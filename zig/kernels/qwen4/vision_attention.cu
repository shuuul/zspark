// The vision tower's attention (spec 019 WS-02): every patch of an image attends to every patch of the same
// image, none of another, 16 heads of 72 values, softmax scaled by 72^-0.5, FlashAttention-style on the tensor
// cores (bf16 m16n8k16, fp32 sums, online softmax in base 2).
//
// A block is (a tile of 128 queries of one image, one head); its 8 warps take 16 query rows each and walk the
// image's keys 64 at a time from the image's first patch. A tile starts at its image's start plus a multiple of
// 128 and its keys at the image's start, so a query row's sums run over the same keys in the same order whatever
// images share the call or where they sit: a row's bits never depend on the rows beside it, nor on the other
// images. Keys past the image are masked to -inf and loaded as zeros; query rows past it are not stored.
//
// The head's 72 values are padded to 80 for Q K^T (five k16 steps, values 72 to 79 zero in shared memory and so
// in Q's fragments) and P V takes nine n8 tiles of V by ldmatrix.trans. Q goes through K's two stages once, then
// stays in registers (20 a lane).
//
// Shared memory a block (rows of 88 bf16, 176 B: 8 rows an ldmatrix fall on 8 distinct 16-byte bank groups):
//
//   region          bytes    what for                                   could be less
//   K, 2 stages     22,528   64 keys x 88 a stage; Q's 128 rows first    the 16 pad values a row (8 of them the
//   V, 2 stages     22,528   64 keys x 88 a stage                        k16 step's, 8 for banks): 18%
//   total           45,056   two blocks an SM (8 warps each, 16 a SM)
//
// Provenance: zspark's own, after FlashAttention 2's tiling (Dao, 2023) and zspark's text attention's fragments
// (`attention_mma.cuh`); the arithmetic is SDPA's non-causal attention over `cu_seqlens` in transformers 5.18's
// `Qwen3_5VisionAttention` as TensorFold 0.6.5 runs it. Its sums' order is not PyTorch's: near, not to the bit.

#include <stdint.h>

#include <cuda_bf16.h>

#include "../common/copies.cuh"
#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using qmm_frag::cp16z;
using qmm_frag::commit;
using qmm_frag::ldmatrix2;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::smem;
using qmm_frag::wait;

constexpr int HIDDEN = 1152;
constexpr int HEADS = 16;
constexpr int HEAD = 72;
constexpr int CHUNKS = HEAD / 8;    // 16-byte pieces of a head's row
constexpr int BM = 128;             // queries a block
constexpr int BN = 64;              // keys a step
constexpr int WARPS = BM / 16;
constexpr int THREADS = WARPS * 32;
constexpr int LD = 88;              // a shared row, bf16
constexpr int KD = 80;              // Q K^T's depth
constexpr int STAGE = BN * LD;      // bf16 a stage
constexpr int SMEM = 4 * STAGE * 2;

using common_copies::ldmatrix4t;
using common_copies::ldmatrix2t;
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// rows [from, from + n) of one head's K or V (part 1 or 2 of qkv) into a stage; rows at or past `end` zeros.
__device__ __forceinline__ void load_rows(__nv_bfloat16* stage, const __nv_bfloat16* qkv, int part, int head,
                                          int from, int n, int end) {
  for (int i = threadIdx.x; i < n * CHUNKS; i += THREADS) {
    const int r = i / CHUNKS, c = i % CHUNKS;
    const int row = from + r;
    const bool read = row < end;
    const __nv_bfloat16* src = qkv + static_cast<size_t>(read ? row : from) * 3 * HIDDEN + part * HIDDEN + head * HEAD + c * 8;
    cp16z(stage + r * LD + c * 8, src, read);
  }
}

// tiles int32 [T, 4]: {first query, image start, image length, unused}.
__global__ void __launch_bounds__(THREADS, 2)
    vision_attention_kernel(const __nv_bfloat16* __restrict__ qkv, const int4* __restrict__ tiles,
                            __nv_bfloat16* __restrict__ out, float scale_log2) {
  extern __shared__ __align__(128) unsigned char raw[];
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(raw);   // [2][BN][LD], Q's [BM][LD] first
  __nv_bfloat16* vs = ks + 2 * STAGE;                            // [2][BN][LD]
  const int4 tile = tiles[blockIdx.x];
  const int head = blockIdx.y;
  const int q0 = tile.x, start = tile.y, end = tile.y + tile.z;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  // the K stages' values 72 to 79 are Q K^T's padding: zero once, never written by a load
  for (int i = threadIdx.x; i < 2 * BN; i += THREADS) *reinterpret_cast<uint4*>(ks + i * LD + HEAD) = make_uint4(0, 0, 0, 0);
  load_rows(ks, qkv, 0, head, q0, BM, end);
  commit();
  wait<0>();
  __syncthreads();
  uint32_t qa[KD / 16][4];
#pragma unroll
  for (int kk = 0; kk < KD / 16; ++kk)
    ldmatrix4(qa[kk], ks + (warp * 16 + (lane & 15)) * LD + kk * 16 + (lane >> 4) * 8);
  __syncthreads();

  float o[HEAD / 8][4];
#pragma unroll
  for (int n = 0; n < HEAD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

  const int steps = (tile.z + BN - 1) / BN;
  load_rows(ks, qkv, 1, head, start, BN, end);
  load_rows(vs, qkv, 2, head, start, BN, end);
  commit();
  for (int j = 0; j < steps; ++j) {
    const int buf = j & 1;
    if (j + 1 < steps) {
      load_rows(ks + (buf ^ 1) * STAGE, qkv, 1, head, start + (j + 1) * BN, BN, end);
      load_rows(vs + (buf ^ 1) * STAGE, qkv, 2, head, start + (j + 1) * BN, BN, end);
    }
    commit();
    wait<1>();
    __syncthreads();
    const __nv_bfloat16* kt = ks + buf * STAGE;
    const __nv_bfloat16* vt = vs + buf * STAGE;

    float s[BN / 8][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KD / 16; ++kk) {
#pragma unroll
      for (int p = 0; p < BN / 16; ++p) {
        uint32_t b[4];
        ldmatrix4(b, kt + (p * 16 + (lane & 7) + ((lane >> 4) << 3)) * LD + kk * 16 + ((lane >> 3) & 1) * 8);
        mma(s[2 * p], qa[kk], b[0], b[1]);
        mma(s[2 * p + 1], qa[kk], b[2], b[3]);
      }
    }

    // the online softmax over this step's keys; a lane holds rows lane / 4 and lane / 4 + 8
    const int key0 = start + j * BN + 2 * (lane & 3);
    float top[2] = {m[0], m[1]};
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = key0 + n * 8 + (e & 1);
        const float v = key < end ? s[n][e] * scale_log2 : -INFINITY;
        s[n][e] = v;
        top[e >> 1] = fmaxf(top[e >> 1], v);
      }
    }
    float alpha[2], sum[2] = {0.f, 0.f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      top[r] = fmaxf(top[r], __shfl_xor_sync(0xffffffffu, top[r], 1));
      top[r] = fmaxf(top[r], __shfl_xor_sync(0xffffffffu, top[r], 2));
      alpha[r] = exp2f(m[r] - top[r]);
      m[r] = top[r];
    }
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float p = exp2f(s[n][e] - m[e >> 1]);
        s[n][e] = p;
        sum[e >> 1] += p;
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) l[r] = l[r] * alpha[r] + sum[r];
#pragma unroll
    for (int n = 0; n < HEAD / 8; ++n) {
      o[n][0] *= alpha[0];
      o[n][1] *= alpha[0];
      o[n][2] *= alpha[1];
      o[n][3] *= alpha[1];
    }

    // O += P V, P's bf16 A fragments straight from the S accumulators
#pragma unroll
    for (int t = 0; t < BN / 16; ++t) {
      uint32_t pa[4] = {pack(s[2 * t][0], s[2 * t][1]), pack(s[2 * t][2], s[2 * t][3]),
                        pack(s[2 * t + 1][0], s[2 * t + 1][1]), pack(s[2 * t + 1][2], s[2 * t + 1][3])};
      const __nv_bfloat16* vrow = vt + (t * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        uint32_t b[4];
        ldmatrix4t(b, vrow + n * 16 + (lane >> 4) * 8);
        mma(o[2 * n], pa, b[0], b[1]);
        mma(o[2 * n + 1], pa, b[2], b[3]);
      }
      uint32_t b[2];
      ldmatrix2t(b, vrow + 64);
      mma(o[8], pa, b[0], b[1]);
    }
    __syncthreads();
  }

  // the row sums are each lane's share: add the quad's, then O / l
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    l[r] = 1.f / l[r];
  }
  const int row0 = q0 + warp * 16 + (lane >> 2);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = row0 + r * 8;
    if (row >= end) continue;
    __nv_bfloat16* dst = out + static_cast<size_t>(row) * HIDDEN + head * HEAD + 2 * (lane & 3);
#pragma unroll
    for (int n = 0; n < HEAD / 8; ++n)
      *reinterpret_cast<uint32_t*>(dst + n * 8) = pack(o[n][2 * r] * l[r], o[n][2 * r + 1] * l[r]);
  }
}

}  // namespace

// out bf16 [P, 1152] = each image's attention over its own patches, from qkv bf16 [P, 3456] (q and k rotated).
// tiles int32 [T, 4]: a block's {first query, image start, image length, 0}; an image of n patches has
// ceil(n / 128) tiles starting at its start plus multiples of 128 (`vision.zig` writes them).
extern "C" int zs_qwen4_vision_attention(const zs_view* qkv_, const zs_view* tiles_, const zs_view* out_) {
  zs::begin();
  const zs::View qkv(qkv_), tiles(tiles_), out(out_);
  const int64_t p = qkv.size(0);
  ZS_REQUIRE(qkv.defined() && qkv.dtype() == ZS_BF16 && qkv.dim() == 2 && qkv.is_contiguous() && qkv.size(1) == 3 * HIDDEN,
             "qkv: bf16 [P, 3456]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.dim() == 2 && out.is_contiguous() && out.size(0) == p &&
                 out.size(1) == HIDDEN,
             "out: bf16 [P, 1152]");
  ZS_REQUIRE(tiles.defined() && tiles.dtype() == ZS_I32 && tiles.dim() == 2 && tiles.is_contiguous() &&
                 tiles.size(1) == 4 && tiles.size(0) >= 1 && tiles.size(0) <= INT32_MAX,
             "tiles: int32 [T, 4]");
  static bool configured = false;
  if (!configured) {
    ZS_CUDA_CHECK(cudaFuncSetAttribute(vision_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    configured = true;
  }
  const float scale_log2 = 1.4426950408889634f / sqrtf(static_cast<float>(HEAD));
  vision_attention_kernel<<<dim3(static_cast<unsigned>(tiles.size(0)), HEADS), THREADS, SMEM, zs::stream()>>>(
      qkv.data_ptr<__nv_bfloat16>(), tiles.data_ptr<int4>(), out.data_ptr<__nv_bfloat16>(), scale_log2);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
