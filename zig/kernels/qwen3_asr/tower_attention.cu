// Qwen3-ASR's audio tower attention (spec 138 WS-03): every token of a block of at most 104 attends to every
// token of the same block and to none of another (block-diagonal, non-causal, `cu_seqlens` of 104-token blocks),
// 16 heads of 64 values (1.7B; 0.6B 14), softmax scaled by 64^-0.5, FlashAttention-style on the tensor cores (bf16 m16n8k16,
// fp32 sums, online softmax in base 2).
//
// `qwen4/vision_attention.cu`'s kernel at head 64 (no padding of the head: Q K^T is four k16 steps, P V eight n8
// tiles by ldmatrix.trans) over a fused q | k | v of [N, 3W]. A block is (a tile of 128 queries of one segment,
// one head); its 8 warps take 16 query rows each and walk the segment's keys 64 at a time from its first token.
// A tile starts at its segment's start plus a multiple of 128 and its keys at the segment's start, so a query
// row's sums run over the same keys in the same order whatever segments share the call or where they sit: a
// row's bits never depend on the rows beside it (the row rule; the encoder's blocks are independent of the
// audio around them). Keys past the segment are masked to -inf and loaded as zeros; query rows past it are not
// stored. A 104-token block is one tile, two key steps, 24 of the 128 query rows idle.
//
// Shared memory a block (rows of 72 bf16, 144 B: 8 rows an ldmatrix fall on 8 distinct 16-byte bank groups):
//
//   region          bytes    what for                                   could be less
//   K, 2 stages     18,432   64 keys x 72 a stage; Q's 128 rows first    the 8 pad values a row (banks): 11%
//   V, 2 stages     18,432   64 keys x 72 a stage                        as K; a block's 104 keys fit two
//   total           36,864   two blocks an SM (8 warps each, 16 a SM)    stages, so no ring is needed either
//
// Not CUTLASS: its FMHA does not take segments of their own; the form is zspark's vision attention, which is
// held to SDPA there. What could be shared later: this and `vision_attention.cu` differ in HEAD, HIDDEN and the
// head's padding alone; a kernel templated on them would serve both (and Unlimited-OCR's head 64, spec 137).
//
// Provenance: zspark's own, after FlashAttention 2's tiling (Dao, 2023) and `qwen4/vision_attention.cu`; the
// arithmetic is FlashAttention-2 varlen over `cu_seqlens` in the qwen-asr 0.0.6 package's
// `Qwen3ASRAudioAttention` (transformers backend), as the spec-138 fixtures run it (SDPA per block). Its sums'
// order is not PyTorch's: near, not to the bit.

#include <stdint.h>

#include <cuda_bf16.h>

#include "../common/copies.cuh"
#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using qmm_frag::cp16z;
using qmm_frag::commit;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::smem;
using qmm_frag::wait;

// The tower's width is a template argument (spec 140 WS-09): 1,024 (1.7B, 16 heads) and 896 (0.6B, 14 heads), each
// its own instantiation, so 1.7B's code is what it was with the constant; the heads are the grid's y.
constexpr int HEAD = 64;
constexpr int CHUNKS = HEAD / 8;    // 16-byte pieces of a head's row
constexpr int BM = 128;             // queries a block
constexpr int BN = 64;              // keys a step
constexpr int WARPS = BM / 16;
constexpr int THREADS = WARPS * 32;
constexpr int LD = 72;              // a shared row, bf16
constexpr int STAGE = BN * LD;      // bf16 a stage
constexpr int SMEM = 4 * STAGE * 2;

using common_copies::ldmatrix4t;
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// rows [from, from + n) of one head's Q, K or V (part 0, 1 or 2 of qkv) into a stage; rows at or past `end` zeros.
template <int HIDDEN>
__device__ __forceinline__ void tower_rows(__nv_bfloat16* stage, const __nv_bfloat16* qkv, int part, int head,
                                           int from, int n, int end) {
  for (int i = threadIdx.x; i < n * CHUNKS; i += THREADS) {
    const int r = i / CHUNKS, c = i % CHUNKS;
    const int row = from + r;
    const bool read = row < end;
    const __nv_bfloat16* src = qkv + static_cast<size_t>(read ? row : from) * 3 * HIDDEN + part * HIDDEN + head * HEAD + c * 8;
    cp16z(stage + r * LD + c * 8, src, read);
  }
}

// tiles int32 [T, 4]: {first query, segment start, segment length, unused}.
template <int HIDDEN>
__global__ void __launch_bounds__(THREADS, 2)
    tower_attention_kernel(const __nv_bfloat16* __restrict__ qkv, const int4* __restrict__ tiles,
                           __nv_bfloat16* __restrict__ out, float scale_log2) {
  extern __shared__ __align__(128) unsigned char raw[];
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(raw);   // [2][BN][LD], Q's [BM][LD] first
  __nv_bfloat16* vs = ks + 2 * STAGE;                            // [2][BN][LD]
  const int4 tile = tiles[blockIdx.x];
  const int head = blockIdx.y;
  const int q0 = tile.x, start = tile.y, end = tile.y + tile.z;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  tower_rows<HIDDEN>(ks, qkv, 0, head, q0, BM, end);
  commit();
  wait<0>();
  __syncthreads();
  uint32_t qa[HEAD / 16][4];
#pragma unroll
  for (int kk = 0; kk < HEAD / 16; ++kk)
    ldmatrix4(qa[kk], ks + (warp * 16 + (lane & 15)) * LD + kk * 16 + (lane >> 4) * 8);
  __syncthreads();

  float o[HEAD / 8][4];
#pragma unroll
  for (int n = 0; n < HEAD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

  const int steps = (tile.z + BN - 1) / BN;
  tower_rows<HIDDEN>(ks, qkv, 1, head, start, BN, end);
  tower_rows<HIDDEN>(vs, qkv, 2, head, start, BN, end);
  commit();
  for (int j = 0; j < steps; ++j) {
    const int buf = j & 1;
    if (j + 1 < steps) {
      tower_rows<HIDDEN>(ks + (buf ^ 1) * STAGE, qkv, 1, head, start + (j + 1) * BN, BN, end);
      tower_rows<HIDDEN>(vs + (buf ^ 1) * STAGE, qkv, 2, head, start + (j + 1) * BN, BN, end);
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
    for (int kk = 0; kk < HEAD / 16; ++kk) {
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
      for (int n = 0; n < HEAD / 16; ++n) {
        uint32_t b[4];
        ldmatrix4t(b, vrow + n * 16 + (lane >> 4) * 8);
        mma(o[2 * n], pa, b[0], b[1]);
        mma(o[2 * n + 1], pa, b[2], b[3]);
      }
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

template <int HIDDEN>
void launch(const zs::View& qkv, const zs::View& tiles, const zs::View& out) {
  static bool configured = false;
  if (!configured) {
    ZS_CUDA_CHECK(cudaFuncSetAttribute(tower_attention_kernel<HIDDEN>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    configured = true;
  }
  const float scale_log2 = 1.4426950408889634f / sqrtf(static_cast<float>(HEAD));
  tower_attention_kernel<HIDDEN><<<dim3(static_cast<unsigned>(tiles.size(0)), HIDDEN / HEAD), THREADS, SMEM, zs::stream()>>>(
      qkv.data_ptr<__nv_bfloat16>(), tiles.data_ptr<int4>(), out.data_ptr<__nv_bfloat16>(), scale_log2);
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

// out bf16 [N, W] = each segment's attention over its own tokens, from qkv bf16 [N, 3W] (q, k, v; head h at h * 64
// in each), W 1,024 (16 heads) or 896 (14). tiles int32 [T, 4]: a block's {first query, segment start, segment
// length, 0}; a segment of n tokens has ceil(n / 128) tiles starting at its start plus multiples of 128 (`tower.zig`
// writes them).
extern "C" int zs_qwen3_asr_tower_attention(const zs_view* qkv_, const zs_view* tiles_, const zs_view* out_) {
  zs::begin();
  const zs::View qkv(qkv_), tiles(tiles_), out(out_);
  const int64_t n = qkv.size(0);
  const int64_t width = out.defined() && out.dim() == 2 ? out.size(1) : 0;
  ZS_REQUIRE(width == 1024 || width == 896, "out: bf16 [N, W], W 1024 or 896 (heads of 64)");
  ZS_REQUIRE(qkv.defined() && qkv.dtype() == ZS_BF16 && qkv.dim() == 2 && qkv.is_contiguous() && qkv.size(1) == 3 * width,
             "qkv: bf16 [N, 3W]");
  ZS_REQUIRE(out.dtype() == ZS_BF16 && out.is_contiguous() && out.size(0) == n, "out: bf16 [N, W]");
  ZS_REQUIRE(tiles.defined() && tiles.dtype() == ZS_I32 && tiles.dim() == 2 && tiles.is_contiguous() &&
                 tiles.size(1) == 4 && tiles.size(0) >= 1 && tiles.size(0) <= INT32_MAX,
             "tiles: int32 [T, 4]");
  if (width == 1024)
    launch<1024>(qkv, tiles, out);
  else
    launch<896>(qkv, tiles, out);
  return zs::status();
}
