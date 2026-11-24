// Unlimited-OCR's vision attention (spec 137 WS-03): SAM's (12 heads of 64, a 14 x 14 window or a whole view,
// decomposed relative positions added to the scores) and CLIP's (16 heads of 64, a whole view, no bias), each
// query attending to every key of its own segment and none of another. FlashAttention-style on the tensor cores
// (bf16 m16n8k16, fp32 sums, online softmax in base 2), after `qwen4/vision_attention.cu` with head 64 (no
// padding of Q K^T's depth, eight n8 tiles of V) and the bias.
//
// A block is (a tile of 128 queries of one segment, one head); its 8 warps take 16 query rows each and walk the
// segment's keys 64 at a time from its first. A tile starts at its segment's start plus a multiple of 128 and its
// keys at the segment's start, so a query row's sums run over the same keys in the same order whatever segments
// share the call: a row's bits never depend on the rows beside it, nor on the other segments or views. Keys past
// the segment are masked to -inf and loaded as zeros; query rows past it are not stored.
//
// The bias (SAM's `add_decomposed_rel_pos` then SDPA's `attn_mask`): the score of query q and key k of a segment
// of width W is (q . k) / 8 + bf(rel_h[q][k / W] + rel_w[q][k % W]), rel_h and rel_w the bf16 einsums of the
// unscaled q with the height and width tables (`ocr_relpos_kernel`, below), their sum rounded to bf16 as PyTorch's
// bf16 add. A window's padded tokens are keys like the others (zero rows after norm1, so their k and v are qkv's
// bias), as in `window_partition`.
//
// Shared memory a block (rows of 72 bf16, 144 B: 8 rows an ldmatrix fall on 8 distinct 16-byte bank groups):
//
//   region          bytes    what for                                   could be less
//   K, 2 stages     18,432   64 keys x 72 a stage; Q's 128 rows first    the 8 pad values a row, for banks: 11%
//   V, 2 stages     18,432   64 keys x 72 a stage
//   rel_w (SAM)     6,144    128 rows x bias_ld(W) bf16 of rel_w (W     the 64 x 64 view keeps a lane's 16 values
//                   to       values, zeros past them), an odd multiple  in registers instead (WIDE); rel_h is read
//                   10,240   of 8 for banks: a window 24, a crop 40     from global memory (L1), a value a step there
//   total           36,864   CLIP and the 64 x 64 view's global blocks; SAM's windows 43,008, a crop's global blocks
//                            47,104: two blocks an SM in every form (128 registers, 28 bytes spilled with a bias)
//
// The bias was first read from global memory a score at a time (15 TFLOPS on SAM's global blocks); staged with the
// block's K and V it is 43 (spec 137 WS-07), still under qwen4's unbiased 75: rel_h and rel_w staged whole made the
// biased blocks one an SM (59,392 and 71,680 B, 147 registers). rel_h out of the stage and the 64 x 64 view's
// rel_w in registers make them two an SM: a global block of the global view 1.23 to 1.06 ms, the global view's windows 178
// to 145 us, a page's tower 1.8 ms less (spec 142 WS-03); the blocks stay bound on the softmax's issue (ncu: 37% of
// issue slots busy, 25 to 31% occupancy).
//
// Provenance: zspark's own, after FlashAttention 2's tiling (Dao, 2023) and `qwen4/vision_attention.cu`; the
// arithmetic is the checkpoint's `deepencoder.py` (`Attention`, `get_rel_pos`, `add_decomposed_rel_pos`,
// `NoTPAttention`) under `infer`'s autocast. Its sums' order is not PyTorch's: near, not to the bit.

#include <stdint.h>

#include <cuda_bf16.h>

#include "../common/copies.cuh"
#include "../common/math.cuh"
#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using common_math::bf;
using qmm_frag::cp16z;
using qmm_frag::commit;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::smem;
using qmm_frag::wait;

constexpr int HEAD = 64;
constexpr int CHUNKS = HEAD / 8;    // 16-byte pieces of a head's row
constexpr int BM = 128;             // queries a block
constexpr int BN = 64;              // keys a step
constexpr int WARPS = BM / 16;
constexpr int THREADS = WARPS * 32;
constexpr int LD = 72;              // a shared row, bf16
constexpr int STAGE = BN * LD;      // bf16 a stage
constexpr int SMEM = 4 * STAGE * 2;
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BIAS_LD_MAX = 248;    // rel_w's widest staged row, bf16: SMEM + BM rows of it within 101,376 B

using common_copies::ldmatrix4t;
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// A segment row's qkv: the row itself, or with `map` the token map[row] holds, `pad` (qkv's bias) where it holds -1.
__device__ __forceinline__ const __nv_bfloat16* qkv_row(const __nv_bfloat16* qkv, const int* map, const __nv_bfloat16* pad,
                                                        int hidden, int row) {
  if (map == nullptr) return qkv + static_cast<size_t>(row) * 3 * hidden;
  const int t = map[row];
  return t < 0 ? pad : qkv + static_cast<size_t>(t) * 3 * hidden;
}

// rows [from, from + n) of one head's Q, K or V (part 0, 1 or 2 of qkv) into a stage; rows at or past `end` zeros.
__device__ __forceinline__ void load_rows(__nv_bfloat16* stage, const __nv_bfloat16* qkv, const int* map,
                                          const __nv_bfloat16* pad, int hidden, int part, int head, int from, int n,
                                          int end) {
  for (int i = threadIdx.x; i < n * CHUNKS; i += THREADS) {
    const int r = i / CHUNKS, c = i % CHUNKS;
    const int row = from + r;
    const bool read = row < end;
    const __nv_bfloat16* src = qkv_row(qkv, map, pad, hidden, read ? row : from) + part * hidden + head * HEAD + c * 8;
    cp16z(stage + r * LD + c * 8, src, read);
  }
}

// The bias's row in shared memory, in bf16: `stride` (kh + W) rounded up to an odd multiple of 8 so that the 8
// rows a quad's lanes read fall on distinct banks.
__host__ __device__ __forceinline__ int bias_ld(int stride) {
  int ld = (stride + 7) / 8 * 8;
  if ((ld / 8) % 2 == 0) ld += 8;
  return ld;
}

// tiles int32 [T, 4]: {first query, segment start, segment length, segment width}. rel bf16 [heads, P, stride]
// (rel_h's kh values then rel_w's W) unless NONE, the segment square (kh = W, stride 2 W), W even. STAGED takes
// rel_w's `BM` rows after K and V (`bias_ld(W)`) and reads a key's rel_h from global memory (L1), so a crop's
// global blocks (47,104 B) and the windows (43,008) run two blocks an SM; WIDE (W = BN, the 64 x 64 view, where a
// step of keys is one kh and the same kw's) holds a lane's 16 rel_w values in registers and one rel_h a row a step:
// no stage, two blocks an SM. Either reads the values the first cut staged whole, so the scores keep their bits.
enum Bias { NONE, STAGED, WIDE };

template <Bias BIAS>
__global__ void __launch_bounds__(THREADS, 2)
    ocr_attention_kernel(const __nv_bfloat16* __restrict__ qkv, const int* __restrict__ map,
                         const __nv_bfloat16* __restrict__ pad, const int4* __restrict__ tiles,
                         const __nv_bfloat16* __restrict__ rel, __nv_bfloat16* __restrict__ out, int hidden, int rows,
                         int stride, float scale_log2) {
  extern __shared__ __align__(128) unsigned char raw[];
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(raw);   // [2][BN][LD], Q's [BM][LD] first
  __nv_bfloat16* vs = ks + 2 * STAGE;                            // [2][BN][LD]
  __nv_bfloat16* bs = vs + 2 * STAGE;                            // [BM][bias_ld(W)] when STAGED
  const int4 tile = tiles[blockIdx.x];
  const int head = blockIdx.y;
  const int q0 = tile.x, start = tile.y, end = tile.y + tile.z, width = tile.w;
  const int kh_count = BIAS != NONE ? tile.z / width : 0;
  const int bld = BIAS == STAGED ? bias_ld(width) : 0;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (BIAS != NONE && (kh_count != width || 2 * width != stride || (BIAS == WIDE && width != BN))) __trap();

  load_rows(ks, qkv, map, pad, hidden, 0, head, q0, BM, end);
  commit();
  if (BIAS == STAGED) {
    // the tile's rows of rel_w, 4 bytes a lane (zeros past the segment)
    const int words = width / 2;
    for (int i = threadIdx.x; i < BM * words; i += THREADS) {
      const int r = i / words, c = i - r * words;
      const int row = q0 + r;
      uint32_t v = 0;
      if (row < end && row < rows)
        v = __ldg(reinterpret_cast<const uint32_t*>(rel + (static_cast<size_t>(head) * rows + row) * stride + kh_count) + c);
      *reinterpret_cast<uint32_t*>(bs + r * bld + 2 * c) = v;
    }
  }
  wait<0>();
  __syncthreads();
  uint32_t qa[HEAD / 16][4];
#pragma unroll
  for (int kk = 0; kk < HEAD / 16; ++kk)
    ldmatrix4(qa[kk], ks + (warp * 16 + (lane & 15)) * LD + kk * 16 + (lane >> 4) * 8);
  __syncthreads();

  // this lane's two query rows, their rows of rel_h in global memory (a row past the segment reads its last row:
  // never stored) and of rel_w, staged or in registers
  const int row0 = q0 + warp * 16 + (lane >> 2);
  const __nv_bfloat16* bias0 = bs + (warp * 16 + (lane >> 2)) * bld;
  const __nv_bfloat16* bias1 = bias0 + 8 * bld;
  const __nv_bfloat16* relh0 = nullptr;
  const __nv_bfloat16* relh1 = nullptr;
  uint32_t rw[2][BIAS == WIDE ? BN / 8 : 1];
  if (BIAS != NONE) {
    const int last = min(end, rows) - 1;
    relh0 = rel + (static_cast<size_t>(head) * rows + min(row0, last)) * stride;
    relh1 = rel + (static_cast<size_t>(head) * rows + min(row0 + 8, last)) * stride;
  }
  if (BIAS == WIDE) {
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      const int kw = 2 * (lane & 3) + 8 * n;
      rw[0][n] = __ldg(reinterpret_cast<const uint32_t*>(relh0 + kh_count + kw));
      rw[1][n] = __ldg(reinterpret_cast<const uint32_t*>(relh1 + kh_count + kw));
    }
  }

  float o[HEAD / 8][4];
#pragma unroll
  for (int n = 0; n < HEAD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

  const int steps = (tile.z + BN - 1) / BN;
  load_rows(ks, qkv, map, pad, hidden, 1, head, start, BN, end);
  load_rows(vs, qkv, map, pad, hidden, 2, head, start, BN, end);
  commit();
  for (int j = 0; j < steps; ++j) {
    const int buf = j & 1;
    if (j + 1 < steps) {
      load_rows(ks + (buf ^ 1) * STAGE, qkv, map, pad, hidden, 1, head, start + (j + 1) * BN, BN, end);
      load_rows(vs + (buf ^ 1) * STAGE, qkv, map, pad, hidden, 2, head, start + (j + 1) * BN, BN, end);
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

    // scaled scores plus the bias, then the online softmax over this step's keys; a lane holds rows lane / 4
    // and lane / 4 + 8, keys key0 + 8 n and the one after (W even: the pair shares its kh)
    const int key0 = j * BN + 2 * (lane & 3);   // within the segment
    int kh = 0, kw = 0;
    if (BIAS != NONE) {
      kh = key0 / width;
      kw = key0 - kh * width;
    }
    float wh0 = 0.f, wh1 = 0.f;   // WIDE: the step's one kh
    if (BIAS == WIDE) {
      wh0 = __bfloat162float(__ldg(relh0 + j));
      wh1 = __bfloat162float(__ldg(relh1 + j));
    }
    float top[2] = {m[0], m[1]};
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      // past the segment: masked (a bias segment's length is even, so a pair is in or out together)
      const bool real0 = start + key0 + n * 8 < end, real1 = start + key0 + n * 8 + 1 < end;
      float b[4];
      if (BIAS != NONE && real0) {
        __nv_bfloat162 w0, w1;
        float h0 = wh0, h1 = wh1;
        if (BIAS == WIDE) {
          w0 = *reinterpret_cast<const __nv_bfloat162*>(&rw[0][n]);
          w1 = *reinterpret_cast<const __nv_bfloat162*>(&rw[1][n]);
        } else {
          w0 = *reinterpret_cast<const __nv_bfloat162*>(bias0 + kw);
          w1 = *reinterpret_cast<const __nv_bfloat162*>(bias1 + kw);
          h0 = __bfloat162float(__ldg(relh0 + kh));
          h1 = __bfloat162float(__ldg(relh1 + kh));
        }
        b[0] = bf(h0 + __low2float(w0));
        b[1] = bf(h0 + __high2float(w0));
        b[2] = bf(h1 + __low2float(w1));
        b[3] = bf(h1 + __high2float(w1));
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v = -INFINITY;
        if (e & 1 ? real1 : real0) {
          // the product scaled, then the bias's FMA on it: the order of the first cut's contraction, pinned
          v = __fmul_rn(s[n][e], scale_log2);
          if (BIAS != NONE) v = __fmaf_rn(b[e], LOG2E, v);
        }
        s[n][e] = v;
        top[e >> 1] = fmaxf(top[e >> 1], v);
      }
      if (BIAS == STAGED) {
        kw += 8;
        if (kw >= width) {
          kw -= width;
          ++kh;
        }
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
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = row0 + r * 8;
    if (row >= end) continue;
    const int t = map == nullptr ? row : map[row];   // a padded row is not stored
    if (t < 0) continue;
    __nv_bfloat16* dst = out + static_cast<size_t>(t) * hidden + head * HEAD + 2 * (lane & 3);
#pragma unroll
    for (int n = 0; n < HEAD / 8; ++n)
      *reinterpret_cast<uint32_t*>(dst + n * 8) = pack(o[n][2 * r] * l[r], o[n][2 * r + 1] * l[r]);
  }
}

// rel bf16 [heads, P, kh + W]: for row p (place i = p % len in its segment of `len` rows, `W` wide, at height
// i / W and column i % W) and head h, rel_h[k] = bf(q . table_h[i / W - k + kh - 1]) for k < kh = len / W, then
// rel_w[k] = bf(q . table_w[i % W - k + W - 1]) for k < W; q the head's unscaled 64 values, the sum fp32
// (`get_rel_pos` with q and k of one size, `torch.einsum` on bf16). A block a row, a lane a term k: its table row
// widened to fp32 in 64 registers once, then every head's q (broadcast from shared memory) against it. A term's sum
// is the first cut's order (spec 137 WS-03), kept to the bit: eight chains of eight values, each a product then
// seven FMAs, met as ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7)). The first cut met the chains of eight lanes by
// `shfl.xor`, stored a term from one lane in eight and loaded q 2 bytes a step: 0.33 ms a global block on a global
// view, 0.14 now, 0.22 against 0.48 on six crops, 0.11 against 0.22 their windows (spec 137 WS-07). Eight rows a
// block (their q staged together) were slower: the windows' blocks of one warp, four an SM.
// Shared memory: the row's q widened to fp32, hidden floats (3,072 B for SAM's 768); nothing else.
__global__ void __launch_bounds__(128)
    ocr_relpos_kernel(const __nv_bfloat16* __restrict__ qkv, const int* __restrict__ map,
                      const __nv_bfloat16* __restrict__ table_h, const __nv_bfloat16* __restrict__ table_w,
                      __nv_bfloat16* __restrict__ rel, int hidden, int rows, int len, int width) {
  extern __shared__ __align__(16) unsigned char raw[];
  float* q = reinterpret_cast<float*>(raw);
  const int p = blockIdx.x;
  const int token = map == nullptr ? p : map[p];
  if (token < 0) {
    // a padded window row: its scores are never stored, its row of rel zeros
    const int stride = len / width + width;
    for (int h = 0; h < hidden / HEAD; ++h)
      if (threadIdx.x < stride) rel[(static_cast<size_t>(h) * rows + p) * stride + threadIdx.x] = __float2bfloat16_rn(0.f);
    return;
  }
  // the row's q, 16 bytes a load, every load issued before the first is waited for (up to four a lane)
  {
    const uint4* src = reinterpret_cast<const uint4*>(qkv + static_cast<size_t>(token) * 3 * hidden);
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = threadIdx.x + u * blockDim.x;
      if (c < hidden / 8) v[u] = __ldg(src + c);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = threadIdx.x + u * blockDim.x;
      if (c < hidden / 8) {
        const uint32_t w[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
        float4 lo, hi;
        lo.x = __uint_as_float(w[0] << 16), lo.y = __uint_as_float(w[0] & 0xFFFF0000u);
        lo.z = __uint_as_float(w[1] << 16), lo.w = __uint_as_float(w[1] & 0xFFFF0000u);
        hi.x = __uint_as_float(w[2] << 16), hi.y = __uint_as_float(w[2] & 0xFFFF0000u);
        hi.z = __uint_as_float(w[3] << 16), hi.w = __uint_as_float(w[3] & 0xFFFF0000u);
        reinterpret_cast<float4*>(q)[2 * c] = lo;
        reinterpret_cast<float4*>(q)[2 * c + 1] = hi;
      }
    }
  }
  const int i = p % len, kh = len / width, stride = kh + width;
  const int y = i / width, x = i % width;
  const int heads = hidden / HEAD;
  const int k = threadIdx.x;
  const bool real = k < stride;
  float t[HEAD];
  {
    const int kk = real ? k : 0;
    const __nv_bfloat16* row = kk < kh ? table_h + static_cast<size_t>(y - kk + kh - 1) * HEAD
                                       : table_w + static_cast<size_t>(x - (kk - kh) + width - 1) * HEAD;
#pragma unroll
    for (int c = 0; c < HEAD / 8; ++c) {
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(row) + c);
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        t[8 * c + 2 * e] = __uint_as_float(w[e] << 16);
        t[8 * c + 2 * e + 1] = __uint_as_float(w[e] & 0xFFFF0000u);
      }
    }
  }
  __syncthreads();
  if (!real) return;
  for (int h = 0; h < heads; ++h) {
    float chain[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const float4 a = *reinterpret_cast<const float4*>(q + h * HEAD + c * 8);
      const float4 b = *reinterpret_cast<const float4*>(q + h * HEAD + c * 8 + 4);
      float sum = a.x * t[8 * c];
      sum = fmaf(a.y, t[8 * c + 1], sum);
      sum = fmaf(a.z, t[8 * c + 2], sum);
      sum = fmaf(a.w, t[8 * c + 3], sum);
      sum = fmaf(b.x, t[8 * c + 4], sum);
      sum = fmaf(b.y, t[8 * c + 5], sum);
      sum = fmaf(b.z, t[8 * c + 6], sum);
      sum = fmaf(b.w, t[8 * c + 7], sum);
      chain[c] = sum;
    }
    const float sum = ((chain[0] + chain[1]) + (chain[2] + chain[3])) + ((chain[4] + chain[5]) + (chain[6] + chain[7]));
    rel[(static_cast<size_t>(h) * rows + p) * stride + k] = __float2bfloat16_rn(sum);
  }
}

}  // namespace

// out bf16 [N, hidden] = each segment's attention over its own rows, from qkv bf16 [N, 3 * hidden] (q, k, v, each
// heads x 64), hidden 768 or 1024. The P segment rows are qkv's N rows, or with `map` (int32 [P]) the token each
// holds, -1 a window's padding, whose q, k and v are `pad` (bf16 [3 * hidden], qkv's bias: a zero row's) and whose
// output is not stored. tiles int32 [T, 4]: {first query, segment start, segment length, segment width}, a segment
// of n rows cut in ceil(n / 128) tiles from its start (`vision_sam.zig` writes them). rel bf16 [heads, P, length /
// width + width] (`zs_unlimited_ocr_vision_relpos`), or null for no bias.
extern "C" int zs_unlimited_ocr_vision_attention(const zs_view* qkv_, const zs_view* map_, const zs_view* pad_,
                                                 const zs_view* tiles_, const zs_view* rel_, const zs_view* out_) {
  zs::begin();
  const zs::View qkv(qkv_), map(map_), pad(pad_), tiles(tiles_), rel(rel_), out(out_);
  ZS_REQUIRE(qkv.defined() && qkv.dtype() == ZS_BF16 && qkv.dim() == 2 && qkv.is_contiguous() &&
                 (qkv.size(1) == 3 * 768 || qkv.size(1) == 3 * 1024),
             "qkv: bf16 [N, 3 * hidden], hidden 768 or 1024");
  const int64_t n = qkv.size(0), hidden = qkv.size(1) / 3, heads = hidden / HEAD;
  ZS_REQUIRE(!map.defined() || (map.dtype() == ZS_I32 && map.is_contiguous() && pad.defined() &&
                                pad.dtype() == ZS_BF16 && pad.numel() == 3 * hidden),
             "map: int32 [P] with pad bf16 [3 * hidden]");
  const int64_t p = map.defined() ? map.numel() : n;
  ZS_REQUIRE(p <= INT32_MAX && n <= INT32_MAX, "P and N within int32");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.dim() == 2 && out.is_contiguous() && out.size(0) == n &&
                 out.size(1) == hidden,
             "out: bf16 [N, hidden]");
  ZS_REQUIRE(tiles.defined() && tiles.dtype() == ZS_I32 && tiles.dim() == 2 && tiles.is_contiguous() &&
                 tiles.size(1) == 4 && tiles.size(0) >= 1 && tiles.size(0) <= INT32_MAX,
             "tiles: int32 [T, 4]");
  ZS_REQUIRE(!rel.defined() || (rel.dtype() == ZS_BF16 && rel.dim() == 3 && rel.size(0) == heads && rel.size(1) == p),
             "rel: bf16 [heads, P, kh + W]");
  const int stride = rel.defined() ? static_cast<int>(rel.size(2)) : 0;
  const int width = stride / 2;
  ZS_REQUIRE(stride % 4 == 0 && bias_ld(width) <= BIAS_LD_MAX, "rel: kh = W even, W at most 248");
  static bool configured = false;
  if (!configured) {
    ZS_CUDA_CHECK(cudaFuncSetAttribute(ocr_attention_kernel<NONE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    ZS_CUDA_CHECK(cudaFuncSetAttribute(ocr_attention_kernel<WIDE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    ZS_CUDA_CHECK(cudaFuncSetAttribute(ocr_attention_kernel<STAGED>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       SMEM + BM * BIAS_LD_MAX * 2));
    configured = true;
  }
  const float scale_log2 = LOG2E / sqrtf(static_cast<float>(HEAD));
  const dim3 grid(static_cast<unsigned>(tiles.size(0)), static_cast<unsigned>(heads));
  if (rel.defined() && width == BN)
    ocr_attention_kernel<WIDE><<<grid, THREADS, SMEM, zs::stream()>>>(
        qkv.data_ptr<__nv_bfloat16>(), map.data_ptr<int>(), pad.data_ptr<__nv_bfloat16>(), tiles.data_ptr<int4>(), rel.data_ptr<__nv_bfloat16>(),
        out.data_ptr<__nv_bfloat16>(), static_cast<int>(hidden), static_cast<int>(p), stride, scale_log2);
  else if (rel.defined())
    ocr_attention_kernel<STAGED><<<grid, THREADS, SMEM + BM * bias_ld(width) * 2, zs::stream()>>>(
        qkv.data_ptr<__nv_bfloat16>(), map.data_ptr<int>(), pad.data_ptr<__nv_bfloat16>(), tiles.data_ptr<int4>(), rel.data_ptr<__nv_bfloat16>(),
        out.data_ptr<__nv_bfloat16>(), static_cast<int>(hidden), static_cast<int>(p), stride, scale_log2);
  else
    ocr_attention_kernel<NONE><<<grid, THREADS, SMEM, zs::stream()>>>(
        qkv.data_ptr<__nv_bfloat16>(), map.data_ptr<int>(), pad.data_ptr<__nv_bfloat16>(), tiles.data_ptr<int4>(), nullptr, out.data_ptr<__nv_bfloat16>(),
        static_cast<int>(hidden), static_cast<int>(p), 0, scale_log2);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// rel bf16 [heads, P, len / W + W] from q of qkv bf16 [N, 3 * hidden], the P segment rows qkv's or with `map`
// (int32 [P]) the token each holds (-1 padding: its row zeros), P a multiple of `len` (segments of `len` rows, `W`
// wide), table_h bf16 [2 (len / W) - 1, 64] and table_w bf16 [2 W - 1, 64].
extern "C" int zs_unlimited_ocr_vision_relpos(const zs_view* qkv_, const zs_view* map_, const zs_view* table_h_,
                                              const zs_view* table_w_, const zs_view* rel_, int64_t len, int64_t width) {
  zs::begin();
  const zs::View qkv(qkv_), map(map_), table_h(table_h_), table_w(table_w_), rel(rel_);
  ZS_REQUIRE(qkv.defined() && qkv.dtype() == ZS_BF16 && qkv.dim() == 2 && qkv.is_contiguous() && qkv.size(1) % (3 * HEAD) == 0,
             "qkv: bf16 [N, 3 * hidden]");
  ZS_REQUIRE(!map.defined() || (map.dtype() == ZS_I32 && map.is_contiguous()), "map: int32 [P]");
  const int64_t p = map.defined() ? map.numel() : qkv.size(0), hidden = qkv.size(1) / 3, heads = hidden / HEAD;
  ZS_REQUIRE(width >= 1 && len >= width && len % width == 0 && p % len == 0 && p <= INT32_MAX, "len: W's multiple, P's divisor");
  const int64_t kh = len / width;
  ZS_REQUIRE(table_h.defined() && table_h.dtype() == ZS_BF16 && table_h.numel() == (2 * kh - 1) * HEAD, "table_h: bf16 [2 kh - 1, 64]");
  ZS_REQUIRE(table_w.defined() && table_w.dtype() == ZS_BF16 && table_w.numel() == (2 * width - 1) * HEAD, "table_w: bf16 [2 W - 1, 64]");
  ZS_REQUIRE(rel.defined() && rel.dtype() == ZS_BF16 && rel.numel() == heads * p * (kh + width), "rel: bf16 [heads, P, kh + W]");
  if (p == 0) return zs::status();
  ZS_REQUIRE(kh + width <= 128 && hidden <= 4 * 8 * 32, "kh + W at most 128, hidden at most 1,024");
  ocr_relpos_kernel<<<static_cast<unsigned>(p), static_cast<unsigned>((kh + width + 31) / 32 * 32),
                      static_cast<size_t>(hidden) * 4, zs::stream()>>>(
      qkv.data_ptr<__nv_bfloat16>(), map.data_ptr<int>(), table_h.data_ptr<__nv_bfloat16>(), table_w.data_ptr<__nv_bfloat16>(),
      rel.data_ptr<__nv_bfloat16>(), static_cast<int>(hidden), static_cast<int>(p), static_cast<int>(len), static_cast<int>(width));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
