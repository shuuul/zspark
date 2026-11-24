// DFlash2 drafter, its element-wise steps: the two-tap grouped dynamic convolution over a block's
// rows (`dconv`), q and k normed a head and rotated at the row's position with v copied (`prep`), a row's
// largest logits (`top`), and those picks dotted again against the stored head's rows (`rescore`). The context
// is flat (position p at row p) or a ring (p at p mod capacity). GLM-5.3-Flash and Qwen3.8 Flash Next both
// launch these.
// Semantics from TensorFold 0.6.6 (cb2ebf0) families/glm5_next/cuda/dflash2.py (_dconv_kernel, _prep_kernel,
// Drafter._rotary, Drafter._slots; Triton), Apache-2.0; rewritten for zspark in CUDA C++.
// Hand-written because none of it is a GEMM CUTLASS has: element-wise maps and a norm over 128 values.
//
// Compiled `--fmad=false`: every fused multiply-add is written (`fmaf`, `fma`), so that
// zig/src/decode/dflash2/kernels_ref.zig is the same arithmetic to the bit. What is zspark's and not TensorFold's:
// - `dconv` and the rotation fuse the second product into the first, rounded alone (zig/kernels/glm5/hc.cu's
//   reading of a Triton sum of products); `prep`'s 1 / sqrt is a division, not `rsqrt`.
// - The rotary phases are made here from the position and not uploaded: inv_freq by square roots, the phase's
//   sine and cosine by `sincos_of`, in fp64 operations that IEEE 754 rounds one way (stated at each).
// - `prep` writes k and v where the context keeps them, at the row's slot, in place of an index copy after it.
// - `top` is an exact selection with the lower token among equals, where `torch.topk` promises no order.
// - `rescore` is zspark's alone (spec 098 WS-04): the four-bit head's picks held to the stored head's rows,
//   which TensorFold's drafter does not do.
// A row of `prep` is a function of its own values and position; the block's attention is
// dflash2_attend.cu's. C ABI names are `zs_dflash2_*`, model-free (spec 136 WS-03).

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

constexpr int HD = 128;         // a head's values
constexpr int HALF = HD / 2;    // a rotary pair is values d and d + HALF
constexpr int LOG_HALF = 6;

__device__ __forceinline__ float value(__nv_bfloat16 x) { return __bfloat162float(x); }

// 1 / n!, n to 18: the factorial is exact in fp64 and the quotient is rounded once.
__host__ __device__ constexpr double inverse_factorial(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= i;
  return 1.0 / f;
}

// The coefficient of x^n in sin or cos: (-1)^(n / 2) / n!.
__host__ __device__ constexpr double term(int n) { return (n / 2) % 2 ? -inverse_factorial(n) : inverse_factorial(n); }

// theta^(-d / HALF) for d under HALF = 2^6, as fp32. With r_b = theta^(2^b / 64), six square roots of theta
// one after another (r_5 first), the product p of the r_b whose bit b is set in d, taken from b = 5 down, and
// 1 / p rounded to fp32: square root, product and quotient are each rounded once by IEEE 754, so the host gives
// the same bits. TensorFold's is torch's fp32 `1 / theta ** (2 d / 128)`, a rounding or two away.
__device__ __forceinline__ float inv_freq(double theta, int d) {
  double r = theta, p = 1.0;
  for (int b = LOG_HALF - 1; b >= 0; --b) {
    r = sqrt(r);
    if ((d >> b) & 1) p = p * r;
  }
  return static_cast<float>(1.0 / p);
}

// sin and cos of x (an fp32 phase, 0 to 2^24) in fp64, rounded to fp32. k = floor(x 2/pi + 1/2); r = x - k pi/2
// with pi/2 in two parts, each step one fma; z = r r; sin r = r + (r z) P(z), P the terms x^3 to x^15 over x^3,
// and cos r = 1 + z Q(z), Q the terms x^2 to x^16 over x^2, both by Horner from the highest term, one fma a
// step; then the quadrant k mod 4. Within 3e-16 of the true value (|r| <= 0.79: the first terms left out are
// 5e-17 and 2e-18), so the fp32 is the true value's rounding but for a phase in 1e8 or so that falls on a tie,
// a rounding away. No library sine: CUDA's and the host's differ in their last bits.
__device__ __forceinline__ void sincos_of(double x, float* sine, float* cosine) {
  const double k = floor(fma(x, 0x1.45f306dc9c883p-1, 0.5));
  double r = fma(-k, 0x1.921fb54442d18p+0, x);
  r = fma(-k, 0x1.1a62633145c07p-54, r);
  const double z = r * r;
  double p = term(15);
  for (int n = 13; n >= 3; n -= 2) p = fma(p, z, term(n));
  const double s = fma(r * z, p, r);
  p = term(16);
  for (int n = 14; n >= 2; n -= 2) p = fma(p, z, term(n));
  const double c = fma(z, p, 1.0);
  switch (static_cast<long long>(k) & 3) {
    case 0: *sine = static_cast<float>(s), *cosine = static_cast<float>(c); break;
    case 1: *sine = static_cast<float>(c), *cosine = static_cast<float>(-s); break;
    case 2: *sine = static_cast<float>(-s), *cosine = static_cast<float>(-c); break;
    default: *sine = static_cast<float>(-c), *cosine = static_cast<float>(s); break;
  }
}

// Thread (row, c): the kernel's two taps of the channel are bf16(base + dyn of the channel's group), and
// y = bf16(x k0 + prev k1), prev the row before (0 for the block's first row); with a residual
// out = bf16(res + y). DYN [R, 2, 2, G] (branch, tap, group), BASE [2, 2, D] (branch, tap, channel).
__global__ void dconv_kernel(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ DYN,
                             const __nv_bfloat16* __restrict__ BASE, const __nv_bfloat16* RES, __nv_bfloat16* OUT,
                             int R, int D, int G, int BRANCH) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const int r = static_cast<int>(i / D), c = static_cast<int>(i % D);
  const __nv_bfloat16* dyn = DYN + ((size_t)r * 2 + BRANCH) * 2 * G + c / (D / G);
  const __nv_bfloat16* base = BASE + (size_t)BRANCH * 2 * D + c;
  const float k0 = bf(value(base[0]) + value(dyn[0]));
  const float k1 = bf(value(base[D]) + value(dyn[G]));
  const float prev = r > 0 ? value(X[i - D]) : 0.0f;
  float y = bf(fmaf(prev, k1, value(X[i]) * k0));
  if (RES != nullptr) y = bf(value(RES[i]) + y);
  OUT[i] = __float2bfloat16_rn(y);
}

// Block (row, head) of HALF threads, thread d the pair (d, d + HALF) = (a, b) of the head's 128 values: the
// row's H query heads from QR [R, H * 128], then its KV key heads and its KV value heads from KVR
// [R, 2 KV * 128]. A query or key head:
//   rstd = 1 / sqrt((fold of the 64 a a + fold of the 64 b b) / 128 + eps), the squares exact in fp32;
//   a = bf16(a rstd w[d]), b = bf16(b rstd w[d + HALF]), w the query's or the key's norm;
//   phase = fp32(position) inv_freq(d) in fp32, (sin, cos) = sincos_of(phase);
//   out[d] = bf16(a cos - b sin), out[d + HALF] = bf16(b cos + a sin), the second product fused.
// A value head is copied. The row stands at position POS + row: its query goes to QO [H, L, 128] row `row`, its
// key and value to KO, VO [KV, CAP, 128] at the position's slot (the position, or modulo CAP in a ring; a flat
// slot past CAP is not written).
__global__ void __launch_bounds__(HALF)
    dflash2_prep_kernel(const __nv_bfloat16* __restrict__ QR, const __nv_bfloat16* __restrict__ KVR,
                const __nv_bfloat16* __restrict__ QN, const __nv_bfloat16* __restrict__ KN, const int* __restrict__ POS, __nv_bfloat16* __restrict__ QO,
                __nv_bfloat16* __restrict__ KO, __nv_bfloat16* __restrict__ VO, float eps, double theta, int H, int KV,
                int L, int CAP, int RING) {
  __shared__ float sa[HALF];
  __shared__ float sb[HALF];
  const int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  const __nv_bfloat16* src =
      head < H ? QR + ((size_t)row * H + head) * HD : KVR + ((size_t)row * 2 * KV + (head - H)) * HD;
  const long long position = (long long)POS[0] + row;
  const long long slot = RING ? position % CAP : position;
  if (head >= H + KV) {
    if (slot >= CAP) return;
    __nv_bfloat16* dst = VO + ((size_t)(head - H - KV) * CAP + slot) * HD;
    dst[d] = src[d];
    dst[HALF + d] = src[HALF + d];
    return;
  }
  const float a = value(src[d]), b = value(src[HALF + d]);
  sa[d] = a * a;
  sb[d] = b * b;
  __syncthreads();
  for (int n = HALF / 2; n >= 1; n >>= 1) {
    if (d < n) {
      sa[d] += sa[d + n];
      sb[d] += sb[d + n];
    }
    __syncthreads();
  }
  const float rstd = 1.0f / sqrtf((sa[0] + sb[0]) / HD + eps);
  const __nv_bfloat16* w = head < H ? QN : KN;
  const float na = bf(a * rstd * value(w[d])), nb = bf(b * rstd * value(w[HALF + d]));
  float sine, cosine;
  sincos_of(static_cast<float>(position) * inv_freq(theta, d), &sine, &cosine);
  __nv_bfloat16* dst;
  if (head < H) {
    dst = QO + ((size_t)head * L + row) * HD;
  } else {
    if (slot >= CAP) return;
    dst = KO + ((size_t)(head - H) * CAP + slot) * HD;
  }
  dst[d] = __float2bfloat16_rn(fmaf(-nb, sine, na * cosine));
  dst[HALF + d] = __float2bfloat16_rn(fmaf(na, sine, nb * cosine));
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

bool rows_of(const zs::View& x, int64_t rows, int64_t d) {
  return x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(0) == rows &&
         x.size(1) == d;
}

// A context's keys or values: bf16 [KV, capacity, 128].
bool context_of(const zs::View& c) {
  return c.defined() && c.dtype() == ZS_BF16 && c.dim() == 3 && c.size(0) >= 1 && c.size(0) <= 4096 &&
         c.size(1) >= 1 && c.size(1) <= INT32_MAX / HD && c.size(2) == HD;
}

// The four-bit head's picks of a draft row, dotted again against the main model's own rows of the
// vocabulary: the pick's token names the rank's row `token - OFFSET` (top's tokens are the whole
// vocabulary's, the rank's rows its half), and its value the fp32 sum of the products with the normed
// draft row, one `fmaf` a value in the row's order. A pick out of the rank's half is -inf: top leaves
// none, and the guard keeps a stray read out of the rows.
//
// A block a pick, grid (picks, rows), 32 threads (spec 143 WS-09): the warp stages the pick's row and the draft
// row in shared memory, 16 bytes a lane where the rows allow it, and lane 0 runs the sum, the same `fmaf`s in the
// same order as a thread a pick did. Shared memory: the two rows as bf16, 4 x COLS bytes (10,240 at Flash Next's
// 2,560, 24,576 at 6,144), nothing else; COLS above RESCORE_STAGED takes `rescore_kernel` below, a thread a pick
// reading global memory. The thread a pick was 124 us a launch of 7 rows x 16 picks (WS-03's trace): 7 blocks,
// each thread's 2-byte loads in a dependent loop, latency-bound.
constexpr int RESCORE_STAGED = 12288;

__global__ void rescore_staged_kernel(const int* TOKENS, const __nv_bfloat16* ROWS, const __nv_bfloat16* FINAL,
                                      float* VALUES, int COLS, int TOP, int OFFSET, int NROWS, int QUADS) {
  extern __shared__ __align__(16) __nv_bfloat16 staged[];
  const int pick = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
  const int token = TOKENS[(size_t)row * TOP + pick] - OFFSET;
  float* out = VALUES + (size_t)row * TOP + pick;
  if (token < 0 || token >= NROWS) {
    if (lane == 0) *out = __int_as_float(0xff800000);
    return;
  }
  const __nv_bfloat16* w = ROWS + (size_t)token * COLS;
  const __nv_bfloat16* x = FINAL + (size_t)row * COLS;
  __nv_bfloat16 *sx = staged, *sw = staged + COLS;
  if (QUADS) {
    for (int d = lane * 8; d < COLS; d += 32 * 8) {
      *reinterpret_cast<uint4*>(sx + d) = *reinterpret_cast<const uint4*>(x + d);
      *reinterpret_cast<uint4*>(sw + d) = *reinterpret_cast<const uint4*>(w + d);
    }
  } else {
    for (int d = lane; d < COLS; d += 32) {
      sx[d] = x[d];
      sw[d] = w[d];
    }
  }
  __syncwarp();
  if (lane != 0) return;
  float acc = 0.0f;
#pragma unroll 8
  for (int d = 0; d < COLS; ++d) acc = fmaf(value(sx[d]), value(sw[d]), acc);
  *out = acc;
}

// The same, a thread a pick from global memory: rows wider than RESCORE_STAGED.
__global__ void rescore_kernel(const int* TOKENS, const __nv_bfloat16* ROWS, const __nv_bfloat16* FINAL,
                               float* VALUES, int COLS, int TOP, int OFFSET, int NROWS) {
  const int row = blockIdx.x, t = threadIdx.x;
  if (t >= TOP) return;
  const int token = TOKENS[(size_t)row * TOP + t] - OFFSET;
  float acc = __int_as_float(0xff800000);
  if (token >= 0 && token < NROWS) {
    const __nv_bfloat16* w = ROWS + (size_t)token * COLS;
    const __nv_bfloat16* x = FINAL + (size_t)row * COLS;
    acc = 0.0f;
    for (int d = 0; d < COLS; ++d) acc = fmaf(value(x[d]), value(w[d]), acc);
  }
  VALUES[(size_t)row * TOP + t] = acc;
}

// A row's largest, in the order `top` defines: a candidate (v, k) comes before (w, m) when v > w, or v == w (the
// zeros equal) and k < m; k < 0 is none, which comes after every number; a NaN is never a candidate. It is a total
// order on the candidates, so the picks are the same whatever the tree that compares them.
__device__ __forceinline__ bool before(float ov, int o, float mv, int m) {
  return o >= 0 && (m < 0 || ov > mv || (ov == mv && o < m));
}

// The block's first candidate of every thread's (v, k): a butterfly in each warp, then the warps' in warp 0.
// sv and sk hold a value a warp.
template <int THREADS>
__device__ __forceinline__ void block_first(float& v, int& k, float* sv, int* sk) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, m);
    const int o = __shfl_xor_sync(0xffffffffu, k, m);
    if (before(ov, o, v, k)) { v = ov; k = o; }
  }
  if (lane == 0) { sv[warp] = v; sk[warp] = k; }
  __syncthreads();
  if (warp == 0) {
    v = lane < THREADS / 32 ? sv[lane] : 0.0f;
    k = lane < THREADS / 32 ? sk[lane] : -1;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, m);
      const int o = __shfl_xor_sync(0xffffffffu, k, m);
      if (before(ov, o, v, k)) { v = ov; k = o; }
    }
    if (lane == 0) { sv[0] = v; sk[0] = k; }
  }
  __syncthreads();
  v = sv[0];
  k = sk[0];
  __syncthreads();
}

// The first TOP of n candidates in that order, `get(j, v, k)` the j-th: pick by pick, each thread's first among
// those after the last pick, then the block's. Thread 0 gets each pick in `put(p, v, k)`, k < 0 once none is left.
template <int THREADS, typename Get, typename Put>
__device__ __forceinline__ void select_first(Get get, int n, int top, Put put) {
  __shared__ float sv[THREADS / 32];
  __shared__ int sk[THREADS / 32];
  const int t = threadIdx.x;
  float last_v = __int_as_float(0x7f800000);
  int last_k = -1;
  for (int p = 0; p < top; ++p) {
    float v = 0.0f;
    int k = -1;
    for (int j = t; j < n; j += THREADS) {
      float c;
      int key;
      get(j, c, key);
      // after the last pick; a NaN is after nothing
      if (key < 0 || !(c < last_v || (c == last_v && key > last_k))) continue;
      if (before(c, key, v, k)) { v = c; k = key; }
    }
    block_first<THREADS>(v, k, sv, sk);
    if (k < 0) {
      // none left, and none after
      if (t == 0) for (int q = p; q < top; ++q) put(q, 0.0f, -1);
      return;
    }
    if (t == 0) put(p, v, k);
    last_v = v;
    last_k = k;
  }
}

// A row's `TOP` largest logits, largest first and the lower token among equals; a NaN is none. An exact
// selection, chain.zig's `largest` on the host; a row short of TOP numbers ends in -inf at the first token.
//
// Split along the row (spec 143 WS-09): grid (slices, rows), a block takes TOP_SLICE columns and leaves its first
// TOP in `top_slice_kernel`, then `top_merge_kernel` takes a row's first TOP of its slices' candidates. The
// order is total, so a row's picks are the bits of one block's over the whole row (the kernel before WS-09: a
// block a row, 1,024 threads, a tree of shared memory a pick, 170 us a launch of 7 rows x 124,160 columns on 7 of
// 48 SMs). A row of one slice takes no merge: its block writes the picks. Shared memory: a value and a token a
// warp (64 B), nothing else; scratch: the candidates, fp32 and int32 [rows, slices, TOP], from the caller.
constexpr int TOP_THREADS = 256;
constexpr int TOP_SLICE = 4096;

__device__ __forceinline__ void final_pick(float* values, int* tokens, int p, float v, int k, int offset) {
  values[p] = k < 0 ? __int_as_float(0xff800000) : v;
  tokens[p] = offset + (k < 0 ? 0 : k);
}

__global__ void __launch_bounds__(TOP_THREADS) top_slice_kernel(const __nv_bfloat16* LOGITS, float* VALUES,
                                                                int* TOKENS, float* CV, int* CK, int COLS, int TOP,
                                                                int OFFSET) {
  const int slice = blockIdx.x, row = blockIdx.y, slices = gridDim.x;
  const int first = slice * TOP_SLICE;
  const int n = COLS - first < TOP_SLICE ? COLS - first : TOP_SLICE;
  const __nv_bfloat16* x = LOGITS + (size_t)row * COLS + first;
  auto get = [&](int j, float& c, int& key) {
    c = value(x[j]);
    key = first + j;
  };
  if (slices == 1) {
    float* values = VALUES + (size_t)row * TOP;
    int* tokens = TOKENS + (size_t)row * TOP;
    select_first<TOP_THREADS>(get, n, TOP, [&](int p, float v, int k) { final_pick(values, tokens, p, v, k, OFFSET); });
  } else {
    const size_t at = ((size_t)row * slices + slice) * TOP;
    select_first<TOP_THREADS>(get, n, TOP, [&](int p, float v, int k) {
      CV[at + p] = v;
      CK[at + p] = k;
    });
  }
}

__global__ void __launch_bounds__(TOP_THREADS) top_merge_kernel(float* VALUES, int* TOKENS, const float* CV,
                                                                const int* CK, int SLICES, int TOP, int OFFSET) {
  const int row = blockIdx.x;
  const size_t at = (size_t)row * SLICES * TOP;
  float* values = VALUES + (size_t)row * TOP;
  int* tokens = TOKENS + (size_t)row * TOP;
  select_first<TOP_THREADS>(
      [&](int j, float& c, int& key) {
        c = CV[at + j];
        key = CK[at + j];
      },
      SLICES * TOP, TOP, [&](int p, float v, int k) { final_pick(values, tokens, p, v, k, OFFSET); });
}

int64_t top_slices(int64_t cols) { return (cols + TOP_SLICE - 1) / TOP_SLICE; }

}  // namespace

extern "C" {

// x bf16 [R, D], dyn bf16 [R, 4 * G] (a row's branch, tap, group: `kernel_projection`'s output), base bf16
// [2, 2, D], res bf16 [R, D] or null -> out bf16 [R, D], which may be res and may not be x. G divides D.
int zs_dflash2_dconv(const zs_view* x_, const zs_view* dyn_, const zs_view* base_, const zs_view* res_,
                          const zs_view* out_, int64_t branch) {
  zs::begin();
  const zs::View x(x_), dyn(dyn_), base(base_), res(res_), out(out_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(0) >= 1 &&
             x.size(1) >= 1 && x.numel() <= INT32_MAX, "x: bf16 [R, D]");
  const int64_t rows = x.size(0), d = x.size(1);
  ZS_REQUIRE(dyn.defined() && dyn.dtype() == ZS_BF16 && dyn.dim() == 2 && dyn.is_contiguous() && dyn.size(0) == rows &&
             dyn.size(1) >= 4 && dyn.size(1) % 4 == 0 && d % (dyn.size(1) / 4) == 0, "dyn: bf16 [R, 4 * G], G dividing D");
  ZS_REQUIRE(base.defined() && base.dtype() == ZS_BF16 && base.numel() == 4 * d, "base: bf16 [2, 2, D]");
  ZS_REQUIRE(!res.defined() || rows_of(res, rows, d), "res: bf16 [R, D]");
  ZS_REQUIRE(rows_of(out, rows, d) && out.data_ptr() != x.data_ptr(), "out: bf16 [R, D], not x");
  ZS_REQUIRE(branch == 0 || branch == 1, "branch: 0 or 1");
  dconv_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), dyn.data_ptr<__nv_bfloat16>(), base.data_ptr<__nv_bfloat16>(),
      res.defined() ? res.data_ptr<__nv_bfloat16>() : nullptr, out.data_ptr<__nv_bfloat16>(), static_cast<int>(rows),
      static_cast<int>(d), static_cast<int>(dyn.size(1) / 4), static_cast<int>(branch));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// q_rows bf16 [R, H * 128], a row's H query heads, or null for context rows (H is 0, and q is null); kv_rows
// bf16 [R, 2 KV * 128], a row's KV key heads, then its KV value heads; q_norm, k_norm bf16 [128]; pos int32
// [1], read on the device, row r standing at position pos + r (under 2^24). q bf16 [H, L, 128], L >= R, takes
// row r at its row r; k and v bf16 [KV, capacity, 128] take it at row pos + r, or (pos + r) mod capacity when
// `ring`. Flat, pos + R <= capacity is the caller's word.
int zs_dflash2_prep(const zs_view* q_rows_, const zs_view* kv_rows_, const zs_view* q_norm_,
                         const zs_view* k_norm_, const zs_view* pos_, const zs_view* q_, const zs_view* k_,
                         const zs_view* v_, double eps, double theta, int64_t ring) {
  zs::begin();
  const zs::View qr(q_rows_), kvr(kv_rows_), qn(q_norm_), kn(k_norm_), pos(pos_), q(q_), k(k_), v(v_);
  ZS_REQUIRE(context_of(k) && context_of(v) && v.size(0) == k.size(0) && v.size(1) == k.size(1),
             "k, v: bf16 [KV, capacity, 128]");
  const int64_t kv = k.size(0), cap = k.size(1);
  ZS_REQUIRE(kvr.defined() && kvr.dtype() == ZS_BF16 && kvr.dim() == 2 && kvr.is_contiguous() && kvr.size(0) >= 1 &&
             kvr.size(0) <= INT32_MAX && kvr.size(1) == 2 * kv * HD, "kv_rows: bf16 [R, 2 KV * 128]");
  const int64_t rows = kvr.size(0);
  ZS_REQUIRE(qr.defined() == q.defined(), "q_rows and q: both, or neither for context rows");
  ZS_REQUIRE(!qr.defined() || (qr.dtype() == ZS_BF16 && qr.dim() == 2 && qr.is_contiguous() && qr.size(0) == rows &&
                               qr.size(1) >= HD && qr.size(1) % HD == 0 && qr.size(1) / HD <= 4096),
             "q_rows: bf16 [R, H * 128]");
  const int64_t heads = qr.defined() ? qr.size(1) / HD : 0;
  ZS_REQUIRE(qn.defined() && qn.dtype() == ZS_BF16 && qn.numel() == HD && kn.defined() && kn.dtype() == ZS_BF16 &&
             kn.numel() == HD, "q_norm, k_norm: bf16 [128]");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(heads == 0 || (q.defined() && q.dtype() == ZS_BF16 && q.dim() == 3 && q.size(0) == heads &&
                            q.size(1) >= rows && q.size(1) <= INT32_MAX && q.size(2) == HD),
             "q: bf16 [H, R or more, 128]");
  ZS_REQUIRE(theta > 1.0, "theta above 1");
  dflash2_prep_kernel<<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>(heads + 2 * kv)), HALF, 0, zs::stream()>>>(
      heads ? qr.data_ptr<__nv_bfloat16>() : nullptr, kvr.data_ptr<__nv_bfloat16>(), qn.data_ptr<__nv_bfloat16>(),
      kn.data_ptr<__nv_bfloat16>(),
      pos.data_ptr<int>(), heads ? q.data_ptr<__nv_bfloat16>() : nullptr, k.data_ptr<__nv_bfloat16>(),
      v.data_ptr<__nv_bfloat16>(), static_cast<float>(eps), theta, static_cast<int>(heads), static_cast<int>(kv),
      heads ? static_cast<int>(q.size(1)) : 0, static_cast<int>(cap), ring ? 1 : 0);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// The int32 values `zs_dflash2_top`'s scratch holds for `rows` rows of `cols` logits and `top` picks: a candidate's
// value and token a slice (0 where a row is one slice).
int64_t zs_dflash2_top_scratch(int64_t rows, int64_t cols, int64_t top) {
  const int64_t slices = top_slices(cols);
  return slices > 1 ? 2 * rows * slices * top : 0;
}

// logits bf16 [rows, tokens] -> a row's `top` largest, values fp32 and tokens int32 [rows, top] (`offset` the row's
// first token); scratch int32 of `zs_dflash2_top_scratch` values, or null where that is 0.
int zs_dflash2_top(const zs_view* logits_, const zs_view* values_, const zs_view* tokens_, const zs_view* scratch_,
                   int64_t offset) {
  zs::begin();
  const zs::View logits(logits_), values(values_), tokens(tokens_), scratch(scratch_);
  ZS_REQUIRE(logits.defined() && logits.dtype() == ZS_BF16 && logits.dim() == 2 && logits.is_contiguous() &&
             logits.size(0) >= 1 && logits.size(0) <= 65535 && logits.size(1) >= 1 && logits.size(1) < (1LL << 30),
             "logits: bf16 [rows, tokens]");
  const int64_t rows = logits.size(0);
  ZS_REQUIRE(values.defined() && values.dtype() == ZS_F32 && values.dim() == 2 && values.is_contiguous() &&
             values.size(0) == rows && values.size(1) >= 1 && values.size(1) <= logits.size(1),
             "values: fp32 [rows, top], top tokens at most");
  ZS_REQUIRE(tokens.defined() && tokens.dtype() == ZS_I32 && tokens.dim() == 2 && tokens.is_contiguous() &&
             tokens.size(0) == rows && tokens.size(1) == values.size(1), "tokens: int32 [rows, top]");
  ZS_REQUIRE(offset >= 0 && offset + logits.size(1) < (1LL << 31), "offset: the row's first token");
  const int64_t cols = logits.size(1), top = values.size(1), slices = top_slices(cols);
  const int64_t need = zs_dflash2_top_scratch(rows, cols, top);
  ZS_REQUIRE(need == 0 || (scratch.defined() && scratch.dtype() == ZS_I32 && scratch.is_contiguous() &&
                           scratch.numel() >= need && need < (1LL << 31)),
             "scratch: int32 of zs_dflash2_top_scratch values");
  float* cv = need ? reinterpret_cast<float*>(scratch.data_ptr<int>()) : nullptr;
  int* ck = need ? scratch.data_ptr<int>() + need / 2 : nullptr;
  top_slice_kernel<<<dim3(static_cast<unsigned>(slices), static_cast<unsigned>(rows)), TOP_THREADS, 0, zs::stream()>>>(
      logits.data_ptr<__nv_bfloat16>(), values.data_ptr<float>(), tokens.data_ptr<int>(), cv, ck,
      static_cast<int>(cols), static_cast<int>(top), static_cast<int>(offset));
  ZS_CUDA_LAUNCH_CHECK();
  if (slices > 1) {
    top_merge_kernel<<<static_cast<unsigned>(rows), TOP_THREADS, 0, zs::stream()>>>(
        values.data_ptr<float>(), tokens.data_ptr<int>(), cv, ck, static_cast<int>(slices), static_cast<int>(top),
        static_cast<int>(offset));
    ZS_CUDA_LAUNCH_CHECK();
  }
  return zs::status();
}

// tokens int32 [rows, top] of whole-vocabulary ids (top's), rows bf16 [half, dims] the rank's rows of the
// vocabulary as the main checkpoint stores them, final bf16 [rows, dims] the normed draft rows the four-bit
// head read -> values fp32 [rows, top], each pick dotted again with its row (`offset` the rank's first
// token). The four-bit head's copy stays the proposer; these values are the stored head's at its picks.
int zs_dflash2_rescore(const zs_view* tokens_, const zs_view* rows_, const zs_view* final_,
                            const zs_view* values_, int64_t offset) {
  zs::begin();
  const zs::View tokens(tokens_), head(rows_), fin(final_), values(values_);
  ZS_REQUIRE(tokens.defined() && tokens.dtype() == ZS_I32 && tokens.dim() == 2 && tokens.is_contiguous() &&
                 tokens.size(0) >= 1 && tokens.size(0) <= 65535, "tokens: int32 [rows, top]");
  const int64_t rows = tokens.size(0), top = tokens.size(1);
  ZS_REQUIRE(top >= 1 && top <= 128, "top: 1 to 128");
  ZS_REQUIRE(head.defined() && head.dtype() == ZS_BF16 && head.dim() == 2 && head.is_contiguous() &&
                 head.size(0) >= 1 && head.size(0) <= INT32_MAX && head.size(1) >= 1 && head.size(1) <= INT32_MAX,
             "rows: bf16 [half, dims]");
  const int64_t dims = head.size(1);
  ZS_REQUIRE(rows_of(fin, rows, dims), "final: bf16 [rows, dims]");
  ZS_REQUIRE(values.defined() && values.dtype() == ZS_F32 && values.dim() == 2 && values.is_contiguous() &&
                 values.size(0) == rows && values.size(1) == top, "values: fp32 [rows, top]");
  ZS_REQUIRE(offset >= 0 && offset + head.size(0) < (1LL << 31), "offset: the rank's first token");
  if (dims <= RESCORE_STAGED) {
    const bool quads = dims % 8 == 0 && (reinterpret_cast<uintptr_t>(head.data_ptr()) & 15) == 0 &&
                       (reinterpret_cast<uintptr_t>(fin.data_ptr()) & 15) == 0;
    rescore_staged_kernel<<<dim3(static_cast<unsigned>(top), static_cast<unsigned>(rows)), 32,
                            static_cast<size_t>(4 * dims), zs::stream()>>>(
        tokens.data_ptr<int>(), head.data_ptr<__nv_bfloat16>(), fin.data_ptr<__nv_bfloat16>(),
        values.data_ptr<float>(), static_cast<int>(dims), static_cast<int>(top), static_cast<int>(offset),
        static_cast<int>(head.size(0)), quads ? 1 : 0);
  } else {
    rescore_kernel<<<static_cast<unsigned>(rows), 128, 0, zs::stream()>>>(
        tokens.data_ptr<int>(), head.data_ptr<__nv_bfloat16>(), fin.data_ptr<__nv_bfloat16>(),
        values.data_ptr<float>(), static_cast<int>(dims), static_cast<int>(top), static_cast<int>(offset),
        static_cast<int>(head.size(0)));
  }
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
