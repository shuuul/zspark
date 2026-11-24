// GLM-5.3-Flash's hyper-connections and norms. The residual is four streams of D values a row. A site reads
// them out (24 mix dots over the streams, a Sinkhorn over 16 of them, the collapsed row normed) and, once the
// branch has run on both ranks, writes it back into the four. Every value is a function of its own row alone:
// fp32 sums in an order the shapes fix, each bf16 rounding where the model has one.
// Semantics from TensorFold 0.6.6 (cb2ebf0) families/glm5_next/cuda/glue.py (_rmsnorm, _hc_partial, _hc_finish,
// _hc_post, _residual_add, _stream_mean; Triton), Apache-2.0; rewritten for zspark in CUDA C++, without the
// 64-group sums only the MLX 4-bit matmul reads. Hand-written because none of it is a GEMM: 24 dots a row with
// the streams' squares beside them, a 4x4 iteration, and element-wise maps.
//
// Two things are fixed here that the Triton source leaves to its compiler, and TensorFold's exported tensors
// have not yet been held against either (spec 044 WS-07):
// - A sum over a vector folds its upper half onto its lower, down to one value (`fold`): what Flash Next's
//   write-back found Triton's `tl.sum` to do for 256 squares (zig/kernels/qwen4/hc.cu).
// - A product added to a sum is one fused multiply-add, written as `fmaf`, the first product of a chain
//   rounded on its own.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/copies.cuh"
#include "../common/glue.cuh"
#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_copies;

using namespace common_math;

using namespace common_glue;

constexpr int STREAMS = 4;             // the kernels name the streams one by one, as glue.py does
constexpr int MIX = STREAMS * (STREAMS + 2);  // a site's dots: 4 for the read-out, 4 for the write, 16 among them
constexpr int PARTS = 16;              // glue.py's HC_BLOCKS: the K blocks a row's dots are summed over
constexpr int SUB = 128;               // values of a block one step folds
constexpr int MAX_STEPS = 8;           // steps a block: S * D / (PARTS * SUB), 8 at D 4096
constexpr int ENTRY = 32;              // a part's row: the 24 dots, then the squares' sum at 24
constexpr int MAX_D = 4096;            // a row a block folds in shared memory
constexpr int THREADS = 256;

// The sum of N values, the upper half folded onto the lower until one is left: at(i) + at(i + N / 2) first.
// Seen from the top that is the even indices' sum plus the odd ones'.
template <int N, class At>
__device__ __forceinline__ float fold(const At& at, int first, int step) {
  if constexpr (N == 1) {
    return at(first);
  } else {
    return fold<N / 2>(at, first, step * 2) + fold<N / 2>(at, first + step, step * 2);
  }
}

// Four of a sum, folded: (a + c) + (b + d).
__device__ __forceinline__ float fold4(float a, float b, float c, float d) { return (a + c) + (b + d); }

// Block (row, b): block b of the row's WIDE values, KB = WIDE / PARTS of them in steps of SUB. Thread (m, t)
// folds step t's products with dot m's weights (bf16 by bf16, exact in fp32); a dot is its steps' sums added in
// order. The last 64 threads sum each column's squares over the steps in order, and those fold. part
// [R, PARTS, ENTRY].
__global__ void __launch_bounds__(THREADS)
    mix_kernel(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ FN, float* __restrict__ part,
               int WIDE) {
  __shared__ float dots[MIX * MAX_STEPS];
  __shared__ float squares[SUB];
  const int r = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  const int KB = WIDE / PARTS, steps = KB / SUB;
  const __nv_bfloat16* x = X + (size_t)r * WIDE + (size_t)b * KB;
  if (tid < MIX * steps) {
    const int m = tid / steps, t = tid % steps;
    const __nv_bfloat16* w = FN + (size_t)m * WIDE + (size_t)b * KB + t * SUB;
    const __nv_bfloat16* xs = x + t * SUB;
    dots[tid] = fold<SUB>([&](int k) { return __bfloat162float(w[k]) * __bfloat162float(xs[k]); }, 0, 1);
  } else if (tid >= THREADS - 64) {
    for (int k = tid - (THREADS - 64); k < SUB; k += 64) {
      float total = 0.f;
      for (int t = 0; t < steps; ++t) {
        const float v = __bfloat162float(x[t * SUB + k]);
        total += v * v;
      }
      squares[k] = total;
    }
  }
  __syncthreads();
  float* out = part + ((size_t)r * PARTS + b) * ENTRY;
  if (tid < MIX) {
    float acc = 0.f;
    for (int t = 0; t < steps; ++t) acc += dots[tid * steps + t];
    out[tid] = acc;
  } else if (tid == MIX) {
    out[MIX] = fold<SUB>([&](int k) { return squares[k]; }, 0, 1);
  }
}

// Block r: the row's read-out. Its first thread adds the parts in order, scales the dots by the streams' rms
// and makes the gates: pre = sigmoid + hc_eps for the collapse, post = 2 sigmoid for the write, and comb, a
// softmax over each row of the 4x4 then ITERS rounds of columns and rows brought to sum to one (the first
// round is the softmax and the columns). Then every thread: the collapsed value bf16(sum of pre_s x_s), its
// rms over D (squares folded), and normed = bf16(w bf16(c / rms)).
template <class W>
__global__ void __launch_bounds__(THREADS)
    pre_kernel(const __nv_bfloat16* __restrict__ X, const float* __restrict__ part, const float* __restrict__ base,
               const float* __restrict__ scale, const W* __restrict__ norm, __nv_bfloat16* __restrict__ out,
               float* __restrict__ post, float* __restrict__ comb, float eps_norm, float hc_eps, int D, int ITERS) {
  __shared__ float sq[MAX_D];
  __shared__ __nv_bfloat16 row[MAX_D];
  __shared__ float pre[STREAMS];
  __shared__ float inverse;
  const int r = blockIdx.x, tid = threadIdx.x;
  if (tid == 0) {
    float mix[MIX], ss = 0.f;
    for (int m = 0; m < MIX; ++m) mix[m] = 0.f;
    for (int b = 0; b < PARTS; ++b) {
      const float* p = part + ((size_t)r * PARTS + b) * ENTRY;
      for (int m = 0; m < MIX; ++m) mix[m] += p[m];
      ss += p[MIX];
    }
    const float rinv = 1.0f / sqrtf(ss / (STREAMS * D) + eps_norm);
    for (int m = 0; m < MIX; ++m) mix[m] = fmaf(mix[m] * rinv, scale[m < STREAMS ? 0 : m < 2 * STREAMS ? 1 : 2], base[m]);
    for (int s = 0; s < STREAMS; ++s) {
      pre[s] = sigmoidf_(mix[s]) + hc_eps;
      post[(size_t)r * STREAMS + s] = 2.0f * sigmoidf_(mix[STREAMS + s]);
    }
    float c[STREAMS][STREAMS];
    for (int i = 0; i < STREAMS; ++i) {
      const float* l = mix + 2 * STREAMS + i * STREAMS;
      const float most = fmaxf(fmaxf(l[0], l[1]), fmaxf(l[2], l[3]));
      float e[STREAMS];
      for (int j = 0; j < STREAMS; ++j) e[j] = expf(l[j] - most);
      const float total = fold4(e[0], e[1], e[2], e[3]);
      for (int j = 0; j < STREAMS; ++j) c[i][j] = e[j] / total + hc_eps;
    }
    for (int round = 0; round < ITERS; ++round) {
      if (round > 0)
        for (int i = 0; i < STREAMS; ++i) {
          const float total = fold4(c[i][0], c[i][1], c[i][2], c[i][3]) + hc_eps;
          for (int j = 0; j < STREAMS; ++j) c[i][j] = c[i][j] / total;
        }
      for (int j = 0; j < STREAMS; ++j) {
        const float total = fold4(c[0][j], c[1][j], c[2][j], c[3][j]) + hc_eps;
        for (int i = 0; i < STREAMS; ++i) c[i][j] = c[i][j] / total;
      }
    }
    for (int i = 0; i < STREAMS; ++i)
      for (int j = 0; j < STREAMS; ++j) comb[(size_t)r * STREAMS * STREAMS + i * STREAMS + j] = c[i][j];
  }
  __syncthreads();
  const __nv_bfloat16* x = X + (size_t)r * STREAMS * D;
  for (int d = tid; d < D; d += THREADS) {
    float v = pre[0] * __bfloat162float(x[d]);
    for (int s = 1; s < STREAMS; ++s) v = fmaf(pre[s], __bfloat162float(x[s * D + d]), v);
    const __nv_bfloat16 c = __float2bfloat16_rn(v);
    row[d] = c;
    sq[d] = __bfloat162float(c) * __bfloat162float(c);
  }
  __syncthreads();
  fold_shared(sq, D);
  if (tid == 0) inverse = 1.0f / sqrtf(sq[0] / D + eps_norm);
  __syncthreads();
  for (int d = tid; d < D; d += THREADS)
    out[(size_t)r * D + d] =
        __float2bfloat16_rn(weight_of(norm[d]) * bf(__bfloat162float(row[d]) * inverse));
}

// `mix_kernel` with a step's values loaded eight at a time (ZSPARK_GLM_HC_OCTS, spec 073 WS-05): thread (m, t)
// makes 16 loads of 16 bytes from the weights and 16 from the row where the base makes 128 of two bytes from
// each, which is what bounds it (its SM's load slots, not memory: it costs the same from L2). The same bits: the
// fold's last three levels are over a value's index mod 8, so a lane folds its 16 values as the base does and
// the eight lanes fold after. X and FN aligned to 16 bytes.
struct Oct {
  float v[8];
};
__device__ __forceinline__ Oct operator+(const Oct& a, const Oct& b) {
  Oct o;
  for (int i = 0; i < 8; ++i) o.v[i] = a.v[i] + b.v[i];
  return o;
}
__device__ __forceinline__ Oct products(const __nv_bfloat16* w, const __nv_bfloat16* x) {
  const uint4 a = *reinterpret_cast<const uint4*>(w), b = *reinterpret_cast<const uint4*>(x);
  Oct o;
  o.v[0] = bf16_low(a.x) * bf16_low(b.x), o.v[1] = bf16_high(a.x) * bf16_high(b.x);
  o.v[2] = bf16_low(a.y) * bf16_low(b.y), o.v[3] = bf16_high(a.y) * bf16_high(b.y);
  o.v[4] = bf16_low(a.z) * bf16_low(b.z), o.v[5] = bf16_high(a.z) * bf16_high(b.z);
  o.v[6] = bf16_low(a.w) * bf16_low(b.w), o.v[7] = bf16_high(a.w) * bf16_high(b.w);
  return o;
}
template <int N, class At>
__device__ __forceinline__ Oct fold_octs(const At& at, int first, int step) {
  if constexpr (N == 1) {
    return at(first);
  } else {
    return fold_octs<N / 2>(at, first, step * 2) + fold_octs<N / 2>(at, first + step, step * 2);
  }
}

__global__ void __launch_bounds__(THREADS)
    mix_octs_kernel(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ FN,
                    float* __restrict__ part, int WIDE) {
  __shared__ float dots[MIX * MAX_STEPS];
  __shared__ float squares[SUB];
  const int r = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  const int KB = WIDE / PARTS, steps = KB / SUB;
  const __nv_bfloat16* x = X + (size_t)r * WIDE + (size_t)b * KB;
  if (tid < MIX * steps) {
    const int m = tid / steps, t = tid % steps;
    const __nv_bfloat16* w = FN + (size_t)m * WIDE + (size_t)b * KB + t * SUB;
    const __nv_bfloat16* xs = x + t * SUB;
    const Oct lanes = fold_octs<SUB / 8>([&](int c) { return products(w + 8 * c, xs + 8 * c); }, 0, 1);
    dots[tid] = fold<8>([&](int j) { return lanes.v[j]; }, 0, 1);
  } else if (tid >= THREADS - 64) {
    for (int k = tid - (THREADS - 64); k < SUB; k += 64) {
      float total = 0.f;
      for (int t = 0; t < steps; ++t) {
        const float v = __bfloat162float(x[t * SUB + k]);
        total += v * v;
      }
      squares[k] = total;
    }
  }
  __syncthreads();
  float* out = part + ((size_t)r * PARTS + b) * ENTRY;
  if (tid < MIX) {
    float acc = 0.f;
    for (int t = 0; t < steps; ++t) acc += dots[tid * steps + t];
    out[tid] = acc;
  } else if (tid == MIX) {
    out[MIX] = fold<SUB>([&](int k) { return squares[k]; }, 0, 1);
  }
}

// `mix_octs_kernel` for a prompt's window (spec 107 WS-03), from `rows_from` rows: alone 21.7 us for 61 at 32 rows,
// 27.7 for 83 at 64, but at 2 and 4 rows 9.4 and 10.6 for 7.5 and 10.0, and 14.1 for 15.3 at 8; a round's rows (2
// to 8) keep the octs kernel. At 2,048 rows the octs kernel took 2.09 ms a call in serve (2.17 alone), seven times its rows'
// 64 MB at the memory's rate: every block read its part of the 24 weight rows (48 KB) for one row, and its
// sixteen 16-byte loads of the row from L1 again for each of the 24 dots. Here a block holds part b for a range
// of rows: thread (m, t) keeps its step's 128 weights in registers through the range, and the rows come GROUP at
// a time into shared memory by cp.async, the next group while this one is summed. Each row's products, folds
// and sums are the octs kernel's expressions in its order: the same bits, whatever the range or the group.
// Shared memory, a block of 256 threads (202 registers: one block an SM, so the grid is 48 blocks):
//
//   region    bytes                          for
//   rows      2 x GROUP x KB x 2 = 32,768    two groups' x, KB = 1,024 values at D 4096: the one summed, the next
//   dots      GROUP x 24 x 8 x 4 = 6,144     a row's step sums, added in order after
//   squares   GROUP x 128 x 4 = 4,096        a row's column squares, folded after
//
// 43,008 B. The weights are 64 registers a thread, not shared memory. Alone at 2,048 rows 470 us for 2,173,
// 129 for 525 at 512 (`ZSPARK_GLM5_PROFILE=mix:2048`): issue-bound now (0.5 a scheduler, 96 M instructions,
// each row's x widened by all 24 dots' threads), not at its bytes (153 GB/s); a third group in flight the same.
constexpr int GROUP = 8;
constexpr int64_t rows_from = 16;
constexpr int ROW_OCTS = MAX_STEPS * SUB / 8;  // a row's part at most, in 16-byte octs
constexpr int RING = 2;                        // groups in shared memory: the one summed and the next (3: the same)
constexpr int RING_BYTES = RING * GROUP * ROW_OCTS * 16;

// Where a row's oct c sits in a group: step t's 16 octs trade places by c ^ t, so that the eight steps' threads
// of a warp, reading the same oct of their own steps, meet eight banks and not one (an 8-way conflict held the
// kernel at the shared memory's rate, 1.33 ms at 2,048 rows).
__device__ __forceinline__ int swizzled(int c) { return (c & ~15) | ((c & 15) ^ ((c >> 4) & 7)); }

__device__ __forceinline__ Oct products(uint4 a, uint4 b) {
  Oct o;
  o.v[0] = bf16_low(a.x) * bf16_low(b.x), o.v[1] = bf16_high(a.x) * bf16_high(b.x);
  o.v[2] = bf16_low(a.y) * bf16_low(b.y), o.v[3] = bf16_high(a.y) * bf16_high(b.y);
  o.v[4] = bf16_low(a.z) * bf16_low(b.z), o.v[5] = bf16_high(a.z) * bf16_high(b.z);
  o.v[6] = bf16_low(a.w) * bf16_low(b.w), o.v[7] = bf16_high(a.w) * bf16_high(b.w);
  return o;
}

__global__ void __launch_bounds__(THREADS)
    mix_rows_kernel(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ FN,
                    float* __restrict__ part, int R, int WIDE) {
  extern __shared__ __align__(16) uint4 rows_ring[];
  uint4 (*rows)[GROUP * ROW_OCTS] = reinterpret_cast<uint4 (*)[GROUP * ROW_OCTS]>(rows_ring);
  __shared__ float dots[GROUP][MIX * MAX_STEPS];
  __shared__ float squares[GROUP][SUB];
  const int b = blockIdx.y, tid = threadIdx.x;
  const int KB = WIDE / PARTS, steps = KB / SUB, octs = KB / 8;
  const int groups = (R + GROUP - 1) / GROUP;
  const bool dotter = tid < MIX * steps;
  const int m = dotter ? tid / steps : 0, t = dotter ? tid % steps : 0;
  uint4 held[SUB / 8];
  if (dotter) {
    const uint4* w = reinterpret_cast<const uint4*>(FN + (size_t)m * WIDE + (size_t)b * KB + t * SUB);
#pragma unroll
    for (int c = 0; c < SUB / 8; ++c) held[c] = w[c];
  }
  // group g's rows into buffer s: rows past R are not copied, and their stale bytes give sums nobody stores
  auto load = [&](int s, int g) {
    const int first = g * GROUP, count = min(GROUP, R - first);
    for (int i = tid; i < count * octs; i += THREADS) {
      const int j = i / octs, c = i % octs;
      cp16(&rows[s][j * octs + swizzled(c)], X + (size_t)(first + j) * WIDE + (size_t)b * KB + 8 * c);
    }
  };
#pragma unroll
  for (int k = 0; k < RING - 1; ++k) {
    const int g = blockIdx.x + k * gridDim.x;
    if (g < groups) load(k, g);
    cp_commit();
  }
  int s = 0;
  for (int g = blockIdx.x; g < groups; g += gridDim.x, s = s + 1 == RING ? 0 : s + 1) {
    const int first = g * GROUP, count = min(GROUP, R - first);
    // every thread has finished reading the last group's buffer, its dots and its squares
    __syncthreads();
    const int ahead = g + (RING - 1) * gridDim.x;
    if (ahead < groups) load(s == 0 ? RING - 1 : s - 1, ahead);
    cp_commit();
    cp_wait<RING - 1>();
    __syncthreads();
    if (dotter) {
      for (int j = 0; j < count; ++j) {
        const uint4* xs = rows[s] + j * octs + t * (SUB / 8);
        const Oct lanes = fold_octs<SUB / 8>([&](int c) { return products(held[c], xs[c ^ t]); }, 0, 1);
        dots[j][tid] = fold<8>([&](int k) { return lanes.v[k]; }, 0, 1);
      }
    } else if (tid >= THREADS - 64) {
      const __nv_bfloat16* staged = reinterpret_cast<const __nv_bfloat16*>(rows[s]);
      for (int j = 0; j < count; ++j)
        for (int k = tid - (THREADS - 64); k < SUB; k += 64) {
          float total = 0.f;
          for (int u = 0; u < steps; ++u) {
            const float v = __bfloat162float(staged[j * KB + u * SUB + (((k >> 3) ^ u) << 3) + (k & 7)]);
            total += v * v;
          }
          squares[j][k] = total;
        }
    }
    __syncthreads();
    if (tid < MIX) {
      for (int j = 0; j < count; ++j) {
        float acc = 0.f;
        for (int u = 0; u < steps; ++u) acc += dots[j][tid * steps + u];
        part[((size_t)(first + j) * PARTS + b) * ENTRY + tid] = acc;
      }
    } else if (tid >= 32 && tid < 32 + count) {
      const int j = tid - 32;
      part[((size_t)(first + j) * PARTS + b) * ENTRY + MIX] = fold<SUB>([&](int k) { return squares[j][k]; }, 0, 1);
    }
  }
  cp_wait<0>();
}

// `pre_kernel` with its first thread's work on the first warp's lanes (ZSPARK_GLM_HC_LANES, spec 073 WS-05):
// lane m adds part m's 16 blocks in order and scales its dot, lanes 0 to 7 make the gates, and lane (i, j) of
// the first 16 holds comb[i][j] through the rounds, a row's and a column's sums by two shuffles each. The base
// spends 0.96 us a round on one thread, 18 of its 27 us at 20 rounds. The same bits: fold4 is (a + c) + (b + d),
// which the lane two away and then the lane one away give in that order, a sum's two terms either way round;
// every other expression is the base's.
template <class W>
__global__ void __launch_bounds__(THREADS)
    pre_lanes_kernel(const __nv_bfloat16* __restrict__ X, const float* __restrict__ part,
                     const float* __restrict__ base, const float* __restrict__ scale,
                     const W* __restrict__ norm, __nv_bfloat16* __restrict__ out, float* __restrict__ post,
                     float* __restrict__ comb, float eps_norm, float hc_eps, int D, int ITERS) {
  __shared__ float sq[MAX_D];
  __shared__ __nv_bfloat16 row[MAX_D];
  __shared__ float mixed[32];
  __shared__ float pre[STREAMS];
  __shared__ float inverse;
  const int r = blockIdx.x, tid = threadIdx.x;
  if (tid < 32) {
    float acc = 0.f;
    if (tid <= MIX)
      for (int b = 0; b < PARTS; ++b) acc += part[((size_t)r * PARTS + b) * ENTRY + tid];
    const float ss = __shfl_sync(0xffffffffu, acc, MIX);
    const float rinv = 1.0f / sqrtf(ss / (STREAMS * D) + eps_norm);
    const int m = tid < MIX ? tid : 0;
    const float v = fmaf(acc * rinv, scale[m < STREAMS ? 0 : m < 2 * STREAMS ? 1 : 2], base[m]);
    if (tid < STREAMS) pre[tid] = sigmoidf_(v) + hc_eps;
    else if (tid < 2 * STREAMS) post[(size_t)r * STREAMS + tid - STREAMS] = 2.0f * sigmoidf_(v);
    mixed[tid] = v;
    __syncwarp();
    // lanes past the 16 run the rounds on ones, so that every lane of the warp meets every shuffle
    float c = 1.0f;
    if (tid < STREAMS * STREAMS) {
      const float* l = mixed + 2 * STREAMS + (tid / STREAMS) * STREAMS;
      const float most = fmaxf(fmaxf(l[0], l[1]), fmaxf(l[2], l[3]));
      float e[STREAMS];
      for (int j = 0; j < STREAMS; ++j) e[j] = expf(l[j] - most);
      const float total = fold4(e[0], e[1], e[2], e[3]);
      c = e[tid % STREAMS] / total + hc_eps;
    }
    for (int round = 0; round < ITERS; ++round) {
      if (round > 0) {
        const float pair = c + __shfl_xor_sync(0xffffffffu, c, 2);
        const float total = (pair + __shfl_xor_sync(0xffffffffu, pair, 1)) + hc_eps;
        c = c / total;
      }
      const float pair = c + __shfl_xor_sync(0xffffffffu, c, 8);
      const float total = (pair + __shfl_xor_sync(0xffffffffu, pair, 4)) + hc_eps;
      c = c / total;
    }
    if (tid < STREAMS * STREAMS) comb[(size_t)r * STREAMS * STREAMS + tid] = c;
  }
  __syncthreads();
  const __nv_bfloat16* x = X + (size_t)r * STREAMS * D;
  for (int d = tid; d < D; d += THREADS) {
    float v = pre[0] * __bfloat162float(x[d]);
    for (int s = 1; s < STREAMS; ++s) v = fmaf(pre[s], __bfloat162float(x[s * D + d]), v);
    const __nv_bfloat16 c = __float2bfloat16_rn(v);
    row[d] = c;
    sq[d] = __bfloat162float(c) * __bfloat162float(c);
  }
  __syncthreads();
  fold_shared(sq, D);
  if (tid == 0) inverse = 1.0f / sqrtf(sq[0] / D + eps_norm);
  __syncthreads();
  for (int d = tid; d < D; d += THREADS)
    out[(size_t)r * D + d] =
        __float2bfloat16_rn(weight_of(norm[d]) * bf(__bfloat162float(row[d]) * inverse));
}

// Thread (row, d): each stream takes bf16(post_s branch + sum over the streams k, in order, of comb[k][s] x_k).
// In place: a thread reads its four values before it writes one.
template <typename G_T>
__global__ void post_kernel(__nv_bfloat16* __restrict__ X, const G_T* __restrict__ G, const float* __restrict__ post,
                            const float* __restrict__ comb, long long RS, int R, int D, int WORLD) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const int r = static_cast<int>(i / D), d = static_cast<int>(i % D);
  const float branch = branch_of(G, RS, WORLD, i);
  __nv_bfloat16* x = X + (size_t)r * STREAMS * D + d;
  float v[STREAMS];
  for (int k = 0; k < STREAMS; ++k) v[k] = __bfloat162float(x[(size_t)k * D]);
  const float* c = comb + (size_t)r * STREAMS * STREAMS;
  for (int s = 0; s < STREAMS; ++s) {
    float mixed = c[s] * v[0];
    for (int k = 1; k < STREAMS; ++k) mixed = fmaf(c[k * STREAMS + s], v[k], mixed);
    x[(size_t)s * D] = __float2bfloat16_rn(fmaf(post[(size_t)r * STREAMS + s], branch, mixed));
  }
}

// Thread (row, d): bf16((x_0 + x_1 + x_2 + x_3) / 4), the streams added in order. `OS`: elements from one row of
// `out` to the next.
__global__ void mean_kernel(const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ out, int R, int D,
                            long long OS) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const int r = static_cast<int>(i / D), d = static_cast<int>(i % D);
  const __nv_bfloat16* x = X + (size_t)r * STREAMS * D + d;
  float acc = __bfloat162float(x[0]);
  for (int s = 1; s < STREAMS; ++s) acc = acc + __bfloat162float(x[(size_t)s * D]);
  out[(size_t)r * OS + d] = __float2bfloat16_rn(acc / STREAMS);
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

// The window kernel's blocks along the rows: with the 16 parts, one a SM (185 registers), and no more than the
// groups.
int64_t row_blocks(int64_t rows) {
  static int sms = 0;
  if (sms == 0) sms = zs::device().multiProcessorCount;
  const int64_t groups = (rows + GROUP - 1) / GROUP, most = (sms + PARTS - 1) / PARTS;
  return groups < most ? groups : most;
}

bool power_of_two(int64_t n) { return n >= 1 && (n & (n - 1)) == 0; }

// x bf16 [R, 4 * D], rows contiguous, D a power of two the kernels fold.
bool streams_of(const zs::View& x) {
  return x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(1) % STREAMS == 0 &&
         power_of_two(x.size(1) / STREAMS) && x.size(1) / STREAMS <= MAX_D && x.size(0) <= INT32_MAX;
}

// The ranks' partials fp32 [world, rows or more, D]: a rank's rows start `size(1) * D` elements after the last's.
bool gathered_of(const zs::View& g, int64_t rows, int64_t d, bool bf16 = false) {
  return g.defined() && (g.dtype() == ZS_F32 || (bf16 && g.dtype() == ZS_BF16)) && g.dim() == 3 && g.size(0) >= 1 && g.size(0) <= 8 && g.size(1) >= rows &&
         g.size(2) == d;
}

bool floats_of(const zs::View& v, int64_t count) {
  return v.defined() && v.dtype() == ZS_F32 && v.numel() >= count;
}

// `zs_glm5_hc_forms`: the read-out's two launches by the forms above. Off unless switched on.
bool lanes = false, octs = false;

template <class W>
void launch_pre(const zs::View& x, const zs::View& part, const zs::View& base, const zs::View& scale, const W* norm,
                const zs::View& out, const zs::View& post, const zs::View& comb, double eps, double hc_eps, int64_t d,
                int64_t iters) {
  (lanes ? pre_lanes_kernel<W> : pre_kernel<W>)<<<static_cast<unsigned>(x.size(0)), THREADS, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), part.data_ptr<float>(), base.data_ptr<float>(), scale.data_ptr<float>(), norm,
      out.data_ptr<__nv_bfloat16>(), post.data_ptr<float>(), comb.data_ptr<float>(), static_cast<float>(eps),
      static_cast<float>(hc_eps), static_cast<int>(d), static_cast<int>(iters));
}

}  // namespace

extern "C" {

// x bf16 [R, 4 * D], fn bf16 [24, 4 * D] -> part fp32 [R, 16, 32]: each of the 16 blocks' 24 dots and, at 24,
// its squares' sum. 4 * D is in blocks of 2048 (16 of 128 values), 16,384 at most.
int zs_glm5_hc_mix(const zs_view* x_, const zs_view* fn_, const zs_view* part_) {
  zs::begin();
  const zs::View x(x_), fn(fn_), part(part_);
  ZS_REQUIRE(streams_of(x), "x: bf16 [R, 4 * D], D a power of two to 4096");
  const int64_t rows = x.size(0), wide = x.size(1);
  ZS_REQUIRE(wide % (PARTS * SUB) == 0 && wide / (PARTS * SUB) <= MAX_STEPS, "4 * D in blocks of 2048, 8 at most");
  ZS_REQUIRE(fn.defined() && fn.dtype() == ZS_BF16 && fn.dim() == 2 && fn.is_contiguous() && fn.size(0) == MIX &&
             fn.size(1) == wide, "fn: bf16 [24, 4 * D]");
  ZS_REQUIRE(floats_of(part, rows * PARTS * ENTRY), "part: fp32 [R, 16, 32]");
  const bool aligned = (reinterpret_cast<uintptr_t>(x.data_ptr<__nv_bfloat16>()) & 15) == 0 &&
                       (reinterpret_cast<uintptr_t>(fn.data_ptr<__nv_bfloat16>()) & 15) == 0;
  if (octs && aligned && rows >= rows_from) {
    static bool opened = false;
    if (!opened) {
      ZS_CUDA_CHECK(cudaFuncSetAttribute(mix_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, RING_BYTES));
      opened = true;
    }
    mix_rows_kernel<<<dim3(static_cast<unsigned>(row_blocks(rows)), PARTS), THREADS, RING_BYTES, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), fn.data_ptr<__nv_bfloat16>(), part.data_ptr<float>(), static_cast<int>(rows),
        static_cast<int>(wide));
  } else
    (octs && aligned ? mix_octs_kernel : mix_kernel)<<<dim3(static_cast<unsigned>(rows), PARTS), THREADS, 0,
                                                         zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), fn.data_ptr<__nv_bfloat16>(), part.data_ptr<float>(), static_cast<int>(wide));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// The read-out by `pre_lanes_kernel` and `mix_octs_kernel` (nonzero) or by the base kernels: the same bits.
void zs_glm5_hc_forms(int lanes_on, int octs_on) {
  lanes = lanes_on != 0;
  octs = octs_on != 0;
}

// x bf16 [R, 4 * D] and its part -> out bf16 [R, D], the collapsed row normed by `norm` bf16 or fp32 [D]; post fp32
// [R, 4] and comb fp32 [R, 4, 4] for the write-back. base fp32 [24], scale fp32 [3].
int zs_glm5_hc_pre(const zs_view* x_, const zs_view* part_, const zs_view* base_, const zs_view* scale_,
                   const zs_view* norm_, const zs_view* out_, const zs_view* post_, const zs_view* comb_, double eps,
                   double hc_eps, int64_t iters) {
  zs::begin();
  const zs::View x(x_), part(part_), base(base_), scale(scale_), norm(norm_), out(out_), post(post_), comb(comb_);
  ZS_REQUIRE(streams_of(x), "x: bf16 [R, 4 * D], D a power of two to 4096");
  const int64_t rows = x.size(0), d = x.size(1) / STREAMS;
  ZS_REQUIRE(floats_of(part, rows * PARTS * ENTRY), "part: fp32 [R, 16, 32]");
  ZS_REQUIRE(floats_of(base, MIX) && floats_of(scale, 3), "base: fp32 [24], scale: fp32 [3]");
  ZS_REQUIRE(norm.defined() && (norm.dtype() == ZS_BF16 || norm.dtype() == ZS_F32) && norm.numel() == d,
             "norm: bf16 or fp32 [D]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * d,
             "out: bf16 [R, D]");
  ZS_REQUIRE(floats_of(post, rows * STREAMS) && floats_of(comb, rows * STREAMS * STREAMS),
             "post: fp32 [R, 4], comb: fp32 [R, 4, 4]");
  ZS_REQUIRE(iters >= 1 && iters <= 1024, "1 Sinkhorn round or more");
  if (norm.dtype() == ZS_F32)
    launch_pre(x, part, base, scale, norm.data_ptr<float>(), out, post, comb, eps, hc_eps, d, iters);
  else
    launch_pre(x, part, base, scale, norm.data_ptr<__nv_bfloat16>(), out, post, comb, eps, hc_eps, d, iters);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// x bf16 [R, 4 * D] in place takes the branch: gathered fp32 or bf16 [world, R or more, D], the ranks' partials, with
// the read-out's post and comb.
int zs_glm5_hc_post(const zs_view* x_, const zs_view* gathered_, const zs_view* post_, const zs_view* comb_) {
  zs::begin();
  const zs::View x(x_), g(gathered_), post(post_), comb(comb_);
  ZS_REQUIRE(streams_of(x), "x: bf16 [R, 4 * D], D a power of two to 4096");
  const int64_t rows = x.size(0), d = x.size(1) / STREAMS;
  ZS_REQUIRE(gathered_of(g, rows, d, true), "gathered: fp32 or bf16 [world, R or more, D]");
  ZS_REQUIRE(floats_of(post, rows * STREAMS) && floats_of(comb, rows * STREAMS * STREAMS),
             "post: fp32 [R, 4], comb: fp32 [R, 4, 4]");
  if (g.dtype() == ZS_BF16)
    post_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), g.data_ptr<__nv_bfloat16>(), post.data_ptr<float>(), comb.data_ptr<float>(),
        g.size(1) * d, static_cast<int>(rows), static_cast<int>(d), static_cast<int>(g.size(0)));
  else
    post_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), g.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
        g.size(1) * d, static_cast<int>(rows), static_cast<int>(d), static_cast<int>(g.size(0)));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// p fp32 [R, D] in place, each value rounded to bf16 (round to nearest even) and kept fp32: a quantized row's partial
// in a round whose partials stay fp32 for its bf16 rows (spec 116 WS-04), the bits its bf16 partial would give post.
__global__ void round_kernel(float* __restrict__ p, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __bfloat162float(__float2bfloat16_rn(p[i]));
}

int zs_glm5_round_bf16(const zs_view* p_) {
  zs::begin();
  const zs::View p(p_);
  ZS_REQUIRE(p.defined() && p.dtype() == ZS_F32 && p.dim() == 2 && p.is_contiguous(), "p: fp32 [R, D]");
  round_kernel<<<blocks(p.numel()), 256, 0, zs::stream()>>>(p.data_ptr<float>(), p.numel());
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// x bf16 [R, 4 * D] -> out bf16 [R, D], the streams' mean. `out`'s rows may be strided: D columns of a wider
// buffer (a tap among the drafter's, side by side).
int zs_glm5_stream_mean(const zs_view* x_, const zs_view* out_) {
  zs::begin();
  const zs::View x(x_), out(out_);
  ZS_REQUIRE(streams_of(x), "x: bf16 [R, 4 * D], D a power of two to 4096");
  const int64_t rows = x.size(0), d = x.size(1) / STREAMS;
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16, "out: bf16 [R, D]");
  const bool strided = !out.is_contiguous();
  ZS_REQUIRE(strided ? out.size(0) >= rows && out.size(1) == d && out.row_stride() > d : out.numel() >= rows * d,
             "out: bf16 [R, D], contiguous or D columns of wider rows");
  mean_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(x.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(),
                                                          static_cast<int>(rows), static_cast<int>(d),
                                                          strided ? out.row_stride() : d);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
