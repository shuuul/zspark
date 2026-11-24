// zspark's own (spec 114 WS-03), after experts.cu's grouped NVFP4 kernel (TensorFold 0.6.5, 609ca41, Apache-2.0),
// which stays as it is beside this file and is what Flash Next's experts and GLM's MTP layer run. The routed
// experts W4A4, as vLLM's image default runs FP4-Local's (b12x `mode="a4"`): the pairs' rows quantized to NVFP4
// by experts_quant.cu, the weights NVFP4 as stored, and the products on the block-scaled FP4 instruction
// (experts_a4.cuh): no weight is decoded to bf16, an mma takes 64 inputs and their scales as they are.
//
// The plan, the units (an item's 32 output columns a warp, M matrices), the pairs as mma rows 16 at a time, the
// units in whole rounds (EVEN) and down's epilogues are experts.cu's; the expert's factor here is alpha
// (weight_scale_2 / gs). Gate and up's epilogue (EPI 4, spec 114 WS-07) makes down's NVFP4 rows from their fp32
// SwiGLU as the card's vLLM does (experts_a4.cuh `swiglu_a4`): no bf16 SwiGLU is written and read back, no second
// quantize launch. This is the W4A4 form of a prompt's window up to 255 rows; experts_tile_a4.cu takes
// four tiles of an expert's pairs under one reading of its weights, with the same bits: a pair's sums are an mma
// row over its own quantized row and the weights, k64 steps in order from zero, then times alpha, then the
// epilogue, in both.
//
// Weights, per expert and column block of 32 and matrix, a (32 columns, 64 inputs) block of 72 uint4 a k64 step
// (experts_a4.cuh, `experts.packA4`): [E][N/32][K/64][M][72].
//
// Shared memory: none. A lane reads its own words from global memory into registers, two k64 steps ahead
// (`ld.global.nc`):
//   held a lane, a k64 step                bytes   what it is
//   B registers, M matrices x 4 n8 tiles  32 x M    weight on its way
//   B scales, M matrices                  16 x M    its quad's column words (all four lanes read them)
//   A registers, rows q and q + 8            16    the two pairs' codes, a0 to a3
//   SFA word                                  4    row q's (lane c 0) or q + 8's (c 1) four scales
// Nothing is staged, so nothing could be less; the activation's 20 bytes a lane are an eighth of what the bf16
// rows were (experts.cu: 64 bytes), and come from L2 (the plan's rows, written by experts_quant.cu or by gate and
// up's epilogue just before). Gate and up's epilogue adds no shared memory: a group's largest magnitude and its
// code bytes meet on the lane quad by shuffles, and a lane holds four 8-byte code words and four scale bytes.
//
// `nvfp4_expert_a16_kernel` (spec 115) is the A16 form on the same pack, a step's, a verify window's and a short
// prompt window's (the card runs those rows' experts A16: `VLLM_B12X_MOE_FP4_FORCE_A16`, and a captured batch's MoE
// A16 whatever its kind): bf16 rows times the weights decoded to bf16 in registers (exact: an e2m1 value times its
// e4m3 scale, experts.cu's `fp4pair`, `scale_pair`, `mul_pair`, copied here) on bf16 m16n8k16, four a k64 step.
// The A4 pack's B register j of lane (q, c) holds column q's inputs 32 j + 8 c .. + 7; sub-step (j, h) gives the mma
// that lane's inputs {2h, 2h + 4} as its k pair (2c, 2c + 1) and {2h + 1, 2h + 5} as (2c + 8, 2c + 9), A's the same
// inputs of its rows (one uint4 a row's eight, `__byte_perm`): a permutation of K inside each lane's eight, both
// operands alike. Their scale is the column's group 2 j + c / 2. Times weight_scale_2, then gate and up's SwiGLU in
// fp32 rounded to bf16 once (`swiglu_b12x`) into down's bf16 rows, or down's fp32 slots. Its registers a lane, a k64
// step: B and scales as above, A 64 bytes (two rows' 16 inputs); no shared memory.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/experts.cuh"
#include "../zs/host.h"
#include "codes.cuh"
#include "experts_a4.cuh"

namespace {

using namespace nvfp4_codes;

__device__ __forceinline__ uint4 ld_nc4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

template <int M>
struct StageA4 {
  uint4 w[M][2];  // tiles 0 and 1, then 2 and 3: b0, b1 a tile
  uint4 s[M];     // the scale words of columns q, 8 + q, 16 + q, 24 + q
  uint32_t a[4];  // rows q, q + 8: inputs 8c .. 8c + 7, then 32 + 8c ..
  uint32_t sfa;
};

template <int M>
__device__ __forceinline__ void load_a4(StageA4<M>& st, const uint4* blk, int g, int lane, int q, int c,
                                        const uint8_t* c0, const uint8_t* c1, const uint8_t* sf, bool v0, bool v1) {
  const uint4* b = blk + (size_t)g * (M * A4_BLOCK4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m][0] = ld_nc4(b + m * A4_BLOCK4 + lane);
    st.w[m][1] = ld_nc4(b + m * A4_BLOCK4 + 32 + lane);
    st.s[m] = ld_nc4(b + m * A4_BLOCK4 + 64 + q);
  }
  const int at = g * 32 + 4 * c;
  st.a[0] = v0 ? __ldg(reinterpret_cast<const uint32_t*>(c0 + at)) : 0u;
  st.a[2] = v0 ? __ldg(reinterpret_cast<const uint32_t*>(c0 + at + 16)) : 0u;
  st.a[1] = v1 ? __ldg(reinterpret_cast<const uint32_t*>(c1 + at)) : 0u;
  st.a[3] = v1 ? __ldg(reinterpret_cast<const uint32_t*>(c1 + at + 16)) : 0u;
  st.sfa = sf ? __ldg(reinterpret_cast<const uint32_t*>(sf + g * 4)) : 0u;
}

template <int M>
__device__ __forceinline__ void compute_a4(float (&acc)[M][1][NTW][4], const StageA4<M>& st) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const uint4& w = st.w[m][j >> 1];
      mma_a4(acc[m][0][j], st.a[0], st.a[1], st.a[2], st.a[3], a4_comp(w, 2 * (j & 1)), a4_comp(w, 2 * (j & 1) + 1),
             st.sfa, a4_comp(st.s[m], j));
    }
}

// Pair rows come from the plan's quantized rows (codes [P, K/2], scales [P, K/16], row i the plan's member i);
// items of expert `skip` are left alone; an item's pairs go 16 at a time.
template <int M, int EPI, int WARPS, bool EVEN>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_a4_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                           const uint4* __restrict__ W, const float* __restrict__ alpha, int KG, int NB,
                           const int* __restrict__ items, const int* __restrict__ counts,
                           const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip,
                           DownA4 dq) {
  const int lane = threadIdx.x & 31, q = lane >> 2, c = lane & 3;
  const int units = __ldg(counts) * NB;
  int unit = blockIdx.x * WARPS + (threadIdx.x >> 5), stride = gridDim.x * WARPS;
  if constexpr (EVEN) {
    const int rounds = (units + stride - 1) / stride;
    if (rounds > 1) {
      const long long active = (units + rounds - 1) / rounds;
      const int mine = static_cast<int>(unit * active / stride);
      if (static_cast<int>((unit + 1) * active / stride) == mine) return;
      unit = mine;
      stride = static_cast<int>(active);
    }
  }
  const size_t crow = (size_t)KG * 32, srow = (size_t)KG * 4;
  for (; unit < units; unit += stride) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * A4_BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + q < cnt, v1 = r0 + q + 8 < cnt;
      const uint8_t* c0 = codes + (size_t)(first + r0 + q) * crow;
      const uint8_t* c1 = codes + (size_t)(first + r0 + q + 8) * crow;
      const bool vs = c == 0 ? v0 : c == 1 ? v1 : false;
      const uint8_t* sf = vs ? scales + (size_t)(first + r0 + q + 8 * c) * srow : nullptr;
      float acc[M][1][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
      constexpr int D = 2;
      StageA4<M> st[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
        if (d < KG) load_a4<M>(st[d], blk, d, lane, q, c, c0, c1, sf, v0, v1);
      for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
          const int g = g0 + d;
          if (g < KG) {
            compute_a4<M>(acc, st[d]);
            if (g + D < KG) load_a4<M>(st[d], blk, g + D, lane, q, c, c0, c1, sf, v0, v1);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float a = __ldg(alpha + e * M + m);
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int k = 0; k < 4; ++k) acc[m][0][j][k] *= a;
      }
      const int pr0 = v0 ? __ldg(members + first + r0 + q) : 0;
      const int pr1 = v1 ? __ldg(members + first + r0 + q + 8) : 0;
      static_assert((M == 2) == (EPI == 4), "gate and up: their SwiGLU into down's rows");
      if constexpr (EPI == 4 || EPI == 7)
        activation_a4<EPI == 7>(acc[0][0], acc[M - 1][0], cb, c, dq, __ldg(dq.gs + e), N, (size_t)(first + r0 + q),
                               (size_t)(first + r0 + q + 8), v0, v1, limit, reinterpret_cast<__nv_bfloat16*>(out), pr0, pr1);
      else
        epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * c, pr0, pr1, v0, v1, limit);
    }
  }
}

template <int M>
struct StageA16 {
  uint4 w[M][2];   // as StageA4's
  uint4 s[M];
  uint4 xa[2];     // row q: inputs 8c .. 8c + 7, then 32 + 8c ..
  uint4 xb[2];     // row q + 8
};

template <int M>
__device__ __forceinline__ void load_a16(StageA16<M>& st, const uint4* blk, int g, int lane, int q, int c,
                                         const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  const uint4* b = blk + (size_t)g * (M * A4_BLOCK4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m][0] = ld_nc4(b + m * A4_BLOCK4 + lane);
    st.w[m][1] = ld_nc4(b + m * A4_BLOCK4 + 32 + lane);
    st.s[m] = ld_nc4(b + m * A4_BLOCK4 + 64 + q);
  }
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  const int k0 = g * 64 + 8 * c;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    st.xa[j] = v0 ? __ldg(reinterpret_cast<const uint4*>(x0 + k0 + 32 * j)) : zero;
    st.xb[j] = v1 ? __ldg(reinterpret_cast<const uint4*>(x1 + k0 + 32 * j)) : zero;
  }
}

// sel[j]: the scale word's byte 2 j + c / 2 twice, for scale_pair
template <int M>
__device__ __forceinline__ void compute_a16(float (&acc)[M][1][NTW][4], const StageA16<M>& st,
                                            const uint32_t (&sel)[2]) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t pa = a4_comp(st.xa[j], h), qa = a4_comp(st.xa[j], h + 2);
        const uint32_t pb = a4_comp(st.xb[j], h), qb = a4_comp(st.xb[j], h + 2);
        const uint32_t a0 = __byte_perm(pa, qa, 0x5410u), a2 = __byte_perm(pa, qa, 0x7632u);
        const uint32_t a1 = __byte_perm(pb, qb, 0x5410u), a3 = __byte_perm(pb, qb, 0x7632u);
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
          const uint32_t word = a4_comp(st.w[m][t >> 1], 2 * (t & 1) + j);
          const uint32_t ss = scale_pair(a4_comp(st.s[m], t), sel[j]);
          mma(acc[m][0][t], a0, a1, a2, a3, mul_pair(fp4pair(word, 8 * h), ss), mul_pair(fp4pair(word, 8 * h + 4), ss));
        }
      }
}

// bf16 rows on the A4 pack: pair p reads X row p / slots when slots > 0 (X holds tokens), else X row p (down's
// bf16 rows by pair). EPI 5: M = 2, the SwiGLU into bf16 out [pairs, N] at the pair's row, rounded once; 0: fp32
// out; 3: bf16 out. Units, rounds and items as nvfp4_expert_a4_kernel's.
template <int M, int EPI, int WARPS, bool EVEN>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_a16_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                            const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                            const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out,
                            int N, float limit, int skip) {
  const int lane = threadIdx.x & 31, q = lane >> 2, c = lane & 3;
  const int units = __ldg(counts) * NB;
  int unit = blockIdx.x * WARPS + (threadIdx.x >> 5), stride = gridDim.x * WARPS;
  if constexpr (EVEN) {
    const int rounds = (units + stride - 1) / stride;
    if (rounds > 1) {
      const long long active = (units + rounds - 1) / rounds;
      const int mine = static_cast<int>(unit * active / stride);
      if (static_cast<int>((unit + 1) * active / stride) == mine) return;
      unit = mine;
      stride = static_cast<int>(active);
    }
  }
  uint32_t sel[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const uint32_t b = 2 * j + (c >> 1);
    sel[j] = b | 0x40u | (b << 8) | 0x4000u;
  }
  for (; unit < units; unit += stride) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * A4_BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + q < cnt, v1 = r0 + q + 8 < cnt;
      const int pr0 = v0 ? __ldg(members + first + r0 + q) : 0;
      const int pr1 = v1 ? __ldg(members + first + r0 + q + 8) : 0;
      const __nv_bfloat16* x0 = X + (size_t)(slots ? pr0 / slots : pr0) * x_stride;
      const __nv_bfloat16* x1 = X + (size_t)(slots ? pr1 / slots : pr1) * x_stride;
      float acc[M][1][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int t = 0; t < NTW; ++t) acc[m][0][t][0] = acc[m][0][t][1] = acc[m][0][t][2] = acc[m][0][t][3] = 0.f;
      constexpr int D = 2;
      StageA16<M> st[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
        if (d < KG) load_a16<M>(st[d], blk, d, lane, q, c, x0, x1, v0, v1);
      for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
          const int g = g0 + d;
          if (g < KG) {
            compute_a16<M>(acc, st[d], sel);
            if (g + D < KG) load_a16<M>(st[d], blk, g + D, lane, q, c, x0, x1, v0, v1);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float a = __ldg(scale + e * M + m);
#pragma unroll
        for (int t = 0; t < NTW; ++t)
#pragma unroll
          for (int k = 0; k < 4; ++k) acc[m][0][t][k] *= a;
      }
      static_assert((M == 2) == (EPI == 5), "gate and up: their SwiGLU");
      if constexpr (EPI == 5) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if (!(h ? v1 : v0)) continue;
          __nv_bfloat16* row = reinterpret_cast<__nv_bfloat16*>(out) + (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
          for (int t = 0; t < NTW; ++t) {
            const float o0 = swiglu_b12x(acc[0][0][t][2 * h], acc[1][0][t][2 * h], limit);
            const float o1 = swiglu_b12x(acc[0][0][t][2 * h + 1], acc[1][0][t][2 * h + 1], limit);
            *reinterpret_cast<__nv_bfloat162*>(row + cb * COLS + 8 * t + 2 * c) = __floats2bfloat162_rn(o0, o1);
          }
        }
      } else {
        epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * c, pr0, pr1, v0, v1, limit);
      }
    }
  }
}

template <int M, int EPI>
void launch_a16(const zs::View& x, int x_stride, int slots, const zs::View& w, const zs::View& scale, int kg, int nb,
                const zs::View& items, const zs::View& counts, const zs::View& members, void* out, int n, float limit,
                int skip, int64_t max_units, bool even) {
  constexpr int WARPS = 4;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_a16_kernel<M, EPI, WARPS, false>, WARPS * 32,
                                                  0);
    sms = zs::device().multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = (max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const uint4* wp = reinterpret_cast<const uint4*>(w.data_ptr());
  if (even && need > grid)
    nvfp4_expert_a16_kernel<M, EPI, WARPS, true><<<grid, WARPS * 32, 0, zs::stream()>>>(
        xp, x_stride, slots, wp, scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(), counts.data_ptr<int>(),
        members.data_ptr<int>(), out, n, limit, skip);
  else
    nvfp4_expert_a16_kernel<M, EPI, WARPS, false><<<grid, WARPS * 32, 0, zs::stream()>>>(
        xp, x_stride, slots, wp, scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(), counts.data_ptr<int>(),
        members.data_ptr<int>(), out, n, limit, skip);
  ZS_CUDA_LAUNCH_CHECK();
}

template <int M, int EPI>
void launch(const zs::View& codes, const zs::View& scales, const zs::View& w, const zs::View& alpha, int kg, int nb,
            const zs::View& items, const zs::View& counts, const zs::View& members, void* out, int n, float limit,
            int skip, int64_t max_units, bool even, const DownA4& dq) {
  constexpr int WARPS = 4;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_a4_kernel<M, EPI, WARPS, false>, WARPS * 32, 0);
    sms = zs::device().multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  // a bound, not a count: the items are known on the device alone, and a graph holds this grid
  const int64_t need = (max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  const uint8_t* cp = reinterpret_cast<const uint8_t*>(codes.data_ptr());
  const uint8_t* sp = reinterpret_cast<const uint8_t*>(scales.data_ptr());
  const uint4* wp = reinterpret_cast<const uint4*>(w.data_ptr());
  if (even && need > grid)
    nvfp4_expert_a4_kernel<M, EPI, WARPS, true><<<grid, WARPS * 32, 0, zs::stream()>>>(
        cp, sp, wp, alpha.data_ptr<float>(), kg, nb, items.data_ptr<int>(), counts.data_ptr<int>(),
        members.data_ptr<int>(), out, n, limit, skip, dq);
  else
    nvfp4_expert_a4_kernel<M, EPI, WARPS, false><<<grid, WARPS * 32, 0, zs::stream()>>>(
        cp, sp, wp, alpha.data_ptr<float>(), kg, nb, items.data_ptr<int>(), counts.data_ptr<int>(),
        members.data_ptr<int>(), out, n, limit, skip, dq);
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

void nvfp4_experts_tile_a4_cuda(int64_t tiles, int64_t epi, const zs::View& codes, const zs::View& scales, const zs::View& w,
                                const zs::View& alpha, int64_t kg, int64_t nb, const zs::View& items,
                                const zs::View& counts, const zs::View& members, void* out, int64_t n,
                                double limit, int64_t skip, int64_t max_units, const DownA4& dq);

extern "C" {

// Grouped W4A4 experts on a plan: codes u8 [P, K/2] and scales u8 [P, K/16] the plan's rows (experts_quant.cu),
// w [E, N/32, K/64, M, 72] uint4 as int32 [.., 288], alpha fp32 [E, M]. epi 4: M = 2, the SwiGLU of gate and up as
// b12x's, quantized into down's rows (down_codes u8 [P, N/2], down_scales u8 [P, N/16], row i the plan's member i,
// down_gs fp32 [E]); out null, or bf16 [pairs, N] for a check: the values quantized, at each pair's row. 0: fp32 out;
// 3: bf16 out; 6: fp32 out rounded to bf16, the card's FC2 store (spec 143 WS-08); the down_* views null. kg = K/64, nb = N/32, n the out rows' width. tiles 1: this file's kernel, an
// item's pairs 16 at a time (even: its units in whole rounds); 4: experts_tile_a4.cu's, items of up to 64 pairs
// under one reading of the weights, 8 up to 128. None changes a bit. max_units bounds the grid.
int zs_nvfp4_experts_a4(int64_t epi, int64_t tiles, int64_t even, const zs_view* codes_, const zs_view* scales_,
                        const zs_view* w_, const zs_view* alpha_, int64_t kg, int64_t nb, const zs_view* items_,
                        const zs_view* counts_, const zs_view* members_, const zs_view* out_, int64_t n,
                        double limit, int64_t skip, int64_t max_units, const zs_view* down_codes_,
                        const zs_view* down_scales_, const zs_view* down_gs_) {
  zs::begin();
  const zs::View codes(codes_), scales(scales_), w(w_), alpha(alpha_), items(items_), counts(counts_),
      members(members_), down_codes(down_codes_), down_scales(down_scales_), down_gs(down_gs_);
  zs::View out(out_);
  ZS_REQUIRE(epi == 0 || epi == 3 || epi == 4 || epi == 6 || epi == 7,
             "epilogue 0 (fp32 down), 3 (bf16 down), 4 (SwiGLU), 6 (fp32 down rounded to bf16), 7 (relu2)");
  ZS_REQUIRE(tiles == 1 || tiles == 4 || tiles == 8, "tiles: 1, 4 or 8 tiles of pairs a reading");
  const int64_t m = epi == 4 ? 2 : 1;
  ZS_REQUIRE(kg >= 1 && nb >= 1 && n == nb * 32, "K in steps of 64, N in blocks of 32 columns");
  ZS_REQUIRE(codes.defined() && codes.dtype() == ZS_U8 && scales.defined() && scales.dtype() == ZS_U8 &&
                 reinterpret_cast<uintptr_t>(codes.data_ptr()) % 16 == 0 &&
                 codes.numel() >= members.numel() * kg * 32 && scales.numel() >= members.numel() * kg * 4,
             "codes u8 [P, K/2] on 16 bytes, scales u8 [P, K/16]");
  ZS_REQUIRE(w.defined() && w.dtype() == ZS_I32 && w.size(-1) == 288 && w.size(-2) == m &&
                 w.numel() == w.size(0) * nb * kg * m * 288 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
             "w: [E, N/32, K/64, M, 288]");
  ZS_REQUIRE(alpha.defined() && alpha.dtype() == ZS_F32 && alpha.numel() == w.size(0) * m, "alpha: [E, M] fp32");
  ZS_REQUIRE(items.defined() && counts.defined() && members.defined(), "plan: items, counts and members");
  DownA4 dq{nullptr, nullptr, nullptr};
  if (epi == 4 || epi == 7) {
    ZS_REQUIRE(!out.defined() || (out.dtype() == ZS_BF16 && out.numel() >= members.numel() * n),
               "out: null, or bf16 [pairs, N]");
    ZS_REQUIRE(down_codes.defined() && down_codes.dtype() == ZS_U8 && down_scales.defined() &&
                   down_scales.dtype() == ZS_U8 && reinterpret_cast<uintptr_t>(down_codes.data_ptr()) % 16 == 0 &&
                   down_codes.numel() >= members.numel() * ((n + 63) / 64 * 32) &&
                   down_scales.numel() >= members.numel() * ((n + 63) / 64 * 4),
               "down's rows: codes u8 [P, N/2] on 16 bytes, scales u8 [P, N/16]");
    ZS_REQUIRE(down_gs.defined() && down_gs.dtype() == ZS_F32 && down_gs.numel() == w.size(0), "down_gs: fp32 [E]");
    dq = {reinterpret_cast<uint8_t*>(down_codes.data_ptr()), reinterpret_cast<uint8_t*>(down_scales.data_ptr()),
          down_gs.data_ptr<float>()};
  } else {
    ZS_REQUIRE(out.defined() && out.dtype() == (epi == 3 ? ZS_BF16 : ZS_F32), "out dtype");
  }
  void* o = out.defined() ? out.data_ptr() : nullptr;
  if (tiles == 4 || tiles == 8) {
    ZS_REQUIRE(counts.numel() >= 4, "counts: items, experts and the tile kernel's two counters, zero between launches");
    nvfp4_experts_tile_a4_cuda(tiles, epi, codes, scales, w, alpha, kg, nb, items, counts, members, o, n, limit, skip,
                               max_units, dq);
    return zs::status();
  }
  const int k = static_cast<int>(kg), b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  if (epi == 4) launch<2, 4>(codes, scales, w, alpha, k, b, items, counts, members, o, nn, lim, sk, max_units, even, dq);
  else if (epi == 7) launch<1, 7>(codes, scales, w, alpha, k, b, items, counts, members, o, nn, lim, sk, max_units, even, dq);
  else if (epi == 0) launch<1, 0>(codes, scales, w, alpha, k, b, items, counts, members, o, nn, lim, sk, max_units, even, dq);
  else if (epi == 6) launch<1, 6>(codes, scales, w, alpha, k, b, items, counts, members, o, nn, lim, sk, max_units, even, dq);
  else launch<1, 3>(codes, scales, w, alpha, k, b, items, counts, members, o, nn, lim, sk, max_units, even, dq);
  return zs::status();
}

// Grouped experts on a plan, bf16 rows on the A4 pack (spec 115): x bf16 [rows, K] at x_stride elements a row
// (slots > 0: pair p reads row p / slots; 0: row p), w as zs_nvfp4_experts_a4's, scale fp32 [E, M] weight_scale_2.
// epi 5: M = 2, the SwiGLU rounded to bf16 once into out bf16 [pairs, N]; 0: fp32 out, 3: bf16 out, M = 1. Items
// of up to 16 pairs (a narrow plan), each read again for every 16; even: its units in whole rounds. The same bits
// whatever else the call holds.
int zs_nvfp4_experts_a16(int64_t epi, int64_t even, const zs_view* x_, int64_t x_stride, int64_t slots,
                         const zs_view* w_, const zs_view* scale_, int64_t kg, int64_t nb, const zs_view* items_,
                         const zs_view* counts_, const zs_view* members_, const zs_view* out_, int64_t n, double limit,
                         int64_t skip, int64_t max_units) {
  zs::begin();
  const zs::View x(x_), w(w_), scale(scale_), items(items_), counts(counts_), members(members_), out(out_);
  ZS_REQUIRE(epi == 0 || epi == 3 || epi == 5, "epilogue 0 (fp32 down), 3 (bf16 down) or 5 (SwiGLU)");
  const int64_t m = epi == 5 ? 2 : 1;
  ZS_REQUIRE(kg >= 1 && nb >= 1 && n == nb * 32, "K in steps of 64, N in blocks of 32 columns");
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                 x_stride % 8 == 0 && x_stride >= kg * 64,
             "x: bf16 rows on 16 bytes, a stride of whole 8 inputs");
  ZS_REQUIRE(w.defined() && w.dtype() == ZS_I32 && w.size(-1) == 288 && w.size(-2) == m &&
                 w.numel() == w.size(0) * nb * kg * m * 288 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
             "w: [E, N/32, K/64, M, 288]");
  ZS_REQUIRE(scale.defined() && scale.dtype() == ZS_F32 && scale.numel() == w.size(0) * m, "scale: [E, M] fp32");
  ZS_REQUIRE(items.defined() && counts.defined() && members.defined(), "plan: items, counts and members");
  ZS_REQUIRE(out.defined() && out.dtype() == (epi == 0 ? ZS_F32 : ZS_BF16), "out: [pairs, N], fp32 (epi 0) or bf16");
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  void* o = out.data_ptr();
  if (epi == 5) launch_a16<2, 5>(x, xs, sl, w, scale, k, b, items, counts, members, o, nn, lim, sk, max_units, even);
  else if (epi == 0) launch_a16<1, 0>(x, xs, sl, w, scale, k, b, items, counts, members, o, nn, lim, sk, max_units, even);
  else launch_a16<1, 3>(x, xs, sl, w, scale, k, b, items, counts, members, o, nn, lim, sk, max_units, even);
  return zs::status();
}

}  // extern "C"
