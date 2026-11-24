// From TensorFold 1.2.1 (0afe9dd5), Apache-2.0: `nvfp4_expert_half`, the decode form's SwiGLU gate and up with two
// warps a unit (16 columns each) for a call whose every half unit fits the resident grid (one row's eleven items
// of Flash Next's 20 column blocks on 768 warps). Changed for zspark: ATen and c10 replaced by zs/host.h, and the
// launcher at the end is zspark's (nvfp4_experts_cuda's arguments, as experts_wide.cu's). `ld_nc`, `scale_sel`,
// `comp`, `mma`, `swiglu`, `relu2` and the epilogue's arithmetic are experts.cu's, `fp4pair`, `scale_pair` and
// `mul_pair` codes.cuh's. The mma and K order are the classic unit's and the epilogue the same, so a pair's bits
// are experts.cu's whatever kernel or call. GLM's epilogues (relu squared, fp32 and bf16 down) are not
// instantiated: the dispatch takes the half kernel for the gate and up alone.
// Not CUTLASS, for experts.cu's reasons.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../common/experts.cuh"
#include "../zs/host.h"
#include "codes.cuh"

namespace {

using namespace nvfp4_codes;

constexpr int BLOCK4 = 36;           // uint4 a (32 columns, 32 inputs) block: a lane's code words, then the scales

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_nc2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

// Half a unit's stage: the lane's words of n8 tiles 2jh and 2jh + 1, all its scales, its inputs.
template <int M>
struct Half {
  uint2 w[M];
  uint4 s[M];
  uint2 xa[2], xb[2];
};

// byte (j & 1) * 2 + (gq & 1) of a scale word, in the low and high halves: column gq's scale for tiles j even, odd
__device__ __forceinline__ void scale_sel(uint32_t (&sel)[2], int gq) {
  const uint32_t lo = gq & 1, hi = 2 + (gq & 1);
  sel[0] = lo | 0x40u | (lo << 8) | 0x4000u;
  sel[1] = hi | 0x40u | (hi << 8) | 0x4000u;
}

template <int M>
__device__ __forceinline__ void load_half(Half<M>& st, const uint4* blk, int g, int lane, int t, int jh,
                                          const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m] = ld_nc2(reinterpret_cast<const uint2*>(b + m * BLOCK4 + lane) + jh);
    st.s[m] = ld_nc(b + m * BLOCK4 + 32 + (lane >> 3));
  }
  const uint2 zero = make_uint2(0u, 0u);
  const int k0 = g * 32 + 4 * t;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    st.xa[h] = v0 ? __ldg(reinterpret_cast<const uint2*>(x0 + k0 + 16 * h)) : zero;
    st.xb[h] = v1 ? __ldg(reinterpret_cast<const uint2*>(x1 + k0 + 16 * h)) : zero;
  }
}

// compute_stage's sums for tiles j = 2jh + jj, in zspark's arithmetic as the classic unit's: the e4m3 scale of a
// weight column folded into its bf16 pair before the mma (`scale_pair` then `mul_pair`, the lane's own quad's
// scale word selected by `sel[j & 1]`), the mma adding into the sums — the same multiply-adds in the same K and
// block order as a whole unit's warp, so a pair's bits are the classic kernel's. TensorFold's version instead
// mma'ed unscaled into a fresh p and added fma(P, scale) a block, the arithmetic experts.cu departed from
// (spec 107 WS-03: the same value within fp32 rounding, not the same bits), so it is not kept here.
template <int M>
__device__ __forceinline__ void compute_half(float (&acc)[M][2][4], const Half<M>& st, int jh,
                                             const uint32_t (&sel)[2]) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
      const uint32_t sw = h == 0 ? (jh ? st.s[m].y : st.s[m].x) : (jh ? st.s[m].w : st.s[m].z);
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const uint32_t word = jj ? st.w[m].y : st.w[m].x;
        const uint32_t ss = scale_pair(sw, sel[jj]);
        mma(acc[m][jj], a0, a1, a2, a3, mul_pair(fp4pair(word, 8 * h), ss), mul_pair(fp4pair(word, 8 * h + 4), ss));
      }
    }
}

// experts.cuh's epilogue on a half unit's two n8 tiles.
template <int EPI, int M>
__device__ __forceinline__ void epilogue_half(const float (&acc)[M][2][4], void* out, int N, int col0, int pr0,
                                              int pr1, bool v0, bool v1, float limit) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int col = col0 + 8 * j;
      const float a0 = acc[0][j][2 * h], a1 = acc[0][j][2 * h + 1];
      if constexpr (EPI == 0) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
      } else {
        float o0, o1;
        if constexpr (EPI == 1) {
          o0 = relu2(a0);
          o1 = relu2(a1);
        } else if constexpr (EPI == 2) {
          o0 = swiglu(a0, acc[M - 1][j][2 * h], limit);
          o1 = swiglu(a1, acc[M - 1][j][2 * h + 1], limit);
        } else {
          o0 = a0;
          o1 = a1;
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

// The decode form with two warps a unit (16 columns each): the same sums, twice the warps for few items.
template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_half(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                      const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out,
                      int N, float limit, int skip) {
  constexpr int D = 2;
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB * 2;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / (NB * 2), rest = unit - it * NB * 2, cb = rest >> 1, jh = rest & 1;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
      const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
      const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
      const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
      const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
      const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
      float acc[M][2][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
      uint32_t sel[2];
      scale_sel(sel, gq);
      Half<M> st[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
        if (d < KG) load_half<M>(st[d], blk, d, lane, t, jh, x0, x1, v0, v1);
      for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
          const int g = g0 + d;
          if (g < KG) {
            compute_half<M>(acc, st[d], jh, sel);
            if (g + D < KG) load_half<M>(st[d], blk, g + D, lane, t, jh, x0, x1, v0, v1);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float g = __ldg(scale + e * M + m);
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][j][q] *= g;
      }
      epilogue_half<EPI, M>(acc, out, N, cb * COLS + 16 * jh + 2 * t, pr0, pr1, v0, v1, limit);
    }
  }
}

// The call the dispatch hands over: SwiGLU gate and up (epilogue 2), launched where every half unit is promised
// a warp. The grid covers the plan's half units, `max_units` a bound for a captured launch whose count is known
// on the device alone.
void launch_half(const zs::View& x, int x_stride, int slots, const zs::View& w, const zs::View& scale, int kg,
                 int nb, const zs::View& items, const zs::View& counts, const zs::View& members, zs::View& out,
                 int n, float limit, int skip, int64_t max_units) {
  constexpr int WARPS = 4;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_half<2, 2, WARPS>, WARPS * 32, 0);
    sms = zs::device().multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = (2 * max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  nvfp4_expert_half<2, 2, WARPS><<<grid, WARPS * 32, 0, zs::stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

// The half kernel's resident warps (its occupancy times the SMs): the dispatch's "every half unit fits" bound.
int64_t nvfp4_experts_half_resident() {
  constexpr int WARPS = 4;
  static int warps = 0;
  if (warps == 0) {
    int per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_half<2, 2, WARPS>, WARPS * 32, 0);
    warps = (per_sm < 1 ? 1 : per_sm) * zs::device().multiProcessorCount * WARPS;
  }
  return warps;
}

// nvfp4_experts_cuda's arguments (experts.cu), for the two-warps-a-unit gate and up.
void nvfp4_experts_half_cuda(int64_t epi, const zs::View& x, int64_t x_stride, int64_t slots, const zs::View& w,
                             const zs::View& scale, int64_t kg, int64_t nb, const zs::View& items,
                             const zs::View& counts, const zs::View& members, zs::View& out, int64_t n,
                             double limit, int64_t skip, int64_t max_units) {
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  if (epi == 2) launch_half(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else zs::fail("nvfp4 experts half: epilogue 2 (SwiGLU gate and up) only");
}
