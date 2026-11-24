// From TensorFold 0.6.5 (609ca41), Apache-2.0; changed for zspark: ATen and c10 replaced by zs/host.h, `fp4pair`
// (but its indentation), `scale_pair` and `mul_pair` codes.cuh's, the same text (spec 121), and the
// launcher at the end hands a call of more than one tile of rows a pass to experts_wide.cu (spec 021, zspark's
// own: this kernel reads an expert's weights again for every 16 of its pairs, 1.7 to 2.0 times an expert in a
// 512-row window) and instantiates epilogue 1 (relu squared, one up matrix; spec 151, Nemotron's experts). The device code here is TensorFold's but for one change (spec 107 WS-03), and is what a step and
// a window of up to 16 rows run. Grouped NVFP4 experts on the shared plan: each weight is its e2m1 value times its
// 16-input block's e4m3 scale in bf16, exact, and the mma adds the block's products into acc, blocks in order; one
// pair an mma row, so no pair affects another; each (expert, matrix) scale multiplies once before the epilogue.
// TensorFold takes P = mma from zero, then acc = fma(P, scale, acc) for each 16-input block; zspark folds the
// scale into the weights as vLLM's b12x W4A16 prefill does (`bfloat2_mul` before its mma.sync), which drops four
// fmaf an mma and lets the window kernels' mma add into their sums: experts_tile.cu at 2048 rows, plan, gate and
// up and down, 14.42 ms to 13.80 on GLM's shapes and 7.66 to 7.29 on Flash Next's, a step's the same (147 to 149
// us at one row). Its sums differ from TensorFold's by fp32 rounding: down against the host's f64 (experts_test),
// at most 1.45e-7 of the sum of the terms' magnitudes and 8.4e-9 on average, against TensorFold's 1.33e-7 and 7.0e-9.
// Not CUTLASS: its SM120 NVFP4 and grouped kernels take FP4 on both sides in M tiles of 128 contiguous rows; these
// rows are bf16, about a dozen an expert, gathered through the plan.

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

// byte (j & 1) * 2 + (gq & 1) of a scale word, in the low and high halves: column gq's scale for tiles j even, odd
__device__ __forceinline__ void scale_sel(uint32_t (&sel)[2], int gq) {
  const uint32_t lo = gq & 1, hi = 2 + (gq & 1);
  sel[0] = lo | 0x40u | (lo << 8) | 0x4000u;
  sel[1] = hi | 0x40u | (hi << 8) | 0x4000u;
}

template <int M>
struct Stage {
  uint4 w[M];          // the lane's words: n8 tiles 0-3, 32 inputs each
  uint4 s[M];          // the scales of the quad that holds column gq, bytes [block][tile][column]
  uint2 xa[2], xb[2];  // rows gq and gq + 8: inputs 4t .. 4t + 3 of each 16-input block
};

template <int M>
__device__ __forceinline__ void load_stage(Stage<M>& st, const uint4* blk, int g, int lane, int t,
                                           const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m] = ld_nc(b + m * BLOCK4 + lane);
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

template <int M>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<M>& st,
                                              const uint32_t (&sel)[2]) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = comp(st.w[m], j);
        const uint32_t ss = scale_pair(comp(st.s[m], 2 * h + (j >> 1)), sel[j & 1]);
        mma(acc[m][0][j], a0, a1, a2, a3, mul_pair(fp4pair(word, 8 * h), ss), mul_pair(fp4pair(word, 8 * h + 4), ss));
      }
    }
}

template <int M>
__device__ __forceinline__ void k_loop(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                       const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  uint32_t sel[2];
  scale_sel(sel, lane >> 2);
  constexpr int D = 2;
  Stage<M> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M>(acc, st[d], sel);
        if (g + D < KG) load_stage<M>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// Pair p reads X row p / slots when slots > 0 (X holds tokens), else X row p; items of expert ``skip`` are left
// alone (the shared expert, run elsewhere), and an item's pairs go 16 at a time.
// EVEN (zspark's own, spec 073): the units over the warps in whole rounds. TensorFold's stride gives warp w the
// units w, w + W, ...; with a few more units than the grid's W warps (40 experts of 20 units on 768) the units
// past W are read by a few warps alone, each at its own chain's pace, while the rest of the grid has ended. Here
// the R = ceil(units / W) rounds go to A = ceil(units / R) warps, spread over the blocks, R units each: as many
// loads in flight at the end as at the start. Which warp sums a unit changes; what it sums does not.
template <int M, int EPI, int WARPS, bool EVEN>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                        const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                        float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
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
  for (; unit < units; unit += stride) {
    const int it = unit / NB, cb = unit - it * NB;
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
      float acc[M][1][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
      k_loop<M>(acc, blk, KG, lane, t, x0, x1, v0, v1);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float g = __ldg(scale + e * M + m);
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
      }
      epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
    }
  }
}

template <int M, int EPI, bool EVEN>
void launch(const zs::View& x, int x_stride, int slots, const zs::View& w, const zs::View& scale, int kg,
            int nb, const zs::View& items, const zs::View& counts, const zs::View& members, zs::View& out,
            int n, float limit, int skip, int64_t max_units) {
  constexpr int WARPS = 4;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_kernel<M, EPI, WARPS, false>, WARPS * 32, 0);
    sms = zs::device().multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = (max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  // the whole rounds only where a launch can have more units than warps: below, the stride is one round
  if (EVEN && need > grid)
    nvfp4_expert_kernel<M, EPI, WARPS, true><<<grid, WARPS * 32, 0, zs::stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
        reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
        counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  else
    nvfp4_expert_kernel<M, EPI, WARPS, false><<<grid, WARPS * 32, 0, zs::stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
        reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
        counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

void nvfp4_experts_wide_cuda(int64_t, const zs::View&, int64_t, int64_t, const zs::View&, const zs::View&, int64_t,
                             int64_t, const zs::View&, const zs::View&, const zs::View&, zs::View&, int64_t, double,
                             int64_t, int64_t);

void nvfp4_experts_tile_cuda(int64_t, const zs::View&, int64_t, int64_t, const zs::View&, const zs::View&, int64_t,
                             int64_t, const zs::View&, const zs::View&, const zs::View&, zs::View&, int64_t, double,
                             int64_t, int64_t);
int64_t nvfp4_experts_half_resident();
void nvfp4_experts_half_cuda(int64_t, const zs::View&, int64_t, int64_t, const zs::View&, const zs::View&, int64_t,
                             int64_t, const zs::View&, const zs::View&, const zs::View&, zs::View&, int64_t, double,
                             int64_t, int64_t);


// tiles: the m16 tiles of rows one reading of an expert's weights serves. 1 is the kernel above; 2 is
// experts_wide.cu's, 4 experts_tile.cu's, with the same bits. even: the kernel above with its units in whole rounds, the same bits.
void nvfp4_experts_cuda(int64_t epi, int64_t tiles, int64_t even, const zs::View& x, int64_t x_stride, int64_t slots,
                        const zs::View& w, const zs::View& scale, int64_t kg, int64_t nb, const zs::View& items,
                        const zs::View& counts, const zs::View& members, zs::View& out, int64_t n, double limit,
                        int64_t skip, int64_t max_units) {
  if (tiles == 4) {
    nvfp4_experts_tile_cuda(epi, x, x_stride, slots, w, scale, kg, nb, items, counts, members, out, n, limit, skip,
                            max_units);
    return;
  }
  if (tiles != 1) {
    nvfp4_experts_wide_cuda(epi, x, x_stride, slots, w, scale, kg, nb, items, counts, members, out, n, limit, skip,
                            max_units);
    return;
  }
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  // TensorFold 1.2.1's experts_waves.pick: two warps a unit while every half unit fits the half kernel's
  // resident warps (the SwiGLU gate and up alone; the other epilogues keep the kernels below).
  if (epi == 2 && even && 2 * max_units <= nvfp4_experts_half_resident()) {
    nvfp4_experts_half_cuda(epi, x, x_stride, slots, w, scale, kg, nb, items, counts, members, out, n, limit,
                            skip, max_units);
    return;
  }
  if (epi == 2 && even) launch<2, 2, true>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 2) launch<2, 2, false>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 0 && even) launch<1, 0, true>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 0) launch<1, 0, false>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 3 && even) launch<1, 3, true>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 3) launch<1, 3, false>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 1 && even) launch<1, 1, true>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 1) launch<1, 1, false>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else zs::fail("nvfp4 experts: epilogue 0 (fp32 down), 1 (relu squared up), 2 (SwiGLU) or 3 (bf16 down)");
}
