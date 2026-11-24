// zspark's own (spec 021): the grouped NVFP4 experts of a wide window, an expert's weights read once for up to
// 16 RT pairs. A departure from TensorFold 0.6.5, whose kernel (experts.cu beside this, taken unchanged) reads
// them once for every 16 pairs of an expert: its device code is the model here, with RT m16 tiles of rows held
// under each stage of weights, as TensorFold's bf16 prefill form holds them (cuda/experts_prefill.cu, 609ca41,
// Apache-2.0). ld_nc, fp4pair, scale_pair, mul_pair, the stage's layout and the arithmetic (the scale folded into
// the weights, the mma adding into the sums) are experts.cu's.
//
// Measured on spark-b (spec 021): with RT 2 on items of 32 pairs a 512-row window's routed experts take 7 to 8%
// less than experts.cu (it reads 1.2 to 1.4 times an expert where that reads 1.7 to 2.0). RT 2 is all that is
// launched. The tiles cost registers (168 a thread for gate and up against 128, 128 for down against 96), so
// fewer warps an SM have loads in flight: RT 4, items of 64 pairs, a third stage held ahead and a
// prefetch.global.L2 further on were each slower than RT 2 on items of 32, and a call in which no expert has
// more than 16 pairs (a step, a verify window, 16 rows) is 3% slower here than in experts.cu, which keeps it.
//
// The arithmetic is experts.cu's to the bit: per 16-input block acc = fma(P, e4m3 scale, acc) in block order, one
// pair an mma row, each (expert, matrix) scale multiplied once before the epilogue. A pair's sums are its own
// accumulators whichever tile of the pass it sits in, so a pair's bits depend on no other pair and are those of
// experts.cu, whatever the plan's item size and whatever RT.
// Not CUTLASS, for experts.cu's reasons: bf16 rows against NVFP4 weights, a dozen rows an expert, gathered
// through the plan.

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

// A lane's rows of one pass: tile r holds pairs r0 + 16 r + gq and that + 8.
template <int RT>
struct Rows {
  const __nv_bfloat16* x0[RT];
  const __nv_bfloat16* x1[RT];
  int p0[RT], p1[RT];
  bool v0[RT], v1[RT];
};

template <int M, int RT>
struct Stage {
  uint4 w[M];              // the lane's words: n8 tiles 0-3, 32 inputs each
  uint4 s[M];              // the scales of the quad that holds column gq, bytes [block][tile][column]
  uint2 xa[RT][2], xb[RT][2];  // each tile's rows gq and gq + 8: inputs 4t .. 4t + 3 of each 16-input block
};

template <int M, int RT>
__device__ __forceinline__ void load_stage(Stage<M, RT>& st, const uint4* blk, int g, int lane, int t,
                                           const Rows<RT>& rows) {
  const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m] = ld_nc(b + m * BLOCK4 + lane);
    st.s[m] = ld_nc(b + m * BLOCK4 + 32 + (lane >> 3));
  }
  const uint2 zero = make_uint2(0u, 0u);
  const int k0 = g * 32 + 4 * t;
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      st.xa[r][h] = rows.v0[r] ? __ldg(reinterpret_cast<const uint2*>(rows.x0[r] + k0 + 16 * h)) : zero;
      st.xb[r][h] = rows.v1[r] ? __ldg(reinterpret_cast<const uint2*>(rows.x1[r] + k0 + 16 * h)) : zero;
    }
}

// nt: the tiles of this pass that hold a pair, the same for every lane of the warp.
template <int M, int RT>
__device__ __forceinline__ void compute_stage(float (&acc)[M][RT][NTW][4], const Stage<M, RT>& st, int nt,
                                              const uint32_t (&sel)[2]) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const uint32_t word = comp(st.w[m], j);
        const uint32_t ss = scale_pair(comp(st.s[m], 2 * h + (j >> 1)), sel[j & 1]);
        const uint32_t b0 = mul_pair(fp4pair(word, 8 * h), ss), b1 = mul_pair(fp4pair(word, 8 * h + 4), ss);
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (r >= nt) break;
          mma(acc[m][r][j], st.xa[r][h].x, st.xb[r][h].x, st.xa[r][h].y, st.xb[r][h].y, b0, b1);
        }
      }
    }
}

template <int M, int RT>
__device__ __forceinline__ void k_loop(float (&acc)[M][RT][NTW][4], const uint4* blk, int KG, int lane, int t,
                                       const Rows<RT>& rows, int nt) {
  uint32_t sel[2];
  scale_sel(sel, lane >> 2);
  constexpr int D = 2;
  Stage<M, RT> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M, RT>(st[d], blk, d, lane, t, rows);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M, RT>(acc, st[d], nt, sel);
        if (g + D < KG) load_stage<M, RT>(st[d], blk, g + D, lane, t, rows);
      }
    }
  }
}

// experts.cu's kernel with an item's pairs taken 16 RT at a time: pair p reads X row p / slots when slots > 0,
// else X row p; items of expert ``skip`` are left alone.
template <int M, int EPI, int WARPS, int RT>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_wide_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                             const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                             const int* __restrict__ items, const int* __restrict__ counts,
                             const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16 * RT) {
      const int nt = min(RT, (cnt - r0 + 15) >> 4);
      Rows<RT> rows;
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const int m0 = r0 + 16 * r + gq;
        rows.v0[r] = m0 < cnt;
        rows.v1[r] = m0 + 8 < cnt;
        rows.p0[r] = rows.v0[r] ? __ldg(members + first + m0) : 0;
        rows.p1[r] = rows.v1[r] ? __ldg(members + first + m0 + 8) : 0;
        rows.x0[r] = X + (size_t)(slots ? rows.p0[r] / slots : rows.p0[r]) * x_stride;
        rows.x1[r] = X + (size_t)(slots ? rows.p1[r] / slots : rows.p1[r]) * x_stride;
      }
      float acc[M][RT][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
      k_loop<M, RT>(acc, blk, KG, lane, t, rows, nt);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float g = __ldg(scale + e * M + m);
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= g;
      }
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= nt) break;
        epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, rows.p0[r], rows.p1[r], rows.v0[r], rows.v1[r],
                             limit);
      }
    }
  }
}

template <int M, int EPI, int RT>
void launch(const zs::View& x, int x_stride, int slots, const zs::View& w, const zs::View& scale, int kg,
            int nb, const zs::View& items, const zs::View& counts, const zs::View& members, zs::View& out,
            int n, float limit, int skip, int64_t max_units) {
  constexpr int WARPS = 4;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_wide_kernel<M, EPI, WARPS, RT>, WARPS * 32,
                                                  0);
    sms = zs::device().multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = (max_units + WARPS - 1) / WARPS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  nvfp4_expert_wide_kernel<M, EPI, WARPS, RT><<<grid, WARPS * 32, 0, zs::stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

// nvfp4_experts_cuda's arguments (experts.cu), for two tiles of rows a pass.
void nvfp4_experts_wide_cuda(int64_t epi, const zs::View& x, int64_t x_stride, int64_t slots, const zs::View& w,
                             const zs::View& scale, int64_t kg, int64_t nb, const zs::View& items,
                             const zs::View& counts, const zs::View& members, zs::View& out, int64_t n,
                             double limit, int64_t skip, int64_t max_units) {
  constexpr int RT = 2;
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  if (epi == 2) launch<2, 2, RT>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 0) launch<1, 0, RT>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 3) launch<1, 3, RT>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 1) launch<1, 1, RT>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else zs::fail("nvfp4 experts: epilogue 0 (fp32 down), 1 (relu squared up), 2 (SwiGLU) or 3 (bf16 down)");
}
