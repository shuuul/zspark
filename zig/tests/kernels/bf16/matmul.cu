// BF16 matmul for the linears a checkpoint keeps unquantized: out (M, N) = x (M, K) @ W^T with W [N, K] as stored.
// Each element adds its K products 16 at a time in order in fp32, so no row affects another; K slices follow from
// the weight's shape alone and add in slice order.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/bf16.py (Triton), Apache-2.0; rewritten for
// zspark in CUDA C++.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../../../kernels/bf16/reduce.cuh"
#include "../../../kernels/common/experts.cuh"
#include "../../../kernels/zs/host.h"

namespace {

constexpr int WARPS = 4;        // a block: 4 warps of 32 output columns
constexpr int ROWS = 16;        // rows a block

__device__ __forceinline__ uint32_t pair(const __nv_bfloat16* p) {
  return __ldg(reinterpret_cast<const uint32_t*>(p));
}

// Block (column block, row tile, K slice). A warp holds 16 rows by 32 columns (NTW n8 tiles) of fp32 sums.
template <bool F32>
__global__ void __launch_bounds__(WARPS * 32)
    b16_kernel(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W, void* __restrict__ out,
               float* __restrict__ part, int M, int N, int K, int SK, int ldx) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gq = lane >> 2, t = lane & 3;
  const int col0 = (blockIdx.x * WARPS + warp) * COLS;
  if (col0 >= N) return;
  const int r0 = blockIdx.y * ROWS + gq, r1 = r0 + 8;
  const bool v0 = r0 < M, v1 = r1 < M;
  const __nv_bfloat16* x0 = X + (size_t)r0 * ldx + 2 * t;
  const __nv_bfloat16* x1 = X + (size_t)r1 * ldx + 2 * t;
  const __nv_bfloat16* w[NTW];
  bool wv[NTW];
#pragma unroll
  for (int j = 0; j < NTW; ++j) {
    const int n = col0 + 8 * j + gq;
    wv[j] = n < N;
    w[j] = W + (size_t)(wv[j] ? n : 0) * K + 2 * t;
  }
  float acc[NTW][4];
#pragma unroll
  for (int j = 0; j < NTW; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  const int per = K / SK, k0 = blockIdx.z * per;
  for (int k = k0; k < k0 + per; k += 16) {
    const uint32_t a0 = v0 ? pair(x0 + k) : 0u, a2 = v0 ? pair(x0 + k + 8) : 0u;
    const uint32_t a1 = v1 ? pair(x1 + k) : 0u, a3 = v1 ? pair(x1 + k + 8) : 0u;
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const uint32_t b0 = wv[j] ? pair(w[j] + k) : 0u, b1 = wv[j] ? pair(w[j] + k + 8) : 0u;
      mma(acc[j], a0, a1, a2, a3, b0, b1);
    }
  }

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? r1 : r0) * N;
#pragma unroll
    for (int j = 0; j < NTW; ++j)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int col = col0 + 8 * j + 2 * t + c;
        if (col >= N) continue;
        const float v = acc[j][2 * h + c];
        if (SK > 1) part[(size_t)blockIdx.z * M * N + row + col] = v;
        else if constexpr (F32) reinterpret_cast<float*>(out)[row + col] = v;
        else reinterpret_cast<__nv_bfloat16*>(out)[row + col] = __float2bfloat16_rn(v);
      }
  }
}

template <bool F32>
void launch(const zs::View& x, const zs::View& w, zs::View& out, const zs::View& part, int N, int K, int SK) {
  const int M = static_cast<int>(x.size(0));
  const dim3 grid((N + WARPS * COLS - 1) / (WARPS * COLS), (M + ROWS - 1) / ROWS, SK);
  b16_kernel<F32><<<grid, WARPS * 32, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), w.data_ptr<__nv_bfloat16>(), out.data_ptr(),
      part.defined() ? part.data_ptr<float>() : nullptr, M, N, K, SK, static_cast<int>(x.stride(0)));
  ZS_CUDA_LAUNCH_CHECK();
  if (SK > 1) {
    const long long total = static_cast<long long>(M) * N;
    b16_reduce<F32><<<static_cast<unsigned>((total + 255) / 256), 256, 0, zs::stream()>>>(
        part.data_ptr<float>(), out.data_ptr(), total, SK);
    ZS_CUDA_LAUNCH_CHECK();
  }
}

}  // namespace

// x (M, K) bf16, w [N, K] bf16, out (M, N) bf16 or fp32; ``part`` (SK, M, N) fp32 scratch when sk > 1.
extern "C" int zs_b16_matmul(const zs_view* x_, const zs_view* w_, const zs_view* out_, const zs_view* part_,
                             int64_t sk) {
  zs::begin();
  const zs::View x(x_), w(w_), part(part_);
  zs::View out(out_);
  ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.dim() == 2 && w.is_contiguous(), "w: [N, K] bf16");
  const int64_t n = w.size(0), k = w.size(1);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.size(1) == k && x.size(0) >= 1,
             "x: (M, K) bf16");
  ZS_REQUIRE(reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0 && x.stride(0) % 2 == 0,
             "x rows must start on 4-byte boundaries");
  ZS_REQUIRE(sk >= 1 && k % 16 == 0 && k % sk == 0 && (k / sk) % 16 == 0, "K in whole steps of 16 a slice");
  const int64_t m = x.size(0);
  ZS_REQUIRE(out.defined() && out.is_contiguous() && out.dim() == 2 && out.size(0) == m && out.size(1) == n &&
             (out.dtype() == ZS_F32 || out.dtype() == ZS_BF16), "out: (M, N) bf16 or fp32");
  ZS_REQUIRE(sk == 1 || (part.defined() && part.dtype() == ZS_F32 && part.numel() >= sk * m * n),
             "part: (SK, M, N) fp32");
  if (out.dtype() == ZS_F32) launch<true>(x, w, out, part, static_cast<int>(n), static_cast<int>(k), static_cast<int>(sk));
  else launch<false>(x, w, out, part, static_cast<int>(n), static_cast<int>(k), static_cast<int>(sk));
  return zs::status();
}
