// A rank's fp32 partial of a full-width row, and the sum of the ranks' (spec 050).
// Rank 0 is added first. The branch sum is rounded to bf16 once and then taken as hc write-back mode 1;
// the MoE sum is left fp32 and taken as mode 3, which rounds it once. Modes 0-3 of hc.cu are not changed.
// The fold is mode 2's fma chain (routed slots in order, then the shared expert) without that rounding.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"

namespace {

__global__ void add_kernel(const float* __restrict__ in, void* __restrict__ out, int ranks, int rows, int dims, int to_bf16) {
  const int n = rows * dims;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sum = 0.f;
  for (int r = 0; r < ranks; ++r) sum += in[(size_t)r * n + i];
  if (to_bf16) static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(sum);
  else static_cast<float*>(out)[i] = sum;
}

// y fp32 [rows * slots, dims], its last slot unread; shared fp32 [rows, dims]; wts fp32 [rows, slots].
// out fp32 [rows, dims] is mode 2's sum before the write-back rounds it.
__global__ void fold_kernel(const float* __restrict__ y, const float* __restrict__ shared, const float* __restrict__ wts,
                            float* __restrict__ out, int rows, int slots, int dims) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = rows * dims;
  if (i >= n) return;
  const int r = i / dims;
  const int d = i - r * dims;
  float acc = 0.f;
  for (int k = 0; k < slots; ++k) {
    const float from = k < slots - 1 ? y[((size_t)r * slots + k) * dims + d] : shared[(size_t)r * dims + d];
    acc = fmaf(from, wts[(size_t)r * slots + k], acc);
  }
  out[i] = acc;
}

}  // namespace

extern "C" {

// gathered fp32 [ranks, rows, dims] -> into, bf16 when `to_bf16` else fp32, both [rows, dims]. Rank 0 first.
int zs_qwen4_partial_add(const zs_view* gathered, const zs_view* into, int64_t to_bf16) {
  zs::begin();
  const zs::View in(gathered), out(into);
  ZS_REQUIRE(in.defined() && in.dtype() == ZS_F32 && in.dim() == 3 && in.is_contiguous(), "gathered: fp32 [ranks, rows, dims]");
  const int ranks = static_cast<int>(in.size(0));
  const int rows = static_cast<int>(in.size(1));
  const int dims = static_cast<int>(in.size(2));
  ZS_REQUIRE(ranks >= 1 && rows >= 1 && dims >= 1, "gathered: a rank and a row");
  const bool bf16 = to_bf16 != 0;
  ZS_REQUIRE(out.defined() && out.is_contiguous() && out.dim() == 2 && out.size(0) >= rows && out.size(1) == dims,
             "into: [rows, dims]");
  ZS_REQUIRE(out.dtype() == (bf16 ? ZS_BF16 : ZS_F32), "into: bf16 for a branch, fp32 for a MoE sum");
  const int threads = 256;
  const int blocks = (rows * dims + threads - 1) / threads;
  add_kernel<<<blocks, threads, 0, zs::stream()>>>(in.data_ptr<float>(), out.data_ptr(), ranks, rows, dims, bf16 ? 1 : 0);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// One rank's slots, folded by weight in slot order, into fp32 [rows, dims].
int zs_qwen4_moe_fold(const zs_view* y_, const zs_view* shared_, const zs_view* wts_, const zs_view* out_) {
  zs::begin();
  const zs::View y(y_), shared(shared_), wts(wts_), out(out_);
  ZS_REQUIRE(wts.defined() && wts.dtype() == ZS_F32 && wts.dim() == 2 && wts.is_contiguous(), "weights: fp32 [rows, slots]");
  const int rows = static_cast<int>(wts.size(0));
  const int slots = static_cast<int>(wts.size(1));
  ZS_REQUIRE(slots >= 1 && out.defined() && out.dtype() == ZS_F32 && out.dim() == 2 && out.is_contiguous() &&
                 out.size(0) >= rows,
             "out: fp32 [rows, dims]");
  const int dims = static_cast<int>(out.size(1));
  ZS_REQUIRE(y.defined() && y.dtype() == ZS_F32 && y.is_contiguous() && y.numel() >= (long long)rows * slots * dims,
             "y: fp32 [rows * slots, dims]");
  ZS_REQUIRE(shared.defined() && shared.dtype() == ZS_F32 && shared.is_contiguous() && shared.numel() >= (long long)rows * dims,
             "shared: fp32 [rows, dims]");
  const int threads = 256;
  const int blocks = (rows * dims + threads - 1) / threads;
  fold_kernel<<<blocks, threads, 0, zs::stream()>>>(y.data_ptr<float>(), shared.data_ptr<float>(), wts.data_ptr<float>(),
                                                    out.data_ptr<float>(), rows, slots, dims);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
