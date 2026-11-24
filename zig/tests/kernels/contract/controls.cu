// Spec 158 WS-02's planted controls: each kernel breaks one gate of the kernel contract on purpose, so that a
// gate which reports nothing is proven broken. Linked into the test binaries alone.
//
//   zs_contract_max    a row maximum by fmaxf, which drops a NaN                      gate 1
//   zs_contract_alias  a row sum that writes into its input                           gate 2
//   zs_contract_tail   a row sum whose whole-tile loop loses the ragged tail           gate 3
//   zs_contract_half   a row sum in bf16 instead of fp32                               gate 4
//   zs_contract_sum    the correct fp32 sum (the other controls' own reference)

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "../../../kernels/zs/host.h"

namespace {

__device__ __forceinline__ float row_max(const __nv_bfloat16* x, long long d) {
  float m = -INFINITY;
  for (long long i = 0; i < d; ++i) m = fmaxf(m, __bfloat162float(x[i]));
  return m;
}

template <bool WRITE_INPUT>
__device__ __forceinline__ float row_sum(__nv_bfloat16* x, long long d) {
  float s = 0.f;
  for (long long i = 0; i < d; ++i) s += __bfloat162float(x[i]);
  if (WRITE_INPUT) x[0] = __float2bfloat16(__bfloat162float(x[0]) + 1.f);
  return s;
}

__device__ __forceinline__ float row_sum_tail(const __nv_bfloat16* x, long long d) {
  const long long whole = d - d % 256;
  float s = 0.f;
  for (long long i = 0; i < whole; ++i) s += __bfloat162float(x[i]);
  return s;
}

__device__ __forceinline__ float row_sum_bf16(const __nv_bfloat16* x, long long d) {
  __nv_bfloat16 s = __float2bfloat16(0.f);
  for (long long i = 0; i < d; ++i) s = __float2bfloat16(__bfloat162float(s) + __bfloat162float(x[i]));
  return __bfloat162float(s);
}

// One thread a row: the tests' shapes are small and their reference is this chain in f64.
__global__ void contract_max_kernel(const __nv_bfloat16* x, long long stride, float* out, long long d) {
  out[blockIdx.x] = row_max(x + (size_t)blockIdx.x * stride, d);
}

__global__ void contract_sum_kernel(const __nv_bfloat16* x, long long stride, float* out, long long d) {
  out[blockIdx.x] = row_sum<false>(const_cast<__nv_bfloat16*>(x + (size_t)blockIdx.x * stride), d);
}

__global__ void contract_alias_kernel(__nv_bfloat16* x, long long stride, float* out, long long d) {
  __nv_bfloat16* row = x + (size_t)blockIdx.x * stride;
  out[blockIdx.x] = row_sum<true>(row, d);
}

__global__ void contract_tail_kernel(const __nv_bfloat16* x, long long stride, float* out, long long d) {
  out[blockIdx.x] = row_sum_tail(x + (size_t)blockIdx.x * stride, d);
}

__global__ void contract_half_kernel(const __nv_bfloat16* x, long long stride, float* out, long long d) {
  out[blockIdx.x] = row_sum_bf16(x + (size_t)blockIdx.x * stride, d);
}

}  // namespace

namespace {

// x bf16 [R, D] contiguous, out fp32 [R, 1] contiguous: the controls' one shape.
bool rows_of(const zs::View& x, const zs::View& out) {
  return x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && out.defined() &&
         out.dtype() == ZS_F32 && out.dim() == 2 && out.size(0) >= x.size(0) && out.size(1) == 1 &&
         out.is_contiguous();
}

}  // namespace

extern "C" {

int zs_contract_max(const zs_view* x_, const zs_view* out_) {
  zs::begin();
  const zs::View x(x_), out(out_);
  ZS_REQUIRE(rows_of(x, out), "x bf16 [R, D] and out fp32 [R, 1]");
  contract_max_kernel<<<static_cast<unsigned>(x.size(0)), 1, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), x.stride(0), out.data_ptr<float>(), x.size(1));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

int zs_contract_sum(const zs_view* x_, const zs_view* out_) {
  zs::begin();
  const zs::View x(x_), out(out_);
  ZS_REQUIRE(rows_of(x, out), "x bf16 [R, D] and out fp32 [R, 1]");
  contract_sum_kernel<<<static_cast<unsigned>(x.size(0)), 1, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), x.stride(0), out.data_ptr<float>(), x.size(1));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

int zs_contract_alias(const zs_view* x_, const zs_view* out_) {
  zs::begin();
  const zs::View x(x_), out(out_);
  ZS_REQUIRE(rows_of(x, out), "x bf16 [R, D] and out fp32 [R, 1]");
  contract_alias_kernel<<<static_cast<unsigned>(x.size(0)), 1, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), x.stride(0), out.data_ptr<float>(), x.size(1));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

int zs_contract_tail(const zs_view* x_, const zs_view* out_) {
  zs::begin();
  const zs::View x(x_), out(out_);
  ZS_REQUIRE(rows_of(x, out), "x bf16 [R, D] and out fp32 [R, 1]");
  contract_tail_kernel<<<static_cast<unsigned>(x.size(0)), 1, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), x.stride(0), out.data_ptr<float>(), x.size(1));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

int zs_contract_half(const zs_view* x_, const zs_view* out_) {
  zs::begin();
  const zs::View x(x_), out(out_);
  ZS_REQUIRE(rows_of(x, out), "x bf16 [R, D] and out fp32 [R, 1]");
  contract_half_kernel<<<static_cast<unsigned>(x.size(0)), 1, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), x.stride(0), out.data_ptr<float>(), x.size(1));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
