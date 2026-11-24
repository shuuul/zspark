// The rank-order sum of a reduce-scatter by rows (spec 106 WS-02), after TensorFold v1.0.12's
// `zig/kernels/cuda/rank_sum.cu` (d309dad9, 88943c57): each rank adds only the rows it glues, of partials that
// each peer sent it into a gathered buffer where an all-gather would have put them (`ranks/ranks.zig`).
// zspark's own: the sum is `qwen4/partial.cu`'s add, not TensorFold's. It starts from 0.f and adds rank 0
// first, so a -0 partial gives +0 as it does today; TensorFold starts from rank 0's term.
// Every element is its own: no order of threads or blocks shows. A thread takes four neighbouring values of one
// row (float4) where the width allows, one otherwise; the arithmetic is the same either way. No shared memory.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"

namespace {

// in fp32 [ranks, rows, dims]; out [count, dims], bf16 or fp32; out row i is in row first + i, rank 0 first.
__global__ void rank_sum_kernel(const float* __restrict__ in, void* __restrict__ out, int ranks, long long stride,
                                long long first, long long n, int to_bf16) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sum = 0.f;
  for (int r = 0; r < ranks; ++r) sum += in[r * stride + first + i];
  if (to_bf16) static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(sum);
  else static_cast<float*>(out)[i] = sum;
}

// The same, four values a thread: `n`, `stride` and `first` in values, all multiples of 4.
__global__ void rank_sum4_kernel(const float* __restrict__ in, void* __restrict__ out, int ranks, long long stride,
                                 long long first, long long n, int to_bf16) {
  const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q * 4 >= n) return;
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int r = 0; r < ranks; ++r) {
    const float4 v = *reinterpret_cast<const float4*>(in + r * stride + first + q * 4);
    sum.x += v.x;
    sum.y += v.y;
    sum.z += v.z;
    sum.w += v.w;
  }
  if (to_bf16) {
    __nv_bfloat16* o = static_cast<__nv_bfloat16*>(out) + q * 4;
    o[0] = __float2bfloat16_rn(sum.x);
    o[1] = __float2bfloat16_rn(sum.y);
    o[2] = __float2bfloat16_rn(sum.z);
    o[3] = __float2bfloat16_rn(sum.w);
  } else {
    *reinterpret_cast<float4*>(static_cast<float*>(out) + q * 4) = sum;
  }
}

}  // namespace

extern "C" {

// gathered fp32 [ranks, rows, dims]: rows first to first + count summed over the ranks, rank 0 first, into
// `into` [count, dims], bf16 when `to_bf16` else fp32.
int zs_rank_sum(const zs_view* gathered, int64_t first, const zs_view* into, int64_t to_bf16) {
  zs::begin();
  const zs::View in(gathered), out(into);
  ZS_REQUIRE(in.defined() && in.dtype() == ZS_F32 && in.dim() == 3 && in.is_contiguous(), "gathered: fp32 [ranks, rows, dims]");
  const long long ranks = in.size(0), rows = in.size(1), dims = in.size(2);
  ZS_REQUIRE(ranks >= 1 && ranks <= 64 && rows >= 1 && dims >= 1, "gathered: a rank and a row");
  const bool bf16 = to_bf16 != 0;
  ZS_REQUIRE(out.defined() && out.is_contiguous() && out.dim() == 2 && out.size(1) == dims && out.size(0) >= 1,
             "into: [count, dims]");
  const long long count = out.size(0);
  ZS_REQUIRE(first >= 0 && first + count <= rows, "the rows inside the gathered window");
  ZS_REQUIRE(out.dtype() == (bf16 ? ZS_BF16 : ZS_F32), "into: bf16 for a branch, fp32 for a MoE sum");
  const long long stride = rows * dims;
  const long long n = count * dims;
  const int threads = 256;
  const bool four = dims % 4 == 0 && (reinterpret_cast<uintptr_t>(in.data_ptr()) & 15) == 0 &&
                    (reinterpret_cast<uintptr_t>(out.data_ptr()) & (bf16 ? 7 : 15)) == 0;
  if (four) {
    const long long quads = n / 4;
    rank_sum4_kernel<<<(unsigned)((quads + threads - 1) / threads), threads, 0, zs::stream()>>>(
        in.data_ptr<float>(), out.data_ptr(), (int)ranks, stride, first * dims, n, bf16 ? 1 : 0);
  } else {
    rank_sum_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, zs::stream()>>>(
        in.data_ptr<float>(), out.data_ptr(), (int)ranks, stride, first * dims, n, bf16 ? 1 : 0);
  }
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
