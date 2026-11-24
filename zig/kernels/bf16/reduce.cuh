// The K slices of a split BF16 matmul summed in slice order, one fp32 add a slice. Shared by gemm.cu and the tests' matmul.cu
// so both give an element the same bits.
#pragma once

#include <cuda_bf16.h>

namespace {

// Element i of SK slices of ``total`` elements each. A kernel that reads a split matmul's result calls this on
// the slices instead of reading what b16_reduce would have written: the same adds in the same order, one launch
// fewer (spec 028). The loads go out eight at a time and are added after, in slice order: an add after each
// load kept two in flight, and 32 slices took 3.4 us where they take 2 (spec 095 WS-04). Slice 0 is taken as
// it is (not 0 + it), so a -0 keeps its sign: the same bits as one add a slice. Two slices stay a load and an
// add each, which batching made slower at 16 rows.
__device__ __forceinline__ float b16_slices(const float* __restrict__ part, long long i, long long total, int SK) {
  if (SK <= 2) {
    float acc = part[i];
    if (SK == 2) acc += part[total + i];
    return acc;
  }
  constexpr int B = 8;
  float acc = 0.f;
  for (int s = 0; s < SK; s += B) {
    float v[B];
#pragma unroll
    for (int j = 0; j < B; ++j)
      if (s + j < SK) v[j] = part[(long long)(s + j) * total + i];
#pragma unroll
    for (int j = 0; j < B; ++j)
      if (s + j < SK) acc = s + j == 0 ? v[j] : acc + v[j];
  }
  return acc;
}

template <bool F32>
__global__ void b16_reduce(const float* __restrict__ part, void* __restrict__ out, long long total, int SK) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float acc = b16_slices(part, i, total, SK);
  if constexpr (F32) reinterpret_cast<float*>(out)[i] = acc;
  else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

}  // namespace
