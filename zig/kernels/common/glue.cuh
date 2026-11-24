// zspark's (spec 136 WS-01): the device helpers `glue.cu`'s kernels and glm5/hc.cu's read-out and write-back share,
// moved here from hc.cu with their text unchanged; `norm_row`, the norm of a row held in registers, moved from
// unlimited_ocr/moe.cu with its text unchanged (spec 142 WS-04), for its combine and glue.cu's bf16 residual. A
// kernel takes them with `using namespace common_glue;`. They are inlined: an object's instructions are what the
// files' own copies compiled to.
#pragma once

#include <cuda_bf16.h>

#include "math.cuh"

namespace common_glue {

// The sum over shared memory of P values (a power of two), by a block's threads, the upper half folded onto the
// lower until one is left: sq[0].
__device__ __forceinline__ void fold_shared(float* sq, int P) {
  for (int n = P / 2; n >= 1; n >>= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) sq[i] += sq[i + n];
    __syncthreads();
  }
}

// A norm's weight as the checkpoint stores it: bf16, or fp32 (FP4-Local keeps the model's post-attention and
// final norms so; the product is the same fp32 expression either way).
__device__ __forceinline__ float weight_of(__nv_bfloat16 w) { return __bfloat162float(w); }
__device__ __forceinline__ float weight_of(float w) { return w; }

// The ranks' fp32 partials of a branch added rank 0 first and rounded to bf16. G [WORLD, .., D], RS elements
// from one rank's rows to the next's.
__device__ __forceinline__ float widened(float v) { return v; }
__device__ __forceinline__ float widened(__nv_bfloat16 v) { return __bfloat162float(v); }

// The partials fp32, or bf16 in a quantized window (spec 116 WS-04: vLLM's and SGLang's, rounded by their writers).
template <typename G_T>
__device__ __forceinline__ float branch_of(const G_T* __restrict__ G, long long RS, int WORLD, long long i) {
  float acc = widened(G[i]);
  for (int k = 1; k < WORLD; ++k) acc = acc + widened(G[(long long)k * RS + i]);
  return common_math::bf(acc);
}

// The norm of a row whose D values a block of NORM_THREADS threads holds, thread t at v[k] = value t + 256 k (zeros
// past D), P the next power of two of D (512 or more): out = bf16(w bf16(v / rms)), `zs_rmsnorm`'s bits. Its squares
// fold over P exactly as `fold_shared` does (the upper half onto the lower, P / 2 down to 1): the folds of 1,024 to
// 256 stay in a thread's registers, those of 128 to 32 meet in shared memory (NORM_THREADS floats) and the last five
// in a warp's shuffles, each the same a + b of the same two values. Shared memory a block: 1,028 B (the folds' 256
// floats, the inverse).
constexpr int NORM_THREADS = 256;
constexpr int MAX_OWN = 16;  // values a thread owns: D to 4,096

template <int OWN>
__device__ __forceinline__ void norm_row(float (&v)[OWN], const __nv_bfloat16* __restrict__ W, __nv_bfloat16* out, int D,
                                         int P, float eps) {
  using common_math::bf;
  __shared__ float fold[NORM_THREADS];
  __shared__ float inverse;
  const int tid = threadIdx.x;
  float sq[OWN];
#pragma unroll
  for (int k = 0; k < OWN; ++k) sq[k] = v[k] * v[k];
  // folds of P / 2 down to 256 (index i + n onto i, both this thread's: n a multiple of 256)
#pragma unroll
  for (int own = OWN / 2; own >= 1; own >>= 1)
    if (own < P / NORM_THREADS)
#pragma unroll
      for (int k = 0; k < own; ++k) sq[k] += sq[k + own];
  fold[tid] = sq[0];
  __syncthreads();
  for (int n = NORM_THREADS / 2; n >= 32; n >>= 1) {
    if (tid < n) fold[tid] += fold[tid + n];
    __syncthreads();
  }
  if (tid < 32) {
    float x = fold[tid];
#pragma unroll
    for (int n = 16; n >= 1; n >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu, x, n);
      if (tid < n) x += other;
    }
    if (tid == 0) inverse = 1.0f / sqrtf(x / D + eps);
  }
  __syncthreads();
  const float inv = inverse;
#pragma unroll
  for (int k = 0; k < OWN; ++k) {
    const int d = tid + NORM_THREADS * k;
    if (d < D) out[d] = __float2bfloat16_rn(__bfloat162float(W[d]) * bf(v[k] * inv));
  }
}

}  // namespace common_glue
