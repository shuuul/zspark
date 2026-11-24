// zspark's (spec 121 WS-04): the device one-liners that were the same text in many kernels, each a single copy.
// A kernel takes them with `using namespace common_math;` in its own namespace; a file whose helper of the same
// name differs keeps its own. They are inlined: an object's instructions are what its own copies compiled to.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace common_math {

// A value rounded to bf16 and back: one rounding of the model's.
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// The float of a bf16's bits in a word's low half, and in its high half: exact (spec 136 WS-01).
__device__ __forceinline__ float bf16_low(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_high(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bsig(float x) { return bf(1.0f / (1.0f + expf(-x))); }
__device__ __forceinline__ float bsilu(float x) { return bf(x / (1.0f + expf(-x))); }

// A warp's sum by xor shuffles 16, 8, 4, 2, 1: every lane ends with the same bits.
__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float minus_inf() { return -__int_as_float(0x7f800000); }

}  // namespace common_math
