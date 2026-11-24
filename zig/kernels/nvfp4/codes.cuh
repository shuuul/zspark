// zspark's (spec 121 WS-04): the NVFP4 and FP8 code decoders that were the same text in the nvfp4 kernels, TensorFold's
// (0.6.5, 609ca41, Apache-2.0: qmmf.cu, prompt.cu, experts.cu) and zspark's own, now one copy each; spec 136 WS-01 adds
// the e4m3 value (`e4m3f`: qmmf.cu's, qwen4/attention_mma.cuh's, glm5/mla_q8.cu's) and the signed e2m1 code
// (`fp4_e2m1`: qwen4/attention.cu's), the same text. A kernel takes
// them with `using namespace nvfp4_codes;`. prompt.cu's and prompt_bias.cu's fp8pair(w, unit) is another function
// and stays in those files.
#pragma once

#include <cuda_fp8.h>
#include <stdint.h>

namespace nvfp4_codes {

// bf16x2 of two e2m1 nibbles (bits [s, s + 4) and [16 + s, 20 + s)): exponent and mantissa into bf16's, times 2^126
// (exact).
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
    const uint32_t v = w >> s;
    const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
    return r;
}

// bf16x2 of two e4m3 bytes (bits [0, 8) and [8, 16)): exponent and mantissa into bf16's, times 2^120.
__device__ __forceinline__ uint32_t fp8pair(uint32_t w) {
    const uint32_t x = (w & 0xFFu) | ((w & 0xFF00u) << 8);
    const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
    return r;
}

// The bf16 pair (s, s) of the e4m3 scale in byte `sel` picks twice (__byte_perm): its fields into bf16's, times
// 2^120 (exact, subnormals too; a scale is never negative).
__device__ __forceinline__ uint32_t scale_pair(uint32_t word, uint32_t sel) {
  const uint32_t t = (__byte_perm(word, 0u, sel) & 0x007F007Fu) << 4;
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
  return r;
}

// The value of an e4m3 code (bits [0, 8)), through the half the conversion gives: exact, subnormals and the sign too.
__device__ __forceinline__ float e4m3f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

// a times b, bf16 pairs: exact for an e2m1 value times an e4m3 one (at most 6 significant bits, 2^-10 to 2,688)
__device__ __forceinline__ uint32_t mul_pair(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}

// e2m1's magnitude code of a quotient clamped to +-6, nearest with ties to even.
__device__ __forceinline__ uint32_t fp4_e2m1_code(float q) {
  const float a = fabsf(q);
  return uint32_t(a > 0.25f) + uint32_t(a >= 0.75f) + uint32_t(a > 1.25f) + uint32_t(a >= 1.75f) +
         uint32_t(a > 2.5f) + uint32_t(a >= 3.5f) + uint32_t(a > 5.0f);
}

// e2m1's code of a value clamped to +-6 (or past it, as 6), the sign in bit 3, none on a zero; a NaN 0.
__device__ __forceinline__ uint32_t fp4_e2m1(float q) {
  const uint32_t m = fp4_e2m1_code(q);
  return (q < 0.0f && m != 0u) ? (m | 8u) : m;
}

}  // namespace nvfp4_codes
