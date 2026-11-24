// What the text attention's kernels share (moved from zig/kernels/qwen4/ by spec 138 WS-04): a block's sizes, the e4m3 mma and the fragments beside
// common/qmm_frag.cuh's, and the NVFP4 values' codes as e4m3. Moved out of attention.cu as they were (spec 073
// WS-08, for a kernel split by tiles, withdrawn in spec 121); attention.cu's header says where each comes from.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../common/copies.cuh"
#include "../common/math.cuh"
#include "../common/qmm_frag.cuh"
#include "../nvfp4/codes.cuh"

namespace {

using namespace common_copies;

using namespace common_math;

using nvfp4_codes::e4m3f;

constexpr int WIDE = 256;      // the most values of a head; the threads of a prep or chunk block
constexpr int CHUNK = 512;     // keys of a row's list a chunk block attends
constexpr int TILE = 64;       // keys a step of the online softmax takes
constexpr int HEADS = 16;      // an mma's rows: the query heads a KV head serves, at most
constexpr int THREADS = 128;   // a chunk block's four warps: 16 keys of a tile each, a quarter of a head's values each
constexpr int MAX_G = HEADS;

// For an e4m3 scale code (sign clear), the e4m3 codes of the scale's value times each e2m1 magnitude over 6, byte c
// for magnitude code c: what fp32 (s * magnitude / 6) rounds to, nearest even. In units of 2^-9 the scale is an
// integer S and the product N / 12 with N = S * (twice the magnitude); a tie of that rounding is an fp32 value, so
// rounding the fp32 quotient gives the same code.
struct ByCode {
  uint64_t of[128];
};
constexpr ByCode nvfp4_by_code() {
  ByCode table{};
  const int twice[8] = {0, 1, 2, 3, 4, 6, 8, 12};
  for (int scale = 0; scale < 128; ++scale) {
    const int exponent = scale >> 3, mantissa = scale & 7;
    const uint64_t S = exponent == 0 ? mantissa : static_cast<uint64_t>(8 + mantissa) << (exponent - 1);
    uint64_t packed = 0;
    for (int code = 1; code < 8; ++code) {
      const uint64_t N = S * twice[code];
      // the step is 2^k where N / 12 lies in [8, 16) 2^k, and 1 below 8
      int k = 0;
      while (N >= (static_cast<uint64_t>(16 * 12) << k)) ++k;
      const uint64_t unit = static_cast<uint64_t>(12) << k;
      uint64_t r = N / unit;
      const uint64_t rest = N - r * unit;
      if (2 * rest > unit || (2 * rest == unit && (r & 1))) ++r;
      if (r == 16) {
        r = 8;
        ++k;
      }
      const uint64_t byte = r < 8 ? r : (static_cast<uint64_t>(k + 1) << 3 | (r - 8));
      packed |= byte << (code * 8);
    }
    table.of[scale] = packed;
  }
  return table;
}
__device__ const ByCode BY_CODE = nvfp4_by_code();

// The bytes a head's values take as NVFP4: two codes a byte, a scale a 16 values, up to a multiple of 8.
__host__ __device__ constexpr int nvfp4_width(int d) { return (d / 2 + d / 16 + 7) & ~7; }

// The e4m3 mma: 16 x 8 products over 32 codes a step, fp32 sums. The fragments are the bf16 mma's with two codes
// where it has one value, so ldmatrix loads them from rows of codes as it does from rows of bf16.
__device__ __forceinline__ void mma8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Four words of four codes each, transposed: out[i] holds byte i of a, b, c and d.
__device__ __forceinline__ void transpose4(uint32_t (&out)[4], uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  const uint32_t ab = __byte_perm(a, b, 0x5140), ab2 = __byte_perm(a, b, 0x7362);
  const uint32_t cd = __byte_perm(c, d, 0x5140), cd2 = __byte_perm(c, d, 0x7362);
  out[0] = __byte_perm(ab, cd, 0x5410);
  out[1] = __byte_perm(ab, cd, 0x7632);
  out[2] = __byte_perm(ab2, cd2, 0x5410);
  out[3] = __byte_perm(ab2, cd2, 0x7632);
}

}  // namespace
