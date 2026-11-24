// zspark's (spec 121 WS-04): the copies into and out of shared memory that were the same text in several kernels.
// cp16, cp_commit and cp_wait are experts.cuh's (TensorFold 0.6.5, 609ca41, Apache-2.0), which takes them from
// here; qmm_frag.cuh's cp16 is another body (the address through `smem`) and stays. A kernel takes them with
// `using namespace common_copies;`.
// The one home of the ldmatrix family (spec 158 WS-01); qmm_frag.cuh keeps TensorFold's unchanged body.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace common_copies {

using qmm_frag::ldmatrix2;
using qmm_frag::ldmatrix4;

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ldmatrix of two 8x8 tiles read down their columns: the B fragment of an mma whose K runs over the rows.
__device__ __forceinline__ void ldmatrix2t(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(qmm_frag::smem(p)));
}

// The same of four 8x8 tiles.
__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(qmm_frag::smem(p)));
}

}  // namespace common_copies
