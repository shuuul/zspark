// GLM-5.3-Flash's KDA chain and replay with the state loaded and stored four floats at a time
// (ZSPARK_GLM_KDA_QUADS, spec 073 WS-05): kda.cu's chain and replay (kda_common.cuh, TensorFold 0.6.6, cb2ebf0,
// families/glm5_next/cuda/kda.cu, Apache-2.0) with `load4` and `store4` one 16-byte access a row of a thread's
// 4 x 4 block of the state, for four 4-byte ones. A scalar store a thread sends its 32-byte sector again each
// time: a chain's 2.1 MB of state was 8.09 MB of sectors written (ncu, lts__t_sectors_op_write; 252,751 sectors),
// as Flash Next's was (qwen4/gdn.cu, QUADS). No arithmetic differs, so these are kda.cu's bits; kda_test holds the
// two together. Compile with --fmad=false, as kda.cu. kda.cu's entries launch these when the switch is on and the
// states are 16-byte aligned.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace glm5_kda_quads {

// A thread's four state values of a row as one 16-byte load or store.
__device__ __forceinline__ void load4(float (&row)[4], const float* p) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    row[0] = v.x, row[1] = v.y, row[2] = v.z, row[3] = v.w;
}
__device__ __forceinline__ void store4(float* p, const float (&row)[4]) {
    *reinterpret_cast<float4*>(p) = make_float4(row[0], row[1], row[2], row[3]);
}

}  // namespace glm5_kda_quads

#define KDA_NAMESPACE glm5_kda_quads
#include "kda_common.cuh"
#undef KDA_NAMESPACE
