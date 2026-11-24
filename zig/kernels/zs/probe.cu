// Toolchain probe: reports the GPU and runs one kernel, so a build that links but cannot launch fails here.

#include <cuda_runtime.h>
#include <cstdint>

#include "probe.h"

namespace {

__global__ void iota_sum_kernel(uint64_t* out, int n) {
    uint64_t acc = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += static_cast<uint64_t>(i);
    atomicAdd(reinterpret_cast<unsigned long long*>(out), static_cast<unsigned long long>(acc));
}

}  // namespace

extern "C" int zspark_probe(zspark_device_info* info) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return static_cast<int>(err);
    cudaDeviceProp prop;
    if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) return static_cast<int>(err);
    size_t free_bytes = 0, total_bytes = 0;
    if ((err = cudaMemGetInfo(&free_bytes, &total_bytes)) != cudaSuccess) return static_cast<int>(err);

    uint64_t* sum = nullptr;
    if ((err = cudaMalloc(&sum, sizeof(uint64_t))) != cudaSuccess) return static_cast<int>(err);
    cudaMemset(sum, 0, sizeof(uint64_t));
    iota_sum_kernel<<<1, 256>>>(sum, ZSPARK_PROBE_N);
    uint64_t host_sum = 0;
    err = cudaMemcpy(&host_sum, sum, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(sum);
    if (err != cudaSuccess) return static_cast<int>(err);

    for (int i = 0; i < 256; ++i) info->name[i] = prop.name[i];
    info->major = prop.major;
    info->minor = prop.minor;
    info->free_bytes = free_bytes;
    info->total_bytes = total_bytes;
    info->kernel_sum = host_sum;
    return 0;
}

extern "C" const char* zspark_cuda_error(int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); }
