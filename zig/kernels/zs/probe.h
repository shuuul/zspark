// C ABI shared by zig/kernels/zs/probe.cu and zig/src/cuda/cuda.zig; keep the two in step.
#pragma once

#include <stdint.h>

#define ZSPARK_PROBE_N 4096

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zspark_device_info {
    char name[256];
    int32_t major;
    int32_t minor;
    uint64_t free_bytes;
    uint64_t total_bytes;
    uint64_t kernel_sum;
} zspark_device_info;

int zspark_probe(zspark_device_info* info);
const char* zspark_cuda_error(int code);

#ifdef __cplusplus
}
#endif
