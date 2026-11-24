// What the launchers need of the host: a tensor view, this thread's stream, the device, and a failure record.
// Stands where ATen and c10 stood in TensorFold's kernels.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"

namespace zs {

struct View {
    zs_view v;

    View() : v{} {}
    View(const zs_view* p) : v(p ? *p : zs_view{}) {}

    bool defined() const { return v.ptr != nullptr; }
    void* data_ptr() const { return v.ptr; }
    template <class T> T* data_ptr() const { return static_cast<T*>(v.ptr); }
    int dim() const { return v.ndim; }
    int64_t size(int i) const { return v.size[i < 0 ? i + v.ndim : i]; }
    int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < v.ndim; ++i) n *= v.size[i];
        return n;
    }
    // Elements from one row to the next (dim 0 of a 2-D view); other dims are dense.
    int64_t row_stride() const { return v.row_stride ? v.row_stride : (v.ndim >= 2 ? v.size[v.ndim - 1] : 1); }
    int64_t stride(int i) const { return v.ndim == 2 && i == 0 ? row_stride() : 1; }
    bool is_contiguous() const { return v.ndim != 2 || v.row_stride == 0 || v.row_stride == v.size[1]; }
    int dtype() const { return v.dtype; }
    int64_t element_size() const {
        switch (v.dtype) {
            case ZS_U8: case ZS_E4M3: return 1;
            case ZS_BF16: case ZS_F16: case ZS_U16: return 2;
            case ZS_I64: return 8;
            default: return 4;
        }
    }
};

cudaStream_t stream();
const cudaDeviceProp& device();

// Records this call's first failure; begin() clears it and status() is what the launcher returns.
void begin();
void fail(const char* message);
void fail_cuda(cudaError_t code, const char* what);
int status();

}  // namespace zs

#define ZS_CUDA_CHECK(expr)                                         \
    do {                                                            \
        const cudaError_t zs_code = (expr);                         \
        if (zs_code != cudaSuccess) zs::fail_cuda(zs_code, #expr);  \
    } while (0)
#define ZS_CUDA_LAUNCH_CHECK() ZS_CUDA_CHECK(cudaGetLastError())
// In a launcher's entry point: refuse bad arguments before anything is launched.
#define ZS_REQUIRE(cond, message)       \
    do {                                \
        if (!(cond)) {                  \
            zs::fail(message);          \
            return zs::status();        \
        }                               \
    } while (0)
