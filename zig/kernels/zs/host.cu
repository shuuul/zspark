// This thread's stream and failure record, and the device's properties.

#include <cstdio>

#include "host.h"

namespace {

thread_local cudaStream_t current_stream = nullptr;
thread_local char message[512] = "";
thread_local bool failed = false;

}  // namespace

namespace zs {

cudaStream_t stream() { return current_stream; }

const cudaDeviceProp& device() {
    static cudaDeviceProp prop;
    static bool known = false;
    if (!known) {
        int id = 0;
        ZS_CUDA_CHECK(cudaGetDevice(&id));
        ZS_CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        known = true;
    }
    return prop;
}

void begin() {
    failed = false;
    message[0] = 0;
}

void fail(const char* text) {
    if (failed) return;
    failed = true;
    snprintf(message, sizeof(message), "%s", text);
}

void fail_cuda(cudaError_t code, const char* what) {
    if (failed) return;
    failed = true;
    snprintf(message, sizeof(message), "%s: %s", what, cudaGetErrorString(code));
}

int status() { return failed ? 1 : 0; }

}  // namespace zs

extern "C" const char* zs_last_error(void) { return message; }
extern "C" void zs_set_stream(void* stream) { current_stream = static_cast<cudaStream_t>(stream); }
