// From TensorFold 0.6.6 (cb2ebf0) families/glm5_next/cuda/kda.cu and kda.cpp, Apache-2.0; changed for zspark: ATen
// and c10 replaced by zs/host.h, a C ABI over zs_view, and the four bindings by two entries. The device code is
// upstream's but its wide chain (kda_prep, step and out kernels), which windows of more than 48 rows leave for
// kda_chunk.cu (spec 116 WS-06). Since spec 121 it is in kda_common.cuh, which kda_quads.cu compiles too: its four
// loops over a thread's state read and write through `load4` and `store4`, here upstream's loops (the same
// instructions, cuobjdump -sass), and `bf`, `warp_sum` and `sigmoidf_` are common/math.cuh's, the same text.
// Compile with --fmad=false, as upstream does.
// GLM-5.3-Flash's KDA (Kimi delta attention) on CUDA: one block of 1024 threads per head, a chain of R rows.
//
// Per row, following the Hugging Face definition: the depthwise conv over [conv state; q|k|v rows] (4 taps,
// fp32) with SiLU and one bf16 rounding; fp32 L2 norms of q and k (eps inside the sum, q times DK^-0.5); the
// per-channel decay g_i = exp(lower * sigmoid(exp(A_log) * (a_i + dt_bias_i))) in fp32; beta = bf16(sigmoid(b));
// the delta rule in fp32 (decay along the key channel, read with k, correct toward v, read out with q);
// read-out rounded to bf16; then the gated RMSNorm bf16(w * (y * rsqrt(mean(y^2) + eps)) * sigmoid(gate)).
// Warp w owns value rows 4w .. 4w + 3, lane l the key columns 4l .. 4l + 3. The state update is one routine
// (``update4``) shared with ``replay``, compiled without FMA contraction, so a replayed prefix of a window gives
// the bits of the serial steps.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../zs/host.h"

namespace {

// A thread's four state values of a row, as upstream's loops read and write them: four 4-byte accesses.
__device__ __forceinline__ void load4(float (&row)[4], const float* p) {
#pragma unroll
    for (int i = 0; i < 4; ++i) row[i] = p[i];
}
__device__ __forceinline__ void store4(float* p, const float (&row)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = row[i];
}

}  // namespace

#define KDA_NAMESPACE
#include "kda_common.cuh"
#undef KDA_NAMESPACE

namespace {

template <typename T>
T* ptr(const zs::View& x) { return x.defined() && x.numel() ? x.data_ptr<T>() : nullptr; }

bool holds(const zs::View& x, int dtype) { return x.defined() && x.is_contiguous() && x.dtype() == dtype; }

// bf16 rows that may be strided: [rows or more, width or more]
bool rows_of(const zs::View& x, int64_t rows, int64_t width) {
    return x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.size(0) >= rows && x.size(1) >= width;
}

}  // namespace

// kda_quads.cu's chain and replay: this file's kernels with the state moved four floats at a time, the same
// bits. Launched when `zs_glm5_kda_quads` switched them on and the states are 16-byte aligned.
namespace glm5_kda_quads {
__global__ void chain_kernel(int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
                             const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ G,
                             int g_stride, const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
                             const float* __restrict__ state_in, const float* __restrict__ a_log,
                             const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ norm_w, float eps,
                             float lower, int rows, __nv_bfloat16* __restrict__ out, float* __restrict__ state_out,
                             float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save,
                             float* __restrict__ g_save, float* __restrict__ b_save);
__global__ void replay_layers_kernel(int H, const float* __restrict__ state_in, size_t state_stride,
                                     const float* __restrict__ k_save, const __nv_bfloat16* __restrict__ v_save,
                                     const float* __restrict__ g_save, const float* __restrict__ b_save,
                                     size_t kv_stride, size_t b_stride, int rows, float* __restrict__ state_out);
}  // namespace glm5_kda_quads

namespace {
bool quads = false;
bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
}  // namespace

extern "C" {

void zs_glm5_kda_quads(int on) { quads = on != 0; }

// A chain of ``rows`` rows from ``state_in``, H heads (a rank's): p bf16 [rows, q | k | v | ... | b at b_off],
// a and g bf16 [rows, H * 128] (the decay's and the output gate's rows), conv state bf16 [3, 3 H 128], conv
// weight bf16 [3 H 128, 4], state fp32 [H, 128, 128] -> out bf16 [rows, H * 128], the state after the last row
// (or null), and what a replay of each row needs (k, g fp32 [rows, H, 128], v bf16 [rows, H, 128], beta fp32
// [rows, H]; or null). Upstream's chain_wide (three kernels from 64 rows, the same bits) is not taken: windows
// of more than 48 rows run chunked (kda_chunk.cu, spec 116 WS-06).
int zs_glm5_kda_chain(const zs_view* p_, int64_t b_off, const zs_view* a_, const zs_view* g_, const zs_view* cs_,
                      const zs_view* cw_, const zs_view* state_in_, const zs_view* a_log_, const zs_view* dt_bias_,
                      const zs_view* norm_w_, double eps, double lower, int64_t rows, const zs_view* out_,
                      const zs_view* state_out_, const zs_view* k_save_, const zs_view* v_save_,
                      const zs_view* g_save_, const zs_view* b_save_) {
    zs::begin();
    const zs::View P(p_), A(a_), G(g_), cs(cs_), cw(cw_), state_in(state_in_), a_log(a_log_), dt_bias(dt_bias_);
    const zs::View norm_w(norm_w_), out(out_), state_out(state_out_), k_save(k_save_), v_save(v_save_);
    const zs::View g_save(g_save_), b_save(b_save_);
    ZS_REQUIRE(holds(a_log, ZS_F32) && a_log.numel() >= 1 && a_log.numel() <= 1024, "A_log: fp32 [H]");
    const int64_t H = a_log.numel(), C = 3 * H * DK;
    ZS_REQUIRE(rows >= 1 && rows <= INT32_MAX / (H * DK), "rows");
    ZS_REQUIRE(b_off >= C && rows_of(P, rows, b_off + H), "p: bf16 [rows, q | k | v | ... | b]");
    ZS_REQUIRE(rows_of(A, rows, H * DK) && rows_of(G, rows, H * DV), "a and g: bf16 [rows, H * 128]");
    ZS_REQUIRE(P.stride(0) <= INT32_MAX && A.stride(0) <= INT32_MAX && G.stride(0) <= INT32_MAX, "row strides");
    ZS_REQUIRE(holds(cs, ZS_BF16) && cs.numel() == (TAPS - 1) * C && holds(cw, ZS_BF16) && cw.numel() == C * TAPS,
               "conv state [3, 3 H 128] and weight [3 H 128, 4]: contiguous bf16");
    ZS_REQUIRE(holds(state_in, ZS_F32) && state_in.numel() == H * DV * DK, "state: fp32 [H, 128, 128]");
    ZS_REQUIRE(holds(dt_bias, ZS_F32) && dt_bias.numel() == H * DK && holds(norm_w, ZS_BF16) && norm_w.numel() == DV,
               "dt_bias fp32 [H * 128] and norm bf16 [128]");
    ZS_REQUIRE(holds(out, ZS_BF16) && out.numel() >= rows * H * DV, "out: bf16 [rows, H * 128]");
    ZS_REQUIRE(!state_out.defined() || (holds(state_out, ZS_F32) && state_out.numel() == H * DV * DK),
               "state out: fp32 [H, 128, 128]");
    const bool saves = k_save.defined();
    ZS_REQUIRE(!saves || (holds(k_save, ZS_F32) && k_save.numel() >= rows * H * DK && holds(v_save, ZS_BF16) &&
                          v_save.numel() >= rows * H * DV && holds(g_save, ZS_F32) &&
                          g_save.numel() >= rows * H * DK && holds(b_save, ZS_F32) && b_save.numel() >= rows * H),
               "replay scratch: k and g fp32 [rows, H, 128], v bf16 [rows, H, 128], beta fp32 [rows, H]");
    auto stream = zs::stream();
    {
        const bool fours = quads && aligned16(ptr<float>(state_in)) && aligned16(ptr<float>(state_out));
        (fours ? glm5_kda_quads::chain_kernel : chain_kernel)<<<(unsigned)H, 1024, 0, stream>>>(
            (int)H, ptr<__nv_bfloat16>(P), (int)P.stride(0), (int)b_off, ptr<__nv_bfloat16>(A), (int)A.stride(0),
            ptr<__nv_bfloat16>(G), (int)G.stride(0), ptr<__nv_bfloat16>(cs), ptr<__nv_bfloat16>(cw),
            ptr<float>(state_in), ptr<float>(a_log), ptr<float>(dt_bias), ptr<__nv_bfloat16>(norm_w), (float)eps,
            (float)lower, (int)rows, ptr<__nv_bfloat16>(out), ptr<float>(state_out), ptr<float>(k_save),
            ptr<__nv_bfloat16>(v_save), ptr<float>(g_save), ptr<float>(b_save));
        ZS_CUDA_LAUNCH_CHECK();
    }
    return zs::status();
}

// The states after the first ``rows`` rows of the last chain, from the states it started at, for ``layers``
// layers in one launch: layer l's states fp32 [H, 128, 128] at l * state_stride elements of ``state_in`` and
// ``state_out``, its saved k, v and g at l * kv_stride and its beta at l * b_stride.
int zs_glm5_kda_replay_layers(const zs_view* state_in_, int64_t state_stride, const zs_view* k_save_,
                              const zs_view* v_save_, const zs_view* g_save_, const zs_view* b_save_,
                              int64_t kv_stride, int64_t b_stride, int64_t layers, int64_t heads, int64_t rows,
                              const zs_view* state_out_) {
    zs::begin();
    const zs::View state_in(state_in_), k_save(k_save_), v_save(v_save_), g_save(g_save_), b_save(b_save_);
    const zs::View state_out(state_out_);
    ZS_REQUIRE(layers >= 1 && heads >= 1 && rows >= 1 && layers * heads <= INT32_MAX, "layers, heads and rows");
    ZS_REQUIRE(state_stride >= heads * DV * DK && kv_stride >= rows * heads * DK && b_stride >= rows * heads,
               "a layer's strides hold its state and its saved rows");
    ZS_REQUIRE(holds(state_in, ZS_F32) && holds(state_out, ZS_F32) &&
               state_in.numel() >= (layers - 1) * state_stride + heads * DV * DK &&
               state_out.numel() >= (layers - 1) * state_stride + heads * DV * DK, "states: fp32, a layer a stride");
    ZS_REQUIRE(holds(k_save, ZS_F32) && holds(v_save, ZS_BF16) && holds(g_save, ZS_F32) && holds(b_save, ZS_F32) &&
               k_save.numel() >= (layers - 1) * kv_stride + rows * heads * DK &&
               v_save.numel() >= (layers - 1) * kv_stride + rows * heads * DV &&
               g_save.numel() >= (layers - 1) * kv_stride + rows * heads * DK &&
               b_save.numel() >= (layers - 1) * b_stride + rows * heads, "replay scratch of every layer");
    const bool fours = quads && state_stride % 4 == 0 && aligned16(ptr<float>(state_in)) && aligned16(ptr<float>(state_out));
    (fours ? glm5_kda_quads::replay_layers_kernel : replay_layers_kernel)<<<(unsigned)(layers * heads), 1024, 0, zs::stream()>>>(
        (int)heads, ptr<float>(state_in), (size_t)state_stride, ptr<float>(k_save), ptr<__nv_bfloat16>(v_save),
        ptr<float>(g_save), ptr<float>(b_save), (size_t)kv_stride, (size_t)b_stride, (int)rows,
        ptr<float>(state_out));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
