// spec 151 WS-06: the two-ranks Nemotron backbone's own combine — one rank's MoE slots folded into its
// fp32 partial, which `common/rank_sum.cu` then sums rank 0 first (the serial `addMoeNorm` order: the
// first `nr` slots weighted by the gate, the rest — the shared experts' rows — added unweighted).

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"
#include "glue_math.cuh"

namespace {
using tf_glue::bf;

// A rank's share of a MoE layer's output, for the ranks' sum (TensorFold `tp.py`'s `_moe_partial`):
// out = sum_s<NR fma(wt_s, y_s) + sum_s>=NR y_s, in slot order, fp32 and unrounded — the rank's combine
// partial, gathered and added rank 0 first (`common/rank_sum.cu`), rounded once where the residual
// takes it.
extern "C" __global__ void tf_nemo_moe_partial(const void* __restrict__ Yk, int y_f32,
                                               const float* __restrict__ WT, float* __restrict__ OUT,
                                               int D, int NR, int NS) {
    const int64_t row = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float routed = 0.0f, shared = 0.0f;
        for (int s = 0; s < NS; ++s) {
            const int64_t at = (row * NS + s) * D + i;
            const float y = y_f32 ? static_cast<const float*>(Yk)[at] : bf(static_cast<const __nv_bfloat16*>(Yk)[at]);
            if (s < NR) routed = __fmaf_rn(WT[row * NS + s], y, routed);
            else shared = __fadd_rn(shared, y);
        }
        OUT[row * D + i] = __fadd_rn(routed, shared);
    }
}

}  // namespace

extern "C" {

// yk fp32 or bf16 [R, NS, D] and wt fp32 [R, NS] -> out fp32 [R, D]: this rank's MoE slots combined in
// slot order (the first `nr` routed slots weighted, the rest added), for the ranks' partial sum.
int zs_nemotron_moe_partial(const zs_view* yk_, const zs_view* wt_, const zs_view* out_, int64_t nr) {
    zs::begin();
    const zs::View yk(yk_), wt(wt_), out(out_);
    ZS_REQUIRE(yk.defined() && (yk.dtype() == ZS_F32 || yk.dtype() == ZS_BF16) && yk.dim() == 3 &&
                   yk.is_contiguous(),
               "yk: fp32 or bf16 [R, NS, D]");
    const int64_t rows = yk.size(0), ns = yk.size(1), d = yk.size(2);
    ZS_REQUIRE(d % 64 == 0 && d <= 8192, "D: a multiple of 64");
    ZS_REQUIRE(nr >= 0 && nr <= ns, "nr: the routed slots, 0 to NS");
    ZS_REQUIRE(wt.defined() && wt.dtype() == ZS_F32 && wt.is_contiguous() && wt.numel() >= rows * ns,
               "wt: fp32 [R, NS]");
    ZS_REQUIRE(out.defined() && out.dtype() == ZS_F32 && out.is_contiguous() && out.numel() >= rows * d,
               "out: fp32 [R, D]");
    if (rows > 0)
        tf_nemo_moe_partial<<<static_cast<unsigned>(rows), 256, 0, zs::stream()>>>(
            yk.data_ptr(), yk.dtype() == ZS_F32 ? 1 : 0, wt.data_ptr<float>(), out.data_ptr<float>(),
            static_cast<int>(d), static_cast<int>(nr), static_cast<int>(ns));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
