// Spec 018 WS-B3, the pool's (`--parallel` N): ple.cu's convolution for several streams in one launch. A row
// reads its own stream's tail for the taps before its segment, and its own segment's earlier rows inside it,
// so a stream's bits do not depend on the rows of another. sid int32 [rows] is the row's stream. seg int32
// [slots, 4] is first row, rows, kept, first position. The tail pointer is ptrs int64 [slots, stride] at col.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

__device__ __forceinline__ float conv_normed(const __nv_bfloat16* __restrict__ gated, const float* __restrict__ pss,
                                             const float* __restrict__ nc, float eps, int row, int e, int D, int S) {
    const float rinv = 1.0f / sqrtf(pss[(size_t)row * S + e / D] / D + eps);
    return bf(__bfloat162float(gated[(size_t)row * S * D + e]) * rinv * nc[e]);
}

// ple.cu's conv_kernel, a row at a time across the round. local = g - a0 is the row inside its segment.
__global__ void conv_slots_kernel(const __nv_bfloat16* __restrict__ gated, const float* __restrict__ pss,
                                  const float* __restrict__ nc, const __nv_bfloat16* __restrict__ cw,
                                  __nv_bfloat16* __restrict__ H, __nv_bfloat16* __restrict__ nrow,
                                  const int* __restrict__ SID, const int* __restrict__ SEG,
                                  const int64_t* __restrict__ PTRS, int stride, int col, float eps, int R, int D,
                                  int S, int TAPS, int DIL, int SLOTS) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int W = S * D;
    if (i >= (long long)R * W) return;
    const int g = static_cast<int>(i / W), e = static_cast<int>(i % W), NT = (TAPS - 1) * DIL;
    const int slot = SID[g];
    // the slot tables are the host's (spec 103): a row whose slot or segment would read past them, or past the
    // round's rows, is left as it was. A row inside keeps its bits: nothing below changes for it.
    if ((unsigned)slot >= (unsigned)SLOTS) return;
    const int a0 = SEG[slot * 4];
    const int local = g - a0;
    if (a0 < 0 || local < 0 || local >= SEG[slot * 4 + 1] || a0 + SEG[slot * 4 + 1] > R) return;
    const __nv_bfloat16* tail = reinterpret_cast<const __nv_bfloat16*>(PTRS[static_cast<size_t>(slot) * stride + col]);
    const float own = conv_normed(gated, pss, nc, eps, g, e, D, S);
    float acc = 0.f;
    for (int t = 0; t < TAPS; ++t) {
        const int at = local + t * DIL;
        const float xv = at < NT ? __bfloat162float(tail[(size_t)at * W + e])
                                 : (at - NT == local ? own
                                                     : conv_normed(gated, pss, nc, eps, a0 + (at - NT), e, D, S));
        acc += __bfloat162float(cw[(size_t)e * TAPS + t]) * xv;
    }
    const float ple = bf(__bfloat162float(gated[i]) + bsilu(bf(acc)));
    H[i] = __float2bfloat16_rn(__bfloat162float(H[i]) + ple);
    nrow[i] = __float2bfloat16_rn(own);
}

}  // namespace

extern "C" int zs_qwen4_ple_conv_slots(const zs_view* gated_, const zs_view* pss_, const zs_view* norm_conv_,
                                       const zs_view* conv_, const zs_view* h_, const zs_view* nrow_,
                                       const zs_view* sid_, const zs_view* seg_, const zs_view* ptrs_,
                                       int64_t stride, int64_t col, double eps, int64_t streams, int64_t dilation) {
    zs::begin();
    const zs::View gated(gated_), pss(pss_), nc(norm_conv_), conv(conv_), h(h_), nrow(nrow_);
    const zs::View sid(sid_), seg(seg_), ptrs(ptrs_);
    ZS_REQUIRE(h.defined() && h.dtype() == ZS_BF16 && h.dim() == 2 && h.is_contiguous() && streams >= 1 &&
               h.size(1) % streams == 0, "h: bf16 [R, S * D]");
    const int64_t rows = h.size(0), wide = h.size(1), d = wide / streams;
    ZS_REQUIRE(gated.defined() && gated.dtype() == ZS_BF16 && gated.is_contiguous() && gated.size(0) == rows &&
               gated.size(1) == wide, "gated: bf16 [R, S * D]");
    ZS_REQUIRE(pss.defined() && pss.dtype() == ZS_F32 && pss.is_contiguous() && pss.size(0) == rows &&
               pss.size(1) == streams, "pss: fp32 [R, S]");
    ZS_REQUIRE(nc.defined() && nc.dtype() == ZS_F32 && nc.numel() == wide, "norm_conv: fp32 [S * D]");
    ZS_REQUIRE(conv.defined() && conv.dtype() == ZS_BF16 && conv.dim() == 2 && conv.is_contiguous() &&
               conv.size(0) == wide && conv.size(1) >= 1, "conv: bf16 [S * D, taps]");
    const int64_t taps = conv.size(1);
    ZS_REQUIRE(dilation >= 1, "dilation: 1 or more");
    ZS_REQUIRE(nrow.defined() && nrow.dtype() == ZS_BF16 && nrow.is_contiguous() && nrow.size(0) == rows &&
               nrow.size(1) == wide, "nrow: bf16 [R, S * D]");
    ZS_REQUIRE(sid.defined() && sid.dtype() == ZS_I32 && sid.is_contiguous() && sid.numel() == rows, "sid: int32 [R]");
    ZS_REQUIRE(seg.defined() && seg.dtype() == ZS_I32 && seg.is_contiguous() && seg.dim() == 2 && seg.size(1) == 4,
               "seg: int32 [slots, 4]");
    ZS_REQUIRE(ptrs.defined() && ptrs.dtype() == ZS_I64 && ptrs.is_contiguous() && ptrs.dim() == 2 &&
               ptrs.size(0) == seg.size(0) && ptrs.size(1) == stride && col >= 0 && col < stride,
               "ptrs: int64 [slots, stride]");
    const unsigned blocks = static_cast<unsigned>((rows * wide + 255) / 256);
    conv_slots_kernel<<<blocks, 256, 0, zs::stream()>>>(
        gated.data_ptr<__nv_bfloat16>(), pss.data_ptr<float>(), nc.data_ptr<float>(), conv.data_ptr<__nv_bfloat16>(),
        h.data_ptr<__nv_bfloat16>(), nrow.data_ptr<__nv_bfloat16>(), sid.data_ptr<int>(), seg.data_ptr<int>(),
        ptrs.data_ptr<int64_t>(), static_cast<int>(stride), static_cast<int>(col), static_cast<float>(eps),
        static_cast<int>(rows), static_cast<int>(d), static_cast<int>(streams), static_cast<int>(taps),
        static_cast<int>(dilation), static_cast<int>(seg.size(0)));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}
