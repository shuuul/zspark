// Nemotron's router: fp32 logits in K slices (one fma chain a slice), then each row's top-k and the shared slots.
// From TensorFold 1.1.5 (792cdbab, unchanged at 1b31cc74) zig/kernels/cuda/nemotron_route.cu, Apache-2.0;
// changed for zspark: the zs launchers at the file's end. The kernels' device code is upstream's byte for byte.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "../zs/host.h"
#include "glue_math.cuh"

using namespace tf_glue;

// Block (16-row tile, 16-expert block, K slice s): PART[s][row][e] = x[row] . w[e] over the slice, in index order.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_router(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                   float* __restrict__ PART, int R, int D, int E, int SK) {
    extern __shared__ uint32_t tile[];  // 16 input rows, then 16 expert rows, each per / 2 words plus one of padding
    const int per = D / SK, words = per / 2, stride = words + 1;
    const int row0 = blockIdx.x * 16, e0 = blockIdx.y * 16, k0 = blockIdx.z * per;
    uint32_t* xs = tile;
    uint32_t* ws = tile + 16 * stride;
    for (int c = threadIdx.x; c < 16 * words; c += 256) {
        const int r = c / words, w = c % words;
        const uint32_t* xr = reinterpret_cast<const uint32_t*>(X + static_cast<int64_t>(row0 + r) * D + k0);
        const uint32_t* wr = reinterpret_cast<const uint32_t*>(W + static_cast<int64_t>(e0 + r) * D + k0);
        xs[r * stride + w] = row0 + r < R ? xr[w] : 0u;
        ws[r * stride + w] = e0 + r < E ? wr[w] : 0u;
    }
    __syncthreads();
    const int rr = threadIdx.x / 16, ee = threadIdx.x % 16;
    const uint32_t* xp = xs + rr * stride;
    const uint32_t* wp = ws + ee * stride;
    float acc = 0.0f;
    for (int w = 0; w < words; ++w) {
        const uint32_t a = xp[w], b = wp[w];
        acc = __fmaf_rn(__uint_as_float(a << 16), __uint_as_float(b << 16), acc);
        acc = __fmaf_rn(__uint_as_float(a & 0xFFFF0000u), __uint_as_float(b & 0xFFFF0000u), acc);
    }
    if (row0 + rr < R && e0 + ee < E) PART[(static_cast<int64_t>(blockIdx.z) * R + row0 + rr) * E + e0 + ee] = acc;
}

namespace {

constexpr int MAXC = 8;  // experts a lane holds: up to 256 experts

// The larger of two (value, index) picks: a higher value, or the lower index on a tie (values are never NaN).
__device__ __forceinline__ bool beats(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

}  // namespace

// Sum logits' slices in order; rank sigmoid scores plus bias before adding the shared slots.
extern "C" __global__ void __launch_bounds__(32) tf_nemo_topk(const float* __restrict__ PART, const float* __restrict__ BIAS,
                                                               int* __restrict__ IDX, float* __restrict__ WT, int R, float scaling, int E,
                                                               int SK, int TOPK, int NS, int norm) {
    const int64_t r = blockIdx.x;
    const int lane = threadIdx.x;
    float score[MAXC], sel[MAXC];
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int e = lane + 32 * c;
        score[c] = 0.0f;
        sel[c] = -INFINITY;
        if (e >= E) continue;
        float logit = 0.0f;
        for (int s = 0; s < SK; ++s) logit = __fadd_rn(logit, PART[(s * R + r) * E + e]);
        score[c] = tf_sigmoid(logit);
        const float v = __fadd_rn(score[c], BIAS[e]);
        sel[c] = v != v ? -INFINITY : v;  // a NaN logit is never picked ahead of a number
    }
    float total = 0.0f;
    float probs[MAXC];
    int ids[MAXC];
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (lane + 32 * c < E && beats(sel[c], lane + 32 * c, bv, bi)) bv = sel[c], bi = lane + 32 * c;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (beats(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        float p = 0.0f;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (bi == lane + 32 * c) p = score[c], sel[c] = -INFINITY;
        p = __shfl_sync(0xffffffffu, p, bi % 32);
        if (k < MAXC) probs[k] = p, ids[k] = bi;
        total = __fadd_rn(total, p);
    }
    if (lane != 0) return;
    for (int k = 0; k < NS; ++k) {
        float w = 1.0f;
        int id = E + (k - TOPK);
        if (k < TOPK) {
            id = ids[k];
            w = norm ? __fmul_rn(__fdiv_rn(probs[k], __fadd_rn(total, 1e-20f)), scaling) : __fmul_rn(probs[k], scaling);
        }
        IDX[r * NS + k] = id;
        WT[r * NS + k] = w;
    }
}

extern "C" {

// x bf16 [R, D], w bf16 [E, D] -> part fp32 [SK, R, E]: the router's logits in SK K-slices, one staged 16x16
// tile a (row tile, expert block, slice). SK divides D / 64, so a slice is a whole number of 64-groups; the
// caller picks it by glue.routerSlices (the largest divisor of D / 64 up to 6).
int zs_nemotron_router(const zs_view* x_, const zs_view* w_, const zs_view* part_) {
    zs::begin();
    const zs::View x(x_), w(w_), part(part_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous(), "x: bf16 [R, D]");
    const int64_t rows = x.size(0), d = x.size(1);
    ZS_REQUIRE(d % 64 == 0, "D: a multiple of 64");
    ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.dim() == 2 && w.is_contiguous() && w.size(1) == d,
               "w: bf16 [E, D]");
    const int64_t e = w.size(0);
    ZS_REQUIRE(part.defined() && part.dtype() == ZS_F32 && part.dim() == 3 && part.is_contiguous() &&
                   part.size(1) == rows && part.size(2) == e,
               "part: fp32 [SK, R, E]");
    const int64_t sk = part.size(0);
    ZS_REQUIRE(sk >= 1 && (d / 64) % sk == 0, "SK: divides D / 64");
    ZS_REQUIRE(rows <= INT32_MAX && e <= INT32_MAX, "R, E: int32");
    // the tiles' staging reads a row slice as words of two bf16
    ZS_REQUIRE(reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 4 == 0,
               "x, w: 4-byte aligned");
    const int64_t per = d / sk, words = per / 2;
    const unsigned shared = static_cast<unsigned>(2 * 16 * (words + 1) * 4);
    if (rows > 0 && e > 0)
        tf_nemo_router<<<dim3((static_cast<unsigned>(rows) + 15) / 16, (static_cast<unsigned>(e) + 15) / 16,
                              static_cast<unsigned>(sk)),
                         256, shared, zs::stream()>>>(x.data_ptr<__nv_bfloat16>(), w.data_ptr<__nv_bfloat16>(),
                                                      part.data_ptr<float>(), static_cast<int>(rows),
                                                      static_cast<int>(d), static_cast<int>(e),
                                                      static_cast<int>(sk));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// part fp32 [SK, R, E] (zs_nemotron_router's slices, or any caller's), bias fp32 [E] -> idx int32 and wt fp32
// [R, NS]: a warp a row sums the slices in order, takes the TOPK highest sigmoid scores plus bias (the lower id
// on a tie), weights prob / (sum + 1e-20) * scaling when `norm`, and appends NS - TOPK shared slots (ids E..,
// weight 1). TOPK and NS stay within the lanes' registers: both no more than 8.
int zs_nemotron_topk(const zs_view* part_, const zs_view* bias_, const zs_view* idx_, const zs_view* wt_,
                     double scaling, int64_t topk, int64_t ns, int64_t norm) {
    zs::begin();
    const zs::View part(part_), bias(bias_), idx(idx_), wt(wt_);
    ZS_REQUIRE(part.defined() && part.dtype() == ZS_F32 && part.dim() == 3 && part.is_contiguous(),
               "part: fp32 [SK, R, E]");
    const int64_t sk = part.size(0), rows = part.size(1), e = part.size(2);
    ZS_REQUIRE(e >= 1 && e <= 32 * 8, "E: up to 256 experts");
    ZS_REQUIRE(topk >= 1 && topk <= 8 && ns >= topk && ns <= 8, "topk, ns: 1 to 8 slots (ns = topk + shared)");
    ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_F32 && bias.numel() >= e, "bias: fp32 [E]");
    ZS_REQUIRE(idx.defined() && idx.dtype() == ZS_I32 && idx.is_contiguous() && idx.numel() >= rows * ns,
               "idx: int32 [R, NS]");
    ZS_REQUIRE(wt.defined() && wt.dtype() == ZS_F32 && wt.is_contiguous() && wt.numel() >= rows * ns,
               "wt: fp32 [R, NS]");
    ZS_REQUIRE(rows <= INT32_MAX, "R: int32");
    if (rows > 0)
        tf_nemo_topk<<<static_cast<unsigned>(rows), 32, 0, zs::stream()>>>(
            part.data_ptr<float>(), bias.data_ptr<float>(), idx.data_ptr<int>(), wt.data_ptr<float>(),
            static_cast<int>(rows), static_cast<float>(scaling), static_cast<int>(e), static_cast<int>(sk),
            static_cast<int>(topk), static_cast<int>(ns), norm ? 1 : 0);
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
