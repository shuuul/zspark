// Nemotron's token rows and RMSNorms: one block a row, fixed-order fp32 sums, so a row's bits ignore its window.
// From TensorFold 1.1.5 (792cdbab, unchanged at 1b31cc74) zig/kernels/cuda/nemotron_norms.cu, Apache-2.0;
// changed for zspark: the zs launchers at the file's end, and tf_nemo_embed dropped — it dequantizes MLX
// affine 4-bit token rows, a storage the NVFP4 checkpoint does not have (the embeddings are BF16, zs_embed
// gathers them). The kept kernels' device code is upstream's byte for byte, XS outputs included: the fp32
// 64-group sums fed TensorFold's 4-bit matmuls, which zspark's NVFP4/FP8 linears never read — the caller
// lends them scratch.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"
#include "glue_math.cuh"

namespace tf_glue {

constexpr int NT = 256;   // threads a row block
constexpr int MAXE = 16;  // elements a thread holds: rows up to NT * MAXE = 4096 wide

// Sum of x^2 over a row: each thread's fma chain in index order, warp butterflies, then warps in order.
__device__ __forceinline__ float row_ss(const float (&x)[MAXE], int D, float* red) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXE; ++k)
        if (threadIdx.x + k * NT < D) acc = __fmaf_rn(x[k], x[k], acc);
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    float ss = red[0];
    for (int w = 1; w < NT / 32; ++w) ss = __fadd_rn(ss, red[w]);
    return ss;
}

// 1 / sqrt(ss / n + eps) with IEEE division and square root.
__device__ __forceinline__ float inv_rms(float ss, int n, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(n)), eps)));
}

// y = bf16((x * inv) * w) into OUT, and each 64-group's sum of the stored values, in order, into XS.
__device__ __forceinline__ void scale_store(const float (&x)[MAXE], int D, float inv, const __nv_bfloat16* __restrict__ W,
                                            __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, float* ybuf) {
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        const __nv_bfloat16 y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[k], inv), bf(W[i])));
        OUT[i] = y;
        ybuf[i] = bf(y);
    }
    __syncthreads();
    for (int g = threadIdx.x; g < D / 64; g += NT) {
        float s = 0.0f;
        for (int j = 0; j < 64; ++j) s = __fadd_rn(s, ybuf[g * 64 + j]);
        XS[g] = s;
    }
}

}  // namespace tf_glue

using namespace tf_glue;

// h = bf16(x + r) (or x alone without r), y = bf16(rmsnorm(h) * w), xs = y's 64-group sums.
extern "C" __global__ void __launch_bounds__(NT) tf_nemo_add_rmsnorm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ R,
                                                                       const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ H,
                                                                       __nv_bfloat16* __restrict__ Y, float* __restrict__ XS, float eps, int D) {
    __shared__ float red[NT / 32];
    __shared__ float ybuf[NT * MAXE];
    const int64_t row = blockIdx.x;
    float x[MAXE];
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        float v = bf(X[row * D + i]);
        if (R != nullptr) {
            const __nv_bfloat16 h = __float2bfloat16_rn(__fadd_rn(v, bf(R[row * D + i])));
            H[row * D + i] = h;
            v = bf(h);
        }
        x[k] = v;
    }
    const float inv = inv_rms(row_ss(x, D, red), D, eps);
    scale_store(x, D, inv, W, Y + row * D, XS + row * (D / 64), ybuf);
}

// delta = bf16(sum_k<NR fma(wt_k, y_k) + sum_k>=NR y_k) in slot order, h = bf16(x + delta), then add_rmsnorm's tail.
extern "C" __global__ void __launch_bounds__(NT) tf_nemo_add_moe_norm(const __nv_bfloat16* __restrict__ X, const void* __restrict__ Yk,
                                                                        int y_f32, const float* __restrict__ WT, const __nv_bfloat16* __restrict__ W,
                                                                        __nv_bfloat16* __restrict__ HN, __nv_bfloat16* __restrict__ Y,
                                                                        float* __restrict__ XS, float eps, int D, int NR, int NS) {
    __shared__ float red[NT / 32];
    __shared__ float ybuf[NT * MAXE];
    const int64_t row = blockIdx.x;
    float x[MAXE];
#pragma unroll
    for (int k = 0; k < MAXE; ++k) {
        const int i = threadIdx.x + k * NT;
        if (i >= D) continue;
        float routed = 0.0f, shared = 0.0f;
        for (int s = 0; s < NS; ++s) {
            const int64_t at = (row * NS + s) * D + i;
            const float y = y_f32 ? static_cast<const float*>(Yk)[at] : bf(static_cast<const __nv_bfloat16*>(Yk)[at]);
            if (s < NR) routed = __fmaf_rn(WT[row * NS + s], y, routed);
            else shared = __fadd_rn(shared, y);
        }
        const float delta = rbf(__fadd_rn(routed, shared));
        const __nv_bfloat16 h = __float2bfloat16_rn(__fadd_rn(bf(X[row * D + i]), delta));
        HN[row * D + i] = h;
        x[k] = bf(h);
    }
    const float inv = inv_rms(row_ss(x, D, red), D, eps);
    scale_store(x, D, inv, W, Y + row * D, XS + row * (D / 64), ybuf);
}

// MTP (row, part) blocks normalize embedding and hidden halves side by side, with group sums.
extern "C" __global__ void __launch_bounds__(NT) tf_nemo_concat_norms(const __nv_bfloat16* __restrict__ E, const __nv_bfloat16* __restrict__ Hd,
                                                                        const __nv_bfloat16* __restrict__ WE, const __nv_bfloat16* __restrict__ WH,
                                                                        __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, float eps, int D) {
    __shared__ float red[NT / 32];
    __shared__ float ybuf[NT * MAXE];
    const int64_t row = blockIdx.x;
    const int part = blockIdx.y;
    const __nv_bfloat16* src = (part == 0 ? E : Hd) + row * D;
    float x[MAXE];
#pragma unroll
    for (int k = 0; k < MAXE; ++k)
        if (threadIdx.x + k * NT < D) x[k] = bf(src[threadIdx.x + k * NT]);
    const float inv = inv_rms(row_ss(x, D, red), D, eps);
    scale_store(x, D, inv, part == 0 ? WE : WH, OUT + row * 2 * D + part * D, XS + row * (2 * D / 64) + part * (D / 64), ybuf);
}

// Block (row, group) of GS = XD / groups values: n = bf16(v * inv), out = bf16(w * n), and the 64-group sums.
extern "C" __global__ void __launch_bounds__(128) tf_nemo_group_rmsnorm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                          __nv_bfloat16* __restrict__ OUT, float* __restrict__ XS, float eps, int XD,
                                                                          int GS) {
    constexpr int T = 128, M = 8;  // groups up to T * M = 1024 wide
    __shared__ float red[T / 32];
    __shared__ float ybuf[T * M];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * XD + blockIdx.y * GS;
    float v[M];
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < M; ++k) {
        const int i = threadIdx.x + k * T;
        if (i >= GS) continue;
        v[k] = bf(X[at + i]);
        acc = __fmaf_rn(v[k], v[k], acc);
    }
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    float ss = red[0];
    for (int w = 1; w < T / 32; ++w) ss = __fadd_rn(ss, red[w]);
    const float inv = inv_rms(ss, GS, eps);
#pragma unroll
    for (int k = 0; k < M; ++k) {
        const int i = threadIdx.x + k * T;
        if (i >= GS) continue;
        const float n = rbf(__fmul_rn(v[k], inv));
        const __nv_bfloat16 o = __float2bfloat16_rn(__fmul_rn(bf(W[blockIdx.y * GS + i]), n));
        OUT[at + i] = o;
        ybuf[i] = bf(o);
    }
    __syncthreads();
    for (int g = threadIdx.x; g < GS / 64; g += T) {
        float s = 0.0f;
        for (int j = 0; j < 64; ++j) s = __fadd_rn(s, ybuf[g * 64 + j]);
        XS[at / 64 + g] = s;
    }
}

extern "C" {

// x bf16 [R, D], r bf16 [R, D] or null (no residual, h untouched), w bf16 [D] -> h, y bf16 [R, D],
// xs fp32 [R, D / 64] (the row's 64-group sums; nothing reads them yet).
int zs_nemotron_add_rmsnorm(const zs_view* x_, const zs_view* r_, const zs_view* w_, const zs_view* h_,
                            const zs_view* y_, const zs_view* xs_, double eps) {
    zs::begin();
    const zs::View x(x_), r(r_), w(w_), h(h_), y(y_), xs(xs_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous(), "x: bf16 [R, D]");
    const int64_t rows = x.size(0), d = x.size(1);
    ZS_REQUIRE(d % 64 == 0 && d <= NT * MAXE, "D: a multiple of 64 up to 4096");
    if (r.defined())
        ZS_REQUIRE(r.dtype() == ZS_BF16 && r.is_contiguous() && r.numel() >= rows * d, "r: bf16 [R, D]");
    ZS_REQUIRE(!r.defined() || (h.defined() && h.dtype() == ZS_BF16 && h.is_contiguous() && h.numel() >= rows * d),
               "h: bf16 [R, D] with r");
    ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.numel() >= d, "w: bf16 [D]");
    ZS_REQUIRE(y.defined() && y.dtype() == ZS_BF16 && y.is_contiguous() && y.numel() >= rows * d, "y: bf16 [R, D]");
    ZS_REQUIRE(xs.defined() && xs.dtype() == ZS_F32 && xs.is_contiguous() && xs.numel() >= rows * (d / 64),
               "xs: fp32 [R, D / 64]");
    if (rows > 0)
        tf_nemo_add_rmsnorm<<<static_cast<unsigned>(rows), NT, 0, zs::stream()>>>(
            x.data_ptr<__nv_bfloat16>(), r.defined() ? r.data_ptr<__nv_bfloat16>() : nullptr,
            w.data_ptr<__nv_bfloat16>(), h.defined() ? h.data_ptr<__nv_bfloat16>() : nullptr,
            y.data_ptr<__nv_bfloat16>(), xs.data_ptr<float>(), static_cast<float>(eps), static_cast<int>(d));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// x bf16 [R, D], yk [R, NS, D] fp32 or bf16 (the expert slots' rows), wt fp32 [R, NS] -> hn, y bf16 [R, D],
// xs fp32 [R, D / 64]; the first NR slots are the routed experts' (fused by wt), the rest the shared halves'.
int zs_nemotron_add_moe_norm(const zs_view* x_, const zs_view* yk_, const zs_view* wt_, const zs_view* w_,
                             const zs_view* hn_, const zs_view* y_, const zs_view* xs_, double eps,
                             int64_t nr) {
    zs::begin();
    const zs::View x(x_), yk(yk_), wt(wt_), w(w_), hn(hn_), y(y_), xs(xs_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous(), "x: bf16 [R, D]");
    const int64_t rows = x.size(0), d = x.size(1);
    ZS_REQUIRE(d % 64 == 0 && d <= NT * MAXE, "D: a multiple of 64 up to 4096");
    ZS_REQUIRE(yk.defined() && (yk.dtype() == ZS_F32 || yk.dtype() == ZS_BF16) && yk.dim() == 3 &&
                   yk.is_contiguous() && yk.size(0) == rows && yk.size(2) == d,
               "yk: fp32 or bf16 [R, NS, D]");
    const int64_t ns = yk.size(1);
    ZS_REQUIRE(nr >= 0 && nr <= ns, "nr: the routed slots, 0 to NS");
    ZS_REQUIRE(wt.defined() && wt.dtype() == ZS_F32 && wt.is_contiguous() && wt.numel() >= rows * ns,
               "wt: fp32 [R, NS]");
    ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.numel() >= d, "w: bf16 [D]");
    ZS_REQUIRE(hn.defined() && hn.dtype() == ZS_BF16 && hn.is_contiguous() && hn.numel() >= rows * d, "hn: bf16 [R, D]");
    ZS_REQUIRE(y.defined() && y.dtype() == ZS_BF16 && y.is_contiguous() && y.numel() >= rows * d, "y: bf16 [R, D]");
    ZS_REQUIRE(xs.defined() && xs.dtype() == ZS_F32 && xs.is_contiguous() && xs.numel() >= rows * (d / 64),
               "xs: fp32 [R, D / 64]");
    if (rows > 0)
        tf_nemo_add_moe_norm<<<static_cast<unsigned>(rows), NT, 0, zs::stream()>>>(
            x.data_ptr<__nv_bfloat16>(), yk.data_ptr(), yk.dtype() == ZS_F32 ? 1 : 0, wt.data_ptr<float>(),
            w.data_ptr<__nv_bfloat16>(), hn.data_ptr<__nv_bfloat16>(), y.data_ptr<__nv_bfloat16>(),
            xs.data_ptr<float>(), static_cast<float>(eps), static_cast<int>(d), static_cast<int>(nr),
            static_cast<int>(ns));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// e, hd bf16 [R, D], we, wh bf16 [D] -> out bf16 [R, 2 D] ([rmsnorm(e) * we | rmsnorm(hd) * wh]),
// xs fp32 [R, 2 D / 64]: the MTP block's input halves normed side by side.
int zs_nemotron_concat_norms(const zs_view* e_, const zs_view* hd_, const zs_view* we_, const zs_view* wh_,
                             const zs_view* out_, const zs_view* xs_, double eps) {
    zs::begin();
    const zs::View e(e_), hd(hd_), we(we_), wh(wh_), out(out_), xs(xs_);
    ZS_REQUIRE(e.defined() && e.dtype() == ZS_BF16 && e.dim() == 2 && e.is_contiguous(), "e: bf16 [R, D]");
    const int64_t rows = e.size(0), d = e.size(1);
    ZS_REQUIRE(d % 64 == 0 && d <= NT * MAXE, "D: a multiple of 64 up to 4096");
    ZS_REQUIRE(hd.defined() && hd.dtype() == ZS_BF16 && hd.is_contiguous() && hd.numel() >= rows * d, "hd: bf16 [R, D]");
    ZS_REQUIRE(we.defined() && we.dtype() == ZS_BF16 && we.numel() >= d, "we: bf16 [D]");
    ZS_REQUIRE(wh.defined() && wh.dtype() == ZS_BF16 && wh.numel() >= d, "wh: bf16 [D]");
    ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * 2 * d,
               "out: bf16 [R, 2 D]");
    ZS_REQUIRE(xs.defined() && xs.dtype() == ZS_F32 && xs.is_contiguous() && xs.numel() >= rows * (2 * d / 64),
               "xs: fp32 [R, 2 D / 64]");
    if (rows > 0)
        tf_nemo_concat_norms<<<dim3(static_cast<unsigned>(rows), 2), NT, 0, zs::stream()>>>(
            e.data_ptr<__nv_bfloat16>(), hd.data_ptr<__nv_bfloat16>(), we.data_ptr<__nv_bfloat16>(),
            wh.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(), xs.data_ptr<float>(),
            static_cast<float>(eps), static_cast<int>(d));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// x bf16 [R, XD], w bf16 [groups * GS] -> out bf16 [R, XD], xs fp32 [R, XD / 64]: a group RMSNorm (the Mamba
// blocks' 8 groups) of GS = XD / groups values, a block a (row, group), GS a multiple of 64 up to 1024.
int zs_nemotron_group_rmsnorm(const zs_view* x_, const zs_view* w_, const zs_view* out_, const zs_view* xs_,
                              double eps, int64_t groups) {
    zs::begin();
    const zs::View x(x_), w(w_), out(out_), xs(xs_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous(), "x: bf16 [R, XD]");
    const int64_t rows = x.size(0), xd = x.size(1);
    ZS_REQUIRE(groups >= 1 && xd % groups == 0, "groups: divides XD");
    const int64_t gs = xd / groups;
    ZS_REQUIRE(gs % 64 == 0 && gs <= 128 * 8, "GS: a multiple of 64 up to 1024");
    ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.numel() >= xd, "w: bf16 [groups * GS]");
    ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * xd, "out: bf16 [R, XD]");
    ZS_REQUIRE(xs.defined() && xs.dtype() == ZS_F32 && xs.is_contiguous() && xs.numel() >= rows * (xd / 64),
               "xs: fp32 [R, XD / 64]");
    if (rows > 0)
        tf_nemo_group_rmsnorm<<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>(groups)), 128, 0,
                                zs::stream()>>>(x.data_ptr<__nv_bfloat16>(), w.data_ptr<__nv_bfloat16>(),
                                                out.data_ptr<__nv_bfloat16>(), xs.data_ptr<float>(),
                                                static_cast<float>(eps), static_cast<int>(xd),
                                                static_cast<int>(gs));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
