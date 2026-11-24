// From TensorFold 1.1.5 (792cdbab, unchanged at 1b31cc74) zig/kernels/cuda/nemotron_mamba.cu, Apache-2.0;
// changed for zspark: the zs launchers at the file's end (upstream's flags, --fmad=false among them, in
// build.zig), and tf_nemo_conv_rows / tf_nemo_conv_commit dropped — a prompt chunk's conv that does not replay
// the kept rows; zspark's prompt path is mamba_wide.cu, which does, with these kernels' arithmetic. The kept
// kernels' device code is upstream's byte for byte.
// Nemotron's Mamba-2 conv and scan: state lags a window, its kept rows replaying first in the same loop body.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"
#include "glue_math.cuh"

using namespace tf_glue;

// out = bf16(silu(bf16(bias + w0 t0 + w1 t1 + w2 t2 + w3 cur))), the taps in that fma order.
__device__ __forceinline__ __nv_bfloat16 conv4(float bias, const float (&w)[4], float t0, float t1, float t2, float cur) {
    float acc = bias;
    acc = __fmaf_rn(w[0], t0, acc);
    acc = __fmaf_rn(w[1], t1, acc);
    acc = __fmaf_rn(w[2], t2, acc);
    acc = __fmaf_rn(w[3], cur, acc);
    return __float2bfloat16_rn(tf_silu(rbf(acc)));
}

// Thread a channel: BASE holds the last three committed inputs; RAW[1 - parity]'s kept rows replay, then the window's.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_conv(const __nv_bfloat16* __restrict__ P, __nv_bfloat16* __restrict__ BASE,
                                                                 __nv_bfloat16* __restrict__ RAW, __nv_bfloat16* __restrict__ XC,
                                                                 const float* __restrict__ CW, const float* __restrict__ CB,
                                                                 const int* __restrict__ META, int R, int PROJ, int XOFF, int CD, int RMAX) {
    const int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= CD) return;
    const int parity = META[1], pk = META[2];
    float t0 = bf(BASE[ch]), t1 = bf(BASE[CD + ch]), t2 = bf(BASE[2 * CD + ch]);
    const float w[4] = {CW[ch], CW[CD + ch], CW[2 * CD + ch], CW[3 * CD + ch]};
    const float bias = CB[ch];
    for (int i = 0; i < pk + R; ++i) {
        const bool prev = i < pk;
        const float cur = prev ? bf(RAW[(static_cast<int64_t>(1 - parity) * RMAX + i) * CD + ch])
                               : bf(P[static_cast<int64_t>(i - pk) * PROJ + XOFF + ch]);
        if (!prev) {
            const int64_t at = (static_cast<int64_t>(parity) * RMAX + (i - pk)) * CD + ch;
            XC[at] = conv4(bias, w, t0, t1, t2, cur);
            RAW[at] = __float2bfloat16_rn(cur);
        }
        t0 = t1;
        t1 = t2;
        t2 = cur;
        if (i == pk - 1) {
            BASE[ch] = __float2bfloat16_rn(t0);
            BASE[CD + ch] = __float2bfloat16_rn(t1);
            BASE[2 * CD + ch] = __float2bfloat16_rn(t2);
        }
    }
}

// Block (head, 32 value rows), states split as scan_rows.cu splits them; kept rows replay first, then the window's.
extern "C" __global__ void __launch_bounds__(128) tf_nemo_scan(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ XC,
                                                                 float* __restrict__ DT, float* __restrict__ S, const float* __restrict__ A,
                                                                 const float* __restrict__ DSK, const float* __restrict__ DTB,
                                                                 const int* __restrict__ META, __nv_bfloat16* __restrict__ Y, int R, float lo,
                                                                 float hi, int PROJ, int XD, int CD, int DTOFF, int H, int DH, int NG, int RMAX) {
    constexpr int DS = 128, TPR = 4, NJ = DS / TPR / 4;
    __shared__ float4 bs[DS / 4], cs[DS / 4];
    const int h = blockIdx.x, g = h / (H / NG);
    const int local = threadIdx.x / TPR, q = threadIdx.x % TPR, d = blockIdx.y * 32 + local;
    const int parity = META[1], pk = META[2];
    const float a = A[h], dsk = DSK[h], dtb = DTB[h];
    float* s0 = S + (static_cast<int64_t>(h) * DH + d) * DS;
    float s[4 * NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 4 * (TPR * j + q));
        s[4 * j] = t.x;
        s[4 * j + 1] = t.y;
        s[4 * j + 2] = t.z;
        s[4 * j + 3] = t.w;
    }
    for (int i = 0; i < pk + R; ++i) {
        const bool prev = i < pk;
        const int buf = prev ? 1 - parity : parity, row = prev ? i : i - pk;
        const int64_t base = (static_cast<int64_t>(buf) * RMAX + row) * CD;
        __syncthreads();
        if (threadIdx.x < DS / 4) {
            const int c = threadIdx.x;
            const __nv_bfloat16* b = XC + base + XD + g * DS + 4 * c;
            const __nv_bfloat16* cc = XC + base + XD + NG * DS + g * DS + 4 * c;
            bs[c] = make_float4(bf(b[0]), bf(b[1]), bf(b[2]), bf(b[3]));
            cs[c] = make_float4(bf(cc[0]), bf(cc[1]), bf(cc[2]), bf(cc[3]));
        }
        __syncthreads();
        const int64_t dt_at = (static_cast<int64_t>(buf) * RMAX + row) * H + h;
        float dt;
        if (prev) {
            dt = DT[dt_at];
        } else {
            const float v = __fadd_rn(bf(P[static_cast<int64_t>(row) * PROJ + DTOFF + h]), dtb);
            dt = fminf(fmaxf(tf_softplus(v), lo), hi);
            if (blockIdx.y == 0 && threadIdx.x == 0) DT[dt_at] = dt;
        }
        const float x = bf(XC[base + h * DH + d]);
        const float da = tf_exp(__fmul_rn(a, dt)), xdt = __fmul_rn(x, dt);
        float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const float4 b = bs[TPR * j + q], c = cs[TPR * j + q];
            s[4 * j] = __fmaf_rn(xdt, b.x, __fmul_rn(s[4 * j], da));
            s[4 * j + 1] = __fmaf_rn(xdt, b.y, __fmul_rn(s[4 * j + 1], da));
            s[4 * j + 2] = __fmaf_rn(xdt, b.z, __fmul_rn(s[4 * j + 2], da));
            s[4 * j + 3] = __fmaf_rn(xdt, b.w, __fmul_rn(s[4 * j + 3], da));
            m[0] = __fmaf_rn(s[4 * j], c.x, m[0]);
            m[1] = __fmaf_rn(s[4 * j + 1], c.y, m[1]);
            m[2] = __fmaf_rn(s[4 * j + 2], c.z, m[2]);
            m[3] = __fmaf_rn(s[4 * j + 3], c.w, m[3]);
        }
        if (i == pk - 1) {
#pragma unroll
            for (int j = 0; j < NJ; ++j)
                *reinterpret_cast<float4*>(s0 + 4 * (TPR * j + q)) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
        }
        float out = __fadd_rn(__fadd_rn(m[0], m[1]), __fadd_rn(m[2], m[3]));
        out = __fadd_rn(out, __shfl_xor_sync(0xffffffffu, out, 1));
        out = __fadd_rn(out, __shfl_xor_sync(0xffffffffu, out, 2));
        if (prev || q != 0) continue;
        const float y = rbf(__fmaf_rn(x, dsk, out));
        const float gz = rbf(tf_silu(bf(P[static_cast<int64_t>(row) * PROJ + h * DH + d])));
        Y[static_cast<int64_t>(row) * XD + h * DH + d] = __float2bfloat16_rn(__fmul_rn(gz, y));
    }
}

extern "C" {

// A window's conv, R <= RMAX rows. p bf16 [R, PROJ] (in_proj's rows: z of XD, then the CD conv channels x, B,
// C, then dt), base bf16 [3, CD] (the last three committed inputs), raw and xc bf16 [2, RMAX, CD] (the parity
// buffers), cw fp32 [4, CD] (tap-major), cb fp32 [CD], meta int32 [3] {pos, parity, prev_keep}: the kept rows
// of raw[1 - parity] replay into base, then the window's rows go to raw[parity] and xc[parity].
int zs_nemotron_mamba_conv(const zs_view* p_, const zs_view* base_, const zs_view* raw_, const zs_view* xc_,
                           const zs_view* cw_, const zs_view* cb_, const zs_view* meta_, int64_t xd) {
    zs::begin();
    const zs::View p(p_), base(base_), raw(raw_), xc(xc_), cw(cw_), cb(cb_), meta(meta_);
    ZS_REQUIRE(base.defined() && base.dtype() == ZS_BF16 && base.dim() == 2 && base.size(0) == 3 && base.is_contiguous(),
               "base: bf16 [3, CD]");
    const int64_t cd = base.size(1);
    ZS_REQUIRE(raw.defined() && raw.dtype() == ZS_BF16 && raw.dim() == 3 && raw.size(0) == 2 && raw.size(2) == cd,
               "raw: bf16 [2, RMAX, CD]");
    const int64_t rmax = raw.size(1);
    ZS_REQUIRE(xc.defined() && xc.dtype() == ZS_BF16 && xc.dim() == 3 && xc.size(0) == 2 && xc.size(1) == rmax &&
                   xc.size(2) == cd,
               "xc: bf16 [2, RMAX, CD]");
    ZS_REQUIRE(p.defined() && p.dtype() == ZS_BF16 && p.dim() == 2 && xd >= 0 && p.size(1) >= xd + cd,
               "p: bf16 [R, PROJ], its conv channels at XD");
    const int64_t rows = p.size(0);
    ZS_REQUIRE(rows >= 1 && rows <= rmax, "rows: 1 to RMAX");
    ZS_REQUIRE(cw.defined() && cw.dtype() == ZS_F32 && cw.numel() == 4 * cd, "cw: fp32 [4, CD]");
    ZS_REQUIRE(cb.defined() && cb.dtype() == ZS_F32 && cb.numel() == cd, "cb: fp32 [CD]");
    ZS_REQUIRE(meta.defined() && meta.dtype() == ZS_I32 && meta.numel() >= 3, "meta: int32 [3]");
    tf_nemo_conv<<<static_cast<unsigned>((cd + 255) / 256), 256, 0, zs::stream()>>>(
        p.data_ptr<__nv_bfloat16>(), base.data_ptr<__nv_bfloat16>(), raw.data_ptr<__nv_bfloat16>(),
        xc.data_ptr<__nv_bfloat16>(), cw.data_ptr<float>(), cb.data_ptr<float>(), meta.data_ptr<int>(),
        static_cast<int>(rows), static_cast<int>(p.row_stride()), static_cast<int>(xd), static_cast<int>(cd),
        static_cast<int>(rmax));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// A window's scan after its conv: p as the conv's, xc bf16 [2, RMAX, CD], dt fp32 [2, RMAX, H] (the rows' dt,
// kept for the replay), ssm fp32 [H, DH, 128] (lagging a window), a, dsk, dtb fp32 [H] (A = -exp(A_log), D,
// dt_bias), meta as the conv's -> y bf16 [R, XD] = silu(z) * (C . state + D x), before the group norm. dt is
// softplus(dt + dt_bias) clamped to [lo, hi]; groups share B and C of 128 states over XD / H-wide heads.
int zs_nemotron_mamba_scan(const zs_view* p_, const zs_view* xc_, const zs_view* dt_, const zs_view* ssm_,
                           const zs_view* a_, const zs_view* dsk_, const zs_view* dtb_, const zs_view* meta_,
                           const zs_view* y_, double lo, double hi, int64_t groups) {
    zs::begin();
    const zs::View p(p_), xc(xc_), dt(dt_), ssm(ssm_), a(a_), dsk(dsk_), dtb(dtb_), meta(meta_), y(y_);
    ZS_REQUIRE(y.defined() && y.dtype() == ZS_BF16 && y.dim() == 2 && y.is_contiguous(), "y: bf16 [R, XD]");
    const int64_t rows = y.size(0), xd = y.size(1);
    ZS_REQUIRE(a.defined() && a.dtype() == ZS_F32 && a.numel() >= 1, "a: fp32 [H]");
    const int64_t h = a.numel();
    ZS_REQUIRE(xd % h == 0 && (xd / h) % 32 == 0, "XD: H heads of a multiple of 32");
    const int64_t dh = xd / h;
    ZS_REQUIRE(dsk.defined() && dsk.dtype() == ZS_F32 && dsk.numel() == h, "dsk: fp32 [H]");
    ZS_REQUIRE(dtb.defined() && dtb.dtype() == ZS_F32 && dtb.numel() == h, "dtb: fp32 [H]");
    ZS_REQUIRE(xc.defined() && xc.dtype() == ZS_BF16 && xc.dim() == 3 && xc.size(0) == 2, "xc: bf16 [2, RMAX, CD]");
    const int64_t rmax = xc.size(1), cd = xc.size(2);
    ZS_REQUIRE(groups >= 1 && h % groups == 0 && cd == xd + 2 * groups * 128, "CD: XD then B and C of 128 a group");
    ZS_REQUIRE(rows >= 1 && rows <= rmax, "rows: 1 to RMAX");
    ZS_REQUIRE(p.defined() && p.dtype() == ZS_BF16 && p.dim() == 2 && p.size(0) == rows && p.size(1) >= xd + cd + h,
               "p: bf16 [R, PROJ] of z, the conv channels, dt");
    ZS_REQUIRE(dt.defined() && dt.dtype() == ZS_F32 && dt.numel() == 2 * rmax * h, "dt: fp32 [2, RMAX, H]");
    ZS_REQUIRE(ssm.defined() && ssm.dtype() == ZS_F32 && ssm.numel() == h * dh * 128, "ssm: fp32 [H, DH, 128]");
    ZS_REQUIRE(meta.defined() && meta.dtype() == ZS_I32 && meta.numel() >= 3, "meta: int32 [3]");
    tf_nemo_scan<<<dim3(static_cast<unsigned>(h), static_cast<unsigned>(dh / 32)), 128, 0, zs::stream()>>>(
        p.data_ptr<__nv_bfloat16>(), xc.data_ptr<__nv_bfloat16>(), dt.data_ptr<float>(), ssm.data_ptr<float>(),
        a.data_ptr<float>(), dsk.data_ptr<float>(), dtb.data_ptr<float>(), meta.data_ptr<int>(),
        y.data_ptr<__nv_bfloat16>(), static_cast<int>(rows), static_cast<float>(lo), static_cast<float>(hi),
        static_cast<int>(p.row_stride()), static_cast<int>(xd), static_cast<int>(cd), static_cast<int>(xd + cd),
        static_cast<int>(h), static_cast<int>(dh), static_cast<int>(groups), static_cast<int>(rmax));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
