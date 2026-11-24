// Nemotron's Mamba-2 prompt path: a chunk of any rows in one conv and one scan launch, each row's arithmetic
// that of mamba.cu's tf_nemo_conv and tf_nemo_scan (TensorFold's step kernels), so a prompt's rows, outputs and
// state are the bits of the same rows run as serial steps. zspark's own (spec 151 WS-02), Apache-2.0.
//
// Why not TensorFold's prefill: its scan_rows.cu runs other transcendentals than the step's (a prompt's state
// would differ from its steps'), and tf_nemo_conv_rows starts from BASE without replaying the kept rows. Why
// hand-written: a recurrence over rows, no GEMM in it (zig/kernels/AGENTS.md, "CUTLASS").
//
// The state model is the step kernels': ssm, base and dt lag a window; meta {pos, parity, prev_keep} says which
// rows of the parity buffers replay first (raw/xc/dt[1 - parity][0, pk)). A chunk of R rows replays those pk
// rows, then runs its R rows, and commits ssm and base through all but its last row, which it leaves in the
// parity buffers' slot 0 (raw/xc/dt[parity][0]): exactly the state R serial steps leave. The caller then sets
// meta {pos + R, parity ^ 1, 1} (mamba.zig's Meta.afterPrompt), and the next window or chunk replays that row.
//
// Built with --fmad=false (build.zig), as the step kernels are: every fma is written as __fmaf_rn, every other
// product and sum is IEEE, in the step kernels' order. nemo_mamba_a is TensorFold's
// torch_ops/nemotron_constants.cu tf_nemotron_mamba_a_f32_kernel's arithmetic (expf, then the sign bit), built
// with the flags upstream gives it (-O3 --fmad=false --ftz=false).
//
// Measured beside a live server on the GPU (not a clean measurement; spec 151 WS-02): a 2,048-row chunk at
// Transcribe's dimensions, conv 0.23 ms, scan 1.49 ms, the mixer with its norm 1.86 ms; 45,000 rows in chunks of
// 2,048 41 ms a layer; 2,048 serial steps of the window kernels 34 ms.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"
#include "glue_math.cuh"

namespace {

using namespace tf_glue;

constexpr int CONV_ROWS = 16;  // rows a conv thread runs, its taps slid from one row to the next
constexpr int SCAN_TILE = 16;  // rows a scan block stages at once
constexpr int DS = 128;        // states a group

// mamba.cu's conv4, op for op: bf16(silu(bf16(bias + w0 t0 + w1 t1 + w2 t2 + w3 cur))).
__device__ __forceinline__ __nv_bfloat16 wide_conv4(float bias, const float (&w)[4], float t0, float t1, float t2, float cur) {
    float acc = bias;
    acc = __fmaf_rn(w[0], t0, acc);
    acc = __fmaf_rn(w[1], t1, acc);
    acc = __fmaf_rn(w[2], t2, acc);
    acc = __fmaf_rn(w[3], cur, acc);
    return __float2bfloat16_rn(tf_silu(rbf(acc)));
}

// The conv's input j of the virtual sequence [BASE's three; raw[1 - parity]'s pk kept rows; the chunk's rows].
__device__ __forceinline__ float wide_input(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ BASE,
                                            const __nv_bfloat16* __restrict__ RAW, int j, int ch, int pk, int parity,
                                            int PROJ, int XOFF, int CD, int RMAX) {
    if (j < 0) return bf(BASE[static_cast<int64_t>(j + 3) * CD + ch]);
    if (j < pk) return bf(RAW[(static_cast<int64_t>(1 - parity) * RMAX + j) * CD + ch]);
    return bf(P[static_cast<int64_t>(j - pk) * PROJ + XOFF + ch]);
}

}  // namespace

// Thread (channel, CONV_ROWS rows): each row's taps are the three inputs before it, so rows need not run in
// order; XW [R, CD] gets every row's conv, and the last row's input and conv go to raw/xc[parity][0].
extern "C" __global__ void __launch_bounds__(256) nemo_conv_wide(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ BASE,
                                                                 __nv_bfloat16* __restrict__ RAW, __nv_bfloat16* __restrict__ XC,
                                                                 __nv_bfloat16* __restrict__ XW, const float* __restrict__ CW,
                                                                 const float* __restrict__ CB, const int* __restrict__ META, int R,
                                                                 int PROJ, int XOFF, int CD, int RMAX) {
    const int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= CD) return;
    const int parity = META[1], pk = META[2];
    const int r0 = blockIdx.y * CONV_ROWS, r1 = min(R, r0 + CONV_ROWS);
    const float w[4] = {CW[ch], CW[CD + ch], CW[2 * CD + ch], CW[3 * CD + ch]};
    const float bias = CB[ch];
    float t0 = wide_input(P, BASE, RAW, pk + r0 - 3, ch, pk, parity, PROJ, XOFF, CD, RMAX);
    float t1 = wide_input(P, BASE, RAW, pk + r0 - 2, ch, pk, parity, PROJ, XOFF, CD, RMAX);
    float t2 = wide_input(P, BASE, RAW, pk + r0 - 1, ch, pk, parity, PROJ, XOFF, CD, RMAX);
    for (int r = r0; r < r1; ++r) {
        const float cur = bf(P[static_cast<int64_t>(r) * PROJ + XOFF + ch]);
        const __nv_bfloat16 out = wide_conv4(bias, w, t0, t1, t2, cur);
        XW[static_cast<int64_t>(r) * CD + ch] = out;
        if (r == R - 1) {
            const int64_t at = static_cast<int64_t>(parity) * RMAX * CD + ch;
            XC[at] = out;
            RAW[at] = __float2bfloat16_rn(cur);
        }
        t0 = t1;
        t1 = t2;
        t2 = cur;
    }
}

// BASE <- the inputs before the chunk's last row (its kept prefix's last three), after nemo_conv_wide read BASE.
extern "C" __global__ void __launch_bounds__(256) nemo_conv_wide_commit(const __nv_bfloat16* __restrict__ P, __nv_bfloat16* __restrict__ BASE,
                                                                        const __nv_bfloat16* __restrict__ RAW, const int* __restrict__ META,
                                                                        int R, int PROJ, int XOFF, int CD, int RMAX) {
    const int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= CD) return;
    const int parity = META[1], pk = META[2];
    const int c = pk + R - 1;  // inputs committed: the kept rows and all of the chunk's but its last
    float v[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) v[k] = wide_input(P, BASE, RAW, c - 3 + k, ch, pk, parity, PROJ, XOFF, CD, RMAX);
#pragma unroll
    for (int k = 0; k < 3; ++k) BASE[static_cast<int64_t>(k) * CD + ch] = __float2bfloat16_rn(v[k]);
}

// Block (head, 32 value rows) of 128 threads, a value row's 128 states over 4 lanes, as tf_nemo_scan; the rows
// run in order inside, SCAN_TILE at a time staged in shared memory:
//   bs, cs   [SCAN_TILE][32] float4   the rows' B and C of the head's group (16 KiB)
//   xs, gzs  [SCAN_TILE][32] float    the block's 32 x values a row, and bf16(silu(z)) (4 KiB)
//   dts, das [SCAN_TILE]     float    a row's dt and exp(A dt), one value a head
// B and C cannot be fewer: every value row of the head reads all 128 of each a row. 20.6 KiB keep the 128
// blocks resident at once (the GB10's 100 KiB a multiprocessor; a tile of 32 rows, 41 KiB, ran them in two
// waves), and the next tile's loads are in registers (stage_*) while this tile's rows run: unstaged, the
// loads were two thirds of a 2,048-row chunk's scan. Each value's arithmetic is tf_nemo_scan's: dt, da, the
// state update and the four-way sums in its order, the commit through the chunk's row before the last, y and
// its gate.
extern "C" __global__ void __launch_bounds__(128) nemo_scan_wide(const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ XC,
                                                                 const __nv_bfloat16* __restrict__ XW, float* __restrict__ DT,
                                                                 float* __restrict__ S, const float* __restrict__ A,
                                                                 const float* __restrict__ DSK, const float* __restrict__ DTB,
                                                                 const int* __restrict__ META, __nv_bfloat16* __restrict__ Y, int R,
                                                                 float lo, float hi, int PROJ, int XD, int CD, int DTOFF, int H, int DH,
                                                                 int NG, int RMAX) {
    constexpr int TPR = 4, NJ = DS / TPR / 4, NT = 128;
    constexpr int NBC = SCAN_TILE * (DS / 2) / NT;  // B and C loads a thread a tile, four values each
    constexpr int NXZ = SCAN_TILE * 32 / NT;        // x and z values a thread a tile
    __shared__ float4 bs[SCAN_TILE][DS / 4], cs[SCAN_TILE][DS / 4];
    __shared__ float xs[SCAN_TILE][32], gzs[SCAN_TILE][32];
    __shared__ float dts[SCAN_TILE], das[SCAN_TILE];
    const int h = blockIdx.x, g = h / (H / NG);
    const int local = threadIdx.x / TPR, q = threadIdx.x % TPR, d0 = blockIdx.y * 32, d = d0 + local;
    const int parity = META[1], pk = META[2];
    const int total = pk + R;
    const float a = A[h], dsk = DSK[h], dtb = DTB[h];
    // virtual row i's conv outputs: a kept row's from xc[1 - parity], the chunk's from XW
    auto xrow = [&](int i) -> const __nv_bfloat16* {
        return i < pk ? XC + (static_cast<int64_t>(1 - parity) * RMAX + i) * CD : XW + static_cast<int64_t>(i - pk) * CD;
    };
    uint2 stage_bc[NBC];
    __nv_bfloat16 stage_x[NXZ], stage_z[NXZ];
    float stage_dt = 0.0f;
    auto fetch = [&](int i0) {
        const int n = min(SCAN_TILE, total - i0);
#pragma unroll
        for (int e = 0; e < NBC; ++e) {
            const int idx = threadIdx.x + e * NT, k = idx / (DS / 2), part = idx % (DS / 2);
            if (k < n) {
                const int c = part % (DS / 4);
                const __nv_bfloat16* b = xrow(i0 + k) + XD + (part < DS / 4 ? 0 : NG * DS) + g * DS + 4 * c;
                stage_bc[e] = *reinterpret_cast<const uint2*>(b);
            }
        }
#pragma unroll
        for (int e = 0; e < NXZ; ++e) {
            const int idx = threadIdx.x + e * NT, k = idx / 32, dd = idx % 32, i = i0 + k;
            if (k < n) {
                stage_x[e] = xrow(i)[h * DH + d0 + dd];
                if (i >= pk) stage_z[e] = P[static_cast<int64_t>(i - pk) * PROJ + h * DH + d0 + dd];
            }
        }
        if (threadIdx.x < n) {
            const int i = i0 + threadIdx.x;
            stage_dt = i < pk ? DT[(static_cast<int64_t>(1 - parity) * RMAX + i) * H + h]
                              : bf(P[static_cast<int64_t>(i - pk) * PROJ + DTOFF + h]);
        }
    };
    auto store = [&](int i0) {
        const int n = min(SCAN_TILE, total - i0);
#pragma unroll
        for (int e = 0; e < NBC; ++e) {
            const int idx = threadIdx.x + e * NT, k = idx / (DS / 2), part = idx % (DS / 2);
            if (k < n) {
                const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&stage_bc[e]);
                const float4 v = make_float4(bf(b[0]), bf(b[1]), bf(b[2]), bf(b[3]));
                if (part < DS / 4)
                    bs[k][part] = v;
                else
                    cs[k][part - DS / 4] = v;
            }
        }
#pragma unroll
        for (int e = 0; e < NXZ; ++e) {
            const int idx = threadIdx.x + e * NT, k = idx / 32, dd = idx % 32;
            if (k < n) {
                xs[k][dd] = bf(stage_x[e]);
                gzs[k][dd] = i0 + k < pk ? 0.0f : rbf(tf_silu(bf(stage_z[e])));
            }
        }
        if (threadIdx.x < n) {
            const int k = threadIdx.x, i = i0 + k;
            float dt = stage_dt;
            if (i >= pk) {
                const float v = __fadd_rn(stage_dt, dtb);
                dt = fminf(fmaxf(tf_softplus(v), lo), hi);
                if (i == total - 1 && blockIdx.y == 0) DT[static_cast<int64_t>(parity) * RMAX * H + h] = dt;
            }
            dts[k] = dt;
            das[k] = tf_exp(__fmul_rn(a, dt));
        }
    };
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
    fetch(0);
    for (int i0 = 0; i0 < total; i0 += SCAN_TILE) {
        const int n = min(SCAN_TILE, total - i0);
        __syncthreads();
        store(i0);
        __syncthreads();
        if (i0 + SCAN_TILE < total) fetch(i0 + SCAN_TILE);
        for (int k = 0; k < n; ++k) {
            const int i = i0 + k;
            const float x = xs[k][local];
            const float da = das[k], xdt = __fmul_rn(x, dts[k]);
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const float4 b = bs[k][TPR * j + q], c = cs[k][TPR * j + q];
                s[4 * j] = __fmaf_rn(xdt, b.x, __fmul_rn(s[4 * j], da));
                s[4 * j + 1] = __fmaf_rn(xdt, b.y, __fmul_rn(s[4 * j + 1], da));
                s[4 * j + 2] = __fmaf_rn(xdt, b.z, __fmul_rn(s[4 * j + 2], da));
                s[4 * j + 3] = __fmaf_rn(xdt, b.w, __fmul_rn(s[4 * j + 3], da));
                m[0] = __fmaf_rn(s[4 * j], c.x, m[0]);
                m[1] = __fmaf_rn(s[4 * j + 1], c.y, m[1]);
                m[2] = __fmaf_rn(s[4 * j + 2], c.z, m[2]);
                m[3] = __fmaf_rn(s[4 * j + 3], c.w, m[3]);
            }
            if (i == total - 2) {
#pragma unroll
                for (int j = 0; j < NJ; ++j)
                    *reinterpret_cast<float4*>(s0 + 4 * (TPR * j + q)) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2], s[4 * j + 3]);
            }
            float out = __fadd_rn(__fadd_rn(m[0], m[1]), __fadd_rn(m[2], m[3]));
            out = __fadd_rn(out, __shfl_xor_sync(0xffffffffu, out, 1));
            out = __fadd_rn(out, __shfl_xor_sync(0xffffffffu, out, 2));
            if (i < pk || q != 0) continue;
            const float y = rbf(__fmaf_rn(x, dsk, out));
            Y[static_cast<int64_t>(i - pk) * XD + h * DH + d] = __float2bfloat16_rn(__fmul_rn(gzs[k][local], y));
        }
    }
}

// A = -exp(A_log), as TensorFold's tf_nemotron_mamba_a_f32_kernel computes it at load.
extern "C" __global__ void __launch_bounds__(128) nemo_mamba_a(const float* __restrict__ logarithms, float* __restrict__ coefficients, int n) {
    const int at = blockIdx.x * 128 + threadIdx.x;
    if (at >= n) return;
    const float magnitude = ::expf(logarithms[at]);
    coefficients[at] = __uint_as_float(__float_as_uint(magnitude) ^ 0x80000000u);
}

extern "C" {

// A prompt chunk's conv, R >= 1 rows of any count: p bf16 [R, PROJ] (its conv channels at XD), base bf16 [3, CD],
// raw and xc bf16 [2, RMAX, CD], cw fp32 [4, CD], cb fp32 [CD], meta int32 [3] -> xw bf16 [R, CD] the rows' conv;
// base commits through the row before the last, which goes to raw/xc[parity][0].
int zs_nemotron_mamba_conv_wide(const zs_view* p_, const zs_view* base_, const zs_view* raw_, const zs_view* xc_,
                                const zs_view* xw_, const zs_view* cw_, const zs_view* cb_, const zs_view* meta_,
                                int64_t xd) {
    zs::begin();
    const zs::View p(p_), base(base_), raw(raw_), xc(xc_), xw(xw_), cw(cw_), cb(cb_), meta(meta_);
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
    ZS_REQUIRE(rows >= 1 && (rows + CONV_ROWS - 1) / CONV_ROWS <= 65535, "rows: 1 to 65535 * 16");
    ZS_REQUIRE(xw.defined() && xw.dtype() == ZS_BF16 && xw.is_contiguous() && xw.numel() >= rows * cd,
               "xw: bf16 [R, CD]");
    ZS_REQUIRE(cw.defined() && cw.dtype() == ZS_F32 && cw.numel() == 4 * cd, "cw: fp32 [4, CD]");
    ZS_REQUIRE(cb.defined() && cb.dtype() == ZS_F32 && cb.numel() == cd, "cb: fp32 [CD]");
    ZS_REQUIRE(meta.defined() && meta.dtype() == ZS_I32 && meta.numel() >= 3, "meta: int32 [3]");
    const unsigned cblocks = static_cast<unsigned>((cd + 255) / 256);
    nemo_conv_wide<<<dim3(cblocks, static_cast<unsigned>((rows + CONV_ROWS - 1) / CONV_ROWS)), 256, 0, zs::stream()>>>(
        p.data_ptr<__nv_bfloat16>(), base.data_ptr<__nv_bfloat16>(), raw.data_ptr<__nv_bfloat16>(),
        xc.data_ptr<__nv_bfloat16>(), xw.data_ptr<__nv_bfloat16>(), cw.data_ptr<float>(), cb.data_ptr<float>(),
        meta.data_ptr<int>(), static_cast<int>(rows), static_cast<int>(p.row_stride()), static_cast<int>(xd),
        static_cast<int>(cd), static_cast<int>(rmax));
    ZS_CUDA_LAUNCH_CHECK();
    nemo_conv_wide_commit<<<cblocks, 256, 0, zs::stream()>>>(
        p.data_ptr<__nv_bfloat16>(), base.data_ptr<__nv_bfloat16>(), raw.data_ptr<__nv_bfloat16>(),
        meta.data_ptr<int>(), static_cast<int>(rows), static_cast<int>(p.row_stride()), static_cast<int>(xd),
        static_cast<int>(cd), static_cast<int>(rmax));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// A prompt chunk's scan after its conv: p, xc, dt, ssm, a, dsk, dtb, meta as zs_nemotron_mamba_scan's, xw bf16
// [R, CD] the chunk's conv -> y bf16 [R, XD]; ssm commits through the row before the last, whose dt goes to
// dt[parity][0].
int zs_nemotron_mamba_scan_wide(const zs_view* p_, const zs_view* xc_, const zs_view* xw_, const zs_view* dt_,
                                const zs_view* ssm_, const zs_view* a_, const zs_view* dsk_, const zs_view* dtb_,
                                const zs_view* meta_, const zs_view* y_, double lo, double hi, int64_t groups) {
    zs::begin();
    const zs::View p(p_), xc(xc_), xw(xw_), dt(dt_), ssm(ssm_), a(a_), dsk(dsk_), dtb(dtb_), meta(meta_), y(y_);
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
    ZS_REQUIRE(groups >= 1 && h % groups == 0 && cd == xd + 2 * groups * DS, "CD: XD then B and C of 128 a group");
    ZS_REQUIRE(rows >= 1, "rows: at least 1");
    ZS_REQUIRE(p.defined() && p.dtype() == ZS_BF16 && p.dim() == 2 && p.size(0) == rows && p.size(1) >= xd + cd + h,
               "p: bf16 [R, PROJ] of z, the conv channels, dt");
    ZS_REQUIRE(xw.defined() && xw.dtype() == ZS_BF16 && xw.is_contiguous() && xw.numel() >= rows * cd,
               "xw: bf16 [R, CD]");
    ZS_REQUIRE(dt.defined() && dt.dtype() == ZS_F32 && dt.numel() == 2 * rmax * h, "dt: fp32 [2, RMAX, H]");
    ZS_REQUIRE(ssm.defined() && ssm.dtype() == ZS_F32 && ssm.numel() == h * dh * DS, "ssm: fp32 [H, DH, 128]");
    ZS_REQUIRE(meta.defined() && meta.dtype() == ZS_I32 && meta.numel() >= 3, "meta: int32 [3]");
    nemo_scan_wide<<<dim3(static_cast<unsigned>(h), static_cast<unsigned>(dh / 32)), 128, 0, zs::stream()>>>(
        p.data_ptr<__nv_bfloat16>(), xc.data_ptr<__nv_bfloat16>(), xw.data_ptr<__nv_bfloat16>(), dt.data_ptr<float>(),
        ssm.data_ptr<float>(), a.data_ptr<float>(), dsk.data_ptr<float>(), dtb.data_ptr<float>(), meta.data_ptr<int>(),
        y.data_ptr<__nv_bfloat16>(), static_cast<int>(rows), static_cast<float>(lo), static_cast<float>(hi),
        static_cast<int>(p.row_stride()), static_cast<int>(xd), static_cast<int>(cd), static_cast<int>(xd + cd),
        static_cast<int>(h), static_cast<int>(dh), static_cast<int>(groups), static_cast<int>(rmax));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// alog fp32 [N] -> a fp32 [N] = -exp(alog): the Mamba heads' decay at load.
int zs_nemotron_mamba_a(const zs_view* alog_, const zs_view* a_) {
    zs::begin();
    const zs::View alog(alog_), a(a_);
    ZS_REQUIRE(alog.defined() && alog.dtype() == ZS_F32 && alog.is_contiguous(), "alog: fp32 [N]");
    ZS_REQUIRE(a.defined() && a.dtype() == ZS_F32 && a.is_contiguous() && a.numel() == alog.numel(), "a: fp32 [N]");
    const int64_t n = alog.numel();
    if (n > 0)
        nemo_mamba_a<<<static_cast<unsigned>((n + 127) / 128), 128, 0, zs::stream()>>>(alog.data_ptr<float>(),
                                                                                        a.data_ptr<float>(),
                                                                                        static_cast<int>(n));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
