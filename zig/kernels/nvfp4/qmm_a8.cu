// zspark's own (spec 114 WS-04), after qmmf.cu's lane matmul (TensorFold 0.6.5, 609ca41, Apache-2.0) and
// qmmf_tile.cu, which stay as they are beside this file. MXFP8 and block-FP8 linears with their activations
// quantized as the card's vLLM quantizes them (b12x), products on FP8 tensor cores (W8A8), in every phase:
//   out (M, N) = quant(x) (M, K) @ W, x bf16, W e4m3 in qmmf's fragment order [npad/64][K/64][8][32][2][8].
//
// The activations (vLLM's choice, read in its source, 2026-10-19):
// - MXFP8 (vLLM `kMxfp8Dynamic`, b12x `gemm/blockscaled/_quantize.py`): a row's 32 inputs share an e8m0 exponent
//   e = clamp(ceil(log2(amax / 448)), -127, 127), 0 for a block of zeros; each input is rne_satfinite_e4m3(x / 2^e).
//   b12x takes the log through Triton's approximate lg2 and divides by `div.full`; here e is the least integer with
//   2^e >= rn(amax / 448), read from the quotient's bits, which those agree with but within an ulp of a power of two.
// - Block FP8 (vLLM `kFp8Dynamic128Sym`, `per_token_group_fp8_quant`, use_ue8m0 off): a row's 128 inputs share an
//   fp32 s = max(amax, 1e-10) / 448, each input rne_satfinite_e4m3(clamp(x / s, -448, 448)), both quotients IEEE.
// - Per-tensor FP8: positive input_scale uses the checkpoint's scalar in both tile forms, with no amax reduction.
//   Weights retain their stored scalar through FP8G's tile scales; shared memory is unchanged.
// - Every row, whatever the window: vLLM's b12x quantizes MXFP8 at more than 8 rows only, which would make a row's
//   bits hang on its window; zspark takes the every-phase form (spec 114's 2026-10-19 decision).
// The products (b12x's arithmetic):
// - MXFP8: `mma.sync m16n8k32 kind::mxf8f6f4.block_scale.scale_vec::1X` with both exponents in the instruction, one
//   chain over a K slice; slices added in slice order (qmmf's), each from zero.
// - Block FP8: a 128-input block's products from zero by the same instruction with unit exponents, then
//   acc = fma(block sum, s_x * s_w, acc) in block order (b12x `_block_fp8_gemv.py`: "each 128-block sum is scaled
//   in FP32 by rhs_scale * lhs_scale and accumulated in FP32").
// K's order: the weight's bytes stay where qmmf's bf16 k16 steps read them, a lane's eight of a 32-input block as
// the k32 B fragment, so within each 32-input block both operands are read in the order `perm` gives (b12x
// permutes K within its 128-blocks the same way); a block's scale is unchanged by it.
//
// Forms, one bits: a window of 1 to 16 rows (bm 16) either quantizes x in the block, stage by stage, where its
// weight is on the way, or — where `work` is given and the caller asks (block FP8 with a static activation scale;
// spec 153 WS-05) — quantizes once by quant_a8_kernel into `work` and reads from there, as bm 64's rows do.
// Every form encodes with `encode16`, so a row's bytes and scales are the same in all, and a row's products are an
// mma row's alone; K slices come from the weight's shape (`nvfp4.splitK`), summed in slice order by a cluster or in
// the block.
//
// Shared memory (the SM's 102,400 B for all its blocks), a stage of 128 inputs (two of the weight's 64-groups):
//   form (warps, stages)          x a stage                    weight   its scales   + once            a block   an SM
//   bm 16 (4, 3), quantized here  16 rows bf16, 4,096          8,192    256 (MX)     x e4m3 2,048,     39,744    2
//                                                                                    scales 64
//   bm 64 (4, 3)                  64 rows e4m3, 8,192          8,192    256          -                 49,920    2
//   Block FP8's weight scale is 4 B a stage (one a tile, `tile_scales`), the stage's line rounded to 128 B. The
//   rows' scales of the bm 64 form go to registers from global (L2) a stage ahead, so that two blocks
//   fit an SM; the cluster's partial sums reuse the stages once the loop is over. What could be less: bm 16 stages
//   16 rows of x whatever the window (at one row 15 of 16 hold zeros), as qmmf_tile.cu does; its weight on the way
//   an SM (48 KB) is qmmf_tile.cu's.

#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using namespace qmm_frag;

enum Mode : int { MXFP8 = 2, FP8G = 3 };                  // nvfp4.Mode's

constexpr int GS = 128;                                   // inputs a stage: a block FP8 block, four MX blocks
constexpr int BN = 64;                                    // a block's columns: one weight tile

// Position p of a 32-input block holds input perm(p): how qmmf's fragment bytes read as a k32 B fragment.
__host__ __device__ constexpr int perm(int p) {
    return (p & 16) + 2 * ((p & 15) >> 2) + (p & 1) + 8 * ((p >> 1) & 1);
}
// the order `encode16` writes by hand: position 2 holds input 8, position 4 input 2
static_assert(perm(1) == 1 && perm(2) == 8 && perm(4) == 2 && perm(16) == 16, "perm");

__device__ __forceinline__ void unpack16(uint4 a, uint4 b, float (&v)[16]) {
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[2 * i] = __uint_as_float(w[i] << 16);
        v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
    }
}

__device__ __forceinline__ float amax16(const float (&v)[16]) {
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(v[i]));
    return m;
}

// The largest |x| of a scale's group: two lanes' 16 inputs (MX), eight lanes' (block FP8), side by side.
template <int MODE>
__device__ __forceinline__ float group_amax(float m) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    if constexpr (MODE == FP8G) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
    }
    return m;
}

// The least e with 2^e >= rn(amax / 448), in [-127, 127]; 0 for amax 0.
__device__ __forceinline__ int mx_exponent(float amax) {
    if (!(amax > 0.0f)) return 0;
    const int bits = __float_as_int(__fdiv_rn(amax, 448.0f));
    const int ex = (bits >> 23) & 0xFF, man = bits & 0x7FFFFF;
    const int e = ex == 0 ? (32 - __clz(man - 1)) - 149 : ex - 127 + (man != 0);
    return min(max(e, -127), 127);
}

__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
    return static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3));
}

// 16 inputs of a row (half of a 32-input block, in order) as their 16 e4m3 bytes in `perm`'s order, under the scale
// of their group, whose largest |x| is `amax`; `scale` its bits: the e8m0 byte (MX) or the fp32 (block FP8).
template <int MODE>
__device__ __forceinline__ uint4 encode16(const float (&v)[16], float amax, uint32_t& scale, float input_scale) {
    float q[16];
    if constexpr (MODE == MXFP8) {
        const int e = mx_exponent(amax);
        scale = static_cast<uint32_t>(e + 127);
        const float f = __int_as_float((127 - min(e, 126)) << 23);   // 2^-e, exact (e 127 only past bf16's range)
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = v[i] * f;
    } else {
        const float s = input_scale > 0 ? input_scale : __fdiv_rn(fmaxf(amax, 1e-10f), 448.0f);
        scale = __float_as_uint(s);
        // v / s rounded as IEEE division (Markstein: r = rn(1/s), q0 = rn(v r), the residual exact by fma, then
        // rn(q0 + res r) is the rounded quotient; `nvfp4_a8_test.zig` holds it to division on the host): __fdiv_rn
        // an input made the decode form, which quantizes in every block, 1.3 to 2.3 times slower than qmmf_tile.cu
        const float r = __frcp_rn(s);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float q0 = __fmul_rn(v[i], r);
            q[i] = fminf(fmaxf(copysignf(fmaf(fmaf(-q0, s, v[i]), r, q0), v[i]), -448.0f), 448.0f);   // -0 kept
        }
    }
    // input pairs (0,1) (2,3) (4,5) (6,7) to positions 0, 4, 8, 12; (8,9) ... (14,15) to 2, 6, 10, 14
    const uint32_t p0 = e4m3x2(q[0], q[1]), p1 = e4m3x2(q[8], q[9]), p2 = e4m3x2(q[2], q[3]), p3 = e4m3x2(q[10], q[11]);
    const uint32_t p4 = e4m3x2(q[4], q[5]), p5 = e4m3x2(q[12], q[13]), p6 = e4m3x2(q[6], q[7]), p7 = e4m3x2(q[14], q[15]);
    return make_uint4(p0 | (p1 << 16), p2 | (p3 << 16), p4 | (p5 << 16), p6 | (p7 << 16));
}

// x (M, K) bf16 -> xq (M, K) e4m3 in `perm`'s order and the rows' scales (MX: e8m0 [M, K/32]; block FP8: fp32
// [M, K/128]). A thread 16 inputs; a scale's group is 2 or 8 neighbouring lanes, never split (K % 128 == 0).
template <int MODE>
__global__ void __launch_bounds__(256) quant_a8_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ xq,
                                                       void* __restrict__ xs, int M, int K, int ldx, float input_scale) {
    const long long parts = K / 16, total = static_cast<long long>(M) * parts;
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long u = t < total ? t : total - 1;         // a whole group past the end: computed, not stored
    const int row = static_cast<int>(u / parts), j = static_cast<int>(u % parts);
    const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(row) * ldx + j * 16);
    float v[16];
    unpack16(src[0], src[1], v);
    uint32_t scale;
    const uint4 q = encode16<MODE>(v, input_scale > 0 ? 0 : group_amax<MODE>(amax16(v)), scale, input_scale);
    if (t >= total) return;
    *reinterpret_cast<uint4*>(xq + static_cast<size_t>(row) * K + j * 16) = q;
    if constexpr (MODE == MXFP8) {
        if ((j & 1) == 0) reinterpret_cast<uint8_t*>(xs)[static_cast<size_t>(row) * (K / 32) + j / 2] = static_cast<uint8_t>(scale);
    } else {
        if ((j & 7) == 0) reinterpret_cast<uint32_t*>(xs)[static_cast<size_t>(row) * (K / 128) + j / 8] = scale;
    }
}

// m16n8k32 e4m3 x e4m3 with ue8m0 block scales: sa a row's (lane % 4 == 0: row g, == 1: row g + 8), sb a column's
// (lane % 4 == 0), BYTE the byte of each that this k32 step takes.
template <int BYTE>
__device__ __forceinline__ void mma_mx(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                       uint32_t sb) {
    asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%12, %13}, %11, {%12, %13};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BYTE), "n"(0));
}

// The same with unit exponents, from a zero accumulator: a block FP8 block's first k32 step.
__device__ __forceinline__ void mma_unit0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10}, %11, {%12, %12}, %11, {%12, %12};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z), "r"(0x7F7F7F7Fu), "n"(0));
}

__device__ __forceinline__ void mma_unit(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %11}, %10, {%11, %11};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x7F7F7F7Fu), "n"(0));
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }

template <int MODE, int BM, int WM, int WN, int STAGES, bool QUANT>
struct A8Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;               // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;                // n8 tiles a warp
    static constexpr int XROW = QUANT ? GS * 2 : GS;      // a row's bytes a stage: bf16, or e4m3 quantized already
    static constexpr int X = BM * XROW, W = BN * GS;
    static constexpr int SB = MODE == MXFP8 ? 2 * BN * 2 : 4;
    static constexpr int STAGE = (X + W + SB + 127) / 128 * 128;
    static constexpr int XQ = QUANT ? BM * GS : 0;        // the stage's rows quantized (bm 16)
    static constexpr int XS = QUANT ? BM * 4 : 0;         // and their scales
    static constexpr int LOOP = STAGES * STAGE + XQ + XS;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;
    static constexpr int SMEM = LOOP > PARTIALS ? LOOP : PARTIALS;
    static_assert(!QUANT || BM * 8 == THREADS, "the block quantizes 16 inputs a thread");
};

// A block: BM rows by one 64-column tile, and (CLUSTER) one K slice. QUANT: x bf16 (M, K), row stride ldx, quantized
// here; else x is quant_a8_kernel's xq (M, K) and xs its scales. w as qmmf's; bs MX: e8m0 [npad/64][K/64][64][2],
// block FP8: fp32 [npad/64][K/64] (a tile's one scale a 64-group).
template <int MODE, int BM, int WM, int WN, int STAGES, bool QUANT, bool F32, bool CLUSTER, bool FUSE>
__global__ void __launch_bounds__(WM * WN * 32, 2) qmma8_kernel(
        const void* __restrict__ x, const void* __restrict__ xs, const unsigned char* __restrict__ w,
        const uint8_t* __restrict__ bs, void* __restrict__ out, int M, int N, int K, int SK, int ldx, int group, float input_scale) {
    using T = A8Tile<MODE, BM, WM, WN, STAGES, QUANT>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS, KG64 = K / 64, slice_groups = KG / SK, per = FUSE ? KG : slice_groups;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y, tile = n0 / BN, slice = FUSE ? 0 : blockIdx.z, g0 = slice * per;
    unsigned char* const xq_buf = buf + STAGES * T::STAGE;
    unsigned char* const xs_buf = xq_buf + T::XQ;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        if constexpr (QUANT) {                             // rows of 256 B, as they are: the block reads them once
            for (int c = tid; c < BM * 16; c += T::THREADS) {
                const int r = c / 16, ch = c % 16;
                const int row = min(m0 + r, M - 1);
                cp16z(p + r * T::XROW + ch * 16,
                      reinterpret_cast<const __nv_bfloat16*>(x) + static_cast<size_t>(row) * ldx + g * GS + ch * 8, m0 + r < M);
            }
        } else {                                           // rows of 128 B, swizzled for ldmatrix
            for (int c = tid; c < BM * 8; c += T::THREADS) {
                const int r = c / 8, ch = c % 8;
                const int row = min(m0 + r, M - 1);
                cp16z(p + r * T::XROW + swz<8>(r, ch) * 16,
                      reinterpret_cast<const uint8_t*>(x) + static_cast<size_t>(row) * K + g * GS + ch * 16, m0 + r < M);
            }
        }
        unsigned char* pw = p + T::X;
        const unsigned char* src = w + (static_cast<size_t>(tile) * KG64 + 2 * g) * 4096;
        for (int c = tid; c < T::W / 16; c += T::THREADS) cp16(pw + c * 16, src + c * 16);
        unsigned char* ps = pw + T::W;
        if constexpr (MODE == MXFP8) {
            if (tid < T::SB / 16) cp16(ps + tid * 16, bs + (static_cast<size_t>(tile) * KG64 + 2 * g) * 128 + tid * 16);
        } else {
            if (tid == 0) cp4(ps, reinterpret_cast<const float*>(bs) + static_cast<size_t>(tile) * KG64 + 2 * g);
        }
    };
    // the rows' scales of stage g (forms that read xs): MX a lane's row's four exponents (its row by lane % 4, as
    // the mma takes them), block FP8 its two rows' fp32
    auto row_scales = [&](int g, uint32_t (&v)[T::MT][2]) {
#pragma unroll
        for (int i = 0; i < T::MT; ++i) {
            const int lo = m0 + wm * (BM / WM) + i * 16 + (lane >> 2);
            if constexpr (MODE == MXFP8) {
                const int r = min((lane & 1) ? lo + 8 : lo, M - 1);
                v[i][0] = __ldg(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(xs) + static_cast<size_t>(r) * (K / 32)) + g);
                v[i][1] = 0;
            } else {
                const uint32_t* s = reinterpret_cast<const uint32_t*>(xs);
                v[i][0] = __ldg(s + static_cast<size_t>(min(lo, M - 1)) * (K / 128) + g);
                v[i][1] = __ldg(s + static_cast<size_t>(min(lo + 8, M - 1)) * (K / 128) + g);
            }
        }
    };

    float acc[T::MT][T::NT][4];
    float tot[FUSE ? T::MT : 1][FUSE ? T::NT : 1][4];   // FUSE: the slices' sum so far, in slice order
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    uint32_t sa_next[T::MT][2], sa[T::MT][2];
    if constexpr (!QUANT) row_scales(g0, sa_next);
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, g0 + s);
        commit();
    }
    for (int it = 0; it < per; ++it) {
        if constexpr (FUSE) {
            if (it > 0 && it % slice_groups == 0) {        // a slice ends: add it to the sum, start the next from zero
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            tot[i][j][e] = it == slice_groups ? acc[i][j][e] : tot[i][j][e] + acc[i][j][e];
                            acc[i][j][e] = 0.0f;
                        }
            }
        }
        if constexpr (!QUANT) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i) sa[i][0] = sa_next[i][0], sa[i][1] = sa_next[i][1];
            if (it + 1 < per) row_scales(g0 + it + 1, sa_next);
        }
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, g0 + next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const unsigned char* pa = p;
        if constexpr (QUANT) {                             // the stage's rows to e4m3: a thread 16 inputs of a row
            const int r = tid >> 3, j = tid & 7;
            const uint4* src = reinterpret_cast<const uint4*>(p + r * T::XROW + j * 32);
            float v[16];
            unpack16(src[0], src[1], v);
            uint32_t scale;
            const uint4 q = encode16<MODE>(v, input_scale > 0 ? 0 : group_amax<MODE>(amax16(v)), scale, input_scale);
            *reinterpret_cast<uint4*>(xq_buf + r * GS + swz<8>(r, j) * 16) = q;
            if constexpr (MODE == MXFP8) {
                if ((j & 1) == 0) xs_buf[r * 4 + j / 2] = static_cast<uint8_t>(scale);
            } else {
                if (j == 0) reinterpret_cast<uint32_t*>(xs_buf)[r] = scale;
            }
            __syncthreads();
            pa = xq_buf;
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int lo = wm * (BM / WM) + i * 16 + (lane >> 2);
                if constexpr (MODE == MXFP8) {
                    sa[i][0] = reinterpret_cast<const uint32_t*>(xs_buf)[(lane & 1) ? lo + 8 : lo];
                } else {
                    sa[i][0] = reinterpret_cast<const uint32_t*>(xs_buf)[lo];
                    sa[i][1] = reinterpret_cast<const uint32_t*>(xs_buf)[lo + 8];
                }
            }
        }
        const unsigned char* pw = p + T::X;
        const unsigned char* ps = pw + T::W;
        uint4 wq[T::NT][2];                                // a lane's 16 bytes of each 64-group, read once
        uint32_t sb[T::NT];                                // MX: its column's four exponents, a k32 step's a byte
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int jj = wn * T::NT + j;
            wq[j][0] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
            wq[j][1] = reinterpret_cast<const uint4*>(pw + 4096)[jj * 32 + lane];
            if constexpr (MODE == MXFP8) {
                const int col = jj * 8 + (lane >> 2);
                sb[j] = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(ps + col * 2)) |
                        (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(ps + 128 + col * 2)) << 16);
            }
        }
        float d[MODE == FP8G ? T::MT : 1][MODE == FP8G ? T::NT : 1][4];
#pragma unroll
        for (int ks = 0; ks < GS / 32; ++ks) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = ks * 2 + (lane >> 4);
                ldmatrix4(a[i], pa + r * GS + swz<8>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t b0 = comp(wq[j][ks >> 1], (ks & 1) * 2), b1 = comp(wq[j][ks >> 1], (ks & 1) * 2 + 1);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if constexpr (MODE == MXFP8) {
                        if (ks == 0) mma_mx<0>(acc[i][j], a[i], b0, b1, sa[i][0], sb[j]);
                        else if (ks == 1) mma_mx<1>(acc[i][j], a[i], b0, b1, sa[i][0], sb[j]);
                        else if (ks == 2) mma_mx<2>(acc[i][j], a[i], b0, b1, sa[i][0], sb[j]);
                        else mma_mx<3>(acc[i][j], a[i], b0, b1, sa[i][0], sb[j]);
                    } else {
                        if (ks == 0) mma_unit0(d[i][j], a[i], b0, b1);
                        else mma_unit(d[i][j], a[i], b0, b1);
                    }
                }
            }
        }
        if constexpr (MODE == FP8G) {                      // the block's sums, scaled into acc in block order
            const float ws = *reinterpret_cast<const float*>(ps);
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const float s0 = __uint_as_float(sa[i][0]) * ws, s1 = __uint_as_float(sa[i][1]) * ws;
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] = __fmaf_rn(d[i][j][e], e < 2 ? s0 : s1, acc[i][j][e]);
            }
        }
    }
    wait<0>();
    __syncthreads();
    if constexpr (FUSE) {                                  // the last slice; one slice: acc is the sum already
        if (SK > 1) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] = tot[i][j][e] + acc[i][j][e];
        }
    }
    if constexpr (CLUSTER) {                               // qmmf's: slice 0 adds its peers' sums in slice order
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
        if (slice != 0) {
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) mine[((i * T::NT + j) * 4 + e) * T::THREADS + tid] = acc[i][j][e];
        }
        cluster.sync();
        if (slice == 0) {
            for (int peer = 1; peer < SK; ++peer) {
                const float* theirs = cluster.map_shared_rank(mine, peer);
#pragma unroll
                for (int i = 0; i < T::MT; ++i)
#pragma unroll
                    for (int j = 0; j < T::NT; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[i][j][e] = acc[i][j][e] + theirs[((i * T::NT + j) * 4 + e) * T::THREADS + tid];
            }
        }
        cluster.sync();
        if (slice != 0) return;
    }
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (F32) {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0)
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
}

template <int MODE, int BM, int WM, int WN, int STAGES, bool QUANT, bool F32, bool CLUSTER>
void launch(const void* x, const void* xs, const zs::View& w, const zs::View& bs, zs::View& out, int M, int N, int K,
            int SK, int ldx, float input_scale) {
    using T = A8Tile<MODE, BM, WM, WN, STAGES, QUANT>;
    constexpr bool FUSE = !CLUSTER;
    auto kernel = qmma8_kernel<MODE, BM, WM, WN, STAGES, QUANT, F32, CLUSTER, FUSE>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        configured = true;
    }
    cudaLaunchConfig_t config = {};
    const int rows_t = (M + BM - 1) / BM;
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K))));
    config.gridDim = dim3(rows_t * ((N + BN - 1) / BN), 1, CLUSTER ? SK : 1);
    config.blockDim = dim3(T::THREADS);
    config.dynamicSmemBytes = T::SMEM;
    config.stream = zs::stream();
    cudaLaunchAttribute attr[1];
    if (CLUSTER) {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = SK;
        config.attrs = attr;
        config.numAttrs = 1;
    }
    ZS_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, x, xs, reinterpret_cast<const unsigned char*>(w.data_ptr()),
        reinterpret_cast<const uint8_t*>(bs.data_ptr()), out.data_ptr(), M, N, K, SK, ldx, group, input_scale));
}

template <int MODE, bool F32>
void by_form(int bm, bool xq16, bool cluster, const void* x, const void* xs, const zs::View& w, const zs::View& bs, zs::View& out,
             int M, int N, int K, int SK, int ldx, float input_scale) {
    if (bm == 16 && xq16) {
        if (cluster) launch<MODE, 16, 1, 4, 3, false, F32, true>(x, xs, w, bs, out, M, N, K, SK, ldx, input_scale);
        else launch<MODE, 16, 1, 4, 3, false, F32, false>(x, xs, w, bs, out, M, N, K, SK, ldx, input_scale);
    } else if (bm == 16) {
        if (cluster) launch<MODE, 16, 1, 4, 3, true, F32, true>(x, xs, w, bs, out, M, N, K, SK, ldx, input_scale);
        else launch<MODE, 16, 1, 4, 3, true, F32, false>(x, xs, w, bs, out, M, N, K, SK, ldx, input_scale);
    } else {
        if (cluster) launch<MODE, 64, 1, 4, 3, false, F32, true>(x, xs, w, bs, out, M, N, K, SK, ldx, input_scale);
        else launch<MODE, 64, 1, 4, 3, false, F32, false>(x, xs, w, bs, out, M, N, K, SK, ldx, input_scale);
    }
}

template <int MODE>
void quantize(const zs::View& x, uint8_t* xq, void* xs, int M, int K, int ldx, float input_scale) {
    const long long total = static_cast<long long>(M) * (K / 16);
    const int blocks = static_cast<int>((total + 255) / 256);
    quant_a8_kernel<MODE><<<blocks, 256, 0, zs::stream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), xq, xs,
                                                            M, K, ldx, input_scale);
    ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

extern "C" {

// out (M, n) = x (M, K) bf16, quantized as vLLM's (MX mode 2: e4m3 under an e8m0 a 32 inputs; block FP8 mode 3:
// e4m3 under an fp32 a 128), times W on FP8 tensor cores. w e4m3 in qmmf's fragment order (npad x K bytes); bs MX:
// u8 [npad/64, K/64, 64, 2]; block FP8: fp32 [npad/64, K/64], a tile's one scale a 64-group. bm 16 (1 to 16 rows)
// quantizes in the block; bm 64 into `work` first, u8 of at least M * K * 33 / 32 bytes. sk K slices by
// shape, summed in slice order in the block, or by a cluster where `fuse` is 0: the same bits. A 128-row tile on
// 8 warps was measured and withdrawn: at 512 to 4,096 rows of GLM's shapes 9 to 50% slower than bm 64 (2 stages,
// 159 to 194 registers, or spills at two blocks an SM; spec 114 WS-04).
int zs_nvfp4_qmm_a8(const zs_view* x_, const zs_view* w_, const zs_view* bs_, const zs_view* out_,
                    const zs_view* work_, int64_t mode, int64_t n, int64_t sk, int64_t npad, int64_t bm, int fuse,
                    int f32, double input_scale) {
    zs::begin();
    const zs::View x(x_), w(w_), bs(bs_), work(work_);
    zs::View out(out_);
    ZS_REQUIRE(mode == MXFP8 || mode == FP8G, "mode: 2 (MXFP8) or 3 (block FP8)");
    ZS_REQUIRE(std::isfinite(input_scale) && input_scale >= 0 && (mode == FP8G || input_scale == 0),
               "input_scale: zero for dynamic quantization, positive for per-tensor FP8");
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2, "x: (M, K) bf16");
    const int64_t m = x.size(0), k = x.size(1);
    ZS_REQUIRE(m >= 1 && (bm == 16 || bm == 64) && (bm != 16 || m <= 16), "bm 16 (1 to 16 rows) or 64");
    ZS_REQUIRE(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (m == 1 || x.stride(0) % 8 == 0),
               "x rows must start on 16-byte boundaries");
    ZS_REQUIRE(sk >= 1 && sk <= 8 && k % GS == 0 && (k / GS) % sk == 0, "K in whole stages of 128, split evenly in 1 to 8");
    ZS_REQUIRE(n >= 1 && npad % BN == 0 && n <= npad, "n within npad, a multiple of 64");
    ZS_REQUIRE(w.defined() && w.numel() * w.element_size() == npad * k, "weight bytes do not match npad and K");
    if (mode == MXFP8)
        ZS_REQUIRE(bs.defined() && bs.numel() * bs.element_size() == (npad / 64) * (k / 64) * 128, "bs: u8 [npad/64, K/64, 64, 2]");
    else
        ZS_REQUIRE(bs.defined() && bs.dtype() == ZS_F32 && bs.numel() == (npad / 64) * (k / 64), "bs: [npad/64, K/64] fp32");
    ZS_REQUIRE(out.defined() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
               out.dtype() == (f32 ? ZS_F32 : ZS_BF16), "out: (M, n)");
    ZS_REQUIRE((bm == 16 && !work.defined()) ||
                   (work.defined() && work.numel() * work.element_size() >= m * k + m * k / 32 &&
                    reinterpret_cast<uintptr_t>(work.data_ptr()) % 16 == 0),
               "work: M * K * 33 / 32 bytes on a 16-byte boundary");
    const bool cluster = sk > 1 && fuse == 0;
    ZS_REQUIRE(!cluster || zs::device().major >= 9, "K slices by a cluster need sm_90 or later");
    const int M = static_cast<int>(m), N = static_cast<int>(n), K = static_cast<int>(k), SK = static_cast<int>(sk);
    const int ldx = m == 1 ? K : static_cast<int>(x.stride(0));
    const void* xa = x.data_ptr();
    const void* xs = nullptr;
    bool xq16 = false;
    if (work.defined() && (bm != 16 || (mode == FP8G && input_scale > 0))) {
        uint8_t* xq = reinterpret_cast<uint8_t*>(work.data_ptr());
        uint8_t* scales = xq + static_cast<size_t>(m) * k;
        if (mode == MXFP8) quantize<MXFP8>(x, xq, scales, M, K, ldx, 0);
        else quantize<FP8G>(x, xq, scales, M, K, ldx, static_cast<float>(input_scale));
        xa = xq;
        xs = scales;
        xq16 = true;
    }
    if (mode == MXFP8) {
        if (f32) by_form<MXFP8, true>(static_cast<int>(bm), xq16, cluster, xa, xs, w, bs, out, M, N, K, SK, ldx, 0);
        else by_form<MXFP8, false>(static_cast<int>(bm), xq16, cluster, xa, xs, w, bs, out, M, N, K, SK, ldx, 0);
    } else {
        if (f32) by_form<FP8G, true>(static_cast<int>(bm), xq16, cluster, xa, xs, w, bs, out, M, N, K, SK, ldx, static_cast<float>(input_scale));
        else by_form<FP8G, false>(static_cast<int>(bm), xq16, cluster, xa, xs, w, bs, out, M, N, K, SK, ldx, static_cast<float>(input_scale));
    }
    return zs::status();
}

}  // extern "C"
