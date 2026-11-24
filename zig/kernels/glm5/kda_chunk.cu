// GLM-5.3-Flash's KDA over a prompt's window, chunked (spec 116 WS-06): zspark's own, after FLA's `chunk_kda`
// (vLLM's Triton fallback, `fla/ops/kda`), the delta rule in its WY form over chunks of 64 rows on absolute
// positions. Compile with --fmad=false, as kda.cu.
//
// The window's rows are cut into chunks [b, b + 64) with b a multiple of 64 of the sequence's positions, the
// first one from the window's first row (the caller says where it stands in its chunk: `phase`, pos % 64). In a
// chunk with the state S_b before it (s[v][k], value-major, as kda.cu), G the cumulative sum of the log decay
// (fp32, row by row from the chunk's first), per key channel:
//   A_kk[i][j] = beta_i sum_c k_ic k_jc e^(G_ic - G_jc), j < i;  A_qk[i][j] = sum_c q_ic k_jc e^(G_ic - G_jc), j <= i
//   U = (I + A_kk)^-1 (beta v), W = (I + A_kk)^-1 (beta k e^G): one forward substitution in fp32 a column
//   v_new = U - W S_b;  o = (q e^G) S_b + A_qk v_new;  S = e^G_last S_b + (k e^(G_last - G))^T v_new
// The conv, SiLU, norms, decay and beta are kda.cu's chain's, the same arithmetic (its q, k and v bits). The
// products e^(G_i - G_j) are taken as e^(G_i - G_r) e^(G_r - G_j) with r the first row of i's block of 16,
// fp32 on the CUDA cores (FLA takes the blocks below the diagonal through tf32 dots); the solve is fp32 (FLA's
// inverse is rounded to bf16); U, W, q e^G, k e^(G_last - G), A_qk and v_new are bf16 into m16n8k16 with fp32
// sums, the state fp32 across chunks and bf16 as an operand, as FLA's. o is rounded to bf16 into kda.cu's
// gated RMSNorm (`out_kernel`, moved here).
//
// A row's bits: every tile has the shape of a whole chunk, a row past the chunk's last (or past the window's)
// is zeros, and a row's sums over the chunk's rows meet zeros for every row after it. So row i's output depends
// on the state at its chunk's start and the chunk's rows up to i, and the state after a chunk on its rows: the
// same bits whether the window ends at the chunk or runs on, as long as windows are cut at multiples of 64
// (`decode/draft.zig`'s alignment). Windows of up to 48 rows keep kda.cu's chain and its steps' bits.
//
// Two kernels a window:
//   intra_kernel, block (chunk, head), 256 threads: a chunk's conv, norms, decay, A_kk and A_qk, the solve; it
//     writes U, -W, q e^G, k e^(G_last - G), A_qk (bf16) and e^G_last (fp32) for the chunk.
//   state_kernel, block (head, half of the values), 4 warps of 16 value rows: the chunks in order, the state's
//     rows a warp's in mma accumulators, read back as the A fragments of the next products; the next chunk's
//     operands are asked into L2 while one is summed (0.14 ms of 1.3 at 4,096 rows). Where a mark is asked at a
//     chunk's start inside the window (`Asked`, spec 116), the state it holds there is written out before the
//     chunk's products: the bits a window cut at that chunk leaves, with no chunk run twice.
// At 4,096 rows and 32 heads (a rank's): intra 3.07 ms, state 1.16, out 0.46 (nsys); intra reads its conv inputs
// 16 rows ahead (`conv_column`: 5.8 ms when each row was read once a tap, its loads one at a time). It is one
// block an SM, 8 warps, at 0.26 issued a scheduler: what is left is latency (the exponents, the barriers).
//
// Shared memory, intra_kernel (99,200 B of the 101,376 a block may take: one block an SM):
//   qa, ka     2 x 16,384  q's and k's conv output (bf16, its bits) of the chunk's 64 rows: every product reads
//                          them; bf16 is what they are, so no less
//   G          32,768      the decay's cumulative sum fp32 [64][128]: every exponent; fp32 is the floor
//   L           8,064      A_kk's strict lower triangle fp32, packed: the solve's matrix
//   tq, tk     2 x 8,192   a block of 16 rows of q and k times e^(G_i - G_r): the i side of a block's products
//   tj          8,448      a block of 16 rows of k times e^(G_r - G_j), rows padded to 132 floats so 16 lanes
//                          reading 16 rows by float4 meet no bank twice
//   invq, invk, beta 768
// state_kernel (71,168 B, one block an SM, 64 blocks of 4 warps):
//   ws, qs, ks 3 x 17,408  -W, q e^G, k e^(G_last - G) bf16 [64][136]: rows padded to 272 B so ldmatrix's
//                          eight rows fall in eight bank groups; each read by all four warps
//   as         9,216       A_qk [64][72]
//   us         9,216       U's 64 values of this block [64][72]
//   gs           512       e^G_last fp32 [128]
// Each region is a chunk's operand, staged once and read by four warps; a second stage would not fit.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../common/copies.cuh"
#include "../common/math.cuh"
#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using namespace common_copies;

using namespace common_math;

constexpr int DK = 128, DV = 128, TAPS = 4, BT = 64, SB = 16;
constexpr int TJ = 132;                                  // tj's row, floats
constexpr int RS = 136, RA = 72;                         // state_kernel's rows, bf16
constexpr int CHUNK_ELEMS = 4 * BT * DK + BT * BT;       // a chunk-head's bf16 scratch: U, -W, q e^G, k e^(.), A_qk
constexpr int INTRA_SMEM = 2 * BT * DK * 2 + BT * DK * 4 + (BT * (BT - 1) / 2) * 4 + 2 * SB * DK * 4 + SB * TJ * 4 + 3 * BT * 4;
constexpr int STATE_SMEM = 3 * BT * RS * 2 + 2 * BT * RA * 2 + DK * 4;
constexpr int MOST_ASKED = 64;                           // a 4,096-row window's chunk starts past its first

// The states asked inside a window: before chunk `chunk[i]` (rising), into `to[i]` fp32 [H, 128, 128].
struct Asked {
    int n;
    int chunk[MOST_ASKED];
    float* to[MOST_ASKED];
};

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&b);
}

// A chunk's rows of the window: [first, first + count).
__device__ __forceinline__ void chunk_rows(int c, int c0, int rows, int& first, int& count) {
    first = c == 0 ? 0 : c0 + (c - 1) * BT;
    const int end = c == 0 ? c0 : first + BT;
    count = (end < rows ? end : rows) - first;
}

// The packed lower triangle's row i: entries j < i.
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

__device__ __forceinline__ int tri(int i) { return i * (i - 1) / 2; }

// The conv over one channel `ch` of the chunk's n rows from r0, kda.cu's taps in its order: acc over taps 0 .. 3
// of w[tap] x[r + tap], x the conv state's three rows then the window's. The inputs are read 16 rows ahead of
// the sums (a row once, not once a tap), so a thread's loads are in flight together; `put(i, acc)` takes row
// i's sum, rows from n on are not summed.
template <class Put>
__device__ __forceinline__ void conv_column(const __nv_bfloat16* __restrict__ cs, int C,
                                            const __nv_bfloat16* __restrict__ P, int p_stride, int ch,
                                            const float (&w)[TAPS], int r0, int n, Put put) {
    auto input = [&](int at) {
        return at < TAPS - 1 ? __bfloat162float(cs[(size_t)at * C + ch])
                             : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * p_stride + ch]);
    };
    const __nv_bfloat16* row = P + (size_t)r0 * p_stride + ch;   // x[r0 + i + 3] is the window's row r0 + i
    float win[TAPS - 1];
#pragma unroll
    for (int tap = 0; tap < TAPS - 1; ++tap) win[tap] = input(r0 + tap);
#pragma unroll
    for (int i0 = 0; i0 < BT; i0 += SB) {
        float in[SB];
#pragma unroll
        for (int e = 0; e < SB; ++e) in[e] = i0 + e < n ? __bfloat162float(row[(size_t)(i0 + e) * p_stride]) : 0.0f;
#pragma unroll
        for (int e = 0; e < SB; ++e) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS - 1; ++tap) acc = acc + w[tap] * win[tap];
            acc = acc + w[TAPS - 1] * in[e];
            if (i0 + e < n) put(i0 + e, acc);
#pragma unroll
            for (int tap = 0; tap < TAPS - 2; ++tap) win[tap] = win[tap + 1];
            win[TAPS - 2] = in[e];
        }
    }
}

template <bool FULL>
__global__ void __launch_bounds__(256) intra_kernel(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A,
        int a_stride, const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower, int rows, int c0, int chunks,
        __nv_bfloat16* __restrict__ scratch, float* __restrict__ gl) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* qa = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* ka = qa + BT * DK;
    float* G = reinterpret_cast<float*>(ka + BT * DK);
    float* L = G + BT * DK;
    float* tq = L + BT * (BT - 1) / 2;
    float* tk = tq + SB * DK;
    float* tj = tk + SB * DK;
    float* invq = tj + SB * TJ;
    float* invk = invq + BT;
    float* beta = invk + BT;

    const int C = 3 * H * DK;
    const int c = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    int r0, n;
    chunk_rows(c, c0, rows, r0, n);
    const size_t blk = (size_t)h * chunks + c, NB = (size_t)H * chunks;
    __nv_bfloat16* u_out = scratch + blk * BT * DK;
    __nv_bfloat16* w_out = scratch + NB * BT * DK + blk * BT * DK;
    __nv_bfloat16* q_out = scratch + 2 * NB * BT * DK + blk * BT * DK;
    __nv_bfloat16* k_out = scratch + 3 * NB * BT * DK + blk * BT * DK;
    __nv_bfloat16* a_out = scratch + 4 * NB * BT * DK + blk * BT * BT;

    // the conv of q (threads 0 .. 127) and k (128 .. 255): kda.cu's taps over [conv state; rows]
    {
        const int ch = (t / DK) * H * DK + h * DK + (t % DK);
        float w[TAPS];
#pragma unroll
        for (int tap = 0; tap < TAPS; ++tap) w[tap] = __bfloat162float(cw[(size_t)ch * TAPS + tap]);
        __nv_bfloat16* dst = t < DK ? qa : ka;
        const int col = t % DK;
        for (int i = n; i < BT; ++i) dst[i * DK + col] = __float2bfloat16_rn(0.0f);
        conv_column(cs, C, P, p_stride, ch, w, r0, n,
                    [&](int i, float acc) { dst[i * DK + col] = __float2bfloat16_rn(acc / (1.0f + expf(-acc))); });
    }
    // the decay's log, summed row by row (threads 0 .. 127, a channel each), and beta (128 .. 191)
    if (t < DK) {
        const float rate = expf(a_log[h]), bias = dt_bias[h * DK + t];
        float g = 0.0f;
#pragma unroll
        for (int i0 = 0; i0 < BT; i0 += SB) {
            float in[SB];
#pragma unroll
            for (int e = 0; e < SB; ++e) in[e] = i0 + e < n ? __bfloat162float(A[(size_t)(r0 + i0 + e) * a_stride + h * DK + t]) : 0.0f;
#pragma unroll
            for (int e = 0; e < SB; ++e) {
                if (i0 + e < n) g = g + lower * sigmoidf_(rate * (in[e] + bias));
                G[(i0 + e) * DK + t] = g;                   // rows past the chunk keep its last: no exponent grows
            }
        }
    } else if (t < DK + BT) {
        const int i = t - DK;
        beta[i] = i < n ? bf(sigmoidf_(__bfloat162float(P[(size_t)(r0 + i) * p_stride + b_off + h]))) : 0.0f;
    }
    __syncthreads();
    // the norms: kda.cu's sums (a lane's four, then the warp's tree), a warp eight rows
    for (int i = warp; i < BT; i += 8) {
#pragma unroll
        for (int which = 0; which < 2; ++which) {
            const __nv_bfloat16* x = (which == 0 ? qa : ka) + i * DK + lane * 4;
            float ss = 0.0f;
#pragma unroll
            for (int e = 0; e < 4; ++e) { const float v = __bfloat162float(x[e]); ss = ss + v * v; }
            ss = warp_sum(ss);
            if (lane == 0) (which == 0 ? invq : invk)[i] = 1.0f / sqrtf(ss + 1e-6f);
        }
    }
    __syncthreads();
    const float qscale = 1.0f / sqrtf((float)DK);
    auto qn = [&](int i, int ch) { return (__bfloat162float(qa[i * DK + ch]) * invq[i]) * qscale; };
    auto kn = [&](int i, int ch) { return __bfloat162float(ka[i * DK + ch]) * invk[i]; };

    // A_kk into L, A_qk to global, a block of 16 rows at a time against the blocks at or before it
    for (int I = 0; I < BT / SB; ++I) {
        const int r = I * SB;
        const int ii = t >> 4, jj = t & 15, i = r + ii;
        if (r < n) {
            for (int e = t; e < SB * DK; e += 256) {
                const int row = r + e / DK, ch = e % DK;
                const float f = expf(G[row * DK + ch] - G[r * DK + ch]);
                if (FULL) tq[e] = qn(row, ch) * f;
                tk[e] = kn(row, ch) * f;
            }
        }
        for (int J = 0; J < BT / SB; ++J) {
            const int j = J * SB + jj;
            const bool live = r < n && J <= I;
            if (live) {
                __syncthreads();
                for (int e = t; e < SB * DK; e += 256) {
                    const int row = J * SB + e / DK, ch = e % DK;
                    tj[(e / DK) * TJ + ch] = kn(row, ch) * expf(G[r * DK + ch] - G[row * DK + ch]);
                }
                __syncthreads();
            }
            float aq = 0.0f, ak = 0.0f;
            if (live) {
                const float4* xq = reinterpret_cast<const float4*>(tq + ii * DK);
                const float4* xk = reinterpret_cast<const float4*>(tk + ii * DK);
                const float4* y = reinterpret_cast<const float4*>(tj + jj * TJ);
#pragma unroll 8
                for (int e = 0; e < DK / 4; ++e) {
                    const float4 b = y[e];
                    if (FULL) {
                        const float4 a = xq[e];
                        aq = fmaf(a.x, b.x, aq); aq = fmaf(a.y, b.y, aq); aq = fmaf(a.z, b.z, aq); aq = fmaf(a.w, b.w, aq);
                    }
                    const float4 a = xk[e];
                    ak = fmaf(a.x, b.x, ak); ak = fmaf(a.y, b.y, ak); ak = fmaf(a.z, b.z, ak); ak = fmaf(a.w, b.w, ak);
                }
            }
            const bool in = i < n && j < n;
            if (FULL) a_out[i * BT + j] = __float2bfloat16_rn(live && in && (J < I || jj <= ii) ? aq : 0.0f);
            if (j < i) L[tri(i) + j] = live && in ? beta[i] * ak : 0.0f;
        }
        __syncthreads();
    }

    // the solve, a column a thread: U's (v, threads 0 .. 127), W's (k, 128 .. 255); rows past n are zeros
    {
        float x[BT];
        if (t < DV) {
            const int ch = 2 * H * DK + h * DK + t;
            float w[TAPS];
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) w[tap] = __bfloat162float(cw[(size_t)ch * TAPS + tap]);
#pragma unroll
            for (int i = 0; i < BT; ++i) x[i] = 0.0f;
            conv_column(cs, C, P, p_stride, ch, w, r0, n,
                        [&](int i, float acc) { x[i] = beta[i] * bf(acc / (1.0f + expf(-acc))); });
        } else {
            const int ch = t - DV;
#pragma unroll
            for (int i = 0; i < BT; ++i) x[i] = i < n ? beta[i] * (kn(i, ch) * expf(G[i * DK + ch])) : 0.0f;
        }
        // x_i = b_i - sum_{j < i} L_ij x_j, the sum in four strands by j % 4, then (s0 + s1) + (s2 + s3)
#pragma unroll
        for (int i = 1; i < BT; ++i) {
            float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float* l = L + tri(i);
#pragma unroll
            for (int j = 0; j < i; ++j) s[j & 3] = fmaf(l[j], x[j], s[j & 3]);
            x[i] = x[i] - ((s[0] + s[1]) + (s[2] + s[3]));
        }
        if (t < DV) {
#pragma unroll
            for (int i = 0; i < BT; ++i) u_out[i * DK + t] = __float2bfloat16_rn(x[i]);
        } else {
#pragma unroll
            for (int i = 0; i < BT; ++i) w_out[i * DK + t - DV] = __float2bfloat16_rn(-x[i]);
        }
    }
    // q e^G (threads 0 .. 127), k e^(G_last - G) and e^G_last (128 .. 255)
    if (t < DK) {
        if (FULL)
            for (int i = 0; i < BT; ++i)
                q_out[i * DK + t] = __float2bfloat16_rn(i < n ? qn(i, t) * expf(G[i * DK + t]) : 0.0f);
    } else {
        const int ch = t - DK;
        const float last = G[(n - 1) * DK + ch];
        for (int i = 0; i < BT; ++i)
            k_out[i * DK + ch] = __float2bfloat16_rn(i < n ? kn(i, ch) * expf(last - G[i * DK + ch]) : 0.0f);
        gl[blk * DK + ch] = expf(last);
    }
}

// Block (head, half): warp w holds the state's value rows v0 + 16 w .. + 15 over all 128 keys, fp32, as the
// accumulators of 16 m16n8 tiles (lane (g, q): rows g and g + 8, keys 8 n + 2 q and + 1).
template <bool FULL>
__global__ void __launch_bounds__(128) state_kernel(
        int H, int rows, int c0, int chunks, const float* __restrict__ state_in, float* __restrict__ state_out,
        const __nv_bfloat16* __restrict__ scratch, const float* __restrict__ gl, __nv_bfloat16* __restrict__ y,
        const __grid_constant__ Asked asked) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* ws = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* qs = ws + BT * RS;
    __nv_bfloat16* ks = qs + BT * RS;
    __nv_bfloat16* as = ks + BT * RS;
    __nv_bfloat16* us = as + BT * RA;
    float* gs = reinterpret_cast<float*>(us + BT * RA);

    const int h = blockIdx.x, v0 = blockIdx.y * 64;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31, g = lane >> 2, q = lane & 3;
    const int vw = v0 + warp * 16;                         // this warp's first value row
    const size_t NB = (size_t)H * chunks;
    const size_t sbase = (size_t)h * DV * DK;

    float s[16][4];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int k = nt * 8 + 2 * q;
        s[nt][0] = state_in[sbase + (size_t)(vw + g) * DK + k];
        s[nt][1] = state_in[sbase + (size_t)(vw + g) * DK + k + 1];
        s[nt][2] = state_in[sbase + (size_t)(vw + g + 8) * DK + k];
        s[nt][3] = state_in[sbase + (size_t)(vw + g + 8) * DK + k + 1];
    }

    int next = 0;
    for (int c = 0; c < chunks; ++c) {
        if (next < asked.n && asked.chunk[next] == c) {     // the state before this chunk, a mark's
            float* to = asked.to[next++];
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const int k = nt * 8 + 2 * q;
                to[sbase + (size_t)(vw + g) * DK + k] = s[nt][0];
                to[sbase + (size_t)(vw + g) * DK + k + 1] = s[nt][1];
                to[sbase + (size_t)(vw + g + 8) * DK + k] = s[nt][2];
                to[sbase + (size_t)(vw + g + 8) * DK + k + 1] = s[nt][3];
            }
        }
        int r0, n;
        chunk_rows(c, c0, rows, r0, n);
        const size_t blk = (size_t)h * chunks + c;
        const __nv_bfloat16* u_in = scratch + blk * BT * DK;
        const __nv_bfloat16* w_in = scratch + NB * BT * DK + blk * BT * DK;
        const __nv_bfloat16* q_in = scratch + 2 * NB * BT * DK + blk * BT * DK;
        const __nv_bfloat16* k_in = scratch + 3 * NB * BT * DK + blk * BT * DK;
        const __nv_bfloat16* a_in = scratch + 4 * NB * BT * DK + blk * BT * BT;
        __syncthreads();                                   // the last chunk's reads are done
        for (int e = t; e < BT * DK / 8; e += 128) {        // 16 bytes a copy
            const int row = e / (DK / 8), seg = e % (DK / 8);
            qmm_frag::cp16(ws + row * RS + seg * 8, w_in + row * DK + seg * 8);
            qmm_frag::cp16(ks + row * RS + seg * 8, k_in + row * DK + seg * 8);
            if (FULL) qmm_frag::cp16(qs + row * RS + seg * 8, q_in + row * DK + seg * 8);
        }
        for (int e = t; e < BT * 64 / 8; e += 128) {
            const int row = e / 8, seg = e % 8;
            qmm_frag::cp16(us + row * RA + seg * 8, u_in + row * DK + v0 + seg * 8);
            if (FULL) qmm_frag::cp16(as + row * RA + seg * 8, a_in + row * BT + seg * 8);
        }
        if (t < DK / 4) qmm_frag::cp16(gs + t * 4, gl + blk * DK + t * 4);
        qmm_frag::commit();
        if (c + 1 < chunks) {                              // the next chunk's operands on their way to L2
            const size_t next = blk + 1;
            for (int e = t; e < 4 * BT; e += 128) {        // 128-byte lines: W, k, q a row's two, U's half one
                const int row = e >> 2, part = e & 3;
                const __nv_bfloat16* line =
                        part < 3 ? scratch + (size_t)(part == 0 ? 1 : part == 1 ? 3 : 2) * NB * BT * DK + next * BT * DK + row * DK
                                 : scratch + next * BT * DK + row * DK + v0;
                prefetch_l2(line);
                if (part < 3) prefetch_l2(line + 64);
            }
            if (FULL && t < BT) prefetch_l2(scratch + 4 * NB * BT * DK + next * BT * BT + t * BT);
        }
        qmm_frag::wait<0>();
        __syncthreads();

        // the state as bf16 A fragments over k
        uint32_t sa[8][4];
#pragma unroll
        for (int ks_ = 0; ks_ < 8; ++ks_) {
            sa[ks_][0] = pack(s[2 * ks_][0], s[2 * ks_][1]);
            sa[ks_][1] = pack(s[2 * ks_][2], s[2 * ks_][3]);
            sa[ks_][2] = pack(s[2 * ks_ + 1][0], s[2 * ks_ + 1][1]);
            sa[ks_][3] = pack(s[2 * ks_ + 1][2], s[2 * ks_ + 1][3]);
        }
        // v_new^T [16 values][64 rows] = U^T + S^T (-W)^T
        float vn[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int j = nt * 8 + 2 * q, v = warp * 16 + g;
            vn[nt][0] = __bfloat162float(us[j * RA + v]);
            vn[nt][1] = __bfloat162float(us[(j + 1) * RA + v]);
            vn[nt][2] = __bfloat162float(us[j * RA + v + 8]);
            vn[nt][3] = __bfloat162float(us[(j + 1) * RA + v + 8]);
        }
#pragma unroll
        for (int ks_ = 0; ks_ < 8; ++ks_) {
#pragma unroll
            for (int nt = 0; nt < 8; nt += 2) {
                const int m = lane >> 3;
                uint32_t b[4];
                qmm_frag::ldmatrix4(b, ws + (nt * 8 + (m >> 1) * 8 + (lane & 7)) * RS + ks_ * 16 + (m & 1) * 8);
                qmm_frag::mma(vn[nt], sa[ks_], b[0], b[1]);
                qmm_frag::mma(vn[nt + 1], sa[ks_], b[2], b[3]);
            }
        }
        uint32_t va[4][4];
#pragma unroll
        for (int kj = 0; kj < 4; ++kj) {
            va[kj][0] = pack(vn[2 * kj][0], vn[2 * kj][1]);
            va[kj][1] = pack(vn[2 * kj][2], vn[2 * kj][3]);
            va[kj][2] = pack(vn[2 * kj + 1][0], vn[2 * kj + 1][1]);
            va[kj][3] = pack(vn[2 * kj + 1][2], vn[2 * kj + 1][3]);
        }
        if (FULL) {
            // o^T [16 values][64 rows] = S^T (q e^G)^T + v_new^T A_qk^T
            float o[8][4];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
#pragma unroll
            for (int ks_ = 0; ks_ < 8; ++ks_) {
#pragma unroll
                for (int nt = 0; nt < 8; nt += 2) {
                    const int m = lane >> 3;
                    uint32_t b[4];
                    qmm_frag::ldmatrix4(b, qs + (nt * 8 + (m >> 1) * 8 + (lane & 7)) * RS + ks_ * 16 + (m & 1) * 8);
                    qmm_frag::mma(o[nt], sa[ks_], b[0], b[1]);
                    qmm_frag::mma(o[nt + 1], sa[ks_], b[2], b[3]);
                }
            }
#pragma unroll
            for (int kj = 0; kj < 4; ++kj) {
#pragma unroll
                for (int nt = 0; nt < 8; nt += 2) {
                    if (kj * 16 > nt * 8 + 15) continue;     // both tiles' rows are before every j of the step
                    const int m = lane >> 3;
                    uint32_t b[4];
                    qmm_frag::ldmatrix4(b, as + (nt * 8 + (m >> 1) * 8 + (lane & 7)) * RA + kj * 16 + (m & 1) * 8);
                    qmm_frag::mma(o[nt], va[kj], b[0], b[1]);
                    qmm_frag::mma(o[nt + 1], va[kj], b[2], b[3]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int i = nt * 8 + 2 * q, v = vw + g;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int row = i + (e & 1), col = v + (e >> 1) * 8;
                    if (row < n) y[((size_t)(r0 + row) * H + h) * DV + col] = __float2bfloat16_rn(o[nt][e]);
                }
            }
        }
        // S^T = S^T e^G_last + v_new^T (k e^(G_last - G))
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const float g0 = gs[nt * 8 + 2 * q], g1 = gs[nt * 8 + 2 * q + 1];
            s[nt][0] = s[nt][0] * g0;
            s[nt][1] = s[nt][1] * g1;
            s[nt][2] = s[nt][2] * g0;
            s[nt][3] = s[nt][3] * g1;
        }
#pragma unroll
        for (int kj = 0; kj < 4; ++kj) {
#pragma unroll
            for (int nt = 0; nt < 16; nt += 2) {
                const int m = lane >> 3;
                uint32_t b[4];
                ldmatrix4t(b, ks + (kj * 16 + (m & 1) * 8 + (lane & 7)) * RS + nt * 8 + (m >> 1) * 8);
                qmm_frag::mma(s[nt], va[kj], b[0], b[1]);
                qmm_frag::mma(s[nt + 1], va[kj], b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int k = nt * 8 + 2 * q;
        state_out[sbase + (size_t)(vw + g) * DK + k] = s[nt][0];
        state_out[sbase + (size_t)(vw + g) * DK + k + 1] = s[nt][1];
        state_out[sbase + (size_t)(vw + g + 8) * DK + k] = s[nt][2];
        state_out[sbase + (size_t)(vw + g + 8) * DK + k + 1] = s[nt][3];
    }
}

// Block (row, head), 128 threads: the gated RMSNorm of the read-out (kda.cu's chain's, from its rows).
__global__ void __launch_bounds__(128) out_kernel(
        int H, const __nv_bfloat16* __restrict__ y_in, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float rinv;
    const size_t base = ((size_t)r * H + h) * DV;
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float y = __bfloat162float(y_in[base + lane * 4 + i]); ss = ss + y * y; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = __bfloat162float(y_in[base + t]) * rinv;
    const float yw = __bfloat162float(norm_w[t]) * yn;
    const float gate = __bfloat162float(G[(size_t)r * g_stride + h * DV + t]);
    out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
}

template <typename T>
T* ptr(const zs::View& x) { return x.defined() && x.numel() ? x.data_ptr<T>() : nullptr; }

bool holds(const zs::View& x, int dtype) { return x.defined() && x.is_contiguous() && x.dtype() == dtype; }

bool rows_of(const zs::View& x, int64_t rows, int64_t width) {
    return x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.size(0) >= rows && x.size(1) >= width;
}

}  // namespace

extern "C" {

// The chunks a window of `rows` rows from a position `phase` past a multiple of 64 is cut into.
int64_t zs_glm5_kda_chunks(int64_t rows, int64_t phase) {
    const int64_t c0 = BT - phase % BT;
    return rows <= c0 ? 1 : 1 + (rows - c0 + BT - 1) / BT;
}

// The bf16 scratch elements of a chunk and a head (U, -W, q e^G, k e^(G_last - G), A_qk); fp32 e^G_last is 128.
int64_t zs_glm5_kda_chunk_elems() { return CHUNK_ELEMS; }

// `zs_glm5_kda_chain`'s window chunked: the same operands but no saves, `phase` the window's first position
// modulo 64, `scratch` bf16 and `last` fp32 for zs_glm5_kda_chunks(rows, phase) x H chunks (zs_glm5_kda_chunk_elems
// and 128 elements each), `y_tmp` bf16 [rows, H, 128]. Without `out` only the state after the rows is written:
// no read-out, nor the products that only it needs. `asked` (n of them, at most zs_glm5_kda_most_asked) are the
// states after the window's first `asked_rows[i]` rows, rising, each a chunk's start inside the window, into
// `asked_to[i]` fp32 [H, 128, 128].
int64_t zs_glm5_kda_most_asked() { return MOST_ASKED; }

int zs_glm5_kda_chunked(const zs_view* p_, int64_t b_off, const zs_view* a_, const zs_view* g_, const zs_view* cs_,
                        const zs_view* cw_, const zs_view* state_in_, const zs_view* a_log_, const zs_view* dt_bias_,
                        const zs_view* norm_w_, double eps, double lower, int64_t rows, int64_t phase,
                        const zs_view* out_, const zs_view* state_out_, const zs_view* scratch_, const zs_view* last_,
                        const zs_view* y_tmp_, int64_t asked_n, const int64_t* asked_rows, const zs_view* asked_to) {
    zs::begin();
    const zs::View P(p_), A(a_), G(g_), cs(cs_), cw(cw_), state_in(state_in_), a_log(a_log_), dt_bias(dt_bias_);
    const zs::View norm_w(norm_w_), out(out_), state_out(state_out_), scratch(scratch_), last(last_), y_tmp(y_tmp_);
    ZS_REQUIRE(holds(a_log, ZS_F32) && a_log.numel() >= 1 && a_log.numel() <= 1024, "A_log: fp32 [H]");
    const int64_t H = a_log.numel(), C = 3 * H * DK;
    ZS_REQUIRE(rows >= 1 && rows <= INT32_MAX / (H * DK) && phase >= 0 && phase < BT, "rows and phase");
    const int64_t chunks = zs_glm5_kda_chunks(rows, phase);
    ZS_REQUIRE(b_off >= C && rows_of(P, rows, b_off + H), "p: bf16 [rows, q | k | v | ... | b]");
    ZS_REQUIRE(rows_of(A, rows, H * DK), "a: bf16 [rows, H * 128]");
    ZS_REQUIRE(P.stride(0) <= INT32_MAX && A.stride(0) <= INT32_MAX, "row strides");
    ZS_REQUIRE(holds(cs, ZS_BF16) && cs.numel() == (TAPS - 1) * C && holds(cw, ZS_BF16) && cw.numel() == C * TAPS,
               "conv state [3, 3 H 128] and weight [3 H 128, 4]: contiguous bf16");
    ZS_REQUIRE(holds(state_in, ZS_F32) && state_in.numel() == H * DV * DK && holds(state_out, ZS_F32) &&
               state_out.numel() == H * DV * DK, "states: fp32 [H, 128, 128]");
    ZS_REQUIRE(holds(dt_bias, ZS_F32) && dt_bias.numel() == H * DK, "dt_bias fp32 [H * 128]");
    ZS_REQUIRE(holds(scratch, ZS_BF16) && scratch.numel() >= chunks * H * CHUNK_ELEMS && holds(last, ZS_F32) &&
               last.numel() >= chunks * H * DK, "chunk scratch: bf16 and fp32, a chunk and head each");
    const bool full = out.defined();
    if (full) {
        ZS_REQUIRE(rows_of(G, rows, H * DV) && G.stride(0) <= INT32_MAX, "g: bf16 [rows, H * 128]");
        ZS_REQUIRE(holds(norm_w, ZS_BF16) && norm_w.numel() == DV, "norm: bf16 [128]");
        ZS_REQUIRE(holds(out, ZS_BF16) && out.numel() >= rows * H * DV, "out: bf16 [rows, H * 128]");
        ZS_REQUIRE(holds(y_tmp, ZS_BF16) && y_tmp.numel() >= rows * H * DV, "read-out: bf16 [rows, H, 128]");
    }
    const int c0 = (int)(BT - phase);
    ZS_REQUIRE(asked_n >= 0 && asked_n <= MOST_ASKED && (asked_n == 0 || (asked_rows && asked_to)), "asked states");
    Asked asked{};
    asked.n = (int)asked_n;
    for (int64_t i = 0; i < asked_n; ++i) {
        const int64_t at = asked_rows[i];
        ZS_REQUIRE(at >= c0 && at < rows && (at - c0) % BT == 0 && (i == 0 || at > asked_rows[i - 1]),
                   "an asked state: a chunk's start inside the window, rising");
        const zs::View to(&asked_to[i]);
        ZS_REQUIRE(holds(to, ZS_F32) && to.numel() == H * DV * DK, "an asked state: fp32 [H, 128, 128]");
        asked.chunk[i] = 1 + (int)((at - c0) / BT);
        asked.to[i] = to.data_ptr<float>();
    }
    static bool configured = false;
    if (!configured) {
        ZS_CUDA_CHECK(cudaFuncSetAttribute(intra_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, INTRA_SMEM));
        ZS_CUDA_CHECK(cudaFuncSetAttribute(intra_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, INTRA_SMEM));
        ZS_CUDA_CHECK(cudaFuncSetAttribute(state_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, STATE_SMEM));
        ZS_CUDA_CHECK(cudaFuncSetAttribute(state_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, STATE_SMEM));
        configured = true;
    }
    auto stream = zs::stream();
    const dim3 intra((unsigned)chunks, (unsigned)H);
    (full ? intra_kernel<true> : intra_kernel<false>)<<<intra, 256, INTRA_SMEM, stream>>>(
        (int)H, ptr<__nv_bfloat16>(P), (int)P.stride(0), (int)b_off, ptr<__nv_bfloat16>(A), (int)A.stride(0),
        ptr<__nv_bfloat16>(cs), ptr<__nv_bfloat16>(cw), ptr<float>(a_log), ptr<float>(dt_bias), (float)lower,
        (int)rows, c0, (int)chunks, ptr<__nv_bfloat16>(scratch), ptr<float>(last));
    ZS_CUDA_LAUNCH_CHECK();
    (full ? state_kernel<true> : state_kernel<false>)<<<dim3((unsigned)H, 2), 128, STATE_SMEM, stream>>>(
        (int)H, (int)rows, c0, (int)chunks, ptr<float>(state_in), ptr<float>(state_out), ptr<__nv_bfloat16>(scratch),
        ptr<float>(last), ptr<__nv_bfloat16>(y_tmp), asked);
    ZS_CUDA_LAUNCH_CHECK();
    if (full) {
        out_kernel<<<dim3((unsigned)rows, (unsigned)H), DV, 0, stream>>>(
            (int)H, ptr<__nv_bfloat16>(y_tmp), ptr<__nv_bfloat16>(G), (int)G.stride(0), ptr<__nv_bfloat16>(norm_w),
            (float)eps, ptr<__nv_bfloat16>(out));
        ZS_CUDA_LAUNCH_CHECK();
    }
    return zs::status();
}

}  // extern "C"
