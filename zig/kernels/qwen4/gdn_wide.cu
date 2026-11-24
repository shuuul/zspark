// Flash Next's Gated DeltaNet for a wide window, with the bits of gdn.cu's chain kernel: a front end over every row
// at once (conv, SiLU, the norms of q and k, the gates), a chain that holds only the delta rule, and a back end
// over every row at once (the gated RMSNorm). Compile with --fmad=false, as gdn.cu.
//
// The front and back ends are from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/gdn_io.cu, Apache-2.0, which
// keeps the chain kernel's operations and lane order; changed for zspark: ATen replaced by zs/host.h, one sequence
// (the conv window is [conv state; projection rows], no table of streams), and the back end works in place.
//
// The chain's layout is TensorFold's gdn_prefill.cu (a state row in the registers of two threads, a stage of 32
// rows' keys, queries and values loaded a stage ahead), but its arithmetic is not: gdn_prefill.cu contracts each
// product and sum into an FMA and sums a row in four strides, so its bits are a prompt's own. Here every sum is
// taken in the chain kernel's order. That kernel's thread (warp w, lane l) holds state rows 4w .. 4w + 3 and
// columns 4l .. 4l + 3, sums its four products from zero and then the 32 lanes with shfl_xor at 16, 8, 4, 2, 1.
// Addition commutes bit for bit, so after the first four levels a lane holds the sum over the lanes of its own
// parity, taken as a tree that pairs lane l with l ^ 16, then ^ 8, ^ 4, ^ 2; the last level adds the two parities.
// A thread here holds the 16 lanes of one parity of one state row, takes that tree in its registers and meets the
// other parity with one shfl_xor at 1.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../zs/host.h"
#include "gdn.cuh"

namespace {

using namespace qwen4_gdn;

// conv + SiLU of channel c for row r: taps from [conv state (3 rows) | projection rows], as the chain kernel
__device__ __forceinline__ float conv_act(const __nv_bfloat16* P, const __nv_bfloat16* cs, const __nv_bfloat16* cw,
                                          int r, int c, int C, int PW) {
    float acc = 0.0f;
#pragma unroll
    for (int tap = 0; tap < TAPS; ++tap) {
        const int at = r + tap;
        const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                      : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
        acc = acc + __bfloat162float(cw[c * TAPS + tap]) * x;
    }
    return bf(acc / (1.0f + expf(-acc)));
}

// Block (row, head): heads < NK L2-norm q (times DK^-0.5) and k as the chain kernel's warps 0 and 1; the others
// write a value head's v, g and beta.
template <int NK, int NV>
__global__ void __launch_bounds__(128) front_kernel(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        float* __restrict__ q, float* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ g,
        float* __restrict__ beta) {
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int r = blockIdx.x, head = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float xs[2][DK];
    if (head < NK) {
        xs[0][t] = conv_act(P, cs, cw, r, head * DK + t, C, PW);
        xs[1][t] = conv_act(P, cs, cw, r, NK * DK + head * DK + t, C, PW);
        __syncthreads();
        if (warp < 2) {
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = xs[warp][lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            float* out = (warp == 0 ? q : k) + ((size_t)r * NK + head) * DK + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) out[i] = v4[i] * inv;
        }
        return;
    }
    const int hv = head - NK;
    v[((size_t)r * NV + hv) * DV + t] = __float2bfloat16_rn(conv_act(P, cs, cw, r, 2 * NK * DK + hv * DV + t, C, PW));
    if (t == 0) {
        const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
        const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
        g[(size_t)r * NV + hv] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
        beta[(size_t)r * NV + hv] = bf(sigmoidf_(b));
    }
}

// Block (row, value head): the gated RMSNorm on warp 0's sum as the chain kernel, then bf16 out and its 32-channel
// group sums. ``y`` is the chain's read-out, already rounded to bf16, and is overwritten by the layer's output.
template <int NK, int NV>
__global__ void __launch_bounds__(128) back_kernel(
        __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ norm_w,
        float eps, float* __restrict__ xs) {
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int r = blockIdx.x, hv = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys[DV];
    __shared__ float rinv;
    ys[t] = __bfloat162float(y[((size_t)r * NV + hv) * DV + t]);
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float yy = ys[lane * 4 + i]; ss = ss + yy * yy; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
    const float z = __bfloat162float(P[(size_t)r * PW + C + hv * DV + t]);
    const float o = bf(yn * sigmoidf_(z));
    y[((size_t)r * NV + hv) * DV + t] = __float2bfloat16_rn(o);
    const float gs = warp_sum(o);
    if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
}

// A stage's keys, queries, gates and value rows, loaded into registers a stage ahead (in flight while the current
// stage computes) and then put where the chain reads them: keys and queries as float4s, values bf16.
template <int THREADS, int ROWS, int STEPS, bool OUT>
struct Prefetch {
    static constexpr int KQ = STEPS * (DK / 4) / THREADS, VV = STEPS * ROWS / 8 / THREADS;
    static_assert(KQ * THREADS == STEPS * (DK / 4) && VV * THREADS * 8 == STEPS * ROWS, "whole slots a thread");
    float4 kr[KQ], qr[KQ];
    uint4 vr[VV];
    float gr, br;

    __device__ __forceinline__ void fetch(const float* q, const float* k, const __nv_bfloat16* v, const float* g,
                                          const float* beta, int t0, int W, int hk, int hv, int head, int key_head,
                                          int col0) {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS, st = i / (DK / 4), c = i % (DK / 4);
            const size_t at = (static_cast<size_t>(min(t0 + st, W - 1)) * hk + key_head) * DK + 4 * c;
            kr[it] = *reinterpret_cast<const float4*>(k + at);
            if constexpr (OUT) qr[it] = *reinterpret_cast<const float4*>(q + at);
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int i = threadIdx.x + it * THREADS, st = i / (ROWS / 8), c = i % (ROWS / 8);
            vr[it] = *reinterpret_cast<const uint4*>(v + (static_cast<size_t>(min(t0 + st, W - 1)) * hv + head) * DV
                                                     + col0 + 8 * c);
        }
        if (threadIdx.x < STEPS) {
            const size_t at = static_cast<size_t>(min(t0 + static_cast<int>(threadIdx.x), W - 1)) * hv + head;
            gr = g[at];
            br = beta[at];
        }
    }

    __device__ __forceinline__ void store(float4 (*ks)[DK / 4], float4 (*qs)[DK / 4], float* gs, float* bs,
                                          __nv_bfloat16 (*vs)[ROWS]) const {
#pragma unroll
        for (int it = 0; it < KQ; ++it) {
            const int i = threadIdx.x + it * THREADS;
            ks[i / (DK / 4)][i % (DK / 4)] = kr[it];
            if constexpr (OUT) qs[i / (DK / 4)][i % (DK / 4)] = qr[it];
        }
#pragma unroll
        for (int it = 0; it < VV; ++it) {
            const int i = threadIdx.x + it * THREADS;
            *reinterpret_cast<uint4*>(&vs[i / (ROWS / 8)][8 * (i % (ROWS / 8))]) = vr[it];
        }
        if (threadIdx.x < STEPS) {
            gs[threadIdx.x] = gr;
            bs[threadIdx.x] = br;
        }
    }
};

// The 16 lanes of one parity summed as the chain kernel's warp_sum pairs them: lane l with l ^ 16, ^ 8, ^ 4, ^ 2,
// which are this thread's slots j ^ 8, ^ 4, ^ 2, ^ 1.
__device__ __forceinline__ float parity_sum(float (&p)[16]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) p[j] = p[j] + p[j + 8];
#pragma unroll
    for (int j = 0; j < 4; ++j) p[j] = p[j] + p[j + 4];
#pragma unroll
    for (int j = 0; j < 2; ++j) p[j] = p[j] + p[j + 2];
    return p[0] + p[1];
}

// Block (value head, ROWS of its state rows), two threads a state row: thread 2 * row + half holds the columns
// 8j + 4 * half .. + 3 for j of 0 .. 15, the chain kernel's lanes 2j + half. W rows of the window from ``state``;
// with OUT each row's read-out goes to ``y`` (bf16, as the chain kernel rounds it), without it only the state
// moves (a replay).
template <int ROWS, int STEPS, bool OUT>
__global__ void __launch_bounds__(2 * ROWS) wide_chain_kernel(
        const float* __restrict__ q, const float* __restrict__ k, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta, const float* __restrict__ state,
        float* __restrict__ last, __nv_bfloat16* __restrict__ y, int W, int hk, int hv) {
    __shared__ float4 ks[STEPS][DK / 4], qs[OUT ? STEPS : 1][DK / 4];
    __shared__ float gs[STEPS], bs[STEPS];
    __shared__ __align__(16) __nv_bfloat16 vs[STEPS][ROWS];
    const int head = blockIdx.x, local = threadIdx.x >> 1, row = blockIdx.y * ROWS + local, half = threadIdx.x & 1;
    const int key_head = head / (hv / hk);
    float s[64];
    const float* s0 = state + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float4 t = *reinterpret_cast<const float4*>(s0 + 8 * j + 4 * half);
        s[4 * j] = t.x; s[4 * j + 1] = t.y; s[4 * j + 2] = t.z; s[4 * j + 3] = t.w;
    }
    Prefetch<2 * ROWS, ROWS, STEPS, OUT> pf;
    pf.fetch(q, k, v, g, beta, 0, W, hk, hv, head, key_head, blockIdx.y * ROWS);
    for (int t0 = 0; t0 < W; t0 += STEPS) {
        const int n = min(STEPS, W - t0);
        __syncthreads();
        pf.store(ks, qs, gs, bs, vs);
        __syncthreads();
        if (t0 + STEPS < W) pf.fetch(q, k, v, g, beta, t0 + STEPS, W, hk, hv, head, key_head, blockIdx.y * ROWS);
        for (int tt = 0; tt < n; ++tt) {
            const float gt = gs[tt], bt = bs[tt];
            const float vt = __bfloat162float(vs[tt][local]);
            // decay, read (k), correct toward v: gdn.cu's ``update``, a lane's four products summed from zero
            float p[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = ks[tt][2 * j + half];
                float kv = 0.0f;
                s[4 * j] = s[4 * j] * gt;
                kv = kv + s[4 * j] * kk.x;
                s[4 * j + 1] = s[4 * j + 1] * gt;
                kv = kv + s[4 * j + 1] * kk.y;
                s[4 * j + 2] = s[4 * j + 2] * gt;
                kv = kv + s[4 * j + 2] * kk.z;
                s[4 * j + 3] = s[4 * j + 3] * gt;
                kv = kv + s[4 * j + 3] * kk.w;
                p[j] = kv;
            }
            float mem = parity_sum(p);
            mem = mem + __shfl_xor_sync(0xffffffffu, mem, 1);
            const float delta = (vt - mem) * bt;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float4 kk = ks[tt][2 * j + half];
                s[4 * j] = s[4 * j] + kk.x * delta;
                s[4 * j + 1] = s[4 * j + 1] + kk.y * delta;
                s[4 * j + 2] = s[4 * j + 2] + kk.z * delta;
                s[4 * j + 3] = s[4 * j + 3] + kk.w * delta;
                if constexpr (OUT) {
                    const float4 qq = qs[tt][2 * j + half];
                    float o = 0.0f;
                    o = o + s[4 * j] * qq.x;
                    o = o + s[4 * j + 1] * qq.y;
                    o = o + s[4 * j + 2] * qq.z;
                    o = o + s[4 * j + 3] * qq.w;
                    p[j] = o;
                }
            }
            if constexpr (OUT) {
                float out = parity_sum(p);
                out = out + __shfl_xor_sync(0xffffffffu, out, 1);
                if (half == 0) y[(static_cast<size_t>(t0 + tt) * hv + head) * DV + row] = __float2bfloat16_rn(out);
            }
        }
    }
    float* s1 = last + (static_cast<size_t>(head) * DV + row) * DK;
#pragma unroll
    for (int j = 0; j < 16; ++j)
        *reinterpret_cast<float4*>(s1 + 8 * j + 4 * half) = make_float4(s[4 * j], s[4 * j + 1], s[4 * j + 2],
                                                                        s[4 * j + 3]);
}

constexpr int STEPS = 32;

template <typename T>
T* ptr(const zs::View& x) { return x.defined() && x.numel() ? x.data_ptr<T>() : nullptr; }

bool holds(const zs::View& x, int dtype) { return x.defined() && x.is_contiguous() && x.dtype() == dtype; }

// 48 value heads fill the device a block each; 24 (one rank of two) take half a head's state rows a block.
template <bool OUT>
void launch_chain(const float* q, const float* k, const __nv_bfloat16* v, const float* g, const float* beta,
                  const float* state, float* last, __nv_bfloat16* y, int rows, int nk, int nv) {
    auto stream = zs::stream();
    if (nv == 48)
        wide_chain_kernel<128, STEPS, OUT><<<dim3(nv, 1), 256, 0, stream>>>(q, k, v, g, beta, state, last, y, rows,
                                                                            nk, nv);
    else
        wide_chain_kernel<64, STEPS, OUT><<<dim3(nv, 2), 128, 0, stream>>>(q, k, v, g, beta, state, last, y, rows,
                                                                           nk, nv);
}

}  // namespace

extern "C" {

// zs_qwen4_gdn_chain for a wide window, bit for bit: p bf16 [rows, PW], conv state bf16 [3, C], conv weight bf16
// [C, 4], state fp32 [nv, 128, 128] -> out bf16 [rows, nv * 128], its 32-group sums xs, the state after the last
// row, and what a replay needs (k, v, g, beta), which the chain reads here; q fp32 [rows, nk, 128] is scratch.
int zs_qwen4_gdn_chain_wide(const zs_view* p_, const zs_view* cs_, const zs_view* cw_, const zs_view* state_in_,
                         const zs_view* a_log_, const zs_view* dt_bias_, const zs_view* norm_w_, double eps,
                         int64_t rows, const zs_view* out_, const zs_view* xs_, const zs_view* state_out_,
                         const zs_view* q_, const zs_view* k_, const zs_view* v_, const zs_view* g_,
                         const zs_view* b_) {
    zs::begin();
    const zs::View P(p_), cs(cs_), cw(cw_), state_in(state_in_), a_log(a_log_), dt_bias(dt_bias_), norm_w(norm_w_);
    const zs::View out(out_), xs(xs_), state_out(state_out_), q(q_), k(k_), v(v_), g(g_), b(b_);
    ZS_REQUIRE(holds(P, ZS_BF16) && holds(cs, ZS_BF16) && holds(cw, ZS_BF16) && holds(norm_w, ZS_BF16) &&
               holds(out, ZS_BF16), "p, conv state, conv weight, norm and out: contiguous bf16");
    ZS_REQUIRE(holds(state_in, ZS_F32) && holds(state_out, ZS_F32) && holds(a_log, ZS_F32) && holds(dt_bias, ZS_F32) &&
               holds(xs, ZS_F32), "states, A_log, dt_bias and group sums: contiguous fp32");
    const int64_t nv = a_log.numel(), nk = nv / 3;
    ZS_REQUIRE(nv == 48 || nv == 24, "48 or 24 value heads");
    const int64_t conv = 2 * nk * 128 + nv * 128, pw = conv + nv * 128 + 2 * nv;
    ZS_REQUIRE(rows >= 1 && P.dim() == 2 && P.size(0) >= rows && P.size(1) == pw, "p: [rows, PW]");
    ZS_REQUIRE(cs.numel() == 3 * conv && cw.numel() == conv * 4, "conv state [3, C] and weight [C, 4]");
    ZS_REQUIRE(dt_bias.numel() == nv && norm_w.numel() == 128, "dt_bias [nv] and norm [128]");
    ZS_REQUIRE(state_in.numel() == nv * 128 * 128 && state_out.numel() == nv * 128 * 128,
               "states must be [nv, 128, 128]");
    ZS_REQUIRE(out.numel() >= rows * nv * 128 && xs.numel() >= rows * nv * 4, "out [rows, nv * 128] and its sums");
    ZS_REQUIRE(holds(q, ZS_F32) && q.numel() >= rows * nk * 128 && holds(k, ZS_F32) && k.numel() >= rows * nk * 128 &&
               holds(v, ZS_BF16) && v.numel() >= rows * nv * 128 && holds(g, ZS_F32) && g.numel() >= rows * nv &&
               holds(b, ZS_F32) && b.numel() >= rows * nv,
               "q and k [rows, nk, 128], v [rows, nv, 128], g and beta [rows, nv]");
    auto stream = zs::stream();
    const dim3 front((unsigned)rows, (unsigned)(nk + nv)), back((unsigned)rows, (unsigned)nv);
    if (nv == 48) {
        front_kernel<16, 48><<<front, 128, 0, stream>>>(
            ptr<__nv_bfloat16>(P), ptr<__nv_bfloat16>(cs), ptr<__nv_bfloat16>(cw), ptr<float>(a_log),
            ptr<float>(dt_bias), ptr<float>(q), ptr<float>(k), ptr<__nv_bfloat16>(v), ptr<float>(g), ptr<float>(b));
    } else {
        front_kernel<8, 24><<<front, 128, 0, stream>>>(
            ptr<__nv_bfloat16>(P), ptr<__nv_bfloat16>(cs), ptr<__nv_bfloat16>(cw), ptr<float>(a_log),
            ptr<float>(dt_bias), ptr<float>(q), ptr<float>(k), ptr<__nv_bfloat16>(v), ptr<float>(g), ptr<float>(b));
    }
    ZS_CUDA_LAUNCH_CHECK();
    launch_chain<true>(ptr<float>(q), ptr<float>(k), ptr<__nv_bfloat16>(v), ptr<float>(g), ptr<float>(b),
                       ptr<float>(state_in), ptr<float>(state_out), ptr<__nv_bfloat16>(out), (int)rows, (int)nk,
                       (int)nv);
    ZS_CUDA_LAUNCH_CHECK();
    if (nv == 48)
        back_kernel<16, 48><<<back, 128, 0, stream>>>(ptr<__nv_bfloat16>(out), ptr<__nv_bfloat16>(P),
                                                      ptr<__nv_bfloat16>(norm_w), (float)eps, ptr<float>(xs));
    else
        back_kernel<8, 24><<<back, 128, 0, stream>>>(ptr<__nv_bfloat16>(out), ptr<__nv_bfloat16>(P),
                                                     ptr<__nv_bfloat16>(norm_w), (float)eps, ptr<float>(xs));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// zs_qwen4_gdn_replay through the wide chain's layout, bit for bit: the state after the first ``rows`` rows of the
// last chain, from the state it started at.
int zs_qwen4_gdn_replay_wide(const zs_view* state_in_, const zs_view* k_save_, const zs_view* v_save_,
                          const zs_view* g_save_, const zs_view* b_save_, int64_t rows, const zs_view* state_out_) {
    zs::begin();
    const zs::View state_in(state_in_), k(k_save_), v(v_save_), g(g_save_), b(b_save_), state_out(state_out_);
    ZS_REQUIRE(holds(state_in, ZS_F32) && holds(state_out, ZS_F32) && state_in.numel() == state_out.numel(),
               "states: contiguous fp32 of one shape");
    ZS_REQUIRE(holds(k, ZS_F32) && holds(v, ZS_BF16) && holds(g, ZS_F32) && holds(b, ZS_F32) && g.dim() == 2 &&
               g.size(0) >= rows && rows >= 1, "replay scratch of at least rows rows");
    const int64_t nv = g.size(1), nk = nv / 3;
    ZS_REQUIRE((nv == 48 || nv == 24) && state_in.numel() == nv * 128 * 128, "48 or 24 value heads of [128, 128]");
    launch_chain<false>(nullptr, ptr<float>(k), ptr<__nv_bfloat16>(v), ptr<float>(g), ptr<float>(b),
                        ptr<float>(state_in), ptr<float>(state_out), nullptr, (int)rows, (int)nk, (int)nv);
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
