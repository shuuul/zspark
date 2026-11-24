// From TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/gdn.cu and gdn.cpp, Apache-2.0; changed for zspark: ATen and
// c10 replaced by zs/host.h, a C ABI over zs_view; `bf`, `warp_sum`, `sigmoidf_` and `softplusf_` are
// common/math.cuh's, the shape and `update` gdn.cuh's, the same text (spec 121). Compile with --fmad=false, as
// upstream does.
// Flash Next's Gated DeltaNet on CUDA: one block of 1024 threads per value head, a chain of R rows.
//
// Per row: the depthwise conv over [conv state; projection rows] (4 taps, fp32) with SiLU (one bf16
// rounding), fp32 L2 norms of q and k (eps inside the sum, q times DK^-0.5), g = exp(-exp(A_log) *
// softplus(a + dt_bias)) in fp32, beta = bf16(sigmoid(b)), the delta-rule update and read-out in fp32,
// and the sigmoid-gated RMSNorm of the read-out. Warp w owns state rows 4w .. 4w + 3, lane l the columns
// 4l .. 4l + 3. The state update is one routine (``update``) shared with ``replay``, compiled without FMA
// contraction, so a replayed prefix of a window gives the bits of the serial steps.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../zs/host.h"
#include "gdn.cuh"

namespace {

using namespace qwen4_gdn;

// NK key heads and NV value heads: the whole layer (16, 48) or one tensor-parallel rank's share (8, 24)

// A thread's 4 x 4 block of a head's state, to and from registers. QUADS (spec 073 WS-03, `ZSPARK_GDN_QUADS`): a
// row of the block as one 16-byte access and not four of 4 bytes; the same values, so the same bits. Nsight
// Compute counted 381,000 sectors written for the chain's 3.07 MB of state (96,000 sectors): each scalar store
// sends its sector again. Shared memory is as the base's (q, k, v, y rows of 128 floats, 2,060 B a block).
// SPREAD (spec 143 WS-05, upstream 995f98af; on with the switch: a layer's 48 heads with QUADS, a rank's 24): a
// row's four k.state and four q.state products are summed in one butterfly each (`spread_sum`, `update_spread`), 16
// shuffles a warp a row where eight warp_sums take 40; each sum is warp_sum's pairs in its order, so the base's bits
// (`gdn_test`'s four-float test holds them). The chain 101.8 to 89.5 ms over a traced decode's 2,664 launches, the
// replay -3%.
template <bool QUADS>
__device__ __forceinline__ void load_state(float (&s)[4][4], const float* head, int warp, int lane) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float* at = head + (size_t)(warp * 4 + j) * DK + lane * 4;
        if constexpr (QUADS) {
            const float4 q = *reinterpret_cast<const float4*>(at);
            s[j][0] = q.x; s[j][1] = q.y; s[j][2] = q.z; s[j][3] = q.w;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] = at[i];
        }
    }
}

template <bool QUADS>
__device__ __forceinline__ void store_state(const float (&s)[4][4], float* head, int warp, int lane) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float* at = head + (size_t)(warp * 4 + j) * DK + lane * 4;
        if constexpr (QUADS) {
            *reinterpret_cast<float4*>(at) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) at[i] = s[j][i];
        }
    }
}

// `replay_rows`, the delta rule alone over saved rows, is gdn.cuh's: gdn_slots.cu replays a slot's deferred rows
// with it too (spec 144 WS-09).

// AHEAD (windows of 2+ rows): the conv's 4 weights and last 3 inputs stay in registers and each row's projection
// values (conv input, z, b, a) load a row ahead; one row loads each where it is used. Same ops in the same order.
//
// `behind` (spec 089 WS-03): a device int, the rows at the front of the saves that the state does not hold yet,
// since the window before was kept in part and its replay left to this launch. They are replayed on the state
// in registers first (`replay_rows`, the replay kernel's loop, so the bits of a replay stored and loaded
// again), and this chain's rows are saved after them: a later replay runs over both. The state is then read
// once a round and not twice. The replay's staging is `replay_rows`'s.
template <int NK, int NV, bool AHEAD, bool QUADS, bool SPREAD>
__global__ void __launch_bounds__(1024) chain_kernel(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ state_in,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs, float* __restrict__ state_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save, const int* __restrict__ behind) {
    constexpr int C = 2 * NK * DK + NV * DV;          // conv channels: q | k | v
    constexpr int PW = C + NV * DV + 2 * NV;          // projection row: qkv | z | b | a
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV];
    __shared__ float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
    load_state<QUADS>(s, state_in + sbase, warp, lane);
    const int kept = behind != nullptr && k_save != nullptr ? *behind : 0;
    if (kept > 0) {
        replay_rows<NK, NV, SPREAD>(s, k_save, v_save, g_save, b_save, kept, hv, hk, t, warp, lane);
        k_save += (size_t)kept * NK * DK; v_save += (size_t)kept * NV * DV; g_save += kept * NV; b_save += kept * NV;
    }
    float w[TAPS] = {}, win[TAPS - 1] = {};
    __nv_bfloat16 xin = {}, zin = {}, bin = {}, ain = {};
    const __nv_bfloat16 *pz = P + C + hv * DV + (t < DV ? t : 0), *pb = P + C + NV * DV + hv, *pa = pb + NV;
    if (AHEAD && rows > 0) {
        if (c >= 0) {
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) w[tap] = __bfloat162float(cw[c * TAPS + tap]);
#pragma unroll
            for (int tap = 0; tap < TAPS - 1; ++tap) win[tap] = __bfloat162float(cs[tap * C + c]);
            xin = P[c];
        }
        if (t < DV) zin = pz[0];
        if (warp == 2 && lane == 0) { bin = pb[0]; ain = pa[0]; }
    }
    for (int r = 0; r < rows; ++r) {
        const __nv_bfloat16 xr = xin, zr = zin, br = bin, ar = ain;
        if (AHEAD && r + 1 < rows) {
            const size_t next = (size_t)(r + 1) * PW;
            if (c >= 0) xin = P[next + c];
            if (t < DV) zin = pz[next];
            if (warp == 2 && lane == 0) { bin = pb[next]; ain = pa[next]; }
        }
        if (c >= 0) {
            float acc = 0.0f;
            if constexpr (AHEAD) {
                const float xn = __bfloat162float(xr);
#pragma unroll
                for (int tap = 0; tap < TAPS - 1; ++tap) acc = acc + w[tap] * win[tap];
                acc = acc + w[TAPS - 1] * xn;
                win[0] = win[1]; win[1] = win[2]; win[2] = xn;
            } else {
#pragma unroll
                for (int tap = 0; tap < TAPS; ++tap) {
                    const int at = r + tap;
                    const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                                  : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
                    acc = acc + __bfloat162float(cw[c * TAPS + tap]) * x;
                }
            }
            const float act = bf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        }
        __syncthreads();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) x[lane * 4 + i] = v4[i] * inv;
        } else if (warp == 2 && lane == 0) {
            const float b = __bfloat162float(AHEAD ? br : pb[(size_t)r * PW]);
            const float a = __bfloat162float(AHEAD ? ar : pa[(size_t)r * PW]);
            gates[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gates[1] = bf(sigmoidf_(b));
        }
        __syncthreads();
        const float g = gates[0], beta = gates[1];
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
        if constexpr (SPREAD) {
            update_spread(s, kk, vs, warp, lane, g, beta);
            float o[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o[j] = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o[j] = o[j] + s[j][i] * qq[i];
            }
            const float sum = spread_sum<4>(o, lane);
            if ((lane & 7) == 0) ys[warp * 4 + (lane >> 3)] = bf(sum);
        } else {
            update(s, kk, vs, warp, g, beta);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
                o = warp_sum(o);
                if (lane == 0) ys[warp * 4 + j] = bf(o);
            }
        }
        // what a replay of this row needs
        if (k_save != nullptr) {
            if (t < DK && hv % (NV / NK) == 0) k_save[((size_t)r * NK + hk) * DK + t] = ks[t];
            if (t < DV) v_save[((size_t)r * NV + hv) * DV + t] = __float2bfloat16_rn(vs[t]);
            if (t == 0) { g_save[r * NV + hv] = g; b_save[r * NV + hv] = beta; }
        }
        __syncthreads();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { const float y = ys[lane * 4 + i]; ss = ss + y * y; }
            ss = warp_sum(ss);
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        __syncthreads();
        if (t < DV) {
            const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
            const float z = __bfloat162float(AHEAD ? zr : pz[(size_t)r * PW]);
            const float o = bf(yn * sigmoidf_(z));
            out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
            const float gs = warp_sum(o);
            if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
        }
        __syncthreads();
    }
    if (state_out != nullptr) store_state<QUADS>(s, state_out + sbase, warp, lane);
}

template <int NK, int NV, bool QUADS, bool SPREAD>
__global__ void __launch_bounds__(1024) replay_kernel(
        const float* __restrict__ state_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save,
        const float* __restrict__ b_save, int rows, float* __restrict__ state_out) {
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
    load_state<QUADS>(s, state_in + sbase, warp, lane);
    replay_rows<NK, NV, SPREAD>(s, k_save, v_save, g_save, b_save, rows, hv, hk, t, warp, lane);
    store_state<QUADS>(s, state_out + sbase, warp, lane);
}

// A landing (spec 146 WS-07): `replay_kernel` for every state a landing brings to its position, in one launch. An
// entry is one DeltaNet layer of one stream: the state its rows start from, its replay scratch, the rows it is
// behind by, and the other side of its pair, which the replay writes. Entries share no buffer, so their blocks run
// in any order. A state row's update reads only its own dot with k (`update`'s warp sum over one row), so a head's
// 128 rows split over LAND_SPLIT blocks of LAND_THREADS threads: warp w of part p is the head's warp 8 p + w, with
// the lanes, the 4 x 4 block, the staged rows and the sums `replay_kernel` gives that warp: the same bits.
// The landing was 36 launches in series, one a layer, of a block of 1,024 threads a head, alone on its SM (59
// registers): 40 us each at a rank's 24 heads, half the SMs idle, 1.44 ms a landing (spec 146 WS-02). Here a
// grid of (entries, heads x parts) holds four blocks an SM, so one block's state loads and stores overlap another's
// rows, and one launch floor is paid a landing.
// Shared memory a block: k and v of RR = 2 rows as fp32 (1,024 B each), g and beta (16 B): `replay_rows`'s staging,
// sized by the block's own threads; registers, not shared memory, bound the blocks an SM.
constexpr int LAND_THREADS = 256, LAND_SPLIT = 1024 / LAND_THREADS, LAND_MOST = 64;

// A launch's entries, by value (3,328 B, under the classic 4 KiB of parameters); `__grid_constant__` reads them
// from the parameter space where a block indexes them, with no copy to local memory.
struct Lands {
    const float* state_in[LAND_MOST];
    float* state_out[LAND_MOST];
    const float* k[LAND_MOST];
    const __nv_bfloat16* v[LAND_MOST];
    const float* g[LAND_MOST];
    const float* b[LAND_MOST];
    int rows[LAND_MOST];
};

template <int NK, int NV, bool QUADS, bool SPREAD>
__global__ void __launch_bounds__(LAND_THREADS, 4) land_kernel(const __grid_constant__ Lands lands) {
    const int e = blockIdx.x;
    const int hv = blockIdx.y / LAND_SPLIT, part = blockIdx.y % LAND_SPLIT, hk = hv / (NV / NK);
    const int t = threadIdx.x, lane = t & 31, warp = part * (LAND_THREADS / 32) + (t >> 5);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
    load_state<QUADS>(s, lands.state_in[e] + sbase, warp, lane);
    replay_rows<NK, NV, SPREAD, LAND_THREADS>(s, lands.k[e], lands.v[e], lands.g[e], lands.b[e], lands.rows[e], hv,
                                              hk, t, warp, lane);
    store_state<QUADS>(s, lands.state_out[e] + sbase, warp, lane);
}


}  // namespace

namespace {

// `zs_qwen4_gdn_quads`: the chain and the replay through their QUADS and SPREAD forms, unless switched off.
bool quads = true;

// A state a float4 may be read from: absent or on a 16-byte boundary.
bool aligned(const zs::View& x) { return !x.defined() || ((uintptr_t)x.data_ptr<float>() & 15) == 0; }

template <typename T>
T* ptr(const zs::View& x) { return x.defined() && x.numel() ? x.data_ptr<T>() : nullptr; }

bool holds(const zs::View& x, int dtype) { return x.defined() && x.is_contiguous() && x.dtype() == dtype; }

}  // namespace

// The switch as gdn_slots.cu reads it, for the pool's chain and replay (spec 096 WS-01).
namespace zs_qwen4_gdn { bool quads_on() { return quads; } }

void gdn_chain_cuda(const zs::View& P, const zs::View& cs, const zs::View& cw, const zs::View& state_in,
                    const zs::View& a_log, const zs::View& dt_bias, const zs::View& norm_w, double eps,
                    int64_t rows, zs::View& out, zs::View& xs, zs::View& state_out, zs::View& k_save,
                    zs::View& v_save, zs::View& g_save, zs::View& b_save, const zs::View& behind) {
    auto stream = zs::stream();
    const int nv = (int)a_log.numel();
    auto launch = [&](auto kernel, int blocks) {
        kernel<<<blocks, 1024, 0, stream>>>(
        ptr<__nv_bfloat16>(P), ptr<__nv_bfloat16>(cs), ptr<__nv_bfloat16>(cw), ptr<float>(state_in),
        ptr<float>(a_log), ptr<float>(dt_bias), ptr<__nv_bfloat16>(norm_w), (float)eps, (int)rows,
        ptr<__nv_bfloat16>(out), ptr<float>(xs), ptr<float>(state_out), ptr<float>(k_save),
        ptr<__nv_bfloat16>(v_save), ptr<float>(g_save), ptr<float>(b_save), ptr<int>(behind));
    };
    if (nv == 48 && quads && aligned(state_in) && aligned(state_out)) launch(rows > 1 ? chain_kernel<16, 48, true, true, true> : chain_kernel<16, 48, false, true, true>, 48);
    else if (nv == 48) launch(rows > 1 ? chain_kernel<16, 48, true, false, false> : chain_kernel<16, 48, false, false, false>, 48);
    else if (nv == 24 && quads) launch(rows > 1 ? chain_kernel<8, 24, true, false, true> : chain_kernel<8, 24, false, false, true>, 24);
    else if (nv == 24) launch(rows > 1 ? chain_kernel<8, 24, true, false, false> : chain_kernel<8, 24, false, false, false>, 24);
    else zs::fail("gdn chain: 48 or 24 value heads");
    ZS_CUDA_LAUNCH_CHECK();
}

void gdn_replay_cuda(const zs::View& state_in, const zs::View& k_save, const zs::View& v_save,
                     const zs::View& g_save, const zs::View& b_save, int64_t rows, zs::View& state_out) {
    auto stream = zs::stream();
    const int nv = (int)g_save.size(1);
    auto launch = [&](auto kernel, int blocks) {
        kernel<<<blocks, 1024, 0, stream>>>(
            ptr<float>(state_in), ptr<float>(k_save), ptr<__nv_bfloat16>(v_save), ptr<float>(g_save),
            ptr<float>(b_save), (int)rows, ptr<float>(state_out));
    };
    if (nv == 48 && quads && aligned(state_in) && aligned(state_out)) launch(replay_kernel<16, 48, true, true>, 48);
    else if (nv == 48) launch(replay_kernel<16, 48, false, false>, 48);
    else if (nv == 24 && quads) launch(replay_kernel<8, 24, false, true>, 24);
    else if (nv == 24) launch(replay_kernel<8, 24, false, false>, 24);
    else zs::fail("gdn replay: 48 or 24 value heads");
    ZS_CUDA_LAUNCH_CHECK();
}

extern "C" {

// A chain of ``rows`` rows from ``state_in``: p bf16 [rows, PW] (qkv | z | b | a), conv state bf16 [3, C], conv
// weight bf16 [C, 4], state fp32 [nv, 128, 128] -> out bf16 [rows, nv * 128], its 32-group sums xs, the state after
// the last row, and what a replay of each row needs (k, v, g, beta). `behind` (i32 [1], optional): the saved
// rows the launch finds there and replays first; its own rows are saved after them, and the caller keeps room.
int zs_qwen4_gdn_chain(const zs_view* p_, const zs_view* cs_, const zs_view* cw_, const zs_view* state_in_,
                    const zs_view* a_log_, const zs_view* dt_bias_, const zs_view* norm_w_, double eps, int64_t rows,
                    const zs_view* out_, const zs_view* xs_, const zs_view* state_out_, const zs_view* k_save_,
                    const zs_view* v_save_, const zs_view* g_save_, const zs_view* b_save_,
                    const zs_view* behind_) {
    zs::begin();
    const zs::View behind(behind_);
    ZS_REQUIRE(!behind.defined() || (holds(behind, ZS_I32) && behind.numel() == 1), "behind: one i32");
    const zs::View P(p_), cs(cs_), cw(cw_), state_in(state_in_), a_log(a_log_), dt_bias(dt_bias_), norm_w(norm_w_);
    const zs::View out(out_), xs(xs_), state_out(state_out_), k_save(k_save_), v_save(v_save_), g_save(g_save_),
        b_save(b_save_);
    ZS_REQUIRE(holds(P, ZS_BF16) && holds(cs, ZS_BF16) && holds(cw, ZS_BF16) && holds(norm_w, ZS_BF16) &&
               holds(out, ZS_BF16), "p, conv state, conv weight, norm and out: contiguous bf16");
    ZS_REQUIRE(holds(state_in, ZS_F32) && holds(a_log, ZS_F32) && holds(dt_bias, ZS_F32) && holds(xs, ZS_F32),
               "state, A_log, dt_bias and group sums: contiguous fp32");
    const int64_t nv = a_log.numel(), nk = nv / 3;
    ZS_REQUIRE(nv == 48 || nv == 24, "48 or 24 value heads");
    const int64_t conv = 2 * nk * 128 + nv * 128, pw = conv + nv * 128 + 2 * nv;
    ZS_REQUIRE(rows >= 1 && P.dim() == 2 && P.size(0) >= rows && P.size(1) == pw, "p: [rows, PW]");
    ZS_REQUIRE(cs.numel() == 3 * conv && cw.numel() == conv * 4, "conv state [3, C] and weight [C, 4]");
    ZS_REQUIRE(dt_bias.numel() == nv && norm_w.numel() == 128, "dt_bias [nv] and norm [128]");
    ZS_REQUIRE(state_in.numel() == nv * 128 * 128, "state must be [nv, 128, 128]");
    ZS_REQUIRE(!state_out.defined() || (holds(state_out, ZS_F32) && state_out.numel() == nv * 128 * 128),
               "state out must be [nv, 128, 128]");
    ZS_REQUIRE(out.numel() >= rows * nv * 128 && xs.numel() >= rows * nv * 4, "out [rows, nv * 128] and its sums");
    ZS_REQUIRE(!k_save.defined() || (holds(k_save, ZS_F32) && k_save.numel() >= rows * nk * 128 &&
                                     holds(v_save, ZS_BF16) && v_save.numel() >= rows * nv * 128 &&
                                     holds(g_save, ZS_F32) && g_save.numel() >= rows * nv &&
                                     holds(b_save, ZS_F32) && b_save.numel() >= rows * nv),
               "replay scratch: k [rows, nk, 128], v [rows, nv, 128], g and beta [rows, nv]");
    zs::View o = out, x = xs, so = state_out, ks = k_save, vs = v_save, gs = g_save, bs = b_save;
    gdn_chain_cuda(P, cs, cw, state_in, a_log, dt_bias, norm_w, eps, rows, o, x, so, ks, vs, gs, bs, behind);
    return zs::status();
}

// The state after the first ``rows`` rows of the last chain, from the state it started at.
int zs_qwen4_gdn_replay(const zs_view* state_in_, const zs_view* k_save_, const zs_view* v_save_, const zs_view* g_save_,
                     const zs_view* b_save_, int64_t rows, const zs_view* state_out_) {
    zs::begin();
    const zs::View state_in(state_in_), k_save(k_save_), v_save(v_save_), g_save(g_save_), b_save(b_save_);
    zs::View state_out(state_out_);
    ZS_REQUIRE(holds(state_in, ZS_F32) && holds(state_out, ZS_F32) && state_in.numel() == state_out.numel(),
               "states: contiguous fp32 of one shape");
    ZS_REQUIRE(holds(k_save, ZS_F32) && holds(v_save, ZS_BF16) && holds(g_save, ZS_F32) && holds(b_save, ZS_F32) &&
               g_save.dim() == 2 && g_save.size(0) >= rows, "replay scratch of at least rows rows");
    gdn_replay_cuda(state_in, k_save, v_save, g_save, b_save, rows, state_out);
    return zs::status();
}

// One DeltaNet state a landing replays (`zs_qwen4_gdn_land`): state_out = the state after the first ``rows`` rows
// of what ``k``, ``v``, ``g`` and ``b`` saved, from state_in, as `zs_qwen4_gdn_replay` gives it.
typedef struct zs_qwen4_gdn_landing {
    zs_view state_in, k, v, g, b, state_out;
    int64_t rows;
} zs_qwen4_gdn_landing;

// `count` entries' replays (`zs_qwen4_gdn_replay` each), one launch for every LAND_MOST of them; the entries of a
// call have one shape (48 or 24 value heads) and share no state. Bits: `zs_qwen4_gdn_replay`'s (spec 146 WS-07).
int zs_qwen4_gdn_land(const zs_qwen4_gdn_landing* entries, int64_t count) {
    zs::begin();
    ZS_REQUIRE(count >= 0 && (count == 0 || entries != nullptr), "entries: count of them");
    if (count == 0) return zs::status();
    const int64_t nv = zs::View(&entries[0].g).size(1), nk = nv / 3;
    ZS_REQUIRE(nv == 48 || nv == 24, "48 or 24 value heads");
    for (int64_t at = 0; at < count; at += LAND_MOST) {
        const int64_t n = count - at < LAND_MOST ? count - at : LAND_MOST;
        Lands lands = {};
        bool fours = quads;
        for (int64_t i = 0; i < n; ++i) {
            const zs_qwen4_gdn_landing& e = entries[at + i];
            const zs::View state_in(&e.state_in), k(&e.k), v(&e.v), g(&e.g), b(&e.b), state_out(&e.state_out);
            ZS_REQUIRE(holds(state_in, ZS_F32) && holds(state_out, ZS_F32) && state_in.numel() == nv * 128 * 128 &&
                       state_out.numel() == nv * 128 * 128, "states: contiguous fp32 [nv, 128, 128]");
            ZS_REQUIRE(holds(k, ZS_F32) && holds(v, ZS_BF16) && holds(g, ZS_F32) && holds(b, ZS_F32) &&
                       g.dim() == 2 && g.size(1) == nv && e.rows >= 1 && g.size(0) >= e.rows &&
                       b.numel() >= e.rows * nv && k.numel() >= e.rows * nk * 128 && v.numel() >= e.rows * nv * 128,
                       "replay scratch of at least rows rows, one shape for every entry");
            ZS_REQUIRE(e.rows <= INT32_MAX, "rows: an int");
            fours = fours && aligned(state_in) && aligned(state_out);
            lands.state_in[i] = state_in.data_ptr<float>();
            lands.state_out[i] = state_out.data_ptr<float>();
            lands.k[i] = k.data_ptr<float>();
            lands.v[i] = v.data_ptr<__nv_bfloat16>();
            lands.g[i] = g.data_ptr<float>();
            lands.b[i] = b.data_ptr<float>();
            lands.rows[i] = (int)e.rows;
        }
        const dim3 grid((unsigned)n, (unsigned)(nv * LAND_SPLIT));
        auto stream = zs::stream();
        // `replay_kernel`'s forms: rows of four floats and butterfly sums with the switch; the same bits either way
        if (nv == 48 && fours) land_kernel<16, 48, true, true><<<grid, LAND_THREADS, 0, stream>>>(lands);
        else if (nv == 48) land_kernel<16, 48, false, false><<<grid, LAND_THREADS, 0, stream>>>(lands);
        else if (fours) land_kernel<8, 24, true, true><<<grid, LAND_THREADS, 0, stream>>>(lands);
        else if (quads) land_kernel<8, 24, false, true><<<grid, LAND_THREADS, 0, stream>>>(lands);
        else land_kernel<8, 24, false, false><<<grid, LAND_THREADS, 0, stream>>>(lands);
        ZS_CUDA_LAUNCH_CHECK();
    }
    return zs::status();
}

// The chain and the replay of a whole layer (48 value heads) load and store the state in rows of four floats, and
// at 48 or 24 heads sum a row's products in butterflies (1), or a float and a warp_sum at a time (0, the base): the
// same bits. Set before a graph is captured.
void zs_qwen4_gdn_quads(int on) { quads = on != 0; }

}  // extern "C"
