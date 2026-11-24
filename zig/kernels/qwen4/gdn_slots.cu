// Spec 018 WS-B3, the pool's (`--parallel` N): gdn.cu's chain and replay for several streams in one round.
// The arithmetic is gdn.cu's, a slot at a time, so a stream's rows are the bits of that stream alone; a window
// of drafts defers its state as a session's does (spec 144 WS-09, `chain_slots_kernel`'s HELD).
// Grid (slots, value heads). A slot with no rows returns at once, and so does `skip`, the one segment the
// wide kernels own. seg int32 [slots, 4] is first row, rows, kept, first position. ptrs int64 [slots, stride]
// holds that layer's state_in, state_out and conv at columns col, col + 1 and col + 2. The projection, the
// output and the replay scratch are the round's, indexed by row. Compile with --fmad=false, as gdn.cu does.
// SPREAD (spec 143 WS-10, gdn.cu's from WS-05; on with `zs_qwen4_gdn_quads`, at 48 heads and a rank's 24): a row's
// four k.state and four q.state products are summed in one butterfly each (gdn.cuh's `spread_sum`,
// `update_spread`), in the chain, its replay of rows behind and the replay kernel: warp_sum's pairs in its order,
// so the base's bits (`rounds_behind_test`'s four-float test holds them).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../zs/host.h"
#include "gdn.cuh"

namespace {

using namespace qwen4_gdn;

// gdn.cu's load_state and store_state (spec 073 WS-03): a row of a thread's 4 x 4 block as one 16-byte access
// under QUADS, the same values so the same bits. A slot's state comes from the pointer table and not a view, so
// the 16-byte boundary is checked here, a block at a time (every state is a buffer of its own, so it holds).
template <bool QUADS>
__device__ __forceinline__ void load_state(float (&s)[4][4], const float* head, int warp, int lane) {
    const bool quad = QUADS && (reinterpret_cast<uintptr_t>(head) & 15) == 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float* at = head + (size_t)(warp * 4 + j) * DK + lane * 4;
        if (quad) {
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
    const bool quad = QUADS && (reinterpret_cast<uintptr_t>(head) & 15) == 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float* at = head + (size_t)(warp * 4 + j) * DK + lane * 4;
        if (quad) {
            *reinterpret_cast<float4*>(at) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) at[i] = s[j][i];
        }
    }
}

// gdn.cu's chain_kernel's rows for one slot, on the state in registers: the conv, the norms, the gates, the update
// and the read-out of each row, and what a replay of the row needs where `k_save` is given. AHEAD is gdn.cu's split:
// windows of 2 or more rows prefetch, a single row loads each value where it is used.
template <int NK, int NV, bool AHEAD, bool SPREAD>
__device__ __forceinline__ void chain_rows(
        float (&s)[4][4], int rows, const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out, float* __restrict__ xs,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save, int hv, int hk, int t, int warp, int lane) {
    constexpr int C = 2 * NK * DK + NV * DV;
    constexpr int PW = C + NV * DV + 2 * NV;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV];
    __shared__ float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
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
}

// One slot's chain, both of gdn.cu's forms in one launch (spec 144 WS-09: the AHEAD launch and the single row's
// were two, each returning at once for the other's slots). QUADS: the state's rows of four floats (spec 096 WS-01).
//
// HELD (spec 144 WS-09), int64 [slots, hstride], or null: a slot's deferred rows, as a session's `behind` (specs
// 089 WS-03, 096 WS-02). Column 0 is how many rows at the front of the slot's own replay scratch the state does
// not hold yet; columns hcol .. hcol + 3 are this layer's k, v, g and beta of that scratch. Those rows are
// replayed on the state in registers first (gdn.cuh's `replay_rows`, the replay kernel's loop: the bits of a
// replay stored and loaded again). A window of 2 to `wide` - 1 rows (`forward.stores` false) then saves its rows
// after them and writes no state: they wait there for the next chain, whatever the round keeps of them (the host
// lands a slot whose rows would pass its scratch first, `rounds.stage`: a block's saves of k would race the replay
// of another value head of its key head). Any other window writes the state after its rows to state_out and saves
// to the round's scratch, as before.
template <int NK, int NV, bool QUADS, bool SPREAD>
__global__ void __launch_bounds__(1024) chain_slots_kernel(
        const int* __restrict__ SEG, const int64_t* __restrict__ PTRS, int stride, int col, int skip,
        const int64_t* __restrict__ HELD, int hstride, int hcol, int wide,
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    constexpr int PW = 2 * NK * DK + 2 * NV * DV + 2 * NV;
    const int slot = static_cast<int>(blockIdx.x);
    const int rows = SEG[slot * 4 + 1];
    if (rows <= 0 || slot == skip) return;
    const int first = SEG[slot * 4];
    const int64_t* cols = PTRS + (static_cast<size_t>(slot) * stride + col);
    const __nv_bfloat16* cs = reinterpret_cast<const __nv_bfloat16*>(cols[2]);
    const float* state_in = reinterpret_cast<const float*>(cols[0]);
    float* state_out = reinterpret_cast<float*>(cols[1]);
    P += static_cast<size_t>(first) * PW;
    out += static_cast<size_t>(first) * NV * DV;
    xs += static_cast<size_t>(first) * (NV * DV / 32);
    const int hv = static_cast<int>(blockIdx.y), hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
    load_state<QUADS>(s, state_in + sbase, warp, lane);
    const int64_t* held = HELD != nullptr ? HELD + static_cast<size_t>(slot) * hstride : nullptr;
    const int behind = held != nullptr ? static_cast<int>(held[0]) : 0;
    const bool deferred = held != nullptr && rows >= 2 && rows < wide;
    if (behind > 0 || deferred) {
        float* hk_save = reinterpret_cast<float*>(held[hcol]);
        __nv_bfloat16* hv_save = reinterpret_cast<__nv_bfloat16*>(held[hcol + 1]);
        float* hg_save = reinterpret_cast<float*>(held[hcol + 2]);
        float* hb_save = reinterpret_cast<float*>(held[hcol + 3]);
        if (behind > 0) replay_rows<NK, NV, SPREAD>(s, hk_save, hv_save, hg_save, hb_save, behind, hv, hk, t, warp, lane);
        if (deferred) {
            k_save = hk_save + static_cast<size_t>(behind) * NK * DK;
            v_save = hv_save + static_cast<size_t>(behind) * NV * DV;
            g_save = hg_save + static_cast<size_t>(behind) * NV;
            b_save = hb_save + static_cast<size_t>(behind) * NV;
        }
    }
    if (!deferred && k_save != nullptr) {
        k_save += static_cast<size_t>(first) * NK * DK;
        v_save += static_cast<size_t>(first) * NV * DV;
        g_save += static_cast<size_t>(first) * NV;
        b_save += static_cast<size_t>(first) * NV;
    }
    if (rows == 1) {
        chain_rows<NK, NV, false, SPREAD>(s, rows, P, cs, cw, a_log, dt_bias, norm_w, eps, out, xs, k_save, v_save, g_save,
                                  b_save, hv, hk, t, warp, lane);
    } else {
        chain_rows<NK, NV, true, SPREAD>(s, rows, P, cs, cw, a_log, dt_bias, norm_w, eps, out, xs, k_save, v_save, g_save,
                                 b_save, hv, hk, t, warp, lane);
    }
    if (!deferred && state_out != nullptr) store_state<QUADS>(s, state_out + sbase, warp, lane);
}

// gdn.cu's replay_kernel for one slot: the first `kept` rows of the slot, from the state the chain started at,
// written over the state the chain left. A slot that kept every row, or none, returns at once, and so does a
// window of 2 to `wide` - 1 rows, which the chain deferred (`chain_slots_kernel`'s HELD); `wide` 0 defers none.
template <int NK, int NV, bool QUADS, bool SPREAD>
__global__ void __launch_bounds__(1024) replay_slots_kernel(
        const int* __restrict__ SEG, const int64_t* __restrict__ PTRS, int stride, int col, int skip, int wide,
        const float* __restrict__ k_save, const __nv_bfloat16* __restrict__ v_save,
        const float* __restrict__ g_save, const float* __restrict__ b_save) {
    const int slot = static_cast<int>(blockIdx.x);
    const int rows = SEG[slot * 4 + 1];
    const int kept = SEG[slot * 4 + 2];
    if (rows <= 0 || kept >= rows || kept <= 0 || slot == skip || (rows >= 2 && rows < wide)) return;
    const int first = SEG[slot * 4];
    const int64_t* cols = PTRS + (static_cast<size_t>(slot) * stride + col);
    const float* state_in = reinterpret_cast<const float*>(cols[0]);
    float* state_out = reinterpret_cast<float*>(cols[1]);
    k_save += static_cast<size_t>(first) * NK * DK;
    v_save += static_cast<size_t>(first) * NV * DV;
    g_save += static_cast<size_t>(first) * NV;
    b_save += static_cast<size_t>(first) * NV;
    const int hv = static_cast<int>(blockIdx.y), hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
    load_state<QUADS>(s, state_in + sbase, warp, lane);
    for (int r = 0; r < kept; ++r) {
        if (t < DV) vs[t] = __bfloat162float(v_save[((size_t)r * NV + hv) * DV + t]);
        __syncthreads();
        float kk[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) kk[i] = k_save[((size_t)r * NK + hk) * DK + lane * 4 + i];
        if constexpr (SPREAD) update_spread(s, kk, vs, warp, lane, g_save[r * NV + hv], b_save[r * NV + hv]);
        else update(s, kk, vs, warp, g_save[r * NV + hv], b_save[r * NV + hv]);
        __syncthreads();
    }
    store_state<QUADS>(s, state_out + sbase, warp, lane);
}

}  // namespace

namespace {

template <typename T>
T* ptr(const zs::View& x) { return x.defined() && x.numel() ? x.data_ptr<T>() : nullptr; }

bool holds(const zs::View& x, int dtype) { return x.defined() && x.is_contiguous() && x.dtype() == dtype; }

}  // namespace

// gdn.cu's `zs_qwen4_gdn_quads` switch (`ZSPARK_GDN_QUADS`), which the slots' kernels follow too: QUADS and SPREAD.
namespace zs_qwen4_gdn { bool quads_on(); }

extern "C" {

// One chain a slot of the round, one launch. `skip` is the slot the wide kernels run, or -1. Saves what a replay
// reads when the scratch views are present. `held` (int64 [slots, hstride], or null) is each slot's deferred rows
// (`chain_slots_kernel`): with it a window of 2 to `wide` - 1 rows saves into the slot's scratch after the rows it
// holds and writes no state.
int zs_qwen4_gdn_chain_slots(const zs_view* seg_, const zs_view* ptrs_, int64_t stride, int64_t col, int64_t skip,
                             const zs_view* held_, int64_t hcol, int64_t wide,
                             const zs_view* p_, const zs_view* cw_, const zs_view* a_log_, const zs_view* dt_bias_,
                             const zs_view* norm_w_, double eps, const zs_view* out_, const zs_view* xs_,
                             const zs_view* k_save_, const zs_view* v_save_, const zs_view* g_save_,
                             const zs_view* b_save_) {
    zs::begin();
    const zs::View seg(seg_), ptrs(ptrs_), held(held_), P(p_), cw(cw_), a_log(a_log_), dt_bias(dt_bias_), norm_w(norm_w_);
    const zs::View out(out_), xs(xs_), k_save(k_save_), v_save(v_save_), g_save(g_save_), b_save(b_save_);
    ZS_REQUIRE(holds(seg, ZS_I32) && seg.dim() == 2 && seg.size(1) == 4, "seg: int32 [slots, 4]");
    const int64_t slots = seg.size(0);
    ZS_REQUIRE(holds(ptrs, ZS_I64) && ptrs.dim() == 2 && ptrs.size(0) == slots && ptrs.size(1) == stride &&
               col >= 0 && col + 2 < stride, "ptrs: int64 [slots, stride], three columns from col");
    ZS_REQUIRE(!held.defined() || (holds(held, ZS_I64) && held.dim() == 2 && held.size(0) == slots && hcol >= 1 &&
                                   hcol + 3 < held.size(1) && wide >= 2),
               "held: int64 [slots, columns], the count and four columns from hcol");
    ZS_REQUIRE(holds(P, ZS_BF16) && holds(cw, ZS_BF16) && holds(norm_w, ZS_BF16) && holds(out, ZS_BF16),
               "p, conv weight, norm and out: contiguous bf16");
    ZS_REQUIRE(holds(a_log, ZS_F32) && holds(dt_bias, ZS_F32) && holds(xs, ZS_F32),
               "A_log, dt_bias and group sums: contiguous fp32");
    const int64_t nv = a_log.numel();
    ZS_REQUIRE(nv == 48 || nv == 24, "48 or 24 value heads");
    ZS_REQUIRE(slots >= 1 && slots <= 64 && skip >= -1 && skip < slots, "slots, and skip one of them or -1");
    const int64_t* held_ptr = held.defined() ? held.data_ptr<int64_t>() : nullptr;
    const int hstride = held.defined() ? static_cast<int>(held.size(1)) : 0;
    auto launch = [&](auto kernel) {
        const dim3 grid(static_cast<unsigned>(slots), static_cast<unsigned>(nv));
        kernel<<<grid, 1024, 0, zs::stream()>>>(
            seg.data_ptr<int>(), ptrs.data_ptr<int64_t>(), static_cast<int>(stride), static_cast<int>(col),
            static_cast<int>(skip), held_ptr, hstride, static_cast<int>(hcol), static_cast<int>(wide),
            ptr<__nv_bfloat16>(P), ptr<__nv_bfloat16>(cw), ptr<float>(a_log), ptr<float>(dt_bias),
            ptr<__nv_bfloat16>(norm_w), static_cast<float>(eps), ptr<__nv_bfloat16>(out), ptr<float>(xs),
            ptr<float>(k_save), ptr<__nv_bfloat16>(v_save), ptr<float>(g_save), ptr<float>(b_save));
    };
    const bool quads = zs_qwen4_gdn::quads_on();
    if (nv == 48) launch(quads ? chain_slots_kernel<16, 48, true, true> : chain_slots_kernel<16, 48, false, false>);
    else launch(quads ? chain_slots_kernel<8, 24, true, true> : chain_slots_kernel<8, 24, false, false>);
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// The state after each slot's kept rows, from the state its chain started at. A slot that kept every row is
// left as the chain wrote it. `skip` is the wide segment, replayed by the wide kernel; a window of 2 to `wide` - 1
// rows was deferred by its chain and is left too (`wide` 0: none was).
int zs_qwen4_gdn_replay_slots(const zs_view* seg_, const zs_view* ptrs_, int64_t stride, int64_t col, int64_t skip,
                              int64_t wide, const zs_view* k_save_, const zs_view* v_save_, const zs_view* g_save_,
                              const zs_view* b_save_) {
    zs::begin();
    const zs::View seg(seg_), ptrs(ptrs_), k_save(k_save_), v_save(v_save_), g_save(g_save_), b_save(b_save_);
    ZS_REQUIRE(holds(seg, ZS_I32) && seg.dim() == 2 && seg.size(1) == 4, "seg: int32 [slots, 4]");
    const int64_t slots = seg.size(0);
    ZS_REQUIRE(holds(ptrs, ZS_I64) && ptrs.dim() == 2 && ptrs.size(0) == slots && ptrs.size(1) == stride &&
               col >= 0 && col + 2 < stride, "ptrs: int64 [slots, stride]");
    ZS_REQUIRE(holds(k_save, ZS_F32) && holds(v_save, ZS_BF16) && holds(g_save, ZS_F32) && holds(b_save, ZS_F32),
               "replay scratch: k and g and beta fp32, v bf16");
    const int64_t nv = g_save.size(1);
    ZS_REQUIRE(nv == 48 || nv == 24, "48 or 24 value heads");
    ZS_REQUIRE(slots >= 1 && skip >= -1 && skip < slots && wide >= 0, "skip is one slot or -1");
    auto stream = zs::stream();
    const dim3 grid(static_cast<unsigned>(slots), static_cast<unsigned>(nv));
    auto launch = [&](auto kernel) {
        kernel<<<grid, 1024, 0, stream>>>(
            seg.data_ptr<int>(), ptrs.data_ptr<int64_t>(), static_cast<int>(stride), static_cast<int>(col),
            static_cast<int>(skip), static_cast<int>(wide), k_save.data_ptr<float>(),
            v_save.data_ptr<__nv_bfloat16>(), g_save.data_ptr<float>(), b_save.data_ptr<float>());
    };
    const bool quads = zs_qwen4_gdn::quads_on();
    if (nv == 48) launch(quads ? replay_slots_kernel<16, 48, true, true> : replay_slots_kernel<16, 48, false, false>);
    else launch(quads ? replay_slots_kernel<8, 24, true, true> : replay_slots_kernel<8, 24, false, false>);
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
