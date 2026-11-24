// What gdn.cu, gdn_slots.cu and gdn_wide.cu share (spec 121 WS-04): the head's shape and the delta-rule step,
// `update`, TensorFold 0.6.5's (609ca41, families/qwen4_exp/cuda/gdn.cu, Apache-2.0) as gdn.cu had it, and the
// replay's loop `replay_rows`, gdn.cu's until the pool's chain replayed its deferred rows with it (spec 144 WS-09).
// The state's loads and stores stay in each file: gdn_slots.cu's check the alignment at run time. Every object that
// includes this is compiled with --fmad=false (build.zig), so a replay keeps the serial steps' bits.
#pragma once

#include "../common/math.cuh"

namespace qwen4_gdn {

using namespace common_math;

constexpr int DK = 128, DV = 128, TAPS = 4;

// One delta-rule step on this thread's 4 x 4 block of the state: decay, read (k), correct toward v.
__device__ __forceinline__ void update(float (&s)[4][4], const float (&kk)[4], const float* vrow, int warp,
                                       float g, float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

// N values' warp sums in one butterfly (TensorFold 995f98af, kernels/cuda/qwen35_slots.cuh `spread_sum`, Apache-2.0):
// at each of warp_sum's levels a lane adds its partner's half of the values it keeps, so each sum is made of
// warp_sum's pairs in warp_sum's order and has its bits (an addition's two operands commute exactly); value j ends
// in the lanes where lane >> (5 - log2 N) == j. Four sums take 6 shuffles here, 20 as four warp_sums.
template <int N>
__device__ __forceinline__ float spread_sum(float (&v)[N], int lane) {
#pragma unroll
    for (int level = 0; level < 5; ++level) {
        const int m = 16 >> level, h = N >> (level + 1);
        if (h >= 1) {
            const bool up = lane & m;
#pragma unroll
            for (int i = 0; i < h; ++i) {
                const float give = up ? v[i] : v[h + i], keep = up ? v[h + i] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, give, m);
            }
        } else {
            v[0] += __shfl_xor_sync(0xffffffffu, v[0], m);
        }
    }
    return v[0];
}

// `update` with its four k.state sums in one butterfly (spec 143 WS-05), each sum then read from lane 8j: the same
// operations on each value in the same order, so the same bits; 10 shuffles a row where `update` takes 20.
__device__ __forceinline__ void update_spread(float (&s)[4][4], const float (&kk)[4], const float* vrow, int warp,
                                              int lane, float g, float beta) {
    float kv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        kv[j] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv[j] = kv[j] + s[j][i] * kk[i];
        }
    }
    const float sums = spread_sum<4>(kv, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float delta = (vrow[warp * 4 + j] - __shfl_sync(0xffffffffu, sums, j << 3)) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

// The delta rule alone over rows `0 .. rows - 1` of what a chain saved, on the state in registers: a replay's
// loop, and the start of a chain that goes on from a window kept in part (`behind`).
//
// The saves are staged RR rows at a time (spec 096 WS-02): the block's THREADS threads load a row's k and v each
// (RR = THREADS / DK), one barrier, then the rows' updates read shared memory alone, so a row waits on no load and
// no barrier. The same values reach the same `update` in the same order: the same bits. Shared memory: k and v of
// RR rows as fp32 (4,096 B each at 1,024 threads), g and beta (64 B), 8,256 B beside the chain's 2,060; one block
// an SM either way. A block of fewer threads holds part of a head's state rows (gdn.cu's `land_kernel`, spec 146
// WS-07): `t` is its own thread, `warp` the warp's place in the head (its rows 4 warp .. + 3), which `update` reads.
template <int NK, int NV, bool SPREAD, int THREADS = 1024>
__device__ __forceinline__ void replay_rows(float (&s)[4][4], const float* k_save, const __nv_bfloat16* v_save,
                                            const float* g_save, const float* b_save, int rows, int hv, int hk,
                                            int t, int warp, int lane) {
    constexpr int RR = THREADS / DK;
    static_assert(RR >= 1 && RR * DK == THREADS, "a block stages whole rows");
    __shared__ __align__(16) float rk[RR][DK], rv[RR][DV];
    __shared__ float rg[RR], rb[RR];
    const int j = t / DK, col = t % DK;
    for (int r0 = 0; r0 < rows; r0 += RR) {
        const int n = rows - r0 < RR ? rows - r0 : RR;
        if (j < n) {
            rk[j][col] = k_save[((size_t)(r0 + j) * NK + hk) * DK + col];
            rv[j][col] = __bfloat162float(v_save[((size_t)(r0 + j) * NV + hv) * DV + col]);
        }
        if (t < n) { rg[t] = g_save[(r0 + t) * NV + hv]; rb[t] = b_save[(r0 + t) * NV + hv]; }
        __syncthreads();
        for (int i = 0; i < n; ++i) {
            const float4 k4 = *reinterpret_cast<const float4*>(&rk[i][lane * 4]);
            const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
            if constexpr (SPREAD) update_spread(s, kk, rv[i], warp, lane, rg[i], rb[i]);
            else update(s, kk, rv[i], warp, rg[i], rb[i]);
        }
        __syncthreads();
    }
}

}  // namespace qwen4_gdn
