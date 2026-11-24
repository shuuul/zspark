// KDA's chain and replay kernels, TensorFold 0.6.6's (cb2ebf0, families/glm5_next/cuda/kda.cu, Apache-2.0) device
// code but its wide chain, moved here from kda.cu (spec 121 WS-04) so that kda.cu and kda_quads.cu compile the one
// text. The only change: the four loops over a thread's 4 x 4 block of the state read and write a row through
// `load4` and `store4`, which the including file defines before it includes this, in the namespace it names by
// KDA_NAMESPACE (empty: the anonymous one): kda.cu as upstream's four scalar loads and stores, kda_quads.cu as one
// 16-byte access. Each object compiles to the instructions it had (cuobjdump -sass, spec 121). Compiled with
// --fmad=false by both, as upstream does.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../common/math.cuh"

namespace KDA_NAMESPACE {

using namespace common_math;

constexpr int DK = 128, DV = 128, TAPS = 4;

// zspark's own (spec 107 WS-03): warp_sum of four values at once, its tree (xor 16, 8, 4, 2, 1, each lane its own
// value plus its partner's) and so its bits, in 6 shuffles for 20. After the first two levels a lane keeps one of
// the four, j = 2 * bit 4 + bit 3 of the lane: the sum of x[j] ends on the lanes of that j.
__device__ __forceinline__ float warp_sum4(const float (&x)[4], int lane) {
    const bool hi = lane & 16, b = lane & 8;
    const float r0 = __shfl_xor_sync(0xffffffffu, hi ? x[0] : x[2], 16);
    const float r1 = __shfl_xor_sync(0xffffffffu, hi ? x[1] : x[3], 16);
    const float a0 = (hi ? x[2] : x[0]) + r0, a1 = (hi ? x[3] : x[1]) + r1;
    const float r = __shfl_xor_sync(0xffffffffu, b ? a0 : a1, 8);
    float c = (b ? a1 : a0) + r;
    for (int o = 4; o; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    return c;
}

// One delta-rule step on this thread's 4 x 4 block of the state: per-channel decay, read (k), correct toward v.
// The four reads are summed by warp_sum4 (warp_sum's bits) and every lane gets the four back in 3 shuffles.
__device__ __forceinline__ void update4(float (&s)[4][4], const float (&kk)[4], const float (&gg)[4],
                                        const float (&vv)[4], int lane, float beta) {
    float kv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        kv[j] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * gg[i];
            kv[j] = kv[j] + s[j][i] * kk[i];
        }
    }
    const float c = warp_sum4(kv, lane);
    const float c8 = __shfl_xor_sync(0xffffffffu, c, 8);
    // this lane's pair (j with bit 4 of the lane), then the other pair from lane ^ 16
    const bool hi = lane & 16, b = lane & 8;
    const float m0 = b ? c8 : c, m1 = b ? c : c8;
    const float o0 = __shfl_xor_sync(0xffffffffu, m0, 16), o1 = __shfl_xor_sync(0xffffffffu, m1, 16);
    const float sum[4] = {hi ? o0 : m0, hi ? o1 : m1, hi ? m0 : o0, hi ? m1 : o1};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float delta = (vv[j] - sum[j]) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

__global__ void __launch_bounds__(1024) chain_kernel(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ state_in, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, float lower, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ state_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    const int C = 3 * H * DK;                         // conv channels: q | k | v
    const int h = blockIdx.x;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV], gs[DK];
    __shared__ float beta_s, rinv;
    int c = -1;
    if (t < 3 * DK) c = (t / DK) * H * DK + h * DK + (t % DK);
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j) load4(s[j], state_in + sbase + (size_t)(warp * 4 + j) * DK + lane * 4);
    const float decay_rate = expf(a_log[h]);
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[(size_t)at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * p_stride + c]);
                acc = acc + __bfloat162float(cw[(size_t)c * TAPS + tap]) * x;
            }
            const float act = bf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        } else if (t >= 512 && t < 512 + DK) {
            const int i = t - 512;
            const float a = __bfloat162float(A[(size_t)r * a_stride + h * DK + i]) + dt_bias[h * DK + i];
            gs[i] = expf(lower * sigmoidf_(decay_rate * a));
        } else if (t == 1023) {
            beta_s = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
        }
        __syncthreads();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = v4[i] * inv;
                if (warp == 0) y = y * (1.0f / sqrtf((float)DK));
                x[lane * 4 + i] = y;
            }
        }
        __syncthreads();
        const float beta = beta_s;
        float kk[4], qq[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; gg[i] = gs[lane * 4 + i]; }
        const float vv[4] = {vs[warp * 4], vs[warp * 4 + 1], vs[warp * 4 + 2], vs[warp * 4 + 3]};
        update4(s, kk, gg, vv, lane, beta);
        float o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o[j] = o[j] + s[j][i] * qq[i];
        }
        const float y = warp_sum4(o, lane);
        if ((lane & 7) == 0) ys[warp * 4 + (lane >> 4) * 2 + ((lane >> 3) & 1)] = bf(y);
        if (k_save != nullptr) {
            const size_t base = ((size_t)r * H + h) * DK;
            if (t < DK) { k_save[base + t] = ks[t]; g_save[base + t] = gs[t]; }
            if (t >= DK && t < DK + DV) v_save[base + t - DK] = __float2bfloat16_rn(vs[t - DK]);
            if (t == 0) b_save[r * H + h] = beta;
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
            const float yn = ys[t] * rinv;
            const float yw = __bfloat162float(norm_w[t]) * yn;
            const float gate = __bfloat162float(G[(size_t)r * g_stride + h * DV + t]);
            out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
        }
        __syncthreads();
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) store4(state_out + sbase + (size_t)(warp * 4 + j) * DK + lane * 4, s[j]);
    }
}

// A replay of the saved rows from the states a chain started at, all layers at once: block (layer, head);
// per-layer strides of the state buffers and saved rows.
__global__ void __launch_bounds__(1024) replay_layers_kernel(
        int H, const float* __restrict__ state_in, size_t state_stride, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        size_t kv_stride, size_t b_stride, int rows, float* __restrict__ state_out) {
    const int layer = blockIdx.x / H, h = blockIdx.x % H;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const float* sin = state_in + layer * state_stride;
    float* sout = state_out + layer * state_stride;
    const float* ks = k_save + layer * kv_stride;
    const __nv_bfloat16* vsv = v_save + layer * kv_stride;
    const float* gsv = g_save + layer * kv_stride;
    const float* bsv = b_save + layer * b_stride;
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j) load4(s[j], sin + sbase + (size_t)(warp * 4 + j) * DK + lane * 4);
    for (int r = 0; r < rows; ++r) {
        const size_t base = ((size_t)r * H + h) * DK;
        if (t < DV) vs[t] = __bfloat162float(vsv[base + t]);
        __syncthreads();
        float kk[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[base + lane * 4 + i]; gg[i] = gsv[base + lane * 4 + i]; }
        const float vv[4] = {vs[warp * 4], vs[warp * 4 + 1], vs[warp * 4 + 2], vs[warp * 4 + 3]};
        update4(s, kk, gg, vv, lane, bsv[r * H + h]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) store4(sout + sbase + (size_t)(warp * 4 + j) * DK + lane * 4, s[j]);
}

}  // namespace KDA_NAMESPACE
