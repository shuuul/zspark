// Nemotron-3.5-Transcribe's perception attention prep (spec 151 WS-03): q and k of a TransformerEncoder layer
// normed per head and rotated, in place on the w_qkv output. NeMo's TransformerEncoder with qk_norm
// (`transformer_encoder.py` `MultiHeadAttention.forward`, commit 774bc30): q = LayerNorm(q) per head of Dh
// (weight and bias bf16, fp32 inside, one rounding), then `RotaryPositionalEncoding` on the first ROT of the
// head's dims, half-split (rotate_half, not interleaved): for pair p of ROT / 2, out_p = x_p cos - x_{p+H} sin,
// out_{p+H} = x_{p+H} cos + x_p sin, H = ROT / 2, the angle the row's frame index inside its chunk times
// inv_freq[p] = rope_base^(-2p/ROT); cos and sin are bf16 and the products round to bf16 as PyTorch's two
// elementwise ops do (`attention_prep.cu`'s order). A chunk is a segment of `offsets`; a row's position is its
// index inside its own, so its bits are its own whatever segments share the call. v passes through.
//
// A block is a row and a warp a head (HEADS warps, 32 lanes over a head's Dh <= 128 values, at most 4 each);
// the normed head lives in the warp's shared row for the pairs' second reads. Shared memory: HEADS * Dh bf16,
// 2,560 B at 10 heads of 128. The position lookup is a binary search of `offsets` [S + 1].
//
// Provenance: zspark's own; the norm's arithmetic is `common/layernorm.cu`'s (Welford's sum is PyTorch's
// nn.LayerNorm's to the bit only in fp32; near), the rotation `common/attention_prep.cu`'s roundings.
//
// Why not CUTLASS: not a GEMM.

#include <stdint.h>

#include <cuda_bf16.h>

#include "../zs/host.h"

namespace {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// qkv bf16 [R, 3 W]: parts q, k, v of W = HEADS * DH a row; offsets int32 [S + 1] the segments' first rows;
// q_norm, k_norm bf16 [DH] weight and bias; inv fp32 [ROT / 2].
__global__ void nemotron_prep_kernel(__nv_bfloat16* __restrict__ qkv, const int* __restrict__ offsets, int segs,
                                     const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ qb,
                                     const __nv_bfloat16* __restrict__ kw, const __nv_bfloat16* __restrict__ kb,
                                     const float* __restrict__ inv, int heads, int dh, int rot, float eps) {
    extern __shared__ __nv_bfloat16 normed[];  // [heads][dh]
    const int row = blockIdx.x, head = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (head >= heads) return;
    // the row's position inside its segment: the last offset not past it
    int lo = 0, hi = segs;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (offsets[mid] <= row) lo = mid; else hi = mid;
    }
    const float pos = static_cast<float>(row - offsets[lo]);
    const int w = heads * dh;
    const int half = rot / 2;
    for (int part = 0; part < 2; ++part) {
        __nv_bfloat16* base = qkv + static_cast<size_t>(row) * 3 * w + part * w + head * dh;
        const __nv_bfloat16* weight = part == 0 ? qw : kw;
        const __nv_bfloat16* bias = part == 0 ? qb : kb;
        float x[4];
        float sum = 0.f, squares = 0.f;
        const int per = dh / 32;
        for (int e = 0; e < per; ++e) {
            x[e] = __bfloat162float(base[lane + e * 32]);
            sum += x[e];
            squares += x[e] * x[e];
        }
        sum = warp_sum(sum);
        squares = warp_sum(squares);
        const float mean = sum / static_cast<float>(dh);
        const float rstd = rsqrtf(squares / static_cast<float>(dh) - mean * mean + eps);
        for (int e = 0; e < per; ++e) {
            const int t = lane + e * 32;
            normed[head * dh + t] = __float2bfloat16_rn(
                (x[e] - mean) * rstd * __bfloat162float(weight[t]) + __bfloat162float(bias[t]));
        }
        __syncwarp();
        for (int p = lane; p < half; p += 32) {
            const float a = pos * inv[p];
            const float c = bf(cosf(a)), s = bf(sinf(a));
            const float n1 = __bfloat162float(normed[head * dh + p]);
            const float n2 = __bfloat162float(normed[head * dh + p + half]);
            base[p] = __float2bfloat16_rn(bf(n1 * c) + bf(-n2 * s));
            base[p + half] = __float2bfloat16_rn(bf(n2 * c) + bf(n1 * s));
        }
        __syncwarp();
        // the unrotated tail (and the rotated dims' second reads done): write the normed values back
        for (int e = 0; e < per; ++e) {
            const int t = lane + e * 32;
            if (t >= rot) base[t] = normed[head * dh + t];
        }
        __syncwarp();
    }
}

}  // namespace

// qkv bf16 [R, 3 W] in place, offsets int32 [S + 1] (0 .. R), qw/qb and kw/kb bf16 [DH], inv fp32 [ROT / 2]:
// per head LayerNorm of q and k then RoPE on the first ROT dims of DH (heads heads, dh 128, rot <= dh even).
extern "C" int zs_nemotron_perception_prep(const zs_view* qkv_, const zs_view* offsets_, const zs_view* qw_,
                                           const zs_view* qb_, const zs_view* kw_, const zs_view* kb_,
                                           const zs_view* inv_, int64_t heads_, int64_t rot_, double eps) {
    zs::begin();
    const zs::View qkv(qkv_), offsets(offsets_), qw(qw_), qb(qb_), kw(kw_), kb(kb_), inv(inv_);
    const int64_t heads = heads_, rot = rot_;
    ZS_REQUIRE(qkv.defined() && qkv.dtype() == ZS_BF16 && qkv.dim() == 2 && qkv.is_contiguous() &&
                   qkv.size(0) >= 1 && qkv.size(0) <= INT32_MAX && qkv.size(1) % 3 == 0,
               "qkv: bf16 [R, 3W]");
    const int64_t width = qkv.size(1) / 3;
    ZS_REQUIRE(heads >= 1 && heads <= 32 && width % heads == 0, "heads divides W");
    const int64_t dh = width / heads;
    ZS_REQUIRE(dh % 32 == 0 && dh <= 128, "a head's dims in 32s, at most 128");
    ZS_REQUIRE(rot >= 2 && rot <= dh && rot % 2 == 0, "rot <= a head's dims, even");
    ZS_REQUIRE(offsets.defined() && offsets.dtype() == ZS_I32 && offsets.is_contiguous() && offsets.size(0) >= 2,
               "offsets: int32 [S + 1]");
    ZS_REQUIRE(qw.defined() && qw.dtype() == ZS_BF16 && qw.numel() == dh && qb.defined() && qb.numel() == dh &&
                   kw.defined() && kw.numel() == dh && kb.defined() && kb.numel() == dh,
               "q_norm, k_norm: bf16 [DH], weight and bias");
    ZS_REQUIRE(inv.defined() && inv.dtype() == ZS_F32 && inv.numel() == rot / 2, "inv: fp32 [ROT / 2]");
    nemotron_prep_kernel<<<static_cast<unsigned>(qkv.size(0)), static_cast<unsigned>(heads * 32),
                           heads * dh * sizeof(__nv_bfloat16), zs::stream()>>>(
        qkv.data_ptr<__nv_bfloat16>(), offsets.data_ptr<int>(), static_cast<int>(offsets.size(0) - 1),
        qw.data_ptr<__nv_bfloat16>(), qb.data_ptr<__nv_bfloat16>(), kw.data_ptr<__nv_bfloat16>(),
        kb.data_ptr<__nv_bfloat16>(), inv.data_ptr<float>(), static_cast<int>(heads), static_cast<int>(dh),
        static_cast<int>(rot), static_cast<float>(eps));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}
