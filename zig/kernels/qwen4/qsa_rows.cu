// Spec 018 WS-B3, the pool's (`--parallel` N): qsa.cu's pooling for several streams in one launch. Grid
// (slots, blocks). A slot with no rows, or a block that slot's window does not complete, returns at once.
// The indexer's keys are the layer's pool; a slot's table and its pooled matrix come from ptrs. seg int32
// [slots, 4] is first row, rows, kept, first position. The arithmetic is qsa.cu's pool_kernel, the block's first
// position rotated under its slot's ROPES[slot] (an image prompt's, attention_rope.cuh) where it has one.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/math.cuh"
#include "../zs/host.h"
#include "attention_rope.cuh"

namespace {

using namespace common_math;

constexpr int WIDE = 256;
constexpr int EXTENT = 512, EXTENT_BITS = 9;

__device__ __forceinline__ int stored_at(const int* __restrict__ TABLE, int pos) {
    return (TABLE[pos >> EXTENT_BITS] << EXTENT_BITS) | (pos & (EXTENT - 1));
}

// Block (slot, bi). Pooled block b = first_pos / RATIO + bi, if that slot's rows complete it.
__global__ void __launch_bounds__(WIDE)
    pool_slots_kernel(const int* __restrict__ SEG, const int64_t* __restrict__ PTRS, int stride, int pooled_col,
                      int table_col, const int64_t* __restrict__ BASES, const float* __restrict__ W,
                      const float* __restrict__ INV, float eps, int DI, int HALF, int RATIO,
                      const int* const* __restrict__ ROPES, int S1, int S2) {
    __shared__ float squares[WIDE];
    __shared__ float mean[WIDE];
    const int slot = static_cast<int>(blockIdx.x);
    const int rows = SEG[slot * 4 + 1];
    if (rows <= 0) return;
    const int p0 = SEG[slot * 4 + 3];
    const int b = p0 / RATIO + static_cast<int>(blockIdx.y), d = threadIdx.x;
    if ((long long)RATIO * b + RATIO > (long long)p0 + rows) return;
    const int* TABLE = reinterpret_cast<const int*>(PTRS[static_cast<size_t>(slot) * stride + table_col]);
    const __nv_bfloat16* IKC = reinterpret_cast<const __nv_bfloat16*>(BASES[(size_t)slot * 3 + 2]);
    __nv_bfloat16* POOLED = reinterpret_cast<__nv_bfloat16*>(PTRS[static_cast<size_t>(slot) * stride + pooled_col]);
    const bool live = d < DI;
    float x = 0.f;
    if (live) {
        const __nv_bfloat16* at = IKC + (size_t)stored_at(TABLE, RATIO * b) * DI + d;
        float acc = __bfloat162float(at[0]);
        for (int k = 1; k < RATIO; ++k) acc = acc + __bfloat162float(at[(size_t)k * DI]);
        x = bf(acc / RATIO);
    }
    mean[d] = x;
    squares[d] = x * x;
    __syncthreads();
    for (int half = WIDE / 2; half > 0; half >>= 1) {
        if (d < half) squares[d] += squares[d + half];
        __syncthreads();
    }
    if (!live) return;
    const float rinv = 1.0f / sqrtf(squares[0] / DI + eps);
    const float xn = bf(x * rinv * W[d]);
    float out = xn;
    if (d < 2 * HALF) {
        const int partner = d < HALF ? d + HALF : d - HALF, i = d < HALF ? d : d - HALF;
        const float xpn = bf(mean[partner] * rinv * W[partner]);
        const int* const ROPE = ROPES == nullptr ? nullptr : ROPES[slot];
        const float ang = static_cast<float>(rope_axis(ROPE, RATIO * b, i, S1, S2)) * INV[i];
        const float c = cosf(ang), s = sinf(ang);
        out = d < HALF ? xn * c - xpn * s : xpn * s + xn * c;
    }
    POOLED[(size_t)b * DI + d] = __float2bfloat16_rn(out);
}

}  // namespace

// ikc is the layer's pool. Each slot's pooled matrix and extent table are pointers in ptrs. `blocks` is the
// longest window's grid, rows / ratio + 2; a shorter slot returns inside the kernel. ropes int64 [slots], each
// slot's rope int32 [2 + 3 capacity] or 0 (its own positions), or null for every slot (spec 019 WS-05).
extern "C" int zs_qwen4_attn_pool_slots(const zs_view* seg_, const zs_view* ptrs_, int64_t stride, int64_t pooled_col,
                                        int64_t table_col, const zs_view* bases_, const zs_view* ikc_, const zs_view* scale_,
                                        const zs_view* inv_freq_, double eps, int64_t ratio, int64_t blocks,
                                        const zs_view* ropes_, int64_t s1, int64_t s2) {
    zs::begin();
    const zs::View seg(seg_), ptrs(ptrs_), bases(bases_), ikc(ikc_), scale(scale_), inv_freq(inv_freq_), ropes(ropes_);
    ZS_REQUIRE(seg.defined() && seg.dtype() == ZS_I32 && seg.is_contiguous() && seg.dim() == 2 && seg.size(1) == 4,
               "seg: int32 [slots, 4]");
    const int64_t slots = seg.size(0);
    ZS_REQUIRE(ptrs.defined() && ptrs.dtype() == ZS_I64 && ptrs.is_contiguous() && ptrs.dim() == 2 &&
               ptrs.size(0) == slots && ptrs.size(1) == stride && pooled_col >= 0 && pooled_col < stride &&
               table_col >= 0 && table_col < stride, "ptrs: int64 [slots, stride]");
    ZS_REQUIRE(bases.defined() && bases.dtype() == ZS_I64 && bases.is_contiguous() && bases.dim() == 2 &&
               bases.size(0) == slots && bases.size(1) == 3, "bases: int64 [slots, 3] of k, v, indexer keys");
    ZS_REQUIRE(ikc.defined() && ikc.dtype() == ZS_BF16 && ikc.dim() == 2 && ikc.is_contiguous(),
               "ikc: bf16 [capacity, index_dim]");
    const int64_t di = ikc.size(1), half = inv_freq.numel();
    ZS_REQUIRE(di >= 1 && di <= WIDE, "an indexer key holds at most 256 values");
    ZS_REQUIRE(scale.defined() && scale.dtype() == ZS_F32 && scale.numel() == di, "scale: fp32 [index_dim]");
    ZS_REQUIRE(inv_freq.defined() && inv_freq.dtype() == ZS_F32 && half >= 1 && 2 * half <= di, "inv_freq");
    ZS_REQUIRE(ratio >= 1 && EXTENT % ratio == 0 && ikc.size(0) % EXTENT == 0, "ratio divides 512");
    ZS_REQUIRE(slots >= 1 && slots <= 65535 && blocks >= 1 && blocks <= 65535, "slots and blocks");
    ZS_REQUIRE(!ropes.defined() || (ropes.dtype() == ZS_I64 && ropes.is_contiguous() && ropes.numel() >= slots),
               "ropes: int64 [slots]");
    pool_slots_kernel<<<dim3(static_cast<unsigned>(slots), static_cast<unsigned>(blocks)), WIDE, 0, zs::stream()>>>(
        seg.data_ptr<int>(), ptrs.data_ptr<int64_t>(), static_cast<int>(stride), static_cast<int>(pooled_col),
        static_cast<int>(table_col), bases.data_ptr<int64_t>(), scale.data_ptr<float>(),
        inv_freq.data_ptr<float>(), static_cast<float>(eps), static_cast<int>(di), static_cast<int>(half),
        static_cast<int>(ratio),
        ropes.defined() ? reinterpret_cast<const int* const*>(ropes.data_ptr<int64_t>()) : nullptr,
        static_cast<int>(s1), static_cast<int>(s2));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}
