// A row of logits' log-sum-exp and its best k tokens (OpenAI `logprobs`, spec 101 WS-02), one block a row.
// Taken from TensorFold 1.0.12 (833bbfc), zig/kernels/cuda/logprob_rows.cuh, Apache-2.0: the namespace below is
// upstream's byte for byte; the launcher after it is zspark's (sample.cu's `tf_logprob_rows` launched it there).
// Hand-written as upstream's: a selection with a strict total order (logit down, id up), not a product.
//
// Every word of a row is a function of that row's logits alone: a fixed tree (lanes, then warp 0 over the 32
// warps), each thread's columns in a fixed stride, and nothing of the grid. So a row's words are the same in
// a launch of one row and among others, as the row rule asks of the logits they are made from.
//
// zspark calls it with `pick` = V (no pick: the chosen token's logit is read on the host, from the row it
// chooses from, so out[1] is NaN and unread) and no rank (out[42] is 0).
//
// Shared memory: 32 x 4 floats and words of the warps' parts and the last pick taken (0.4 KB), the least a
// tree over 32 warps meets in; registers hold the rest.
// logprobs (lanes/logprob.zig): log normalizer, the pick's logit, the k best ids and logits (logit desc, id asc), rank.
#pragma once

#include <cuda_bf16.h>
#include <math.h>

namespace tf_logprob {

constexpr unsigned TG = 1024;  // one block a row, as tf_logprob_rows launches it
constexpr unsigned NSG = TG / 32;
constexpr unsigned FULL = 0xFFFFFFFFu;
constexpr unsigned LP_TOP = 20;
constexpr unsigned LP_WORDS = 3 + 2 * LP_TOP;

// (m, s): a running max and the sum of exp(x - m) under it, merged with another part's
__device__ __forceinline__ void lse_merge(float& m, float& s, float m2, float s2) {
    if (m2 == -INFINITY) return;
    if (m == -INFINITY) {
        m = m2;
        s = s2;
    } else if (m2 > m) {
        s = s * expf(m - m2) + s2;
        m = m2;
    } else {
        s += s2 * expf(m2 - m);
    }
}

__device__ __forceinline__ bool better(float v, unsigned i, float bv, unsigned bi) { return v > bv || (v == bv && i < bi); }

// One row: a fixed reduction tree (lane tree, then warp 0 over the warps), so a row's words never depend on timing.
__device__ void logprob_row(const __nv_bfloat16* L, unsigned V, unsigned pick, unsigned k, bool rank, unsigned* out) {
    __shared__ float part_m[NSG], part_s[NSG], part_v[NSG];
    __shared__ unsigned part_i[NSG];
    __shared__ float prev_v;
    __shared__ unsigned prev_i;
    const unsigned t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    float m = -INFINITY, s = 0.0f;
    for (unsigned i = t; i < V; i += TG) {
        const float v = __bfloat162float(L[i]);
        if (v == -INFINITY) continue;
        if (v > m) {
            s = s * expf(m - v) + 1.0f;
            m = v;
        } else {
            s += expf(v - m);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float m2 = __shfl_down_sync(FULL, m, off), s2 = __shfl_down_sync(FULL, s, off);
        lse_merge(m, s, m2, s2);
    }
    if (lane == 0) {
        part_m[warp] = m;
        part_s[warp] = s;
    }
    if (t == 0) {
        prev_v = INFINITY;
        prev_i = 0;
    }
    __syncthreads();
    if (warp == 0) {
        m = part_m[lane];
        s = part_s[lane];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float m2 = __shfl_down_sync(FULL, m, off), s2 = __shfl_down_sync(FULL, s, off);
            lse_merge(m, s, m2, s2);
        }
        if (lane == 0) {
            out[0] = __float_as_uint(m + logf(s));
            out[1] = __float_as_uint(pick < V ? __bfloat162float(L[pick]) : NAN);
        }
    }
    if (rank) {
        // vLLM's rank: the entries whose logit is at or above the pick's, the pick itself included
        const float pk = pick < V ? __bfloat162float(L[pick]) : INFINITY;
        unsigned above = 0;
        for (unsigned i = t; i < V; i += TG) above += __bfloat162float(L[i]) >= pk ? 1u : 0u;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) above += __shfl_down_sync(FULL, above, off);
        if (lane == 0) part_i[warp] = above;
        __syncthreads();
        if (warp == 0) {
            above = part_i[lane];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) above += __shfl_down_sync(FULL, above, off);
            if (lane == 0) out[2 + 2 * LP_TOP] = above;
        }
        __syncthreads();
    } else if (t == 0) {
        out[2 + 2 * LP_TOP] = 0;
    }
    for (unsigned r = 0; r < k; ++r) {
        // the best entry after the last one taken: a strict total order, so the result is exact
        const float pv = prev_v;
        const unsigned pi = prev_i;
        float bv = -INFINITY;
        unsigned bi = 0xFFFFFFFFu;
        for (unsigned i = t; i < V; i += TG) {
            const float v = __bfloat162float(L[i]);
            if ((v < pv || (v == pv && i > pi)) && better(v, i, bv, bi)) {
                bv = v;
                bi = i;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float v2 = __shfl_down_sync(FULL, bv, off);
            const unsigned i2 = __shfl_down_sync(FULL, bi, off);
            if (better(v2, i2, bv, bi)) {
                bv = v2;
                bi = i2;
            }
        }
        if (lane == 0) {
            part_v[warp] = bv;
            part_i[warp] = bi;
        }
        __syncthreads();
        if (warp == 0) {
            bv = part_v[lane];
            bi = part_i[lane];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float v2 = __shfl_down_sync(FULL, bv, off);
                const unsigned i2 = __shfl_down_sync(FULL, bi, off);
                if (better(v2, i2, bv, bi)) {
                    bv = v2;
                    bi = i2;
                }
            }
            if (lane == 0) {
                out[2 + r] = bi;
                out[2 + LP_TOP + r] = __float_as_uint(bv);
                prev_v = bv;
                prev_i = bi;
            }
        }
        __syncthreads();
    }
}

}  // namespace tf_logprob

#include "../zs/host.h"

namespace {

__global__ void __launch_bounds__(tf_logprob::TG)
    logprob_rows_kernel(const __nv_bfloat16* __restrict__ logits, unsigned V, unsigned k, unsigned* __restrict__ out) {
  const unsigned row = blockIdx.x;
  tf_logprob::logprob_row(logits + static_cast<size_t>(row) * V, V, V, k, false, out + static_cast<size_t>(row) * tf_logprob::LP_WORDS);
}

}  // namespace

extern "C" {

// logits bf16 [rows, V] (rows contiguous), k 0 to 20 -> out uint32 [rows, 43]: per row [0] the fp32
// log-sum-exp of the row, [2 + j] and [22 + j] for j < k the j-th best id and its logit (fp32 bits), by logit
// down and id up; the rest unspecified.
int zs_logprob_rows(const zs_view* logits_, int64_t k, const zs_view* out_) {
  zs::begin();
  const zs::View logits(logits_), out(out_);
  ZS_REQUIRE(logits.defined() && logits.dtype() == ZS_BF16 && logits.dim() == 2 && logits.is_contiguous() &&
                 logits.size(0) >= 1 && logits.size(0) <= 65535 && logits.size(1) >= 1 && logits.size(1) <= (1 << 30),
             "logits: bf16 [rows, V], contiguous");
  ZS_REQUIRE(k >= 0 && k <= tf_logprob::LP_TOP, "k: 0 to 20");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_U32 && out.is_contiguous() &&
                 out.numel() >= logits.size(0) * tf_logprob::LP_WORDS,
             "out: uint32 [rows, 43]");
  const unsigned rows = static_cast<unsigned>(logits.size(0)), V = static_cast<unsigned>(logits.size(1));
  logprob_rows_kernel<<<rows, tf_logprob::TG, 0, zs::stream()>>>(logits.data_ptr<__nv_bfloat16>(), V,
                                                                  static_cast<unsigned>(k), out.data_ptr<unsigned>());
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
