// Unlimited-OCR's MoE glue (spec 137 WS-04): a row's routing from its gate's bf16 logits, the routed experts'
// outputs weighted and added to the shared MLP's and to the residual. The residual's plain add is common/glue.cu's
// (`zs_residual_add`, `zs_residual_rmsnorm` with a bf16 branch; spec 142 WS-04 moved it there from here).
// Semantics from the checkpoint's modeling_deepseekv2.py (MoEGate.forward with scoring_func softmax, topk_method
// greedy, norm_topk_prob false, routed_scaling_factor 1.0; DeepseekV2MoE.moe_infer; the decoder layer's adds),
// Apache-2.0, read 2026-10-25; written for zspark in CUDA C++. Under `infer`'s bf16 autocast the gate's
// `F.linear` is a bf16 product (the spec's Decisions): its logits come here as bf16, the softmax is fp32.
// qwen4/moe.cu's selection does not serve: it renormalizes over the picks, rounds the weights to bf16 and keeps a
// shared expert's sigmoid slot. Hand-written because none of it is a GEMM. No shared memory but the route's row
// and the norm's (below).
//
// The combine may carry the RMS norm that reads its sum next (WS-07: the next layer's input norm, the final norm),
// in one launch a row where there were two: common/glue.cu's `zs_rmsnorm` to the bit, by common/glue.cuh's
// `norm_row` (its header has the folds). Shared memory a block: 1,028 B (the folds' 256 floats, the inverse).

#include <cuda_bf16.h>
#include <stdint.h>

#include <type_traits>

#include "../common/glue.cuh"
#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;
using common_glue::MAX_OWN;
using common_glue::NORM_THREADS;
using common_glue::norm_row;

constexpr int MAX_EXPERTS = 256;
constexpr int MAX_TOP = 16;

// Block r, a thread an expert: the softmax of the row's E logits in fp32 (the largest, then the exps' sum, each
// a fold over a power of two in a fixed order), then the TOP largest probabilities, the lower id first among
// equal ones, written in ascending id with their probabilities unnormalized (norm_topk_prob false, scale 1).
// An expert's place is the count of experts before it in that order, which is total: places 0 .. E - 1 once each.
// With MEMBERS (one row alone, WS-07), the row's plan too: what common/plan.cu's `zs_experts_plan` writes for TOP
// distinct picks in ascending id, an item a pick (expert, slot, 1) and the slots in order, in this launch.
__global__ void __launch_bounds__(MAX_EXPERTS)
    unlimited_ocr_route_kernel(const __nv_bfloat16* __restrict__ LOGITS, int* __restrict__ PICKS,
                               float* __restrict__ WEIGHTS, int E, int TOP, int F, int* __restrict__ MEMBERS,
                               int* __restrict__ ITEMS, int* __restrict__ COUNTS) {
  __shared__ float fold[MAX_EXPERTS];
  __shared__ float p[MAX_EXPERTS];
  __shared__ int chosen[MAX_EXPERTS];
  const int r = blockIdx.x, e = threadIdx.x;
  const float x = e < E ? __bfloat162float(LOGITS[(size_t)r * E + e]) : minus_inf();
  fold[e] = x;
  __syncthreads();
  for (int half = F / 2; half > 0; half >>= 1) {
    if (e < half) fold[e] = fmaxf(fold[e], fold[e + half]);
    __syncthreads();
  }
  const float most = fold[0];
  __syncthreads();
  const float ex = e < E ? expf(x - most) : 0.f;
  fold[e] = ex;
  __syncthreads();
  for (int half = F / 2; half > 0; half >>= 1) {
    if (e < half) fold[e] = fold[e] + fold[e + half];
    __syncthreads();
  }
  const float prob = ex / fold[0];
  p[e] = prob;
  __syncthreads();
  int place = 0;
  if (e < E)
    for (int o = 0; o < E; ++o) place += (p[o] > prob) | ((p[o] == prob) & (o < e));
  chosen[e] = e < E && place < TOP;
  __syncthreads();
  if (chosen[e]) {
    int slot = 0;
    for (int o = 0; o < e; ++o) slot += chosen[o];
    PICKS[(size_t)r * TOP + slot] = e;
    WEIGHTS[(size_t)r * TOP + slot] = prob;
    if (MEMBERS != nullptr) {
      MEMBERS[slot] = slot;
      ITEMS[3 * slot] = e;
      ITEMS[3 * slot + 1] = slot;
      ITEMS[3 * slot + 2] = 1;
    }
  }
  if (MEMBERS != nullptr && e == 0) {
    COUNTS[0] = TOP;
    COUNTS[1] = TOP;
  }
}

// Thread (row, d): h = bf16(h + bf16(bf16(sum_k w_k y_k) + shared)), the sum in fp32 over the row's picks in
// their order (ascending id), each y_k the expert's bf16 output: moe_infer's fp32 products and sum rounded to
// bf16, the shared MLP's output added in bf16, then the residual's bf16 add.
__global__ void unlimited_ocr_combine_kernel(const __nv_bfloat16* __restrict__ Y, const float* __restrict__ WEIGHTS,
                                             const __nv_bfloat16* __restrict__ SHARED, __nv_bfloat16* __restrict__ H,
                                             int R, int D, int TOP) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const int r = static_cast<int>(i / D), d = static_cast<int>(i % D);
  float sum = 0.f;
  for (int k = 0; k < TOP; ++k)
    sum += WEIGHTS[(size_t)r * TOP + k] * __bfloat162float(Y[((size_t)r * TOP + k) * D + d]);
  const float moe = bf(bf(sum) + __bfloat162float(SHARED[i]));
  H[i] = __float2bfloat16_rn(__bfloat162float(H[i]) + moe);
}

// Block r: the combine's sum of the row into H, then its norm into OUT.
template <int OWN>
__global__ void __launch_bounds__(NORM_THREADS)
    unlimited_ocr_combine_norm_kernel(const __nv_bfloat16* __restrict__ Y, const float* __restrict__ WEIGHTS,
                                      const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ H,
                                      const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ OUT, int D,
                                      int P, int TOP, float eps) {
  const int r = blockIdx.x, tid = threadIdx.x;
  float v[OWN];
#pragma unroll
  for (int k = 0; k < OWN; ++k) {
    const int d = tid + NORM_THREADS * k;
    v[k] = 0.f;
    if (d >= D) continue;
    const size_t i = (size_t)r * D + d;
    float x = __bfloat162float(X[i]);
    float sum = 0.f;
    for (int j = 0; j < TOP; ++j)
      sum += WEIGHTS[(size_t)r * TOP + j] * __bfloat162float(Y[((size_t)r * TOP + j) * D + d]);
    x = bf(bf(sum) + x);
    const float h = bf(__bfloat162float(H[i]) + x);
    H[i] = __float2bfloat16_rn(h);
    v[k] = h;
  }
  norm_row<OWN>(v, W, OUT + (size_t)r * D, D, P, eps);
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

// The fused form's launch: OUT bf16 [R, D] the norm of the sum, by w bf16 [D].
int combine_norm(const zs::View& y, const zs::View& weights, const zs::View& x, const zs::View& h, const zs_view* w_,
             const zs_view* out_, double eps, int64_t top) {
  const zs::View w(w_), out(out_);
  const int64_t rows = h.size(0), d = h.size(1);
  ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.numel() == d, "w: bf16 [D]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() == rows * d,
             "out: bf16 [R, D]");
  int p = 1;
  while (p < d) p *= 2;
  ZS_REQUIRE(p >= 512 && p <= NORM_THREADS * MAX_OWN, "the norm's D: 257 to 4,096");
  const __nv_bfloat16* yp = y.data_ptr<__nv_bfloat16>();
  const float* wp = weights.data_ptr<float>();
  auto launch = [&](auto own) {
    unlimited_ocr_combine_norm_kernel<decltype(own)::value><<<static_cast<unsigned>(rows), NORM_THREADS, 0,
                                                              zs::stream()>>>(
        yp, wp, x.data_ptr<__nv_bfloat16>(), h.data_ptr<__nv_bfloat16>(), w.data_ptr<__nv_bfloat16>(),
        out.data_ptr<__nv_bfloat16>(), static_cast<int>(d), p, static_cast<int>(top), static_cast<float>(eps));
  };
  if (p <= 2048) launch(std::integral_constant<int, 8>{});
  else launch(std::integral_constant<int, 16>{});
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // namespace

extern "C" {

// logits bf16 [R, E] (the gate's product) -> picks int32 and weights fp32 [R, top], ascending id. One row may
// take its plan here too (members int32 [top] or more, items [top, 3] or more, counts [2] or more), as
// `zs_experts_plan` with any tile would write it; null for none.
int zs_unlimited_ocr_moe_route(const zs_view* logits_, const zs_view* picks_, const zs_view* weights_,
                               const zs_view* members_, const zs_view* items_, const zs_view* counts_) {
  zs::begin();
  const zs::View logits(logits_), picks(picks_), weights(weights_);
  ZS_REQUIRE(logits.defined() && logits.dtype() == ZS_BF16 && logits.dim() == 2 && logits.is_contiguous(),
             "logits: bf16 [R, E]");
  const int64_t rows = logits.size(0), experts = logits.size(1);
  ZS_REQUIRE(rows >= 1 && rows <= INT32_MAX && experts >= 1 && experts <= MAX_EXPERTS, "1 to 256 experts");
  ZS_REQUIRE(picks.defined() && picks.dtype() == ZS_I32 && picks.dim() == 2 && picks.is_contiguous() &&
                 picks.size(0) == rows,
             "picks: int32 [R, top]");
  const int64_t top = picks.size(1);
  ZS_REQUIRE(top >= 1 && top <= MAX_TOP && top <= experts, "1 to 16 picks a row");
  ZS_REQUIRE(weights.defined() && weights.dtype() == ZS_F32 && weights.is_contiguous() && weights.numel() == rows * top,
             "weights: fp32 [R, top]");
  int* plan[3] = {nullptr, nullptr, nullptr};
  if (members_ != nullptr) {
    const zs::View members(members_), items(items_), counts(counts_);
    ZS_REQUIRE(rows == 1, "a plan here for one row alone");
    ZS_REQUIRE(members.defined() && members.dtype() == ZS_I32 && members.numel() >= top && items.defined() &&
                   items.dtype() == ZS_I32 && items.numel() >= 3 * top && counts.defined() &&
                   counts.dtype() == ZS_I32 && counts.numel() >= 2,
               "the plan: int32 members [top], items [top, 3], counts [2]");
    plan[0] = members.data_ptr<int>();
    plan[1] = items.data_ptr<int>();
    plan[2] = counts.data_ptr<int>();
  }
  int fold = 1;
  while (fold < experts) fold *= 2;
  unlimited_ocr_route_kernel<<<static_cast<unsigned>(rows), static_cast<unsigned>(fold < 32 ? 32 : fold), 0,
                               zs::stream()>>>(logits.data_ptr<__nv_bfloat16>(), picks.data_ptr<int>(),
                                               weights.data_ptr<float>(), static_cast<int>(experts),
                                               static_cast<int>(top), fold, plan[0], plan[1], plan[2]);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// y bf16 [R top, D] (row r top + k: the row's k-th pick's output), weights fp32 [R, top], shared bf16 [R, D] ->
// h bf16 [R, D] in place; with w bf16 [D], out bf16 [R, D] = the norm of h by w (`zs_rmsnorm`'s bits).
int zs_unlimited_ocr_moe_combine(const zs_view* y_, const zs_view* weights_, const zs_view* shared_, const zs_view* h_,
                                 const zs_view* w_, const zs_view* out_, double eps) {
  zs::begin();
  const zs::View y(y_), weights(weights_), shared(shared_), h(h_);
  ZS_REQUIRE(h.defined() && h.dtype() == ZS_BF16 && h.dim() == 2 && h.is_contiguous(), "h: bf16 [R, D]");
  const int64_t rows = h.size(0), d = h.size(1);
  ZS_REQUIRE(weights.defined() && weights.dtype() == ZS_F32 && weights.dim() == 2 && weights.is_contiguous() &&
                 weights.size(0) == rows,
             "weights: fp32 [R, top]");
  const int64_t top = weights.size(1);
  ZS_REQUIRE(y.defined() && y.dtype() == ZS_BF16 && y.is_contiguous() && y.numel() >= rows * top * d,
             "y: bf16 [R top, D]");
  ZS_REQUIRE(shared.defined() && shared.dtype() == ZS_BF16 && shared.is_contiguous() && shared.numel() == rows * d,
             "shared: bf16 [R, D]");
  if (w_ != nullptr) return combine_norm(y, weights, shared, h, w_, out_, eps, top);
  unlimited_ocr_combine_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(
      y.data_ptr<__nv_bfloat16>(), weights.data_ptr<float>(), shared.data_ptr<__nv_bfloat16>(),
      h.data_ptr<__nv_bfloat16>(), static_cast<int>(rows), static_cast<int>(d), static_cast<int>(top));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
