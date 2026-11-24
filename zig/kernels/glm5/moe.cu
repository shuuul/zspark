// GLM-5.3-Flash's MoE glue: each row's top experts and their weights from the router's logits, the SwiGLU of
// the dense and shared MLPs, and a rank's share of the block, the slots times their weights. A row routes and
// runs on its own.
// Semantics from TensorFold 0.6.6 (cb2ebf0) families/glm5_next/cuda/glue.py (_topk, _swiglu, _combine; Triton),
// Apache-2.0; rewritten for zspark in CUDA C++, without the 64-group sums only the MLX 4-bit matmul reads. The
// selection follows zig/kernels/qwen4/moe.cu's (places counted, not TOP rounds of a reduction). Hand-written
// because none of it is a GEMM: a selection and element-wise maps. The router itself is a BF16 linear whose K
// slices the selection sums (bf16/reduce.cuh), not glue.py's `_router_part`.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../bf16/reduce.cuh"
#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

constexpr int MAX_EXPERTS = 1024;
constexpr int MAX_TOP = 32;

// Block r, a thread an expert: row r's TOP experts by sigmoid(logit) + bias, largest first, the lower id among
// equals. A pick's weight is its sigmoid alone, over the picks' sum in pick order (+ 1e-20) when NORM, times
// `scale`; then the shared expert (id NE) in slot TOP with weight 1. logits [R, NE], picks and weights
// [R, TOP + 1].
// An expert's place in that order is how many experts come before it, and the order is total, so the places are
// 0 to NE - 1 once each. A thread counts its expert's place among the 32 of its warp; one of the row's first TOP
// is among its warp's first TOP, so those are the candidates, [warps, min(TOP, 32)], and a thread a candidate
// counts its place among them: under TOP, it is that pick.
// With SK above 0 the logits are not yet there: ``part`` holds the router's K slices [SK, R, NE] unsummed, and
// each thread sums its expert's in slice order (b16_reduce's sum) and writes the logit it routed by.
__global__ void __launch_bounds__(MAX_EXPERTS)
    topk_kernel(float* __restrict__ logits, const float* __restrict__ part, const float* __restrict__ bias,
                int* __restrict__ picks, float* __restrict__ weights, float scale, int NE, int TOP, int R, int SK,
                int NORM) {
  __shared__ float value[MAX_EXPERTS];
  __shared__ float score[MAX_EXPERTS];
  // a warp's candidates, largest first; one it does not have is NONE, which comes after every expert
  __shared__ float cand[MAX_EXPERTS];
  __shared__ int cand_id[MAX_EXPERTS];
  __shared__ int chosen[MAX_TOP];
  constexpr int NONE = 0x7fffffff;
  const int r = blockIdx.x, tid = threadIdx.x, slots = TOP + 1;
  const int each = TOP < 32 ? TOP : 32, candidates = static_cast<int>(blockDim.x) / 32 * each;
  const long long span = (long long)R * NE;  // elements a slice
  float mine = 0.f;
  if (tid < NE) {
    float logit;
    if (SK > 0) {
      logit = b16_slices(part, (long long)r * NE + tid, span, SK);
      logits[(size_t)r * NE + tid] = logit;
    } else {
      logit = logits[(size_t)r * NE + tid];
    }
    const float s = 1.0f / (1.0f + expf(-logit));
    score[tid] = s;
    mine = s + bias[tid];
    value[tid] = mine;
  }
  if (tid < candidates) {
    cand[tid] = -__int_as_float(0x7f800000);
    cand_id[tid] = NONE;
  }
  // a place no candidate takes (a NaN counts no place but the first) is an expert that exists (spec 081)
  if (tid < TOP) chosen[tid] = tid;
  __syncthreads();
  if (tid < NE) {
    // before it: every larger value, and an equal one of a lower id
    const int first = tid & ~31, end = first + 32 < NE ? first + 32 : NE;
    int place = 0;
    for (int e = first; e < tid; ++e) place += value[e] >= mine;
    for (int e = tid + 1; e < end; ++e) place += value[e] > mine;
    if (place < each) {
      cand[(tid >> 5) * each + place] = mine;
      cand_id[(tid >> 5) * each + place] = tid;
    }
  }
  __syncthreads();
  if (tid < candidates && cand_id[tid] != NONE) {
    const float v = cand[tid];
    const int id = cand_id[tid];
    int place = 0;
    // no branch: a warp's threads would part ways at each candidate
    for (int c = 0; c < candidates; ++c) place += (cand[c] > v) | ((cand[c] == v) & (cand_id[c] < id));
    if (place < TOP) chosen[place] = id;
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.f;
    for (int k = 0; k < TOP; ++k) total += score[chosen[k]];
    for (int k = 0; k < TOP; ++k) {
      float w = score[chosen[k]];
      if (NORM) w = w / (total + 1e-20f);
      picks[(size_t)r * slots + k] = chosen[k];
      weights[(size_t)r * slots + k] = w * scale;
    }
    picks[(size_t)r * slots + TOP] = NE;
    weights[(size_t)r * slots + TOP] = 1.0f;
  }
}

// Thread (row, d): a rank's fp32 share of the block, y_k w_k added in slot order from zero, each one fused
// multiply-add; the shared expert is the last slot, its weight 1. y [R * slots, D] fp32; with `shared`, fp32
// [R, D], the last slot's values are read there (a BF16 expert's rows, which no launch writes into y).
__device__ __forceinline__ void put(float* out, float v) { *out = v; }
__device__ __forceinline__ void put(__nv_bfloat16* out, float v) { *out = __float2bfloat16_rn(v); }

// `out` bf16 is the fp32 sum rounded once (a quantized window's partial, spec 116 WS-04).
template <typename OUT>
__global__ void moe_combine_kernel(const float* __restrict__ y, const float* __restrict__ shared,
                                   const float* __restrict__ wts, OUT* __restrict__ out, int R, int D, int slots) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const int r = static_cast<int>(i / D), d = static_cast<int>(i % D);
  const int routed = shared != nullptr ? slots - 1 : slots;
  float acc = 0.f;
  for (int k = 0; k < routed; ++k) acc = fmaf(y[((size_t)r * slots + k) * D + d], wts[(size_t)r * slots + k], acc);
  if (shared != nullptr) acc = fmaf(shared[i], wts[(size_t)r * slots + slots - 1], acc);
  put(out + i, acc);
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

}  // namespace

extern "C" {

// logits fp32 [R, experts], bias fp32 [experts] -> picks int32 and weights fp32 [R, top + 1]; the last slot is
// the shared expert. With ``part``, fp32 [SK, R, experts], the logits are the router's K slices unsummed
// (zs_b16_gemm without ``out``): the selection sums them and writes ``logits`` as well.
int zs_glm5_moe_topk(const zs_view* logits_, const zs_view* bias_, const zs_view* picks_, const zs_view* weights_,
                     int64_t top, double scale, int64_t norm, const zs_view* part_) {
  zs::begin();
  const zs::View logits(logits_), bias(bias_), picks(picks_), weights(weights_), part(part_);
  ZS_REQUIRE(logits.defined() && logits.dtype() == ZS_F32 && logits.dim() == 2 && logits.is_contiguous(),
             "logits: fp32 [R, experts]");
  const int64_t rows = logits.size(0), experts = logits.size(1);
  ZS_REQUIRE(experts >= 1 && experts <= MAX_EXPERTS, "at most 1024 routed experts");
  ZS_REQUIRE(top >= 1 && top <= MAX_TOP && top <= experts, "1 to 32 picks a row");
  ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_F32 && bias.numel() == experts, "bias: fp32 [experts]");
  ZS_REQUIRE(picks.defined() && picks.dtype() == ZS_I32 && picks.is_contiguous() &&
             picks.numel() >= rows * (top + 1), "picks: int32 [R, top + 1]");
  ZS_REQUIRE(weights.defined() && weights.dtype() == ZS_F32 && weights.is_contiguous() &&
             weights.numel() >= rows * (top + 1), "weights: fp32 [R, top + 1]");
  int64_t sk = 0;
  if (part.defined()) {
    ZS_REQUIRE(part.dtype() == ZS_F32 && part.dim() == 3 && part.is_contiguous() && part.size(1) == rows &&
               part.size(2) == experts && part.size(0) >= 1 && part.size(0) <= INT32_MAX && rows <= INT32_MAX,
               "part: fp32 [SK, R, experts]");
    sk = part.size(0);
  }
  // a thread an expert, in whole warps
  const unsigned threads = static_cast<unsigned>((experts + 31) / 32 * 32);
  topk_kernel<<<static_cast<unsigned>(rows), threads, 0, zs::stream()>>>(
      logits.data_ptr<float>(), sk > 0 ? part.data_ptr<float>() : nullptr, bias.data_ptr<float>(),
      picks.data_ptr<int>(), weights.data_ptr<float>(), static_cast<float>(scale), static_cast<int>(experts),
      static_cast<int>(top), static_cast<int>(rows), static_cast<int>(sk), norm != 0);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// y fp32 [R * slots, D], wts fp32 [R, slots] -> out fp32 or bf16 [R, D], this rank's share of the block. ``shared``,
// fp32 [R, D] or null, stands for y's last slot.
int zs_glm5_moe_combine(const zs_view* y_, const zs_view* wts_, const zs_view* out_, const zs_view* shared_) {
  zs::begin();
  const zs::View y(y_), wts(wts_), out(out_), shared(shared_);
  ZS_REQUIRE(wts.defined() && wts.dtype() == ZS_F32 && wts.dim() == 2 && wts.is_contiguous() && wts.size(1) >= 1,
             "wts: fp32 [R, slots]");
  ZS_REQUIRE(out.defined() && (out.dtype() == ZS_F32 || out.dtype() == ZS_BF16) && out.dim() == 2 &&
                 out.is_contiguous() && out.size(0) <= wts.size(0) && out.size(0) <= INT32_MAX,
             "out: fp32 or bf16 [R, D]");
  const int64_t rows = out.size(0), d = out.size(1), slots = wts.size(1);
  ZS_REQUIRE(y.defined() && y.dtype() == ZS_F32 && y.is_contiguous() && y.numel() >= rows * slots * d,
             "y: fp32 [R * slots, D]");
  ZS_REQUIRE(!shared.defined() || (shared.dtype() == ZS_F32 && shared.is_contiguous() && shared.numel() >= rows * d),
             "shared: fp32 [R, D]");
  const float* sh = shared.defined() ? shared.data_ptr<float>() : nullptr;
  if (out.dtype() == ZS_BF16)
    moe_combine_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(y.data_ptr<float>(), sh, wts.data_ptr<float>(),
                                                                   out.data_ptr<__nv_bfloat16>(), static_cast<int>(rows),
                                                                   static_cast<int>(d), static_cast<int>(slots));
  else
    moe_combine_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(y.data_ptr<float>(), sh, wts.data_ptr<float>(),
                                                                   out.data_ptr<float>(), static_cast<int>(rows),
                                                                   static_cast<int>(d), static_cast<int>(slots));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
