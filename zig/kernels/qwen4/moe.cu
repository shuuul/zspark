// Flash Next's MoE glue: each row's top experts and their weights from the router's logits, the shared expert's
// SwiGLU. A row routes and runs on its own.
// Semantics from TensorFold 0.6.5 (609ca41) cuda/moe.py (_topk_rows, Triton) and
// families/qwen4_exp/cuda/nvfp4_moe.py (shared_act), Apache-2.0; rewritten for zspark in CUDA C++. Hand-written
// because none of it is a GEMM: a selection and an element-wise map.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../bf16/reduce.cuh"
#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

constexpr int MAX_EXPERTS = 1024;
constexpr int MAX_TOP = 32;

// Block r, a thread an expert: row r's TOP experts by fp32 logit, largest first, the lower id among equal logits,
// with weights bf16(exp(l_k - l_0) / sum) (the sum in pick order); then the shared expert (id NE) in slot TOP with
// weight bf16(sigmoid(bf16(its gate logit))). logits [R, NE + 1], picks and weights [R, TOP + 1].
// An expert's place in that order is how many experts come before it, and the order is total, so the places are
// 0 to NE - 1 once each. A thread counts its expert's place among the 32 of its warp; one of the row's first TOP
// is among its warp's first TOP, so those are the candidates, [warps, min(TOP, 32)], and a thread a candidate
// counts its place among them: under TOP, it is that pick. TOP rounds of a reduction over 1,024 slots, which
// this replaces, took 16.7 us a launch for the same picks and the same weights' bits (spec 028).
// With SK above 0 the logits are not yet there: ``part`` holds the router's K slices [SK, R, NE + 1] unsummed,
// and each thread sums its expert's in slice order (b16_reduce's sum) and writes the logit it routed by.
__global__ void __launch_bounds__(MAX_EXPERTS)
    topk_kernel(float* __restrict__ logits, const float* __restrict__ part, int* __restrict__ picks,
                float* __restrict__ weights, int NE, int TOP, int R, int SK) {
  __shared__ float value[MAX_EXPERTS];
  // a warp's candidates, largest first; one it does not have is NONE, which comes after every expert
  __shared__ float cand[MAX_EXPERTS];
  __shared__ int cand_id[MAX_EXPERTS];
  __shared__ float taken[MAX_TOP];
  __shared__ float exps[MAX_TOP];
  __shared__ int chosen[MAX_TOP];
  constexpr int NONE = 0x7fffffff;
  const int r = blockIdx.x, tid = threadIdx.x, NL = NE + 1, slots = TOP + 1;
  const int each = TOP < 32 ? TOP : 32, candidates = static_cast<int>(blockDim.x) / 32 * each;
  const long long span = (long long)R * NL;  // elements a slice
  float mine = 0.f;
  if (tid < NE) {
    if (SK > 0) {
      mine = b16_slices(part, (long long)r * NL + tid, span, SK);
      logits[(size_t)r * NL + tid] = mine;
    } else {
      mine = logits[(size_t)r * NL + tid];
    }
    value[tid] = mine;
  }
  if (tid < candidates) {
    cand[tid] = -__int_as_float(0x7f800000);
    cand_id[tid] = NONE;
  }
  // A place no candidate takes keeps an expert that exists: with a NaN among a row's logits several candidates
  // count the same place and others are left, which were shared memory never written (spec 081). Rows without
  // one write every place below, so their bits are what they were.
  if (tid < TOP) {
    chosen[tid] = tid;
    taken[tid] = -__int_as_float(0x7f800000);
  }
  __syncthreads();
  if (tid < NE) {
    // before it: every larger logit, and an equal one of a lower id
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
    if (place < TOP) {
      chosen[place] = id;
      taken[place] = v;
    }
  }
  __syncthreads();
  if (tid < TOP) exps[tid] = expf(taken[tid] - taken[0]);
  __syncthreads();
  if (tid == 0) {
    float total = 0.f;
    for (int k = 0; k < TOP; ++k) total += exps[k];
    for (int k = 0; k < TOP; ++k) {
      picks[(size_t)r * slots + k] = chosen[k];
      weights[(size_t)r * slots + k] = bf(exps[k] / total);
    }
    float shared_gate = 0.f;
    if (SK > 0) {
      shared_gate = b16_slices(part, (long long)r * NL + NE, span, SK);
      logits[(size_t)r * NL + NE] = shared_gate;
    } else {
      shared_gate = logits[(size_t)r * NL + NE];
    }
    const float gate = bf(shared_gate);
    picks[(size_t)r * slots + TOP] = NE;
    weights[(size_t)r * slots + TOP] = bf(1.0f / (1.0f + expf(-gate)));
  }
}

// Thread (row, i): the shared expert's act = bf16(bf16(silu(gate)) * up) from gu [R, 2 NI] (gate | up).
__global__ void shared_act_kernel(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ act, int R, int NI) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * NI) return;
  const int r = static_cast<int>(i / NI), c = static_cast<int>(i % NI);
  const float gate = __bfloat162float(gu[(size_t)r * 2 * NI + c]);
  const float up = __bfloat162float(gu[(size_t)r * 2 * NI + NI + c]);
  act[i] = __float2bfloat16_rn(bf(gate / (1.0f + expf(-gate))) * up);
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

}  // namespace

extern "C" {

// logits fp32 [R, experts + 1] -> picks int32 and weights fp32 [R, top + 1]; the last slot is the shared expert.
// With ``part``, fp32 [SK, R, experts + 1], the logits are the router's K slices unsummed (zs_b16_gemm without
// ``out``): the selection sums them and writes ``logits`` as well.
int zs_qwen4_moe_topk(const zs_view* logits_, const zs_view* picks_, const zs_view* weights_, int64_t top,
                   const zs_view* part_) {
  zs::begin();
  const zs::View logits(logits_), picks(picks_), weights(weights_), part(part_);
  ZS_REQUIRE(logits.defined() && logits.dtype() == ZS_F32 && logits.dim() == 2 && logits.is_contiguous(),
             "logits: fp32 [R, experts + 1]");
  const int64_t rows = logits.size(0), experts = logits.size(1) - 1;
  ZS_REQUIRE(experts >= 1 && experts < MAX_EXPERTS, "at most 1023 routed experts");
  ZS_REQUIRE(top >= 1 && top <= MAX_TOP && top <= experts, "1 to 32 picks a row");
  ZS_REQUIRE(picks.defined() && picks.dtype() == ZS_I32 && picks.is_contiguous() &&
             picks.numel() >= rows * (top + 1), "picks: int32 [R, top + 1]");
  ZS_REQUIRE(weights.defined() && weights.dtype() == ZS_F32 && weights.is_contiguous() &&
             weights.numel() >= rows * (top + 1), "weights: fp32 [R, top + 1]");
  int64_t sk = 0;
  if (part.defined()) {
    ZS_REQUIRE(part.dtype() == ZS_F32 && part.dim() == 3 && part.is_contiguous() && part.size(1) == rows &&
               part.size(2) == experts + 1 && part.size(0) >= 1 && part.size(0) <= INT32_MAX && rows <= INT32_MAX,
               "part: fp32 [SK, R, experts + 1]");
    sk = part.size(0);
  }
  // a thread an expert, in whole warps
  const unsigned threads = static_cast<unsigned>((experts + 31) / 32 * 32);
  topk_kernel<<<static_cast<unsigned>(rows), threads, 0, zs::stream()>>>(
      logits.data_ptr<float>(), sk > 0 ? part.data_ptr<float>() : nullptr, picks.data_ptr<int>(),
      weights.data_ptr<float>(), static_cast<int>(experts), static_cast<int>(top), static_cast<int>(rows),
      static_cast<int>(sk));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// gu bf16 [R, 2 width] (gate | up) -> act bf16 [R, width].
int zs_qwen4_moe_shared_act(const zs_view* gu_, const zs_view* act_) {
  zs::begin();
  const zs::View gu(gu_), act(act_);
  ZS_REQUIRE(gu.defined() && gu.dtype() == ZS_BF16 && gu.dim() == 2 && gu.is_contiguous() && gu.size(1) % 2 == 0,
             "gu: bf16 [R, 2 width]");
  const int64_t rows = gu.size(0), width = gu.size(1) / 2;
  ZS_REQUIRE(act.defined() && act.dtype() == ZS_BF16 && act.is_contiguous() && act.numel() >= rows * width,
             "act: bf16 [R, width]");
  shared_act_kernel<<<blocks(rows * width), 256, 0, zs::stream()>>>(
      gu.data_ptr<__nv_bfloat16>(), act.data_ptr<__nv_bfloat16>(), static_cast<int>(rows), static_cast<int>(width));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
