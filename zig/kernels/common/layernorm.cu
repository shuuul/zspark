// LayerNorm with weight and bias over rows of any width in 8s up to 8,192, with the residual add before it
// (spec 138 WS-03, Qwen3-ASR's audio tower, d 1024, eps 1e-5): h = bf(h + branch) written back when a branch is
// given, then y = LayerNorm(h) in fp32, mean and variance in two passes, (x - mean) * rstd * weight + bias,
// rounded once. One block a row, a thread 8 values (a 16-byte vector), the warps' sums met in shared memory in
// warp order: the order depends on the width alone, so a row's bits are its own (the row rule).
//
// `qwen4/vision.cu`'s `norm_kernel` is the same arithmetic at a fixed 1,152; it could launch this one.
//
// Shared memory: the warps' sums, 32 floats (128 B) a block; nothing else.
//
// Provenance: zspark's own, after `qwen4/vision.cu`; the arithmetic is PyTorch's `nn.LayerNorm` on bf16 rows. Its
// Welford sum and the order of its sums are not PyTorch's: near, not to the bit.

#include <stdint.h>

#include <cuda_bf16.h>

#include "../zs/host.h"

namespace {

constexpr int MAX_THREADS = 1024;

__device__ __forceinline__ float layernorm_block_sum(float v, float* shared) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int warps = blockDim.x / 32;
  __syncthreads();
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  float total = 0.f;
  for (int w = 0; w < warps; ++w) total += shared[w];
  return total;
}

__global__ void __launch_bounds__(MAX_THREADS) layernorm_kernel(__nv_bfloat16* __restrict__ h,
                                                               const __nv_bfloat16* __restrict__ branch,
                                                               const __nv_bfloat16* __restrict__ weight,
                                                               const __nv_bfloat16* __restrict__ bias,
                                                               __nv_bfloat16* __restrict__ y, int width, float eps) {
  __shared__ float shared[32];
  const size_t base = static_cast<size_t>(blockIdx.x) * width;
  const int c = threadIdx.x * 8;
  const bool live = c < width;
  float x[8];
  if (live) {
    uint4 raw = *reinterpret_cast<const uint4*>(h + base + c);
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
    if (branch != nullptr) {
      const uint4 add = *reinterpret_cast<const uint4*>(branch + base + c);
      const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(&add);
      uint4 sum;
      __nv_bfloat16* s = reinterpret_cast<__nv_bfloat16*>(&sum);
#pragma unroll
      for (int i = 0; i < 8; ++i) s[i] = __float2bfloat16_rn(__bfloat162float(v[i]) + __bfloat162float(a[i]));
      *reinterpret_cast<uint4*>(h + base + c) = sum;
      raw = sum;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = __bfloat162float(v[i]);
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = 0.f;
  }
  if (y == nullptr) return;
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum += x[i];
  const float mean = layernorm_block_sum(sum, shared) / static_cast<float>(width);
  float squares = 0.f;
  if (live) {
#pragma unroll
    for (int i = 0; i < 8; ++i) squares += (x[i] - mean) * (x[i] - mean);
  }
  const float rstd = rsqrtf(layernorm_block_sum(squares, shared) / static_cast<float>(width) + eps);
  if (!live) return;
  const uint4 gw = *reinterpret_cast<const uint4*>(weight + c), gb = *reinterpret_cast<const uint4*>(bias + c);
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&gw);
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&gb);
  uint4 out;
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    o[i] = __float2bfloat16_rn((x[i] - mean) * rstd * __bfloat162float(w[i]) + __bfloat162float(b[i]));
  *reinterpret_cast<uint4*>(y + base + c) = out;
}

bool rows_of(const zs::View& v, int64_t p, int64_t width) {
  return v.defined() && v.dtype() == ZS_BF16 && v.is_contiguous() && v.dim() == 2 && v.size(0) == p &&
         v.size(1) == width;
}

}  // namespace

// h bf16 [P, W] += branch (when given, rounded to bf16 and written back), then y = LayerNorm(h) by weight and
// bias bf16 [W] (when y is given). W a multiple of 8, at most 8,192.
extern "C" int zs_layernorm(const zs_view* h_, const zs_view* branch_, const zs_view* weight_, const zs_view* bias_,
                            const zs_view* y_, double eps) {
  zs::begin();
  const zs::View h(h_), branch(branch_), weight(weight_), bias(bias_), y(y_);
  const int64_t p = h.size(0), width = h.size(1);
  ZS_REQUIRE(h.defined() && h.dim() == 2 && width % 8 == 0 && width >= 8 && width <= 8 * MAX_THREADS && p >= 1 &&
                 p <= INT32_MAX && rows_of(h, p, width),
             "h: bf16 [P, W], W in 8s up to 8192");
  ZS_REQUIRE(!branch.defined() || rows_of(branch, p, width), "branch: bf16 [P, W]");
  ZS_REQUIRE(branch.defined() || y.defined(), "an add, a norm or both");
  if (y.defined()) {
    ZS_REQUIRE(rows_of(y, p, width), "y: bf16 [P, W]");
    ZS_REQUIRE(weight.defined() && weight.dtype() == ZS_BF16 && weight.numel() == width && bias.defined() &&
                   bias.dtype() == ZS_BF16 && bias.numel() == width,
               "weight and bias: bf16 [W]");
  }
  const int threads = static_cast<int>((width / 8 + 31) / 32 * 32);
  layernorm_kernel<<<static_cast<unsigned>(p), threads, 0, zs::stream()>>>(
      h.data_ptr<__nv_bfloat16>(), branch.defined() ? branch.data_ptr<__nv_bfloat16>() : nullptr,
      y.defined() ? weight.data_ptr<__nv_bfloat16>() : nullptr, y.defined() ? bias.data_ptr<__nv_bfloat16>() : nullptr,
      y.defined() ? y.data_ptr<__nv_bfloat16>() : nullptr, static_cast<int>(width), static_cast<float>(eps));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
