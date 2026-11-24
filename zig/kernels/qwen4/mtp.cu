// Flash Next's MTP head, before its layer: the norms of the next token's embedding and of the main model's final
// streams, and the sum that makes the head's own streams of them. Every value is a function of its own row alone,
// with fp32 sums in a fixed order and each bf16 rounding where the model has one.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/glue.py (`_rmsnorm` with one group,
// `_add_streams`; Triton), Apache-2.0; rewritten for zspark in CUDA C++, without the 32-group sums only the MLX
// 4-bit matmul reads. Triton leaves the order of the squares' sum to its compiler; here it is fixed.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"

namespace {

constexpr int BLOCK = 256;      // threads of a row's block; a row's width is a multiple of it

// Block (row) of BLOCK threads: thread t sums the squares of values t, t + BLOCK, ... in order, the threads' sums
// fold in halves, and each value leaves as bf16(x / rms * scale), the rms over the whole row.
__global__ void __launch_bounds__(BLOCK)
    rmsnorm_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ scale,
                   __nv_bfloat16* __restrict__ out, float eps, int W) {
  __shared__ float squares[BLOCK];
  const int tid = threadIdx.x;
  const size_t row = (size_t)blockIdx.x * W;
  float acc = 0.f;
  for (int i = tid; i < W; i += BLOCK) {
    const float v = __bfloat162float(x[row + i]);
    acc += v * v;
  }
  squares[tid] = acc;
  __syncthreads();
  for (int half = BLOCK / 2; half > 0; half >>= 1) {
    if (tid < half) squares[tid] += squares[tid + half];
    __syncthreads();
  }
  const float rinv = 1.0f / sqrtf(squares[0] / W + eps);
  for (int i = tid; i < W; i += BLOCK)
    out[row + i] = __float2bfloat16_rn(__bfloat162float(x[row + i]) * rinv * scale[i]);
}

// Block (tile, row) of rmsnorm_kernel's shape, after TensorFold ca8f24bb `rms_tiles.cuh` (spec 152 WS-07):
// a 10,240-wide row on one block leaves most of the GB10's 48 SMs idle, so each of `tiles` blocks repeats
// this file's inv_rms (the same 256-thread loop and shared tree, so the row's bits are unchanged) and writes
// only its own WIDTH / tiles columns. Not taken device code: TF sums differently (`__fmaf_rn`/`__fadd_rn`,
// 1,024 threads); zspark's order is kept.
__global__ void __launch_bounds__(BLOCK)
    rmsnorm_tiles_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ scale,
                         __nv_bfloat16* __restrict__ out, float eps, int W, int tiles) {
  __shared__ float squares[BLOCK];
  const int tid = threadIdx.x;
  const size_t row = (size_t)blockIdx.y * W;
  float acc = 0.f;
  for (int i = tid; i < W; i += BLOCK) {
    const float v = __bfloat162float(x[row + i]);
    acc += v * v;
  }
  squares[tid] = acc;
  __syncthreads();
  for (int half = BLOCK / 2; half > 0; half >>= 1) {
    if (tid < half) squares[tid] += squares[tid + half];
    __syncthreads();
  }
  const float rinv = 1.0f / sqrtf(squares[0] / W + eps);
  const int span = W / tiles, first = blockIdx.x * span;
  for (int i = tid; i < span; i += BLOCK) {
    const int c = first + i;
    out[row + c] = __float2bfloat16_rn(__bfloat162float(x[row + c]) * rinv * scale[c]);
  }
}

// Thread (row, d): each stream's value bf16(e + hs_s).
__global__ void add_streams_kernel(const __nv_bfloat16* __restrict__ e, const __nv_bfloat16* __restrict__ hs,
                                   __nv_bfloat16* __restrict__ out, int R, int D, int S) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const int r = static_cast<int>(i / D), d = static_cast<int>(i % D);
  const float ev = __bfloat162float(e[i]);
  for (int s = 0; s < S; ++s) {
    const size_t at = ((size_t)r * S + s) * D + d;
    out[at] = __float2bfloat16_rn(ev + __bfloat162float(hs[at]));
  }
}

}  // namespace

extern "C" {

// x bf16 [R, W], scale fp32 [W] -> out bf16 [R, W]: each row over its own RMS, times the scale.
int zs_qwen4_mtp_rmsnorm(const zs_view* x_, const zs_view* scale_, const zs_view* out_, double eps) {
  zs::begin();
  const zs::View x(x_), scale(scale_), out(out_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(1) % BLOCK == 0,
             "x: bf16 [R, W], W in blocks of 256");
  const int64_t rows = x.size(0), wide = x.size(1);
  ZS_REQUIRE(scale.defined() && scale.dtype() == ZS_F32 && scale.numel() == wide, "scale: fp32 [W]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * wide,
             "out: bf16 [R, W]");
  // At Flash Next's MTP width each tile block repeats the row's sums and writes a slice (same bytes as
  // rmsnorm_kernel; TensorFold ca8f24bb's sm_121 row of its tiles table: 10 tiles at rows <= 4, else 1).
  const int tiles = (wide == 10240) ? (rows <= 4 ? 10 : 1) : 1;
  if (tiles > 1) {
    const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(rows));
    rmsnorm_tiles_kernel<<<grid, BLOCK, 0, zs::stream()>>>(x.data_ptr<__nv_bfloat16>(),
        scale.data_ptr<float>(), out.data_ptr<__nv_bfloat16>(), static_cast<float>(eps),
        static_cast<int>(wide), tiles);
  } else {
    rmsnorm_kernel<<<static_cast<unsigned>(rows), BLOCK, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), scale.data_ptr<float>(), out.data_ptr<__nv_bfloat16>(),
        static_cast<float>(eps), static_cast<int>(wide));
  }
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// e bf16 [R, D], hs bf16 [R * S, D] -> out bf16 [R, S * D].
int zs_qwen4_mtp_add_streams(const zs_view* e_, const zs_view* hs_, const zs_view* out_, int64_t streams) {
  zs::begin();
  const zs::View e(e_), hs(hs_), out(out_);
  ZS_REQUIRE(e.defined() && e.dtype() == ZS_BF16 && e.dim() == 2 && e.is_contiguous(), "e: bf16 [R, D]");
  const int64_t rows = e.size(0), d = e.size(1);
  ZS_REQUIRE(streams >= 1 && hs.defined() && hs.dtype() == ZS_BF16 && hs.is_contiguous() &&
             hs.numel() >= rows * streams * d, "hs: bf16 [R * S, D]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * streams * d,
             "out: bf16 [R, S * D]");
  add_streams_kernel<<<static_cast<unsigned>((rows * d + 255) / 256), 256, 0, zs::stream()>>>(
      e.data_ptr<__nv_bfloat16>(), hs.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(),
      static_cast<int>(rows), static_cast<int>(d), static_cast<int>(streams));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
