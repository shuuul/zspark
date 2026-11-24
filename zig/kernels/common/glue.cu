// The row glue two families launch (spec 136 WS-01): a row's RMS norm, the ranks' partials of a branch added in rank
// order, a branch added to a residual, and the SwiGLU of a dense MLP's gate and up. GLM-5.3-Flash's layers, Qwen3.8
// Flash Next's DFlash2 drafter, Qwen3-ASR's language model launch them. Moved from zig/kernels/glm5/hc.cu
// (`zs_glm5_rmsnorm`, `zs_glm5_branch`, `zs_glm5_residual_add`, the last in spec 138 WS-04) and
// glm5/moe.cu (`zs_glm5_swiglu`), the kernels' text unchanged and their entries renamed by layer; hc.cu's shared
// helpers are `glue.cuh`'s. Unlimited-OCR's residual add of a bf16 branch, alone or with the norm after it, moved
// from unlimited_ocr/moe.cu (`zs_unlimited_ocr_add`, spec 142 WS-04): `residual_kernel` reads a bf16 branch as
// `branch_of` does (its one rounding of a bf16 value is exact, so h = bf16(h + x) as there), and
// `residual_held_norm_kernel` is its fused kernel's text. Every value is a function of its own row alone: fp32 sums in an order the shapes fix,
// each bf16 rounding where the model has one.
// Semantics from TensorFold 0.6.6 (cb2ebf0) families/glm5_next/cuda/glue.py (_rmsnorm, _swiglu, the drafter's `_row`;
// Triton), Apache-2.0; rewritten for zspark in CUDA C++. Hand-written because none of it is a GEMM: a row's
// reduction and element-wise maps. No shared memory but the norm's row of squares (4 * MAX_D bytes, folded in
// place) and its one inverse.

#include <cuda_bf16.h>
#include <stdint.h>

#include <type_traits>

#include "../bf16/reduce.cuh"
#include "../zs/host.h"
#include "glue.cuh"
#include "math.cuh"

namespace {

using namespace common_math;

using namespace common_glue;

constexpr int MAX_D = 4096;  // a row the norm folds in shared memory
constexpr int THREADS = 256;

// Block r: out = bf16(w bf16(x / rms)), rms over the row's D values, their squares folded over the next power
// of two (zeros beyond D, as glue.py's masked load).
template <class Weight>
__global__ void __launch_bounds__(THREADS)
    rmsnorm_kernel(const __nv_bfloat16* __restrict__ X, long long x_stride, const Weight* __restrict__ W,
                   __nv_bfloat16* __restrict__ out, long long o_stride, float eps, int D, int P) {
  __shared__ float sq[MAX_D];
  __shared__ float inverse;
  const int r = blockIdx.x, tid = threadIdx.x;
  const __nv_bfloat16* x = X + (size_t)r * x_stride;
  for (int d = tid; d < P; d += THREADS) {
    const float v = d < D ? __bfloat162float(x[d]) : 0.f;
    sq[d] = v * v;
  }
  __syncthreads();
  fold_shared(sq, P);
  if (tid == 0) inverse = 1.0f / sqrtf(sq[0] / D + eps);
  __syncthreads();
  for (int d = tid; d < D; d += THREADS)
    out[(size_t)r * o_stride + d] = __float2bfloat16_rn(weight_of(W[d]) * bf(__bfloat162float(x[d]) * inverse));
}

// Thread (row, d): out = the branch, where nothing is added to it yet (the drafter's `_row`).
__global__ void branch_kernel(__nv_bfloat16* __restrict__ out, const float* __restrict__ G, long long RS, int R, int D,
                              int WORLD) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  out[i] = __float2bfloat16_rn(branch_of(G, RS, WORLD, i));
}

// Thread (row, i): act = bf16(bf16(silu(min(gate, L))) * clip(up, -L, L)) from gu [R, 2 W] (gate | up).
__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ act, float limit,
                              int R, int W) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * W) return;
  const int r = static_cast<int>(i / W), c = static_cast<int>(i % W);
  const float gate = fminf(__bfloat162float(gu[(size_t)r * 2 * W + c]), limit);
  const float up = fminf(fmaxf(__bfloat162float(gu[(size_t)r * 2 * W + W + c]), -limit), limit);
  act[i] = __float2bfloat16_rn(bf(gate / (1.0f + expf(-gate))) * up);
}

// Thread (row, d): x = bf16(x + branch), a plain residual (GLM's MTP layer, Qwen3-ASR's layers; Unlimited-OCR's
// bf16 branch, one rank's).
template <typename G_T>
__global__ void residual_kernel(__nv_bfloat16* __restrict__ X, const G_T* __restrict__ G, long long RS, int R, int D,
                                int WORLD) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  X[i] = __float2bfloat16_rn(__bfloat162float(X[i]) + branch_of(G, RS, WORLD, i));
}

// Block r: rmsnorm_kernel after residual_kernel of one rank, in one launch (spec 138 WS-06): x = bf16(x + bf16(the
// branch)), the branch a split matmul's K slices added in slice order (b16_slices, b16_reduce's sum), then out =
// bf16(w bf16(x / rms)) of the new x, its squares folded as rmsnorm_kernel folds them. The two kernels' bits; the
// slices' reduce, the add and the norm were three launches of 1 to 3.5 us at a decoding row. A row's block is
// RN_THREADS wide: at one row the block is the launch, and 256 threads took 10.4 us over o's four slices of 2,048
// (fold_shared's order does not depend on the threads).
constexpr int RN_THREADS = 1024;
template <class Weight>
__global__ void __launch_bounds__(RN_THREADS)
    residual_rmsnorm_kernel(__nv_bfloat16* __restrict__ X, const float* __restrict__ PART, long long total, int SK,
                            const Weight* __restrict__ W, __nv_bfloat16* __restrict__ out, float eps, int D, int P) {
  __shared__ float sq[MAX_D];
  __shared__ float inverse;
  const int r = blockIdx.x, tid = threadIdx.x;
  __nv_bfloat16* x = X + (size_t)r * D;
  for (int d = tid; d < P; d += RN_THREADS) {
    float v = 0.f;
    if (d < D) {
      const __nv_bfloat16 sum =
          __float2bfloat16_rn(__bfloat162float(x[d]) + bf(b16_slices(PART, (long long)r * D + d, total, SK)));
      x[d] = sum;
      v = __bfloat162float(sum);
    }
    sq[d] = v * v;
  }
  __syncthreads();
  fold_shared(sq, P);
  if (tid == 0) inverse = 1.0f / sqrtf(sq[0] / D + eps);
  __syncthreads();
  for (int d = tid; d < D; d += RN_THREADS)
    out[(size_t)r * D + d] = __float2bfloat16_rn(weight_of(W[d]) * bf(__bfloat162float(x[d]) * inverse));
}

// Block r: x = bf16(x + branch), the branch bf16 [R, D] (Unlimited-OCR's), then out = the norm of the new x by w bf16
// [D] (`norm_row`, `zs_rmsnorm`'s bits). D 257 to 4,096: OWN 8 to a P of 2,048, 16 above.
template <int OWN>
__global__ void __launch_bounds__(NORM_THREADS)
    residual_held_norm_kernel(const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ H,
                              const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ OUT, int D, int P,
                              float eps) {
  const int r = blockIdx.x, tid = threadIdx.x;
  float v[OWN];
#pragma unroll
  for (int k = 0; k < OWN; ++k) {
    const int d = tid + NORM_THREADS * k;
    v[k] = 0.f;
    if (d >= D) continue;
    const size_t i = (size_t)r * D + d;
    float x = __bfloat162float(X[i]);
    const float h = bf(__bfloat162float(H[i]) + x);
    H[i] = __float2bfloat16_rn(h);
    v[k] = h;
  }
  norm_row<OWN>(v, W, OUT + (size_t)r * D, D, P, eps);
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

// The ranks' partials fp32 [world, rows or more, D]: a rank's rows start `size(1) * D` elements after the last's.
bool gathered_of(const zs::View& g, int64_t rows, int64_t d) {
  return g.defined() && g.dtype() == ZS_F32 && g.dim() == 3 && g.size(0) >= 1 && g.size(0) <= 8 && g.size(1) >= rows &&
         g.size(2) == d;
}

// zs_residual_rmsnorm's bf16 branch: unlimited_ocr/moe.cu's fused add's launch, moved.
int residual_held_norm(const zs::View& h, const zs::View& x, const zs::View& w, const zs::View& out, double eps) {
  const int64_t rows = h.size(0), d = h.size(1);
  ZS_REQUIRE(x.dim() == 2 && x.is_contiguous() && x.size(0) == rows && x.size(1) == d, "part: bf16 [R, D]");
  ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.numel() == d, "w: bf16 [D]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() == rows * d,
             "out: bf16 [R, D]");
  int p = 1;
  while (p < d) p *= 2;
  ZS_REQUIRE(p >= 512 && p <= NORM_THREADS * MAX_OWN, "the norm's D: 257 to 4,096");
  auto launch = [&](auto own) {
    residual_held_norm_kernel<decltype(own)::value><<<static_cast<unsigned>(rows), NORM_THREADS, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), h.data_ptr<__nv_bfloat16>(), w.data_ptr<__nv_bfloat16>(),
        out.data_ptr<__nv_bfloat16>(), static_cast<int>(d), p, static_cast<float>(eps));
  };
  if (p <= 2048) launch(std::integral_constant<int, 8>{});
  else launch(std::integral_constant<int, 16>{});
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // namespace

extern "C" {

// x bf16 [R, D] (rows may be a wider buffer's first D columns) and w bf16 or fp32 [D] -> out bf16 [R, D], rows
// contiguous or strided alike. D to 4096.
int zs_rmsnorm(const zs_view* x_, const zs_view* w_, const zs_view* out_, double eps) {
  zs::begin();
  const zs::View x(x_), w(w_), out(out_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.size(0) <= INT32_MAX, "x: bf16 [R, D]");
  const int64_t rows = x.size(0), d = x.size(1);
  ZS_REQUIRE(d >= 1 && d <= MAX_D, "D to 4096");
  ZS_REQUIRE(w.defined() && (w.dtype() == ZS_BF16 || w.dtype() == ZS_F32) && w.numel() == d, "w: bf16 or fp32 [D]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.dim() == 2 && out.size(0) >= rows && out.size(1) == d,
             "out: bf16 [R, D]");
  int p = 1;
  while (p < d) p *= 2;
  if (w.dtype() == ZS_F32)
    rmsnorm_kernel<<<static_cast<unsigned>(rows), THREADS, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), x.row_stride(), w.data_ptr<float>(), out.data_ptr<__nv_bfloat16>(),
        out.row_stride(), static_cast<float>(eps), static_cast<int>(d), p);
  else
    rmsnorm_kernel<<<static_cast<unsigned>(rows), THREADS, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), x.row_stride(), w.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(),
        out.row_stride(), static_cast<float>(eps), static_cast<int>(d), p);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// gathered fp32 [world, R or more, D] -> out bf16 [R, D]: the ranks' partials added rank 0 first.
int zs_branch(const zs_view* gathered_, const zs_view* out_) {
  zs::begin();
  const zs::View g(gathered_), out(out_);
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.dim() == 2 && out.is_contiguous() &&
                 out.size(0) <= INT32_MAX,
             "out: bf16 [R, D]");
  const int64_t rows = out.size(0), d = out.size(1);
  ZS_REQUIRE(gathered_of(g, rows, d), "gathered: fp32 [world, R or more, D]");
  branch_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(
      out.data_ptr<__nv_bfloat16>(), g.data_ptr<float>(), g.size(1) * d, static_cast<int>(rows), static_cast<int>(d),
      static_cast<int>(g.size(0)));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// x bf16 [R, D] in place takes the branch, gathered fp32 [world, R or more, D], or one rank's bf16 [R, D].
int zs_residual_add(const zs_view* x_, const zs_view* gathered_) {
  zs::begin();
  const zs::View x(x_), g(gathered_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(0) <= INT32_MAX,
             "x: bf16 [R, D]");
  const int64_t rows = x.size(0), d = x.size(1);
  if (g.defined() && g.dtype() == ZS_BF16) {
    ZS_REQUIRE(g.dim() == 2 && g.is_contiguous() && g.size(0) == rows && g.size(1) == d, "branch: bf16 [R, D]");
    residual_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), g.data_ptr<__nv_bfloat16>(), rows * d, static_cast<int>(rows),
        static_cast<int>(d), 1);
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
  }
  ZS_REQUIRE(gathered_of(g, rows, d), "gathered: fp32 [world, R or more, D]");
  residual_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), g.data_ptr<float>(), g.size(1) * d, static_cast<int>(rows), static_cast<int>(d),
      static_cast<int>(g.size(0)));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// x bf16 [R, D] in place takes the branch, part fp32 [SK, R, D] (a split matmul's K slices, or SK 1 the product),
// then out bf16 [R, D] = x's RMS norm with w bf16 or fp32 [D]. D to 4096. Or part bf16 [R, D], the branch as its
// product wrote it, and w bf16 [D]: D 257 to 4,096 (`residual_held_norm_kernel`).
int zs_residual_rmsnorm(const zs_view* x_, const zs_view* part_, const zs_view* w_, const zs_view* out_, double eps) {
  zs::begin();
  const zs::View x(x_), part(part_), w(w_), out(out_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(0) <= INT32_MAX,
             "x: bf16 [R, D]");
  const int64_t rows = x.size(0), d = x.size(1);
  ZS_REQUIRE(d >= 1 && d <= MAX_D, "D to 4096");
  if (part.defined() && part.dtype() == ZS_BF16) return residual_held_norm(x, part, w, out, eps);
  ZS_REQUIRE(part.defined() && part.dtype() == ZS_F32 && part.dim() == 3 && part.is_contiguous() &&
                 part.size(0) >= 1 && part.size(0) <= 32 && part.size(1) == rows && part.size(2) == d,
             "part: fp32 [SK, R, D]");
  ZS_REQUIRE(w.defined() && (w.dtype() == ZS_BF16 || w.dtype() == ZS_F32) && w.numel() == d, "w: bf16 or fp32 [D]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.dim() == 2 && out.is_contiguous() &&
                 out.size(0) >= rows && out.size(1) == d,
             "out: bf16 [R, D]");
  int p = 1;
  while (p < d) p *= 2;
  const int sk = static_cast<int>(part.size(0));
  if (w.dtype() == ZS_F32)
    residual_rmsnorm_kernel<<<static_cast<unsigned>(rows), RN_THREADS, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), part.data_ptr<float>(), rows * d, sk, w.data_ptr<float>(),
        out.data_ptr<__nv_bfloat16>(), static_cast<float>(eps), static_cast<int>(d), p);
  else
    residual_rmsnorm_kernel<<<static_cast<unsigned>(rows), RN_THREADS, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), part.data_ptr<float>(), rows * d, sk, w.data_ptr<__nv_bfloat16>(),
        out.data_ptr<__nv_bfloat16>(), static_cast<float>(eps), static_cast<int>(d), p);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// gu bf16 [R, 2 width] (gate | up) -> act bf16 [R, width], gate and up held to `limit`.
int zs_swiglu(const zs_view* gu_, const zs_view* act_, double limit) {
  zs::begin();
  const zs::View gu(gu_), act(act_);
  ZS_REQUIRE(gu.defined() && gu.dtype() == ZS_BF16 && gu.dim() == 2 && gu.is_contiguous() && gu.size(1) % 2 == 0,
             "gu: bf16 [R, 2 width]");
  const int64_t rows = gu.size(0), width = gu.size(1) / 2;
  ZS_REQUIRE(act.defined() && act.dtype() == ZS_BF16 && act.is_contiguous() && act.numel() >= rows * width,
             "act: bf16 [R, width]");
  ZS_REQUIRE(limit > 0, "a limit above zero");
  swiglu_kernel<<<blocks(rows * width), 256, 0, zs::stream()>>>(
      gu.data_ptr<__nv_bfloat16>(), act.data_ptr<__nv_bfloat16>(), static_cast<float>(limit), static_cast<int>(rows),
      static_cast<int>(width));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
