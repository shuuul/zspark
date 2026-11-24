// A token's embedding row into each residual stream. Flash Next and GLM-5.3-Flash both start a forward here.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/exl3_mm.py (_embed) and 0.6.6 (cb2ebf0)
// families/glm5_next/cuda/glue.py (_embed_b16), Triton, Apache-2.0; rewritten for zspark in CUDA C++. It was
// zig/kernels/qwen4/hc.cu's until a second model launched it (spec 044).
//
// zs_embed_rows: a prompt's placeholder rows take an encoder's features (Flash Next's image, spec 019; Qwen3-ASR's
// audio, spec 138). Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/image_rows.py (embed) and
// forward.py (compute's `features`: `h.index_copy_(0, target, features.repeat(1, streams))`), PyTorch, Apache-2.0;
// written for zspark in CUDA C++. A copy: no arithmetic, so a row's bits are its feature's whatever the window.
// It was zig/kernels/qwen4/embed_image.cu's `zs_qwen4_embed_image` until a second model launched it.
// No shared memory.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"

namespace {

// Thread (row, d): the token's embedding value into each stream. An id outside the table reads nothing and its
// row is zero: TensorFold's `_embed` has no such bound (spec 081), and the ids are another kernel's or a host's.
__global__ void embed_kernel(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
                             __nv_bfloat16* __restrict__ out, int R, int D, int S, int V) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const int r = static_cast<int>(i / D), d = static_cast<int>(i % D);
  const unsigned id = static_cast<unsigned>(ids[r]);
  const __nv_bfloat16 v = id < static_cast<unsigned>(V) ? table[(size_t)id * D + d] : __float2bfloat16(0.f);
  for (int s = 0; s < S; ++s) out[((size_t)r * S + s) * D + d] = v;
}

// Block (pair, stream): row PAIRS[2 pair] of OUT takes feature row PAIRS[2 pair + 1] in stream blockIdx.y, eight
// values a thread a step (D a multiple of 8).
__global__ void embed_rows_kernel(const __nv_bfloat16* __restrict__ FEATURES, const int* __restrict__ PAIRS,
                                  __nv_bfloat16* __restrict__ OUT, int D, int S) {
  const int row = PAIRS[2 * blockIdx.x], from = PAIRS[2 * blockIdx.x + 1], s = blockIdx.y;
  const uint4* const src = reinterpret_cast<const uint4*>(FEATURES + (size_t)from * D);
  uint4* const dst = reinterpret_cast<uint4*>(OUT + ((size_t)row * S + s) * D);
  for (int i = threadIdx.x; i < D / 8; i += blockDim.x) dst[i] = src[i];
}

}  // namespace

extern "C" {

// ids int32 [R], table bf16 [V, D] -> out bf16 [R, S * D], the row in each stream.
int zs_embed(const zs_view* ids_, const zs_view* table_, const zs_view* out_, int64_t streams) {
  zs::begin();
  const zs::View ids(ids_), table(table_), out(out_);
  ZS_REQUIRE(ids.defined() && ids.dtype() == ZS_I32, "ids: int32 [R]");
  ZS_REQUIRE(table.defined() && table.dtype() == ZS_BF16 && table.dim() == 2, "table: bf16 [V, D]");
  const int64_t rows = ids.numel(), d = table.size(1);
  ZS_REQUIRE(table.size(0) >= 1 && table.size(0) <= INT32_MAX, "table: 1 to 2^31 - 1 rows");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() == rows * streams * d,
             "out: bf16 [R, S * D]");
  embed_kernel<<<static_cast<unsigned>((rows * d + 255) / 256), 256, 0, zs::stream()>>>(
      ids.data_ptr<int>(), table.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(), static_cast<int>(rows),
      static_cast<int>(d), static_cast<int>(streams), static_cast<int>(table.size(0)));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}


// features bf16 [N, D], pairs int32 [K, 2] (a row of out, then its feature's row; each row at most once) ->
// rows of out bf16 [R, S * D] in each of the S streams. The rows not listed are left as they are.
int zs_embed_rows(const zs_view* features_, const zs_view* pairs_, const zs_view* out_, int64_t streams) {
  zs::begin();
  const zs::View features(features_), pairs(pairs_), out(out_);
  ZS_REQUIRE(features.defined() && features.dtype() == ZS_BF16 && features.dim() == 2 && features.is_contiguous(),
             "features: bf16 [N, D]");
  const int64_t d = features.size(1);
  ZS_REQUIRE(d >= 8 && d % 8 == 0 && d <= INT32_MAX, "features: D a multiple of 8");
  ZS_REQUIRE(pairs.defined() && pairs.dtype() == ZS_I32 && pairs.dim() == 2 && pairs.size(1) == 2 &&
             pairs.is_contiguous() && pairs.size(0) >= 1 && pairs.size(0) <= INT32_MAX, "pairs: int32 [K, 2]");
  ZS_REQUIRE(streams >= 1 && streams <= 65535 && out.defined() && out.dtype() == ZS_BF16 && out.dim() == 2 &&
             out.is_contiguous() && out.size(1) == streams * d, "out: bf16 [R, S * D]");
  embed_rows_kernel<<<dim3(static_cast<unsigned>(pairs.size(0)), static_cast<unsigned>(streams)), 128, 0,
                      zs::stream()>>>(features.data_ptr<__nv_bfloat16>(), pairs.data_ptr<int>(),
                                      out.data_ptr<__nv_bfloat16>(), static_cast<int>(d), static_cast<int>(streams));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
