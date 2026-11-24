// GLM-5.3-Flash's KDA between windows: the conv's window of every layer after a commit. A copy, hand-written
// because nothing computes.
// Semantics from TensorFold 0.6.6 (cb2ebf0) families/glm5_next/cuda/forward.py (_conv_shift, _shift_conv;
// Triton), Apache-2.0; rewritten for zspark in CUDA C++.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"

namespace {

constexpr int WINDOW = 3;  // a 4-tap conv keeps three rows

// Thread (layer, channel): the layer's three window rows become rows keep .. keep + 2 of [the window; the new
// rows]. A thread reads its three values before it writes one, so the window shifts in place.
__global__ void conv_shift_kernel(__nv_bfloat16* __restrict__ conv, const __nv_bfloat16* __restrict__ proj, int keep,
                                  long long proj_layer, long long proj_row, int C, int layers) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)layers * C) return;
  const long long l = i / C, c = i % C;
  __nv_bfloat16* window = conv + l * WINDOW * C;
  __nv_bfloat16 v[WINDOW];
#pragma unroll
  for (int j = 0; j < WINDOW; ++j) {
    const int src = keep + j;
    v[j] = src < WINDOW ? window[(long long)src * C + c] : proj[l * proj_layer + (long long)(src - WINDOW) * proj_row + c];
  }
#pragma unroll
  for (int j = 0; j < WINDOW; ++j) window[(long long)j * C + c] = v[j];
}

}  // namespace

extern "C" {

// conv bf16 [layers, 3, C], in place; proj bf16, layer l's row r at l * proj_layer + r * proj_row elements, its
// first C values the conv's channels, `keep` rows of it or more.
int zs_glm5_kda_conv_shift(const zs_view* conv_, const zs_view* proj_, int64_t keep, int64_t proj_layer,
                           int64_t proj_row) {
  zs::begin();
  const zs::View conv(conv_), proj(proj_);
  ZS_REQUIRE(conv.defined() && conv.dtype() == ZS_BF16 && conv.dim() == 3 && conv.is_contiguous() &&
             conv.size(1) == WINDOW && conv.size(0) >= 1 && conv.size(2) >= 1 && conv.size(2) <= INT32_MAX &&
             conv.size(0) <= INT32_MAX, "conv: bf16 [layers, 3, C]");
  const int64_t layers = conv.size(0), c = conv.size(2);
  ZS_REQUIRE(keep >= 1 && keep <= INT32_MAX - WINDOW && proj_row >= c && proj_layer >= keep * proj_row,
             "keep rows of C channels or more a layer");
  ZS_REQUIRE(proj.defined() && proj.dtype() == ZS_BF16 && proj.is_contiguous() &&
             proj.numel() >= (layers - 1) * proj_layer + (keep - 1) * proj_row + c, "proj: bf16, keep rows a layer");
  conv_shift_kernel<<<static_cast<unsigned>((layers * c + 255) / 256), 256, 0, zs::stream()>>>(
      conv.data_ptr<__nv_bfloat16>(), proj.data_ptr<__nv_bfloat16>(), static_cast<int>(keep), proj_layer, proj_row,
      static_cast<int>(c), static_cast<int>(layers));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
