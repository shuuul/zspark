// Qwen3-ASR's audio tower steps between its GEMMs (spec 138 WS-03): the convolutions' patches (im2col), the
// convolutions' bias and GELU, and the per-chunk sinusoidal positions. The GEMMs are `bf16/gemm_bias.cu`'s, the
// LayerNorms `common/layernorm.cu`'s and the attention `tower_attention.cu`'s.
//
// The conv stack is three Conv2d of kernel 3, stride 2, padding 1 over (frequency, time), as GEMMs: a layer's
// input is channels last, [chunks, F, T, C], and its patches a row an output position, the columns (kf, kt, c),
// nine taps of C channels (`tower.zig` orders the stored [480, C, 3, 3] weight's columns the same at load). An
// output position's row is (chunk, f, t), or (chunk, t, f) for the third layer, whose rows are then `conv_out`'s
// [chunks * 13, 16 * 480] input as is (its weight's columns reordered to (f, c) at load). Each output depends on
// its chunk alone: a chunk's bits are the same whatever chunks share the call.
//
// PyTorch's cuDNN Conv2d rounds the product to bf16 and adds its bias in a second op (`at::_convolution`), then
// F.gelu reads the bf16 sum: here the GEMM writes the rounded product (a zero bias) and `bias_gelu` takes
// bf(gelu(bf(x + bias))). The positions: `SinusoidsPositionEmbedding`'s rows 0..12 rounded to bf16 (the host
// computes them, `tower.zig`), added to `conv_out`'s bf16 rows as PyTorch's bf16 add, restarting each chunk.
//
// Shared memory: none.
//
// Provenance: zspark's own CUDA for the qwen-asr 0.0.6 package's `Qwen3ASRAudioEncoder.forward` (transformers
// backend, modeling_qwen3_asr.py 669-737).

#include <stdint.h>

#include <cuda_bf16.h>

#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

// Output position r's (chunk, fo, to), rows (chunk, f, t) or with `time_major` (chunk, t, f).
__device__ __forceinline__ void place(int64_t r, int fo_n, int to_n, bool time_major, int64_t& b, int& fo, int& to) {
  const int64_t per = static_cast<int64_t>(fo_n) * to_n;
  b = r / per;
  const int within = static_cast<int>(r - b * per);
  if (time_major) {
    to = within / fo_n;
    fo = within % fo_n;
  } else {
    fo = within / to_n;
    to = within % to_n;
  }
}

// col [rows, 9 * C]: x [B, F, T, C] channels last, C a multiple of 8; a thread one 16-byte piece of one tap.
__global__ void tower_patches_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ col,
                                     int64_t items, int f_n, int t_n, int c, int fo_n, int to_n, bool time_major) {
  const int pieces = c / 8;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < items;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int piece = static_cast<int>(i % pieces);
    const int64_t rt = i / pieces;
    const int tap = static_cast<int>(rt % 9);
    const int64_t r = rt / 9;
    int64_t b;
    int fo, to;
    place(r, fo_n, to_n, time_major, b, fo, to);
    const int f = 2 * fo - 1 + tap / 3, t = 2 * to - 1 + tap % 3;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (f >= 0 && f < f_n && t >= 0 && t < t_n)
      v = *reinterpret_cast<const uint4*>(x + ((b * f_n + f) * t_n + t) * c + piece * 8);
    *reinterpret_cast<uint4*>(col + (r * 9 + tap) * c + piece * 8) = v;
  }
}

// col [rows, width]: the one-channel input x [B, F, T]; columns 0..8 the taps, the rest zero (K padded to 8s).
__global__ void tower_patches1_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ col,
                                      int64_t items, int width, int f_n, int t_n, int fo_n, int to_n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < items;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int k = static_cast<int>(i % width);
    const int64_t r = i / width;
    int64_t b;
    int fo, to;
    place(r, fo_n, to_n, false, b, fo, to);
    __nv_bfloat16 v = __float2bfloat16_rn(0.f);
    if (k < 9) {
      const int f = 2 * fo - 1 + k / 3, t = 2 * to - 1 + k % 3;
      if (f >= 0 && f < f_n && t >= 0 && t < t_n) v = x[(b * f_n + f) * t_n + t];
    }
    col[i] = v;
  }
}

// x [M, N] = bf(gelu(bf(x + bias))), exact GELU (`F.gelu`), in place; 8 values a thread.
__global__ void tower_bias_gelu_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias,
                                       int64_t vectors, int n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < vectors;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int c = static_cast<int>((i * 8) % n);
    uint4 raw = reinterpret_cast<uint4*>(x)[i];
    const uint4 braw = *reinterpret_cast<const uint4*>(bias + c);
    __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(&raw);
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&braw);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float s = bf(__bfloat162float(v[e]) + __bfloat162float(b[e]));
      v[e] = __float2bfloat16_rn(s * 0.5f * (1.f + erff(s * 0.70710678118654752440f)));
    }
    reinterpret_cast<uint4*>(x)[i] = raw;
  }
}

// x [N, W] += table [P, W] row (r mod P), rounded to bf16, in place.
__global__ void tower_positions_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ table,
                                       int64_t vectors, int per_chunk, int width) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < vectors;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t r = (i * 8) / width;
    const int c = static_cast<int>((i * 8) % width);
    uint4 raw = reinterpret_cast<uint4*>(x)[i];
    const uint4 praw = *reinterpret_cast<const uint4*>(table + (r % per_chunk) * width + c);
    __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(&raw);
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&praw);
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = __float2bfloat16_rn(__bfloat162float(v[e]) + __bfloat162float(p[e]));
    reinterpret_cast<uint4*>(x)[i] = raw;
  }
}

unsigned grid_of(int64_t items) {
  const int64_t blocks = (items + 255) / 256;
  return static_cast<unsigned>(blocks < 48 * 64 ? (blocks < 1 ? 1 : blocks) : 48 * 64);
}

}  // namespace

// col bf16 [B * Fo * To, K] = the patches of x bf16 [B, F, T, C] channels last for a k3 s2 p1 convolution; K is
// 9 * C (C a multiple of 8), or for C = 1 a multiple of 8 of at least 9, its columns past 9 zero. Rows are
// (chunk, f, t), or (chunk, t, f) when `time_major`.
extern "C" int zs_qwen3_asr_tower_patches(const zs_view* x_, const zs_view* col_, int64_t time_major) {
  zs::begin();
  const zs::View x(x_), col(col_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 4 && x.is_contiguous(), "x: bf16 [B, F, T, C]");
  const int64_t b = x.size(0), f = x.size(1), t = x.size(2), c = x.size(3);
  const int64_t fo = (f - 1) / 2 + 1, to = (t - 1) / 2 + 1;
  const int64_t rows = b * fo * to;
  ZS_REQUIRE(f >= 1 && t >= 1 && f <= 4096 && t <= 4096, "F and T within 4096");
  ZS_REQUIRE(c == 1 || (c % 8 == 0 && c <= 4096), "C one or a multiple of 8");
  ZS_REQUIRE(col.defined() && col.dtype() == ZS_BF16 && col.dim() == 2 && col.is_contiguous() && col.size(0) == rows,
             "col: bf16 [B * Fo * To, K]");
  ZS_REQUIRE(c == 1 ? (col.size(1) >= 9 && col.size(1) % 8 == 0) : col.size(1) == 9 * c, "K: 9 C, or 9 padded to 8s");
  ZS_REQUIRE(c != 1 || time_major == 0, "the one-channel layer's rows are (chunk, f, t)");
  if (c == 1) {
    const int64_t items = rows * col.size(1);
    tower_patches1_kernel<<<grid_of(items), 256, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), col.data_ptr<__nv_bfloat16>(), items, static_cast<int>(col.size(1)),
        static_cast<int>(f), static_cast<int>(t), static_cast<int>(fo), static_cast<int>(to));
  } else {
    const int64_t items = rows * 9 * (c / 8);
    tower_patches_kernel<<<grid_of(items), 256, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), col.data_ptr<__nv_bfloat16>(), items, static_cast<int>(f), static_cast<int>(t),
        static_cast<int>(c), static_cast<int>(fo), static_cast<int>(to), time_major != 0);
  }
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// x bf16 (M, N) = bf(gelu(bf(x + bias))) in place, bias bf16 [N], N a multiple of 8.
extern "C" int zs_qwen3_asr_tower_bias_gelu(const zs_view* x_, const zs_view* bias_) {
  zs::begin();
  const zs::View x(x_), bias(bias_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(1) % 8 == 0,
             "x: bf16 (M, N), N in 8s");
  ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_BF16 && bias.numel() == x.size(1), "bias: bf16 [N]");
  const int64_t vectors = x.numel() / 8;
  tower_bias_gelu_kernel<<<grid_of(vectors), 256, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), bias.data_ptr<__nv_bfloat16>(), vectors, static_cast<int>(x.size(1)));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// x bf16 [N, W] += table bf16 [P, W] at row (r mod P), rounded to bf16, in place.
extern "C" int zs_qwen3_asr_tower_positions(const zs_view* x_, const zs_view* table_) {
  zs::begin();
  const zs::View x(x_), table(table_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(1) % 8 == 0,
             "x: bf16 [N, W], W in 8s");
  ZS_REQUIRE(table.defined() && table.dtype() == ZS_BF16 && table.dim() == 2 && table.is_contiguous() &&
                 table.size(1) == x.size(1) && table.size(0) >= 1,
             "table: bf16 [P, W]");
  const int64_t vectors = x.numel() / 8;
  tower_positions_kernel<<<grid_of(vectors), 256, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), table.data_ptr<__nv_bfloat16>(), vectors, static_cast<int>(table.size(0)),
      static_cast<int>(x.size(1)));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
