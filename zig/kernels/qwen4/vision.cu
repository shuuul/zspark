// The vision tower's steps between its linears (spec 019 WS-02): the learned position table resampled to an
// image's grid and added to the patches, the 2D rope on q and k in place, and the residual add with the
// LayerNorm after it. Each works on one patch row at a time from that row's own values and its place
// (`places`, int32 [P, 4]: the patch's row and column in its image, the image's height and width in patches),
// so a row's bits never depend on the rows or the images beside it.
//
// Provenance: zspark's own CUDA for transformers 5.18's `Qwen3_5VisionModel.forward` as TensorFold 0.6.5 runs
// it: `get_vision_interpolation_indices_and_weights` (bilinear, align_corners, merge-block order) and the
// weighted `pos_embed` sum; `Qwen3_5VisionRotaryEmbedding` and `apply_rotary_pos_emb_vision`; `nn.LayerNorm`
// (eps 1e-6). PyTorch's roundings are kept where they are elementwise (a bf16 result rounded once per op, no
// fused multiply-add where PyTorch has two ops); its LayerNorm's Welford sum and the order of its sums are not:
// the values are near, not to the bit.
//
// Shared memory: the LayerNorm's 5 warps' two sums, 40 B a block; nothing else.

#include <stdint.h>

#include <cuda_bf16.h>

#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

constexpr int HIDDEN = 1152;
constexpr int SIDE = 48;            // the position table's side: 2,304 = 48 x 48
constexpr int HEADS = 16;
constexpr int HEAD = 72;
constexpr int HALF = HEAD / 2;      // rotate_half's pairs: d and d + 36
constexpr int FREQS = HEAD / 4;     // 18 frequencies an axis
constexpr int NORM_THREADS = HIDDEN / 8;   // 144: one 16-byte vector of 8 values a thread

// One axis' two taps and weights, `_interpolation_axis_taps_weights` with align_corners and border clamping.
__device__ __forceinline__ void taps(int index, int size, int (&tap)[2], float (&weight)[2]) {
  const float src = __fdiv_rn(__fmul_rn(static_cast<float>(index), static_cast<float>(SIDE - 1)),
                              static_cast<float>(size - 1 > 1 ? size - 1 : 1));
  const float floor = floorf(src);
  for (int o = 0; o < 2; ++o) {
    const int raw = static_cast<int>(floor) + o;
    tap[o] = raw < 0 ? 0 : (raw > SIDE - 1 ? SIDE - 1 : raw);
    const float distance = fabsf(__fsub_rn(__fsub_rn(src, floor), static_cast<float>(o)));
    const float w = __fsub_rn(1.f, distance);
    weight[o] = w > 0.f ? w : 0.f;
  }
}

// embed = bf(embed + bias), written back, then h = bf(embed + bf(sum of the four taps' table rows times their
// weights)): one block a patch, a thread a value. `embed` is the patches' product rounded to bf16 without its
// bias: PyTorch's Conv3d adds the bias to the rounded product and rounds again (measured on the export's
// `patch_embed`: 3.6e-3 worst row with one rounding, 5.5e-4 with two). The taps' sum runs in their order (h0w0,
// h0w1, h1w0, h1w1) from zero, as PyTorch's sum(1).
__global__ void pos_kernel(__nv_bfloat16* __restrict__ embed, const __nv_bfloat16* __restrict__ bias,
                           const __nv_bfloat16* __restrict__ table, const int4* __restrict__ places,
                           __nv_bfloat16* __restrict__ out) {
  const int patch = blockIdx.x;
  const int4 place = places[patch];
  int ht[2], wt[2];
  float hw[2], ww[2];
  taps(place.x, place.z, ht, hw);
  taps(place.y, place.w, wt, ww);
  const __nv_bfloat16* rows[4] = {table + (ht[0] * SIDE + wt[0]) * HIDDEN, table + (ht[0] * SIDE + wt[1]) * HIDDEN,
                                  table + (ht[1] * SIDE + wt[0]) * HIDDEN, table + (ht[1] * SIDE + wt[1]) * HIDDEN};
  const float weights[4] = {__fmul_rn(hw[0], ww[0]), __fmul_rn(hw[0], ww[1]), __fmul_rn(hw[1], ww[0]),
                            __fmul_rn(hw[1], ww[1])};
  for (int c = threadIdx.x; c < HIDDEN; c += blockDim.x) {
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(rows[t][c]), weights[t]));
    const size_t at = static_cast<size_t>(patch) * HIDDEN + c;
    const __nv_bfloat16 e = __float2bfloat16_rn(__fadd_rn(__bfloat162float(embed[at]), __bfloat162float(bias[c])));
    embed[at] = e;
    out[at] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(e), bf(sum)));
  }
}

// q and k of a patch rotated in place in qkv [P, 3 * 1152] (q, k, v; a head's 72 values each), in fp32 with
// PyTorch's separate roundings, `(x * cos) + (rotate_half(x) * sin)`, then rounded to bf16. Angles: position times
// inv_freq[j] = 1 / 10000^(2j / 36), the row's for j of [0, 18) and the column's for [18, 36), twice. A block a
// patch: 36 pairs by 8 lanes, a lane's angle once for its 4 of the 32 heads of q and k.
constexpr int ROPE_LANES = 8;
__global__ void __launch_bounds__(HALF * ROPE_LANES) rope_kernel(__nv_bfloat16* __restrict__ qkv,
                                                                  const int4* __restrict__ places, float theta) {
  const int patch = blockIdx.x;
  const int4 place = places[patch];
  __nv_bfloat16* row = qkv + static_cast<size_t>(patch) * 3 * HIDDEN;
  const int d = threadIdx.x % HALF, lane = threadIdx.x / HALF;
  const int j = d % FREQS;
  const float inv = __fdiv_rn(1.f, powf(theta, __fdiv_rn(static_cast<float>(2 * j), static_cast<float>(2 * FREQS))));
  const float angle = __fmul_rn(static_cast<float>(d < FREQS ? place.x : place.y), inv);
  // cosf and sinf as PyTorch's cos and sin call them (sincosf is not promised the same bits)
  const float c = cosf(angle), s = sinf(angle);
#pragma unroll
  for (int i = lane; i < 2 * HEADS; i += ROPE_LANES) {
    __nv_bfloat16* x = row + i * HEAD;   // q's 16 heads, then k's
    const float x1 = __bfloat162float(x[d]), x2 = __bfloat162float(x[d + HALF]);
    x[d] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(-x2, s)));
    x[d + HALF] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s)));
  }
}

__device__ __forceinline__ float block_sum(float v, float* shared) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  constexpr int WARPS = (NORM_THREADS + 31) / 32;
  __syncthreads();
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) total += shared[w];
  return total;
}

// With `branch`: h = bf(h + branch), written back. Then, with `y`: y = LayerNorm(h) in fp32, mean and variance
// in two passes, (x - mean) * rstd * weight + bias, rounded once. One block a row, 144 threads of 8 values.
// The 144 threads are 4.5 warps: the fifth's upper half has no values and adds zeros.
__global__ void __launch_bounds__(160) norm_kernel(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ branch,
                                                   const __nv_bfloat16* __restrict__ weight,
                                                   const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y,
                                                   float eps) {
  __shared__ float shared[8];
  const size_t base = static_cast<size_t>(blockIdx.x) * HIDDEN;
  const bool live = threadIdx.x < NORM_THREADS;
  const int c = threadIdx.x * 8;
  float x[8];
  if (live) {
    uint4 raw = *reinterpret_cast<const uint4*>(h + base + c);
    const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
    if (branch != nullptr) {
      uint4 add = *reinterpret_cast<const uint4*>(branch + base + c);
      const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(&add);
      uint4 sum;
      __nv_bfloat16* s = reinterpret_cast<__nv_bfloat16*>(&sum);
#pragma unroll
      for (int i = 0; i < 8; ++i) s[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(v[i]), __bfloat162float(a[i])));
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
  const float mean = block_sum(sum, shared) / static_cast<float>(HIDDEN);
  float squares = 0.f;
  if (live) {
#pragma unroll
    for (int i = 0; i < 8; ++i) squares += (x[i] - mean) * (x[i] - mean);
  }
  const float rstd = rsqrtf(block_sum(squares, shared) / static_cast<float>(HIDDEN) + eps);
  if (!live) return;
  uint4 gw = *reinterpret_cast<const uint4*>(weight + c), gb = *reinterpret_cast<const uint4*>(bias + c);
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

bool places_of(const zs::View& v, int64_t p) {
  return v.defined() && v.dtype() == ZS_I32 && v.is_contiguous() && v.dim() == 2 && v.size(0) == p && v.size(1) == 4;
}

}  // namespace

// embed bf16 [P, 1152] += bias bf16 [1152] in place; out bf16 [P, 1152] = embed + the position table [2304, 1152]
// resampled at each patch's place, int32 [P, 4] {row, column, height, width}.
extern "C" int zs_qwen4_vision_pos(const zs_view* embed_, const zs_view* bias_, const zs_view* table_,
                                   const zs_view* places_, const zs_view* out_) {
  zs::begin();
  const zs::View embed(embed_), bias(bias_), table(table_), places(places_), out(out_);
  const int64_t p = embed.size(0);
  ZS_REQUIRE(p >= 1 && rows_of(embed, p, HIDDEN), "embed: bf16 [P, 1152]");
  ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_BF16 && bias.numel() == HIDDEN, "bias: bf16 [1152]");
  ZS_REQUIRE(rows_of(table, SIDE * SIDE, HIDDEN), "table: bf16 [2304, 1152]");
  ZS_REQUIRE(places_of(places, p), "places: int32 [P, 4]");
  ZS_REQUIRE(rows_of(out, p, HIDDEN), "out: bf16 [P, 1152]");
  ZS_REQUIRE(p <= INT32_MAX, "P within int32");
  pos_kernel<<<static_cast<unsigned>(p), 128, 0, zs::stream()>>>(
      embed.data_ptr<__nv_bfloat16>(), bias.data_ptr<__nv_bfloat16>(), table.data_ptr<__nv_bfloat16>(), places.data_ptr<int4>(),
      out.data_ptr<__nv_bfloat16>());
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// qkv bf16 [P, 3456]: q and k rotated in place by each patch's row and column (places int32 [P, 4]).
extern "C" int zs_qwen4_vision_rope(const zs_view* qkv_, const zs_view* places_, double theta) {
  zs::begin();
  const zs::View qkv(qkv_), places(places_);
  const int64_t p = qkv.size(0);
  ZS_REQUIRE(p >= 1 && rows_of(qkv, p, 3 * HIDDEN), "qkv: bf16 [P, 3456]");
  ZS_REQUIRE(places_of(places, p), "places: int32 [P, 4]");
  ZS_REQUIRE(theta > 1.0, "theta above 1");
  rope_kernel<<<static_cast<unsigned>(p), HALF * ROPE_LANES, 0, zs::stream()>>>(qkv.data_ptr<__nv_bfloat16>(),
                                                                  places.data_ptr<int4>(), static_cast<float>(theta));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// h bf16 [P, 1152] += branch (when given, rounded to bf16 and written back), then y = LayerNorm(h) by weight and
// bias bf16 [1152] (when y is given).
extern "C" int zs_qwen4_vision_norm(const zs_view* h_, const zs_view* branch_, const zs_view* weight_,
                                    const zs_view* bias_, const zs_view* y_, double eps) {
  zs::begin();
  const zs::View h(h_), branch(branch_), weight(weight_), bias(bias_), y(y_);
  const int64_t p = h.size(0);
  ZS_REQUIRE(p >= 1 && rows_of(h, p, HIDDEN), "h: bf16 [P, 1152]");
  ZS_REQUIRE(!branch.defined() || rows_of(branch, p, HIDDEN), "branch: bf16 [P, 1152]");
  ZS_REQUIRE(branch.defined() || y.defined(), "an add, a norm or both");
  if (y.defined()) {
    ZS_REQUIRE(rows_of(y, p, HIDDEN), "y: bf16 [P, 1152]");
    ZS_REQUIRE(weight.defined() && weight.dtype() == ZS_BF16 && weight.numel() == HIDDEN && bias.defined() &&
                   bias.dtype() == ZS_BF16 && bias.numel() == HIDDEN,
               "weight and bias: bf16 [1152]");
  }
  norm_kernel<<<static_cast<unsigned>(p), 160, 0, zs::stream()>>>(
      h.data_ptr<__nv_bfloat16>(), branch.defined() ? branch.data_ptr<__nv_bfloat16>() : nullptr,
      y.defined() ? weight.data_ptr<__nv_bfloat16>() : nullptr, y.defined() ? bias.data_ptr<__nv_bfloat16>() : nullptr,
      y.defined() ? y.data_ptr<__nv_bfloat16>() : nullptr, static_cast<float>(eps));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
