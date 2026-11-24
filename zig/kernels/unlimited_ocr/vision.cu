// Unlimited-OCR's vision tower between its linears (spec 137 WS-03): the patches' im2col, SAM's position add, the
// LayerNorms of SAM (bf16 residual, eps 1e-6), of its neck (LayerNorm2d) and of CLIP (fp32 residual, eps 1e-5)
// with the residual add before them and the windowed layout around them, CLIP's embedding, the concat of CLIP's
// and SAM's features, and the placing of the projector's rows among `image_newline` and `view_seperator`. Every
// kernel works on one row (a token, a window row, an output row) from that row's own values, so a row's bits
// never depend on the rows or the views beside it (the row rule).
//
// The roundings are the checkpoint's own code under `infer`'s `torch.autocast(bf16)`: a LayerNorm runs in fp32
// and its fp32 output is cast to bf16 by the linear after it (one rounding); SAM's residual stream is bf16 (`x =
// shortcut + x`, both bf16), CLIP's fp32 (`pre_layrnorm`'s fp32 output plus bf16 branches); `LayerNorm2d` is
// elementwise PyTorch on bf16 (`u` the bf16 mean, `x - u` rounded, `pow` and the second mean fp32 by autocast).
// PyTorch's Welford LayerNorm and the order of its sums are not kept: near, not to the bit.
//
// Shared memory: a LayerNorm's warps' sums, 8 floats a block (32 B); nothing else.
//
// Provenance: zspark's own CUDA for the checkpoint's `deepencoder.py` (`ImageEncoderViT`, `Block`,
// `window_partition`, `LayerNorm2d`, `CLIPVisionEmbeddings`, `VitModel`, `NoTPTransformerBlock`) and
// `modeling_unlimitedocr.py`'s `UnlimitedOCRModel.forward` (the concat and the newline/separator layout).

#include <stdint.h>

#include <cuda_bf16.h>

#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

// cols bf16 [B * G * G, 768] from pixels bf16 [B, 3, S, S], G = S / 16: column c * 256 + ky * 16 + kx, the patch
// conv's weight [768, 3, 16, 16] flattened. A thread copies one row of 16 pixels (32 B).
__global__ void ocr_patches_kernel(const __nv_bfloat16* __restrict__ pixels, __nv_bfloat16* __restrict__ cols, int side,
                                   int64_t pieces) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pieces) return;
  const int g = side / 16;
  const int ky = static_cast<int>(i % 16), c = static_cast<int>(i / 16 % 3);
  const int64_t t = i / 48;
  const int gx = static_cast<int>(t % g), gy = static_cast<int>(t / g % g);
  const int64_t b = t / (static_cast<int64_t>(g) * g);
  const uint4* src = reinterpret_cast<const uint4*>(pixels + ((b * 3 + c) * side + gy * 16 + ky) * side + gx * 16);
  uint4* dst = reinterpret_cast<uint4*>(cols + t * 768 + c * 256 + ky * 16);
  dst[0] = src[0];
  dst[1] = src[1];
}

// cols bf16 [B * Ho * Wo, 9 * C] from x bf16 [B * H * W, C] (channels last), a 3 x 3 conv of padding 1 and
// `stride`: column (ky * 3 + kx) * C + c, the weight permuted at load to [out, ky, kx, C]. A thread moves 8
// channels.
__global__ void ocr_im2col_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ cols, int h, int w,
                                  int c, int stride, int ho, int wo, int64_t pieces) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pieces) return;
  const int chunks = c / 8;
  const int part = static_cast<int>(i % chunks);
  const int tap = static_cast<int>(i / chunks % 9);
  const int64_t t = i / (9 * chunks);
  const int ox = static_cast<int>(t % wo), oy = static_cast<int>(t / wo % ho);
  const int64_t b = t / (static_cast<int64_t>(ho) * wo);
  const int iy = oy * stride - 1 + tap / 3, ix = ox * stride - 1 + tap % 3;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (iy >= 0 && iy < h && ix >= 0 && ix < w)
    v = *reinterpret_cast<const uint4*>(x + ((b * h + iy) * w + ix) * c + part * 8);
  *reinterpret_cast<uint4*>(cols + t * 9 * c + tap * c + part * 8) = v;
}

// h = bf(bf(h + bias) + pos[t % gg]) in place: the patch conv's rounded product, its bias (the conv's rounding),
// then the resampled position table (`x + get_abs_pos_sam(...)`, both bf16). A thread 8 values.
__global__ void ocr_sam_pos_kernel(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ bias,
                                   const __nv_bfloat16* __restrict__ pos, int gg, int64_t pieces) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pieces) return;
  constexpr int D = 768, CH = D / 8;
  const int part = static_cast<int>(i % CH);
  const int64_t t = i / CH;
  __nv_bfloat16* row = h + t * D + part * 8;
  const __nv_bfloat16* b = bias + part * 8;
  const __nv_bfloat16* p = pos + static_cast<int64_t>(t % gg) * D + part * 8;
#pragma unroll
  for (int e = 0; e < 8; ++e)
    row[e] = __float2bfloat16_rn(bf(__bfloat162float(row[e]) + __bfloat162float(b[e])) + __bfloat162float(p[e]));
}

// A block's sum over its threads in a fixed order: the warps' xor sums, then the warps' in order.
template <int THREADS> __device__ __forceinline__ float block_sum(float x, float* sums) {
  x = warp_sum(x);
  const int warp = threadIdx.x / 32;
  __syncthreads();
  if (threadIdx.x % 32 == 0) sums[warp] = x;
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int w = 0; w < THREADS / 32; ++w) total += sums[w];
  return total;
}

template <class H> __device__ __forceinline__ float load_h(const H* p);
template <> __device__ __forceinline__ float load_h<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float load_h<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <class H> __device__ __forceinline__ H store_h(float v);
template <> __device__ __forceinline__ float store_h<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 store_h<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// One row of y = LayerNorm(h[t]) after h[t] += branch[t]: a block a row, a thread four values. `quirk`: the neck's
// LayerNorm2d on bf16 (`u` rounded to bf16, `x - u` rounded, the variance's mean fp32, a division).
template <int D, class H>
__global__ void __launch_bounds__(D / 4)
    ocr_norm_kernel(H* __restrict__ h, const __nv_bfloat16* __restrict__ branch,
                    const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ bias,
                    __nv_bfloat16* __restrict__ y, float eps, int quirk) {
  constexpr int THREADS = D / 4;
  __shared__ float sums[THREADS / 32];
  const int64_t r = blockIdx.x;
  const int c0 = threadIdx.x * 4;
  H* row = h + r * D + c0;
  float v[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) v[e] = load_h(row + e);
  if (branch != nullptr) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const H s = store_h<H>(v[e] + __bfloat162float(branch[r * D + c0 + e]));
      row[e] = s;
      v[e] = load_h(&s);
    }
  }
  if (y == nullptr) return;
  float sum = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) sum += v[e];
  float mean = block_sum<THREADS>(sum, sums) / D;
  if (quirk) mean = bf(mean);
  float d[4], sq = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    d[e] = quirk ? bf(v[e] - mean) : v[e] - mean;
    sq += d[e] * d[e];
  }
  const float var = block_sum<THREADS>(sq, sums) / D;
  __nv_bfloat16* out = y + r * D + c0;
  if (quirk) {
    const float root = __fsqrt_rn(var + eps);
#pragma unroll
    for (int e = 0; e < 4; ++e)
      out[e] = __float2bfloat16_rn(__bfloat162float(weight[c0 + e]) * __fdiv_rn(d[e], root) + __bfloat162float(bias[c0 + e]));
  } else {
    const float rstd = rsqrtf(var + eps);
#pragma unroll
    for (int e = 0; e < 4; ++e)
      out[e] = __float2bfloat16_rn(d[e] * rstd * __bfloat162float(weight[c0 + e]) + __bfloat162float(bias[c0 + e]));
  }
}

// CLIP's embedding and pre_layrnorm: row r of view v = r / (1 + gg), place i = r % (1 + gg); e = the class
// embedding (i = 0) or SAM's feature i - 1 (bf16, `patch_embeds`); x = bf(e + pos[i]); h[r] = LayerNorm(x),
// fp32 (the residual stream, eps 1e-5). A block a row, 256 threads of four values.
__global__ void __launch_bounds__(256)
    ocr_clip_embed_kernel(const __nv_bfloat16* __restrict__ sam, const __nv_bfloat16* __restrict__ cls,
                          const __nv_bfloat16* __restrict__ pos, const __nv_bfloat16* __restrict__ weight,
                          const __nv_bfloat16* __restrict__ bias, float* __restrict__ h, int gg, float eps) {
  constexpr int D = 1024, THREADS = 256;
  __shared__ float sums[THREADS / 32];
  const int64_t r = blockIdx.x;
  const int64_t v = r / (1 + gg);
  const int i = static_cast<int>(r % (1 + gg));
  const int c0 = threadIdx.x * 4;
  const __nv_bfloat16* e = i == 0 ? cls : sam + (v * gg + i - 1) * D;
  float x[4], sum = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    x[k] = bf(__bfloat162float(e[c0 + k]) + __bfloat162float(pos[static_cast<int64_t>(i) * D + c0 + k]));
    sum += x[k];
  }
  const float mean = block_sum<THREADS>(sum, sums) / D;
  float sq = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    x[k] -= mean;
    sq += x[k] * x[k];
  }
  const float rstd = rsqrtf(block_sum<THREADS>(sq, sums) / D + eps);
#pragma unroll
  for (int k = 0; k < 4; ++k)
    h[r * D + c0 + k] = x[k] * rstd * __bfloat162float(weight[c0 + k]) + __bfloat162float(bias[c0 + k]);
}

// x bf16 [V * gg, 2048] = cat(CLIP's fp32 rows past the class token, SAM's bf16 features), cast to bf16 as the
// projector's autocast reads it. A thread 4 values.
__global__ void ocr_concat_kernel(const float* __restrict__ clip, const __nv_bfloat16* __restrict__ sam,
                                  __nv_bfloat16* __restrict__ x, int gg, int64_t pieces) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pieces) return;
  const int part = static_cast<int>(i % 512);
  const int64_t r = i / 512;
  const int64_t v = r / gg, t = r % gg;
  __nv_bfloat16* dst = x + r * 2048 + part * 4;
  if (part < 256) {
    const float* src = clip + (v * (1 + gg) + 1 + t) * 1024 + part * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[e] = __float2bfloat16_rn(src[e]);
  } else {
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(sam + r * 1024 + (part - 256) * 4);
  }
}

// out[to[i]] = src[from[i]], or `image_newline` (from -1) or `view_seperator` (from -2): rows of 1280.
__global__ void ocr_place_kernel(const __nv_bfloat16* __restrict__ src, const int* __restrict__ from,
                                 const int* __restrict__ to, const __nv_bfloat16* __restrict__ newline,
                                 const __nv_bfloat16* __restrict__ separator, __nv_bfloat16* __restrict__ out) {
  const int64_t i = blockIdx.x;
  const int f = from[i];
  const __nv_bfloat16* row = f >= 0 ? src + static_cast<int64_t>(f) * 1280 : (f == -1 ? newline : separator);
  __nv_bfloat16* dst = out + static_cast<int64_t>(to[i]) * 1280;
  for (int c = threadIdx.x; c < 1280 / 8; c += blockDim.x)
    reinterpret_cast<uint4*>(dst)[c] = reinterpret_cast<const uint4*>(row)[c];
}

unsigned blocks(int64_t n, int threads) { return static_cast<unsigned>((n + threads - 1) / threads); }

bool aligned(const zs::View& v) { return reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0; }

}  // namespace

// cols bf16 [B * G * G, 768] from pixels bf16 [B, 3, S, S], S a multiple of 16.
extern "C" int zs_unlimited_ocr_vision_patches(const zs_view* pixels_, const zs_view* cols_) {
  zs::begin();
  const zs::View pixels(pixels_), cols(cols_);
  ZS_REQUIRE(pixels.defined() && pixels.dtype() == ZS_BF16 && pixels.dim() == 4 && pixels.size(1) == 3 &&
                 pixels.size(2) == pixels.size(3) && pixels.size(2) % 16 == 0 && aligned(pixels),
             "pixels: bf16 [B, 3, S, S], S a multiple of 16");
  const int64_t side = pixels.size(2), g = side / 16, t = pixels.size(0) * g * g;
  ZS_REQUIRE(cols.defined() && cols.dtype() == ZS_BF16 && cols.is_contiguous() && cols.numel() == t * 768 && aligned(cols),
             "cols: bf16 [B * G * G, 768]");
  const int64_t pieces = t * 48;
  ocr_patches_kernel<<<blocks(pieces, 256), 256, 0, zs::stream()>>>(pixels.data_ptr<__nv_bfloat16>(),
                                                                     cols.data_ptr<__nv_bfloat16>(), static_cast<int>(side), pieces);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// cols bf16 [B * Ho * Wo, 9 * C] from x bf16 [B * H * W, C], a 3 x 3 window of padding 1 and `stride` 1 or 2.
extern "C" int zs_unlimited_ocr_vision_im2col(const zs_view* x_, const zs_view* cols_, int64_t batch, int64_t h, int64_t w,
                                              int64_t stride) {
  zs::begin();
  const zs::View x(x_), cols(cols_);
  ZS_REQUIRE(stride == 1 || stride == 2, "stride: 1 or 2");
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(0) == batch * h * w &&
                 x.size(1) % 8 == 0 && aligned(x),
             "x: bf16 [B * H * W, C], C a multiple of 8");
  const int64_t c = x.size(1), ho = (h - 1) / stride + 1, wo = (w - 1) / stride + 1;
  ZS_REQUIRE(cols.defined() && cols.dtype() == ZS_BF16 && cols.numel() >= batch * ho * wo * 9 * c && aligned(cols),
             "cols: bf16 [B * Ho * Wo, 9 * C]");
  const int64_t pieces = batch * ho * wo * 9 * (c / 8);
  ocr_im2col_kernel<<<blocks(pieces, 256), 256, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), cols.data_ptr<__nv_bfloat16>(), static_cast<int>(h), static_cast<int>(w),
      static_cast<int>(c), static_cast<int>(stride), static_cast<int>(ho), static_cast<int>(wo), pieces);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// h bf16 [N, 768] = bf(bf(h + bias) + pos[t % gg]), pos bf16 [gg, 768].
extern "C" int zs_unlimited_ocr_vision_sam_pos(const zs_view* h_, const zs_view* bias_, const zs_view* pos_) {
  zs::begin();
  const zs::View h(h_), bias(bias_), pos(pos_);
  ZS_REQUIRE(h.defined() && h.dtype() == ZS_BF16 && h.dim() == 2 && h.is_contiguous() && h.size(1) == 768, "h: bf16 [N, 768]");
  ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_BF16 && bias.numel() == 768, "bias: bf16 [768]");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_BF16 && pos.dim() == 2 && pos.size(1) == 768 && pos.size(0) >= 1 &&
                 h.size(0) % pos.size(0) == 0,
             "pos: bf16 [gg, 768], N a multiple of gg");
  const int64_t pieces = h.size(0) * 96;
  ocr_sam_pos_kernel<<<blocks(pieces, 256), 256, 0, zs::stream()>>>(h.data_ptr<__nv_bfloat16>(), bias.data_ptr<__nv_bfloat16>(),
                                                                     pos.data_ptr<__nv_bfloat16>(), static_cast<int>(pos.size(0)), pieces);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// h [N, D] (bf16, or fp32 for D 1024) += branch[from[t]] when `branch`, then y bf16 [R, D] = LayerNorm(h[to[r]])
// (a zero row where to[r] is -1) when `y`. D 256, 768 or 1024. `quirk` 1: the neck's LayerNorm2d (bf16 h, no
// branch).
extern "C" int zs_unlimited_ocr_vision_norm(const zs_view* h_, const zs_view* branch_, const zs_view* weight_,
                                            const zs_view* bias_, const zs_view* y_, double eps, int64_t quirk) {
  zs::begin();
  const zs::View h(h_), branch(branch_), weight(weight_), bias(bias_), y(y_);
  ZS_REQUIRE(h.defined() && h.dim() == 2 && h.is_contiguous() && (h.dtype() == ZS_BF16 || h.dtype() == ZS_F32),
             "h: [N, D] bf16 or fp32");
  const int64_t n = h.size(0), d = h.size(1);
  ZS_REQUIRE(d == 256 || d == 768 || d == 1024, "D: 256, 768 or 1024");
  ZS_REQUIRE(h.dtype() == ZS_BF16 || d == 1024, "an fp32 h is CLIP's, D 1024");
  ZS_REQUIRE(!branch.defined() || (branch.dtype() == ZS_BF16 && branch.dim() == 2 && branch.size(1) == d), "branch: bf16 [M, D]");
  ZS_REQUIRE(!branch.defined() || branch.size(0) >= n, "branch: a row a token");
  ZS_REQUIRE(y.defined() || branch.defined(), "a norm, an add or both");
  ZS_REQUIRE(!y.defined() || (y.dtype() == ZS_BF16 && y.dim() == 2 && y.is_contiguous() && y.size(1) == d &&
                              weight.defined() && weight.numel() == d && bias.defined() && bias.numel() == d),
             "y: bf16 [R, D] with weight and bias bf16 [D]");
  ZS_REQUIRE(!y.defined() || y.size(0) == n, "y: a row a token");
  ZS_REQUIRE(quirk == 0 || (h.dtype() == ZS_BF16 && !branch.defined()), "quirk: bf16 h and no branch");
  if (n == 0) return zs::status();
  const auto* b = branch.data_ptr<__nv_bfloat16>();
  const auto* w = weight.data_ptr<__nv_bfloat16>();
  const auto* bi = bias.data_ptr<__nv_bfloat16>();
  auto* yo = y.data_ptr<__nv_bfloat16>();
  const float e = static_cast<float>(eps);
  const int q = static_cast<int>(quirk);
  const dim3 grid(static_cast<unsigned>(n));
  if (h.dtype() == ZS_F32) ocr_norm_kernel<1024, float><<<grid, 256, 0, zs::stream()>>>(h.data_ptr<float>(), b, w, bi, yo, e, q);
  else if (d == 1024) ocr_norm_kernel<1024, __nv_bfloat16><<<grid, 256, 0, zs::stream()>>>(h.data_ptr<__nv_bfloat16>(), b, w, bi, yo, e, q);
  else if (d == 768) ocr_norm_kernel<768, __nv_bfloat16><<<grid, 192, 0, zs::stream()>>>(h.data_ptr<__nv_bfloat16>(), b, w, bi, yo, e, q);
  else ocr_norm_kernel<256, __nv_bfloat16><<<grid, 64, 0, zs::stream()>>>(h.data_ptr<__nv_bfloat16>(), b, w, bi, yo, e, q);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// h fp32 [V * (1 + gg), 1024] = pre_layrnorm(bf(cat(cls, sam[v]) + pos)), sam bf16 [V * gg, 1024], pos bf16
// [1 + gg, 1024].
extern "C" int zs_unlimited_ocr_vision_clip_embed(const zs_view* sam_, const zs_view* cls_, const zs_view* pos_, const zs_view* weight_,
                                                  const zs_view* bias_, const zs_view* h_, double eps) {
  zs::begin();
  const zs::View sam(sam_), cls(cls_), pos(pos_), weight(weight_), bias(bias_), h(h_);
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_BF16 && pos.dim() == 2 && pos.size(1) == 1024 && pos.size(0) >= 2, "pos: bf16 [1 + gg, 1024]");
  const int64_t gg = pos.size(0) - 1;
  ZS_REQUIRE(sam.defined() && sam.dtype() == ZS_BF16 && sam.dim() == 2 && sam.size(1) == 1024 && sam.size(0) % gg == 0,
             "sam: bf16 [V * gg, 1024]");
  const int64_t v = sam.size(0) / gg;
  ZS_REQUIRE(cls.defined() && cls.dtype() == ZS_BF16 && cls.numel() == 1024, "cls: bf16 [1024]");
  ZS_REQUIRE(weight.defined() && weight.dtype() == ZS_BF16 && weight.numel() == 1024 && bias.defined() &&
                 bias.dtype() == ZS_BF16 && bias.numel() == 1024,
             "weight, bias: bf16 [1024]");
  ZS_REQUIRE(h.defined() && h.dtype() == ZS_F32 && h.numel() == v * (1 + gg) * 1024, "h: fp32 [V * (1 + gg), 1024]");
  ocr_clip_embed_kernel<<<static_cast<unsigned>(v * (1 + gg)), 256, 0, zs::stream()>>>(
      sam.data_ptr<__nv_bfloat16>(), cls.data_ptr<__nv_bfloat16>(), pos.data_ptr<__nv_bfloat16>(), weight.data_ptr<__nv_bfloat16>(),
      bias.data_ptr<__nv_bfloat16>(), h.data_ptr<float>(), static_cast<int>(gg), static_cast<float>(eps));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// x bf16 [V * gg, 2048] = cat(bf(clip fp32 [V * (1 + gg), 1024] past each view's class row), sam bf16 [V * gg, 1024]).
extern "C" int zs_unlimited_ocr_vision_concat(const zs_view* clip_, const zs_view* sam_, const zs_view* x_) {
  zs::begin();
  const zs::View clip(clip_), sam(sam_), x(x_);
  ZS_REQUIRE(sam.defined() && sam.dtype() == ZS_BF16 && sam.dim() == 2 && sam.size(1) == 1024, "sam: bf16 [V * gg, 1024]");
  ZS_REQUIRE(clip.defined() && clip.dtype() == ZS_F32 && clip.dim() == 2 && clip.size(1) == 1024 && clip.size(0) > sam.size(0),
             "clip: fp32 [V * (1 + gg), 1024]");
  const int64_t views = clip.size(0) - sam.size(0);
  ZS_REQUIRE(sam.size(0) % views == 0, "clip and sam: the same views");
  const int64_t gg = sam.size(0) / views;
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.numel() == sam.size(0) * 2048, "x: bf16 [V * gg, 2048]");
  const int64_t pieces = sam.size(0) * 512;
  ocr_concat_kernel<<<blocks(pieces, 256), 256, 0, zs::stream()>>>(clip.data_ptr<float>(), sam.data_ptr<__nv_bfloat16>(),
                                                                   x.data_ptr<__nv_bfloat16>(), static_cast<int>(gg), pieces);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// out bf16 [R, 1280]: out[to[i]] = src[from[i]] (from >= 0), `newline` (-1) or `separator` (-2), i over from's n.
extern "C" int zs_unlimited_ocr_vision_place(const zs_view* src_, const zs_view* from_, const zs_view* to_, const zs_view* newline_,
                                             const zs_view* separator_, const zs_view* out_) {
  zs::begin();
  const zs::View src(src_), from(from_), to(to_), newline(newline_), separator(separator_), out(out_);
  ZS_REQUIRE(from.defined() && from.dtype() == ZS_I32 && to.defined() && to.dtype() == ZS_I32 && to.numel() == from.numel(),
             "from, to: int32 [n]");
  ZS_REQUIRE(!src.defined() || (src.dtype() == ZS_BF16 && src.dim() == 2 && src.size(1) == 1280 && aligned(src)), "src: bf16 [M, 1280]");
  ZS_REQUIRE(newline.defined() && newline.dtype() == ZS_BF16 && newline.numel() == 1280 && aligned(newline) && separator.defined() &&
                 separator.dtype() == ZS_BF16 && separator.numel() == 1280 && aligned(separator),
             "newline, separator: bf16 [1280]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.dim() == 2 && out.size(1) == 1280 && aligned(out), "out: bf16 [R, 1280]");
  if (from.numel() == 0) return zs::status();
  ocr_place_kernel<<<static_cast<unsigned>(from.numel()), 160, 0, zs::stream()>>>(
      src.data_ptr<__nv_bfloat16>(), from.data_ptr<int>(), to.data_ptr<int>(), newline.data_ptr<__nv_bfloat16>(),
      separator.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>());
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
