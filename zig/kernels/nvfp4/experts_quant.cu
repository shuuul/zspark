// zspark's own (spec 114 WS-03): the W4A4 routed experts' activations, quantized to NVFP4 once a call, before
// experts_a4.cu or experts_tile_a4.cu multiplies them. Each pair of the plan gets its row, from the bf16 row it
// reads (X row p / slots for gate and up; down's rows are made in gate and up's epilogue since WS-07, `swiglu_a4`
// in experts_a4.cuh, and this kernel no longer reads the SwiGLU's bf16 for them), in the quantization of
// experts_a4.cuh with its own expert's global scale: codes [P, K/2] (two e2m1 a byte, the lower input in the low
// nibble) and scales [P, K/16] (e4m3), row i the plan's i-th member, so that an item's rows are contiguous for
// the products' kernels, which read nothing else of the pair. A row's bits are its own: one thread makes one
// group of 16 inputs from that group alone.
//
// A block takes an item (its pairs, 16 to 64 by the plan) and its threads the item's (pair, group)s, a row's
// groups on neighbouring threads: 32 bytes read and 9 written a thread. Shared memory: none.
// Dense linears also use this quantization. Their normalized FP4 value times its e4m3 block scale is
// exact in bf16 (at most seven significand bits); the global scale is applied by the existing GEMM.
// This reuses its tensor-core product without an extra rounding. Shared memory: none.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"
#include "experts_a4.cuh"

namespace {

constexpr int THREADS = 256;

__global__ void __launch_bounds__(THREADS)
    nvfp4_experts_quant_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                               const float* __restrict__ gs, const int* __restrict__ items,
                               const int* __restrict__ counts, const int* __restrict__ members,
                               uint8_t* __restrict__ codes, uint8_t* __restrict__ scales, int K, int skip) {
  const int groups = K / 16;
  const int n_items = __ldg(counts);
  for (int it = blockIdx.x; it < n_items; it += gridDim.x) {
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const float g = __ldg(gs + e);
    for (int at = threadIdx.x; at < cnt * groups; at += THREADS) {
      const int r = at / groups, grp = at - r * groups;
      const int p = __ldg(members + first + r);
      const uint4* src =
          reinterpret_cast<const uint4*>(X + (size_t)(slots ? p / slots : p) * x_stride + grp * 16);
      const uint4 lo = __ldg(src), hi = __ldg(src + 1);
      const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      float v[16];
      float most = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v[2 * i] = __uint_as_float(w[i] << 16);
        v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
        most = fmaxf(most, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
      }
      float t[7];
      bool zero;
      const uint32_t sb = a4_scale(most, g, t, zero);
      uint32_t c0 = 0, c1 = 0;
      if (!zero) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          c0 |= a4_code(v[i], t) << (4 * i);
          c1 |= a4_code(v[8 + i], t) << (4 * i);
        }
      }
      const size_t row = (size_t)(first + r);
      *reinterpret_cast<uint2*>(codes + row * (K / 2) + grp * 8) = make_uint2(c0, c1);
      scales[row * groups + grp] = static_cast<uint8_t>(sb);
    }
  }
}

// One thread per 16-input block; no values from another row enter the scale.
__global__ void __launch_bounds__(THREADS)
    nvfp4_linear_quant_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
                              int groups, int k, int stride, float gs) {
  const int at = blockIdx.x * blockDim.x + threadIdx.x;
  if (at >= groups) return;
  const int row = at / (k / 16), col = at % (k / 16) * 16;
  float v[16], most = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = __bfloat162float(x[(size_t)row * stride + col + i]);
    most = fmaxf(most, fabsf(v[i]));
  }
  float t[7];
  bool zero;
  const float sf = a4_e4m3f(a4_scale(most, gs, t, zero));
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const uint32_t code = zero ? 0 : a4_code(v[i], t);
    const uint32_t mag = code & 7u;
    const float value = mag < 2 ? mag * 0.5f : __uint_as_float(((mag / 2 + 126u) << 23) | ((mag & 1u) << 22));
    const float signed_value = __uint_as_float(__float_as_uint(value) | ((code & 8u) << 28));
    out[(size_t)row * k + col + i] = __float2bfloat16_rn(signed_value * sf);
  }
}

}  // namespace

extern "C" {

// x bf16 rows (pair p reads row p / slots when slots > 0, else row p), K inputs (a multiple of 64), gs fp32 [E]:
// codes u8 [P, K/2] and scales u8 [P, K/16], row i the plan's member i; items of expert `skip` are left alone.
// max_items bounds the grid: the items are known on the device alone.
int zs_nvfp4_experts_quant(const zs_view* x_, int64_t x_stride, int64_t slots, const zs_view* gs_,
                           const zs_view* items_, const zs_view* counts_, const zs_view* members_,
                           const zs_view* codes_, const zs_view* scales_, int64_t k, int64_t skip,
                           int64_t max_items) {
  zs::begin();
  const zs::View x(x_), gs(gs_), items(items_), counts(counts_), members(members_);
  zs::View codes(codes_), scales(scales_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x_stride % 8 == 0 &&
                 reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
             "x: bf16 rows on 16-byte boundaries");
  ZS_REQUIRE(k >= 64 && k % 64 == 0 && k <= x_stride, "K: a multiple of 64 inputs within a row");
  ZS_REQUIRE(gs.defined() && gs.dtype() == ZS_F32, "gs: fp32 [E]");
  ZS_REQUIRE(items.defined() && counts.defined() && members.defined(), "plan: items, counts and members");
  ZS_REQUIRE(codes.defined() && codes.dtype() == ZS_U8 && scales.defined() && scales.dtype() == ZS_U8 &&
                 reinterpret_cast<uintptr_t>(codes.data_ptr()) % 16 == 0 &&
                 codes.numel() * 8 == scales.numel() * 64 && codes.numel() >= members.numel() * (k / 2),
             "codes u8 [P, K/2] on 16 bytes, scales u8 [P, K/16], P the plan's pairs at least");
  const int64_t grid = max_items < 4096 ? max_items : 4096;
  if (grid < 1) return zs::status();
  nvfp4_experts_quant_kernel<<<static_cast<int>(grid), THREADS, 0, zs::stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(x_stride), static_cast<int>(slots),
      gs.data_ptr<float>(), items.data_ptr<int>(), counts.data_ptr<int>(), members.data_ptr<int>(),
      reinterpret_cast<uint8_t*>(codes.data_ptr()), reinterpret_cast<uint8_t*>(scales.data_ptr()),
      static_cast<int>(k), static_cast<int>(skip));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

int zs_nvfp4_linear_quant(const zs_view* x_, const zs_view* out_, double gs) {
  zs::begin();
  const zs::View x(x_);
  zs::View out(out_);
  ZS_REQUIRE(x.defined() && out.defined() && x.dtype() == ZS_BF16 && out.dtype() == ZS_BF16 &&
                 x.dim() == 2 && out.dim() == 2 && x.size(0) == out.size(0) && x.size(1) == out.size(1) &&
                 x.size(1) % 16 == 0 && x.stride(1) == 1 && out.is_contiguous(),
             "x/out: bf16 [M,K], K a multiple of 16, contiguous inputs and outputs within each row");
  ZS_REQUIRE(gs > 0 && isfinite(gs), "gs: positive finite reciprocal input scale");
  const int groups = static_cast<int>(x.numel() / 16);
  if (groups == 0) return zs::status();
  nvfp4_linear_quant_kernel<<<(groups + THREADS - 1) / THREADS, THREADS, 0, zs::stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      groups, static_cast<int>(x.size(1)), static_cast<int>(x.stride(0)), static_cast<float>(gs));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
