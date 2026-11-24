// zspark's own (spec 114 WS-03): what the W4A4 routed experts share (experts_quant.cu, experts_a4.cu,
// experts_tile_a4.cu). NVFP4 weights times NVFP4 activations on the block-scaled FP4 tensor-core instruction, as the
// card's vLLM (FP4-Local's recommended launch, spec 115) runs a prompt's rows: b12x's W4A16 experts with
// `B12X_W4A16_A4_PREFILL`, A4 on a prompt's rows (`vllm/utils/b12x.py` 301-357); its steps' and verify windows' rows
// A16, as experts_a4.cu's `nvfp4_expert_a16_kernel` runs them on this pack.
//
// An activation's 16-input group is quantized as b12x's `quantize_block_fp4` (`_lib/intrinsics.py`), in its order
// of operations: gs is the activation global scale, 1 / input_scale (vLLM keeps the reciprocal); gate and up's is
// the layer's, 1 / the largest gate or up input_scale of its experts (spec 115: the card's A4 prompt path takes
// `a1_gscale = a1.amin()`, `fused_moe/b12x.py` 520-534; FP4-Local's differ by expert), down's each expert's; the
// group's scale is e4m3(min(max|x| * gs / 6, 448)), rounded to nearest even; each value is the e2m1 nearest
// x / (scale / gs) by `quantize_and_pack_16`'s thresholds (ties to even); a scale of zero leaves the group zeros. The
// epilogue's factor (alpha) is weight_scale_2 / gs, computed on the host in fp32 as b12x's
// `_prepare_modelopt_nvfp4_runtime_alphas` does. `numeric.nvfp4Group` is the host's copy, to the bit.
//
// The instruction: `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32
// .ue4m3`, run and held to an f64 sum on the GB10 (cuda-gb10, 2026-10-19). Lane (q, c) = (l / 4, l % 4): A register
// r holds row q + 8 (r % 2), inputs 32 (r / 2) + 8 c .. + 7, the lower input in the low nibble: a row's packed
// bytes as they are; B register j column q, inputs 32 j + 8 c .. + 7; the SFA word of lane c in {0, 1} row q + 8 c's
// four scales, byte i the group of inputs 16 i .. 16 i + 15; the SFB word column q's four. The sums as m16n8k16's.
//
// The row rule: a pair's quantized row is made from its own bf16 row and its layer's or expert's gs, and its sums are
// an mma row over its own operands, k64 steps in order from zero, whatever kernel or tile holds it.
//
// Down's rows (spec 114 WS-07) are made in gate and up's epilogue (`swiglu_a4`), as the card's A4 prompt path makes
// them in its FC1 epilogue (spec 115; b12x `moe/_shared/kernels/w4a16/prefill_a4.py` `_stage_epilogue`, `_activate`,
// `_store_fc1`): gate's and up's fp32 sums times alpha each rounded to bf16, clipped, sigma = rcp.approx.ftz(1 +
// exp(-gate)) with exp as ex2.approx.ftz of gate * log2(e) (its `fast_math` exp), the product bf16(bf16(gate *
// sigma) * up), then quantized as above with down's gs (`quantize_block_fp4`, its `fast_quant` off). zspark's bf16
// path rounds gate, up, silu and the product (`experts.cuh` `swiglu`); the A16 form on this pack (experts_a4.cu
// `nvfp4_expert_a16_kernel`, a step's and a verify window's) rounds once, `swiglu_b12x`. A group of 16 is 16
// columns of one row: two n8 tiles on one lane quad, its largest magnitude met by two shuffles (a max: no order),
// each lane coding its four values; nothing of another row enters.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/experts.cuh"

// Where gate and up's epilogue puts down's rows: codes u8 [P, width/2], scales u8 [P, width/16], row i the plan's
// member i, and each expert's gs for down (1 / input_scale). Outside the anonymous namespace: experts_a4.cu hands
// it to experts_tile_a4.cu's launcher.
struct DownA4 {
  uint8_t* codes;
  uint8_t* scales;
  const float* gs;
};

namespace {

// uint4 a (32 columns, 64 inputs) block of a matrix: lane l's B registers of n8 tiles 0 and 1 at l, of tiles 2 and
// 3 at 32 + l (b0, b1 a tile), then at 64 + q the scale words of columns q, 8 + q, 16 + q, 24 + q.
constexpr int A4_BLOCK4 = 72;

__device__ __forceinline__ void mma_a4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                       uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, {%11}, {%12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "h"((unsigned short)0),
        "h"((unsigned short)0));
}

__device__ __forceinline__ uint32_t a4_comp(const uint4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

// The e4m3 byte nearest x (ties to even), saturating at 448.
__device__ __forceinline__ uint32_t a4_e4m3(float x) {
  unsigned short pair;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(pair) : "f"(0.f), "f"(x));
  return pair & 0xFFu;
}

__device__ __forceinline__ float a4_e4m3f(uint32_t b) {
  // exponent field 0 is subnormal: (b & 7) * 2^-9; else (8 + m) * 2^(e - 10)
  const uint32_t e = (b >> 3) & 15u, m = b & 7u;
  return e == 0 ? __uint2float_rn(m) * 0.001953125f : __uint_as_float(((e + 120u) << 23) | (m << 20));
}

// The e2m1 code of |v| against `quantize_and_pack_16`'s thresholds t (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5 times the
// value scale), the sign bit for a negative value that is not zero. Its chain of ranges ((t0, t1) -> 1, [t1, t2] ->
// 2, ...) as the count of the bounds passed, the thresholds rising: the same code without a branch (a NaN 0).
__device__ __forceinline__ uint32_t a4_code(float v, const float (&t)[7]) {
  const float a = fabsf(v);
  const uint32_t n = (uint32_t)(a > t[0]) + (uint32_t)(a >= t[1]) + (uint32_t)(a > t[2]) + (uint32_t)(a >= t[3]) +
                     (uint32_t)(a > t[4]) + (uint32_t)(a >= t[5]) + (uint32_t)(a > t[6]);
  return (n != 0 && v < 0.f) ? (n | 8u) : n;
}

// The group's scale byte from its largest magnitude, and the thresholds its values are coded against.
__device__ __forceinline__ uint32_t a4_scale(float most, float gs, float (&t)[7], bool& zero) {
  const float raw = fminf(__fdiv_rn(__fmul_rn(most, gs), 6.f), 448.f);
  const uint32_t sb = a4_e4m3(raw);
  const float sf = a4_e4m3f(sb);
  zero = sf == 0.f || gs == 0.f;
  const float vs = zero ? 0.f : __fdiv_rn(sf, gs);
  const float k[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
#pragma unroll
  for (int i = 0; i < 7; ++i) t[i] = __fmul_rn(vs, k[i]);
  return sb;
}

// The gated activation on fp32 sums already times alpha, then one bf16 rounding: the A16 form's (b12x W4A16's
// decode SwiGLU, fp32 then cast; zspark's exp and reciprocal IEEE).
__device__ __forceinline__ float swiglu_b12x(float g, float u, float limit) {
  if (limit > 0.f) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
  }
  const float s = __frcp_rn(1.f + expf(-g));
  return __bfloat162float(__float2bfloat16_rn(g * s * u));
}

__device__ __forceinline__ float a4_bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// The card's A4 FC1 activation (`prefill_a4.py` `_activate` on `_stage_epilogue`'s bf16 tile): gate and up rounded
// to bf16, clipped, its fast sigmoid, bf16(bf16(gate * sigma) * up).
template <bool RELU2>
__device__ __forceinline__ float activation_card(float g, float u, float limit) {
  if constexpr (RELU2) {
    const float r = fmaxf(a4_bf(g), 0.f);
    return a4_bf(r * r);
  }
  g = a4_bf(g);
  u = a4_bf(u);
  if (limit > 0.f) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
  }
  float e, s;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(e) : "f"(-g * 1.44269504f));
  asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(s) : "f"(1.f + e));
  return a4_bf(a4_bf(g * s) * u);
}

// One m16 tile of gate's and up's sums (times alpha), a warp's column block cb (32 columns: groups 2 cb and
// 2 cb + 1), into down's rows at plan rows at0 (row q) and at1 (q + 8). Lane (q, c) holds columns 8 j + 2c, + 1 of
// tile j: of group G (tiles 2G, 2G + 1) inputs 2c, 2c + 1 (byte c) and 8 + 2c, 9 + 2c (byte 4 + c). Lane c writes
// group c % 2 of row q + 8 (c / 2). Every lane of the warp calls it (the shuffles); v0, v1 say which rows are real.
// act, when not null: the bf16 values quantized, at pair rows pr0, pr1 [., width] (a check's, not a call's).
template <bool RELU2>
__device__ __forceinline__ void activation_a4(const float (&g)[NTW][4], const float (&u)[NTW][4], int cb, int c,
                                              const DownA4& dq, float gs, int width, size_t at0, size_t at1, bool v0,
                                              bool v1, float limit, __nv_bfloat16* act, int pr0, int pr1) {
  uint2 word[2][2];
  uint32_t sbyte[2][2];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int G = 0; G < 2; ++G) {
      const float x0 = activation_card<RELU2>(g[2 * G][2 * h], u[2 * G][2 * h], limit);
      const float x1 = activation_card<RELU2>(g[2 * G][2 * h + 1], u[2 * G][2 * h + 1], limit);
      const float x2 = activation_card<RELU2>(g[2 * G + 1][2 * h], u[2 * G + 1][2 * h], limit);
      const float x3 = activation_card<RELU2>(g[2 * G + 1][2 * h + 1], u[2 * G + 1][2 * h + 1], limit);
      float most = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
      most = fmaxf(most, __shfl_xor_sync(0xFFFFFFFFu, most, 1));
      most = fmaxf(most, __shfl_xor_sync(0xFFFFFFFFu, most, 2));
      float t[7];
      bool zero;
      sbyte[h][G] = a4_scale(most, gs, t, zero);
      uint32_t lo = 0, hi = 0;
      if (!zero) {
        lo = (a4_code(x0, t) | (a4_code(x1, t) << 4)) << (8 * c);
        hi = (a4_code(x2, t) | (a4_code(x3, t) << 4)) << (8 * c);
      }
      lo |= __shfl_xor_sync(0xFFFFFFFFu, lo, 1);
      hi |= __shfl_xor_sync(0xFFFFFFFFu, hi, 1);
      lo |= __shfl_xor_sync(0xFFFFFFFFu, lo, 2);
      hi |= __shfl_xor_sync(0xFFFFFFFFu, hi, 2);
      word[h][G] = make_uint2(lo, hi);
      if (act != nullptr && (h ? v1 : v0)) {
        __nv_bfloat16* row = act + (size_t)(h ? pr1 : pr0) * width + cb * 32 + 16 * G + 2 * c;
        *reinterpret_cast<__nv_bfloat162*>(row) = __floats2bfloat162_rn(x0, x1);
        *reinterpret_cast<__nv_bfloat162*>(row + 8) = __floats2bfloat162_rn(x2, x3);
      }
    }
  const int h = c >> 1, G = c & 1;
  if (!(h ? v1 : v0)) return;
  const uint2 w = h ? (G ? word[1][1] : word[1][0]) : (G ? word[0][1] : word[0][0]);
  const uint32_t sb = h ? (G ? sbyte[1][1] : sbyte[1][0]) : (G ? sbyte[0][1] : sbyte[0][0]);
  const size_t at = h ? at1 : at0;
  const int grp = 2 * cb + G;
  const int padded_width = (width + 63) / 64 * 64;
  *reinterpret_cast<uint2*>(dq.codes + at * (padded_width / 2) + grp * 8) = w;
  dq.scales[at * (padded_width / 16) + grp] = static_cast<uint8_t>(sb);
}

}  // namespace
