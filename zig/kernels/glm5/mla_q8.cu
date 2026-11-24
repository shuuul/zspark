// GLM-5.3-Flash's DSA chunks over an FP8 cache with the queries and the softmax's weights in FP8, as vLLM runs
// them on this card (spec 114 WS-05; `mla.cu` has the latent, the chunks, the merge and the C ABI). vLLM's GLM
// attend on `fp8_ds_mla` requires its FP8 compute mode (eugr/spark-vllm-b12x a7f76ae9f18c, FlashInfer b12x
// `attention/_shared/mla/traits.py` 297-331) and takes it in decode and extend alike (`prefill.py` 425-445,
// `prefill_mg.py` 2933, 3278, 3515-3560), so a row's form is the same in every phase. Its steps, from
// `decode_math.py`:
//   queries: a (head, group of 128) scale s = pow2ceil(max(amax, 1e-4) * (1 / 448)), codes e4m3(clamp(x / s))
//     (`s0_quantize_q_to_smem` 300-560);
//   scores: a group's 128 products by the block-scaled e4m3 mma from zero (the query's scale its ue8m0 byte, the
//     key's codes raw at unit scale), then the groups' sums times the key's fp32 scales (`s1_qk_nope_block_scaled`
//     707-883);
//   P.V: a (head, group) scale of a tile, sc = max(max over its keys of |p s_g|, 1e-10) * (1 / 448); the weight
//     w = p s_g (1 / sc) as two e4m3 codes, HIGH = e4m3(clamp(w)) and LOW = e4m3(clamp(w - HIGH)); HIGH . codes
//     and LOW . codes by the plain e4m3 mma into one fp32 sum, which the head's running output takes times sc
//     (`s6_xv_nope` 2042-2427, `_quant_e4m3_residual_byte` 3309-3320).
// Departures: a tile is 32 keys (vLLM's 64), so P's scales are near vLLM's and not its bits; the softmax is
// `mla.cu`'s (expf, the scores times SCALE after the groups' sum); V's fragment is built by `ldmatrix.trans` and
// two byte permutes (vLLM: eight 4-byte loads and six), so the keys' order within an mma is another fixed one.
// The row rule: a query's scales are its own head's, a tile's keys are fixed by the row's own positions or list,
// and an mma's element is a function of its own row of A and column of B, so a window's row, a serial step and a
// prompt chunk give the same bits (`dsa_test`).
// Block (row, 32 heads, chunk) of 8 warps as `mla_chunk_kernel`: two groups of four, a group's 16 heads the mmas'
// rows; for the scores a warp takes 8 keys of a tile, for P.V a quarter of the latent, which is one scale group.
// The records stay as stored: the scores' B is `ldmatrix` straight from the stage (528 bytes a key, so a matrix's
// eight rows fall in eight bank groups), with no widening and no bf16 tile.
// Shared memory, 38,528 B a block, two blocks an SM (`mla_chunk_kernel<true>`: 91,648, one):
//   q8 [HEADS, LW] e4m3         16,384  the heads' query codes, read by ldmatrix every step of every tile's
//                                       score (swizzled by 16-byte pieces);
//   qe [HEADS] u32                 128  a head's four ue8m0 scale bytes, the scores' scale A;
//   stage [TILE, REC]           16,896  a tile's records by cp.async, the bytes in flight: both products' B and
//                                       the keys' scales; one tile, the next asked for into L1 meanwhile;
//   pf [2, 4, 32] float4         4,096  a lane's p of two heads and two keys, where a group's warps meet: a
//                                       lane's P.V fragment is its own place in the four warps (no transpose);
//   part [2, 4, HEADS] fp32      1,024  a warp's largest score and its sum of p.
// Could be less: pf could be the queries' unused rows when H - 32 hb is 16 or less (not tried); a second stage
// would leave one block an SM. The window form (WHOLE, below) adds none: the row's running output is po's
// [R, H, LW] in L2, not shared memory (its 64 KB beside these 38.5 are past the 101,376 B a block may take), nor registers
// (64 more a thread past `__launch_bounds__(256, 2)`'s 128).
// WHOLE (spec 116 WS-01, `mla.cu`'s window form, from `zs_glm5_mla_window`'s rows): a block runs its row's chunks
// in order and folds each as `mla_merge_kernel` does (`mla.cuh` `fold_of`, `folded`): the merge's bits. 128
// registers, 48 bytes spilled (the chunk form 80). Alone (`profile_test.zig` `attend8`, a list of 2,051 keys a
// row, off, on, off, on, machine alone, 2026-10-20): a 4,096-row window 24.85 / 24.87 ms with the merge to
// 12.70 / 12.59, 512 rows 3.13 / 3.09 to 1.67 / 1.67, 256 1.55 to 0.92, 128 0.74 to 0.53, 64 0.34 to 0.32-0.41,
// 32 0.19 to 0.23, 8 0.073 to 0.226; dropping the running output's lines from L2 after their last read is 0.3
// to 0.5 ms of the 4,096 rows' 12.7 (their 268 MB not written back).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../common/copies.cuh"
#include "../common/qmm_frag.cuh"
#include "../nvfp4/codes.cuh"
#include "mla.cuh"

namespace glm5_mla {
namespace {

using namespace common_copies;

constexpr int FR8 = LW / 4 / 16;   // a warp's quarter of the latent in 16-value blocks, a pair of mmas each
constexpr int SMEM_Q8 = HEADS * LW + HEADS * 4 + TILE * REC + 2 * 4 * 32 * 16 + 2 * 4 * HEADS * 4;
static_assert(SMEM_Q8 == 38528 && SMEM_Q8 <= 49152 && 2 * (SMEM_Q8 + 1024) <= 102400, "no opt-in; two blocks an SM");
static_assert(LW / 4 == GROUP && TILE == 32, "a warp's quarter is a scale group; a tile is one k32 step");

// The block-scaled e4m3 mma from zero: A's scale the BYTE-th byte of sa (lane 4 g: row g, 4 g + 1: row g + 8),
// B's unit (0x7F).
template <int BYTE>
__device__ __forceinline__ void mma_mx0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa) {
  const float z = 0.0f;
  asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%12, %12, %12, %12}, %10, {%13, %14}, %11, {%13, %14};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(0x7F7F7F7Fu), "f"(z), "n"(BYTE),
        "n"(0));
}

template <int BYTE>
__device__ __forceinline__ void mma_mx(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa) {
  asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%12, %13}, %11, {%12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(0x7F7F7F7Fu), "n"(BYTE), "n"(0));
}

// The plain e4m3 mma, accumulating (`cuda-gb10`: its fragments are the block-scaled one's).
__device__ __forceinline__ void mma8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t e4m3(float v) {
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float clamp448(float v) { return fmaxf(-448.f, fminf(448.f, v)); }

// 8 of a query's values from p, or zeros without reading.
__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* v, bool read) {
  const uint4 x = read ? *reinterpret_cast<const uint4*>(p) : make_uint4(0, 0, 0, 0);
  const uint32_t w[4] = {x.x, x.y, x.z, x.w};
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

// The scores' four groups, step k32 by step over the latent, a group's sum from zero (its scale byte the group).
template <int G>
__device__ __forceinline__ void group_sum(float (&c)[4], const unsigned char* q8, const unsigned char* stage, int head,
                                          int key, int lane, uint32_t sa) {
  for (int s = 0; s < GROUP / 32; ++s) {
    const int step = G * (GROUP / 32) + s;
    uint32_t a[4], b[2];
    qmm_frag::ldmatrix4(a, q8 + head * LW + ((step * 2 + (lane >> 4)) ^ (head % 8)) * 16);
    qmm_frag::ldmatrix2(b, stage + key * REC + (step * 2 + ((lane >> 3) & 1)) * 16);
    if (s == 0) mma_mx0<G>(c, a, b[0], b[1], sa);
    else mma_mx<G>(c, a, b[0], b[1], sa);
  }
}

template <bool WHOLE>
__global__ void __launch_bounds__(THREADS, 2)
    q8_chunk_kernel(const __nv_bfloat16* __restrict__ QA, const unsigned char* __restrict__ LR,
                    const int* __restrict__ POS, const int* __restrict__ TABLE, float* __restrict__ PO,
                    float* __restrict__ PM, float* __restrict__ PL, const int* __restrict__ TOKENS,
                    const int* __restrict__ COUNTS, int WIDTH, float SCALE, int H, int NCH,
                    __nv_bfloat16* __restrict__ OUT) {
  extern __shared__ __align__(16) unsigned char shared[];
  unsigned char* const q8 = shared;                                        // [HEADS, LW]
  unsigned char* const qe = q8 + HEADS * LW;                               // [HEADS, GROUPS]
  unsigned char* const stage = qe + HEADS * 4;                             // [TILE, REC]
  float4* const pf = reinterpret_cast<float4*>(stage + TILE * REC);        // [2, 4, 32]
  float* const part_m = reinterpret_cast<float*>(pf + 2 * 4 * 32);         // [4, HEADS]
  float* const part_l = part_m + 4 * HEADS;                                // [4, HEADS]
  const int r = blockIdx.x, hb = blockIdx.y, tid = threadIdx.x;
  const int n = keys_of(POS, COUNTS, r);
  const int* const list = COUNTS && COUNTS[r] > 0 ? TOKENS + (size_t)r * WIDTH : nullptr;
  if (!WHOLE && (int)blockIdx.z * CHUNK >= n) return;
  const int G = min(HEADS, H - hb * HEADS);
  const int grp = tid / 128, h0 = grp * 16, warp = tid / 32 % 4, lane = tid % 32, row8 = lane >> 2, pair = lane & 3;
  {
    // a thread half a (head, group): 64 values, the pair's largest |value| by one shuffle; zeros past H
    const int head = tid / 8, g = tid / 2 % 4, half = tid % 2;
    const __nv_bfloat16* in = QA + ((size_t)r * H + hb * HEADS + head) * LW + g * GROUP + half * 64;
    float most = 0.f;
    for (int j = 0; j < 8; ++j) {
      float v[8];
      load8(in + j * 8, v, head < G);
      for (int i = 0; i < 8; ++i) most = fmaxf(most, fabsf(v[i]));
    }
    most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, 1));
    // pow2ceil: a mantissa's bits round the exponent up
    uint32_t bits = __float_as_uint(fmaxf(most, 1e-4f) * (1.f / 448.f));
    if (bits & 0x7fffffu) bits = (bits & 0xff800000u) + 0x800000u;
    const float inv = 1.f / __uint_as_float(bits);
    // read again (from L1) rather than held: 64 registers a thread
    for (int j = 0; j < 4; ++j) {
      float v[16];
      load8(in + j * 16, v, head < G);
      load8(in + j * 16 + 8, v + 8, head < G);
      uint32_t w[4];
      for (int i = 0; i < 4; ++i)
        w[i] = e4m3(clamp448(v[4 * i] * inv)) | e4m3(clamp448(v[4 * i + 1] * inv)) << 8 |
               e4m3(clamp448(v[4 * i + 2] * inv)) << 16 | e4m3(clamp448(v[4 * i + 3] * inv)) << 24;
      const int ch = g * 8 + half * 4 + j;
      *reinterpret_cast<uint4*>(q8 + head * LW + (ch ^ (head % 8)) * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    }
    if (half == 0) qe[head * 4 + g] = static_cast<unsigned char>(bits >> 23);
  }
  // WHOLE: the row's running m and l of heads h0 + row8 and h0 + row8 + 8, as the merge's from nothing
  float run_m[2] = {minus_inf(), minus_inf()}, run_l[2] = {0.f, 0.f};
  const int chunks = WHOLE ? min((n + CHUNK - 1) / CHUNK, NCH) : 1;
  for (int ci = 0; ci < chunks; ++ci) {
    const int c = WHOLE ? ci : blockIdx.z, start = c * CHUNK;
    float o[FR8][8];
    for (int f = 0; f < FR8; ++f)
      for (int e = 0; e < 8; ++e) o[f][e] = 0.f;
    // of heads h0 + row8 and h0 + row8 + 8
    float m[2] = {minus_inf(), minus_inf()}, l[2] = {0.f, 0.f};
    const int keys = min(n - start, CHUNK);
    for (int t0 = 0; t0 < keys; t0 += TILE) {
      {
        // a thread a key's eighth of its record, the row found once; the next tile's same bytes asked for
        const int key = tid >> 3, ki = start + t0 + key;
        const bool here = ki < n;
        const int row = here ? placed(TABLE, list ? list[ki] : ki) : 0;
        const unsigned char* from = LR + (size_t)row * REC;
        for (int ch = tid & 7; ch < REC / 16; ch += 8) qmm_frag::cp16z(stage + key * REC + ch * 16, from + ch * 16, here);
        const int kn = ki + TILE;
        // WHOLE: across a chunk's end too, the next chunk being this block's
        if (kn < n && (WHOLE || t0 + TILE < keys)) {
          const unsigned char* next = LR + (size_t)placed(TABLE, list ? list[kn] : kn) * REC + (tid & 7) * 64;
          ask(next);
          if ((tid & 7) == 7) ask(next + 64);
        }
      }
      qmm_frag::commit();
      qmm_frag::wait<0>();
      __syncthreads();
      float alpha[2];
      {
        const int head = h0 + lane % 16, key = warp * 8 + (lane & 7);
        const int k0 = warp * 8 + pair * 2;
        const int mine[2] = {k0, k0 + 1};
        const uint32_t sa = *reinterpret_cast<const uint32_t*>(qe + (h0 + row8 + 8 * (pair & 1)) * 4);
        float cg[GROUPS][4];
        group_sum<0>(cg[0], q8, stage, head, key, lane, sa);
        group_sum<1>(cg[1], q8, stage, head, key, lane, sa);
        group_sum<2>(cg[2], q8, stage, head, key, lane, sa);
        group_sum<3>(cg[3], q8, stage, head, key, lane, sa);
        float c0[4];
        for (int e = 0; e < 4; ++e) {
          const float* sk = reinterpret_cast<const float*>(stage + mine[e & 1] * REC + LW);
          c0[e] = fmaf(cg[3][e], sk[3], fmaf(cg[2][e], sk[2], fmaf(cg[1][e], sk[1], cg[0][e] * sk[0])));
        }
        float sc[2][2] = {{c0[0], c0[1]}, {c0[2], c0[3]}};
        bool live[2];
        for (int j = 0; j < 2; ++j) {
          live[j] = start + t0 + mine[j] < n;
          for (int h = 0; h < 2; ++h) sc[h][j] = live[j] ? sc[h][j] * SCALE : minus_inf();
        }
        for (int h = 0; h < 2; ++h) {
          float top = fmaxf(sc[h][0], sc[h][1]);
          top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
          top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
          if (pair == 0) part_m[warp * HEADS + h0 + row8 + 8 * h] = top;
        }
        __syncthreads();
        float pk[2][2];
        for (int h = 0; h < 2; ++h) {
          const int at = h0 + row8 + 8 * h;
          const float tile_m = fmaxf(fmaxf(part_m[at], part_m[HEADS + at]), fmaxf(part_m[2 * HEADS + at], part_m[3 * HEADS + at]));
          const bool active = tile_m != minus_inf();
          const float next_m = active ? fmaxf(m[h], tile_m) : m[h];
          alpha[h] = active ? (m[h] == minus_inf() ? 0.f : expf(m[h] - next_m)) : 1.f;
          m[h] = next_m;
          for (int j = 0; j < 2; ++j) pk[h][j] = active && live[j] ? expf(sc[h][j] - next_m) : 0.f;
          float sum = pk[h][0] + pk[h][1];
          sum += __shfl_xor_sync(0xffffffffu, sum, 1);
          sum += __shfl_xor_sync(0xffffffffu, sum, 2);
          if (pair == 0) part_l[warp * HEADS + at] = sum;
          // a head past H keeps p = 0
          if (at >= G) pk[h][0] = pk[h][1] = 0.f;
        }
        pf[(grp * 4 + warp) * 32 + lane] = make_float4(pk[0][0], pk[0][1], pk[1][0], pk[1][1]);
      }
      __syncthreads();
      {
        // A, this lane's: heads h0 + row8 (a0, a2) and + 8 (a1, a3); k slot 4 pair + i of a0 / a1 is the key
        // 8 (i / 2) + 2 pair + i % 2, of a2 / a3 that plus 16, which warp i / 2 (+ 2) held at this lane
        float u[4][4];   // [warp w][head lo key, lo key + 1, hi key, hi key + 1], times the key's scale of this group
        float most[2] = {0.f, 0.f};
        for (int w = 0; w < 4; ++w) {
          const float4 x = pf[(grp * 4 + w) * 32 + lane];
          const int k = w * 8 + pair * 2;
          const float s0 = *reinterpret_cast<const float*>(stage + k * REC + LW + warp * 4);
          const float s1 = *reinterpret_cast<const float*>(stage + (k + 1) * REC + LW + warp * 4);
          u[w][0] = x.x * s0;
          u[w][1] = x.y * s1;
          u[w][2] = x.z * s0;
          u[w][3] = x.w * s1;
          most[0] = fmaxf(most[0], fmaxf(fabsf(u[w][0]), fabsf(u[w][1])));
          most[1] = fmaxf(most[1], fmaxf(fabsf(u[w][2]), fabsf(u[w][3])));
        }
        float scl[2], inv[2];
        for (int h = 0; h < 2; ++h) {
          most[h] = fmaxf(most[h], __shfl_xor_sync(0xffffffffu, most[h], 1));
          most[h] = fmaxf(most[h], __shfl_xor_sync(0xffffffffu, most[h], 2));
          scl[h] = fmaxf(most[h], 1e-10f) * (1.f / 448.f);
          inv[h] = 1.f / scl[h];
        }
        uint32_t hi[4] = {0, 0, 0, 0}, lo[4] = {0, 0, 0, 0};
        for (int w = 0; w < 4; ++w)
          for (int e = 0; e < 4; ++e) {
            const int h = e / 2;
            const float x = clamp448(u[w][e] * inv[h]);
            const uint32_t top = e4m3(x);
            const uint32_t rest = e4m3(clamp448(x - nvfp4_codes::e4m3f(top)));
            // register: (w / 2) * 2 + h; byte: (w % 2) * 2 + e % 2
            const int reg = (w / 2) * 2 + h, sh = 8 * ((w % 2) * 2 + e % 2);
            hi[reg] |= top << sh;
            lo[reg] |= rest << sh;
          }
        for (int f = 0; f < FR8; ++f) {
          // V's B: keys 0-7, 8-15, 16-23, 24-31 of 16 values; a lane's register holds keys 2 pair, 2 pair + 1 of
          // values 2 row8, 2 row8 + 1: even and odd values apart are k slots 4 pair + i of the keys above
          uint32_t v[4];
          ldmatrix4t(v, stage + lane * REC + warp * GROUP + f * 16);
          const uint32_t e0 = __byte_perm(v[0], v[1], 0x6420), d0 = __byte_perm(v[0], v[1], 0x7531);
          const uint32_t e1 = __byte_perm(v[2], v[3], 0x6420), d1 = __byte_perm(v[2], v[3], 0x7531);
          float even[4] = {0.f, 0.f, 0.f, 0.f}, odd[4] = {0.f, 0.f, 0.f, 0.f};
          mma8(even, hi, e0, e1);
          mma8(even, lo, e0, e1);
          mma8(odd, hi, d0, d1);
          mma8(odd, lo, d0, d1);
          // column j of even is value 2 j of the block, of odd 2 j + 1: this lane's values 4 pair .. 4 pair + 3
          const float x[8] = {even[0], odd[0], even[1], odd[1], even[2], odd[2], even[3], odd[3]};
          for (int e = 0; e < 8; ++e) o[f][e] = fmaf(x[e], scl[e / 4], o[f][e] * alpha[e / 4]);
        }
        for (int h = 0; h < 2; ++h) {
          const int at = h0 + row8 + 8 * h;
          l[h] = l[h] * alpha[h] + (((part_l[at] + part_l[HEADS + at]) + part_l[2 * HEADS + at]) + part_l[3 * HEADS + at]);
        }
      }
      __syncthreads();
    }
    const size_t base = ((size_t)r * NCH + c) * H + hb * HEADS + h0;
    const int left = G - h0;
    if constexpr (WHOLE) {
      // the merge's step for this chunk, a thread its own values; the running output is the row's of po as fp32
      // [R, H, LW], which only this thread reads or writes, and a row's last fold writes bf16(o / l) and drops the
      // lines from L2
      const size_t slot = (size_t)r * H + hb * HEADS + h0;
      const bool last = ci + 1 == chunks;
  #pragma unroll
      for (int h = 0; h < 2; ++h) {
        if (row8 + 8 * h >= left) continue;
        const Fold w = fold_of(run_m[h], m[h], l[h]);
        run_l[h] = folded(run_l[h], l[h], w);
        run_m[h] = w.m;
        float* const at = PO + (slot + row8 + 8 * h) * LW + warp * GROUP + pair * 4;
        __nv_bfloat16* const to = OUT + ((size_t)r * H + hb * HEADS + h0 + row8 + 8 * h) * LW + warp * GROUP + pair * 4;
        for (int f = 0; f < FR8; ++f) {
          float4 x = ci == 0 ? make_float4(0.f, 0.f, 0.f, 0.f) : *reinterpret_cast<const float4*>(at + f * 16);
          x.x = folded(x.x, o[f][4 * h], w);
          x.y = folded(x.y, o[f][4 * h + 1], w);
          x.z = folded(x.z, o[f][4 * h + 2], w);
          x.w = folded(x.w, o[f][4 * h + 3], w);
          if (!last) {
            *reinterpret_cast<float4*>(at + f * 16) = x;
          } else {
            const float ll = run_l[h];
            *reinterpret_cast<__nv_bfloat162*>(to + f * 16) = __floats2bfloat162_rn(x.x / ll, x.y / ll);
            *reinterpret_cast<__nv_bfloat162*>(to + f * 16 + 2) = __floats2bfloat162_rn(x.z / ll, x.w / ll);
          }
        }
        // a line is 32 values: two blocks of 16 of the four lanes of this head
        if (last && ci > 0) {
          __syncwarp(0xfu << (lane & ~3));
          if (pair == 0)
            for (int f = 0; f < FR8; f += 2) drop_line(at + f * 16);
        }
      }
    } else {
      for (int f = 0; f < FR8; ++f) {
        const int d = warp * GROUP + f * 16 + pair * 4;
        if (row8 < left) *reinterpret_cast<float4*>(PO + (base + row8) * LW + d) = make_float4(o[f][0], o[f][1], o[f][2], o[f][3]);
        if (row8 + 8 < left)
          *reinterpret_cast<float4*>(PO + (base + row8 + 8) * LW + d) = make_float4(o[f][4], o[f][5], o[f][6], o[f][7]);
      }
      if (warp == 0 && pair == 0) {
        if (row8 < left) {
          PM[base + row8] = m[0];
          PL[base + row8] = l[0];
        }
        if (row8 + 8 < left) {
          PM[base + row8 + 8] = m[1];
          PL[base + row8 + 8] = l[1];
        }
      }
    }
  }
}

}  // namespace

void q8_chunks(dim3 grid, cudaStream_t stream, const void* qa, const unsigned char* cache, const int* pos,
               const int* table, float* po, float* pm, float* pl, const int* tokens, const int* counts, int width,
               float scale, int h, int nch, void* out) {
  // under the 48 KB a block takes without asking
  (out ? q8_chunk_kernel<true> : q8_chunk_kernel<false>)<<<grid, THREADS, SMEM_Q8, stream>>>(
      static_cast<const __nv_bfloat16*>(qa), cache, pos, table, po, pm, pl, tokens, counts, width, scale, h, nch,
      static_cast<__nv_bfloat16*>(out));
}

}  // namespace glm5_mla
