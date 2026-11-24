// GLM-5.3-Flash's DSA projections and cache writes beside `mla.cu`'s attention (split from it in spec 121 WS-06):
// absorb takes kv_b's key rows into a row's query, qa = bf16(q W_k); expand takes its value rows out of the
// attention's result, out = bf16(W_v ol); the writes put a row's latent into the cache, bf16 or an FP8 record.
// Semantics from TensorFold 0.6.6 (cb2ebf0) families/glm5_next/cuda/latent.py (_absorb_q, _expand_v, _lat_write;
// Triton), Apache-2.0; rewritten for zspark in CUDA C++: the BF16 kv_b alone (the EXL3 pack's).
// Absorb and expand are GEMMs, a head each, and are by hand (spec 068 WS-03): a serial step is one row against
// 256 KB of a head's weights, 32 heads a rank, so its time is the weights' bytes, which the warps read once
// straight into their fragments; a CUTLASS tile stages 128 rows' worth to shared memory whatever R is, and takes
// a launch a head. Not measured against it. TensorFold's Triton sums a row's products in fp32 in its compiler's
// order; here they are the tensor cores' (the owner, 2026-10-09), near and not equal. Measured on the GB10 (CUDA
// 13, nsys, 8 layers of the pack, both ranks in one process, H 32, DK and DV 256; a call's mean): a serial step's
// absorb 269 -> 41 us and expand 231 -> 67 us over 520 calls; a 64-row window's 1.79 ms -> 51 us and 8.06 ms ->
// 78 us over 32. Two warps a block for expand were no faster (69 us).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../common/qmm_frag.cuh"
#include "mla.cuh"
#include "../zs/host.h"

namespace {

using namespace glm5_mla;


constexpr int MAX_D = 256;     // a head's key or value width, at most

// Absorb and expand are a GEMM a head, R rows by the head's block of kv_b, on the tensor cores: m16n8k16 steps
// on bf16 with an fp32 accumulator. For every element (row r, output n) of both:
//   the sum starts at +0 and takes the K products 16 at a step, steps in the order of k (k = 16 s .. 16 s + 15
//   at step s), each step one mma that adds its 16 products to the running sum.
// K is DK (absorb) or 512 (expand), so the steps are the weights' shape alone. An mma's element is a function of
// its own row of A and its own column of B and of nothing else in the fragments (`mla.cu`'s chunk kernel
// stands on the same), so a row's bits depend neither on the rows that share its mma, nor on the slot it has there,
// nor on R: a window's row r is slot r % 16 and a serial step's is slot 0, with the same bits. Rows past R are
// zero fragments, never read and never stored; an output past DV likewise. How many rows a warp carries (MT
// tiles of 16) changes which launches hold an element and nothing of its arithmetic.
// Within a step a thread's 4 consecutive k come by one 8-byte load and fill its two halves of the fragment, so
// the mma's column 2 p + i is k = 16 s + 4 p + i and its column 8 + 2 p + i is k = 16 s + 4 p + 2 + i (p the
// lane's pair, i 0 or 1), in A and B alike. The fragments are loaded from global memory as they are used: a
// serial step is the weights' bytes (256 KB a head and kernel, read once), and nothing is staged.
constexpr int NT = 4;          // 8-output tiles a warp carries: 32 outputs
constexpr int PW = 4;          // warps a block

// 4 consecutive bf16, or zeros without reading.
__device__ __forceinline__ uint64_t quad(const __nv_bfloat16* p, bool read) {
  return read ? *reinterpret_cast<const uint64_t*>(p) : 0ull;
}

__device__ __forceinline__ void put2(__nv_bfloat16* p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

// Block (MT * 16 rows, head, 4 warps of 32 outputs each): qa[r][n] = bf16(sum over k of q[r][k] W[k][n]), W the
// head's first DK rows of kv_b [H, DK + DV, LW]. A warp's 32 outputs n0 .. n0 + 31 are 4 tiles, tile t the
// outputs n0 + 4 j + t (j the mma's column): a thread's 8-byte load of row k at n0 + 4 j is that row's value in
// each of the four tiles, so a step reads 64 bytes of 16 rows of W, whole.
template <int MT>
__global__ void __launch_bounds__(PW * 32)
    absorb_kernel(const __nv_bfloat16* __restrict__ Q, long long q_stride, const __nv_bfloat16* __restrict__ W,
                  __nv_bfloat16* __restrict__ OUT, int R, int H, int DK, int DV) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, row8 = lane >> 2, pair = lane & 3;
  const int h = blockIdx.y;
  constexpr int NB = LW / (PW * NT * 8);   // blocks across the latent
  const int n0 = ((blockIdx.x % NB) * PW + warp) * NT * 8;
  const int r0 = (blockIdx.x / NB) * MT * 16;
  const __nv_bfloat16* w = W + (size_t)h * (DK + DV) * LW + n0 + row8 * 4;
  const __nv_bfloat16* q = Q + (size_t)h * DK;
  float c[MT][NT][4];
  for (int m = 0; m < MT; ++m)
    for (int t = 0; t < NT; ++t)
      for (int e = 0; e < 4; ++e) c[m][t][e] = 0.f;
  for (int k = pair * 4; k < DK; k += 16) {
    uint64_t x[4];
    for (int i = 0; i < 4; ++i) x[i] = quad(w + (size_t)(k + i) * LW, true);
    uint32_t b[NT][2];
    for (int t = 0; t < NT; ++t) {
      b[t][0] = static_cast<uint32_t>((x[0] >> (16 * t)) & 0xffff) | static_cast<uint32_t>((x[1] >> (16 * t)) & 0xffff) << 16;
      b[t][1] = static_cast<uint32_t>((x[2] >> (16 * t)) & 0xffff) | static_cast<uint32_t>((x[3] >> (16 * t)) & 0xffff) << 16;
    }
    for (int m = 0; m < MT; ++m) {
      const int r = r0 + m * 16 + row8;
      if (r0 + m * 16 >= R) break;
      const uint64_t low = quad(q + (long long)r * q_stride + k, r < R);
      const uint64_t high = quad(q + (long long)(r + 8) * q_stride + k, r + 8 < R);
      const uint32_t a[4] = {static_cast<uint32_t>(low), static_cast<uint32_t>(high), static_cast<uint32_t>(low >> 32),
                             static_cast<uint32_t>(high >> 32)};
      for (int t = 0; t < NT; ++t) qmm_frag::mma(c[m][t], a, b[t][0], b[t][1]);
    }
  }
  // this thread's columns 2 p and 2 p + 1 of the four tiles: the outputs n0 + 8 p .. n0 + 8 p + 7
  for (int m = 0; m < MT; ++m)
    for (int half = 0; half < 2; ++half) {
      const int r = r0 + m * 16 + row8 + 8 * half;
      if (r >= R) continue;
      __nv_bfloat16* out = OUT + ((size_t)r * H + h) * LW + n0 + pair * 8;
      put2(out, c[m][0][2 * half], c[m][1][2 * half]);
      put2(out + 2, c[m][2][2 * half], c[m][3][2 * half]);
      put2(out + 4, c[m][0][2 * half + 1], c[m][1][2 * half + 1]);
      put2(out + 6, c[m][2][2 * half + 1], c[m][3][2 * half + 1]);
    }
}

// Block (MT * 16 rows, head, 4 warps of 32 outputs each): out[r][n] = bf16(sum over the latent's k of ol[r][k]
// W[DK + n][k]). A warp's tile t is the outputs n0 + 8 t .. n0 + 8 t + 7; a thread's 8-byte load of W's row n
// is its B fragment of that tile.
template <int MT>
__global__ void __launch_bounds__(PW * 32)
    expand_kernel(const __nv_bfloat16* __restrict__ OL, const __nv_bfloat16* __restrict__ W,
                  __nv_bfloat16* __restrict__ OUT, int R, int H, int DK, int DV) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, row8 = lane >> 2, pair = lane & 3;
  const int h = blockIdx.y;
  const int nb = (DV + PW * NT * 8 - 1) / (PW * NT * 8);   // blocks across a head's outputs
  const int n0 = ((blockIdx.x % nb) * PW + warp) * NT * 8;
  const int r0 = (blockIdx.x / nb) * MT * 16;
  if (n0 >= DV) return;
  const __nv_bfloat16* w = W + ((size_t)h * (DK + DV) + DK + n0 + row8) * LW;
  const __nv_bfloat16* ol = OL + (size_t)h * LW;
  float c[MT][NT][4];
  for (int m = 0; m < MT; ++m)
    for (int t = 0; t < NT; ++t)
      for (int e = 0; e < 4; ++e) c[m][t][e] = 0.f;
  for (int k = pair * 4; k < LW; k += 16) {
    uint32_t b[NT][2];
    for (int t = 0; t < NT; ++t) {
      const uint64_t x = quad(w + (size_t)t * 8 * LW + k, n0 + t * 8 + row8 < DV);
      b[t][0] = static_cast<uint32_t>(x);
      b[t][1] = static_cast<uint32_t>(x >> 32);
    }
    for (int m = 0; m < MT; ++m) {
      const int r = r0 + m * 16 + row8;
      if (r0 + m * 16 >= R) break;
      const uint64_t low = quad(ol + (size_t)r * H * LW + k, r < R);
      const uint64_t high = quad(ol + (size_t)(r + 8) * H * LW + k, r + 8 < R);
      const uint32_t a[4] = {static_cast<uint32_t>(low), static_cast<uint32_t>(high), static_cast<uint32_t>(low >> 32),
                             static_cast<uint32_t>(high >> 32)};
      for (int t = 0; t < NT; ++t) qmm_frag::mma(c[m][t], a, b[t][0], b[t][1]);
    }
  }
  for (int m = 0; m < MT; ++m)
    for (int half = 0; half < 2; ++half) {
      const int r = r0 + m * 16 + row8 + 8 * half;
      if (r >= R) continue;
      __nv_bfloat16* out = OUT + ((size_t)r * H + h) * DV;
      for (int t = 0; t < NT; ++t)
        for (int i = 0; i < 2; ++i) {
          const int n = n0 + t * 8 + pair * 2 + i;
          if (n < DV) out[n] = __float2bfloat16_rn(c[m][t][2 * half + i]);
        }
    }
}
// `expand_kernel` asking for the weights before it loads them (ZSPARK_GLM_MLA_ASK, spec 073 WS-05). The base
// reads a head's 256 rows of kv_b a step of 32 bytes at a time, 128 rows a block, and waits for each: 63 us
// from DRAM against 10 from L2, 8.4 MB at 133 GB/s. A prefetch is not waited for (`cuda-gb10`): ASK below zero
// asks for a thread's sixteen lines of its four rows before the first step, ASK above it for the step that
// many ahead. No arithmetic is touched: the base's bits.
template <int MT>
__global__ void __launch_bounds__(PW * 32)
    expand_asked_kernel(const __nv_bfloat16* __restrict__ OL, const __nv_bfloat16* __restrict__ W,
                  __nv_bfloat16* __restrict__ OUT, int R, int H, int DK, int DV, int ASK) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, row8 = lane >> 2, pair = lane & 3;
  const int h = blockIdx.y;
  const int nb = (DV + PW * NT * 8 - 1) / (PW * NT * 8);   // blocks across a head's outputs
  const int n0 = ((blockIdx.x % nb) * PW + warp) * NT * 8;
  const int r0 = (blockIdx.x / nb) * MT * 16;
  if (n0 >= DV) return;
  const __nv_bfloat16* w = W + ((size_t)h * (DK + DV) + DK + n0 + row8) * LW;
  const __nv_bfloat16* ol = OL + (size_t)h * LW;
  float c[MT][NT][4];
  for (int m = 0; m < MT; ++m)
    for (int t = 0; t < NT; ++t)
      for (int e = 0; e < 4; ++e) c[m][t][e] = 0.f;
  if (ASK < 0)
    for (int t = 0; t < NT; ++t)
      for (int j = 0; j < 4; ++j)
        if (n0 + t * 8 + row8 < DV) ask(w + (size_t)t * 8 * LW + (pair * 4 + j) * 32);
  for (int k = pair * 4; k < LW; k += 16) {
    if (ASK > 0 && k + ASK * 16 < LW)
      for (int t = 0; t < NT; ++t)
        if (n0 + t * 8 + row8 < DV) ask(w + (size_t)t * 8 * LW + k + ASK * 16);
    uint32_t b[NT][2];
    for (int t = 0; t < NT; ++t) {
      const uint64_t x = quad(w + (size_t)t * 8 * LW + k, n0 + t * 8 + row8 < DV);
      b[t][0] = static_cast<uint32_t>(x);
      b[t][1] = static_cast<uint32_t>(x >> 32);
    }
    for (int m = 0; m < MT; ++m) {
      const int r = r0 + m * 16 + row8;
      if (r0 + m * 16 >= R) break;
      const uint64_t low = quad(ol + (size_t)r * H * LW + k, r < R);
      const uint64_t high = quad(ol + (size_t)(r + 8) * H * LW + k, r + 8 < R);
      const uint32_t a[4] = {static_cast<uint32_t>(low), static_cast<uint32_t>(high), static_cast<uint32_t>(low >> 32),
                             static_cast<uint32_t>(high >> 32)};
      for (int t = 0; t < NT; ++t) qmm_frag::mma(c[m][t], a, b[t][0], b[t][1]);
    }
  }
  for (int m = 0; m < MT; ++m)
    for (int half = 0; half < 2; ++half) {
      const int r = r0 + m * 16 + row8 + 8 * half;
      if (r >= R) continue;
      __nv_bfloat16* out = OUT + ((size_t)r * H + h) * DV;
      for (int t = 0; t < NT; ++t)
        for (int i = 0; i < 2; ++i) {
          const int n = n0 + t * 8 + pair * 2 + i;
          if (n < DV) out[n] = __float2bfloat16_rn(c[m][t][2 * half + i]);
        }
    }
}
static_assert(LW % (PW * NT * 8) == 0 && LW % 16 == 0, "absorb's blocks and expand's steps cover the latent");

// Thread (row, k): the row's latent into the cache at its position.
__global__ void write_kernel(const __nv_bfloat16* __restrict__ LAT, long long stride, __nv_bfloat16* __restrict__ LC,
                             const int* __restrict__ POS, const int* __restrict__ TABLE, int R) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * LW) return;
  const long long r = i / LW, k = i % LW;
  LC[(long long)placed(TABLE, POS[0] + (int)r) * LW + k] = LAT[r * stride + k];
}

// Block (row) of 128 threads, a warp a group of 128 values, a thread 4 of them: the row's latent as an FP8 record
// at its position. The group's largest |value| is a warp's max (any order gives it); scale = largest / 448, or 1
// for a group of zeros; a code is e4m3(value / scale), saturated at 448.
__global__ void __launch_bounds__(LW / 4)
    write_fp8_kernel(const __nv_bfloat16* __restrict__ LAT, long long stride, unsigned char* __restrict__ LC,
                     const int* __restrict__ POS, const int* __restrict__ TABLE) {
  const int r = blockIdx.x, t = threadIdx.x;
  const __nv_bfloat16* in = LAT + (long long)r * stride + t * 4;
  float v[4];
  for (int i = 0; i < 4; ++i) v[i] = __bfloat162float(in[i]);
  float most = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  for (int o = 16; o >= 1; o >>= 1) most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, o));
  const float scale = most > 0.f ? most / 448.f : 1.f;
  uint32_t codes = 0;
  for (int i = 0; i < 4; ++i)
    codes |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v[i] / scale, __NV_SATFINITE, __NV_E4M3)) << (8 * i);
  unsigned char* rec = LC + (long long)placed(TABLE, POS[0] + r) * REC;
  *reinterpret_cast<uint32_t*>(rec + t * 4) = codes;
  if (t % 32 == 0) reinterpret_cast<float*>(rec + LW)[t / 32] = scale;
}

// kv_b as a rank holds it: bf16 [H, DK + DV, LW], a head's key rows then its value rows. Its H, or 0.
int64_t heads_of(const zs::View& w, int64_t dk, int64_t dv) {
  if (!w.defined() || w.dtype() != ZS_BF16 || w.dim() != 3 || !w.is_contiguous()) return 0;
  if (w.size(1) != dk + dv || w.size(2) != LW || w.size(0) > INT32_MAX) return 0;
  return w.size(0);
}

// Rows a warp carries: one tile of 16 for a step and its drafts, two for a prompt's window.
template <class Launch>
void by_rows(int64_t rows, const Launch& launch) {
  if (rows <= 16) launch(absorb_kernel<1>, expand_kernel<1>, 16);
  else launch(absorb_kernel<2>, expand_kernel<2>, 32);
}

// `zs_glm5_mla_ask`: expand's ASK, 0 the base kernel.
int asked = 0;

bool aligned8(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 7) == 0; }

}  // namespace

extern "C" {

// q bf16 [R, H * DK] (rows may be strided by a multiple of 4 values) and kv_b bf16 [H, DK + DV, 512] -> out bf16
// [R, H, 512]. DK a multiple of 16 to 256; the buffers 8-byte aligned.
int zs_glm5_mla_absorb(const zs_view* q_, const zs_view* kv_b_, int64_t dk, const zs_view* out_) {
  zs::begin();
  const zs::View q(q_), w(kv_b_), out(out_);
  ZS_REQUIRE(dk >= 16 && dk <= MAX_D && dk % 16 == 0, "DK: a multiple of 16 to 256");
  ZS_REQUIRE(w.defined() && w.dim() == 3 && w.size(1) > dk, "kv_b: bf16 [H, DK + DV, 512]");
  const int64_t dv = w.size(1) - dk, h = heads_of(w, dk, dv);
  ZS_REQUIRE(h >= 1, "kv_b: bf16 [H, DK + DV, 512]");
  ZS_REQUIRE(q.defined() && q.dtype() == ZS_BF16 && q.dim() == 2 && q.size(1) == h * dk && q.size(0) >= 1 &&
             q.size(0) <= INT32_MAX - 32, "q: bf16 [R, H * DK]");
  const int64_t rows = q.size(0);
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * h * LW,
             "out: bf16 [R, H, 512]");
  ZS_REQUIRE(q.row_stride() % 4 == 0 && aligned8(q.data_ptr<__nv_bfloat16>()) && aligned8(w.data_ptr<__nv_bfloat16>()) &&
             aligned8(out.data_ptr<__nv_bfloat16>()), "q, kv_b, out: 8-byte aligned, q's rows a multiple of 4 values apart");
  by_rows(rows, [&](auto absorb, auto, int64_t tile) {
    const dim3 grid(static_cast<unsigned>((rows + tile - 1) / tile * (LW / (PW * NT * 8))), static_cast<unsigned>(h));
    absorb<<<grid, PW * 32, 0, zs::stream()>>>(
        q.data_ptr<__nv_bfloat16>(), q.row_stride(), w.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(),
        static_cast<int>(rows), static_cast<int>(h), static_cast<int>(dk), static_cast<int>(dv));
  });
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// ol bf16 [R, H, 512] and kv_b bf16 [H, DK + DV, 512] -> out bf16 [R, H * DV]. DV to 256; ol and kv_b 8-byte
// aligned.
int zs_glm5_mla_expand(const zs_view* ol_, const zs_view* kv_b_, int64_t dk, const zs_view* out_) {
  zs::begin();
  const zs::View ol(ol_), w(kv_b_), out(out_);
  ZS_REQUIRE(dk >= 1 && w.defined() && w.dim() == 3 && w.size(1) > dk && w.size(1) - dk <= MAX_D,
             "kv_b: bf16 [H, DK + DV, 512], DV to 256");
  const int64_t dv = w.size(1) - dk, h = heads_of(w, dk, dv);
  ZS_REQUIRE(h >= 1, "kv_b: bf16 [H, DK + DV, 512]");
  ZS_REQUIRE(ol.defined() && ol.dtype() == ZS_BF16 && ol.dim() == 3 && ol.is_contiguous() && ol.size(0) >= 1 &&
             ol.size(0) <= INT32_MAX - 32 && ol.size(1) == h && ol.size(2) == LW, "ol: bf16 [R, H, 512]");
  const int64_t rows = ol.size(0);
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * h * dv,
             "out: bf16 [R, H * DV]");
  ZS_REQUIRE(aligned8(ol.data_ptr<__nv_bfloat16>()) && aligned8(w.data_ptr<__nv_bfloat16>()),
             "ol, kv_b: 8-byte aligned");
  by_rows(rows, [&](auto, auto expand, int64_t tile) {
    const dim3 grid(static_cast<unsigned>((rows + tile - 1) / tile * ((dv + PW * NT * 8 - 1) / (PW * NT * 8))),
                    static_cast<unsigned>(h));
    if (asked != 0 && tile == 16)
      expand_asked_kernel<1><<<grid, PW * 32, 0, zs::stream()>>>(
          ol.data_ptr<__nv_bfloat16>(), w.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(),
          static_cast<int>(rows), static_cast<int>(h), static_cast<int>(dk), static_cast<int>(dv), asked);
    else
      expand<<<grid, PW * 32, 0, zs::stream()>>>(
          ol.data_ptr<__nv_bfloat16>(), w.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(),
          static_cast<int>(rows), static_cast<int>(h), static_cast<int>(dk), static_cast<int>(dv));
  });
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// Expand up to 16 rows by `expand_asked_kernel` with this ASK, or by the base at 0: the same bits.
void zs_glm5_mla_ask(int ask_steps) { asked = ask_steps; }

// lat bf16 [R, 512] (rows may be a wider buffer's columns) into cache bf16 [capacity, 512], or an FP8 cache u8
// [capacity, 528] as its records, at rows pos .. pos + R - 1; pos int32 [1], read on the device, with pos + R <=
// context <= capacity the caller's word for it.
int zs_glm5_mla_write(const zs_view* lat_, const zs_view* cache_, const zs_view* pos_, const zs_view* table_,
                      int64_t context) {
  zs::begin();
  const zs::View lat(lat_), cache(cache_), pos(pos_);
  ZS_REQUIRE(lat.defined() && lat.dtype() == ZS_BF16 && lat.dim() == 2 && lat.size(1) == LW && lat.size(0) >= 1 &&
             lat.size(0) <= INT32_MAX / LW, "lat: bf16 [R, 512]");
  const bool fp8 = cache.defined() && cache.dtype() == ZS_U8;
  ZS_REQUIRE(cache.defined() && cache.dim() == 2 && cache.is_contiguous() &&
             ((cache.dtype() == ZS_BF16 && cache.size(1) == LW) || (fp8 && cache.size(1) == REC)),
             "cache: bf16 [capacity, 512] or u8 [capacity, 528]");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(context >= lat.size(0) && context <= cache.size(0), "the cache holds pos + R rows");
  const int* table = nullptr;
  if (table_) {
    const zs::View view(table_);
    if (view.defined()) {
      ZS_REQUIRE(view.dtype() == ZS_I32 && view.is_contiguous(), "table: int32 [extents]");
      table = view.data_ptr<int>();
    }
  }
  if (fp8) {
    write_fp8_kernel<<<static_cast<unsigned>(lat.size(0)), LW / 4, 0, zs::stream()>>>(
        lat.data_ptr<__nv_bfloat16>(), lat.row_stride(), cache.data_ptr<unsigned char>(), pos.data_ptr<int>(), table);
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
  }
  write_kernel<<<blocks(lat.size(0) * LW), 256, 0, zs::stream()>>>(
      lat.data_ptr<__nv_bfloat16>(), lat.row_stride(), cache.data_ptr<__nv_bfloat16>(), pos.data_ptr<int>(), table,
      static_cast<int>(lat.size(0)));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
