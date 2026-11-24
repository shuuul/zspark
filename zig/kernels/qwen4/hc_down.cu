// Flash Next's hyper-connection read-out at a prompt window's rows (spec 148 WS-03): the norm, the down projection
// and the activation as one launch. The streams h are read as they are and normed in the tile's shared memory, so
// the normed rows never reach memory; a block walks the whole K and folds the product's K slices in registers, so
// the fp32 product is never written either. hc.cu's `normed_kernel` and `act_kernel` and bf16/gemm.cu stay as they
// are; this gives their bits, and zig/tests/models/qwen4/blocks/hc_fused_test.zig holds every output byte to them.
// With `zs_qwen4_hc_up_mix_streams` (hc_window.cu, the norm formed again in the up product's epilogue) a read-out
// is three launches where it was five: the write-back without its norm, this, the up and mix.
//
// The bits:
// - the norm, `normed_kernel`'s: rinv = 1 / sqrt(sum / D + eps), the stream's block sums of squares (pss) added
//   from 0 in block order, each value bf16((h * rinv) * scale), IEEE division and square root, no contraction;
// - the product, `zs_b16_gemm`'s: each element's K slices (K / SK values, SK by the weight's shape, `bf16.splitK`)
//   each a k16 mma chain from zero in K order, as CUTLASS's Sm80 GEMM and the hand matmul it is held to
//   (zig/tests/kernels/bf16/matmul.cu); the slices added from slice 0 in slice order, slice 0 taken as it is:
//   `b16_slices`' sum, and CUTLASS's serial split-K's (fma(1, acc_z, D)) from `bf16.sums_from` rows;
// - the activation, `act_kernel`'s on that sum: v = bf16(sum / S), act bf16(silu(v)), the gates bf16(2 sigmoid(v)).
// The tile a row lies in decides which block computes it, never its arithmetic: a row's bits are its own.
//
// The form: block (column tile, row tile) multiplies 64 rows by 112 columns over the whole K in stages of 32
// values, h, the weight and the stage's scale values copied by cp.async; each stage's h is normed in place between
// its arrival and the first ldmatrix (a stage lies in one stream: one 1 / rms a row, all four computed at the
// block's start). A slice's sums run in `acc` from zero and are added to `tot` at its last stage. Mia's AI Lab's
// recipe forms the norm in the down tile's shared memory too (Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold, MIT,
// clone fd3f43e1 of 2026-10-27, patch 0014, flashnext_hcf.cu `hcn::down_body`). Measured at Flash Next's shapes,
// 2,048 rows (nsys, spark-a and spark-b, spec 148 WS-03), against `normed` 405 us, CUTLASS's GEMM 316 and `act` 10:
// - their tile (16 MT rows by half the columns, 7 warps, the weight's fragments from L2 straight into registers):
//   887 to 926 us;
// - CUTLASS's serial split-K form (128 by 112 a block, a block a slice, the sums through fp32 in L2 between
//   slices, a semaphore a tile): 900 us, 545 with the scale staged once a block; the sums' 170 MB in and out of
//   L2 and the semaphores' waits bound it;
// - this form: 387 us, 410 on 128-row blocks (one a SM). It takes about 250 us at any width to 1,024 rows (the
//   K walk's latency, few blocks), hence `hc.fused_rows`.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/glue.py (Triton), Apache-2.0.
//
// Shared memory a block (8 warps, 64 rows by 112 columns), 3 stages of 32 values of K:
//
//   region                                          a stage   what for
//   h, 64 rows by 32 of K, rows 80 B apart          5,120     the rows (A), normed in place; a row padded by 16 B
//   the weight, 112 rows by 32 of K, 80 B apart     8,960     so that ldmatrix's eight rows fall in eight bank groups
//   scale, 32 fp32                                  128       the stage's scale values
//   rinv, 4 streams by 64 fp32, after the stages    1,024     the block's rows' 1 / rms in each stream
//
// 43,648 B, two blocks an SM. What could be less: the padding (a swizzle, 2,816 B). 112 columns: 3 tiles hold 336
// of the 324 (the 128 of gemm.cu's tile pad to 384).

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/math.cuh"
#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;
using qmm_frag::cp16z;

constexpr int BN = 112;              // columns a block
constexpr int BK = 32;               // values of K a stage
constexpr int LD = BK + 8;           // a staged row's elements: 80 bytes
constexpr int WARPS = 8;             // 4 by rows (16 MI each) and 2 by columns (56 each)
constexpr int THREADS = WARPS * 32;
constexpr int NT = BN / 2 / 8;       // a warp's n8 tiles: 7
constexpr int STREAMS_MAX = 4;

template <int MI>
struct Tile {
  static constexpr int BM = 64 * MI;               // rows a block
  static constexpr int STAGES = MI == 1 ? 3 : 4;
  static constexpr int STAGE = (BM + BN) * LD;     // bf16 elements
  static constexpr int STAGE_BYTES = STAGE * 2 + BK * 4;  // and the stage's 32 scale values
  static constexpr int SMEM = STAGES * STAGE_BYTES + STREAMS_MAX * BM * 4;
};

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

using qmm_frag::ldmatrix4;
using qmm_frag::ldmatrix2;

// `normed_kernel`'s 1 / rms of stream s of row r: the block sums in order from 0.
__device__ __forceinline__ float rinv_of(const float* __restrict__ pss, long long r, int s, int NC, int S, int D,
                                         float eps) {
  float total = 0.f;
  for (int c = 0; c < NC; ++c) total = __fadd_rn(total, pss[(r * NC + c) * S + s]);
  return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(D)), eps)));
}

// Stage of K values k .. k + BK: h's rows, the weight's, the scale's 32 values; 16-byte units. A row past M or N
// is zeros (computed, never stored).
template <int MI>
__device__ __forceinline__ void fetch(unsigned char* to, const __nv_bfloat16* __restrict__ H,
                                      const __nv_bfloat16* __restrict__ W, const float* __restrict__ SCALE, int M,
                                      int N, int K, int row0, int n0, int k) {
  constexpr int BM = Tile<MI>::BM;
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(to);
  for (int u = threadIdx.x; u < (BM + BN) * (BK / 8) + BK / 4; u += THREADS) {
    if (u >= (BM + BN) * (BK / 8)) {
      const int q = u - (BM + BN) * (BK / 8);
      cp16z(to + Tile<MI>::STAGE * 2 + q * 16, SCALE + k + q * 4, true);
      continue;
    }
    const int line = u / (BK / 8), at = (u % (BK / 8)) * 8;
    if (line < BM) {
      const int r = row0 + line;
      cp16z(st + line * LD + at, H + (size_t)min(r, M - 1) * K + k + at, r < M);
    } else {
      const int n = n0 + line - BM;
      cp16z(st + line * LD + at, W + (size_t)min(n, N - 1) * K + k + at, n < N);
    }
  }
}

// Block (column tile, row tile) over the whole K. Warp (wm, wn) holds rows wm 16 MI .. + 16 MI and columns
// wn 56 .. + 56 of the block; lane (gq, t) holds, for m tile i and n tile j, rows gq and gq + 8 at columns 2 t and
// 2 t + 1: acc[i][j] for the slice in hand, tot[i][j] the slices before it summed.
template <int MI>
__global__ void __launch_bounds__(THREADS)
    down_window_kernel(const __nv_bfloat16* __restrict__ H, const float* __restrict__ PSS,
                       const float* __restrict__ SCALE, const __nv_bfloat16* __restrict__ W,
                       __nv_bfloat16* __restrict__ ACT, __nv_bfloat16* __restrict__ INJ, int M, int N, int K, int SK,
                       int D, int S, int low, float eps) {
  using T = Tile<MI>;
  constexpr int BM = T::BM, STAGES = T::STAGES;
  extern __shared__ __align__(16) unsigned char shared[];
  float* rinv_s = reinterpret_cast<float*>(shared + STAGES * T::STAGE_BYTES);  // [S][BM]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gq = lane >> 2, t = lane & 3;
  const int wm = warp >> 1, wn = warp & 1;
  const int n0 = blockIdx.x * BN, row0 = blockIdx.y * BM;
  const int KT = K / BK, slice_stages = K / SK / BK;

#pragma unroll
  for (int p = 0; p < STAGES - 1; ++p) {
    if (p < KT) fetch<MI>(shared + p * T::STAGE_BYTES, H, W, SCALE, M, N, K, row0, n0, p * BK);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  for (int i = threadIdx.x; i < S * BM; i += THREADS)
    rinv_s[i] = rinv_of(PSS, min(row0 + i % BM, M - 1), i / BM, D / 256, S, D, eps);

  float acc[MI][NT][4], tot[MI][NT][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = tot[i][j][e] = 0.f;

  for (int kt = 0; kt < KT; ++kt) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
    __syncthreads();  // stage kt landed for all; stage kt - 1's reads are over
    unsigned char* stage = shared + (kt % STAGES) * T::STAGE_BYTES;
    __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(stage);
    const __nv_bfloat16* B = A + BM * LD;
    const float* scale_s = reinterpret_cast<const float*>(stage + T::STAGE * 2);
    const float* rv_s = rinv_s + ((kt * BK) / D) * BM;
    // normed_kernel's value of each staged h, in place: BM rows by four units of eight
#pragma unroll
    for (int q = 0; q < BM * (BK / 8) / THREADS; ++q) {
      const int u = threadIdx.x + q * THREADS, line = u / (BK / 8), at = (u % (BK / 8)) * 8;
      uint4* cell = reinterpret_cast<uint4*>(A + line * LD + at);
      const uint4 raw = *cell;
      const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&raw);
      const float rv = rv_s[line];
      const float4 sa = *reinterpret_cast<const float4*>(scale_s + at);
      const float4 sb = *reinterpret_cast<const float4*>(scale_s + at + 4);
      const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
      __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        o8[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(hv[i]), rv), sv[i]));
      *cell = *reinterpret_cast<const uint4*>(o8);
    }
    __syncthreads();
    if (kt + STAGES - 1 < KT)
      fetch<MI>(shared + ((kt + STAGES - 1) % STAGES) * T::STAGE_BYTES, H, W, SCALE, M, N, K, row0, n0,
                (kt + STAGES - 1) * BK);
    asm volatile("cp.async.commit_group;\n" ::);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t a[MI][4];
#pragma unroll
      for (int i = 0; i < MI; ++i)
        ldmatrix4(a[i], A + (wm * 16 * MI + i * 16 + (lane & 15)) * LD + kk + (lane >> 4) * 8);
      // n tiles 2 p and 2 p + 1 from one x4: lanes 0-15 the first, 16-31 the second; the seventh from an x2
#pragma unroll
      for (int p = 0; p < NT / 2; ++p) {
        uint32_t b[4];
        ldmatrix4(b, B + (wn * 56 + p * 16 + (lane >> 4) * 8 + (lane & 7)) * LD + kk + ((lane >> 3) & 1) * 8);
#pragma unroll
        for (int i = 0; i < MI; ++i) {
          mma(acc[i][2 * p], a[i], b[0], b[1]);
          mma(acc[i][2 * p + 1], a[i], b[2], b[3]);
        }
      }
      {
        uint32_t b[2];
        ldmatrix2(b, B + (wn * 56 + (NT - 1) * 8 + (lane & 7)) * LD + kk + ((lane >> 3) & 1) * 8);
#pragma unroll
        for (int i = 0; i < MI; ++i) mma(acc[i][NT - 1], a[i], b[0], b[1]);
      }
    }
    if ((kt + 1) % slice_stages == 0) {
      // a K slice is over: the GEMM's sum of slices, slice 0 as it is, then each added in order
      const bool first = kt + 1 == slice_stages;
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            tot[i][j][e] = first ? acc[i][j][e] : __fadd_rn(tot[i][j][e], acc[i][j][e]);
            acc[i][j][e] = 0.f;
          }
    }
  }

#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = row0 + wm * 16 * MI + i * 16 + gq + 8 * (e >> 1), c = n0 + wn * 56 + j * 8 + 2 * t + (e & 1);
        if (r >= M || c >= N) continue;
        // act_kernel's arithmetic on the summed element
        const float x = bf(__fdiv_rn(tot[i][j][e], static_cast<float>(S)));
        if (c < low) ACT[(size_t)r * low + c] = __float2bfloat16_rn(bsilu(x));
        else if (INJ != nullptr) INJ[(size_t)r * S + (c - low)] = __float2bfloat16_rn(2.0f * bsig(x));
      }
}

template <int MI>
int launch(const zs::View& h, const zs::View& pss, const zs::View& scale, const zs::View& w, const zs::View& act,
           const zs::View& inject, int64_t rows, int64_t n, int64_t wide, int64_t sk, int64_t dims, int64_t streams,
           int64_t low, double eps) {
  using T = Tile<MI>;
  static bool ready = false;
  if (!ready) {
    ZS_CUDA_CHECK(cudaFuncSetAttribute(down_window_kernel<MI>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
    ready = true;
  }
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>((rows + T::BM - 1) / T::BM));
  down_window_kernel<MI><<<grid, THREADS, T::SMEM, zs::stream()>>>(
      h.data_ptr<__nv_bfloat16>(), pss.data_ptr<float>(), scale.data_ptr<float>(), w.data_ptr<__nv_bfloat16>(),
      act.data_ptr<__nv_bfloat16>(), n == low ? nullptr : inject.data_ptr<__nv_bfloat16>(), static_cast<int>(rows),
      static_cast<int>(n), static_cast<int>(wide), static_cast<int>(sk), static_cast<int>(dims),
      static_cast<int>(streams), static_cast<int>(low), static_cast<float>(eps));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // namespace

extern "C" {

// h bf16 [R, S * D], pss fp32 [R, D / 256, S] (the write-back's), scale fp32 [S * D], w bf16 [N, S * D] (the down
// projection, then the inject gates' rows where it has them; N = low or low + S, to 336), in ``sk`` K slices of
// whole stages -> act bf16 [R, low] and, where N = low + S, inject bf16 [R, S]: `zs_qwen4_hc_normed`, `zs_b16_gemm`
// and `zs_qwen4_hc_act` in one launch, their bits.
int zs_qwen4_hc_down_window(const zs_view* h_, const zs_view* pss_, const zs_view* scale_, const zs_view* w_,
                            const zs_view* act_, const zs_view* inject_, int64_t streams, int64_t low, int64_t sk,
                            double eps) {
  zs::begin();
  const zs::View h(h_), pss(pss_), scale(scale_), w(w_), act(act_), inject(inject_);
  ZS_REQUIRE(h.defined() && h.dtype() == ZS_BF16 && h.dim() == 2 && h.is_contiguous() && streams >= 1 &&
                 streams <= STREAMS_MAX && h.size(1) % streams == 0 && (h.size(1) / streams) % 256 == 0,
             "h: bf16 [R, S * D], S to 4, D in blocks of 256");
  const int64_t rows = h.size(0), wide = h.size(1), dims = wide / streams;
  ZS_REQUIRE(rows >= 1 && rows <= INT32_MAX / wide, "rows: 1 and more, the shape within int32");
  ZS_REQUIRE(pss.defined() && pss.dtype() == ZS_F32 && pss.numel() >= rows * (dims / 256) * streams,
             "pss: fp32 [R, D / 256, S]");
  ZS_REQUIRE(scale.defined() && scale.dtype() == ZS_F32 && scale.numel() == wide &&
                 reinterpret_cast<uintptr_t>(scale.data_ptr()) % 16 == 0, "scale: fp32 [S * D] from 16 bytes");
  ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.dim() == 2 && w.is_contiguous() && w.size(1) == wide &&
                 reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(h.data_ptr()) % 16 == 0, "w: bf16 [N, S * D]; w and h from 16 bytes");
  const int64_t n = w.size(0);
  ZS_REQUIRE(n >= 1 && n <= 3 * BN && (n == low || n == low + streams), "N: low or low + S, to 336");
  ZS_REQUIRE(sk >= 1 && wide % sk == 0 && (wide / sk) % BK == 0, "K slices of whole stages");
  ZS_REQUIRE(act.defined() && act.dtype() == ZS_BF16 && act.is_contiguous() && act.numel() >= rows * low,
             "act: bf16 [R, low]");
  ZS_REQUIRE(n == low || (inject.defined() && inject.dtype() == ZS_BF16 && inject.numel() >= rows * streams),
             "inject: bf16 [R, S]");
  ZS_REQUIRE((rows + 63) / 64 < 65536, "rows within the grid");
  // 64-row blocks, two a SM: 387 us at 2,048 rows where 128-row blocks, one a SM, took 410 (spec 148 WS-03)
  return launch<1>(h, pss, scale, w, act, inject, rows, n, wide, sk, dims, streams, low, eps);
}

}  // extern "C"
