// Flash Next's hyper-connection read-out at a prompt window's rows (spec 095 WS-03, spec 087's candidate 4): the up
// projection and `mix` as one launch, the up product never written. hc.cu's `mix_kernel` and bf16/gemm.cu stay as
// they are; this gives their bits, and zig/tests/models/qwen4/blocks/hc_window_test.zig holds it to them.
//
// Why not CUTLASS's epilogue: `mix` sums a row's four streams at a column d, and they are the product's columns
// d, D + d, 2 D + d and 3 D + d, which a CUTLASS tile of contiguous columns never holds together. Here a block's
// 128 product columns are 32 values of d in each of the four streams, and a warp's eight n8 tiles are its 16
// values of d in each stream, so every thread holds a (row, d)'s four products in registers and mixes them there.
// Why not a decoding round's form (16 rows a block, a warp a stream meeting in shared memory, the weight read
// again for every 16 rows; spec 070's fused read-out, withdrawn in spec 121): 1.48 ms at 2,048 rows where the GEMM and `mix` take
// 0.34 and 0.47 (ncu, spec 095 WS-03).
//
// The bits: an element's product is the k16 mma chain from zero over K in order, as CUTLASS's Sm80 GEMM on every
// tile (its warp's k iterations in order inside each stage, the stages in order), then bf16 by round to nearest
// (LinearCombination with ScaleType::Nothing); then hc.cu's `mix_kernel` exactly: total += bf(bsig(up_s) *
// normed_s) for s in order from 0.f, bf16(total / S). The tile a row lies in decides which block computes it,
// never its arithmetic: a row's bits are its own.
//
// Shared memory a block (8 warps, 128 rows by 32 values of d), STAGES stages of 32 values of K:
//
//   region                                         a stage     what for
//   act, 128 rows by 32 of K, rows 80 B apart      10,240      the rows (A); a row padded by 16 B so that
//   the weight, 128 rows by 32 of K, 80 B apart    10,240      ldmatrix's eight rows fall in eight bank groups
//
// 2 stages are 40,960 B, two blocks an SM; 3 are 61,440, one. Nothing else is staged: `normed` (8 bytes a value of
// d, read once) goes from global memory to registers in the epilogue, asked into L2 by a prefetch when the block
// starts, and the products stay in registers. What could be less: the 16 B of padding a row (a swizzle instead,
// 6,144 B of a 3-stage block). The streams' form (`zs_qwen4_hc_up_mix_streams`, spec 148 WS-03) reads h where
// this reads `normed` and adds its rows' 1 / rms after the stages, 128 rows by 4 streams fp32, 2,048 B: 43,008 B
// on 2 stages, still two blocks an SM.
//
// ncu at Flash Next's shapes (2,048, 1,024, 512, 256, 128, 64 rows; us a launch against the GEMM and `mix`): 465 /
// 795, 280 / 431, 173 / 253, 127 / 154, 74 / 93, 76 / 66; 3 stages 499 at 2,048. Without the prefetch 500 at
// 2,048 rows, half the cycles waiting on `normed`; with `normed` held in registers from the start, 606.
// `hc.window_rows` (128) is the gate.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/glue.py (Triton), Apache-2.0; this fused form
// is zspark's own.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/copies.cuh"
#include "../common/math.cuh"
#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using namespace common_copies;

using namespace common_math;

constexpr int BM = 128;          // rows a block
constexpr int BD = 32;           // values of d a block, in each of the four streams
constexpr int S4 = 4;            // streams
constexpr int BN = BD * S4;      // product columns a block
constexpr int BK = 32;           // values of K a stage
constexpr int LD = BK + 8;       // a staged row's elements: 80 bytes
constexpr int WARPS = 8;         // 4 by rows (32 each) and 2 by d (16 each)
constexpr int THREADS = WARPS * 32;

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

using qmm_frag::ldmatrix4;

// Stage `kt`'s act rows and weight rows into `to` (A's 128 rows, then the weight's): 16-byte units, four a row,
// four a thread of each. A row past M is not copied (its products are stored nowhere).
template <class El>
__device__ __forceinline__ void fetch(El* to, const El* __restrict__ X, const El* __restrict__ W, int M, int D, int K,
                                      int ldx, int row0, int d0, int kt) {
#pragma unroll
  for (int i = 0; i < 2 * BM * (BK / 8) / THREADS; ++i) {
    const int u = threadIdx.x + i * THREADS, line = u / (BK / 8), at = (u % (BK / 8)) * 8;
    if (line < BM) {
      if (row0 + line < M) cp16(to + line * LD + at, X + (size_t)(row0 + line) * ldx + kt * BK + at);
    } else {
      const int b = line - BM, s = b / BD, j = b % BD;
      cp16(to + line * LD + at, W + (size_t)(s * D + d0 + j) * K + kt * BK + at);
    }
  }
}

// Block (d tile, row tile). Warp (wm, wn) holds rows wm 32 .. + 32 and d values wn 16 .. + 16 of the block; its n8
// tile j = 2 s + h is stream s's d values h 8 .. + 8 of those. Lane (gq, t) holds, for m tile i and n tile j, rows
// gq and gq + 8 at columns 2 t and 2 t + 1: acc[i][j][2 e + c].
// With STREAMS, `normed` is the streams h themselves and each normed value is formed in the epilogue as
// hc.cu's `normed_kernel` forms it (hc_down.cu's `rinv_of`, bf16((h * rinv) * scale)), its rows' 1 / rms in shared
// memory after the stages: the normed rows are never read from memory, nor written by anyone (spec 148 WS-03).
__device__ __forceinline__ float rinv_of(const float* __restrict__ pss, long long r, int s, int NC, int S, int D,
                                         float eps) {
  float total = 0.f;
  for (int c = 0; c < NC; ++c) total = __fadd_rn(total, pss[(r * NC + c) * S + s]);
  return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(total, static_cast<float>(D)), eps)));
}

template <class El, int STAGES, bool STREAMS>
__global__ void __launch_bounds__(THREADS)
    up_mix_window_kernel(const El* __restrict__ X, const El* __restrict__ W, const __nv_bfloat16* __restrict__ normed,
                         __nv_bfloat16* __restrict__ mixed, int M, int D, int K, int ldx, const float* __restrict__ pss,
                         const float* __restrict__ scale, float eps) {
  extern __shared__ __align__(16) unsigned char shared[];
  El* stages = reinterpret_cast<El*>(shared);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gq = lane >> 2, t = lane & 3;
  const int wm = warp >> 1, wn = warp & 1;
  const int d0 = blockIdx.x * BD, row0 = blockIdx.y * BM, KT = K / BK;
  float* rinv_s = reinterpret_cast<float*>(shared + STAGES * 2 * BM * LD * sizeof(El));
  if constexpr (STREAMS) {
    // read after the main loop's barriers
    for (int u = threadIdx.x; u < BM * S4; u += THREADS)
      rinv_s[u] = rinv_of(pss, min(row0 + u / S4, M - 1), u % S4, D / 256, S4, D, eps);
  }

  // the epilogue's normed lines (64 bytes a row and stream) asked into L2 before the products, so that they arrive
  // while the products run: read after them from memory, a launch waited on them for half its cycles (ncu, 2,048
  // rows); held in registers from here instead, the launch was slower still
  for (int u = threadIdx.x; u < BM * S4; u += THREADS) {
    const int r = row0 + u / S4;
    if (r < M) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(normed + ((size_t)r * S4 + u % S4) * D + d0));
  }

  float acc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) fetch<El>(stages + s * 2 * BM * LD, X, W, M, D, K, ldx, row0, d0, s);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  for (int kt = 0; kt < KT; ++kt) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < KT) fetch<El>(stages + (next % STAGES) * 2 * BM * LD, X, W, M, D, K, ldx, row0, d0, next);
    asm volatile("cp.async.commit_group;\n" ::);
    const El* A = stages + (kt % STAGES) * 2 * BM * LD;
    const El* B = A + BM * LD;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t a[2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i) ldmatrix4(a[i], A + (wm * 32 + i * 16 + (lane & 15)) * LD + kk + (lane >> 4) * 8);
      // tiles j and j + 1 (a stream's two halves) from one x4: lanes 0-15 the first, 16-31 the second
#pragma unroll
      for (int s = 0; s < S4; ++s) {
        uint32_t b[4];
        ldmatrix4(b, B + (s * BD + wn * 16 + (lane >> 4) * 8 + (lane & 7)) * LD + kk + ((lane >> 3) & 1) * 8);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          mma(acc[i][2 * s], a[i], b[0], b[1]);
          mma(acc[i][2 * s + 1], a[i], b[2], b[3]);
        }
      }
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::);

  // mix_kernel's arithmetic on the four streams' products a thread holds
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int r = row0 + wm * 32 + i * 16 + gq + 8 * e;
      if (r >= M) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int d = d0 + wn * 16 + h * 8 + 2 * t;
        __nv_bfloat162 n[S4];
#pragma unroll
        for (int s = 0; s < S4; ++s)
          n[s] = __ldg(reinterpret_cast<const __nv_bfloat162*>(normed + ((size_t)r * S4 + s) * D + d));
        if constexpr (STREAMS) {
#pragma unroll
          for (int s = 0; s < S4; ++s) {
            const float rv = rinv_s[(r - row0) * S4 + s];
            const float2 sc = __ldg(reinterpret_cast<const float2*>(scale + s * D + d));
            n[s] = __halves2bfloat162(__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(n[s].x), rv), sc.x)),
                                      __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(n[s].y), rv), sc.y)));
          }
        }
        float total[2] = {0.f, 0.f};
#pragma unroll
        for (int s = 0; s < S4; ++s) {
          total[0] += bf(bsig(bf(acc[i][2 * s + h][2 * e])) * __bfloat162float(n[s].x));
          total[1] += bf(bsig(bf(acc[i][2 * s + h][2 * e + 1])) * __bfloat162float(n[s].y));
        }
        *reinterpret_cast<__nv_bfloat162*>(mixed + (size_t)r * D + d) =
            __halves2bfloat162(__float2bfloat16_rn(total[0] / S4), __float2bfloat16_rn(total[1] / S4));
      }
    }
}

template <class El, int STAGES, bool STREAMS = false>
int launch(const zs::View& act, const zs::View& up, const zs::View& normed, const zs::View& mixed,
           int64_t rows, int64_t d, int64_t k, const float* pss = nullptr, const float* scale = nullptr,
           float eps = 0.f) {
  constexpr int bytes = STAGES * 2 * BM * LD * static_cast<int>(sizeof(El)) + (STREAMS ? BM * S4 * 4 : 0);
  static bool ready = false;
  if (!ready) {
    ZS_CUDA_CHECK(cudaFuncSetAttribute(up_mix_window_kernel<El, STAGES, STREAMS>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    ready = true;
  }
  const dim3 grid(static_cast<unsigned>(d / BD), static_cast<unsigned>((rows + BM - 1) / BM));
  up_mix_window_kernel<El, STAGES, STREAMS><<<grid, THREADS, bytes, zs::stream()>>>(
      act.data_ptr<El>(), up.data_ptr<El>(), normed.data_ptr<__nv_bfloat16>(), mixed.data_ptr<__nv_bfloat16>(),
      static_cast<int>(rows), static_cast<int>(d), static_cast<int>(k), static_cast<int>(act.stride(0)), pss, scale,
      eps);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // namespace

extern "C" {

// act [R, low] and up [S * D, low] (one K slice), both bf16, normed bf16 [R, S * D] -> mixed bf16 [R, D]:
// zs_b16_gemm and zs_qwen4_hc_mix in one launch, their bits. S is 4, D a multiple of 32, low of 32.
// ``stages`` 2 or 3 (0: 2), the same bits.
int zs_qwen4_hc_up_mix_window(const zs_view* act_, const zs_view* up_, const zs_view* normed_, const zs_view* mixed_,
                              int64_t streams, int64_t stages) {
  zs::begin();
  const zs::View act(act_), up(up_), normed(normed_), mixed(mixed_);
  ZS_REQUIRE(up.defined() && up.dtype() == ZS_BF16 && up.dim() == 2 && up.is_contiguous(), "up: [S * D, K] bf16");
  const int64_t wide = up.size(0), k = up.size(1);
  ZS_REQUIRE(streams == S4 && wide % (S4 * BD) == 0 && k % BK == 0, "4 streams, D in tiles of 32, K of 32");
  const int64_t d = wide / streams;
  ZS_REQUIRE(act.defined() && act.dtype() == ZS_BF16 && act.dim() == 2 && act.size(1) == k && act.size(0) >= 1 &&
             reinterpret_cast<uintptr_t>(act.data_ptr()) % 16 == 0 && act.stride(0) % 8 == 0 &&
             reinterpret_cast<uintptr_t>(up.data_ptr()) % 16 == 0,
             "act: [R, K] bf16, rows on 16 bytes");
  const int64_t rows = act.size(0);
  ZS_REQUIRE(rows <= INT32_MAX / wide && rows / BM < 65535, "shape within int32 and the grid");
  ZS_REQUIRE(normed.defined() && normed.dtype() == ZS_BF16 && normed.is_contiguous() && normed.numel() >= rows * wide &&
             reinterpret_cast<uintptr_t>(normed.data_ptr()) % 4 == 0, "normed: bf16 [R, S * D]");
  ZS_REQUIRE(mixed.defined() && mixed.dtype() == ZS_BF16 && mixed.is_contiguous() && mixed.numel() >= rows * d &&
             reinterpret_cast<uintptr_t>(mixed.data_ptr()) % 4 == 0, "mixed: bf16 [R, D]");
  ZS_REQUIRE(stages >= 0 && stages <= 3 && stages != 1, "stages: 0, 2 or 3");
  return stages == 3 ? launch<__nv_bfloat16, 3>(act, up, normed, mixed, rows, d, k)
                     : launch<__nv_bfloat16, 2>(act, up, normed, mixed, rows, d, k);
}

// `zs_qwen4_hc_up_mix_window` with the streams h bf16 [R, S * D] in place of the normed rows, normed in the
// epilogue from pss fp32 [R, D / 256, S] (the write-back's) and scale fp32 [S * D] as `zs_qwen4_hc_normed` norms
// them: zs_qwen4_hc_normed, zs_b16_gemm and zs_qwen4_hc_mix in one launch, their bits, on 2 stages (spec 148 WS-03).
int zs_qwen4_hc_up_mix_streams(const zs_view* act_, const zs_view* up_, const zs_view* h_, const zs_view* pss_,
                               const zs_view* scale_, const zs_view* mixed_, int64_t streams, double eps) {
  zs::begin();
  const zs::View act(act_), up(up_), h(h_), pss(pss_), scale(scale_), mixed(mixed_);
  ZS_REQUIRE(up.defined() && up.dtype() == ZS_BF16 && up.dim() == 2 && up.is_contiguous(), "up: [S * D, K] bf16");
  const int64_t wide = up.size(0), k = up.size(1);
  ZS_REQUIRE(streams == S4 && wide % (S4 * BD) == 0 && k % BK == 0, "4 streams, D in tiles of 32, K of 32");
  const int64_t d = wide / streams;
  ZS_REQUIRE(d % 256 == 0, "D in blocks of 256");
  ZS_REQUIRE(act.defined() && act.dtype() == ZS_BF16 && act.dim() == 2 && act.size(1) == k && act.size(0) >= 1 &&
             reinterpret_cast<uintptr_t>(act.data_ptr()) % 16 == 0 && act.stride(0) % 8 == 0 &&
             reinterpret_cast<uintptr_t>(up.data_ptr()) % 16 == 0,
             "act: [R, K] bf16, rows on 16 bytes");
  const int64_t rows = act.size(0);
  ZS_REQUIRE(rows <= INT32_MAX / wide && rows / BM < 65535, "shape within int32 and the grid");
  ZS_REQUIRE(h.defined() && h.dtype() == ZS_BF16 && h.is_contiguous() && h.numel() >= rows * wide &&
             reinterpret_cast<uintptr_t>(h.data_ptr()) % 4 == 0, "h: bf16 [R, S * D]");
  ZS_REQUIRE(pss.defined() && pss.dtype() == ZS_F32 && pss.numel() >= rows * (d / 256) * streams,
             "pss: fp32 [R, D / 256, S]");
  ZS_REQUIRE(scale.defined() && scale.dtype() == ZS_F32 && scale.numel() == wide &&
             reinterpret_cast<uintptr_t>(scale.data_ptr()) % 8 == 0, "scale: fp32 [S * D]");
  ZS_REQUIRE(mixed.defined() && mixed.dtype() == ZS_BF16 && mixed.is_contiguous() && mixed.numel() >= rows * d &&
             reinterpret_cast<uintptr_t>(mixed.data_ptr()) % 4 == 0, "mixed: bf16 [R, D]");
  return launch<__nv_bfloat16, 2, true>(act, up, h, mixed, rows, d, k, pss.data_ptr<float>(),
                                        scale.data_ptr<float>(), static_cast<float>(eps));
}

}  // extern "C"
