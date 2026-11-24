// Flash Next's sparse key selection (QSA), decode path: the indexer's keys are pooled RATIO positions to a block,
// a row scores every complete block before it and attends the TOP best, in block order, followed by the positions
// of its unfinished block. A row with at most TOP complete blocks attends every position. A row's list comes from
// its own indexer queries and the pooled blocks before it: equal scores go to the lower block, so neither the
// window nor the bound of a launch changes it.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/attention.py (_pool, _pool_block, _scores,
// _select, _select_tiles), Triton, Apache-2.0; rewritten for zspark in CUDA C++. The
// query-by-block products are a dense GEMM and run on CUTLASS (bf16/gemm.cu, one K slice) from the Zig side; what
// is here is the pooling, the sum over indexer heads, and the selection, none of which is a GEMM. TensorFold has
// two selections that give the same lists (a bisection in registers, a radix select in tiles); this is the radix
// select, for any block count.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/math.cuh"
#include "../common/qmm_frag.cuh"
#include "../zs/host.h"
#include "attention_rope.cuh"

namespace {

using namespace common_math;

constexpr int WIDE = 256;      // the most values of an indexer key; a pool block's threads
constexpr int SEL = 1024;      // a selection block's threads

// ZS_EXTENTS is spec 018's layout, the pool's (`--parallel` N): qsa_table.cu defines it and
// includes this file, and without it this file is the kernels it always was (attention.cu says the same of
// itself, and what the layout is). With it the indexer's raw keys IKC are a pool of extents of 512 positions
// behind TABLE, and the selection writes a listed row's keys as the pool's rows, so that the attend kernel reads
// a list as it always did. A pooled block's RATIO positions lie in one extent (RATIO divides 512, which the
// launchers require): one load of the table a block pooled and one a block listed. The pooled keys themselves
// stay a sequence's own, in order: one GEMM scores them. Which blocks a row takes, and in which order, does not
// change: only what is written for a key does.
#ifdef ZS_EXTENTS
constexpr int EXTENT = 512, EXTENT_BITS = 9;
__device__ __forceinline__ int stored_at(const int* __restrict__ TABLE, int pos) {
  return (TABLE[pos >> EXTENT_BITS] << EXTENT_BITS) | (pos & (EXTENT - 1));
}
#define ZS_TABLE_PARAM const int* __restrict__ TABLE,
#else
#define ZS_TABLE_PARAM
#endif

// Block i of WIDE threads: pooled block b = POS0 / RATIO + i, if rows POS0 .. POS0 + R - 1 complete it (a block
// the window completes is pooled again from the same keys, to the same bits):
//   x = bf16((k_0 + ... + k_RATIO-1, in order, fp32) / RATIO); bf16(x / rms(x) * W), rms from a tree over the WIDE
//   threads; then the rotate-half RoPE of the block's first position on the first 2 HALF values, rounded to bf16:
//   under ROPE (an image prompt's, attention_rope.cuh; null in the table's object, a round's being qsa_rows.cu's)
//   that position's axis of three.
__global__ void __launch_bounds__(WIDE)
    pool_kernel(ZS_TABLE_PARAM const __nv_bfloat16* __restrict__ IKC, __nv_bfloat16* __restrict__ POOLED,
                const int* __restrict__ POS0, const float* __restrict__ W, const float* __restrict__ INV, float eps,
                int R, int DI, int HALF, int RATIO, const int* __restrict__ ROPE, int S1, int S2) {
  __shared__ float squares[WIDE];
  __shared__ float mean[WIDE];
  const int p0 = POS0[0], b = p0 / RATIO + blockIdx.x, d = threadIdx.x;
  if ((long long)RATIO * b + RATIO > (long long)p0 + R) return;
  const bool live = d < DI;
  float x = 0.f;
  if (live) {
#ifdef ZS_EXTENTS
    const __nv_bfloat16* at = IKC + (size_t)stored_at(TABLE, RATIO * b) * DI + d;   // the block's first row
#else
    const __nv_bfloat16* at = IKC + (size_t)RATIO * b * DI + d;
#endif
    float acc = __bfloat162float(at[0]);
    for (int k = 1; k < RATIO; ++k) acc = acc + __bfloat162float(at[(size_t)k * DI]);
    x = bf(acc / RATIO);
  }
  mean[d] = x;
  squares[d] = x * x;
  __syncthreads();
  for (int half = WIDE / 2; half > 0; half >>= 1) {
    if (d < half) squares[d] += squares[d + half];
    __syncthreads();
  }
  if (!live) return;
  const float rinv = 1.0f / sqrtf(squares[0] / DI + eps);
  const float xn = bf(x * rinv * W[d]);
  float out = xn;
  if (d < 2 * HALF) {
    const int partner = d < HALF ? d + HALF : d - HALF, i = d < HALF ? d : d - HALF;
    const float xpn = bf(mean[partner] * rinv * W[partner]);
    const float ang = static_cast<float>(rope_axis(ROPE, RATIO * b, i, S1, S2)) * INV[i];
    const float c = cosf(ang), s = sinf(ang);
    out = d < HALF ? xn * c - xpn * s : xpn * s + xn * c;
  }
  POOLED[(size_t)b * DI + d] = __float2bfloat16_rn(out);
}

// Thread (row, block): the block's score for a sparse row, the sum over the indexer heads in order of
// relu(q_h . k) over sqrt(DI). DOTS [R * HI, NBL] holds the products; SC [R, NB].
__global__ void scores_kernel(const float* __restrict__ DOTS, const int* __restrict__ POS0, float* __restrict__ SC,
                              int R, int HI, int NBL, int NB, int DI, int RATIO, int TOP) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * NBL) return;
  const int r = static_cast<int>(i / NBL), b = static_cast<int>(i % NBL);
  const int complete = (POS0[0] + r + 1) / RATIO;
  if (complete <= TOP || b >= complete) return;
  float total = 0.f;
  for (int h = 0; h < HI; ++h) total = total + fmaxf(DOTS[((size_t)r * HI + h) * NBL + b], 0.f);
  SC[(size_t)r * NB + b] = total / sqrtf(static_cast<float>(DI));
}

// A score's bits as an unsigned key that orders as the scores do.
__device__ __forceinline__ uint32_t order_key(float v) {
  const uint32_t bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// The block's inclusive prefix of the value each thread holds, every thread taking part: a warp's lanes by
// shuffle, the 32 warps' totals over warp 0, each warp's offset added on. Integer sums, so the result is what
// the counts sum to and not what order the threads ran in; scratch holds the warps' totals, then their offsets.
template <typename T>
__device__ __forceinline__ T block_inclusive(T v, T* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const T up = __shfl_up_sync(0xffffffffu, v, off);
    if (lane >= off) v += up;
  }
  if (lane == 31) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    const T own = scratch[lane];
    T t = own;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const T up = __shfl_up_sync(0xffffffffu, t, off);
      if (lane >= off) t += up;
    }
    scratch[lane] = t - own;
  }
  __syncthreads();
  const T base = scratch[warp];
  __syncthreads();
  return v + base;
}

// The largest value a thread of the block holds, in scratch. A shuffle down leaves the warp's value in its
// lane 0; the block's is lane 0 of warp 0's.
__device__ __forceinline__ int block_maximum(int v, int* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const int up = __shfl_down_sync(0xffffffffu, v, off);
    v = max(v, up);
  }
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = scratch[lane];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const int up = __shfl_down_sync(0xffffffffu, v, off);
      v = max(v, up);
    }
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
}

// Block r of SEL threads. A row with at most TOP complete blocks keeps its length in NK and SPARSE 0. Otherwise
// its list IDS[r] is the positions of its TOP blocks by score, the lower block among equal scores, in block order,
// then the positions of its unfinished block; NK its length, SPARSE 1.
// The cut (the TOP-th largest key) comes from a radix select, a byte a pass from the top, with exact integer
// counts. Then each thread counts the keys above and at the cut in its run of consecutive blocks, a scan over the
// block gives each run its place, and the run writes its chosen blocks: every key above the cut, and the first
// ``need`` at it. The scans (the digit's histogram summed up, the runs' prefix) are over integers, so the digit
// and each run's place are the one-thread fold's: the digit is the largest whose suffix holds ``need`` (a suffix
// only grows toward the lower digits, the one the scan from 255 down stopped at), and a run's place is the count
// of the chosen keys before it, whatever thread counted them.
// With ZS_EXTENTS a key is written as the pool's row of its position, a block's RATIO keys from one load of TABLE.
__global__ void __launch_bounds__(SEL)
    select_kernel(ZS_TABLE_PARAM const float* __restrict__ SC, const int* __restrict__ POS0, int* __restrict__ IDS,
                  int* __restrict__ NK, int* __restrict__ SPARSE, int NB, int RATIO, int TOP, int IDW) {
  __shared__ int hist[256];
  __shared__ int scan[256];        // the histogram's inclusive sums, and ``total`` at 255
  __shared__ int scratch[32];
  __shared__ long long sums[32];   // the runs' counts, above and equal packed in one
  __shared__ uint32_t cut_shared;
  __shared__ int need_shared;      // the cut's rank among the keys sharing its leading bytes
  const int r = blockIdx.x, tid = threadIdx.x;
  const int end = POS0[0] + r + 1, complete = end / RATIO;
  if (complete <= TOP) {
    if (tid == 0) {
      NK[r] = end;
      SPARSE[r] = 0;
    }
    return;
  }
  const float* row = SC + (size_t)r * NB;
  if (tid == 0) {
    cut_shared = 0;
    need_shared = TOP;
  }
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    const uint32_t lead = cut_shared;
    for (int b = tid; b < complete; b += SEL) {
      const uint32_t key = order_key(row[b]);
      if (pass > 0 && (key >> (shift + 8)) != (lead >> (shift + 8))) continue;
      atomicAdd(&hist[(key >> shift) & 0xFF], 1);
    }
    __syncthreads();
    // the cut's digit: the histogram summed up over the block, then the largest digit whose suffix holds
    // ``need``; the keys above it are the whole less the digit's inclusive sum
    const int h = tid < 256 ? hist[tid] : 0;
    const int incl = block_inclusive(h, scratch);
    if (tid < 256) scan[tid] = incl;
    __syncthreads();
    const int total = scan[255];
    int best = 0;
    if (tid < 256 && total - incl + h >= need_shared) best = tid;
    const int digit = block_maximum(best, scratch);
    if (tid == 0) {
      need_shared -= total - scan[digit];
      cut_shared = lead | (static_cast<uint32_t>(digit) << shift);
    }
    __syncthreads();
  }
  const uint32_t cut = cut_shared;
  const int need = need_shared;
  const int per = (complete + SEL - 1) / SEL;
  const int b0 = min(tid * per, complete), b1 = min(b0 + per, complete);
  int na = 0, ne = 0;
  for (int b = b0; b < b1; ++b) {
    const uint32_t key = order_key(row[b]);
    na += key > cut;
    ne += key == cut;
  }
  // each run's place: the counts packed one long long (neither half reaches 2^31, so no half carries into the
  // other), their exclusive prefix one scan
  const long long mine = ((long long)ne << 32) | (unsigned int)na;
  const long long before = block_inclusive(mine, sums) - mine;
  int* list = IDS + (size_t)r * IDW;
  int seen = (int)(before >> 32), placed = (int)before + min(seen, need);
  for (int b = b0; b < b1; ++b) {
    const uint32_t key = order_key(row[b]);
    bool chosen = key > cut;
    if (key == cut) chosen = ++seen <= need;
    if (chosen) {
#ifdef ZS_EXTENTS
      const int first = stored_at(TABLE, b * RATIO);
      for (int k = 0; k < RATIO; ++k) list[placed * RATIO + k] = first + k;
#else
      for (int k = 0; k < RATIO; ++k) list[placed * RATIO + k] = b * RATIO + k;
#endif
      ++placed;
    }
  }
  // the unfinished block's positions, one a thread
  if (tid < RATIO) {
    const int tail = RATIO * complete + tid;
#ifdef ZS_EXTENTS
    if (tail < end) list[TOP * RATIO + tid] = stored_at(TABLE, tail);
#else
    if (tail < end) list[TOP * RATIO + tid] = tail;
#endif
  }
  if (tid == 0) {
    NK[r] = TOP * RATIO + end - RATIO * complete;
    SPARSE[r] = 1;
  }
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

// A prompt window's scores in one launch (spec 095 WS-03, spec 087's candidate 8): what the GEMM of a tile of
// rows into DOTS and scores_kernel give, without DOTS. Block (x, y): pooled blocks 64 x .. 64 x + 63 for rows
// 64 y .. 64 y + 63, a warp 16 rows by the 64 blocks (eight n8 tiles). For each head in order, an element's
// product is the k16 mma chain from zero over DI in order, as CUTLASS's Sm80 GEMM of one K slice into fp32 gives
// it (bf16/gemm.cu, the selection's GEMM), and its relu is added to the row's total for the block; then the total
// over sqrt(DI), as scores_kernel. Only the entries scores_kernel writes are written.
// Shared memory: the 64 pooled keys, DI + 8 values a row (rows 16 bytes off a multiple of 32 words, so that
// ldmatrix's eight rows fall in eight bank groups): 17,408 bytes at DI 128, 33,792 at 256. The queries are read
// from global memory into the mma's registers, 8 bytes a value of a row a warp, each read by every block of x.
constexpr int SW_ROWS = 64, SW_BLOCKS = 64, SW_THREADS = 128;

__global__ void __launch_bounds__(SW_THREADS)
    scores_window_kernel(const __nv_bfloat16* __restrict__ IQ, const __nv_bfloat16* __restrict__ POOLED,
                         const int* __restrict__ POS0, float* __restrict__ SC, int R, int HI, int DI, int NBL, int NB,
                         int RATIO, int TOP) {
  extern __shared__ __align__(16) __nv_bfloat16 keys[];
  const int LD = DI + 8;
  const int b0 = blockIdx.x * SW_BLOCKS, r0 = blockIdx.y * SW_ROWS;
  const int p0 = POS0[0];
  const int last = min(r0 + SW_ROWS, R) - 1;
  const int most = (p0 + last + 1) / RATIO;   // the most complete blocks of the block's rows
  if (most <= TOP || b0 >= most) return;
  // the 64 keys, 16 bytes a thread a step; keys past NBL are zeros and never written out
  const int units = DI / 8;
  for (int u = threadIdx.x; u < SW_BLOCKS * units; u += SW_THREADS) {
    const int b = u / units, at = (u % units) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b0 + b < NBL) v = *reinterpret_cast<const uint4*>(POOLED + (size_t)(b0 + b) * DI + at);
    *reinterpret_cast<uint4*>(keys + b * LD + at) = v;
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int ra = r0 + warp * 16 + gq, rb = ra + 8;
  float total[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int c = 0; c < 4; ++c) total[j][c] = 0.f;
  for (int h = 0; h < HI; ++h) {
    const __nv_bfloat16* qa = IQ + ((size_t)ra * HI + h) * DI;
    const __nv_bfloat16* qb = IQ + ((size_t)rb * HI + h) * DI;
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[j][c] = 0.f;
    for (int kk = 0; kk < DI; kk += 16) {
      uint32_t a[4];
      a[0] = ra < R ? __ldg(reinterpret_cast<const uint32_t*>(qa + kk + 2 * t)) : 0u;
      a[1] = rb < R ? __ldg(reinterpret_cast<const uint32_t*>(qb + kk + 2 * t)) : 0u;
      a[2] = ra < R ? __ldg(reinterpret_cast<const uint32_t*>(qa + kk + 8 + 2 * t)) : 0u;
      a[3] = rb < R ? __ldg(reinterpret_cast<const uint32_t*>(qb + kk + 8 + 2 * t)) : 0u;
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        uint32_t b[4];
        qmm_frag::ldmatrix4(b, keys + (j * 8 + (lane >> 4) * 8 + (lane & 7)) * LD + kk + ((lane >> 3) & 1) * 8);
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(acc[j + 1][0]), "+f"(acc[j + 1][1]), "+f"(acc[j + 1][2]), "+f"(acc[j + 1][3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]));
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) total[j][c] = total[j][c] + fmaxf(acc[j][c], 0.f);
  }
  const float root = sqrtf(static_cast<float>(DI));
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const int r = e == 0 ? ra : rb;
    if (r >= R) continue;
    const int complete = (p0 + r + 1) / RATIO;
    if (complete <= TOP) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int b = b0 + j * 8 + 2 * t + c;
        if (b < complete) SC[(size_t)r * NB + b] = total[j][2 * e + c] / root;
      }
  }
}

}  // namespace

extern "C" {

// ikc bf16 [capacity, DI] holding rows 0 .. pos + R - 1 -> the blocks of pooled bf16 [blocks, DI] that rows
// pos .. pos + R - 1 complete. scale fp32 [DI], inv_freq fp32 [HALF]; ``context`` is pos + R on the host.
// With ZS_EXTENTS (zs_qwen4_attn_pool_table): ikc is a pool of 512-row extents behind table int32 [extents], as
// zs_qwen4_attn_prep_table wrote it; pooled is the sequence's own, in order. ratio divides 512.
#ifdef ZS_EXTENTS
int zs_qwen4_attn_pool_table(const zs_view* table_, const zs_view* ikc_, const zs_view* pooled_,
                             const zs_view* pos_, const zs_view* scale_,
#else
int zs_qwen4_attn_pool(const zs_view* ikc_, const zs_view* pooled_, const zs_view* pos_, const zs_view* scale_,
#endif
                    const zs_view* inv_freq_, double eps, int64_t rows, int64_t ratio, int64_t context
#ifndef ZS_EXTENTS
                    , const zs_view* rope_, int64_t s1, int64_t s2
#endif
                    ) {
  zs::begin();
  const zs::View ikc(ikc_), pooled(pooled_), pos(pos_), scale(scale_), inv_freq(inv_freq_);
  ZS_REQUIRE(ikc.defined() && ikc.dtype() == ZS_BF16 && ikc.dim() == 2 && ikc.is_contiguous(),
             "ikc: bf16 [capacity, index_dim]");
  const int64_t di = ikc.size(1), half = inv_freq.numel();
  ZS_REQUIRE(di >= 1 && di <= WIDE, "an indexer key holds at most 256 values");
  ZS_REQUIRE(pooled.defined() && pooled.dtype() == ZS_BF16 && pooled.dim() == 2 && pooled.is_contiguous() &&
             pooled.size(1) == di, "pooled: bf16 [blocks, index_dim]");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(scale.defined() && scale.dtype() == ZS_F32 && scale.numel() == di, "scale: fp32 [index_dim]");
  ZS_REQUIRE(inv_freq.defined() && inv_freq.dtype() == ZS_F32 && half >= 1 && 2 * half <= di,
             "inv_freq: fp32 [half], 2 half within an indexer key");
#ifdef ZS_EXTENTS
  const zs::View table(table_);
  ZS_REQUIRE(table.defined() && table.dtype() == ZS_I32 && table.is_contiguous() && ikc.size(0) % EXTENT == 0 &&
             table.numel() <= ikc.size(0) / EXTENT && ratio >= 1 && EXTENT % ratio == 0,
             "table: int32 [extents] over a pool of 512-row extents, ratio dividing 512");
  ZS_REQUIRE(rows >= 1 && context >= rows && context <= table.numel() * EXTENT && context / ratio <= pooled.size(0),
             "the table holds pos + R rows and pooled their complete blocks");
#define ZS_TABLE table.data_ptr<int>(),
#else
  ZS_REQUIRE(rows >= 1 && ratio >= 1 && context >= rows && context <= ikc.size(0) &&
             context / ratio <= pooled.size(0), "the caches hold pos + R rows and their complete blocks");
#define ZS_TABLE
#endif
#ifdef ZS_EXTENTS
  const int* const rope_words = nullptr;    // the table's one sequence is text; a round's rows: qsa_rows.cu
  const int64_t s1 = 0, s2 = 0;
#else
  // rope int32 [2 + 3 capacity], or null: the blocks' own positions (attention_rope.cuh). Only positions under its
  // length rope[1] read the table, and `Rope.set` keeps that within the capacity, whatever the context.
  const zs::View rope(rope_);
  ZS_REQUIRE(!rope.defined() || (rope.dtype() == ZS_I32 && rope.is_contiguous() && rope.numel() >= 2 &&
                                 (rope.numel() - 2) % 3 == 0),
             "rope: int32 [2 + 3 capacity]");
  const int* const rope_words = rope.defined() ? rope.data_ptr<int>() : nullptr;
#endif
  pool_kernel<<<static_cast<unsigned>(rows / ratio + 2), WIDE, 0, zs::stream()>>>(
      ZS_TABLE ikc.data_ptr<__nv_bfloat16>(), pooled.data_ptr<__nv_bfloat16>(), pos.data_ptr<int>(), scale.data_ptr<float>(),
      inv_freq.data_ptr<float>(), static_cast<float>(eps), static_cast<int>(rows), static_cast<int>(di),
      static_cast<int>(half), static_cast<int>(ratio), rope_words, static_cast<int>(s1), static_cast<int>(s2));
#undef ZS_TABLE
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// dots fp32 [R * heads, blocks]: each indexer query against the first ``blocks`` = (pos + R) / ratio pooled blocks.
// -> scores fp32 [R, NB] (scratch), ids int32 [R, IDW] with IDW >= (top + 1) ratio, nk and sparse int32 [R].
// With ZS_EXTENTS (zs_qwen4_attn_select_table): ids holds, for each key, its row in the pools behind table int32
// [extents], which is what zs_attn_attend_table reads; ratio divides 512. The entries of positions below
// pos + R are read, and no other.
#ifdef ZS_EXTENTS
int zs_qwen4_attn_select_table(const zs_view* table_, const zs_view* dots_, const zs_view* pos_,
                               const zs_view* scores_, const zs_view* ids_,
#else
int zs_qwen4_attn_select(const zs_view* dots_, const zs_view* pos_, const zs_view* scores_, const zs_view* ids_,
#endif
                      const zs_view* nk_, const zs_view* sparse_, int64_t heads, int64_t dim, int64_t ratio,
                      int64_t top) {
  zs::begin();
  const zs::View dots(dots_), pos(pos_), scores(scores_), ids(ids_), nk(nk_), sparse(sparse_);
  ZS_REQUIRE(dots.defined() && dots.dtype() == ZS_F32 && dots.dim() == 2 && dots.is_contiguous() && heads >= 1 &&
             dots.size(0) % heads == 0, "dots: fp32 [R * heads, blocks]");
  const int64_t rows = dots.size(0) / heads, nbl = dots.size(1);
  ZS_REQUIRE(rows >= 1 && dim >= 1 && ratio >= 1 && top >= 1, "rows, dim, ratio and top are positive");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(scores.defined() && scores.dtype() == ZS_F32 && scores.dim() == 2 && scores.is_contiguous() &&
             scores.size(0) >= rows && scores.size(1) >= nbl, "scores: fp32 [R, NB], NB at least the blocks");
  ZS_REQUIRE(ids.defined() && ids.dtype() == ZS_I32 && ids.dim() == 2 && ids.is_contiguous() &&
             ids.size(0) >= rows && ids.size(1) >= (top + 1) * ratio, "ids: int32 [R, IDW], IDW >= (top + 1) ratio");
  ZS_REQUIRE(nk.defined() && nk.dtype() == ZS_I32 && nk.numel() >= rows && sparse.defined() &&
             sparse.dtype() == ZS_I32 && sparse.numel() >= rows, "nk, sparse: int32 [R]");
  ZS_REQUIRE(nbl <= INT32_MAX / ratio && scores.size(1) <= INT32_MAX, "blocks within int32");
#ifdef ZS_EXTENTS
  const zs::View table(table_);
  ZS_REQUIRE(table.defined() && table.dtype() == ZS_I32 && table.is_contiguous() && EXTENT % ratio == 0 &&
             nbl * ratio <= table.numel() * EXTENT,
             "table: int32 [extents] holding the blocks' positions, ratio dividing 512");
#define ZS_TABLE table.data_ptr<int>(),
#else
#define ZS_TABLE
#endif
  const auto stream = zs::stream();
  scores_kernel<<<blocks(rows * nbl), 256, 0, stream>>>(
      dots.data_ptr<float>(), pos.data_ptr<int>(), scores.data_ptr<float>(), static_cast<int>(rows),
      static_cast<int>(heads), static_cast<int>(nbl), static_cast<int>(scores.size(1)), static_cast<int>(dim),
      static_cast<int>(ratio), static_cast<int>(top));
  ZS_CUDA_LAUNCH_CHECK();
  select_kernel<<<static_cast<unsigned>(rows), SEL, 0, stream>>>(
      ZS_TABLE scores.data_ptr<float>(), pos.data_ptr<int>(), ids.data_ptr<int>(), nk.data_ptr<int>(),
      sparse.data_ptr<int>(), static_cast<int>(scores.size(1)), static_cast<int>(ratio), static_cast<int>(top),
      static_cast<int>(ids.size(1)));
#undef ZS_TABLE
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// A prompt window's selection at once: iq bf16 [R, heads, dim] and pooled bf16 [at least nbl, dim], the first
// nbl = (pos + R) / ratio of them complete -> scores, ids, nk, sparse as zs_qwen4_attn_select's, for all R rows:
// scores_window_kernel and one select_kernel, where zs_qwen4_attn_select takes a GEMM, scores_kernel and
// select_kernel a tile of rows. The same lists: a row's scores are the same bits, and its list is its own.
// With ZS_EXTENTS (zs_qwen4_attn_select_window_table, spec 149 WS-02, a pool's prompt round): the pooled keys
// as a slot holds them, in order, and ids the pool's rows behind table, as zs_qwen4_attn_select_table's.
#ifdef ZS_EXTENTS
int zs_qwen4_attn_select_window_table(const zs_view* table_, const zs_view* iq_, const zs_view* pooled_,
                                      const zs_view* pos_,
#else
int zs_qwen4_attn_select_window(const zs_view* iq_, const zs_view* pooled_, const zs_view* pos_,
#endif
                                const zs_view* scores_, const zs_view* ids_, const zs_view* nk_,
                                const zs_view* sparse_, int64_t rows, int64_t heads, int64_t dim, int64_t nbl,
                                int64_t ratio, int64_t top) {
  zs::begin();
  const zs::View iq(iq_), pooled(pooled_), pos(pos_), scores(scores_), ids(ids_), nk(nk_), sparse(sparse_);
  ZS_REQUIRE(rows >= 1 && heads >= 1 && ratio >= 1 && top >= 1 && nbl >= 1, "rows, heads, ratio, top, blocks");
  ZS_REQUIRE(dim >= 16 && dim % 16 == 0 && dim <= WIDE, "dim: a multiple of 16, at most 256");
  ZS_REQUIRE(iq.defined() && iq.dtype() == ZS_BF16 && iq.is_contiguous() && iq.numel() == rows * heads * dim &&
             reinterpret_cast<uintptr_t>(iq.data_ptr<__nv_bfloat16>()) % 4 == 0, "iq: bf16 [rows, heads, dim]");
  ZS_REQUIRE(pooled.defined() && pooled.dtype() == ZS_BF16 && pooled.dim() == 2 && pooled.is_contiguous() &&
             pooled.size(1) == dim && pooled.size(0) >= nbl &&
             reinterpret_cast<uintptr_t>(pooled.data_ptr<__nv_bfloat16>()) % 16 == 0,
             "pooled: bf16 [blocks, dim] holding the complete blocks, on 16 bytes");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(scores.defined() && scores.dtype() == ZS_F32 && scores.dim() == 2 && scores.is_contiguous() &&
             scores.size(0) >= rows && scores.size(1) >= nbl, "scores: fp32 [R, NB], NB at least the blocks");
  ZS_REQUIRE(ids.defined() && ids.dtype() == ZS_I32 && ids.dim() == 2 && ids.is_contiguous() &&
             ids.size(0) >= rows && ids.size(1) >= (top + 1) * ratio, "ids: int32 [R, IDW], IDW >= (top + 1) ratio");
  ZS_REQUIRE(nk.defined() && nk.dtype() == ZS_I32 && nk.numel() >= rows && sparse.defined() &&
             sparse.dtype() == ZS_I32 && sparse.numel() >= rows, "nk, sparse: int32 [R]");
  ZS_REQUIRE(nbl <= INT32_MAX / ratio && scores.size(1) <= INT32_MAX && rows * heads <= INT32_MAX,
             "blocks and rows within int32");
#ifdef ZS_EXTENTS
  const zs::View table(table_);
  ZS_REQUIRE(table.defined() && table.dtype() == ZS_I32 && table.is_contiguous() && EXTENT % ratio == 0 &&
             nbl * ratio <= table.numel() * EXTENT,
             "table: int32 [extents] holding the blocks' positions, ratio dividing 512");
#define ZS_TABLE table.data_ptr<int>(),
#else
#define ZS_TABLE
#endif
  const auto stream = zs::stream();
  const dim3 grid(static_cast<unsigned>((nbl + SW_BLOCKS - 1) / SW_BLOCKS),
                  static_cast<unsigned>((rows + SW_ROWS - 1) / SW_ROWS));
  const size_t bytes = (size_t)SW_BLOCKS * (dim + 8) * sizeof(__nv_bfloat16);
  scores_window_kernel<<<grid, SW_THREADS, bytes, stream>>>(
      iq.data_ptr<__nv_bfloat16>(), pooled.data_ptr<__nv_bfloat16>(), pos.data_ptr<int>(), scores.data_ptr<float>(),
      static_cast<int>(rows), static_cast<int>(heads), static_cast<int>(dim), static_cast<int>(nbl),
      static_cast<int>(scores.size(1)), static_cast<int>(ratio), static_cast<int>(top));
  ZS_CUDA_LAUNCH_CHECK();
  select_kernel<<<static_cast<unsigned>(rows), SEL, 0, stream>>>(
      ZS_TABLE scores.data_ptr<float>(), pos.data_ptr<int>(), ids.data_ptr<int>(), nk.data_ptr<int>(),
      sparse.data_ptr<int>(), static_cast<int>(scores.size(1)), static_cast<int>(ratio), static_cast<int>(top),
      static_cast<int>(ids.size(1)));
#undef ZS_TABLE
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
