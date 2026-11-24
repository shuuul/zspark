// zspark's own (spec 114 WS-03): the W4A4 routed experts of a prompt's window, 64 pairs of an expert (four m16
// tiles) under one reading of its weights, as experts_tile.cu holds them for bf16 rows. The pairs' rows are the
// plan's NVFP4 rows (experts_quant.cu), the weights NVFP4 as stored, the products the block-scaled FP4 instruction
// (experts_a4.cuh). The block's shape, the units, the ring of stages and the epilogue (the up warp's sums handed
// to the gate warp through shared memory for SwiGLU) are experts_tile.cu's; a stage is 64 inputs, one mma's. Gate
// and up's epilogue is experts_a4.cu's EPI 4: down's rows quantized from the fp32 SwiGLU.
//
// The arithmetic is experts_a4.cu's to the bit: a lane's A registers and SFA word hold the same row's same bytes,
// its B registers and SFB word the same column's, each mma adds into acc, k64 steps in order from zero; then times
// alpha (weight_scale_2 / gs); then the epilogue. A pair's sums are its own accumulators whichever tile it sits in.
//
// A block is 8 warps on one item (an expert's 64 pairs at most) and 8 / MW column blocks of 32: with gate and up
// (MW 2) a warp is one matrix of one column block; down (MW 1) a warp a column block. RH 2 (items of 128 pairs, a
// prompt's window where an expert has more than 64: spec 114 WS-03) adds 8 warps on the item's second 64 pairs,
// which read the weight blocks the first 8 copied: the weights are read once for 128 pairs, each warp's
// accumulators those of 64.
//
// A block takes its units ((item, column group) pairs) as it frees, from a counter in the plan's counts (spec 114
// WS-09): with units dealt by the grid's stride, a block's column group was fixed and the blocks on long items
// ended up to 25% later than the rest at a 4,096-row window; taken as they free, the blocks end together and an
// expert's items on one column group run near each other, the later reading in L2 what the earlier brought.
// Which block runs a unit changes no bit: a unit's sums and outputs are its own.
//
// Shared memory a block, RH 1: STAGES1 stages of 11,520 B (46,080 B at four), two blocks an SM in 94 KB of the 100;
// RH 2: STAGES2 stages of 13,824 B (96,768 B at seven, dynamic), one block an SM, six stages of weights in flight
// as two RH 1 blocks have; then 16 B for the unit taken next:
//   a stage                            bytes   what it is
//   the rows' codes, fragment order    2,048   tile r, lane l: one uint4 {a0, a1, a2, a3}, one LDS.128 an mma
//                                    x RH      row tile, no bank met twice by a quarter-warp; copied 4 bytes a
//                                              cp.async (512 a stage a 64 pairs, two a thread)
//   the rows' scales                     256   row R's word at R: lane c 0 reads row q's, c 1 row q + 8's
//                                    x RH
//   each column block's weights        9,216   72 uint4 a block: its (32 columns, 64 inputs), as stored
//   after the stages: the unit next       16   a block's, once (a static __shared__ before the dynamic stages put
//                                              their copies off the weights' 32-byte sectors: twice the sectors)
// After the K loop the first 8,192 B (x RH) hand the up sums over, 16 floats a lane of each up warp a tile; the
// gate warp then makes down's NVFP4 rows from both in registers (experts_a4.cuh `swiglu_a4`, spec 114 WS-07: no
// shared memory added, the quad's shuffles meet a group's largest magnitude and its code bytes). What
// could be less: the rows are a fifth of a stage (a third at RH 2; bf16 rows were half of experts_tile.cu's); the
// weights are what buys speed. Rows past the item's pairs are not copied and their stale bytes give sums nobody
// stores.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/experts.cuh"
#include "../zs/host.h"
#include "experts_a4.cuh"

namespace {

constexpr int RT = 4;                   // m16 tiles of pairs a warp
constexpr int CW = 8;                   // warps over the column blocks
constexpr int WSTAGE = A4_BLOCK4 * 16;  // 1,152: a (32 columns, 64 inputs) block

template <int RH>
struct TileA4 {
  static constexpr int PAIRS = 16 * RT * RH;  // pairs an item holds at most
  static constexpr int WARPS = CW * RH;
  static constexpr int THREADS = 32 * WARPS;
  static constexpr int XA = RT * RH * 32 * 16;  // the rows' codes in fragment order
  static constexpr int SA = PAIRS * 4;          // the rows' scale words
  static constexpr int STAGE = XA + SA + CW * WSTAGE;
  static constexpr int STAGES = RH == 1 ? 4 : 7;
  static constexpr int SMEM = STAGES * STAGE + 16;  // and the unit taken next, after the stages
  static_assert(THREADS * 2 == PAIRS * 8, "a thread copies two 4-byte words of codes a stage");
  static_assert(SMEM <= 101376, "an SM's shared memory");
  static_assert(STAGES * STAGE >= 8192 * RH, "the hand-over fits");
};

__device__ __forceinline__ void cp4a(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}

template <int MW, int EPI, int RH>
__global__ void __launch_bounds__(TileA4<RH>::THREADS, 3 - RH)
    nvfp4_expert_tile_a4_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                                const uint4* __restrict__ W, const float* __restrict__ alpha, int KG, int NB,
                                const int* __restrict__ items, int* __restrict__ counts,
                                const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                int skip, DownA4 dq) {
  using T = TileA4<RH>;
  constexpr int THREADS = T::THREADS, XA = T::XA, SA = T::SA, STAGE = T::STAGE, STAGES = T::STAGES;
  constexpr int CBS = CW / MW;
  static_assert((MW == 2) == (EPI == 4), "gate and up: their SwiGLU into down's rows");
  extern __shared__ __align__(16) unsigned char sm[];
  // the unit this block takes after the one it is on; not a static __shared__, which would move the stages off
  // the 32-byte alignment their copies keep with the weights' sectors (twice the sectors a copy, spec 114 WS-09)
  int& taken = *reinterpret_cast<int*>(sm + STAGES * STAGE);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = lane >> 2, c = lane & 3;
  // warp: its column block's warp wc and its half of the pairs; the first half's warps copy the weights
  const int wc = warp % CW, half = warp / CW;
  const int m = wc % MW, cbw = wc / MW;
  const int groups = (NB + CBS - 1) / CBS;
  const int units = counts[0] * groups;
  const size_t crow = (size_t)KG * 32, srow = (size_t)KG * 4;
  unsigned char* const wown = sm + XA + SA + wc * WSTAGE;
  // the two code words this thread copies a stage: row R = idx / 8, word w = idx % 8 lands in tile R / 16, lane
  // (R % 8) * 4 + w % 4, register (w / 4) * 2 + (R / 8) % 2
  int crow_of[2], cdst[2], cw[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tid + i * THREADS, R = idx >> 3, w = idx & 7;
    crow_of[i] = R;
    cw[i] = w;
    cdst[i] = (((R >> 4) * 32 + (R & 7) * 4 + (w & 3)) * 4 + (w >> 2) * 2 + ((R >> 3) & 1)) * 4;
  }
  // Units are taken as blocks free (counts[2]), after a first one each by its index: a block on items of 128 pairs
  // and one on items of 2 end together, and an expert's items' units on one column group, a few apart in the list,
  // run at once, the later reading the weights the earlier brought into L2.
  for (int unit = blockIdx.x; unit < units;) {
    // every warp has finished the last unit's reads of the stages, of the hand-over and of `taken` here
    __syncthreads();
    if (tid == 0) taken = gridDim.x + atomicAdd(counts + 2, 1);
    const int it = unit / groups, cg = unit - it * groups;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) {
      __syncthreads();
      unit = taken;
      continue;
    }
    const int cb = cg * CBS + cbw;
    const bool live = cb < NB;
    // the block's tiles of 16 pairs, and this warp's (half 0's are the more)
    const int ntb = min((cnt + 15) >> 4, RT);
    const int nt = min(max(((cnt + 15) >> 4) - half * RT, 0), RT);
    const int stride = MW * A4_BLOCK4;
    const uint4* const blk = W + ((size_t)e * NB + (live ? cb : 0)) * (size_t)KG * stride + m * A4_BLOCK4;
    auto load = [&](int s, int g) {
      unsigned char* const st = sm + s * STAGE;
#pragma unroll
      for (int i = 0; i < 2; ++i)
        if (crow_of[i] < cnt) cp4a(st + cdst[i], codes + (size_t)(first + crow_of[i]) * crow + g * 32 + cw[i] * 4);
      if (tid < cnt) cp4a(st + XA + tid * 4, scales + (size_t)(first + tid) * srow + g * 4);
      if (live && half == 0) {
        const uint4* b = blk + (size_t)g * stride;
        cp16(wown + s * STAGE + lane * 16, b + lane);
        cp16(wown + s * STAGE + (32 + lane) * 16, b + 32 + lane);
        if (lane < 8) cp16(wown + s * STAGE + (64 + lane) * 16, b + 64 + lane);
      }
    };
    bool v1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) v1[r] = 16 * (half * RT + r) + q + 8 < cnt;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < KG) load(s, s);
      cp_commit();
    }
    float acc[1][RT][NTW][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[0][r][j][0] = acc[0][r][j][1] = acc[0][r][j][2] = acc[0][r][j][3] = 0.f;
    for (int g = 0; g < KG; ++g) {
      cp_wait<STAGES - 2>();
      __syncthreads();
      const int next = g + STAGES - 1;
      if (next < KG) load(next % STAGES, next);
      cp_commit();
      if (live) {
        const unsigned char* const stage = sm + (g % STAGES) * STAGE;
        const uint4* const wb = reinterpret_cast<const uint4*>(stage + XA + SA + wc * WSTAGE);
        const uint4 w0 = wb[lane], w1 = wb[32 + lane], sb = wb[64 + q];
        const uint4* const xa = reinterpret_cast<const uint4*>(stage);
        const uint32_t* const sa = reinterpret_cast<const uint32_t*>(stage + XA);
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (r >= nt) break;
          const int tile = half * RT + r;
          const uint4 x = xa[tile * 32 + lane];
          const uint32_t sfa = sa[tile * 16 + q + 8 * (c & 1)];
#pragma unroll
          for (int j = 0; j < NTW; ++j) {
            const uint4& w = j < 2 ? w0 : w1;
            mma_a4(acc[0][r][j], x.x, x.y, x.z, x.w, a4_comp(w, 2 * (j & 1)), a4_comp(w, 2 * (j & 1) + 1), sfa,
                   a4_comp(sb, j));
          }
        }
      }
    }
    cp_wait<0>();
    if (live) {
      const float a = __ldg(alpha + e * MW + m);
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int k = 0; k < 4; ++k) acc[0][r][j][k] *= a;
    }
    const int col = cb * COLS + 2 * c;
    if constexpr (MW == 1) {
      if (live) {
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (r >= nt) break;
          const int r0 = 16 * (half * RT + r) + q;
          const bool v0 = r0 < cnt;
          const int p0 = v0 ? __ldg(members + first + r0) : 0, p1 = v1[r] ? __ldg(members + first + r0 + 8) : 0;
          if constexpr (EPI == 7)
            activation_a4<true>(acc[0][r], acc[0][r], cb, c, dq, __ldg(dq.gs + e), N,
                                (size_t)(first + r0), (size_t)(first + r0 + 8), v0, v1[r], limit,
                                reinterpret_cast<__nv_bfloat16*>(out), p0, p1);
          else
            epilogue<EPI, 1, RT>(acc, r, out, N, col, p0, p1, v0, v1[r], limit);
        }
      }
    } else {
      float* hand = reinterpret_cast<float*>(sm) + (half * CBS + cbw) * (NTW * 4 * 32);
      // every warp meets the block's barriers, as many as half 0's tiles
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= ntb) break;
        const bool mine = r < nt;
        __syncthreads();
        if (live && mine && m == 1) {
#pragma unroll
          for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) hand[(j * 4 + k) * 32 + lane] = acc[0][r][j][k];
        }
        __syncthreads();
        if (live && mine && m == 0) {
          float two[2][1][NTW][4];
#pragma unroll
          for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              two[0][0][j][k] = acc[0][r][j][k];
              two[1][0][j][k] = hand[(j * 4 + k) * 32 + lane];
            }
          const int r0 = 16 * (half * RT + r) + q;
          const bool v0 = r0 < cnt;
          const int p0 = v0 ? __ldg(members + first + r0) : 0, p1 = v1[r] ? __ldg(members + first + r0 + 8) : 0;
          activation_a4<false>(two[0][0], two[1][0], cb, c, dq, __ldg(dq.gs + e), N, (size_t)(first + r0), (size_t)(first + r0 + 8),
                              v0, v1[r], limit, reinterpret_cast<__nv_bfloat16*>(out), p0, p1);
        }
      }
    }
    // thread 0 wrote it before the K loop's first barrier
    unit = taken;
  }
  // the last block out leaves the counters at zero for the next launch: every block has taken its last unit
  if (tid == 0 && atomicAdd(counts + 3, 1) == (int)gridDim.x - 1) {
    atomicExch(counts + 2, 0);
    atomicExch(counts + 3, 0);
  }
}

template <int MW, int EPI, int RH>
void launch(const zs::View& codes, const zs::View& scales, const zs::View& w, const zs::View& alpha, int kg, int nb,
            const zs::View& items, const zs::View& counts, const zs::View& members, void* out, int n, float limit,
            int skip, int64_t max_units, const DownA4& dq) {
  using T = TileA4<RH>;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaFuncSetAttribute(nvfp4_expert_tile_a4_kernel<MW, EPI, RH>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         T::SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_tile_a4_kernel<MW, EPI, RH>, T::THREADS,
                                                  T::SMEM);
    sms = zs::device().multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  // max_units counts (item, column block) units: a block takes CW / MW of them
  constexpr int CBS = CW / MW;
  const int64_t need = (max_units + CBS - 1) / CBS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  nvfp4_expert_tile_a4_kernel<MW, EPI, RH><<<grid, T::THREADS, T::SMEM, zs::stream()>>>(
      reinterpret_cast<const uint8_t*>(codes.data_ptr()), reinterpret_cast<const uint8_t*>(scales.data_ptr()),
      reinterpret_cast<const uint4*>(w.data_ptr()), alpha.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out, n, limit, skip, dq);
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

// zs_nvfp4_experts_a4's arguments (experts_a4.cu), for four tiles of pairs a reading (tiles 4: the plan's items of
// up to 64) or eight (tiles 8: up to 128).
void nvfp4_experts_tile_a4_cuda(int64_t tiles, int64_t epi, const zs::View& codes, const zs::View& scales, const zs::View& w,
                                const zs::View& alpha, int64_t kg, int64_t nb, const zs::View& items,
                                const zs::View& counts, const zs::View& members, void* out, int64_t n,
                                double limit, int64_t skip, int64_t max_units, const DownA4& dq) {
  const int k = static_cast<int>(kg), b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  if (tiles == 8) {
    if (epi == 4) launch<2, 4, 2>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
    else if (epi == 7) launch<1, 7, 2>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
    else if (epi == 0) launch<1, 0, 2>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
    else if (epi == 6) launch<1, 6, 2>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
    else launch<1, 3, 2>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
    return;
  }
  if (epi == 4) launch<2, 4, 1>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
  else if (epi == 7) launch<1, 7, 1>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
  else if (epi == 0) launch<1, 0, 1>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
  else if (epi == 6) launch<1, 6, 1>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
  else launch<1, 3, 1>(codes, scales, w, alpha, k, b, items, counts, members, out, nn, lim, sk, max_units, dq);
}
