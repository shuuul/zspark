// The routing plan of the grouped experts: pairs (row, slot) grouped by expert into items of at most T pairs.
// From TensorFold 0.6.5 (609ca41) experts.cu and experts.cpp, Apache-2.0; changed for zspark: the plan kernels
// alone, ATen and c10 replaced by zs/host.h, a C ABI over zs_view, which takes items of 32 pairs too (spec 021)
// and of 128 (spec 114 WS-03).

#include <stdint.h>

#include "../zs/host.h"

namespace {

constexpr int PLAN_THREADS = 1024;
constexpr int EMAX = 1024;
constexpr int PSMALL = 1024;       // pairs the one-block plan takes; wider calls rank in blocks of PLAN_THREADS

// Whole block: thread e gives expert e's pair count, gets its first member; items of <= T pairs land in expert order.
__device__ int place_items(int c, int E, int T, int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int wsum[3][32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tiles = (c + T - 1) / T;
  int a = c, b = tiles, u = c > 0;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int ua = __shfl_up_sync(0xffffffffu, a, o), ub = __shfl_up_sync(0xffffffffu, b, o);
    const int uu = __shfl_up_sync(0xffffffffu, u, o);
    if (lane >= o) {
      a += ua;
      b += ub;
      u += uu;
    }
  }
  if (lane == 31) {
    wsum[0][warp] = a;
    wsum[1][warp] = b;
    wsum[2][warp] = u;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      int v = wsum[s][lane];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int w = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += w;
      }
      wsum[s][lane] = v;
    }
  }
  __syncthreads();
  const int off = (warp ? wsum[0][warp - 1] : 0) + a - c;
  const int ioff = (warp ? wsum[1][warp - 1] : 0) + b - tiles;
  if (tid == 0) {
    counts[0] = wsum[1][31];
    counts[1] = wsum[2][31];
  }
  if (tid < E)
    for (int j = 0; j < tiles; ++j) {
      int* it = items + 3 * (ioff + j);
      it[0] = tid;
      it[1] = off + T * j;
      it[2] = min(T, c - T * j);
    }
  return off;
}

// One block: pairs p = row * slots + slot grouped by expert (pair order within an expert).
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_kernel(const int* __restrict__ picks, int P, int E, int T, int* __restrict__ members,
                int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int pk[PSMALL];
  __shared__ int cnt[EMAX];
  __shared__ int off[EMAX];
  const int tid = threadIdx.x;
  for (int e = tid; e < EMAX; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = picks[p];
    pk[p] = e;
    atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int o = place_items(tid < E ? cnt[tid] : 0, E, T, items, counts);
  if (tid < E) off[tid] = o;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = pk[p];
    int rank = 0;
    for (int q = 0; q < p; ++q) rank += pk[q] == e;
    members[off[e] + rank] = p;
  }
}

// Wide calls, pass 1: block b ranks its pairs within each expert in pair order and writes its counts to hist[b][e].
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_rank(const int* __restrict__ picks, int P, int E, int* __restrict__ rank, int* __restrict__ hist) {
  __shared__ int cnt[EMAX];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int e = tid; e < E; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  const int p = blockIdx.x * PLAN_THREADS + tid;
  const bool ok = p < P;
  const int e = ok ? picks[p] : -1 - lane;
  const unsigned same = __match_any_sync(0xffffffffu, e);
  const int below = __popc(same & ((1u << lane) - 1u));
  for (int w = 0; w < PLAN_THREADS / 32; ++w) {
    if (warp == w) {
      const int base = ok ? cnt[e] : 0;
      __syncwarp();
      if (ok && below == 0) cnt[e] = base + __popc(same);
      if (ok) rank[p] = base + below;
    }
    __syncthreads();
  }
  for (int x = tid; x < E; x += PLAN_THREADS) hist[(size_t)blockIdx.x * E + x] = cnt[x];
}

// Pass 2, one block: each block's first member for each expert in place of its count, and the items.
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_offsets(int nblk, int E, int T, int* __restrict__ hist, int* __restrict__ items, int* __restrict__ counts) {
  const int tid = threadIdx.x;
  int c = 0;
  if (tid < E)
    for (int b = 0; b < nblk; ++b) {
      int* h = hist + (size_t)b * E + tid;
      const int v = *h;
      *h = c;
      c += v;
    }
  const int off = place_items(c, E, T, items, counts);
  if (tid < E)
    for (int b = 0; b < nblk; ++b) hist[(size_t)b * E + tid] += off;
}

// Pass 3: every pair to its place.
__global__ void plan_scatter(const int* __restrict__ picks, int P, int E, const int* __restrict__ rank,
                             const int* __restrict__ hist, int* __restrict__ members) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < P) members[hist[(size_t)(p / PLAN_THREADS) * E + picks[p]] + rank[p]] = p;
}


void experts_plan_cuda(const zs::View& picks, int64_t pairs, int64_t experts, int64_t tile, zs::View& members,
                       zs::View& items, zs::View& counts, zs::View& rank, zs::View& hist) {
  const auto stream = zs::stream();
  const int P = static_cast<int>(pairs), E = static_cast<int>(experts), T = static_cast<int>(tile);
  if (P <= PSMALL) {
    plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(picks.data_ptr<int>(), P, E, T, members.data_ptr<int>(),
                                                 items.data_ptr<int>(), counts.data_ptr<int>());
    ZS_CUDA_LAUNCH_CHECK();
    return;
  }
  const int nblk = (P + PLAN_THREADS - 1) / PLAN_THREADS;
  if (rank.numel() < P || hist.numel() < (int64_t)nblk * E) {
    zs::fail("experts: plan scratch too small");
    return;
  }
  plan_rank<<<nblk, PLAN_THREADS, 0, stream>>>(picks.data_ptr<int>(), P, E, rank.data_ptr<int>(),
                                               hist.data_ptr<int>());
  ZS_CUDA_LAUNCH_CHECK();
  plan_offsets<<<1, PLAN_THREADS, 0, stream>>>(nblk, E, T, hist.data_ptr<int>(), items.data_ptr<int>(),
                                               counts.data_ptr<int>());
  ZS_CUDA_LAUNCH_CHECK();
  plan_scatter<<<(P + 255) / 256, 256, 0, stream>>>(picks.data_ptr<int>(), P, E, rank.data_ptr<int>(),
                                                    hist.data_ptr<int>(), members.data_ptr<int>());
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

// picks: each pair's expert, pair p = row * slots + slot. members [pairs], items [max items, 3] (expert, first,
// count), counts [2] (items, distinct experts); rank [pairs] and hist [ceil(pairs / 1024) * experts] are scratch
// that calls of more than 1024 pairs use.
extern "C" int zs_experts_plan(const zs_view* picks_, int64_t pairs, int64_t experts, int64_t tile,
                               const zs_view* members_, const zs_view* items_, const zs_view* counts_,
                               const zs_view* rank_, const zs_view* hist_) {
  zs::begin();
  const zs::View picks(picks_);
  zs::View members(members_), items(items_), counts(counts_), rank(rank_), hist(hist_);
  const zs::View* buffers[] = {&picks, &members, &items, &counts, &rank, &hist};
  for (const zs::View* t : buffers) ZS_REQUIRE(t->defined() && t->dtype() == ZS_I32, "plan buffers are int32");
  ZS_REQUIRE(picks.numel() >= pairs, "picks: one id a pair");
  ZS_REQUIRE(members.numel() >= pairs && counts.numel() >= 2, "plan buffers too small");
  ZS_REQUIRE(tile == 16 || tile == 32 || tile == 64 || tile == 128,
             "items hold 16 pairs (decode), 32, 64 or 128 (wide windows; 128 W4A4's, spec 114 WS-03)");
  ZS_REQUIRE(experts <= EMAX, "experts: at most 1024 experts");
  experts_plan_cuda(picks, pairs, experts, tile, members, items, counts, rank, hist);
  return zs::status();
}
