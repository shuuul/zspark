// zspark's own (spec 105 WS-03): the grouped NVFP4 experts of a prompt's window, 64 pairs of an expert (four m16
// tiles) under one reading of its weights, where experts_wide.cu holds 32. Its rows and its weights are staged
// through shared memory by cp.async, the rows once for the block's 8 warps, not loaded by every warp into its
// registers: that is what frees the registers for twice the tiles (spec 087 WS-04 found a second pair of tiles
// in registers slower, its sums or its rows spilling). fp4pair, scale_pair, mul_pair, the weights' layout and the
// epilogue are experts.cu's and experts_wide.cu's.
//
// The arithmetic is experts.cu's to the bit: a lane's A registers hold the same inputs as there (row gq and
// gq + 8, inputs 4t .. 4t + 3 of each 16-input block), so every mma sees the same operands; each weight is its
// e2m1 value times its block's e4m3 scale in bf16 (exact) and the mma adds into acc, blocks in order; each
// (expert, matrix) scale multiplied once before the epilogue. A pair's sums are its own accumulators whichever tile it sits in, so a
// pair's bits depend on no other pair, whatever the plan's item size.
//
// A block is 8 warps on one item (an expert's 64 pairs at most) and 8 / MW column blocks of 32: with gate and
// up (MW 2) a warp is one matrix of one column block, and the up warp hands its sums to the gate warp through
// shared memory for SwiGLU; down (MW 1) a warp a column block. 128 registers a thread, two blocks an SM.
//
// Measured alone at 2048 rows (ZSPARK_NVFP4_PROFILE, spark-a): plan, gate and up and down 7.73 ms against
// experts_wide.cu's 10.04 with all 512 experts routed, about 80% of DRAM for the 1.4 GB they read. Tried and
// slower or no faster: the weights in registers by ld.global.nc, double-buffered as experts_wide.cu holds them
// (9.57 ms; 7.96 once held to 128 registers for two blocks an SM; barrier and scoreboard stalls), two groups of
// 32 inputs a stage for half the barriers (7.97), and the stages run on across a block's units (7.83).
// Spec 107 WS-03: the rows staged in the mma's fragment order and the scale folded into the weights, 7.29 ms
// (skew, Flash Next's shapes; 13.80 on GLM's, from 14.42). Staged as rows, a lane's two 8-byte reads came in the
// wrong register order for the mma and the SASS moved four registers an HMMA; staged as fragments, one LDS.128
// feeds an mma. ncu on GLM's at 2048 rows: gate and up 8.26 ms, the tensor pipe 53% active, down 5.28 and 41%,
// the first stalls math-pipe throttle and the barrier.
//
// Shared memory a block: STAGES stages of 8,704 B, 43,520 B: a stage is the rows' 32 inputs (64 rows of 64 B,
// 4,096 B in fragment order: for tile r, lane (gq, t) and half h, the lane's four A registers (rows gq and gq + 8,
// inputs 4t .. 4t + 3 of the half) as one 16-byte chunk, the halves swapped by gq's low bit so a quarter-warp's
// 16-byte reads meet no bank twice; copied 4 bytes a cp.async) and each warp's (32 columns, 32 inputs) block (576 B, 4,608 B); after the K loop the first 8,192 B hand
// the up sums over, 16 floats a lane of each up warp a tile. Two blocks an SM take 87 KB of the 100. What could
// be less: fewer stages (each one is 8.7 KB; five keep the weights' latency covered at two blocks an SM); rows
// past the item's pairs are not copied and their stale bytes give sums nobody stores.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../common/experts.cuh"
#include "../zs/host.h"
#include "codes.cuh"

namespace {

using namespace nvfp4_codes;

constexpr int BLOCK4 = 36;            // uint4 a (32 columns, 32 inputs) block: a lane's code words, then the scales
constexpr int RT = 4;                 // m16 tiles of pairs a unit
constexpr int PAIRS = 16 * RT;        // pairs an item holds at most
constexpr int WARPS = 8;
constexpr int THREADS = 32 * WARPS;
constexpr int XROW = 64;              // bytes a staged row a stage: 32 inputs
constexpr int XSTAGE = PAIRS * XROW;  // 4,096
constexpr int WSTAGE = BLOCK4 * 16;   // 576: a warp's (32 columns, 32 inputs) block
constexpr int STAGE = XSTAGE + WARPS * WSTAGE;
constexpr int STAGES = 5;
static_assert(THREADS == PAIRS * 4, "a thread copies one 16-byte chunk of one row a stage");

// A stage's rows as the mma's A fragments: lane (gq, t) of tile r reads one uint4 for each 16-input half h, its
// four registers in order: rows 16 r + gq and 16 r + gq + 8, their inputs 16 h + 4 t, + 1 (word 8 h + 2 t), then
// + 2, + 3 (the word after). Lanes of odd gq trade halves of their 128 bytes, so a quarter-warp meets eight banks.
__device__ __forceinline__ int xin(int r, int gq, int h, int t) { return (r * 8 + gq) * 8 + ((h * 4 + t) ^ ((gq & 1) * 4)); }

__device__ __forceinline__ void cp4(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}

struct Weights {
  uint4 w;  // the lane's words: n8 tiles 0-3, 32 inputs each
  uint4 s;  // the quad's scales, bytes [block][tile][column]
};

// One stage's products into acc. Each weight is its e2m1 value times its 16-input block's scale, exactly, before
// the mma, which adds into the sums: the lane's B registers are column gq's, so it reads the scales of the quad
// that holds columns gq & ~1 and picks gq's bytes (sel).
__device__ __forceinline__ void compute(float (&acc)[RT][NTW][4], const Weights& st, const unsigned char* xs,
                                        int gq, int t, int nt, const uint32_t (&sel)[2]) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint4 x[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) x[r] = reinterpret_cast<const uint4*>(xs)[xin(r, gq, h, t)];
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const uint32_t word = comp(st.w, j);
      const uint32_t ss = scale_pair(comp(st.s, 2 * h + (j >> 1)), sel[j & 1]);
      const uint32_t b0 = mul_pair(fp4pair(word, 8 * h), ss), b1 = mul_pair(fp4pair(word, 8 * h + 4), ss);
#pragma unroll
      for (int r = 0; r < RT; ++r) mma(acc[r][j], x[r].x, x[r].y, x[r].z, x[r].w, b0, b1);
    }
  }
}

// MW matrices (2: gate and up, 1: down), WARPS / MW column blocks a block; pair p reads X row p / slots when
// slots > 0, else X row p; items of expert ``skip`` are left alone.
template <int MW, int EPI>
__global__ void __launch_bounds__(THREADS, 2)
    nvfp4_expert_tile_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                             const uint4* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                             const int* __restrict__ items, const int* __restrict__ counts,
                             const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  constexpr int CBS = WARPS / MW;
  __shared__ __align__(16) unsigned char xs[STAGES * STAGE];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, t = lane & 3;
  const int m = warp % MW, c = warp / MW;
  const int groups = (NB + CBS - 1) / CBS;
  const int units = __ldg(counts) * groups;
  // the row and chunk this thread copies a stage
  const int crow = tid >> 2, cchunk = tid & 3;
  unsigned char* const wown = xs + XSTAGE + warp * WSTAGE;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / groups, cg = unit - it * groups;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const int cb = cg * CBS + c;
    const bool live = cb < NB;
    const int nt = (cnt + 15) >> 4;
    const bool copies = crow < cnt;
    const __nv_bfloat16* src = X;
    if (copies) {
      const int p = __ldg(members + first + crow);
      src = X + (size_t)(slots ? p / slots : p) * x_stride + cchunk * 8;
    }
    // words 4 q .. 4 q + 3 of the row's 16 a stage: word w lands in xin(r, gq, w / 8, (w % 8) / 2) at register
    // (w % 2) * 2 + (row % 16) / 8
    const int rr = crow >> 4, rg = crow & 7, rh = (crow >> 3) & 1, q = cchunk;
    unsigned char* const dst = xs + xin(rr, rg, q >> 1, 2 * (q & 1)) * 16 + rh * 4;
    unsigned char* const dst2 = xs + xin(rr, rg, q >> 1, 2 * (q & 1) + 1) * 16 + rh * 4;
    const int stride = MW * BLOCK4;
    const uint4* const blk = W + ((size_t)e * NB + (live ? cb : 0)) * (size_t)KG * stride + m * BLOCK4;
    // a stage: the rows' group g, and each warp's block of it
    auto load = [&](int s, int g) {
      if (copies) {
        const __nv_bfloat16* from = src + g * 32;
        cp4(dst + s * STAGE, from);
        cp4(dst + s * STAGE + 8, from + 2);
        cp4(dst2 + s * STAGE, from + 4);
        cp4(dst2 + s * STAGE + 8, from + 6);
      }
      if (live) {
        const uint4* b = blk + (size_t)g * stride;
        cp16(wown + s * STAGE + lane * 16, b + lane);
        if (lane < 4) cp16(wown + s * STAGE + (32 + lane) * 16, b + 32 + lane);
      }
    };
    bool v1[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) v1[r] = 16 * r + gq + 8 < cnt;
    // byte (j & 1) * 2 + (gq & 1) of a scale word, in the low and high halves
    const uint32_t lo = gq & 1, hi = 2 + (gq & 1);
    const uint32_t sel[2] = {lo | 0x40u | (lo << 8) | 0x4000u, hi | 0x40u | (hi << 8) | 0x4000u};

    // every warp has finished the last unit's reads of the stages (and of the hand-over) here
    __syncthreads();
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
        const unsigned char* const stage = xs + (g % STAGES) * STAGE;
        const uint4* const wb = reinterpret_cast<const uint4*>(stage + XSTAGE + warp * WSTAGE);
        Weights st;
        st.w = wb[lane];
        st.s = wb[32 + (gq >> 1)];
        compute(acc[0], st, stage, gq, t, nt, sel);
      }
    }
    cp_wait<0>();
    if (live) {
      const float gs = __ldg(scale + e * MW + m);
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[0][r][j][q] *= gs;
    }
    const int col = cb * COLS + 2 * t;
    if constexpr (MW == 1) {
      if (live) {
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (r >= nt) break;
          const int r0 = 16 * r + gq;
          const bool v0 = r0 < cnt;
          const int p0 = v0 ? __ldg(members + first + r0) : 0, p1 = v1[r] ? __ldg(members + first + r0 + 8) : 0;
          epilogue<EPI, 1, RT>(acc, r, out, N, col, p0, p1, v0, v1[r], limit);
        }
      }
    } else {
      float* hand = reinterpret_cast<float*>(xs) + c * (NTW * 4 * 32);
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= nt) break;
        __syncthreads();
        if (live && m == 1) {
#pragma unroll
          for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) hand[(j * 4 + q) * 32 + lane] = acc[0][r][j][q];
        }
        __syncthreads();
        if (live && m == 0) {
          float two[2][1][NTW][4];
#pragma unroll
          for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              two[0][0][j][q] = acc[0][r][j][q];
              two[1][0][j][q] = hand[(j * 4 + q) * 32 + lane];
            }
          const int r0 = 16 * r + gq;
          const bool v0 = r0 < cnt;
          const int p0 = v0 ? __ldg(members + first + r0) : 0, p1 = v1[r] ? __ldg(members + first + r0 + 8) : 0;
          epilogue<EPI, 2, 1>(two, 0, out, N, col, p0, p1, v0, v1[r], limit);
        }
      }
    }
  }
}

template <int MW, int EPI>
void launch(const zs::View& x, int x_stride, int slots, const zs::View& w, const zs::View& scale, int kg, int nb,
            const zs::View& items, const zs::View& counts, const zs::View& members, zs::View& out, int n,
            float limit, int skip, int64_t max_units) {
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nvfp4_expert_tile_kernel<MW, EPI>, THREADS, 0);
    sms = zs::device().multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  // max_units counts (item, column block) units: a block takes WARPS / MW of them
  constexpr int CBS = WARPS / MW;
  const int64_t need = (max_units + CBS - 1) / CBS;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  nvfp4_expert_tile_kernel<MW, EPI><<<grid, THREADS, 0, zs::stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots, reinterpret_cast<const uint4*>(w.data_ptr()),
      scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(), counts.data_ptr<int>(), members.data_ptr<int>(),
      out.data_ptr(), n, limit, skip);
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

// nvfp4_experts_cuda's arguments (experts.cu), for four tiles of rows a reading: the plan's items of 64 pairs.
void nvfp4_experts_tile_cuda(int64_t epi, const zs::View& x, int64_t x_stride, int64_t slots, const zs::View& w,
                             const zs::View& scale, int64_t kg, int64_t nb, const zs::View& items,
                             const zs::View& counts, const zs::View& members, zs::View& out, int64_t n,
                             double limit, int64_t skip, int64_t max_units) {
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  if (epi == 2) launch<2, 2>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 0) launch<1, 0>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 3) launch<1, 3>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else if (epi == 1) launch<1, 1>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_units);
  else zs::fail("nvfp4 experts: epilogue 0 (fp32 down), 1 (relu squared up), 2 (SwiGLU) or 3 (bf16 down)");
}
