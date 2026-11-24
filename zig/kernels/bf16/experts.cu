// Grouped routed experts on bf16 weights (spec 137 WS-04, Unlimited-OCR's DeepseekV2 MoE): a call's (row, pick)
// pairs grouped by expert (common/plan.cu's `zs_experts_plan`, items of 64 pairs), then two products a pair:
//   up:   act[s] = bf16(bf16(silu(bf16(x[row] . gate_e))) * bf16(x[row] . up_e)), s the pair's place in the plan;
//   down: y[pair] = bf16(act[s] . down_e).
// The rounding is HF's under bf16 autocast (each projection's bf16 output, silu in fp32 rounded, the bf16
// product), common/experts.cuh's `swiglu` with no limit. Written for zspark in CUDA C++; TensorFold's grouped
// experts are 4-bit only.
//
// Hand-written because CUTLASS's grouped GEMM takes a problem's rows contiguous and its sizes on the host: here
// an item's rows are gathered from a plan the device wrote, and the SwiGLU of two weight rows meets in one
// thread. m16n8k16 bf16 tensor cores, cp.async in two stages, ldmatrix from swizzled rows.
//
// Block (column tile, item): 64 pairs, four warps of 16 pairs, 64 weight rows (up: 32 gate rows and the 32 up
// rows of the same columns, so a thread holds both halves of its outputs; down: 64 output columns), K in tiles
// of 64 from 0 in order. A pair's sums are its own row's against the weight, the same whatever pairs share its
// item: the row rule holds by construction. Items past the plan's count return at once.
//
// Shared memory a block (32,768 B, three blocks an SM):
//   region     bytes   holds
//   A stages   16,384  two stages of 64 pairs x 64 inputs (bf16), zeros past the item's pairs
//   B stages   16,384  two stages of 64 weight rows x 64 inputs
// Everything else is in registers. A single row's decode reads every expert it picks through one item each: the
// weight bytes are the bound (WS-07's), not the 63 idle pair slots.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/math.cuh"
#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

constexpr int BM = 64;        // pairs an item
constexpr int BN = 64;        // weight rows a block
constexpr int BK = 64;        // inputs a K tile
constexpr int THREADS = 128;
constexpr int CH = BK / 8;    // 16-byte chunks a staged row

__device__ __forceinline__ int at(int r, int c) { return r * BK + ((c ^ (r & 7)) << 3); }

// UP: x bf16 [R, K] (pair p reads row p / TOP), w [E, 2 I, K] (gate rows then up rows), out act [pairs, I] at the
// pair's place. Otherwise: x is act [pairs, K] at the place, w [E, N, K], out y [pairs, N] at the pair.
template <bool UP>
__global__ void __launch_bounds__(THREADS)
    bf16_experts_kernel(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                        const int* __restrict__ members, const int* __restrict__ items, const int* __restrict__ counts,
                        __nv_bfloat16* __restrict__ OUT, int K, int N, int TOP) {
  __shared__ __align__(128) __nv_bfloat16 as[2][BM * BK];
  __shared__ __align__(128) __nv_bfloat16 bs[2][BN * BK];
  const int item = blockIdx.y;
  if (item >= counts[0]) return;
  const int expert = items[3 * item], first = items[3 * item + 1], count = items[3 * item + 2];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  // up: columns j0 .. j0 + 31 of act (N = I); down: columns n0 .. n0 + 63 of y
  const int col0 = blockIdx.x * (UP ? BN / 2 : BN);
  const size_t wrows = UP ? 2 * (size_t)N : (size_t)N;
  const __nv_bfloat16* w = W + (size_t)expert * wrows * K;

  auto stage = [&](int buf, int k0) {
    for (int i = tid; i < BM * CH; i += THREADS) {
      const int r = i / CH, c = i % CH;
      const bool real = r < count;
      const int place = first + (real ? r : 0);
      const size_t row = UP ? (size_t)(members[place] / TOP) : (size_t)place;
      qmm_frag::cp16z(&as[buf][at(r, c)], X + row * K + k0 + c * 8, real);
    }
    for (int i = tid; i < BN * CH; i += THREADS) {
      const int r = i / CH, c = i % CH;
      const size_t wr = UP ? (r < BN / 2 ? (size_t)col0 + r : (size_t)N + col0 + r - BN / 2) : (size_t)col0 + r;
      qmm_frag::cp16(&bs[buf][at(r, c)], w + wr * K + k0 + c * 8);
    }
    qmm_frag::commit();
  };

  float acc[BN / 8][4];
#pragma unroll
  for (int j = 0; j < BN / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  const int tiles = K / BK;
  stage(0, 0);
  for (int kt = 0; kt < tiles; ++kt) {
    if (kt + 1 < tiles) {
      stage((kt + 1) & 1, (kt + 1) * BK);
      qmm_frag::wait<1>();
    } else {
      qmm_frag::wait<0>();
    }
    __syncthreads();
    const __nv_bfloat16* a_s = as[kt & 1];
    const __nv_bfloat16* b_s = bs[kt & 1];
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      uint32_t a[4];
      qmm_frag::ldmatrix4(a, &a_s[at(warp * 16 + (lane & 15), kk * 2 + (lane >> 4))]);
#pragma unroll
      for (int np = 0; np < BN / 16; ++np) {
        uint32_t b[4];
        qmm_frag::ldmatrix4(b, &b_s[at(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1))]);
        qmm_frag::mma(acc[2 * np], a, b[0], b[1]);
        qmm_frag::mma(acc[2 * np + 1], a, b[2], b[3]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = warp * 16 + g + 8 * h;
    if (r >= count) continue;
    const int place = first + r;
    if constexpr (UP) {
      __nv_bfloat16* out = OUT + (size_t)place * N;
#pragma unroll
      for (int j = 0; j < BN / 16; ++j) {
        const int col = col0 + j * 8 + 2 * t;
        float v[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float gv = bf(acc[j][2 * h + e]), uv = bf(acc[j + BN / 16][2 * h + e]);
          v[e] = bf(gv / (1.f + expf(-gv))) * uv;
        }
        *reinterpret_cast<__nv_bfloat162*>(out + col) = __floats2bfloat162_rn(v[0], v[1]);
      }
    } else {
      __nv_bfloat16* out = OUT + (size_t)members[place] * N;
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int col = col0 + j * 8 + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(out + col) = __floats2bfloat162_rn(acc[j][2 * h], acc[j][2 * h + 1]);
      }
    }
  }
}

bool ints(const zs::View& v) { return v.defined() && v.dtype() == ZS_I32 && v.is_contiguous(); }

}  // namespace

extern "C" {

// x bf16 [R, H] (pair p = row * top + slot), gu bf16 [E, 2 I, H] (an expert's gate rows, then its up rows), the
// plan (members [pairs], items [max items, 3], counts [2]) of tile 64 -> act bf16 [pairs, I] in plan order.
// max_items bounds the grid (ceil(pairs / 64) + E covers every plan).
int zs_bf16_experts_up(const zs_view* x_, const zs_view* gu_, const zs_view* members_, const zs_view* items_,
                       const zs_view* counts_, int64_t top, int64_t max_items, const zs_view* act_) {
  zs::begin();
  const zs::View x(x_), gu(gu_), members(members_), items(items_), counts(counts_), act(act_);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous(), "x: bf16 [R, H]");
  ZS_REQUIRE(gu.defined() && gu.dtype() == ZS_BF16 && gu.dim() == 3 && gu.is_contiguous() && gu.size(2) == x.size(1),
             "gu: bf16 [E, 2 I, H]");
  const int64_t hidden = x.size(1), inter = gu.size(1) / 2;
  ZS_REQUIRE(hidden % BK == 0 && gu.size(1) % 2 == 0 && inter % (BN / 2) == 0, "H a multiple of 64, I of 32");
  ZS_REQUIRE(act.defined() && act.dtype() == ZS_BF16 && act.dim() == 2 && act.is_contiguous() &&
                 act.size(1) == inter && act.size(0) >= x.size(0) * top,
             "act: bf16 [pairs, I]");
  ZS_REQUIRE(ints(members) && ints(items) && ints(counts) && items.numel() >= 3 * max_items && top >= 1,
             "the plan: int32 members, items, counts");
  if (max_items == 0) return zs::status();
  const dim3 grid(static_cast<unsigned>(inter / (BN / 2)), static_cast<unsigned>(max_items));
  bf16_experts_kernel<true><<<grid, THREADS, 0, zs::stream()>>>(
      x.data_ptr<__nv_bfloat16>(), gu.data_ptr<__nv_bfloat16>(), members.data_ptr<int>(), items.data_ptr<int>(),
      counts.data_ptr<int>(), act.data_ptr<__nv_bfloat16>(), static_cast<int>(hidden), static_cast<int>(inter),
      static_cast<int>(top));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// act bf16 [pairs, I] in plan order, down bf16 [E, H, I], the plan -> y bf16 [pairs, H] at each pair.
int zs_bf16_experts_down(const zs_view* act_, const zs_view* down_, const zs_view* members_, const zs_view* items_,
                         const zs_view* counts_, int64_t max_items, const zs_view* y_) {
  zs::begin();
  const zs::View act(act_), down(down_), members(members_), items(items_), counts(counts_), y(y_);
  ZS_REQUIRE(act.defined() && act.dtype() == ZS_BF16 && act.dim() == 2 && act.is_contiguous(), "act: bf16 [pairs, I]");
  ZS_REQUIRE(down.defined() && down.dtype() == ZS_BF16 && down.dim() == 3 && down.is_contiguous() &&
                 down.size(2) == act.size(1),
             "down: bf16 [E, H, I]");
  const int64_t inter = act.size(1), hidden = down.size(1);
  ZS_REQUIRE(inter % BK == 0 && hidden % BN == 0, "I a multiple of 64, H of 64");
  ZS_REQUIRE(y.defined() && y.dtype() == ZS_BF16 && y.dim() == 2 && y.is_contiguous() && y.size(1) == hidden &&
                 y.size(0) <= act.size(0),
             "y: bf16 [pairs, H]");
  ZS_REQUIRE(ints(members) && ints(items) && ints(counts) && items.numel() >= 3 * max_items,
             "the plan: int32 members, items, counts");
  if (max_items == 0) return zs::status();
  const dim3 grid(static_cast<unsigned>(hidden / BN), static_cast<unsigned>(max_items));
  bf16_experts_kernel<false><<<grid, THREADS, 0, zs::stream()>>>(
      act.data_ptr<__nv_bfloat16>(), down.data_ptr<__nv_bfloat16>(), members.data_ptr<int>(), items.data_ptr<int>(),
      counts.data_ptr<int>(), y.data_ptr<__nv_bfloat16>(), static_cast<int>(inter), static_cast<int>(hidden), 1);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
