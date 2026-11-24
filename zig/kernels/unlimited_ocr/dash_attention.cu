// Unlimited-OCR-2's DASH (SR-SWA) attend (spec 147 WS-04): the attention of each row over its local window and the
// 32-key blocks its router chose, a bias a chosen block. Plain MHA, 10 heads of 128; q and the caches come rotated
// (page2d RoPE), the block ids and their bias from the router (WS-03).
// Semantics from the checkpoint's own code with Triton off, modeling_deepseekv2.py's `_decode_attend` (a step, fp32)
// and dash_attention.py's `_forward_dense` (a prompt, SDPA under a mask) (Apache-2.0, read 2026-10-27); written for
// zspark in CUDA C++. TensorFold has no such model; SGLang's wheel attends the same set (its Triton gather kernel is
// speed only, the spec's Decisions).
//
// A row at position i sees key j iff j <= i and (j > i - W, its local window, no bias; or j's block j / 32 is one
// of the row's chosen ids, with that slot's bias). A key in the window takes no bias even when its block was chosen
// (HF's `where(local, 0, base)`: the window wins); the router chooses only blocks <= i / 32 - W / 32, so only the
// window's first block can be both. Score = q.k * scale + bias, HF's order (the bias unscaled); kept here in log2
// units, fmaf(q.k, scale log2 e, bias log2 e). An id < 0 is an absent slot (fewer blocks than slots); ids are
// distinct. Departures from HF, kept wider: the prompt's bias stays fp32 (HF's SDPA mask is cast to bf16), and
// scores and weights are fp32 throughout (online softmax), the weights rounded to bf16 only as P.V's operand.
//
// Hand-written because CUTLASS has no attention for sm_121 over a per-row gathered key set with a per-block bias
// (512 keys a row at most, no GEMM shape); FlashAttention-style on m16n8k16 bf16 tensor cores (common/qmm_frag.cuh),
// in the family of unlimited_ocr/attention.cu.
//
// The unit is one aligned block of 32 keys in both kernels. A row's units are its chosen blocks and its window's
// blocks, in ascending block order. For each unit a warp computes the row's partial with the same code
// (`unit_scores`, `unit_weights`, `unit_values`): its largest score x, its weights' sum and its P.V from zero; and
// the partials are folded in ascending order by `fold` from the empty state (max -inf, sum 0, values 0). The tile
// kernel folds as it goes; the step kernel computes each unit in a block of its own and the last block folds them
// in order. So a row's bits are the same in either kernel, alone or among any rows: WS-05 may choose a kernel by
// cost. A unit a row does not take, met in a tile because another row does, folds as the identity (x = -inf: the
// old state times exp2(0) = 1 plus zero), and a warp none of whose rows takes it skips it. Keys past the tile's
// last position are zeros in shared memory, never seen, so a weight of 0 times a value is 0.
//
// Measured (147 WS-04, a quick timing, back to back in L2, not a measurement round; 10 heads, 512 keys a row):
// the step kernel 12.4 us at 1 row, 23.2 at 4, 74.5 at 16; the tile kernel 39.0 at 1 to 4 rows, 51 at 553, 168
// at 2,048. Its error: 1.7e-3 relative rms from fp64 over the fixtures, as attention.cu's (bf16 P and output).
//
// Shared memory, the tile kernel (block: 64 rows of a head, four warps of 16 rows; 30,756 B, three blocks an SM):
//   region  bytes   holds                                                         could be less
//   kv      16,384  a unit's 32 keys and 32 values; the 64 queries before          a second stage (+16 KB) would hide
//                   the first unit                                                 the load; single-staged as v1
//   ids      4,096  the rows' chosen ids, 64 x 16 slots (12 used)                 12 slots: 3,072
//   bias     4,096  their bias in log2 units                                       12 slots: 3,072
//   marks    2,048  a bit a block for the union of the rows' units (16,384 blocks, 524,288 positions)
//   list     4,132  the union in ascending order (64 x 16 + 8 at most), its count 12 slots: 3,108
// The step kernel (block: one unit of one row and head, one warp; 16,473 B, six blocks an SM): kv (the query's
// 16-row A operand,
// one real row, staged in its keys' half before), the row's sorted units (84 B) and the last-block flag.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/copies.cuh"
#include "../common/math.cuh"
#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

constexpr int D = 128;            // a head's values
constexpr int BLK = 32;           // keys a unit: the router's block
constexpr int ROWS = 64;          // query rows a tile block
constexpr int THREADS = 128;      // four warps of 16 rows
constexpr int CHUNKS = D / 8;     // 16-byte chunks a row of a head
constexpr int MAXSEL = 16;        // chosen slots a row at most
constexpr int MAXW = 128;         // the window at most
constexpr int MARKS = 512;        // words of block marks: 16,384 blocks
constexpr int SEGS = MAXSEL + MAXW / BLK + 1;   // a row's units at most (21)
constexpr int LIST = ROWS * MAXSEL + (ROWS + MAXW) / BLK + 2;
constexpr int PART = D + 2;       // a unit's partial: max, sum, then D values (fp32)
constexpr float LOG2E = 1.4426950408889634f;

using common_copies::ldmatrix4t;

// Chunk c of row r at chunk c ^ (r % 8): ldmatrix's eight rows in eight bank groups.
__device__ __forceinline__ int at(int r, int c) { return r * D + ((c ^ (r & 7)) << 3); }

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// One row's view of a unit: its position, whether it chose the unit's block and with what bias (log2 units).
struct Seen {
  long long pos;
  bool chosen;
  float bias;
};

// The warp's 16 x 32 raw scores q.k over the unit's keys in ks.
__device__ __forceinline__ void unit_scores(const uint32_t (&qf)[D / 16][4], const __nv_bfloat16* ks, int lane,
                                            float (&s)[BLK / 8][4]) {
#pragma unroll
  for (int j = 0; j < BLK / 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) {
#pragma unroll
    for (int np = 0; np < BLK / 16; ++np) {
      uint32_t b[4];
      qmm_frag::ldmatrix4(b, &ks[at(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1))]);
      qmm_frag::mma(s[2 * np], qf[kk], b[0], b[1]);
      qmm_frag::mma(s[2 * np + 1], qf[kk], b[2], b[3]);
    }
  }
}

// Masks and biases the scores, then each of the thread's two rows' largest x and weights' sum (over the row's
// quad, lanes in a fixed order); s becomes the weights exp2(score - x).
__device__ __forceinline__ void unit_weights(float (&s)[BLK / 8][4], long long key0, int t, const Seen& a,
                                             const Seen& b, long long W, float scale_log2, float& x_a, float& x_b,
                                             float& sum_a, float& sum_b) {
  x_a = minus_inf();
  x_b = minus_inf();
#pragma unroll
  for (int j = 0; j < BLK / 8; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const Seen& r = e < 2 ? a : b;
      const long long key = key0 + j * 8 + 2 * t + (e & 1);
      const bool local = key <= r.pos && key > r.pos - W;
      const bool seen = local || (r.chosen && key <= r.pos);
      s[j][e] = seen ? __fmaf_rn(s[j][e], scale_log2, local ? 0.f : r.bias) : minus_inf();
    }
    x_a = fmaxf(x_a, fmaxf(s[j][0], s[j][1]));
    x_b = fmaxf(x_b, fmaxf(s[j][2], s[j][3]));
  }
  x_a = fmaxf(x_a, __shfl_xor_sync(0xffffffffu, x_a, 1));
  x_a = fmaxf(x_a, __shfl_xor_sync(0xffffffffu, x_a, 2));
  x_b = fmaxf(x_b, __shfl_xor_sync(0xffffffffu, x_b, 1));
  x_b = fmaxf(x_b, __shfl_xor_sync(0xffffffffu, x_b, 2));
  // a row that sees none of the unit's keys: every weight exp2(-inf) = 0
  const float base_a = x_a == minus_inf() ? 0.f : x_a, base_b = x_b == minus_inf() ? 0.f : x_b;
  sum_a = 0.f;
  sum_b = 0.f;
#pragma unroll
  for (int j = 0; j < BLK / 8; ++j) {
    s[j][0] = exp2f(__fsub_rn(s[j][0], base_a));
    s[j][1] = exp2f(__fsub_rn(s[j][1], base_a));
    s[j][2] = exp2f(__fsub_rn(s[j][2], base_b));
    s[j][3] = exp2f(__fsub_rn(s[j][3], base_b));
    sum_a = __fadd_rn(sum_a, __fadd_rn(s[j][0], s[j][1]));
    sum_b = __fadd_rn(sum_b, __fadd_rn(s[j][2], s[j][3]));
  }
  sum_a = __fadd_rn(sum_a, __shfl_xor_sync(0xffffffffu, sum_a, 1));
  sum_a = __fadd_rn(sum_a, __shfl_xor_sync(0xffffffffu, sum_a, 2));
  sum_b = __fadd_rn(sum_b, __shfl_xor_sync(0xffffffffu, sum_b, 1));
  sum_b = __fadd_rn(sum_b, __shfl_xor_sync(0xffffffffu, sum_b, 2));
}

// The weights as P.V's A operand (bf16), two k16 steps over the unit's 32 keys.
__device__ __forceinline__ void unit_operand(const float (&s)[BLK / 8][4], uint32_t (&p)[BLK / 16][4]) {
#pragma unroll
  for (int kk = 0; kk < BLK / 16; ++kk) {
    p[kk][0] = pack(s[2 * kk][0], s[2 * kk][1]);
    p[kk][1] = pack(s[2 * kk][2], s[2 * kk][3]);
    p[kk][2] = pack(s[2 * kk + 1][0], s[2 * kk + 1][1]);
    p[kk][3] = pack(s[2 * kk + 1][2], s[2 * kk + 1][3]);
  }
}

// Value columns 16 dp .. 16 dp + 15 of the unit's P.V, from zero: n-tiles 2 dp and 2 dp + 1.
__device__ __forceinline__ void unit_values(const uint32_t (&p)[BLK / 16][4], const __nv_bfloat16* vs, int lane,
                                            int dp, float (&o)[2][4]) {
#pragma unroll
  for (int kk = 0; kk < BLK / 16; ++kk) {
    uint32_t b[4];
    ldmatrix4t(b, &vs[at(kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3), dp * 2 + (lane >> 4))]);
    if (kk == 0) {
      qmm_frag::mma0(o[0], p[kk], b[0], b[1]);
      qmm_frag::mma0(o[1], p[kk], b[2], b[3]);
    } else {
      qmm_frag::mma(o[0], p[kk], b[0], b[1]);
      qmm_frag::mma(o[1], p[kk], b[2], b[3]);
    }
  }
}

// The scales that fold a unit's partial (largest x) into a running state (largest m): old * f_old + part * f_new.
// The empty state (m = -inf, its sum and values 0) keeps factor 1; a unit that saw nothing (x = -inf) gives 0.
__device__ __forceinline__ void fold_scales(float m, float x, float& most, float& f_old, float& f_new) {
  most = fmaxf(m, x);
  f_old = m == minus_inf() ? 1.f : exp2f(__fsub_rn(m, most));
  f_new = x == minus_inf() ? 0.f : exp2f(__fsub_rn(x, most));
}

__device__ __forceinline__ float fold(float old, float f_old, float part, float f_new) {
  return __fmaf_rn(old, f_old, __fmul_rn(part, f_new));
}

__device__ __forceinline__ float finish(float value, float sum) { return sum > 0.f ? __fdiv_rn(value, sum) : 0.f; }

// Block (row tile, head). Q [R, H, D] at positions POS0 .. POS0 + R - 1 (AT[0]); the caches [cap, H, D] hold
// 0 .. POS0 + R - 1; IDS, BIAS [R, SEL]; out [R, H D] (row stride OS).
__global__ void __launch_bounds__(THREADS)
    unlimited_ocr_dash_tile_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                   const __nv_bfloat16* __restrict__ VC, const int* __restrict__ IDS,
                                   const float* __restrict__ BIAS, int SEL, __nv_bfloat16* __restrict__ OUT,
                                   long long OS, int R, int H, const int* __restrict__ AT, int W, float scale_log2) {
  __shared__ __align__(128) __nv_bfloat16 kv[2 * BLK * D];
  __shared__ int ids[ROWS][MAXSEL];
  __shared__ float bias[ROWS][MAXSEL];
  __shared__ unsigned marks[MARKS];
  __shared__ int list[LIST];
  __shared__ int listed;
  __nv_bfloat16* ks = kv;
  __nv_bfloat16* vs = kv + BLK * D;
  const int h = blockIdx.y, row0 = blockIdx.x * ROWS, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const long long pos0 = AT[0];
  const int last_row = min(R, row0 + ROWS) - 1;
  const long long first_pos = pos0 + row0, last_key = pos0 + last_row;
  const int last_block = static_cast<int>(last_key / BLK);

  // the block's queries, staged in kv; rows past R are zeros
  for (int i = tid; i < ROWS * CHUNKS; i += THREADS) {
    const int rr = i / CHUNKS, c = i % CHUNKS;
    const bool real = row0 + rr < R;
    qmm_frag::cp16z(&kv[at(rr, c)], Q + ((size_t)(real ? row0 + rr : 0) * H + h) * D + c * 8, real);
  }
  qmm_frag::commit();
  for (int i = tid; i < MARKS; i += THREADS) marks[i] = 0u;
  __syncthreads();
  // the rows' chosen blocks and the window's, marked; an id past the tile's last block no row sees
  for (int i = tid; i < ROWS * MAXSEL; i += THREADS) {
    const int rr = i / MAXSEL, k = i % MAXSEL;
    int id = -1;
    float b = 0.f;
    if (row0 + rr < R && k < SEL) {
      id = IDS[(size_t)(row0 + rr) * SEL + k];
      b = __fmul_rn(BIAS[(size_t)(row0 + rr) * SEL + k], LOG2E);
    }
    ids[rr][k] = id;
    bias[rr][k] = b;
    if (id >= 0 && id <= last_block) atomicOr(&marks[id >> 5], 1u << (id & 31));
  }
  const int local0 = static_cast<int>(max(first_pos - W + 1, 0LL) / BLK);
  for (int b = local0 + tid; b <= last_block; b += THREADS) atomicOr(&marks[b >> 5], 1u << (b & 31));
  __syncthreads();
  // the marks in ascending order: warp 0, a lane a run of words, its count's exclusive scan over the lanes
  if (warp == 0) {
    const int words = last_block / 32 + 1, per = (words + 31) / 32;
    const int w0 = min(words, lane * per), w1 = min(words, w0 + per);
    int count = 0;
    for (int w = w0; w < w1; ++w) count += __popc(marks[w]);
    int before = count;
#pragma unroll
    for (int x = 1; x < 32; x <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, before, x);
      if (lane >= x) before += y;
    }
    if (lane == 31) listed = before;
    before -= count;
    for (int w = w0; w < w1; ++w)
      for (unsigned bits = marks[w]; bits != 0u; bits &= bits - 1u) list[before++] = w * 32 + __ffs(bits) - 1;
  }
  qmm_frag::wait<0>();
  __syncthreads();
  uint32_t qf[D / 16][4];
  for (int kk = 0; kk < D / 16; ++kk) qmm_frag::ldmatrix4(qf[kk], &kv[at(warp * 16 + (lane & 15), kk * 2 + (lane >> 4))]);
  const int units = listed;
  __syncthreads();

  // this thread's two rows: g and g + 8 of its warp's 16
  const int ra = warp * 16 + g, rb = ra + 8;
  const bool real_a = row0 + ra < R, real_b = row0 + rb < R;
  Seen sa{first_pos + ra, false, 0.f}, sb{first_pos + rb, false, 0.f};
  float o[D / 8][4];
#pragma unroll
  for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
  float m_a = minus_inf(), m_b = minus_inf(), l_a = 0.f, l_b = 0.f;
  for (int u = 0; u < units; ++u) {
    const int blk = list[u];
    const long long key0 = (long long)blk * BLK;
    for (int i = tid; i < BLK * CHUNKS; i += THREADS) {
      const int kr = i / CHUNKS, c = i % CHUNKS;
      const long long key = key0 + kr;
      const bool real = key <= last_key;
      const size_t from = ((size_t)(real ? key : 0) * H + h) * D + c * 8;
      qmm_frag::cp16z(&ks[at(kr, c)], KC + from, real);
      qmm_frag::cp16z(&vs[at(kr, c)], VC + from, real);
    }
    qmm_frag::commit();
    // which of the thread's rows take the unit, and with what bias
    sa.chosen = sb.chosen = false;
    for (int k = 0; k < SEL; ++k) {
      if (ids[ra][k] == blk) {
        sa.chosen = true;
        sa.bias = bias[ra][k];
      }
      if (ids[rb][k] == blk) {
        sb.chosen = true;
        sb.bias = bias[rb][k];
      }
    }
    const bool takes_a = real_a && (sa.chosen || (blk <= sa.pos / BLK && blk >= max(sa.pos - W + 1, 0LL) / BLK));
    const bool takes_b = real_b && (sb.chosen || (blk <= sb.pos / BLK && blk >= max(sb.pos - W + 1, 0LL) / BLK));
    qmm_frag::wait<0>();
    __syncthreads();
    if (__any_sync(0xffffffffu, takes_a || takes_b)) {
      float s[BLK / 8][4];
      unit_scores(qf, ks, lane, s);
      float x_a, x_b, sum_a, sum_b;
      unit_weights(s, key0, t, sa, sb, W, scale_log2, x_a, x_b, sum_a, sum_b);
      float n_a, fo_a, fn_a, n_b, fo_b, fn_b;
      fold_scales(m_a, x_a, n_a, fo_a, fn_a);
      fold_scales(m_b, x_b, n_b, fo_b, fn_b);
      l_a = fold(l_a, fo_a, sum_a, fn_a);
      l_b = fold(l_b, fo_b, sum_b, fn_b);
      m_a = n_a;
      m_b = n_b;
      uint32_t p[BLK / 16][4];
      unit_operand(s, p);
#pragma unroll
      for (int dp = 0; dp < D / 16; ++dp) {
        float v[2][4];
        unit_values(p, vs, lane, dp, v);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          o[2 * dp + n][0] = fold(o[2 * dp + n][0], fo_a, v[n][0], fn_a);
          o[2 * dp + n][1] = fold(o[2 * dp + n][1], fo_a, v[n][1], fn_a);
          o[2 * dp + n][2] = fold(o[2 * dp + n][2], fo_b, v[n][2], fn_b);
          o[2 * dp + n][3] = fold(o[2 * dp + n][3], fo_b, v[n][3], fn_b);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < D / 8; ++j) {
    const int col = h * D + j * 8 + 2 * t;
    if (real_a)
      *reinterpret_cast<__nv_bfloat162*>(OUT + (size_t)(row0 + ra) * OS + col) =
          __floats2bfloat162_rn(finish(o[j][0], l_a), finish(o[j][1], l_a));
    if (real_b)
      *reinterpret_cast<__nv_bfloat162*>(OUT + (size_t)(row0 + rb) * OS + col) =
          __floats2bfloat162_rn(finish(o[j][2], l_b), finish(o[j][3], l_b));
  }
}

// Block (unit slot, head, row), one warp: the row's units in ascending order (its chosen ids, its window's blocks;
// at most SEGS), this block's slot's partial to PARTS, and the block that finishes last, by a counter a (row, head),
// folds the row's partials in slot order and resets the counter. A slot past the row's units writes nothing.
__global__ void __launch_bounds__(32)
    unlimited_ocr_dash_step_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                                   const __nv_bfloat16* __restrict__ VC, const int* __restrict__ IDS,
                                   const float* __restrict__ BIAS, int SEL, __nv_bfloat16* __restrict__ OUT,
                                   long long OS, int H, const int* __restrict__ AT, int W, float scale_log2,
                                   float* __restrict__ PARTS, int* __restrict__ DONE, int SLOTS) {
  __shared__ __align__(128) __nv_bfloat16 kv[2 * BLK * D];
  __shared__ int units[SEGS];
  __shared__ int count;
  __shared__ bool last;
  __nv_bfloat16* ks = kv;
  __nv_bfloat16* vs = kv + BLK * D;
  const int slot = blockIdx.x, h = blockIdx.y, r = blockIdx.z, lane = threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const long long pos = (long long)AT[0] + r;
  const int top = static_cast<int>(pos / BLK), local0 = static_cast<int>(max(pos - W + 1, 0LL) / BLK);
  const size_t pair = (size_t)r * H + h;

  // the query as row 0 of the A operand, rows 1 to 15 zeros
  for (int i = lane; i < 16 * CHUNKS; i += 32) {
    const int rr = i / CHUNKS, c = i % CHUNKS;
    qmm_frag::cp16z(&ks[at(rr, c)], Q + (pair * D + c * 8), rr == 0);
  }
  qmm_frag::commit();
  // the row's units: chosen ids before the window (sorted), the window's blocks, chosen ids past the row (none
  // in HF's choice; folded as the identity in the tile kernel, so left out here alike)
  if (lane == 0) {
    int n = 0;
    for (int k = 0; k < SEL; ++k) {
      const int id = IDS[(size_t)r * SEL + k];
      if (id < 0 || id >= local0) continue;
      int i = n++;
      for (; i > 0 && units[i - 1] > id; --i) units[i] = units[i - 1];
      units[i] = id;
    }
    for (int b = local0; b <= top; ++b) units[n++] = b;
    count = n;
  }
  qmm_frag::wait<0>();
  __syncwarp();
  uint32_t qf[D / 16][4];
  for (int kk = 0; kk < D / 16; ++kk) qmm_frag::ldmatrix4(qf[kk], &ks[at(lane & 15, kk * 2 + (lane >> 4))]);
  __syncwarp();
  const int n = count;
  float* parts = PARTS + pair * SLOTS * PART;

  if (slot < n) {
    const int blk = units[slot];
    const long long key0 = (long long)blk * BLK;
    for (int i = lane; i < BLK * CHUNKS; i += 32) {
      const int kr = i / CHUNKS, c = i % CHUNKS;
      const long long key = key0 + kr;
      const bool real = key <= pos;
      const size_t from = ((size_t)(real ? key : 0) * H + h) * D + c * 8;
      qmm_frag::cp16z(&ks[at(kr, c)], KC + from, real);
      qmm_frag::cp16z(&vs[at(kr, c)], VC + from, real);
    }
    qmm_frag::commit();
    Seen sa{pos, false, 0.f}, sb{pos, false, 0.f};
    for (int k = 0; k < SEL; ++k)
      if (IDS[(size_t)r * SEL + k] == blk) {
        sa.chosen = true;
        sa.bias = __fmul_rn(BIAS[(size_t)r * SEL + k], LOG2E);
      }
    qmm_frag::wait<0>();
    __syncwarp();
    float s[BLK / 8][4];
    unit_scores(qf, ks, lane, s);
    float x_a, x_b, sum_a, sum_b;
    unit_weights(s, key0, t, sa, sb, W, scale_log2, x_a, x_b, sum_a, sum_b);
    uint32_t p[BLK / 16][4];
    unit_operand(s, p);
    float* part = parts + (size_t)slot * PART;
#pragma unroll
    for (int dp = 0; dp < D / 16; ++dp) {
      float v[2][4];
      unit_values(p, vs, lane, dp, v);
      if (g == 0) {
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
          part[2 + (2 * dp + nn) * 8 + 2 * t] = v[nn][0];
          part[2 + (2 * dp + nn) * 8 + 2 * t + 1] = v[nn][1];
        }
      }
    }
    if (lane == 0) {
      part[0] = x_a;
      part[1] = sum_a;
    }
  }
  __threadfence();
  __syncwarp();
  if (lane == 0) last = atomicAdd(&DONE[pair], 1) == SLOTS - 1;
  __syncwarp();
  if (!last) return;
  __threadfence();
  // the last block folds the row's partials in order, a lane four values
  float m = minus_inf(), l = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c = 0; c < n; ++c) {
    const float* part = parts + (size_t)c * PART;
    float most, f_old, f_new;
    fold_scales(m, __ldcg(part), most, f_old, f_new);
    l = fold(l, f_old, __ldcg(part + 1), f_new);
#pragma unroll
    for (int d = 0; d < 4; ++d) o[d] = fold(o[d], f_old, __ldcg(part + 2 + 4 * lane + d), f_new);
    m = most;
  }
  __nv_bfloat16* out = OUT + (size_t)r * OS + (size_t)h * D + 4 * lane;
  reinterpret_cast<__nv_bfloat162*>(out)[0] = __floats2bfloat162_rn(finish(o[0], l), finish(o[1], l));
  reinterpret_cast<__nv_bfloat162*>(out)[1] = __floats2bfloat162_rn(finish(o[2], l), finish(o[3], l));
  if (lane == 0) DONE[pair] = 0;
}

bool cache_of(const zs::View& c, int64_t heads) {
  return c.defined() && c.dtype() == ZS_BF16 && c.dim() == 3 && c.is_contiguous() && c.size(1) == heads &&
         c.size(2) == D && c.size(0) <= (int64_t)MARKS * 32 * BLK;
}

// The arguments both entry points take; 0 when they hold.
int check(const zs::View& q, const zs::View& kc, const zs::View& vc, const zs::View& at, const zs::View& ids,
          const zs::View& bias, int64_t window, const zs::View& out) {
  ZS_REQUIRE(q.defined() && q.dtype() == ZS_BF16 && q.dim() == 3 && q.is_contiguous() && q.size(2) == D,
             "q: bf16 [R, H, 128]");
  const int64_t rows = q.size(0), heads = q.size(1);
  ZS_REQUIRE(rows >= 1 && rows <= 65535 && heads >= 1 && heads <= 65535, "q: rows and heads");
  ZS_REQUIRE(cache_of(kc, heads) && cache_of(vc, heads) && vc.size(0) == kc.size(0),
             "caches: bf16 [cap, H, 128], cap at most 524,288");
  ZS_REQUIRE(at.defined() && at.dtype() == ZS_I32 && at.numel() >= 1 && at.is_contiguous(), "at: int32 [1]");
  ZS_REQUIRE(rows <= kc.size(0), "rows past the cache's capacity");
  ZS_REQUIRE(ids.defined() && ids.dtype() == ZS_I32 && ids.dim() == 2 && ids.is_contiguous() && ids.size(0) == rows &&
                 ids.size(1) >= 1 && ids.size(1) <= MAXSEL,
             "ids: int32 [R, at most 16]");
  ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_F32 && bias.dim() == 2 && bias.is_contiguous() &&
                 bias.size(0) == rows && bias.size(1) == ids.size(1),
             "bias: fp32 [R, ids' slots]");
  ZS_REQUIRE(window >= 1 && window <= MAXW, "a window of 1 to 128");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.dim() == 2 && out.size(0) == rows &&
                 out.size(1) == heads * D && out.stride(1) == 1,
             "out: bf16 [R, H 128]");
  return 0;
}

}  // namespace

extern "C" {

// q bf16 [R, H, 128] at positions pos0 .. pos0 + R - 1 (at int32 [1], read on the device), the caches bf16
// [cap, H, 128] holding positions 0 .. pos0 + R - 1, ids int32 [R, S] (a slot < 0 absent) and bias fp32 [R, S] the
// router's, window W, scale the scores' (1 / sqrt(128)) -> out bf16 [R, H 128] (rows may be strided). The tile
// kernel, a block 64 rows: for a prompt's chunk. Its rows' bits are the step kernel's.
int zs_unlimited_ocr_dash_tile(const zs_view* q_, const zs_view* kc_, const zs_view* vc_, const zs_view* at_,
                               const zs_view* ids_, const zs_view* bias_, int64_t window, double scale,
                               const zs_view* out_) {
  zs::begin();
  const zs::View q(q_), kc(kc_), vc(vc_), at(at_), ids(ids_), bias(bias_), out(out_);
  if (check(q, kc, vc, at, ids, bias, window, out) != 0) return zs::status();
  const int64_t rows = q.size(0), heads = q.size(1);
  const float scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  const dim3 grid(static_cast<unsigned>((rows + ROWS - 1) / ROWS), static_cast<unsigned>(heads));
  unlimited_ocr_dash_tile_kernel<<<grid, THREADS, 0, zs::stream()>>>(
      q.data_ptr<__nv_bfloat16>(), kc.data_ptr<__nv_bfloat16>(), vc.data_ptr<__nv_bfloat16>(), ids.data_ptr<int>(),
      bias.data_ptr<float>(), static_cast<int>(ids.size(1)), out.data_ptr<__nv_bfloat16>(), out.row_stride(),
      static_cast<int>(rows), static_cast<int>(heads), at.data_ptr<int>(), static_cast<int>(window), scale_log2);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// The same rows and arguments on the step kernel, a block a unit of a (row, head): for steps and verify windows.
// parts fp32 [R H, slots, 130] or more and done int32 [R H] or more, done zero before the first call and left zero
// by each; slots = S + ceil(W / 32) + 1 (`dash_attention.zig`'s `slots`).
int zs_unlimited_ocr_dash_step(const zs_view* q_, const zs_view* kc_, const zs_view* vc_, const zs_view* at_,
                               const zs_view* ids_, const zs_view* bias_, int64_t window, double scale,
                               const zs_view* parts_, const zs_view* done_, const zs_view* out_) {
  zs::begin();
  const zs::View q(q_), kc(kc_), vc(vc_), at(at_), ids(ids_), bias(bias_), out(out_), parts(parts_), done(done_);
  if (check(q, kc, vc, at, ids, bias, window, out) != 0) return zs::status();
  const int64_t rows = q.size(0), heads = q.size(1);
  const int64_t slots = ids.size(1) + (window + BLK - 1) / BLK + 1;
  ZS_REQUIRE(parts.defined() && parts.dtype() == ZS_F32 && parts.is_contiguous() &&
                 parts.numel() >= rows * heads * slots * PART,
             "parts: fp32 [R H, slots, 130]");
  ZS_REQUIRE(done.defined() && done.dtype() == ZS_I32 && done.is_contiguous() && done.numel() >= rows * heads,
             "done: int32 [R H]");
  const float scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  const dim3 grid(static_cast<unsigned>(slots), static_cast<unsigned>(heads), static_cast<unsigned>(rows));
  unlimited_ocr_dash_step_kernel<<<grid, 32, 0, zs::stream()>>>(
      q.data_ptr<__nv_bfloat16>(), kc.data_ptr<__nv_bfloat16>(), vc.data_ptr<__nv_bfloat16>(), ids.data_ptr<int>(),
      bias.data_ptr<float>(), static_cast<int>(ids.size(1)), out.data_ptr<__nv_bfloat16>(), out.row_stride(),
      static_cast<int>(heads), at.data_ptr<int>(), static_cast<int>(window), scale_log2, parts.data_ptr<float>(),
      done.data_ptr<int>(), static_cast<int>(slots));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
