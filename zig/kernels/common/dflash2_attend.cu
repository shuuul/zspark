// DFlash2 drafter: a block's attention over the context's keys and values, an online softmax on
// the tensor cores. For the query of head h and row r, at position POS + r:
//   out = bf16(softmax((q . k) / sqrt(128) over the keys it sees) . v)
// A query sees the context from its position - WINDOW on and the whole block (to its own row when causal). Key
// kk is row kk of its KV head's K and V [capacity, 128], or row kk mod capacity in a ring.
// GLM-5.3-Flash and Qwen3.8 Flash Next both launch this. Semantics from TensorFold 0.6.6 (cb2ebf0)
// families/glm5_next/cuda/dflash2.py (_dattn_kernel; Triton), Apache-2.0; rewritten for zspark in CUDA C++ on
// zig/kernels/glm5/mla.cu's steps (K and V apart, 128 wide): as TensorFold's, tiles of keys with a running
// maximum and probabilities rounded to bf16 for the second product.
// zspark's: the tile is 32 keys, the launch's keys are split in chunks of CHUNK by position and the chunks
// merged in chunk order, and the reductions' orders are those stated at the kernel.
// Measured (spec 067): at the published shapes (a rank's 16 heads, 8 rows, 2,055 keys) 0.024 ms a launch,
// where a kernel that kept a query's scores and summed in fp32 (a thread a value, no tensor core) took 0.13.
// A query's values come from its own row of Q, its position and the keys it sees by position: the chunks and
// tiles count from the first position the launch's first row sees, so neither the ring nor the capacity
// changes a bit. A key a query does not see is read when another row of the launch sees it and has p = 0.
// Hand-written because the attention is not a GEMM CUTLASS has (zig/kernels/glm5/mla.cu's header).
// C ABI names are `zs_dflash2_*`, model-free (spec 136 WS-03).

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "copies.cuh"
#include "math.cuh"
#include "qmm_frag.cuh"
#include "../zs/host.h"

namespace {

using namespace common_copies;

using namespace common_math;

constexpr int HD = 128;
constexpr int CHUNK = 256;            // keys a chunk block attends: 128 and 256 measured alike, 512 a fifth slower
constexpr int TILE = 32;              // keys a step of the online softmax takes
constexpr int ROWS = 16;              // an mma's rows: queries of one KV head
constexpr int THREADS = 128;          // a chunk block's four warps: 8 keys of a tile each, a quarter of a value each
constexpr int FR = HD / 32;           // the 8-value fragments of a value a warp carries
constexpr int DC = HD / 8;            // 16-byte pieces of a row
// a chunk block's shared memory: the queries, a tile's keys and values, a tile's bf16(p), the warps' max and sum
constexpr int SMEM = ROWS * HD * 2 + 2 * TILE * HD * 2 + ROWS * TILE * 2 + 2 * 4 * ROWS * 4;

// Block (KV head, group of 16 of its queries, chunk) of 4 warps. Query qi of a KV head is its query head qi / N,
// row qi % N; a group past them holds zero queries. The launch's keys are the positions base .. POS + N - 1 in
// a row, base the first position row 0 sees; chunk c is keys [c CHUNK, (c + 1) CHUNK) of them, a TILE at a
// time. Each tile, as mla.cu's:
//   score = (q . k) * SCALE, one fp32 chain of m16n8k16 steps over the head, 16 values at a step, -inf for a
//   key the query does not see;
//   m' = max(m, tile max); p = exp(score - m'); l = l exp(m - m') + sum p;
//   o = o exp(m - m') + (bf16(p) . v, one fp32 chain of steps over the tile's keys, 16 at a step).
// sum p adds a thread's two keys, the four threads of a warp's 8 keys as a tree, then the four warps in order.
// An mma's element is a function of its own row of A and column of B, so a query's values depend on neither
// the queries beside it nor the warp that holds them. The chunk's o, m and l go to PO [H, N, NCH, 128], PM and
// PL [H, N, NCH]; a chunk past the keys writes nothing. Shared rows are swizzled by 16-byte pieces.
__global__ void __launch_bounds__(THREADS)
    dflash2_chunk_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                 const __nv_bfloat16* __restrict__ V, const int* __restrict__ POS, float* __restrict__ PO,
                 float* __restrict__ PM, float* __restrict__ PL, float SCALE, int N, int G, int CAP, int WINDOW,
                 int CAUSAL, int RING, int NCH, int GROUPS) {
  extern __shared__ __align__(16) unsigned char shared[];
  __nv_bfloat16* const q = reinterpret_cast<__nv_bfloat16*>(shared);   // [ROWS, HD]
  __nv_bfloat16* const kt = q + ROWS * HD;                             // [TILE, HD]
  __nv_bfloat16* const vt = kt + TILE * HD;                            // [TILE, HD]
  __nv_bfloat16* const p = vt + TILE * HD;                             // [ROWS, TILE]
  float* const part_m = reinterpret_cast<float*>(p + ROWS * TILE);     // [4, ROWS]
  float* const part_l = part_m + 4 * ROWS;                             // [4, ROWS]
  const int kvh = blockIdx.x / GROUPS, group = blockIdx.x % GROUPS, c = blockIdx.y, tid = threadIdx.x;
  const long long s = POS[0];
  const long long base = max(min(s - WINDOW, s), 0LL);
  const int total = static_cast<int>(s + N - 1 - base + 1);
  const int start = c * CHUNK;
  if (start >= total) return;
  const int first = static_cast<int>(RING ? base % CAP : base);
  const int valid = min(ROWS, G * N - group * ROWS);
  const int warp = tid / 32, lane = tid % 32, row8 = lane >> 2, pair = lane & 3;
  const __nv_bfloat16* keys = K + (size_t)kvh * CAP * HD;
  const __nv_bfloat16* values = V + (size_t)kvh * CAP * HD;
  for (int i = tid; i < ROWS * DC; i += THREADS) {
    const int g = i / DC, ch = i % DC;
    uint4 x = make_uint4(0, 0, 0, 0);
    if (g < valid) {
      const int qi = group * ROWS + g;
      const int qh = kvh * G + qi / N, row = qi % N;
      x = *reinterpret_cast<const uint4*>(Q + ((size_t)qh * N + row) * HD + ch * 8);
    }
    *reinterpret_cast<uint4*>(q + g * HD + (ch ^ (g % 8)) * 8) = x;
  }
  for (int i = tid; i < ROWS * TILE / 8; i += THREADS) *reinterpret_cast<uint4*>(p + i * 8) = make_uint4(0, 0, 0, 0);
  float o[FR][4];
  for (int f = 0; f < FR; ++f)
    for (int e = 0; e < 4; ++e) o[f][e] = 0.f;
  // of queries row8 and row8 + 8: the first and last key index each sees
  int lo[2], hi[2];
  for (int h = 0; h < 2; ++h) {
    const int row = (group * ROWS + row8 + 8 * h) % N;
    const long long qpos = s + row;
    lo[h] = static_cast<int>(max(min(qpos - WINDOW, s), 0LL) - base);
    hi[h] = static_cast<int>((CAUSAL ? qpos : s + N - 1) - base);
  }
  float m[2] = {minus_inf(), minus_inf()}, l[2] = {0.f, 0.f}, alpha[2] = {1.f, 1.f};
  const int count = min(total - start, CHUNK);
  for (int t0 = 0; t0 < count; t0 += TILE) {
    for (int i = tid; i < TILE * DC; i += THREADS) {
      const int key = i / DC, ch = i % DC, ki = start + t0 + key;
      int at = first + (ki < total ? ki : 0);
      if (RING && at >= CAP) at -= CAP;
      qmm_frag::cp16z(kt + key * HD + (ch ^ (key % 8)) * 8, keys + (size_t)at * HD + ch * 8, ki < total);
      qmm_frag::cp16z(vt + key * HD + (ch ^ (key % 8)) * 8, values + (size_t)at * HD + ch * 8, ki < total);
    }
    qmm_frag::commit();
    qmm_frag::wait<0>();
    __syncthreads();
    {
      float c0[4] = {0.f, 0.f, 0.f, 0.f};
      const int head = lane % 16, key = warp * 8 + (lane & 7);
      for (int step = 0; step < DC / 2; ++step) {
        uint32_t a[4], b[2];
        qmm_frag::ldmatrix4(a, q + head * HD + ((step * 2 + (lane >> 4)) ^ (head % 8)) * 8);
        qmm_frag::ldmatrix2(b, kt + key * HD + ((step * 2 + ((lane >> 3) & 1)) ^ (key % 8)) * 8);
        qmm_frag::mma(c0, a, b[0], b[1]);
      }
      const int k0 = warp * 8 + pair * 2;
      const int mine[2] = {k0, k0 + 1};
      float sc[2][2] = {{c0[0], c0[1]}, {c0[2], c0[3]}};
      bool live[2][2];
      for (int h = 0; h < 2; ++h)
        for (int j = 0; j < 2; ++j) {
          const int ki = start + t0 + mine[j];
          live[h][j] = ki >= lo[h] && ki <= hi[h];
          sc[h][j] = live[h][j] ? sc[h][j] * SCALE : minus_inf();
        }
      for (int h = 0; h < 2; ++h) {
        float top = fmaxf(sc[h][0], sc[h][1]);
        top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
        top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
        if (pair == 0) part_m[warp * ROWS + row8 + 8 * h] = top;
      }
      __syncthreads();
      for (int h = 0; h < 2; ++h) {
        const int at = row8 + 8 * h;
        const float tile_m = fmaxf(fmaxf(part_m[at], part_m[ROWS + at]), fmaxf(part_m[2 * ROWS + at], part_m[3 * ROWS + at]));
        const bool active = tile_m != minus_inf();
        const float next_m = active ? fmaxf(m[h], tile_m) : m[h];
        alpha[h] = active ? (m[h] == minus_inf() ? 0.f : expf(m[h] - next_m)) : 1.f;
        m[h] = next_m;
        float pk[2];
        for (int j = 0; j < 2; ++j) pk[j] = active && live[h][j] ? expf(sc[h][j] - next_m) : 0.f;
        float sum = pk[0] + pk[1];
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        if (pair == 0) part_l[warp * ROWS + at] = sum;
        if (at < valid)
          for (int j = 0; j < 2; ++j) p[at * TILE + ((mine[j] / 8) ^ (at % 4)) * 8 + mine[j] % 8] = __float2bfloat16_rn(pk[j]);
      }
    }
    __syncthreads();
    {
      float acc[FR][4];
      for (int f = 0; f < FR; ++f)
        for (int e = 0; e < 4; ++e) acc[f][e] = 0.f;
      const int head = lane % 16;
      for (int step = 0; step < TILE / 16; ++step) {
        uint32_t a[4];
        qmm_frag::ldmatrix4(a, p + head * TILE + ((step * 2 + (lane >> 4)) ^ (head % 4)) * 8);
        const int key = step * 16 + (lane & 15);
        for (int f = 0; f < FR; ++f) {
          uint32_t b[2];
          ldmatrix2t(b, vt + key * HD + ((warp * FR + f) ^ (key % 8)) * 8);
          qmm_frag::mma(acc[f], a, b[0], b[1]);
        }
      }
      for (int h = 0; h < 2; ++h) {
        const int at = row8 + 8 * h;
        l[h] = l[h] * alpha[h] + (((part_l[at] + part_l[ROWS + at]) + part_l[2 * ROWS + at]) + part_l[3 * ROWS + at]);
      }
      const float a_low = alpha[0], a_high = alpha[1];
      for (int f = 0; f < FR; ++f) {
        o[f][0] = o[f][0] * a_low + acc[f][0];
        o[f][1] = o[f][1] * a_low + acc[f][1];
        o[f][2] = o[f][2] * a_high + acc[f][2];
        o[f][3] = o[f][3] * a_high + acc[f][3];
      }
    }
    __syncthreads();
  }
  for (int h = 0; h < 2; ++h) {
    const int g = row8 + 8 * h;
    if (g >= valid) continue;
    const int qi = group * ROWS + g;
    const size_t at = ((size_t)(kvh * G + qi / N) * N + qi % N) * NCH + c;
    for (int f = 0; f < FR; ++f) {
      const int d = (warp * FR + f) * 8 + pair * 2;
      PO[at * HD + d] = o[f][2 * h];
      PO[at * HD + d + 1] = o[f][2 * h + 1];
    }
    if (warp == 0 && pair == 0) {
      PM[at] = m[h];
      PL[at] = l[h];
    }
  }
}

// Thread (query head, row, d): the chunks' partials merged in chunk order, then bf16(o / l).
__global__ void dflash2_merge_kernel(const float* __restrict__ PO, const float* __restrict__ PM, const float* __restrict__ PL,
                             const int* __restrict__ POS, __nv_bfloat16* __restrict__ OUT, int N, int NH, int WINDOW,
                             int NCH) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)N * NH * HD) return;
  const int d = static_cast<int>(i % HD), row = static_cast<int>(i / HD % N), qh = static_cast<int>(i / HD / N);
  const long long s = POS[0];
  const int total = static_cast<int>(s + N - max(min(s - WINDOW, s), 0LL));
  const int chunks = min((total + CHUNK - 1) / CHUNK, NCH);
  float m = minus_inf(), l = 0.f, o = 0.f;
  for (int c = 0; c < chunks; ++c) {
    const size_t at = ((size_t)qh * N + row) * NCH + c;
    const float cm = PM[at], cl = PL[at], co = PO[at * HD + d];
    const bool active = cl > 0.f;
    const float next_m = active ? fmaxf(m, cm) : m;
    const float a = active ? (m == minus_inf() ? 0.f : expf(m - next_m)) : 1.f;
    const float b = active ? expf(cm - next_m) : 0.f;
    o = o * a + co * b;
    l = l * a + cl * b;
    m = next_m;
  }
  OUT[((size_t)row * NH + qh) * HD + d] = __float2bfloat16_rn(o / l);
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

// A context's keys or values: bf16 [KV, capacity, 128], rows that cp.async reads 16 bytes at a time.
bool context_of(const zs::View& c) {
  return c.defined() && c.dtype() == ZS_BF16 && c.dim() == 3 && c.is_contiguous() && c.size(0) >= 1 &&
         c.size(0) <= 4096 && c.size(1) >= 1 && c.size(1) <= INT32_MAX / HD && c.size(2) == HD &&
         reinterpret_cast<uintptr_t>(c.data_ptr<__nv_bfloat16>()) % 16 == 0;
}

}  // namespace

extern "C" {

// The values `parts` holds for H query heads, N rows and a window: the chunks' partials.
int64_t zs_dflash2_attend_parts(int64_t heads, int64_t n, int64_t window) {
  return heads * n * ((window + n + CHUNK - 1) / CHUNK) * (HD + 2);
}

// q bf16 [H, N, 128], the block's queries at positions pos .. pos + N - 1, over k and v bf16 [KV, capacity, 128]
// -> out bf16 [N, H * 128]. parts fp32, the caller's scratch.
int zs_dflash2_attend(const zs_view* q_, const zs_view* k_, const zs_view* v_, const zs_view* pos_,
                           const zs_view* out_, const zs_view* parts_, int64_t window, int64_t causal,
                           int64_t ring) {
  zs::begin();
  const zs::View q(q_), k(k_), v(v_), pos(pos_), out(out_), parts(parts_);
  ZS_REQUIRE(context_of(k) && context_of(v) && v.size(0) == k.size(0) && v.size(1) == k.size(1),
             "k, v: bf16 [KV, capacity, 128], 16-byte aligned");
  const int64_t kv = k.size(0), cap = k.size(1);
  ZS_REQUIRE(q.defined() && q.dtype() == ZS_BF16 && q.dim() == 3 && q.is_contiguous() && q.size(0) >= kv &&
             q.size(0) % kv == 0 && q.size(0) <= 4096 && q.size(1) >= 1 && q.size(1) <= 4096 && q.size(2) == HD &&
             reinterpret_cast<uintptr_t>(q.data_ptr<__nv_bfloat16>()) % 16 == 0,
             "q: bf16 [H, N, 128], KV dividing H, 16-byte aligned");
  const int64_t heads = q.size(0), n = q.size(1), g = heads / kv;
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= n * heads * HD,
             "out: bf16 [N, H * 128]");
  ZS_REQUIRE(window >= 0 && window <= (1 << 24), "window: positions back");
  ZS_REQUIRE(!ring || window + n <= cap, "a ring holds window + N rows");
  const int64_t nch = (window + n + CHUNK - 1) / CHUNK, groups = (g * n + ROWS - 1) / ROWS;
  ZS_REQUIRE(parts.defined() && parts.dtype() == ZS_F32 && parts.is_contiguous() &&
             parts.numel() >= zs_dflash2_attend_parts(heads, n, window) && kv * groups <= 65535 && nch <= 65535,
             "parts: fp32, zs_dflash2_attend_parts values");
  float* po = parts.data_ptr<float>();
  float* pm = po + heads * n * nch * HD;
  float* pl = pm + heads * n * nch;
  dflash2_chunk_kernel<<<dim3(static_cast<unsigned>(kv * groups), static_cast<unsigned>(nch)), THREADS, SMEM, zs::stream()>>>(
      q.data_ptr<__nv_bfloat16>(), k.data_ptr<__nv_bfloat16>(), v.data_ptr<__nv_bfloat16>(), pos.data_ptr<int>(), po,
      pm, pl, static_cast<float>(1.0 / sqrt(static_cast<double>(HD))), static_cast<int>(n), static_cast<int>(g),
      static_cast<int>(cap), static_cast<int>(window), causal ? 1 : 0, ring ? 1 : 0, static_cast<int>(nch),
      static_cast<int>(groups));
  ZS_CUDA_LAUNCH_CHECK();
  dflash2_merge_kernel<<<blocks(n * heads * HD), 256, 0, zs::stream()>>>(
      po, pm, pl, pos.data_ptr<int>(), out.data_ptr<__nv_bfloat16>(), static_cast<int>(n), static_cast<int>(heads),
      static_cast<int>(window), static_cast<int>(nch));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
