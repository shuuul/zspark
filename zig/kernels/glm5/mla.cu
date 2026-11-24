// GLM-5.3-Flash's DSA over the latent cache, dense rows: a token's key and value are one bf16 row of LW = 512
// values (kv_a's output, normed), and kv_b's blocks are taken into the query and out of the result, so nothing
// a head wide is stored. For a row and one of its heads:
//   qa = bf16(q W_k), W_k the head's key rows of kv_b [DK, LW];
//   ol = bf16(softmax((qa . latent) SCALE over the keys 0 .. pos) . latent);
//   out = bf16(W_v ol), W_v the head's value rows [DV, LW].
// A row's values come from its own query, its position and the cache rows before it: the chunks split its keys
// by position alone and are merged in chunk order, so neither the window, the cache's capacity nor a launch's
// context changes a bit.
// Absorb, expand and the cache writes are `mla_proj.cu`'s (spec 121 WS-06).
// Semantics from TensorFold 0.6.6 (cb2ebf0) families/glm5_next/cuda/latent.py (_tile, _dense_chunks,
// _sparse_chunks, _merge; Triton), Apache-2.0; rewritten for zspark in CUDA C++. A row past the dense limit
// attends the indexer's list (index.cu, spec 046) through the same chunk kernel, its keys gathered by the list.
// Where Triton leaves a reduction's order to its compiler it is fixed here and stated at the kernel; TensorFold's
// exported tensors have not been held against them (spec 044 WS-07).
// Hand-written because the attention is not a GEMM CUTLASS has (a row masks the keys by its own position and
// carries an online softmax from tile to tile; zig/kernels/common/attention.cu, whose steps these are, with the
// latent as key and value both).

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../common/copies.cuh"
#include "../common/qmm_frag.cuh"
#include "mla.cuh"
#include "../zs/host.h"

namespace {

using namespace common_copies;

using namespace glm5_mla;

constexpr int FR = LW / 32;    // the 8-value fragments of the latent a warp carries
constexpr int DC = LW / 8;     // 16-byte pieces of a latent
// a chunk block's shared memory: the heads' queries, a tile's latents, a tile's bf16(p), the warps' max and sum
constexpr int SMEM = HEADS * LW * 2 + TILE * LW * 2 + HEADS * TILE * 2 + 2 * 4 * HEADS * 4;
// The FP8 chunk block's shared memory, one block an SM either way (bf16's 52,224 B leaves room for no second):
//   queries [HEADS, LW] bf16             32,768  the heads' absorbed rows, read by every tile
//   kv tile [TILE, LW] bf16              32,768  the records widened (exactly), as bf16's tile: the mma's B
//   p [GROUPS, HEADS, TILE] bf16          8,192  bf16(p s_g) a group: P.V's A, the scale a key's and group's
//   max, sum [2, 4, HEADS] fp32           1,024  the warps' partials
//   stage [TILE, REC]                    16,896  the next tile's records by cp.async, while this one is widened
//   total                                91,648  could be less: the stage could be the kv tile's second half
//                                                were the tile widened in place (not tried; one block an SM anyway)
constexpr int SMEM_FP8 = HEADS * LW * 2 + TILE * LW * 2 + GROUPS * HEADS * TILE * 2 + 2 * 4 * HEADS * 4 + TILE * REC;
static_assert(REC % 16 == 0 && SMEM_FP8 <= 101376, "an FP8 record is whole 16-byte pieces; the block fits an SM");

// 16 e4m3 codes as 16 bf16, exactly (every e4m3 value is a bf16 one).
__device__ __forceinline__ void widen16(const uint4& codes, uint4& low, uint4& high) {
  const uint32_t w[4] = {codes.x, codes.y, codes.z, codes.w};
  uint32_t out[8];
  for (int i = 0; i < 4; ++i)
    for (int h = 0; h < 2; ++h) {
      const __half2_raw two = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w[i] >> (16 * h)), __NV_E4M3);
      const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&two));
      const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
      out[i * 2 + h] = *reinterpret_cast<const uint32_t*>(&b);
    }
  low = make_uint4(out[0], out[1], out[2], out[3]);
  high = make_uint4(out[4], out[5], out[6], out[7]);
}

// Block (row, 32 heads, chunk) of 8 warps: the row's keys [c CHUNK, (c + 1) CHUNK) of 0 .. POS + r against 32 of
// its heads, a TILE at a time, on the tensor cores. The warps are two groups of four, a group the 16 heads that
// are the rows of its mmas, zero queries past H; both read the one staged tile, so a key is loaded once for a
// rank's 32 heads (TensorFold's HB_WIDE, which it keeps for a prompt's chunks). Each tile, each group:
//   score = (qa . latent) * SCALE, one fp32 chain of m16n8k16 steps over the latent, 16 values at a step;
//   m' = max(m, tile max); p = exp(score - m'); l = l exp(m - m') + sum p;
//   o = o exp(m - m') + (bf16(p) . latent, one fp32 chain of steps over the tile's keys, 16 at a step).
// The softmax runs where the scores land: a thread holds two heads' scores of two keys (k, k + 1, k even), so
// sum p adds those two, then the four threads of a warp's 8 keys as a tree (pairs, then the pairs), then the
// group's four warps in order. Every thread of a head carries its m, l and exp(m - m'). A key past the row's
// position is zeros with p = 0. An mma's element is a function of its own row of A and column of B, so a head's
// values depend on neither the other heads nor the group or warp that holds them: a head's bits are those of
// the block of 16 heads and 4 warps this kernel was until spec 093 (`dsa_test`, unchanged). The chunk's o, m and
// l go to PO [R, NCH, H, LW], PM and PL [R, NCH, H]; a chunk past the row's keys writes nothing.
// A row whose COUNTS is positive attends its list instead (the indexer's, spec 046): its keys are the latents at
// TOKENS [R, WIDTH] in list order, COUNTS[r] of them, and everything after the address is the same steps, so a
// listed row's bits are its own query's, its list's and those latents'. A block is one row, so it is wholly
// one or the other.
// Shared rows are swizzled by 16-byte pieces so ldmatrix's eight rows fall in eight bank groups; a tile's latents
// come by cp.async, zeros past the row's keys, and stay for both products. A thread loads an eighth of one
// key's latent (its row found once, by the list and the table) and asks for the same eighth of the key a tile
// on (`ask`, two lines), which no arithmetic sees.
// Measured alone (spec 093 WS-04, `ZSPARK_GLM5_PROFILE=attend`, a list of 2,051 keys from twelve caches, the
// kernel's own us in a trace): 89, 92, 94, 100, 105 us at 1, 2, 3, 5, 6 rows and 179, 184 at 7, 8 as blocks of 16
// heads; 65, 68, 69, 72, 75, 80, 86 as this one, and a 512-row window 7.0 ms to 3.5. Of that, the 32 heads a
// block alone: 93 to 107 at 1 to 6 rows, 111 at 7; the stores of two values, the one lookup a thread and the
// asking took the rest (from L2 69 to 60, the asking the gap to L2: 25 to 40 us down to 5 to 18). ncu at 2
// rows from L2: 18 cycles a warp between instructions, 6.5 of them on a shared load; the tensor pipe at 17%
// of its peak while active. A block is its warps' own chains, not the SM's products: 16 heads a block of four
// warps took the same 2.8 us a tile as 32 of eight, and the score's 32 steps as four sums in turn changed
// nothing. Chunks of 256 keys (9 partials a row) took 55, 59, 64, 70, 75, 81 us at 1 to 6 rows and of 128 keys
// 36, 45, 55, 78, 85, 95, but their partials cost a 512-row window 1.2 ms a launch (4.5 + 1.5 for 3.9 + 0.8)
// and the merge 2 to 9 us: not taken. Not tried: 16 warps a block, the score's K by halves.
// Shared memory, 68,608 B a block, one block an SM (spec 093 WS-04):
//   q     32,768  the 32 heads' queries, read by ldmatrix every step of every tile's score: 128 registers a
//                 lane if kept there instead; half of it is idle when H - 32 hb is 16 or less;
//   kv    32,768  a tile's 32 latents, the bytes in flight; one tile, since a second (a tile asked for under
//                 the products of the one before) does not fit an SM beside q;
//   p      2,048  bf16(p) [32, 32], the second product's A, which every warp of a group reads;
//   part   1,024  a warp's largest score and its sum of p, [4, 32] each, where the group's warps meet.
// Before: 16 heads a block, 50,688 B, which is one block an SM too (two need 50,176 or less), so 7 rows' 70
// blocks were two waves, and both of a row's blocks loaded each key.
// FP8 (an FP8 cache's records, `REC`): the tile's records come by cp.async into `stage` (16,896 B) and are
// widened there to the bf16 tile, the codes exactly, so the products are the bf16 path's mmas on codes. The
// score keeps a chain a group (8 steps each, the latent's quarters in order) and is
//   fma(c3, s3, fma(c2, s2, fma(c1, s1, c0 s0))) * SCALE, s_g the key's group scales;
// p is the same, and the second product's A is bf16(p s_g) for the group a warp's quarter of the latent is (a
// warp's 16 fragments are one group), 2,048 B a group. 91,648 B, one block an SM as the bf16 kernel. Since spec 114
// WS-05 an FP8 cache is attended by `mla_q8.cu` (vLLM's FP8 queries and weights) unless `zs_glm5_mla_q8(0)`.
// WHOLE (spec 116 WS-01, from `zs_glm5_mla_window`'s rows, 128 by default): a block (row, 32 heads) runs the
// row's chunks one after the other, each from nothing as a chunk block does (the same tiles, sums and order),
// and at a chunk's end folds it into the row's output by `mla.cuh` `fold_of` and `folded`, `mla_merge_kernel`'s
// step in chunk order from m = -inf, l = o = 0; the last fold writes bf16(o / l). So a window's row has the bits
// of the chunk blocks and the merge, which steps and drafts keep. The running m and l are a thread's registers
// (every thread of a head holds them); the running o, as many floats as the chunk's own, is the row's of po
// taken as fp32 [R, H, LW] (po need hold R rows of one chunk, not of every chunk), read and written only by the
// thread that owns those values, in L2 while the row runs, its lines dropped from L2 unwritten after the last
// read (`discard.global.L2`). No shared memory is added: the table above
// stands. What it saves at a 16k prompt's 4,096-row window (`profile_test.zig` `attend:4096`, a list of 2,051
// keys a row, machine alone): the five chunk blocks' 1.34 GB of fp32 partials written and read back by the
// merge, and four of a row's five loads of its queries; the fifth chunk's 0 to 3 keys are one tile of the row's
// block, not a block of their own, and its first keys are asked for under the fourth chunk's last tile. bf16
// cache 34.0 to 30.5 ms, FP8 records under bf16 queries 38.0 to 26.9; `q8_chunk_kernel` below. Under 128 rows
// a block a chunk wins (64 rows 0.54 against 0.64 ms, 128 rows 1.07 against 1.02), and steps and drafts keep it.
// Registers: 254, no spill, either form (one block an SM by its shared memory anyway).
template <bool FP8, bool WHOLE>
__global__ void __launch_bounds__(THREADS)
    mla_chunk_kernel(const __nv_bfloat16* __restrict__ QA, const void* __restrict__ LC_,
                 const int* __restrict__ POS, const int* __restrict__ TABLE, float* __restrict__ PO, float* __restrict__ PM,
                 float* __restrict__ PL, const int* __restrict__ TOKENS, const int* __restrict__ COUNTS, int WIDTH,
                 float SCALE, int H, int NCH, __nv_bfloat16* __restrict__ OUT) {
  extern __shared__ __align__(16) unsigned char shared[];
  __nv_bfloat16* const q = reinterpret_cast<__nv_bfloat16*>(shared);   // [HEADS, LW]
  __nv_bfloat16* const kv = q + HEADS * LW;                            // [TILE, LW]
  __nv_bfloat16* const p = kv + TILE * LW;                             // [HEADS, TILE], FP8 [GROUPS, HEADS, TILE]
  constexpr int PS = FP8 ? GROUPS : 1;
  float* const part_m = reinterpret_cast<float*>(p + PS * HEADS * TILE);  // [4, HEADS]: a warp's largest score
  float* const part_l = part_m + 4 * HEADS;                            // [4, HEADS]: a warp's sum of p
  unsigned char* const stage = reinterpret_cast<unsigned char*>(part_l + 4 * HEADS);  // FP8: [TILE, REC]
  const __nv_bfloat16* const LC = static_cast<const __nv_bfloat16*>(LC_);
  const unsigned char* const LR = static_cast<const unsigned char*>(LC_);
  const int r = blockIdx.x, hb = blockIdx.y, tid = threadIdx.x;
  const int n = keys_of(POS, COUNTS, r);
  const int* const list = COUNTS && COUNTS[r] > 0 ? TOKENS + (size_t)r * WIDTH : nullptr;
  if (!WHOLE && (int)blockIdx.z * CHUNK >= n) return;
  const int G = min(HEADS, H - hb * HEADS);
  // the group's 16 heads start at h0; warp is this warp's place among the group's four
  const int h0 = (tid / 128) * 16, warp = tid / 32 % 4, lane = tid % 32, row8 = lane >> 2, pair = lane & 3;
  const __nv_bfloat16* q_in = QA + ((size_t)r * H + hb * HEADS) * LW;
  for (int i = tid; i < HEADS * DC; i += THREADS) {
    const int g = i / DC, ch = i % DC;
    uint4 x = make_uint4(0, 0, 0, 0);
    if (g < G) x = *reinterpret_cast<const uint4*>(q_in + (size_t)g * LW + ch * 8);
    *reinterpret_cast<uint4*>(q + g * LW + (ch ^ (g % 8)) * 8) = x;
  }
  for (int i = tid; i < PS * HEADS * TILE / 8; i += THREADS) *reinterpret_cast<uint4*>(p + i * 8) = make_uint4(0, 0, 0, 0);
  // WHOLE: the row's running m and l of heads h0 + row8 and h0 + row8 + 8, as the merge's from nothing
  float run_m[2] = {minus_inf(), minus_inf()}, run_l[2] = {0.f, 0.f};
  const int chunks = WHOLE ? min((n + CHUNK - 1) / CHUNK, NCH) : 1;
  for (int ci = 0; ci < chunks; ++ci) {
    const int c = WHOLE ? ci : blockIdx.z, start = c * CHUNK;
    float o[FR][4];
    for (int f = 0; f < FR; ++f)
      for (int e = 0; e < 4; ++e) o[f][e] = 0.f;
    // of heads h0 + row8 and h0 + row8 + 8
    float m[2] = {minus_inf(), minus_inf()}, l[2] = {0.f, 0.f}, alpha[2] = {1.f, 1.f};
    const int keys = min(n - start, CHUNK);
    for (int t0 = 0; t0 < keys; t0 += TILE) {
      {
        // a thread a key's eighth: the row found once, then 128 bytes of it in 16-byte pieces
        const int key = tid >> 3, ki = start + t0 + key;
        const bool here = ki < n;
        const int row = here ? placed(TABLE, list ? list[ki] : ki) : 0;
        const int kn = ki + TILE;
        if constexpr (FP8) {
          const unsigned char* from = LR + (size_t)row * REC;
          for (int ch = tid & 7; ch < REC / 16; ch += 8) qmm_frag::cp16z(stage + key * REC + ch * 16, from + ch * 16, here);
          if (kn < n && (WHOLE || t0 + TILE < keys)) {
            const unsigned char* next = LR + (size_t)placed(TABLE, list ? list[kn] : kn) * REC + (tid & 7) * 64;
            ask(reinterpret_cast<const __nv_bfloat16*>(next));
            if ((tid & 7) == 7) ask(reinterpret_cast<const __nv_bfloat16*>(next + 64));
          }
        } else {
          const __nv_bfloat16* from = LC + (size_t)row * LW;
          for (int ch = (tid & 7) * 8; ch < (tid & 7) * 8 + 8; ++ch)
            qmm_frag::cp16z(kv + key * LW + (ch ^ (key % 8)) * 8, from + ch * 8, here);
          // the next tile's same bytes asked for now and not waited for: they come under this tile's products
          if (kn < n && (WHOLE || t0 + TILE < keys)) {
            const __nv_bfloat16* next = LC + (size_t)placed(TABLE, list ? list[kn] : kn) * LW + (tid & 7) * 64;
            ask(next);
            ask(next + 32);
          }
        }
      }
      qmm_frag::commit();
      qmm_frag::wait<0>();
      __syncthreads();
      if constexpr (FP8) {
        // a thread a key's eighth: 64 codes, four pieces of 16, widened into the swizzled bf16 tile
        const int key = tid >> 3;
        for (int i = 0; i < 4; ++i) {
          const int cp = (tid & 7) * 4 + i;
          uint4 low, high;
          widen16(*reinterpret_cast<const uint4*>(stage + key * REC + cp * 16), low, high);
          *reinterpret_cast<uint4*>(kv + key * LW + ((2 * cp) ^ (key % 8)) * 8) = low;
          *reinterpret_cast<uint4*>(kv + key * LW + ((2 * cp + 1) ^ (key % 8)) * 8) = high;
        }
        __syncthreads();
      }
      {
        // this warp's 8 keys: lanes 0-7 address their low 8 values of a step, 8-15 the high 8
        float c0[4] = {0.f, 0.f, 0.f, 0.f};
        const int head = h0 + lane % 16, key = warp * 8 + (lane & 7);
        const int k0 = warp * 8 + pair * 2;
        const int mine[2] = {k0, k0 + 1};
        if constexpr (FP8) {
          float cg[GROUPS][4];
          for (int g = 0; g < GROUPS; ++g)
            for (int e = 0; e < 4; ++e) cg[g][e] = 0.f;
          for (int step = 0; step < DC / 2; ++step) {
            uint32_t a[4], b[2];
            qmm_frag::ldmatrix4(a, q + head * LW + ((step * 2 + (lane >> 4)) ^ (head % 8)) * 8);
            qmm_frag::ldmatrix2(b, kv + key * LW + ((step * 2 + ((lane >> 3) & 1)) ^ (key % 8)) * 8);
            qmm_frag::mma(cg[step / (GROUP / 16)], a, b[0], b[1]);
          }
          for (int e = 0; e < 4; ++e) {
            const float* sk = reinterpret_cast<const float*>(stage + mine[e & 1] * REC + LW);
            c0[e] = fmaf(cg[3][e], sk[3], fmaf(cg[2][e], sk[2], fmaf(cg[1][e], sk[1], cg[0][e] * sk[0])));
          }
        } else {
          for (int step = 0; step < DC / 2; ++step) {
            uint32_t a[4], b[2];
            qmm_frag::ldmatrix4(a, q + head * LW + ((step * 2 + (lane >> 4)) ^ (head % 8)) * 8);
            qmm_frag::ldmatrix2(b, kv + key * LW + ((step * 2 + ((lane >> 3) & 1)) ^ (key % 8)) * 8);
            qmm_frag::mma(c0, a, b[0], b[1]);
          }
        }
        float sc[2][2] = {{c0[0], c0[1]}, {c0[2], c0[3]}};
        bool live[2];
        for (int j = 0; j < 2; ++j) {
          live[j] = start + t0 + mine[j] < n;
          for (int h = 0; h < 2; ++h) sc[h][j] = live[j] ? sc[h][j] * SCALE : minus_inf();
        }
        for (int h = 0; h < 2; ++h) {
          float top = fmaxf(sc[h][0], sc[h][1]);
          top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
          top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
          if (pair == 0) part_m[warp * HEADS + h0 + row8 + 8 * h] = top;
        }
        __syncthreads();
        for (int h = 0; h < 2; ++h) {
          const int at = h0 + row8 + 8 * h;
          const float tile_m = fmaxf(fmaxf(part_m[at], part_m[HEADS + at]), fmaxf(part_m[2 * HEADS + at], part_m[3 * HEADS + at]));
          const bool active = tile_m != minus_inf();
          const float next_m = active ? fmaxf(m[h], tile_m) : m[h];
          alpha[h] = active ? (m[h] == minus_inf() ? 0.f : expf(m[h] - next_m)) : 1.f;
          m[h] = next_m;
          float pk[2];
          for (int j = 0; j < 2; ++j) pk[j] = active && live[j] ? expf(sc[h][j] - next_m) : 0.f;
          float sum = pk[0] + pk[1];
          sum += __shfl_xor_sync(0xffffffffu, sum, 1);
          sum += __shfl_xor_sync(0xffffffffu, sum, 2);
          if (pair == 0) part_l[warp * HEADS + at] = sum;
          // a head past H keeps p = 0
          if (at < G)
            for (int j = 0; j < 2; ++j) {
              const int place = at * TILE + ((mine[j] / 8) ^ (at % 4)) * 8 + mine[j] % 8;
              if constexpr (FP8) {
                const float* sk = reinterpret_cast<const float*>(stage + mine[j] * REC + LW);
                for (int g = 0; g < GROUPS; ++g) p[g * HEADS * TILE + place] = __float2bfloat16_rn(pk[j] * sk[g]);
              } else {
                p[place] = __float2bfloat16_rn(pk[j]);
              }
            }
        }
      }
      __syncthreads();
      {
        float acc[FR][4];
        for (int f = 0; f < FR; ++f)
          for (int e = 0; e < 4; ++e) acc[f][e] = 0.f;
        const int head = h0 + lane % 16;
        // FP8: the group of this warp's quarter of the latent
        const __nv_bfloat16* const pw = FP8 ? p + warp * HEADS * TILE : p;
        for (int step = 0; step < TILE / 16; ++step) {
          uint32_t a[4];
          qmm_frag::ldmatrix4(a, pw + head * TILE + ((step * 2 + (lane >> 4)) ^ (head % 4)) * 8);
          // lanes 0-7 address this step's first eight keys, 8-15 its next eight
          const int key = step * 16 + (lane & 15);
          for (int f = 0; f < FR; ++f) {
            uint32_t b[2];
            ldmatrix2t(b, kv + key * LW + ((warp * FR + f) ^ (key % 8)) * 8);
            qmm_frag::mma(acc[f], a, b[0], b[1]);
          }
        }
        for (int h = 0; h < 2; ++h) {
          const int at = h0 + row8 + 8 * h;
          l[h] = l[h] * alpha[h] + (((part_l[at] + part_l[HEADS + at]) + part_l[2 * HEADS + at]) + part_l[3 * HEADS + at]);
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
    const size_t base = ((size_t)r * NCH + c) * H + hb * HEADS + h0;
    const int left = G - h0;
    if constexpr (WHOLE) {
      // the merge's step for this chunk, as `q8_chunk_kernel<true>`'s: po as fp32 [R, H, LW] the running output, a
      // thread's own values, the last fold bf16(o / l) and the lines dropped
      const size_t slot = (size_t)r * H + hb * HEADS + h0;
      const bool last = ci + 1 == chunks;
  #pragma unroll
      for (int h = 0; h < 2; ++h) {
        if (row8 + 8 * h >= left) continue;
        const Fold w = fold_of(run_m[h], m[h], l[h]);
        run_l[h] = folded(run_l[h], l[h], w);
        run_m[h] = w.m;
        float* const at = PO + (slot + row8 + 8 * h) * LW + warp * FR * 8 + pair * 2;
        __nv_bfloat16* const to = OUT + ((size_t)r * H + hb * HEADS + h0 + row8 + 8 * h) * LW + warp * FR * 8 + pair * 2;
        for (int f = 0; f < FR; ++f) {
          float2 x = ci == 0 ? make_float2(0.f, 0.f) : *reinterpret_cast<const float2*>(at + f * 8);
          x.x = folded(x.x, o[f][2 * h], w);
          x.y = folded(x.y, o[f][2 * h + 1], w);
          if (!last) *reinterpret_cast<float2*>(at + f * 8) = x;
          else *reinterpret_cast<__nv_bfloat162*>(to + f * 8) = __floats2bfloat162_rn(x.x / run_l[h], x.y / run_l[h]);
        }
        // a line is 32 values: four fragments of 8 of the four lanes of this head
        if (last && ci > 0) {
          __syncwarp(0xfu << (lane & ~3));
          if (pair == 0)
            for (int f = 0; f < FR; f += 4) drop_line(at + f * 8);
        }
      }
    } else {
      for (int f = 0; f < FR; ++f) {
        const int d = (warp * FR + f) * 8 + pair * 2;
        if (row8 < left) *reinterpret_cast<float2*>(PO + (base + row8) * LW + d) = make_float2(o[f][0], o[f][1]);
        if (row8 + 8 < left) *reinterpret_cast<float2*>(PO + (base + row8 + 8) * LW + d) = make_float2(o[f][2], o[f][3]);
      }
      if (warp == 0 && pair == 0) {
        if (row8 < left) {
          PM[base + row8] = m[0];
          PL[base + row8] = l[0];
        }
        if (row8 + 8 < left) {
          PM[base + row8 + 8] = m[1];
          PL[base + row8 + 8] = l[1];
        }
      }
    }
  }
}

// Thread (row, head, d): the row's chunk partials merged in chunk order (`mla.cuh` `fold_of`), then bf16(o / l).
__global__ void mla_merge_kernel(const float* __restrict__ PO, const float* __restrict__ PM, const float* __restrict__ PL,
                             const int* __restrict__ POS, const int* __restrict__ COUNTS,
                             __nv_bfloat16* __restrict__ OUT, int R, int H, int NCH) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * H * LW) return;
  const int d = static_cast<int>(i % LW), h = static_cast<int>(i / LW % H), r = static_cast<int>(i / LW / H);
  const int n = keys_of(POS, COUNTS, r);
  const int chunks = min((n + CHUNK - 1) / CHUNK, NCH);
  float m = minus_inf(), l = 0.f, o = 0.f;
  for (int c = 0; c < chunks; ++c) {
    const size_t base = ((size_t)r * NCH + c) * H + h;
    const float cm = PM[base], cl = PL[base], co = PO[base * LW + d];
    const Fold w = fold_of(m, cm, cl);
    o = folded(o, co, w);
    l = folded(l, cl, w);
    m = w.m;
  }
  OUT[i] = __float2bfloat16_rn(o / l);
}

// `zs_glm5_mla_q8`: an FP8 cache's chunks with FP8 queries (`mla_q8.cu`), or 0 bf16 queries over the codes.
int q8 = 1;
// `zs_glm5_mla_window`: from this many rows a block a row's chunks in order, merged as it goes; 0 never. 128 but
// in tests: `dsa.zig` `Buffers` sizes po for it.
int window_rows = 128;

}  // namespace

extern "C" {

// An FP8 cache's chunks with the queries and P in FP8 as vLLM's (`mla_q8.cu`, the default), or with 0 bf16
// queries over the codes (`mla_chunk_kernel<true>`): not the same bits.
void zs_glm5_mla_q8(int on) { q8 = on; }

// From `rows` rows (0 never; 128, fixed but for tests) `zs_glm5_mla_attend` runs a block a row's 32 heads over its
// chunks in order, each folded into the row's output as `mla_merge_kernel` folds it, with no partials but the
// running output (spec 116 WS-01); fewer rows keep a block a chunk and the merge. The same bits.
void zs_glm5_mla_window(int rows) { window_rows = rows; }

// qa bf16 [R, H, 512], row r at position pos + r, over cache bf16 [capacity, 512] (or an FP8 cache's records, u8
// [capacity, 528]) rows 0 .. pos + r -> out bf16 [R, H, 512]. po fp32 [P, NCH, H, 512], pm and pl fp32 [P, NCH,
// H]: the chunks' partials, P >= R, NCH * 512 >= context >= pos + R; from the window form's rows po is the running
// output, [R, H, 512], so P NCH >= R. With tokens int32 [R, width] and counts int32 [R] (both or neither), a row
// whose count is positive attends its count's first tokens, positions through the table, and no other key; the
// others are dense rows. The indexer gives a count of 0 only to a row of at most width keys (the dense limit is
// the list's width), so with lists no row has more keys than width, and po need cover no more: NCH * 512 >=
// min(context, width).
int zs_glm5_mla_attend(const zs_view* qa_, const zs_view* cache_, const zs_view* pos_, const zs_view* table_,
                       const zs_view* po_, const zs_view* pm_, const zs_view* pl_, const zs_view* out_,
                       const zs_view* tokens_, const zs_view* counts_, double scale, int64_t context) {
  zs::begin();
  const zs::View qa(qa_), cache(cache_), pos(pos_), po(po_), pm(pm_), pl(pl_), out(out_);
  ZS_REQUIRE(qa.defined() && qa.dtype() == ZS_BF16 && qa.dim() == 3 && qa.is_contiguous() && qa.size(2) == LW &&
             qa.size(0) >= 1 && qa.size(1) >= 1 && qa.size(0) <= INT32_MAX && qa.size(1) <= INT32_MAX,
             "qa: bf16 [R, H, 512]");
  const int64_t rows = qa.size(0), h = qa.size(1);
  const bool fp8 = cache.defined() && cache.dtype() == ZS_U8;
  ZS_REQUIRE(cache.defined() && cache.dim() == 2 && cache.is_contiguous() &&
             ((cache.dtype() == ZS_BF16 && cache.size(1) == LW) || (fp8 && cache.size(1) == REC)),
             "cache: bf16 [capacity, 512] or u8 [capacity, 528]");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(context >= rows && context <= cache.size(0), "the cache holds pos + R rows");
  ZS_REQUIRE(po.defined() && po.dtype() == ZS_F32 && po.dim() == 4 && po.is_contiguous() && po.size(0) >= 1 &&
             po.size(1) >= 1 && po.size(2) == h && po.size(3) == LW, "po: fp32 [P, NCH, H, 512]");
  const int64_t nch = po.size(1);
  ZS_REQUIRE(pm.defined() && pm.dtype() == ZS_F32 && pm.numel() >= po.size(0) * nch * h && pl.defined() &&
             pl.dtype() == ZS_F32 && pl.numel() >= po.size(0) * nch * h, "pm, pl: fp32 [R, NCH, H]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * h * LW,
             "out: bf16 [R, H, 512]");
  const int* table = nullptr;
  if (table_) {
    const zs::View view(table_);
    if (view.defined()) {
      ZS_REQUIRE(view.dtype() == ZS_I32 && view.is_contiguous(), "table: int32 [extents]");
      table = view.data_ptr<int>();
    }
  }
  const int* tokens = nullptr;
  const int* counts = nullptr;
  int width = 0;
  if (tokens_ || counts_) {
    ZS_REQUIRE(tokens_ && counts_, "tokens and counts: both or neither");
    const zs::View list(tokens_), count(counts_);
    ZS_REQUIRE(list.defined() && list.dtype() == ZS_I32 && list.dim() == 2 && list.is_contiguous() &&
               list.size(0) >= rows && list.size(1) >= 1, "tokens: int32 [R, width]");
    ZS_REQUIRE(count.defined() && count.dtype() == ZS_I32 && count.is_contiguous() && count.numel() >= rows,
               "counts: int32 [R]");
    tokens = list.data_ptr<int>();
    counts = count.data_ptr<int>();
    width = static_cast<int>(list.size(1));
  }
  const int64_t span = tokens ? std::min<int64_t>(context, width) : context;
  ZS_REQUIRE(span <= nch * CHUNK, "po's chunks cover the keys a row attends");
  static bool configured[2] = {false, false};
  if (!configured[fp8]) {
    if (fp8) {
      ZS_CUDA_CHECK(cudaFuncSetAttribute(mla_chunk_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_FP8));
      ZS_CUDA_CHECK(cudaFuncSetAttribute(mla_chunk_kernel<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_FP8));
    } else {
      ZS_CUDA_CHECK(cudaFuncSetAttribute(mla_chunk_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
      ZS_CUDA_CHECK(cudaFuncSetAttribute(mla_chunk_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    }
    configured[fp8] = true;
  }
  const bool whole = window_rows > 0 && rows >= window_rows;
  // the chunk form's partials are po's [R, NCH, H, 512]; the window form's running output po's [R, H, 512]
  ZS_REQUIRE(whole ? po.size(0) * nch >= rows : po.size(0) >= rows, "po: the rows' partials, or their running output");
  // the window form drops po's lines from L2 by 128 bytes: they must be po's alone (a Buffer's are 256-aligned)
  ZS_REQUIRE(!whole || reinterpret_cast<uintptr_t>(po.data_ptr<float>()) % 128 == 0, "po: 128-byte aligned");
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>((h + HEADS - 1) / HEADS),
                  whole ? 1u : static_cast<unsigned>((span + CHUNK - 1) / CHUNK));
  const auto stream = zs::stream();
  __nv_bfloat16* const to = whole ? out.data_ptr<__nv_bfloat16>() : nullptr;
  if (fp8 && q8 != 0)
    glm5_mla::q8_chunks(grid, stream, qa.data_ptr<__nv_bfloat16>(), cache.data_ptr<unsigned char>(), pos.data_ptr<int>(),
                        table, po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), tokens, counts, width,
                        static_cast<float>(scale), static_cast<int>(h), static_cast<int>(nch), to);
  else
    (fp8 ? (whole ? mla_chunk_kernel<true, true> : mla_chunk_kernel<true, false>)
         : (whole ? mla_chunk_kernel<false, true> : mla_chunk_kernel<false, false>))<<<grid, THREADS, fp8 ? SMEM_FP8 : SMEM, stream>>>(
        qa.data_ptr<__nv_bfloat16>(), cache.data_ptr(), pos.data_ptr<int>(), table, po.data_ptr<float>(),
        pm.data_ptr<float>(), pl.data_ptr<float>(), tokens, counts, width, static_cast<float>(scale),
        static_cast<int>(h), static_cast<int>(nch), to);
  ZS_CUDA_LAUNCH_CHECK();
  if (whole) return zs::status();
  mla_merge_kernel<<<blocks(rows * h * LW), 256, 0, stream>>>(
      po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), pos.data_ptr<int>(), counts,
      out.data_ptr<__nv_bfloat16>(), static_cast<int>(rows), static_cast<int>(h), static_cast<int>(nch));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
