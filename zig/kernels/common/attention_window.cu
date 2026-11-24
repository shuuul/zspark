// The text attention over a prompt window's rows (spec 105 WS-05; moved from zig/kernels/qwen4/ with attention.cu
// by spec 138 WS-04, its text unchanged): attention.cu's attn_chunk_kernel and
// attn_merge_kernel as one launch, a block a (row, KV head) that attends the row's chunks one after the other and
// mixes each chunk's partial into the row's in registers. The bits are attention.cu's.
// Why: at a window of 2,048 rows over 2,052 keys a row, attn_chunk_kernel writes 252 MB of fp32 partials a
// launch and attn_merge_kernel reads them back (6.4 to 7.0 ms and 1.3 ms, a 16k prompt's serve trace), and each
// of a row's five chunk blocks loads the row's queries again. A window has blocks enough without the chunks: 2,048
// rows of 2 KV heads are 4,096 blocks, 28 waves of 144. A reply's 1 to 16 rows stay on attention.cu, whose
// chunks put a row's keys on five blocks.
// The arithmetic, attention.cu's to the bit:
//   a chunk: attn_chunk_kernel's tile loop as it is (the same TILE, the same order of every sum, the same FP8 and
//     NVFP4 paths), m, l, o and u from nothing at the chunk's first key;
//   the row: the chunks mixed in chunk order from nothing, as attn_merge_kernel does: a chunk of l > 0 is mixed,
//     m' = max(m, m_c), a = exp(m - m') (0 where m is -inf), b = exp(m_c - m'), o = fma(o, a, o_c b),
//     l = fma(l, a, l_c b), then bf16(o / l). Written with __fmaf_rn and __fmul_rn as attn_merge_kernel's
//     object has them (cuobjdump -sass, CUDA 13.0): another object is not bound to contract it the same way.
// A thread holds the merge's state of the values it carries: heads row8 and row8 + 8, its warp's fragments.
// What moves the data, not the arithmetic, departs from attention.cu (alone, `attend:2048:16384:fp8`, 3 blocks an
// SM; ncu, 2026-10-18): the chunks and merge as one launch 7.47 to 5.92 ms; the next tile's rows and scales asked
// a tile ahead into registers, 5.75; FP8 values asked before the scores, 5.62; and the transposed values' stores
// without bank conflicts, 3.97. attention.cu's FP8 block has lane l of a warp store chunk l % 16 of a four of
// keys: rows 16 x 80 bytes apart, one bank, 16 ways (260 M of the shared stores' 382 M wavefronts conflicted,
// the top stall MIO throttle). Here lanes 0-15 take sixteen fours of one chunk, consecutive words, and lanes
// 16-31 the next chunk with its words in the other order, 16 banks over.
// From attention.cu (TensorFold 0.6.5 609ca41 attention.py's _tile, _chunk and _merge behind it, Apache-2.0).
// The contiguous caches, a lone stream's (attention.zig's `Cache.table` null); with ZS_ROWS (spec 149 WS-02,
// compiled only from qwen4/attention_window_rows.cu) the pool's round as attention_rows.cu's attend reads it: row r
// of stream SID[r] at POS0[r], its caches at BASES[SID[r]], its dense keys at the pool's rows through TABLES[SID[r]]
// (512-position extents, common/attention.cuh's stored_at), its listed ids the pool's rows already. Only where a
// key lies changes, never a sum, so a row's bits are attend_rows' as they are attend's here.
//
// A block's shared memory (D = 256, the model's; 128 threads): attention.cu's chunk block's, nothing added.
//   | region | bf16 | FP8 | holds | less? |
//   | q | 8,192 | 4,096 | the 16 mma rows of queries, 12 in use, loaded once a row now | 6,144 with G rows |
//   | kv | 32,768 | 20,480 | a tile's keys, then its values in their place | the tile itself |
//   | p | 2,048 | 2,048 (1,280 used) | a tile's weights as the value product's A | no |
//   | part_m, part_l, scales, key_at | 768 | 1,352 | the warps' meeting, a tile's scales and rows | no |
//   | total | 43,776 | 27,976 | | 2 blocks an SM bf16, 3 FP8 |
// The partials attention.cu writes are in registers: 2 x FR x 4 floats and 6 more a thread.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../common/qmm_frag.cuh"
#include "attention_mma.cuh"
#include "../zs/host.h"

namespace {

#ifdef ZS_ROWS
#define ZS_WINDOW_KERNEL attn_window_rows_kernel
static_assert(CHUNK == 512, "an extent is a chunk");
#else
#define ZS_WINDOW_KERNEL attn_window_kernel
#endif

// Block (row, kv head) of 4 warps: attention.cu's attn_chunk_kernel over each chunk c of the row's list while
// c CHUNK < n, then its attn_merge_kernel over those chunks, the result to OUT [R, H, D] bf16.
template <int FR, bool F8, bool V4>
__global__ void __launch_bounds__(THREADS)
    ZS_WINDOW_KERNEL(
#ifdef ZS_ROWS
                       const int* __restrict__ SID, const int* const* __restrict__ TABLES,
                       const int64_t* __restrict__ BASES, const void* __restrict__ Q,
                       const void* __restrict__ KC_ARG, const void* __restrict__ VC_ARG,
#else
                       const void* __restrict__ Q, const void* __restrict__ KC, const void* __restrict__ VC,
#endif
                       const float* __restrict__ QS, const float* __restrict__ KS, const float* __restrict__ VS,
                       const int* __restrict__ POS0, __nv_bfloat16* __restrict__ OUT, const int* __restrict__ IDS,
                       const int* __restrict__ NK, const int* __restrict__ SPARSE, float SCALE, int H, int HK,
                       int G, int IDW) {
  constexpr int D = 32 * FR;
  constexpr int DC = D / 8;                    // 16-byte chunks of a head in bf16
  constexpr int DS = DC < 8 ? DC : 8;          // rows a swizzle cycles over
  constexpr int QC = F8 ? D / 16 : DC;         // 16-byte chunks of a head as the scores read it
  constexpr int QS8 = QC < 8 ? QC : 8;
  constexpr int TC = TILE / 8;
  __shared__ __align__(16) __nv_bfloat16 q[F8 ? HEADS * D / 2 : HEADS * D];   // F8: codes
  // a tile's keys, then its values; F8: the keys' codes, then the values' transposed, D rows 80 bytes apart
  __shared__ __align__(16) __nv_bfloat16 kv[F8 ? D * 40 : TILE * D];
  __shared__ __align__(16) __nv_bfloat16 p[HEADS * TILE];  // a tile's bf16(p), F8: the codes of p s_v / u
  __shared__ float part_m[4][HEADS];                        // each warp's largest score of a head in the tile
  __shared__ float part_l[4][HEADS];                        // each warp's sum of p
  __shared__ float q_scale[HEADS];                          // F8: s_q SCALE
  __shared__ float k_scale[TILE], v_scale[TILE];            // F8: a tile's s_k and s_v
  __shared__ float v_top[2];                                // F8: the largest s_v of the tile's halves
  __shared__ int key_at[TILE];                              // a tile's cache rows, -1 past the row's keys
  const int r = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
#ifdef ZS_ROWS
  const void* KC = reinterpret_cast<const void*>(BASES[(size_t)SID[r] * 3]);
  const void* VC = reinterpret_cast<const void*>(BASES[(size_t)SID[r] * 3 + 1]);
  const int* TABLE = TABLES[SID[r]];
  int n = POS0[r] + 1;
#else
  int n = POS0[0] + r + 1;
#endif
  bool sparse = false;
  if (NK != nullptr && SPARSE[r] != 0) {
    sparse = true;
    n = NK[r];
  }
  const int warp = tid / 32, lane = tid % 32, row8 = lane >> 2, pair = lane & 3;
  const char* q_in = static_cast<const char*>(Q) + ((size_t)r * H + hk * G) * (QC * 16);
  for (int i = tid; i < HEADS * QC; i += THREADS) {
    const int g = i / QC, ch = i % QC;
    uint4 x = make_uint4(0, 0, 0, 0);
    if (g < G) x = *reinterpret_cast<const uint4*>(q_in + (g * QC + ch) * 16);
    *reinterpret_cast<uint4*>(reinterpret_cast<char*>(q) + (g * QC + (ch ^ (g % QS8))) * 16) = x;
  }
  for (int i = tid; i < HEADS * TC; i += THREADS) *reinterpret_cast<uint4*>(p + i * 8) = make_uint4(0, 0, 0, 0);
  if (F8 && tid < HEADS) q_scale[tid] = tid < G ? QS[(size_t)r * H + hk * G + tid] * SCALE : 0.f;
  constexpr int APART = 80;
  uint8_t* const p8 = reinterpret_cast<uint8_t*>(p);
  uint8_t* const vt = reinterpret_cast<uint8_t*>(kv);
  // the row's: attn_merge_kernel's m, l and o of the values this thread carries, heads row8 and row8 + 8
  float row_m[2] = {minus_inf(), minus_inf()}, row_l[2] = {0.f, 0.f};
  float row_o[FR][4];
  for (int f = 0; f < FR; ++f)
    for (int e = 0; e < 4; ++e) row_o[f][e] = 0.f;
  // a thread of the first two warps: the next tile's row of its key and that row's scales, asked a tile ahead
  int next_at = -1;
  float next_ks = 0.f, next_vs = 0.f;
  const auto ask = [&](int ki) {
    if (tid < TILE) {
#ifdef ZS_ROWS
      next_at = ki < n ? (sparse ? IDS[(size_t)r * IDW + ki] : (TABLE[ki >> 9] << 9) | (ki & (CHUNK - 1))) : -1;
#else
      next_at = ki < n ? (sparse ? IDS[(size_t)r * IDW + ki] : ki) : -1;
#endif
      if (F8) {
        next_ks = next_at >= 0 ? KS[(size_t)next_at * HK + hk] : 0.f;
        next_vs = next_at >= 0 ? VS[(size_t)next_at * HK + hk] : 0.f;
      }
    }
  };
  ask(tid);
  for (int start = 0; start < n; start += CHUNK) {
    float unit[2] = {0.f, 0.f};
    float o[FR][4];
    for (int f = 0; f < FR; ++f)
      for (int e = 0; e < 4; ++e) o[f][e] = 0.f;
    // of heads row8 and row8 + 8
    float m[2] = {minus_inf(), minus_inf()}, l[2] = {0.f, 0.f}, alpha[2] = {1.f, 1.f};
    const int keys = min(n - start, CHUNK);
    for (int t0 = 0; t0 < keys; t0 += TILE) {
      if (tid < TILE) {
        key_at[tid] = next_at;
        if (F8) {
          k_scale[tid] = next_ks;
          float top = next_vs;
          v_scale[tid] = top;
          // these are the first two warps, whole
          for (int other = 16; other > 0; other >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, other));
          if (lane == 0) v_top[warp] = top;
        }
      }
      __syncthreads();
      // the tile after this one: the chunk's next, or the next chunk's first
      ask((t0 + TILE < keys ? start + t0 + TILE : start + CHUNK) + tid);
      for (int i = tid; i < TILE * QC; i += THREADS) {
        const int key = i / QC, ch = i % QC, at = key_at[key];
        qmm_frag::cp16z(reinterpret_cast<char*>(kv) + (key * QC + (ch ^ (key % QS8))) * 16,
                        static_cast<const char*>(KC) + (((size_t)(at >= 0 ? at : 0) * HK + hk) * QC + ch) * 16,
                        at >= 0);
      }
      qmm_frag::commit();
      // FP8 values (not NVFP4) asked before the scores, into registers, so that their latency runs under them
      constexpr int VTURNS = (TILE / 4 * QC + THREADS - 1) / THREADS;
      uint4 vin[F8 && !V4 ? VTURNS : 1][4];
      if (F8 && !V4)
        for (int turn = 0; turn < VTURNS; ++turn) {
          const int i = tid + turn * THREADS, four = i % (TILE / 4), ch = i / (TILE / 4);
          for (int j = 0; j < 4; ++j) {
            const int at = i < TILE / 4 * QC ? key_at[four * 4 + j] : -1;
            vin[turn][j] = at >= 0 ? *reinterpret_cast<const uint4*>(static_cast<const char*>(VC) +
                                                                      (((size_t)at * HK + hk) * QC + ch) * 16)
                                   : make_uint4(0, 0, 0, 0);
          }
        }
      qmm_frag::wait<0>();
      __syncthreads();
      {
        // this warp's 16 keys: lanes 0-15 address the first eight, 16-31 the next; odd eights of lanes the high 8
        float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
        const int head = lane % 16, key = warp * 16 + (lane & 7) + (lane >> 4) * 8;
        const char* q_row = reinterpret_cast<const char*>(q) + head * QC * 16;
        const char* k_row = reinterpret_cast<const char*>(kv) + key * QC * 16;
        for (int step = 0; step < QC / 2; ++step) {
          uint32_t a[4], b[4];
          qmm_frag::ldmatrix4(a, q_row + ((step * 2 + (lane >> 4)) ^ (head % QS8)) * 16);
          qmm_frag::ldmatrix4(b, k_row + ((step * 2 + ((lane >> 3) & 1)) ^ (key % QS8)) * 16);
          if (F8) {
            mma8(c0, a, b[0], b[1]);
            mma8(c1, a, b[2], b[3]);
          } else {
            qmm_frag::mma(c0, a, b[0], b[1]);
            qmm_frag::mma(c1, a, b[2], b[3]);
          }
        }
        const int k0 = warp * 16 + pair * 2;
        const int mine[4] = {k0, k0 + 1, k0 + 8, k0 + 9};
        float sc[2][4] = {{c0[0], c0[1], c1[0], c1[1]}, {c0[2], c0[3], c1[2], c1[3]}};
        bool live[4];
        for (int j = 0; j < 4; ++j) {
          live[j] = key_at[mine[j]] >= 0;
          for (int h = 0; h < 2; ++h) {
            const float scaled = F8 ? sc[h][j] * k_scale[mine[j]] * q_scale[row8 + 8 * h] : sc[h][j] * SCALE;
            sc[h][j] = live[j] ? scaled : minus_inf();
          }
        }
        for (int h = 0; h < 2; ++h) {
          float top = fmaxf(fmaxf(sc[h][0], sc[h][1]), fmaxf(sc[h][2], sc[h][3]));
          top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
          top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
          if (pair == 0) part_m[warp][row8 + 8 * h] = top;
        }
        __syncthreads();
        for (int h = 0; h < 2; ++h) {
          const int head = row8 + 8 * h;
          const float tile_m =
              fmaxf(fmaxf(part_m[0][head], part_m[1][head]), fmaxf(part_m[2][head], part_m[3][head]));
          const bool active = tile_m != minus_inf();
          const float next_m = active ? fmaxf(m[h], tile_m) : m[h];
          alpha[h] = active ? (m[h] == minus_inf() ? 0.f : expf(m[h] - next_m)) : 1.f;
          m[h] = next_m;
          float pk[4];
          for (int j = 0; j < 4; ++j) pk[j] = active && live[j] ? expf(sc[h][j] - next_m) : 0.f;
          float sum = ((pk[0] + pk[1]) + pk[2]) + pk[3];
          sum += __shfl_xor_sync(0xffffffffu, sum, 1);
          sum += __shfl_xor_sync(0xffffffffu, sum, 2);
          if (pair == 0) part_l[warp][head] = sum;
          // a head past G keeps p = 0
          if (F8) {
            const float most = (active ? expf(tile_m - next_m) : 0.f) * fmaxf(v_top[0], v_top[1]);
            const float inv = most > 0.f ? 448.f / most : 0.f;
            unit[h] = most / 448.f;
            // attention.cu's comment: an infinite inv gives a weight of zero the code zero through fmaxf
            if (head < G)
              for (int j = 0; j < 4; ++j)
                p8[head * APART + mine[j]] =
                    __nv_cvt_float_to_fp8(fmaxf(pk[j] * v_scale[mine[j]] * inv, 0.f), __NV_SATFINITE, __NV_E4M3);
          } else if (head < G) {
            for (int j = 0; j < 4; ++j)
              p[head * TILE + ((mine[j] / 8) ^ (head % 8)) * 8 + mine[j] % 8] = __float2bfloat16_rn(pk[j]);
          }
        }
      }
      // the values take the keys' place: the scores above are the last thing to read them
      if (F8) {
        // four keys' codes of 16 values a turn, into those values' rows
        for (int i = tid; i < TILE / 4 * QC; i += THREADS) {
          const int four = i % (TILE / 4), ch = i / (TILE / 4);
          uint32_t in[4][4];
          for (int j = 0; j < 4; ++j) {
            const int at = key_at[four * 4 + j];
            uint4 x = make_uint4(0, 0, 0, 0);
            if (V4) {
              if (at >= 0) {
                const uint8_t* head_v = static_cast<const uint8_t*>(VC) + ((size_t)at * HK + hk) * nvfp4_width(D);
                const uint64_t fours = *reinterpret_cast<const uint64_t*>(head_v + ch * 8);
                const uint64_t by_code = BY_CODE.of[head_v[D / 2 + ch] & 127];
                // __byte_perm picks a byte of by_code by each code's low three bits; the fourth is the sign
                const uint32_t low = static_cast<uint32_t>(by_code), high = static_cast<uint32_t>(by_code >> 32);
                uint32_t words[4];
                for (int word = 0; word < 4; ++word) {
                  const uint32_t codes = static_cast<uint32_t>(fours >> (word * 16)) & 0xffff;
                  words[word] = __byte_perm(low, high, codes) | (codes & 0x8) << 4 | (codes & 0x80) << 8 |
                                (codes & 0x800) << 12 | (codes & 0x8000) << 16;
                }
                x = make_uint4(words[0], words[1], words[2], words[3]);
              }
            } else {
              x = vin[(i - tid) / THREADS][j];
            }
            in[j][0] = x.x;
            in[j][1] = x.y;
            in[j][2] = x.z;
            in[j][3] = x.w;
          }
          const bool odd = ch & 1;
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            // a warp is 16 keys' fours of two chunks: the odd chunk's lanes take the words in the other order, so
            // that their rows lie 16 banks from the even's
            const int word = odd ? w ^ 1 : w;
            uint32_t out[4];
            transpose4(out, odd ? in[0][w ^ 1] : in[0][w], odd ? in[1][w ^ 1] : in[1][w],
                       odd ? in[2][w ^ 1] : in[2][w], odd ? in[3][w ^ 1] : in[3][w]);
            for (int e = 0; e < 4; ++e)
              *reinterpret_cast<uint32_t*>(vt + (ch * 16 + word * 4 + e) * APART + four * 4) = out[e];
          }
        }
      } else {
        for (int i = tid; i < TILE * DC; i += THREADS) {
          const int key = i / DC, ch = i % DC, at = key_at[key];
          qmm_frag::cp16z(kv + key * D + (ch ^ (key % DS)) * 8,
                          static_cast<const __nv_bfloat16*>(VC) + ((size_t)(at >= 0 ? at : 0) * HK + hk) * D +
                              ch * 8,
                          at >= 0);
        }
        qmm_frag::commit();
        qmm_frag::wait<0>();
      }
      __syncthreads();
      {
        float acc[FR][4];
        for (int f = 0; f < FR; ++f)
          for (int e = 0; e < 4; ++e) acc[f][e] = 0.f;
        const int head = lane % 16;
        if (F8) {
          for (int step = 0; step < TILE / 32; ++step) {
            uint32_t a[4];
            qmm_frag::ldmatrix4(a, p8 + head * APART + (step * 2 + (lane >> 4)) * 16);
            // lanes 0-7 address eight values' rows at this step's first 16 keys, 8-15 at its next 16
            for (int f = 0; f < FR; ++f) {
              uint32_t b[2];
              qmm_frag::ldmatrix2(b, vt + ((warp * FR + f) * 8 + (lane & 7)) * APART + step * 32 +
                                         ((lane >> 3) & 1) * 16);
              mma8(acc[f], a, b[0], b[1]);
            }
          }
          for (int f = 0; f < FR; ++f) {
            acc[f][0] *= unit[0];
            acc[f][1] *= unit[0];
            acc[f][2] *= unit[1];
            acc[f][3] *= unit[1];
          }
        } else {
          for (int step = 0; step < TILE / 16; ++step) {
            uint32_t a[4];
            qmm_frag::ldmatrix4(a, p + head * TILE + ((step * 2 + (lane >> 4)) ^ (head % 8)) * 8);
            // lanes 0-7 address this step's first eight keys, 8-15 its next eight
            const int key = step * 16 + (lane & 15);
            for (int f = 0; f < FR; ++f) {
              uint32_t b[2];
              ldmatrix2t(b, kv + key * D + ((warp * FR + f) ^ (key % DS)) * 8);
              qmm_frag::mma(acc[f], a, b[0], b[1]);
            }
          }
        }
        for (int h = 0; h < 2; ++h) {
          const int at = row8 + 8 * h;
          l[h] = l[h] * alpha[h] + (((part_l[0][at] + part_l[1][at]) + part_l[2][at]) + part_l[3][at]);
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
    // the chunk's partial (attention.cu's PO, PM, PL) mixed into the row's, attn_merge_kernel's step
    for (int h = 0; h < 2; ++h) {
      const bool active = l[h] > 0.f;
      const float next_m = active ? fmaxf(row_m[h], m[h]) : row_m[h];
      const float a = active ? (row_m[h] == minus_inf() ? 0.f : expf(row_m[h] - next_m)) : 1.f;
      const float b = active ? expf(m[h] - next_m) : 0.f;
      for (int f = 0; f < FR; ++f)
        for (int e = 2 * h; e < 2 * h + 2; ++e) row_o[f][e] = __fmaf_rn(row_o[f][e], a, __fmul_rn(o[f][e], b));
      row_l[h] = __fmaf_rn(row_l[h], a, __fmul_rn(l[h], b));
      row_m[h] = next_m;
    }
  }
  const size_t base = (size_t)r * H + hk * G;
  for (int f = 0; f < FR; ++f) {
    const int d = (warp * FR + f) * 8 + pair * 2;
    for (int h = 0; h < 2; ++h) {
      const int head = row8 + 8 * h;
      if (head < G) {
        __nv_bfloat162 two;
        two.x = __float2bfloat16_rn(__fdiv_rn(row_o[f][2 * h], row_l[h]));
        two.y = __float2bfloat16_rn(__fdiv_rn(row_o[f][2 * h + 1], row_l[h]));
        *reinterpret_cast<__nv_bfloat162*>(OUT + (base + head) * D + d) = two;
      }
    }
  }
}

bool is(const zs::View& v, int dtype, int dims) {
  return v.defined() && v.dtype() == dtype && v.dim() == dims && v.is_contiguous();
}

}  // namespace

extern "C" {

// zs_attn_attend's arguments and meaning (attention.cu) without its partials, on the contiguous caches: the
// same out, bit for bit, in one launch. ZS_ROWS: zs_attn_attend_rows' (attention_rows.cu) likewise, kc and vc
// the pools that bases point into, pos int32 [R].
#ifdef ZS_ROWS
int zs_attn_attend_window_rows(const zs_view* sid_, const zs_view* tables_, const zs_view* bases_,
                               const zs_view* q_, const zs_view* kc_, const zs_view* vc_, const zs_view* pos_,
#else
int zs_attn_attend_window(const zs_view* q_, const zs_view* kc_, const zs_view* vc_, const zs_view* pos_,
#endif
                          const zs_view* out_, const zs_view* ids_, const zs_view* nk_, const zs_view* sparse_,
                          const zs_view* qs_, const zs_view* ks_, const zs_view* vs_, double scale, int64_t context) {
  zs::begin();
  const zs::View q(q_), kc(kc_), vc(vc_), pos(pos_), out(out_), ids(ids_), nk(nk_), sparse(sparse_), qs(qs_),
      ks(ks_), vs(vs_);
  const bool f8 = kc.defined() && kc.dtype() == ZS_E4M3;
  const int stored = f8 ? ZS_E4M3 : ZS_BF16;
  ZS_REQUIRE(is(q, stored, 3), "q: [R, H, D] in the cache's dtype");
  const int64_t rows = q.size(0), h = q.size(1), d = q.size(2);
  ZS_REQUIRE(is(kc, stored, 3) && kc.size(2) == d && kc.size(1) >= 1 && h % kc.size(1) == 0,
             "kc: bf16 or e4m3 [capacity, HK, D], HK dividing H");
  const bool v4 = f8 && vc.defined() && vc.dtype() == ZS_U8;
  ZS_REQUIRE(is(vc, v4 ? ZS_U8 : stored, 3) && vc.size(0) == kc.size(0) && vc.size(1) == kc.size(1) &&
             vc.size(2) == (v4 ? nvfp4_width(static_cast<int>(d)) : d),
             "vc: kc's shape and dtype, or beside e4m3 keys uint8 [capacity, HK, NVFP4 bytes of a head]");
  const int64_t hk = kc.size(1), g = h / hk;
  if (f8) {
    ZS_REQUIRE(qs.defined() && qs.dtype() == ZS_F32 && qs.is_contiguous() && qs.numel() >= rows * h,
               "qs: fp32 [R, H]");
    ZS_REQUIRE(is(ks, ZS_F32, 2) && ks.size(0) == kc.size(0) && ks.size(1) == hk && is(vs, ZS_F32, 2) &&
               vs.size(0) == kc.size(0) && vs.size(1) == hk, "ks, vs: fp32 [capacity, HK]");
  } else {
    ZS_REQUIRE(!qs.defined() && !ks.defined() && !vs.defined(), "qs, ks and vs come with an e4m3 cache");
  }
  ZS_REQUIRE(rows >= 1 && (d == 32 || d == 64 || d == 128 || d == 256) && g <= MAX_G,
             "32, 64, 128 or 256 values a head, at most 16 query heads a KV head");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * h * d,
             "out: bf16 [R, H, D]");
#ifdef ZS_ROWS
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == rows, "posr: int32 [R]");
  const zs::View sid(sid_), tables(tables_), bases(bases_);
  ZS_REQUIRE(sid.defined() && sid.dtype() == ZS_I32 && sid.is_contiguous() && sid.numel() == rows, "sid: int32 [R]");
  ZS_REQUIRE(tables.defined() && tables.dtype() == ZS_I64 && tables.is_contiguous() && tables.numel() >= 1,
             "tables: int64 [slots], each slot's extent table");
  ZS_REQUIRE(bases.defined() && bases.dtype() == ZS_I64 && bases.is_contiguous() && bases.dim() == 2 &&
             bases.size(1) == 3 && bases.size(0) >= 1, "bases: int64 [slots, 3] of k, v, indexer keys");
  ZS_REQUIRE(kc.size(0) % CHUNK == 0 && context >= 1 && context <= kc.size(0), "the pools hold the furthest row");
#else
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(context >= rows && context <= kc.size(0), "the caches hold pos + R rows");
#endif
  const bool select = ids.defined();
  int64_t idw = 0;
  if (select) {
    ZS_REQUIRE(is(ids, ZS_I32, 2) && ids.size(0) >= rows, "ids: int32 [R, IDW]");
    idw = ids.size(1);
    ZS_REQUIRE(nk.defined() && nk.dtype() == ZS_I32 && nk.numel() >= rows && sparse.defined() &&
               sparse.dtype() == ZS_I32 && sparse.numel() >= rows, "nk, sparse: int32 [R]");
  } else {
    ZS_REQUIRE(!nk.defined() && !sparse.defined(), "ids, nk and sparse come together");
  }
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(hk));
  const auto stream = zs::stream();
  const int* idp = select ? ids.data_ptr<int>() : nullptr;
  const int* nkp = select ? nk.data_ptr<int>() : nullptr;
  const int* spp = select ? sparse.data_ptr<int>() : nullptr;
  const float* qsp = f8 ? qs.data_ptr<float>() : nullptr;
  const float* ksp = f8 ? ks.data_ptr<float>() : nullptr;
  const float* vsp = f8 ? vs.data_ptr<float>() : nullptr;
#ifdef ZS_ROWS
#define ZS_WINDOW_TABLE \
  sid.data_ptr<int>(), reinterpret_cast<const int* const*>(tables.data_ptr<int64_t>()), bases.data_ptr<int64_t>(),
#else
#define ZS_WINDOW_TABLE
#endif
#define ZS_WINDOW(FR, F8, V4)                                                                                    \
  ZS_WINDOW_KERNEL<FR, F8, V4><<<grid, THREADS, 0, stream>>>(                                                    \
      ZS_WINDOW_TABLE q.data_ptr<char>(), kc.data_ptr<char>(), vc.data_ptr<char>(), qsp, ksp, vsp, pos.data_ptr<int>(),          \
      out.data_ptr<__nv_bfloat16>(), idp, nkp, spp, static_cast<float>(scale), static_cast<int>(h),               \
      static_cast<int>(hk), static_cast<int>(g), static_cast<int>(idw))
  switch (d + (f8 ? 1 : 0) + (v4 ? 1 : 0)) {
    case 32: ZS_WINDOW(1, false, false); break;
    case 64: ZS_WINDOW(2, false, false); break;
    case 128: ZS_WINDOW(4, false, false); break;
    case 256: ZS_WINDOW(8, false, false); break;
    case 33: ZS_WINDOW(1, true, false); break;
    case 65: ZS_WINDOW(2, true, false); break;
    case 129: ZS_WINDOW(4, true, false); break;
    case 257: ZS_WINDOW(8, true, false); break;
    case 34: ZS_WINDOW(1, true, true); break;
    case 66: ZS_WINDOW(2, true, true); break;
    case 130: ZS_WINDOW(4, true, true); break;
    default: ZS_WINDOW(8, true, true); break;
  }
#undef ZS_WINDOW
#undef ZS_WINDOW_TABLE
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
