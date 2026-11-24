// The text attention over a prompt window's rows, a group of rows a block (spec 138 WS-06): attention_window.cu's
// chunks and merge in one launch, with the 16 rows of each mma the (row, query head) pairs of 16 / G rows instead
// of one row's G heads, and MT such mmas a block, so a tile of keys and values is loaded once for MT 16 / G rows.
// The bits are attention.cu's: every row's arithmetic is its own, step for step.
// Why: a block of attention_window.cu is a (row, KV head), its mma's 16 rows the G query heads, so at G = 2
// (Qwen3-ASR's language model, 16 query heads over 8) 14 of the 16 rows are zeros and every row loads its keys and
// values for itself. A 600 s clip's prompt (7,862 rows in windows of 512) spent 2.02 s of its 2.8 in that kernel
// (nsys, spec 138 WS-06).
// The arithmetic, attention.cu's to the bit (its attn_chunk_kernel and attn_merge_kernel):
//   a score is an mma's element, a function of its own row of A (the row's head's query) and column of B (the
//     key) over the same m16n8k16 steps in the same order, whatever the other rows of the mma are;
//   the chunks are the row's keys [c CHUNK, (c + 1) CHUNK), its tiles of TILE keys from the chunk's first, warp w
//     the keys w 16 .. w 16 + 15 of a tile, and every sum in attn_chunk_kernel's order: a thread's four p, then
//     shfl_xor 1 and 2, then the four warps in order; p . v over the tile's keys in steps of 16;
//   the chunks mixed in chunk order in registers, as attention_window.cu (its __fmaf_rn and __fmul_rn).
// A group's rows end at different keys. A row's key past its own list is masked as attention.cu masks it (a
// score of -inf, p = 0), though its key and value are loaded for a longer row of the group: p . v adds 0 v
// for it where attention.cu adds 0 0, the same sum. A tile past a row's last key is inactive for it (every
// score -inf), and leaves its m, l and o as they are, as attention.cu never runs that tile; a chunk past it has l
// = 0 and is not mixed, as attn_merge_kernel does not read it.
// bf16 caches only, contiguous, dense rows (no selection lists), G dividing 16, D to 128.
// From attention.cu (TensorFold 0.6.5 609ca41 attention.py's _tile, _chunk and _merge behind it, Apache-2.0).
//
// A block's shared memory (D = 128, MT = 2; 128 threads):
//   | region | bytes | holds | less? |
//   | q | 8,192 | the MT x 16 mma rows' queries, loaded once | no: every tile reads them |
//   | kv | 16,384 | a tile's keys, then its values in their place | the tile itself |
//   | p | 4,096 | a tile's bf16(p) of the MT x 16 rows, the value product's A | no |
//   | part_m, part_l, key_at | 1,280 | the warps' meeting, a tile's keys | no |
//   | total | 29,952 | | 3 blocks an SM |

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/qmm_frag.cuh"
#include "attention_mma.cuh"
#include "../zs/host.h"

namespace {

// Block (group of MT 16 / G rows, KV head) of 4 warps.
template <int FR, int MT>
__global__ void __launch_bounds__(THREADS)
    attn_group_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KC,
                      const __nv_bfloat16* __restrict__ VC, const int* __restrict__ POS0,
                      __nv_bfloat16* __restrict__ OUT, float SCALE, int R, int H, int HK, int G) {
  constexpr int D = 32 * FR;
  constexpr int DC = D / 8;            // 16-byte chunks of a head
  constexpr int DS = DC < 8 ? DC : 8;  // rows a swizzle cycles over
  constexpr int ROWS = MT * HEADS;
  __shared__ __align__(16) __nv_bfloat16 q[ROWS * D];
  __shared__ __align__(16) __nv_bfloat16 kv[TILE * D];
  __shared__ __align__(16) __nv_bfloat16 p[ROWS * TILE];
  __shared__ float part_m[4][ROWS];
  __shared__ float part_l[4][ROWS];
  __shared__ int key_at[TILE];
  const int hk = blockIdx.y, tid = threadIdx.x;
  const int per = HEADS / G;  // rows an mma holds
  const int r0 = blockIdx.x * MT * per;
  const int pos0 = POS0[0];
  const int n_max = pos0 + min(r0 + MT * per, R);
  const int warp = tid / 32, lane = tid % 32, row8 = lane >> 2, pair = lane & 3;
  // mma row i of the block: tile i / 16, its row (i % 16) / G of the tile's, head (i % 16) % G
  for (int i = tid; i < ROWS * DC; i += THREADS) {
    const int row = i / DC, ch = i % DC, w = row % HEADS;
    const int r = r0 + row / HEADS * per + w / G;
    uint4 x = make_uint4(0, 0, 0, 0);
    if (r < R) x = *reinterpret_cast<const uint4*>(Q + ((size_t)r * H + hk * G + w % G) * D + ch * 8);
    *reinterpret_cast<uint4*>(q + (row * DC + (ch ^ (row % DS))) * 8) = x;
  }
  // the keys of this thread's rows (t, h): mma row t 16 + row8 + 8 h
  int n[MT][2];
  for (int t = 0; t < MT; ++t)
    for (int h = 0; h < 2; ++h) {
      const int w = row8 + 8 * h, r = r0 + t * per + w / G;
      n[t][h] = r < R ? pos0 + r + 1 : 0;
    }
  float row_m[MT][2], row_l[MT][2], row_o[MT][FR][4];
  for (int t = 0; t < MT; ++t) {
    for (int h = 0; h < 2; ++h) {
      row_m[t][h] = minus_inf();
      row_l[t][h] = 0.f;
    }
    for (int f = 0; f < FR; ++f)
      for (int e = 0; e < 4; ++e) row_o[t][f][e] = 0.f;
  }
  for (int start = 0; start < n_max; start += CHUNK) {
    float o[MT][FR][4], m[MT][2], l[MT][2], alpha[MT][2];
    bool active[MT][2];
    for (int t = 0; t < MT; ++t) {
      for (int h = 0; h < 2; ++h) {
        m[t][h] = minus_inf();
        l[t][h] = 0.f;
        alpha[t][h] = 1.f;
      }
      for (int f = 0; f < FR; ++f)
        for (int e = 0; e < 4; ++e) o[t][f][e] = 0.f;
    }
    const int keys = min(n_max - start, CHUNK);
    for (int t0 = 0; t0 < keys; t0 += TILE) {
      if (tid < TILE) {
        const int ki = start + t0 + tid;
        key_at[tid] = ki < n_max ? ki : -1;
      }
      __syncthreads();
      for (int i = tid; i < TILE * DC; i += THREADS) {
        const int key = i / DC, ch = i % DC, at = key_at[key];
        qmm_frag::cp16z(kv + (key * DC + (ch ^ (key % DS))) * 8,
                        KC + ((size_t)(at >= 0 ? at : 0) * HK + hk) * D + ch * 8, at >= 0);
      }
      qmm_frag::commit();
      qmm_frag::wait<0>();
      __syncthreads();
      {
        // this warp's 16 keys: lanes 0-15 address the first eight, 16-31 the next; odd eights of lanes the high 8
        float c0[MT][4], c1[MT][4];
        for (int t = 0; t < MT; ++t)
          for (int e = 0; e < 4; ++e) c0[t][e] = c1[t][e] = 0.f;
        const int head = lane % 16, key = warp * 16 + (lane & 7) + (lane >> 4) * 8;
        const __nv_bfloat16* k_row = kv + key * D;
        for (int step = 0; step < DC / 2; ++step) {
          uint32_t b[4];
          qmm_frag::ldmatrix4(b, k_row + ((step * 2 + ((lane >> 3) & 1)) ^ (key % DS)) * 8);
          for (int t = 0; t < MT; ++t) {
            uint32_t a[4];
            qmm_frag::ldmatrix4(a, q + (t * HEADS + head) * D + ((step * 2 + (lane >> 4)) ^ (head % DS)) * 8);
            qmm_frag::mma(c0[t], a, b[0], b[1]);
            qmm_frag::mma(c1[t], a, b[2], b[3]);
          }
        }
        const int k0 = warp * 16 + pair * 2;
        const int mine[4] = {k0, k0 + 1, k0 + 8, k0 + 9};
        float sc[MT][2][4];
        bool live[MT][2][4];
        for (int t = 0; t < MT; ++t) {
          const float in[2][4] = {{c0[t][0], c0[t][1], c1[t][0], c1[t][1]}, {c0[t][2], c0[t][3], c1[t][2], c1[t][3]}};
          for (int h = 0; h < 2; ++h) {
            for (int j = 0; j < 4; ++j) {
              live[t][h][j] = start + t0 + mine[j] < n[t][h];
              sc[t][h][j] = live[t][h][j] ? in[h][j] * SCALE : minus_inf();
            }
            float top = fmaxf(fmaxf(sc[t][h][0], sc[t][h][1]), fmaxf(sc[t][h][2], sc[t][h][3]));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
            if (pair == 0) part_m[warp][t * HEADS + row8 + 8 * h] = top;
          }
        }
        __syncthreads();
        for (int t = 0; t < MT; ++t)
          for (int h = 0; h < 2; ++h) {
            const int head8 = row8 + 8 * h, at = t * HEADS + head8;
            const float tile_m = fmaxf(fmaxf(part_m[0][at], part_m[1][at]), fmaxf(part_m[2][at], part_m[3][at]));
            const bool act = tile_m != minus_inf();
            active[t][h] = act;
            const float next_m = act ? fmaxf(m[t][h], tile_m) : m[t][h];
            alpha[t][h] = act ? (m[t][h] == minus_inf() ? 0.f : expf(m[t][h] - next_m)) : 1.f;
            m[t][h] = next_m;
            float pk[4];
            for (int j = 0; j < 4; ++j) pk[j] = act && live[t][h][j] ? expf(sc[t][h][j] - next_m) : 0.f;
            float sum = ((pk[0] + pk[1]) + pk[2]) + pk[3];
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            if (pair == 0) part_l[warp][at] = sum;
            for (int j = 0; j < 4; ++j)
              p[at * TILE + ((mine[j] / 8) ^ (head8 % 8)) * 8 + mine[j] % 8] = __float2bfloat16_rn(pk[j]);
          }
      }
      // the values take the keys' place: the scores above are the last thing to read them
      for (int i = tid; i < TILE * DC; i += THREADS) {
        const int key = i / DC, ch = i % DC, at = key_at[key];
        qmm_frag::cp16z(kv + key * D + (ch ^ (key % DS)) * 8, VC + ((size_t)(at >= 0 ? at : 0) * HK + hk) * D + ch * 8,
                        at >= 0);
      }
      qmm_frag::commit();
      qmm_frag::wait<0>();
      __syncthreads();
      {
        const int head = lane % 16;
        for (int t = 0; t < MT; ++t) {
          float acc[FR][4];
          for (int f = 0; f < FR; ++f)
            for (int e = 0; e < 4; ++e) acc[f][e] = 0.f;
          for (int step = 0; step < TILE / 16; ++step) {
            uint32_t a[4];
            qmm_frag::ldmatrix4(a, p + (t * HEADS + head) * TILE + ((step * 2 + (lane >> 4)) ^ (head % 8)) * 8);
            // lanes 0-7 address this step's first eight keys, 8-15 its next eight
            const int key = step * 16 + (lane & 15);
            for (int f = 0; f < FR; ++f) {
              uint32_t b[2];
              ldmatrix2t(b, kv + key * D + ((warp * FR + f) ^ (key % DS)) * 8);
              qmm_frag::mma(acc[f], a, b[0], b[1]);
            }
          }
          for (int h = 0; h < 2; ++h) {
            if (!active[t][h]) continue;
            const int at = t * HEADS + row8 + 8 * h;
            l[t][h] = l[t][h] * alpha[t][h] + (((part_l[0][at] + part_l[1][at]) + part_l[2][at]) + part_l[3][at]);
            const float a = alpha[t][h];
            for (int f = 0; f < FR; ++f) {
              o[t][f][2 * h] = o[t][f][2 * h] * a + acc[f][2 * h];
              o[t][f][2 * h + 1] = o[t][f][2 * h + 1] * a + acc[f][2 * h + 1];
            }
          }
        }
      }
      __syncthreads();
    }
    // the chunk's partial mixed into the row's, attn_merge_kernel's step (attention_window.cu's form)
    for (int t = 0; t < MT; ++t)
      for (int h = 0; h < 2; ++h) {
        const bool act = l[t][h] > 0.f;
        const float next_m = act ? fmaxf(row_m[t][h], m[t][h]) : row_m[t][h];
        const float a = act ? (row_m[t][h] == minus_inf() ? 0.f : expf(row_m[t][h] - next_m)) : 1.f;
        const float b = act ? expf(m[t][h] - next_m) : 0.f;
        for (int f = 0; f < FR; ++f)
          for (int e = 2 * h; e < 2 * h + 2; ++e)
            row_o[t][f][e] = __fmaf_rn(row_o[t][f][e], a, __fmul_rn(o[t][f][e], b));
        row_l[t][h] = __fmaf_rn(row_l[t][h], a, __fmul_rn(l[t][h], b));
        row_m[t][h] = next_m;
      }
  }
  for (int t = 0; t < MT; ++t)
    for (int h = 0; h < 2; ++h) {
      const int w = row8 + 8 * h, r = r0 + t * per + w / G;
      if (r >= R) continue;
      __nv_bfloat16* out = OUT + ((size_t)r * H + hk * G + w % G) * D;
      for (int f = 0; f < FR; ++f) {
        const int d = (warp * FR + f) * 8 + pair * 2;
        __nv_bfloat162 two;
        two.x = __float2bfloat16_rn(__fdiv_rn(row_o[t][f][2 * h], row_l[t][h]));
        two.y = __float2bfloat16_rn(__fdiv_rn(row_o[t][f][2 * h + 1], row_l[t][h]));
        *reinterpret_cast<__nv_bfloat162*>(out + d) = two;
      }
    }
}

bool is(const zs::View& v, int dtype, int dims) {
  return v.defined() && v.dtype() == dtype && v.dim() == dims && v.is_contiguous();
}

}  // namespace

extern "C" {

// zs_attn_attend's arguments and meaning (attention.cu) for dense rows on bf16 contiguous caches, without its
// partials: the same out, bit for bit, in one launch. G = H / HK dividing 16, D 32, 64 or 128.
int zs_attn_attend_group(const zs_view* q_, const zs_view* kc_, const zs_view* vc_, const zs_view* pos_,
                         const zs_view* out_, double scale, int64_t context) {
  zs::begin();
  const zs::View q(q_), kc(kc_), vc(vc_), pos(pos_), out(out_);
  ZS_REQUIRE(is(q, ZS_BF16, 3), "q: bf16 [R, H, D]");
  const int64_t rows = q.size(0), h = q.size(1), d = q.size(2);
  ZS_REQUIRE(is(kc, ZS_BF16, 3) && kc.size(2) == d && kc.size(1) >= 1 && h % kc.size(1) == 0,
             "kc: bf16 [capacity, HK, D], HK dividing H");
  ZS_REQUIRE(is(vc, ZS_BF16, 3) && vc.size(0) == kc.size(0) && vc.size(1) == kc.size(1) && vc.size(2) == d,
             "vc: kc's shape");
  const int64_t hk = kc.size(1), g = h / hk;
  ZS_REQUIRE(rows >= 1 && rows <= INT32_MAX && (d == 32 || d == 64 || d == 128) && HEADS % g == 0,
             "32, 64 or 128 values a head, query heads a KV head dividing 16");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * h * d,
             "out: bf16 [R, H, D]");
  ZS_REQUIRE(context >= rows && context <= kc.size(0), "the caches hold pos + R rows");
  constexpr int MT = 2;
  const int64_t group = MT * (HEADS / g);
  const dim3 grid(static_cast<unsigned>((rows + group - 1) / group), static_cast<unsigned>(hk));
  const auto stream = zs::stream();
#define ZS_GROUP(FR)                                                                                             \
  attn_group_kernel<FR, MT><<<grid, THREADS, 0, stream>>>(                                                       \
      q.data_ptr<__nv_bfloat16>(), kc.data_ptr<__nv_bfloat16>(), vc.data_ptr<__nv_bfloat16>(), pos.data_ptr<int>(), \
      out.data_ptr<__nv_bfloat16>(), static_cast<float>(scale), static_cast<int>(rows), static_cast<int>(h),      \
      static_cast<int>(hk), static_cast<int>(g))
  switch (d) {
    case 32: ZS_GROUP(1); break;
    case 64: ZS_GROUP(2); break;
    default: ZS_GROUP(4); break;
  }
#undef ZS_GROUP
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
