// The text attention, decode and prompt rows (spec 138 WS-04: moved from zig/kernels/qwen4/attention.cu, the
// kernels' text unchanged and the entries renamed by layer, when Qwen3-ASR's language model became its second
// family; Flash Next's prep and gate stay there). A row's online softmax over its keys in fixed chunks, then the
// chunks merged: the chunks split a row's key list by index alone and are merged in chunk order, so neither the
// window, the cache's capacity nor the context bound of a launch changes a bit. The selection that builds a sparse
// row's key list is qwen4/qsa.cu's.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/attention.py (_tile, _chunks, _chunk, _merge,
// _merge_row), Triton, Apache-2.0; rewritten for zspark in CUDA C++: the cache as bf16 or as FP8 (below), not
// TensorFold's int8 and int4. Where Triton left a reduction's order to its compiler (a tile's products) the order
// here is fixed and stated at the kernel.
// Hand-written because the attention is not a GEMM CUTLASS has: each row reads its own gathered key list, masks
// it by its own length, and carries an online softmax from tile to tile. Its products are m16n8k16 steps written
// by hand (common/qmm_frag.cuh), the heads of a KV head as an mma's rows.
// The FP8 cache is zspark's own: a head's keys and values as e4m3 codes of x / s, s the head's largest magnitude
// over 448, an fp32 a (position, KV head); the queries likewise. The GB10 multiplies e4m3 by e4m3 on its tensor
// cores (m16n8k32), so both products run on the stored codes, half the bytes and half the steps of bf16: q . k as
// stored, p . v with p as e4m3 too and a tile's value codes transposed, since they lie along a head and that
// product runs along the keys.
// Beside FP8 keys the values may be NVFP4 (e2m1 codes, an e4m3 scale a 16 values, the head's scale over both):
// 4.5 bits a value. They become e4m3 codes as a tile is loaded, a rounding of its own, and the product is the FP8
// one: the block-scaled FP4 mma would want the softmax weights as e2m1 too, and the scales lie across the keys it
// sums over.

#include "attention.cuh"

namespace {

// Block (row, kv head, chunk) of 4 warps: the row's keys [c CHUNK, (c + 1) CHUNK) of its list against the G query
// heads of the KV head, a TILE at a time, on the tensor cores. A dense row's list is positions 0 .. POS0 + r; a
// sparse row's is IDS[r] (NK[r] keys). The heads are the 16 rows of every mma, zero queries past G. Each tile:
//   score = (q . k) * SCALE, q . k one fp32 chain of m16n8k16 steps over the head's values, 16 at a step;
//   m' = max(m, tile max); p = exp(score - m'); l = l exp(m - m') + sum p;
//   o = o exp(m - m') + (bf16(p) . v, one fp32 chain of steps over the tile's keys, 16 at a step).
// The softmax runs where the scores land: a thread holds two heads' scores of four keys (k, k + 1, k + 8, k + 9,
// k even), so sum p adds those four in that order, then the four threads of a warp's 16 keys as a tree (pairs,
// then pairs of pairs), then the four warps in order. Every thread of a head carries its m, l and exp(m - m').
// A key past the row's list is zeros with p = 0. An mma's element is a function of its own row of A and column
// of B, so a head's values depend on neither the other heads nor the warp that holds them. The chunk's o, m and l
// go to PO [R, NCH, H, D], PM and PL [R, NCH, H]; a chunk past the row's keys writes nothing.
// FR: the 8-value fragments of a head a warp carries, D = 32 FR. Shared rows are swizzled by 16-byte chunks so
// ldmatrix's eight rows fall in eight bank groups. A tile's keys and bf16 values come by cp.async, zeros past the
// row's keys: through registers the copies were over half the kernel's time. What a block holds in shared memory
// bounds the blocks an SM runs, so the FP8 tile keeps no second copy of its values.
// F8, the FP8 cache: Q, KC and VC are e4m3 codes with the scales QS [R, H], KS and VS [capacity, HK].
//   score = (code q . code k, m16n8k32 steps, 32 codes at a step) * s_k * (s_q SCALE);
//   o takes u (e4m3(p s_v / u) . code v, m16n8k32 steps over the tile's keys, 32 at a step), u a 448th of
//   exp(tile max - m') times the tile's largest s_v: no p s_v of the tile exceeds 448 u. l sums p as above.
//   A p s_v of zero has the code zero whatever u is: 448 / u is infinite where the tile lies far under m'
//   (spec 043: zero times that was a head of NaN).
// The tile's value codes are held transposed, [D, TILE], in rows 80 bytes apart, as are p's codes [HEADS, TILE]:
// eight such rows fall in eight bank groups without a swizzle.
// V4, values as NVFP4 (attn_prep_kernel's layout) beside FP8 keys: a value becomes the e4m3 code nearest
// (its e2m1 magnitude) (its 16's scale) / 6 as it is loaded, so that code times s_v stands for it and the product
// is the FP8 one. BY_CODE holds a scale's seven codes; a value picks its own.
template <int FR, bool F8, bool V4>
__global__ void __launch_bounds__(THREADS)
    ZS_NAME(chunk)(
#ifdef ZS_ROWS
                 const int* __restrict__ SID, const int* const* __restrict__ TABLES,
                 const int64_t* __restrict__ BASES,
#else
                 ZS_TABLE_PARAM
#endif
                 const void* __restrict__ Q, const void* __restrict__ KC_ARG, const void* __restrict__ VC_ARG,
                 const float* __restrict__ QS, const float* __restrict__ KS, const float* __restrict__ VS,
                 const int* __restrict__ POS0, float* __restrict__ PO, float* __restrict__ PM,
                 float* __restrict__ PL, const int* __restrict__ IDS, const int* __restrict__ NK,
                 const int* __restrict__ SPARSE, float SCALE, int H, int HK, int G, int NCH, int IDW) {
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
  const int r = blockIdx.x, hk = blockIdx.y, c = blockIdx.z, tid = threadIdx.x;
#ifdef ZS_ROWS
  const void* KC = reinterpret_cast<const void*>(BASES[(size_t)SID[r] * 3]);
  const void* VC = reinterpret_cast<const void*>(BASES[(size_t)SID[r] * 3 + 1]);
  const int* TABLE = TABLES[SID[r]];
  int n = POS0[r] + 1;
#else
  const void* KC = KC_ARG;
  const void* VC = VC_ARG;
  int n = POS0[0] + r + 1;
#endif
  bool sparse = false;
  if (NK != nullptr && SPARSE[r] != 0) {
    sparse = true;
    n = NK[r];
  }
  const int start = c * CHUNK;
  if (start >= n) return;
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
  float unit[2] = {0.f, 0.f};
  float o[FR][4];
  for (int f = 0; f < FR; ++f)
    for (int e = 0; e < 4; ++e) o[f][e] = 0.f;
  // of heads row8 and row8 + 8
  float m[2] = {minus_inf(), minus_inf()}, l[2] = {0.f, 0.f}, alpha[2] = {1.f, 1.f};
  const int keys = min(n - start, CHUNK);
#ifdef ZS_EXTENTS
  // a dense row's chunk c is positions [c CHUNK, (c + 1) CHUNK): one extent
  const int dense_base = !sparse && tid < TILE ? TABLE[c] << EXTENT_BITS : 0;
#endif
  for (int t0 = 0; t0 < keys; t0 += TILE) {
    if (tid < TILE) {
      const int ki = start + t0 + tid;
#ifdef ZS_EXTENTS
      const int at = ki < n ? (sparse ? IDS[(size_t)r * IDW + ki] : dense_base | (t0 + tid)) : -1;
#else
      const int at = ki < n ? (sparse ? IDS[(size_t)r * IDW + ki] : ki) : -1;
#endif
      key_at[tid] = at;
      if (F8) {
        k_scale[tid] = at >= 0 ? KS[(size_t)at * HK + hk] : 0.f;
        float top = at >= 0 ? VS[(size_t)at * HK + hk] : 0.f;
        v_scale[tid] = top;
        // these are the first two warps, whole
        for (int other = 16; other > 0; other >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, other));
        if (lane == 0) v_top[warp] = top;
      }
    }
    __syncthreads();
    for (int i = tid; i < TILE * QC; i += THREADS) {
      const int key = i / QC, ch = i % QC, at = key_at[key];
      qmm_frag::cp16z(reinterpret_cast<char*>(kv) + (key * QC + (ch ^ (key % QS8))) * 16,
                      static_cast<const char*>(KC) + (((size_t)(at >= 0 ? at : 0) * HK + hk) * QC + ch) * 16, at >= 0);
    }
    qmm_frag::commit();
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
          // 448 / most is past fp32 for a most under 1.3e-36 (a tile 77 or more under the row's largest score,
          // or values that small): inv is then infinite, a weight that is not zero takes the largest code
          // under a unit of 2.9e-39 at most, and a weight of zero (a key past the row's list, values of zeros
          // with their scale of zero, a p that underflowed) times inv is NaN. fmaxf gives its other operand
          // for a NaN, so that code is zero, as it is wherever inv is finite; no other product changes, none
          // being under zero.
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
        const int four = i / QC, ch = i % QC;
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
          } else if (at >= 0) {
            x = *reinterpret_cast<const uint4*>(static_cast<const char*>(VC) +
                                                (((size_t)at * HK + hk) * QC + ch) * 16);
          }
          in[j][0] = x.x;
          in[j][1] = x.y;
          in[j][2] = x.z;
          in[j][3] = x.w;
        }
        for (int word = 0; word < 4; ++word) {
          uint32_t out[4];
          transpose4(out, in[0][word], in[1][word], in[2][word], in[3][word]);
          for (int e = 0; e < 4; ++e)
            *reinterpret_cast<uint32_t*>(vt + (ch * 16 + word * 4 + e) * APART + four * 4) = out[e];
        }
      }
    } else {
      for (int i = tid; i < TILE * DC; i += THREADS) {
        const int key = i / DC, ch = i % DC, at = key_at[key];
        qmm_frag::cp16z(kv + key * D + (ch ^ (key % DS)) * 8,
                        static_cast<const __nv_bfloat16*>(VC) + ((size_t)(at >= 0 ? at : 0) * HK + hk) * D + ch * 8,
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
            qmm_frag::ldmatrix2(b, vt + ((warp * FR + f) * 8 + (lane & 7)) * APART + step * 32 + ((lane >> 3) & 1) * 16);
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
  const size_t base = ((size_t)r * NCH + c) * H + hk * G;
  for (int f = 0; f < FR; ++f) {
    const int d = (warp * FR + f) * 8 + pair * 2;
    if (row8 < G) {
      PO[(base + row8) * D + d] = o[f][0];
      PO[(base + row8) * D + d + 1] = o[f][1];
    }
    if (row8 + 8 < G) {
      PO[(base + row8 + 8) * D + d] = o[f][2];
      PO[(base + row8 + 8) * D + d + 1] = o[f][3];
    }
  }
  if (warp == 0 && pair == 0) {
    if (row8 < G) {
      PM[base + row8] = m[0];
      PL[base + row8] = l[0];
    }
    if (row8 + 8 < G) {
      PM[base + row8 + 8] = m[1];
      PL[base + row8 + 8] = l[1];
    }
  }
}

// Thread (row, head, d): the row's chunk partials merged in chunk order, then bf16(o / l).
//   m' = max(m, m_c); o = o exp(m - m') + o_c exp(m_c - m'); l likewise.
__global__ void ZS_NAME(merge)(const float* __restrict__ PO, const float* __restrict__ PM, const float* __restrict__ PL,
                             const int* __restrict__ POS0, __nv_bfloat16* __restrict__ OUT,
                             const int* __restrict__ NK, const int* __restrict__ SPARSE, int R, int H, int D,
                             int NCH) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * H * D) return;
  const int d = static_cast<int>(i % D), h = static_cast<int>(i / D % H), r = static_cast<int>(i / D / H);
#ifdef ZS_ROWS
  int n = POS0[r] + 1;
#else
  int n = POS0[0] + r + 1;
#endif
  if (NK != nullptr && SPARSE[r] != 0) n = NK[r];
  const int chunks = min((n + CHUNK - 1) / CHUNK, NCH);
  float m = minus_inf(), l = 0.f, o = 0.f;
  for (int c = 0; c < chunks; ++c) {
    const size_t base = ((size_t)r * NCH + c) * H + h;
    const float cm = PM[base], cl = PL[base], co = PO[base * D + d];
    const bool active = cl > 0.f;
    const float next_m = active ? fmaxf(m, cm) : m;
    const float a = active ? (m == minus_inf() ? 0.f : expf(m - next_m)) : 1.f;
    const float b = active ? expf(cm - next_m) : 0.f;
    o = o * a + co * b;
    l = l * a + cl * b;
    m = next_m;
  }
  OUT[i] = __float2bfloat16_rn(o / l);
}

}  // namespace

extern "C" {

// q bf16 [R, H, D] against kc, vc bf16 [capacity, HK, D] holding rows 0 .. pos + R - 1 -> out bf16 [R, H, D].
// Row r attends positions 0 .. pos + r, or, with the selection's ids int32 [R, IDW], nk and sparse int32 [R] (all
// three or none), its ids where sparse. po fp32 [R, NCH, H, D], pm and pl fp32 [R, NCH, H] are scratch: NCH chunks
// of 512 keys must cover the longest list. ``context`` (pos + R on the host) bounds the chunks launched.
// The FP8 cache: q, kc and vc are e4m3 with qs fp32 [R, H], ks and vs fp32 [capacity, HK]; null for bf16.
// vc may be NVFP4 beside e4m3 keys: uint8 [capacity, HK, bytes of a head], as prep wrote it.
// With ZS_EXTENTS the caches are pools behind table, as prep's. zs_attn_attend_table: ids holds the pool's
// rows (zs_qwen4_attn_select_table wrote them).
#if defined(ZS_ROWS)
int zs_attn_attend_rows(const zs_view* sid_, const zs_view* tables_, const zs_view* bases_, const zs_view* q_,
                              const zs_view* kc_,
#elif defined(ZS_EXTENTS)
int zs_attn_attend_table(const zs_view* table_, const zs_view* q_, const zs_view* kc_,
#else
int zs_attn_attend(const zs_view* q_, const zs_view* kc_,
#endif
                      const zs_view* vc_, const zs_view* pos_,
                      const zs_view* po_, const zs_view* pm_, const zs_view* pl_, const zs_view* out_,
                      const zs_view* ids_, const zs_view* nk_, const zs_view* sparse_, const zs_view* qs_,
                      const zs_view* ks_, const zs_view* vs_, double scale, int64_t context) {
  zs::begin();
  const zs::View q(q_), kc(kc_), vc(vc_), pos(pos_), po(po_), pm(pm_), pl(pl_), out(out_), ids(ids_), nk(nk_),
      sparse(sparse_), qs(qs_), ks(ks_), vs(vs_);
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
#ifdef ZS_ROWS
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == rows, "posr: int32 [R]");
#else
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
#endif
  ZS_REQUIRE(is(po, ZS_F32, 4) && po.size(0) >= rows && po.size(2) == h && po.size(3) == d && po.size(1) >= 1,
             "po: fp32 [R, NCH, H, D]");
  const int64_t nch = po.size(1);
  ZS_REQUIRE(pm.defined() && pm.dtype() == ZS_F32 && pm.numel() >= po.size(0) * nch * h && pl.defined() &&
             pl.dtype() == ZS_F32 && pl.numel() >= po.size(0) * nch * h, "pm, pl: fp32 [R, NCH, H]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * h * d,
             "out: bf16 [R, H, D]");
#if defined(ZS_ROWS)
  const zs::View sid(sid_), tables(tables_), bases(bases_);
  ZS_REQUIRE(sid.defined() && sid.dtype() == ZS_I32 && sid.is_contiguous() && sid.numel() == rows, "sid: int32 [R]");
  ZS_REQUIRE(tables.defined() && tables.dtype() == ZS_I64 && tables.is_contiguous() && tables.numel() >= 1,
             "tables: int64 [slots], each slot's extent table");
  ZS_REQUIRE(bases.defined() && bases.dtype() == ZS_I64 && bases.is_contiguous() && bases.dim() == 2 &&
             bases.size(1) == 3 && bases.size(0) >= 1, "bases: int64 [slots, 3] of k, v, indexer keys");
  ZS_REQUIRE(kc.size(0) % CHUNK == 0 && context >= 1 && context <= kc.size(0), "the pools hold the furthest row");
#define ZS_TABLE sid.data_ptr<int>(), reinterpret_cast<const int* const*>(tables.data_ptr<int64_t>()), bases.data_ptr<int64_t>(),
#elif defined(ZS_EXTENTS)
  const zs::View table(table_);
  ZS_REQUIRE(table.defined() && table.dtype() == ZS_I32 && table.is_contiguous() && kc.size(0) % CHUNK == 0 &&
             table.numel() <= kc.size(0) / CHUNK && context >= rows && context <= table.numel() * CHUNK,
             "table: int32 [extents] over pools of 512-row extents, holding pos + R");
#define ZS_TABLE table.data_ptr<int>(),
#else
  ZS_REQUIRE(context >= rows && context <= kc.size(0), "the caches hold pos + R rows");
#define ZS_TABLE
#endif
  const bool select = ids.defined();
  int64_t idw = 0, longest = context;
  if (select) {
    ZS_REQUIRE(is(ids, ZS_I32, 2) && ids.size(0) >= rows, "ids: int32 [R, IDW]");
    idw = ids.size(1);
    ZS_REQUIRE(nk.defined() && nk.dtype() == ZS_I32 && nk.numel() >= rows && sparse.defined() &&
               sparse.dtype() == ZS_I32 && sparse.numel() >= rows, "nk, sparse: int32 [R]");
    if (longest > idw - 1) longest = idw - 1;      // a dense row under a selection holds fewer keys than IDW
  } else {
    ZS_REQUIRE(!nk.defined() && !sparse.defined(), "ids, nk and sparse come together");
  }
  ZS_REQUIRE(longest <= nch * CHUNK, "po's chunks cover the longest key list");
  const int64_t chunks = (longest + CHUNK - 1) / CHUNK;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(hk), static_cast<unsigned>(chunks));
  const auto stream = zs::stream();
  const int* idp = select ? ids.data_ptr<int>() : nullptr;
  const int* nkp = select ? nk.data_ptr<int>() : nullptr;
  const int* spp = select ? sparse.data_ptr<int>() : nullptr;
  const float* qsp = f8 ? qs.data_ptr<float>() : nullptr;
  const float* ksp = f8 ? ks.data_ptr<float>() : nullptr;
  const float* vsp = f8 ? vs.data_ptr<float>() : nullptr;
#define ZS_CHUNKS(FR, F8, V4)                                                                                    \
  ZS_NAME(chunk)<FR, F8, V4><<<grid, THREADS, 0, stream>>>(                                                       \
      ZS_TABLE q.data_ptr<char>(), kc.data_ptr<char>(), vc.data_ptr<char>(), qsp, ksp, vsp, pos.data_ptr<int>(), \
      po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), idp, nkp, spp,                          \
      static_cast<float>(scale), static_cast<int>(h), static_cast<int>(hk), static_cast<int>(g),                \
      static_cast<int>(nch), static_cast<int>(idw))
  switch (d + (f8 ? 1 : 0) + (v4 ? 1 : 0)) {
    case 32: ZS_CHUNKS(1, false, false); break;
    case 64: ZS_CHUNKS(2, false, false); break;
    case 128: ZS_CHUNKS(4, false, false); break;
    case 256: ZS_CHUNKS(8, false, false); break;
    case 33: ZS_CHUNKS(1, true, false); break;
    case 65: ZS_CHUNKS(2, true, false); break;
    case 129: ZS_CHUNKS(4, true, false); break;
    case 257: ZS_CHUNKS(8, true, false); break;
    case 34: ZS_CHUNKS(1, true, true); break;
    case 66: ZS_CHUNKS(2, true, true); break;
    case 130: ZS_CHUNKS(4, true, true); break;
    default: ZS_CHUNKS(8, true, true); break;
  }
#undef ZS_CHUNKS
#undef ZS_TABLE
  ZS_CUDA_LAUNCH_CHECK();
  ZS_NAME(merge)<<<blocks(rows * h * d), 256, 0, stream>>>(
      po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), pos.data_ptr<int>(),
      out.data_ptr<__nv_bfloat16>(), nkp, spp, static_cast<int>(rows), static_cast<int>(h), static_cast<int>(d),
      static_cast<int>(nch));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
