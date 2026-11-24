// GLM-5.3-Flash's indexer: a token's key (LayerNorm of wk, with bias) and its gate, the pool of four, the
// scores and the selection. A row's list is its own query against the pools complete before it, ties to the
// lower pool, so a row among others is that row alone and a window is its serial steps.
// Semantics from TensorFold 0.6.6 (cb2ebf0) families/glm5_next/cuda/sparse.py (_index_write, _select_rows,
// select_tokens), Triton, Apache-2.0; rewritten for zspark in CUDA C++. The pool, the pooled keys' and the
// queries' storage and the score are the card's vLLM's instead (spec 116 WS-05, the owner): vLLM's GLM-5.3
// `nvidia/pooled_indexer.py` and `ops/glm_kpool.py` (`_fwht_quant_kernel`, `_write_pool`, `_prefill_pool_kernel`)
// and b12x's paged indexer (`dsa_indexer/reference.py`), which SGLang's `kpool_fp8_index.py` matches: the pool an
// fp32 softmax sum rounded once to bf16; a pooled key and each head's query rotated by the D-point Walsh-Hadamard
// transform (stages of stride 1, 2, .., D / 2, the lower value a + b and the upper a - b, then times
// 1 / sqrt(D)), rounded to bf16, and stored e4m3 under one power-of-two scale, 2^ceil(log2(max(max|x|, 1e-4) /
// 448)), the key's kept beside its codes (D + 4 bytes a pool: 132 at GLM's 128), the query's folded into its
// head weight with 1 / sqrt(D H); the score sum_h relu(q8_h . k8) w_h, times the key's scale, the dot on FP8
// tensor cores (`mma` m16n8k32 e4m3, fp32 sums). Where vLLM's Triton is approximate (`tl.exp`, `tl.log2`, the
// fp32 division) this takes the exact operation, so a value is at vLLM's precision or above. TensorFold keeps
// the pooled keys and the queries bf16; the host model (`layers/index.zig`) is this arithmetic, the dot's sum
// one product at a time, so a score is held near it, not to the bit. Compiled `--fmad=false`, so every other
// sum is the additions written.
// Keys and gates are a pool of extents of 512 positions behind table int32 [extents] (spec 018, from the first
// version: logical extent i lives at table[i]). A pool is four positions and 512 is a multiple of four, so a
// pool never straddles two extents: one load of the table a token. The pooled keys stay contiguous, in pool
// order. The list is token positions, not the pool's rows.
// Hand-written: the norm and the pool are four-wide softmaxes, the rotation a butterfly in a warp's lanes and
// the selection a radix over the scores; the score is one GEMM, but its rows are one token's heads, reduced over
// them with a ReLU between, and each token stops at its own pool count, which no CUTLASS epilogue does.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../common/qmm_frag.cuh"
#include "../zs/host.h"

namespace {

constexpr int EXTENT = 512;
constexpr int POOL = 4;
constexpr int SEL = 256;
constexpr int WIDE = 256;  // an index key's values at the write, at most
constexpr int CODED = 128; // and a rotated key's or query's: one scale, a warp of four values a lane
constexpr int SCORE_POOLS = 64;    // a score block's pools: four warps of two n8 tiles
constexpr int SCORE_HEADS = 32;    // and its heads at most: two m16 tiles
constexpr int SCORE_ROWS = 8;      // and its rows, whose queries it stages
constexpr int SCORE_THREADS = 128;
constexpr float FP8_MAX = 448.f;

template <class T>
__device__ __forceinline__ T* at(T* base, const int* table, int pos, int dim, int d) {
  const int physical = table[pos >> 9];
  const int row = physical * EXTENT + (pos & (EXTENT - 1));
  return base + (static_cast<size_t>(row) * dim + d);
}

__device__ __forceinline__ float bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ uint32_t order_key(float v) {
  const uint32_t bits = __float_as_uint(v + 0.f);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// One block a row, one thread a value. The mean and the variance are one sum each, in dimension order, by
// the block's first thread over the values the others brought (TensorFold's _index_write is a warp a row and
// tl.sum; one thread a row for everything was 9.5 to 12.8 us alone, spec 093). Shared memory: the row as fp32
// [256], 1,024 B, and the mean and the inverse deviation; the sums' order needs the values in one thread.
__global__ void __launch_bounds__(WIDE)
    write_kernel(const __nv_bfloat16* __restrict__ K, const float* __restrict__ GATE,
                 const __nv_bfloat16* __restrict__ LN_W, const __nv_bfloat16* __restrict__ LN_B,
                 __nv_bfloat16* __restrict__ KEYS, __nv_bfloat16* __restrict__ GATES, const int* __restrict__ TABLE,
                 const int* __restrict__ POS0, int dim) {
  __shared__ float row[WIDE];
  __shared__ float mean_shared, inv_shared;
  const int r = blockIdx.x, d = threadIdx.x;
  const float x = d < dim ? __bfloat162float(K[(size_t)r * dim + d]) : 0.f;
  if (d < dim) row[d] = x;
  __syncthreads();
  if (d == 0) {
    const float width = static_cast<float>(dim);
    float sum = 0.f;
    for (int i = 0; i < dim; ++i) sum += row[i];
    const float mean = sum / width;
    float squares = 0.f;
    for (int i = 0; i < dim; ++i) {
      const float centred = row[i] - mean;
      squares += centred * centred;
    }
    mean_shared = mean;
    inv_shared = 1.f / sqrtf(squares / width + 1e-6f);
  }
  __syncthreads();
  if (d >= dim) return;
  const int pos = POS0[0] + r;
  const float y = (x - mean_shared) * inv_shared * __bfloat162float(LN_W[d]) + __bfloat162float(LN_B[d]);
  *at(KEYS, TABLE, pos, dim, d) = __float2bfloat16(y);
  *at(GATES, TABLE, pos, dim, d) = __float2bfloat16(GATE[(size_t)r * dim + d]);
}

// The D-point Walsh-Hadamard transform of the warp's vector, value 4 l + j in lane l's v[j] (lanes from D / 4
// hold zeros, and stay among themselves), stage by stage of stride 1 to D / 2 as vLLM's `_fwht128`: the lower of
// a pair a + b, the upper a - b. Then times `rot` (1 / sqrt(D)) and rounded to bf16, as vLLM rounds both a key
// and a query before their scale.
__device__ __forceinline__ void rotate(float (&v)[4], int lane, int dim, float rot) {
  for (int s = 1; s < 4 && s < dim; s *= 2)
    for (int j = 0; j < 4; ++j)
      if ((j & s) == 0) {
        const float a = v[j], b = v[j + s];
        v[j] = a + b;
        v[j + s] = a - b;
      }
  for (int s = 4; s < dim; s *= 2) {
    const int x = s / 4;
    for (int j = 0; j < 4; ++j) {
      const float other = __shfl_xor_sync(0xffffffffu, v[j], x);
      v[j] = (lane & x) ? other - v[j] : v[j] + other;
    }
  }
  for (int j = 0; j < 4; ++j) v[j] = bf16(v[j] * rot);
}

// The warp's rotated vector as e4m3 codes under its scale, 2^ceil(log2(max(max|x|, 1e-4) / 448)) taken exactly
// (vLLM's `tl.log2` is the approximate one; at a power of two it may round the other way), the codes x / scale
// clamped to 448 and rounded to nearest even. Returns the scale; lane l's four codes are `codes`, value 4 l first.
__device__ __forceinline__ float quantize(const float (&v)[4], uint32_t& codes) {
  float most = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  for (int x = 16; x >= 1; x /= 2) most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, x));
  const float ratio = fmaxf(most, 1e-4f) / FP8_MAX;
  int e;
  const float m = frexpf(ratio, &e);  // ratio = m 2^e, m in [0.5, 1): 2^e, or 2^(e - 1) when m is a half
  if (m == 0.5f) --e;
  const float scale = ldexpf(1.f, e), inv = ldexpf(1.f, -e);  // x / scale is x times 2^-e, the same rounding
  codes = 0;
  for (int j = 0; j < 4; ++j) {
    const float q = fmaxf(-FP8_MAX, fminf(FP8_MAX, v[j] * inv));
    codes |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3)) << (8 * j);
  }
  return scale;
}

// A warp a pool, four values a lane: pool pos / 4 + 4 i + w for block i and warp w. A window of R rows
// completes at most R / 4 + 1 pools, and the grid is those (a block a pool of the context started 16,384 at a
// 65,536 bucket for the one that worked, 10.2 us alone against 2.1, spec 093). vLLM's pool: the four gates plus
// `ape` under a softmax, max first, then the keys times their weights added in token order with the weights'
// sum in fp32, divided by that sum (at least 1e-20) and rounded once to bf16; then rotated and stored as codes
// u8 [pools, D] and scales fp32 [pools].
__global__ void __launch_bounds__(128)
    pool_kernel(const __nv_bfloat16* __restrict__ KEYS, const __nv_bfloat16* __restrict__ GATES,
                const int* __restrict__ TABLE, const __nv_bfloat16* __restrict__ APE, uint8_t* __restrict__ CODES,
                float* __restrict__ SCALES, const int* __restrict__ POS0, int rows, int dim, float rot) {
  const int pos = POS0[0];
  const int lane = threadIdx.x & 31;
  const int p = pos / POOL + static_cast<int>(blockIdx.x) * 4 + (threadIdx.x >> 5);
  if (p >= (pos + rows) / POOL) return;
  const bool live = 4 * lane < dim;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (live) {
    float logit[POOL][4];
    float highest[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (int t = 0; t < POOL; ++t)
      for (int j = 0; j < 4; ++j) {
        const int d = 4 * lane + j;
        logit[t][j] = __bfloat162float(*at(GATES, TABLE, p * POOL + t, dim, d)) +
                      __bfloat162float(APE[(size_t)t * dim + d]);
        highest[j] = fmaxf(highest[j], logit[t][j]);
      }
    for (int j = 0; j < 4; ++j) {
      const int d = 4 * lane + j;
      float numerator = 0.f, denominator = 0.f;
      for (int t = 0; t < POOL; ++t) {
        const float weight = expf(logit[t][j] - highest[j]);
        numerator = numerator + __bfloat162float(*at(KEYS, TABLE, p * POOL + t, dim, d)) * weight;
        denominator = denominator + weight;
      }
      v[j] = bf16(numerator / fmaxf(denominator, 1e-20f));
    }
  }
  rotate(v, lane, dim, rot);
  uint32_t codes;
  const float scale = quantize(v, codes);
  if (live) *reinterpret_cast<uint32_t*>(CODES + (size_t)p * dim + 4 * lane) = codes;
  if (lane == 0) SCALES[p] = scale;
}

// A warp a head of a row (vLLM's `fwht128_quant_fp8`): the query rotated and stored as codes u8 [R, H, D], its
// scale folded into the head's weight, wq = (w scale) / sqrt(D H), two roundings in that order.
__global__ void __launch_bounds__(128)
    query_kernel(const __nv_bfloat16* __restrict__ QI, const float* __restrict__ W, uint8_t* __restrict__ Q8,
                 float* __restrict__ WQ, int vectors, int dim, float rot, float norm) {
  const int lane = threadIdx.x & 31;
  const int v_at = static_cast<int>(blockIdx.x) * 4 + (threadIdx.x >> 5);
  if (v_at >= vectors) return;
  const bool live = 4 * lane < dim;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (live) {
    const uint2 x = *reinterpret_cast<const uint2*>(QI + (size_t)v_at * dim + 4 * lane);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&x.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&x.y));
    v[0] = lo.x;
    v[1] = lo.y;
    v[2] = hi.x;
    v[3] = hi.y;
  }
  rotate(v, lane, dim, rot);
  uint32_t codes;
  const float scale = quantize(v, codes);
  if (live) *reinterpret_cast<uint32_t*>(Q8 + (size_t)v_at * dim + 4 * lane) = codes;
  if (lane == 0) WQ[v_at] = (W[v_at] * scale) * norm;
}

__device__ __forceinline__ void mma8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Block (64 pools, 8 rows), four warps of 16 pools: for each row, dots [H, 64] = q8_r [H, D] @ k8^T on the FP8
// tensor cores, the heads the rows of two m16n8k32 mmas, D / 32 steps in order from zero (a D of 16 fills its
// step's upper half with zeros). An mma's element is a function of its own row of A and column of B, so a pool's
// dot depends neither on the pools beside it nor on the rows beside it: a block's rows share its keys and
// nothing else. Then a pool's score: relu(dot_h) wq_h over a lane's four heads (h, h + 8, h + 16, h + 24 for
// g = h), added in that pairing, then over the eight lanes of a column by `shfl_xor` 4, 8 and 16, times the
// pool's scale. Pools a row has not completed are -inf, up to the block of 64 the context's last pool is in.
// Registers: a warp's keys as B fragments, (D / 32) x 2 x 2 words, loaded once and read by every row.
// Shared memory: the rows' queries as codes [8][32][128], 32,768 B, each head's 16-byte chunks XORed by the head
// (so `ldmatrix`'s eight rows meet eight banks), staged by cp.async with heads past H and values past D zeros:
// one block of 128 threads takes 33 KB, three an SM. The four warps read every query of the block's rows, so a
// query is fetched once a block, not once a warp.
__global__ void __launch_bounds__(SCORE_THREADS)
    score_kernel(const uint8_t* __restrict__ Q8, const float* __restrict__ WQ, const uint8_t* __restrict__ K8,
                 const float* __restrict__ KS, const int* __restrict__ POS0, float* __restrict__ SC, int rows,
                 int heads, int dim, int nb) {
  __shared__ __align__(16) uint8_t stage[SCORE_ROWS][SCORE_HEADS][CODED];
  const int r0 = blockIdx.y * SCORE_ROWS, p0 = blockIdx.x * SCORE_POOLS, tid = threadIdx.x;
  const int pos = POS0[0];
  const int here = min(SCORE_ROWS, rows - r0);
  const int widest = (pos + r0 + here) / POOL;  // the block's last row's complete pools
  if (p0 >= widest) {
    for (int i = 0; i < here; ++i)
      if (tid < SCORE_POOLS && p0 + tid < nb) SC[(size_t)(r0 + i) * nb + p0 + tid] = -INFINITY;
    return;
  }
  const int chunks = dim / 16;
  for (int c = tid; c < SCORE_ROWS * SCORE_HEADS * (CODED / 16); c += SCORE_THREADS) {
    const int i = c / (SCORE_HEADS * (CODED / 16)), h = (c / (CODED / 16)) % SCORE_HEADS, k = c % (CODED / 16);
    const bool read = i < here && h < heads && k < chunks;
    const uint8_t* src = Q8 + (read ? ((size_t)(r0 + i) * heads + h) * dim + 16 * k : 0);
    qmm_frag::cp16z(&stage[i][h][(k ^ (h & 7)) * 16], read ? src : Q8, read);
  }
  qmm_frag::commit();

  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int steps = (dim + 31) / 32;
  uint32_t b[CODED / 32][2][2];
  float ks[2][2];
  for (int n = 0; n < 2; ++n) {
    const int p = p0 + warp * 16 + n * 8 + g;
    const bool live = p < widest;
    for (int s = 0; s < CODED / 32; ++s)
      for (int half = 0; half < 2; ++half) {
        const int k = 32 * s + 16 * half + 4 * t;
        b[s][n][half] = live && s < steps && k < dim ? *reinterpret_cast<const uint32_t*>(K8 + (size_t)p * dim + k) : 0u;
      }
    for (int e = 0; e < 2; ++e) {
      const int q = p0 + warp * 16 + n * 8 + 2 * t + e;
      ks[n][e] = q < widest ? KS[q] : 0.f;
    }
  }
  qmm_frag::wait<0>();
  __syncthreads();

  for (int i = 0; i < here; ++i) {
    const int r = r0 + i;
    const int complete = (pos + r + 1) / POOL;
    float* out = SC + (size_t)r * nb;
    if (p0 >= complete) {
      if (tid < SCORE_POOLS && p0 + tid < nb) out[p0 + tid] = -INFINITY;
      continue;
    }
    float c[2][2][4];
    for (int m = 0; m < 2; ++m)
      for (int n = 0; n < 2; ++n)
        for (int e = 0; e < 4; ++e) c[m][n][e] = 0.f;
    for (int s = 0; s < CODED / 32; ++s) {
      if (s >= steps) break;
      for (int m = 0; m < 2; ++m) {
        const int h = m * 16 + (lane & 15), k = 2 * s + (lane >> 4);
        uint32_t a[4];
        qmm_frag::ldmatrix4(a, &stage[i][h][(k ^ (h & 7)) * 16]);
        for (int n = 0; n < 2; ++n) mma8(c[m][n], a, b[s][n][0], b[s][n][1]);
      }
    }
    const float* wq = WQ + (size_t)r * heads;
    float w[4];
    for (int j = 0; j < 4; ++j) w[j] = g + 8 * j < heads ? wq[g + 8 * j] : 0.f;
    for (int n = 0; n < 2; ++n)
      for (int e = 0; e < 2; ++e) {
        float sum = (fmaxf(c[0][n][e], 0.f) * w[0] + fmaxf(c[0][n][2 + e], 0.f) * w[1]) +
                    (fmaxf(c[1][n][e], 0.f) * w[2] + fmaxf(c[1][n][2 + e], 0.f) * w[3]);
        for (int x = 4; x <= 16; x *= 2) sum = sum + __shfl_xor_sync(0xffffffffu, sum, x);
        const int p = p0 + warp * 16 + n * 8 + 2 * t + e;
        if (g == 0 && p < nb) out[p] = p < complete ? sum * ks[n][e] : -INFINITY;
      }
  }
}

// Block r. A row with at most TOP complete pools keeps a count of 0. Otherwise its list is the TOP pools by
// score, the lower pool among equal scores, in pool order, then the unfinished pool's visible tokens (at most
// three); the rest of the row is -1.
__global__ void __launch_bounds__(SEL)
    select_kernel(const float* __restrict__ SC, const int* __restrict__ POS0, int* __restrict__ TOKENS,
                  int* __restrict__ COUNTS, int nb, int top, int width) {
  __shared__ int hist[256];
  __shared__ int above[SEL];
  __shared__ int equal[SEL];
  __shared__ uint32_t cut_shared;
  __shared__ int need_shared;
  const int r = blockIdx.x, tid = threadIdx.x;
  const int end = POS0[0] + r + 1, complete = end / POOL;
  int* list = TOKENS + (size_t)r * width;
  for (int i = tid; i < width; i += SEL) list[i] = -1;
  __syncthreads();
  if (complete <= top) {
    if (tid == 0) COUNTS[r] = 0;
    return;
  }
  const float* row = SC + (size_t)r * nb;
  if (tid == 0) {
    cut_shared = 0;
    need_shared = top;
  }
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    const uint32_t lead = cut_shared;
    for (int b = tid; b < complete; b += SEL) {
      const uint32_t key = order_key(row[b]);
      if (pass > 0 && (key >> (shift + 8)) != (lead >> (shift + 8))) continue;
      atomicAdd(&hist[(key >> shift) & 0xFF], 1);
    }
    __syncthreads();
    if (tid == 0) {
      int larger = 0, digit = 255;
      for (; digit > 0; --digit) {
        if (larger + hist[digit] >= need_shared) break;
        larger += hist[digit];
      }
      need_shared -= larger;
      cut_shared = lead | (static_cast<uint32_t>(digit) << shift);
    }
    __syncthreads();
  }
  const uint32_t cut = cut_shared;
  const int need = need_shared;
  const int per = (complete + SEL - 1) / SEL;
  const int b0 = min(tid * per, complete), b1 = min(b0 + per, complete);
  int na = 0, ne = 0;
  for (int b = b0; b < b1; ++b) {
    const uint32_t key = order_key(row[b]);
    na += key > cut;
    ne += key == cut;
  }
  above[tid] = na;
  equal[tid] = ne;
  __syncthreads();
  if (tid == 0) {
    int sa = 0, se = 0;
    for (int u = 0; u < SEL; ++u) {
      const int a = above[u], e = equal[u];
      above[u] = sa;
      equal[u] = se;
      sa += a;
      se += e;
    }
  }
  __syncthreads();
  int seen = equal[tid], placed = above[tid] + min(seen, need);
  for (int b = b0; b < b1; ++b) {
    const uint32_t key = order_key(row[b]);
    bool chosen = key > cut;
    if (key == cut) chosen = ++seen <= need;
    if (chosen) {
      for (int k = 0; k < POOL; ++k) list[placed * POOL + k] = b * POOL + k;
      ++placed;
    }
  }
  if (tid == 0) {
    const int rem = end - POOL * complete;
    for (int t = 0; t < rem && t + 1 < POOL; ++t) list[top * POOL + t] = POOL * complete + t;
    COUNTS[r] = top * POOL + rem;
  }
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

}  // namespace

extern "C" {

// k bf16 [R, D] and gate fp32 [R, D] of rows pos .. pos + R - 1, ln_w and ln_b bf16 [D], into keys and gates
// bf16 [extents * 512, D] through table int32 [extents]. ``context`` is pos + R.
int zs_glm5_index_write(const zs_view* k_, const zs_view* gate_, const zs_view* ln_w_, const zs_view* ln_b_,
                        const zs_view* keys_, const zs_view* gates_, const zs_view* table_, const zs_view* pos_,
                        int64_t rows, int64_t context) {
  zs::begin();
  const zs::View k(k_), gate(gate_), ln_w(ln_w_), ln_b(ln_b_), keys(keys_), gates(gates_), table(table_), pos(pos_);
  ZS_REQUIRE(k.defined() && k.dtype() == ZS_BF16 && k.dim() == 2 && k.is_contiguous() && k.size(0) >= rows,
             "k: bf16 [R, D]");
  const int64_t dim = k.size(1);
  ZS_REQUIRE(dim >= 1 && dim <= WIDE, "an index key holds at most 256 values");
  ZS_REQUIRE(gate.defined() && gate.dtype() == ZS_F32 && gate.dim() == 2 && gate.is_contiguous() &&
                 gate.size(0) >= rows && gate.size(1) == dim,
             "gate: fp32 [R, D]");
  ZS_REQUIRE(ln_w.defined() && ln_w.dtype() == ZS_BF16 && ln_w.is_contiguous() && ln_w.numel() == dim &&
                 ln_b.defined() && ln_b.dtype() == ZS_BF16 && ln_b.is_contiguous() && ln_b.numel() == dim,
             "ln_w, ln_b: bf16 [D]");
  ZS_REQUIRE(keys.defined() && keys.dtype() == ZS_BF16 && keys.dim() == 2 && keys.is_contiguous() &&
                 keys.size(1) == dim && keys.size(0) % EXTENT == 0 && gates.defined() && gates.dtype() == ZS_BF16 &&
                 gates.dim() == 2 && gates.is_contiguous() && gates.size(0) == keys.size(0) && gates.size(1) == dim,
             "keys, gates: bf16 [extents * 512, D]");
  ZS_REQUIRE(table.defined() && table.dtype() == ZS_I32 && table.is_contiguous() && table.numel() >= 1 &&
                 table.numel() * EXTENT <= keys.size(0),
             "table: int32 [extents] over the keys' extents");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(rows >= 1 && context >= rows && context <= table.numel() * EXTENT && context <= INT32_MAX,
             "the table holds pos + R rows");
  write_kernel<<<static_cast<unsigned>(rows), static_cast<unsigned>((dim + 31) / 32 * 32), 0, zs::stream()>>>(
      k.data_ptr<__nv_bfloat16>(), gate.data_ptr<float>(), ln_w.data_ptr<__nv_bfloat16>(),
      ln_b.data_ptr<__nv_bfloat16>(), keys.data_ptr<__nv_bfloat16>(), gates.data_ptr<__nv_bfloat16>(),
      table.data_ptr<int>(), pos.data_ptr<int>(), static_cast<int>(dim));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// The pools rows pos .. pos + R - 1 complete, into codes u8 [pools, D] and scales fp32 [pools], contiguous in pool
// order. ape is bf16 [4, D]; D a power of two from 16 to 128.
int zs_glm5_index_pool(const zs_view* keys_, const zs_view* gates_, const zs_view* table_, const zs_view* ape_,
                       const zs_view* codes_, const zs_view* scales_, const zs_view* pos_, int64_t rows,
                       int64_t context) {
  zs::begin();
  const zs::View keys(keys_), gates(gates_), table(table_), ape(ape_), codes(codes_), scales(scales_), pos(pos_);
  ZS_REQUIRE(keys.defined() && keys.dtype() == ZS_BF16 && keys.dim() == 2 && keys.is_contiguous() &&
                 keys.size(0) % EXTENT == 0,
             "keys: bf16 [extents * 512, D]");
  const int64_t dim = keys.size(1);
  ZS_REQUIRE(dim >= 16 && dim <= CODED && (dim & (dim - 1)) == 0, "a rotated index key: 16 to 128 values, a power of two");
  ZS_REQUIRE(gates.defined() && gates.dtype() == ZS_BF16 && gates.dim() == 2 && gates.is_contiguous() &&
                 gates.size(0) == keys.size(0) && gates.size(1) == dim,
             "gates: bf16 [extents * 512, D]");
  ZS_REQUIRE(table.defined() && table.dtype() == ZS_I32 && table.is_contiguous() &&
                 table.numel() * EXTENT <= keys.size(0),
             "table: int32 [extents] over the keys' extents");
  ZS_REQUIRE(ape.defined() && ape.dtype() == ZS_BF16 && ape.dim() == 2 && ape.is_contiguous() && ape.size(0) == POOL &&
                 ape.size(1) == dim,
             "ape: bf16 [4, D]");
  ZS_REQUIRE(codes.defined() && codes.dtype() == ZS_U8 && codes.dim() == 2 && codes.is_contiguous() &&
                 codes.size(1) == dim && scales.defined() && scales.dtype() == ZS_F32 && scales.is_contiguous() &&
                 scales.numel() == codes.size(0),
             "codes: u8 [pools, D], scales: fp32 [pools]");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(rows >= 1 && context >= rows && context / POOL <= codes.size(0) && context <= table.numel() * EXTENT &&
                 context <= INT32_MAX,
             "the codes hold the pools pos + R completes, and the table their rows");
  const unsigned pools = static_cast<unsigned>(rows / POOL + 1);
  pool_kernel<<<(pools + 3) / 4, 128, 0, zs::stream()>>>(
      keys.data_ptr<__nv_bfloat16>(), gates.data_ptr<__nv_bfloat16>(), table.data_ptr<int>(),
      ape.data_ptr<__nv_bfloat16>(), codes.data_ptr<uint8_t>(), scales.data_ptr<float>(), pos.data_ptr<int>(),
      static_cast<int>(rows), static_cast<int>(dim), static_cast<float>(1.0 / sqrt(static_cast<double>(dim))));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// qi bf16 [R, H, D] and the head weights fp32 [R, H] -> q8 u8 [R, H, D] and wq fp32 [R, H]: each head's query
// rotated and stored as codes, its scale folded into its weight with 1 / sqrt(D H).
int zs_glm5_index_query(const zs_view* qi_, const zs_view* weights_, const zs_view* q8_, const zs_view* wq_) {
  zs::begin();
  const zs::View qi(qi_), weights(weights_), q8(q8_), wq(wq_);
  ZS_REQUIRE(qi.defined() && qi.dtype() == ZS_BF16 && qi.dim() == 3 && qi.is_contiguous() && qi.size(0) >= 1,
             "qi: bf16 [R, H, D]");
  const int64_t rows = qi.size(0), heads = qi.size(1), dim = qi.size(2);
  ZS_REQUIRE(heads >= 1 && heads <= SCORE_HEADS && dim >= 16 && dim <= CODED && (dim & (dim - 1)) == 0,
             "at most 32 heads, and a rotated index key of 16 to 128 values, a power of two");
  ZS_REQUIRE(weights.defined() && weights.dtype() == ZS_F32 && weights.is_contiguous() &&
                 weights.numel() >= rows * heads,
             "weights: fp32 [R, H]");
  ZS_REQUIRE(q8.defined() && q8.dtype() == ZS_U8 && q8.is_contiguous() && q8.numel() >= rows * heads * dim &&
                 wq.defined() && wq.dtype() == ZS_F32 && wq.is_contiguous() && wq.numel() >= rows * heads,
             "q8: u8 [R, H, D], wq: fp32 [R, H]");
  const int64_t vectors = rows * heads;
  ZS_REQUIRE(vectors <= INT32_MAX / 2, "rows times heads in int32");
  query_kernel<<<static_cast<unsigned>((vectors + 3) / 4), 128, 0, zs::stream()>>>(
      qi.data_ptr<__nv_bfloat16>(), weights.data_ptr<float>(), q8.data_ptr<uint8_t>(), wq.data_ptr<float>(),
      static_cast<int>(vectors), static_cast<int>(dim), static_cast<float>(1.0 / sqrt(static_cast<double>(dim))),
      static_cast<float>(1.0 / sqrt(static_cast<double>(dim * heads))));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// q8 u8 [R, H, D], wq fp32 [R, H], the pools' codes u8 [pools, D] and scales fp32 [pools] -> scores fp32 [R, NB].
// A pool past the row is -inf up to the block of 64 the context's last pool is in, and not written after it.
// ``context`` is pos + R or a graph's bucket over it, and NB is at least its complete pools.
int zs_glm5_index_score(const zs_view* q8_, const zs_view* wq_, const zs_view* codes_, const zs_view* scales_,
                        const zs_view* pos_, const zs_view* scores_, int64_t context) {
  zs::begin();
  const zs::View q8(q8_), wq(wq_), codes(codes_), scales(scales_), pos(pos_), scores(scores_);
  ZS_REQUIRE(q8.defined() && q8.dtype() == ZS_U8 && q8.dim() == 3 && q8.is_contiguous() && q8.size(0) >= 1,
             "q8: u8 [R, H, D]");
  const int64_t rows = q8.size(0), heads = q8.size(1), dim = q8.size(2);
  ZS_REQUIRE(heads >= 1 && heads <= SCORE_HEADS && dim >= 16 && dim <= CODED && dim % 16 == 0,
             "at most 32 heads, and an index key of 16 to 128 values in sixteens");
  ZS_REQUIRE(wq.defined() && wq.dtype() == ZS_F32 && wq.is_contiguous() && wq.numel() >= rows * heads,
             "wq: fp32 [R, H]");
  ZS_REQUIRE(codes.defined() && codes.dtype() == ZS_U8 && codes.dim() == 2 && codes.is_contiguous() &&
                 codes.size(1) == dim && scales.defined() && scales.dtype() == ZS_F32 && scales.is_contiguous() &&
                 scales.numel() == codes.size(0),
             "codes: u8 [pools, D], scales: fp32 [pools]");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(scores.defined() && scores.dtype() == ZS_F32 && scores.dim() == 2 && scores.is_contiguous() &&
                 scores.size(0) >= rows && context >= rows && context / POOL <= scores.size(1) &&
                 context / POOL <= codes.size(0) && context <= INT32_MAX,
             "scores: fp32 [R, NB], NB and the codes at least the pools pos + R completes");
  ZS_REQUIRE(rows <= 65535 * SCORE_ROWS, "at most 524,280 rows a launch");
  // The grid is the context's pools, not the buffer's: nothing reads a score past a row's complete pools.
  const int64_t nb = scores.size(1);
  const dim3 grid(static_cast<unsigned>(context / POOL / SCORE_POOLS + 1),
                  static_cast<unsigned>((rows + SCORE_ROWS - 1) / SCORE_ROWS));
  score_kernel<<<grid, SCORE_THREADS, 0, zs::stream()>>>(
      q8.data_ptr<uint8_t>(), wq.data_ptr<float>(), codes.data_ptr<uint8_t>(), scales.data_ptr<float>(),
      pos.data_ptr<int>(), scores.data_ptr<float>(), static_cast<int>(rows), static_cast<int>(heads),
      static_cast<int>(dim), static_cast<int>(nb));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// scores fp32 [R, NB] -> tokens int32 [R, width] and counts int32 [R]. width is the kept pools' tokens and
// three of the unfinished pool. A row still inside the dense window (at most ``kept`` complete pools) has a
// count of 0 and a row of -1.
int zs_glm5_index_select(const zs_view* scores_, const zs_view* pos_, const zs_view* tokens_, const zs_view* counts_,
                         int64_t kept, int64_t context) {
  zs::begin();
  const zs::View scores(scores_), pos(pos_), tokens(tokens_), counts(counts_);
  ZS_REQUIRE(scores.defined() && scores.dtype() == ZS_F32 && scores.dim() == 2 && scores.is_contiguous(),
             "scores: fp32 [R, NB]");
  const int64_t rows = scores.size(0), nb = scores.size(1);
  ZS_REQUIRE(rows >= 1 && rows <= 65535 && kept >= 1 && context >= rows && context / POOL <= nb,
             "one block a row, and a score for every pool pos + R completes");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  const int64_t width = kept * POOL + POOL - 1;
  ZS_REQUIRE(tokens.defined() && tokens.dtype() == ZS_I32 && tokens.dim() == 2 && tokens.is_contiguous() &&
                 tokens.size(0) >= rows && tokens.size(1) >= width,
             "tokens: int32 [R, width], width the kept pools and the tail");
  ZS_REQUIRE(counts.defined() && counts.dtype() == ZS_I32 && counts.is_contiguous() && counts.numel() >= rows,
             "counts: int32 [R]");
  select_kernel<<<static_cast<unsigned>(rows), SEL, 0, zs::stream()>>>(
      scores.data_ptr<float>(), pos.data_ptr<int>(), tokens.data_ptr<int>(), counts.data_ptr<int>(),
      static_cast<int>(nb), static_cast<int>(kept), static_cast<int>(tokens.size(1)));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
