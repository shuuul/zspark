// Flash Next's attention block, decode path: the prep (per-head RMSNorm, rotate-half RoPE, the cache rows) and the
// output gate. The attention over each row's keys between them is common/attention.cu's (spec 138 WS-04 moved it
// there), and the selection that builds a sparse row's key list is in qsa.cu.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/glue.py (_attn_prep, _prep_row, _attn_gate),
// Triton, Apache-2.0; rewritten for zspark in CUDA C++: without the 32-group sums; the cache as bf16 or as FP8
// (common/attention.cu), not TensorFold's int8 and int4. Where Triton left a reduction's order to its compiler (the
// squares of a head) the order here is fixed and stated at the kernel.
// Hand-written because none of it is a GEMM: a head's reduction and element-wise maps.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "../common/attention.cuh"
#include "attention_rope.cuh"
#include "../zs/host.h"

namespace {

// Block (row, head) of WIDE threads over the stacked projection P [R, PW]:
//   [q_0 | gate_0 | ... | q_NQ-1 | gate_NQ-1] [k: NKV x HD] [v: NKV x HD] [indexer q: NI x IHD] [indexer k: IHD].
// Heads 0 .. NQ + NKV + NI - 1 are a query, a key or an indexer query: x / rms(x) * scale rounded to bf16, the
// first 2 HALF values rotated by axis * INV (rotate-half) and rounded again, the axis the row's position or, under
// ROPE (an image prompt's, attention_rope.cuh; a slot's ROPES[SID[r]] in a round), one of its three. rms sums the squares in a tree
// over the WIDE threads, zeros past the head's width. A query goes to Q, a key to the cache row POS0 + r with its
// value beside it as projected, an indexer query to IQ. The last head copies the raw indexer key to its cache row.
// With KS (the FP8 cache) Q, KC and VC hold e4m3 codes: of the rounded query or key, and of the value, each over
// its head's s = (largest magnitude) / 448, which goes to QS [R, NQ], KS and VS [capacity, NKV]. A head of zeros
// has s = 0 and codes of zero.
// With VW (values as NVFP4, VW bytes a head in VC) a value's 16s each take an e4m3 scale code, of (the 16's
// largest magnitude) / (the head's) * 448, and a value the e2m1 code of x / u, u = (the scale code's value) s_v / 6:
// HD / 2 bytes of codes, the even value in the low half, then HD / 16 scale codes. A 16 whose code is zero holds
// zeros.
__global__ void __launch_bounds__(WIDE)
    ZS_NAME(prep)(
#ifdef ZS_ROWS
                const int* __restrict__ SID, const int* const* __restrict__ TABLES,
                const int64_t* __restrict__ BASES, const int* const* __restrict__ ROPES,
#else
                ZS_TABLE_PARAM
#endif
                const __nv_bfloat16* __restrict__ P, const int* __restrict__ POS0, const float* __restrict__ QW,
                const float* __restrict__ KW, const float* __restrict__ IW, const float* __restrict__ INV,
                void* __restrict__ Q, void* __restrict__ KC, void* __restrict__ VC, __nv_bfloat16* __restrict__ IQ,
                __nv_bfloat16* __restrict__ IKC, float* __restrict__ QS, float* __restrict__ KS,
                float* __restrict__ VS, float eps, int PW, int NQ, int NKV, int HD, int NI, int IHD, int HALF,
                int VW, const int* __restrict__ ROPE, int S1, int S2) {
  __shared__ float squares[WIDE];
  __shared__ float beside[WIDE];
  const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
#ifdef ZS_ROWS
  // A slot's k, v and indexer keys. Identity tables on separate pools do not alias; one pool repeats the pointer.
  KC = reinterpret_cast<void*>(BASES[(size_t)SID[r] * 3]);
  VC = reinterpret_cast<void*>(BASES[(size_t)SID[r] * 3 + 1]);
  IKC = reinterpret_cast<__nv_bfloat16*>(BASES[(size_t)SID[r] * 3 + 2]);
  const int* TABLE = TABLES[SID[r]];
  const int pos = POS0[r];
  // a slot's rotary words (an image prompt's), or null: its rows' own positions (spec 019 WS-05)
  ROPE = ROPES == nullptr ? nullptr : ROPES[SID[r]];
#else
  const int pos = POS0[0] + r;
#endif
#ifdef ZS_EXTENTS
  const int kept = stored_at(TABLE, pos);    // the caches' row of this position
#else
  const int kept = pos;
#endif
  const __nv_bfloat16* row = P + (size_t)r * PW;
  const int index0 = NQ * 2 * HD + 2 * NKV * HD;
  if (head == NQ + NKV + NI) {
    if (d < IHD) IKC[(size_t)kept * IHD + d] = row[index0 + NI * IHD + d];
    return;
  }
  int width = HD, src;
  const float* W;
  if (head < NQ) {
    src = head * 2 * HD;
    W = QW;
  } else if (head < NQ + NKV) {
    src = NQ * 2 * HD + (head - NQ) * HD;
    W = KW;
  } else {
    width = IHD;
    src = index0 + (head - NQ - NKV) * IHD;
    W = IW;
  }
  const bool live = d < width;
  const float x = live ? __bfloat162float(row[src + d]) : 0.f;
  squares[d] = x * x;
  __syncthreads();
  for (int half = WIDE / 2; half > 0; half >>= 1) {
    if (d < half) squares[d] += squares[d + half];
    __syncthreads();
  }
  const bool codes = KS != nullptr && head < NQ + NKV;
  if (!live && !codes) return;
  __nv_bfloat16 stored = __float2bfloat16_rn(0.f);
  if (live) {
    const float rinv = 1.0f / sqrtf(squares[0] / width + eps);
    const float xn = bf(x * rinv * W[d]);
    float out = xn;
    if (d < 2 * HALF) {
      const int partner = d < HALF ? d + HALF : d - HALF, i = d < HALF ? d : d - HALF;
      const float xpn = bf(__bfloat162float(row[src + partner]) * rinv * W[partner]);
      const float ang = static_cast<float>(rope_axis(ROPE, pos, i, S1, S2)) * INV[i];
      const float c = cosf(ang), s = sinf(ang);
      out = d < HALF ? xn * c - xpn * s : xpn * s + xn * c;
    }
    stored = __float2bfloat16_rn(out);
  }
  if (!codes) {
    if (head < NQ) {
      static_cast<__nv_bfloat16*>(Q)[((size_t)r * NQ + head) * HD + d] = stored;
    } else if (head < NQ + NKV) {
      const size_t slot = ((size_t)kept * NKV + head - NQ) * HD + d;
      static_cast<__nv_bfloat16*>(KC)[slot] = stored;
      static_cast<__nv_bfloat16*>(VC)[slot] = row[NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d];
    } else {
      IQ[((size_t)r * NI + head - NQ - NKV) * IHD + d] = stored;
    }
    return;
  }
  // every thread is here: the head's largest magnitudes, of the query or key and of the value beside a key
  const bool key = head >= NQ;
  const float a = __bfloat162float(stored);
  const float v = key && live ? __bfloat162float(row[NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d]) : 0.f;
  __syncthreads();
  squares[d] = fabsf(a);
  beside[d] = fabsf(v);
  __syncthreads();
  for (int half = WIDE / 2; half > 0; half >>= 1) {
    if (d < half) {
      squares[d] = fmaxf(squares[d], squares[d + half]);
      beside[d] = fmaxf(beside[d], beside[d + half]);
    }
    __syncthreads();
  }
  const float top = squares[0], top_v = beside[0];
  uint32_t fours = 0, scale_code = 0;
  if (VW > 0 && key) {
    // a warp holds two 16s of the head; threads past the head hold zeros
    float most = fabsf(v);
    for (int other = 8; other > 0; other >>= 1) most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, other));
    scale_code = top_v > 0.f ? __nv_cvt_float_to_fp8(most / top_v * 448.f, __NV_SATFINITE, __NV_E4M3) : 0;
    const float u = e4m3f(scale_code) * top_v / 448.f / 6.f;
    fours = u > 0.f ? nvfp4_codes::fp4_e2m1(v / u) : 0;
    fours |= __shfl_xor_sync(0xffffffffu, fours, 1) << 4;   // an even thread: its code, then its neighbour's
  }
  if (!live) return;
  const uint8_t code = top > 0.f ? __nv_cvt_float_to_fp8(a / top * 448.f, __NV_SATFINITE, __NV_E4M3) : 0;
  if (!key) {
    static_cast<uint8_t*>(Q)[((size_t)r * NQ + head) * HD + d] = code;
    if (d == 0) QS[(size_t)r * NQ + head] = top / 448.f;
    return;
  }
  const size_t at = (size_t)kept * NKV + head - NQ;
  static_cast<uint8_t*>(KC)[at * HD + d] = code;
  if (VW > 0) {
    uint8_t* const head_v = static_cast<uint8_t*>(VC) + at * VW;
    if (d % 2 == 0) head_v[d / 2] = static_cast<uint8_t>(fours);
    if (d % 16 == 0) head_v[HD / 2 + d / 16] = static_cast<uint8_t>(scale_code);
  } else {
    static_cast<uint8_t*>(VC)[at * HD + d] =
        top_v > 0.f ? __nv_cvt_float_to_fp8(v / top_v * 448.f, __NV_SATFINITE, __NV_E4M3) : 0;
  }
  if (d == 0) {
    KS[at] = top / 448.f;
    VS[at] = top_v / 448.f;
  }
}

#ifndef ZS_EXTENTS
// Thread (row, head, d): bf16(o * sigmoid(gate)), the gate from the projection's [q | gate] pair of the head.
__global__ void ZS_NAME(gate)(const __nv_bfloat16* __restrict__ O, const __nv_bfloat16* __restrict__ P,
                            __nv_bfloat16* __restrict__ OUT, int R, int PW, int NQ, int HD) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * NQ * HD) return;
  const int d = static_cast<int>(i % HD), h = static_cast<int>(i / HD % NQ), r = static_cast<int>(i / HD / NQ);
  const float o = __bfloat162float(O[i]);
  const float g = __bfloat162float(P[(size_t)r * PW + h * 2 * HD + HD + d]);
  OUT[i] = __float2bfloat16_rn(o / (1.0f + expf(-g)));
}
#endif

}  // namespace

extern "C" {

// p bf16 [R, PW] (the stacked projection, see attn_prep_kernel), pos int32 [1] (the first row's position), scales fp32
// [HD], [HD], [IHD], inv_freq fp32 [HALF] -> q bf16 [R, NQ, HD], iq bf16 [R, NI, IHD], and rows pos .. pos + R - 1
// of kc, vc bf16 [capacity, NKV, HD] and ikc bf16 [capacity, IHD]. ``context`` is pos + R as the host knows it.
// The FP8 cache: q, kc and vc are e4m3 with qs fp32 [R, NQ], ks and vs fp32 [capacity, NKV]; null for bf16.
// vc may be uint8 [capacity, NKV, bytes of a head as NVFP4] beside e4m3 keys: the values go there as NVFP4.
// With ZS_EXTENTS (zs_qwen4_attn_prep_table): kc, vc, ikc,
// ks and vs are pools, capacity a multiple of 512, and table int32 [extents] says where each 512 positions of the
// sequence lie in them; its entries are the caller's to keep below capacity / 512 and apart, the launcher reads
// no device memory. Only the entries of positions below pos + R are read.
#if defined(ZS_ROWS)
int zs_qwen4_attn_prep_rows(const zs_view* sid_, const zs_view* tables_, const zs_view* bases_, const zs_view* p_,
                            const zs_view* pos_,
#elif defined(ZS_EXTENTS)
int zs_qwen4_attn_prep_table(const zs_view* table_, const zs_view* p_, const zs_view* pos_,
#else
int zs_qwen4_attn_prep(const zs_view* p_, const zs_view* pos_,
#endif
                    const zs_view* q_scale_, const zs_view* k_scale_,
                    const zs_view* iq_scale_, const zs_view* inv_freq_, const zs_view* q_, const zs_view* kc_,
                    const zs_view* vc_, const zs_view* iq_, const zs_view* ikc_, const zs_view* qs_,
                    const zs_view* ks_, const zs_view* vs_, double eps, int64_t context
#if defined(ZS_ROWS)
                    , const zs_view* ropes_, int64_t s1, int64_t s2
#elif !defined(ZS_EXTENTS)
                    , const zs_view* rope_, int64_t s1, int64_t s2
#endif
                    ) {
  zs::begin();
  const zs::View p(p_), pos(pos_), q_scale(q_scale_), k_scale(k_scale_), iq_scale(iq_scale_), inv_freq(inv_freq_),
      q(q_), kc(kc_), vc(vc_), iq(iq_), ikc(ikc_), qs(qs_), ks(ks_), vs(vs_);
  ZS_REQUIRE(is(p, ZS_BF16, 2), "p: bf16 [R, PW]");
  const bool f8 = kc.defined() && kc.dtype() == ZS_E4M3;
  const int stored = f8 ? ZS_E4M3 : ZS_BF16;
  ZS_REQUIRE(is(q, stored, 3) && q.size(0) == p.size(0), "q: [R, heads, head_dim] in the cache's dtype");
  ZS_REQUIRE(is(kc, stored, 3) && kc.size(2) == q.size(2), "kc: bf16 or e4m3 [capacity, kv_heads, head_dim]");
  const bool v4 = f8 && vc.defined() && vc.dtype() == ZS_U8;
  ZS_REQUIRE(is(vc, v4 ? ZS_U8 : stored, 3) && vc.size(0) == kc.size(0) && vc.size(1) == kc.size(1) &&
             vc.size(2) == (v4 ? nvfp4_width(static_cast<int>(kc.size(2))) : kc.size(2)),
             "vc: kc's shape and dtype, or beside e4m3 keys uint8 [capacity, kv_heads, NVFP4 bytes of a head]");
  ZS_REQUIRE(!v4 || kc.size(2) % 32 == 0, "NVFP4 values: head_dim a multiple of 32");
  ZS_REQUIRE(is(iq, ZS_BF16, 3) && iq.size(0) == p.size(0), "iq: bf16 [R, index_heads, index_dim]");
  ZS_REQUIRE(is(ikc, ZS_BF16, 2) && ikc.size(1) == iq.size(2), "ikc: bf16 [capacity, index_dim]");
  const int64_t rows = p.size(0), pw = p.size(1), nq = q.size(1), hd = q.size(2), nkv = kc.size(1), ni = iq.size(1),
                ihd = iq.size(2), half = inv_freq.numel();
  if (f8) {
    ZS_REQUIRE(qs.defined() && qs.dtype() == ZS_F32 && qs.is_contiguous() && qs.numel() >= rows * nq,
               "qs: fp32 [R, heads]");
    ZS_REQUIRE(is(ks, ZS_F32, 2) && ks.size(0) == kc.size(0) && ks.size(1) == nkv && is(vs, ZS_F32, 2) &&
               vs.size(0) == kc.size(0) && vs.size(1) == nkv, "ks, vs: fp32 [capacity, kv_heads]");
  } else {
    ZS_REQUIRE(!qs.defined() && !ks.defined() && !vs.defined(), "qs, ks and vs come with an e4m3 cache");
  }
  ZS_REQUIRE(hd >= 1 && hd <= WIDE && ihd >= 1 && ihd <= WIDE, "a head holds at most 256 values");
  ZS_REQUIRE(pw == nq * 2 * hd + 2 * nkv * hd + (ni + 1) * ihd, "p holds q|gate pairs, k, v, indexer q, indexer k");
#ifdef ZS_ROWS
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == rows, "posr: int32 [R]");
#else
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
#endif
  ZS_REQUIRE(q_scale.defined() && q_scale.dtype() == ZS_F32 && q_scale.numel() == hd, "q_scale: fp32 [head_dim]");
  ZS_REQUIRE(k_scale.defined() && k_scale.dtype() == ZS_F32 && k_scale.numel() == hd, "k_scale: fp32 [head_dim]");
  ZS_REQUIRE(iq_scale.defined() && iq_scale.dtype() == ZS_F32 && iq_scale.numel() == ihd,
             "iq_scale: fp32 [index_dim]");
  ZS_REQUIRE(inv_freq.defined() && inv_freq.dtype() == ZS_F32 && half >= 1 && 2 * half <= hd && 2 * half <= ihd,
             "inv_freq: fp32 [half], 2 half within a head and an indexer head");
#if defined(ZS_ROWS)
  const zs::View sid(sid_), tables(tables_), bases(bases_);
  ZS_REQUIRE(sid.defined() && sid.dtype() == ZS_I32 && sid.is_contiguous() && sid.numel() == rows, "sid: int32 [R]");
  ZS_REQUIRE(tables.defined() && tables.dtype() == ZS_I64 && tables.is_contiguous() && tables.numel() >= 1,
             "tables: int64 [slots], each slot's extent table");
  ZS_REQUIRE(bases.defined() && bases.dtype() == ZS_I64 && bases.is_contiguous() && bases.dim() == 2 &&
             bases.size(1) == 3 && bases.size(0) >= 1, "bases: int64 [slots, 3] of k, v, indexer keys");
  ZS_REQUIRE(kc.size(0) % CHUNK == 0 && ikc.size(0) == kc.size(0) && context >= 1 && context <= kc.size(0),
             "the pools hold the furthest row");
  // ropes int64 [slots], each slot's rope int32 [2 + 3 capacity] (attention_rope.cuh) or 0 for its rows' own
  // positions; null: every slot's are (spec 019 WS-05)
  const zs::View ropes(ropes_);
  ZS_REQUIRE(!ropes.defined() || (ropes.dtype() == ZS_I64 && ropes.is_contiguous() && ropes.numel() >= tables.numel()),
             "ropes: int64 [slots]");
#define ZS_TABLE sid.data_ptr<int>(), reinterpret_cast<const int* const*>(tables.data_ptr<int64_t>()), bases.data_ptr<int64_t>(), \
                 ropes.defined() ? reinterpret_cast<const int* const*>(ropes.data_ptr<int64_t>()) : nullptr,
#elif defined(ZS_EXTENTS)
  const zs::View table(table_);
  ZS_REQUIRE(table.defined() && table.dtype() == ZS_I32 && table.is_contiguous() && kc.size(0) % CHUNK == 0 &&
             ikc.size(0) == kc.size(0) && table.numel() <= kc.size(0) / CHUNK && context >= rows &&
             context <= table.numel() * CHUNK, "table: int32 [extents] over pools of 512-row extents, holding pos + R");
#define ZS_TABLE table.data_ptr<int>(),
#else
  ZS_REQUIRE(context >= rows && context <= kc.size(0) && context <= ikc.size(0), "the caches hold pos + R rows");
#define ZS_TABLE
#endif
  ZS_REQUIRE(rows >= 1 && pw <= INT32_MAX, "at least one row");
#if defined(ZS_ROWS)
  const int* const rope_words = nullptr;    // a row's are its slot's, ROPES[SID[r]]
#elif defined(ZS_EXTENTS)
  const int* const rope_words = nullptr;    // the table's one sequence is text
  const int64_t s1 = 0, s2 = 0;
#else
  // rope int32 [2 + 3 capacity], or null: the rows' own positions (attention_rope.cuh). Only positions under its
  // length rope[1] read the table, and `Rope.set` keeps that within the capacity, whatever the context.
  const zs::View rope(rope_);
  ZS_REQUIRE(!rope.defined() || (rope.dtype() == ZS_I32 && rope.is_contiguous() && rope.numel() >= 2 &&
                                 (rope.numel() - 2) % 3 == 0),
             "rope: int32 [2 + 3 capacity]");
  const int* const rope_words = rope.defined() ? rope.data_ptr<int>() : nullptr;
#endif
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(nq + nkv + ni + 1));
  ZS_NAME(prep)<<<grid, WIDE, 0, zs::stream()>>>(
      ZS_TABLE p.data_ptr<__nv_bfloat16>(), pos.data_ptr<int>(), q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
      iq_scale.data_ptr<float>(), inv_freq.data_ptr<float>(), q.data_ptr<char>(), kc.data_ptr<char>(),
      vc.data_ptr<char>(), iq.data_ptr<__nv_bfloat16>(), ikc.data_ptr<__nv_bfloat16>(),
      f8 ? qs.data_ptr<float>() : nullptr, f8 ? ks.data_ptr<float>() : nullptr,
      f8 ? vs.data_ptr<float>() : nullptr, static_cast<float>(eps), static_cast<int>(pw), static_cast<int>(nq),
      static_cast<int>(nkv), static_cast<int>(hd), static_cast<int>(ni), static_cast<int>(ihd),
      static_cast<int>(half), v4 ? static_cast<int>(vc.size(2)) : 0, rope_words, static_cast<int>(s1),
      static_cast<int>(s2));
#undef ZS_TABLE
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

#ifndef ZS_EXTENTS
// o bf16 [R, NQ, HD] and the stacked projection p bf16 [R, PW] -> out bf16 [R, NQ * HD].
int zs_qwen4_attn_gate(const zs_view* o_, const zs_view* p_, const zs_view* out_) {
  zs::begin();
  const zs::View o(o_), p(p_), out(out_);
  ZS_REQUIRE(is(o, ZS_BF16, 3), "o: bf16 [R, heads, head_dim]");
  const int64_t rows = o.size(0), nq = o.size(1), hd = o.size(2);
  ZS_REQUIRE(is(p, ZS_BF16, 2) && p.size(0) == rows && p.size(1) >= nq * 2 * hd && p.size(1) <= INT32_MAX,
             "p: bf16 [R, PW] with the q|gate pairs first");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.numel() >= rows * nq * hd,
             "out: bf16 [R, heads * head_dim]");
  ZS_NAME(gate)<<<blocks(rows * nq * hd), 256, 0, zs::stream()>>>(
      o.data_ptr<__nv_bfloat16>(), p.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(),
      static_cast<int>(rows), static_cast<int>(p.size(1)), static_cast<int>(nq), static_cast<int>(hd));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
#endif

}  // extern "C"
