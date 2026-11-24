// The text attention's prep two families launch (spec 142 WS-04): a row's q, k and v heads from the stacked
// projection, q and k each over its RMS norm where the model has one, rotated by split-half RoPE, and the row's k
// and v written to the caches at its position. Qwen3-ASR's language model takes the normed GQA form;
// Unlimited-OCR-2 uses the page2d MHA form described below, before its DASH attend.
// Moved from qwen3_asr/attention.cu (spec 138 WS-04), its text unchanged but the norm made optional; Unlimited-OCR's
// own prep (unlimited_ocr/attention.cu's, spec 137 WS-04) was the same arithmetic on another thread map and is gone:
// each value's products, sums and roundings are the same expressions, a test holds the two to the bit.
// Semantics from the qwen-asr package's transformers backend, modeling_qwen3_asr.py
// (Qwen3ASRThinkerTextAttention, Qwen3ASRThinkerTextRMSNorm, apply_rotary_pos_emb, rotate_half,
// Qwen3ASRThinkerTextRotaryEmbedding; Apache-2.0, read 2026-10-25), and Unlimited-OCR's modeling_deepseekv2.py
// (SlidingWindowLlamaAttention, read 2026-10-25), in bf16 as HF runs them: each product and sum rounded to bf16
// where torch rounds it. TensorFold has no such model; Flash Next's prep (qwen4/attention.cu) takes a gate and an
// indexer key beside its heads, so it does not fit.
// Unlimited-OCR-2's page2d RoPE (spec 147 WS-05, `zs_attn_prep_page2d`) is the same kernel with a row's coordinates
// in place of its position: `_sparse_base_impl.py`'s 2d rotary and `_llama_apply_rotary_pos_emb` (read
// 2026-10-27), the router's prep's rotation (unlimited_ocr/router.cu) on the attention's heads.
// Without inverse frequencies (NemotronH's NoPE attention, spec 151; vLLM's nemotron_h.py NemotronHAttention, no
// rotary) the heads go out as normed, unrotated.
// Hand-written because it is a row's element-wise maps and a head's one reduction. Shared memory: the two to four
// warps' partial sums of squares (16 B).

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"
#include "math.cuh"

namespace {

using namespace common_math;

// Block (row r, head slot s) of D / 2 threads, thread t the values t and t + D / 2 of the head: slots 0 .. NQ - 1
// are q's heads, NQ .. NQ + NKV - 1 k's, whose block also copies v's head of the same index. The head:
//   with norm weights, n = bf16(w bf16(x rsqrt(sum x^2 / D + eps))), the squares summed a thread (t, then
//   t + D / 2), a warp by warp_sum, then the warps in order; without, n = x;
//   at position p = POS[0] + r, the angle a = fp32(p) INV[t] (HF's fp32 product), c = bf16(cos a), s = bf16(sin a);
//   with COORDS (int32 [R, 3]: a row's token, its page, a word not read), page2d: a = fp32(tok) INV[t] for
//   t < n_tok, a = fp32(page mod page_mod) INV[t] after (page_mod 0: the page itself); k and v still go to p;
//   out_t = bf16(bf16(n_t c) + bf16(-n_{t+D/2} s)), out_{t+D/2} = bf16(bf16(n_{t+D/2} c) + bf16(n_t s)).
// A row reads its own values and its own position alone.
__global__ void attn_prep_split_kernel(const __nv_bfloat16* __restrict__ QKV, long long stride,
                                       const __nv_bfloat16* __restrict__ QW, const __nv_bfloat16* __restrict__ KW,
                                       const float* __restrict__ INV, const int* __restrict__ POS,
                                       __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KC,
                                       __nv_bfloat16* __restrict__ VC, int NQ, int NKV, int D, float eps,
                                       const int* __restrict__ COORDS, int n_tok, int page_mod) {
  __shared__ float partial[4];
  const int r = blockIdx.x, slot = blockIdx.y, t = threadIdx.x, half = D / 2;
  const bool is_q = slot < NQ;
  const int kh = slot - NQ;
  const __nv_bfloat16* x = QKV + (size_t)r * stride + (size_t)slot * D;
  const float x1 = __bfloat162float(x[t]), x2 = __bfloat162float(x[t + half]);
  float n1 = x1, n2 = x2;
  if (QW != nullptr) {
    float ss = warp_sum(x1 * x1 + x2 * x2);
    if ((t & 31) == 0) partial[t >> 5] = ss;
    __syncthreads();
    ss = partial[0];
    for (int w = 1; w < half / 32; ++w) ss += partial[w];
    const float inv = rsqrtf(ss / D + eps);
    const __nv_bfloat16* W = is_q ? QW : KW;
    n1 = bf(__bfloat162float(W[t]) * bf(x1 * inv));
    n2 = bf(__bfloat162float(W[t + half]) * bf(x2 * inv));
  }
  const int p = POS[0] + r;
  if (INV == nullptr) {  // no rotation (NoPE, Nemotron's attention, spec 151): the heads as normed
    if (is_q) {
      __nv_bfloat16* q = Q + ((size_t)r * NQ + slot) * D;
      q[t] = __float2bfloat16_rn(n1);
      q[t + half] = __float2bfloat16_rn(n2);
      return;
    }
    const size_t at = ((size_t)p * NKV + kh) * D;
    KC[at + t] = __float2bfloat16_rn(n1);
    KC[at + t + half] = __float2bfloat16_rn(n2);
    const __nv_bfloat16* v = x + (size_t)NKV * D;
    VC[at + t] = v[t];
    VC[at + t + half] = v[t + half];
    return;
  }
  float a;
  if (COORDS == nullptr) {
    a = static_cast<float>(p) * INV[t];
  } else {
    int page = COORDS[r * 3 + 1];
    if (page_mod > 0) page = ((page % page_mod) + page_mod) % page_mod;
    a = static_cast<float>(t < n_tok ? COORDS[r * 3] : page) * INV[t];
  }
  const float c = bf(cosf(a)), s = bf(sinf(a));
  const __nv_bfloat16 o1 = __float2bfloat16_rn(bf(n1 * c) + bf(-n2 * s));
  const __nv_bfloat16 o2 = __float2bfloat16_rn(bf(n2 * c) + bf(n1 * s));
  if (is_q) {
    __nv_bfloat16* q = Q + ((size_t)r * NQ + slot) * D;
    q[t] = o1;
    q[t + half] = o2;
    return;
  }
  const size_t at = ((size_t)p * NKV + kh) * D;
  KC[at + t] = o1;
  KC[at + t + half] = o2;
  const __nv_bfloat16* v = x + (size_t)NKV * D;
  VC[at + t] = v[t];
  VC[at + t + half] = v[t + half];
}

int attn_prep(const zs_view* qkv_, const zs_view* q_norm_, const zs_view* k_norm_, const zs_view* inv_,
              const zs_view* pos_, const zs_view* q_, const zs_view* kc_, const zs_view* vc_, double eps,
              const zs_view* coords_, int64_t n_tok, int64_t page_mod, bool paged) {
  zs::begin();
  const zs::View qkv(qkv_), qw(q_norm_), kw(k_norm_), inv(inv_), pos(pos_), q(q_), kc(kc_), vc(vc_);
  ZS_REQUIRE(q.defined() && q.dtype() == ZS_BF16 && q.dim() == 3 && q.is_contiguous(), "q: bf16 [R, NQ, D]");
  const int64_t rows = q.size(0), nq = q.size(1), d = q.size(2);
  ZS_REQUIRE(d == 64 || d == 128 || d == 256, "D 64, 128 or 256");
  ZS_REQUIRE(kc.defined() && kc.dtype() == ZS_BF16 && kc.dim() == 3 && kc.size(2) == d && vc.defined() &&
                 vc.dtype() == ZS_BF16 && vc.dim() == 3 && vc.size(0) == kc.size(0) && vc.size(1) == kc.size(1) &&
                 vc.size(2) == d,
             "kc, vc: bf16 [capacity, NKV, D]");
  const int64_t nkv = kc.size(1);
  ZS_REQUIRE(nkv >= 1 && nq % nkv == 0 && nq + nkv <= 65535, "NKV dividing NQ");
  ZS_REQUIRE(qkv.defined() && qkv.dtype() == ZS_BF16 && qkv.dim() == 2 && qkv.size(0) == rows &&
                 qkv.size(1) == (nq + 2 * nkv) * d && rows >= 1 && rows <= INT32_MAX,
             "qkv: bf16 [R, (NQ + 2 NKV) D]");
  ZS_REQUIRE(qw.defined() == kw.defined(), "q_norm and k_norm both or neither");
  ZS_REQUIRE(!qw.defined() || (qw.dtype() == ZS_BF16 && qw.numel() == d && kw.dtype() == ZS_BF16 && kw.numel() == d),
             "q_norm, k_norm: bf16 [D]");
  ZS_REQUIRE((!inv.defined() && !paged) || (inv.defined() && inv.dtype() == ZS_F32 && inv.numel() == d / 2),
             "inv: fp32 [D / 2], or null for no rotation");
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() >= 1, "pos: int32 [1]");
  const zs::View co(coords_);
  ZS_REQUIRE(co.defined() == paged, "coords: given for page2d alone");
  ZS_REQUIRE(!co.defined() || (co.dtype() == ZS_I32 && co.is_contiguous() && co.numel() == rows * 3 && n_tok >= 0 &&
                               n_tok <= d / 2 && page_mod >= 0),
             "coords: int32 [R, 3], n_tok in 0 .. D / 2, page_mod >= 0");
  attn_prep_split_kernel<<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>(nq + nkv)),
                           static_cast<unsigned>(d / 2), 0, zs::stream()>>>(
      qkv.data_ptr<__nv_bfloat16>(), qkv.row_stride(), qw.defined() ? qw.data_ptr<__nv_bfloat16>() : nullptr,
      kw.defined() ? kw.data_ptr<__nv_bfloat16>() : nullptr, inv.defined() ? inv.data_ptr<float>() : nullptr, pos.data_ptr<int>(),
      q.data_ptr<__nv_bfloat16>(), kc.data_ptr<__nv_bfloat16>(), vc.data_ptr<__nv_bfloat16>(), static_cast<int>(nq),
      static_cast<int>(nkv), static_cast<int>(d), static_cast<float>(eps), co.defined() ? co.data_ptr<int>() : nullptr,
      static_cast<int>(n_tok), static_cast<int>(page_mod));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // namespace

extern "C" {

// qkv bf16 [R, (NQ + 2 NKV) D] (q's heads, k's, v's; rows may be strided), q_norm, k_norm bf16 [D] or both null (no
// norm), inv fp32 [D / 2] (RoPE's inverse frequencies) or null (no rotation: NoPE, Nemotron's), pos int32 [1] or more (the window's first position, on the
// device; what follows it is not read) -> q bf16 [R, NQ, D]; k and v into kc, vc bf16 [capacity, NKV, D] at rows
// pos .. pos + R - 1, which the caller keeps within capacity. D 64, 128 or 256.
int zs_attn_prep(const zs_view* qkv_, const zs_view* q_norm_, const zs_view* k_norm_, const zs_view* inv_,
                 const zs_view* pos_, const zs_view* q_, const zs_view* kc_, const zs_view* vc_, double eps) {
  return attn_prep(qkv_, q_norm_, k_norm_, inv_, pos_, q_, kc_, vc_, eps, nullptr, 0, 0, false);
}

// The same without norms, rotated by page2d RoPE: coords int32 [R, 3] (a row's token and page, a word not read), inv's
// first n_tok pairs the token axis's, the rest the page axis's, the page taken mod page_mod (0: as it is).
int zs_attn_prep_page2d(const zs_view* qkv_, const zs_view* inv_, const zs_view* pos_, const zs_view* coords_,
                        int64_t n_tok, int64_t page_mod, const zs_view* q_, const zs_view* kc_, const zs_view* vc_) {
  return attn_prep(qkv_, nullptr, nullptr, inv_, pos_, q_, kc_, vc_, 0, coords_, n_tok, page_mod, true);
}

}  // extern "C"
