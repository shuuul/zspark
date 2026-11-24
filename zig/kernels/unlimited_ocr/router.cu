// Unlimited-OCR-2's DASH router (spec 147 WS-03): from a row's projection `q_proj`, `k_proj` (one stacked bf16
// GEMM before, bf16/gemm.cu's CUTLASS path, `bf16.Linear`), its index query and key over Gemma RMSNorm and page2d
// RoPE; a completed 32-key block's summary key `k_bar` (head_cls's softmax over the block); then a row's remote
// block scores, its page-aware candidates, the top 16, their 1.5-entmax weights, the first 12 and their bias.
// Semantics from the checkpoint's own HF code with Triton off (modeling_deepseekv2.py `_router_qk`,
// `_block_scores_infer`, `_infer_route_select`; dash_attention.py `_RMSNorm`, `entmax15`; _sparse_base_impl.py
// `DeepseekV2IndexerRotary2D`, `_build_page_aware_image_block_mask`; modeling_unlimitedocr.py
// `_uocr_dash_decode_allow`; Apache-2.0, read 2026-10-27), written for zspark in CUDA C++. Hand-written: past the
// GEMM nothing is a product of two matrices (a 128-wide norm, a block's 32-key softmax, a row's dots against its
// own blocks, a selection), and Triton is not taken (the owner, 2026-10-27).
//
// What departs from HF, on purpose: `nan_to_num` on the hidden rows, the index rows and the entmax input is not
// done (finite inputs give finite values; a NaN here is a fault upstream); the scores' dots and the sums run in this
// file's fixed orders, not cuBLAS's or torch's (fp32 either way); the top 16 breaks a tie by the lower block, where
// torch's `topk` gives no order; the entmax always takes 16 slots, where HF takes min(16, blocks): a slot past the
// row's remote blocks holds -1e4 as HF's non-candidates do, and gets weight 0 whenever one candidate exists (the
// support lies within 1 of the largest), so only ids of invalid slots differ, which nobody reads.
//
// The row rule: a row's index q, its scores, ids and bias read its own projection, its own coordinates, and the
// k_bar of blocks that end at least 128 keys before it; a block's k_bar reads that block's 32 index keys alone.
// So a row has the same bits alone, in a prompt's chunk, in a verify window or as a step. Positions come from a
// device word (`at[0]`, the window's first; row r at at[0] + r), so a captured graph replays at any position.

#include <cuda_bf16.h>
#include <stdint.h>

#include <cmath>

#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

constexpr int D = 128;          // the index head's width (sparse_index_dim)
constexpr int BLOCK = 32;       // keys a block (sparse_block_size)
constexpr int LOCAL_BLOCKS = 4; // the local window in blocks (sparse_local_block): never remote
constexpr int TOP_M = 16;       // the entmax's candidates (sparse_topk_blocks / dash_topm)
constexpr float ACT_CLAMP = 1.0e4f;  // dash_attention.py _ACT_CLAMP
constexpr float ENTMAX_NEG = -1.0e4f;  // modeling_deepseekv2.py _ENTMAX_NEG: a non-candidate's entmax input
constexpr float SCORE_NEG = -1.0e9f;   // _SCORE_NEG: a block that is not remote, in the scores written for tests

__device__ __forceinline__ float clamp_act(float x) { return fminf(fmaxf(x, -ACT_CLAMP), ACT_CLAMP); }

// Grid (row r, side s: 0 q, 1 k), D threads, thread t the value t. HF (`_router_qk`, `_RMSNorm.forward`, autocast
// off): the projection's bf16 x; x32 = x; r = rsqrt(sum x32^2 / D + eps), the squares summed a warp by warp_sum and
// the four warps in order; y = clamp(x32 r), y = clamp(y (1 + w)) with 1 + w in fp32, rounded to bf16 once. Then
// page2d RoPE (`DeepseekV2IndexerRotary2D`, cat(ang, ang), rotate_half) in bf16 as torch runs it: pair j < D / 2
// turns by a = fp32(tok) INV[j] for j < n_tok, by a = fp32(page mod page_mod) INV[j] after (the page axis);
// c = bf16(cos a), s = bf16(sin a); out_j = bf16(bf16(n_j c) + bf16(-n_{j+64} s)),
// out_{j+64} = bf16(bf16(n_{j+64} c) + bf16(n_j s)). q goes to IDXQ's row r, k to IDXK's row at[0] + r (the
// cache, linear by position); a position at or past `cap` is skipped. Shared memory a block: the normed row (512 B)
// and the warps' partial sums (16 B); nothing could be less (the rotation reads a partner's value).
__global__ void __launch_bounds__(D)
    ocr_router_prep_kernel(const __nv_bfloat16* __restrict__ QK, long long stride, const int* __restrict__ AT,
                           const int* __restrict__ COORDS, const __nv_bfloat16* __restrict__ QW,
                           const __nv_bfloat16* __restrict__ KW, const float* __restrict__ INV, int n_tok,
                           int page_mod, float eps, __nv_bfloat16* __restrict__ IDXQ,
                           __nv_bfloat16* __restrict__ IDXK, long long cap) {
  __shared__ float normed[D];
  __shared__ float partial[D / 32];
  const int r = blockIdx.x, side = blockIdx.y, t = threadIdx.x;
  const long long pos = (long long)AT[0] + r;
  if (side == 1 && pos >= cap) return;
  const float x = __bfloat162float(QK[(long long)r * stride + side * D + t]);
  const float sq = warp_sum(x * x);
  if ((t & 31) == 0) partial[t >> 5] = sq;
  __syncthreads();
  float sum = 0.f;
  for (int w = 0; w < D / 32; w++) sum += partial[w];
  const float inv_rms = rsqrtf(sum / (float)D + eps);
  const float w = __bfloat162float((side == 0 ? QW : KW)[t]);
  const float y = clamp_act(clamp_act(x * inv_rms) * (1.0f + w));
  normed[t] = bf(y);
  __syncthreads();
  const int half = D / 2, j = t < half ? t : t - half;
  const int tok = COORDS[r * 3 + 0];
  int page = COORDS[r * 3 + 1];
  if (page_mod > 0) page = ((page % page_mod) + page_mod) % page_mod;
  const float angle = (j < n_tok ? (float)tok : (float)page) * INV[j];
  const float c = bf(cosf(angle)), s = bf(sinf(angle));
  const float out = t < half ? bf(bf(normed[t] * c) + bf(-normed[t + half] * s))
                             : bf(bf(normed[t] * c) + bf(normed[t - half] * s));
  if (side == 0)
    IDXQ[(long long)r * D + t] = __float2bfloat16_rn(out);
  else
    IDXK[pos * D + t] = __float2bfloat16_rn(out);
}

// Block r, D threads: when the row at p = at[0] + r is a block's last key (p % 32 == 31), that block b = p / 32's
// summary key (`_block_scores_infer`'s head_cls branch; a completed block has every key valid):
//   aw_t = clamp(fp32(sum_d k_td h_d) / sqrt(D)) for its 32 keys t, the sum a lane's four values (d = l, l + 32,
//   l + 64, l + 96 in order) then warp_sum; w_t = exp(aw_t - max) / max(sum_t exp(..), 1e-6), the max and the sum
//   over the 32 by xor shuffles; k_bar_d = sum_t w_t k_td in t order.
// It reads the block's 32 index keys alone, so recomputing a block gives the same bits: a caller may run it over
// any window that holds a block's last row (a verify window whose last rows are dropped recomputes it with the
// rows that replace them). Shared memory a block: the 32 weights (128 B); the keys are read from global (8 KB,
// L2), twice, rather than staged: staging them would be 8 KB a block for a kernel a prompt runs once a block.
__global__ void __launch_bounds__(D)
    ocr_router_kbar_kernel(const __nv_bfloat16* __restrict__ IDXK, const __nv_bfloat16* __restrict__ HC,
                           const int* __restrict__ AT, float inv_sqrt_d, float* __restrict__ KBAR, long long blocks) {
  __shared__ float weight[BLOCK];
  const long long pos = (long long)AT[0] + blockIdx.x;
  if (pos % BLOCK != BLOCK - 1) return;
  const long long b = pos / BLOCK;
  if (b >= blocks) return;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const __nv_bfloat16* keys = IDXK + b * BLOCK * D;
  float h[4];
  for (int i = 0; i < 4; i++) h[i] = __bfloat162float(HC[lane + 32 * i]);
  for (int k = warp; k < BLOCK; k += D / 32) {
    float s = 0.f;
    for (int i = 0; i < 4; i++) s += __bfloat162float(keys[k * D + lane + 32 * i]) * h[i];
    s = warp_sum(s);
    if (lane == 0) weight[k] = clamp_act(s * inv_sqrt_d);
  }
  __syncthreads();
  if (warp == 0) {
    const float aw = weight[lane];
    float most = aw;
    for (int o = 16; o; o >>= 1) most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, o));
    const float e = expf(aw - most);
    const float total = warp_sum(e);
    weight[lane] = e / fmaxf(total, 1e-6f);
  }
  __syncthreads();
  float acc = 0.f;
  for (int k = 0; k < BLOCK; k++) acc += weight[k] * __bfloat162float(keys[k * D + t]);
  KBAR[b * D + t] = acc;
}

constexpr int ROUTE_THREADS = 128;

// The larger of two (value, block) pairs: the larger value, the lower block among equal ones.
__device__ __forceinline__ void better(float& v, int& id, float v2, int id2) {
  if (v2 > v || (v2 == v && id2 >= 0 && (id < 0 || id2 < id))) {
    v = v2;
    id = id2;
  }
}

__device__ __forceinline__ bool same_page(const int* __restrict__ BPAGE, int b, int qp) {
  const int lo = BPAGE[2 * b], hi = BPAGE[2 * b + 1];
  return qp >= 0 && lo >= 0 && lo <= qp && qp <= hi;
}

// Block r, 128 threads, the row at p = at[0] + r with its query page qp (COORDS' third column, -1 none):
//  1. Its remote blocks are b <= p / 32 - 4 (n = max(0, p / 32 - 3) of them, at most `blocks`). For each (a thread
//     a block, strided), z_b = clamp(fp32(sum_d q_d k_bar_bd) * fp32(1 / sqrt(D))), the dot in d order.
//  2. Candidates (`_build_page_aware_image_block_mask` and `_uocr_dash_decode_allow`, which agree): the remote
//     blocks holding an image token of page qp, [bmin, bmax] of BPAGE covering qp; none (qp -1, or no such block
//     remote) opens every remote block. HF's prefill tests each key's page; an interval is the same while a
//     block's image pages are contiguous, which a page's run of image tokens makes them (every row of both
//     fixtures agrees). x_b = z_b * fp32(scale) on a candidate, -1e4 otherwise (two products, as HF's).
//  3. The top 16 by x, the lower block first among equal values, by 16 rounds of a block's arg-max (a chosen
//     value is set to -inf); slots past n hold -1e4 and no block.
//  4. Thread 0: HF's sort-based 1.5-entmax over the 16 (`_Entmax15.forward`): X = (x - x_0) / 2; running sums
//     of X and of X^2 in slot order; mean, mean_sq, ss = rho (mean_sq - mean^2), delta = (1 - ss) / rho,
//     tau = mean - sqrt(max(delta, 0)); the support is the count of tau_rho <= X_rho (at least 1);
//     p = max(X - tau_support, 0)^2. Each product and sum rounded where torch rounds it (no contraction).
//  5. The first K slots: valid where the block is a candidate; logp = log(max(p, floor)); mu the valid logps'
//     sum in slot order over their count (at least 1); bias = (logp - mu) * fp32(1 / sigma). IDS gets the block of
//     a valid slot and -1 otherwise, BIAS its bias and 0. With SCORES (tests): z_b for b < n, -1e9 after, to its width.
// Shared memory a block: X, the row's x a block (4 B a remote block, `blocks` of them: the dynamic part), q
// (512 B), the warps' arg-max (32 B), the top 16 (128 B). The x could be recomputed in each round instead of
// held, at 16 times the dots; at 4 B a block (16 KB for 131,072 keys) holding it is cheaper.
__global__ void __launch_bounds__(ROUTE_THREADS)
    ocr_router_route_kernel(const __nv_bfloat16* __restrict__ IDXQ, const float* __restrict__ KBAR,
                            const int* __restrict__ AT, const int* __restrict__ COORDS,
                            const int* __restrict__ BPAGE, long long blocks, float inv_sqrt_d, float scale,
                            float floor_, float inv_sigma, int K, int* __restrict__ IDS, float* __restrict__ BIAS,
                            float* __restrict__ SCORES, int width) {
  extern __shared__ float X[];
  __shared__ float q[D];
  __shared__ float warp_v[ROUTE_THREADS / 32];
  __shared__ int warp_id[ROUTE_THREADS / 32];
  __shared__ float top_x[TOP_M];
  __shared__ int top_id[TOP_M];
  const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const long long pos = (long long)AT[0] + r;
  long long remote = pos / BLOCK - (LOCAL_BLOCKS - 1);
  if (remote < 0) remote = 0;
  if (remote > blocks) remote = blocks;
  const int n = (int)remote;
  const int qp = COORDS[r * 3 + 2];
  for (int d = t; d < D; d += ROUTE_THREADS) q[d] = __bfloat162float(IDXQ[(long long)r * D + d]);
  __syncthreads();
  bool any_same = false;
  for (int b = t; b < n; b += ROUTE_THREADS) {
    const float4* kb = reinterpret_cast<const float4*>(KBAR + (long long)b * D);
    float dot = 0.f;
    for (int i = 0; i < D / 4; i++) {
      const float4 k = kb[i];
      dot += q[4 * i] * k.x;
      dot += q[4 * i + 1] * k.y;
      dot += q[4 * i + 2] * k.z;
      dot += q[4 * i + 3] * k.w;
    }
    const float z = clamp_act(dot * inv_sqrt_d);
    X[b] = z;
    if (SCORES != nullptr && b < width) SCORES[(long long)r * width + b] = z;
    any_same |= same_page(BPAGE, b, qp);
  }
  if (SCORES != nullptr)
    for (int b = n + t; b < width; b += ROUTE_THREADS) SCORES[(long long)r * width + b] = SCORE_NEG;
  any_same = __syncthreads_or(any_same);
  for (int b = t; b < n; b += ROUTE_THREADS)
    X[b] = (!any_same || same_page(BPAGE, b, qp)) ? __fmul_rn(X[b], scale) : ENTMAX_NEG;
  __syncthreads();
  for (int k = 0; k < TOP_M; k++) {
    float v = minus_inf();
    int id = -1;
    for (int b = t; b < n; b += ROUTE_THREADS)
      if (X[b] != minus_inf()) better(v, id, X[b], b);  // -inf: chosen in an earlier round
    for (int o = 16; o; o >>= 1) {
      const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
      const int id2 = __shfl_xor_sync(0xffffffffu, id, o);
      better(v, id, v2, id2);
    }
    if (lane == 0) {
      warp_v[warp] = v;
      warp_id[warp] = id;
    }
    __syncthreads();
    if (t == 0) {
      float bv = warp_v[0];
      int bid = warp_id[0];
      for (int w = 1; w < ROUTE_THREADS / 32; w++) better(bv, bid, warp_v[w], warp_id[w]);
      if (bid >= 0) {
        top_x[k] = bv;
        top_id[k] = bid;
        X[bid] = minus_inf();
      } else {
        top_x[k] = ENTMAX_NEG;
        top_id[k] = -1;
      }
    }
    __syncthreads();
  }
  if (t != 0) return;
  float xs[TOP_M], tau[TOP_M];
  const float most = top_x[0];
  float cs = 0.f, cs2 = 0.f;
  int support = 0;
  for (int i = 0; i < TOP_M; i++) {
    xs[i] = __fsub_rn(top_x[i], most) / 2.0f;
    cs = __fadd_rn(cs, xs[i]);
    cs2 = __fadd_rn(cs2, __fmul_rn(xs[i], xs[i]));
    const float rho = (float)(i + 1);
    const float mean = cs / rho, mean_sq = cs2 / rho;
    const float ss = __fmul_rn(rho, __fsub_rn(mean_sq, __fmul_rn(mean, mean)));
    const float delta = __fsub_rn(1.0f, ss) / rho;
    tau[i] = __fsub_rn(mean, sqrtf(fmaxf(delta, 0.f)));
    support += tau[i] <= xs[i];
  }
  if (support < 1) support = 1;
  const float tau_star = tau[support - 1];
  float logp[TOP_M];
  bool valid[TOP_M];
  float sum = 0.f;
  int count = 0;
  for (int i = 0; i < K; i++) {
    const float y = fmaxf(__fsub_rn(xs[i], tau_star), 0.f);
    const float p = __fmul_rn(y, y);
    const int b = top_id[i];
    valid[i] = b >= 0 && (!any_same || same_page(BPAGE, b, qp));
    logp[i] = logf(fmaxf(p, floor_));
    if (valid[i]) {
      sum = __fadd_rn(sum, logp[i]);
      count++;
    }
  }
  const float mu = sum / (float)(count > 0 ? count : 1);
  for (int i = 0; i < K; i++) {
    IDS[(long long)r * K + i] = valid[i] ? top_id[i] : -1;
    BIAS[(long long)r * K + i] = valid[i] ? __fmul_rn(__fsub_rn(logp[i], mu), inv_sigma) : 0.f;
  }
}

}  // namespace

extern "C" {

// qk bf16 [R, 2 D] (rows may be strided): q_proj's then k_proj's outputs; at int32 [>= 1]: the window's first
// position; coords int32 [R, 3]: a row's (tok, page, query page); q_norm, k_norm bf16 [D]; inv fp32 [D / 2]: the
// pairs' inverse frequencies, the first n_tok the token axis's, the rest the page axis's; idx_q bf16 [R, D] out;
// idx_k bf16 [cap, D]: the index keys by position, row at[0] + r written.
int zs_unlimited_ocr_router_prep(const zs_view* qk_, const zs_view* at_, const zs_view* coords_, const zs_view* q_norm_,
                                 const zs_view* k_norm_, const zs_view* inv_, int64_t n_tok, int64_t page_mod,
                                 double eps, const zs_view* idx_q_, const zs_view* idx_k_) {
  zs::begin();
  const zs::View qk(qk_), at(at_), coords(coords_), qw(q_norm_), kw(k_norm_), inv(inv_), idx_q(idx_q_), idx_k(idx_k_);
  ZS_REQUIRE(qk.defined() && qk.dtype() == ZS_BF16 && qk.dim() == 2 && qk.size(1) == 2 * D, "qk: bf16 [R, 256]");
  const int64_t rows = qk.size(0);
  ZS_REQUIRE(rows >= 1 && rows <= INT32_MAX, "1 row or more");
  ZS_REQUIRE(at.defined() && at.dtype() == ZS_I32 && at.numel() >= 1, "at: int32, the first position");
  ZS_REQUIRE(coords.defined() && coords.dtype() == ZS_I32 && coords.is_contiguous() && coords.numel() == rows * 3,
             "coords: int32 [R, 3]");
  ZS_REQUIRE(qw.defined() && qw.dtype() == ZS_BF16 && qw.numel() == D && kw.defined() && kw.dtype() == ZS_BF16 &&
                 kw.numel() == D,
             "q_norm, k_norm: bf16 [128]");
  ZS_REQUIRE(inv.defined() && inv.dtype() == ZS_F32 && inv.numel() == D / 2, "inv: fp32 [64]");
  ZS_REQUIRE(n_tok >= 0 && n_tok <= D / 2 && page_mod >= 0, "n_tok in 0 .. 64, page_mod >= 0");
  ZS_REQUIRE(idx_q.defined() && idx_q.dtype() == ZS_BF16 && idx_q.is_contiguous() && idx_q.numel() == rows * D,
             "idx_q: bf16 [R, 128]");
  ZS_REQUIRE(idx_k.defined() && idx_k.dtype() == ZS_BF16 && idx_k.is_contiguous() && idx_k.dim() == 2 &&
                 idx_k.size(1) == D,
             "idx_k: bf16 [capacity, 128]");
  ocr_router_prep_kernel<<<dim3(static_cast<unsigned>(rows), 2), D, 0, zs::stream()>>>(
      qk.data_ptr<__nv_bfloat16>(), qk.row_stride(), at.data_ptr<int>(), coords.data_ptr<int>(),
      qw.data_ptr<__nv_bfloat16>(), kw.data_ptr<__nv_bfloat16>(), inv.data_ptr<float>(), static_cast<int>(n_tok),
      static_cast<int>(page_mod), static_cast<float>(eps), idx_q.data_ptr<__nv_bfloat16>(),
      idx_k.data_ptr<__nv_bfloat16>(), idx_k.size(0));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// idx_k bf16 [cap, D]; head_cls bf16 [D]; at int32 [>= 1]; rows: the window's; kbar fp32 [blocks, D]: the blocks
// that the window's rows complete are written, the others kept.
int zs_unlimited_ocr_router_kbar(const zs_view* idx_k_, const zs_view* head_cls_, const zs_view* at_, int64_t rows,
                                 const zs_view* kbar_) {
  zs::begin();
  const zs::View idx_k(idx_k_), hc(head_cls_), at(at_), kbar(kbar_);
  ZS_REQUIRE(idx_k.defined() && idx_k.dtype() == ZS_BF16 && idx_k.is_contiguous() && idx_k.dim() == 2 &&
                 idx_k.size(1) == D,
             "idx_k: bf16 [capacity, 128]");
  ZS_REQUIRE(hc.defined() && hc.dtype() == ZS_BF16 && hc.numel() == D, "head_cls: bf16 [128]");
  ZS_REQUIRE(at.defined() && at.dtype() == ZS_I32 && at.numel() >= 1, "at: int32, the first position");
  ZS_REQUIRE(rows >= 1 && rows <= INT32_MAX, "1 row or more");
  ZS_REQUIRE(kbar.defined() && kbar.dtype() == ZS_F32 && kbar.is_contiguous() && kbar.dim() == 2 &&
                 kbar.size(1) == D && kbar.size(0) * BLOCK <= idx_k.size(0),
             "kbar: fp32 [blocks, 128], blocks * 32 within idx_k's rows");
  ocr_router_kbar_kernel<<<static_cast<unsigned>(rows), D, 0, zs::stream()>>>(
      idx_k.data_ptr<__nv_bfloat16>(), hc.data_ptr<__nv_bfloat16>(), at.data_ptr<int>(),
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), kbar.data_ptr<float>(), kbar.size(0));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// idx_q bf16 [R, D]; kbar fp32 [blocks, D]; at int32 [>= 1]; coords int32 [R, 3]; bpage int32 [blocks, 2]: a
// block's least and largest image page (-1 for a block without an image token); scale: the entmax input's factor
// (dash_scale / sqrt(D)); floor, sigma: the bias's; ids int32 [R, K] and bias fp32 [R, K] out, K <= 16; scores
// fp32 [R, W] or null.
int zs_unlimited_ocr_router_route(const zs_view* idx_q_, const zs_view* kbar_, const zs_view* at_,
                                  const zs_view* coords_, const zs_view* bpage_, double scale, double floor_,
                                  double sigma, const zs_view* ids_, const zs_view* bias_, const zs_view* scores_) {
  zs::begin();
  const zs::View idx_q(idx_q_), kbar(kbar_), at(at_), coords(coords_), bpage(bpage_), ids(ids_), bias(bias_);
  ZS_REQUIRE(idx_q.defined() && idx_q.dtype() == ZS_BF16 && idx_q.is_contiguous() && idx_q.dim() == 2 &&
                 idx_q.size(1) == D,
             "idx_q: bf16 [R, 128]");
  const int64_t rows = idx_q.size(0);
  ZS_REQUIRE(rows >= 1 && rows <= INT32_MAX, "1 row or more");
  ZS_REQUIRE(kbar.defined() && kbar.dtype() == ZS_F32 && kbar.is_contiguous() && kbar.dim() == 2 &&
                 kbar.size(1) == D,
             "kbar: fp32 [blocks, 128]");
  const int64_t blocks = kbar.size(0);
  ZS_REQUIRE(blocks >= 1 && blocks * 4 <= 96 * 1024, "1 to 24,576 blocks");
  ZS_REQUIRE(at.defined() && at.dtype() == ZS_I32 && at.numel() >= 1, "at: int32, the first position");
  ZS_REQUIRE(coords.defined() && coords.dtype() == ZS_I32 && coords.is_contiguous() && coords.numel() == rows * 3,
             "coords: int32 [R, 3]");
  ZS_REQUIRE(bpage.defined() && bpage.dtype() == ZS_I32 && bpage.is_contiguous() && bpage.numel() >= blocks * 2,
             "bpage: int32 [blocks, 2]");
  ZS_REQUIRE(ids.defined() && ids.dtype() == ZS_I32 && ids.is_contiguous() && ids.dim() == 2 &&
                 ids.size(0) == rows && ids.size(1) >= 1 && ids.size(1) <= TOP_M,
             "ids: int32 [R, K], K 1 to 16");
  const int64_t K = ids.size(1);
  ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_F32 && bias.is_contiguous() && bias.numel() == rows * K,
             "bias: fp32 [R, K]");
  ZS_REQUIRE(floor_ > 0 && sigma > 0, "a positive floor and sigma");
  float* scores = nullptr;
  int width = 0;
  if (scores_ != nullptr) {
    const zs::View s(scores_);
    ZS_REQUIRE(s.defined() && s.dtype() == ZS_F32 && s.is_contiguous() && s.dim() == 2 && s.size(0) == rows,
               "scores: fp32 [R, W]");
    scores = s.data_ptr<float>();
    width = static_cast<int>(s.size(1));
  }
  const size_t dynamic = static_cast<size_t>(blocks) * sizeof(float);
  if (dynamic > 48 * 1024)
    ZS_CUDA_CHECK(cudaFuncSetAttribute(ocr_router_route_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(dynamic)));
  ocr_router_route_kernel<<<static_cast<unsigned>(rows), ROUTE_THREADS, dynamic, zs::stream()>>>(
      idx_q.data_ptr<__nv_bfloat16>(), kbar.data_ptr<float>(), at.data_ptr<int>(), coords.data_ptr<int>(),
      bpage.data_ptr<int>(), blocks, static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), static_cast<float>(scale),
      static_cast<float>(floor_), static_cast<float>(1.0 / sigma), static_cast<int>(K), ids.data_ptr<int>(),
      bias.data_ptr<float>(), scores, width);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
