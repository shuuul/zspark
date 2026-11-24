// Flash Next's n-gram embedding branch (PLE): the gathered table rows side by side as a row's embedding, the gate
// of its value projection by each stream's key-query product, and the dilated convolution over the gated rows that
// is added to the streams. A row reads its own values and the normed rows before it (the sequence's tail, then the
// window's earlier rows, which hold what the tail would after their commit), so a window equals its serial steps.
// Sums run over a block's fixed threads, then a fixed tree; each bf16 rounding is where the model has one.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/glue.py (_ple_embed_bf16, _ple_gate,
// _ple_conv, Triton), Apache-2.0; rewritten for zspark in CUDA C++, without the 32-group sums only the MLX 4-bit
// matmul reads. Hand-written because none of it is a matmul: the two projections between run on bf16/gemm.cu.
//
// The retired EXL3 packs' packed-row decode (rows_kernel, zs_qwen4_ple_rows) is in tag
// retired-checkpoints-2026-10-21.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

constexpr int BLOCK = 256;      // threads of a gate block; thread i sums values i, i + BLOCK, ... of a stream

// Thread i over every value: out = bf16(v * scale). Row r's heads lie side by side in both, so the index is one.
__global__ void ple_embed_kernel(const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out, float scale,
                             long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  out[i] = __float2bfloat16_rn(__bfloat162float(v[i]) * scale);
}

// The block's BLOCK partial sums to one, in a tree that depends on nothing but BLOCK; every thread gets the total.
__device__ __forceinline__ float reduce(float* sums, int tid, float mine) {
  sums[tid] = mine;
  __syncthreads();
  for (int half = BLOCK / 2; half > 0; half >>= 1) {
    if (tid < half) sums[tid] += sums[tid + half];
    __syncthreads();
  }
  const float total = sums[0];
  __syncthreads();
  return total;
}

// Block (row, stream) of BLOCK threads. kn = bf16(key / rms * norm_key), qn = bf16(h / rms * norm_query);
// gate = bf16(sum bf16(kn qn) / sqrt(D)), then bf16(sign(gate) sqrt(max(|gate|, 1e-6))) (0 stays 0);
// gated = bf16(bf16(sigmoid(gate)) * value), and pss [R, S] its squares summed.
__global__ void __launch_bounds__(BLOCK)
    ple_gate_kernel(const __nv_bfloat16* __restrict__ keys, const __nv_bfloat16* __restrict__ vals,
                const __nv_bfloat16* __restrict__ H, const float* __restrict__ nk, const float* __restrict__ nq,
                __nv_bfloat16* __restrict__ gated, float* __restrict__ pss, float eps, int D, int S) {
  __shared__ float sums[BLOCK];
  const int r = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
  const size_t base = ((size_t)r * S + s) * D;
  float kss = 0.f, qss = 0.f;
  for (int d = tid; d < D; d += BLOCK) {
    const float kv = __bfloat162float(keys[base + d]), hv = __bfloat162float(H[base + d]);
    kss += kv * kv;
    qss += hv * hv;
  }
  const float krinv = 1.0f / sqrtf(reduce(sums, tid, kss) / D + eps);
  const float qrinv = 1.0f / sqrtf(reduce(sums, tid, qss) / D + eps);
  float dot = 0.f;
  for (int d = tid; d < D; d += BLOCK) {
    const float kn = bf(__bfloat162float(keys[base + d]) * krinv * nk[(size_t)s * D + d]);
    const float qn = bf(__bfloat162float(H[base + d]) * qrinv * nq[(size_t)s * D + d]);
    dot += bf(kn * qn);
  }
  const float raw = bf(reduce(sums, tid, dot) / sqrtf(static_cast<float>(D)));
  const float mag = sqrtf(fmaxf(fabsf(raw), 1e-6f));
  const float sg = bsig(bf(raw > 0.f ? mag : (raw < 0.f ? -mag : 0.f)));
  float gss = 0.f;
  for (int d = tid; d < D; d += BLOCK) {
    const float gv = bf(sg * __bfloat162float(vals[(size_t)r * D + d]));
    gated[base + d] = __float2bfloat16_rn(gv);
    gss += gv * gv;
  }
  const float total = reduce(sums, tid, gss);
  if (tid == 0) pss[(size_t)r * S + s] = total;
}

// A gated value over its stream's RMS, times norm_conv, in bf16: what the convolution reads and the tail keeps.
// Two products and one rounding, so the bits are the same wherever it is evaluated.
__device__ __forceinline__ float conv_normed(const __nv_bfloat16* __restrict__ gated, const float* __restrict__ pss,
                                             const float* __restrict__ nc, float eps, int row, int e, int D, int S) {
  const float rinv = 1.0f / sqrtf(pss[(size_t)row * S + e / D] / D + eps);
  return bf(__bfloat162float(gated[(size_t)row * S * D + e]) * rinv * nc[e]);
}

// Thread (row, e) over the S * D values. Tap t reads row r + t * DIL of [tail (NT = (TAPS - 1) * DIL rows); the
// window's normed rows], so the last tap is the row itself; taps are summed in order in fp32.
// h += bf16(gated + bf16(silu(bf16(sum)))), and nrow keeps the row's normed values for the tail.
__global__ void conv_kernel(const __nv_bfloat16* __restrict__ gated, const float* __restrict__ pss,
                            const float* __restrict__ nc, const __nv_bfloat16* __restrict__ tail,
                            const __nv_bfloat16* __restrict__ cw, __nv_bfloat16* __restrict__ H,
                            __nv_bfloat16* __restrict__ nrow, float eps, int R, int D, int S, int TAPS, int DIL) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int W = S * D;
  if (i >= (long long)R * W) return;
  const int r = static_cast<int>(i / W), e = static_cast<int>(i % W), NT = (TAPS - 1) * DIL;
  const float own = conv_normed(gated, pss, nc, eps, r, e, D, S);
  float acc = 0.f;
  for (int t = 0; t < TAPS; ++t) {
    const int at = r + t * DIL;
    const float xv = at < NT ? __bfloat162float(tail[(size_t)at * W + e])
                             : (at - NT == r ? own : conv_normed(gated, pss, nc, eps, at - NT, e, D, S));
    acc += __bfloat162float(cw[(size_t)e * TAPS + t]) * xv;
  }
  const float ple = bf(__bfloat162float(gated[i]) + bsilu(bf(acc)));
  H[i] = __float2bfloat16_rn(__bfloat162float(H[i]) + ple);
  nrow[i] = __float2bfloat16_rn(own);
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

// bf16 [R, S * D], dense.
bool streams_of(const zs::View& h, int64_t streams) {
  return h.defined() && h.dtype() == ZS_BF16 && h.dim() == 2 && h.is_contiguous() && streams >= 1 &&
         h.size(0) >= 1 && h.size(1) >= streams && h.size(1) % streams == 0;
}

}  // namespace

extern "C" {

// v bf16 [R * heads, dh], the gathered rows (row r's head h at r * heads + h) -> out bf16 [R, heads * dh], each
// value times the table's scale and rounded.
int zs_qwen4_ple_embed(const zs_view* v_, const zs_view* out_, int64_t heads, double scale) {
  zs::begin();
  const zs::View v(v_), out(out_);
  ZS_REQUIRE(v.defined() && v.dtype() == ZS_BF16 && v.dim() == 2 && v.is_contiguous(), "v: bf16 [R * heads, dh]");
  ZS_REQUIRE(heads >= 1 && v.size(0) % heads == 0, "v holds whole rows of heads");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.dim() == 2 && out.is_contiguous() &&
             out.size(0) == v.size(0) / heads && out.size(1) == heads * v.size(1), "out: bf16 [R, heads * dh]");
  const long long total = v.numel();
  if (total == 0) return zs::status();
  ple_embed_kernel<<<blocks(total), 256, 0, zs::stream()>>>(v.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(),
                                                       static_cast<float>(scale), total);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// keys and h bf16 [R, S * D], vals bf16 [R, D], norm_key and norm_query fp32 [S * D] -> gated bf16 [R, S * D]
// and pss fp32 [R, S], each stream's gated squares summed.
int zs_qwen4_ple_gate(const zs_view* keys_, const zs_view* vals_, const zs_view* h_, const zs_view* norm_key_,
                   const zs_view* norm_query_, const zs_view* gated_, const zs_view* pss_, double eps,
                   int64_t streams) {
  zs::begin();
  const zs::View keys(keys_), vals(vals_), h(h_), nk(norm_key_), nq(norm_query_), gated(gated_), pss(pss_);
  ZS_REQUIRE(streams_of(h, streams), "h: bf16 [R, S * D]");
  const int64_t rows = h.size(0), wide = h.size(1), d = wide / streams;
  ZS_REQUIRE(streams_of(keys, streams) && keys.size(0) == rows && keys.size(1) == wide, "keys: bf16 [R, S * D]");
  ZS_REQUIRE(vals.defined() && vals.dtype() == ZS_BF16 && vals.dim() == 2 && vals.is_contiguous() &&
             vals.size(0) == rows && vals.size(1) == d, "vals: bf16 [R, D]");
  ZS_REQUIRE(nk.defined() && nk.dtype() == ZS_F32 && nk.numel() == wide, "norm_key: fp32 [S * D]");
  ZS_REQUIRE(nq.defined() && nq.dtype() == ZS_F32 && nq.numel() == wide, "norm_query: fp32 [S * D]");
  ZS_REQUIRE(streams_of(gated, streams) && gated.size(0) == rows && gated.size(1) == wide,
             "gated: bf16 [R, S * D]");
  ZS_REQUIRE(pss.defined() && pss.dtype() == ZS_F32 && pss.dim() == 2 && pss.is_contiguous() &&
             pss.size(0) == rows && pss.size(1) == streams, "pss: fp32 [R, S]");
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(streams));
  ple_gate_kernel<<<grid, BLOCK, 0, zs::stream()>>>(
      keys.data_ptr<__nv_bfloat16>(), vals.data_ptr<__nv_bfloat16>(), h.data_ptr<__nv_bfloat16>(),
      nk.data_ptr<float>(), nq.data_ptr<float>(), gated.data_ptr<__nv_bfloat16>(), pss.data_ptr<float>(),
      static_cast<float>(eps), static_cast<int>(d), static_cast<int>(streams));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// gated bf16 [R, S * D] and pss fp32 [R, S] from the gate, norm_conv fp32 [S * D], tail bf16
// [(taps - 1) * dilation, S * D] (the sequence's last normed rows, oldest first), conv bf16 [S * D, taps]:
// h bf16 [R, S * D] takes the branch in place, and nrow bf16 [R, S * D] gets the rows the tail slides over.
int zs_qwen4_ple_conv(const zs_view* gated_, const zs_view* pss_, const zs_view* norm_conv_, const zs_view* tail_,
                   const zs_view* conv_, const zs_view* h_, const zs_view* nrow_, double eps, int64_t streams,
                   int64_t dilation) {
  zs::begin();
  const zs::View gated(gated_), pss(pss_), nc(norm_conv_), tail(tail_), conv(conv_), h(h_), nrow(nrow_);
  ZS_REQUIRE(streams_of(h, streams), "h: bf16 [R, S * D]");
  const int64_t rows = h.size(0), wide = h.size(1), d = wide / streams;
  ZS_REQUIRE(streams_of(gated, streams) && gated.size(0) == rows && gated.size(1) == wide,
             "gated: bf16 [R, S * D]");
  ZS_REQUIRE(pss.defined() && pss.dtype() == ZS_F32 && pss.dim() == 2 && pss.is_contiguous() &&
             pss.size(0) == rows && pss.size(1) == streams, "pss: fp32 [R, S]");
  ZS_REQUIRE(nc.defined() && nc.dtype() == ZS_F32 && nc.numel() == wide, "norm_conv: fp32 [S * D]");
  ZS_REQUIRE(conv.defined() && conv.dtype() == ZS_BF16 && conv.dim() == 2 && conv.is_contiguous() &&
             conv.size(0) == wide && conv.size(1) >= 1, "conv: bf16 [S * D, taps]");
  const int64_t taps = conv.size(1);
  ZS_REQUIRE(dilation >= 1, "dilation: 1 or more");
  ZS_REQUIRE(tail.defined() && tail.dtype() == ZS_BF16 && tail.numel() == (taps - 1) * dilation * wide,
             "tail: bf16 [(taps - 1) * dilation, S * D]");
  ZS_REQUIRE(streams_of(nrow, streams) && nrow.size(0) == rows && nrow.size(1) == wide, "nrow: bf16 [R, S * D]");
  ZS_REQUIRE(h.data_ptr() != nrow.data_ptr() && h.data_ptr() != gated.data_ptr() &&
             nrow.data_ptr() != gated.data_ptr(), "h, gated and nrow are three buffers");
  conv_kernel<<<blocks(rows * wide), 256, 0, zs::stream()>>>(
      gated.data_ptr<__nv_bfloat16>(), pss.data_ptr<float>(), nc.data_ptr<float>(), tail.data_ptr<__nv_bfloat16>(),
      conv.data_ptr<__nv_bfloat16>(), h.data_ptr<__nv_bfloat16>(), nrow.data_ptr<__nv_bfloat16>(),
      static_cast<float>(eps), static_cast<int>(rows), static_cast<int>(d), static_cast<int>(streams),
      static_cast<int>(taps), static_cast<int>(dilation));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
