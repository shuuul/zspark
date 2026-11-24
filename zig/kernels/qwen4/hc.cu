// Flash Next's hyper-connections: the residual is S streams of D values a row, written back into after each
// branch and read out (norm, down, SiLU and inject gates, up, mix) before the next. Every value is a function
// of its own row alone, with fp32 sums in a fixed order and each bf16 rounding where the model has one. The
// embedding that fills the streams is common/embed.cu.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/glue.py and exl3_mm.py (Triton), Apache-2.0;
// rewritten for zspark in CUDA C++, without the 32-group sums only the MLX 4-bit matmul reads.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../bf16/reduce.cuh"
#include "../common/math.cuh"
#include "../zs/host.h"

namespace {

using namespace common_math;

constexpr int BLOCK = 256;      // values of a stream a block of the write-back squares and sums
constexpr int LANES = BLOCK / 4;  // the threads of such a block: four values each, read and written as one

// Four bf16 values side by side, moved as 8 bytes.
struct alignas(8) Quad {
  __nv_bfloat16 v[4];
};

// Whether a buffer starts where `bytes`-wide loads may: rows and blocks keep the alignment from there.
bool aligned(const void* p, uintptr_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; }

// Block (row, c) of LANES threads, four values each: values [c BLOCK, (c + 1) BLOCK) of each stream. MODE 0
// leaves the streams,
// 1 adds a bf16 branch, 2 the MoE's slots times their weights (routed slots in order, then the shared expert's
// own rows as the last slot, rounded once); each stream takes bf16(h + branch * inject_s). The shared
// expert's row r starts at shared + r * SH: SH is D for a buffer of its own, slots * D where it sits in y's last
// slot (an EXL3 pack's table computes it there).
// The product is not rounded before the sum, though glue.py's source reads
// (hv + (branch * inj).to(bfloat16).to(float32)).to(bfloat16): the kernel Triton compiles from it rounds once.
// TensorFold's exported streams say so: replayed on the host from its own h, branch, slots, weights and gates,
// bf16(h + branch * inject) gives every value of h_mid, the next h_in and final.h (0 of 67 million of each differ
// over five references, NVFP4, hybrid and EXL3) and the source's two roundings miss one value in ten (2.8e-3 of the rms).
// Both factors are bf16, so the fp32 product is exact and a contraction to fma changes nothing; the slots' sum
// does contract (acc + y * w as one fma), which is also what the exported streams follow.
// pss [R, D / BLOCK, S]: each stream's squares summed over the block.
template <int MODE>
__global__ void __launch_bounds__(LANES)
    writeback_kernel(__nv_bfloat16* __restrict__ H, float* __restrict__ pss, const __nv_bfloat16* __restrict__ branch,
                     const __nv_bfloat16* __restrict__ inject, const float* __restrict__ y,
                     const float* __restrict__ wts, const float* __restrict__ shared, long long SH, int D, int S,
                     int slots) {
  __shared__ float squares[BLOCK];
  const int r = blockIdx.x, c = blockIdx.y, tid = threadIdx.x, d = c * BLOCK + tid * 4;
  const int NC = D / BLOCK;
  float b[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (MODE == 1) {
    const Quad q = *reinterpret_cast<const Quad*>(branch + (size_t)r * D + d);
    for (int j = 0; j < 4; ++j) b[j] = __bfloat162float(q.v[j]);
  } else if constexpr (MODE == 2) {
    for (int k = 0; k < slots; ++k) {
      const float* from = k < slots - 1 ? y + ((size_t)r * slots + k) * D + d : shared + (size_t)r * SH + d;
      const float4 v = *reinterpret_cast<const float4*>(from);
      const float w = wts[(size_t)r * slots + k];
      b[0] = fmaf(v.x, w, b[0]);
      b[1] = fmaf(v.y, w, b[1]);
      b[2] = fmaf(v.z, w, b[2]);
      b[3] = fmaf(v.w, w, b[3]);
    }
    for (int j = 0; j < 4; ++j) b[j] = bf(b[j]);
  } else if constexpr (MODE == 3) {
    // the slots' sum as the experts' own launch left it (mode 2's fma chain, spec 073 WS-07), rounded here
    const float4 v = *reinterpret_cast<const float4*>(y + (size_t)r * D + d);
    b[0] = bf(v.x);
    b[1] = bf(v.y);
    b[2] = bf(v.z);
    b[3] = bf(v.w);
  }
  for (int s = 0; s < S; ++s) {
    Quad* at = reinterpret_cast<Quad*>(H + ((size_t)r * S + s) * D + d);
    Quad q = *at;
    float hv[4];
    for (int j = 0; j < 4; ++j) hv[j] = __bfloat162float(q.v[j]);
    if constexpr (MODE != 0) {
      const float inj = __bfloat162float(inject[(size_t)r * S + s]);
      for (int j = 0; j < 4; ++j) {
        hv[j] = bf(hv[j] + b[j] * inj);
        q.v[j] = __float2bfloat16_rn(hv[j]);
      }
      *at = q;
    }
    // The block's 256 squares fold halves onto halves, from 128 apart down to 1. A thread holds four values
    // side by side, so the folds from 128 to 4 apart pair threads 32 to 1 apart, value for value: the first
    // through shared memory (the block is two warps), the rest by shuffles in the first warp; the folds 2 and
    // 1 apart are within its first thread. The product is rounded on its own, whatever it is added to.
    for (int j = 0; j < 4; ++j) squares[tid * 4 + j] = __fmul_rn(hv[j], hv[j]);
    __syncthreads();
    if (tid < 32) {
      float total[4];
      for (int j = 0; j < 4; ++j) total[j] = squares[tid * 4 + j] + squares[(tid + 32) * 4 + j];
      for (int half = 16; half > 0; half >>= 1)
        for (int j = 0; j < 4; ++j) total[j] += __shfl_down_sync(0xffffffffu, total[j], half);
      if (tid == 0) pss[((size_t)r * NC + c) * S + s] = (total[0] + total[2]) + (total[1] + total[3]);
    }
    __syncthreads();
  }
}

// Block (row, stream, c) of LANES threads over BLOCK values, four a thread: bf16(h / rms_s * scale), rms_s from
// the stream's block sums in order, which the block's first thread sums for all of them.
__global__ void __launch_bounds__(LANES)
    normed_kernel(const __nv_bfloat16* __restrict__ H, const float* __restrict__ pss, const float* __restrict__ scale,
                  __nv_bfloat16* __restrict__ normed, float eps, int R, int D, int S) {
  __shared__ float inverse;
  const long long first = (long long)blockIdx.x * BLOCK;
  const int wide = S * D;
  const int r = static_cast<int>(first / wide), s = static_cast<int>(first % wide) / D, NC = D / BLOCK;
  if (threadIdx.x == 0) {
    float total = 0.f;
    for (int c = 0; c < NC; ++c) total += pss[((size_t)r * NC + c) * S + s];
    inverse = 1.0f / sqrtf(total / D + eps);
  }
  __syncthreads();
  const long long i = first + threadIdx.x * 4;
  const float4 sc = *reinterpret_cast<const float4*>(scale + i % wide);
  const Quad q = *reinterpret_cast<const Quad*>(H + i);
  Quad out;
  out.v[0] = __float2bfloat16_rn(__bfloat162float(q.v[0]) * inverse * sc.x);
  out.v[1] = __float2bfloat16_rn(__bfloat162float(q.v[1]) * inverse * sc.y);
  out.v[2] = __float2bfloat16_rn(__bfloat162float(q.v[2]) * inverse * sc.z);
  out.v[3] = __float2bfloat16_rn(__bfloat162float(q.v[3]) * inverse * sc.w);
  *reinterpret_cast<Quad*>(normed + i) = out;
}

// The write-back and the norm after it as one launch (spec 073 WS-03, `ZSPARK_HC_JOINED`): block (row, stream), D / 4
// threads, a thread four values of its stream. `writeback_kernel`'s values and sums in its order, then
// `normed_kernel`'s from the same registers: h is read once and not twice, and a site is a launch less. The same
// bits as the two (a test holds them). A row's streams are its blocks' alone. ExLlamaV3 (151539c,
// exllamav3/cpp/norm.cu, RES_IN) joins a residual add and a norm likewise; none of its code is here. A block a row over all four
// streams (nine barriers, a stream's fold after another's) was the first form and slower than the two launches
// under 16 rows (5.7 against 4.3 us at a row, 6.9 against 4.4 at two); a stream a block has three barriers and
// four times the blocks. With slots each stream's block sums them again: they are in L2, the routed table's
// launch wrote them.
// Shared memory, a block (a stream of a row): squares, D fp32 (10,240 B at D 2560), for the fold 128 apart,
// which only the upper warp of each pair needs to leave there (half would do); sums, D / 256 fp32 (40 B), the
// pss values again for the stream's own norm; inverse. 10.3 KB, so 9 blocks an SM.
constexpr int SITE_D = 4096;
template <int MODE>
__global__ void __launch_bounds__(SITE_D / 4)
    site_kernel(__nv_bfloat16* __restrict__ H, float* __restrict__ pss, const __nv_bfloat16* __restrict__ branch,
                const __nv_bfloat16* __restrict__ inject, const float* __restrict__ y, const float* __restrict__ wts,
                const float* __restrict__ shared, long long SH, int D, int S, int slots,
                const float* __restrict__ scale, __nv_bfloat16* __restrict__ normed, float eps) {
  __shared__ float squares[SITE_D];
  __shared__ float sums[SITE_D / BLOCK];
  __shared__ float inverse;
  const int r = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, d = tid * 4, c = tid / LANES, in = tid % LANES;
  const int NC = D / BLOCK;
  float b[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (MODE == 1) {
    const Quad q = *reinterpret_cast<const Quad*>(branch + (size_t)r * D + d);
    for (int j = 0; j < 4; ++j) b[j] = __bfloat162float(q.v[j]);
  } else if constexpr (MODE == 2) {
    for (int k = 0; k < slots; ++k) {
      const float* from = k < slots - 1 ? y + ((size_t)r * slots + k) * D + d : shared + (size_t)r * SH + d;
      const float4 v = *reinterpret_cast<const float4*>(from);
      const float w = wts[(size_t)r * slots + k];
      b[0] = fmaf(v.x, w, b[0]);
      b[1] = fmaf(v.y, w, b[1]);
      b[2] = fmaf(v.z, w, b[2]);
      b[3] = fmaf(v.w, w, b[3]);
    }
    for (int j = 0; j < 4; ++j) b[j] = bf(b[j]);
  } else if constexpr (MODE == 3) {
    // the slots' sum as the experts' own launch left it (mode 2's fma chain, spec 073 WS-07), rounded here
    const float4 v = *reinterpret_cast<const float4*>(y + (size_t)r * D + d);
    b[0] = bf(v.x);
    b[1] = bf(v.y);
    b[2] = bf(v.z);
    b[3] = bf(v.w);
  }
  const size_t i = ((size_t)r * S + s) * D + d;
  Quad* at = reinterpret_cast<Quad*>(H + i);
  Quad q = *at;
  float hv[4];
  for (int j = 0; j < 4; ++j) hv[j] = __bfloat162float(q.v[j]);
  if constexpr (MODE != 0) {
    const float inj = __bfloat162float(inject[(size_t)r * S + s]);
    for (int j = 0; j < 4; ++j) {
      hv[j] = bf(hv[j] + b[j] * inj);
      q.v[j] = __float2bfloat16_rn(hv[j]);
    }
    *at = q;
  }
  // `writeback_kernel`'s fold, a pair of warps a block of 256 values
  for (int j = 0; j < 4; ++j) squares[tid * 4 + j] = __fmul_rn(hv[j], hv[j]);
  __syncthreads();
  if (in < 32) {
    float total[4];
    for (int j = 0; j < 4; ++j) total[j] = squares[tid * 4 + j] + squares[(tid + 32) * 4 + j];
    for (int half = 16; half > 0; half >>= 1)
      for (int j = 0; j < 4; ++j) total[j] += __shfl_down_sync(0xffffffffu, total[j], half);
    if (in == 0) {
      const float sum = (total[0] + total[2]) + (total[1] + total[3]);
      pss[((size_t)r * NC + c) * S + s] = sum;
      sums[c] = sum;
    }
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.f;
    for (int k = 0; k < NC; ++k) total += sums[k];
    inverse = 1.0f / sqrtf(total / D + eps);
  }
  __syncthreads();
  const float4 sc = *reinterpret_cast<const float4*>(scale + (size_t)s * D + d);
  const float inv = inverse;
  Quad out;
  out.v[0] = __float2bfloat16_rn(__bfloat162float(q.v[0]) * inv * sc.x);
  out.v[1] = __float2bfloat16_rn(__bfloat162float(q.v[1]) * inv * sc.y);
  out.v[2] = __float2bfloat16_rn(__bfloat162float(q.v[2]) * inv * sc.z);
  out.v[3] = __float2bfloat16_rn(__bfloat162float(q.v[3]) * inv * sc.w);
  *reinterpret_cast<Quad*>(normed + i) = out;
}

// Thread (row, i) over the down projection's [low | inject] fp32 values: act = bf16(silu(bf16(v / S))), and the
// inject gates bf16(2 bf16(sigmoid(bf16(v / S)))). dn is the projection's SK slices [SK, R, ndn], summed here in
// slice order (one slice: the projection itself).
__global__ void act_kernel(const float* __restrict__ dn, __nv_bfloat16* __restrict__ act,
                           __nv_bfloat16* __restrict__ inject, int R, int ndn, int low, int S, int SK) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)R * ndn;
  if (i >= total) return;
  const int r = static_cast<int>(i / ndn), c = static_cast<int>(i % ndn);
  const float v = bf(b16_slices(dn, i, total, SK) / S);
  if (c < low) {
    act[(size_t)r * low + c] = __float2bfloat16_rn(bsilu(v));
  } else if (inject != nullptr) {
    inject[(size_t)r * S + (c - low)] = __float2bfloat16_rn(2.0f * bsig(v));
  }
}

// Thread (row, d): mixed = bf16((sum over streams in order of bf16(bf16(sigmoid(up_s)) * normed_s)) / S).
__global__ void mix_kernel(const __nv_bfloat16* __restrict__ up, const __nv_bfloat16* __restrict__ normed,
                           __nv_bfloat16* __restrict__ mixed, int R, int D, int S) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const int r = static_cast<int>(i / D), d = static_cast<int>(i % D);
  float total = 0.f;
  for (int s = 0; s < S; ++s) {
    const size_t at = ((size_t)r * S + s) * D + d;
    total += bf(bsig(__bfloat162float(up[at])) * __bfloat162float(normed[at]));
  }
  mixed[i] = __float2bfloat16_rn(total / S);
}

// Thread (row, d): out = mean over S streams of h. mean_mode 0: f32 acc / S, rn-even (GLM). 1: round to bf16
// after each add in stream order, then / S. `OS`: elements from one row of `out` to the next (a tap among
// the five, side by side).
__global__ void mean_kernel(const __nv_bfloat16* __restrict__ H, __nv_bfloat16* __restrict__ out, int R, int D,
                            int S, int mean_mode, long long OS) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * D) return;
  const int r = static_cast<int>(i / D), d = static_cast<int>(i % D);
  const __nv_bfloat16* x = H + ((size_t)r * S) * D + d;
  if (mean_mode == 0) {
    float acc = __bfloat162float(x[0]);
    for (int s = 1; s < S; ++s) acc = acc + __bfloat162float(x[(size_t)s * D]);
    out[(size_t)r * OS + d] = __float2bfloat16_rn(acc / S);
  } else {
    __nv_bfloat16 acc = x[0];
    for (int s = 1; s < S; ++s)
      acc = __float2bfloat16_rn(__bfloat162float(acc) + __bfloat162float(x[(size_t)s * D]));
    out[(size_t)r * OS + d] = __float2bfloat16_rn(__bfloat162float(acc) / S);
  }
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

bool streams_of(const zs::View& h, int64_t streams) {
  return h.defined() && h.dtype() == ZS_BF16 && h.dim() == 2 && h.is_contiguous() && streams >= 1 &&
         h.size(1) % streams == 0 && (h.size(1) / streams) % BLOCK == 0;
}

}  // namespace

namespace {

// What `zs_qwen4_hc_site` adds to a write-back: the norm's scale and outputs.
struct Norm {
  const zs_view* scale;
  const zs_view* normed;
  double eps;
};

// h bf16 [R, S * D] in place, pss fp32 [R, D / 256, S]. mode 0: the squares alone; 1: branch bf16 [R, D] and
// inject bf16 [R, S]; 2: y fp32 [R * slots, D] (its last slot unread), the shared expert's fp32 [R, D] in that
// slot's place, wts fp32 [R, slots] and inject; without ``shared`` y's last slot is read. 3: y fp32 [R, D], mode
// 2's sum of the slots before its rounding (an EXL3 table's `forwardSum`), and inject: mode 2's bits.
int writeback(const zs_view* h_, const zs_view* pss_, int64_t mode, const zs_view* branch_,
              const zs_view* inject_, const zs_view* y_, const zs_view* wts_, const zs_view* shared_,
              int64_t streams, const Norm* norm) {
  zs::begin();
  const zs::View h(h_), pss(pss_), branch(branch_), inject(inject_), y(y_), wts(wts_), shared(shared_);
  ZS_REQUIRE(streams_of(h, streams), "h: bf16 [R, S * D], D in blocks of 256");
  const int64_t rows = h.size(0), d = h.size(1) / streams;
  ZS_REQUIRE(pss.defined() && pss.dtype() == ZS_F32 && pss.numel() >= rows * (d / BLOCK) * streams,
             "pss: fp32 [R, D / 256, S]");
  ZS_REQUIRE(mode >= 0 && mode <= 3, "mode 0-3");
  ZS_REQUIRE(aligned(h.data_ptr<__nv_bfloat16>(), 8) && (mode != 1 || aligned(branch.data_ptr<__nv_bfloat16>(), 8)),
             "h and branch start on 8 bytes");
  ZS_REQUIRE(mode == 0 || (inject.defined() && inject.dtype() == ZS_BF16 && inject.numel() >= rows * streams),
             "inject: bf16 [R, S]");
  ZS_REQUIRE(mode != 1 || (branch.defined() && branch.dtype() == ZS_BF16 && branch.is_contiguous() &&
                           branch.numel() >= rows * d), "branch: bf16 [R, D]");
  int64_t slots = 0;
  if (mode == 2) {
    ZS_REQUIRE(wts.defined() && wts.dtype() == ZS_F32 && wts.dim() == 2 && wts.is_contiguous() &&
               wts.size(0) >= rows, "wts: fp32 [R, slots]");
    slots = wts.size(1);
    ZS_REQUIRE(slots >= 1 && y.defined() && y.dtype() == ZS_F32 && y.numel() >= rows * slots * d,
               "y: fp32 [R * slots, D]");
    ZS_REQUIRE(!shared.defined() || (shared.dtype() == ZS_F32 && shared.is_contiguous() && shared.numel() >= rows * d),
               "shared: fp32 [R, D]");
  }
  ZS_REQUIRE(mode != 2 || (aligned(y.data_ptr<float>(), 16) &&
                           (!shared.defined() || aligned(shared.data_ptr<float>(), 16))),
             "y and shared start on 16 bytes");
  ZS_REQUIRE(mode != 3 || (y.defined() && y.dtype() == ZS_F32 && y.is_contiguous() && y.numel() >= rows * d &&
                           aligned(y.data_ptr<float>(), 16)), "the slots' sum: fp32 [R, D] from 16 bytes");
  const float* shared_rows = nullptr;
  long long shared_stride = d;
  if (mode == 2) {
    if (shared.defined()) {
      shared_rows = shared.data_ptr<float>();
    } else {
      shared_rows = y.data_ptr<float>() + (slots - 1) * d;
      shared_stride = slots * d;
    }
  }
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(d / BLOCK));
  const auto stream = zs::stream();
  __nv_bfloat16* hp = h.data_ptr<__nv_bfloat16>();
  const int di = static_cast<int>(d), si = static_cast<int>(streams), sl = static_cast<int>(slots);
  if (norm != nullptr) {
    const zs::View scale(norm->scale), normed(norm->normed);
    const int64_t wide = h.size(1);
    ZS_REQUIRE(d <= SITE_D, "a joined site: D to 4096");
    ZS_REQUIRE(scale.defined() && scale.dtype() == ZS_F32 && scale.numel() == wide, "scale: fp32 [S * D]");
    ZS_REQUIRE(normed.defined() && normed.dtype() == ZS_BF16 && normed.is_contiguous() &&
               normed.numel() >= rows * wide, "normed: bf16 [R, S * D]");
    ZS_REQUIRE(aligned(normed.data_ptr<__nv_bfloat16>(), 8) && aligned(scale.data_ptr<float>(), 16),
               "normed starts on 8 bytes, scale on 16");
    const dim3 row_blocks(static_cast<unsigned>(rows), static_cast<unsigned>(streams));
    const unsigned threads = static_cast<unsigned>(d / 4);
    const float* sp = scale.data_ptr<float>();
    __nv_bfloat16* np = normed.data_ptr<__nv_bfloat16>();
    const float e = static_cast<float>(norm->eps);
    if (mode == 0)
      site_kernel<0><<<row_blocks, threads, 0, stream>>>(hp, pss.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr,
                                                         nullptr, 0, di, si, sl, sp, np, e);
    else if (mode == 1)
      site_kernel<1><<<row_blocks, threads, 0, stream>>>(hp, pss.data_ptr<float>(), branch.data_ptr<__nv_bfloat16>(),
                                                         inject.data_ptr<__nv_bfloat16>(), nullptr, nullptr, nullptr,
                                                         0, di, si, sl, sp, np, e);
    else if (mode == 2)
      site_kernel<2><<<row_blocks, threads, 0, stream>>>(hp, pss.data_ptr<float>(), nullptr,
                                                         inject.data_ptr<__nv_bfloat16>(), y.data_ptr<float>(),
                                                         wts.data_ptr<float>(), shared_rows, shared_stride, di, si,
                                                         sl, sp, np, e);
    else
      site_kernel<3><<<row_blocks, threads, 0, stream>>>(hp, pss.data_ptr<float>(), nullptr,
                                                         inject.data_ptr<__nv_bfloat16>(), y.data_ptr<float>(),
                                                         nullptr, nullptr, 0, di, si, sl, sp, np, e);
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
  }
  if (mode == 0)
    writeback_kernel<0><<<grid, LANES, 0, stream>>>(hp, pss.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr, nullptr,
                                                    0, di, si, sl);
  else if (mode == 1)
    writeback_kernel<1><<<grid, LANES, 0, stream>>>(hp, pss.data_ptr<float>(), branch.data_ptr<__nv_bfloat16>(),
                                                    inject.data_ptr<__nv_bfloat16>(), nullptr, nullptr, nullptr, 0, di,
                                                    si, sl);
  else if (mode == 2)
    writeback_kernel<2><<<grid, LANES, 0, stream>>>(hp, pss.data_ptr<float>(), nullptr,
                                                    inject.data_ptr<__nv_bfloat16>(), y.data_ptr<float>(),
                                                    wts.data_ptr<float>(), shared_rows, shared_stride, di, si, sl);
  else
    writeback_kernel<3><<<grid, LANES, 0, stream>>>(hp, pss.data_ptr<float>(), nullptr,
                                                    inject.data_ptr<__nv_bfloat16>(), y.data_ptr<float>(), nullptr,
                                                    nullptr, 0, di, si, sl);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // namespace

extern "C" {

int zs_qwen4_hc_writeback(const zs_view* h, const zs_view* pss, int64_t mode, const zs_view* branch,
                       const zs_view* inject, const zs_view* y, const zs_view* wts, const zs_view* shared,
                       int64_t streams) {
  return writeback(h, pss, mode, branch, inject, y, wts, shared, streams, nullptr);
}

// `zs_qwen4_hc_writeback` and `zs_qwen4_hc_normed` of the same h as one launch: the same bits.
int zs_qwen4_hc_site(const zs_view* h, const zs_view* pss, int64_t mode, const zs_view* branch, const zs_view* inject,
                  const zs_view* y, const zs_view* wts, const zs_view* shared, int64_t streams,
                  const zs_view* scale, const zs_view* normed, double eps) {
  const Norm norm = {scale, normed, eps};
  return writeback(h, pss, mode, branch, inject, y, wts, shared, streams, &norm);
}

// h bf16 [R, S * D], pss from the write-back, scale fp32 [S * D] -> normed bf16 [R, S * D].
int zs_qwen4_hc_normed(const zs_view* h_, const zs_view* pss_, const zs_view* scale_, const zs_view* normed_,
                    double eps, int64_t streams) {
  zs::begin();
  const zs::View h(h_), pss(pss_), scale(scale_), normed(normed_);
  ZS_REQUIRE(streams_of(h, streams), "h: bf16 [R, S * D], D in blocks of 256");
  const int64_t rows = h.size(0), wide = h.size(1), d = wide / streams;
  ZS_REQUIRE(pss.defined() && pss.dtype() == ZS_F32 && pss.numel() >= rows * (d / BLOCK) * streams,
             "pss: fp32 [R, D / 256, S]");
  ZS_REQUIRE(scale.defined() && scale.dtype() == ZS_F32 && scale.numel() == wide, "scale: fp32 [S * D]");
  ZS_REQUIRE(normed.defined() && normed.dtype() == ZS_BF16 && normed.is_contiguous() &&
             normed.numel() >= rows * wide, "normed: bf16 [R, S * D]");
  ZS_REQUIRE(aligned(h.data_ptr<__nv_bfloat16>(), 8) && aligned(normed.data_ptr<__nv_bfloat16>(), 8) &&
             aligned(scale.data_ptr<float>(), 16), "h and normed start on 8 bytes, scale on 16");
  normed_kernel<<<static_cast<unsigned>(rows * wide / BLOCK), LANES, 0, zs::stream()>>>(
      h.data_ptr<__nv_bfloat16>(), pss.data_ptr<float>(), scale.data_ptr<float>(), normed.data_ptr<__nv_bfloat16>(),
      static_cast<float>(eps), static_cast<int>(rows), static_cast<int>(d), static_cast<int>(streams));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// dn fp32 [SK, R, low (+ S)], the down projection's K slices unsummed (zs_b16_gemm without ``out``), or [R, ..]
// for the projection itself -> act bf16 [R, low] and, when the projection carries them, inject gates bf16
// [R, S].
int zs_qwen4_hc_act(const zs_view* dn_, const zs_view* act_, const zs_view* inject_, int64_t streams,
                 int64_t low) {
  zs::begin();
  const zs::View dn(dn_), act(act_), inject(inject_);
  ZS_REQUIRE(dn.defined() && dn.dtype() == ZS_F32 && (dn.dim() == 2 || dn.dim() == 3) && dn.is_contiguous(),
             "dn: fp32 [R, N] or [SK, R, N]");
  const int64_t sk = dn.dim() == 3 ? dn.size(0) : 1, rows = dn.size(dn.dim() - 2), ndn = dn.size(dn.dim() - 1);
  ZS_REQUIRE(sk >= 1 && sk <= INT32_MAX, "1 slice or more");
  ZS_REQUIRE(ndn == low || ndn == low + streams, "dn holds low values, then S inject gates or none");
  ZS_REQUIRE(act.defined() && act.dtype() == ZS_BF16 && act.is_contiguous() && act.numel() >= rows * low,
             "act: bf16 [R, low]");
  ZS_REQUIRE(ndn == low || (inject.defined() && inject.dtype() == ZS_BF16 && inject.numel() >= rows * streams),
             "inject: bf16 [R, S]");
  act_kernel<<<blocks(rows * ndn), 256, 0, zs::stream()>>>(
      dn.data_ptr<float>(), act.data_ptr<__nv_bfloat16>(), ndn == low ? nullptr : inject.data_ptr<__nv_bfloat16>(),
      static_cast<int>(rows), static_cast<int>(ndn),
      static_cast<int>(low), static_cast<int>(streams), static_cast<int>(sk));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// up and normed bf16 [R, S * D] -> mixed bf16 [R, D].
int zs_qwen4_hc_mix(const zs_view* up_, const zs_view* normed_, const zs_view* mixed_, int64_t streams) {
  zs::begin();
  const zs::View up(up_), normed(normed_), mixed(mixed_);
  ZS_REQUIRE(streams_of(up, streams), "up: bf16 [R, S * D]");
  const int64_t rows = up.size(0), d = up.size(1) / streams;
  ZS_REQUIRE(normed.defined() && normed.dtype() == ZS_BF16 && normed.is_contiguous() &&
             normed.numel() >= rows * up.size(1), "normed: bf16 [R, S * D]");
  ZS_REQUIRE(mixed.defined() && mixed.dtype() == ZS_BF16 && mixed.is_contiguous() && mixed.numel() >= rows * d,
             "mixed: bf16 [R, D]");
  mix_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(
      up.data_ptr<__nv_bfloat16>(), normed.data_ptr<__nv_bfloat16>(), mixed.data_ptr<__nv_bfloat16>(),
      static_cast<int>(rows), static_cast<int>(d), static_cast<int>(streams));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}


// h bf16 [R, S * D] -> out bf16 [R, D], the streams' mean. `out`'s rows may be strided: D columns of a wider
// buffer (a tap among the drafter's, side by side). mean_mode 0: GLM f32-acc; 1: bf16 after each add.
int zs_qwen4_hc_stream_mean(const zs_view* h_, const zs_view* out_, int64_t streams, int64_t mean_mode) {
  zs::begin();
  const zs::View h(h_), out(out_);
  ZS_REQUIRE(h.defined() && h.dtype() == ZS_BF16 && h.dim() == 2 && h.is_contiguous() && streams >= 1 &&
                 h.size(1) % streams == 0 && h.size(0) <= INT32_MAX,
             "h: bf16 [R, S * D]");
  const int64_t rows = h.size(0), d = h.size(1) / streams;
  ZS_REQUIRE(d >= 1 && mean_mode >= 0 && mean_mode <= 1, "D >= 1, mean_mode 0 or 1");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16, "out: bf16 [R, D]");
  const bool strided = !out.is_contiguous();
  ZS_REQUIRE(strided ? out.size(0) >= rows && out.size(1) == d && out.row_stride() > d : out.numel() >= rows * d,
             "out: bf16 [R, D], contiguous or D columns of wider rows");
  mean_kernel<<<blocks(rows * d), 256, 0, zs::stream()>>>(
      h.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(), static_cast<int>(rows), static_cast<int>(d),
      static_cast<int>(streams), static_cast<int>(mean_mode), strided ? out.row_stride() : d);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // extern "C"
