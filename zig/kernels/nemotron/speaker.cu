// Nemotron-3.5-Transcribe's Sortformer speaker kernels (spec 151 WS-11): the pieces of the PEE's
// diarization expert (`parallel_expert_encoder.py`'s `_forward_online_impl` -> `forward_streaming_step`,
// `sortformer_modules.py`, commit 774bc30) that the perception encoder's kernels do not already cover:
//
// - `zs_nemotron_speaker_prep`: the 512-dim Sortformer encoder's attention prep — RoPE on the whole head
//   (`self_attention_model: "rope"`, `rotary_fraction: 1.0`) with no qk norms (`qk_norm: false`), in
//   place on the w_qkv rows. The same math as `perception.cu`'s `nemotron_prep_kernel` minus the norms:
//   rotate_half pairs p and p + ROT / 2, the angle a row's index inside its `offsets` segment times
//   inv_freq[p] = rope_base^(-2p/ROT); cos and sin bf16 and the products round to bf16 as PyTorch's two
//   elementwise ops.
// - `zs_nemotron_speaker_conv_input`: pack Conv1d(k=3, pad=1, groups=1) input as [T, H*3],
//   masking input rows past `gate`. The caller multiplies by the stored [F*H, H*3] weights
//   with the shared BF16 tensor-core GEMM, then applies ReLU to [T*F, H].
// - `zs_nemotron_speaker_fuse`: `_fuse_diar_and_asr` — y = asr_norm(x) + spk_kernel_scale *
//   (diar_norm(act > 0.5) @ diar_kernel) a row at a time, the norms LayerNorms (eps) with their own
//   weight and bias and diar_kernel the sinusoid buffer [S, D]. `act` is the bf16 probabilities;
//   thresholded features are 0/1. One rounding on the output.
//
// The row rule: a block is a row and nothing else is read between rows.
//
// Provenance: zspark's own; the norms' arithmetic is `common/layernorm.cu`'s (fp32 means, Welford's
// sums in lane order — PyTorch's nn.LayerNorm near, not to the bit) and the rotation
// `perception.cu`'s roundings.
//
// Why not CUTLASS: only `speaker_fuse`'s [1, S] x [S, D] inner product is a matmul, and a scalar-loop
// one at that.

#include <stdint.h>

#include <cuda_bf16.h>

#include "../zs/host.h"

namespace {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// qkv bf16 [R, 3 W]: parts q, k, v of W = HEADS * DH a row; offsets int32 [S + 1] the segments' first
// rows; inv fp32 [ROT / 2]. As `nemotron_prep_kernel` minus the q/k LayerNorms.
__global__ void nemotron_speaker_prep_kernel(__nv_bfloat16* __restrict__ qkv, const int* __restrict__ offsets,
                                             int segs, const float* __restrict__ inv, int heads, int dh, int rot) {
    const int row = blockIdx.x, head = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (head >= heads) return;
    int lo = 0, hi = segs;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (offsets[mid] <= row) lo = mid; else hi = mid;
    }
    const float pos = static_cast<float>(row - offsets[lo]);
    const int w = heads * dh;
    const int half = rot / 2;
    for (int part = 0; part < 2; ++part) {
        __nv_bfloat16* base = qkv + static_cast<size_t>(row) * 3 * w + part * w + head * dh;
        for (int p = lane; p < half; p += 32) {
            const float a = pos * inv[p];
            const float c = bf(cosf(a)), s = bf(sinf(a));
            const float n1 = __bfloat162float(base[p]);
            const float n2 = __bfloat162float(base[p + half]);
            base[p] = __float2bfloat16_rn(bf(n1 * c) + bf(-n2 * s));
            base[p + half] = __float2bfloat16_rn(bf(n2 * c) + bf(n1 * s));
        }
        __syncwarp();
    }
}

// Conv1d's input columns follow the stored [output channel, input channel, kernel offset] weight.
__global__ void nemotron_speaker_conv_input_kernel(const __nv_bfloat16* __restrict__ emb,
                                                    __nv_bfloat16* __restrict__ packed,
                                                    int t, int h, int gate) {
    const int row = blockIdx.x;
    for (int col = threadIdx.x; col < 3 * h; col += blockDim.x) {
        const int channel = col / 3, src = row + col % 3 - 1;
        packed[static_cast<size_t>(row) * 3 * h + col] =
            (src >= 0 && src < t && src < gate) ? emb[static_cast<size_t>(src) * h + channel]
                                              : __float2bfloat16_rn(0.f);
    }
}

// x bf16 [N] in place ReLU (`forward_speaker_sigmoids`' F.relu between the two hidden linears).
__global__ void nemotron_speaker_relu_kernel(__nv_bfloat16* __restrict__ x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(x[i]);
    if (!(v > 0.f)) x[i] = __float2bfloat16_rn(0.f);  // relu maps 0 and -0 to 0, negatives to 0
}

// logits bf16 [R, S] -> preds bf16 [R, S]: sigmoid masked at `gate` rows (`forward_speaker_sigmoids`'
// sigmoid then `apply_mask_to_preds`, one rounding each — sigmoid's f32 then the bf16 store).
__global__ void nemotron_speaker_sigmoid_kernel(const __nv_bfloat16* __restrict__ logits,
                                                __nv_bfloat16* __restrict__ preds, size_t rows, int s, size_t gate) {
    const size_t r = blockIdx.x;
    const bool live = r < gate;
    for (int e = threadIdx.x; e < s; e += blockDim.x) {
        const float l = live ? __bfloat162float(logits[r * s + e]) : 0.f;
        preds[r * s + e] = __float2bfloat16_rn(live ? 1.f / (1.f + __expf(-l)) : 0.f);
    }
}

// x bf16 [R, D] the encoder's states, act bf16 [R, S] speaker probabilities, aw/ab the ASR norm's and
// nw/nb the speaker norm's weights and biases bf16, kernel bf16 [S, D] the sinusoid; y[r] =
// LN(x[r]; aw, ab) + scale * sum_s LN((act[r] > 0.5); nw, nb)[s] * kernel[s], bf16 one rounding. D a
// multiple of 8 up to 8,192, S <= 32. A block a row, the threads striding D and the norms' sums met in
// shared (the same Welford split as `layernorm.cu`: sum and sum of squared deviations).
__global__ void nemotron_speaker_fuse_kernel(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ act,
                                             const __nv_bfloat16* __restrict__ aw,
                                             const __nv_bfloat16* __restrict__ ab,
                                             const __nv_bfloat16* __restrict__ nw,
                                             const __nv_bfloat16* __restrict__ nb,
                                             const __nv_bfloat16* __restrict__ kernel,
                                             __nv_bfloat16* __restrict__ y, int d, int s, float scale, float eps) {
    __shared__ float shared[32];
    __shared__ float spk[32];
    const size_t base = static_cast<size_t>(blockIdx.x) * d;
    const int per = (d + blockDim.x - 1) / blockDim.x;
    float sum = 0.f;
    for (int e = 0; e < per; ++e) {
        const int t = threadIdx.x + e * blockDim.x;
        if (t < d) sum += __bfloat162float(x[base + t]);
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = sum;
    __syncthreads();
    const float mean = shared[0] / static_cast<float>(d);  // one warp: the sums are one lane's
    float sq = 0.f;
    for (int e = 0; e < per; ++e) {
        const int t = threadIdx.x + e * blockDim.x;
        if (t < d) {
            const float v = __bfloat162float(x[base + t]) - mean;
            sq += v * v;
        }
    }
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) shared[1] = sq;
    __syncthreads();
    const float rstd = rsqrtf(shared[1] / static_cast<float>(d) + eps);
    if (threadIdx.x < static_cast<unsigned>(s)) {
        const float a = __bfloat162float(act[static_cast<size_t>(blockIdx.x) * s + threadIdx.x]);
        spk[threadIdx.x] = a > 0.5f ? 1.f : 0.f;
    }
    __syncthreads();
    float ssum = 0.f, ssq = 0.f;
    for (int e = 0; e < s; ++e) {
        ssum += spk[e];
        ssq += spk[e] * spk[e];
    }
    const float smean = ssum / static_cast<float>(s);
    const float srstd = rsqrtf(ssq / static_cast<float>(s) - smean * smean + eps);
    __syncthreads();
    if (threadIdx.x < static_cast<unsigned>(s))
        spk[threadIdx.x] = bf((spk[threadIdx.x] - smean) * srstd * __bfloat162float(nw[threadIdx.x]) +
                              __bfloat162float(nb[threadIdx.x]));
    __syncthreads();
    for (int t = threadIdx.x; t < d; t += blockDim.x) {
        float out = bf((__bfloat162float(x[base + t]) - mean) * rstd * __bfloat162float(aw[t]) +
                       __bfloat162float(ab[t]));
        float inf = 0.f;
        for (int e = 0; e < s; ++e) inf += spk[e] * __bfloat162float(kernel[static_cast<size_t>(e) * d + t]);
        y[base + t] = __float2bfloat16_rn(out + scale * inf);
    }
}

}  // namespace

// qkv bf16 [R, 3W] in place, offsets int32 [S + 1], inv fp32 [ROT / 2]: RoPE alone on the first ROT dims
// of each head's DH (heads, DH <= 128 a multiple of 32, rot <= DH even).
extern "C" int zs_nemotron_speaker_prep(const zs_view* qkv_, const zs_view* offsets_, const zs_view* inv_,
                                        int64_t heads_, int64_t rot_) {
    zs::begin();
    const zs::View qkv(qkv_), offsets(offsets_), inv(inv_);
    const int64_t heads = heads_, rot = rot_;
    ZS_REQUIRE(qkv.defined() && qkv.dtype() == ZS_BF16 && qkv.dim() == 2 && qkv.is_contiguous() &&
                   qkv.size(0) >= 1 && qkv.size(0) <= INT32_MAX && qkv.size(1) % 3 == 0,
               "qkv: bf16 [R, 3W]");
    const int64_t width = qkv.size(1) / 3;
    ZS_REQUIRE(heads >= 1 && heads <= 32 && width % heads == 0, "heads divides W");
    const int64_t dh = width / heads;
    ZS_REQUIRE(dh % 32 == 0 && dh <= 128, "a head's dims in 32s, at most 128");
    ZS_REQUIRE(rot >= 2 && rot <= dh && rot % 2 == 0, "rot <= a head's dims, even");
    ZS_REQUIRE(offsets.defined() && offsets.dtype() == ZS_I32 && offsets.is_contiguous() && offsets.size(0) >= 2,
               "offsets: int32 [S + 1]");
    ZS_REQUIRE(inv.defined() && inv.dtype() == ZS_F32 && inv.numel() == rot / 2, "inv: fp32 [ROT / 2]");
    nemotron_speaker_prep_kernel<<<static_cast<unsigned>(qkv.size(0)), static_cast<unsigned>(heads * 32), 0,
                                 zs::stream()>>>(qkv.data_ptr<__nv_bfloat16>(), offsets.data_ptr<int>(),
                                                 static_cast<int>(offsets.size(0) - 1), inv.data_ptr<float>(),
                                                 static_cast<int>(heads), static_cast<int>(dh),
                                                 static_cast<int>(rot));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// Caller-owned im2col scratch for the full-channel subpixel Conv1d.
extern "C" int zs_nemotron_speaker_conv_input(const zs_view* emb_, const zs_view* packed_, int64_t gate_) {
    zs::begin();
    const zs::View emb(emb_), packed(packed_);
    const int64_t gate = gate_;
    ZS_REQUIRE(emb.defined() && emb.dtype() == ZS_BF16 && emb.dim() == 2 && emb.is_contiguous() &&
                   emb.size(0) >= 1 && emb.size(0) <= INT32_MAX,
               "emb: bf16 [T, H]");
    const int64_t t = emb.size(0), h = emb.size(1);
    ZS_REQUIRE(h >= 32 && h <= 256 && h % 32 == 0, "H in 32s up to 256");
    ZS_REQUIRE(gate >= 0 && gate <= t, "gate <= T");
    ZS_REQUIRE(packed.defined() && packed.dtype() == ZS_BF16 && packed.dim() == 2 && packed.is_contiguous() &&
                   packed.size(0) == t && packed.size(1) == 3 * h,
               "packed: bf16 [T, H*3]");
    nemotron_speaker_conv_input_kernel<<<static_cast<unsigned>(t), 128, 0, zs::stream()>>>(
        emb.data_ptr<__nv_bfloat16>(), packed.data_ptr<__nv_bfloat16>(),
        static_cast<int>(t), static_cast<int>(h), static_cast<int>(gate));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// x bf16 [R, D], act bf16 [R, S], aw/ab + nw/nb the LayerNorms' bf16 [D]/[S], kernel bf16 [S, D] the
// sinusoid, y bf16 [R, D]: y = LN(x) + scale * LN(act > 0.5) @ kernel.
extern "C" int zs_nemotron_speaker_fuse(const zs_view* x_, const zs_view* act_, const zs_view* aw_,
                                        const zs_view* ab_, const zs_view* nw_, const zs_view* nb_,
                                        const zs_view* kernel_, const zs_view* y_, double scale_, double eps_) {
    zs::begin();
    const zs::View x(x_), act(act_), aw(aw_), ab(ab_), nw(nw_), nb(nb_), kernel(kernel_), y(y_);
    const int64_t r = x.size(0), d = x.size(1), s = act.size(1);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && r >= 1 &&
                   r <= INT32_MAX && d % 8 == 0 && d >= 32 && d <= 8192,
               "x: bf16 [R, D], D in 8s 32..8192");
    ZS_REQUIRE(act.defined() && act.dtype() == ZS_BF16 && act.dim() == 2 && act.is_contiguous() &&
                   act.size(0) == r && s >= 1 && s <= 32,
               "act: bf16 [R, S]");
    ZS_REQUIRE(aw.defined() && aw.numel() == d && ab.defined() && ab.numel() == d, "asr_norm: bf16 [D]");
    ZS_REQUIRE(nw.defined() && nw.numel() == s && nb.defined() && nb.numel() == s, "diar_norm: bf16 [S]");
    ZS_REQUIRE(kernel.defined() && kernel.dtype() == ZS_BF16 && kernel.numel() == s * d,
               "kernel: bf16 [S, D]");
    ZS_REQUIRE(y.defined() && y.dtype() == ZS_BF16 && y.dim() == 2 && y.is_contiguous() && y.size(0) == r &&
                   y.size(1) == d,
               "y: bf16 [R, D]");
    const int threads = 32; // The mean/variance reductions use one warp; each lane strides all D.
    nemotron_speaker_fuse_kernel<<<static_cast<unsigned>(r), threads, 0, zs::stream()>>>(
        x.data_ptr<__nv_bfloat16>(), act.data_ptr<__nv_bfloat16>(), aw.data_ptr<__nv_bfloat16>(),
        ab.data_ptr<__nv_bfloat16>(), nw.data_ptr<__nv_bfloat16>(), nb.data_ptr<__nv_bfloat16>(),
        kernel.data_ptr<__nv_bfloat16>(), y.data_ptr<__nv_bfloat16>(), static_cast<int>(d),
        static_cast<int>(s), static_cast<float>(scale_), static_cast<float>(eps_));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// x bf16 [N] in place.
extern "C" int zs_nemotron_speaker_relu(const zs_view* x_) {
    zs::begin();
    const zs::View x(x_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.is_contiguous(), "x: bf16 contiguous");
    const size_t n = x.numel();
    if (n == 0) return zs::status();
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
    nemotron_speaker_relu_kernel<<<blocks, threads, 0, zs::stream()>>>(x.data_ptr<__nv_bfloat16>(), n);
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// logits bf16 [R, S] -> preds bf16 [R, S], rows past `gate` zeroed; S <= 32.
extern "C" int zs_nemotron_speaker_sigmoid(const zs_view* logits_, const zs_view* preds_, int64_t gate_) {
    zs::begin();
    const zs::View logits(logits_), preds(preds_);
    const int64_t gate = gate_;
    ZS_REQUIRE(logits.defined() && logits.dtype() == ZS_BF16 && logits.dim() == 2 && logits.is_contiguous() &&
                   preds.defined() && preds.dtype() == ZS_BF16 && preds.dim() == 2 && preds.is_contiguous() &&
                   logits.size(0) == preds.size(0) && logits.size(1) == preds.size(1) && logits.size(1) <= 32,
               "logits and preds: bf16 [R, S <= 32]");
    ZS_REQUIRE(gate >= 0 && gate <= logits.size(0), "gate within R");
    nemotron_speaker_sigmoid_kernel<<<static_cast<unsigned>(logits.size(0)), 32, 0, zs::stream()>>>(
        logits.data_ptr<__nv_bfloat16>(), preds.data_ptr<__nv_bfloat16>(), static_cast<size_t>(logits.size(0)),
        static_cast<int>(logits.size(1)), static_cast<size_t>(gate));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}
