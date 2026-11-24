// bf16 rows times NVFP4 or MXFP8 weights read as stored, with the linear's bias and an activation in the
// epilogue: out (M, N) = act(x (M, K) @ W^T * scale + bias), for many rows at once (the vision tower's linears,
// spec 019 WS-08). A weight is exact in bf16 (an e2m1 code times its e4m3 block scale, an e4m3 byte times its
// power of two), the products are summed in one fp32 chain over K on the tensor cores (bf16 `m16n8k16`), the
// tensor's scale (ModelOpt's `weight_scale_2`, 1 for MXFP8) multiplies the sum, the bias is added in fp32 and
// the result rounded to bf16 once; an activation then takes that bf16 value in fp32 and is rounded again:
// `bf16/gemm_bias.cu`'s epilogue (cuBLASLt's bias epilogue, then PyTorch's elementwise GELU), so a BF16 and a
// quantized linear of the tower differ by their weights alone.
//
// Provenance: the main loop is TensorFold 0.6.5's (609ca41) `nvfp4/prompt.cu` (Apache-2.0), its device code
// copied unchanged but for the epilogue and the F32 output dropped; prompt.cu stays as taken. Mode 1 is its FP8
// case (e4m3 bytes, no block scales, the tensor scale in the epilogue) for Nemotron's encoder (spec 151 WS-03). The weights' layouts are `weights/nvfp4.zig`'s `Fp4Linear` and `Mx8Linear`
// (words [npad/64][K/64][8][32][2], bytes [npad/64][K/64][8][32][2][8], block scales [npad/64][K/64][64][4|2]).
// Why not CUTLASS: its SM120 block-scaled kernels take FP4 or FP8 on both sides, and bf16 rows times FP4 or
// FP8 weights is not an instruction here (`cuda-gb10`); the activations stay bf16 (the owner: as stored).
//
// The row rule: a row's sum runs over K in the order of K's groups of 64 alone, no K split, and the tile (by
// shape) and the blocks' order change no element's arithmetic; a row past M is staged as zeros and never
// stored. So a row's bits are its own whatever the rows beside it.
//
// Shared memory a block (STAGES ring of cp.async stages; nothing else staged, the epilogue reads registers):
//
// | Region a stage | Bytes, BM 128 x BN 128 | For | Could be less |
// | --- | --- | --- | --- |
// | x: BM rows of 64 bf16 inputs, swizzled | 16,384 | ldmatrix A fragments | no: an m16 tile's rows a k16 step |
// | w: BN columns of 64 inputs | 4,096 (NVFP4) or 8,192 (MXFP8) | the B fragments, decoded in registers | no |
// | block scales | 512 (NVFP4: 4 e4m3 a column) or 256 (MXFP8: 2 ue8m0) | the B columns' scales | no |
//
// A stage is rounded up to 128 B: NVFP4 20,992, MXFP8 24,832. Tile 0 (128 x 128, 8 warps, 3 stages): 62,976 or
// 74,496 B, one block an SM; tile 4 (128 x 128, 4 warps, 2 stages): 41,984 or 49,664 B, two an SM; tile 5 (64 x
// 128, 2 warps, 2 stages): 25,600 or 33,280 B. Tile 4 is the one taken (`weights/nvfp4.zig`, `biasedTile`):
// two blocks an SM were 10% faster than tile 0's one on the tower at 4,096 patches. The x region is the most of a stage: rows are shared by the
// BN columns, so it is the larger BN, not a smaller BM, that would amortize it further (not tried).

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../common/qmm_frag.cuh"
#include "../zs/host.h"
#include "codes.cuh"

namespace {

using namespace nvfp4_codes;

using namespace qmm_frag;

enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2 };

constexpr int GS = 64;                                    // inputs a pipeline stage

// bf16x2 of two e4m3 bytes (bits [0, 8) and [8, 16)) times ``unit`` (bf16x2 2^120 x a power of two): exact products.
__device__ __forceinline__ uint32_t fp8pair(uint32_t w, uint32_t unit) {
    const uint32_t x = (w & 0xFFu) | ((w & 0xFF00u) << 8);
    const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(unit), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ uint32_t mul2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
    return r;
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }

__device__ __forceinline__ float bf_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// The activation of the bias-added sum, as `bf16/gemm_bias.cu`'s `Act`: 0 none (rounded once by the store), 1
// gelu_pytorch_tanh and 2 exact GELU, each of the sum rounded to bf16 first.
template <int ACT>
__device__ __forceinline__ float act(float x) {
    if (ACT == 0) return x;
    const float v = bf_round(x);
    if (ACT == 1) {
        const float kBeta = 1.41421356237309504880f * 1.12837916709551257390f * 0.5f;
        const float inner = kBeta * (v + 0.044715f * v * v * v);
        return 0.5f * v * (1.f + tanhf(inner));
    }
    return v * 0.5f * (1.f + erff(v * 0.70710678118654752440f));
}

template <int MODE, int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NT = BN / WN / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;
    static constexpr int W = MODE == FP4 ? BN * GS / 2 : BN * GS;
    static constexpr int S = MODE == FP4 ? BN * 4 : MODE == MXFP8 ? BN * 2 : 0;   // block scales a group
    static constexpr int STAGE = (X + W + S + 127) / 128 * 128;
    static constexpr int SMEM = STAGES * STAGE;
};

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, int ACT>
__global__ void __launch_bounds__(WM * WN * 32) prompt_bias_kernel(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out, int M, int N, int K,
        int npad, int ldx, int group) {
    using T = Tile<MODE, BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16,
                  x + static_cast<size_t>(min(m0 + r, M - 1)) * ldx + g * GS + ch * 8, m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : 64 * GS;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            if (n0 + t * 64 < npad)
                cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
        if constexpr (T::S > 0) {
            constexpr int PER = T::S / (BN / 64);
            unsigned char* ps = pw + T::W;
            for (int c = tid; c < T::S / 16; c += T::THREADS) {
                const int t = c / (PER / 16), off = c % (PER / 16);
                if (n0 + t * 64 < npad)
                    cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64 + t) * KG + g) * PER + off * 16);
            }
        }
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    for (int g = 0; g < KG; ++g) {
        wait<STAGES - 2>();
        __syncthreads();
        if (g + STAGES - 1 < KG) load((g + STAGES - 1) % STAGES, g + STAGES - 1);
        commit();
        const unsigned char* p = stage(g % STAGES);
        const unsigned char* pw = p + T::X;
        const uint8_t* ps = p + T::X + T::W;
        uint4 wq[T::NT];
        uint32_t sv[T::NT][4];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int jj = wn * T::NT + j;
            const int col = (jj & 7) * 8 + (lane >> 2);
            if constexpr (MODE == FP4) {
                const uint2 u = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                wq[j] = make_uint4(u.x, u.y, 0u, 0u);
                const uint32_t b4 = *reinterpret_cast<const uint32_t*>(ps + (jj >> 3) * 256 + col * 4);
#pragma unroll
                for (int blk = 0; blk < 4; ++blk)
                    sv[j][blk] = fp8pair(((b4 >> (8 * blk)) & 0xFFu) * 0x101u, 0x7B807B80u);
            } else {
                wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
                if constexpr (MODE == MXFP8) {              // FP8 has no block scales
                    const uint32_t e2 = *reinterpret_cast<const uint16_t*>(ps + (jj >> 3) * 128 + col * 2);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint32_t u = (((e2 >> (8 * h)) & 0xFFu) + 120u) << 7;
                        sv[j][h] = u | (u << 16);
                    }
                }
            }
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                uint32_t b0, b1;
                if constexpr (MODE == FP4) {
                    const uint32_t word = kt < 2 ? wq[j].x : wq[j].y;
                    b0 = mul2(fp4pair(word, (kt & 1) * 8), sv[j][kt]);
                    b1 = mul2(fp4pair(word, (kt & 1) * 8 + 4), sv[j][kt]);
                } else {
                    const uint32_t word = comp(wq[j], kt);
                    const uint32_t unit = MODE == MXFP8 ? sv[j][kt >> 1] : 0x7B807B80u;
                    b0 = fp8pair(word & 0xFFFFu, unit);
                    b1 = fp8pair(word >> 16, unit);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
    // the epilogue: scale, bias, one rounding, then the activation of the rounded value (gemm_bias.cu's)
#pragma unroll
    for (int j = 0; j < T::NT; ++j) {
        const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
        const float b0 = col < N ? __bfloat162float(bias[col]) : 0.f;
        const float b1 = col + 1 < N ? __bfloat162float(bias[col + 1]) : 0.f;
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = act<ACT>(acc[i][j][2 * h] * scale + b0);
                const float v1 = act<ACT>(acc[i][j][2 * h + 1] * scale + b1);
                __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0)
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                else {
                    if (col < N) dst[0] = __float2bfloat16_rn(v0);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                }
            }
    }
}

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, int ACT>
void launch(const zs::View& x, const zs::View& w, const zs::View& bs, double scale, const zs::View& bias,
            zs::View& out, int N, int npad) {
    using T = Tile<MODE, BM, BN, WM, WN, STAGES>;
    const int M = x.size(0), K = x.size(1);
    auto kernel = prompt_bias_kernel<MODE, BM, BN, WM, WN, STAGES, ACT>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        configured = true;
    }
    const int rows_t = (M + BM - 1) / BM;
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K * 2))));
    kernel<<<rows_t * ((N + BN - 1) / BN), T::THREADS, T::SMEM, zs::stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const unsigned char*>(w.data_ptr()),
        bs.defined() ? reinterpret_cast<const uint8_t*>(bs.data_ptr()) : nullptr, static_cast<float>(scale),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M,
        N, K, npad, M == 1 ? K : static_cast<int>(x.stride(0)), group);
    ZS_CUDA_LAUNCH_CHECK();
}

template <int MODE, int ACT>
void by_tile(int tile, const zs::View& x, const zs::View& w, const zs::View& bs, double scale, const zs::View& bias,
             zs::View& out, int N, int npad) {
    switch (tile) {
        case 4: launch<MODE, 128, 128, 2, 2, 2, ACT>(x, w, bs, scale, bias, out, N, npad); break;
        case 5: launch<MODE, 64, 128, 1, 2, 2, ACT>(x, w, bs, scale, bias, out, N, npad); break;
        default: launch<MODE, 128, 128, 2, 4, 3, ACT>(x, w, bs, scale, bias, out, N, npad); break;
    }
}

template <int MODE>
void by_act(int act, int tile, const zs::View& x, const zs::View& w, const zs::View& bs, double scale,
            const zs::View& bias, zs::View& out, int N, int npad) {
    if (act == 0) by_tile<MODE, 0>(tile, x, w, bs, scale, bias, out, N, npad);
    else if (act == 1) by_tile<MODE, 1>(tile, x, w, bs, scale, bias, out, N, npad);
    else by_tile<MODE, 2>(tile, x, w, bs, scale, bias, out, N, npad);
}

bool aligned16(const zs::View& x) {
    return reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0);
}

}  // namespace

// out bf16 (M, n) = act(x bf16 (M, K) @ W^T * scale + bias bf16 [n]): mode 0 NVFP4 (`Fp4Linear`'s words and e4m3
// block scales [npad/64, K/64, 64, 4]), 1 per-tensor FP8 (`Fp8Linear`'s bytes, `bs` null), 2 MXFP8 (`Mx8Linear`'s
// bytes and ue8m0 exponents [npad/64, K/64, 64, 2]);
// act 0 none, 1 GELU tanh, 2 exact GELU; ``tile`` 0 (128 x 128, 8 warps, 3 stages), 4 (128 x 128, 4 warps, 2
// stages) or 5 (64 x 128, 2 warps, 2 stages) never changes a row's bits. n may pass the weight's own outputs up to
// npad: those columns' codes are zero, so with a zero bias they are written as exact zeros.
extern "C" int zs_nvfp4_prompt_bias(const zs_view* x_, const zs_view* w_, const zs_view* bs_, double scale,
                                    const zs_view* bias_, const zs_view* out_, int64_t mode, int64_t n, int64_t npad,
                                    int64_t tile, int64_t act) {
    zs::begin();
    const zs::View x(x_), w(w_), bs(bs_), bias(bias_);
    zs::View out(out_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.size(0) >= 1 && x.stride(1) == 1, "x: (M, K) bf16");
    ZS_REQUIRE(aligned16(x), "x rows must start on 16-byte boundaries");
    ZS_REQUIRE((mode == 0 || mode == 1 || mode == 2) && (tile == 0 || tile == 4 || tile == 5) && act >= 0 && act <= 2,
               "mode 0, 1 or 2, tile 0, 4 or 5, act 0-2");
    const int64_t m = x.size(0), k = x.size(1);
    ZS_REQUIRE(k % 64 == 0 && npad % 128 == 0 && n >= 1 && n <= npad, "K in groups of 64, n padded to 128");
    ZS_REQUIRE(m <= INT32_MAX && k <= INT32_MAX && npad <= INT32_MAX, "shape within int32");
    ZS_REQUIRE(w.defined() && w.numel() * w.element_size() == npad * k / (mode == 0 ? 2 : 1),
               "weight bytes do not match n and K");
    if (mode == 1) {
        ZS_REQUIRE(!bs.defined(), "FP8 takes no block scales");
    } else {
        ZS_REQUIRE(bs.defined() && bs.numel() == (k / 64) * npad * (mode == 0 ? 4 : 2), "block scales [npad/64, K/64, 64, 4|2]");
    }
    ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_BF16 && bias.is_contiguous() && bias.numel() == n, "bias: [n] bf16");
    ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.dim() == 2 && out.size(0) == m &&
                   out.size(1) == n,
               "out: (M, n) bf16");
    const int N = static_cast<int>(n), NP = static_cast<int>(npad), T = static_cast<int>(tile), A = static_cast<int>(act);
    if (mode == 0) by_act<FP4>(A, T, x, w, bs, scale, bias, out, N, NP);
    else if (mode == 1) by_act<FP8>(A, T, x, w, bs, scale, bias, out, N, NP);
    else by_act<MXFP8>(A, T, x, w, bs, scale, bias, out, N, NP);
    return zs::status();
}
