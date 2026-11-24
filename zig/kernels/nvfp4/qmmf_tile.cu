// zspark's own (spec 070, WS-08), after qmmf.cu's FP8G lane matmul (TensorFold 0.6.5, 609ca41, Apache-2.0), which
// stays as it is beside this file. out (M, N) = x (M, K) bf16 @ W for 1 to 16 rows of a block-FP8 weight whose
// 64-column tiles each have ONE scale a group of 64 inputs (a checkpoint's 128x128 blocks always do).
//
// What differs from qmmf_kernel<FP8G>, and why:
// - The scale. qmmf reads an fp32 a column and group, 256 B beside a tile's 4,096 B of codes: 5.9% of a launch's
//   bytes, and of its time, since the kernel runs at the memory's rate (190.5 us without them 178.8 at
//   16,384 x 2,560). Here it is one fp32 a (tile, group), [npad/64, K/64], asked for with the stage (4 B by
//   cp.async). The arithmetic is qmmf's: acc = fma(a group's products, its scale, acc) in group order.
// - The K slices. qmmf sums them through a cluster of SK blocks, which costs time as the cluster grows
//   (2,560 x 6,144: 73.5 us at 2 slices, 77.9 at 4, 86.2 at 8, the read-only kernel 72.2). `fuse`: one block runs
//   its tile's slices in turn and adds them in slice order, qmmf's own FUSE (its prompt rows') on the 16-row
//   tile. The host asks for it from 40 tiles (`nvfp4.fused_tiles`): below, the blocks are too few for the 48 SMs
//   (20 tiles: 20.3 us fused against 19.6 clustered; 6 tiles: 56 against 25.7) and the cluster is kept.
// Bits: qmmf_kernel<FP8G>'s for the same weight, element for element; the row rule as there (a row is an mma row,
// K slices by shape). `nvfp4_test.zig` holds both, with a row alone against the row in a window.
//
// Measured standalone, the kernels' own durations (nsys), medians, 1 to 16 rows alike, 24+ copies of a weight in
// turn, spark-b alone, us. Read-only: WS-01's read kernel (128 threads, its best grid) over the bytes qmmf reads.
// The whole table, with a round's launches: `.agents/skills/cuda-gb10/references/fp8-linears.md`.
//   N x K, tiles x slices (linear)                       qmmf   this   read-only
//   16,384 x 2,560, 256 x 1 (DeltaNet qkv, z)           192.4  181.1     206.5
//   13,312 x 2,560, 208 x 1 (attention q, k, v)         155.7  147.7     160.0
//    2,560 x 6,144,  40 x 8 (out_proj, o_proj; fused)    86.4   67.7      72.2
//    2,560 x 2,560,  40 x 4 (the draft head's fc; fused) 37.3   29.3      32.6
//      324 x 10,240,  6 x 8 (read-out down; cluster)     25.7   24.5      19.0
//   10,240 x 320,   160 x 1 (read-out up)                17.2   16.1      17.2
//    1,280 x 2,560,  20 x 4 (shared gate, up; cluster)   19.5   18.5      17.3
//    2,560 x 640,    40 x 1 (shared down)                 9.3    8.65      9.2
//  248,320 x 2,560, 3880 x 1 (lm_head)                  2,871  2,708     2,927
//
// Shared memory (the SM's 102,400 B for all its blocks; a block is 4 warps):
//   region                  bytes a warp   a block (4 stages)   an SM (3 blocks)   what it is
//   codes, a stage 4,096       1,024          16,384               49,152          weight on its way
//   x, a stage 2,048             512           8,192               24,576          16 rows of 64 inputs, ldmatrix's
//   scale, a stage 4 (+124)       32             512                1,536          one fp32, the line's rest unused
//   in all                     1,568          25,088               75,264          + 1,024 reserved a block
//   The cluster's partial sums (4,096 B) reuse the stages once the loop is over, as in qmmf.
// What could still be less: x is staged for 16 rows whatever the window (rows past M are zeros, not read): a
// 1-row launch could keep 128 B a stage and a fifth stage of codes. Not built: a stage of 6,144 B without the
// scales, four blocks an SM instead of three, was timed and gained nothing (179.3 against 178.8 us), so the
// weight on its way an SM is not what bounds this kernel; the bytes are.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../common/qmm_frag.cuh"
#include "../zs/host.h"
#include "codes.cuh"

namespace {

using namespace nvfp4_codes;

using namespace qmm_frag;

constexpr int GS = 64;                                    // inputs a pipeline stage
constexpr int BM = 16, BN = 64, WN = 4, STAGES = 4;       // qmmf's decode tile: 16 rows by 64 columns, 4 warps
constexpr int THREADS = WN * 32;
constexpr int NT = BN / WN / 8;                           // n8 tiles a warp
constexpr int ROW = GS * 2, CHUNKS = ROW / 16;
constexpr int X = BM * ROW, W = BN * GS;
constexpr int STAGE = (X + W + 4 + 127) / 128 * 128;      // on 128-byte lines, as qmmf's
constexpr int PARTIALS = NT * 4 * THREADS * 4;
constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }

// A block is one 64-column tile (blockIdx.x) and, unless FUSE, one K slice (blockIdx.z). w: e4m3 bytes in qmmf's
// fragment order [npad/64][K/64][8][32][2][8]; ts: fp32 [npad/64][K/64].
template <bool F32, bool CLUSTER, bool FUSE>
__global__ void __launch_bounds__(THREADS) qmmt_kernel(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const float* __restrict__ ts,
        void* __restrict__ out, int M, int N, int K, int SK, int ldx) {
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / GS, slice_groups = KG / SK, per = FUSE ? KG : slice_groups;
    const int tile = blockIdx.x, n0 = tile * BN, slice = FUSE ? 0 : blockIdx.z, g0 = slice * per;

    auto stage = [&](int s) { return buf + s * STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        {                                                  // a thread a 16-byte chunk of x: 16 rows by 8 chunks
            const int r = tid / CHUNKS, ch = tid % CHUNKS;
            const int row = min(r, M - 1);
            cp16z(p + r * ROW + swz<CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8, r < M);
        }
        unsigned char* pw = p + X;
        const unsigned char* src = w + (static_cast<size_t>(tile) * KG + g) * W;
        for (int c = tid; c < W / 16; c += THREADS) cp16(pw + c * 16, src + c * 16);
        if (tid == 0) cp4(pw + W, ts + static_cast<size_t>(tile) * KG + g);
    };

    float acc[NT][4];
    float tot[NT][4];                                      // FUSE: the slices' sum so far, in slice order
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f, tot[j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, g0 + s);
        commit();
    }
    for (int it = 0; it < per; ++it) {
        if constexpr (FUSE) {
            if (it > 0 && it % slice_groups == 0) {        // a slice ends: add it to the sum, start the next from zero
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[j][e] = it == slice_groups ? acc[j][e] : tot[j][e] + acc[j][e];
                        acc[j][e] = 0.0f;
                    }
            }
        }
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, g0 + next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const unsigned char* pw = p + X;
        const float scale = *reinterpret_cast<const float*>(pw + W);   // the tile's, for this group
        float d[NT][4];
        uint4 wq[NT];                                      // a lane's 16 bytes for the stage, read once
#pragma unroll
        for (int j = 0; j < NT; ++j) wq[j] = reinterpret_cast<const uint4*>(pw)[(wn * NT + j) * 32 + lane];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[4];
            {
                const int r = (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a, p + r * ROW + swz<CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint32_t word = comp(wq[j], kt);
                const uint32_t b0 = fp8pair(word & 0xFFFFu), b1 = fp8pair(word >> 16);
                if (kt == 0) mma0(d[j], a, b0, b1);
                else mma(d[j], a, b0, b1);
            }
        }
        // the group's products, scaled into acc in group order: qmmf's FP8G step with one scale for the tile
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[j][e] = __fmaf_rn(d[j][e], scale, acc[j][e]);
    }
    wait<0>();
    __syncthreads();
    if constexpr (FUSE) {                                  // the last slice; one slice: acc is the sum already
        if (SK > 1) {
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[j][e] = tot[j][e] + acc[j][e];
        }
    }
    if constexpr (CLUSTER) {                               // qmmf's: slice 0 adds its peers' sums in slice order
        auto cluster = cooperative_groups::this_cluster();
        float* mine = reinterpret_cast<float*>(buf);
        if (slice != 0) {
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) mine[(j * 4 + e) * THREADS + tid] = acc[j][e];
        }
        cluster.sync();
        if (slice == 0) {
            for (int peer = 1; peer < SK; ++peer) {
                const float* theirs = cluster.map_shared_rank(mine, peer);
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[j][e] = acc[j][e] + theirs[(j * 4 + e) * THREADS + tid];
            }
        }
        cluster.sync();
        if (slice != 0) return;
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = (lane >> 2) + h * 8;
            if (row >= M) continue;
            const float v0 = acc[j][2 * h], v1 = acc[j][2 * h + 1];   // qmmf multiplies by the tensor's scale, 1 here
            if (F32) {
                float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                if (col < N) dst[0] = v0;
                if (col + 1 < N) dst[1] = v1;
            } else {
                __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0)
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                else {
                    if (col < N) dst[0] = __float2bfloat16_rn(v0);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                }
            }
        }
    }
}

template <bool F32, bool CLUSTER, bool FUSE>
void launch(const zs::View& x, const zs::View& w, const zs::View& ts, zs::View& out, int N, int K, int SK) {
    const int M = x.size(0);
    auto kernel = qmmt_kernel<F32, CLUSTER, FUSE>;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3((N + BN - 1) / BN, 1, FUSE ? 1 : SK);
    config.blockDim = dim3(THREADS);
    config.dynamicSmemBytes = SMEM;                        // 25,088 B: under the 49,152 a block has without asking
    config.stream = zs::stream();
    cudaLaunchAttribute attr[1];
    if (CLUSTER) {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 1;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = SK;
        config.attrs = attr;
        config.numAttrs = 1;
    }
    ZS_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const unsigned char*>(w.data_ptr()), ts.data_ptr<float>(), out.data_ptr(), M, N, K, SK,
        M == 1 ? K : static_cast<int>(x.stride(0))));
}

template <bool F32>
void by_slices(bool cluster, bool fuse, const zs::View& x, const zs::View& w, const zs::View& ts, zs::View& out,
               int N, int K, int SK) {
    if (fuse) launch<F32, false, true>(x, w, ts, out, N, K, SK);
    else if (cluster) launch<F32, true, false>(x, w, ts, out, N, K, SK);
    else launch<F32, false, false>(x, w, ts, out, N, K, SK);
}

}  // namespace

extern "C" {

// out (M, n) = x (M, K) bf16 @ W, 1 to 16 rows: w e4m3 in qmmf's fragment order (npad x K bytes), ts fp32
// [npad/64, K/64], a 64-column tile's one scale a group of 64 inputs. sk K slices, summed in slice order by a
// cluster or, fuse != 0, inside each block: the same bits, and zs_nvfp4_qmmf's mode 3 for column scales that
// repeat ts.
int zs_nvfp4_qmmf_tile(const zs_view* x_, const zs_view* w_, const zs_view* ts_, const zs_view* out_, int64_t n,
                       int64_t sk, int64_t npad, int fuse, int f32) {
    zs::begin();
    const zs::View x(x_), w(w_), ts(ts_);
    zs::View out(out_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2, "x: (M, K) bf16");
    const int64_t m = x.size(0), k = x.size(1);
    ZS_REQUIRE(m >= 1 && m <= BM, "1 to 16 rows");
    ZS_REQUIRE(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (m == 1 || x.stride(0) % 8 == 0),
               "x rows must start on 16-byte boundaries");
    ZS_REQUIRE(sk >= 1 && sk <= 8 && k % 64 == 0 && (k / 64) % sk == 0, "K in whole groups of 64, split evenly in 1 to 8");
    ZS_REQUIRE(n >= 1 && npad % 64 == 0 && n <= npad, "n within npad, a multiple of 64");
    ZS_REQUIRE(w.defined() && w.numel() * w.element_size() == npad * k, "weight bytes do not match npad and K");
    ZS_REQUIRE(ts.defined() && ts.dtype() == ZS_F32 && ts.numel() == (npad / 64) * (k / 64), "ts: [npad/64, K/64] fp32");
    ZS_REQUIRE(out.defined() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
               out.dtype() == (f32 ? ZS_F32 : ZS_BF16), "out: (M, n)");
    const bool fused = fuse != 0 && sk > 1, cluster = !fused && sk > 1;
    ZS_REQUIRE(!cluster || zs::device().major >= 9, "K slices by a cluster need sm_90 or later");
    const int N = static_cast<int>(n), K = static_cast<int>(k), SK = static_cast<int>(sk);
    if (f32) by_slices<true>(cluster, fused, x, w, ts, out, N, K, SK);
    else by_slices<false>(cluster, fused, x, w, ts, out, N, K, SK);
    return zs::status();
}

}  // extern "C"
