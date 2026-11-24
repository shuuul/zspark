// Our own kernels for Nemotron's weight layouts and the serial feed (torch-op replacements live in torch_ops/).
// From TensorFold 1.1.5 (792cdbab, unchanged at 1b31cc74) zig/kernels/cuda/nemotron_ops.cu, Apache-2.0;
// changed for zspark: the zs launchers at the file's end, and tf_pack_dense and tf_transpose_pad16 dropped —
// they repack MLX 4-bit weights into the lane matmul's tiles, a storage the NVFP4 checkpoint does not have
// (zspark's own NVFP4/FP8 linears read the checkpoint's). The kept kernels' device code is upstream's byte for byte.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"

// A serial round's end on the device: its token becomes the next window's input; pos += 1, parity flips, keep 1.
extern "C" __global__ void tf_serial_feed(const int* __restrict__ sampled, int* __restrict__ ids, int* __restrict__ meta,
                                          int* __restrict__ history) {
    const int tok = sampled[0];
    const int pos = meta[0];
    ids[0] = tok;
    history[pos + 1] = tok;
    meta[0] = pos + 1;
    meta[1] = meta[1] ^ 1;
    meta[2] = 1;
}

namespace {

// Exclusive prefix sum over a 128-thread block (four warps); `total` gets the sum.
__device__ __forceinline__ int scan128(int v, int* total) {
    __shared__ int ws[4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int s = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, s, o);
        if (lane >= o) s += u;
    }
    if (lane == 31) ws[warp] = s;
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += ws[w];
    *total = ws[0] + ws[1] + ws[2] + ws[3];
    __syncthreads();
    return before + s - v;
}

}  // namespace

// Routed slots retain plan_kernel item and member order for experts below E.
extern "C" __global__ void __launch_bounds__(128) tf_plan_routed(const int* __restrict__ picks, int rows, int slots, int routed,
                                                                int E, int T, int* __restrict__ members,
                                                                int* __restrict__ items, int* __restrict__ counts) {
    __shared__ int pk[16 * 8];
    const int tid = threadIdx.x, P = rows * slots;
    for (int p = tid; p < P; p += blockDim.x) pk[p] = p % slots < routed ? picks[p] : -1;
    __syncthreads();
    int c = 0;
    if (tid < E)
        for (int p = 0; p < P; ++p) c += pk[p] == tid;
    const int tiles = (c + T - 1) / T;
    int nmembers, ntiles, nused;
    const int off = scan128(c, &nmembers);
    const int ioff = scan128(tiles, &ntiles);
    scan128(c > 0, &nused);
    if (tid == 0) {
        counts[0] = ntiles;
        counts[1] = nused;
    }
    if (tid >= E) return;
    for (int j = 0; j < tiles; ++j) {
        int* it = items + 3 * (ioff + j);
        it[0] = tid;
        it[1] = off + T * j;
        it[2] = min(T, c - T * j);
    }
    int k = off;
    for (int p = 0; p < P && k < off + c; ++p)
        if (pk[p] == tid) members[k++] = p;
}

// y[row] += R x[row] at rows r * row_mul + row_add (r < rows): a lesson's change past the 4-bit codes.
extern "C" __global__ void __launch_bounds__(128) tf_rest_rows(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                               const __nv_bfloat16* __restrict__ r, void* __restrict__ y,
                                                               int y_stride, int y_f32, int rows, int row_mul, int row_add,
                                                               int n, int k) {
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (j >= n) return;
    const __nv_bfloat16* rj = r + static_cast<size_t>(j) * k;
    for (int r0 = 0; r0 < rows; r0 += 16) {
        const int tile = min(16, rows - r0);
        float acc[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) acc[t] = 0.f;
        for (int i = lane * 8; i < k; i += 256) {
            const uint4 wv = *reinterpret_cast<const uint4*>(rj + i);
            const __nv_bfloat16* w8 = reinterpret_cast<const __nv_bfloat16*>(&wv);
            float wf[8];
#pragma unroll
            for (int q = 0; q < 8; ++q) wf[q] = __bfloat162float(w8[q]);
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                if (t < tile) {
                    const size_t row = static_cast<size_t>(r0 + t) * row_mul + row_add;
                    const uint4 xv = *reinterpret_cast<const uint4*>(x + row * x_stride + i);
                    const __nv_bfloat16* x8 = reinterpret_cast<const __nv_bfloat16*>(&xv);
                    float s = 0.f;
#pragma unroll
                    for (int q = 0; q < 8; ++q) s += wf[q] * __bfloat162float(x8[q]);
                    acc[t] += s;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (t < tile) {
                float v = acc[t];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) {
                    const size_t row = static_cast<size_t>(r0 + t) * row_mul + row_add;
                    if (y_f32) {
                        float* yp = static_cast<float*>(y) + row * y_stride + j;
                        *yp += v;
                    } else {
                        __nv_bfloat16* yp = static_cast<__nv_bfloat16*>(y) + row * y_stride + j;
                        *yp = __float2bfloat16(__bfloat162float(*yp) + v);
                    }
                }
            }
        }
    }
}

extern "C" {

// sampled int32 [1], meta int32 [3] ({pos, parity, keep}, read and written on the device), history int32 (the
// tokens' record, written at pos + 1) -> ids int32 [1]: one serial round's end on the device — its token
// becomes the next window's input, pos += 1, parity flips, keep 1.
int zs_nemotron_serial_feed(const zs_view* sampled_, const zs_view* ids_, const zs_view* meta_,
                            const zs_view* history_) {
    zs::begin();
    const zs::View sampled(sampled_), ids(ids_), meta(meta_), history(history_);
    ZS_REQUIRE(sampled.defined() && sampled.dtype() == ZS_I32 && sampled.numel() >= 1, "sampled: int32 [1]");
    ZS_REQUIRE(ids.defined() && ids.dtype() == ZS_I32 && ids.numel() >= 1, "ids: int32 [1]");
    ZS_REQUIRE(meta.defined() && meta.dtype() == ZS_I32 && meta.numel() >= 3, "meta: int32 [3]");
    ZS_REQUIRE(history.defined() && history.dtype() == ZS_I32 && history.numel() >= 1, "history: int32 [N]");
    tf_serial_feed<<<1, 1, 0, zs::stream()>>>(sampled.data_ptr<int>(), ids.data_ptr<int>(), meta.data_ptr<int>(),
                                              history.data_ptr<int>());
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// picks int32 [rows * slots] (the routed slot ids; slots after `routed` are the shared halves' and are skipped)
// -> members int32 (the pick indices grouped by expert, member order kept), items int32 [3 * (experts +
// pairs / tile)] triples {expert, first, count} of at most `tile` members, counts int32 [2] {items, experts
// used}. One block of 128: rows up to 16, slots up to 8, experts up to 128 (its shared table and its prefix sums).
int zs_nemotron_plan_routed(const zs_view* picks_, const zs_view* members_, const zs_view* items_,
                            const zs_view* counts_, int64_t rows, int64_t slots, int64_t routed, int64_t e,
                            int64_t tile) {
    zs::begin();
    const zs::View picks(picks_), members(members_), items(items_), counts(counts_);
    ZS_REQUIRE(rows >= 0 && rows <= 16 && slots >= 1 && slots <= 8 && routed >= 0 && routed <= slots,
               "rows, slots, routed: 16 rows of 8 slots, the routed first");
    ZS_REQUIRE(e >= 0 && e <= 128 && tile >= 1, "experts up to 128, a tile of at least 1");
    const int64_t p = rows * slots;
    ZS_REQUIRE(picks.defined() && picks.dtype() == ZS_I32 && picks.is_contiguous() && picks.numel() >= p,
               "picks: int32 [rows * slots]");
    ZS_REQUIRE(members.defined() && members.dtype() == ZS_I32 && members.is_contiguous() &&
                   members.numel() >= rows * routed,
               "members: int32 [rows * routed]");
    // an expert's members make at most ceil(P / tile) items, so E + ceil(P / tile) covers every plan
    ZS_REQUIRE(items.defined() && items.dtype() == ZS_I32 && items.is_contiguous() &&
                   items.numel() >= 3 * (e + (p + tile - 1) / tile),
               "items: int32 [3 * (experts + pairs / tile)]");
    ZS_REQUIRE(counts.defined() && counts.dtype() == ZS_I32 && counts.numel() >= 2, "counts: int32 [2]");
    tf_plan_routed<<<1, 128, 0, zs::stream()>>>(picks.data_ptr<int>(), static_cast<int>(rows),
                                                static_cast<int>(slots), static_cast<int>(routed),
                                                static_cast<int>(e), static_cast<int>(tile),
                                                members.data_ptr<int>(), items.data_ptr<int>(),
                                                counts.data_ptr<int>());
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// y[row * row_mul + row_add] += r @ x[row] for row < rows, a warp a column j: r bf16 [N, K], x bf16 rows of
// K at x_stride elements (a 2-D view's row stride), y fp32 or bf16 at y_stride. The reads are 16 bytes:
// K and both strides are multiples of 8, the bases 16-byte aligned.
int zs_nemotron_rest_rows(const zs_view* x_, const zs_view* r_, const zs_view* y_, int64_t rows, int64_t row_mul,
                          int64_t row_add) {
    zs::begin();
    const zs::View x(x_), r(r_), y(y_);
    ZS_REQUIRE(r.defined() && r.dtype() == ZS_BF16 && r.dim() == 2 && r.is_contiguous(), "r: bf16 [N, K]");
    const int64_t n = r.size(0), k = r.size(1);
    ZS_REQUIRE(k % 8 == 0 && k <= INT32_MAX && n <= INT32_MAX, "K: a multiple of 8");
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16, "x: bf16");
    const int64_t x_stride = x.dim() == 2 ? x.row_stride() : k;
    ZS_REQUIRE(y.defined() && (y.dtype() == ZS_F32 || y.dtype() == ZS_BF16), "y: fp32 or bf16");
    const int64_t y_stride = y.dim() == 2 ? y.row_stride() : n;
    ZS_REQUIRE(rows >= 0 && row_mul >= 0 && row_add >= 0, "rows, row_mul, row_add: nonnegative");
    if (rows > 0) {
        const int64_t last = (rows - 1) * row_mul + row_add;
        ZS_REQUIRE(x_stride % 8 == 0 && x.numel() >= last * x_stride + k, "x: bf16 rows of K at the view's stride");
        ZS_REQUIRE(y.numel() >= last * y_stride + n, "y: rows of N at the view's stride");
    }
    // 16-byte vector reads
    ZS_REQUIRE(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(r.data_ptr()) % 16 == 0,
               "x, r: 16-byte aligned");
    if (rows > 0 && n > 0)
        tf_rest_rows<<<static_cast<unsigned>((n + 3) / 4), 128, 0, zs::stream()>>>(
            x.data_ptr<__nv_bfloat16>(), static_cast<int>(x_stride), r.data_ptr<__nv_bfloat16>(), y.data_ptr(),
            static_cast<int>(y_stride), y.dtype() == ZS_F32 ? 1 : 0, static_cast<int>(rows),
            static_cast<int>(row_mul), static_cast<int>(row_add), static_cast<int>(n), static_cast<int>(k));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
