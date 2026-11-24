// Spec 018 WS-B4, the pool's (`--parallel` N). The selection for a captured round: one GEMM a slot, of the
// round's row count, because a graph cannot vary its launches, all in one zs_b16_gemm_entries launch. A slot
// with no rows returns from the pack and from its GEMM at once, both reading its count in SEG (spec 144).
// Scores and the list are qsa.cu's, a round row at a time: its position is posr[r], its queries' products are
// the packed rows the pack wrote for its slot, and a listed key is the pool's row through that slot's table.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"

namespace {

constexpr int SEL = 1024;
constexpr int EXTENT = 512, EXTENT_BITS = 9;

__device__ __forceinline__ int stored_at(const int* __restrict__ TABLE, int pos) {
    return (TABLE[pos >> EXTENT_BITS] << EXTENT_BITS) | (pos & (EXTENT - 1));
}

__device__ __forceinline__ uint32_t order_key(float v) {
    const uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Copies a live slot's indexer queries, [rows, heads, dim] at the segment's first row, into the packed
// region the slot's GEMM reads. An empty slot returns before a load.
__global__ void pack_queries_kernel(const int* __restrict__ SEG, const __nv_bfloat16* __restrict__ IQ,
                                    __nv_bfloat16* __restrict__ Q, int heads, int dim, int gem_rows) {
    const int slot = static_cast<int>(blockIdx.x);
    const int rows = SEG[slot * 4 + 1];
    if (rows <= 0) return;
    const int first = SEG[slot * 4];
    const int n = rows * heads * dim;
    const __nv_bfloat16* src = IQ + (size_t)first * heads * dim;
    __nv_bfloat16* dst = Q + (size_t)slot * gem_rows * heads * dim;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// Thread (round row, block). DOTS holds one packed [gem_rows * heads, nbl] region a slot; row r reads its
// slot's local row. The sum is qsa.cu's scores_kernel.
__global__ void scores_slots_kernel(const float* __restrict__ DOTS, int slot_stride, const int* __restrict__ SEG,
                                    const int* __restrict__ SID, const int* __restrict__ POSR, float* __restrict__ SC,
                                    int R, int HI, int NBL, int NB, int DI, int RATIO, int TOP) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)R * NBL) return;
    const int r = static_cast<int>(i / NBL), b = static_cast<int>(i % NBL);
    const int slot = SID[r];
    const int local = r - SEG[slot * 4];
    if (local < 0 || local >= SEG[slot * 4 + 1]) return;
    const int complete = (POSR[r] + 1) / RATIO;
    if (complete <= TOP || b >= complete) return;
    const float* dots = DOTS + (size_t)slot * slot_stride;
    float total = 0.f;
    for (int h = 0; h < HI; ++h) total = total + fmaxf(dots[((size_t)local * HI + h) * NBL + b], 0.f);
    SC[(size_t)r * NB + b] = total / sqrtf(static_cast<float>(DI));
}

// Block r of SEL threads. qsa.cu's select_kernel, with the row's own position and its slot's table.
__global__ void __launch_bounds__(SEL) select_slots_kernel(const float* __restrict__ SC, const int* __restrict__ SID,
                                                           const int* __restrict__ POSR, const int64_t* __restrict__ TABLES,
                                                           int* __restrict__ IDS, int* __restrict__ NK,
                                                           int* __restrict__ SPARSE, int NB, int RATIO, int TOP, int IDW) {
    __shared__ int hist[256];
    __shared__ int above[SEL];
    __shared__ int equal[SEL];
    __shared__ uint32_t cut_shared;
    __shared__ int need_shared;
    const int r = blockIdx.x, tid = threadIdx.x;
    const int end = POSR[r] + 1, complete = end / RATIO;
    if (complete <= TOP) {
        if (tid == 0) {
            NK[r] = end;
            SPARSE[r] = 0;
        }
        return;
    }
    const int* TABLE = reinterpret_cast<const int*>(TABLES[SID[r]]);
    const float* row = SC + (size_t)r * NB;
    if (tid == 0) {
        cut_shared = 0;
        need_shared = TOP;
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
    int* list = IDS + (size_t)r * IDW;
    for (int b = b0; b < b1; ++b) {
        const uint32_t key = order_key(row[b]);
        bool chosen = key > cut;
        if (key == cut) chosen = ++seen <= need;
        if (chosen) {
            const int first = stored_at(TABLE, b * RATIO);
            for (int k = 0; k < RATIO; ++k) list[placed * RATIO + k] = first + k;
            ++placed;
        }
    }
    if (tid == 0) {
        for (int t = 0; t < RATIO; ++t) {
            const int tail = RATIO * complete + t;
            if (tail < end) list[TOP * RATIO + t] = stored_at(TABLE, tail);
        }
        NK[r] = TOP * RATIO + end - RATIO * complete;
        SPARSE[r] = 1;
    }
}

unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

}  // namespace

extern "C" {

// iq bf16 holds the round's indexer queries, row-major [rows, heads, dim]. queries receives one packed region
// a slot, [gem_rows, heads, dim], gem_rows the round's row count. A slot with no rows returns at once.
int zs_qwen4_attn_pack_queries(const zs_view* seg_, const zs_view* iq_, const zs_view* queries_, int64_t heads,
                               int64_t dim, int64_t gem_rows) {
    zs::begin();
    const zs::View seg(seg_), iq(iq_), queries(queries_);
    ZS_REQUIRE(seg.defined() && seg.dtype() == ZS_I32 && seg.is_contiguous() && seg.dim() == 2 && seg.size(1) == 4,
               "seg: int32 [slots, 4]");
    const int64_t slots = seg.size(0);
    ZS_REQUIRE(iq.defined() && iq.dtype() == ZS_BF16 && iq.is_contiguous() && heads >= 1 && dim >= 1 && gem_rows >= 1,
               "iq: bf16 queries, one row heads * dim");
    ZS_REQUIRE(iq.numel() >= gem_rows * heads * dim, "iq holds the round's queries");
    ZS_REQUIRE(queries.defined() && queries.dtype() == ZS_BF16 && queries.is_contiguous() &&
                   queries.numel() >= slots * gem_rows * heads * dim,
               "queries: one [gem_rows, heads, dim] region a slot");
    ZS_REQUIRE(slots >= 1 && slots <= 65535, "slots");
    pack_queries_kernel<<<static_cast<unsigned>(slots), 256, 0, zs::stream()>>>(
        seg.data_ptr<int>(), iq.data_ptr<__nv_bfloat16>(), queries.data_ptr<__nv_bfloat16>(), static_cast<int>(heads),
        static_cast<int>(dim), static_cast<int>(gem_rows));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// dots is one packed region a slot, slot_stride floats apart, each region's products strided by its own nbl
// (dots' size(1) is not that stride: the region is wider than the GEMM's columns). The rest is qsa.cu's select,
// one round row a block, positions from posr and storage rows from the slot's table.
int zs_qwen4_attn_select_slots(const zs_view* dots_, int64_t slot_stride, const zs_view* seg_, const zs_view* sid_,
                               const zs_view* posr_, const zs_view* tables_, const zs_view* scores_, const zs_view* ids_,
                               const zs_view* nk_, const zs_view* sparse_, int64_t rows, int64_t heads, int64_t dim,
                               int64_t nbl, int64_t ratio, int64_t top) {
    zs::begin();
    const zs::View dots(dots_), seg(seg_), sid(sid_), posr(posr_), tables(tables_);
    const zs::View scores(scores_), ids(ids_), nk(nk_), sparse(sparse_);
    ZS_REQUIRE(dots.defined() && dots.dtype() == ZS_F32 && dots.is_contiguous() && heads >= 1 && rows >= 1 && nbl >= 1 &&
                   slot_stride >= rows * heads * nbl,
               "dots: fp32 regions, one a slot, holding [rows * heads, nbl]");
    ZS_REQUIRE(seg.defined() && seg.dtype() == ZS_I32 && seg.dim() == 2 && seg.size(1) == 4, "seg: int32 [slots, 4]");
    const int64_t slots = seg.size(0);
    ZS_REQUIRE(sid.defined() && sid.dtype() == ZS_I32 && sid.numel() >= rows && posr.defined() && posr.dtype() == ZS_I32 &&
                   posr.numel() >= rows,
               "sid, posr: int32 [rows]");
    ZS_REQUIRE(tables.defined() && tables.dtype() == ZS_I64 && tables.numel() >= slots, "tables: int64 [slots]");
    ZS_REQUIRE(scores.defined() && scores.dtype() == ZS_F32 && scores.dim() == 2 && scores.is_contiguous() &&
                   scores.size(0) >= rows && scores.size(1) >= nbl,
               "scores: fp32 [rows, NB]");
    ZS_REQUIRE(ids.defined() && ids.dtype() == ZS_I32 && ids.dim() == 2 && ids.is_contiguous() && ids.size(0) >= rows &&
                   ids.size(1) >= (top + 1) * ratio,
               "ids: int32 [rows, IDW]");
    ZS_REQUIRE(nk.defined() && nk.dtype() == ZS_I32 && nk.numel() >= rows && sparse.defined() && sparse.dtype() == ZS_I32 &&
                   sparse.numel() >= rows,
               "nk, sparse: int32 [rows]");
    ZS_REQUIRE(dim >= 1 && ratio >= 1 && top >= 1 && EXTENT % ratio == 0 && slots >= 1 &&
                   dots.numel() >= slots * slot_stride,
               "a listed block lies in one extent");
    const auto stream = zs::stream();
    scores_slots_kernel<<<blocks(rows * nbl), 256, 0, stream>>>(
        dots.data_ptr<float>(), static_cast<int>(slot_stride), seg.data_ptr<int>(), sid.data_ptr<int>(),
        posr.data_ptr<int>(), scores.data_ptr<float>(), static_cast<int>(rows), static_cast<int>(heads),
        static_cast<int>(nbl), static_cast<int>(scores.size(1)), static_cast<int>(dim), static_cast<int>(ratio),
        static_cast<int>(top));
    ZS_CUDA_LAUNCH_CHECK();
    select_slots_kernel<<<static_cast<unsigned>(rows), SEL, 0, stream>>>(
        scores.data_ptr<float>(), sid.data_ptr<int>(), posr.data_ptr<int>(), tables.data_ptr<int64_t>(),
        ids.data_ptr<int>(), nk.data_ptr<int>(), sparse.data_ptr<int>(), static_cast<int>(scores.size(1)),
        static_cast<int>(ratio), static_cast<int>(top), static_cast<int>(ids.size(1)));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
