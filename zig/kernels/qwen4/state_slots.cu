// Spec 018 WS-B3, the pool's (`--parallel` N): state.cu's window shift for every slot in one launch. A slot with
// no rows returns at once. Each slot's window is its own; fresh is the round's rows, and a slot reads the
// rows of its segment. seg int32 [slots, 4] is first row, rows, kept, first position. The window pointer is
// ptrs int64 [slots, stride] at column col.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"

namespace {

constexpr int MAX_TAPS = 32;

__global__ void shift_slots_kernel(const int* __restrict__ SEG, const int64_t* __restrict__ PTRS, int stride,
                                   int col, const __nv_bfloat16* __restrict__ fresh, int fresh_row, int T, int C) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int slots = static_cast<int>(gridDim.y);
    // the grid is (blocks, slots): blockIdx.y is the slot, so a slot's threads share nothing with another's
    const int slot = static_cast<int>(blockIdx.y);
    const int c = static_cast<int>(i);
    if (c >= C || slot >= slots) return;
    const int rows = SEG[slot * 4 + 1];
    if (rows <= 0) return;
    const int first = SEG[slot * 4];
    const int keep = SEG[slot * 4 + 2];
    __nv_bfloat16* window = reinterpret_cast<__nv_bfloat16*>(PTRS[static_cast<size_t>(slot) * stride + col]) + c;
    __nv_bfloat16 held[MAX_TAPS];
    for (int j = 0; j < T; ++j) {
        const int src = keep + j;
        held[j] = src < T ? window[(size_t)src * C]
                          : fresh[(static_cast<size_t>(first + src - T)) * fresh_row + c];
    }
    for (int j = 0; j < T; ++j) window[(size_t)j * C] = held[j];
}

// `shift_slots_kernel` for `count` layers at once (spec 144 WS-09, as state.cu's `shift_each` for a session, spec 073
// WS-07): blockIdx.z is the layer, its window in ptrs at column col0 + z * col_step and its new rows at fresh[z].
// The same copies: the same bits.
constexpr int MAX_LAYERS = 64;
struct Freshes {
    const __nv_bfloat16* fresh[MAX_LAYERS];
};

__global__ void shift_slots_each_kernel(const int* __restrict__ SEG, const int64_t* __restrict__ PTRS, int stride,
                                        int col0, int col_step, const Freshes freshes, int fresh_row, int T, int C) {
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int slot = static_cast<int>(blockIdx.y);
    const int layer = static_cast<int>(blockIdx.z);
    if (c >= C) return;
    const int rows = SEG[slot * 4 + 1];
    if (rows <= 0) return;
    const int first = SEG[slot * 4];
    const int keep = SEG[slot * 4 + 2];
    const int col = col0 + layer * col_step;
    __nv_bfloat16* window = reinterpret_cast<__nv_bfloat16*>(PTRS[static_cast<size_t>(slot) * stride + col]) + c;
    const __nv_bfloat16* fresh = freshes.fresh[layer];
    __nv_bfloat16 held[MAX_TAPS];
    for (int j = 0; j < T; ++j) {
        const int src = keep + j;
        held[j] = src < T ? window[(size_t)src * C]
                          : fresh[(static_cast<size_t>(first + src - T)) * fresh_row + c];
    }
    for (int j = 0; j < T; ++j) window[(size_t)j * C] = held[j];
}

}  // namespace

// `zs_qwen4_shift_slots` of `count` layers in one launch: layer i's window at ptrs column col0 + i * col_step,
// its new rows freshes[i] (bf16, row r at r * fresh_row). One shape for all.
extern "C" int zs_qwen4_shift_slots_each(const zs_view* seg_, const zs_view* ptrs_, int64_t stride, int64_t col0,
                                         int64_t col_step, const zs_view* freshes, int64_t count, int64_t fresh_row,
                                         int64_t taps, int64_t channels) {
    zs::begin();
    const zs::View seg(seg_), ptrs(ptrs_);
    ZS_REQUIRE(seg.defined() && seg.dtype() == ZS_I32 && seg.is_contiguous() && seg.dim() == 2 && seg.size(1) == 4,
               "seg: int32 [slots, 4]");
    const int64_t slots = seg.size(0);
    ZS_REQUIRE(count >= 1 && count <= MAX_LAYERS && freshes != nullptr, "1 to 64 layers");
    ZS_REQUIRE(ptrs.defined() && ptrs.dtype() == ZS_I64 && ptrs.is_contiguous() && ptrs.dim() == 2 &&
                   ptrs.size(0) == slots && ptrs.size(1) == stride && col0 >= 0 && col_step >= 0 &&
                   col0 + (count - 1) * col_step < stride,
               "ptrs: int64 [slots, stride]");
    ZS_REQUIRE(taps >= 1 && taps <= MAX_TAPS && channels >= 1 && fresh_row >= channels, "1 to 32 window rows");
    ZS_REQUIRE(slots >= 1 && slots <= 65535, "slots");
    Freshes table;
    for (int64_t i = 0; i < count; ++i) {
        const zs::View fresh(freshes + i);
        ZS_REQUIRE(fresh.defined() && fresh.dtype() == ZS_BF16, "fresh: bf16 rows");
        table.fresh[i] = fresh.data_ptr<__nv_bfloat16>();
    }
    const unsigned blocks = static_cast<unsigned>((channels + 255) / 256);
    shift_slots_each_kernel<<<dim3(blocks, static_cast<unsigned>(slots), static_cast<unsigned>(count)), 256, 0,
                              zs::stream()>>>(
        seg.data_ptr<int>(), ptrs.data_ptr<int64_t>(), static_cast<int>(stride), static_cast<int>(col0),
        static_cast<int>(col_step), table, static_cast<int>(fresh_row), static_cast<int>(taps),
        static_cast<int>(channels));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

// One shift of every slot's window. old is not one buffer: each slot's is the pointer in ptrs. fresh bf16
// holds the round's new rows, row r at r * fresh_row, its first channels the window's. `keep` is seg's.
extern "C" int zs_qwen4_shift_slots(const zs_view* seg_, const zs_view* ptrs_, int64_t stride, int64_t col,
                                    const zs_view* fresh_, int64_t fresh_row, int64_t taps, int64_t channels) {
    zs::begin();
    const zs::View seg(seg_), ptrs(ptrs_), fresh(fresh_);
    ZS_REQUIRE(seg.defined() && seg.dtype() == ZS_I32 && seg.is_contiguous() && seg.dim() == 2 && seg.size(1) == 4,
               "seg: int32 [slots, 4]");
    const int64_t slots = seg.size(0);
    ZS_REQUIRE(ptrs.defined() && ptrs.dtype() == ZS_I64 && ptrs.is_contiguous() && ptrs.dim() == 2 &&
               ptrs.size(0) == slots && ptrs.size(1) == stride && col >= 0 && col < stride,
               "ptrs: int64 [slots, stride]");
    ZS_REQUIRE(fresh.defined() && fresh.dtype() == ZS_BF16 && fresh_row >= channels, "fresh: bf16 rows");
    ZS_REQUIRE(taps >= 1 && taps <= MAX_TAPS && channels >= 1, "1 to 32 window rows");
    ZS_REQUIRE(slots >= 1 && slots <= 65535, "slots");
    const unsigned blocks = static_cast<unsigned>((channels + 255) / 256);
    shift_slots_kernel<<<dim3(blocks, static_cast<unsigned>(slots)), 256, 0, zs::stream()>>>(
        seg.data_ptr<int>(), ptrs.data_ptr<int64_t>(), static_cast<int>(stride), static_cast<int>(col),
        fresh.data_ptr<__nv_bfloat16>(), static_cast<int>(fresh_row), static_cast<int>(taps),
        static_cast<int>(channels));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}
