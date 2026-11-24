// What the text attention's two halves share (spec 138 WS-04, moved out of zig/kernels/qwen4/attention.cu as it
// was): the macros of its three objects (the contiguous caches, ZS_EXTENTS's table, ZS_ROWS's round) and the
// launchers' checks. qwen4/attention.cu (Flash Next's prep and gate) and common/attention.cu (the attention
// itself, which Qwen3-ASR's language model launches too) include it; attention_table.cu and attention_rows.cu
// include both, so it is read once an object.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "qmm_frag.cuh"
#include "attention_mma.cuh"
#include "../zs/host.h"

namespace {


// ZS_EXTENTS is spec 018's layout, the pool's (`--parallel` N) (attention.zig's `Cache.table`):
// attention_table.cu and attention_rows.cu define it and include the two halves, and without it they are the
// kernels they always were. With it the caches are a pool of extents of CHUNK positions, and a sequence's position p
// lies at the pool's row TABLE[p >> 9] * CHUNK + p % CHUNK (TABLE int32, an extent's place in the pool a 512
// positions of the sequence): one load a row in prep and one a chunk block on a dense row, whose keys are one
// extent's. A listed row's keys (the `*_table` entries): IDS holds the pool's rows, which the selection wrote
// through the table (qsa.cu under the same macro), so a listed row is read as the contiguous kernel reads it.
// Only where a key or a value is stored changes, never the order of a sum, so the bits are the contiguous caches'.
#ifdef ZS_EXTENTS
static_assert(CHUNK == 512, "an extent is a chunk");
constexpr int EXTENT_BITS = 9;
__device__ __forceinline__ int stored_at(const int* __restrict__ TABLE, int pos) {
  return (TABLE[pos >> EXTENT_BITS] << EXTENT_BITS) | (pos & (CHUNK - 1));
}
#define ZS_TABLE_PARAM const int* __restrict__ TABLE,
#else
#define ZS_TABLE_PARAM
#endif

// ZS_ROWS is a mixed round (spec 018 WS-B3), compiled only from attention_rows.cu, which also defines
// ZS_EXTENTS 2. Row r belongs to stream SID[r], sits at POS0[r], and its extent table is TABLES[SID[r]].
// Without the macro the kernels below are the ones they were: POS0 is one position and TABLE one sequence.

// A kernel's name says which of the three objects it is in (attn_prep_kernel, attn_prep_table_kernel,
// attn_prep_rows_kernel), a trace having only the name (zig/kernels/AGENTS.md, "The C ABI").
#if defined(ZS_ROWS)
#define ZS_NAME(stem) attn_##stem##_rows_kernel
#elif defined(ZS_EXTENTS)
#define ZS_NAME(stem) attn_##stem##_table_kernel
#else
#define ZS_NAME(stem) attn_##stem##_kernel
#endif


unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

bool is(const zs::View& v, int dtype, int dims) {
  return v.defined() && v.dtype() == dtype && v.dim() == dims && v.is_contiguous();
}

}  // namespace
