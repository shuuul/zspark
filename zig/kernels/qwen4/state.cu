// Sliding windows a committed sequence keeps: the DeltaNet conv's last inputs and the PLE conv's tail.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/forward.py (_shift_windows, Triton),
// Apache-2.0; rewritten for zspark in CUDA C++. Hand-written because it is a copy, not arithmetic.

#include <cuda_bf16.h>
#include <stdint.h>

#include "../zs/host.h"

namespace {

constexpr int MAX_TAPS = 32;

// Thread (layer, channel): window rows j < T become rows keep + j of [old (T rows); new rows], the old rows read
// before any is written.
__global__ void shift_kernel(__nv_bfloat16* __restrict__ old, const __nv_bfloat16* __restrict__ fresh, int keep,
                             int layers, int T, int C, long long fresh_layer, long long fresh_row) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)layers * C) return;
  const int l = static_cast<int>(i / C), c = static_cast<int>(i % C);
  __nv_bfloat16* window = old + (size_t)l * T * C + c;
  __nv_bfloat16 rows[MAX_TAPS];
  for (int j = 0; j < T; ++j) {
    const int src = keep + j;
    rows[j] = src < T ? window[(size_t)src * C] : fresh[l * fresh_layer + (long long)(src - T) * fresh_row + c];
  }
  for (int j = 0; j < T; ++j) window[(size_t)j * C] = rows[j];
}

// Several windows in one launch (spec 073 WS-07), `shift_kernel`'s copy for each: block (channels, window), the
// windows' and their new rows' addresses passed by value. What ExLlamaV3 does for its state rewinds
// (exllamav3_ext/gdn.cu at 151539c, `batched_conv_rewind_kernel`: a batch of jobs by value, a job a row of the
// grid; MIT, Copyright (c) 2025 Turboderp); the copy is this file's. A commit's 36 launches of 1.7 us become one
// of 22 with the windows out of L2, 10 to 12 with them in it (spark-b, a trace). No shared memory.
constexpr int MAX_WINDOWS = 64;
struct Windows {
  __nv_bfloat16* old[MAX_WINDOWS];
  const __nv_bfloat16* fresh[MAX_WINDOWS];
};

__global__ void shift_each_kernel(const Windows windows, int keep, int T, int C, long long fresh_row) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  __nv_bfloat16* window = windows.old[blockIdx.y] + c;
  const __nv_bfloat16* fresh = windows.fresh[blockIdx.y] + c;
  __nv_bfloat16 rows[MAX_TAPS];
  for (int j = 0; j < T; ++j) {
    const int src = keep + j;
    rows[j] = src < T ? window[(size_t)src * C] : fresh[(long long)(src - T) * fresh_row];
  }
  for (int j = 0; j < T; ++j) window[(size_t)j * C] = rows[j];
}

}  // namespace

// old bf16 [layers, taps, channels] in place; fresh bf16 holds each layer's new rows, layer ``l`` row ``r`` at
// l * fresh_layer + r * fresh_row elements, its first ``channels`` values the window's. ``keep`` rows are kept.
extern "C" int zs_qwen4_shift_windows(const zs_view* old_, const zs_view* fresh_, int64_t keep, int64_t fresh_layer,
                                   int64_t fresh_row, int64_t fresh_rows) {
  zs::begin();
  const zs::View old(old_), fresh(fresh_);
  ZS_REQUIRE(old.defined() && old.dtype() == ZS_BF16 && old.dim() == 3, "old: bf16 [layers, taps, channels]");
  const int64_t layers = old.size(0), taps = old.size(1), channels = old.size(2);
  ZS_REQUIRE(taps >= 1 && taps <= MAX_TAPS, "1 to 32 window rows");
  ZS_REQUIRE(fresh.defined() && fresh.dtype() == ZS_BF16 && fresh_row >= channels, "fresh: bf16 rows of the window");
  ZS_REQUIRE(keep >= 1 && keep <= fresh_rows, "keep 1 to the new rows");
  ZS_REQUIRE(fresh.numel() >= (layers - 1) * fresh_layer + (fresh_rows - 1) * fresh_row + channels,
             "fresh does not hold the rows read");
  const long long total = layers * channels;
  shift_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, zs::stream()>>>(
      old.data_ptr<__nv_bfloat16>(), fresh.data_ptr<__nv_bfloat16>(), static_cast<int>(keep),
      static_cast<int>(layers), static_cast<int>(taps), static_cast<int>(channels), fresh_layer, fresh_row);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// `zs_qwen4_shift_windows` for `count` windows of one layer each, in one launch: olds[i] bf16 [1, taps, channels]
// in place from freshes[i]'s rows, `fresh_row` elements apart. The same copies.
extern "C" int zs_qwen4_shift_each(const zs_view* olds, const zs_view* freshes, int64_t count, int64_t keep,
                                   int64_t fresh_row, int64_t fresh_rows) {
  zs::begin();
  ZS_REQUIRE(count >= 1 && count <= MAX_WINDOWS && olds != nullptr && freshes != nullptr, "1 to 64 windows");
  Windows windows;
  int64_t taps = 0, channels = 0;
  for (int64_t i = 0; i < count; ++i) {
    const zs::View old(olds + i), fresh(freshes + i);
    ZS_REQUIRE(old.defined() && old.dtype() == ZS_BF16 && old.dim() == 3 && old.size(0) == 1,
               "old: bf16 [1, taps, channels]");
    if (i == 0) {
      taps = old.size(1);
      channels = old.size(2);
    }
    ZS_REQUIRE(old.size(1) == taps && old.size(2) == channels, "the windows have one shape");
    ZS_REQUIRE(fresh.defined() && fresh.dtype() == ZS_BF16 && fresh_row >= channels, "fresh: bf16 rows of the window");
    ZS_REQUIRE(fresh.numel() >= (fresh_rows - 1) * fresh_row + channels, "fresh does not hold the rows read");
    windows.old[i] = old.data_ptr<__nv_bfloat16>();
    windows.fresh[i] = fresh.data_ptr<__nv_bfloat16>();
  }
  ZS_REQUIRE(taps >= 1 && taps <= MAX_TAPS, "1 to 32 window rows");
  ZS_REQUIRE(keep >= 1 && keep <= fresh_rows, "keep 1 to the new rows");
  const dim3 grid(static_cast<unsigned>((channels + 255) / 256), static_cast<unsigned>(count));
  shift_each_kernel<<<grid, 256, 0, zs::stream()>>>(windows, static_cast<int>(keep), static_cast<int>(taps),
                                                    static_cast<int>(channels), fresh_row);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
