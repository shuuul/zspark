// The rotary position a value of a row takes (spec 019): its own position, or for an image prompt one of three.
// Semantics from TensorFold 0.6.5 (609ca41) families/qwen4_exp/cuda/image_rows.py (rope_axis, MODE 2), Triton,
// Apache-2.0; rewritten for zspark in CUDA C++.
//
// ROPE int32 [2 + 3 capacity] is a sequence's (attention.zig's `Rope`): ROPE[0] the delta, ROPE[1] the length the
// table covers, then (t, h, w) for each position below it. A position under the length takes the table's t, h or
// w by its rotary pair i: h where i % 3 == 1 and i < 3 S1, w where i % 3 == 2 and i < 3 S2, else t (the
// interleaved sections of `mrope_section`); a position past it, pos + delta. Read from device memory, so a graph
// captured before the image serves it. A null ROPE is the text path as it was; a text sequence's words are zeros,
// so its axis is pos + 0, the same integer.

#pragma once

__device__ __forceinline__ int rope_axis(const int* __restrict__ ROPE, int pos, int i, int S1, int S2) {
  if (ROPE == nullptr) return pos;
  if (pos >= ROPE[1]) return pos + ROPE[0];
  const int* const at = ROPE + 2 + 3 * (size_t)pos;
  return (i % 3 == 1 && i < 3 * S1) ? at[1] : (i % 3 == 2 && i < 3 * S2) ? at[2] : at[0];
}
