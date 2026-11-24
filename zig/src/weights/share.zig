//! A rank's share of a tensor's bytes. Host only: bytes in, bytes out.
//!
//! GLM-5.3-Flash cuts every wide tensor this way (TensorFold's `split.py`), and Flash Next's NVFP4 loader
//! slices the same axes (`families/qwen4_exp/cuda/weights.py`). Half of the first axis is a slice of the
//! tensor's own bytes. Half of the second axis, a matrix's columns, is not contiguous and is copied. What a
//! model does with the cut (which tensors, a DeltaNet's heads) stays in that model.

const std = @import("std");
const Allocator = std.mem.Allocator;
const safetensors = @import("safetensors.zig");

/// Ranks these splits were written for. A caller of `take` passes its own.
pub const ranks = 2;

/// What a rank holds of a tensor.
pub const Split = enum {
    /// Half of the first dimension: the rank's heads, or its half of an inner width's outputs.
    row,
    /// Half of the second dimension of a matrix: its half of an inner width's inputs.
    col,
    /// All of it, on both ranks.
    whole,
    /// `lm_head`: the rule replicates it and the loader then keeps the rank's half of the vocabulary's rows.
    /// One name for the two steps.
    vocab,
    /// Not read.
    drop,

    /// The bytes of `total` a rank of `ranks` holds.
    pub fn share(self: Split, total: u64) u64 {
        return switch (self) {
            .row, .col, .vocab => total / ranks,
            .whole => total,
            .drop => 0,
        };
    }
};

/// `tensor` as `rank` of `of` holds it.
pub fn take(arena: Allocator, tensor: safetensors.Tensor, split: Split, rank: usize, of: usize) Allocator.Error!safetensors.Tensor {
    std.debug.assert(of >= 1 and rank < of);
    switch (split) {
        .whole => return tensor,
        .drop => unreachable,
        .row, .vocab => {
            std.debug.assert(tensor.shape.len >= 1 and tensor.shape[0] % of == 0);
            const shape = try arena.dupe(u64, tensor.shape);
            shape[0] /= of;
            const half = tensor.data.len / of;
            return .{ .dtype = tensor.dtype, .shape = shape, .data = tensor.data[rank * half ..][0..half] };
        },
        .col => {
            std.debug.assert(tensor.shape.len >= 2 and tensor.shape[1] % of == 0);
            const shape = try arena.dupe(u64, tensor.shape);
            shape[1] /= of;
            const outer: usize = @intCast(tensor.shape[0]);
            const inner = tensor.data.len / outer;
            const half = inner / of;
            // on a cache line: the bytes are read as int16 and uploaded
            const data = try arena.alignedAlloc(u8, .@"64", outer * half);
            for (0..outer) |o| @memcpy(data[o * half ..][0..half], tensor.data[o * inner + rank * half ..][0..half]);
            return .{ .dtype = tensor.dtype, .shape = shape, .data = data };
        },
    }
}
