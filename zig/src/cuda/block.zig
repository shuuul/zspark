//! Where each part of one device allocation begins (`cuda.cut`): a part at a multiple of `alignment`, as
//! `cudaMalloc` aligns a buffer of its own, so a kernel reading a part sees what it saw in a buffer of its own.
//! Host only: nothing here reaches the device.

const std = @import("std");

/// What `cudaMalloc` aligns an allocation to.
pub const alignment = 256;

/// Where a part of `size` bytes begins when the parts before it end at `end`; a part of no byte takes none.
pub fn place(end: usize, size: usize) usize {
    return if (size == 0) end else std.mem.alignForward(usize, end, alignment);
}
