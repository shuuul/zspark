//! What the device tests of every area share: a buffer from a host slice and a buffer's elements back. Every
//! file that imports this one reaches `cuda.zig`, so no root of the host tier lists one; `weights/numeric_fixture.zig`
//! is the host-safe side.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../src/cuda/cuda.zig");

pub fn device(comptime T: type, dtype: cuda.DType, shape: []const usize, host: []const T) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(dtype, shape);
    errdefer buffer.deinit();
    try buffer.upload(std.mem.sliceAsBytes(host));
    return buffer;
}

pub fn fetch(comptime T: type, gpa: Allocator, buffer: cuda.Buffer) ![]T {
    try cuda.synchronize();
    const host = try gpa.alloc(T, buffer.view.count());
    errdefer gpa.free(host);
    try buffer.download(std.mem.sliceAsBytes(host));
    return host;
}
