//! `ranks.zig`'s tests: where a rank's partial lies in a gathered buffer and what a reader is given, on views
//! over host memory (nothing is launched), and what a build without NCCL answers.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../src/cuda/cuda.zig");
const nccl = @import("../../src/ranks/nccl.zig");
const ranks = @import("../../src/ranks/ranks.zig");

test "a gathered buffer is tight: a rank's partial lies rank * rows * dims values in, whatever rows the buffer was made for" {
    const world = 2;
    const most = 5;
    const dims = 4;
    var memory: [world * most * dims]f32 = undefined;
    const buffer: cuda.View = .init(&memory, .f32, &.{ world, most, dims });
    for (1..most + 1) |rows| {
        const all = ranks.gathered(buffer, rows);
        try testing.expectEqual(@intFromPtr(&memory), @intFromPtr(all.ptr.?));
        try testing.expectEqualSlices(i64, &.{ world, @intCast(rows), dims }, all.size[0..3]);
        try testing.expectEqual(world * rows * dims, all.count());
        for (0..world) |rank| {
            const own = ranks.partial(buffer, rank, rows);
            // what the kernels read: rank r's rows `size(1) * dims` values after rank r - 1's (`hc.cu`)
            try testing.expectEqual(@intFromPtr(&memory[rank * rows * dims]), @intFromPtr(own.ptr.?));
            try testing.expectEqualSlices(i64, &.{ @intCast(rows), dims }, own.size[0..2]);
            // and of the reader's view the same place: an all-gather in place sends from there
            try testing.expectEqual(@intFromPtr(own.ptr.?), @intFromPtr(ranks.partial(all, rank, rows).ptr.?));
        }
    }
}

test "loopback gathers nothing and serves a process that holds every rank, and no other" {
    var memory: [2 * 3 * 4]f32 = undefined;
    for (&memory, 0..) |*value, index| value.* = @floatFromInt(index);
    const gather: ranks.Gather = .loopback;
    try gather.run(ranks.gathered(.init(&memory, .f32, &.{ 2, 3, 4 }), 2));
    for (memory, 0..) |value, index| try testing.expectEqual(@as(f32, @floatFromInt(index)), value);
    try testing.expect(gather.serves(2, 2));
    try testing.expect(!gather.serves(1, 2));
}

test "a build without NCCL says so: no communicator, no id, no version" {
    if (nccl.linked) return error.SkipZigTest;
    try testing.expectError(error.NcclNotLinked, nccl.version());
    try testing.expectError(error.NcclNotLinked, nccl.uniqueId());
    try testing.expectError(error.NcclNotLinked, nccl.Comm.init(2, .{ .internal = @splat(0) }, 0));
}

test "a split window's shares: whole rows in rank order, each rank's first a multiple of 16 rows, every row once" {
    for (1..300) |rows| for (1..5) |world| {
        var at: usize = 0;
        for (0..world) |rank| {
            const own = ranks.share(rows, world, rank);
            try testing.expectEqual(at, own[0]);
            try testing.expect(own[1] >= own[0]);
            // a rank's first row 16 rows in, so 128 bytes and more in every row-indexed buffer (spec 106 WS-02)
            if (own[1] > own[0]) try testing.expectEqual(@as(usize, 0), own[0] % ranks.cut_rows);
            at = own[1];
        }
        try testing.expectEqual(rows, at);
    };
    try testing.expectEqual([2]usize{ 0, 1024 }, ranks.share(2048, 2, 0));
    try testing.expectEqual([2]usize{ 1024, 2048 }, ranks.share(2048, 2, 1));
    try testing.expectEqual([2]usize{ 0, 32 }, ranks.share(33, 2, 0));
    try testing.expectEqual([2]usize{ 32, 33 }, ranks.share(33, 2, 1));
    try testing.expectEqual([2]usize{ 16, 16 }, ranks.share(16, 2, 1));
}
