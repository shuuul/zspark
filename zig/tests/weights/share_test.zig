//! `weights/share.zig`: `take` slices rows and copies columns.

const std = @import("std");
const testing = std.testing;
const safetensors = @import("../../src/weights/safetensors.zig");
const share = @import("../../src/weights/share.zig");

test "half of the rows is a slice, half of the columns a copy, a whole itself" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var bytes: [4 * 6 * 2]u8 = undefined;
    for (0..4) |r| for (0..6) |c| {
        bytes[(r * 6 + c) * 2] = @intCast(r);
        bytes[(r * 6 + c) * 2 + 1] = @intCast(c);
    };
    const matrix: safetensors.Tensor = .{ .dtype = .BF16, .shape = &.{ 4, 6 }, .data = &bytes };
    for (0..share.ranks) |rank| {
        const rows = try share.take(arena, matrix, .row, rank, share.ranks);
        try testing.expectEqualSlices(u64, &.{ 2, 6 }, rows.shape);
        try testing.expectEqual(@intFromPtr(&bytes[rank * 24]), @intFromPtr(rows.data.ptr));
        const cols = try share.take(arena, matrix, .col, rank, share.ranks);
        try testing.expectEqualSlices(u64, &.{ 4, 3 }, cols.shape);
        for (0..4) |r| for (0..3) |c| {
            try testing.expectEqual(@as(u8, @intCast(r)), cols.data[(r * 3 + c) * 2]);
            try testing.expectEqual(@as(u8, @intCast(rank * 3 + c)), cols.data[(r * 3 + c) * 2 + 1]);
        };
        try testing.expectEqual(@intFromPtr(&bytes), @intFromPtr((try share.take(arena, matrix, .whole, rank, share.ranks)).data.ptr));
    }
}
