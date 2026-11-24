//! `taps.zig`'s tests: synthetic asymmetric taps, the mean against the first stream, the row rule, off-by-one.

const std = @import("std");
const numeric = @import("../../../src/weights/numeric.zig");
const taps = @import("../../../src/models/qwen4/taps.zig");

const layers = taps.layers;
fn fillResidual(layer: u32, rows: usize, hidden: usize, nstreams: usize, out: []u16) void {
    std.debug.assert(out.len == rows * nstreams * hidden);
    for (0..rows) |r| {
        for (0..nstreams) |s| {
            for (0..hidden) |d| {
                const v = @as(f32, @floatFromInt(layer + 1)) * 1.25 + @as(f32, @floatFromInt(s)) * 0.375 +
                    @as(f32, @floatFromInt(r)) * 0.0625 + @as(f32, @floatFromInt(d)) * 0.00390625;
                out[(r * nstreams + s) * hidden + d] = numeric.bf16FromF32(v);
            }
        }
    }
}

test "asymmetric taps: swapping layers 5 and 15 fails" {
    const hidden: usize = 8;
    const nstreams: usize = 4;
    const rows: usize = 3;
    var t = try taps.Taps.init(std.testing.allocator, &layers, hidden, nstreams, rows, .f32_acc);
    defer t.deinit();

    var residual: [3 * 4 * 8]u16 = undefined;
    for (layers) |layer| {
        fillResidual(layer, rows, hidden, nstreams, &residual);
        try std.testing.expect(t.capture(layer, &residual));
    }

    const wide = layers.len * hidden;
    const swapped = try std.testing.allocator.dupe(u16, t.values);
    defer std.testing.allocator.free(swapped);
    for (0..rows) |r| {
        const a = swapped[r * wide ..][0..hidden];
        const b = swapped[r * wide + hidden ..][0..hidden];
        var tmp: [8]u16 = undefined;
        @memcpy(&tmp, a);
        @memcpy(a, b);
        @memcpy(b, &tmp);
    }
    try std.testing.expect(!std.mem.eql(u16, t.values, swapped));
}

test "the mean is not the first stream" {
    const hidden: usize = 8;
    const nstreams: usize = 4;
    var residual: [4 * 8]u16 = undefined;
    fillResidual(5, 1, hidden, nstreams, &residual);

    var mean: [8]u16 = undefined;
    taps.streamMean(&residual, &mean, nstreams, .f32_acc);
    try std.testing.expect(!std.mem.eql(u16, mean[0..], residual[0..hidden]));
}

test "a row alone equals that row among others" {
    const hidden: usize = 8;
    const nstreams: usize = 4;
    const rows: usize = 3;
    var residual: [3 * 4 * 8]u16 = undefined;
    fillResidual(15, rows, hidden, nstreams, &residual);

    var batched: [3 * 8]u16 = undefined;
    taps.streamMeanRows(&residual, &batched, rows, nstreams, .f32_acc);

    var alone: [8]u16 = undefined;
    taps.streamMean(residual[0 .. nstreams * hidden], &alone, nstreams, .f32_acc);
    try std.testing.expectEqualSlices(u16, &alone, batched[0..hidden]);
}

test "off-by-one layer taps are not captured" {
    const hidden: usize = 8;
    const nstreams: usize = 4;
    var t = try taps.Taps.init(std.testing.allocator, &layers, hidden, nstreams, 1, .f32_acc);
    defer t.deinit();

    var residual: [4 * 8]u16 = undefined;
    fillResidual(5, 1, hidden, nstreams, &residual);
    try std.testing.expect(!t.capture(4, &residual));
    try std.testing.expect(!t.capture(6, &residual));
    try std.testing.expect(std.mem.allEqual(u16, t.values, 0));
    try std.testing.expect(t.capture(5, &residual));
    try std.testing.expect(!std.mem.allEqual(u16, t.slot(0, 0), 0));
    try std.testing.expectEqual(@as(?usize, 0), taps.tapIndex(5));
    try std.testing.expectEqual(@as(?usize, 1), taps.tapIndex(15));
    try std.testing.expectEqual(@as(?usize, 4), taps.tapIndex(45));
    try std.testing.expectEqual(@as(?usize, null), taps.tapIndex(4));
    try std.testing.expectEqual(@as(?usize, null), taps.tapIndex(6));
}

test "f32_acc and bf16 means differ when a large stream swallows a small one" {
    const nstreams: usize = 4;
    var residual: [4]u16 = .{
        numeric.bf16FromF32(256.0),
        numeric.bf16FromF32(1.0),
        numeric.bf16FromF32(1.0),
        numeric.bf16FromF32(1.0),
    };
    var acc: [1]u16 = undefined;
    var rounded: [1]u16 = undefined;
    taps.streamMean(&residual, &acc, nstreams, .f32_acc);
    taps.streamMean(&residual, &rounded, nstreams, .bf16);
    try std.testing.expect(acc[0] != rounded[0]);
    try std.testing.expectEqual(numeric.bf16FromF32(64.75), acc[0]);
    try std.testing.expectEqual(numeric.bf16FromF32(64.0), rounded[0]);
}
