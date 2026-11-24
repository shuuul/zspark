//! `state.zig`'s tests.

const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const state = @import("../../../../src/models/qwen4/blocks/state.zig");
const shiftWindows = state.shiftWindows;

test "windows slide to the last kept rows" {
    const gpa = std.testing.allocator;
    const layers = 2;
    const taps = 3;
    const channels = 300;
    const rows = 5;
    const width = channels + 7; // a new row is wider than the window
    var prng: std.Random.DefaultPrng = .init(3);
    const old = try gpa.alloc(u16, layers * taps * channels);
    defer gpa.free(old);
    const fresh = try gpa.alloc(u16, layers * rows * width);
    defer gpa.free(fresh);
    prng.random().bytes(std.mem.sliceAsBytes(old));
    prng.random().bytes(std.mem.sliceAsBytes(fresh));
    const got = try gpa.alloc(u16, old.len);
    defer gpa.free(got);

    var fresh_dev: cuda.Buffer = try .init(.bf16, &.{ layers, rows, width });
    defer fresh_dev.deinit();
    try fresh_dev.upload(std.mem.sliceAsBytes(fresh));
    // fewer kept rows than the window holds, as many, and more
    for ([_]usize{ 1, 2, 3, 5 }) |keep| {
        var old_dev: cuda.Buffer = try .init(.bf16, &.{ layers, taps, channels });
        defer old_dev.deinit();
        try old_dev.upload(std.mem.sliceAsBytes(old));
        try shiftWindows(old_dev.view, .{ .view = fresh_dev.view, .layer = rows * width, .row = width, .rows = rows }, keep);
        try cuda.synchronize();
        try old_dev.download(std.mem.sliceAsBytes(got));
        for (0..layers) |l| for (0..taps) |j| {
            const src = keep + j;
            const want = if (src < taps) old[(l * taps + src) * channels ..][0..channels] else fresh[(l * rows + src - taps) * width ..][0..channels];
            try std.testing.expectEqualSlices(u16, want, got[(l * taps + j) * channels ..][0..channels]);
        };
    }
}

test "several windows shifted in one launch are each window's own shift" {
    const gpa = std.testing.allocator;
    const count = 5;
    const taps = 3;
    const channels = 700;
    const rows = 4;
    const width = channels + 9;
    var prng: std.Random.DefaultPrng = .init(4);
    const host = try gpa.alloc(u16, @max(taps * channels, rows * width));
    defer gpa.free(host);
    const want = try gpa.alloc(u16, taps * channels);
    defer gpa.free(want);
    const got = try gpa.alloc(u16, taps * channels);
    defer gpa.free(got);
    var olds: [2][count]cuda.Buffer = undefined;
    var freshes: [count]cuda.Buffer = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        olds[0][i].deinit();
        olds[1][i].deinit();
        freshes[i].deinit();
    };
    for ([_]usize{ 1, 2, 3, 4 }) |keep| {
        for (0..made) |i| {
            olds[0][i].deinit();
            olds[1][i].deinit();
            freshes[i].deinit();
        }
        made = 0;
        var old_views: [count]cuda.View = undefined;
        var fresh_views: [count]cuda.View = undefined;
        for (0..count) |i| {
            prng.random().bytes(std.mem.sliceAsBytes(host));
            olds[0][i] = try .init(.bf16, &.{ 1, taps, channels });
            errdefer olds[0][i].deinit();
            olds[1][i] = try .init(.bf16, &.{ 1, taps, channels });
            errdefer olds[1][i].deinit();
            try olds[0][i].upload(std.mem.sliceAsBytes(host[0 .. taps * channels]));
            try olds[1][i].upload(std.mem.sliceAsBytes(host[0 .. taps * channels]));
            prng.random().bytes(std.mem.sliceAsBytes(host));
            freshes[i] = try .init(.bf16, &.{ rows, width });
            try freshes[i].upload(std.mem.sliceAsBytes(host[0 .. rows * width]));
            made += 1;
            old_views[i] = olds[1][i].view;
            fresh_views[i] = freshes[i].view;
            try shiftWindows(olds[0][i].view, .{ .view = freshes[i].view, .layer = 0, .row = width, .rows = rows }, keep);
        }
        try state.shiftEach(&old_views, &fresh_views, width, rows, keep);
        try cuda.synchronize();
        for (0..count) |i| {
            try olds[0][i].download(std.mem.sliceAsBytes(want));
            try olds[1][i].download(std.mem.sliceAsBytes(got));
            try std.testing.expectEqualSlices(u16, want, got);
        }
    }
}
