//! `hc.zig`'s read-out at a prompt window's rows (spec 095 WS-03): the fused forms against the launches they
//! stand for, bit for bit, at window widths ragged and whole, a row alone as among others; and the launches alone
//! at the model's shapes for a profiler.
//!
//! `ZSPARK_HC_WINDOW_PROFILE=FORM:ROWS` launches one form `passes` times at the model's shapes on random rows:
//! `base` (the up GEMM on the window's tile, then `mix`), `window2` and `window3` (`upMixWindow` on 2 or 3
//! stages), `act` (the read-out's GEMM, its K slices summed in the GEMM from
//! `bf16.sums_from` rows, then `activate`). Unset, the profile test is skipped.
//!
//!   ZSPARK_HC_WINDOW_PROFILE=window2:2048 ncu --metrics gpu__time_duration.sum TEST

const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const bf16 = @import("../../../../src/weights/bf16.zig");
const hc = @import("../../../../src/models/qwen4/blocks/hc.zig");
const numeric_fixture = @import("../../../weights/numeric_fixture.zig");
const gpu = @import("../../../cuda/testing.zig");
const randomBf16 = numeric_fixture.randomBf16;
const device = gpu.device;

const testing = std.testing;

/// Flash Next's read-out: 4 streams of 2,560, a rank of 320 and 4 inject gates.
const flash = struct {
    const dims = 2560;
    const streams = 4;
    const wide = streams * dims;
    const low = 320;
    const n = low + streams;
    const sk = 32;
};

fn fetch(gpa: std.mem.Allocator, buffer: cuda.Buffer, count: usize) ![]u16 {
    try cuda.synchronize();
    const host = try gpa.alloc(u16, count);
    errdefer gpa.free(host);
    try buffer.downloadAt(0, std.mem.sliceAsBytes(host));
    return host;
}

/// `upMixWindow` against the up GEMM (bf16) and `mix`, at the model's shapes:
/// windows of every width the kernel's tiles cut differently (one row, under a tile, a tile and one more, ragged
/// prompt windows of the trace, 2,048), each stage count; then rows alone.
fn windowAgainstLaunches(dtype: cuda.DType) !void {
    const gpa = testing.allocator;
    const most = 2048;
    var prng: std.Random.DefaultPrng = .init(951);
    const random = prng.random();
    const normed_host = try gpa.alloc(u16, most * flash.wide);
    defer gpa.free(normed_host);
    randomBf16(random, normed_host, 1.5);
    const act_host = try gpa.alloc(u16, most * flash.low);
    defer gpa.free(act_host);
    randomBf16(random, act_host, 0.6);
    const up_host = try gpa.alloc(u16, flash.wide * flash.low);
    defer gpa.free(up_host);
    randomBf16(random, up_host, 0.06);

    var normed = try device(u16, .bf16, &.{ most, flash.wide }, normed_host);
    defer normed.deinit();
    var act = try device(u16, dtype, &.{ most, flash.low }, act_host);
    defer act.deinit();
    var up_weight = try device(u16, dtype, &.{ flash.wide, flash.low }, up_host);
    defer up_weight.deinit();
    var up: cuda.Buffer = try .init(.bf16, &.{ most, flash.wide });
    defer up.deinit();
    var mixed: [2]cuda.Buffer = undefined;
    mixed[0] = try .init(.bf16, &.{ most, flash.dims });
    defer mixed[0].deinit();
    mixed[1] = try .init(.bf16, &.{ most, flash.dims });
    defer mixed[1].deinit();

    const widths = [_]usize{ 1, 17, 128, 129, 300, 1024, 1100, 1943, 2011, 2048 };
    for (widths) |rows| {
        // the launches as they are: the GEMM on the window's tile, `mix`
        const tile = bf16.Tile.auto.of(rows, 1, flash.wide, flash.low);
        try cuda.launched(cuda.kernels.zs_b16_gemm(&act.rows(rows), &up_weight.view, &up.rows(rows), null, 1, tile));
        try hc.mix(up.rows(rows), normed.rows(rows), mixed[0].rows(rows), flash.streams);
        const want = try fetch(gpa, mixed[0], rows * flash.dims);
        defer gpa.free(want);
        for ([_]usize{ 2, 3 }) |stages| {
            try mixed[1].zero();
            try hc.upMixWindow(act.rows(rows), up_weight.view, normed.rows(rows), mixed[1].rows(rows), flash.streams, stages);
            const got = try fetch(gpa, mixed[1], rows * flash.dims);
            defer gpa.free(got);
            try testing.expectEqualSlices(u16, want, got);
        }
        // the values are not all alike
        var distinct: usize = 0;
        for (want[1..], want[0 .. want.len - 1]) |a, b| distinct += @intFromBool(a != b);
        try testing.expect(distinct > want.len / 2);
    }

    // rows of a 2,048-row window alone, each in a window of its own, at another row tile's place
    const all = try fetch(gpa, mixed[1], most * flash.dims);
    defer gpa.free(all);
    for ([_]usize{ 0, 127, 128, 1000, 2047 }) |row| {
        try mixed[1].zero();
        const at_act: cuda.View = .init(@as([*]u8, @ptrCast(act.view.ptr.?)) + row * flash.low * 2, dtype, &.{ 1, flash.low });
        const at_normed: cuda.View = .init(@as([*]u8, @ptrCast(normed.view.ptr.?)) + row * flash.wide * 2, .bf16, &.{ 1, flash.wide });
        try hc.upMixWindow(at_act, up_weight.view, at_normed, mixed[1].rows(1), flash.streams, 2);
        const one = try fetch(gpa, mixed[1], flash.dims);
        defer gpa.free(one);
        try testing.expectEqualSlices(u16, all[row * flash.dims ..][0..flash.dims], one);
    }
}

test "the read-out's up and mix as one launch at a window's rows give the GEMM's and mix's bits, a row alone as among others" {
    try windowAgainstLaunches(.bf16);
}

const Form = enum { base, act, window2, window3 };

test "PROFILE the read-out's up and mix at a window's rows" {
    const asked = std.c.getenv("ZSPARK_HC_WINDOW_PROFILE") orelse return error.SkipZigTest;
    const text = std.mem.span(asked);
    const colon = std.mem.indexOfScalar(u8, text, ':') orelse return error.InvalidArgument;
    const form = std.meta.stringToEnum(Form, text[0..colon]) orelse return error.InvalidArgument;
    const rows = try std.fmt.parseInt(usize, text[colon + 1 ..], 10);
    const passes = 6;
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(95);
    const random = prng.random();

    const normed_host = try gpa.alloc(u16, rows * flash.wide);
    defer gpa.free(normed_host);
    randomBf16(random, normed_host, 1.0);
    const act_host = try gpa.alloc(u16, rows * flash.low);
    defer gpa.free(act_host);
    randomBf16(random, act_host, 0.5);
    const up_host = try gpa.alloc(u16, flash.wide * flash.low);
    defer gpa.free(up_host);
    randomBf16(random, up_host, 0.05);
    const down_host = try gpa.alloc(u16, flash.n * flash.wide);
    defer gpa.free(down_host);
    randomBf16(random, down_host, 0.02);

    var normed = try device(u16, .bf16, &.{ rows, flash.wide }, normed_host);
    defer normed.deinit();
    var act = try device(u16, .bf16, &.{ rows, flash.low }, act_host);
    defer act.deinit();
    var up_weight = try device(u16, .bf16, &.{ flash.wide, flash.low }, up_host);
    defer up_weight.deinit();
    var down_weight = try device(u16, .bf16, &.{ flash.n, flash.wide }, down_host);
    defer down_weight.deinit();
    var up: cuda.Buffer = try .init(.bf16, &.{ rows, flash.wide });
    defer up.deinit();
    var mixed: cuda.Buffer = try .init(.bf16, &.{ rows, flash.dims });
    defer mixed.deinit();
    var inject: cuda.Buffer = try .init(.bf16, &.{ rows, flash.streams });
    defer inject.deinit();
    var dn: cuda.Buffer = try .init(.f32, &.{ rows, flash.n });
    defer dn.deinit();
    var part: cuda.Buffer = try .init(.f32, &.{ flash.sk, rows, flash.n });
    defer part.deinit();

    const up_tile = bf16.Tile.auto.of(rows, 1, flash.wide, flash.low);
    const down_tile = bf16.Tile.auto.of(rows, flash.sk, flash.n, flash.wide);
    for (0..passes) |_| switch (form) {
        .base => {
            try cuda.launched(cuda.kernels.zs_b16_gemm(&act.view, &up_weight.view, &up.view, null, 1, up_tile));
            try hc.mix(up.view, normed.view, mixed.view, flash.streams);
        },
        .window2 => try hc.upMixWindow(act.view, up_weight.view, normed.view, mixed.view, flash.streams, 2),
        .window3 => try hc.upMixWindow(act.view, up_weight.view, normed.view, mixed.view, flash.streams, 3),
        .act => {
            const summed = rows >= bf16.sums_from;
            if (summed) {
                try cuda.launched(cuda.kernels.zs_b16_gemm(&normed.view, &down_weight.view, &dn.view, &part.view, flash.sk, down_tile));
                try hc.activate(.one(dn.view), act.view, inject.view, flash.streams, flash.low);
            } else {
                try cuda.launched(cuda.kernels.zs_b16_gemm(&normed.view, &down_weight.view, null, &part.view, flash.sk, down_tile));
                try hc.activate(.of(part.view, flash.sk, rows, flash.n), act.view, inject.view, flash.streams, flash.low);
            }
        },
    };
    try cuda.synchronize();
    std.debug.print("PROFILE {s} at {d} rows: {d} passes, up tile {d}, down tile {d}\n", .{ @tagName(form), rows, passes, up_tile, down_tile });
}
