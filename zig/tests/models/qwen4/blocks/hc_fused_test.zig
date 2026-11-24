//! `hc.zig`'s read-out at a prompt window's rows in three launches (spec 148 WS-03, after Mia's AI Lab's
//! `flashnext_hc_fuse_tests.zig`): `downWindow` and `upMixStreams` against the chain they stand for (`normed`, the
//! down GEMM as the linear launches it, `activate`, `upMixWindow`), every output byte, at window widths ragged and
//! whole, with the inject gates and without; then rows alone as among others; and the launches alone
//! at the model's shapes for a profiler.
//!
//! `ZSPARK_HC_FUSED_PROFILE=FORM:ROWS` launches one form `passes` times on random rows: `chain` (`normed`, the down
//! GEMM, `activate`, `upMixWindow`), `fused` (`downWindow`, `upMixStreams`), `down` (`downWindow` alone) and `up`
//! (`upMixStreams` alone). Unset, the profile test is skipped.
//!
//!   ZSPARK_HC_FUSED_PROFILE=fused:2048 ncu --metrics gpu__time_duration.sum TEST

const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const bf16 = @import("../../../../src/weights/bf16.zig");
const hc = @import("../../../../src/models/qwen4/blocks/hc.zig");
const numeric_fixture = @import("../../../weights/numeric_fixture.zig");
const gpu = @import("../../../cuda/testing.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const randomBf16 = numeric_fixture.randomBf16;
const device = gpu.device;

const testing = std.testing;

/// Flash Next's read-out: 4 streams of 2,560, a rank of 320 and 4 inject gates.
const flash = struct {
    const dims = 2560;
    const streams = 4;
    const wide = streams * dims;
    const low = 320;
    const blocks = dims / hc.block;
    const eps: f32 = 1e-6;
};

fn fetch(comptime T: type, gpa: std.mem.Allocator, view: cuda.View, count: usize) ![]T {
    try cuda.synchronize();
    const host = try gpa.alloc(T, count);
    errdefer gpa.free(host);
    try cuda.download(.init(view.ptr.?, view.dtype, &.{count}), std.mem.sliceAsBytes(host));
    return host;
}

/// A view of `count` rows of `width` from row `first` of `view`'s buffer.
fn at(view: cuda.View, dtype: cuda.DType, width: usize, first: usize, count: usize) cuda.View {
    return .init(@as([*]u8, @ptrCast(view.ptr.?)) + first * width * dtype.size(), dtype, &.{ count, width });
}

/// The inputs of a read-out of up to `most` rows: the streams, their squares, the scale and the two weights.
const Rig = struct {
    h: cuda.Buffer,
    pss: cuda.Buffer,
    scale: cuda.Buffer,
    down: cuda.Buffer,
    up: cuda.Buffer,
    n: usize,
    sk: usize,
    // the chain's
    normed: cuda.Buffer,
    dn: cuda.Buffer,
    part: cuda.Buffer,
    // each side's outputs: act, inject, mixed
    act: [2]cuda.Buffer,
    inject: [2]cuda.Buffer,
    mixed: [2]cuda.Buffer,

    fn init(gpa: std.mem.Allocator, most: usize, inject: bool, seed: u64) !Rig {
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        const n: usize = flash.low + @as(usize, if (inject) flash.streams else 0);
        const sk = bf16.splitK(n, flash.wide);
        const h_host = try gpa.alloc(u16, most * flash.wide);
        defer gpa.free(h_host);
        randomBf16(random, h_host, 1.5);
        // a few rows far larger and far smaller, as streams are
        for (0..most) |row| if (row % 7 == 3) {
            for (h_host[row * flash.wide ..][0..flash.wide]) |*v| v.* = numeric.bf16FromF32(random.floatNorm(f32) * if (row % 2 == 0) @as(f32, 40.0) else 0.01);
        };
        const scale_host = try gpa.alloc(f32, flash.wide);
        defer gpa.free(scale_host);
        for (scale_host) |*v| v.* = 0.4 + random.float(f32) * 1.2;
        const down_host = try gpa.alloc(u16, n * flash.wide);
        defer gpa.free(down_host);
        randomBf16(random, down_host, 0.02);
        const up_host = try gpa.alloc(u16, flash.wide * flash.low);
        defer gpa.free(up_host);
        randomBf16(random, up_host, 0.06);

        var self: Rig = undefined;
        self.n = n;
        self.sk = sk;
        self.h = try device(u16, .bf16, &.{ most, flash.wide }, h_host);
        self.pss = try .init(.f32, &.{ most, flash.blocks, flash.streams });
        self.scale = try device(f32, .f32, &.{flash.wide}, scale_host);
        self.down = try device(u16, .bf16, &.{ n, flash.wide }, down_host);
        self.up = try device(u16, .bf16, &.{ flash.wide, flash.low }, up_host);
        self.normed = try .init(.bf16, &.{ most, flash.wide });
        self.dn = try .init(.f32, &.{ most, n });
        self.part = try .init(.f32, &.{ sk, most, n });
        for (0..2) |i| {
            self.act[i] = try .init(.bf16, &.{ most, flash.low });
            self.inject[i] = try .init(.bf16, &.{ most, flash.streams });
            self.mixed[i] = try .init(.bf16, &.{ most, flash.dims });
        }
        // the squares, as the write-back leaves them (mode 0 leaves h)
        try hc.writeback(self.h.view, self.pss.view, .none, flash.streams);
        return self;
    }

    fn deinit(self: *Rig) void {
        inline for (.{ "h", "pss", "scale", "down", "up", "normed", "dn", "part" }) |name| @field(self, name).deinit();
        for (0..2) |i| {
            self.act[i].deinit();
            self.inject[i].deinit();
            self.mixed[i].deinit();
        }
    }

    fn injects(self: Rig) bool {
        return self.n > flash.low;
    }

    /// The chain `forward.readoutAfter` runs without the fused launches, over rows [first, first + rows) into
    /// side 0's buffers at row 0. The down GEMM as `bf16.Linear.slices` launches it.
    fn chain(self: *Rig, first: usize, rows: usize) !void {
        const h = at(self.h.view, .bf16, flash.wide, first, rows);
        const pss = at(self.pss.view, .f32, flash.blocks * flash.streams, first, rows);
        const normed = self.normed.rows(rows);
        try hc.normed(h, pss, self.scale.view, normed, flash.eps, flash.streams);
        const tile = bf16.Tile.auto.of(rows, self.sk, self.n, flash.wide);
        const slices: cuda.Slices = if (rows < bf16.sums_from) slices: {
            try cuda.launched(cuda.kernels.zs_b16_gemm(&normed, &self.down.view, null, &self.part.view, @intCast(self.sk), tile));
            break :slices .of(self.part.view, self.sk, rows, self.n);
        } else slices: {
            try cuda.launched(cuda.kernels.zs_b16_gemm(&normed, &self.down.view, &self.dn.rows(rows), &self.part.view, @intCast(self.sk), tile));
            break :slices .one(self.dn.rows(rows));
        };
        try hc.activate(slices, self.act[0].rows(rows), if (self.injects()) self.inject[0].view else null, flash.streams, flash.low);
        try hc.upMixWindow(self.act[0].rows(rows), self.up.view, normed, self.mixed[0].rows(rows), flash.streams, 2);
    }

    /// The fused launches over the same rows into side 1's buffers at row 0.
    fn fused(self: *Rig, first: usize, rows: usize) !void {
        const h = at(self.h.view, .bf16, flash.wide, first, rows);
        const pss = at(self.pss.view, .f32, flash.blocks * flash.streams, first, rows);
        try hc.downWindow(h, pss, self.scale.view, self.down.view, self.sk, self.act[1].rows(rows), if (self.injects()) self.inject[1].view else null, flash.eps, flash.streams, flash.low);
        try hc.upMixStreams(self.act[1].rows(rows), self.up.view, h, pss, self.scale.view, self.mixed[1].rows(rows), flash.eps, flash.streams);
    }

    /// Side 1's outputs over `rows` rows are side 0's, byte for byte.
    fn expectSame(self: Rig, gpa: std.mem.Allocator, rows: usize) !void {
        const outputs = [_]struct { [2]cuda.Buffer, usize }{
            .{ self.act, flash.low },
            .{ self.inject, flash.streams },
            .{ self.mixed, flash.dims },
        };
        for (outputs, 0..) |output, i| {
            if (i == 1 and !self.injects()) continue;
            const want = try fetch(u16, gpa, output[0][0].view, rows * output[1]);
            defer gpa.free(want);
            const got = try fetch(u16, gpa, output[0][1].view, rows * output[1]);
            defer gpa.free(got);
            try testing.expectEqualSlices(u16, want, got);
            // the values are not all alike
            var distinct: usize = 0;
            for (want[1..], want[0 .. want.len - 1]) |a, b| distinct += @intFromBool(a != b);
            try testing.expect(rows == 1 or distinct > want.len / 4);
        }
    }

    fn clear(self: *Rig) !void {
        for (0..2) |i| {
            try self.act[i].zero();
            try self.inject[i].zero();
            try self.mixed[i].zero();
        }
    }
};

fn fusedAgainstChain(inject: bool) !void {
    const gpa = testing.allocator;
    const most = 2048;
    var rig: Rig = try .init(gpa, most, inject, if (inject) 1481 else 1482);
    defer rig.deinit();
    try testing.expectEqual(@as(usize, 32), rig.sk);
    // under a tile, each tile's edge, either side of `bf16.sums_from`, ragged prompt windows, a whole window
    const widths = [_]usize{ 1, 17, 32, 33, 48, 64, 128, 129, 300, 1000, 1023, 1024, 1100, 1943, 2048 };
    for (widths) |rows| {
        try rig.clear();
        try rig.chain(0, rows);
        try rig.fused(0, rows);
        try rig.expectSame(gpa, rows);
    }
    // rows of the 2,048-row window alone, each at another tile's place, against the chain over all of them
    try rig.clear();
    try rig.chain(0, most);
    for ([_]usize{ 0, 31, 47, 48, 1000, 2047 }) |row| {
        try rig.act[1].zero();
        try rig.mixed[1].zero();
        try rig.fused(row, 1);
        const outputs = [_]struct { [2]cuda.Buffer, usize }{ .{ rig.act, flash.low }, .{ rig.mixed, flash.dims } };
        for (outputs) |output| {
            const all = try fetch(u16, gpa, output[0][0].view, most * output[1]);
            defer gpa.free(all);
            const one = try fetch(u16, gpa, output[0][1].view, output[1]);
            defer gpa.free(one);
            try testing.expectEqualSlices(u16, all[row * output[1] ..][0..output[1]], one);
        }
    }
}

test "the read-out's norm, down, activation, up and mix in two launches give every byte of the chain's, a row alone as among others" {
    try fusedAgainstChain(true);
}

test "the mixer's read-out, without inject gates, in two launches gives the chain's bytes" {
    try fusedAgainstChain(false);
}

const Form = enum { chain, fused, down, up };

test "PROFILE the read-out in two launches at a window's rows" {
    const asked = std.c.getenv("ZSPARK_HC_FUSED_PROFILE") orelse return error.SkipZigTest;
    const text = std.mem.span(asked);
    const colon = std.mem.indexOfScalar(u8, text, ':') orelse return error.InvalidArgument;
    const form = std.meta.stringToEnum(Form, text[0..colon]) orelse return error.InvalidArgument;
    const rows = try std.fmt.parseInt(usize, text[colon + 1 ..], 10);
    const passes = 6;
    var rig: Rig = try .init(testing.allocator, rows, true, 148);
    defer rig.deinit();
    for (0..passes) |_| switch (form) {
        .chain => try rig.chain(0, rows),
        .fused => try rig.fused(0, rows),
        .down => try hc.downWindow(rig.h.view, rig.pss.view, rig.scale.view, rig.down.view, rig.sk, rig.act[1].view, rig.inject[1].view, flash.eps, flash.streams, flash.low),
        .up => try hc.upMixStreams(rig.act[1].view, rig.up.view, rig.h.view, rig.pss.view, rig.scale.view, rig.mixed[1].view, flash.eps, flash.streams),
    };
    try cuda.synchronize();
    std.debug.print("PROFILE {s} at {d} rows: {d} passes\n", .{ @tagName(form), rows, passes });
}
