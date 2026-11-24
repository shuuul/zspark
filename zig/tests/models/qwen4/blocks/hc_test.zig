//! `hc.zig`'s tests: each kernel against the same steps on the host, and a row alone against the row among others.

const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const hc = @import("../../../../src/models/qwen4/blocks/hc.zig");
const activate = hc.activate;
const block = hc.block;
const embed = hc.embed;
const mix = hc.mix;
const normed = hc.normed;
const writeback = hc.writeback;

const testing = std.testing;
const hc_fixture = @import("hc_fixture.zig");
const numeric_fixture = @import("../../../weights/numeric_fixture.zig");
const bf = numeric.roundBf16;
const sigmoid = numeric_fixture.sigmoid;
const device = hc_fixture.device;
const fetch = hc_fixture.fetch;
const fx = hc_fixture.fx;
const Inputs = hc_fixture.Inputs;
const randomBf16 = hc_fixture.randomBf16;

/// bf16 values a rounding apart at most: exp and fused multiplies differ between the host and the device.
fn expectClose(want: f32, got_bits: u16) !void {
    const got = numeric.f32FromBf16(got_bits);
    try testing.expectApproxEqAbs(want, got, @abs(want) / 64.0 + 1e-30);
}

test "embed copies the token's row into every stream" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1);
    const vocab = 9;
    const table = try gpa.alloc(u16, vocab * fx.dims);
    defer gpa.free(table);
    randomBf16(prng.random(), table, 1.0);
    const ids = [_]i32{ 7, 0, 8 };

    var table_dev = try device(u16, .bf16, &.{ vocab, fx.dims }, table);
    defer table_dev.deinit();
    var ids_dev = try device(i32, .i32, &.{ids.len}, &ids);
    defer ids_dev.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ ids.len, fx.wide });
    defer out.deinit();
    try embed(ids_dev.view, table_dev.view, out.view, fx.streams);
    const got = try fetch(u16, gpa, out);
    defer gpa.free(got);
    for (ids, 0..) |id, r| for (0..fx.streams) |s| {
        try testing.expectEqualSlices(u16, table[@as(usize, @intCast(id)) * fx.dims ..][0..fx.dims], got[(r * fx.streams + s) * fx.dims ..][0..fx.dims]);
    };
}

test "embed reads no row outside its table: an id out of range gives zeros, the rows beside it their own" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(81);
    const vocab = 9;
    const table = try gpa.alloc(u16, vocab * fx.dims);
    defer gpa.free(table);
    randomBf16(prng.random(), table, 1.0);
    const ids = [_]i32{ 7, -1, vocab, 0, std.math.maxInt(i32), std.math.minInt(i32), 8, 1 << 20 };

    var table_dev = try device(u16, .bf16, &.{ vocab, fx.dims }, table);
    defer table_dev.deinit();
    var ids_dev = try device(i32, .i32, &.{ids.len}, &ids);
    defer ids_dev.deinit();
    // not zero before the launch, so a zero row is the kernel's
    const before = try gpa.alloc(u16, ids.len * fx.wide);
    defer gpa.free(before);
    @memset(before, numeric.bf16FromF32(3.0));
    var out = try device(u16, .bf16, &.{ ids.len, fx.wide }, before);
    defer out.deinit();
    try embed(ids_dev.view, table_dev.view, out.view, fx.streams);
    const got = try fetch(u16, gpa, out);
    defer gpa.free(got);
    const zeros: [fx.dims]u16 = @splat(0);
    for (ids, 0..) |id, r| for (0..fx.streams) |s| {
        const row = got[(r * fx.streams + s) * fx.dims ..][0..fx.dims];
        if (id >= 0 and id < vocab) {
            try testing.expectEqualSlices(u16, table[@as(usize, @intCast(id)) * fx.dims ..][0..fx.dims], row);
        } else {
            try testing.expectEqualSlices(u16, &zeros, row);
        }
    };
}

/// The read-out and both write-backs for `rows` rows of fixed random inputs; the caller owns the results.
const Run = struct {
    normed: []u16,
    act: []u16,
    inject: []u16,
    mixed: []u16,
    after_branch: []u16,
    after_slots: []u16,
    /// the squares of the streams as they were given, and after the slots' write-back
    pss: []f32,
    pss_slots: []f32,

    fn deinit(self: Run, gpa: std.mem.Allocator) void {
        inline for (comptime std.meta.fieldNames(Run)) |name| gpa.free(@field(self, name));
    }
};

fn run(gpa: std.mem.Allocator, in: Inputs, rows: usize) !Run {
    var h = try device(u16, .bf16, &.{ rows, fx.wide }, in.h[0 .. rows * fx.wide]);
    defer h.deinit();
    var scale = try device(f32, .f32, &.{fx.wide}, in.scale);
    defer scale.deinit();
    var down = try device(f32, .f32, &.{ rows, fx.low + fx.streams }, in.down[0 .. rows * (fx.low + fx.streams)]);
    defer down.deinit();
    var up = try device(u16, .bf16, &.{ rows, fx.wide }, in.up[0 .. rows * fx.wide]);
    defer up.deinit();
    var branch = try device(u16, .bf16, &.{ rows, fx.dims }, in.branch[0 .. rows * fx.dims]);
    defer branch.deinit();
    var y = try device(f32, .f32, &.{ rows * fx.slots, fx.dims }, in.y[0 .. rows * fx.slots * fx.dims]);
    defer y.deinit();
    var shared = try device(f32, .f32, &.{ rows, fx.dims }, in.shared[0 .. rows * fx.dims]);
    defer shared.deinit();
    var weights = try device(f32, .f32, &.{ rows, fx.slots }, in.weights[0 .. rows * fx.slots]);
    defer weights.deinit();

    var pss: cuda.Buffer = try .init(.f32, &.{ rows, fx.dims / block, fx.streams });
    defer pss.deinit();
    var normed_dev: cuda.Buffer = try .init(.bf16, &.{ rows, fx.wide });
    defer normed_dev.deinit();
    var act: cuda.Buffer = try .init(.bf16, &.{ rows, fx.low });
    defer act.deinit();
    var inject: cuda.Buffer = try .init(.bf16, &.{ rows, fx.streams });
    defer inject.deinit();
    var mixed: cuda.Buffer = try .init(.bf16, &.{ rows, fx.dims });
    defer mixed.deinit();

    try writeback(h.view, pss.view, .none, fx.streams);
    try normed(h.view, pss.view, scale.view, normed_dev.view, fx.eps, fx.streams);
    try activate(.one(down.view), act.view, inject.view, fx.streams, fx.low);
    try mix(up.view, normed_dev.view, mixed.view, fx.streams);
    var result: Run = undefined;
    result.pss = try fetch(f32, gpa, pss);
    result.normed = try fetch(u16, gpa, normed_dev);
    result.act = try fetch(u16, gpa, act);
    result.inject = try fetch(u16, gpa, inject);
    result.mixed = try fetch(u16, gpa, mixed);
    try writeback(h.view, pss.view, .{ .branch = .{ .branch = branch.view, .inject = inject.view } }, fx.streams);
    result.after_branch = try fetch(u16, gpa, h);
    try writeback(h.view, pss.view, .{ .slots = .{ .y = y.view, .shared = shared.view, .weights = weights.view, .inject = inject.view } }, fx.streams);
    result.after_slots = try fetch(u16, gpa, h);
    result.pss_slots = try fetch(f32, gpa, pss);

    // an EXL3 pack's slots: the shared expert's rows in the last slot of `y` and no buffer of their own, the
    // same streams bit for bit
    const joined = try gpa.dupe(f32, in.y[0 .. rows * fx.slots * fx.dims]);
    defer gpa.free(joined);
    for (0..rows) |r| @memcpy(joined[(r * fx.slots + fx.slots - 1) * fx.dims ..][0..fx.dims], in.shared[r * fx.dims ..][0..fx.dims]);
    try y.upload(std.mem.sliceAsBytes(joined));
    try h.upload(std.mem.sliceAsBytes(result.after_branch));
    try writeback(h.view, pss.view, .none, fx.streams);
    try writeback(h.view, pss.view, .{ .slots = .{ .y = y.view, .shared = null, .weights = weights.view, .inject = inject.view } }, fx.streams);
    const from_last_slot = try fetch(u16, gpa, h);
    defer gpa.free(from_last_slot);
    try std.testing.expectEqualSlices(u16, result.after_slots, from_last_slot);
    return result;
}

test "hyper-connection steps match the host" {
    const gpa = testing.allocator;
    const rows = 3;
    const in: Inputs = try .init(gpa, rows);
    defer in.deinit(gpa);
    const got = try run(gpa, in, rows);
    defer got.deinit(gpa);

    for (0..rows) |r| {
        for (0..fx.streams) |s| {
            var total: f32 = 0;
            for (0..fx.dims / block) |c| {
                var sum: f64 = 0;
                for (0..block) |i| {
                    const v: f64 = numeric.f32FromBf16(in.h[(r * fx.streams + s) * fx.dims + c * block + i]);
                    sum += v * v;
                }
                const part = got.pss[(r * (fx.dims / block) + c) * fx.streams + s];
                try testing.expectApproxEqRel(sum, @as(f64, part), 1e-5);
                total += part;
            }
            const rinv = 1.0 / @sqrt(total / fx.dims + fx.eps);
            for (0..fx.dims) |d| {
                const e = s * fx.dims + d;
                try expectClose(numeric.f32FromBf16(in.h[r * fx.wide + e]) * rinv * in.scale[e], got.normed[r * fx.wide + e]);
            }
        }
        for (0..fx.low) |i| {
            const v = bf(in.down[r * (fx.low + fx.streams) + i] / fx.streams);
            try expectClose(bf(v * sigmoid(v)), got.act[r * fx.low + i]);
        }
        for (0..fx.streams) |s| {
            const v = bf(in.down[r * (fx.low + fx.streams) + fx.low + s] / fx.streams);
            try expectClose(2.0 * bf(sigmoid(v)), got.inject[r * fx.streams + s]);
        }
        for (0..fx.dims) |d| {
            var total: f32 = 0;
            for (0..fx.streams) |s| {
                const at = (r * fx.streams + s) * fx.dims + d;
                total += bf(bf(sigmoid(numeric.f32FromBf16(in.up[at]))) * numeric.f32FromBf16(got.normed[at]));
            }
            try expectClose(total / fx.streams, got.mixed[r * fx.dims + d]);
        }
    }
}

/// A block's squares as the write-back sums them: fp32, halves folded onto halves from 128 apart down to 1.
fn blockSquares(values: []const u16) f32 {
    var squares: [block]f32 = undefined;
    for (&squares, values) |*square, bits| {
        const v = numeric.f32FromBf16(bits);
        square.* = v * v;
    }
    var half: usize = block / 2;
    while (half > 0) : (half /= 2) {
        for (0..half) |i| squares[i] += squares[i + half];
    }
    return squares[0];
}

test "a block's squares are one tree of fp32 sums and a stream's norm their sum in order: the host's bits" {
    // The sums' order is part of a row's bits: a write-back or a read-out that is rearranged for speed must
    // still give these, which the host computes in the kernels' order (no value here goes through exp).
    const gpa = testing.allocator;
    const rows = 5;
    const in: Inputs = try .init(gpa, rows);
    defer in.deinit(gpa);
    const got = try run(gpa, in, rows);
    defer got.deinit(gpa);
    const chunks = fx.dims / block;
    for (0..rows) |r| for (0..fx.streams) |s| {
        var total: f32 = 0;
        for (0..chunks) |c| {
            const at = (r * fx.streams + s) * fx.dims + c * block;
            const part = blockSquares(in.h[at..][0..block]);
            try testing.expectEqual(@as(u32, @bitCast(part)), @as(u32, @bitCast(got.pss[(r * chunks + c) * fx.streams + s])));
            try testing.expectEqual(@as(u32, @bitCast(blockSquares(got.after_slots[at..][0..block]))), @as(u32, @bitCast(got.pss_slots[(r * chunks + c) * fx.streams + s])));
            total += part;
        }
        const rinv = 1.0 / @sqrt(total / fx.dims + @as(f32, fx.eps));
        for (0..fx.dims) |d| {
            const e = s * fx.dims + d;
            try testing.expectEqual(numeric.bf16FromF32(numeric.f32FromBf16(in.h[r * fx.wide + e]) * rinv * in.scale[e]), got.normed[r * fx.wide + e]);
        }
    };
}

test "a write-back is bf16(h + branch * inject), the product unrounded: TensorFold's compiled kernel, bit for bit" {
    // glue.py's source rounds the product to bf16 before the sum; the kernel Triton compiles from it does not, as
    // TensorFold's exported streams show (zig/kernels/qwen4/hc.cu). The host's arithmetic here is exact in the
    // same order (the product of two bf16 values is exact in fp32; the slots' sum is a chain of fma), so the
    // bits are compared, and the source's two roundings must give other bits on this fixture.
    const gpa = testing.allocator;
    const rows = 3;
    const in: Inputs = try .init(gpa, rows);
    defer in.deinit(gpa);
    const got = try run(gpa, in, rows);
    defer got.deinit(gpa);

    var twice_differs: usize = 0;
    for (0..rows) |r| {
        for (0..fx.streams) |s| {
            const inj = numeric.f32FromBf16(got.inject[r * fx.streams + s]);
            for (0..fx.dims) |d| {
                const at = (r * fx.streams + s) * fx.dims + d;
                const h = numeric.f32FromBf16(in.h[at]);
                const branch = numeric.f32FromBf16(in.branch[r * fx.dims + d]);
                const first = numeric.bf16FromF32(h + branch * inj);
                try testing.expectEqual(first, got.after_branch[at]);
                twice_differs += @intFromBool(numeric.bf16FromF32(h + bf(branch * inj)) != first);

                var acc: f32 = 0;
                for (0..fx.slots - 1) |k| acc = @mulAdd(f32, in.y[(r * fx.slots + k) * fx.dims + d], in.weights[r * fx.slots + k], acc);
                acc = @mulAdd(f32, in.shared[r * fx.dims + d], in.weights[r * fx.slots + fx.slots - 1], acc);
                const base = numeric.f32FromBf16(first);
                const second = numeric.bf16FromF32(base + bf(acc) * inj);
                try testing.expectEqual(second, got.after_slots[at]);
                twice_differs += @intFromBool(numeric.bf16FromF32(base + bf(bf(acc) * inj)) != second);
            }
        }
    }
    // the two arithmetics part on a good share of the fixture's values, so the bits above tell them apart
    try testing.expect(twice_differs > rows * fx.wide / 20);
}

test "a row's hyper-connection bits do not depend on the rows beside it" {
    const gpa = testing.allocator;
    const in: Inputs = try .init(gpa, 5);
    defer in.deinit(gpa);
    const alone = try run(gpa, in, 1);
    defer alone.deinit(gpa);
    const among = try run(gpa, in, 5);
    defer among.deinit(gpa);
    try testing.expectEqualSlices(u16, alone.normed, among.normed[0..fx.wide]);
    try testing.expectEqualSlices(u16, alone.act, among.act[0..fx.low]);
    try testing.expectEqualSlices(u16, alone.inject, among.inject[0..fx.streams]);
    try testing.expectEqualSlices(u16, alone.mixed, among.mixed[0..fx.dims]);
    try testing.expectEqualSlices(u16, alone.after_branch, among.after_branch[0..fx.wide]);
    try testing.expectEqualSlices(u16, alone.after_slots, among.after_slots[0..fx.wide]);
}

/// `hc.site` against `writeback` then `normed` on the same inputs, `rows` rows from row `first`: h, the squares,
/// and the normed rows, which it returns.
fn siteAgainstLaunches(gpa: std.mem.Allocator, mode: usize, own_shared: bool, first: usize, rows: usize, all: usize) ![]u16 {
    const dims = 2560;
    const streams = 4;
    const slots = 11;
    const wide = streams * dims;
    var prng: std.Random.DefaultPrng = .init(733);
    const random = prng.random();
    const h_host = try gpa.alloc(u16, all * wide);
    defer gpa.free(h_host);
    randomBf16(random, h_host, 2.0);
    const branch_host = try gpa.alloc(u16, all * dims);
    defer gpa.free(branch_host);
    randomBf16(random, branch_host, 1.0);
    const inject_host = try gpa.alloc(u16, all * streams);
    defer gpa.free(inject_host);
    randomBf16(random, inject_host, 1.0);
    const y_host = try gpa.alloc(f32, all * slots * dims);
    defer gpa.free(y_host);
    for (y_host) |*x| x.* = random.floatNorm(f32);
    const shared_host = try gpa.alloc(f32, all * dims);
    defer gpa.free(shared_host);
    for (shared_host) |*x| x.* = random.floatNorm(f32);
    const weights_host = try gpa.alloc(f32, all * slots);
    defer gpa.free(weights_host);
    for (weights_host) |*x| x.* = random.float(f32);
    const scale_host = try gpa.alloc(f32, wide);
    defer gpa.free(scale_host);
    for (scale_host) |*x| x.* = 1.0 + 0.1 * random.floatNorm(f32);

    var branch = try device(u16, .bf16, &.{ rows, dims }, branch_host[first * dims ..][0 .. rows * dims]);
    defer branch.deinit();
    var inject = try device(u16, .bf16, &.{ rows, streams }, inject_host[first * streams ..][0 .. rows * streams]);
    defer inject.deinit();
    var y = try device(f32, .f32, &.{ rows * slots, dims }, y_host[first * slots * dims ..][0 .. rows * slots * dims]);
    defer y.deinit();
    var shared = try device(f32, .f32, &.{ rows, dims }, shared_host[first * dims ..][0 .. rows * dims]);
    defer shared.deinit();
    var weights = try device(f32, .f32, &.{ rows, slots }, weights_host[first * slots ..][0 .. rows * slots]);
    defer weights.deinit();
    var scale = try device(f32, .f32, &.{wide}, scale_host);
    defer scale.deinit();
    // mode 3: the slots already summed, a chain of fma in slot order on the host (what the retired EXL3 experts'
    // `combine` launched), which is mode 2's bits
    const sum_host = try gpa.alloc(f32, rows * dims);
    defer gpa.free(sum_host);
    for (0..rows) |r| for (0..dims) |d| {
        var acc: f32 = 0;
        for (0..slots) |s| acc = @mulAdd(f32, y_host[((first + r) * slots + s) * dims + d], weights_host[(first + r) * slots + s], acc);
        sum_host[r * dims + d] = acc;
    };
    var sum = try device(f32, .f32, &.{ rows, dims }, sum_host);
    defer sum.deinit();
    const pending: hc.Pending = switch (mode) {
        0 => .none,
        1 => .{ .branch = .{ .branch = branch.view, .inject = inject.view } },
        3 => .{ .sum = .{ .sum = sum.view, .inject = inject.view } },
        else => .{ .slots = .{ .y = y.view, .shared = if (own_shared) shared.view else null, .weights = weights.view, .inject = inject.view } },
    };
    var results: [2]struct { h: []u16, pss: []f32, normed: []u16 } = undefined;
    var made: usize = 0;
    defer for (results[0..made]) |r| {
        gpa.free(r.h);
        gpa.free(r.pss);
        gpa.free(r.normed);
    };
    for (&results, 0..) |*result, joined| {
        var h = try device(u16, .bf16, &.{ rows, wide }, h_host[first * wide ..][0 .. rows * wide]);
        defer h.deinit();
        var pss: cuda.Buffer = try .init(.f32, &.{ rows, dims / block, streams });
        defer pss.deinit();
        var out: cuda.Buffer = try .init(.bf16, &.{ rows, wide });
        defer out.deinit();
        if (joined == 1) {
            try hc.site(h.view, pss.view, pending, scale.view, out.view, 1e-6, streams);
        } else {
            try writeback(h.view, pss.view, pending, streams);
            try normed(h.view, pss.view, scale.view, out.view, 1e-6, streams);
        }
        result.h = try fetch(u16, gpa, h);
        errdefer gpa.free(result.h);
        result.pss = try fetch(f32, gpa, pss);
        errdefer gpa.free(result.pss);
        result.normed = try fetch(u16, gpa, out);
        made += 1;
    }
    try testing.expectEqualSlices(u16, results[0].h, results[1].h);
    try testing.expectEqualSlices(u32, @ptrCast(results[0].pss), @ptrCast(results[1].pss));
    try testing.expectEqualSlices(u16, results[0].normed, results[1].normed);
    return gpa.dupe(u16, results[1].normed);
}

test "the write-back and the norm as one launch give the two launches' bits in every mode, a row alone as among others" {
    const gpa = testing.allocator;
    const all = 5;
    const wide = 4 * 2560;
    for ([_]struct { mode: usize, own: bool }{ .{ .mode = 0, .own = false }, .{ .mode = 1, .own = false }, .{ .mode = 2, .own = false }, .{ .mode = 2, .own = true } }) |case| {
        const window = try siteAgainstLaunches(gpa, case.mode, case.own, 0, all, all);
        defer gpa.free(window);
        for (0..all) |row| {
            const alone = try siteAgainstLaunches(gpa, case.mode, case.own, row, 1, all);
            defer gpa.free(alone);
            try testing.expectEqualSlices(u16, window[row * wide ..][0..wide], alone);
        }
    }
    // the slots summed before the write-back (`hc.Pending.sum`) are the slots' mode, bit for bit
    for ([_]usize{ 1, all }) |rows| {
        const slots = try siteAgainstLaunches(gpa, 2, false, 0, rows, all);
        defer gpa.free(slots);
        const summed = try siteAgainstLaunches(gpa, 3, false, 0, rows, all);
        defer gpa.free(summed);
        try testing.expectEqualSlices(u16, slots, summed);
    }
}
