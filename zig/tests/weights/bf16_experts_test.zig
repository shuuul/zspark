//! `bf16.zig`'s `Experts` (`zig/kernels/bf16/experts.cu`) on random weights: the up product's SwiGLU and the down
//! product against fp64 sums with the kernel's roundings, and a row alone against the row among others, across
//! calls whose plans take one block (up to 1,024 pairs) and several.

const std = @import("std");
const testing = std.testing;
const bf16 = @import("../../src/weights/bf16.zig");
const cuda = @import("../../src/cuda/cuda.zig");
const experts_file = @import("../../src/weights/experts.zig");
const numeric = @import("../../src/weights/numeric.zig");
const numeric_fixture = @import("numeric_fixture.zig");

const fx = struct {
    const experts = 8;
    const hidden = 128;
    const width = 64;
    const top = 3;

    fn bf(v: f64) f64 {
        return numeric.roundBf16(@floatCast(v));
    }

    fn of(bits: u16) f64 {
        return numeric.f32FromBf16(bits);
    }

    fn upload(dtype: cuda.DType, shape: []const usize, bytes: []const u8) !cuda.Buffer {
        var buffer: cuda.Buffer = try .init(dtype, shape);
        errdefer buffer.deinit();
        try buffer.upload(bytes);
        return buffer;
    }

    const Run = struct { act: []u16, y: []u16 };

    /// The two products for `rows` rows of `x` and their picks; the act rows put back in pair order.
    fn run(gpa: std.mem.Allocator, x: []const u16, picks: []const i32, rows: usize, gu: cuda.Buffer, down: cuda.Buffer) !Run {
        const pairs = rows * top;
        var xb = try upload(.bf16, &.{ rows, hidden }, std.mem.sliceAsBytes(x[0 .. rows * hidden]));
        defer xb.deinit();
        var pb = try upload(.i32, &.{ rows, top }, std.mem.sliceAsBytes(picks[0..pairs]));
        defer pb.deinit();
        var plan: experts_file.Plan = try .init(rows, top, experts);
        defer plan.deinit();
        try plan.routeAs(pb.view, rows, .{ .item = bf16.Experts.item, .tiles = 4 });
        var act: cuda.Buffer = try .init(.bf16, &.{ pairs, width });
        defer act.deinit();
        var y: cuda.Buffer = try .init(.bf16, &.{ pairs, hidden });
        defer y.deinit();
        try bf16.Experts.up(xb.view, gu.view, plan.members.view, plan.items.view, plan.counts.view, top, act.view);
        try bf16.Experts.down(act.view, down.view, plan.members.view, plan.items.view, plan.counts.view, y.view);
        try cuda.synchronize();
        const members = try gpa.alloc(i32, pairs);
        defer gpa.free(members);
        try plan.members.downloadAt(0, std.mem.sliceAsBytes(members));
        const placed = try gpa.alloc(u16, pairs * width);
        defer gpa.free(placed);
        try act.download(std.mem.sliceAsBytes(placed));
        const out: Run = .{ .act = try gpa.alloc(u16, pairs * width), .y = try gpa.alloc(u16, pairs * hidden) };
        for (members, 0..) |pair, place| @memcpy(out.act[@as(usize, @intCast(pair)) * width ..][0..width], placed[place * width ..][0..width]);
        try y.download(std.mem.sliceAsBytes(out.y));
        return out;
    }
};

test "bf16 grouped experts against fp64 sums, and a row alone against the row among others" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(137);
    const random = prng.random();
    const E = fx.experts;
    const H = fx.hidden;
    const W = fx.width;
    const most = 400; // 1,200 pairs: a plan of several blocks
    const gu_bits = try gpa.alloc(u16, E * 2 * W * H);
    defer gpa.free(gu_bits);
    numeric_fixture.randomBf16(random, gu_bits, 0.1);
    const down_bits = try gpa.alloc(u16, E * H * W);
    defer gpa.free(down_bits);
    numeric_fixture.randomBf16(random, down_bits, 0.1);
    const x = try gpa.alloc(u16, most * H);
    defer gpa.free(x);
    numeric_fixture.randomBf16(random, x, 1.0);
    const picks = try gpa.alloc(i32, most * fx.top);
    defer gpa.free(picks);
    for (0..most) |r| {
        var order: [E]i32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        random.shuffle(i32, &order);
        // rows that crowd expert 0, so that one expert's items run past one
        if (r % 3 == 0) order[0] = 0;
        var seen: [E]bool = @splat(false);
        var k: usize = 0;
        for (order) |e| {
            if (k == fx.top) break;
            if (seen[@intCast(e)]) continue;
            seen[@intCast(e)] = true;
            picks[r * fx.top + k] = e;
            k += 1;
        }
    }
    var gu = try fx.upload(.bf16, &.{ E, 2 * W, H }, std.mem.sliceAsBytes(gu_bits));
    defer gu.deinit();
    var down = try fx.upload(.bf16, &.{ E, H, W }, std.mem.sliceAsBytes(down_bits));
    defer down.deinit();

    const whole = try fx.run(gpa, x, picks, most, gu, down);
    defer gpa.free(whole.act);
    defer gpa.free(whole.y);
    // the host's: fp64 sums, the kernel's roundings; a sum's error stated against its summed magnitudes
    var worst_act: f64 = 0;
    var worst_y: f64 = 0;
    for (0..most * fx.top) |pair| {
        const r = pair / fx.top;
        const e: usize = @intCast(picks[pair]);
        var act: [W]f64 = undefined;
        for (0..W) |j| {
            var g: f64 = 0;
            var u: f64 = 0;
            var mag: f64 = 0;
            for (0..H) |i| {
                const xv = fx.of(x[r * H + i]);
                const gw = fx.of(gu_bits[(e * 2 * W + j) * H + i]);
                const uw = fx.of(gu_bits[(e * 2 * W + W + j) * H + i]);
                g += xv * gw;
                u += xv * uw;
                mag += @abs(xv) * (@abs(gw) + @abs(uw));
            }
            const gv = fx.bf(g);
            act[j] = fx.bf(fx.bf(gv / (1.0 + @exp(-gv))) * fx.bf(u));
            const got = fx.of(whole.act[pair * W + j]);
            const err = @abs(got - act[j]);
            worst_act = @max(worst_act, err / (mag + 1e-30));
            try testing.expect(err <= mag / 64.0 + 1e-6);
        }
        for (0..H) |n| {
            var sum: f64 = 0;
            var mag: f64 = 0;
            for (0..W) |j| {
                const a = fx.of(whole.act[pair * W + j]);
                const w = fx.of(down_bits[(e * H + n) * W + j]);
                sum += a * w;
                mag += @abs(a * w);
            }
            const err = @abs(fx.of(whole.y[pair * H + n]) - sum);
            worst_y = @max(worst_y, err / (mag + 1e-30));
            try testing.expect(err <= mag / 128.0 + 1e-6);
        }
    }
    // a row alone and in calls of other sizes: the same bits
    for ([_]usize{ 1, 7, 50 }) |rows| {
        const part = try fx.run(gpa, x, picks, rows, gu, down);
        defer gpa.free(part.act);
        defer gpa.free(part.y);
        try testing.expectEqualSlices(u16, whole.act[0 .. rows * fx.top * W], part.act);
        try testing.expectEqualSlices(u16, whole.y[0 .. rows * fx.top * H], part.y);
    }
    const alone = try fx.run(gpa, x[(most - 1) * H ..], picks[(most - 1) * fx.top ..], 1, gu, down);
    defer gpa.free(alone.act);
    defer gpa.free(alone.y);
    try testing.expectEqualSlices(u16, whole.y[(most - 1) * fx.top * H ..], alone.y);
}
