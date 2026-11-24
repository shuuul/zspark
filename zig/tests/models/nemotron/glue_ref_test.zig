//! `glue_ref.zig`'s tests: the transcendentals' accuracy, a tie in the top-k, the plan's and the feed's logic.

const std = @import("std");
const m = @import("../../../src/models/nemotron/glue_ref.zig");

const testing = std.testing;

fn ulps(got: f32, want: f64) f64 {
    const w: f32 = @floatCast(want);
    if (w == 0 or !std.math.isFinite(w)) return if (got == w) 0 else std.math.inf(f64);
    const e = std.math.frexp(w).exponent;
    const ulp = std.math.ldexp(@as(f64, 1.0), @max(e - 24, -149));
    return @abs(@as(f64, got) - want) / ulp;
}

test "exp stays within 1 ulp of the f64 exp, including subnormal and overflowing results" {
    var worst: f64 = 0;
    var x: f32 = -103.9;
    while (x < 88.72) : (x += 0.000377) worst = @max(worst, ulps(m.exp(x), @exp(@as(f64, x))));
    try testing.expect(worst <= 1.0);
    try testing.expectEqual(std.math.inf(f32), m.exp(88.73));
    try testing.expectEqual(@as(f32, 1.0), m.exp(0.0));
    try testing.expectEqual(@as(f32, 1.0), m.exp(-0.0));
    try testing.expectEqual(@as(f32, 0.0), m.exp(-200.0));
    try testing.expect(std.math.isNan(m.exp(std.math.nan(f32))));
}

test "log1pUnit stays within 1 ulp of the f64 log1p over [0, 1]" {
    var worst: f64 = 0;
    var u: f32 = 0;
    while (u <= 1.0) : (u += 0.0000037) worst = @max(worst, ulps(m.log1pUnit(u), std.math.log1p(@as(f64, u))));
    for ([_]f32{ 1e-30, 1e-10, 5.9604645e-8, 3.861785e-3, 0.49958834, 0.5, 0.50000006, 1.0 }) |v| worst = @max(worst, ulps(m.log1pUnit(v), std.math.log1p(@as(f64, v))));
    try testing.expect(worst <= 1.0);
    try testing.expectEqual(@as(f32, 0.0), m.log1pUnit(0.0));
}

test "bf16 rounding is round to nearest even, and a NaN is 0x7fff" {
    try testing.expectEqual(@as(u16, 0x3f80), m.f32ToBf16(1.0));
    try testing.expectEqual(@as(u16, 0x3f80), m.f32ToBf16(@bitCast(@as(u32, 0x3f808000)))); // a tie rounds to even
    try testing.expectEqual(@as(u16, 0x3f82), m.f32ToBf16(@bitCast(@as(u32, 0x3f818000))));
    try testing.expectEqual(@as(u16, 0x7fff), m.f32ToBf16(std.math.nan(f32)));
}

test "top-k picks by score plus bias, the lowest id on a tie, then the shared slots" {
    var part: [8]f32 = .{ 0, 2, -1, 2, 0.5, -3, 1, 2 };
    const bias: [8]f32 = .{ 0, 0, 0, 0, 0, 0, 3, 0 };
    var idx: [4]i32 = undefined;
    var wt: [4]f32 = undefined;
    m.topk(&part, &bias, &idx, &wt, 1, 2.0, 8, 1, 2, 4, false);
    try testing.expectEqualSlices(i32, &.{ 6, 1, 8, 9 }, &idx);
    try testing.expectEqual(m.sigmoid(1) * 2.0, wt[0]);
    try testing.expectEqual(@as(f32, 1), wt[3]);
}

test "serial feed records the token and advances pos, parity and keep" {
    const sampled: i32 = 41;
    var ids: [1]i32 = .{-1};
    var meta: [3]i32 = .{ 7, 1, 3 };
    var history: [16]i32 = @splat(-1);
    m.serialFeed(&.{sampled}, &ids, &meta, &history);
    try testing.expectEqual(@as(i32, 41), ids[0]);
    try testing.expectEqual(@as(i32, 41), history[8]);
    try testing.expectEqualSlices(i32, &.{ 8, 0, 1 }, &meta);
}

test "plan routed groups member order by expert and counts the used ones" {
    // 2 rows of 4 slots, the first 2 routed: picks at slots 2, 3 are skipped
    const picks: [8]i32 = .{ 5, 5, 0, 0, 3, 5, 0, 0 };
    var members: [4]i32 = @splat(-1);
    var items: [3 * (6 + 8)]i32 = @splat(-1);
    var counts: [2]i32 = undefined;
    m.planRouted(&picks, &members, &items, &counts, 2, 4, 2, 6, 2);
    // expert 3: pick 4; expert 5: picks 0, 1, 5 -> members [4, 0, 1, 5], items 3:2 and 5:{1,3}:2+1
    try testing.expectEqualSlices(i32, &.{ 4, 0, 1, 5 }, &members);
    try testing.expectEqualSlices(i32, &.{ 3, 0, 1, 5, 1, 2, 5, 3, 1 }, items[0..9]);
    try testing.expectEqualSlices(i32, &.{ 3, 2 }, &counts);
}

test "rest rows adds r x at the mapped rows, fp32 and bf16" {
    const k = 16;
    var x: [2 * k]u16 = undefined;
    var r: [3 * k]u16 = undefined;
    for (&x, 0..) |*v, i| v.* = m.f32ToBf16(@as(f32, @floatFromInt(i % 5)) * 0.25);
    for (&r, 0..) |*v, i| v.* = m.f32ToBf16(@as(f32, @floatFromInt(i % 7)) * 0.125 - 0.5);
    var y32: [6]f32 = .{ 1, 2, 3, 10, 20, 30 };
    var y16: [6]u16 = .{ m.f32ToBf16(1), m.f32ToBf16(2), m.f32ToBf16(3), m.f32ToBf16(10), m.f32ToBf16(20), m.f32ToBf16(30) };
    m.restRows(&x, k, &r, &y32, null, 3, 2, 1, 0, 3, k);
    m.restRows(&x, k, &r, null, &y16, 3, 2, 1, 0, 3, k);
    for (0..2) |rr| for (0..3) |j| {
        var dot: f64 = 0;
        for (0..k) |i| dot += @as(f64, m.bf16ToF32(r[j * k + i])) * m.bf16ToF32(x[rr * k + i]);
        const base: f64 = if (rr == 0) @floatFromInt(j + 1) else @floatFromInt((j + 1) * 10);
        try testing.expectApproxEqAbs(base + dot, y32[rr * 3 + j], 1e-3);
        try testing.expectApproxEqAbs(base + dot, m.bf16ToF32(y16[rr * 3 + j]), 4e-2);
    };
}

test "keyed greedy takes the highest value, the lowest id on a tie, and the top share" {
    const vals: [8]f32 = .{ 1.0, 3.0, 3.0, -2.0, 0.5, 0.5, 0.5, 0.5 };
    const ids: [8]i64 = .{ 9, 2, 7, 4, 40, 30, 20, 10 };
    var out: [2]i32 = undefined;
    var prob: [2]f32 = undefined;
    m.keyedGreedy(&vals, &ids, &out, &prob, 2, 4, 2);
    // row 0: the tie at 3.0 goes to the lower id 2; row 1: all tied, id 10
    try testing.expectEqualSlices(i32, &.{ 2, 10 }, &out);
    // both rows' kept two share the top value: 1 / (e^0 + e^0)
    try testing.expectApproxEqAbs(@as(f32, 0.5), prob[0], 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.5), prob[1], 1e-7);
}

/// A small Mamba layer for the host twins: 2 heads of 4 in one group, state 128.
const mamba_fx = struct {
    const s: m.Shape = .{ .proj = 8 + 264 + 2, .xd = 8, .cd = 264, .heads = 2, .dh = 4, .groups = 1, .rmax = 16 };

    const Run = struct {
        base: [3 * s.cd]u16 = @splat(0),
        raw: [2 * s.rmax * s.cd]u16 = @splat(0),
        xc: [2 * s.rmax * s.cd]u16 = @splat(0),
        dt: [2 * s.rmax * s.heads]f32 = @splat(0),
        state: [s.heads * s.dh * 128]f32 = @splat(0),
        parity: usize = 0,
        pk: usize = 0,

        /// conv then scan of a window, then TensorFold's verify (parity flips) and commit(keep).
        fn window(r: *Run, p: []const u16, y: []u16, rows: usize, keep: usize, w: Weights) void {
            m.conv(p, &r.base, &r.raw, &r.xc, &w.cw, &w.cb, r.parity, r.pk, rows, s);
            m.scan(p, &r.xc, &r.dt, &r.state, &w.a, &w.dsk, &w.dtb, y, r.parity, r.pk, rows, 0, std.math.inf(f32), s);
            r.parity ^= 1;
            r.pk = keep;
        }
    };

    const Weights = struct {
        cw: [4 * s.cd]f32,
        cb: [s.cd]f32,
        a: [s.heads]f32 = .{ -0.5, -3.0 },
        dsk: [s.heads]f32 = .{ 1.0, 0.25 },
        dtb: [s.heads]f32 = .{ -1.0, 0.5 },
    };
};

test "the host's Mamba window is its serial steps, and rejected rows leave the state alone" {
    var prng: std.Random.DefaultPrng = .init(157);
    const random = prng.random();
    const s = mamba_fx.s;
    var w: mamba_fx.Weights = .{ .cw = undefined, .cb = undefined };
    for (&w.cw) |*v| v.* = m.rbf(random.floatNorm(f32) * 0.3);
    for (&w.cb) |*v| v.* = m.rbf(random.floatNorm(f32) * 0.1);
    var p: [12 * s.proj]u16 = undefined;
    for (&p) |*v| v.* = m.f32ToBf16(random.floatNorm(f32));
    var ya: [12 * s.xd]u16 = undefined;
    var yb: [12 * s.xd]u16 = undefined;
    // a: a window of 6 keeping 2, then a window of rows 6..10, then a step of row 10
    var a: mamba_fx.Run = .{};
    a.window(p[0 .. 6 * s.proj], ya[0 .. 6 * s.xd], 6, 2, w);
    a.window(p[6 * s.proj .. 10 * s.proj], ya[6 * s.xd .. 10 * s.xd], 4, 4, w);
    a.window(p[10 * s.proj .. 11 * s.proj], ya[10 * s.xd .. 11 * s.xd], 1, 1, w);
    // b: rows 0, 1, then 6..11 as serial steps
    var b: mamba_fx.Run = .{};
    for ([_]usize{ 0, 1, 6, 7, 8, 9, 10 }) |r| b.window(p[r * s.proj ..][0..s.proj], yb[r * s.xd ..][0..s.xd], 1, 1, w);
    try testing.expectEqualSlices(u16, ya[0 .. 2 * s.xd], yb[0 .. 2 * s.xd]);
    try testing.expectEqualSlices(u16, ya[6 * s.xd .. 11 * s.xd], yb[6 * s.xd .. 11 * s.xd]);
    try testing.expectEqualSlices(f32, &a.state, &b.state);
    try testing.expectEqualSlices(u16, &a.base, &b.base);
}
