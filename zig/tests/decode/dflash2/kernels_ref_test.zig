//! `kernels_ref.zig`'s tests: the rotary phases and the exponential against the library's in fp64, the convolution's
//! roundings, which positions a query sees.

const std = @import("std");
const testing = std.testing;
const numeric = @import("../../../src/weights/numeric.zig");
const host = @import("../../../src/decode/dflash2/kernels_ref.zig");

const one = numeric.bf16FromF32(1.0);

test "DFlash2 host: inv_freq is theta^(-d / 64) to an fp32 rounding, and 1 at d = 0" {
    try testing.expectEqual(@as(f32, 1.0), host.invFreq(10000.0, 0));
    try testing.expectEqual(@as(f32, 0.01), host.invFreq(10000.0, 32));
    for (0..host.half) |d| {
        const want = std.math.pow(f64, 10000.0, -@as(f64, @floatFromInt(d)) / 64.0);
        try testing.expectApproxEqRel(want, @as(f64, host.invFreq(10000.0, d)), 6e-8);
    }
}

test "DFlash2 host: a phase's sine and cosine are the library's to an fp32 rounding, in every quadrant and far out" {
    const start = host.sinCos(0.0);
    try testing.expectEqual(@as(f32, 0.0), start.sin);
    try testing.expectEqual(@as(f32, 1.0), start.cos);
    var prng: std.Random.DefaultPrng = .init(11);
    const random = prng.random();
    var off: usize = 0;
    const count = 200_000;
    for (0..count) |i| {
        // a third near zero, a third over a few turns, a third to 2^24
        const top: f32 = switch (i % 3) {
            0 => 1.0,
            1 => 64.0,
            else => 16777216.0,
        };
        const x: f32 = random.float(f32) * top;
        const got = host.sinCos(x);
        const sin: f32 = @floatCast(@sin(@as(f64, x)));
        const cos: f32 = @floatCast(@cos(@as(f64, x)));
        try testing.expectApproxEqAbs(sin, got.sin, 6e-8);
        try testing.expectApproxEqAbs(cos, got.cos, 6e-8);
        if (sin != got.sin or cos != got.cos) off += 1;
    }
    // the fp64 values are a few 1e-16 apart: their fp32 roundings part on a tie alone
    try testing.expect(off <= 2);
}

test "DFlash2 host: a convolution rounds its kernel to bf16 before the product, and the first row has no tap behind it" {
    const dims = 4;
    const groups = 2;
    var base: [4 * dims]u16 = @splat(one);
    var dyn: [4 * groups]u16 = @splat(0);
    // branch 1, tap 0, group 0: 1 + 1.5 * 2^-9 is 1 in bf16; unrounded, 255 times it would round to 256
    dyn[2 * groups] = numeric.bf16FromF32(1.5 / 512.0);
    const x = [dims]u16{ numeric.bf16FromF32(255), numeric.bf16FromF32(255), one, one };
    const before = [dims]u16{ numeric.bf16FromF32(3), 0, 0, numeric.bf16FromF32(3) };
    var out: [dims]u16 = undefined;
    host.dconvRow(&x, null, &dyn, &base, 1, null, &out);
    try testing.expectEqualSlices(u16, &x, &out);
    // tap 1 of branch 1 is 1 everywhere: the row before is added
    host.dconvRow(&x, &before, &dyn, &base, 1, null, &out);
    try testing.expectEqual(@as(f32, 258), numeric.f32FromBf16(out[0]));
    try testing.expectEqual(@as(f32, 4), numeric.f32FromBf16(out[3]));
    // branch 0 reads its own taps: with its second tap 0 the row before is not seen
    for (base[dims .. 2 * dims]) |*tap| tap.* = 0;
    host.dconvRow(&x, &before, &dyn, &base, 0, null, &out);
    try testing.expectEqualSlices(u16, &x, &out);
    // the residual is added to the rounded product and the sum rounded again: 255 + 1/2 is 256 in bf16
    const res = [dims]u16{ numeric.bf16FromF32(0.5), 0, 0, 0 };
    host.dconvRow(&x, null, &dyn, &base, 1, &res, &out);
    try testing.expectEqual(@as(f32, 256), numeric.f32FromBf16(out[0]));
}

test "DFlash2 host: a query sees the context a window back and the whole block, or the block to itself when causal" {
    const block: host.Block = .{ .pos = 5000, .rows = 8, .window = 2047 };
    try testing.expectEqual(@as(usize, 2953), block.seen(0).first);
    try testing.expectEqual(@as(usize, 2956), block.seen(3).first);
    try testing.expectEqual(@as(usize, 5007), block.seen(3).last);
    const early: host.Block = .{ .pos = 100, .rows = 8, .window = 2047 };
    try testing.expectEqual(@as(usize, 0), early.seen(7).first);
    const narrow: host.Block = .{ .pos = 100, .rows = 8, .window = 2, .causal = true };
    try testing.expectEqual(@as(usize, 98), narrow.seen(0).first);
    try testing.expectEqual(@as(usize, 100), narrow.seen(5).first);
    try testing.expectEqual(@as(usize, 105), narrow.seen(5).last);
}

test "DFlash2 host: equal keys give the mean of the values a query sees" {
    const block: host.Block = .{ .pos = 3, .rows = 2, .window = 2 };
    const keys: [5 * host.head]u16 = @splat(one);
    var values: [5 * host.head]u16 = undefined;
    for (0..5) |p| @memset(values[p * host.head ..][0..host.head], numeric.bf16FromF32(@floatFromInt(1 + p)));
    const q: [host.head]u16 = @splat(one);
    var out: [host.head]u16 = undefined;
    // row 0 at position 3 sees 1, 2 and the block's 3, 4: values 2, 3, 4, 5
    host.attendRow(&q, &keys, &values, block, 0, &out);
    try testing.expectEqual(@as(f32, 3.5), numeric.f32FromBf16(out[0]));
    // row 1 at position 4 sees 2 and the block: values 3, 4, 5
    host.attendRow(&q, &keys, &values, block, 1, &out);
    try testing.expectEqual(@as(f32, 4), numeric.f32FromBf16(out[host.head - 1]));
}
