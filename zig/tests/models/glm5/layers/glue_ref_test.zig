//! `glue_ref.zig`'s tests: the order of a fold, what a read-out's gates are, a selection's ties.

const std = @import("std");
const testing = std.testing;
const numeric = @import("../../../../src/weights/numeric.zig");
const host = @import("../../../../src/models/glm5/layers/glue_ref.zig");
const shared = @import("../../../../src/decode/glue_ref.zig");

test "a fold adds the upper half onto the lower, not one value after another" {
    var four = [_]f32{ 1e8, 1, -1e8, 1 };
    try testing.expectEqual(@as(f32, 2), host.fold(&four));
    // 16777216 + 1 is lost one after another; folded, the ones meet first
    var eight = [_]f32{ 16777216, 1, 0, 0, 0, 1, 0, 0 };
    try testing.expectEqual(@as(f32, 16777218), host.fold(&eight));
    var one = [_]f32{7};
    try testing.expectEqual(@as(f32, 7), host.fold(&one));
}

test "a read-out's comb sums to one by row and by column, and equal streams collapse to themselves" {
    var prng: std.Random.DefaultPrng = .init(3);
    const random = prng.random();
    const dims = 512;
    var x: [4 * dims]u16 = undefined;
    for (x[0..dims]) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    for (1..4) |s| @memcpy(x[s * dims ..][0..dims], x[0..dims]);
    var dots: [host.mix * 4 * dims]u16 = undefined;
    for (&dots) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.05);
    var base: [host.mix]f32 = undefined;
    for (&base) |*b| b.* = random.floatNorm(f32);
    var norm: [dims]u16 = undefined;
    @memset(&norm, numeric.bf16FromF32(1));
    var part: [host.parts * host.entry]f32 = undefined;
    host.hcMix(&x, &dots, &part);
    var squares: f64 = 0;
    for (x) |bits| squares += @as(f64, numeric.f32FromBf16(bits)) * numeric.f32FromBf16(bits);
    var summed: f64 = 0;
    for (0..host.parts) |b| summed += part[b * host.entry + host.mix];
    try testing.expectApproxEqRel(squares, summed, 1e-5);

    var out: [dims]u16 = undefined;
    var post: [4]f32 = undefined;
    var comb: [16]f32 = undefined;
    var scratch: [dims]f32 = undefined;
    host.hcPre(&x, &part, .{ .base = &base, .scale = &.{ 0.7, 1.3, 0.9 }, .norm = .{ .bf16 = &norm }, .eps = 1e-5, .hc_eps = 1e-6, .iters = 20 }, &out, &post, &comb, &scratch);
    for (0..4) |i| {
        var row: f32 = 0;
        var column: f32 = 0;
        for (0..4) |j| {
            row += comb[i * 4 + j];
            column += comb[j * 4 + i];
        }
        // the columns are brought to one last, the rows as near as twenty rounds bring them
        try testing.expectApproxEqAbs(@as(f32, 1), row, 2e-3);
        try testing.expectApproxEqAbs(@as(f32, 1), column, 1e-4);
        try testing.expect(post[i] > 0 and post[i] < 2);
    }
    // the normed row has rms 1 under a weight of ones
    var rms: f64 = 0;
    for (out) |bits| rms += @as(f64, numeric.f32FromBf16(bits)) * numeric.f32FromBf16(bits);
    try testing.expectApproxEqAbs(@as(f64, 1), @sqrt(rms / dims), 2e-3);
}

test "a selection takes the lower id among equals, by score and bias together, and weighs by the score alone" {
    // 1 and 4 tie by logit; 2's bias lifts it over both; 5 ties 0 through its bias
    const logits = [_]f32{ 0, 2, 0, -1, 2, -30 };
    const bias = [_]f32{ 0, 0, 0.75, 0, 0, 0.5 };
    var picks: [5]i32 = undefined;
    var weights: [5]f32 = undefined;
    host.select(&logits, &bias, &picks, &weights, 2.5, true);
    try testing.expectEqualSlices(i32, &.{ 2, 1, 4, 0, 6 }, &picks);
    const s2: f32 = 0.5;
    const s1: f32 = 1.0 / (1.0 + @exp(@as(f32, -2)));
    const total = ((s2 + s1) + s1) + 0.5;
    try testing.expectApproxEqRel(s2 / total * 2.5, weights[0], 1e-6);
    try testing.expectApproxEqRel(s1 / total * 2.5, weights[1], 1e-6);
    try testing.expectEqual(weights[1], weights[2]);
    try testing.expectEqual(@as(f32, 1), weights[4]);
    host.select(&logits, &bias, &picks, &weights, 1, false);
    try testing.expectEqual(s2, weights[0]);
}

test "SwiGLU holds the gate under the limit and the up within it; a share is the slots in order" {
    const f = numeric.bf16FromF32;
    const gu = [_]u16{ f(50), f(-50), f(1), f(3), f(-40), f(40), f(2), f(-2) };
    var act: [4]u16 = undefined;
    shared.swiglu(&gu, &act, 10);
    const v = numeric.f32FromBf16;
    try testing.expectApproxEqRel(@as(f32, 10 * -10), v(act[0]), 1e-2);
    try testing.expect(@abs(v(act[1])) < 1e-15);
    try testing.expectApproxEqRel(@as(f32, 0.7310586 * 2), v(act[2]), 1e-2);
    const y = [_]f32{ 1, 2, 10, 20, 100, 200 };
    var out: [2]f32 = undefined;
    shared.combine(&y, &.{ 0.5, 0.25, 1 }, &out);
    try testing.expectEqualSlices(f32, &.{ 103, 206 }, &out);
}
