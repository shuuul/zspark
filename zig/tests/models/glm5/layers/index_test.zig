//! `index.zig`: the cache's bytes, the pool's rounding, the rotation and the FP8 coding, and the selection's ties
//! and tail.

const std = @import("std");
const testing = std.testing;
const index = @import("../../../../src/models/glm5/layers/index.zig");
const numeric = @import("../../../../src/weights/numeric.zig");

const dim = 8;

fn bits(value: f32) u16 {
    return numeric.bf16FromF32(value);
}

test "the cache's bytes a token and a rank's, by item" {
    const bf16_item = index.item(.bf16);
    try testing.expectEqual(@as(u64, 1024), bf16_item.latent);
    try testing.expectEqual(@as(u64, 256), bf16_item.key);
    try testing.expectEqual(@as(u64, 256), bf16_item.gate);
    // a pooled key: 128 codes and an fp32 scale over four tokens (64, bf16, before spec 116 WS-05)
    try testing.expectEqual(@as(u64, 33), bf16_item.pooled);
    try testing.expectEqual(@as(u64, 1569), bf16_item.aToken());
    try testing.expectEqual(@as(u64, 18828), index.bytesAToken(.bf16));
    try testing.expectEqual(@as(u64, 1057), index.item(.fp8_latent).aToken());
    try testing.expectEqual(@as(u64, 12684), index.bytesAToken(.fp8_latent));

    const cases = [_]u64{ 2051, 81920, 262144, 1048576 };
    const bf16_rank = [_]u64{ 38616228, 1542389760, 4935647232, 19742588928 };
    const fp8_rank = [_]u64{ 26014884, 1039073280, 3325034496, 13300137984 };
    for (cases, bf16_rank, fp8_rank) |tokens, bf16_bytes, fp8_bytes| {
        try testing.expectEqual(bf16_bytes, index.rankBytes(tokens, .bf16));
        try testing.expectEqual(fp8_bytes, index.rankBytes(tokens, .fp8_latent));
        try testing.expectEqual(bf16_bytes, index.item(.bf16).aToken() * index.caches * tokens);
    }
    try testing.expectEqual(@as(usize, 2051), index.denseLimit(2048, 4));
    try testing.expectEqual(@as(usize, 512), index.keptPools(2048, 4));
    try testing.expectEqual(@as(usize, 2051), index.listWidth(512, 4));

    var text: [800]u8 = undefined;
    var w: std.Io.Writer = .fixed(&text);
    try index.writeCache(&w, 262144);
    const printed = w.buffered();
    try testing.expect(std.mem.indexOf(u8, printed, "18828 over 12 caches") != null);
    try testing.expect(std.mem.indexOf(u8, printed, "4935647232") != null);
    try testing.expect(std.mem.indexOf(u8, printed, "3325034496") != null);
    try testing.expect(std.mem.indexOf(u8, printed, "1024 a token") != null);
    try testing.expect(std.mem.indexOf(u8, printed, "33 a token") != null);
}

test "a pool is written when its fourth token arrives, and not before" {
    try testing.expectEqual(@as(usize, 0), index.poolsCompleted(0, 3, 4).count);
    try testing.expectEqual(@as(usize, 1), index.poolsCompleted(0, 4, 4).count);
    const across = index.poolsCompleted(2, 6, 4);
    try testing.expectEqual(@as(usize, 0), across.first);
    try testing.expectEqual(@as(usize, 2), across.count);
    const later = index.poolsCompleted(5, 3, 4);
    try testing.expectEqual(@as(usize, 1), later.first);
    try testing.expectEqual(@as(usize, 1), later.count);
}

test "the pool's softmax sums in fp32 and rounds once, as vLLM's" {
    var keys: [4][dim]u16 = undefined;
    var gates: [4][dim]u16 = undefined;
    var ape: [4 * dim]u16 = undefined;
    for (0..4) |t| {
        for (0..dim) |d| {
            keys[t][d] = bits(@as(f32, @floatFromInt(t + 1)) * 0.1 + @as(f32, @floatFromInt(d)) * 0.01);
            gates[t][d] = bits(@as(f32, @floatFromInt(d)) * 0.05 - @as(f32, @floatFromInt(t)) * 0.2);
            ape[t * dim + d] = bits(@as(f32, @floatFromInt(t)) * 0.01);
        }
    }
    var out: [dim]u16 = undefined;
    index.poolRow(.{ &keys[0], &keys[1], &keys[2], &keys[3] }, .{ &gates[0], &gates[1], &gates[2], &gates[3] }, &ape, &out);
    // one column by hand, the kernel's order
    const d: usize = 3;
    var logit: [4]f32 = undefined;
    var highest: f32 = -std.math.inf(f32);
    for (0..4) |t| {
        logit[t] = numeric.f32FromBf16(gates[t][d]) + numeric.f32FromBf16(ape[t * dim + d]);
        highest = @max(highest, logit[t]);
    }
    var numerator: f32 = 0;
    var denominator: f32 = 0;
    for (0..4) |t| {
        const weight = @exp(logit[t] - highest);
        numerator += numeric.f32FromBf16(keys[t][d]) * weight;
        denominator += weight;
    }
    try testing.expectEqual(numeric.bf16FromF32(numerator / denominator), out[d]);
}

test "the rotation is the Walsh-Hadamard transform, and the coding a power-of-two scale" {
    // a one-hot vector rotates to +-1 / sqrt(16) in the Hadamard row's signs
    var v: [16]f32 = @splat(0);
    v[3] = 1;
    index.rotate(&v);
    for (v, 0..) |x, i| {
        const sign: f32 = if (@popCount(i & 3) % 2 == 0) 1 else -1;
        try testing.expectEqual(sign * 0.25, x);
    }
    // the largest magnitude 0.25: 0.25 / 448 lies in (2^-11, 2^-10], so the scale is 2^-10 and the code 256
    var codes: [16]u8 = undefined;
    const scale = index.quantize(&v, &codes);
    try testing.expectEqual(std.math.ldexp(@as(f32, 1), -10), scale);
    for (codes, v) |code, x| try testing.expectEqual(x / scale, numeric.f32FromE4m3(code));
    // at a power of two the scale is the ratio itself, not twice it
    var exact = [_]f32{ 448.0 * 0.5, 1 };
    var two: [2]u8 = undefined;
    try testing.expectEqual(@as(f32, 0.5), index.quantize(&exact, &two));
    try testing.expectEqual(@as(f32, 448), numeric.f32FromE4m3(two[0]));
    // nothing below 1e-4 sets the scale
    var zeros: [4]f32 = @splat(0);
    var four: [4]u8 = undefined;
    try testing.expectEqual(std.math.ldexp(@as(f32, 1), -22), index.quantize(&zeros, &four));
}

test "ties keep the lower pool, and the unfinished pool's tokens are always listed" {
    const gpa = testing.allocator;
    // eight pools, keep two: pools 1 and 4 tie for the best, 6 is next, the rest lose
    const scores = [_]f32{ 0.1, 1.0, 0.2, 0.0, 1.0, -1.0, 0.5, 0.3 };
    var selected: [2]u32 = undefined;
    try index.selectPools(gpa, &scores, 2, &selected);
    try testing.expectEqualSlices(u32, &.{ 1, 4 }, &selected);

    // position 10: complete pools (10+1)/4 = 2, so a keep of 2 is still dense (count 0) and the tail is token 8, 9, 10
    var tokens: [2 * 4 + 3]i32 = undefined;
    const dense = index.selectTokens(&selected, 10, 4, &tokens);
    try testing.expectEqual(@as(usize, 0), dense);
    try testing.expectEqualSlices(i32, &.{ 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10 }, &tokens);

    // position 11 completes pool 2: (12)/4 = 3 > 2, sparse, and the tail is empty
    const sparse = index.selectTokens(&selected, 11, 4, &tokens);
    try testing.expectEqual(@as(usize, 8), sparse);
    try testing.expectEqual(@as(i32, -1), tokens[8]);
}

test "a row's list ignores pools that are not complete at its position" {
    const gpa = testing.allocator;
    var scores = [_]f32{ 0.2, 3.0, 0.4, 9.0, -std.math.inf(f32), -std.math.inf(f32) };
    var early: [2]u32 = undefined;
    try index.selectPools(gpa, &scores, 2, &early);
    try testing.expectEqualSlices(u32, &.{ 1, 3 }, &early);
    scores[4] = 100.0;
    scores[5] = 100.0;
    var later: [2]u32 = undefined;
    try index.selectPools(gpa, scores[0..4], 2, &later);
    try testing.expectEqualSlices(u32, &early, &later);
}
