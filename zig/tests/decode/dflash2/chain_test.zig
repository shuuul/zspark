//! `chain.zig`'s tests on small made candidates. TensorFold's picks on recorded ones are WS-13's fixture.

const std = @import("std");
const testing = std.testing;
const sampling = @import("../../../src/decode/sampling.zig");
const chain = @import("../../../src/decode/dflash2/chain.zig");

test "the merge keeps the largest of both ranks' candidates, equal logits by token id" {
    const halves = [_]chain.Half{
        .{ .values = &.{ 5.0, 3.0, 3.0, 1.0 }, .tokens = &.{ 7, 9, 2, 4 } },
        .{ .values = &.{ 6.0, 3.0, 2.5, 0.5 }, .tokens = &.{ 107, 101, 103, 100 } },
    };
    var tokens: [4]u32 = undefined;
    var values: [4]f64 = undefined;
    chain.merge(&halves, &tokens, &values);
    try testing.expectEqualSlices(u32, &.{ 107, 7, 2, 9 }, &tokens);
    try testing.expectEqualSlices(f64, &.{ 6.0, 5.0, 3.0, 3.0 }, &values);
    // one rank: its own, in its order
    chain.merge(halves[0..1], &tokens, &values);
    try testing.expectEqualSlices(u32, &.{ 7, 2, 9, 4 }, &tokens);
}

/// Books of rank 2 over a vocabulary of 6: predecessor rows all ones but token 5's, successor rows that make
/// token t's edge `t` times the projected row's first value (and for token 5 before it, three times that).
const pred = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1 };
const succ = [_]f32{ 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0 };
const books: chain.Books = .{ .predecessor = &pred, .successor = &succ, .rank = 2 };

test "a greedy chain takes the best logit plus 0.6 of the edge from the token before, the first of equals" {
    // position 0: token 1 leads by its logit, token 4's edge (4 x 0.5) passes it; position 1 after token 4:
    // equal scores, the first; position 2: no edge (a projected row of zeros)
    const c: chain.Candidates = .{
        .tokens = &.{ 1, 4, 2, 2, 3, 0, 0, 5, 1 },
        .values = &.{ 2.0, 1.2, 0.0, 1.0, 1.0, 0.0, 0.1, 0.3, 0.2 },
        .projected = &.{ 0.5, 9.0, 0.0, 9.0, 0.0, 0.0 },
        .top = 3,
    };
    var out: [3]u32 = undefined;
    try testing.expectEqual(@as(usize, 3), chain.choose(books, c, 0, 10, null, 0.0, false, &out));
    try testing.expectEqualSlices(u32, &.{ 4, 2, 5 }, &out);
    // a temperature of 0 is greedy too
    try testing.expectEqual(@as(usize, 3), chain.choose(books, c, 0, 10, .{ .seed = 1, .temperature = 0 }, 0.0, false, &out));
    try testing.expectEqualSlices(u32, &.{ 4, 2, 5 }, &out);
    // the edge is from the token before: after token 5 (a predecessor row of 3) token 2's passes token 1's logit
    const after: chain.Candidates = .{ .tokens = &.{ 1, 2 }, .values = &.{ 1.5, 0.0 }, .projected = &.{ 2.0, 0.0 }, .top = 2 };
    try testing.expectEqual(@as(usize, 1), chain.choose(books, after, 0, 10, null, 0.0, false, &out));
    try testing.expectEqual(@as(u32, 1), out[0]);
    try testing.expectEqual(@as(usize, 1), chain.choose(books, after, 5, 10, null, 0.0, false, &out));
    try testing.expectEqual(@as(u32, 2), out[0]);
}

test "the chain stops before the pick that takes the product under the confidence, and keeps the first" {
    // each position: two candidates a logit of ln 3 apart, no edge: the pick's probability is 3/4
    const ln3 = @log(3.0);
    const c: chain.Candidates = .{
        .tokens = &.{ 1, 2, 1, 2, 1, 2, 1, 2 },
        .values = &.{ ln3, 0, ln3, 0, ln3, 0, ln3, 0 },
        .projected = &@as([8]f64, @splat(0)),
        .top = 2,
    };
    var out: [4]u32 = undefined;
    try testing.expectEqual(@as(usize, 4), chain.choose(books, c, 0, 1, null, 0.0, false, &out));
    // 0.75, 0.5625, 0.42, 0.32
    try testing.expectEqual(@as(usize, 4), chain.choose(books, c, 0, 1, null, 0.3, false, &out));
    try testing.expectEqual(@as(usize, 2), chain.choose(books, c, 0, 1, null, 0.5, false, &out));
    try testing.expectEqual(@as(usize, 1), chain.choose(books, c, 0, 1, null, 0.6, false, &out));
    try testing.expectEqual(@as(usize, 1), chain.choose(books, c, 0, 1, null, 0.99, false, &out));
}

test "a sampled pick is the score over the temperature plus 0.7 of the keyed Gumbel noise at its position" {
    const c: chain.Candidates = .{
        .tokens = &.{ 1, 2, 3, 4, 1, 2, 3, 4 },
        .values = &.{ 1.0, 0.9, 0.8, 0.7, 0.2, 0.1, 0.4, 0.3 },
        .projected = &.{ 0.1, 0.0, 0.2, 0.0 },
        .top = 4,
    };
    var differed = false;
    for (0..64) |seed| {
        const noise: chain.Noise = .{ .seed = seed, .temperature = 0.8 };
        var out: [2]u32 = undefined;
        try testing.expectEqual(@as(usize, 2), chain.choose(books, c, 0, 20, noise, 0.0, false, &out));
        var prev: u32 = 0;
        for (0..2) |d| {
            var want: u32 = 0;
            var most: f64 = -std.math.inf(f64);
            for (c.tokens[d * 4 ..][0..4], c.values[d * 4 ..][0..4]) |token, value| {
                const edge = @as(f64, succ[token * 2]) * (@as(f64, pred[prev * 2]) * c.projected[d * 2]);
                const pick = (value + 0.6 * edge) / 0.8 + 0.7 * -@log(-@log(sampling.uniform(seed, 20 + d, token)));
                if (pick > most) {
                    most = pick;
                    want = token;
                }
            }
            try testing.expectEqual(want, out[d]);
            prev = want;
        }
        var greedy: [2]u32 = undefined;
        _ = chain.choose(books, c, 0, 20, null, 0.0, false, &greedy);
        differed = differed or !std.mem.eql(u32, &greedy, &out);
        // the same seed, the same chain; another position, its own noise
        var again: [2]u32 = undefined;
        _ = chain.choose(books, c, 0, 20, noise, 0.0, false, &again);
        try testing.expectEqualSlices(u32, &out, &again);
    }
    try testing.expect(differed);
}

test "a rank's top candidates: the largest first, the lower token among equals, NaN none, at the rank's offset" {
    const numeric = @import("../../../src/weights/numeric.zig");
    const raw = [_]f32{ 0.5, 2.0, -1.0, 2.0, std.math.nan(f32), 3.0, 0.5, 2.0, -4.0, 0.25 };
    var logits: [raw.len]u16 = undefined;
    for (&logits, raw) |*bits, value| bits.* = numeric.bf16FromF32(value);
    var values: [6]f32 = undefined;
    var tokens: [6]i32 = undefined;
    chain.largest(&logits, 100, &values, &tokens);
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 2.0, 2.0, 2.0, 0.5, 0.5 }, &values);
    try std.testing.expectEqualSlices(i32, &.{ 105, 101, 103, 107, 100, 106 }, &tokens);
    // against a sort of every value, on random rows with many equals
    var prng: std.Random.DefaultPrng = .init(11);
    const random = prng.random();
    var row: [300]u16 = undefined;
    for (0..20) |_| {
        for (&row) |*bits| bits.* = numeric.bf16FromF32(@floatFromInt(random.intRangeLessThan(i32, -20, 20)));
        var order: [row.len]u16 = undefined;
        for (&order, 0..) |*one, index| one.* = @intCast(index);
        const Context = struct {
            row: []const u16,
            fn before(self: @This(), a: u16, b: u16) bool {
                const x = numeric.f32FromBf16(self.row[a]);
                const y = numeric.f32FromBf16(self.row[b]);
                return x > y or (x == y and a < b);
            }
        };
        std.mem.sort(u16, &order, Context{ .row = &row }, Context.before);
        var v: [16]f32 = undefined;
        var t: [16]i32 = undefined;
        chain.largest(&row, 7, &v, &t);
        for (v, t, order[0..16]) |value, token, index| {
            try std.testing.expectEqual(numeric.f32FromBf16(row[index]), value);
            try std.testing.expectEqual(@as(i32, index) + 7, token);
        }
    }
}

test "a pick under a mask is the best of the candidates it leaves, and none where it leaves none" {
    const rank = 2;
    const zero: [64 * rank]f32 = @splat(0);
    const none: chain.Books = .{ .predecessor = &zero, .successor = &zero, .rank = rank };
    const c: chain.Candidates = .{ .tokens = &.{ 3, 5, 6, 40 }, .values = &.{ 4.0, 3.0, 2.0, 1.0 }, .projected = &.{ 0, 0 }, .top = 4 };
    try std.testing.expectEqual(@as(u32, 3), chain.pick(none, c, 0, 0, 1, null, null, false).?.token);
    // tokens 5 and 6 allowed: 5 is the larger; its probability is among those two
    const one = chain.pick(none, c, 0, 0, 1, null, &.{ 0b1100000, 0 }, true).?;
    try std.testing.expectEqual(@as(u32, 5), one.token);
    try std.testing.expectApproxEqAbs(1.0 / (1.0 + @exp(-1.0)), one.p, 1e-12);
    try std.testing.expectEqual(@as(u32, 40), chain.pick(none, c, 0, 0, 1, null, &.{ 0, 0b100000000 }, false).?.token);
    try std.testing.expect(chain.pick(none, c, 0, 0, 1, null, &.{ 0b10, 0 }, false) == null);
    // a mask shorter than the vocabulary allows nothing past it
    try std.testing.expect(chain.pick(none, c, 0, 0, 1, null, &.{0}, false) == null);
}

test "a pick's noise-aware confidence is the kit's: the scores, with the whole keyed noise when sampling, over its temperature" {
    const rank = 2;
    const zero: [64 * rank]f32 = @splat(0);
    const none: chain.Books = .{ .predecessor = &zero, .successor = &zero, .rank = rank };
    const tokens = [_]u32{ 3, 5, 6, 40 };
    const values = [_]f64{ 4.0, 3.0, 2.0, 1.0 };
    const c: chain.Candidates = .{ .tokens = &tokens, .values = &values, .projected = &.{ 0, 0 }, .top = 4 };
    // greedy: softmax(score / 2) at the pick, and the product's probability beside it unchanged
    const greedy = chain.pick(none, c, 0, 0, 9, null, null, true).?;
    var sum: f64 = 0;
    for (values) |v| sum += @exp((v - 4.0) / chain.noisy_beta);
    try std.testing.expectApproxEqAbs(1.0 / sum, greedy.noisy, 1e-12);
    var plain: f64 = 0;
    for (values) |v| plain += @exp(v - 4.0);
    try std.testing.expectApproxEqAbs(1.0 / plain, greedy.p, 1e-12);
    // sampled: z = (value / T + g) / 2, g the keyed Gumbel draw of the candidate at the position
    const noise: chain.Noise = .{ .seed = 11, .temperature = 0.8 };
    const sampled = chain.pick(none, c, 0, 0, 9, noise, null, true).?;
    var z: [4]f64 = undefined;
    for (&z, tokens, values) |*one, token, v| one.* = (v / 0.8 + -@log(-@log(sampling.uniform(11, 9, token)))) / chain.noisy_beta;
    const most = std.mem.max(f64, &z);
    var all: f64 = 0;
    for (z) |one| all += @exp(one - most);
    const at = std.mem.indexOfScalar(u32, &tokens, sampled.token).?;
    try std.testing.expectApproxEqAbs(@exp(z[at] - most) / all, sampled.noisy, 1e-12);
    // unjudged, nothing is computed
    try std.testing.expectEqual(@as(f64, 1), chain.pick(none, c, 0, 0, 9, noise, null, false).?.noisy);
}
