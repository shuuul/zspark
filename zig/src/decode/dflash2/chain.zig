//! DFlash2's choice of drafts on the host, from `dflash2.py:Drafter.candidates` and `chain`: both ranks'
//! candidates of a position merged, then one picked a position by its logit, the selector's edge from the
//! token before it and, when sampling, the target's keyed noise; the chain stops under a confidence. fp64 in
//! a fixed order. TensorFold's edge is a BLAS product, whose order of additions is not this one's: the picks
//! are held to its on recorded candidates (spec 067 WS-13), not the scores to the bit.
//!
//! WS-03's Qwen drafter imports `Books`, `Half`, `Candidates`, `Noise`, `Pick`, `largest`, `merge`, `pick`,
//! `choose`, `max_top` and `noisy_beta`.

const std = @import("std");
const sampling = @import("../sampling.zig");
const numeric = @import("../../weights/numeric.zig");
const config_file = @import("config.zig");

/// The edge's weight in a score and the noise's in a pick (`dflash2.py`'s `EDGE`, `NOISE`).
const edge_weight = 0.6;
const noise_weight = 0.7;

/// The most candidates a position has after the merge: `selector_top_k`.
pub const max_top = 64;

/// The selector's codebooks as the chain reads them: a token's row of `rank` fp32 values.
pub const Books = struct {
    predecessor: []const f32,
    successor: []const f32,
    rank: usize,

    fn pred(self: Books, token: u32) []const f32 {
        return self.predecessor[@as(usize, token) * self.rank ..][0..self.rank];
    }

    fn succ(self: Books, token: u32) []const f32 {
        return self.successor[@as(usize, token) * self.rank ..][0..self.rank];
    }
};

/// A rank's candidates of one position as the block pass packs them: its `top` largest logits of its half of
/// the vocabulary, largest first, and their token ids over the whole vocabulary.
pub const Half = struct {
    values: []const f32,
    tokens: []const i32,
};

/// A rank's half of one position's candidates: the `values.len` largest of `logits`, bf16 bits a token of the
/// rank's rows of the vocabulary, largest first and the lower token among equals, each token `offset` on. NaN
/// is no candidate. What `torch.topk` gives but among equals, where it promises no order.
pub fn largest(logits: []const u16, offset: usize, values: []f32, tokens: []i32) void {
    const count = values.len;
    std.debug.assert(tokens.len == count and count <= logits.len);
    var held: usize = 0;
    for (logits, 0..) |bits, index| {
        const value = numeric.f32FromBf16(bits);
        if (std.math.isNan(value)) continue;
        if (held == count and !(value > values[count - 1])) continue;
        // above every held value it is not larger than: equals stay in the order of their tokens
        var at = @min(held, count - 1);
        while (at > 0 and value > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            tokens[at] = tokens[at - 1];
        }
        values[at] = value;
        tokens[at] = @intCast(offset + index);
        if (held < count) held += 1;
    }
    for (values[held..], tokens[held..]) |*value, *token| {
        value.* = -std.math.inf(f32);
        token.* = @intCast(offset);
    }
}

/// The `top` best of the ranks' candidates of one position: by logit, the largest first, and by token id
/// where two are equal (`np.lexsort((tokens, -values))`). `tokens` and `values` take `top` each.
pub fn merge(halves: []const Half, tokens: []u32, values: []f64) void {
    const top = tokens.len;
    std.debug.assert(values.len == top and top <= max_top and halves.len <= config_file.world);
    const Entry = struct {
        value: f64,
        token: u32,

        fn before(_: void, a: @This(), b: @This()) bool {
            return a.value > b.value or (a.value == b.value and a.token < b.token);
        }
    };
    var all: [config_file.world * max_top]Entry = undefined;
    var n: usize = 0;
    for (halves) |half| {
        std.debug.assert(half.values.len == top and half.tokens.len == top);
        for (half.values, half.tokens) |value, token| {
            all[n] = .{ .value = value, .token = @intCast(token) };
            n += 1;
        }
    }
    std.mem.sort(Entry, all[0..n], {}, Entry.before);
    for (tokens, values, all[0..top]) |*token, *value, entry| {
        token.* = entry.token;
        value.* = entry.value;
    }
}

/// What a chain is chosen from: `depth` positions of `top` candidates, and each position's projected row.
pub const Candidates = struct {
    /// [depth, top]
    tokens: []const u32,
    values: []const f64,
    /// [depth, rank]
    projected: []const f64,
    top: usize,

    pub fn depth(self: Candidates) usize {
        return self.tokens.len / self.top;
    }
};

/// How a chain is noised: the request's seed and temperature. Null, or a temperature of 0 or below: greedy.
pub const Noise = struct {
    seed: u64,
    temperature: f64,
};

/// A position's pick, and the probability its candidates' scores give it (1 where none is asked). `noisy` is
/// the kit's noise-aware confidence of the same pick (its patch 0018, `dflash2.noisy_confidence`): the softmax
/// of the scores plus the target's keyed noise, whole, over `noisy_beta`. No rule here reads it; a record's
/// chains carry it (`generate ... chains`) so that the kit's stop rule is played beside the product's.
pub const Pick = struct { token: u32, p: f64, noisy: f64 = 1 };

/// The kit's `NOISY_BETA`.
pub const noisy_beta = 2.0;

/// The candidate of position `d` (the sequence's `first + d`) after the token `prev`: the largest score, a
/// candidate's logit and the selector's edge from `prev` over the temperature, with the keyed noise when
/// sampling; the first of equals. With `allowed`, a constraint's mask of the vocabulary, only its candidates
/// stand, and null is none of them. `judged`: the pick's probability among the standing candidates is wanted.
pub fn pick(books: Books, c: Candidates, d: usize, prev: u32, first: u64, noise: ?Noise, allowed: ?[]const u32, judged: bool) ?Pick {
    const top = c.top;
    std.debug.assert(top <= max_top and d < c.depth() and c.projected.len == c.depth() * books.rank);
    const sampled = noise != null and noise.?.temperature > 0;
    const temperature: f64 = if (sampled) noise.?.temperature else 1.0;
    var score: [max_top]f64 = undefined;
    var drawn: [max_top]f64 = undefined;
    var score_tokens: [max_top]u32 = undefined;
    const tokens = c.tokens[d * top ..][0..top];
    const values = c.values[d * top ..][0..top];
    const row = c.projected[d * books.rank ..][0..books.rank];
    const from = books.pred(prev);
    var best: ?usize = null;
    var best_pick: f64 = undefined;
    var highest: f64 = -std.math.inf(f64);
    var standing: usize = 0;
    for (tokens, values) |token, value| {
        if (allowed) |mask| if (token / 32 >= mask.len or mask[token / 32] >> @intCast(token % 32) & 1 == 0) continue;
        var edge: f64 = 0;
        for (books.succ(token), from, row) |to, back, at| edge += @as(f64, to) * (@as(f64, back) * at);
        const own = (value + edge_weight * edge) / temperature;
        highest = @max(highest, own);
        const draw = if (sampled) -@log(-@log(sampling.uniform(noise.?.seed, first + d, token))) else 0;
        const noised = if (sampled) own + noise_weight * draw else own;
        // the first of equals, as `np.argmax`
        if (best == null or noised > best_pick) {
            best = standing;
            best_pick = noised;
        }
        score[standing] = own;
        drawn[standing] = (own + draw) / noisy_beta;
        score_tokens[standing] = token;
        standing += 1;
    }
    const chosen = best orelse return null;
    if (!judged) return .{ .token = score_tokens[chosen], .p = 1 };
    var prob: [max_top]f64 = undefined;
    for (score[0..standing], prob[0..standing]) |one, *p| p.* = @exp(one - highest);
    const p = prob[chosen] / sampling.pairwiseSum(prob[0..standing]);
    const most = std.mem.max(f64, drawn[0..standing]);
    for (drawn[0..standing], prob[0..standing]) |one, *q| q.* = @exp(one - most);
    return .{ .token = score_tokens[chosen], .p = p, .noisy = prob[chosen] / sampling.pairwiseSum(prob[0..standing]) };
}

/// The drafts for the positions `first` onward after the token `anchor`, into `out`; returns how many. With a
/// `confidence` above 0 the chain ends before the first pick that takes the product of the picks'
/// probabilities under it, the first pick always kept. `noisy` uses each pick's noise-aware confidence,
/// which is what a drafter with `Drafter.noisy` stops on.
pub fn choose(books: Books, c: Candidates, anchor: u32, first: u64, noise: ?Noise, confidence: f64, noisy: bool, out: []u32) usize {
    std.debug.assert(out.len >= c.depth());
    var prev = anchor;
    var chain: f64 = 1.0;
    var kept: usize = 0;
    for (0..c.depth()) |d| {
        const one = pick(books, c, d, prev, first, noise, null, confidence > 0).?;
        chain *= if (noisy) one.noisy else one.p;
        if (confidence > 0 and d > 0 and chain < confidence) break;
        prev = one.token;
        out[kept] = prev;
        kept += 1;
    }
    return kept;
}
