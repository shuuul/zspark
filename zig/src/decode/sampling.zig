//! TensorFold's keyed sampling, on the host: a row of bf16 logits, the settings, a seed and the absolute position
//! give a token. The draw is a pure function of (seed, position, token id), with no generator state, so a
//! drafted token can be checked against the one serial decoding picks.
//!
//! From TensorFold 0.6.5: `engine/exact_sampling.py` (`Sampling`, `seed_for`, `uniform`, `choose_rows`) and
//! `cuda/sampling.py` (`sample_rows`, `nucleus_rows`), the path `families/qwen4_exp/cuda/decode.py`'s
//! `Engine.sample` takes on one rank for the main head. It has three rules:
//!
//! - **Temperature 0 or below**: the largest logit, the lower id among equals.
//! - **Top-k on**: `torch.topk` reads `top_k + 8` candidates, then the host keeps the `top_k` largest of them by
//!   (logit down, id up); each is widened to fp64 and divided by the temperature; top-p cuts on probabilities
//!   over those `top_k` alone; min-p cuts at ln(min_p) below the top; then the largest of `scaled - ln(-ln(u))`
//!   over what is left wins, the earlier in the order among equals.
//! - **Top-k off** (`top_k` 0): the same draw over the whole row, but the logits are multiplied by the
//!   temperature's reciprocal (PyTorch's CUDA division by a scalar), and top-p cuts on integer masses
//!   `floor(exp(scaled - top) * 2^40)` against `ceil(top_p * total)` in exact arithmetic.
//!
//! `torch.topk` on the device selects by the logits' bits: +0 ahead of -0, and the lower ids among equal bits.
//! PyTorch does not promise that order, but it is what TensorFold's build does (checked on 630 rows built to
//! tie) and so what decides a pick when zeros or more than eight equal logits straddle the cut; it is
//! reproduced here. The other two rules order by value, the zeros as one.
//!
//! Departures from TensorFold, none of which moves a pick in practice:
//!
//! - `exp` and `log` are Zig's (compiler-rt). TensorFold's are glibc's through numpy and, for the masses, the
//!   device's. Each is within an ulp of the others, so a pick could differ only where two scores, or a
//!   cumulative probability and `top_p`, agree to the last bit or two of an fp64.
//! - A row with a NaN, a +inf or no finite logit is refused instead of sampled.

const std = @import("std");
const Allocator = std.mem.Allocator;
const numeric = @import("../weights/numeric.zig");

/// `exact_sampling.Sampling`, with its defaults (which are also the Flash Next checkpoint's
/// `generation_config.json`). TensorFold's servers fall back to the checkpoint's values, and to greedy without.
pub const Sampling = struct {
    /// The key: a request's `seed` (`seedFromInt`), else `seedFor` of the prompt.
    seed: u64,
    /// 0 or below decodes greedily; a positive value below 1e-6 counts as 1e-6.
    temperature: f64 = 1.0,
    /// 0 turns top-k off, and with it selects the whole-row rule.
    top_k: u32 = 20,
    /// Cuts only strictly between 0 and 1; anything else is off.
    top_p: f64 = 0.95,
    /// Keeps tokens at least `min_p` times as likely as the likeliest, after temperature; 0 is off.
    min_p: f64 = 0.0,

    pub const Invalid = error{ InvalidTemperature, InvalidTopP, InvalidMinP };

    /// What `server/request_options.py`'s `parse_numbers` refuses: a number that is not finite, and a `min_p`
    /// outside 0 to 1. (It clamps a negative `top_k` to 0, which the unsigned field leaves to the caller.)
    /// Nothing else is refused, by it or by what stands before it on any of TensorFold's three APIs (spec
    /// 056): a temperature below 0 is `greedy`, one above 2 is sampled with, and a `top_p` at or below 0 or
    /// above 1 is one that does not cut (`cuts`). vLLM and SGLang refuse a negative temperature and a `top_p`
    /// outside (0, 1]; TensorFold is the reference here, so no such bound is added.
    pub fn validate(self: Sampling) Invalid!void {
        if (!std.math.isFinite(self.temperature)) return error.InvalidTemperature;
        if (!std.math.isFinite(self.top_p)) return error.InvalidTopP;
        if (!(self.min_p >= 0.0 and self.min_p <= 1.0)) return error.InvalidMinP;
    }

    pub fn greedy(self: Sampling) bool {
        return self.temperature <= 0.0;
    }

    /// A checkpoint's defaults, as TensorFold's `cli.py` `_generation_config` reads `generation_config.json`:
    /// its `temperature`, `top_k`, `top_p` and `min_p` where they are numbers; `do_sample` false is greedy, and
    /// true without a temperature is temperature 1. A config that sets no temperature either way, or no config,
    /// decodes greedily.
    pub fn fromGenerationConfig(config: ?std.json.Value, seed: u64) Sampling {
        var s: Sampling = .{ .seed = seed, .temperature = 0.0 };
        const c = config orelse return s;
        if (c != .object) return s;
        if (number(c.object.get("temperature"))) |v| s.temperature = v;
        if (number(c.object.get("top_k"))) |v| s.top_k = if (v > 0) std.math.lossyCast(u32, v) else 0;
        if (number(c.object.get("top_p"))) |v| s.top_p = v;
        if (number(c.object.get("min_p"))) |v| s.min_p = v;
        if (c.object.get("do_sample")) |sample| if (sample == .bool) {
            if (!sample.bool) s.temperature = 0.0 else if (number(c.object.get("temperature")) == null) s.temperature = 1.0;
        };
        return s;
    }

    fn number(v: ?std.json.Value) ?f64 {
        return switch (v orelse return null) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => null,
        };
    }

    pub fn cuts(self: Sampling) bool {
        return self.top_p > 0.0 and self.top_p < 1.0;
    }
};

/// A request's integer seed as the key: its low 64 bits, as `seed & 0xFFFFFFFFFFFFFFFF` does for a negative one.
pub fn seedFromInt(seed: i64) u64 {
    return @bitCast(seed);
}

/// `exact_sampling.seed_for`: the key of a request without a seed. SHA-256 of the prompt's ids in decimal joined
/// by commas, then `|` and the salt (`TENSORFOLD_SEED_SALT`, 0 unless set); the first 8 bytes as a little-endian
/// integer, without its top bit.
pub fn seedFor(tokens: []const i32, salt: i64) u64 {
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    var text: [24]u8 = undefined;
    for (tokens, 0..) |token, i| {
        if (i != 0) hash.update(",");
        hash.update(std.fmt.bufPrint(&text, "{d}", .{token}) catch unreachable);
    }
    hash.update(std.fmt.bufPrint(&text, "|{d}", .{salt}) catch unreachable);
    const digest = hash.finalResult();
    return std.mem.readInt(u64, digest[0..8], .little) & (1 << 63) - 1;
}

/// splitmix64's finaliser.
fn mix(value: u64) u64 {
    var x = value;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// The part of the hash a row's tokens share.
fn positionKey(seed: u64, position: u64) u64 {
    return mix(mix(seed +% 0x9E3779B97F4A7C15) ^ (position *% 0xD1B54A32D192ED03));
}

fn uniformKeyed(key: u64, token: u64) f64 {
    return @as(f64, @floatFromInt(mix(key ^ token) >> 11)) * 0x1p-53 + 0x1p-54;
}

/// `exact_sampling.uniform`: a double strictly between 0 and 1 from (seed, position, token id), 53 hashed bits
/// at the centre of their 2^-53 cell. `position` is the index in the sequence of the token being chosen: the
/// prompt's length for the first one.
pub fn uniform(seed: u64, position: u64, token: u64) f64 {
    return uniformKeyed(positionKey(seed, position), token);
}

/// Token `token`'s score: its scaled logit plus Gumbel noise. The largest wins.
fn score(scaled: f64, key: u64, token: u32) f64 {
    return scaled - @log(-@log(uniformKeyed(key, token)));
}

/// numpy's `sum` over fp64: eight running sums up to 128 values, halves above that. `choose_rows` normalises
/// its probabilities by this sum, so its order of additions is part of the result.
pub fn pairwiseSum(values: []const f64) f64 {
    const n = values.len;
    if (n < 8) {
        var sum: f64 = 0.0;
        for (values) |v| sum += v;
        return sum;
    }
    if (n <= 128) {
        var r: [8]f64 = values[0..8].*;
        var i: usize = 8;
        while (i < n - n % 8) : (i += 8) {
            inline for (0..8) |j| r[j] += values[i + j];
        }
        var sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < n) : (i += 1) sum += values[i];
        return sum;
    }
    var half = n / 2;
    half -= half % 8;
    return pairwiseSum(values[0..half]) + pairwiseSum(values[half..]);
}

/// A token's share of the mass, in units of 2^-40 of the top token's: `nucleus_rows`'s `MASS`.
const mass_unit = 0x1p40;

pub fn massOf(scaled: f64, top: f64) u64 {
    const d = scaled - top;
    if (d < -28.0) return 0; // exp(-28) * 2^40 is 0.76: the floor is 0 from here down
    return @intFromFloat(@floor(@exp(d) * mass_unit));
}

/// `ceil(Fraction(top_p) * total)`: the mass the nucleus must reach, with `top_p` taken as the exact binary
/// fraction the double holds. `top_p` is strictly between 0 and 1.
pub fn massNeeded(top_p: f64, total: u64) u64 {
    const bits: u64 = @bitCast(top_p);
    const exponent = (bits >> 52) & 0x7ff;
    const fraction = bits & (1 << 52) - 1;
    const mantissa = if (exponent == 0) fraction else fraction | 1 << 52;
    const shift = 1075 - @max(exponent, 1); // top_p is mantissa * 2^-shift, and shift is at least 53
    const product = @as(u128, mantissa) * total;
    if (shift >= 128) return @intFromBool(product != 0);
    const s: u7 = @intCast(shift);
    return @intCast((product >> s) + @intFromBool(product & ((@as(u128, 1) << s) - 1) != 0));
}

/// bf16 bits as a 16-bit key that orders as the values do, except that -0 (`minus_zero`) sits one below +0
/// (`plus_zero`), which is the order `torch.topk` selects in. The finite values fill `lowest_finite` to
/// `highest_finite`, the infinities sit just outside and the NaNs beyond them.
pub fn keyOf(bits: u16) u16 {
    return if (bits & 0x8000 != 0) ~bits else bits | 0x8000;
}

/// The bf16 bits `keyOf` made `key` of.
pub fn bitsOf(key: u16) u16 {
    return if (key >= 0x8000) key & 0x7fff else ~key;
}

fn valueOf(key: u16) f64 {
    return numeric.f32FromBf16(bitsOf(key));
}

pub const lowest_finite = 0x0080;
const highest_finite = 0xff7f;
pub const plus_zero = 0x8000;
pub const minus_zero = 0x7fff;
const lanes = 32;
const Lanes = @Vector(lanes, u16);

/// `keyOf` over a window; with `fold` the two zeros share `plus_zero`, so the keys order exactly as the values.
fn keysOf(bits: Lanes, comptime fold: bool) Lanes {
    const sign: Lanes = @splat(0x8000);
    const zero: Lanes = @splat(0);
    const keys = @select(u16, bits & sign != zero, ~bits, bits | sign);
    if (!fold) return keys;
    return @select(u16, keys == @as(Lanes, @splat(minus_zero)), @as(Lanes, @splat(plus_zero)), keys);
}

/// `exact_sampling.MARGIN`: the candidates TensorFold reads beyond `top_k`.
pub const margin = 8;

/// A row's start in the order `choose` reads it, found by something other than `choose` (a kernel, for a
/// draft): the first `keys.len` tokens of a row of `len` by (key down, id up), their `keyOf` keys and ids, and
/// the row's smallest key.
pub const Leading = struct {
    keys: []const u16,
    ids: []const u32,
    smallest: u16,
    len: usize,
    /// The sum over the row of exp(v - largest) for v > largest - 24, within `sum_slack` of it relatively;
    /// null where it is not known, and a nucleus is then not settled.
    sum: ?f64 = null,

    /// How far a `sum` may be from the row's: a kernel's is fp32 terms added in fp32, 1e-5 off at most.
    const sum_slack = 4e-5;
};

/// The start of a row scored in parts (GLM's ranks hold half the vocabulary each, spec 045), from each part's
/// own start: `parts` in the row's order, a part's `ids` already the row's, its `len` the part's, and its start
/// every token of the part at or above its last key (what a pick kernel asked to a rank leaves). The joined
/// start, in `keys` and `ids`, is the parts' tokens at or above the highest of the parts' last keys, below
/// which a part may hold tokens it did not give; it is never shorter than the shortest part's. Null where a
/// part gave nothing. The sum is the parts' brought to the row's largest logit: it also counts a part's logits
/// within 24 of that part's largest and not of the row's, each under e^-24, which `chooseLeading` allows for.
pub fn joinLeading(parts: []const Leading, keys: []u16, ids: []u32) ?Leading {
    var cut: u16 = 0;
    var top: u16 = 0;
    var smallest: u16 = 0xffff;
    var len: usize = 0;
    for (parts) |part| {
        if (part.keys.len == 0) return null;
        std.debug.assert(part.keys.len == part.ids.len and part.keys.len <= part.len);
        if (part.keys.len < part.len) cut = @max(cut, part.keys[part.keys.len - 1]);
        top = @max(top, part.keys[0]);
        smallest = @min(smallest, part.smallest);
        len += part.len;
    }
    // the parts' ids ascend with the parts, so among equal keys the earlier part's come first
    var at: [8]usize = @splat(0);
    std.debug.assert(parts.len <= at.len);
    var n: usize = 0;
    while (true) : (n += 1) {
        var from: ?usize = null;
        for (parts, 0..) |part, index| {
            if (at[index] == part.keys.len or part.keys[at[index]] < cut) continue;
            if (from == null or part.keys[at[index]] > parts[from.?].keys[at[from.?]]) from = index;
        }
        const index = from orelse break;
        keys[n] = parts[index].keys[at[index]];
        ids[n] = parts[index].ids[at[index]];
        at[index] += 1;
    }
    var sum: ?f64 = 0;
    for (parts) |part| {
        const own = part.sum orelse {
            sum = null;
            break;
        };
        // a part with nothing finite adds nothing; a row without a finite top is refused by its reader
        if (part.keys[0] < lowest_finite or top > highest_finite) continue;
        sum.? += own * @exp(valueOf(part.keys[0]) - valueOf(top));
    }
    return .{ .keys = keys[0..n], .ids = ids[0..n], .smallest = smallest, .len = len, .sum = sum };
}

/// The row's first tokens a nucleus over the whole row is settled from, where it can be (`chooseLeading`).
const nucleus_leading = 64;

pub const Error = error{
    /// The row holds a NaN or +inf, or nothing finite.
    NonFiniteLogits,
};

/// The id of the largest logit, the lower id among equals (and +0 equals -0): `logits.argmax` in TensorFold.
/// Only the largest is looked at, so a NaN with its sign bit set goes unseen here.
pub fn argmax(logits: []const u16) Error!i32 {
    std.debug.assert(logits.len != 0);
    var best: u16 = 0;
    var i: usize = 0;
    while (i + lanes <= logits.len) : (i += lanes) {
        best = @max(best, @reduce(.Max, keysOf(logits[i..][0..lanes].*, true)));
    }
    for (logits[i..]) |bits| best = @max(best, keyOf(bits));
    if (best < lowest_finite or best > highest_finite) return error.NonFiniteLogits;
    const equal: u16 = if (best == plus_zero) minus_zero else best;
    // the windows before the first that holds it are passed 32 logits at a time
    var first: usize = 0;
    while (first + lanes <= logits.len) : (first += lanes) {
        if (@reduce(.Or, keysOf(logits[first..][0..lanes].*, true) == @as(Lanes, @splat(best)))) break;
    }
    for (logits[first..], first..) |bits, id| {
        const key = keyOf(bits);
        if (key == best or key == equal) return @intCast(id);
    }
    unreachable;
}

/// How many of a row's first tokens (`Leading`) can settle `choose` under `s` for a row of `len`: the largest
/// alone for a greedy choice, `top_k` and its margin for top-k sampling, `nucleus_leading` for a nucleus at a
/// temperature of 1 or under; null where the whole row is read: no cut at all, or a temperature over 1.
pub fn leadingNeeded(s: Sampling, len: usize) ?usize {
    if (s.greedy()) return 1;
    if (s.top_k == 0) return if (s.cuts() and s.temperature <= 1) @min(nucleus_leading, len) else null;
    return @min(@as(usize, s.top_k) + margin, len);
}

/// Scratch for rows of up to `vocab` logits. `choose` allocates nothing.
pub const Sampler = struct {
    /// [65536] how many of the row's logits have each key; `gather` turns the kept keys' counts into offsets.
    counts: []u32,
    /// [vocab] the kept tokens, by logit down then id up.
    ids: []u32,
    /// [vocab] their keys.
    keys: []u16,
    /// [vocab] their logits over the temperature.
    scaled: []f64,
    /// [vocab] their probabilities, for top-k's top-p.
    probs: []f64,

    pub fn init(gpa: Allocator, vocab: usize) Allocator.Error!Sampler {
        const counts = try gpa.alloc(u32, 1 << 16);
        errdefer gpa.free(counts);
        const ids = try gpa.alloc(u32, vocab);
        errdefer gpa.free(ids);
        const keys = try gpa.alloc(u16, vocab);
        errdefer gpa.free(keys);
        const scaled = try gpa.alloc(f64, vocab);
        errdefer gpa.free(scaled);
        const probs = try gpa.alloc(f64, vocab);
        return .{ .counts = counts, .ids = ids, .keys = keys, .scaled = scaled, .probs = probs };
    }

    pub fn deinit(self: *Sampler, gpa: Allocator) void {
        gpa.free(self.counts);
        gpa.free(self.ids);
        gpa.free(self.keys);
        gpa.free(self.scaled);
        gpa.free(self.probs);
        self.* = undefined;
    }

    /// The token at `position` (its index in the sequence) from that position's row of bf16 logits. The same
    /// row, settings, seed and position always give the same token.
    pub fn choose(self: *Sampler, logits: []const u16, position: u64, s: Sampling) Error!i32 {
        std.debug.assert(logits.len != 0 and logits.len <= self.ids.len);
        if (s.greedy()) return argmax(logits);

        @memset(self.counts, 0);
        for (logits) |bits| self.counts[keyOf(bits)] += 1;
        for (self.counts[0 .. lowest_finite - 1]) |count| {
            if (count != 0) return error.NonFiniteLogits; // a NaN with its sign bit set
        }
        for (self.counts[highest_finite + 1 ..]) |count| {
            if (count != 0) return error.NonFiniteLogits; // +inf or a NaN
        }
        var top: u16 = highest_finite;
        while (self.counts[top] == 0) : (top -= 1) {
            if (top == lowest_finite) return error.NonFiniteLogits; // nothing, or only -inf
        }

        const temperature = @max(s.temperature, 1e-6);
        const kept = if (s.top_k != 0)
            self.keepTopK(logits, top, temperature, s)
        else
            self.keepNucleus(logits, top, temperature, s);
        return self.draw(kept, position, s);
    }

    /// `choose` for a row known by its start alone: the token `choose` gives for the whole row, or null where
    /// the start does not settle it, and the caller reads the row. It is settled for a greedy choice whose
    /// largest logit is not a zero (the zeros' order is not the argmax's), and for top-k sampling with the
    /// `top_k` and its margin within the start; never where the row holds a NaN or +inf, or the start a -inf,
    /// which `choose` refuses or reads its own way.
    pub fn chooseLeading(self: *Sampler, row: Leading, position: u64, s: Sampling) ?i32 {
        if (leadingNeeded(s, row.len) == null) return null;
        std.debug.assert(row.keys.len == row.ids.len and row.keys.len <= row.len and row.len <= self.ids.len);
        if (row.keys.len == 0) return null;
        const top = row.keys[0];
        if (top < lowest_finite or top > highest_finite or row.smallest < lowest_finite - 1) return null;
        if (s.greedy()) return if (top == plus_zero or top == minus_zero) null else @intCast(row.ids[0]);
        if (s.top_k == 0) return self.leadNucleus(row, position, s);
        const candidates = @min(@as(usize, s.top_k) + margin, row.len);
        if (candidates > row.keys.len or row.keys[candidates - 1] < lowest_finite) return null;
        @memcpy(self.keys[0..candidates], row.keys[0..candidates]);
        @memcpy(self.ids[0..candidates], row.ids[0..candidates]);
        const kept = self.cutTopK(candidates, row.len, @max(s.temperature, 1e-6), s);
        return self.draw(kept, position, s);
    }

    /// `chooseLeading` under a nucleus over the whole row (spec 065). The row's mass is the start's, which is
    /// exact here, and the rest's, which is at most what `row.sum` leaves beyond the start at temperature 1,
    /// shrunk as a temperature of 1 or under shrinks the start's last token. Where the cut falls at the same
    /// token with no rest and with all of that bound, the kept tokens are the row's and so is the draw.
    fn leadNucleus(self: *Sampler, row: Leading, position: u64, s: Sampling) ?i32 {
        const sum = row.sum orelse return null;
        const temperature = @max(s.temperature, 1e-6);
        const n = row.keys.len;
        // the zeros are one value to this rule and two keys to the start's order
        for (row.keys) |key| if (key < lowest_finite or key == plus_zero or key == minus_zero) return null;
        const reciprocal = 1.0 / temperature;
        const top_value = valueOf(row.keys[0]);
        const top_scaled = top_value * reciprocal;
        var lead: u64 = 0;
        var lead_plain: f64 = 0;
        for (row.keys) |key| {
            lead += massOf(valueOf(key) * reciprocal, top_scaled);
            lead_plain += @exp(valueOf(key) - top_value);
        }
        var rest: u64 = 0;
        if (n < row.len) {
            // what the sum leaves out, logits 24 and more under the largest, is at most e^-24 each
            const plain = @max(sum * (1 + Leading.sum_slack) + @as(f64, @floatFromInt(row.len)) * @exp(-24.0) - lead_plain, 0);
            const last = valueOf(row.keys[n - 1]) - top_value;
            rest = @intFromFloat(@ceil(plain * @exp(last * (reciprocal - 1)) * mass_unit));
        }
        var kept: usize = 0;
        for ([_]u64{ lead, lead + rest }) |total| {
            const need = massNeeded(s.top_p, total);
            var below: usize = 0;
            var running: u64 = 0;
            for (row.keys) |key| {
                const mass = massOf(valueOf(key) * reciprocal, top_scaled);
                if (mass != 0 and running + mass >= need) break;
                running += mass;
                below += 1;
            }
            // the cut is past the start, or not where the other total puts it
            if (below == n and n < row.len) return null;
            if (kept != 0 and kept != below + 1) return null;
            kept = @min(below + 1, n);
        }
        if (s.min_p > 0.0) {
            const floor = top_scaled + @log(s.min_p);
            var above: usize = 0;
            while (above < kept and valueOf(row.keys[above]) * reciprocal >= floor) above += 1;
            kept = above;
        }
        @memcpy(self.ids[0..kept], row.ids[0..kept]);
        for (self.scaled[0..kept], row.keys[0..kept]) |*value, key| value.* = valueOf(key) * reciprocal;
        return self.draw(kept, position, s);
    }

    /// The kept tokens' keyed draw: the largest score, the first among equals.
    fn draw(self: *Sampler, kept: usize, position: u64, s: Sampling) i32 {
        const key = positionKey(s.seed, position);
        var best: usize = 0;
        var best_score = -std.math.inf(f64);
        for (self.scaled[0..kept], self.ids[0..kept], 0..) |scaled, id, i| {
            const value = score(scaled, key, id);
            if (value > best_score) {
                best_score = value;
                best = i;
            }
        }
        return @intCast(self.ids[best]);
    }

    /// `sample_rows` into `choose_rows`: how many tokens of the order survive top-k, then top-p over those, then
    /// min-p; their ids and scaled logits are left in `ids` and `scaled`.
    fn keepTopK(self: *Sampler, logits: []const u16, top: u16, temperature: f64, s: Sampling) usize {
        // `torch.topk`'s candidates: top_k and a margin, +0 ahead of -0, the lower ids among equal bits
        const candidates = @min(@as(usize, s.top_k) + margin, logits.len);
        self.gather(logits, top, candidates, false);
        return self.cutTopK(candidates, logits.len, temperature, s);
    }

    /// `keepTopK` after its candidates are in `keys` and `ids`; `len` is the row's length.
    fn cutTopK(self: *Sampler, candidates: usize, len: usize, temperature: f64, s: Sampling) usize {
        // `lexsort` then orders them by value and id, to which the two zeros are one value
        if (std.mem.findScalar(u16, self.keys[0..candidates], minus_zero)) |minus| {
            const plus = std.mem.findScalar(u16, self.keys[0..minus], plus_zero) orelse minus;
            const end = std.mem.findNonePos(u16, self.keys[0..candidates], minus, &.{minus_zero}) orelse candidates;
            std.mem.sort(u32, self.ids[plus..end], {}, std.sort.asc(u32));
        }
        const k: usize = @min(s.top_k, len);
        const scaled = self.scaled[0..k];
        for (scaled, self.keys[0..k]) |*value, key| value.* = valueOf(key) / temperature;
        var kept = k;
        if (s.cuts()) {
            const probs = self.probs[0..k];
            for (probs, scaled) |*p, value| p.* = @exp(value - scaled[0]);
            const sum = pairwiseSum(probs);
            var below: usize = 0;
            var running: f64 = 0.0;
            for (probs) |p| {
                running += p / sum;
                if (!(running < s.top_p)) break; // the running sum never falls, so the first miss ends the count
                below += 1;
            }
            kept = @min(kept, below + 1);
        }
        if (s.min_p > 0.0) {
            const floor = scaled[0] + @log(s.min_p);
            var above: usize = 0;
            while (above < kept and scaled[above] >= floor) above += 1;
            kept = above;
        }
        return kept;
    }

    /// `nucleus_rows`: the same over the whole row, with top-p cut on integer masses. TensorFold tries the top
    /// 1,024 candidates first and reads the whole row only when the cut is not inside them; both give this.
    fn keepNucleus(self: *Sampler, logits: []const u16, highest: u16, temperature: f64, s: Sampling) usize {
        // this rule orders by value alone, so the two zeros count as one key
        self.counts[plus_zero] += self.counts[minus_zero];
        self.counts[minus_zero] = 0;
        const top: u16 = if (highest == minus_zero) plus_zero else highest;
        const reciprocal = 1.0 / temperature;
        const top_scaled = valueOf(top) * reciprocal;
        var kept = logits.len;
        if (s.cuts()) {
            var total: u64 = 0;
            var key: u16 = top;
            while (key >= lowest_finite) : (key -= 1) {
                const count = self.counts[key];
                if (count != 0) total += count * massOf(valueOf(key) * reciprocal, top_scaled);
            }
            const need = massNeeded(s.top_p, total);
            // tokens whose running mass, themselves included, is still short of the need; then one more
            var below: usize = 0;
            var running: u64 = 0;
            key = top;
            while (key >= lowest_finite) : (key -= 1) {
                const count = self.counts[key];
                if (count == 0) continue;
                const mass = massOf(valueOf(key) * reciprocal, top_scaled);
                const short = if (mass == 0) count else @min(count, (need - running - 1) / mass);
                below += short;
                if (short < count) break;
                running += count * mass;
            }
            kept = @min(kept, below + 1);
        }
        if (s.min_p > 0.0) {
            const floor = top_scaled + @log(s.min_p);
            var above: usize = 0;
            var key: u16 = top;
            while (key >= lowest_finite - 1 and above < kept) : (key -= 1) {
                if (!(valueOf(key) * reciprocal >= floor)) break;
                above += self.counts[key];
            }
            kept = @min(kept, above);
        }
        self.gather(logits, top, kept, true);
        for (self.scaled[0..kept], self.keys[0..kept]) |*value, key| value.* = valueOf(key) * reciprocal;
        return kept;
    }

    /// The first `count` tokens by (key down, id up) into `ids` and `keys`: a counting sort on the keys that
    /// reach the cut, which reads the row in id order so equal keys keep it. `fold` says the zeros were counted
    /// as one key. Consumes `counts`.
    fn gather(self: *Sampler, logits: []const u16, top: u16, count: usize, comptime fold: bool) void {
        var low = top;
        var before: u32 = 0;
        while (true) : (low -= 1) {
            const here = self.counts[low];
            self.counts[low] = before; // where this key's tokens start in the order
            before += here;
            if (before >= count) break;
        }
        const cut: Lanes = @splat(low);
        var id: usize = 0;
        while (id < logits.len) {
            // most windows hold nothing that reaches the cut
            if (id + lanes <= logits.len and !@reduce(.Or, keysOf(logits[id..][0..lanes].*, fold) >= cut)) {
                id += lanes;
                continue;
            }
            const end = @min(id + lanes, logits.len);
            while (id < end) : (id += 1) {
                var key = keyOf(logits[id]);
                if (fold and key == minus_zero) key = plus_zero;
                if (key < low) continue;
                const slot = self.counts[key];
                if (slot >= count) continue; // the cut's own key holds more tokens than are wanted
                self.ids[slot] = @intCast(id);
                self.keys[slot] = key;
                self.counts[key] = slot + 1;
            }
        }
    }
};
