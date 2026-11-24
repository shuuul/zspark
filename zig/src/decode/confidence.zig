//! What a draft head gives a token it drafts: its probability from a row of logits on the host, which a
//! model's draft rule holds against a threshold (Flash Next's `mtp_rule.Rule`, GLM's `mtp.Head`). Host only.

const std = @import("std");
const Allocator = std.mem.Allocator;
const numeric = @import("../weights/numeric.zig");

/// bf16's -inf: a logit a head did not score, or a constraint took.
pub const minus_infinity: u16 = 0xff80;

/// A logit's value.
pub fn value(bits: u16) f32 {
    return numeric.f32FromBf16(bits);
}

/// The probability, at temperature 1, the logits give `token`. Logits more than 24 below the largest are left
/// out of the sum: together they are under 1e-5 of it for any vocabulary here.
pub fn probability(logits: []const u16, token: usize) f32 {
    return @floatCast(@exp(numeric.f32FromBf16(logits[token]) - normalizer(logits)));
}

/// The probability a logit (bf16 bits) has where `sum` is the row's `normalizer`: `probability` before its
/// narrowing, for a head that holds the sum from `Normalizer.of`.
pub fn fromSum(bits: u16, sum: f64) f64 {
    return @exp(@as(f64, numeric.f32FromBf16(bits)) - sum);
}

/// The probability of a logit read from the device's candidates: `bits` is the logit, `largest_bits` the
/// row's largest, `sum` the device's sum of exp(v - largest) (`Picked.sum`); `normalizer` from them.
pub fn fromLeading(bits: u16, largest_bits: u16, sum: f64) f64 {
    return @exp(@as(f64, numeric.f32FromBf16(bits)) - (@as(f64, numeric.f32FromBf16(largest_bits)) + @log(sum)));
}

/// The logarithm of the sum `probability` divides by: a logit less it is the logarithm of its probability.
pub fn normalizer(logits: []const u16) f64 {
    var largest: f32 = -std.math.inf(f32);
    for (logits) |bits| largest = @max(largest, numeric.f32FromBf16(bits));
    var sum: f64 = 0;
    for (logits) |bits| {
        const v = numeric.f32FromBf16(bits);
        if (v > largest - 24) sum += @exp(v - largest);
    }
    return largest + @log(sum);
}

/// `normalizer` for a head that asks once a draft, to the bit: the same terms added in the same order, but a
/// term is computed once for each bf16 pattern the row holds. A draft head's row of 80,000 logits holds a few
/// thousand patterns, and an `exp` for each logit was most of what a draft cost the host (spec 028).
pub const Normalizer = struct {
    /// [65536] exp(v - largest) for the bf16 pattern v, where `stamps` says it is this row's: the fp64
    /// exponential of the fp32 difference, as `normalizer`'s sum takes it
    terms: []f64,
    /// [65536] the row (`row`) whose `terms` entry this is
    stamps: []u32,
    row: u32 = 0,

    pub fn init(gpa: Allocator) Allocator.Error!Normalizer {
        const terms = try gpa.alloc(f64, 1 << 16);
        errdefer gpa.free(terms);
        const stamps = try gpa.alloc(u32, 1 << 16);
        @memset(stamps, 0);
        return .{ .terms = terms, .stamps = stamps };
    }

    pub fn deinit(self: *Normalizer, gpa: Allocator) void {
        gpa.free(self.terms);
        gpa.free(self.stamps);
        self.* = undefined;
    }

    /// `normalizer(logits)`. Logits left out of `logits` because they are -inf change nothing: -inf is never
    /// the largest of a row that has a finite logit, and adds no term.
    pub fn of(self: *Normalizer, logits: []const u16) f64 {
        // the terms are of one row's largest logit: a new row starts them over
        self.row +%= 1;
        if (self.row == 0) {
            @memset(self.stamps, 0);
            self.row = 1;
        }
        var largest: f32 = -std.math.inf(f32);
        for (logits) |bits| largest = @max(largest, numeric.f32FromBf16(bits));
        var sum: f64 = 0;
        for (logits) |bits| {
            if (self.stamps[bits] != self.row) {
                const v = numeric.f32FromBf16(bits);
                // a pattern outside the sum is held as NaN, which no term is
                self.terms[bits] = if (v > largest - 24) @exp(@as(f64, v - largest)) else std.math.nan(f64);
                self.stamps[bits] = self.row;
            }
            const term = self.terms[bits];
            if (term == term) sum += term;
        }
        return largest + @log(sum);
    }
};
