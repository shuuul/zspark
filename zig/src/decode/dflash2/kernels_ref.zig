//! DFlash2's kernels on the host, a row or a query at a time, in the order `zig/kernels/common/dflash2.cu`
//! takes: what its tests hold the device to, bit for bit. Every fused multiply-add of the kernels is a
//! `@mulAdd` here, and the sine, cosine and exponential are the kernels' own fp64 steps, not the library's. No
//! device, no allocation.
//!
//! WS-03's Qwen drafter imports `dconvRow`, `prepHead`, `attendRow`, `rescore`, `Block`, `head`, `tile` and `chunk`.

const std = @import("std");
const numeric = @import("../../weights/numeric.zig");

fn fold(values: []f32) f32 {
    std.debug.assert(std.math.isPowerOfTwo(values.len));
    var n = values.len / 2;
    while (n >= 1) : (n /= 2) {
        for (values[0..n], values[n .. 2 * n]) |*low, high| low.* += high;
    }
    return values[0];
}

/// A head's values, and half of them: a rotary pair is values d and d + half.
pub const head = 128;
pub const half = head / 2;

fn value(bits: u16) f32 {
    return numeric.f32FromBf16(bits);
}

/// 1 / n!, n to 18: the factorial is exact in fp64 and the quotient is rounded once.
fn inverseFactorial(n: usize) f64 {
    var f: f64 = 1.0;
    for (1..n + 1) |i| f *= @floatFromInt(i);
    return 1.0 / f;
}

/// The coefficient of x^n in sin or cos.
fn term(n: usize) f64 {
    return if ((n / 2) % 2 == 1) -inverseFactorial(n) else inverseFactorial(n);
}

/// theta^(-d / 64) as fp32: the square roots of theta one after another, the product of those whose bit is
/// set in d from the highest down, and its inverse rounded to fp32.
pub fn invFreq(theta: f64, d: usize) f32 {
    std.debug.assert(d < half);
    var r = theta;
    var p: f64 = 1.0;
    var b: usize = 6;
    while (b > 0) {
        b -= 1;
        r = @sqrt(r);
        if ((d >> @intCast(b)) & 1 == 1) p = p * r;
    }
    return @floatCast(1.0 / p);
}

pub const SinCos = struct { sin: f32, cos: f32 };

/// sin and cos of a phase (0 to 2^24) in fp64, rounded to fp32: the quadrant by floor(x 2/pi + 1/2), the rest
/// by pi/2 in two parts, the series to x^15 and x^16 by Horner.
pub fn sinCos(x: f64) SinCos {
    const k = @floor(@mulAdd(f64, x, 0x1.45f306dc9c883p-1, 0.5));
    var r = @mulAdd(f64, -k, 0x1.921fb54442d18p+0, x);
    r = @mulAdd(f64, -k, 0x1.1a62633145c07p-54, r);
    const z = r * r;
    var p = term(15);
    var n: usize = 13;
    while (n >= 3) : (n -= 2) p = @mulAdd(f64, p, z, term(n));
    const s = @mulAdd(f64, r * z, p, r);
    p = term(16);
    n = 14;
    while (n >= 2) : (n -= 2) p = @mulAdd(f64, p, z, term(n));
    const c = @mulAdd(f64, z, p, 1.0);
    const quadrant: u2 = @truncate(@as(u64, @bitCast(@as(i64, @intFromFloat(k)))));
    return switch (quadrant) {
        0 => .{ .sin = @floatCast(s), .cos = @floatCast(c) },
        1 => .{ .sin = @floatCast(c), .cos = @floatCast(-s) },
        2 => .{ .sin = @floatCast(-s), .cos = @floatCast(-c) },
        3 => .{ .sin = @floatCast(-c), .cos = @floatCast(s) },
    };
}

/// The rotation of pair `d` at `position`: the phase is fp32(position) times `invFreq` in fp32.
pub fn rotary(theta: f64, position: usize, d: usize) SinCos {
    const phase = @as(f32, @floatFromInt(position)) * invFreq(theta, d);
    return sinCos(phase);
}

/// A row of the convolution: x bf16 [dims] and the row before it (null for the block's first), the row's dyn
/// bf16 [2, 2, groups] (branch, tap, group), base bf16 [2, 2, dims] -> out bf16 [dims], with the residual
/// added when there is one.
pub fn dconvRow(x: []const u16, before: ?[]const u16, dyn: []const u16, base: []const u16, branch: usize, residual: ?[]const u16, out: []u16) void {
    const dims = x.len;
    const groups = dyn.len / 4;
    std.debug.assert(base.len == 4 * dims and dims % groups == 0 and branch < 2);
    const size = dims / groups;
    for (out, 0..) |*o, c| {
        const k0 = numeric.roundBf16(value(base[branch * 2 * dims + c]) + value(dyn[branch * 2 * groups + c / size]));
        const k1 = numeric.roundBf16(value(base[(branch * 2 + 1) * dims + c]) + value(dyn[(branch * 2 + 1) * groups + c / size]));
        const prev: f32 = if (before) |row| value(row[c]) else 0.0;
        var y = numeric.roundBf16(@mulAdd(f32, prev, k1, value(x[c]) * k0));
        if (residual) |res| y = numeric.roundBf16(value(res[c]) + y);
        o.* = numeric.bf16FromF32(y);
    }
}

/// A query or key head of a row at `position`: normed with `norm` bf16 [128] and rotated.
pub fn prepHead(x: *const [head]u16, norm: *const [head]u16, position: usize, eps: f32, theta: f64, out: *[head]u16) void {
    var sa: [half]f32 = undefined;
    var sb: [half]f32 = undefined;
    for (&sa, &sb, x[0..half], x[half..]) |*a, *b, low, high| {
        a.* = value(low) * value(low);
        b.* = value(high) * value(high);
    }
    const rstd = 1.0 / @sqrt((fold(&sa) + fold(&sb)) / @as(f32, head) + eps);
    for (0..half) |d| {
        const a = numeric.roundBf16(value(x[d]) * rstd * value(norm[d]));
        const b = numeric.roundBf16(value(x[half + d]) * rstd * value(norm[half + d]));
        const turn = rotary(theta, position, d);
        out[d] = numeric.bf16FromF32(@mulAdd(f32, -b, turn.sin, a * turn.cos));
        out[half + d] = numeric.bf16FromF32(@mulAdd(f32, a, turn.sin, b * turn.cos));
    }
}

/// A block of `rows` rows whose first stands at `pos`, over a context a query sees `window` positions back.
pub const Block = struct {
    pos: usize,
    rows: usize,
    window: usize,
    causal: bool = false,

    /// The positions row `row`'s query sees, first to last, all of them.
    pub fn seen(self: Block, row: usize) struct { first: usize, last: usize } {
        const qpos = self.pos + row;
        return .{ .first = @min(qpos -| self.window, self.pos), .last = if (self.causal) qpos else self.pos + self.rows - 1 };
    }
};

/// The keys a step of the attention's online softmax takes, and those a chunk of them holds
/// (`zig/kernels/common/dflash2_attend.cu`'s TILE and CHUNK).
pub const tile = 32;
pub const chunk = 256;

/// The query q bf16 [128] of the block's row `row` over a KV head's keys and values by position, bf16
/// [positions, 128] -> out bf16 [128]: `dflash2_attend.cu`'s steps. The keys count from the first position the
/// block's first row sees, a tile at a time with a running largest score, a probability rounded to bf16 before
/// it weighs a value, the chunks merged in order. What the device does otherwise is the order of its fp32 sums
/// inside an mma and its `expf`: a rounding apart, which may turn a bf16 probability to its neighbour.
pub fn attendRow(q: *const [head]u16, keys: []const u16, values: []const u16, block: Block, row: usize, out: *[head]u16) void {
    const range = block.seen(row);
    const base = block.seen(0).first;
    const total = block.pos + block.rows - base;
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, head)));
    const none = -std.math.inf(f32);
    var m: f32 = none;
    var l: f32 = 0.0;
    var o: [head]f32 = @splat(0.0);
    var start: usize = 0;
    while (start < total) : (start += chunk) {
        var cm: f32 = none;
        var cl: f32 = 0.0;
        var co: [head]f32 = @splat(0.0);
        var t0 = start;
        while (t0 < @min(start + chunk, total)) : (t0 += tile) {
            var sc: [tile]f32 = @splat(none);
            var most: f32 = none;
            for (&sc, 0..) |*one, j| {
                const position = base + t0 + j;
                if (t0 + j >= total or position < range.first or position > range.last) continue;
                var dot: f32 = 0.0;
                for (q, keys[position * head ..][0..head]) |a, b| dot = @mulAdd(f32, value(a), value(b), dot);
                one.* = dot * scale;
                most = @max(most, one.*);
            }
            if (most == none) continue;
            const next = @max(cm, most);
            const alpha: f32 = if (cm == none) 0.0 else @exp(cm - next);
            cm = next;
            var p: [tile]f32 = undefined;
            for (&p, sc) |*one, score| one.* = if (score == none) 0.0 else @exp(score - next);
            // a warp's 8 keys: pairs, the pairs as a tree; then the four warps in order
            var sum: f32 = undefined;
            for (0..4) |w| {
                const e = p[w * 8 ..][0..8];
                const eight = ((e[0] + e[1]) + (e[2] + e[3])) + ((e[4] + e[5]) + (e[6] + e[7]));
                sum = if (w == 0) eight else sum + eight;
            }
            cl = cl * alpha + sum;
            for (&co, 0..) |*one, d| {
                var acc: f32 = 0.0;
                for (p, 0..) |weight, j| {
                    if (weight == 0.0) continue;
                    acc = @mulAdd(f32, numeric.roundBf16(weight), value(values[(base + t0 + j) * head + d]), acc);
                }
                one.* = one.* * alpha + acc;
            }
        }
        if (!(cl > 0.0)) continue;
        const next = @max(m, cm);
        const a: f32 = if (m == none) 0.0 else @exp(m - next);
        const b: f32 = @exp(cm - next);
        for (&o, co) |*one, part| one.* = one.* * a + part * b;
        l = l * a + cl * b;
        m = next;
    }
    for (out, o) |*one, sum| one.* = numeric.bf16FromF32(sum / l);
}

/// The four-bit head's picks of a block's rows, dotted again against the stored rows (`kernels.rescore`):
/// tokens [rows x top] of whole-vocabulary ids, head_rows [half x dims] the rank's rows of the vocabulary as
/// the main checkpoint stores them, final [rows x dims] the normed draft rows, `offset` the rank's first
/// token -> values [rows x top], each pick's dot with its row, one `@mulAdd` a value in the row's order. A
/// pick out of the rank's half is -inf.
pub fn rescore(tokens: []const i32, head_rows: []const u16, final: []const u16, values: []f32, top: usize, dims: usize, offset: usize) void {
    const rows = final.len / dims;
    const rows_of_head = head_rows.len / dims;
    std.debug.assert(tokens.len == rows * top and values.len == rows * top and head_rows.len % dims == 0);
    for (0..rows) |row| for (0..top) |at| {
        const place = row * top + at;
        const index = @as(i64, tokens[place]) - @as(i64, @intCast(offset));
        if (index < 0 or index >= rows_of_head) {
            values[place] = -std.math.inf(f32);
            continue;
        }
        var acc: f32 = 0.0;
        for (0..dims) |d| acc = @mulAdd(f32, value(final[row * dims + d]), value(head_rows[@as(usize, @intCast(index)) * dims + d]), acc);
        values[place] = acc;
    };
}
