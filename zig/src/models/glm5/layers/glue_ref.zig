//! GLM-5.3-Flash's glue on the host, a row at a time, in the order `zig/kernels/glm5/hc.cu` and `moe.cu` take:
//! what their tests hold the device to, and what a comparison replays. No device, no allocation.

const std = @import("std");
const numeric = @import("../../../weights/numeric.zig");
const shared = @import("../../../decode/glue_ref.zig");

const value = shared.value;
const fold4 = shared.fold4;
const branchOf = shared.branchOf;

pub const streams = 4;
/// A site's dots: 4 for the read-out, 4 for the write, 16 among the streams.
pub const mix = streams * (streams + 2);
/// The K blocks a row's dots are summed over, the values one step of a block folds, and a part's row (the 24
/// dots, then the squares' sum at 24).
pub const parts = 16;
pub const sub = 128;
pub const entry = 32;

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// The sum of `values` (a power of two of them), the upper half folded onto the lower until one is left.
/// `values` is the scratch.
pub fn fold(values: []f32) f32 {
    std.debug.assert(std.math.isPowerOfTwo(values.len));
    var n = values.len / 2;
    while (n >= 1) : (n /= 2) {
        for (values[0..n], values[n .. 2 * n]) |*low, high| low.* += high;
    }
    return values[0];
}

/// x bf16 [4 * dims], dots bf16 [24, 4 * dims] -> part [16, 32].
pub fn hcMix(x: []const u16, dots: []const u16, part: []f32) void {
    const wide = x.len;
    const block = wide / parts;
    const steps = block / sub;
    std.debug.assert(dots.len == mix * wide and part.len == parts * entry and block % sub == 0);
    var products: [sub]f32 = undefined;
    for (0..parts) |b| {
        const out = part[b * entry ..][0..entry];
        @memset(out, 0);
        for (0..mix) |m| {
            var acc: f32 = 0;
            for (0..steps) |t| {
                const at = b * block + t * sub;
                for (&products, x[at..][0..sub], dots[m * wide + at ..][0..sub]) |*p, xv, wv| p.* = value(wv) * value(xv);
                acc += fold(&products);
            }
            out[m] = acc;
        }
        for (&products, 0..) |*p, k| {
            var total: f32 = 0;
            for (0..steps) |t| {
                const v = value(x[b * block + t * sub + k]);
                total += v * v;
            }
            p.* = total;
        }
        out[mix] = fold(&products);
    }
}

/// A site's constants: base fp32 [24], scale fp32 [3], the norm's weight [dims] as stored, bf16 or fp32
/// (FP4-Local's post-attention norms).
pub const Site = struct {
    base: []const f32,
    scale: []const f32,
    norm: Norm,
    eps: f32,
    hc_eps: f32,
    iters: usize,
};

/// x bf16 [4 * dims] and its part -> out bf16 [dims], post [4], comb [4, 4]. `scratch` holds dims values.
pub fn hcPre(x: []const u16, part: []const f32, site: Site, out: []u16, post: *[streams]f32, comb: *[streams * streams]f32, scratch: []f32) void {
    const dims = out.len;
    std.debug.assert(x.len == streams * dims and scratch.len == dims);
    var dots: [mix]f32 = @splat(0);
    var ss: f32 = 0;
    for (0..parts) |b| {
        for (&dots, part[b * entry ..][0..mix]) |*dot, p| dot.* += p;
        ss += part[b * entry + mix];
    }
    const rinv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(streams * dims)) + site.eps);
    for (&dots, 0..) |*dot, m| dot.* = @mulAdd(f32, dot.* * rinv, site.scale[if (m < streams) 0 else if (m < 2 * streams) 1 else 2], site.base[m]);
    var pre: [streams]f32 = undefined;
    for (0..streams) |s| {
        pre[s] = sigmoid(dots[s]) + site.hc_eps;
        post[s] = 2.0 * sigmoid(dots[streams + s]);
    }
    for (0..streams) |i| {
        const l = dots[2 * streams + i * streams ..][0..streams];
        const most = @max(@max(l[0], l[1]), @max(l[2], l[3]));
        var e: [streams]f32 = undefined;
        for (&e, l) |*one, logit| one.* = @exp(logit - most);
        const total = fold4(e[0], e[1], e[2], e[3]);
        for (0..streams) |j| comb[i * streams + j] = e[j] / total + site.hc_eps;
    }
    for (0..site.iters) |round| {
        if (round > 0) for (0..streams) |i| {
            const c = comb[i * streams ..][0..streams];
            const total = fold4(c[0], c[1], c[2], c[3]) + site.hc_eps;
            for (c) |*one| one.* = one.* / total;
        };
        for (0..streams) |j| {
            const total = fold4(comb[j], comb[streams + j], comb[2 * streams + j], comb[3 * streams + j]) + site.hc_eps;
            for (0..streams) |i| comb[i * streams + j] = comb[i * streams + j] / total;
        }
    }
    for (out, scratch, 0..) |*o, *sq, d| {
        var v = pre[0] * value(x[d]);
        for (1..streams) |s| v = @mulAdd(f32, pre[s], value(x[s * dims + d]), v);
        o.* = numeric.bf16FromF32(v);
        sq.* = value(o.*) * value(o.*);
    }
    const inverse = 1.0 / @sqrt(fold(scratch) / @as(f32, @floatFromInt(dims)) + site.eps);
    for (out, 0..) |*o, d| o.* = numeric.bf16FromF32(site.norm.at(d) * numeric.roundBf16(value(o.*) * inverse));
}

/// A norm's weight as the checkpoint stores it; the product is the same fp32 expression either way.
pub const Norm = union(enum) {
    bf16: []const u16,
    f32: []const f32,

    pub fn at(self: Norm, d: usize) f32 {
        return switch (self) {
            .bf16 => |w| value(w[d]),
            .f32 => |w| w[d],
        };
    }
};

/// x bf16 [4 * dims] in place takes the branch, each rank's partial fp32 [dims].
pub fn hcPost(x: []u16, gathered: []const []const f32, post: *const [streams]f32, comb: *const [streams * streams]f32) void {
    const dims = x.len / streams;
    for (0..dims) |d| {
        const branch = branchOf(gathered, d);
        var v: [streams]f32 = undefined;
        for (&v, 0..) |*one, k| one.* = value(x[k * dims + d]);
        for (0..streams) |s| {
            var mixed = comb[s] * v[0];
            for (1..streams) |k| mixed = @mulAdd(f32, comb[k * streams + s], v[k], mixed);
            x[s * dims + d] = numeric.bf16FromF32(@mulAdd(f32, post[s], branch, mixed));
        }
    }
}

/// x bf16 [dims] in place: bf16(x + branch).
pub fn residualAdd(x: []u16, gathered: []const []const f32) void {
    for (x, 0..) |*one, d| one.* = numeric.bf16FromF32(value(one.*) + branchOf(gathered, d));
}

/// out bf16 [dims]: the branch alone.
pub fn branchRow(gathered: []const []const f32, out: []u16) void {
    for (out, 0..) |*one, d| one.* = numeric.bf16FromF32(branchOf(gathered, d));
}

/// x bf16 [4 * dims] -> out bf16 [dims], the streams' mean.
pub fn streamMean(x: []const u16, out: []u16) void {
    const dims = out.len;
    for (out, 0..) |*o, d| {
        var acc = value(x[d]);
        for (1..streams) |s| acc += value(x[s * dims + d]);
        o.* = numeric.bf16FromF32(acc / streams);
    }
}

/// out = bf16(w bf16(x / rms)). `scratch` holds the next power of two of x's length.
pub fn rmsnorm(x: []const u16, w: Norm, out: []u16, eps: f32, scratch: []f32) void {
    std.debug.assert(scratch.len == std.math.ceilPowerOfTwoAssert(usize, x.len));
    @memset(scratch, 0);
    for (scratch[0..x.len], x) |*sq, bits| sq.* = value(bits) * value(bits);
    const inverse = 1.0 / @sqrt(fold(scratch) / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, 0..) |*o, xv, d| o.* = numeric.bf16FromF32(w.at(d) * numeric.roundBf16(value(xv) * inverse));
}

/// A row's top experts by sigmoid(logit) + bias, the lower id among equals; a pick's weight is its sigmoid over
/// the picks' sum (when `norm`) times `scale`; the shared expert (id `logits.len`, weight 1) is the last slot.
pub fn select(logits: []const f32, bias: []const f32, picks: []i32, weights: []f32, scale: f32, norm: bool) void {
    const top = picks.len - 1;
    var total: f32 = 0;
    for (0..top) |k| {
        var best: ?usize = null;
        var most: f32 = 0;
        candidates: for (logits, bias, 0..) |logit, b, e| {
            for (picks[0..k]) |taken| if (taken == e) continue :candidates;
            const choice = sigmoid(logit) + b;
            if (best == null or choice > most) {
                best = e;
                most = choice;
            }
        }
        picks[k] = @intCast(best.?);
        weights[k] = sigmoid(logits[best.?]);
        total += weights[k];
    }
    for (weights[0..top]) |*w| w.* = (if (norm) w.* / (total + 1e-20) else w.*) * scale;
    picks[top] = @intCast(logits.len);
    weights[top] = 1.0;
}
