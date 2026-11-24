//! Samples from one rate to another (spec 138 WS-02), as the reference's `librosa.resample` does by default:
//! `soxr_hq`, a linear-phase low-pass whose pass band ends at 0.913 of the lower rate's Nyquist and whose stop band
//! starts at it. libsoxr is LGPL and is neither vendored nor ported: this is a Kaiser-windowed sinc of the same
//! band edges, its attenuation (126 dB) and pass-band end fitted to soxr_hq's output on speech-like and white
//! signals (`tools/reference/audio_cases.py` holds the figures), applied as one exact rational polyphase filter
//! where soxr cascades stages. Against soxr_hq: samples within about 1e-4, log-mel bins below 7.4 kHz within
//! 1e-4, the last bins (soxr's transition band, which its cascade shapes otherwise) within about 1e-2.
//!
//! The output has soxr's length, `round(n * to / from)`, and its time origin: output sample k is input time
//! `k * from / to`, the filter's delay removed, the input taken as zeros past both ends.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// The filter's design (fitted to soxr_hq; see the header).
pub const pass_end = 0.913;
pub const stop_begin = 1.0;
pub const attenuation = 126.0;

/// Lanes of the dot product; a phase's taps are padded to a multiple.
const lanes = 8;
const V = @Vector(lanes, f32);

/// The polyphase filter for one rate pair: `up` phases of `taps` coefficients each, reversed so a phase's dot
/// product walks the input forwards.
pub const Filter = struct {
    up: u64,
    down: u64,
    /// The prototype's length and its centre (`(len - 1) / 2`) on the upsampled grid.
    len: u64,
    centre: u64,
    /// `[up][taps]`: phase p's taps, the one for the newest input sample last; leading zeros pad each phase to
    /// `taps`, a multiple of `lanes`.
    coefs: []f32,
    taps: usize,

    pub fn init(a: Allocator, from: u32, to: u32) Allocator.Error!Filter {
        const g = std.math.gcd(from, to);
        const up: u64 = to / g;
        const down: u64 = from / g;
        const q: f64 = @floatFromInt(@max(up, down));
        const width = (stop_begin - pass_end) / q;
        const cutoff = (pass_end + stop_begin) / 2.0 / q;
        const beta = 0.1102 * (attenuation - 8.7);
        var n: u64 = @intFromFloat(@ceil((attenuation - 7.95) / (2.285 * std.math.pi * width)));
        n += 1;
        n = n / (2 * up) * 2 * up + 1;
        const centre = (n - 1) / 2;
        const per = (n + up - 1) / up;
        const taps = (per + lanes - 1) / lanes * lanes;
        const coefs = try a.alloc(f32, up * taps);
        @memset(coefs, 0);
        const half: f64 = @floatFromInt(centre);
        const norm = bessel0(beta);
        const gain: f64 = @floatFromInt(up);
        for (0..up) |p| {
            // tap i of the phase meets the input `per - 1 - i` samples before the newest
            var m: u64 = p;
            var i: usize = 0;
            while (m < n) : ({
                m += up;
                i += 1;
            }) {
                const z = @as(f64, @floatFromInt(m)) - half;
                const r = z / half;
                const x = cutoff * z;
                const sinc = if (z == 0) 1.0 else @sin(std.math.pi * x) / (std.math.pi * x);
                const h = gain * cutoff * sinc * bessel0(beta * @sqrt(@max(0.0, 1.0 - r * r))) / norm;
                coefs[p * taps + taps - 1 - i] = @floatCast(h);
            }
        }
        return .{ .up = up, .down = down, .len = n, .centre = centre, .coefs = coefs, .taps = taps };
    }

    pub fn deinit(self: Filter, a: Allocator) void {
        a.free(self.coefs);
    }

    /// soxr's output length for `n` input samples: `round(n * up / down)`.
    pub fn outputs(self: Filter, n: usize) usize {
        return @intCast((2 * @as(u128, n) * self.up + self.down) / (2 * self.down));
    }

    /// Output samples `[first, first + out.len)` of `x`.
    pub fn run(self: Filter, x: []const f32, first: usize, out: []f32) void {
        const taps = self.taps;
        for (out, first..) |*y, k| {
            const t = @as(u64, k) * self.down + self.centre;
            const p = t % self.up;
            const newest: i64 = @intCast(t / self.up);
            const oldest = newest - @as(i64, @intCast(taps)) + 1;
            const c = self.coefs[p * taps ..][0..taps];
            if (oldest >= 0 and newest < x.len) {
                const from: usize = @intCast(oldest);
                var sum: V = @splat(0);
                var i: usize = 0;
                while (i < taps) : (i += lanes) {
                    const xv: V = x[from + i ..][0..lanes].*;
                    const cv: V = c[i..][0..lanes].*;
                    sum += xv * cv;
                }
                y.* = @reduce(.Add, sum);
            } else {
                var sum: f32 = 0;
                for (c, 0..) |coef, i| {
                    const j = oldest + @as(i64, @intCast(i));
                    if (j >= 0 and j < x.len) sum += x[@intCast(j)] * coef;
                }
                y.* = sum;
            }
        }
    }
};

/// The zeroth-order modified Bessel function of the first kind, by its series (the Kaiser window's).
fn bessel0(x: f64) f64 {
    var sum: f64 = 1;
    var term: f64 = 1;
    const q = x * x / 4;
    var k: f64 = 1;
    while (k < 500) : (k += 1) {
        term *= q / (k * k);
        sum += term;
        if (term < sum * 1e-17) break;
    }
    return sum;
}

/// Outputs below this many are computed on the calling thread alone.
const parallel_outputs = 1 << 16;

/// `x` at rate `from` as samples at rate `to`; a copy where the rates are equal. The caller frees the result.
pub fn resample(a: Allocator, x: []const f32, from: u32, to: u32) Allocator.Error![]f32 {
    std.debug.assert(from > 0 and to > 0);
    if (from == to) return a.dupe(f32, x);
    const filter: Filter = try .init(a, from, to);
    defer filter.deinit(a);
    const out = try a.alloc(f32, filter.outputs(x.len));
    errdefer a.free(out);
    const cpus = std.Thread.getCpuCount() catch 1;
    const workers = @min(@min(cpus, 16), @max(1, out.len / parallel_outputs));
    if (workers <= 1) {
        filter.run(x, 0, out);
        return out;
    }
    var threads: [16]?std.Thread = @splat(null);
    const each = (out.len + workers - 1) / workers;
    for (1..workers) |w| {
        const lo = @min(out.len, w * each);
        const hi = @min(out.len, lo + each);
        threads[w] = std.Thread.spawn(.{}, Filter.run, .{ filter, x, lo, out[lo..hi] }) catch null;
        if (threads[w] == null) filter.run(x, lo, out[lo..hi]);
    }
    filter.run(x, 0, out[0..@min(out.len, each)]);
    for (threads[1..workers]) |t| if (t) |thread| thread.join();
    return out;
}
