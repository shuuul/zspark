//! The log-mel features of 16 kHz samples (spec 138 WS-02), as transformers 4.57.6's `WhisperFeatureExtractor`
//! computes them with Qwen3-ASR-1.7B's `preprocessor_config.json` (128 bins, `n_fft` 400, hop 160, dither 0) and
//! its processor's `padding=True, truncation=False`, so no padding to 30 s: frames of 400 samples every 160, the
//! samples reflected by 200 at both ends (`center`), a periodic Hann window, the power of a 400-point DFT, the
//! Slaney mel filters (Slaney scale and area norm, 0 to 8 kHz), `log10` of at least 1e-10, every value raised to
//! the clip's maximum less 8, then `(x + 4) / 4`. The last frame is dropped as Whisper's extractor drops it, so
//! `n` samples give `n / 160` frames, the count the processor's attention mask gives.
//!
//! The DFT is a mixed-radix Stockham FFT (400 = 4 * 4 * 5 * 5) in f32, eight frames a vector; the reference's
//! torch path is f32 as well (`torch.stft`). The maximum is the whole clip's, so a frame's value depends on the
//! clip around it: the features are made once a clip, not streamed.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const rate = 16000;
pub const n_fft = 400;
pub const hop = 160;
pub const bins = 128;
/// The power spectrum's bins, `n_fft / 2 + 1`.
pub const freqs = n_fft / 2 + 1;

/// Frames for `n` samples at 16 kHz: `n / 160` (Whisper's extractor drops the STFT's last frame).
pub fn frames(n: usize) usize {
    return n / hop;
}

/// The audio tower's output tokens for `n` mel frames (Qwen3-ASR's `_get_feat_extract_output_lengths`): 13 a full
/// chunk of 100 frames, and for the rest the three stride-2 convolutions' length, `(r - 1) // 2 + 1` thrice with
/// Python's floor division, so a remainder of 0 gives 0.
pub fn tokens(n: usize) usize {
    const r: i64 = @intCast(n % 100);
    var l = r;
    for (0..3) |_| l = @divFloor(l - 1, 2) + 1;
    return @as(usize, @intCast(l)) + n / 100 * 13;
}

const lanes = 8;
const V = @Vector(lanes, f32);
const radices = [_]usize{ 4, 4, 5, 5 };

/// What the features need besides the samples, made once: the window, the FFT's twiddles and the filters.
pub const Tables = struct {
    window: [n_fft]f32,
    /// Stage s's twiddles: `[n_fft / p][p]` as cos and sin of `-2 pi k r / (span * p)`.
    twiddle_re: [radices.len][n_fft]f32,
    twiddle_im: [radices.len][n_fft]f32,
    /// The roots of unity of a radix-p DFT, `exp(-2 pi i j / p)`.
    root_re: [6][6]f32,
    root_im: [6][6]f32,
    /// Filter m's non-zero weights over power bins `[first[m], first[m] + count[m])`.
    first: [bins]usize,
    count: [bins]usize,
    weights: [bins][freqs]f32,

    /// Fills the tables in place (about 110 KB; `std.heap` them rather than the stack).
    pub fn init(t: *Tables) void {
        for (&t.window, 0..) |*w, i| {
            const x: f64 = @floatFromInt(i);
            w.* = @floatCast(0.5 - 0.5 * @cos(2.0 * std.math.pi * x / n_fft));
        }
        var span: usize = 1;
        for (radices, 0..) |p, s| {
            for (0..n_fft / p) |j| {
                const k = j % span;
                for (0..p) |r| {
                    const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(k * r)) / @as(f64, @floatFromInt(span * p));
                    t.twiddle_re[s][j * p + r] = @floatCast(@cos(angle));
                    t.twiddle_im[s][j * p + r] = @floatCast(@sin(angle));
                }
            }
            span *= p;
        }
        for (2..6) |p| for (0..p) |j| {
            const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(p));
            t.root_re[p][j] = @floatCast(@cos(angle));
            t.root_im[p][j] = @floatCast(@sin(angle));
        };
        filters(t);
    }
};

/// transformers' `hertz_to_mel` and `mel_to_hertz` on the Slaney scale.
fn hertzToMel(f: f64) f64 {
    const logstep = 27.0 / @log(6.4);
    if (f >= 1000.0) return 15.0 + @log(f / 1000.0) * logstep;
    return 3.0 * f / 200.0;
}

fn melToHertz(m: f64) f64 {
    const logstep = @log(6.4) / 27.0;
    if (m >= 15.0) return 1000.0 * @exp(logstep * (m - 15.0));
    return 200.0 * m / 3.0;
}

/// transformers' `mel_filter_bank(201, 128, 0, 8000, 16000, norm="slaney", mel_scale="slaney")`: triangles between
/// mel-spaced edges over the bins' frequencies `linspace(0, 8000, 201)`, each scaled by `2 / (its upper edge - its
/// lower edge)`, computed in f64 and kept as f32 (the torch path's `mel_filters.to(float32)`).
fn filters(t: *Tables) void {
    var edges: [bins + 2]f64 = undefined;
    const lo = hertzToMel(0.0);
    const hi = hertzToMel(8000.0);
    const step = (hi - lo) / @as(f64, bins + 1);
    for (&edges, 0..) |*e, i| e.* = melToHertz(if (i == bins + 1) hi else lo + @as(f64, @floatFromInt(i)) * step);
    for (0..bins) |m| {
        const norm = 2.0 / (edges[m + 2] - edges[m]);
        t.first[m] = freqs;
        t.count[m] = 0;
        for (0..freqs) |k| {
            const f = 8000.0 * @as(f64, @floatFromInt(k)) / @as(f64, freqs - 1);
            const down = -(edges[m] - f) / (edges[m + 1] - edges[m]);
            const up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            const w = @max(0.0, @min(down, up)) * norm;
            t.weights[m][k] = @floatCast(w);
            if (w > 0) {
                if (t.first[m] == freqs) t.first[m] = k;
                t.count[m] = k + 1 - t.first[m];
            }
        }
        if (t.first[m] == freqs) t.first[m] = 0;
    }
}

/// The sample `i` of the padded clip (`i` from -200): numpy's and torch's `reflect`, the edge not repeated.
fn reflected(x: []const f32, i: i64) f32 {
    const n: i64 = @intCast(x.len);
    if (n == 1) return x[0];
    const period = 2 * (n - 1);
    var j = @mod(i, period);
    if (j >= n) j = period - j;
    return x[@intCast(j)];
}

/// Eight frames' 400-point DFTs at once, lane f the frame f; `re` and `im` in, the spectrum out in natural order.
fn fft(t: *const Tables, re: *[n_fft]V, im: *[n_fft]V) void {
    var bre: [n_fft]V = undefined;
    var bim: [n_fft]V = undefined;
    var in_re = re;
    var in_im = im;
    var out_re = &bre;
    var out_im = &bim;
    var span: usize = 1;
    for (radices, 0..) |p, s| {
        const stride = n_fft / p;
        for (0..stride) |j| {
            const k = j % span;
            var vre: [5]V = undefined;
            var vim: [5]V = undefined;
            for (0..p) |r| {
                const xr = in_re[j + r * stride];
                const xi = in_im[j + r * stride];
                const wr: V = @splat(t.twiddle_re[s][j * p + r]);
                const wi: V = @splat(t.twiddle_im[s][j * p + r]);
                vre[r] = xr * wr - xi * wi;
                vim[r] = xr * wi + xi * wr;
            }
            const base = (j / span) * span * p + k;
            for (0..p) |q| {
                var sr: V = vre[0];
                var si: V = vim[0];
                for (1..p) |r| {
                    const at = (q * r) % p;
                    const wr: V = @splat(t.root_re[p][at]);
                    const wi: V = @splat(t.root_im[p][at]);
                    sr += vre[r] * wr - vim[r] * wi;
                    si += vre[r] * wi + vim[r] * wr;
                }
                out_re[base + q * span] = sr;
                out_im[base + q * span] = si;
            }
        }
        std.mem.swap(*[n_fft]V, &in_re, &out_re);
        std.mem.swap(*[n_fft]V, &in_im, &out_im);
        span *= p;
    }
    // four stages: the result is back in `re` and `im`
    comptime std.debug.assert(radices.len % 2 == 0);
}

/// The `log10` mel values of frames `[first, first + count)` into `out` (`[bins][stride]`, column `first + f`),
/// before the clip's maximum is applied; returns their maximum.
fn logMel(t: *const Tables, x: []const f32, first: usize, count: usize, out: []f32, stride: usize) f32 {
    var top: f32 = -std.math.inf(f32);
    var f0 = first;
    while (f0 < first + count) : (f0 += lanes) {
        var re: [n_fft]V = undefined;
        var im: [n_fft]V = undefined;
        const live = @min(lanes, first + count - f0);
        for (0..n_fft) |i| {
            var v: [lanes]f32 = @splat(0);
            for (0..live) |f| {
                const at = @as(i64, @intCast((f0 + f) * hop + i)) - n_fft / 2;
                const s = if (at >= 0 and at < x.len) x[@intCast(at)] else reflected(x, at);
                v[f] = s * t.window[i];
            }
            re[i] = v;
            im[i] = @splat(0);
        }
        fft(t, &re, &im);
        var power: [freqs]V = undefined;
        for (&power, 0..) |*pw, k| pw.* = re[k] * re[k] + im[k] * im[k];
        for (0..bins) |m| {
            var sum: V = @splat(0);
            const w = t.weights[m][t.first[m]..][0..t.count[m]];
            for (w, t.first[m]..) |weight, k| sum += power[k] * @as(V, @splat(weight));
            const floor: V = @splat(1e-10);
            const lg = @log10(@max(sum, floor));
            const arr: [lanes]f32 = lg;
            for (0..live) |f| {
                out[m * stride + f0 + f] = arr[f];
                top = @max(top, arr[f]);
            }
        }
    }
    return top;
}

/// Frames below this many are computed on the calling thread alone.
const parallel_frames = 4096;

const Part = struct {
    t: *const Tables,
    x: []const f32,
    first: usize,
    count: usize,
    out: []f32,
    stride: usize,
    top: f32 = -std.math.inf(f32),

    fn run(self: *Part) void {
        self.top = logMel(self.t, self.x, self.first, self.count, self.out, self.stride);
    }
};

/// The features of `x` (16 kHz) as `[128][frames(x.len)]`, bin-major as the extractor's `input_features[0]`; empty
/// where `x` holds fewer than 160 samples. The caller frees the result.
pub fn features(a: Allocator, tables: *const Tables, x: []const f32) Allocator.Error![]f32 {
    const t_len = frames(x.len);
    const out = try a.alloc(f32, bins * t_len);
    if (t_len == 0) return out;
    const cpus = std.Thread.getCpuCount() catch 1;
    const workers = @min(@min(cpus, 16), @max(1, t_len / parallel_frames));
    var parts: [16]Part = undefined;
    var threads: [16]?std.Thread = @splat(null);
    const each = (t_len + workers - 1) / workers;
    for (0..workers) |w| {
        const lo = @min(t_len, w * each);
        parts[w] = .{ .t = tables, .x = x, .first = lo, .count = @min(t_len, lo + each) - lo, .out = out, .stride = t_len };
    }
    for (1..workers) |w| {
        threads[w] = std.Thread.spawn(.{}, Part.run, .{&parts[w]}) catch null;
        if (threads[w] == null) parts[w].run();
    }
    parts[0].run();
    var top: f32 = -std.math.inf(f32);
    for (parts[0..workers], threads[0..workers]) |*part, thread| {
        if (thread) |th| th.join();
        top = @max(top, part.top);
    }
    const least = top - 8.0;
    for (out) |*v| v.* = (@max(v.*, least) + 4.0) / 4.0;
    return out;
}
