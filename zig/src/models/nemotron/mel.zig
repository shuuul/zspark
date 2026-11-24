//! The Nemotron-3.5-Transcribe perception's log-mel features and a clip's count of audio rows (spec 151 WS-03),
//! host f32 as NeMo runs its `AudioToMelSpectrogramPreprocessor` (it insists on fp32, `audio_preprocessing.py`
//! `AudioPreprocessor.forward`, then casts to the module's bf16; NeMo commit 774bc30): pre-emphasis 0.97 on the
//! samples (`x[0]` kept, `x[t] - 0.97 x[t-1]`; FilterbankFeatures.forward, `features.py`), a centered STFT of
//! `n_fft` 512 with `n_fft`/2 zeros at both ends and the checkpoint's own hann window of `win` samples centered
//! in the frame (torch.stft, `pad_mode="constant"`), the magnitude squared (`sqrt` then `mag_power` 2), the
//! checkpoint's `fb` mel table, `log(x + 2^-24)` ("add" zero guard). No dither (training only), no normalization
//! (`normalize` null), no `pad_to` (0). A clip is chunked into `chunk_seconds` spans first (the row count of
//! `nemo/collections/common/data/lhotse/audio_token_estimator.py` and `speechlm2/parts/encoder_chunking.py`),
//! so the features here are one chunk's.
//!
//! `fb` and `window` are the checkpoint's stored `perception.preprocessor.featurizer.*` tensors (bf16, read
//! into f32 slices), not recomputed tables. The FFT is a radix-2 Cooley-Tukey in f32, twiddles from f64;
//! `n_fft` must be a power of two.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// The featurizer's shape, from `config.zig`'s `Perception`.
pub const Params = struct {
    n_fft: usize,
    /// `window_size` in samples (400).
    window: usize,
    /// `window_stride` in samples (160).
    hop: usize,
    /// The mel bins (128).
    features: usize,
    /// Sample rate (16,000).
    rate: usize,

    /// The STFT's pad a side (`n_fft`/2, centered).
    pub fn pad(self: Params) usize {
        return self.n_fft / 2;
    }

    /// Mel frames of `samples` (`get_seq_len`: `(n + 2 pad - n_fft) / hop`).
    pub fn frames(self: Params, samples: usize) usize {
        return (samples + 2 * self.pad() - self.n_fft) / self.hop;
    }

    /// The frames `torch.stft` really emits over the padded signal (`floor(n / hop) + 1`), one more than
    /// `frames` reports as valid: the last is a partial frame `get_seq_len` masks out (`features.py`'s
    /// pad-to-`seq_len` step). The ParallelExpertEncoder tiles this padded width, not `frames`.
    pub fn stftFrames(self: Params, samples: usize) usize {
        return self.frames(samples) + 1;
    }
};

/// A chunk's span in samples: `round(chunk_seconds * rate)` clamped to at least the smallest count that gives
/// two frames (a tail under it folds into the previous chunk; `encoder_chunking._split_audio_into_chunks`).
pub fn chunkSpans(arena: Allocator, p: Params, seconds: f64, samples: usize) Allocator.Error![]const [2]usize {
    var min_chunk = p.hop;
    for (0..16) |_| {
        if (p.frames(min_chunk) >= 2) break;
        min_chunk += p.hop;
    } else min_chunk = 2 * p.hop;
    const chunk = @max(1, @max(min_chunk, @as(usize, @intFromFloat(@round(seconds * @as(f64, @floatFromInt(p.rate)))))));
    var spans: std.ArrayList([2]usize) = .empty;
    var begin: usize = 0;
    while (begin < samples) : (begin += chunk) {
        try spans.append(arena, .{ begin, @min(begin + chunk, samples) });
    }
    if (spans.items.len == 0) try spans.append(arena, .{ 0, 0 });
    var items = spans.items;
    if (items.len > 1 and items[items.len - 1][1] - items[items.len - 1][0] < min_chunk) {
        items[items.len - 2][1] = items[items.len - 1][1];
        items = items[0 .. items.len - 1];
    }
    return items;
}

/// A clip's audio rows: the estimator's count, one chunk's `ceil(frames / stacking)` summed over the spans.
pub fn rows(arena: Allocator, p: Params, seconds: f64, stacking: usize, samples: usize) Allocator.Error!usize {
    const spans = try chunkSpans(arena, p, seconds, samples);
    var count: usize = 0;
    for (spans) |span| count += @max(1, (p.frames(span[1] - span[0]) + stacking - 1) / stacking);
    return count;
}

/// The features' working memory: one frame's DFT scratch and the pre-emphasized chunk.
pub const Scratch = struct {
    re: []f32,
    im: []f32,
    order: []u32,
    signal: []f32,

    pub fn init(a: Allocator, p: Params, samples: usize) (Allocator.Error || error{UnsupportedShape})!Scratch {
        return .{
            .re = try a.alloc(f32, p.n_fft),
            .im = try a.alloc(f32, p.n_fft),
            .order = try bitReverse(a, p.n_fft),
            .signal = try a.alloc(f32, samples),
        };
    }

    pub fn deinit(self: *Scratch, a: Allocator) void {
        a.free(self.re);
        a.free(self.im);
        a.free(self.order);
        a.free(self.signal);
        self.* = undefined;
    }
};

fn bitReverse(a: Allocator, n: usize) (Allocator.Error || error{UnsupportedShape})![]u32 {
    const order = try a.alloc(u32, n);
    var bits: usize = 0;
    while (@as(usize, 1) << @intCast(bits) < n) bits += 1;
    if (@as(usize, 1) << @intCast(bits) != n) return error.UnsupportedShape;
    for (order, 0..) |*o, i| o.* = @bitReverse(@as(u32, @intCast(i))) >> @intCast(32 - @as(u6, @intCast(bits)));
    return order;
}

fn fft(p: Params, order: []const u32, re: []f32, im: []f32) void {
    const n = p.n_fft;
    // bit-reversed gather, then lg(n) stages of the radix-2 butterfly
    {
        const r = order;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const j = r[i];
            if (j > i) {
                std.mem.swap(f32, &re[i], &re[j]);
                std.mem.swap(f32, &im[i], &im[j]);
            }
        }
    }
    var span: usize = 1;
    while (span < n) : (span *= 2) {
        const step = @as(f64, -std.math.pi) / @as(f64, @floatFromInt(span));
        var base: usize = 0;
        while (base < n) : (base += 2 * span) {
            for (0..span) |k| {
                const angle = step * @as(f64, @floatFromInt(k));
                const wr: f32 = @floatCast(@cos(angle));
                const wi: f32 = @floatCast(@sin(angle));
                const i = base + k;
                const j = i + span;
                const xr = re[j] * wr - im[j] * wi;
                const xi = re[j] * wi + im[j] * wr;
                re[j] = re[i] - xr;
                im[j] = im[i] - xi;
                re[i] += xr;
                im[i] += xi;
            }
        }
    }
}

/// One chunk's log-mel into `out` `[frames, features]` row-major (a row a frame, the encoder's (T, C) after its
/// transpose). `samples` the chunk's samples, `signal` scratch its length, `window` [`p.window`] and `fb`
/// [`p.features`, `n_fft`/2 + 1] the checkpoint's tables in f32.
pub fn features(p: Params, scratch: *Scratch, fb: []const f32, window: []const f32, samples: []const f32, out: []f32) void {
    const t_len = p.frames(samples.len);
    const freqs = p.n_fft / 2 + 1;
    // pre-emphasis (the pad's zeros stay zero; `FilterbankFeatures.forward`)
    const x = scratch.signal[0..samples.len];
    if (samples.len > 0) {
        x[0] = samples[0];
        for (1..samples.len) |i| x[i] = samples[i] - 0.97 * samples[i - 1];
    }
    const off = (p.n_fft - p.window) / 2;
    for (0..t_len) |f| {
        for (0..p.n_fft) |i| {
            const at = @as(i64, @intCast(f * p.hop + i)) - @as(i64, @intCast(p.pad()));
            scratch.re[i] = if (i >= off and i < off + p.window and at >= 0 and at < x.len) x[@intCast(at)] * window[i - off] else 0;
            scratch.im[i] = 0;
        }
        fft(p, scratch.order, scratch.re, scratch.im);
        for (0..p.features) |m| {
            var sum: f32 = 0;
            for (0..freqs) |k| sum += fb[m * freqs + k] * (scratch.re[k] * scratch.re[k] + scratch.im[k] * scratch.im[k]);
            out[f * p.features + m] = @log(sum + std.math.pow(f32, 2, -24));
        }
    }
}

/// `normalize_type="per_feature"` on `[T, features]` row-major as `features.py`'s `normalize_batch` runs it:
/// each mel bin's mean and sample std (ddof 1) over the first `valid` frames, `(x - mean) / (std + 1e-5)`
/// there, zeros at and past `valid` (the STFT's partial tail and any padding). The ParallelExpertEncoder
/// applies it to the whole clip's mel before its windows (`parallel_expert_encoder.py`'s
/// `_forward_online_impl`); NeMo refuses a `seq_len` under 2 upstream, the lone-frame std's NaN->0 then
/// `+ 1e-5` is the same as this denominator clamp.
pub fn normalizePerFeature(p: Params, x: []f32, valid: usize) void {
    const at = x.len / p.features;
    for (0..p.features) |m| {
        var mean: f64 = 0;
        for (0..@min(valid, at)) |f| mean += x[f * p.features + m];
        mean /= @floatFromInt(@max(1, @min(valid, at)));
        var sq: f64 = 0;
        for (0..@min(valid, at)) |f| {
            const d = x[f * p.features + m] - mean;
            sq += d * d;
        }
        const denom = @max(1, @min(valid, at) - 1);
        const sigma = @sqrt(sq / @as(f64, @floatFromInt(denom))) + 1e-5;
        for (0..at) |f| {
            const v = x[f * p.features + m];
            x[f * p.features + m] = if (f < valid) @floatCast((@as(f64, v) - mean) / sigma) else 0;
        }
    }
}
