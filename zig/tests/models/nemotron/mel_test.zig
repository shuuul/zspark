//! `mel.zig`'s tests (spec 151 WS-03; `tools/check.sh nemotron mel`): the log-mel against an fp64 host
//! reference of NeMo's FilterbankFeatures written from the source (`features.py`, commit 774bc30) on a
//! synthetic waveform, with `fb` and `window` built as NeMo builds them (librosa's slaney mel 0..8,000 Hz, a
//! symmetric hann of 400), and `rows` the `AudioTokenEstimator`'s counts on the edge lengths. Host only.

const std = @import("std");
const testing = std.testing;
const mel = @import("../../../src/models/nemotron/mel.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");

const p: mel.Params = .{ .n_fft = 512, .window = 400, .hop = 160, .features = 128, .rate = 16000 };

/// librosa's slaney mel filterbank (fmin 0, fmax 8000, slaney area norm) and torch's symmetric hann window,
/// in f32 as the checkpoint stores them in bf16.
fn tables(a: std.mem.Allocator) !struct { fb: []f32, window: []f32 } {
    const freqs = p.n_fft / 2 + 1;
    const fb = try a.alloc(f32, p.features * freqs);
    const window = try a.alloc(f32, p.window);
    for (window, 0..) |*w, i| w.* = @floatCast(0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, p.window - 1)));
    const hzToMel = struct {
        fn f(hz: f64) f64 {
            const logstep = 27.0 / @log(6.4);
            return if (hz >= 1000) 15.0 + @log(hz / 1000.0) * logstep else 3.0 * hz / 200.0;
        }
    }.f;
    const melToHz = struct {
        fn f(m: f64) f64 {
            const logstep = @log(6.4) / 27.0;
            return if (m >= 15.0) 1000.0 * @exp(logstep * (m - 15.0)) else 200.0 * m / 3.0;
        }
    }.f;
    var edges: [p.features + 2]f64 = undefined;
    const lo = hzToMel(0);
    const hi = hzToMel(8000);
    const step = (hi - lo) / @as(f64, p.features + 1);
    for (&edges, 0..) |*e, i| e.* = melToHz(if (i == p.features + 1) hi else lo + @as(f64, @floatFromInt(i)) * step);
    for (0..p.features) |m| {
        const enorm = 2.0 / (edges[m + 2] - edges[m]);
        for (0..freqs) |k| {
            const f = 8000.0 * @as(f64, @floatFromInt(k)) / @as(f64, freqs - 1);
            const down = -(edges[m] - f) / (edges[m + 1] - edges[m]);
            const up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            fb[m * freqs + k] = @floatCast(@max(0.0, @min(down, up)) * enorm);
        }
    }
    return .{ .fb = fb, .window = window };
}

/// Two sines plus a deterministic noise, [-0.5, 0.5].
fn waveform(a: std.mem.Allocator, n: usize, seed: u64) ![]f32 {
    const x = try a.alloc(f32, n);
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    for (x, 0..) |*v, i| {
        const t = @as(f64, @floatFromInt(i)) / 16000;
        v.* = @floatCast(0.2 * @sin(2 * std.math.pi * 440 * t) + 0.1 * @sin(2 * std.math.pi * 1300 * t) + 0.05 * (random.float(f64) - 0.5));
    }
    return x;
}

/// NeMo's math in fp64: preemph 0.97, centered STFT (pad 256, the window centered in the frame), power
/// spectrum, the fb matmul, log(x + 2^-24) — straight from `FilterbankFeatures.forward`.
fn reference(p_: mel.Params, fb: []const f32, window: []const f32, samples: []const f32, out: []f64) void {
    const t_len = p_.frames(samples.len);
    const freqs = p_.n_fft / 2 + 1;
    const off = (p_.n_fft - p_.window) / 2;
    var pre: [1 << 20]f64 = undefined;
    std.debug.assert(samples.len <= pre.len);
    if (samples.len > 0) {
        pre[0] = samples[0];
        for (1..samples.len) |i| pre[i] = samples[i] - 0.97 * @as(f64, samples[i - 1]);
    }
    var re: [1024]f64 = undefined;
    var im: [1024]f64 = undefined;
    for (0..t_len) |f| {
        for (0..p_.n_fft) |i| {
            const at = @as(i64, @intCast(f * p_.hop + i)) - @as(i64, @intCast(p_.pad()));
            re[i] = if (i >= off and i < off + p_.window and at >= 0 and at < samples.len) pre[@intCast(at)] * window[i - off] else 0;
            im[i] = 0;
        }
        // direct DFT (reference speed is fine at this size)
        var power: [1024]f64 = undefined;
        for (0..freqs) |k| {
            var sr: f64 = 0;
            var si: f64 = 0;
            for (0..p_.n_fft) |i| {
                const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(k * i)) / @as(f64, @floatFromInt(p_.n_fft));
                sr += re[i] * @cos(angle) - im[i] * @sin(angle);
                si += re[i] * @sin(angle) + im[i] * @cos(angle);
            }
            power[k] = sr * sr + si * si;
        }
        for (0..p_.features) |m| {
            var sum: f64 = 0;
            for (0..freqs) |k| sum += @as(f64, fb[m * freqs + k]) * power[k];
            out[f * p_.features + m] = @log(sum + std.math.pow(f64, 2, -24));
        }
    }
}

test "mel: a synthetic clip's features against NeMo's math in fp64" {
    const a = testing.allocator;
    const t = try tables(a);
    defer a.free(t.fb);
    defer a.free(t.window);
    const samples = try waveform(a, 5000, 7);
    defer a.free(samples);
    const t_len = p.frames(samples.len);
    const mine = try a.alloc(f32, t_len * p.features);
    defer a.free(mine);
    var scratch: mel.Scratch = try .init(a, p, samples.len);
    defer scratch.deinit(a);
    mel.features(p, &scratch, t.fb, t.window, samples, mine);
    const want = try a.alloc(f64, t_len * p.features);
    defer a.free(want);
    reference(p, t.fb, t.window, samples, want);
    // f32 (fft, the roundings) against fp64: each value within 1e-5 absolute and 1e-4 of the log's scale
    var worst: f64 = 0;
    for (mine, want) |got, exp| worst = @max(worst, @abs(@as(f64, got) - exp));
    try testing.expect(worst < 1e-4);
}

test "rows: the estimator's counts on the edge lengths" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // 0.5 s: 8,000 samples -> 50 frames -> ceil(50/8) = 7
    try testing.expectEqual(@as(usize, 7), try mel.rows(a, p, 30.0, 8, 8000));
    // exactly 30 s: one chunk of 3,000 frames -> 375
    try testing.expectEqual(@as(usize, 375), try mel.rows(a, p, 30.0, 8, 480000));
    // 30 s + 1 sample: the one-sample tail folds into the previous chunk -> still 375
    try testing.expectEqual(@as(usize, 375), try mel.rows(a, p, 30.0, 8, 480001));
    // a tail that lives: 480,000 + 320 -> second chunk of 2 frames -> 375 + 1
    try testing.expectEqual(@as(usize, 376), try mel.rows(a, p, 30.0, 8, 480320));
    // 3600 s: 120 full chunks -> 45,000
    try testing.expectEqual(@as(usize, 45000), try mel.rows(a, p, 30.0, 8, 57600000));
    // a lone span under the minimum still counts once (no previous chunk to fold into)
    try testing.expectEqual(@as(usize, 1), try mel.rows(a, p, 30.0, 8, 160));
    try testing.expectEqual(@as(usize, 1), try mel.rows(a, p, 30.0, 8, 320));
    // 61 s: 30 + 30 + 1, the last of 16,000 samples (>= the 320 minimum) -> 375 + 375 + ceil(100/8)
    try testing.expectEqual(@as(usize, 375 + 375 + 13), try mel.rows(a, p, 30.0, 8, 976000));
    // 60 s + 160 samples: the tail folds, the grown chunk is 480,160 -> 375 + ceil(3,001/8)
    try testing.expectEqual(@as(usize, 375 + 376), try mel.rows(a, p, 30.0, 8, 960160));
    // a smaller chunk size: 10 s spans -> 125 each
    try testing.expectEqual(@as(usize, 375), try mel.rows(a, p, 10.0, 8, 480000));
}

test "mel: the reference fixtures' clips (ZSPARK_REFERENCE_NEMOTRON)" {
    // WS-04's exports: each NAME.mel.safetensors holds `pcm.16k` F32 [n] and `mel.f32` [128, T]
    // (AudioToMelSpectrogramPreprocessor on the checkpoint's preprocessor config), `manifest.json` the
    // plugin's single-pass `audio_placeholders`. `mel.f32`'s last column is the STFT's partial frame
    // `get_seq_len` counts out — compare `frames(n)` valid frames only.
    const a = testing.allocator;
    const io = testing.io;
    const dir_path = testing.environ.getPosix("ZSPARK_REFERENCE_NEMOTRON") orelse return error.SkipZigTest;
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    const t = try tables(a);
    defer a.free(t.fb);
    defer a.free(t.window);
    const raw = try dir.readFileAlloc(io, "manifest.json", a, .limited(1 << 22));
    defer a.free(raw);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const manifest = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), raw, .{});
    const clips = manifest.object.get("clips").?.object;
    var entries = dir.iterate();
    var seen: usize = 0;
    var worst: f64 = 0;
    while (try entries.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".mel.safetensors")) continue;
        const name = entry.name[0..std.mem.indexOfScalar(u8, entry.name, '.').?];
        var file = try safetensors.File.open(a, io, dir, entry.name);
        defer file.close();
        const pcm_t = file.get("pcm.16k").?;
        const mel_t = file.get("mel.f32").?;
        const pcm = std.mem.bytesAsSlice(f32, pcm_t.data);
        const mel_want = std.mem.bytesAsSlice(f32, mel_t.data); // [features][T] bin-major
        const T: usize = @intCast(mel_t.shape[1]);
        const valid = p.frames(pcm.len);
        try testing.expectEqual(@as(usize, @intCast(mel_t.shape[0])), p.features);
        try testing.expectEqual(valid + 1, T);
        // rows: the plugin's single-pass placeholder count
        const clip = clips.get(name).?.object;
        const want_rows: usize = @intCast(clip.get("audio_placeholders").?.integer);
        try testing.expectEqual(want_rows, @max(1, (valid + 7) / 8));
        // the log-mel over the whole clip (features() emits [frames, features] row-major)
        const got = try a.alloc(f32, valid * p.features);
        defer a.free(got);
        var scratch: mel.Scratch = try .init(a, p, pcm.len);
        defer scratch.deinit(a);
        mel.features(p, &scratch, t.fb, t.window, @as([]const f32, @alignCast(pcm)), got);
        var clip_worst: f64 = 0;
        for (0..valid) |f| for (0..p.features) |m| {
            clip_worst = @max(clip_worst, @abs(@as(f64, got[f * p.features + m]) - mel_want[m * T + f]));
        };
        std.debug.print("{s}: n={d} T={d} rows={d} worst={d:.6}\n", .{ name, pcm.len, T, want_rows, clip_worst });
        worst = @max(worst, clip_worst);
        seen += 1;
    }
    try testing.expect(seen > 0);
    try testing.expect(worst < 1e-3);
}
