//! `zig/src/audio/` against the `qwen-asr` 0.0.6 package's steps (`fixtures/audio/cases.json`, written by
//! `tools/reference/audio_cases.py`): libsndfile's decode through librosa, `soxr_hq`, `float_range_normalize`
//! and transformers 4.57.6's Whisper extractor. Each file decodes to libsndfile's samples, gives the package's
//! 16 kHz clip and its features within the bounds below, or is refused as the layer refuses it; the features
//! of a clip both sides compute alike are within the spec's 1e-4.

const std = @import("std");
const audio = @import("../../src/audio/audio.zig");
const mel = audio.mel;
const safetensors = @import("../../src/weights/safetensors.zig");

const Case = struct {
    name: []const u8,
    bytes: []const u8,
    format: ?[]const u8 = null,
    rate: ?u32 = null,
    samples: ?usize = null,
    decoded: ?[]const u8 = null,
    clip_samples: ?usize = null,
    clip: ?[]const u8 = null,
    frames: ?usize = null,
    mel: ?[]const u8 = null,
    problem: ?[]const u8 = null,
};

const Cases = struct {
    versions: std.json.Value,
    cases: []Case,
    synthetic: struct { samples: usize, frames: usize, every: usize, mel: []const u8 },
    tokens: [][2]usize,
};

/// What the fixture keeps of the samples (`audio_cases.py`'s `DECODED_EVERY`, `CLIP_EVERY`).
const decoded_every = 4;
const clip_every = 2;

/// The bounds, measured on the cases (in brackets) and kept a little above; spec 138 WS-02's entry has the
/// figures. A format's decode against libsndfile's: WAV and FLAC are the same integers scaled alike (0), MP3 is
/// minimp3's against mpg123's (1.5e-5, the same length: both trim the LAME tag's delay and padding), Vorbis
/// stb_vorbis's against libvorbis's (1.5e-7).
const bound = struct {
    fn decoded(format: []const u8) f32 {
        if (std.mem.eql(u8, format, "mp3")) return 5e-5;
        if (std.mem.eql(u8, format, "vorbis")) return 1e-6;
        return 0;
    }
    /// The 16 kHz clip against soxr_hq's, where the rate is not 16 kHz, past its first `edge_samples` (1.1e-4):
    /// soxr's first two outputs of a ratio that is not an integer differ from the filter's (its own start; 1e-2).
    const resampled = 3e-4;
    const edge_samples = 2;
    /// The features within the pass band: 1e-4 where nothing was resampled (8.3e-7), `mel_resampled` where
    /// something was (4.3e-4); above it (soxr's transition band, or what an upsampled file has none of)
    /// `mel_above` (5.3e-3); the first `edge_frames` frames of a resampled clip, which its first samples reach,
    /// `mel_edge` (0.44, the quietest bins of a 22.05 kHz file's first frame).
    const mel_exact = 1e-4;
    const mel_resampled = 1e-3;
    const mel_above = 1e-2;
    const edge_frames = 2;
    const mel_edge = 1.0;
};

/// The first filter whose band reaches past `resample.pass_end` of `rate`'s Nyquist.
fn passBand(t: *const mel.Tables, rate: u32) usize {
    const limit = audio.resample.pass_end * @as(f64, @floatFromInt(rate)) / 2;
    for (0..mel.bins) |m| {
        const last = t.first[m] + t.count[m];
        if (@as(f64, @floatFromInt(last * mel.rate / mel.n_fft)) > limit) return m;
    }
    return mel.bins;
}

fn floats(gpa: std.mem.Allocator, b64: []const u8) ![]f32 {
    const d = std.base64.standard.Decoder;
    const bytes = try gpa.alloc(u8, try d.calcSizeForSlice(b64));
    defer gpa.free(bytes);
    try d.decode(bytes, b64);
    const out = try gpa.alloc(f32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return out;
}

fn raw(gpa: std.mem.Allocator, b64: []const u8) ![]u8 {
    const d = std.base64.standard.Decoder;
    const out = try gpa.alloc(u8, try d.calcSizeForSlice(b64));
    try d.decode(out, b64);
    return out;
}

/// The largest distance between `got[i * every]` and `want[i]`.
fn distance(got: []const f32, want: []align(1) const f32, every: usize) f32 {
    var worst: f32 = 0;
    for (want, 0..) |w, i| worst = @max(worst, @abs(got[i * every] - w));
    return worst;
}

/// `audio_cases.py`'s `signal`: the same f64 arithmetic in the same order, cast to f32.
fn signal(gpa: std.mem.Allocator, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    var state: u64 = 0x2545F4914F6CDD1D;
    const pi = std.math.pi;
    for (out, 0..) |*v, i| {
        const t = @as(f64, @floatFromInt(i)) / 16000.0;
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const noise = (@as(f64, @floatFromInt(state >> 40)) / @as(f64, 1 << 24) - 0.5) * 0.02;
        const x = 0.3 * @sin(2 * pi * (220.0 * t + 600.0 * t * t)) +
            0.2 * @sin(2 * pi * 3150.0 * t) * (0.5 + 0.5 * @sin(2 * pi * 3.0 * t)) + noise;
        v.* = @floatCast(x);
    }
    return out;
}

fn tables(gpa: std.mem.Allocator) !*mel.Tables {
    const t = try gpa.create(mel.Tables);
    t.init();
    return t;
}

test "the features of a clip both sides compute alike are the Whisper extractor's within 1e-4" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Cases, gpa, @embedFile("fixtures/audio/cases.json"), .{});
    defer parsed.deinit();
    const s = parsed.value.synthetic;
    const x = try signal(gpa, s.samples);
    defer gpa.free(x);
    const t = try tables(gpa);
    defer gpa.destroy(t);
    const got = try mel.features(gpa, t, x);
    defer gpa.free(got);
    try std.testing.expectEqual(s.frames, mel.frames(x.len));
    const want = try floats(gpa, s.mel);
    defer gpa.free(want);
    const kept = (s.frames + s.every - 1) / s.every;
    try std.testing.expectEqual(mel.bins * kept, want.len);
    var worst: f32 = 0;
    for (0..mel.bins) |m| for (0..kept) |j| {
        worst = @max(worst, @abs(got[m * s.frames + j * s.every] - want[m * kept + j]));
    };
    std.debug.print("synthetic: {d} frames, features within {e}\n", .{ s.frames, worst });
    try std.testing.expect(worst <= bound.mel_exact);
}

test "every file decodes to libsndfile's samples and gives the package's clip and features, or is refused" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Cases, gpa, @embedFile("fixtures/audio/cases.json"), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("4.57.6", parsed.value.versions.object.get("transformers").?.string);
    const t = try tables(gpa);
    defer gpa.destroy(t);
    var wrong: usize = 0;
    for (parsed.value.cases) |c| {
        const bytes = try raw(gpa, c.bytes);
        defer gpa.free(bytes);
        const d = audio.decode.decode(gpa, bytes, 1200) catch |err| {
            if (c.problem == null or !std.mem.eql(u8, c.problem.?, @errorName(err))) {
                std.debug.print("{s}: {t} where the reference gave {s}\n", .{ c.name, err, c.problem orelse "samples" });
                wrong += 1;
            }
            continue;
        };
        defer gpa.free(d.samples);
        if (c.problem) |p| {
            std.debug.print("{s}: decoded where it is refused ({s})\n", .{ c.name, p });
            wrong += 1;
            continue;
        }
        const format = c.format.?;
        const want_decoded = try floats(gpa, c.decoded.?);
        defer gpa.free(want_decoded);
        if (d.rate != c.rate.? or d.samples.len != c.samples.?) {
            std.debug.print("{s}: {d} samples at {d} Hz where libsndfile gave {d} at {d}\n", .{ c.name, d.samples.len, d.rate, c.samples.?, c.rate.? });
            wrong += 1;
            continue;
        }
        const decoded_off = distance(d.samples, want_decoded, decoded_every);

        const clip = try audio.clip(gpa, bytes, 1200);
        defer clip.deinit(gpa);
        const want_clip = try floats(gpa, c.clip.?);
        defer gpa.free(want_clip);
        if (clip.samples.len != c.clip_samples.?) {
            std.debug.print("{s}: a clip of {d} samples where the package's has {d}\n", .{ c.name, clip.samples.len, c.clip_samples.? });
            wrong += 1;
            continue;
        }
        const clip_off = distance(clip.samples, want_clip, clip_every);

        const got = try mel.features(gpa, t, clip.samples);
        defer gpa.free(got);
        const want_mel = try floats(gpa, c.mel.?);
        defer gpa.free(want_mel);
        const frames = c.frames.?;
        try std.testing.expectEqual(frames, mel.frames(clip.samples.len));
        // the bins whose filters end below the pass band's end (the source's rate's or 16 kHz's, the lower), and
        // the frames past the first two, which a non-integer ratio's first samples reach (`bound.edge`)
        const band = passBand(t, @min(d.rate, audio.rate));
        const resampled = d.rate != audio.rate;
        const skip: usize = if (resampled) bound.edge_frames else 0;
        var inside: f32 = 0;
        var above: f32 = 0;
        var edge: f32 = 0;
        for (0..mel.bins) |m| for (0..frames) |f| {
            const e = @abs(got[m * frames + f] - want_mel[m * frames + f]);
            if (f < skip) edge = @max(edge, e) else if (m < band) inside = @max(inside, e) else above = @max(above, e);
        };
        const clip_inner = distance(clip.samples[if (resampled) bound.edge_samples else 0..], want_clip[if (resampled) bound.edge_samples / clip_every else 0..], clip_every);
        std.debug.print("{s}: decoded within {e}; clip within {e}, past its first {d} samples {e}; features below bin {d} within {e}, above {e}, first frames {e}\n", .{ c.name, decoded_off, clip_off, bound.edge_samples, clip_inner, band, inside, above, edge });
        const mel_inside: f32 = if (resampled or decoded_off > 0) bound.mel_resampled else bound.mel_exact;
        const clip_bound = if (resampled) bound.resampled else bound.decoded(format);
        if (decoded_off > bound.decoded(format) or clip_inner > clip_bound or inside > mel_inside or
            above > @max(mel_inside, bound.mel_above) or edge > bound.mel_edge)
        {
            std.debug.print("{s}: past its bounds\n", .{c.name});
            wrong += 1;
        }
    }
    try std.testing.expectEqual(0, wrong);
}

test "the tower's token count is the package's for every frame count" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Cases, gpa, @embedFile("fixtures/audio/cases.json"), .{});
    defer parsed.deinit();
    for (parsed.value.tokens) |pair| try std.testing.expectEqual(pair[1], mel.tokens(pair[0]));
}

test "a sine resampled to 16 kHz is the sine at 16 kHz" {
    const gpa = std.testing.allocator;
    for ([_]u32{ 8000, 11025, 22050, 44100, 48000 }) |from| {
        const n = from; // one second
        const x = try gpa.alloc(f32, n);
        defer gpa.free(x);
        const hz = 1000.0;
        for (x, 0..) |*v, i| v.* = @floatCast(0.5 * @sin(2 * std.math.pi * hz * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(from))));
        const y = try audio.resample.resample(gpa, x, from, audio.rate);
        defer gpa.free(y);
        try std.testing.expectEqual(@as(usize, audio.rate), y.len);
        // away from the ends, where the filter meets the zeros past them
        var worst: f64 = 0;
        for (y[800 .. y.len - 800], 800..) |v, k| {
            const want = 0.5 * @sin(2 * std.math.pi * hz * @as(f64, @floatFromInt(k)) / audio.rate);
            worst = @max(worst, @abs(@as(f64, v) - want));
        }
        std.debug.print("1 kHz from {d} Hz: within {e} of the sine\n", .{ from, worst });
        try std.testing.expect(worst < 2e-5);
    }
}

test "a clip's peak above 1 is divided down and clipped as the package's float_range_normalize" {
    var x = [_]f32{ 0.5, -2.0, 1.0, 0.25 };
    audio.normalize(&x);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, -1.0, 0.5, 0.125 }, &x);
    var y = [_]f32{ 0.5, -0.75 };
    audio.normalize(&y);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, -0.75 }, &y);
}

test "a file past the length asked for is refused before it is all decoded, in vLLM's words otherwise" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Cases, gpa, @embedFile("fixtures/audio/cases.json"), .{});
    defer parsed.deinit();
    const bytes = try raw(gpa, parsed.value.cases[0].bytes);
    defer gpa.free(bytes);
    try std.testing.expectError(error.TooLong, audio.decode.decode(gpa, bytes, 0.1));
    try std.testing.expectEqualStrings("Invalid or unsupported audio file.", audio.message(error.Unsupported));
    try std.testing.expect(std.mem.startsWith(u8, audio.message(error.Invalid), "Invalid or corrupted audio data."));
}

fn tensorFloats(file: *const safetensors.File, name: []const u8) ![]align(1) const f32 {
    const t = file.get(name) orelse return error.MissingTensor;
    if (t.dtype != .F32) return error.WrongDType;
    return std.mem.bytesAsSlice(f32, t.data);
}

/// The worst distance over the features of `[bins][frames]`: past the first `skip` frames, within the band below
/// `band`, and above it.
fn melDistance(got: []const f32, want: []align(1) const f32, frames: usize, skip: usize, band: usize) [2]f32 {
    var worst: [2]f32 = .{ 0, 0 };
    for (0..mel.bins) |m| for (skip..frames) |f| {
        const d = @abs(got[m * frames + f] - want[m * frames + f]);
        const at: usize = @intFromBool(m >= band);
        worst[at] = @max(worst[at], d);
    };
    return worst;
}

test "the reference's clips (ZSPARK_REFERENCE_QWEN3_ASR) give its samples and features" {
    // spec 138 WS-01's exports: each `NAME.*.safetensors` holds the package's `pcm.src` (the file as librosa
    // reads it, `[channels][n]` or `[n]`), its 16 kHz clip `pcm.16k` and the torch path's features `mel.f32`;
    // `clips/NAME.flac` is the file
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = std.testing.environ.getPosix("ZSPARK_REFERENCE_QWEN3_ASR") orelse return error.SkipZigTest;
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    const t = try tables(gpa);
    defer gpa.destroy(t);
    var seen: usize = 0;
    var entries = dir.iterate();
    while (try entries.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        const name = entry.name[0..std.mem.indexOfScalar(u8, entry.name, '.').?];
        var file = try safetensors.File.open(gpa, io, dir, entry.name);
        defer file.close();
        const want_mel = try tensorFloats(&file, "mel.f32");
        const pcm16 = try tensorFloats(&file, "pcm.16k");
        const src = try tensorFloats(&file, "pcm.src");
        const frames = mel.frames(pcm16.len);
        try std.testing.expectEqual(mel.bins * frames, want_mel.len);

        // the features of the package's own clip: the spec's 1e-4, everywhere
        const x = try gpa.alloc(f32, pcm16.len);
        defer gpa.free(x);
        for (x, pcm16) |*v, w| v.* = w;
        const got_mel = try mel.features(gpa, t, x);
        defer gpa.free(got_mel);
        const exact = melDistance(got_mel, want_mel, frames, 0, mel.bins)[0];

        // the file through the layer: its decode against librosa's, then its clip and features
        const flac = try std.fmt.allocPrint(gpa, "clips/{s}.flac", .{name});
        defer gpa.free(flac);
        const bytes = try dir.readFileAlloc(io, flac, gpa, .limited(1 << 28));
        defer gpa.free(bytes);
        const d = try audio.decode.decode(gpa, bytes, 1200);
        defer gpa.free(d.samples);
        const shape = file.get("pcm.src").?.shape;
        const n: usize = @intCast(shape[shape.len - 1]);
        const c: usize = if (shape.len == 2) @intCast(shape[0]) else 1;
        try std.testing.expectEqual(n, d.samples.len);
        var decoded: f32 = 0;
        for (d.samples, 0..) |s, i| {
            var sum: f32 = 0;
            for (0..c) |ch| sum += src[ch * n + i];
            decoded = @max(decoded, @abs(s - sum / @as(f32, @floatFromInt(c))));
        }
        const clip = try audio.clip(gpa, bytes, 1200);
        defer clip.deinit(gpa);
        try std.testing.expectEqual(pcm16.len, clip.samples.len);
        const resampled = clip.source_rate != mel.rate;
        const edge: usize = if (resampled) bound.edge_samples else 0;
        const samples = distance(clip.samples[edge..], pcm16[edge..], 1);
        const clip_mel = try mel.features(gpa, t, clip.samples);
        defer gpa.free(clip_mel);
        const band = passBand(t, @min(clip.source_rate, mel.rate));
        const skip: usize = if (resampled) bound.edge_frames else 0;
        const inner = melDistance(clip_mel, want_mel, frames, skip, band);
        var first: f32 = 0;
        for (0..mel.bins) |m| for (0..@min(frames, skip)) |f| {
            first = @max(first, @abs(clip_mel[m * frames + f] - want_mel[m * frames + f]));
        };
        std.debug.print("{s}: {d} Hz x{d}, {d} frames; features of its clip within {e}; decode {e}, clip {e}, " ++
            "features {e} in band, {e} above, {e} first frames\n", .{ entry.name, clip.source_rate, c, frames, exact, decoded, samples, inner[0], inner[1], first });
        try std.testing.expect(exact <= bound.mel_exact);
        try std.testing.expect(decoded == 0);
        try std.testing.expect(samples <= @as(f32, if (resampled) bound.resampled else 0));
        try std.testing.expect(inner[0] <= @as(f32, if (resampled) bound.mel_resampled else bound.mel_exact));
        try std.testing.expect(inner[1] <= bound.mel_above);
        try std.testing.expect(first <= bound.mel_edge);
        seen += 1;
    }
    try std.testing.expect(seen > 0);
}
