//! An audio file's bytes to the 16 kHz mono samples and log-mel features a speech model reads, on the CPU
//! (spec 138 WS-02). The reference is the `qwen-asr` 0.0.6 package's `normalize_audio_input` and the Whisper
//! extractor its processor calls: the file decoded (`decode.zig`; the package reads it with librosa over
//! libsndfile), the channels averaged, resampled to 16 kHz when its rate differs (`resample.zig`, librosa's
//! default `soxr_hq`), divided by its peak where that passes 1 and clipped to [-1, 1] (`float_range_normalize`),
//! then the features (`mel.zig`). The package's 1200 s cap is the engine's (`models/qwen3_asr/served.zig`),
//! which passes it here as `max_seconds`. Its 0.5 s floor pads only the pieces of a clip it splits past 1200 s:
//! a short clip alone goes unpadded (`split_audio_into_chunks` returns it whole first), as here.
//!
//! Host only: this layer imports no other of zspark's.

const std = @import("std");
const Allocator = std.mem.Allocator;
pub const decode = @import("decode.zig");
pub const resample = @import("resample.zig");
pub const mel = @import("mel.zig");
pub const split = @import("split.zig");

pub const Error = decode.Error;
pub const rate = mel.rate;

/// vLLM's 400 for a file it cannot take (`speech_to_text/base/serving.py`, `multimodal/media/audio.py`): an
/// unknown format is "Invalid or unsupported audio file."; a known one its decoder refuses gets the decoders'
/// own words.
pub fn message(e: Error) []const u8 {
    return switch (e) {
        error.Unsupported => "Invalid or unsupported audio file.",
        error.Invalid => "Invalid or corrupted audio data. Ensure the input is a valid audio or video file (e.g. a complete WAV, MP3, or MP4).",
        error.TooLong => "Audio exceeds maximum allowed duration.",
        error.OutOfMemory => "Out of memory decoding the audio.",
    };
}

/// A clip as the model's processor takes it: 16 kHz mono samples in [-1, 1].
pub const Clip = struct {
    samples: []f32,
    /// The file's own rate and channels, and its length at that rate.
    source_rate: u32,
    channels: u32,
    seconds: f64,

    pub fn deinit(self: Clip, a: Allocator) void {
        a.free(self.samples);
    }
};

/// The file's samples as `qwen-asr`'s `normalize_audio_input` gives them; at most `max_seconds` of the file.
pub fn clip(a: Allocator, bytes: []const u8, max_seconds: f64) Error!Clip {
    const d = try decode.decode(a, bytes, max_seconds);
    var samples = d.samples;
    if (d.rate != rate) {
        defer a.free(d.samples);
        samples = try resample.resample(a, d.samples, d.rate, rate);
    }
    normalize(samples);
    return .{ .samples = samples, .source_rate = d.rate, .channels = d.channels, .seconds = d.seconds() };
}

/// `float_range_normalize`: divided by the peak where it passes 1, then clipped to [-1, 1].
pub fn normalize(x: []f32) void {
    var peak: f32 = 0;
    for (x) |v| peak = @max(peak, @abs(v));
    if (peak > 1.0) for (x) |*v| {
        v.* /= peak;
    };
    for (x) |*v| v.* = std.math.clamp(v.*, -1.0, 1.0);
}
