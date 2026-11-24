//! Additional ASR codecs against one generated stereo PCM reference (fixtures/audio/README.md).

const std = @import("std");
const audio = @import("../../src/audio/audio.zig");
const testing = std.testing;

const wav = @embedFile("../fixtures/audio/stereo_ref.wav");
const m4a = @embedFile("../fixtures/audio/stereo_aac.m4a");
const adts = @embedFile("../fixtures/audio/stereo_adts.aac");

test "lossless M4A, AIFF and Wave64 decode to the original stereo signal averaged to mono" {
    const a = testing.allocator;
    const reference = try audio.decode.decode(a, wav, 0.25);
    defer a.free(reference.samples);
    try testing.expectEqual(@as(usize, 12000), reference.samples.len);
    for ([_][]const u8{
        @embedFile("../fixtures/audio/stereo_alac.m4a"),
        @embedFile("../fixtures/audio/stereo.aiff"),
        @embedFile("../fixtures/audio/stereo.w64"),
    }) |bytes| {
        const decoded = try audio.decode.decode(a, bytes, 0.25);
        defer a.free(decoded.samples);
        try testing.expectEqual(@as(u32, 48000), decoded.rate);
        try testing.expectEqual(@as(u32, 2), decoded.channels);
        try testing.expectEqualSlices(f32, reference.samples, decoded.samples);
    }
}

test "AAC and Opus recordings reach the ASR clip with their signal, source rate and duration" {
    const id3 = "ID3\x04\x00\x00\x00\x00\x00\x00";
    for ([_]struct { name: []const u8, bytes: []const u8 }{
        .{ .name = "M4A AAC", .bytes = m4a },
        .{ .name = "ADTS AAC", .bytes = adts },
        .{ .name = "ID3 and ADTS AAC", .bytes = id3 ++ adts },
        .{ .name = "Ogg Opus", .bytes = @embedFile("../fixtures/audio/stereo_opus.ogg") },
        .{ .name = "WebM Opus", .bytes = @embedFile("../fixtures/audio/stereo_opus.webm") },
    }) |case| {
        errdefer std.debug.print("codec fixture: {s}\n", .{case.name});
        const clip = try audio.clip(testing.allocator, case.bytes, 1);
        defer clip.deinit(testing.allocator);
        try testing.expectEqual(@as(u32, 48000), clip.source_rate);
        try testing.expectEqual(@as(u32, 2), clip.channels);
        try testing.expectApproxEqAbs(@as(f64, 0.25), clip.seconds, 0.035);
        try testing.expectEqual(@as(usize, @intFromFloat(@round(clip.seconds * audio.rate))), clip.samples.len);

        // The middle 0.1 s spans 44 cycles. Measure the 440 Hz amplitude independently of codec delay.
        var sine: f64 = 0;
        var cosine: f64 = 0;
        const signal = clip.samples[800..2400];
        for (signal, 0..) |sample, i| {
            const phase = 2.0 * std.math.pi * 440 * @as(f64, @floatFromInt(i)) / audio.rate;
            sine += sample * @sin(phase);
            cosine += sample * @cos(phase);
        }
        const amplitude = 2 * @sqrt(sine * sine + cosine * cosine) / @as(f64, @floatFromInt(signal.len));
        try testing.expectApproxEqAbs(@as(f64, 0.2), amplitude, 0.025);
        try testing.expectError(error.TooLong, audio.clip(testing.allocator, case.bytes, 0.1));
    }
    const opus = try audio.decode.decode(testing.allocator, @embedFile("../fixtures/audio/stereo_opus.ogg"), 1);
    defer testing.allocator.free(opus.samples);
    try testing.expectEqual(@as(usize, 12000), opus.samples.len);
}

test "a truncated M4A container is invalid" {
    try testing.expectError(error.Invalid, audio.clip(testing.allocator, m4a[0..32], 1));
}
