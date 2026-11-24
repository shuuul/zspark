//! An audio file's bytes to mono f32 samples at its own rate (spec 138 WS-02): WAV (RIFF and RF64) through dr_wav,
//! FLAC through dr_flac, MP3 through dr_mp3 (its encoder delay and padding trimmed from the LAME tag, as
//! libsndfile's mpg123 does) and Ogg Vorbis through stb_vorbis, all vendored under `zig/vendor/` and compiled by
//! `build.zig` (`audioLibrary`). The format is sniffed from the bytes, whatever a name or a media type says. The
//! channels are averaged, sample by sample in channel order and then divided, as the reference's
//! `np.mean(audio, axis=-1)` on float32 does for a few channels. Symphonia decodes AAC-LC and ALAC
//! in MP4/M4A, ADTS AAC-LC and Opus in Ogg or WebM. This extends the reference's libsndfile path.

const std = @import("std");
const Allocator = std.mem.Allocator;

extern fn zs_audio_open(kind: c_int, data: [*]const u8, len: usize, channels: *c_uint, rate: *c_uint, frames: *c_ulonglong) ?*anyopaque;
extern fn zs_audio_read(handle: *anyopaque, out: [*]f32, count: c_ulonglong) c_ulonglong;
extern fn zs_audio_close(handle: ?*anyopaque) void;
extern fn zs_vorbis_open(data: [*]const u8, len: usize, channels: *c_uint, rate: *c_uint) ?*anyopaque;
extern fn zs_vorbis_read(handle: *anyopaque, out: [*]f32, count: c_ulonglong, channels: c_uint) c_ulonglong;
extern fn zs_vorbis_close(handle: ?*anyopaque) void;
extern fn zs_audio_decode_codecs(data: [*]const u8, len: usize, context: *Collector, samples: *const fn (*Collector, [*]const f32, usize, u32, usize) callconv(.c) bool) c_int;

pub const Error = Allocator.Error || error{
    /// Not a format decoded here.
    Unsupported,
    /// A format decoded here, but its decoder refused the bytes or found no sample in them.
    Invalid,
    /// Longer than the caller's `max_seconds`.
    TooLong,
};

/// The first three values match `dr_libs.c`'s decoder kinds.
pub const Format = enum(c_int) { wav = 1, flac = 2, mp3 = 3, vorbis, mp4, adts, opus, webm };

/// The format the bytes are decoded as, by their first bytes; null for any other.
pub fn sniff(bytes: []const u8) ?Format {
    if (bytes.len >= 12) {
        if ((std.mem.eql(u8, bytes[0..4], "RIFF") or std.mem.eql(u8, bytes[0..4], "RF64") or
            std.mem.eql(u8, bytes[0..4], "RIFX")) and std.mem.eql(u8, bytes[8..12], "WAVE")) return .wav;
        if (std.mem.eql(u8, bytes[0..4], "FORM") and
            (std.mem.eql(u8, bytes[8..12], "AIFF") or std.mem.eql(u8, bytes[8..12], "AIFC"))) return .wav;
        if (std.mem.eql(u8, bytes[4..8], "ftyp")) return .mp4;
    }
    if (std.mem.startsWith(u8, bytes, "riff\x2E\x91\xCF\x11\xA5\xD6\x28\xDB\x04\xC1\x00\x00")) return .wav;
    if (std.mem.startsWith(u8, bytes, "fLaC")) return .flac;
    if (std.mem.startsWith(u8, bytes, "\x1A\x45\xDF\xA3")) return .webm;
    // The first Ogg page carries the codec's identification packet.
    if (std.mem.startsWith(u8, bytes, "OggS") and bytes.len >= 27) {
        const at = 27 + @as(usize, bytes[26]);
        if (at > bytes.len) return null;
        if (std.mem.startsWith(u8, bytes[at..], "\x01vorbis")) return .vorbis;
        if (std.mem.startsWith(u8, bytes[at..], "OpusHead")) return .opus;
        return null;
    }
    const frames = skipId3(bytes);
    if (mpegFrame(frames)) return .mp3;
    if (frames.len >= 7 and frames[0] == 0xff and frames[1] & 0xf6 == 0xf0 and
        (frames[2] >> 2) & 0xf < 13) return .adts;
    return null;
}

/// The bytes after an ID3v2 tag where one leads them (its size is four 7-bit bytes, a footer 10 more).
fn skipId3(bytes: []const u8) []const u8 {
    if (bytes.len < 10 or !std.mem.startsWith(u8, bytes, "ID3")) return bytes;
    var size: usize = 0;
    for (bytes[6..10]) |b| {
        if (b & 0x80 != 0) return bytes;
        size = size << 7 | b;
    }
    size += 10;
    if (bytes[5] & 0x10 != 0) size += 10;
    return if (size <= bytes.len) bytes[size..] else bytes[bytes.len..];
}

/// Whether an MPEG audio frame header starts the bytes: the sync, a version and a layer that exist, a bitrate
/// and a rate that are not reserved. An AAC ADTS header shares the sync but has layer 0.
fn mpegFrame(bytes: []const u8) bool {
    if (bytes.len < 4 or bytes[0] != 0xFF or bytes[1] & 0xE0 != 0xE0) return false;
    const version = (bytes[1] >> 3) & 3;
    const layer = (bytes[1] >> 1) & 3;
    const bitrate = bytes[2] >> 4;
    const rate = (bytes[2] >> 2) & 3;
    return version != 1 and layer != 0 and bitrate != 15 and rate != 3;
}

/// Mono samples at the file's rate; the caller frees `samples`.
pub const Decoded = struct {
    samples: []f32,
    rate: u32,
    channels: u32,

    pub fn seconds(self: Decoded) f64 {
        return @as(f64, @floatFromInt(self.samples.len)) / @as(f64, @floatFromInt(self.rate));
    }
};

/// The most channels taken (stb_vorbis's own bound; WAV's 16-bit count is cut to it).
pub const max_channels = 16;
/// Frames read a call.
const chunk = 4096;

/// The file's samples, averaged to one channel, at most `max_seconds` of them at its rate.
pub fn decode(a: Allocator, bytes: []const u8, max_seconds: f64) Error!Decoded {
    const format = sniff(bytes) orelse return error.Unsupported;
    switch (format) {
        .mp4, .opus, .webm => return decodeCodecs(a, bytes, max_seconds),
        .adts => return decodeCodecs(a, skipId3(bytes), max_seconds),
        else => {},
    }
    var channels: c_uint = 0;
    var rate: c_uint = 0;
    var listed: c_ulonglong = 0;
    const handle = switch (format) {
        .vorbis => zs_vorbis_open(bytes.ptr, bytes.len, &channels, &rate),
        else => zs_audio_open(@backingInt(format), bytes.ptr, bytes.len, &channels, &rate, &listed),
    } orelse return error.Invalid;
    defer switch (format) {
        .vorbis => zs_vorbis_close(handle),
        else => zs_audio_close(handle),
    };
    if (channels == 0 or channels > max_channels or rate == 0) return error.Invalid;
    var collector: Collector = .{ .allocator = a, .max_seconds = max_seconds };
    defer collector.mono.deinit(a);
    if (listed > frameLimit(max_seconds, rate)) return error.TooLong;
    // The header's count is untrusted. Use it only as a bounded allocation hint.
    try collector.mono.ensureTotalCapacity(a, @intCast(@min(listed, 1 << 24)));
    const frame = try a.alloc(f32, chunk * channels);
    defer a.free(frame);
    while (true) {
        const got: usize = @intCast(switch (format) {
            .vorbis => zs_vorbis_read(handle, frame.ptr, chunk, channels),
            else => zs_audio_read(handle, frame.ptr, chunk),
        });
        if (got == 0) break;
        try collector.append(frame.ptr, got, rate, channels);
    }
    return collector.finish();
}

fn frameLimit(max_seconds: f64, rate: u32) u64 {
    return @intFromFloat(@max(0, @min(@floor(max_seconds * @as(f64, @floatFromInt(rate))), 1 << 40)));
}

const Collector = struct {
    allocator: Allocator,
    max_seconds: f64,
    mono: std.ArrayList(f32) = .empty,
    rate: u32 = 0,
    channels: u32 = 0,
    failure: ?Error = null,

    fn append(self: *Collector, interleaved: [*]const f32, frames: usize, rate: u32, channels: usize) Error!void {
        if (rate == 0 or channels == 0 or channels > max_channels) return error.Invalid;
        if (self.rate == 0) {
            self.rate = rate;
            self.channels = @intCast(channels);
        }
        if (rate != self.rate or channels != self.channels) return error.Invalid;
        if (frames > frameLimit(self.max_seconds, rate) - self.mono.items.len) return error.TooLong;
        const into = try self.mono.addManyAsSlice(self.allocator, frames);
        if (channels == 1) {
            @memcpy(into, interleaved[0..frames]);
            return;
        }
        const scale: f32 = @floatFromInt(channels);
        for (into, 0..) |*s, i| {
            var sum: f32 = 0;
            for (interleaved[i * channels ..][0..channels]) |v| sum += v;
            s.* = sum / scale;
        }
    }

    fn finish(self: *Collector) Error!Decoded {
        if (self.mono.items.len == 0) return error.Invalid;
        return .{ .samples = try self.mono.toOwnedSlice(self.allocator), .rate = self.rate, .channels = self.channels };
    }

    fn samples(self: *Collector, interleaved: [*]const f32, frames: usize, rate: u32, channels: usize) callconv(.c) bool {
        self.append(interleaved, frames, rate, channels) catch |err| {
            self.failure = err;
            return false;
        };
        return true;
    }
};

fn decodeCodecs(a: Allocator, bytes: []const u8, max_seconds: f64) Error!Decoded {
    var collector: Collector = .{ .allocator = a, .max_seconds = max_seconds };
    defer collector.mono.deinit(a);
    return switch (zs_audio_decode_codecs(bytes.ptr, bytes.len, &collector, Collector.samples)) {
        0 => collector.finish(),
        2 => error.Unsupported,
        3 => collector.failure orelse error.Invalid,
        else => error.Invalid,
    };
}
