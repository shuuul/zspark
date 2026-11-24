//! WebP through libwebp's decoder (`zig/vendor/libwebp/`), as Pillow 12.3 opened it for TensorFold 0.6.5: the
//! features first (the size, checked before a pixel is allocated; the alpha), then the picture as RGBA where
//! the bitstream has alpha and as RGB where it has none, and the orientation from the container's first EXIF
//! chunk, else its first XMP. Pillow decodes through libwebp's animation decoder, which for one still frame
//! gives the same pixels. An animation is its first frame (spec 019 WS-04, as vLLM's Pillow loads it): the
//! frame's chunks decoded as a still picture and put on the canvas, which libwebp's animation decoder starts
//! transparent; Pillow reads that canvas as RGBA where the file says it has alpha, else as RGB, its uncovered
//! pixels black.

const std = @import("std");
const Allocator = std.mem.Allocator;
const picture = @import("picture");

/// libwebp's `WebPBitstreamFeatures` (`src/webp/decode.h`).
const Features = extern struct {
    width: c_int,
    height: c_int,
    has_alpha: c_int,
    has_animation: c_int,
    format: c_int,
    pad: [5]u32,
};

/// `WEBP_DECODER_ABI_VERSION` of the vendored release (1.6.0); `WebPGetFeatures` is an inline that passes it.
const abi_version = 0x0210;
const status_ok = 0;

extern fn WebPGetFeaturesInternal(data: [*]const u8, len: usize, features: *Features, version: c_int) c_int;
extern fn WebPDecodeRGBAInto(data: [*]const u8, len: usize, out: [*]u8, out_size: usize, stride: c_int) ?[*]u8;
extern fn WebPDecodeRGBInto(data: [*]const u8, len: usize, out: [*]u8, out_size: usize, stride: c_int) ?[*]u8;

pub fn decode(a: Allocator, data: []const u8) picture.Error!picture.Samples {
    // Pillow's `_accept`: RIFF, WEBP, then one of the three first chunks
    if (data.len < 16 or !std.mem.startsWith(u8, data, "RIFF") or !std.mem.eql(u8, data[8..12], "WEBP")) return error.Invalid;
    const first = data[12..16];
    if (!std.mem.eql(u8, first, "VP8 ") and !std.mem.eql(u8, first, "VP8L") and !std.mem.eql(u8, first, "VP8X")) return error.Invalid;
    var f: Features = undefined;
    if (WebPGetFeaturesInternal(data.ptr, data.len, &f, abi_version) != status_ok) return error.Invalid;
    if (f.width <= 0 or f.height <= 0) return error.Invalid;
    try picture.fits(@intCast(f.width), @intCast(f.height));
    const meta = metadata(data);
    const o = try picture.tags.orientation(meta.exif, meta.xmp, .refuse);
    if (f.has_animation != 0) return firstFrame(a, data, f, o);
    const w: usize = @intCast(f.width);
    const h: usize = @intCast(f.height);
    const channels: usize = if (f.has_alpha != 0) 4 else 3;
    const px = try a.alloc(u8, w * h * channels);
    errdefer a.free(px);
    const stride: c_int = @intCast(w * channels);
    const done = if (channels == 4) WebPDecodeRGBAInto(data.ptr, data.len, px.ptr, px.len, stride) else WebPDecodeRGBInto(data.ptr, data.len, px.ptr, px.len, stride);
    if (done == null) return error.Invalid;
    return .{ .width = w, .height = h, .channels = channels, .data = px, .orientation = o };
}

/// An animation's first frame on its canvas (`decode`'s doc comment).
fn firstFrame(a: Allocator, data: []const u8, f: Features, orientation: u32) picture.Error!picture.Samples {
    const frame = chunk(data, "ANMF") orelse return error.Invalid;
    if (frame.len < 16) return error.Invalid;
    const x = 2 * @as(usize, std.mem.readInt(u24, frame[0..3], .little));
    const y = 2 * @as(usize, std.mem.readInt(u24, frame[3..6], .little));
    const fw = @as(usize, std.mem.readInt(u24, frame[6..9], .little)) + 1;
    const fh = @as(usize, std.mem.readInt(u24, frame[9..12], .little)) + 1;
    const cw: usize = @intCast(f.width);
    const ch: usize = @intCast(f.height);
    if (x + fw > cw or y + fh > ch) return error.Invalid;
    // the frame's chunks (ALPH and VP8, or VP8L) as a still file of an extended header
    const body = frame[16..];
    const still = try a.alloc(u8, 12 + 18 + body.len);
    defer a.free(still);
    @memcpy(still[0..4], "RIFF");
    std.mem.writeInt(u32, still[4..8], @intCast(still.len - 8), .little);
    @memcpy(still[8..16], "WEBPVP8X");
    std.mem.writeInt(u32, still[16..20], 10, .little);
    @memset(still[20..24], 0);
    still[20] = 0x10; // alpha: the frame's own bitstream says whether it has any
    std.mem.writeInt(u24, still[24..27], @intCast(fw - 1), .little);
    std.mem.writeInt(u24, still[27..30], @intCast(fh - 1), .little);
    @memcpy(still[30..], body);
    const px = try a.alloc(u8, fw * fh * 4);
    defer a.free(px);
    if (WebPDecodeRGBAInto(still.ptr, still.len, px.ptr, px.len, @intCast(fw * 4)) == null) return error.Invalid;
    const channels: usize = if (f.has_alpha != 0) 4 else 3;
    const canvas = try a.alloc(u8, cw * ch * channels);
    errdefer a.free(canvas);
    @memset(canvas, 0);
    for (0..fh) |row| for (0..fw) |col| {
        const from = px[(row * fw + col) * 4 ..][0..4];
        const to = canvas[((y + row) * cw + x + col) * channels ..][0..channels];
        @memcpy(to, from[0..channels]);
    };
    return .{ .width = cw, .height = ch, .channels = channels, .data = canvas, .orientation = orientation };
}

/// The body of the first chunk of `kind` after the header, if it fits in the data.
fn chunk(data: []const u8, kind: *const [4]u8) ?[]const u8 {
    var at: usize = 12;
    while (data.len - at >= 8) {
        const size: usize = std.mem.readInt(u32, data[at + 4 ..][0..4], .little);
        if (size > data.len - at - 8) return null;
        if (std.mem.eql(u8, data[at..][0..4], kind)) return data[at + 8 ..][0..size];
        at += 8 + size + (size & 1);
        if (at > data.len) return null;
    }
    return null;
}

const Metadata = struct { exif: ?[]const u8 = null, xmp: ?[]const u8 = null };

/// The first EXIF and the first XMP chunk of an extended (VP8X) file, as libwebp's demuxer hands them to Pillow;
/// a simple file has neither. The walk stops at a chunk that runs past the data.
fn metadata(data: []const u8) Metadata {
    var m: Metadata = .{};
    if (!std.mem.eql(u8, data[12..16], "VP8X")) return m;
    var at: usize = 12;
    while (data.len - at >= 8) {
        const size: usize = std.mem.readInt(u32, data[at + 4 ..][0..4], .little);
        if (size > data.len - at - 8) break;
        const body = data[at + 8 ..][0..size];
        const kind = data[at..][0..4];
        if (m.exif == null and std.mem.eql(u8, kind, "EXIF")) m.exif = body;
        if (m.xmp == null and std.mem.eql(u8, kind, "XMP ")) m.xmp = body;
        at += 8 + size + (size & 1);
        if (at > data.len) break;
    }
    return m;
}
