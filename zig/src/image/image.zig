//! A picture's bytes to RGB on the CPU (spec 019 WS-07): PNG, JPEG, GIF's first frame and WebP, as TensorFold
//! 0.6.5's `vision/images.py` `_decode` loaded them through Pillow 12.3: the size checked against `ImageLimits`
//! before a pixel is allocated, `ImageOps.exif_transpose`, then any alpha over white (vLLM's `rgba_to_rgb`
//! default too). PNG is TensorFold 1.0.12's own decoder (the vendored `picture`), Pillow's bits; JPEG is
//! stb_image's and WebP libwebp's, both vendored under `zig/vendor/` and compiled by `build.zig` (`image_c`).
//! A `data:` URL's bytes and an `https` fetch are the vendored host's (`server/images.zig`, `media_fetch.zig`).
//! BMP, TGA, PSD and binary PNM are stb_image's too (`stb.zig`, WS-04), formats vLLM's Pillow opens and 0.6.5
//! refused: vLLM's `open_image` is the reference for those. An animated PNG or WebP is its first frame, as vLLM's
//! Pillow loads it (WS-04), where 0.6.5 refused it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const picture = @import("picture");
pub const gif = @import("gif.zig");
pub const jpeg = @import("jpeg.zig");
pub const webp = @import("webp.zig");
pub const stb = @import("stb.zig");

pub const Rgb = picture.Rgb;
pub const Error = picture.Error || error{TooManyBytes};

/// 0.6.5's `ImageLimits` for one picture: its encoded bytes, then (`picture.fits`) a side and the pixels.
pub const max_bytes = 10 << 20;
pub const max_side = picture.max_side;
pub const max_pixels = picture.max_pixels;

pub const Format = enum { png, jpeg, gif, webp, bmp, psd, pnm, tga, unknown };

/// The format Pillow would open the bytes as, by their first bytes; a TGA, which has no signature, by its
/// header as stb_image reads it.
pub fn sniff(bytes: []const u8) Format {
    if (std.mem.startsWith(u8, bytes, "GIF87a") or std.mem.startsWith(u8, bytes, "GIF89a")) return .gif;
    switch (picture.sniff(bytes)) {
        .png => return .png,
        .jpeg => return .jpeg,
        .webp => return .webp,
        .unknown => {},
    }
    if (std.mem.startsWith(u8, bytes, "BM")) return .bmp;
    if (std.mem.startsWith(u8, bytes, "8BPS")) return .psd;
    if (bytes.len >= 3 and bytes[0] == 'P' and (bytes[1] == '5' or bytes[1] == '6') and std.ascii.isWhitespace(bytes[2])) return .pnm;
    if (stb.takesTga(bytes)) return .tga;
    return .unknown;
}

/// The picture as oriented RGB rows, three bytes a pixel; the caller frees it with `Rgb.deinit`.
pub fn decode(a: Allocator, bytes: []const u8) Error!Rgb {
    if (bytes.len > max_bytes) return error.TooManyBytes;
    const s = switch (sniff(bytes)) {
        .png => picture.png.decode(a, bytes) catch |e| switch (e) {
            error.Animated => try firstPng(a, bytes),
            else => return e,
        },
        .jpeg => try jpeg.decode(a, bytes),
        .gif => try gif.decode(a, bytes),
        .webp => try webp.decode(a, bytes),
        .bmp, .psd, .pnm, .tga => try stb.decode(a, bytes),
        .unknown => return error.Invalid,
    };
    defer a.free(s.data);
    return picture.finish(a, s);
}

/// An animated PNG's first frame, as vLLM's Pillow loads it (spec 019 WS-04): its default image, the image data,
/// which APNG makes the first frame whether or not a frame control comes before it, and which Pillow composites
/// over nothing. The file without its animation chunks (acTL, fcTL, fdAT) is that picture still.
fn firstPng(a: Allocator, bytes: []const u8) Error!picture.Samples {
    const still = try a.alloc(u8, bytes.len);
    defer a.free(still);
    const head = 8;
    @memcpy(still[0..head], bytes[0..head]);
    var len: usize = head;
    var at: usize = head;
    while (bytes.len - at >= 12) {
        const size: usize = std.mem.readInt(u32, bytes[at..][0..4], .big);
        if (size > bytes.len - at - 12) return error.Invalid;
        const whole = bytes[at..][0 .. 12 + size];
        const kind = whole[4..8];
        if (!std.mem.eql(u8, kind, "acTL") and !std.mem.eql(u8, kind, "fcTL") and !std.mem.eql(u8, kind, "fdAT")) {
            @memcpy(still[len..][0..whole.len], whole);
            len += whole.len;
        }
        at += whole.len;
    }
    @memcpy(still[len..][0 .. bytes.len - at], bytes[at..]);
    len += bytes.len - at;
    return picture.png.decode(a, still[0..len]);
}

/// 0.6.5's words for each refusal, which a client reads.
pub fn problem(err: Error) []const u8 {
    return switch (err) {
        error.TooManyBytes => "image is empty or exceeds the encoded byte limit",
        error.Invalid => "image bytes are invalid or unsupported; use JPEG, PNG or WebP",
        // the rest as 1.0.12 words them (its libjpeg-turbo refusals are its own JPEG decoder's, never called here)
        else => |e| picture.problem(e),
    };
}
