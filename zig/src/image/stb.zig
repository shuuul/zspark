//! BMP, TGA, PSD and PNM (binary P5 and P6) through stb_image (`stb_image.c`; spec 019 WS-04), the formats
//! vLLM's Pillow opens beside JPEG, PNG, GIF and WebP: the size read from the header and checked before a pixel
//! is allocated, then the picture as RGBA where it has alpha (gray and alpha, or RGBA) and as RGB where it has
//! none; `picture.finish` puts the alpha over white, as vLLM's `rgba_to_rgb`. None carries an orientation tag
//! Pillow's `exif_transpose` reads. stb_image reads PSD's merged picture, as Pillow's first frame is.

const std = @import("std");
const Allocator = std.mem.Allocator;
const picture = @import("picture");

extern fn zs_image_info(data: [*]const u8, len: usize, width: *c_int, height: *c_int, channels: *c_int) c_int;
extern fn zs_image_is_tga(data: [*]const u8, len: usize) c_int;
extern fn zs_image_stb(data: [*]const u8, len: usize, out: [*]u8, width: c_int, height: c_int, channels: c_int) c_int;

/// Whether stb_image identifies the header as TGA, the only format with no signature in `image.sniff`.
pub fn takesTga(data: []const u8) bool {
    return data.len > 0 and zs_image_is_tga(data.ptr, data.len) != 0;
}

pub fn decode(a: Allocator, data: []const u8) picture.Error!picture.Samples {
    var w: c_int = 0;
    var h: c_int = 0;
    var n: c_int = 0;
    if (data.len == 0 or zs_image_info(data.ptr, data.len, &w, &h, &n) == 0 or w <= 0 or h <= 0) return error.Invalid;
    try picture.fits(@intCast(w), @intCast(h));
    const channels: usize = if (n == 2 or n == 4) 4 else 3;
    const width: usize = @intCast(w);
    const height: usize = @intCast(h);
    const out = try a.alloc(u8, width * height * channels);
    errdefer a.free(out);
    if (zs_image_stb(data.ptr, data.len, out.ptr, w, h, @intCast(channels)) != 0) return error.Invalid;
    return .{ .width = width, .height = height, .channels = channels, .data = out };
}
