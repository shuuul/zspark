//! JPEG through stb_image (`stb_image.c`), behind the markers read as Pillow's open read them (TensorFold 1.0.12's
//! `jpeg.walk`): the size checked before a pixel is allocated, CMYK and data that stops before its end marker
//! refused, EXIF and XMP for the orientation. stb_image's IDCT and chroma upsampling are not libjpeg-turbo's,
//! so the pixels are near Pillow's, not equal (`tests/image/jpeg_test.zig` measures how near); TensorFold 1.0.12
//! loads libjpeg-turbo 3.1.4.1 at run time for Pillow's bits, which zspark does not (spec 019's Decisions).
//! stb_image reads neither arithmetic coding nor lossless JPEG, which libjpeg-turbo does: those are refused.

const std = @import("std");
const Allocator = std.mem.Allocator;
const picture = @import("picture");

extern fn zs_image_jpeg(data: [*]const u8, len: usize, out: [*]u8, width: c_int, height: c_int) c_int;

pub fn decode(a: Allocator, data: []const u8) picture.Error!picture.Samples {
    var m: picture.jpeg.Markers = .{};
    defer m.exif.deinit(a);
    try picture.jpeg.walk(a, data, &m);
    try picture.fits(m.width, m.height);
    if (m.components == 4) return error.CmykJpeg;
    if (!m.ended) return error.Invalid; // Pillow's load refused the bytes that stop before the end marker
    const exif: ?[]const u8 = if (m.exif.items.len > 0) m.exif.items else null;
    const o = try picture.tags.orientation(exif, m.xmp, if (m.dpi) .refuse else .ignore);
    const w: usize = m.width;
    const h: usize = m.height;
    const rgb = try a.alloc(u8, w * h * 3);
    errdefer a.free(rgb);
    if (zs_image_jpeg(data.ptr, data.len, rgb.ptr, m.width, m.height) != 0) return error.Invalid;
    return .{ .width = w, .height = h, .channels = 3, .data = rgb, .orientation = o };
}
