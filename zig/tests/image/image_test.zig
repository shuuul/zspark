//! `zig/src/image/` against TensorFold 0.6.5's `_decode` over Pillow 12.3.0 (`fixtures/image/cases.json`, written
//! by `tools/reference/image_cases.py`): every case decoded to the same RGB, or refused in the same words.
//! PNG, GIF and WebP are Pillow's bytes; JPEG is stb_image's and is held to a measured distance. BMP, TGA, PSD
//! and PPM, which 0.6.5 refused, are vLLM's steps over Pillow (spec 019 WS-04), stb_image's bytes equal to them.

const std = @import("std");
const image = @import("../../src/image/image.zig");

const Case = struct {
    name: []const u8,
    format: []const u8,
    bytes: []const u8,
    note: ?[]const u8 = null,
    width: ?usize = null,
    height: ?usize = null,
    rgb: ?[]const u8 = null,
    problem: ?[]const u8 = null,
};

const Cases = struct {
    versions: std.json.Value,
    cases: []Case,
};

/// Where zspark refuses on purpose what Pillow decoded, and how.
const departures = [_]struct { name: []const u8, err: image.Error }{
    .{ .name = "jpeg_cmyk", .err = error.CmykJpeg }, // as TensorFold 1.0.12: Pillow's CMYK formula is not stb_image's
    .{ .name = "psd_gray", .err = error.Invalid }, // stb_image reads a PSD of RGB colour alone (spec 019 WS-04)
    .{ .name = "ppm_p3", .err = error.Invalid }, // PNM's text forms: stb_image reads P5 and P6 alone (spec 019 WS-04)
};

/// stb_image's JPEG against libjpeg-turbo 3.1.4.1's, over every JPEG case: the largest and the mean distance of
/// a sample. Measured on the vendored stb_image 2.30 (its PATCHES.md): 3 and 0.078 over 78,864 samples, from
/// its integer IDCT, YCbCr conversion and the rounding of its chroma upsampling.
const jpeg_max = 3;
const jpeg_mean = 0.1;

fn departure(name: []const u8) ?@TypeOf(departures[0]) {
    for (departures) |d| if (std.mem.eql(u8, d.name, name)) return d;
    return null;
}

fn decoded(gpa: std.mem.Allocator, b64: []const u8) ![]u8 {
    const d = std.base64.standard.Decoder;
    const out = try gpa.alloc(u8, try d.calcSizeForSlice(b64));
    try d.decode(out, b64);
    return out;
}

test "every picture decodes to Pillow's RGB under TensorFold 0.6.5's steps, or is refused in its words" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(Cases, gpa, @embedFile("fixtures/image/cases.json"), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("12.3.0", parsed.value.versions.object.get("pillow").?.string);
    var jpeg_worst: u32 = 0;
    var jpeg_sum: u64 = 0;
    var jpeg_samples: u64 = 0;
    var formats = std.StringHashMap(void).init(gpa);
    defer formats.deinit();
    var wrong: usize = 0;
    for (parsed.value.cases) |c| {
        try formats.put(c.format, {});
        const bytes = try decoded(gpa, c.bytes);
        defer gpa.free(bytes);
        const departed = departure(c.name);
        var got = image.decode(gpa, bytes) catch |err| {
            const words = c.problem orelse "";
            const meant = if (departed) |d| d.err == err else false;
            if (!meant and !std.mem.eql(u8, words, image.problem(err))) {
                std.debug.print("{s}: {t} where Pillow gave {s}\n", .{ c.name, err, if (c.problem == null) "pixels" else words });
                wrong += 1;
            }
            continue;
        };
        defer got.deinit(gpa);
        if (departed != null or c.problem != null) {
            std.debug.print("{s}: decoded where Pillow refused it: {s}\n", .{ c.name, c.problem orelse "(a departure)" });
            wrong += 1;
            continue;
        }
        const want = try decoded(gpa, c.rgb.?);
        defer gpa.free(want);
        if (c.width.? != got.width or c.height.? != got.height) {
            std.debug.print("{s}: {d} x {d} where Pillow gave {d} x {d}\n", .{ c.name, got.width, got.height, c.width.?, c.height.? });
            wrong += 1;
            continue;
        }
        if (!std.mem.eql(u8, c.format, "jpeg")) {
            if (!std.mem.eql(u8, want, got.pixels)) {
                const at = std.mem.indexOfDiff(u8, want, got.pixels).?;
                std.debug.print("{s}: byte {d} is {d} where Pillow gave {d}\n", .{ c.name, at, got.pixels[at], want[at] });
                wrong += 1;
            }
            continue;
        }
        var worst: u32 = 0;
        var sum: u64 = 0;
        for (want, got.pixels) |x, y| {
            const d = @abs(@as(i32, x) - y);
            worst = @max(worst, d);
            sum += d;
        }
        jpeg_worst = @max(jpeg_worst, worst);
        jpeg_sum += sum;
        jpeg_samples += want.len;
    }
    try std.testing.expectEqual(0, wrong);
    for ([_][]const u8{ "png", "jpeg", "gif", "webp", "bmp", "tga", "psd", "ppm", "other" }) |f| try std.testing.expect(formats.contains(f));
    const mean = @as(f64, @floatFromInt(jpeg_sum)) / @as(f64, @floatFromInt(jpeg_samples));
    if (jpeg_worst > jpeg_max or mean > jpeg_mean) std.debug.print("jpeg against Pillow: max {d}, mean {d:.3} over {d} samples\n", .{ jpeg_worst, mean, jpeg_samples });
    try std.testing.expect(jpeg_worst <= jpeg_max);
    try std.testing.expect(mean <= jpeg_mean);
}

test "the bounds refuse before a pixel is allocated: bytes, a side, the pixels" {
    var none: [0]u8 = .{};
    var fixed = std.heap.FixedBufferAllocator.init(&none);
    const a = fixed.allocator();
    // a PNG header of 9000 x 1, a GIF screen of 9000 x 1, a WebP of 5000 x 5000: refused with no memory at all
    const png = "\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR\x00\x00\x23\x28\x00\x00\x00\x01\x08\x02\x00\x00\x00\x3c\x41\x92\x12";
    try std.testing.expectError(error.Oversized, image.decode(a, png ++ "\x00\x00\x00\x00IDAT\x35\xaf\x06\x1e\x00\x00\x00\x00IEND\xae\x42\x60\x82"));
    const gif = "GIF89a\x28\x23\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00;";
    try std.testing.expectError(error.Oversized, image.decode(a, gif));
    // a lossless WebP's header: 14 bits a side less one, so 5000 x 5000 is 4999 and 4999
    const side: u32 = 4999;
    var webp = "RIFF\x1a\x00\x00\x00WEBPVP8L\x0d\x00\x00\x00\x2f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00".*;
    std.mem.writeInt(u32, webp[21..25], side | side << 14, .little);
    try std.testing.expectError(error.Oversized, image.decode(a, &webp));
    const big = try std.testing.allocator.alloc(u8, image.max_bytes + 1);
    defer std.testing.allocator.free(big);
    @memcpy(big[0..png.len], png);
    try std.testing.expectError(error.TooManyBytes, image.decode(a, big));
}

test "every allocation that fails is an error, never a crash or a leak" {
    const parsed = try std.json.parseFromSlice(Cases, std.testing.allocator, @embedFile("fixtures/image/cases.json"), .{});
    defer parsed.deinit();
    for ([_][]const u8{ "png_palette4_alpha", "png_orient6", "gif_small_frame_transparent", "jpeg_orient6", "webp_lossy_alpha" }) |name| {
        const c = for (parsed.value.cases) |c| {
            if (std.mem.eql(u8, c.name, name)) break c;
        } else return error.NoSuchCase;
        const bytes = try decoded(std.testing.allocator, c.bytes);
        defer std.testing.allocator.free(bytes);
        try std.testing.checkAllAllocationFailures(std.testing.allocator, decodeFreed, .{bytes});
    }
}

fn decodeFreed(a: std.mem.Allocator, bytes: []const u8) !void {
    var got = try image.decode(a, bytes);
    got.deinit(a);
}
