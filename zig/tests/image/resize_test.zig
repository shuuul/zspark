//! `zig/src/image/resize.zig` against Pillow 12.3.0 (spec 147 WS-02): made pictures resized and padded by Pillow
//! (`fixtures/image/resize_cases.json`, `tools/reference/unlimited_ocr/resize_cases.py`), and where
//! `ZSPARK_REFERENCE_UNLIMITED_OCR_2` names it, the checkpoint's own global views and tiles of its fixture
//! pages: every byte equal.

const std = @import("std");
const testing = std.testing;
const image = @import("../../src/image/image.zig");
const picture = @import("picture");
const resize = @import("../../src/image/resize.zig");
const safetensors = @import("../../src/weights/safetensors.zig");

const Case = struct {
    op: []const u8,
    name: []const u8,
    width: usize,
    height: usize,
    rgb: []const u8,
    to: [2]usize,
    fill: u8,
    out_size: [2]usize,
    sha256: []const u8,
};

fn hex(bytes: []const u8) [64]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

test "image resize and pad equal Pillow's bytes on made pictures" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(struct { cases: []Case }, gpa, @embedFile("fixtures/image/resize_cases.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const d = std.base64.standard.Decoder;
    for (parsed.value.cases) |c| {
        const src = try gpa.alloc(u8, try d.calcSizeForSlice(c.rgb));
        defer gpa.free(src);
        try d.decode(src, c.rgb);
        try testing.expectEqual(c.width * c.height * 3, src.len);
        const isPad = std.mem.eql(u8, c.op, "pad");
        const got = if (isPad)
            try resize.pad(gpa, src, c.width, c.height, c.to[0], c.to[1], c.fill)
        else
            try resize.resize(gpa, src, c.width, c.height, c.to[0], c.to[1]);
        defer gpa.free(got);
        try testing.expectEqual(c.out_size[0] * c.out_size[1] * 3, got.len);
        testing.expectEqualStrings(c.sha256, &hex(got)) catch |e| {
            std.debug.print("case {s}\n", .{c.name});
            return e;
        };
    }
}

fn openReference() !std.Io.Dir {
    const path = testing.environ.getPosix("ZSPARK_REFERENCE_UNLIMITED_OCR_2") orelse return error.SkipZigTest;
    return std.Io.Dir.cwd().openDir(testing.io, path, .{});
}

/// HF's input conversion drops alpha; the server's image loader composites over white instead.
fn rgbDroppingAlpha(gpa: std.mem.Allocator, bytes: []const u8) !image.Rgb {
    const s = try picture.png.decode(gpa, bytes);
    defer gpa.free(s.data);
    const out = try gpa.alloc(u8, s.width * s.height * 3);
    for (0..s.width * s.height) |i| @memcpy(out[i * 3 ..][0..3], s.data[i * s.channels ..][0..3]);
    return .{ .width = s.width, .height = s.height, .pixels = out };
}

fn pageOf(gpa: std.mem.Allocator, dir: std.Io.Dir, path: []const u8) !image.Rgb {
    const bytes = try dir.readFileAlloc(testing.io, path, gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    return rgbDroppingAlpha(gpa, bytes);
}

test "image pad and resize equal the checkpoint's views on its fixture pages" {
    // spec 147's export (tools/reference/unlimited_ocr/export_v2.py): `one_page` is paper_p01 in gundam mode
    // (a global view and crops, `images_spatial_crop` their grid), `two_pages` paper_p01 and paper_p03 in
    // base mode (a global view a page, no crops).
    var dir = try openReference();
    defer dir.close(testing.io);
    const gpa = testing.allocator;
    const cases = [_]struct { name: []const u8, pages: []const []const u8 }{
        .{ .name = "one_page", .pages = &.{"pages/paper_p01.png"} },
        .{ .name = "two_pages", .pages = &.{ "pages/paper_p01.png", "pages/paper_p03.png" } },
    };
    for (cases) |c| {
        var path_buf: [40]u8 = undefined;
        var file = try safetensors.File.open(gpa, testing.io, dir, try std.fmt.bufPrint(&path_buf, "{s}.safetensors", .{c.name}));
        defer file.close();
        const want = file.get("views.global_u8").?;
        const spatial = file.get("images_spatial_crop").?;
        for (c.pages, 0..) |page_path, p| {
            var page = try pageOf(gpa, dir, page_path);
            defer page.deinit(gpa);
            const got = try resize.pad(gpa, page.pixels, page.width, page.height, 1024, 1024, 127);
            defer gpa.free(got);
            const global = want.data[p * 1024 * 1024 * 3 ..][0 .. 1024 * 1024 * 3];
            testing.expect(std.mem.eql(u8, global, got)) catch |e| {
                std.debug.print("global view of {s} page {d} differs\n", .{ c.name, p });
                return e;
            };
        }
        const crops = file.get("views.crops_u8") orelse continue;
        // the crops are cut from one resize of the page to (640 w, 640 h): stitched back, equal to it
        const gw: usize = @intCast(std.mem.readInt(i32, spatial.data[0..4], .little));
        const gh: usize = @intCast(std.mem.readInt(i32, spatial.data[4..8], .little));
        const tiles = crops.shape[0];
        try testing.expectEqual(gw * gh, tiles);
        var page = try pageOf(gpa, dir, c.pages[0]);
        defer page.deinit(gpa);
        const whole = try resize.resize(gpa, page.pixels, page.width, page.height, 640 * gw, 640 * gh);
        defer gpa.free(whole);
        for (0..tiles) |t| for (0..640) |y| {
            const at = (((t / gw) * 640 + y) * gw * 640 + (t % gw) * 640) * 3;
            try testing.expectEqualSlices(u8, crops.data[(t * 640 + y) * 640 * 3 ..][0 .. 640 * 3], whole[at..][0 .. 640 * 3]);
        };
    }
}
