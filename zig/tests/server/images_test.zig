//! `images.zig`'s tests on the host (spec 019 WS-04): the `Offer` against TensorFold 0.6.5's processor
//! (`zig/tests/fixtures/vision/cases.json`, WS-01's), alone and under the vendored host's `expand`, which writes
//! each image's tokens where the template put its placeholder; and its refusals, in vLLM's words. Every test's
//! name says "image", for `tools/check.sh server image`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const testing = std.testing;
const api = @import("engine_api");
const host = @import("server");
const images = @import("../../src/server/images.zig");
const vision = @import("../../src/text/vision.zig");
const fixture = @import("served_fixture.zig");

/// Flash Next's settings, as its config.json and processor_config.json give them (`vision_test.zig`'s).
const flash: vision.Settings = .{ .patch = 16, .temporal = 2, .merge = 2, .image_token = 248056, .video_token = 248057, .vision_start = 248053, .vision_end = 248054 };

const pictures = .{
    .{ "shapes", @embedFile("fixtures/vision/shapes.png") },
    .{ "bands", @embedFile("fixtures/vision/bands.png") },
    .{ "tiny", @embedFile("fixtures/vision/tiny.png") },
};

fn picture(name: []const u8) []const u8 {
    inline for (pictures) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
    unreachable;
}

fn field(o: Value, name: []const u8) Value {
    return o.object.get(name).?;
}

fn int(v: Value) u32 {
    return @intCast(v.integer);
}

test "image cases: each image's patches, grid, tokens and content hash are the processor's at the host's share of the budget" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const doc = try std.json.parseFromSliceLeaky(Value, arena, @embedFile("fixtures/vision/cases.json"), .{});
    const offer: images.Offer = .init(flash);
    var it = field(doc, "cases").object.iterator();
    var checked: usize = 0;
    while (it.next()) |e| {
        const case = e.value_ptr.*;
        const names = field(case, "images").array.items;
        const grids = field(case, "image_grid_thw").array.items;
        const hashes = field(case, "image_hashes").array.items;
        var sha: std.crypto.hash.sha2.Sha256 = .init(.{});
        var tokens: u32 = 0;
        for (names, grids, hashes) |name, grid, hash| {
            // the host's share of 0.6.5's 4,096 (`expand`'s `image_tokens / n_images`)
            const p = try offer.prepare(arena, picture(name.string), images.image_tokens / @as(u32, @intCast(names.len)));
            try testing.expectEqual([2]u32{ int(grid.array.items[1]), int(grid.array.items[2]) }, [2]u32{ p.gh, p.gw });
            try testing.expectEqual(try std.fmt.parseInt(u64, hash.string[0..16], 16), p.hash);
            sha.update(std.mem.sliceAsBytes(p.pixels));
            tokens += p.tokens;
        }
        const pv = field(case, "pixel_values");
        try testing.expectEqualStrings(field(pv, "sha256").string, &std.fmt.bytesToHex(sha.finalResult(), .lower));
        try testing.expectEqual(int(field(case, "visual_tokens")), tokens);
        checked += 1;
    }
    try testing.expectEqual(2, checked);
}

test "image cases under the host's expand: the prompt's ids, each image's rows and the positions and delta are 0.6.5's" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const doc = try std.json.parseFromSliceLeaky(Value, arena, @embedFile("fixtures/vision/cases.json"), .{});
    const offer: images.Offer = .init(flash);
    var bytes: fixture.Bytes = .{};
    var it = field(doc, "cases").object.iterator();
    while (it.next()) |e| {
        const case = e.value_ptr.*;
        // the encoded render: 0.6.5's ids with each image's run of placeholders one placeholder, as the template wrote it
        var rendered: std.ArrayList(u32) = .empty;
        var last: u32 = 0;
        for (field(case, "ids").array.items) |v| {
            const id = int(v);
            if (!(id == flash.image_token and last == flash.image_token)) try rendered.append(arena, id);
            last = id;
        }
        var parts: std.ArrayList(u8) = .empty;
        for (field(case, "images").array.items, 0..) |name, i| {
            const b64 = try arena.alloc(u8, std.base64.standard.Encoder.calcSize(picture(name.string).len));
            _ = std.base64.standard.Encoder.encode(b64, picture(name.string));
            try parts.print(arena, "{s}{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:image/png;base64,{s}\"}}}}", .{ if (i > 0) "," else "", b64 });
        }
        const messages = switch (try host.json.parse(arena, try std.fmt.allocPrint(arena, "[{{\"role\":\"user\",\"content\":[{s}]}}]", .{parts.items}))) {
            .ok => |v| v,
            .err => return error.NotJson,
        };
        var cx: host.errors.Cx = .{ .a = arena };
        const out = (host.images.expand(offer.vision(), null, bytes.text(), &cx, messages, rendered.items, 0) catch |err| {
            std.debug.print("refused: {s}\n", .{cx.message});
            return err;
        }).?;
        const ids = field(case, "ids").array.items;
        try testing.expectEqual(ids.len, out.ids.len);
        for (ids, out.ids) |want, got| try testing.expectEqual(int(want), got);
        const spans = field(case, "image_spans").array.items;
        try testing.expectEqual(spans.len, out.images.len);
        for (spans, out.images) |span, img| try testing.expectEqual([2]u32{ int(span.array.items[0]), int(span.array.items[1]) }, [2]u32{ img.at, img.at + img.tokens });

        // what the model makes of the request (`models/qwen4/served.zig`): the layout over the expanded prompt
        const grids = try arena.alloc([3]u32, out.images.len);
        for (out.images, grids) |img, *g| g.* = .{ 1, img.gh, img.gw };
        var problem: vision.Problem = .{};
        var layout = try vision.Layout.init(arena, out.ids, grids, .{ .start = flash.vision_start, .image = flash.image_token, .end = flash.vision_end, .video = flash.video_token }, flash.merge, &problem);
        try testing.expectEqual(@as(i32, @intCast(field(case, "rope_delta").integer)), layout.shift);
        const positions = field(case, "position_ids").array.items;
        for (0..out.ids.len) |p| {
            const got = layout.position(@intCast(p));
            for (0..3) |axis| try testing.expectEqual(@as(i32, @intCast(positions[axis].array.items[p].integer)), got[axis]);
        }
    }
}

test "image formats the host decodes: PNG, JPEG, GIF's first frame, WebP and stb_image's BMP, PPM each prepared" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const offer: images.Offer = .init(flash);
    for ([_][]const u8{ @embedFile("fixtures/vision/shapes.png"), @embedFile("fixtures/vision/shapes.jpeg"), @embedFile("fixtures/vision/shapes.gif"), @embedFile("fixtures/vision/shapes.webp") }) |bytes| {
        const p = try offer.prepare(arena, bytes, images.image_tokens);
        try testing.expectEqual([3]u32{ 16, 16, 64 }, [3]u32{ p.gh, p.gw, p.tokens });
    }
    // a 32 x 32 BMP and binary PPM written here: one merged row of 2 x 2 patches, a token
    var bmp: [54 + 32 * 32 * 3]u8 = @splat(0x80);
    @memcpy(bmp[0..2], "BM");
    std.mem.writeInt(u32, bmp[2..6], bmp.len, .little);
    std.mem.writeInt(u32, bmp[10..14], 54, .little);
    std.mem.writeInt(u32, bmp[14..18], 40, .little);
    std.mem.writeInt(i32, bmp[18..22], 32, .little);
    std.mem.writeInt(i32, bmp[22..26], 32, .little);
    std.mem.writeInt(u16, bmp[26..28], 1, .little);
    std.mem.writeInt(u16, bmp[28..30], 24, .little);
    std.mem.writeInt(u32, bmp[30..34], 0, .little);
    std.mem.writeInt(u32, bmp[34..38], 32 * 32 * 3, .little);
    const head = "P6\n32 32\n255\n";
    var ppm: [head.len + 32 * 32 * 3]u8 = @splat(0x40);
    @memcpy(ppm[0..head.len], head);
    for ([_][]const u8{ &bmp, &ppm }) |bytes| {
        const p = try offer.prepare(arena, bytes, images.image_tokens);
        try testing.expectEqual([3]u32{ 2, 2, 1 }, [3]u32{ p.gh, p.gw, p.tokens });
    }
    // a low detail's cap is the host's (`expand`'s 256), not the processor's
    const low = try offer.prepare(arena, picture("bands"), 16);
    try testing.expect(low.tokens <= 16);
}

test "image refusals are vLLM's words: an unknown format, a named gap, damaged bytes, too many pixels, an aspect past 200" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const offer: images.Offer = .init(flash);
    const png = picture("tiny");
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "not a picture", "Failed to load image: cannot identify image file" },
        .{ "II*\x00\x08\x00\x00\x00", "Failed to load image: TIFF images are not decoded by this server; send JPEG, PNG, GIF, WebP, BMP, TGA, PSD or PPM" },
        .{ "\x00\x00\x00\x1cftypavif\x00\x00\x00\x00", "Failed to load image: AVIF images are not decoded by this server; send JPEG, PNG, GIF, WebP, BMP, TGA, PSD or PPM" },
        .{ "\x00\x00\x01\x00\x01\x00", "Failed to load image: ICO images are not decoded by this server; send JPEG, PNG, GIF, WebP, BMP, TGA, PSD or PPM" },
        .{ "P3\n2 1\n255\n255 0 0 0 128 255\n", "Failed to load image: PBM, PGM and PPM images in text or 1-bit form (P1 to P4) are not decoded by this server; send JPEG, PNG, GIF, WebP, BMP, TGA, PSD or PPM" },
        .{ png[0 .. png.len / 2], "Failed to load image: image file is truncated or damaged" },
    };
    for (cases) |case| {
        const err = if (offer.prepare(arena, case[0], images.image_tokens)) |_| return error.Prepared else |e| e;
        errdefer std.debug.print("{s}: {t}\n", .{ case[1], err });
        try testing.expectEqualStrings(case[1], images.explain(err).?);
    }
    try testing.expectEqualStrings("Image dimensions exceed the maximum of 8192 pixels a side or 16777216 pixels", images.explain(error.Oversized).?);
    try testing.expectEqualStrings("absolute aspect ratio must be smaller than 200", images.explain(error.AspectRatio).?);
    try testing.expectEqual(null, images.explain(error.OutOfMemory));
}

test "image offer: the host's Info.vision is the checkpoint's markers at 0.6.5's budgets" {
    const offer: images.Offer = .init(flash);
    const v: api.Vision = offer.vision();
    try testing.expectEqual(flash.image_token, v.image_token);
    try testing.expectEqualSlices(u32, &.{ flash.vision_start, flash.image_token, flash.vision_end }, v.markers);
    try testing.expectEqual(4096, v.image_tokens);
    try testing.expectEqual(4, v.max_images);
    try testing.expectEqual(null, v.wrap);
    try testing.expectEqual(null, v.video);
}

test "image settings of the checkpoint images are served on: FP4-Local's files give Flash Next's" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set; skipped without. Its JSON files alone are read.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    var dir = try std.Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    var problem: vision.Problem = .{};
    const s = (vision.Settings.load(arena_state.allocator(), testing.io, dir, &problem) catch |err| {
        std.debug.print("{s}\n", .{problem.message()});
        return err;
    }).?;
    try testing.expectEqual(flash, s);
}
