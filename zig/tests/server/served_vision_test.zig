//! `served.zig`'s images on the host (spec 019 WS-04): `Hi` behind TensorFold's server with `images.zig`'s
//! `Offer` as its `Info.vision`, asked over a socket on the three APIs. The text is `served_fixture.zig`'s
//! `Bytes`, whose render writes an image part as its placeholder byte. What Flash Next makes of an image is
//! `models/qwen4/`'s (the GPU, after WS-08). Every test's name says "image", for `tools/check.sh server image`.

const std = @import("std");
const testing = std.testing;
const images = @import("../../src/server/images.zig");
const vision = @import("../../src/text/vision.zig");
const fixture = @import("served_fixture.zig");
const Stand = fixture.Stand;
const expectRefused = fixture.expectRefused;
const bodyOf = fixture.bodyOf;

/// Flash Next's geometry, its markers the bytes `Bytes` writes.
const settings: vision.Settings = .{ .patch = 16, .temporal = 2, .merge = 2, .image_token = fixture.image_token, .video_token = null, .vision_start = 2, .vision_end = 3 };
const tiny = @embedFile("fixtures/vision/tiny.png");

fn dataUrl(arena: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    const out = try arena.alloc(u8, std.base64.standard.Encoder.calcSize(bytes.len));
    return std.fmt.allocPrint(arena, "data:image/png;base64,{s}", .{std.base64.standard.Encoder.encode(out, bytes)});
}

fn chatError(arena: std.mem.Allocator, message: []const u8) ![]const u8 {
    return std.fmt.allocPrint(arena, "{{\"error\": {{\"message\": {f}, \"type\": \"invalid_request_error\"}}}}", .{std.json.fmt(message, .{})});
}

fn chatWith(arena: std.mem.Allocator, url: []const u8) ![]const u8 {
    return std.fmt.allocPrint(arena, "{{\"model\":\"hi\",\"max_tokens\":4,\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{s}\"}}}},{{\"type\":\"text\",\"text\":\"what\"}}]}}]}}", .{url});
}

test "an image part reaches the model on the three APIs: its tower input, and its tokens where the template put its placeholder" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const offer: images.Offer = .init(settings);
    var stand: Stand = undefined;
    try stand.initWith(gpa, .{ .vision = offer.vision() });
    defer stand.deinit();
    const url = try dataUrl(arena, tiny);
    const b64 = url["data:image/png;base64,".len..];

    const asked = [_]struct { []const u8, []const u8 }{
        .{ "/v1/chat/completions", try chatWith(arena, url) },
        .{ "/v1/responses", try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",\"max_output_tokens\":4,\"input\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"input_image\",\"image_url\":\"{s}\"}},{{\"type\":\"input_text\",\"text\":\"what\"}}]}}]}}", .{url}) },
        .{ "/v1/messages", try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",\"max_tokens\":4,\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"image\",\"source\":{{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"{s}\"}}}},{{\"type\":\"text\",\"text\":\"what\"}}]}}]}}", .{b64}) },
    };
    for (asked, 1..) |case, n| {
        _ = try bodyOf(try stand.post(arena, case[0], case[1]), "HTTP/1.1 200 OK\r\n");
        const behind = &stand.served.behind;
        try testing.expectEqual(n, behind.with_images.load(.acquire));
        // tiny.png, 48 x 40: a grid of 2 x 4 patches, 2 tokens (0.6.5's `pair` case gives it the same)
        try testing.expectEqual(1, behind.images.len);
        const img = behind.images[0];
        try testing.expectEqual([4]u32{ 2, 4, 2, 1 }, [4]u32{ img.gh, img.gw, img.tokens, img.at });
        // "|", the image's 2 rows, "what"
        try testing.expectEqual([2]usize{ 2, 7 }, [2]usize{ behind.image_rows, behind.prompt_len });
    }
}

test "image refusals are 400s before anything is decoded: vLLM's words for bytes the server cannot decode, the host's for the rest" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const offer: images.Offer = .init(settings);
    var stand: Stand = undefined;
    try stand.initWith(gpa, .{ .vision = offer.vision() });
    defer stand.deinit();

    const cases = [_]struct { []const u8, []const u8 }{
        .{ try dataUrl(arena, "not a picture"), "Failed to load image: cannot identify image file" },
        .{ try dataUrl(arena, "II*\x00\x08\x00\x00\x00"), "Failed to load image: TIFF images are not decoded by this server; send JPEG, PNG, GIF, WebP, BMP, TGA, PSD or PPM" },
        .{ try dataUrl(arena, tiny[0 .. tiny.len / 2]), "Failed to load image: image file is truncated or damaged" },
        // without --vision-urls; and a scheme the host never fetches
        .{ "https://example.com/a.png", "image URLs are off on this server; send the image as a data URL, or start the server with --vision-urls" },
        .{ "http://example.com/a.png", "images require data URLs or public HTTPS URLs" },
    };
    for (cases) |case| try expectRefused(try stand.post(arena, "/v1/chat/completions", try chatWith(arena, case[0])), try chatError(arena, case[1]));

    // more images than 0.6.5's four
    var parts: std.ArrayList(u8) = .empty;
    const url = try dataUrl(arena, tiny);
    for (0..5) |_| try parts.print(arena, "{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{s}\"}}}},", .{url});
    const five = try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",\"messages\":[{{\"role\":\"user\",\"content\":[{s}{{\"type\":\"text\",\"text\":\"what\"}}]}}]}}", .{parts.items});
    try expectRefused(try stand.post(arena, "/v1/chat/completions", five), try chatError(arena, "a request may hold at most 4 images (--vision-max-images); it has 5"));
    try testing.expectEqual(0, stand.served.behind.replies.load(.acquire));
}

test "an image part where the model offers no vision is today's 400, on the three APIs" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.init(gpa);
    defer stand.deinit();
    const url = try dataUrl(arena, tiny);
    const said = "this server accepts text parts only; image, audio and video inputs are unsupported";
    try expectRefused(try stand.post(arena, "/v1/chat/completions", try chatWith(arena, url)), try chatError(arena, said));
    const responses = try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",\"input\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"input_image\",\"image_url\":\"{s}\"}}]}}]}}", .{url});
    try expectRefused(try stand.post(arena, "/v1/responses", responses), try chatError(arena, said));
    const messages = try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",\"max_tokens\":4,\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"image\",\"source\":{{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"{s}\"}}}}]}}]}}", .{url["data:image/png;base64,".len..]});
    try expectRefused(try stand.post(arena, "/v1/messages", messages), try std.fmt.allocPrint(arena, "{{\"type\": \"error\", \"error\": {{\"type\": \"invalid_request_error\", \"message\": {f}}}}}", .{std.json.fmt(said, .{})}));
    try testing.expectEqual(0, stand.served.behind.replies.load(.acquire));
}

test "a text-only request whose text spells the image placeholder is its text, not an image (vLLM): a model with no shielded markers" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const offer: images.Offer = .init(settings);
    // Unlimited-OCR's: no marker is shielded, so the `<image>` a client writes is the placeholder's id
    var offered = offer.vision();
    offered.markers = &.{};
    var stand: Stand = undefined;
    try stand.initWith(gpa, .{ .vision = offered });
    defer stand.deinit();
    const behind = &stand.served.behind;

    // the placeholder's byte in a text part and in a string content: the host's 500 before, a reply now
    for ([_][]const u8{
        "{\"model\":\"hi\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"\\u0001what\"}]}]}",
        "{\"model\":\"hi\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"\\u0001what\"}]}",
    }) |body| {
        _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", body), "HTTP/1.1 200 OK\r\n");
        // "|", the placeholder's id as text, "what"; no image reached the model
        try testing.expectEqual([2]usize{ 1, 6 }, [2]usize{ behind.image_rows, behind.prompt_len });
        try testing.expectEqual(0, behind.with_images.load(.acquire));
    }
    // with an image part the placeholder the text writes is the image's
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", try chatWith(arena, try dataUrl(arena, tiny))), "HTTP/1.1 200 OK\r\n");
    try testing.expectEqual(1, behind.with_images.load(.acquire));
}
