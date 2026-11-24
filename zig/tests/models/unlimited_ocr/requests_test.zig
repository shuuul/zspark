//! `requests.zig`'s tests on the host (spec 147 WS-06): the page template under the host's masked encode and
//! `expand` gives HF's `input_ids` for spec 147's fixtures (one page in gundam mode, two in base mode),
//! whichever way a client writes the page prompt; a placeholder count `infer` refuses is refused; the rules
//! read the reply alone and guard its pages.

const std = @import("std");
const testing = std.testing;
const api = @import("engine_api");
const host = @import("server");
const template = @import("template");
const segments_file = @import("../../../src/server/segments.zig");
const ngram = @import("../../../src/decode/ngram.zig");
const masks = @import("../../../src/decode/masks.zig");
const requests = @import("../../../src/models/unlimited_ocr/requests.zig");
const preprocess = @import("../../../src/models/unlimited_ocr/preprocess.zig");

test "unlimited_ocr requests: the mode, offer and template" {
    const Offer = requests.Offer;
    // without a mode: HF's, gundam alone in its request, base among several
    try testing.expectEqual(preprocess.Mode.gundam, try Offer.modeOf(.{ .images = 1 }));
    try testing.expectEqual(preprocess.Mode.base, try Offer.modeOf(.{ .images = 2 }));
    // a mode named (SGLang's `_resolve_mode_v2`): stripped, lower case, auto gundam; tiny, small, large not served
    try testing.expectEqual(preprocess.Mode.gundam, try Offer.modeOf(.{ .images = 2, .mode = " Auto\n" }));
    try testing.expectEqual(preprocess.Mode.gundam, try Offer.modeOf(.{ .images = 2, .mode = "GUNDAM" }));
    try testing.expectEqual(preprocess.Mode.base, try Offer.modeOf(.{ .images = 1, .mode = "base" }));
    try testing.expectError(error.ImageModeUnserved, Offer.modeOf(.{ .images = 1, .mode = "large" }));
    try testing.expectError(error.ImageModeUnknown, Offer.modeOf(.{ .images = 1, .mode = "huge" }));
    try testing.expectError(error.ImageModeUnknown, Offer.modeOf(.{ .images = 1, .mode = "a much longer word" }));
    const v = Offer.vision();
    try testing.expectEqual(@as(u32, 128), v.max_images);
    // every page's share holds its largest gundam grid (16 tiles, 1 x 16), a request may ask gundam for each
    try testing.expectEqual(@as(u32, 2033), v.image_tokens / v.max_images);
}

test "unlimited_ocr requests: the rules read the reply and guard its pages" {
    var logits: [16]u16 = undefined;
    const prompt = [_]u32{ 3, 4, 3 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 8, .no_repeat = .{ .size = 2 }, .eos = &.{1} };
    // the reply alone, so the prompt's `3 4` bans nothing; two pages owed, the end banned
    const pages = [_]u32{ 0, preprocess.page_token_id, preprocess.image_token_id, preprocess.page_token_id, preprocess.image_token_id, 3, 4, 3 };
    var rules: requests.Rules = .of(request, &pages, 1);
    try testing.expectEqual(@as(u32, 2), rules.guard.?.expected);
    try testing.expectEqual(ngram.Rule.page.window, rules.ngram.?.window);
    var seen: [pages.len + 1]u32 = undefined;
    @memcpy(seen[0..pages.len], &pages);
    seen[pages.len] = 3;
    @memset(&logits, 0);
    rules.apply(&seen, pages.len, &logits);
    try testing.expectEqual(@as(u16, 0), logits[4]);
    try testing.expectEqual(masks.minus_infinity, logits[1]);
    // the request's cap; a request that ignores the end token has no guard
    var capped = request;
    capped.max_page_tokens = 300;
    try testing.expectEqual(@as(u32, 300), requests.Rules.of(capped, &pages, 1).guard.?.cap);
    capped.eos = &.{};
    try testing.expect(requests.Rules.of(capped, &pages, 1).guard == null);
    // one page: the image rule's window
    try testing.expectEqual(ngram.Rule.image.window, requests.Rules.of(request, pages[0..3], 1).ngram.?.window);
}

test "unlimited_ocr requests: SGLang's periodic tail rule (custom_params' rep_*) forces the end token first" {
    const rule: api.TailRepeat = .{ .min_size = 2, .max_size = 3, .min_count = 3 };
    // `_tail_has_repeat`: a tail of 2 or 3 tokens three times back to back
    try testing.expect(requests.tailRepeats(&.{ 9, 5, 6, 5, 6, 5, 6 }, rule));
    try testing.expect(requests.tailRepeats(&.{ 1, 2, 3, 1, 2, 3, 1, 2, 3 }, rule));
    try testing.expect(!requests.tailRepeats(&.{ 5, 6, 5, 6, 7, 6 }, rule));
    try testing.expect(!requests.tailRepeats(&.{ 5, 6, 5, 6 }, rule));
    // a size of 1 is 1 when the minimum is 0, a size past the reply none
    try testing.expect(requests.tailRepeats(&.{ 4, 4, 4 }, .{ .max_size = 1, .min_count = 3 }));
    try testing.expect(!requests.tailRepeats(&.{ 4, 4, 4 }, .{ .min_size = 2, .max_size = 9, .min_count = 3 }));
    // in the rules: before the guard and the n-gram rule; off unless max_size > 0 and min_count >= 2
    const prompt = [_]u32{ 0, preprocess.page_token_id, preprocess.image_token_id, preprocess.page_token_id, preprocess.image_token_id, 3 };
    var request: api.Request = .{ .prompt = &prompt, .max_tokens = 8, .eos = &.{1}, .tail_repeat = rule };
    const seen = prompt ++ [_]u32{ 5, 6, 5, 6, 5, 6 };
    var logits: [16]u16 = undefined;
    var rules: requests.Rules = .of(request, &prompt, 1);
    @memset(&logits, 0);
    rules.apply(&seen, prompt.len, &logits);
    try testing.expectEqual(@as(u16, 0), logits[1]);
    try testing.expectEqual(masks.minus_infinity, logits[2]);
    request.tail_repeat.min_count = 1;
    rules = .of(request, &prompt, 1);
    @memset(&logits, 0);
    rules.apply(&seen, prompt.len, &logits);
    try testing.expectEqual(masks.minus_infinity, logits[1]); // two pages owed: the guard bans the end
}

test "unlimited_ocr requests: the template and the host's expand give HF's input_ids, the page prompt written any way infer takes" {
    // Runs with ZSPARK_UNLIMITED_OCR_2 (the checkpoint's tokenizer) and ZSPARK_REFERENCE_UNLIMITED_OCR_2 set.
    const path = testing.environ.getPosix("ZSPARK_UNLIMITED_OCR_2") orelse return error.SkipZigTest;
    const reference = testing.environ.getPosix("ZSPARK_REFERENCE_UNLIMITED_OCR_2") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var problem: []const u8 = "";
    const text = host.hf_text.HfText.load(gpa, testing.io, path, null, arena, &problem) catch |err| {
        std.debug.print("{s}: {s}\n", .{ path, problem });
        return err;
    };
    defer text.deinit();
    const source = requests.template;
    text.source = source;
    text.compiled = try template.compile(gpa, source, null);
    var segments: segments_file.Segments = try .init(gpa, testing.io, text.text(), &text.tok, segments_file.default_bytes);
    defer segments.deinit();
    const fast = segments.text();
    try testing.expectEqual(preprocess.page_token_id, fast.tokenId("<PAGE>").?);
    var ref = try std.Io.Dir.cwd().openDir(testing.io, reference, .{});
    defer ref.close(testing.io);

    const Fixture = struct { input_ids: []u32, prompt: []const u8 };
    const cases = [_]struct { name: []const u8, pages: []const []const u8, texts: []const ?[]const u8 }{
        .{ .name = "one_page.json", .pages = &.{"pages/paper_p01.png"}, .texts = &.{ null, "Multi page parsing.", "<PAGE><image>Multi page parsing." } },
        .{ .name = "two_pages.json", .pages = &.{ "pages/paper_p01.png", "pages/paper_p03.png" }, .texts = &.{ null, "Multi page parsing.", "<PAGE><image><PAGE><image>Multi page parsing.", "<image>Multi page parsing." } },
    };
    for (cases) |c| {
        const raw = try ref.readFileAlloc(testing.io, c.name, arena, .limited(64 << 20));
        const doc = try std.json.parseFromSliceLeaky(Fixture, arena, try std.mem.replaceOwned(u8, arena, raw, "Infinity", "0"), .{ .ignore_unknown_fields = true });
        var parts: std.ArrayList(u8) = .empty;
        for (c.pages, 0..) |page, i| {
            const png = try ref.readFileAlloc(testing.io, page, arena, .limited(64 << 20));
            const b64 = try arena.alloc(u8, std.base64.standard.Encoder.calcSize(png.len));
            _ = std.base64.standard.Encoder.encode(b64, png);
            try parts.print(arena, "{s}{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:image/png;base64,{s}\"}}}}", .{ if (i > 0) "," else "", b64 });
        }
        for (c.texts) |asked| {
            const body = if (asked) |t|
                try std.fmt.allocPrint(arena, "[{{\"role\":\"user\",\"content\":[{s},{{\"type\":\"text\",\"text\":\"{s}\"}}]}}]", .{ parts.items, t })
            else
                try std.fmt.allocPrint(arena, "[{{\"role\":\"user\",\"content\":[{s}]}}]", .{parts.items});
            const messages = (try host.json.parse(arena, body)).ok;
            const rendered = fast.renderIds(arena, messages, .{ .add_generation_prompt = true }, &problem) catch |err| {
                std.debug.print("{s} {?s}: {s}\n", .{ c.name, asked, problem });
                return err;
            };
            var cx: host.errors.Cx = .{ .a = arena };
            const out = (host.images.expand(requests.Offer.vision(), null, fast, &cx, messages, rendered, 0) catch |err| {
                std.debug.print("{s} {?s}: refused: {s}\n", .{ c.name, asked, cx.message });
                return err;
            }).?;
            testing.expectEqualSlices(u32, doc.input_ids, out.ids) catch |err| {
                std.debug.print("{s} {?s}: the ids differ from HF's\n", .{ c.name, asked });
                return err;
            };
            try testing.expectEqual(c.pages.len, out.images.len);
        }
    }

    // a placeholder count `infer` refuses: two among one image, or one among three
    const refused = [_][]const u8{
        "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,\"}},{\"type\":\"text\",\"text\":\"<image><image>x\"}]}]",
        "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,\"}},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,\"}},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:,\"}},{\"type\":\"text\",\"text\":\"<image><image>x\"}]}]",
    };
    for (refused) |body| {
        const messages = (try host.json.parse(arena, body)).ok;
        if (fast.renderIds(arena, messages, .{ .add_generation_prompt = true }, &problem)) |_| return error.TestUnexpectedResult else |_| {}
        try testing.expect(std.mem.indexOf(u8, problem, "placeholders but") != null);
    }
    // a message with no image part: the client's `<PAGE>` and `<image>` are its text
    const plain = (try host.json.parse(arena, "[{\"role\":\"user\",\"content\":\"<PAGE><image>x\"}]")).ok;
    const ids = try fast.renderIds(arena, plain, .{ .add_generation_prompt = true }, &problem);
    try testing.expectEqual(@as(usize, 0), std.mem.count(u32, ids, &.{preprocess.page_token_id}));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u32, ids, &.{preprocess.image_token_id}));
}
