//! Unlimited-OCR-2's long documents on the device (`parts.zig`, spec 147 WS-08), on `forward_dash_test.zig`'s load
//! (its one test calls `document`): the `two_pages` fixture's pages twice, a document of four (HF's image rows from
//! the fixture's embeddings, the prompt's ids around them), greedy with `serve`'s rules:
//!
//! - one part of the four pages against the same reply run by hand (eager prompt windows, steps without graphs):
//!   the same tokens, every one: one window is the request as it stands;
//! - parts of two pages (`documents.plan` with `max_batch_pages` 2): the second takes the first's tail over (its
//!   keys, values, index keys and text), and the reply ends by a stop with the four pages' markers; its tokens
//!   against the one part's are printed (SGLang's approximation: near, not equal);
//! - three pages in parts of 2: the last part, one page, takes over that page less its last image row (`straddle`);
//! - the state's bytes a row: two states' device bytes apart.

const std = @import("std");
const testing = std.testing;
const api = @import("engine_api");
const cuda = @import("../../../src/cuda/cuda.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const documents = @import("../../../src/models/unlimited_ocr/documents.zig");
const parts = @import("../../../src/models/unlimited_ocr/parts.zig");
const requests = @import("../../../src/models/unlimited_ocr/requests.zig");
const forward = @import("../../../src/models/unlimited_ocr/forward.zig");
const model_file = @import("../../../src/models/unlimited_ocr/model.zig");
const ocr_testing = @import("testing.zig");

const page_token = 128827;
const image_token = 128815;
const eos = 1;
const max_tokens = 6000;

/// Greedy, every token kept.
const Greedy = struct {
    out: std.ArrayList(u32) = .empty,

    fn sink(self: *Greedy) parts.Sink {
        return .{ .ctx = self, .choose = choose, .give = give, .cancelled = never, .prefilled = nothing };
    }
    fn choose(_: *anyopaque, logits: []u16, _: usize) anyerror!i32 {
        return sampling.argmax(logits);
    }
    fn give(ctx: *anyopaque, token: i32) bool {
        const self: *Greedy = @ptrCast(@alignCast(ctx));
        self.out.append(testing.allocator, @intCast(token)) catch return true;
        return false;
    }
    fn never(_: *anyopaque) bool {
        return false;
    }
    fn nothing(_: *anyopaque) void {}
};

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

pub fn document(m: model_file.Model) !void {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c = m.config;

    // the fixture's two pages twice
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/two_pages.safetensors", .{env("ZSPARK_REFERENCE_UNLIMITED_OCR_2").?});
    var file: safetensors.File = try .open(gpa, testing.io, .cwd(), path);
    defer file.close();
    const ids: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, file.get("input_ids").?.data));
    const e = file.get("prefill.embeds").?;
    var embeds: cuda.Buffer = try .init(.bf16, &.{ @intCast(e.shape[0]), @intCast(e.shape[1]) });
    defer embeds.deinit();
    try embeds.upload(e.data);
    const p = (try documents.parse(arena, ids, page_token, image_token)).?;
    try testing.expectEqual(@as(usize, 2), p.pages());
    const order = [_]usize{ 0, 1, 0, 1 };
    var doc_ids: std.ArrayList(i32) = .empty;
    try doc_ids.appendSlice(arena, ids[0..p.prefix]);
    const rows = try arena.alloc(usize, order.len);
    var image_rows: usize = 0;
    for (order, rows) |j, *r| {
        try doc_ids.appendSlice(arena, ids[p.starts[j]..p.ends[j]]);
        r.* = p.ends[j] - p.starts[j] - 1;
        image_rows += r.*;
    }
    try doc_ids.appendSlice(arena, ids[p.instruction..]);
    var features: cuda.Buffer = try .init(.bf16, &.{ image_rows, c.hidden });
    defer features.deinit();
    var at: usize = 0;
    for (order, rows) |j, r| {
        try features.copyRows(at, embeds, p.starts[j] + 1, r);
        at += r;
    }
    const doc = (try parts.Document.of(arena, doc_ids.items, features, rows)).?;
    const n = doc_ids.items.len;

    // the state's bytes a row
    const capacity = n + max_tokens + 64;
    const before = cuda.account.bytes;
    var state: forward.State = try .init(gpa, c, capacity);
    defer state.deinit();
    const one = cuda.account.bytes - before;
    {
        var more: forward.State = try .init(gpa, c, capacity + 32768);
        defer more.deinit();
        const row = @as(f64, @floatFromInt(cuda.account.bytes - before - 2 * one)) / 32768.0;
        std.debug.print("unlimited_ocr parts: the state {d} bytes a row ({d:.1} MB a 2,033-row gundam page, {d:.2} GB at 131,072 rows)\n", .{ row, row * 2033 / 1e6, row * 131072 / 1e9 });
    }
    var b: forward.Buffers = try .init(c, 1024, 1);
    defer b.deinit();
    const logits = try gpa.alloc(u16, c.vocab);
    defer gpa.free(logits);
    const request: api.Request = .{ .prompt = &.{}, .max_tokens = 0, .eos = &.{eos} };
    const run: parts.Run = .{ .model = m, .state = &state, .buffers = &b, .logits = logits };

    // by hand: the prompt in eager windows, steps without graphs
    var hand: std.ArrayList(u32) = .empty;
    defer hand.deinit(gpa);
    {
        var rules: requests.Rules = .of(request, doc_ids.items, eos);
        var runs: [order.len]forward.Image.Run = undefined;
        for (&runs, 0..) |*r, j| r.* = .{ .first = doc.prompt.starts[j] + 1, .rows = rows[j] };
        state.reset();
        state.image = .{ .features = features, .runs = &runs };
        try state.paged(doc_ids.items);
        while (state.pos < n) {
            const w = @min(b.rows, n - state.pos);
            const window = doc_ids.items[state.pos..][0..w];
            try forward.forwardScoring(m, &state, &b, window, if (state.pos + w == n) .last else .none);
            try forward.commit(m, &state, b, window, w);
        }
        var seen: std.ArrayList(u32) = .empty;
        defer seen.deinit(gpa);
        for (doc_ids.items) |id| try seen.append(gpa, @intCast(id));
        while (hand.items.len < max_tokens) {
            try forward.logitsRow(b, 0, logits);
            rules.apply(seen.items, n, logits);
            const token: u32 = @intCast(try sampling.argmax(logits));
            if (token == eos) break;
            try hand.append(gpa, token);
            try seen.append(gpa, token);
            const step = [_]i32{@intCast(token)};
            try forward.forwardScoring(m, &state, &b, &step, .last);
            try forward.commit(m, &state, b, &step, 1);
        }
    }

    // one part
    var whole: Greedy = .{};
    defer whole.out.deinit(gpa);
    const t0 = cuda.monotonic();
    const one_part = [_]documents.Part{.{ .lo = 0, .hi = 4 }};
    const a = try parts.reply(gpa, arena, run, doc, &one_part, request, max_tokens, whole.sink());
    const t1 = cuda.monotonic();
    try testing.expectEqual(parts.Finish.stop, a.finish);
    try testing.expectEqual(@as(usize, 0), a.reused);
    try testing.expectEqualSlices(u32, hand.items, whole.out.items);

    // parts of two pages
    var split: Greedy = .{};
    defer split.out.deinit(gpa);
    const planned = (try documents.plan(arena, 4, 274, capacity, max_tokens, 2)).?;
    try testing.expectEqual(@as(usize, 2), planned.len);
    const t2 = cuda.monotonic();
    const w = try parts.reply(gpa, arena, run, doc, planned, request, max_tokens, split.sink());
    const t3 = cuda.monotonic();
    const marks = std.mem.count(u32, split.out.items, &.{page_token});
    const distance = try ocr_testing.editDistance(gpa, split.out.items, whole.out.items);
    std.debug.print("unlimited_ocr parts: 4 pages ({d} prompt rows) one part {d} tokens in {d:.1} s = by hand; parts of 2: {d} parts, {d} taken over, {t}, {d} tokens, {d} <PAGE>s (one part {d}) in {d:.1} s, edit distance {d}\n", .{
        n,                                      whole.out.items.len, @as(f64, @floatFromInt(t1 - t0)) / 1e9, w.parts, w.reused, w.finish, split.out.items.len, marks, std.mem.count(u32, whole.out.items, &.{page_token}),
        @as(f64, @floatFromInt(t3 - t2)) / 1e9, distance,
    });
    try testing.expectEqual(@as(usize, 2), w.parts);
    try testing.expectEqual(@as(usize, 1), w.reused);
    try testing.expectEqual(parts.Finish.stop, w.finish);
    try testing.expect(distance * 5 < whole.out.items.len);

    // three pages in parts of 2: the last part of one page takes over its page less its last image row (`straddle`),
    // which its prompt reads
    var three_ids: std.ArrayList(i32) = .empty;
    try three_ids.appendSlice(arena, doc_ids.items[0..doc.prompt.ends[2]]);
    try three_ids.appendSlice(arena, doc_ids.items[doc.prompt.instruction..]);
    const three_rows = rows[0..3];
    const three_features: cuda.Buffer = .{ .view = features.span(0, three_rows[0] + three_rows[1] + three_rows[2]) };
    const three = (try parts.Document.of(arena, three_ids.items, three_features, three_rows)).?;
    const three_parts = (try documents.plan(arena, 3, 274, capacity, max_tokens, 2)).?;
    try testing.expectEqual(@as(usize, 2), three_parts.len);
    const last_ids = try documents.partIds(arena, three.ids, three.prompt, three_parts[1], 0);
    const first_ids = try documents.partIds(arena, three.ids, three.prompt, three_parts[0], 1);
    const tail = (try documents.tailCopy(arena, first_ids, last_ids, page_token, image_token, 32, 1, true)).?;
    try testing.expect(tail.straddle);
    var straddled: Greedy = .{};
    defer straddled.out.deinit(gpa);
    const s = try parts.reply(gpa, arena, run, three, three_parts, request, max_tokens, straddled.sink());
    const s_distance = try ocr_testing.editDistance(gpa, straddled.out.items, whole.out.items[0..@min(whole.out.items.len, straddled.out.items.len)]);
    std.debug.print("unlimited_ocr parts: 3 pages in parts of 2 (the last a straddled page): {d} taken over, {t}, {d} tokens, {d} <PAGE>s, edit distance {d} to the 4 pages' reply's start\n", .{ s.reused, s.finish, straddled.out.items.len, std.mem.count(u32, straddled.out.items, &.{page_token}), s_distance });
    try testing.expectEqual(@as(usize, 1), s.reused);
    try testing.expectEqual(parts.Finish.stop, s.finish);
    try testing.expectEqual(@as(usize, 3), std.mem.count(u32, straddled.out.items, &.{page_token}));
}
