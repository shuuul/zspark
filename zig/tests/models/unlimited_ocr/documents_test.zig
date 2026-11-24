//! Unlimited-OCR-2's long documents (`documents.zig`, spec 147 WS-08), host only: each port against the value
//! SGLang's v2 wheel gives for the same arguments (`doc_split.py`, `session/uocr_doc_reuse.py`, read 2026-10-28;
//! the values worked from its code, the one case where zspark departs said so).

const std = @import("std");
const testing = std.testing;
const documents = @import("../../../src/models/unlimited_ocr/documents.zig");

const Part = documents.Part;
const P: i32 = 128827;
const I: i32 = 128815;

fn expectPlan(expected: ?[]const Part, n: u32, mode: []const u8, context: u64, max_tokens: u64, max_pages: ?u32) !void {
    const got = try documents.plan(testing.allocator, n, documents.tokensPerPage(mode), context, max_tokens, max_pages);
    defer if (got) |g| testing.allocator.free(g);
    if (expected) |e| {
        try testing.expect(got != null);
        try testing.expectEqualSlices(Part, e, got.?);
    } else try testing.expect(got == null);
}

test "unlimited_ocr documents: plan_batches" {
    // one page; a budget of none
    try expectPlan(null, 1, "", 131072, 1000, null);
    try expectPlan(null, 10, "base", 131072, 0, 2);
    // 30 base pages fit (131,007 // 274 = 478)
    try expectPlan(null, 30, "base", 131072, 1000, null);
    // a page limit cuts them, whatever the mode (gundam fits 86)
    const tens = [_]Part{ .{ .lo = 0, .hi = 10 }, .{ .lo = 10, .hi = 20 }, .{ .lo = 20, .hi = 30 } };
    try expectPlan(&tens, 30, "base", 131072, 1000, 10);
    try expectPlan(&tens, 30, "", 131072, 1000, 10);
    try expectPlan(&.{ .{ .lo = 0, .hi = 10 }, .{ .lo = 10, .hi = 20 }, .{ .lo = 20, .hi = 25 } }, 25, "base", 131072, 1000, 10);
    try expectPlan(&.{ .{ .lo = 0, .hi = 2 }, .{ .lo = 2, .hi = 4 } }, 4, "base", 131072, 50, 2);
    try expectPlan(&.{ .{ .lo = 0, .hi = 2 }, .{ .lo = 2, .hi = 4 }, .{ .lo = 4, .hi = 5 } }, 5, "base", 131072, 50, 2);
    try expectPlan(&.{ .{ .lo = 0, .hi = 1 }, .{ .lo = 1, .hi = 2 }, .{ .lo = 2, .hi = 3 } }, 3, "base", 131072, 10, 1);
    // the context alone cuts them
    try expectPlan(&.{ .{ .lo = 0, .hi = 478 }, .{ .lo = 478, .hi = 600 } }, 600, "base", 131072, 1000, null);
    try expectPlan(&.{ .{ .lo = 0, .hi = 86 }, .{ .lo = 86, .hi = 100 } }, 100, "gundam", 131072, 1000, null);
    // SGLang's [(0, 2, 0)] (every page fits, the budget clamped) is the request as it stands here
    try expectPlan(null, 2, "base", 1000, 900, null);
    try testing.expectEqual(@as(u32, 1514), documents.tokensPerPage("auto"));
}

test "unlimited_ocr documents: lookahead, output cap, page base, own pages" {
    const parts = [_]Part{ .{ .lo = 0, .hi = 10 }, .{ .lo = 10, .hi = 20 }, .{ .lo = 20, .hi = 25 } };
    try testing.expectEqual(@as(u32, 1), documents.lookahead(&parts, 0));
    try testing.expectEqual(@as(u32, 1), documents.lookahead(&parts, 1));
    try testing.expectEqual(@as(u32, 0), documents.lookahead(&parts, 2));
    try testing.expectEqual(@as(?u64, 200256), documents.outputCap(10, 20000));
    try testing.expectEqual(@as(?u64, null), documents.outputCap(3, 0));
    // a part that reads its pages numbers them from lo + 1; a reused one's page 1 is the page before lo
    try testing.expectEqual(@as(i32, 10), documents.pageBase(parts[1], false));
    try testing.expectEqual(@as(i32, 9), documents.pageBase(parts[1], true));
    // the guard's expected pages: a reused part's prompt holds the context page and 10 own and a lookahead
    try testing.expectEqual(@as(u32, 10), documents.ownPages(12, true, 1));
    try testing.expectEqual(@as(u32, 10), documents.ownPages(11, false, 1));
    try testing.expectEqual(@as(u32, 5), documents.ownPages(5, false, 0));
}

/// BOS, then `rows.len` pages of `<PAGE>` and `rows[i]` image ids, then a three-id instruction.
fn doc(a: std.mem.Allocator, rows: []const usize) ![]i32 {
    var ids: std.ArrayList(i32) = .empty;
    try ids.append(a, 0);
    for (rows) |r| {
        try ids.append(a, P);
        try ids.appendNTimes(a, I, r);
    }
    try ids.appendSlice(a, &.{ 5, 6, 7 });
    return ids.toOwnedSlice(a);
}

test "unlimited_ocr documents: parse_prompt and the part's ids" {
    const a = testing.allocator;
    const ids = try doc(a, &.{ 273, 273, 273 });
    defer a.free(ids);
    const p = (try documents.parse(a, ids, P, I)).?;
    defer p.deinit(a);
    try testing.expectEqual(@as(usize, 1), p.prefix);
    try testing.expectEqualSlices(usize, &.{ 1, 275, 549 }, p.starts);
    try testing.expectEqualSlices(usize, &.{ 275, 549, 823 }, p.ends);
    try testing.expectEqual(@as(usize, 823), p.instruction);
    // pages 1 and 2: BOS, their spans, the instruction (`split_messages` and `_retag_markers`)
    const part = try documents.partIds(a, ids, p, .{ .lo = 1, .hi = 2 }, 1);
    defer a.free(part);
    const expected = try doc(a, &.{ 273, 273 });
    defer a.free(expected);
    try testing.expectEqualSlices(i32, expected, part);
    // an image before the first marker, a marker without images, an image in the instruction
    try testing.expect(try documents.parse(a, &.{ 0, I, P, I }, P, I) == null);
    try testing.expect(try documents.parse(a, &.{ 0, P, P, I }, P, I) == null);
    try testing.expect(try documents.parse(a, &.{ 0, P, I, 5, I }, P, I) == null);
    try testing.expect(try documents.parse(a, &.{ 0, 5 }, P, I) == null);
}

test "unlimited_ocr documents: plan_tail_copy" {
    const a = testing.allocator;
    // two own pages and a lookahead page, then the next part: the lookahead page and one more
    const old = try doc(a, &.{ 273, 273, 273 });
    defer a.free(old);
    const new = try doc(a, &.{ 273, 273 });
    defer a.free(new);
    {
        // end 549, start 549 - 128 - 549 % 32 = 416; the lookahead page (549 .. 822) copied whole
        const t = (try documents.tailCopy(a, old, new, P, I, 32, 1, true)).?;
        defer t.deinit(a);
        try testing.expectEqual(@as(usize, 275), t.marker);
        try testing.expectEqual(@as(usize, 416), t.start);
        try testing.expectEqual(@as(usize, 823), t.copy_end);
        try testing.expectEqual(@as(usize, 408), t.copy_len);
        try testing.expectEqual(@as(usize, 275), t.kept_from);
        try testing.expectEqual(@as(usize, 2), t.old_pages);
        try testing.expectEqual(@as(usize, 2), t.new_pages);
        try testing.expectEqual(@as(usize, 1), t.copied_pages);
        try testing.expect(!t.straddle);
        try testing.expectEqual(@as(usize, 682), t.ids.len);
        try testing.expectEqualSlices(i32, &.{ P, I, I }, t.ids[0..3]);
        try testing.expectEqualSlices(i32, old[416..823], t.ids[1..408]);
        try testing.expectEqualSlices(i32, new[275..549], t.ids[408..]);
    }
    {
        // a one-page part leaves its copied page's last image row to its prompt
        const new1 = try doc(a, &.{273});
        defer a.free(new1);
        const t = (try documents.tailCopy(a, old, new1, P, I, 32, 1, true)).?;
        defer t.deinit(a);
        try testing.expectEqual(@as(usize, 822), t.copy_end);
        try testing.expectEqual(@as(usize, 407), t.copy_len);
        try testing.expectEqual(@as(usize, 274), t.kept_from);
        try testing.expectEqual(@as(usize, 1), t.copied_pages);
        try testing.expect(t.straddle);
        try testing.expectEqual(@as(usize, 408), t.ids.len);
    }
    {
        // no lookahead: the tail alone, the new part's pages after it
        const old2 = try doc(a, &.{ 273, 273 });
        defer a.free(old2);
        const t = (try documents.tailCopy(a, old2, new, P, I, 32, 0, true)).?;
        defer t.deinit(a);
        try testing.expectEqual(@as(usize, 549), t.copy_end);
        try testing.expectEqual(@as(usize, 134), t.copy_len);
        try testing.expectEqual(@as(usize, 1), t.kept_from);
        try testing.expectEqual(@as(usize, 0), t.copied_pages);
        try testing.expectEqual(@as(usize, 682), t.ids.len);
    }
    {
        // a lookahead page that is not the new part's first: not copied
        const old3 = try doc(a, &.{ 273, 40 });
        defer a.free(old3);
        const t = (try documents.tailCopy(a, old3, new, P, I, 32, 1, true)).?;
        defer t.deinit(a);
        try testing.expectEqual(@as(usize, 128), t.start);
        try testing.expectEqual(@as(usize, 275), t.copy_end);
        try testing.expectEqual(@as(usize, 0), t.copied_pages);
    }
    {
        // a last own page shorter than the tail
        const old4 = try doc(a, &.{ 100, 273 });
        defer a.free(old4);
        try testing.expect(try documents.tailCopy(a, old4, new, P, I, 32, 1, true) == null);
    }
}

test "unlimited_ocr documents: text_tail and the extra marker" {
    const p: u32 = @intCast(P);
    const out = [_]u32{ 11, 12, p, 13, 14, 15, p, 16, 17 };
    const ones: [300]u32 = @splat(1);
    try testing.expectEqual(@as(?usize, 7), documents.textTail(&out, 128, p));
    try testing.expectEqual(@as(?usize, 7), documents.textTail(&out, 2, p));
    try testing.expectEqual(@as(?usize, 172), documents.textTail(&ones, 128, p));
    try testing.expectEqual(@as(?usize, null), documents.textTail(&.{}, 128, p));
    try testing.expectEqualSlices(u32, out[0..6], documents.cutAtExtraMarker(&out, p, 1));
    try testing.expectEqualSlices(u32, &out, documents.cutAtExtraMarker(&out, p, 2));
}
