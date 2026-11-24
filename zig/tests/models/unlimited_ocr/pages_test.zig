//! Unlimited-OCR-2's page coordinates (`pages.zig`, spec 147 WS-05), host only: a prompt made by hand, and with
//! `$ZSPARK_REFERENCE_UNLIMITED_OCR_2` both cases of `tools/reference/unlimited_ocr/export_v2.py` (HF's own
//! `prefill.tok`, `prefill.page`, `prefill.query_page_ids`, `state.*` and every reply step's `step.tok`, `step.page`),
//! equal to the value.

const std = @import("std");
const testing = std.testing;
const pages = @import("../../../src/models/unlimited_ocr/pages.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");

const Coords = pages.Coords;
const block = 32;
const tokens: pages.Tokens = .{ .image = 128815, .page = 128827, .per_page = 273 };

test "unlimited_ocr pages: a prompt by hand" {
    // <PAGE> image x40, <PAGE> image x30, then text, the image tokens reaching into block 2
    const I = tokens.image;
    const P = tokens.page;
    var ids: [80]i32 = undefined;
    ids[0] = 1;
    ids[1] = P;
    @memset(ids[2..42], I);
    ids[42] = P;
    @memset(ids[43..73], I);
    for (ids[73..], 0..) |*id, i| id.* = 1000 + @as(i32, @intCast(i));
    var coords: [80]Coords = undefined;
    var blocks: [3]pages.Pages = undefined;
    var starts: [81]i32 = undefined;
    var track = try pages.plan(&ids, tokens, block, &coords, &blocks, &starts);
    // bucket 0: the row before the first marker; 1: the first page; 2: the second; after the last image the
    // instruction's rows go back to bucket max(2 - 2, 0) = 0
    try testing.expectEqual(Coords{ .tok = 0, .page = 0, .qp = -1 }, coords[0]);
    try testing.expectEqual(Coords{ .tok = 0, .page = 1, .qp = 0 }, coords[1]);
    try testing.expectEqual(Coords{ .tok = 40, .page = 1, .qp = 0 }, coords[41]);
    try testing.expectEqual(Coords{ .tok = 0, .page = 2, .qp = 1 }, coords[42]);
    try testing.expectEqual(Coords{ .tok = 30, .page = 2, .qp = 1 }, coords[72]);
    try testing.expectEqual(Coords{ .tok = 1, .page = 0, .qp = -1 }, coords[73]);
    try testing.expectEqual(Coords{ .tok = 7, .page = 0, .qp = -1 }, coords[79]);
    // block 0 holds page 0's image rows, block 1 both pages', block 2 page 1's up to row 72
    try testing.expectEqual(pages.Pages{ .least = 0, .most = 0 }, blocks[0]);
    try testing.expectEqual(pages.Pages{ .least = 0, .most = 1 }, blocks[1]);
    try testing.expectEqual(pages.Pages{ .least = 1, .most = 1 }, blocks[2]);
    try testing.expectEqualSlices(i32, &.{ 1, 41, 31 }, track.starts);
    try testing.expectEqual(@as(i32, 2), track.markers);
    try testing.expectEqual(@as(i32, 2), track.prompt_pages);
    // the reply: text, then a <PAGE> to the first page's bucket, then a third that stays in the last bucket
    try testing.expectEqual(Coords{ .tok = 8, .page = 0, .qp = -1 }, track.advance(5));
    try testing.expectEqual(Coords{ .tok = 41, .page = 1, .qp = 0 }, track.advance(P));
    try testing.expectEqual(Coords{ .tok = 42, .page = 1, .qp = 0 }, track.advance(5));
    try testing.expectEqual(Coords{ .tok = 31, .page = 2, .qp = 1 }, track.advance(P));
    try testing.expectEqual(Coords{ .tok = 31, .page = 2, .qp = 2 }, track.advance(P));
    // no image, or no marker up to the last image token
    try testing.expectError(error.Unpaged, pages.plan(ids[73..], tokens, block, coords[0..7], blocks[0..1], &starts));
    try testing.expectError(error.Unpaged, pages.plan(ids[2..42], tokens, block, coords[0..40], blocks[0..2], &starts));
}

test "unlimited_ocr pages: against HF's" {
    // Runs with ZSPARK_REFERENCE_UNLIMITED_OCR_2 set; skipped without it.
    const root = testing.environ.getPosix("ZSPARK_REFERENCE_UNLIMITED_OCR_2") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    for ([_][]const u8{ "one_page", "two_pages" }) |name| {
        var buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/{s}.safetensors", .{ root, name });
        var file: safetensors.File = try .open(gpa, testing.io, .cwd(), path);
        defer file.close();
        const of = struct {
            fn get(f: *const safetensors.File, key: []const u8) []const i32 {
                return @alignCast(std.mem.bytesAsSlice(i32, f.get(key).?.data));
            }
        };
        const ids = of.get(&file, "input_ids");
        const coords = try gpa.alloc(Coords, ids.len);
        defer gpa.free(coords);
        const blocks = try gpa.alloc(pages.Pages, (ids.len + block - 1) / block);
        defer gpa.free(blocks);
        const starts = try gpa.alloc(i32, ids.len + 1);
        defer gpa.free(starts);
        var track = try pages.plan(ids, tokens, block, coords, blocks, starts);
        for (coords, of.get(&file, "prefill.tok"), of.get(&file, "prefill.page"), of.get(&file, "prefill.query_page_ids"), 0..) |c, t, p, q, i| {
            errdefer std.debug.print("{s}: prompt row {d}: {any}, HF's ({d}, {d}, {d})\n", .{ name, i, c, t, p, q });
            try testing.expectEqual(Coords{ .tok = t, .page = p, .qp = q }, c);
        }
        for (blocks, of.get(&file, "state.bmin"), of.get(&file, "state.bmax")) |b, lo, hi| try testing.expectEqual(pages.Pages{ .least = lo, .most = hi }, b);
        try testing.expectEqualSlices(i32, of.get(&file, "state.pf_cnt"), track.starts);
        try testing.expectEqual(of.get(&file, "state.mcum")[0], track.markers);
        try testing.expectEqual(of.get(&file, "state.pp")[0], track.prompt_pages);
        for (of.get(&file, "step.token"), of.get(&file, "step.tok"), of.get(&file, "step.page"), 0..) |token, t, p, i| {
            const c = track.advance(token);
            errdefer std.debug.print("{s}: step {d}: {any}, HF's ({d}, {d})\n", .{ name, i, c, t, p });
            try testing.expectEqual(t, c.tok);
            try testing.expectEqual(p, c.page);
        }
    }
}
