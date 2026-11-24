//! `page_guard.zig`'s tests: each branch of the checkpoint's `_uocr_page_count_guard`, the incremental scan against
//! a scan of the whole reply at every length, and, where `ZSPARK_REFERENCE_UNLIMITED_OCR_2` names spec 147's
//! fixtures, HF's greedy replies passed through the guard and v2's n-gram rule (the reply's tokens alone) step by
//! step: no chosen token is banned, and the end is allowed only once every page is written.

const std = @import("std");
const testing = std.testing;
const page_guard = @import("../../src/decode/page_guard.zig");
const ngram = @import("../../src/decode/ngram.zig");
const masks = @import("../../src/decode/masks.zig");

const Guard = page_guard.Guard;
const ids = struct {
    const page: u32 = 128827;
    const eos: u32 = 1;
};

fn guardOf(expected: u32, cap: u32) Guard {
    return .{ .expected = expected, .cap = cap, .page = ids.page, .eos = ids.eos };
}

/// HF's rule over the whole reply, scanned at once.
fn reference(expected: u32, cap: u32, reply: []const u32) Guard.Action {
    if (expected == 0) return .none;
    const marks: u32 = @intCast(std.mem.count(u32, reply, &.{ids.page}));
    const page_len = if (std.mem.lastIndexOfScalar(u32, reply, ids.page)) |last| reply.len - 1 - last else reply.len;
    if (marks > expected) return .{ .force = ids.eos };
    if (cap > 0 and page_len >= cap) return .{ .force = if (marks < expected) ids.page else ids.eos };
    if (marks < expected) return .{ .ban = ids.eos };
    return .none;
}

test "page guard: each branch of HF's guard" {
    const P = ids.page;
    const Case = struct { expected: u32, cap: u32, reply: []const u32, want: Guard.Action };
    const cases = [_]Case{
        // no marker in the prompt: nothing, whatever the reply
        .{ .expected = 0, .cap = 2, .reply = &.{ P, P, 5, 5, 5 }, .want = .none },
        // pages owed: the end banned, before the first marker too
        .{ .expected = 2, .cap = 0, .reply = &.{}, .want = .{ .ban = ids.eos } },
        .{ .expected = 2, .cap = 0, .reply = &.{ P, 7, 8 }, .want = .{ .ban = ids.eos } },
        // every page written: the row as it stands, the last page under the cap
        .{ .expected = 2, .cap = 4, .reply = &.{ P, 7, P, 8, 9, 9 }, .want = .none },
        // a marker too many: the end alone
        .{ .expected = 1, .cap = 0, .reply = &.{ P, 7, P }, .want = .{ .force = ids.eos } },
        // a page at the cap: the next marker while one is owed, else the end
        .{ .expected = 2, .cap = 3, .reply = &.{ P, 7, 8, 9 }, .want = .{ .force = P } },
        .{ .expected = 2, .cap = 3, .reply = &.{ P, 7, P, 8, 9, 9 }, .want = .{ .force = ids.eos } },
        // tokens before the first marker count toward its page
        .{ .expected = 1, .cap = 3, .reply = &.{ 7, 8, 9 }, .want = .{ .force = P } },
        .{ .expected = 1, .cap = 3, .reply = &.{ 7, 8 }, .want = .{ .ban = ids.eos } },
    };
    for (cases) |c| {
        var g = guardOf(c.expected, c.cap);
        try testing.expectEqual(c.want, g.verdict(c.reply));
        try testing.expectEqual(c.want, reference(c.expected, c.cap, c.reply));
    }
    try testing.expectEqual(@as(u32, 2), page_guard.pagesIn(&[_]i32{ 0, @intCast(P), 128815, @intCast(P), 5 }, P));
}

test "page guard: the logits it leaves, a banned token forced kept at 0" {
    var logits: [8]u16 = .{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100 };
    // ban: the end token alone at -inf
    var g: Guard = .{ .expected = 2, .cap = 0, .page = 6, .eos = 1 };
    try testing.expectEqual(Guard.Action{ .ban = 1 }, g.apply(&.{6}, &logits));
    try testing.expectEqual(masks.minus_infinity, logits[1]);
    try testing.expectEqual(@as(u16, 0x3f80), logits[0]);
    // force: every other token at -inf, the forced one's logit kept
    var f: Guard = .{ .expected = 2, .cap = 2, .page = 6, .eos = 1 };
    try testing.expectEqual(Guard.Action{ .force = 6 }, f.apply(&.{ 6, 3, 3 }, &logits));
    for (logits, 0..) |v, i| try testing.expectEqual(if (i == 6) @as(u16, 0x40e0) else masks.minus_infinity, v);
    // the end token is -inf here (as if the n-gram rule had banned it): forced, it is kept at 0, the row's largest
    try testing.expectEqual(masks.minus_infinity, logits[1]);
    var h: Guard = .{ .expected = 1, .cap = 0, .page = 6, .eos = 1 };
    try testing.expectEqual(Guard.Action{ .force = 1 }, h.apply(&.{ 6, 6 }, &logits));
    for (logits, 0..) |v, i| try testing.expectEqual(if (i == 1) @as(u16, 0) else masks.minus_infinity, v);
}

test "page guard: the incremental scan equals the whole reply's at every length, steps of any size, a shorter reply rescanned" {
    var prng: std.Random.DefaultPrng = .init(147);
    const r = prng.random();
    var reply: [400]u32 = undefined;
    for (0..200) |_| {
        for (&reply) |*t| t.* = if (r.uintLessThan(u32, 9) == 0) ids.page else 2 + r.uintLessThan(u32, 50);
        const expected = r.uintLessThan(u32, 12);
        const cap = r.uintLessThan(u32, 3) * r.uintLessThan(u32, 40);
        var g = guardOf(expected, cap);
        var at: usize = 0;
        while (at < reply.len) {
            try testing.expectEqual(reference(expected, cap, reply[0..at]), g.verdict(reply[0..at]));
            // mostly a step of one, as a reply grows; sometimes several, or back a few tokens to a shorter reply
            const move = r.uintLessThan(usize, 20);
            at = if (move == 0) at -| (1 + r.uintLessThan(usize, 8)) else if (move == 1) at + 1 + r.uintLessThan(usize, 20) else at + 1;
        }
    }
}

test "page guard: HF's greedy replies on spec 147's fixtures pass the guard and v2's n-gram rule at every step" {
    // Runs with ZSPARK_REFERENCE_UNLIMITED_OCR_2 set; skipped without it.
    const path = testing.environ.getPosix("ZSPARK_REFERENCE_UNLIMITED_OCR_2") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var dir = try std.Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    const Fixture = struct { input_ids: []u32, tokens: []u32, expected_pages: u32, finish_reason: []const u8, provenance: struct { ngram: []const u8, window: []const u8 } };
    var logits: [129280]u16 = undefined;
    for ([_][]const u8{ "one_page.json", "two_pages.json" }) |name| {
        const raw = try dir.readFileAlloc(testing.io, name, gpa, .limited(64 << 20));
        defer gpa.free(raw);
        // Python's json writes an infinite near-tie gap as `Infinity`, which JSON has not; those fields are not read
        const text = try std.mem.replaceOwned(u8, gpa, raw, "Infinity", "0");
        defer gpa.free(text);
        const doc = try std.json.parseFromSlice(Fixture, gpa, text, .{ .ignore_unknown_fields = true });
        defer doc.deinit();
        const f = doc.value;
        const rule: ngram.Rule = .{ .size = try std.fmt.parseInt(u32, f.provenance.ngram, 10), .window = try std.fmt.parseInt(u32, f.provenance.window, 10) };
        var g: Guard = .init(f.input_ids, ids.page, ids.eos, Guard.default_cap);
        try testing.expectEqual(f.expected_pages, g.expected);
        try testing.expectEqualStrings("stop", f.finish_reason);
        // each token, then the end after the last: chosen from a row the rules left it in
        for (0..f.tokens.len + 1) |step| {
            const chosen = if (step < f.tokens.len) f.tokens[step] else ids.eos;
            @memset(&logits, 0);
            _ = ngram.apply(rule, f.tokens[0..step], &.{}, &logits);
            _ = g.apply(f.tokens[0..step], &logits);
            testing.expect(logits[chosen] != masks.minus_infinity) catch |e| {
                std.debug.print("{s}: step {d} bans {d}\n", .{ name, step, chosen });
                return e;
            };
        }
        try testing.expectEqual(f.expected_pages, g.marks);
    }
}
