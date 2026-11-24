//! `ngram.zig`'s tests: the expected ban sets are the checkpoint's own loop, ported in
//! `tools/reference/ngram_cases.py` (fixture `ngram_cases.bin`).

const std = @import("std");
const ngram = @import("../../src/decode/ngram.zig");
const masks = @import("../../src/decode/masks.zig");

const testing = std.testing;
const cases = @embedFile("fixtures/ngram_cases.bin");

const Reader = struct {
    at: usize = 8,

    fn int(self: *Reader) u32 {
        defer self.at += 4;
        return std.mem.readInt(u32, cases[self.at..][0..4], .little);
    }

    fn list(self: *Reader, gpa: std.mem.Allocator) ![]u32 {
        const out = try gpa.alloc(u32, self.int());
        for (out) |*v| v.* = self.int();
        return out;
    }
};

test "ngram: ban sets equal the checkpoint's loop on every fixture case" {
    const gpa = testing.allocator;
    try testing.expectEqualStrings("ZSNGRAM1", cases[0..8]);
    var reader: Reader = .{};
    const count = reader.int();
    var banning: usize = 0;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    var logits: [150_000]u16 = undefined;
    for (0..count) |case| {
        const size = reader.int();
        const window = reader.int();
        const whitelist = try reader.list(gpa);
        defer gpa.free(whitelist);
        const sequence = try reader.list(gpa);
        defer gpa.free(sequence);
        const expected = try reader.list(gpa);
        defer gpa.free(expected);
        const rule: ngram.Rule = .{ .size = size, .window = window, .whitelist = whitelist };

        out.clearRetainingCapacity();
        try ngram.banned(gpa, rule, sequence, &.{}, &out);
        testing.expectEqualSlices(u32, expected, out.items) catch |err| {
            std.debug.print("case {d}: n {d} window {d} length {d}\n", .{ case, size, window, sequence.len });
            return err;
        };
        if (expected.len > 0) banning += 1;

        // the sequence split anywhere into base and ahead is the same row
        const cut = sequence.len / 3;
        out.clearRetainingCapacity();
        try ngram.banned(gpa, rule, sequence[0..cut], sequence[cut..], &out);
        try testing.expectEqualSlices(u32, expected, out.items);

        // on the logits: -inf exactly where banned
        @memset(&logits, 0x3f80);
        _ = ngram.apply(rule, sequence, &.{}, &logits);
        var seen: usize = 0;
        for (logits, 0..) |bits, token| {
            if (bits == masks.minus_infinity) {
                try testing.expect(std.mem.indexOfScalar(u32, expected, @intCast(token)) != null);
                seen += 1;
            }
        }
        try testing.expectEqual(expected.len, seen);
    }
    try testing.expect(banning > 50);
}

fn build(gpa: std.mem.Allocator, phrase: []const u32, gap: usize) ![]u32 {
    // 50 tokens of other text, the phrase, `gap` tokens, the phrase's first 34 tokens
    var seq: std.ArrayList(u32) = .empty;
    errdefer seq.deinit(gpa);
    for (0..50) |i| try seq.append(gpa, @intCast(5000 + i));
    try seq.appendSlice(gpa, phrase);
    for (0..gap) |i| try seq.append(gpa, @intCast(9000 + i));
    try seq.appendSlice(gpa, phrase[0..34]);
    return seq.toOwnedSlice(gpa);
}

test "ngram: the 35/128 and 35/1024 settings ban a repeat only inside their window" {
    const gpa = testing.allocator;
    var phrase: [80]u32 = undefined;
    for (&phrase, 0..) |*t, i| t.* = @intCast(20_000 + i * 7);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    // the earlier 35-gram starts `gap + 80 + 0` tokens before the end minus 34... measured from the end: the
    // phrase's start is gap + 80 + 34 tokens back, so it is in a window of w iff gap + 114 <= w
    for ([_]struct { gap: usize, image: bool, page: bool }{
        .{ .gap = 14, .image = true, .page = true },
        .{ .gap = 15, .image = false, .page = true },
        .{ .gap = 100, .image = false, .page = true },
        .{ .gap = 910, .image = false, .page = true },
        .{ .gap = 911, .image = false, .page = false },
    }) |c| {
        const seq = try build(gpa, &phrase, c.gap);
        defer gpa.free(seq);
        out.clearRetainingCapacity();
        try ngram.banned(gpa, .image, seq, &.{}, &out);
        try testing.expectEqual(c.image, out.items.len == 1 and out.items[0] == phrase[34]);
        out.clearRetainingCapacity();
        try ngram.banned(gpa, .page, seq, &.{}, &out);
        try testing.expectEqual(c.page, out.items.len == 1 and out.items[0] == phrase[34]);
        // a whitelisted token is never banned
        out.clearRetainingCapacity();
        try ngram.banned(gpa, .{ .size = 35, .window = 1024, .whitelist = &.{phrase[34]} }, seq, &.{}, &out);
        try testing.expectEqual(@as(usize, 0), out.items.len);
    }
}

test "ngram: a row's set is its own sequence's, whatever the window's other rows hold" {
    const gpa = testing.allocator;
    const rule: ngram.Rule = .{ .size = 3, .window = 64 };
    // base ends 7 8; the sequence holds 7 8 9 and 7 8 4 earlier
    const base = [_]u32{ 1, 7, 8, 9, 2, 7, 8, 4, 3, 7, 8 };
    const drafts = [_]u32{ 9, 2, 7, 8, 5 };
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    // each row alone: base plus the drafts before it, joined by hand
    var joined: [16]u32 = undefined;
    @memcpy(joined[0..base.len], &base);
    var logits_window: [drafts.len + 1][12]u16 = undefined;
    for (&logits_window) |*row| @memset(row, 0);
    // the window's rows are filled in reverse order and with every draft present to the later ones
    var row: usize = drafts.len + 1;
    while (row > 0) {
        row -= 1;
        _ = ngram.apply(rule, &base, drafts[0..row], &logits_window[row]);
    }
    for (0..drafts.len + 1) |i| {
        // a serial step: the sequence as one slice
        @memcpy(joined[base.len..][0..i], drafts[0..i]);
        var serial: [12]u16 = @splat(0);
        _ = ngram.apply(rule, joined[0 .. base.len + i], &.{}, &serial);
        try testing.expectEqualSlices(u16, &serial, &logits_window[i]);
    }
    // row 0 bans 9 and 4; row 4 (base 9 2 7 8) bans the same and 5 is not in this vocabulary of 12... checked above
    out.clearRetainingCapacity();
    try ngram.banned(gpa, rule, &base, &.{}, &out);
    try testing.expectEqualSlices(u32, &.{ 4, 9 }, out.items);
    out.clearRetainingCapacity();
    try ngram.banned(gpa, rule, &base, drafts[0..1], &out);
    // the sequence now ends 8 9: no earlier "8 9" continues, 8 9 2 does
    try testing.expectEqualSlices(u32, &.{2}, out.items);
}

test "ngram: size 1 bans the window's tokens, a short sequence nothing" {
    const gpa = testing.allocator;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try ngram.banned(gpa, .{ .size = 1, .window = 3 }, &.{ 1, 2, 3, 4, 5 }, &.{}, &out);
    try testing.expectEqualSlices(u32, &.{ 3, 4, 5 }, out.items);
    out.clearRetainingCapacity();
    try ngram.banned(gpa, .image, &.{ 1, 2, 3 }, &.{}, &out);
    try testing.expectEqual(@as(usize, 0), out.items.len);
}
