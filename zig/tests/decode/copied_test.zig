//! `copied.zig`'s tests on the host: the drafter around a made-up inner drafter, handed tokens as `Decoder` hands
//! them (the rows kept, the last at `position`), and Recall's record asked through it (`recall.zig`).

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../src/cuda/cuda.zig");
const draft = @import("../../src/decode/draft.zig");
const masks_file = @import("../../src/decode/masks.zig");
const copied = @import("../../src/decode/copied.zig");
const recall = @import("../../src/decode/recall.zig");

/// An inner drafter that drafts `chain` whatever it is asked, as many as there is room for.
const Fixed = struct {
    chain: []const i32 = &.{},
    asked: usize = 0,

    fn drafter(self: *Fixed) draft.Drafter {
        return .{ .context = self, .draft = draftFn };
    }

    fn draftFn(context: *anyopaque, _: cuda.View, _: []const i32, _: usize, _: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Fixed = @ptrCast(@alignCast(context));
        self.asked += 1;
        const n = @min(self.chain.len, out.len);
        @memcpy(out[0..n], self.chain[0..n]);
        return n;
    }
};

const no_streams: cuda.View = undefined;

/// One call as the decoder makes it after a window that kept `tokens`, the last of them at `position`.
fn ask(drafter: draft.Drafter, tokens: []const i32, position: usize, out: []i32) ![]const i32 {
    return out[0..try drafter.draft(drafter.context, no_streams, tokens, position, null, out)];
}

// a prompt that holds 10..29, and a reply that has begun to repeat it: 10 11 12 13 14 15 16
const prompt = [_]i32{ 1, 2, 3 } ++ blk: {
    var run: [20]i32 = undefined;
    for (&run, 0..) |*token, i| token.* = 10 + @as(i32, @intCast(i));
    break :blk run;
} ++ [_]i32{ 4, 5, 6 };
const begun = [_]i32{ 10, 11, 12, 13, 14, 15, 16 };

test "what followed the earlier occurrence is the window where the inner drafter's first draft agrees, and the drafter's own chain where it does not" {
    var inner: Fixed = .{};
    var copy: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer copy.deinit();
    try copy.begin(&prompt);
    var out: [15]i32 = undefined;

    // agreed: 17 is what followed 10..16 in the prompt, and the copy runs on to the prompt's own end
    inner.chain = &.{ 17, 99, 98 };
    const position = prompt.len + begun.len - 1;
    try testing.expectEqualSlices(i32, &.{ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 4, 5 }, try ask(copy.drafter(), &begun, position, &out));
    try testing.expectEqual(1, copy.stats.offered);
    try testing.expectEqual(15, copy.stats.rows);

    // the window kept 13 rows: 12 copied rows were the model's choice
    inner.chain = &.{ 77, 78 };
    const kept = [_]i32{ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 40 };
    // not agreed (nothing follows 28 40 in the context at all): the drafter's chain, untouched
    try testing.expectEqualSlices(i32, &.{ 77, 78 }, try ask(copy.drafter(), &kept, position + kept.len, &out));
    try testing.expectEqual(12, copy.stats.kept);
    try testing.expectEqual(1, copy.stats.offered);

    // a match the drafter does not agree with is found and not offered
    var again: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer again.deinit();
    try again.begin(&prompt);
    inner.chain = &.{ 55, 56 };
    try testing.expectEqualSlices(i32, &.{ 55, 56 }, try ask(again.drafter(), &begun, position, &out));
    try testing.expectEqual(1, again.stats.found);
    try testing.expectEqual(0, again.stats.offered);
    // and with no room for a draft the inner drafter is still asked (it absorbs the rows), and nothing is offered
    const asked = inner.asked;
    try testing.expectEqual(0, (try ask(again.drafter(), &.{16}, position, out[0..0])).len);
    try testing.expectEqual(asked + 1, inner.asked);
}

test "a copy ends at a token that ends a reply, and never crosses a position whose token is unknown" {
    var inner: Fixed = .{ .chain = &.{17} };
    var copy: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{20});
    defer copy.deinit();
    try copy.begin(&prompt);
    var out: [15]i32 = undefined;
    try testing.expectEqualSlices(i32, &.{ 17, 18, 19, 20 }, try ask(copy.drafter(), &begun, prompt.len + begun.len - 1, &out));

    // a state held from an earlier request: the decoder shows the rows after it alone, and `begin` was not told
    // the prompt, so the positions before are unknown and nothing is copied from them
    var held: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer held.deinit();
    try held.begin(&.{});
    try testing.expectEqualSlices(i32, &.{17}, try ask(held.drafter(), &begun, prompt.len + begun.len - 1, &out));
    try testing.expectEqual(0, held.stats.found);
    try testing.expectEqual(prompt.len + begun.len, held.context.items.len);

    // told the prompt's first 15 tokens alone (1 2 3, 10..21): the copy stops where the unknown positions begin
    var part: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer part.deinit();
    try part.begin(prompt[0..15]);
    try testing.expectEqualSlices(i32, &.{ 17, 18, 19, 20, 21 }, try ask(part.drafter(), &begun, prompt.len + begun.len - 1, &out));
}

test "the context follows the decoder: a prompt's windows shown again change nothing, and a new reply forgets the last" {
    var inner: Fixed = .{ .chain = &.{17} };
    var copy: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer copy.deinit();
    try copy.begin(&prompt);
    var out: [15]i32 = undefined;
    // the prompt's windows, as `Decoder.advance` hands them: the tokens after each row, nothing to draft
    _ = try ask(copy.drafter(), prompt[1..9], 8, out[0..0]);
    _ = try ask(copy.drafter(), prompt[9..], prompt.len - 1, out[0..0]);
    for (prompt, copy.context.items) |token, held| try testing.expectEqual(@as(u32, @bitCast(token)), held);
    try testing.expectEqual(15, (try ask(copy.drafter(), &begun, prompt.len + begun.len - 1, &out)).len);

    // the next request holds none of it
    try copy.begin(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    try testing.expectEqualSlices(i32, &.{17}, try ask(copy.drafter(), &begun, 8 + begun.len - 1, &out));
}

test "after four windows of copies with no row kept the lookup is silent for sixteen rounds" {
    var inner: Fixed = .{};
    var copy: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer copy.deinit();
    // a context that repeats 1..8 five times, so every tail has an earlier occurrence
    var repeated: [40]i32 = undefined;
    for (&repeated, 0..) |*token, i| token.* = 1 + @as(i32, @intCast(i % 8));
    try copy.begin(&repeated);
    var out: [15]i32 = undefined;
    var position: usize = repeated.len - 1;
    var last: [1]i32 = .{repeated[repeated.len - 1]};
    // each round the drafter agrees with the copy's first token, and the model then keeps the last row alone
    for (0..4) |_| {
        inner.chain = &.{1 + @mod(last[0], 8)};
        try testing.expect((try ask(copy.drafter(), &last, position, &out)).len > 1);
        position += 1;
        last[0] = 1 + @mod(last[0], 8);
    }
    try testing.expectEqual(4, copy.stats.offered);
    for (0..16) |_| {
        inner.chain = &.{1 + @mod(last[0], 8)};
        try testing.expectEqual(1, (try ask(copy.drafter(), &last, position, &out)).len);
        position += 1;
        last[0] = 1 + @mod(last[0], 8);
    }
    try testing.expectEqual(4, copy.stats.offered);
    try testing.expectEqual(0, copy.stats.kept);
    inner.chain = &.{1 + @mod(last[0], 8)};
    try testing.expect((try ask(copy.drafter(), &last, position, &out)).len > 1);
}

test "a lookup asked ahead of the drafter offers what the drafter's call would, and `copyOf` the same copy beside it" {
    var inner: Fixed = .{};
    var by_call: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer by_call.deinit();
    var by_ahead: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer by_ahead.deinit();
    try by_call.begin(&prompt);
    try by_ahead.begin(&prompt);
    const position = prompt.len + begun.len - 1;
    const kept = [_]i32{ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 40 };
    // agreed, then not agreed and judged, then agreed again: each window the same both ways
    const rounds = [_]struct { tokens: []const i32, position: usize, chain: []const i32, room: usize }{
        .{ .tokens = &begun, .position = position, .chain = &.{ 17, 99 }, .room = 15 },
        .{ .tokens = &kept, .position = position + kept.len, .chain = &.{ 77, 78 }, .room = 6 },
        .{ .tokens = &.{ 11, 12, 13, 14, 15, 16 }, .position = position + kept.len + 6, .chain = &.{ 17, 1 }, .room = 4 },
    };
    for (rounds) |one| {
        inner.chain = one.chain;
        var out_call: [15]i32 = undefined;
        var out_ahead: [15]i32 = undefined;
        const want = try ask(by_call.drafter(), one.tokens, one.position, out_call[0..one.room]);
        const found = try by_ahead.ahead(one.tokens, one.position, one.room);
        const got = try ask(by_ahead.drafter(), one.tokens, one.position, out_ahead[0..one.room]);
        try testing.expectEqualSlices(i32, want, got);
        // what another holder of the same drafts makes of what was found
        var beside: [15]i32 = undefined;
        const n = @min(one.chain.len, one.room);
        @memcpy(beside[0..n], one.chain[0..n]);
        const rows = if (found.len == 0) null else copied.Copied.copyOf(found, &.{}, beside[0..one.room]);
        try testing.expectEqualSlices(i32, want, beside[0 .. rows orelse n]);
    }
    try testing.expectEqual(by_call.stats, by_ahead.stats);
    try testing.expectEqual(2, by_call.stats.offered);
}

/// A record of Recall's in a directory of the test's own, closed by the caller.
fn recallStore(tmp: *const std.testing.TmpDir, a: std.mem.Allocator) !*recall.Store {
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/recall", .{tmp.sub_path});
    var problem: []const u8 = "";
    const identity: recall.Identity = .{ .digest = @splat(7), .vocab = 1000, .name = "test" };
    return (try recall.open(testing.allocator, testing.io, a, identity, .{ .dir = dir }, null, &problem)).?;
}

// a prompt that repeats nothing, and the reply an earlier request gave to it
const fx = struct {
    const asked = blk: {
        var run: [40]i32 = undefined;
        for (&run, 0..) |*token, i| token.* = 100 + @as(i32, @intCast(i));
        break :blk run;
    };
    const answered = blk: {
        var run: [30]i32 = undefined;
        for (&run, 0..) |*token, i| token.* = 300 + @as(i32, @intCast(i));
        break :blk run;
    };
};

test "Recall: a request fx.asked again is offered the earlier reply from the record where its context holds none, judged by the record's rates" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const store = try recallStore(&tmp, arena_state.allocator());
    defer store.close();
    var inner: Fixed = .{};
    var out: [15]i32 = undefined;

    // the first request: nothing recalled (the record held nothing of it), and its reply goes in at its end
    var first: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer first.deinit();
    first.store = store;
    try first.begin(&fx.asked);
    inner.chain = &.{ 301, 999 };
    try testing.expectEqualSlices(i32, &.{ 301, 999 }, try ask(first.drafter(), fx.answered[0..1], fx.asked.len, &out));
    _ = try ask(first.drafter(), fx.answered[1..], fx.asked.len + fx.answered.len - 1, out[0..0]);
    first.end();
    try testing.expectEqual(0, first.stats.recalled);

    // the same request again: the context repeats nothing, and the record holds what followed
    var again: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer again.deinit();
    again.store = store;
    try again.begin(&fx.asked);
    // the prompt's windows, as `Decoder.advance` hands them, are no rounds: the bar stays the reader's prior
    _ = try ask(again.drafter(), fx.asked[1..], fx.asked.len - 1, out[0..0]);
    try testing.expectEqual(2, again.reader.?.head);
    try testing.expectEqualSlices(i32, fx.answered[1..16], try ask(again.drafter(), fx.answered[0..1], fx.asked.len, &out));
    try testing.expectEqual(1, again.stats.recalled);
    try testing.expectEqual(15, again.stats.recalled_rows);
    // the window kept ten copied rows: they are Recall's, and the record's rates took them
    const proposals = store.shared.rates.proposals;
    _ = try ask(again.drafter(), fx.answered[1..12], fx.asked.len + 11, out[0..0]);
    try testing.expectEqual(10, again.stats.recalled_kept);
    try testing.expectEqual(10, again.stats.kept);
    try testing.expectEqual(proposals + 1, store.shared.rates.proposals);
    var line: [96]u8 = undefined;
    try testing.expectEqualStrings(", 10 of 15 copied, 10 of 15 recalled", again.stats.line(&line));

    // a head that does not begin as the record does is not overruled: its chain stands
    var other: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer other.deinit();
    other.store = store;
    try other.begin(&fx.asked);
    inner.chain = &.{ 555, 556 };
    try testing.expectEqualSlices(i32, &.{ 555, 556 }, try ask(other.drafter(), fx.answered[0..1], fx.asked.len, &out));
    try testing.expectEqual(0, other.stats.offered);

    // without a store nothing is recalled, as before Recall
    var off: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer off.deinit();
    try off.begin(&fx.asked);
    inner.chain = &.{ 301, 999 };
    try testing.expectEqualSlices(i32, &.{ 301, 999 }, try ask(off.drafter(), fx.answered[0..1], fx.asked.len, &out));
    off.end();
}

test "Recall: the context's own copy comes first, and a record read only takes nothing in" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const store = try recallStore(&tmp, arena_state.allocator());
    defer store.close();
    var inner: Fixed = .{};
    var out: [15]i32 = undefined;
    // the record holds `prompt` followed by another text than the prompt's own continuation
    var seeded: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer seeded.deinit();
    seeded.store = store;
    try seeded.begin(prompt[0..13]);
    _ = try ask(seeded.drafter(), &.{ 70, 71, 72, 73, 74, 75, 76, 77, 78, 79 }, 22, out[0..0]);
    seeded.end();
    // the context's copy of 10..16's continuation is offered, not the record's
    var copy: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer copy.deinit();
    copy.store = store;
    try copy.begin(&prompt);
    inner.chain = &.{17};
    try testing.expectEqualSlices(i32, &.{ 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 4, 5 }, try ask(copy.drafter(), &begun, prompt.len + begun.len - 1, &out));
    try testing.expectEqual(0, copy.stats.recalled);

    const before = store.stats().ids;
    store.writes = false;
    defer store.writes = true;
    var reading: copied.Copied = try .init(testing.allocator, inner.drafter(), &.{});
    defer reading.deinit();
    reading.store = store;
    try reading.begin(&fx.asked);
    _ = try ask(reading.drafter(), &fx.answered, fx.asked.len + fx.answered.len - 1, out[0..0]);
    reading.end();
    try testing.expectEqual(before, store.stats().ids);
}
