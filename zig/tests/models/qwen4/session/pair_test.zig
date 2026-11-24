//! `pair.zig`'s tests: both ranks in one process (`Of(2)`), a prompt run from where it parts from the sequence
//! the pair holds (spec 094). Every reply is compared, token for token, with the same two ranks' reply from an
//! empty state: a prompt sent again, the next turn, a message edited, the sequence left gone back to, a turn
//! after a reply a client cut, after a reply the head did not draft. The head whole beside rank 0 and split,
//! and with an attention layer whose caches are bf16, FP8, or FP8 keys with NVFP4 values.

const std = @import("std");
const testing = std.testing;
const fixtures = @import("../testing.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const layout = @import("../../../../src/models/qwen4/layout.zig");
const model = @import("../../../../src/models/qwen4/model.zig");
const ranks = @import("../../../../src/models/qwen4/ranks_forward.zig");
const random = @import("../../../../src/models/qwen4/checks/random.zig");
const pair_file = @import("../../../../src/models/qwen4/session/pair.zig");
const book_file = @import("../../../../src/decode/book.zig");

const Allocator = std.mem.Allocator;
const Both = pair_file.Of(2);
const Held = ranks.Held(2);

const reply_settings = [_]sampling.Sampling{
    .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 },
    .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 },
};

/// A pair that goes on from what it holds, and one emptied before each reply that keeps no state.
const Case = struct {
    pair: *Both,
    empty: *Both,
    settings: sampling.Sampling,
    by: pair_file.Drafts,
    taken: usize = 0,

    const Got = struct { cached: usize, tokens: []const i32 };

    /// The reply to `prompt` on the pair as it stands, which must be the emptied pair's.
    fn turn(self: *Case, prompt: []const i32, how: pair_file.Turn, by: pair_file.Drafts, max_tokens: usize, hang_at: ?usize) !Got {
        const pair = self.pair;
        const before = pair.tokens.items.len;
        const got = try pair.reply(prompt, self.settings, &.{}, by, how, null, max_tokens, hang_at, null);
        try self.empty.reset();
        const from = self.empty.tokens.items.len;
        const want = try self.empty.reply(prompt, self.settings, &.{}, by, .{}, null, max_tokens, hang_at, null);
        try testing.expectEqual(@as(usize, 0), want.cached);
        try testing.expectEqual(@as(usize, 0), want.taken);
        try testing.expectEqual(want.finish, got.finish);
        try testing.expectEqualSlices(i32, self.empty.tokens.items[from..], pair.tokens.items[before..]);
        // rank 0's account is what the ranks hold
        var all: [32]usize = undefined;
        const held = pair.book.positions(&all);
        try testing.expectEqual(held.len, pair.shelf.held.items.len);
        for (held, pair.shelf.held.items) |at, one| for (one) |snap| try testing.expectEqual(at, snap.pos);
        try testing.expectEqual(pair.state.pos + 1, pair.seen.items.len);
        if (pair.head) |h| if (by.depth > 0) try testing.expectEqual(pair.state.pos, h.state.position.host);
        self.taken += got.taken;
        return .{ .cached = got.cached, .tokens = pair.tokens.items[before..] };
    }
};

/// The turns of a conversation and what a client does to it. `other` is a token for an edit: none of `first`'s.
fn conversation(gpa: Allocator, case: *Case, first: []const i32, starts: []const usize, more: []const i32, other: i32) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const by = case.by;
    const count = 14;
    const tail = case.pair.book.options.marks.tail;

    // from nothing
    const one = try case.turn(first, .{ .starts = starts }, by, count, null);
    try testing.expectEqual(@as(usize, 0), one.cached);
    const reply = try arena.dupe(i32, one.tokens);
    // sent again: from the mark before its end
    const again = try case.turn(first, .{ .starts = starts }, by, count, null);
    try testing.expectEqual(first.len - tail, again.cached);
    // the next turn: all of the prompt and of the reply but its last token, which no window ran
    var next: std.ArrayList(i32) = .empty;
    try next.appendSlice(arena, first);
    try next.appendSlice(arena, reply);
    try next.appendSlice(arena, more);
    var next_starts: std.ArrayList(usize) = .empty;
    try next_starts.appendSlice(arena, starts);
    try next_starts.append(arena, first.len + reply.len);
    const second = try case.turn(next.items, .{ .starts = next_starts.items }, by, count, null);
    try testing.expectEqual(first.len + reply.len - 1, second.cached);
    // a message edited, past the second start: from a state below the edit
    const edit = starts[1] + 3;
    const edited = try arena.dupe(i32, next.items);
    edited[edit] = other;
    const third = try case.turn(edited, .{ .starts = next_starts.items }, by, count, null);
    try testing.expect(third.cached > 0 and third.cached < edit);
    // the turn as it was: the states the edited one left above the edit are none of it, and the one it took
    // a token before the edit serves
    const fourth = try case.turn(next.items, .{ .starts = next_starts.items }, by, count, null);
    try testing.expectEqual(edit - 1, fourth.cached);
    // a client that left after five tokens, and its next turn over what it got
    const cut = try case.turn(next.items, .{ .starts = next_starts.items }, by, count, 5);
    try testing.expectEqual(@as(usize, 5), cut.tokens.len);
    var after: std.ArrayList(i32) = .empty;
    try after.appendSlice(arena, next.items);
    try after.appendSlice(arena, cut.tokens);
    try after.appendSlice(arena, more);
    const fifth = try case.turn(after.items, .{ .starts = next_starts.items }, by, count, null);
    try testing.expect(fifth.cached >= next.items.len - tail and fifth.cached < next.items.len + 5);
    // the state after the token that closes the reasoning: a prompt that parts two tokens after it
    const closing = try case.turn(first, .{ .starts = starts, .close = reply[3] }, by, count, null);
    try testing.expectEqualSlices(i32, reply, closing.tokens);
    var parted: std.ArrayList(i32) = .empty;
    try parted.appendSlice(arena, first);
    try parted.appendSlice(arena, reply[0..5]);
    try parted.append(arena, if (reply[5] == other) other + 1 else other);
    const close_at = first.len + (std.mem.indexOfScalar(i32, reply, reply[3]).? + 1);
    const sixth = try case.turn(parted.items, .{ .starts = starts }, by, count, null);
    try testing.expect(sixth.cached >= close_at and sixth.cached <= first.len + 4);
    // a reply nobody drafted: the head lags the state, and the next prompt runs whole, as on one rank
    if (case.pair.head != null) {
        _ = try case.turn(first, .{ .starts = starts }, .{}, count, null);
        const seventh = try case.turn(first, .{ .starts = starts }, by, count, null);
        try testing.expectEqual(@as(usize, 0), seventh.cached);
    }
    try testing.expect(case.taken > 0);
}

const Head = enum { none, whole, split };

const storages = [_]attention.Storage{ .bf16, .fp8, .fp8_nvfp4 };

/// `caches`: an attention layer after the DeltaNet ones, its caches so stored; null, the model without one.
fn randomPair(gpa: Allocator, arena: Allocator, head: Head, rows: usize, scored: usize, capacity: usize, eager: bool, caches: ?attention.Storage) !*Both {
    const built = try random.sharesOf(2, gpa, arena, 42, 0, rows, .{ .head = switch (head) {
        .none => .none,
        .whole => .whole,
        .split => .split,
    }, .caches = caches });
    return Both.createLoopbackScoring(gpa, .{ .ranks = built, .max_rows = rows }, rows, scored, capacity, eager);
}

/// States every few tokens, so that a short prompt takes several, inside its windows and at their ends.
const close_marks: book_file.Options = .{ .marks = .{ .budget = 8, .stride = 16, .near = 2, .tail = 2 }, .reply_stride = 5, .reply_marks = 2 };

fn randomTurns(head: Head, eager: bool, caches: ?attention.Storage) !void {
    return randomTurnsOf(head, eager, caches, 4, 4);
}

/// `randomTurns` in windows of `rows` rows, `scored` of them scored: fewer, and a prompt's window is wider than
/// the rows each layer keeps, its states asked of it before it runs.
fn randomTurnsOf(head: Head, eager: bool, caches: ?attention.Storage, rows: usize, scored: usize) !void {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(94);
    const first = try arena.alloc(i32, 27);
    for (first) |*id| id.* = prng.random().intRangeLessThan(i32, 1, 300);
    const more = [_]i32{ 9, 301, 44, 17, 250 };
    for (reply_settings) |settings| {
        const pair = try randomPair(gpa, arena, head, rows, scored, 256, eager, caches);
        if (caches) |kv| for (pair.state.ranks) |rank| {
            try testing.expectEqual(@as(usize, 1), rank.caches.len);
            try testing.expectEqual(kv.dtype(), rank.caches[0].k.view.dtype);
            try testing.expectEqual(kv.valueDtype(), rank.caches[0].v.view.dtype);
            try testing.expectEqual(kv != .bf16, rank.caches[0].k_scale != null);
        };
        defer pair.destroy();
        try pair.marking(close_marks);
        const empty = try randomPair(gpa, arena, head, rows, scored, 256, eager, caches);
        defer empty.destroy();
        try empty.marking(.{ .marks = .{ .budget = 0 } });
        var case: Case = .{ .pair = pair, .empty = empty, .settings = settings, .by = .{ .depth = if (head == .none) 0 else scored - 1 } };
        try conversation(gpa, &case, first, &.{ 0, 6, 19 }, &more, 319);
    }
}

test "two ranks keep the sequence they ran, the head whole beside rank 0: each reply is the one from an empty state" {
    try randomTurns(.whole, true, null);
}

test "two ranks keep the sequence they ran, the head split over them: each reply is the one from an empty state" {
    try randomTurns(.split, true, null);
}

test "two ranks keep the sequence they ran, through graphs and with no head: each reply is the one from an empty state" {
    try randomTurns(.split, false, null);
    try randomTurns(.whole, false, null);
    try randomTurns(.none, true, null);
}

test "two ranks keep the sequence they ran in prompt windows wider than the rows kept, the states asked ahead: each reply is the one from an empty state" {
    try randomTurnsOf(.split, true, null, 12, 4);
    try randomTurnsOf(.whole, false, .fp8, 12, 4);
}

test "two ranks keep the sequence they ran over attention caches in each storage, the head split: each reply is the one from an empty state" {
    for (storages) |kv| try randomTurns(.split, true, kv);
}

test "two ranks keep the sequence they ran over attention caches in each storage, the head whole and through graphs: each reply is the one from an empty state" {
    for (storages) |kv| try randomTurns(.whole, true, kv);
    for (storages[1..]) |kv| try randomTurns(.split, false, kv);
}

test "two ranks of a checkpoint's first layers keep the sequence they ran: the attention's rows, the PLE, the head rewound" {
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();

    // three DeltaNet layers and an attention layer, the PLE on the first
    const window = 8;
    const layers = 4;
    const ids = try model.DraftHead.defaultIds(arena);
    const first = [_]i32{ 151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198 };
    const more = [_]i32{ 151645, 198, 151644, 872, 198, 3838, 30, 151645, 198, 151644, 77091, 198 };
    // the caches bf16 with the head whole and split, then coded, the head's own cache bf16 beside them. One load
    // a rank for each way the head is held (`head_split` changes the load), its storages by
    // `checkpoints.storing`; the pair and its reference from an empty state borrow it, one running at a time.
    const Way = struct { split: bool, kv: [2]attention.Storage };
    for ([_]Way{ .{ .split = false, .kv = .{ .bf16, .fp8_nvfp4 } }, .{ .split = true, .kv = .{ .bf16, .fp8 } } }) |way| {
        var loaded: [2]model.Model = undefined;
        var held: usize = 0;
        defer for (loaded[0..held]) |*share| share.deinit();
        for (&loaded, 0..) |*share, rank| {
            share.* = try .loadWith(gpa, config, found, .{ .layers = layers, .max_rows = window, .mtp = true, .draft_vocab = ids, .share = .of(@intCast(rank)), .head_split = way.split, .kv = way.kv[0], .head_kv = layout.head_storage });
            held += 1;
        }
        for (way.kv) |kv| {
            var ranks_m: [2]model.Model = undefined;
            for (&ranks_m, loaded) |*share, one| share.* = fixtures.checkpoints.storing(one, kv, layout.head_storage);
            var both: [2]*Both = undefined;
            var made: usize = 0;
            defer for (both[0..made]) |one| one.destroy();
            for (&both) |*one| {
                one.* = try Both.createLoopbackBorrowing(gpa, .{ .ranks = ranks_m, .max_rows = window }, window, 256, false);
                made += 1;
            }
            try both[0].marking(close_marks);
            try both[1].marking(.{ .marks = .{ .budget = 0 } });
            var case: Case = .{ .pair = both[0], .empty = both[1], .settings = reply_settings[0], .by = .{ .depth = 4 } };
            try conversation(gpa, &case, &first, &.{ 0, 11, 23 }, &more, 220);
        }
    }
}
