//! `book.zig`'s tests, on the host: where a prompt goes on from the states two ranks hold, turn after
//! turn, and that no state is ever gone back to at a position the sequence ran again with other tokens.

const std = @import("std");
const testing = std.testing;
const book_file = @import("../../src/decode/book.zig");
const Book = book_file.Book;
const Back = book_file.Back;

fn digest(tokens: []const i32) u64 {
    return std.hash.Wyhash.hash(7, std.mem.sliceAsBytes(tokens));
}

/// A pair's account driven as `pair.zig`'s `reply` drives it, with no device: each state taken is the digest
/// of the tokens below it, and a state gone back to must be the prompt's.
const Sim = struct {
    gpa: std.mem.Allocator,
    book: Book,
    seen: std.ArrayList(i32) = .empty,
    pos: usize = 0,
    trusted: bool = false,
    states: std.AutoHashMapUnmanaged(usize, u64) = .empty,
    /// A prompt window's rows at most, and the last one's (`Decoder.promptRows`).
    wide: usize,
    room: usize,
    /// The think block's closing tag in a reply's tokens.
    close: i32 = -2,
    taken: usize = 0,

    fn init(gpa: std.mem.Allocator, options: book_file.Options, wide: usize, room: usize) !Sim {
        return .{ .gpa = gpa, .book = try .init(gpa, options), .wide = wide, .room = room };
    }

    fn deinit(self: *Sim) void {
        self.states.deinit(self.gpa);
        self.seen.deinit(self.gpa);
        self.book.deinit(self.gpa);
    }

    fn rowsOf(self: *const Sim, pos: usize, len: usize) usize {
        const left = len - pos;
        const last = @min(self.wide, self.room);
        return if (left <= last) left else @min(self.wide, left - last);
    }

    fn take(self: *Sim, at: usize) !void {
        var all: [64]usize = undefined;
        const list = self.book.positions(&all);
        // the newest, and the shelf's order
        try testing.expectEqual(at, list[list.len - 1]);
        try testing.expect(!self.states.contains(at));
        try self.states.put(self.gpa, at, digest(self.seen.items[0..at]));
        self.taken += 1;
    }

    fn drop(self: *Sim, at: usize) !void {
        try testing.expect(self.states.remove(at));
    }

    fn held(self: *Sim) !void {
        var all: [64]usize = undefined;
        const list = self.book.positions(&all);
        try testing.expectEqual(list.len, self.states.count());
        try testing.expect(list.len <= self.book.options.marks.budget + self.book.options.reply_marks);
        for (list, 0..) |at, i| {
            if (i > 0) try testing.expect(list[i - 1] < at);
            try testing.expect(at <= self.pos);
            try testing.expectEqual(digest(self.seen.items[0..at]), self.states.get(at).?);
        }
    }

    /// The reply to `prompt`: `reply` tokens in rounds that keep `round` rows, absorbed by the head or not.
    /// Returns the prompt's tokens that were not run.
    fn run(self: *Sim, prompt: []const i32, starts: []const usize, reply: []const i32, round: usize, absorbing: bool) !usize {
        // what one rank's `resume` gives, by its words
        var all: [64]usize = undefined;
        const before = self.book.positions(&all);
        const shared = if (self.trusted) std.mem.indexOfDiff(i32, self.seen.items, prompt) orelse @min(self.seen.items.len, prompt.len) else 0;
        var expected: usize = 0;
        if (self.trusted and self.pos < shared) {
            expected = self.pos;
        } else for (before) |at| {
            if (at < shared) expected = @max(expected, at);
        }

        const start = self.book.start(self.seen.items, prompt, self.trusted, self.pos, null);
        try testing.expectEqual(shared, start.shared);
        for (start.drops) |at| {
            try testing.expect(at >= shared);
            try self.drop(at);
        }
        switch (start.to) {
            .stay => {},
            .back => |at| {
                // the state's tokens are the prompt's, and the token after them, which the head absorbed
                try testing.expect(at < prompt.len and at < shared);
                try testing.expectEqual(digest(prompt[0..at]), self.states.get(at).?);
                self.pos = at;
            },
            .reset => self.pos = 0,
        }
        self.trusted = false;
        const cached = self.pos;
        try testing.expectEqual(expected, cached);
        try testing.expect(cached < prompt.len);
        self.seen.shrinkRetainingCapacity(cached);
        try self.seen.appendSlice(self.gpa, prompt[cached..]);
        try self.held();

        const inside = absorbing and self.book.options.marks.budget > 0;
        if (inside) {
            var planner = self.book.plan(.{ .pos = cached, .len = prompt.len, .shared = shared, .starts = starts });
            while (planner.from < prompt.len) self.book.window(&planner, self.rowsOf(planner.from, prompt.len));
            for (self.book.planned(&planner)) |at| try self.drop(at);
            while (self.pos < prompt.len) {
                const stop = self.pos + self.rowsOf(self.pos, prompt.len);
                for (self.book.due) |at| {
                    if (at <= self.pos or at > stop) continue;
                    self.book.took(at);
                    try self.take(at);
                }
                self.pos = stop;
            }
        }
        self.pos = prompt.len;
        // the reply's first token is chosen, and each round runs the tokens chosen before it
        var made: usize = 0;
        var closing = inside;
        try self.seen.append(self.gpa, reply[0]);
        made += 1;
        while (made < reply.len) {
            const rows = @min(round, reply.len - made);
            try self.seen.appendSlice(self.gpa, reply[made..][0 .. rows - 1]);
            const window = self.seen.items[self.pos..][0..rows];
            if (inside) {
                if (closing) if (std.mem.indexOfScalar(i32, window, self.close)) |row| {
                    closing = false;
                    if (self.book.close(self.pos + row + 1)) |at| try self.drop(at);
                    try self.take(self.pos + row + 1);
                };
                if (self.book.stride(self.pos, rows)) |one| {
                    if (one.drop) |at| try self.drop(at);
                    try testing.expect(one.at > self.pos and one.at <= self.pos + rows);
                    try self.take(one.at);
                }
            }
            self.pos += rows;
            try self.seen.append(self.gpa, reply[made + rows - 1]);
            made += rows;
        }
        try testing.expectEqual(self.pos + 1, self.seen.items.len);
        self.trusted = absorbing;
        try self.held();
        return cached;
    }
};

fn filled(gpa: std.mem.Allocator, random: std.Random, len: usize) ![]i32 {
    const out = try gpa.alloc(i32, len);
    for (out) |*id| id.* = random.intRangeLessThan(i32, 10, 60);
    return out;
}

test "two ranks' states: a prompt sent again, the next turn, an edit, a reply cut, as one rank keeps them" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(94);
    const random = prng.random();
    // one rank's defaults, a pair's windows: 512 rows, the last 16
    var sim: Sim = try .init(gpa, .{}, 512, 16);
    defer sim.deinit();

    const prompt = try filled(arena, random, 1200);
    const starts = [_]usize{ 0, 30, 700 };
    const reply = try filled(arena, random, 300);
    // from empty: the starts, the prompt's end less two; the strides of 512 hold a start each and take none
    try testing.expectEqual(@as(usize, 0), try sim.run(prompt, &starts, reply, 5, true));
    var all: [16]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 30, 700, 1198, 1454 }, sim.book.positions(&all));
    // sent again: the two rows after the mark before its end, and the state one token before where it parted
    // from the reply; the reply's own state is gone
    try testing.expectEqual(@as(usize, 1198), try sim.run(prompt, &starts, reply, 5, true));
    try testing.expectEqualSlices(usize, &.{ 30, 700, 1198, 1199, 1455 }, sim.book.positions(&all));
    // the next turn holds the prompt and the whole reply: nothing is run again
    var turn: std.ArrayList(i32) = .empty;
    try turn.appendSlice(arena, sim.seen.items);
    try turn.appendSlice(arena, try filled(arena, random, 40));
    const turn_starts = [_]usize{ 0, 30, 700, 1500 };
    try testing.expectEqual(@as(usize, 1499), try sim.run(turn.items, &turn_starts, reply, 16, true));
    try testing.expectEqualSlices(usize, &.{ 30, 700, 1198, 1199, 1455, 1500, 1538, 1794 }, sim.book.positions(&all));
    // the second message edited: back to the first's start, and everything above is gone
    const edited = try arena.dupe(i32, turn.items);
    edited[100] = 5;
    try testing.expectEqual(@as(usize, 30), try sim.run(edited, &turn_starts, reply, 16, true));
    try testing.expectEqualSlices(usize, &.{ 30, 99, 542, 700, 1500, 1538, 1794 }, sim.book.positions(&all));
    // and the turn as it was: the states of the edited one above the edit are no state of it
    try testing.expectEqual(@as(usize, 99), try sim.run(turn.items, &turn_starts, reply, 16, true));
    // a client that left after 20 tokens of 300, then its next turn: the reply's tokens it got, and more
    var cut: std.ArrayList(i32) = .empty;
    try cut.appendSlice(arena, turn.items);
    try cut.appendSlice(arena, reply[0..20]);
    try cut.appendSlice(arena, try filled(arena, random, 9));
    try testing.expectEqual(@as(usize, 1538), try sim.run(cut.items, &turn_starts, reply, 16, true));
    // a reply the head did not absorb leaves nothing to go on from
    _ = try sim.run(cut.items, &turn_starts, reply, 1, false);
    try testing.expectEqual(@as(usize, 0), try sim.run(cut.items, &turn_starts, reply, 16, true));
}

test "two ranks' states: no state is gone back to at a position run again with other tokens, whatever the turns" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(940);
    const random = prng.random();
    for (0..40) |_| {
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const options: book_file.Options = .{
            .marks = .{ .budget = random.intRangeAtMost(usize, 0, 4), .stride = random.intRangeAtMost(usize, 0, 12), .near = random.intRangeAtMost(usize, 0, 4), .tail = random.intRangeAtMost(usize, 0, 3) },
            .reply_stride = random.intRangeAtMost(usize, 0, 9),
            .reply_marks = random.intRangeAtMost(usize, 0, 2),
        };
        var sim: Sim = try .init(gpa, options, random.intRangeAtMost(usize, 4, 24), random.intRangeAtMost(usize, 1, 16));
        defer sim.deinit();
        var prompt: std.ArrayList(i32) = .empty;
        try prompt.appendSlice(arena, try filled(arena, random, random.intRangeAtMost(usize, 1, 60)));
        var went_back: usize = 0;
        for (0..60) |_| {
            switch (random.intRangeLessThan(usize, 0, 6)) {
                // the next turn: all that was run, and more
                0 => {
                    prompt.clearRetainingCapacity();
                    try prompt.appendSlice(arena, sim.seen.items);
                    try prompt.appendSlice(arena, try filled(arena, random, random.intRangeAtMost(usize, 1, 30)));
                },
                // an edit somewhere, the rest as it was or cut
                1 => if (prompt.items.len > 0) {
                    const at = random.intRangeLessThan(usize, 0, prompt.items.len);
                    prompt.items[at] = random.intRangeLessThan(i32, 0, 9);
                    if (random.boolean()) prompt.shrinkRetainingCapacity(at + 1);
                },
                // a reply cut, and what the client got of it
                2 => if (sim.seen.items.len > 2) {
                    prompt.clearRetainingCapacity();
                    try prompt.appendSlice(arena, sim.seen.items[0..random.intRangeLessThan(usize, 1, sim.seen.items.len)]);
                    try prompt.appendSlice(arena, try filled(arena, random, random.intRangeAtMost(usize, 0, 5)));
                },
                // sent again
                3, 4 => {},
                // another conversation
                else => {
                    prompt.clearRetainingCapacity();
                    try prompt.appendSlice(arena, try filled(arena, random, random.intRangeAtMost(usize, 1, 60)));
                },
            }
            if (prompt.items.len == 0) try prompt.append(arena, 1);
            var starts: [4]usize = undefined;
            var n: usize = 0;
            var at: usize = 0;
            while (n < starts.len) : (n += 1) {
                at += random.intRangeAtMost(usize, 0, 20);
                if (at >= prompt.items.len) break;
                starts[n] = at;
                at += 1;
            }
            const reply = try filled(arena, random, random.intRangeAtMost(usize, 1, 40));
            if (random.intRangeLessThan(usize, 0, 4) == 0) reply[random.intRangeLessThan(usize, 0, reply.len)] = sim.close;
            const cached = try sim.run(prompt.items, starts[0..n], reply, random.intRangeAtMost(usize, 1, 16), random.intRangeLessThan(usize, 0, 8) != 0);
            if (cached > 0) went_back += 1;
        }
        // with a budget the turns found states to go on from
        if (options.marks.budget > 1 and options.marks.tail > 0) try testing.expect(went_back > 0 and sim.taken > 0);
    }
}

test "GLM's reach: a prompt sent again goes back to before its end, a longer one to the end it shares" {
    const gpa = testing.allocator;
    var book: Book = try .init(gpa, .{ .through = false });
    defer book.deinit(gpa);
    try book.reserve(gpa, 3, 0);
    for ([_]usize{ 1, 17, 28 }) |at| book.put(at, false);
    try testing.expectEqual(@as(?usize, 1), (book.under(book.reach(17, 17)) orelse return error.None).pos);
    try testing.expectEqual(@as(?usize, 17), (book.under(book.reach(17, 32)) orelse return error.None).pos);
    try testing.expectEqual(@as(?Back, null), book.under(book.reach(0, 17)));
    book.clear();
    book.put(17, true);
    try testing.expectEqual(@as(?Back, null), book.under(book.reach(17, 17)));
}

test "the states a window holds: past its first row, up to its last, as rows into it" {
    var out: [3]u32 = undefined;
    const at = [_]usize{ 4, 10, 11, 25, 26 };
    try testing.expectEqualSlices(u32, &.{ 6, 7 }, book_file.inside(&at, 4, 7, &out).?);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 16 }, book_file.inside(&at, 9, 16, &out).?);
    try testing.expectEqualSlices(u32, &.{}, book_file.inside(&at, 11, 13, &out).?);
    try testing.expectEqual(@as(?[]u32, null), book_file.inside(&at, 0, 30, &out));
}
