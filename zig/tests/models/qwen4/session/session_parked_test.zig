//! `session.zig`'s tests of the sequences it leaves (`parked.zig`, spec 064): conversations in turn on one
//! session, each going on from where a held turn would, with the tokens and the drafts of a session that runs
//! every prompt from an empty state.

const std = @import("std");
const testing = std.testing;
const fixtures = @import("../testing.zig");
const Counting = fixtures.sessions.Counting;
const sampling = @import("../../../../src/decode/sampling.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const model = @import("../../../../src/models/qwen4/model.zig");
const parked_file = @import("../../../../src/models/qwen4/session/parked.zig");
const session_file = @import("../../../../src/models/qwen4/session/session.zig");
const Session = session_file.Session;

const count = 12;

/// A conversation: its prompt so far, and the last reply to it.
const Talk = struct {
    prompt: std.ArrayList(i32) = .empty,
    record: [count]i32 = undefined,
    /// The state's position after its last reply, and its last prompt's length.
    end: usize = 0,
    asked: usize = 0,
};

test "conversations in turn on one session each go on from their own end: tokens, drafts and the rows run" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // a head with an attention layer over DeltaNet layers: the head's entries are what is put back
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer m.deinit();
    try inTurn(m);
}

test "conversations in turn over an attention layer and a PLE: the caches' rows are put back" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    try inTurn(m);
}

fn inTurn(m: model.Model) !void {
    const gpa = testing.allocator;
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = 0 });
    defer fresh.destroy();
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const sampled: sampling.Sampling = .{ .seed = 9, .temperature = 0.8 };

    // `whole`: a turn's prompt is the sequence held and a message, as a template that writes a reply into the
    // history as it was sent; else the prompt's last token is another, as Qwen's is after a think block, and
    // the turn starts from the mark two before the prompt's end. `kept`: the sequences the lot holds.
    for ([_]struct { whole: bool, kept: usize, talks: usize }{
        .{ .whole = true, .kept = 4, .talks = 2 },
        .{ .whole = false, .kept = 4, .talks = 2 },
        .{ .whole = false, .kept = 4, .talks = 3 },
        .{ .whole = true, .kept = 1, .talks = 3 },
        .{ .whole = false, .kept = 0, .talks = 2 },
    }) |case| {
        const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = 4, .stride = 32, .parked = case.kept });
        defer session.destroy();
        var random: std.Random.DefaultPrng = .init(64);
        var talks: [3]Talk = @splat(.{});
        defer for (&talks) |*talk| talk.prompt.deinit(gpa);
        // a system prompt of 40 tokens they share, then 88 of each one's own
        var system: [40]i32 = undefined;
        for (&system) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
        var served: usize = 0;
        for (0..4) |round| for (talks[0..case.talks], 0..) |*talk, which| {
            const first = round == 0;
            if (first) {
                try talk.prompt.appendSlice(gpa, &system);
                for (0..88) |_| try talk.prompt.append(gpa, random.random().intRangeLessThan(i32, 1, 250));
            } else {
                if (!case.whole) {
                    talk.prompt.items.len -= 1;
                    try talk.prompt.append(gpa, 251);
                }
                try talk.prompt.appendSlice(gpa, &talk.record);
                for (0..20) |i| try talk.prompt.append(gpa, @intCast(100 + which * 30 + i));
            }
            const settings = if ((round + which) % 2 == 0) greedy else sampled;
            const expected = try fixtures.serial.serialReply(gpa, m, talk.prompt.items, count, settings);
            defer gpa.free(expected);
            const reply = try session.reply(talk.prompt.items, settings, .{ .max_tokens = count, .depth = 3, .record = &talk.record }, counting.sink());
            try testing.expectEqualSlices(i32, expected, &talk.record);
            fresh.trusted = false;
            var again: [count]i32 = undefined;
            const alone = try fresh.reply(talk.prompt.items, settings, .{ .max_tokens = count, .depth = 3, .record = &again }, counting.sink());
            try testing.expectEqualSlices(i32, &again, &talk.record);
            try testing.expectEqual(alone.drafts, reply.drafts);
            // the last row's logits, bit for bit: a cache row that is another sequence's moves them before a token
            try testing.expectEqualSlices(u16, fresh.logits, session.logits);
            try testing.expectEqual(alone.stats.accepted, reply.stats.accepted);
            // a held turn's start: the state as it stood, or the mark two before the prompt's end
            const held = if (case.whole) talk.end else talk.asked -| 2;
            if (!first and reply.cached == held) served += 1;
            if (!first and case.kept >= case.talks - 1) try testing.expectEqual(held, reply.cached);
            if (!first and case.kept == 0) try testing.expect(reply.cached < 40);
            talk.asked = talk.prompt.items.len;
            talk.end = session.state.pos;
            // what went is freed by the reply's end, but the states kept for their buffers (spec 066)
            try testing.expectEqual(@as(usize, 0), session.lot.late.items.len);
            try testing.expect(session.lot.spare.count() <= parked_file.states);
            if (case.kept == 0) try testing.expectEqual(@as(usize, 0), session.lot.spare.count());
        };
        if (case.kept > 0) try testing.expect(session.lot.parked > 0 and session.lot.bytes() > 0);
        // a sequence's end is held in the buffers of a state that went, once one has
        if (case.kept > 0) try testing.expect(session.lot.reused * 2 >= session.lot.parked);
        if (case.kept >= case.talks - 1) try testing.expectEqual(3 * case.talks, session.lot.taken);
    }
}

test "a conversation's first prompt after a long one holds the long one's rows in the store made after its reply: no buffer made, the replies a fresh session's, one over its harness too" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // a head over DeltaNet layers, then an attention layer and a PLE
    var rows = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer rows.deinit();
    var tiny = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer tiny.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    for ([_]model.Model{ rows, tiny }) |m| {
        const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = 0 });
        defer fresh.destroy();
        const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = 4, .stride = 32, .parked = 4 });
        defer session.destroy();
        var random: std.Random.DefaultPrng = .init(65);
        // a long conversation, two of their own, the long one again (each new one parts from the one before
        // at its first token), then one over the long one's first 240 tokens, as over a harness, which parts
        // from it near its end; the long one last, its rows put back from that store
        var talks: [4]std.ArrayList(i32) = @splat(.empty);
        defer for (&talks) |*talk| talk.deinit(gpa);
        for (&talks, [_]usize{ 300, 40, 24, 30 }, 0..) |*talk, length, which| {
            if (which == 3) try talk.appendSlice(gpa, talks[0].items[0..240]) else try talk.append(gpa, @intCast(240 + which));
            for (1..length) |_| try talk.append(gpa, random.random().intRangeLessThan(i32, 1, 240));
        }
        for ([_]usize{ 0, 1, 2, 0, 3, 0 }) |which| {
            var record: [count]i32 = undefined;
            const reply = try session.reply(talks[which].items, greedy, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
            fresh.trusted = false;
            var again: [count]i32 = undefined;
            const alone = try fresh.reply(talks[which].items, greedy, .{ .max_tokens = count, .depth = 3, .record = &again }, counting.sink());
            try testing.expectEqualSlices(i32, &again, &record);
            try testing.expectEqual(alone.drafts, reply.drafts);
            try testing.expectEqualSlices(u16, fresh.logits, session.logits);
        }
        // every lowering found its store ready, and none made a buffer while its prompt waited; the long one,
        // lowered at 240 of its 300 and more, moved into a store of its own rows after the reply (and the one
        // over its harness, lowered at 240 by the last prompt)
        try testing.expect(session.lot.housed_ready >= 4);
        try testing.expectEqual(@as(u64, 0), session.lot.housed_made);
        try testing.expectEqual(@as(usize, 0), session.lot.lowering_made);
        try testing.expect(session.lot.compacted >= 1);
        try testing.expect(session.lot.parked >= 4);
    }
}

/// What the harness test runs with: `kept` told, as the server tells it; not told; told, and the lot held to
/// no bytes.
const Told = enum { kept, plain, capped };

test "a harness's state outlives a long conversation, a second harness and the conversations that fill the lot" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = 0 });
    defer fresh.destroy();
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    // a message starts with this token, as a template's opener
    const opener = 252;
    const harness = 60;

    for ([_]Told{ .kept, .plain, .capped }) |told| {
        const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = 4, .stride = 0, .near = 4, .parked_bytes = if (told == .capped) 0 else 1 << 40 });
        defer session.destroy();
        var random: std.Random.DefaultPrng = .init(640);
        var prompt: std.ArrayList(i32) = .empty;
        defer prompt.deinit(gpa);
        var starts: std.ArrayList(usize) = .empty;
        defer starts.deinit(gpa);
        var record: [count]i32 = undefined;
        // two harnesses, `harness` tokens each and then the first user message's opener
        var systems: [2][harness]i32 = undefined;
        for (&systems, 0..) |*system, which| {
            for (system) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
            system[0] = @intCast(240 + which);
        }
        // Each step is a conversation's first prompt over harness `over` (null: none, a prompt of its own) and
        // `turns` turns after it; `from` is where the first prompt must start with `kept` told.
        const Step = struct { over: ?usize, turns: usize, from: ?usize };
        for ([_]Step{
            // a conversation long enough for the snapshots' budget to thin the harness's state out
            .{ .over = 0, .turns = 8, .from = 0 },
            // a new one over the same harness starts at its end
            .{ .over = 0, .turns = 1, .from = harness },
            // the other harness, and back: the first one's state went to the lot with its sequence
            .{ .over = 1, .turns = 2, .from = 0 },
            .{ .over = 0, .turns = 1, .from = harness },
            .{ .over = 1, .turns = 0, .from = harness },
            // five conversations of no harness fill the lot's four places: the two harnesses' stay
            .{ .over = null, .turns = 0, .from = 0 },
            .{ .over = null, .turns = 0, .from = 0 },
            .{ .over = null, .turns = 0, .from = 0 },
            .{ .over = null, .turns = 0, .from = 0 },
            .{ .over = null, .turns = 0, .from = 0 },
            .{ .over = 0, .turns = 0, .from = harness },
            .{ .over = 1, .turns = 0, .from = harness },
        }, 0..) |step, nth| {
            prompt.clearRetainingCapacity();
            starts.clearRetainingCapacity();
            if (step.over) |which| {
                try starts.append(gpa, 0);
                try prompt.appendSlice(gpa, &systems[which]);
            }
            for (0..step.turns + 1) |turn| {
                if (turn > 0) {
                    try starts.append(gpa, prompt.items.len);
                    try prompt.append(gpa, opener);
                    try prompt.appendSlice(gpa, &record);
                }
                try starts.append(gpa, prompt.items.len);
                try prompt.append(gpa, opener);
                for (0..24) |_| try prompt.append(gpa, random.random().intRangeLessThan(i32, 1, 240));
                const kept: ?usize = if (step.over != null and told != .plain) harness else null;
                const reply = try session.reply(prompt.items, greedy, .{ .max_tokens = count, .depth = 3, .starts = starts.items, .kept = kept, .record = &record }, counting.sink());
                fresh.trusted = false;
                var again: [count]i32 = undefined;
                const alone = try fresh.reply(prompt.items, greedy, .{ .max_tokens = count, .depth = 3, .record = &again }, counting.sink());
                try testing.expectEqualSlices(i32, &again, &record);
                try testing.expectEqual(alone.drafts, reply.drafts);
                try testing.expectEqualSlices(u16, fresh.logits, session.logits);
                if (turn == 0 and nth > 0) {
                    const from = step.from.?;
                    // told nothing, or with no byte for the lot, the harness is run again from below its end
                    switch (told) {
                        .kept => try testing.expectEqual(from, reply.cached),
                        .plain => if (from == harness) try testing.expect(reply.cached < harness),
                        .capped => if (from == harness and nth > 1) try testing.expect(reply.cached < harness),
                    }
                }
                if (told == .capped) try testing.expectEqual(@as(usize, 0), session.lot.bytes());
            }
            if (nth == 0 and told != .plain) {
                // the long conversation kept the harness's state among its four
                try testing.expectEqual(@as(usize, harness), session.snapshots.items[0].pos);
                const expected = try fixtures.serial.serialReply(gpa, m, prompt.items, count, greedy);
                defer gpa.free(expected);
                try testing.expectEqualSlices(i32, expected, &record);
            }
        }
        if (told == .kept) {
            var held: usize = 0;
            for (session.lot.list.items) |one| held += @intFromBool(one.shared != null);
            try testing.expect(held >= 1);
        }
    }
}

test "more harnesses in turn than the lot's places: each new conversation over one starts at its end, the last holder of a harness counted apart (spec 108)" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = 0 });
    defer fresh.destroy();
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const opener = 252;
    const harness = 60;
    // two more harnesses than the lot's four places
    const many = 6;

    for ([_]Told{ .kept, .plain, .capped }) |told| {
        const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = 4, .stride = 32, .near = 4, .parked_bytes = if (told == .capped) 0 else 1 << 40 });
        defer session.destroy();
        // a stride's snapshot stands inside each harness, below its end: the sequence left holds it too
        var random: std.Random.DefaultPrng = .init(1080);
        var systems: [many][harness]i32 = undefined;
        for (&systems, 0..) |*system, which| {
            for (system) |*token| token.* = random.random().intRangeLessThan(i32, 1, 230);
            system[0] = @intCast(230 + which);
        }
        var prompt: std.ArrayList(i32) = .empty;
        defer prompt.deinit(gpa);
        var starts: std.ArrayList(usize) = .empty;
        defer starts.deinit(gpa);
        var record: [count]i32 = undefined;
        // each harness's first conversation, then a second one over each in the same order
        for (0..2) |round| for (0..many) |which| {
            prompt.clearRetainingCapacity();
            starts.clearRetainingCapacity();
            try starts.append(gpa, 0);
            try prompt.appendSlice(gpa, &systems[which]);
            try starts.append(gpa, prompt.items.len);
            try prompt.append(gpa, opener);
            for (0..24) |_| try prompt.append(gpa, random.random().intRangeLessThan(i32, 1, 230));
            const kept: ?usize = if (told != .plain) harness else null;
            const reply = try session.reply(prompt.items, greedy, .{ .max_tokens = count, .depth = 3, .starts = starts.items, .kept = kept, .record = &record }, counting.sink());
            fresh.trusted = false;
            var again: [count]i32 = undefined;
            const alone = try fresh.reply(prompt.items, greedy, .{ .max_tokens = count, .depth = 3, .record = &again }, counting.sink());
            try testing.expectEqualSlices(i32, &again, &record);
            try testing.expectEqual(alone.drafts, reply.drafts);
            try testing.expectEqualSlices(u16, fresh.logits, session.logits);
            if (round == 0) try testing.expect(reply.cached < harness);
            if (round == 1) switch (told) {
                .kept => try testing.expectEqual(@as(usize, harness), reply.cached),
                // four places: the first two harnesses' went
                .plain => if (which < 2) try testing.expect(reply.cached < harness),
                .capped => try testing.expect(reply.cached < harness),
            };
        };
        switch (told) {
            // the five harnesses the session does not stand on, each the last to hold its state
            .kept => {
                try testing.expectEqual(@as(usize, many - 1), session.lot.list.items.len);
                try testing.expectEqual(@as(usize, many - 1), session.lot.spared());
            },
            .plain => try testing.expectEqual(@as(usize, 4), session.lot.list.items.len),
            .capped => try testing.expectEqual(@as(usize, 0), session.lot.bytes()),
        }
    }
}

test "the lot's count: the oldest of those not the last to hold a harness goes past `most`, the oldest last holder past `harnesses` (spec 108)" {
    const gpa = testing.allocator;
    var lot: parked_file.Lot = .{ .gpa = gpa, .most = 4 };
    defer lot.deinit();
    // A sequence of no rows and no state: its tokens, the harness it holds the end of, when it was left.
    const add = struct {
        fn one(into: *parked_file.Lot, first: i32, shared: ?usize) !void {
            const tokens = try into.gpa.alloc(i32, 9);
            @memset(tokens, 1);
            tokens[0] = first;
            into.clock += 1;
            try into.list.append(into.gpa, .{ .tokens = tokens, .from = 8, .to = 8, .states = .empty, .spans = try into.gpa.alloc(parked_file.Span, 0), .head = null, .left = into.clock, .shared = shared });
        }
    }.one;
    // two conversations over harness 0 (the older is not its last holder), three of none
    try add(&lot, 0, 4);
    for (0..3) |_| try add(&lot, 100, null);
    try add(&lot, 0, 4);
    try testing.expect(!lot.last(0) and lot.last(4));
    try testing.expectEqual(@as(usize, 1), lot.spared());
    // four counted: room for one more takes the oldest of them, the first over harness 0
    lot.makeRoom();
    try testing.expectEqual(@as(usize, 4), lot.list.items.len);
    try testing.expectEqual(@as(i32, 100), lot.list.items[0].tokens[0]);
    // harnesses of their own, each its last holder: none counted, and past `harnesses` the oldest goes
    for (0..parked_file.harnesses + 2) |which| try add(&lot, @intCast(1000 + which), 4);
    lot.makeRoom();
    try testing.expectEqual(parked_file.harnesses - 1, lot.spared());
    try testing.expectEqual(@as(usize, 3), lot.list.items.len - lot.spared());
    // the harness of 0's sequence was among the oldest last holders, and went
    for (lot.list.items) |one| try testing.expect(one.tokens[0] != 0);
    try testing.expectEqual(@as(i32, 1000 + 3), lot.list.items[3].tokens[0]);
    // the victim for room is still one that is not a last holder
    try testing.expectEqual(@as(i32, 100), lot.list.items[lot.victim().?].tokens[0]);
}

test "the states kept for their buffers: every one while a reply runs, `states` of them past its end" {
    const gpa = testing.allocator;
    // a state without buffers frees nothing
    const empty: forward.Snapshot = .{ .pos = 0, .recurrent = &.{}, .conv = &.{}, .tails = &.{}, .history = &.{}, .gpa = gpa };
    var lot: parked_file.Lot = .{ .gpa = gpa, .most = 1 };
    defer lot.deinit();
    for (0..parked_file.states + 2) |_| lot.give(empty);
    try testing.expectEqual(parked_file.states + 2, lot.spare.count());
    lot.sweep();
    try testing.expectEqual(parked_file.states, lot.spare.count());
    try testing.expect(lot.lend() != null);
    try testing.expectEqual(@as(usize, 0), lot.spareBytes());
    // a lot that parks nothing keeps none
    var none: parked_file.Lot = .{ .gpa = gpa, .most = 0 };
    defer none.deinit();
    none.give(empty);
    try testing.expectEqual(@as(usize, 0), none.spare.count());
}
