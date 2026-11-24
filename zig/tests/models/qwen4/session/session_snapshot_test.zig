//! `session.zig`'s tests of turns and edits: which snapshots are held as a conversation grows, and where a
//! prompt that parts from the sequence runs from (the previous prompt's end, a message's start, the mark
//! before a prompt's end).

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const Counting = fixtures.sessions.Counting;
const deepest = fixtures.sessions.deepest;
const cuda = @import("../../../../src/cuda/cuda.zig");
const marks_file = @import("../../../../src/decode/marks.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const text_frontend = @import("../../../../src/text/frontend.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const draft = @import("../../../../src/models/qwen4/draft.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const session_file = @import("../../../../src/models/qwen4/session/session.zig");
const Options = session_file.Options;
const Reply = session_file.Reply;
const Session = session_file.Session;

/// The positions held after snapshots at `marks`, in order, each taken in a prompt `end` long, `budget` kept.
fn heldAfter(held: *std.ArrayList(usize), marks: []const usize, end: usize, budget: usize) void {
    for (marks) |mark| {
        held.appendAssumeCapacity(mark);
        if (held.items.len > budget) _ = held.orderedRemove(marks_file.thinned(held.items, end, null));
    }
}

test "snapshots thin out toward a sequence's start: the previous prompt's end outlives a new turn" {
    var buffer: [16]usize = undefined;
    var held: std.ArrayList(usize) = .initBuffer(&buffer);

    // A first prompt of 4054 tokens, then a second turn's marks, as they were taken before spec 026: the mark
    // two before the first prompt's end is the one an edited second turn starts from. The rule of then dropped
    // it, as the nearest to the one below it, and the edit ran 514 tokens again from 3584.
    heldAfter(&held, &.{ 512, 1024, 1536, 2048, 2560, 3072, 3584, 4052 }, 4054, 8);
    heldAfter(&held, &.{ 4096, 4098 }, 4100, 8);
    try testing.expectEqualSlices(usize, &.{ 1024, 2048, 2560, 3072, 3584, 4052, 4096, 4098 }, held.items);
    // and as they are taken now
    held.clearRetainingCapacity();
    heldAfter(&held, &.{ 512, 1024, 1536, 2048, 2560, 3072, 3542, 4052 }, 4054, 8);
    heldAfter(&held, &.{4098}, 4100, 8);
    try testing.expectEqualSlices(usize, &.{ 1024, 1536, 2048, 2560, 3072, 3542, 4052, 4098 }, held.items);

    // 16256 tokens in windows of 512, then a second turn: both turns' marks stay, and whatever token a later
    // prompt parts at, the tokens run again are within a stride of twice those that follow it.
    held.clearRetainingCapacity();
    var marks: [32]usize = undefined;
    for (marks[0..31], 1..) |*mark, i| mark.* = if (i == 31) 15744 else i * 512;
    marks[31] = 16254;
    heldAfter(&held, &marks, 16256, 8);
    heldAfter(&held, &.{16299}, 16301, 8);
    try testing.expect(std.mem.indexOfScalar(usize, held.items, 16254) != null);
    try testing.expect(std.mem.indexOfScalar(usize, held.items, 16299) != null);
    var worst: usize = 0;
    for (1..16301) |parted| {
        var below: usize = 0;
        for (held.items) |pos| {
            if (pos < parted) below = pos;
        }
        const again = parted - below;
        worst = @max(worst, again -| 2 * (16301 - parted));
    }
    try testing.expect(worst <= 512);
}

test "an edited last message starts from the previous prompt's end, after two turns and after three, and a short turn is one window" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // windows of up to 32 rows; four snapshots, a stride of 32
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const options: Options = .{ .window = 4, .capacity = 256, .snapshots = 4, .stride = 32 };
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, options);
    defer session.destroy();
    try testing.expectEqual(32, session.buffers.rows);
    // one that starts every prompt from an empty state: the head's drafts must be the same too
    const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 0 });
    defer fresh.destroy();

    const count = 12;
    var random: std.Random.DefaultPrng = .init(26);
    var first: [128]i32 = undefined;
    for (&first) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var message: [16]i32 = undefined;
    for (&message, 0..) |*token, i| token.* = @intCast(100 + i);
    var record: [count]i32 = undefined;
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const sampled: sampling.Sampling = .{ .seed = 9, .temperature = 0.8 };

    var prompt: std.ArrayList(i32) = .empty;
    defer prompt.deinit(gpa);
    // A turn keeps `keep` of the prompt before it, then, as Qwen's template does when a reply enters the
    // history, another last token, the reply and a new message; an edit keeps `keep` and adds other tokens.
    const Turn = struct {
        keep: usize,
        reply: bool,
        add: []const i32,
        settings: sampling.Sampling,
        cached: usize,
        windows: usize,
        held: []const usize,
    };
    const turns = [_]Turn{
        // 128 tokens met first: three whole windows, each ending where it crossed a stride, and the last,
        // with the mark two before the prompt's end taken inside it
        .{ .keep = 0, .reply = false, .add = &first, .settings = greedy, .cached = 0, .windows = 4, .held = &.{ 32, 64, 96, 126 } },
        // a second turn, 156 tokens: the 30 after the mark are one window, and its own mark is kept beside the
        // first prompt's, the oldest going
        .{ .keep = 127, .reply = true, .add = &message, .settings = sampled, .cached = 126, .windows = 1, .held = &.{ 64, 96, 126, 154 } },
        // its last message edited: from the first prompt's end, 126, not from a stride's 96
        .{ .keep = 148, .reply = false, .add = &.{ 7, 6, 5, 4, 3, 2, 1, 9 }, .settings = greedy, .cached = 126, .windows = 1, .held = &.{ 96, 126, 147, 154 } },
        // a third turn, 184 tokens, after the edited one
        .{ .keep = 155, .reply = true, .add = &message, .settings = greedy, .cached = 154, .windows = 1, .held = &.{ 96, 126, 154, 182 } },
        // and its last message edited: from the second prompt's end
        .{ .keep = 176, .reply = false, .add = &.{ 1, 2, 3, 4, 60, 61, 62, 63 }, .settings = sampled, .cached = 154, .windows = 1, .held = &.{ 96, 154, 175, 182 } },
        // edited again, alike for four tokens: from where the first edit parted
        .{ .keep = 180, .reply = false, .add = &.{ 70, 71, 72, 73 }, .settings = greedy, .cached = 175, .windows = 1, .held = &.{ 96, 154, 175, 182 } },
        // the first turn's message edited: from a stride's mark, and the rest in windows
        .{ .keep = 120, .reply = false, .add = &message, .settings = sampled, .cached = 96, .windows = 2, .held = &.{ 96, 119, 134 } },
    };
    for (turns) |turn| {
        prompt.shrinkRetainingCapacity(turn.keep);
        if (turn.reply) {
            try prompt.append(gpa, 251);
            try prompt.appendSlice(gpa, &record);
        }
        try prompt.appendSlice(gpa, turn.add);
        const expected = try fixtures.serial.serialReply(gpa, m, prompt.items, count, turn.settings);
        defer gpa.free(expected);
        const reply = try session.reply(prompt.items, turn.settings, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
        try testing.expectEqualSlices(i32, expected, &record);
        try testing.expectEqual(turn.cached, reply.cached);
        try testing.expectEqual(turn.windows, reply.windows);
        const held = try fixtures.sessions.snapshotPositions(session, gpa);
        defer gpa.free(held);
        try testing.expectEqualSlices(usize, turn.held, held);
        fresh.trusted = false;
        var again: [count]i32 = undefined;
        const whole = try fresh.reply(prompt.items, turn.settings, .{ .max_tokens = count, .depth = 3, .record = &again }, counting.sink());
        try testing.expectEqual(0, whole.cached);
        try testing.expectEqualSlices(i32, &again, &record);
        try testing.expectEqual(whole.drafts, reply.drafts);
        try testing.expectEqual(whole.stats.accepted, reply.stats.accepted);
    }
}

/// A conversation as a template would write it, for the tests: each message is an opener, a role, its content
/// and a closer; the prompt ends with the reply's header. `starts` are the openers' positions.
const Talk = struct {
    gpa: Allocator,
    tokens: std.ArrayList(i32) = .empty,
    starts: std.ArrayList(usize) = .empty,

    const opener = 251;
    const closer = 252;

    fn deinit(self: *Talk) void {
        self.tokens.deinit(self.gpa);
        self.starts.deinit(self.gpa);
    }

    fn message(self: *Talk, role: i32, content: []const i32) !void {
        try self.starts.append(self.gpa, self.tokens.items.len);
        try self.tokens.appendSlice(self.gpa, &.{ opener, role });
        try self.tokens.appendSlice(self.gpa, content);
        try self.tokens.append(self.gpa, closer);
    }

    /// The reply's header, which ends the prompt; `last` is its last token.
    fn header(self: *Talk, last: i32) !void {
        try self.starts.append(self.gpa, self.tokens.items.len);
        try self.tokens.appendSlice(self.gpa, &.{ opener, 9, last });
    }

    /// The reply enters the history as it was decoded, closed.
    fn said(self: *Talk, reply: []const i32) !void {
        try self.tokens.appendSlice(self.gpa, reply);
        try self.tokens.append(self.gpa, closer);
    }

    /// Back to the first `len` tokens.
    fn cut(self: *Talk, len: usize) void {
        self.tokens.shrinkRetainingCapacity(len);
        while (self.starts.items.len > 0 and self.starts.items[self.starts.items.len - 1] >= len) _ = self.starts.pop();
    }
};

test "snapshots stand at the messages' starts: an edited message runs again from its start, the last one and an earlier one" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // windows of up to 32 rows; six snapshots, a stride of 32, a start standing for the marks within 8 after it
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const options: Options = .{ .window = 4, .capacity = 256, .snapshots = 6, .stride = 32, .near = 8 };
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, options);
    defer session.destroy();
    // one that starts every prompt from an empty state: the head's drafts must be the same too
    const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 0 });
    defer fresh.destroy();

    const count = 12;
    var random: std.Random.DefaultPrng = .init(30);
    var words: [160]i32 = undefined;
    for (&words) |*token| token.* = random.random().intRangeLessThan(i32, 10, 250);
    var record: [count]i32 = undefined;
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const sampled: sampling.Sampling = .{ .seed = 9, .temperature = 0.8 };

    var talk: Talk = .{ .gpa = gpa };
    defer talk.deinit();
    const Step = enum { first, second, second_edited, third, third_edited, earlier_edited, left_out, first_edited, long, crossing, many };
    const Expected = struct { step: Step, settings: sampling.Sampling, cached: usize, windows: usize, taken: usize, held: []const usize };
    // What each request must find held, run and leave. The positions: a system block at 0, the first message
    // at 43, the reply's header at 66; a turn adds the reply (12 tokens), its closer, a message and a header.
    const steps = [_]Expected{
        // 69 tokens met first: the stride's mark inside the system block, then both starts, and the mark two
        // before the end, kept though the header's start is one below it
        .{ .step = .first, .settings = greedy, .cached = 0, .windows = 3, .taken = 4, .held = &.{ 32, 43, 66, 67 } },
        // a second turn, the reply as it was decoded: on from the live state, and the new message's start at 82,
        // its header's at 95 and the mark at 96 are taken inside the one window. One too many: the first
        // header's start goes, the mark above it serving what it served but a header written otherwise
        .{ .step = .second, .settings = sampled, .cached = 80, .windows = 1, .taken = 3, .held = &.{ 32, 43, 67, 82, 95, 96 } },
        // its message edited: from that message's start, after the reply, where it was the header before the
        // reply (66) until spec 030; the mark where it parted is within 8 of the start and not taken, and the
        // two it parted below give their buffers to the two it takes
        .{ .step = .second_edited, .settings = greedy, .cached = 82, .windows = 1, .taken = 2, .held = &.{ 32, 43, 67, 82, 95, 96 } },
        // a third turn: three more, and the second header's start, the oldest and the first prompt's mark go
        .{ .step = .third, .settings = greedy, .cached = 109, .windows = 1, .taken = 3, .held = &.{ 43, 82, 96, 111, 124, 125 } },
        .{ .step = .third_edited, .settings = sampled, .cached = 111, .windows = 1, .taken = 2, .held = &.{ 43, 82, 96, 111, 124, 125 } },
        // the second turn's message edited, the turns after it gone: from its start, still held
        .{ .step = .earlier_edited, .settings = greedy, .cached = 82, .windows = 1, .taken = 2, .held = &.{ 43, 82, 106, 107 } },
        // a turn whose reply the template writes otherwise than it was decoded (the header's last token is
        // another): from the mark before the prompt's end, one row past the header's start (106), which it
        // parted from until spec 041; the reply runs again as history. The header's start then stands for
        // the mark, and the place where the prompt parted, one past the mark, is not taken
        .{ .step = .left_out, .settings = sampled, .cached = 107, .windows = 1, .taken = 3, .held = &.{ 43, 82, 107, 122, 135, 136 } },
        // the first message edited: from its start, after the system block
        .{ .step = .first_edited, .settings = greedy, .cached = 43, .windows = 2, .taken = 2, .held = &.{ 43, 78, 79 } },
        // another conversation, one message of 153 tokens: the stride's marks inside it, where a window that
        // crossed a multiple of 32 ends, the header's start and the mark
        .{ .step = .long, .settings = sampled, .cached = 0, .windows = 5, .taken = 5, .held = &.{ 32, 64, 96, 153, 154 } },
        // a message of 20 tokens and one of 60: the window that crosses 32 holds the second's start at 23, and
        // the stride's mark at 32 is not taken beside it
        .{ .step = .crossing, .settings = greedy, .cached = 0, .windows = 3, .taken = 3, .held = &.{ 23, 86, 87 } },
        // and one of fourteen messages, nine tokens each: six states are taken, not fifteen, at starts but the
        // mark before the end
        .{ .step = .many, .settings = greedy, .cached = 0, .windows = 5, .taken = 6, .held = &.{} },
    };
    for (steps) |expected| {
        switch (expected.step) {
            .first => {
                try talk.message(1, words[0..40]);
                try talk.message(2, words[40..60]);
                try talk.header(9);
            },
            .second, .third => {
                try talk.said(&record);
                try talk.message(2, words[60..70]);
                try talk.header(9);
            },
            .second_edited => {
                // alike for three tokens of its content
                talk.cut(82 + 2 + 3);
                try talk.tokens.appendSlice(gpa, words[80..87]);
                try talk.tokens.append(gpa, Talk.closer);
                try talk.header(9);
            },
            .third_edited => {
                talk.cut(111 + 2 + 4);
                try talk.tokens.appendSlice(gpa, words[90..96]);
                try talk.tokens.append(gpa, Talk.closer);
                try talk.header(9);
            },
            .earlier_edited => {
                talk.cut(82 + 2 + 1);
                try talk.tokens.appendSlice(gpa, words[100..120]);
                try talk.tokens.append(gpa, Talk.closer);
                try talk.header(9);
            },
            .left_out => {
                // the header ends otherwise in the history than it did before the reply
                talk.tokens.items[talk.tokens.items.len - 1] = 8;
                try talk.said(&record);
                try talk.message(2, words[60..70]);
                try talk.header(9);
            },
            .first_edited => {
                talk.cut(43 + 2 + 2);
                try talk.tokens.appendSlice(gpa, words[120..150]);
                try talk.tokens.append(gpa, Talk.closer);
                try talk.header(9);
            },
            .long => {
                talk.cut(0);
                try talk.message(3, words[0..150]);
                try talk.header(9);
            },
            .crossing => {
                talk.cut(0);
                try talk.message(5, words[0..20]);
                try talk.message(6, words[20..80]);
                try talk.header(9);
            },
            .many => {
                talk.cut(0);
                for (0..14) |i| try talk.message(4, words[i * 6 ..][0..6]);
                try talk.header(9);
            },
        }
        const prompt = talk.tokens.items;
        const serial = try fixtures.serial.serialReply(gpa, m, prompt, count, expected.settings);
        defer gpa.free(serial);
        const reply = try session.reply(prompt, expected.settings, .{ .max_tokens = count, .depth = 3, .record = &record, .starts = talk.starts.items }, counting.sink());
        errdefer std.debug.print("step {t}\n", .{expected.step});
        try testing.expectEqualSlices(i32, serial, &record);
        try testing.expectEqual(expected.cached, reply.cached);
        try testing.expectEqual(expected.windows, reply.windows);
        try testing.expectEqual(expected.taken, reply.taken);
        const held = try fixtures.sessions.snapshotPositions(session, gpa);
        defer gpa.free(held);
        if (expected.step == .many) {
            // what taking every start in turn and thinning would have left: the last ones dense, older ones thin
            var buffer: [8]usize = undefined;
            var kept: std.ArrayList(usize) = .initBuffer(&buffer);
            heldAfter(&kept, talk.starts.items[1..], prompt.len, 6);
            heldAfter(&kept, &.{prompt.len - 2}, prompt.len, 6);
            try testing.expectEqualSlices(usize, kept.items, held);
            try testing.expectEqualSlices(usize, &.{ 127, 126, 117 }, &.{ held[5], held[4], held[3] });
            for (held[0..5]) |pos| try testing.expect(std.mem.indexOfScalar(usize, talk.starts.items, pos) != null);
        } else try testing.expectEqualSlices(usize, expected.held, held);
        fresh.trusted = false;
        var again: [count]i32 = undefined;
        const whole = try fresh.reply(prompt, expected.settings, .{ .max_tokens = count, .depth = 3, .record = &again, .starts = talk.starts.items }, counting.sink());
        try testing.expectEqual(0, whole.cached);
        try testing.expectEqual(0, whole.taken);
        try testing.expectEqualSlices(i32, &again, &record);
        try testing.expectEqual(whole.drafts, reply.drafts);
        try testing.expectEqual(whole.stats.accepted, reply.stats.accepted);
    }
}

/// The committed state's bits: each DeltaNet layer's state on its current side and conv inputs, the PLE tails
/// and the position. Two sessions that hold the same sequence hold the same, however each came to it.
/// Every snapshot a session holds, as bytes: each one's DeltaNet states, conv inputs, PLE tails and histories.
fn heldBytes(gpa: Allocator, session: *const Session) ![]u8 {
    try cuda.synchronize();
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (session.snapshots.items) |one| {
        for ([_][]cuda.Buffer{ one.recurrent, one.conv, one.tails }) |list| {
            for (list) |buffer| try buffer.download(try out.addManyAsSlice(gpa, buffer.bytes()));
        }
        for (one.history) |h| try out.appendSlice(gpa, std.mem.sliceAsBytes(h));
        try out.appendSlice(gpa, std.mem.asBytes(&one.pos));
    }
    return out.toOwnedSlice(gpa);
}

fn committedBytes(gpa: Allocator, session: *const Session) ![]u8 {
    try cuda.synchronize();
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    const state = session.state;
    for (state.recurrent, state.conv, 0..) |pair, conv, linear| {
        // the state at `pos`: the committed side, replayed over the rows it is behind by
        var at: cuda.Buffer = try forward.Snapshot.likeOf(pair[0]);
        defer at.deinit();
        try forward.committed(state, linear, at);
        for ([_]cuda.Buffer{ at, conv }) |buffer| try buffer.download(try out.addManyAsSlice(gpa, buffer.bytes()));
    }
    for (state.ple) |branch| {
        try branch.tail.download(try out.addManyAsSlice(gpa, branch.tail.bytes()));
        try out.appendSlice(gpa, std.mem.sliceAsBytes(branch.history));
    }
    try out.appendSlice(gpa, std.mem.asBytes(&state.pos));
    return out.toOwnedSlice(gpa);
}

test "the mark before a prompt's end is kept beside the header's start: the prompt sent again runs two rows from it, and its reply and state are those run from the start" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // prompts in windows of up to 384 rows, where the shared scratch holds a state (`forward.defers`), replies
    // in windows of up to 16
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 384);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const window = draft.forward_max_rows;
    // as served, but a start stands for the marks within 8 after it and not 32: all but the mark two before the end
    // no sequence is left (`parked`): the conversation cut back would give the lot three of the states counted here
    const now: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = window, .capacity = 512, .near = 8, .parked = 0 });
    defer now.destroy();
    try now.prepare(window - 1, 1);
    // the same with every state replayed inside its window's forward, and none after the first token
    const inside: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = window, .capacity = 512, .near = 8, .settle = false });
    defer inside.destroy();
    // as it was until spec 041: no mark before the end, so the header's start is the nearest state
    const was: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = window, .capacity = 512, .near = 8, .tail = 0 });
    defer was.destroy();
    // and one that starts every prompt from an empty state
    const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = window, .capacity = 512, .snapshots = 0 });
    defer fresh.destroy();

    const count = 12;
    var random: std.Random.DefaultPrng = .init(41);
    var words: [200]i32 = undefined;
    for (&words) |*token| token.* = random.random().intRangeLessThan(i32, 10, 250);
    var record: [count]i32 = undefined;
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const sampled: sampling.Sampling = .{ .seed = 9, .temperature = 0.8 };

    var talk: Talk = .{ .gpa = gpa };
    defer talk.deinit();
    const Step = enum { first, again, third, turn, edited, edited_again, short, cut_back, long };
    // `cached` and `before`: where the prompt runs from with the mark and without; `unused`: the states left
    // waiting with their buffers after it.
    const Expected = struct { step: Step, settings: sampling.Sampling, cached: usize, before: usize, taken: usize, graphed: bool, held: []const usize, unused: usize = 0, settled: usize = 0 };
    // `settled`: the states of the prompt's last window that got their DeltaNet part after the first token.
    // A system block at 0, the first message at 43, the reply's header at 96, the prompt's end at 99.
    const steps = [_]Expected{
        // met first, in one window of 99 rows: the message's start, the header's and the mark one row above
        // it, each replayed on from the one below, after the reply's first token
        .{ .step = .first, .settings = greedy, .cached = 0, .before = 0, .taken = 3, .graphed = false, .held = &.{ 43, 96, 97 }, .settled = 3 },
        // sent again: two rows where it ran three, on the graph of two rows; the place where it parted, one
        // past the mark, is not taken: the header's start stands for it as it did
        .{ .step = .again, .settings = sampled, .cached = 97, .before = 96, .taken = 0, .graphed = true, .held = &.{ 43, 96, 97 } },
        .{ .step = .third, .settings = greedy, .cached = 97, .before = 96, .taken = 0, .graphed = true, .held = &.{ 43, 96, 97 } },
        // the next turn, whose template writes the header's last token otherwise: from the mark, the reply
        // run again as history, in one wide window that holds the new message's start (112), its header's
        // (139) and the mark (140)
        .{ .step = .turn, .settings = sampled, .cached = 97, .before = 96, .taken = 3, .graphed = false, .held = &.{ 43, 96, 97, 112, 139, 140 }, .settled = 3 },
        // that message edited after its third token: from its start as before, and the two states it parts
        // below give their buffers to the two it takes
        .{ .step = .edited, .settings = greedy, .cached = 112, .before = 112, .taken = 2, .graphed = false, .held = &.{ 43, 96, 97, 112, 139, 140 }, .settled = 2 },
        .{ .step = .edited_again, .settings = sampled, .cached = 140, .before = 139, .taken = 0, .graphed = true, .held = &.{ 43, 96, 97, 112, 139, 140 } },
        // a turn of three words after the reply as it was decoded: eleven rows from the live state, one
        // narrow window on its graph, whose states are taken once it has run; the header's start (161) is
        // near the message's and left out, the mark is not
        .{ .step = .short, .settings = greedy, .cached = 153, .before = 153, .taken = 2, .graphed = true, .held = &.{ 43, 96, 97, 112, 139, 140, 155, 162 } },
        // the conversation cut back to its first prompt: the mark kept since the first step, and the five
        // states above it wait with their buffers
        .{ .step = .cut_back, .settings = sampled, .cached = 97, .before = 96, .taken = 0, .graphed = true, .held = &.{ 43, 96, 97 }, .unused = 5 },
        // another conversation: its three states are written into three of those, inside the forward, a
        // window of 153 rows being more than a state's buffer holds of the chain's scratch (150)
        .{ .step = .long, .settings = greedy, .cached = 0, .before = 0, .taken = 3, .graphed = false, .held = &.{ 22, 150, 151 }, .unused = 5 },
    };
    var bytes: usize = 0;
    for (steps) |expected| {
        switch (expected.step) {
            .first => {
                try talk.message(1, words[0..40]);
                try talk.message(2, words[40..90]);
                try talk.header(9);
            },
            .again, .third, .edited_again => {},
            .turn => {
                talk.tokens.items[talk.tokens.items.len - 1] = 8;
                try talk.said(&record);
                try talk.message(2, words[90..114]);
                try talk.header(9);
            },
            .edited => {
                talk.cut(112 + 2 + 3);
                try talk.tokens.appendSlice(gpa, words[150..171]);
                try talk.tokens.append(gpa, Talk.closer);
                try talk.header(9);
            },
            .short => {
                try talk.said(&record);
                try talk.message(2, words[130..133]);
                try talk.header(9);
            },
            .cut_back => {
                talk.cut(96);
                try talk.header(9);
            },
            .long => {
                talk.cut(0);
                try talk.message(5, words[0..19]);
                try talk.message(6, words[20..145]);
                try talk.header(9);
            },
        }
        const prompt = talk.tokens.items;
        errdefer std.debug.print("step {t}\n", .{expected.step});
        const serial = try fixtures.serial.serialReply(gpa, m, prompt, count, expected.settings);
        defer gpa.free(serial);
        var old: [count]i32 = undefined;
        var cold: [count]i32 = undefined;
        // the same reply, whichever state it was resumed from
        const reply = try now.reply(prompt, expected.settings, .{ .max_tokens = count, .depth = deepest, .record = &record, .starts = talk.starts.items }, counting.sink());
        const before = try was.reply(prompt, expected.settings, .{ .max_tokens = count, .depth = deepest, .record = &old, .starts = talk.starts.items }, counting.sink());
        fresh.trusted = false;
        const whole = try fresh.reply(prompt, expected.settings, .{ .max_tokens = count, .depth = deepest, .record = &cold, .starts = talk.starts.items }, counting.sink());
        try testing.expectEqualSlices(i32, serial, &record);
        try testing.expectEqualSlices(i32, &old, &record);
        try testing.expectEqualSlices(i32, &cold, &record);
        for ([_]Reply{ before, whole }) |other| {
            try testing.expectEqual(other.drafts, reply.drafts);
            try testing.expectEqual(other.stats, reply.stats);
        }
        // and the same state after it, bit for bit
        const state = try committedBytes(gpa, now);
        defer gpa.free(state);
        for ([_]*const Session{ was, fresh }) |other| {
            const theirs = try committedBytes(gpa, other);
            defer gpa.free(theirs);
            try testing.expectEqualSlices(u8, theirs, state);
        }
        try testing.expectEqual(expected.cached, reply.cached);
        try testing.expectEqual(expected.before, before.cached);
        try testing.expectEqual(0, whole.cached);
        try testing.expectEqual(expected.taken, reply.taken);
        try testing.expectEqual(expected.graphed, reply.graphed);
        try testing.expectEqual(expected.settled, reply.settled);
        // the states it holds are those replayed inside the forward, bit for bit
        var warm: [count]i32 = undefined;
        const within = try inside.reply(prompt, expected.settings, .{ .max_tokens = count, .depth = deepest, .record = &warm, .starts = talk.starts.items }, counting.sink());
        try testing.expectEqualSlices(i32, &warm, &record);
        try testing.expectEqual(0, within.settled);
        try testing.expectEqual(reply.cached, within.cached);
        const kept = try heldBytes(gpa, now);
        defer gpa.free(kept);
        const theirs = try heldBytes(gpa, inside);
        defer gpa.free(theirs);
        try testing.expectEqualSlices(u8, theirs, kept);
        const held = try fixtures.sessions.snapshotPositions(now, gpa);
        defer gpa.free(held);
        try testing.expectEqualSlices(usize, expected.held, held);
        try testing.expectEqual(expected.unused, now.unused.items.len);
        // No step frees a state's buffers, and none allocates where one waits: the bytes held only grow, by
        // a state for each one taken beyond those waiting.
        const one = now.snapshots.items[0].bytes();
        try testing.expectEqual(one * (held.len + expected.unused), now.snapshotBytes());
        try testing.expect(now.snapshotBytes() >= bytes and now.snapshotBytes() <= one * now.options.snapshots);
        bytes = now.snapshotBytes();
    }
}

test "a served conversation's snapshots stand where its template opens a message" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer m.deinit();
    // every message and the reply's header open with `<|end|>`, which closes none
    const source = "<|end|>system be brief\n{% for message in messages %}<|end|>{{ message.role }} {{ message.content }}\n{% endfor %}<|end|>assistant ";
    var template_diagnostic: @import("../../../../src/text/template.zig").Diagnostic = .{};
    const parsed = try @import("../../../../src/text/template.zig").Template.parse(gpa, source, &template_diagnostic);
    var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
    const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
    var frontend: text_frontend.Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
    defer frontend.deinit(gpa);
    const opener = try frontend.probeOpener(gpa);
    try testing.expect(opener != null);
    frontend.openers = &.{opener.?};

    // every start is kept, however near the one before it
    const session: *Session = try .create(gpa, testing.io, m, &frontend.tokenizer, &.{}, .{ .window = 4, .capacity = 256, .near = 1 });
    defer session.destroy();
    var counting: Counting = .{};
    const body =
        \\[{"role":"user","content":"the cat sat on the mat"},{"role":"assistant","content":"and then"},
        \\ {"role":"user","content":"the dog ran over the hill and far away"}]
    ;
    const list = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), body, .{});
    var text_diagnostic: text_frontend.Diagnostic = .{};
    const ids = try frontend.prompt(arena.allocator(), .{ .messages = list, .thinking = false }, &text_diagnostic);
    const prompt = try arena.allocator().alloc(i32, ids.len);
    for (prompt, ids) |*token, id| token.* = @intCast(id);
    // as `served.zig` asks a reply of a session: the prompt's ids with where the template opens a message
    const starts = try frontend.starts(arena.allocator(), prompt);
    const reply = try session.reply(prompt, .{ .seed = 0, .temperature = 0 }, .{ .max_tokens = 6, .depth = 3, .starts = starts }, counting.sink());
    try testing.expectEqual(0, reply.cached);

    // the system block's, the three messages' and the header's
    try testing.expectEqual(5, starts.len);
    try testing.expectEqual(0, starts[0]);
    const held = try fixtures.sessions.snapshotPositions(session, gpa);
    defer gpa.free(held);
    // the header's is two tokens or fewer from the prompt's end here, so the mark there is the same or follows
    var expected: std.ArrayList(usize) = .empty;
    try expected.appendSlice(arena.allocator(), starts[1..]);
    if (prompt.len - 2 > starts[4]) try expected.append(arena.allocator(), prompt.len - 2);
    try testing.expectEqualSlices(usize, expected.items, held);
}
