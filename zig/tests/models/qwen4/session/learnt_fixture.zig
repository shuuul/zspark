//! What `learnt_test.zig` and `checkpoint_prefix_test.zig` share (spec 133 WS-02 moved it here): conversations
//! over a harness held to a session that runs each prompt whole (`fx.Bench`), a state written by one session
//! and read by the next (`readBack`), and the states that leave a session read by the next load's (`spiltBack`).

const std = @import("std");
const testing = std.testing;
const fixtures = @import("../testing.zig");
const Counting = fixtures.sessions.Counting;
const sampling = @import("../../../../src/decode/sampling.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const learnt = @import("../../../../src/models/qwen4/session/learnt.zig");
const learned = @import("../../../../src/decode/learned.zig");
const learned_blocks = @import("../../../../src/decode/learned_blocks.zig");
const model = @import("../../../../src/models/qwen4/model.zig");
const session_file = @import("../../../../src/models/qwen4/session/session.zig");
const Session = session_file.Session;

pub const count = 12;
pub const harness = 60;
/// A message starts with this token, as a template's opener.
pub const opener = 252;
pub const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };

pub const fx = struct {
    pub const options: session_file.Options = .{ .window = 4, .capacity = 512, .snapshots = 4, .stride = 0, .near = 4 };

    /// Conversations over a harness, each prompt checked against a session that runs it whole.
    pub const Bench = struct {
        fresh: *Session,
        counting: Counting = .{},
        random: std.Random.DefaultPrng = .init(641),
        system: [harness]i32 = undefined,
        prompt: std.ArrayList(i32) = .empty,
        starts: std.ArrayList(usize) = .empty,
        record: [count]i32 = undefined,

        pub fn init(m: model.Model, tokens: *const tokenizer.Tokenizer) !Bench {
            var self: Bench = .{ .fresh = try .create(testing.allocator, testing.io, m, tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = 0 }) };
            for (&self.system) |*token| token.* = self.random.random().intRangeLessThan(i32, 1, 250);
            return self;
        }

        pub fn deinit(self: *Bench) void {
            self.fresh.destroy();
            self.prompt.deinit(testing.allocator);
            self.starts.deinit(testing.allocator);
        }

        /// A conversation's first prompt, over the harness or with none.
        pub fn begin(self: *Bench, over: bool) !void {
            self.prompt.clearRetainingCapacity();
            self.starts.clearRetainingCapacity();
            if (over) {
                try self.starts.append(testing.allocator, 0);
                try self.prompt.appendSlice(testing.allocator, &self.system);
            }
            try self.message();
        }

        /// The last reply and a new message after the prompt.
        pub fn turn(self: *Bench) !void {
            try self.starts.append(testing.allocator, self.prompt.items.len);
            try self.prompt.append(testing.allocator, opener);
            try self.prompt.appendSlice(testing.allocator, &self.record);
            try self.message();
        }

        pub fn message(self: *Bench) !void {
            try self.starts.append(testing.allocator, self.prompt.items.len);
            try self.prompt.append(testing.allocator, opener);
            for (0..24) |_| try self.prompt.append(testing.allocator, self.random.random().intRangeLessThan(i32, 1, 240));
        }

        /// The prompt on `session`, held to a pass over the whole of it: ids, drafts and the last row's
        /// logits. Returns the tokens the session did not run.
        pub fn ask(self: *Bench, session: *Session, kept: ?usize) !usize {
            const reply = try session.reply(self.prompt.items, greedy, .{ .max_tokens = count, .depth = 3, .starts = self.starts.items, .kept = kept, .record = &self.record }, self.counting.sink());
            self.fresh.trusted = false;
            var again: [count]i32 = undefined;
            const alone = try self.fresh.reply(self.prompt.items, greedy, .{ .max_tokens = count, .depth = 3, .record = &again }, self.counting.sink());
            try testing.expectEqualSlices(i32, &again, &self.record);
            try testing.expectEqual(alone.drafts, reply.drafts);
            try testing.expectEqualSlices(u16, self.fresh.logits, session.logits);
            return reply.cached;
        }
    };
};

pub fn readBack(m: model.Model) !void {
    const gpa = testing.allocator;
    const io = testing.io;
    const worth, const block = .{ learnt.worth, learned_blocks.block_tokens };
    defer learnt.worth, learned_blocks.block_tokens = .{ worth, block };
    // the harness's 60 positions: three blocks of 16, and its rows past them in the state's file
    learnt.worth, learned_blocks.block_tokens = .{ 16, 16 };
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: learnt.Store = try .open(gpa, io, tmp.dir, 1, 1 << 30);
    defer store.close();
    var bench: fx.Bench = try .init(m, &tokens);
    defer bench.deinit();

    {
        // the load that learns: the harness's state is written after its first reply, and once
        const first: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer first.destroy();
        first.learned = &store;
        try bench.begin(true);
        try testing.expectEqual(0, try bench.ask(first, harness));
        try testing.expectEqual(1, store.written);
        try bench.turn();
        _ = try bench.ask(first, harness);
        try bench.begin(true);
        try testing.expectEqual(harness, try bench.ask(first, harness));
        try testing.expectEqual(1, store.written);
        try testing.expectEqual(0, store.read);
    }
    {
        // the load after: a new conversation starts where the harness ends, and goes on as any other
        const second: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer second.destroy();
        second.learned = &store;
        try bench.begin(true);
        try testing.expectEqual(harness, try bench.ask(second, harness));
        try testing.expectEqual(1, store.read);
        try bench.turn();
        try testing.expect(try bench.ask(second, harness) > harness);
        // the state is the session's own now
        try bench.begin(true);
        try testing.expectEqual(harness, try bench.ask(second, harness));
        try testing.expectEqual(1, store.read);
        try testing.expectEqual(1, store.written);
    }
    {
        // a session that holds another conversation reads it too, and that conversation goes on after: the
        // rows the file wrote over are the lot's
        const third: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer third.destroy();
        third.learned = &store;
        var other: fx.Bench = try .init(m, &tokens);
        defer other.deinit();
        other.random = .init(77);
        try other.begin(false);
        _ = try other.ask(third, null);
        try other.turn();
        _ = try other.ask(third, null);
        try bench.begin(true);
        try testing.expectEqual(harness, try bench.ask(third, harness));
        try testing.expectEqual(2, store.read);
        try other.turn();
        try testing.expect(try other.ask(third, null) > harness);
        try bench.turn();
        try testing.expect(try bench.ask(third, harness) > harness);
    }
    {
        // not told where the harness ends, a session reads nothing; nor one of another identity
        const untold: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer untold.destroy();
        untold.learned = &store;
        try bench.begin(true);
        try testing.expectEqual(0, try bench.ask(untold, null));
        var elsewhere: learnt.Store = try .open(gpa, io, tmp.dir, 2, 1 << 30);
        defer elsewhere.close();
        const another: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer another.destroy();
        another.learned = &elsewhere;
        try bench.begin(true);
        try testing.expectEqual(0, try bench.ask(another, harness));
        try testing.expectEqual(0, elsewhere.read);
        try testing.expectEqual(1, elsewhere.written);
    }
    {
        // a file of the right tokens whose parts are another model's is left unread, before a row is written
        var system: [harness + 1]i32 = undefined;
        @memcpy(system[0..harness], &bench.system);
        system[harness] = opener;
        store.discard(&system);
        try store.putHost(&system, &.{ "abcdefgh", "ijklmnop" }, &.{});
        const fourth: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer fourth.destroy();
        fourth.learned = &store;
        try bench.begin(true);
        try testing.expectEqual(0, try bench.ask(fourth, harness));
        try bench.begin(true);
        try testing.expectEqual(harness, try bench.ask(fourth, harness));
    }
}

/// Six conversations of two turns on a session that keeps four beside its own: the ones that go for room are
/// written as they go, the rest when the server stops. A session of another load then takes each one's next
/// turn from its file, with the reply a pass over the whole prompt gives; with the flag off nothing is
/// written.
pub fn spiltBack(m: model.Model) !void {
    const gpa = testing.allocator;
    const io = testing.io;
    const spill = @import("../../../../src/decode/spill.zig");
    const floor, const worth, const block = .{ spill.floor, spill.worth, learned_blocks.block_tokens };
    defer spill.floor, spill.worth, learned_blocks.block_tokens = .{ floor, worth, block };
    spill.floor, spill.worth, learned_blocks.block_tokens = .{ 16, 16, 16 };
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: learnt.Store = try .open(gpa, io, tmp.dir, 1, 1 << 30);
    defer store.close();
    const conversations = 6;
    var benches: [conversations]fx.Bench = undefined;
    var made: usize = 0;
    defer for (benches[0..made]) |*bench| bench.deinit();
    for (&benches, 0..) |*bench, i| {
        bench.* = try .init(m, &tokens);
        made += 1;
        bench.random = .init(900 + i);
    }
    // where the last message of a conversation's second prompt starts, and how long that prompt is
    var last: usize = 0;
    var asked: usize = 0;
    {
        const first: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer first.destroy();
        learnt.spilling(first, &store);
        for (&benches) |*bench| {
            try bench.begin(false);
            _ = try bench.ask(first, null);
            try bench.turn();
            _ = try bench.ask(first, null);
            last = bench.starts.items[bench.starts.items.len - 1];
            asked = bench.prompt.items.len;
        }
        // the first went when the sixth was left; a state is one file, written once
        try testing.expectEqual(1, store.written);
        learnt.flush(first, true);
        try testing.expectEqual(conversations, store.written);
        learnt.flush(first, true);
        try testing.expectEqual(conversations, store.written);
        try testing.expectEqual(0, store.read);
    }
    // the highest state of the prompt: its last message's start, or the mark before its end
    const at = store.longest(benches[0].prompt.items, asked).? - 1;
    try testing.expect(at >= last and at < asked and last >= spill.floor);
    {
        const second: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer second.destroy();
        learnt.spilling(second, &store);
        for (&benches, 1..) |*bench, n| {
            try bench.turn();
            try testing.expectEqual(at, try bench.ask(second, null));
            try testing.expectEqual(n, store.read);
        }
        // a prompt the session holds itself reads nothing
        try benches[conversations - 1].turn();
        try testing.expect(try benches[conversations - 1].ask(second, null) > at);
        try testing.expectEqual(conversations, store.read);
    }
    {
        // the flag off: nothing is read, and nothing more written
        const off: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer off.destroy();
        try testing.expectEqual(0, try benches[0].ask(off, null));
        learnt.flush(off, true);
        try testing.expectEqual(conversations, store.read);
    }
    {
        // a file cut short is deleted, and the prompt is run whole with the same reply
        const prompt = benches[1].prompt.items;
        const len = store.longest(prompt, prompt.len).?;
        const name = learned.nameOf(learnt.keyOf(prompt[0..len]));
        const file = try store.dir.openFile(io, &name, .{ .mode = .read_write });
        try file.setLength(io, try file.length(io) - 1);
        file.close(io);
        const third: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer third.destroy();
        learnt.spilling(third, &store);
        _ = try benches[1].ask(third, null);
        try testing.expect(!store.has(prompt[0..len]));
    }
}
