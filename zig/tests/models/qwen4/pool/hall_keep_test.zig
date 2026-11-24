//! Tests of the states a pool's stream keeps (`hall_keep.zig`, spec 054): each one's bytes against a slot
//! run a row at a time, a reply from each, the rows beside one taken, a round cut for the reasoning's end,
//! the pool's rows run again beside a session's, and the store's bounds.

const std = @import("std");
const fixtures = @import("../testing.zig");
const keep_fixture = @import("hall_keep_fixture.zig");
const Collect = keep_fixture.Collect;
const Shape = keep_fixture.Shape;
const open = keep_fixture.open;
const step = keep_fixture.step;
const ask = keep_fixture.ask;
const expectLeading = keep_fixture.expectLeading;
const expectSerial = keep_fixture.expectSerial;
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const prefix = @import("../../../../src/decode/prefix.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const draft = @import("../../../../src/models/qwen4/draft.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const keep_file = @import("../../../../src/models/qwen4/pool/hall_keep.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const session_file = @import("../../../../src/models/qwen4/session/session.zig");
const Engine = hall_engine.Engine;
const fetchBytes = fixtures.attending.fetchBytes;

const greedy: sampling.Sampling = .{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
const sampled = keep_fixture.sampled;

const Got = keep_fixture.Got;
const one = keep_fixture.one;
const countOf = keep_fixture.countOf;
const positions = keep_fixture.positions;
const randomTokens = keep_fixture.randomTokens;
const starts_of_150 = keep_fixture.starts_of_150;
const keptAsSerial = keep_fixture.keptAsSerial;

test "a pool stream's kept states are a serial slot's: taken within a round and by chunks cut at them, a wide chunk and a narrow one, a reply beside, with a PLE and trunk attention" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    // 40 rows a window: the chain's wide kernels; 24: the slots' kernel
    try keptAsSerial(gpa, m, .{ .prompt_rows = 40 }, false);
    try keptAsSerial(gpa, m, .{ .prompt_rows = 24 }, false);
    try keptAsSerial(gpa, m, .{ .slots = 2, .prompt_rows = 40, .chunk = 20 }, true);
}

test "a pool stream's kept states are a serial slot's on each cache storage: the head's pooled keys and its length" {
    const gpa = testing.allocator;
    defer fixtures.random_model.caches = .bf16;
    for ([_]attention.Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
        defer m.deinit();
        try keptAsSerial(gpa, m, .{ .prompt_rows = 48 }, false);
        try keptAsSerial(gpa, m, .{ .slots = 2, .prompt_rows = 48, .chunk = 20 }, true);
    }
}

/// After a prompt and its reply on `m`: a prompt that goes on from each kept state with other tokens finds
/// that state, and its reply is the one a row at a time gives.
fn repliesFromEach(gpa: Allocator, m: model_file.Model, shape: Shape) !void {
    var random: std.Random.DefaultPrng = .init(55);
    var prompt: [150]i32 = undefined;
    randomTokens(random.random(), &prompt);
    var engine = try open(gpa, m, shape);
    engine.bindPool();
    defer engine.deinit();
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    const first = try one(&engine, &collect, ask(&prompt, sampled, 44, &starts_of_150, null));
    defer gpa.free(first.tokens);
    var sequence: std.ArrayList(i32) = .empty;
    defer sequence.deinit(gpa);
    try sequence.appendSlice(gpa, &prompt);
    try sequence.appendSlice(gpa, first.tokens);
    const held = try positions(gpa, &engine, true);
    defer gpa.free(held);
    try testing.expect(held.len >= 6);
    for ([_]prefix.Kind{ .chunk, .message, .tail, .stride, .end }) |kind| try testing.expect(countOf(&engine, kind) >= 1);
    // a pool that holds nothing: its drafts are the ones a state's head must give
    var empty = try open(gpa, m, shape);
    empty.bindPool();
    defer empty.deinit();
    const first_chain = engine.pool.chains.items[0];
    const tail = [_]i32{ 251, 252, 7, 9 };
    for (held) |pos| {
        errdefer std.debug.print("from the state at {d}\n", .{pos});
        var next: std.ArrayList(i32) = .empty;
        defer next.deinit(gpa);
        // the token the head took in there, then tokens the sequence does not have
        try next.appendSlice(gpa, sequence.items[0 .. pos + 1]);
        try next.appendSlice(gpa, &tail);
        const cold = try fixtures.serial.serialReply(gpa, m, next.items, 10, sampled);
        defer gpa.free(cold);
        var reports: [2]hall_file.Report = undefined;
        for ([_]*Engine{ &empty, &engine }, &reports) |pool, *report| {
            collect.clear(0);
            try testing.expectEqual(if (pool == &engine) pos else 0, try pool.begin(0, ask(next.items, sampled, 10, &.{}, null), 0));
            if (pool == &engine) {
                // the slot's pooled keys below the state are the ones kept with it
                const keep = engine.keeps.at(first_chain, pos).?;
                const slot = engine.slots[0];
                for (keep.pooled.list, 0..) |pooled, i| {
                    if (pooled.view.ptr == null) continue;
                    try expectLeading(gpa, if (i < slot.caches.len) slot.caches[i].pooled else slot.head.?.pooled, pooled);
                }
            }
            var spins: usize = 0;
            while (!collect.ended[0]) : (spins += 1) {
                if (spins == 256) return error.Stuck;
                try step(pool, &collect);
            }
            try testing.expectEqualSlices(i32, cold, collect.lists[0].items);
            // its chain and its states go: the first request's are what is asked
            report.* = pool.retire(0, .failed, 0).?;
        }
        try testing.expectEqual(reports[0].drafts, reports[1].drafts);
        try testing.expectEqual(reports[0].drafted, reports[1].drafted);
        try testing.expectEqual(reports[0].accepted, reports[1].accepted);
    }
}

test "a reply from each state a pool stream kept is the cold reply: a prompt's, a chunk's, a stride's and the end's, the head's cache and the trunk's restored" {
    const gpa = testing.allocator;
    defer fixtures.random_model.caches = .bf16;
    for ([_]attention.Storage{ .bf16, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
        defer m.deinit();
        try repliesFromEach(gpa, m, .{ .prompt_rows = 48 });
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    try repliesFromEach(gpa, m, .{ .prompt_rows = 40 });
}

test "a seat's rows do not change when the seat beside it takes states in the round: its reply and its end are the ones it has alone" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var random: std.Random.DefaultPrng = .init(56);
    var prompt: [150]i32 = undefined;
    randomTokens(random.random(), &prompt);
    const mine = [_]i32{ 21, 22, 23, 24, 25, 26, 27 };
    const replied = 60;

    var ends: [2][]u8 = undefined;
    var made: usize = 0;
    defer for (ends[0..made]) |bytes| gpa.free(bytes);
    var replies: [2][]i32 = undefined;
    defer for (replies[0..made]) |reply| gpa.free(reply);
    for ([_]bool{ false, true }) |beside| {
        var engine = try open(gpa, m, .{ .slots = 2, .prompt_rows = 48, .chunk = 20 });
        engine.bindPool();
        defer engine.deinit();
        var collect: Collect = .{ .gpa = gpa };
        defer collect.deinit();
        _ = try engine.begin(0, ask(&mine, sampled, replied, &.{}, null), 0);
        for (0..2) |_| try step(&engine, &collect);
        if (beside) _ = try engine.begin(1, ask(&prompt, greedy, 4, &starts_of_150, null), 0);
        var spins: usize = 0;
        while (!collect.ended[0]) : (spins += 1) {
            if (spins == 512) return error.Stuck;
            try step(&engine, &collect);
        }
        // every state of the prompt beside was taken inside a round that held this reply's window
        if (beside) try testing.expect(engine.keeps.took.within >= 3);
        if (beside) engine.preempt(1);
        _ = engine.retire(0, .ended, replied);
        const chain = engine.pool.chains.items[0];
        const end = engine.keeps.at(chain, mine.len + replied - 1).?;
        try testing.expectEqual(prefix.Kind.end, end.kind);
        var all: std.ArrayList(u8) = .empty;
        defer all.deinit(gpa);
        for (end.state.recurrent, end.state.conv) |recurrent, conv| {
            for ([_]cuda.Buffer{ recurrent, conv }) |buffer| {
                const bytes = try fetchBytes(gpa, buffer);
                defer gpa.free(bytes);
                try all.appendSlice(gpa, bytes);
            }
        }
        for (end.pooled.list) |pooled| {
            const bytes = try fetchBytes(gpa, pooled);
            defer gpa.free(bytes);
            try all.appendSlice(gpa, bytes);
        }
        ends[made] = try all.toOwnedSlice(gpa);
        replies[made] = try gpa.dupe(i32, collect.lists[0].items);
        made += 1;
    }
    const serial = try fixtures.serial.serialReply(gpa, m, &mine, replied, sampled);
    defer gpa.free(serial);
    try testing.expectEqualSlices(i32, serial, replies[0]);
    try testing.expectEqualSlices(i32, replies[0], replies[1]);
    try testing.expect(std.mem.eql(u8, ends[0], ends[1]));
}

test "a round kept only up to the reasoning's end: the reply is the serial one, the state after the close a serial slot's, and a prompt that parts in the content runs from it" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var random: std.Random.DefaultPrng = .init(35);
    var prompt: [40]i32 = undefined;
    randomTokens(random.random(), &prompt);
    const replied = 24;
    const each = [_]sampling.Sampling{ greedy, .{ .seed = 9, .temperature = 0.8 } };
    var tried: usize = 0;
    var cuts: usize = 0;
    var ends: usize = 0;
    // The reply's token `index` stands for `</think>`.
    for (0..replied - 2) |index| {
        const settings = each[index % 2];
        const expected = try fixtures.serial.serialReply(gpa, m, &prompt, replied, settings);
        defer gpa.free(expected);
        const close = expected[index];
        if (std.mem.indexOfScalar(i32, expected[0..index], close) != null) continue;
        errdefer std.debug.print("token {d} closes the reasoning\n", .{index});
        tried += 1;
        var engine = try open(gpa, m, .{ .prompt_rows = 48, .reply_stride = 0 });
        engine.bindPool();
        defer engine.deinit();
        var collect: Collect = .{ .gpa = gpa };
        defer collect.deinit();
        _ = try engine.begin(0, ask(&prompt, settings, replied, &.{}, close), 0);
        var spins: usize = 0;
        while (!collect.ended[0]) : (spins += 1) {
            if (spins == 256) return error.Stuck;
            try step(&engine, &collect);
            // A random head's drafts are seldom kept. The window's drafts are the reply's own next tokens, so
            // every row is kept and the close falls inside a window; now and then the last one is another.
            const seat = &engine.seats[0].?;
            if (seat.phase != .decode) continue;
            try testing.expectEqual(expected[seat.made - 1], seat.last);
            const n = @min(1 + random.random().uintLessThan(usize, 6), replied - seat.made);
            @memcpy(seat.drafts[0..n], expected[seat.made..][0..n]);
            if (n > 0 and random.random().uintLessThan(usize, 4) == 0) seat.drafts[n - 1] = 255;
            seat.draft_n = n;
        }
        const got: Got = .{ .tokens = try gpa.dupe(i32, collect.lists[0].items), .cached = 0 };
        defer gpa.free(got.tokens);
        _ = engine.retire(0, .ended, replied);
        try testing.expectEqualSlices(i32, expected, got.tokens);
        cuts += engine.keeps.took.cut;
        const at = prompt.len + index + 1;
        const chain = engine.pool.chains.items[0];
        const state = engine.keeps.at(chain, at).?;
        try testing.expectEqual(prefix.Kind.reasoning, state.kind);
        try testing.expectEqual(1, countOf(&engine, .reasoning));
        ends += @intFromBool(engine.keeps.took.cut == 0);
        var sequence: std.ArrayList(i32) = .empty;
        defer sequence.deinit(gpa);
        try sequence.appendSlice(gpa, &prompt);
        try sequence.appendSlice(gpa, expected);
        try testing.expectEqual(engine.keeps.list.items.len, try expectSerial(gpa, m, &engine, sequence.items, .{}));

        // the reasoning and the token after its close as they were decoded, then other tokens
        sequence.shrinkRetainingCapacity(at + 1);
        try sequence.appendSlice(gpa, &.{ 7, 6, 5, 251 });
        const cold = try fixtures.serial.serialReply(gpa, m, sequence.items, 8, each[1 - index % 2]);
        defer gpa.free(cold);
        const second = try one(&engine, &collect, ask(sequence.items, each[1 - index % 2], 8, &.{}, null));
        defer gpa.free(second.tokens);
        try testing.expectEqual(at, second.cached);
        try testing.expectEqualSlices(i32, cold, second.tokens);
    }
    try testing.expect(tried >= 12);
    // a round's accepted drafts held the close and were cut there, and elsewhere the round ended on it
    try testing.expect(cuts >= 4 and ends >= 2);
}

/// A conversation's requests as WS-01's `pool_states.py` sends them, as tokens: what each prompt is and where
/// its messages start.
const Talk = struct {
    gpa: Allocator,
    random: std.Random,
    prompt: std.ArrayList(i32) = .empty,
    starts: std.ArrayList(usize) = .empty,

    const user = [_]i32{ 251, 252 };
    const assistant = [_]i32{ 251, 253 };
    const closing = [_]i32{254};

    fn deinit(self: *Talk) void {
        self.prompt.deinit(self.gpa);
        self.starts.deinit(self.gpa);
    }

    /// Cuts the conversation back to `len` tokens.
    fn back(self: *Talk, len: usize) void {
        self.prompt.shrinkRetainingCapacity(len);
        while (self.starts.items.len > 0 and self.starts.items[self.starts.items.len - 1] >= len) _ = self.starts.pop();
    }

    /// A user's message of `text`, then the header the reply follows.
    fn say(self: *Talk, text: []const i32) !void {
        try self.starts.append(self.gpa, self.prompt.items.len);
        try self.prompt.appendSlice(self.gpa, &user);
        try self.prompt.appendSlice(self.gpa, text);
        try self.prompt.appendSlice(self.gpa, &closing);
        try self.starts.append(self.gpa, self.prompt.items.len);
        try self.prompt.appendSlice(self.gpa, &assistant);
    }

    /// The reply's tokens, as the next prompt holds them.
    fn heard(self: *Talk, reply: []const i32) !void {
        try self.prompt.appendSlice(self.gpa, reply);
        try self.prompt.appendSlice(self.gpa, &closing);
    }
};

/// Where a request is decoded: a session, or a pool of one seat.
const Path = union(enum) {
    session: *session_file.Session,
    pool: struct { engine: *Engine, collect: *Collect },

    const Halt = struct {
        taken: usize = 0,
        /// The reply ends at the first window's end at or past this many tokens.
        after: usize,
        list: *std.ArrayList(i32),
        gpa: Allocator,

        fn take(context: *anyopaque, token: i32) void {
            const self: *Halt = @ptrCast(@alignCast(context));
            self.taken += 1;
            self.list.append(self.gpa, token) catch @panic("oom");
        }

        fn halt(context: *anyopaque) bool {
            const self: *Halt = @ptrCast(@alignCast(context));
            return self.taken >= self.after;
        }
    };

    /// The reply to `talk`'s prompt, into `reply`, ended at `count` tokens or hung up at the first window's end
    /// at or past `hang`; `short` of the tokens given then never reached the client. Returns the prompt's
    /// rows that ran.
    ///
    /// `given` is what the session gave a reply hung up, which the pool then gives too. Where the client
    /// lacks some of it (`short`), so that both go on from a state inside the reply, the pool's windows are
    /// drafted from those tokens: a random head's drafts are seldom kept, and a state on a stride would
    /// always stand where a round of one row ends.
    fn run(self: Path, gpa: Allocator, all: *std.ArrayList(i32), talk: *const Talk, count: usize, hang: ?usize, short: usize, reply: *std.ArrayList(i32), given: *std.ArrayList(i32)) !usize {
        reply.clearRetainingCapacity();
        // every reply as its client has it, for the two paths to be held against each other
        defer all.appendSlice(gpa, reply.items) catch @panic("oom");
        const prompt = talk.prompt.items;
        switch (self) {
            .session => |session| {
                var halt: Halt = .{ .after = hang orelse std.math.maxInt(usize), .list = reply, .gpa = gpa };
                var counting: fixtures.sessions.Counting = .{};
                const done = try session.reply(prompt, greedy, .{ .max_tokens = count, .depth = 6, .opening = 6, .starts = talk.starts.items, .tap = .{ .context = &halt, .takeFn = Halt.take, .haltFn = Halt.halt } }, counting.sink());
                if (hang != null) {
                    given.clearRetainingCapacity();
                    try given.appendSlice(gpa, reply.items);
                }
                reply.shrinkRetainingCapacity(reply.items.len - @min(short, reply.items.len -| 1));
                return prompt.len - done.cached;
            },
            .pool => |pool| {
                const engine = pool.engine;
                const collect = pool.collect;
                collect.clear(0);
                const cached = try engine.begin(0, ask(prompt, greedy, count, talk.starts.items, null), 0);
                const sent = &collect.lists[0];
                const want = given.items;
                var spins: usize = 0;
                while (!collect.ended[0] and (hang == null or sent.items.len < want.len)) : (spins += 1) {
                    if (spins == 4096) return error.Stuck;
                    try step(engine, collect);
                    const seat = &engine.seats[0].?;
                    if (hang == null or short == 0 or seat.phase != .decode or seat.made >= want.len) continue;
                    try testing.expectEqual(want[seat.made - 1], seat.last);
                    const n = @min(1 + (spins * 7 + 3) % 6, want.len - seat.made);
                    @memcpy(seat.drafts[0..n], want[seat.made..][0..n]);
                    seat.draft_n = n;
                }
                try reply.appendSlice(gpa, sent.items);
                if (hang != null) {
                    try testing.expect(reply.items.len >= want.len);
                    try testing.expectEqualSlices(i32, want, reply.items[0..want.len]);
                    // the client has what the session's had
                    reply.shrinkRetainingCapacity(if (short == 0) reply.items.len else want.len);
                }
                reply.shrinkRetainingCapacity(reply.items.len - @min(short, reply.items.len -| 1));
                _ = engine.retire(0, if (collect.ended[0]) .ended else .cancelled, reply.items.len);
                return prompt.len - cached;
            },
        }
    }
};

/// The eleven requests of WS-01's table and the hang-up between them, on `path`: the prompt's rows each ran,
/// in order. `seed` places the edits and the hang-up.
fn conversation(gpa: Allocator, path: Path, seed: u64, document: usize, rows: *[12]usize, replies: *std.ArrayList(i32), given: *std.ArrayList(i32)) !void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    var reply: std.ArrayList(i32) = .empty;
    defer reply.deinit(gpa);
    var at: usize = 0;
    const count = 12;
    for ([_]usize{ document, document * 2 / 3 }, 0..) |size, nth| {
        var talk: Talk = .{ .gpa = gpa, .random = random };
        defer talk.deinit();
        const text = try gpa.alloc(i32, size + 12);
        defer gpa.free(text);
        randomTokens(random, text);
        var question: [10]i32 = undefined;
        randomTokens(random, &question);
        // the conversation of one message: sent, sent again, its question changed, and changed again from
        // the same token on
        if (nth == 1) {
            try talk.say(text);
            _ = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
            rows[at] = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
            const edit = text.len - 1 - random.uintLessThan(usize, 10);
            for (0..2) |_| {
                talk.back(0);
                randomTokens(random, text[edit..]);
                try talk.say(text);
                rows[at + 1] = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
                at += 1;
            }
            at += 1;
            break;
        }
        try talk.say(text);
        rows[at] = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
        try talk.heard(reply.items);
        const second = talk.prompt.items.len;
        try talk.say(&question);
        rows[at + 1] = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
        try talk.heard(reply.items);
        const third = talk.prompt.items.len;
        var other: [10]i32 = undefined;
        randomTokens(random, &other);
        try talk.say(&other);
        rows[at + 2] = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
        // again
        rows[at + 3] = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
        // the third turn's question changed from a token of it on
        talk.back(third);
        randomTokens(random, other[random.uintLessThan(usize, other.len)..]);
        try talk.say(&other);
        rows[at + 4] = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
        // a long reply hung up at a row, of which the client may not have had the last tokens given
        talk.back(third);
        randomTokens(random, &other);
        try talk.say(&other);
        const hang = 20 + random.uintLessThan(usize, 90);
        const short = if (random.boolean()) 0 else 1 + random.uintLessThan(usize, 18);
        rows[at + 5] = try path.run(gpa, replies, &talk, 160, hang, short, &reply, given);
        try talk.heard(reply.items);
        randomTokens(random, &other);
        try talk.say(&other);
        rows[at + 6] = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
        // the second turn's question changed, the turns after it gone
        talk.back(second);
        randomTokens(random, question[random.uintLessThan(usize, question.len)..]);
        try talk.say(&question);
        rows[at + 7] = try path.run(gpa, replies, &talk, count, null, 0, &reply, given);
        at += 8;
    }
    std.debug.assert(at == 11);
    rows[11] = 0;
}

test "a pool of one seat runs at most fifteen rows more of a prompt than a session does, over a conversation's requests with the edits and the hang-up placed at random" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const context = 1024;
    var worse: usize = 0;
    var better: usize = 0;
    for (0..20) |trial| {
        const document: usize = 60 + 23 * trial;
        const session: *session_file.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = context, .prompt_rows = 48, .stride = 48, .reply_stride = 32 });
        defer session.destroy();
        var engine = try open(gpa, m, .{ .context = context, .prompt_rows = 48, .reply_stride = 32 });
        engine.bindPool();
        defer engine.deinit();
        var collect: Collect = .{ .gpa = gpa };
        defer collect.deinit();
        var rows: [2][12]usize = undefined;
        var replies: [2]std.ArrayList(i32) = .{ .empty, .empty };
        defer for (&replies) |*list| list.deinit(gpa);
        var given: std.ArrayList(i32) = .empty;
        defer given.deinit(gpa);
        try conversation(gpa, .{ .session = session }, 100 + trial, document, &rows[0], &replies[0], &given);
        try conversation(gpa, .{ .pool = .{ .engine = &engine, .collect = &collect } }, 100 + trial, document, &rows[1], &replies[1], &given);
        // the same conversation on both: every reply's tokens
        try testing.expectEqualSlices(i32, replies[0].items, replies[1].items);
        for (rows[0], rows[1], 0..) |alone, pooled, request| {
            errdefer std.debug.print("trial {d}, request {d}: the session ran {d} rows, the pool {d}\n", .{ trial, request, alone, pooled });
            try testing.expect(pooled <= alone + 15);
            worse += @intFromBool(pooled > alone);
            better += @intFromBool(pooled < alone);
        }
    }
    // After a hang-up the pool's state on a stride stands below the session's, or its next one below where
    // the client stopped; and a reply's end of an earlier turn, which a session no longer holds, serves an
    // edited question. So the bound is not met by never differing.
    try testing.expect(worse >= 1 and better >= 20);
}

test "the store's bounds: the states' bytes stay inside the budget and their count under the cap, a chain holds eight of a prompt's kinds and two strides, and each reply is the serial one" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var random: std.Random.DefaultPrng = .init(57);
    var prompt: [400]i32 = undefined;
    randomTokens(random.random(), &prompt);
    var starts: [16]usize = undefined;
    // past `near` of each other: every one is a state the plan wants, and it lets go as it takes
    for (&starts, 0..) |*start, i| start.* = i * 33;

    for ([_]struct { states: usize, cap: usize }{ .{ .states = 100, .cap = prefix.states_max }, .{ .states = 5, .cap = prefix.states_max }, .{ .states = 100, .cap = 4 } }) |bound| {
        errdefer std.debug.print("{d} states of bytes, {d} at most\n", .{ bound.states, bound.cap });
        var probe = try open(gpa, m, .{ .context = 1024, .prompt_rows = 16 });
        probe.bindPool();
        const each = keep_file.bytesAt(&probe, 500);
        probe.deinit();
        var engine = try open(gpa, m, .{ .context = 1024, .prompt_rows = 16, .states_bytes = bound.states * each });
        engine.bindPool();
        defer engine.deinit();
        engine.store.limit = bound.cap;
        var collect: Collect = .{ .gpa = gpa };
        defer collect.deinit();
        var most: usize = 0;
        for ([_]usize{ 400, 330 }) |len| {
            const expected = try fixtures.serial.serialReply(gpa, m, prompt[0..len], 120, sampled);
            defer gpa.free(expected);
            collect.clear(0);
            _ = try engine.begin(0, ask(prompt[0..len], sampled, 120, &starts, null), 0);
            var spins: usize = 0;
            while (!collect.ended[0]) : (spins += 1) {
                if (spins == 1024) return error.Stuck;
                try step(&engine, &collect);
                try testing.expect(engine.keeps.bytes() <= engine.keeps.budget);
                try testing.expect(engine.keeps.list.items.len <= bound.cap);
                try testing.expectEqual(engine.store.marks.items.len, engine.keeps.list.items.len);
                most = @max(most, engine.keeps.list.items.len);
                for (engine.pool.chains.items) |chain| {
                    var kinds: [3]usize = .{ 0, 0, 0 };
                    for (engine.keeps.list.items) |keep| {
                        if (keep.chain != chain) continue;
                        kinds[0] += @intFromBool(keep.kind.thins());
                        kinds[1] += @intFromBool(keep.kind == .stride);
                        kinds[2] += @intFromBool(keep.kind == .chunk);
                    }
                    try testing.expect(kinds[0] <= prefix.prompt_marks and kinds[1] <= prefix.reply_marks and kinds[2] <= prefix.reply_marks);
                }
            }
            try testing.expectEqualSlices(i32, expected, collect.lists[0].items);
            _ = engine.retire(0, .ended, 120);
            try testing.expect(engine.keeps.bytes() <= engine.keeps.budget);
        }
        // 24 chunk ends and 12 starts a prompt: where neither bound is near a chain keeps eight of the
        // prompt's kinds, two chunk ends, two strides and an end
        if (bound.states == 100 and bound.cap == prefix.states_max) try testing.expect(most >= 14 and most <= 26);
        if (bound.states == 5) try testing.expect(most <= 5 and most >= 3);
        if (bound.cap == 4) try testing.expectEqual(4, most);
    }
}

test "a turn's states are written over buffers made before it: none is made before its first token, and none at all once the budget is full" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var random: std.Random.DefaultPrng = .init(58);
    var prompt: [400]i32 = undefined;
    randomTokens(random.random(), &prompt);
    var starts: [16]usize = undefined;
    for (&starts, 0..) |*start, i| start.* = i * 33;

    var probe = try open(gpa, m, .{ .context = 1024, .prompt_rows = 64 });
    probe.bindPool();
    const each = keep_file.bytesAt(&probe, 500);
    probe.deinit();
    var engine = try open(gpa, m, .{ .context = 1024, .prompt_rows = 64, .states_bytes = 40 * each });
    engine.bindPool();
    defer engine.deinit();
    engine.readyStates();
    const most = engine.keeps.spares;
    try testing.expectEqual(most, engine.keeps.spare.items.len);
    try testing.expectEqual(most, engine.keeps.took.made);
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    var made: [14]usize = undefined;
    for (&made, 0..) |*after, turn| {
        // each goes on from the one before it, a turn of twenty tokens
        const len = 100 + turn * 20;
        const before = engine.keeps.took.made;
        const taken = engine.keeps.took.within + engine.keeps.took.standing;
        collect.clear(0);
        _ = try engine.begin(0, ask(prompt[0..len], sampled, 24, &starts, null), 0);
        var spins: usize = 0;
        while (collect.lists[0].items.len == 0 and !collect.ended[0]) : (spins += 1) {
            if (spins == 64) return error.Stuck;
            try step(&engine, &collect);
        }
        // the turn kept states of its prompt, and made no buffer for them
        if (turn > 0) {
            try testing.expect(engine.keeps.took.within + engine.keeps.took.standing > taken);
            try testing.expectEqual(before, engine.keeps.took.made);
        }
        while (!collect.ended[0]) : (spins += 1) {
            if (spins == 1024) return error.Stuck;
            try step(&engine, &collect);
        }
        _ = engine.retire(0, .ended, 24);
        try testing.expectEqual(most, engine.keeps.spare.items.len);
        try testing.expect(engine.keeps.bytes() <= engine.keeps.budget);
        after.* = engine.keeps.took.made;
    }
    // the budget filled on the way, and from there a state's buffers are those of one that went
    try testing.expect(made[made.len - 1] > most and made[made.len - 1] <= 41);
    try testing.expectEqual(made[made.len - 4], made[made.len - 1]);
    try testing.expect(engine.keeps.list.items.len >= 30);
}

test "what goes when a chain holds one state too many is what a session lets go: the oldest far from the end, never the newest" {
    // `marks.thinned` over the positions a chain holds and those it still plans: a prompt of sixteen messages
    // keeps eight, as `Session.plan` keeps them
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var random: std.Random.DefaultPrng = .init(58);
    var prompt: [640]i32 = undefined;
    randomTokens(random.random(), &prompt);
    var starts: [16]usize = undefined;
    for (&starts, 0..) |*start, i| start.* = i * 40;
    const session: *session_file.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = 1024, .prompt_rows = 48, .stride = 0, .reply_stride = 0 });
    defer session.destroy();
    var counting: fixtures.sessions.Counting = .{};
    _ = try session.reply(&prompt, greedy, .{ .max_tokens = 2, .starts = &starts }, counting.sink());
    const alone = try fixtures.sessions.snapshotPositions(session, gpa);
    defer gpa.free(alone);

    var engine = try open(gpa, m, .{ .context = 1024, .prompt_rows = 48, .reply_stride = 0 });
    engine.bindPool();
    defer engine.deinit();
    // the plan's positions alone, as the session with no stride
    engine.keeps.chunks = false;
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    _ = try engine.begin(0, ask(&prompt, greedy, 2, &starts, null), 0);
    while (engine.seats[0].?.phase == .fill) {
        try step(&engine, &collect);
        // what the plan lets go on the way goes then, not at the prompt's end
        try testing.expect(engine.keeps.list.items.len <= prefix.prompt_marks);
    }
    const pooled = try positions(gpa, &engine, true);
    defer gpa.free(pooled);
    try testing.expectEqual(prefix.prompt_marks, pooled.len);
    try testing.expectEqualSlices(usize, alone, pooled);
    engine.preempt(0);
}

test "a reply taken up again keeps its strides where the reply uninterrupted keeps them: they are counted from the newest state below its end, not from the end it goes on from" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var random: std.Random.DefaultPrng = .init(59);
    var prompt: [60]i32 = undefined;
    randomTokens(random.random(), &prompt);
    const stride = 32;
    const replied = 150;
    const expected = try fixtures.serial.serialReply(gpa, m, &prompt, replied, sampled);
    defer gpa.free(expected);
    for ([_]usize{ 21, 40, 77 }) |hang| {
        errdefer std.debug.print("hung up after {d} tokens\n", .{hang});
        var engine = try open(gpa, m, .{ .prompt_rows = 48, .reply_stride = stride });
        engine.bindPool();
        defer engine.deinit();
        var collect: Collect = .{ .gpa = gpa };
        defer collect.deinit();
        _ = try engine.begin(0, ask(&prompt, sampled, replied, &.{}, null), 0);
        while (collect.lists[0].items.len < hang) try step(&engine, &collect);
        const given = collect.lists[0].items.len;
        try testing.expectEqualSlices(i32, expected[0..given], collect.lists[0].items);
        // its client took every token: the state at the end is kept, and the reply goes on from it
        _ = engine.retire(0, .cancelled, given);
        var again: std.ArrayList(i32) = .empty;
        defer again.deinit(gpa);
        try again.appendSlice(gpa, &prompt);
        try again.appendSlice(gpa, expected[0..given]);
        var rest = ask(again.items, sampled, replied - given, &.{}, null);
        rest.depth = 6;
        collect.clear(0);
        try testing.expectEqual(prompt.len + given - 1, try engine.begin(0, rest, given));
        while (!collect.ended[0]) try step(&engine, &collect);
        try testing.expectEqualSlices(i32, expected[given..], collect.lists[0].items);
        _ = engine.retire(0, .ended, replied);
        // Each stride's state stands a round below a multiple of the stride from the mark before the prompt's
        // end, on the chain it was cut on and on the one it went on with: where a round of one row first
        // could pass it, or a few rows on where a draft was kept.
        var strides: usize = 0;
        for (engine.keeps.list.items) |keep| {
            if (keep.kind != .stride) continue;
            strides += 1;
            const past = keep.pos + 15 - (prompt.len - 2);
            try testing.expect(past >= stride and past % stride <= 4);
        }
        try testing.expect(strides >= 2);
    }
}

test "the state where a shared prefix ends is the pool's last to go: a prompt of many messages keeps it, other conversations' states go before it, and a conversation over the harness starts there" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    const harness = 80;
    // `told`: the request names the harness's end (`Ask.kept`), as the server does; else it is a start like another
    for ([_]bool{ true, false }) |told| {
        var random: std.Random.DefaultPrng = .init(64);
        var engine = try open(gpa, m, .{ .context = 1024, .prompt_rows = 48, .reply_stride = 0 });
        engine.bindPool();
        defer engine.deinit();
        engine.keeps.chunks = false;
        engine.store.limit = 12;
        var collect: Collect = .{ .gpa = gpa };
        defer collect.deinit();
        // a harness, then sixteen messages of 30 tokens: more starts than a chain holds states
        var prompt: [harness + 16 * 30]i32 = undefined;
        randomTokens(random.random(), &prompt);
        var starts: [17]usize = undefined;
        starts[0] = 0;
        for (starts[1..], 0..) |*start, i| {
            start.* = harness + i * 30;
            prompt[start.*] = 252;
        }
        var first = ask(&prompt, greedy, 4, &starts, null);
        if (told) first.kept = harness;
        const long = try one(&engine, &collect, first);
        defer gpa.free(long.tokens);
        var shared: usize = 0;
        for (engine.store.marks.items) |mark| {
            if (mark.shared) shared += 1;
            try testing.expectEqual(mark.shared, told and mark.pos == harness);
        }
        try testing.expectEqual(@as(usize, @intFromBool(told)), shared);
        // other conversations, each idle after its reply: more states than the store holds
        for (0..6) |_| {
            var other: [130]i32 = undefined;
            randomTokens(random.random(), &other);
            const got = try one(&engine, &collect, ask(&other, greedy, 4, &.{ 0, 40, 80, 120 }, null));
            gpa.free(got.tokens);
            try testing.expectEqual(@as(usize, 0), got.cached);
        }
        try testing.expectEqual(engine.store.bound(), engine.store.marks.items.len);
        try testing.expectEqual(engine.keeps.list.items.len, engine.store.marks.items.len);
        // a new conversation over the harness
        var next: [harness + 50]i32 = undefined;
        randomTokens(random.random(), &next);
        @memcpy(next[0 .. harness + 1], prompt[0 .. harness + 1]);
        var second = ask(&next, greedy, 6, &.{ 0, harness, harness + 40 }, null);
        if (told) second.kept = harness;
        const got = try one(&engine, &collect, second);
        defer gpa.free(got.tokens);
        if (told) try testing.expectEqual(@as(usize, harness), got.cached) else try testing.expect(got.cached < harness);
        const expected = try fixtures.serial.serialReply(gpa, m, &next, 6, greedy);
        defer gpa.free(expected);
        try testing.expectEqualSlices(i32, expected, got.tokens);
    }
}
