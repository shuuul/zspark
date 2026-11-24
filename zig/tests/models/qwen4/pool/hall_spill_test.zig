//! Tests of the pool's states on disk (`hall_spill.zig`, spec 074): a state that leaves for room and one held
//! at the stop are files, a pool opened after starts a conversation's next prompt from one with the cold
//! reply and a serial slot's state, and a session and a pool read each other's.

const std = @import("std");
const fixtures = @import("../testing.zig");
const keep_fixture = @import("hall_keep_fixture.zig");
const Collect = keep_fixture.Collect;
const Shape = keep_fixture.Shape;
const step = keep_fixture.step;
const ask = keep_fixture.ask;
const expectSerial = keep_fixture.expectSerial;
const Allocator = std.mem.Allocator;
const testing = std.testing;
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const spill = @import("../../../../src/decode/spill.zig");
const learned_blocks = @import("../../../../src/decode/learned_blocks.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const hall_spill = @import("../../../../src/models/qwen4/pool/hall_spill.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const session_file = @import("../../../../src/models/qwen4/session/session.zig");
const Engine = hall_engine.Engine;
const Store = hall_spill.Store;

const sampled: sampling.Sampling = .{ .seed = 7, .temperature = 0.8 };
const context = 512;
const replied = 24;
const starts_of_150 = [_]usize{ 0, 37, 90, 141 };
const tail = [_]i32{ 251, 252, 7, 9, 3, 8 };

const Got = struct { tokens: []i32, cached: usize, report: hall_file.Report };

const fx = struct {
    /// The floor, the worth and the block of a test's small prompts, until `restore`: its states' rows lie
    /// in blocks of 16 positions.
    const Small = struct {
        floor: usize,
        worth: usize,
        block: u32,

        fn set() Small {
            const was: Small = .{ .floor = spill.floor, .worth = spill.worth, .block = learned_blocks.block_tokens };
            spill.floor, spill.worth, learned_blocks.block_tokens = .{ 16, 16, 16 };
            return was;
        }

        fn restore(self: Small) void {
            spill.floor, spill.worth, learned_blocks.block_tokens = .{ self.floor, self.worth, self.block };
        }
    };

    /// A pool of one seat with `spare` extents beside what it must have.
    fn open(gpa: Allocator, m: model_file.Model, spare: usize) !Engine {
        return openOf(gpa, m, context, spare);
    }

    fn openOf(gpa: Allocator, m: model_file.Model, capacity: usize, spare: usize) !Engine {
        var engine = try Engine.open(gpa, m, &.{}, .{
            .slots = 1,
            .session = .{ .capacity = capacity, .prompt_rows = 32, .reply_stride = 16 },
            .pool = (extents.lowerBound(capacity, 1) + spare) * extents.positions,
            .chunk = 24,
            .states_bytes = hall_file.default_states_bytes,
        });
        engine.io = testing.io;
        return engine;
    }

    /// Once the engine stands where it stays.
    fn bind(engine: *Engine, store: ?*Store) void {
        engine.bindPool();
        if (store) |held| hall_spill.attach(engine, held);
    }

    /// One request alone on slot 0, to its end: the reply's tokens, the caller's.
    fn one(engine: *Engine, collect: *Collect, prompt: []const i32, starts: []const usize) !Got {
        collect.clear(0);
        const cached = try engine.begin(0, ask(prompt, sampled, replied, starts, null), 0);
        var spins: usize = 0;
        while (!collect.ended[0]) : (spins += 1) {
            if (spins == 4096) return error.Stuck;
            try step(engine, collect);
        }
        const tokens = try engine.gpa.dupe(i32, collect.lists[0].items);
        return .{ .tokens = tokens, .cached = cached, .report = engine.retire(0, .ended, tokens.len).? };
    }

    /// A conversation: its first prompt of 150 tokens, then each turn the reply and a message after it.
    const Talk = struct {
        gpa: Allocator,
        prompt: std.ArrayList(i32) = .empty,
        starts: std.ArrayList(usize) = .empty,

        fn init(gpa: Allocator, seed: u64) !Talk {
            return of(gpa, seed, 150, &starts_of_150);
        }

        fn of(gpa: Allocator, seed: u64, len: usize, starts: []const usize) !Talk {
            var self: Talk = .{ .gpa = gpa };
            errdefer self.deinit();
            var random: std.Random.DefaultPrng = .init(seed);
            try self.prompt.resize(gpa, len);
            for (self.prompt.items) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
            try self.starts.appendSlice(gpa, starts);
            return self;
        }

        fn deinit(self: *Talk) void {
            self.prompt.deinit(self.gpa);
            self.starts.deinit(self.gpa);
        }

        fn turn(self: *Talk, reply: []const i32) !void {
            try self.prompt.appendSlice(self.gpa, reply);
            try self.starts.append(self.gpa, self.prompt.items.len);
            try self.prompt.appendSlice(self.gpa, &tail);
        }

        /// The prompt on `engine`, held to a row at a time: the reply's tokens.
        fn run(self: *Talk, m: model_file.Model, engine: *Engine, collect: *Collect) !Got {
            const got = try one(engine, collect, self.prompt.items, self.starts.items);
            errdefer self.gpa.free(got.tokens);
            const cold = try fixtures.serial.serialReply(self.gpa, m, self.prompt.items, replied, sampled);
            defer self.gpa.free(cold);
            try testing.expectEqualSlices(i32, cold, got.tokens);
            return got;
        }
    };
};

/// A conversation's first turn on a pool that then stops; its second on a pool of another load, from the
/// file, beside a pool that holds nothing.
fn stoppedAndRead(gpa: Allocator, m: model_file.Model) !void {
    const small: fx.Small = .set();
    defer small.restore();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(gpa, testing.io, tmp.dir, 1, 1 << 30);
    defer store.close();
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    var talk: fx.Talk = try .init(gpa, 71);
    defer talk.deinit();
    {
        var engine = try fx.open(gpa, m, 24);
        fx.bind(&engine, &store);
        defer engine.deinit();
        const first = try talk.run(m, &engine, &collect);
        defer gpa.free(first.tokens);
        try testing.expectEqual(0, first.cached);
        try talk.turn(first.tokens);
        try testing.expectEqual(0, store.written);
        hall_spill.flush(&engine);
        try testing.expectEqual(1, store.written);
        hall_spill.flush(&engine);
        try testing.expectEqual(1, store.written);
    }
    // the highest state of the first prompt: the mark before its end, or its last message's start
    const at = store.longest(talk.prompt.items, talk.prompt.items.len).? - 1;
    try testing.expect(at >= 141 and at < 150);
    var reports: [2]hall_file.Report = undefined;
    for ([_]?*Store{ null, &store }, &reports) |held, *report| {
        var engine = try fx.open(gpa, m, 24);
        fx.bind(&engine, held);
        defer engine.deinit();
        const second = try talk.run(m, &engine, &collect);
        defer gpa.free(second.tokens);
        report.* = second.report;
        try testing.expectEqual(if (held == null) 0 else at, second.cached);
        if (held == null) continue;
        try testing.expectEqual(1, store.read);
        // the state read, and those taken above it, are a serial slot's
        var sequence: std.ArrayList(i32) = .empty;
        defer sequence.deinit(gpa);
        try sequence.appendSlice(gpa, talk.prompt.items);
        try sequence.appendSlice(gpa, second.tokens);
        const held_states = engine.keeps.list.items.len;
        try testing.expectEqual(held_states, try expectSerial(gpa, m, &engine, sequence.items, .{ .context = context }));
        var from_file = false;
        for (engine.keeps.list.items) |keep| from_file = from_file or keep.pos == at;
        try testing.expect(from_file);
        // the pool holds it now: the prompt again reads nothing
        const again = try talk.run(m, &engine, &collect);
        defer gpa.free(again.tokens);
        try testing.expect(again.cached >= at);
        try testing.expectEqual(1, store.read);
    }
    try testing.expectEqual(reports[0].drafts, reports[1].drafts);
    try testing.expectEqual(reports[0].drafted, reports[1].drafted);
    try testing.expectEqual(reports[0].accepted, reports[1].accepted);
}

test "a state the pool holds when it stops is a file, and the next load's pool starts the conversation's next prompt from it: the cold reply and a serial slot's state, each cache storage" {
    const gpa = testing.allocator;
    defer fixtures.random_model.caches = .bf16;
    for ([_]attention.Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
        defer m.deinit();
        try stoppedAndRead(gpa, m);
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    try stoppedAndRead(gpa, m);
}

test "two conversations over one prefix that the pool holds when it stops share its blocks on disk, and the next loads' pools start each from its files with the cold reply" {
    const gpa = testing.allocator;
    const io = testing.io;
    const small: fx.Small = .set();
    defer small.restore();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(gpa, io, tmp.dir, 1, 1 << 30);
    defer store.close();
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    var talks = [_]fx.Talk{ try .init(gpa, 71), try .init(gpa, 72) };
    defer for (&talks) |*talk| talk.deinit();
    // the second begins with the first's 100 tokens: blocks 0 to 5 (through token 96) are the same
    @memcpy(talks[1].prompt.items[0..100], talks[0].prompt.items[0..100]);
    {
        var engine = try fx.open(gpa, m, 24);
        fx.bind(&engine, &store);
        defer engine.deinit();
        for (&talks) |*talk| {
            const first = try talk.run(m, &engine, &collect);
            defer gpa.free(first.tokens);
            try talk.turn(first.tokens);
        }
        hall_spill.flush(&engine);
        try testing.expectEqual(2, store.written);
    }
    // each state names its whole blocks; the files are the blocks they name, the shared ones once
    var names: std.AutoHashMapUnmanaged(learned_blocks.Stem, void) = .empty;
    defer names.deinit(gpa);
    var named: usize = 0;
    try testing.expectEqual(2, store.held.items.len);
    for (store.held.items) |entry| {
        try testing.expect(entry.blocks.len >= 8);
        named += entry.blocks.len;
        for (entry.blocks) |one| try names.put(gpa, one, {});
    }
    try testing.expectEqual(names.count(), try filesIn(io, store.dir, ".blk"));
    try testing.expectEqual(named - 6, names.count());
    for (&talks, 1..) |*talk, n| {
        const at = store.longest(talk.prompt.items, talk.prompt.items.len).? - 1;
        var engine = try fx.open(gpa, m, 24);
        fx.bind(&engine, &store);
        defer engine.deinit();
        const second = try talk.run(m, &engine, &collect);
        defer gpa.free(second.tokens);
        try testing.expectEqual(at, second.cached);
        try testing.expectEqual(n, store.read);
    }
}

test "a conversation whose extent the pool takes back is written as it goes, once, and its next prompt starts from the file; with the flag off it starts from nothing" {
    const gpa = testing.allocator;
    const small: fx.Small = .set();
    defer small.restore();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(gpa, testing.io, tmp.dir, 1, 1 << 30);
    defer store.close();
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    var cached: [2]usize = undefined;
    for ([_]?*Store{ &store, null }, &cached) |held, *took| {
        // extents for the seat and one conversation beside it
        var engine = try fx.open(gpa, m, 0);
        fx.bind(&engine, held);
        defer engine.deinit();
        var talks: [3]fx.Talk = undefined;
        var made: usize = 0;
        defer for (talks[0..made]) |*talk| talk.deinit();
        for (&talks, 0..) |*talk, i| {
            talk.* = try .init(gpa, 80 + i);
            made += 1;
            const first = try talk.run(m, &engine, &collect);
            defer gpa.free(first.tokens);
            try talk.turn(first.tokens);
        }
        if (held != null) try testing.expect(store.written >= 1);
        const written = store.written;
        const second = try talks[0].run(m, &engine, &collect);
        defer gpa.free(second.tokens);
        took.* = second.cached;
        if (held != null) {
            try testing.expectEqual(1, store.read);
            // each prompt's state is one file however often its chain is shortened: the three first
            // prompts' and the second of the conversation that went on
            try testing.expect(store.written >= written and store.written <= talks.len);
            hall_spill.flush(&engine);
            try testing.expectEqual(talks.len + 1, store.written);
        }
    }
    try testing.expect(cached[0] >= 141 and cached[0] < 150);
    try testing.expectEqual(0, cached[1]);
    try testing.expectEqual(1, store.read);
}

test "a conversation whose extent the pool takes back while the writer has still to write it: its next prompt starts from the writer's memory with the reply a pool without the flag gives, and the stop writes it" {
    const gpa = testing.allocator;
    const io = testing.io;
    const small: fx.Small = .set();
    defer small.restore();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(gpa, io, tmp.dir, 1, 1 << 30);
    defer store.close();
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    var replies: [2][]i32 = undefined;
    var got: usize = 0;
    defer for (replies[0..got]) |reply| gpa.free(reply);
    for ([_]?*Store{ &store, null }) |held| {
        var engine = try fx.open(gpa, m, 0);
        fx.bind(&engine, held);
        defer engine.deinit();
        // the writer begins nothing: what leaves stays in its memory
        if (held != null) store.writer.?.hold(true);
        var talks: [3]fx.Talk = undefined;
        var made: usize = 0;
        defer for (talks[0..made]) |*talk| talk.deinit();
        for (&talks, 0..) |*talk, i| {
            talk.* = try .init(gpa, 80 + i);
            made += 1;
            const first = try talk.run(m, &engine, &collect);
            defer gpa.free(first.tokens);
            try talk.turn(first.tokens);
        }
        const second = try talks[0].run(m, &engine, &collect);
        replies[got] = second.tokens;
        got += 1;
        if (held == null) {
            try testing.expectEqual(0, second.cached);
            continue;
        }
        try testing.expect(store.written >= 1);
        try testing.expectEqual(0, try binsIn(io, store.dir));
        try testing.expectEqual(0, store.writer.?.done);
        try testing.expect(second.cached >= 141 and second.cached < 150);
        try testing.expectEqual(1, store.read);
        store.writer.?.hold(false);
        hall_spill.flush(&engine);
        try testing.expectEqual(talks.len + 1, store.written);
        try testing.expectEqual(store.written, try binsIn(io, store.dir));
        try testing.expectEqual(store.written, store.writer.?.done);
    }
    try testing.expectEqualSlices(i32, replies[1], replies[0]);
}

fn filesIn(io: std.Io, dir: std.Io.Dir, ending: []const u8) !usize {
    var n: usize = 0;
    var entries = dir.iterate();
    while (try entries.next(io)) |entry| n += @intFromBool(std.mem.endsWith(u8, entry.name, ending));
    return n;
}

fn binsIn(io: std.Io, dir: std.Io.Dir) !usize {
    var n: usize = 0;
    var entries = dir.iterate();
    while (try entries.next(io)) |entry| n += @intFromBool(std.mem.endsWith(u8, entry.name, ".st"));
    return n;
}

/// Conversations of 600 tokens in a pool with extents for two: each that goes is written with its rows in two
/// extents, and the one that comes takes them in the order they were freed. Every reply is the one a pool
/// that holds nothing and has room gives, drafts and all; a pool of the next load reads each file.
fn acrossExtents(gpa: Allocator, m: model_file.Model) !void {
    const small: fx.Small = .set();
    defer small.restore();
    const capacity = 1536;
    const starts = [_]usize{ 0, 211, 430, 588 };
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(gpa, testing.io, tmp.dir, 1, 1 << 30);
    defer store.close();
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    var cold = try fx.openOf(gpa, m, capacity, 24);
    fx.bind(&cold, null);
    defer cold.deinit();
    var talks: [3]fx.Talk = undefined;
    var made: usize = 0;
    defer for (talks[0..made]) |*talk| talk.deinit();
    var parted = false;
    {
        var engine = try fx.openOf(gpa, m, capacity, 0);
        fx.bind(&engine, &store);
        defer engine.deinit();
        for (&talks, 0..) |*talk, i| {
            talk.* = try .of(gpa, 60 + i, 600, &starts);
            made += 1;
            try both(gpa, talk, &cold, &engine, &collect, 0);
        }
        try testing.expect(store.written >= 1);
        // the first again, from its file, then the second from its own
        for (talks[0..2], 1..) |*talk, n| {
            try both(gpa, talk, &cold, &engine, &collect, null);
            try testing.expectEqual(n, store.read);
        }
        for (engine.pool.chains.items) |chain| {
            const table = chain.table.items;
            if (table.len < 2) continue;
            for (table[1..], table[0 .. table.len - 1]) |next, before| parted = parted or next != before + 1;
        }
        hall_spill.flush(&engine);
    }
    // a chain whose extents do not follow one another was among those written
    try testing.expect(parted);
    var engine = try fx.openOf(gpa, m, capacity, 24);
    fx.bind(&engine, &store);
    defer engine.deinit();
    for (&talks, 3..) |*talk, n| {
        try both(gpa, talk, &cold, &engine, &collect, null);
        try testing.expectEqual(n, store.read);
    }
}

/// `talk`'s prompt on a pool that holds nothing of it and on `engine`: the same reply, drafted alike; the
/// conversation then goes on. `held`: what `engine` starts from, null for a state of 588 or above.
fn both(gpa: Allocator, talk: *fx.Talk, cold: *Engine, engine: *Engine, collect: *Collect, held: ?usize) !void {
    const want = try fx.one(cold, collect, talk.prompt.items, talk.starts.items);
    defer gpa.free(want.tokens);
    // the cold pool keeps nothing of it: the next turn there is run whole again
    for (cold.pool.chains.items) |chain| if (!chain.live) {
        try cold.store.release(chain);
        break;
    };
    @import("../../../../src/models/qwen4/pool/hall_keep.zig").sweep(cold);
    const got = try fx.one(engine, collect, talk.prompt.items, talk.starts.items);
    defer gpa.free(got.tokens);
    try testing.expectEqual(0, want.cached);
    if (held) |at| try testing.expectEqual(at, got.cached) else try testing.expect(got.cached >= 588);
    try testing.expectEqualSlices(i32, want.tokens, got.tokens);
    try testing.expectEqual(want.report.drafts, got.report.drafts);
    try testing.expectEqual(want.report.drafted, got.report.drafted);
    try testing.expectEqual(want.report.accepted, got.report.accepted);
    try talk.turn(got.tokens);
}

test "a state whose rows lie in several extents, in an order they were freed in, is written and read: each reply is a cold pool's, a head over DeltaNet layers and trunk attention" {
    const gpa = testing.allocator;
    defer fixtures.random_model.caches = .bf16;
    for ([_]attention.Storage{ .bf16, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
        defer m.deinit();
        try acrossExtents(gpa, m);
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    try acrossExtents(gpa, m);
}

test "a session and a pool read each other's files: the reply is the cold one either way" {
    const gpa = testing.allocator;
    const io = testing.io;
    const small: fx.Small = .set();
    defer small.restore();
    const count = 12;
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    defer fixtures.random_model.caches = .bf16;
    for ([_]attention.Storage{ .bf16, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
        defer m.deinit();
        var diagnostic: tokenizer.Diagnostic = .{};
        var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
        defer tokens.deinit(gpa);
        var tmp = testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var store: Store = try .open(gpa, io, tmp.dir, 1, 1 << 30);
        defer store.close();
        var collect: Collect = .{ .gpa = gpa };
        defer collect.deinit();
        var counting: fixtures.sessions.Counting = .{};
        const options: session_file.Options = .{ .capacity = context, .prompt_rows = 32, .reply_stride = 16 };

        // a session's, read by a pool
        var talk: fx.Talk = try .init(gpa, 91);
        defer talk.deinit();
        {
            const session: *session_file.Session = try .create(gpa, io, m, &tokens, &.{}, options);
            defer session.destroy();
            hall_spill.learnt.spilling(session, &store);
            var record: [count]i32 = undefined;
            _ = try session.reply(talk.prompt.items, greedy, .{ .max_tokens = count, .depth = 3, .starts = talk.starts.items, .record = &record }, counting.sink());
            try talk.turn(&record);
            hall_spill.learnt.flush(session, true);
            try testing.expectEqual(1, store.written);
        }
        {
            var engine = try fx.open(gpa, m, 24);
            fx.bind(&engine, &store);
            defer engine.deinit();
            const got = try talk.run(m, &engine, &collect);
            defer gpa.free(got.tokens);
            try testing.expect(got.cached >= 141 and got.cached < 150);
            try testing.expectEqual(1, store.read);
        }

        // a pool's, read by a session
        var other: fx.Talk = try .init(gpa, 92);
        defer other.deinit();
        {
            var engine = try fx.open(gpa, m, 24);
            fx.bind(&engine, &store);
            defer engine.deinit();
            const got = try other.run(m, &engine, &collect);
            defer gpa.free(got.tokens);
            try other.turn(got.tokens);
            hall_spill.flush(&engine);
            try testing.expectEqual(2, store.written);
        }
        {
            const session: *session_file.Session = try .create(gpa, io, m, &tokens, &.{}, options);
            defer session.destroy();
            hall_spill.learnt.spilling(session, &store);
            const fresh: *session_file.Session = try .create(gpa, io, m, &tokens, &.{}, .{ .capacity = context, .prompt_rows = 32, .snapshots = 0 });
            defer fresh.destroy();
            var record: [2][count]i32 = undefined;
            const reply = try session.reply(other.prompt.items, greedy, .{ .max_tokens = count, .depth = 3, .starts = other.starts.items, .record = &record[0] }, counting.sink());
            const alone = try fresh.reply(other.prompt.items, greedy, .{ .max_tokens = count, .depth = 3, .record = &record[1] }, counting.sink());
            try testing.expect(reply.cached >= 141 and reply.cached < 150);
            try testing.expectEqual(2, store.read);
            try testing.expectEqualSlices(i32, &record[1], &record[0]);
            try testing.expectEqual(alone.drafts, reply.drafts);
            try testing.expectEqualSlices(u16, fresh.logits, session.logits);
        }
    }
}

test "a server's states over its life: a sequence its session left is carried into the pool when it opens and goes to disk with the pool's when it stops, and the next load's session starts each conversation's next turn from its file" {
    const gpa = testing.allocator;
    const io = testing.io;
    const small: fx.Small = .set();
    defer small.restore();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(gpa, io, tmp.dir, 1, 1 << 30);
    defer store.close();
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    const options: hall_file.EngineOptions = .{
        .slots = 2,
        .session = .{ .capacity = context, .prompt_rows = 32, .reply_stride = 16 },
        .pool = (extents.lowerBound(context, 2) + 8) * extents.positions,
        .chunk = 8,
    };
    const Core = hall_file.Core;
    const drive = struct {
        /// What a lone stream's prompt did not run, a slot each.
        var told: [4]?usize = @splat(null);

        fn heldOf(_: *anyopaque, slot: usize, cached: usize) void {
            told[slot] = cached;
        }

        /// The requests admitted on `slots`, to their ends.
        fn drain(core: *Core, held: *Collect, slots: []const usize) !void {
            var spins: usize = 0;
            while (spins < 4096) : (spins += 1) {
                const done = for (slots) |slot| {
                    if (!held.ended[slot]) break false;
                } else true;
                if (done) return;
                var out = held.out();
                out.held = heldOf;
                if (core.round(out) != .ran) return error.Failed;
            }
            return error.Stuck;
        }

        /// `talk`'s prompt alone, held to a row at a time, and the conversation's next turn made.
        fn turn(core: *Core, held: *Collect, model: model_file.Model, talk: *fx.Talk) !usize {
            const taken = try core.admit(.{ .prompt = talk.prompt.items, .settings = sampled, .max_tokens = replied, .depth = 6, .opening = 6, .starts = talk.starts.items }, &.{});
            held.clear(taken.slot);
            told[taken.slot] = null;
            try drain(core, held, &.{taken.slot});
            const cold = try fixtures.serial.serialReply(talk.gpa, model, talk.prompt.items, replied, sampled);
            defer talk.gpa.free(cold);
            try testing.expectEqualSlices(i32, cold, held.lists[taken.slot].items);
            _ = core.retire(taken.slot, .ended, cold.len);
            try talk.turn(cold);
            return told[taken.slot] orelse taken.cached;
        }
    };
    var talks: [3]fx.Talk = undefined;
    var made: usize = 0;
    defer for (talks[0..made]) |*talk| talk.deinit();
    for (&talks, 0..) |*talk, i| {
        talk.* = try .init(gpa, 40 + i);
        made += 1;
    }
    {
        const core = try Core.create(gpa, io, m, &tokens, &.{}, options);
        defer core.destroy();
        core.spilling(&store);
        // two conversations on the lone session: it stands on the second and has left the first
        for (talks[0..2]) |*talk| try testing.expectEqual(0, try drive.turn(core, &collect, m, talk));
        try testing.expect(core.engine == null);
        try testing.expectEqual(0, store.written);
        // two streams at once open the pool: the sequence left is carried with the one the session stood on
        const other = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
        const first = try core.admit(.{ .prompt = talks[2].prompt.items, .settings = sampled, .max_tokens = replied, .depth = 6, .opening = 6, .starts = talks[2].starts.items }, &.{});
        const second = try core.admit(.{ .prompt = &other, .settings = sampled, .max_tokens = 8, .depth = 6, .opening = 6 }, &.{});
        collect.clear(first.slot);
        collect.clear(second.slot);
        try drive.drain(core, &collect, &.{ first.slot, second.slot });
        try testing.expect(core.engine != null);
        try testing.expectEqual(0, store.written);
        try talks[2].turn(collect.lists[first.slot].items);
        _ = core.retire(first.slot, .ended, replied);
        _ = core.retire(second.slot, .ended, 8);
    }
    // the stop wrote the three the pool held: the two carried and the one that ran in it
    try testing.expectEqual(3, store.written);
    const core = try Core.create(gpa, io, m, &tokens, &.{}, options);
    defer core.destroy();
    core.spilling(&store);
    for (&talks, 1..) |*talk, n| {
        const cached = try drive.turn(core, &collect, m, talk);
        try testing.expect(cached >= 141 and cached < 150);
        try testing.expectEqual(n, store.read);
    }
}

test "--learn: the pool begins a prompt from the shared prefix's file the lone session wrote at the request's kept, with the cold reply" {
    const gpa = testing.allocator;
    const io = testing.io;
    const worth = hall_spill.learnt.worth;
    defer hall_spill.learnt.worth = worth;
    hall_spill.learnt.worth = 16;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(gpa, io, tmp.dir, 1, 1 << 30);
    defer store.close();
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    const options: hall_file.EngineOptions = .{
        .slots = 2,
        .session = .{ .capacity = context, .prompt_rows = 32, .reply_stride = 16 },
        .pool = (extents.lowerBound(context, 2) + 8) * extents.positions,
        .chunk = 8,
    };
    const Core = hall_file.Core;
    const kept = starts_of_150[2];
    const drive = struct {
        var told: [4]?usize = @splat(null);

        fn heldOf(_: *anyopaque, slot: usize, cached: usize) void {
            told[slot] = cached;
        }

        fn drain(core: *Core, held: *Collect, slots: []const usize) !void {
            var spins: usize = 0;
            while (spins < 4096) : (spins += 1) {
                const done = for (slots) |slot| {
                    if (!held.ended[slot]) break false;
                } else true;
                if (done) return;
                var out = held.out();
                out.held = heldOf;
                if (core.round(out) != .ran) return error.Failed;
            }
            return error.Stuck;
        }
    };
    var talk: fx.Talk = try .init(gpa, 93);
    defer talk.deinit();
    {
        // the load that learns: the lone session writes the state where the request's shared prefix ends
        const core = try Core.create(gpa, io, m, &tokens, &.{}, options);
        defer core.destroy();
        core.session.?.learned = &store;
        const taken = try core.admit(.{ .prompt = talk.prompt.items, .settings = sampled, .max_tokens = replied, .depth = 6, .opening = 6, .starts = talk.starts.items, .kept = kept }, &.{});
        collect.clear(taken.slot);
        try drive.drain(core, &collect, &.{taken.slot});
        _ = core.retire(taken.slot, .ended, replied);
        try testing.expectEqual(1, store.written);
    }
    // the load after: a second stream opens the pool, and a prompt that shares the prefix and no more begins
    // there from the file
    const core = try Core.create(gpa, io, m, &tokens, &.{}, options);
    defer core.destroy();
    core.session.?.learned = &store;
    var shares = try talk.prompt.clone(gpa);
    defer shares.deinit(gpa);
    for (shares.items[kept + 1 ..], 0..) |*token, i| token.* = @intCast(1 + i % 200);
    const other = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const first = try core.admit(.{ .prompt = &other, .settings = sampled, .max_tokens = 8, .depth = 6, .opening = 6 }, &.{});
    const second = try core.admit(.{ .prompt = shares.items, .settings = sampled, .max_tokens = replied, .depth = 6, .opening = 6, .starts = starts_of_150[0..3], .kept = kept }, &.{});
    try testing.expect(core.engine != null);
    collect.clear(first.slot);
    collect.clear(second.slot);
    drive.told = @splat(null);
    try drive.drain(core, &collect, &.{ first.slot, second.slot });
    try testing.expectEqual(1, store.read);
    try testing.expectEqual(kept, drive.told[second.slot] orelse second.cached);
    const cold = try fixtures.serial.serialReply(gpa, m, shares.items, replied, sampled);
    defer gpa.free(cold);
    try testing.expectEqualSlices(i32, cold, collect.lists[second.slot].items);
    _ = core.retire(first.slot, .ended, 8);
    _ = core.retire(second.slot, .ended, replied);
}
