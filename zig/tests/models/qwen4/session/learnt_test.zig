//! `learnt.zig`: a state written by one session and read by another gives the reply a pass over the whole
//! prompt gives, on each cache storage; what is never read; the probe. And the states that leave a session
//! (`--spill-gib`): written as they go and when it stops, read by the next load's.

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

const learnt_fixture = @import("learnt_fixture.zig");
const count = learnt_fixture.count;
const harness = learnt_fixture.harness;
const opener = learnt_fixture.opener;
const greedy = learnt_fixture.greedy;
const fx = learnt_fixture.fx;
const readBack = learnt_fixture.readBack;
const spiltBack = learnt_fixture.spiltBack;

test "a state written by one session is read by the next: a head over DeltaNet layers, each cache storage" {
    const gpa = testing.allocator;
    defer fixtures.random_model.caches = .bf16;
    for ([_]@TypeOf(fixtures.random_model.caches){ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
        defer m.deinit();
        try readBack(m);
    }
}

test "a state written by one session is read by the next: an attention layer and a PLE" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    try readBack(m);
}

test "a state that leaves a session is written, and the next load's session goes on from it: a head over DeltaNet layers, each cache storage" {
    const gpa = testing.allocator;
    defer fixtures.random_model.caches = .bf16;
    for ([_]@TypeOf(fixtures.random_model.caches){ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
        defer m.deinit();
        try spiltBack(m);
    }
}

test "a state that leaves a session is written, and the next load's session goes on from it: an attention layer and a PLE" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    try spiltBack(m);
}

test "one block store: a harness's learned state and a conversation's spilt one over it write the harness's blocks once, under the learned root, and the next load reads both: a head over DeltaNet layers, and an attention layer and a PLE" {
    const gpa = testing.allocator;
    {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
        defer m.deinit();
        try bothBack(m);
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    try bothBack(m);
}

/// `--learn` and `--spill-gib` on one load, as `learnt.open` opens them: one `learned.Blocks` under the
/// learned root, each kind's state files under its own.
const Both = struct {
    shared: *learned.Blocks,
    learn: learnt.Store,
    spill: learnt.Store,

    fn open(learn_root: std.Io.Dir, spill_root: std.Io.Dir) !Both {
        const io = testing.io;
        const shared: *learned.Blocks = try .open(testing.allocator, io, learn_root, 1, .{ .learned = true, .spilled = true, .spilled_root = spill_root });
        errdefer shared.close();
        var learn: learnt.Store = try .openKind(testing.allocator, io, learn_root, 1, 1 << 30, .learned, shared);
        errdefer learn.close();
        const spill: learnt.Store = try .openKind(testing.allocator, io, spill_root, 1, 1 << 30, .spilled, shared);
        try learn.sweep();
        return .{ .shared = shared, .learn = learn, .spill = spill };
    }

    fn close(self: *Both) void {
        self.learn.close();
        self.spill.close();
        self.shared.close();
    }

    /// The blocks the two kinds' states name, each once, and how many both name.
    fn union_(self: *const Both) !struct { all: usize, both: usize } {
        var seen: std.AutoHashMapUnmanaged(learned_blocks.Stem, u2) = .empty;
        defer seen.deinit(testing.allocator);
        for ([_]*const learnt.Store{ &self.learn, &self.spill }, [_]u2{ 1, 2 }) |store, bit| for (store.held.items) |entry| for (entry.blocks) |one| {
            const slot = try seen.getOrPut(testing.allocator, one);
            slot.value_ptr.* = if (slot.found_existing) slot.value_ptr.* | bit else bit;
        };
        var both: usize = 0;
        var it = seen.valueIterator();
        while (it.next()) |bits| both += @intFromBool(bits.* == 3);
        return .{ .all = seen.count(), .both = both };
    }
};

fn blocksIn(dir: std.Io.Dir) !usize {
    var n: usize = 0;
    var entries = dir.iterate();
    while (try entries.next(testing.io)) |entry| n += @intFromBool(std.mem.endsWith(u8, entry.name, ".blk"));
    return n;
}

/// Each block file's inode: a block written again is another file, renamed over it.
fn inodesIn(dir: std.Io.Dir, out: *std.StringHashMapUnmanaged(std.Io.File.INode)) !void {
    var entries = dir.iterate();
    while (try entries.next(testing.io)) |entry| if (std.mem.endsWith(u8, entry.name, ".blk")) {
        const stat = try dir.statFile(testing.io, entry.name, .{});
        try out.put(testing.allocator, try testing.allocator.dupe(u8, entry.name), stat.inode);
    };
}

fn bothBack(m: model.Model) !void {
    const gpa = testing.allocator;
    const io = testing.io;
    const spill = @import("../../../../src/decode/spill.zig");
    const floor, const worth, const spilt_worth, const block = .{ spill.floor, learnt.worth, spill.worth, learned_blocks.block_tokens };
    defer spill.floor, learnt.worth, spill.worth, learned_blocks.block_tokens = .{ floor, worth, spilt_worth, block };
    // the harness's 60 positions: three blocks of 16 that the conversation's state names too
    spill.floor, learnt.worth, spill.worth, learned_blocks.block_tokens = .{ 16, 16, 16, 16 };
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const learn_root = try tmp.dir.createDirPathOpen(io, "learned", .{ .open_options = .{ .iterate = true } });
    defer learn_root.close(io);
    const spill_root = try tmp.dir.createDirPathOpen(io, "spilled", .{ .open_options = .{ .iterate = true } });
    defer spill_root.close(io);
    var talk: fx.Bench = try .init(m, &tokens);
    defer talk.deinit();
    // the same harness, another first message
    var again: fx.Bench = try .init(m, &tokens);
    defer again.deinit();
    again.random = .init(77);
    {
        var both: Both = try .open(learn_root, spill_root);
        defer both.close();
        const first: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer first.destroy();
        first.learned = &both.learn;
        learnt.spilling(first, &both.spill);
        try talk.begin(true);
        try testing.expectEqual(0, try talk.ask(first, harness));
        try testing.expectEqual(1, both.learn.written);
        var harnessed: std.StringHashMapUnmanaged(std.Io.File.INode) = .empty;
        defer {
            var names = harnessed.keyIterator();
            while (names.next()) |name| gpa.free(name.*);
            harnessed.deinit(gpa);
        }
        try inodesIn(both.learn.dir, &harnessed);
        try testing.expectEqual(3, harnessed.count());
        try talk.turn();
        _ = try talk.ask(first, harness);
        learnt.flush(first, true);
        try testing.expect(both.spill.written >= 1);
        // the spilt states name the harness's blocks and leave their files as they were
        var it = harnessed.iterator();
        while (it.next()) |one| try testing.expectEqual(one.value_ptr.*, (try both.learn.dir.statFile(io, one.key_ptr.*, .{})).inode);
        const named = try both.union_();
        try testing.expect(named.both >= 3);
        try testing.expectEqual(named.all, try blocksIn(both.learn.dir));
        try testing.expectEqual(0, try blocksIn(both.spill.dir));
    }
    {
        // the next load: a new conversation reads the harness's state, the one that left goes on from its own
        var both: Both = try .open(learn_root, spill_root);
        defer both.close();
        const named = try both.union_();
        try testing.expectEqual(named.all, try blocksIn(both.learn.dir));
        const second: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer second.destroy();
        second.learned = &both.learn;
        learnt.spilling(second, &both.spill);
        try again.begin(true);
        try testing.expectEqual(harness, try again.ask(second, harness));
        try testing.expectEqual(1, both.learn.read);
        try talk.turn();
        try testing.expect(try talk.ask(second, harness) > harness);
        try testing.expectEqual(1, both.spill.read);
    }
}

test "a state that left a session and the writer has still to write is read from its memory, then from its file, each time with the whole prompt's reply: a head over DeltaNet layers, each cache storage" {
    const gpa = testing.allocator;
    defer fixtures.random_model.caches = .bf16;
    for ([_]@TypeOf(fixtures.random_model.caches){ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
        defer m.deinit();
        try stillListed(m);
    }
}

test "a state that left a session and the writer has still to write is read from its memory, then from its file, each time with the whole prompt's reply: an attention layer and a PLE" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    try stillListed(m);
}

/// `spiltBack`'s six conversations with the writer held: the one that went is no file, and its next turn on
/// the same session starts from the writer's memory. Once the writer is let go and the session stops, every
/// state is a file, and another load's session takes the same conversation's turn after from its file.
fn stillListed(m: model.Model) !void {
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
    {
        const first: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
        defer first.destroy();
        learnt.spilling(first, &store);
        const writer = store.writer.?;
        writer.hold(true);
        for (&benches) |*bench| {
            try bench.begin(false);
            _ = try bench.ask(first, null);
            try bench.turn();
            _ = try bench.ask(first, null);
        }
        try testing.expectEqual(1, store.written);
        try testing.expectEqual(0, try binsIn(io, store.dir));
        const prompt = benches[0].prompt.items;
        const at = store.longest(prompt, prompt.len).? - 1;
        try benches[0].turn();
        try testing.expectEqual(at, try benches[0].ask(first, null));
        try testing.expectEqual(1, store.read);
        try testing.expectEqual(0, try binsIn(io, store.dir));
        try testing.expectEqual(0, writer.done);
        writer.hold(false);
        learnt.flush(first, true);
        try testing.expectEqual(store.written, try binsIn(io, store.dir));
        try testing.expectEqual(store.written, writer.done);
        try testing.expectEqual(0, writer.jobs.items.len);
    }
    const second: *Session = try .create(gpa, io, m, &tokens, &.{}, fx.options);
    defer second.destroy();
    learnt.spilling(second, &store);
    const prompt = benches[0].prompt.items;
    const at = store.longest(prompt, prompt.len).? - 1;
    try benches[0].turn();
    try testing.expectEqual(at, try benches[0].ask(second, null));
    try testing.expectEqual(2, store.read);
}

fn binsIn(io: std.Io, dir: std.Io.Dir) !usize {
    var n: usize = 0;
    var entries = dir.iterate();
    while (try entries.next(io)) |entry| n += @intFromBool(std.mem.endsWith(u8, entry.name, ".st"));
    return n;
}

test "the probe's identity is one load's: the same twice, another's with other weights, and the session serves after it" {
    const gpa = testing.allocator;
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var ids: [3]u64 = undefined;
    for (&ids, [_]bool{ false, false, true }) |*id, tiny| {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = if (tiny) try fixtures.tiny_model.tinyModel(gpa, arena.allocator()) else try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
        defer m.deinit();
        const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, fx.options);
        defer session.destroy();
        id.* = try learnt.identity(session, 5);
        try testing.expectEqual(id.*, try learnt.identity(session, 5));
        try testing.expect(id.* != try learnt.identity(session, 6));
        var bench: fx.Bench = try .init(m, &tokens);
        defer bench.deinit();
        try bench.begin(true);
        try testing.expectEqual(0, try bench.ask(session, harness));
    }
    try testing.expectEqual(ids[0], ids[1]);
    try testing.expect(ids[0] != ids[2]);
}
