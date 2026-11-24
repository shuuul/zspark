//! What `hall_carry_test.zig` and `checkpoint_prefix_test.zig` share (spec 133 WS-02 moved it here): the
//! scheduler a core reports to, a core as `serve` makes it, and the hand-over (`handOver`): what a session
//! held when a second stream opened the pool is in the pool as a serial slot's, and the stream that was live
//! goes on with the reply it would have had.

const std = @import("std");
const fixtures = @import("../testing.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const request_file = @import("../../../../src/server/request.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const keep_fixture = @import("hall_keep_fixture.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const session = @import("../../../../src/models/qwen4/session/session.zig");
const Ask = hall_file.Ask;
const Core = hall_file.Core;
const Engine = hall_engine.Engine;
const fetchBytes = fixtures.attending.fetchBytes;

pub const greedy: sampling.Sampling = .{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
pub const sampled: sampling.Sampling = .{ .seed = 7, .temperature = 0.8 };
pub const context = 512;
pub const reply_stride = 16;
pub const starts_of_150 = [_]usize{ 0, 37, 90, 141 };

/// What the scheduler is to a core: the tokens given each slot, and a request that waits once slot 0 has
/// been given `wait_at` tokens.
pub const Bench = struct {
    gpa: Allocator,
    lists: [4]std.ArrayList(i32) = @splat(.empty),
    ended: [4]bool = @splat(false),
    held: [4]?usize = @splat(null),
    failed: bool = false,
    wait_at: ?usize = null,

    pub fn deinit(self: *Bench) void {
        for (&self.lists) |*list| list.deinit(self.gpa);
    }

    pub fn reset(self: *Bench, slot: usize) void {
        self.lists[slot].clearRetainingCapacity();
        self.ended[slot] = false;
        self.held[slot] = null;
    }

    pub fn out(self: *Bench) hall_file.Out {
        return .{ .ctx = self, .emit = emit, .end = end, .fail = fail, .held = told, .waiting = waiting };
    }

    pub fn emit(ctx: *anyopaque, slot: usize, tokens: []const i32) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.lists[slot].appendSlice(self.gpa, tokens) catch @panic("oom");
    }

    pub fn end(ctx: *anyopaque, slot: usize, _: request_file.Finish) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
    }

    pub fn fail(ctx: *anyopaque, slot: usize, message: []const u8) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
        self.failed = true;
        std.debug.print("slot {d} failed: {s}\n", .{ slot, message });
    }

    pub fn told(ctx: *anyopaque, slot: usize, cached: usize) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.held[slot] = cached;
    }

    pub fn waiting(ctx: *anyopaque) bool {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        return if (self.wait_at) |n| self.lists[0].items.len >= n else false;
    }

    pub fn drain(self: *Bench, core: *Core, slots: []const usize) !void {
        var spins: usize = 0;
        while (spins < 2048) : (spins += 1) {
            const done = for (slots) |slot| {
                if (!self.ended[slot]) break false;
            } else true;
            if (done) break;
            if (core.round(self.out()) != .ran) return error.Failed;
        }
        for (slots) |slot| try testing.expect(self.ended[slot]);
        try testing.expect(!self.failed);
    }

    /// One request from its admission to its end with nothing beside it: how much of its prompt was held.
    pub fn one(self: *Bench, core: *Core, ask: Ask) !usize {
        const taken = try core.admit(ask, &.{});
        self.reset(taken.slot);
        try self.drain(core, &.{taken.slot});
        _ = core.retire(taken.slot, .ended, self.lists[taken.slot].items.len);
        return self.held[taken.slot] orelse taken.cached;
    }
};

pub fn coreFor(gpa: Allocator, m: model_file.Model, tokens: *const tokenizer.Tokenizer, slots: usize) !*Core {
    return coreRows(gpa, m, tokens, slots, null);
}

/// `coreFor`, or, with `pool_rows`, as `serve --max-num-batched-tokens` makes it (spec 105 WS-07): the lone
/// session's prompt windows the model's rows, the pool's rounds `pool_rows`.
pub fn coreRows(gpa: Allocator, m: model_file.Model, tokens: *const tokenizer.Tokenizer, slots: usize, pool_rows: ?usize) !*Core {
    return Core.create(gpa, testing.io, m, tokens, &.{}, .{
        .slots = slots,
        .session = .{ .capacity = context, .prompt_rows = if (pool_rows != null) m.max_rows else 32, .reply_stride = reply_stride },
        .pool = (extents.lowerBound(context, slots) + 8) * extents.positions,
        .chunk = 8,
        .rows = pool_rows,
    });
}

pub fn askFor(prompt: []const i32, settings: sampling.Sampling, count: usize, starts: []const usize) Ask {
    return .{ .prompt = prompt, .settings = settings, .max_tokens = count, .depth = session.default_depth, .opening = session.default_depth, .starts = starts };
}

/// How many states `held` hands over: its snapshots, those on its reply's stride and the one it stands at,
/// which may be one of them.
pub fn statesOf(held: *const session.Session) usize {
    var n: usize = 1;
    for ([_][]const @import("../../../../src/models/qwen4/forward.zig").Snapshot{ held.snapshots.items, held.strides.items }) |list| {
        for (list) |one| n += @intFromBool(one.pos != held.state.pos);
    }
    return n;
}

/// The rows of `sequence` below its last token that `engine` holds on the chain its carried states stand on
/// are the rows a slot leaves that ran `sequence` a row a round: every array of each attention layer's pool
/// and of the head's.
pub fn expectRows(gpa: Allocator, m: model_file.Model, engine: *Engine, sequence: []const i32) !void {
    const pos = sequence.len - 1;
    const carried = for (engine.keeps.list.items) |keep| {
        if (!keep.chain.live and keep.chain.len() == pos) break keep.chain;
    } else return error.NothingCarried;
    try testing.expectEqualSlices(i32, sequence[0..pos], carried.tokens.items);
    var serial = try keep_fixture.open(gpa, m, .{ .slots = 1, .context = context, .prompt_rows = 1, .reply_stride = 0 });
    serial.bindPool();
    defer serial.deinit();
    serial.keeps.chunks = false;
    var collect: keep_fixture.Collect = .{ .gpa = gpa };
    defer collect.deinit();
    const longer = try gpa.alloc(i32, sequence.len + 1);
    defer gpa.free(longer);
    @memcpy(longer[0..sequence.len], sequence);
    longer[sequence.len] = 1;
    _ = try serial.begin(0, keep_fixture.ask(longer, greedy, 1, &.{}, null), 0);
    while (serial.slots[0].pos < pos) try keep_fixture.step(&serial, &collect);
    try testing.expectEqual(pos, serial.slots[0].pos);
    const chain = serial.seats[0].?.chain;
    var compared: usize = 0;
    for (0..engine.trunk.len + @intFromBool(engine.head_pool != null)) |i| {
        const got: attention.Pool = if (i < engine.trunk.len) engine.trunk[i] else engine.head_pool.?;
        const want: attention.Pool = if (i < engine.trunk.len) serial.trunk[i] else serial.head_pool.?;
        for ([_]?cuda.Buffer{ got.k, got.v, got.k_scale, got.v_scale, got.index_keys }, [_]?cuda.Buffer{ want.k, want.v, want.k_scale, want.v_scale, want.index_keys }) |a, b| {
            const mine = try fetchBytes(gpa, a orelse continue);
            defer gpa.free(mine);
            const theirs = try fetchBytes(gpa, b.?);
            defer gpa.free(theirs);
            const row = a.?.bytes() / @as(usize, @intCast(a.?.view.size[0]));
            for (0..pos) |p| {
                errdefer std.debug.print("layer {d}, position {d} of {d}\n", .{ i, p, pos });
                try testing.expect(std.mem.eql(u8, mine[carried.row(p) * row ..][0..row], theirs[chain.row(p) * row ..][0..row]));
            }
            // the rows are written, not a pool's zeros
            try testing.expect(!std.mem.allEqual(u8, mine[carried.row(0) * row ..][0 .. pos * row], 0));
            compared += 1;
        }
    }
    try testing.expect(compared >= 3);
    serial.preempt(0);
}

pub const Way = enum {
    /// The second stream comes while the first one's reply is under way.
    live,
    /// It comes with a first one that has not run a row: the session holds the conversation before, whole.
    idle,
    /// As `live`, with a conversation before that shares the prompt's first two messages: the one the session
    /// left parted from the carried one there, and its rows below come from the session's caches.
    parted,
};

/// A prompt of 150 tokens in four messages and its reply of 44 on `m`, the pool opened `way`. Then what the
/// session held is in the pool as a serial slot's, the stream that was live gives the serial reply from the
/// row it stood at, and the conversation's next turn and an earlier message edited run at most 15 rows more
/// than a session that was never left runs of them. The conversation the session left before is carried too
/// and goes on from its states.
pub fn handOver(gpa: Allocator, m: model_file.Model, tokens: *const tokenizer.Tokenizer, way: Way) !void {
    errdefer std.debug.print("the pool opened {t}\n", .{way});
    var random: std.Random.DefaultPrng = .init(54);
    var prompt: [150]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    const count = 44;
    const expected = try fixtures.serial.serialReply(gpa, m, &prompt, count, sampled);
    defer gpa.free(expected);
    // the next turn, and the first prompt's third message edited after its first token
    var turn: std.ArrayList(i32) = .empty;
    defer turn.deinit(gpa);
    try turn.appendSlice(gpa, &prompt);
    try turn.appendSlice(gpa, expected);
    const turn_start = turn.items.len;
    for (0..16) |i| try turn.append(gpa, @intCast(100 + i));
    var edited: [120]i32 = undefined;
    @memcpy(edited[0..91], prompt[0..91]);
    for (edited[91..], 0..) |*token, i| token.* = @intCast(60 + i);
    const turn_starts = starts_of_150 ++ [_]usize{turn_start};

    // what a session holds of the two, never left
    const alone: *session.Session = try .create(gpa, testing.io, m, tokens, &.{}, .{ .capacity = context, .prompt_rows = 32, .reply_stride = reply_stride });
    defer alone.destroy();
    var unasked: Bench = .{ .gpa = gpa };
    var want: [3]usize = undefined;
    for ([_][]const i32{ &prompt, turn.items, &edited }, [_][]const usize{ &starts_of_150, &turn_starts, starts_of_150[0..3] }, &want) |asked, starts, *cached| {
        cached.* = (try alone.reply(asked, sampled, .{ .max_tokens = count, .starts = starts }, .{ .context = &unasked, .writeFn = Core.unasked })).cached;
    }
    try testing.expectEqual(0, want[0]);
    try testing.expect(want[1] >= 150 + count - 2 and want[2] >= 37);

    const core = try coreFor(gpa, m, tokens, 2);
    defer core.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();
    var sequence: std.ArrayList(i32) = .empty;
    defer sequence.deinit(gpa);
    try sequence.appendSlice(gpa, &prompt);
    var slots: [2]usize = undefined;
    var first = askFor(&prompt, sampled, count, &starts_of_150);
    // the server's account of a shared prefix: the second message's start
    first.kept = starts_of_150[1];
    var earlier: std.ArrayList(i32) = .empty;
    defer earlier.deinit(gpa);
    try before(&bench, core, prompt[0], &earlier, if (way == .parted) prompt[0..starts_of_150[2]] else &.{});
    const other = [_]i32{ 9, 8, 7 };
    switch (way) {
        .live, .parted => {
            slots[0] = (try core.admit(first, &.{})).slot;
            bench.reset(slots[0]);
            bench.wait_at = 20;
            _ = core.round(bench.out());
            bench.wait_at = null;
            const sent = bench.lists[slots[0]].items.len;
            try testing.expect(sent >= 20 and sent < count and !bench.ended[slots[0]]);
            try sequence.appendSlice(gpa, bench.lists[slots[0]].items);
        },
        .idle => {
            _ = try bench.one(core, first);
            try testing.expectEqualSlices(i32, expected, bench.lists[0].items);
            try sequence.appendSlice(gpa, expected);
            slots[0] = (try core.admit(askFor(&.{ 3, 1, 4, 1, 5 }, greedy, 6, &.{}), &.{})).slot;
            bench.reset(slots[0]);
        },
    }
    const held = core.session.?;
    try testing.expect(core.engine == null and held.trusted);
    try testing.expectEqual(sequence.items.len, held.state.pos + 1);
    const states = statesOf(held);
    // a prompt's states, one on the reply's stride and the end
    try testing.expect(held.snapshots.items.len >= 2 and held.strides.items.len >= 1);
    // and buffers of the conversation before, more than the state it stands at takes (one that shares the
    // prompt's start went on in its states)
    if (way != .parted) try testing.expect(held.unused.items.len + held.spare.items.len >= 2);
    // the conversation before is a sequence the session left, with its states
    try testing.expectEqual(1, held.lot.list.items.len);
    const left = held.lot.list.items[0].states.items.len;
    try testing.expect(left >= 1);

    slots[1] = (try core.admit(askFor(&other, greedy, 6, &.{}), &.{})).slot;
    bench.reset(slots[1]);
    try testing.expect(core.session == null);
    const engine = &core.engine.?;
    try testing.expectEqual(states + left, engine.keeps.list.items.len);
    try testing.expectEqual(states + left, engine.store.marks.items.len);
    // the state where the request said a shared prefix ends is one to the store too (a prompt that went on
    // from the conversation before's state above it made none there)
    var shared: usize = 0;
    for (engine.store.marks.items) |mark| {
        try testing.expectEqual(mark.pos == starts_of_150[1] and way != .parted, mark.shared);
        shared += @intFromBool(mark.shared);
    }
    try testing.expectEqual(@intFromBool(way != .parted), shared);
    // the buffers the session kept are the store's ready ones: none was made for them
    if (way != .parted) try testing.expect(engine.keeps.spare.items.len >= 1 and engine.keeps.took.made < engine.keeps.spare.items.len);
    // each is a serial slot's state, and the rows below them a serial slot's rows
    try testing.expectEqual(states, try keep_fixture.expectSerial(gpa, m, engine, sequence.items, .{ .context = context }));
    try expectRows(gpa, m, engine, sequence.items);
    // and so is what the session left: a chain of its own, its states a serial slot's, its rows a serial slot's
    // (and, where it parted from the prompt, the carried chain's states below that are on it too)
    const along = try keep_fixture.expectSerial(gpa, m, engine, earlier.items, .{ .context = context });
    if (way == .parted) try testing.expect(along > left) else try testing.expectEqual(left, along);
    try expectRows(gpa, m, engine, earlier.items);
    if (way != .idle) {
        // the stream goes on from the row it stood at: its last token's row is the one it runs again
        try testing.expectEqual(sequence.items.len - 1, engine.seats[slots[0]].?.cached);
    }
    try bench.drain(core, &slots);
    if (way != .idle) try testing.expectEqualSlices(i32, expected, bench.lists[slots[0]].items);
    for (slots) |slot| _ = core.retire(slot, .ended, bench.lists[slot].items.len);

    for ([_][]const i32{ turn.items, &edited }, [_][]const usize{ &turn_starts, starts_of_150[0..3] }, want[1..]) |asked, starts, cached| {
        const reply = try fixtures.serial.serialReply(gpa, m, asked, count, sampled);
        defer gpa.free(reply);
        const got = try bench.one(core, askFor(asked, sampled, count, starts));
        errdefer std.debug.print("a prompt of {d}: {d} held in the pool, {d} in a session\n", .{ asked.len, got, cached });
        try testing.expectEqualSlices(i32, reply, bench.lists[0].items);
        try testing.expect(got + 15 >= cached);
    }
    // the conversation the session left goes on from its states in the pool, with the reply of a cold run (a
    // session's, as `serialReply` holds fewer rows than it)
    try earlier.appendSlice(gpa, &.{ 11, 12, 13, 14, 15, 16, 17, 18 });
    const cold = try coreFor(gpa, m, tokens, 2);
    defer cold.destroy();
    var fresh: Bench = .{ .gpa = gpa };
    defer fresh.deinit();
    try testing.expectEqual(0, try fresh.one(cold, askFor(earlier.items, sampled, count, &.{})));
    const reply = fresh.lists[0].items;
    const got = try bench.one(core, askFor(earlier.items, sampled, count, &.{}));
    errdefer std.debug.print("the conversation left: {d} of {d} held in the pool\n", .{ got, earlier.items.len });
    try testing.expectEqualSlices(i32, reply, bench.lists[0].items);
    try testing.expect(got + 8 + 15 >= earlier.items.len);
}

/// A conversation that shares nothing with the next one, whole: the session leaves it for the next (its
/// lot), and its states' buffers beyond the lot's are the session's unused ones, which the state it stands
/// at and the pool's ready buffers are made of; or one that begins with `shares`. Its prompt and reply into
/// `into`, where given.
pub fn before(bench: *Bench, core: *Core, differs: i32, into: ?*std.ArrayList(i32), shares: []const i32) !void {
    var random: std.Random.DefaultPrng = .init(58);
    var prompt: [240]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    if (prompt[0] == differs) prompt[0] += 1;
    @memcpy(prompt[0..shares.len], shares);
    try testing.expectEqual(0, try bench.one(core, askFor(&prompt, sampled, 40, &.{ 0, 34, 68, 102, 136, 170, 204 })));
    if (into) |list| {
        try list.appendSlice(bench.gpa, &prompt);
        try list.appendSlice(bench.gpa, bench.lists[0].items);
    }
    const held = core.session.?;
    try testing.expect(held.snapshots.items.len >= 7 and held.strides.items.len >= 1);
}

pub fn tinyTokens(gpa: Allocator) !tokenizer.Tokenizer {
    var diagnostic: tokenizer.Diagnostic = .{};
    return tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
}
