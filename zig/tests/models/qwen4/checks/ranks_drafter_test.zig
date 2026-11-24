//! The `ranks-check DIR reference=REF drafter=DIR` wiring (spec 126 WS-01) as the loopback reference holds it:
//! `ranks_drafter.Given` over two ranks in one process — the drafter of an opened checkpoint loaded and
//! `initHeld` beside a pair's model and buffers, given to the pair (`pair.flash`), its drafted replies the
//! pair's serial ones, greedy and sampled, and the ranks clean after each. The drafter's checkpoint is the
//! tests' random one under the published names, read through a `Checkpoint` as the opened file is, so the
//! wiring is the run's minus the disk. What loopback cannot show stays N1's: the NCCL telling, rank 1's
//! mirror, the real checkpoint's files.

const std = @import("std");
const testing = std.testing;
const sampling = @import("../../../../src/decode/sampling.zig");
const config_file = @import("../../../../src/decode/dflash2/config.zig");
const random = @import("../../../../src/models/qwen4/checks/random.zig");
const drafter_testing = @import("../../../../src/models/qwen4/checks/random_drafter.zig");
const ranks_drafter = @import("../../../../src/models/qwen4/checks/ranks_drafter.zig");
const rig = @import("../../../../src/models/qwen4/dflash2/rig.zig");
const pair_file = @import("../../../../src/models/qwen4/session/pair.zig");
const safetensors = @import("../../../../src/weights/safetensors.zig");

const Both = pair_file.Of(2);
const world = config_file.world;
/// Wide enough that a round can take the drafter's block (`depth` is the pair's `max_rows` less one).
const rows_cap = 16;
const capacity = 64;

/// The random main model, its pair, and the same drafter the run opens: a `Checkpoint` of the random
/// tensors under the published names, `Opened` over it, `Given` beside the pair.
const Held = struct {
    arena_state: std.heap.ArenaAllocator,
    pair: *Both,
    opened: rig.Opened,
    given: *ranks_drafter.Given(world),
    tensors: std.StringArrayHashMapUnmanaged(safetensors.Tensor),

    fn init(self: *Held, taps: []const u32, stored: bool) !void {
        const gpa = testing.allocator;
        self.arena_state = .init(gpa);
        errdefer self.arena_state.deinit();
        const arena = self.arena_state.allocator();
        const built = try random.shares(2, gpa, arena, 42, 0, rows_cap);
        self.pair = try Both.createLoopback(gpa, .{ .ranks = built, .max_rows = rows_cap }, rows_cap, capacity, true);
        errdefer self.pair.destroy();
        var c = try drafter_testing.config(arena, random.fx.shape.hidden, random.fx.shape.vocab, random.fx.layers, taps);
        c.block = 8;
        var source: drafter_testing.Source = .empty;
        var prng: std.Random.DefaultPrng = .init(468);
        try drafter_testing.put(arena, prng.random(), c, &source);
        self.tensors = .empty;
        errdefer self.tensors.deinit(gpa);
        var it = source.iterator();
        while (it.next()) |entry| try self.tensors.put(gpa, entry.key_ptr.*, entry.value_ptr.*);
        self.opened = .{ .path = "the tests' random drafter", .config = c, .checkpoint = .{ .shards = &.{}, .tensors = self.tensors, .gpa = gpa } };
        self.given = try ranks_drafter.Given(world).init(gpa, &self.opened, self.pair, stored, false);
    }

    fn deinit(self: *Held) void {
        const gpa = testing.allocator;
        self.given.deinit(gpa);
        self.pair.destroy();
        self.tensors.deinit(gpa);
        self.arena_state.deinit();
    }
};

/// Target ranks and the drafter's context stand at the same position after a reply.
fn expectClean(held: *Held) !void {
    const pair = held.pair;
    try testing.expectEqual(pair.state.ranks[0].pos, pair.state.ranks[1].pos);
    try testing.expectEqual(pair.state.pos, pair.state.ranks[0].pos);
    try testing.expectEqual(pair.state.pos + 1, pair.seen.items.len);
    try testing.expectEqual(pair.state.pos, held.given.flash.pair.context_end);
}

test "Given drafts from an opened drafter the pair's serial reply, greedy" {
    const gpa = testing.allocator;
    var held: Held = undefined;
    try held.init(&.{ 0, 2 }, true);
    defer held.deinit();
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 10;
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };

    try testing.expect(held.pair.flash == &held.given.flash.d);
    const serial = try held.pair.reply(&prompt, settings, &.{}, .{}, .{}, null, count, null, null);
    const serial_ids = try gpa.dupe(i32, held.pair.tokens.items);
    defer gpa.free(serial_ids);

    try held.pair.reset();
    held.given.flash.d.reset(settings);
    const from = held.pair.tokens.items.len;
    const drafted = try held.pair.reply(&prompt, settings, &.{}, .{ .given = held.given.flash.d.drafter() }, .{}, null, count, null, null);
    try testing.expectEqual(serial.tokens, drafted.tokens);
    try testing.expectEqualSlices(i32, serial_ids, held.pair.tokens.items[from..]);
    try testing.expect(drafted.stats.drafted > 0);
    try expectClean(&held);
}

test "a second reply, sampled, after the first's reset is serial again" {
    const gpa = testing.allocator;
    var held: Held = undefined;
    try held.init(&.{ 0, 2 }, true);
    defer held.deinit();
    const prompt = [_]i32{ 9, 40, 3 };
    const count = 8;
    const first: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const second: sampling.Sampling = .{ .seed = 5, .temperature = 0.8 };

    held.given.flash.d.reset(first);
    _ = try held.pair.reply(&prompt, first, &.{}, .{ .given = held.given.flash.d.drafter() }, .{}, null, count, null, null);
    try expectClean(&held);

    try held.pair.reset();
    held.given.flash.d.reset(second);
    const serial_from = held.pair.tokens.items.len;
    const serial = try held.pair.reply(&prompt, second, &.{}, .{}, .{}, null, count, null, null);
    const serial_ids = try gpa.dupe(i32, held.pair.tokens.items[serial_from..]);
    defer gpa.free(serial_ids);
    try held.pair.reset();
    held.given.flash.d.reset(second);
    const from = held.pair.tokens.items.len;
    const drafted = try held.pair.reply(&prompt, second, &.{}, .{ .given = held.given.flash.d.drafter() }, .{}, null, count, null, null);
    try testing.expectEqual(serial.tokens, drafted.tokens);
    try testing.expectEqualSlices(i32, serial_ids, held.pair.tokens.items[from..]);
    try testing.expect(drafted.stats.drafted > 0);
    try expectClean(&held);
}

test "the pair drafts by its own DFlash2 where no drafter is given (serve --drafter on two ranks): two turns, the second going on from the first, the serial replies" {
    const gpa = testing.allocator;
    var held: Held = undefined;
    try held.init(&.{ 0, 2 }, true);
    defer held.deinit();
    try testing.expect(held.pair.head == null);
    const prompt = [_]i32{ 7, 21, 200, 3, 3 };
    const count = 10;
    const samplings = [_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 5, .temperature = 0.8 } };
    for (samplings) |settings| {
        // the second turn: the first's prompt, its reply and two more tokens, which go on from its state
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var replies: [2][2][]i32 = undefined;
        var drafted: u64 = 0;
        for ([_]usize{ 0, 7 }, 0..) |depth, mode| {
            try held.pair.reset();
            held.given.flash.d.reset(settings);
            var turn: std.ArrayList(i32) = .empty;
            try turn.appendSlice(arena, &prompt);
            for (0..2) |t| {
                const from = held.pair.tokens.items.len;
                const reply = try held.pair.reply(turn.items, settings, &.{}, .{ .depth = depth }, .{}, null, count, null, null);
                if (depth > 0) drafted += reply.stats.drafted;
                replies[t][mode] = try arena.dupe(i32, held.pair.tokens.items[from..]);
                try turn.appendSlice(arena, replies[t][mode]);
                try turn.appendSlice(arena, &.{ 33, 34 });
            }
            // the drafter's context stands where the state does (a serial reply absorbs nothing into it)
            if (depth > 0) try expectClean(&held);
        }
        for (replies) |pair| try testing.expectEqualSlices(i32, pair[0], pair[1]);
        try testing.expect(drafted > 0);
    }
}

test "the drafter stored and mixed drafts the serial reply (spec 135)" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3, 9 };
    const count = 12;
    const settings: sampling.Sampling = .{ .seed = 3, .temperature = 0.8 };
    for ([_]bool{ true, false }) |stored| {
        var held: Held = undefined;
        try held.init(&.{ 0, 2 }, stored);
        defer held.deinit();
        // the mixed drafter proposes by its four-bit head, its picks scored against the borrowed stored rows;
        // the stored drafter borrows the target's head itself and owns no proposer copy
        try testing.expectEqual(!stored, held.given.flash.d.heads != null);
        if (held.given.flash.d.heads) |heads| try testing.expect(heads[0] == .four);
        try testing.expectEqual(stored, held.given.flash.w[0].layers[0].q == .bf16);
        const serial_from = held.pair.tokens.items.len;
        _ = try held.pair.reply(&prompt, settings, &.{}, .{}, .{}, null, count, null, null);
        const serial = try gpa.dupe(i32, held.pair.tokens.items[serial_from..]);
        defer gpa.free(serial);
        try held.pair.reset();
        held.given.flash.d.reset(settings);
        const from = held.pair.tokens.items.len;
        const reply = try held.pair.reply(&prompt, settings, &.{}, .{ .given = held.given.flash.d.drafter() }, .{}, null, count, null, null);
        try testing.expectEqual(count, reply.tokens);
        try testing.expect(reply.stats.drafted > 0);
        try testing.expectEqualSlices(i32, serial, held.pair.tokens.items[from..][0..count]);
        try expectClean(&held);
    }
}
