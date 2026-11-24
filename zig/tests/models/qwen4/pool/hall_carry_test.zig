//! Tests of the hand-over (`hall_carry.zig`, spec 054 WS-03): what a session held when a second stream opened
//! the pool is in the pool afterwards, bit for bit what a slot run a row at a time holds there; the stream
//! that was live goes on from its row with the reply it would have had; the next requests of the session's
//! conversation run the rows a session would run; and the store's bounds hold over what was carried.

const std = @import("std");
const fixtures = @import("../testing.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const session = @import("../../../../src/models/qwen4/session/session.zig");
const Ask = hall_file.Ask;
const Core = hall_file.Core;
const Engine = hall_engine.Engine;
const fetchBytes = fixtures.attending.fetchBytes;

const carry_fixture = @import("hall_carry_fixture.zig");
const greedy = carry_fixture.greedy;
const sampled = carry_fixture.sampled;
const context = carry_fixture.context;
const reply_stride = carry_fixture.reply_stride;
const starts_of_150 = carry_fixture.starts_of_150;
const Bench = carry_fixture.Bench;
const coreFor = carry_fixture.coreFor;
const coreRows = carry_fixture.coreRows;
const askFor = carry_fixture.askFor;
const statesOf = carry_fixture.statesOf;
const expectRows = carry_fixture.expectRows;
const Way = carry_fixture.Way;
const handOver = carry_fixture.handOver;
const before = carry_fixture.before;
const tinyTokens = carry_fixture.tinyTokens;

test "a second stream opens the pool: the session's states and rows are carried as a serial slot's, the live stream goes on from its row, and the conversation's next turns run a session's rows, on each cache storage" {
    try carriedOnEachStorage();
}

fn carriedOnEachStorage() !void {
    const gpa = testing.allocator;
    var tokens = try tinyTokens(gpa);
    defer tokens.deinit(gpa);
    defer fixtures.random_model.caches = .bf16;
    for ([_]attention.Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        errdefer std.debug.print("caches {t}\n", .{storage});
        fixtures.random_model.caches = storage;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
        defer m.deinit();
        try handOver(gpa, m, &tokens, .live);
        try handOver(gpa, m, &tokens, .idle);
        try handOver(gpa, m, &tokens, .parted);
    }
}

test "a second stream opens the pool over a trunk that attends and a PLE: the rows of the trunk's caches are carried too" {
    try carriedOverTrunk();
}

fn carriedOverTrunk() !void {
    const gpa = testing.allocator;
    var tokens = try tinyTokens(gpa);
    defer tokens.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    try handOver(gpa, m, &tokens, .live);
    try handOver(gpa, m, &tokens, .idle);
    try handOver(gpa, m, &tokens, .parted);
}

/// The reply of `count` tokens to `prompt`, and the logits its last token was chosen from: from a session
/// that is never left, or from a stream moved to the pool once it has sent `wait_at` tokens. The pool's rows
/// settled on the device (`Engine.settleRound`, spec 144 WS-03) are never read back, so its row is the last
/// one read whole: `settle` false reads every row whole, and the row is the last one's; true settles them, as
/// served, and at least one row must have been.
fn lastRow(gpa: Allocator, m: model_file.Model, tokens: *const tokenizer.Tokenizer, prompt: []const i32, count: usize, wait_at: ?usize, pool_rows: ?usize, settle: bool, reply: *std.ArrayList(i32)) ![]u16 {
    const core = try coreRows(gpa, m, tokens, 2, pool_rows);
    defer core.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();
    try before(&bench, core, prompt[0], null, &.{});
    const live = try core.admit(askFor(prompt, sampled, count, &.{}), &.{});
    bench.reset(live.slot);
    reply.clearRetainingCapacity();
    if (wait_at) |n| {
        bench.wait_at = n;
        _ = core.round(bench.out());
        bench.wait_at = null;
        const sent = bench.lists[live.slot].items.len;
        try testing.expect(sent >= n and sent < count and !bench.ended[live.slot]);
        const other = try core.admit(askFor(&.{ 40, 41, 42 }, greedy, 3, &.{}), &.{});
        bench.reset(other.slot);
        if (pool_rows) |rows| try testing.expectEqual(rows, core.engine.?.m.max_rows);
        core.engine.?.settle_on_device = settle;
        try testing.expectEqual(prompt.len + sent - 1, core.engine.?.seats[live.slot].?.cached);
        try bench.drain(core, &.{ live.slot, other.slot });
        try testing.expectEqual(settle, core.engine.?.settled > 0);
        try reply.appendSlice(gpa, bench.lists[live.slot].items);
        return gpa.dupe(u16, core.engine.?.logits[live.slot]);
    }
    try bench.drain(core, &.{live.slot});
    try testing.expect(core.engine == null);
    try reply.appendSlice(gpa, bench.lists[live.slot].items);
    return gpa.dupe(u16, core.session.?.logits);
}

test "a live stream moved to the pool mid-reply: its remaining tokens and the logits each was chosen from are those of the reply never moved" {
    const gpa = testing.allocator;
    var tokens = try tinyTokens(gpa);
    defer tokens.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var random: std.Random.DefaultPrng = .init(56);
    var prompt: [70]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var stayed: std.ArrayList(i32) = .empty;
    defer stayed.deinit(gpa);
    var moved: std.ArrayList(i32) = .empty;
    defer moved.deinit(gpa);
    // a reply's last row read back whole, so each length gives the row of one more position; and the reply
    // again with the pool's rows settled on the device, as served
    for (14..30) |count| {
        errdefer std.debug.print("a reply of {d} tokens\n", .{count});
        const want = try lastRow(gpa, m, &tokens, &prompt, count, null, null, false, &stayed);
        defer gpa.free(want);
        const got = try lastRow(gpa, m, &tokens, &prompt, count, 4, null, false, &moved);
        defer gpa.free(got);
        try testing.expectEqualSlices(i32, stayed.items, moved.items);
        try testing.expectEqualSlices(u16, want, got);
        gpa.free(try lastRow(gpa, m, &tokens, &prompt, count, 4, null, true, &moved));
        try testing.expectEqualSlices(i32, stayed.items, moved.items);
    }
}

test "a session's windows wider than the pool's rounds (--max-num-batched-tokens): a stream moved to the pool mid-reply gives the reply and logits of the default's, never moved" {
    const gpa = testing.allocator;
    var tokens = try tinyTokens(gpa);
    defer tokens.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // loaded for 96 rows: the session's prompts run in windows of 96, the pool's rounds are made for 48
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 96);
    defer m.deinit();
    var random: std.Random.DefaultPrng = .init(57);
    var prompt: [130]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var stayed: std.ArrayList(i32) = .empty;
    defer stayed.deinit(gpa);
    var moved: std.ArrayList(i32) = .empty;
    defer moved.deinit(gpa);
    for ([_]usize{ 14, 21, 29 }) |count| {
        errdefer std.debug.print("a reply of {d} tokens\n", .{count});
        const want = try lastRow(gpa, m, &tokens, &prompt, count, null, null, false, &stayed);
        defer gpa.free(want);
        const got = try lastRow(gpa, m, &tokens, &prompt, count, 4, 48, false, &moved);
        defer gpa.free(got);
        try testing.expectEqualSlices(i32, stayed.items, moved.items);
        try testing.expectEqualSlices(u16, want, got);
        gpa.free(try lastRow(gpa, m, &tokens, &prompt, count, 4, 48, true, &moved));
        try testing.expectEqualSlices(i32, stayed.items, moved.items);
    }
}

test "more states than the store's budget holds: the newest are carried, the bound holds and no buffer is left" {
    const gpa = testing.allocator;
    var tokens = try tinyTokens(gpa);
    defer tokens.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var random: std.Random.DefaultPrng = .init(57);
    var prompt: [150]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    try cuda.synchronize();
    const buffers = cuda.account.buffers;
    const bytes = cuda.account.bytes;
    {
        const core = try coreFor(gpa, m, &tokens, 2);
        defer core.destroy();
        var bench: Bench = .{ .gpa = gpa };
        defer bench.deinit();
        try before(&bench, core, prompt[0], null, &.{});
        _ = try bench.one(core, askFor(&prompt, sampled, 44, &starts_of_150));
        const held = core.session.?;
        try testing.expect(held.unused.items.len + held.spare.items.len >= 2);
        const states = statesOf(held);
        try testing.expect(states >= 5);
        const each = held.snapshots.items[0].bytes();
        // three of them and their pooled keys, of the five and more
        core.options.states_bytes = 3 * each + each / 2;
        const top = held.state.pos;
        const a = try core.admit(askFor(&.{ 3, 1, 4, 1, 5 }, greedy, 6, &.{}), &.{});
        const b = try core.admit(askFor(&.{ 9, 8, 7 }, greedy, 6, &.{}), &.{});
        const engine = &core.engine.?;
        try testing.expect(engine.keeps.bytes() <= core.options.states_bytes);
        try testing.expectEqual(3, engine.keeps.list.items.len);
        try testing.expectEqual(3, engine.store.marks.items.len);
        // the store's order let the lowest go: the end stays, and each one kept stands above each that went
        var lowest: usize = top;
        for (engine.keeps.list.items) |keep| lowest = @min(lowest, keep.pos);
        try testing.expect(engine.keeps.at(engine.keeps.list.items[0].chain, top) != null and lowest > 90);
        bench.reset(a.slot);
        bench.reset(b.slot);
        try bench.drain(core, &.{ a.slot, b.slot });
        _ = core.retire(a.slot, .ended, 6);
        _ = core.retire(b.slot, .ended, 6);
        try testing.expect(engine.keeps.bytes() <= core.options.states_bytes);
    }
    try testing.expectEqual(buffers, cuda.account.buffers);
    try testing.expectEqual(bytes, cuda.account.bytes);
}
