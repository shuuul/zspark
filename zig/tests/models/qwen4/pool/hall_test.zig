//! `hall.zig`'s tests: replies served in shared rounds against each alone, one live stream against the
//! session, the caches' one storage, forks and kept states in the pool, a constrained reply cut and resumed.
//! The replies' ends, the pool's sizes and the wide prompt chunk are `hall_streams_test.zig`'s.

const std = @import("std");
const fixtures = @import("../testing.zig");
const hall_fixture = @import("hall_fixture.zig");
const greedy = hall_fixture.greedy;
const Collect = hall_fixture.Collect;
const engineFor = hall_fixture.engineFor;
const Bench = hall_fixture.Bench;
const coreFor = hall_fixture.coreFor;
const askFor = hall_fixture.askFor;
const openPool = hall_fixture.openPool;
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const request_file = @import("../../../../src/server/request.zig");
const grammar = @import("../../../../src/text/grammar.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const gdn = @import("../../../../src/models/qwen4/blocks/gdn.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const stage_file = @import("../../../../src/models/qwen4/pool/rounds_stage.zig");
const session = @import("../../../../src/models/qwen4/session/session.zig");
const Engine = hall_engine.Engine;
const Core = hall_file.Core;
const Out = hall_file.Out;
const default_parallel = hall_file.default_parallel;
const layout_file = @import("../../../../src/models/qwen4/layout.zig");
const head_storage = layout_file.head_storage;

/// The trunk's storage `layout.storageFor` gives. It reads nothing of the config.
fn storageFor(parallel: usize, asked: ?attention.Storage) attention.Storage {
    return storageOn(parallel, 1, asked);
}

fn storageOn(parallel: usize, ranks: usize, asked: ?attention.Storage) attention.Storage {
    const config: @import("../../../../src/models/qwen4/config.zig").Config = undefined;
    return layout_file.storageFor(config, parallel, ranks, asked).trunk;
}

fn runPrompts(engine: *Engine, prompts: []const []const i32, count: usize) ![][]i32 {
    for (prompts, 0..) |prompt, i| {
        _ = try engine.begin(i, .{ .prompt = prompt, .settings = greedy, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, 0);
    }
    const lists = try engine.gpa.alloc(std.ArrayList(i32), prompts.len);
    defer engine.gpa.free(lists);
    const ended = try engine.gpa.alloc(bool, prompts.len);
    defer engine.gpa.free(ended);
    for (lists) |*list| list.* = .empty;
    @memset(ended, false);
    var collect: Collect = .{ .gpa = engine.gpa, .lists = lists, .ended = ended };
    const out: Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail };
    var spins: usize = 0;
    while (spins < 64) : (spins += 1) {
        const done = for (ended) |flag| {
            if (!flag) break false;
        } else true;
        if (done) break;
        switch (engine.round(out)) {
            .ran => engine.publishAhead(out),
            .short => return error.Short,
            .failed => return error.Failed,
        }
    }
    const out_tokens = try engine.gpa.alloc([]i32, prompts.len);
    for (lists, ended, out_tokens) |*list, flag, *row| {
        try testing.expect(flag);
        row.* = try engine.gpa.dupe(i32, list.items);
        list.deinit(engine.gpa);
    }
    for (prompts, 0..) |_, i| _ = engine.retire(i, .ended, count);
    return out_tokens;
}

fn freeRows(gpa: Allocator, rows: [][]i32) void {
    for (rows) |row| gpa.free(row);
    gpa.free(rows);
}

test "shared rounds: two replies served equal each alone" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    const prompts = [_][]const i32{ &.{ 1, 2, 3, 4 }, &.{ 8, 9, 10 } };
    const count: usize = 6;
    const serial_a = try fixtures.serial.serialReply(gpa, m, prompts[0], count, greedy);
    defer gpa.free(serial_a);
    const serial_b = try fixtures.serial.serialReply(gpa, m, prompts[1], count, greedy);
    defer gpa.free(serial_b);

    var together = try engineFor(gpa, m, 2, 8);
    together.bindPool();
    defer together.deinit();
    const both = try runPrompts(&together, &prompts, count);
    defer freeRows(gpa, both);
    try testing.expectEqualSlices(i32, serial_a, both[0]);
    try testing.expectEqualSlices(i32, serial_b, both[1]);

    var alone = try engineFor(gpa, m, 1, 8);
    alone.bindPool();
    defer alone.deinit();
    const one = try runPrompts(&alone, prompts[0..1], count);
    defer freeRows(gpa, one);
    const two = try runPrompts(&alone, prompts[1..2], count);
    defer freeRows(gpa, two);
    try testing.expectEqualSlices(i32, both[0], one[0]);
    try testing.expectEqualSlices(i32, both[1], two[0]);
}

test "shared rounds over extents never zeroed: two replies over an attending trunk, the extents and every scratch NaN patterns as made, equal each alone" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    try testing.expect(m.attention != null);
    const prompts = [_][]const i32{ &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, &.{ 8, 9, 10 } };
    const count: usize = 8;
    const serial_a = try fixtures.serial.serialReply(gpa, m, prompts[0], count, greedy);
    defer gpa.free(serial_a);
    const serial_b = try fixtures.serial.serialReply(gpa, m, prompts[1], count, greedy);
    defer gpa.free(serial_b);
    // `attention.Pool.init` fills the extents as it makes them
    cuda.Scratch.poisoned = true;
    defer cuda.Scratch.poisoned = false;
    var together = try engineFor(gpa, m, 2, 8);
    together.bindPool();
    defer together.deinit();
    const both = try runPrompts(&together, &prompts, count);
    defer freeRows(gpa, both);
    try testing.expectEqualSlices(i32, serial_a, both[0]);
    try testing.expectEqualSlices(i32, serial_b, both[1]);
    // the extents the first two left, taken again
    const again = try runPrompts(&together, &.{ prompts[1], prompts[0] }, count);
    defer freeRows(gpa, again);
    try testing.expectEqualSlices(i32, serial_b, again[0]);
    try testing.expectEqualSlices(i32, serial_a, again[1]);
}

test "a slot's next window is asked ahead from its first token on, a draft at a time, as the head's passes know them" {
    // spec 146 WS-06: `begin` set the count to 0 and went on through `extend`, which asks nothing of a window
    // not begun, so no pass gathered a row and the stage gathered them all
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    var engine = try engineFor(gpa, m, 2, 8);
    defer engine.deinit();
    var ahead = try stage_file.Ahead.init(gpa, m, engine.slots.len);
    defer ahead.deinit(gpa);
    try testing.expect(ahead.branches > 0);
    ahead.begin(m, engine.slots, 1, 5);
    try testing.expectEqual(1, ahead.asked[1]);
    const one = ahead.gathers[ahead.branches].held();
    try testing.expect(one > 0);
    ahead.extend(m, engine.slots, 1, 6);
    try testing.expectEqual(2, ahead.asked[1]);
    try testing.expectEqual(2 * one, ahead.gathers[ahead.branches].held());
    try testing.expectEqual(0, ahead.asked[0]);
}

test "shared rounds: a served prompt chunk equals the reply alone" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var long: [24]i32 = undefined;
    for (&long, 0..) |*token, i| token.* = @intCast(i + 1);
    const short = [_]i32{ 40, 41 };
    const count: usize = 4;
    const serial_long = try fixtures.serial.serialReply(gpa, m, &long, count, greedy);
    defer gpa.free(serial_long);
    const prompts = [_][]const i32{ &short, &long };

    var together = try engineFor(gpa, m, 2, 8);
    together.bindPool();
    defer together.deinit();
    const both = try runPrompts(&together, &prompts, count);
    defer freeRows(gpa, both);
    try testing.expectEqualSlices(i32, serial_long, both[1]);

    var alone = try engineFor(gpa, m, 1, 8);
    alone.bindPool();
    defer alone.deinit();
    const one = try runPrompts(&alone, prompts[1..2], count);
    defer freeRows(gpa, one);
    try testing.expectEqualSlices(i32, both[1], one[0]);
}

test "one live stream under --parallel holds the tokens --parallel 1 holds: a second turn, an edited message, a prompt sent again" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const core = try coreFor(gpa, m, &tokens, &.{}, 4, 256);
    defer core.destroy();
    try testing.expect(core.captured().count > 0);
    try testing.expect(core.captured().nanoseconds > 0);
    try testing.expectEqual(core.session.?.captured().nanoseconds, core.captured().nanoseconds);
    // `--parallel 1`'s session, made as `serve` makes it for the same model
    const alone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, core.options.session);
    defer alone.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();

    const count = 12;
    var random: std.Random.DefaultPrng = .init(51);
    var first: [100]i32 = undefined;
    for (&first) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var message: [16]i32 = undefined;
    for (&message, 0..) |*token, i| token.* = @intCast(100 + i);
    const sampled: sampling.Sampling = .{ .seed = 9, .temperature = 0.8 };
    var record: [count]i32 = undefined;
    var prompt: std.ArrayList(i32) = .empty;
    defer prompt.deinit(gpa);
    var starts: std.ArrayList(usize) = .empty;
    defer starts.deinit(gpa);

    // A turn keeps `keep` of the prompt before it, then, as Qwen's template does when a reply enters the
    // history, another last token, the reply and a new message; an edit keeps `keep` and adds other tokens.
    // `messages` of the starts before it stand, and the added tokens are a message.
    const Turn = struct { keep: usize, reply: bool, add: []const i32, settings: sampling.Sampling, messages: usize, held: usize };
    const turns = [_]Turn{
        // met first: two messages, the second from 60. `held` is what `--parallel 1` holds of each prompt:
        // the one before it but its last two tokens, and a message's start
        .{ .keep = 0, .reply = false, .add = &first, .settings = greedy, .messages = 0, .held = 0 },
        .{ .keep = 99, .reply = true, .add = &message, .settings = sampled, .messages = 2, .held = 98 },
        // the last message edited
        .{ .keep = 99 + 1 + count, .reply = false, .add = &.{ 7, 6, 5, 4, 3, 2, 1, 9 }, .settings = greedy, .messages = 2, .held = 98 },
        // and sent again
        .{ .keep = 99 + 1 + count + 8, .reply = false, .add = &.{}, .settings = greedy, .messages = 3, .held = 118 },
        // the first turn's second message edited, after the token that opens it
        .{ .keep = 61, .reply = false, .add = &message, .settings = sampled, .messages = 1, .held = 60 },
    };
    for (turns, 0..) |turn, nth| {
        prompt.shrinkRetainingCapacity(turn.keep);
        starts.shrinkRetainingCapacity(turn.messages);
        if (nth == 0) {
            try starts.appendSlice(gpa, &.{ 0, 60 });
            try prompt.appendSlice(gpa, turn.add);
        } else {
            if (turn.reply) {
                try prompt.append(gpa, 251);
                try prompt.appendSlice(gpa, &record);
            }
            if (turn.add.len > 0 and turn.keep != 61) try starts.append(gpa, prompt.items.len);
            try prompt.appendSlice(gpa, turn.add);
        }
        const expected = try fixtures.serial.serialReply(gpa, m, prompt.items, count, turn.settings);
        defer gpa.free(expected);
        var ask = askFor(prompt.items, turn.settings, count);
        ask.starts = starts.items;
        const got = try bench.one(core, ask);
        defer gpa.free(got.tokens);
        try testing.expectEqualSlices(i32, expected, got.tokens);
        try testing.expectEqual(request_file.Finish.length, got.finish);
        var unasked: Bench = .{ .gpa = gpa };
        const reply = try alone.reply(prompt.items, turn.settings, .{ .max_tokens = count, .starts = starts.items, .record = &record }, .{ .context = &unasked, .writeFn = Core.unasked });
        try testing.expectEqualSlices(i32, expected, &record);
        // what the request is told is what `--parallel 1` tells it
        try testing.expectEqual(reply.cached, got.cached);
        try testing.expectEqual(turn.held, got.cached);
        try testing.expectEqual(nth + 1, core.reports);
    }
    // the second turn held the first prompt but its last tokens, and the session was never left
    try testing.expect(core.engine == null);
}

test "a load's caches are one storage: a request's tokens are the same before a second stream, as it comes, beside three others and after" {
    // what `serve` resolves, and what `--parallel 1` and `--kv-dtype` keep
    try testing.expectEqual(attention.Storage.fp8, storageFor(default_parallel, null));
    try testing.expectEqual(attention.Storage.fp8, storageFor(2, null));
    try testing.expectEqual(attention.Storage.bf16, storageFor(1, null));
    // two ranks hold half the KV heads each: bf16 at any --parallel (spec 144 WS-01)
    try testing.expectEqual(attention.Storage.bf16, storageOn(default_parallel, 2, null));
    try testing.expectEqual(attention.Storage.fp8, storageOn(default_parallel, 2, .fp8));
    try testing.expectEqual(attention.Storage.bf16, storageFor(default_parallel, .bf16));
    try testing.expectEqual(attention.Storage.fp8_nvfp4, storageFor(1, .fp8_nvfp4));
    // the head's own cache: bf16 whatever `--kv-dtype` asks (spec 052)
    try testing.expectEqual(attention.Storage.bf16, head_storage);

    const gpa = testing.allocator;
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    defer fixtures.random_model.caches = .bf16;
    defer fixtures.random_model.head_caches = null;
    // The pool's arrays are not zeroed (spec 101), and the rows no round wrote are compared too: they are the
    // poisoning's bytes in every load, not what an earlier load left where this one's pool lands.
    cuda.Scratch.poisoned = true;
    defer cuda.Scratch.poisoned = false;
    // The head's pool after the rounds below on an all-bf16 load. The random trunk does not attend, so a
    // head in bf16 beside coded caches reads the same rows and leaves these bytes, in the pool as alone.
    var plain: [3]?[]u8 = @splat(null);
    defer for (plain) |held| if (held) |bytes| gpa.free(bytes);
    // bf16 as `--kv-dtype bf16` asks, the second what the defaults give; then the head beside NVFP4 values,
    // and what an explicit `--kv-dtype` gives: the head bf16 beside it
    for ([_]Storages{
        .{ .kv = storageFor(default_parallel, .bf16), .head = head_storage },
        .{ .kv = storageFor(default_parallel, null), .head = head_storage },
        .{ .kv = storageFor(default_parallel, .fp8_nvfp4), .head = head_storage },
        .{ .kv = storageFor(1, .fp8), .head = head_storage },
    }) |case| {
        fixtures.random_model.caches = case.kv;
        fixtures.random_model.head_caches = case.head;
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 80);
        defer m.deinit();
        try sameBeforeBesideAfter(gpa, m, &tokens, case, &plain);
    }
}

const Storages = hall_fixture.Storages;
const sameBeforeBesideAfter = hall_fixture.sameBeforeBesideAfter;

test "the token that ends a reply is not given: alone and in the pool a reply's tokens and their count are the same, and each leaves its line" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const prompt = [_]i32{ 1, 2, 3, 4 };
    const serial = try fixtures.serial.serialReply(gpa, m, &prompt, 12, greedy);
    defer gpa.free(serial);
    // a reply that ends at its sixth token or before, and one that ends at its first
    for ([_]usize{ 5, 0 }) |at| {
        const eos = [1]u32{@intCast(serial[at])};
        const ends = std.mem.indexOfScalar(i32, serial, serial[at]).?;
        const core = try coreFor(gpa, m, &tokens, &eos, 2, 256);
        defer core.destroy();
        var bench: Bench = .{ .gpa = gpa };
        defer bench.deinit();

        const lone = try bench.one(core, askFor(&prompt, greedy, 12));
        defer gpa.free(lone.tokens);
        try testing.expectEqualSlices(i32, serial[0..ends], lone.tokens);
        try testing.expectEqual(request_file.Finish.stop, lone.finish);
        // the line `--parallel 1` writes
        try testing.expectEqual(1, core.reports);
        try testing.expect(std.mem.startsWith(u8, core.line[0..core.line_len], "prompt 4 tokens, 0 held, in "));
        try testing.expect(std.mem.indexOf(u8, core.line[0..core.line_len], "snapshots") != null);

        try openPool(&bench, core);
        try testing.expectEqual(3, core.reports);
        const pooled = try bench.one(core, askFor(&prompt, greedy, 12));
        defer gpa.free(pooled.tokens);
        try testing.expectEqualSlices(i32, lone.tokens, pooled.tokens);
        try testing.expectEqual(request_file.Finish.stop, pooled.finish);
        try testing.expectEqual(4, core.reports);
        var want: [64]u8 = undefined;
        // the session's state before the prompt's end was carried into the pool: the prompt goes on from it
        const head = try std.fmt.bufPrint(&want, "prompt 4 tokens, 2 held, in 0.00 s; reply {d} tokens in ", .{ends});
        try testing.expect(std.mem.startsWith(u8, core.line[0..core.line_len], head));
        try testing.expect(std.mem.indexOf(u8, core.line[0..core.line_len], "in the pool beside 0") != null);
    }
}

test "in the pool a reply's end is kept: the next turn runs from it, and a prompt that goes on otherwise from a state below" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const count = 12;
    var random: std.Random.DefaultPrng = .init(52);
    var first: [40]i32 = undefined;
    for (&first) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    const sampled: sampling.Sampling = .{ .seed = 3, .temperature = 0.8 };

    var cached: [2]usize = undefined;
    for ([_]bool{ true, false }, &cached) |keep_ends, *found| {
        const core = try coreFor(gpa, m, &tokens, &.{}, 2, 256);
        defer core.destroy();
        var bench: Bench = .{ .gpa = gpa };
        defer bench.deinit();
        try openPool(&bench, core);
        try testing.expect(core.captured().nanoseconds > 0);
        try testing.expectEqual(core.engine.?.graphs.cache.capturing(), core.captured().nanoseconds);
        core.engine.?.keep_ends = keep_ends;

        const opening = try bench.one(core, askFor(&first, greedy, count));
        defer gpa.free(opening.tokens);
        try testing.expectEqual(0, opening.cached);
        var prompt: std.ArrayList(i32) = .empty;
        defer prompt.deinit(gpa);
        try prompt.appendSlice(gpa, &first);
        try prompt.appendSlice(gpa, opening.tokens);
        try prompt.appendSlice(gpa, &.{ 100, 101, 102, 103 });
        const expected = try fixtures.serial.serialReply(gpa, m, prompt.items, count, sampled);
        defer gpa.free(expected);
        const second = try bench.one(core, askFor(prompt.items, sampled, count));
        defer gpa.free(second.tokens);
        try testing.expectEqualSlices(i32, expected, second.tokens);
        found.* = second.cached;
        if (!keep_ends) continue;
        // the reply's rows but its last token's, which no row has run: what a session's state holds
        try testing.expectEqual(first.len + count - 1, second.cached);

        // The reply's last token is another: the head took the first one in with the state at the reply's
        // end, so this prompt starts below it, and its tokens are an empty state's all the same.
        prompt.shrinkRetainingCapacity(first.len + count - 1);
        try prompt.appendSlice(gpa, &.{ 200, 201 });
        const parted = try fixtures.serial.serialReply(gpa, m, prompt.items, count, greedy);
        defer gpa.free(parted);
        const third = try bench.one(core, askFor(prompt.items, greedy, count));
        defer gpa.free(third.tokens);
        try testing.expectEqualSlices(i32, parted, third.tokens);
        try testing.expect(third.cached < first.len + count - 1);
    }
    // without it the turn starts from the prompt's last chunk
    try testing.expect(cached[1] < first.len and cached[1] < cached[0]);
}

test "a fork at an extent's edge copies no row and gives the reply an empty pool gives" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const count = 6;
    var random: std.Random.DefaultPrng = .init(53);
    var long: [extents.positions + 8]i32 = undefined;
    for (&long) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var edited = long;
    edited[extents.positions + 4] = 251;

    var replies: [2][]i32 = undefined;
    for (&replies, 0..) |*reply, nth| {
        const core = try coreFor(gpa, m, &tokens, &.{}, 2, 1024);
        defer core.destroy();
        var bench: Bench = .{ .gpa = gpa };
        defer bench.deinit();
        try openPool(&bench, core);
        if (nth == 0) {
            // the prompt's chunks of 32 rows leave a state at 512, where an extent ends
            const met = try bench.one(core, askFor(&long, greedy, count));
            gpa.free(met.tokens);
        }
        const got = try bench.one(core, askFor(&edited, greedy, count));
        reply.* = got.tokens;
        try testing.expectEqual(@as(usize, if (nth == 0) extents.positions else 0), got.cached);
    }
    defer for (replies) |reply| gpa.free(reply);
    try testing.expectEqualSlices(i32, replies[1], replies[0]);
}

test "a constrained reply cut for a request that waits goes on under its constraint from the tokens it sent, in its session and moved to the pool" {
    const json_schema = @import("../../../../src/text/json_schema.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const eos = [1]u32{tokens.addedId("<|end|>").?};
    var engine: json_schema.Engine = try .init(gpa, &tokens, m.shape.vocab, &eos);
    defer engine.deinit(gpa);
    var schema_diagnostic: grammar.Diagnostic = .{};
    const schema =
        \\{"type":"object","properties":{"ok":{"type":"boolean"},"kind":{"enum":["a","ab","abc",12,null]},"list":{"type":"array","items":{"type":"boolean"},"maxItems":3}},"required":["ok","kind"]}
    ;
    const constraint = try engine.interface().compile(gpa, .{ .json_schema = schema }, &schema_diagnostic);
    defer constraint.deinit();
    const mask = try gpa.alloc(u32, constraint.words);
    defer gpa.free(mask);

    const count = 96;
    const prompt = [_]i32{ 5, 17, 200, 3, 3, 91 };
    const sampled: sampling.Sampling = .{ .seed = 11, .temperature = 0.9 };
    const expected = try fixtures.serial.serialConstrained(gpa, m, &prompt, count, sampled, constraint, null, &eos);
    defer gpa.free(expected);
    const ended = expected[expected.len - 1] == eos[0];
    const document = expected[0 .. expected.len - @intFromBool(ended)];

    var ask = askFor(&prompt, sampled, count);
    ask.constraint = constraint;
    ask.mask = mask;
    {
        // The request that waited has gone before it was admitted: the lone stream goes on in its session,
        // from the state and under the constraint where its tokens left them.
        const core = try coreFor(gpa, m, &tokens, &eos, 2, 256);
        defer core.destroy();
        var bench: Bench = .{ .gpa = gpa };
        defer bench.deinit();
        const live = try core.admit(ask, &.{});
        bench.reset(live.slot);
        bench.wait_at = 2;
        _ = core.round(bench.out());
        bench.wait_at = null;
        const sent = bench.lists[live.slot].items.len;
        try testing.expect(sent >= 2 and sent < document.len and !bench.ended[live.slot]);
        try bench.drain(core, &.{live.slot});
        try testing.expect(core.engine == null);
        try testing.expectEqualSlices(i32, document, bench.lists[live.slot].items);
        // told once, what its prompt found: nothing, and not the rows its own reply left
        try testing.expectEqual(0, bench.held[live.slot].?);
        _ = core.retire(live.slot, .ended, document.len);
        try testing.expectEqual(1, core.reports);
        var want: [64]u8 = undefined;
        const head = try std.fmt.bufPrint(&want, "prompt 6 tokens, 0 held, in ", .{});
        try testing.expect(std.mem.startsWith(u8, core.line[0..core.line_len], head));
        const tail = try std.fmt.bufPrint(&want, "; reply {d} tokens in ", .{document.len});
        try testing.expect(std.mem.indexOf(u8, core.line[0..core.line_len], tail) != null);
    }
    const core = try coreFor(gpa, m, &tokens, &eos, 2, 256);
    defer core.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();
    const live = try core.admit(ask, &.{});
    bench.reset(live.slot);
    bench.wait_at = 2;
    _ = core.round(bench.out());
    bench.wait_at = null;
    const sent = bench.lists[live.slot].items.len;
    // the document is under way, neither at its start nor ended
    try testing.expect(sent >= 2 and sent < document.len and !bench.ended[live.slot]);
    const other = try core.admit(askFor(&.{ 40, 41, 42 }, greedy, 4), &.{});
    bench.reset(other.slot);
    try testing.expect(core.engine != null);
    // carried into the pool, it goes on from its last token's row and not from its prompt (spec 054)
    try testing.expectEqual(prompt.len + sent - 1, core.engine.?.seats[live.slot].?.cached);
    try bench.drain(core, &.{ live.slot, other.slot });
    try testing.expectEqualSlices(i32, document, bench.lists[live.slot].items);
    try testing.expectEqual(if (ended) request_file.Finish.stop else request_file.Finish.length, bench.finish[live.slot].?);
    _ = core.retire(live.slot, .ended, document.len);
    _ = core.retire(other.slot, .ended, 4);
}

test "in the pool a wide prompt chunk leaves every layer its own conv window: the reply is serial's, and so is one decoding beside the chunk" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 64);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const count = 24;
    const sampled: sampling.Sampling = .{ .seed = 3, .temperature = 0.8 };
    var random: std.Random.DefaultPrng = .init(54);
    var long: [72]i32 = undefined;
    for (&long) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    const short = [_]i32{ 40, 41, 42 };
    const serial_long = try fixtures.serial.serialReply(gpa, m, &long, count, sampled);
    defer gpa.free(serial_long);
    const serial_short = try fixtures.serial.serialReply(gpa, m, &short, count, sampled);
    defer gpa.free(serial_short);

    // A chunk of `gdn.wide_rows` rows or more takes the wide kernels, which write every layer's projection
    // into one buffer. The model has more DeltaNet layers than one, or the last layer's rows would do.
    try testing.expect(gdn.wide_rows <= 40);
    var deltas: usize = 0;
    for (m.layers) |layer| deltas += @intFromBool(layer.block == .delta_net);
    try testing.expect(deltas >= 2);

    // alone in the pool, the prompt in chunks of 40 and 32 rows; then beside a reply, in chunks of 40 and 32
    // with that reply's window in the round
    for ([_]bool{ false, true }) |beside| {
        const core = try Core.create(gpa, testing.io, m, &tokens, &.{}, .{
            .slots = 2,
            .session = .{ .capacity = 256, .prompt_rows = 40 },
            .pool = extents.lowerBound(256, 2) * extents.positions,
            .chunk = 40,
        });
        defer core.destroy();
        var bench: Bench = .{ .gpa = gpa };
        defer bench.deinit();
        try openPool(&bench, core);
        try testing.expect(core.engine.?.buffers.projection_wide != null);
        if (!beside) {
            const got = try bench.one(core, askFor(&long, sampled, count));
            defer gpa.free(got.tokens);
            try testing.expectEqualSlices(i32, serial_long, got.tokens);
            continue;
        }
        const first = try core.admit(askFor(&short, sampled, count), &.{});
        bench.reset(first.slot);
        // the short prompt's reply is under way when the long prompt comes
        for (0..3) |_| _ = core.round(bench.out());
        try testing.expect(bench.lists[first.slot].items.len >= 1 and !bench.ended[first.slot]);
        const second = try core.admit(askFor(&long, sampled, count), &.{});
        bench.reset(second.slot);
        try bench.drain(core, &.{ first.slot, second.slot });
        try testing.expectEqualSlices(i32, serial_short, bench.lists[first.slot].items);
        try testing.expectEqualSlices(i32, serial_long, bench.lists[second.slot].items);
        _ = core.retire(first.slot, .ended, count);
        _ = core.retire(second.slot, .ended, count);
    }
}
