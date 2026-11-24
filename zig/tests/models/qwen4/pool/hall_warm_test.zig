//! `hall_warm.zig`'s tests: a follow-up after a warm pass is its cold reply and goes on from the pass's states,
//! on the lone session and in the pool; a pass steps aside for a live request within one round and is declined
//! beside one.

const std = @import("std");
const fixtures = @import("../testing.zig");
const greedy = @import("hall_fixture.zig").greedy;
const Allocator = std.mem.Allocator;
const testing = std.testing;
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const request_file = @import("../../../../src/server/request.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const session = @import("../../../../src/models/qwen4/session/session.zig");
const Ask = hall_file.Ask;
const Core = hall_file.Core;
const Out = hall_file.Out;

const count = 12;
const sampled: sampling.Sampling = .{ .seed = 7, .temperature = 0.8 };

/// What a core's rounds gave, a slot each, and whether a request waits.
const Bench = struct {
    gpa: Allocator,
    lists: [4]std.ArrayList(i32) = @splat(.empty),
    ended: [4]bool = @splat(false),
    finish: [4]?request_file.Finish = @splat(null),
    held: [4]?usize = @splat(null),
    waits: bool = false,
    failed: bool = false,

    fn deinit(self: *Bench) void {
        for (&self.lists) |*list| list.deinit(self.gpa);
    }

    fn reset(self: *Bench, slot: usize) void {
        self.lists[slot].clearRetainingCapacity();
        self.ended[slot] = false;
        self.finish[slot] = null;
        self.held[slot] = null;
    }

    fn out(self: *Bench) Out {
        return .{ .ctx = self, .emit = emit, .end = end, .fail = fail, .held = told, .waiting = waiting };
    }

    fn emit(ctx: *anyopaque, slot: usize, tokens: []const i32) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.lists[slot].appendSlice(self.gpa, tokens) catch @panic("oom");
    }

    fn end(ctx: *anyopaque, slot: usize, finish: request_file.Finish) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
        self.finish[slot] = finish;
    }

    fn fail(ctx: *anyopaque, slot: usize, message: []const u8) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
        self.failed = true;
        std.debug.print("slot {d} failed: {s}\n", .{ slot, message });
    }

    fn told(ctx: *anyopaque, slot: usize, cached: usize) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.held[slot] = cached;
    }

    fn waiting(ctx: *anyopaque) bool {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        return self.waits;
    }

    fn drain(self: *Bench, core: *Core, slots: []const usize) !void {
        var spins: usize = 0;
        while (spins < 400) : (spins += 1) {
            const done = for (slots) |slot| {
                if (!self.ended[slot]) break false;
            } else true;
            if (done) break;
            switch (core.round(self.out())) {
                .ran => {},
                .short => return error.Short,
                .failed => return error.Failed,
            }
        }
        for (slots) |slot| try testing.expect(self.ended[slot]);
        try testing.expect(!self.failed);
    }

    const Got = struct { tokens: []i32, cached: usize, finish: request_file.Finish };

    fn one(self: *Bench, core: *Core, ask: Ask) !Got {
        const taken = try core.admit(ask, &.{});
        self.reset(taken.slot);
        try self.drain(core, &.{taken.slot});
        const tokens = try self.gpa.dupe(i32, self.lists[taken.slot].items);
        _ = core.retire(taken.slot, .ended, tokens.len);
        return .{ .tokens = tokens, .cached = self.held[taken.slot] orelse taken.cached, .finish = self.finish[taken.slot].? };
    }
};

fn coreFor(gpa: Allocator, m: model_file.Model, tokens: *const tokenizer.Tokenizer) !*Core {
    const context = 256;
    return Core.create(gpa, testing.io, m, tokens, &.{}, .{
        .slots = 2,
        .session = .{ .capacity = context, .prompt_rows = 32 },
        .pool = extents.lowerBound(context, 2) * extents.positions,
        .chunk = 8,
    });
}

fn askFor(prompt: []const i32, settings: sampling.Sampling, max_tokens: usize, warm: bool) Ask {
    return .{ .prompt = prompt, .settings = settings, .max_tokens = max_tokens, .depth = session.default_depth, .opening = session.default_depth, .warm = warm };
}

/// Opens `core`'s pool: two requests at once, run to their ends.
fn openPool(bench: *Bench, core: *Core) !void {
    const a = try core.admit(askFor(&.{ 9, 8, 7 }, greedy, 2, false), &.{});
    const b = try core.admit(askFor(&.{ 6, 5 }, greedy, 2, false), &.{});
    bench.reset(a.slot);
    bench.reset(b.slot);
    try testing.expect(core.engine != null);
    try bench.drain(core, &.{ a.slot, b.slot });
    _ = core.retire(a.slot, .ended, 2);
    _ = core.retire(b.slot, .ended, 2);
}

/// A conversation's three prompts: the first turn, the warm pass (the first, its reply rendered otherwise
/// than decoded, as a template that leaves the reasoning out renders it, and the next turn's header), and
/// the next turn, which goes on from the pass's prompt.
const Talk = struct {
    first: [40]i32,
    warm: std.ArrayList(i32) = .empty,
    next: std.ArrayList(i32) = .empty,

    fn init(seed: u64) Talk {
        var talk: Talk = .{ .first = undefined };
        var random: std.Random.DefaultPrng = .init(seed);
        for (&talk.first) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
        return talk;
    }

    fn deinit(self: *Talk, gpa: Allocator) void {
        self.warm.deinit(gpa);
        self.next.deinit(gpa);
    }

    fn after(self: *Talk, gpa: Allocator, reply: []const i32) !void {
        try self.warm.appendSlice(gpa, &self.first);
        try self.warm.append(gpa, 251);
        try self.warm.appendSlice(gpa, reply[3..]);
        try self.warm.appendSlice(gpa, &.{ 252, 100, 101 });
        try self.next.appendSlice(gpa, self.warm.items);
        try self.next.appendSlice(gpa, &.{ 120, 121, 122, 123, 124, 125, 253, 102 });
    }
};

test "a follow-up after a warm pass is its cold reply, greedy and seeded, on the lone session and in the pool, and runs from two tokens before the pass's end" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);

    for ([_]bool{ false, true }) |pool| {
        for ([_]sampling.Sampling{ greedy, sampled }) |settings| {
            var cached: [2]usize = undefined;
            for ([_]bool{ true, false }, &cached) |warm, *found| {
                const core = try coreFor(gpa, m, &tokens);
                defer core.destroy();
                var bench: Bench = .{ .gpa = gpa };
                defer bench.deinit();
                if (pool) try openPool(&bench, core);
                var talk: Talk = .init(61);
                defer talk.deinit(gpa);
                const opening = try bench.one(core, askFor(&talk.first, greedy, count, false));
                defer gpa.free(opening.tokens);
                try talk.after(gpa, opening.tokens);
                if (warm) {
                    const pass = try bench.one(core, askFor(talk.warm.items, greedy, 1, true));
                    defer gpa.free(pass.tokens);
                    try testing.expect(pass.finish != .cut);
                }
                const expected = try fixtures.serial.serialReply(gpa, m, talk.next.items, count, settings);
                defer gpa.free(expected);
                const got = try bench.one(core, askFor(talk.next.items, settings, count, false));
                defer gpa.free(got.tokens);
                try testing.expectEqualSlices(i32, expected, got.tokens);
                found.* = got.cached;
                try testing.expectEqual(pool, core.engine != null);
            }
            // the pass's prompt but its last two tokens (`marks.Options.tail`); cold, the first turn's at most
            try testing.expect(cached[0] >= talk_warm_len - 2);
            try testing.expect(cached[1] <= 40);
        }
    }
}

/// The warm prompt's length in `Talk`: 40, the separator, the reply but its first three, three of the header.
const talk_warm_len = 40 + 1 + (count - 3) + 3;

test "a live request that comes while a warm pass is the lone stream waits one round, in which the pass ends unrun, and the pool stays shut; a pass beside a live stream is declined" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const core = try coreFor(gpa, m, &tokens);
    defer core.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();
    var talk: Talk = .init(62);
    defer talk.deinit(gpa);
    const opening = try bench.one(core, askFor(&talk.first, greedy, count, false));
    defer gpa.free(opening.tokens);
    try talk.after(gpa, opening.tokens);
    const other = [_]i32{ 30, 31, 32, 33, 34, 35, 36 };

    // the pass is the lone stream; a request comes before its round
    const pass = try core.admit(askFor(talk.warm.items, greedy, 1, true), &.{});
    bench.reset(pass.slot);
    try testing.expectError(error.Wait, core.admit(askFor(&other, greedy, count, false), &.{}));
    bench.waits = true;
    try testing.expect(core.round(bench.out()) == .ran);
    try testing.expect(bench.ended[pass.slot]);
    try testing.expectEqual(request_file.Finish.cut, bench.finish[pass.slot].?);
    try testing.expectEqual(0, bench.lists[pass.slot].items.len);
    _ = core.retire(pass.slot, .ended, 0);
    bench.waits = false;
    // the request is the lone stream after it, its reply its own
    const expected = try fixtures.serial.serialReply(gpa, m, &other, count, greedy);
    defer gpa.free(expected);
    const live = try core.admit(askFor(&other, greedy, count, false), &.{});
    bench.reset(live.slot);
    // beside it a pass is declined
    try testing.expectError(error.Declined, core.admit(askFor(talk.warm.items, greedy, 1, true), &.{}));
    try bench.drain(core, &.{live.slot});
    try testing.expectEqualSlices(i32, expected, bench.lists[live.slot].items);
    _ = core.retire(live.slot, .ended, count);
    try testing.expect(core.engine == null);

    // a lone pass that would run more than a prompt window past what the session ran is declined
    var long: [200]i32 = undefined;
    for (&long, 0..) |*token, i| token.* = @intCast(1 + i % 240);
    try testing.expectError(error.Declined, core.admit(askFor(&long, greedy, 1, true), &.{}));
}

test "in the pool a warm pass is ended in the round a live stream comes to, which runs the live prompt's chunk; alone with a request waiting it ends unrun; beside a live stream it is declined" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const core = try coreFor(gpa, m, &tokens);
    defer core.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();
    try openPool(&bench, core);
    var talk: Talk = .init(63);
    defer talk.deinit(gpa);
    const opening = try bench.one(core, askFor(&talk.first, greedy, count, false));
    defer gpa.free(opening.tokens);
    try talk.after(gpa, opening.tokens);
    var other: [30]i32 = undefined;
    for (&other, 0..) |*token, i| token.* = @intCast(60 + i);
    const expected = try fixtures.serial.serialReply(gpa, m, &other, count, greedy);
    defer gpa.free(expected);
    const engine = &core.engine.?;

    // a pass of more rows than a round runs, which has run one, then a live request beside it
    var long: [80]i32 = undefined;
    for (&long, 0..) |*token, i| token.* = @intCast(1 + (i * 7) % 240);
    const pass = try core.admit(askFor(&long, greedy, 1, true), &.{});
    bench.reset(pass.slot);
    const before = engine.seats[pass.slot].?.filled;
    try testing.expect(core.round(bench.out()) == .ran);
    const ran = engine.seats[pass.slot].?.filled;
    try testing.expect(ran > before and ran < long.len);
    const live = try core.admit(askFor(&other, greedy, count, false), &.{});
    bench.reset(live.slot);
    try testing.expect(core.round(bench.out()) == .ran);
    try testing.expect(bench.ended[pass.slot] and bench.finish[pass.slot].? == .cut);
    try testing.expectEqual(ran, engine.seats[pass.slot].?.filled);
    try testing.expect(engine.seats[live.slot].?.filled > 0);
    _ = core.retire(pass.slot, .ended, 0);
    // beside the live stream a pass is declined
    try testing.expectError(error.Declined, core.admit(askFor(talk.warm.items, greedy, 1, true), &.{}));
    try bench.drain(core, &.{live.slot});
    try testing.expectEqualSlices(i32, expected, bench.lists[live.slot].items);
    _ = core.retire(live.slot, .ended, count);

    // alone, with a request waiting for the scheduler to admit it, a pass ends before its first chunk
    const alone = try core.admit(askFor(talk.warm.items, greedy, 1, true), &.{});
    bench.reset(alone.slot);
    const held = engine.seats[alone.slot].?.filled;
    bench.waits = true;
    try testing.expect(core.round(bench.out()) == .ran);
    try testing.expect(bench.ended[alone.slot] and bench.finish[alone.slot].? == .cut);
    try testing.expectEqual(held, engine.seats[alone.slot].?.filled);
    _ = core.retire(alone.slot, .ended, 0);
}
