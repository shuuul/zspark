//! `guard.zig`'s tests, through `masks.zig` as a decoder asks it: TensorFold 1.0.5's own cases
//! (`zig/src/core/lanes/stream.zig`'s and `engine_test.zig`'s, with their tokens: the close is 90, 91, 92,
//! `</think>` 91, the end 96, the cycle 11, 12, 13 after 70 tokens), then what zspark adds: a window against
//! its serial steps, a reply taken up again, a grammar inside.

const std = @import("std");
const lanes = @import("lanes");
const grammar = @import("../../src/text/grammar.zig");
const guard_file = @import("../../src/decode/guard.zig");
const masks_file = @import("../../src/decode/masks.zig");
const Guard = guard_file.Guard;
const Ask = guard_file.Ask;
const Masks = masks_file.Masks;
const Error = grammar.Error;

const testing = std.testing;

const fx = struct {
    const vocab = 2048;
    const words = vocab / 32;
    const close = [_]u32{ 90, 91, 92 };
    const end: u32 = 91;
    const eos: u32 = 96;
    /// What the model writes after `</think>` where nothing is forced there.
    const own_trail: u32 = 300;
};

/// The tests' model, TensorFold's fake target in the guard's cases: the token after `reply`, a function of
/// the reply alone. It thinks in distinct tokens, then in a cycle, and answers once `</think>` stands.
const Model = struct {
    cycle_after: usize = 70,
    pattern: []const u32 = &.{ 11, 12, 13 },
    /// The answer is a cycle too, without an end.
    answer_cycles: bool = false,
    /// It closes its reasoning itself after this many tokens.
    closes_at: ?usize = null,

    fn next(self: Model, reply: []const u32) u32 {
        if (std.mem.lastIndexOfScalar(u32, reply, fx.end)) |at| {
            const after = reply.len - at - 1;
            if (after == 0) return fx.own_trail;
            const answer = [_]u32{ 40, 41, 42 };
            if (self.answer_cycles) return answer[(after - 1) % answer.len];
            return if (after - 1 < answer.len) answer[after - 1] else fx.eos;
        }
        if (self.closes_at) |at| if (reply.len == at) return fx.end;
        if (reply.len < self.cycle_after) return @intCast(200 + reply.len);
        return self.pattern[(reply.len - self.cycle_after) % self.pattern.len];
    }
};

const How = enum {
    /// A window of one row a token.
    serial,
    /// Windows of the model's own next tokens as drafts, as a head that is always right would draft them:
    /// a draft where a token is forced is wrong, and every fifth window has a wrong draft of its own.
    drafted,
    /// The drafts a drafter makes with the constraint's guide: a forced token where one is known.
    guided,
};

/// `tokens` taken one by one, each a window of one row whose choice it was.
fn give(masks: *Masks, tokens: []const i32) !void {
    for (tokens) |token| {
        _ = try masks.window(&.{});
        try masks.commit(1, token);
    }
}

/// One row's logits: `preferred` above the rest. bf16 1.0 and 2.0.
fn chooseUnder(masks: *Masks, row: usize, logits: []u16, preferred: u32) !u32 {
    @memset(logits, 0x3f80);
    logits[preferred] = 0x4000;
    try masks.apply(row, logits);
    if (logits[preferred] != masks_file.minus_infinity) return preferred;
    for (logits, 0..) |logit, token| if (logit != masks_file.minus_infinity) return @intCast(token);
    return error.ConstraintFailed;
}

/// The reply of `model` under `masks`, `most` tokens at most, decoded `how`. `sure` counts the drafts the
/// constraint made certain before their window ran.
fn decode(gpa: std.mem.Allocator, masks: *Masks, model: Model, most: usize, how: How, sure: ?*usize) ![]u32 {
    var reply: std.ArrayList(u32) = .empty;
    errdefer reply.deinit(gpa);
    const logits = try gpa.alloc(u16, fx.vocab);
    defer gpa.free(logits);
    var windows: usize = 0;
    while (reply.items.len < most) : (windows += 1) {
        var drafts: [7]i32 = undefined;
        var history: [7]u32 = undefined;
        var count: usize = 0;
        const room = @min(drafts.len, most - reply.items.len - 1);
        if (how != .serial) while (count < room) {
            var path: std.ArrayList(u32) = .empty;
            defer path.deinit(gpa);
            try path.appendSlice(gpa, reply.items);
            try path.appendSlice(gpa, history[0..count]);
            var token = model.next(path.items);
            if (how == .guided) switch (try masks.next(drafts[0..count], true)) {
                .refused, .closed => break,
                .free, .among => {},
                .token => |forced| token = @intCast(forced),
            };
            if (how == .drafted and windows % 5 == 4 and count == 2) token = 1999;
            history[count] = token;
            drafts[count] = @intCast(token);
            count += 1;
            if (token == fx.eos) break;
        };
        if (sure) |total| total.* += masks.certain(drafts[0..count]);
        const ran = try masks.window(drafts[0..count]);
        var chosen: [8]u32 = undefined;
        var keep: usize = 0;
        while (keep <= ran) {
            var path: std.ArrayList(u32) = .empty;
            defer path.deinit(gpa);
            try path.appendSlice(gpa, reply.items);
            try path.appendSlice(gpa, history[0..keep]);
            chosen[keep] = try chooseUnder(masks, keep, logits, model.next(path.items));
            keep += 1;
            if (chosen[keep - 1] == fx.eos or keep > ran or chosen[keep - 1] != history[keep - 1]) break;
        }
        try masks.commit(keep, @intCast(chosen[keep - 1]));
        try reply.appendSlice(gpa, chosen[0..keep]);
        if (chosen[keep - 1] == fx.eos) break;
    }
    return reply.toOwnedSlice(gpa);
}

const Run = struct {
    reply: []u32,
    period: ?u32,
    sure: usize,
};

/// `model`'s reply under `ask` and no grammar.
fn run(model: Model, ask: Ask, most: usize, how: How) !Run {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const guard = try Guard.init(arena.allocator(), ask, null, fx.words, most);
    var mask: [fx.words]u32 = undefined;
    var masks: Masks = try .init(guard.constraint(), &mask, @intCast(fx.end));
    var sure: usize = 0;
    const reply = try decode(gpa, &masks, model, most, how, &sure);
    return .{ .reply = reply, .period = guard.period(), .sure = sure };
}

/// The reply every way of decoding gives, checked equal: a window's rows are its serial steps.
fn runAll(model: Model, ask: Ask, most: usize) !Run {
    const gpa = testing.allocator;
    const serial = try run(model, ask, most, .serial);
    errdefer gpa.free(serial.reply);
    for ([_]How{ .drafted, .guided }) |how| {
        const other = try run(model, ask, most, how);
        defer gpa.free(other.reply);
        try testing.expectEqualSlices(u32, serial.reply, other.reply);
        try testing.expectEqual(serial.period, other.period);
    }
    return serial;
}

fn guardOf(budget: u32, loop: bool) Ask {
    return Ask.of(budget, loop, &fx.close, fx.end).?;
}

test "what is guarded: a budget or the loop guard, with the tokens that close the block" {
    try testing.expectEqual(null, Ask.of(0, false, &fx.close, fx.end));
    try testing.expectEqual(null, Ask.of(9, true, &.{}, fx.end));
    // the loop is read against `</think>`: without it only a budget is held
    try testing.expectEqual(null, Ask.of(0, true, &fx.close, null));
    try testing.expect(!Ask.of(9, true, &fx.close, null).?.loop);
    try testing.expect(Ask.of(0, true, &fx.close, fx.end).?.loop);
}

test "TensorFold: the loop guard forces nothing where no cycle fires" {
    // stream.zig, "loop guard stops at the cap even when no cycle fires": twenty distinct tokens, ten kept
    const plain: Model = .{ .cycle_after = 1000 };
    const guarded = try runAll(plain, guardOf(0, true), 10);
    defer testing.allocator.free(guarded.reply);
    try testing.expectEqual(10, guarded.reply.len);
    for (guarded.reply, 0..) |token, index| try testing.expectEqual(plain.next(guarded.reply[0..index]), token);
    try testing.expectEqual(null, guarded.period);
    try testing.expectEqual(0, guarded.sure);
}

test "TensorFold: the loop guard fires, closes thinking, and latches once" {
    // stream.zig's case of that name, token for token: 64 distinct, then 257 of one
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const guard = try Guard.init(arena.allocator(), guardOf(0, true), null, fx.words, 400);
    var mask: [fx.words]u32 = undefined;
    var masks: Masks = try .init(guard.constraint(), &mask, @intCast(fx.end));
    var tokens: std.ArrayList(i32) = .empty;
    defer tokens.deinit(gpa);
    for (0..lanes.stream.loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..257) |_| try tokens.append(gpa, 7);
    // one token short nothing is forced and nothing has fired
    try masks.advance(tokens.items[0..320]);
    try testing.expectEqual(null, guard.period());
    try testing.expect(try masks.next(&.{}, true) == .free);
    try give(&masks, tokens.items[320..]);
    try testing.expectEqual(1, guard.period().?);
    // the close, a token a place, each the only one that may come; another is refused
    for (fx.close) |token| {
        try testing.expect(try masks.next(&.{8}, true) == .refused);
        try testing.expectEqual(@as(i32, @intCast(token)), (try masks.next(&.{}, true)).token);
        try give(&masks, &.{@intCast(token)});
    }
    try testing.expect(try masks.next(&.{}, true) == .free);
    try give(&masks, &.{8});
    try testing.expectEqual(1, guard.period().?);
    // and the answer may cycle as it likes
    for (0..400 - 325) |_| try give(&masks, &.{8});
    try testing.expect(try masks.next(&.{}, true) == .free);
    try testing.expectEqual(1, guard.period().?);
}

test "TensorFold: the thinking budget's forced close is the same drafted, plain and guided" {
    // engine_test.zig: budget 9, `emitted[8] == 90`
    const budgeted = try runAll(.{ .cycle_after = 1000 }, guardOf(9, false), 400);
    defer testing.allocator.free(budgeted.reply);
    for (budgeted.reply[0..8], 0..) |token, index| try testing.expectEqual(200 + index, token);
    try testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, budgeted.reply[8..]);
    try testing.expectEqual(null, budgeted.period);
    // the close is drafted without a model and known before its window runs: a window begins at it
    const guided = try run(.{ .cycle_after = 1000 }, guardOf(9, false), 400, .guided);
    defer testing.allocator.free(guided.reply);
    try testing.expectEqual(fx.close.len, guided.sure);
    // a budget of one: the reply's first token is the close's
    const at_once = try runAll(.{ .cycle_after = 1000 }, guardOf(1, false), 400);
    defer testing.allocator.free(at_once.reply);
    try testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, at_once.reply);
}

test "TensorFold: the loop guard matches drafted and plain, reports its period, and answers after close" {
    // engine_test.zig's case of that name: the cycle of three completes at token 329
    const guarded = try runAll(.{}, guardOf(0, true), 400);
    defer testing.allocator.free(guarded.reply);
    try testing.expectEqual(3, guarded.period.?);
    try testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, guarded.reply[329..]);
    try testing.expectEqualSlices(u32, &.{ 12, 13, 11 }, guarded.reply[326..329]);
    // the guard off, a budget that is never reached: the block stays open and the reply runs to its length
    const off = try runAll(.{}, guardOf(100_000, false), 400);
    defer testing.allocator.free(off.reply);
    try testing.expectEqual(400, off.reply.len);
    try testing.expectEqual(null, off.period);
    try testing.expectEqual(null, std.mem.indexOfScalar(u32, off.reply, 90));
}

test "TensorFold: a cycle in the answer does not refire or reclose" {
    const guarded = try runAll(.{ .answer_cycles = true }, guardOf(0, true), 800);
    defer testing.allocator.free(guarded.reply);
    try testing.expectEqual(800, guarded.reply.len);
    try testing.expectEqual(3, guarded.period.?);
    try testing.expectEqual(1, std.mem.count(u32, guarded.reply, &.{90}));
}

test "the budget and the loop at one index, and one after: the close replaces the token, a completed cycle wins" {
    const gpa = testing.allocator;
    // the cycle completes with token 329 (index 328): a budget of 329 replaces that token
    const budget_first = try runAll(.{}, guardOf(329, true), 400);
    defer gpa.free(budget_first.reply);
    try testing.expectEqual(null, budget_first.period);
    try testing.expectEqualSlices(u32, &.{ 13, 90, 91, 92, 40 }, budget_first.reply[327..332]);
    const loop_first = try runAll(.{}, guardOf(330, true), 400);
    defer gpa.free(loop_first.reply);
    try testing.expectEqual(3, loop_first.period.?);
    try testing.expectEqualSlices(u32, &.{ 11, 90, 91, 92, 40 }, loop_first.reply[328..333]);
    try testing.expectEqual(1, std.mem.count(u32, loop_first.reply, &.{90}));
}

test "a </think> of the model's own ends the budget and the guard" {
    const gpa = testing.allocator;
    // closed at token 5: neither a budget of 9 nor a cycle in what follows forces anything
    const model: Model = .{ .closes_at = 5, .answer_cycles = true };
    const closed = try runAll(model, guardOf(9, true), 700);
    defer gpa.free(closed.reply);
    try testing.expectEqual(700, closed.reply.len);
    try testing.expectEqual(null, closed.period);
    try testing.expectEqual(null, std.mem.indexOfScalar(u32, closed.reply, 90));
    try testing.expectEqualSlices(u32, &.{ 204, 91, fx.own_trail, 40, 41 }, closed.reply[4..9]);
    // one token later the budget has the place: the model's `</think>` is not written, the close is
    const late = try runAll(.{ .closes_at = 8, .cycle_after = 1000 }, guardOf(9, false), 700);
    defer gpa.free(late.reply);
    try testing.expectEqualSlices(u32, &.{ 207, 90, 91, 92, 40, 41, 42, 96 }, late.reply[7..]);
}

test "a period is the committed reply's: a draft that completes the cycle and is not kept reports none" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const guard = try Guard.init(arena.allocator(), guardOf(0, true), null, fx.words, 400);
    var mask: [fx.words]u32 = undefined;
    var masks: Masks = try .init(guard.constraint(), &mask, @intCast(fx.end));
    var tokens: [320]i32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = if (i < 64) @intCast(1000 + i) else 7;
    try masks.advance(&tokens);
    // the draft 7 completes the cycle: the row after it must be the close's first, so a second 7 is cut
    try testing.expectEqual(1, try masks.window(&.{ 7, 7 }));
    try testing.expectEqual(null, guard.period());
    // the model chose another token at row 0: the cycle never completed
    try masks.commit(1, 9);
    try testing.expectEqual(null, guard.period());
    try testing.expect(try masks.next(&.{}, true) == .free);
}

test "a reply taken up again stands where it stood: the same tokens force the same close" {
    const gpa = testing.allocator;
    const whole = try run(.{}, guardOf(0, true), 400, .serial);
    defer gpa.free(whole.reply);
    // a stream moved to another decoder: a new `Masks` over the same constraint, advanced over what was sent
    for ([_]usize{ 100, 329, 330, 331, 332, 335 }) |sent| {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const guard = try Guard.init(arena.allocator(), guardOf(0, true), null, fx.words, 400);
        var mask: [fx.words]u32 = undefined;
        var first: Masks = try .init(guard.constraint(), &mask, @intCast(fx.end));
        try first.advance(@ptrCast(whole.reply[0 .. sent / 2]));
        var masks: Masks = try .init(guard.constraint(), &mask, @intCast(fx.end));
        try masks.advance(@ptrCast(whole.reply[0..sent]));
        try testing.expectEqual(if (sent >= 329) @as(?u32, 3) else null, guard.period());
        const model: Model = .{};
        const logits = try gpa.alloc(u16, fx.vocab);
        defer gpa.free(logits);
        for (sent..whole.reply.len) |index| {
            _ = try masks.window(&.{});
            const token = try chooseUnder(&masks, 0, logits, model.next(whole.reply[0..index]));
            try testing.expectEqual(whole.reply[index], token);
            try masks.commit(1, @intCast(token));
        }
    }
}

/// A grammar a test makes up: place i allows `allowed[i]`, and the reply ends after the last place.
const Scripted = struct {
    allowed: []const []const u32,

    const table: grammar.Constraint.Table = .{ .start = start, .fill = fill, .accept = accept, .ended = ended, .deinit = deinit };

    fn constraint(self: *Scripted) grammar.Constraint {
        return .{ .context = self, .words = fx.words, .table = &table };
    }

    fn start(_: *anyopaque) Error!grammar.Position {
        return @fromBackingInt(@intCast(0));
    }

    fn fill(context: *anyopaque, at: grammar.Position, mask: []u32) Error!void {
        const self: *Scripted = @ptrCast(@alignCast(context));
        @memset(mask, 0);
        for (self.allowed[@backingInt(at)]) |token| mask[token / 32] |= @as(u32, 1) << @intCast(token % 32);
    }

    fn accept(context: *anyopaque, at: grammar.Position, token: u32) Error!?grammar.Position {
        const self: *Scripted = @ptrCast(@alignCast(context));
        return if (std.mem.indexOfScalar(u32, self.allowed[@backingInt(at)], token) != null) @fromBackingInt(@intCast(@backingInt(at) + 1)) else null;
    }

    fn ended(context: *anyopaque, at: grammar.Position) bool {
        const self: *Scripted = @ptrCast(@alignCast(context));
        return @backingInt(at) == self.allowed.len;
    }

    fn deinit(_: *anyopaque) void {}
};

test "a grammar inside begins after the forced </think>: the close's trail where the grammar takes it, the grammar's own lead where not" {
    const gpa = testing.allocator;
    // the grammar's lead is the close's trail (92), as a template's blank line after `</think>` is
    var same: Scripted = .{ .allowed = &.{ &.{92}, &.{ 40, 50 }, &.{fx.eos} } };
    // another lead: the trail is given up
    var other: Scripted = .{ .allowed = &.{ &.{77}, &.{ 40, 50 }, &.{fx.eos} } };
    const expected = [_][]const u32{ &.{ 200, 201, 90, 91, 92, 40, 96 }, &.{ 200, 201, 90, 91, 77, 40, 96 } };
    for ([_]*Scripted{ &same, &other }, expected) |scripted, tokens| {
        for ([_]How{ .serial, .drafted, .guided }) |how| {
            var arena: std.heap.ArenaAllocator = .init(gpa);
            defer arena.deinit();
            const guard = try Guard.init(arena.allocator(), guardOf(3, true), scripted.constraint(), fx.words, 64);
            var mask: [fx.words]u32 = undefined;
            var masks: Masks = try .init(guard.constraint(), &mask, @intCast(fx.end));
            try testing.expect(!masks.ended());
            const reply = try decode(gpa, &masks, .{ .cycle_after = 1000 }, 64, how, null);
            defer gpa.free(reply);
            try testing.expectEqualSlices(u32, tokens, reply);
            try testing.expect(masks.ended());
        }
    }
    // without a guard the same grammar after the model's own `</think>`: the reasoning is free, as before
    var mask: [fx.words]u32 = undefined;
    var masks: Masks = try .init(other.constraint(), &mask, @intCast(fx.end));
    const reply = try decode(gpa, &masks, .{ .cycle_after = 1000, .closes_at = 4 }, 64, .serial, null);
    defer gpa.free(reply);
    try testing.expectEqualSlices(u32, &.{ 200, 201, 202, 203, 91, 77, 40, 96 }, reply);
}

test "a reply longer than the guard was made for fails and writes nothing past its places" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const guard = try Guard.init(arena.allocator(), guardOf(0, true), null, fx.words, 4);
    var mask: [fx.words]u32 = undefined;
    var masks: Masks = try .init(guard.constraint(), &mask, null);
    var tokens: [guard_file.Guard.rows_ahead + 4]i32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i);
    try masks.advance(&tokens);
    try testing.expectError(error.ConstraintFailed, masks.advance(&.{5}));
}
