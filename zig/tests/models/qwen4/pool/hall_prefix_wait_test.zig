//! Tests of the prefix wait (spec 152 WS-06, `decode/prefix_wait.zig`): a prompt admitted beside another
//! whose fill shares a cut waits for it (`error.Wait`, `Core.wait_offer`), goes on from the state the fill
//! kept once its round reaches the cut, is freed by the bound and by the leader ending, and asks for no wait
//! when it is off or the prompts share nothing. `check.sh qwen4 hall_keep` runs it; no checkpoint.

const std = @import("std");
const testing = std.testing;
const fixtures = @import("../testing.zig");
const hall_fixture = @import("hall_fixture.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const prefix_wait = @import("../../../../src/decode/prefix_wait.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");

const greedy = hall_fixture.greedy;
const Core = hall_file.Core;
const Bench = hall_fixture.Bench;
const askFor = hall_fixture.askFor;

/// A `Bench` that drives `core` for `rounds` rounds, each a prompt chunk of the fills it holds.
fn run(core: *Core, bench: *Bench, rounds: usize) !void {
    for (0..rounds) |_| switch (core.round(bench.out())) {
        .ran => {},
        .short => return error.Short,
        .failed => return error.Failed,
    };
}

/// A core of `slots` over a small model, the wait `ns` nanoseconds.
fn waitingCore(gpa: std.mem.Allocator, m: anytype, tokens: *const tokenizer.Tokenizer, slots: usize, ns: i96) !*Core {
    const core = try hall_fixture.coreFor(gpa, m, tokens, &.{}, slots, 512);
    core.prefix_wait = .{ .ns = ns };
    return core;
}

test "a request whose prompt shares a cut another fills waits, then goes on from the kept state" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const core = try waitingCore(gpa, m, &tokens, 3, 5 * std.time.ns_per_s);
    defer core.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();

    // the first prompt opens the pool and fills beside the lone stream's migration: two requests at once
    const opener = try core.admit(askFor(&.{ 3, 3, 3, 3 }, greedy, 2), &.{});
    bench.reset(opener.slot);
    const ask = askFor(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 }, greedy, 4);
    const filling = try core.admit(ask, &.{});
    bench.reset(filling.slot);
    try testing.expect(core.engine != null);
    // one chunk (8 rows) is run: the fill stands inside its prompt, its end a cut it keeps
    try run(core, &bench, 1);

    // the same prompt waits on the fill, and holds the wait until its round reaches the end
    var waiter = ask;
    try testing.expectError(error.Wait, core.admit(waiter, &.{}));
    const waiting = core.wait_offer.?;
    // the cut is the turn mark the plan keeps, two below the end (`Options.tail`): `pick` asks the state
    // for `at + 1` tokens against the fill's own prompt, so `at` stops one short of the end
    try testing.expect(waiting.at == ask.prompt.len - 2);
    waiter.wait = waiting;
    try testing.expectError(error.Wait, core.admit(waiter, &.{}));
    try run(core, &bench, 4); // the fill's prompt is run whole
    const taken = try core.admit(waiter, &.{});
    bench.reset(taken.slot);
    // the kept state at the turn mark served the prompt: its two last rows are run again
    try testing.expectEqual(ask.prompt.len - 2, taken.cached);
    _ = core.retire(filling.slot, .ended, 4);
    _ = core.retire(taken.slot, .ended, 4);
    _ = core.retire(opener.slot, .ended, 2);
    try testing.expectEqual(@as(u64, 1), core.prefix_wait.waits);
}

test "the bound releases the waiter, which fills for itself" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const core = try waitingCore(gpa, m, &tokens, 3, 1); // a nanosecond: every second ask is over it
    defer core.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();

    _ = try core.admit(askFor(&.{ 9, 9 }, greedy, 2), &.{});
    const long = askFor(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, greedy, 2);
    const filling = try core.admit(long, &.{});
    bench.reset(filling.slot);
    try run(core, &bench, 1);

    try testing.expectError(error.Wait, core.admit(long, &.{}));
    var waiter = long;
    waiter.wait = core.wait_offer.?;
    // over the bound at once: the waiter is admitted and fills, the wait counted a timeout
    const taken = try core.admit(waiter, &.{});
    try testing.expect(taken.cached < long.prompt.len);
    _ = core.retire(filling.slot, .ended, 2);
    _ = core.retire(taken.slot, .ended, 2);
    try testing.expectEqual(@as(u64, 1), core.prefix_wait.timeouts);
}

test "off and unrelated: the wait is asked of nothing" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const core = try waitingCore(gpa, m, &tokens, 2, 0);
    defer core.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();

    _ = try core.admit(askFor(&.{ 7, 7 }, greedy, 2), &.{});
    const long = askFor(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, greedy, 2);
    const filling = try core.admit(long, &.{});
    bench.reset(filling.slot);
    try run(core, &bench, 1);
    // 0: no wait is asked; both seats are taken, so the request waits for a seat as it always did
    try testing.expectError(error.Wait, core.admit(long, &.{}));
    try testing.expectEqual(@as(u64, 0), core.prefix_wait.waits);
    _ = core.retire(filling.slot, .ended, 2);
}
