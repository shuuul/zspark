//! `hall.zig` and `slots.zig` on FP4-Local, the checkpoint `ZSPARK_GLM_FP4_LOCAL` names (spec 068 WS-04): its first four layers
//! with both ranks in the process, as a session of four slots holds them (`session.Options.parallel`).
//! Sequences that share a window are held bit for bit to each alone (`slots_fixture.zig`'s check, on FP4-Local's
//! kernels and sizes); replies decoded several at once, drafted by the MTP layer and by the DFlash2 drafter
//! `ZSPARK_GLM_DRAFTER` names, are held token for token to the serial reply of the model alone and to the
//! same reply decoded with no other beside it, whose rounds and drafts they repeat. 13 GiB on the device.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const shared = @import("../testing.zig");
const cuda = @import("../../../../src/cuda/cuda.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const flash_config = @import("../../../../src/decode/dflash2/config.zig");
const flash_layout = @import("../../../../src/decode/dflash2/layout.zig");
const rig = @import("../../../../src/models/glm5/dflash2/rig.zig");
const hall_file = @import("../../../../src/models/glm5/pool/hall.zig");
const hall_fixture = @import("hall_fixture.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const session_file = @import("../../../../src/models/glm5/session/session.zig");
const slots_fixture = @import("slots_fixture.zig");

const Session = session_file.SessionOf(layout.world);
const capacity = 96;
const slots = 4;

/// A chat's opening as the template writes it, and ids of plain text after it.
const opening = [_]i32{ 154822, 154824, 154827, 852, 279, 1156, 4236, 10245, 5109, 11, 18630, 553, 75994, 13, 154828, 154841, 154842 };

fn ids(arena: Allocator, seed: u64, count: usize) ![]i32 {
    var prng: std.Random.DefaultPrng = .init(seed);
    const out = try arena.alloc(i32, count);
    for (out) |*one| one.* = prng.random().intRangeLessThan(i32, 200, 150_000);
    return out;
}

/// The session's sequences sharing windows against each alone: every scored row's logits and what each
/// leaves, bit for bit.
fn bits(arena: Allocator, session: *Session, vocab: usize) !void {
    var loop: struct { m: Session.H.Model, state: Session.H.State, b: Session.H.Buffers } = .{ .m = session.m, .state = try .init(session.m, capacity), .b = session.b };
    defer loop.state.deinit();
    const text = try ids(arena, 0x68a, 160);
    // windows cut short, a prompt's chunk scored at its end, a sequence that starts late, a slot taken again
    const all = [_]slots_fixture.Sequence{
        .{ .tokens = text[0..40], .plan = &.{ .{ 7, 7 }, .{ 1, 1 }, .{ 5, 2 }, .{ 8, 8 }, .{ 3, 1 }, .{ 1, 1 } }, .slot = 2 },
        .{ .tokens = text[40..90], .plan = &.{ .{ 24, 24 }, .{ 2, 2 }, .{ 6, 3 }, .{ 1, 1 }, .{ 4, 4 } }, .slot = 0, .score = .last },
        .{ .tokens = text[90..120], .plan = &.{ .{ 1, 1 }, .{ 8, 5 }, .{ 2, 2 }, .{ 6, 6 } }, .slot = 3, .start = 2 },
        .{ .tokens = text[120..160], .plan = &.{ .{ 12, 12 }, .{ 3, 3 }, .{ 7, 4 } }, .slot = 2, .start = 7, .score = .none },
    };
    var want: [all.len]slots_fixture.Run = undefined;
    for (&want, all) |*run, one| run.* = try slots_fixture.alone(arena, &loop, vocab, one);
    const rounds = try slots_fixture.among(arena, &loop, &session.hall.?.slots, vocab, &all, &want);
    try testing.expect(rounds >= 6);
}

fn several(drafter: ?[]const u8) !void {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    // the drafter's directory as a command opens it: read from where it is, nothing of it written
    var drafting: rig.Opened = .{};
    defer drafting.close(io);
    if (drafter) |where| {
        drafting.path = where;
        drafting.dir = try std.Io.Dir.cwd().openDir(io, where, .{});
        drafting.config = try flash_config.Config.read(arena, io, drafting.dir.?, &drafting.diagnostic);
        drafting.checkpoint = try .open(gpa, io, .cwd(), where);
        drafting.layout = try flash_layout.Layout.init(arena, &drafting.config, &drafting.checkpoint.?, &drafting.diagnostic);
        try testing.expect(try rig.fit(arena, &drafting, &c, 4));
    }
    const session = try Session.create(gpa, arena, io, &checkpoint, &c, .{}, .{ .layers = 4, .rows = 64, .capacity = capacity, .parallel = slots, .drafter = if (drafter != null) &drafting else null }, "test");
    defer session.destroy();
    try session.meet();
    const hall = &session.hall.?;
    try testing.expectEqual(@as(usize, slots), hall.streams.len);
    try testing.expect((hall.flash != null) == (drafter != null) and (hall.head != null) == (drafter == null));
    if (drafter == null) try bits(arena, session, c.vocab);

    const text = try ids(arena, 0x68b, 120);
    const none: []const u32 = &.{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    // more requests than slots, a prompt in several chunks beside replies, one that does not draft
    const cases = [_]hall_fixture.Case{
        .{ .request = .{ .prompt = &opening, .settings = greedy, .eos = none, .max_tokens = 20, .fresh = true } },
        .{ .request = .{ .prompt = text[0..9], .settings = .{ .seed = 7, .temperature = 0.8 }, .eos = none, .max_tokens = 24, .fresh = true } },
        .{ .request = .{ .prompt = text[10..70], .settings = .{ .seed = 8, .temperature = 0.8 }, .eos = none, .max_tokens = 16, .fresh = true }, .start = 1 },
        .{ .request = .{ .prompt = text[70..71], .settings = greedy, .eos = none, .max_tokens = 18, .fresh = true }, .start = 2 },
        .{ .request = .{ .prompt = text[80..102], .settings = .{ .seed = 9, .temperature = 0.8 }, .eos = none, .max_tokens = 14, .drafts = false, .fresh = true }, .start = 2 },
        .{ .request = .{ .prompt = text[100..118], .settings = .{ .seed = 10, .temperature = 1.0 }, .eos = none, .max_tokens = 22, .fresh = true }, .start = 5 },
    };
    var serial: [cases.len][]i32 = undefined;
    var alone: [cases.len]hall_fixture.Served = undefined;
    for (&serial, &alone, cases) |*want, *one, case| {
        want.* = try shared.serialReply(arena, session.m, c.vocab, case.request.prompt, case.request.max_tokens, case.request.settings);
        var lone = case;
        lone.start = 0;
        one.* = try hall_fixture.serve(arena, hall, &.{lone});
        try testing.expectEqual(@as(usize, 1), one.beside);
        try testing.expectEqualSlices(i32, want.*, one.tokens[0].items);
    }
    const before = hall.rounds;
    const among = try hall_fixture.serve(arena, hall, &cases);
    try testing.expectEqual(@as(usize, slots), among.beside);
    var drafted: usize = 0;
    var accepted: usize = 0;
    for (serial, alone, among.tokens, among.ended, cases) |want, one, got, ended, case| {
        try testing.expectEqualSlices(i32, want, got.items);
        // beside others a reply's rounds, its drafts and the drafts kept are those it has alone
        try testing.expectEqual(one.ended[0].?.stats, ended.?.stats);
        try testing.expectEqual(hall_file.Finish.length, ended.?.finish);
        if (!case.request.drafts) try testing.expectEqual(@as(usize, 0), ended.?.stats.drafted);
        drafted += ended.?.stats.drafted;
        accepted += ended.?.stats.accepted;
    }
    try testing.expect(drafted > 0);
    // streams that share a round take fewer rounds than their windows
    try testing.expect(hall.rounds - before < windows(among));
    std.debug.print("several at once, {s}: {d} rounds for {d} replies' {d} windows, {d} of {d} drafts kept\n", .{ if (drafter == null) "MTP" else "DFlash2", hall.rounds - before, cases.len, windows(among), accepted, drafted });
}

fn windows(served: hall_fixture.Served) usize {
    var sum: usize = 0;
    for (served.ended) |ended| sum += ended.?.stats.rounds;
    return sum;
}

/// What `streams` replies of `count` tokens each cost a hall after their prompts' round: its rounds, the
/// nanoseconds and the tokens given.
fn timed(hall: *Session.Hall, prompts: []const []const i32, count: usize, drafts: bool) ![3]u64 {
    const Sink = struct {
        given: u64 = 0,
        left: [16]usize = undefined,
        ended_count: usize = 0,

        pub fn tokens(self: *@This(), _: usize, given: []const i32) void {
            self.given += given.len;
        }

        pub fn ended(self: *@This(), slot: usize, _: hall_file.Ended) void {
            self.left[self.ended_count] = slot;
            self.ended_count += 1;
        }
    };
    var sink: Sink = .{};
    for (prompts) |prompt| _ = (try hall.admit(.{ .prompt = prompt, .settings = .{ .seed = 0, .temperature = 0 }, .eos = &.{}, .max_tokens = count, .drafts = drafts, .fresh = true })).?;
    // the prompts' round, and each reply's first token
    _ = try hall.round(&sink);
    try cuda.synchronize();
    const rounds = hall.rounds;
    const given = sink.given;
    const start = cuda.monotonic();
    while (hall.live() > 0) {
        _ = try hall.round(&sink);
        for (sink.left[0..sink.ended_count]) |slot| hall.retire(slot);
        sink.ended_count = 0;
    }
    try cuda.synchronize();
    return .{ hall.rounds - rounds, cuda.monotonic() - start, sink.given - given };
}

test "FP4-Local's first layers, several at once: what a round costs by the streams in it (ZSPARK_GLM_HALL_TIMES; a timing, the machine alone)" {
    _ = testing.environ.getPosix("ZSPARK_GLM_HALL_TIMES") orelse return error.SkipZigTest;
    _ = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    try times(null);
    if (testing.environ.getPosix("ZSPARK_GLM_DRAFTER")) |drafter| try times(drafter);
}

/// A session's and a hall's rounds timed, drafted by the MTP layer and not at all, or by the drafter at `drafter`.
fn times(drafter: ?[]const u8) !void {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL").?;
    const name = if (drafter == null) "MTP drafts" else "DFlash2 drafts";
    const ways: []const bool = if (drafter == null) &.{ false, true } else &.{true};
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    var drafting: rig.Opened = .{};
    defer drafting.close(io);
    if (drafter) |where| {
        drafting.path = where;
        drafting.dir = try std.Io.Dir.cwd().openDir(io, where, .{});
        drafting.config = try flash_config.Config.read(arena, io, drafting.dir.?, &drafting.diagnostic);
        drafting.checkpoint = try .open(gpa, io, .cwd(), where);
        drafting.layout = try flash_layout.Layout.init(arena, &drafting.config, &drafting.checkpoint.?, &drafting.diagnostic);
        try testing.expect(try rig.fit(arena, &drafting, &c, 4));
    }
    const by: ?*rig.Opened = if (drafter != null) &drafting else null;
    const count = 129;
    const text = try ids(arena, 0x68c, 8 * 32);
    var prompts: [8][]const i32 = undefined;
    for (&prompts, 0..) |*one, at| one.* = text[at * 32 ..][0..32];
    std.debug.print("4 layers, both ranks in the process, windows of 512 rows, 2,051 positions a slot; replies of {d} tokens after 32 of prompt, greedy\n", .{count});
    {
        // one reply at a time: the session, its windows captured graphs
        const session = try Session.create(gpa, arena, io, &checkpoint, &c, .{}, .{ .layers = 4, .rows = 512, .capacity = 2051, .drafter = by }, "test");
        defer session.destroy();
        try session.meet();
        var out: [count]i32 = undefined;
        for (ways) |drafts| for (0..2) |turn| {
            const done = try session.reply(prompts[0], .{ .seed = 0, .temperature = 0 }, .{ .eos = &.{}, .drafts = drafts, .fresh = true }, &out);
            if (turn == 0) continue;
            const ms = @as(f64, @floatFromInt(done.rest_ns)) / 1e6;
            std.debug.print("  session, {s}: {d} rounds, {d:.3} ms a round, {d:.1} tokens a second\n", .{ if (drafts) name else "no drafts", done.stats.rounds, ms / @as(f64, @floatFromInt(done.stats.rounds)), @as(f64, @floatFromInt(done.made - 1)) * 1e3 / ms });
        };
    }
    const session = try Session.create(gpa, arena, io, &checkpoint, &c, .{}, .{ .layers = 4, .rows = 512, .capacity = 2051, .parallel = 8, .drafter = by }, "test");
    defer session.destroy();
    try session.meet();
    const hall = &session.hall.?;
    for (ways) |drafts| for ([_]usize{ 1, 2, 4, 8 }) |streams| {
        var best: [3]u64 = .{ 0, std.math.maxInt(u64), 0 };
        for (0..3) |_| {
            const got = try timed(hall, prompts[0..streams], count, drafts);
            if (got[1] < best[1]) best = got;
        }
        const ms = @as(f64, @floatFromInt(best[1])) / 1e6;
        std.debug.print("  hall of 8, {d} streams, {s}: {d} rounds, {d:.3} ms a round, {d:.1} tokens a second in all\n", .{ streams, if (drafts) name else "no drafts", best[0], ms / @as(f64, @floatFromInt(best[0])), @as(f64, @floatFromInt(best[2])) * 1e3 / ms });
    };
}

test "FP4-Local's first layers, several at once: sequences that share a window give each one's bits alone, and replies drafted by the MTP layer are the serial replies and their rounds alone" {
    try several(null);
}

test "FP4-Local's first layers, several at once by DFlash2: replies drafted from each slot's own context are the serial replies and their rounds alone" {
    const drafter = testing.environ.getPosix("ZSPARK_GLM_DRAFTER") orelse return error.SkipZigTest;
    try several(drafter);
}
