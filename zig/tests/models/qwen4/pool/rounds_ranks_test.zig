//! `rounds_ranks.zig`'s tests: a rank's rounds of several streams, the share alone in this process (the other
//! rank's partial left at zero, the add still run, as `ranks_forward.zig`'s `Model.alone`), each rank in
//! turn. Every row of four packed streams is the row its stream has alone through `launch`, and what a slot
//! then holds is the lone state's: the random shares with an attention layer in each storage. The
//! checkpoint's first four layers (its PLE) on a share run on `mtp_ranks_test.zig`'s two-rank load
//! (`rounds_ranks_fixture.checkpointPacked`, spec 133 WS-08). Both ranks' partials meeting is `ranks-check ...
//! streams`'s, over NCCL: this file holds one rank's launches. What a rank refuses is at the bottom.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const random = @import("../../../../src/models/qwen4/checks/random.zig");
const check = @import("../../../../src/models/qwen4/checks/ranks_streams.zig");
const rounds_ranks = @import("../../../../src/models/qwen4/pool/rounds_ranks.zig");
const ranks = @import("../../../../src/models/qwen4/ranks_forward.zig");
const rounds_ranks_fixture = @import("rounds_ranks_fixture.zig");
const expectRows = rounds_ranks_fixture.expectRows;
const packedIsAlone = rounds_ranks_fixture.packedIsAlone;

const One = ranks.Held(1);
const Desk = rounds_ranks.Desk;
const storages = [_]?attention.Storage{ null, .bf16, .fp8, .fp8_nvfp4 };

test "a rank's rounds on the random shares, an attention layer's caches in each storage: every row of four packed streams is the row its stream has alone, and a slot holds the lone state" {
    const gpa = testing.allocator;
    for (storages) |kv| for (0..2) |rank| {
        errdefer std.debug.print("kv {?t}, rank {d}\n", .{ kv, rank });
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const built = try random.sharesOf(1, gpa, arena_state.allocator(), 42, rank, check.rows, .{ .caches = kv });
        var m: One.Model = .{ .ranks = built, .first = rank, .max_rows = check.rows, .alone = true };
        defer m.deinit();
        try testing.expectEqual(@as(usize, random.fx.layers) + @intFromBool(kv != null), m.ranks[0].layers.len);
        try packedIsAlone(gpa, m, 50);
    };
}

test "a rank's rounds write each stream's DFlash2 taps, the rows its stream's launch alone writes, one stream and two" {
    // spec 126 WS-05: the pool's rounds wrote no taps (`rounds.computeOver`); a lone launch's are the host mean
    // of its streams (`dflash2/ranks_loopback_test.zig`)
    const gpa = testing.allocator;
    const taps = [_]u32{ 0, 2 };
    for (0..2) |rank| {
        errdefer std.debug.print("rank {d}\n", .{rank});
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const built = try random.sharesOf(1, gpa, arena_state.allocator(), 42, rank, check.rows, .{ .caches = .bf16 });
        var m: One.Model = .{ .ranks = built, .first = rank, .max_rows = check.rows, .alone = true };
        defer m.deinit();
        const share = m.ranks[0];
        const width = taps.len * share.shape.hidden;
        var b: One.Buffers = try .init(gpa, m, check.rows, check.room);
        defer b.deinit();
        try b.setTaps(&taps, .f32_acc);
        var lone_b: One.Buffers = try .init(gpa, m, check.rows, check.room);
        defer lone_b.deinit();
        try lone_b.setTaps(&taps, .f32_acc);
        var lone: [2]One.State = undefined;
        for (&lone) |*state| state.* = try .init(gpa, m, check.rows, check.room);
        defer for (&lone) |*state| state.deinit();
        var desk: Desk = try .init(gpa, share, 2, check.rows, check.room, check.count);
        defer desk.deinit();
        try desk.place(0, 0, &.{0});
        try desk.place(1, 0, &.{1});
        const want = try gpa.alloc(u16, check.rows * width);
        defer gpa.free(want);
        const got = try gpa.alloc(u16, check.rows * width);
        defer gpa.free(got);
        var tokens: [8]i32 = undefined;
        var prng: std.Random.DefaultPrng = .init(126);
        for (&tokens) |*id| id.* = prng.random().intRangeLessThan(i32, 1, 300);
        const rounds_of = [_][]const rounds_ranks.Part{
            &.{ .{ .slot = 0, .rows = 5 }, .{ .slot = 1, .rows = 3 } },
            &.{.{ .slot = 0, .rows = 1 }},
        };
        const tokens_of = [_][]const i32{ &tokens, tokens[2..3] };
        for (rounds_of, tokens_of) |parts, round_tokens| {
            try b.ranks[0].taps.?.zero();
            try rounds_ranks.shared(m, &desk, &b, parts, round_tokens);
            var keeps: [2]usize = undefined;
            for (parts, 0..) |part, i| keeps[i] = part.rows;
            try rounds_ranks.commit(m, &desk, &b, keeps[0..parts.len]);
            var at: usize = 0;
            for (parts) |part| {
                const own = round_tokens[at..][0..part.rows];
                try One.forward(m, &lone[part.slot], &lone_b, own);
                try One.commit(m, &lone[part.slot], lone_b, own, own.len);
                const n = part.rows * width;
                try cuda.synchronize();
                try lone_b.ranks[0].taps.?.downloadAt(0, std.mem.sliceAsBytes(want[0..n]));
                try b.ranks[0].taps.?.downloadAt(at * width * 2, std.mem.sliceAsBytes(got[0..n]));
                try testing.expect(std.mem.indexOfNone(u16, want[0..n], &.{ 0, 0x8000 }) != null);
                try testing.expectEqualSlices(u16, want[0..n], got[0..n]);
                try expectRows(b.ranks[0].h, at, lone_b.ranks[0].h, 0, part.rows);
                at += part.rows;
            }
        }
    }
}

test "a rank refuses a round its tables do not cover, a table past its pool, a commit of no round and a short keep of a wide segment" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const built = try random.sharesOf(1, gpa, arena_state.allocator(), 42, 1, check.rows, .{ .caches = .fp8 });
    var m: One.Model = .{ .ranks = built, .first = 1, .max_rows = check.rows, .alone = true };
    defer m.deinit();
    var b: One.Buffers = try .init(gpa, m, check.rows, check.room);
    defer b.deinit();
    var desk: Desk = try .init(gpa, m.ranks[0], 2, check.rows, check.room, 4);
    defer desk.deinit();
    const tokens = @as([40]i32, @splat(7));

    // no table told yet; an entry past the pool; a table that does not go on from what stands; a slot not held
    try testing.expectError(error.OutOfStep, rounds_ranks.shared(m, &desk, &b, &.{.{ .slot = 0, .rows = 2 }}, tokens[0..2]));
    try testing.expectError(error.OutOfStep, desk.place(0, 0, &.{4}));
    try testing.expectError(error.OutOfStep, desk.place(0, 1, &.{1}));
    try testing.expectError(error.OutOfStep, desk.place(2, 0, &.{1}));
    try testing.expectEqual(0, desk.placed[0]);
    try desk.place(0, 0, &.{3});
    try desk.place(1, 0, &.{1});
    // a slot twice, rows that are not the tokens', a commit before any round
    try testing.expectError(error.OutOfStep, rounds_ranks.shared(m, &desk, &b, &.{ .{ .slot = 0, .rows = 1 }, .{ .slot = 0, .rows = 1 } }, tokens[0..2]));
    try testing.expectError(error.OutOfStep, rounds_ranks.shared(m, &desk, &b, &.{.{ .slot = 0, .rows = 3 }}, tokens[0..2]));
    try testing.expectError(error.OutOfStep, rounds_ranks.commit(m, &desk, &b, &.{1}));

    try rounds_ranks.shared(m, &desk, &b, &.{ .{ .slot = 1, .rows = 2 }, .{ .slot = 0, .rows = 38 } }, &tokens);
    // keeps for other parts than ran, none, more than ran, and fewer of a segment the wide kernels took
    try testing.expectError(error.OutOfStep, rounds_ranks.commit(m, &desk, &b, &.{2}));
    try testing.expectError(error.OutOfStep, rounds_ranks.commit(m, &desk, &b, &.{ 0, 38 }));
    try testing.expectError(error.OutOfStep, rounds_ranks.commit(m, &desk, &b, &.{ 3, 38 }));
    try testing.expectError(error.OutOfStep, rounds_ranks.commit(m, &desk, &b, &.{ 2, 5 }));
    try testing.expectEqual(0, desk.slots[0].pos);
    try rounds_ranks.commit(m, &desk, &b, &.{ 1, 38 });
    try testing.expectEqual(38, desk.slots[0].pos);
    try testing.expectEqual(1, desk.slots[1].pos);
    // the round is closed
    try testing.expectError(error.OutOfStep, rounds_ranks.commit(m, &desk, &b, &.{ 1, 38 }));
}
