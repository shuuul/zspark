//! `rounds.zig`'s deferred DeltaNet states (spec 144 WS-09): windows of drafts kept in part leave their rows behind
//! in a slot's scratch, the next chain replays them, and a slot whose window cannot go on from them lands first,
//! as a session's state does (`forward.stores`, `forward.landsAt`). Each round's rows and each slot's state at its
//! position are the stream's alone, bit for bit, on the random model (`testing.zig`'s `tiny_model`).

const std = @import("std");
const testing = std.testing;
const fixtures = @import("../testing.zig");
const tinyModel = fixtures.tiny_model.tinyModel;
const tiny = fixtures.tiny_model.tiny;
const cuda = @import("../../../../src/cuda/cuda.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const gdn = @import("../../../../src/models/qwen4/blocks/gdn.zig");
const rounds = @import("../../../../src/models/qwen4/pool/rounds.zig");
const Segment = rounds.Segment;
const Slot = rounds.Slot;

fn rowWidth(buffer: cuda.Buffer) usize {
    return buffer.bytes() / @as(usize, @intCast(buffer.view.size[0]));
}

fn expectRows(got: cuda.Buffer, got_row: usize, want: cuda.Buffer, want_row: usize, count: usize) !void {
    const width = rowWidth(got);
    std.debug.assert(width == rowWidth(want));
    const gpa = testing.allocator;
    const left = try gpa.alloc(u8, width * count);
    defer gpa.free(left);
    const right = try gpa.alloc(u8, width * count);
    defer gpa.free(right);
    try cuda.synchronize();
    try got.downloadAt(got_row * width, left);
    try want.downloadAt(want_row * width, right);
    try testing.expectEqualSlices(u8, right, left);
}

fn expectBuffer(got: cuda.Buffer, want: cuda.Buffer) !void {
    try expectRows(got, 0, want, 0, @intCast(got.view.size[0]));
}

/// The slot's state at its position is the session's: the DeltaNet states replayed over the rows each is behind
/// by, the conv windows, the PLE's tails and histories.
fn expectSlot(slot: Slot, state: forward.State) !void {
    try testing.expectEqual(state.pos, slot.pos);
    for (state.recurrent, 0..) |want, linear| {
        var at = try forward.Snapshot.likeOf(want[0]);
        defer at.deinit();
        try forward.committed(state, linear, at);
        var got = try forward.Snapshot.likeOf(want[0]);
        defer got.deinit();
        try rounds.committed(slot, linear, got);
        try expectBuffer(got, at);
    }
    for (slot.conv, state.conv) |window, want| try expectBuffer(window, want);
    for (slot.ple, state.ple) |branch, want| {
        try expectBuffer(branch.tail, want.tail);
        try testing.expectEqualSlices(u32, want.history, branch.history);
    }
}

/// One round: the widths and keeps of slot 0's and slot 1's windows (0 rows: the slot sits out), and whether it
/// runs as the pool's engine does, every row staged as kept, the keeps replayed after (`replayKept`).
const Plan = struct { rows: [2]usize, keeps: [2]usize, after: bool = false };

test "deferred rounds: windows of drafts kept in part go on from the rows behind, land past the scratch and before a wide chunk, as each stream alone" {
    try deferredRounds(null);
}

test "deferred rounds through the slots' rows of four floats and butterfly sums give the base's bits: each round's rows and each slot's states" {
    // `gdn.setQuads`'s two sides through the same rounds: the base (a float and a warp_sum at a time) and the
    // slots' kernels in their QUADS form (spec 143 WS-10); each side's rounds also equal each stream alone
    const gpa = testing.allocator;
    var sides: [2]std.ArrayList(u8) = .{ .empty, .empty };
    defer for (&sides) |*side| side.deinit(gpa);
    defer gdn.setQuads(true);
    for (&sides, [_]bool{ false, true }) |*side, on| {
        gdn.setQuads(on);
        try deferredRounds(side);
    }
    try testing.expect(sides[0].items.len > 0);
    try testing.expectEqualSlices(u8, sides[0].items, sides[1].items);
}

/// What a round left, appended to `recorded`: its rows' logits, then each slot's committed DeltaNet states and
/// conv windows.
fn record(recorded: *std.ArrayList(u8), logits: cuda.Buffer, rows: usize, slots: []const Slot) !void {
    const gpa = testing.allocator;
    try cuda.synchronize();
    try logits.downloadAt(0, try recorded.addManyAsSlice(gpa, rows * rowWidth(logits)));
    for (slots) |slot| {
        for (slot.recurrent, 0..) |pair, linear| {
            var at = try forward.Snapshot.likeOf(pair[0]);
            defer at.deinit();
            try rounds.committed(slot, linear, at);
            try cuda.synchronize();
            try at.download(try recorded.addManyAsSlice(gpa, at.bytes()));
        }
        for (slot.conv) |window| try window.download(try recorded.addManyAsSlice(gpa, window.bytes()));
    }
}

/// The rounds below, each checked against the streams alone; with `recorded`, each round's rows and states kept.
fn deferredRounds(recorded: ?*std.ArrayList(u8)) !void {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    // slot 0: 5 of 7 rows kept, behind 5 to 30, then a window that passes the 32 rows and lands; a single row on
    // rows behind (replayed in the chain, which stores); a window of 31 that lands first; a wide chunk of 33 that
    // lands first. Slot 1 beside it with other widths and keeps, and rounds in the engine's order.
    const plans = [_]Plan{
        .{ .rows = .{ 7, 4 }, .keeps = .{ 5, 2 } },
        .{ .rows = .{ 7, 3 }, .keeps = .{ 5, 3 }, .after = true },
        .{ .rows = .{ 7, 0 }, .keeps = .{ 5, 0 } },
        .{ .rows = .{ 7, 6 }, .keeps = .{ 5, 1 }, .after = true },
        .{ .rows = .{ 7, 1 }, .keeps = .{ 5, 1 } },
        .{ .rows = .{ 7, 5 }, .keeps = .{ 5, 4 } },
        .{ .rows = .{ 7, 2 }, .keeps = .{ 3, 2 }, .after = true },
        .{ .rows = .{ 1, 8 }, .keeps = .{ 1, 3 } },
        .{ .rows = .{ 4, 8 }, .keeps = .{ 2, 3 }, .after = true },
        .{ .rows = .{ 31, 8 }, .keeps = .{ 20, 8 } },
        .{ .rows = .{ 6, 8 }, .keeps = .{ 6, 7 } },
        .{ .rows = .{ 33, 5 }, .keeps = .{ 33, 5 } },
        .{ .rows = .{ 3, 2 }, .keeps = .{ 2, 2 }, .after = true },
    };
    var pools: [2]attention.Pool = undefined;
    // every row a stream reads it wrote, from position 0: the pools need no zeros
    for (&pools) |*pool| pool.* = try .init(m.attention.?.shape, 1);
    defer for (&pools) |*pool| pool.deinit();
    var slots: [2]Slot = undefined;
    var alone: [2]forward.State = undefined;
    for (&slots, &alone, &pools) |*slot, *state, *pool| {
        slot.* = try .open(gpa, m, &.{pool.*}, null, attention.chunk);
        try slot.table.set(&.{0});
        state.* = try .init(gpa, m, tiny.max_rows, attention.chunk);
    }
    defer for (&slots, &alone) |*slot, *state| {
        slot.deinit();
        state.deinit();
    };
    var shared: rounds.Shared = try .init(tiny.max_rows, m.shape.delta_net);
    defer shared.deinit();
    var round: rounds.Round = try .init(m, 2, tiny.max_rows);
    defer round.deinit();
    var together: forward.Buffers = try .init(gpa, m, tiny.max_rows, attention.chunk);
    defer together.deinit();
    var alone_buf: forward.Buffers = try .init(gpa, m, tiny.max_rows, attention.chunk);
    defer alone_buf.deinit();
    var saves: [1]gdn.Scratch = .{try .init(tiny.max_rows, m.shape.delta_net)};
    defer saves[0].deinit();

    var tokens: [2][64]i32 = undefined;
    var landed = false;
    for (plans, 0..) |plan, r| {
        errdefer std.debug.print("round {d}\n", .{r});
        var segments: [2]Segment = undefined;
        var staged: [2]Segment = undefined;
        var n: usize = 0;
        // a wide chunk is the round's last segment
        const order: [2]usize = if (plan.rows[0] >= gdn.wide_rows) .{ 1, 0 } else .{ 0, 1 };
        for (order) |s| {
            const rows = plan.rows[s];
            if (rows == 0) continue;
            for (tokens[s][0..rows], 0..) |*token, i| token.* = @intCast((r * 37 + s * 101 + i * 13 + slots[s].pos) % 300 + 1);
            segments[n] = .{ .slot = s, .tokens = tokens[s][0..rows], .keep = plan.keeps[s] };
            staged[n] = .{ .slot = s, .tokens = tokens[s][0..rows], .keep = rows };
            if (slots[s].behind > 0 and forward.landsAt(slots[s].behind, rows, rounds.heldRows())) landed = true;
            n += 1;
        }
        if (plan.after) {
            try rounds.stage(gpa, m, &slots, &round, &together, staged[0..n]);
            try rounds.compute(m, &slots, &shared, &together, &round, staged[0..n], null, .all, null, &saves);
            try rounds.replayKept(m, &slots, &round, segments[0..n], &saves);
            try rounds.commit(m, &slots, &together, &round, segments[0..n]);
        } else {
            try rounds.run(gpa, m, &slots, &shared, &round, &together, segments[0..n]);
        }
        var at: usize = 0;
        for (segments[0..n]) |segment| {
            const state = &alone[segment.slot];
            try forward.forward(m, state, &alone_buf, segment.tokens);
            try forward.commit(m, state, alone_buf, segment.tokens, segment.keep);
            try expectRows(together.logits, at, alone_buf.logits, 0, segment.tokens.len);
            try expectSlot(slots[segment.slot], state.*);
            at += segment.tokens.len;
        }
        if (recorded) |list| try record(list, together.logits, at, &slots);
    }
    // the plan reached every path it names: rows left behind, and slots landed before going on
    try testing.expect(landed);
}
