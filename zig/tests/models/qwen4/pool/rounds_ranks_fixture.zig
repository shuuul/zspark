//! `rounds_ranks_test.zig`'s check of a rank's rounds, which `mtp_ranks_test.zig` also runs on the checkpoint's
//! first layers it loads for two ranks (spec 133 WS-08 moved it here): four packed streams on one share alone,
//! row for row against each stream alone, then what each slot holds against the lone state.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const model = @import("../../../../src/models/qwen4/model.zig");
const check = @import("../../../../src/models/qwen4/checks/ranks_streams.zig");
const rounds = @import("../../../../src/models/qwen4/pool/rounds.zig");
const rounds_ranks = @import("../../../../src/models/qwen4/pool/rounds_ranks.zig");
const ranks = @import("../../../../src/models/qwen4/ranks_forward.zig");

const One = ranks.Held(1);
const Desk = rounds_ranks.Desk;

/// A share alone in this process for the check: `share` as rank `rank`, windows of the check's rows.
pub fn alone(share: model.Model, rank: usize) One.Model {
    return .{ .ranks = .{share}, .first = rank, .max_rows = check.rows, .alone = true };
}

/// The check on a checkpoint's first four layers: the share holds the PLE and one attention layer, and its
/// four packed streams are each the stream alone.
pub fn checkpointPacked(gpa: Allocator, m: One.Model) !void {
    var ples: usize = 0;
    var attending: usize = 0;
    for (m.ranks[0].layers) |layer| {
        ples += @intFromBool(layer.ple != null);
        attending += @intFromBool(layer.block == .attention);
    }
    try testing.expect(ples == 1 and attending == 1);
    try packedIsAlone(gpa, m, 51);
}

/// This rank's half of a row, the other half zero: the share is alone here.
pub fn ownHalf(b: One.Buffers, row: usize, out: []u16) One.Error!void {
    const half: usize = @intCast(b.ranks[0].logits.view.size[1]);
    @memset(out, 0);
    try cuda.synchronize();
    try b.ranks[0].logits.downloadAt(row * half * 2, std.mem.sliceAsBytes(out[0..half]));
}

pub fn rowWidth(buffer: cuda.Buffer) usize {
    return buffer.bytes() / @as(usize, @intCast(buffer.view.size[0]));
}

pub fn expectRows(got: cuda.Buffer, got_row: usize, want: cuda.Buffer, want_row: usize, count: usize) !void {
    const width = rowWidth(got);
    try testing.expectEqual(rowWidth(want), width);
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

pub fn expectBuffer(got: cuda.Buffer, want: cuda.Buffer) !void {
    try expectRows(got, 0, want, 0, @intCast(got.view.size[0]));
}

/// What `slot` holds is what `state` holds after the same tokens alone: the DeltaNet states and conv windows,
/// the PLE's tails and history, every cache row as stored, where `table` says it lies, and the pooled keys.
fn expectSlot(share: model.Model, slot: rounds.Slot, table: []const i32, state: forward.State) !void {
    try testing.expectEqual(state.pos, slot.pos);
    for (state.recurrent, 0..) |want, linear| {
        var at: cuda.Buffer = try forward.Snapshot.likeOf(want[0]);
        defer at.deinit();
        try forward.committed(state, linear, at);
        // the slot's at its position too: windows of drafts leave rows behind (spec 144 WS-09)
        var got: cuda.Buffer = try forward.Snapshot.likeOf(want[0]);
        defer got.deinit();
        try rounds.committed(slot, linear, got);
        try expectBuffer(got, at);
    }
    for (slot.conv, state.conv) |window, want| try expectBuffer(window, want);
    for (slot.ple, state.ple) |branch, want| {
        try expectBuffer(branch.tail, want.tail);
        try testing.expectEqualSlices(u32, want.history, branch.history);
    }
    try testing.expectEqual(state.caches.len, slot.caches.len);
    for (slot.caches, state.caches) |cache, want| {
        var at: usize = 0;
        while (at < slot.pos) : (at += attention.chunk) {
            const n = @min(attention.chunk, slot.pos - at);
            const place = @as(usize, @intCast(table[at / attention.chunk])) * attention.chunk;
            try expectRows(cache.k, place, want.k, at, n);
            try expectRows(cache.v, place, want.v, at, n);
            try expectRows(cache.index_keys, place, want.index_keys, at, n);
            try testing.expectEqual(want.k_scale != null, cache.k_scale != null);
            if (cache.k_scale) |scale| try expectRows(scale, place, want.k_scale.?, at, n);
            if (cache.v_scale) |scale| try expectRows(scale, place, want.v_scale.?, at, n);
        }
        const blocks = slot.pos / share.attention.?.shape.ratio;
        if (blocks > 0) try expectRows(cache.pooled, 0, want.pooled, 0, blocks);
    }
}

/// The check's four streams on `m`, one share alone: packed against lone, row for row, then each slot
/// against its stream run alone once more.
pub fn packedIsAlone(gpa: Allocator, m: One.Model, seed: u64) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const asked = try check.prompts(arena_state.allocator(), seed);
    const share = m.ranks[0];
    var state: One.State = try .init(gpa, m, check.rows, check.room);
    defer state.deinit();
    var b: One.Buffers = try .init(gpa, m, check.rows, check.room);
    defer b.deinit();
    var desk: Desk = try .init(gpa, share, check.streams, check.rows, check.room, check.count);
    defer desk.deinit();
    var pool: extents.Pool = try .init(gpa, .{ .extents = check.count, .capacity = check.room, .slots = check.streams });
    defer pool.deinit();
    const row = try gpa.alloc(u16, share.head.n * 2);
    defer gpa.free(row);
    var told: std.ArrayList(i32) = .empty;
    defer told.deinit(gpa);
    const taking: check.Taking = .{ .row = row, .told = &told, .read = ownHalf, .gpa = gpa };

    var lone: check.Run = .{};
    defer lone.deinit(gpa);
    try check.alone(m, &state, &b, taking, asked, &lone);
    var together: check.Run = .{};
    defer together.deinit(gpa);
    try check.together(gpa, m, &desk, &b, &pool, taking, asked, &together);
    if (together.differs(&lone)) |where| {
        std.debug.print("stream {d} among the others parts from the stream alone at row {?d}\n", .{ where.stream, where.row });
        return error.TestExpectedEqual;
    }
    // no table is the identity: a second extent was taken after the other streams' first
    try testing.expectEqualSlices(i32, &.{ 0, 4 }, desk.tables[0][0..desk.placed[0]]);
    try testing.expectEqualSlices(i32, &.{1}, desk.tables[1][0..desk.placed[1]]);
    try testing.expectEqualSlices(i32, &.{2}, desk.tables[2][0..desk.placed[2]]);
    try testing.expectEqualSlices(i32, &.{ 3, 5 }, desk.tables[3][0..desk.placed[3]]);

    for (asked, 0..) |prompt, stream| {
        for (&state.ranks) |*one| try one.reset();
        state.pos = 0;
        var at: usize = 0;
        while (at < prompt.len) {
            const window = prompt[at..@min(at + check.rows, prompt.len)];
            try One.forward(m, &state, &b, window);
            try One.commit(m, &state, b, window, window.len);
            at += window.len;
        }
        for (together.tokens[stream][0 .. check.made - 1]) |token| {
            try One.forward(m, &state, &b, &.{token});
            try One.commit(m, &state, b, &.{token}, 1);
        }
        try testing.expectEqual(prompt.len + check.made - 1, desk.slots[stream].pos);
        try expectSlot(share, desk.slots[stream], desk.tables[stream], state.ranks[0]);
    }
}
