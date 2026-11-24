//! `rounds.zig`'s tests. The random model (`testing.zig`'s `tiny_model`) is one DeltaNet layer with a tiny PLE
//! and one attention layer. A checkpoint's first four layers are the gate at the bottom, one scattered round,
//! not the whole acceptance.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const tinyModel = fixtures.tiny_model.tinyModel;
const tiny = fixtures.tiny_model.tiny;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const gdn = @import("../../../../src/models/qwen4/blocks/gdn.zig");
const model = @import("../../../../src/models/qwen4/model.zig");
const rounds_file = @import("../../../../src/models/qwen4/pool/rounds.zig");
const taps_file = @import("../../../../src/models/qwen4/taps.zig");
const Gem = rounds_file.Gem;
const Graphs = rounds_file.Graphs;
const Round = rounds_file.Round;
const Segment = rounds_file.Segment;
const Shared = rounds_file.Shared;
const Slot = rounds_file.Slot;
const address = rounds_file.address;
const chunkRows = rounds_file.chunkRows;
const commit = rounds_file.commit;
const compute = rounds_file.compute;
const promptBeside = rounds_file.promptBeside;
const prompt_chunk = rounds_file.prompt_chunk;
const reserve = rounds_file.reserve;
const run = rounds_file.run;
const stage = rounds_file.stage;
const zs_qwen4_attn_prep_rows = cuda.kernels.zs_qwen4_attn_prep_rows;

fn storedAt(table: []const i32, pos: usize) usize {
    return @as(usize, @intCast(table[pos >> 9])) * attention.chunk + (pos & (attention.chunk - 1));
}

fn rowWidth(buffer: cuda.Buffer) usize {
    return buffer.bytes() / @as(usize, @intCast(buffer.view.size[0]));
}

fn extentBytes(gpa: Allocator, buffer: cuda.Buffer, extent: usize) ![]u8 {
    std.debug.assert((extent + 1) * attention.chunk <= @as(usize, @intCast(buffer.view.size[0])));
    return gpa.alloc(u8, rowWidth(buffer) * attention.chunk);
}

fn downloadExtent(buffer: cuda.Buffer, extent: usize, dest: []u8) !void {
    try buffer.downloadAt(extent * attention.chunk * rowWidth(buffer), dest);
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

fn expectCacheRows(got: attention.Cache, table: []const i32, want: attention.Cache, pos: usize, count: usize) !void {
    for (0..count) |i| {
        const place = storedAt(table, pos + i);
        try expectRows(got.k, place, want.k, pos + i, 1);
        try expectRows(got.v, place, want.v, pos + i, 1);
        try expectRows(got.index_keys, place, want.index_keys, pos + i, 1);
    }
}

/// Blocks the window completes, pooled by logical block. A listed row attends through these keys, so a pool
/// that read another stream's indexer keys fails here even when the selected set would not have moved.
fn expectPooled(got: attention.Cache, want: attention.Cache, pos: usize, rows: usize, ratio: usize) !void {
    const end = pos + rows;
    var b = pos / ratio;
    while (ratio * (b + 1) <= end) : (b += 1) try expectRows(got.pooled, b, want.pooled, b, 1);
}

/// The round equals each stream alone: hidden rows, logits, the committed DeltaNet state and conv window, the
/// PLE tail, the cache rows the window wrote, and the pooled keys of every block it completed.
fn expectRound(gpa: Allocator, m: model.Model, slots: []Slot, shared: *Shared, round: *Round, together: *forward.Buffers, alone_state: []forward.State, alone: *forward.Buffers, segments: []const Segment, places: []const []const i32) !void {
    const starts = try gpa.alloc(usize, slots.len);
    defer gpa.free(starts);
    for (slots, starts) |slot, *start| start.* = slot.pos;
    try run(gpa, m, slots, shared, round, together, segments);
    var at: usize = 0;
    for (segments) |segment| {
        const state = &alone_state[segment.slot];
        std.debug.assert(state.pos == starts[segment.slot]);
        try forward.forward(m, state, alone, segment.tokens);
        try forward.commit(m, state, alone.*, segment.tokens, segment.keep);
        const n = segment.tokens.len;
        try expectRows(together.h, at, alone.h, 0, n);
        try expectRows(together.logits, at, alone.logits, 0, n);
        try expectSlot(m, &slots[segment.slot], state, places[segment.slot], starts[segment.slot], n);
        at += n;
    }
}

/// What `slot` holds after `n` rows from `start` is what `state` holds: the DeltaNet state and conv window,
/// the PLE tail, the cache rows written and the pooled keys of every block completed.
fn expectSlot(m: model.Model, slot: *const Slot, state: *const forward.State, places: []const i32, start: usize, n: usize) !void {
    try testing.expectEqual(state.pos, slot.pos);
    for (state.recurrent, 0..) |want, linear| {
        // each state at its position: its committed side, replayed over the rows it is behind by (a session's,
        // spec 096; a slot's, spec 144 WS-09)
        var at: cuda.Buffer = try forward.Snapshot.likeOf(want[0]);
        defer at.deinit();
        try forward.committed(state.*, linear, at);
        var got: cuda.Buffer = try forward.Snapshot.likeOf(want[0]);
        defer got.deinit();
        try rounds_file.committed(slot.*, linear, got);
        try expectBuffer(got, at);
    }
    for (slot.conv, state.conv) |window, want| try expectBuffer(window, want);
    for (slot.ple, state.ple) |branch, want| {
        try expectBuffer(branch.tail, want.tail);
        try testing.expectEqualSlices(u32, want.history, branch.history);
    }
    for (slot.caches, state.caches) |cache, want| {
        try expectCacheRows(cache, places, want, start, n);
        try expectPooled(cache, want, start, n, m.attention.?.shape.ratio);
    }
}

/// A pool for slots `placeAt` moves: their windows read the rows below the position that none wrote, which the
/// state alone holds as its contiguous caches' zeros. A pool is not zeroed (`attention.Pool.init`), since no
/// served sequence reads a row it has not written; these are.
fn placedPool(shape: attention.Shape, count: usize) !attention.Pool {
    var pool: attention.Pool = try .init(shape, count);
    errdefer pool.deinit();
    for ([_]?cuda.Buffer{ pool.k, pool.v, pool.k_scale, pool.v_scale, pool.index_keys }) |array| if (array) |b| try b.zero();
    return pool;
}

fn placeAt(slot: *Slot, state: *forward.State, pos: usize) !void {
    // the rows a window of drafts kept whole left behind (spec 096) are replayed before the position moves
    try forward.land(state);
    try rounds_file.land(slot);
    slot.pos = pos;
    state.pos = pos;
    try state.position.set(pos);
}

const identity_table = [_]i32{ 0, 1, 2, 3, 4, 5 };
// Interleaved and descending. A window at 2,002 reads places 8, 6, 4 and 2; one at 2,101 reads 9, 7, 5, 3
// and 1. Neither place one writes is a place the other reads.
const scattered_a = [_]i32{ 8, 6, 4, 2, 0, 0 };
const scattered_b = [_]i32{ 9, 7, 5, 3, 1, 0 };
const scattered_extents: usize = 10;

fn openSlots(gpa: Allocator, m: model.Model, pools: []attention.Pool, tables: []const []const i32, out: []Slot) !void {
    for (out, 0..) |*slot, s| {
        const pool = if (pools.len == 1) &pools[0] else &pools[s];
        slot.* = try .open(gpa, m, &.{pool.*}, null, tiny.capacity);
        try slot.table.set(tables[s]);
    }
}

// A captured round against the same launches run directly, two streams of unequal widths and one slot empty,
// then one stream whose graph keeps every row and whose replay, after the logits, keeps two. The fixed
// selection, its empty slot skipped on the device (spec 144 port 5), against each stream's own (`run`'s).
test "shared rounds: a graph equals direct launches" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    const prefix = [_]Segment{
        .{ .slot = 0, .tokens = &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, .keep = 8 },
        .{ .slot = 1, .tokens = &.{ 11, 12, 13, 14, 15 }, .keep = 5 },
    };
    const compared = [_]Segment{
        .{ .slot = 0, .tokens = &.{ 21, 22, 23 }, .keep = 3 },
        .{ .slot = 1, .tokens = &.{ 31, 32 }, .keep = 2 },
    };
    const empty = [_]i32{ 0, 0, 0, 0, 0, 0 };
    const modes = [_]struct { shared: bool, extents: usize, tables: [3][]const i32 }{
        .{ .shared = true, .extents = scattered_extents, .tables = .{ &scattered_a, &scattered_b, &empty } },
        .{ .shared = false, .extents = 6, .tables = .{ &identity_table, &identity_table, &identity_table } },
    };
    for (modes) |mode| {
        var pools_a: [3]attention.Pool = undefined;
        var pools_b: [3]attention.Pool = undefined;
        var pools_c: [3]attention.Pool = undefined;
        const n: usize = if (mode.shared) 1 else 3;
        for (0..n) |i| {
            pools_a[i] = try placedPool(m.attention.?.shape, mode.extents);
            pools_b[i] = try placedPool(m.attention.?.shape, mode.extents);
            pools_c[i] = try placedPool(m.attention.?.shape, mode.extents);
        }
        defer for (0..n) |i| {
            pools_a[i].deinit();
            pools_b[i].deinit();
            pools_c[i].deinit();
        };
        var slots_a: [3]Slot = undefined;
        var slots_b: [3]Slot = undefined;
        var slots_c: [3]Slot = undefined;
        try openSlots(gpa, m, pools_a[0..n], &mode.tables, &slots_a);
        try openSlots(gpa, m, pools_b[0..n], &mode.tables, &slots_b);
        try openSlots(gpa, m, pools_c[0..n], &mode.tables, &slots_c);
        defer for (&slots_a, &slots_b, &slots_c) |*a, *b, *c| {
            a.deinit();
            b.deinit();
            c.deinit();
        };
        var shared_a: Shared = try .init(tiny.max_rows, m.shape.delta_net);
        defer shared_a.deinit();
        var shared_b: Shared = try .init(tiny.max_rows, m.shape.delta_net);
        defer shared_b.deinit();
        var shared_c: Shared = try .init(tiny.max_rows, m.shape.delta_net);
        defer shared_c.deinit();
        var round_a: Round = try .init(m, 3, tiny.max_rows);
        defer round_a.deinit();
        var round_b: Round = try .init(m, 3, tiny.max_rows);
        defer round_b.deinit();
        var round_c: Round = try .init(m, 3, tiny.max_rows);
        defer round_c.deinit();
        var buf_a: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer buf_a.deinit();
        var buf_b: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer buf_b.deinit();
        var buf_c: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer buf_c.deinit();
        var graphs: Graphs = try .init(gpa, m, 3, tiny.max_rows, tiny.capacity);
        defer graphs.deinit();
        var gem_b: Gem = try .init(3, tiny.max_rows, m.attention.?.shape, tiny.capacity);
        defer gem_b.deinit();
        try run(gpa, m, &slots_a, &shared_a, &round_a, &buf_a, &prefix);
        try run(gpa, m, &slots_b, &shared_b, &round_b, &buf_b, &prefix);
        try run(gpa, m, &slots_c, &shared_c, &round_c, &buf_c, &prefix);
        for ([_]*[3]Slot{ &slots_a, &slots_b, &slots_c }) |slots| {
            slots[0].pos = 2000;
            slots[1].pos = 2100;
        }
        const rows: usize = 5;
        const context = forward.Graphs.bucket(buf_a, 2102);
        const free_before = try cuda.memory();
        const bytes_before = cuda.account.bytes;
        const used = round_a.used;
        // Refuse a short projection after several launches have already been recorded. Preparation must
        // restore the round even when the compute fails, and the abandoned graph must never run.
        {
            const projection_rows = buf_a.projections[0].view.size[0];
            defer buf_a.projections[0].view.size[0] = projection_rows;
            buf_a.projections[0].view.size[0] = rows - 1;
            try testing.expectError(error.RoundPastProjection, graphs.prepare(m, &slots_a, &shared_a, &buf_a, &round_a, rows, context, null));
            try testing.expectEqual(used, round_a.used);
            try testing.expectEqual(0, graphs.cache.count());
            try expectRows(buf_a.h, 0, buf_b.h, 0, rows);
        }
        try graphs.prepare(m, &slots_a, &shared_a, &buf_a, &round_a, rows, context, null);
        try testing.expectEqual(used, round_a.used);
        try testing.expectEqual(1, graphs.cache.count());
        try testing.expect(graphs.cache.contains(.{ .rows = rows, .context = context }));
        try expectRows(buf_a.h, 0, buf_b.h, 0, rows);
        try graphs.prepare(m, &slots_a, &shared_a, &buf_a, &round_a, rows, context, null);
        try testing.expectEqual(1, graphs.cache.count());
        const free_after = try cuda.memory();
        std.debug.print("shared graph: count={d} capture_ms={d:.1} device_free={d}->{d} buffers={d}->{d}\n", .{ graphs.cache.count(), @as(f64, @floatFromInt(graphs.cache.capturing())) / 1e6, free_before.free, free_after.free, bytes_before, cuda.account.bytes });
        try stage(gpa, m, &slots_a, &round_a, &buf_a, &compared);
        try graphs.launch(rows, context);
        try commit(m, &slots_a, &buf_a, &round_a, &compared);
        try stage(gpa, m, &slots_b, &round_b, &buf_b, &compared);
        try compute(m, &slots_b, &shared_b, &buf_b, &round_b, &compared, context, .all, &gem_b, null);
        try commit(m, &slots_b, &buf_b, &round_b, &compared);
        try expectRows(buf_a.h, 0, buf_b.h, 0, rows);
        try expectRows(buf_a.logits, 0, buf_b.logits, 0, rows);
        try run(gpa, m, &slots_c, &shared_c, &round_c, &buf_c, &compared);
        try expectRows(buf_b.h, 0, buf_c.h, 0, rows);
        try expectRows(buf_b.logits, 0, buf_c.logits, 0, rows);
        for (slots_a, slots_b, 0..) |a, b, s| {
            try testing.expectEqual(b.pos, a.pos);
            try testing.expectEqual(b.behind, a.behind);
            for (a.recurrent, a.current, b.recurrent, b.current, a.held, b.held) |pair, side, want, want_side, held, held_want| {
                try testing.expectEqual(want_side, side);
                try expectBuffer(pair[side], want[want_side]);
                // the rows a window of drafts left behind (spec 144 WS-09), as the graph and the launches saved them
                if (a.behind > 0) {
                    try expectRows(held.k, 0, held_want.k, 0, a.behind);
                    try expectRows(held.v, 0, held_want.v, 0, a.behind);
                    try expectRows(held.g, 0, held_want.g, 0, a.behind);
                    try expectRows(held.beta, 0, held_want.beta, 0, a.behind);
                }
            }
            for (a.conv, b.conv) |window, want| try expectBuffer(window, want);
            for (a.ple, b.ple) |branch, want| {
                try expectBuffer(branch.tail, want.tail);
                try testing.expectEqualSlices(u32, want.history, branch.history);
            }
            if (s < 2) {
                const pos: usize = if (s == 0) 2000 else 2100;
                for (0..compared[s].tokens.len) |i| {
                    const place = storedAt(mode.tables[s], pos + i);
                    try expectRows(a.caches[0].k, place, b.caches[0].k, place, 1);
                    try expectRows(a.caches[0].v, place, b.caches[0].v, place, 1);
                    try expectRows(a.caches[0].index_keys, place, b.caches[0].index_keys, place, 1);
                }
                try expectPooled(a.caches[0], b.caches[0], pos, compared[s].tokens.len, m.attention.?.shape.ratio);
            }
        }
    }
}

test "shared rounds: two streams of unequal widths equal each alone" {
    try unequalWidths();
}

fn unequalWidths() !void {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    // A prefix writes indexer keys, then the compared round sits one stream under 2,051 positions and one past
    // it, so the listed stream's pooling reads those keys. Zeros would not show a pool that read the other stream.
    const prefix = [_]Segment{
        .{ .slot = 0, .tokens = &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, .keep = 8 },
        .{ .slot = 1, .tokens = &.{ 11, 12, 13, 14, 15 }, .keep = 5 },
    };
    const compared = [_]Segment{
        .{ .slot = 0, .tokens = &.{ 21, 22, 23 }, .keep = 3 },
        .{ .slot = 1, .tokens = &.{ 31, 32 }, .keep = 2 },
    };
    const modes = [_]struct { shared: bool, extents: usize, tables: [2][]const i32 }{
        .{ .shared = true, .extents = scattered_extents, .tables = .{ &scattered_a, &scattered_b } },
        .{ .shared = false, .extents = 6, .tables = .{ &identity_table, &identity_table } },
    };
    for (modes) |mode| {
        var pools: [2]attention.Pool = undefined;
        const n: usize = if (mode.shared) 1 else 2;
        for (0..n) |i| pools[i] = try placedPool(m.attention.?.shape, mode.extents);
        defer for (pools[0..n]) |*pool| pool.deinit();
        var slots: [2]Slot = undefined;
        for (&slots, 0..) |*slot, s| {
            const pool = if (mode.shared) &pools[0] else &pools[s];
            slot.* = try .open(gpa, m, &.{pool.*}, null, tiny.capacity);
            try slot.table.set(mode.tables[s]);
        }
        defer for (&slots) |*slot| slot.deinit();
        var shared: Shared = try .init(tiny.max_rows, m.shape.delta_net);
        defer shared.deinit();
        var round: Round = try .init(m, 2, tiny.max_rows);
        defer round.deinit();
        var together: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer together.deinit();
        var alone_buf: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer alone_buf.deinit();
        var alone: [2]forward.State = undefined;
        alone[0] = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer alone[0].deinit();
        alone[1] = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer alone[1].deinit();
        try placeAt(&slots[1], &alone[1], 32);
        try expectRound(gpa, m, &slots, &shared, &round, &together, &alone, &alone_buf, &prefix, &mode.tables);
        try placeAt(&slots[0], &alone[0], 2000);
        try placeAt(&slots[1], &alone[1], 2100);
        try expectRound(gpa, m, &slots, &shared, &round, &together, &alone, &alone_buf, &compared, &mode.tables);
    }
}

test "shared rounds: replay keeps one stream whole and one row of the other" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    const segments = [_]Segment{
        .{ .slot = 0, .tokens = &.{ 1, 2, 3, 4 }, .keep = 4 },
        .{ .slot = 1, .tokens = &.{ 8, 9, 10, 11 }, .keep = 1 },
    };
    const modes = [_]struct { shared: bool, extents: usize, tables: [2][]const i32 }{
        .{ .shared = true, .extents = scattered_extents, .tables = .{ &scattered_a, &scattered_b } },
        .{ .shared = false, .extents = 6, .tables = .{ &identity_table, &identity_table } },
    };
    for (modes) |mode| {
        var pools: [2]attention.Pool = undefined;
        const n: usize = if (mode.shared) 1 else 2;
        for (0..n) |i| pools[i] = try placedPool(m.attention.?.shape, mode.extents);
        defer for (pools[0..n]) |*pool| pool.deinit();
        var slots: [2]Slot = undefined;
        for (&slots, 0..) |*slot, s| {
            const pool = if (mode.shared) &pools[0] else &pools[s];
            slot.* = try .open(gpa, m, &.{pool.*}, null, tiny.capacity);
            try slot.table.set(mode.tables[s]);
        }
        defer for (&slots) |*slot| slot.deinit();
        slots[1].pos = 16;
        var shared: Shared = try .init(tiny.max_rows, m.shape.delta_net);
        defer shared.deinit();
        var round: Round = try .init(m, 2, tiny.max_rows);
        defer round.deinit();
        var together: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer together.deinit();
        var alone_buf: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer alone_buf.deinit();
        var alone: [2]forward.State = undefined;
        alone[0] = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer alone[0].deinit();
        alone[1] = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer alone[1].deinit();
        try alone[1].position.set(16);
        alone[1].pos = 16;
        try expectRound(gpa, m, &slots, &shared, &round, &together, &alone, &alone_buf, &segments, &mode.tables);
    }
}

test "shared rounds: a stream at the start, the middle and the end of four" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    const segments = [_]Segment{
        .{ .slot = 0, .tokens = &.{ 1, 2 }, .keep = 2 },
        .{ .slot = 1, .tokens = &.{ 3, 4, 5 }, .keep = 3 },
        .{ .slot = 2, .tokens = &.{6}, .keep = 1 },
        .{ .slot = 3, .tokens = &.{ 7, 8, 9, 10 }, .keep = 4 },
    };
    const scattered = [_][]const i32{
        &.{0},
        &.{1},
        &.{2},
        &.{3},
    };
    const same = [_][]const i32{ &.{0}, &.{0}, &.{0}, &.{0} };
    // Scattered: one pool, a place a stream, every position inside the first extent. Identity: a pool a stream,
    // so four copies of place 0 do not alias.
    for ([_]struct { shared: bool, tables: []const []const i32 }{
        .{ .shared = true, .tables = &scattered },
        .{ .shared = false, .tables = &same },
    }) |mode| {
        var pools: [4]attention.Pool = undefined;
        const n: usize = if (mode.shared) 1 else 4;
        for (0..n) |i| pools[i] = try placedPool(m.attention.?.shape, if (mode.shared) 4 else 1);
        defer for (pools[0..n]) |*pool| pool.deinit();
        var slots: [4]Slot = undefined;
        var alone: [4]forward.State = undefined;
        for (&slots, &alone, mode.tables, [_]usize{ 0, 20, 40, 70 }, 0..) |*slot, *state, table, pos, s| {
            const pool = if (mode.shared) &pools[0] else &pools[s];
            slot.* = try .open(gpa, m, &.{pool.*}, null, attention.chunk);
            state.* = try .init(gpa, m, tiny.max_rows, attention.chunk);
            try slot.table.set(table);
            try placeAt(slot, state, pos);
        }
        defer for (&slots) |*slot| slot.deinit();
        defer for (&alone) |*state| state.deinit();
        var shared: Shared = try .init(tiny.max_rows, m.shape.delta_net);
        defer shared.deinit();
        var round: Round = try .init(m, 4, tiny.max_rows);
        defer round.deinit();
        var together: forward.Buffers = try .init(gpa, m, tiny.max_rows, attention.chunk);
        defer together.deinit();
        var alone_buf: forward.Buffers = try .init(gpa, m, tiny.max_rows, attention.chunk);
        defer alone_buf.deinit();
        try expectRound(gpa, m, &slots, &shared, &round, &together, &alone, &alone_buf, &segments, mode.tables);
    }
}

test "shared rounds: a chunk beside two windows" {
    try chunkBesideWindows();
}

fn chunkBesideWindows() !void {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    var wide_tokens: [32]i32 = undefined;
    for (&wide_tokens, 0..) |*token, i| token.* = @intCast(i + 1);
    // The flag's path: the replies' windows, then the chunk. 32 rows is what this model can hold of
    // `prompt_chunk`, and it is the wide path a real chunk takes.
    try testing.expectEqual(@as(usize, 512), prompt_chunk);
    try testing.expectEqual(@as(usize, 512), chunkRows(2048, prompt_chunk));
    try testing.expectEqual(wide_tokens.len, chunkRows(wide_tokens.len, prompt_chunk));
    const replies = [_]Segment{
        .{ .slot = 1, .tokens = &.{41}, .keep = 1 },
        .{ .slot = 2, .tokens = &.{42}, .keep = 1 },
    };
    const segments = try promptBeside(gpa, &replies, 0, &wide_tokens, prompt_chunk);
    defer gpa.free(segments);
    try testing.expectEqual(@as(usize, 3), segments.len);
    try testing.expectEqual(@as(usize, 1), segments[0].slot);
    try testing.expectEqual(@as(usize, 2), segments[1].slot);
    try testing.expectEqual(@as(usize, 0), segments[2].slot);
    try testing.expectEqual(wide_tokens.len, segments[2].tokens.len);
    var pool: attention.Pool = try placedPool(m.attention.?.shape, 3);
    defer pool.deinit();
    var slots: [3]Slot = undefined;
    var alone: [3]forward.State = undefined;
    // One extent each, three places: the windows read only their own rows, not the chunk's.
    const tables = [_][]const i32{ &.{0}, &.{1}, &.{2} };
    for (&slots, &alone, tables, [_]usize{ 0, 64, 96 }) |*slot, *state, table, pos| {
        slot.* = try .open(gpa, m, &.{pool}, null, tiny.capacity);
        state.* = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        try slot.table.set(table);
        try placeAt(slot, state, pos);
    }
    defer for (&slots) |*slot| slot.deinit();
    defer for (&alone) |*state| state.deinit();
    var shared: Shared = try .init(tiny.max_rows, m.shape.delta_net);
    defer shared.deinit();
    var round: Round = try .init(m, 3, tiny.max_rows);
    defer round.deinit();
    var together: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
    defer together.deinit();
    var alone_buf: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
    defer alone_buf.deinit();
    try expectRound(gpa, m, &slots, &shared, &round, &together, &alone, &alone_buf, segments, &tables);
}

test "shared rounds: a fork inside an extent equals the same tokens from the start" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    const capacity = 2048;
    const at: usize = 137;
    var sequence: [153]i32 = undefined;
    for (&sequence, 0..) |*token, i| token.* = @intCast((i % 17) + 1);
    var chains: extents.Pool = try .init(gpa, .{ .extents = 6, .capacity = capacity, .slots = 2 });
    defer chains.deinit();
    const source_chain = try chains.open();
    try chains.grow(source_chain, at);
    var pool: attention.Pool = try .init(m.attention.?.shape, 6);
    defer pool.deinit();
    var source: Slot = try .open(gpa, m, &.{pool}, null, capacity);
    defer source.deinit();
    try source.table.set(source_chain.table.items);
    var shared: Shared = try .init(tiny.max_rows, m.shape.delta_net);
    defer shared.deinit();
    var round: Round = try .init(m, 1, tiny.max_rows);
    defer round.deinit();
    var together: forward.Buffers = try .init(gpa, m, tiny.max_rows, capacity);
    defer together.deinit();
    var serial: forward.State = try .init(gpa, m, tiny.max_rows, capacity);
    defer serial.deinit();
    var serial_buf: forward.Buffers = try .init(gpa, m, tiny.max_rows, capacity);
    defer serial_buf.deinit();

    var cursor: usize = 0;
    while (cursor < at) {
        const n = @min(16, at - cursor);
        const window = [_]Segment{.{ .slot = 0, .tokens = sequence[cursor..][0..n], .keep = n }};
        try run(gpa, m, (&source)[0..1], &shared, &round, &together, &window);
        try forward.forward(m, &serial, &serial_buf, sequence[cursor..][0..n]);
        try forward.commit(m, &serial, serial_buf, sequence[cursor..][0..n], n);
        cursor += n;
    }
    try testing.expectEqual(at, source.pos);
    try testing.expectEqual(at, serial.pos);
    chains.append(source_chain, sequence[0..at]);
    const forked = try chains.fork(source_chain, at);
    try testing.expect(forked.copy != null);
    try pool.copy(forked.copy.?);
    var fork: Slot = try .open(gpa, m, &.{pool}, null, capacity);
    defer fork.deinit();
    try fork.table.set(forked.chain.table.items);
    // the rows the source's windows of 16 left behind (spec 144 WS-09) replayed, so its sides are its state
    try rounds_file.land(&source);
    for (fork.recurrent, source.recurrent) |*pair, from| {
        try pair[0].copyFrom(from[0]);
        try pair[1].copyFrom(from[1]);
    }
    @memcpy(fork.current, source.current);
    for (fork.conv, source.conv) |*window, from| try window.copyFrom(from);
    for (fork.ple, source.ple) |*branch, from| {
        try branch.tail.copyFrom(from.tail);
        @memcpy(branch.history, from.history);
    }
    fork.pos = source.pos;
    try fork.caches[0].inherit(m.attention.?.shape, source.caches[0], at);

    // The fork writes its own extent in this same pool. The source's rows are the extent it already held.
    const source_extent: usize = @intCast(source_chain.table.items[0]);
    const snap_k = try extentBytes(gpa, pool.k, source_extent);
    defer gpa.free(snap_k);
    const snap_v = try extentBytes(gpa, pool.v, source_extent);
    defer gpa.free(snap_v);
    const snap_index = try extentBytes(gpa, pool.index_keys, source_extent);
    defer gpa.free(snap_index);
    const snap_state = try gpa.alloc(u8, source.recurrent[0][source.current[0]].bytes());
    defer gpa.free(snap_state);
    try cuda.synchronize();
    try downloadExtent(pool.k, source_extent, snap_k);
    try downloadExtent(pool.v, source_extent, snap_v);
    try downloadExtent(pool.index_keys, source_extent, snap_index);
    try source.recurrent[0][source.current[0]].download(snap_state);

    const rest = sequence[at..];
    const window = [_]Segment{.{ .slot = 0, .tokens = rest, .keep = rest.len }};
    try run(gpa, m, (&fork)[0..1], &shared, &round, &together, &window);
    try forward.forward(m, &serial, &serial_buf, rest);
    try forward.commit(m, &serial, serial_buf, rest, rest.len);
    try expectRows(together.h, 0, serial_buf.h, 0, rest.len);
    try expectRows(together.logits, 0, serial_buf.logits, 0, rest.len);
    // the serial state at its position: its committed side, replayed over the rows it is behind by (spec 096)
    var serial_at = try forward.Snapshot.likeOf(serial.recurrent[0][0]);
    defer serial_at.deinit();
    try forward.committed(serial, 0, serial_at);
    var fork_at = try forward.Snapshot.likeOf(serial.recurrent[0][0]);
    defer fork_at.deinit();
    try rounds_file.committed(fork, 0, fork_at);
    try expectBuffer(fork_at, serial_at);
    try expectBuffer(fork.conv[0], serial.conv[0]);
    try expectBuffer(fork.ple[0].tail, serial.ple[0].tail);

    const after_k: []u8 = try gpa.alloc(u8, snap_k.len);
    defer gpa.free(after_k);
    const after_v: []u8 = try gpa.alloc(u8, snap_v.len);
    defer gpa.free(after_v);
    const after_index: []u8 = try gpa.alloc(u8, snap_index.len);
    defer gpa.free(after_index);
    const after_state: []u8 = try gpa.alloc(u8, snap_state.len);
    defer gpa.free(after_state);
    try cuda.synchronize();
    try downloadExtent(pool.k, source_extent, after_k);
    try downloadExtent(pool.v, source_extent, after_v);
    try downloadExtent(pool.index_keys, source_extent, after_index);
    try source.recurrent[0][source.current[0]].download(after_state);
    try testing.expectEqualSlices(u8, snap_k, after_k);
    try testing.expectEqualSlices(u8, snap_v, after_v);
    try testing.expectEqualSlices(u8, snap_index, after_index);
    try testing.expectEqualSlices(u8, snap_state, after_state);
    try expectCacheRows(fork.caches[0], forked.chain.table.items, serial.caches[0], at, rest.len);
}

test "shared rounds: the head's cache uses the trunk's table" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    var trunk: attention.Pool = try .init(m.attention.?.shape, 6);
    defer trunk.deinit();
    var head_pool: attention.Pool = try .init(m.attention.?.shape, 6);
    defer head_pool.deinit();
    var slot: Slot = try .open(gpa, m, &.{trunk}, &head_pool, tiny.capacity);
    defer slot.deinit();
    const head = slot.head.?;
    try testing.expect(slot.caches[0].table.?.ptr == slot.table.device.view.ptr);
    try testing.expect(head.table.?.ptr == slot.table.device.view.ptr);
    // Extent 0 sits at place 1, so the written row is not the raw position. The pool holds six extents.
    const head_table = [_]i32{ 1, 0, 0, 0, 0, 0 };
    try slot.table.set(&head_table);

    const head_len: usize = 100;
    var mixed: cuda.Buffer = try .init(.bf16, &.{ 1, tiny.shape.hidden });
    defer mixed.deinit();
    var prng: std.Random.DefaultPrng = .init(3);
    const bytes = try gpa.alloc(u8, mixed.bytes());
    defer gpa.free(bytes);
    prng.random().bytes(bytes);
    try mixed.upload(bytes);
    var scratch: attention.Scratch = try .init(tiny.attn, 1, tiny.capacity);
    defer scratch.deinit();
    const weights = m.layers[1].block.attention;
    try weights.proj.forward(mixed.view, scratch.pa.view);
    var sid: cuda.Buffer = try .init(.i32, &.{1});
    defer sid.deinit();
    var pos: cuda.Buffer = try .init(.i32, &.{1});
    defer pos.deinit();
    var tables: cuda.Buffer = try .init(.i64, &.{1});
    defer tables.deinit();
    const zero: i32 = 0;
    const position: i32 = head_len;
    const table_ptr: i64 = address(slot.table.device.view);
    const head_bases = [_]i64{ address(head.k.view), address(head.v.view), address(head.index_keys.view) };
    var bases: cuda.Buffer = try .init(.i64, &.{ 1, 3 });
    defer bases.deinit();
    try sid.upload(std.mem.asBytes(&zero));
    try pos.upload(std.mem.asBytes(&position));
    try tables.upload(std.mem.asBytes(&table_ptr));
    try bases.upload(std.mem.sliceAsBytes(&head_bases));
    const rotary = m.attention.?.rotary;
    try cuda.launched(zs_qwen4_attn_prep_rows(&sid.view, &tables.view, &bases.view, &scratch.pa.view, &pos.view, &weights.q_scale.view, &weights.k_scale.view, &weights.index_q_scale.view, &rotary.inv_freq.view, &scratch.q.view, &head.k.view, &head.v.view, &scratch.iq.view, &head.index_keys.view, null, null, null, tiny.attn.eps, head_len + 1, null, 0, 0));

    var contiguous: attention.Cache = try .contiguous(tiny.attn, tiny.capacity);
    defer contiguous.deinit();
    var position_dev: attention.Position = try .init();
    defer position_dev.deinit();
    try position_dev.set(head_len);
    try attention.prep(tiny.attn, weights, rotary, contiguous, scratch, scratch.pa.view, position_dev.device.view, 1, head_len + 1, null);
    const place = storedAt(&head_table, head_len);
    try testing.expect(place != head_len);
    try expectRows(head.index_keys, place, contiguous.index_keys, head_len, 1);
    const width = rowWidth(head.index_keys);
    const raw = try gpa.alloc(u8, width);
    defer gpa.free(raw);
    const zeros = try gpa.alloc(u8, width);
    defer gpa.free(zeros);
    @memset(zeros, 0);
    try cuda.synchronize();
    try head.index_keys.downloadAt(head_len * width, raw);
    try testing.expectEqualSlices(u8, zeros, raw);
}

// One load of the first four layers for both checks below (spec 133 WS-02): its rows the chunks', the
// round's check on it narrowed to the 48 rows it was written for (`checkpoints.narrowed`).
test "shared rounds checkpoint: the first four layers, two streams; and a prompt chunk of 128 rows and of 64 beside a window leaves what halves of it and a row at a time leave" {
    const gpa = testing.allocator;
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    var m: model.Model = try .loadWith(gpa, config, found, .{ .layers = 4, .max_rows = chunks_rows, .mtp = false });
    defer m.deinit();
    try cuda.synchronize();
    try checkpointRound(gpa, "fp4-local", fixtures.checkpoints.narrowed(m, 48));
    try checkpointChunks(gpa, m);
}

fn checkpointRound(gpa: Allocator, label: []const u8, m: model.Model) !void {
    const rows = m.max_rows;
    std.debug.print("shared rounds checkpoint: {s}, four layers\n", .{label});
    const capacity = tiny.capacity;
    var pool: attention.Pool = try placedPool(m.attention.?.shape, scattered_extents);
    defer pool.deinit();
    var slots: [2]Slot = undefined;
    slots[0] = try .open(gpa, m, &.{pool}, null, capacity);
    defer slots[0].deinit();
    slots[1] = try .open(gpa, m, &.{pool}, null, capacity);
    defer slots[1].deinit();
    try slots[0].table.set(&scattered_a);
    try slots[1].table.set(&scattered_b);
    var shared: Shared = try .init(rows, m.shape.delta_net);
    defer shared.deinit();
    var round: Round = try .init(m, 2, rows);
    defer round.deinit();
    var together: forward.Buffers = try .init(gpa, m, rows, capacity);
    defer together.deinit();
    var alone_buf: forward.Buffers = try .init(gpa, m, rows, capacity);
    defer alone_buf.deinit();
    var alone: [2]forward.State = undefined;
    alone[0] = try .init(gpa, m, rows, capacity);
    defer alone[0].deinit();
    alone[1] = try .init(gpa, m, rows, capacity);
    defer alone[1].deinit();
    try placeAt(&slots[0], &alone[0], 2000);
    try placeAt(&slots[1], &alone[1], 2100);
    const segments = [_]Segment{
        .{ .slot = 0, .tokens = &.{ 1, 2, 3 }, .keep = 3 },
        .{ .slot = 1, .tokens = &.{ 4, 5 }, .keep = 2 },
    };
    try expectRound(gpa, m, &slots, &shared, &round, &together, &alone, &alone_buf, &segments, &.{ &scattered_a, &scattered_b });

    // A window beside a prompt chunk on the wide kernels, in buffers that keep fewer rows than a round has, as
    // `serve`'s do: every layer's projection is then written into the one `projection_wide`, and each layer's
    // conv window must still be its own (spec 051; the first four layers hold more DeltaNet layers than one).
    var deltas: usize = 0;
    for (m.layers) |layer| deltas += @intFromBool(layer.block == .delta_net);
    try testing.expect(deltas >= 2);
    var served: forward.Buffers = try .initFor(gpa, m, m.layers, rows, rows - 1, capacity);
    defer served.deinit();
    try testing.expect(served.projection_wide != null);
    var chunk: [40]i32 = undefined;
    for (&chunk, 0..) |*token, i| token.* = @intCast(100 + i * 7);
    try testing.expect(chunk.len >= gdn.wide_rows);
    const beside = [_]Segment{
        .{ .slot = 0, .tokens = &.{ 6, 7, 8 }, .keep = 3 },
        .{ .slot = 1, .tokens = &chunk, .keep = chunk.len },
    };
    try expectRound(gpa, m, &slots, &shared, &round, &served, &alone, &alone_buf, &beside, &.{ &scattered_a, &scattered_b });
    // and the rows after it, which read those windows
    const after = [_]Segment{
        .{ .slot = 0, .tokens = &.{ 9, 10 }, .keep = 2 },
        .{ .slot = 1, .tokens = &.{ 11, 12, 13 }, .keep = 3 },
    };
    try expectRound(gpa, m, &slots, &shared, &round, &served, &alone, &alone_buf, &after, &.{ &scattered_a, &scattered_b });
}

// `serve --prompt-chunk` (spec 074, WS-04): a prompt beside a reply in chunks of 128 rows, of 64 and of fewer
// leaves one state. The chunk of a round against the same rows alone in windows of half its width, and
// against a row at a time; the widths are the flag's on a checkpoint, where the tests above stop at 40.
const chunks_rows = 144;

fn checkpointChunks(gpa: Allocator, m: model.Model) !void {
    const rows = chunks_rows;
    const capacity = tiny.capacity;
    var pool: attention.Pool = try placedPool(m.attention.?.shape, scattered_extents);
    defer pool.deinit();
    var shared: Shared = try .init(rows, m.shape.delta_net);
    defer shared.deinit();
    var round: Round = try .init(m, 2, rows);
    defer round.deinit();
    // buffers that keep fewer rows than a round has, as `serve`'s do
    var together: forward.Buffers = try .initFor(gpa, m, m.layers, rows, rows - 1, capacity);
    defer together.deinit();
    var alone_buf: forward.Buffers = try .init(gpa, m, rows, capacity);
    defer alone_buf.deinit();
    var prompt: [192]i32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(100 + i * 7);
    const start = 2100;
    // the prompt in a round: a chunk of 128 rows and one of 64, each beside a reply's window
    var slots: [2]Slot = undefined;
    slots[0] = try .open(gpa, m, &.{pool}, null, capacity);
    defer slots[0].deinit();
    slots[1] = try .open(gpa, m, &.{pool}, null, capacity);
    defer slots[1].deinit();
    try slots[0].table.set(&scattered_a);
    try slots[1].table.set(&scattered_b);
    slots[0].pos = 2000;
    slots[1].pos = start;
    try run(gpa, m, &slots, &shared, &round, &together, &.{
        .{ .slot = 0, .tokens = &.{ 6, 7, 8 }, .keep = 3 },
        .{ .slot = 1, .tokens = prompt[0..128], .keep = 128 },
    });
    try run(gpa, m, &slots, &shared, &round, &together, &.{
        .{ .slot = 0, .tokens = &.{ 9, 10 }, .keep = 2 },
        .{ .slot = 1, .tokens = prompt[128..], .keep = 64 },
    });
    // alone: in windows of 64, 64, 32 and 32 rows, and a row at a time
    for ([_][]const usize{ &.{ 64, 64, 32, 32 }, &.{} }) |widths| {
        errdefer std.debug.print("alone in windows of {any}\n", .{widths});
        var state: forward.State = try .init(gpa, m, rows, capacity);
        defer state.deinit();
        state.pos = start;
        try state.position.set(start);
        var at: usize = 0;
        var w: usize = 0;
        while (at < prompt.len) : (w += 1) {
            const n = if (widths.len == 0) 1 else widths[w];
            try forward.forward(m, &state, &alone_buf, prompt[at..][0..n]);
            try forward.commit(m, &state, alone_buf, prompt[at..][0..n], n);
            at += n;
        }
        try expectSlot(m, &slots[1], &state, &scattered_b, start, prompt.len);
        // the last row's hidden state and logits: the round's last chunk ends where this window does
        try expectRows(together.h, 2 + 63, alone_buf.h, if (widths.len == 0) 0 else 31, 1);
        try expectRows(together.logits, 2 + 63, alone_buf.logits, if (widths.len == 0) 0 else 31, 1);
    }
}

test "shared rounds: a short pool commits nothing" {
    const gpa = testing.allocator;
    // Five extents, two streams: one holds three and one holds two, so the free list is empty while the
    // second stream still has room in its capacity. `reserve` is what a round calls before `stage`.
    var pool = try extents.Pool.init(gpa, .{ .extents = 5, .capacity = 3 * extents.positions, .slots = 2 });
    defer pool.deinit();
    const a = try pool.open();
    const b = try pool.open();
    try pool.grow(a, 3 * extents.positions);
    try pool.grow(b, 2 * extents.positions);
    try testing.expectEqual(@as(usize, 0), pool.free_count);
    pool.append(a, &.{ 1, 2, 3 });
    pool.append(b, &.{ 4, 5 });
    const a_len = a.len();
    const b_len = b.len();
    const a_table = a.table.items.len;
    const b_table = b.table.items.len;
    const chains = [_]*extents.Chain{ a, b };
    const uptos = [_]usize{ 3 * extents.positions, 2 * extents.positions + 1 };
    try testing.expectError(error.Short, reserve(&pool, &chains, &uptos));
    try testing.expectEqual(a_len, a.len());
    try testing.expectEqual(b_len, b.len());
    try testing.expectEqual(a_table, a.table.items.len);
    try testing.expectEqual(b_table, b.table.items.len);
    try testing.expectEqual(@as(usize, 0), pool.counts().free);
    try testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, a.tokens.items);
    try testing.expectEqualSlices(i32, &.{ 4, 5 }, b.tokens.items);
}

/// The tiny model's two layers, both DFlash2 taps: the first's completed streams are the second's `h_in`, the
/// second's the final streams.
const tap_layers = [_]u32{ 0, 1 };

/// Each tap layer's completed streams as a passive tap on a stream alone hands them, `rows` rows of `width`
/// (`dflash2/ranks_loopback_test.zig`'s).
const Streams = struct {
    layers: usize,
    seen: [tap_layers.len][]u16,
    width: usize,
    rows: usize,

    fn tap(self: *Streams) forward.Tap {
        return .{ .context = self, .at = at, .passive = true };
    }

    fn at(context: *anyopaque, point: forward.Point, layer: usize, view: cuda.View) anyerror!void {
        const self: *Streams = @ptrCast(@alignCast(context));
        for (tap_layers, 0..) |one, slot| {
            const after = one + 1;
            const here = if (after < self.layers) point == .h_in and layer == after else point == .final_h and layer == self.layers;
            if (here) try cuda.download(view, std.mem.sliceAsBytes(self.seen[slot][0 .. self.rows * self.width]));
        }
    }
};

/// After a round over `together`, whose taps are set: each segment's stream alone on `alone`, whose are not,
/// gives the round's streams and logits to the bit, so the capture moved none of them, and each of its rows'
/// taps are the host mean of that stream's streams after each tap layer.
fn expectTaps(gpa: Allocator, m: model.Model, together: *const forward.Buffers, alone_state: []forward.State, alone: *forward.Buffers, segments: []const Segment) !void {
    std.debug.assert(alone.taps == null);
    const hidden = m.shape.hidden;
    const nstreams = m.shape.streams;
    const width = tap_layers.len * hidden;
    var seen: [tap_layers.len][]u16 = undefined;
    for (&seen) |*one| one.* = try gpa.alloc(u16, tiny.max_rows * nstreams * hidden);
    defer for (seen) |one| gpa.free(one);
    const mean = try gpa.alloc(u16, tiny.max_rows * hidden);
    defer gpa.free(mean);
    const want = try gpa.alloc(u16, tiny.max_rows * width);
    defer gpa.free(want);
    const got = try gpa.alloc(u16, tiny.max_rows * width);
    defer gpa.free(got);
    var at: usize = 0;
    for (segments) |segment| {
        const n = segment.tokens.len;
        var streams: Streams = .{ .layers = m.layers.len, .seen = seen, .width = nstreams * hidden, .rows = n };
        const state = &alone_state[segment.slot];
        try forward.forwardTapped(m, state, alone, segment.tokens, streams.tap());
        try forward.commit(m, state, alone.*, segment.tokens, segment.keep);
        try expectRows(together.h, at, alone.h, 0, n);
        try expectRows(together.logits, at, alone.logits, 0, n);
        for (seen, 0..) |one, slot| {
            taps_file.streamMeanRows(one[0 .. n * nstreams * hidden], mean[0 .. n * hidden], n, nstreams, together.tap_mean);
            for (0..n) |r| @memcpy(want[r * width + slot * hidden ..][0..hidden], mean[r * hidden ..][0..hidden]);
        }
        try testing.expect(std.mem.indexOfNone(u16, want[0 .. n * width], &.{ 0, 0x8000 }) != null);
        try cuda.download(together.taps.?.span(at, n), std.mem.sliceAsBytes(got[0 .. n * width]));
        try testing.expectEqualSlices(u16, want[0 .. n * width], got[0 .. n * width]);
        at += n;
    }
}

test "shared rounds: each stream's DFlash2 taps are its streams' mean after the tap layers, one stream and several, graphed or not" {
    // spec 126 WS-05: `rounds.compute` wrote no taps, while the hall's DFlash2 seats read them
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try tinyModel(gpa, arena_state.allocator());
    defer m.deinit();
    try testing.expectEqual(tap_layers.len, m.layers.len);
    var alone_buf: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
    defer alone_buf.deinit();

    // one stream: a prompt's rows, then a reply's row
    {
        var pool: attention.Pool = try placedPool(m.attention.?.shape, 1);
        defer pool.deinit();
        var slots: [1]Slot = undefined;
        slots[0] = try .open(gpa, m, &.{pool}, null, tiny.capacity);
        defer slots[0].deinit();
        try slots[0].table.set(&identity_table);
        var alone: [1]forward.State = undefined;
        alone[0] = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer alone[0].deinit();
        var shared: Shared = try .init(tiny.max_rows, m.shape.delta_net);
        defer shared.deinit();
        var round: Round = try .init(m, 1, tiny.max_rows);
        defer round.deinit();
        var together: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer together.deinit();
        try together.setTaps(&tap_layers, .f32_acc);
        const steps = [_][]const Segment{
            &.{.{ .slot = 0, .tokens = &.{ 1, 2, 3, 4, 5 }, .keep = 5 }},
            &.{.{ .slot = 0, .tokens = &.{6}, .keep = 1 }},
        };
        for (steps) |segments| {
            try together.taps.?.zero();
            try run(gpa, m, &slots, &shared, &round, &together, segments);
            try expectTaps(gpa, m, &together, &alone, &alone_buf, segments);
        }
    }

    // three: two replies' windows and a prompt chunk of the wide path beside them
    {
        var wide_tokens: [32]i32 = undefined;
        for (&wide_tokens, 0..) |*token, i| token.* = @intCast(i + 1);
        const replies = [_]Segment{
            .{ .slot = 1, .tokens = &.{ 41, 43 }, .keep = 2 },
            .{ .slot = 2, .tokens = &.{42}, .keep = 1 },
        };
        const segments = try promptBeside(gpa, &replies, 0, &wide_tokens, prompt_chunk);
        defer gpa.free(segments);
        var pool: attention.Pool = try placedPool(m.attention.?.shape, 3);
        defer pool.deinit();
        var slots: [3]Slot = undefined;
        var alone: [3]forward.State = undefined;
        const tables = [_][]const i32{ &.{0}, &.{1}, &.{2} };
        for (&slots, &alone, tables, [_]usize{ 0, 64, 96 }) |*slot, *state, table, pos| {
            slot.* = try .open(gpa, m, &.{pool}, null, tiny.capacity);
            state.* = try .init(gpa, m, tiny.max_rows, tiny.capacity);
            try slot.table.set(table);
            try placeAt(slot, state, pos);
        }
        defer for (&slots) |*slot| slot.deinit();
        defer for (&alone) |*state| state.deinit();
        var shared: Shared = try .init(tiny.max_rows, m.shape.delta_net);
        defer shared.deinit();
        var round: Round = try .init(m, 3, tiny.max_rows);
        defer round.deinit();
        var together: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer together.deinit();
        try together.setTaps(&tap_layers, .f32_acc);
        try together.taps.?.zero();
        try run(gpa, m, &slots, &shared, &round, &together, segments);
        try expectTaps(gpa, m, &together, &alone, &alone_buf, segments);
    }

    // two of unequal widths, the second round from a graph captured after the taps were set
    {
        var pools: [2]attention.Pool = undefined;
        for (&pools) |*pool| pool.* = try placedPool(m.attention.?.shape, 6);
        defer for (&pools) |*pool| pool.deinit();
        var slots: [2]Slot = undefined;
        var alone: [2]forward.State = undefined;
        for (&slots, &alone, &pools) |*slot, *state, pool| {
            slot.* = try .open(gpa, m, &.{pool}, null, tiny.capacity);
            state.* = try .init(gpa, m, tiny.max_rows, tiny.capacity);
            try slot.table.set(&identity_table);
        }
        defer for (&slots) |*slot| slot.deinit();
        defer for (&alone) |*state| state.deinit();
        var shared: Shared = try .init(tiny.max_rows, m.shape.delta_net);
        defer shared.deinit();
        var round: Round = try .init(m, 2, tiny.max_rows);
        defer round.deinit();
        var together: forward.Buffers = try .init(gpa, m, tiny.max_rows, tiny.capacity);
        defer together.deinit();
        try together.setTaps(&tap_layers, .f32_acc);
        var graphs: Graphs = try .init(gpa, m, 2, tiny.max_rows, tiny.capacity);
        defer graphs.deinit();
        const prefix = [_]Segment{
            .{ .slot = 0, .tokens = &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, .keep = 8 },
            .{ .slot = 1, .tokens = &.{ 11, 12, 13, 14, 15 }, .keep = 5 },
        };
        try together.taps.?.zero();
        try run(gpa, m, &slots, &shared, &round, &together, &prefix);
        try expectTaps(gpa, m, &together, &alone, &alone_buf, &prefix);
        const compared = [_]Segment{
            .{ .slot = 0, .tokens = &.{ 21, 22, 23 }, .keep = 3 },
            .{ .slot = 1, .tokens = &.{ 31, 32 }, .keep = 2 },
        };
        const rows: usize = 5;
        const context = forward.Graphs.bucket(together, 11);
        try graphs.prepare(m, &slots, &shared, &together, &round, rows, context, null);
        try together.taps.?.zero();
        try stage(gpa, m, &slots, &round, &together, &compared);
        try graphs.launch(rows, context);
        try commit(m, &slots, &together, &round, &compared);
        try expectTaps(gpa, m, &together, &alone, &alone_buf, &compared);
    }
}
