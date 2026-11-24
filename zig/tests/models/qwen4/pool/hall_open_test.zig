//! `hall_open.zig`'s tests: the extents' bytes and the rest's, said before a pool is made, against what its
//! opening asks of the device, and what `serve`'s load says fits.

const std = @import("std");
const fixtures = @import("../testing.zig");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_seats = @import("../../../../src/models/qwen4/pool/hall_seats.zig");
const hall_open = @import("../../../../src/models/qwen4/pool/hall_open.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const gdn = @import("../../../../src/models/qwen4/blocks/gdn.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const mtp = @import("../../../../src/models/qwen4/mtp.zig");
const rounds = @import("../../../../src/models/qwen4/pool/rounds.zig");
const Engine = hall_engine.Engine;
const Fit = hall_open.Fit;
const Rest = hall_open.Rest;
const extentsBytes = hall_open.extentsBytes;

/// The buffers' account when `openAfter` has made the extents and nothing else.
const Seen = struct {
    bytes: usize = 0,
    places: usize = 0,

    fn see(context: *anyopaque, rooms: hall_engine.Rooms) void {
        const self: *Seen = @ptrCast(@alignCast(context));
        self.bytes = cuda.account.bytes;
        self.places = rooms.pool.refs.len;
    }
};

/// Opens a pool of `slots` streams of `context` tokens over `m` and holds `extentsBytes` to what the
/// opening asked of the device for its extents, and `Rest` to what it asked after them, both to the byte
/// by the buffers' account. Returns the extents' bytes.
fn expectExtents(m: model_file.Model, context: usize, slots: usize) !usize {
    return expectOpened(m, context, slots, true);
}

fn expectOpened(m: model_file.Model, context: usize, slots: usize, lazy: bool) !usize {
    const gpa = testing.allocator;
    const pool = hall_file.poolFor(context, slots);
    const before = cuda.account.bytes;
    var seen: Seen = .{};
    var engine = try Engine.openAfter(gpa, m, &.{}, .{
        .slots = slots,
        .session = .{ .capacity = context, .prompt_rows = 32 },
        .pool = pool,
        .lazy_graphs = lazy,
    }, .{ .context = &seen, .call = Seen.see });
    defer engine.deinit();
    try testing.expectEqual(pool / extents.positions, seen.places);
    try testing.expectEqual(seen.bytes - before, extentsBytes(m, seen.places));
    // the rest, whole: a capture of the first graphs makes no buffer either
    const rest: Rest = .of(m, slots, context);
    try testing.expectEqual(cuda.account.bytes - seen.bytes, rest.bytes());
    try testing.expect(rest.slots > 0 and rest.windows > 0 and (rest.head > 0) == (m.mtp != null));
    return seen.bytes - before;
}

/// Each allocator of the rest alone against the `bytesFor` beside it, so that no two err against each
/// other inside `Rest`'s sum: a round's tables, a layer's replay, the forward's buffers, the graphs'
/// regions, a slot, and the head's buffers lent the trunk's attention scratch or with their own.
fn expectParts(m: model_file.Model, context: usize, slots: usize) !void {
    const gpa = testing.allocator;
    const rows = m.max_rows;
    const scored = @min(rows, slots * 17 + 1);
    var at = cuda.account.bytes;
    var round: rounds.Round = try .init(m, slots, rows);
    defer round.deinit();
    try testing.expectEqual(cuda.account.bytes - at, rounds.Round.bytesFor(m, slots, rows));
    at = cuda.account.bytes;
    var replay: gdn.Scratch = try .init(rows, m.shape.delta_net);
    defer replay.deinit();
    try testing.expectEqual(cuda.account.bytes - at, gdn.Scratch.bytesFor(rows, m.shape.delta_net));
    at = cuda.account.bytes;
    var buffers: forward.Buffers = try .initFor(gpa, m, m.layers, rows, scored, context);
    defer buffers.deinit();
    try testing.expectEqual(cuda.account.bytes - at, forward.Buffers.bytesFor(m, m.layers, rows, scored, context, false));
    at = cuda.account.bytes;
    var graphs: rounds.Graphs = try .init(gpa, m, slots, rows, context);
    defer graphs.deinit();
    try testing.expectEqual(cuda.account.bytes - at, rounds.Graphs.bytesFor(m, slots, rows, context));

    var n_attn: usize = 0;
    for (m.layers) |layer| n_attn += @intFromBool(layer.block == .attention);
    const trunk = try gpa.alloc(attention.Pool, n_attn);
    defer gpa.free(trunk);
    var made: usize = 0;
    defer for (trunk[0..made]) |*pool| pool.deinit();
    for (trunk) |*pool| {
        pool.* = try .init(m.attention.?.shape, 1);
        made += 1;
    }
    var head_pool: ?attention.Pool = if (m.mtp != null) try .init(m.attention.?.headShape(), 1) else null;
    defer if (head_pool) |*pool| pool.deinit();
    at = cuda.account.bytes;
    var slot: rounds.Slot = try .open(gpa, m, trunk, if (head_pool) |*pool| pool else null, context);
    defer slot.deinit();
    try testing.expectEqual(cuda.account.bytes - at, rounds.Slot.bytesFor(m, context, m.mtp != null));

    if (m.mtp == null) return;
    at = cuda.account.bytes;
    var lent: mtp.Buffers = try .initBeside(gpa, m, rows, context, &buffers);
    defer lent.deinit();
    try testing.expectEqual(cuda.account.bytes - at, mtp.Buffers.bytesFor(m, rows, context, true));
    at = cuda.account.bytes;
    var own: mtp.Buffers = try .init(gpa, m, rows, context);
    defer own.deinit();
    try testing.expectEqual(cuda.account.bytes - at, mtp.Buffers.bytesFor(m, rows, context, false));
    at = cuda.account.bytes;
    var head_round: hall_seats.HeadRound = try .init(slots, rows);
    defer head_round.deinit();
    try testing.expectEqual(cuda.account.bytes - at, hall_seats.HeadRound.bytesFor(slots, rows));
}

test "the extents' bytes said before a pool is made are what its opening asks of the device: the head's pool in each storage, a trunk layer's, by the extents" {
    const gpa = testing.allocator;
    defer fixtures.random_model.caches = .bf16;
    defer fixtures.random_model.head_caches = null;
    const Case = struct { kv: attention.Storage, head: ?attention.Storage };
    // the head's pool alone (the random trunk does not attend), in its storage
    var by_storage: [3]usize = undefined;
    for ([_]Case{
        .{ .kv = .bf16, .head = null },
        .{ .kv = .fp8, .head = null },
        .{ .kv = .fp8_nvfp4, .head = null },
        .{ .kv = .fp8, .head = .bf16 },
        .{ .kv = .fp8_nvfp4, .head = .bf16 },
    }, 0..) |case, i| {
        fixtures.random_model.caches = case.kv;
        fixtures.random_model.head_caches = case.head;
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
        defer m.deinit();
        const small = try expectExtents(m, 256, 2);
        const large = try expectExtents(m, 1536, 4);
        try expectParts(m, 1536, 4);
        // linear in the extents: one extent's bytes say any pool's
        try testing.expectEqual(small, 3 * extentsBytes(m, 1));
        try testing.expectEqual(large, 12 * extentsBytes(m, 1));
        try testing.expect(small > 0);
        if (i < by_storage.len) by_storage[i] = small else try testing.expectEqual(by_storage[0], small);
    }
    // the storages differ, so a figure taken for the wrong one would have failed above
    try testing.expect(by_storage[1] < by_storage[0] and by_storage[2] < by_storage[1]);

    // a trunk attention layer's pool, no head
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var tiny = try fixtures.tiny_model.tinyModel(gpa, arena_state.allocator());
    defer tiny.deinit();
    const trunk = try expectExtents(tiny, fixtures.tiny_model.tiny.capacity, 2);
    try testing.expectEqual(trunk, attention.Pool.bytesFor(fixtures.tiny_model.tiny.attn, extents.wanted(fixtures.tiny_model.tiny.capacity, 2)));
    try expectParts(tiny, fixtures.tiny_model.tiny.capacity, 2);
}

test "the rest of a pool said before it is made is what its opening asks after the extents: one stream and several, a draft head, a context the indexer selects in, the first graphs captured" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    // rows past the short projection's and the small plan's, so the wide buffers are in it
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), &.{ 3, 5, 8, 13, 21, 34 }, 160);
    defer m.deinit();
    try testing.expect(m.mtp.?.draft != null);
    for ([_][2]usize{ .{ 256, 1 }, .{ 512, 2 }, .{ 8192, 3 }, .{ 1024, 8 } }) |case| {
        errdefer std.debug.print("context {d}, {d} streams\n", .{ case[0], case[1] });
        _ = try expectOpened(m, case[0], case[1], true);
        try expectParts(m, case[0], case[1]);
    }
    _ = try expectOpened(m, 512, 2, false);
    // by part: the slots' with the streams, the windows' with the context where the indexer scores its blocks
    const two: Rest = .of(m, 2, 8192);
    const four: Rest = .of(m, 4, 8192);
    try testing.expectEqual(2 * two.slots, four.slots);
    try testing.expect(four.windows > two.windows and four.head > two.head);
    try testing.expect(Rest.of(m, 2, 16384).windows > two.windows);

    var tiny = try fixtures.tiny_model.tinyModel(gpa, arena_state.allocator());
    defer tiny.deinit();
    _ = try expectOpened(tiny, fixtures.tiny_model.tiny.capacity, 3, false);
}

test "a pool's rounds made for fewer rows than the model's: its rest is a model's of those rows, and its rounds are that wide (spec 105 WS-07)" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    // `serve --max-num-batched-tokens` loads the model for a lone prompt's wider windows; the pool keeps its own
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), &.{ 3, 5, 8, 13, 21, 34 }, 160);
    defer m.deinit();
    var narrow = m;
    narrow.max_rows = 96;
    const context = 1024;
    const pool = hall_file.poolFor(context, 2);
    var seen: Seen = .{};
    var engine = try Engine.openAfter(gpa, m, &.{}, .{
        .slots = 2,
        .session = .{ .capacity = context, .prompt_rows = 160 },
        .pool = pool,
        .chunk = 160,
        .rows = narrow.max_rows,
        .lazy_graphs = true,
    }, .{ .context = &seen, .call = Seen.see });
    defer engine.deinit();
    try testing.expectEqual(cuda.account.bytes - seen.bytes, Rest.of(narrow, 2, context).bytes());
    try testing.expect(Rest.of(narrow, 2, context).bytes() < Rest.of(m, 2, context).bytes());
    try testing.expectEqual(narrow.max_rows, engine.m.max_rows);
    try testing.expectEqual(narrow.max_rows, engine.prompt_rows);
    try testing.expectEqual(narrow.max_rows, engine.chunk);
}

test "a checkpoint's first layers: the rest of a pool said before it is made is what its opening asks, with the PLE, an attention layer and the head" {
    const layout = @import("../../../../src/models/qwen4/layout.zig");
    const gpa = testing.allocator;
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    const layers = fixtures.checkpoints.attentionLayers(found) orelse return error.SkipZigTest;
    // the storages `serve` gives four streams, and rows past the prompt window
    const caches = layout.storageFor(config, 4, 1, null);
    var m: model_file.Model = try .loadWith(gpa, config, found, .{ .layers = layers, .max_rows = 600, .draft_vocab = try model_file.DraftHead.defaultIds(arena), .kv = caches.trunk, .head_kv = caches.head });
    defer m.deinit();
    for ([_][2]usize{ .{ 2048, 2 }, .{ 32768, 4 } }) |case| {
        _ = try expectOpened(m, case[0], case[1], true);
        try expectParts(m, case[0], case[1]);
    }
}

test "what serve's load says of a pool: its extents fit what is left less the reserve, and what is said to fit instead does" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    const each = extentsBytes(m, 1);
    const context: usize = 262144;
    // A session larger than the rest gives back more than the rest takes, so the extents alone decide: the
    // cases of spec 082, as they were.
    const session = Rest.of(m, 8, context).bytes();
    // the defaults' pool to the byte: it fits what is left when that is its bytes and the reserve
    const asked = Fit.of(m, context, 4, 2048 * each + cuda.reserve, session, 0);
    try testing.expectEqual(@as(usize, 1_048_576), asked.positions);
    try testing.expectEqual(2048 * each, asked.bytes);
    try testing.expect(asked.fits and asked.extents_fit);
    try testing.expectEqual(@as(usize, 4), asked.parallel);
    try testing.expectEqual(context, asked.context);
    // one byte less and it does not; three streams of this context do, or four of the context below
    const short = Fit.of(m, context, 4, 2048 * each + cuda.reserve - 1, session, 0);
    try testing.expect(!short.fits and !short.extents_fit);
    try testing.expectEqual(@as(usize, 3), short.parallel);
    try testing.expectEqual(context - extents.positions, short.context);
    try testing.expect(Fit.of(m, context, short.parallel, short.left, session, 0).fits);
    try testing.expect(Fit.of(m, short.context, 4, short.left, session, 0).fits);
    // the owner's case: eight streams of the default context beside what four of them fit
    const eight = Fit.of(m, context, 8, 2048 * each + cuda.reserve, session, 0);
    try testing.expectEqual(@as(usize, 2_097_152), eight.positions);
    try testing.expect(!eight.fits);
    try testing.expectEqual(@as(usize, 4), eight.parallel);
    try testing.expectEqual(@as(usize, 131072), eight.context);
    // nothing left: the session alone, and no context
    const none = Fit.of(m, context, 4, cuda.reserve, session, 0);
    try testing.expect(!none.fits);
    try testing.expectEqual(@as(usize, 1), none.parallel);
    try testing.expectEqual(@as(usize, 0), none.context);
}

test "what serve's load says of a pool whose extents fit: the rest must fit what they leave with the session's bytes given back, and what is said to fit instead does" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    const context: usize = 8192;
    const pool = extentsBytes(m, extents.wanted(context, 4));
    const rest = Rest.of(m, 4, context).bytes();
    try testing.expect(rest > 0);
    // no session to give back: the extents and the rest, each with the reserve kept, to the byte
    const both = Fit.of(m, context, 4, pool + rest + cuda.reserve, 0, 0);
    try testing.expect(both.fits and both.extents_fit);
    try testing.expectEqual(rest, both.rest.bytes());
    try testing.expectEqual(@as(usize, 4), both.parallel);
    try testing.expect(both.context >= context);
    // one byte less: the extents fit and the rest does not, which the extents' rule alone passed
    const short = Fit.of(m, context, 4, pool + rest + cuda.reserve - 1, 0, 0);
    try testing.expect(!short.fits and short.extents_fit);
    try testing.expect(short.parallel < 4 and short.context < context);
    // a session of half the rest gives that much back, and no more
    const half = rest / 2;
    try testing.expect(Fit.of(m, context, 4, pool + (rest - half) + cuda.reserve, half, 0).fits);
    const held = Fit.of(m, context, 4, pool + (rest - half) + cuda.reserve - 1, half, 0);
    try testing.expect(!held.fits and held.extents_fit);
    // a session of the rest or more: the extents alone decide
    try testing.expect(Fit.of(m, context, 4, pool + cuda.reserve, rest, 0).fits);
    try testing.expect(!Fit.of(m, context, 4, pool + cuda.reserve - 1, 1 << 40, 0).extents_fit);
    // What a refusal says would fit does, and the next above it does not. Over a range of what is left,
    // from the session alone to the pool a byte short.
    var left = cuda.reserve;
    while (left < pool + (rest - half) + cuda.reserve) : (left += (pool + rest) / 37 + 1) {
        const fit = Fit.of(m, context, 4, left, half, 0);
        try testing.expect(!fit.fits);
        if (fit.parallel > 1) try testing.expect(Fit.of(m, context, fit.parallel, left, half, 0).fits);
        try testing.expect(fit.parallel < 4 and !Fit.of(m, context, fit.parallel + 1, left, half, 0).fits);
        if (fit.context > 0) try testing.expect(Fit.of(m, fit.context, 4, left, half, 0).fits);
        try testing.expect(fit.context < context and !Fit.of(m, fit.context + extents.positions, 4, left, half, 0).fits);
    }
}

test "a pool's short projection holds its widest round without a wide segment: the other slots' full windows and a prompt chunk one row short of the wide kernels" {
    const window = @import("../../../../src/decode/masks.zig").max_rows;
    const shortRows = hall_open.shortRows;
    for (1..9) |slots| {
        const short = shortRows(576, slots);
        // spec 086: three windows and 31 rows of a prompt were 79 rows in buffers of 65
        try testing.expect(short >= (slots - 1) * window + gdn.wide_rows - 1);
        // every slot's window, and the row a prompt's last logits take beside them
        try testing.expect(short >= slots * window + 1);
        try testing.expect(short < 576);
    }
    try testing.expectEqual(@as(usize, 79), shortRows(576, 4));
    try testing.expectEqual(@as(usize, 40), shortRows(40, 4));

    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 128);
    defer m.deinit();
    var engine = try Engine.open(gpa, m, &.{}, .{
        .slots = 4,
        .session = .{ .capacity = 256, .prompt_rows = 32 },
        .pool = hall_file.poolFor(256, 4),
        .lazy_graphs = true,
    });
    defer engine.deinit();
    try testing.expectEqual(shortRows(128, 4), engine.buffers.kept);
    for (engine.buffers.projections) |projection| try testing.expectEqual(@as(i64, @intCast(shortRows(128, 4))), projection.view.size[0]);
    try testing.expect(engine.buffers.projections.len > 0 and engine.buffers.projection_wide != null);
}
