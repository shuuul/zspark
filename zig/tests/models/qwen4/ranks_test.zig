//! Two ranks of Flash Next in one process (spec 050). The random model is sliced once and held twice: either
//! rank's streams match, a window matches its serial steps, and a drafted reply matches those steps. The MTP head
//! whole beside rank 0's share (spec 090) drafts a reply that is the two ranks' serial one; on a checkpoint's
//! first layers, `mtp_ranks_test.zig`'s load of both head forms (`ranks_fixture.wholeHeadDrafted`, spec 133
//! WS-08). A shelf's states (spec 094) are a cold run's, inside a window too. The gate is
//! `ranks_gate_test.zig`'s.

const std = @import("std");
const testing = std.testing;
const fixtures = @import("testing.zig");
const cuda = @import("../../../src/cuda/cuda.zig");
const draft = @import("../../../src/decode/draft.zig");
const picking = @import("../../../src/decode/pick.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const attention = @import("../../../src/models/qwen4/blocks/attention.zig");
const cache = @import("../../../src/models/qwen4/blocks/attention_cache.zig");
const forward = @import("../../../src/models/qwen4/forward.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const mtp_file = @import("../../../src/models/qwen4/mtp.zig");
const ranks = @import("../../../src/models/qwen4/ranks_forward.zig");
const partials = @import("../../../src/models/qwen4/partials.zig");
const ranks_file = @import("../../../src/ranks/ranks.zig");
const random = @import("../../../src/models/qwen4/checks/random.zig");
const ranks_check = @import("../../../src/models/qwen4/checks/ranks_check.zig");
const pair_file = @import("../../../src/models/qwen4/session/pair.zig");
const book_file = @import("../../../src/decode/book.zig");

const Allocator = std.mem.Allocator;
const Pair = ranks.Held(2);
const Both = pair_file.Of(2);

const ns = struct {
    pub const Model = Pair.Model;
    pub const State = Pair.State;
    pub const Buffers = Pair.Buffers;
    pub const Graphs = Pair.Graphs;
    pub const Error = Pair.Error;
    pub const forward = Pair.forward;
    pub const forwardScoring = Pair.forwardScoring;
    pub const commit = Pair.commit;
    pub const logitsRow = Pair.logitsRow;
    pub const streams = Pair.streams;
};

const Decoder = draft.Decoder(ns);
const storages = [_]attention.Storage{ .bf16, .fp8, .fp8_nvfp4 };
const rows_cap = 4;

fn argmax(bits: []const u16) i32 {
    var best: usize = 0;
    var value: f32 = -std.math.inf(f32);
    for (bits, 0..) |one, id| {
        const v = numeric.f32FromBf16(one);
        if (v > value) {
            value = v;
            best = id;
        }
    }
    return @intCast(best);
}

fn rowBytes(buffer: cuda.Buffer) usize {
    return buffer.bytes() / @as(usize, @intCast(buffer.view.size[0]));
}

fn downloadRow(buffer: cuda.Buffer, row: usize, dest: []u8) !void {
    const width = rowBytes(buffer);
    try buffer.downloadAt(row * width, dest[0..width]);
}

fn expectBuffer(gpa: Allocator, a: cuda.Buffer, b: cuda.Buffer) !void {
    const left = try gpa.alloc(u8, a.bytes());
    defer gpa.free(left);
    const right = try gpa.alloc(u8, b.bytes());
    defer gpa.free(right);
    try a.download(left);
    try b.download(right);
    try testing.expectEqualSlices(u8, left, right);
}

fn expectState(gpa: Allocator, a: forward.State, b: forward.State) !void {
    try testing.expectEqual(a.pos, b.pos);
    // Which side is current, and how many rows it is behind by, depend on the windows committed (a window of
    // drafts kept whole leaves its rows behind, spec 096); the state at `pos` (`forward.committed`) does not.
    for (a.recurrent, b.recurrent, 0..) |left, right, linear| {
        var at_a = try forward.Snapshot.likeOf(left[0]);
        defer at_a.deinit();
        var at_b = try forward.Snapshot.likeOf(right[0]);
        defer at_b.deinit();
        try forward.committed(a, linear, at_a);
        try forward.committed(b, linear, at_b);
        try cuda.synchronize();
        try expectBuffer(gpa, at_a, at_b);
    }
    for (a.conv, b.conv) |left, right| try expectBuffer(gpa, left, right);
}

test "ranks-check plan: launched and through graphs, in loopback" {
    try ranks_check.loopback(testing.allocator, null);
}

test "ranks-check plan over an attention layer, its caches in each storage: launched and through graphs, in loopback" {
    for (storages) |kv| try ranks_check.loopback(testing.allocator, kv);
}

test "a rank's pool bytes a position are its kv heads, the index keys whole" {
    var whole: attention.Shape = .{ .dims = 256, .heads = 4, .kv_heads = 2, .head_dim = 32, .index_heads = 2, .index_dim = 64, .rotary_dim = 16, .rope_theta = 1e7, .budget = 2048, .ratio = 4, .eps = 1e-6 };
    var half = whole;
    half.heads = 2;
    half.kv_heads = 1;
    const index = whole.index_dim * @sizeOf(u16);
    var last: usize = std.math.maxInt(usize);
    // a coded cache's scales are a head's too: what is not the index keys halves in every storage
    for (storages) |kv| {
        whole.kv = kv;
        half.kv = kv;
        const full = cache.Pool.positionBytes(whole);
        const rank = cache.Pool.positionBytes(half);
        try testing.expect(rank < full and rank < last);
        try testing.expectEqual(full - index, 2 * (rank - index));
        last = rank;
    }
}

/// What a run over the random model with an attention layer left: every row's logits, and each rank's
/// committed cache rows as stored.
const Attended = struct {
    logits: []u16,
    rows: [2][]u8,

    fn deinit(self: Attended, gpa: Allocator) void {
        gpa.free(self.logits);
        for (self.rows) |one| gpa.free(one);
    }
};

fn appendRows(gpa: Allocator, out: *std.ArrayList(u8), view: cuda.View) !void {
    try cuda.download(view, try out.addManyAsSlice(gpa, view.count() * view.dtype.size()));
}

/// `tokens` through both shares in windows of `step`, each row's hidden state the same on either rank.
fn attend(gpa: Allocator, held: Pair.Model, tokens: []const i32, step: usize) !Attended {
    const vocab = random.fx.shape.vocab;
    var state: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer state.deinit();
    var buffers: Pair.Buffers = try .init(gpa, held, rows_cap, 64);
    defer buffers.deinit();
    const logits = try gpa.alloc(u16, tokens.len * vocab);
    errdefer gpa.free(logits);
    const h_row = rowBytes(buffers.ranks[0].h);
    var at: usize = 0;
    while (at < tokens.len) : (at += step) {
        const window = tokens[at..@min(at + step, tokens.len)];
        try Pair.forward(held, &state, &buffers, window);
        for (0..window.len) |row| {
            var mine: [2048]u8 = undefined;
            var other: [2048]u8 = undefined;
            try downloadRow(buffers.ranks[0].h, row, mine[0..h_row]);
            try downloadRow(buffers.ranks[1].h, row, other[0..h_row]);
            try testing.expectEqualSlices(u8, mine[0..h_row], other[0..h_row]);
            try Pair.logitsRow(buffers, row, logits[(at + row) * vocab ..][0..vocab]);
        }
        try Pair.commit(held, &state, buffers, window, window.len);
    }
    try cuda.synchronize();
    var rows: [2]std.ArrayList(u8) = .{ .empty, .empty };
    errdefer for (&rows) |*one| one.deinit(gpa);
    for (&rows, state.ranks) |*out, rank| {
        try testing.expectEqual(tokens.len, rank.pos);
        for (rank.caches) |one| {
            const kv = held.ranks[0].attention.?.shape.kv;
            try testing.expectEqual(kv.dtype(), one.k.view.dtype);
            try testing.expectEqual(kv.scaled(), one.k_scale != null and one.v_scale != null);
            try appendRows(gpa, out, one.k.rows(tokens.len));
            try appendRows(gpa, out, one.v.rows(tokens.len));
            if (one.k_scale) |scale| try appendRows(gpa, out, scale.rows(tokens.len));
            if (one.v_scale) |scale| try appendRows(gpa, out, scale.rows(tokens.len));
            try appendRows(gpa, out, one.index_keys.rows(tokens.len));
        }
    }
    return .{ .logits = logits, .rows = .{ try rows[0].toOwnedSlice(gpa), try rows[1].toOwnedSlice(gpa) } };
}

test "both shares of the random model over an attention layer, its caches in each storage: a window is its serial steps, logits and the rows as stored" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(94);
    // past the attention's budget of 16 positions, so the indexer picks the blocks
    var tokens: [41]i32 = undefined;
    for (&tokens) |*id| id.* = prng.random().intRangeLessThan(i32, 1, 300);
    var first: ?[]u16 = null;
    defer if (first) |kept| gpa.free(kept);
    var bytes: [storages.len]usize = undefined;
    for (storages, &bytes) |kv, *stored| {
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const built = try random.sharesOf(2, gpa, arena_state.allocator(), 42, 0, rows_cap, .{ .caches = kv });
        var held: Pair.Model = .{ .ranks = built, .max_rows = rows_cap };
        defer held.deinit();
        try testing.expectEqual(random.fx.layers + 1, held.ranks[0].layers.len);
        const serial = try attend(gpa, held, &tokens, 1);
        defer serial.deinit(gpa);
        for ([_]usize{ 4, 3 }) |step| {
            const windowed = try attend(gpa, held, &tokens, step);
            defer windowed.deinit(gpa);
            try testing.expectEqualSlices(u16, serial.logits, windowed.logits);
            for (serial.rows, windowed.rows) |want, got| try testing.expectEqualSlices(u8, want, got);
        }
        stored.* = serial.rows[0].len;
        // the storage is what the layer reads: a coded cache's logits are not bf16's
        if (first) |plain| {
            try testing.expect(!std.mem.eql(u16, plain, serial.logits));
        } else first = try gpa.dupe(u16, serial.logits);
    }
    try testing.expect(bytes[1] < bytes[0] and bytes[2] < bytes[1]);
}

test "a split window's rank-order sum of its rows is the whole sum's, -0 and every binade alike" {
    const gpa = testing.allocator;
    const rows = 37;
    const dims = 2560;
    const values = try gpa.alloc(f32, 2 * rows * dims);
    defer gpa.free(values);
    var prng: std.Random.DefaultPrng = .init(1060);
    for (values, 0..) |*v, i| {
        const r = prng.random();
        v.* = switch (i % 7) {
            0 => -0.0,
            1 => 0.0,
            else => (r.float(f32) - 0.5) * std.math.pow(f32, 2.0, @floatFromInt(r.intRangeLessThan(i32, -20, 20))),
        };
    }
    var gathered: cuda.Buffer = try .init(.f32, &.{ 2, rows, dims });
    defer gathered.deinit();
    try gathered.upload(std.mem.sliceAsBytes(values));
    for ([_]cuda.DType{ .bf16, .f32 }) |dtype| {
        var whole: cuda.Buffer = try .init(dtype, &.{ rows, dims });
        defer whole.deinit();
        var split: cuda.Buffer = try .init(dtype, &.{ rows, dims });
        defer split.deinit();
        try split.zero();
        if (dtype == .bf16) try partials.roundSum(gathered.view, whole.view) else try partials.add(gathered.view, whole.view);
        for ([_][2]usize{ .{ 0, 32 }, .{ 32, 37 } }) |own| {
            try ranks_file.sumRows(gathered.view, own[0], split.span(own[0], own[1] - own[0]));
        }
        try cuda.synchronize();
        try expectBuffer(gpa, whole, split);
    }
}

/// What a run of windows leaves on both ranks, for the split glue's test: each window's streams, mixed rows and
/// logits, both ranks', then the caches' rows.
const Left = struct {
    rows: std.ArrayList(u8) = .empty,
    /// Where the caches' rows start in `rows`, after the windows'.
    caches: usize = 0,

    fn deinit(self: *Left, gpa: Allocator) void {
        self.rows.deinit(gpa);
    }
};

fn splitRun(gpa: Allocator, held: Pair.Model, tokens: []const i32, windows: []const usize, rows_max: usize, capacity: usize, state: *Pair.State) !Left {
    var buffers: Pair.Buffers = try .init(gpa, held, rows_max, capacity);
    defer buffers.deinit();
    var left: Left = .{};
    errdefer left.deinit(gpa);
    var at: usize = 0;
    for (windows) |rows| {
        const window = tokens[at..][0..rows];
        try Pair.forward(held, state, &buffers, window);
        for (buffers.ranks) |one| {
            try appendRows(gpa, &left.rows, one.h.rows(rows));
            try appendRows(gpa, &left.rows, one.mixed.rows(rows));
            try appendRows(gpa, &left.rows, one.logits.rows(rows));
        }
        try Pair.commit(held, state, buffers, window, rows);
        at += rows;
    }
    try cuda.synchronize();
    left.caches = left.rows.items.len;
    for (state.ranks) |rank| for (rank.caches) |one| {
        try appendRows(gpa, &left.rows, one.k.rows(at));
        try appendRows(gpa, &left.rows, one.v.rows(at));
        try appendRows(gpa, &left.rows, one.index_keys.rows(at));
    };
    return left;
}

test "a window's glue split by rows over the ranks gives the whole glue's bits (spec 106 WS-02)" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const rows_max = 48;
    const built = try random.sharesOf(2, gpa, arena_state.allocator(), 42, 0, rows_max, .{ .caches = .bf16 });
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_max };
    defer held.deinit();
    var prng: std.Random.DefaultPrng = .init(106);
    const capacity = 160;
    var tokens: [48 + 33 + 17 + 35 + 5]i32 = undefined;
    for (&tokens) |*id| id.* = prng.random().intRangeLessThan(i32, 1, 300);
    // an even cut; odd row counts, which leave rank 1 one row (the 32nd, the 16th) or an odd 3 (from the 32nd),
    // so its rows start on a cut of 16 rows and not half way (spec 106 WS-05); and a window under the threshold
    const windows = [_]usize{ 48, 33, 17, 35, 5 };
    const saved = ranks.split_rows;
    defer ranks.split_rows = saved;

    ranks.split_rows = std.math.maxInt(usize);
    try testing.expect(!Pair.splits(held, 48, null));
    var whole_state: Pair.State = try .init(gpa, held, rows_max, capacity);
    defer whole_state.deinit();
    var whole = try splitRun(gpa, held, &tokens, &windows, rows_max, capacity, &whole_state);
    defer whole.deinit(gpa);
    // the serial run: a row a window, nothing split
    const steps: [tokens.len]usize = @splat(1);
    var serial_state: Pair.State = try .init(gpa, held, rows_max, capacity);
    defer serial_state.deinit();
    var serial = try splitRun(gpa, held, &tokens, &steps, rows_max, capacity, &serial_state);
    defer serial.deinit(gpa);

    ranks.split_rows = 17;
    for (windows[0..4]) |rows| try testing.expect(Pair.splits(held, rows, null));
    try testing.expect(!Pair.splits(held, 5, null) and !Pair.splits(held, 16, null));
    try testing.expectEqual([2]usize{ 16, 17 }, ranks_file.share(17, 2, 1));
    try testing.expectEqual([2]usize{ 32, 35 }, ranks_file.share(35, 2, 1));
    var split_state: Pair.State = try .init(gpa, held, rows_max, capacity);
    defer split_state.deinit();
    var split = try splitRun(gpa, held, &tokens, &windows, rows_max, capacity, &split_state);
    defer split.deinit(gpa);

    try testing.expectEqualSlices(u8, whole.rows.items, split.rows.items);
    for (whole_state.ranks, split_state.ranks) |a, b| try expectState(gpa, a, b);
    // the caches' rows and the states as the serial run leaves them
    try testing.expectEqualSlices(u8, serial.rows.items[serial.caches..], split.rows.items[split.caches..]);
    for (serial_state.ranks, split_state.ranks) |a, b| try expectState(gpa, a, b);
}

test "a rank's snapshot restores both shares to a cold run's state" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const built = try random.shares(2, gpa, arena_state.allocator(), 42, 0, rows_cap);
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_cap };
    defer held.deinit();
    const prompt = [_]i32{ 4, 11, 28 };
    var warm: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer warm.deinit();
    var cold: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer cold.deinit();
    var buffers: Pair.Buffers = try .init(gpa, held, rows_cap, 64);
    defer buffers.deinit();
    try Pair.forward(held, &warm, &buffers, &prompt);
    try Pair.commit(held, &warm, buffers, &prompt, prompt.len);
    var snaps: [2]forward.Snapshot = undefined;
    for (&snaps, 0..) |*snap, i| snap.* = try forward.Snapshot.take(gpa, warm.ranks[i]);
    defer for (&snaps) |*snap| snap.deinit();
    try testing.expectEqual(prompt.len, snaps[0].pos);
    try testing.expectEqual(snaps[0].pos, snaps[1].pos);
    const spoil = [_]i32{9};
    try Pair.forward(held, &warm, &buffers, &spoil);
    try Pair.commit(held, &warm, buffers, &spoil, 1);
    for (snaps, 0..) |snap, i| try snap.restore(&warm.ranks[i]);
    try Pair.forward(held, &cold, &buffers, &prompt);
    try Pair.commit(held, &cold, buffers, &prompt, prompt.len);
    try expectState(gpa, warm.ranks[0], cold.ranks[0]);
    try expectState(gpa, warm.ranks[1], cold.ranks[1]);
}

test "a rank's shelf: a state taken inside a window is the one a cold run commits there, over buffers a dropped one left, and no position twice" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const built = try random.shares(2, gpa, arena_state.allocator(), 42, 0, rows_cap);
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_cap };
    defer held.deinit();
    const prompt = [_]i32{ 4, 11, 28, 7 };
    var warm: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer warm.deinit();
    var cold: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer cold.deinit();
    var buffers: Pair.Buffers = try .init(gpa, held, rows_cap, 64);
    defer buffers.deinit();
    var other: Pair.Buffers = try .init(gpa, held, rows_cap, 64);
    defer other.deinit();
    var shelf: Pair.Shelf = .{ .gpa = gpa };
    defer shelf.deinit();
    var asked: Pair.Asked = .{};
    defer asked.deinit(gpa);
    // twice: the second round's states are written over the buffers the first's left
    for (0..2) |round| {
        for (&warm.ranks) |*one| try one.reset();
        warm.pos = 0;
        try Pair.forward(held, &warm, &buffers, &prompt);
        for ([_]usize{ 1, 3 }) |keep| try Pair.keepWithinLocal(&shelf, &asked, held, &warm, &buffers, &prompt, keep);
        // below the newest, and where one stands already
        try testing.expectError(error.OutOfStep, Pair.keepWithinLocal(&shelf, &asked, held, &warm, &buffers, &prompt, 2));
        try testing.expectError(error.OutOfStep, Pair.keepWithinLocal(&shelf, &asked, held, &warm, &buffers, &prompt, 3));
        try Pair.commit(held, &warm, buffers, &prompt, prompt.len);
        try Pair.keepLocal(&shelf, &warm);
        try testing.expectError(error.OutOfStep, Pair.keepLocal(&shelf, &warm));
        try testing.expectEqual(@as(usize, 3), shelf.held.items.len);
        try testing.expectEqual(@as(usize, 0), shelf.spare.count());
        for ([_]usize{ 3, 1, 4 }) |at| {
            try Pair.backLocal(&shelf, &warm, at);
            try testing.expectEqual(at, warm.pos);
            for (&cold.ranks) |*one| try one.reset();
            cold.pos = 0;
            try Pair.forward(held, &cold, &other, prompt[0..at]);
            try Pair.commit(held, &cold, other, prompt[0..at], at);
            for (warm.ranks, cold.ranks) |a, b| try expectState(gpa, a, b);
        }
        try testing.expectError(error.OutOfStep, Pair.backLocal(&shelf, &warm, 2));
        try testing.expectError(error.OutOfStep, Pair.dropLocal(&shelf, 2));
        for ([_]usize{ 1, 4, 3 }) |at| try Pair.dropLocal(&shelf, at);
        try testing.expectError(error.OutOfStep, Pair.backLocal(&shelf, &warm, 3));
        try testing.expectEqual(@as(usize, 0), shelf.held.items.len);
        try testing.expectEqual(@as(usize, 3), shelf.spare.count());
        _ = round;
    }
}

test "either share of the random model: the same streams, a window its serial steps, and drafted tokens" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const built = try random.shares(2, gpa, arena_state.allocator(), 42, 0, rows_cap);
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_cap };
    defer held.deinit();
    const vocab = random.fx.shape.vocab;
    const tokens = [_]i32{ 4, 11, 28 };

    var window_state: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer window_state.deinit();
    var buffers: Pair.Buffers = try .init(gpa, held, rows_cap, 64);
    defer buffers.deinit();
    try Pair.forward(held, &window_state, &buffers, &tokens);

    const h_row = rowBytes(buffers.ranks[0].h);
    var window_h = try gpa.alloc(u8, tokens.len * h_row);
    defer gpa.free(window_h);
    var window_logits = try gpa.alloc(u16, tokens.len * vocab);
    defer gpa.free(window_logits);
    for (0..tokens.len) |row| {
        try downloadRow(buffers.ranks[0].h, row, window_h[row * h_row ..][0..h_row]);
        var other: [2048]u8 = undefined;
        try downloadRow(buffers.ranks[1].h, row, other[0..h_row]);
        try testing.expectEqualSlices(u8, window_h[row * h_row ..][0..h_row], other[0..h_row]);
        try Pair.logitsRow(buffers, row, window_logits[row * vocab ..][0..vocab]);
    }

    const half = vocab / 2;
    var picks: [2]cuda.Buffer = undefined;
    for (&picks) |*out| {
        out.* = try .init(.u32, &.{picking.Picked.wordsFor(half)});
        try picking.Picked.ask(out.*, 32);
    }
    defer for (&picks) |*out| out.deinit();
    var picked: [2]picking.Picked = undefined;
    for (&picks, &picked, 0..) |out, *one, rank| {
        try picking.pick(buffers.ranks[rank].logits.rows(1), out.view);
        var raw: [picking.Picked.words]u32 = undefined;
        try out.downloadAt(0, std.mem.sliceAsBytes(&raw));
        one.* = .read(&raw);
    }
    var joined_scratch: picking.Joined = .{};
    const joined = picking.joined(&picked, half, &joined_scratch) orelse {
        std.debug.print("a half's pick gave no start\n", .{});
        return error.Unexpected;
    };
    var sampler: sampling.Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    const greedy: sampling.Sampling = .{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
    const led = sampler.chooseLeading(joined, 0, greedy) orelse return error.Unexpected;
    try testing.expectEqual(argmax(window_logits[0..vocab]), led);

    try Pair.commit(held, &window_state, buffers, &tokens, tokens.len);

    var serial_state: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer serial_state.deinit();
    var serial_h: [2048]u8 = undefined;
    var serial_logits: [320]u16 = undefined;
    for (tokens, 0..) |token, row| {
        const step = [1]i32{token};
        try Pair.forward(held, &serial_state, &buffers, &step);
        try downloadRow(buffers.ranks[0].h, 0, serial_h[0..h_row]);
        try testing.expectEqualSlices(u8, window_h[row * h_row ..][0..h_row], serial_h[0..h_row]);
        try downloadRow(buffers.ranks[1].h, 0, serial_h[0..h_row]);
        try testing.expectEqualSlices(u8, window_h[row * h_row ..][0..h_row], serial_h[0..h_row]);
        try Pair.logitsRow(buffers, 0, &serial_logits);
        try testing.expectEqualSlices(u16, window_logits[row * vocab ..][0..vocab], &serial_logits);
        try Pair.commit(held, &serial_state, buffers, &step, 1);
    }
    try expectState(gpa, window_state.ranks[0], serial_state.ranks[0]);
    try expectState(gpa, window_state.ranks[1], serial_state.ranks[1]);

    // A greedy reply, then the same tokens drafted. The oracle's drafts are the serial choices.
    var fresh: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer fresh.deinit();
    const prompt = [_]i32{7};
    try Pair.forward(held, &fresh, &buffers, &prompt);
    try Pair.logitsRow(buffers, 0, &serial_logits);
    try Pair.commit(held, &fresh, buffers, &prompt, 1);
    // The last chosen token is not committed: a drafted round leaves it for the next, so the states meet at
    // the token before it.
    var expected: [3]i32 = undefined;
    for (&expected, 0..) |*token, i| {
        token.* = argmax(&serial_logits);
        if (i + 1 == expected.len) break;
        const step = [1]i32{token.*};
        try Pair.forward(held, &fresh, &buffers, &step);
        try Pair.logitsRow(buffers, 0, &serial_logits);
        try Pair.commit(held, &fresh, buffers, &step, 1);
    }

    var drafted_state: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer drafted_state.deinit();
    const logits = try gpa.alloc(u16, vocab);
    defer gpa.free(logits);
    var oracle: Oracle = .{ .reply = &expected, .first = prompt.len };
    var decoder: Decoder = .{
        .m = held,
        .state = &drafted_state,
        .b = &buffers,
        .graphs = null,
        .sampler = &sampler,
        .settings = greedy,
        .eos = &.{},
        .logits = logits,
    };
    var chosen: [draft.forward_max_rows]i32 = undefined;
    var drafts: [draft.forward_max_rows]i32 = undefined;
    const n = try decoder.prompt(oracle.drafter(), &prompt, &chosen);
    try testing.expectEqual(prompt.len, drafted_state.pos);
    var out: [expected.len]i32 = undefined;
    out[0] = chosen[n - 1];
    var made: usize = 1;
    var kept: []const i32 = chosen[0..n];
    var held_tokens: [draft.forward_max_rows]i32 = undefined;
    while (made < out.len) {
        @memcpy(held_tokens[0..kept.len], kept);
        const count = try decoder.draft(oracle.drafter(), held_tokens[0..kept.len], out.len - made - 1, &drafts);
        const keep = try decoder.round(out[made - 1], drafts[0..count], &chosen);
        @memcpy(out[made..][0..keep], chosen[0..keep]);
        made += keep;
        kept = chosen[0..keep];
    }
    try testing.expectEqualSlices(i32, &expected, &out);
    try expectState(gpa, fresh.ranks[0], drafted_state.ranks[0]);
}

/// A prompt's windows take the states of `due` within themselves, as `pair.zig`'s `Keeping.window` does where
/// the prompt is told per pass.
const Taking = struct {
    pair: *Both,
    due: []const usize,

    fn within(context: *anyopaque, run: []const i32, last: bool) !void {
        _ = last;
        const self: *Taking = @ptrCast(@alignCast(context));
        const from = self.pair.state.pos;
        for (self.due) |at| {
            if (at <= from or at > from + run.len) continue;
            Pair.holdWithin(self.pair.m, &self.pair.state, &self.pair.b, &self.pair.shelf, &self.pair.asked, run, at - from) catch return error.DraftFailed;
        }
    }
};

test "a prompt told once: the follower's frames run the windows the direct telling does" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // both ranks of each pair in one process: `a` the direct telling, `b` the frames' (`promptFollowed`)
    var pairs: [2]*Both = undefined;
    var made: usize = 0;
    defer for (pairs[0..made]) |one| one.destroy();
    for (&pairs) |*one| {
        const built = try random.sharesOf(2, gpa, arena, 42, 0, rows_cap, .{ .head = .split });
        one.* = try Both.createLoopback(gpa, .{ .ranks = built, .max_rows = rows_cap }, rows_cap, 64, true);
        made += 1;
    }
    const a = pairs[0];
    const b = pairs[1];
    const greedy: sampling.Sampling = .{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
    var prng: std.Random.DefaultPrng = .init(94);
    // three windows of 4 rows: the head drafts between the first two, and the last is committed whole
    const first = try arena.alloc(i32, 11);
    for (first) |*id| id.* = prng.random().intRangeLessThan(i32, 1, 300);
    const due = [_]usize{6};

    var rounds: Decoder = .{
        .m = a.m,
        .state = &a.state,
        .b = &a.b,
        .graphs = null,
        .sampler = &a.sampler,
        .settings = greedy,
        .eos = &.{},
        .logits = a.logits,
    };
    var chosen: [draft.forward_max_rows]i32 = undefined;

    // the first prompt: told where the state stands, the head drafting, a state taken inside the second
    // window. The frames go to `b` a few tokens at a time; `a` runs the direct telling.
    const stay: book_file.Resumed = .{ .shared = 0, .to = .stay, .drops = &.{} };
    a.head.?.settings = greedy;
    var frame: Both.PromptFrame = undefined;
    var at: usize = 0;
    while (at < first.len) {
        at = try a.promptFrame(stay, &.{}, &.{}, &due, first, at, 5, greedy, &.{}, true, &frame);
        try b.promptFollowed(&frame.arguments, frame.words[0..frame.said]);
    }
    var taking: Taking = .{ .pair = a, .due = &due };
    rounds.within = .{ .context = &taking, .window = Taking.within };
    _ = try rounds.prompt(a.head.?.drafter(), first, &chosen);
    rounds.within = null;
    try testing.expectEqual(first.len, a.state.pos);
    try testing.expectEqual(first.len, b.state.pos);

    try expectState(gpa, a.state.ranks[0], b.state.ranks[0]);
    try expectState(gpa, a.state.ranks[1], b.state.ranks[1]);
    try testing.expectEqual(a.head.?.state.position.host, b.head.?.state.position.host);
    try testing.expectEqual(a.shelf.held.items.len, b.shelf.held.items.len);
    for (a.shelf.held.items, b.shelf.held.items) |one, two| {
        try testing.expectEqual(one[0].pos, two[0].pos);
        try testing.expectEqual(one[1].pos, two[1].pos);
    }

    // the next prompt goes on from the state taken inside the first: the state at its end goes, the ranks
    // back to 6, the head where the prompt's windows left it, and a state taken inside the next prompt too
    try Pair.hold(a.m, &a.state, &a.shelf);
    try Pair.hold(b.m, &b.state, &b.shelf);
    const junk = [_]i32{ 7, 8 };
    for (pairs[0..made]) |one| {
        try Pair.forward(one.m, &one.state, &one.b, &junk);
        try Pair.commit(one.m, &one.state, one.b, &junk, junk.len);
    }
    const second = try arena.alloc(i32, first.len + 7);
    @memcpy(second[0..first.len], first);
    for (second[first.len..]) |*id| id.* = prng.random().intRangeLessThan(i32, 1, 300);
    const dropped = [_]usize{first.len};
    const taken_within = [_]usize{first.len + 1};
    const back: book_file.Resumed = .{ .shared = 6, .to = .{ .back = 6 }, .drops = &.{} };
    at = 0;
    while (at < second.len) {
        at = try a.promptFrame(back, &dropped, &.{}, &taken_within, second, at, 5, greedy, &.{}, true, &frame);
        try b.promptFollowed(&frame.arguments, frame.words[0..frame.said]);
    }
    try Pair.drop(a.m, &a.shelf, first.len);
    try Pair.back(a.m, &a.state, &a.shelf, 6);
    try a.head.?.rewind(6, greedy);
    var again: Taking = .{ .pair = a, .due = &taken_within };
    rounds.within = .{ .context = &again, .window = Taking.within };
    _ = try rounds.prompt(a.head.?.drafter(), second, &chosen);
    rounds.within = null;

    try expectState(gpa, a.state.ranks[0], b.state.ranks[0]);
    try expectState(gpa, a.state.ranks[1], b.state.ranks[1]);
    try testing.expectEqual(a.head.?.state.position.host, b.head.?.state.position.host);
    try testing.expectEqual(a.shelf.held.items.len, b.shelf.held.items.len);
    for (a.shelf.held.items, b.shelf.held.items) |one, two| {
        try testing.expectEqual(one[0].pos, two[0].pos);
        try testing.expectEqual(one[1].pos, two[1].pos);
    }

    // A third prompt back at 6 whose plan drops the state at 6 itself, as a long prompt's plan does with its
    // budget full: rank 0 goes back first and drops after, and the frames' telling must too (it once dropped
    // first, and a two-rank serve parted on a 4,000-token prompt).
    const third = try arena.alloc(i32, 6 + 9);
    @memcpy(third[0..6], first[0..6]);
    for (third[6..]) |*id| id.* = prng.random().intRangeLessThan(i32, 1, 300);
    const above_six = [_]usize{first.len + 1};
    const planned_drop = [_]usize{6};
    const third_due = [_]usize{9};
    at = 0;
    while (at < third.len) {
        at = try a.promptFrame(back, &above_six, &planned_drop, &third_due, third, at, 5, greedy, &.{}, true, &frame);
        try b.promptFollowed(&frame.arguments, frame.words[0..frame.said]);
    }
    try Pair.drop(a.m, &a.shelf, first.len + 1);
    try Pair.back(a.m, &a.state, &a.shelf, 6);
    try a.head.?.rewind(6, greedy);
    try Pair.drop(a.m, &a.shelf, 6);
    var third_taking: Taking = .{ .pair = a, .due = &third_due };
    rounds.within = .{ .context = &third_taking, .window = Taking.within };
    _ = try rounds.prompt(a.head.?.drafter(), third, &chosen);
    rounds.within = null;

    try expectState(gpa, a.state.ranks[0], b.state.ranks[0]);
    try expectState(gpa, a.state.ranks[1], b.state.ranks[1]);
    try testing.expectEqual(a.head.?.state.position.host, b.head.?.state.position.host);
    try testing.expectEqual(@as(usize, 1), b.shelf.held.items.len);
    try testing.expectEqual(a.shelf.held.items.len, b.shelf.held.items.len);
    for (a.shelf.held.items, b.shelf.held.items) |one, two| {
        try testing.expectEqual(one[0].pos, two[0].pos);
        try testing.expectEqual(one[1].pos, two[1].pos);
    }
}

test "the gathered partials are read only as gathered: a window's bits over buffers left dirty" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const built = try random.shares(2, gpa, arena_state.allocator(), 42, 0, rows_cap);
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_cap };
    defer held.deinit();
    var state: Pair.State = try .init(gpa, held, rows_cap, 64);
    defer state.deinit();
    var buffers: Pair.Buffers = try .init(gpa, held, rows_cap, 64);
    defer buffers.deinit();
    const vocab = random.fx.shape.vocab;
    const tokens = [_]i32{ 4, 11, 28, 7, 9 };
    const logits = try gpa.alloc(u16, tokens.len * vocab);
    defer gpa.free(logits);
    const again = try gpa.alloc(u16, vocab);
    defer gpa.free(again);

    for (tokens, 0..) |token, at| {
        const step = [1]i32{token};
        try Pair.forward(held, &state, &buffers, &step);
        try Pair.logitsRow(buffers, 0, logits[at * vocab ..][0..vocab]);
        try Pair.commit(held, &state, buffers, &step, 1);
    }
    // the same rows over partials and sums left dirty: only what the ranks wrote and the gathers bring is
    // read, so the bits are the clean run's
    for ([_]*cuda.Buffer{ &buffers.branch, &buffers.moe_parts, &buffers.branch_bf16, &buffers.moe_sum }) |buffer| {
        const dirt = try gpa.alloc(u8, buffer.bytes());
        defer gpa.free(dirt);
        @memset(dirt, 0xaa);
        try buffer.upload(dirt);
    }
    for (&state.ranks) |*one| try one.reset();
    state.pos = 0;
    for (tokens, 0..) |token, at| {
        const step = [1]i32{token};
        try Pair.forward(held, &state, &buffers, &step);
        try Pair.logitsRow(buffers, 0, again);
        try testing.expectEqualSlices(u16, logits[at * vocab ..][0..vocab], again);
        try Pair.commit(held, &state, buffers, &step, 1);
    }
}

const Oracle = struct {
    reply: []const i32,
    first: usize,

    fn propose(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?@import("../../../src/decode/masks.zig").Guide, out: []i32) anyerror!usize {
        const self: *Oracle = @ptrCast(@alignCast(context));
        std.debug.assert(guide == null);
        try testing.expectEqual(@as(i64, @intCast(tokens.len)), streams.size[0]);
        var made: usize = 0;
        while (made < out.len) : (made += 1) {
            const at = position + 1 + made;
            if (at < self.first or at - self.first >= self.reply.len) break;
            out[made] = self.reply[at - self.first];
        }
        return made;
    }

    fn drafter(self: *Oracle) draft.Drafter {
        return .{ .context = self, .draft = propose };
    }
};

const ranks_fixture = @import("ranks_fixture.zig");
const replyOn = ranks_fixture.replyOn;
const reply_settings = ranks_fixture.reply_settings;

test "the MTP head whole beside rank 0's share of the random model: the drafted reply is the two ranks' serial one, greedy and sampled" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const built = try random.sharesDrafting(2, gpa, arena_state.allocator(), 42, 0, rows_cap);
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_cap };
    defer held.deinit();
    try testing.expect(held.ranks[0].mtp != null and held.ranks[1].mtp == null);
    // the share's own sizes are a rank's half, the head's the whole model's
    try testing.expectEqual(random.fx.expert_width / 2, held.ranks[0].shape.width);
    try testing.expectEqual(random.fx.expert_width, mtp_file.over(held.ranks[0]).shape.width);
    // longer than a window, so the head absorbs a prompt window before the last
    const prompt = [_]i32{ 5, 300, 17, 17, 0, 251 };
    const count = 40;
    for (reply_settings) |settings| {
        const serial = try replyOn(gpa, held, &prompt, count, settings, false, false, 0, 64);
        defer gpa.free(serial.tokens);
        for ([_]bool{ true, false }) |graphed| for ([_]f32{ 0, mtp_file.Head.default_chain }) |chain| {
            const got = try replyOn(gpa, held, &prompt, count, settings, true, graphed, chain, 64);
            defer gpa.free(got.tokens);
            try testing.expectEqualSlices(i32, serial.tokens, got.tokens);
            try testing.expectEqual(count - 1, got.stats.rounds + got.stats.accepted);
            // a head of random weights is rarely right; without a floor it still fills every window
            if (chain == 0) try testing.expect(got.stats.drafted > got.stats.rounds);
        };
    }
}
