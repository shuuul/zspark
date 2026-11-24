//! `ple.zig`'s tests: each kernel against the same steps on the host, a row alone against the row among others,
//! a window against its serial steps. The n-gram ids and the gather: `ple_gather_test.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const qwen4_config = @import("../../../../src/models/qwen4/config.zig");
const layout = @import("../../../../src/models/qwen4/layout.zig");
const ple_file = @import("../../../../src/models/qwen4/blocks/ple.zig");
const Ahead = ple_file.Ahead;
const Buffers = ple_file.Buffers;
const Shape = ple_file.Shape;
const State = ple_file.State;
const Table = ple_file.Table;
const Weights = ple_file.Weights;
const commit = ple_file.commit;
const forward = ple_file.forward;
const gather = ple_file.gather;
const gatherAfter = ple_file.gatherAfter;
const historyAfter = ple_file.historyAfter;
const leading = ple_file.leading;
const ngramIds = ple_file.ngramIds;
const stage = ple_file.stage;
const uploaded = ple_file.uploaded;
const Config = qwen4_config.Config;
const Ngram = qwen4_config.Ngram;

const testing = std.testing;
const rows_fixture = @import("ple_fixture.zig");
const fetch = rows_fixture.fetch;
const ngramOf = rows_fixture.ngramOf;
const Tensor = @import("../../../../src/weights/safetensors.zig").Tensor;
const numeric_fixture = @import("../../../weights/numeric_fixture.zig");
const bf = numeric.roundBf16;
const sigmoid = numeric_fixture.sigmoid;

/// The fixture's sizes: the published model's 3-grams, taps and dilation over a small table and small streams.
const fx = struct {
    const shape: Shape = .{ .hidden = 320, .streams = 4, .heads = 4, .dims = 16, .taps = 4, .dilation = 3, .ngram = 3 };
    const wide = shape.streams * shape.hidden;
    const emb = shape.heads * shape.dims;
    const eos = 9;
    const eps = 1e-6;
    const max_rows = 12;
    /// eos alone, doubled and at the window's end; more rows than the tail holds
    const tokens = [max_rows]u32{ 5, 17, 9, 3, 3, 299, 9, 9, 1, 40, 41, 9 };
};

/// A PLE with random weights, a bf16 table, and the rows it is run on.
const Model = struct {
    arena: std.heap.ArenaAllocator,
    ngram: Ngram,
    table: Table,
    weights: Weights,
    scratch: *cuda.Scratch,
    key: []u16,
    value: []u16,
    norm_key: []f32,
    norm_query: []f32,
    norm_conv: []f32,
    conv: []u16,
    /// bf16 bits [max_rows, wide]: the streams before the branch
    h: []u16,
    /// bf16 bits [tail rows, wide]: a tail to start from, as if rows had been committed
    tail: []u16,
    /// What `Table.gather` gives for row `id`, by direct dequantization.
    rows: []u16,

    fn init(seed: u64) !Model {
        var self: Model = undefined;
        self.arena = .init(testing.allocator);
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        self.ngram = try ngramOf(arena, 300, 3, 2, 100, 8, 0, fx.emb, 0);

        // two shards of bf16 rows, each an e4m3 code up to a magnitude of 15 times 0.05, none of them NaN
        const scale: f32 = 0.05;
        const counts = [2]usize{ 200, 224 };
        self.rows = try arena.alloc(u16, 424 * fx.shape.dims);
        for (self.rows) |*bits| {
            const code = random.uintLessThan(u8, 88) | (@as(u8, random.uintLessThan(u8, 2)) << 7);
            bits.* = numeric.bf16FromF32(numeric.f32FromE4m3(code) * scale);
        }
        const bytes = std.mem.sliceAsBytes(self.rows);
        const shards = try arena.alloc(layout.Table.Shard, 2);
        var first: usize = 0;
        for (counts, shards) |count, *shard| {
            const shape = try arena.dupe(u64, &.{ count, fx.shape.dims });
            shard.* = .{ .weight = .{ .dtype = .BF16, .shape = shape, .data = bytes[first * fx.shape.dims * 2 ..][0 .. count * fx.shape.dims * 2] } };
            first += count;
        }
        self.table = try .init(arena, .{ .scheme = .bf16, .shards = shards, .scale = null, .rows = 424, .dims = fx.shape.dims });

        self.key = try arena.alloc(u16, fx.wide * fx.emb);
        self.value = try arena.alloc(u16, fx.shape.hidden * fx.emb);
        self.conv = try arena.alloc(u16, fx.wide * fx.shape.taps);
        self.h = try arena.alloc(u16, fx.max_rows * fx.wide);
        self.tail = try arena.alloc(u16, fx.shape.tailRows() * fx.wide);
        for (self.key) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.3);
        for (self.value) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.3);
        for (self.conv) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.5);
        for (self.h) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
        for (self.tail) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
        self.norm_key = try arena.alloc(f32, fx.wide);
        self.norm_query = try arena.alloc(f32, fx.wide);
        self.norm_conv = try arena.alloc(f32, fx.wide);
        for (self.norm_key, self.norm_query, self.norm_conv) |*k, *q, *c| {
            k.* = 1.0 + 0.2 * random.floatNorm(f32);
            q.* = 1.0 + 0.2 * random.floatNorm(f32);
            c.* = 1.0 + 0.2 * random.floatNorm(f32);
        }
        self.scratch = try .create(testing.allocator);
        errdefer self.scratch.destroy();
        self.weights = try .initHost(fx.shape, fx.eps, self.table.scale, .{
            .key = std.mem.sliceAsBytes(self.key),
            .value = std.mem.sliceAsBytes(self.value),
            .norm_key = self.norm_key,
            .norm_query = self.norm_query,
            .norm_conv = self.norm_conv,
            .conv = std.mem.sliceAsBytes(self.conv),
        }, self.scratch.rows(fx.max_rows));
        return self;
    }

    fn deinit(self: *Model) void {
        self.weights.deinit();
        self.scratch.destroy();
        self.arena.deinit();
    }

    /// A sequence whose tail is the fixture's.
    fn sequence(self: Model, gpa: Allocator) !State {
        var seq: State = try .init(gpa, fx.shape, fx.eos);
        errdefer seq.deinit(gpa);
        try seq.tail.upload(std.mem.sliceAsBytes(self.tail));
        return seq;
    }
};

/// Every buffer of one window, read back.
const Window = struct {
    ids: []u64,
    keys: []u16,
    vals: []u16,
    gated: []u16,
    pss: []f32,
    nrow: []u16,
    h: []u16,

    fn deinit(self: Window, gpa: Allocator) void {
        inline for (comptime std.meta.fieldNames(Window)) |name| gpa.free(@field(self, name));
    }
};

/// The fixture's first `rows` rows as one window from the fixture's tail, nothing committed.
fn window(gpa: Allocator, model: Model, rows: usize) !Window {
    var seq = try model.sequence(gpa);
    defer seq.deinit(gpa);
    var buffers: Buffers = try .init(gpa, fx.shape, rows);
    defer buffers.deinit(gpa);
    var h = try uploaded(.bf16, &.{ rows, fx.wide }, std.mem.sliceAsBytes(model.h[0 .. rows * fx.wide]));
    defer h.deinit();
    try stage(model.ngram, model.table, seq, buffers, fx.tokens[0..rows]);
    try forward(model.weights, buffers, seq, h.view, rows);
    var result: Window = undefined;
    result.ids = try gpa.dupe(u64, buffers.ids);
    errdefer gpa.free(result.ids);
    result.keys = try fetch(u16, gpa, buffers.keys.view);
    errdefer gpa.free(result.keys);
    result.vals = try fetch(u16, gpa, buffers.vals.view);
    errdefer gpa.free(result.vals);
    result.gated = try fetch(u16, gpa, buffers.gated.view);
    errdefer gpa.free(result.gated);
    result.pss = try fetch(f32, gpa, buffers.pss.view);
    errdefer gpa.free(result.pss);
    result.nrow = try fetch(u16, gpa, buffers.nrow.view);
    errdefer gpa.free(result.nrow);
    result.h = try fetch(u16, gpa, h.view);
    return result;
}

fn dot(w: []const u16, x: []const u16) f64 {
    var sum: f64 = 0;
    for (w, x) |a, b| sum += @as(f64, numeric.f32FromBf16(a)) * numeric.f32FromBf16(b);
    return sum;
}

test "the PLE's projections, gate and convolution match the host" {
    const gpa = testing.allocator;
    var model: Model = try .init(31);
    defer model.deinit();
    const rows = 5;
    const got = try window(gpa, model, rows);
    defer got.deinit(gpa);
    const hidden = fx.shape.hidden;
    const hidden_f: f32 = @floatFromInt(hidden);
    const streams = fx.shape.streams;
    const tail_rows = fx.shape.tailRows();

    var want_ids: [rows * fx.shape.heads]u64 = undefined;
    ngramIds(model.ngram, fx.eos, &.{ fx.eos, fx.eos }, fx.tokens[0..rows], &want_ids);
    try testing.expectEqualSlices(u64, &want_ids, got.ids);
    for (0..rows) |r| {
        // the embedding: the token's table rows side by side
        var emb: [fx.emb]u16 = undefined;
        for (0..fx.shape.heads) |head_at| {
            const id: usize = @intCast(want_ids[r * fx.shape.heads + head_at]);
            @memcpy(emb[head_at * fx.shape.dims ..][0..fx.shape.dims], model.rows[id * fx.shape.dims ..][0..fx.shape.dims]);
        }
        for (0..fx.wide) |e| {
            const want = dot(model.key[e * fx.emb ..][0..fx.emb], &emb);
            try testing.expectApproxEqAbs(want, @as(f64, numeric.f32FromBf16(got.keys[r * fx.wide + e])), @abs(want) / 64.0 + 1e-3);
        }
        for (0..hidden) |d| {
            const want = dot(model.value[d * fx.emb ..][0..fx.emb], &emb);
            try testing.expectApproxEqAbs(want, @as(f64, numeric.f32FromBf16(got.vals[r * hidden + d])), @abs(want) / 64.0 + 1e-3);
        }

        for (0..streams) |s| {
            // the gate, from the device's keys and values
            const keys = got.keys[(r * streams + s) * hidden ..][0..hidden];
            const query = model.h[(r * streams + s) * hidden ..][0..hidden];
            var kss: f32 = 0;
            var qss: f32 = 0;
            for (keys, query) |k, q| {
                kss += numeric.f32FromBf16(k) * numeric.f32FromBf16(k);
                qss += numeric.f32FromBf16(q) * numeric.f32FromBf16(q);
            }
            const krinv = 1.0 / @sqrt(kss / hidden_f + fx.eps);
            const qrinv = 1.0 / @sqrt(qss / hidden_f + fx.eps);
            var sum: f32 = 0;
            for (keys, query, 0..) |k, q, d| {
                const kn = bf(numeric.f32FromBf16(k) * krinv * model.norm_key[s * hidden + d]);
                const qn = bf(numeric.f32FromBf16(q) * qrinv * model.norm_query[s * hidden + d]);
                sum += bf(kn * qn);
            }
            const raw = bf(sum / @sqrt(hidden_f));
            const mag = @sqrt(@max(@abs(raw), 1e-6));
            const sg = bf(sigmoid(bf(if (raw > 0) mag else if (raw < 0) -mag else 0)));
            var gss: f64 = 0;
            for (0..hidden) |d| {
                const want = bf(sg * numeric.f32FromBf16(got.vals[r * hidden + d]));
                const gated: f32 = numeric.f32FromBf16(got.gated[(r * streams + s) * hidden + d]);
                try testing.expectApproxEqAbs(want, gated, @abs(want) / 64.0 + 1e-30);
                gss += @as(f64, gated) * gated;
            }
            const pss = got.pss[r * streams + s];
            try testing.expectApproxEqRel(gss, @as(f64, pss), 1e-5);

            // the convolution, from the device's gated rows: taps over [tail; the window's normed rows]
            const rinv = 1.0 / @sqrt(pss / hidden_f + fx.eps);
            for (0..hidden) |d| {
                const e = s * hidden + d;
                const gated: f32 = numeric.f32FromBf16(got.gated[r * fx.wide + e]);
                try testing.expectEqual(numeric.bf16FromF32(gated * rinv * model.norm_conv[e]), got.nrow[r * fx.wide + e]);
                var acc: f32 = 0;
                for (0..fx.shape.taps) |t| {
                    const at = r + t * fx.shape.dilation;
                    const x = if (at < tail_rows) model.tail[at * fx.wide + e] else got.nrow[(at - tail_rows) * fx.wide + e];
                    acc += numeric.f32FromBf16(model.conv[e * fx.shape.taps + t]) * numeric.f32FromBf16(x);
                }
                const conv = bf(acc);
                const branch = bf(gated + bf(conv * sigmoid(conv)));
                const before: f32 = numeric.f32FromBf16(model.h[r * fx.wide + e]);
                try testing.expectApproxEqAbs(bf(before + branch), numeric.f32FromBf16(got.h[r * fx.wide + e]), (@abs(before) + @abs(branch)) / 64.0 + 1e-30);
            }
        }
    }
}

test "a row's PLE bits do not depend on the rows after it" {
    const gpa = testing.allocator;
    var model: Model = try .init(32);
    defer model.deinit();
    const alone = try window(gpa, model, 1);
    defer alone.deinit(gpa);
    // up to the 12 rows of the fixture; the linears' wider tile past 16 rows is covered by bf16.zig's own test
    for ([_]usize{ 2, 5, fx.max_rows }) |rows| {
        const among = try window(gpa, model, rows);
        defer among.deinit(gpa);
        try testing.expectEqualSlices(u16, alone.keys, among.keys[0..fx.wide]);
        try testing.expectEqualSlices(u16, alone.vals, among.vals[0..fx.shape.hidden]);
        try testing.expectEqualSlices(u16, alone.gated, among.gated[0..fx.wide]);
        try testing.expectEqualSlices(u32, @ptrCast(alone.pss), @ptrCast(among.pss[0..fx.shape.streams]));
        try testing.expectEqualSlices(u16, alone.nrow, among.nrow[0..fx.wide]);
        try testing.expectEqualSlices(u16, alone.h, among.h[0..fx.wide]);
    }
}

test "rows gathered ahead are the bytes the window's own gather reads, across every seam, and no other window's" {
    const gpa = testing.allocator;
    var model: Model = try .init(35);
    defer model.deinit();
    var seq: State = try .init(gpa, fx.shape, fx.eos);
    defer seq.deinit(gpa);
    var plain: Buffers = try .init(gpa, fx.shape, fx.max_rows);
    defer plain.deinit(gpa);
    var staged: Buffers = try .init(gpa, fx.shape, fx.max_rows);
    defer staged.deinit(gpa);
    var ahead: Ahead = try .init(gpa, fx.shape, fx.max_rows);
    defer ahead.deinit(gpa);
    const stride = model.table.stride();
    var served: usize = 0;
    // The window [seam, end) after [from, seam): a window before it shorter than the history, an eos on either
    // side of the seam, a window of one row and of all that is left.
    for (0..fx.max_rows - 1) |from| for (from + 1..fx.max_rows) |seam| for (seam + 1..fx.max_rows + 1) |end| {
        const run = fx.tokens[from..seam];
        const next = fx.tokens[seam..end];
        const words = next.len * fx.shape.heads * stride;
        @memset(seq.history, fx.eos);
        if (from > 0) historyAfter(seq.history, fx.tokens[0..from], from);
        ahead.start(model.ngram, model.table, seq, run, next);
        historyAfter(seq.history, run, run.len);
        try gather(model.ngram, model.table, seq, plain, next);
        // another window is not served: one id off
        const ids = plain.ids[0 .. next.len * fx.shape.heads];
        ids[ids.len - 1] ^= 1;
        try testing.expectEqual(0, ahead.take(ids).len);
        ids[ids.len - 1] ^= 1;
        try testing.expectEqual(served, ahead.served);
        // a window that ends a row sooner is served: a row's ids are its own tokens' and those before it
        if (next.len > 1) {
            const fewer = ids.len - fx.shape.heads;
            try testing.expectEqualSlices(u16, plain.host[0 .. fewer * stride], ahead.take(ids[0..fewer]));
            served += 1;
        }
        @memset(staged.host, 0xdead);
        try gatherAfter(model.ngram, model.table, seq, &staged, &ahead, next);
        served += 1;
        try testing.expectEqual(served, ahead.served);
        try testing.expectEqualSlices(u64, ids, staged.ids[0..ids.len]);
        try testing.expectEqualSlices(u16, plain.host[0..words], staged.host[0..words]);
        // rows handed over are gone with their memory: the same window once more gathers its own
        @memset(staged.host, 0xdead);
        try gatherAfter(model.ngram, model.table, seq, &staged, &ahead, next);
        try testing.expectEqual(served, ahead.served);
        try testing.expectEqualSlices(u16, plain.host[0..words], staged.host[0..words]);
        // and the same window where the one before it was kept in part, so that it follows other tokens: its
        // own gather's bytes, whether those rows were held or not
        if (run.len > 1) {
            @memset(seq.history, fx.eos);
            if (from > 0) historyAfter(seq.history, fx.tokens[0..from], from);
            ahead.start(model.ngram, model.table, seq, run, next);
            historyAfter(seq.history, run, run.len - 1);
            const again = fx.tokens[seam - 1 .. end - 1];
            try gather(model.ngram, model.table, seq, plain, again);
            @memset(staged.host, 0xdead);
            try gatherAfter(model.ngram, model.table, seq, &staged, &ahead, again);
            served = ahead.served;
            try testing.expectEqualSlices(u16, plain.host[0..words], staged.host[0..words]);
        }
    };

    // nothing is held where there is no window to expect, or one wider than the buffers
    ahead.start(model.ngram, model.table, seq, fx.tokens[0..2], &.{});
    try testing.expectEqual(0, ahead.held());

    // A round's window asked for as its tokens come (the kept token, then with each draft): whatever was
    // asked, the window staged gets its own gather's bytes, the rows held for its first tokens taken and
    // the others read; a draft that turned out another's is not served past it.
    for (0..fx.max_rows) |from| for (from + 1..fx.max_rows + 1) |end| for (0..end - from + 1) |told| {
        const round = fx.tokens[from..end];
        const words = round.len * fx.shape.heads * stride;
        @memset(seq.history, fx.eos);
        if (from > 0) historyAfter(seq.history, fx.tokens[0..from], from);
        for (1..told + 1) |known| ahead.start(model.ngram, model.table, seq, &.{}, round[0..known]);
        try testing.expectEqual(told * fx.shape.heads, if (told == 0) 0 else ahead.held());
        try gather(model.ngram, model.table, seq, plain, round);
        @memset(staged.host, 0xdead);
        const before = ahead.served;
        try gatherAfter(model.ngram, model.table, seq, &staged, &ahead, round);
        try testing.expectEqual(before + @intFromBool(told > 0), ahead.served);
        try testing.expectEqualSlices(u16, plain.host[0..words], staged.host[0..words]);
        if (told > 1) {
            // the last token told was not the one staged: the rows before it are taken, its own read
            var other: [fx.max_rows]u32 = undefined;
            @memcpy(other[0..told], round[0..told]);
            other[told - 1] = round[told - 1] ^ 1;
            ahead.start(model.ngram, model.table, seq, &.{}, other[0 .. told - 1]);
            ahead.start(model.ngram, model.table, seq, &.{}, other[0..told]);
            @memset(staged.host, 0xdead);
            try gatherAfter(model.ngram, model.table, seq, &staged, &ahead, round);
            try testing.expectEqualSlices(u16, plain.host[0..words], staged.host[0..words]);
        }
    };
    served = ahead.served;

    // a table that lacks a row: the rows ahead are refused and the window's own gather says so
    var short = model.table;
    short.shards = model.table.shards[0..1];
    short.starts = model.table.starts[0..2];
    @memset(seq.history, fx.eos);
    try testing.expectError(error.RowOutOfRange, gather(model.ngram, short, seq, plain, fx.tokens[2..]));
    ahead.start(model.ngram, short, seq, fx.tokens[0..2], fx.tokens[2..]);
    historyAfter(seq.history, fx.tokens[0..2], 2);
    try testing.expectError(error.RowOutOfRange, gatherAfter(model.ngram, short, seq, &staged, &ahead, fx.tokens[2..]));
    try testing.expectEqual(served, ahead.served);
}

/// What a sequence of windows leaves: every committed row's streams, and the state after the last commit.
const Decoded = struct {
    h: []u16,
    tail: []u16,
    history: [fx.shape.ngram - 1]u32,

    fn deinit(self: Decoded, gpa: Allocator) void {
        gpa.free(self.h);
        gpa.free(self.tail);
    }
};

/// Runs the fixture's tokens through windows of `chunks[i][0]` rows of which `chunks[i][1]` are kept; the next
/// window starts after the kept rows.
fn decode(gpa: Allocator, model: Model, weights: Weights, chunks: []const [2]usize) !Decoded {
    var seq = try model.sequence(gpa);
    defer seq.deinit(gpa);
    var buffers: Buffers = try .init(gpa, fx.shape, fx.max_rows);
    defer buffers.deinit(gpa);
    var h: cuda.Buffer = try .init(.bf16, &.{ fx.max_rows, fx.wide });
    defer h.deinit();
    const out = try gpa.alloc(u16, fx.max_rows * fx.wide);
    errdefer gpa.free(out);
    var at: usize = 0;
    for (chunks) |chunk| {
        const rows, const keep = chunk;
        const view = leading(h, rows);
        const part: cuda.Buffer = .{ .view = view };
        try part.upload(std.mem.sliceAsBytes(model.h[at * fx.wide ..][0 .. rows * fx.wide]));
        const tokens = fx.tokens[at..][0..rows];
        try stage(model.ngram, model.table, seq, buffers, tokens);
        try forward(weights, buffers, seq, view, rows);
        try commit(&seq, buffers, tokens, keep);
        try cuda.synchronize();
        const kept: cuda.Buffer = .{ .view = leading(h, keep) };
        try kept.download(std.mem.sliceAsBytes(out[at * fx.wide ..][0 .. keep * fx.wide]));
        at += keep;
    }
    try testing.expectEqual(@as(usize, fx.max_rows), at);
    var result: Decoded = .{ .h = out, .tail = try fetch(u16, gpa, seq.tail.view), .history = undefined };
    @memcpy(&result.history, seq.history);
    return result;
}

test "a PLE window equals its serial steps, tail and history committed between them" {
    const gpa = testing.allocator;
    var model: Model = try .init(33);
    defer model.deinit();
    const serial_chunks: [fx.max_rows][2]usize = @splat(.{ 1, 1 });
    const serial = try decode(gpa, model, model.weights, &serial_chunks);
    defer serial.deinit(gpa);
    try testing.expectEqualSlices(u32, fx.tokens[fx.max_rows - 2 ..], &serial.history);

    const plans = [_][]const [2]usize{
        // one window of more rows than the tail holds
        &.{.{ 12, 12 }},
        // drafts cut short: rows past the kept ones are run again by the next window
        &.{ .{ 5, 3 }, .{ 4, 1 }, .{ 8, 8 } },
        &.{ .{ 2, 1 }, .{ 11, 10 }, .{ 1, 1 } },
        &.{ .{ 12, 9 }, .{ 3, 3 } },
    };
    for (plans) |plan| {
        const windows = try decode(gpa, model, model.weights, plan);
        defer windows.deinit(gpa);
        try testing.expectEqualSlices(u16, serial.h, windows.h);
        try testing.expectEqualSlices(u16, serial.tail, windows.tail);
        try testing.expectEqualSlices(u32, &serial.history, &windows.history);
    }

    // the tail really is read: from a zero tail the first row differs
    var fresh: State = try .init(gpa, fx.shape, fx.eos);
    defer fresh.deinit(gpa);
    var buffers: Buffers = try .init(gpa, fx.shape, 1);
    defer buffers.deinit(gpa);
    var h = try uploaded(.bf16, &.{ 1, fx.wide }, std.mem.sliceAsBytes(model.h[0..fx.wide]));
    defer h.deinit();
    try stage(model.ngram, model.table, fresh, buffers, fx.tokens[0..1]);
    try forward(model.weights, buffers, fresh, h.view, 1);
    const from_zero = try fetch(u16, gpa, h.view);
    defer gpa.free(from_zero);
    try testing.expect(!std.mem.eql(u16, from_zero, serial.h[0..fx.wide]));
}

test "a bf16 table's scale goes through the embedding kernel, window and serial alike" {
    const gpa = testing.allocator;
    var model: Model = try .init(34);
    defer model.deinit();
    // the same rows read as a bf16 table with a scale of its own
    var scaled = model.weights;
    scaled.scale = 0.75;
    const serial_chunks: [fx.max_rows][2]usize = @splat(.{ 1, 1 });
    const serial = try decode(gpa, model, scaled, &serial_chunks);
    defer serial.deinit(gpa);
    const windows = try decode(gpa, model, scaled, &.{ .{ 7, 4 }, .{ 8, 8 } });
    defer windows.deinit(gpa);
    try testing.expectEqualSlices(u16, serial.h, windows.h);
    try testing.expectEqualSlices(u16, serial.tail, windows.tail);
    const unscaled = try decode(gpa, model, model.weights, &serial_chunks);
    defer unscaled.deinit(gpa);
    try testing.expect(!std.mem.eql(u16, unscaled.h, serial.h));
}

test "the published checkpoint's PLE: a window equals its serial steps on the real weights" {
    // Needs ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory. With the n-gram shards present the rows
    // are the table's; while they are still downloading, random rows stand in for them.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var diagnostic: qwen4_config.Diagnostic = .{};
    const config: Config = try .read(arena, io, dir, &diagnostic);
    var checkpoint: @import("../../../../src/weights/safetensors.zig").Checkpoint = try .open(gpa, io, .cwd(), path);
    defer checkpoint.close();
    const found: layout.Layout = try .init(arena, &config, &checkpoint, &diagnostic);
    const layer = config.ple_layers[0];
    const shape: Shape = .of(config);
    const ngram: Ngram = try .init(arena, config, 0);
    const rows = 8;

    // the layout drops the whole layer while the table is missing, so the other tensors are taken by name
    const whole: ?layout.Ple = if (found.layers[layer]) |l| l.ple.? else null;
    const base = try std.fmt.allocPrint(arena, "{s}layers.{d}.ple", .{ found.prefix, layer });
    const names = [_][]const u8{ "key_proj.weight", "value_proj.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight", "conv1d.weight" };
    var tensors: [names.len]Tensor = undefined;
    for (names, &tensors) |name, *tensor| {
        tensor.* = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.{s}", .{ base, name })) orelse return error.SkipZigTest;
    }
    const ple: layout.Ple = whole orelse .{
        .index = 0,
        .ngram = ngram,
        .table = .{ .scheme = .bf16, .shards = &.{}, .scale = null, .rows = 0, .dims = ngram.dims },
        .key = .{ .name = "key_proj", .scheme = .bf16, .rows = @intCast(shape.wide()), .cols = config.ple_dim, .weight = tensors[0] },
        .value = .{ .name = "value_proj", .scheme = .bf16, .rows = config.hidden, .cols = config.ple_dim, .weight = tensors[1] },
        .norm_key = .{ .name = "norm_key", .tensor = tensors[2], .gamma = true },
        .norm_query = .{ .name = "norm_query", .tensor = tensors[3], .gamma = true },
        .norm_conv = .{ .name = "norm_conv", .tensor = tensors[4], .gamma = true },
        .conv = .{ .name = "conv1d", .tensor = tensors[5], .channels = @intCast(shape.wide()), .taps = config.ple_kernel },
    };
    var table: ?Table = if (whole) |p| try .init(gpa, p.table) else null;
    defer if (table) |*t| t.deinit(gpa);
    var scratch: cuda.Scratch = .init(gpa);
    defer scratch.deinit();
    var weights: Weights = try .init(gpa, config, ple, found.norms_centred.?, if (table) |t| t.scale else 1.0, scratch.rows(rows));
    defer weights.deinit();

    var prng: std.Random.DefaultPrng = .init(41);
    const random = prng.random();
    const tokens = [rows]u32{ 151644, 872, 198, 9707, config.ple_eos, 11, 1879, 0 };
    const h_host = try arena.alloc(u16, rows * shape.wide());
    for (h_host) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    const stand_in = try arena.alloc(u16, rows * shape.embedding());
    for (stand_in) |*bits| {
        const sign: f32 = if (random.boolean()) 1 else -1;
        bits.* = numeric.bf16FromF32(sign * 0.01 * numeric.f32FromE4m3(random.uintLessThan(u8, 0x7f)));
    }

    var results: [2][]u16 = undefined;
    var tails: [2][]u16 = undefined;
    for ([_]usize{ rows, 1 }, &results, &tails) |step, *result, *tail| {
        var seq: State = try .init(gpa, shape, config.ple_eos);
        defer seq.deinit(gpa);
        var buffers: Buffers = try .init(gpa, shape, rows);
        defer buffers.deinit(gpa);
        var h = try uploaded(.bf16, &.{ rows, shape.wide() }, std.mem.sliceAsBytes(h_host));
        defer h.deinit();
        var at: usize = 0;
        while (at < rows) : (at += step) {
            const view = cuda.View.init(@ptrFromInt(@intFromPtr(h.view.ptr.?) + at * shape.wide() * 2), .bf16, &.{ step, shape.wide() });
            if (table) |t| {
                try stage(ngram, t, seq, buffers, tokens[at..][0..step]);
            } else {
                @memcpy(buffers.host[0 .. step * shape.embedding()], stand_in[at * shape.embedding() ..][0 .. step * shape.embedding()]);
                try buffers.upload(step, shape.dims);
            }
            try forward(weights, buffers, seq, view, step);
            try commit(&seq, buffers, tokens[at..][0..step], step);
            try cuda.synchronize();
        }
        result.* = try fetch(u16, arena, h.view);
        tail.* = try fetch(u16, arena, seq.tail.view);
    }
    try testing.expectEqualSlices(u16, results[1], results[0]);
    try testing.expectEqualSlices(u16, tails[1], tails[0]);
    var change: f64 = 0;
    for (results[0], h_host) |after, before| {
        const value = numeric.f32FromBf16(after);
        try testing.expect(std.math.isFinite(value));
        change += @abs(value - numeric.f32FromBf16(before));
    }
    try testing.expect(change > 0);
    std.debug.print("PLE on {s}: layer {d}, table {s}, {d} rows, mean |branch| {d:.4}\n", .{
        path, layer, if (table != null) "from the checkpoint" else "missing", rows, change / @as(f64, @floatFromInt(h_host.len)),
    });
}
