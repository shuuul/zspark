//! `mtp.zig`'s tests: each kernel against the same steps on the host and a row alone against the row among
//! others; a window of head rows against its serial steps; the drafts and the head under a guide; the draft
//! head. A checkpoint's layers with its head and against TensorFold: `mtp_checkpoint_test.zig`. What the device
//! finds in a scored row and where a chain ends: `mtp_pick_test.zig`.

const std = @import("std");
const fixtures = @import("testing.zig");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../src/cuda/cuda.zig");
const masks_file = @import("../../../src/decode/masks.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const attention = @import("../../../src/models/qwen4/blocks/attention.zig");
const draft = @import("../../../src/models/qwen4/draft.zig");
const forward = @import("../../../src/models/qwen4/forward.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const mtp_file = @import("../../../src/models/qwen4/mtp.zig");
const Buffers = mtp_file.Buffers;
const Graphs = mtp_file.Graphs;
const Head = mtp_file.Head;
const State = mtp_file.State;
const addStreams = mtp_file.addStreams;
const minus_infinity = mtp_file.minus_infinity;
const probability = mtp_file.probability;
const rmsnorm = mtp_file.rmsnorm;
const run = mtp_file.run;

const testing = std.testing;
const heads = @import("mtp_fixture.zig");
const device = heads.device;
const fetch = fixtures.attending.fetch;
const randomBf16 = heads.randomBf16;
const testSubset = heads.testSubset;
const headReply = heads.headReply;
const headReplyBeside = heads.headReplyBeside;

const fx = struct {
    const dims = 512;
    const streams = 4;
    const wide = streams * dims;
    const eps = 1e-6;
};

/// `rmsnorm` of `x` [rows, width]; the caller owns the result.
fn normOnDevice(gpa: std.mem.Allocator, x: []const u16, scale: []const f32, rows: usize, width: usize) ![]u16 {
    var x_dev = try device(u16, .bf16, &.{ rows, width }, x);
    defer x_dev.deinit();
    var scale_dev = try device(f32, .f32, &.{width}, scale);
    defer scale_dev.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ rows, width });
    defer out.deinit();
    try rmsnorm(x_dev.view, scale_dev.view, out.view, fx.eps);
    return fetch(u16, gpa, out);
}

test "rmsnorm: a row over its own RMS, the same alone and among others" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(5);
    const random = prng.random();
    // the widths the head norms: an embedding's, and a row of streams
    for ([_]usize{ fx.dims, fx.wide }) |width| {
        const rows = 5;
        const x = try gpa.alloc(u16, rows * width);
        defer gpa.free(x);
        // rows of different magnitudes, so one read with another's RMS is far off
        for (0..rows) |r| randomBf16(random, x[r * width ..][0..width], @floatFromInt(1 + 3 * r));
        const scale = try gpa.alloc(f32, width);
        defer gpa.free(scale);
        for (scale) |*v| v.* = 0.5 + random.float(f32);

        const got = try normOnDevice(gpa, x, scale, rows, width);
        defer gpa.free(got);
        for (0..rows) |r| {
            var squares: f64 = 0;
            for (x[r * width ..][0..width]) |bits| {
                const v: f64 = numeric.f32FromBf16(bits);
                squares += v * v;
            }
            const rinv: f32 = @floatCast(1.0 / @sqrt(squares / @as(f64, @floatFromInt(width)) + fx.eps));
            for (0..width) |i| {
                const want = numeric.f32FromBf16(x[r * width + i]) * rinv * scale[i];
                const have = numeric.f32FromBf16(got[r * width + i]);
                try testing.expectApproxEqAbs(want, have, @abs(want) / 64.0 + 1e-30);
            }
            const alone = try normOnDevice(gpa, x[r * width ..][0..width], scale, 1, width);
            defer gpa.free(alone);
            try testing.expectEqualSlices(u16, alone, got[r * width ..][0..width]);
        }
    }
}

test "rmsnorm: a 10,240-wide row over tiles keeps the one-block kernel's bytes" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(7);
    const random = prng.random();
    const width = 10240; // Flash Next's streams width: the tiled launch is taken at 4 rows or fewer
    const rows = 5; // over four rows the one-block kernel runs: its rows are the reference
    const x = try gpa.alloc(u16, rows * width);
    defer gpa.free(x);
    for (0..rows) |r| randomBf16(random, x[r * width ..][0..width], @floatFromInt(1 + r));
    const scale = try gpa.alloc(f32, width);
    defer gpa.free(scale);
    for (scale) |*v| v.* = 0.5 + random.float(f32);

    const whole = try normOnDevice(gpa, x, scale, rows, width);
    defer gpa.free(whole);
    // one row and several rows take the tiled launch; each tile block repeats the row's sums, so the bytes match
    for ([_]usize{ 1, 3 }) |n| {
        const tiled = try normOnDevice(gpa, x[0 .. n * width], scale, n, width);
        defer gpa.free(tiled);
        try testing.expectEqualSlices(u16, whole[0 .. n * width], tiled);
    }
}

test "addStreams: the embedding branch added to each stream, rounded once" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(6);
    const random = prng.random();
    const rows = 3;
    const e = try gpa.alloc(u16, rows * fx.dims);
    defer gpa.free(e);
    randomBf16(random, e, 1.0);
    const hs = try gpa.alloc(u16, rows * fx.wide);
    defer gpa.free(hs);
    randomBf16(random, hs, 1.0);

    var e_dev = try device(u16, .bf16, &.{ rows, fx.dims }, e);
    defer e_dev.deinit();
    var hs_dev = try device(u16, .bf16, &.{ rows * fx.streams, fx.dims }, hs);
    defer hs_dev.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ rows, fx.wide });
    defer out.deinit();
    try addStreams(e_dev.view, hs_dev.view, out.view, fx.streams);
    const got = try fetch(u16, gpa, out);
    defer gpa.free(got);
    // one fp32 sum of two bf16 values and one rounding: the host gives the same bits
    for (0..rows) |r| for (0..fx.streams) |s| for (0..fx.dims) |d| {
        const at = (r * fx.streams + s) * fx.dims + d;
        const want = numeric.bf16FromF32(numeric.f32FromBf16(e[r * fx.dims + d]) + numeric.f32FromBf16(hs[at]));
        try testing.expectEqual(want, got[at]);
    };

    // a row alone
    for (0..rows) |r| {
        var one: cuda.Buffer = try .init(.bf16, &.{ 1, fx.wide });
        defer one.deinit();
        try addStreams(e_dev.span(r, 1), hs_dev.span(r * fx.streams, fx.streams), one.view, fx.streams);
        const alone = try fetch(u16, gpa, one);
        defer gpa.free(alone);
        try testing.expectEqualSlices(u16, alone, got[r * fx.wide ..][0..fx.wide]);
    }
}

/// What absorbing a sequence leaves: every row's head streams, the logits of the rows that were scored, and the
/// cache's rows.
const Trace = struct {
    h: []u16,
    logits: []u16,
    scored: []bool,
    k: []u16,
    v: []u16,
    index_keys: []u16,

    fn deinit(self: Trace, gpa: Allocator) void {
        inline for (comptime std.meta.fieldNames(Trace)) |name| gpa.free(@field(self, name));
    }
};

/// An image prompt's rotary positions for the head's cache (`attention.Rope.set`).
const Rotated = struct { positions: []const [3]i32, delta: i32 };

/// Absorbs `tokens.len` rows of `input` in windows of `sizes`, taken in turn. With `chain`, two chained drafts
/// follow every window, as `Head` makes them, and are trimmed before the next. `rope`: the rows rotate by it.
fn absorb(gpa: Allocator, m: model.Model, input: cuda.Buffer, tokens: []const i32, sizes: []const usize, graphed: bool, chain: bool, rope: ?Rotated) !Trace {
    const capacity = 64;
    const wide = m.shape.wide();
    const vocab = m.shape.vocab;
    const total = tokens.len;
    var state: State = try .init(m, capacity);
    defer state.deinit();
    if (rope) |r| try state.rope.set(r.positions, r.delta);
    var b: Buffers = try .init(gpa, m, m.max_rows, capacity);
    defer b.deinit();
    var graphs: Graphs = try .init(gpa);
    defer graphs.deinit();
    const h = try gpa.alloc(u16, total * wide);
    errdefer gpa.free(h);
    const logits = try gpa.alloc(u16, total * vocab);
    errdefer gpa.free(logits);
    @memset(logits, 0);
    const scored = try gpa.alloc(bool, total);
    errdefer gpa.free(scored);
    @memset(scored, false);

    var at: usize = 0;
    var turn: usize = 0;
    while (at < total) : (turn += 1) {
        const rows = @min(sizes[turn % sizes.len], total - at);
        try state.trim();
        try std.testing.expectEqual(at, state.position.host);
        const streams = input.span(at, rows);
        if (graphed) try graphs.run(m, &state, &b, streams, tokens[at..][0..rows], true) else try run(m, &state, &b, streams, tokens[at..][0..rows], true);
        try state.position.advance(rows);
        try cuda.synchronize();
        try b.layer.h.downloadAt(0, std.mem.sliceAsBytes(h[at * wide ..][0 .. rows * wide]));
        try forward.logitsRow(b.layer, 0, logits[(at + rows - 1) * vocab ..][0..vocab]);
        scored[at + rows - 1] = true;
        at += rows;
        if (!chain) continue;
        for ([_]usize{ rows - 1, 0 }, [_]i32{ 3, 9 }) |last, token| {
            const own = b.layer.h.span(last, 1);
            if (graphed) try graphs.run(m, &state, &b, own, &.{token}, true) else try run(m, &state, &b, own, &.{token}, true);
            try state.position.advance(1);
            state.drafted += 1;
        }
    }
    try state.trim();
    try cuda.synchronize();
    const shape = m.attention.?.headShape();
    // u16s of a position's keys: bf16 values, or pairs of FP8 codes
    const key = shape.kv_heads * shape.head_dim * shape.kv.dtype().size() / 2;
    const k = try gpa.alloc(u16, total * key);
    errdefer gpa.free(k);
    try state.cache.k.downloadAt(0, std.mem.sliceAsBytes(k));
    const value = shape.kv_heads * shape.kv.valueWidth(shape.head_dim) * shape.kv.valueDtype().size() / 2;
    const v = try gpa.alloc(u16, total * value);
    errdefer gpa.free(v);
    try state.cache.v.downloadAt(0, std.mem.sliceAsBytes(v));
    const index_keys = try gpa.alloc(u16, total * shape.index_dim);
    errdefer gpa.free(index_keys);
    try state.cache.index_keys.downloadAt(0, std.mem.sliceAsBytes(index_keys));
    return .{ .h = h, .logits = logits, .scored = scored, .k = k, .v = v, .index_keys = index_keys };
}

test "prepare preserves head buffers and device position, and keys distinguish stream addresses, rows and scoring" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelWithHead(gpa, arena.allocator());
    defer m.deinit();
    var state: State = try .init(m, 64);
    defer state.deinit();
    var b: Buffers = try .init(gpa, m, m.max_rows, 64);
    defer b.deinit();
    var input: cuda.Buffer = try .init(.bf16, &.{ 2, m.shape.wide() });
    defer input.deinit();
    try input.zero();
    try b.layer.h.zero();
    try b.layer.logits.zero();
    var graphs: Graphs = try .init(gpa);
    defer graphs.deinit();
    const Case = struct { streams: cuda.View, rows: usize, score: bool };
    const cases = [_]Case{
        .{ .streams = input.span(0, 1), .rows = 1, .score = true },
        .{ .streams = input.span(1, 1), .rows = 1, .score = true },
        .{ .streams = input.span(0, 1), .rows = 1, .score = false },
        .{ .streams = input.view, .rows = 2, .score = true },
        .{ .streams = b.layer.h.span(0, 1), .rows = 1, .score = true },
    };
    for (cases, 0..) |case, i| {
        try graphs.prepare(m, &state, &b, case.streams, case.rows, 64, case.score);
        try testing.expectEqual(i + 1, graphs.cache.count());
        try testing.expect(graphs.cache.contains(.{ .rows = case.rows, .context = 64, .streams = @intFromPtr(case.streams.ptr.?), .score = case.score }));
        try graphs.prepare(m, &state, &b, case.streams, case.rows, 64, case.score);
        try testing.expectEqual(i + 1, graphs.cache.count());
    }
    try cuda.synchronize();
    const bytes = try gpa.alloc(u8, @max(b.layer.h.bytes(), b.layer.logits.bytes()));
    defer gpa.free(bytes);
    for ([_]cuda.Buffer{ b.layer.h, b.layer.logits }) |buffer| {
        try buffer.download(bytes[0..buffer.bytes()]);
        try testing.expect(std.mem.allEqual(u8, bytes[0..buffer.bytes()], 0));
    }
    try testing.expectEqual(0, state.position.host);
    var position: [1]i32 = undefined;
    try state.position.device.download(std.mem.sliceAsBytes(&position));
    try testing.expectEqual(0, position[0]);
    try graphs.run(m, &state, &b, input.span(0, 1), &.{5}, true);
    try testing.expectEqual(cases.len, graphs.cache.count());
}

test "a window of head rows equals its serial steps, eager and through graphs, and chained drafts leave no trace" {
    (try windowEqualsSteps(null)).deinit(testing.allocator);
}

test "with an image's rotary positions (spec 019) a window of head rows equals its serial steps, eager and through graphs, and the positions are read" {
    // 4 rows of text, a 4 x 4 grid at t 4, 4 rows of text from 8 on; past the table each row at pos - 12
    var positions: [24][3]i32 = undefined;
    for (&positions, 0..) |*p, i| {
        const at: i32 = @intCast(i);
        p.* = if (i < 4) .{ at, at, at } else if (i < 20) .{ 4, 4 + @divFloor(at - 4, 4), 4 + @mod(at - 4, 4) } else .{ at - 12, at - 12, at - 12 };
    }
    const rope: Rotated = .{ .positions = &positions, .delta = -12 };
    const plain = try windowEqualsSteps(null);
    defer plain.deinit(testing.allocator);
    const rotated = try windowEqualsSteps(rope);
    defer rotated.deinit(testing.allocator);
    // the text rows before the image are the plain ones', every row from the image's second on is not
    const row = plain.h.len / 30;
    try testing.expectEqualSlices(u16, plain.h[0 .. 5 * row], rotated.h[0 .. 5 * row]);
    for (5..30) |r| try testing.expect(!std.mem.eql(u16, plain.h[r * row ..][0..row], rotated.h[r * row ..][0..row]));
}

test "with FP8 caches, and with NVFP4 values, a window of head rows still equals its serial steps, and the head's reply is the serial one" {
    defer fixtures.random_model.caches = .bf16;
    for ([_]attention.Storage{ .fp8, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        (try windowEqualsSteps(null)).deinit(testing.allocator);
        try headRepliesSerially();
    }
}

test "the head's cache in bf16 beside coded caches: its rows are an all-bf16 load's, a window its serial steps, the reply the serial one" {
    defer fixtures.random_model.caches = .bf16;
    defer fixtures.random_model.head_caches = null;
    const gpa = testing.allocator;
    const plain = try mixedTrace(gpa);
    defer plain.deinit(gpa);
    for ([_]attention.Storage{ .fp8, .fp8_nvfp4 }) |storage| {
        fixtures.random_model.caches = storage;
        fixtures.random_model.head_caches = .bf16;
        // The random model's trunk does not attend, so the head reads the same streams whatever the
        // trunk's storage: with its own cache in bf16 every row and every cached key and value is the
        // all-bf16 load's, through a scratch whose queries are coded for the other caches.
        const mixed = try mixedTrace(gpa);
        defer mixed.deinit(gpa);
        inline for (.{ "h", "k", "v", "index_keys", "logits" }) |name| try testing.expectEqualSlices(u16, @field(plain, name), @field(mixed, name));
        (try windowEqualsSteps(null)).deinit(testing.allocator);
        try headRepliesSerially();
    }
}

/// The head's rows over 30 positions in windows, with chained drafts between them, under `fixtures.random_model.caches` and
/// `fixtures.random_model.head_caches`.
fn mixedTrace(gpa: Allocator) !Trace {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelWithHead(gpa, arena.allocator());
    defer m.deinit();
    var prng: std.Random.DefaultPrng = .init(8);
    const random = prng.random();
    const total = 30;
    const wide = m.shape.wide();
    const streams = try gpa.alloc(u16, total * wide);
    defer gpa.free(streams);
    randomBf16(random, streams, 1.5);
    var input = try device(u16, .bf16, &.{ total, wide }, streams);
    defer input.deinit();
    var tokens: [total]i32 = undefined;
    for (&tokens) |*token| token.* = random.intRangeLessThan(i32, 0, @intCast(m.shape.vocab));
    return absorb(gpa, m, input, &tokens, &.{ 3, 1, 4 }, true, true, null);
}

/// The serial trace, for the caller to free.
fn windowEqualsSteps(rope: ?Rotated) !Trace {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelWithHead(gpa, arena.allocator());
    defer m.deinit();
    var prng: std.Random.DefaultPrng = .init(8);
    const random = prng.random();
    // past position 20 the head's attention selects blocks
    const total = 30;
    const wide = m.shape.wide();
    const streams = try gpa.alloc(u16, total * wide);
    defer gpa.free(streams);
    randomBf16(random, streams, 1.5);
    var input = try device(u16, .bf16, &.{ total, wide }, streams);
    defer input.deinit();
    var tokens: [total]i32 = undefined;
    for (&tokens) |*token| token.* = random.intRangeLessThan(i32, 0, @intCast(m.shape.vocab));

    const serial = try absorb(gpa, m, input, &tokens, &.{1}, false, false, rope);
    errdefer serial.deinit(gpa);
    try testing.expect(std.mem.allEqual(bool, serial.scored, true));
    const Case = struct { sizes: []const usize, graphed: bool, chain: bool };
    for ([_]Case{
        .{ .sizes = &.{ 4, 3, 1, 2 }, .graphed = false, .chain = false },
        .{ .sizes = &.{ 4, 3, 1, 2 }, .graphed = true, .chain = false },
        .{ .sizes = &.{ 2, 4, 1 }, .graphed = false, .chain = true },
        .{ .sizes = &.{ 3, 1, 4 }, .graphed = true, .chain = true },
        .{ .sizes = &.{1}, .graphed = true, .chain = true },
    }) |case| {
        const got = try absorb(gpa, m, input, &tokens, case.sizes, case.graphed, case.chain, rope);
        defer got.deinit(gpa);
        try testing.expectEqualSlices(u16, serial.h, got.h);
        try testing.expectEqualSlices(u16, serial.k, got.k);
        try testing.expectEqualSlices(u16, serial.v, got.v);
        try testing.expectEqualSlices(u16, serial.index_keys, got.index_keys);
        const vocab = m.shape.vocab;
        for (got.scored, 0..) |was, row| if (was) try testing.expectEqualSlices(u16, serial.logits[row * vocab ..][0..vocab], got.logits[row * vocab ..][0..vocab]);
    }
    return serial;
}

test "the head drafts and the reply is the serial one, greedy and sampled" {
    try headRepliesSerially();
}

test "a chain that pays by a fitted calibration too: fewer drafts, and the reply's ids those of the probability squared alone and of the serial decoding" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), try testSubset(arena.allocator(), 320));
    defer m.deinit();
    const prompt = [_]i32{ 5, 300, 17, 17, 0, 251 };
    const count = 40;
    // a random head gives a draft a few hundredths, which a pack's calibration says lands more often than
    // that: it cuts what the head is sure of. So one that says less of such a draft, and a row so cheap that
    // the square of two such pays for it where this product of two does not
    const squared: mtp_file.Paying.Costs = .{ .row = 1e-8, .level = 1e-9 };
    var fitted = squared;
    fitted.landing = .{ .slope = 1, .levels = @splat(-6) };
    var differed = false;
    for ([_]sampling.Sampling{
        .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 },
        .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 },
    }) |settings| {
        const serial = try fixtures.serial.serialReply(gpa, m, &prompt, count, settings);
        defer gpa.free(serial);
        for ([_]bool{ true, false }) |graphed| {
            const plain = try headReplyBeside(gpa, m, &prompt, count, settings, graphed, 0, false, squared);
            defer gpa.free(plain.tokens);
            const judged = try headReplyBeside(gpa, m, &prompt, count, settings, graphed, 0, false, fitted);
            defer gpa.free(judged.tokens);
            try testing.expectEqualSlices(i32, serial, plain.tokens);
            try testing.expectEqualSlices(i32, serial, judged.tokens);
            try testing.expect(judged.stats.drafted <= plain.stats.drafted);
            differed = differed or judged.stats.drafted < plain.stats.drafted;
        }
    }
    try testing.expect(differed);
}

fn headRepliesSerially() !void {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // with the full head, then with a draft head that scores two tokens in three
    const subset = try testSubset(arena.allocator(), 320);
    for ([_]?[]const u32{ null, subset }) |draft_vocab| {
        var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), draft_vocab);
        defer m.deinit();
        // longer than a window, so the head absorbs a prompt window before the last
        const prompt = [_]i32{ 5, 300, 17, 17, 0, 251 };
        const count = 40;
        for ([_]sampling.Sampling{
            .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 },
            .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 },
        }) |settings| {
            const serial = try fixtures.serial.serialReply(gpa, m, &prompt, count, settings);
            defer gpa.free(serial);
            for ([_]bool{ true, false }) |graphed| for ([_]f32{ 0, 0.7 }) |confidence| {
                const got = try headReply(gpa, m, &prompt, count, settings, graphed, confidence);
                defer gpa.free(got.tokens);
                try testing.expectEqualSlices(i32, serial, got.tokens);
                try testing.expectEqual(count - 1, got.stats.rounds + got.stats.accepted);
                // a head of random weights is rarely right; without a confidence it still fills every window
                if (confidence == 0) try testing.expect(got.stats.drafted > got.stats.rounds);
            };
        }
    }
}

/// A guide a test scripts: what it says of each draft, and a draft of the head's own it refuses.
const Scripted = struct {
    says: []const masks_file.Next,
    /// The drafts it must be shown, where the test knows them.
    expected: ?[]const i32 = null,
    refuse: ?usize = null,

    fn guide(self: *Scripted) masks_file.Guide {
        return .{ .context = self, .nextFn = next };
    }

    fn next(context: *anyopaque, drafts: []const i32, more: bool) masks_file.Error!masks_file.Next {
        const self: *Scripted = @ptrCast(@alignCast(context));
        if (self.expected) |expected| {
            if (!std.mem.eql(i32, expected[0..drafts.len], drafts)) return error.ConstraintFailed;
        }
        if (self.refuse) |at| if (drafts.len == at + 1) return .refused;
        return if (more) self.says[drafts.len] else .closed;
    }
};

test "the head under a guide: forced drafts stand in its chain as its own would, and it chooses among what a mask leaves" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), null);
    defer m.deinit();
    const vocab = m.shape.vocab;
    var sampler: sampling.Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    var prng: std.Random.DefaultPrng = .init(22);
    const rows = 3;
    const host = try arena.allocator().alloc(u16, rows * m.shape.wide());
    randomBf16(prng.random(), host, 1.0);
    var streams = try device(u16, .bf16, &.{ rows, m.shape.wide() }, host);
    defer streams.deinit();
    const tokens = [rows]i32{ 4, 250, 9 };
    const settings: sampling.Sampling = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 1, .min_p = 0 };
    const depth = 5;
    const every = try arena.allocator().alloc(u32, (vocab + 31) / 32);
    @memset(every, std.math.maxInt(u32));

    for ([_]bool{ false, true }) |graphed| {
        // The chain, written without `Head`: the rows absorbed, then each draft chosen from the last step's
        // logits and run from the last step's streams. `scores` keeps the logits each was chosen from.
        var chain: [depth]i32 = undefined;
        const scores = try arena.allocator().alloc(u16, depth * vocab);
        {
            var state: State = try .init(m, 64);
            defer state.deinit();
            var b: Buffers = try .init(gpa, m, m.max_rows, 64);
            defer b.deinit();
            try run(m, &state, &b, streams.view, &tokens, true);
            try state.position.advance(rows);
            var last: usize = rows - 1;
            for (&chain, 0..) |*token, i| {
                const row = scores[i * vocab ..][0..vocab];
                try forward.logitsRow(b.layer, 0, row);
                token.* = try sampler.choose(row, rows + 1 + i, settings);
                try run(m, &state, &b, b.layer.h.span(last, 1), &.{token.*}, true);
                try state.position.advance(1);
                last = 0;
            }
        }
        // and with no guide the head drafts it
        {
            var head: Head = try .init(gpa, m, &sampler, settings, .{ .rows = m.max_rows, .capacity = 64, .graphs = graphed });
            defer head.deinit();
            head.confidence = 0;
            head.chain = 0;
            const d = head.drafter();
            var out: [depth]i32 = undefined;
            try testing.expectEqual(depth, try d.draft(d.context, streams.view, &tokens, rows, null, &out));
            try testing.expectEqualSlices(i32, &chain, &out);
            try testing.expectEqual(depth - 1, head.state.drafted);
            try testing.expectEqualSlices(u16, scores[(depth - 1) * vocab ..][0..vocab], head.logits);
        }
        // drafts and logits that differ, or a wrong one could stand for another
        try testing.expect(chain[0] != chain[1] and chain[1] != chain[2] and chain[3] != chain[4]);
        for (1..depth) |i| try testing.expect(!std.mem.eql(u16, scores[(i - 1) * vocab ..][0..vocab], scores[i * vocab ..][0..vocab]));

        // Each choice of which drafts the guide forces, the token being the chain's own: the head, run through
        // them only when it must choose after them, drafts the rest of the chain to the bit.
        for (0..1 << depth) |forced| {
            var says: [depth]masks_file.Next = undefined;
            for (&says, 0..) |*say, i| say.* = if (forced >> @intCast(i) & 1 == 1) .{ .token = chain[i] } else if ((i + @popCount(forced)) % 2 == 0) .free else .{ .among = every };
            var scripted: Scripted = .{ .says = &says, .expected = &chain };
            var head: Head = try .init(gpa, m, &sampler, settings, .{ .rows = m.max_rows, .capacity = 64, .graphs = graphed });
            defer head.deinit();
            head.confidence = 0;
            head.chain = 0;
            var out: [depth]i32 = undefined;
            const d = head.drafter();
            try testing.expectEqual(depth, try d.draft(d.context, streams.view, &tokens, rows, scripted.guide(), &out));
            try testing.expectEqualSlices(i32, &chain, &out);
            // it ran the drafts before the last it chose itself, and none after
            const steps: usize = if (forced == (1 << depth) - 1) 0 else depth - 1 - @clz(@as(u5, @truncate(~forced)));
            try testing.expectEqual(steps, head.state.drafted);
            // and the last it chose, it chose from the logits the chain has there
            if (forced != (1 << depth) - 1) try testing.expectEqualSlices(u16, scores[steps * vocab ..][0..vocab], head.logits);
        }

        // A mask without the chain's token: the head's draft is one of the mask's.
        // A draft of its own refused: taken back. Closed: nothing more. A mask of nothing: nothing more.
        const Case = struct { at: usize, kind: enum { narrowed, refused, closed, nothing } };
        for ([_]Case{ .{ .at = 2, .kind = .narrowed }, .{ .at = 0, .kind = .narrowed }, .{ .at = 3, .kind = .refused }, .{ .at = 0, .kind = .refused }, .{ .at = depth - 1, .kind = .refused }, .{ .at = 2, .kind = .closed }, .{ .at = 1, .kind = .nothing } }) |case| {
            const narrow = try gpa.alloc(u32, every.len);
            defer gpa.free(narrow);
            @memset(narrow, 0);
            var pair: [2]i32 = undefined;
            var found: usize = 0;
            for (0..vocab) |id| {
                if (found == 2) break;
                if (id == chain[case.at]) continue;
                pair[found] = @intCast(id);
                found += 1;
                if (case.kind == .narrowed) narrow[id / 32] |= @as(u32, 1) << @intCast(id % 32);
            }
            var says: [depth]masks_file.Next = @splat(.free);
            var scripted: Scripted = .{ .says = &says };
            switch (case.kind) {
                .narrowed, .nothing => says[case.at] = .{ .among = narrow },
                .closed => says[case.at] = .closed,
                .refused => scripted.refuse = case.at,
            }
            var head: Head = try .init(gpa, m, &sampler, settings, .{ .rows = m.max_rows, .capacity = 64, .graphs = graphed });
            defer head.deinit();
            head.confidence = 0;
            head.chain = 0;
            var out: [depth]i32 = undefined;
            const d = head.drafter();
            const n = try d.draft(d.context, streams.view, &tokens, rows, scripted.guide(), &out);
            try testing.expectEqualSlices(i32, chain[0..case.at], out[0..case.at]);
            if (case.kind == .narrowed) {
                try testing.expectEqual(depth, n);
                try testing.expect(out[case.at] == pair[0] or out[case.at] == pair[1]);
            } else try testing.expectEqual(case.at, n);
        }

        // The head's confidence is its own, from before the mask. Where each mask leaves one token the head
        // gave little, that token is certain among the mask's and is a guess all the same: it is drafted when
        // the window has nothing else, and nothing is drafted on from it. So is the chain's own first draft,
        // which this head gives less than 0.7.
        try testing.expect(probability(scores[0..vocab], @intCast(chain[0])) < 0.5);
        for ([_]bool{ false, true }) |narrowed| {
            const one = try gpa.alloc(u32, every.len);
            defer gpa.free(one);
            @memset(one, 0);
            const other: usize = if (chain[0] == 0) 1 else 0;
            one[other / 32] |= @as(u32, 1) << @intCast(other % 32);
            var says: [depth]masks_file.Next = @splat(.{ .among = if (narrowed) one else every });
            var scripted: Scripted = .{ .says = &says };
            var head: Head = try .init(gpa, m, &sampler, settings, .{ .rows = m.max_rows, .capacity = 64, .graphs = graphed });
            defer head.deinit();
            var out: [depth]i32 = undefined;
            const d = head.drafter();
            try testing.expectEqual(1, try d.draft(d.context, streams.view, &tokens, rows, scripted.guide(), &out));
            try testing.expectEqual(if (narrowed) @as(i32, @intCast(other)) else chain[0], out[0]);
        }
    }
}

test "the draft head: the full head's logits of its tokens within NVFP4's step, and drafts only among them" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const subset = try testSubset(arena.allocator(), 320);
    var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), subset);
    defer m.deinit();
    const vocab = m.shape.vocab;
    var sampler: sampling.Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    var prng: std.Random.DefaultPrng = .init(21);
    const rows = 3;
    const host = try arena.allocator().alloc(u16, rows * m.shape.wide());
    randomBf16(prng.random(), host, 1.0);
    var streams = try device(u16, .bf16, &.{ rows, m.shape.wide() }, host);
    defer streams.deinit();
    const tokens = [rows]i32{ 4, 250, 9 };

    for ([_]bool{ false, true }) |graphed| {
        var head: Head = try .init(gpa, m, &sampler, .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 1, .min_p = 0 }, .{ .rows = m.max_rows, .capacity = 64, .graphs = graphed });
        defer head.deinit();
        head.confidence = 0;
        head.chain = 0;
        var out: [5]i32 = undefined;
        const d = head.drafter();
        const n = try d.draft(d.context, streams.view, &tokens, rows, null, &out);
        try testing.expectEqual(out.len, n);
        // sampled at temperature 1 over everything: a token the draft head does not score would turn up
        for (out) |token| try testing.expect(@mod(token, 3) != 0);

        // the row the last draft was chosen from, scored by the full head
        try m.head.forward(head.b.layer.mixed.span(0, 1), head.b.layer.logits.rows(1));
        const full = try gpa.alloc(u16, vocab);
        defer gpa.free(full);
        try forward.logitsRow(head.b.layer, 0, full);
        var diff: f64 = 0;
        var ref: f64 = 0;
        for (full, head.logits, 0..) |theirs, mine, id| {
            if (id % 3 == 0) {
                try testing.expectEqual(minus_infinity, mine);
                continue;
            }
            const a: f64 = numeric.f32FromBf16(mine);
            const r: f64 = numeric.f32FromBf16(theirs);
            diff += (a - r) * (a - r);
            ref += r * r;
        }
        try testing.expect(@sqrt(diff / ref) < 0.15);
    }
}

test "a draft pays by the stream's rate: a stream whose rounds keep little verifies a draft one that keeps much does not" {
    const costs = mtp_file.Paying.published;
    var poor: mtp_file.Paying = .begin(costs);
    var rich: mtp_file.Paying = .begin(costs);
    try testing.expectApproxEqAbs(2.0 / (1 + costs.row + costs.level), poor.rate(), 1e-12);
    for (0..40) |_| {
        // a round of one token whose one draft failed, and one that kept seven of eight
        poor.done(1, .{ .drafts = 1, .scored = 2, .said = 0 });
        rich.done(8, .{ .drafts = 8, .scored = 8, .said = 0 });
    }
    try testing.expect(poor.rate() < 1.0 and rich.rate() > 4.0);
    const low = poor.floors();
    const high = rich.floors();
    // a second draft at 0.4 through its chain: verified in the first and not in the second, where the
    // chain would not have gone on to it either
    try testing.expect(low[0] < 0.3 and low[1] < 0.4 and high[0] > 0.4 and high[1] > high[0]);
    // between the two floors a draft is verified and ends its chain; under both it is not drafted, unless it
    // is the chain's first, which always is; a floor on the product is not read where costs are
    const rule: mtp_file.Rule = .{ .paying = rich, .chain = 0.99 };
    const between = (high[0] + high[1]) / 2;
    try testing.expectEqual(@TypeOf(rule.little(0, 0, 0)){ .low = false, .closing = false }, rule.little(0.9, high[1] + 0.01, 1));
    try testing.expectEqual(@TypeOf(rule.little(0, 0, 0)){ .low = false, .closing = true }, rule.little(0.9, between, 1));
    try testing.expectEqual(@TypeOf(rule.little(0, 0, 0)){ .low = true, .closing = false }, rule.little(0.9, high[0] - 0.01, 1));
    try testing.expectEqual(@TypeOf(rule.little(0, 0, 0)){ .low = false, .closing = true }, rule.little(0.9, high[0] - 0.01, 0));
    try testing.expect((mtp_file.Rule{ .paying = rich, .confidence = 0.95 }).little(0.9, 0.9, 0).low);
    // the rate follows a round's modelled cost and nothing timed: the same rounds, the same floors
    var again: mtp_file.Paying = .begin(costs);
    for (0..40) |_| again.done(1, .{ .drafts = 1, .scored = 2, .said = 0 });
    try testing.expectEqual(low, again.floors());
    // a head kept less often than it said is drafted from less far: three drafts after the first said to
    // keep two, none kept
    for (0..40) |_| again.done(1, .{ .drafts = 4, .scored = 4, .said = 2 });
    var said: mtp_file.Paying = .begin(costs);
    for (0..40) |_| said.done(1, .{ .drafts = 4, .scored = 4, .said = 0 });
    try testing.expect(again.floors()[0] > 1.5 * said.floors()[0] and again.got < 0.2 and said.got == 1);
}
