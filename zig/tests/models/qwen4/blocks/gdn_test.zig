//! `gdn.zig`'s tests: a window against the same steps on the host and against serial steps, and a replayed
//! prefix against the serial state.

const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const gdn = @import("../../../../src/models/qwen4/blocks/gdn.zig");
const numeric_fixture = @import("../../../weights/numeric_fixture.zig");
const gpu = @import("../../../cuda/testing.zig");
const bf = numeric.roundBf16;
const sigmoid = numeric_fixture.sigmoid;
const device = gpu.device;
const fetch = gpu.fetch;
const Scratch = gdn.Scratch;
const Shape = gdn.Shape;
const Weights = gdn.Weights;
const chain = gdn.chain;
const key_dim = gdn.key_dim;
const replayFromThrough = gdn.replayFromThrough;
const replayThrough = gdn.replayThrough;
const taps = gdn.taps;
const value_dim = gdn.value_dim;

const testing = std.testing;
const Allocator = std.mem.Allocator;

const Fixture = FixtureOf(.{});

/// A fixture of a layer's heads (`Shape{}`) or one rank's share.
fn FixtureOf(comptime heads: Shape) type {
    return struct {
        const Self = @This();
        const shape: Shape = heads;
        const rows = 5;
        const eps = 1e-6;

        projection: []u16,
        conv_state: []u16,
        conv_weight: []u16,
        state: []f32,
        a_log: []f32,
        dt_bias: []f32,
        norm: []u16,

        fn init(gpa: Allocator) !Self {
            return initRows(gpa, rows);
        }

        /// A fixture of `count` rows: `rows` of them for the tests of a step's window, more for a wide one.
        fn initRows(gpa: Allocator, count: usize) !Self {
            var prng: std.Random.DefaultPrng = .init(5);
            const random = prng.random();
            const self: Self = .{
                .projection = try gpa.alloc(u16, count * shape.projection()),
                .conv_state = try gpa.alloc(u16, (taps - 1) * shape.conv()),
                .conv_weight = try gpa.alloc(u16, shape.conv() * taps),
                .state = try gpa.alloc(f32, shape.state()),
                .a_log = try gpa.alloc(f32, shape.value_heads),
                .dt_bias = try gpa.alloc(f32, shape.value_heads),
                .norm = try gpa.alloc(u16, value_dim),
            };
            for (self.projection) |*v| v.* = numeric.bf16FromF32(random.floatNorm(f32));
            for (self.conv_state) |*v| v.* = numeric.bf16FromF32(random.floatNorm(f32));
            for (self.conv_weight) |*v| v.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.5);
            for (self.state) |*v| v.* = random.floatNorm(f32) * 0.1;
            for (self.a_log) |*v| v.* = random.floatNorm(f32) * 0.5 - 1.0;
            for (self.dt_bias) |*v| v.* = random.floatNorm(f32);
            for (self.norm) |*v| v.* = numeric.bf16FromF32(0.5 + random.float(f32));
            return self;
        }

        fn deinit(self: Self, gpa: Allocator) void {
            inline for (comptime std.meta.fieldNames(Self)) |name| gpa.free(@field(self, name));
        }
    };
}

const Result = struct {
    out: []u16,
    state: []f32,
    /// out's 32-group sums
    sums: []f32,
    /// what a replay reads: `Scratch`'s four buffers
    k: []f32,
    v: []u16,
    g: []f32,
    beta: []f32,

    fn deinit(self: Result, gpa: Allocator) void {
        inline for (comptime std.meta.fieldNames(Result)) |name| gpa.free(@field(self, name));
    }
};

/// Which kernels a test's chain and replay run through; a window under `wide_rows` rows is narrow either way.
const Through = enum { narrow, wide };

/// `count` rows from row `first` of the fixture as one chain from `state`; `keep` replays that many rows instead of
/// taking the chain's own final state.
fn runChain(gpa: Allocator, fx: anytype, state: []const f32, first: usize, count: usize, keep: ?usize) !Result {
    return runChainThrough(.narrow, gpa, fx, state, first, count, keep);
}

fn runChainThrough(through: Through, gpa: Allocator, fx: anytype, state: []const f32, first: usize, count: usize, keep: ?usize) !Result {
    return runChainStaged(through, gpa, fx, state, first, count, keep, 0);
}

/// `runChainThrough`, the replay to `keep` rows in two parts where `stage` is not 0: to `stage` rows, then on
/// from that state (`replayFrom`).
fn runChainStaged(through: Through, gpa: Allocator, fx: anytype, state: []const f32, first: usize, count: usize, keep: ?usize, stage: usize) !Result {
    const shape = @TypeOf(fx).shape;
    const pw = shape.projection();
    const channels = shape.conv();
    // the conv window before row `first`: the stored state's rows, then the fixture's own earlier rows
    const window = try gpa.alloc(u16, (taps - 1) * channels);
    defer gpa.free(window);
    for (0..taps - 1) |tap| {
        const at = first + tap;
        const source = if (at < taps - 1) fx.conv_state[at * channels ..][0..channels] else fx.projection[(at - (taps - 1)) * pw ..][0..channels];
        @memcpy(window[tap * channels ..][0..channels], source);
    }

    var projection = try device(u16, .bf16, &.{ count, pw }, fx.projection[first * pw ..][0 .. count * pw]);
    defer projection.deinit();
    var conv_state = try device(u16, .bf16, &.{ taps - 1, channels }, window);
    defer conv_state.deinit();
    var conv_weight = try device(u16, .bf16, &.{ channels, taps }, fx.conv_weight);
    defer conv_weight.deinit();
    var state_in = try device(f32, .f32, &.{ shape.value_heads, value_dim, key_dim }, state);
    defer state_in.deinit();
    var a_log = try device(f32, .f32, &.{shape.value_heads}, fx.a_log);
    defer a_log.deinit();
    var dt_bias = try device(f32, .f32, &.{shape.value_heads}, fx.dt_bias);
    defer dt_bias.deinit();
    var norm = try device(u16, .bf16, &.{value_dim}, fx.norm);
    defer norm.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ count, shape.out() });
    defer out.deinit();
    var sums: cuda.Buffer = try .init(.f32, &.{ count, shape.out() / 32 });
    defer sums.deinit();
    var state_out: cuda.Buffer = try .init(.f32, &.{ shape.value_heads, value_dim, key_dim });
    defer state_out.deinit();
    var scratch: Scratch = try .init(count, shape);
    defer scratch.deinit();
    var work: cuda.Buffer = try .init(.f32, &.{ count, shape.key_heads, key_dim });
    defer work.deinit();

    const weights: Weights = .{ .conv = conv_weight.view, .a_log = a_log.view, .dt_bias = dt_bias.view, .norm = norm.view };
    try chain(weights, .{ .projection = projection.view, .conv_state = conv_state.view, .state_in = state_in.view, .state_out = state_out.view, .out = out.view, .sums = sums.view, .scratch = &scratch, .work = if (through == .wide) work.view else null }, count, @TypeOf(fx).eps);
    var staged: cuda.Buffer = try .init(.f32, &.{ shape.value_heads, value_dim, key_dim });
    defer staged.deinit();
    if (keep) |rows| {
        if (stage == 0) {
            try replayThrough(if (through == .wide) .wide else .narrow, state_in.view, scratch, rows, state_out.view);
        } else {
            try replayThrough(if (through == .wide) .wide else .narrow, state_in.view, scratch, stage, staged.view);
            try replayFromThrough(if (through == .wide) .wide else .narrow, staged.view, scratch, stage, rows - stage, state_out.view);
        }
    }
    return .{
        .out = try fetch(u16, gpa, out),
        .state = try fetch(f32, gpa, state_out),
        .sums = try fetch(f32, gpa, sums),
        .k = try fetch(f32, gpa, scratch.k),
        .v = try fetch(u16, gpa, scratch.v),
        .g = try fetch(f32, gpa, scratch.g),
        .beta = try fetch(f32, gpa, scratch.beta),
    };
}

test "a DeltaNet window equals its serial steps, and a replayed prefix their state" {
    const gpa = testing.allocator;
    const fx: Fixture = try .init(gpa);
    defer fx.deinit(gpa);
    const rows = Fixture.rows;
    const width = Fixture.shape.out();

    const window = try runChain(gpa, fx, fx.state, 0, rows, null);
    defer window.deinit(gpa);

    const keep = 3;
    const state = try gpa.dupe(f32, fx.state);
    defer gpa.free(state);
    var kept: ?[]f32 = null;
    defer if (kept) |k| gpa.free(k);
    for (0..rows) |r| {
        const step = try runChain(gpa, fx, state, r, 1, null);
        defer step.deinit(gpa);
        try testing.expectEqualSlices(u16, step.out, window.out[r * width ..][0..width]);
        @memcpy(state, step.state);
        if (r + 1 == keep) kept = try gpa.dupe(f32, state);
    }
    try testing.expectEqualSlices(u32, @ptrCast(state), @ptrCast(window.state));

    const replayed = try runChain(gpa, fx, fx.state, 0, rows, keep);
    defer replayed.deinit(gpa);
    try testing.expectEqualSlices(u32, @ptrCast(kept.?), @ptrCast(replayed.state));
}

test "the DeltaNet chain and replay through rows of four floats and butterfly sums give the base's bits: a window, a row alone, a replayed prefix, a layer's heads and a rank's" {
    inline for (.{ Shape{}, Shape{ .key_heads = 8, .value_heads = 24 } }) |heads| try expectQuadsBase(FixtureOf(heads));
}

/// `gdn.setQuads`'s two sides on one fixture: the base (a float and a warp_sum at a time) and rows of four floats
/// (48 heads) with a row's sums in butterflies (spec 143 WS-05).
fn expectQuadsBase(comptime F: type) !void {
    const gpa = testing.allocator;
    const fx: F = try .init(gpa);
    defer fx.deinit(gpa);
    const rows = F.rows;
    // a window, each row alone from the fixture's state (the row rule's two sides), and a replay to 3 rows
    const Run = struct { first: usize, count: usize, keep: ?usize };
    var runs: [rows + 2]Run = undefined;
    runs[0] = .{ .first = 0, .count = rows, .keep = null };
    runs[1] = .{ .first = 0, .count = rows, .keep = 3 };
    for (0..rows) |r| runs[r + 2] = .{ .first = r, .count = 1, .keep = null };
    for (runs) |run| {
        gdn.setQuads(false);
        const base = try runChain(gpa, fx, fx.state, run.first, run.count, run.keep);
        defer base.deinit(gpa);
        gdn.setQuads(true);
        defer gdn.setQuads(true);
        const quads = try runChain(gpa, fx, fx.state, run.first, run.count, run.keep);
        defer quads.deinit(gpa);
        try testing.expectEqualSlices(u16, base.out, quads.out);
        try testing.expectEqualSlices(u32, @ptrCast(base.state), @ptrCast(quads.state));
        try testing.expectEqualSlices(u32, @ptrCast(base.sums), @ptrCast(quads.sums));
        try testing.expectEqualSlices(u32, @ptrCast(base.k), @ptrCast(quads.k));
        try testing.expectEqualSlices(u16, base.v, quads.v);
        try testing.expectEqualSlices(u32, @ptrCast(base.g), @ptrCast(quads.g));
        try testing.expectEqualSlices(u32, @ptrCast(base.beta), @ptrCast(quads.beta));
    }
}

test "a chain behind by the rows a window kept gives the bits of their replay and then the chain" {
    const gpa = testing.allocator;
    const fx: Fixture = try .init(gpa);
    defer fx.deinit(gpa);
    const shape = Fixture.shape;
    const pw = shape.projection();
    const channels = shape.conv();
    // a window of 4 rows of which 2 are kept, one of 3 from there of which 1 is kept, one of 2: as replays
    const first = try runChain(gpa, fx, fx.state, 0, 4, 2);
    defer first.deinit(gpa);
    const second = try runChain(gpa, fx, first.state, 2, 3, null);
    defer second.deinit(gpa);
    const mid = try runChain(gpa, fx, first.state, 2, 3, 1);
    defer mid.deinit(gpa);
    const third = try runChain(gpa, fx, mid.state, 3, 2, null);
    defer third.deinit(gpa);

    for ([_]bool{ false, true }) |quads| {
        gdn.setQuads(quads);
        defer gdn.setQuads(true);
        // the same windows from the one state, never written: each chain replays what it is behind by
        var state_in = try device(f32, .f32, &.{ shape.value_heads, value_dim, key_dim }, fx.state);
        defer state_in.deinit();
        var state_out: cuda.Buffer = try .init(.f32, &.{ shape.value_heads, value_dim, key_dim });
        defer state_out.deinit();
        var conv_weight = try device(u16, .bf16, &.{ channels, taps }, fx.conv_weight);
        defer conv_weight.deinit();
        var a_log = try device(f32, .f32, &.{shape.value_heads}, fx.a_log);
        defer a_log.deinit();
        var dt_bias = try device(f32, .f32, &.{shape.value_heads}, fx.dt_bias);
        defer dt_bias.deinit();
        var norm = try device(u16, .bf16, &.{value_dim}, fx.norm);
        defer norm.deinit();
        var out: cuda.Buffer = try .init(.bf16, &.{ 4, shape.out() });
        defer out.deinit();
        var sums: cuda.Buffer = try .init(.f32, &.{ 4, shape.out() / 32 });
        defer sums.deinit();
        var scratch: Scratch = try .init(Fixture.rows, shape);
        defer scratch.deinit();
        var behind: cuda.Buffer = try .init(.i32, &.{1});
        defer behind.deinit();
        const weights: Weights = .{ .conv = conv_weight.view, .a_log = a_log.view, .dt_bias = dt_bias.view, .norm = norm.view };
        const Window = struct { from: usize, rows: usize, behind: i32, want: Result };
        for ([_]Window{
            .{ .from = 0, .rows = 4, .behind = 0, .want = first },
            .{ .from = 2, .rows = 3, .behind = 2, .want = second },
            .{ .from = 3, .rows = 2, .behind = 3, .want = third },
        }, 0..) |window, index| {
            const tail = try gpa.alloc(u16, (taps - 1) * channels);
            defer gpa.free(tail);
            for (0..taps - 1) |tap| {
                const at = window.from + tap;
                const source = if (at < taps - 1) fx.conv_state[at * channels ..][0..channels] else fx.projection[(at - (taps - 1)) * pw ..][0..channels];
                @memcpy(tail[tap * channels ..][0..channels], source);
            }
            var conv_state = try device(u16, .bf16, &.{ taps - 1, channels }, tail);
            defer conv_state.deinit();
            var projection = try device(u16, .bf16, &.{ window.rows, pw }, fx.projection[window.from * pw ..][0 .. window.rows * pw]);
            defer projection.deinit();
            try behind.upload(std.mem.asBytes(&window.behind));
            try chain(weights, .{ .projection = projection.view, .conv_state = conv_state.view, .state_in = state_in.view, .state_out = state_out.view, .out = out.rows(window.rows), .sums = sums.rows(window.rows), .scratch = &scratch, .behind = behind.view }, window.rows, Fixture.eps);
            const got = try fetch(u16, gpa, out);
            defer gpa.free(got);
            // the first window's reference is its replay to 2 rows: its rows are the second's test
            if (index > 0) {
                try testing.expectEqualSlices(u16, window.want.out, got[0..window.want.out.len]);
                const state = try fetch(f32, gpa, state_out);
                defer gpa.free(state);
                try testing.expectEqualSlices(u32, @ptrCast(window.want.state), @ptrCast(state));
            }
            if (index == 1) {
                // a state inside the second window: one replay over the rows behind and the window's first
                try gdn.replay(state_in.view, scratch, 3, state_out.view);
                const state = try fetch(f32, gpa, state_out);
                defer gpa.free(state);
                try testing.expectEqualSlices(u32, @ptrCast(mid.state), @ptrCast(state));
            }
        }
    }
}

test "a wide DeltaNet window gives the chain kernel's bits, and a wide replay the replay's" {
    const gpa = testing.allocator;
    // two stages of 32 rows and part of a third
    const rows = 75;
    const fx: Fixture = try .initRows(gpa, rows);
    defer fx.deinit(gpa);
    const width = Fixture.shape.out();

    // a stage's edge on either side, and a window that starts inside the fixture
    for ([_][2]usize{ .{ 0, rows }, .{ 0, gdn.wide_rows }, .{ 0, gdn.wide_rows + 1 }, .{ 0, 2 * gdn.wide_rows }, .{ 7, rows - 7 } }) |window| {
        const narrow = try runChainThrough(.narrow, gpa, fx, fx.state, window[0], window[1], null);
        defer narrow.deinit(gpa);
        const wide = try runChainThrough(.wide, gpa, fx, fx.state, window[0], window[1], null);
        defer wide.deinit(gpa);
        try testing.expectEqualSlices(u16, narrow.out, wide.out);
        inline for (.{ "state", "sums", "k", "g", "beta" }) |name| try testing.expectEqualSlices(u32, @ptrCast(@field(narrow, name)), @ptrCast(@field(wide, name)));
        try testing.expectEqualSlices(u16, narrow.v, wide.v);
    }

    // the wide window against serial steps of the chain kernel, and its state after them
    const wide = try runChainThrough(.wide, gpa, fx, fx.state, 0, rows, null);
    defer wide.deinit(gpa);
    const state = try gpa.dupe(f32, fx.state);
    defer gpa.free(state);
    for (0..rows) |r| {
        const step = try runChain(gpa, fx, state, r, 1, null);
        defer step.deinit(gpa);
        try testing.expectEqualSlices(u16, step.out, wide.out[r * width ..][0..width]);
        @memcpy(state, step.state);
    }
    try testing.expectEqualSlices(u32, @ptrCast(state), @ptrCast(wide.state));

    // a prefix replayed through either layout, from a chain run through either
    for ([_]usize{ 1, gdn.wide_rows - 1, gdn.wide_rows, gdn.wide_rows + 1, rows - 1, rows }) |keep| {
        const narrow = try runChainThrough(.narrow, gpa, fx, fx.state, 0, rows, keep);
        defer narrow.deinit(gpa);
        const replayed = try runChainThrough(.wide, gpa, fx, fx.state, 0, rows, keep);
        defer replayed.deinit(gpa);
        try testing.expectEqualSlices(u32, @ptrCast(narrow.state), @ptrCast(replayed.state));
        // and a replay gone on from a state replayed to an earlier row, through either layout
        for ([_]usize{ 1, 5, gdn.wide_rows, rows - 3 }) |stage| {
            if (stage >= keep) continue;
            for ([_]Through{ .narrow, .wide }) |through| {
                const staged = try runChainStaged(through, gpa, fx, fx.state, 0, rows, keep, stage);
                defer staged.deinit(gpa);
                try testing.expectEqualSlices(u32, @ptrCast(narrow.state), @ptrCast(staged.state));
            }
        }
    }
}

test "a landing gives each state the bits of its replay: many states of a layer's heads and a rank's, past one launch" {
    inline for (.{ Shape{}, Shape{ .key_heads = 8, .value_heads = 24 } }) |heads| {
        gdn.setQuads(false);
        try expectLandReplays(FixtureOf(heads));
        gdn.setQuads(true);
        try expectLandReplays(FixtureOf(heads));
    }
}

/// `gdn.land` over more states than one launch takes (64), each from its own rows of one chain's saves and its
/// own count of them, under `wide_rows` and past it, against `gdn.replay` of each (spec 146 WS-07).
fn expectLandReplays(comptime F: type) !void {
    const gpa = testing.allocator;
    const shape = F.shape;
    const rows = 40;
    const fx: F = try .initRows(gpa, rows);
    defer fx.deinit(gpa);
    var projection = try device(u16, .bf16, &.{ rows, shape.projection() }, fx.projection);
    defer projection.deinit();
    var conv_state = try device(u16, .bf16, &.{ taps - 1, shape.conv() }, fx.conv_state);
    defer conv_state.deinit();
    var conv_weight = try device(u16, .bf16, &.{ shape.conv(), taps }, fx.conv_weight);
    defer conv_weight.deinit();
    var state_in = try device(f32, .f32, &.{ shape.value_heads, value_dim, key_dim }, fx.state);
    defer state_in.deinit();
    var a_log = try device(f32, .f32, &.{shape.value_heads}, fx.a_log);
    defer a_log.deinit();
    var dt_bias = try device(f32, .f32, &.{shape.value_heads}, fx.dt_bias);
    defer dt_bias.deinit();
    var norm = try device(u16, .bf16, &.{value_dim}, fx.norm);
    defer norm.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ rows, shape.out() });
    defer out.deinit();
    var sums: cuda.Buffer = try .init(.f32, &.{ rows, shape.out() / 32 });
    defer sums.deinit();
    var last: cuda.Buffer = try .init(.f32, &.{ shape.value_heads, value_dim, key_dim });
    defer last.deinit();
    var scratch: Scratch = try .init(rows, shape);
    defer scratch.deinit();
    const weights: Weights = .{ .conv = conv_weight.view, .a_log = a_log.view, .dt_bias = dt_bias.view, .norm = norm.view };
    try chain(weights, .{ .projection = projection.view, .conv_state = conv_state.view, .state_in = state_in.view, .state_out = last.view, .out = out.view, .sums = sums.view, .scratch = &scratch }, rows, F.eps);

    const count = 70;
    var replayed: cuda.Buffer = try .init(.f32, &.{ count, shape.value_heads, value_dim, key_dim });
    defer replayed.deinit();
    var landed: cuda.Buffer = try .init(.f32, &.{ count, shape.value_heads, value_dim, key_dim });
    defer landed.deinit();
    var entries: [count]gdn.Landing = undefined;
    for (&entries, 0..) |*entry, i| {
        // rows `first ..` of the saves, 1 to 35 of them, from the state the chain started at or the one it left
        const first = i % 5;
        const n = 1 + (i * 7) % (rows - 5);
        var own = scratch;
        own.k.view = scratch.k.span(first, n);
        own.v.view = scratch.v.span(first, n);
        own.g.view = scratch.g.span(first, n);
        own.beta.view = scratch.beta.span(first, n);
        const from = if (i % 3 == 0) last.view else state_in.view;
        try gdn.replay(from, own, n, replayed.span(i, 1));
        entry.* = .of(from, own, n, landed.span(i, 1));
    }
    try gdn.land(&entries);
    const want = try fetch(f32, gpa, replayed);
    defer gpa.free(want);
    const got = try fetch(f32, gpa, landed);
    defer gpa.free(got);
    try testing.expectEqualSlices(u32, @ptrCast(want), @ptrCast(got));
}

test "a DeltaNet chain matches the host" {
    const gpa = testing.allocator;
    const fx: Fixture = try .init(gpa);
    defer fx.deinit(gpa);
    const shape = Fixture.shape;
    const rows = 3;
    const pw = shape.projection();
    const channels = shape.conv();
    const got = try runChain(gpa, fx, fx.state, 0, rows, null);
    defer got.deinit(gpa);

    const state = try gpa.dupe(f32, fx.state);
    defer gpa.free(state);
    const act = try gpa.alloc(f32, channels);
    defer gpa.free(act);
    const ratio = shape.value_heads / shape.key_heads;
    for (0..rows) |r| {
        const row = fx.projection[r * pw ..][0..pw];
        for (0..channels) |c| {
            var acc: f32 = 0;
            for (0..taps) |tap| {
                const at = r + tap;
                const x = if (at < taps - 1) fx.conv_state[at * channels + c] else fx.projection[(at - (taps - 1)) * pw + c];
                acc += numeric.f32FromBf16(fx.conv_weight[c * taps + tap]) * numeric.f32FromBf16(x);
            }
            act[c] = bf(acc * sigmoid(acc));
        }
        for (0..shape.value_heads) |hv| {
            const hk = hv / ratio;
            var q: [key_dim]f32 = undefined;
            var k: [key_dim]f32 = undefined;
            var qq: f32 = 0;
            var kk: f32 = 0;
            for (0..key_dim) |i| {
                q[i] = act[hk * key_dim + i];
                k[i] = act[shape.key_heads * key_dim + hk * key_dim + i];
                qq += q[i] * q[i];
                kk += k[i] * k[i];
            }
            for (&q, &k) |*qv, *kv| {
                qv.* *= 1.0 / @sqrt(qq + 1e-6) / @sqrt(@as(f32, key_dim));
                kv.* *= 1.0 / @sqrt(kk + 1e-6);
            }
            const b = numeric.f32FromBf16(row[channels + shape.out() + hv]);
            const a = numeric.f32FromBf16(row[channels + shape.out() + shape.value_heads + hv]);
            const x = a + fx.dt_bias[hv];
            const softplus = if (x > 20.0) x else std.math.log1p(@exp(x));
            const g = @exp(-@exp(fx.a_log[hv]) * softplus);
            const beta = bf(sigmoid(b));

            var y: [value_dim]f32 = undefined;
            var yy: f32 = 0;
            for (0..value_dim) |j| {
                const s = state[(hv * value_dim + j) * key_dim ..][0..key_dim];
                var kv: f32 = 0;
                for (s, k) |*sv, kvv| {
                    sv.* *= g;
                    kv += sv.* * kvv;
                }
                const delta = (act[2 * shape.key_heads * key_dim + hv * value_dim + j] - kv) * beta;
                var o: f32 = 0;
                for (s, k, q) |*sv, kvv, qv| {
                    sv.* += kvv * delta;
                    o += sv.* * qv;
                }
                y[j] = bf(o);
                yy += y[j] * y[j];
            }
            const rinv = 1.0 / @sqrt(yy / value_dim + Fixture.eps);
            for (0..value_dim) |t| {
                const yn = bf(bf(y[t] * rinv) * numeric.f32FromBf16(fx.norm[t]));
                const z = numeric.f32FromBf16(row[channels + hv * value_dim + t]);
                const want = bf(yn * sigmoid(z));
                const have = numeric.f32FromBf16(got.out[r * shape.out() + hv * value_dim + t]);
                // sums run in another order on the device, and each rounding to bf16 can land a step apart
                try testing.expectApproxEqAbs(want, have, @abs(want) / 16.0 + 2e-3);
            }
        }
    }
    for (state, got.state) |want, have| try testing.expectApproxEqAbs(want, have, @abs(want) * 1e-3 + 1e-4);
}
