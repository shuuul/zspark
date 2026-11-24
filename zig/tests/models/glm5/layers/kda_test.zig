//! `kda.zig`'s tests on random weights: a window, chunks, a window cut short and serial steps bit for bit (up to
//! `kda.recurrent_rows`), a chunked window cut at multiples of `kda.chunk` bit for bit, and against the recurrent
//! chain within a bound, a mark at a chunk's start, two ranks' heads against the whole layer's, and the ranks' sum
//! against the layer on the host in fp64, which is what holds the wiring (which column, which taps, which head).
//! The recurrent chain's own bits are TensorFold's device code as it stands; the references hold them to
//! TensorFold's outputs (spec 044 WS-07).

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const face = @import("../../../../src/weights/face.zig");
const config_file = @import("../../../../src/models/glm5/layers/config.zig");
const fixtures = @import("../../../../src/models/glm5/checks/fixtures.zig");
const host = @import("../../../../src/models/glm5/layers/glue_ref.zig");
const kda = @import("../../../../src/models/glm5/layers/kda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const shared = @import("../testing.zig");
const device = shared.device;
const Source = fixtures.Source;

const fx = struct {
    const dims = 256;
    /// The whole layer's heads; a rank holds half.
    const heads = 4;
    const inner = heads * kda.head_dim;
    const layers = 2;

    fn config(arena: Allocator) !config_file.Config {
        var diagnostic: config_file.Diagnostic = .{};
        var c = try config_file.Config.parse(arena, fixtures.config, &diagnostic);
        c.hidden = dims;
        c.kda_heads = heads;
        return c;
    }

    fn name(arena: Allocator, layer: usize, leaf: []const u8) ![]const u8 {
        return std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.self_attn.{s}", .{ layer, leaf });
    }

    fn putBf16(arena: Allocator, random: std.Random, source: *Source, layer: usize, leaf: []const u8, shape: []const u64, spread: f32) !void {
        var count: usize = 1;
        for (shape) |dim| count *= @intCast(dim);
        const bits = try arena.alloc(u16, count);
        shared.randomBf16(random, bits, spread);
        try source.put(arena, try name(arena, layer, leaf), .{ .dtype = .BF16, .shape = try arena.dupe(u64, shape), .data = std.mem.sliceAsBytes(bits) });
    }

    fn putF32(arena: Allocator, random: std.Random, source: *Source, layer: usize, leaf: []const u8, count: usize, spread: f32) !void {
        const values = try arena.alloc(f32, count);
        shared.randomF32(random, values, spread);
        try source.put(arena, try name(arena, layer, leaf), .{ .dtype = .F32, .shape = try arena.dupe(u64, &.{count}), .data = std.mem.sliceAsBytes(values) });
    }

    /// A whole KDA layer of random tensors under the checkpoint's names.
    fn putLayer(arena: Allocator, random: std.Random, source: *Source, layer: usize) !void {
        for ([_][]const u8{ "q", "k", "v" }) |one| {
            try putBf16(arena, random, source, layer, try std.fmt.allocPrint(arena, "{s}_proj.weight", .{one}), &.{ inner, dims }, 0.08);
            try putBf16(arena, random, source, layer, try std.fmt.allocPrint(arena, "{s}_conv1d.weight", .{one}), &.{ inner, 1, kda.window + 1 }, 0.6);
        }
        try putBf16(arena, random, source, layer, "f_a_proj.weight", &.{ kda.head_dim, dims }, 0.08);
        try putBf16(arena, random, source, layer, "g_a_proj.weight", &.{ kda.head_dim, dims }, 0.08);
        try putBf16(arena, random, source, layer, "f_b_proj.weight", &.{ inner, kda.head_dim }, 0.1);
        try putBf16(arena, random, source, layer, "g_b_proj.weight", &.{ inner, kda.head_dim }, 0.1);
        try putBf16(arena, random, source, layer, "b_proj.weight", &.{ heads, dims }, 0.08);
        try putBf16(arena, random, source, layer, "o_proj.weight", &.{ dims, inner }, 0.05);
        try putBf16(arena, random, source, layer, "o_norm.weight", &.{kda.head_dim}, 1.0);
        try putF32(arena, random, source, layer, "A_log", heads, 0.5);
        try putF32(arena, random, source, layer, "dt_bias", inner, 0.5);
    }

    /// Layer `layer` whole, from the tensors as they stand.
    fn whole(arena: Allocator, source: Source, c: *const config_file.Config, layer: usize, rows: cuda.Scratch.Rows) !kda.Weights {
        var parts: kda.Parts = undefined;
        inline for (comptime std.meta.fieldNames(kda.Parts), kda.Parts.leaves) |field, leaf| {
            const data = source.get(try name(arena, layer, leaf)).?.data;
            @field(parts, field) = if (comptime @FieldType(kda.Parts, field) == face.Part) .{ .bf16 = data } else data;
        }
        return kda.Weights.init(testing.allocator, parts, c, rows);
    }
};

fn bf(v: f64) f64 {
    return numeric.roundBf16(@floatCast(v));
}

fn sigmoid(v: f64) f64 {
    return 1.0 / (1.0 + @exp(-v));
}

/// A whole KDA layer over `rows` rows of x from an empty sequence, in fp64 with the device's bf16 roundings
/// (`kda.cu` `chain_kernel`, read as arithmetic): out [rows, dims].
fn hostLayer(arena: Allocator, source: Source, c: *const config_file.Config, layer: usize, x: []const u16, rows: usize, out: []f64) !void {
    const T = struct {
        fn bits(a: Allocator, s: Source, l: usize, leaf: []const u8) ![]align(1) const u16 {
            return @ptrCast(s.get(try fx.name(a, l, leaf)).?.data);
        }
        fn floats(a: Allocator, s: Source, l: usize, leaf: []const u8) ![]align(1) const f32 {
            return @ptrCast(s.get(try fx.name(a, l, leaf)).?.data);
        }
        fn at(w: []align(1) const u16, i: usize) f64 {
            return numeric.f32FromBf16(w[i]);
        }
        /// to[i] = bf16(w[first + i] . from), w's rows `from.len` wide
        fn product(w: []align(1) const u16, first: usize, from: []const f64, to: []f64) void {
            for (to, 0..) |*one, i| {
                var sum: f64 = 0;
                for (from, 0..) |v, j| sum += at(w, (first + i) * from.len + j) * v;
                one.* = bf(sum);
            }
        }
    };
    const hd = kda.head_dim;
    const proj = [3][]align(1) const u16{ try T.bits(arena, source, layer, "q_proj.weight"), try T.bits(arena, source, layer, "k_proj.weight"), try T.bits(arena, source, layer, "v_proj.weight") };
    const taps = [3][]align(1) const u16{ try T.bits(arena, source, layer, "q_conv1d.weight"), try T.bits(arena, source, layer, "k_conv1d.weight"), try T.bits(arena, source, layer, "v_conv1d.weight") };
    const f_a = try T.bits(arena, source, layer, "f_a_proj.weight");
    const g_a = try T.bits(arena, source, layer, "g_a_proj.weight");
    const f_b = try T.bits(arena, source, layer, "f_b_proj.weight");
    const g_b = try T.bits(arena, source, layer, "g_b_proj.weight");
    const b = try T.bits(arena, source, layer, "b_proj.weight");
    const o = try T.bits(arena, source, layer, "o_proj.weight");
    const norm = try T.bits(arena, source, layer, "o_norm.weight");
    const a_log = try T.floats(arena, source, layer, "A_log");
    const dt_bias = try T.floats(arena, source, layer, "dt_bias");

    const state = try arena.alloc(f64, fx.heads * hd * hd);
    @memset(state, 0);
    // q, k and v before the conv, every row: the conv reads three rows back
    const pre = try arena.alloc(f64, 3 * rows * fx.inner);
    const xr = try arena.alloc(f64, fx.dims);
    const y = try arena.alloc(f64, fx.inner);
    var low: [hd]f64 = undefined;
    var beta: [fx.heads]f64 = undefined;
    for (0..rows) |r| {
        for (xr, x[r * fx.dims ..][0..fx.dims]) |*to, from| to.* = numeric.f32FromBf16(from);
        for (0..3) |kind| T.product(proj[kind], 0, xr, pre[(kind * rows + r) * fx.inner ..][0..fx.inner]);
        T.product(b, 0, xr, &beta);
        for (0..fx.heads) |h| {
            var act: [3][hd]f64 = undefined;
            for (0..3) |kind| for (0..hd) |i| {
                const ch = h * hd + i;
                var acc: f64 = 0;
                for (0..kda.window + 1) |tap| {
                    if (r + tap < kda.window) continue;
                    acc += T.at(taps[kind], ch * (kda.window + 1) + tap) * pre[(kind * rows + r + tap - kda.window) * fx.inner + ch];
                }
                act[kind][i] = bf(acc * sigmoid(acc));
            };
            for (0..2) |kind| {
                var ss: f64 = 0;
                for (act[kind]) |v| ss += v * v;
                const inv = 1.0 / @sqrt(ss + 1e-6) * if (kind == 0) 1.0 / @sqrt(@as(f64, hd)) else 1.0;
                for (&act[kind]) |*v| v.* *= inv;
            }
            T.product(f_a, 0, xr, &low);
            var decay: [hd]f64 = undefined;
            T.product(f_b, h * hd, &low, &decay);
            for (&decay, 0..) |*g, i| g.* = @exp(c.gate_lower_bound * sigmoid(@exp(@as(f64, a_log[h])) * (g.* + dt_bias[h * hd + i])));
            T.product(g_a, 0, xr, &low);
            var gate: [hd]f64 = undefined;
            T.product(g_b, h * hd, &low, &gate);
            const s = state[h * hd * hd ..][0 .. hd * hd];
            var read: [hd]f64 = undefined;
            var ss: f64 = 0;
            for (0..hd) |j| {
                var kv: f64 = 0;
                for (0..hd) |i| {
                    s[j * hd + i] *= decay[i];
                    kv += s[j * hd + i] * act[1][i];
                }
                const delta = (act[2][j] - kv) * bf(sigmoid(beta[h]));
                var sum: f64 = 0;
                for (0..hd) |i| {
                    s[j * hd + i] += act[1][i] * delta;
                    sum += s[j * hd + i] * act[0][i];
                }
                read[j] = bf(sum);
                ss += read[j] * read[j];
            }
            const rinv = 1.0 / @sqrt(ss / @as(f64, hd) + c.eps);
            for (0..hd) |j| y[h * hd + j] = bf(T.at(norm, j) * (read[j] * rinv) * sigmoid(gate[j]));
        }
        for (out[r * fx.dims ..][0..fx.dims], 0..) |*to, d| {
            var sum: f64 = 0;
            for (y, 0..) |v, ch| sum += T.at(o, d * fx.inner + ch) * v;
            to.* = sum;
        }
    }
}

/// What a run leaves: every layer's output rows and committed state, as bits.
const Played = struct {
    out: [fx.layers][]u32,
    state: [fx.layers][]u32,
};

/// `x`'s rows through the layers as `plan` cuts them: each step a window of its first number of rows from the
/// first row not kept yet, of which its second number is kept. `chunked` overrides the chain a window's rows
/// choose.
fn play(arena: Allocator, weights: []const kda.Weights, x: cuda.Buffer, plan: []const [2]usize, chunked: ?bool) !Played {
    const total: usize = @intCast(x.view.size[0]);
    var most: usize = 1;
    for (plan) |step| most = @max(most, step[0]);
    var state: kda.State = try .init(weights.len, weights[0].heads, weights[0].proj.n, @max(most, kda.recurrent_rows + 1));
    defer state.deinit();
    state.chunked = chunked;
    var outs: [fx.layers]cuda.Buffer = undefined;
    for (&outs) |*out| {
        out.* = try .init(.f32, &.{ total, fx.dims });
        try out.zero();
    }
    defer for (&outs) |*out| out.deinit();
    var at: usize = 0;
    for (plan) |step| {
        for (weights, 0..) |layer, index| try state.forward(layer, index, at, x.span(at, step[0]), outs[index].span(at, step[0]));
        try state.commit(step[0], step[1]);
        at += step[1];
    }
    // a plan may stop short of `x`'s rows (a mark's state); the rows after it are zeros
    try testing.expect(at <= total);
    var played: Played = undefined;
    for (0..fx.layers) |index| {
        played.out[index] = try shared.fetch(u32, arena, outs[index]);
        played.state[index] = try arena.alloc(u32, weights[0].heads * kda.head_dim * kda.head_dim);
        try cuda.download(state.committed(index), std.mem.sliceAsBytes(played.state[index]));
    }
    return played;
}

fn expectSame(want: Played, got: Played) !void {
    for (0..fx.layers) |index| {
        try testing.expectEqualSlices(u32, want.out[index], got.out[index]);
        try testing.expectEqualSlices(u32, want.state[index], got.state[index]);
    }
}

/// Every value finite, and not all of them zero.
fn expectLive(bits: []const u32) !void {
    var any = false;
    for (bits) |one| {
        const v: f32 = @bitCast(one);
        try testing.expect(std.math.isFinite(v));
        any = any or v != 0;
    }
    try testing.expect(any);
}

/// The ranks' sum against the fp64 layer, of the largest output: bf16 roundings that fall the other way.
const host_bound = 1.5e-3;

const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    pool: cuda.Scratch,
    c: config_file.Config,
    source: Source,
    x: cuda.Buffer,

    fn init(self: *Fixture, seed: u64, rows: usize) !void {
        self.arena_state = .init(testing.allocator);
        const arena = self.arena_state.allocator();
        self.pool = .init(testing.allocator);
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        self.c = try fx.config(arena);
        self.source = .empty;
        for (0..fx.layers) |layer| try fx.putLayer(arena, random, &self.source, layer);
        const x = try arena.alloc(u16, rows * fx.dims);
        shared.randomBf16(random, x, 1.0);
        self.x = try device(u16, .bf16, &.{ rows, fx.dims }, x);
    }

    fn deinit(self: *Fixture) void {
        self.x.deinit();
        self.pool.deinit();
        self.arena_state.deinit();
    }

    /// The layers as `rank` holds them.
    fn load(self: *Fixture, rank: usize, rows: usize) ![fx.layers]kda.Weights {
        var weights: [fx.layers]kda.Weights = undefined;
        for (&weights, 0..) |*one, layer| one.* = try .load(self.arena_state.allocator(), testing.allocator, self.source, &self.c, layer, rank, .{ .pool = &self.pool, .count = rows });
        return weights;
    }
};

test "a window, chunks, a window cut short and serial steps leave the same rows and the same state" {
    const rows = 9;
    var f: Fixture = undefined;
    try f.init(21, rows);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    var weights = try f.load(1, rows);
    defer for (&weights) |*one| one.deinit();

    const steps: [rows][2]usize = @splat(.{ 1, 1 });
    const serial = try play(arena, &weights, f.x, &steps, null);
    for (serial.out, serial.state) |out, state| {
        try expectLive(out);
        try expectLive(state);
    }
    try expectSame(serial, try play(arena, &weights, f.x, &.{.{ rows, rows }}, null));
    try expectSame(serial, try play(arena, &weights, f.x, &.{ .{ 4, 4 }, .{ 5, 5 } }, null));
    try expectSame(serial, try play(arena, &weights, f.x, &.{ .{ 2, 2 }, .{ 1, 1 }, .{ 6, 6 } }, null));
    // a draft's window of which the first rows are kept: one, two (inside the conv's reach), most
    try expectSame(serial, try play(arena, &weights, f.x, &.{ .{ 5, 1 }, .{ 8, 8 } }, null));
    try expectSame(serial, try play(arena, &weights, f.x, &.{ .{ 5, 2 }, .{ 7, 3 }, .{ 4, 4 } }, null));
    try expectSame(serial, try play(arena, &weights, f.x, &.{ .{ 9, 8 }, .{ 1, 1 } }, null));
}

test "the chain and the replay by rows of four floats (ZSPARK_GLM_KDA_QUADS) are the base's bits, a row alone too" {
    const rows = 9;
    var f: Fixture = undefined;
    try f.init(23, rows);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    var weights = try f.load(1, rows);
    defer for (&weights) |*one| one.deinit();

    const steps: [rows][2]usize = @splat(.{ 1, 1 });
    const base = try play(arena, &weights, f.x, &steps, null);
    const base_cut = try play(arena, &weights, f.x, &.{ .{ 5, 2 }, .{ 7, 3 }, .{ 4, 4 } }, null);
    try expectSame(base, base_cut);
    kda.setQuads(true);
    defer kda.setQuads(false);
    // serial steps, a window, and windows cut short, which the replay finishes
    try expectSame(base, try play(arena, &weights, f.x, &steps, null));
    try expectSame(base, try play(arena, &weights, f.x, &.{.{ rows, rows }}, null));
    try expectSame(base, try play(arena, &weights, f.x, &.{ .{ 5, 1 }, .{ 8, 8 } }, null));
    try expectSame(base, try play(arena, &weights, f.x, &.{ .{ 5, 2 }, .{ 7, 3 }, .{ 4, 4 } }, null));
    try expectSame(base, try play(arena, &weights, f.x, &.{ .{ 9, 8 }, .{ 1, 1 } }, null));
}

test "windows up to the recurrent chain's rows are serial steps, a window cut short included" {
    const rows = kda.recurrent_rows;
    var f: Fixture = undefined;
    try f.init(22, rows);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    var weights = try f.load(0, rows);
    defer for (&weights) |*one| one.deinit();

    const steps: [rows][2]usize = @splat(.{ 1, 1 });
    const serial = try play(arena, &weights, f.x, &steps, null);
    try expectSame(serial, try play(arena, &weights, f.x, &.{.{ rows, rows }}, null));
    try expectSame(serial, try play(arena, &weights, f.x, &.{ .{ rows, 33 }, .{ rows - 33, rows - 33 } }, null));
    try expectSame(serial, try play(arena, &weights, f.x, &.{ .{ 3, 3 }, .{ rows - 3, 5 }, .{ rows - 8, rows - 8 } }, null));
}

/// Windows that start `lead` rows into the sequence (after steps of up to `kda.recurrent_rows`), then run the
/// rest of `total` rows chunked, cut where `cuts` (absolute positions) say.
fn cutPlan(arena: Allocator, lead: usize, total: usize, cuts: []const usize) ![]const [2]usize {
    var plan: std.ArrayList([2]usize) = .empty;
    var at: usize = 0;
    while (at < lead) {
        const n = @min(kda.recurrent_rows, lead - at);
        try plan.append(arena, .{ n, n });
        at += n;
    }
    for (cuts) |cut| {
        try plan.append(arena, .{ cut - at, cut - at });
        at = cut;
    }
    try plan.append(arena, .{ total - at, total - at });
    return plan.items;
}

test "a chunked window cut at chunks' starts leaves the same rows and state as uncut, from any position" {
    const rows = 63 + 200;
    var f: Fixture = undefined;
    try f.init(24, rows);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    var weights = try f.load(0, rows);
    defer for (&weights) |*one| one.deinit();

    // a lead of 0, 5 and 63 rows: the first chunk whole, short, one row; every piece after the lead is chunked
    const Case = struct { lead: usize, cuts: []const []const usize };
    const cases = [_]Case{
        .{ .lead = 0, .cuts = &.{ &.{64}, &.{128}, &.{ 64, 128 } } },
        .{ .lead = 5, .cuts = &.{ &.{64}, &.{128}, &.{ 64, 128 } } },
        .{ .lead = 63, .cuts = &.{ &.{128}, &.{192}, &.{ 128, 192 } } },
    };
    for (cases) |case| {
        const total = case.lead + 200;
        const x: cuda.Buffer = .{ .view = f.x.span(0, total) };
        const whole = try play(arena, &weights, x, try cutPlan(arena, case.lead, total, &.{}), null);
        for (whole.out, whole.state) |out, state| {
            try expectLive(out);
            try expectLive(state);
        }
        for (case.cuts) |cuts| {
            for (cuts) |cut| try testing.expect(cut % kda.chunk == 0);
            try expectSame(whole, try play(arena, &weights, x, try cutPlan(arena, case.lead, total, cuts), null));
        }
    }
}

/// The largest difference of `got` from `want` against `want`'s largest magnitude.
fn apart(want: []const u32, got: []const u32) [2]f64 {
    var worst: f64 = 0;
    var scale: f64 = 0;
    for (want, got) |a, b| {
        const w: f64 = @as(f32, @bitCast(a));
        worst = @max(worst, @abs(@as(f64, @as(f32, @bitCast(b))) - w));
        scale = @max(scale, @abs(w));
    }
    return .{ worst, scale };
}

/// The chunked chain against the recurrent fp32 one over a window: bf16 operands in its products (FLA's), the
/// largest output and state difference against their largest magnitude.
const chunked_bound = 2e-2;

test "the chunked chain is the recurrent fp32 chain's within a bound" {
    const rows = 5 + 200;
    var f: Fixture = undefined;
    try f.init(25, rows);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    var weights = try f.load(1, rows);
    defer for (&weights) |*one| one.deinit();

    const plan = try cutPlan(arena, 5, rows, &.{});
    const recurrent = try play(arena, &weights, f.x, plan, false);
    const chunked = try play(arena, &weights, f.x, plan, null);
    for (0..fx.layers) |index| {
        const out = apart(recurrent.out[index], chunked.out[index]);
        const state = apart(recurrent.state[index], chunked.state[index]);
        std.debug.print("kda chunked against recurrent, layer {d}: out {e} of {e}, state {e} of {e}\n", .{ index, out[0], out[1], state[0], state[1] });
        try testing.expect(out[0] <= chunked_bound * out[1] and state[0] <= chunked_bound * state[1]);
        try testing.expect(out[1] > 1e-3 and state[1] > 1e-3);
    }
}

test "a mark asked at a chunk's start inside a chunked window is the state of the window cut there" {
    const lead = 5;
    const rows = 200;
    const total = lead + rows;
    var f: Fixture = undefined;
    try f.init(26, total);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    var weights = try f.load(0, total);
    defer for (&weights) |*one| one.deinit();

    var state: kda.State = try .init(fx.layers, weights[0].heads, weights[0].proj.n, rows);
    defer state.deinit();
    const out: cuda.Buffer = try .init(.f32, &.{ total, fx.dims });
    defer {
        var one = out;
        one.deinit();
    }
    for (0..lead) |at| {
        for (&weights, 0..) |layer, index| try state.forward(layer, index, at, f.x.span(at, 1), out.span(at, 1));
        try state.commit(1, 1);
    }
    // asked before the window runs, which writes their states as it passes them
    var marks: [2]kda.Mark = undefined;
    var taken: usize = 0;
    defer for (marks[0..taken]) |*one| one.deinit();
    for ([_]usize{ 64, 128 }, &marks) |at, *mark| {
        mark.* = try .init(fx.layers, weights[0].heads);
        taken += 1;
        state.ask(at - lead, mark.*);
    }
    for (&weights, 0..) |layer, index| try state.forward(layer, index, lead, f.x.span(lead, rows), out.span(lead, rows));
    try testing.expect(state.window.chunked);
    for ([_]usize{ 64, 128 }, marks) |at, mark| {
        try testing.expect(state.wrote(at - lead, mark));
        try mark.convWithin(state, at - lead);
    }
    try state.commit(rows, rows);
    // the window that wrote them is the window that wrote none
    const whole = try play(arena, &weights, f.x, try cutPlan(arena, lead, total, &.{}), null);
    try testing.expectEqualSlices(u32, whole.out[fx.layers - 1], try shared.fetch(u32, arena, out));
    for (0..fx.layers) |index| {
        const got = try arena.alloc(u32, weights[0].heads * kda.head_dim * kda.head_dim);
        try cuda.download(state.committed(index), std.mem.sliceAsBytes(got));
        try testing.expectEqualSlices(u32, whole.state[index], got);
    }
    for ([_]usize{ 64, 128 }, marks) |at, mark| {
        const cut = try play(arena, &weights, f.x, try cutPlan(arena, lead, at, &.{}), null);
        for (0..fx.layers) |index| {
            const got = try arena.alloc(u32, weights[0].heads * kda.head_dim * kda.head_dim);
            try cuda.download(mark.committed(index), std.mem.sliceAsBytes(got));
            try testing.expectEqualSlices(u32, cut.state[index], got);
        }
    }
}

test "two ranks hold the whole layer's heads: the same states, partials that sum to its output" {
    const rows = 6;
    var f: Fixture = undefined;
    try f.init(23, rows);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    var all: [fx.layers]kda.Weights = undefined;
    for (&all, 0..) |*one, layer| one.* = try fx.whole(arena, f.source, &f.c, layer, .{ .pool = &f.pool, .count = rows });
    defer for (&all) |*one| one.deinit();
    try testing.expectEqual(@as(usize, fx.heads), all[0].heads);
    const want = try play(arena, &all, f.x, &.{ .{ 4, 4 }, .{ 2, 2 } }, null);

    const sum = try arena.alloc(f64, rows * fx.dims);
    const one_head = kda.head_dim * kda.head_dim;
    var parts: [layout.world]Played = undefined;
    for (&parts, 0..) |*part, rank| {
        var weights = try f.load(rank, rows);
        defer for (&weights) |*one| one.deinit();
        try testing.expectEqual(@as(usize, fx.heads / 2), weights[0].heads);
        // the projections split K alike, so a head's rows are the whole layer's to the bit
        try testing.expectEqual(all[0].proj.plain().?.sk, weights[0].proj.plain().?.sk);
        part.* = try play(arena, &weights, f.x, &.{ .{ 4, 4 }, .{ 2, 2 } }, null);
        for (0..fx.layers) |index| {
            const first = rank * (fx.heads / 2) * one_head;
            try testing.expectEqualSlices(u32, want.state[index][first..][0 .. (fx.heads / 2) * one_head], part.state[index]);
        }
    }
    const x = try shared.fetch(u16, arena, f.x);
    const modelled = try arena.alloc(f64, rows * fx.dims);
    var worst: f64 = 0;
    var scale: f64 = 0;
    var off: f64 = 0;
    for (0..fx.layers) |index| {
        @memset(sum, 0);
        for (parts) |part| for (sum, part.out[index]) |*to, bits| {
            to.* += @as(f32, @bitCast(bits));
        };
        try hostLayer(arena, f.source, &f.c, index, x, rows, modelled);
        for (sum, modelled) |got, model| off = @max(off, @abs(got - model));
        for (sum, want.out[index]) |got, bits| {
            const whole: f64 = @as(f32, @bitCast(bits));
            worst = @max(worst, @abs(got - whole));
            scale = @max(scale, @abs(whole));
        }
    }
    std.debug.print("kda ranks against the whole layer: {e} of {e}; against the host: {e}\n", .{ worst, scale, off });
    try testing.expect(off <= host_bound * scale);
    try testing.expect(scale > 1e-3);
    try testing.expect(worst <= 2e-5 * scale);
}
