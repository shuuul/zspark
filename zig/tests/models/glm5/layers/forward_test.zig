//! `forward.zig`'s and `loopback.zig`'s tests on a random GLM: five layers of both attentions and both MLPs at sizes the kernels
//! take, two shares in loopback, no checkpoint. A window at every row count against serial steps bit for bit,
//! windows cut short, captured graphs against the eager forward, and a first window layer by layer: every
//! site against the host's glue and every branch against its layer's weights loaded alone.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const config_file = @import("../../../../src/models/glm5/layers/config.zig");
const dsa = @import("../../../../src/models/glm5/layers/dsa.zig");
const forward = @import("../../../../src/models/glm5/layers/forward.zig");
const host = @import("../../../../src/models/glm5/layers/glue_ref.zig");
const kda = @import("../../../../src/models/glm5/layers/kda.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const loopback = @import("../../../../src/models/glm5/layers/loopback.zig");
const mlp = @import("../../../../src/models/glm5/layers/mlp.zig");
const random_file = @import("../../../../src/models/glm5/checks/random.zig");
const ranks_forward = @import("../../../../src/models/glm5/layers/ranks_forward.zig");
const shared = @import("../testing.zig");
const Source = random_file.Source;

const fx = random_file.fx;
const Fixture = shared.Fixture;
const Loop = shared.Loop;

const Played = struct {
    /// bf16 [total, vocab]: a row's logits, rank 0's half and then rank 1's.
    logits: []u16,
    /// Each rank's committed KDA states and latents.
    states: [layout.world][]u32,
    latents: [layout.world][]u16,
    /// Where the buffers keep the drafter's taps: bf16 [total, taps x dims] a rank, a kept row's. Else empty.
    taps: [layout.world][]u16 = @splat(&.{}),
    /// With taps: bf16 [total, dims], rank 0's mean of the streams after the last layer.
    hidden: []u16 = &.{},
};

/// The fixture's ids through `loop` from an empty sequence as `plan` cuts them: each step a window of its first
/// number of rows from the first row not kept yet, of which its second number is kept. Through `graphs` if given.
fn play(arena: Allocator, loop: *Loop, graphs: ?*loopback.Graphs, tokens: []const i32, plan: []const [2]usize) !Played {
    const half = fx.vocab / layout.world;
    var played: Played = undefined;
    played.logits = try arena.alloc(u16, tokens.len * fx.vocab);
    const rows_of = try arena.alloc(u16, fx.max_rows * half);
    const wide = fx.dims * loop.b.ranks[0].tap_layers.len;
    played.taps = @splat(&.{});
    played.hidden = &.{};
    if (wide > 0) {
        for (&played.taps) |*taps| taps.* = try arena.alloc(u16, tokens.len * wide);
        played.hidden = try arena.alloc(u16, tokens.len * fx.dims);
    }
    try loop.state.reset();
    var at: usize = 0;
    for (plan) |step| {
        const window = tokens[at..][0..step[0]];
        if (graphs) |g| try g.forward(loop.m, &loop.state, &loop.b, window) else try loopback.forward(loop.m, &loop.state, &loop.b, window);
        try cuda.synchronize();
        for (loop.b.ranks, 0..) |b, rank| {
            try cuda.download(b.logits.rows(step[1]), std.mem.sliceAsBytes(rows_of[0 .. step[1] * half]));
            for (0..step[1]) |r| @memcpy(played.logits[(at + r) * fx.vocab + rank * half ..][0..half], rows_of[r * half ..][0..half]);
            if (wide > 0) try cuda.download(b.taps.?.rows(step[1]), std.mem.sliceAsBytes(played.taps[rank][at * wide ..][0 .. step[1] * wide]));
        }
        if (wide > 0) try cuda.download(loop.b.ranks[0].hidden.rows(step[1]), std.mem.sliceAsBytes(played.hidden[at * fx.dims ..][0 .. step[1] * fx.dims]));
        try loopback.commit(loop.m, &loop.state, loop.b, window, step[1]);
        at += step[1];
    }
    try testing.expectEqual(tokens.len, at);
    try cuda.synchronize();
    for (loop.state.ranks, 0..) |state, rank| {
        const chains = state.kda.?;
        const one = chains.heads * kda.head_dim * kda.head_dim;
        played.states[rank] = try arena.alloc(u32, chains.layers * one);
        for (0..chains.layers) |index| try cuda.download(chains.committed(index), std.mem.sliceAsBytes(played.states[rank][index * one ..][0..one]));
        const cache = state.dsa.?;
        played.latents[rank] = try arena.alloc(u16, cache.layers * tokens.len * dsa.latent);
        for (0..cache.layers) |index| {
            var kept = cache.layer(index);
            kept.size[0] = @intCast(tokens.len);
            try cuda.download(kept, std.mem.sliceAsBytes(played.latents[rank][index * tokens.len * dsa.latent ..][0 .. tokens.len * dsa.latent]));
        }
    }
    return played;
}

fn expectSame(want: Played, got: Played) !void {
    try testing.expectEqualSlices(u16, want.logits, got.logits);
    for (0..layout.world) |rank| {
        try testing.expectEqualSlices(u32, want.states[rank], got.states[rank]);
        try testing.expectEqualSlices(u16, want.latents[rank], got.latents[rank]);
        // where both kept taps
        if (want.taps[rank].len > 0 and got.taps[rank].len > 0) try testing.expectEqualSlices(u16, want.taps[rank], got.taps[rank]);
    }
}

/// Every logit finite, the rows not all alike, and both ranks' latents the same.
fn expectLive(played: Played) !void {
    for (played.logits) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
    try testing.expect(!std.mem.eql(u16, played.logits[0..fx.vocab], played.logits[played.logits.len - fx.vocab ..]));
    for (played.states) |state| for (state) |bits| try testing.expect(std.math.isFinite(@as(f32, @bitCast(bits))));
    try testing.expectEqualSlices(u16, played.latents[0], played.latents[1]);
}

/// `total` rows in windows of `rows`, the last one shorter, all kept.
fn windows(arena: Allocator, total: usize, rows: usize) ![]const [2]usize {
    var plan: std.ArrayList([2]usize) = .empty;
    var at: usize = 0;
    while (at < total) : (at += @min(rows, total - at)) try plan.append(arena, @splat(@min(rows, total - at)));
    return plan.items;
}

test "a window at every row count to the maximum, and a window's rows alone, give the serial steps' logits and state" {
    // held to serial steps: one form, a step's (`testing.oneForm`)
    const forms = shared.oneForm();
    defer forms.restore();
    // windows of 1, 2, ... 80 rows: 3240 positions, over six of DSA's chunks
    const total = fx.max_rows * (fx.max_rows + 1) / 2;
    var fixture: Fixture = undefined;
    try fixture.init(51, total);
    defer fixture.deinit();
    const arena = fixture.arena_state.allocator();
    cuda.Scratch.poisoned = true;
    defer cuda.Scratch.poisoned = false;
    var loop = try fixture.loop(total);
    defer loop.deinit();

    const growing = try arena.alloc([2]usize, fx.max_rows);
    for (growing, 1..) |*step, rows| step.* = @splat(rows);
    const serial = try play(arena, &loop, null, fixture.tokens, try windows(arena, total, 1));
    try expectLive(serial);
    try expectSame(serial, try play(arena, &loop, null, fixture.tokens, growing));
    try expectSame(serial, try play(arena, &loop, null, fixture.tokens, try windows(arena, total, fx.max_rows)));
}

test "the drafter's taps change no bit of the forward; a row's taps alone are its taps in a window and in a graph" {
    // held to serial steps: one form, a step's (`testing.oneForm`)
    const forms = shared.oneForm();
    defer forms.restore();
    const total = 400;
    var fixture: Fixture = undefined;
    try fixture.init(67, total);
    defer fixture.deinit();
    const arena = fixture.arena_state.allocator();
    cuda.Scratch.poisoned = true;
    defer cuda.Scratch.poisoned = false;
    var plain = try fixture.loop(total + fx.max_rows);
    defer plain.deinit();
    var tapped = try fixture.loop(total + fx.max_rows);
    defer tapped.deinit();
    // after the second layer and after the last, whose tap is the mean the head reads
    const layers = [_]u32{ 1, fx.kinds.len - 1 };
    try tapped.b.setTaps(&layers);
    const wide = layers.len * fx.dims;

    const whole = try windows(arena, total, fx.max_rows);
    const off = try play(arena, &plain, null, fixture.tokens, whole);
    try testing.expectEqual(@as(usize, 0), off.taps[0].len);
    const on = try play(arena, &tapped, null, fixture.tokens, whole);
    try expectLive(on);
    try expectSame(off, on);
    // both ranks hold the same taps; the last layer's is the head's mean, the other is not
    try testing.expectEqualSlices(u16, on.taps[0], on.taps[1]);
    for (0..total) |r| {
        const row = on.taps[0][r * wide ..][0..wide];
        try testing.expectEqualSlices(u16, on.hidden[r * fx.dims ..][0..fx.dims], row[fx.dims..]);
        try testing.expect(!std.mem.eql(u16, row[0..fx.dims], row[fx.dims..]));
        for (row) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
    }

    // a row alone, windows cut short, and both through graphs captured with the taps in them
    try expectSame(on, try play(arena, &tapped, null, fixture.tokens, try windows(arena, total, 1)));
    var cut: std.ArrayList([2]usize) = .empty;
    var at: usize = 0;
    var keep: usize = 1;
    while (at < total) : (keep = keep % 29 + 1) {
        const rows = @min(fx.max_rows, total - at);
        try cut.append(arena, .{ rows, @min(keep, rows) });
        at += @min(keep, rows);
    }
    try expectSame(on, try play(arena, &tapped, null, fixture.tokens, cut.items));
    var graphs: loopback.Graphs = try .init(testing.allocator);
    defer graphs.deinit();
    try expectSame(on, try play(arena, &tapped, &graphs, fixture.tokens, cut.items));
    try expectSame(on, try play(arena, &tapped, &graphs, fixture.tokens, cut.items));
}

test "windows cut short and captured graphs give the eager windows' bits, twice over; both ranks hold the same streams" {
    // held to serial steps: one form, a step's (`testing.oneForm`)
    const forms = shared.oneForm();
    defer forms.restore();
    const total = 600;
    var fixture: Fixture = undefined;
    try fixture.init(52, total);
    defer fixture.deinit();
    const arena = fixture.arena_state.allocator();
    cuda.Scratch.poisoned = true;
    defer cuda.Scratch.poisoned = false;
    // room for the rows a cut window ran and did not keep
    var loop = try fixture.loop(total + fx.max_rows);
    defer loop.deinit();

    const whole = try play(arena, &loop, null, fixture.tokens, try windows(arena, total, fx.max_rows));
    try expectLive(whole);
    try expectSame(whole, try play(arena, &loop, null, fixture.tokens, try windows(arena, total, fx.max_rows)));

    // windows of which 1, 2, ... rows are kept, then of every size as the sequence's end allows
    var cut: std.ArrayList([2]usize) = .empty;
    var at: usize = 0;
    var keep: usize = 1;
    while (at < total) : (keep = keep % 37 + 1) {
        const rows = @min(fx.max_rows, total - at);
        try cut.append(arena, .{ rows, @min(keep, rows) });
        at += @min(keep, rows);
    }
    try expectSame(whole, try play(arena, &loop, null, fixture.tokens, cut.items));

    var graphs: loopback.Graphs = try .init(testing.allocator);
    defer graphs.deinit();
    try expectSame(whole, try play(arena, &loop, &graphs, fixture.tokens, cut.items));
    const captured = graphs.cache.count();
    try expectSame(whole, try play(arena, &loop, &graphs, fixture.tokens, cut.items));
    try testing.expectEqual(captured, graphs.cache.count());

    // a window from an empty sequence: held as a graph, its rows read one by one, its last row scored alone
    try loop.state.reset();
    const window = fixture.tokens[0..fx.max_rows];
    try testing.expect(graphs.holds(loop.state, loop.b, window.len));
    try testing.expect(!graphs.holds(loop.state, loop.b, 39));
    try loopback.forward(loop.m, &loop.state, &loop.b, window);
    const streams = try shared.fetch(u16, arena, loop.b.ranks[0].x);
    try testing.expectEqualSlices(u16, streams, try shared.fetch(u16, arena, loop.b.ranks[1].x));
    const row = try arena.alloc(u16, fx.vocab);
    const last = try arena.alloc(u16, fx.vocab);
    for ([_]usize{ 0, 41, window.len - 1 }) |r| {
        try loopback.logitsRow(loop.b, r, row);
        try testing.expectEqualSlices(u16, whole.logits[r * fx.vocab ..][0..fx.vocab], row);
    }
    try loopback.forwardScoring(loop.m, &loop.state, &loop.b, window, .last);
    try loopback.logitsRow(loop.b, 0, last);
    try testing.expectEqualSlices(u16, row, last);
    const finals = try shared.fetch(u16, arena, loop.b.ranks[1].final);
    try loopback.forwardScoring(loop.m, &loop.state, &loop.b, window, .none);
    try testing.expectEqualSlices(u16, finals, try shared.fetch(u16, arena, loop.b.ranks[0].final));
}

fn val(bits: u16) f64 {
    return numeric.f32FromBf16(bits);
}

/// A site of layer `layer` as the checkpoint names it, for the host's glue.
fn hostSite(arena: Allocator, source: Source, c: *const config_file.Config, layer: usize, comptime which: []const u8, comptime norm: []const u8) !struct { dots: []const u16, site: host.Site } {
    const p = try std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.", .{layer});
    const get = struct {
        fn get(a: Allocator, s: Source, under: []const u8, leaf: []const u8) ![]const u8 {
            return s.get(try std.mem.concat(a, u8, &.{ under, leaf })).?.data;
        }
    }.get;
    const copy = struct {
        fn copy(comptime T: type, a: Allocator, bytes: []const u8) ![]T {
            const values = try a.alloc(T, bytes.len / @sizeOf(T));
            @memcpy(std.mem.sliceAsBytes(values), bytes);
            return values;
        }
    }.copy;
    return .{
        .dots = try copy(u16, arena, try get(arena, source, p, "hc_" ++ which ++ "_fn")),
        .site = .{
            .base = try copy(f32, arena, try get(arena, source, p, "hc_" ++ which ++ "_base")),
            .scale = try copy(f32, arena, try get(arena, source, p, "hc_" ++ which ++ "_scale")),
            .norm = switch (source.get(try std.mem.concat(arena, u8, &.{ p, norm ++ ".weight" })).?.dtype) {
                .F32 => .{ .f32 = try copy(f32, arena, try get(arena, source, p, norm ++ ".weight")) },
                else => .{ .bf16 = try copy(u16, arena, try get(arena, source, p, norm ++ ".weight")) },
            },
            .eps = @floatCast(c.eps),
            .hc_eps = @floatCast(c.hc_eps),
            .iters = c.sinkhorn,
        },
    };
}

/// Holds one site of the window against the host: rank 0's read-out of the streams `x` within a rounding, and
/// after the write-back of `gathered` with the device's gates the streams' bits, which `x` then is.
const Glue = struct {
    arena: Allocator,
    rows: usize,
    /// bf16 [rows, 4 dims], the host's streams
    x: []u16,

    fn readOut(self: Glue, b: forward.Buffers, dots: []const u16, site: host.Site) !void {
        const normed = try shared.fetch(u16, self.arena, b.normed);
        const part = try self.arena.alloc(f32, host.parts * host.entry);
        const out = try self.arena.alloc(u16, fx.dims);
        const scratch = try self.arena.alloc(f32, fx.dims);
        var post: [host.streams]f32 = undefined;
        var comb: [host.streams * host.streams]f32 = undefined;
        for (0..self.rows) |r| {
            const row = self.x[r * host.streams * fx.dims ..][0 .. host.streams * fx.dims];
            host.hcMix(row, dots, part);
            host.hcPre(row, part, site, out, &post, &comb, scratch);
            // a value the streams cancel in is held to the row's size, not to its own
            var size: f64 = 0;
            for (out) |want| size += val(want) * val(want);
            const rms = @sqrt(size / fx.dims);
            try testing.expect(rms > 0.05);
            for (out, normed[r * fx.dims ..][0..fx.dims]) |want, got| try testing.expectApproxEqAbs(val(want), val(got), @abs(val(want)) / 64.0 + rms / 1024.0);
        }
    }

    fn writeBack(self: Glue, b: forward.Buffers, gathered: cuda.Buffer) !void {
        const post = try shared.fetch(f32, self.arena, b.post);
        const comb = try shared.fetch(f32, self.arena, b.comb);
        const partials = try shared.fetch(f32, self.arena, gathered);
        for (0..self.rows) |r| {
            var ranks: [layout.world][]const f32 = undefined;
            for (&ranks, 0..) |*one, rank| one.* = partials[(rank * self.rows + r) * fx.dims ..][0..fx.dims];
            host.hcPost(self.x[r * host.streams * fx.dims ..][0 .. host.streams * fx.dims], &ranks, post[r * host.streams ..][0..host.streams], comb[r * host.streams * host.streams ..][0 .. host.streams * host.streams]);
        }
    }

    fn expectStreams(self: Glue, b: forward.Buffers) !void {
        const got = try shared.fetch(u16, self.arena, b.x);
        try testing.expectEqualSlices(u16, self.x, got[0..self.x.len]);
    }
};

test "a first window layer by layer: each site is its layer's on the host, each branch its layer's weights alone, the head its rank's rows" {
    const rows = 3;
    var fixture: Fixture = undefined;
    try fixture.init(53, rows);
    defer fixture.deinit();
    const arena = fixture.arena_state.allocator();
    const gpa = testing.allocator;
    const source = fixture.source;
    const c = &fixture.c;
    const capacity = 16;
    var loop = try fixture.loop(capacity);
    defer loop.deinit();
    // the window's form, as `loopback.forward` sets it: bf16 rows and fp32 partials, which the host's write-back reads
    for (loop.m.ranks) |m| forward.setForm(m, forward.quantizedFrom(rows));
    try loop.state.reset();
    try loopback.stage(&loop.b, fixture.tokens);
    for (loop.m.ranks, &loop.b.ranks) |m, *b| try forward.embed(m, b, rows);

    const table: []align(1) const u16 = @ptrCast(source.get(layout.prefix ++ "embed_tokens.weight").?.data);
    var glue: Glue = .{ .arena = arena, .rows = rows, .x = try arena.alloc(u16, rows * host.streams * fx.dims) };
    for (fixture.tokens, 0..) |id, r| for (0..host.streams) |s|
        @memcpy(glue.x[(r * host.streams + s) * fx.dims ..][0..fx.dims], table[@as(usize, @intCast(id)) * fx.dims ..][0..fx.dims]);
    for (loop.b.ranks) |b| try glue.expectStreams(b);

    var alone: cuda.Buffer = try .init(.f32, &.{ rows, fx.dims });
    defer alone.deinit();
    const partial = struct {
        fn of(a: Allocator, gathered: cuda.Buffer, rank: usize) ![]u32 {
            const all = try shared.fetch(u32, a, gathered);
            return all[rank * rows * fx.dims ..][0 .. rows * fx.dims];
        }
    }.of;
    for (fx.kinds, fx.mlps, 0..) |kind, block, index| {
        const attn = try hostSite(arena, source, c, index, "attn", "input_layernorm");
        for (loop.m.ranks, &loop.state.ranks, &loop.b.ranks, 0..) |m, *state, *b, rank| {
            try forward.attention(m, state, b, index, rows, capacity, loopback.partial(loop.b.attn, rank, rows), null);
            try glue.readOut(b.*, attn.dots, attn.site);
            // the layer's own weights, from an empty sequence of their own
            switch (kind) {
                .linear_attention => {
                    var weights: kda.Weights = try .load(arena, gpa, source, c, index, rank, fixture.rows());
                    defer weights.deinit();
                    var fresh: kda.State = try .init(1, weights.heads, weights.proj.n, fx.max_rows);
                    defer fresh.deinit();
                    try fresh.reset();
                    try fresh.forward(weights, 0, 0, b.normed.rows(rows), alone.view);
                },
                .deepseek_sparse_attention => {
                    var weights: dsa.Weights = try .load(arena, source, try std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.self_attn.", .{index}), c, rank, fixture.rows());
                    defer weights.deinit();
                    var cache: dsa.Cache = try weights.cacheFor(1, capacity);
                    defer cache.deinit();
                    var buffers: dsa.Buffers = try .init(weights, capacity, fx.max_rows);
                    defer buffers.deinit();
                    try weights.forward(cache, 0, buffers, b.normed.rows(rows), alone.view);
                },
            }
            try testing.expectEqualSlices(u32, try shared.fetch(u32, arena, alone), try partial(arena, loop.b.attn, rank));
        }
        try glue.writeBack(loop.b.ranks[0], loop.b.attn);

        const ffn = try hostSite(arena, source, c, index, "ffn", "post_attention_layernorm");
        for (loop.m.ranks, &loop.b.ranks, 0..) |m, *b, rank| {
            try forward.mlp(m, b, index, rows, loopback.gathered(loop.b.attn, rows), loopback.partial(loop.b.ffn, rank, rows));
            try glue.expectStreams(b.*);
            try glue.readOut(b.*, ffn.dots, ffn.site);
            switch (block) {
                .dense => {
                    var weights: mlp.Dense = try .load(arena, source, layout.prefix ++ "layers.{d}.mlp.", .{index}, c, rank, fixture.rows());
                    defer weights.deinit();
                    var buffers: mlp.Dense.Buffers = try .init(weights.width, fx.max_rows);
                    defer buffers.deinit();
                    try weights.run(buffers, b.normed.rows(rows), alone.view);
                },
                .sparse => {
                    var weights: mlp.Moe = try .load(arena, gpa, source, c, index, rank, fixture.rows());
                    defer weights.deinit();
                    var buffers: mlp.Moe.Buffers = try .init(weights, fx.max_rows);
                    defer buffers.deinit();
                    try weights.forward(buffers, b.normed.rows(rows), alone.view, m.pool.quantize_from);
                },
            }
            try testing.expectEqualSlices(u32, try shared.fetch(u32, arena, alone), try partial(arena, loop.b.ffn, rank));
        }
        try glue.writeBack(loop.b.ranks[0], loop.b.ffn);
        for (&loop.b.ranks) |*b| {
            try forward.finish(b, rows, loopback.gathered(loop.b.ffn, rows));
            try glue.expectStreams(b.*);
        }
    }

    // the head: the streams' mean and its norm the host's bits, a logit the product with its row of lm_head
    const norm = try arena.alloc(f32, fx.dims);
    @memcpy(std.mem.sliceAsBytes(norm), source.get(layout.prefix ++ "norm.weight").?.data);
    const lm_head: []align(1) const u16 = @ptrCast(source.get("lm_head.weight").?.data);
    const hidden = try arena.alloc(u16, fx.dims);
    const final = try arena.alloc(u16, fx.dims);
    const scratch = try arena.alloc(f32, fx.dims);
    const half = fx.vocab / layout.world;
    var phased: [layout.world][]u16 = undefined;
    for (loop.m.ranks, &loop.b.ranks, 0..) |m, *b, rank| {
        try forward.head(m, b, rows, .all);
        const got_hidden = try shared.fetch(u16, arena, b.hidden);
        const got_final = try shared.fetch(u16, arena, b.final);
        const logits = try shared.fetch(u16, arena, b.logits);
        for (0..rows) |r| {
            host.streamMean(glue.x[r * host.streams * fx.dims ..][0 .. host.streams * fx.dims], hidden);
            host.rmsnorm(hidden, .{ .f32 = norm }, final, @floatCast(c.eps), scratch);
            try testing.expectEqualSlices(u16, hidden, got_hidden[r * fx.dims ..][0..fx.dims]);
            try testing.expectEqualSlices(u16, final, got_final[r * fx.dims ..][0..fx.dims]);
            var size: f64 = 0;
            const want = try arena.alloc(f64, half);
            for (want, 0..) |*one, j| {
                one.* = 0;
                for (final, 0..) |f, d| one.* += val(f) * val(lm_head[(rank * half + j) * fx.dims + d]);
                size += one.* * one.*;
            }
            const rms = @sqrt(size / half);
            try testing.expect(rms > 0.1);
            for (want, logits[r * half ..][0..half]) |w, g| try testing.expectApproxEqAbs(w, val(g), 1e-3 * rms + @abs(w) / 128.0);
        }
        phased[rank] = logits;
    }

    // and the loopback's own order of the phases gives these bits
    try loop.state.reset();
    try loopback.forward(loop.m, &loop.state, &loop.b, fixture.tokens);
    for (loop.b.ranks, phased) |b, want| try testing.expectEqualSlices(u16, want, try shared.fetch(u16, arena, b.logits));
}

test "a window's glue split by rows over the ranks gives the whole glue's bits (spec 107 WS-03)" {
    // held to serial steps: one form, a step's (`testing.oneForm`)
    const forms = shared.oneForm();
    defer forms.restore();
    const total = 80 + 33 + 17 + 35 + 5;
    var fixture: Fixture = undefined;
    try fixture.init(107, total);
    defer fixture.deinit();
    const arena = fixture.arena_state.allocator();
    var loop = try fixture.loop(total);
    defer loop.deinit();
    for (&loop.m.ranks) |*rank| rank.bf16_partials = false;
    const saved = ranks_forward.split_rows;
    defer ranks_forward.split_rows = saved;
    // an even cut; odd row counts, which leave rank 1 one row or an odd 3 from a cut of 16; a window under it
    const plan = [_][2]usize{ @splat(80), @splat(33), @splat(17), @splat(35), @splat(5) };
    ranks_forward.split_rows = std.math.maxInt(usize);
    try testing.expect(!loopback.splits(loop.b, 80));
    const whole = try play(arena, &loop, null, fixture.tokens, &plan);
    try expectLive(whole);
    const serial = try play(arena, &loop, null, fixture.tokens, try windows(arena, total, 1));
    ranks_forward.split_rows = 17;
    for (plan[0..4]) |step| try testing.expect(loopback.splits(loop.b, step[0]));
    try testing.expect(!loopback.splits(loop.b, 5) and !loopback.splits(loop.b, 16));
    const split = try play(arena, &loop, null, fixture.tokens, &plan);
    try expectSame(whole, split);
    try expectSame(serial, split);

    // bf16 partials in the quantized windows (FP4-Local's, spec 116 WS-04), the card's forms: split, the whole's
    // bits again
    forward.bf16_rows = forward.card_bf16_rows;
    for (&loop.m.ranks) |*rank| rank.bf16_partials = true;
    defer for (&loop.m.ranks) |*rank| {
        rank.bf16_partials = false;
    };
    const half_split = try play(arena, &loop, null, fixture.tokens, &plan);
    ranks_forward.split_rows = std.math.maxInt(usize);
    const half_whole = try play(arena, &loop, null, fixture.tokens, &plan);
    try expectSame(half_whole, half_split);
    try expectLive(half_whole);
    try testing.expect(!std.mem.eql(u16, whole.logits, half_whole.logits));
}
