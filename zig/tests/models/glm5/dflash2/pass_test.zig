//! `pass.zig`'s tests on a random drafter: a context update and a block pass against the same steps here (the
//! kernels' host models, and every linear a dot in f64 rounded where the device rounds), which holds the
//! wiring of `_taps_compute` and `_layer`; and the bits the passes owe: a context row's K and V the same
//! alone, in a batch and through the ring's wrap, a block the same twice, flat and in the ring, and the same
//! rows on both ranks. The published drafter's: `pass_checkpoint_test.zig`.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const cuda = @import("../../../../src/cuda/cuda.zig");
const bf16 = @import("../../../../src/weights/bf16.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const glm5_testing = @import("../testing.zig");
const glue = @import("../../../../src/models/glm5/layers/glue_ref.zig");
const config_file = @import("../../../../src/decode/dflash2/config.zig");
const drafter_testing = @import("../../../../src/models/glm5/checks/random_drafter.zig");
const host = @import("../../../../src/decode/dflash2/kernels_ref.zig");
const kernels = @import("../../../../src/decode/dflash2/kernels.zig");
const pass = @import("../../../../src/models/glm5/dflash2/pass.zig");
const dflash2_pass = @import("../../../../src/decode/dflash2/pass.zig");
const weights = @import("../../../../src/decode/dflash2/weights.zig");
const pass_fixture = @import("pass_fixture.zig");
const drafts = pass_fixture.drafts;
const value = pass_fixture.value;

const world = config_file.world;
const dims = 512;
const vocab = 128;
const half = vocab / world;
const head = host.head;

/// A random drafter's two shares, and a main model's table and rows of the vocabulary beside it.
const Fx = struct {
    arena_state: *std.heap.ArenaAllocator,
    pool: *cuda.Scratch,
    c: *config_file.Config,
    source: drafter_testing.Source,
    w: [world]weights.Weights,
    table_host: []u16,
    table: cuda.Buffer,
    head_host: []u16,
    heads: [world]bf16.Linear,

    fn init(storage: weights.Storage) !Fx {
        const gpa = testing.allocator;
        const arena_state = try gpa.create(std.heap.ArenaAllocator);
        arena_state.* = .init(gpa);
        const arena = arena_state.allocator();
        const pool = try arena.create(cuda.Scratch);
        pool.* = .init(gpa);
        var prng: std.Random.DefaultPrng = .init(1067);
        const random = prng.random();
        const c = try arena.create(config_file.Config);
        c.* = try drafter_testing.config(arena, dims, vocab, 5, &.{ 1, 3 });
        var source: drafter_testing.Source = .empty;
        try drafter_testing.put(arena, random, c.*, &source);
        const rows: cuda.Scratch.Rows = .{ .pool = pool, .count = dflash2_pass.batch };
        var w: [world]weights.Weights = undefined;
        for (&w, 0..) |*one, rank| one.* = try .load(arena, gpa, source, c, rank, storage, rows);
        const table_host = try arena.alloc(u16, vocab * dims);
        glm5_testing.randomBf16(random, table_host, 1.0);
        const head_host = try arena.alloc(u16, vocab * dims);
        glm5_testing.randomBf16(random, head_host, 1.0 / @sqrt(@as(f32, dims)));
        var heads: [world]bf16.Linear = undefined;
        for (&heads, 0..) |*one, rank| one.* = try .init(std.mem.sliceAsBytes(head_host[rank * half * dims ..][0 .. half * dims]), half, dims, rows);
        return .{ .arena_state = arena_state, .pool = pool, .c = c, .source = source, .w = w, .table_host = table_host, .table = try glm5_testing.device(u16, .bf16, &.{ vocab, dims }, table_host), .head_host = head_host, .heads = heads };
    }

    fn deinit(self: *Fx) void {
        for (&self.w) |*one| one.deinit();
        for (&self.heads) |*one| one.deinit();
        self.table.deinit();
        self.pool.deinit();
        self.arena_state.deinit();
        testing.allocator.destroy(self.arena_state);
    }

    fn main(self: Fx) [world]dflash2_pass.Main {
        return .{ .{ .embed = self.table.view, .head = .{ .bf16 = self.heads[0] }, .rows = self.heads[0].weight.view }, .{ .embed = self.table.view, .head = .{ .bf16 = self.heads[1] }, .rows = self.heads[1].weight.view } };
    }

    fn pair(self: Fx, capacity: usize, ring: bool) !pass.Pair {
        return .init(testing.allocator, self.w, self.c, capacity, ring, half);
    }

    /// `count` rows of taps, the same on both ranks.
    fn taps(self: Fx, seed: u64, count: usize) ![world]cuda.Buffer {
        var prng: std.Random.DefaultPrng = .init(seed);
        const values = try self.arena_state.allocator().alloc(u16, count * self.c.tapWidth());
        glm5_testing.randomBf16(prng.random(), values, 1.0);
        var first = try glm5_testing.device(u16, .bf16, &.{ count, self.c.tapWidth() }, values);
        errdefer first.deinit();
        return .{ first, try glm5_testing.device(u16, .bf16, &.{ count, self.c.tapWidth() }, values) };
    }
};

fn drop(buffers: *[world]cuda.Buffer) void {
    for (buffers) |*one| one.deinit();
}

/// Rows `row` onward of the matrix `name` from column `col`, times x, in f64.
fn dots(fx: Fx, name: []const u8, row: usize, col: usize, x: []const u16, out: []f64) void {
    const tensor = fx.source.get(name).?;
    const k: usize = @intCast(tensor.shape[1]);
    for (out, 0..) |*o, n| {
        var sum: f64 = 0;
        for (x, 0..) |bits, at| {
            const w = std.mem.readInt(u16, tensor.data[((row + n) * k + col + at) * 2 ..][0..2], .little);
            sum += @as(f64, value(w)) * value(bits);
        }
        o.* = sum;
    }
}

/// `dots` rounded to bf16, as a linear writes a bf16 output.
fn linear(fx: Fx, arena: Allocator, name: []const u8, row: usize, col: usize, x: []const u16, out: []u16) !void {
    const wide = try arena.alloc(f64, out.len);
    dots(fx, name, row, col, x, wide);
    for (out, wide) |*o, v| o.* = numeric.bf16FromF32(@floatCast(v));
}

fn vector(fx: Fx, name: []const u8) []const u16 {
    const data = fx.source.get(name).?.data;
    return @alignCast(std.mem.bytesAsSlice(u16, data));
}

fn norm(arena: Allocator, x: []const u16, w: []const u16, out: []u16, eps: f32) !void {
    const scratch = try arena.alloc(f32, std.math.ceilPowerOfTwoAssert(usize, x.len));
    glue.rmsnorm(x, .{ .bf16 = w }, out, eps, scratch);
}

/// The steps of `dflash2.py` on the host: the context by position, and a block over it.
const Model = struct {
    fx: Fx,
    arena: Allocator,
    /// bf16 [positions, 128] a (layer, rank, KV head).
    k: [][]u16,
    v: [][]u16,
    end: usize = 0,

    fn init(fx: Fx, arena: Allocator, positions: usize) !Model {
        const c = fx.c;
        const count = c.layers * world * c.rankKvHeads();
        const k = try arena.alloc([]u16, count);
        const v = try arena.alloc([]u16, count);
        for (k, v) |*one, *other| {
            one.* = try arena.alloc(u16, positions * head);
            other.* = try arena.alloc(u16, positions * head);
        }
        return .{ .fx = fx, .arena = arena, .k = k, .v = v };
    }

    fn at(self: Model, layer: usize, rank: usize, kv: usize) usize {
        return (layer * world + rank) * self.fx.c.rankKvHeads() + kv;
    }

    /// A row's keys and values, bf16 [2 kv 128], into the context at `position`.
    fn keep(self: Model, layer: usize, rank: usize, row: []const u16, position: usize) void {
        const c = self.fx.c;
        const kv = c.rankKvHeads();
        const k_norm = vector(self.fx, self.name(layer, "self_attn.k_norm.weight"));
        for (0..kv) |h| {
            const slot = self.at(layer, rank, h);
            host.prepHead(row[h * head ..][0..head], k_norm[0..head], position, @floatCast(c.eps), c.theta, self.k[slot][position * head ..][0..head]);
            @memcpy(self.v[slot][position * head ..][0..head], row[(kv + h) * head ..][0..head]);
        }
    }

    fn name(self: Model, layer: usize, leaf: []const u8) []const u8 {
        return std.fmt.allocPrint(self.arena, "layers.{d}.{s}", .{ layer, leaf }) catch unreachable;
    }

    /// A row's k | v of `rank` from x.
    fn kvRow(self: Model, layer: usize, rank: usize, x: []const u16) ![]u16 {
        const inner = self.fx.c.rankKvHeads() * head;
        const row = try self.arena.alloc(u16, 2 * inner);
        try linear(self.fx, self.arena, self.name(layer, "self_attn.k_proj.weight"), rank * inner, 0, x, row[0..inner]);
        try linear(self.fx, self.arena, self.name(layer, "self_attn.v_proj.weight"), rank * inner, 0, x, row[inner..]);
        return row;
    }

    /// `_taps_compute` of one row.
    fn add(self: *Model, tap: []const u16) !void {
        const c = self.fx.c;
        const projected = try self.arena.alloc(u16, dims);
        try linear(self.fx, self.arena, "fc.weight", 0, 0, tap, projected);
        const ctx = try self.arena.alloc(u16, dims);
        try norm(self.arena, projected, vector(self.fx, "hidden_norm.weight"), ctx, @floatCast(c.eps));
        for (0..c.layers) |layer| for (0..world) |rank| self.keep(layer, rank, try self.kvRow(layer, rank, ctx), self.end);
        self.end += 1;
    }

    /// One convolution over the block's rows: `out` may be the residual.
    fn conv(self: Model, x: []const u16, dyn: []const u16, base: []const u16, branch: usize, residual: ?[]const u16, out: []u16) void {
        const n = self.fx.c.block;
        const wide = dyn.len / n;
        for (0..n) |i| {
            const res: ?[]const u16 = if (residual) |all| all[i * dims ..][0..dims] else null;
            host.dconvRow(x[i * dims ..][0..dims], if (i == 0) null else x[(i - 1) * dims ..][0..dims], dyn[i * wide ..][0..wide], base, branch, res, out[i * dims ..][0..dims]);
        }
    }

    /// The ranks' partials [world][block, dims] as the branch's rows.
    fn gather(self: Model, parts: [world][]f32) ![]u16 {
        const out = try self.arena.alloc(u16, parts[0].len);
        var i: usize = 0;
        while (i < out.len) : (i += dims) glue.branchRow(&.{ parts[0][i..][0..dims], parts[1][i..][0..dims] }, out[i..][0..dims]);
        return out;
    }

    /// `_block_compute` with `pending` first: the block's rows after the last layer.
    fn block(self: *Model, pending: u32) ![]u16 {
        const c = self.fx.c;
        const arena = self.arena;
        const n = c.block;
        const eps: f32 = @floatCast(c.eps);
        const x = try arena.alloc(u16, n * dims);
        for (0..n) |i| @memcpy(x[i * dims ..][0..dims], self.fx.table_host[(if (i == 0) pending else c.mask) * dims ..][0..dims]);
        const normed = try arena.alloc(u16, n * dims);
        const mixed = try arena.alloc(u16, n * dims);
        const dyn = try arena.alloc(u16, n * c.kernelWidth());
        const q_inner = c.rankHeads() * head;
        const kv = c.rankKvHeads();
        const shared = c.rankHeads() / kv;
        const width = c.rankWidth();
        const seen: host.Block = .{ .pos = self.end, .rows = n, .window = c.window.?, .causal = c.causal };
        for (0..c.layers) |layer| {
            const a_base = vector(self.fx, self.name(layer, "attention_conv.base_kernel"));
            const m_base = vector(self.fx, self.name(layer, "mlp_conv.base_kernel"));
            const in_norm = vector(self.fx, self.name(layer, "input_layernorm.weight"));
            for (0..n) |i| {
                try norm(arena, x[i * dims ..][0..dims], in_norm, normed[i * dims ..][0..dims], eps);
                try linear(self.fx, arena, self.name(layer, "attention_conv.kernel_projection.weight"), 0, 0, normed[i * dims ..][0..dims], dyn[i * c.kernelWidth() ..][0..c.kernelWidth()]);
            }
            self.conv(normed, dyn, a_base, 0, null, mixed);
            var parts: [world][]f32 = undefined;
            for (&parts, 0..) |*part, rank| {
                part.* = try arena.alloc(f32, n * dims);
                const q = try arena.alloc(u16, n * q_inner);
                const q_norm = vector(self.fx, self.name(layer, "self_attn.q_norm.weight"));
                for (0..n) |i| {
                    const row = mixed[i * dims ..][0..dims];
                    const raw = try arena.alloc(u16, q_inner);
                    try linear(self.fx, arena, self.name(layer, "self_attn.q_proj.weight"), rank * q_inner, 0, row, raw);
                    for (0..c.rankHeads()) |h| host.prepHead(raw[h * head ..][0..head], q_norm[0..head], self.end + i, eps, c.theta, q[i * q_inner + h * head ..][0..head]);
                    self.keep(layer, rank, try self.kvRow(layer, rank, row), self.end + i);
                }
                const attended = try arena.alloc(u16, q_inner);
                const wide = try arena.alloc(f64, dims);
                for (0..n) |i| {
                    for (0..c.rankHeads()) |h| {
                        const slot = self.at(layer, rank, h / shared);
                        host.attendRow(q[i * q_inner + h * head ..][0..head], self.k[slot], self.v[slot], seen, i, attended[h * head ..][0..head]);
                    }
                    dots(self.fx, self.name(layer, "self_attn.o_proj.weight"), 0, rank * q_inner, attended, wide);
                    for (part.*[i * dims ..][0..dims], wide) |*o, value_of| o.* = @floatCast(value_of);
                }
            }
            self.conv(try self.gather(parts), dyn, a_base, 1, x, x);
            const post_norm = vector(self.fx, self.name(layer, "post_attention_layernorm.weight"));
            for (0..n) |i| {
                try norm(arena, x[i * dims ..][0..dims], post_norm, normed[i * dims ..][0..dims], eps);
                try linear(self.fx, arena, self.name(layer, "mlp_conv.kernel_projection.weight"), 0, 0, normed[i * dims ..][0..dims], dyn[i * c.kernelWidth() ..][0..c.kernelWidth()]);
            }
            self.conv(normed, dyn, m_base, 0, null, mixed);
            for (&parts, 0..) |*part, rank| {
                const gate = try arena.alloc(u16, width);
                const up = try arena.alloc(u16, width);
                const act = try arena.alloc(u16, width);
                const wide = try arena.alloc(f64, dims);
                for (0..n) |i| {
                    const row = mixed[i * dims ..][0..dims];
                    try linear(self.fx, arena, self.name(layer, "mlp.gate_proj.weight"), rank * width, 0, row, gate);
                    try linear(self.fx, arena, self.name(layer, "mlp.up_proj.weight"), rank * width, 0, row, up);
                    for (act, gate, up) |*a, g, u| {
                        const silu = value(numeric.bf16FromF32(value(g) / (1.0 + @exp(-value(g)))));
                        a.* = numeric.bf16FromF32(silu * value(u));
                    }
                    dots(self.fx, self.name(layer, "mlp.down_proj.weight"), 0, rank * width, act, wide);
                    for (part.*[i * dims ..][0..dims], wide) |*o, value_of| o.* = @floatCast(value_of);
                }
            }
            self.conv(try self.gather(parts), dyn, m_base, 1, x, x);
        }
        return x;
    }
};

/// A layer's K or V of a rank by position, bf16 [KV heads, positions, 128], from a flat context.
fn fetchFlat(gpa: Allocator, buffer: cuda.Buffer) ![]u16 {
    return glm5_testing.fetch(u16, gpa, buffer);
}

fn expectNear(want: []const u16, got: []const u16, tolerance: f32, what: []const u8) !void {
    var worst: f32 = 0;
    for (want, got) |w, g| worst = @max(worst, @abs(value(w) - value(g)));
    if (!(worst <= tolerance)) {
        std.debug.print("{s}: off by {d}, over {d}\n", .{ what, worst, tolerance });
        return error.TestExpectedApproxEqAbs;
    }
}

test "DFlash2 pass: a context update and a block are dflash2.py's steps on the host, both ranks" {
    var fx: Fx = try .init(.stored);
    defer fx.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c = fx.c;
    const kept = 6;
    const pending = 17;
    var taps = try fx.taps(1, kept);
    defer drop(&taps);
    const taps_host = try glm5_testing.fetch(u16, arena, taps[0]);

    var pair = try fx.pair(kept, false);
    defer pair.deinit();
    try pair.addTaps(taps, 0, kept);
    try pair.block(fx.main(), pending);
    try cuda.synchronize();

    var model: Model = try .init(fx, arena, kept + c.block);
    for (0..kept) |row| try model.add(taps_host[row * c.tapWidth() ..][0..c.tapWidth()]);
    const x = try model.block(pending);

    const n = c.block;
    const kv = c.rankKvHeads();
    for (0..world) |rank| {
        // the context's rows and the block's own, by position: a linear's sum is in another order on the
        // device, so its bf16 may be a neighbour's
        for (0..c.layers) |layer| {
            const k = try fetchFlat(arena, pair.ranks[rank].k[layer]);
            const v = try fetchFlat(arena, pair.ranks[rank].v[layer]);
            for (0..kv) |h| {
                const slot = model.at(layer, rank, h);
                const count = if (layer == 0) (kept + n) * head else kept * head;
                try expectNear(model.k[slot][0..count], k[h * (kept + n) * head ..][0..count], 0.004, "k");
                try expectNear(model.v[slot][0..count], v[h * (kept + n) * head ..][0..count], 0.004, "v");
            }
        }
        const got = try glm5_testing.fetch(u16, arena, pair.ranks[rank].x);
        try expectNear(x, got, 0.02, "the block's rows");

        const final = try arena.alloc(u16, (n - 1) * dims);
        for (0..n - 1) |i| try norm(arena, x[(i + 1) * dims ..][0..dims], vector(fx, "norm.weight"), final[i * dims ..][0..dims], @floatCast(c.eps));
        try expectNear(final, try glm5_testing.fetch(u16, arena, pair.ranks[rank].final), 0.02, "the drafts' rows");
        // the head and the projection on the device's own rows: exact but for a sum's order
        const rows = try glm5_testing.fetch(u16, arena, pair.ranks[rank].final);
        const logits = try glm5_testing.fetch(u16, arena, pair.ranks[rank].logits);
        const selected = try glm5_testing.fetch(u16, arena, pair.ranks[rank].selected);
        const want_logits = try arena.alloc(u16, half);
        const want_selected = try arena.alloc(u16, c.selector_rank);
        for (0..n - 1) |i| {
            const row = rows[i * dims ..][0..dims];
            const wide = try arena.alloc(f64, half);
            for (wide, 0..) |*o, token| {
                var sum: f64 = 0;
                for (row, fx.head_host[(rank * half + token) * dims ..][0..dims]) |a, b| sum += @as(f64, value(a)) * value(b);
                o.* = sum;
            }
            for (want_logits, wide) |*o, w| o.* = numeric.bf16FromF32(@floatCast(w));
            try expectNear(want_logits, logits[i * half ..][0..half], 0.004, "logits");
            try linear(fx, arena, "candidate_selector.hidden_projection.weight", 0, 0, row, want_selected);
            try expectNear(want_selected, selected[i * c.selector_rank ..][0..c.selector_rank], 0.004, "projected");
        }
    }
    // the gathered branches are the same on both ranks, and so are the rows
    const first = try glm5_testing.fetch(u16, arena, pair.ranks[0].x);
    const second = try glm5_testing.fetch(u16, arena, pair.ranks[1].x);
    try testing.expectEqualSlices(u16, first, second);
}

/// Every layer's K and V of both ranks at positions `from` to `to`, by position.
fn contextRows(arena: Allocator, pair: pass.Pair, from: usize, to: usize) ![]u16 {
    const c = pair.ranks[0].c;
    const kv = c.rankKvHeads();
    const out = try arena.alloc(u16, world * c.layers * 2 * kv * (to - from) * head);
    var at: usize = 0;
    for (pair.ranks) |rank| for (rank.k, rank.v) |k, v| for ([_]cuda.Buffer{ k, v }) |buffer| {
        const all = try glm5_testing.fetch(u16, arena, buffer);
        for (0..kv) |h| for (from..to) |position| {
            const row = if (rank.ring) position % rank.cap else position;
            @memcpy(out[at..][0..head], all[(h * rank.cap + row) * head ..][0..head]);
            at += head;
        };
    };
    return out;
}

test "DFlash2 pass: a context row's K and V are the same bits alone, in a batch of 64 and through the ring's wrap; a block is the same twice, flat, in the ring and as a graph, and moves no context" {
    var fx: Fx = try .init(.mixed);
    defer fx.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c = fx.c;
    const ring_rows = c.ringRows();
    const total = ring_rows + 108;
    const early = 10;
    var taps = try fx.taps(2, total);
    defer drop(&taps);

    // flat, in batches of 64
    var flat = try fx.pair(total, false);
    defer flat.deinit();
    try testing.expect(!flat.ranks[0].ring);
    try flat.addTaps(taps, 0, total);
    // flat, a row a launch
    var alone = try fx.pair(total, false);
    defer alone.deinit();
    for (0..total) |row| try alone.addTaps(taps, row, 1);
    // the ring, its wrap inside a batch
    var ring = try fx.pair(total, true);
    defer ring.deinit();
    try testing.expect(ring.ranks[0].ring and ring.ranks[0].cap == ring_rows);
    try ring.addTaps(taps, 0, early);
    try ring.addTaps(taps, early, total - early);
    try testing.expect((early + 2 * dflash2_pass.batch) < ring_rows and early + 3 * dflash2_pass.batch > ring_rows);
    // as graphs: an update a (row count, first row), captured at its first run and launched again after
    var graphs: dflash2_pass.Graphs = try .init(testing.allocator);
    defer graphs.deinit();
    var graphed = try fx.pair(total, true);
    defer graphed.deinit();
    graphed.graphs = &graphs;
    for (0..early) |row| try graphed.addTaps(taps, row, 1);
    try graphed.addTaps(taps, early, total - early);
    try testing.expectEqual(early + (total - early + dflash2_pass.batch - 1) / dflash2_pass.batch, graphs.updates.count());
    try cuda.synchronize();

    const want = try contextRows(arena, flat, 0, total);
    try testing.expectEqualSlices(u16, want, try contextRows(arena, alone, 0, total));
    const recent = try contextRows(arena, flat, total - ring_rows, total);
    try testing.expectEqualSlices(u16, recent, try contextRows(arena, ring, total - ring_rows, total));
    try testing.expectEqualSlices(u16, recent, try contextRows(arena, graphed, total - ring_rows, total));

    // a block: twice, after another pending token between, flat and in the ring
    try flat.block(fx.main(), 5);
    const first = try drafts(arena, flat);
    try flat.block(fx.main(), 9);
    const other = try drafts(arena, flat);
    try testing.expect(!std.mem.eql(u16, first, other));
    try flat.block(fx.main(), 5);
    try testing.expectEqualSlices(u16, first, try drafts(arena, flat));
    try ring.block(fx.main(), 5);
    try testing.expectEqualSlices(u16, first, try drafts(arena, ring));
    // the block's one graph: captured, then launched for another token and for this one again
    try graphed.block(fx.main(), 5);
    try testing.expectEqualSlices(u16, first, try drafts(arena, graphed));
    try graphed.block(fx.main(), 9);
    try testing.expectEqualSlices(u16, other, try drafts(arena, graphed));
    try graphed.block(fx.main(), 5);
    try testing.expectEqualSlices(u16, first, try drafts(arena, graphed));
    // and the context it read is as it was: what a query sees, the window back
    try testing.expectEqualSlices(u16, want, try contextRows(arena, flat, 0, total));
    try testing.expectEqualSlices(u16, try contextRows(arena, flat, total - c.window.?, total), try contextRows(arena, ring, total - c.window.?, total));

    // a block's own K and V stand where the next rows go: rows kept after a block are the rows kept without
    var later = try fx.taps(3, 2 * c.block);
    defer drop(&later);
    var plain = try fx.pair(early + 2 * c.block, false);
    defer plain.deinit();
    var drafted = try fx.pair(early + 2 * c.block, false);
    defer drafted.deinit();
    for ([_]*pass.Pair{ &plain, &drafted }) |one| try one.addTaps(taps, 0, early);
    try drafted.block(fx.main(), 5);
    for ([_]*pass.Pair{ &plain, &drafted }) |one| try one.addTaps(later, 0, 2 * c.block);
    try cuda.synchronize();
    try testing.expectEqualSlices(u16, try contextRows(arena, plain, 0, early + 2 * c.block), try contextRows(arena, drafted, 0, early + 2 * c.block));
}

test "DFlash2 pass: the gathered candidates hold each rank's largest logits and then their tokens at the rank's place, the host's of its logits" {
    var fx: Fx = try .init(.mixed);
    defer fx.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c = fx.c;
    const chain = @import("../../../../src/decode/dflash2/chain.zig");
    var taps = try fx.taps(4, 20);
    defer drop(&taps);
    var pair = try fx.pair(20, false);
    defer pair.deinit();
    try pair.addTaps(taps, 0, 20);
    try pair.block(fx.main(), 5);
    try cuda.synchronize();
    const rows = c.block - 1;
    const top = c.top_k;
    try testing.expectEqual(world * 2 * rows * top, pair.tops.view.count());
    const words = try glm5_testing.fetch(u32, arena, pair.tops);
    const values = try arena.alloc(f32, top);
    const tokens = try arena.alloc(i32, top);
    for (pair.ranks, 0..) |rank, r| {
        const logits = try glm5_testing.fetch(u16, arena, rank.logits);
        const mine = words[r * 2 * rows * top ..][0 .. 2 * rows * top];
        for (0..rows) |row| {
            chain.largest(logits[row * half ..][0..half], r * half, values, tokens);
            try testing.expectEqualSlices(u32, @ptrCast(values), mine[row * top ..][0..top]);
            try testing.expectEqualSlices(u32, @ptrCast(tokens), mine[(rows + row) * top ..][0..top]);
            // a rank's tokens are of its own rows of the vocabulary
            for (tokens) |token| try testing.expect(@as(usize, @intCast(token)) / half == r);
        }
    }
}

test "DFlash2 pass: a kept context restored, into the pair it left, a flat one and a ring at another turn, gives a cold pair's rows and block" {
    var fx: Fx = try .init(.mixed);
    defer fx.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c = fx.c;
    const ring_rows = c.ringRows();
    // each end: inside the window, and past the ring's wrap
    for ([_]usize{ c.window.? - 7, ring_rows + 41 }) |end| {
        const total = end + 2 * ring_rows + 30;
        var taps = try fx.taps(4, total);
        defer drop(&taps);
        var cold = try fx.pair(end, false);
        defer cold.deinit();
        try cold.addTaps(taps, 0, end);
        try cold.block(fx.main(), 5);
        const want = try drafts(arena, cold);
        const seen = @min(end, c.window.?);
        const rows = try contextRows(arena, cold, end - seen, end);

        // taken as graphs run, before a block and more rows write over it
        var graphs: dflash2_pass.Graphs = try .init(testing.allocator);
        defer graphs.deinit();
        var hot = try fx.pair(total, true);
        defer hot.deinit();
        hot.graphs = &graphs;
        try hot.addTaps(taps, 0, end);
        var kept: pass.Kept = try .take(&hot);
        defer kept.deinit();
        try testing.expectEqual(end, kept.end);
        try testing.expectEqual(seen, kept.rows);
        try hot.block(fx.main(), 9);
        try hot.addTaps(taps, end, total - end);
        try hot.block(fx.main(), 9);
        try testing.expect(!std.mem.eql(u16, want, try drafts(arena, hot)));

        // a ring that reaches `end` at another turn than the one it was taken from, and a flat pair
        var turned = try fx.pair(total, true);
        defer turned.deinit();
        try turned.addTaps(taps, 0, 3 * dflash2_pass.batch + 5);
        var fresh = try fx.pair(end + 9, false);
        defer fresh.deinit();
        for ([_]*pass.Pair{ &hot, &turned, &fresh }) |one| {
            try kept.restore(one);
            try testing.expectEqual(end, one.context_end);
            try testing.expectEqualSlices(u16, rows, try contextRows(arena, one.*, end - seen, end));
            try one.block(fx.main(), 5);
            try testing.expectEqualSlices(u16, want, try drafts(arena, one.*));
        }
        // and what follows a restore is what follows in a pair that never left
        try hot.addTaps(taps, end, 2 * dflash2_pass.batch + 3);
        var through = try fx.pair(total, false);
        defer through.deinit();
        try through.addTaps(taps, 0, end + 2 * dflash2_pass.batch + 3);
        try through.block(fx.main(), 5);
        const next = try drafts(arena, through);
        try hot.block(fx.main(), 5);
        try testing.expectEqualSlices(u16, next, try drafts(arena, hot));
    }
}

test "DFlash2 seats: a job granted no row is absorbed without a block, and drafts where it is granted one (spec 152 WS-08, TensorFold #1779)" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const drafter_file = @import("../../../../src/models/glm5/dflash2/drafter.zig");
    const loopback = @import("../../../../src/models/glm5/layers/loopback.zig");
    const job_file = @import("../../../../src/models/glm5/draft.zig");
    const fx_random = @import("../../../../src/models/glm5/checks/random.zig").fx;
    const rows = 9;
    const capacity = 64;
    var fixture: glm5_testing.Fixture = undefined;
    try fixture.init(467, rows + 1);
    defer fixture.deinit();
    var loop = try fixture.loop(capacity);
    defer loop.deinit();
    var c = try drafter_testing.config(arena, fx_random.dims, fx_random.vocab, 5, &.{ 1, 3 });
    var source: drafter_testing.Source = .empty;
    var prng: std.Random.DefaultPrng = .init(468);
    try drafter_testing.put(arena, prng.random(), c, &source);
    try loop.b.setTaps(c.taps);
    const scratch: cuda.Scratch.Rows = .{ .pool = &fixture.pool, .count = dflash2_pass.batch };
    var w: [world]weights.Weights = .{
        try .load(arena, gpa, source, &c, 0, .mixed, scratch),
        try .load(arena, gpa, source, &c, 1, .mixed, scratch),
    };
    defer for (&w) |*one| one.deinit();
    var books = try weights.Codebooks.load(gpa, source, &c);
    defer books.deinit(gpa);
    // the window the jobs' kept rows come from: taps rows are the window's rows, kept the tokens after them
    const tokens = fixture.tokens;
    try loop.state.reset();
    try loopback.forward(loop.m, &loop.state, &loop.b, tokens[0..rows]);
    try loopback.commit(loop.m, &loop.state, loop.b, tokens[0..rows], rows);
    var pair = try pass.Pair.init(gpa, w, &c, capacity, false, fx_random.vocab / world);
    defer pair.deinit();
    var seats: pass.SeatsOf(world) = try .init(gpa, &pair, 2);
    defer seats.deinit();
    var d: drafter_file.Drafter = try .init(gpa, &c, &pair, .{ .m = loop.m, .main = &loop.b }, books, .{ .seed = 0, .temperature = 0 });
    defer d.deinit();
    var full = [_]bool{ false, false };
    var granted: [8]i32 = undefined;
    var jobs = [_]job_file.Job{
        .{ .slot = 0, .from = 1, .kept = tokens[1..4], .position = 3, .settings = .{ .seed = 0, .temperature = 0 }, .ends = &.{}, .guide = null, .out = &.{}, .full = &full[0] },
        .{ .slot = 1, .from = 4, .kept = tokens[4..rows], .position = 5, .settings = .{ .seed = 0, .temperature = 0 }, .ends = &.{}, .guide = null, .out = &granted, .full = &full[1] },
    };
    try seats.draft(&d, &jobs);
    try cuda.synchronize();
    // no block for the zero-grant job: the pass ran the other's alone, its context taken in beside it
    try testing.expectEqual(@as(usize, 0), jobs[0].count);
    try testing.expectEqual(@as(usize, 1), seats.blocks);
    try testing.expectEqual(@as(usize, 1), seats.order[0]);
    try testing.expectEqual(@as(usize, 1), d.passes);
    try testing.expect(jobs[1].count > 0);
    var pos: i32 = 0;
    try seats.seats[0].pos[0].downloadAt(0, std.mem.asBytes(&pos));
    try testing.expectEqual(@as(i32, 3), pos);
    // granted rows again, slot 0 drafts on the state the skipped round kept
    var later: [8]i32 = undefined;
    var again = [_]job_file.Job{
        .{ .slot = 0, .from = 4, .kept = tokens[4..7], .position = 6, .settings = .{ .seed = 0, .temperature = 0 }, .ends = &.{}, .guide = null, .out = &later, .full = &full[0] },
    };
    try seats.draft(&d, &again);
    try testing.expect(again[0].count > 0);
    try seats.seats[0].pos[0].downloadAt(0, std.mem.asBytes(&pos));
    try testing.expectEqual(@as(i32, 6), pos);
}
