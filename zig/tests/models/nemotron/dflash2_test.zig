//! Nemotron's DFlash2 drafter on a synthetic one at its schema (spec 151 WS-07): the config accepted and
//! refused in words, a context update and a block pass against `dflash2.py`'s steps on the host, and a
//! block the same bits alone and beside another. The drafter's schema is GLM's: `sliding_window` null (the
//! whole context seen, no ring) and a GQA whose q_inner is not hidden. Both ranks in one process.
//! The pair is `decode/dflash2/pass.zig`'s own with the smallest `Target` its generics ask for: the target
//! (the taps' capture, `Main`, `initRig`) is ForwardSlice's.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const cuda = @import("../../../src/cuda/cuda.zig");
const bf16 = @import("../../../src/weights/bf16.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const gpu = @import("../../cuda/testing.zig");
const numeric_fixture = @import("../../weights/numeric_fixture.zig");
const glue = @import("../../../src/models/glm5/layers/glue_ref.zig");
const config_file = @import("../../../src/decode/dflash2/config.zig");
const drafter_testing = @import("../../../src/models/nemotron/checks/random_drafter.zig");
const host = @import("../../../src/decode/dflash2/kernels_ref.zig");
const dflash2_pass = @import("../../../src/decode/dflash2/pass.zig");
const drafter = @import("../../../src/models/nemotron/dflash2.zig");
const weights = @import("../../../src/decode/dflash2/weights.zig");

const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;
const Error = config_file.Error;
const fixture = drafter_testing.fixture;
const world = config_file.world;

const dims = 512;
const vocab = 128;
const half = vocab / world;
const head = host.head;

const randomBf16 = numeric_fixture.randomBf16;
const device = gpu.device;
const fetch = gpu.fetch;

/// The fixture with `from` (once in it) replaced by `to` is refused with `expected` and a message holding `needle`.
fn expectRefused(expected: Error, from: []const u8, to: []const u8, needle: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, fixture, from));
    const text = try std.mem.replaceOwned(u8, arena.allocator(), fixture, from, to);
    var diagnostic: Diagnostic = .{};
    try testing.expectError(expected, Config.parseFull(arena.allocator(), text, &diagnostic));
    if (std.mem.indexOf(u8, diagnostic.message, needle) == null) {
        std.debug.print("\"{s}\" is not in \"{s}\"\n", .{ needle, diagnostic.message });
        return error.TestExpectedEqual;
    }
}

test "a Nemotron drafter's config: the whole context, a GQA's inner width, the six taps" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try Config.parseFull(arena.allocator(), fixture, &diagnostic);
    try testing.expectEqual(@as(usize, 512), c.hidden);
    // Nemotron's GQA: 8 heads of 128 over hidden 512, q_inner 1024, one KV head a rank
    try testing.expectEqual(@as(usize, 4), c.rankHeads());
    try testing.expectEqual(@as(usize, 1), c.rankKvHeads());
    try testing.expectEqual(@as(usize, 128), c.rankWidth());
    try testing.expectEqual(@as(?usize, null), c.window);
    try testing.expect(!c.causal);
    try testing.expectEqualSlices(u32, &.{ 1, 3 }, c.taps);
    try testing.expectEqual(@as(usize, 1024), c.tapWidth());
    try testing.expectEqual(@as(usize, 128), c.kernelWidth());
    try testing.expectEqual(@as(u32, 120), c.mask);
    // no window, no ring: the context is capacity + a block, flat
    try testing.expectEqual(@as(usize, 0), c.ringRows());
    try testing.expectEqual(@as(usize, 3000), c.seen(3000));
    try testing.expectEqual(@as(?[]const u8, null), try c.mismatch(arena.allocator(), 512, 128, 4));
    try testing.expect(std.mem.indexOf(u8, (try c.mismatch(arena.allocator(), 512, 128, 52)).?, "4 layers") != null);
}

test "the published Nemotron DFlash2 sizes: hidden 2688, taps 1/5/19/29/41/51 of 52, no window" {
    const p = drafter.published;
    try testing.expectEqual(@as(usize, 2688), p.hidden);
    try testing.expectEqual(@as(usize, 6), p.layers);
    try testing.expectEqual(@as(usize, 32), p.heads);
    try testing.expectEqual(@as(usize, 2), p.kv_heads);
    try testing.expectEqual(@as(usize, 128), p.head_dim);
    try testing.expectEqual(@as(usize, 6144), p.intermediate);
    try testing.expectEqual(@as(?usize, null), p.window);
    try testing.expectEqual(@as(usize, 16), p.block);
    try testing.expectEqual(@as(usize, 16), p.group);
    try testing.expectEqual(@as(usize, 256), p.selector_rank);
    try testing.expectEqual(@as(usize, 16), p.top_k);
    try testing.expectEqual(@as(usize, 131072), p.vocab);
    try testing.expectEqual(@as(u32, 990), p.mask);
    try testing.expectEqualSlices(u32, &.{ 1, 5, 19, 29, 41, 51 }, &p.taps);
    try testing.expectEqual(@as(usize, 52), p.target_layers);
}

test "what a Nemotron drafter's config is refused for" {
    try expectRefused(error.UnsupportedModel, "DFlash2DraftModel", "NemotronHForCausalLM", "not a DFlash2 drafter");
    try expectRefused(error.UnsupportedModel, "\"sliding_window\":null", "\"sliding_window\":2048", "keeps none");
    try expectRefused(error.UnsupportedModel, "\"conv_kernel_size\":2", "\"conv_kernel_size\":3", "two taps");
    try expectRefused(error.UnsupportedModel, "\"head_dim\":128", "\"head_dim\":64", "heads of 128");
    try expectRefused(error.UnsupportedModel, "\"is_causal\":false", "\"is_causal\":true", "both ways");
    try expectRefused(error.InvalidConfig, "\"num_key_value_heads\":2", "\"num_key_value_heads\":1", "each take half");
    try expectRefused(error.InvalidConfig, "\"num_key_value_heads\":2", "\"num_key_value_heads\":3", "each take half");
    try expectRefused(error.InvalidConfig, "\"conv_group_size\":16", "\"conv_group_size\":24", "does not divide");
    try expectRefused(error.InvalidConfig, "\"block_size\":4", "\"block_size\":1", "block_size is 1");
    try expectRefused(error.InvalidConfig, "[1,3]", "[3,1]", "rising layers");
    try expectRefused(error.InvalidConfig, "[1,3]", "[1,4]", "rising layers");
    try expectRefused(error.InvalidConfig, "\"mask_token_id\":120", "\"mask_token_id\":128", "not in the vocabulary");
    try expectRefused(error.InvalidConfig, "\"sliding_window\":null,", "", "no sliding_window");
    try expectRefused(error.InvalidConfig, "\"rope_theta\":10000.0", "\"theta\":1", "no rope_theta");
}

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
        var prng: std.Random.DefaultPrng = .init(2067);
        const random = prng.random();
        const c = try arena.create(config_file.Config);
        c.* = try drafter_testing.config(arena, dims, vocab, 5, &.{ 1, 3 });
        var source: drafter_testing.Source = .empty;
        try drafter_testing.put(arena, random, c.*, &source);
        const rows: cuda.Scratch.Rows = .{ .pool = pool, .count = dflash2_pass.batch };
        var w: [world]weights.Weights = undefined;
        for (&w, 0..) |*one, rank| one.* = try .load(arena, gpa, source, c, rank, storage, rows);
        const table_host = try arena.alloc(u16, vocab * dims);
        randomBf16(random, table_host, 1.0);
        const head_host = try arena.alloc(u16, vocab * dims);
        randomBf16(random, head_host, 1.0 / @sqrt(@as(f32, dims)));
        var heads: [world]bf16.Linear = undefined;
        for (&heads, 0..) |*one, rank| one.* = try .init(std.mem.sliceAsBytes(head_host[rank * half * dims ..][0 .. half * dims]), half, dims, rows);
        return .{ .arena_state = arena_state, .pool = pool, .c = c, .source = source, .w = w, .table_host = table_host, .table = try device(u16, .bf16, &.{ vocab, dims }, table_host), .head_host = head_host, .heads = heads };
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

    /// Both ranks of the shared pair, no Target needed: `decode/dflash2/pass.zig`'s own.
    const Pair = dflash2_pass.PairOf(world, struct {
        pub const ranks = @import("../../../src/ranks/ranks.zig");
        pub const lead = @import("../../../src/ranks/lead.zig");
        pub const Op = enum(u32) { flash_taps, flash_block, flash_round };
        pub const wide = false;
        pub const Job = struct {};
        pub fn Of(comptime held: usize) type {
            _ = held;
            return struct {};
        }
        pub const owns_heads = false;
        pub fn initRig() void {}
        pub fn initRigHeld() void {}
    });

    fn pair(self: Fx, capacity: usize, ring: bool) !Pair {
        return .init(testing.allocator, self.w, self.c, capacity, ring, half);
    }

    /// `count` rows of taps, the same on both ranks.
    fn taps(self: Fx, seed: u64, count: usize) ![world]cuda.Buffer {
        var prng: std.Random.DefaultPrng = .init(seed);
        const values = try self.arena_state.allocator().alloc(u16, count * self.c.tapWidth());
        randomBf16(prng.random(), values, 1.0);
        var first = try device(u16, .bf16, &.{ count, self.c.tapWidth() }, values);
        errdefer first.deinit();
        return .{ first, try device(u16, .bf16, &.{ count, self.c.tapWidth() }, values) };
    }
};

fn drop(buffers: *[world]cuda.Buffer) void {
    for (buffers) |*one| one.deinit();
}

fn value(bits: u16) f32 {
    return numeric.f32FromBf16(bits);
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

/// The steps of `dflash2.py` on the host: the context by position, and a block over it — the whole of it,
/// Nemotron's drafter seeing no window.
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

    /// `_block_compute` with `pending` first: the block's rows after the last layer. No window: every
    /// position back to 0 is seen (`cap` is the whole context).
    fn block(self: *Model, pending: u32, cap: usize) ![]u16 {
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
        const seen: host.Block = .{ .pos = self.end, .rows = n, .window = c.window orelse cap, .causal = c.causal };
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

fn fetchFlat(gpa: Allocator, buffer: cuda.Buffer) ![]u16 {
    return fetch(u16, gpa, buffer);
}

fn expectNear(want: []const u16, got: []const u16, tolerance: f32, what: []const u8) !void {
    var worst: f32 = 0;
    for (want, got) |w, g| worst = @max(worst, @abs(value(w) - value(g)));
    if (!(worst <= tolerance)) {
        std.debug.print("{s}: off by {d}, over {d}\n", .{ what, worst, tolerance });
        return error.TestExpectedApproxEqAbs;
    }
}

/// `expectNear`, but the tolerance grows with the slice's own size: a residual stream's values reach tens,
/// where one bf16 step is bigger than the whole small-value bound, and a fp32 sum's order costs a few such
/// steps beside a cancelling element. The bound is the peak's share, so a wrong element — off by a value's
/// size, not a share — still fails.
fn expectScaled(want: []const u16, got: []const u16, relative: f32, what: []const u8) !void {
    var peak: f32 = 0;
    for (want) |w| peak = @max(peak, @abs(value(w)));
    const bound = @max(@as(f32, 0.02), peak * relative);
    var worst: f32 = 0;
    for (want, got) |w, g| worst = @max(worst, @abs(value(w) - value(g)));
    if (!(worst <= bound)) {
        std.debug.print("{s}: off by {d}, bound {d}\n", .{ what, worst, bound });
        return error.TestExpectedApproxEqAbs;
    }
}

test "Nemotron DFlash2 pass: a context update and a block are dflash2.py's steps on the host, both ranks" {
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
    const taps_host = try fetch(u16, arena, taps[0]);

    var pair = try fx.pair(kept, false);
    defer pair.deinit();
    // no window: a flat context of capacity + block, never a ring
    try testing.expect(!pair.ranks[0].ring and pair.ranks[0].cap == kept + c.block);
    try pair.addTaps(taps, 0, kept);
    try pair.block(fx.main(), pending);
    try cuda.synchronize();

    var model: Model = try .init(fx, arena, kept + c.block);
    for (0..kept) |row| try model.add(taps_host[row * c.tapWidth() ..][0..c.tapWidth()]);
    const x = try model.block(pending, pair.ranks[0].cap);

    const n = c.block;
    const kv = c.rankKvHeads();
    for (0..world) |rank| {
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
        const got = try fetch(u16, arena, pair.ranks[rank].x);
        try expectScaled(x, got, 0.015, "the block's rows");

        const final = try arena.alloc(u16, (n - 1) * dims);
        for (0..n - 1) |i| try norm(arena, x[(i + 1) * dims ..][0..dims], vector(fx, "norm.weight"), final[i * dims ..][0..dims], @floatCast(c.eps));
        try expectScaled(final, try fetch(u16, arena, pair.ranks[rank].final), 0.015, "the drafts' rows");
        const rows = try fetch(u16, arena, pair.ranks[rank].final);
        const logits = try fetch(u16, arena, pair.ranks[rank].logits);
        const selected = try fetch(u16, arena, pair.ranks[rank].selected);
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
    const first = try fetch(u16, arena, pair.ranks[0].x);
    const second = try fetch(u16, arena, pair.ranks[1].x);
    try testing.expectEqualSlices(u16, first, second);
}

/// The drafts' rows a block left: x, logits and the projected rows, both ranks'.
fn drafts(arena: Allocator, pair: Fx.Pair) ![]u16 {
    var all: std.ArrayList(u16) = .empty;
    for (pair.ranks) |rank| for ([_]cuda.Buffer{ rank.logits, rank.selected, rank.x }) |buffer| try all.appendSlice(arena, try fetch(u16, arena, buffer));
    return all.items;
}

test "Nemotron DFlash2 pass: a block is the same bits twice, alone and beside another pair, and sees the whole flat context" {
    var fx: Fx = try .init(.mixed);
    defer fx.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const total = 300;
    var taps = try fx.taps(2, total);
    defer drop(&taps);

    var flat = try fx.pair(total, false);
    defer flat.deinit();
    try flat.addTaps(taps, 0, total);
    // asked for a ring, there is none: no window holds the context whole and flat
    var wanted = try fx.pair(total, true);
    defer wanted.deinit();
    try testing.expect(!wanted.ranks[0].ring);
    try wanted.addTaps(taps, 0, total);

    try flat.block(fx.main(), 5);
    const first = try drafts(arena, flat);
    try flat.block(fx.main(), 9);
    const other = try drafts(arena, flat);
    try testing.expect(!std.mem.eql(u16, first, other));
    try flat.block(fx.main(), 5);
    try testing.expectEqualSlices(u16, first, try drafts(arena, flat));
    // beside another context at the same end: the same bits
    try wanted.block(fx.main(), 5);
    try testing.expectEqualSlices(u16, first, try drafts(arena, wanted));
}
