//! `dsa.zig`'s tests on random weights: a window, chunks, windows cut short and serial steps bit for bit over a
//! chunk's boundary (512 keys), two ranks' heads against the whole layer's, and chosen rows of the ranks' sum
//! against the layer on the host in fp64. The absorb and the expand alone: each element near its fp64 sum
//! (their products are the tensor cores', so no host order is theirs), and a row alone against the row in a
//! window, bit for bit.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const face = @import("../../../../src/weights/face.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const config_file = @import("../../../../src/models/glm5/layers/config.zig");
const fixtures = @import("../../../../src/models/glm5/checks/fixtures.zig");
const dsa = @import("../../../../src/models/glm5/layers/dsa.zig");
const host = @import("../../../../src/models/glm5/layers/glue_ref.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const shared = @import("../testing.zig");
const device = shared.device;
const Source = fixtures.Source;

const fx = struct {
    const dims = 256;
    /// The whole layer's heads: three blocks of the attention's 16, the last of 4; a rank's 18 are 16 and 2.
    const heads = 36;
    const key_dim = 64;
    const value_dim = 64;
    const q_rank = 64;
    const layers = 2;
    /// Positions the tests run: past a chunk of 512 keys.
    const total = 540;
    const capacity = 600;

    fn config(arena: Allocator) !config_file.Config {
        var diagnostic: config_file.Diagnostic = .{};
        var c = try config_file.Config.parse(arena, fixtures.config, &diagnostic);
        c.hidden = dims;
        c.heads = heads;
        c.key_dim = key_dim;
        c.value_dim = value_dim;
        c.q_rank = q_rank;
        c.index_heads = 2;
        c.index_dim = 16;
        // the positions run past the fixture's dense limit of 11; this layer's test is the dense attention,
        // so the limit stays beyond them and the indexer, which load requires, changes no row
        c.index_top = 540;
        // the layers loaded here are DSA layers: a linear is read by its layer's kind (`layout.kindIn`)
        const types = try arena.alloc(config_file.LayerType, c.layers);
        @memset(types, .deepseek_sparse_attention);
        c.layer_types = types;
        return c;
    }

    fn prefix(arena: Allocator, layer: usize) ![]const u8 {
        return std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.self_attn.", .{layer});
    }

    fn put(arena: Allocator, random: std.Random, source: *Source, layer: usize, leaf: []const u8, shape: []const u64, spread: f32, mean: f32) !void {
        var count: usize = 1;
        for (shape) |dim| count *= @intCast(dim);
        const values = try arena.alloc(u16, count);
        for (values) |*one| one.* = numeric.bf16FromF32(mean + random.floatNorm(f32) * spread);
        const name = try std.mem.concat(arena, u8, &.{ try prefix(arena, layer), leaf });
        try source.put(arena, name, .{ .dtype = .BF16, .shape = try arena.dupe(u64, shape), .data = std.mem.sliceAsBytes(values) });
    }

    /// A whole DSA layer of random tensors under the checkpoint's names; scores a few units apart.
    fn putLayer(arena: Allocator, random: std.Random, source: *Source, layer: usize) !void {
        try put(arena, random, source, layer, "q_a_proj.weight", &.{ q_rank, dims }, 0.08, 0);
        try put(arena, random, source, layer, "kv_a_proj_with_mqa.weight", &.{ dsa.latent, dims }, 0.08, 0);
        try put(arena, random, source, layer, "q_a_layernorm.weight", &.{q_rank}, 0.2, 1);
        try put(arena, random, source, layer, "kv_a_layernorm.weight", &.{dsa.latent}, 0.2, 1);
        try put(arena, random, source, layer, "q_b_proj.weight", &.{ heads * key_dim, q_rank }, 0.1, 0);
        try put(arena, random, source, layer, "kv_b_proj.weight", &.{ heads * (key_dim + value_dim), dsa.latent }, 0.1, 0);
        try put(arena, random, source, layer, "o_proj.weight", &.{ dims, heads * value_dim }, 0.05, 0);
        const index_dim: u64 = 16;
        const index_heads: u64 = 2;
        try put(arena, random, source, layer, "indexer.wk.weight", &.{ index_dim, dims }, 0.06, 0);
        try put(arena, random, source, layer, "indexer.k_norm.weight", &.{index_dim}, 0.2, 1);
        try put(arena, random, source, layer, "indexer.k_norm.bias", &.{index_dim}, 0.05, 0);
        try put(arena, random, source, layer, "indexer.wq_b.weight", &.{ index_heads * index_dim, q_rank }, 0.1, 0);
        try put(arena, random, source, layer, "indexer.weights_proj.weight", &.{ index_heads, dims }, 0.1, 0);
        try put(arena, random, source, layer, "indexer.index_kpool_compress_gate", &.{ index_dim, dims }, 0.06, 0);
        try put(arena, random, source, layer, "indexer.index_kpool_compress_ape", &.{ 4, index_dim }, 0.1, 0);
    }

    fn bits(arena: Allocator, source: Source, layer: usize, leaf: []const u8) ![]align(1) const u16 {
        return @ptrCast(source.get(try std.mem.concat(arena, u8, &.{ try prefix(arena, layer), leaf })).?.data);
    }

    /// Layer `layer` whole, from the tensors as they stand.
    fn whole(arena: Allocator, source: Source, c: *const config_file.Config, layer: usize, rows: cuda.Scratch.Rows) !dsa.Weights {
        var parts: dsa.Parts = undefined;
        inline for (comptime std.meta.fieldNames(dsa.Parts), dsa.Parts.leaves) |field, leaf| {
            const data = std.mem.sliceAsBytes(try bits(arena, source, layer, leaf));
            @field(parts, field) = if (comptime @FieldType(dsa.Parts, field) == face.Part) .{ .bf16 = data } else data;
        }
        return dsa.Weights.init(arena, parts, c, rows);
    }
};

/// What a run leaves: every layer's output rows and latents, as bits, and the last layer's last attention.
const Played = struct {
    out: [fx.layers][]u32,
    /// a token's latent as the cache stores it: bf16, or an FP8 record
    cache: [fx.layers][]align(2) u8,
    /// bf16 [rows, heads, 512] of the last window
    ol: []u16,
};

/// `x`'s rows through the layers as `plan` cuts them: each step a window of its first number of rows from the
/// first row not kept yet, of which its second number is kept.
fn play(arena: Allocator, weights: []const dsa.Weights, x: cuda.Buffer, plan: []const [2]usize) !Played {
    return playAs(.bf16, arena, weights, x, plan);
}

/// `play` over a cache whose latents are held as `storage`.
fn playAs(storage: dsa.Storage, arena: Allocator, weights: []const dsa.Weights, x: cuda.Buffer, plan: []const [2]usize) !Played {
    const total: usize = @intCast(x.view.size[0]);
    var most: usize = 1;
    for (plan) |step| most = @max(most, step[0]);
    const index_dim: ?usize = if (weights[0].index) |index| index.dim else null;
    var cache: dsa.Cache = try .holdingAs(storage, weights.len, fx.capacity, index_dim);
    defer cache.deinit();
    // what a window cut short left past the kept rows is never read: NaNs show a read
    const nans = try arena.alloc(u8, cache.rows.bytes());
    @memset(nans, 0xff);
    try cache.rows.upload(nans);
    var buffers: dsa.Buffers = try .init(weights[0], fx.capacity, most);
    defer buffers.deinit();
    var outs: [fx.layers]cuda.Buffer = undefined;
    for (&outs) |*out| {
        out.* = try .init(.f32, &.{ total, fx.dims });
        try out.zero();
    }
    defer for (&outs) |*out| out.deinit();
    var last: usize = 0;
    for (plan) |step| {
        for (weights, 0..) |layer, index| try layer.forward(cache, index, buffers, x.span(cache.pos, step[0]), outs[index].span(cache.pos, step[0]));
        try cache.commit(step[1]);
        last = step[0];
    }
    try testing.expectEqual(total, cache.pos);
    var played: Played = undefined;
    for (0..fx.layers) |index| {
        played.out[index] = try shared.fetch(u32, arena, outs[index]);
        played.cache[index] = try arena.alignedAlloc(u8, .of(u16), total * storage.width());
        try cuda.download(.init(cache.layer(index).ptr.?, .u8, &.{ total, storage.width() }), played.cache[index]);
    }
    played.ol = try arena.alloc(u16, last * weights[0].heads * dsa.latent);
    try cuda.download(buffers.ol.rows(last), std.mem.sliceAsBytes(played.ol));
    return played;
}

fn expectSame(want: Played, got: Played) !void {
    for (0..fx.layers) |index| {
        try testing.expectEqualSlices(u8, want.cache[index], got.cache[index]);
        try testing.expectEqualSlices(u32, want.out[index], got.out[index]);
    }
}

/// Windows of `rows` rows, all kept, over the fixture's positions.
fn windows(arena: Allocator, rows: usize) ![]const [2]usize {
    var plan: std.ArrayList([2]usize) = .empty;
    var first: usize = 0;
    while (first < fx.total) : (first += rows) {
        const count = @min(rows, fx.total - first);
        try plan.append(arena, .{ count, count });
    }
    return plan.items;
}

const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    pool: cuda.Scratch,
    c: config_file.Config,
    source: Source,
    x: cuda.Buffer,

    fn init(self: *Fixture, seed: u64) !void {
        self.arena_state = .init(testing.allocator);
        const arena = self.arena_state.allocator();
        self.pool = .init(testing.allocator);
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        self.c = try fx.config(arena);
        self.source = .empty;
        for (0..fx.layers) |layer| try fx.putLayer(arena, random, &self.source, layer);
        const x = try arena.alloc(u16, fx.total * fx.dims);
        shared.randomBf16(random, x, 1.0);
        self.x = try device(u16, .bf16, &.{ fx.total, fx.dims }, x);
    }

    fn deinit(self: *Fixture) void {
        self.x.deinit();
        self.pool.deinit();
        self.arena_state.deinit();
    }

    /// The layers as `rank` holds them.
    fn load(self: *Fixture, rank: usize, rows: usize) ![fx.layers]dsa.Weights {
        const arena = self.arena_state.allocator();
        var weights: [fx.layers]dsa.Weights = undefined;
        for (&weights, 0..) |*one, layer| one.* = try .load(arena, self.source, try fx.prefix(arena, layer), &self.c, rank, .{ .pool = &self.pool, .count = rows });
        return weights;
    }
};

test "a window, chunks, a window cut short and serial steps leave the same rows and latents over a chunk's boundary" {
    var f: Fixture = undefined;
    try f.init(31);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    var weights = try f.load(1, 64);
    defer for (&weights) |*one| one.deinit();

    const serial = try play(arena, &weights, f.x, try windows(arena, 1));
    for (serial.out) |out| {
        var any = false;
        for (out) |one| {
            try testing.expect(std.math.isFinite(@as(f32, @bitCast(one))));
            any = any or one != 0;
        }
        try testing.expect(any);
    }
    // 512 is a window's first row, then its last, then inside one
    try expectSame(serial, try play(arena, &weights, f.x, try windows(arena, 64)));
    try expectSame(serial, try play(arena, &weights, f.x, try windows(arena, 57)));
    try expectSame(serial, try play(arena, &weights, f.x, try windows(arena, 60)));
    // drafts' windows of which the first rows are kept, the boundary among the rows dropped and among the kept
    var plan: std.ArrayList([2]usize) = .empty;
    try plan.appendSlice(arena, &.{ .{ 1, 1 }, .{ 9, 2 }, .{ 17, 17 } });
    for (0..7) |_| try plan.append(arena, .{ 64, 64 });
    try plan.appendSlice(arena, &.{ .{ 40, 30 }, .{ 20, 1 }, .{ 20, 3 }, .{ 9, 9 }, .{ 29, 29 } });
    try expectSame(serial, try play(arena, &weights, f.x, plan.items));
    // the window kernel from 60 rows: its windows the serial steps' bits too
    dsa.setWindow(60);
    defer dsa.setWindow(dsa.window_rows);
    try expectSame(serial, try play(arena, &weights, f.x, try windows(arena, 64)));
    try expectSame(serial, try play(arena, &weights, f.x, try windows(arena, 60)));
}

test "an FP8 cache: each token's record its latent's, windows and serial steps the same bits in both forms, the outputs near bf16's" {
    var f: Fixture = undefined;
    try f.init(31);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    var weights = try f.load(1, 64);
    defer for (&weights) |*one| one.deinit();

    const serial = try playAs(.fp8, arena, &weights, f.x, try windows(arena, 1));
    try expectSame(serial, try playAs(.fp8, arena, &weights, f.x, try windows(arena, 64)));
    try expectSame(serial, try playAs(.fp8, arena, &weights, f.x, try windows(arena, 57)));
    var plan: std.ArrayList([2]usize) = .empty;
    try plan.appendSlice(arena, &.{ .{ 1, 1 }, .{ 9, 2 }, .{ 17, 17 } });
    for (0..7) |_| try plan.append(arena, .{ 64, 64 });
    try plan.appendSlice(arena, &.{ .{ 40, 30 }, .{ 20, 1 }, .{ 20, 3 }, .{ 9, 9 }, .{ 29, 29 } });
    try expectSame(serial, try playAs(.fp8, arena, &weights, f.x, plan.items));
    // the window kernel from 60 rows, both forms of the queries (below)
    dsa.setWindow(60);
    defer dsa.setWindow(dsa.window_rows);
    try expectSame(serial, try playAs(.fp8, arena, &weights, f.x, try windows(arena, 64)));

    const wide = try play(arena, &weights, f.x, try windows(arena, 64));
    var worst: f64 = 0;
    for (0..fx.layers) |index| {
        // each record is the host's quantization of the bf16 cache's latent, byte for byte
        const lat: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, wide.cache[index]));
        for (0..fx.total) |t| {
            const rec = serial.cache[index][t * dsa.fp8_record ..][0..dsa.fp8_record];
            for (0..dsa.latent / 128) |g| {
                var most: f32 = 0;
                for (0..128) |i| most = @max(most, @abs(numeric.f32FromBf16(lat[t * dsa.latent + g * 128 + i])));
                const scale: f32 = if (most > 0) most / 448.0 else 1.0;
                try testing.expectEqual(scale, std.mem.bytesToValue(f32, rec[dsa.latent + g * 4 ..][0..4]));
                for (0..128) |i| try testing.expectEqual(numeric.e4m3FromF32(numeric.f32FromBf16(lat[t * dsa.latent + g * 128 + i]) / scale), rec[g * 128 + i]);
            }
        }
        // the layer's output against bf16's, of its largest
        var top: f64 = 0;
        var off: f64 = 0;
        for (wide.out[index], serial.out[index]) |a, b| {
            const x: f64 = @as(f32, @bitCast(a));
            const y: f64 = @as(f32, @bitCast(b));
            try testing.expect(std.math.isFinite(y));
            top = @max(top, @abs(x));
            off = @max(off, @abs(x - y));
        }
        worst = @max(worst, off / top);
    }
    std.debug.print("fp8 cache: a layer's output at most {d:.4} of its largest from bf16's\n", .{worst});
    try testing.expect(worst < fp8_bound);

    // bf16 queries over the same records (`ZSPARK_GLM_MLA_Q8=0`): its windows its serial steps too, and the FP8
    // queries' and weights' own error measured against it
    dsa.setQ8(false);
    defer dsa.setQ8(true);
    const plain = try playAs(.fp8, arena, &weights, f.x, try windows(arena, 1));
    try expectSame(plain, try playAs(.fp8, arena, &weights, f.x, try windows(arena, 64)));
    var apart: f64 = 0;
    var plain_worst: f64 = 0;
    for (0..fx.layers) |index| {
        try testing.expectEqualSlices(u8, plain.cache[index], serial.cache[index]);
        var top: f64 = 0;
        var off: f64 = 0;
        var off_wide: f64 = 0;
        for (plain.out[index], serial.out[index], wide.out[index]) |a, b, w| {
            const x: f64 = @as(f32, @bitCast(a));
            top = @max(top, @abs(x));
            off = @max(off, @abs(x - @as(f32, @bitCast(b))));
            off_wide = @max(off_wide, @abs(@as(f64, @as(f32, @bitCast(w))) - x));
        }
        apart = @max(apart, off / top);
        plain_worst = @max(plain_worst, off_wide / top);
    }
    std.debug.print("fp8 cache: FP8 queries at most {d:.4} of the largest from bf16 queries', which are {d:.4} from bf16's cache\n", .{ apart, plain_worst });
    try testing.expect(apart < fp8_bound);
}

/// An FP8 cache's layer output from bf16's, of the largest: E4M3's 2^-4 a value, averaged over the keys.
const fp8_bound = 0.05;

/// FP8 queries' output from the host's on the bf16 queries: a query's e4m3 rounding, 2^-4 a value over a scale up
/// to twice its group's largest, moves the scores of row 0 (spread some 2.5) and so its weights.
const fp8_queries = 0.1;

/// FP8 queries' output from the host's on the same queries: the weights' HIGH and LOW codes keep some 7 bits,
/// then bf16's rounding of the output.
const fp8_weights = 0.01;

/// A token's record as `zs_glm5_mla_write` makes it: a group's largest |value| / 448 (1 for zeros), codes e4m3.
fn record(rec: []u8, values: []const f32) void {
    for (0..dsa.latent / 128) |g| {
        var most: f32 = 0;
        for (values[g * 128 ..][0..128]) |v| most = @max(most, @abs(v));
        const s: f32 = if (most > 0) most / 448.0 else 1.0;
        for (0..128) |i| rec[g * 128 + i] = numeric.e4m3FromF32(values[g * 128 + i] / s);
        @memcpy(rec[dsa.latent + g * 4 ..][0..4], std.mem.asBytes(&s));
    }
}

test "an FP8 cache's chunks with FP8 queries: rows made by hand at the scales' ends near the host, a row alone its window's bits" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(114);
    const random = prng.random();
    const latent = dsa.latent;
    const heads = 36;
    const rows = 4;
    const positions = 1100;
    // a dense row attends every position: the lists are as wide
    const width = positions;
    const scale: f64 = 0.0625;

    // keys: noise, a group of zeros, a group so small its weight p s_g underflows the weights' floor
    const values = try arena.alloc(f32, positions * latent);
    for (values) |*v| v.* = random.floatNorm(f32) * 0.3;
    for (0..positions) |t| {
        if (t % 7 == 3) @memset(values[t * latent + 128 ..][0..128], 0);
        if (t % 11 == 5) for (values[t * latent + 256 ..][0..128]) |*v| {
            v.* *= 1e-30;
        };
    }
    const records = try arena.alloc(u8, positions * dsa.fp8_record);
    for (0..positions) |t| record(records[t * dsa.fp8_record ..][0..dsa.fp8_record], values[t * latent ..][0..latent]);
    // what the host attends: the records' values
    const keys = try arena.alloc(f64, positions * latent);
    for (0..positions) |t| for (0..latent) |d| {
        const s: f32 = std.mem.bytesToValue(f32, records[t * dsa.fp8_record + latent + d / 128 * 4 ..][0..4]);
        keys[t * latent + d] = @as(f64, numeric.f32FromE4m3(records[t * dsa.fp8_record + d])) * s;
    };

    // row 0: dense over two chunks, queries of 6 so that its weights are far from even, its head 0 zeros; row 1: a list whose first key's score is far over the
    // rest, so later tiles are far under the largest; row 2: five keys, the tile's rest past its list; row 3:
    // a list over a chunk's boundary, queries of 1e-6 (the queries' scale floor)
    const qa = try arena.alloc(u16, rows * heads * latent);
    shared.randomBf16(random, qa, 0.3);
    for (qa[0 .. heads * latent]) |*q| q.* = numeric.bf16FromF32(numeric.f32FromBf16(q.*) * 20);
    @memset(qa[0..latent], 0);
    var tokens: [rows * width]i32 = @splat(-1);
    const counts = [_]i32{ 0, 600, 5, 520 };
    for (1..rows) |r| for (0..@as(usize, @intCast(counts[r]))) |t| {
        tokens[r * width + t] = @intCast((r * 31 + t * 7) % positions);
    };
    const first: usize = @intCast(tokens[1 * width]);
    for (0..heads) |h| for (0..latent) |d| {
        qa[(1 * heads + h) * latent + d] = numeric.bf16FromF32(@floatCast(20.0 * keys[first * latent + d]));
    };
    for (qa[3 * heads * latent ..][0 .. heads * latent]) |*q| q.* = numeric.bf16FromF32(numeric.f32FromBf16(q.*) * 1e-6);
    const at: i32 = positions - rows;

    var cache = try shared.device(u8, .u8, &.{ positions, dsa.fp8_record }, records);
    defer cache.deinit();
    var qa_dev = try shared.device(u16, .bf16, &.{ rows, heads, latent }, qa);
    defer qa_dev.deinit();
    var tok_dev = try shared.device(i32, .i32, &.{ rows, width }, &tokens);
    defer tok_dev.deinit();
    var count_dev = try shared.device(i32, .i32, &.{rows}, &counts);
    defer count_dev.deinit();
    var pos = try shared.device(i32, .i32, &.{1}, &.{at});
    defer pos.deinit();
    var out = try cuda.Buffer.init(.bf16, &.{ rows, heads, latent });
    defer out.deinit();
    const chunks = (positions + dsa.chunk - 1) / dsa.chunk;
    var po = try cuda.Buffer.init(.f32, &.{ rows, chunks, heads, latent });
    defer po.deinit();
    var pm = try cuda.Buffer.init(.f32, &.{ rows, chunks, heads });
    defer pm.deinit();
    var pl = try cuda.Buffer.init(.f32, &.{ rows, chunks, heads });
    defer pl.deinit();
    const parts = [3]cuda.View{ po.view, pm.view, pl.view };
    const lists: [2]cuda.View = .{ tok_dev.view, count_dev.view };

    try dsa.attendRows(qa_dev.view, cache.view, pos.view, null, parts, out.view, lists, scale, positions);
    const got = try shared.fetch(u16, arena, out);
    dsa.setQ8(false);
    try dsa.attendRows(qa_dev.view, cache.view, pos.view, null, parts, out.view, lists, scale, positions);
    dsa.setQ8(true);
    const plain = try shared.fetch(u16, arena, out);

    // against the host's fp64 softmax over the records' values, of the head's largest mass |p| |v| (a value whose
    // own mass is far under it, as row 1's group of zeros in its first key where the rest weigh e^-50, keeps no
    // more than the largest's precision): with the bf16 queries, and with them quantized as the kernel does,
    // which leaves the weights' own rounding
    var worst = [3]f64{ 0, 0, 0 };
    for (0..rows) |r| {
        const n: usize = if (counts[r] > 0) @intCast(counts[r]) else @as(usize, @intCast(at)) + r + 1;
        const score = try arena.alloc(f64, n);
        for (0..heads) |h| {
            var q: [latent]f64 = undefined;
            var q8: [latent]f64 = undefined;
            for (&q, qa[(r * heads + h) * latent ..][0..latent]) |*one, bits| one.* = numeric.f32FromBf16(bits);
            for (0..latent / 128) |g| {
                var most: f32 = 0;
                for (q[g * 128 ..][0..128]) |v| most = @max(most, @abs(@as(f32, @floatCast(v))));
                var raw: u32 = @bitCast(@max(most, 1e-4) * (1.0 / 448.0));
                if (raw & 0x7fffff != 0) raw = (raw & 0xff800000) + 0x800000;
                const s: f32 = @bitCast(raw);
                for (0..128) |i| {
                    const code = numeric.e4m3FromF32(std.math.clamp(@as(f32, @floatCast(q[g * 128 + i])) * (1.0 / s), -448, 448));
                    q8[g * 128 + i] = @as(f64, numeric.f32FromE4m3(code)) * s;
                }
            }
            for ([_]*const [latent]f64{ &q, &q8 }, 0..) |query, form| {
                var top: f64 = -std.math.inf(f64);
                for (0..n) |t| {
                    const key: usize = if (counts[r] > 0) @intCast(tokens[r * width + t]) else t;
                    var dot: f64 = 0;
                    for (query, keys[key * latent ..][0..latent]) |x, y| dot += x * y;
                    score[t] = dot * scale;
                    top = @max(top, score[t]);
                }
                var sum: f64 = 0;
                for (score) |*one| {
                    one.* = @exp(one.* - top);
                    sum += one.*;
                }
                var want: [latent]f64 = @splat(0);
                var mass: f64 = 0;
                for (0..latent) |d| {
                    var sum_abs: f64 = 0;
                    for (0..n) |t| {
                        const key: usize = if (counts[r] > 0) @intCast(tokens[r * width + t]) else t;
                        want[d] += score[t] / sum * keys[key * latent + d];
                        sum_abs += score[t] / sum * @abs(keys[key * latent + d]);
                    }
                    mass = @max(mass, sum_abs);
                }
                const pairs: []const struct { []const u16, usize } = if (form == 0) &.{ .{ got, 0 }, .{ plain, 1 } } else &.{.{ got, 2 }};
                for (pairs) |pair| for (0..latent) |d| {
                    const have: f64 = numeric.f32FromBf16(pair[0][(r * heads + h) * latent + d]);
                    try testing.expect(std.math.isFinite(have));
                    worst[pair[1]] = @max(worst[pair[1]], @abs(have - want[d]) / mass);
                };
            }
        }
    }
    std.debug.print("fp8 queries: an output at most {d:.4} of its head's largest mass from the host's, {d:.4} from the host's on the same FP8 queries; bf16 queries {d:.4}\n", .{ worst[0], worst[2], worst[1] });
    try testing.expect(worst[0] < fp8_queries);
    try testing.expect(worst[2] < fp8_weights);

    // a row alone at its position: its window's bits
    var one_qa = try cuda.Buffer.init(.bf16, &.{ 1, heads, latent });
    defer one_qa.deinit();
    var one_tok = try cuda.Buffer.init(.i32, &.{ 1, width });
    defer one_tok.deinit();
    var one_count = try cuda.Buffer.init(.i32, &.{1});
    defer one_count.deinit();
    var one_out = try cuda.Buffer.init(.bf16, &.{ 1, heads, latent });
    defer one_out.deinit();
    var one_pos = try cuda.Buffer.init(.i32, &.{1});
    defer one_pos.deinit();
    for (0..rows) |r| {
        const here: i32 = at + @as(i32, @intCast(r));
        try one_qa.upload(std.mem.sliceAsBytes(qa[r * heads * latent ..][0 .. heads * latent]));
        try one_tok.upload(std.mem.sliceAsBytes(tokens[r * width ..][0..width]));
        try one_count.upload(std.mem.sliceAsBytes(counts[r..][0..1]));
        try one_pos.upload(std.mem.asBytes(&here));
        try dsa.attendRows(one_qa.view, cache.view, one_pos.view, null, parts, one_out.view, .{ one_tok.view, one_count.view }, scale, positions);
        try testing.expectEqualSlices(u16, got[r * heads * latent ..][0 .. heads * latent], try shared.fetch(u16, arena, one_out));
    }
}

test "a window's rows a block a row over their chunks, each folded as the merge folds it: the chunk blocks' and merge's bits" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(116);
    const random = prng.random();
    const latent = dsa.latent;
    // two blocks of heads, the second of 4
    const heads = 36;
    const positions = 1100;
    const width = positions;
    // lists ending a chunk's keys exactly and 1, 2 and 3 keys into the next, short ones, the widest, dense rows
    const counts = [_]i32{ 1024, 513, 514, 515, 0, 1, 32, 1100, 512, 0 };
    const rows = counts.len;
    const values = try arena.alloc(f32, positions * latent);
    for (values) |*v| v.* = random.floatNorm(f32) * 0.3;
    const records = try arena.alloc(u8, positions * dsa.fp8_record);
    for (0..positions) |t| record(records[t * dsa.fp8_record ..][0..dsa.fp8_record], values[t * latent ..][0..latent]);
    const latents = try arena.alloc(u16, positions * latent);
    for (latents, values) |*bits, v| bits.* = numeric.bf16FromF32(v);
    const qa = try arena.alloc(u16, rows * heads * latent);
    shared.randomBf16(random, qa, 2.0);
    var tokens: [rows * width]i32 = @splat(-1);
    for (0..rows) |r| for (0..@as(usize, @intCast(counts[r]))) |t| {
        tokens[r * width + t] = @intCast((r * 31 + t * 7) % positions);
    };
    const at: i32 = positions - rows;

    var records_dev = try shared.device(u8, .u8, &.{ positions, dsa.fp8_record }, records);
    defer records_dev.deinit();
    var latents_dev = try shared.device(u16, .bf16, &.{ positions, latent }, latents);
    defer latents_dev.deinit();
    var qa_dev = try shared.device(u16, .bf16, &.{ rows, heads, latent }, qa);
    defer qa_dev.deinit();
    var tok_dev = try shared.device(i32, .i32, &.{ rows, width }, &tokens);
    defer tok_dev.deinit();
    var count_dev = try shared.device(i32, .i32, &.{rows}, &counts);
    defer count_dev.deinit();
    var pos = try shared.device(i32, .i32, &.{1}, &.{at});
    defer pos.deinit();
    var out = try cuda.Buffer.init(.bf16, &.{ rows, heads, latent });
    defer out.deinit();
    const chunks = (positions + dsa.chunk - 1) / dsa.chunk;
    var po = try cuda.Buffer.init(.f32, &.{ rows, chunks, heads, latent });
    defer po.deinit();
    var pm = try cuda.Buffer.init(.f32, &.{ rows, chunks, heads });
    defer pm.deinit();
    var pl = try cuda.Buffer.init(.f32, &.{ rows, chunks, heads });
    defer pl.deinit();
    const parts = [3]cuda.View{ po.view, pm.view, pl.view };
    const lists: [2]cuda.View = .{ tok_dev.view, count_dev.view };
    // the window form's po as `Buffers` holds it past 127 rows: the rows' running output alone, [rows, heads,
    // 512] in [held, chunks, ...]
    const held = (rows + chunks - 1) / chunks;
    var tight = try cuda.Buffer.init(.f32, &.{ held, chunks, heads, latent });
    defer tight.deinit();
    const tight_parts = [3]cuda.View{ tight.view, pm.view, pl.view };

    defer dsa.setWindow(dsa.window_rows);
    defer dsa.setQ8(true);
    // the FP8 cache with FP8 queries, with bf16 queries, the bf16 cache
    for ([_]struct { cuda.View, bool }{ .{ records_dev.view, true }, .{ records_dev.view, false }, .{ latents_dev.view, true } }) |form| {
        dsa.setQ8(form[1]);
        dsa.setWindow(0);
        try dsa.attendRows(qa_dev.view, form[0], pos.view, null, parts, out.view, lists, 0.0625, positions);
        const chunked = try shared.fetch(u16, arena, out);
        for (chunked) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
        dsa.setWindow(1);
        try dsa.attendRows(qa_dev.view, form[0], pos.view, null, parts, out.view, lists, 0.0625, positions);
        try testing.expectEqualSlices(u16, chunked, try shared.fetch(u16, arena, out));
        try dsa.attendRows(qa_dev.view, form[0], pos.view, null, tight_parts, out.view, lists, 0.0625, positions);
        try testing.expectEqualSlices(u16, chunked, try shared.fetch(u16, arena, out));
    }
}

fn bf(v: f64) f64 {
    return numeric.roundBf16(@floatCast(v));
}

fn val(w: []align(1) const u16, i: usize) f64 {
    return numeric.f32FromBf16(w[i]);
}

/// to[i] = bf16(w[first + i] . from), w's rows `from.len` wide
fn product(w: []align(1) const u16, first: usize, from: []const f64, to: []f64) void {
    for (to, 0..) |*one, i| {
        var sum: f64 = 0;
        for (from, 0..) |v, j| sum += val(w, (first + i) * from.len + j) * v;
        one.* = bf(sum);
    }
}

/// to = bf16(w bf16(from / rms))
fn rmsnorm(w: []align(1) const u16, from: []const f64, to: []f64, eps: f64) void {
    var ss: f64 = 0;
    for (from) |v| ss += v * v;
    const inverse = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(from.len)) + eps);
    for (to, from, 0..) |*one, v, i| one.* = bf(val(w, i) * bf(v * inverse));
}

/// The whole DSA layer's row at position `pos` in fp64 with the model's bf16 roundings, over the latents `keys`
/// [pos + 1, 512] (the row's own the last): out [dims], and the row's latent.
fn hostRow(arena: Allocator, source: Source, c: *const config_file.Config, layer: usize, x: []const u16, keys: []const u16, pos: usize, out: []f64, lat: *[dsa.latent]f64) !void {
    const q_a = try fx.bits(arena, source, layer, "q_a_proj.weight");
    const kv_a = try fx.bits(arena, source, layer, "kv_a_proj_with_mqa.weight");
    const q_b = try fx.bits(arena, source, layer, "q_b_proj.weight");
    const kv_b = try fx.bits(arena, source, layer, "kv_b_proj.weight");
    const o = try fx.bits(arena, source, layer, "o_proj.weight");
    var xr: [fx.dims]f64 = undefined;
    for (&xr, x) |*to, from| to.* = numeric.f32FromBf16(from);
    var low: [fx.q_rank]f64 = undefined;
    var q_normed: [fx.q_rank]f64 = undefined;
    product(q_a, 0, &xr, &low);
    rmsnorm(try fx.bits(arena, source, layer, "q_a_layernorm.weight"), &low, &q_normed, c.eps);
    var kv_low: [dsa.latent]f64 = undefined;
    product(kv_a, 0, &xr, &kv_low);
    rmsnorm(try fx.bits(arena, source, layer, "kv_a_layernorm.weight"), &kv_low, lat, c.eps);
    const scores = try arena.alloc(f64, pos + 1);
    const per_head = fx.key_dim + fx.value_dim;
    const y = try arena.alloc(f64, fx.heads * fx.value_dim);
    for (0..fx.heads) |h| {
        var q: [fx.key_dim]f64 = undefined;
        product(q_b, h * fx.key_dim, &q_normed, &q);
        var qa: [dsa.latent]f64 = undefined;
        for (&qa, 0..) |*one, n| {
            var sum: f64 = 0;
            for (q, 0..) |v, k| sum += v * val(kv_b, (h * per_head + k) * dsa.latent + n);
            one.* = bf(sum);
        }
        var top: f64 = -std.math.inf(f64);
        for (scores, 0..) |*score, key| {
            var sum: f64 = 0;
            for (qa, keys[key * dsa.latent ..][0..dsa.latent]) |v, k| sum += v * numeric.f32FromBf16(k);
            score.* = sum / @sqrt(@as(f64, fx.key_dim));
            top = @max(top, score.*);
        }
        var ol: [dsa.latent]f64 = @splat(0);
        var total: f64 = 0;
        for (scores, 0..) |score, key| {
            const p = @exp(score - top);
            total += p;
            for (&ol, keys[key * dsa.latent ..][0..dsa.latent]) |*one, k| one.* += p * numeric.f32FromBf16(k);
        }
        for (&ol) |*one| one.* = bf(one.* / total);
        product(kv_b, h * per_head + fx.key_dim, &ol, y[h * fx.value_dim ..][0..fx.value_dim]);
    }
    for (out, 0..) |*to, d| {
        var sum: f64 = 0;
        for (y, 0..) |v, j| sum += val(o, d * y.len + j) * v;
        to.* = sum;
    }
}

/// The ranks' sum against the fp64 layer, of the largest output: bf16 roundings that fall the other way.
const host_bound = 4e-3;

test "two ranks hold the whole layer's heads: the same latents and attention, partials that sum to its output, which is the host's" {
    var f: Fixture = undefined;
    try f.init(32);
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const plan = try windows(arena, 60);
    var all: [fx.layers]dsa.Weights = undefined;
    for (&all, 0..) |*one, layer| one.* = try fx.whole(arena, f.source, &f.c, layer, .{ .pool = &f.pool, .count = 60 });
    defer for (&all) |*one| one.deinit();
    try testing.expectEqual(@as(usize, fx.heads), all[0].heads);
    const want = try play(arena, &all, f.x, plan);

    var parts: [layout.world]Played = undefined;
    const half = fx.heads / 2;
    for (&parts, 0..) |*part, rank| {
        var weights = try f.load(rank, 60);
        defer for (&weights) |*one| one.deinit();
        try testing.expectEqual(@as(usize, half), weights[0].heads);
        part.* = try play(arena, &weights, f.x, plan);
        for (0..fx.layers) |index| try testing.expectEqualSlices(u8, want.cache[index], part.cache[index]);
        // a head's attention is the whole layer's to the bit, whichever block of 16 it falls in
        try testing.expectEqual(all[0].q_b.plain().?.sk, weights[0].q_b.plain().?.sk);
        const rows = want.ol.len / (fx.heads * dsa.latent);
        for (0..rows) |r| try testing.expectEqualSlices(u16, want.ol[(r * fx.heads + rank * half) * dsa.latent ..][0 .. half * dsa.latent], part.ol[r * half * dsa.latent ..][0 .. half * dsa.latent]);
    }
    const x = try shared.fetch(u16, arena, f.x);
    const modelled = try arena.alloc(f64, fx.dims);
    var lat: [dsa.latent]f64 = undefined;
    var worst: f64 = 0;
    var scale: f64 = 0;
    var off: f64 = 0;
    var off_lat: f64 = 0;
    for (0..fx.layers) |index| {
        for (0..fx.total * fx.dims) |i| {
            const whole: f64 = @as(f32, @bitCast(want.out[index][i]));
            var sum: f64 = 0;
            for (parts) |part| sum += @as(f32, @bitCast(part.out[index][i]));
            worst = @max(worst, @abs(sum - whole));
            scale = @max(scale, @abs(whole));
        }
        const latents: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, want.cache[index]));
        // the first rows, a window's edge, and both sides of the chunk's boundary
        for ([_]usize{ 0, 1, 2, 59, 60, 511, 512, 513, fx.total - 1 }) |pos| {
            try hostRow(arena, f.source, &f.c, index, x[pos * fx.dims ..][0..fx.dims], latents[0 .. (pos + 1) * dsa.latent], pos, modelled, &lat);
            for (lat, latents[pos * dsa.latent ..][0..dsa.latent]) |model, bits| off_lat = @max(off_lat, @abs(model - numeric.f32FromBf16(bits)));
            for (modelled, 0..) |model, d| {
                var sum: f64 = 0;
                for (parts) |part| sum += @as(f32, @bitCast(part.out[index][pos * fx.dims + d]));
                off = @max(off, @abs(sum - model));
            }
        }
    }
    std.debug.print("dsa ranks against the whole layer: {e} of {e}; against the host: {e}, latents {e}\n", .{ worst, scale, off, off_lat });
    try testing.expect(scale > 1e-3);
    try testing.expect(worst <= 2e-5 * scale);
    try testing.expect(off <= host_bound * scale);
    try testing.expect(off_lat <= 0.05);
}

/// What holds absorb and expand to their fp64 sums. An element is an fp32 chain of at most 32 steps of 16
/// products and then one rounding to bf16: half a bf16 step of the sum (2^-8 of it) and the chain's fp32
/// roundings, each under 2^-24 of the magnitudes summed so far (48 of them counted, 16 for a step's inside).
fn nearSum(exact: f64, magnitudes: f64) f64 {
    return @abs(exact) / 256.0 + 3e-6 * magnitudes;
}

test "absorb and expand: every element near its fp64 sum, and a row's bits the same alone, in any window and at any slot" {
    const heads = 3;
    // a key width that is no power of two; values that end inside an 8-output tile, past a warp's 32 and a block's 128
    const key_dim = 48;
    const total = 70;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(68);
    const random = prng.random();
    var worst: f64 = 0;
    for ([_]usize{ 44, 136 }) |value_dim| {
        const per_head = key_dim + value_dim;
        const lw = heads * dsa.latent;
        const ow = heads * value_dim;
        const w = try arena.alloc(u16, heads * per_head * dsa.latent);
        shared.randomBf16(random, w, 0.1);
        const q = try arena.alloc(u16, total * heads * key_dim);
        shared.randomBf16(random, q, 1.0);
        const ol = try arena.alloc(u16, total * lw);
        shared.randomBf16(random, ol, 1.0);
        // a row of zeros among the others
        @memset(q[5 * heads * key_dim ..][0 .. heads * key_dim], 0);
        @memset(ol[5 * lw ..][0..lw], 0);
        // the queries again as the first columns of wider rows
        const stride = heads * key_dim + 12;
        const q_wide = try arena.alloc(u16, total * stride);
        @memset(q_wide, 0x7fc0);
        for (0..total) |r| @memcpy(q_wide[r * stride ..][0 .. heads * key_dim], q[r * heads * key_dim ..][0 .. heads * key_dim]);
        const nans = try arena.alloc(u16, total * lw);
        @memset(nans, 0x7fc1);

        var w_dev = try device(u16, .bf16, &.{ heads, per_head, dsa.latent }, w);
        defer w_dev.deinit();
        var q_dev = try device(u16, .bf16, &.{ total, heads * key_dim }, q);
        defer q_dev.deinit();
        var q_wide_dev = try device(u16, .bf16, &.{ total, stride }, q_wide);
        defer q_wide_dev.deinit();
        var ol_dev = try device(u16, .bf16, &.{ total, heads, dsa.latent }, ol);
        defer ol_dev.deinit();
        var qa_dev = try device(u16, .bf16, &.{ total, heads, dsa.latent }, nans);
        defer qa_dev.deinit();
        var o_dev = try device(u16, .bf16, &.{ total, ow }, nans[0 .. total * ow]);
        defer o_dev.deinit();

        // every row alone: a serial step's launch
        for (0..total) |r| {
            try dsa.absorb(q_dev.span(r, 1), w_dev.view, key_dim, qa_dev.span(r, 1));
            try dsa.expand(ol_dev.span(r, 1), w_dev.view, key_dim, o_dev.span(r, 1));
        }
        const qa = try shared.fetch(u16, arena, qa_dev);
        const o = try shared.fetch(u16, arena, o_dev);

        for (0..total) |r| for (0..heads) |h| {
            for (0..dsa.latent) |n| {
                var sum: f64 = 0;
                var magnitudes: f64 = 0;
                for (0..key_dim) |k| {
                    const term = val(q, (r * heads + h) * key_dim + k) * val(w, (h * per_head + k) * dsa.latent + n);
                    sum += term;
                    magnitudes += @abs(term);
                }
                const off = @abs(val(qa, (r * heads + h) * dsa.latent + n) - sum);
                if (magnitudes > 0) worst = @max(worst, off / nearSum(sum, magnitudes));
                try testing.expect(off <= nearSum(sum, magnitudes));
            }
            for (0..value_dim) |n| {
                var sum: f64 = 0;
                var magnitudes: f64 = 0;
                for (0..dsa.latent) |k| {
                    const term = val(ol, (r * heads + h) * dsa.latent + k) * val(w, (h * per_head + key_dim + n) * dsa.latent + k);
                    sum += term;
                    magnitudes += @abs(term);
                }
                const off = @abs(val(o, (r * heads + h) * value_dim + n) - sum);
                if (magnitudes > 0) worst = @max(worst, off / nearSum(sum, magnitudes));
                try testing.expect(off <= nearSum(sum, magnitudes));
            }
        };

        // windows of (first row, rows): a row at every slot of an mma, tiles full and cut, one tile of rows a
        // warp (to 16 rows) and two, a block's rows cut in its first tile and in its second
        for ([_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 15 }, .{ 0, 16 }, .{ 3, 17 }, .{ 9, 31 }, .{ 10, 32 }, .{ 5, 33 }, .{ 7, 40 }, .{ 6, 64 }, .{ 21, 49 }, .{ 0, total } }, 0..) |window, turn| {
            // expand's asking forms (`ZSPARK_GLM_MLA_ASK`) in turn, against the rows the base kernel gave alone
            dsa.setAsk(([_]i32{ 0, -1, 1, 2, 5 })[turn % 5]);
            defer dsa.setAsk(0);
            const first = window[0];
            const rows = window[1];
            // what the window does not write stays: a store past its rows would show beside it
            var held = try device(u16, .bf16, &.{ total, heads, dsa.latent }, nans);
            defer held.deinit();
            var held_o = try device(u16, .bf16, &.{ total, ow }, nans[0 .. total * ow]);
            defer held_o.deinit();
            // the strided queries for the odd windows
            var q_view = q_dev.span(first, rows);
            if (rows % 2 == 1) {
                q_view = .init(@ptrCast(@as([*]u8, @ptrCast(q_wide_dev.view.ptr.?)) + first * stride * 2), .bf16, &.{ rows, heads * key_dim });
                q_view.row_stride = stride;
            }
            try dsa.absorb(q_view, w_dev.view, key_dim, held.span(first, rows));
            try dsa.expand(ol_dev.span(first, rows), w_dev.view, key_dim, held_o.span(first, rows));
            const got = try shared.fetch(u16, arena, held);
            const got_o = try shared.fetch(u16, arena, held_o);
            try testing.expectEqualSlices(u16, qa[first * lw ..][0 .. rows * lw], got[first * lw ..][0 .. rows * lw]);
            try testing.expectEqualSlices(u16, o[first * ow ..][0 .. rows * ow], got_o[first * ow ..][0 .. rows * ow]);
            for (got[0 .. first * lw]) |one| try testing.expectEqual(@as(u16, 0x7fc1), one);
            for (got[(first + rows) * lw ..]) |one| try testing.expectEqual(@as(u16, 0x7fc1), one);
            for (got_o[0 .. first * ow]) |one| try testing.expectEqual(@as(u16, 0x7fc1), one);
            for (got_o[(first + rows) * ow ..]) |one| try testing.expectEqual(@as(u16, 0x7fc1), one);
        }
    }
    std.debug.print("dsa absorb and expand against fp64: the worst element at {d:.3} of its bound\n", .{worst});
}
