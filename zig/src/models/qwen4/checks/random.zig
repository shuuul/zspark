//! A random Flash Next sliced into its ranks' shares (spec 050), from a seed: what `zspark ranks-check random`
//! runs on both ranks and the tests hold twice. Every tensor is drawn whole from one stream before it is cut,
//! so ranks `first..first + held` of a seed are the same bytes whichever process makes them. `sharesDrafting`
//! gives rank 0 a random MTP head, whole beside its share (spec 090), drawn from a stream of its own;
//! `sharesSplit` gives each rank its half of that head (`model.Mtp.split`). `sharesOf` with `caches` adds an
//! attention layer after the DeltaNet ones, a rank its half of the heads, its caches in that storage (spec 094).

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("../blocks/attention.zig");
const cuda = @import("../../../cuda/cuda.zig");
const experts = @import("../../../weights/experts.zig");
const face = @import("../../../weights/face.zig");
const numeric = @import("../../../weights/numeric.zig");
const bf16 = @import("../../../weights/bf16.zig");
const gdn = @import("../blocks/gdn.zig");
const moe = @import("../blocks/moe.zig");
const model = @import("../model.zig");
const share_file = @import("../share.zig");

const world: usize = share_file.world;

pub const fx = struct {
    /// The whole model's shape. Its expert width is twice the tests' random model's, so a rank's half is 64:
    /// a bf16 face's K is a multiple of 64.
    pub const shape: model.Shape = .{ .hidden = 256, .streams = 4, .low = 64, .eps = 1e-6, .vocab = 320, .top = 2, .routed = 8, .width = expert_width };
    pub const layers = 3;
    pub const expert_width = 128;
    /// The MTP head's attention, whole on rank 0: three query heads a KV head, as the tests' random head.
    pub const head_attention: attention.Shape = .{ .dims = shape.hidden, .heads = 6, .kv_heads = 2, .head_dim = 32, .index_heads = 2, .index_dim = 64, .rotary_dim = 16, .rope_theta = 1e7, .budget = 16, .ratio = 4, .eps = shape.eps };

    /// One rank's shape: half the DeltaNet's heads and half the experts' width.
    pub fn half() model.Shape {
        var out = shape;
        out.width = expert_width / world;
        out.delta_net.key_heads /= world;
        out.delta_net.value_heads /= world;
        return out;
    }
};

pub fn randomBf16(arena: Allocator, random: std.Random, count: usize, spread: f32) ![]const u8 {
    const bits = try arena.alloc(u16, count);
    for (bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32) * spread);
    return std.mem.sliceAsBytes(bits);
}

pub fn randomFloats(arena: Allocator, random: std.Random, count: usize, base: f32, spread: f32) ![]f32 {
    const out = try arena.alloc(f32, count);
    for (out) |*v| v.* = base + random.floatNorm(f32) * spread;
    return out;
}

/// NVFP4 experts [routed][n, k] of random codes.
pub fn randomMatrix(arena: Allocator, random: std.Random, n: usize, k: usize) !experts.Matrix {
    const count = fx.shape.routed;
    const codes = try arena.alloc(u8, count * n * k / 2);
    random.bytes(codes);
    const block_scales = try arena.alloc(u8, count * n * k / 16);
    for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
    return .{
        .codes = try experts.Matrix.parts(arena, codes, count),
        .block_scales = try experts.Matrix.parts(arena, block_scales, count),
        .scales = try randomFloats(arena, random, count, 0.02, 0.0),
    };
}

fn halfRows(bytes: []const u8, n: usize, rank: usize) []const u8 {
    const row = bytes.len / n;
    const half = n / world;
    return bytes[rank * half * row ..][0 .. half * row];
}

fn halfCols(arena: Allocator, bytes: []const u8, n: usize, row_bytes: usize, rank: usize) ![]const u8 {
    const half = row_bytes / world;
    const out = try arena.alloc(u8, n * half);
    for (0..n) |r| @memcpy(out[r * half ..][0..half], bytes[r * row_bytes + rank * half ..][0..half]);
    return out;
}

/// A rank's rows of a DeltaNet [q | k | v] tensor, `inner` bytes a row.
fn halfQkv(arena: Allocator, bytes: []const u8, q: usize, v: usize, inner: usize, rank: usize) ![]const u8 {
    const qh = q / world;
    const vh = v / world;
    const out = try arena.alloc(u8, (2 * qh + vh) * inner);
    const segments = [_]struct { at: usize, n: usize, src: usize }{
        .{ .at = 0, .n = qh, .src = rank * qh },
        .{ .at = qh, .n = qh, .src = q + rank * qh },
        .{ .at = 2 * qh, .n = vh, .src = 2 * q + rank * vh },
    };
    for (segments) |seg| @memcpy(out[seg.at * inner ..][0 .. seg.n * inner], bytes[seg.src * inner ..][0 .. seg.n * inner]);
    return out;
}

/// A rank's half of the shared expert's [gate | up] rows.
fn halfGateUp(arena: Allocator, bytes: []const u8, width: usize, hidden: usize, rank: usize) ![]const u8 {
    const row = hidden * 2;
    const half = width / world;
    const out = try arena.alloc(u8, 2 * half * row);
    @memcpy(out[0 .. half * row], bytes[rank * half * row ..][0 .. half * row]);
    @memcpy(out[half * row ..], bytes[(width + rank * half) * row ..][0 .. half * row]);
    return out;
}

/// A connection is whole on every rank: drawn once, made for each held rank.
fn connections(comptime held: usize, arena: Allocator, random: std.Random, inject: bool, scratch: [held]cuda.Scratch.Rows) ![held]model.HyperConnection {
    const s = fx.shape;
    const wide = s.wide();
    const down = try randomBf16(arena, random, s.low * wide, 0.05);
    const gates: ?[]const u8 = if (inject) try randomBf16(arena, random, s.streams * wide, 0.05) else null;
    const up = try randomBf16(arena, random, wide * s.low, 0.2);
    const scale = try randomFloats(arena, random, wide, 1.0, 0.1);
    var out: [held]model.HyperConnection = undefined;
    for (&out, scratch) |*one, rows| one.* = try .init(down, gates, up, scale, s.low, rows);
    return out;
}

const Moe = struct {
    router: []const u8,
    gate: experts.Matrix,
    up: experts.Matrix,
    down: experts.Matrix,
    shared_gate_up: []const u8,
    shared_down: []const u8,

    fn draw(arena: Allocator, random: std.Random) !Moe {
        const full = fx.shape;
        const width = fx.expert_width;
        // in the order the stream is drawn
        const gate = try randomMatrix(arena, random, width, full.hidden);
        const up = try randomMatrix(arena, random, width, full.hidden);
        const down = try randomMatrix(arena, random, full.hidden, width);
        const router = try randomBf16(arena, random, (full.routed + 1) * full.hidden, 0.3);
        const shared_gate_up = try randomBf16(arena, random, 2 * width * full.hidden, 0.06);
        const shared_down = try randomBf16(arena, random, full.hidden * width, 0.1);
        return .{ .router = router, .gate = gate, .up = up, .down = down, .shared_gate_up = shared_gate_up, .shared_down = shared_down };
    }

    /// Every expert at its whole width: the MTP head's, which is not split.
    fn whole(self: Moe, gpa: Allocator, staging: *experts.Staging, scratch: cuda.Scratch.Rows) !moe.Weights {
        const full = fx.shape;
        const width = fx.expert_width;
        return .{
            .router = try bf16.Linear.init(self.router, full.routed + 1, full.hidden, scratch),
            .stored = .{ .nvfp4 = .{
                .routed = .{ .nvfp4 = try experts.Experts.init(gpa, staging, full.routed, width, full.hidden, self.gate, self.up, self.down) },
                .shared_gate_up = try face.Face.init(self.shared_gate_up, 2 * width, full.hidden, scratch),
                .shared_down = try face.Face.init(self.shared_down, full.hidden, width, scratch),
            } },
        };
    }

    fn of(self: Moe, gpa: Allocator, arena: Allocator, staging: *experts.Staging, scratch: cuda.Scratch.Rows, rank: usize) !moe.Weights {
        const full = fx.shape;
        const width = fx.expert_width;
        const half = fx.half();
        return .{
            .router = try bf16.Linear.init(self.router, full.routed + 1, full.hidden, scratch),
            .stored = .{ .nvfp4 = .{
                .routed = .{ .nvfp4 = try experts.Experts.init(
                    gpa,
                    staging,
                    full.routed,
                    half.width,
                    full.hidden,
                    try self.gate.rowsOf(arena, width, rank, world),
                    try self.up.rowsOf(arena, width, rank, world),
                    try self.down.colsOf(arena, full.hidden, width, rank, world),
                ) },
                .shared_gate_up = try face.Face.init(try halfGateUp(arena, self.shared_gate_up, width, full.hidden, rank), width, full.hidden, scratch),
                .shared_down = try face.Face.init(try halfCols(arena, self.shared_down, full.hidden, width * 2, rank), full.hidden, half.width, scratch),
            } },
        };
    }
};

/// Ranks `first..first + held` of the random model of `seed`, each with windows of up to `max_rows` rows. A
/// model owns its scratch pool (`model.Model.deinit`).
pub fn shares(comptime held: usize, gpa: Allocator, arena: Allocator, seed: u64, first: usize, max_rows: usize) ![held]model.Model {
    return sharesWith(held, gpa, arena, seed, first, max_rows, .{ .head = .none });
}

/// `shares`, rank 0 with a random MTP head whole beside its share, as `Model.loadWith` leaves a checkpoint's
/// (`model.Mtp.whole`); rank 1 has none. The trunk's bytes are `shares`'.
pub fn sharesDrafting(comptime held: usize, gpa: Allocator, arena: Allocator, seed: u64, first: usize, max_rows: usize) ![held]model.Model {
    return sharesWith(held, gpa, arena, seed, first, max_rows, .{ .head = .whole });
}

/// `sharesDrafting`'s head split: each rank its half of the attention's heads and of the experts' width, its
/// part of the draft ids, the rest whole, as `Model.loadWith` leaves a checkpoint's under `head_split`. The
/// head's bytes are `sharesDrafting`'s, cut.
pub fn sharesSplit(comptime held: usize, gpa: Allocator, arena: Allocator, seed: u64, first: usize, max_rows: usize) ![held]model.Model {
    return sharesWith(held, gpa, arena, seed, first, max_rows, .{ .head = .split });
}

pub const Head = enum { none, whole, split };

/// What a random model holds beside its DeltaNet layers.
pub const Extra = struct {
    head: Head = .none,
    /// An attention layer after the DeltaNet ones, as the checkpoint's fourth: `splitAttention` whole, a rank
    /// its half of the query and KV heads and of the output's columns, the indexer whole, and each rank's
    /// caches in this storage. The head's own cache stays bf16 (`layout.head_storage`). Null: none, the
    /// model `shares` gives.
    caches: ?attention.Storage = null,
};

/// `shares` with a head and an attention layer as `extra` says.
pub fn sharesOf(comptime held: usize, gpa: Allocator, arena: Allocator, seed: u64, first: usize, max_rows: usize, extra: Extra) ![held]model.Model {
    return sharesWith(held, gpa, arena, seed, first, max_rows, extra);
}

/// A split head's attention, whole: twice `fx.head_attention`'s heads, so a rank's half is that shape (half of
/// its own six heads' columns is no multiple of 64, which a bf16 linear wants).
fn splitAttention() attention.Shape {
    var out = fx.head_attention;
    out.heads *= world;
    out.kv_heads *= world;
    return out;
}

fn deviceFloats(values: []const f32) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(.f32, &.{values.len});
    errdefer buffer.deinit();
    try buffer.upload(std.mem.sliceAsBytes(values));
    return buffer;
}

/// A random MTP head of the whole model's sizes, its drafts scored over the ids of `head_bytes` [vocab,
/// hidden] that are not multiples of three: an odd count, so a split head's ranks hold parts of two lengths.
/// `cut`: that rank's half of it, every tensor drawn whole first. A failed init leaks what it made: the
/// process stops on it.
fn randomHead(gpa: Allocator, arena: Allocator, seed: u64, head_bytes: []const u8, staging: *experts.Staging, rows: cuda.Scratch.Rows, cut: ?usize, trunk: attention.Storage) !model.Mtp {
    const s = fx.shape;
    const shape = if (cut != null) splitAttention() else fx.head_attention;
    var prng: std.Random.DefaultPrng = .init(seed ^ 0x6d74705f68656164);
    const random = prng.random();
    const spread = 1.0 / @sqrt(@as(f32, @floatFromInt(s.hidden)));
    const one = [1]cuda.Scratch.Rows{rows};
    const drawn: Moe = try .draw(arena, random);
    var ids: std.ArrayList(u32) = .empty;
    for (0..s.vocab) |id| if (id % 3 != 0) try ids.append(arena, @intCast(id));
    const norm_embedding = try deviceFloats(try randomFloats(arena, random, s.hidden, 1.0, 0.1));
    const norm_hidden = try deviceFloats(try randomFloats(arena, random, s.wide(), 1.0, 0.1));
    const fc_embedding: face.Face = try .init(try randomBf16(arena, random, s.hidden * s.hidden, spread), s.hidden, s.hidden, rows);
    const fc_hidden: face.Face = try .init(try randomBf16(arena, random, s.hidden * s.hidden, spread), s.hidden, s.hidden, rows.times(s.streams));
    const attn_hc = (try connections(1, arena, random, true, one))[0];
    const mlp_hc = (try connections(1, arena, random, true, one))[0];
    const q_rows = shape.heads * 2 * shape.head_dim;
    const kv_rows = shape.kv_heads * shape.head_dim;
    var host: attention.Weights.Host = .{
        .q = try randomBf16(arena, random, q_rows * s.hidden, spread),
        .k = try randomBf16(arena, random, kv_rows * s.hidden, spread),
        .v = try randomBf16(arena, random, kv_rows * s.hidden, spread),
        .index_qk = try randomBf16(arena, random, (shape.index_heads + 1) * shape.index_dim * s.hidden, spread),
        .o = try randomBf16(arena, random, s.hidden * shape.out(), 0.1),
        .q_norm = try randomFloats(arena, random, shape.head_dim, 1.0, 0.1),
        .k_norm = try randomFloats(arena, random, shape.head_dim, 1.0, 0.1),
        .index_q_norm = try randomFloats(arena, random, shape.index_dim, 1.0, 0.1),
        .index_k_norm = try randomFloats(arena, random, shape.index_dim, 1.0, 0.1),
    };
    const mixer = (try connections(1, arena, random, false, one))[0];
    if (cut) |rank| {
        // the projections' rows are a head's, the output's columns: half of each, the indexer whole
        host.q = halfRows(host.q, q_rows, rank);
        host.k = halfRows(host.k, kv_rows, rank);
        host.v = halfRows(host.v, kv_rows, rank);
        host.o = try halfCols(arena, host.o, s.hidden, shape.out() * 2, rank);
    }
    return .{
        .norm_embedding = norm_embedding,
        .norm_hidden = norm_hidden,
        .fc_embedding = fc_embedding,
        .fc_hidden = fc_hidden,
        .layer = .{
            .attn_hc = attn_hc,
            .mlp_hc = mlp_hc,
            .block = .{ .attention = try .init(fx.head_attention, host, rows) },
            .moe = if (cut) |rank| try drawn.of(gpa, arena, staging, rows, rank) else try drawn.whole(gpa, staging, rows),
        },
        .mixer = mixer,
        .draft = try .init(gpa, .{ .bf16 = head_bytes }, s.vocab, s.hidden, if (cut) |rank| model.Mtp.Split.partOf(ids.items, rank) else ids.items),
        .max_rows = rows.count,
        // as `Model.loadWith` leaves it: the shape says the trunk's storage, `head` the head's own cache's
        .whole = if (cut != null) null else .{ .shape = s, .attention = .{ .shape = stored(shape, trunk), .rotary = try .init(shape), .head = .bf16 } },
        .split = if (cut != null) .{ .ids = try gpa.dupe(u32, ids.items) } else null,
    };
}

fn stored(shape: attention.Shape, kv: attention.Storage) attention.Shape {
    var out = shape;
    out.kv = kv;
    return out;
}

/// A rank's half of an attention layer drawn whole (`splitAttention`): the projections' rows are a head's and
/// the output's columns, half of each; the indexer and the norms whole.
fn halfAttention(arena: Allocator, whole: attention.Weights.Host, rank: usize) !attention.Weights.Host {
    const shape = splitAttention();
    var out = whole;
    out.q = halfRows(whole.q, shape.heads * 2 * shape.head_dim, rank);
    out.k = halfRows(whole.k, shape.kv_heads * shape.head_dim, rank);
    out.v = halfRows(whole.v, shape.kv_heads * shape.head_dim, rank);
    out.o = try halfCols(arena, whole.o, fx.shape.hidden, shape.out() * 2, rank);
    return out;
}

fn randomAttention(arena: Allocator, random: std.Random, shape: attention.Shape) !attention.Weights.Host {
    const s = fx.shape;
    const spread = 1.0 / @sqrt(@as(f32, @floatFromInt(s.hidden)));
    return .{
        .q = try randomBf16(arena, random, shape.heads * 2 * shape.head_dim * s.hidden, spread),
        .k = try randomBf16(arena, random, shape.kv_heads * shape.head_dim * s.hidden, spread),
        .v = try randomBf16(arena, random, shape.kv_heads * shape.head_dim * s.hidden, spread),
        .index_qk = try randomBf16(arena, random, (shape.index_heads + 1) * shape.index_dim * s.hidden, spread),
        .o = try randomBf16(arena, random, s.hidden * shape.out(), 0.1),
        .q_norm = try randomFloats(arena, random, shape.head_dim, 1.0, 0.1),
        .k_norm = try randomFloats(arena, random, shape.head_dim, 1.0, 0.1),
        .index_q_norm = try randomFloats(arena, random, shape.index_dim, 1.0, 0.1),
        .index_k_norm = try randomFloats(arena, random, shape.index_dim, 1.0, 0.1),
    };
}

fn sharesWith(comptime held: usize, gpa: Allocator, arena: Allocator, seed: u64, first: usize, max_rows: usize, extra: Extra) ![held]model.Model {
    const head = extra.head;
    std.debug.assert(first + held <= world);
    const full = fx.shape;
    const half = fx.half();
    const dn = full.delta_net;
    const q = dn.key_heads * gdn.key_dim;
    const v = dn.value_heads * gdn.value_dim;
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    var staging: experts.Staging = .{};
    defer staging.deinit(gpa);
    var pools: [held]*cuda.Scratch = undefined;
    var made: usize = 0;
    errdefer for (pools[0..made]) |pool| pool.destroy();
    for (&pools) |*pool| {
        pool.* = try .create(gpa);
        made += 1;
    }
    var scratch: [held]cuda.Scratch.Rows = undefined;
    for (&scratch, pools) |*rows, pool| rows.* = pool.rows(max_rows);

    const embed_bytes = try randomBf16(arena, random, full.vocab * full.hidden, 1.0);
    var embed: [held]cuda.Buffer = undefined;
    for (&embed) |*table| {
        table.* = try .init(.bf16, &.{ full.vocab, full.hidden });
        try table.upload(embed_bytes);
    }

    var layers: [held][]model.Layer = undefined;
    for (&layers) |*one| one.* = try gpa.alloc(model.Layer, fx.layers + @as(usize, @intFromBool(extra.caches != null)));
    for (0..fx.layers) |index| {
        const attn = try connections(held, arena, random, true, scratch);
        const mlp = try connections(held, arena, random, true, scratch);
        const qkv = try randomBf16(arena, random, dn.conv() * full.hidden, 0.06);
        const z = try randomBf16(arena, random, dn.out() * full.hidden, 0.06);
        const b = try randomBf16(arena, random, dn.value_heads * full.hidden, 0.06);
        const a = try randomBf16(arena, random, dn.value_heads * full.hidden, 0.06);
        const conv = try randomBf16(arena, random, dn.conv() * gdn.taps, 0.5);
        const a_log = try randomFloats(arena, random, dn.value_heads, -1.0, 0.5);
        const dt_bias = try randomFloats(arena, random, dn.value_heads, 0.0, 1.0);
        const norm = try randomBf16(arena, random, gdn.value_dim, 1.0);
        const out = try randomBf16(arena, random, full.hidden * dn.out(), 0.02);
        const experts_drawn: Moe = try .draw(arena, random);
        for (0..held) |i| {
            const rank = first + i;
            const heads = half.delta_net.value_heads;
            layers[i][index] = .{
                .attn_hc = attn[i],
                .mlp_hc = mlp[i],
                .block = .{ .delta_net = try .init(.{
                    .projection = &.{
                        try halfQkv(arena, qkv, q, v, full.hidden * 2, rank),
                        halfRows(z, dn.out(), rank),
                        halfRows(b, dn.value_heads, rank),
                        halfRows(a, dn.value_heads, rank),
                    },
                    .conv = try halfQkv(arena, conv, q, v, gdn.taps * 2, rank),
                    .a_log = a_log[rank * heads ..][0..heads],
                    .dt_bias = dt_bias[rank * heads ..][0..heads],
                    .norm = norm,
                    .out = try halfCols(arena, out, full.hidden, dn.out() * 2, rank),
                }, half, scratch[i]) },
                .moe = try experts_drawn.of(gpa, arena, &staging, scratch[i], rank),
            };
        }
    }
    // a rank's attention is `fx.head_attention`'s sizes, the half of `splitAttention`, as a split head's is:
    // a model has one shape for both (`model.Model.attention`)
    const attending: ?attention.Shape = if (extra.caches) |kv| stored(fx.head_attention, kv) else null;
    if (attending) |shape| {
        const attn = try connections(held, arena, random, true, scratch);
        const mlp = try connections(held, arena, random, true, scratch);
        const drawn = try randomAttention(arena, random, splitAttention());
        const experts_drawn: Moe = try .draw(arena, random);
        for (0..held) |i| {
            layers[i][fx.layers] = .{
                .attn_hc = attn[i],
                .mlp_hc = mlp[i],
                .block = .{ .attention = try .init(shape, try halfAttention(arena, drawn, first + i), scratch[i]) },
                .moe = try experts_drawn.of(gpa, arena, &staging, scratch[i], first + i),
            };
        }
    }
    const mix = try connections(held, arena, random, false, scratch);
    const head_bytes = try randomBf16(arena, random, full.vocab * full.hidden, 0.1);
    var out: [held]model.Model = undefined;
    for (&out, 0..) |*one, i| {
        one.* = .{
            .shape = half,
            .embed = embed[i],
            .layers = layers[i],
            .mixer = mix[i],
            .head = try .init(halfRows(head_bytes, full.vocab, first + i), full.vocab / world, full.hidden, scratch[i]),
            .max_rows = max_rows,
            .scratch = pools[i],
            .gpa = gpa,
        };
        const trunk: attention.Storage = extra.caches orelse .bf16;
        switch (head) {
            .none => {},
            .whole => if (first + i == 0) {
                one.mtp = try randomHead(gpa, arena, seed, head_bytes, &staging, scratch[i], null, trunk);
            },
            .split => one.mtp = try randomHead(gpa, arena, seed, head_bytes, &staging, scratch[i], first + i, trunk),
        }
        // the trunk's layer and a split head's half share it; the head's own cache is bf16 beside a coded trunk
        if (attending != null or head == .split) {
            const shape = attending orelse fx.head_attention;
            one.attention = .{ .shape = shape, .rotary = try .init(shape), .head = .bf16 };
        }
    }
    return out;
}
