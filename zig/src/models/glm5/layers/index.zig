//! GLM-5.3-Flash's indexer on the host: the cache's bytes, the key's norm and the selection in the order
//! TensorFold's `cuda/sparse.py` takes (`_index_write`, `_select_rows`, `select_tokens`), and the pool, the
//! rotation, the FP8 coding and the score as the card's vLLM (`index.cu`, spec 116 WS-05). No device. A row's
//! list is its own query against the pools complete before it, ties to the lower pool, so a row among others is
//! that row alone.
//!
//! The score's dot here is an fp32 sum of e4m3 products, one at a time; the device's is on the FP8 tensor cores,
//! 32 products a step: a score is held near this one, not to the bit, and a list that parts from it is a tie
//! when the two pools' scores meet at the cut. `scoreBf16` is TensorFold's score over bf16 pooled keys and
//! queries, which its fixtures record, to measure what the coding moves.

const std = @import("std");
const numeric = @import("../../../weights/numeric.zig");
const gib = @import("../../../cuda/units.zig").gib;

/// DSA layers and the MTP layer's own cache.
pub const caches = 12;
/// A latent's width, an index key's, and the tokens a pool gathers.
pub const latent_dim = 512;
pub const index_dim = 128;
pub const pool = 4;
/// LayerNorm's epsilon, `sparse._index_write`'s.
pub const eps: f32 = 1e-6;

/// What one token stores in one layer's cache: the latent bf16 or FP8; the indexer's keys and gates bf16 and
/// its pooled keys FP8 either way.
pub const Storage = enum { bf16, fp8_latent };
/// e4m3's largest finite value.
pub const fp8_max: f32 = 448;

pub const Item = struct {
    /// Bytes of the latent, the index key, the gate, and one token's share of a pooled key.
    latent: u64,
    key: u64,
    gate: u64,
    pooled: u64,

    pub fn aToken(self: Item) u64 {
        return self.latent + self.key + self.gate + self.pooled;
    }
};

/// One layer's bytes a token. The latent as FP8 is half its bf16 row (e4m3, the scale not stored); the
/// indexer's keys and gates bf16, a pooled key 128 e4m3 codes and an fp32 scale, 132 bytes over its four
/// tokens. Twelve caches are `bytesAToken`.
pub fn item(storage: Storage) Item {
    return .{
        .latent = if (storage == .bf16) latent_dim * 2 else latent_dim,
        .key = index_dim * 2,
        .gate = index_dim * 2,
        .pooled = (index_dim + @sizeOf(f32)) / pool,
    };
}

pub fn bytesAToken(storage: Storage) u64 {
    return item(storage).aToken() * caches;
}

/// One rank's cache at `tokens`, the twelve caches. Both ranks hold the same rows.
pub fn rankBytes(tokens: u64, storage: Storage) u64 {
    return bytesAToken(storage) * tokens;
}

/// Tokens a dense row reads: `index_topk + index_kpool - 1`.
pub fn denseLimit(top_tokens: usize, pool_size: usize) usize {
    return top_tokens + pool_size - 1;
}

/// Pools a sparse row keeps: `index_topk / index_kpool`.
pub fn keptPools(top_tokens: usize, pool_size: usize) usize {
    return top_tokens / pool_size;
}

/// The selected list's width: the kept pools' tokens and the unfinished pool's, less its last.
pub fn listWidth(kept: usize, pool_size: usize) usize {
    return kept * pool_size + pool_size - 1;
}

/// Pools complete at `position` (the token itself included): `(position + 1) / pool`.
pub fn completePools(position: usize, pool_size: usize) usize {
    return (position + 1) / pool_size;
}

/// Pools a window at `pos` of `rows` completes: from `pos / pool`, while the pool's last token is in the window.
pub fn poolsCompleted(pos: usize, rows: usize, pool_size: usize) struct { first: usize, count: usize } {
    const first = pos / pool_size;
    const end = (pos + rows) / pool_size;
    return .{ .first = first, .count = if (end > first) end - first else 0 };
}

/// The position a prompt chunk's indexer rows split across the ranks from (`ZSPARK_GLM_INDEX_SPLIT`, spec
/// 152 WS-10, TensorFold 1.2.1's `TF_GLM_INDEX_SPLIT`): over two ranks or more each rank scores and selects
/// its block of the rows and the lists are all-gathered, so every rank holds every list before the attention.
/// A window at or past this position whose chunk splits; 0 never. Decode windows and loopback never split.
/// The same kernels and per-row inputs: a row's list is the full indexer's, bit for bit.
pub var split_from: usize = 2048;

/// Whether a prompt chunk at `start` splits its indexer rows over `world` ranks: from `from` on (0 never),
/// and only over two ranks or more (a process holding every rank picks every row itself).
pub fn splitAt(start: usize, from: usize, world: usize) bool {
    return from > 0 and world > 1 and start >= from;
}

fn value(bits: u16) f32 {
    return numeric.f32FromBf16(bits);
}

/// `wk`'s row after LayerNorm with bias, stored bf16. The mean and the variance are one sum each over the
/// width; the weight and the bias are the stored bf16, promoted as the kernel promotes them.
pub fn normKey(x: []const u16, weight: []const u16, bias: []const u16, out: []u16) void {
    const width: f32 = @floatFromInt(x.len);
    var sum: f32 = 0;
    for (x) |bits| sum += value(bits);
    const mean = sum / width;
    var squares: f32 = 0;
    for (x) |bits| {
        const centred = value(bits) - mean;
        squares += centred * centred;
    }
    const inv = 1.0 / @sqrt(squares / width + eps);
    for (out, x, weight, bias) |*stored, bits, scale, shift| {
        stored.* = numeric.bf16FromF32((value(bits) - mean) * inv * value(scale) + value(shift));
    }
}

/// The gate row stored bf16, as `_index_write` stores `igr`.
pub fn storeGate(gate: []const f32, out: []u16) void {
    for (out, gate) |*stored, v| stored.* = numeric.bf16FromF32(v);
}

/// Pool `p`'s key as vLLM's `_prefill_pool_kernel`: a softmax over the four gates plus `ape` bf16 [4, dim],
/// max first, the keys times their weights and the weights added in token order in fp32, the sum divided by the
/// weights' (at least 1e-20) and rounded once to bf16. `keys` and `gates` are the four tokens' stored rows.
pub fn poolRow(keys: [pool][]const u16, gates: [pool][]const u16, ape: []const u16, out: []u16) void {
    const dim = out.len;
    for (0..dim) |d| {
        var logit: [pool]f32 = undefined;
        var highest: f32 = -std.math.inf(f32);
        for (0..pool) |t| {
            logit[t] = value(gates[t][d]) + value(ape[t * dim + d]);
            highest = @max(highest, logit[t]);
        }
        var numerator: f32 = 0;
        var denominator: f32 = 0;
        for (0..pool) |t| {
            const weight = @exp(logit[t] - highest);
            numerator += value(keys[t][d]) * weight;
            denominator += weight;
        }
        out[d] = numeric.bf16FromF32(numerator / @max(denominator, 1e-20));
    }
}

/// `1 / sqrt(dim)`, the rotation's normalization (vLLM's 0.08838834764831845 at 128).
pub fn rotation(dim: usize) f32 {
    return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim))));
}

/// `1 / sqrt(dim heads)`, folded into each head's weight with its query's scale (1 / 64 at 128 and 32).
pub fn headNorm(dim: usize, heads: usize) f32 {
    return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim * heads))));
}

/// The Walsh-Hadamard transform of `v` (a power of two long) as vLLM's `_fwht128`: stages of stride 1, 2, ..,
/// the lower of a pair a + b and the upper a - b, then times `rotation` and rounded to bf16, in place.
pub fn rotate(v: []f32) void {
    var s: usize = 1;
    while (s < v.len) : (s *= 2) {
        for (0..v.len) |i| {
            if (i & s != 0) continue;
            const a = v[i];
            const b = v[i + s];
            v[i] = a + b;
            v[i + s] = a - b;
        }
    }
    const rot = rotation(v.len);
    for (v) |*x| x.* = numeric.roundBf16(x.* * rot);
}

/// A rotated vector's codes, e4m3 under one power-of-two scale, 2^ceil(log2(max(max|x|, 1e-4) / 448)); the
/// codes x / scale clamped to 448, to nearest even. Returns the scale.
pub fn quantize(v: []const f32, codes: []u8) f32 {
    var most: f32 = 0;
    for (v) |x| most = @max(most, @abs(x));
    const ratio = @max(most, 1e-4) / fp8_max;
    const parts = std.math.frexp(ratio);
    const e = if (parts.significand == 0.5) parts.exponent - 1 else parts.exponent;
    const inv = std.math.ldexp(@as(f32, 1), -e);
    for (codes, v) |*code, x| code.* = numeric.e4m3FromF32(std.math.clamp(x * inv, -fp8_max, fp8_max));
    return std.math.ldexp(@as(f32, 1), e);
}

/// A pooled key bf16 [dim] as the cache stores it: rotated, its codes and its scale.
pub fn encodeKey(pooled: []const u16, codes: []u8) f32 {
    var v: [index_dim]f32 = undefined;
    const row = v[0..pooled.len];
    for (row, pooled) |*x, bits| x.* = value(bits);
    rotate(row);
    return quantize(row, codes);
}

/// A row's queries bf16 [heads, dim] and head weights fp32 [heads] as the score reads them: each head's query
/// rotated and coded, its scale folded into its weight, `(w scale) headNorm`.
pub fn encodeQuery(qi: []const u16, weights: []const f32, codes: []u8, folded: []f32) void {
    const heads = weights.len;
    const dim = qi.len / heads;
    for (0..heads) |h| {
        const scale = encodeKey(qi[h * dim ..][0..dim], codes[h * dim ..][0..dim]);
        folded[h] = (weights[h] * scale) * headNorm(dim, heads);
    }
}

/// One pool's score as b12x's reference: `sum_h relu(q8_h . k8) wq_h`, heads in order, times the key's scale.
/// The dot is an fp32 sum of the codes' products, one at a time.
pub fn scorePool(q8: []const u8, wq: []const f32, codes: []const u8, scale: f32) f32 {
    const dim = codes.len;
    var sum: f32 = 0;
    for (wq, 0..) |w, h| {
        var dot: f32 = 0;
        for (q8[h * dim ..][0..dim], codes) |q, k| dot += numeric.f32FromE4m3(q) * numeric.f32FromE4m3(k);
        sum += @max(dot, 0) * w;
    }
    return sum * scale;
}

/// TensorFold's score over its bf16 pooled key: `sum_h (w_h / sqrt(heads)) relu(qi_h . pool / sqrt(dim))`,
/// heads in order. `qi` is bf16 [heads, dim].
pub fn scoreBf16(qi: []const u16, weights: []const f32, pooled: []const u16) f32 {
    const heads = weights.len;
    const dim = pooled.len;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
    const wscale = 1.0 / @sqrt(@as(f32, @floatFromInt(heads)));
    var sum: f32 = 0;
    for (0..heads) |h| {
        var dot: f32 = 0;
        const query = qi[h * dim ..][0..dim];
        for (query, pooled) |q, k| dot += value(q) * value(k);
        sum += (weights[h] * wscale) * @max(dot * scale, 0);
    }
    return sum;
}

/// A float's unsigned order, `-0` counted as `+0` (`sparse._order_key`).
pub fn orderKey(score: f32) u32 {
    const bits: u32 = @bitCast(score + 0.0);
    const sign: u32 = @bitCast(@as(i32, @bitCast(bits)) >> 31);
    return bits ^ (sign | 0x80000000);
}

/// The `kept` best pools, ties to the lower index, written ascending. `scores` is one a complete pool, in
/// pool order, and at least `kept` long; a pool past the row is `-inf` and loses to every finite score.
pub fn selectPools(gpa: std.mem.Allocator, scores: []const f32, kept: usize, out: []u32) std.mem.Allocator.Error!void {
    std.debug.assert(out.len == kept and scores.len >= kept);
    const order = try gpa.alloc(u32, scores.len);
    defer gpa.free(order);
    for (order, 0..) |*pool_index, n| pool_index.* = @intCast(n);
    const ctx: Sort = .{ .scores = scores };
    std.mem.sort(u32, order, ctx, Sort.higher);
    @memcpy(out, order[0..kept]);
    std.mem.sort(u32, out, {}, struct {
        fn lessThan(_: void, a: u32, b: u32) bool {
            return a < b;
        }
    }.lessThan);
}

const Sort = struct {
    scores: []const f32,

    fn higher(self: Sort, a: u32, b: u32) bool {
        const ka = orderKey(self.scores[a]);
        const kb = orderKey(self.scores[b]);
        if (ka != kb) return ka > kb;
        return a < b;
    }
};

/// The kept pools' tokens in pool order, then the unfinished pool's tokens that this position can see.
/// Returns the count a sparse row attends (0 while the row is still inside the dense window). `tokens` is
/// `listWidth` long and is padded with -1.
pub fn selectTokens(selected: []const u32, position: usize, pool_size: usize, tokens: []i32) usize {
    const kept = selected.len;
    @memset(tokens, -1);
    for (selected, 0..) |p, i| {
        for (0..pool_size) |t| tokens[i * pool_size + t] = @intCast(@as(usize, p) * pool_size + t);
    }
    const npool = completePools(position, pool_size);
    var tail: usize = 0;
    var at = kept * pool_size;
    var tok = npool * pool_size;
    while (tail + 1 < pool_size and tok <= position) {
        tokens[at] = @intCast(tok);
        tail += 1;
        at += 1;
        tok += 1;
    }
    if (npool > kept) return kept * pool_size + tail;
    return 0;
}

/// `info`'s cache lines: the bytes a token by item, and one rank's bytes at `tokens`.
pub fn writeCache(stdout: *std.Io.Writer, tokens: u64) std.Io.Writer.Error!void {
    const bf16_item = item(.bf16);
    const fp8_item = item(.fp8_latent);
    try stdout.print("  cache      {d} bytes a token a layer, {d} over {d} caches\n", .{ bf16_item.aToken(), bytesAToken(.bf16), caches });
    const names = [_][]const u8{ "latent", "index key", "gate", "pooled key" };
    const bf16_bytes = [_]u64{ bf16_item.latent, bf16_item.key, bf16_item.gate, bf16_item.pooled };
    inline for (names, bf16_bytes) |name, bytes| {
        try stdout.print("    {s:<10} {d:>5} a token  {d:>12} a rank at {d}\n", .{ name, bytes, bytes * caches * tokens, tokens });
    }
    const bf16_rank = rankBytes(tokens, .bf16);
    const fp8_rank = rankBytes(tokens, .fp8_latent);
    try stdout.print("    bf16         {d} a rank ({d:.6} GiB); latent fp8 {d} ({d:.6} GiB)\n", .{
        bf16_rank, gib(bf16_rank), fp8_rank, gib(fp8_rank),
    });
    try stdout.print("    fp8 latent   {d} a token a layer (the latent {d}, the indexer unchanged)\n", .{ fp8_item.aToken(), fp8_item.latent });
    try stdout.print("    the pooled keys e4m3 with an fp32 scale, {d} bytes a pool\n", .{index_dim + @sizeOf(f32)});
}
