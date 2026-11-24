//! `perception.zig`'s tests (spec 151 WS-03; `tools/check.sh nemotron perception`): the prep kernel (qk
//! LayerNorm + RoPE) and the chunk attention against the host's fp32 with the model's roundings, a row alone
//! against the row among others bit for bit, and the whole encoder on the fixture's synthetic checkpoint
//! against a host reference of the layer stack — a chunk alone the same bits as among a batch.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const config_file = @import("../../../src/models/nemotron/config.zig");
const perception_file = @import("../../../src/models/nemotron/perception.zig");
const mel_file = @import("../../../src/models/nemotron/mel.zig");
const fixture = @import("fixture.zig");
const Perception = perception_file.Perception;

extern fn zs_nemotron_perception_prep(qkv: *const cuda.View, offsets: *const cuda.View, qw: *const cuda.View, qb: *const cuda.View, kw: *const cuda.View, kb: *const cuda.View, inv: *const cuda.View, heads: i64, rot: i64, eps: f64) c_int;
extern fn zs_nemotron_perception_attention(qkv: *const cuda.View, tiles: *const cuda.View, out: *const cuda.View) c_int;

fn f32v(bits: u16) f32 {
    return perception_file.f32Of(bits);
}
fn bf(x: f32) f32 {
    return perception_file.f32Of(perception_file.bf16Of(x));
}

fn upload(_: std.mem.Allocator, comptime dtype: cuda.DType, shape: []const usize, bytes: []const u8) !cuda.Buffer {
    var b: cuda.Buffer = try .init(dtype, shape);
    errdefer b.deinit();
    try b.upload(bytes);
    return b;
}

fn download(gpa: std.mem.Allocator, v: cuda.View) ![]u16 {
    const out = try gpa.alloc(u16, v.count());
    try cuda.download(v, std.mem.sliceAsBytes(out));
    return out;
}

const DH = 128;
const HEADS = 1;
const D = HEADS * DH;
const ROT = 64;

/// Host reference of `zs_nemotron_perception_prep`: LayerNorm per head on q and k, then RoPE half-split.
fn prepRef(x: []f32, offsets: []const usize, qw: []const f32, qb: []const f32, kw: []const f32, kb: []const f32, inv: []const f32) void {
    const rows = x.len / (3 * D);
    for (0..rows) |r| {
        var seg: usize = 0;
        for (offsets[1..], 1..) |e, i| {
            if (r >= e) seg = i;
        }
        const pos = @as(f32, @floatFromInt(r - offsets[seg]));
        for ([_][]const f32{ qw, kw }, [_][]const f32{ qb, kb }, 0..) |w, b, part| {
            const base = r * 3 * D + part * D;
            for (0..HEADS) |h| {
                const row = base + h * DH;
                var sum: f64 = 0;
                var sq: f64 = 0;
                for (0..DH) |t| {
                    const v = x[row + t];
                    sum += v;
                    sq += @as(f64, v) * v;
                }
                const mean = sum / DH;
                const rstd = 1.0 / @sqrt(sq / DH - mean * mean + 1e-5);
                var normed: [DH]f32 = undefined;
                for (0..DH) |t| normed[t] = bf(@as(f32, @floatCast((x[row + t] - mean) * rstd)) * w[t] + b[t]);
                for (0..ROT / 2) |i| {
                    const a = pos * inv[i];
                    const c = bf(@cos(a));
                    const s = bf(@sin(a));
                    const n1 = normed[i];
                    const n2 = normed[i + ROT / 2];
                    x[row + i] = bf(bf(n1 * c) + bf(-n2 * s));
                    x[row + i + ROT / 2] = bf(bf(n2 * c) + bf(n1 * s));
                }
                for (ROT..DH) |t| x[row + t] = normed[t];
            }
        }
    }
}

fn invFreq() [ROT / 2]f32 {
    var inv: [ROT / 2]f32 = undefined;
    for (&inv, 0..) |*v, i| v.* = @floatCast(1.0 / std.math.pow(f64, 10000, @as(f64, @floatFromInt(2 * i)) / ROT));
    return inv;
}

fn randomBf(gpa: std.mem.Allocator, n: usize, seed: u64) ![]u16 {
    const out = try gpa.alloc(u16, n);
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    for (out) |*v| v.* = perception_file.bf16Of(random.float(f32) - 0.5);
    return out;
}

test "prep: qk LayerNorm and RoPE against the host, a row alone among others" {
    const a = testing.allocator;
    for ([_]usize{ 3, 37 }) |rows| {
        const offs_h = [_]i32{ 0, @intCast(@min(rows, 5)), @intCast(rows) };
        const qkv_h = try randomBf(a, rows * 3 * D, 11);
        defer a.free(qkv_h);
        const qw_h = try randomBf(a, DH, 3);
        const qb_h = try randomBf(a, DH, 4);
        const kw_h = try randomBf(a, DH, 5);
        const kb_h = try randomBf(a, DH, 6);
        defer a.free(qw_h);
        defer a.free(qb_h);
        defer a.free(kw_h);
        defer a.free(kb_h);
        var qkv = try upload(a, .bf16, &.{ rows, 3 * D }, std.mem.sliceAsBytes(qkv_h));
        defer qkv.deinit();
        var offs = try upload(a, .i32, &.{offs_h.len}, std.mem.sliceAsBytes(&offs_h));
        defer offs.deinit();
        var qw = try upload(a, .bf16, &.{DH}, std.mem.sliceAsBytes(qw_h));
        var qb = try upload(a, .bf16, &.{DH}, std.mem.sliceAsBytes(qb_h));
        var kw = try upload(a, .bf16, &.{DH}, std.mem.sliceAsBytes(kw_h));
        var kb = try upload(a, .bf16, &.{DH}, std.mem.sliceAsBytes(kb_h));
        defer qw.deinit();
        defer qb.deinit();
        defer kw.deinit();
        defer kb.deinit();
        const inv = invFreq();
        var inv_b = try upload(a, .f32, &.{ROT / 2}, std.mem.sliceAsBytes(&inv));
        defer inv_b.deinit();
        try cuda.launched(zs_nemotron_perception_prep(&qkv.view, &offs.view, &qw.view, &qb.view, &kw.view, &kb.view, &inv_b.view, HEADS, ROT, 1e-5));
        const got = try download(a, qkv.view);
        defer a.free(got);
        // the reference on the same bf16 inputs
        const xr = try a.alloc(f32, rows * 3 * D);
        defer a.free(xr);
        for (xr, qkv_h) |*v, b| v.* = f32v(b);
        const offs_us = [_]usize{ 0, @min(rows, 5), rows };
        const qwf = try a.alloc(f32, DH);
        const qbf = try a.alloc(f32, DH);
        const kwf = try a.alloc(f32, DH);
        const kbf = try a.alloc(f32, DH);
        defer a.free(qwf);
        defer a.free(qbf);
        defer a.free(kwf);
        defer a.free(kbf);
        for (qwf, qw_h) |*v, b| v.* = f32v(b);
        for (qbf, qb_h) |*v, b| v.* = f32v(b);
        for (kwf, kw_h) |*v, b| v.* = f32v(b);
        for (kbf, kb_h) |*v, b| v.* = f32v(b);
        prepRef(xr, &offs_us, qwf, qbf, kwf, kbf, &inv);
        // bit for bit where the kernel's arithmetic is the reference's: the kernel sums in another order
        var worst: f64 = 0;
        for (got, xr) |g, e| worst = @max(worst, @abs(@as(f64, f32v(g)) - e));
        try testing.expect(worst < 3e-3);
        // a row alone: the same bits
        const one_h = qkv_h[0 .. 3 * D];
        var one = try upload(a, .bf16, &.{ 1, 3 * D }, std.mem.sliceAsBytes(one_h));
        defer one.deinit();
        const offs1 = [_]i32{ 0, 1 };
        var offs1_b = try upload(a, .i32, &.{offs1.len}, std.mem.sliceAsBytes(&offs1));
        defer offs1_b.deinit();
        try cuda.launched(zs_nemotron_perception_prep(&one.view, &offs1_b.view, &qw.view, &qb.view, &kw.view, &kb.view, &inv_b.view, HEADS, ROT, 1e-5));
        const single = try download(a, one.view);
        defer a.free(single);
        try testing.expectEqualSlices(u16, single, got[0 .. 3 * D]);
    }
}

/// Host reference of `zs_nemotron_perception_attention`: fp32 SDPA per segment, scale 1/sqrt(128).
fn attentionRef(qkv: []const f32, offs: []const usize, out: []f32) void {
    for (offs[0 .. offs.len - 1], offs[1..]) |start, end| {
        for (start..end) |r| {
            for (0..HEADS) |h| {
                var scores: [4096]f32 = undefined;
                var top: f32 = -std.math.inf(f32);
                for (start..end) |j| {
                    var s: f64 = 0;
                    for (0..DH) |t| s += @as(f64, qkv[r * 3 * D + h * DH + t]) * qkv[j * 3 * D + D + h * DH + t];
                    scores[j - start] = @as(f32, @floatCast(s / @sqrt(@as(f64, DH))));
                    top = @max(top, scores[j - start]);
                }
                var l: f64 = 0;
                for (start..end) |j| {
                    scores[j - start] = @exp(scores[j - start] - top);
                    l += scores[j - start];
                }
                for (0..DH) |t| {
                    var o: f64 = 0;
                    for (start..end) |j| o += @as(f64, scores[j - start]) * qkv[j * 3 * D + 2 * D + h * DH + t];
                    out[r * D + h * DH + t] = @floatCast(o / l);
                }
            }
        }
    }
}

fn writeTiles(a: std.mem.Allocator, offs: []const usize) !struct { host: [][4]i32, buffer: cuda.Buffer } {
    var list: std.ArrayList([4]i32) = .empty;
    for (offs[0 .. offs.len - 1], offs[1..]) |start, end| {
        var q = start;
        while (q < end) : (q += 128) try list.append(a, .{ @intCast(q), @intCast(start), @intCast(end - start), 0 });
    }
    const owned = try list.toOwnedSlice(a);
    return .{ .host = owned, .buffer = try upload(a, .i32, &.{ owned.len, 4 }, std.mem.sliceAsBytes(owned)) };
}

test "attention: a chunk's own tokens against the host, segments alone and together" {
    const a = testing.allocator;
    const offs_us = [_]usize{ 0, 5, 133, 200 };
    const rows = offs_us[offs_us.len - 1];
    const qkv_h = try randomBf(a, rows * 3 * D, 21);
    defer a.free(qkv_h);
    var qkv = try upload(a, .bf16, &.{ rows, 3 * D }, std.mem.sliceAsBytes(qkv_h));
    defer qkv.deinit();
    var out = try cuda.Buffer.init(.bf16, &.{ rows, D });
    defer out.deinit();
    const made = try writeTiles(a, &offs_us);
    defer a.free(made.host);
    var tiles = made.buffer;
    defer tiles.deinit();
    try cuda.launched(zs_nemotron_perception_attention(&qkv.view, &tiles.view, &out.view));
    const got = try download(a, out.view);
    defer a.free(got);
    const xr = try a.alloc(f32, rows * 3 * D);
    const want = try a.alloc(f32, rows * D);
    defer a.free(xr);
    defer a.free(want);
    for (xr, qkv_h) |*v, b| v.* = f32v(b);
    attentionRef(xr, &offs_us, want);
    // fp32 scores vs the kernel's bf16 mma: within 2e-2 of a value
    var worst: f64 = 0;
    var denom: f64 = 0;
    for (got, want) |g, e| {
        worst = @max(worst, @abs(@as(f64, f32v(g)) - e));
        denom = @max(denom, @abs(e));
    }
    try testing.expect(worst < 2e-2);
    // the middle segment alone: the same bits as among the batch
    const lo = offs_us[1];
    const hi = offs_us[2];
    const seg_h = qkv_h[lo * 3 * D .. hi * 3 * D];
    var seg = try upload(a, .bf16, &.{ hi - lo, 3 * D }, std.mem.sliceAsBytes(seg_h));
    defer seg.deinit();
    var seg_out = try cuda.Buffer.init(.bf16, &.{ hi - lo, D });
    defer seg_out.deinit();
    const seg_offs = [_]usize{ 0, hi - lo };
    const made2 = try writeTiles(a, &seg_offs);
    defer a.free(made2.host);
    var tiles2 = made2.buffer;
    defer tiles2.deinit();
    try cuda.launched(zs_nemotron_perception_attention(&seg.view, &tiles2.view, &seg_out.view));
    const alone = try download(a, seg_out.view);
    defer a.free(alone);
    try testing.expectEqualSlices(u16, alone, got[lo * D .. hi * D]);
}

/// Host reference of one encoder forward over stacked rows (fp32, bf16 roundings where the model has them):
/// pre_encode.proj, embed_norm, per layer x+attn(norm1 x), x+ffn(norm2 x), final_norm, perception.proj.
const Ref = struct {
    c: config_file.Config,
    checkpoint: *const safetensors.Checkpoint,

    fn vec(self: Ref, name: []const u8, a: std.mem.Allocator) ![]f32 {
        const t = self.checkpoint.get(name).?;
        const out = try a.alloc(f32, t.data.len / 2);
        for (out, std.mem.bytesAsSlice(u16, t.data)) |*v, b| v.* = f32v(b);
        return out;
    }

    /// An FP8 linear's [n, k] in f32 (e4m3 x its tensor scale; the fixture's scales are 1.0).
    fn fp8(self: Ref, name: []const u8, a: std.mem.Allocator) ![]f32 {
        var b: [160]u8 = undefined;
        const w = self.checkpoint.get(std.fmt.bufPrint(&b, "{s}.weight", .{name}) catch unreachable).?;
        const s = self.checkpoint.get(std.fmt.bufPrint(&b, "{s}.weight_scale", .{name}) catch unreachable).?;
        const scale = std.mem.bytesToValue(f32, s.data[0..4]);
        const out = try a.alloc(f32, w.data.len);
        for (out, w.data) |*v, code| v.* = fixture.e4m3Value(code) * scale;
        return out;
    }

    fn norm(x: []f32, at_row: usize, w: []const f32, b: []const f32, out: []f32) void {
        var sum: f64 = 0;
        var sq: f64 = 0;
        for (0..w.len) |t| {
            sum += x[at_row + t];
            sq += @as(f64, x[at_row + t]) * x[at_row + t];
        }
        const mean = sum / @as(f64, @floatFromInt(w.len));
        const rstd = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(w.len)) - mean * mean + 1e-5);
        for (0..w.len) |t| out[at_row + t] = bf(@as(f32, @floatCast((x[at_row + t] - mean) * rstd)) * w[t] + b[t]);
    }

    fn linear(w: []const f32, bias: ?[]const f32, x: []const f32, rows: usize, k: usize, n: usize, out: []f32) void {
        for (0..rows) |r| {
            for (0..n) |j| {
                var acc: f64 = if (bias) |b| b[j] else 0;
                for (0..k) |i| acc += @as(f64, x[r * k + i]) * w[j * k + i];
                out[r * n + j] = @floatCast(acc);
            }
        }
    }

    fn gelu(x: f32) f32 {
        // Abramowitz-Stegun 7.1.26 (1e-7 max error; the comparison's tolerance is far looser)
        const t = 1.0 / (1.0 + 0.2316419 * @abs(x));
        const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        const phi = 1.0 - @exp(-x * x / 2.0) * poly / @sqrt(2.0 * std.math.pi);
        const erf = if (x >= 0) phi else -phi;
        return x * 0.5 * (1 + erf);
    }

    /// `stacked` bf16-bits rows [rows, enc_stack] -> out f32 [rows, output_dim]. `keys` a segment's key
    /// rows where the PEE's padding mask cuts (empty: the whole segment, the non-PEE path).
    fn run(self: Ref, a: std.mem.Allocator, e: config_file.Perception, stacked: []const u16, offs: []const usize, keys: []const usize, rows: usize) ![]f32 {
        const d = e.d_model;
        const ff = e.ffWidth();
        const rot = @as(usize, @intFromFloat(e.rotary_fraction * @as(f64, @floatFromInt(e.headDim()))));
        var x = try a.alloc(f32, rows * d);
        const y = try a.alloc(f32, rows * d);
        const qkv = try a.alloc(f32, rows * 3 * d);
        const attn = try a.alloc(f32, rows * d);
        const ffn = try a.alloc(f32, rows * ff);
        const br = try a.alloc(f32, rows * d);
        const sin = try a.alloc(f32, rows * e.stacked());
        {
            const w = try self.vec("perception.encoder.asr_encoder.pre_encode.proj.weight", a);
            for (sin, 0..) |*v, i| v.* = f32v(stacked[i]);
            linear(w, null, sin, rows, e.stacked(), d, x);
        }
        {
            const w = try self.vec("perception.encoder.asr_encoder.embed_norm.weight", a);
            const b = try self.vec("perception.encoder.asr_encoder.embed_norm.bias", a);
            for (0..rows) |r| norm(x, r * d, w, b, x);
        }
        var name: [160]u8 = undefined;
        var leaf: [160]u8 = undefined;
        var inv: [256]f32 = undefined;
        for (0..rot / 2) |i| inv[i] = @floatCast(1.0 / std.math.pow(f64, e.rope_base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rot))));
        for (0..e.layers) |l| {
            const p = std.fmt.bufPrint(&name, "{s}{d}", .{ "perception.encoder.asr_encoder.layers.", l }) catch unreachable;
            const n1w = try self.vec(cat(&leaf, p, ".norm1.weight"), a);
            const n1b = try self.vec(cat(&leaf, p, ".norm1.bias"), a);
            const n2w = try self.vec(cat(&leaf, p, ".norm2.weight"), a);
            const n2b = try self.vec(cat(&leaf, p, ".norm2.bias"), a);
            for (0..rows) |r| norm(x, r * d, n1w, n1b, y);
            {
                const w = try self.fp8(cat(&leaf, p, ".attn.w_qkv"), a);
                linear(w, null, y, rows, d, 3 * d, qkv);
                const qw = try self.vec(cat(&leaf, p, ".attn.q_norm.weight"), a);
                const qb = try self.vec(cat(&leaf, p, ".attn.q_norm.bias"), a);
                const kw = try self.vec(cat(&leaf, p, ".attn.k_norm.weight"), a);
                const kb = try self.vec(cat(&leaf, p, ".attn.k_norm.bias"), a);
                // per-head q/k norms + rope, positions from each row's segment
                const dh = e.headDim();
                const heads = e.heads;
                for (0..rows) |r| {
                    var seg: usize = 0;
                    for (offs[1..], 1..) |end, i| {
                        if (r >= end) seg = i;
                    }
                    const pos = @as(f32, @floatFromInt(r - offs[seg]));
                    for (0..2) |part| {
                        const w_n = if (part == 0) qw else kw;
                        const b_n = if (part == 0) qb else kb;
                        for (0..heads) |h| {
                            const row = r * 3 * d + part * d + h * dh;
                            norm(qkv, row, w_n, b_n, qkv);
                            for (0..rot / 2) |i| {
                                const ang = pos * inv[i];
                                const c = bf(@cos(ang));
                                const s = bf(@sin(ang));
                                const n1 = qkv[row + i];
                                const n2 = qkv[row + i + rot / 2];
                                qkv[row + i] = bf(bf(n1 * c) + bf(-n2 * s));
                                qkv[row + i + rot / 2] = bf(bf(n2 * c) + bf(n1 * s));
                            }
                        }
                    }
                }
                // attention per segment, fp32
                for (offs[0 .. offs.len - 1], offs[1..], 0..) |start, end, si| {
                    const key_end = if (keys.len > 0) start + keys[si] else end;
                    for (start..end) |r| {
                        for (0..heads) |h| {
                            var scores: [8192]f32 = undefined;
                            var top: f32 = -std.math.inf(f32);
                            for (start..key_end) |j| {
                                var s: f64 = 0;
                                for (0..dh) |t| s += @as(f64, qkv[r * 3 * d + h * dh + t]) * qkv[j * 3 * d + d + h * dh + t];
                                scores[j - start] = @floatCast(s / @sqrt(@as(f64, @floatFromInt(dh))));
                                top = @max(top, scores[j - start]);
                            }
                            var ls: f64 = 0;
                            for (start..key_end) |j| {
                                scores[j - start] = @exp(scores[j - start] - top);
                                ls += scores[j - start];
                            }
                            for (0..dh) |t| {
                                var o: f64 = 0;
                                for (start..key_end) |j| o += @as(f64, scores[j - start]) * qkv[j * 3 * d + 2 * d + h * dh + t];
                                attn[r * d + h * dh + t] = @floatCast(o / ls);
                            }
                        }
                    }
                }
                const ow = try self.fp8(cat(&leaf, p, ".attn.out_proj"), a);
                const ob = try self.vec(cat(&leaf, p, ".attn.out_proj.bias"), a);
                linear(ow, ob, attn, rows, d, d, br);
                for (0..rows * d) |i| x[i] = bf(bf(x[i]) + bf(br[i]));
                for (0..rows) |r| norm(x, r * d, n2w, n2b, y);
                const f0w = try self.fp8(cat(&leaf, p, ".ffn.net.0"), a);
                const f0b = try self.vec(cat(&leaf, p, ".ffn.net.0.bias"), a);
                linear(f0w, f0b, y, rows, d, ff, ffn);
                for (ffn) |*v| v.* = gelu(bf(v.*));
                const f3w = try self.fp8(cat(&leaf, p, ".ffn.net.3"), a);
                const f3b = try self.vec(cat(&leaf, p, ".ffn.net.3.bias"), a);
                linear(f3w, f3b, ffn, rows, ff, d, br);
                for (0..rows * d) |i| x[i] = bf(bf(x[i]) + bf(br[i]));
            }
        }
        {
            const w = try self.vec("perception.encoder.asr_encoder.final_norm.weight", a);
            const b = try self.vec("perception.encoder.asr_encoder.final_norm.bias", a);
            for (0..rows) |r| norm(x, r * d, w, b, x);
            if (self.c.pe_encoder) {
                const aw = try self.vec("perception.encoder.asr_norm.weight", a);
                const ab = try self.vec("perception.encoder.asr_norm.bias", a);
                for (0..rows) |r| norm(x, r * d, aw, ab, x);
            }
            const w2 = try self.vec("perception.proj.weight", a);
            const b2 = try self.vec("perception.proj.bias", a);
            const out = try a.alloc(f32, rows * e.output_dim);
            linear(w2, b2, x, rows, d, e.output_dim, out);
            return out;
        }
    }
};

fn cat(buffer: []u8, prefix: []const u8, leaf: []const u8) []const u8 {
    return std.fmt.bufPrint(buffer, "{s}{s}", .{ prefix, leaf }) catch unreachable;
}

/// Stacked bf16 rows for `samples` through `mel.features` per chunk (the host path `encode` runs).
fn stackedRows(self: *Perception, arena: std.mem.Allocator, samples: []const f32, spans: []const [2]usize, rows: usize) ![]u16 {
    const p = self.params;
    const sub = self.subsampling;
    const rows_bf = try arena.alloc(u16, rows * self.stacked);
    @memset(rows_bf, 0);
    const frames = try arena.alloc(f32, self.rows_per_chunk * sub * p.features);
    var row0: usize = 0;
    for (spans) |span| {
        const t_len = p.frames(span[1] - span[0]);
        mel_file.features(p, &self.mel_scratch, self.fb, self.window, samples[span[0]..span[1]], frames[0 .. t_len * p.features]);
        for (0..t_len) |f| {
            const r = row0 + f / sub;
            for (0..p.features) |m| rows_bf[r * self.stacked + (f % sub) * p.features + m] = perception_file.bf16Of(frames[f * p.features + m]);
        }
        row0 += (t_len + sub - 1) / sub;
    }
    return rows_bf;
}

fn loadFixture(gpa: std.mem.Allocator, arena: std.mem.Allocator, max_chunks: usize, pe: bool) !struct { c: config_file.Config, checkpoint: safetensors.Checkpoint, p: Perception } {
    var diagnostic: config_file.Diagnostic = .{};
    const text = if (pe) blk: {
        // the checkpoint mounts a ParallelExpertEncoder: the same swap prompt_test's realConfig makes
        const swapped = try std.mem.replaceOwned(u8, arena, fixture.config_text, "\"encoder_chunk_size_seconds\":30.0,", "\"encoder_chunk_size_seconds\":30.0,\"pe_encoder_config\":{\"asr_encoder_type\":\"transformer\"},");
        break :blk swapped;
    } else fixture.config_text;
    const c = try config_file.Config.parse(arena, text, fixture.backbone_text, fixture.quant_text, &diagnostic);
    var checkpoint = try fixture.synthetic(arena, c, 3);
    fixture.fill(&checkpoint, 7);
    const p = try Perception.load(gpa, arena, &c, &checkpoint, max_chunks);
    return .{ .c = c, .checkpoint = checkpoint, .p = p };
}

fn cosine(got: []const u16, want: []const f32) f64 {
    var dot: f64 = 0;
    var ga: f64 = 0;
    var wa: f64 = 0;
    for (got, want) |g, w| {
        const gf = @as(f64, f32v(g));
        dot += gf * w;
        ga += gf * gf;
        wa += @as(f64, w) * w;
    }
    return dot / @sqrt(ga * wa + 1e-30);
}

/// The stacked rows a PEE window's segment holds: `cs..ce` frames of `mel_rows`, plus its segment offsets,
/// key length (`ceil(min(ce-cs, valid-cs)/sub)`) and kept (out, from) pairs — the same bookkeeping
/// `encodePe` writes.
fn peStacked(self: *Perception, arena: std.mem.Allocator, mel_rows: []const f32, windows: []const [5]usize, valid: usize, batch_rows: usize) !struct { rows: []u16, offs: []usize, keys: []usize, kept: [][2]usize } {
    const p = self.params;
    const sub = self.subsampling;
    const rows_bf = try arena.alloc(u16, batch_rows * self.stacked);
    @memset(rows_bf, 0);
    const offs = try arena.alloc(usize, windows.len + 1);
    const keys = try arena.alloc(usize, windows.len);
    const kept_list = try arena.alloc([2]usize, batch_rows);
    var row0: usize = 0;
    var kept_count: usize = 0;
    offs[0] = 0;
    for (windows, 0..) |w, i| {
        const cs, const ce, const core_rows, const left_drop, const out_row = w;
        const seg_rows = (ce - cs + sub - 1) / sub;
        for (cs..ce) |f| {
            const r = row0 + (f - cs) / sub;
            for (0..p.features) |m| rows_bf[r * self.stacked + (f - cs) % sub * p.features + m] = perception_file.bf16Of(mel_rows[f * p.features + m]);
        }
        keys[i] = @min(seg_rows, (@min(ce - cs, if (cs < valid) valid - cs else 0) + sub - 1) / sub);
        for (0..core_rows) |k| {
            kept_list[kept_count] = .{ out_row + k, row0 + left_drop + k };
            kept_count += 1;
        }
        offs[i + 1] = row0 + seg_rows;
        row0 += seg_rows;
    }
    return .{ .rows = rows_bf[0 .. row0 * self.stacked], .offs = offs, .keys = keys, .kept = kept_list[0..kept_count] };
}

test "encoder: the whole stack on synthetic weights against the host reference" {
    const a = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    var made = try loadFixture(a, ar, 4, false);
    defer made.p.deinit();
    const seconds = 30.0;
    // 42 s of synthetic audio: one full chunk + a 12 s tail, as the fixture's 30 s chunks split it
    const samples = try waveformLen(ar, 42 * 16000, 5);
    const spans = try mel_file.chunkSpans(ar, made.p.params, seconds, samples.len);
    const total = try mel_file.rows(ar, made.p.params, seconds, made.p.subsampling, samples.len);
    var out = try cuda.Buffer.init(.bf16, &.{ total, made.p.output_dim });
    defer out.deinit();
    try made.p.encode(ar, samples, out.view);
    const got = try download(a, out.view);
    defer a.free(got);
    // the host reference on the same stacked rows
    const offs = try ar.alloc(usize, spans.len + 1);
    var row: usize = 0;
    offs[0] = 0;
    for (spans, 1..) |span, i| {
        row += (made.p.params.frames(span[1] - span[0]) + made.p.subsampling - 1) / made.p.subsampling;
        offs[i] = row;
    }
    const stacked = try stackedRows(&made.p, ar, samples, spans, row);
    const want = try (Ref{ .c = made.c, .checkpoint = &made.checkpoint }).run(ar, made.c.perception, stacked, offs, &.{}, row);
    const cos = cosine(got, want);
    std.debug.print("encoder end-to-end cosine {d:.6} over {d} rows\n", .{ cos, row });
    try testing.expect(cos > 0.995);
    // a chunk encoded alone == the same chunk in the batch, bit for bit
    var out1 = try cuda.Buffer.init(.bf16, &.{ made.p.rows_per_chunk, made.p.output_dim });
    defer out1.deinit();
    try made.p.encode(ar, samples[spans[0][0]..spans[0][1]], out1.view);
    const alone = try download(a, out1.view);
    defer a.free(alone);
    const first_rows = offs[1] - offs[0];
    try testing.expectEqualSlices(u16, alone[0 .. first_rows * made.p.output_dim], got[0 .. first_rows * made.p.output_dim]);
}

test "encoder: the PEE's whole-clip mel, per-feature norm, context windows and kept cores" {
    const a = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    var made = try loadFixture(a, ar, 2, true);
    defer made.p.deinit();
    try testing.expect(made.p.pe);
    const p_ = made.p.params;
    const sub = made.p.subsampling;
    // 75 s: several windows of pe_core*sub = 1,000 mel frames a core (the fixture's pe defaults), 4,201
    // valid frames + the STFT's partial = 4,202 tiled
    const n = 75 * 16000;
    const samples = try waveformLen(ar, n, 5);
    const valid = p_.frames(n);
    const width = p_.stftFrames(n);
    const encoded_len = @max(1, (valid + sub - 1) / sub);
    try testing.expectEqual(encoded_len, try made.p.rows(ar, n));
    var out = try cuda.Buffer.init(.bf16, &.{ encoded_len, made.p.output_dim });
    defer out.deinit();
    try made.p.encode(ar, samples, out.view);
    const got = try download(a, out.view);
    defer a.free(got);
    // the host path: the clip's mel once, per_feature over `valid`, the same windows' stacked rows
    const mel_rows = try ar.alloc(f32, width * p_.features);
    @memset(mel_rows, 0);
    var scratch = made.p.mel_scratch;
    scratch.signal = try ar.alloc(f32, n);
    mel_file.features(p_, &scratch, made.p.fb, made.p.window, samples, mel_rows[0 .. valid * p_.features]);
    mel_file.normalizePerFeature(p_, mel_rows, valid);
    const num_windows = @max(1, (width + made.p.pe_core - 1) / made.p.pe_core);
    const windows = try ar.alloc([5]usize, num_windows);
    var batch_rows: usize = 0;
    var out_row: usize = 0;
    for (windows, 0..) |*w, i| {
        const start = i * made.p.pe_core;
        const end = @min(start + made.p.pe_core, width);
        const cs = if (start >= made.p.pe_left) start - made.p.pe_left else 0;
        const ce = @min(end + made.p.pe_right, width);
        const core_rows = @min(encoded_len, (end + sub - 1) / sub) - (start + sub - 1) / sub;
        w.* = .{ cs, ce, core_rows, (start - cs) / sub, out_row };
        out_row += core_rows;
        batch_rows += (ce - cs + sub - 1) / sub;
    }
    const built = try peStacked(&made.p, ar, mel_rows, windows, valid, batch_rows);
    const projected = try (Ref{ .c = made.c, .checkpoint = &made.checkpoint }).run(ar, made.c.perception, built.rows, built.offs, built.keys, built.rows.len / made.p.stacked);
    const want = try ar.alloc(f32, out_row * made.p.output_dim);
    for (built.kept) |pair| {
        for (0..made.p.output_dim) |m|
            want[pair[0] * made.p.output_dim + m] = projected[pair[1] * made.p.output_dim + m];
    }
    const cos = cosine(got, want);
    std.debug.print("PEE end-to-end cosine {d:.6} over {d} kept of {d} batch rows, {d} windows\n", .{ cos, out_row, batch_rows, num_windows });
    try testing.expect(cos > 0.995);
    // the first window's kept rows are position-independent: a short clip that is one window alone gives
    // the same bits — it isn't, the per-feature norm is over the whole clip; instead check rows are the
    // estimator's count and deterministic: encode twice, the same bits
    var out2 = try cuda.Buffer.init(.bf16, &.{ encoded_len, made.p.output_dim });
    defer out2.deinit();
    try made.p.encode(ar, samples, out2.view);
    const again = try download(a, out2.view);
    defer a.free(again);
    try testing.expectEqualSlices(u16, got, again);
}

fn waveformLen(a: std.mem.Allocator, n: usize, seed: u64) ![]f32 {
    const x = try a.alloc(f32, n);
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    for (x, 0..) |*v, i| {
        const t = @as(f64, @floatFromInt(i)) / 16000;
        v.* = @floatCast(0.15 * @sin(2 * std.math.pi * 220 * t) + 0.1 * @sin(2 * std.math.pi * 997 * t) + 0.03 * (random.float(f64) - 0.5));
    }
    return x;
}
