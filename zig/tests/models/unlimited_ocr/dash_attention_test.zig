//! Unlimited-OCR-2's DASH attend (`kernels/unlimited_ocr/dash_attention.cu`, spec 147 WS-04): on the fixtures of
//! the checkpoint's own HF code with Triton off (`$ZSPARK_REFERENCE_UNLIMITED_OCR_2`, `export_v2.py`), every
//! layer's prompt against `prefill.attend` (HF's SDPA in bf16 under a bf16 mask) and the detailed steps against
//! `detail.attend` (`_decode_attend`, fp32), each also against an fp64 sum on the host over the same inputs; and
//! the row rule: a row's bits the same through the tile kernel and the step kernel, alone, in a prompt's chunks
//! and in a verify window. On random rows the same, with the window's first block chosen, slots absent and
//! rows before any remote block, every value finite.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const dash = @import("../../../src/models/unlimited_ocr/dash_attention.zig");

const D = 128;
const H = 10;
const SEL = 12;
const W = 128;
const layers = 12;
const scale = 1.0 / @sqrt(@as(f64, D));

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

fn of(bits: u16) f64 {
    return numeric.f32FromBf16(bits);
}

fn upload(dtype: cuda.DType, shape: []const usize, bytes: []const u8) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(dtype, shape);
    errdefer buffer.deinit();
    try buffer.upload(bytes);
    return buffer;
}

/// The caches of one layer on the device, with the host's bits for the fp64 sums.
const Caches = struct {
    k: []const u16,
    v: []const u16,
    heads: usize,
    kc: cuda.Buffer,
    vc: cuda.Buffer,

    fn init(k: []const u16, v: []const u16, cap: usize, heads: usize) !Caches {
        var kc = try upload(.bf16, &.{ cap, heads, D }, std.mem.sliceAsBytes(k));
        errdefer kc.deinit();
        const vc = try upload(.bf16, &.{ cap, heads, D }, std.mem.sliceAsBytes(v));
        return .{ .k = k, .v = v, .heads = heads, .kc = kc, .vc = vc };
    }

    fn deinit(self: *Caches) void {
        self.kc.deinit();
        self.vc.deinit();
    }
};

/// out bf16 [rows, heads D] of `form` for q [rows, heads, D] at positions pos0 .., ids and bias [rows, SEL].
fn run(gpa: std.mem.Allocator, form: dash.Form, c: Caches, q: []const u16, rows: usize, pos0: usize, ids: []const i32, bias: []const f32) ![]u16 {
    const heads = c.heads;
    var qb = try upload(.bf16, &.{ rows, heads, D }, std.mem.sliceAsBytes(q[0 .. rows * heads * D]));
    defer qb.deinit();
    var ib = try upload(.i32, &.{ rows, SEL }, std.mem.sliceAsBytes(ids[0 .. rows * SEL]));
    defer ib.deinit();
    var bb = try upload(.f32, &.{ rows, SEL }, std.mem.sliceAsBytes(bias[0 .. rows * SEL]));
    defer bb.deinit();
    const word = [1]i32{@intCast(pos0)};
    var at = try upload(.i32, &.{1}, std.mem.sliceAsBytes(&word));
    defer at.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ rows, heads * D });
    defer out.deinit();
    const slots = dash.slots(SEL, W);
    var parts: cuda.Buffer = try .init(.f32, &.{ rows * heads * slots, D + 2 });
    defer parts.deinit();
    var done: cuda.Buffer = try .init(.i32, &.{rows * heads});
    defer done.deinit();
    try done.zero();
    try dash.attend(form, qb.view, c.kc.view, c.vc.view, at.view, ib.view, bb.view, W, scale, .{ .parts = parts.view, .done = done.view }, out.view);
    const got = try gpa.alloc(u16, rows * heads * D);
    errdefer gpa.free(got);
    try cuda.synchronize();
    try out.download(std.mem.sliceAsBytes(got));
    if (form == .step) {
        // the counters are left zero for the next call
        const left = try gpa.alloc(i32, rows * heads);
        defer gpa.free(left);
        try done.download(std.mem.sliceAsBytes(left));
        for (left) |x| try testing.expectEqual(@as(i32, 0), x);
    }
    return got;
}

/// One (row, head)'s attend in fp64 over the keys HF's `_decode_attend` keeps: values in `want`, the weighted
/// magnitudes in `mag`.
fn reference(c: Caches, q: []const u16, h: usize, pos: usize, ids: []const i32, bias: []const f32, want: *[D]f64, mag: *[D]f64) void {
    const heads = c.heads;
    var keys: [SEL * 32 + W]usize = undefined;
    var adds: [SEL * 32 + W]f64 = undefined;
    var n: usize = 0;
    const low = if (pos + 1 > W) pos + 1 - W else 0;
    for (ids, bias) |id, b| {
        if (id < 0) continue;
        const k0: usize = @as(usize, @intCast(id)) * 32;
        for (k0..k0 + 32) |j| if (j < low) {
            keys[n] = j;
            adds[n] = b;
            n += 1;
        };
    }
    for (low..pos + 1) |j| {
        keys[n] = j;
        adds[n] = 0;
        n += 1;
    }
    var scores: [SEL * 32 + W]f64 = undefined;
    var most: f64 = -std.math.inf(f64);
    for (0..n) |i| {
        var dot: f64 = 0;
        for (0..D) |d| dot += of(q[h * D + d]) * of(c.k[(keys[i] * heads + h) * D + d]);
        scores[i] = dot * scale + adds[i];
        most = @max(most, scores[i]);
    }
    var total: f64 = 0;
    for (scores[0..n]) |*s| {
        s.* = @exp(s.* - most);
        total += s.*;
    }
    want.* = @splat(0);
    mag.* = @splat(0);
    for (0..n) |i| for (0..D) |d| {
        const v = of(c.v[(keys[i] * heads + h) * D + d]);
        want[d] += scores[i] / total * v;
        mag[d] += scores[i] / total * @abs(v);
    };
}

const Stats = struct {
    max: f64 = 0,
    sum: f64 = 0,
    n: usize = 0,
    sq: f64 = 0,
    ref_sq: f64 = 0,

    fn add(self: *Stats, got: f64, want: f64) void {
        const e = @abs(got - want);
        self.max = @max(self.max, e);
        self.sum += e;
        self.sq += e * e;
        self.ref_sq += want * want;
        self.n += 1;
    }

    fn print(self: Stats, what: []const u8) void {
        std.debug.print("dash {s}: max abs {e:.3}, mean abs {e:.3}, rel rms {e:.3} ({d} values)\n", .{ what, self.max, self.sum / @as(f64, @floatFromInt(@max(self.n, 1))), @sqrt(self.sq / @max(self.ref_sq, 1e-300)), self.n });
    }
};

/// The kernel against fp64 over its own inputs: within its roundings (bf16 weights in P.V, a bf16 output).
fn againstHost(c: Caches, q: []const u16, got: []const u16, h: usize, pos: usize, ids: []const i32, bias: []const f32, stats: *Stats) !void {
    var want: [D]f64 = undefined;
    var mag: [D]f64 = undefined;
    reference(c, q, h, pos, ids, bias, &want, &mag);
    for (0..D) |d| {
        const g = of(got[h * D + d]);
        stats.add(g, want[d]);
        if (!(@abs(g - want[d]) <= mag[d] / 100.0 + @abs(want[d]) / 128.0 + 1e-6)) {
            std.debug.print("pos {d} head {d} d {d}: {d} against {d}\n", .{ pos, h, d, g, want[d] });
            return error.TestUnexpectedResult;
        }
    }
}

fn slice(comptime T: type, t: safetensors.Tensor) []const T {
    return @alignCast(std.mem.bytesAsSlice(T, t.data));
}

/// The router's slots with the invalid ones absent (an invalid slot's id is tie-break noise, export_v2.py).
fn absentInvalid(gpa: std.mem.Allocator, ids: []const i32, valid: []const u8) ![]i32 {
    const out = try gpa.alloc(i32, ids.len);
    for (out, ids, valid) |*o, id, ok| o.* = if (ok != 0) id else -1;
    return out;
}

/// Bounds held on HF's outputs, a relative rms over a case's values (measured 1.1e-3 on the prompts, 1.7e-3 on
/// the steps, spec 147 WS-04): the prompt's SDPA rounds its mask's bias and its weights to bf16, the steps'
/// attend is fp32, so the steps' error is the kernel's own bf16 weights in P.V and its bf16 output.
const prompt_rel_rms = 4e-3;
const step_rel_rms = 4e-3;

test "DASH attend: the fixtures' prompt and steps against HF and fp64, a row's bits the same in either kernel" {
    const gpa = testing.allocator;
    const dir = env("ZSPARK_REFERENCE_UNLIMITED_OCR_2") orelse return error.SkipZigTest;
    for ([_][]const u8{ "two_pages", "one_page" }) |name| {
        var buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/{s}.safetensors", .{ dir, name });
        var file: safetensors.File = try .open(gpa, testing.io, .cwd(), path);
        defer file.close();
        const pq = file.get("prefill.q").?;
        const S: usize = @intCast(pq.shape[1]);
        const cache_k = file.get("cache.k").?;
        const cap: usize = @intCast(cache_k.shape[1]);
        const all_q = slice(u16, pq);
        const all_attend = slice(u16, file.get("prefill.attend").?);
        const all_k = slice(u16, cache_k);
        const all_v = slice(u16, file.get("cache.v").?);
        const p_ids = try absentInvalid(gpa, slice(i32, file.get("prefill.block_ids").?), file.get("prefill.valid").?.data);
        defer gpa.free(p_ids);
        const p_bias = slice(f32, file.get("prefill.bias").?);
        const s_ids = try absentInvalid(gpa, slice(i32, file.get("step.block_ids").?), file.get("step.valid").?.data);
        defer gpa.free(s_ids);
        const s_bias = slice(f32, file.get("step.bias").?);
        const calls = slice(i32, file.get("detail.calls").?);
        const d_q = slice(u16, file.get("detail.q").?);
        const d_attend = slice(f32, file.get("detail.attend").?);
        var prompt_hf: Stats = .{};
        var prompt_host: Stats = .{};
        var step_hf: Stats = .{};
        var step_host: Stats = .{};
        for (0..layers) |l| {
            const lane = cap * H * D;
            var c = try Caches.init(all_k[l * lane ..][0..lane], all_v[l * lane ..][0..lane], cap, H);
            defer c.deinit();
            const q = all_q[l * S * H * D ..][0 .. S * H * D];
            const ids = p_ids[l * S * SEL ..][0 .. S * SEL];
            const bias = p_bias[l * S * SEL ..][0 .. S * SEL];
            const tile = try run(gpa, .tile, c, q, S, 0, ids, bias);
            defer gpa.free(tile);
            // the row rule: the step kernel on the prompt's rows, the prompt in two chunks, rows alone
            const step = try run(gpa, .step, c, q, S, 0, ids, bias);
            defer gpa.free(step);
            try testing.expectEqualSlices(u16, tile, step);
            const cut = 200;
            const tail = try run(gpa, .tile, c, q[cut * H * D ..], S - cut, cut, ids[cut * SEL ..], bias[cut * SEL ..]);
            defer gpa.free(tail);
            try testing.expectEqualSlices(u16, tile[cut * H * D ..], tail);
            for ([_]usize{ 0, 131, cut, S - 1 }) |r| {
                const alone = try run(gpa, .tile, c, q[r * H * D ..], 1, r, ids[r * SEL ..], bias[r * SEL ..]);
                defer gpa.free(alone);
                try testing.expectEqualSlices(u16, tile[r * H * D ..][0 .. H * D], alone);
            }
            const ref = all_attend[l * S * H * D ..][0 .. S * H * D];
            for (tile, ref) |g, w| prompt_hf.add(of(g), of(w));
            if (l == 0 or l == layers - 1) for (0..S) |r| for (0..H) |h| {
                try againstHost(c, q[r * H * D ..][0 .. H * D], tile[r * H * D ..][0 .. H * D], h, r, ids[r * SEL ..][0..SEL], bias[r * SEL ..][0..SEL], &prompt_host);
            };
            // the detailed steps: call i feeds the token at S + i - 1, step row i - 1
            for (calls, 0..) |call, d| {
                const row: usize = @intCast(call - 1);
                const pos = S + row;
                const sq = d_q[(d * layers + l) * H * D ..][0 .. H * D];
                const si = s_ids[(row * layers + l) * SEL ..][0..SEL];
                const sb = s_bias[(row * layers + l) * SEL ..][0..SEL];
                const one = try run(gpa, .step, c, sq, 1, pos, si, sb);
                defer gpa.free(one);
                const by_tile = try run(gpa, .tile, c, sq, 1, pos, si, sb);
                defer gpa.free(by_tile);
                try testing.expectEqualSlices(u16, one, by_tile);
                // a verify window of three rows ending at the step, its rows before it another step's query
                if (row >= 2) {
                    var wq: [3 * H * D]u16 = undefined;
                    var wi: [3 * SEL]i32 = undefined;
                    var wb: [3 * SEL]f32 = undefined;
                    for (0..3) |k| {
                        const other = d_q[(((d + k + 1) % calls.len) * layers + l) * H * D ..][0 .. H * D];
                        @memcpy(wq[k * H * D ..][0 .. H * D], if (k == 2) sq else other);
                        @memcpy(wi[k * SEL ..][0..SEL], s_ids[((row - 2 + k) * layers + l) * SEL ..][0..SEL]);
                        @memcpy(wb[k * SEL ..][0..SEL], s_bias[((row - 2 + k) * layers + l) * SEL ..][0..SEL]);
                    }
                    for ([_]dash.Form{ .step, .tile }) |form| {
                        const window = try run(gpa, form, c, &wq, 3, pos - 2, &wi, &wb);
                        defer gpa.free(window);
                        try testing.expectEqualSlices(u16, one, window[2 * H * D ..]);
                    }
                }
                const want = d_attend[(d * layers + l) * H * D ..][0 .. H * D];
                for (one, want) |g, w| step_hf.add(of(g), w);
                for (0..H) |h| try againstHost(c, sq, one, h, pos, si, sb, &step_host);
            }
        }
        std.debug.print("dash {s}: {d} prompt rows, {d} detailed steps, 12 layers\n", .{ name, S, calls.len });
        prompt_hf.print("prompt against HF's bf16 SDPA");
        prompt_host.print("prompt against fp64 (layers 0 and 11)");
        step_hf.print("steps against HF's fp32 _decode_attend");
        step_host.print("steps against fp64");
        try testing.expect(@sqrt(prompt_hf.sq / prompt_hf.ref_sq) < prompt_rel_rms);
        try testing.expect(@sqrt(step_hf.sq / step_hf.ref_sq) < step_rel_rms);
    }
}

test "DASH attend on random rows: fp64, finite, the window's first block chosen, a row's bits the same everywhere" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1474);
    const random = prng.random();
    const heads = 2;
    const cap = 700;
    const k = try gpa.alloc(u16, cap * heads * D);
    defer gpa.free(k);
    const v = try gpa.alloc(u16, cap * heads * D);
    defer gpa.free(v);
    const q = try gpa.alloc(u16, cap * heads * D);
    defer gpa.free(q);
    for (k) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32));
    for (v) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32));
    for (q) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32) * 2);
    // a row's choice: up to 12 distinct blocks at most pos / 32 - 4, some slots absent, the window's first block
    // (pos / 32 - 4) taken by every third row; a bias of either sign, some large
    const ids = try gpa.alloc(i32, cap * SEL);
    defer gpa.free(ids);
    const bias = try gpa.alloc(f32, cap * SEL);
    defer gpa.free(bias);
    for (0..cap) |pos| {
        const row_ids = ids[pos * SEL ..][0..SEL];
        const row_bias = bias[pos * SEL ..][0..SEL];
        @memset(row_ids, -1);
        @memset(row_bias, 0);
        const qblk = pos / 32;
        if (qblk < 4) continue;
        const remote = qblk - 4 + 1;
        var n: usize = 0;
        if (pos % 3 == 0) {
            row_ids[0] = @intCast(qblk - 4);
            n = 1;
        }
        const want = @min(remote, random.uintLessThan(usize, SEL + 1));
        var tries: usize = 0;
        while (n < want and tries < 200) : (tries += 1) {
            const b: i32 = @intCast(random.uintLessThan(usize, remote));
            if (std.mem.indexOfScalar(i32, row_ids[0..n], b) == null) {
                row_ids[n] = b;
                n += 1;
            }
        }
        random.shuffle(i32, row_ids);
        for (row_bias, row_ids) |*b, id| b.* = if (id >= 0) random.floatNorm(f32) * @as(f32, if (pos % 7 == 0) 30 else 1) else 0;
    }
    var c = try Caches.init(k, v, cap, heads);
    defer c.deinit();
    const tile = try run(gpa, .tile, c, q, cap, 0, ids, bias);
    defer gpa.free(tile);
    const step = try run(gpa, .step, c, q, cap, 0, ids, bias);
    defer gpa.free(step);
    try testing.expectEqualSlices(u16, tile, step);
    var host: Stats = .{};
    for (0..cap) |r| for (0..heads) |h| {
        for (tile[r * heads * D ..][h * D ..][0..D]) |bits| try testing.expect(std.math.isFinite(of(bits)));
        try againstHost(c, q[r * heads * D ..][0 .. heads * D], tile[r * heads * D ..][0 .. heads * D], h, r, ids[r * SEL ..][0..SEL], bias[r * SEL ..][0..SEL], &host);
    };
    host.print("random rows against fp64");
    // chunks of odd sizes on either kernel, and windows of a few rows
    var first: usize = 0;
    for ([_]usize{ 37, 1, 90, 64, 129, 5, 3 }, 0..) |n, i| {
        const form: dash.Form = if (i % 2 == 0) .tile else .step;
        const part = try run(gpa, form, c, q[first * heads * D ..], n, first, ids[first * SEL ..], bias[first * SEL ..]);
        defer gpa.free(part);
        try testing.expectEqualSlices(u16, tile[first * heads * D ..][0 .. n * heads * D], part);
        first += n;
    }
    const last = try run(gpa, .tile, c, q[first * heads * D ..], cap - first, first, ids[first * SEL ..], bias[first * SEL ..]);
    defer gpa.free(last);
    try testing.expectEqualSlices(u16, tile[first * heads * D ..], last);
}
