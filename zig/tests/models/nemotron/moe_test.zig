//! `moe.zig`'s tests: synthetic NVFP4 experts (the manifest's shapes shrunk) against an fp64 host — the
//! routed picks and weights of `glue_ref.zig`'s top-k, each slot's relu-squared expert rows, the shared
//! halves and the weighted sum — and a row's bits the same in a serial step, a verify window and prompt
//! chunks of 1 to 2,049 rows, through `combine`'s normed output too. Routing ties pick the lower id.

const std = @import("std");
const cuda = @import("../../../src/cuda/cuda.zig");
const gpu = @import("../../cuda/testing.zig");
const experts_file = @import("../../../src/weights/experts.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const nvfp4 = @import("../../../src/weights/nvfp4.zig");
const moe = @import("../../../src/models/nemotron/moe.zig");
const ref = @import("../../../src/models/nemotron/glue_ref.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;
const fetch = gpu.fetch;
const tobf = ref.f32ToBf16;
const bf = ref.bf16ToF32;

const fx = struct {
    // a small Transcribe-shaped layer: 8 routed experts of 64 over a hidden of 128, a shared of 128
    const shape: moe.Shape = .{ .dims = 128, .routed = 8, .width = 64, .top = 3, .scale = 2.5, .norm = true, .eps = 1e-5 };
    const count = shape.count();
    const slots = shape.slots();
};

/// An NVFP4 matrix as the checkpoint stores it, per expert, and its fp64 dequantization.
const TestMatrix = struct {
    codes: []u8,
    block_scales: []u8,
    scales: []f32,
    /// [experts, n, k] as the host dequantizes it
    dense: []f64,

    fn init(gpa: Allocator, random: std.Random, experts: usize, n: usize, k: usize) !TestMatrix {
        const codes = try gpa.alloc(u8, experts * n * k / 2);
        errdefer gpa.free(codes);
        random.bytes(codes);
        const block_scales = try gpa.alloc(u8, experts * n * k / 16);
        errdefer gpa.free(block_scales);
        for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
        const scales = try gpa.alloc(f32, experts);
        errdefer gpa.free(scales);
        for (scales) |*scale| scale.* = 0.01 + 0.02 * random.float(f32);
        const dense = try gpa.alloc(f64, experts * n * k);
        for (dense, 0..) |*value, i| {
            const byte = codes[i / 2];
            const code = if (i % 2 == 0) byte & 0xf else byte >> 4;
            value.* = @as(f64, nvfp4.e2m1[code]) * numeric.f32FromE4m3(block_scales[i / 16]) * scales[i / (n * k)];
        }
        return .{ .codes = codes, .block_scales = block_scales, .scales = scales, .dense = dense };
    }

    fn deinit(self: TestMatrix, gpa: Allocator) void {
        gpa.free(self.codes);
        gpa.free(self.block_scales);
        gpa.free(self.scales);
        gpa.free(self.dense);
    }

    fn matrix(self: TestMatrix, arena: Allocator) !experts_file.Matrix {
        const count = self.scales.len;
        return .{ .codes = try experts_file.Matrix.parts(arena, self.codes, count), .block_scales = try experts_file.Matrix.parts(arena, self.block_scales, count), .scales = self.scales };
    }
};

/// One layer's synthetic weights: the router rows, the bias, the routed experts, the shared one — and the
/// fold the device makes of them: `count` dense expert matrices, the shared's halves last.
const Weights = struct {
    router: []u16,
    bias: []f32,
    up: TestMatrix,
    down: TestMatrix,
    shared_up: TestMatrix,
    shared_down: TestMatrix,
    norm: []u16,
    /// [count, width * dims]: routed up dense, then the shared up's row halves.
    up_all: []f64,
    /// [count, dims * width]: routed down dense, then the shared down's column halves.
    down_all: []f64,

    fn init(gpa: Allocator, random: std.Random) !Weights {
        const s = fx.shape;
        const router = try gpa.alloc(u16, s.routed * s.dims);
        for (router) |*v| v.* = tobf(random.floatNorm(f32) * 0.5);
        const bias = try gpa.alloc(f32, s.routed);
        for (bias) |*v| v.* = random.floatNorm(f32) * 0.3;
        const norm = try gpa.alloc(u16, s.dims);
        for (norm) |*v| v.* = tobf(1.0 + random.floatNorm(f32) * 0.2);
        var w: Weights = .{
            .router = router,
            .bias = bias,
            .up = try .init(gpa, random, s.routed, s.width, s.dims),
            .down = try .init(gpa, random, s.routed, s.dims, s.width),
            .shared_up = try .init(gpa, random, 1, 2 * s.width, s.dims),
            .shared_down = try .init(gpa, random, 1, s.dims, 2 * s.width),
            .norm = norm,
            .up_all = try gpa.alloc(f64, s.count() * s.width * s.dims),
            .down_all = try gpa.alloc(f64, s.count() * s.dims * s.width),
        };
        @memcpy(w.up_all[0 .. s.routed * s.width * s.dims], w.up.dense);
        @memcpy(w.down_all[0 .. s.routed * s.dims * s.width], w.down.dense);
        @memcpy(w.up_all[s.routed * s.width * s.dims ..], w.shared_up.dense);
        // shared down [dims, 2 width] -> each pseudo-expert's [dims, width] column half
        for (0..2) |half| for (0..s.dims) |r| {
            const at = (s.routed + half) * s.dims * s.width + r * s.width;
            @memcpy(w.down_all[at..][0..s.width], w.shared_down.dense[r * 2 * s.width + half * s.width ..][0..s.width]);
        };
        return w;
    }

    fn deinit(w: Weights, gpa: Allocator) void {
        gpa.free(w.router);
        gpa.free(w.bias);
        gpa.free(w.norm);
        gpa.free(w.up_all);
        gpa.free(w.down_all);
        w.up.deinit(gpa);
        w.down.deinit(gpa);
        w.shared_up.deinit(gpa);
        w.shared_down.deinit(gpa);
    }

    fn host(w: Weights, arena: Allocator) !moe.Host {
        return .{
            .router = w.router,
            .bias = w.bias,
            .up = try w.up.matrix(arena),
            .down = try w.down.matrix(arena),
            .shared_up = try w.shared_up.matrix(arena),
            .shared_down = try w.shared_down.matrix(arena),
        };
    }
};

const Dot = struct { value: f64, magnitude: f64 };

fn dot(weights: []const f64, x_bits: []const u16) Dot {
    var result: Dot = .{ .value = 0, .magnitude = 0 };
    for (weights, x_bits) |weight, bits| {
        const term = weight * bf(bits);
        result.value += term;
        result.magnitude += @abs(term);
    }
    return result;
}

/// Expert `e`'s relu² rows for input `x_bits` (e in the folded [0, count) space): up's dot rounded to bf16,
/// relu, squared, rounded again — the kernel epilogue's two roundings — then down's dot over the bf16 rows.
/// Writes the slot's output row; returns the largest down dot's magnitude for the tolerance.
fn slotRow(w: Weights, e: usize, x_bits: []const u16, act: []u16, out: []f64) f64 {
    const s = fx.shape;
    const up_dense = w.up_all[e * s.width * s.dims ..][0 .. s.width * s.dims];
    const down_dense = w.down_all[e * s.dims * s.width ..][0 .. s.dims * s.width];
    for (0..s.width) |col| {
        const u = dot(up_dense[col * s.dims ..][0..s.dims], x_bits);
        const r = @as(f64, bf(tobf(@floatCast(@max(u.value, 0)))));
        act[col] = tobf(@floatCast(r * r));
    }
    var magnitude: f64 = 0;
    for (0..s.dims) |col| {
        const d = dot(down_dense[col * s.width ..][0..s.width], act);
        out[col] = d.value;
        magnitude = @max(magnitude, d.magnitude);
    }
    return magnitude;
}

test "the moe layer's slots and weights match the fp64 host, ties to the lower id" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1510);
    const random = prng.random();
    const w = try Weights.init(gpa, random);
    defer w.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    var layer: moe.Moe = try .init(gpa, &staging, fx.shape, try w.host(arena.allocator()));
    defer layer.deinit();
    try testing.expect(!layer.experts.gated);
    try testing.expectEqual(fx.count, layer.experts.count);
    try testing.expectEqual(fx.shape.dims, layer.experts.dims);
    try testing.expectEqual(fx.shape.width, layer.experts.width);

    const rows = 4;
    var sc: moe.Scratch = try .init(fx.shape, rows, false);
    defer sc.deinit();
    const x_bits = try gpa.alloc(u16, rows * fx.shape.dims);
    defer gpa.free(x_bits);
    for (x_bits) |*bits| bits.* = tobf(random.floatNorm(f32));
    var x_dev = try gpu.device(u16, .bf16, &.{ rows, fx.shape.dims }, x_bits);
    defer x_dev.deinit();
    try moe.window(layer, sc, x_dev.view);
    try cuda.synchronize();

    const picks = try fetch(i32, gpa, sc.picks);
    defer gpa.free(picks);
    const wts = try fetch(f32, gpa, sc.wts);
    defer gpa.free(wts);
    const y = try fetch(f32, gpa, sc.y);
    defer gpa.free(y);

    // the host's top-k over the router's fp64 logits: the kernels' sliced fma chains round in fp32, so the
    // ids and weights are checked against the fp64 scores (a tie would take the lower id either way)
    const act = try gpa.alloc(u16, fx.shape.width);
    defer gpa.free(act);
    const out = try gpa.alloc(f64, fx.shape.dims);
    defer gpa.free(out);
    for (0..rows) |row| {
        const x_row = x_bits[row * fx.shape.dims ..][0..fx.shape.dims];
        var scores: [fx.shape.routed]f64 = undefined;
        for (0..fx.shape.routed) |e| {
            var logit: f64 = 0;
            for (0..fx.shape.dims) |i| logit += @as(f64, bf(x_row[i])) * bf(w.router[e * fx.shape.dims + i]);
            scores[e] = 1.0 / (1.0 + @exp(-logit));
        }
        var sel = scores;
        for (&sel, w.bias) |*v, b| v.* += b;
        var picked: [fx.shape.top]usize = undefined;
        var total: f64 = 0;
        for (0..fx.shape.top) |k| {
            var bi: usize = 0;
            for (1..fx.shape.routed) |e| if (sel[e] > sel[bi] or (sel[e] == sel[bi] and e < bi)) {
                bi = e;
            };
            picked[k] = bi;
            total += scores[bi];
            sel[bi] = -std.math.inf(f64);
        }
        for (0..fx.slots) |slot| {
            const e = picks[row * fx.slots + slot];
            if (slot < fx.shape.top) {
                try testing.expectEqual(@as(i32, @intCast(picked[slot])), e);
                const want_wt = scores[picked[slot]] / total * fx.shape.scale;
                try testing.expectApproxEqAbs(want_wt, @as(f64, wts[row * fx.slots + slot]), 1e-4);
            } else {
                try testing.expectEqual(@as(i32, @intCast(fx.shape.routed + slot - fx.shape.top)), e);
                try testing.expectEqual(@as(f32, 1), wts[row * fx.slots + slot]);
            }
            // the slot's output row against the fp64 expert's (the kernel's fp32 down sums within it)
            const magnitude = slotRow(w, @intCast(e), x_row, act, out);
            for (0..fx.shape.dims) |col| {
                try testing.expectApproxEqAbs(out[col], y[(row * fx.slots + slot) * fx.shape.dims + col], magnitude * 1e-4 + 1e-8);
            }
        }
    }
}

test "the moe layer's rows are a serial step's bits, in a window and prompt chunks of 1 to 2049" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1511);
    const random = prng.random();
    const w = try Weights.init(gpa, random);
    defer w.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    var layer: moe.Moe = try .init(gpa, &staging, fx.shape, try w.host(arena.allocator()));
    defer layer.deinit();

    const max_rows = 2049;
    const s = fx.shape;
    var sc: moe.Scratch = try .init(s, max_rows, false);
    defer sc.deinit();
    var sc_step: moe.Scratch = try .init(s, moe.window_rows, false);
    defer sc_step.deinit();
    const x_bits = try gpa.alloc(u16, max_rows * s.dims);
    defer gpa.free(x_bits);
    for (x_bits) |*bits| bits.* = tobf(random.floatNorm(f32) * 1.5);
    const resid_bits = try gpa.alloc(u16, max_rows * s.dims);
    defer gpa.free(resid_bits);
    for (resid_bits) |*bits| bits.* = tobf(random.floatNorm(f32) * 0.5);
    var norm_dev = try gpu.device(u16, .bf16, &.{s.dims}, w.norm);
    defer norm_dev.deinit();
    var resid_dev = try gpu.device(u16, .bf16, &.{ max_rows, s.dims }, resid_bits);
    defer resid_dev.deinit();
    var x_dev = try gpu.device(u16, .bf16, &.{ max_rows, s.dims }, x_bits);
    defer x_dev.deinit();

    const y_serial = try gpa.alloc(f32, moe.window_rows * fx.slots * s.dims);
    defer gpa.free(y_serial);
    const wts_serial = try gpa.alloc(f32, moe.window_rows * fx.slots);
    defer gpa.free(wts_serial);
    const picks_serial = try gpa.alloc(i32, moe.window_rows * fx.slots);
    defer gpa.free(picks_serial);
    const out_serial = try gpa.alloc(u16, moe.window_rows * s.dims);
    defer gpa.free(out_serial);
    const h_serial = try gpa.alloc(u16, moe.window_rows * s.dims);
    defer gpa.free(h_serial);

    // every row of a verify window's span as its own one-row window: slots and combined rows the step's
    for (0..moe.window_rows) |row| {
        var h_dev: cuda.Buffer = try .init(.bf16, &.{ 1, s.dims });
        defer h_dev.deinit();
        var out_dev: cuda.Buffer = try .init(.bf16, &.{ 1, s.dims });
        defer out_dev.deinit();
        try moe.window(layer, sc_step, x_dev.span(row, 1));
        try moe.combine(layer, sc_step, resid_dev.span(row, 1), norm_dev.view, h_dev.view, out_dev.view, 1);
        try cuda.synchronize();
        try sc_step.y.downloadAt(0, std.mem.sliceAsBytes(y_serial[row * fx.slots * s.dims ..][0 .. fx.slots * s.dims]));
        try sc_step.wts.downloadAt(0, std.mem.sliceAsBytes(wts_serial[row * fx.slots ..][0..fx.slots]));
        try sc_step.picks.downloadAt(0, std.mem.sliceAsBytes(picks_serial[row * fx.slots ..][0..fx.slots]));
        try h_dev.download(std.mem.sliceAsBytes(h_serial[row * s.dims ..][0..s.dims]));
        try out_dev.download(std.mem.sliceAsBytes(out_serial[row * s.dims ..][0..s.dims]));
    }

    for ([_]usize{ 1, 17, 300, 2049 }) |rows| {
        var h_dev: cuda.Buffer = try .init(.bf16, &.{ rows, s.dims });
        defer h_dev.deinit();
        var out_dev: cuda.Buffer = try .init(.bf16, &.{ rows, s.dims });
        defer out_dev.deinit();
        if (rows <= moe.window_rows) try moe.window(layer, sc, x_dev.span(0, rows)) else try moe.prompt(layer, sc, x_dev.span(0, rows));
        try moe.combine(layer, sc, resid_dev.span(0, rows), norm_dev.view, h_dev.view, out_dev.view, rows);
        try cuda.synchronize();
        const y = try gpa.alloc(f32, rows * fx.slots * s.dims);
        defer gpa.free(y);
        try sc.y.downloadAt(0, std.mem.sliceAsBytes(y));
        const wts = try gpa.alloc(f32, rows * fx.slots);
        defer gpa.free(wts);
        try sc.wts.downloadAt(0, std.mem.sliceAsBytes(wts));
        const picks = try gpa.alloc(i32, rows * fx.slots);
        defer gpa.free(picks);
        try sc.picks.downloadAt(0, std.mem.sliceAsBytes(picks));
        const h = try fetch(u16, gpa, h_dev);
        defer gpa.free(h);
        const out = try fetch(u16, gpa, out_dev);
        defer gpa.free(out);
        // the serial rows' slots and combined rows, bit for bit
        const covered = @min(rows, moe.window_rows);
        try testing.expectEqualSlices(i32, picks_serial[0 .. covered * fx.slots], picks[0 .. covered * fx.slots]);
        try testing.expectEqualSlices(u32, @ptrCast(wts_serial[0 .. covered * fx.slots]), @ptrCast(wts[0 .. covered * fx.slots]));
        try testing.expectEqualSlices(u32, @ptrCast(y_serial[0 .. covered * fx.slots * s.dims]), @ptrCast(y[0 .. covered * fx.slots * s.dims]));
        try testing.expectEqualSlices(u16, h_serial[0 .. covered * s.dims], h[0 .. covered * s.dims]);
        try testing.expectEqualSlices(u16, out_serial[0 .. covered * s.dims], out[0 .. covered * s.dims]);
        for (out) |v| try testing.expect(std.math.isFinite(bf(v)));
    }

    // and a step after the prompt chunks: row 3's bits again, whatever ran before
    {
        var h_dev: cuda.Buffer = try .init(.bf16, &.{ 1, s.dims });
        defer h_dev.deinit();
        var out_dev: cuda.Buffer = try .init(.bf16, &.{ 1, s.dims });
        defer out_dev.deinit();
        try moe.window(layer, sc_step, x_dev.span(3, 1));
        try moe.combine(layer, sc_step, resid_dev.span(3, 1), norm_dev.view, h_dev.view, out_dev.view, 1);
        try cuda.synchronize();
        const h = try fetch(u16, gpa, h_dev);
        defer gpa.free(h);
        const out = try fetch(u16, gpa, out_dev);
        defer gpa.free(out);
        try testing.expectEqualSlices(u16, h_serial[3 * s.dims ..][0..s.dims], h);
        try testing.expectEqualSlices(u16, out_serial[3 * s.dims ..][0..s.dims], out);
    }
}
