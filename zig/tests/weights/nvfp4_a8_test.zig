//! `qmm_a8.cu`'s tests through `nvfp4.zig`: block FP8 and MXFP8 linears whose rows are quantized as the card's
//! vLLM quantizes them (spec 114 WS-04). A host model quantizes each row as vLLM does, so a product's error is
//! the fp32 sums' alone; a row's bits are its own at any row count, tile and K-slice sum.

const std = @import("std");
const cuda = @import("../../src/cuda/cuda.zig");
const numeric = @import("../../src/weights/numeric.zig");
const nvfp4 = @import("../../src/weights/nvfp4.zig");
const Allocator = std.mem.Allocator;

const Mode = enum { mx, block, tensor };

/// One row's k inputs as vLLM's quantization gives them back: block FP8 `per_token_group_fp8_quant` (an fp32
/// scale a 128, s = max(amax, 1e-10) / 448), MXFP8 b12x's `_quantize.py` (an e8m0 a 32, the least 2^e not under
/// amax / 448), e4m3 by round to nearest even, saturated at 448.
fn hostRow(mode: Mode, row: []const u16, out: []f64) void {
    const group: usize = if (mode == .mx) 32 else 128;
    var first: usize = 0;
    while (first < row.len) : (first += group) {
        var amax: f32 = 0;
        for (row[first..][0..group]) |bits| amax = @max(amax, @abs(numeric.f32FromBf16(bits)));
        for (row[first..][0..group], out[first..][0..group]) |bits, *value| {
            const v = numeric.f32FromBf16(bits);
            switch (mode) {
                .mx => {
                    const e = mxExponent(amax);
                    const q = std.math.clamp(v * std.math.ldexp(@as(f32, 1), -@min(e, 126)), -448, 448);
                    value.* = std.math.ldexp(@as(f64, numeric.f32FromE4m3(numeric.e4m3FromF32(q))), e);
                },
                .block, .tensor => {
                    const s: f32 = if (mode == .tensor) 0.03125 else @max(amax, 1e-10) / 448.0;
                    const q = std.math.clamp(v / s, -448, 448);
                    value.* = @as(f64, numeric.f32FromE4m3(numeric.e4m3FromF32(q))) * s;
                },
            }
        }
    }
}

fn mxExponent(amax: f32) i32 {
    if (!(amax > 0)) return 0;
    const ratio = amax / 448.0;
    var e: i32 = -149;
    while (std.math.ldexp(@as(f32, 1), e) < ratio) e += 1;
    return std.math.clamp(e, -127, 127);
}

test "block FP8's quotient by a reciprocal and a residual gives IEEE division's e4m3 byte, as qmm_a8.cu takes it" {
    var prng: std.Random.DefaultPrng = .init(1142);
    const random = prng.random();
    for (0..4_000_000) |i| {
        // a finite bf16 input and its group's largest |x| at or above it, from the subnormals to the top
        var bits = random.int(u16);
        if (bits & 0x7f80 == 0x7f80) bits &= 0xbfff;
        const v = numeric.f32FromBf16(bits);
        const amax: f32 = if (i % 4 == 0) @abs(v) else @abs(v) * (1 + 1000 * random.float(f32));
        if (!std.math.isFinite(amax)) continue;
        const s = @max(amax, 1e-10) / 448.0;
        const r = 1.0 / s;
        const q0 = v * r;
        const q = std.math.copysign(@mulAdd(f32, @mulAdd(f32, -q0, s, v), r, q0), v);
        // the byte always; the quotient's bits where the residual cannot underflow (the rest are e4m3 zeros)
        const byte = numeric.e4m3FromF32(std.math.clamp(q, -448, 448));
        const exact = numeric.e4m3FromF32(std.math.clamp(v / s, -448, 448));
        if (byte != exact or (@abs(v / s) >= 0x1p-24 and @as(u32, @bitCast(q)) != @as(u32, @bitCast(v / s)))) {
            std.debug.print("{e} / {e}: {e} for {e}\n", .{ v, s, q, v / s });
            return error.TestUnexpectedResult;
        }
    }
}

fn randomE4m3(random: std.Random, out: []u8) void {
    for (out) |*byte| {
        byte.* = random.int(u8);
        if (byte.* & 0x7f == 0x7f) byte.* &= 0xfe; // not NaN
    }
}

const Weights = struct {
    dense: []f64,
    mx: ?nvfp4.Mx8Linear = null,
    block: ?nvfp4.Fp8BlockLinear = null,
    input_scale: f32 = 0,

    fn init(gpa: Allocator, mode: Mode, random: std.Random, n: usize, k: usize) !Weights {
        const codes = try gpa.alloc(u8, n * k);
        defer gpa.free(codes);
        randomE4m3(random, codes);
        const dense = try gpa.alloc(f64, n * k);
        errdefer gpa.free(dense);
        switch (mode) {
            .mx => {
                const exponents = try gpa.alloc(u8, n * k / 32);
                defer gpa.free(exponents);
                for (exponents) |*byte| byte.* = 113 + random.uintLessThan(u8, 12);
                for (dense, codes, 0..) |*value, code, i|
                    value.* = std.math.ldexp(@as(f64, numeric.f32FromE4m3(code)), @as(i32, exponents[i / 32]) - 127);
                return .{ .dense = dense, .mx = try .init(gpa, codes, exponents, n, k) };
            },
            .block, .tensor => {
                const scales = try gpa.alloc(f32, (n + 127) / 128 * (k / 128));
                defer gpa.free(scales);
                for (scales) |*value| value.* = 0.001 + 0.01 * random.float(f32);
                for (dense, codes, 0..) |*value, code, i|
                    value.* = @as(f64, numeric.f32FromE4m3(code)) * scales[i / k / 128 * (k / 128) + i % k / 128];
                return .{ .dense = dense, .block = try .init(gpa, codes, scales, n, k, .{}), .input_scale = if (mode == .tensor) 0.03125 else 0 };
            },
        }
    }

    fn deinit(self: *Weights, gpa: Allocator) void {
        if (self.mx) |*linear| linear.deinit();
        if (self.block) |*linear| linear.deinit();
        gpa.free(self.dense);
    }

    fn quantize(self: *Weights, rows: cuda.Scratch.Rows) !void {
        if (self.mx) |*linear| try std.testing.expect(try linear.quantizeActivations(rows));
        if (self.block) |*linear| {
            if (self.input_scale > 0) try linear.quantizeStaticActivations(rows, self.input_scale) else try std.testing.expect(try linear.quantizeActivations(rows));
        }
    }

    fn forward(self: Weights, x: cuda.View, out: cuda.View) !void {
        if (self.mx) |linear| return linear.forward(x, out);
        return self.block.?.forward(x, out);
    }
};

/// Every output of `rows` rows of x against the host's quantized rows times the weight in fp64.
fn checkHost(gpa: Allocator, mode: Mode, dense: []const f64, x: []const u16, got: []const f32, rows: usize, n: usize, k: usize) !void {
    const q = try gpa.alloc(f64, k);
    defer gpa.free(q);
    for (0..rows) |row| {
        hostRow(mode, x[row * k ..][0..k], q);
        for (0..n) |col| {
            var want: f64 = 0;
            var magnitude: f64 = 0;
            for (q, dense[col * k ..][0..k]) |a, w| {
                want += a * w;
                magnitude += @abs(a * w);
            }
            const value: f64 = got[row * n + col];
            if (!(@abs(want - value) <= magnitude * 1e-5 + 1e-30)) {
                std.debug.print("{t} row {d} col {d}: {e} for {e} (sum of |terms| {e})\n", .{ mode, row, col, value, want, magnitude });
                return error.TestExpectedApproxEqAbs;
            }
        }
    }
}

// one K slice; four by a cluster (10 tiles); two summed in each block (40 tiles)
const shapes = [_]struct { n: usize, k: usize, slices: usize }{
    .{ .n = 200, .k = 256, .slices = 1 },
    .{ .n = 640, .k = 2560, .slices = 4 },
    .{ .n = 2560, .k = 1024, .slices = 2 },
};

test "FP8 linears on quantized rows match vLLM's quantization and keep a row's bits at any row count" {
    const gpa = std.testing.allocator;
    const most = 300;
    const cases = shapes ++ [_]@TypeOf(shapes[0]){.{ .n = 64, .k = 2688, .slices = 2 }};
    for ([_]Mode{ .mx, .block, .tensor }) |mode| for (cases, 0..) |shape, seed| {
        const n = shape.n;
        const k = shape.k;
        try std.testing.expectEqual(shape.slices, nvfp4.splitK(n, k));
        var prng: std.Random.DefaultPrng = .init(seed * 2 + @as(u64, @backingInt(mode)) + 1140);
        const random = prng.random();
        var weights: Weights = try .init(gpa, mode, random, n, k);
        defer weights.deinit(gpa);
        var scratch: cuda.Scratch = .init(gpa);
        defer scratch.deinit();
        try weights.quantize(scratch.rows(most));

        const x_host = try gpa.alloc(u16, most * k);
        defer gpa.free(x_host);
        // rows of differing scale, so that groups' scales differ
        for (x_host, 0..) |*value, i| value.* = numeric.bf16FromF32(random.floatNorm(f32) * std.math.ldexp(@as(f32, 1), @intCast(i / 32 % 9)));
        @memcpy(x_host[(most - 1) * k ..][0..k], x_host[0..k]);
        var x: cuda.Buffer = try .init(.bf16, &.{ most, k });
        defer x.deinit();
        try x.upload(std.mem.sliceAsBytes(x_host));
        var out: cuda.Buffer = try .init(.f32, &.{ most, n });
        defer out.deinit();
        const got = try gpa.alloc(f32, most * n);
        defer gpa.free(got);
        const first = try gpa.alloc(f32, most * n);
        defer gpa.free(first);
        const row0 = try gpa.alloc(f32, n);
        defer gpa.free(row0);

        try weights.forward(x.span(0, 1), out.rows(1));
        try out.download(std.mem.sliceAsBytes(got));
        @memcpy(row0, got[0..n]);
        try checkHost(gpa, mode, weights.dense, x_host, got, 1, n, k);
        // the in-block quantization (bm 16) to 16 rows, then 64-row tiles on the quantize kernel's bytes
        for ([_]usize{ 2, 6, 16, 17, 70, 257, most }) |rows| {
            // the window's rows end with x's first row, so its last output row is that row's
            try weights.forward(x.span(most - rows, rows), out.rows(rows));
            try out.download(std.mem.sliceAsBytes(got));
            try std.testing.expectEqualSlices(u32, @ptrCast(row0), @ptrCast(got[(rows - 1) * n ..][0..n]));
            if (rows == most) @memcpy(first, got);
            try checkHost(gpa, mode, weights.dense, x_host[(most - rows) * k ..], got, @min(rows, 24), n, k);
        }
        // the last 16 rows alone in the block as among 300 in a tile of the quantize kernel's bytes
        try weights.forward(x.span(most - 16, 16), out.rows(16));
        try out.download(std.mem.sliceAsBytes(got));
        try std.testing.expectEqualSlices(u32, @ptrCast(first[(most - 16) * n ..][0 .. 16 * n]), @ptrCast(got[0 .. 16 * n]));
        // a bf16 out is the fp32 out rounded once, in each tile (spec 116 WS-04: a quantized window's partials)
        var half: cuda.Buffer = try .init(.bf16, &.{ most, n });
        defer half.deinit();
        const halves = try gpa.alloc(u16, most * n);
        defer gpa.free(halves);
        for ([_]usize{ 16, most }) |rows| {
            try weights.forward(x.span(most - rows, rows), out.rows(rows));
            try cuda.download(out.rows(rows), std.mem.sliceAsBytes(got[0 .. rows * n]));
            try weights.forward(x.span(most - rows, rows), half.rows(rows));
            try cuda.download(half.rows(rows), std.mem.sliceAsBytes(halves[0 .. rows * n]));
            for (got[0 .. rows * n], halves[0 .. rows * n]) |wide, narrow| try std.testing.expectEqual(numeric.bf16FromF32(wide), narrow);
        }
    };
}

test "a K-split linear on quantized rows sums its slices in the block as a cluster does, bit for bit" {
    const gpa = std.testing.allocator;
    const n = 640;
    const k = 2560;
    const most = 300;
    const sk = nvfp4.splitK(n, k);
    try std.testing.expectEqual(@as(usize, 4), sk);
    for ([_]Mode{ .mx, .block }) |mode| {
        var prng: std.Random.DefaultPrng = .init(1143 + @as(u64, @backingInt(mode)));
        const random = prng.random();
        var weights: Weights = try .init(gpa, mode, random, n, k);
        defer weights.deinit(gpa);
        const w = if (weights.mx) |linear| linear.w8.view else weights.block.?.w8.view;
        const bs = if (weights.mx) |linear| linear.exponents.view else weights.block.?.tile_scales.?.view;
        const npad = if (weights.mx) |linear| linear.npad else weights.block.?.npad;
        const x_host = try gpa.alloc(u16, most * k);
        defer gpa.free(x_host);
        for (x_host) |*value| value.* = numeric.bf16FromF32(random.floatNorm(f32));
        var x: cuda.Buffer = try .init(.bf16, &.{ most, k });
        defer x.deinit();
        try x.upload(std.mem.sliceAsBytes(x_host));
        var work: cuda.Buffer = try .init(.u8, &.{most * k / 32 * 33});
        defer work.deinit();
        var out: cuda.Buffer = try .init(.f32, &.{ most, n });
        defer out.deinit();
        var got: [2][]f32 = undefined;
        for (&got) |*one| one.* = try gpa.alloc(f32, most * n);
        defer for (got) |one| gpa.free(one);
        // the in-block quantization (bm 16), then the quantize kernel's bytes (bm 64)
        for ([_]usize{ 1, 16, 70, most }) |rows| {
            for (&got, 0..) |*one, fuse| {
                const x_view = x.span(0, rows);
                const out_view = out.rows(rows);
                const bm: i64 = if (rows <= nvfp4.a8_tile_rows) 16 else 64;
                try cuda.launched(cuda.kernels.zs_nvfp4_qmm_a8(&x_view, &w, &bs, &out_view, &work.view, if (mode == .mx) 2 else 3, n, @intCast(sk), @intCast(npad), bm, @intCast(fuse), 1, 0));
                try out.download(std.mem.sliceAsBytes(one.*));
            }
            try std.testing.expectEqualSlices(u32, @ptrCast(got[0][0 .. rows * n]), @ptrCast(got[1][0 .. rows * n]));
        }
    }
}

test "the bm-16 tile on the quantize kernel's bytes is the in-block quantization's, bit for bit" {
    // spec 153 WS-05: Nemotron's static-FP8 mamba projections (FP8G, input_scale > 0) quantize x once by
    // quant_a8_kernel at every row count, the bm-16 tile then reading the pre-quantized bytes as bm 64 does.
    // Same encode16 in both -> the same e4m3 bytes and scales -> the same products; proven here on the fp32 out.
    const gpa = std.testing.allocator;
    const n = 640;
    const k = 2560;
    const most = 16;
    const sk = nvfp4.splitK(n, k);
    var prng: std.Random.DefaultPrng = .init(15305);
    const random = prng.random();
    var weights: Weights = try .init(gpa, .tensor, random, n, k);
    defer weights.deinit(gpa);
    const w = weights.block.?.w8.view;
    const bs = weights.block.?.tile_scales.?.view;
    const npad = weights.block.?.npad;
    const x_host = try gpa.alloc(u16, most * k);
    defer gpa.free(x_host);
    for (x_host, 0..) |*value, i| value.* = numeric.bf16FromF32(random.floatNorm(f32) * std.math.ldexp(@as(f32, 1), @intCast(i % 11)));
    var x: cuda.Buffer = try .init(.bf16, &.{ most, k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_host));
    var work: cuda.Buffer = try .init(.u8, &.{most * k / 32 * 33});
    defer work.deinit();
    var out: cuda.Buffer = try .init(.f32, &.{ most, n });
    defer out.deinit();
    const a = try gpa.alloc(f32, most * n);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, most * n);
    defer gpa.free(b);
    for ([_]usize{ 1, 4, 16 }) |rows| for ([_]bool{ false, true }) |fused| {
        const x_view = x.span(0, rows);
        const out_view = out.rows(rows);
        const fuse: c_int = @intFromBool(fused);
        // null work: the tile quantizes x in the block, stage by stage
        try cuda.launched(cuda.kernels.zs_nvfp4_qmm_a8(&x_view, &w, &bs, &out_view, null, 3, n, @intCast(sk), @intCast(npad), 16, fuse, 1, 0.03125));
        try out.download(std.mem.sliceAsBytes(a));
        // work given: quant_a8_kernel first, the tile reading its bytes
        try cuda.launched(cuda.kernels.zs_nvfp4_qmm_a8(&x_view, &w, &bs, &out_view, &work.view, 3, n, @intCast(sk), @intCast(npad), 16, fuse, 1, 0.03125));
        try out.download(std.mem.sliceAsBytes(b));
        try std.testing.expectEqualSlices(u32, @ptrCast(a[0 .. rows * n]), @ptrCast(b[0 .. rows * n]));
    };
}

test "FP8 linears on quantized rows keep zeros, tiny and huge rows finite and as vLLM's" {
    const gpa = std.testing.allocator;
    const n = 200;
    const k = 256;
    const rows = 24;
    for ([_]Mode{ .mx, .block, .tensor }) |mode| {
        var prng: std.Random.DefaultPrng = .init(1141 + @as(u64, @backingInt(mode)));
        const random = prng.random();
        var weights: Weights = try .init(gpa, mode, random, n, k);
        defer weights.deinit(gpa);
        var scratch: cuda.Scratch = .init(gpa);
        defer scratch.deinit();
        try weights.quantize(scratch.rows(rows));
        const x_host = try gpa.alloc(u16, rows * k);
        defer gpa.free(x_host);
        for (x_host, 0..) |*value, i| {
            const row = i / k;
            const scale: f32 = switch (row % 6) {
                0 => 0,
                1 => 1e-38, // bf16's subnormals and smallest normals
                2 => 1e30,
                3 => if (i % 7 == 0) 1e20 else 1e-20, // one large input a group
                4 => 1e-6,
                else => 1,
            };
            value.* = numeric.bf16FromF32(random.floatNorm(f32) * scale);
        }
        var x: cuda.Buffer = try .init(.bf16, &.{ rows, k });
        defer x.deinit();
        try x.upload(std.mem.sliceAsBytes(x_host));
        var out: cuda.Buffer = try .init(.f32, &.{ rows, n });
        defer out.deinit();
        const got = try gpa.alloc(f32, rows * n);
        defer gpa.free(got);
        // 24 rows through the quantize kernel, then the first 16 quantized in the block
        for ([_]usize{ rows, 16 }) |m| {
            try weights.forward(x.span(0, m), out.rows(m));
            try out.download(std.mem.sliceAsBytes(got));
            for (got[0 .. m * n]) |value| try std.testing.expect(std.math.isFinite(value));
            try checkHost(gpa, mode, weights.dense, x_host, got, m, n, k);
        }
    }
}

test "an MXFP8 linear quantizes a call's rows from its pool's quantize_from on, each row the bits of its form alone" {
    const gpa = std.testing.allocator;
    const most = 70;
    const n = 640;
    const k = 2560;
    var prng: std.Random.DefaultPrng = .init(1150);
    const random = prng.random();
    var weights: Weights = try .init(gpa, .mx, random, n, k);
    defer weights.deinit(gpa);
    var scratch: cuda.Scratch = .init(gpa);
    defer scratch.deinit();
    try weights.quantize(scratch.rows(most));

    const x_host = try gpa.alloc(u16, most * k);
    defer gpa.free(x_host);
    for (x_host, 0..) |*value, i| value.* = numeric.bf16FromF32(random.floatNorm(f32) * std.math.ldexp(@as(f32, 1), @intCast(i / 32 % 9)));
    var x: cuda.Buffer = try .init(.bf16, &.{ most, k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_host));
    var out: cuda.Buffer = try .init(.f32, &.{ most, n });
    defer out.deinit();
    // every row bf16, then every row quantized
    const whole: [2][]f32 = .{ try gpa.alloc(f32, most * n), try gpa.alloc(f32, most * n) };
    defer for (whole) |one| gpa.free(one);
    for (whole, [_]usize{ std.math.maxInt(usize), 0 }) |one, from| {
        scratch.quantize_from = from;
        try weights.forward(x.view, out.view);
        try out.download(std.mem.sliceAsBytes(one));
    }
    try std.testing.expect(!std.mem.eql(u32, @ptrCast(whole[0]), @ptrCast(whole[1])));
    const got = try gpa.alloc(f32, most * n);
    defer gpa.free(got);
    // a split past the in-block quantization's 16 rows and within it
    for ([_]usize{ 1, 30, 60, most }) |from| {
        scratch.quantize_from = from;
        try weights.forward(x.view, out.view);
        try out.download(std.mem.sliceAsBytes(got));
        try std.testing.expectEqualSlices(u32, @ptrCast(whole[0][0 .. from * n]), @ptrCast(got[0 .. from * n]));
        try std.testing.expectEqualSlices(u32, @ptrCast(whole[1][from * n ..]), @ptrCast(got[from * n ..]));
    }
    scratch.quantize_from = 0;
}

test "NVFP4 linear activation rounding matches FP4 products and keeps a row alone across tiles" {
    const gpa = std.testing.allocator;
    const most = 70;
    const n = 64;
    const input_scale: f32 = 0.7;
    const weight_scale: f32 = 0.019;
    const gs: f32 = 1.0 / input_scale;
    const alpha: f32 = weight_scale / gs;
    for ([_]usize{ 256, 2688 }) |k| {
        var prng: std.Random.DefaultPrng = .init(k + 151);
        const random = prng.random();
        const codes = try gpa.alloc(u8, n * k / 2);
        defer gpa.free(codes);
        random.bytes(codes);
        const scales = try gpa.alloc(u8, n * k / 16);
        defer gpa.free(scales);
        for (scales) |*scale| scale.* = 16 + random.uintLessThan(u8, 96);
        var linear: nvfp4.Fp4Linear = try .init(gpa, codes, scales, n, k, weight_scale);
        defer linear.deinit();
        var scratch: cuda.Scratch = .init(gpa);
        defer scratch.deinit();
        _ = try scratch.lend(.part, .bf16, &.{ most * 2, k + 128 });
        try linear.quantizeActivations(scratch.rows(most), input_scale);
        const inputs = try gpa.alloc(u16, most * k);
        defer gpa.free(inputs);
        for (inputs, 0..) |*bits, i| {
            const scale: f32 = switch (i / k) {
                0 => 0,
                1 => 1e-30,
                2 => 1e20,
                else => 3,
            };
            bits.* = numeric.bf16FromF32(random.floatNorm(f32) * scale);
        }
        var x: cuda.Buffer = try .init(.bf16, &.{ most, k });
        defer x.deinit();
        try x.upload(std.mem.sliceAsBytes(inputs));
        var out: cuda.Buffer = try .init(.f32, &.{ most, n });
        defer out.deinit();
        const got = try gpa.alloc(f32, most * n);
        defer gpa.free(got);
        const baseline = try gpa.alloc(f32, n);
        defer gpa.free(baseline);
        const quantized = try gpa.alloc(f64, k);
        defer gpa.free(quantized);
        try linear.forward(x.span(most - 1, 1), out.rows(1));
        try cuda.download(out.rows(1), std.mem.sliceAsBytes(baseline));
        for ([_]usize{ 16, 17, 33, most }) |rows| {
            const first = most - rows;
            try linear.forward(x.span(first, rows), out.rows(rows));
            try cuda.download(out.rows(rows), std.mem.sliceAsBytes(got[0 .. rows * n]));
            try std.testing.expectEqualSlices(u32, @ptrCast(baseline), @ptrCast(got[(rows - 1) * n ..][0..n]));
            for (0..rows) |row| {
                for (0..k / 16) |group| {
                    var values: [16]f32 = undefined;
                    for (&values, 0..) |*value, i| value.* = numeric.f32FromBf16(inputs[(first + row) * k + group * 16 + i]);
                    const q = numeric.nvfp4Group(values, gs);
                    for (0..16) |i| quantized[group * 16 + i] = @as(f64, nvfp4.e2m1[@as(u4, @truncate(q.codes >> @intCast(i * 4)))]) * numeric.f32FromE4m3(q.scale);
                }
                for (0..n) |col| {
                    var sum: f64 = 0;
                    var magnitude: f64 = 0;
                    for (quantized, 0..) |value, i| {
                        const byte = codes[(col * k + i) / 2];
                        const code = if (i % 2 == 0) byte & 15 else byte >> 4;
                        const term = value * nvfp4.e2m1[code] * numeric.f32FromE4m3(scales[col * k / 16 + i / 16]);
                        sum += term;
                        magnitude += @abs(term);
                    }
                    try std.testing.expectApproxEqAbs(sum * alpha, @as(f64, got[row * n + col]), magnitude * alpha * 1e-5 + 1e-30);
                }
            }
        }
    }
}
