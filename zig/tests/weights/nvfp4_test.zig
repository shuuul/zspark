//! `nvfp4.zig`'s tests.

const std = @import("std");
const cuda = @import("../../src/cuda/cuda.zig");
const numeric = @import("../../src/weights/numeric.zig");
const nvfp4 = @import("../../src/weights/nvfp4.zig");
const Allocator = std.mem.Allocator;
const Fp4Linear = nvfp4.Fp4Linear;
const Fp8BlockLinear = nvfp4.Fp8BlockLinear;
const Fp8Linear = nvfp4.Fp8Linear;
const Mx8Linear = nvfp4.Mx8Linear;
const checkLinear = @import("testing.zig").linears.checkLinear;
const e2m1 = nvfp4.e2m1;
const splitK = nvfp4.splitK;
const vision_ref = @import("../../src/models/qwen4/vision_ref.zig");

test "nvfp4 splitK depends on the shape alone" {
    try std.testing.expectEqual(@as(usize, 1), splitK(2560, 64));
    try std.testing.expectEqual(@as(usize, 4), splitK(640, 2560));
    try std.testing.expectEqual(@as(usize, 1), splitK(248320, 2560));
}

const Shape = struct { n: usize, k: usize };
// one K slice, then four
const shapes = [_]Shape{ .{ .n = 200, .k = 256 }, .{ .n = 640, .k = 2560 } };
// rows through every tile and the fused form
const decode_rows = [_]usize{ 1, 3, 16, 17, 33, 70, 300 };

fn checkPaths(gpa: Allocator, linear: anytype, dense: []const f64) !void {
    try checkLinear(gpa, linear, dense, &decode_rows, .decode);
    const tiles: usize = if (linear.k > 1024) 1 else 6;
    for (0..tiles) |tile| try checkLinear(gpa, linear, dense, &.{ 1, 130, 257 }, .{ .prompt = @intCast(tile) });
}

test "FP8 matmuls match the host and keep a row's bits across row counts, chunks and tiles" {
    const gpa = std.testing.allocator;
    for (shapes) |shape| {
        var prng: std.Random.DefaultPrng = .init(shape.n);
        const codes = try gpa.alloc(u8, shape.n * shape.k);
        defer gpa.free(codes);
        randomE4m3(prng.random(), codes);
        const scale = 1.0 / 448.0;
        const dense = try gpa.alloc(f64, codes.len);
        defer gpa.free(dense);
        for (dense, codes) |*value, code| value.* = @as(f64, numeric.f32FromE4m3(code)) * scale;

        var linear: Fp8Linear = try .init(gpa, codes, shape.n, shape.k, scale);
        defer linear.deinit();
        try checkPaths(gpa, linear, dense);
    }
}

/// Exact GELU as `prompt_bias.cu`'s epilogue applies it: the sum rounded to bf16 first, then
/// `x * 0.5 * (1 + erf(x / sqrt(2)))` of that rounded value (`vision_ref.erf`).
fn geluErf(want: f64) f64 {
    const v = @as(f64, numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(want))));
    return 0.5 * v * (1 + vision_ref.erf(v / std.math.sqrt2));
}

test "an FP8 linear's biased prompt path matches the host and keeps a row's bits across tiles" {
    const gpa = std.testing.allocator;
    for (shapes) |shape| {
        var prng: std.Random.DefaultPrng = .init(shape.n + 7);
        const random = prng.random();
        const codes = try gpa.alloc(u8, shape.n * shape.k);
        defer gpa.free(codes);
        randomE4m3(random, codes);
        const scale = 1.0 / 448.0;
        const dense = try gpa.alloc(f64, codes.len);
        defer gpa.free(dense);
        for (dense, codes) |*value, code| value.* = @as(f64, numeric.f32FromE4m3(code)) * scale;
        const bias = try gpa.alloc(u16, shape.n);
        defer gpa.free(bias);
        for (bias) |*b| b.* = numeric.bf16FromF32(random.float(f32) - 0.5);

        var linear: Fp8Linear = try .init(gpa, codes, shape.n, shape.k, scale);
        defer linear.deinit();
        var bias_b: cuda.Buffer = try .init(.bf16, &.{shape.n});
        defer bias_b.deinit();
        try bias_b.upload(std.mem.sliceAsBytes(bias));
        const got = try gpa.alloc(u16, 130 * shape.n);
        defer gpa.free(got);
        var x_prng: std.Random.DefaultPrng = .init(shape.k);
        const x_bits = try gpa.alloc(u16, 130 * shape.k);
        defer gpa.free(x_bits);
        for (x_bits) |*bits| bits.* = numeric.bf16FromF32(x_prng.random().float(f32) - 0.5);
        var x: cuda.Buffer = try .init(.bf16, &.{ 130, shape.k });
        defer x.deinit();
        try x.upload(std.mem.sliceAsBytes(x_bits));
        for ([_]nvfp4.Activation{ .none, .gelu_erf }) |act| {
            var reference: ?[]u16 = null;
            defer if (reference) |r| gpa.free(r);
            for ([_]u3{ 0, 4, 5 }) |tile| {
                var out: cuda.Buffer = try .init(.bf16, &.{ 130, shape.n });
                defer out.deinit();
                nvfp4.biased_tile = tile;
                try linear.forwardBiased(x.view, bias_b.view, out.view, act);
                try out.download(std.mem.sliceAsBytes(got));
                if (reference == null) {
                    reference = try gpa.dupe(u16, got);
                    // the host's fp64 product with the bias, near the kernel's bf16 chain
                    var worst: f64 = 0;
                    for (0..130 * shape.n) |i| {
                        const row = i / shape.n;
                        const col = i % shape.n;
                        var want: f64 = @as(f64, numeric.f32FromBf16(bias[col]));
                        for (0..shape.k) |t| want += @as(f64, numeric.f32FromBf16(x_bits[row * shape.k + t])) * dense[col * shape.k + t];
                        if (act == .gelu_erf) want = geluErf(want);
                        const v = @as(f64, numeric.f32FromBf16(got[i]));
                        worst = @max(worst, @abs(v - want) / @max(1.0, @abs(want)));
                    }
                    std.debug.print("fp8 biased {t}: worst rel {d:.6}\n", .{ act, worst });
                    try std.testing.expect(worst < 2e-2);
                } else try std.testing.expectEqualSlices(u16, reference.?, got);
            }
            nvfp4.biased_tile = null;
        }
    }
}

test "NVFP4 matmuls match the host and keep a row's bits across row counts, chunks and tiles" {
    const gpa = std.testing.allocator;
    for (shapes) |shape| {
        var prng: std.Random.DefaultPrng = .init(shape.n + 1);
        const random = prng.random();
        const codes = try gpa.alloc(u8, shape.n * shape.k / 2);
        defer gpa.free(codes);
        random.bytes(codes);
        const block_scales = try gpa.alloc(u8, shape.n * shape.k / 16);
        defer gpa.free(block_scales);
        // positive scales within a few binades, as a checkpoint's are
        for (block_scales) |*byte| byte.* = 0x20 + random.uintLessThan(u8, 0x30);
        const scale = 0.0173;
        const dense = try gpa.alloc(f64, shape.n * shape.k);
        defer gpa.free(dense);
        for (dense, 0..) |*value, i| {
            const byte = codes[i / 2];
            const code = if (i % 2 == 0) byte & 0xf else byte >> 4;
            value.* = @as(f64, e2m1[code]) * numeric.f32FromE4m3(block_scales[i / 16]) * @as(f32, scale);
        }

        var linear: Fp4Linear = try .init(gpa, codes, block_scales, shape.n, shape.k, scale);
        defer linear.deinit();
        try checkPaths(gpa, linear, dense);
    }
}

test "block FP8 matmul matches the host and keeps a row's bits across row counts" {
    const gpa = std.testing.allocator;
    for (shapes) |shape| {
        var prng: std.Random.DefaultPrng = .init(shape.n + 2);
        const random = prng.random();
        const codes = try gpa.alloc(u8, shape.n * shape.k);
        defer gpa.free(codes);
        randomE4m3(random, codes);
        const block_rows = (shape.n + 127) / 128;
        const block_scales = try gpa.alloc(f32, block_rows * (shape.k / 128));
        defer gpa.free(block_scales);
        for (block_scales) |*value| value.* = 0.001 + 0.01 * random.float(f32);
        const dense = try gpa.alloc(f64, codes.len);
        defer gpa.free(dense);
        for (dense, codes, 0..) |*value, code, i| {
            const output = i / shape.k;
            const input = i % shape.k;
            value.* = @as(f64, numeric.f32FromE4m3(code)) * block_scales[output / 128 * (shape.k / 128) + input / 128];
        }

        var linear: Fp8BlockLinear = try .init(gpa, codes, block_scales, shape.n, shape.k, .{});
        defer linear.deinit();
        try checkLinear(gpa, linear, dense, &decode_rows, .decode);
    }
}

test "MXFP8 matmuls match the host and keep a row's bits across row counts, alone or stacked" {
    const gpa = std.testing.allocator;
    // one K slice, four, and the stacks the faces make: parts that meet inside a tile, one of 16 outputs
    const stacks = [_]struct { k: usize, parts: []const usize }{
        .{ .k = 256, .parts = &.{200} },
        .{ .k = 2560, .parts = &.{640} },
        .{ .k = 2560, .parts = &.{ 100, 92, 16, 16 } },
    };
    for (stacks, 0..) |stack, seed| {
        var prng: std.Random.DefaultPrng = .init(seed + 11);
        const random = prng.random();
        var n: usize = 0;
        for (stack.parts) |rows| n += rows;
        const k = stack.k;
        const codes = try gpa.alloc(u8, n * k);
        defer gpa.free(codes);
        randomE4m3(random, codes);
        const exponents = try gpa.alloc(u8, n * k / 32);
        defer gpa.free(exponents);
        // powers of two as a checkpoint's weights take them, 2^-14 to 2^-3, and the ends of the range
        for (exponents) |*byte| byte.* = 113 + random.uintLessThan(u8, 12);
        exponents[0] = 1;
        exponents[1] = 160;
        const dense = try gpa.alloc(f64, n * k);
        defer gpa.free(dense);
        for (dense, codes, 0..) |*value, code, i| {
            const output = i / k;
            const input = i % k;
            value.* = std.math.ldexp(@as(f64, numeric.f32FromE4m3(code)), @as(i32, exponents[output * (k / 32) + input / 32]) - 127);
        }
        var parts: [4]nvfp4.Mx8Rows = undefined;
        var first: usize = 0;
        for (stack.parts, parts[0..stack.parts.len]) |rows, *part| {
            part.* = .{ .codes = codes[first * k ..][0 .. rows * k], .exponents = exponents[first * k / 32 ..][0 .. rows * k / 32], .n = rows };
            first += rows;
        }
        var linear: Mx8Linear = try .initStacked(gpa, parts[0..stack.parts.len], k);
        defer linear.deinit();
        try std.testing.expectEqual(n, linear.n);
        try checkLinear(gpa, linear, dense, &decode_rows, .decode);
        // read back as bf16 rows: each weight as the host widens it, the padding's outputs gone
        const rows_of = try linear.bf16Rows(gpa);
        defer gpa.free(rows_of);
        try std.testing.expectEqual(n * k * 2, rows_of.len);
        for (dense, 0..) |value, i| {
            const bits = std.mem.readInt(u16, rows_of[i * 2 ..][0..2], .little);
            try std.testing.expectEqual(numeric.bf16FromF32(@floatCast(value)), bits);
        }
    }
}

// one K slice; four by a cluster; two summed in each block (40 tiles); a stack whose second part ends in a tile
const tile_shapes = [_][]const usize{ &.{200}, &.{640}, &.{2560}, &.{ 128, 4 }, &.{2048} };
const tile_inputs = [_]usize{ 256, 2560, 1024, 1024, 4096 };

test "the tile-scale FP8 matmul gives the lane matmul's bits, a row alone as in a window" {
    const gpa = std.testing.allocator;
    const before = nvfp4.tile_rows;
    defer nvfp4.tile_rows = before;
    for (tile_shapes, tile_inputs) |outputs, k| {
        var prng: std.Random.DefaultPrng = .init(outputs[0] + 3);
        const random = prng.random();
        var parts: [2]nvfp4.Rows = undefined;
        var scales: [2][]f32 = undefined;
        var n: usize = 0;
        for (outputs, 0..) |rows, i| {
            const codes = try gpa.alloc(u8, rows * k);
            randomE4m3(random, codes);
            scales[i] = try gpa.alloc(f32, (rows + 127) / 128 * (k / 128));
            for (scales[i]) |*value| value.* = 0.001 + 0.01 * random.float(f32);
            parts[i] = .{ .codes = codes, .block_scales = .of(scales[i]), .n = rows };
            n += rows;
        }
        defer for (parts[0..outputs.len], scales[0..outputs.len]) |part, values| {
            gpa.free(part.codes);
            gpa.free(values);
        };
        nvfp4.tile_rows = 16;
        var linear: Fp8BlockLinear = try .initStacked(gpa, parts[0..outputs.len], k, .{});
        defer linear.deinit();
        try std.testing.expect(linear.tile_scales != null);
        try std.testing.expectEqual(outputs[0] == 2560 or outputs[0] == 2048, nvfp4.splitK(n, k) > 1 and (n + 63) / 64 >= nvfp4.fused_tiles);

        const x_bits = try gpa.alloc(u16, 16 * k);
        defer gpa.free(x_bits);
        for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
        const base = try gpa.alloc(u32, 16 * n);
        defer gpa.free(base);
        const got = try gpa.alloc(u32, 16 * n);
        defer gpa.free(got);
        const alone = try gpa.alloc(u32, n);
        defer gpa.free(alone);
        for ([_]cuda.DType{ .f32, .bf16 }) |dtype| for ([_]usize{ 1, 3, 9, 16 }) |m| {
            var x: cuda.Buffer = try .init(.bf16, &.{ m, k });
            defer x.deinit();
            try x.upload(std.mem.sliceAsBytes(x_bits[0 .. m * k]));
            var out: cuda.Buffer = try .init(dtype, &.{ m, n });
            defer out.deinit();
            const bytes = m * n * dtype.size();
            nvfp4.tile_rows = 0; // the lane matmul
            try linear.forward(x.view, out.view);
            try cuda.synchronize();
            try out.download(std.mem.sliceAsBytes(base)[0..bytes]);
            nvfp4.tile_rows = 16;
            try linear.forward(x.view, out.view);
            try cuda.synchronize();
            try out.download(std.mem.sliceAsBytes(got)[0..bytes]);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(base)[0..bytes], std.mem.sliceAsBytes(got)[0..bytes]);
            // the first row of every window is the row alone
            const row = std.mem.sliceAsBytes(got)[0 .. n * dtype.size()];
            if (m == 1) @memcpy(std.mem.sliceAsBytes(alone)[0..row.len], row) else try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(alone)[0..row.len], row);
        };
    }
}

test "a stack whose parts meet inside a tile with their own scales keeps the lane matmul" {
    const gpa = std.testing.allocator;
    const before = nvfp4.tile_rows;
    defer nvfp4.tile_rows = before;
    nvfp4.tile_rows = 16;
    var prng: std.Random.DefaultPrng = .init(7);
    const codes = try gpa.alloc(u8, 100 * 256);
    defer gpa.free(codes);
    randomE4m3(prng.random(), codes);
    const first = [_]f32{ 0.004, 0.006 };
    const second = [_]f32{ 0.005, 0.007 };
    var linear: Fp8BlockLinear = try .initStacked(gpa, &.{ .{ .codes = codes, .block_scales = .of(&first), .n = 100 }, .{ .codes = codes[0 .. 92 * 256], .block_scales = .of(&second), .n = 92 } }, 256, .{});
    defer linear.deinit();
    try std.testing.expect(linear.tile_scales == null);
}

fn randomE4m3(random: std.Random, out: []u8) void {
    for (out) |*byte| {
        byte.* = random.int(u8);
        if (byte.* & 0x7f == 0x7f) byte.* &= 0xfe; // not NaN
    }
}

test "a lane matmul's block that sums its tile's K slices gives the cluster's bits, and a row's own at any row count" {
    const gpa = std.testing.allocator;
    const before = nvfp4.lane_fused_tiles;
    defer nvfp4.lane_fused_tiles = before;
    // 40 tiles of two K slices
    const n = 2560;
    const k = 1024;
    try std.testing.expectEqual(@as(usize, 2), splitK(n, k));
    var prng: std.Random.DefaultPrng = .init(105);
    const random = prng.random();
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    randomE4m3(random, codes);
    const exponents = try gpa.alloc(u8, n * k / 32);
    defer gpa.free(exponents);
    for (exponents) |*byte| byte.* = 113 + random.uintLessThan(u8, 12);
    var mx8: Mx8Linear = try .init(gpa, codes, exponents, n, k);
    defer mx8.deinit();
    const block_scales = try gpa.alloc(u8, n * k / 16);
    defer gpa.free(block_scales);
    for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
    var fp4: Fp4Linear = try .init(gpa, codes[0 .. n * k / 2], block_scales, n, k, 0.02);
    defer fp4.deinit();
    const most = 40;
    const x_host = try gpa.alloc(u16, most * k);
    defer gpa.free(x_host);
    for (x_host) |*value| value.* = numeric.bf16FromF32(random.floatNorm(f32));
    @memcpy(x_host[(most - 1) * k ..][0..k], x_host[0..k]);
    var x: cuda.Buffer = try .init(.bf16, &.{ most, k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_host));
    var out: cuda.Buffer = try .init(.bf16, &.{ most, n });
    defer out.deinit();
    const got = try gpa.alloc(u16, most * n);
    defer gpa.free(got);
    const first = try gpa.alloc(u16, most * n);
    defer gpa.free(first);
    inline for (.{ mx8, fp4 }) |linear| {
        // every form at 1 row, then each count's rows against those, the last row as the first
        var row0: [n]u16 = undefined;
        for ([_]usize{ 1, 3, 16, 17, 32, 33 }, 0..) |rows, at| {
            for ([_]usize{ 0, 40 }) |tiles| {
                nvfp4.lane_fused_tiles = tiles;
                // the window's rows end with x's first row, so its last output row is that row's
                const window = x.span(most - rows, rows);
                try linear.forward(window, out.rows(rows));
                try out.download(std.mem.sliceAsBytes(got[0 .. most * n]));
                if (tiles == 0) @memcpy(first[0 .. rows * n], got[0 .. rows * n]) else try std.testing.expectEqualSlices(u16, first[0 .. rows * n], got[0 .. rows * n]);
                if (at == 0 and tiles == 0) @memcpy(&row0, got[0..n]) else try std.testing.expectEqualSlices(u16, &row0, got[(rows - 1) * n ..][0..n]);
            }
        }
    }
}

test "an MXFP8 lane matmul's 128-row tile gives the 64-row tile's bits, and a row's own at any row count" {
    const gpa = std.testing.allocator;
    const before = nvfp4.lane_wide_rows;
    defer nvfp4.lane_wide_rows = before;
    var prng: std.Random.DefaultPrng = .init(1052);
    const random = prng.random();
    const most = 520;
    // one K slice on ragged columns, then four summed in the block
    for ([_]Shape{ .{ .n = 200, .k = 256 }, .{ .n = 640, .k = 2560 } }, [_]usize{ 1, 4 }) |shape, slices| {
        const n = shape.n;
        const k = shape.k;
        try std.testing.expectEqual(slices, splitK(n, k));
        const codes = try gpa.alloc(u8, n * k);
        defer gpa.free(codes);
        randomE4m3(random, codes);
        const exponents = try gpa.alloc(u8, n * k / 32);
        defer gpa.free(exponents);
        for (exponents) |*byte| byte.* = 113 + random.uintLessThan(u8, 12);
        var mx8: Mx8Linear = try .init(gpa, codes, exponents, n, k);
        defer mx8.deinit();
        const x_host = try gpa.alloc(u16, most * k);
        defer gpa.free(x_host);
        for (x_host) |*value| value.* = numeric.bf16FromF32(random.floatNorm(f32));
        @memcpy(x_host[(most - 1) * k ..][0..k], x_host[0..k]);
        var x: cuda.Buffer = try .init(.bf16, &.{ most, k });
        defer x.deinit();
        try x.upload(std.mem.sliceAsBytes(x_host));
        var out: cuda.Buffer = try .init(.bf16, &.{ most, n });
        defer out.deinit();
        const got = try gpa.alloc(u16, most * n);
        defer gpa.free(got);
        const first = try gpa.alloc(u16, most * n);
        defer gpa.free(first);
        const row0 = try gpa.alloc(u16, n);
        defer gpa.free(row0);
        nvfp4.lane_wide_rows = 0;
        try mx8.forward(x.span(0, 1), out.rows(1));
        try out.download(std.mem.sliceAsBytes(got));
        @memcpy(row0, got[0..n]);
        for ([_]usize{ 256, 257, 300, 513, most }) |rows| {
            for ([_]usize{ 0, 256 }) |wide| {
                nvfp4.lane_wide_rows = wide;
                // the window's rows end with x's first row, so its last output row is that row's
                try mx8.forward(x.span(most - rows, rows), out.rows(rows));
                try out.download(std.mem.sliceAsBytes(got));
                if (wide == 0) @memcpy(first[0 .. rows * n], got[0 .. rows * n]) else try std.testing.expectEqualSlices(u16, first[0 .. rows * n], got[0 .. rows * n]);
                try std.testing.expectEqualSlices(u16, row0, got[(rows - 1) * n ..][0..n]);
            }
        }
    }
}

test "a block FP8 lane matmul's 128-row tile gives the 64-row tile's bits, and a row's own at any row count" {
    const gpa = std.testing.allocator;
    const before = nvfp4.lane_wide_rows;
    defer nvfp4.lane_wide_rows = before;
    var prng: std.Random.DefaultPrng = .init(1072);
    const random = prng.random();
    const most = 520;
    // one K slice on ragged columns, then four summed in the block
    for ([_]Shape{ .{ .n = 200, .k = 256 }, .{ .n = 640, .k = 2560 } }, [_]usize{ 1, 4 }) |shape, slices| {
        const n = shape.n;
        const k = shape.k;
        try std.testing.expectEqual(slices, splitK(n, k));
        const codes = try gpa.alloc(u8, n * k);
        defer gpa.free(codes);
        randomE4m3(random, codes);
        const block_scales = try gpa.alloc(f32, (n + 127) / 128 * (k / 128));
        defer gpa.free(block_scales);
        for (block_scales) |*value| value.* = 0.001 + 0.01 * random.float(f32);
        var mx8: Fp8BlockLinear = try .init(gpa, codes, block_scales, n, k, .{});
        defer mx8.deinit();
        const x_host = try gpa.alloc(u16, most * k);
        defer gpa.free(x_host);
        for (x_host) |*value| value.* = numeric.bf16FromF32(random.floatNorm(f32));
        @memcpy(x_host[(most - 1) * k ..][0..k], x_host[0..k]);
        var x: cuda.Buffer = try .init(.bf16, &.{ most, k });
        defer x.deinit();
        try x.upload(std.mem.sliceAsBytes(x_host));
        var out: cuda.Buffer = try .init(.bf16, &.{ most, n });
        defer out.deinit();
        const got = try gpa.alloc(u16, most * n);
        defer gpa.free(got);
        const first = try gpa.alloc(u16, most * n);
        defer gpa.free(first);
        const row0 = try gpa.alloc(u16, n);
        defer gpa.free(row0);
        nvfp4.lane_wide_rows = 0;
        try mx8.forward(x.span(0, 1), out.rows(1));
        try out.download(std.mem.sliceAsBytes(got));
        @memcpy(row0, got[0..n]);
        for ([_]usize{ 256, 257, 300, 513, most }) |rows| {
            for ([_]usize{ 0, 256 }) |wide| {
                nvfp4.lane_wide_rows = wide;
                // the window's rows end with x's first row, so its last output row is that row's
                try mx8.forward(x.span(most - rows, rows), out.rows(rows));
                try out.download(std.mem.sliceAsBytes(got));
                if (wide == 0) @memcpy(first[0 .. rows * n], got[0 .. rows * n]) else try std.testing.expectEqualSlices(u16, first[0 .. rows * n], got[0 .. rows * n]);
                try std.testing.expectEqualSlices(u16, row0, got[(rows - 1) * n ..][0..n]);
            }
        }
    }
}
