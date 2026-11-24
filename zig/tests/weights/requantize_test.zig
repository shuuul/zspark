//! `requantize.zig`'s tests. The fixture's codes and scales are TensorFold's own, exported by `tools/reference/requantize_cases.py`.

const std = @import("std");
const numeric = @import("../../src/weights/numeric.zig");
const nvfp4 = @import("../../src/weights/nvfp4.zig");
const requantize = @import("../../src/weights/requantize.zig");
const Quantized = requantize.Quantized;
const Rows = requantize.Rows;
const Weight = requantize.Weight;
const dequantize = requantize.dequantize;
const group = requantize.group;
const quantize = requantize.quantize;
const tiny = requantize.tiny;

const testing = std.testing;

const fx = struct {
    const cases = @embedFile("fixtures/requantize_cases.bin");

    /// A value's code among 0..7 in fp64, ties to the smaller: the rule without fp32's rounding.
    fn nearest(steps: f64) u8 {
        var best: u8 = 0;
        for (nvfp4.e2m1[0..8], 0..) |magnitude, code| {
            if (@abs(@abs(steps) - magnitude) < @abs(@abs(steps) - nvfp4.e2m1[best])) best = @intCast(code);
        }
        return best;
    }

    fn codeAt(codes: []const u8, i: usize) u8 {
        return if (i % 2 == 0) codes[i / 2] & 0xf else codes[i / 2] >> 4;
    }
};

test "hand-made blocks: the scales, each midpoint to the smaller code, signs, and a block too small to scale" {
    const n = 2;
    const k = 32;
    var values: [n * k]f32 = @splat(0);
    // the expert's largest value is 6 x 448 x 2^-10 = 2.625, so its scale is 2^-10 exactly (2.625 times the
    // fp32 reciprocal of 2688 rounds there) and a value's count of steps is value / (block scale x 2^-10),
    // exact where the count is: the midpoints are met, not approached
    const steps = [16]f32{ 6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, -6 };
    const want = [16]u8{ 7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15 };
    // block scale 448, then 2 (its largest is 12 x 2^-10)
    for (steps, 0..) |count, i| {
        values[i] = count * 448.0 / 1024.0;
        values[16 + i] = count * 2.0 / 1024.0;
    }
    // just off a midpoint on either side, and values beyond the largest code's reach
    values[32..48].* = .{ 12, 2.5002, 2.4998, -7.0002, -6.9998, 9.9, 10.1, 0.4998, -0.4998, 0.5002, -0.5002, 0, -0.0, 3, -3, 11.9 };
    for (values[32..48]) |*value| value.* *= 1.0 / 1024.0;
    // a block whose largest is below half the smallest scale's reach: scale 0, codes 0 whatever the sign
    values[48] = -1e-9;
    values[49] = 1e-9;

    var codes: [n * k / 2]u8 = undefined;
    var block_scales: [n * k / group]u8 = undefined;
    const expert = quantize(&values, n, k, &codes, &block_scales);
    try testing.expectEqual(@as(f32, 1.0 / 1024.0), expert);
    try testing.expectEqualSlices(u8, &.{ 0x7e, 0x40, 0x40, 0x00 }, &block_scales);
    for (want, 0..) |code, i| {
        try testing.expectEqual(code, fx.codeAt(&codes, i));
        try testing.expectEqual(code, fx.codeAt(&codes, 16 + i));
    }
    // steps of the third block: value x 1024 / 2
    const third = [16]u8{ 7, 3, 2, 14, 13, 6, 7, 0, 8, 1, 9, 0, 0, 3, 11, 7 };
    for (third, 0..) |code, i| try testing.expectEqual(code, fx.codeAt(&codes, 32 + i));
    try testing.expectEqualSlices(u8, &@as([8]u8, @splat(0)), codes[24..32]);

    // an expert of zeros: the floor of its scale, no block scale, no code
    const zeros: [k]f32 = @splat(0);
    try testing.expectEqual(tiny, quantize(&zeros, 1, k, codes[0..16], block_scales[0..2]));
    try testing.expectEqualSlices(u8, &.{ 0, 0 }, block_scales[0..2]);
    try testing.expectEqualSlices(u8, &@as([16]u8, @splat(0)), codes[0..16]);
}

test "requantized weights stay within the format's step of their source, and are the nearest codes" {
    const gpa = testing.allocator;
    const count = 5;
    const n = 256;
    const k = 384;
    var prng: std.Random.DefaultPrng = .init(11);
    const random = prng.random();

    // block FP8 as the checkpoint's MTP experts are: Gaussian values on a block's scale, the scales bf16
    const codes = try gpa.alloc(u8, count * n * k);
    defer gpa.free(codes);
    const scale_bits = try gpa.alloc(u16, count * (n / 128) * (k / 128));
    defer gpa.free(scale_bits);
    const bf16_bits = try gpa.alloc(u16, n * k);
    defer gpa.free(bf16_bits);
    const fp8_scales = try gpa.alloc(f32, n);
    defer gpa.free(fp8_scales);
    const weights = try gpa.alloc(Weight, count);
    defer gpa.free(weights);
    for (codes) |*code| code.* = numeric.e4m3FromF32(@min(448.0, @max(-448.0, 110.0 * random.floatNorm(f32))));
    for (scale_bits) |*bits| bits.* = numeric.bf16FromF32(1e-4 + 4e-4 * random.float(f32));
    for (bf16_bits) |*bits| bits.* = numeric.bf16FromF32(0.02 * random.floatNorm(f32));
    for (fp8_scales) |*scale| scale.* = 1e-4 + 2e-4 * random.float(f32);
    for (weights, 0..) |*weight, e| weight.* = .{ .fp8_block = .{
        .codes = codes[e * n * k ..][0 .. n * k],
        .block_scales = .{ .data = std.mem.sliceAsBytes(scale_bits[e * (n / 128) * (k / 128) ..][0 .. (n / 128) * (k / 128)]), .bf16 = true },
        .n = n,
    } };
    // and the other storages an MTP layer may have: bf16, FP8 with a scale an output, FP8 with one scale
    weights[2] = .{ .bf16 = std.mem.sliceAsBytes(bf16_bits) };
    weights[3] = .{ .fp8 = .{ .codes = codes[3 * n * k ..][0 .. n * k], .scales = .of(fp8_scales) } };
    weights[4] = .{ .fp8 = .{ .codes = codes[4 * n * k ..][0 .. n * k], .scales = .of(fp8_scales[0..1]) } };

    var quantized: Quantized = try .init(gpa, weights, n, k);
    defer quantized.deinit(gpa);
    const matrix = quantized.matrix();
    try testing.expectEqual(@as(usize, count), matrix.codes.len);

    const source = try gpa.alloc(f32, n * k);
    defer gpa.free(source);
    const got = try gpa.alloc(f32, n * k);
    defer gpa.free(got);
    for (weights, 0..) |weight, e| {
        weight.values(n, k, source);
        // the source as the host reads the storage, in fp64 up to the bf16 rounding
        for ([_]usize{ 0, 129 * k + 200, n * k - 1 }) |i| {
            const row = i / k;
            const column = i % k;
            const stored: f64 = switch (weight) {
                .bf16 => numeric.f32FromBf16(bf16_bits[i]),
                .fp8_block => |rows| @as(f64, numeric.f32FromE4m3(rows.codes[i])) * numeric.f32FromBf16(scale_bits[(e * (n / 128) + row / 128) * (k / 128) + column / 128]),
                .fp8 => |scaled| @as(f64, numeric.f32FromE4m3(scaled.codes[i])) * fp8_scales[if (e == 3) row else 0],
                .f32 => unreachable,
            };
            try testing.expectApproxEqRel(stored, @as(f64, source[i]), 1.0 / 256.0);
            try testing.expectEqual(numeric.f32FromBf16(numeric.bf16FromF32(source[i])), source[i]);
        }

        var largest: f32 = 0;
        for (source) |value| largest = @max(largest, @abs(value));
        const expert = matrix.scales[e];
        try testing.expectApproxEqRel(largest / 2688.0, expert, 2e-7);
        dequantize(matrix.codes[e], matrix.block_scales[e], expert, got);
        var error_sum: f64 = 0;
        var square_sum: f64 = 0;
        for (0..n * k / group) |b| {
            var block_largest: f32 = 0;
            for (source[b * group ..][0..group]) |value| block_largest = @max(block_largest, @abs(value));
            const scale: f64 = numeric.f32FromE4m3(matrix.block_scales[e][b]);
            // the block's scale is the e4m3 value nearest largest / 6 / expert scale: within half its 3-bit step
            const ideal = @as(f64, block_largest) / 6.0 / expert;
            try testing.expect(scale > 0);
            try testing.expect(@abs(scale - ideal) <= ideal / 16.0 * 1.0001);
            const step = scale * expert;
            for (source[b * group ..][0..group], got[b * group ..][0..group], 0..) |value, back, i| {
                // the codes are 0.5 apart up to 2, 1 apart to 4, 2 apart to 6: a value is within half a gap of
                // its code, and the scale's rounding may leave the block's largest past 6 by a sixteenth
                const steps = @abs(@as(f64, value)) / step;
                const gap: f64 = if (steps <= 2) 0.5 else if (steps <= 4) 1 else 2;
                const reach = if (steps > 6) steps - 6 else gap / 2;
                try testing.expect(@abs(@as(f64, back) - value) <= step * reach * 1.0001);
                try testing.expect(steps <= 6.0 * (1.0 + 1.0 / 15.0) * 1.0001);
                // and it is the nearest code, to the smaller on a tie, with the value's sign
                const code = fx.codeAt(matrix.codes[e], b * group + i);
                const exact = fx.nearest(@as(f64, value) / step);
                if (code & 7 != exact) {
                    // only where fp32 put the quotient on the other side of a midpoint it sits on in fp64
                    const midpoint = (@as(f64, nvfp4.e2m1[code & 7]) + nvfp4.e2m1[exact]) / 2.0;
                    try testing.expect(@abs(steps - midpoint) <= 1e-6 * midpoint);
                }
                try testing.expectEqual(value < 0, code >= 8);
                error_sum += (@as(f64, back) - value) * (@as(f64, back) - value);
                square_sum += @as(f64, value) * value;
            }
        }
        // Gaussian weights in blocks of 16 lose about a tenth of their rms to four bits
        const relative = @sqrt(error_sum / square_sum);
        try testing.expect(relative > 0.02 and relative < 0.16);
    }

    // one thread gives what many gave
    for (weights, 0..) |weight, e| {
        weight.values(n, k, source);
        const alone_codes = try gpa.alloc(u8, n * k / 2);
        defer gpa.free(alone_codes);
        const alone_scales = try gpa.alloc(u8, n * k / group);
        defer gpa.free(alone_scales);
        const expert = quantize(source, n, k, alone_codes, alone_scales);
        try testing.expectEqual(matrix.scales[e], expert);
        try testing.expectEqualSlices(u8, matrix.codes[e], alone_codes);
        try testing.expectEqualSlices(u8, matrix.block_scales[e], alone_scales);
    }
}

test "TensorFold's quantizer, bit for bit: block-FP8 and bf16 experts from its fixture" {
    const gpa = testing.allocator;
    const bytes = fx.cases;
    try testing.expectEqualStrings("ZSREQNT1", bytes[0..8]);
    var at: usize = 8;
    const Reader = struct {
        fn int(source: []const u8, offset: *usize) usize {
            defer offset.* += 4;
            return std.mem.readInt(u32, source[offset.*..][0..4], .little);
        }
        fn take(source: []const u8, offset: *usize, count: usize) []const u8 {
            defer offset.* += count;
            return source[offset.*..][0..count];
        }
    };
    const cases = Reader.int(bytes, &at);
    var kinds: [2]usize = @splat(0);
    for (0..cases) |_| {
        const kind = Reader.int(bytes, &at);
        const count = Reader.int(bytes, &at);
        const n = Reader.int(bytes, &at);
        const k = Reader.int(bytes, &at);
        kinds[kind] += 1;
        const weights = try gpa.alloc(Weight, count);
        defer gpa.free(weights);
        if (kind == 0) {
            const blocks = (n + 127) / 128 * (k / 128);
            const codes = Reader.take(bytes, &at, count * n * k);
            const scales = Reader.take(bytes, &at, count * blocks * 2);
            for (weights, 0..) |*weight, e| weight.* = .{ .fp8_block = .{
                .codes = codes[e * n * k ..][0 .. n * k],
                .block_scales = .{ .data = scales[e * blocks * 2 ..][0 .. blocks * 2], .bf16 = true },
                .n = n,
            } };
        } else {
            const values = Reader.take(bytes, &at, count * n * k * 2);
            for (weights, 0..) |*weight, e| weight.* = .{ .bf16 = values[e * n * k * 2 ..][0 .. n * k * 2] };
        }
        const want_codes = Reader.take(bytes, &at, count * n * k / 2);
        const want_scales = Reader.take(bytes, &at, count * n * k / group);
        const want_experts = Reader.take(bytes, &at, count * 4);

        var quantized: Quantized = try .init(gpa, weights, n, k);
        defer quantized.deinit(gpa);
        try testing.expectEqualSlices(u8, want_experts, std.mem.sliceAsBytes(quantized.scales));
        try testing.expectEqualSlices(u8, want_scales, quantized.block_scales);
        try testing.expectEqualSlices(u8, want_codes, quantized.codes);
        // the fixture reaches the corners: a block with no scale, a subnormal scale, a negative zero
        if (kind == 1) {
            var zero_scales: usize = 0;
            var subnormal: usize = 0;
            for (want_scales) |scale| {
                zero_scales += @intFromBool(scale == 0);
                subnormal += @intFromBool(scale > 0 and scale < 8);
            }
            var negative_zeros: usize = 0;
            for (0..want_codes.len * 2) |i| negative_zeros += @intFromBool(fx.codeAt(want_codes, i) == 8);
            try testing.expect(zero_scales > 0 and subnormal > 0 and negative_zeros > 0);
        }
    }
    try testing.expectEqual(bytes.len, at);
    try testing.expect(kinds[0] > 0 and kinds[1] > 0);
}

test "chosen rows quantize as the matrix of those rows does, whatever the threads" {
    const gpa = testing.allocator;
    const n = 300;
    const k = 128;
    var prng: std.Random.DefaultPrng = .init(12);
    const random = prng.random();
    const bits = try gpa.alloc(u16, n * k);
    defer gpa.free(bits);
    // rows of different magnitudes: the scale must be the chosen rows' largest, not the tensor's
    for (bits, 0..) |*value, i| value.* = numeric.bf16FromF32(random.floatNorm(f32) * @as(f32, @floatFromInt(1 + i / k % 7)));
    const weight: Weight = .{ .bf16 = std.mem.sliceAsBytes(bits) };
    var ids: [97]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 37 + 5) % n);

    const gathered = try gpa.alloc(f32, ids.len * k);
    defer gpa.free(gathered);
    for (ids, 0..) |id, i| weight.row(n, k, id, gathered[i * k ..][0..k]);
    const codes = try gpa.alloc(u8, ids.len * k / 2);
    defer gpa.free(codes);
    const block_scales = try gpa.alloc(u8, ids.len * k / group);
    defer gpa.free(block_scales);
    const scale = quantize(gathered, ids.len, k, codes, block_scales);

    var rows: Rows = try .init(gpa, weight, n, k, &ids);
    defer rows.deinit(gpa);
    try testing.expectEqual(scale, rows.scale);
    try testing.expectEqualSlices(u8, codes, rows.codes);
    try testing.expectEqualSlices(u8, block_scales, rows.block_scales);
    try testing.expectError(error.UnsupportedShape, Rows.init(gpa, weight, n, k, &.{n}));
}
