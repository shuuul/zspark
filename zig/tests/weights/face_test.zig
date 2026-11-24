//! `face.zig`'s tests.

const std = @import("std");
const fixtures = @import("testing.zig");
const cuda = @import("../../src/cuda/cuda.zig");
const face_file = @import("../../src/weights/face.zig");
const numeric = @import("../../src/weights/numeric.zig");
const nvfp4 = @import("../../src/weights/nvfp4.zig");
const Allocator = std.mem.Allocator;
const Face = face_file.Face;
const Part = face_file.Part;
const bf16 = @import("../../src/weights/bf16.zig");

/// A face's parts on the host, random, with the weight [n, k] as the host dequantizes it.
const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    parts: []Part,
    /// [n, k]
    dense: []f64,
    n: usize,
    k: usize,

    const Kind = enum { bf16, fp8_block };
    const Spec = struct { kind: Kind, n: usize };

    fn init(gpa: Allocator, seed: u64, specs: []const Spec, k: usize) !Fixture {
        var made: Fixture = .{ .arena = .init(gpa), .parts = undefined, .dense = undefined, .n = 0, .k = k };
        errdefer made.arena.deinit();
        const arena = made.arena.allocator();
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        for (specs) |spec| made.n += spec.n;
        made.parts = try arena.alloc(Part, specs.len);
        made.dense = try arena.alloc(f64, made.n * k);
        var first: usize = 0;
        for (specs, made.parts) |spec, *part| {
            const dense = made.dense[first * k ..][0 .. spec.n * k];
            switch (spec.kind) {
                .bf16 => {
                    const bits = try arena.alloc(u16, spec.n * k);
                    for (bits, dense) |*b, *d| {
                        b.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.05);
                        d.* = numeric.f32FromBf16(b.*);
                    }
                    part.* = .{ .bf16 = std.mem.sliceAsBytes(bits) };
                },
                .fp8_block => {
                    const codes = try arena.alloc(u8, spec.n * k);
                    for (codes) |*byte| {
                        byte.* = random.int(u8);
                        if (byte.* & 0x7f == 0x7f) byte.* &= 0xfe; // not NaN
                    }
                    // a part's blocks count from its own first row, whatever it is stacked under
                    const inputs = nvfp4.Fp8BlockLinear.Block.of(k).inputs;
                    const scales = try arena.alloc(f32, (spec.n + 127) / 128 * (k / inputs));
                    for (scales) |*value| value.* = 0.0002 + 0.002 * random.float(f32);
                    for (dense, codes, 0..) |*d, code, i| d.* = @as(f64, numeric.f32FromE4m3(code)) * scales[i / k / 128 * (k / inputs) + i % k / inputs];
                    part.* = .{ .fp8_block = .{ .codes = codes, .block_scales = .of(scales), .n = spec.n } };
                },
            }
            first += spec.n;
        }
        return made;
    }

    fn deinit(self: *Fixture) void {
        self.arena.deinit();
    }
};

const fx = struct {
    const Case = struct { specs: []const Fixture.Spec, k: usize };
    // DeltaNet's order (two block-FP8 parts whose blocks do not line up once stacked, then two BF16 rows), three
    // runs with a BF16 part first, and a run that starts off the 16-byte boundary a lone row could be written at
    const cases = [_]Case{
        .{ .specs = &.{ .{ .kind = .fp8_block, .n = 136 }, .{ .kind = .fp8_block, .n = 200 }, .{ .kind = .bf16, .n = 4 }, .{ .kind = .bf16, .n = 4 } }, .k = 256 },
        .{ .specs = &.{ .{ .kind = .bf16, .n = 64 }, .{ .kind = .fp8_block, .n = 520 }, .{ .kind = .bf16, .n = 33 } }, .k = 2560 },
        .{ .specs = &.{ .{ .kind = .bf16, .n = 3 }, .{ .kind = .fp8_block, .n = 130 } }, .k = 256 },
        // a hyper-connection's read-out: `down` with the inject gates' rows under it, and `up`, whose 320 inputs
        // are in blocks of 64
        .{ .specs = &.{ .{ .kind = .fp8_block, .n = 320 }, .{ .kind = .fp8_block, .n = 4 } }, .k = 2560 },
        .{ .specs = &.{.{ .kind = .fp8_block, .n = 1300 }}, .k = 320 },
        // the read-out in BF16, as FP4-Local stores it: one matmul whose K splits, so its slices are summed
        .{ .specs = &.{ .{ .kind = .bf16, .n = 320 }, .{ .kind = .bf16, .n = 4 } }, .k = 2560 },
    };
    // rows through every tile of both matmuls and the lane matmul's fused form
    const rows = [_]usize{ 1, 3, 16, 17, 33, 70, 300 };
    const max_rows = 300;
};

/// `face` over the first `m` rows of `x_bits`, into `into` as `dtype`.
fn through(face: Face, x_bits: []const u16, first: usize, m: usize, dtype: cuda.DType, into: []u8) !void {
    var x: cuda.Buffer = try .init(.bf16, &.{ m, face.k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_bits[first * face.k ..][0 .. m * face.k]));
    var out: cuda.Buffer = try .init(dtype, &.{ m, face.n });
    defer out.deinit();
    try face.forward(x.view, out.view);
    try cuda.synchronize();
    try out.download(into);
}

test "a face mixing block FP8 and BF16 matches the host" {
    const gpa = std.testing.allocator;
    for (fx.cases, 0..) |case, seed| {
        var host: Fixture = try .init(gpa, seed, case.specs, case.k);
        defer host.deinit();
        var scratch: cuda.Scratch = .init(gpa);
        defer scratch.deinit();
        var face: Face = try .initParts(gpa, host.parts, case.k, scratch.rows(fx.max_rows));
        defer face.deinit();
        try std.testing.expectEqual(host.n, face.n);
        try fixtures.linears.checkLinear(gpa, face, host.dense, &fx.rows, .decode);
    }
}

test "a mixed face's window equals its rows as serial steps, and each part's own matmul" {
    const gpa = std.testing.allocator;
    for (fx.cases, 0..) |case, seed| {
        var host: Fixture = try .init(gpa, seed, case.specs, case.k);
        defer host.deinit();
        var scratch: cuda.Scratch = .init(gpa);
        defer scratch.deinit();
        var face: Face = try .initParts(gpa, host.parts, case.k, scratch.rows(fx.max_rows));
        defer face.deinit();
        const n = face.n;
        var prng: std.Random.DefaultPrng = .init(seed + 100);
        const random = prng.random();
        const x_bits = try gpa.alloc(u16, fx.max_rows * case.k);
        defer gpa.free(x_bits);
        for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));

        for ([_]cuda.DType{ .bf16, .f32 }) |dtype| {
            const width = dtype.size();
            const window = try gpa.alloc(u8, fx.max_rows * n * width);
            defer gpa.free(window);
            const got = try gpa.alloc(u8, fx.max_rows * n * width);
            defer gpa.free(got);
            try through(face, x_bits, 0, fx.max_rows, dtype, window);
            // every row alone: a serial step
            for (0..fx.max_rows) |row| {
                try through(face, x_bits, row, 1, dtype, got[0 .. n * width]);
                try std.testing.expectEqualSlices(u8, window[row * n * width ..][0 .. n * width], got[0 .. n * width]);
            }
            // shorter windows, and windows that start elsewhere
            for (fx.rows) |m| for ([_]usize{ 0, fx.max_rows - m }) |first| {
                try through(face, x_bits, first, m, dtype, got[0 .. m * n * width]);
                try std.testing.expectEqualSlices(u8, window[first * n * width ..][0 .. m * n * width], got[0 .. m * n * width]);
            };

            // each run's columns are what its storage's own linear gives for those parts
            var at: usize = 0;
            var first: usize = 0;
            while (first < host.parts.len) {
                var end = first + 1;
                while (end < host.parts.len and std.meta.activeTag(host.parts[end]) == std.meta.activeTag(host.parts[first])) end += 1;
                var alone: Face = try .initParts(gpa, host.parts[first..end], case.k, scratch.rows(fx.max_rows));
                defer alone.deinit();
                try std.testing.expectEqual(@as(usize, 1), alone.count);
                try through(alone, x_bits, 0, fx.max_rows, dtype, got[0 .. fx.max_rows * alone.n * width]);
                for (0..fx.max_rows) |row| {
                    try std.testing.expectEqualSlices(u8, got[row * alone.n * width ..][0 .. alone.n * width], window[(row * n + at) * width ..][0 .. alone.n * width]);
                }
                at += alone.n;
                first = end;
            }
            try std.testing.expectEqual(n, at);
        }
    }
}

test "a face's unsummed slices add up, in slice order, to its fp32 rows" {
    // Spec 028: the kernel after a split matmul sums its slices. Here the host adds the slices as that kernel
    // does: it must give `forward`'s bits.
    const gpa = std.testing.allocator;
    var folded_matmuls: usize = 0;
    for (fx.cases, 0..) |case, seed| {
        var host: Fixture = try .init(gpa, seed, case.specs, case.k);
        defer host.deinit();
        var scratch: cuda.Scratch = .init(gpa);
        defer scratch.deinit();
        var face: Face = try .initParts(gpa, host.parts, case.k, scratch.rows(fx.max_rows));
        defer face.deinit();
        const n = face.n;
        var prng: std.Random.DefaultPrng = .init(seed + 200);
        const random = prng.random();
        const x_bits = try gpa.alloc(u16, fx.max_rows * case.k);
        defer gpa.free(x_bits);
        for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
        var x: cuda.Buffer = try .init(.bf16, &.{ fx.max_rows, case.k });
        defer x.deinit();
        try x.upload(std.mem.sliceAsBytes(x_bits));
        var out: cuda.Buffer = try .init(.f32, &.{ fx.max_rows, n });
        defer out.deinit();
        const want = try gpa.alloc(u32, fx.max_rows * n);
        defer gpa.free(want);
        const got = try gpa.alloc(u32, fx.max_rows * n);
        defer gpa.free(got);

        for (fx.rows) |m| {
            const rows = out.rows(m);
            try face.forward(x.rows(m), rows);
            try cuda.synchronize();
            try cuda.download(rows, std.mem.sliceAsBytes(want[0 .. m * n]));

            {
                try out.zero();
                const slices = try face.slices(x.rows(m), rows);
                try cuda.synchronize();
                const sk: usize = @intCast(slices.view.size[0]);
                try std.testing.expectEqual(@as(i64, @intCast(m)), slices.view.size[1]);
                try std.testing.expectEqual(@as(i64, @intCast(n)), slices.view.size[2]);
                const parts = try gpa.alloc(f32, sk * m * n);
                defer gpa.free(parts);
                try cuda.download(slices.view, std.mem.sliceAsBytes(parts));
                for (got[0 .. m * n], 0..) |*bits, i| {
                    var total: f32 = parts[i];
                    for (1..sk) |s| total += parts[s * m * n + i];
                    bits.* = @bitCast(total);
                }
                try std.testing.expectEqualSlices(u32, want[0 .. m * n], got[0 .. m * n]);
                folded_matmuls += @intFromBool(sk > 1);
            }
        }
    }
    // the fixture's read-outs do split: the fold is exercised, not skipped
    try std.testing.expect(folded_matmuls > 0);
}

test "BF16 parts alone are the stacked BF16 linear" {
    const gpa = std.testing.allocator;
    var host: Fixture = try .init(gpa, 7, &.{ .{ .kind = .bf16, .n = 96 }, .{ .kind = .bf16, .n = 32 }, .{ .kind = .bf16, .n = 2 } }, 256);
    defer host.deinit();
    var scratch: cuda.Scratch = .init(gpa);
    defer scratch.deinit();
    var face: Face = try .initParts(gpa, host.parts, 256, scratch.rows(40));
    defer face.deinit();
    try std.testing.expectEqual(@as(usize, 1), face.count);
    try std.testing.expect(face.staging == null);
    var linear: bf16.Linear = try .initStacked(&.{ host.parts[0].bf16, host.parts[1].bf16, host.parts[2].bf16 }, 256, scratch.rows(40));
    defer linear.deinit();

    var prng: std.Random.DefaultPrng = .init(11);
    const x_bits = try gpa.alloc(u16, 40 * 256);
    defer gpa.free(x_bits);
    for (x_bits) |*bits| bits.* = numeric.bf16FromF32(prng.random().floatNorm(f32));
    var x: cuda.Buffer = try .init(.bf16, &.{ 40, 256 });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_bits));
    var out: cuda.Buffer = try .init(.bf16, &.{ 40, face.n });
    defer out.deinit();
    const want = try gpa.alloc(u16, 40 * face.n);
    defer gpa.free(want);
    const got = try gpa.alloc(u16, 40 * face.n);
    defer gpa.free(got);
    try linear.forward(x.view, out.view);
    try cuda.synchronize();
    try out.download(std.mem.sliceAsBytes(want));
    try out.zero();
    try face.forward(x.view, out.view);
    try cuda.synchronize();
    try out.download(std.mem.sliceAsBytes(got));
    try std.testing.expectEqualSlices(u16, want, got);
}
