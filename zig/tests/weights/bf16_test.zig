//! `bf16.zig`'s tests.

const std = @import("std");
const fixtures = @import("testing.zig");
const bf16 = @import("../../src/weights/bf16.zig");
const cuda = @import("../../src/cuda/cuda.zig");
const numeric = @import("../../src/weights/numeric.zig");
const Linear = bf16.Linear;
const Tile = bf16.Tile;
const splitK = bf16.splitK;

test "splitK depends on the shape alone" {
    try std.testing.expectEqual(@as(usize, 1), splitK(248320, 2560));
    try std.testing.expectEqual(@as(usize, 1), splitK(10240, 2560));
    try std.testing.expectEqual(@as(usize, 32), splitK(320, 10240));
    try std.testing.expectEqual(@as(usize, 8), splitK(513, 2560));
}

test "BF16 matmul matches the host and keeps a row's bits across row counts" {
    const gpa = std.testing.allocator;
    // one K slice with a ragged last column tile, eight slices, and the hyper-connection's 32
    const shapes = [_][2]usize{ .{ 2000, 128 }, .{ 513, 2560 }, .{ 324, 10240 } };
    for (shapes) |shape| {
        const n, const k = shape;
        var prng: std.Random.DefaultPrng = .init(n);
        const random = prng.random();
        const bits = try gpa.alloc(u16, n * k);
        defer gpa.free(bits);
        const dense = try gpa.alloc(f64, n * k);
        defer gpa.free(dense);
        for (bits, dense) |*b, *d| {
            b.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.05);
            d.* = numeric.f32FromBf16(b.*);
        }
        var scratch: cuda.Scratch = .init(gpa);
        defer scratch.deinit();
        var linear: Linear = try .init(std.mem.sliceAsBytes(bits), n, k, scratch.rows(40));
        defer linear.deinit();
        try fixtures.linears.checkLinear(gpa, linear, dense, &.{ 1, 3, 16, 17, 40 }, .decode);
        try fixtures.linears.checkLinear(gpa, linear, dense, &.{ 1, 3, 16, 17, 40 }, .{ .prompt = 0 });
    }
}

test "the CUTLASS GEMM gives the hand-written matmul's bits at every row count and tile" {
    const gpa = std.testing.allocator;
    const max_rows = 2048;
    // one K slice with a ragged last column tile, then ragged weights in 8, 32, 4 and 2 K slices (the router,
    // the read-out, `out_proj`'s and the shared expert's down's splits): from 1,024 rows the 128-row tiles sum
    // them in the block (spec 095 WS-03; 1,100 a ragged row tile), else b16_reduce, the reference its own
    const shapes = [_][2]usize{ .{ 2000, 128 }, .{ 513, 2560 }, .{ 324, 10240 }, .{ 2600, 6144 }, .{ 2600, 640 } };
    for (shapes) |shape| {
        const n, const k = shape;
        var prng: std.Random.DefaultPrng = .init(n + 7);
        const random = prng.random();
        const bits = try gpa.alloc(u16, n * k);
        defer gpa.free(bits);
        for (bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.05);
        const x_bits = try gpa.alloc(u16, max_rows * k);
        defer gpa.free(x_bits);
        for (x_bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32));
        var scratch: cuda.Scratch = .init(gpa);
        defer scratch.deinit();
        var linear: Linear = try .init(std.mem.sliceAsBytes(bits), n, k, scratch.rows(max_rows));
        defer linear.deinit();

        const want = try gpa.alloc(u32, max_rows * n);
        defer gpa.free(want);
        const got = try gpa.alloc(u32, max_rows * n);
        defer gpa.free(got);
        for ([_]cuda.DType{ .f32, .bf16 }) |dtype| {
            const width: usize = if (dtype == .f32) 4 else 2;
            // the reference: every row through the hand-written kernel
            try run(linear, x_bits, max_rows, dtype, null, std.mem.sliceAsBytes(want)[0 .. max_rows * n * width]);
            for ([_]usize{ 1, 3, 16, 17, 40, 300, 1000, 1100, 2048 }) |m| for ([_]Tile{ .auto, .narrow, .wide, .slim, .deep, .half }) |tile| {
                if (m > 16 and (tile == .slim or tile == .deep)) continue;
                const bytes = std.mem.sliceAsBytes(got)[0 .. m * n * width];
                try run(linear, x_bits, m, dtype, tile, bytes);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want)[0..bytes.len], bytes);
            };
            // a window's slices are its GEMM's sum, one slice, those bits; a round's are the split unsummed
            if (dtype == .f32) try expectSum(linear, x_bits, max_rows, &.{ 1, 16, 17, 1023, 1024, 1100, 2048 }, std.mem.sliceAsBytes(want)[0 .. max_rows * n * 4]);
        }
    }
}

/// `slices` at each of `widths` rows of `x_bits`, summed in slice order on the host, against `want`, the
/// reference's fp32 rows: one slice from `bf16.sums_from` rows, the weight's split under it.
fn expectSum(linear: Linear, x_bits: []const u16, max_rows: usize, widths: []const usize, want: []const u8) !void {
    const gpa = std.testing.allocator;
    var x: cuda.Buffer = try .init(.bf16, &.{ max_rows, linear.k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_bits[0 .. max_rows * linear.k]));
    var out: cuda.Buffer = try .init(.f32, &.{ max_rows, linear.n });
    defer out.deinit();
    const n = linear.n;
    for (widths) |m| {
        try out.zero();
        const slices = try linear.slices(x.rows(m), out.rows(m));
        try cuda.synchronize();
        const sk: usize = @intCast(slices.view.size[0]);
        try std.testing.expectEqual(if (m >= bf16.sums_from) 1 else linear.sk, sk);
        const parts = try gpa.alloc(f32, sk * m * n);
        defer gpa.free(parts);
        try cuda.download(slices.view, std.mem.sliceAsBytes(parts));
        const got = try gpa.alloc(f32, m * n);
        defer gpa.free(got);
        for (got, 0..) |*total, i| {
            total.* = parts[i];
            for (1..sk) |s| total.* += parts[s * m * n + i];
        }
        try std.testing.expectEqualSlices(u8, want[0 .. m * n * 4], std.mem.sliceAsBytes(got));
    }
}

/// The hand-written kernel the CUTLASS GEMM replaced (`zig/tests/kernels/bf16/matmul.cu`): the same K slices and
/// reduction, so the same bits.
fn reference(linear: Linear, x: cuda.View, out: cuda.View) cuda.Error!void {
    try linear.scratch.enter(x);
    const part: ?*const cuda.View = if (linear.part) |*part| &part.view else null;
    return cuda.launched(zs_b16_matmul(&x, &linear.weight.view, &out, part, @intCast(linear.sk)));
}

extern fn zs_b16_matmul(x: *const cuda.View, w: *const cuda.View, out: *const cuda.View, part: ?*const cuda.View, sk: i64) c_int;

/// The first `m` rows of `x_bits` through the hand-written kernel (`tile` null) or the GEMM, into `into`.
fn run(linear: Linear, x_bits: []const u16, m: usize, dtype: cuda.DType, tile: ?Tile, into: []u8) !void {
    var x: cuda.Buffer = try .init(.bf16, &.{ m, linear.k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_bits[0 .. m * linear.k]));
    var out: cuda.Buffer = try .init(dtype, &.{ m, linear.n });
    defer out.deinit();
    if (tile) |t| try linear.gemm(x.view, out.view, t) else try reference(linear, x.view, out.view);
    try cuda.synchronize();
    try out.download(into);
}

test "the entries' GEMM gives each running entry zs_b16_gemm's bits and leaves a skipped entry's region as it was" {
    const gpa = std.testing.allocator;
    // the pool's indexer GEMM (spec 144 port 5): K the indexer's 128, N a bucket's blocks with a ragged last
    // column tile; x and out regions at fixed strides wider than an entry's rows, as `Gem`'s
    const k = 128;
    const n = 1000;
    const held = 48;
    const count = 3;
    const out_stride = held * n + 4;
    var prng: std.Random.DefaultPrng = .init(144);
    const random = prng.random();
    const x_bits = try gpa.alloc(u16, count * held * k);
    defer gpa.free(x_bits);
    for (x_bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32));
    var x: cuda.Buffer = try .init(.bf16, &.{ count * held, k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_bits));
    const w_bits = try gpa.alloc(u16, count * n * k);
    defer gpa.free(w_bits);
    for (w_bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.05);
    // the entries' weights in one buffer, each its own view as a slot's pooled keys are
    var w: cuda.Buffer = try .init(.bf16, &.{ count * n, k });
    defer w.deinit();
    try w.upload(std.mem.sliceAsBytes(w_bits));
    var views: [count]cuda.View = undefined;
    for (&views, 0..) |*view, z| view.* = w.span(z * n, n);
    var out: cuda.Buffer = try .init(.f32, &.{ count, out_stride });
    defer out.deinit();
    var one: cuda.Buffer = try .init(.f32, &.{ held, n });
    defer one.deinit();
    var live: cuda.Buffer = try .init(.i32, &.{ count, 4 });
    defer live.deinit();
    const sentinel: u32 = 0x7fc0_0abc;
    const got = try gpa.alloc(u32, count * out_stride);
    defer gpa.free(got);
    const want = try gpa.alloc(u32, held * n);
    defer gpa.free(want);
    // the narrow tile and the wide one, as zs_b16_gemm's tile 0 picks them by the rows
    for ([_]usize{ 12, 16, 17, 40 }) |m| {
        // the middle entry empty, each count in column 1 as the round's `seg` holds it
        const counts = [count * 4]i32{ 0, @intCast(m), 0, 0, 0, 0, 0, 0, 0, @intCast(m), 0, 0 };
        try live.upload(std.mem.sliceAsBytes(&counts));
        @memset(got, sentinel);
        try out.upload(std.mem.sliceAsBytes(got));
        try cuda.launched(cuda.kernels.zs_b16_gemm_entries(&x.view, @intCast(m), held * k, &views, count, &out.view, out_stride, &live.view, 1));
        try cuda.synchronize();
        try out.download(std.mem.sliceAsBytes(got));
        for (0..count) |z| {
            const region = got[z * out_stride ..][0 .. m * n];
            if (z == 1) {
                for (region) |bits| try std.testing.expectEqual(sentinel, bits);
                continue;
            }
            const rows = x.span(z * held, m);
            const product = one.rows(m);
            try cuda.launched(cuda.kernels.zs_b16_gemm(&rows, &views[z], &product, null, 1, 0));
            try cuda.synchronize();
            try cuda.download(product, std.mem.sliceAsBytes(want[0 .. m * n]));
            try std.testing.expectEqualSlices(u32, want[0 .. m * n], region);
        }
    }
}
