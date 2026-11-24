//! `linear.zig`'s tests.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../src/cuda/cuda.zig");
const numeric = @import("../../src/weights/numeric.zig");
const linear_file = @import("../../src/weights/linear.zig");
const Linear = linear_file.Linear;

/// `linear` on the first `rows` rows of `x`, as fp32.
fn run(gpa: std.mem.Allocator, linear: Linear, x: cuda.Buffer, rows: usize) ![]f32 {
    var out: cuda.Buffer = try .init(.f32, &.{ rows, linear.outputs() });
    defer out.deinit();
    try linear.forward(x.rows(rows), out.view);
    try cuda.synchronize();
    const host = try gpa.alloc(f32, rows * linear.outputs());
    errdefer gpa.free(host);
    try out.download(std.mem.sliceAsBytes(host));
    return host;
}

test "a linear in either storage: stacked parts in order, four bits near the stored, a row's bits its own" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const k = 256;
    const sizes = [_]usize{ 96, 160 };
    const max_rows = 5;
    var prng: std.Random.DefaultPrng = .init(4);
    const random = prng.random();
    var parts: [sizes.len][]const u8 = undefined;
    var dense: [sizes.len][]const u16 = undefined;
    for (&parts, &dense, sizes) |*part, *kept, n| {
        const bits = try arena.alloc(u16, n * k);
        for (bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32) * 0.05);
        part.* = std.mem.sliceAsBytes(bits);
        kept.* = bits;
    }
    const x_host = try arena.alloc(u16, max_rows * k);
    for (x_host) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32));
    var x: cuda.Buffer = try .init(.bf16, &.{ max_rows, k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_host));

    var scratch: cuda.Scratch = .init(gpa);
    defer scratch.deinit();
    var stored: Linear = try .init(arena, gpa, &parts, k, .bf16, scratch.rows(max_rows));
    defer stored.deinit();
    var four: Linear = try .init(arena, gpa, &parts, k, .four, scratch.rows(max_rows));
    defer four.deinit();
    try testing.expect(stored == .bf16 and four == .four);
    try testing.expectEqual(@as(usize, 256), stored.outputs());
    try testing.expectEqual(@as(usize, 256), four.outputs());
    // 4.5 bits a weight beside 16
    try testing.expectEqual(stored.bytes() * 9 / 32, four.bytes());

    const want = try run(arena, stored, x, max_rows);
    const got = try run(arena, four, x, max_rows);
    // the stored one's outputs are its parts' rows in order, against the host
    for ([_][2]usize{ .{ 0, 0 }, .{ 2, 95 }, .{ 3, 96 }, .{ 4, 255 } }) |at| {
        const row, const output = at;
        const part: usize = @intFromBool(output >= sizes[0]);
        const within = output - part * sizes[0];
        var sum: f64 = 0;
        for (dense[part][within * k ..][0..k], x_host[row * k ..][0..k]) |w, v| sum += @as(f64, numeric.f32FromBf16(w)) * numeric.f32FromBf16(v);
        try testing.expectApproxEqAbs(sum, @as(f64, want[row * 256 + output]), 2e-3);
    }
    var err: f64 = 0;
    var size: f64 = 0;
    for (want, got) |w, g| {
        err += (@as(f64, w) - g) * (@as(f64, w) - g);
        size += @as(f64, w) * w;
    }
    // E2M1 with a scale a group of 16: about a tenth of a weight's size
    try testing.expect(err > 0 and @sqrt(err / size) < 0.15);

    // a row alone is the row among five
    for ([_]Linear{ stored, four }, [_][]const f32{ want, got }) |linear, among| {
        const alone = try run(arena, linear, x, 1);
        try testing.expectEqualSlices(u32, @ptrCast(among[0..256]), @ptrCast(alone));
    }

    // a stack that is not rows of `k`
    try testing.expectError(error.UnsupportedShape, Linear.init(arena, gpa, &.{parts[0][0 .. k * 2 + 2]}, k, .four, scratch.rows(max_rows)));
}

test "per-tensor FP8 parts as ModelOpt stores them: one stack, the host's dequantized product, a row's bits its own" {
    const modelopt = @import("../../src/weights/modelopt.zig");
    const reader = @import("../../src/weights/checkpoint_reader.zig");
    const linears = @import("testing.zig").linears;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const k = 192;
    const sizes = [_]usize{ 72, 120 };
    const scale: f32 = 0.0123;
    var prng: std.Random.DefaultPrng = .init(15);
    const random = prng.random();
    var parts: [sizes.len]modelopt.Linear = undefined;
    const dense = try arena.alloc(f64, (sizes[0] + sizes[1]) * k);
    var at: usize = 0;
    const scale_bytes = std.mem.asBytes(&scale);
    for (&parts, sizes) |*part, n| {
        const codes = try arena.alloc(u8, n * k);
        for (codes) |*c| {
            c.* = numeric.e4m3FromF32(random.floatNorm(f32) * 40);
            dense[at] = @as(f64, numeric.f32FromE4m3(c.*)) * scale;
            at += 1;
        }
        part.* = .{
            .scheme = .fp8,
            .rows = n,
            .cols = k,
            .weight = .{ .dtype = .F8_E4M3, .shape = &.{}, .data = codes },
            .scale_2 = .{ .dtype = .F32, .shape = &.{}, .data = scale_bytes },
        };
    }
    var scratch: cuda.Scratch = .init(gpa);
    defer scratch.deinit();
    var diagnostic: reader.Diagnostic = .{};
    var stack: Linear = try .stored(arena, gpa, &parts, scratch.rows(1025), &diagnostic);
    defer stack.deinit();
    try testing.expect(stack == .fp8);
    try testing.expectEqual(sizes[0] + sizes[1], stack.outputs());
    try testing.expectEqual(@as(u64, 256 * k), stack.bytes());
    // the lane matmul's tiles and fused slices by row count; the first row's bits the same in each
    try linears.checkLinear(gpa, stack.fp8, dense, &.{ 1, 2, 16, 17, 300, 1025 }, .decode);

    // A fused export may retain independent tensor scales; neither codes nor scales may be requantized.
    const other: f32 = 0.5;
    var mixed = parts;
    mixed[1].scale_2 = .{ .dtype = .F32, .shape = &.{}, .data = std.mem.asBytes(&other) };
    var separate: Linear = try .stored(arena, gpa, &mixed, scratch.rows(1025), &diagnostic);
    defer separate.deinit();
    for (mixed[1].weight.data, 0..) |code, i| dense[sizes[0] * k + i] = @as(f64, numeric.f32FromE4m3(code)) * other;
    try linears.checkLinear(gpa, separate.fp8_block, dense, &.{ 1, 2, 16, 17, 300, 1025 }, .decode);
}
