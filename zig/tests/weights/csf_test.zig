//! `csf.zig`'s tests: an encoder written here from vLLM PR 60646's reference (`_encode`: a row's base its
//! smallest byte, capped at 240; bytes outside base + 0..15 as exceptions) round-trips random scales with
//! bytes out of every window, the refusals, and, with `ZSPARK_FLASH_NEXT_FP4_LOCAL`, a checkpoint's streams
//! decoded and read back scale by scale through a second, independent reader.

const std = @import("std");
const csf = @import("../../src/weights/csf.zig");
const safetensors = @import("../../src/weights/safetensors.zig");
const Shape = csf.Shape;
const Checkpoint = safetensors.Checkpoint;

const Encoded = struct {
    fixed: []u8,
    /// U32 words, little-endian
    exceptions: []u8,
};

fn encode(arena: std.mem.Allocator, scales: []const u8, rows: usize, cols: usize) !Encoded {
    const half = cols / 2;
    const fixed = try arena.alloc(u8, rows * (1 + half));
    var words: std.ArrayList(u8) = .empty;
    for (0..rows) |row| {
        const values = scales[row * cols ..][0..cols];
        const base = @min(std.mem.min(u8, values), csf.max_base);
        const slab = row / csf.slab;
        const r = row % csf.slab;
        const block = fixed[slab * csf.slab * (1 + half) ..];
        block[r] = base;
        const codes = block[csf.slab + r * half ..][0..half];
        @memset(codes, 0);
        for (values, 0..) |value, col| {
            const outlier = value < base or value - base > 15;
            if (outlier) {
                var word: [4]u8 = undefined;
                std.mem.writeInt(u32, &word, @as(u32, @intCast(row * cols + col)) | @as(u32, value) << 24, .little);
                try words.appendSlice(arena, &word);
            } else {
                codes[col / 2] |= (value - base) << @intCast(4 * (col % 2));
            }
        }
    }
    return .{ .fixed = fixed, .exceptions = words.items };
}

/// One scale straight from the streams, by a binary search of the sorted exceptions.
fn scaleAt(shape: Shape, fixed: []const u8, exceptions: []const u8, row: u64, col: u64) u8 {
    const half: usize = @intCast(shape.cols / 2);
    const flat: u32 = @intCast(row * shape.cols + col);
    var low: usize = 0;
    var high: usize = exceptions.len / 4;
    while (low < high) {
        const mid = (low + high) / 2;
        const word = std.mem.readInt(u32, exceptions[mid * 4 ..][0..4], .little);
        if (word & 0xff_ffff == flat) return @intCast(word >> 24);
        if (word & 0xff_ffff < flat) low = mid + 1 else high = mid;
    }
    const block: usize = @intCast((row / csf.slab) * csf.slab * (1 + half));
    const r: usize = @intCast(row % csf.slab);
    const byte = fixed[block + csf.slab + r * half + @as(usize, @intCast(col / 2))];
    return fixed[block + r] + (if (col % 2 == 0) byte & 0xf else byte >> 4);
}

test "decodes what the reference encoder stored, out-of-window bytes included" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(0);
    const random = prng.random();
    // a routed expert's gate and down projections' scale shapes
    for ([_][2]usize{ .{ 640, 160 }, .{ 2560, 40 } }) |dims| {
        const rows, const cols = dims;
        const scales = try arena.alloc(u8, rows * cols);
        for (scales) |*byte| byte.* = random.intRangeLessThan(u8, 0x30, 0x40);
        // bytes outside any row's window: above it, and below a row whose base another byte sets
        for (0..64) |_| scales[random.uintLessThan(usize, scales.len)] = 0x7e;
        scales[3 * cols + 1] = 0x00;
        const stored = try encode(arena, scales, rows, cols);
        try std.testing.expect(stored.exceptions.len / 4 >= 64);
        const shape = Shape.ofFixed(&.{ rows / 16, 16 * (1 + cols / 2) }).?;
        try std.testing.expectEqual(@as(u64, rows), shape.rows);
        try std.testing.expectEqual(@as(u64, cols), shape.cols);
        const out = try arena.alloc(u8, rows * cols);
        try csf.decode(shape, stored.fixed, stored.exceptions, out);
        try std.testing.expectEqualSlices(u8, scales, out);
        for (0..200) |_| {
            const at = random.uintLessThan(usize, scales.len);
            try std.testing.expectEqual(scales[at], scaleAt(shape, stored.fixed, stored.exceptions, at / cols, at % cols));
        }
    }
}

test "a window decodes as the whole cut: each rank's rows of gate, its columns of down, exceptions at the cut's edges" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(4);
    const random = prng.random();
    // gate's scales [640, 160] cut by rows, down's [2560, 40] by columns, as `share_layout.sliceLinear` cuts them
    for ([_][3]usize{ .{ 640, 160, 0 }, .{ 2560, 40, 1 } }) |dims| {
        const rows, const cols, const by_cols = dims;
        const scales = try arena.alloc(u8, rows * cols);
        for (scales) |*byte| byte.* = random.intRangeLessThan(u8, 0x30, 0x40);
        for (0..64) |_| scales[random.uintLessThan(usize, scales.len)] = 0x7e;
        // an exception on each side of the cut, and at the matrix's first and last bytes
        if (by_cols == 1) {
            for (0..rows) |row| if (row % 7 == 0) {
                scales[row * cols + cols / 2 - 1] = 0x01;
                scales[row * cols + cols / 2] = 0x7f;
            };
        } else for (0..cols) |col| {
            scales[(rows / 2 - 1) * cols + col] = 0x02;
            scales[(rows / 2) * cols + col] = 0x7d;
        }
        scales[0] = 0x00;
        scales[scales.len - 1] = 0x7c;
        const stored = try encode(arena, scales, rows, cols);
        const shape: Shape = .{ .rows = rows, .cols = cols };
        for (0..2) |rank| {
            const window: csf.Window = if (by_cols == 1)
                .{ .row = 0, .rows = rows, .col = rank * cols / 2, .cols = cols / 2 }
            else
                .{ .row = rank * rows / 2, .rows = rows / 2, .col = 0, .cols = cols };
            const out = try arena.alloc(u8, @intCast(window.count()));
            try csf.decodeWindow(shape, stored.fixed, stored.exceptions, window, out);
            const w: usize = @intCast(window.cols);
            for (0..@intCast(window.rows)) |r| {
                const from = (@as(usize, @intCast(window.row)) + r) * cols + @as(usize, @intCast(window.col));
                try std.testing.expectEqualSlices(u8, scales[from..][0..w], out[r * w ..][0..w]);
            }
        }
        // an odd column, a window past the matrix, an output of the wrong size
        var one: [2]u8 = undefined;
        try std.testing.expectError(error.InvalidCsf, csf.decodeWindow(shape, stored.fixed, stored.exceptions, .{ .row = 0, .rows = 1, .col = 1, .cols = 2 }, &one));
        try std.testing.expectError(error.InvalidCsf, csf.decodeWindow(shape, stored.fixed, stored.exceptions, .{ .row = rows, .rows = 1, .col = 0, .cols = 2 }, &one));
        try std.testing.expectError(error.InvalidCsf, csf.decodeWindow(shape, stored.fixed, stored.exceptions, .{ .row = 0, .rows = 1, .col = 0, .cols = 4 }, &one));
    }
}

test "refuses a base above 240, an exception past the matrix and streams of the wrong length" {
    const shape: Shape = .{ .rows = 16, .cols = 4 };
    var fixed: [16 * 3]u8 = @splat(0x38);
    var out: [64]u8 = undefined;
    try csf.decode(shape, &fixed, &.{}, &out);
    // every base 0x38, every code byte 0x38: an even column's code 8, an odd one's 3
    for (out, 0..) |byte, i| try std.testing.expectEqual(@as(u8, if (i % 2 == 0) 0x40 else 0x3b), byte);
    fixed[7] = 241;
    try std.testing.expectError(error.InvalidCsf, csf.decode(shape, &fixed, &.{}, &out));
    fixed[7] = 0x38;
    var word: [4]u8 = undefined;
    std.mem.writeInt(u32, &word, 64 | 0x40 << 24, .little);
    try std.testing.expectError(error.InvalidCsf, csf.decode(shape, &fixed, &word, &out));
    std.mem.writeInt(u32, &word, 63 | 0x40 << 24, .little);
    try csf.decode(shape, &fixed, &word, &out);
    try std.testing.expectEqual(@as(u8, 0x40), out[63]);
    try std.testing.expectError(error.InvalidCsf, csf.decode(shape, fixed[1..], &.{}, &out));
    try std.testing.expectError(error.InvalidCsf, csf.decode(shape, &fixed, word[0..3], &out));
    try std.testing.expectError(error.InvalidCsf, csf.decode(.{ .rows = 8, .cols = 4 }, fixed[0..24], &.{}, out[0..32]));
    try std.testing.expect(Shape.ofFixed(&.{ 2, 17 }) == null);
    try std.testing.expectEqual([2]u64{ 40, 1296 }, (Shape{ .rows = 640, .cols = 160 }).fixedDims());
}

test "a checkpoint's csf scales: decoded, read back one by one, e4m3 bytes that are no NaN" {
    const path = std.testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var checkpoint: Checkpoint = try .open(gpa, std.testing.io, std.Io.Dir.cwd(), path);
    defer checkpoint.close();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(105);
    const random = prng.random();
    // the first and last layers, a few experts each, every projection
    var checked: usize = 0;
    for ([_]u32{ 0, 47 }) |layer| for ([_]u32{ 0, 255, 511 }) |expert| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        const base = try std.fmt.allocPrint(arena, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ layer, expert, proj });
        const fixed = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight_scale{s}", .{ base, csf.fixed_suffix })).?;
        const exceptions = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight_scale{s}", .{ base, csf.exceptions_suffix })).?;
        const weight = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight", .{base})).?;
        try std.testing.expect(checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight_scale", .{base})) == null);
        const shape = Shape.ofFixed(fixed.shape).?;
        // a scale every 16 inputs, two codes a byte
        try std.testing.expectEqual(weight.shape[0], shape.rows);
        try std.testing.expectEqual(weight.shape[1] * 2 / 16, shape.cols);
        const out = try arena.alloc(u8, @intCast(shape.count()));
        try csf.decode(shape, fixed.data, exceptions.data, out);
        // the exceptions are sorted, as the read-back's search needs
        var last: i64 = -1;
        var i: usize = 0;
        while (i < exceptions.data.len) : (i += 4) {
            const position: i64 = std.mem.readInt(u32, exceptions.data[i..][0..4], .little) & 0xff_ffff;
            try std.testing.expect(position > last);
            last = position;
        }
        // positive and finite: an amax over 6 is never negative or NaN
        for (out) |byte| try std.testing.expect(byte < 0x7f);
        for (0..512) |_| {
            const at = random.uintLessThan(usize, out.len);
            try std.testing.expectEqual(out[at], scaleAt(shape, fixed.data, exceptions.data, at / shape.cols, at % shape.cols));
        }
        checked += 1;
    };
    try std.testing.expectEqual(@as(usize, 18), checked);
}
