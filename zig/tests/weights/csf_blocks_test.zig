//! `csf_blocks.zig`'s tests: planes encoded by flashinfer's own `_csf_blocks.py` (`tools/reference/csf_blocks_cases.py`,
//! in the card's container) decoded whole and by windows, the refusals of `validate`, and, with
//! `ZSPARK_GLM_FP4_LOCAL`, the checkpoint's planes against the SHA-256 of the reference's decoding, a rank's
//! windows against the whole and one layer's every plane validated.

const std = @import("std");
const csf_blocks = @import("../../src/weights/csf_blocks.zig");
const safetensors = @import("../../src/weights/safetensors.zig");
const Shape = csf_blocks.Shape;
const Streams = csf_blocks.Streams;
const Window = csf_blocks.Window;

const cases = @embedFile("fixtures/csf_blocks_cases.bin");

pub const Plane = struct {
    shape: Shape,
    logical: []const u8,
    streams: Streams,
};

pub const Stored = struct {
    name: []const u8,
    shape: Shape,
    digest: [32]u8,
};

pub const Fixture = struct {
    planes: []Plane,
    stored: []Stored,
};

const Reader = struct {
    bytes: []const u8,
    at: usize = 0,

    fn take(self: *Reader, n: usize) []const u8 {
        defer self.at += n;
        return self.bytes[self.at..][0..n];
    }

    fn word(self: *Reader) u32 {
        return std.mem.readInt(u32, self.take(4)[0..4], .little);
    }
};

pub fn fixture(arena: std.mem.Allocator) !Fixture {
    var r: Reader = .{ .bytes = cases };
    try std.testing.expectEqualStrings("ZSCSF321", r.take(8));
    const planes = try arena.alloc(Plane, r.word());
    for (planes) |*plane| {
        const rows = r.word();
        const cols = r.word();
        const logical = r.take(rows * cols);
        var counts: [5]u32 = undefined;
        for (&counts) |*n| n.* = r.word();
        plane.* = .{ .shape = .{ .rows = rows, .cols = cols }, .logical = logical, .streams = .{
            .bases = r.take(counts[0]),
            .codes = r.take(counts[1]),
            .page_bases = r.take(4 * counts[2]),
            .offsets = r.take(2 * counts[3]),
            .exceptions = r.take(2 * counts[4]),
        } };
    }
    const stored = try arena.alloc(Stored, r.word());
    for (stored) |*one| {
        const name = r.take(r.word());
        const rows = r.word();
        const cols = r.word();
        one.* = .{ .name = name, .shape = .{ .rows = rows, .cols = cols }, .digest = r.take(32)[0..32].* };
    }
    try std.testing.expectEqual(cases.len, r.at);
    return .{ .planes = planes, .stored = stored };
}

fn expectWindow(arena: std.mem.Allocator, plane: Plane, window: Window) !void {
    const out = try arena.alloc(u8, @intCast(window.count()));
    try csf_blocks.decodeWindow(plane.shape, plane.streams, window, out);
    const cols: usize = @intCast(plane.shape.cols);
    const w: usize = @intCast(window.cols);
    for (0..@intCast(window.rows)) |r| {
        const from = (@as(usize, @intCast(window.row)) + r) * cols + @as(usize, @intCast(window.col));
        try std.testing.expectEqualSlices(u8, plane.logical[from..][0..w], out[r * w ..][0..w]);
    }
}

test "decodes every plane the reference encoded: one column, clipped tiles, pages of 32 tiles" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const found = try fixture(arena);
    try std.testing.expectEqual(@as(usize, 9), found.planes.len);
    var pages: u64 = 0;
    var exceptions: usize = 0;
    for (found.planes) |plane| {
        try std.testing.expectEqual(plane.shape, Shape.ofStreams(plane.streams.bases.len, plane.streams.codes.len).?);
        try csf_blocks.validate(plane.shape, plane.streams);
        const out = try arena.alloc(u8, @intCast(plane.shape.count()));
        try csf_blocks.decode(plane.shape, plane.streams, out);
        try std.testing.expectEqualSlices(u8, plane.logical, out);
        pages = @max(pages, plane.shape.pages());
        exceptions += plane.streams.exceptions.len / 2;
    }
    // the last two span two pages, and the planes hold exceptions to decode
    try std.testing.expectEqual(@as(u64, 2), pages);
    try std.testing.expect(exceptions > 500);
}

test "a window decodes as the whole's rows and columns: each rank's half both ways, and windows at random" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const found = try fixture(arena);
    var prng: std.Random.DefaultPrng = .init(107);
    const random = prng.random();
    for (found.planes) |plane| {
        const rows = plane.shape.rows;
        const cols = plane.shape.cols;
        for (0..2) |rank| {
            try expectWindow(arena, plane, .{ .row = rank * rows / 2, .rows = rows / 2, .col = 0, .cols = cols });
            if (cols >= 2) try expectWindow(arena, plane, .{ .row = 0, .rows = rows, .col = rank * (cols / 2), .cols = cols / 2 });
        }
        for (0..8) |_| {
            const row = random.uintLessThan(u64, rows);
            const col = random.uintLessThan(u64, cols);
            try expectWindow(arena, plane, .{ .row = row, .rows = 1 + random.uintLessThan(u64, rows - row), .col = col, .cols = 1 + random.uintLessThan(u64, cols - col) });
        }
    }
}

test "refuses a base above 240, records out of order or off a block's columns, a page not from 0, wrong lengths" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const found = try fixture(arena);
    // [64, 5]: a block of one column on the right, records in most blocks
    const plane = found.planes[3];
    try std.testing.expectEqual(Shape{ .rows = 64, .cols = 5 }, plane.shape);
    const out = try arena.alloc(u8, @intCast(plane.shape.count()));
    const Part = enum { bases, exceptions, offsets, page_bases };
    const Spoil = struct { part: Part, at: usize, value: u8 };
    var two: usize = 0;
    while (std.mem.readInt(u16, plane.streams.offsets[2 * two + 2 ..][0..2], .little) -% std.mem.readInt(u16, plane.streams.offsets[2 * two ..][0..2], .little) < 2) two += 1;
    const first = std.mem.readInt(u16, plane.streams.offsets[2 * two ..][0..2], .little);
    const spoils = [_]Spoil{
        .{ .part = .bases, .at = 7, .value = 241 },
        // the block's second record at its first's position: not sorted
        .{ .part = .exceptions, .at = 2 * (first + 1), .value = plane.streams.exceptions[2 * first] },
        // a position past a block's 64
        .{ .part = .exceptions, .at = 2 * first, .value = 64 },
        .{ .part = .offsets, .at = 0, .value = 1 },
        .{ .part = .page_bases, .at = 0, .value = 2 },
    };
    for (spoils) |spoil| {
        var streams = plane.streams;
        const slot = switch (spoil.part) {
            .bases => &streams.bases,
            .exceptions => &streams.exceptions,
            .offsets => &streams.offsets,
            .page_bases => &streams.page_bases,
        };
        const copy = try arena.dupe(u8, slot.*);
        copy[spoil.at] = spoil.value;
        slot.* = copy;
        try std.testing.expectError(error.InvalidCsf, csf_blocks.decode(plane.shape, streams, out));
    }
    // a record on the missing column of a block of one: a plane of odd columns whose last block (the bottom
    // right one, one column) has records, which end the arena
    var odd: usize = 0;
    for (found.planes) |each| {
        if (each.shape.cols % 2 == 0 or each.shape.pages() != 1) continue;
        const blocks: usize = @intCast(each.shape.blocks());
        const count = std.mem.readInt(u16, each.streams.offsets[2 * blocks ..][0..2], .little) -
            std.mem.readInt(u16, each.streams.offsets[2 * blocks - 2 ..][0..2], .little);
        if (count == 0) continue;
        const exceptions = try arena.dupe(u8, each.streams.exceptions);
        const last = exceptions.len - 2;
        const record = std.mem.readInt(u16, exceptions[last..][0..2], .little);
        try std.testing.expectEqual(@as(u16, 0), record & 1);
        std.mem.writeInt(u16, exceptions[last..][0..2], record | 1, .little);
        var streams = each.streams;
        streams.exceptions = exceptions;
        const bytes = try arena.alloc(u8, @intCast(each.shape.count()));
        try std.testing.expectError(error.InvalidCsf, csf_blocks.decode(each.shape, streams, bytes));
        odd += 1;
    }
    try std.testing.expect(odd > 0);
    var short = plane.streams;
    short.codes = short.codes[1..];
    try std.testing.expectError(error.InvalidCsf, csf_blocks.decode(plane.shape, short, out));
    short = plane.streams;
    short.exceptions = short.exceptions[2..];
    try std.testing.expectError(error.InvalidCsf, csf_blocks.decode(plane.shape, short, out));
    try std.testing.expectError(error.InvalidCsf, csf_blocks.decodeWindow(plane.shape, plane.streams, .{ .row = 32, .rows = 33, .col = 0, .cols = 1 }, out[0..33]));
    try std.testing.expectError(error.InvalidCsf, csf_blocks.decode(plane.shape, plane.streams, out[1..]));
    try std.testing.expect(Shape.ofStreams(48, 96) == null);
    try std.testing.expectEqual(Shape{ .rows = 2048, .cols = 256 }, Shape.ofStreams(2048, 262144).?);
}

fn streamsOf(checkpoint: *const safetensors.Checkpoint, arena: std.mem.Allocator, module: []const u8) !Streams {
    var streams: Streams = undefined;
    inline for (.{ "bases", "codes", "page_bases", "offsets", "exceptions" }) |part| {
        const name = try std.fmt.allocPrint(arena, "{s}.weight_scale" ++ @field(csf_blocks, part ++ "_suffix"), .{module});
        @field(streams, part) = (checkpoint.get(name) orelse return error.MissingTensor).data;
    }
    return streams;
}

test "a checkpoint's planes: the reference's decoding, a rank's windows, one layer's every plane validated" {
    const path = std.testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var checkpoint: safetensors.Checkpoint = try .open(gpa, std.testing.io, std.Io.Dir.cwd(), path);
    defer checkpoint.close();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const found = try fixture(arena);
    try std.testing.expectEqual(@as(usize, 6), found.stored.len);
    for (found.stored) |stored| {
        const streams = try streamsOf(&checkpoint, arena, stored.name);
        const shape = Shape.ofStreams(streams.bases.len, streams.codes.len).?;
        try std.testing.expectEqual(stored.shape, shape);
        const whole = try arena.alloc(u8, @intCast(shape.count()));
        try csf_blocks.decode(shape, streams, whole);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(whole, &digest, .{});
        try std.testing.expectEqualSlices(u8, &stored.digest, &digest);
        // e4m3 scales of an amax over 6: positive and finite
        for (whole) |byte| try std.testing.expect(byte < 0x7f);
        // a rank's rows of gate and up, its columns of down, as `modelopt.cut` asks them
        const by_cols = std.mem.endsWith(u8, stored.name, "down_proj");
        const plane: Plane = .{ .shape = shape, .logical = whole, .streams = streams };
        for (0..2) |rank| try expectWindow(arena, plane, if (by_cols)
            .{ .row = 0, .rows = shape.rows, .col = rank * shape.cols / 2, .cols = shape.cols / 2 }
        else
            .{ .row = rank * shape.rows / 2, .rows = shape.rows / 2, .col = 0, .cols = shape.cols });
    }
    // layer 3's 288 experts, every projection
    var planes: usize = 0;
    for (0..288) |expert| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        const module = try std.fmt.allocPrint(arena, "model.language_model.layers.3.mlp.experts.{d}.{s}", .{ expert, proj });
        const streams = try streamsOf(&checkpoint, arena, module);
        try csf_blocks.validate(Shape.ofStreams(streams.bases.len, streams.codes.len).?, streams);
        planes += 1;
    };
    try std.testing.expectEqual(@as(usize, 864), planes);
}
