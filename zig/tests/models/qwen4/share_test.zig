//! `share.zig`: two ranks' slices put back together are the tensor, and the published checkpoint's share is
//! 38.835 GiB (72.350 split in half and 2.660 whole).

const std = @import("std");
const testing = std.testing;
const config_file = @import("../../../src/models/qwen4/config.zig");
const share = @import("../../../src/models/qwen4/share.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");

const Tensor = safetensors.Tensor;

fn fill(bytes: []u8, cols: usize) void {
    for (bytes, 0..) |*byte, i| byte.* = @intCast((i / cols * 3 + i % cols * 5) % 251);
}

fn place(into: []u8, whole: Tensor, part: Tensor, group: share.Group, rank: u32, heads: share.Heads) void {
    switch (group.cut()) {
        .whole, .unread => @memcpy(into, part.data),
        .row => {
            const half = whole.data.len / share.world;
            @memcpy(into[rank * half ..][0..half], part.data);
        },
        .col => {
            const outer: usize = @intCast(whole.shape[0]);
            const inner = whole.data.len / outer;
            const half = inner / share.world;
            for (0..outer) |o| @memcpy(into[o * inner + rank * half ..][0..half], part.data[o * half ..][0..half]);
        },
        .qkv => {
            const q: usize = heads.q;
            const v: usize = heads.v;
            const full: usize = @intCast(whole.shape[0]);
            const inner = whole.data.len / full;
            const qh = q / share.world;
            const vh = v / share.world;
            const segments = [_]struct { at: usize, n: usize, src: usize }{
                .{ .at = rank * qh, .n = qh, .src = 0 },
                .{ .at = q + rank * qh, .n = qh, .src = qh },
                .{ .at = 2 * q + rank * vh, .n = vh, .src = 2 * qh },
            };
            for (segments) |one| {
                @memcpy(into[one.at * inner ..][0 .. one.n * inner], part.data[one.src * inner ..][0 .. one.n * inner]);
            }
        },
    }
}

fn expectRoundTrip(arena: std.mem.Allocator, whole: Tensor, group: share.Group, heads: share.Heads) !void {
    if (group.cut() == .whole or group.cut() == .unread) {
        for (0..share.world) |rank| {
            const part = try share.take(arena, whole, group, .of(@intCast(rank)), heads);
            try testing.expectEqual(@intFromPtr(whole.data.ptr), @intFromPtr(part.data.ptr));
        }
        return;
    }
    const back = try arena.alloc(u8, whole.data.len);
    @memset(back, 0);
    for (0..share.world) |rank| {
        const part = try share.take(arena, whole, group, .of(@intCast(rank)), heads);
        place(back, whole, part, group, @intCast(rank), heads);
    }
    try testing.expectEqualSlices(u8, whole.data, back);
}

test "two ranks' rows, columns and DeltaNet heads reassemble to the tensor" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const heads: share.Heads = .{ .q = 4, .v = 8 };
    var rows: [8 * 6]u8 = undefined;
    var cols: [4 * 8]u8 = undefined;
    var qkv: [16 * 3]u8 = undefined;
    fill(&rows, 6);
    fill(&cols, 8);
    fill(&qkv, 3);
    const cases = [_]struct { tensor: Tensor, group: share.Group }{
        .{ .tensor = .{ .dtype = .U8, .shape = &.{ 8, 6 }, .data = &rows }, .group = .head },
        .{ .tensor = .{ .dtype = .U8, .shape = &.{ 4, 8 }, .data = &cols }, .group = .expert_cols },
        .{ .tensor = .{ .dtype = .U8, .shape = &.{ 16, 3 }, .data = &qkv }, .group = .qkv },
    };
    for (cases) |case| try expectRoundTrip(arena, case.tensor, case.group, heads);
    const halved = try share.take(arena, cases[0].tensor, .head, .of(0), heads);
    try testing.expectEqual(@intFromPtr(cases[0].tensor.data.ptr), @intFromPtr(halved.data.ptr));
    try testing.expectEqualSlices(u64, &.{ 4, 6 }, halved.shape);
}

test "a mixed-precision checkpoint: its MXFP8 codes and exponents cut alike, and each half reassembles" {
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    var checkpoint: safetensors.Checkpoint = try .open(testing.allocator, testing.io, .cwd(), path);
    defer checkpoint.close();
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try std.Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    var diagnostic: config_file.Diagnostic = .{};
    const heads = share.Heads.of(try config_file.Config.read(arena, testing.io, dir, &diagnostic));
    // every MXFP8 linear of a DeltaNet layer and an attention layer, each cut its way
    const modules = [_][]const u8{
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj",
        "model.language_model.layers.0.mlp.shared_expert.down_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj",
    };
    for (modules) |module| {
        const weight = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight", .{module})).?;
        const scale = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight_scale", .{module})).?;
        try testing.expectEqual(safetensors.DType.U8, scale.dtype);
        try testing.expectEqual(weight.shape[1] / 32, scale.shape[1]);
        const group = share.groupOf(module);
        try expectRoundTrip(arena, weight, group, heads);
        try expectRoundTrip(arena, scale, group, heads);
        for (0..share.world) |rank| {
            const codes = try share.take(arena, weight, group, .of(@intCast(rank)), heads);
            const exponents = try share.take(arena, scale, group, .of(@intCast(rank)), heads);
            // what `Mx8Linear.init` takes: an exponent a 32 of the rank's inputs, which come in 64s
            try testing.expectEqual(codes.shape[0], exponents.shape[0]);
            try testing.expectEqual(codes.shape[1] / 32, exponents.shape[1]);
            try testing.expect(codes.shape[1] % 64 == 0);
        }
    }
}
