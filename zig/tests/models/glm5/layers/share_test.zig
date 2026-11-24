//! `share.zig`'s tests: each split on tensors whose bytes say where they came from, and a ModelOpt linear of each
//! storage and split cut for both ranks against its whole (`modelopt_fixture.zig`, and FP4-Local's own where
//! `ZSPARK_GLM_FP4_LOCAL` names it).

const std = @import("std");
const testing = std.testing;
const safetensors = @import("../../../../src/weights/safetensors.zig");
const fixtures = @import("../../../../src/models/glm5/checks/fixtures.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const share = @import("../../../../src/models/glm5/layers/share.zig");
const config_file = @import("../../../../src/models/glm5/layers/config.zig");
const modelopt = @import("../../../../src/weights/modelopt.zig");
const modelopt_test = @import("../../../weights/modelopt_test.zig");
const modelopt_fixture = @import("modelopt_fixture.zig");

const Source = fixtures.Source;

test "half of the rows is a slice, half of the columns a copy, a whole itself" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // [4, 6] of two bytes an element: element (r, c) holds r and c
    var bytes: [4 * 6 * 2]u8 = undefined;
    for (0..4) |r| for (0..6) |c| {
        bytes[(r * 6 + c) * 2] = @intCast(r);
        bytes[(r * 6 + c) * 2 + 1] = @intCast(c);
    };
    const matrix: safetensors.Tensor = .{ .dtype = .BF16, .shape = &.{ 4, 6 }, .data = &bytes };
    for (0..layout.world) |rank| {
        const rows = try share.take(arena, matrix, .row, rank);
        try testing.expectEqualSlices(u64, &.{ 2, 6 }, rows.shape);
        try testing.expectEqual(@intFromPtr(&bytes[rank * 24]), @intFromPtr(rows.data.ptr));
        try testing.expectEqual(@as(usize, 24), rows.data.len);
        const vocab = try share.take(arena, matrix, .vocab, rank);
        try testing.expectEqual(@intFromPtr(rows.data.ptr), @intFromPtr(vocab.data.ptr));

        const cols = try share.take(arena, matrix, .col, rank);
        try testing.expectEqualSlices(u64, &.{ 4, 3 }, cols.shape);
        try testing.expectEqual(@as(usize, 24), cols.data.len);
        for (0..4) |r| for (0..3) |c| {
            try testing.expectEqual(@as(u8, @intCast(r)), cols.data[(r * 3 + c) * 2]);
            try testing.expectEqual(@as(u8, @intCast(rank * 3 + c)), cols.data[(r * 3 + c) * 2 + 1]);
        };
        try testing.expectEqual(@intFromPtr(&bytes), @intFromPtr((try share.take(arena, matrix, .whole, rank)).data.ptr));
    }
}

/// `module`'s two ranks' shares against the whole the map stores, by the split its kind has.
fn expectShares(arena: std.mem.Allocator, source: anytype, c: *const config_file.Config, module: []const u8, split: layout.Split) !void {
    var diagnostic: config_file.Diagnostic = .{};
    const kind = layout.kindIn(c.*, try std.fmt.allocPrint(arena, "{s}.weight", .{module})).?;
    try testing.expectEqual(split, kind.split);
    var dims: [3]u64 = undefined;
    const shape = kind.shape(c.*, &dims);
    const whole = try modelopt.read(arena, source, module, c.quant.find(module), shape[0], shape[1], &diagnostic);
    var halves: [2]modelopt.Linear = undefined;
    for (&halves, 0..) |*half, rank| half.* = share.modeloptLinear(arena, source, c, module, rank, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message});
        return err;
    };
    if (split == .whole) {
        for (halves) |half| try testing.expectEqual(whole.weight.data.ptr, half.weight.data.ptr);
        return;
    }
    try modelopt_test.expectHalves(arena, whole, halves, split);
}

test "a ModelOpt linear of each storage cut for both ranks: MXFP8, csf_32x2_v1 and plain NVFP4, BF16, the head" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.parse(arena, modelopt_fixture.config, &diagnostic);
    const checkpoint = try modelopt_fixture.synthetic(arena, c, 3);
    const p = layout.prefix ++ "layers.";
    const cases = [_]struct { []const u8, layout.Split }{
        .{ p ++ "0.self_attn.q_proj", .row },
        .{ p ++ "1.self_attn.o_proj", .col },
        .{ p ++ "3.mlp.experts.2.gate_proj", .row },
        .{ p ++ "3.mlp.experts.2.down_proj", .col },
        .{ p ++ "4.mlp.experts.1.down_proj", .col },
        .{ p ++ "4.mlp.experts.5.up_proj", .row },
        .{ p ++ "4.self_attn.kv_b_proj", .row },
        .{ p ++ "0.self_attn.f_b_proj", .row },
        .{ p ++ "3.self_attn.q_a_proj", .whole },
        .{ "lm_head", .vocab },
    };
    for (cases) |case| expectShares(arena, &checkpoint, &c, case[0], case[1]) catch |err| {
        std.debug.print("{s}: {t}\n", .{ case[0], err });
        return err;
    };
    // 96 inputs do not halve on MXFP8's blocks of 32; a norm is no linear
    try testing.expectError(error.Indivisible, share.modeloptLinear(arena, &checkpoint, &c, p ++ "0.mlp.down_proj", 0, &diagnostic));
    try testing.expectError(error.UnexpectedTensor, share.modeloptLinear(arena, &checkpoint, &c, p ++ "0.input_layernorm", 0, &diagnostic));
}

test "FP4-Local's linears of each storage cut for both ranks, the bytes against the whole as stored" {
    // Runs with ZSPARK_GLM_FP4_LOCAL set to the checkpoint's directory; skipped without. Reads a few linears.
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.read(arena, io, dir, &diagnostic);
    var checkpoint: safetensors.Checkpoint = try .open(testing.allocator, io, .cwd(), path);
    defer checkpoint.close();
    const p = layout.prefix ++ "layers.";
    const cases = [_]struct { []const u8, layout.Split }{
        // MXFP8: KDA's heads and their output, the dense MLP's inputs, the MTP layer's attention output
        .{ p ++ "0.self_attn.q_proj", .row },
        .{ p ++ "0.self_attn.o_proj", .col },
        .{ p ++ "0.mlp.down_proj", .col },
        .{ p ++ "45.self_attn.o_proj", .col },
        .{ p ++ "45.mlp.shared_experts.down_proj", .col },
        // FP8_PB_WO, its scales stored 4-D: MLA's heads and output, a shared expert
        .{ p ++ "3.self_attn.q_b_proj", .row },
        .{ p ++ "3.self_attn.o_proj", .col },
        .{ p ++ "43.mlp.shared_experts.gate_proj", .row },
        .{ p ++ "43.mlp.shared_experts.down_proj", .col },
        .{ p ++ "3.self_attn.kv_a_proj_with_mqa", .whole },
        // NVFP4: csf_32x2_v1 in the model, plain in the MTP layer
        .{ p ++ "3.mlp.experts.0.gate_proj", .row },
        .{ p ++ "44.mlp.experts.287.down_proj", .col },
        .{ p ++ "45.mlp.experts.0.up_proj", .row },
        .{ p ++ "45.mlp.experts.17.down_proj", .col },
        // BF16 (native-QB): KDA's f_b, MLA's kv_b
        .{ p ++ "0.self_attn.f_b_proj", .row },
        .{ p ++ "3.self_attn.kv_b_proj", .row },
        .{ "lm_head", .vocab },
    };
    for (cases) |case| expectShares(arena, &checkpoint, &c, case[0], case[1]) catch |err| {
        std.debug.print("{s}: {t}\n", .{ case[0], err });
        return err;
    };
}
