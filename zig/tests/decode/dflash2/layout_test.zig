//! `layout.zig`'s tests: the split of each tensor as `dflash2.py:Drafter.__init__` slices it, the shapes a
//! config makes, and the drafter's own header where `ZSPARK_GLM_DRAFTER` names its directory.

const std = @import("std");
const testing = std.testing;
const safetensors = @import("../../../src/weights/safetensors.zig");
const config_file = @import("../../../src/decode/dflash2/config.zig");
const layout_file = @import("../../../src/decode/dflash2/layout.zig");
const drafter_testing = @import("../../../src/models/glm5/checks/random_drafter.zig");
const Split = layout_file.Split;
const Storage = layout_file.Storage;
const rule = layout_file.rule;

test "a rank's slice of each tensor is Drafter.__init__'s, name by name" {
    const cases = [_]struct { []const u8, ?Split }{
        // `[r * H * hd:(r + 1) * H * hd]`, `[r * KV * hd:...]`, `[r * inter:...]`: rows
        .{ "layers.0.self_attn.q_proj.weight", .row },
        .{ "layers.4.self_attn.k_proj.weight", .row },
        .{ "layers.4.self_attn.v_proj.weight", .row },
        .{ "layers.2.mlp.gate_proj.weight", .row },
        .{ "layers.2.mlp.up_proj.weight", .row },
        // `[:, r * H * hd:...]`, `[:, r * inter:...]`: columns
        .{ "layers.3.self_attn.o_proj.weight", .col },
        .{ "layers.3.mlp.down_proj.weight", .col },
        // read whole by both ranks
        .{ "fc.weight", .whole },
        .{ "hidden_norm.weight", .whole },
        .{ "norm.weight", .whole },
        .{ "candidate_selector.hidden_projection.weight", .whole },
        .{ "candidate_selector.predecessor_codebook", .whole },
        .{ "candidate_selector.successor_codebook", .whole },
        .{ "layers.1.input_layernorm.weight", .whole },
        .{ "layers.1.post_attention_layernorm.weight", .whole },
        .{ "layers.1.attention_conv.base_kernel", .whole },
        .{ "layers.1.attention_conv.kernel_projection.weight", .whole },
        .{ "layers.1.mlp_conv.base_kernel", .whole },
        .{ "layers.1.mlp_conv.kernel_projection.weight", .whole },
        .{ "layers.1.self_attn.q_norm.weight", .whole },
        .{ "layers.1.self_attn.k_norm.weight", .whole },
        // not the drafter's
        .{ "layers.1.fc.weight", null },
        .{ "self_attn.q_proj.weight", null },
        .{ "layers.x.self_attn.q_proj.weight", null },
        .{ "layers.1.self_attn.rotary", null },
        .{ "lm_head.weight", null },
    };
    for (cases) |case| try testing.expectEqual(case[1], rule(case[0]));
    try testing.expectEqual(@as(usize, 21), layout_file.kinds.len);
}

test "what TensorFold holds in four bits, and what zspark keeps as stored of it" {
    // `_quantize4` in `Drafter.__init__`: fc, both kernel projections, qkv and kv, o, gate and up, down
    const four = [_][]const u8{
        "fc.weight",                        "layers.0.attention_conv.kernel_projection.weight", "layers.0.mlp_conv.kernel_projection.weight",
        "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",                 "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight", "layers.0.mlp.gate_proj.weight",                    "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    };
    var counted: usize = 0;
    for (layout_file.kinds) |kind| {
        if (kind.tensorfold == .four) counted += 1;
        // zspark quantizes nothing TensorFold leaves, and the codebooks are the host's in both
        if (kind.storage == .four) try testing.expectEqual(Storage.four, kind.tensorfold);
        try testing.expectEqual(kind.tensorfold == .host, kind.storage == .host);
    }
    try testing.expectEqual(four.len, counted);
    for (four) |name| try testing.expectEqual(Storage.four, layout_file.kinds[layout_file.find(name).?].tensorfold);
    const kept = [_][]const u8{ "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight", "layers.0.attention_conv.kernel_projection.weight", "layers.0.mlp_conv.kernel_projection.weight" };
    for (kept) |name| try testing.expectEqual(Storage.bf16, layout_file.kinds[layout_file.find(name).?].storage);
    // 4.5 bits a weight of 16; fp32 of bf16
    try testing.expectEqual(@as(u64, 90), Storage.four.bytes(320));
    try testing.expectEqual(@as(u64, 640), Storage.host.bytes(320));
}

/// A drafter's file of headers alone, every tensor `c` asks for with zeros behind it.
pub fn synthetic(arena: std.mem.Allocator, c: config_file.Config) !safetensors.Checkpoint {
    var checkpoint: safetensors.Checkpoint = .{ .shards = &.{}, .tensors = .empty, .gpa = arena };
    for (layout_file.kinds) |kind| {
        var dims: [3]u64 = undefined;
        var tensor: safetensors.Tensor = .{ .dtype = .BF16, .shape = try arena.dupe(u64, kind.shape(c, &dims)), .data = &.{} };
        tensor.data = try arena.alloc(u8, tensor.count() * 2);
        if (!kind.layer) {
            try checkpoint.tensors.put(arena, kind.leaf, tensor);
        } else for (0..c.layers) |layer| {
            try checkpoint.tensors.put(arena, try std.fmt.allocPrint(arena, "layers.{d}.{s}", .{ layer, kind.leaf }), tensor);
        }
    }
    return checkpoint;
}

test "a layout finds a config's tensors, and says which is missing, misshapen or not its own" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.parse(arena, drafter_testing.fixture, &diagnostic);
    var checkpoint = try synthetic(arena, c);
    const layout = try layout_file.Layout.init(arena, &c, &checkpoint, &diagnostic);
    // six of the drafter and fifteen a layer
    try testing.expectEqual(@as(usize, 6 + 2 * 15), layout.tensors);
    const q = layout.found[layout_file.find("layers.0.self_attn.q_proj.weight").?];
    try testing.expectEqual(@as(u64, 2 * 512 * 512 * 2), q.bytes);
    try testing.expectEqual(@as(u64, 2 * 256 * 512 * 2), q.rank);
    try testing.expect(layout.device(true, false) < layout.device(false, false));
    try testing.expect(layout.device(false, false) < layout.device(false, true));
    try testing.expectEqual(@as(u64, 2 * 128 * 64 * 4), layout.host());

    const name = "layers.1.mlp_conv.base_kernel";
    const held = checkpoint.tensors.get(name).?;
    _ = checkpoint.tensors.swapRemove(name);
    try testing.expectError(error.MissingTensor, layout_file.Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, name) != null);
    var wrong = held;
    wrong.shape = &.{ 2, 512, 2 };
    try checkpoint.tensors.put(arena, name, wrong);
    try testing.expectError(error.UnexpectedTensor, layout_file.Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, "BF16 { 2, 2, 512 }") != null);
    try checkpoint.tensors.put(arena, name, held);
    try checkpoint.tensors.put(arena, "layers.1.self_attn.bias", held);
    try testing.expectError(error.UnknownTensor, layout_file.Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, "layers.1.self_attn.bias") != null);
}

test "the DFlash2 drafter's header holds the 81 tensors its config asks for, and a rank 1162 MiB of them" {
    // Runs with ZSPARK_GLM_DRAFTER set to the drafter's directory; skipped without. The header only: no tensor is read.
    const path = testing.environ.getPosix("ZSPARK_GLM_DRAFTER") orelse return error.SkipZigTest;
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var diagnostic: config_file.Diagnostic = .{};
    const c = config_file.Config.read(arena.allocator(), io, dir, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message});
        return err;
    };
    // revision bf582e4e
    try testing.expectEqual(@as(usize, 4096), c.hidden);
    try testing.expectEqual(@as(usize, 154880), c.vocab);
    try testing.expectEqual(@as(usize, 5), c.layers);
    try testing.expectEqual(@as(usize, 16), c.rankHeads());
    try testing.expectEqual(@as(usize, 4), c.rankKvHeads());
    try testing.expectEqual(@as(usize, 6144), c.rankWidth());
    try testing.expectEqual(@as(?usize, 2047), c.window);
    try testing.expectEqual(@as(usize, 8), c.block);
    try testing.expectEqual(@as(u32, 154856), c.mask);
    try testing.expectEqualSlices(u32, &.{ 5, 14, 24, 33, 42 }, c.taps);
    try testing.expectEqual(@as(usize, 2176), c.ringRows());
    var checkpoint: safetensors.Checkpoint = try .open(testing.allocator, io, .cwd(), path);
    defer checkpoint.close();
    const layout = layout_file.Layout.init(arena.allocator(), &c, &checkpoint, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message});
        return err;
    };
    try testing.expectEqual(@as(usize, 81), layout.tensors);
    const mib: f64 = 1024.0 * 1024.0;
    try testing.expectApproxEqAbs(@as(f64, 2233.7), @as(f64, @floatFromInt(layout.bytes)) / mib, 0.1);
    try testing.expectApproxEqAbs(@as(f64, 1162.4), @as(f64, @floatFromInt(layout.device(false, true))) / mib, 0.1);
    try testing.expectApproxEqAbs(@as(f64, 328.7), @as(f64, @floatFromInt(layout.device(true, false))) / mib, 0.1);
    try testing.expectApproxEqAbs(@as(f64, 414.9), @as(f64, @floatFromInt(layout.device(false, false))) / mib, 0.1);
}
