//! `layout.zig`'s tests: the split rule against `split.py`'s lists name by name, the shapes a config makes, a
//! ModelOpt checkpoint's linears by the map (`modelopt_fixture.zig`), and FP4-Local's own headers where
//! `ZSPARK_GLM_FP4_LOCAL` names it.

const std = @import("std");
const testing = std.testing;
const safetensors = @import("../../../../src/weights/safetensors.zig");
const config_file = @import("../../../../src/models/glm5/layers/config.zig");
const fixtures = @import("../../../../src/models/glm5/checks/fixtures.zig");
const layout_file = @import("../../../../src/models/glm5/layers/layout.zig");
const modelopt_fixture = @import("modelopt_fixture.zig");
const Split = layout_file.Split;
const rule = layout_file.rule;

test "the split rule gives what split.py's ROW, COL and REP lists give, name by name" {
    const p = layout_file.prefix;
    const cases = [_]struct { []const u8, ?Split }{
        // ROW
        .{ p ++ "layers.3.mlp.shared_experts.up_proj.weight", .row },
        .{ p ++ "layers.0.mlp.gate_proj.weight", .row },
        .{ p ++ "layers.44.self_attn.k_proj.weight", .row },
        .{ p ++ "layers.44.self_attn.v_conv1d.weight", .row },
        .{ p ++ "layers.1.self_attn.f_b_proj.weight", .row },
        .{ p ++ "layers.1.self_attn.b_proj.weight", .row },
        .{ p ++ "layers.1.self_attn.A_log", .row },
        .{ p ++ "layers.1.self_attn.dt_bias", .row },
        .{ p ++ "layers.3.self_attn.q_b_proj.weight", .row },
        .{ p ++ "layers.45.self_attn.kv_b_proj.weight", .row },
        // COL
        .{ p ++ "layers.3.mlp.shared_experts.down_proj.weight", .col },
        .{ p ++ "layers.2.mlp.down_proj.weight", .col },
        .{ p ++ "layers.2.self_attn.o_proj.weight", .col },
        .{ p ++ "layers.3.self_attn.o_proj.weight", .col },
        // REP
        .{ p ++ "embed_tokens.weight", .whole },
        .{ p ++ "norm.weight", .whole },
        .{ p ++ "layers.3.self_attn.q_a_layernorm.weight", .whole },
        .{ p ++ "layers.9.post_attention_layernorm.weight", .whole },
        .{ p ++ "layers.9.hc_attn_fn", .whole },
        .{ p ++ "layers.9.hc_ffn_scale", .whole },
        .{ p ++ "layers.9.mlp.gate.weight", .whole },
        .{ p ++ "layers.9.mlp.gate.e_score_correction_bias", .whole },
        .{ p ++ "layers.3.self_attn.indexer.wq_b.weight", .whole },
        .{ p ++ "layers.3.self_attn.indexer.k_norm.bias", .whole },
        .{ p ++ "layers.3.self_attn.q_a_proj.weight", .whole },
        .{ p ++ "layers.3.self_attn.kv_a_proj_with_mqa.weight", .whole },
        .{ p ++ "layers.0.self_attn.g_a_proj.weight", .whole },
        .{ p ++ "layers.0.self_attn.o_norm.weight", .whole },
        .{ p ++ "layers.45.eh_proj.weight", .whole },
        .{ p ++ "layers.45.hnorm.weight", .whole },
        .{ p ++ "layers.45.shared_head.norm.weight", .whole },
        // the head: replicated by the rule, halved by the loader
        .{ "lm_head.weight", .vocab },
        .{ "model.visual.blocks.0.attn.qkv.weight", .drop },
        // a ModelOpt linear's parts have its weight's split (FP4-Local's experts, its MXFP8 and block FP8);
        // the activations' scales are not read
        .{ p ++ "layers.3.mlp.experts.0.gate_proj.weight", .row },
        .{ p ++ "layers.3.mlp.experts.0.gate_proj.weight_scale.csf32_codes", .row },
        .{ p ++ "layers.3.mlp.experts.0.down_proj.weight_scale.csf32_exceptions", .col },
        .{ p ++ "layers.45.mlp.experts.0.down_proj.weight_scale", .col },
        .{ p ++ "layers.45.mlp.experts.0.down_proj.weight_scale_2", .col },
        .{ p ++ "layers.3.mlp.experts.0.up_proj.input_scale", .drop },
        .{ p ++ "layers.0.self_attn.o_proj.weight_scale", .col },
        .{ p ++ "layers.3.self_attn.q_a_proj.weight_scale", .whole },
        .{ "lm_head.weight_scale", .vocab },
        // no rule: TensorFold raises
        .{ p ++ "layers.3.self_attn.rotary.weight", null },
        .{ p ++ "layers.3.input_layernorm.weight_scale", null },
        .{ p ++ "layers.3.mlp.experts.0.gate_proj.weight_scale.csf32_more", null },
        .{ p ++ "layers.3.mlp.experts.0.self_attn.q_proj.weight", null },
    };
    for (cases) |case| try testing.expectEqual(case[1], rule(case[0]));
    // every kind of the table answers for itself
    for (layout_file.kinds) |kind| {
        var name: [160]u8 = undefined;
        const full = switch (kind.where) {
            .top => kind.leaf,
            .expert => try std.fmt.bufPrint(&name, p ++ "layers.5.mlp.experts.17.{s}", .{kind.leaf}),
            else => try std.fmt.bufPrint(&name, p ++ "layers.5.{s}", .{kind.leaf}),
        };
        try testing.expectEqual(@as(?Split, kind.split), rule(full));
    }
}

test "a rank's share of a tensor's bytes, and the shapes a config makes" {
    try testing.expectEqual(@as(u64, 50), Split.row.share(100));
    try testing.expectEqual(@as(u64, 50), Split.col.share(100));
    try testing.expectEqual(@as(u64, 50), Split.vocab.share(100));
    try testing.expectEqual(@as(u64, 100), Split.whole.share(100));
    try testing.expectEqual(@as(u64, 0), Split.drop.share(100));

    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.parse(arena.allocator(), fixtures.config, &diagnostic);
    var dims: [3]u64 = undefined;
    for (layout_file.kinds) |kind| {
        const expected: []const u64 = if (std.mem.eql(u8, kind.leaf, "hc_attn_fn")) &.{ 24, 256 } else if (std.mem.eql(u8, kind.leaf, "self_attn.kv_b_proj.weight")) &.{ 128, 512 } else if (std.mem.eql(u8, kind.leaf, "self_attn.q_conv1d.weight")) &.{ 512, 1, 4 } else if (std.mem.eql(u8, kind.leaf, "self_attn.indexer.wq_b.weight")) &.{ 32, 24 } else if (std.mem.eql(u8, kind.leaf, "eh_proj.weight")) &.{ 64, 128 } else continue;
        try testing.expectEqualSlices(u64, expected, kind.shape(c, &dims));
    }
    // which layers hold what: the fourth is DSA with a MoE, and the MTP's (4) is as it but has no sites
    const holds = layout_file.Layout.holds;
    try testing.expect(holds(c, .kda, 0) and !holds(c, .kda, 3) and !holds(c, .kda, 4));
    try testing.expect(!holds(c, .dsa, 0) and holds(c, .dsa, 3) and holds(c, .dsa, 4));
    try testing.expect(holds(c, .dense, 2) and !holds(c, .dense, 3) and !holds(c, .dense, 4));
    try testing.expect(holds(c, .expert, 3) and holds(c, .moe, 4) and !holds(c, .moe, 0));
    try testing.expect(holds(c, .site, 3) and !holds(c, .site, 4) and holds(c, .every, 4));
    try testing.expect(holds(c, .mtp, 4) and !holds(c, .mtp, 3));
}

test "a whole checkpoint is taken; one tensor missing, misshapen or unplaced is refused by its name" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.parse(arena, modelopt_fixture.config, &diagnostic);
    var checkpoint = try modelopt_fixture.synthetic(arena, c, 2);
    const Layout = layout_file.Layout;

    const whole = try Layout.init(arena, &c, &checkpoint, &diagnostic);
    try testing.expectEqual(checkpoint.tensors.count(), whole.tensors);
    try testing.expectEqual(@as(usize, 0), whole.dropped);
    // a rank holds its halves and every replicated tensor: more than half, less than all
    try testing.expect(whole.rank * 2 > whole.bytes and whole.rank < whole.bytes);
    var counted: u64 = 0;
    for (whole.found) |found| counted += found.count;
    try testing.expectEqual(@as(u64, whole.tensors), counted);

    // the vision tower is counted and not read
    try checkpoint.tensors.put(arena, "model.visual.blocks.0.attn.qkv.weight", .{ .dtype = .BF16, .shape = &.{ 2, 3 }, .data = &@as([12]u8, @splat(0)) });
    const seen = try Layout.init(arena, &c, &checkpoint, &diagnostic);
    try testing.expectEqual(@as(usize, 1), seen.dropped);
    try testing.expectEqual(@as(u64, 12), seen.dropped_bytes);
    try testing.expectEqual(whole.rank, seen.rank);

    // a language-model tensor the table does not name
    const stray = layout_file.prefix ++ "layers.3.self_attn.rotary.weight";
    try checkpoint.tensors.put(arena, stray, .{ .dtype = .BF16, .shape = &.{2}, .data = &@as([4]u8, @splat(0)) });
    try testing.expectError(error.UnknownTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, stray) != null);
    try testing.expect(checkpoint.tensors.swapRemove(stray));

    // a DSA layer's tensor under a KDA layer: the rule knows the name, the config asks for none there
    const misplaced = layout_file.prefix ++ "layers.0.self_attn.q_a_proj.weight";
    try checkpoint.tensors.put(arena, misplaced, .{ .dtype = .BF16, .shape = &.{2}, .data = &@as([4]u8, @splat(0)) });
    try testing.expectError(error.UnknownTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(checkpoint.tensors.swapRemove(misplaced));

    // the wrong shape, then the wrong dtype
    const name = layout_file.prefix ++ "layers.3.self_attn.kv_b_proj.weight";
    const kept = checkpoint.tensors.get(name).?;
    try checkpoint.tensors.put(arena, name, .{ .dtype = .BF16, .shape = &.{ 8, 128 }, .data = kept.data });
    try testing.expectError(error.UnexpectedTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, name) != null);
    try checkpoint.tensors.put(arena, name, .{ .dtype = .F32, .shape = kept.shape, .data = kept.data });
    try testing.expectError(error.UnexpectedTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try checkpoint.tensors.put(arena, name, kept);

    // missing: an expert's part, and the MTP layer's own
    for ([_][]const u8{ layout_file.prefix ++ "layers.3.mlp.experts.5.down_proj.weight", layout_file.prefix ++ "layers.4.eh_proj.weight", "lm_head.weight" }) |gone| {
        const was = checkpoint.tensors.get(gone).?;
        try testing.expect(checkpoint.tensors.swapRemove(gone));
        try testing.expectError(error.MissingTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
        try testing.expect(std.mem.indexOf(u8, diagnostic.message, gone) != null);
        try checkpoint.tensors.put(arena, gone, was);
    }
    _ = try Layout.init(arena, &c, &checkpoint, &diagnostic);
}

test "a name's kind in a config: one leaf of KDA's and MLA's layers, or of the model's and the MTP's, by its layer and the map" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.parse(arena.allocator(), modelopt_fixture.config, &diagnostic);
    const kindIn = layout_file.kindIn;
    const p = layout_file.prefix;
    try testing.expectEqual(layout_file.Where.kda, kindIn(c, p ++ "layers.0.self_attn.o_proj.weight").?.where);
    try testing.expectEqual(layout_file.Where.dsa, kindIn(c, p ++ "layers.3.self_attn.o_proj.weight").?.where);
    // the MTP layer's attention is MLA's
    try testing.expectEqual(layout_file.Where.dsa, kindIn(c, p ++ "layers.4.self_attn.o_proj.weight").?.where);
    try testing.expectEqual(safetensors.DType.F32, kindIn(c, p ++ "norm.weight").?.dtype);
    try testing.expect(kindIn(c, p ++ "layers.4.mlp.experts.5.down_proj.weight").?.linear);
    // no such layer, no KDA tensor in an MLA layer, no trellis
    try testing.expect(kindIn(c, p ++ "layers.9.self_attn.o_proj.weight") == null);
    try testing.expect(kindIn(c, p ++ "layers.3.self_attn.q_proj.weight") == null);
    try testing.expect(kindIn(c, p ++ "layers.3.mlp.experts.0.down_proj.trellis") == null);
    try testing.expect(layout_file.kindOf(p ++ "layers.3.self_attn.q_proj.weight") != null);
}

test "a ModelOpt checkpoint's linears as its map stores them: every entry met, a part missing or misstored refused" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.parse(arena, modelopt_fixture.config, &diagnostic);
    var checkpoint = try modelopt_fixture.synthetic(arena, c, 1);
    const Layout = layout_file.Layout;

    const whole = Layout.init(arena, &c, &checkpoint, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message});
        return err;
    };
    try testing.expectEqual(checkpoint.tensors.count(), whole.tensors);
    try testing.expectEqual(@as(usize, modelopt_fixture.mxfp8), whole.schemes.get(.mxfp8));
    try testing.expectEqual(@as(usize, modelopt_fixture.nvfp4), whole.schemes.get(.nvfp4));
    try testing.expectEqual(@as(usize, modelopt_fixture.bf16), whole.schemes.get(.bf16));
    try testing.expectEqual(@as(usize, 0), whole.schemes.get(.fp8_block));
    try testing.expectEqual(@as(usize, modelopt_fixture.csf32), whole.csf32);
    try testing.expectEqual(@as(usize, modelopt_fixture.mapped), whole.mapped);
    try testing.expect(whole.rank * 2 > whole.bytes and whole.rank < whole.bytes);
    // every tensor has its rule
    var it = checkpoint.tensors.iterator();
    while (it.next()) |entry| try testing.expect(rule(entry.key_ptr.*) != null);

    // a stream missing, a scale of the wrong dtype, an exception stream of the wrong width
    const expert = layout_file.prefix ++ "layers.3.mlp.experts.4.down_proj.weight_scale";
    const gone = expert ++ ".csf32_offsets";
    const was = checkpoint.tensors.get(gone).?;
    try testing.expect(checkpoint.tensors.swapRemove(gone));
    try testing.expectError(error.MissingTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, gone) != null);
    try checkpoint.tensors.put(arena, gone, was);
    const scale_name = layout_file.prefix ++ "layers.1.self_attn.o_proj.weight_scale";
    const scale = checkpoint.tensors.get(scale_name).?;
    try checkpoint.tensors.put(arena, scale_name, .{ .dtype = .F32, .shape = scale.shape, .data = scale.data });
    try testing.expectError(error.UnexpectedTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, scale_name) != null);
    try checkpoint.tensors.put(arena, scale_name, scale);
    const exceptions_name = expert ++ ".csf32_exceptions";
    const exceptions = checkpoint.tensors.get(exceptions_name).?;
    try checkpoint.tensors.put(arena, exceptions_name, .{ .dtype = .U32, .shape = exceptions.shape, .data = exceptions.data });
    try testing.expectError(error.UnexpectedTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try checkpoint.tensors.put(arena, exceptions_name, exceptions);
    _ = try Layout.init(arena, &c, &checkpoint, &diagnostic);

    // a map entry no linear is, and one storing a linear per-tensor FP8
    const stray = try std.mem.replaceOwned(u8, arena, modelopt_fixture.config, "\"lm_head\":", "\"model.language_model.layers.3.mlp.gate_proj\":{\"quant_algo\":\"MXFP8\",\"group_size\":32},\"lm_head\":");
    const strayed = try config_file.Config.parse(arena, stray, &diagnostic);
    try testing.expectError(error.UnknownTensor, Layout.init(arena, &strayed, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, "layers.3.mlp.gate_proj") != null);
}

test "FP4-Local's headers: every tensor in GLM's layout by the map, every entry of the language model met" {
    // Runs with ZSPARK_GLM_FP4_LOCAL set to the checkpoint's directory; skipped without. Headers only.
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
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
    try testing.expectEqual(@as(usize, 494), c.quant.layers.len);
    var checkpoint: safetensors.Checkpoint = try .open(testing.allocator, io, .cwd(), path);
    defer checkpoint.close();
    // 40 shards, the input scales' and native-QB's under `tensors/`
    try testing.expectEqual(@as(usize, 42), checkpoint.shards.len);
    try testing.expectEqual(@as(usize, 295_041), checkpoint.tensors.count());
    const layout = layout_file.Layout.init(arena.allocator(), &c, &checkpoint, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message});
        return err;
    };
    // snapshot 76348dd: 294,494 tensors of the language model, 547 of the vision tower
    try testing.expectEqual(@as(usize, 294_494), layout.tensors);
    try testing.expectEqual(@as(usize, 547), layout.dropped);
    // KDA's f, g and b, DSA's kv_b and indexer, the routers and eh_proj BF16; KDA, the dense MLPs, the MTP
    // layer's attention and shared expert and the head MXFP8; MLA and the shared experts FP8_PB_WO; the
    // routed experts NVFP4, csf_32x2_v1 but the MTP layer's
    try testing.expectEqual(@as(usize, 258), layout.schemes.get(.bf16));
    try testing.expectEqual(@as(usize, 157), layout.schemes.get(.mxfp8));
    try testing.expectEqual(@as(usize, 170), layout.schemes.get(.fp8_block));
    try testing.expectEqual(@as(usize, 37_152), layout.schemes.get(.nvfp4));
    try testing.expectEqual(@as(usize, 36_288), layout.csf32);
    try testing.expectEqual(@as(usize, 370), layout.mapped);
    try testing.expectEqual(@as(u64, 177_030_962_328), layout.bytes);
    try testing.expectEqual(@as(u64, 94_603_131_032), layout.rank);
    var it = checkpoint.tensors.iterator();
    while (it.next()) |entry| try testing.expect(rule(entry.key_ptr.*) != null);
}
