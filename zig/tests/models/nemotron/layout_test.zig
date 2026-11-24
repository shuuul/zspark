//! `layout.zig`'s tests.

const std = @import("std");
const testing = std.testing;
const config_file = @import("../../../src/models/nemotron/config.zig");
const layout_file = @import("../../../src/models/nemotron/layout.zig");
const fixture = @import("fixture.zig");

const holds = layout_file.holds;
const placed = layout_file.placed;

test "the kinds' shapes, and which layers hold them" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try fixture.parse(arena.allocator(), &diagnostic);
    var dims: [3]u64 = undefined;
    for (layout_file.kinds) |one| {
        const expected: []const u64 = if (std.mem.eql(u8, one.leaf, "mixer.in_proj.weight"))
            &.{ 772, 128 }
        else if (std.mem.eql(u8, one.leaf, "mixer.conv1d.weight"))
            &.{ 512, 1, 4 }
        else if (std.mem.eql(u8, one.leaf, "mixer.o_proj.weight"))
            &.{ 128, 256 }
        else if (std.mem.eql(u8, one.leaf, "eh_proj.weight"))
            &.{ 128, 256 }
        else if (std.mem.eql(u8, one.leaf, "attn.w_qkv.weight"))
            &.{ 384, 128 }
        else if (std.mem.eql(u8, one.leaf, "perception.proj.weight"))
            &.{ 128, 128 }
        else if (std.mem.eql(u8, one.leaf, "perception.preprocessor.featurizer.fb"))
            &.{ 1, 32, 33 }
        else if (std.mem.eql(u8, one.leaf, "perception.preprocessor.featurizer.window"))
            &.{16}
        else if (std.mem.eql(u8, one.leaf, "mixer.gate.weight"))
            &.{ 4, 128 }
        else if (std.mem.eql(u8, one.leaf, "perception.encoder.asr_encoder.pre_encode.proj.weight"))
            &.{ 128, 64 }
        else
            continue;
        if (one.diar() and c.diar == null) continue; // a diarization kind without the expert mounted
        try testing.expectEqualSlices(u64, expected, one.shape(c, &dims));
    }
    // layers 0 mamba, 1 moe, 2 attention, 3 moe; the MTP block's 0 attention, 1 MoE (its last)
    try testing.expect(holds(c, .mamba, 0) and !holds(c, .mamba, 1));
    try testing.expect(holds(c, .attention, 2) and !holds(c, .attention, 3));
    try testing.expect(holds(c, .moe, 1) and holds(c, .expert, 3) and !holds(c, .moe, 0));
    try testing.expect(holds(c, .every, 2) and !holds(c, .top, 0));
    try testing.expect(holds(c, .mtp, 0) and holds(c, .mtp, 1));
    try testing.expect(holds(c, .mtp_attention, 0) and !holds(c, .mtp_attention, 1));
    try testing.expect(holds(c, .mtp_moe, 1) and holds(c, .mtp_expert, 1) and !holds(c, .mtp_moe, 0));
    try testing.expect(!holds(c, .mtp_final, 0) and holds(c, .mtp_final, 1));
    try testing.expect(holds(c, .encoder, 0));
}

test "a whole checkpoint is taken: every tensor by name, its linears as the map stores them" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try fixture.parse(arena, &diagnostic);
    var checkpoint = try fixture.synthetic(arena, c, 7);

    const whole = try layout_file.Layout.init(arena, &c, &checkpoint, &diagnostic);
    try testing.expectEqual(checkpoint.tensors.count(), whole.tensors);
    try testing.expectEqual(@as(usize, fixture.nvfp4), whole.schemes.get(.nvfp4));
    try testing.expectEqual(@as(usize, fixture.fp8), whole.schemes.get(.fp8));
    try testing.expectEqual(@as(usize, fixture.bf16), whole.schemes.get(.bf16));
    try testing.expectEqual(@as(usize, 0), whole.schemes.get(.fp8_block));
    try testing.expectEqual(@as(usize, 0), whole.schemes.get(.mxfp8));
    try testing.expectEqual(@as(usize, fixture.mapped), whole.mapped);
    var counted: usize = 0;
    for (whole.found) |found| counted += found.count;
    try testing.expectEqual(whole.tensors, counted);
    // every tensor of the synthetic checkpoint has its place
    var it = checkpoint.tensors.iterator();
    while (it.next()) |entry| try testing.expect(placed(c, entry.key_ptr.*));
    try testing.expect(placed(c, "backbone.layers.1.mixer.experts.0.up_proj.weight_scale"));
    try testing.expect(placed(c, "llm.mtp.layers.1.mixer.experts.2.down_proj.input_scale"));
    try testing.expect(!placed(c, "backbone.layers.0.mixer.q_proj.weight"));
    try testing.expect(!placed(c, "backbone.layers.2.mixer.in_proj.weight"));
    try testing.expect(!placed(c, "llm.mtp.layers.2.norm.weight"));
    try testing.expect(!placed(c, "perception.encoder.asr_encoder.layers.4.norm1.weight"));
    try testing.expect(!placed(c, "vision.blocks.0.attn.qkv.weight"));

    // the diarization encoder and the CTC heads are counted and not read
    try checkpoint.tensors.put(arena, "perception.encoder.diarization_model.encoder.layers.0.attn.out_proj.weight", .{ .dtype = .BF16, .shape = &.{ 2, 3 }, .data = &@as([12]u8, @splat(0)) });
    try checkpoint.tensors.put(arena, "ctc_timestamp.decoder.residual_scale", .{ .dtype = .F32, .shape = &.{}, .data = &@as([4]u8, @splat(0)) });
    const again = try layout_file.Layout.init(arena, &c, &checkpoint, &diagnostic);
    try testing.expectEqual(@as(usize, 2), again.skipped);
    try testing.expectEqual(@as(u64, 16), again.skipped_bytes);
    try testing.expectEqual(whole.tensors, again.tensors);
}

test "a tensor missing, misshapen, unplaced or unmapped is refused by its name" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try fixture.parse(arena, &diagnostic);
    var checkpoint = try fixture.synthetic(arena, c, 3);
    const Layout = layout_file.Layout;

    // one gone
    const gone = "backbone.layers.0.mixer.conv1d.weight";
    const was = checkpoint.tensors.get(gone).?;
    try testing.expect(checkpoint.tensors.swapRemove(gone));
    try testing.expectError(error.MissingTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, gone) != null);
    try checkpoint.tensors.put(arena, gone, was);

    // one of another dtype: the embedding read BF16
    const named = "backbone.embeddings.weight";
    const tensor = checkpoint.tensors.get(named).?;
    try checkpoint.tensors.put(arena, named, .{ .dtype = .F32, .shape = tensor.shape, .data = tensor.data });
    try testing.expectError(error.UnexpectedTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, named) != null);
    try checkpoint.tensors.put(arena, named, tensor);

    // an NVFP4 block scale of the wrong width
    const scale_name = "backbone.layers.1.mixer.experts.2.up_proj.weight_scale";
    const scale = checkpoint.tensors.get(scale_name).?;
    try checkpoint.tensors.put(arena, scale_name, .{ .dtype = .F8_E4M3, .shape = &.{ 32, 3 }, .data = scale.data[0..96] });
    try testing.expectError(error.UnexpectedTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, scale_name) != null);
    try checkpoint.tensors.put(arena, scale_name, scale);

    // one the layout has no place for
    try checkpoint.tensors.put(arena, "backbone.layers.0.mixer.extra.weight", .{ .dtype = .BF16, .shape = &.{ 1, 1 }, .data = &@as([2]u8, @splat(0)) });
    try testing.expectError(error.UnknownTensor, Layout.init(arena, &c, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, "mixer.extra") != null);
    try testing.expect(checkpoint.tensors.swapRemove("backbone.layers.0.mixer.extra.weight"));

    // a map entry no linear is
    const stray = try std.mem.replaceOwned(u8, arena, fixture.quant_text, "\"mtp.layers.0.eh_proj\":{\"quant_algo\":\"FP8\"},", "\"mtp.layers.0.eh_proj\":{\"quant_algo\":\"FP8\"},\"mtp.layers.0.mixer.gate\":{\"quant_algo\":\"FP8\"},");
    const strayed = try config_file.Config.parse(arena, fixture.config_text, fixture.backbone_text, stray, &diagnostic);
    try testing.expectError(error.UnknownTensor, Layout.init(arena, &strayed, &checkpoint, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, "mtp.layers.0.mixer.gate") != null);
}

test "the manifest's home shard: a tensor's file among the shards" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: config_file.Diagnostic = .{};
    const c = try fixture.parse(arena.allocator(), &diagnostic);
    var checkpoint = try fixture.synthetic(arena.allocator(), c, 1);
    // a synthetic checkpoint has no shards: every name's home is unknown, not a crash
    try testing.expectEqual(@as(?usize, null), layout_file.home(&checkpoint, "backbone.embeddings.weight"));
}
