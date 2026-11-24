//! `layout.zig`'s tests on checkpoints written into a temporary directory: every tensor resolved, what a
//! partial checkpoint lacks, the mixed-precision layout's formats, the refusals; `centredFromMeans` on means alone.

const std = @import("std");
const Allocator = std.mem.Allocator;
const numeric = @import("../../../src/weights/numeric.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const qwen4_config = @import("../../../src/models/qwen4/config.zig");
const layout_file = @import("../../../src/models/qwen4/layout.zig");
const Io = std.Io;
const Error = layout_file.Error;
const Layout = layout_file.Layout;
const Linear = layout_file.Linear;
const Scheme = layout_file.Scheme;
const ranges = layout_file.ranges;
const centredFromMeans = layout_file.centredFromMeans;
const Config = qwen4_config.Config;
const Diagnostic = qwen4_config.Diagnostic;
const Ngram = qwen4_config.Ngram;
const Checkpoint = safetensors.Checkpoint;
const Tensor = safetensors.Tensor;

/// A config small enough to write whole: two layers (DeltaNet, then attention with the PLE) and the MTP head.
const fixture_config =
    \\{"model_type":"qwen4_exp","eos_token_id":7,
    \\"text_config":{"hidden_size":128,"num_hidden_layers":2,"layer_types":["linear_attention","full_attention"],
    \\"vocab_size":100,"rms_norm_eps":1e-06,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,
    \\"indexer_n_heads":1,"indexer_head_dim":32,
    \\"linear_num_key_heads":2,"linear_num_value_heads":2,"linear_key_head_dim":16,"linear_value_head_dim":16,
    \\"linear_conv_kernel_dim":4,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":128,
    \\"shared_expert_intermediate_size":128,"hc_count":2,"hc_lowrank":16,"ple_layer_ids":[2],
    \\"heads_per_ngram":2,"ngram_vocab_size_base":100,"make_ngram_vocab_size_divisible_by":8,"split_ngram_parts":2,
    \\"mtp":{"layer_types":["full_attention"],"num_hidden_layers":1}},
    \\"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{
    \\"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16},
    \\"model.language_model.layers.1.mlp.experts":{"quant_algo":"NVFP4","group_size":16},
    \\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding":{"quant_algo":"NVFP4","group_size":16},
    \\"mtp.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16}}}}
;

const fx = struct {
    /// The csf experts' scales: every row's base, every code byte (an even column 1, an odd 2), and one
    /// exception at flat position `csf_at`.
    const csf_base: u8 = 0x30;
    const csf_codes: u8 = 0x21;
    const csf_at: u32 = 5;
    const csf_exception: u8 = 0x7e;
};

/// Writes the fixture config's checkpoint as one safetensors file, zeros but for the values tests set.
const Fixture = struct {
    gpa: Allocator = std.testing.allocator,
    header: std.ArrayList(u8) = .empty,
    data: std.ArrayList(u8) = .empty,
    /// Tensors whose name contains this are left out.
    omit: []const u8 = "\x00",
    /// What every `hc_norm` holds.
    norm: f32 = 0,
    /// The MTP experts stacked in bf16.
    stacked: bool = false,
    /// A mixed-precision layout, FP4-Local's: MXFP8 in every DeltaNet, attention and shared-expert projection,
    /// the routed experts' NVFP4 scales csf-encoded (`fx.csf_*`). Without it those projections are bf16 and the
    /// scales plain; the n-gram table and the MTP experts are NVFP4 with plain scales either way.
    mixed: bool = false,
    /// The csf fixed stream's slabs of 16 rows: 8 for the experts' 128 rows.
    csf_slabs: u64 = 128 / 16,

    fn deinit(f: *Fixture) void {
        f.header.deinit(f.gpa);
        f.data.deinit(f.gpa);
    }

    fn add(f: *Fixture, name: []const u8, dtype: safetensors.DType, shape: []const u64) ![]u8 {
        var count: u64 = dtype.size();
        for (shape) |dim| count *= dim;
        if (std.mem.indexOf(u8, name, f.omit) != null) return &.{};
        const begin = f.data.items.len;
        try f.header.print(f.gpa, "{s}\"{s}\":{{\"dtype\":\"{t}\",\"shape\":[", .{ if (f.header.items.len == 0) "{" else ",", name, dtype });
        for (shape, 0..) |dim, i| try f.header.print(f.gpa, "{s}{d}", .{ if (i == 0) "" else ",", dim });
        try f.header.print(f.gpa, "],\"data_offsets\":[{d},{d}]}}", .{ begin, begin + count });
        const bytes = try f.data.addManyAsSlice(f.gpa, @intCast(count));
        @memset(bytes, 0);
        return bytes;
    }

    fn addf(f: *Fixture, comptime format: []const u8, args: anytype, dtype: safetensors.DType, shape: []const u64) ![]u8 {
        var buffer: [256]u8 = undefined;
        return f.add(try std.fmt.bufPrint(&buffer, format, args), dtype, shape);
    }

    fn hyperConnection(f: *Fixture, comptime format: []const u8, args: anytype, inject: bool) !void {
        var buffer: [128]u8 = undefined;
        const base = try std.fmt.bufPrint(&buffer, format, args);
        _ = try f.addf("{s}.input_mix_weight_down.weight", .{base}, .BF16, &.{ 16, 256 });
        if (inject) _ = try f.addf("{s}.block_inject_weight.weight", .{base}, .BF16, &.{ 2, 256 });
        _ = try f.addf("{s}.input_mix_weight_up.weight", .{base}, .BF16, &.{ 256, 16 });
        const norm = try f.addf("{s}.hc_norm.weight", .{base}, .BF16, &.{256});
        for (0..norm.len / 2) |i| std.mem.writeInt(u16, norm[i * 2 ..][0..2], numeric.bf16FromF32(f.norm), .little);
    }

    /// A linear as bf16, or as MXFP8 with its scales outside the MTP head of a mixed layout.
    fn dense(f: *Fixture, base: []const u8, comptime field: []const u8, rows: u64, cols: u64) !void {
        if (f.mixed and !std.mem.startsWith(u8, base, "mtp")) {
            _ = try f.addf("{s}." ++ field ++ ".weight", .{base}, .F8_E4M3, &.{ rows, cols });
            const scales = try f.addf("{s}." ++ field ++ ".weight_scale", .{base}, .U8, &.{ rows, cols / 32 });
            @memset(scales, 127);
            return;
        }
        _ = try f.addf("{s}." ++ field ++ ".weight", .{base}, .BF16, &.{ rows, cols });
    }

    fn moe(f: *Fixture, base: []const u8, drafter: bool) !void {
        _ = try f.addf("{s}.gate.weight", .{base}, .BF16, &.{ 4, 128 });
        _ = try f.addf("{s}.shared_expert_gate.weight", .{base}, .BF16, &.{ 1, 128 });
        inline for (.{ "gate_proj", "up_proj", "down_proj" }) |proj| {
            try f.dense(base, "shared_expert." ++ proj, 128, 128);
        }
        if (drafter and f.stacked) {
            _ = try f.addf("{s}.experts.gate_up_proj", .{base}, .BF16, &.{ 4, 256, 128 });
            _ = try f.addf("{s}.experts.down_proj", .{base}, .BF16, &.{ 4, 128, 128 });
            return;
        }
        for (0..4) |i| inline for (.{ "gate_proj", "up_proj", "down_proj" }) |proj| {
            if (f.mixed and !drafter) {
                _ = try f.addf("{s}.experts.{d}." ++ proj ++ ".weight", .{ base, i }, .U8, &.{ 128, 64 });
                const fixed = try f.addf("{s}.experts.{d}." ++ proj ++ ".weight_scale.nvfp4_csf_fixed", .{ base, i }, .U8, &.{ f.csf_slabs, 16 * (1 + 8 / 2) });
                for (0..fixed.len / 80) |slab| {
                    @memset(fixed[slab * 80 ..][0..16], fx.csf_base);
                    @memset(fixed[slab * 80 + 16 ..][0..64], fx.csf_codes);
                }
                const words = try f.addf("{s}.experts.{d}." ++ proj ++ ".weight_scale.nvfp4_csf_exceptions", .{ base, i }, .U32, &.{1});
                if (words.len != 0) std.mem.writeInt(u32, words[0..4], fx.csf_at | @as(u32, fx.csf_exception) << 24, .little);
                _ = try f.addf("{s}.experts.{d}." ++ proj ++ ".weight_scale_2", .{ base, i }, .F32, &.{});
            } else {
                _ = try f.addf("{s}.experts.{d}." ++ proj ++ ".weight", .{ base, i }, .U8, &.{ 128, 64 });
                _ = try f.addf("{s}.experts.{d}." ++ proj ++ ".weight_scale", .{ base, i }, .F8_E4M3, &.{ 128, 8 });
                _ = try f.addf("{s}.experts.{d}." ++ proj ++ ".weight_scale_2", .{ base, i }, .F32, &.{});
            }
        };
    }

    fn attention(f: *Fixture, base: []const u8) !void {
        try f.dense(base, "q_proj", 256, 128);
        try f.dense(base, "k_proj", 64, 128);
        try f.dense(base, "v_proj", 64, 128);
        try f.dense(base, "indexer.index_qk_proj", 64, 128);
        _ = try f.addf("{s}.q_norm.weight", .{base}, .BF16, &.{64});
        _ = try f.addf("{s}.k_norm.weight", .{base}, .BF16, &.{64});
        _ = try f.addf("{s}.indexer.q_layernorm.weight", .{base}, .BF16, &.{32});
        _ = try f.addf("{s}.indexer.k_layernorm.weight", .{base}, .BF16, &.{32});
        try f.dense(base, "o_proj", 128, 128);
    }

    fn i64s(f: *Fixture, name: []const u8, comptime T: type, values: []const T) !void {
        const bytes = try f.add(name, .I64, &.{values.len});
        if (bytes.len == 0) return;
        for (values, 0..) |v, i| std.mem.writeInt(i64, bytes[i * 8 ..][0..8], @intCast(v), .little);
    }

    fn ple(f: *Fixture, base: []const u8, ngram: Ngram) !void {
        var buffer: [128]u8 = undefined;
        try f.i64s(try std.fmt.bufPrint(&buffer, "{s}.ple_embedding.layer_multipliers", .{base}), i64, ngram.multipliers);
        try f.i64s(try std.fmt.bufPrint(&buffer, "{s}.ple_embedding.ngram_heads_offsets", .{base}), u64, ngram.head_offsets);
        try f.i64s(try std.fmt.bufPrint(&buffer, "{s}.ple_embedding.ngram_heads_vocab_sizes", .{base}), u64, ngram.head_sizes);
        // 424 rows of 32 NVFP4 values over two shards, an e4m3 scale per 16
        for ([_]u64{ 200, 224 }, 0..) |rows, i| {
            _ = try f.addf("{s}.ple_embedding.ngram_embedding.shard_{d}.weight", .{ base, i }, .U8, &.{ rows, 16 });
            _ = try f.addf("{s}.ple_embedding.ngram_embedding.shard_{d}.weight_scale", .{ base, i }, .F8_E4M3, &.{ rows, 2 });
        }
        _ = try f.addf("{s}.ple_embedding.ngram_embedding.weight_scale_2", .{base}, .F32, &.{1});
        _ = try f.addf("{s}.key_proj.weight", .{base}, .BF16, &.{ 256, 128 });
        _ = try f.addf("{s}.value_proj.weight", .{base}, .BF16, &.{ 128, 128 });
        inline for (.{ "norm_key", "norm_query", "norm_conv" }) |norm| {
            _ = try f.addf("{s}." ++ norm ++ ".weight", .{base}, .BF16, &.{256});
        }
        _ = try f.addf("{s}.conv1d.weight", .{base}, .BF16, &.{ 256, 1, 4 });
    }

    fn write(f: *Fixture, io: Io, dir: Io.Dir, config: Config) !void {
        var arena: std.heap.ArenaAllocator = .init(f.gpa);
        defer arena.deinit();
        const lm = "model.language_model.";

        _ = try f.add(lm ++ "embed_tokens.weight", .BF16, &.{ 100, 128 });
        try f.hyperConnection(lm ++ "layers.0.attn_hyper_connection", .{}, true);
        try f.hyperConnection(lm ++ "layers.0.mlp_hyper_connection", .{}, true);
        const gdn = lm ++ "layers.0.linear_attn";
        try f.dense(gdn, "in_proj_qkv", 96, 128);
        try f.dense(gdn, "in_proj_z", 32, 128);
        try f.dense(gdn, "in_proj_b", 2, 128);
        try f.dense(gdn, "in_proj_a", 2, 128);
        _ = try f.add(gdn ++ ".conv1d.weight", .BF16, &.{ 96, 1, 4 });
        _ = try f.add(gdn ++ ".A_log", .BF16, &.{2});
        _ = try f.add(gdn ++ ".dt_bias", .BF16, &.{2});
        _ = try f.add(gdn ++ ".norm.weight", .BF16, &.{16});
        try f.dense(gdn, "out_proj", 128, 32);
        try f.moe(lm ++ "layers.0.mlp", false);

        try f.hyperConnection(lm ++ "layers.1.attn_hyper_connection", .{}, true);
        try f.hyperConnection(lm ++ "layers.1.mlp_hyper_connection", .{}, true);
        try f.attention(lm ++ "layers.1.self_attn");
        try f.ple(lm ++ "layers.1.ple", try .init(arena.allocator(), config, 0));
        try f.moe(lm ++ "layers.1.mlp", false);

        try f.hyperConnection(lm ++ "hyper_connection_mixer", .{}, false);
        _ = try f.add("lm_head.weight", .BF16, &.{ 100, 128 });

        _ = try f.add("mtp.pre_fc_norm_embedding.weight", .BF16, &.{128});
        _ = try f.add("mtp.pre_fc_norm_hidden.weight", .BF16, &.{256});
        try f.dense("mtp", "fc_embedding", 128, 128);
        try f.dense("mtp", "fc_hidden", 128, 128);
        try f.hyperConnection("mtp.layers.0.attn_hyper_connection", .{}, true);
        try f.hyperConnection("mtp.layers.0.mlp_hyper_connection", .{}, true);
        try f.attention("mtp.layers.0.self_attn");
        try f.moe("mtp.layers.0.mlp", true);
        try f.hyperConnection("mtp.hyper_connection_mixer", .{}, false);
        // a tensor the model never reads
        _ = try f.add("model.visual.patch_embed.weight", .BF16, &.{ 2, 2 });

        try f.header.append(f.gpa, '}');
        var file = try dir.createFile(io, "model.safetensors", .{});
        defer file.close(io);
        var prefix: [8]u8 = undefined;
        std.mem.writeInt(u64, &prefix, f.header.items.len, .little);
        try file.writeStreamingAll(io, &prefix);
        try file.writeStreamingAll(io, f.header.items);
        try file.writeStreamingAll(io, f.data.items);
    }
};

/// The fixture's config, checkpoint and layout, torn down together.
const Case = struct {
    tmp: std.testing.TmpDir,
    arena: std.heap.ArenaAllocator,
    config: Config,
    checkpoint: Checkpoint,
    diagnostic: Diagnostic = .{},

    fn open(case: *Case, fixture: Fixture) !void {
        const gpa = std.testing.allocator;
        const io = std.testing.io;
        case.tmp = std.testing.tmpDir(.{ .iterate = true });
        errdefer case.tmp.cleanup();
        case.arena = .init(gpa);
        errdefer case.arena.deinit();
        case.diagnostic = .{};
        case.config = try .parse(case.arena.allocator(), fixture_config, null, &case.diagnostic);
        var f = fixture;
        defer f.deinit();
        try f.write(io, case.tmp.dir, case.config);
        case.checkpoint = try .open(gpa, io, case.tmp.dir, ".");
    }

    fn layout(case: *Case) Error!Layout {
        return .init(case.arena.allocator(), &case.config, &case.checkpoint, &case.diagnostic);
    }

    fn close(case: *Case) void {
        case.checkpoint.close();
        case.arena.deinit();
        case.tmp.cleanup();
    }
};

test "resolves every tensor of a whole checkpoint" {
    var case: Case = undefined;
    try case.open(.{});
    defer case.close();
    const layout = try case.layout();

    try std.testing.expect(layout.complete());
    try std.testing.expectEqualStrings("model.language_model.", layout.prefix);
    try std.testing.expectEqual(@as(?bool, true), layout.norms_centred);
    try std.testing.expectEqual(@as(usize, 2), layout.layersFound());

    const first = layout.layers[0].?;
    try std.testing.expectEqualSlices(u64, &.{ 2, 256 }, first.attn_hc.inject.?.weight.shape);
    const gdn = first.block.linear_attention;
    try std.testing.expectEqual(Scheme.bf16, gdn.qkv.scheme);
    try std.testing.expectEqual(@as(u32, 96), gdn.qkv.rows);
    try std.testing.expectEqual(@as(usize, 2), gdn.a_log.len());
    try std.testing.expect(!gdn.norm.gamma and first.attn_hc.norm.gamma);
    try std.testing.expect(first.ple == null);
    const routed = first.moe.experts.each;
    try std.testing.expectEqual(@as(usize, 4), routed.len);
    try std.testing.expectEqual(Scheme.nvfp4, routed[3].down.scheme);
    try std.testing.expectEqualStrings("model.language_model.layers.0.mlp.experts.3.down_proj", routed[3].down.name);
    try std.testing.expectEqualSlices(u64, &.{ 128, 8 }, routed[3].down.scale.?.shape);
    try std.testing.expectEqual(@as(u64, 1), routed[3].down.scale_2.?.count());

    const second = layout.layers[1].?;
    try std.testing.expectEqual(@as(u32, 64), second.block.full_attention.index_qk.rows);
    const ple = second.ple.?;
    try std.testing.expectEqual(Scheme.nvfp4, ple.table.scheme);
    try std.testing.expectEqual(@as(u64, 424), ple.table.rows);
    try std.testing.expectEqual(@as(usize, 2), ple.table.shards.len);
    try std.testing.expect(ple.table.scale != null);

    try std.testing.expect(layout.mixer.?.inject == null);
    const mtp = layout.mtp.?;
    try std.testing.expectEqual(Scheme.nvfp4, mtp.layer.moe.experts.each[0].gate.scheme);
    try std.testing.expectEqualSlices(u64, &.{ 128, 8 }, mtp.layer.moe.experts.each[0].gate.scale.?.shape);

    // 2 x 12 routed nvfp4 and 12 nvfp4 drafters; the n-gram table is counted apart
    try std.testing.expectEqual(@as(usize, 36), layout.stats.linears.get(.nvfp4));
    try std.testing.expectEqual(@as(usize, 0), layout.stats.linears.get(.mxfp8));
    try std.testing.expectEqual(@as(u64, 36 * (128 * 64 + 128 * 8 + 4)), layout.stats.bytes.get(.nvfp4));
    try std.testing.expectEqual(@as(usize, 2), layout.stats.table_shards);
    try std.testing.expectEqual(@as(?Scheme, .nvfp4), layout.stats.table_scheme);
}

test "reports what a partial checkpoint lacks" {
    var case: Case = undefined;
    try case.open(.{ .omit = "layers.1.", .norm = 1 });
    defer case.close();
    const layout = try case.layout();

    try std.testing.expect(!layout.complete());
    try std.testing.expectEqual(@as(?bool, false), layout.norms_centred);
    try std.testing.expect(layout.layers[0] != null and layout.layers[1] == null);
    try std.testing.expect(layout.embed != null and layout.head != null and layout.mixer != null and layout.mtp != null);
    // 8 hyper-connection tensors, 9 attention, 11 PLE (a table's scale is named once a shard tells its
    // scheme), 5 + 4 x 3 MoE weights
    try std.testing.expectEqual(@as(usize, 8 + 9 + 11 + 5 + 12), layout.missing.len);
    try std.testing.expectEqualStrings("model.language_model.layers.1.attn_hyper_connection.input_mix_weight_down.weight", layout.missing[0]);
    for (layout.missing) |name| try std.testing.expect(std.mem.indexOf(u8, name, "layers.1.") != null);
    // layer 0's 4 x 3 routed experts and the MTP layer's, NVFP4 both
    try std.testing.expectEqual(@as(usize, 24), layout.stats.linears.get(.nvfp4));
}

test "waits for a scale in a later shard" {
    var case: Case = undefined;
    try case.open(.{ .omit = "experts.2.up_proj.weight_scale" });
    defer case.close();
    const layout = try case.layout();
    // each layer's expert 2 and the MTP's lack both NVFP4 scales
    try std.testing.expectEqual(@as(usize, 6), layout.missing.len);
    try std.testing.expectEqualStrings("model.language_model.layers.0.mlp.experts.2.up_proj.weight_scale", layout.missing[0]);
    try std.testing.expectEqualStrings("model.language_model.layers.0.mlp.experts.2.up_proj.weight_scale_2", layout.missing[1]);
    try std.testing.expectEqualStrings("mtp.layers.0.mlp.experts.2.up_proj.weight_scale", layout.missing[4]);
    try std.testing.expectEqual(@as(usize, 0), layout.layersFound());
    try std.testing.expect(layout.mtp == null and layout.head != null);
}

test "a stacked bf16 drafter" {
    var case: Case = undefined;
    try case.open(.{ .stacked = true });
    defer case.close();
    const layout = try case.layout();

    try std.testing.expect(layout.complete());
    try std.testing.expectEqualSlices(u64, &.{ 4, 256, 128 }, layout.mtp.?.layer.moe.experts.stacked.gate_up.shape);
}

test "mixed precision: MXFP8 projections, csf-encoded expert scales decoded to the bytes, an NVFP4 table" {
    var case: Case = undefined;
    try case.open(.{ .mixed = true });
    defer case.close();
    const layout = try case.layout();

    try std.testing.expect(layout.complete());
    const first = layout.layers[0].?;
    const gdn = first.block.linear_attention;
    for ([_]Linear{ gdn.qkv, gdn.z, gdn.b, gdn.a, gdn.out, first.moe.shared.gate, first.moe.shared.up, first.moe.shared.down }) |linear|
        try std.testing.expectEqual(Scheme.mxfp8, linear.scheme);
    try std.testing.expectEqualSlices(u64, &.{ 96, 4 }, gdn.qkv.scale.?.shape);
    const attention = layout.layers[1].?.block.full_attention;
    for ([_]Linear{ attention.q, attention.k, attention.v, attention.o, attention.index_qk }) |linear|
        try std.testing.expectEqual(Scheme.mxfp8, linear.scheme);
    // the routers, hyper-connections and head stay bf16
    try std.testing.expectEqual(Scheme.bf16, first.moe.router.scheme);
    try std.testing.expectEqual(Scheme.bf16, layout.head.?.scheme);

    const routed = first.moe.experts.each[2].up;
    try std.testing.expectEqual(Scheme.nvfp4, routed.scheme);
    try std.testing.expect(routed.scale == null and routed.csf != null and routed.scale_2 != null);
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const scales = try routed.blockScales(arena.allocator());
    try std.testing.expectEqual(@as(usize, 128 * 8), scales.len);
    for (scales, 0..) |byte, i| {
        const want: u8 = if (i == fx.csf_at) fx.csf_exception else fx.csf_base + @as(u8, if (i % 2 == 0) 1 else 2);
        try std.testing.expectEqual(want, byte);
    }
    // the drafter's experts keep their plain scales, which are returned as stored
    const drafter = layout.mtp.?.layer.moe.experts.each[0].gate;
    try std.testing.expect(drafter.csf == null);
    try std.testing.expectEqual(drafter.scale.?.data.ptr, (try drafter.blockScales(arena.allocator())).ptr);

    const table = layout.layers[1].?.ple.?.table;
    try std.testing.expectEqual(Scheme.nvfp4, table.scheme);
    try std.testing.expectEqualSlices(u64, &.{ 224, 2 }, table.shards[1].scale.?.shape);
    try std.testing.expectEqual(@as(u64, 1), table.scale.?.count());

    // 2 x 12 routed with csf scales and 12 drafters with plain ones; 8 + 8 MXFP8
    try std.testing.expectEqual(@as(usize, 36), layout.stats.linears.get(.nvfp4));
    try std.testing.expectEqual(@as(usize, 24), layout.stats.csf);
    try std.testing.expectEqual(@as(usize, 16), layout.stats.linears.get(.mxfp8));

    // what a loader holds covers the compressed streams, never decoded bytes
    var held: std.ArrayList([]const u8) = .empty;
    defer held.deinit(std.testing.allocator);
    try ranges(std.testing.allocator, routed, &held);
    try std.testing.expectEqual(@as(usize, 4), held.items.len);
    try std.testing.expectEqual(routed.csf.?.fixed.data.ptr, held.items[1].ptr);
}

test "csf scales wait for their streams, and refuse a fixed stream of the wrong shape" {
    var waiting: Case = undefined;
    try waiting.open(.{ .mixed = true, .omit = "experts.1.down_proj.weight_scale.nvfp4_csf_exceptions" });
    defer waiting.close();
    const partial = try waiting.layout();
    // each layer's expert 1 lacks its exceptions
    try std.testing.expectEqual(@as(usize, 2), partial.missing.len);
    try std.testing.expectEqualStrings("model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale.nvfp4_csf_exceptions", partial.missing[0]);

    var wrong: Case = undefined;
    try wrong.open(.{ .mixed = true, .csf_slabs = 4 });
    defer wrong.close();
    try std.testing.expectError(error.InvalidCheckpoint, wrong.layout());
    try std.testing.expect(std.mem.indexOf(u8, wrong.diagnostic.message, "nvfp4_csf_fixed: U8 { 4, 80 }, expected U8 { 8, 80 }") != null);
}

test "refuses a tensor of the wrong shape" {
    var case: Case = undefined;
    try case.open(.{});
    defer case.close();
    case.config.low = 32;
    try std.testing.expectError(error.InvalidCheckpoint, case.layout());
    try std.testing.expect(std.mem.indexOf(u8, case.diagnostic.message, "input_mix_weight_down.weight") != null);
    try std.testing.expect(std.mem.indexOf(u8, case.diagnostic.message, "expected BF16 { 32, 256 }") != null);
}

test "refuses n-gram constants the config does not derive" {
    var case: Case = undefined;
    try case.open(.{});
    defer case.close();
    case.config.seed = 99;
    try std.testing.expectError(error.InvalidCheckpoint, case.layout());
    try std.testing.expect(std.mem.indexOf(u8, case.diagnostic.message, "layer_multipliers") != null);
}

test "a linear's scheme from its tensors" {
    const t = struct {
        fn of(dtype: safetensors.DType, shape: []const u64) Tensor {
            return .{ .dtype = dtype, .shape = shape, .data = &.{} };
        }
    };
    try std.testing.expectEqual(Scheme.bf16, try Scheme.of(t.of(.BF16, &.{ 2, 2 }), null));
    try std.testing.expectEqual(Scheme.nvfp4, try Scheme.of(t.of(.U8, &.{ 2, 8 }), t.of(.F8_E4M3, &.{ 2, 1 })));
    try std.testing.expectEqual(Scheme.mxfp8, try Scheme.of(t.of(.F8_E4M3, &.{ 2, 64 }), t.of(.U8, &.{ 2, 2 })));
    // the retired checkpoint's per-tensor and block FP8
    try std.testing.expectError(error.UnknownStorage, Scheme.of(t.of(.F8_E4M3, &.{ 2, 128 }), t.of(.F32, &.{})));
    try std.testing.expectError(error.UnknownStorage, Scheme.of(t.of(.F8_E4M3, &.{ 2, 128 }), null));
    try std.testing.expectError(error.UnknownStorage, Scheme.of(t.of(.U8, &.{ 2, 8 }), null));
    try std.testing.expectError(error.UnknownStorage, Scheme.of(t.of(.I64, &.{2}), null));
}

test "norm storage from the hc_norm means" {
    var scales = [_]f64{ 0.9, 1.1, 1.0, 1.2 };
    try std.testing.expectEqual(@as(?bool, false), try centredFromMeans(&scales));
    var centred = [_]f64{ -0.1, 0.05, 0.0 };
    try std.testing.expectEqual(@as(?bool, true), try centredFromMeans(&centred));
    var mixed = [_]f64{ 0.0, 1.0, 0.0, 1.0 };
    try std.testing.expectError(error.Undecided, centredFromMeans(&mixed));
    try std.testing.expectEqual(@as(?bool, null), try centredFromMeans(&.{}));
}

test "a mixed-precision checkpoint: every tensor's format as its map declares, csf experts, MXFP8 projections" {
    // `ZSPARK_FLASH_NEXT_FP4_LOCAL`: headers only, no tensor read but the n-gram constants and norm means
    const path = std.testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var diagnostic: Diagnostic = .{};
    const config: Config = Config.read(arena, io, dir, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message});
        return err;
    };
    var checkpoint: Checkpoint = try .open(gpa, io, dir, ".");
    defer checkpoint.close();
    try std.testing.expectEqual(@as(usize, 21), checkpoint.shards.len);
    const layout = Layout.init(arena, &config, &checkpoint, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message});
        return err;
    };
    try std.testing.expect(layout.complete());

    // 48 x 512 x 3 routed experts with csf scales, the MTP's 512 x 3 with plain ones (W4A16_NVFP4); DeltaNet's
    // five projections in 36 layers, attention's five in 12, the shared expert's three in 48 as MXFP8
    try std.testing.expectEqual(@as(usize, 73_728), layout.stats.csf);
    try std.testing.expectEqual(@as(usize, 73_728 + 1_536), layout.stats.linears.get(.nvfp4));
    try std.testing.expectEqual(@as(usize, 36 * 5 + 12 * 5 + 48 * 3), layout.stats.linears.get(.mxfp8));
    try std.testing.expectEqual(@as(usize, 128), layout.stats.table_shards);
    try std.testing.expectEqual(@as(?Scheme, .nvfp4), layout.stats.table_scheme);
    try std.testing.expectEqual(Scheme.bf16, layout.head.?.scheme);

    // the MTP layer: its experts NVFP4 and listed; its attention, shared expert and fc bf16 and unlisted (the
    // config's `ignore` names them; the reader takes the format from the tensors)
    const mtp = layout.mtp.?;
    const drafter = mtp.layer.moe.experts.each[511].down;
    try std.testing.expect(drafter.scheme == .nvfp4 and drafter.csf == null and drafter.scale != null);
    try std.testing.expect(config.quant.find(drafter.name) != null);
    const head_attention = mtp.layer.block.full_attention;
    for ([_]Linear{ head_attention.q, head_attention.o, head_attention.index_qk, mtp.layer.moe.shared.gate, mtp.fc_hidden }) |linear| {
        try std.testing.expectEqual(Scheme.bf16, linear.scheme);
        try std.testing.expect(config.quant.find(linear.name) == null);
    }

    // every tensor of the language model and the head by its name: what the map declares, the dtype it stores
    var declared: usize = 0;
    for (checkpoint.tensors.keys(), checkpoint.tensors.values()) |name, tensor| {
        if (std.mem.startsWith(u8, name, "model.visual.")) continue;
        const layer = config.quant.find(name) orelse {
            // unlisted: bf16 or fp32, and the n-gram constants
            try std.testing.expect(tensor.dtype == .BF16 or tensor.dtype == .F32 or tensor.dtype == .I64);
            continue;
        };
        declared += 1;
        const leaf = name[std.mem.lastIndexOfScalar(u8, name, '.').? + 1 ..];
        const want: safetensors.DType = switch (layer.algo) {
            .mxfp8 => if (std.mem.eql(u8, leaf, "weight")) .F8_E4M3 else .U8,
            .nvfp4 => if (std.mem.eql(u8, leaf, "weight")) .U8 else if (std.mem.eql(u8, leaf, "nvfp4_csf_fixed")) .U8 else if (std.mem.eql(u8, leaf, "nvfp4_csf_exceptions")) .U32 else if (std.mem.eql(u8, leaf, "weight_scale")) .F8_E4M3 else .F32,
            else => return error.TestUnexpectedResult,
        };
        if (tensor.dtype != want) {
            std.debug.print("{s}: {t}, {t} declares {t}\n", .{ name, tensor.dtype, layer.algo, want });
            return error.TestUnexpectedResult;
        }
        if (layer.scale_encoding == .csf) try std.testing.expect(!std.mem.eql(u8, leaf, "weight_scale"));
    }
    // per routed expert projection weight, input_scale, weight_scale_2 and two streams; per drafter's weight,
    // scale and scale_2; per MXFP8 linear weight and scale; the table's 128 shards of two and its scale_2
    try std.testing.expectEqual(@as(usize, 73_728 * 5 + 1_536 * 3 + 384 * 2 + 128 * 2 + 1), declared);
}
