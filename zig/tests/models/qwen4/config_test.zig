//! `config.zig`'s tests.

const std = @import("std");
const config_file = @import("../../../src/models/qwen4/config.zig");
const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;
const Error = config_file.Error;
const Algo = config_file.Algo;
const LayerType = config_file.LayerType;
const Ngram = config_file.Ngram;

/// A two-layer text config with every field the parser requires; `{s}` takes the quantization block.
const fixture_head =
    \\{"model_type":"qwen4_exp","eos_token_id":7,
    \\"text_config":{"hidden_size":64,"num_hidden_layers":2,"layer_types":["linear_attention","full_attention"],
    \\"vocab_size":100,"rms_norm_eps":1e-06,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
    \\"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,"linear_value_head_dim":16,
    \\"linear_conv_kernel_dim":4,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,
    \\"shared_expert_intermediate_size":32,"hc_count":2,"hc_lowrank":16,"ple_layer_ids":[2],"ple_embed_dim":64,
    \\"heads_per_ngram":2,"ngram_vocab_size_base":100,"make_ngram_vocab_size_divisible_by":8,"split_ngram_parts":2,
    \\"eos_token_id":[9,7],"rope_parameters":{"rope_theta":1000,"partial_rotary_factor":0.25,"mrope_section":[2,1,1]},
    \\"mtp":{"hybrid":true,"layer_types":["full_attention"],"mtp_use_hidden_state_from_layer":null,
    \\"num_hidden_layers":1,"rope_theta":1000}}
;

const fixture_quant =
    \\,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION",
    \\"config_groups":{"group_0":{"input_activations":{"dynamic":false,"num_bits":4,"type":"float","group_size":16},
    \\"weights":{"dynamic":false,"num_bits":4,"type":"float","group_size":16},"targets":["model.language_model.layers.0.mlp.experts"]},
    \\"group_1":{"weights":{"dynamic":false,"num_bits":8,"type":"float"},"targets":["x"]}},
    \\"quantized_layers":{"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16},
    \\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding":{"quant_algo":"NVFP4","group_size":16},
    \\"mtp.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16}}}}
;

fn expectRefused(expected: Error, comptime text: []const u8, needle: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    try std.testing.expectError(expected, Config.parse(arena.allocator(), text, null, &diagnostic));
    try std.testing.expect(std.mem.indexOf(u8, diagnostic.message, needle) != null);
}

test "parses the text config and its derived sizes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const config: Config = try .parse(arena.allocator(), fixture_head ++ fixture_quant,
        \\{"eos_token_id":[7,11]}
    , &diagnostic);

    try std.testing.expectEqual(@as(u32, 64), config.hidden);
    try std.testing.expectEqualSlices(LayerType, &.{ .linear_attention, .full_attention }, config.layer_types);
    try std.testing.expectEqual(@as(u32, 8), config.rotary_dim);
    try std.testing.expectEqual(@as(f64, 1000), config.rope_theta);
    try std.testing.expectEqual([3]u32{ 2, 1, 1 }, config.mrope_section);
    try std.testing.expectEqual(@as(u32, 2 * 2 * 16 + 4 * 16), config.convDim());
    try std.testing.expectEqual(@as(u32, 128), config.wide());
    // defaults TensorFold supplies
    try std.testing.expectEqual(@as(u32, 4), config.index_heads);
    try std.testing.expectEqual(@as(u32, 512), config.topBlocks());
    try std.testing.expectEqual(@as(u32, 3), config.ngram_size);
    try std.testing.expectEqual(@as(u64, 1234), config.seed);
    // ple_layer_ids count from one
    try std.testing.expectEqualSlices(u32, &.{1}, config.ple_layers);
    try std.testing.expectEqual(@as(?usize, 0), config.pleIndex(1));
    try std.testing.expectEqual(@as(?usize, 0), config.tapIndex(5));
    try std.testing.expectEqual(@as(?usize, 1), config.tapIndex(15));
    try std.testing.expectEqual(@as(?usize, 4), config.tapIndex(45));
    try std.testing.expectEqual(@as(?usize, null), config.tapIndex(4));
    try std.testing.expectEqual(@as(?usize, null), config.tapIndex(6));
    try std.testing.expectEqual(@as(?usize, null), config.pleIndex(0));
    try std.testing.expectEqual(@as(u32, 4), config.ngramHeads());
    try std.testing.expectEqual(@as(u32, 16), config.ngramDims());
    // the top-level eos, then generation_config.json's new ones; the n-grams reset on the text config's first
    try std.testing.expectEqualSlices(u32, &.{ 7, 11 }, config.eos);
    try std.testing.expectEqual(@as(u32, 9), config.ple_eos);

    const mtp = config.mtp.?;
    try std.testing.expect(mtp.hybrid and mtp.hidden_state_from_layer == null and !mtp.dedicated_embeddings);

    try std.testing.expectEqualStrings("MIXED_PRECISION", config.quant.algo);
    try std.testing.expectEqual(@as(u32, 16), config.quant.nvfp4_group);
    try std.testing.expectEqual(@as(usize, 3), config.quant.layers.len);
    try std.testing.expectEqual(@as(usize, 2), config.quant.groups.len);
    try std.testing.expectEqual(@as(?u32, 4), config.quant.groups[0].input_bits);
    try std.testing.expectEqual(@as(?u32, null), config.quant.groups[1].input_bits);
    try std.testing.expectEqual(Algo.nvfp4, config.quant.find("model.language_model.layers.0.mlp.experts.3.up_proj").?.algo);
    try std.testing.expectEqual(Algo.nvfp4, config.quant.find("mtp.layers.0.mlp.experts").?.algo);
    try std.testing.expect(config.quant.find("model.language_model.layers.0.mlp.experts_gate") == null);
}

test "refuses checkpoints that are not ModelOpt NVFP4" {
    try expectRefused(error.UnsupportedQuantization, fixture_head ++ "}", "no quantization block");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization":{"group_size":32,"bits":4}}
    , "MLX affine");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization_config":{"quant_method":"exl3","version":"1.4.4","bits":3.05,"codebook":"mul1"}}
    , "quant_method \"exl3\"");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":
        \\{"model.language_model.layers.0.mlp.experts":{"quant_algo":"MXFP8","group_size":32}}}}
    , "beside the experts");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":
        \\{"model.language_model.layers.0.linear_attn.out_proj":{"quant_algo":"MXFP8","group_size":64}}}}
    , "MXFP8's block is 32");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":
        \\{"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16,"weight_scale_encoding":"zip"}}}}
    , "weight_scale_encoding \"zip\"");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":
        \\{"model.language_model.layers.0.linear_attn.out_proj":{"quant_algo":"MXFP8","group_size":32,"weight_scale_encoding":"csf"}}}}
    , "csf encodes NVFP4");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":
        \\{"model.language_model.layers.0.linear_attn.out_proj":{"quant_algo":"NVFP4"}}}}
    , "routed experts");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":
        \\{"lm_head":{"quant_algo":"FP8"}}}}
    , "lm_head is FP8, which zspark no longer reads");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization_config":{"quant_method":"modelopt","config_groups":{"group_0":{"weights":{"num_bits":4,"group_size":32}}}}}
    , "blocks of 32");
    try expectRefused(error.UnsupportedQuantization, fixture_head ++
        \\,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":
        \\{"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16},
        \\"lm_head":{"quant_algo":"FP8_PB_WO","group_size":32}}}}
    , "lm_head is FP8_PB_WO, which zspark no longer reads");
    try expectRefused(error.UnsupportedModel,
        \\{"model_type":"qwen3_5","text_config":{}}
    , "qwen3_5");
    try expectRefused(error.InvalidConfig, "{\"model_type\":\"qwen4_exp\",\"quantization_config\":{\"quant_method\":\"modelopt\"},\"text_config\":{}}", "hidden_size");
}

test "a mixed map: MXFP8 linears, csf-encoded NVFP4 experts, W4A16_NVFP4 drafters, an NVFP4 table" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const config: Config = try .parse(arena.allocator(), fixture_head ++
        \\,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":null,
        \\"ignore":["lm_head","mtp.fc_embedding","mtp.layers.0.self_attn*"],"quantized_layers":{
        \\"model.language_model.layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16,"weight_scale_encoding":"csf"},
        \\"model.language_model.layers.0.linear_attn.in_proj_qkv":{"quant_algo":"MXFP8","group_size":32},
        \\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding":{"quant_algo":"NVFP4","group_size":16},
        \\"model.visual.blocks.0.attn.qkv":{"quant_algo":"MXFP8","group_size":32},
        \\"mtp.layers.0.mlp.experts":{"quant_algo":"W4A16_NVFP4","group_size":16}}}}
    , null, &diagnostic);
    const experts = config.quant.find("model.language_model.layers.0.mlp.experts.3.up_proj").?;
    try std.testing.expectEqual(Algo.nvfp4, experts.algo);
    try std.testing.expectEqual(config_file.ScaleEncoding.csf, experts.scale_encoding);
    const qkv = config.quant.find("model.language_model.layers.0.linear_attn.in_proj_qkv").?;
    try std.testing.expectEqual(Algo.mxfp8, qkv.algo);
    try std.testing.expectEqual(@as(?u32, 32), qkv.group_size);
    try std.testing.expectEqual(config_file.ScaleEncoding.plain, config.quant.find("mtp.layers.0.mlp.experts.0.gate_proj").?.scale_encoding);
    try std.testing.expectEqual(Algo.nvfp4, config.quant.find("model.language_model.layers.1.ple.ple_embedding.ngram_embedding").?.algo);
    // FP4-Local's map: `serve` offers images on it (spec 019 WS-04); on a map of no MXFP8 linear it does not
    try std.testing.expect(config.servesImages());
    const nvfp4: Config = try .parse(arena.allocator(), fixture_head ++ fixture_quant, null, &diagnostic);
    try std.testing.expect(!nvfp4.servesImages());
}

test "n-gram constants" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const config: Config = try .parse(arena.allocator(), fixture_head ++ fixture_quant, null, &diagnostic);

    // primes from 100 on: the first table takes four, the second the next four
    const first: Ngram = try .init(arena.allocator(), config, 0);
    try std.testing.expectEqualSlices(u64, &.{ 101, 103, 107, 109 }, first.head_sizes);
    try std.testing.expectEqualSlices(u64, &.{ 0, 101, 204, 311 }, first.head_offsets);
    try std.testing.expectEqual(@as(u64, 424), first.rows);
    try std.testing.expectEqual(@as(u32, 16), first.dims);
    const second: Ngram = try .init(arena.allocator(), config, 1);
    try std.testing.expectEqualSlices(u64, &.{ 113, 127, 131, 137 }, second.head_sizes);

    for (first.multipliers, second.multipliers) |a, b| try std.testing.expect(a != b);

    // what nvidia/Qwen3.8-Flash-Next-NVFP4 ships for its one table
    var published = config;
    published.vocab = 248320;
    published.heads_per_ngram = 8;
    published.ngram_base = 20_000_000;
    published.ngram_divisor = 128;
    published.ple_dim = 2560;
    const table: Ngram = try .init(arena.allocator(), published, 0);
    try std.testing.expectEqualSlices(i64, &.{ 23703573157769, 20109073645365, 8052911324071 }, table.multipliers);
    try std.testing.expectEqual(@as(usize, 16), table.head_sizes.len);
    try std.testing.expectEqualSlices(u64, &.{ 20000003, 20000023, 20000033 }, table.head_sizes[0..3]);
    try std.testing.expectEqual(@as(u64, 20000171), table.head_sizes[15]);
    try std.testing.expectEqual(@as(u64, 300001275), table.head_offsets[15]);
    try std.testing.expectEqual(@as(u64, 320001536), table.rows);
    try std.testing.expectEqual(@as(u32, 160), table.dims);
}
