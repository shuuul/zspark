//! `config.zig`'s tests.

const std = @import("std");
const testing = std.testing;
const config_file = @import("../../../src/models/nemotron/config.zig");
const modelopt = @import("../../../src/weights/modelopt.zig");
const fixture = @import("fixture.zig");
const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;
const Error = config_file.Error;

/// The file a refusal test edits (`from` must occur once in it).
const File = enum { config, backbone, quant };

fn expectRefused(expected: Error, file: File, from: []const u8, to: []const u8, needle: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var texts = [3][]const u8{ fixture.config_text, fixture.backbone_text, fixture.quant_text };
    const at: usize = switch (file) {
        .config => 0,
        .backbone => 1,
        .quant => 2,
    };
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, texts[at], from));
    texts[at] = try std.mem.replaceOwned(u8, arena.allocator(), texts[at], from, to);
    var diagnostic: Diagnostic = .{};
    try testing.expectError(expected, Config.parse(arena.allocator(), texts[0], texts[1], texts[2], &diagnostic));
    if (std.mem.indexOf(u8, diagnostic.message, needle) == null) {
        std.debug.print("\"{s}\" is not in \"{s}\"\n", .{ needle, diagnostic.message });
        return error.TestExpectedEqual;
    }
}

test "a Nemotron config's backbone: layers by kind, Mamba-2, MoE and the MTP block" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try fixture.parse(arena.allocator(), &diagnostic);
    const b = c.backbone;
    try testing.expectEqual(@as(usize, 131072), b.vocab);
    try testing.expectEqual(@as(usize, 4), b.layers);
    try testing.expectEqual(@as(usize, 1), b.count(.mamba));
    try testing.expectEqual(@as(usize, 1), b.count(.attention));
    try testing.expectEqual(@as(usize, 2), b.count(.moe));
    try testing.expectEqual(@as(usize, 4), b.heads);
    try testing.expectEqual(@as(usize, 2), b.kv_heads);
    try testing.expectEqual(@as(usize, 64), b.head_dim);
    try testing.expectEqual(@as(usize, 256), b.mambaInner());
    try testing.expectEqual(@as(usize, 512), b.convWidth());
    try testing.expectEqual(@as(usize, 772), b.inWidth());
    try testing.expectEqual(@as(usize, 4), b.conv_kernel);
    try testing.expectEqual(@as(usize, 4), b.experts);
    try testing.expectEqual(@as(usize, 2), b.top_k);
    try testing.expectEqual(@as(usize, 32), b.moe_width);
    try testing.expectEqual(@as(usize, 64), b.shared_width);
    try testing.expectEqual(@as(f64, 2.5), b.routed_scale);
    try testing.expect(b.norm_topk and b.rescale_prenorm and b.conv_bias);
    try testing.expectEqualSlices(config_file.LayerType, &.{ .attention, .moe }, b.mtp_types);
    try testing.expectEqual(@as(usize, 1), b.mtp_layers);
    try testing.expectEqual(@as(u32, 1), b.bos);
    try testing.expectEqual(@as(u32, 2), b.eos);
    try testing.expectEqual(@as(u32, 0), b.pad);
}

test "Nemotron derives its layer count when the published config omits num_hidden_layers" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const text = try std.mem.replaceOwned(u8, arena.allocator(), fixture.backbone_text, "\"num_hidden_layers\":4,", "");
    var diagnostic: Diagnostic = .{};
    const b = try config_file.Backbone.parse(arena.allocator(), text, &diagnostic);
    try testing.expectEqual(@as(usize, 4), b.layers);
    try testing.expectEqualSlices(config_file.LayerType, &.{ .mamba, .moe, .attention, .moe }, b.types);
}

test "the speechlm config and its perception module" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try fixture.parse(arena.allocator(), &diagnostic);
    try testing.expectEqualStrings("nemotron3p5", c.prompt_format);
    try testing.expectEqualStrings("<|audio|>", c.audio_locator);
    try testing.expectEqual(@as(f64, 30.0), c.chunk_seconds);
    try testing.expect(!c.pe_encoder);
    try testing.expectEqual(@as(usize, 500), c.pe_core);
    try testing.expectEqual(@as(usize, 50), c.pe_left);
    try testing.expectEqual(@as(usize, 50), c.pe_right);
    try testing.expect(!c.pe_norm);
    try testing.expectEqualSlices(u32, &.{11}, c.eos);
    try testing.expectEqual(@as(u32, 0), c.pad);
    const e = c.perception;
    try testing.expectEqual(@as(usize, 16000), e.sample_rate);
    try testing.expectEqual(@as(usize, 32), e.features);
    try testing.expectEqual(@as(usize, 33), e.fftBins());
    try testing.expectEqual(@as(usize, 16), e.windowFrames());
    try testing.expectEqual(@as(usize, 32), e.feat_in);
    try testing.expectEqual(@as(usize, 128), e.d_model);
    try testing.expectEqual(@as(usize, 1), e.heads);
    try testing.expectEqual(@as(usize, 128), e.headDim());
    try testing.expectEqual(@as(usize, 512), e.ffWidth());
    try testing.expectEqual(@as(usize, 64), e.stacked());
    try testing.expectEqual(@as(usize, 2), e.layers);
    try testing.expect(e.qk_norm);
    try testing.expectEqual(@as(f64, 0.5), e.rotary_fraction);
}

test "the ModelOpt map: FP8 and NVFP4 modules, the MTP's names without llm., excludes" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try fixture.parse(arena.allocator(), &diagnostic);
    try testing.expectEqual(@as(usize, fixture.mapped), c.quant.layers.len);
    try testing.expectEqual(@as(usize, 6), c.quant.exclude.len);
    try testing.expectEqual(@as(?[]const u8, null), c.quant.kv_cache);
    try testing.expectEqual(modelopt.Algo.fp8, c.quant.find("backbone.layers.0.mixer.in_proj").?.algo);
    try testing.expectEqual(modelopt.Algo.nvfp4, c.quant.find("backbone.layers.2.mixer.q_proj").?.algo);
    // `mixer.experts` covers every routed expert's linear beneath it; all four attention projections map NVFP4
    try testing.expectEqual(modelopt.Algo.nvfp4, c.quant.find("backbone.layers.1.mixer.experts.3.up_proj").?.algo);
    try testing.expectEqual(modelopt.Algo.nvfp4, c.quant.find("backbone.layers.2.mixer.v_proj").?.algo);
    // the tensors' llm. prefix is not the map's
    try testing.expect(c.quant.find("llm.mtp.layers.0.eh_proj") != null);
    try testing.expect(c.quant.find("mtp.layers.0.eh_proj") != null);
    try testing.expectEqual(modelopt.Algo.fp8, c.quant.find("llm.mtp.layers.1.mixer.experts.2.down_proj").?.algo);
}

test "a Nemotron config is refused in the model's words" {
    try expectRefused(error.UnsupportedModel, .config, "\"nemo_speechlm\"", "\"qwen3_asr\"", "not Nemotron-3.5-Transcribe");
    try expectRefused(error.UnsupportedModel, .backbone, "\"nemotron_h\"", "\"nemotron_h_for_causal_lm\"", "nemotron_h");
    try expectRefused(error.UnsupportedModel, .backbone, "\"tie_word_embeddings\":false", "\"tie_word_embeddings\":true", "tied to the head");
    try expectRefused(error.UnsupportedModel, .backbone, "\"mamba_hidden_act\":\"silu\"", "\"mamba_hidden_act\":\"gelu\"", "silu");
    try expectRefused(error.UnsupportedModel, .backbone, "\"mlp_hidden_act\":\"relu2\"", "\"mlp_hidden_act\":\"swiglu\"", "relu2");
    try expectRefused(error.UnsupportedModel, .backbone, "\"mamba_ssm_cache_dtype\":\"float32\"", "\"mamba_ssm_cache_dtype\":\"bfloat16\"", "fp32 state");
    try expectRefused(error.UnsupportedModel, .backbone, "\"moe_latent_size\":null", "\"moe_latent_size\":128", "moe_latent_size");
    try expectRefused(error.UnsupportedModel, .backbone, "\"n_shared_experts\":1", "\"n_shared_experts\":2", "one shared expert");
    try expectRefused(error.UnsupportedModel, .backbone, "\"num_nextn_predict_layers\":1", "\"num_nextn_predict_layers\":2", "one");
    try expectRefused(error.UnsupportedModel, .backbone, "\"mamba\",\"moe\",\"attention\",\"moe\"", "\"mamba\",\"moe\",\"kan\",\"moe\"", "\"kan\"");
    try expectRefused(error.InvalidConfig, .backbone, "\"num_hidden_layers\":4", "\"num_hidden_layers\":5", "num_hidden_layers 5");
    try expectRefused(error.InvalidConfig, .backbone, "\"num_key_value_heads\":2", "\"num_key_value_heads\":3", "KV heads");
    try expectRefused(error.InvalidConfig, .backbone, "\"num_experts_per_tok\":2", "\"num_experts_per_tok\":9", "9 experts a token of 4");
    try expectRefused(error.InvalidConfig, .backbone, "\"hidden_size\":128", "\"hidden_size\":60", "blocks of 16");
    try expectRefused(error.UnsupportedModel, .config, "\"window\":\"hann\"", "\"window\":\"hamming\"", "hann");
    try expectRefused(error.UnsupportedModel, .config, "\"attn_mode\":\"full\"", "\"attn_mode\":\"rel_pos\"", "full attention");
    try expectRefused(error.UnsupportedModel, .config, "\"self_attention_model\":\"rope\"", "\"self_attention_model\":\"abs\"", "RoPE");
    try expectRefused(error.UnsupportedModel, .config, "\"output_dim\":128", "\"output_dim\":32", "projects to 32");
    try expectRefused(error.InvalidConfig, .config, "\"n_heads\":1", "\"n_heads\":3", "heads");
    try expectRefused(error.UnsupportedModel, .quant, "\"MIXED_PRECISION\"", "\"FP8\"", "MIXED_PRECISION");
    try expectRefused(error.UnsupportedModel, .quant, "\"backbone.layers.2.mixer.q_proj\":{\"quant_algo\":\"NVFP4\",\"group_size\":16}", "\"backbone.layers.2.mixer.q_proj\":{\"quant_algo\":\"NVFP4\",\"group_size\":32}", "NVFP4 in groups of 16");
    try expectRefused(error.UnsupportedModel, .quant, "\"backbone.layers.2.mixer.q_proj\":{\"quant_algo\":\"NVFP4\",\"group_size\":16}", "\"backbone.layers.2.mixer.q_proj\":{\"quant_algo\":\"MXFP8\",\"weight_block_size\":[1,32]}", "NVFP4 in groups of 16");
    try expectRefused(error.InvalidConfig, .quant, "\"quantized_layers\"", "\"mapped\"", "quantized_layers");
}

test "Mamba inference timestep limits override the parameter initialization range" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cases = [_]struct { json: []const u8, low: f64, high: f64 }{
        .{ .json = "[0.0,{\"__float__\":\"Infinity\"}]", .low = 0, .high = std.math.inf(f64) },
        .{ .json = "[0.02,0.8]", .low = 0.02, .high = 0.8 },
    };
    for (cases) |case| {
        const entry = try std.fmt.allocPrint(arena, "\"time_step_limit\":{s},\"time_step_min\":0.001", .{case.json});
        const text = try std.mem.replaceOwned(u8, arena, fixture.backbone_text, "\"time_step_min\":0.001", entry);
        var diagnostic: Diagnostic = .{};
        const b = try config_file.Backbone.parse(arena, text, &diagnostic);
        try testing.expectEqual(case.low, b.dt_min);
        try testing.expectEqual(case.high, b.dt_max);
    }
    for ([_][]const u8{ "[0.8,0.02]", "[-1,1]", "[0]", "[0,\"invalid\"]" }) |value| {
        const entry = try std.fmt.allocPrint(arena, "\"time_step_limit\":{s},\"time_step_min\":0.001", .{value});
        const text = try std.mem.replaceOwned(u8, arena, fixture.backbone_text, "\"time_step_min\":0.001", entry);
        var diagnostic: Diagnostic = .{};
        try testing.expectError(error.InvalidConfig, config_file.Backbone.parse(arena, text, &diagnostic));
    }
}
