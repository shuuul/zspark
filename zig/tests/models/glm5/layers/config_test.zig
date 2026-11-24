//! `config.zig`'s tests.

const std = @import("std");
const testing = std.testing;
const config_file = @import("../../../../src/models/glm5/layers/config.zig");
const fixtures = @import("../../../../src/models/glm5/checks/fixtures.zig");
const modelopt_fixture = @import("modelopt_fixture.zig");
const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;
const Error = config_file.Error;

const fixture = fixtures.config;
const fixture_quant = fixtures.config_quant;

/// The fixture with `from` (once in it) replaced by `to` is refused with `expected` and a message holding `needle`.
fn expectRefused(expected: Error, from: []const u8, to: []const u8, needle: []const u8) !void {
    return expectRefusedIn(fixture, expected, from, to, needle);
}

fn expectRefusedIn(base: []const u8, expected: Error, from: []const u8, to: []const u8, needle: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, base, from));
    const text = try std.mem.replaceOwned(u8, arena.allocator(), base, from, to);
    var diagnostic: Diagnostic = .{};
    try testing.expectError(expected, Config.parse(arena.allocator(), text, &diagnostic));
    if (std.mem.indexOf(u8, diagnostic.message, needle) == null) {
        std.debug.print("\"{s}\" is not in \"{s}\"\n", .{ needle, diagnostic.message });
        return error.TestExpectedEqual;
    }
}

test "a GLM config's fields, its layers by kind and a site's mix dots" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try Config.parse(arena.allocator(), fixture, &diagnostic);
    try testing.expectEqual(@as(usize, 64), c.hidden);
    try testing.expectEqual(@as(usize, 4), c.layers);
    try testing.expectEqual(@as(usize, 3), c.count(.linear_attention));
    try testing.expectEqual(@as(usize, 1), c.count(.deepseek_sparse_attention));
    try testing.expectEqual(@as(usize, 3), c.countMlp(.dense));
    try testing.expectEqual(@as(usize, 1), c.countMlp(.sparse));
    try testing.expectEqual(@as(usize, 24), c.mix());
    try testing.expectEqualSlices(u32, &.{ 9, 7 }, c.eos);
    try testing.expectEqual(@as(f64, -5.0), c.gate_lower_bound);
    try testing.expectEqual(@as(f64, 2.5), c.routed_scale);
    try testing.expect(c.norm_topk);
    const plain = try std.mem.replaceOwned(u8, arena.allocator(), fixture, "\"swiglu_limit\":10.0,", "\"swiglu_limit\":10.0,\"norm_topk_prob\":false,");
    try testing.expect(!(try Config.parse(arena.allocator(), plain, &diagnostic)).norm_topk);
    try testing.expectEqual(@as(usize, 1), c.mtp_layers);
    try testing.expectEqual(@as(usize, 2), c.quant.layers.len);
    try testing.expect(c.quant.find("model.language_model.layers.4.mlp.experts.1.up_proj").?.a16);
    try testing.expect(c.index_tail);
}

test "a GLM config is refused in the model's words: another model, no ModelOpt block, positions, an odd split" {
    try expectRefused(error.UnsupportedModel, "\"glm5_next\"", "\"qwen4_exp\"", "not GLM-5.3-Flash");
    try expectRefused(error.UnsupportedQuantization, fixture_quant, "}", "names no ModelOpt quantization");
    try expectRefused(error.UnsupportedModel, "\"qk_rope_head_dim\":0", "\"qk_rope_head_dim\":64", "no positions");
    try expectRefused(error.UnsupportedModel, "\"n_shared_experts\":1", "\"n_shared_experts\":2", "one shared expert");
    try expectRefused(error.UnsupportedModel, "\"sigmoid\"", "\"softmax\"", "scores with a sigmoid");
    try expectRefused(error.UnsupportedModel, "\"num_nextn_predict_layers\":1", "\"num_nextn_predict_layers\":2", "2 MTP layers");
    try expectRefused(error.InvalidConfig, "\"num_hidden_layers\":4", "\"num_hidden_layers\":5", "num_hidden_layers 5");
    try expectRefused(error.UnsupportedModel, "\"dense\",\"sparse\"", "\"dense\",\"kan\"", "\"kan\"");
    try expectRefused(error.InvalidConfig, "\"num_attention_heads\":4", "\"num_attention_heads\":3", "half of the heads");
    try expectRefused(error.InvalidConfig, "\"num_experts_per_tok\":2", "\"num_experts_per_tok\":7", "7 experts a token of 6");
    try expectRefused(error.InvalidConfig, "\"index_topk\":8", "\"index_topk\":6", "not whole pools of 4");
    try expectRefused(error.InvalidConfig, "\"text_config\"", "\"text\"", "no text_config");
    try expectRefused(error.InvalidConfig, "\"hc_mult\":4,", "", "no hc_mult");
    try expectRefused(error.UnsupportedModel, "\"hc_mult\":4,", "\"hc_mult\":2,", "four streams");
    try expectRefused(error.UnsupportedModel, "\"head_dim\":128,", "\"head_dim\":64,", "heads of 128");
    try expectRefused(error.UnsupportedModel, "\"kv_lora_rank\":512", "\"kv_lora_rank\":256", "latents of 512");
    try expectRefused(error.UnsupportedModel, "\"short_conv_kernel_size\":4", "\"short_conv_kernel_size\":3", "four taps");
    try expectRefused(error.InvalidConfig, "\"swiglu_limit\":10.0,", "\"swiglu_limit\":10.0,\"norm_topk_prob\":1,", "norm_topk_prob");
}

test "a ModelOpt config: its map read, an entry covering a module; another storage refused in the model's words" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try Config.parse(arena.allocator(), modelopt_fixture.config, &diagnostic);
    const map = c.quant;
    try testing.expectEqual(@as(usize, modelopt_fixture.mapped + 1), map.layers.len);
    const experts = map.find("model.language_model.layers.3.mlp.experts.5.down_proj").?;
    try testing.expectEqual(.nvfp4, experts.algo);
    try testing.expectEqual(.csf32, experts.scale_encoding);
    try testing.expectEqual(@as(?u32, 16), experts.group_size);
    try testing.expectEqual(.plain, map.find("model.language_model.layers.4.mlp.experts.0.up_proj").?.scale_encoding);
    try testing.expectEqual(@as(?u32, 32), map.find("lm_head").?.group_size);
    try testing.expect(map.find("model.language_model.layers.3.mlp.gate") == null);

    const base = modelopt_fixture.config;
    const head = "\"lm_head\":{\"quant_algo\":\"MXFP8\",\"weight_block_size\":[1,32]}";
    try expectRefusedIn(base, error.UnsupportedQuantization, "\"quant_algo\":\"MIXED_PRECISION\"", "\"quant_algo\":\"NVFP4\"", "NVFP4 throughout");
    try expectRefusedIn(base, error.UnsupportedQuantization, head, "\"lm_head\":{\"quant_algo\":\"FP8\"}", "lm_head is stored as");
    try expectRefusedIn(base, error.UnsupportedQuantization, head, "\"lm_head\":{\"quant_algo\":\"NVFP4\",\"group_size\":16}", "in the routed experts");
    try expectRefusedIn(base, error.UnsupportedQuantization, head, "\"lm_head\":{\"quant_algo\":\"MXFP8\",\"group_size\":32,\"weight_scale_encoding\":\"csf_32x2_v1\"}", "lm_head is stored as");
    try expectRefusedIn(base, error.UnsupportedQuantization, head, "\"lm_head\":{\"quant_algo\":\"INT4_AWQ\"}", "lm_head is stored as");
    try expectRefusedIn(base, error.UnsupportedQuantization, "\"csf_32x2_v1\"", "\"csf\"", "in the routed experts");
    try expectRefusedIn(base, error.InvalidConfig, head, "\"lm_head\":{\"group_size\":32}", "no ModelOpt entry");
    try expectRefusedIn(base, error.UnsupportedQuantization, "\"quant_method\":\"modelopt\"", "\"quant_method\":\"awq\"", "NVIDIA's FP4-Local");
}
