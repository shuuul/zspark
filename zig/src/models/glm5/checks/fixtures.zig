//! What the host's tests of this directory and the random model (`random.zig`) are made from, with no device:
//! a four-layer `config.json`, tensors by name as a checkpoint answers, a random NVFP4 linear under a name.
//! Device helpers are `testing.zig`'s.

const std = @import("std");
const numeric = @import("../../../weights/numeric.zig");
const safetensors = @import("../../../weights/safetensors.zig");

/// A four-layer config with every field the parser reads, in three parts for a test to change one.
pub const config_head =
    \\{"model_type":"glm5_next","text_config":{"hidden_size":64,"vocab_size":100,"num_hidden_layers":4,
    \\"layer_types":["linear_attention","linear_attention","linear_attention","deepseek_sparse_attention"],
    \\"mlp_layer_types":["dense","dense","dense","sparse"],"rms_norm_eps":1e-05,"eos_token_id":[9,7],
    \\"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,
    \\"linear_attn_config":{"num_heads":4,"head_dim":128,"short_conv_kernel_size":4,"gate_lower_bound":-5.0},
    \\"num_attention_heads":4,"q_lora_rank":24,"kv_lora_rank":512,"qk_nope_head_dim":16,"v_head_dim":16,
;
pub const config_rest =
    \\"qk_rope_head_dim":0,"n_shared_experts":1,"scoring_func":"sigmoid",
    \\"index_n_heads":2,"index_head_dim":16,"index_topk":8,"index_kpool":4,"index_kpool_always_select_tail":true,
    \\"intermediate_size":96,"n_routed_experts":6,"num_experts_per_tok":2,"moe_intermediate_size":32,
    \\"routed_scaling_factor":2.5,"swiglu_limit":10.0,"num_nextn_predict_layers":1}
;
/// Layer 3's routed experts NVFP4 (W4A4), the MTP layer's W4A16_NVFP4, every other linear BF16.
pub const config_quant = quantOf(&.{"3"}, "4");
pub const config = config_head ++ config_rest ++ config_quant;

/// Tensors by name, as a checkpoint answers `get`.
pub const Source = std.StringHashMapUnmanaged(safetensors.Tensor);

/// A ModelOpt MIXED_PRECISION block whose map names the routed experts of `moe` NVFP4 (W4A4, as FP4-Local's
/// layers 3 to 44) and those of `mtp` W4A16_NVFP4 (its MTP layer's), every other linear BF16.
pub fn quantOf(comptime moe: []const []const u8, comptime mtp: []const u8) []const u8 {
    comptime var map: []const u8 = "";
    inline for (moe) |layer| map = map ++ "\"model.language_model.layers." ++ layer ++ ".mlp.experts\":{\"group_size\":16,\"quant_algo\":\"NVFP4\"},";
    map = map ++ "\"model.language_model.layers." ++ mtp ++ ".mlp.experts\":{\"group_size\":16,\"quant_algo\":\"W4A16_NVFP4\"}";
    return ",\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"MIXED_PRECISION\",\"exclude_modules\":[],\"quantized_layers\":{" ++ map ++ "}}}";
}

/// A random NVFP4 linear under `prefix` in `source`, as FP4-Local stores a routed expert with plain scales: K
/// inputs, N outputs, codes of every value, e4m3 block scales of 0.5 to 1.5 and one `weight_scale_2` that give
/// the weights a spread near `spread`, and an `input_scale` for activations of magnitude to about 4.
pub fn putNvfp4(arena: std.mem.Allocator, random: std.Random, source: *Source, prefix: []const u8, k: usize, n: usize, spread: f64) !void {
    const codes = try arena.alloc(u8, n * k / 2);
    random.bytes(codes);
    try source.put(arena, try std.mem.concat(arena, u8, &.{ prefix, ".weight" }), .{ .dtype = .U8, .shape = try arena.dupe(u64, &.{ n, k / 2 }), .data = codes });
    const scales = try arena.alloc(u8, n * k / 16);
    for (scales) |*one| one.* = numeric.e4m3FromF32(@floatCast(0.5 + random.float(f64)));
    try source.put(arena, try std.mem.concat(arena, u8, &.{ prefix, ".weight_scale" }), .{ .dtype = .F8_E4M3, .shape = try arena.dupe(u64, &.{ n, k / 16 }), .data = scales });
    // an e2m1 code of uniform bits has an rms of 2.93
    const global = try arena.alloc(f32, 1);
    global[0] = @floatCast(spread / 2.93);
    try source.put(arena, try std.mem.concat(arena, u8, &.{ prefix, ".weight_scale_2" }), .{ .dtype = .F32, .shape = &.{}, .data = std.mem.sliceAsBytes(global) });
    const input = try arena.alloc(f32, 1);
    input[0] = @floatCast(4.0 / (448.0 * 6.0) * (0.75 + 0.5 * random.float(f64)));
    try source.put(arena, try std.mem.concat(arena, u8, &.{ prefix, ".input_scale" }), .{ .dtype = .F32, .shape = &.{}, .data = std.mem.sliceAsBytes(input) });
}
