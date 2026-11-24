//! A small Nemotron-3.5-Transcribe checkpoint, weightless (spec 151 WS-01): the three config files' texts at
//! a four-layer size (one layer of each `layers_block_type` kind, the two-layer MTP block, two encoder
//! layers), the quant map as `hf_quant_config.json` writes it, and a `safetensors.Checkpoint` of synthetic
//! tensors under the real names, FP8 and NVFP4 as the map stores them. What `config_test.zig` and
//! `layout_test.zig` read.

const std = @import("std");
const safetensors = @import("../../../src/weights/safetensors.zig");
const modelopt = @import("../../../src/weights/modelopt.zig");
const config_file = @import("../../../src/models/nemotron/config.zig");
const layout_file = @import("../../../src/models/nemotron/layout.zig");

/// `llm_backbone/config.json`: one layer of each kind, a hidden of 128, four heads of 64 over two KV heads,
/// Mamba-2 of four heads of 64 and one group, four experts of 32 with a shared of 64.
pub const backbone_text =
    \\{"model_type":"nemotron_h","architectures":["NemotronHForCausalLM"],
    \\"hidden_size":128,"vocab_size":131072,"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,
    \\"head_dim":64,"rope_theta":10000,"partial_rotary_factor":1.0,"max_position_embeddings":8192,
    \\"layer_norm_epsilon":1e-05,"layers_block_type":["mamba","moe","attention","moe"],
    \\"expand":2,"mamba_num_heads":4,"mamba_head_dim":64,"n_groups":1,"ssm_state_size":128,"conv_kernel":4,
    \\"chunk_size":8,"mamba_hidden_act":"silu","use_conv_bias":true,"mamba_proj_bias":false,
    \\"time_step_min":0.001,"time_step_max":0.1,"mamba_ssm_cache_dtype":"float32",
    \\"n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,"moe_latent_size":null,
    \\"moe_intermediate_size":32,"moe_shared_expert_intermediate_size":64,"mlp_hidden_act":"relu2",
    \\"routed_scaling_factor":2.5,"norm_topk_prob":true,"rescale_prenorm_residual":true,"mlp_bias":false,
    \\"attention_bias":false,"tie_word_embeddings":false,
    \\"num_nextn_predict_layers":1,"mtp_layers_block_type":["attention","moe"],
    \\"bos_token_id":1,"eos_token_id":2,"pad_token_id":0}
;

/// `hf_quant_config.json`: the Mamba projections, the MTP block and the encoder's linears FP8, the routed and
/// shared experts and the attention layer's projections NVFP4 in groups of 16 (as the checkpoint stores them).
pub const quant_text =
    \\{"producer":{"name":"modelopt"},"quantization":{"quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":null,
    \\"quantized_layers":{
    \\"backbone.layers.0.mixer.in_proj":{"quant_algo":"FP8"},
    \\"backbone.layers.0.mixer.out_proj":{"quant_algo":"FP8"},
    \\"backbone.layers.1.mixer.experts":{"quant_algo":"NVFP4","group_size":16},
    \\"backbone.layers.1.mixer.shared_experts.up_proj":{"quant_algo":"NVFP4","group_size":16},
    \\"backbone.layers.1.mixer.shared_experts.down_proj":{"quant_algo":"NVFP4","group_size":16},
    \\"backbone.layers.2.mixer.q_proj":{"quant_algo":"NVFP4","group_size":16},
    \\"backbone.layers.2.mixer.k_proj":{"quant_algo":"NVFP4","group_size":16},
    \\"backbone.layers.2.mixer.v_proj":{"quant_algo":"NVFP4","group_size":16},
    \\"backbone.layers.2.mixer.o_proj":{"quant_algo":"NVFP4","group_size":16},
    \\"backbone.layers.3.mixer.experts":{"quant_algo":"NVFP4","group_size":16},
    \\"backbone.layers.3.mixer.shared_experts.up_proj":{"quant_algo":"NVFP4","group_size":16},
    \\"backbone.layers.3.mixer.shared_experts.down_proj":{"quant_algo":"NVFP4","group_size":16},
    \\"mtp.layers.0.eh_proj":{"quant_algo":"FP8"},
    \\"mtp.layers.0.mixer.q_proj":{"quant_algo":"FP8"},
    \\"mtp.layers.0.mixer.k_proj":{"quant_algo":"FP8"},
    \\"mtp.layers.0.mixer.v_proj":{"quant_algo":"FP8"},
    \\"mtp.layers.0.mixer.o_proj":{"quant_algo":"FP8"},
    \\"mtp.layers.1.mixer.experts":{"quant_algo":"FP8"},
    \\"mtp.layers.1.mixer.shared_experts.up_proj":{"quant_algo":"FP8"},
    \\"mtp.layers.1.mixer.shared_experts.down_proj":{"quant_algo":"FP8"},
    \\"perception.encoder.asr_encoder.layers.0.attn.w_qkv":{"quant_algo":"FP8"},
    \\"perception.encoder.asr_encoder.layers.0.attn.out_proj":{"quant_algo":"FP8"},
    \\"perception.encoder.asr_encoder.layers.0.ffn.net.0":{"quant_algo":"FP8"},
    \\"perception.encoder.asr_encoder.layers.0.ffn.net.3":{"quant_algo":"FP8"},
    \\"perception.encoder.asr_encoder.layers.1.attn.w_qkv":{"quant_algo":"FP8"},
    \\"perception.encoder.asr_encoder.layers.1.attn.out_proj":{"quant_algo":"FP8"},
    \\"perception.encoder.asr_encoder.layers.1.ffn.net.0":{"quant_algo":"FP8"},
    \\"perception.encoder.asr_encoder.layers.1.ffn.net.3":{"quant_algo":"FP8"}},
    \\"exclude_modules":["backbone.embeddings","backbone.layers.0.mixer.conv1d","backbone.layers.1.mixer.gate","lm_head","mtp.layers.1.mixer.gate","ctc_timestamp*"]}}
;

/// config.json: the speechlm part and `perception` (an encoder of two layers, d_model 128 over one head of 128).
pub const config_text =
    \\{"model_type":"nemo_speechlm","architectures":["Nemotron3_5TranscribeForConditionalGeneration"],
    \\"prompt_format":"nemotron3p5","audio_locator_tag":"<|audio|>","encoder_chunk_size_seconds":30.0,
    \\"eos_token_id":[11],"pad_token_id":0,
    \\"perception":{"output_dim":128,
    \\"preprocessor":{"sample_rate":16000,"window_size":0.001,"window_stride":0.01,"window":"hann",
    \\"features":32,"n_fft":64,"log":true,"dither":1e-05},
    \\"encoder":{"feat_in":32,"d_model":128,"n_heads":1,"n_layers":2,"qkv_bias":false,"qk_norm":true,
    \\"ff_expansion":4,"subsampling_factor":2,"attn_mode":"full","self_attention_model":"rope",
    \\"rope_base":10000.0,"rotary_fraction":0.5}}}
;

/// The whole config (the three files' texts).
pub fn parse(arena: std.mem.Allocator, diagnostic: *config_file.Diagnostic) config_file.Error!config_file.Config {
    return config_file.Config.parse(arena, config_text, backbone_text, quant_text, diagnostic);
}

/// The linears by storage as `quant_text` makes them, and its entries' count.
pub const nvfp4 = 24;
pub const fp8 = 25;
pub const bf16 = 0;
pub const mapped = 28;
fn put(arena: std.mem.Allocator, checkpoint: *safetensors.Checkpoint, random: std.Random, name: []const u8, dtype: safetensors.DType, shape: []const u64, zero: bool) !void {
    var tensor: safetensors.Tensor = .{ .dtype = dtype, .shape = try arena.dupe(u64, shape), .data = &.{} };
    const data = try arena.alloc(u8, @intCast(tensor.count() * dtype.size()));
    if (zero) @memset(data, 0) else random.bytes(data);
    tensor.data = data;
    try checkpoint.tensors.put(arena, try arena.dupe(u8, name), tensor);
}

/// The linear `module` of [rows, cols] as `layer` stores it (BF16, FP8 or NVFP4 with plain scales here).
fn putLinear(arena: std.mem.Allocator, checkpoint: *safetensors.Checkpoint, random: std.Random, module: []const u8, layer: ?modelopt.QuantizedLayer, rows: u64, cols: u64) !void {
    var name: [256]u8 = undefined;
    const at = struct {
        fn of(buffer: []u8, m: []const u8, leaf: []const u8) []const u8 {
            return std.fmt.bufPrint(buffer, "{s}.{s}", .{ m, leaf }) catch unreachable;
        }
    }.of;
    switch (modelopt.Scheme.of(layer).?) {
        .bf16 => try put(arena, checkpoint, random, at(&name, module, "weight"), .BF16, &.{ rows, cols }, false),
        .fp8 => {
            try put(arena, checkpoint, random, at(&name, module, "weight"), .F8_E4M3, &.{ rows, cols }, false);
            // a fused stack (q, k, v) shares the tensor scale as ModelOpt exports it: one fixed value here
            const scale_2 = try arena.dupe(u8, &.{ 0, 0, 0x80, 0x3F });
            try checkpoint.tensors.put(arena, try arena.dupe(u8, at(&name, module, "weight_scale")), .{ .dtype = .F32, .shape = &.{}, .data = scale_2 });
            try checkpoint.tensors.put(arena, try arena.dupe(u8, at(&name, module, "input_scale")), .{ .dtype = .F32, .shape = &.{}, .data = try arena.dupe(u8, scale_2) });
        },
        .nvfp4 => {
            try put(arena, checkpoint, random, at(&name, module, "weight"), .U8, &.{ rows, cols / 2 }, false);
            // a fused stack (q, k, v) shares the tensor scale as ModelOpt exports it: one fixed value here
            const scale_2 = try arena.dupe(u8, &.{ 0, 0, 0x80, 0x3F });
            try checkpoint.tensors.put(arena, try arena.dupe(u8, at(&name, module, "weight_scale_2")), .{ .dtype = .F32, .shape = &.{}, .data = scale_2 });
            try put(arena, checkpoint, random, at(&name, module, "weight_scale"), .F8_E4M3, &.{ rows, cols / 16 }, false);
            try checkpoint.tensors.put(arena, try arena.dupe(u8, at(&name, module, "input_scale")), .{ .dtype = .F32, .shape = &.{}, .data = try arena.dupe(u8, scale_2) });
        },
        else => unreachable,
    }
}

/// A checkpoint of every tensor `c` asks for, its linears as the map stores them.
pub fn synthetic(arena: std.mem.Allocator, c: config_file.Config, seed: u64) !safetensors.Checkpoint {
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    var checkpoint: safetensors.Checkpoint = .{ .shards = &.{}, .tensors = .empty, .gpa = arena };
    var name: [256]u8 = undefined;
    for (layout_file.kinds) |one| {
        switch (one.where.stem()) {
            .whole => if (!one.diar() or c.diar != null) try putKind(arena, &checkpoint, random, c, one, one.leaf),
            .backbone => for (0..c.backbone.types.len) |layer| {
                if (!layout_file.holds(c, one.where, layer)) continue;
                if (one.where != .expert) {
                    try putKind(arena, &checkpoint, random, c, one, std.fmt.bufPrint(&name, layout_file.backbone_prefix ++ "{d}.{s}", .{ layer, one.leaf }) catch unreachable);
                } else for (0..c.backbone.experts) |expert| {
                    try putKind(arena, &checkpoint, random, c, one, std.fmt.bufPrint(&name, layout_file.backbone_prefix ++ "{d}.mixer.experts.{d}.{s}", .{ layer, expert, one.leaf }) catch unreachable);
                }
            },
            .mtp => for (0..c.backbone.mtp_types.len) |layer| {
                if (!layout_file.holds(c, one.where, layer)) continue;
                if (one.where != .mtp_expert) {
                    try putKind(arena, &checkpoint, random, c, one, std.fmt.bufPrint(&name, layout_file.mtp_prefix ++ "{d}.{s}", .{ layer, one.leaf }) catch unreachable);
                } else for (0..c.backbone.experts) |expert| {
                    try putKind(arena, &checkpoint, random, c, one, std.fmt.bufPrint(&name, layout_file.mtp_prefix ++ "{d}.mixer.experts.{d}.{s}", .{ layer, expert, one.leaf }) catch unreachable);
                }
            },
            .encoder => for (0..c.perception.layers) |layer| {
                try putKind(arena, &checkpoint, random, c, one, std.fmt.bufPrint(&name, layout_file.encoder_prefix ++ "{d}.{s}", .{ layer, one.leaf }) catch unreachable);
            },
            .diar => if (c.diar) |dd| for (0..dd.layers) |layer| {
                try putKind(arena, &checkpoint, random, c, one, std.fmt.bufPrint(&name, layout_file.diar_prefix ++ "{d}.{s}", .{ layer, one.leaf }) catch unreachable);
            },
        }
    }
    return checkpoint;
}

fn putKind(arena: std.mem.Allocator, checkpoint: *safetensors.Checkpoint, random: std.Random, c: config_file.Config, one: layout_file.Kind, name: []const u8) !void {
    var dims: [3]u64 = undefined;
    const shape = one.shape(c, &dims);
    if (!one.linear) return put(arena, checkpoint, random, name, one.dtype, shape, false);
    const module = name[0 .. name.len - ".weight".len];
    try putLinear(arena, checkpoint, random, module, c.quant.find(module), shape[0], shape[1]);
}

/// Rewrites a `synthetic` checkpoint's bytes into values a forward survives: bf16 tensors uniform in
/// [-0.25, 0.25], F32 scalars 1.0, FP8 codes e4m3 of uniform values in [-0.5, 0.5] (random bytes hold NaNs).
/// LayerNorm weights are set near 1.0 so the encoder's activations keep a sane scale, and the featurizer's
/// `fb` non-negative so its log-mel is real (a mel table's weights never are).
pub fn fill(checkpoint: *safetensors.Checkpoint, seed: u64) void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    var it = checkpoint.tensors.iterator();
    while (it.next()) |entry| {
        const t = entry.value_ptr;
        const name = entry.key_ptr.*;
        switch (t.dtype) {
            .BF16 => {
                const values = std.mem.bytesAsSlice(u16, @constCast(t.data));
                const is_norm = std.mem.endsWith(u8, name, ".weight") and (std.mem.indexOf(u8, name, "norm") != null or std.mem.indexOf(u8, name, "embed_norm") != null);
                const is_fb = std.mem.endsWith(u8, name, "featurizer.fb");
                for (values[0..]) |*v| v.* = bf16Bits(if (is_fb) 0.25 * random.float(f32) else if (is_norm) 0.8 + 0.4 * random.float(f32) else 0.5 * random.float(f32) - 0.25);
            },
            .F32 => {
                const w = std.mem.bytesAsSlice(u32, @constCast(t.data));
                for (w[0..]) |*v| v.* = @as(u32, @bitCast(@as(f32, 1.0)));
            },
            .F8_E4M3 => {
                const w: []u8 = @constCast(t.data);
                for (w[0..]) |*v| v.* = e4m3Bits(1.0 * random.float(f32) - 0.5);
            },
            else => {},
        }
    }
}

/// Round-to-nearest-even f32 -> bf16 bits.
pub fn bf16Bits(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

/// f32 -> the nearest e4m3 code (round half to even on the 3-bit mantissa; enough for test weights).
pub fn e4m3Bits(x: f32) u8 {
    const sign: u8 = if (x < 0) 0x80 else 0;
    const a = @abs(x);
    if (a == 0) return sign;
    const e = std.math.log2(a);
    var exp: i32 = @intFromFloat(@floor(e));
    var mant = a / std.math.pow(f32, 2, @floatFromInt(exp)) - 1.0;
    if (mant > 0.9375) {
        exp += 1;
        mant = 0;
    }
    exp = @max(-6, @min(8, exp));
    var code: u8 = sign | (@as(u8, @intCast(exp + 7)) << 3) | @as(u8, @intFromFloat(@round(mant * 8)));
    if ((code & 0x7F) > 0x7E) code = sign | 0x7E; // 0x7F and 0xFF are NaN in e4m3
    return code;
}

/// An e4m3 code's value (for the host reference).
pub fn e4m3Value(code: u8) f32 {
    const sign: f32 = if (code & 0x80 != 0) -1 else 1;
    const exp: i32 = @as(i32, (code >> 3) & 0xF) - 7;
    const mant: f32 = @floatFromInt(code & 7);
    if (exp == -7) return sign * mant * std.math.pow(f32, 2, -9);
    return sign * (1 + mant / 8) * std.math.pow(f32, 2, @floatFromInt(exp));
}

/// config.json with a ParallelExpertEncoder (`pe_encoder_config`, spec 151 WS-11): the defaults the
/// bundle's PEE writes (`online_inference_length` 100 here, the contexts 6 each so a test clip spans
/// several windows), a two-layer Sortformer of d_model 128 over one head, `tf_d_model` 64, eight
/// speakers, the streaming knobs small enough that a short clip exercises the spkcache compression.
pub const config_text_pe =
    \\{"model_type":"nemo_speechlm","architectures":["Nemotron3_5TranscribeForConditionalGeneration"],
    \\"prompt_format":"nemotron3p5","audio_locator_tag":"<|audio|>","encoder_chunk_size_seconds":30.0,
    \\"eos_token_id":[11],"pad_token_id":0,
    \\"perception":{"output_dim":128,
    \\"preprocessor":{"sample_rate":16000,"window_size":0.001,"window_stride":0.01,"window":"hann",
    \\"features":32,"n_fft":64,"log":true,"dither":1e-05},
    \\"encoder":{"feat_in":32,"d_model":128,"n_heads":1,"n_layers":2,"qkv_bias":false,"qk_norm":true,
    \\"ff_expansion":4,"subsampling_factor":2,"attn_mode":"full","self_attention_model":"rope",
    \\"rope_base":10000.0,"rotary_fraction":0.5}},
    \\"pe_encoder_config":{"asr_encoder_type":"transformer","asr_normalize_type":"per_feature",
    \\"online_inference_length":100,"chunk_left_context":6,"chunk_right_context":6,
    \\"speaker_activity_threshold":0.5,"spk_kernel_scale":1.0,
    \\"diar_fifo_len":6,"diar_spkcache_len":8,"diar_spkcache_update_period":4,
    \\"diarization_model_cfg":{"high_resolution":true,"output_subsampling_factor":1,
    \\"encoder":{"feat_in":32,"d_model":128,"n_heads":1,"n_layers":2,"qkv_bias":false,"qk_norm":false,
    \\"ff_expansion":4,"subsampling_factor":2,"attn_mode":"full","self_attention_model":"rope",
    \\"rope_base":10000.0,"rotary_fraction":1.0,"subsampling":"feature_stacking"},
    \\"sortformer_modules":{"fc_d_model":128,"tf_d_model":64,"num_spks":8,
    \\"use_learnable_sil_emb":true,"spkcache_len":8,"spkcache_sil_frames_per_spk":1,
    \\"spkcache_update_period":4,"fifo_len":6,"pred_score_threshold":0.25,
    \\"scores_boost_latest":0.05,"sil_threshold":0.2,"strong_boost_rate":0.75,
    \\"weak_boost_rate":1.5,"min_pos_scores_rate":0.5}}}}
;
/// Rewrites a `synthetic` checkpoint's bytes so the MTP head's draft is the backbone's own choice
/// (`fill` first, for sane values): every projection and expert zeroed (a zeroed weight's product is 0
/// whatever its scales), every embedding row the same positive vector and `lm_head`'s row 0 dominant, so
/// each row's logits argmax id 0 — and the draft head's, its input rows then all zero, argmaxes id 0 too.
pub fn agreeable(checkpoint: *safetensors.Checkpoint) void {
    var it = checkpoint.tensors.iterator();
    while (it.next()) |entry| {
        const t = entry.value_ptr;
        const name = entry.key_ptr.*;
        switch (t.dtype) {
            .F8_E4M3, .U8 => @memset(@constCast(t.data), 0),
            .BF16 => {
                const values = std.mem.bytesAsSlice(u16, @constCast(t.data));
                if (std.mem.eql(u8, name, "backbone.embeddings.weight")) {
                    for (values) |*v| v.* = bf16Bits(0.5);
                } else if (std.mem.eql(u8, name, "lm_head.weight")) {
                    for (values[0..t.shape[1]]) |*v| v.* = bf16Bits(4.0);
                }
            },
            else => {},
        }
    }
}
