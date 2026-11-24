//! Nemotron-3.5-Transcribe-30B-A4B-NVFP4's tensors by name, dtype and shape from the config (`config.zig`):
//! the checkpoint manifest (spec 151 WS-01). The backbone shards hold `backbone.embeddings`, `backbone.
//! layers.N.*` by the layer's `layers_block_type` kind, `backbone.norm_f` and `lm_head`; `model-mtp-fp8`
//! the MTP block's `llm.mtp.layers.*` (the quant map's `mtp.layers.*`); `perception.safetensors` the mel
//! featurizer, the 32-layer encoder and the projection, the Sortformer diarization encoder the PEE's
//! speaker infusion reads (WS-11), and the CTC heads beside them, counted and not read (`dropped`). A
//! linear's storage is `hf_quant_config.json`'s map (`weights/modelopt.zig`): NVFP4 in groups of 16 with
//! plain scales, per-tensor FP8, or BF16 where the map names nothing. Host only.

const std = @import("std");
const Allocator = std.mem.Allocator;
const safetensors = @import("../../weights/safetensors.zig");
const modelopt = @import("../../weights/modelopt.zig");
const config_file = @import("config.zig");

const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;

pub const Error = error{ MissingTensor, UnexpectedTensor, UnknownTensor } || Allocator.Error;

/// A backbone layer's tensors live under `backbone.layers.N.`, the MTP block's under `llm.mtp.layers.N.`, the
/// encoder's under `perception.encoder.asr_encoder.layers.N.` and the Sortformer encoder's under
/// `perception.encoder.diarization_model.encoder.layers.N.`.
pub const backbone_prefix = "backbone.layers.";
pub const mtp_prefix = "llm.mtp.layers.";
pub const encoder_prefix = "perception.encoder.asr_encoder.layers.";
pub const diar_prefix = "perception.encoder.diarization_model.encoder.layers.";

/// The shard holds a tensor the manifest counts and never resolves: the CTC heads
/// (`model-ctc-bf16.safetensors`), out of scope (the spec's non-goals).
pub fn dropped(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "ctc_timestamp.");
}

/// The diarization expert's tensors where the config mounts none: counted and not read, as the CTC
/// heads' (a `pe_encoder_config` with a `diarization_model_cfg` mounts them — `kinds` reads them).
fn droppedDiar(c: Config, name: []const u8) bool {
    return c.diar == null and std.mem.startsWith(u8, name, "perception.encoder.diarization_model.");
}

/// What a ModelOpt linear stores beside `MODULE.weight` here.
pub const linear_parts = [_][]const u8{ ".weight_scale", ".weight_scale_2", ".input_scale" };

/// The family of layers a `Where` iterates (which prefix `layers.N.` takes).
pub const Stem = enum { whole, backbone, mtp, encoder, diar };

/// Which layers hold a kind of tensor.
pub const Where = enum {
    /// Once, the whole name `leaf`.
    top,
    /// Every backbone layer's (`layers.N.norm`).
    every,
    /// A Mamba-2 layer's (`layers.N.mixer.*`).
    mamba,
    /// A full-attention layer's.
    attention,
    /// A MoE layer's gate and shared expert.
    moe,
    /// `experts` times a MoE layer (`mixer.experts.E.*`).
    expert,
    /// Every layer of the MTP block's (`llm.mtp.layers.N.norm`).
    mtp,
    /// The MTP block's attention layer's (with `eh_proj`, `enorm`, `hnorm`).
    mtp_attention,
    /// Its MoE layer's gate and shared expert.
    mtp_moe,
    /// `experts` times its MoE layer.
    mtp_expert,
    /// Its last layer's (`final_layernorm`).
    mtp_final,
    /// Every perception encoder layer's.
    encoder,
    /// Every Sortformer diarization encoder layer's.
    diar,

    /// The family of layers `where` iterates.
    pub fn stem(self: Where) Stem {
        return switch (self) {
            .top => .whole,
            .every, .mamba, .attention, .moe, .expert => .backbone,
            .mtp, .mtp_attention, .mtp_moe, .mtp_expert, .mtp_final => .mtp,
            .encoder => .encoder,
            .diar => .diar,
        };
    }
};

/// A dimension of a tensor, by what the config makes it.
pub const Dim = enum {
    none,
    one,
    hidden,
    two_hidden,
    vocab,
    /// The query heads' width (`heads` x `head_dim`).
    q_inner,
    /// The KV heads' width (`kv_heads` x `head_dim`).
    kv_inner,
    /// A Mamba-2 mixer's inner width (`mamba_heads` x `mamba_head_dim`).
    mamba_inner,
    mamba_heads,
    /// The depthwise conv's channels (x and the groups' B and C).
    conv_width,
    /// `in_proj`'s outputs (z, x, B, C and dt).
    in_width,
    conv_taps,
    experts,
    /// A routed expert's width (`moe_intermediate_size`).
    moe,
    /// The shared expert's width (`moe_shared_expert_intermediate_size`).
    shared,
    /// The encoder's `d_model`, its FFN (`x ff_expansion`), its fused qkv (3 x `d_model`) and a head's width.
    enc_d,
    enc_ff,
    enc_qkv,
    enc_head,
    /// `pre_encode.proj`'s inputs (`feat_in` mels x `subsampling` stacked).
    enc_stack,
    enc_feat,
    /// The featurizer's: `features` mels, `n_fft`/2 + 1 bins and the window's samples.
    features,
    fft_bins,
    mel_window,
    /// `perception.proj`'s outputs (`output_dim`).
    out_dim,
    /// The Sortformer diarization encoder's (`pe_encoder_config.diarization_model_cfg`, WS-11): the
    /// encoder's `d_model`/`fc_d_model` (512) and FFN (x `ff_expansion`), the fused qkv (3 x `d_model`),
    /// the post-encoder `tf_d_model` (192), its outputs after the x8 upsample conv (with the conv's
    /// kernel of 3), `hidden_to_spks`' inputs (2 x `tf_d_model`), the speakers (`num_spks`), and the
    /// stacked frames of `features` mels x `subsampling`.
    diar_d,
    diar_ff,
    diar_qkv,
    diar_tf,
    diar_tf1,
    diar_tf2,
    diar_conv,
    diar_spk,
    diar_stack,

    pub fn of(self: Dim, c: Config) u64 {
        const b = c.backbone;
        const e = c.perception;
        return switch (self) {
            .none => 0,
            .one => 1,
            .hidden => b.hidden,
            .two_hidden => 2 * b.hidden,
            .vocab => b.vocab,
            .q_inner => b.heads * b.head_dim,
            .kv_inner => b.kv_heads * b.head_dim,
            .mamba_inner => b.mambaInner(),
            .mamba_heads => b.mamba_heads,
            .conv_width => b.convWidth(),
            .in_width => b.inWidth(),
            .conv_taps => b.conv_kernel,
            .experts => b.experts,
            .moe => b.moe_width,
            .shared => b.shared_width,
            .enc_d => e.d_model,
            .enc_ff => e.ffWidth(),
            .enc_qkv => 3 * e.d_model,
            .enc_head => e.headDim(),
            .enc_stack => e.stacked(),
            .enc_feat => e.feat_in,
            .features => e.features,
            .fft_bins => e.fftBins(),
            .mel_window => e.windowFrames(),
            .out_dim => e.output_dim,
            // the diarization dims: a kind asks for them only where `c.diar` mounts the expert
            // (zero where it does not — the kind is then skipped before its shape is read)
            .diar_d => if (c.diar) |dd| dd.d_model else 0,
            .diar_ff => if (c.diar) |dd| dd.ffWidth() else 0,
            .diar_qkv => 3 * (if (c.diar) |dd| dd.d_model else 0),
            .diar_tf => if (c.diar) |dd| dd.tf_d_model * dd.subsampling else 0,
            .diar_tf1 => if (c.diar) |dd| dd.tf_d_model else 0,
            .diar_tf2 => 2 * (if (c.diar) |dd| dd.tf_d_model else 0),
            .diar_conv => 3,
            .diar_spk => if (c.diar) |dd| dd.speakers else 0,
            .diar_stack => if (c.diar) |dd| dd.stacked() else 0,
        };
    }
};

pub const Kind = struct {
    /// The name after `layers.N.` (after `mixer.experts.E.` for an expert's; the whole name for `top`).
    leaf: []const u8,
    dtype: safetensors.DType,
    dims: [3]Dim,
    where: Where,
    /// A linear's weight (`MODULE.weight`, [outputs, inputs]): the map gives its storage and parts.
    linear: bool = false,

    /// Whether the kind belongs to the diarization expert (mounted or not): a `.diar`-stemmed layer
    /// kind, a `.top` leaf under `diarization_model.`, or a shape that reads `c.diar`'s dims.
    pub fn diar(self: Kind) bool {
        for (self.dims) |dim| switch (dim) {
            .diar_d, .diar_ff, .diar_qkv, .diar_tf, .diar_tf1, .diar_tf2, .diar_spk, .diar_stack => return true,
            else => {},
        };
        return self.where == .diar or std.mem.startsWith(u8, self.leaf, "perception.encoder.diarization_model.");
    }

    pub fn shape(self: Kind, c: Config, out: *[3]u64) []const u64 {
        var n: usize = 0;
        for (self.dims) |dim| if (dim != .none) {
            out[n] = dim.of(c);
            n += 1;
        };
        return out[0..n];
    }
};

fn kind(leaf: []const u8, dtype: safetensors.DType, dims: [3]Dim, where: Where) Kind {
    return .{ .leaf = leaf, .dtype = dtype, .dims = dims, .where = where };
}

fn linear(leaf: []const u8, dims: [3]Dim, where: Where) Kind {
    return .{ .leaf = leaf, .dtype = .BF16, .dims = dims, .where = where, .linear = true };
}

/// Every tensor of the checkpoint zspark reads, a line a name (spec 151's manifest: `scratch-151/manifest.tsv`
/// is these names against the published `model.safetensors.index.json`).
pub const kinds = [_]Kind{
    kind("backbone.embeddings.weight", .BF16, .{ .vocab, .hidden, .none }, .top),
    kind("lm_head.weight", .BF16, .{ .vocab, .hidden, .none }, .top),
    kind("backbone.norm_f.weight", .BF16, .{ .hidden, .none, .none }, .top),
    kind("norm.weight", .BF16, .{ .hidden, .none, .none }, .every),
    // Mamba-2: the conv, the SSM's own tensors and the gated norm are BF16; the projections are the map's
    linear("mixer.in_proj.weight", .{ .in_width, .hidden, .none }, .mamba),
    linear("mixer.out_proj.weight", .{ .hidden, .mamba_inner, .none }, .mamba),
    kind("mixer.conv1d.weight", .BF16, .{ .conv_width, .one, .conv_taps }, .mamba),
    kind("mixer.conv1d.bias", .BF16, .{ .conv_width, .none, .none }, .mamba),
    kind("mixer.A_log", .BF16, .{ .mamba_heads, .none, .none }, .mamba),
    kind("mixer.D", .BF16, .{ .mamba_heads, .none, .none }, .mamba),
    kind("mixer.dt_bias", .BF16, .{ .mamba_heads, .none, .none }, .mamba),
    kind("mixer.norm.weight", .BF16, .{ .mamba_inner, .none, .none }, .mamba),
    // full attention: q, k, v and o, the map's
    linear("mixer.q_proj.weight", .{ .q_inner, .hidden, .none }, .attention),
    linear("mixer.k_proj.weight", .{ .kv_inner, .hidden, .none }, .attention),
    linear("mixer.v_proj.weight", .{ .kv_inner, .hidden, .none }, .attention),
    linear("mixer.o_proj.weight", .{ .hidden, .q_inner, .none }, .attention),
    // MoE: the router BF16, its correction bias F32; the shared expert and every routed one the map's
    kind("mixer.gate.weight", .BF16, .{ .experts, .hidden, .none }, .moe),
    kind("mixer.gate.e_score_correction_bias", .F32, .{ .experts, .none, .none }, .moe),
    linear("mixer.shared_experts.up_proj.weight", .{ .shared, .hidden, .none }, .moe),
    linear("mixer.shared_experts.down_proj.weight", .{ .hidden, .shared, .none }, .moe),
    linear("up_proj.weight", .{ .moe, .hidden, .none }, .expert),
    linear("down_proj.weight", .{ .hidden, .moe, .none }, .expert),
    // the MTP block's own norms, its attention layer's fusion and projections, its MoE layer's
    kind("norm.weight", .BF16, .{ .hidden, .none, .none }, .mtp),
    kind("final_layernorm.weight", .BF16, .{ .hidden, .none, .none }, .mtp_final),
    linear("eh_proj.weight", .{ .hidden, .two_hidden, .none }, .mtp_attention),
    kind("enorm.weight", .BF16, .{ .hidden, .none, .none }, .mtp_attention),
    kind("hnorm.weight", .BF16, .{ .hidden, .none, .none }, .mtp_attention),
    linear("mixer.q_proj.weight", .{ .q_inner, .hidden, .none }, .mtp_attention),
    linear("mixer.k_proj.weight", .{ .kv_inner, .hidden, .none }, .mtp_attention),
    linear("mixer.v_proj.weight", .{ .kv_inner, .hidden, .none }, .mtp_attention),
    linear("mixer.o_proj.weight", .{ .hidden, .q_inner, .none }, .mtp_attention),
    kind("mixer.gate.weight", .BF16, .{ .experts, .hidden, .none }, .mtp_moe),
    kind("mixer.gate.e_score_correction_bias", .F32, .{ .experts, .none, .none }, .mtp_moe),
    linear("mixer.shared_experts.up_proj.weight", .{ .shared, .hidden, .none }, .mtp_moe),
    linear("mixer.shared_experts.down_proj.weight", .{ .hidden, .shared, .none }, .mtp_moe),
    linear("up_proj.weight", .{ .moe, .hidden, .none }, .mtp_expert),
    linear("down_proj.weight", .{ .hidden, .moe, .none }, .mtp_expert),
    // the perception module: the featurizer's tables, the feature-stacking projection, the norms, `proj`
    kind("perception.preprocessor.featurizer.fb", .BF16, .{ .one, .features, .fft_bins }, .top),
    kind("perception.preprocessor.featurizer.window", .BF16, .{ .mel_window, .none, .none }, .top),
    kind("perception.encoder.asr_encoder.pre_encode.proj.weight", .BF16, .{ .enc_d, .enc_stack, .none }, .top),
    kind("perception.encoder.asr_encoder.embed_norm.weight", .BF16, .{ .enc_d, .none, .none }, .top),
    kind("perception.encoder.asr_encoder.embed_norm.bias", .BF16, .{ .enc_d, .none, .none }, .top),
    kind("perception.encoder.asr_encoder.final_norm.weight", .BF16, .{ .enc_d, .none, .none }, .top),
    kind("perception.encoder.asr_encoder.final_norm.bias", .BF16, .{ .enc_d, .none, .none }, .top),
    kind("perception.encoder.asr_norm.weight", .BF16, .{ .enc_d, .none, .none }, .top),
    kind("perception.encoder.asr_norm.bias", .BF16, .{ .enc_d, .none, .none }, .top),
    kind("perception.proj.weight", .BF16, .{ .out_dim, .enc_d, .none }, .top),
    kind("perception.proj.bias", .BF16, .{ .out_dim, .none, .none }, .top),
    // an encoder layer: norms and q/k norms BF16 (with biases), w_qkv, out_proj and the FFN the map's
    kind("norm1.weight", .BF16, .{ .enc_d, .none, .none }, .encoder),
    kind("norm1.bias", .BF16, .{ .enc_d, .none, .none }, .encoder),
    kind("norm2.weight", .BF16, .{ .enc_d, .none, .none }, .encoder),
    kind("norm2.bias", .BF16, .{ .enc_d, .none, .none }, .encoder),
    kind("attn.q_norm.weight", .BF16, .{ .enc_head, .none, .none }, .encoder),
    kind("attn.q_norm.bias", .BF16, .{ .enc_head, .none, .none }, .encoder),
    kind("attn.k_norm.weight", .BF16, .{ .enc_head, .none, .none }, .encoder),
    kind("attn.k_norm.bias", .BF16, .{ .enc_head, .none, .none }, .encoder),
    linear("attn.w_qkv.weight", .{ .enc_qkv, .enc_d, .none }, .encoder),
    linear("attn.out_proj.weight", .{ .enc_d, .enc_d, .none }, .encoder),
    kind("attn.out_proj.bias", .BF16, .{ .enc_d, .none, .none }, .encoder),
    linear("ffn.net.0.weight", .{ .enc_ff, .enc_d, .none }, .encoder),
    kind("ffn.net.0.bias", .BF16, .{ .enc_ff, .none, .none }, .encoder),
    linear("ffn.net.3.weight", .{ .enc_d, .enc_ff, .none }, .encoder),
    kind("ffn.net.3.bias", .BF16, .{ .enc_d, .none, .none }, .encoder),
    // the Sortformer diarization expert (WS-11, all BF16 — the quant map names none of it): its own
    // featurizer tables, feature-stacking projection and norms, `diar_norm`, and the sortformer
    // modules' linears (encoder_proj, the upsample conv, the speaker heads, the learnable silence
    // embedding); `diar_kernel` is a computed sinusoid buffer, no tensor
    kind("perception.encoder.diarization_model.preprocessor.featurizer.fb", .BF16, .{ .one, .features, .fft_bins }, .top),
    kind("perception.encoder.diarization_model.preprocessor.featurizer.window", .BF16, .{ .mel_window, .none, .none }, .top),
    kind("perception.encoder.diarization_model.encoder.pre_encode.proj.weight", .BF16, .{ .diar_d, .diar_stack, .none }, .top),
    kind("perception.encoder.diarization_model.encoder.embed_norm.weight", .BF16, .{ .diar_d, .none, .none }, .top),
    kind("perception.encoder.diarization_model.encoder.embed_norm.bias", .BF16, .{ .diar_d, .none, .none }, .top),
    kind("perception.encoder.diarization_model.encoder.final_norm.weight", .BF16, .{ .diar_d, .none, .none }, .top),
    kind("perception.encoder.diarization_model.encoder.final_norm.bias", .BF16, .{ .diar_d, .none, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.encoder_proj.weight", .BF16, .{ .diar_tf1, .diar_d, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.encoder_proj.bias", .BF16, .{ .diar_tf1, .none, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.subpixel_upsample.weight", .BF16, .{ .diar_tf, .diar_tf1, .diar_conv }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.subpixel_upsample.bias", .BF16, .{ .diar_tf, .none, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.first_hidden_to_hidden.weight", .BF16, .{ .diar_tf1, .diar_tf1, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.first_hidden_to_hidden.bias", .BF16, .{ .diar_tf1, .none, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.single_hidden_to_spks.weight", .BF16, .{ .diar_spk, .diar_tf1, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.single_hidden_to_spks.bias", .BF16, .{ .diar_spk, .none, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.hidden_to_spks.weight", .BF16, .{ .diar_spk, .diar_tf2, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.hidden_to_spks.bias", .BF16, .{ .diar_spk, .none, .none }, .top),
    kind("perception.encoder.diarization_model.sortformer_modules.learnable_sil_emb", .BF16, .{ .diar_d, .none, .none }, .top),
    kind("perception.encoder.diar_norm.weight", .BF16, .{ .diar_spk, .none, .none }, .top),
    kind("perception.encoder.diar_norm.bias", .BF16, .{ .diar_spk, .none, .none }, .top),
    // a Sortformer encoder layer: norms, w_qkv (no bias) and out_proj, the FFN — no qk norms
    // (`qk_norm: false`), the attention's bias on out_proj alone
    kind("norm1.weight", .BF16, .{ .diar_d, .none, .none }, .diar),
    kind("norm1.bias", .BF16, .{ .diar_d, .none, .none }, .diar),
    kind("norm2.weight", .BF16, .{ .diar_d, .none, .none }, .diar),
    kind("norm2.bias", .BF16, .{ .diar_d, .none, .none }, .diar),
    kind("attn.w_qkv.weight", .BF16, .{ .diar_qkv, .diar_d, .none }, .diar),
    kind("attn.out_proj.weight", .BF16, .{ .diar_d, .diar_d, .none }, .diar),
    kind("attn.out_proj.bias", .BF16, .{ .diar_d, .none, .none }, .diar),
    kind("ffn.net.0.weight", .BF16, .{ .diar_ff, .diar_d, .none }, .diar),
    kind("ffn.net.0.bias", .BF16, .{ .diar_ff, .none, .none }, .diar),
    kind("ffn.net.3.weight", .BF16, .{ .diar_d, .diar_ff, .none }, .diar),
    kind("ffn.net.3.bias", .BF16, .{ .diar_d, .none, .none }, .diar),
};

/// Whether layer `layer` (a backbone layer of `types.len`, or an MTP block layer of `mtp_types.len`) holds
/// tensors of `where`.
pub fn holds(c: Config, where: Where, layer: usize) bool {
    const b = c.backbone;
    return switch (where) {
        .top => false,
        .every => true,
        .mamba => b.types[layer] == .mamba,
        .attention => b.types[layer] == .attention,
        .moe, .expert => b.types[layer] == .moe,
        .mtp => true,
        .mtp_attention => b.mtp_types[layer] == .attention,
        .mtp_moe, .mtp_expert => b.mtp_types[layer] == .moe,
        .mtp_final => layer + 1 == b.mtp_types.len,
        .encoder => true,
        .diar => true,
    };
}

/// `name` names a tensor `kinds` and the config's layer kinds place: the unknown-tensor scan's ask. A
/// linear's scale parts (`MODULE.weight_scale*` etc.) have their weight's place.
pub fn placed(c: Config, name: []const u8) bool {
    if (dropped(name) or droppedDiar(c, name)) return true;
    var leaf = name;
    var layer: ?usize = null;
    var stem: Stem = .whole;
    for ([_]struct { []const u8, Stem, usize }{
        .{ backbone_prefix, .backbone, c.backbone.types.len },
        .{ mtp_prefix, .mtp, c.backbone.mtp_types.len },
        .{ encoder_prefix, .encoder, c.perception.layers },
        .{ diar_prefix, .diar, if (c.diar) |dd| dd.layers else 0 },
    }) |each| {
        const prefix = each[0];
        const total = each[2];
        if (!std.mem.startsWith(u8, name, prefix)) continue;
        const rest = name[prefix.len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
        const at = std.fmt.parseInt(usize, rest[0..dot], 10) catch return false;
        if (at >= total) return false;
        layer = at;
        stem = each[1];
        leaf = rest[dot + 1 ..];
        break;
    }
    // a linear's parts name the module: fold a part's suffix into its weight's leaf
    var buffer: [160]u8 = undefined;
    for (linear_parts) |suffix| {
        if (std.mem.endsWith(u8, leaf, suffix)) {
            leaf = std.fmt.bufPrint(&buffer, "{s}.weight", .{leaf[0 .. leaf.len - suffix.len]}) catch return false;
            break;
        }
    }
    // an expert's index: the leaf after `mixer.experts.E.`
    var routed = false;
    if (layer != null) {
        const experts = "mixer.experts.";
        if (std.mem.startsWith(u8, leaf, experts)) {
            const rest = leaf[experts.len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
            const at = std.fmt.parseInt(usize, rest[0..dot], 10) catch return false;
            if (at >= c.backbone.experts) return false;
            leaf = rest[dot + 1 ..];
            routed = true;
        }
    }
    for (kinds) |one| {
        if (one.where.stem() != stem or !std.mem.eql(u8, one.leaf, leaf)) continue;
        if (one.diar() and c.diar == null) continue;
        if (routed != (one.where == .expert or one.where == .mtp_expert)) continue;
        if (layer) |at| {
            if (!holds(c, one.where, at)) continue;
        } else if (one.where != .top) continue;
        return true;
    }
    return false;
}

/// A kind's tensors as found: how many and their bytes.
pub const Found = struct {
    count: usize = 0,
    bytes: u64 = 0,
};

pub const Layout = struct {
    /// One a `kinds` entry.
    found: []Found,
    /// The linears by storage, and the map's entries a linear met (every one of them, or `init` refuses).
    schemes: std.enums.EnumArray(modelopt.Scheme, usize) = .initFill(0),
    mapped: usize = 0,
    /// The checkpoint's tensors zspark reads, and their bytes.
    tensors: usize = 0,
    bytes: u64 = 0,
    /// The CTC heads', counted and not read (`dropped`).
    skipped: usize = 0,
    skipped_bytes: u64 = 0,

    /// Finds every tensor the config asks for in `checkpoint`, with its dtype and shape, and refuses a tensor
    /// it did not ask for. The result lives in `arena`.
    pub fn init(arena: Allocator, c: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic) Error!Layout {
        var self: Layout = .{ .found = try arena.alloc(Found, kinds.len) };
        @memset(self.found, .{});
        const met = try arena.alloc(bool, c.quant.layers.len);
        @memset(met, false);
        var name: [192]u8 = undefined;
        for (kinds, self.found) |one, *found| {
            if (one.diar() and c.diar == null) continue;
            switch (one.where.stem()) {
                .whole => try self.take(arena, c, checkpoint, diagnostic, met, one, found, one.leaf),
                .backbone => for (0..c.backbone.types.len) |layer| {
                    if (!holds(c.*, one.where, layer)) continue;
                    if (one.where != .expert) {
                        try self.take(arena, c, checkpoint, diagnostic, met, one, found, std.fmt.bufPrint(&name, backbone_prefix ++ "{d}.{s}", .{ layer, one.leaf }) catch unreachable);
                    } else for (0..c.backbone.experts) |expert| {
                        try self.take(arena, c, checkpoint, diagnostic, met, one, found, std.fmt.bufPrint(&name, backbone_prefix ++ "{d}.mixer.experts.{d}.{s}", .{ layer, expert, one.leaf }) catch unreachable);
                    }
                },
                .mtp => for (0..c.backbone.mtp_types.len) |layer| {
                    if (!holds(c.*, one.where, layer)) continue;
                    if (one.where != .mtp_expert) {
                        try self.take(arena, c, checkpoint, diagnostic, met, one, found, std.fmt.bufPrint(&name, mtp_prefix ++ "{d}.{s}", .{ layer, one.leaf }) catch unreachable);
                    } else for (0..c.backbone.experts) |expert| {
                        try self.take(arena, c, checkpoint, diagnostic, met, one, found, std.fmt.bufPrint(&name, mtp_prefix ++ "{d}.mixer.experts.{d}.{s}", .{ layer, expert, one.leaf }) catch unreachable);
                    }
                },
                .encoder => for (0..c.perception.layers) |layer| {
                    try self.take(arena, c, checkpoint, diagnostic, met, one, found, std.fmt.bufPrint(&name, encoder_prefix ++ "{d}.{s}", .{ layer, one.leaf }) catch unreachable);
                },
                .diar => if (c.diar) |dd| for (0..dd.layers) |layer| {
                    try self.take(arena, c, checkpoint, diagnostic, met, one, found, std.fmt.bufPrint(&name, diar_prefix ++ "{d}.{s}", .{ layer, one.leaf }) catch unreachable);
                },
            }
        }
        for (met, 0..) |was, at| {
            if (!was) {
                diagnostic.set(arena, "quantized_layers names {s}, which is no linear of Nemotron-3.5-Transcribe's layout", .{c.quant.layers[at].name});
                return error.UnknownTensor;
            }
            self.mapped += 1;
        }
        // what is left is the CTC heads, or a tensor this layout has no place for
        var unknown: usize = 0;
        var it = checkpoint.tensors.iterator();
        while (it.next()) |entry| {
            if (!dropped(entry.key_ptr.*) and !droppedDiar(c.*, entry.key_ptr.*)) {
                unknown += 1;
                continue;
            }
            self.skipped += 1;
            self.skipped_bytes += entry.value_ptr.data.len;
        }
        if (unknown != self.tensors) {
            it = checkpoint.tensors.iterator();
            while (it.next()) |entry| if (!placed(c.*, entry.key_ptr.*)) {
                diagnostic.set(arena, "the checkpoint holds {s}, which Nemotron-3.5-Transcribe's layout has no place for", .{entry.key_ptr.*});
                return error.UnknownTensor;
            };
            diagnostic.set(arena, "the checkpoint holds {d} tensors where the config asks for {d}", .{ unknown, self.tensors });
            return error.UnknownTensor;
        }
        return self;
    }

    fn take(self: *Layout, arena: Allocator, c: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic, met: []bool, one: Kind, found: *Found, name: []const u8) Error!void {
        if (one.linear) return self.takeLinear(arena, c, checkpoint, diagnostic, met, one, found, name);
        const tensor = checkpoint.get(name) orelse {
            diagnostic.set(arena, "the checkpoint has no {s}", .{name});
            return error.MissingTensor;
        };
        var dims: [3]u64 = undefined;
        const shape = one.shape(c.*, &dims);
        if (tensor.dtype != one.dtype or !std.mem.eql(u64, tensor.shape, shape)) {
            diagnostic.set(arena, "{s} is {t} {any}; the config makes it {t} {any}", .{ name, tensor.dtype, tensor.shape, one.dtype, shape });
            return error.UnexpectedTensor;
        }
        found.count += 1;
        found.bytes += tensor.data.len;
        self.tensors += 1;
        self.bytes += tensor.data.len;
    }

    /// A linear: `modelopt.read` finds its parts as the map stores the module (`llm.mtp.layers.*` as the map's
    /// `mtp.layers.*`, `config.zig`'s `Quant.findIndex`).
    fn takeLinear(self: *Layout, arena: Allocator, c: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic, met: []bool, one: Kind, found: *Found, name: []const u8) Error!void {
        const module = name[0 .. name.len - ".weight".len];
        const entry = c.quant.findIndex(module);
        if (entry) |at| met[at] = true;
        var dims: [3]u64 = undefined;
        const shape = one.shape(c.*, &dims);
        const stored = modelopt.read(arena, checkpoint, module, if (entry) |at| c.quant.layers[at] else null, shape[0], shape[1], diagnostic) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.MissingTensor => error.MissingTensor,
            error.UnexpectedTensor, error.Indivisible => error.UnexpectedTensor,
        };
        const held = stored.held(.whole);
        found.count += held.count;
        found.bytes += held.bytes;
        self.tensors += held.count;
        self.bytes += held.bytes;
        self.schemes.getPtr(stored.scheme).* += 1;
    }
};

/// The shard that holds `name` (its index in `checkpoint.shards`), or null; a synthetic checkpoint has no
/// shards. The manifest's "home shard" (`model.safetensors.index.json`'s answer).
pub fn home(checkpoint: *const safetensors.Checkpoint, name: []const u8) ?usize {
    for (checkpoint.shards, 0..) |*shard, at| {
        if (shard.get(name) != null) return at;
    }
    return null;
}
