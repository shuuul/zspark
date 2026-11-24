//! Nemotron-3.5-Transcribe-30B-A4B's configs (`model_type` `nemo_speechlm`, TensorFold's family `nemotron`;
//! spec 151): the NeMo speechlm `config.json` (the prompt's format and audio locator, the perception module's
//! shape), the NemotronH backbone's `llm_backbone/config.json` (52 layers, each Mamba-2, MoE or full attention
//! by `layers_block_type`, and an MTP block of `mtp_layers_block_type`), and `hf_quant_config.json`'s ModelOpt
//! MIXED_PRECISION map (`weights/modelopt.zig`): attention and the experts NVFP4 group 16, the Mamba
//! projections, the perception encoder and the whole MTP FP8, the rest BF16. Host only.

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const modelopt = @import("../../weights/modelopt.zig");
const reader = @import("../../weights/checkpoint_reader.zig");
const Parser = reader.Parser;

/// Why a config was refused, in the model's own words.
pub const Diagnostic = reader.Diagnostic;

pub const Error = reader.Error;

/// The name config.json's `model_type` carries.
pub const model_types = [_][]const u8{"nemo_speechlm"};

/// A backbone layer's `layers_block_type` entry (and the MTP block's `mtp_layers_block_type`).
pub const LayerType = enum { mamba, attention, moe };

/// `llm_backbone/config.json` (`model_type` `nemotron_h`): the language model's shape. Slices live in the
/// arena `parse` was given.
pub const Backbone = struct {
    hidden: usize,
    vocab: usize,
    layers: usize,
    /// `layers_block_type`, a `.mamba`, `.attention` or `.moe` a layer.
    types: []const LayerType,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    /// `partial_rotary_factor`: the share of a head's width RoPE touches.
    rotary_factor: f64,
    max_positions: usize,
    eps: f64,
    /// Mamba-2: heads of `mamba_head_dim`, `groups` sharing B and C of `ssm_state`, a depthwise conv of
    /// `conv_kernel` taps (with a bias where `conv_bias`), the scan's chunk, and dt clamped to
    /// `time_step_limit` (independent of time_step_min/max's initializer). `mamba_hidden_act` is silu, the projections carry no bias, the SSM
    /// state is fp32.
    expand: usize,
    mamba_heads: usize,
    mamba_head_dim: usize,
    groups: usize,
    ssm_state: usize,
    conv_kernel: usize,
    chunk: usize,
    conv_bias: bool,
    dt_min: f64,
    dt_max: f64,
    /// MoE: routed experts of `moe_width`, `top_k` a token, one shared expert of `shared_width`, the weights'
    /// scale, `norm_topk_prob` and `rescale_prenorm_residual`. The expert activation is `relu2`.
    experts: usize,
    top_k: usize,
    shared_width: usize,
    moe_width: usize,
    routed_scale: f64,
    norm_topk: bool,
    rescale_prenorm: bool,
    /// `num_nextn_predict_layers` MTP blocks of `mtp_layers_block_type` internal layers each (Transcribe's
    /// one block is attention then MoE).
    mtp_layers: usize,
    mtp_types: []const LayerType,
    bos: u32,
    eos: u32,
    pad: u32,

    /// The mixer's inner width: `mamba_heads` of `mamba_head_dim` (4,096; `hidden` x `expand` is nominal).
    pub fn mambaInner(self: Backbone) usize {
        return self.mamba_heads * self.mamba_head_dim;
    }

    /// The depthwise conv's channels: x of `mambaInner` and each group's B and C of `ssm_state`.
    pub fn convWidth(self: Backbone) usize {
        return self.mambaInner() + 2 * self.groups * self.ssm_state;
    }

    /// `in_proj`'s outputs: z and x of `mambaInner`, the groups' B and C, and a dt a head.
    pub fn inWidth(self: Backbone) usize {
        return 2 * self.mambaInner() + 2 * self.groups * self.ssm_state + self.mamba_heads;
    }

    /// `layers_block_type` holds `kind` this many times.
    pub fn count(self: Backbone, kind: LayerType) usize {
        return std.mem.count(LayerType, self.types, &.{kind});
    }

    /// `text` is `llm_backbone/config.json`. The result lives in `arena`.
    pub fn parse(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) Error!Backbone {
        const p: Parser = .{ .arena = arena, .diagnostic = diagnostic, .where = "llm_backbone/config.json" };
        const root = json.parseFromSliceLeaky(json.Value, arena, text, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return p.fail(error.InvalidConfig, "{s} is not valid JSON ({t})", .{ p.where, err }),
        };
        if (root != .object) return p.fail(error.InvalidConfig, "{s} is not a JSON object", .{p.where});
        const t = root.object;
        const model_type = if (t.get("model_type")) |v| (if (v == .string) v.string else "") else "";
        if (!std.mem.eql(u8, model_type, "nemotron_h"))
            return p.fail(error.UnsupportedModel, "{s} describes model_type \"{s}\", not Nemotron's backbone (nemotron_h)", .{ p.where, model_type });
        if (p.flagOr(t, "tie_word_embeddings", false))
            return p.fail(error.UnsupportedModel, "{s}: the embedding is tied to the head", .{p.where});
        if (p.flagOr(t, "attention_bias", false)) return p.fail(error.UnsupportedModel, "{s}: attention_bias is set", .{p.where});
        if (p.flagOr(t, "mlp_bias", false) or p.flagOr(t, "use_bias", false))
            return p.fail(error.UnsupportedModel, "{s}: a linear carries a bias", .{p.where});
        if (p.flagOr(t, "mamba_proj_bias", false)) return p.fail(error.UnsupportedModel, "{s}: mamba_proj_bias is set", .{p.where});
        if (!std.mem.eql(u8, p.stringOr(t, "mamba_hidden_act", "silu"), "silu"))
            return p.fail(error.UnsupportedModel, "{s}: mamba_hidden_act \"{s}\", not silu", .{ p.where, p.stringOr(t, "mamba_hidden_act", "") });
        if (!std.mem.eql(u8, p.stringOr(t, "mlp_hidden_act", "relu2"), "relu2"))
            return p.fail(error.UnsupportedModel, "{s}: mlp_hidden_act \"{s}\", not relu2", .{ p.where, p.stringOr(t, "mlp_hidden_act", "") });
        if (!std.mem.eql(u8, p.stringOr(t, "mamba_ssm_cache_dtype", "float32"), "float32"))
            return p.fail(error.UnsupportedModel, "{s}: mamba_ssm_cache_dtype \"{s}\"; zspark's Mamba-2 keeps an fp32 state", .{ p.where, p.stringOr(t, "mamba_ssm_cache_dtype", "") });
        if (t.get("scoring_func")) |scoring| {
            const name = if (scoring == .string) scoring.string else "";
            if (!std.mem.eql(u8, name, "sigmoid"))
                return p.fail(error.UnsupportedModel, "{s}: scoring_func is \"{s}\"; zspark's router scores with a sigmoid", .{ p.where, name });
        }
        if (t.get("moe_latent_size")) |latent| if (latent != .null)
            return p.fail(error.UnsupportedModel, "{s}: moe_latent_size is {f}; zspark reads Nemotron's plain experts", .{ p.where, json.fmt(latent, .{}) });
        const shared_experts = try p.uint(t, "n_shared_experts");
        if (shared_experts != 1)
            return p.fail(error.UnsupportedModel, "{s}: n_shared_experts is {d}; zspark reads one shared expert", .{ p.where, shared_experts });
        const types = try kinds(p, t, "layers_block_type");
        // The published Transcribe export omits this redundant field.
        const layers = if (t.contains("num_hidden_layers")) try p.uint(t, "num_hidden_layers") else types.len;
        const dt_limit = try timeStepLimit(p, t);
        const self: Backbone = .{
            .hidden = try p.uint(t, "hidden_size"),
            .vocab = try p.uint(t, "vocab_size"),
            .layers = layers,
            .types = types,
            .heads = try p.uint(t, "num_attention_heads"),
            .kv_heads = try p.uint(t, "num_key_value_heads"),
            .head_dim = try p.uint(t, "head_dim"),
            .rope_theta = numberOr(t, "rope_theta", 10000),
            .rotary_factor = numberOr(t, "partial_rotary_factor", 1.0),
            .max_positions = try p.uint(t, "max_position_embeddings"),
            .eps = numberOr(t, "layer_norm_epsilon", numberOr(t, "norm_eps", 1e-5)),
            .expand = try p.uint(t, "expand"),
            .mamba_heads = try p.uint(t, "mamba_num_heads"),
            .mamba_head_dim = try p.uint(t, "mamba_head_dim"),
            .groups = try p.uint(t, "n_groups"),
            .ssm_state = try p.uint(t, "ssm_state_size"),
            .conv_kernel = try p.uint(t, "conv_kernel"),
            .chunk = try p.uint(t, "chunk_size"),
            .conv_bias = flagOrTrue(t, "use_conv_bias"),
            .dt_min = dt_limit[0],
            .dt_max = dt_limit[1],
            .experts = try p.uint(t, "n_routed_experts"),
            .top_k = try p.uint(t, "num_experts_per_tok"),
            .shared_width = try p.uint(t, "moe_shared_expert_intermediate_size"),
            .moe_width = try p.uint(t, "moe_intermediate_size"),
            .routed_scale = numberOr(t, "routed_scaling_factor", 1.0),
            .norm_topk = p.flagOr(t, "norm_topk_prob", true),
            .rescale_prenorm = p.flagOr(t, "rescale_prenorm_residual", false),
            .mtp_layers = try p.uint(t, "num_nextn_predict_layers"),
            .mtp_types = try kinds(p, t, "mtp_layers_block_type"),
            .bos = @intCast(try p.uint(t, "bos_token_id")),
            .eos = @intCast(try p.uint(t, "eos_token_id")),
            .pad = @intCast(try p.uint(t, "pad_token_id")),
        };
        if (self.types.len != layers)
            return p.fail(error.InvalidConfig, "{s}: layers_block_type names {d} layers, num_hidden_layers {d}", .{ p.where, self.types.len, layers });
        if (self.mtp_layers > 1)
            return p.fail(error.UnsupportedModel, "{s}: {d} MTP blocks; zspark reads one", .{ p.where, self.mtp_layers });
        if (self.mtp_types.len != 2 or self.mtp_types[0] != .attention or self.mtp_types[1] != .moe)
            return p.fail(error.UnsupportedModel, "{s}: mtp_layers_block_type holds no attention layer before its MoE", .{p.where});
        if (self.heads == 0 or self.kv_heads == 0 or self.heads % self.kv_heads != 0)
            return p.fail(error.InvalidConfig, "{s}: {d} heads over {d} KV heads", .{ p.where, self.heads, self.kv_heads });
        if (self.head_dim != 64 and self.head_dim != 128 and self.head_dim != 256)
            return p.fail(error.UnsupportedModel, "{s}: head_dim {d}", .{ p.where, self.head_dim });
        if (self.hidden % 16 != 0 or self.moe_width % 16 != 0 or self.shared_width % 16 != 0)
            return p.fail(error.InvalidConfig, "{s}: widths {d}, {d} and {d} are not whole NVFP4 blocks of 16", .{ p.where, self.hidden, self.moe_width, self.shared_width });
        if (self.top_k == 0 or self.top_k > self.experts)
            return p.fail(error.InvalidConfig, "{s}: {d} experts a token of {d}", .{ p.where, self.top_k, self.experts });
        return self;
    }
};

/// config.json's `perception` (NeMo's `AudioPerceptionModule`): the log-mel preprocessor, the 8x feature
/// stacking and the Transformer encoder over `chunk_seconds` chunks, and the projection to `output_dim`. The
/// diarization encoder and CTC heads beside them are not read. Its `_target_` classes are the library's own.
pub const Perception = struct {
    output_dim: usize,
    /// AudioToMelSpectrogramPreprocessor: 16 kHz, `features` mel bins of `n_fft` (a `window_size`/`window_stride`
    /// hann window, log after `dither`). `window` names the window's shape.
    sample_rate: usize,
    n_fft: usize,
    features: usize,
    window_size: f64,
    window_stride: f64,
    dither: f64,
    log_mel: bool,
    /// TransformerEncoder: `d_model`, `heads` (qk_norm of a head's width), `layers`, FFN of `ff_expansion`,
    /// `subsampling` input frames stacked a feature, full attention with RoPE on `rotary_fraction` of a head.
    feat_in: usize,
    d_model: usize,
    heads: usize,
    layers: usize,
    qk_norm: bool,
    ff_expansion: usize,
    subsampling: usize,
    rope_base: f64,
    rotary_fraction: f64,

    /// A head's width (`d_model`/`heads`).
    pub fn headDim(self: Perception) usize {
        return self.d_model / self.heads;
    }

    /// The FFN's width (`d_model` x `ff_expansion`).
    pub fn ffWidth(self: Perception) usize {
        return self.d_model * self.ff_expansion;
    }

    /// `pre_encode.proj`'s inputs: `feat_in` mels x `subsampling` stacked.
    pub fn stacked(self: Perception) usize {
        return self.feat_in * self.subsampling;
    }

    /// The filterbank's bins: `n_fft`/2 + 1.
    pub fn fftBins(self: Perception) usize {
        return self.n_fft / 2 + 1;
    }

    /// The window's samples (`window_size` of `sample_rate`).
    pub fn windowFrames(self: Perception) usize {
        return @intFromFloat(@round(self.window_size * @as(f64, @floatFromInt(self.sample_rate))));
    }

    /// The `perception` object of config.json.
    pub fn parse(p: Parser, fields: json.ObjectMap) Error!Perception {
        const whole = try p.object(fields, "perception");
        const pre = try p.object(whole, "preprocessor");
        const enc = try p.object(whole, "encoder");
        const window = p.stringOr(pre, "window", "hann");
        if (!std.mem.eql(u8, window, "hann"))
            return p.fail(error.UnsupportedModel, "config.json: the featurizer's window is \"{s}\", not hann", .{window});
        if (p.flagOr(enc, "qkv_bias", false))
            return p.fail(error.UnsupportedModel, "config.json: the encoder's qkv carries a bias", .{});
        if (!std.mem.eql(u8, p.stringOr(enc, "attn_mode", "full"), "full"))
            return p.fail(error.UnsupportedModel, "config.json: the encoder's attn_mode is \"{s}\"; zspark reads full attention", .{p.stringOr(enc, "attn_mode", "")});
        if (!std.mem.eql(u8, p.stringOr(enc, "self_attention_model", "rope"), "rope"))
            return p.fail(error.UnsupportedModel, "config.json: the encoder's self_attention_model is \"{s}\", not RoPE", .{p.stringOr(enc, "self_attention_model", "")});
        const self: Perception = .{
            .output_dim = try p.uint(whole, "output_dim"),
            .sample_rate = try p.uint(pre, "sample_rate"),
            .n_fft = try p.uint(pre, "n_fft"),
            .features = try p.uint(pre, "features"),
            .window_size = numberOr(pre, "window_size", 0.025),
            .window_stride = numberOr(pre, "window_stride", 0.01),
            .dither = numberOr(pre, "dither", 1e-5),
            .log_mel = p.flagOr(pre, "log", true),
            .feat_in = try p.uint(enc, "feat_in"),
            .d_model = try p.uint(enc, "d_model"),
            .heads = try p.uint(enc, "n_heads"),
            .layers = try p.uint(enc, "n_layers"),
            .qk_norm = p.flagOr(enc, "qk_norm", true),
            .ff_expansion = try p.uint(enc, "ff_expansion"),
            .subsampling = try p.uint(enc, "subsampling_factor"),
            .rope_base = numberOr(enc, "rope_base", 10000),
            .rotary_fraction = numberOr(enc, "rotary_fraction", 1.0),
        };
        if (self.heads == 0 or self.d_model % self.heads != 0)
            return p.fail(error.InvalidConfig, "config.json: the encoder's d_model {d} over {d} heads", .{ self.d_model, self.heads });
        if (self.n_fft % 2 != 0 or self.n_fft < self.features or self.features == 0 or self.feat_in == 0 or self.subsampling == 0)
            return p.fail(error.InvalidConfig, "config.json: n_fft {d}, features {d}, feat_in {d}, subsampling {d}", .{ self.n_fft, self.features, self.feat_in, self.subsampling });
        if (self.d_model % 16 != 0 or self.ffWidth() % 16 != 0)
            return p.fail(error.InvalidConfig, "config.json: the encoder's widths {d} and {d} are not whole FP8 rows' blocks", .{ self.d_model, self.ffWidth() });
        return self;
    }
};

/// The PEE's Sortformer diarization expert (`pe_encoder_config.diarization_model_cfg`, spec 151 WS-11): its
/// own featurizer and a TransformerEncoder (feature-stacking `subsampling`, RoPE, no qk norms, biases on
/// the out projection and the FFN — all BF16), then `sortformer_modules` (`encoder_proj` to `tf_d_model`,
/// the subpixel upsample conv x `subsampling`, the speaker heads) and the streaming speaker cache the
/// windowed path keeps (`fifo_len`, `spkcache_len`, `spkcache_update_period`, in encoder rows).
pub const Diar = struct {
    /// `encoder`: `d_model`, `heads`, `layers`, FFN of `ff_expansion`, `subsampling` mel frames a row,
    /// RoPE (`rope_base`) over the whole head (`rotary_fraction` 1.0). `qk_norm` must be false.
    feat_in: usize,
    d_model: usize,
    heads: usize,
    layers: usize,
    ff_expansion: usize,
    subsampling: usize,
    rope_base: f64,
    rotary_fraction: f64,
    /// `sortformer_modules`: `tf_d_model` the post-encoder width, `num_spks`, `use_learnable_sil_emb`.
    tf_d_model: usize,
    speakers: usize,
    /// The streaming knobs the PEE writes over the module's at `_init_streaming_diar`: `diar_fifo_len`,
    /// `diar_spkcache_len`, `diar_spkcache_update_period` (encoder rows), and `high_resolution`.
    /// PEE overrides the diarizer's output_subsampling_factor with the ASR encoder's factor before
    /// constructing it, so native 10 ms probabilities are pooled back onto that encoder grid.
    fifo_len: usize,
    spkcache_len: usize,
    update_period: usize,
    output_factor: usize,

    /// A head's width (`d_model`/`heads`).
    pub fn headDim(self: Diar) usize {
        return self.d_model / self.heads;
    }

    /// The FFN's width (`d_model` x `ff_expansion`).
    pub fn ffWidth(self: Diar) usize {
        return self.d_model * self.ff_expansion;
    }

    /// `pre_encode.proj`'s inputs: `feat_in` mels x `subsampling` stacked.
    pub fn stacked(self: Diar) usize {
        return self.feat_in * self.subsampling;
    }

    /// `pe_encoder_config.diarization_model_cfg` (a sibling of `asr_encoder_cfg`); absent without a PEE,
    /// where this parse never runs. `enc` is its `encoder` object, `mods` its `sortformer_modules`, `pe`
    /// the bundle's own streaming fields.
    pub fn parse(p: Parser, pe: json.ObjectMap) Error!Diar {
        const map = try p.object(pe, "diarization_model_cfg");
        const enc = try p.object(map, "encoder");
        const mods = try p.object(map, "sortformer_modules");
        if (p.flagOr(enc, "qk_norm", false))
            return p.fail(error.UnsupportedModel, "config.json: the diarization encoder's qk_norm is set; zspark's Sortformer layers have none", .{});
        if (p.flagOr(enc, "xscaling", false))
            return p.fail(error.UnsupportedModel, "config.json: the diarization encoder's xscaling is set", .{});
        if (p.flagOr(enc, "qkv_bias", false))
            return p.fail(error.UnsupportedModel, "config.json: the diarization encoder's w_qkv carries a bias", .{});
        if (!std.mem.eql(u8, p.stringOr(enc, "subsampling", "feature_stacking"), "feature_stacking"))
            return p.fail(error.UnsupportedModel, "config.json: the diarization encoder's subsampling is \"{s}\"; zspark reads feature_stacking", .{p.stringOr(enc, "subsampling", "")});
        if (!std.mem.eql(u8, p.stringOr(enc, "attn_mode", "full"), "full"))
            return p.fail(error.UnsupportedModel, "config.json: the diarization encoder's attn_mode is \"{s}\"; zspark reads full attention", .{p.stringOr(enc, "attn_mode", "")});
        if (!std.mem.eql(u8, p.stringOr(enc, "self_attention_model", "rope"), "rope"))
            return p.fail(error.UnsupportedModel, "config.json: the diarization encoder's self_attention_model is \"{s}\", not RoPE", .{p.stringOr(enc, "self_attention_model", "")});
        if (!p.flagOr(map, "high_resolution", false))
            return p.fail(error.UnsupportedModel, "config.json: the diarization model is not high_resolution; zspark reads the 10 ms preds it upsamples", .{});
        const self: Diar = .{
            .feat_in = try p.uint(enc, "feat_in"),
            .d_model = try p.uint(enc, "d_model"),
            .heads = try p.uint(enc, "n_heads"),
            .layers = try p.uint(enc, "n_layers"),
            .ff_expansion = @intFromFloat(numberOr(enc, "ff_expansion", 4)),
            .subsampling = try p.uint(enc, "subsampling_factor"),
            .rope_base = numberOr(enc, "rope_base", 10000),
            .rotary_fraction = numberOr(enc, "rotary_fraction", 1.0),
            .tf_d_model = @intFromFloat(numberOr(mods, "tf_d_model", 192)),
            .speakers = @intFromFloat(numberOr(mods, "num_spks", 8)),
            .fifo_len = @intFromFloat(numberOr(pe, "diar_fifo_len", 40)),
            .spkcache_len = @intFromFloat(numberOr(pe, "diar_spkcache_len", 188)),
            .update_period = @intFromFloat(numberOr(pe, "diar_spkcache_update_period", 300)),
            .output_factor = try p.uint(enc, "subsampling_factor"),
        };
        if (self.heads == 0 or self.d_model % self.heads != 0)
            return p.fail(error.InvalidConfig, "config.json: the diarization encoder's d_model {d} over {d} heads", .{ self.d_model, self.heads });
        if (self.headDim() % 32 != 0 or self.headDim() > 128)
            return p.fail(error.UnsupportedModel, "config.json: the diarization encoder's head width {d}; zspark's prep kernel wants 32s to 128", .{self.headDim()});
        if (self.feat_in == 0 or self.subsampling == 0 or self.speakers == 0)
            return p.fail(error.InvalidConfig, "config.json: the diarization encoder's feat_in {d}, subsampling {d}, speakers {d}", .{ self.feat_in, self.subsampling, self.speakers });
        return self;
    }
};

/// `hf_quant_config.json`'s `quantization`: the ModelOpt MIXED_PRECISION map `quantized_layers` (a name
/// covers every linear beneath it; one absent is BF16) and `exclude_modules` beside it.
pub const Quant = struct {
    layers: []const modelopt.QuantizedLayer,
    /// Names and `*`/`?` wildcards the export left BF16; informational (a module `layers` does not name is
    /// BF16 anyway). `ctc_timestamp*` is among them.
    exclude: []const []const u8 = &.{},
    /// `kv_cache_quant_algo`, null on the NVFP4 build (the card's fp8 KV cache is its launch flag's).
    kv_cache: ?[]const u8 = null,

    /// The entry covering module `name`: the map's names drop a tensor's leading `llm.` (`llm.mtp.layers.N.*`
    /// is `mtp.layers.N.*` here).
    pub fn find(self: Quant, name: []const u8) ?modelopt.QuantizedLayer {
        return modelopt.find(self.layers, mapped(name));
    }

    /// `find`'s entry by its place in `layers`.
    pub fn findIndex(self: Quant, name: []const u8) ?usize {
        return modelopt.findIndex(self.layers, mapped(name));
    }

    fn mapped(name: []const u8) []const u8 {
        return if (std.mem.startsWith(u8, name, "llm.")) name["llm.".len..] else name;
    }

    /// `text` is `hf_quant_config.json`. The result lives in `arena`.
    pub fn parse(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) Error!Quant {
        const p: Parser = .{ .arena = arena, .diagnostic = diagnostic, .where = "hf_quant_config.json" };
        const root = json.parseFromSliceLeaky(json.Value, arena, text, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return p.fail(error.InvalidConfig, "{s} is not valid JSON ({t})", .{ p.where, err }),
        };
        if (root != .object) return p.fail(error.InvalidConfig, "{s} is not a JSON object", .{p.where});
        const formats = "zspark reads Nemotron-3.5-Transcribe's NVFP4 build (ModelOpt MIXED_PRECISION: NVFP4 in groups of 16 with plain scales and per-tensor FP8)";
        const block = try p.object(root.object, "quantization");
        const algo_name = try string(p, block, "quant_algo");
        if (!std.ascii.eqlIgnoreCase(algo_name, "MIXED_PRECISION"))
            return p.fail(error.UnsupportedModel, "{s} stores {s} throughout; {s}", .{ p.where, algo_name, formats });
        var self: Quant = .{ .layers = &.{} };
        if (block.get("kv_cache_quant_algo")) |cached| if (cached == .string) {
            self.kv_cache = cached.string;
        };
        const listed = try p.object(block, "quantized_layers");
        const layers = try p.arena.alloc(modelopt.QuantizedLayer, listed.count());
        for (listed.keys(), listed.values(), layers) |name, value, *layer| {
            if (value != .object) return p.fail(error.InvalidConfig, "{s}: quantized_layers.{s} is not an object", .{ p.where, name });
            layer.* = modelopt.entry(name, value.object) catch |err| return switch (err) {
                error.UnknownAlgo, error.UnknownEncoding => p.fail(error.UnsupportedModel, "{s} is stored as {f}; {s}", .{ name, json.fmt(value, .{}), formats }),
                error.InvalidEntry => p.fail(error.InvalidConfig, "{s}: quantized_layers.{s} is {f}, no ModelOpt entry", .{ p.where, name, json.fmt(value, .{}) }),
            };
            if ((layer.algo != .nvfp4 and layer.algo != .fp8) or modelopt.Scheme.of(layer.*) == null)
                return p.fail(error.UnsupportedModel, "{s} is stored as {f}: {s}", .{ name, json.fmt(value, .{}), formats });
        }
        self.layers = layers;
        if (block.get("exclude_modules")) |list| {
            if (list != .array) return p.fail(error.InvalidConfig, "{s}: exclude_modules is not a list", .{p.where});
            const exclude = try p.arena.alloc([]const u8, list.array.items.len);
            for (exclude, list.array.items) |*name, item| {
                if (item != .string) return p.fail(error.InvalidConfig, "{s}: exclude_modules holds a non-string", .{p.where});
                name.* = item.string;
            }
            self.exclude = exclude;
        }
        return self;
    }
};

/// The checkpoint's own shape, from its three config files.
pub const Config = struct {
    /// `prompt_format` ("nemotron3p5") and the tag a prompt's audio rows carry (`audio_locator_tag`).
    prompt_format: []const u8,
    audio_locator: []const u8,
    /// `encoder_chunk_size_seconds`: a perception chunk's span (30 s).
    chunk_seconds: f64,
    /// Whether the checkpoint mounts a ParallelExpertEncoder (`pe_encoder_path` or `pe_encoder_config`
    /// present and non-empty; the vLLM plugin's `_config_has_pe_encoder`): it owns the encoder's windowing,
    /// so a clip's prompt rows are the estimator's single pass, not the 30 s chunk split.
    pe_encoder: bool,
    /// `pe_encoder_config`'s online windowing, in the encoder's output rows: a window's core
    /// (`online_inference_length`, NeMo's default 500) and its left and right context
    /// (`chunk_left_context`/`chunk_right_context`, 50). Mel frames a side: `x subsampling`.
    pe_core: usize,
    pe_left: usize,
    pe_right: usize,
    /// `pe_encoder_config`'s `asr_normalize_type` (this checkpoint's "per_feature" on the mel before the
    /// windows; null where the bundle normalizes differently — only per_feature is served).
    pe_norm: bool,
    /// The Sortformer diarization expert's shape and streaming knobs (`pe_encoder_config.
    /// diarization_model_cfg` and the bundle's `diar_*` fields); absent without a PEE.
    diar: ?Diar,
    /// config.json's `eos_token_id`, generation_config.json's merged in (`read`).
    eos: []const u32,
    pad: u32,
    backbone: Backbone,
    perception: Perception,
    quant: Quant,

    /// Reads `dir`'s config.json, `llm_backbone/config.json` and `hf_quant_config.json`.
    pub fn read(arena: Allocator, io: Io, dir: Io.Dir, diagnostic: *Diagnostic) (Error || Io.Dir.ReadFileAllocError)!Config {
        const text = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 26));
        const backbone = try dir.readFileAlloc(io, "llm_backbone/config.json", arena, .limited(1 << 26));
        const quant = try dir.readFileAlloc(io, "hf_quant_config.json", arena, .limited(1 << 26));
        var self = try parse(arena, text, backbone, quant, diagnostic);
        if (dir.readFileAlloc(io, "generation_config.json", arena, .limited(1 << 24))) |generated| {
            self.eos = try mergeEos(arena, self.eos, generated, diagnostic);
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }
        return self;
    }

    /// `text` is config.json (config.json's embedded `llm_config` repeats `llm_backbone/config.json`; the file
    /// is read instead, `backbone` its text), `quant` hf_quant_config.json. The result lives in `arena`.
    pub fn parse(arena: Allocator, text: []const u8, backbone: []const u8, quant: []const u8, diagnostic: *Diagnostic) Error!Config {
        const p: Parser = .{ .arena = arena, .diagnostic = diagnostic, .where = "config.json" };
        const root = json.parseFromSliceLeaky(json.Value, arena, text, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return p.fail(error.InvalidConfig, "config.json is not valid JSON ({t})", .{err}),
        };
        if (root != .object) return p.fail(error.InvalidConfig, "config.json is not a JSON object", .{});
        const raw = root.object;
        const model_type = if (raw.get("model_type")) |v| (if (v == .string) v.string else "") else "";
        if (!std.mem.eql(u8, model_type, model_types[0]))
            return p.fail(error.UnsupportedModel, "config.json describes model_type \"{s}\", not Nemotron-3.5-Transcribe ({s})", .{ model_type, model_types[0] });
        const pe_map: json.ObjectMap = if (raw.get("pe_encoder_config")) |v| (if (v == .object) v.object else .{}) else .{};
        const has_pe = hasKey(raw, "pe_encoder_path") or hasKey(raw, "pe_encoder_config");
        const pe_norm_name = if (pe_map.get("asr_normalize_type")) |v| (if (v == .string) v.string else "per_feature") else "per_feature";
        if (has_pe and !std.mem.eql(u8, pe_norm_name, "per_feature"))
            return p.fail(error.UnsupportedModel, "config.json: the encoder's asr_normalize_type is \"{s}\"; zspark serves per_feature", .{pe_norm_name});
        const self: Config = .{
            .prompt_format = try string(p, raw, "prompt_format"),
            .audio_locator = try string(p, raw, "audio_locator_tag"),
            .chunk_seconds = numberOr(raw, "encoder_chunk_size_seconds", 30.0),
            .eos = try ids(p, raw, "eos_token_id"),
            .pad = @intCast(try p.uint(raw, "pad_token_id")),
            .pe_encoder = has_pe,
            .pe_core = @intFromFloat(numberOr(pe_map, "online_inference_length", 500)),
            .pe_left = @intFromFloat(numberOr(pe_map, "chunk_left_context", 50)),
            .pe_right = @intFromFloat(numberOr(pe_map, "chunk_right_context", 50)),
            .pe_norm = has_pe,
            .diar = if (has_pe and pe_map.get("diarization_model_cfg") != null) try Diar.parse(p, pe_map) else null,
            .backbone = try Backbone.parse(arena, backbone, diagnostic),
            .perception = try Perception.parse(p, raw),
            .quant = try Quant.parse(arena, quant, diagnostic),
        };
        if (self.perception.output_dim != self.backbone.hidden)
            return p.fail(error.UnsupportedModel, "config.json: the perception projects to {d}, the backbone's hidden is {d}", .{ self.perception.output_dim, self.backbone.hidden });
        return self;
    }
};

/// The plugin's `_config_has_pe_encoder` test on one field: present and not null, `""`, `false` or `{}`.
fn hasKey(fields: json.ObjectMap, key: []const u8) bool {
    const v = fields.get(key) orelse return false;
    return switch (v) {
        .null, .bool => false,
        .string => |s| s.len > 0,
        .object => |o| o.count() > 0,
        else => true,
    };
}

/// config.json's eos ids plus generation_config.json's (the served model's), in `arena`.
fn mergeEos(arena: Allocator, eos: []const u32, text: []const u8, diagnostic: *Diagnostic) Error![]const u32 {
    const p: Parser = .{ .arena = arena, .diagnostic = diagnostic, .where = "generation_config.json" };
    const root = json.parseFromSliceLeaky(json.Value, arena, text, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return p.fail(error.InvalidConfig, "generation_config.json is not valid JSON ({t})", .{err}),
    };
    if (root != .object or root.object.get("eos_token_id") == null) return eos;
    const extra = try ids(p, root.object, "eos_token_id");
    var out: std.ArrayList(u32) = .empty;
    try out.appendSlice(arena, eos);
    for (extra) |id| {
        for (eos) |known| if (known == id) break;
        try out.append(arena, id);
    }
    return out.items;
}

/// A required field's value, `where` naming its file.
fn field(p: Parser, fields: json.ObjectMap, key: []const u8) Error!json.Value {
    return fields.get(key) orelse p.fail(error.InvalidConfig, "{s} has no {s}", .{ p.where, key });
}

/// A required number.
fn number(p: Parser, fields: json.ObjectMap, key: []const u8) Error!f64 {
    return switch (try field(p, fields, key)) {
        .integer => |v| @floatFromInt(v),
        .float => |v| v,
        else => p.fail(error.InvalidConfig, "{s}: {s} is not a number", .{ p.where, key }),
    };
}

/// `fields[key]` as f64, `default` where it is absent or no number.
fn numberOr(fields: json.ObjectMap, key: []const u8, default: f64) f64 {
    return switch (fields.get(key) orelse return default) {
        .integer => |v| @floatFromInt(v),
        .float => |v| v,
        else => default,
    };
}

/// Inference clamp, independent of the parameter initializer's time_step_min/max.
/// ModelOpt serializes infinity as {"__float__":"Infinity"} in this checkpoint.
fn timeStepLimit(p: Parser, fields: json.ObjectMap) Error![2]f64 {
    const value = fields.get("time_step_limit") orelse return .{ 0, std.math.inf(f64) };
    if (value != .array or value.array.items.len != 2)
        return p.fail(error.InvalidConfig, "{s}: time_step_limit must hold two bounds", .{p.where});
    var bounds: [2]f64 = undefined;
    for (value.array.items, &bounds) |item, *bound| {
        bound.* = switch (item) {
            .integer => |v| @floatFromInt(v),
            .float => |v| v,
            .object => |object| blk: {
                const tag = object.get("__float__") orelse return p.fail(error.InvalidConfig, "{s}: time_step_limit has a nonnumeric bound", .{p.where});
                if (tag == .string and std.mem.eql(u8, tag.string, "Infinity")) break :blk std.math.inf(f64);
                return p.fail(error.InvalidConfig, "{s}: time_step_limit has an invalid infinity tag", .{p.where});
            },
            else => return p.fail(error.InvalidConfig, "{s}: time_step_limit has a nonnumeric bound", .{p.where}),
        };
    }
    if (!std.math.isFinite(bounds[0]) or bounds[0] < 0 or std.math.isNan(bounds[1]) or bounds[1] < bounds[0])
        return p.fail(error.InvalidConfig, "{s}: time_step_limit requires 0 <= lower <= upper", .{p.where});
    return bounds;
}

/// A required string.
fn string(p: Parser, fields: json.ObjectMap, key: []const u8) Error![]const u8 {
    const value = try field(p, fields, key);
    return if (value == .string) value.string else p.fail(error.InvalidConfig, "{s}: {s} is not a string", .{ p.where, key });
}

/// `fields[key]` true unless it says false or is absent.
fn flagOrTrue(fields: json.ObjectMap, key: []const u8) bool {
    const value = fields.get(key) orelse return true;
    return if (value == .bool) value.bool else true;
}

/// A list of `LayerType`'s names (`layers_block_type`, `mtp_layers_block_type`).
fn kinds(p: Parser, fields: json.ObjectMap, key: []const u8) Error![]const LayerType {
    const value = try field(p, fields, key);
    if (value != .array) return p.fail(error.InvalidConfig, "{s}: {s} is not a list", .{ p.where, key });
    const out = try p.arena.alloc(LayerType, value.array.items.len);
    for (out, value.array.items, 0..) |*kind, item, at| {
        const name = if (item == .string) item.string else "";
        kind.* = std.meta.stringToEnum(LayerType, name) orelse
            return p.fail(error.UnsupportedModel, "{s}: {s}[{d}] is \"{s}\", which Nemotron has no layer for", .{ p.where, key, at, name });
    }
    return out;
}

/// One token id or a list of them.
fn ids(p: Parser, fields: json.ObjectMap, key: []const u8) Error![]const u32 {
    const value = try field(p, fields, key);
    const items: []const json.Value = if (value == .array) value.array.items else &.{value};
    const out = try p.arena.alloc(u32, items.len);
    for (out, items) |*id, item| {
        if (item != .integer or item.integer < 0) return p.fail(error.InvalidConfig, "{s}: {s} is not a token id or a list of them", .{ p.where, key });
        id.* = @intCast(item.integer);
    }
    return out;
}
