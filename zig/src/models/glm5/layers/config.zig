//! GLM-5.3-Flash's config.json (`model_type` `glm5_next`, TensorFold's family `glm5_next`): the text model's
//! shape as `families/glm5_next/cuda/weights.py:Config` reads it, and what this port refuses. Host only.

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const modelopt = @import("../../../weights/modelopt.zig");

/// Why a config was refused, in the model's own words.
pub const Diagnostic = struct {
    message: []const u8 = "",

    pub fn set(self: *Diagnostic, arena: Allocator, comptime format: []const u8, args: anytype) void {
        self.message = std.fmt.allocPrint(arena, format, args) catch "out of memory";
    }
};

pub const Error = error{ InvalidConfig, UnsupportedModel, UnsupportedQuantization } || Allocator.Error;

/// The name config.json's `model_type` carries.
pub const model_types = [_][]const u8{"glm5_next"};

pub const LayerType = enum {
    /// KDA: a gated delta rule on a state a head (`self_attn` with `q/k/v_proj`, `A_log`).
    linear_attention,
    /// DSA: attention over latents its indexer selects (`self_attn` with `q_a_proj`, `indexer`).
    deepseek_sparse_attention,
};

/// How the checkpoint stores its linears: a ModelOpt MIXED_PRECISION checkpoint (NVIDIA's FP4-Local), each linear
/// as `quantized_layers` names it, one it does not name BF16 (`weights/modelopt.zig`).
pub const ModelOpt = struct {
    /// `quantized_layers`, the vision tower's among them.
    layers: []const modelopt.QuantizedLayer,

    /// The entry covering module or tensor `name`, if any.
    pub fn find(self: ModelOpt, name: []const u8) ?modelopt.QuantizedLayer {
        return modelopt.find(self.layers, name);
    }
};

pub const MlpType = enum {
    /// One MLP of `intermediate` (`mlp.gate_proj`).
    dense,
    /// Routed experts and a shared one (`mlp.experts.E`, `mlp.shared_experts`).
    sparse,
};

/// The text model. Slices live in the arena `parse` was given.
pub const Config = struct {
    model_type: []const u8,
    hidden: usize,
    vocab: usize,
    /// Layers of the model, the MTP's (`layers.N` from `layers` on) not among them.
    layers: usize,
    layer_types: []const LayerType,
    mlp_types: []const MlpType,
    eps: f64,
    eos: []const u32,

    /// Residual streams (`hc_mult`), the Sinkhorn's iterations and its epsilon.
    streams: usize,
    sinkhorn: usize,
    hc_eps: f64,

    /// KDA: heads, a head's width, the depthwise conv's taps, the gate's lower bound.
    kda_heads: usize,
    kda_head_dim: usize,
    conv_kernel: usize,
    gate_lower_bound: f64,

    /// DSA: heads, the query's and the latent's ranks, a head's key and value widths.
    heads: usize,
    q_rank: usize,
    kv_rank: usize,
    key_dim: usize,
    value_dim: usize,
    /// The indexer: heads, a head's width, the tokens it selects, a pool's tokens, the tail always kept.
    index_heads: usize,
    index_dim: usize,
    index_top: usize,
    index_pool: usize,
    index_tail: bool,

    /// The dense MLP's width; the routed experts, those chosen a token, an expert's width, the weights' scale,
    /// and whether the chosen experts' weights are divided by their sum first (`norm_topk_prob`, true when absent
    /// as in TensorFold).
    intermediate: usize,
    experts: usize,
    top_k: usize,
    moe_width: usize,
    routed_scale: f64,
    norm_topk: bool,
    swiglu_limit: f64,

    /// MTP layers after the model's (`num_nextn_predict_layers`).
    mtp_layers: usize,
    /// The checkpoint's storage: its ModelOpt map.
    quant: ModelOpt,

    pub fn count(self: Config, kind: LayerType) usize {
        return std.mem.count(LayerType, self.layer_types, &.{kind});
    }

    pub fn countMlp(self: Config, kind: MlpType) usize {
        return std.mem.count(MlpType, self.mlp_types, &.{kind});
    }

    /// A hyper-connection site's mix dots: `streams` for the read-out, `streams` for the write and
    /// `streams` squared for the mix among them.
    pub fn mix(self: Config) usize {
        return self.streams * (self.streams + 2);
    }

    /// Reads `dir`'s config.json.
    pub fn read(arena: Allocator, io: Io, dir: Io.Dir, diagnostic: *Diagnostic) (Error || Io.Dir.ReadFileAllocError)!Config {
        const text = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 26));
        return parse(arena, text, diagnostic);
    }

    /// `text` is config.json. The result lives in `arena`.
    pub fn parse(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) Error!Config {
        const p: Parser = .{ .arena = arena, .diagnostic = diagnostic };
        const root = json.parseFromSliceLeaky(json.Value, arena, text, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return p.fail(error.InvalidConfig, "config.json is not valid JSON ({t})", .{err}),
        };
        if (root != .object) return p.fail(error.InvalidConfig, "config.json is not a JSON object", .{});
        const raw = root.object;
        const model_type = if (raw.get("model_type")) |v| (if (v == .string) v.string else "") else "";
        if (!std.mem.eql(u8, model_type, model_types[0]))
            return p.fail(error.UnsupportedModel, "config.json describes model_type \"{s}\", not GLM-5.3-Flash ({s})", .{ model_type, model_types[0] });
        const t = try p.object(raw, "text_config");

        const quant = try quantization(p, raw);
        const layers = try p.uint(t, "num_hidden_layers");
        const layer_types = try p.kinds(LayerType, t, "layer_types");
        const mlp_types = try p.kinds(MlpType, t, "mlp_layer_types");
        if (layer_types.len != layers or mlp_types.len != layers)
            return p.fail(error.InvalidConfig, "config.json: layer_types names {d} layers and mlp_layer_types {d}, num_hidden_layers {d}", .{ layer_types.len, mlp_types.len, layers });
        const kda = try p.object(t, "linear_attn_config");

        const self: Config = .{
            .model_type = model_type,
            .hidden = try p.uint(t, "hidden_size"),
            .vocab = try p.uint(t, "vocab_size"),
            .layers = layers,
            .layer_types = layer_types,
            .mlp_types = mlp_types,
            .eps = try p.number(t, "rms_norm_eps"),
            .eos = try p.ids(t, "eos_token_id"),
            .streams = try p.uint(t, "hc_mult"),
            .sinkhorn = try p.uint(t, "hc_sinkhorn_iters"),
            .hc_eps = try p.number(t, "hc_eps"),
            .kda_heads = try p.uint(kda, "num_heads"),
            .kda_head_dim = try p.uint(kda, "head_dim"),
            .conv_kernel = try p.uint(kda, "short_conv_kernel_size"),
            .gate_lower_bound = try p.number(kda, "gate_lower_bound"),
            .heads = try p.uint(t, "num_attention_heads"),
            .q_rank = try p.uint(t, "q_lora_rank"),
            .kv_rank = try p.uint(t, "kv_lora_rank"),
            .key_dim = try p.uint(t, "qk_nope_head_dim"),
            .value_dim = try p.uint(t, "v_head_dim"),
            .index_heads = try p.uint(t, "index_n_heads"),
            .index_dim = try p.uint(t, "index_head_dim"),
            .index_top = try p.uint(t, "index_topk"),
            .index_pool = try p.uint(t, "index_kpool"),
            .index_tail = try p.flag(t, "index_kpool_always_select_tail"),
            .intermediate = try p.uint(t, "intermediate_size"),
            .experts = try p.uint(t, "n_routed_experts"),
            .top_k = try p.uint(t, "num_experts_per_tok"),
            .moe_width = try p.uint(t, "moe_intermediate_size"),
            .routed_scale = try p.number(t, "routed_scaling_factor"),
            .norm_topk = if (t.get("norm_topk_prob") == null) true else try p.flag(t, "norm_topk_prob"),
            .swiglu_limit = try p.number(t, "swiglu_limit"),
            .mtp_layers = try p.uint(t, "num_nextn_predict_layers"),
            .quant = quant,
        };

        // What the forward is written for; each is a number of the config this port reads as a fact.
        if (try p.uint(t, "qk_rope_head_dim") != 0)
            return p.fail(error.UnsupportedModel, "config.json: qk_rope_head_dim is {d}; zspark's DSA has no positions (0)", .{try p.uint(t, "qk_rope_head_dim")});
        if (try p.uint(t, "n_shared_experts") != 1)
            return p.fail(error.UnsupportedModel, "config.json: n_shared_experts is {d}; zspark reads one shared expert", .{try p.uint(t, "n_shared_experts")});
        const scoring = try p.string(t, "scoring_func");
        if (!std.mem.eql(u8, scoring, "sigmoid"))
            return p.fail(error.UnsupportedModel, "config.json: scoring_func is \"{s}\"; zspark's router scores with a sigmoid", .{scoring});
        if (self.streams != 4)
            return p.fail(error.UnsupportedModel, "config.json: hc_mult is {d}; zspark's hyper-connection kernels are written for four streams", .{self.streams});
        if (self.kda_head_dim != 128 or self.conv_kernel != 4)
            return p.fail(error.UnsupportedModel, "config.json: KDA heads of {d} with a conv of {d} taps; zspark's chain is written for heads of 128 and four taps", .{ self.kda_head_dim, self.conv_kernel });
        if (self.kv_rank != 512)
            return p.fail(error.UnsupportedModel, "config.json: kv_lora_rank is {d}; zspark's DSA attends over latents of 512", .{self.kv_rank});
        if (self.mtp_layers > 1)
            return p.fail(error.UnsupportedModel, "config.json: {d} MTP layers; zspark reads one or none", .{self.mtp_layers});
        if (self.heads % 2 != 0 or self.kda_heads % 2 != 0 or self.vocab % 2 != 0 or self.intermediate % 2 != 0 or self.moe_width % 32 != 0)
            return p.fail(error.InvalidConfig, "config.json: {d} DSA heads, {d} KDA heads, vocabulary {d}, widths {d} and {d}: two ranks each take half of the heads, the vocabulary and the widths", .{ self.heads, self.kda_heads, self.vocab, self.intermediate, self.moe_width });
        if (self.top_k == 0 or self.top_k > self.experts)
            return p.fail(error.InvalidConfig, "config.json: {d} experts a token of {d}", .{ self.top_k, self.experts });
        if (self.index_pool == 0 or self.index_top % self.index_pool != 0)
            return p.fail(error.InvalidConfig, "config.json: index_topk {d} is not whole pools of {d}", .{ self.index_top, self.index_pool });
        return self;
    }
};

/// The checkpoint's storage, read as vLLM's `modelopt_mixed` reads it (spec 107). TensorFold read only EXL3 packs
/// of this family, retired with their code (tag `retired-checkpoints-2026-10-21`, `docs/recipes/README.md`).
fn quantization(p: Parser, raw: json.ObjectMap) Error!ModelOpt {
    const formats = "zspark reads GLM-5.3-Flash from NVIDIA's FP4-Local (ModelOpt MIXED_PRECISION)";
    if (raw.get("quantization_config")) |block| if (block == .object) if (block.object.get("quant_method")) |method| {
        if (method == .string and std.ascii.eqlIgnoreCase(method.string, "modelopt")) return modeloptMap(p, block.object, formats);
    };
    return p.fail(error.UnsupportedQuantization, "config.json names no ModelOpt quantization: {s}, not from an unquantized checkpoint", .{formats});
}

/// `quantized_layers` as GLM stores it: MXFP8 and FP8_PB_WO beside the experts, NVFP4 (csf_32x2_v1 or plain
/// scales, W4A16_NVFP4 among it) only in the routed experts and the vision tower, nothing per-tensor FP8.
fn modeloptMap(p: Parser, block: json.ObjectMap, formats: []const u8) Error!ModelOpt {
    const algo = try p.string(block, "quant_algo");
    if (!std.ascii.eqlIgnoreCase(algo, "MIXED_PRECISION"))
        return p.fail(error.UnsupportedQuantization, "this ModelOpt checkpoint is {s} throughout; {s}", .{ algo, formats });
    const listed = try p.object(block, "quantized_layers");
    const layers = try p.arena.alloc(modelopt.QuantizedLayer, listed.count());
    for (listed.keys(), listed.values(), layers) |name, value, *layer| {
        if (value != .object) return p.fail(error.InvalidConfig, "config.json: quantized_layers.{s} is not an object", .{name});
        layer.* = modelopt.entry(name, value.object) catch |err| return switch (err) {
            error.UnknownAlgo, error.UnknownEncoding => p.fail(error.UnsupportedQuantization, "{s} is stored as {f}; {s}", .{ name, json.fmt(value, .{}), formats }),
            error.InvalidEntry => p.fail(error.InvalidConfig, "config.json: quantized_layers.{s} is {f}, no ModelOpt entry", .{ name, json.fmt(value, .{}) }),
        };
        const experts = std.mem.indexOf(u8, name, ".mlp.experts") != null or std.mem.startsWith(u8, name, "model.visual.");
        if (modelopt.Scheme.of(layer.*) == null or layer.algo == .fp8 or (layer.algo == .nvfp4 and !experts))
            return p.fail(error.UnsupportedQuantization, "{s} is stored as {f}: zspark reads MXFP8 (32) and FP8_PB_WO (128) linears, NVFP4 (16, scales plain or csf_32x2_v1) in the routed experts", .{ name, json.fmt(value, .{}) });
    }
    return .{ .layers = layers };
}

/// config.json's fields by kind, each refusal a line in the diagnostic.
pub const Parser = struct {
    arena: Allocator,
    diagnostic: *Diagnostic,

    pub fn fail(p: Parser, err: Error, comptime format: []const u8, args: anytype) Error {
        p.diagnostic.set(p.arena, format, args);
        return err;
    }

    pub fn field(p: Parser, fields: json.ObjectMap, key: []const u8) Error!json.Value {
        return fields.get(key) orelse p.fail(error.InvalidConfig, "config.json has no {s}", .{key});
    }

    pub fn object(p: Parser, fields: json.ObjectMap, key: []const u8) Error!json.ObjectMap {
        const value = try p.field(fields, key);
        return if (value == .object) value.object else p.fail(error.InvalidConfig, "config.json: {s} is not an object", .{key});
    }

    pub fn uint(p: Parser, fields: json.ObjectMap, key: []const u8) Error!usize {
        const value = try p.field(fields, key);
        if (value == .integer and value.integer >= 0) return @intCast(value.integer);
        return p.fail(error.InvalidConfig, "config.json: {s} is not a count", .{key});
    }

    pub fn number(p: Parser, fields: json.ObjectMap, key: []const u8) Error!f64 {
        return switch (try p.field(fields, key)) {
            .integer => |v| @floatFromInt(v),
            .float => |v| v,
            else => p.fail(error.InvalidConfig, "config.json: {s} is not a number", .{key}),
        };
    }

    pub fn flag(p: Parser, fields: json.ObjectMap, key: []const u8) Error!bool {
        const value = try p.field(fields, key);
        return if (value == .bool) value.bool else p.fail(error.InvalidConfig, "config.json: {s} is not true or false", .{key});
    }

    pub fn string(p: Parser, fields: json.ObjectMap, key: []const u8) Error![]const u8 {
        const value = try p.field(fields, key);
        return if (value == .string) value.string else p.fail(error.InvalidConfig, "config.json: {s} is not a string", .{key});
    }

    /// A list of an enum's names.
    pub fn kinds(p: Parser, comptime Kind: type, fields: json.ObjectMap, key: []const u8) Error![]const Kind {
        const value = try p.field(fields, key);
        if (value != .array) return p.fail(error.InvalidConfig, "config.json: {s} is not a list", .{key});
        const out = try p.arena.alloc(Kind, value.array.items.len);
        for (out, value.array.items, 0..) |*kind, item, at| {
            const name = if (item == .string) item.string else "";
            kind.* = std.meta.stringToEnum(Kind, name) orelse
                return p.fail(error.UnsupportedModel, "config.json: {s}[{d}] is \"{s}\", which zspark's GLM-5.3-Flash has no layer for", .{ key, at, name });
        }
        return out;
    }

    /// One id or a list of them.
    pub fn ids(p: Parser, fields: json.ObjectMap, key: []const u8) Error![]const u32 {
        const value = try p.field(fields, key);
        const items: []const json.Value = if (value == .array) value.array.items else &.{value};
        const out = try p.arena.alloc(u32, items.len);
        for (out, items) |*id, item| {
            if (item != .integer or item.integer < 0) return p.fail(error.InvalidConfig, "config.json: {s} is not a token id or a list of them", .{key});
            id.* = @intCast(item.integer);
        }
        return out;
    }
};
