//! A Qwen3.8 Flash Next checkpoint's config.json: the sizes the forward needs and the quantization block.
//! Fields, defaults and derived values follow TensorFold (families/qwen4_exp/cuda/weight_types.py); the
//! quantization preflight follows families/qwen4_exp/__init__.py. ModelOpt checkpoints that store FP4-Local's
//! formats pass: NVFP4 experts and n-gram tables, MXFP8 or BF16 linears beside them.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const json = std.json;
const modelopt = @import("../../weights/modelopt.zig");

/// Why a checkpoint was refused, in words for the user; set whenever a function taking one fails with
/// anything but OutOfMemory or an I/O error.
pub const Diagnostic = struct {
    message: []const u8 = "",

    pub fn set(self: *Diagnostic, arena: Allocator, comptime format: []const u8, args: anytype) void {
        self.message = std.fmt.allocPrint(arena, format, args) catch "out of memory";
    }
};

pub const Error = error{ InvalidConfig, UnsupportedModel, UnsupportedQuantization } || Allocator.Error;

/// The names config.json's `model_type` carries; the second is what newer exports use.
pub const model_types = [_][]const u8{ "qwen4_exp", "qwen3_8_flash_next" };

pub const LayerType = enum {
    /// A gated DeltaNet block (`linear_attn`).
    linear_attention,
    /// Attention over the keys its indexer selects (`self_attn`).
    full_attention,
};

/// The quantization map's types are shared (`weights/modelopt.zig`); this config refuses what Flash Next does not
/// store (csf_32x2_v1 scales, GLM's).
pub const Algo = modelopt.Algo;
pub const ScaleEncoding = modelopt.ScaleEncoding;
pub const QuantizedLayer = modelopt.QuantizedLayer;

/// One entry of `config_groups`: the modules sharing a weight and activation format.
pub const Group = struct {
    name: []const u8,
    weight_bits: u32,
    weight_group_size: ?u32,
    /// Null where activations stay bf16.
    input_bits: ?u32,
    input_group_size: ?u32,
    input_dynamic: bool,
    targets: []const []const u8,
};

pub const Quantization = struct {
    /// ModelOpt's `quant_algo`, upper case: one algorithm, or MIXED_PRECISION with `layers` naming each.
    algo: []const u8,
    /// Inputs sharing an NVFP4 block scale (`config_groups.group_0.weights.group_size`).
    nvfp4_group: u32,
    layers: []const QuantizedLayer,
    groups: []const Group,

    /// The entry covering module or tensor `name`, if any.
    pub fn find(self: Quantization, name: []const u8) ?QuantizedLayer {
        return modelopt.find(self.layers, name);
    }
};

/// The multi-token-prediction head's block of the text config.
pub const Mtp = struct {
    layers: u32,
    layer_types: []const LayerType,
    rope_theta: f64,
    hybrid: bool,
    /// `mtp_use_hidden_state_from_layer`: null takes the last layer's streams.
    hidden_state_from_layer: ?u32,
    /// `mtp_use_dedicated_embeddings`: false shares the model's token embedding.
    dedicated_embeddings: bool,
};

pub const Config = struct {
    model_type: []const u8,
    hidden: u32,
    layers: u32,
    layer_types: []const LayerType,
    vocab: u32,
    /// `rms_norm_eps`
    eps: f64,
    max_positions: u32,

    // attention
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rope_theta: f64,
    /// Leading dimensions of a head that rotate: head_dim * partial_rotary_factor.
    rotary_dim: u32,
    /// Interleaved t/h/w rotary pairs.
    mrope_section: [3]u32,

    // the indexer choosing which keys an attention layer reads
    index_heads: u32,
    index_kv_heads: u32,
    index_dim: u32,
    index_budget: u32,
    /// Positions pooled into one indexer key.
    index_ratio: u32,

    // DeltaNet (`linear_*`)
    key_heads: u32,
    value_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    conv_kernel: u32,

    // MoE
    experts: u32,
    /// `num_experts_per_tok`
    top_k: u32,
    moe_width: u32,
    shared_width: u32,
    norm_topk_prob: bool,

    // hyper-connections
    /// Residual streams (`hc_count`).
    streams: u32,
    /// `hc_lowrank`
    low: u32,

    // the n-gram embedding (PLE)
    /// Zero-indexed decoder layers carrying a PLE, ascending; config.json's `ple_layer_ids` count from one.
    ple_layers: []const u32,
    ple_dim: u32,
    ple_kernel: u32,
    ngram_size: u32,
    heads_per_ngram: u32,
    ngram_base: u32,
    /// `make_ngram_vocab_size_divisible_by`
    ngram_divisor: u32,
    /// `split_ngram_parts`
    ngram_shards: u32,
    seed: u64,
    /// The token that resets the n-gram history: the text config's first eos.
    ple_eos: u32,

    /// End-of-reply ids: config.json's, then those only generation_config.json names.
    eos: []const u32,
    mtp: ?Mtp,
    /// ModelOpt's block.
    quant: Quantization,

    /// Channels of a DeltaNet layer's convolution: q, k, then v.
    pub fn convDim(self: Config) u32 {
        return 2 * self.key_heads * self.key_head_dim + self.value_heads * self.value_head_dim;
    }

    /// Pooled key blocks an attention layer may select.
    pub fn topBlocks(self: Config) u32 {
        return self.index_budget / self.index_ratio;
    }

    /// The residual streams side by side.
    /// Whether `serve` offers images on this checkpoint (spec 019 WS-04, the owner's decision of 2026-10-17):
    /// Flash Next FP4-Local alone, known by what it stores, NVFP4 experts beside MXFP8 linears. The NVFP4
    /// checkpoint's tower loads too, and its replies to an image are not served: an image part is its 400.
    pub fn servesImages(self: Config) bool {
        if (!std.mem.eql(u8, self.quant.algo, "MIXED_PRECISION")) return false;
        for (self.quant.layers) |layer| if (layer.algo == .mxfp8) return true;
        return false;
    }

    pub fn wide(self: Config) u32 {
        return self.streams * self.hidden;
    }

    /// Hash heads of an n-gram table: `heads_per_ngram` for each n-gram length from 2 to `ngram_size`.
    pub fn ngramHeads(self: Config) u32 {
        return (self.ngram_size - 1) * self.heads_per_ngram;
    }

    /// Values a table row holds: a head's share of the PLE embedding.
    pub fn ngramDims(self: Config) u32 {
        return self.ple_dim / self.ngramHeads();
    }

    /// The position of `layer` among the PLE layers, which seeds its table's hashes.
    pub fn pleIndex(self: Config, layer: usize) ?usize {
        for (self.ple_layers, 0..) |at, index| if (at == layer) return index;
        return null;
    }

    /// The position of `layer` among DFlash2's completed-layer taps (spec 122): layers 5/15/25/35/45 of the
    /// pinned drafter. The list is `taps.layers`; a loaded drafter's `decode/dflash2.Config.taps` is checked
    /// against it when the drafter is seated (WS-03).
    pub fn tapIndex(_: Config, layer: usize) ?usize {
        return @import("taps.zig").tapIndex(layer);
    }

    /// Reads `dir`'s config.json and, when present, generation_config.json.
    pub fn read(arena: Allocator, io: Io, dir: Io.Dir, diagnostic: *Diagnostic) (Error || Io.Dir.ReadFileAllocError)!Config {
        const text = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 26));
        const generation = dir.readFileAlloc(io, "generation_config.json", arena, .limited(1 << 20)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        return parse(arena, text, generation, diagnostic);
    }

    /// `text` is config.json, `generation` generation_config.json if the checkpoint has one. The result
    /// lives in `arena`.
    pub fn parse(arena: Allocator, text: []const u8, generation: ?[]const u8, diagnostic: *Diagnostic) Error!Config {
        const p: Parser = .{ .arena = arena, .diagnostic = diagnostic };
        const raw = try p.document(text, "config.json");
        const t = (try p.object(raw, "text_config")) orelse raw;

        const model_type = (try p.string(raw, "model_type")) orelse "";
        for (model_types) |known| {
            if (std.mem.eql(u8, model_type, known)) break;
        } else return p.fail(error.UnsupportedModel, "config.json describes model_type \"{s}\", not Qwen3.8 Flash Next ({s} or {s})", .{ model_type, model_types[0], model_types[1] });

        const quant = try quantization(p, raw);

        const rope = (try p.object(t, "rope_parameters")) orelse json.ObjectMap.empty;
        const hidden = try p.uint(t, "hidden_size", null);
        const heads = try p.uint(t, "num_attention_heads", null);
        if (heads == 0) return p.fail(error.InvalidConfig, "config.json: num_attention_heads is 0", .{});
        var head_dim = try p.uint(t, "head_dim", 0);
        if (head_dim == 0) head_dim = hidden / heads;
        const partial = try p.number(rope, "partial_rotary_factor", try p.number(t, "partial_rotary_factor", 0.25));

        const layers = try p.uint(t, "num_hidden_layers", null);
        const layer_types = try p.layerTypes(t, "layer_types");
        if (layer_types.len != layers)
            return p.fail(error.InvalidConfig, "config.json: layer_types names {d} layers, num_hidden_layers {d}", .{ layer_types.len, layers });

        var mrope_section: [3]u32 = .{ 11, 11, 10 };
        if (try p.uints(rope, "mrope_section")) |section| {
            if (section.len != 3) return p.fail(error.InvalidConfig, "config.json: mrope_section holds {d} values, expected 3", .{section.len});
            mrope_section = section[0..3].*;
        }

        // the ids count layers from one
        const ple_ids = (try p.uints(t, "ple_layer_ids")) orelse &.{};
        const ple_layers = try arena.alloc(u32, ple_ids.len);
        for (ple_ids, ple_layers) |id, *layer| {
            if (id == 0 or id > layers) return p.fail(error.InvalidConfig, "config.json: ple_layer_ids names layer {d} of {d}", .{ id, layers });
            layer.* = id - 1;
        }
        std.mem.sort(u32, ple_layers, {}, std.sort.asc(u32));
        for (ple_layers[0..ple_layers.len -| 1], ple_layers[@min(1, ple_layers.len)..]) |a, b| {
            if (a == b) return p.fail(error.InvalidConfig, "config.json: ple_layer_ids names layer {d} twice", .{a + 1});
        }

        const text_eos = try p.ids(t, "eos_token_id");
        var eos: std.ArrayList(u32) = .empty;
        for (if (raw.contains("eos_token_id")) try p.ids(raw, "eos_token_id") else text_eos) |id| try appendNew(arena, &eos, id);
        if (generation) |source| {
            const g = try p.document(source, "generation_config.json");
            for (try p.ids(g, "eos_token_id")) |id| try appendNew(arena, &eos, id);
        }
        if (eos.items.len == 0) return p.fail(error.InvalidConfig, "no eos_token_id in config.json or generation_config.json", .{});

        const config: Config = .{
            .model_type = model_type,
            .hidden = hidden,
            .layers = layers,
            .layer_types = layer_types,
            .vocab = try p.uint(t, "vocab_size", null),
            .eps = try p.number(t, "rms_norm_eps", null),
            .max_positions = try p.uint(t, "max_position_embeddings", 0),
            .heads = heads,
            .kv_heads = try p.uint(t, "num_key_value_heads", null),
            .head_dim = head_dim,
            .rope_theta = try p.number(rope, "rope_theta", 10_000_000),
            .rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * partial),
            .mrope_section = mrope_section,
            .index_heads = try p.uint(t, "indexer_n_heads", 4),
            .index_kv_heads = try p.uint(t, "indexer_kv_heads", 1),
            .index_dim = try p.uint(t, "indexer_head_dim", 128),
            .index_budget = try p.uint(t, "indexer_budget", 2048),
            .index_ratio = try p.uint(t, "indexer_compress_ratio", 4),
            .key_heads = try p.uint(t, "linear_num_key_heads", null),
            .value_heads = try p.uint(t, "linear_num_value_heads", null),
            .key_head_dim = try p.uint(t, "linear_key_head_dim", null),
            .value_head_dim = try p.uint(t, "linear_value_head_dim", null),
            .conv_kernel = try p.uint(t, "linear_conv_kernel_dim", null),
            .experts = try p.uint(t, "num_experts", null),
            .top_k = try p.uint(t, "num_experts_per_tok", null),
            .moe_width = try p.uint(t, "moe_intermediate_size", null),
            .shared_width = try p.uint(t, "shared_expert_intermediate_size", null),
            .norm_topk_prob = try p.boolean(t, "norm_topk_prob", true),
            .streams = try p.uint(t, "hc_count", 4),
            .low = try p.uint(t, "hc_lowrank", 320),
            .ple_layers = ple_layers,
            .ple_dim = blk: {
                const dim = try p.uint(t, "ple_embed_dim", 0);
                break :blk if (dim == 0) hidden else dim;
            },
            .ple_kernel = try p.uint(t, "ple_conv_kernel_size", 4),
            .ngram_size = try p.uint(t, "ngram_size", 3),
            .heads_per_ngram = try p.uint(t, "heads_per_ngram", 8),
            .ngram_base = try p.uint(t, "ngram_vocab_size_base", 20_000_000),
            .ngram_divisor = try p.uint(t, "make_ngram_vocab_size_divisible_by", 128),
            .ngram_shards = try p.uint(t, "split_ngram_parts", 128),
            .seed = try p.uint(t, "seed", 1234),
            .ple_eos = if (text_eos.len > 0) text_eos[0] else 0,
            .eos = eos.items,
            .mtp = try mtpBlock(p, t),
            .quant = quant,
        };
        try config.check(p);
        return config;
    }

    /// Sizes the forward divides by or indexes with.
    fn check(self: Config, p: Parser) Error!void {
        inline for (.{ "hidden", "layers", "vocab", "kv_heads", "head_dim", "streams", "low", "experts", "top_k", "moe_width", "shared_width", "key_heads", "value_heads", "key_head_dim", "value_head_dim", "conv_kernel", "index_heads", "index_dim", "index_ratio" }) |field| {
            if (@field(self, field) == 0) return p.fail(error.InvalidConfig, "config.json: {s} is 0", .{field});
        }
        if (self.top_k > self.experts)
            return p.fail(error.InvalidConfig, "config.json: {d} experts a token of {d}", .{ self.top_k, self.experts });
        if (self.ple_layers.len == 0) return;
        if (self.ngram_size < 2 or self.heads_per_ngram == 0 or self.ngram_divisor == 0 or self.ngram_shards == 0 or self.ple_kernel == 0)
            return p.fail(error.InvalidConfig, "config.json: an n-gram embedding needs ngram_size >= 2 and non-zero heads, divisor, shards and kernel", .{});
        if (self.ple_dim % self.ngramHeads() != 0)
            return p.fail(error.InvalidConfig, "config.json: ple_embed_dim {d} does not split over {d} n-gram heads", .{ self.ple_dim, self.ngramHeads() });
    }
};

fn appendNew(arena: Allocator, list: *std.ArrayList(u32), id: u32) Allocator.Error!void {
    if (std.mem.indexOfScalar(u32, list.items, id) == null) try list.append(arena, id);
}

fn mtpBlock(p: Parser, t: json.ObjectMap) Error!?Mtp {
    const block = (try p.object(t, "mtp")) orelse return null;
    const layers = try p.uint(block, "num_hidden_layers", try p.uint(t, "mtp_num_hidden_layers", 1));
    if (layers == 0) return null;
    const layer_types = try p.layerTypes(block, "layer_types");
    // the head TensorFold runs, and the only one a published checkpoint has
    if (layers != 1 or layer_types.len != 1 or layer_types[0] != .full_attention)
        return p.fail(error.UnsupportedModel, "config.json: the MTP head has {d} layers; zspark reads one full_attention layer", .{layers});
    const from = block.get("mtp_use_hidden_state_from_layer") orelse json.Value.null;
    return .{
        .layers = layers,
        .layer_types = layer_types,
        .rope_theta = try p.number(block, "rope_theta", 10_000_000),
        .hybrid = try p.boolean(block, "hybrid", false),
        .hidden_state_from_layer = if (from == .null) null else try p.uint(block, "mtp_use_hidden_state_from_layer", null),
        .dedicated_embeddings = try p.boolean(t, "mtp_use_dedicated_embeddings", false),
    };
}

fn isNgramTable(name: []const u8) bool {
    const marker = ".ple.ple_embedding.ngram_embedding";
    const at = std.mem.indexOf(u8, name, marker) orelse return false;
    const end = at + marker.len;
    return end == name.len or name[end] == '.';
}

fn hasPart(name: []const u8, part: []const u8) bool {
    var parts = std.mem.splitScalar(u8, name, '.');
    while (parts.next()) |found| if (std.mem.eql(u8, found, part)) return true;
    return false;
}

fn isVision(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "model.visual") or std.mem.startsWith(u8, name, "visual");
}

const nvfp4_only = "zspark reads ModelOpt checkpoints of NVFP4 routed experts in blocks of 16 (their scales plain or csf) beside MXFP8 or bf16 linears: NVIDIA's Qwen3.8-Flash-Next-FP4-Local";

fn quantization(p: Parser, raw: json.ObjectMap) Error!Quantization {
    const block = (try p.object(raw, "quantization")) orelse (try p.object(raw, "quantization_config")) orelse
        return p.fail(error.UnsupportedQuantization, "config.json has no quantization block (an unquantized checkpoint). {s}", .{nvfp4_only});
    // MLX affine checkpoints carry bits and group_size without a method
    const method = try std.ascii.allocLowerString(p.arena, (try p.string(block, "quant_method")) orelse "mlx");
    if (!std.mem.eql(u8, method, "modelopt")) {
        const what = if (std.mem.eql(u8, method, "mlx")) "an MLX affine" else method;
        return p.fail(error.UnsupportedQuantization, "this is {s} checkpoint (quant_method \"{s}\"). {s}", .{ what, method, nvfp4_only });
    }
    const algo = try std.ascii.allocUpperString(p.arena, (try p.string(block, "quant_algo")) orelse "NVFP4");
    const mixed = std.mem.eql(u8, algo, "MIXED_PRECISION");
    if (!mixed) _ = try algoOf(p, algo, "the checkpoint");

    var layers: std.ArrayList(QuantizedLayer) = .empty;
    if (try p.object(block, "quantized_layers")) |listed| {
        try layers.ensureTotalCapacity(p.arena, listed.count());
        for (listed.keys(), listed.values()) |name, value| {
            if (value != .object) return p.fail(error.InvalidConfig, "config.json: quantized_layers.{s} is not an object", .{name});
            const named = try std.ascii.allocUpperString(p.arena, (try p.string(value.object, "quant_algo")) orelse "");
            const found = try algoOf(p, named, name);
            // older per-tensor FP8 and block-FP8 linear layouts
            if (found == .fp8 or found == .fp8_block)
                return p.fail(error.UnsupportedQuantization, "{s} is {s}, which zspark no longer reads. {s}", .{ name, named, nvfp4_only });
            if (found == .nvfp4 and !hasPart(name, "experts") and !isNgramTable(name) and !isVision(name))
                return p.fail(error.UnsupportedQuantization, "{s} is NVFP4, which is read only in routed experts and n-gram tables. {s}", .{ name, nvfp4_only });
            const group_size: ?u32 = if (value.object.contains("group_size")) try p.uint(value.object, "group_size", null) else null;
            if (found == .mxfp8 and (group_size orelse 32) != 32)
                return p.fail(error.UnsupportedQuantization, "{s} is MXFP8 in blocks of {d} inputs; MXFP8's block is 32", .{ name, group_size.? });
            if (found == .mxfp8 and (hasPart(name, "experts") or isNgramTable(name)))
                return p.fail(error.UnsupportedQuantization, "{s} is MXFP8, which is read only in the linears beside the experts. {s}", .{ name, nvfp4_only });
            const encoding = (try p.string(value.object, "weight_scale_encoding")) orelse "";
            const scale_encoding: ScaleEncoding = if (encoding.len == 0) .plain else if (std.mem.eql(u8, encoding, "csf")) .csf else return p.fail(error.UnsupportedQuantization, "{s}: weight_scale_encoding \"{s}\"; zspark reads plain and csf block scales", .{ name, encoding });
            if (scale_encoding == .csf and found != .nvfp4)
                return p.fail(error.UnsupportedQuantization, "{s}: csf-encoded scales on {t}; csf encodes NVFP4 block scales", .{ name, found });
            layers.appendAssumeCapacity(.{ .name = name, .algo = found, .group_size = group_size, .scale_encoding = scale_encoding });
        }
    }

    var groups: std.ArrayList(Group) = .empty;
    var nvfp4_group: u32 = 16;
    if (try p.object(block, "config_groups")) |listed| {
        try groups.ensureTotalCapacity(p.arena, listed.count());
        for (listed.keys(), listed.values()) |name, value| {
            if (value != .object) return p.fail(error.InvalidConfig, "config.json: config_groups.{s} is not an object", .{name});
            const weights = (try p.object(value.object, "weights")) orelse json.ObjectMap.empty;
            const inputs = try p.object(value.object, "input_activations");
            const bits = try p.uint(weights, "num_bits", 4);
            const kind = (try p.string(weights, "type")) orelse "float";
            if (!std.mem.eql(u8, kind, "float") or (bits != 4 and bits != 8))
                return p.fail(error.UnsupportedQuantization, "config_groups.{s} stores {d}-bit {s} weights. {s}", .{ name, bits, kind, nvfp4_only });
            const group_size: ?u32 = if (weights.contains("group_size")) try p.uint(weights, "group_size", null) else null;
            if (bits == 4 and (group_size orelse 16) != 16)
                return p.fail(error.UnsupportedQuantization, "config_groups.{s} stores FP4 weights in blocks of {d}. {s}", .{ name, group_size.?, nvfp4_only });
            if (std.mem.eql(u8, name, "group_0")) nvfp4_group = group_size orelse 16;
            groups.appendAssumeCapacity(.{
                .name = name,
                .weight_bits = bits,
                .weight_group_size = group_size,
                .input_bits = if (inputs) |i| try p.uint(i, "num_bits", 8) else null,
                .input_group_size = if (inputs) |i| (if (i.contains("group_size")) try p.uint(i, "group_size", null) else null) else null,
                .input_dynamic = if (inputs) |i| try p.boolean(i, "dynamic", false) else false,
                .targets = (try p.strings(value.object, "targets")) orelse &.{},
            });
        }
    }
    return .{ .algo = algo, .nvfp4_group = nvfp4_group, .layers = layers.items, .groups = groups.items };
}

fn algoOf(p: Parser, name: []const u8, where: []const u8) Error!Algo {
    return Algo.of(name) orelse p.fail(error.UnsupportedQuantization, "{s} is quantized as \"{s}\". {s}", .{ where, name, nvfp4_only });
}

/// Typed reads of a JSON object's fields; a missing field takes its default, or fails where it has none.
const Parser = struct {
    arena: Allocator,
    diagnostic: *Diagnostic,

    fn fail(p: Parser, err: Error, comptime format: []const u8, args: anytype) Error {
        p.diagnostic.set(p.arena, format, args);
        return err;
    }

    fn document(p: Parser, text: []const u8, name: []const u8) Error!json.ObjectMap {
        const root = json.parseFromSliceLeaky(json.Value, p.arena, text, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return p.fail(error.InvalidConfig, "{s} is not valid JSON ({t})", .{ name, err }),
        };
        if (root != .object) return p.fail(error.InvalidConfig, "{s} is not a JSON object", .{name});
        return root.object;
    }

    /// The field, or null where it is absent or JSON null.
    fn present(obj: json.ObjectMap, key: []const u8) ?json.Value {
        const value = obj.get(key) orelse return null;
        return if (value == .null) null else value;
    }

    fn object(p: Parser, obj: json.ObjectMap, key: []const u8) Error!?json.ObjectMap {
        const value = present(obj, key) orelse return null;
        if (value != .object) return p.fail(error.InvalidConfig, "config.json: {s} is not an object", .{key});
        return value.object;
    }

    fn string(p: Parser, obj: json.ObjectMap, key: []const u8) Error!?[]const u8 {
        const value = present(obj, key) orelse return null;
        if (value != .string) return p.fail(error.InvalidConfig, "config.json: {s} is not a string", .{key});
        return value.string;
    }

    fn uint(p: Parser, obj: json.ObjectMap, key: []const u8, default: ?u32) Error!u32 {
        const value = present(obj, key) orelse
            return default orelse p.fail(error.InvalidConfig, "config.json lacks {s}", .{key});
        return p.asUint(value, key);
    }

    fn asUint(p: Parser, value: json.Value, key: []const u8) Error!u32 {
        if (value == .integer) {
            if (std.math.cast(u32, value.integer)) |n| return n;
        }
        return p.fail(error.InvalidConfig, "config.json: {s} is not a non-negative 32-bit integer", .{key});
    }

    fn number(p: Parser, obj: json.ObjectMap, key: []const u8, default: ?f64) Error!f64 {
        const value = present(obj, key) orelse
            return default orelse p.fail(error.InvalidConfig, "config.json lacks {s}", .{key});
        return switch (value) {
            .integer => |n| @floatFromInt(n),
            .float => |f| f,
            else => p.fail(error.InvalidConfig, "config.json: {s} is not a number", .{key}),
        };
    }

    fn boolean(p: Parser, obj: json.ObjectMap, key: []const u8, default: bool) Error!bool {
        const value = present(obj, key) orelse return default;
        if (value != .bool) return p.fail(error.InvalidConfig, "config.json: {s} is not true or false", .{key});
        return value.bool;
    }

    fn uints(p: Parser, obj: json.ObjectMap, key: []const u8) Error!?[]u32 {
        const value = present(obj, key) orelse return null;
        if (value != .array) return p.fail(error.InvalidConfig, "config.json: {s} is not a list", .{key});
        const out = try p.arena.alloc(u32, value.array.items.len);
        for (value.array.items, out) |item, *slot| slot.* = try p.asUint(item, key);
        return out;
    }

    /// Token ids given as one id or a list; none where absent.
    fn ids(p: Parser, obj: json.ObjectMap, key: []const u8) Error![]const u32 {
        const value = present(obj, key) orelse return &.{};
        if (value == .array) return (try p.uints(obj, key)).?;
        const out = try p.arena.alloc(u32, 1);
        out[0] = try p.asUint(value, key);
        return out;
    }

    fn strings(p: Parser, obj: json.ObjectMap, key: []const u8) Error!?[]const []const u8 {
        const value = present(obj, key) orelse return null;
        if (value != .array) return p.fail(error.InvalidConfig, "config.json: {s} is not a list", .{key});
        const out = try p.arena.alloc([]const u8, value.array.items.len);
        for (value.array.items, out) |item, *slot| {
            if (item != .string) return p.fail(error.InvalidConfig, "config.json: {s} holds a non-string", .{key});
            slot.* = item.string;
        }
        return out;
    }

    fn layerTypes(p: Parser, obj: json.ObjectMap, key: []const u8) Error![]const LayerType {
        const names = (try p.strings(obj, key)) orelse return p.fail(error.InvalidConfig, "config.json lacks {s}", .{key});
        const out = try p.arena.alloc(LayerType, names.len);
        for (names, out) |name, *slot| {
            slot.* = std.meta.stringToEnum(LayerType, name) orelse
                return p.fail(error.UnsupportedModel, "config.json: unknown layer type \"{s}\"", .{name});
        }
        return out;
    }
};

/// An n-gram table's hash constants, derived from the config as the checkpoint's own tensors are
/// (TensorFold cuda/ngram.py): each head's prime size and row offset, and a multiplier per n-gram position.
pub const Ngram = struct {
    head_sizes: []const u64,
    head_offsets: []const u64,
    multipliers: []const i64,
    /// Table rows: the head sizes' total, rounded up to the divisor.
    rows: u64,
    /// Values a row holds.
    dims: u32,

    const golden: u64 = 0x9E3779B97F4A7C15;

    /// The table of the `ple_index`-th PLE layer.
    pub fn init(arena: Allocator, config: Config, ple_index: usize) Allocator.Error!Ngram {
        const heads = config.ngramHeads();
        const sizes = try arena.alloc(u64, heads);
        const offsets = try arena.alloc(u64, heads);
        // heads take successive primes from the base on, tables one after another
        var prime: u64 = config.ngram_base - 1;
        for (0..ple_index * heads) |_| prime = nextPrime(prime);
        var total: u64 = 0;
        for (sizes, offsets) |*size, *offset| {
            prime = nextPrime(prime);
            size.* = prime;
            offset.* = total;
            total += prime;
        }
        const multipliers = try arena.alloc(i64, config.ngram_size);
        const half = @max(1, (std.math.maxInt(i64) / @as(u64, @max(config.vocab, 1))) / 2);
        const base = config.seed +% 10007 *% @as(u64, ple_index);
        for (multipliers, 1..) |*multiplier, i| {
            multiplier.* = @intCast(2 * (splitmix64(base +% golden *% i) % half) + 1);
        }
        const divisor: u64 = config.ngram_divisor;
        return .{
            .head_sizes = sizes,
            .head_offsets = offsets,
            .multipliers = multipliers,
            .rows = (total + divisor - 1) / divisor * divisor,
            .dims = config.ngramDims(),
        };
    }

    fn splitmix64(seed: u64) u64 {
        var value = seed +% golden;
        value = (value ^ (value >> 30)) *% 0xBF58476D1CE4E5B9;
        value = (value ^ (value >> 27)) *% 0x94D049BB133111EB;
        return value ^ (value >> 31);
    }

    fn nextPrime(after: u64) u64 {
        var candidate = after + 1;
        while (!isPrime(candidate)) candidate += 1;
        return candidate;
    }

    fn isPrime(value: u64) bool {
        if (value < 2) return false;
        if (value % 2 == 0) return value == 2;
        var divisor: u64 = 3;
        while (divisor * divisor <= value) : (divisor += 2) {
            if (value % divisor == 0) return false;
        }
        return true;
    }
};
