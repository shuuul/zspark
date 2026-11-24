//! Where GLM-5.3-Flash's weights are in NVIDIA's FP4-Local ModelOpt checkpoint: every
//! tensor's name, dtype and shape from the config, and what part of it a rank of two holds (TensorFold's
//! `families/glm5_next/cuda/split.py:rule`, as a table). The model does not fit one Spark, so each of two ranks
//! reads half of every wide tensor and the forward ends each block in a partial sum over that half (spec 044).
//! A linear's storage is the map's (`weights/modelopt.zig`: BF16, MXFP8, FP8_PB_WO or
//! NVFP4 with csf_32x2_v1 or plain scales), its parts cut by the linear's split (spec 107). Host only.

const std = @import("std");
const Allocator = std.mem.Allocator;
const safetensors = @import("../../../weights/safetensors.zig");
const share_file = @import("../../../weights/share.zig");
const modelopt = @import("../../../weights/modelopt.zig");
const csf_blocks = @import("../../../weights/csf_blocks.zig");
const glm5_config = @import("config.zig");
const Config = glm5_config.Config;
const Diagnostic = glm5_config.Diagnostic;

pub const Error = error{ MissingTensor, UnexpectedTensor, UnknownTensor } || Allocator.Error;

pub const prefix = "model.language_model.";
/// Ranks a checkpoint is split over: TensorFold's engine for this family has no other (`weights.py:load`).
pub const world = share_file.ranks;

/// What a rank holds of a tensor. The enum lives in `weights/share.zig` (`take`'s second user is Flash Next).
pub const Split = share_file.Split;

/// A dimension of a tensor, by what the config makes it.
pub const Dim = enum {
    none,
    one,
    three,
    hidden,
    two_hidden,
    vocab,
    /// A site's mix dots, and the streams side by side they are taken over.
    mix,
    streams_hidden,
    kda_inner,
    kda_heads,
    kda_head_dim,
    taps,
    q_rank,
    kv_rank,
    /// DSA's heads by a key's width, by a value's, and by both (`kv_b_proj` writes key and value).
    q_inner,
    o_inner,
    kv_inner,
    index_dim,
    index_heads,
    index_inner,
    pool,
    intermediate,
    experts,
    moe,

    pub fn of(self: Dim, c: Config) u64 {
        return switch (self) {
            .none => 0,
            .one => 1,
            .three => 3,
            .hidden => c.hidden,
            .two_hidden => 2 * c.hidden,
            .vocab => c.vocab,
            .mix => c.mix(),
            .streams_hidden => c.streams * c.hidden,
            .kda_inner => c.kda_heads * c.kda_head_dim,
            .kda_heads => c.kda_heads,
            .kda_head_dim => c.kda_head_dim,
            .taps => c.conv_kernel,
            .q_rank => c.q_rank,
            .kv_rank => c.kv_rank,
            .q_inner => c.heads * c.key_dim,
            .o_inner => c.heads * c.value_dim,
            .kv_inner => c.heads * (c.key_dim + c.value_dim),
            .index_dim => c.index_dim,
            .index_heads => c.index_heads,
            .index_inner => c.index_heads * c.index_dim,
            .pool => c.index_pool,
            .intermediate => c.intermediate,
            .experts => c.experts,
            .moe => c.moe_width,
        };
    }
};

/// Which layers hold a kind of tensor.
pub const Where = enum {
    /// Once, under `prefix` or at the root.
    top,
    /// Every layer and the MTP's.
    every,
    /// The model's layers, not the MTP's (which has a plain residual).
    site,
    kda,
    /// DSA layers and the MTP's.
    dsa,
    dense,
    /// MoE layers and the MTP's.
    moe,
    /// `experts` times a MoE layer.
    expert,
    mtp,
};

pub const Kind = struct {
    /// The name after `layers.N.` (after `mlp.experts.E.` for an expert's; the whole name for `top`).
    leaf: []const u8,
    /// A linear's: as the map says (BF16 where it names none).
    dtype: safetensors.DType,
    dims: [3]Dim,
    split: Split,
    where: Where,
    /// A linear's weight (`MODULE.weight`, [outputs, inputs]): the map gives its storage and parts.
    linear: bool = false,

    pub fn shape(self: Kind, c: Config, out: *[3]u64) []const u64 {
        var n: usize = 0;
        for (self.dims) |dim| if (dim != .none) {
            out[n] = dim.of(c);
            n += 1;
        };
        return out[0..n];
    }
};

fn kind(leaf: []const u8, dtype: safetensors.DType, dims: [3]Dim, split: Split, where: Where) Kind {
    return .{ .leaf = leaf, .dtype = dtype, .dims = dims, .split = split, .where = where };
}

fn linear(leaf: []const u8, dims: [3]Dim, split: Split, where: Where) Kind {
    return .{ .leaf = leaf, .dtype = .BF16, .dims = dims, .split = split, .where = where, .linear = true };
}

/// Every tensor of the language model. `split.py`'s ROW, COL and REP lists, a line a name.
pub const kinds = [_]Kind{
    linear("lm_head.weight", .{ .vocab, .hidden, .none }, .vocab, .top),
    kind(prefix ++ "embed_tokens.weight", .BF16, .{ .vocab, .hidden, .none }, .whole, .top),
    kind(prefix ++ "norm.weight", .F32, .{ .hidden, .none, .none }, .whole, .top),
    kind("input_layernorm.weight", .BF16, .{ .hidden, .none, .none }, .whole, .every),
    // FP4-Local keeps the model's in fp32, the MTP's in bf16
    kind("post_attention_layernorm.weight", .F32, .{ .hidden, .none, .none }, .whole, .site),
    kind("post_attention_layernorm.weight", .BF16, .{ .hidden, .none, .none }, .whole, .mtp),
    // a hyper-connection site before attention and one before the MLP
    kind("hc_attn_fn", .BF16, .{ .mix, .streams_hidden, .none }, .whole, .site),
    kind("hc_attn_base", .F32, .{ .mix, .none, .none }, .whole, .site),
    kind("hc_attn_scale", .F32, .{ .three, .none, .none }, .whole, .site),
    kind("hc_ffn_fn", .BF16, .{ .mix, .streams_hidden, .none }, .whole, .site),
    kind("hc_ffn_base", .F32, .{ .mix, .none, .none }, .whole, .site),
    kind("hc_ffn_scale", .F32, .{ .three, .none, .none }, .whole, .site),
    // KDA: a rank has half the heads
    linear("self_attn.q_proj.weight", .{ .kda_inner, .hidden, .none }, .row, .kda),
    linear("self_attn.k_proj.weight", .{ .kda_inner, .hidden, .none }, .row, .kda),
    linear("self_attn.v_proj.weight", .{ .kda_inner, .hidden, .none }, .row, .kda),
    kind("self_attn.q_conv1d.weight", .BF16, .{ .kda_inner, .one, .taps }, .row, .kda),
    kind("self_attn.k_conv1d.weight", .BF16, .{ .kda_inner, .one, .taps }, .row, .kda),
    kind("self_attn.v_conv1d.weight", .BF16, .{ .kda_inner, .one, .taps }, .row, .kda),
    linear("self_attn.f_a_proj.weight", .{ .kda_head_dim, .hidden, .none }, .whole, .kda),
    linear("self_attn.f_b_proj.weight", .{ .kda_inner, .kda_head_dim, .none }, .row, .kda),
    linear("self_attn.g_a_proj.weight", .{ .kda_head_dim, .hidden, .none }, .whole, .kda),
    linear("self_attn.g_b_proj.weight", .{ .kda_inner, .kda_head_dim, .none }, .row, .kda),
    linear("self_attn.b_proj.weight", .{ .kda_heads, .hidden, .none }, .row, .kda),
    kind("self_attn.A_log", .F32, .{ .kda_heads, .none, .none }, .row, .kda),
    kind("self_attn.dt_bias", .F32, .{ .kda_inner, .none, .none }, .row, .kda),
    kind("self_attn.o_norm.weight", .BF16, .{ .kda_head_dim, .none, .none }, .whole, .kda),
    linear("self_attn.o_proj.weight", .{ .hidden, .kda_inner, .none }, .col, .kda),
    // DSA: the latent's projections whole, the heads' halved; the indexer whole
    linear("self_attn.q_a_proj.weight", .{ .q_rank, .hidden, .none }, .whole, .dsa),
    kind("self_attn.q_a_layernorm.weight", .BF16, .{ .q_rank, .none, .none }, .whole, .dsa),
    linear("self_attn.q_b_proj.weight", .{ .q_inner, .q_rank, .none }, .row, .dsa),
    linear("self_attn.kv_a_proj_with_mqa.weight", .{ .kv_rank, .hidden, .none }, .whole, .dsa),
    kind("self_attn.kv_a_layernorm.weight", .BF16, .{ .kv_rank, .none, .none }, .whole, .dsa),
    linear("self_attn.kv_b_proj.weight", .{ .kv_inner, .kv_rank, .none }, .row, .dsa),
    linear("self_attn.o_proj.weight", .{ .hidden, .o_inner, .none }, .col, .dsa),
    linear("self_attn.indexer.wk.weight", .{ .index_dim, .hidden, .none }, .whole, .dsa),
    kind("self_attn.indexer.k_norm.weight", .BF16, .{ .index_dim, .none, .none }, .whole, .dsa),
    kind("self_attn.indexer.k_norm.bias", .BF16, .{ .index_dim, .none, .none }, .whole, .dsa),
    linear("self_attn.indexer.wq_b.weight", .{ .index_inner, .q_rank, .none }, .whole, .dsa),
    linear("self_attn.indexer.weights_proj.weight", .{ .index_heads, .hidden, .none }, .whole, .dsa),
    kind("self_attn.indexer.index_kpool_compress_gate", .BF16, .{ .index_dim, .hidden, .none }, .whole, .dsa),
    kind("self_attn.indexer.index_kpool_compress_ape", .BF16, .{ .pool, .index_dim, .none }, .whole, .dsa),
    // the dense MLP of the first layers
    linear("mlp.gate_proj.weight", .{ .intermediate, .hidden, .none }, .row, .dense),
    linear("mlp.up_proj.weight", .{ .intermediate, .hidden, .none }, .row, .dense),
    linear("mlp.down_proj.weight", .{ .hidden, .intermediate, .none }, .col, .dense),
    // MoE: the router whole, the shared expert and every routed one halved over the expert's width
    linear("mlp.gate.weight", .{ .experts, .hidden, .none }, .whole, .moe),
    kind("mlp.gate.e_score_correction_bias", .F32, .{ .experts, .none, .none }, .whole, .moe),
    linear("mlp.shared_experts.gate_proj.weight", .{ .moe, .hidden, .none }, .row, .moe),
    linear("mlp.shared_experts.up_proj.weight", .{ .moe, .hidden, .none }, .row, .moe),
    linear("mlp.shared_experts.down_proj.weight", .{ .hidden, .moe, .none }, .col, .moe),
    // a routed expert: the map's format, halved over the expert's width
    linear("gate_proj.weight", .{ .moe, .hidden, .none }, .row, .expert),
    linear("up_proj.weight", .{ .moe, .hidden, .none }, .row, .expert),
    linear("down_proj.weight", .{ .hidden, .moe, .none }, .col, .expert),
    // the MTP layer's own
    linear("eh_proj.weight", .{ .hidden, .two_hidden, .none }, .whole, .mtp),
    kind("enorm.weight", .BF16, .{ .hidden, .none, .none }, .whole, .mtp),
    kind("hnorm.weight", .BF16, .{ .hidden, .none, .none }, .whole, .mtp),
    kind("shared_head.norm.weight", .BF16, .{ .hidden, .none, .none }, .whole, .mtp),
};

/// The kind of the tensor `name` (a linear's weight for its other parts), by its name alone; null where
/// `split.py:rule` has none. Kinds of one leaf in two layer kinds split alike.
pub fn kindOf(name: []const u8) ?Kind {
    return kindWhere(name, null);
}

/// The kind of the tensor `name` in `c`'s checkpoint: of its layer's kind, where one leaf names two (KDA's and
/// MLA's `o_proj`, a norm F32 or the MTP's BF16); null where `c` asks for no such tensor.
pub fn kindIn(c: Config, name: []const u8) ?Kind {
    return kindWhere(name, c);
}

fn kindWhere(name: []const u8, c: ?Config) ?Kind {
    var leaf = name;
    var layer: ?usize = null;
    if (std.mem.indexOf(u8, name, ".layers.")) |at| {
        const rest = name[at + ".layers.".len ..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
        leaf = rest[dot + 1 ..];
    }
    const expert = "mlp.experts.";
    const routed = std.mem.startsWith(u8, leaf, expert);
    if (routed) {
        const rest = leaf[expert.len..];
        leaf = rest[(std.mem.indexOfScalar(u8, rest, '.') orelse return null) + 1 ..];
    }
    for (kinds) |one| {
        if ((one.where == .expert) != routed or !std.mem.eql(u8, one.leaf, leaf)) continue;
        if (c) |config| {
            if (layer) |at| {
                if (at >= config.layers + config.mtp_layers or !Layout.holds(config, one.where, at)) continue;
            } else if (one.where != .top) continue;
        }
        return one;
    }
    return null;
}

/// What a ModelOpt linear stores beside `MODULE.weight` (`weights/modelopt.zig`).
pub const linear_parts = [_][]const u8{
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".weight_scale" ++ csf_blocks.bases_suffix,
    ".weight_scale" ++ csf_blocks.codes_suffix,
    ".weight_scale" ++ csf_blocks.page_bases_suffix,
    ".weight_scale" ++ csf_blocks.offsets_suffix,
    ".weight_scale" ++ csf_blocks.exceptions_suffix,
};

/// What a rank holds of the tensor `name`, by its name alone, as `split.py:rule` answers; null where that
/// raises (no rule for the name). A ModelOpt linear's part has its weight's split, which `modelopt.cut` applies
/// to each part by its format; the activations' `input_scale` is not read.
pub fn rule(name: []const u8) ?Split {
    if (std.mem.startsWith(u8, name, "model.visual.")) return .drop;
    for (linear_parts) |suffix| if (std.mem.endsWith(u8, name, suffix)) {
        var weight: [256]u8 = undefined;
        const one = kindOf(std.fmt.bufPrint(&weight, "{s}.weight", .{name[0 .. name.len - suffix.len]}) catch return null) orelse return null;
        if (!one.linear) return null;
        return if (std.mem.eql(u8, suffix, ".input_scale")) .drop else one.split;
    };
    return (kindOf(name) orelse return null).split;
}

/// A kind's tensors as found: how many, their bytes, and a rank's.
pub const Found = struct {
    count: usize = 0,
    bytes: u64 = 0,
    rank: u64 = 0,
};

pub const Layout = struct {
    /// One a `kinds` entry.
    found: []Found,
    /// The linears by storage, those with csf_32x2_v1 scales, and the map's entries of the
    /// language model they meet (every one of them, or `init` refuses).
    schemes: std.enums.EnumArray(modelopt.Scheme, usize) = .initFill(0),
    csf32: usize = 0,
    mapped: usize = 0,
    /// The language model's tensors, their bytes and a rank's share of them.
    tensors: usize = 0,
    bytes: u64 = 0,
    rank: u64 = 0,
    /// The vision tower's, which no rank reads.
    dropped: usize = 0,
    dropped_bytes: u64 = 0,

    /// Whether layer `layer` (the MTP's is `c.layers`) holds tensors of `where`.
    pub fn holds(c: Config, where: Where, layer: usize) bool {
        const mtp = layer >= c.layers;
        return switch (where) {
            .top => false,
            .every => true,
            .site => !mtp,
            .kda => !mtp and c.layer_types[layer] == .linear_attention,
            .dsa => mtp or c.layer_types[layer] == .deepseek_sparse_attention,
            .dense => !mtp and c.mlp_types[layer] == .dense,
            .moe, .expert => mtp or c.mlp_types[layer] == .sparse,
            .mtp => mtp,
        };
    }

    /// Finds every tensor the config asks for in `checkpoint`, with its dtype and shape, and refuses a
    /// language-model tensor it did not ask for. The result lives in `arena`.
    pub fn init(arena: Allocator, c: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic) Error!Layout {
        var self: Layout = .{ .found = try arena.alloc(Found, kinds.len) };
        @memset(self.found, .{});
        const met = try arena.alloc(bool, c.quant.layers.len);
        @memset(met, false);
        var name: [192]u8 = undefined;
        for (kinds, self.found) |one, *found| {
            if (one.where == .top) {
                try self.take(arena, c, checkpoint, diagnostic, met, one, found, one.leaf);
                continue;
            }
            for (0..c.layers + c.mtp_layers) |layer| {
                if (!holds(c.*, one.where, layer)) continue;
                if (one.where != .expert) {
                    try self.take(arena, c, checkpoint, diagnostic, met, one, found, std.fmt.bufPrint(&name, prefix ++ "layers.{d}.{s}", .{ layer, one.leaf }) catch unreachable);
                } else for (0..c.experts) |expert| {
                    try self.take(arena, c, checkpoint, diagnostic, met, one, found, std.fmt.bufPrint(&name, prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ layer, expert, one.leaf }) catch unreachable);
                }
            }
        }
        for (met, 0..) |was, at| {
            const entry = c.quant.layers[at];
            if (std.mem.startsWith(u8, entry.name, "model.visual.")) continue;
            if (!was) {
                diagnostic.set(arena, "quantized_layers names {s}, which is no linear of GLM-5.3-Flash's layout", .{entry.name});
                return error.UnknownTensor;
            }
            self.mapped += 1;
        }
        // what is left is the vision tower, or a tensor this layout has no place for
        var unknown: usize = 0;
        var it = checkpoint.tensors.iterator();
        while (it.next()) |entry| {
            if (!std.mem.startsWith(u8, entry.key_ptr.*, "model.visual.")) {
                unknown += 1;
                continue;
            }
            self.dropped += 1;
            self.dropped_bytes += entry.value_ptr.data.len;
        }
        if (unknown != self.tensors) {
            it = checkpoint.tensors.iterator();
            while (it.next()) |entry| if (rule(entry.key_ptr.*) == null) {
                diagnostic.set(arena, "the checkpoint holds {s}, which GLM-5.3-Flash's layout has no place for", .{entry.key_ptr.*});
                return error.UnknownTensor;
            };
            diagnostic.set(arena, "the checkpoint holds {d} language-model tensors where the config asks for {d}", .{ unknown, self.tensors });
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
        const rank = one.split.share(tensor.data.len);
        found.count += 1;
        found.bytes += tensor.data.len;
        found.rank += rank;
        self.tensors += 1;
        self.bytes += tensor.data.len;
        self.rank += rank;
    }
    /// A linear: `modelopt.read` finds its parts as the map stores the module.
    fn takeLinear(self: *Layout, arena: Allocator, c: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic, met: []bool, one: Kind, found: *Found, name: []const u8) Error!void {
        const module = name[0 .. name.len - ".weight".len];
        const map = c.quant;
        const entry = modelopt.findIndex(map.layers, module);
        if (entry) |at| met[at] = true;
        var dims: [3]u64 = undefined;
        const shape = one.shape(c.*, &dims);
        const stored = modelopt.read(arena, checkpoint, module, if (entry) |at| map.layers[at] else null, shape[0], shape[1], diagnostic) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.MissingTensor => error.MissingTensor,
            error.UnexpectedTensor, error.Indivisible => error.UnexpectedTensor,
        };
        const held = stored.held(one.split);
        found.count += held.count;
        found.bytes += held.bytes;
        found.rank += held.rank;
        self.tensors += held.count;
        self.bytes += held.bytes;
        self.rank += held.rank;
        self.schemes.getPtr(stored.scheme).* += 1;
        if (stored.csf != null) self.csf32 += 1;
    }
};
