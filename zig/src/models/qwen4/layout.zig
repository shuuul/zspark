//! Where a Flash Next checkpoint keeps each weight: every tensor the forward reads, resolved by name,
//! its storage classified and its dtype and shape checked against the config. Nothing is copied; the leaves
//! point into the checkpoint's mapping. Names and shapes follow TensorFold's single-GPU NVFP4 loader
//! (families/qwen4_exp/cuda/weights.py, reader.py); vision tensors are not read.

const std = @import("std");
const nvfp4 = @import("../../weights/nvfp4.zig");
const Allocator = std.mem.Allocator;
const attention = @import("blocks/attention.zig");
const qwen4_config = @import("config.zig");
const Config = qwen4_config.Config;
const Diagnostic = qwen4_config.Diagnostic;
const Ngram = qwen4_config.Ngram;
const csf = @import("../../weights/csf.zig");
const face = @import("../../weights/face.zig");
const loading = @import("../../weights/loading.zig");
const numeric = @import("../../weights/numeric.zig");
const safetensors = @import("../../weights/safetensors.zig");
const Checkpoint = safetensors.Checkpoint;
const Tensor = safetensors.Tensor;
const layout_builder = @import("layout_builder.zig");
const Builder = layout_builder.Builder;

pub const Error = error{ InvalidCheckpoint, UnsupportedScheme } || Allocator.Error;

/// How a weight is stored.
pub const Scheme = enum {
    /// bf16 values [n, k].
    bf16,
    /// Two E2M1 codes a byte [n, k/2], an e4m3 scale per 16 inputs (`weight_scale`) and one fp32 scale for
    /// the tensor (`weight_scale_2`).
    nvfp4,
    /// e4m3 [n, k] with a ue8m0 scale byte per 32 inputs (`weight_scale`, U8 [n, k/32]): 2^(byte - 127).
    mxfp8,

    pub const Refusal = error{UnknownStorage};

    /// A linear's scheme from its tensors, as TensorFold's cuda/nvfp4/format.py `scheme`. NVFP4 with csf scales
    /// has no `weight_scale` and is told apart by `Builder.linear`. The retired NVFP4 checkpoint's per-tensor
    /// and block FP8 are no longer read (tag retired-checkpoints-2026-10-21).
    pub fn of(weight: Tensor, weight_scale: ?Tensor) Refusal!Scheme {
        switch (weight.dtype) {
            .U8 => if (weight_scale) |s| {
                if (s.dtype == .F8_E4M3) return .nvfp4;
            },
            .F8_E4M3 => if (weight_scale) |s| {
                if (s.dtype == .U8) return .mxfp8;
            },
            .BF16, .F32 => return .bf16,
            else => {},
        }
        return error.UnknownStorage;
    }
};

/// Inputs sharing one MXFP8 scale.
pub const mxfp8_block = 32;

/// NVFP4 block scales stored compressed (`weights/csf.zig`): `weight_scale.nvfp4_csf_fixed` and `_exceptions`.
pub const Csf = struct {
    /// U8 [rows / 16, 16 (1 + cols / 32)]
    fixed: Tensor,
    /// U32 [count]
    exceptions: Tensor,
    /// A rank's share of the matrix the streams store whole (`share_layout.sliceLinear`): the linear's `rows` x
    /// `cols / 16` scales are this window of it. Null: the whole matrix.
    window: ?csf.Window = null,

    /// The linear's `rows` x `cols / 16` e4m3 scales into `out`.
    pub fn decode(self: Csf, rows: u32, cols: u32, out: []u8) csf.Error!void {
        const whole = csf.Shape.ofFixed(self.fixed.shape) orelse return error.InvalidCsf;
        const window = self.window orelse csf.Window.whole(whole);
        if (window.rows != rows or window.cols != cols / 16) return error.InvalidCsf;
        return csf.decodeWindow(whole, self.fixed.data, self.exceptions.data, window, out);
    }
};

/// One projection, y = W x with W [rows, cols]. ModelOpt's `input_scale` (activation quantization) is kept where
/// an NVFP4 linear stores one: the routed experts' W4A4 rows in a prompt's window read it (spec 143 WS-06).
pub const Linear = struct {
    /// The module's name; its tensors are `name.weight`, `name.weight_scale`, ...
    name: []const u8,
    scheme: Scheme,
    /// Outputs.
    rows: u32,
    /// Inputs.
    cols: u32,
    weight: Tensor,
    /// nvfp4: `weight_scale`, e4m3 [rows, cols/16], or null with `csf`. mxfp8: `weight_scale`, U8 [rows, cols/32].
    /// bf16: none.
    scale: ?Tensor = null,
    /// nvfp4: the block scales compressed, in place of `scale`.
    csf: ?Csf = null,
    /// nvfp4: `weight_scale_2`, one fp32.
    scale_2: ?Tensor = null,
    /// nvfp4: `input_scale`, one fp32, where stored.
    input_scale: ?Tensor = null,

    pub fn bytes(self: Linear) u64 {
        const compressed = if (self.csf) |c| c.fixed.data.len + c.exceptions.data.len else 0;
        return self.weight.data.len + compressed + (if (self.scale) |s| s.data.len else 0) + (if (self.scale_2) |s| s.data.len else 0) + (if (self.input_scale) |s| s.data.len else 0);
    }

    /// An NVFP4 linear's e4m3 block scales [rows, cols/16]: the stored bytes, or csf's decoded into `arena`
    /// (what the decoded checkpoint stores, to the byte).
    pub fn blockScales(self: Linear, arena: Allocator) (Allocator.Error || csf.Error)![]const u8 {
        std.debug.assert(self.scheme == .nvfp4);
        if (self.scale) |s| return s.data;
        const out = try arena.alloc(u8, self.rows * (self.cols / 16));
        try self.csf.?.decode(self.rows, self.cols, out);
        return out;
    }

    /// The linear as a face takes it, pointing into the mapping: BF16 rows, or MXFP8 with its scales as stored.
    /// NVFP4 does not sit in a face.
    pub fn part(self: Linear) error{UnsupportedScheme}!face.Part {
        switch (self.scheme) {
            .bf16 => if (self.weight.dtype == .BF16) return .{ .bf16 = self.weight.data },
            .mxfp8 => return .{ .mxfp8 = .{ .codes = self.weight.data, .exponents = self.scale.?.data, .n = self.rows } },
            .nvfp4 => {},
        }
        std.log.err("{s}: stored as {t} ({t}); a projection beside the experts is run as BF16 or MXFP8", .{ self.name, self.scheme, self.weight.dtype });
        return error.UnsupportedScheme;
    }
};

/// A 1-D parameter, BF16, F16 or F32.
pub const Vector = struct {
    name: []const u8,
    tensor: Tensor,
    /// An RMSNorm weight: where the layout's `norms_centred` holds, the stored value is gamma - 1.
    gamma: bool,

    pub fn len(self: Vector) usize {
        return @intCast(self.tensor.count());
    }

    /// Element `index` as stored.
    pub fn value(self: Vector, index: usize) f32 {
        return floatAt(self.tensor, index);
    }
};

/// A depthwise convolution's taps, BF16 [channels, 1, taps].
pub const Conv = struct {
    name: []const u8,
    tensor: Tensor,
    channels: u32,
    taps: u32,

    /// The taps as bf16 bytes, as stored.
    pub fn bf16(self: Conv) []const u8 {
        return self.tensor.data;
    }
};

/// A hyper-connection's read-out: norm, down, SiLU and inject gates, up, the mix over streams.
pub const HyperConnection = struct {
    /// [low, S*D]
    down: Linear,
    /// [S, S*D]; absent in a mixer, which writes nothing back.
    inject: ?Linear,
    /// [S*D, low]
    up: Linear,
    /// [S*D]
    norm: Vector,
};

/// A gated DeltaNet block (`linear_attn`).
pub const DeltaNet = struct {
    /// [2 nk dk + nv dv, D]: q, k, then v
    qkv: Linear,
    /// [nv dv, D]
    z: Linear,
    /// [nv, D]
    b: Linear,
    /// [nv, D]
    a: Linear,
    /// [conv_dim, taps] over the qkv channels
    conv: Conv,
    /// [nv]
    a_log: Vector,
    /// [nv]
    dt_bias: Vector,
    /// [dv]: the gated RMSNorm's weight, used as stored
    norm: Vector,
    /// [D, nv dv]
    out: Linear,
};

/// An attention block (`self_attn`) and its indexer.
pub const Attention = struct {
    /// [heads 2 head_dim, D]: each head's query, then its gate
    q: Linear,
    /// [kv_heads head_dim, D]
    k: Linear,
    /// [kv_heads head_dim, D]
    v: Linear,
    /// [(index_heads + index_kv_heads) index_dim, D]: the indexer's queries, then its key
    index_qk: Linear,
    /// [head_dim]
    q_norm: Vector,
    /// [head_dim]
    k_norm: Vector,
    /// [index_dim]
    index_q_norm: Vector,
    /// [index_dim]: the pooled indexer keys' norm
    index_k_norm: Vector,
    /// [D, heads head_dim]
    o: Linear,
};

/// An n-gram table: rows of `dims` values, split over shards in row order.
pub const Table = struct {
    /// bf16 or nvfp4.
    scheme: Scheme,
    shards: []const Shard,
    /// The table's one scale: `weight_scale` (bf16) or `weight_scale_2` (nvfp4); none multiplies by 1.
    scale: ?Tensor,
    rows: u64,
    dims: u32,

    pub const Shard = struct {
        /// [rows, dims], or [rows, dims/2] of nvfp4 codes
        weight: Tensor,
        /// nvfp4: e4m3 [rows, dims/16]
        scale: ?Tensor = null,
    };
};

/// A layer's n-gram embedding branch (PLE).
pub const Ple = struct {
    /// Its position among the PLE layers.
    index: usize,
    ngram: Ngram,
    table: Table,
    /// [S*D, ple_dim]
    key: Linear,
    /// [D, ple_dim]
    value: Linear,
    /// [S*D]
    norm_key: Vector,
    /// [S*D]
    norm_query: Vector,
    /// [S*D]
    norm_conv: Vector,
    /// [S*D, taps]
    conv: Conv,
};

pub const Expert = struct {
    /// [width, D]
    gate: Linear,
    /// [width, D]
    up: Linear,
    /// [D, width]
    down: Linear,
};

pub const Experts = union(enum) {
    /// A module per expert (`experts.{i}.gate_proj`): NVFP4 in the decoder, FP8 in the MTP layer.
    each: []const Expert,
    /// One tensor for all (an MTP layer left unquantized).
    stacked: Stacked,

    pub const Stacked = struct {
        /// BF16 [E, 2 width, D]: gate rows, then up
        gate_up: Tensor,
        /// BF16 [E, D, width]
        down: Tensor,
    };
};

pub const Moe = struct {
    /// [E, D]: `gate`
    router: Linear,
    /// [1, D]: `shared_expert_gate`
    shared_gate: Linear,
    shared: Expert,
    experts: Experts,
};

pub const Block = union(qwen4_config.LayerType) {
    linear_attention: DeltaNet,
    full_attention: Attention,
};

/// The bytes of every tensor under `value` (a layout struct, optional, union or slice of them), for
/// `Checkpoint.hold`.
pub fn ranges(arena: Allocator, value: anytype, out: *std.ArrayList([]const u8)) Allocator.Error!void {
    const T = @TypeOf(value);
    if (T == Tensor) return out.append(arena, value.data);
    switch (@typeInfo(T)) {
        .@"struct" => inline for (comptime std.meta.fieldNames(T)) |name| try ranges(arena, @field(value, name), out),
        .optional => if (value) |inner| try ranges(arena, inner, out),
        .@"union" => switch (value) {
            inline else => |inner| try ranges(arena, inner, out),
        },
        .pointer => |info| if (info.size == .slice and info.child != u8) for (value) |item| try ranges(arena, item, out),
        else => {},
    }
}

/// `value` with every tensor's bytes where `slot` staged them (`loading.Slot.moved`), its slices that hold tensors
/// copied into `arena` (and nothing else: what holds none, an n-gram's sizes, stays the caller's); a tensor `slot`
/// did not stage keeps the mapping's bytes.
pub fn moved(arena: Allocator, value: anytype, slot: *const loading.Slot) Allocator.Error!@TypeOf(value) {
    const T = @TypeOf(value);
    if (comptime !holdsTensor(T)) return value;
    if (T == Tensor) {
        var out = value;
        out.data = slot.moved(value.data);
        return out;
    }
    switch (@typeInfo(T)) {
        .@"struct" => {
            var out = value;
            inline for (comptime std.meta.fieldNames(T)) |name| @field(out, name) = try moved(arena, @field(value, name), slot);
            return out;
        },
        .optional => return if (value) |inner| try moved(arena, inner, slot) else null,
        .@"union" => switch (value) {
            inline else => |inner, tag| return @unionInit(T, @tagName(tag), try moved(arena, inner, slot)),
        },
        .pointer => |info| {
            if (info.size != .slice or info.child == u8) return value;
            const out = try arena.alloc(info.child, value.len);
            for (value, out) |item, *dest| dest.* = try moved(arena, item, slot);
            return out;
        },
        else => return value,
    }
}

/// Whether a value of `T` holds a `Tensor`, through its fields, options, unions and slices.
fn holdsTensor(comptime T: type) bool {
    return holds(T, &.{});
}

fn holds(comptime T: type, comptime seen: []const type) bool {
    if (T == Tensor) return true;
    for (seen) |one| if (one == T) return false;
    const next = seen ++ .{T};
    return switch (@typeInfo(T)) {
        .@"struct" => |info| inline for (info.field_types) |field| {
            if (holds(field, next)) break true;
        } else false,
        .optional => |info| holds(info.child, next),
        .@"union" => |info| inline for (info.field_types) |field| {
            if (holds(field, next)) break true;
        } else false,
        .pointer => |info| info.size == .slice and holds(info.child, next),
        else => false,
    };
}

pub const Layer = struct {
    attn_hc: HyperConnection,
    mlp_hc: HyperConnection,
    block: Block,
    ple: ?Ple,
    moe: Moe,
};

/// The multi-token-prediction head (`mtp.*`).
pub const Mtp = struct {
    /// [D]: `pre_fc_norm_embedding`
    norm_embedding: Vector,
    /// [S*D]: `pre_fc_norm_hidden`
    norm_hidden: Vector,
    /// [D, D]
    fc_embedding: Linear,
    /// [D, D], applied to each stream
    fc_hidden: Linear,
    layer: Layer,
    mixer: HyperConnection,
};

/// Linears and bytes of each scheme, among the tensors found.
pub const Stats = struct {
    linears: std.enums.EnumArray(Scheme, usize) = .initFill(0),
    bytes: std.enums.EnumArray(Scheme, u64) = .initFill(0),
    /// NVFP4 linears among `linears` whose block scales are csf-encoded.
    csf: usize = 0,
    /// Vectors and convolutions.
    small: usize = 0,
    small_bytes: u64 = 0,
    /// N-gram table shards, their bytes under `table_scheme`.
    table_shards: usize = 0,
    table_bytes: u64 = 0,
    table_scheme: ?Scheme = null,
};

pub const Layout = struct {
    /// Where the tensors are, for the loader to hold a part in memory while it reads it.
    checkpoint: *const Checkpoint,
    /// What precedes `layers.{i}` and `embed_tokens`: "model.language_model." in ModelOpt's exports.
    prefix: []const u8,
    /// Whether RMSNorm weights are stored as gamma - 1; null while no layer's `hc_norm` is there to tell.
    norms_centred: ?bool,
    /// [vocab, D]; each part below is null while any of its tensors is missing.
    embed: ?Linear,
    layers: []const ?Layer,
    /// The read-out after the last layer (`hyper_connection_mixer`).
    mixer: ?HyperConnection,
    /// [vocab, D]: `lm_head`
    head: ?Linear,
    /// Null too where the config has no MTP block.
    mtp: ?Mtp,
    /// Every tensor the model needs that the checkpoint lacks, in model order.
    missing: []const []const u8,
    stats: Stats,

    pub fn complete(self: Layout) bool {
        return self.missing.len == 0;
    }

    pub fn layersFound(self: Layout) usize {
        var found: usize = 0;
        for (self.layers) |layer| found += @intFromBool(layer != null);
        return found;
    }

    /// Resolves the model's tensors in `checkpoint`. A missing tensor is recorded, not an error (a
    /// download may be in progress); a tensor of the wrong dtype, shape or storage fails with `diagnostic`
    /// set. The result lives in `arena` and points into `checkpoint`, which must outlive it.
    pub fn init(arena: Allocator, config: *const Config, checkpoint: *const Checkpoint, diagnostic: *Diagnostic) Error!Layout {
        var b: Builder = .{ .arena = arena, .config = config, .checkpoint = checkpoint, .diagnostic = diagnostic };
        const c = config;
        // ModelOpt names the language model `model.language_model.*`, with lm_head and mtp at the top level
        const prefix: []const u8 = if (checkpoint.get("model.embed_tokens.weight") != null or b.anyStartsWith("model.layers."))
            "model."
        else
            "model.language_model.";

        const norms_centred = try b.normsCentred(prefix);
        const embed = try b.part(Linear, b.linear(try b.fmt("{s}embed_tokens", .{prefix}), c.vocab, c.hidden, .plain));
        const layers = try arena.alloc(?Layer, c.layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try b.layer(try b.fmt("{s}layers.{d}", .{ prefix, i }), c.layer_types[i], c.pleIndex(i), false);
        }
        const mixer = try b.part(HyperConnection, b.hyperConnection(try b.fmt("{s}hyper_connection_mixer", .{prefix}), false));
        const head = try b.part(Linear, b.linear("lm_head", c.vocab, c.hidden, .dense));
        const mtp = if (c.mtp != null) try b.mtp() else null;
        return .{
            .checkpoint = checkpoint,
            .prefix = prefix,
            .norms_centred = norms_centred,
            .embed = embed,
            .layers = layers,
            .mixer = mixer,
            .head = head,
            .mtp = mtp,
            .missing = b.missing.items,
            .stats = b.stats,
        };
    }
};

pub fn floatAt(tensor: Tensor, index: usize) f32 {
    return switch (tensor.dtype) {
        .F16 => @as(f16, @bitCast(std.mem.readInt(u16, tensor.data[index * 2 ..][0..2], .little))),
        .BF16 => numeric.f32FromBf16(std.mem.readInt(u16, tensor.data[index * 2 ..][0..2], .little)),
        .F32 => @bitCast(std.mem.readInt(u32, tensor.data[index * 4 ..][0..4], .little)),
        else => unreachable,
    };
}

/// How a load holds its attention caches: the trunk's keys and values, and the MTP head's own.
pub const Caches = struct {
    trunk: attention.Storage,
    head: attention.Storage = head_storage,
};

/// The storage of a load's caches, the one place a command asks. `parallel` is the streams the load answers
/// together (1 for a command that decodes one), `ranks` the Sparks it runs on, `asked` what `--kv-dtype` or a
/// command's word named. The lone stream's caches and the pool's are alike, so a request's text does not depend
/// on whether a second stream ever came (spec 051). What is asked wins. One stream is bf16, one context, which
/// fits. Above it on one Spark NVFP4 is FP8 from the load: a bf16 pool of the defaults (`hall.poolFor` of four
/// contexts of 262144 tokens, 1,048,576 positions) does not fit beside that model, and its FP8 one does. On
/// two ranks each holds half the KV heads and a bf16 pool fits (65.8 GiB of buffers a rank at `--parallel 4`):
/// bf16, which kept more drafts than FP8 on every text measured, 149 tok/s summed at 4 streams against 134 and
/// one stream 72.4 against 63.8 (spec 144 WS-01; the owner, 2026-10-26). The storage is not chosen by the
/// pool's size: a smaller `--context` leaves it, so that no reply's bits move with it.
pub fn storageFor(_: Config, parallel: usize, ranks: usize, asked: ?attention.Storage) Caches {
    if (asked) |held| return .{ .trunk = held };
    if (parallel <= 1 or ranks > 1) return .{ .trunk = .bf16 };
    return .{ .trunk = .fp8 };
}

/// How the MTP head holds its own cache, in a session and in the pool (spec 052): bf16, whatever
/// `storageFor` gives the trunk and whatever `--kv-dtype` asks (the owner, 2026-10-07: `--kv-dtype` is asked
/// for the trunk's memory, and the head's storage cannot change a text). The trunk verifies every draft, so
/// the head's storage changes how many are kept and never a token: on NVFP4 with FP8 caches a bf16 head keeps
/// 5.0 drafts a round of code where an FP8 one kept 4.3, for 1,008 bytes a position of the head's pool.
pub const head_storage: attention.Storage = .bf16;

/// Scales have means around 1, centred weights around 0; null without a mean to judge by.
pub fn centredFromMeans(means: []f64) error{Undecided}!?bool {
    if (means.len == 0) return null;
    var high: usize = 0;
    for (means) |mean| high += @intFromBool(mean > 0.5);
    const share = @as(f64, @floatFromInt(high)) / @as(f64, @floatFromInt(means.len));
    std.mem.sort(f64, means, {}, std.sort.asc(f64));
    const median = (means[(means.len - 1) / 2] + means[means.len / 2]) / 2;
    const around_one = share >= 0.9 and median >= 0.75 and median <= 1.5;
    const around_zero = share <= 0.1 and median >= -0.5 and median <= 0.25;
    if (!around_one and !around_zero) return error.Undecided;
    return !around_one;
}
