//! A rank's share of GLM-5.3-Flash on the device: the embedding, every layer's two hyper-connection sites, its
//! attention (KDA or DSA) and its MLP (dense or MoE), the final norm and the rank's half of the head; what a
//! sequence keeps (`State`) and what a forward writes on its way (`Buffers`). TensorFold's `weights.py`
//! `Weights` and `forward.py` `State` and `Buffers`. `forward.zig` runs it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("../../../weights/bf16.zig");
const cuda = @import("../../../cuda/cuda.zig");
const experts = @import("../../../weights/experts.zig");
const linear = @import("../../../weights/linear.zig");
const nvfp4 = @import("../../../weights/nvfp4.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const Config = @import("config.zig").Config;
const dsa = @import("dsa.zig");
const hc = @import("hc.zig");
const kda = @import("kda.zig");
const layout = @import("layout.zig");
const mlp = @import("mlp.zig");
const share = @import("share.zig");

pub const LoadError = mlp.LoadError;

/// The tensor `name` as `rank` holds it, on the device in its own shape.
pub fn upload(arena: Allocator, source: anytype, name: []const u8, rank: usize, dtype: safetensors.DType) LoadError!cuda.Buffer {
    const tensor = try share.get(arena, source, name, rank) orelse {
        std.log.err("{s}: not in the checkpoint", .{name});
        return error.MissingTensor;
    };
    if (tensor.dtype != dtype or tensor.shape.len == 0 or tensor.shape.len > 3) return error.UnsupportedShape;
    var shape: [3]usize = undefined;
    for (tensor.shape, shape[0..tensor.shape.len]) |from, *to| to.* = @intCast(from);
    var buffer: cuda.Buffer = try .init(if (dtype == .F32) .f32 else .bf16, shape[0..tensor.shape.len]);
    errdefer buffer.deinit();
    try buffer.upload(tensor.data);
    return buffer;
}

/// A norm's weight as `rank` holds it, on the device as stored: bf16, or fp32 (FP4-Local's post-attention and
/// final norms, which the kernels read so: vLLM rounds them to bf16 at load, zspark keeps them).
pub fn uploadNorm(arena: Allocator, source: anytype, name: []const u8, rank: usize) LoadError!cuda.Buffer {
    const tensor = source.get(name) orelse {
        std.log.err("{s}: not in the checkpoint", .{name});
        return error.MissingTensor;
    };
    return upload(arena, source, name, rank, if (tensor.dtype == .F32) .F32 else .BF16);
}

/// A hyper-connection site's constants and the norm its collapsed row takes.
pub const Site = struct {
    dots: cuda.Buffer,
    base: cuda.Buffer,
    scale: cuda.Buffer,
    norm: cuda.Buffer,

    /// `which` is `attn` or `ffn`, `norm` the layer norm's name.
    fn load(arena: Allocator, source: anytype, layer: usize, comptime which: []const u8, comptime norm: []const u8, rank: usize) LoadError!Site {
        const p = try std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.", .{layer});
        var dots = try upload(arena, source, try std.mem.concat(arena, u8, &.{ p, "hc_" ++ which ++ "_fn" }), rank, .BF16);
        errdefer dots.deinit();
        var base = try upload(arena, source, try std.mem.concat(arena, u8, &.{ p, "hc_" ++ which ++ "_base" }), rank, .F32);
        errdefer base.deinit();
        var scale = try upload(arena, source, try std.mem.concat(arena, u8, &.{ p, "hc_" ++ which ++ "_scale" }), rank, .F32);
        errdefer scale.deinit();
        const weight = try uploadNorm(arena, source, try std.mem.concat(arena, u8, &.{ p, norm ++ ".weight" }), rank);
        return .{ .dots = dots, .base = base, .scale = scale, .norm = weight };
    }

    fn deinit(self: *Site) void {
        self.dots.deinit();
        self.base.deinit();
        self.scale.deinit();
        self.norm.deinit();
    }

    pub fn views(self: Site) hc.Site {
        return .{ .dots = self.dots.view, .base = self.base.view, .scale = self.scale.view, .norm = self.norm.view };
    }
};

pub const Attention = union(enum) { kda: kda.Weights, dsa: dsa.Weights };
pub const Mlp = union(enum) { dense: mlp.Dense, moe: mlp.Moe };

pub const Layer = struct {
    attn_site: Site,
    ffn_site: Site,
    attention: Attention,
    mlp: Mlp,
    /// Its place among the layers of its attention's kind: which state or cache is its own.
    slot: usize,

    fn deinit(self: *Layer) void {
        self.attn_site.deinit();
        self.ffn_site.deinit();
        switch (self.attention) {
            inline else => |*one| one.deinit(),
        }
        switch (self.mlp) {
            inline else => |*one| one.deinit(),
        }
    }
};

pub const Model = struct {
    gpa: Allocator,
    rank: usize,
    dims: usize,
    /// The norms' epsilon, the hyper-connections' and their Sinkhorn's iterations.
    eps: f32,
    hc_eps: f32,
    sinkhorn: usize,
    /// The most rows a forward takes.
    max_rows: usize,
    /// The pool its linears take their scratch from, whose `quantize_from` a window's form sets (spec 115).
    pool: *cuda.Scratch,
    /// bf16 [vocab, dims], whole on each rank
    embed: cuda.Buffer,
    layers: []Layer,
    /// bf16 or fp32 [dims], as stored
    norm: cuda.Buffer,
    /// [vocab / world, dims]: the rank's rows of the vocabulary, from row `rank * vocab / world`, as stored
    /// (BF16, or MXFP8 in FP4-Local)
    head: linear.Linear,
    kda_layers: usize,
    dsa_layers: usize,
    /// How a sequence's DSA caches hold their latents (`--kv-dtype`); the MTP layer's own cache stays bf16.
    kv: dsa.Storage = .bf16,
    /// Whether a window of the quantized forms (spec 115) gives its partials in bf16, as the card's vLLM reduces
    /// them (spec 116 WS-04): FP4-Local's, whose forms they are; a test may keep fp32.
    bf16_partials: bool = false,

    /// `rank`'s share of `source` (anything with `get(name)`: a checkpoint, or a test's map of tensors).
    /// `arena` holds what the load copies on the host and may be dropped after it.
    pub fn load(arena: Allocator, gpa: Allocator, source: anytype, c: *const Config, rank: usize, rows: cuda.Scratch.Rows) LoadError!Model {
        var whole: struct {
            arena: Allocator,
            pub fn begin(self: *@This(), _: Part) LoadError!Allocator {
                return self.arena;
            }
            pub fn end(_: *@This(), _: Part) void {}
        } = .{ .arena = arena };
        return loadEach(gpa, source, c, rank, rows, &whole);
    }

    /// What `loadEach` loads in turn: the embedding, the norm and the head, then a layer.
    pub const Part = union(enum) { top, layer: usize };

    /// `load` a part at a time: `each.begin(part)` gives the arena the part's host copies and names live in,
    /// `each.end(part)` is told it is on the device, so a pack's loader holds and frees a layer's
    /// bytes around it (`pack.zig`).
    fn loadEach(gpa: Allocator, source: anytype, c: *const Config, rank: usize, rows: cuda.Scratch.Rows, each: anytype) LoadError!Model {
        return loadSpan(gpa, source, c, 0, c.layers, rank, rows, each);
    }

    /// `loadEach` of the `count` layers from `from` alone, as a model of that many: `layers[i]` is the
    /// checkpoint's layer `from + i`, a `slot` its place among those loaded. What `compare.zig` runs a layer
    /// at a time, from the streams a reference holds.
    pub fn loadSpan(gpa: Allocator, source: anytype, c: *const Config, from: usize, count: usize, rank: usize, rows: cuda.Scratch.Rows, each: anytype) LoadError!Model {
        std.debug.assert(rank < layout.world and count >= 1 and from + count <= c.layers);
        const top = try each.begin(.top);
        var embed = try upload(top, source, layout.prefix ++ "embed_tokens.weight", rank, .BF16);
        errdefer embed.deinit();
        var norm = try uploadNorm(top, source, layout.prefix ++ "norm.weight", rank);
        errdefer norm.deinit();
        var head = try loadHead(top, source, c, rank, rows);
        errdefer head.deinit();
        each.end(.top);

        const layers = try gpa.alloc(Layer, count);
        errdefer gpa.free(layers);
        var done: usize = 0;
        errdefer for (layers[0..done]) |*one| one.deinit();
        var kda_layers: usize = 0;
        var dsa_layers: usize = 0;
        // NVFP4 experts are packed here, one layer after another
        var staging: experts.Staging = .{ .place = mlp.Moe.placesOn(@TypeOf(source)) };
        defer staging.deinit(gpa);
        if (comptime mlp.Moe.placesOn(@TypeOf(source))) staging.swap = source.swap();
        for (layers, from..) |*one, index| {
            const arena = try each.begin(.{ .layer = index });
            var attn_site: Site = try .load(arena, source, index, "attn", "input_layernorm", rank);
            errdefer attn_site.deinit();
            var ffn_site: Site = try .load(arena, source, index, "ffn", "post_attention_layernorm", rank);
            errdefer ffn_site.deinit();
            var slot: usize = undefined;
            var attention: Attention = switch (c.layer_types[index]) {
                .linear_attention => kind: {
                    slot = kda_layers;
                    kda_layers += 1;
                    break :kind .{ .kda = try .load(arena, gpa, source, c, index, rank, rows) };
                },
                .deepseek_sparse_attention => kind: {
                    slot = dsa_layers;
                    dsa_layers += 1;
                    const prefix = try std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.self_attn.", .{index});
                    break :kind .{ .dsa = try .load(arena, source, prefix, c, rank, rows) };
                },
            };
            errdefer switch (attention) {
                inline else => |*a| a.deinit(),
            };
            const block: Mlp = switch (c.mlp_types[index]) {
                .dense => .{ .dense = try .load(arena, source, layout.prefix ++ "layers.{d}.mlp.", .{index}, c, rank, rows) },
                .sparse => .{ .moe = try .loadWith(arena, gpa, &staging, source, c, index, rank, rows) },
            };
            one.* = .{ .attn_site = attn_site, .ffn_site = ffn_site, .attention = attention, .mlp = block, .slot = slot };
            done += 1;
            each.end(.{ .layer = index });
        }
        try staging.settle();
        return .{ .gpa = gpa, .rank = rank, .dims = c.hidden, .eps = @floatCast(c.eps), .hc_eps = @floatCast(c.hc_eps), .sinkhorn = c.sinkhorn, .max_rows = rows.count, .pool = rows.pool, .embed = embed, .layers = layers, .norm = norm, .head = head, .kda_layers = kda_layers, .dsa_layers = dsa_layers, .bf16_partials = true };
    }

    /// The rank's rows of `lm_head`, as stored: BF16, or MXFP8 read as stored. `arena` holds what the MXFP8
    /// rows are packed in while they are.
    fn loadHead(arena: Allocator, source: anytype, c: *const Config, rank: usize, rows: cuda.Scratch.Rows) LoadError!linear.Linear {
        const n = c.vocab / layout.world;
        switch (try share.partOf(arena, source, c, "lm_head", rank)) {
            .bf16 => |bytes| {
                if (bytes.len != n * c.hidden * 2) return error.UnsupportedShape;
                return .{ .bf16 = try .init(bytes, n, c.hidden, rows) };
            },
            .mxfp8 => |stored| {
                if (stored.n != n) return error.UnsupportedShape;
                var head = nvfp4.Mx8Linear.init(arena, stored.codes, stored.exponents, n, c.hidden) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    error.UnsupportedShape => error.UnsupportedShape,
                    else => error.CudaFailure,
                };
                errdefer head.deinit();
                // its rows quantized to MXFP8, as the card's vLLM's (`share.faceOf`)
                _ = head.quantizeActivations(rows) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.CudaFailure,
                };
                return .{ .mxfp8 = head };
            },
            else => {
                std.log.err("lm_head: zspark reads it BF16 or MXFP8", .{});
                return error.UnsupportedShape;
            },
        }
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*one| one.deinit();
        self.gpa.free(self.layers);
        self.embed.deinit();
        self.norm.deinit();
        self.head.deinit();
        self.* = undefined;
    }

    /// The first layer of a kind, for the shapes its buffers take; null where the model has none.
    fn first(self: Model, comptime field: []const u8, comptime tag: []const u8) ?@FieldType(@FieldType(Layer, field), tag) {
        for (self.layers) |one| switch (@field(one, field)) {
            inline else => |weights, found| if (comptime std.mem.eql(u8, @tagName(found), tag)) return weights,
        };
        return null;
    }

    /// The indexer's key width, when a DSA layer has one.
    pub fn indexDim(self: Model) ?usize {
        const weights = self.first("attention", "dsa") orelse return null;
        const index = weights.index orelse return null;
        return index.dim;
    }
};

/// What a sequence keeps on a rank: every KDA layer's state and conv window, every DSA layer's latents, and
/// its length. The latents are the same on both ranks; the states are a rank's heads.
pub const State = struct {
    kda: ?kda.State,
    dsa: ?dsa.Cache,
    /// Positions committed.
    pos: usize = 0,
    capacity: usize,

    /// For up to `capacity` positions and windows of up to `m.max_rows` rows.
    pub fn init(m: Model, capacity: usize) cuda.Error!State {
        var chains: ?kda.State = if (m.first("attention", "kda")) |weights| try .init(m.kda_layers, weights.heads, weights.proj.n, m.max_rows) else null;
        errdefer if (chains) |*one| one.deinit();
        const latents: ?dsa.Cache = if (m.dsa_layers > 0) try .holdingAs(m.kv, m.dsa_layers, capacity, m.indexDim()) else null;
        return .{ .kda = chains, .dsa = latents, .capacity = capacity };
    }

    pub fn deinit(self: *State) void {
        if (self.kda) |*one| one.deinit();
        if (self.dsa) |*one| one.deinit();
        self.* = undefined;
    }

    /// An empty sequence.
    pub fn reset(self: *State) cuda.Error!void {
        if (self.kda) |*one| try one.reset();
        if (self.dsa) |*one| try one.setPos(0);
        self.pos = 0;
    }

    /// Keeps the first `keep` rows of the last forward of `rows` rows.
    pub fn commit(self: *State, rows: usize, keep: usize) cuda.Error!void {
        try cuda.bound(keep >= 1 and keep <= rows and self.pos + keep <= self.capacity, "glm5 commit: {d} of {d} rows at {d} of a capacity of {d}", .{ keep, rows, self.pos, self.capacity });
        if (self.kda) |*one| try one.commit(rows, keep);
        if (self.dsa) |*one| try one.commit(keep);
        self.pos += keep;
    }

    /// The side the KDA states stand on, which a captured graph is bound to.
    pub fn parity(self: State) u1 {
        return if (self.kda) |one| one.cur else 0;
    }
};

/// What one of the sequences that share a window keeps on a rank (`slots.zig`): a `State` without a window's
/// rows, which are the window's (`window`). The same bits as a `State`'s.
pub const Seq = struct {
    kda: ?kda.Kept,
    dsa: ?dsa.Cache,
    /// Positions committed.
    pos: usize = 0,
    capacity: usize,

    pub fn init(m: Model, capacity: usize) cuda.Error!Seq {
        var self = try bare(m, capacity);
        errdefer self.deinit();
        if (m.dsa_layers > 0) self.dsa = try .holdingAs(m.kv, m.dsa_layers, capacity, m.indexDim());
        return self;
    }

    /// The KDA state alone. A hall's slot takes its latents from the rank's `dsa.Store`.
    pub fn bare(m: Model, capacity: usize) cuda.Error!Seq {
        var chains: ?kda.Kept = if (m.first("attention", "kda")) |weights| try .init(m.kda_layers, weights.heads) else null;
        errdefer if (chains) |*one| one.deinit();
        return .{ .kda = chains, .dsa = null, .capacity = capacity };
    }

    pub fn deinit(self: *Seq) void {
        if (self.kda) |*one| one.deinit();
        if (self.dsa) |*one| one.deinit();
        self.* = undefined;
    }

    /// The device bytes one takes for `capacity` positions.
    pub fn bytes(m: Model, capacity: usize) usize {
        const chains = if (m.first("attention", "kda")) |weights| kda.Kept.bytes(m.kda_layers, weights.heads) else 0;
        return chains + if (m.dsa_layers > 0) dsa.Cache.bytesAs(m.kv, m.dsa_layers, capacity, m.indexDim()) else 0;
    }

    /// An empty sequence.
    pub fn reset(self: *Seq) cuda.Error!void {
        if (self.kda) |*one| try one.reset();
        if (self.dsa) |*one| try one.setPos(0);
        self.pos = 0;
    }

    /// Keeps the first `keep` of its rows `first` .. `first + rows - 1` of the window `in` ran last.
    pub fn commit(self: *Seq, in: ?kda.State, first: usize, rows: usize, keep: usize) cuda.Error!void {
        try cuda.bound(keep >= 1 and keep <= rows and self.pos + keep <= self.capacity, "glm5 commit: {d} of {d} rows at {d} of a capacity of {d}", .{ keep, rows, self.pos, self.capacity });
        if (self.kda) |*one| try one.commit(in.?, first, rows, keep);
        if (self.dsa) |*one| try one.commit(keep);
        self.pos += keep;
    }
};

/// The KDA layers' window for sequences that are each a `Seq`; null where the model has no KDA layer.
pub fn window(m: Model) cuda.Error!?kda.State {
    return if (m.first("attention", "kda")) |weights| try .initWindow(m.kda_layers, weights.heads, weights.proj.n, m.max_rows) else null;
}

/// What a forward of up to `rows` rows writes on a rank.
pub const Buffers = struct {
    rows: usize,
    /// int32 [rows]
    ids: cuda.Buffer,
    /// bf16 [rows, 4 dims]: the residual's four streams
    x: cuda.Buffer,
    /// bf16 [rows, dims]: a site's read-out, the branch's input
    normed: cuda.Buffer,
    /// fp32 [rows, 16, 32], [rows, 4] and [rows, 4, 4]: a site's dots in parts, its write-back gates and mix
    mix: cuda.Buffer,
    post: cuda.Buffer,
    comb: cuda.Buffer,
    dense: ?mlp.Dense.Buffers,
    moe: ?mlp.Moe.Buffers,
    dsa: ?dsa.Buffers,
    /// bf16 [rows, dims]: the streams' mean after the last layer, and that normed
    hidden: cuda.Buffer,
    final: cuda.Buffer,
    /// bf16 [rows, vocab / world]: the rank's rows of the vocabulary
    logits: cuda.Buffer,
    /// With a drafter: the layers whose output it reads, rising, and bf16 [rows, taps x dims], the streams'
    /// mean after each side by side in that order (`forward.py:Buffers.set_taps`, `decode.py:tap_rows`).
    /// Without one nothing is launched for them.
    tap_layers: []const u32 = &.{},
    taps: ?cuda.Buffer = null,

    /// For `m`'s shapes, windows of up to `m.max_rows` rows and a sequence of up to `capacity` positions.
    pub fn init(m: Model, capacity: usize) cuda.Error!Buffers {
        const rows = m.max_rows;
        const dims = m.dims;
        var self: Buffers = undefined;
        self.rows = rows;
        self.tap_layers = &.{};
        self.taps = null;
        self.ids = try .init(.i32, &.{rows});
        errdefer self.ids.deinit();
        self.x = try .init(.bf16, &.{ rows, hc.streams * dims });
        errdefer self.x.deinit();
        self.normed = try .init(.bf16, &.{ rows, dims });
        errdefer self.normed.deinit();
        self.mix = try .init(.f32, &hc.partShape(rows));
        errdefer self.mix.deinit();
        self.post = try .init(.f32, &.{ rows, hc.streams });
        errdefer self.post.deinit();
        self.comb = try .init(.f32, &.{ rows, hc.streams, hc.streams });
        errdefer self.comb.deinit();
        self.hidden = try .init(.bf16, &.{ rows, dims });
        errdefer self.hidden.deinit();
        self.final = try .init(.bf16, &.{ rows, dims });
        errdefer self.final.deinit();
        self.logits = try .init(.bf16, &.{ rows, m.head.outputs() });
        errdefer self.logits.deinit();
        self.dense = if (m.first("mlp", "dense")) |weights| try .init(weights.width, rows) else null;
        errdefer if (self.dense) |*one| one.deinit();
        self.moe = if (m.first("mlp", "moe")) |weights| try .init(weights, rows) else null;
        errdefer if (self.moe) |*one| one.deinit();
        self.dsa = if (m.first("attention", "dsa")) |weights| try .init(weights, capacity, rows) else null;
        return self;
    }

    /// Every forward from here on leaves the streams' mean after each of `layers` in `taps`. Before a graph
    /// is captured over these buffers; `layers` outlives them.
    pub fn setTaps(self: *Buffers, layers: []const u32) cuda.Error!void {
        std.debug.assert(self.taps == null and layers.len > 0);
        const dims: usize = @intCast(self.hidden.view.size[1]);
        self.taps = try .init(.bf16, &.{ self.rows, layers.len * dims });
        self.tap_layers = layers;
    }

    /// Tap `slot`'s columns of the first `rows` rows: [rows, dims] strided over the taps beside it.
    pub fn tap(self: Buffers, slot: usize, rows: usize) cuda.View {
        const taps = self.taps.?.view;
        const dims: usize = @intCast(self.hidden.view.size[1]);
        const base: [*]u8 = @ptrCast(taps.ptr.?);
        var view: cuda.View = .init(@ptrCast(base + slot * dims * 2), .bf16, &.{ rows, dims });
        view.row_stride = taps.size[1];
        return view;
    }

    pub fn deinit(self: *Buffers) void {
        if (self.taps) |*one| one.deinit();
        inline for (.{ "ids", "x", "normed", "mix", "post", "comb", "hidden", "final", "logits" }) |name| @field(self, name).deinit();
        if (self.dense) |*one| one.deinit();
        if (self.moe) |*one| one.deinit();
        if (self.dsa) |*one| one.deinit();
        self.* = undefined;
    }
};
