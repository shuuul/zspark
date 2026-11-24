//! Flash Next's weights on the device, and their upload from a checkpoint's layout.

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("blocks/attention.zig");
const cuda = @import("../../cuda/cuda.zig");
const Drive = @import("../../weights/drive.zig").Drive;
const experts = @import("../../weights/experts.zig");
const face = @import("../../weights/face.zig");
const forms = @import("../../decode/forms.zig");
const Face = face.Face;
const qwen4_config = @import("config.zig");
const gdn = @import("blocks/gdn.zig");
const layout = @import("layout.zig");
const loading_file = @import("../../weights/loading.zig");
const load_file = @import("model_load.zig");
const moe = @import("blocks/moe.zig");
const numeric = @import("../../weights/numeric.zig");
const nvfp4 = @import("../../weights/nvfp4.zig");
const ple = @import("blocks/ple.zig");
const requantize = @import("../../weights/requantize.zig");
const share_file = @import("share.zig");
const share_layout = @import("share_layout.zig");
const safetensors = @import("../../weights/safetensors.zig");

/// The sizes the forward reads.
pub const Shape = struct {
    hidden: usize,
    streams: usize,
    /// The hyper-connections' rank.
    low: usize,
    eps: f32,
    vocab: usize,
    /// Routed experts a row.
    top: usize,
    routed: usize,
    /// An expert's intermediate width.
    width: usize,
    delta_net: gdn.Shape = .{},

    pub fn wide(self: Shape) usize {
        return self.streams * self.hidden;
    }

    pub fn of(config: qwen4_config.Config) Shape {
        return .{
            .hidden = config.hidden,
            .streams = config.streams,
            .low = config.low,
            .eps = @floatCast(config.eps),
            .vocab = config.vocab,
            .top = config.top_k,
            .routed = config.experts,
            .width = config.moe_width,
            .delta_net = .{ .key_heads = config.key_heads, .value_heads = config.value_heads },
        };
    }
};

pub const HyperConnection = struct {
    /// [low (+ streams), streams * hidden]: the down projection, then the inject gates' rows where it has them.
    /// BF16, as stored.
    down: Face,
    /// [streams * hidden, low]
    up: Face,
    /// fp32 [streams * hidden]: the norm's gamma
    scale: cuda.Buffer,
    inject: bool,

    /// `down` and `inject` bf16 [rows, wide], `up` bf16 [wide, low], `scale` the gammas.
    pub fn init(down: []const u8, inject: ?[]const u8, up: []const u8, scale: []const f32, low: usize, scratch: cuda.Scratch.Rows) !HyperConnection {
        const wide = scale.len;
        var down_face: Face = try .initStacked(if (inject) |rows| &.{ down, rows } else &.{down}, wide, scratch);
        errdefer down_face.deinit();
        const up_face: Face = try .init(up, wide, low, scratch);
        return fromFaces(down_face, up_face, scale, inject != null);
    }

    /// Takes the faces; on failure it frees `up` and leaves `down` to the caller.
    pub fn fromFaces(down: Face, up: Face, scale: []const f32, inject: bool) cuda.Error!HyperConnection {
        var up_face = up;
        errdefer up_face.deinit();
        var scale_dev: cuda.Buffer = try .init(.f32, &.{scale.len});
        errdefer scale_dev.deinit();
        try scale_dev.upload(std.mem.sliceAsBytes(scale));
        return .{ .down = down, .up = up_face, .scale = scale_dev, .inject = inject };
    }

    pub fn deinit(self: *HyperConnection) void {
        self.down.deinit();
        self.up.deinit();
        self.scale.deinit();
        self.* = undefined;
    }
};

pub const DeltaNet = struct {
    /// [qkv | z | b | a rows, hidden]; MXFP8 in FP4-Local, one stacked run
    projection: Face,
    /// bf16 [conv channels, 4]
    conv: cuda.Buffer,
    /// fp32 [value heads]
    a_log: cuda.Buffer,
    /// fp32 [value heads]
    dt_bias: cuda.Buffer,
    /// bf16 [128]
    norm: cuda.Buffer,
    /// [hidden, value heads * 128]
    out: Face,

    pub const Host = struct {
        /// The projection's parts in row order (qkv, z, b, a), each bf16 [rows, hidden].
        projection: []const []const u8,
        conv: []const u8,
        a_log: []const f32,
        dt_bias: []const f32,
        norm: []const u8,
        out: []const u8,
    };

    /// What a block keeps beside its two faces.
    pub const Vectors = struct {
        /// bf16 [conv channels, 4]
        conv: []const u8,
        a_log: []const f32,
        dt_bias: []const f32,
        /// bf16 [128]
        norm: []const u8,
    };

    /// A block whose linears are all BF16.
    pub fn init(host: Host, shape: Shape, scratch: cuda.Scratch.Rows) !DeltaNet {
        var projection: Face = try .initStacked(host.projection, shape.hidden, scratch);
        errdefer projection.deinit();
        var out: Face = try .init(host.out, shape.hidden, shape.delta_net.out(), scratch);
        errdefer out.deinit();
        return fromFaces(projection, out, .{ .conv = host.conv, .a_log = host.a_log, .dt_bias = host.dt_bias, .norm = host.norm }, shape);
    }

    /// A block over faces already on the device, which it owns once it returns; on failure they stay the caller's.
    pub fn fromFaces(projection: Face, out: Face, host: Vectors, shape: Shape) !DeltaNet {
        const dn = shape.delta_net;
        if (projection.n != dn.projection() or projection.k != shape.hidden or out.n != shape.hidden or out.k != dn.out()) return error.UnsupportedShape;
        var conv: cuda.Buffer = try .init(.bf16, &.{ dn.conv(), gdn.taps });
        errdefer conv.deinit();
        try conv.upload(host.conv);
        var a_log: cuda.Buffer = try .init(.f32, &.{dn.value_heads});
        errdefer a_log.deinit();
        try a_log.upload(std.mem.sliceAsBytes(host.a_log));
        var dt_bias: cuda.Buffer = try .init(.f32, &.{dn.value_heads});
        errdefer dt_bias.deinit();
        try dt_bias.upload(std.mem.sliceAsBytes(host.dt_bias));
        var norm: cuda.Buffer = try .init(.bf16, &.{gdn.value_dim});
        errdefer norm.deinit();
        try norm.upload(host.norm);
        return .{ .projection = projection, .conv = conv, .a_log = a_log, .dt_bias = dt_bias, .norm = norm, .out = out };
    }

    pub fn deinit(self: *DeltaNet) void {
        self.projection.deinit();
        self.conv.deinit();
        self.a_log.deinit();
        self.dt_bias.deinit();
        self.norm.deinit();
        self.out.deinit();
        self.* = undefined;
    }

    pub fn kernelWeights(self: DeltaNet) gdn.Weights {
        return .{ .conv = self.conv.view, .a_log = self.a_log.view, .dt_bias = self.dt_bias.view, .norm = self.norm.view };
    }
};

pub const Block = union(enum) {
    delta_net: DeltaNet,
    attention: attention.Weights,
};

/// A layer's n-gram embedding branch: its weights on the device and its table in the checkpoint's mapping.
pub const Ple = struct {
    weights: ple.Weights,
    table: ple.Table,
    ngram: qwen4_config.Ngram,
    /// The token that resets the n-grams.
    eos: u32,
};

pub const Layer = struct {
    attn_hc: HyperConnection,
    mlp_hc: HyperConnection,
    block: Block,
    moe: moe.Weights,
    ple: ?Ple = null,

    pub fn deinit(self: *Layer, gpa: Allocator) void {
        if (self.ple) |*branch| {
            branch.weights.deinit();
            branch.table.deinit(gpa);
        }
        self.attn_hc.deinit();
        self.mlp_hc.deinit();
        switch (self.block) {
            .delta_net => |*block| block.deinit(),
            .attention => |*block| block.deinit(),
        }
        self.moe.deinit();
        self.* = undefined;
    }

    /// Its MXFP8 faces take FP8 rows in the windows the pool's form quantizes (`PromptForm.card`), as the card's
    /// vLLM runs them in a prompt's chunks: b12x races A16 against quantized rows by time for each of its linears
    /// at the 4,096-row capacity every chunk of more than its graphs' rows runs (spec 143 WS-08, its modes logged
    /// from the references' own selections). Quantized there: DeltaNet's qkv and z (`in_proj_qkvz`) and its
    /// output, attention's q, k, v and the indexer's projection and its output, the shared expert's gate and up
    /// and its down. A16: DeltaNet's b and a (`in_proj_ba`, 96 outputs), so their columns keep bf16 rows
    /// (`Face.keepBf16From`). The hyper-connections, the router and the PLE are BF16 and keep bf16 rows. Returns
    /// how many MXFP8 runs cannot (their inputs not whole stages of 128).
    pub fn quantizePrompts(self: *Layer, shape: Shape) !usize {
        var kept: usize = 0;
        switch (self.block) {
            .delta_net => |*block| {
                const quantized = try block.projection.quantizeActivations();
                kept += quantized + try block.out.quantizeActivations();
                // b and a end the row (`gdn.Shape.projection`), a rank's share's of its heads at its share's end
                const ba = 2 * shape.delta_net.value_heads * block.projection.n / shape.delta_net.projection();
                if (quantized == 0) try block.projection.keepBf16From(block.projection.n - ba);
            },
            .attention => |*block| kept += try block.proj.quantizeActivations() + try block.o.quantizeActivations(),
        }
        switch (self.moe.stored) {
            .nvfp4 => |*stored| kept += try stored.shared_gate_up.quantizeActivations() + try stored.shared_down.quantizeActivations(),
        }
        return kept;
    }
};

/// The multi-token-prediction head (`mtp.*`): what drafts read. TensorFold's `MTPW` (weight_types.py).
pub const Mtp = struct {
    /// fp32 [hidden]: `pre_fc_norm_embedding`'s gamma, for the drafted token's embedding
    norm_embedding: cuda.Buffer,
    /// fp32 [streams * hidden]: `pre_fc_norm_hidden`'s gamma, for a row's final streams as one norm
    norm_hidden: cuda.Buffer,
    /// [hidden, hidden], for up to `max_rows` rows
    fc_embedding: Face,
    /// [hidden, hidden], applied to each stream: up to `max_rows * streams` rows
    fc_hidden: Face,
    /// `mtp.layers.0`: an attention layer with both hyper-connections and no PLE. Its routed experts are
    /// NVFP4 whatever the checkpoint stores (`model_load.loadMoe`); its faces take BF16 or MXFP8 as a main
    /// layer's do.
    layer: Layer,
    /// The head's own read-out, before the main model's `head`.
    mixer: HyperConnection,
    /// What scores a draft in place of the main model's head; null scores with that head.
    draft: ?DraftHead = null,
    /// What a drafted row and a level of the head cost in this storage (`mtp_rule.Paying`); null, a model made
    /// by a test, drafts by `mtp_rule.Rule.chain`.
    costs: ?Paying.Costs = null,
    /// Calls take up to this many rows.
    max_rows: usize,
    /// Beside a share (rank 0 of two, spec 090): the whole model's sizes and an attention of the head's own,
    /// the share's `Model.shape` and `Model.attention` being a rank's half. Null on one rank, where the head's
    /// are the model's (`mtp.over`).
    whole: ?Whole = null,
    /// Split over the two ranks (`Options.head_split`, spec 090's scheme B): this rank's half of the layer, as
    /// its share of a main layer, and `draft` its part of the draft vocabulary. Null where the head is whole.
    split: ?Split = null,

    pub const Whole = struct { shape: Shape, attention: Model.Attention };

    pub const Split = struct {
        /// Every rank's draft ids, ascending: the columns of the gathered row (`mtp_ranks.zig`).
        ids: []u32,

        /// The most ids a rank scores: rank 0's.
        pub fn widest(self: Split) usize {
            return partOf(self.ids, 0).len;
        }

        /// Rank `rank`'s ids of `ids`, as TensorFold's `np.array_split(ids, world)[rank]`: the ranks' parts in
        /// order, the first ones an id longer where the count does not divide.
        pub fn partOf(ids: []const u32, rank: usize) []const u32 {
            const world: usize = share_file.world;
            const each = ids.len / world;
            const longer = ids.len % world;
            return ids[rank * each + @min(rank, longer) ..][0 .. each + @intFromBool(rank < longer)];
        }
    };

    /// The tokens a scored row's columns stand for: every rank's where the head is split, the draft head's,
    /// or null where the main head scores.
    pub fn scored(self: Mtp) ?[]const u32 {
        if (self.split) |split| return split.ids;
        return if (self.draft) |d| d.ids else null;
    }

    const Paying = @import("mtp.zig").Paying;

    /// The head alone, from `found`; the sizes are `config`'s, the main layers' on one rank and the whole
    /// model's beside a share. Its attention needs the rotary frequencies of `Model.attention` too, or of
    /// `whole`, which `Model.loadWith` makes.
    /// `stored`: its routed experts in the format the checkpoint holds (`Model.Options.drafts_stored`).
    /// `cut`: the rank's half of it (`split`; the caller sets that field), `config` still the whole model's.
    /// `staged`: the head's tensors where a reading staged them (`model_load.Reading`'s part after the head),
    /// whose NVFP4 routed experts the device then places as a main layer's (`model_load.placesMtp`); null: they
    /// are held through the mapping and packed by the host.
    pub fn load(gpa: Allocator, staging: *experts.Staging, config: qwen4_config.Config, found: layout.Layout, scratch: cuda.Scratch.Rows, stored: bool, cut: ?share_file.Share, staged: ?layout.Mtp) Model.LoadError!Mtp {
        const whole_source = staged orelse found.mtp orelse return error.MissingTensors;
        const centred = found.norms_centred orelse return error.MissingTensors;
        var cfg = if (cut != null) share_layout.halved(config) else config;
        const shape: Shape = .of(cfg);
        const attention_shape = attention.Shape.of(&cfg) catch return error.UnsupportedShape;
        var held = if (staged != null) safetensors.Held.none(gpa) else try load_file.hold(gpa, found, whole_source);
        defer held.release();
        var piece: std.heap.ArenaAllocator = .init(gpa);
        defer piece.deinit();
        const placed = staged != null and load_file.placesMtp(whole_source);
        const slice = if (placed) &share_layout.sliceMtpPlaced else &share_layout.sliceMtp;
        const source = if (cut) |one| slice(piece.allocator(), whole_source, one, share_file.Heads.of(config)) catch |err| return load_file.cutShare(err) else whole_source;
        const placing = staging.place;
        staging.place = placed;
        defer staging.place = placing;

        var norm_embedding = try load_file.gamma(gpa, source.norm_embedding, centred);
        errdefer norm_embedding.deinit();
        var norm_hidden = try load_file.gamma(gpa, source.norm_hidden, centred);
        errdefer norm_hidden.deinit();
        var fc_embedding = try load_file.faceOf(gpa, &.{source.fc_embedding}, scratch);
        errdefer fc_embedding.deinit();
        var fc_hidden = try load_file.faceOf(gpa, &.{source.fc_hidden}, scratch.times(shape.streams));
        errdefer fc_hidden.deinit();
        var layer = try load_file.loadLayer(gpa, staging, source.layer, shape, attention_shape, centred, scratch, stored, cut, if (placed) cut else null, .bf16);
        errdefer layer.deinit(gpa);
        // the device is done with the staged bytes before they go
        if (placed) try staging.settle();
        const mixer = try load_file.loadHyperConnection(gpa, source.mixer, centred, scratch);
        return .{ .norm_embedding = norm_embedding, .norm_hidden = norm_hidden, .fc_embedding = fc_embedding, .fc_hidden = fc_hidden, .layer = layer, .mixer = mixer, .costs = Paying.published, .max_rows = scratch.count };
    }

    pub fn deinit(self: *Mtp, gpa: Allocator) void {
        self.norm_embedding.deinit();
        self.norm_hidden.deinit();
        self.fc_embedding.deinit();
        self.fc_hidden.deinit();
        self.layer.deinit(gpa);
        self.mixer.deinit();
        if (self.draft) |*head| head.deinit(gpa);
        if (self.whole) |*own| own.attention.rotary.deinit();
        if (self.split) |own| gpa.free(own.ids);
        self.* = undefined;
    }
};

/// The drafts' head: the main head's rows of a draft vocabulary, requantized to NVFP4 on the host, so a draft
/// reads a twelfth of the bytes the head does. TensorFold's `draft_head`, which holds the same rows as MLX's affine
/// 4-bit (weights.py `quantize4`); zspark has that format's kernel nowhere else, and drafts choose nothing.
pub const DraftHead = struct {
    /// [ids.len, hidden]
    linear: nvfp4.Fp4Linear,
    /// The token each output scores, ascending.
    ids: []u32,

    /// TensorFold's families/qwen4_exp/cuda/draft_vocab.txt: the tokenizer's ids worth drafting, one a line.
    const default_vocab = @embedFile("draft_vocab.txt");

    /// The default draft vocabulary, ascending and without repeats; the caller owns it.
    pub fn defaultIds(gpa: Allocator) (Allocator.Error || error{InvalidDraftVocab})![]u32 {
        return parseIds(gpa, default_vocab);
    }

    /// The ids of `text`, one a line, sorted and without repeats, as TensorFold's `draft_token_ids` reads them.
    pub fn parseIds(gpa: Allocator, text: []const u8) (Allocator.Error || error{InvalidDraftVocab})![]u32 {
        var list: std.ArrayList(u32) = .empty;
        errdefer list.deinit(gpa);
        var lines = std.mem.tokenizeAny(u8, text, " \t\r\n");
        while (lines.next()) |line| try list.append(gpa, std.fmt.parseInt(u32, line, 10) catch return error.InvalidDraftVocab);
        std.mem.sort(u32, list.items, {}, std.sort.asc(u32));
        var kept: usize = 0;
        for (list.items) |id| {
            if (kept != 0 and list.items[kept - 1] == id) continue;
            list.items[kept] = id;
            kept += 1;
        }
        list.shrinkRetainingCapacity(kept);
        return list.toOwnedSlice(gpa);
    }

    /// Rows `ids` (ascending) of `weight` [vocab, hidden], the main head as stored.
    pub fn init(gpa: Allocator, weight: requantize.Weight, vocab: usize, hidden: usize, ids: []const u32) Model.LoadError!DraftHead {
        var rows = requantize.Rows.init(gpa, weight, vocab, hidden, ids) catch |err| return load_file.narrow(err);
        defer rows.deinit(gpa);
        const owned = try gpa.dupe(u32, ids);
        errdefer gpa.free(owned);
        const linear = nvfp4.Fp4Linear.init(gpa, rows.codes, rows.block_scales, ids.len, hidden, rows.scale) catch |err| return load_file.narrow(err);
        return .{ .linear = linear, .ids = owned };
    }

    pub fn deinit(self: *DraftHead, gpa: Allocator) void {
        self.linear.deinit();
        gpa.free(self.ids);
        self.* = undefined;
    }
};

/// `Model.Options.drafts_stored` as the commands pass it; `main.zig` sets it from `ZSPARK_DRAFTS_STORED`.
pub var drafts_stored: bool = false;

/// Flash Next's activations in a prompt's windows (spec 143 WS-06, `forward.bf16_rows`): `card`, the card's vLLM's
/// (MXFP8 linears on FP8 rows, `qmm_a8.cu`; the routed experts W4A4 on their `mxf4nvf4` pack, whose steps and
/// verify windows run A16 on it); `bf16`, every row bf16 in every window on the A16 pack, as before. Steps and
/// verify windows are bf16 rows in both. Read at load: `main.zig` sets it from `ZSPARK_PROMPT_FORM`, a test
/// before its load.
pub const PromptForm = enum { bf16, card };
pub var prompt_form: PromptForm = .bf16;

/// How two ranks hold the MTP head (spec 090): `whole` on rank 0, beside its share (scheme A), or `split`, half
/// on each as a main layer is (scheme B). `Model.Options.head_split` as the commands pass it; `main.zig` sets
/// it from `ZSPARK_RANKS_HEAD`. Split since spec 090's timed pair: 1 to 8% of a stream's tokens a second.
pub const RanksHead = enum { whole, split };
pub var ranks_head: RanksHead = .split;

pub const Model = struct {
    shape: Shape,
    /// bf16 [vocab, hidden]
    embed: cuda.Buffer,
    layers: []Layer,
    /// The read-out after the last layer.
    mixer: HyperConnection,
    /// [vocab, hidden], BF16 as stored
    head: Face,
    /// The attention layers' sizes and rotary frequencies; null in a model without one. The MTP head's layer
    /// is one.
    attention: ?Attention = null,
    /// The drafting head; null where it was not asked for or the checkpoint has none.
    mtp: ?Mtp = null,
    /// Calls take up to this many rows.
    max_rows: usize,
    /// The linears' scratch, theirs to compute in and nothing's to keep (`cuda.Scratch`); freed after them.
    scratch: *cuda.Scratch,
    /// The drive the n-gram table is read from, kept awake while its rows are gathered (`weights/drive.zig`):
    /// one reader a model, which every table's gather tells. Null where the table's file has no direct reads.
    drive: ?*Drive = null,
    gpa: Allocator,

    pub const Attention = struct {
        shape: attention.Shape,
        rotary: attention.Rotary,
        /// How the MTP head holds its own cache, a session's (`mtp.State`) and the pool's (`hall.Engine`'s
        /// `head_pool`): the trunk's storage, `shape.kv`, or bf16 beside a coded trunk (spec 052). The head's drafts are verified by the trunk, so this
        /// changes how many are kept and never a token.
        head: attention.Storage,

        /// `shape` as the head's cache is stored.
        pub fn headShape(self: Attention) attention.Shape {
            var shape = self.shape;
            shape.kv = self.head;
            return shape;
        }
    };

    pub fn deinit(self: *Model) void {
        // whatever gathers (a session's buffers, the pool's) is gone before its model
        if (self.drive) |drive| drive.close(self.gpa);
        if (self.attention) |*shared| shared.rotary.deinit();
        if (self.mtp) |*drafter| drafter.deinit(self.gpa);
        self.embed.deinit();
        for (self.layers) |*layer| layer.deinit(self.gpa);
        self.gpa.free(self.layers);
        self.mixer.deinit();
        self.head.deinit();
        self.scratch.destroy();
        self.* = undefined;
    }

    pub const LoadError = error{ MissingTensors, UnsupportedScheme, UnsupportedShape, UnsupportedLayer } || Allocator.Error || cuda.Error;

    /// Uploads the embedding, the first `layer_count` layers, the mixer and the head of `found`: the DeltaNet and
    /// attention faces and the shared expert take BF16 or MXFP8 on its stored bytes; the embedding, the router,
    /// the hyper-connections, the head and the PLE's projections are BF16. The retired checkpoints' block FP8
    /// is at tag retired-checkpoints-2026-10-21.
    /// Without the MTP head, which `loadWith` takes.
    pub fn load(gpa: Allocator, config: qwen4_config.Config, found: layout.Layout, layer_count: usize, max_rows: usize) LoadError!Model {
        return loadWith(gpa, config, found, .{ .layers = layer_count, .max_rows = max_rows, .mtp = false });
    }

    pub const Options = struct {
        /// The first layers to load; a partial load still takes the mixer, the head and the MTP head.
        layers: usize,
        max_rows: usize,
        /// Whether to load the MTP head where the layout has one (`Layout.mtp`); `Model.mtp` stays null where
        /// it has none. Its experts are requantized to NVFP4 (`zig/src/weights/requantize.zig`), as TensorFold
        /// does, unless `drafts_stored`.
        mtp: bool = true,
        /// The tokens the MTP head's drafts are scored over, ascending (`DraftHead`); null scores them with
        /// the main head.
        draft_vocab: ?[]const u32 = null,
        /// Drafting on what the checkpoint stores, in place of TensorFold's two requantizations at load: no
        /// `DraftHead` whatever `draft_vocab` holds (the main head scores the drafts), and the MTP layer's
        /// routed experts run as stored where they are NVFP4. Experts stored otherwise have no such path and
        /// are refused. A command passes `drafts_stored` (`ZSPARK_DRAFTS_STORED`).
        drafts_stored: bool = false,
        /// How the attention caches, the MTP head's included, hold keys and values.
        kv: attention.Storage = .bf16,
        /// The MTP head's own cache where it is not `kv`: bf16 alone (`Attention.head`, `layout.head_storage`).
        head_kv: ?attention.Storage = null,
        /// A rank of two. Null is one rank, every tensor whole, today's kernels. With `mtp`, rank 0 loads the
        /// MTP head whole beside its share (`Mtp.whole`, spec 090) and rank 1 nothing of it: a draft is
        /// verified by both ranks' window and proposed by one, so the ranks' options may be the same.
        share: ?@import("share.zig").Share = null,
        /// With `share` and `mtp`: each rank loads its half of the head (`Mtp.split`, scheme B) and both
        /// compute a drafted level (`mtp_ranks.zig`). A command passes `ranks_head == .split`.
        head_split: bool = false,
        /// The parts' reading begun before the load (`Reading.prefetch`, serve's while the ranks join), for this
        /// layout, `layers` and `share`; null: the load reads them itself. The caller frees it after the load.
        reading: ?*load_file.Reading = null,
    };

    /// What a load's reading reads of the MTP head (`load_file.Reading.mtp`): `drafting` and `split` as
    /// `loadUnreported` finds them, which a caller that makes the reading finds with `drafts`.
    pub fn mtpReading(drafting: bool, split: bool) @FieldType(load_file.Reading, "mtp") {
        return if (!drafting) .none else if (split) .split else .whole;
    }

    /// Whether a load with these options and this layout reads the MTP head, on `share`'s rank (`loadUnreported`).
    pub fn drafts(found: layout.Layout, options: Options) bool {
        const rank = options.share;
        const split = rank != null and options.head_split;
        return options.mtp and found.mtp != null and (rank == null or rank.?.rank == 0 or split);
    }

    /// `load`, and the MTP head after the main model's.
    pub fn loadWith(gpa: Allocator, config: qwen4_config.Config, found: layout.Layout, options: Options) LoadError!Model {
        const started = cuda.monotonic();
        const model = try loadUnreported(gpa, config, found, options);
        loading_file.report(cuda.monotonic() - started);
        return model;
    }

    fn loadUnreported(gpa: Allocator, config: qwen4_config.Config, found: layout.Layout, options: Options) LoadError!Model {
        const rank = options.share;
        // Whole, rank 0 holds the head and rank 1 none of it; split, each its half.
        const split = rank != null and options.head_split;
        const drafting = drafts(found, options);
        if (rank != null and drafting) {
            // a share's `head` is half a vocabulary: nothing to score a draft with
            if (options.drafts_stored) {
                std.log.err("drafts as stored (ZSPARK_DRAFTS_STORED) score with the model's head, and a rank of two holds half of it: unset it with --rank", .{});
                return error.UnsupportedScheme;
            }
            if (options.draft_vocab == null) {
                std.log.err("the MTP head beside a share scores its drafts with a draft head, and no draft vocabulary was given", .{});
                return error.UnsupportedShape;
            }
        }
        // A column cut and a DeltaNet's three segments are copies. They are made after the layer's tensors
        // are held, and freed when the layer is on the device: cutting the whole model first reads them
        // through the mapping and keeps about 12 GiB beside the device copy until the load ends.
        var cfg = if (rank != null) share_layout.halved(config) else config;
        const heads = share_file.Heads.of(config);
        var piece: std.heap.ArenaAllocator = .init(gpa);
        defer piece.deinit();
        const layer_count = options.layers;
        const max_rows = options.max_rows;
        const pool: *cuda.Scratch = try .create(gpa);
        errdefer pool.destroy();
        const scratch = pool.rows(max_rows);
        const shape: Shape = .of(cfg);
        const centred = found.norms_centred orelse return error.MissingTensors;
        const embed_whole = found.embed orelse return error.MissingTensors;
        // a part at a time, read a part ahead (`load_file.Reading`)
        var own_reading: load_file.Reading = .init(gpa, found, layer_count, rank);
        defer own_reading.deinit();
        own_reading.mtp = mtpReading(drafting, split);
        const reading = options.reading orelse &own_reading;
        std.debug.assert(options.reading == null or reading.layers == layer_count);
        // the MTP head is a staged part where the reading reads it (`Reading.mtp`, set before its first part)
        if (reading.mtp != mtpReading(drafting, split)) return error.UnsupportedShape;
        const embed_arena = try reading.begin(load_file.Reading.embed);
        const embed_read = try reading.moved(embed_arena, embed_whole);
        const embed_linear = if (rank) |one| share_layout.sliceLinear(piece.allocator(), embed_read, one, heads) catch |err| return load_file.cutShare(err) else embed_read;
        var embed: cuda.Buffer = try .init(.bf16, &.{ shape.vocab, shape.hidden });
        errdefer embed.deinit();
        try embed.upload(try load_file.plain(embed_linear));
        reading.end();

        var attention_shape = attention.Shape.of(&cfg) catch return error.UnsupportedShape;
        attention_shape.kv = options.kv;
        const head_kv = options.head_kv orelse options.kv;
        if (head_kv != options.kv and head_kv != .bf16) return error.UnsupportedShape;
        var uses_attention = false;
        // the staged parts' experts are placed by the device from the stage (`Reading.end` keeps a part's bytes
        // until its launches are done)
        // a reading that halves down trades the rest with the peer (`Reading.swapping`)
        if (reading.swapping != (reading.swap != null) or (reading.swapping and rank == null)) return error.UnsupportedShape;
        var staging: experts.Staging = .{ .place = true, .swap = reading.swap };
        defer staging.deinit(gpa);
        const layers = try gpa.alloc(Layer, layer_count);
        errdefer gpa.free(layers);
        var loaded: usize = 0;
        errdefer for (layers[0..loaded]) |*layer| layer.deinit(gpa);
        var drive: ?*Drive = null;
        errdefer if (drive) |reader| reader.close(gpa);
        for (layers, 0..) |*layer, i| {
            const whole = found.layers[i] orelse return error.MissingTensors;
            // The n-gram table stays in the mapping, its rows read as they are gathered, on every checkpoint:
            // its 30 to 51 GiB are not what a layer's load holds. Held, they were read from the disk and
            // uploaded nowhere, and releasing them dropped from the page cache the rows a reply reads next.
            var loading = whole;
            if (loading.ple) |*branch| branch.table.shards = &.{};
            var layer_mem: std.heap.ArenaAllocator = .init(gpa);
            defer layer_mem.deinit();
            const arena = try reading.begin(load_file.Reading.layer(i));
            var read = try reading.moved(arena, loading);
            // the table's rows stay in the mapping, read as they are gathered
            if (read.ple) |*branch| branch.table.shards = whole.ple.?.table.shards;
            const source = if (rank) |one| share_layout.sliceLayerPlaced(layer_mem.allocator(), read, one, heads) catch |err| return load_file.cutShare(err) else read;
            layer.* = try load_file.loadLayer(gpa, &staging, source, shape, attention_shape, centred, scratch, false, null, rank, if (prompt_form == .card) .nvfp4 else .bf16);
            errdefer layer.deinit(gpa);
            reading.end();
            uses_attention = uses_attention or layer.block == .attention;
            if (source.ple) |branch| {
                var table = ple.Table.init(gpa, branch.table) catch |err| return load_file.narrow(err);
                errdefer table.deinit(gpa);
                // Every checkpoint's rows are read from the mapping as they are gathered (bf16 or NVFP4 codes),
                // so each keeps the drive awake: through the shard's own O_DIRECT
                // descriptor, which reads past the page cache and touches no mapping.
                if (drive == null and table.shards.len > 0) if (found.checkpoint.fileOf(table.shards[0])) |file| {
                    drive = Drive.open(gpa, file.direct, file.map.len, .{});
                };
                table.drive = drive;
                try table.readFrom(gpa, found.checkpoint);
                const weights = ple.Weights.init(gpa, cfg, branch, centred, table.scale, scratch) catch |err| return load_file.narrow(err);
                layer.ple = .{ .weights = weights, .table = table, .ngram = branch.ngram, .eos = cfg.ple_eos };
            }
            loaded += 1;
        }

        // a prompt's windows in the card's form (spec 143 WS-06); a call outside a window's form quantizes nothing
        pool.quantize_from = forms.none;
        if (prompt_form == .card) for (layers) |*layer| {
            const kept = try layer.quantizePrompts(shape);
            if (kept > 0) std.log.debug("a layer keeps bf16 rows on {d} MXFP8 runs", .{kept});
        };
        // the MTP head's parts are held, not staged: its experts are packed by the host
        try staging.settle();
        staging.place = false;
        staging.swap = null;
        const top_arena = try reading.begin(reading.top());
        const mixer_whole = try reading.moved(top_arena, found.mixer orelse return error.MissingTensors);
        const mixer_source = if (rank) |one| share_layout.sliceHyper(piece.allocator(), mixer_whole, one, heads) catch |err| return load_file.cutShare(err) else mixer_whole;
        var mixer = try load_file.loadHyperConnection(gpa, mixer_source, centred, scratch);
        errdefer mixer.deinit();
        const head_whole = try reading.moved(top_arena, found.head orelse return error.MissingTensors);
        const head_linear = if (rank) |one| share_layout.sliceLinear(piece.allocator(), head_whole, one, heads) catch |err| return load_file.cutShare(err) else head_whole;
        // Beside a share the draft head is cut from the whole head's rows, so the whole is held (1.3 GiB,
        // released below) and the rank's half, rows of it, lies inside.
        const head_scored = if (rank != null and drafting) head_whole else head_linear;
        var head: Face = try load_file.faceOf(gpa, &.{head_linear}, scratch);
        errdefer head.deinit();
        // the head's sizes are the whole model's, on a share too
        const mtp_staged: ?layout.Mtp = if (drafting) staged: {
            const arena = try reading.begin(reading.mtpPart());
            break :staged try reading.moved(arena, found.mtp.?);
        } else null;
        var mtp: ?Mtp = if (drafting) try Mtp.load(gpa, &staging, config, found, scratch, options.drafts_stored, if (split) rank else null, mtp_staged) else null;
        errdefer if (mtp) |*drafter| drafter.deinit(gpa);
        if (mtp) |*drafter| if (!options.drafts_stored) if (options.draft_vocab) |all| {
            // split, a rank scores its part of the ids, cut from the whole head's rows as rank 0's whole one is
            const ids = if (split) Mtp.Split.partOf(all, rank.?.rank) else all;
            if (split) drafter.split = .{ .ids = try gpa.dupe(u32, all) };
            const weight: requantize.Weight = switch (try head_scored.part()) {
                .bf16 => |bytes| .{ .bf16 = bytes },
                .fp8_block => |rows| .{ .fp8_block = rows },
                .mxfp8 => return error.UnsupportedScheme,
            };
            for (ids) |id| if (id >= shape.vocab) return error.UnsupportedShape;
            drafter.draft = try DraftHead.init(gpa, weight, shape.vocab, shape.hidden, ids);
        };
        if (rank != null and !split) if (mtp) |*drafter| {
            var whole_attention = attention.Shape.of(&config) catch return error.UnsupportedShape;
            whole_attention.kv = options.kv;
            drafter.whole = .{ .shape = .of(config), .attention = .{ .shape = whole_attention, .rotary = try .init(whole_attention), .head = head_kv } };
        };
        const shared: ?Attention = if (uses_attention or mtp != null) .{ .shape = attention_shape, .rotary = try .init(attention_shape), .head = head_kv } else null;
        return .{ .shape = shape, .embed = embed, .layers = layers, .mixer = mixer, .head = head, .attention = shared, .mtp = mtp, .max_rows = max_rows, .scratch = pool, .drive = drive, .gpa = gpa };
    }
};

pub const hold = load_file.hold;
pub const Reading = load_file.Reading;
pub const faceOf = load_file.faceOf;
