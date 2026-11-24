//! Flash Next's attention block on the device, decode path: the stacked projection, the prep that normalizes and
//! rotates the heads and writes the window's rows into the caches, the indexer's selection of the keys a row reads
//! once its context outgrows the budget (QSA), the attention itself, the output gate and the output projection.
//! A row's bits depend on its own input, its position and the cache rows before it, never on the window, the
//! caches' capacity or a launch's bound. The caches are `attention_cache.zig`'s: their storage by dtype, the
//! pools of spec 018, a sequence's view of them.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Face = @import("../../../weights/face.zig").Face;
const cuda = @import("../../../cuda/cuda.zig");
const Config = @import("../config.zig").Config;
const layout = @import("../layout.zig");
const extents = @import("../../../decode/extents.zig");
const gdn = @import("gdn.zig");
const cache_file = @import("attention_cache.zig");

pub const Storage = cache_file.Storage;
pub const Pool = cache_file.Pool;
pub const Table = cache_file.Table;
pub const Cache = cache_file.Cache;

/// Keys of a row's list that one chunk of the attention takes; chunks are merged in order.
pub const chunk = 512;

/// `ZSPARK_ATTN_WINDOW` (spec 105 WS-05): from this many rows a launch on the contiguous caches attends a row a
/// block, its chunks one after the other and their merge in registers (`zig/kernels/common/attention_window.cu`):
/// `attention.cu`'s bits in one launch without the partials. By time alone; 0 never, 1 always (a test).
pub var window_rows: usize = 64;

comptime {
    // an extent of the pool is the positions of one chunk of a dense row: the table is read once a chunk block
    std.debug.assert(extents.positions == chunk and 1 << extents.shift == chunk);
}

/// The sizes of an attention layer.
pub const Shape = struct {
    /// The model's hidden size.
    dims: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// The indexer's query heads; it has one key head.
    index_heads: usize,
    index_dim: usize,
    /// Leading values of a head (and of an indexer head) that rotate.
    rotary_dim: usize,
    rope_theta: f64,
    /// Keys a row attends once it has more complete blocks than `top()`, not counting its unfinished block.
    budget: usize,
    /// Positions pooled into one indexer key.
    ratio: usize,
    eps: f32,
    /// The caches' storage: an option of the load, not of the checkpoint.
    kv: Storage = .bf16,
    /// `mrope_section`: of a row's rotary pairs, how many take an image's t, h and w axes (`Rope`).
    sections: [3]u32 = .{ 11, 11, 10 },

    pub fn of(c: *const Config) error{UnsupportedShape}!Shape {
        if (c.index_kv_heads != 1 or c.heads % c.kv_heads != 0 or c.heads / c.kv_heads > 16 or
            (c.head_dim != 32 and c.head_dim != 64 and c.head_dim != 128 and c.head_dim != 256) or
            c.index_dim > 256 or c.rotary_dim > c.index_dim or c.rotary_dim > c.head_dim) return error.UnsupportedShape;
        return .{
            .dims = c.hidden,
            .heads = c.heads,
            .kv_heads = c.kv_heads,
            .head_dim = c.head_dim,
            .index_heads = c.index_heads,
            .index_dim = c.index_dim,
            .rotary_dim = c.rotary_dim,
            .rope_theta = c.rope_theta,
            .budget = c.index_budget,
            .ratio = c.index_ratio,
            .eps = @floatCast(c.eps),
            .sections = c.mrope_section,
        };
    }

    /// A row of the stacked projection: [q | gate] a head, k, v, the indexer's queries, the indexer's key.
    pub fn projection(self: Shape) usize {
        return self.heads * 2 * self.head_dim + 2 * self.kv_heads * self.head_dim + (self.index_heads + 1) * self.index_dim;
    }

    /// The gated output a row hands the output projection.
    pub fn out(self: Shape) usize {
        return self.heads * self.head_dim;
    }

    /// Blocks a sparse row attends.
    pub fn top(self: Shape) usize {
        return self.budget / self.ratio;
    }

    /// A sparse row's list at its longest, plus one: `top()` blocks and an unfinished block.
    pub fn listWidth(self: Shape) usize {
        return (self.top() + 1) * self.ratio;
    }

    /// Chunks covering the longest list of keys a row reads from caches of `capacity` rows.
    pub fn chunks(self: Shape, capacity: usize) usize {
        return (@min(capacity, self.listWidth() - 1) + chunk - 1) / chunk;
    }

    /// Whether a row in caches of `rows` rows can have more complete blocks than it attends.
    pub fn selects(self: Shape, rows: usize) bool {
        return rows / self.ratio > self.top();
    }
};

/// A layer's attention weights on the device.
pub const Weights = struct {
    /// [projection, dims]: q_proj, k_proj, v_proj and the indexer's index_qk_proj stacked in that order.
    proj: Face,
    /// fp32 [head_dim]: the queries' RMSNorm weight
    q_scale: cuda.Buffer,
    /// fp32 [head_dim]
    k_scale: cuda.Buffer,
    /// fp32 [index_dim]
    index_q_scale: cuda.Buffer,
    /// fp32 [index_dim]: the pooled indexer keys' RMSNorm weight
    index_k_scale: cuda.Buffer,
    /// [dims, heads head_dim]
    o: Face,

    /// The weights on the host: bf16 bytes row-major as the checkpoint stores them, and the norms as scales
    /// (a centred norm's + 1 already taken).
    pub const Host = struct {
        /// [heads 2 head_dim, dims]
        q: []const u8,
        /// [kv_heads head_dim, dims]
        k: []const u8,
        /// [kv_heads head_dim, dims]
        v: []const u8,
        /// [(index_heads + 1) index_dim, dims]
        index_qk: []const u8,
        /// [dims, heads head_dim]
        o: []const u8,
        q_norm: []const f32,
        k_norm: []const f32,
        index_q_norm: []const f32,
        index_k_norm: []const f32,
    };

    /// The norms as scales (a centred norm's + 1 already taken).
    pub const Norms = struct {
        q: []const f32,
        k: []const f32,
        index_q: []const f32,
        index_k: []const f32,
    };

    /// Weights whose linears are all BF16; calls take up to `max_rows` rows.
    pub fn init(shape: Shape, host: Host, scratch: cuda.Scratch.Rows) !Weights {
        std.debug.assert(host.o.len == shape.dims * shape.out() * 2);
        var proj: Face = try .initStacked(&.{ host.q, host.k, host.v, host.index_qk }, shape.dims, scratch);
        errdefer proj.deinit();
        var o: Face = try .init(host.o, shape.dims, shape.out(), scratch);
        errdefer o.deinit();
        return fromFaces(shape, proj, o, .{ .q = host.q_norm, .k = host.k_norm, .index_q = host.index_q_norm, .index_k = host.index_k_norm });
    }

    /// Weights over faces already on the device, which they own once this returns; on failure the faces stay the
    /// caller's.
    pub fn fromFaces(shape: Shape, proj: Face, o: Face, host: Norms) !Weights {
        if (proj.n != shape.projection() or proj.k != shape.dims or o.n != shape.dims or o.k != shape.out()) return error.UnsupportedShape;
        var q_scale = try uploaded(f32, .f32, &.{shape.head_dim}, host.q);
        errdefer q_scale.deinit();
        var k_scale = try uploaded(f32, .f32, &.{shape.head_dim}, host.k);
        errdefer k_scale.deinit();
        var index_q_scale = try uploaded(f32, .f32, &.{shape.index_dim}, host.index_q);
        errdefer index_q_scale.deinit();
        const index_k_scale = try uploaded(f32, .f32, &.{shape.index_dim}, host.index_k);
        return .{ .proj = proj, .q_scale = q_scale, .k_scale = k_scale, .index_q_scale = index_q_scale, .index_k_scale = index_k_scale, .o = o };
    }

    /// From a checkpoint's tensors; `centred` is the layout's `norms_centred`. Each linear is BF16 or MXFP8,
    /// read as stored: neighbours of one storage share a matmul, as TensorFold's `attention_nvfp4` stacks them.
    pub fn load(gpa: Allocator, shape: Shape, tensors: layout.Attention, centred: bool, scratch: cuda.Scratch.Rows) !Weights {
        var proj: Face = try .initParts(gpa, &.{ try tensors.q.part(), try tensors.k.part(), try tensors.v.part(), try tensors.index_qk.part() }, shape.dims, scratch);
        errdefer proj.deinit();
        var o: Face = try .initParts(gpa, &.{try tensors.o.part()}, shape.out(), scratch);
        errdefer o.deinit();
        var norms: [4][]f32 = undefined;
        var made: usize = 0;
        defer for (norms[0..made]) |norm| gpa.free(norm);
        for ([_]layout.Vector{ tensors.q_norm, tensors.k_norm, tensors.index_q_norm, tensors.index_k_norm }) |vector| {
            const scale = try gpa.alloc(f32, vector.len());
            norms[made] = scale;
            made += 1;
            const shift: f32 = if (vector.gamma and centred) 1.0 else 0.0;
            for (scale, 0..) |*value, i| value.* = shift + vector.value(i);
        }
        return fromFaces(shape, proj, o, .{ .q = norms[0], .k = norms[1], .index_q = norms[2], .index_k = norms[3] });
    }

    pub fn deinit(self: *Weights) void {
        self.proj.deinit();
        self.q_scale.deinit();
        self.k_scale.deinit();
        self.index_q_scale.deinit();
        self.index_k_scale.deinit();
        self.o.deinit();
        self.* = undefined;
    }
};

/// The rotary frequencies every attention layer shares: fp32 [rotary_dim / 2], theta^(-i / half) from fp64.
pub const Rotary = struct {
    inv_freq: cuda.Buffer,

    pub fn init(shape: Shape) cuda.Error!Rotary {
        var host: [128]f32 = undefined;
        const half = shape.rotary_dim / 2;
        frequencies(shape, host[0..half]);
        return .{ .inv_freq = try uploaded(f32, .f32, &.{half}, host[0..half]) };
    }

    pub fn frequencies(shape: Shape, out: []f32) void {
        const half: f64 = @floatFromInt(shape.rotary_dim / 2);
        for (out, 0..) |*value, i| value.* = @floatCast(std.math.pow(f64, shape.rope_theta, -@as(f64, @floatFromInt(i)) / half));
    }

    pub fn deinit(self: *Rotary) void {
        self.inv_freq.deinit();
        self.* = undefined;
    }
};

/// A sequence's committed length, on the host and as the int32 [1] the kernels read (a captured launch reads the
/// device's copy, so a graph does not bake a position in).
pub const Position = struct {
    /// i32 [2]: the position, and a word that goes with it in the one copy (`put`).
    words: cuda.Buffer,
    /// The first word, i32 [1]: `words`' memory, not its own.
    device: cuda.Buffer,
    host: usize,

    pub fn init() cuda.Error!Position {
        const words: cuda.Buffer = try .init(.i32, &.{2});
        var self: Position = .{ .words = words, .device = .{ .view = words.span(0, 1) }, .host = 0 };
        errdefer self.words.deinit();
        try self.set(0);
        return self;
    }

    pub fn set(self: *Position, pos: usize) cuda.Error!void {
        return self.put(pos, 0);
    }

    /// `set`, and the second word in the same copy: the rows a sequence's DeltaNet states are behind its
    /// position by (`forward.State.behind`), which each layer's chain reads on the device as attention reads
    /// the position. One copy a commit, as before.
    pub fn put(self: *Position, pos: usize, behind: usize) cuda.Error!void {
        const words: [2]i32 = .{ @intCast(pos), @intCast(behind) };
        try self.words.upload(std.mem.asBytes(&words));
        self.host = pos;
    }

    /// The second word on the device, i32 [1].
    pub fn behindView(self: Position) cuda.View {
        return self.words.span(1, 1);
    }

    /// Commits the first `keep` rows of the window just run. This is all an attention layer's caches need:
    /// the next window starts at the new position and writes its own key, value and indexer-key rows before
    /// any row of it reads them, and a pooled block that took a dropped row is pooled again by the window that
    /// completes it, before a row can score it. No row reads a cache row at or past its own position + 1.
    pub fn advance(self: *Position, keep: usize) cuda.Error!void {
        return self.set(self.host + keep);
    }

    pub fn deinit(self: *Position) void {
        self.words.deinit();
        self.* = undefined;
    }
};

/// The switch of spec 018's layout for caches made by `Cache.init`: null, as everything that serves leaves it,
/// gives a sequence's rows in order and the kernels zspark always ran. Set, each cache made lies in a pool of
/// its own behind a table of its own. It is a test's facility (`Cache.initTabled`): a served layout shares one
/// pool a layer among sequences (`Cache.over`), which is spec 018's step B3. A listed row's ids are then the
/// pool's rows: the selection wrote them through the table (`zig/kernels/qwen4/qsa_table.cu`) and the attend kernel
/// reads a list as it does over caches in order (`attention_table.cu`).
pub const Tabled = struct {
    /// The extents in a fixed shuffle that depends on their number alone (so two caches of one capacity lie
    /// alike, as a sequence's layers do), or in order.
    scattered: bool = true,
};

/// What the selection writes for a window: each row's key list, and the scratch it scores with.
pub const Selection = struct {
    /// Complete blocks the caches can hold.
    blocks: usize,
    /// int32 [rows, (top + 1) ratio]: a sparse row's keys, its chosen blocks in order, then its unfinished block
    ids: cuda.Buffer,
    /// int32 [rows]: keys the row attends
    nk: cuda.Buffer,
    /// int32 [rows]: 1 where the row reads `ids`, 0 where it reads positions 0 .. its own
    sparse: cuda.Buffer,
    /// Token rows `dots` holds: the window, or `gdn.wide_rows` when the window is wider. A captured round asks
    /// for fewer than that, so it is one tile; a prompt window is several, and a row's products do not depend
    /// on the rows beside it.
    scored: usize,
    /// fp32 [scored * index_heads, blocks in use]: one tile of indexer queries against each pooled block
    dots: cuda.Buffer,
    /// fp32 [rows, blocks]
    scores: cuda.Buffer,
    /// int32 [1]: a later tile's position, the window's start plus the rows already scored. A round that fits
    /// in `scored` never writes it.
    tile_pos: cuda.Buffer,

    pub fn init(shape: Shape, rows: usize, capacity: usize) cuda.Error!Selection {
        const blocks = capacity / shape.ratio;
        const scored = @min(rows, gdn.wide_rows);
        var ids: cuda.Buffer = try .init(.i32, &.{ rows, shape.listWidth() });
        errdefer ids.deinit();
        var nk: cuda.Buffer = try .init(.i32, &.{rows});
        errdefer nk.deinit();
        var sparse: cuda.Buffer = try .init(.i32, &.{rows});
        errdefer sparse.deinit();
        var dots: cuda.Buffer = try .init(.f32, &.{ scored * shape.index_heads, blocks });
        errdefer dots.deinit();
        var scores: cuda.Buffer = try .init(.f32, &.{ rows, blocks });
        errdefer scores.deinit();
        const tile_pos: cuda.Buffer = try .init(.i32, &.{1});
        return .{ .blocks = blocks, .ids = ids, .nk = nk, .sparse = sparse, .scored = scored, .dots = dots, .scores = scores, .tile_pos = tile_pos };
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(shape: Shape, rows: usize, capacity: usize) usize {
        const of = cuda.Buffer.bytesOf;
        const blocks = capacity / shape.ratio;
        return of(.i32, &.{ rows, shape.listWidth() }) + 2 * of(.i32, &.{rows}) + of(.f32, &.{ @min(rows, gdn.wide_rows) * shape.index_heads, blocks }) +
            of(.f32, &.{ rows, blocks }) + of(.i32, &.{1});
    }

    pub fn deinit(self: *Selection) void {
        self.ids.deinit();
        self.nk.deinit();
        self.sparse.deinit();
        self.dots.deinit();
        self.scores.deinit();
        self.tile_pos.deinit();
        self.* = undefined;
    }
};

/// Scratch for windows of up to `rows` rows over caches of up to `capacity` rows. `gated` and `branch` are the
/// block's results.
pub const Scratch = struct {
    rows: usize,
    capacity: usize,
    /// bf16 [rows, projection]: the stacked projection
    pa: cuda.Buffer,
    /// bf16 or e4m3 [rows, heads, head_dim]: normalized, rotated queries, stored as the caches are
    q: cuda.Buffer,
    /// fp32 [rows, heads] unless the storage is bf16
    q_scale: ?cuda.Buffer,
    /// bf16 [rows, heads, head_dim]: the queries for a cache held in bf16 beside coded ones (the MTP head's,
    /// spec 052); null where every cache this scratch serves is stored as `q` is
    q_plain: ?cuda.Buffer = null,
    /// bf16 [rows, index_heads, index_dim]: the indexer's normalized, rotated queries
    iq: cuda.Buffer,
    /// fp32 [rows, chunks, heads, head_dim]: each chunk's unnormalized output
    partial_o: cuda.Buffer,
    /// fp32 [rows, chunks, heads]: each chunk's largest score
    partial_m: cuda.Buffer,
    /// fp32 [rows, chunks, heads]: each chunk's sum of exp(score - largest)
    partial_l: cuda.Buffer,
    /// bf16 [rows, heads, head_dim]: the attention's output
    attn: cuda.Buffer,
    /// bf16 [rows, heads head_dim]: the output times the sigmoid of each head's gate
    gated: cuda.Buffer,
    /// bf16 [rows, dims]: the output projection, the branch the next write-back takes
    branch: cuda.Buffer,
    /// Present when the caches can hold more complete blocks than a row attends.
    selection: ?Selection,

    /// `init`, serving caches held as `other` too. Only bf16 beside a coded storage is a second kind.
    pub fn initBeside(shape: Shape, other: Storage, rows: usize, capacity: usize) cuda.Error!Scratch {
        var self: Scratch = try .init(shape, rows, capacity);
        errdefer self.deinit();
        if (other != shape.kv) {
            std.debug.assert(other == .bf16);
            self.q_plain = try .init(.bf16, &.{ rows, shape.heads, shape.head_dim });
        }
        return self;
    }

    /// The device bytes `initBeside` asks, with nothing made.
    pub fn bytesBeside(shape: Shape, other: Storage, rows: usize, capacity: usize) usize {
        const of = cuda.Buffer.bytesOf;
        const nch = shape.chunks(capacity);
        var bytes = of(.bf16, &.{ rows, shape.projection() }) + of(shape.kv.dtype(), &.{ rows, shape.heads, shape.head_dim });
        if (shape.kv.scaled()) bytes += of(.f32, &.{ rows, shape.heads });
        if (other != shape.kv) bytes += of(.bf16, &.{ rows, shape.heads, shape.head_dim });
        bytes += of(.bf16, &.{ rows, shape.index_heads, shape.index_dim });
        bytes += of(.f32, &.{ rows, nch, shape.heads, shape.head_dim }) + 2 * of(.f32, &.{ rows, nch, shape.heads });
        bytes += of(.bf16, &.{ rows, shape.heads, shape.head_dim }) + of(.bf16, &.{ rows, shape.out() }) + of(.bf16, &.{ rows, shape.dims });
        if (shape.selects(capacity)) bytes += Selection.bytesFor(shape, rows, capacity);
        return bytes;
    }

    /// Where a block over `cache` keeps its queries, and their scales: stored as the cache's keys are.
    pub fn queries(self: Scratch, cache: Cache) Queries {
        if (cache.k_scale == null) if (self.q_plain) |plain| return .{ .q = plain, .scale = null };
        return .{ .q = self.q, .scale = self.q_scale };
    }

    pub const Queries = struct { q: cuda.Buffer, scale: ?cuda.Buffer };

    pub fn init(shape: Shape, rows: usize, capacity: usize) cuda.Error!Scratch {
        const nch = shape.chunks(capacity);
        var pa: cuda.Buffer = try .init(.bf16, &.{ rows, shape.projection() });
        errdefer pa.deinit();
        var q: cuda.Buffer = try .init(shape.kv.dtype(), &.{ rows, shape.heads, shape.head_dim });
        errdefer q.deinit();
        var q_scale: ?cuda.Buffer = if (shape.kv.scaled()) try .init(.f32, &.{ rows, shape.heads }) else null;
        errdefer if (q_scale) |*buffer| buffer.deinit();
        var iq: cuda.Buffer = try .init(.bf16, &.{ rows, shape.index_heads, shape.index_dim });
        errdefer iq.deinit();
        var partial_o: cuda.Buffer = try .init(.f32, &.{ rows, nch, shape.heads, shape.head_dim });
        errdefer partial_o.deinit();
        var partial_m: cuda.Buffer = try .init(.f32, &.{ rows, nch, shape.heads });
        errdefer partial_m.deinit();
        var partial_l: cuda.Buffer = try .init(.f32, &.{ rows, nch, shape.heads });
        errdefer partial_l.deinit();
        var attn: cuda.Buffer = try .init(.bf16, &.{ rows, shape.heads, shape.head_dim });
        errdefer attn.deinit();
        var gated: cuda.Buffer = try .init(.bf16, &.{ rows, shape.out() });
        errdefer gated.deinit();
        var branch: cuda.Buffer = try .init(.bf16, &.{ rows, shape.dims });
        errdefer branch.deinit();
        const selection: ?Selection = if (shape.selects(capacity)) try .init(shape, rows, capacity) else null;
        return .{ .rows = rows, .capacity = capacity, .pa = pa, .q = q, .q_scale = q_scale, .iq = iq, .partial_o = partial_o, .partial_m = partial_m, .partial_l = partial_l, .attn = attn, .gated = gated, .branch = branch, .selection = selection };
    }

    /// In the test mode `cuda.Scratch.poisoned`, fills every buffer with NaN patterns: a block that starts
    /// from that reads nothing the block before it left, whichever layer or model ran it.
    pub fn poison(self: Scratch) cuda.Error!void {
        inline for (.{ "pa", "q", "iq", "partial_o", "partial_m", "partial_l", "attn", "gated", "branch" }) |name| try cuda.Scratch.poison(@field(self, name));
        if (self.q_scale) |buffer| try cuda.Scratch.poison(buffer);
        if (self.q_plain) |buffer| try cuda.Scratch.poison(buffer);
        if (self.selection) |selection| {
            inline for (.{ "ids", "nk", "sparse", "dots", "scores", "tile_pos" }) |name| try cuda.Scratch.poison(@field(selection, name));
        }
    }

    pub fn deinit(self: *Scratch) void {
        self.pa.deinit();
        self.q.deinit();
        if (self.q_scale) |*buffer| buffer.deinit();
        if (self.q_plain) |*buffer| buffer.deinit();
        self.iq.deinit();
        self.partial_o.deinit();
        self.partial_m.deinit();
        self.partial_l.deinit();
        self.attn.deinit();
        self.gated.deinit();
        self.branch.deinit();
        if (self.selection) |*selection| selection.deinit();
        self.* = undefined;
    }
};

pub fn uploaded(comptime T: type, dtype: cuda.DType, shape: []const usize, host: []const T) cuda.Error!cuda.Buffer {
    var buffer: cuda.Buffer = try .init(dtype, shape);
    errdefer buffer.deinit();
    try buffer.upload(std.mem.sliceAsBytes(host));
    return buffer;
}

/// The first `rows` rows of `buffer` as a view.
pub fn viewOf(buffer: *const ?cuda.Buffer) ?*const cuda.View {
    return if (buffer.*) |*b| &b.view else null;
}

/// pa bf16 [rows, projection] at positions pos .. pos + rows - 1 -> `scratch`'s q [rows, heads, head_dim] (with
/// its scales where the caches are FP8) and iq bf16 [rows, index_heads, index_dim], and those positions' rows of
/// the caches. `context` is pos + rows. `rope` is the sequence's `Rope`, or null: the rows' own positions (the
/// pool's, whose caches are behind a table, take none).
pub fn prep(shape: Shape, weights: Weights, rotary: Rotary, cache: Cache, scratch: Scratch, pa: cuda.View, pos: cuda.View, rows: usize, context: usize, rope: ?cuda.View) cuda.Error!void {
    const mine = scratch.queries(cache);
    std.debug.assert(cache.table == null or rope == null);
    if (cache.table) |table| return cuda.launched(zs_qwen4_attn_prep_table(&table, &pa, &pos, &weights.q_scale.view, &weights.k_scale.view, &weights.index_q_scale.view, &rotary.inv_freq.view, &mine.q.rows(rows), &cache.k.view, &cache.v.view, &scratch.iq.rows(rows), &cache.index_keys.view, viewOf(&mine.scale), viewOf(&cache.k_scale), viewOf(&cache.v_scale), shape.eps, @intCast(context)));
    return cuda.launched(zs_qwen4_attn_prep(&pa, &pos, &weights.q_scale.view, &weights.k_scale.view, &weights.index_q_scale.view, &rotary.inv_freq.view, &mine.q.rows(rows), &cache.k.view, &cache.v.view, &scratch.iq.rows(rows), &cache.index_keys.view, viewOf(&mine.scale), viewOf(&cache.k_scale), viewOf(&cache.v_scale), shape.eps, @intCast(context), if (rope) |*r| r else null, shape.sections[1], shape.sections[2]));
}

/// Pools the blocks of indexer keys that positions pos .. pos + rows - 1 complete, rotated as `prep` rotates.
pub fn pool(shape: Shape, weights: Weights, rotary: Rotary, cache: Cache, pos: cuda.View, rows: usize, context: usize, rope: ?cuda.View) cuda.Error!void {
    std.debug.assert(cache.table == null or rope == null);
    if (cache.table) |table| return cuda.launched(zs_qwen4_attn_pool_table(&table, &cache.index_keys.view, &cache.pooled.view, &pos, &weights.index_k_scale.view, &rotary.inv_freq.view, shape.eps, @intCast(rows), @intCast(shape.ratio), @intCast(context)));
    return cuda.launched(zs_qwen4_attn_pool(&cache.index_keys.view, &cache.pooled.view, &pos, &weights.index_k_scale.view, &rotary.inv_freq.view, shape.eps, @intCast(rows), @intCast(shape.ratio), @intCast(context), if (rope) |*r| r else null, shape.sections[1], shape.sections[2]));
}

/// Each row's key list from its indexer queries iq bf16 [rows, index_heads, index_dim] and the pooled blocks
/// complete by the window's end. The products are a GEMM over the (pos + rows) / ratio pooled blocks with a
/// single K slice, whose bits for a (query, block) pair depend on neither count. `dots` holds `scored` token
/// rows, so a wider window is that many at a time and the lists land at row `at`. A round that fits in one
/// tile launches the one GEMM and one select it did, and does not write `tile_pos`. With `stored` (a
/// sequence's table, spec 018) a list holds the pool's rows of its keys, for `attend` over a cache behind
/// that table; without, positions.
pub fn select(shape: Shape, selection: Selection, iq: cuda.View, pooled: cuda.View, pos: cuda.View, at: usize, rows: usize, context: usize, stored: ?cuda.View) cuda.Error!void {
    const blocks = context / shape.ratio;
    std.debug.assert(blocks <= selection.blocks and blocks >= 1 and selection.scored >= 1);
    const heads = shape.index_heads;
    const dim = shape.index_dim;
    if (whole_rows != 0 and rows >= whole_rows and dim % 16 == 0) {
        // a prompt's window at once (spec 095): the scores without `dots`, one selection, the same lists
        whole_selections += 1;
        const q: cuda.View = .init(iq.ptr.?, .bf16, &.{ rows * heads, dim });
        const keys: cuda.View = .init(pooled.ptr.?, .bf16, &.{ blocks, dim });
        const scores = selection.scores.span(at, rows);
        const ids = selection.ids.span(at, rows);
        const nk = selection.nk.span(at, rows);
        const sparse = selection.sparse.span(at, rows);
        // behind a table (a pool's round, spec 149 WS-02): the same lists, written as the pool's rows
        if (stored) |table| return cuda.launched(cuda.kernels.zs_qwen4_attn_select_window_table(&table, &q, &keys, &pos, &scores, &ids, &nk, &sparse, @intCast(rows), @intCast(heads), @intCast(dim), @intCast(blocks), @intCast(shape.ratio), @intCast(shape.top())));
        return cuda.launched(zs_qwen4_attn_select_window(&q, &keys, &pos, &scores, &ids, &nk, &sparse, @intCast(rows), @intCast(heads), @intCast(dim), @intCast(blocks), @intCast(shape.ratio), @intCast(shape.top())));
    }
    var done: usize = 0;
    while (done < rows) {
        const n = @min(selection.scored, rows - done);
        const x: cuda.View = .init(tokenPtr(iq, done, heads, dim), .bf16, &.{ n * heads, dim });
        const w: cuda.View = .init(pooled.ptr.?, .bf16, &.{ blocks, dim });
        const dots: cuda.View = .init(selection.dots.view.ptr.?, .f32, &.{ n * heads, blocks });
        try cuda.launched(cuda.kernels.zs_b16_gemm(&x, &w, &dots, null, 1, 0));
        const tile_pos = if (done == 0) pos else shifted: {
            try cuda.launched(zs_qwen4_attn_pos_add(&pos, &selection.tile_pos.view, @intCast(done)));
            break :shifted selection.tile_pos.view;
        };
        try pick(shape, selection, dots, tile_pos, at + done, n, heads, dim, stored);
        done += n;
    }
}

/// A window of this many rows or more is selected at once by `zs_qwen4_attn_select_window` (`_table` behind a table)
/// (`qsa.cu`: the scores of every row in one launch, without `dots`, then one selection) in place of a GEMM, a
/// position, scores and a selection each `Selection.scored` rows; 0 is never. The same lists either way.
pub var whole_rows: usize = 33;
/// Selections that went so: a test's proof that its comparison was of the whole window.
pub var whole_selections: u64 = 0;

fn tokenPtr(iq: cuda.View, tokens: usize, heads: usize, dim: usize) *anyopaque {
    const bytes = tokens * heads * dim * iq.dtype.size();
    return @ptrCast(@as([*]u8, @ptrCast(iq.ptr.?)) + bytes);
}

/// The selection from the products dots fp32 [rows * heads, blocks]: a block's score is the sum over the heads of
/// relu(product) over sqrt(dim); a row with more than `top()` complete blocks takes its best, the lower block
/// among equal scores. The lists are written at row `at` of the selection's buffers.
pub fn pick(shape: Shape, selection: Selection, dots: cuda.View, pos: cuda.View, at: usize, rows: usize, heads: usize, dim: usize, stored: ?cuda.View) cuda.Error!void {
    var scores = selection.scores.span(at, rows);
    var ids = selection.ids.span(at, rows);
    var nk = selection.nk.span(at, rows);
    var sparse = selection.sparse.span(at, rows);
    if (stored) |table| return cuda.launched(cuda.kernels.zs_qwen4_attn_select_table(&table, &dots, &pos, &scores, &ids, &nk, &sparse, @intCast(heads), @intCast(dim), @intCast(shape.ratio), @intCast(shape.top())));
    return cuda.launched(zs_qwen4_attn_select(&dots, &pos, &scores, &ids, &nk, &sparse, @intCast(heads), @intCast(dim), @intCast(shape.ratio), @intCast(shape.top())));
}

/// q [rows, heads, head_dim] (`scratch.q`'s first rows, with `scratch.q_scale` where the caches are FP8) against
/// the caches -> out bf16 [rows, heads, head_dim]. Row r attends positions 0 .. pos + r, or its list in
/// `selection` where that marks it sparse.
pub fn attend(shape: Shape, cache: Cache, scratch: Scratch, selection: ?Selection, q: cuda.View, pos: cuda.View, out: cuda.View, context: usize) cuda.Error!void {
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(shape.head_dim)));
    const mine = scratch.queries(cache);
    const lists: [3]?*const cuda.View = if (selection) |*s| .{ &s.ids.view, &s.nk.view, &s.sparse.view } else .{ null, null, null };
    if (cache.table == null and window_rows != 0 and q.size[0] >= window_rows)
        return cuda.launched(cuda.kernels.zs_attn_attend_window(&q, &cache.k.view, &cache.v.view, &pos, &out, lists[0], lists[1], lists[2], viewOf(&mine.scale), viewOf(&cache.k_scale), viewOf(&cache.v_scale), scale, @intCast(context)));
    if (cache.table) |table| return cuda.launched(zs_attn_attend_table(&table, &q, &cache.k.view, &cache.v.view, &pos, &scratch.partial_o.view, &scratch.partial_m.view, &scratch.partial_l.view, &out, lists[0], lists[1], lists[2], viewOf(&mine.scale), viewOf(&cache.k_scale), viewOf(&cache.v_scale), scale, @intCast(context)));
    return cuda.launched(cuda.kernels.zs_attn_attend(&q, &cache.k.view, &cache.v.view, &pos, &scratch.partial_o.view, &scratch.partial_m.view, &scratch.partial_l.view, &out, lists[0], lists[1], lists[2], viewOf(&mine.scale), viewOf(&cache.k_scale), viewOf(&cache.v_scale), scale, @intCast(context)));
}

/// o bf16 [rows, heads, head_dim] and the stacked projection -> out bf16 [rows, heads head_dim].
/// A sequence's rotary positions where an image prompt gives three a row (spec 019; TensorFold 0.6.5's
/// `State.image_positions` and `rope_delta`, `image_rows.py`): i32 [2 + 3 capacity], the delta, the length the
/// table covers, then (t, h, w) for each position below it (`zig/kernels/qwen4/attention_rope.cuh` reads it). A
/// position at or past the length rotates by itself plus the delta, a reply's rows among them. Zeros, as `init`
/// and `clear` leave it, rotate every row by its own position: a text sequence's bits. The kernels read it from
/// device memory, so the graphs captured at load serve an image sequence too; it is made with the state and
/// lives as long.
pub const Rope = struct {
    words: cuda.Buffer,
    /// The positions the table holds.
    capacity: usize,
    /// What the device holds, as the host last wrote it.
    length: usize = 0,
    delta: i32 = 0,

    pub fn init(capacity: usize) cuda.Error!Rope {
        var words: cuda.Buffer = try .init(.i32, &.{2 + 3 * capacity});
        errdefer words.deinit();
        try words.uploadAt(0, std.mem.asBytes(&[2]i32{ 0, 0 }));
        return .{ .words = words, .capacity = capacity };
    }

    /// An image prompt's positions: `positions` i32 [length, 3] (a row's t, h, w), and `delta`, which the rows
    /// past them add to their own position. Waits for the launches before it, as an upload does.
    pub fn set(self: *Rope, positions: []const [3]i32, delta: i32) cuda.Error!void {
        std.debug.assert(positions.len <= self.capacity);
        if (positions.len > 0) try self.words.uploadAt(2 * @sizeOf(i32), std.mem.sliceAsBytes(positions));
        try self.words.uploadAt(0, std.mem.asBytes(&[2]i32{ delta, @intCast(positions.len) }));
        self.length = positions.len;
        self.delta = delta;
    }

    /// Back to every row at its own position. No copy where it is so already.
    pub fn clear(self: *Rope) cuda.Error!void {
        if (self.length == 0 and self.delta == 0) return;
        return self.set(&.{}, 0);
    }

    /// Another sequence's positions, as the device holds them (`copyLaunched`: in the stream's order).
    pub fn copyFrom(self: *Rope, other: Rope) cuda.Error!void {
        std.debug.assert(other.length <= self.capacity);
        try self.words.copyAt(0, other.words, (2 + 3 * other.length) * @sizeOf(i32));
        self.length = other.length;
        self.delta = other.delta;
    }

    pub fn view(self: Rope) cuda.View {
        return self.words.view;
    }

    pub fn bytesFor(capacity: usize) usize {
        return (2 + 3 * capacity) * @sizeOf(i32);
    }

    pub fn deinit(self: *Rope) void {
        self.words.deinit();
        self.* = undefined;
    }
};

pub fn gate(o: cuda.View, pa: cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_qwen4_attn_gate(&o, &pa, &out));
}

/// The block for a window: mixed bf16 [rows, dims] at positions `position` .. + rows - 1. `scratch.gated` holds
/// the gated attention output and `scratch.branch` its projection, the branch the next write-back takes. The
/// window's rows are written into `cache`; `position.advance(keep)` commits the first `keep` of them. `rope`
/// is the sequence's `Rope` (null: the rows' own positions).
pub fn forward(shape: Shape, weights: Weights, rotary: Rotary, cache: Cache, scratch: Scratch, position: Position, mixed: cuda.View, rows: usize, bound: ?usize, branch_out: ?cuda.View, rope: ?cuda.View) cuda.Error!void {
    const context = bound orelse position.host + rows;
    std.debug.assert(context >= position.host + rows);
    std.debug.assert(rows >= 1 and rows <= scratch.rows and mixed.size[0] == rows);
    std.debug.assert(context <= cache.capacity and cache.capacity <= scratch.capacity);
    const pos = position.device.view;
    const pa = scratch.pa.rows(rows);
    const q = scratch.queries(cache).q.rows(rows);
    const iq = scratch.iq.rows(rows);
    const attn = scratch.attn.rows(rows);
    const gated = scratch.gated.rows(rows);
    try weights.proj.forward(mixed, pa);
    try prep(shape, weights, rotary, cache, scratch, pa, pos, rows, context, rope);
    // TensorFold pools and selects whenever the caches' capacity exceeds the budget. Here every window pools, and
    // the selection runs once the window's last row has more complete blocks than it attends: before that every
    // row is dense under either rule, so the bits are the same and short contexts skip three launches.
    try pool(shape, weights, rotary, cache, pos, rows, context, rope);
    const selection: ?Selection = if (shape.selects(context)) scratch.selection.? else null;
    if (selection) |lists| try select(shape, lists, iq, cache.pooled.view, pos, 0, rows, context, cache.table);
    try attend(shape, cache, scratch, selection, q, pos, attn, context);
    try gate(attn, pa, gated);
    try weights.o.forward(gated, branch_out orelse scratch.branch.rows(rows));
}

extern fn zs_qwen4_attn_prep(p: *const cuda.View, pos: *const cuda.View, q_scale: *const cuda.View, k_scale: *const cuda.View, iq_scale: *const cuda.View, inv_freq: *const cuda.View, q: *const cuda.View, kc: *const cuda.View, vc: *const cuda.View, iq: *const cuda.View, ikc: *const cuda.View, qs: ?*const cuda.View, ks: ?*const cuda.View, vs: ?*const cuda.View, eps: f64, context: i64, rope: ?*const cuda.View, s1: i64, s2: i64) c_int;
extern fn zs_qwen4_attn_pool(ikc: *const cuda.View, pooled: *const cuda.View, pos: *const cuda.View, scale: *const cuda.View, inv_freq: *const cuda.View, eps: f64, rows: i64, ratio: i64, context: i64, rope: ?*const cuda.View, s1: i64, s2: i64) c_int;
extern fn zs_qwen4_attn_select_window(iq: *const cuda.View, pooled: *const cuda.View, pos: *const cuda.View, scores: *const cuda.View, ids: *const cuda.View, nk: *const cuda.View, sparse: *const cuda.View, rows: i64, heads: i64, dim: i64, nbl: i64, ratio: i64, top: i64) c_int;
extern fn zs_qwen4_attn_select(dots: *const cuda.View, pos: *const cuda.View, scores: *const cuda.View, ids: *const cuda.View, nk: *const cuda.View, sparse: *const cuda.View, heads: i64, dim: i64, ratio: i64, top: i64) c_int;
extern fn zs_qwen4_attn_pos_add(pos: *const cuda.View, out: *const cuda.View, add: i64) c_int;
extern fn zs_qwen4_attn_gate(o: *const cuda.View, p: *const cuda.View, out: *const cuda.View) c_int;

extern fn zs_qwen4_attn_prep_table(table: *const cuda.View, p: *const cuda.View, pos: *const cuda.View, q_scale: *const cuda.View, k_scale: *const cuda.View, iq_scale: *const cuda.View, inv_freq: *const cuda.View, q: *const cuda.View, kc: *const cuda.View, vc: *const cuda.View, iq: *const cuda.View, ikc: *const cuda.View, qs: ?*const cuda.View, ks: ?*const cuda.View, vs: ?*const cuda.View, eps: f64, context: i64) c_int;
extern fn zs_qwen4_attn_pool_table(table: *const cuda.View, ikc: *const cuda.View, pooled: *const cuda.View, pos: *const cuda.View, scale: *const cuda.View, inv_freq: *const cuda.View, eps: f64, rows: i64, ratio: i64, context: i64) c_int;
extern fn zs_attn_attend_table(table: *const cuda.View, q: *const cuda.View, kc: *const cuda.View, vc: *const cuda.View, pos: *const cuda.View, po: *const cuda.View, pm: *const cuda.View, pl: *const cuda.View, out: *const cuda.View, ids: ?*const cuda.View, nk: ?*const cuda.View, sparse: ?*const cuda.View, qs: ?*const cuda.View, ks: ?*const cuda.View, vs: ?*const cuda.View, scale: f64, context: i64) c_int;
