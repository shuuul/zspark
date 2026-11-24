//! A NemotronH full-attention layer: NoPE GQA (`heads` query over `kv_heads` KV heads of `head_dim`), the
//! q, k and v projections one stacked `linear.Linear` (`[q | k | v]` of one input, as TensorFold's fused qkv and
//! vLLM's `QKVParallelLinear` lay them; layout.zig keeps the three apart in storage, stacked at load) and o_proj,
//! both read as the checkpoint stores them (NVFP4 group 16 on the backbone, per-tensor FP8 on the MTP block).
//! No q/k norms and no rotary: the checkpoint's headers hold neither and vLLM's nemotron_h.py
//! `NemotronHAttention` applies none (the config's `rope_theta` and `partial_rotary_factor` go unread). The
//! scale is `head_dim`^-1/2.
//!
//! `kernels/common/attention_prep.cu` splits the row's heads (a null `inv`: no rotation) and writes k and v into
//! the bf16 caches at the row's position; `common/attention.cu` and `attention_group.cu` attend: a row's keys are
//! positions 0 .. pos + r, chunks of 512 in index order merged in order, so a row's bits are the same in a serial
//! step, a verify window and a prompt chunk. TensorFold's own kernels (`tf_nemo_kv_write`, `tf_nemo_attn_chunk`,
//! `tf_nemo_attn_merge`) are the same arithmetic in another thread map and are not ported: zspark's common ones
//! Qwen3-ASR and Unlimited-OCR already run keep the row rule the same way. The card's FP8 KV cache launch option
//! is not taken: `kv_cache_quant_algo` is null on this checkpoint and `prep` writes bf16 alone (`attend` reads
//! e4m3 keys where a caller stores them; recording the choice for when the launch flag asks).
//!
//! A sequence's state is one `Cache` a layer (the keys and values alone; the position word is the caller's, so
//! a step's launches are the same at every position). A call writes the window's rows at pos .. pos + rows - 1
//! before any row reads them and bounds its keys at `context`; a rejected window's rows are never read, so a
//! rollback costs nothing. `window` is a step or a verify window (1 to `window_rows` rows); `prompt` a chunk of
//! any rows `Scratch` holds, the same bits a row's serial steps give.

const std = @import("std");
const ranks_file = @import("ranks.zig");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const config_file = @import("config.zig");
const linear = @import("../../weights/linear.zig");
const modelopt = @import("../../weights/modelopt.zig");
const safetensors = @import("../../weights/safetensors.zig");

/// A window's most rows: a step or a verify window (`draft.forward_max_rows`, as mamba.zig's).
pub const window_rows = 16;

/// Keys of a row one chunk of `common/attention.cu` takes, on the narrow attend path.
pub const chunk = 512;

/// From this many rows `attend` takes the group kernel (a group of rows a block); fewer take the chunks and a
/// merge. The same bits.
pub const group_rows = 64;

/// The layer's dimensions; `of` the backbone's (32 query over 2 KV heads of 128, eps the layers' RMS norm's).
pub const Shape = struct {
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,

    pub fn of(b: config_file.Backbone) Shape {
        return .{ .hidden = b.hidden, .heads = b.heads, .kv_heads = b.kv_heads, .head_dim = b.head_dim, .eps = b.eps };
    }

    /// The stacked projection's outputs a row: q's heads, then k's and v's.
    pub fn qkvWidth(s: Shape) usize {
        return (s.heads + 2 * s.kv_heads) * s.head_dim;
    }
};

/// One sequence's keys and values of this layer: bf16 [capacity, kv_heads, head_dim] each, as
/// `attention_prep.cu` writes them and the attends read them. The position the next row writes at is the
/// caller's own device word, given at each call.
pub const Cache = struct {
    keys: cuda.Buffer,
    values: cuda.Buffer,
    capacity: usize,

    pub fn init(s: Shape, capacity: usize) cuda.Error!Cache {
        var keys: cuda.Buffer = try .init(.bf16, &.{ capacity, s.kv_heads, s.head_dim });
        errdefer keys.deinit();
        const values: cuda.Buffer = try .init(.bf16, &.{ capacity, s.kv_heads, s.head_dim });
        return .{ .keys = keys, .values = values, .capacity = capacity };
    }

    pub fn deinit(c: *Cache) void {
        c.keys.deinit();
        c.values.deinit();
    }
};

/// What a call lends, made once for the most rows a prompt chunk takes and shared by every attention layer:
/// qkv bf16 [rows, qkvWidth] (the stacked projection's product), q bf16 [rows, heads, head_dim] and the
/// attended rows bf16 [rows, heads, head_dim] (o_proj's input); po, pm, pl fp32 the narrow attend's chunk
/// partials ([rows, nch, heads, head_dim] and [rows, nch, heads] twice, nch the chunks `capacity` keys make;
/// the group path reads none).
pub const Scratch = struct {
    rows: usize,
    nch: usize,
    /// parts[0] qkv, [1] q, [2] attended, [3..6) po, pm, pl
    parts: [6]cuda.Buffer,
    block: cuda.Buffer,

    /// `rows` the most a `prompt` chunk holds; a `window` takes up to `window_rows` of it. `capacity` the most
    /// positions a row's chunks may reach (the partials are sized by it).
    pub fn init(s: Shape, rows: usize, capacity: usize) cuda.Error!Scratch {
        const r = @max(rows, window_rows);
        const nch = (capacity + chunk - 1) / chunk;
        var sc: Scratch = .{ .rows = r, .nch = nch, .parts = .{
            .shaped(.bf16, &.{ r, s.qkvWidth() }),
            .shaped(.bf16, &.{ r, s.heads, s.head_dim }),
            .shaped(.bf16, &.{ r, s.heads, s.head_dim }),
            .shaped(.f32, &.{ r, nch, s.heads, s.head_dim }),
            .shaped(.f32, &.{ r, nch, s.heads }),
            .shaped(.f32, &.{ r, nch, s.heads }),
        }, .block = undefined };
        sc.block = try cuda.cut(&.{&sc.parts});
        return sc;
    }

    pub fn deinit(sc: *Scratch) void {
        sc.block.deinit();
    }

    /// The stacked projection's product, bf16 [rows, qkvWidth].
    pub fn qkv(sc: Scratch, rows: usize) cuda.View {
        return sc.parts[0].rows(rows);
    }
};

/// The layer's weights on the device, passed by value. `qkv` the q, k and v projections stacked (one
/// `linear.Linear.stored`: NVFP4, per-tensor FP8 or BF16 as the quant map stores them), `o` o_proj's.
pub const Attention = struct {
    shape: Shape,
    qkv: linear.Linear,
    o: linear.Linear,

    /// Reads the layer at `prefix` ++ `layer` ++ `.mixer.*` (`backbone.layers.N.` or `llm.mtp.layers.N.`): the
    /// three projections stacked, then o_proj, from `checkpoint` as `quant` stores them; `rows` the rows a call
    /// takes (`cuda.Scratch`'s). A q/k-norm tensor beside them is refused: this checkpoint ships none and the
    /// port keeps none (NoPE; a build that gained one would want `zs_attn_prep`'s norms instead). `s` the
    /// rows to the rank's heads and `o`'s columns), the full shape's `Shape.of(b)` and null without one.
    pub fn init(arena: Allocator, gpa: Allocator, b: config_file.Backbone, prefix: []const u8, layer: usize, checkpoint: *const safetensors.Checkpoint, quant: config_file.Quant, rows: cuda.Scratch.Rows, diagnostic: *config_file.Diagnostic, s: Shape, share: ?ranks_file.Share) !Attention {
        const full = Shape.of(b);
        var buffers: [4][160]u8 = undefined;
        const at = struct {
            fn module(buffer: []u8, p: []const u8, i: usize, leaf: []const u8) []const u8 {
                return std.fmt.bufPrint(buffer, "{s}{d}.mixer.{s}", .{ p, i, leaf }) catch unreachable;
            }
        }.module;
        for ([_][]const u8{ "q_norm.weight", "k_norm.weight" }) |leaf| {
            if (checkpoint.get(at(&buffers[0], prefix, layer, leaf)) != null) {
                diagnostic.set(arena, "{s}{d}.mixer has {s}: NemotronH's attention is NoPE with no head norms (vLLM's nemotron_h.py)", .{ prefix, layer, leaf });
                return error.UnexpectedTensor;
            }
        }
        var q = try modelopt.read(arena, checkpoint, at(&buffers[0], prefix, layer, "q_proj"), quant.find(at(&buffers[0], prefix, layer, "q_proj")), full.heads * full.head_dim, s.hidden, diagnostic);
        var k = try modelopt.read(arena, checkpoint, at(&buffers[1], prefix, layer, "k_proj"), quant.find(at(&buffers[1], prefix, layer, "k_proj")), full.kv_heads * full.head_dim, s.hidden, diagnostic);
        var v = try modelopt.read(arena, checkpoint, at(&buffers[2], prefix, layer, "v_proj"), quant.find(at(&buffers[2], prefix, layer, "v_proj")), full.kv_heads * full.head_dim, s.hidden, diagnostic);
        var o = try modelopt.read(arena, checkpoint, at(&buffers[3], prefix, layer, "o_proj"), quant.find(at(&buffers[3], prefix, layer, "o_proj")), s.hidden, full.heads * full.head_dim, diagnostic);
        if (share) |sh| {
            // this rank's heads' rows of q, k, v; the same heads' columns of o (its fp32 partial is summed
            // with the peer's, TensorFold `tp.py`'s split_attention)
            q = try modelopt.cut(arena, q, .row, sh.rank, ranks_file.world, diagnostic);
            k = try modelopt.cut(arena, k, .row, sh.rank, ranks_file.world, diagnostic);
            v = try modelopt.cut(arena, v, .row, sh.rank, ranks_file.world, diagnostic);
            o = try modelopt.cut(arena, o, .col, sh.rank, ranks_file.world, diagnostic);
        }
        var qkv: linear.Linear = try .stored(arena, gpa, &.{ q, k, v }, rows, diagnostic);
        errdefer qkv.deinit();
        var stored_o: linear.Linear = try .stored(arena, gpa, &.{o}, rows, diagnostic);
        errdefer stored_o.deinit();
        if (q.scheme == .nvfp4) {
            try qkv.four.quantizeActivations(rows, @max(try inputScale(q), @max(try inputScale(k), try inputScale(v))));
            try stored_o.four.quantizeActivations(rows, try inputScale(o));
        }
        return .{ .shape = s, .qkv = qkv, .o = stored_o };
    }

    fn inputScale(projection: modelopt.Linear) !f32 {
        const tensor = projection.input_scale orelse return error.UnsupportedShape;
        return @bitCast(std.mem.readInt(u32, tensor.data[0..4], .little));
    }

    pub fn deinit(a: *Attention) void {
        a.qkv.deinit();
        a.o.deinit();
    }

    /// input -> out bf16 [rows, heads * head_dim] the attention's product before o_proj. `pos` int32 [1], the
    /// sequence's committed length, on the device: the rows' keys and values land at pos .. pos + rows - 1 and
    /// row r reads positions 0 .. pos + r. `context` bounds the chunks launched (pos + rows or more, within the
    /// caches), which changes no bit. Nothing of the window is committed: the caller advances `pos` for the
    /// rows it keeps, and a rejected row's cache rows are never read.
    pub fn window(a: Attention, cache: Cache, sc: Scratch, x: cuda.View, pos: cuda.View, context: usize, out: cuda.View) cuda.Error!void {
        const rows: usize = @intCast(x.size[0]);
        try cuda.bound(rows >= 1 and rows <= window_rows, "attention.window: {d} rows, 1 to {d}", .{ rows, window_rows });
        return run(a, cache, sc, x, pos, context, out, rows);
    }

    /// A prompt chunk of 1 to `sc.rows` rows, kept whole: as `window`. The same bits the rows' serial steps
    /// give (the cache writes and the attends are a window's, a row's keys bounded at its own position).
    pub fn prompt(a: Attention, cache: Cache, sc: Scratch, x: cuda.View, pos: cuda.View, context: usize, out: cuda.View) cuda.Error!void {
        const rows: usize = @intCast(x.size[0]);
        try cuda.bound(rows >= 1 and rows <= sc.rows, "attention.prompt: {d} rows, 1 to {d}", .{ rows, sc.rows });
        return run(a, cache, sc, x, pos, context, out, rows);
    }

    /// The stacked projection, the prep (no rotation) and the attend over `rows` rows of `x`, its product in
    /// `sc`'s attended rows and `o` into `out` bf16 [rows, hidden]: the layer's output for the caller's
    /// residual add (`o.slices` folds there).
    fn run(a: Attention, cache: Cache, sc: Scratch, x: cuda.View, pos: cuda.View, context: usize, out: cuda.View, rows: usize) cuda.Error!void {
        const s = a.shape;
        const stacked = sc.qkv(rows);
        const q = sc.parts[1].rows(rows);
        try a.qkv.forward(x, stacked);
        try cuda.launched(cuda.kernels.zs_attn_prep(&stacked, null, null, null, &pos, &q, &cache.keys.view, &cache.values.view, s.eps));
        const attended = sc.parts[2].rows(rows);
        const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(s.head_dim)));
        if (rows >= group_rows) {
            try cuda.launched(cuda.kernels.zs_attn_attend_group(&q, &cache.keys.view, &cache.values.view, &pos, &attended, scale, @intCast(context)));
        } else {
            try cuda.launched(cuda.kernels.zs_attn_attend(&q, &cache.keys.view, &cache.values.view, &pos, &sc.parts[3].view, &sc.parts[4].view, &sc.parts[5].view, &attended, null, null, null, null, null, null, scale, @intCast(context)));
        }
        var flat: cuda.View = attended;
        flat.ndim = 2;
        flat.size[1] = @intCast(s.heads * s.head_dim);
        return a.o.forward(flat, out);
    }
};
