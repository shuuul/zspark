//! Qwen3-ASR's language model forward (Qwen3 1.7B): one window of rows after a sequence's state, as
//! `decode/draft.zig`'s `Decoder` asks (`draft.zig`). Semantics from the qwen-asr package's transformers backend,
//! modeling_qwen3_asr.py (Qwen3ASRThinkerTextModel and its decoder layer, attention, MLP; Apache-2.0, read
//! 2026-10-25), in bf16 as stored. A layer:
//!
//! 1. h = bf16(h + o(attention(rope(norms(qkv(norm(h))))))): GQA, a head's q and k over their own RMS norms, then
//!    split-half RoPE at the row's position (`kernels/common/attention_prep.cu`), the attend common/attention.cu's;
//! 2. h = bf16(h + down(bf16(bf16(silu(gate x)) up x))), x = norm(h).
//!
//! Each add takes the norm after it in its launch (`glue.residualNorm`, the K slices of `o` and `down` summed
//! there): the attention's add the post-attention norm, the MLP's the next layer's input norm or the final norm.
//!
//! The window's rows at positions pos .. pos + n - 1 write their keys and values at those rows of the caches
//! before any row reads them, so a window that is never committed leaves nothing a later one reads. A row's bits
//! are its own: the linears' K slices follow the weight's shape, attention's key chunks go in index order and are
//! merged in order whatever the window (the row_rule test).
//!
//! Audio rows: the state's `audio` lays the tower's embeddings (`enc.proj2`, WS-03's, or a fixture's) over the
//! prompt's `<|audio_pad|>` rows after the token embeddings (`decode/placeholders.zig`), so a prompt runs through
//! the decoder's windows as text does. The position is a device word (`State.position`) advanced by `commit`, so a
//! step's launches are the same at every position and `Graphs` captures them once.

const std = @import("std");
const Allocator = std.mem.Allocator;
const config_file = @import("config.zig");
const cuda = @import("../../cuda/cuda.zig");
const glue = @import("../../decode/glue.zig");
const graphs_file = @import("../../decode/graphs.zig");
const model_file = @import("model.zig");
const placeholders = @import("../../decode/placeholders.zig");

pub const Model = model_file.Model;
const Config = config_file.Config;

pub const Error = cuda.Error || error{ WindowTooWide, ContextFull };

/// Keys of a row that one chunk of the attention takes (common/attention.cu's CHUNK).
pub const chunk = 512;

/// From this many rows a window attends a group of rows a block, their chunks and merges in one launch
/// (common/attention_group.cu); fewer take the chunks and a merge (common/attention.cu). The same bits.
pub const window_rows = 64;

/// One sequence: its position, on the host and as the device word the kernels read, and the caches, a layer's
/// keys and values bf16 [capacity, kv_heads, head_dim].
pub const State = struct {
    pos: usize = 0,
    /// int32 [1]: `pos`, for the kernels
    position: cuda.Buffer,
    /// The prompt's audio rows and their embeddings, lent by the caller until `reset`.
    audio: ?placeholders.Placeholders = null,
    keys: []cuda.Buffer,
    values: []cuda.Buffer,
    capacity: usize,
    gpa: Allocator,

    pub fn init(gpa: Allocator, config: Config, capacity: usize) !State {
        var position: cuda.Buffer = try .init(.i32, &.{1});
        errdefer position.deinit();
        try position.zero();
        const keys = try gpa.alloc(cuda.Buffer, config.layers);
        errdefer gpa.free(keys);
        const values = try gpa.alloc(cuda.Buffer, config.layers);
        errdefer gpa.free(values);
        var made: usize = 0;
        errdefer for (keys[0..made], values[0..made]) |*k, *v| {
            k.deinit();
            v.deinit();
        };
        for (keys, values) |*k, *v| {
            k.* = try .init(.bf16, &.{ capacity, config.kv_heads, config.head_dim });
            v.* = cuda.Buffer.init(.bf16, &.{ capacity, config.kv_heads, config.head_dim }) catch |err| {
                k.deinit();
                return err;
            };
            made += 1;
        }
        return .{ .position = position, .keys = keys, .values = values, .capacity = capacity, .gpa = gpa };
    }

    pub fn deinit(self: *State) void {
        for (self.keys, self.values) |*k, *v| {
            k.deinit();
            v.deinit();
        }
        self.gpa.free(self.keys);
        self.gpa.free(self.values);
        self.position.deinit();
        self.* = undefined;
    }

    /// A new sequence in the same caches.
    pub fn reset(self: *State) Error!void {
        self.pos = 0;
        self.audio = null;
        try self.position.zero();
    }
};

/// What a forward writes and nothing keeps.
pub const Buffers = struct {
    rows: usize,
    scored: usize,
    ids: cuda.Buffer,
    /// int32 [rows, 2]: the window's audio rows and their embeddings' rows (`placeholders.within`)
    pairs: cuda.Buffer,
    /// bf16 [rows, hidden]: the residual stream
    h: cuda.Buffer,
    /// bf16 [rows, hidden]: a norm's output
    x: cuda.Buffer,
    qkv: cuda.Buffer,
    /// bf16 [rows, heads, head_dim]
    q: cuda.Buffer,
    attn: cuda.Buffer,
    /// fp32 [rows, hidden]: a branch's output before its add (`glue.residualAdd`'s one rank)
    branch: cuda.Buffer,
    /// bf16 [rows, 2 intermediate]: the MLP's gate and up; act [rows, intermediate]
    gu: cuda.Buffer,
    act: cuda.Buffer,
    /// fp32 [narrow, chunks, heads, head_dim], [narrow, chunks, heads] twice: a narrow window's attention chunks
    /// (fewer than `window_rows` rows)
    po: cuda.Buffer,
    pm: cuda.Buffer,
    pl: cuda.Buffer,
    /// bf16 [rows, hidden]: the final norm's rows
    final: cuda.Buffer,
    /// bf16 [scored, vocab]
    logits: cuda.Buffer,

    pub fn init(config: Config, rows: usize, scored: usize, capacity: usize) !Buffers {
        const c = config;
        const narrow = @min(rows, window_rows - 1);
        const chunks = (capacity + chunk - 1) / chunk;
        const names = .{ "ids", "pairs", "h", "x", "qkv", "q", "attn", "branch", "gu", "act", "po", "pm", "pl", "final", "logits" };
        const shapes = [names.len]struct { cuda.DType, []const usize }{
            .{ .i32, &.{rows} },
            .{ .i32, &.{ rows, 2 } },
            .{ .bf16, &.{ rows, c.hidden } },
            .{ .bf16, &.{ rows, c.hidden } },
            .{ .bf16, &.{ rows, c.qkvWidth() } },
            .{ .bf16, &.{ rows, c.heads, c.head_dim } },
            .{ .bf16, &.{ rows, c.heads * c.head_dim } },
            .{ .f32, &.{ rows, c.hidden } },
            .{ .bf16, &.{ rows, 2 * c.intermediate } },
            .{ .bf16, &.{ rows, c.intermediate } },
            .{ .f32, &.{ narrow, chunks, c.heads, c.head_dim } },
            .{ .f32, &.{ narrow, chunks, c.heads } },
            .{ .f32, &.{ narrow, chunks, c.heads } },
            .{ .bf16, &.{ rows, c.hidden } },
            .{ .bf16, &.{ scored, c.vocab } },
        };
        var self: Buffers = undefined;
        self.rows = rows;
        self.scored = scored;
        var made: usize = 0;
        errdefer inline for (names, 0..) |name, i| if (i < made) @field(self, name).deinit();
        inline for (names, shapes) |name, shape| {
            @field(self, name) = try .init(shape[0], shape[1]);
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *Buffers) void {
        inline for (.{ "ids", "pairs", "h", "x", "qkv", "q", "attn", "branch", "gu", "act", "po", "pm", "pl", "final", "logits" }) |name| @field(self, name).deinit();
        self.* = undefined;
    }

    pub fn scoredRows(self: Buffers) usize {
        return self.scored;
    }
};

/// The forward as captured graphs (`decode/graphs.zig`), a graph a window size, scoring and bucket of context:
/// the window's ids and its audio rows are staged eagerly (`embed`: host copies, which a capture cannot hold),
/// then the layers and the head run as one launch. The kernels and their arguments are `run`'s, so are the bits.
pub const Graphs = graphs_file.Graphs(Key, Graphed);

pub const Key = struct { rows: usize, context: usize };

/// The context a window ending at `end` launches its attention for: a power of two from `bucket_floor`, at most
/// the caches' capacity, so a reply crosses few of them. Chunks past a row's keys return at once, so a wider
/// bound costs a few idle blocks and changes no bit (common/attention.cu).
pub fn bucket(state: State, end: usize) usize {
    return @min(state.capacity, @max(bucket_floor, std.math.ceilPowerOfTwoAssert(usize, end)));
}

/// 4,096: a 140 s clip's reply (1,869 to 2,400 positions) takes one graph, a 600 s one's (7,862 to 10,000) two.
pub const bucket_floor = 4096;

const Graphed = struct {
    pub const Model = model_file.Model;
    pub const State = forward_file.State;
    pub const Buffers = forward_file.Buffers;
    pub const Error = forward_file.Error;

    pub fn enter(m: model_file.Model, state: *forward_file.State, b: *forward_file.Buffers, tokens: []const i32) forward_file.Error!Key {
        try fits(m, state.*, b.*, tokens.len, .all);
        try embed(m, state.*, b, tokens);
        return .{ .rows = tokens.len, .context = bucket(state.*, state.pos + tokens.len) };
    }

    pub fn compute(m: model_file.Model, state: *forward_file.State, b: *forward_file.Buffers, rows: usize, key: Key) forward_file.Error!void {
        return runWithin(m, state.*, b, rows, .all, key.context);
    }

    pub fn ahead(state: forward_file.State, b: forward_file.Buffers, rows: usize) ?Key {
        if (rows == 0 or rows > b.rows or rows > b.scored or state.pos + rows > state.capacity) return null;
        return .{ .rows = rows, .context = bucket(state, state.pos + rows) };
    }
};

const forward_file = @This();

pub const Score = enum { none, last, all };

/// `tokens` as one window after the state, every row scored.
pub fn forward(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
    return forwardScoring(m, state, b, tokens, .all);
}

pub fn forwardScoring(m: Model, state: *State, b: *Buffers, tokens: []const i32, score: Score) Error!void {
    try fits(m, state.*, b.*, tokens.len, score);
    try embed(m, state.*, b, tokens);
    return run(m, state.*, b, tokens.len, score);
}

/// The window's rows of `b.h`: the tokens' embeddings, the state's audio rows laid over those it covers.
pub fn embed(m: Model, state: State, b: *Buffers, tokens: []const i32) Error!void {
    const n = tokens.len;
    try b.ids.uploadAt(0, std.mem.sliceAsBytes(tokens));
    try cuda.launched(cuda.kernels.zs_embed(&b.ids.rows(n), &m.table(), &b.h.rows(n), 1));
    const audio = state.audio orelse return;
    if (!audio.touches(state.pos, n)) return;
    var pairs: [256][2]i32 = undefined;
    var start: usize = 0;
    while (start < n) : (start += pairs.len) {
        const count = @min(pairs.len, n - start);
        const at = audio.within(state.pos + start, count, &pairs);
        if (at == 0) continue;
        for (pairs[0..at]) |*pair| pair[0] += @intCast(start);
        try b.pairs.uploadAt(start * @sizeOf([2]i32), std.mem.sliceAsBytes(pairs[0..at]));
        try placeholders.embed(audio.features, b.pairs.span(start, at), b.h.rows(n), 1);
    }
}

pub fn fits(m: Model, state: State, b: Buffers, n: usize, score: Score) Error!void {
    if (n == 0 or n > b.rows or n > m.max_rows or (score == .all and n > b.scored)) return error.WindowTooWide;
    if (state.pos + n > state.capacity) return error.ContextFull;
}

/// Every layer over the window's `n` rows in `b.h`, then the final norm and the logits `score` names.
pub fn run(m: Model, state: State, b: *Buffers, n: usize, score: Score) Error!void {
    return runWithin(m, state, b, n, score, state.pos + n);
}

/// `run` with the attention's keys bounded by `context` (pos + n or more, within the caches): the same bits. A
/// layer's last add takes the next layer's input norm, the last layer's the final norm, in its launch.
pub fn runWithin(m: Model, state: State, b: *Buffers, n: usize, score: Score, context: usize) Error!void {
    return layers(m, .{ .one = .{ .state = state, .context = context } }, b, n, score);
}

/// One step of several sequences at once (spec 142 WS-06, a long clip's pieces): row j is `tokens[j]` after
/// `states[j]`, every row scored. The linears, norms and the head take the rows as one window; each row's
/// attention is its own sequence's launch, as its serial step's, so a row has the bits it has stepped alone
/// (the row rule). `commit` each state after.
pub fn steps(m: Model, states: []const *State, b: *Buffers, tokens: []const i32) Error!void {
    var many: Many = .{ .states = states };
    const rows = try Stepped.enter(m, &many, b, tokens);
    return Stepped.compute(m, &many, b, rows, rows);
}

/// The sequences `steps` steps, a row each in this order.
pub const Many = struct { states: []const *State };

/// `steps` as captured graphs, a graph a number of rows: always the same `Many` (its states' caches and
/// position words are what a graph holds). Each row attends its whole caches' capacity, whatever its position, so
/// a graph serves every step (no bit changes: chunks past a row's keys return at once).
pub const StepGraphs = graphs_file.Graphs(usize, Stepped);

const Stepped = struct {
    pub const Model = model_file.Model;
    pub const State = Many;
    pub const Buffers = forward_file.Buffers;
    pub const Error = forward_file.Error;

    pub fn enter(m: model_file.Model, many: *Many, b: *forward_file.Buffers, tokens: []const i32) forward_file.Error!usize {
        const n = tokens.len;
        std.debug.assert(many.states.len == n);
        if (n == 0 or n > b.rows or n > m.max_rows or n > b.scored) return error.WindowTooWide;
        for (many.states) |s| if (s.pos + 1 > s.capacity) return error.ContextFull;
        try b.ids.uploadAt(0, std.mem.sliceAsBytes(tokens));
        return n;
    }

    pub fn compute(m: model_file.Model, many: *Many, b: *forward_file.Buffers, rows: usize, _: usize) forward_file.Error!void {
        try cuda.launched(cuda.kernels.zs_embed(&b.ids.rows(rows), &m.table(), &b.h.rows(rows), 1));
        return layers(m, .{ .rows = many.states }, b, rows, .all);
    }

    pub fn ahead(many: Many, b: forward_file.Buffers, rows: usize) ?usize {
        if (rows != many.states.len or rows > b.scored) return null;
        return rows;
    }
};

/// Whose keys a window's rows attend: one sequence's, the window after it with its keys bounded by `context`
/// (pos + n or more, within the caches; no bit changes), or a sequence a row (`steps`).
const Sequences = union(enum) {
    one: struct { state: State, context: usize },
    rows: []const *State,
};

fn layers(m: Model, who: Sequences, b: *Buffers, n: usize, score: Score) Error!void {
    try glue.rmsnorm(b.h.rows(n), m.layers[0].input_norm.view, b.x.rows(n), m.config.eps);
    for (0..m.layers.len) |i| {
        const next: Next = if (i + 1 < m.layers.len) .{ .w = m.layers[i + 1].input_norm.view, .out = b.x } else .{ .w = m.norm.view, .out = b.final };
        try layerNormed(m, i, who, b, n, next);
    }
    try head(m, b, n, score);
}

/// The final norm into `b.final`, then the head: every row's logits, the last row's into row 0, or none.
pub fn finish(m: Model, b: *Buffers, n: usize, score: Score) Error!void {
    try glue.rmsnorm(b.h.rows(n), m.norm.view, b.final.rows(n), m.config.eps);
    return head(m, b, n, score);
}

/// The head over `b.final`'s rows as `score` names.
fn head(m: Model, b: *Buffers, n: usize, score: Score) Error!void {
    switch (score) {
        .none => {},
        .last => try m.head.forward(b.final.span(n - 1, 1), b.logits.rows(1)),
        .all => try m.head.forward(b.final.rows(n), b.logits.rows(n)),
    }
}

/// The norm a layer's last add leaves its rows' output of: the next layer's input norm into `b.x`, or the final
/// norm into `b.final`.
const Next = struct { w: cuda.View, out: cuda.Buffer };

/// Layer `i` over rows 0 .. n - 1 of `b.h`, at the positions `state.position` starts; `context` bounds the keys a
/// launch looks at (pos + n or more, within the caches), which changes no bit.
pub fn layer(m: Model, i: usize, state: State, b: *Buffers, n: usize, context: usize) Error!void {
    try glue.rmsnorm(b.h.rows(n), m.layers[i].input_norm.view, b.x.rows(n), m.config.eps);
    return layerNormed(m, i, .{ .one = .{ .state = state, .context = context } }, b, n, null);
}

/// `layer` with its input norm in `b.x` already, and its last add followed by `next`'s norm when given.
fn layerNormed(m: Model, i: usize, who: Sequences, b: *Buffers, n: usize, next: ?Next) Error!void {
    const c = m.config;
    const l = m.layers[i];
    const h = b.h.rows(n);
    const x = b.x.rows(n);
    try l.qkv.forward(x, b.qkv.rows(n));
    switch (who) {
        .one => |one| try attend(m, i, one.state, b, 0, n, one.context),
        .rows => |states| for (states, 0..) |s, j| try attend(m, i, s.*, b, j, 1, s.capacity),
    }
    try glue.residualNorm(h, try l.o.slices(b.attn.rows(n), b.branch.rows(n)), l.post_norm.view, x, c.eps);
    try l.gu.forward(x, b.gu.rows(n));
    try glue.swiglu(b.gu.rows(n), b.act.rows(n), std.math.inf(f32));
    if (next) |after| {
        try glue.residualNorm(h, try l.down.slices(b.act.rows(n), b.branch.rows(n)), after.w, after.out.rows(n), c.eps);
    } else {
        try l.down.forward(b.act.rows(n), b.branch.rows(n));
        try glue.residualAdd(h, gathered(b.branch, n, c.hidden));
    }
}

/// Layer `i`'s attention of the window's rows `first .. first + n - 1` after `state`: q and k normed and turned,
/// k and v into its caches, then attended over its keys into the same rows of `b.attn`.
fn attend(m: Model, i: usize, state: State, b: *Buffers, first: usize, n: usize, context: usize) Error!void {
    const c = m.config;
    const l = m.layers[i];
    const kc = state.keys[i].view;
    const vc = state.values[i].view;
    const pos = state.position.view;
    const qkv = b.qkv.span(first, n);
    const q = b.q.span(first, n);
    try cuda.launched(cuda.kernels.zs_attn_prep(&qkv, &l.q_norm.view, &l.k_norm.view, &m.inv.view, &pos, &q, &kc, &vc, c.eps));
    const out: cuda.View = .init(b.attn.span(first, n).ptr.?, .bf16, &.{ n, c.heads, c.head_dim });
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim)));
    if (n >= window_rows) {
        try cuda.launched(cuda.kernels.zs_attn_attend_group(&q, &kc, &vc, &pos, &out, scale, @intCast(context)));
    } else {
        try cuda.launched(cuda.kernels.zs_attn_attend(&q, &kc, &vc, &pos, &b.po.view, &b.pm.view, &b.pl.view, &out, null, null, null, null, null, null, scale, @intCast(context)));
    }
}

/// `branch`'s first `n` rows as one rank's partials, fp32 [1, n, hidden].
fn gathered(branch: cuda.Buffer, n: usize, hidden: usize) cuda.View {
    return .init(branch.view.ptr.?, .f32, &.{ 1, n, hidden });
}

/// Makes the window's first `keep` rows part of the sequence: their keys and values are in the caches already.
pub fn commit(_: Model, state: *State, _: Buffers, tokens: []const i32, keep: usize) Error!void {
    std.debug.assert(keep >= 1 and keep <= tokens.len);
    state.pos += keep;
    const word: [1]i32 = .{@intCast(state.pos)};
    try state.position.uploadLaunched(0, std.mem.sliceAsBytes(&word));
}

/// Row `row`'s logits, bf16 bits a token, after the device's launches.
pub fn logitsRow(b: Buffers, row: usize, out: []u16) Error!void {
    return cuda.download(b.logits.span(row, 1), std.mem.sliceAsBytes(out));
}

/// Rows `row` to `row + n` of the window's logits, bf16 [n, vocab].
pub fn logitRows(b: Buffers, row: usize, n: usize) cuda.View {
    return b.logits.span(row, n);
}

/// The window's first `rows` final rows.
pub fn streams(b: Buffers, rows: usize) cuda.View {
    return b.final.rows(rows);
}
