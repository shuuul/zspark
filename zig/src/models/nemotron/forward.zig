//! The backbone's decode-path forward (`cuda_engine.zig`, `cuda_forward.zig`): `tokens` as rows at positions
//! `state.pos` onward, the layers `layers_block_type` gives in order — each a residual RMSNorm, then its
//! mixer (the Mamba-2 conv and scan of `mamba.zig`, the NoPE attention of `attention.zig`, the routed and
//! shared experts of `moe.zig`), its product folded back into the residual — then the final norm and
//! `lm_head`'s logits for the rows `score` names. A window changes nothing a later forward from the same
//! state reads: `commit` advances it, and a verify window's rejected rows leave the states where the kept
//! rows took them (each layer's own rules: the Mamba meta's kept rows replay before the next window; a
//! rejected row's cache rows are never read).
//!
//! `forward`'s contract with `decode/draft.zig` is `namespace`'s below; `Graphs` is its captured form
//! (`decode/graphs.zig`), a window's launches as one launch by `(rows, context bucket)`.

const std = @import("std");
const cuda = @import("../../cuda/cuda.zig");
const placeholders = @import("../../decode/placeholders.zig");
const mamba_file = @import("mamba.zig");
const attention_file = @import("attention.zig");
const moe_file = @import("moe.zig");
const model_file = @import("model.zig");
const state_file = @import("state.zig");
const glue = @import("glue.zig");
const graphs_file = @import("../../decode/graphs.zig");

pub const Error = error{ WindowTooWide, ContextFull, NoMtpHead } || cuda.Error;
pub const Model = model_file.Model;
pub const State = state_file.State;
pub const Buffers = state_file.Buffers;
/// Which rows of a window the head scores. A row's logits are the same bits whichever is asked.
pub const Score = enum { none, last, all };

const forward_file = @This();

/// `tokens` as one window after the state, every row scored (a verify window's; a serial step is `tokens`
/// of one row).
pub fn forward(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
    return forwardScoring(m, state, b, tokens, .all);
}

/// `forward` with `score` naming which rows the head scores: `.all` into `b.logits` in the same row order,
/// `.last` the last row's alone into `b.logits[0]`, `.none` no head launch at all (a prompt's earlier
/// windows).
pub fn forwardScoring(m: Model, state: *State, b: *Buffers, tokens: []const i32, score: Score) Error!void {
    const rows = tokens.len;
    if (rows == 0 or rows > b.rows or rows > m.max_rows or (score == .all and rows > b.scored)) return error.WindowTooWide;
    if (state.pos + rows > state.capacity) return error.ContextFull;
    try embedRows(m, state, b, tokens);
    return run(m, state, b, rows, score, state.pos + rows, false);
}

/// The window's rows of `b.h`: the tokens' embeddings, the state's audio rows laid over those it covers
/// (`Graphed.enter` runs the same eagerly — a graph cannot hold the upload or know the position).
fn embedRows(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
    const rows = tokens.len;
    try b.ids.uploadAt(0, std.mem.sliceAsBytes(tokens));
    try cuda.launched(cuda.kernels.zs_embed(&b.ids.rows(rows), &m.embed.view, &b.h.rows(rows), 1));
    try audioRows(state, b, rows);
}

/// The state's audio rows laid over the embeddings this window covers (`decode/placeholders.zig`), after
/// `zs_embed`: the pairs are staged in `b.pairs` before `placeholders.embed` launches (eager host copies —
/// `Graphed` runs it the same way outside what a graph captures, `qwen3_asr`'s pattern).
fn audioRows(state: *State, b: *Buffers, rows: usize) Error!void {
    const audio = state.audio orelse return;
    if (!audio.touches(state.pos, rows)) return;
    var pairs: [256][2]i32 = undefined;
    var start: usize = 0;
    while (start < rows) : (start += pairs.len) {
        const count = @min(pairs.len, rows - start);
        const at = audio.within(state.pos + start, count, &pairs);
        if (at == 0) continue;
        for (pairs[0..at]) |*pair| pair[0] += @intCast(start);
        try b.pairs.uploadAt(start * @sizeOf([2]i32), std.mem.sliceAsBytes(pairs[0..at]));
        try placeholders.embed(audio.features, b.pairs.span(start, at), b.h.rows(rows), 1);
    }
}

/// The residual delta a block's mixer leaves for the next norm.
const Delta = union(enum) {
    none,
    /// A dense row's product (o_proj / out_proj) to add, bf16 [rows, hidden].
    dense: cuda.View,
    /// A MoE layer's slot rows and weights still in `b.moe_sc` (`addMoeNorm` folds them): the layer's own.
    moe: moe_file.Moe,
};

/// Whether `rows` runs as a prompt chunk (above the window rows) or a window; the mixers take the same
/// rows either way, the mamba's prompts the wide kernels of one chunk.
fn wide(rows: usize) bool {
    return rows > mamba_file.window_rows;
}

/// h = x + delta, the normed rows in `b.y` (x folded into `b.hn` where delta exists): `.moe`'s slots and
/// weights fold into the residual the same way (`addMoeNorm`), the layer's own `Moe` held in `delta`.
fn foldDelta(m: Model, x: cuda.View, delta: Delta, w: cuda.View, b: *Buffers, rows: usize) Error!cuda.View {
    const c = m.config.backbone;
    const hn = b.hn.rows(rows);
    const y = b.y.rows(rows);
    const xs = b.xs.rows(rows);
    switch (delta) {
        .none => try glue.addRmsnorm(x, null, w, hn, y, xs, c.eps),
        .dense => |d| try glue.addRmsnorm(x, d, w, hn, y, xs, c.eps),
        .moe => |one| {
            const s = one.shape;
            const yk: cuda.View = .{ .ptr = b.moe_sc.y.view.ptr, .dtype = .f32, .ndim = 3, .size = .{ @intCast(rows), @intCast(s.slots()), @intCast(s.dims), 0, 0, 0 } };
            try glue.addMoeNorm(x, yk, b.moe_sc.wts.rows(rows), w, hn, y, xs, c.eps, s.top);
        },
    }
    return if (delta == .none) x else hn;
}

/// The slot of `layer` among `layers`, or null (an off-by-one is not a tap).
fn tapIndexOf(layers: []const u32, layer: usize) ?usize {
    for (layers, 0..) |one, i| if (one == layer) return i;
    return null;
}

/// The window's layers over `b.h`'s `rows` rows, then the final norm and the head `score` names. `context`
/// bounds the attention's launched chunks (pos + rows or more, within the caches): a launch bound alone.
/// `staged` says the meta word is already on the device for this window (`Graphed.enter` uploads it
/// eagerly, before the capture): the per-layer `.set` then only runs eagerly, since an `uploadLaunched`
/// inside a capture would replay the capture-time pos/parity/prev_keep on every replay.
fn run(m: Model, state: *State, b: *Buffers, rows: usize, score: Score, context: usize, staged: bool) Error!void {
    var x = b.h.rows(rows); // the residual: embeddings, then each block's h
    var delta: Delta = .none;
    var mi: usize = 0; // mamba blocks so far
    var ai: usize = 0; // attention blocks so far
    const y = b.y.rows(rows);
    for (m.blocks, 0..) |*blk, index| {
        x = try foldDelta(m, x, delta, blk.norm(), b, rows);
        // the residual `foldDelta` left is the previous block's completed output: the tap of layer
        // `index - 1` (vLLM `nemotron_h.py`: `target_layer_ids` + 1 gives `aux_hidden_state_layers`,
        // captured as `hidden_states + residual` after that decoder layer — `interfaces.py`'s
        // `_maybe_add_hidden_state`, called with `idx + 1` after layer `idx` runs). `b.hn` holds it
        // where a delta folded in.
        if (index > 0) {
            if (tapIndexOf(b.tap_layers, index - 1)) |slot| {
                const stream = if (delta == .none) x else b.hn.rows(rows);
                try b.taps.?.copyColumns(slot * @as(usize, @intCast(stream.size[1])) * 2, stream, rows);
            }
        }
        switch (blk.*) {
            .mamba => |*one| {
                try one.in_proj.forward(y, b.proj.rows(rows));
                // the layer's meta row uploaded from its host copy, which `commit` then advances; under a
                // graph `Graphed.enter` staged it eagerly — all layers' host metas advance identically, so
                // `b.meta`'s one word serves them all.
                if (!staged) try state.mamba_meta[mi].set(b.meta);
                if (wide(rows))
                    try one.mixer.prompt(state.mamba[mi], b.mamba_sc, b.proj.rows(rows), b.meta.view, b.gate.rows(rows))
                else
                    try one.mixer.window(state.mamba[mi], b.mamba_sc, b.proj.rows(rows), b.meta.view, b.gate.rows(rows));
                try one.out_proj.forward(b.gate.rows(rows), b.delta.rows(rows));
                delta = .{ .dense = b.delta.rows(rows) };
                mi += 1;
            },
            .attention => |*one| {
                // `window`/`prompt` run the layer whole: qkv, the K/V writes at `pos`, the attend and o_proj
                // into `att` (bf16 [rows, hidden]); the row is the residual's delta.
                if (wide(rows))
                    try one.mixer.prompt(state.attention[ai], b.attn_sc, y, state.position.view, context, b.att.rows(rows))
                else
                    try one.mixer.window(state.attention[ai], b.attn_sc, y, state.position.view, context, b.att.rows(rows));
                delta = .{ .dense = b.att.rows(rows) };
                ai += 1;
            },
            .moe => |*one| {
                if (wide(rows))
                    try moe_file.prompt(one.mixer, b.moe_sc, y)
                else
                    try moe_file.window(one.mixer, b.moe_sc, y);
                delta = .{ .moe = one.mixer };
            },
        }
    }
    // the final norm folds the last delta in — `b.hn` is then the last block's completed output, the tap
    // of the last layer (51 on the published drafter) where its id is asked — then the head scores what
    // `score` names of the normed rows
    _ = try foldDelta(m, x, delta, m.norm_f.view, b, rows);
    if (tapIndexOf(b.tap_layers, m.blocks.len - 1)) |slot| {
        const stream = if (delta == .none) x else b.hn.rows(rows);
        try b.taps.?.copyColumns(slot * @as(usize, @intCast(stream.size[1])) * 2, stream, rows);
    }
    switch (score) {
        .none => {},
        .all => try m.head.forward(y, b.logits.rows(rows)),
        .last => try m.head.forward(y.span(rows - 1, 1), b.logits.rows(1)),
    }
}

/// Keeps the first `keep` rows of the window `tokens` the last forward ran: each mamba layer's meta advances
/// by `afterWindow` (`afterPrompt` for a whole prompt chunk), `pos` and the position word by `keep`.
/// Rejected rows leave no state: a mamba layer's kept rows stay pending in its parity buffers for the next
/// window's replay; the attention caches' rows past `pos` are never read.
pub fn commit(m: Model, state: *State, b: Buffers, tokens: []const i32, keep: usize) cuda.Error!void {
    _ = m;
    _ = b;
    const rows = tokens.len;
    std.debug.assert(keep >= 1 and keep <= rows);
    if (wide(rows)) {
        for (state.mamba_meta) |*meta| meta.* = meta.afterPrompt(@intCast(rows));
        state.pos += rows;
    } else {
        for (state.mamba_meta) |*meta| meta.* = meta.afterWindow(keep);
        state.pos += keep;
    }
    const word: [1]i32 = .{@intCast(state.pos)};
    return state.position.upload(std.mem.sliceAsBytes(&word));
}

/// A row of logits copied to the host (a row's bits are a serial step's), the device waited for.
pub fn logitsRow(b: Buffers, row: usize, out: []u16) cuda.Error!void {
    try cuda.synchronize();
    return cuda.download(b.logits.span(row, 1), std.mem.sliceAsBytes(out));
}

/// The window's final residual streams: `b.hn`'s first `rows` rows, the residual after the last layer's
/// fold (what the MTP head's `hin` reads). Valid until the next forward.
pub fn streams(b: Buffers, rows: usize) cuda.View {
    return b.hn.rows(rows);
}

/// Rows `row` to `row + n` of the window's logits, bf16 [n, vocab].
pub fn logitRows(b: Buffers, row: usize, n: usize) cuda.View {
    return b.logits.span(row, n);
}

/// The forward as captured graphs (`decode/graphs.zig`), a graph a window size and context bucket: the
/// window's ids are staged eagerly (`uploadAt`, which a capture cannot hold), then the layers and the head
/// run as one launch. A row's keys are bounded by its own position, so a bucket's wider `context` costs
/// idle chunks and changes no bit (`common/attention.cu`).
pub const Graphs = graphs_file.Graphs(Key, Graphed);

/// A graph's key: the window's rows and the context bucket its attention launches bound at.
pub const Key = struct { rows: usize, context: usize };

/// The context a window ending at `end` launches its attention for: a power of two from `bucket_floor`, at
/// most the caches' capacity, so a reply crosses few of them.
pub fn bucket(state: State, end: usize) usize {
    return @min(state.capacity, @max(bucket_floor, std.math.ceilPowerOfTwoAssert(usize, end)));
}

/// 4,096: the bucket below which a reply's windows share one graph.
pub const bucket_floor = 4096;

const Graphed = struct {
    pub const Model = model_file.Model;
    pub const State = forward_file.State;
    pub const Buffers = forward_file.Buffers;
    pub const Error = forward_file.Error;

    pub fn enter(m: model_file.Model, state: *forward_file.State, b: *forward_file.Buffers, tokens: []const i32) forward_file.Error!Key {
        const rows = tokens.len;
        if (rows == 0 or rows > b.rows or rows > m.max_rows or rows > b.scored) return error.WindowTooWide;
        if (state.pos + rows > state.capacity) return error.ContextFull;
        // ids, embeddings and the audio rows staged eagerly: host copies and the position-dependent pairs
        // a capture cannot hold (qwen3_asr's `embed`); the mamba meta word too — captured, an
        // `uploadLaunched` would replay the capture-time pos/parity/prev_keep on every replay (the
        // verify-window drift that broke the drafter smoke).
        if (state.mamba_meta.len > 0) try state.mamba_meta[0].set(b.meta);
        try embedRows(m, state, b, tokens);
        return .{ .rows = rows, .context = bucket(state.*, state.pos + rows) };
    }

    pub fn compute(m: model_file.Model, state: *forward_file.State, b: *forward_file.Buffers, rows: usize, key: Key) forward_file.Error!void {
        return run(m, state, b, rows, .all, key.context, true);
    }

    pub fn ahead(state: forward_file.State, b: forward_file.Buffers, rows: usize) ?Key {
        if (rows == 0 or rows > b.rows or rows > b.scored or state.pos + rows > state.capacity) return null;
        return .{ .rows = rows, .context = bucket(state, state.pos + rows) };
    }
};
