//! GLM-5.3-Flash's forward on a rank, in the phases between which the ranks exchange their partials:
//! `loopback.zig` runs them for both ranks in one process, spec 045 for a rank a process. TensorFold's
//! `forward.py` (`layer_forward`, `compute`).
//!
//! A layer is two sites. At each the four streams are read out into one row (`hc.mix`, `hc.pre`), the branch
//! (KDA or DSA, then the MLP) gives the rank's partial in fp32 (bf16 in FP4-Local's quantized windows,
//! `partials`), the ranks' partials are gathered, and every
//! rank writes their sum back into its own streams (`hc.post`), rank 0 first: the streams are the same bits on
//! both ranks, and so is all that is read from them. A prompt's window on two ranks splits that glue by rows
//! (`ranks_forward.zig` `splits`): each rank writes back and reads out its own rows, the same bits.
//!
//! ```mermaid
//! flowchart LR
//!   IDS["ids"] -- "embed" --> X["x: 4 streams"]
//!   X -- "attention: read-out, KDA or DSA" --> PA["partial"]
//!   PA -- "gathered" --> MLP["mlp: write-back, read-out, dense or MoE"]
//!   MLP -- "partial, gathered" --> FIN["finish: write-back"]
//!   FIN -- "next layer" --> X
//!   FIN -- "head: mean, norm, the rank's rows" --> LOGITS["logits: vocab / 2"]
//! ```

const std = @import("std");
const cuda = @import("../../../cuda/cuda.zig");
const dsa = @import("dsa.zig");
const forms = @import("../../../decode/forms.zig");
const hc = @import("hc.zig");
const kda = @import("kda.zig");
const model = @import("model.zig");
const nccl = @import("../../../ranks/nccl.zig");

pub const Error = cuda.Error || nccl.Error;
pub const Model = model.Model;
pub const State = model.State;
pub const Buffers = model.Buffers;

/// The card's forms (spec 115; FP4-Local's recommended vLLM launch, read in its source), by `decode/forms.zig`'s
/// rule: a window of more than this many rows is a prompt's, and its rows take the quantized forms (routed experts
/// W4A4 as the card's A4 prompt path, MXFP8 linears on FP8 rows); a step, a verify window and a prompt window of up
/// to it take bf16 rows (experts A16, MXFP8 linears on bf16 rows), as the card runs the batches its CUDA graphs hold
/// (up to 24 tokens with MTP: their MoE was captured A16) and the MXFP8 plans it may tune A16 (its capture sizes, up
/// to 48). Block-FP8 linears quantize in every form, as the card's. A test lowers it, never under
/// `forward_max_rows`, or raises it past its windows (`testing.oneForm`: one form).
pub var bf16_rows: usize = card_bf16_rows;
pub const card_bf16_rows = 48;
comptime {
    forms.checkBound(card_bf16_rows);
}

/// The first row of a window of `rows` rows that takes the quantized forms: 0, or `rows` where none does.
pub fn quantizedFrom(rows: usize) usize {
    return forms.quantizedFrom(rows, bf16_rows);
}

/// The rows of `m`'s linears' calls from `from` on take the quantized forms until `clearForm` (the pool's
/// `quantize_from`, which the MoE reads too).
pub fn setForm(m: Model, from: usize) void {
    m.pool.quantize_from = from;
}

/// The partials' dtype of a window whose quantized rows start at `from` (spec 116 WS-04): bf16 where every row is
/// quantized and the model's forms are the card's (`Model.bf16_partials`), as the card's vLLM reduces its partials
/// in every phase; else fp32, more than vLLM's. A round of both forms keeps fp32, its quantized rows rounded to bf16
/// by their writer (`roundQuantized`), so that each row's bits are its form's. The MTP layer's and the drafters'
/// partials are fp32: they do not ask.
pub fn partials(m: Model, from: usize) cuda.DType {
    return if (m.bf16_partials and from == 0) .bf16 else .f32;
}

/// The rows of [first, first + count) of an fp32 `partial` that are the quantized form's in a round of both forms,
/// rounded to bf16 in place: a bf16 partial's values (every writer rounds its fp32 sum once, `partials`).
fn roundQuantized(m: Model, partial: cuda.View, first: usize, count: usize) Error!void {
    const from = m.pool.quantize_from;
    if (!m.bf16_partials or partial.dtype != .f32 or from == 0 or from >= first + count) return;
    const lo = @max(first, from);
    return hc.roundBf16(partial.span(lo, first + count - lo));
}

/// The pool's form for a call on the window's rows from `first` alone, which reads `quantize_from` from its own
/// first row (spec 116 WS-03's parts); returns the window's, which the caller puts back.
fn partForm(m: Model, first: usize, count: usize) usize {
    const was = m.pool.quantize_from;
    m.pool.quantize_from = if (was <= first) 0 else @min(was - first, count);
    return was;
}

/// Back to every row quantized, what the MTP layer and the drafters run (spec 115 leaves their forms).
pub fn clearForm(m: Model) void {
    m.pool.quantize_from = 0;
}

/// The window's ids into `b`, before any phase.
pub fn stageIds(b: *Buffers, tokens: []const i32) Error!void {
    try cuda.bound(tokens.len >= 1 and tokens.len <= b.rows, "glm5 stageIds: {d} tokens into buffers of {d} rows", .{ tokens.len, b.rows });
    return b.ids.uploadLaunched(0, std.mem.sliceAsBytes(tokens));
}

/// The ids' rows into each stream.
pub fn embed(m: Model, b: *Buffers, rows: usize) Error!void {
    return hc.embed(b.ids.rows(rows), m.embed.view, b.x.rows(rows));
}

/// A site's read-out: `b.normed`, with the gates and the mix its write-back takes.
pub fn readOut(m: Model, b: *Buffers, rows: usize, site: model.Site) Error!void {
    const x = b.x.rows(rows);
    const part = b.mix.rows(rows);
    try hc.mix(x, site.dots.view, part);
    try hc.pre(x, part, site.views(), b.normed.rows(rows), b.post.rows(rows), b.comb.rows(rows), m.eps, m.hc_eps, m.sinkhorn);
}

/// `readOut` of the window's rows [first, first + count) alone (the split glue, spec 107 WS-03): every kernel of
/// it works a row at a time, so these rows' bits are the whole window's.
pub fn readOutRows(m: Model, b: *Buffers, first: usize, count: usize, site: model.Site) Error!void {
    const x = b.x.span(first, count);
    const part = b.mix.span(first, count);
    try hc.mix(x, site.dots.view, part);
    try hc.pre(x, part, site.views(), b.normed.span(first, count), b.post.span(first, count), b.comb.span(first, count), m.eps, m.hc_eps, m.sinkhorn);
}

/// `finish` of the rows [first, first + count): `gathered` fp32 or bf16 [world, rows, dims] is the whole window's.
pub fn finishRows(b: *Buffers, first: usize, count: usize, gathered: cuda.View) Error!void {
    const rows: usize = @intCast(gathered.size[1]);
    const dims: usize = @intCast(gathered.size[2]);
    const base: [*]u8 = @ptrCast(gathered.ptr.?);
    // the rows from `first` of each rank's partial, a rank's stride still the window's
    const from: cuda.View = .init(@ptrCast(base + first * dims * gathered.dtype.size()), gathered.dtype, &.{ @intCast(gathered.size[0]), rows, dims });
    return hc.post(b.x.span(first, count), from, b.post.span(first, count), b.comb.span(first, count));
}

/// Layer `index`'s attention over the window at the sequence's position: `partial` [rows, dims] (`partials`), the
/// rank's heads. `context` is what the DSA's launch is sized for, from the window's end to the capacity: it
/// changes no bit, and a graph is bound to it.
pub fn attention(m: Model, state: *State, b: *Buffers, index: usize, rows: usize, context: usize, partial: cuda.View, split: ?nccl.Comm) Error!void {
    try readOut(m, b, rows, m.layers[index].attn_site);
    return attend(m, state, b, index, rows, context, partial, split);
}

/// `attention` after its read-out, from the rows in `b.normed`: `compare.zig` puts TensorFold's there.
pub fn attend(m: Model, state: *State, b: *Buffers, index: usize, rows: usize, context: usize, partial: cuda.View, split: ?nccl.Comm) Error!void {
    const layer = m.layers[index];
    const x = b.normed.rows(rows);
    switch (layer.attention) {
        .kda => |weights| try state.kda.?.forward(weights, layer.slot, state.pos, x, partial),
        .dsa => |weights| try weights.forwardAtSplit(state.dsa.?, layer.slot, b.dsa.?, x, partial, context, split),
    }
    return roundQuantized(m, partial, 0, rows);
}

/// `attend` but its out-projection, which `emitRows` makes a part of the rows at a time (spec 116 WS-03).
pub fn attendBody(m: Model, state: *State, b: *Buffers, index: usize, rows: usize, context: usize, split: ?nccl.Comm) Error!void {
    const layer = m.layers[index];
    const x = b.normed.rows(rows);
    switch (layer.attention) {
        .kda => |weights| try state.kda.?.forwardChain(weights, layer.slot, state.pos, x),
        .dsa => |weights| try weights.attendAtSplit(state.dsa.?, layer.slot, b.dsa.?, x, context, split),
    }
}

/// The out-projection of layer `index`'s attention on the window's rows [first, first + count) alone, after
/// `attendBody` or `attendOfBody` over all of them, into those rows of `partial` [rows, dims] (`partials`): a row's bits
/// are the whole's. `chains` holds a KDA layer's chained rows: the sequence's state, or the window's.
pub fn emitRows(m: Model, chains: ?kda.State, b: *Buffers, index: usize, first: usize, count: usize, partial: cuda.View) Error!void {
    {
        const was = partForm(m, first, count);
        defer m.pool.quantize_from = was;
        switch (m.layers[index].attention) {
            .kda => |weights| try chains.?.emitRows(weights, first, count, partial),
            .dsa => |weights| try weights.emitRows(b.dsa.?, first, count, partial),
        }
    }
    return roundQuantized(m, partial, first, count);
}

/// One sequence's rows in a window that several share: rows `first` .. `first + rows - 1`, of `seqs[slot]`.
pub const Span = struct { slot: usize, first: usize, rows: usize, prompt: bool = false };

/// `attention` for a window of `rows` rows that `spans` cut into sequences, each at its own position. What a
/// row shares with no row before it (the projections, the absorbed queries, the outputs) is one launch over
/// the window; the chain of a KDA layer, and a DSA layer's latents and attention, one a sequence. A row's bits
/// are those of its sequence alone.
pub fn attentionOf(m: Model, in: ?kda.State, seqs: []const model.Seq, spans: []const Span, b: *Buffers, index: usize, rows: usize, partial: cuda.View, split: ?nccl.Comm) Error!void {
    try readOut(m, b, rows, m.layers[index].attn_site);
    return attendOf(m, in, seqs, spans, b, index, rows, partial, split);
}

/// `attentionOf` after its read-out, as `attend`.
pub fn attendOf(m: Model, in: ?kda.State, seqs: []const model.Seq, spans: []const Span, b: *Buffers, index: usize, rows: usize, partial: cuda.View, split: ?nccl.Comm) Error!void {
    try attendOfBody(m, in, seqs, spans, b, index, rows, split);
    return emitRows(m, in, b, index, 0, rows, partial);
}

/// `attendOf` but its out-projection (`emitRows`), as `attendBody`.
pub fn attendOfBody(m: Model, in: ?kda.State, seqs: []const model.Seq, spans: []const Span, b: *Buffers, index: usize, rows: usize, split: ?nccl.Comm) Error!void {
    const layer = m.layers[index];
    const x = b.normed.rows(rows);
    switch (layer.attention) {
        .kda => |weights| {
            const chains = in.?;
            try chains.project(weights, layer.slot, x);
            for (spans) |span| try chains.chainKept(weights, layer.slot, seqs[span.slot].kda.?, span.first, span.rows, seqs[span.slot].pos);
        },
        .dsa => |weights| {
            const scratch = b.dsa.?;
            try weights.project(scratch, x);
            try weights.indexProject(scratch, x);
            for (spans) |span| {
                const cache = seqs[span.slot].dsa.?;
                try weights.write(cache, layer.slot, scratch, span.first, span.rows, cache.pos + span.rows);
                // a prompt part's rows at or past `index.split_from` split over the ranks; decode's never do
                const comm = if (span.prompt) split else null;
                try weights.indexRowsSplit(cache, layer.slot, scratch, span.first, span.rows, cache.pos + span.rows, comm);
            }
            try dsa.absorb(scratch.q.rows(rows), weights.kv_b.view, weights.key_dim, scratch.qa.rows(rows));
            for (spans) |span| {
                const cache = seqs[span.slot].dsa.?;
                try weights.attend(cache, layer.slot, scratch, span.first, span.rows, cache.pos + span.rows);
            }
        },
    }
}

/// The attention's write-back from `gathered` fp32 [world, rows or more, dims], then layer `index`'s MLP:
/// `partial` [rows, dims] (`partials`), the rank's half of the inner width.
pub fn mlp(m: Model, b: *Buffers, index: usize, rows: usize, gathered: cuda.View, partial: cuda.View) Error!void {
    try finish(b, rows, gathered);
    try readOut(m, b, rows, m.layers[index].ffn_site);
    return feed(m, b, index, rows, partial);
}

/// `mlp` after its read-out, as `attend`.
pub fn feed(m: Model, b: *Buffers, index: usize, rows: usize, partial: cuda.View) Error!void {
    const x = b.normed.rows(rows);
    switch (m.layers[index].mlp) {
        .dense => |weights| try weights.run(b.dense.?, x, partial),
        .moe => |weights| try weights.forward(b.moe.?, x, partial, m.pool.quantize_from),
    }
    return roundQuantized(m, partial, 0, rows);
}

/// `feed` up to what `fillRows` makes a part of the rows at a time (spec 116 WS-03): a MoE's router and routed
/// experts over every row; nothing of a dense MLP.
pub fn feedBody(m: Model, b: *Buffers, index: usize, rows: usize) Error!void {
    switch (m.layers[index].mlp) {
        .dense => {},
        .moe => |weights| try weights.route(b.moe.?, b.normed.rows(rows), m.pool.quantize_from),
    }
}

/// `feed`'s rest on the window's rows [first, first + count) alone, after `feedBody`, into those rows of
/// `partial` [rows, dims] (`partials`): a dense MLP whole, a MoE's shared expert and sum. A row's bits are the whole's.
pub fn fillRows(m: Model, b: *Buffers, index: usize, rows: usize, first: usize, count: usize, partial: cuda.View) Error!void {
    const x = b.normed.rows(rows);
    {
        const was = partForm(m, first, count);
        defer m.pool.quantize_from = was;
        switch (m.layers[index].mlp) {
            .dense => |weights| try weights.run(b.dense.?, x.span(first, count), partial.span(first, count)),
            .moe => |weights| try weights.fill(b.moe.?, x, first, count, partial),
        }
    }
    return roundQuantized(m, partial, first, count);
}

/// A branch's write-back; the MLP's ends the layer.
pub fn finish(b: *Buffers, rows: usize, gathered: cuda.View) Error!void {
    return hc.post(b.x.rows(rows), gathered, b.post.rows(rows), b.comb.rows(rows));
}

/// After layer `index`: the drafter's tap of it, where it has one.
pub fn tap(b: *Buffers, index: usize, rows: usize) Error!void {
    for (b.tap_layers, 0..) |layer, slot| {
        if (layer == index) try hc.streamMean(b.x.rows(rows), b.tap(slot, rows));
    }
}

/// Which rows of a window the head scores. A row's logits are the same bits whichever is asked.
pub const Score = enum {
    /// Every row, into the same row of `b.logits`.
    all,
    /// The last row alone, into `b.logits`' first row.
    last,
    none,
};

/// After the last layer: the streams' mean (`b.hidden`) and its norm (`b.final`) for every row, which the MTP
/// layer reads of a prompt's rows too, and the rank's rows of the vocabulary (`b.logits`) for those `score` names.
/// The head's rows take the window's form where it scores them all, and bf16 rows where it scores the last alone
/// (the card's head takes the rows it samples, a prompt's last: one row, an A16 plan).
pub fn head(m: Model, b: *Buffers, rows: usize, score: Score) Error!void {
    const hidden = b.hidden.rows(rows);
    const final = b.final.rows(rows);
    try hc.streamMean(b.x.rows(rows), hidden);
    try hc.rmsnorm(hidden, m.norm.view, final, m.eps);
    switch (score) {
        .all => try m.head.forward(final, b.logits.rows(rows)),
        .last => try headPlain(m, b.final.span(rows - 1, 1), b.logits.rows(1)),
        .none => {},
    }
}

/// The head on bf16 rows whatever the window's form: the rows a round samples (spec 115).
pub fn headPlain(m: Model, x: cuda.View, logits: cuda.View) Error!void {
    const was = m.pool.quantize_from;
    m.pool.quantize_from = std.math.maxInt(usize);
    defer m.pool.quantize_from = was;
    return m.head.forward(x, logits);
}

/// The context a window ending at `end` is launched for: a power of two from 8192, so a sequence crosses few
/// of them, and at most the capacity.
pub fn bucket(capacity: usize, end: usize) usize {
    return @min(capacity, @max(8192, std.math.ceilPowerOfTwoAssert(usize, end)));
}
