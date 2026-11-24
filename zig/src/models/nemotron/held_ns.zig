//! Nemotron-3.5-Transcribe's one-held-rank namespace for `decode/draft.zig`'s `Decoder` (spec 151
//! WS-06/WS-08): a process's share of the two-rank split, its `Ranks` gather meeting each block's fp32
//! partial over NCCL. `forward`/`commit`/`logitsRow` run `ranks_forward`'s on this rank's own model,
//! state and buffers. Rank 0 leads: each operation is told to rank 1 over the control channel first
//! (`ranks_forward.Op`, `lead.zig`'s frames), and the follower replays it in `mirror.zig` — the reply's
//! choices, sampler, stop strings and exits all live on rank 0 alone, so the two ranks stay inside the
//! same collective sequence and there is no peer state to keep in step.
//!
//! `logitRows`/`streams` give the drafter this rank's own buffers: `logitsRow`'s gather writes the
//! whole-vocabulary rows beside the halves, so `logprobs` reads the gathered copy — the namespace's
//! `logitRows` view is over `Ranks.row`'s gathered face.

const std = @import("std");
const cuda = @import("../../cuda/cuda.zig");
const exchange = @import("../../ranks/ranks.zig");
const draft_file = @import("draft.zig");
const forward_file = @import("forward.zig");
const lead_file = @import("../../ranks/lead.zig");
const model_file = @import("model.zig");
const ranks_forward = @import("ranks_forward.zig");
const state_file = @import("state.zig");

/// This process's share of the pair: its `Model` and its rank (0 leading, 1 following).
pub const Model = struct {
    rank: usize,
    share: model_file.Model,
    max_rows: usize,
};

/// One sequence's state on this rank: the split's KV and Mamba state are a rank's own.
pub const State = state_file.State;

/// The rank's window buffers beside the pair's `Ranks` (the partials and logits halves the gathers
/// read). `rows`/`scoredRows` are the rank buffers' own.
pub const Buffers = struct {
    rank: *state_file.Buffers,
    ranks: *ranks_forward.Ranks,
    /// The last window's scored rows — `logitsRow`'s `n` (`.all`'s whole window, `.last`'s one).
    scored: usize = 0,
    /// Rank 0's half of the pair's control channel: every windowed operation is told before it runs.
    lead: ?*lead_file.Lead = null,
    /// The gathered whole-vocabulary rows `logitRows` views (`logprobs`' launch reads them), beside
    /// `Ranks.row`'s halves — null on the follower, which joins the gather but keeps no face.
    wide: ?cuda.Buffer = null,

    /// The most rows a window of these buffers holds.
    rows: usize,

    pub fn scoredRows(b: *const Buffers) usize {
        return b.rank.scoredRows();
    }
    /// `wide`'s rows cover the scored window — `scoredRows` of the whole vocabulary.
    pub fn initWide(b: *Buffers) cuda.Error!void {
        b.wide = try .init(.bf16, &.{ b.rank.scoredRows(), @intCast(b.ranks.row.view.size[2] * ranks_forward.world) });
    }
    pub fn deinit(b: *Buffers) void {
        if (b.wide) |*wide| wide.deinit();
        b.rank.deinit();
    }
};

/// The window's gathered logits on device — `n` scored rows' whole vocabulary in `wide`, the faces
/// `logitsRow` fills (`logprobs`' launch reads it there).
pub fn logitRows(b: Buffers, row: usize, n: usize) cuda.View {
    return b.wide.?.span(row, n);
}

/// No graphs: a capture cannot hold the embed's host-side uploads, and the pair's windows are told, not
/// captured. The type exists so `P.Graphs` names something.
pub const Graphs = struct {
    pub fn init() Graphs {
        return .{};
    }
    pub fn deinit(_: *Graphs) void {}
    /// Never called (`P.graphed` keeps `Decoder.graphs` null); `holds` says so where a caller asks.
    pub fn holds(_: Graphs, _: State, _: Buffers, _: usize) bool {
        return false;
    }
    pub fn forward(_: *Graphs, _: Model, _: *State, _: *Buffers, _: []const i32) Error!void {
        unreachable;
    }
};

/// This rank's DFlash2 rig namespace where a `--drafter` is bound: `Target` keeps `lead` so the pair's
/// `flash_*` ops are told by `pass.zig` itself.
pub const draft = @import("dflash2.zig");

/// A window told and run. Rank 0 says `window` (rows, the score mode) with the ids; both ranks run the
/// `stage` (ids, embeddings, the audio rows over them) and the layers to their meet — and the scored
/// rows' logits halves gather right there into `b.wide`, one allGather a window instead of one a row
/// (`logitsRow`/`logitRows` then read the gathered face; the follower does the same inside the op).
pub fn forwardScoring(m: Model, state: *State, b: *Buffers, tokens: []const i32, score: Score) Error!void {
    if (b.lead) |lead| try lead.tell(@backingInt(ranks_forward.Op.window), &.{ @intCast(tokens.len), @backingInt(score), 0 }, tokens);
    try ranks_forward.forward(m.share, state, b.rank, b.ranks, tokens, score, false);
    b.scored = switch (score) {
        .all => tokens.len,
        .last => 1,
        .none => 0,
    };
    if (b.scored == 0) return;
    const wide = b.wide orelse return error.UnsupportedShape;
    if (b.scored > wide.view.size[0]) return error.UnsupportedShape;
    try gather(b.ranks, b.wide, b.scored);
}

/// The `n` scored rows' halves joined into `wide`'s whole-vocabulary rows — the window op's
/// collective; the follower's is the same call (its `wide` null, the halves joined and dropped).
pub fn gather(ranks: *ranks_forward.Ranks, wide: ?cuda.Buffer, n: usize) Error!void {
    const half: usize = @intCast(ranks.row.view.size[2]);
    if (ranks.gather != .loopback) try ranks.gather.run(.init(ranks.row.view.ptr.?, .bf16, &.{ ranks_forward.world, @intCast(n), @intCast(half) }));
    const w = wide orelse return;
    for (0..ranks_forward.world) |rank| {
        try w.copyColumns(rank * half * 2, exchange.partial(ranks.row.view, rank, n), n);
    }
}
/// `forward.zig`'s `forward`: every row scored (a verify window's `round`, the choice read off
/// `logitsRow` after).
pub fn forward(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
    return forwardScoring(m, state, b, tokens, .all);
}

/// A commit told (`keep` rows of the open window) and run.
pub fn commit(m: Model, state: *State, b: Buffers, tokens: []const i32, keep: usize) Error!void {
    if (b.lead) |lead| try lead.tell(@backingInt(ranks_forward.Op.commit), &.{@intCast(keep)}, tokens);
    return forward_file.commit(m.share, state, b.rank.*, tokens, keep);
}

/// Row `row`'s whole-vocabulary logits into `out` — the window's gather already joined the halves in
/// `b.wide`; the follower is told the row read (its `.row` op's own gather was the window's), then
/// rank 0 downloads it.
pub fn logitsRow(b: Buffers, row: usize, out: []u16) Error!void {
    const wide = b.wide orelse return error.UnsupportedShape;
    if (b.lead) |lead| try lead.tell(@backingInt(ranks_forward.Op.row), &.{ @intCast(row), @intCast(b.scored) }, &.{});
    if (out.len != wide.view.size[1] or row >= b.scored) return error.UnsupportedShape;
    try cuda.synchronize();
    try cuda.download(wide.span(row, 1), std.mem.sliceAsBytes(out));
}

/// The window's final residual streams: this rank's `b.hn` rows — the gathered folds make them the
/// same rows the serial `streams` gives, what the MTP head reads.
pub fn streams(b: Buffers, rows: usize) cuda.View {
    return b.rank.hn.rows(rows);
}

pub const Error = ranks_forward.Error || lead_file.Error || cuda.Error || error{UnsupportedShape};

pub const Score = forward_file.Score;
