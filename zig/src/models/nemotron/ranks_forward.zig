//! Nemotron-3.5-Transcribe's forward over the ranks one process holds (spec 151 WS-06): both in lockstep
//! (`Held` — every held rank's layer before the fold's gather, so the loopback meets real partials), or
//! one rank of a pair, the peer's over NCCL; `alone` zeroes the peer's slice (a stand-in, never a gate).
//! A world of one does not come through here; `forward.zig` runs it on the same kernels.
//!
//! A layer's product is fp32 [rows, hidden] in the gathered buffer `parts` (`ranks.partial` its own
//! slice): the MoE's slot rows and shared row fold there first (`glue.moePartial`), the out projections
//! write fp32 straight in; `Gather.run` brings the peer's and `ranks.sumRows` adds them rank 0 first,
//! rounded once into the delta the residual's add-RMSNorm takes. The split's sums run in another order
//! than the serial product's, so a row's logits can differ by the last rounding — the checks compare
//! picks and a step of bf16, not bits. The head scores this rank's half of the vocabulary into `row`'s
//! slice; `logitsRow` gathers a scored row whole.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const placeholders = @import("../../decode/placeholders.zig");
const mamba_file = @import("mamba.zig");
const attention_file = @import("attention.zig");
const moe_file = @import("moe.zig");
const model_file = @import("model.zig");
const forward_file = @import("forward.zig");
const state_file = @import("state.zig");
const ranks_file = @import("ranks.zig");
const exchange = @import("../../ranks/ranks.zig");
const glue = @import("glue.zig");

pub const Model = model_file.Model;
pub const State = state_file.State;
pub const Buffers = state_file.Buffers;
pub const Score = forward_file.Score;
pub const Error = forward_file.Error || exchange.Error;
pub const Share = ranks_file.Share;

/// The world the split is written for (`ranks.zig`'s: two).
pub const world = ranks_file.world;

/// A process's ranks' buffers for a window: the gathered partials and the logits row.
pub const Ranks = struct {
    /// fp32 [world, rows, hidden]: each rank's branch partial, tight for the window.
    parts: cuda.Buffer,
    /// bf16 [world, scored, vocab / world]: each rank's head rows, where a scored row's logits are read.
    row: cuda.Buffer,
    /// This process's rank of the two (0 or 1): the slice it writes.
    rank: usize,
    /// What brings the peer's partial and logits half.
    gather: exchange.Gather,

    pub fn init(m: Model, rows: usize, scored: usize, rank: usize, gather: exchange.Gather) cuda.Error!Ranks {
        const b = m.config.backbone;
        var r: Ranks = .{ .parts = undefined, .row = undefined, .rank = rank, .gather = gather };
        r.parts = try .init(.f32, &.{ world, rows, b.hidden });
        errdefer r.parts.deinit();
        r.row = try .init(.bf16, &.{ world, scored, b.vocab / world });
        return r;
    }

    /// One rank of the two, the other left at zero: `alone`'s peer slice is zeroed before the gather
    /// (nothing is sent), as `Model.alone`'s gate runs it.
    pub fn deinit(r: *Ranks) void {
        r.parts.deinit();
        r.row.deinit();
        r.* = undefined;
    }
};
/// What a leading rank tells its peer before the operation runs (`lead.zig`'s frame kind). One space for
/// the mirror's serve protocol: `window`'s rows staged, then the forward over them — rank 1 replays the
/// same op with the same ids; `commit` keeps a told window; `audio` carries a clip's prompt rows and
/// request-owned PCM for the follower's overlay; `row` reads gathered logits; `draft` runs an MTP
/// draft round; the `flash_*` ops are `dflash2/pass.zig`'s; `end` closes the session.
pub const Op = enum(u32) {
    /// `rows`, `score`, `alone`, then the window's `rows` int32 ids in the frame's blob.
    window = 0,
    /// `keep` — rows of the open window committed.
    commit = 1,
    /// The reply's start: the state reset, `pos` 0; the drafter's seats cleared.
    reset = 2,
    /// `n_rows`, `n_samples`; prompt row ids int32, then 16-kHz f32 PCM in `audio_frame_bytes` frames.
    /// Both Sparks use the same sample bits; the follower encodes them into its own `state.audio`.
    audio = 3,
    /// `row`, `n` — a `logitsRow` the pair gathers.
    row = 4,
    /// `want`, `position`; the window's `chosen` int32 — an MTP `draft` round of `want` asks.
    draft = 5,
    /// The drafter's ops (`dflash2/pass.zig`, `decode/dflash2/follow.zig`).
    flash_taps = 16,
    flash_block = 17,
    flash_round = 18,
    /// The session's last frame.
    end = 30,
    _,
};

/// PCM frames stay below the control channel's 64-MiB payload limit.
pub const audio_frame_bytes: usize = 1 << 20;

/// `forward` split at the embed: `stage` uploads the ids and lays the embeddings and audio rows (the
/// host-side moves a leading rank tells first); `forwardRows` the layers over a window `stage` just
/// made. A replayed rank calls both with the told ids; one process calls `forward`, which is both.
pub fn stage(m: Model, state: *State, b: *Buffers, tokens: []const i32, score: Score) Error!usize {
    const rows = tokens.len;
    if (rows == 0 or rows > b.rows or rows > m.max_rows or (score == .all and rows > b.scored)) return error.WindowTooWide;
    if (state.pos + rows > state.capacity) return error.ContextFull;
    try b.ids.uploadAt(0, std.mem.sliceAsBytes(tokens));
    try cuda.launched(cuda.kernels.zs_embed(&b.ids.rows(rows), &m.embed.view, &b.h.rows(rows), 1));
    try audioRows(state, b, rows);
    return rows;
}

/// The layers over `stage`'s `rows` rows. Eager only: each layer uploads its current host Mamba meta
/// inside `run`. A captured form must stage the meta eagerly, as `forward.zig` does, and capture only
/// kernels that read those live device words; capturing this function would freeze the upload values.
pub fn forwardRows(m: Model, state: *State, b: *Buffers, r: *Ranks, rows: usize, score: Score, alone: bool) Error!void {
    if (rows == 0 or rows > b.rows or rows > m.max_rows or (score == .all and rows > b.scored)) return error.WindowTooWide;
    if (state.pos + rows > state.capacity) return error.ContextFull;
    return run(m, state, b, r, rows, score, state.pos + rows, alone);
}

/// tokens -> this rank's layers, the partials met and folded as `forward.forward`'s. `alone` zeroes the
/// peer's slice instead of exchanging (a check's one-rank stand-in; it does not give the serial bits).
pub fn forward(m: Model, state: *State, b: *Buffers, r: *Ranks, tokens: []const i32, score: Score, alone: bool) Error!void {
    const rows = try stage(m, state, b, tokens, score);
    return forwardRows(m, state, b, r, rows, score, alone);
}

/// The state's audio rows laid over the embeddings this window covers (`decode/placeholders.zig`), after
/// `zs_embed` — `forward.zig`'s `audioRows`, the same bits on each rank.
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

/// Whether `rows` runs as a prompt chunk (above the window rows) or a window; the mixers take the same
/// rows either way, the mamba's prompts the wide kernels of one chunk.
fn wide(rows: usize) bool {
    return rows > mamba_file.window_rows;
}

/// The residual delta a block's mixer leaves for the next norm.
const Delta = union(enum) {
    none,
    /// This rank's fp32 product is in `parts[rank]` (dense rows).
    dense,
    /// A MoE layer's slot rows and shared row in `b.moe_sc`, the layer's own `Moe`.
    moe: moe_file.Moe,
};

/// The peer's slice of `parts` in for the fold: the peer's rows over `gather`, or zeros `alone`.
fn meet(r: *Ranks, rows: usize, alone: bool) Error!void {
    if (alone) try cuda.zeroLaunched(exchange.partial(r.parts.view, r.rank ^ 1, rows));
    if (r.gather != .loopback) try r.gather.run(exchange.gathered(r.parts.view, rows));
}

/// h = x + delta, the normed rows in `b.y`: the ranks' fp32 partials summed rank 0 first into `b.delta`
/// bf16 (one rounding, the serial product's place), then `addRmsnorm`'s fold.
fn foldDelta(m: Model, x: cuda.View, delta: Delta, w: cuda.View, b: *Buffers, r: *Ranks, rows: usize, alone: bool) Error!cuda.View {
    const c = m.config.backbone;
    const hn = b.hn.rows(rows);
    const y = b.y.rows(rows);
    const xs = b.xs.rows(rows);
    switch (delta) {
        .none => try glue.addRmsnorm(x, null, w, hn, y, xs, c.eps),
        .dense => {
            try meet(r, rows, alone);
            try exchange.sumRows(exchange.gathered(r.parts.view, rows), 0, b.delta.rows(rows));
            try glue.addRmsnorm(x, b.delta.rows(rows), w, hn, y, xs, c.eps);
        },
        .moe => |one| {
            // fold this rank's slot rows and shared row into its slice, meet, sum — the fold's slot
            // order and weights are the serial layer's (`zs_nemotron_moe_partial`, `tp.py`'s
            // `_moe_partial`).
            const s = one.shape;
            const yk: cuda.View = .{ .ptr = b.moe_sc.y.view.ptr, .dtype = .f32, .ndim = 3, .size = .{ @intCast(rows), @intCast(s.slots()), @intCast(s.dims), 0, 0, 0 } };
            try glue.moePartial(yk, b.moe_sc.wts.rows(rows), exchange.partial(r.parts.view, r.rank, rows), s.top);
            try meet(r, rows, alone);
            try exchange.sumRows(exchange.gathered(r.parts.view, rows), 0, b.delta.rows(rows));
            try glue.addRmsnorm(x, b.delta.rows(rows), w, hn, y, xs, c.eps);
        },
    }
    return if (delta == .none) x else hn;
}

/// The slot of `layer` among `layers`, or null (an off-by-one is not a tap) — `forward.zig`'s.
fn tapIndexOf(layers: []const u32, layer: usize) ?usize {
    for (layers, 0..) |one, i| if (one == layer) return i;
    return null;
}

/// One rank's position inside the window: the residual, the pending delta and the per-kind counters a
/// block needs. A `Held` lockstep keeps one a rank and interleaves their layers.
const Place = struct {
    x: cuda.View,
    delta: Delta = .none,
    mi: usize = 0,
    ai: usize = 0,
};

/// Block `blk`'s fold of the pending delta, the tap of the layer it completes, then this rank's mixer's
/// product made the next delta — `forward.run`'s block body, the fold's `meet` the only exchange.
fn blockStep(m: Model, state: *State, b: *Buffers, r: *Ranks, rows: usize, context: usize, alone: bool, p: *Place, blk: *model_file.Block, index: usize) Error!void {
    const y = b.y.rows(rows);
    const mine = exchange.partial(r.parts.view, r.rank, rows);
    p.x = try foldDelta(m, p.x, p.delta, blk.norm(), b, r, rows, alone);
    if (index > 0) {
        if (tapIndexOf(b.tap_layers, index - 1)) |slot| {
            const stream = if (p.delta == .none) p.x else b.hn.rows(rows);
            try b.taps.?.copyColumns(slot * @as(usize, @intCast(stream.size[1])) * 2, stream, rows);
        }
    }
    switch (blk.*) {
        .mamba => |*one| {
            try one.in_proj.forward(y, b.proj.rows(rows));
            try state.mamba_meta[p.mi].set(b.meta);
            if (wide(rows))
                try one.mixer.prompt(state.mamba[p.mi], b.mamba_sc, b.proj.rows(rows), b.meta.view, b.gate.rows(rows))
            else
                try one.mixer.window(state.mamba[p.mi], b.mamba_sc, b.proj.rows(rows), b.meta.view, b.gate.rows(rows));
            try one.out_proj.forward(b.gate.rows(rows), mine);
            p.delta = .dense;
            p.mi += 1;
        },
        .attention => |*one| {
            // `window`/`prompt` run the layer whole: qkv, the K/V writes at `pos`, the attend and
            // o_proj — here fp32 into the rank's partial (the serial out's bf16 rounds at the sum).
            if (wide(rows))
                try one.mixer.prompt(state.attention[p.ai], b.attn_sc, y, state.position.view, context, mine)
            else
                try one.mixer.window(state.attention[p.ai], b.attn_sc, y, state.position.view, context, mine);
            p.delta = .dense;
            p.ai += 1;
        },
        .moe => |*one| {
            if (wide(rows))
                try moe_file.prompt(one.mixer, b.moe_sc, y)
            else
                try moe_file.window(one.mixer, b.moe_sc, y);
            p.delta = .{ .moe = one.mixer };
        },
    }
}

/// The last fold: the pending delta met into `b.hn` (the last block's completed output and its tap),
/// then the head `score` names into this rank's slice of `row`.
fn finish(m: Model, b: *Buffers, r: *Ranks, rows: usize, score: Score, alone: bool, p: *Place) Error!void {
    const y = b.y.rows(rows);
    p.x = try foldDelta(m, p.x, p.delta, m.norm_f.view, b, r, rows, alone);
    if (tapIndexOf(b.tap_layers, m.blocks.len - 1)) |slot| {
        const stream = if (p.delta == .none) p.x else b.hn.rows(rows);
        try b.taps.?.copyColumns(slot * @as(usize, @intCast(stream.size[1])) * 2, stream, rows);
    }
    switch (score) {
        .none => {},
        // this rank's half of the vocabulary, into its slice of `row` for `logitsRow`'s gather
        .all => try m.head.forward(y, exchange.partial(r.row.view, r.rank, rows)),
        .last => try m.head.forward(y.span(rows - 1, 1), exchange.partial(r.row.view, r.rank, 1)),
    }
}

/// The window's layers over `b.h`'s `rows` rows, then the final norm and the head `score` names — as
/// `forward.run`'s, each dense product landing fp32 in `parts[rank]` and the MoE slots folded at `meet`.
/// One rank a call: the serial-loopback `Held` drives every rank's `layer` before the next fold instead.
fn run(m: Model, state: *State, b: *Buffers, r: *Ranks, rows: usize, score: Score, context: usize, alone: bool) Error!void {
    var p: Place = .{ .x = b.h.rows(rows) };
    for (m.blocks, 0..) |*blk, index| try blockStep(m, state, b, r, rows, context, alone, &p, blk, index);
    return finish(m, b, r, rows, score, alone, &p);
}

/// The `held` ranks a loopback process runs in lockstep (GLM's `Held`): every rank's layer `index`'s
/// compute — its partial landed — before any rank's next fold reads the peer's slice. `gather.run` is a
/// no-op over `.loopback` because the peer's partial is in place already, made this layer. A process a
/// rank (`held` 1, `.nccl`) drives `run` whole: the exchange itself is the barrier.
pub const Held = struct {
    held: usize,
    m: [world]Model,
    state: [world]*State,
    b: [world]*Buffers,
    ranks: *Ranks,
    alone: bool = false,

    /// `stage`'s window over every held rank in lockstep: every rank's layer `index`'s compute — its
    /// partial landed — before any rank's next fold reads the peer's slice, so a layer's gather is
    /// ranks' `layer` calls apart. `commit` stays the caller's (`forward.commit` a rank, as `ranks` does).
    pub fn forward(h: *Held, tokens: []const i32, score: Score) Error!void {
        const rows = tokens.len;
        if (rows == 0 or rows > h.b[0].rows or rows > h.m[0].max_rows or (score == .all and rows > h.b[0].scored)) return error.WindowTooWide;
        if (h.state[0].pos + rows > h.state[0].capacity) return error.ContextFull;
        const context = h.state[0].pos + rows;
        var places: [world]Place = undefined;
        for (0..h.held) |i| {
            try stageOnly(h.m[i], h.state[i], h.b[i], tokens);
            places[i] = .{ .x = h.b[i].h.rows(rows) };
        }
        for (h.m[0].blocks, 0..) |*blk, index| {
            for (0..h.held) |i| {
                if (h.held != 1) h.ranks.rank = i;
                try blockStep(h.m[i], h.state[i], h.b[i], h.ranks, rows, context, h.alone, &places[i], blk, index);
            }
        }
        for (0..h.held) |i| {
            if (h.held != 1) h.ranks.rank = i;
            try finish(h.m[i], h.b[i], h.ranks, rows, score, h.alone, &places[i]);
        }
        h.ranks.rank = switch (h.ranks.gather) {
            .loopback => 0,
            .nccl => |comm| comm.rank,
        };
    }

    /// The window's ids in `b.ids` and its embeddings (audio over them) in `b.h`: `forward`'s `stage`
    /// per rank, the bounds checked once by the caller.
    fn stageOnly(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
        const rows = tokens.len;
        try b.ids.uploadAt(0, std.mem.sliceAsBytes(tokens));
        try cuda.launched(cuda.kernels.zs_embed(&b.ids.rows(rows), &m.embed.view, &b.h.rows(rows), 1));
        try audioRows(state, b, rows);
    }
};

/// Row `row` of the last scored window's logits, bf16 [vocab] to the host `out`: the ranks' halves over
/// `gather`, each rank's [n, half] slice copied beside the other's into one wide row. `n` rows were
/// scored (`.all`'s window or `.last`'s one); `row` picks among them.
pub fn logitsRow(r: *Ranks, row: usize, n: usize, out: []u16) (exchange.Error || error{UnsupportedShape})!void {
    const half: usize = @intCast(r.row.view.size[2]);
    if (out.len != world * half or row >= n) return error.UnsupportedShape;
    if (r.gather != .loopback) try r.gather.run(.init(r.row.view.ptr.?, .bf16, &.{ world, @intCast(n), @intCast(half) }));
    var wide_row: cuda.Buffer = try .init(.bf16, &.{ n, world * half });
    defer wide_row.deinit();
    for (0..world) |rank| {
        try wide_row.copyColumns(rank * half * 2, exchange.partial(r.row.view, rank, n), n);
    }
    try cuda.synchronize();
    try cuda.download(wide_row.span(row, 1), std.mem.sliceAsBytes(out));
}
