//! The drafter's two passes over a rank's share (`dflash2.py`): a context update, which turns kept rows' taps
//! into each layer's K and V at their positions (`_taps_compute`), and a block pass, which runs the pending
//! token and the masks through the five layers over that context (`_block_compute`, `_layer`) and leaves the
//! rank's logits of the drafts and their projected rows. A layer has two row-parallel partials, the attention's
//! `o` and the MLP's `down`, added over the ranks in rank order: the phases below end and begin at them, as
//! the main model's do, and `PairOf(held, Target)` runs the ranks a process holds: both (the tests' and the
//! reference for two devices), or its own with a `ranks.Gather` over NCCL (spec 045 WS-09). A block pass has
//! eleven gathers: a layer's two partials, tight as the main model's (`zig/src/ranks/ranks.zig`), and at its end
//! each rank's largest logits and their tokens, so that every rank chooses its drafts from the same candidates.
//!
//! A context row's K and V depend on its taps and its position alone: not on the rows added with it, and not
//! on the ring. Inside a block the rows read each other by design.
//!
//! Shared by GLM-5.3-Flash and Qwen3.8 Flash Next (spec 136 WS-03). RMSNorm, the ranks' branch and SwiGLU are
//! `decode/glue.zig`'s. What a model gives is `Target`, a namespace its `dflash2/pass.zig` declares:
//!
//! - `ranks`, `lead`: `ranks/ranks.zig` and `ranks/lead.zig` (`decode/` does not import `ranks/`);
//! - `Op`: its `ranks_forward.Op`, whose `flash_taps`, `flash_block` and `flash_round` a leading rank tells;
//! - `wide`: a prompt's update read where the taps stand, `wide` rows a pass (Flash Next, spec 135 WS-02);
//!   false, `batch` rows copied in, a captured graph where the pair has graphs (GLM);
//! - for `drafter.zig`, `seats.zig` and `rig.zig`: `Of(held)`, `Job`, `owns_heads`, `initRig`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const config_file = @import("config.zig");
const kernels = @import("kernels.zig");
const ring_file = @import("ring.zig");
const weights_file = @import("weights.zig");
const Config = config_file.Config;
const Weights = weights_file.Weights;

/// The row glue the drafter shares with GLM's layers: bf16 RMSNorm, rank-order branch, unclipped SwiGLU.
pub const glue = @import("../glue.zig");

pub const Error = cuda.Error;
/// A pass with its gathers.
pub fn GatherError(comptime Target: type) type {
    return cuda.Error || Target.ranks.Error;
}
pub const world = config_file.world;

/// The most context rows of one update (`Drafter.tap_in`).
pub const batch = 64;

/// The most context rows of one pass over a prompt's taps (spec 135 WS-02): a prompt window's, read from the
/// taps where they stand, never more than the ring holds. The linears are loaded for this many rows.
pub const wide = 2048;

/// `dflash2.py`'s `NO_LIMIT`: the drafter's SwiGLU clips nothing.
const no_limit: f32 = 1e30;

/// A rank's context and what its passes write.
pub const Rank = struct {
    c: *const Config,
    /// The context's rows: the ring's, or `capacity` and a block flat.
    cap: usize,
    ring: bool,
    /// bf16 [KV heads, cap, 128] a layer.
    k: []cuda.Buffer,
    v: []cuda.Buffer,
    /// int32 [1]: the position a launch's first row stands at, staged before it.
    pos: cuda.Buffer,
    /// A context update's rows: the taps bf16 [batch, taps x dims], `fc`'s output and its norm, a layer's k | v.
    tap_in: cuda.Buffer,
    projected: cuda.Buffer,
    ctx: cuda.Buffer,
    kv_rows: cuda.Buffer,
    /// A block: int32 [block] ids; bf16 [block, dims] the rows, their norm, a convolution's output and a
    /// gathered branch; [block, kernel] a row's kernel.
    ids: cuda.Buffer,
    x: cuda.Buffer,
    normed: cuda.Buffer,
    mixed: cuda.Buffer,
    branch: cuda.Buffer,
    dyn: cuda.Buffer,
    /// bf16 [block, heads x 128] the queries as rows and the attention's output; [heads, block, 128] by head.
    q_rows: cuda.Buffer,
    q: cuda.Buffer,
    attended: cuda.Buffer,
    /// The attention's scratch (`kernels.attendParts`).
    parts: cuda.Buffer,
    /// bf16 [block, 2 width] and [block, width].
    gate_up: cuda.Buffer,
    act: cuda.Buffer,
    /// The drafts' rows (the block's but its first): bf16 [block - 1, dims] normed, [block - 1, vocab / world]
    /// the rank's logits, [block - 1, selector rank] projected for the chain.
    final: cuda.Buffer,
    logits: cuda.Buffer,
    selected: cuda.Buffer,
    /// A draft row's largest logits of the rank, fp32 and int32 [block - 1, top_k]: the largest first, the
    /// lower token among equals, the tokens the whole vocabulary's. The rank's place in the pair's `tops`,
    /// which owns the memory.
    top_values: cuda.Buffer,
    top_tokens: cuda.Buffer,
    /// int32 [`kernels.topScratch`]: `kernels.top`'s slices' candidates.
    picks: cuda.Buffer,
    /// Which rank this is, and the first token of its rows of the vocabulary.
    rank: usize,
    first_token: usize,
    gpa: Allocator,

    /// For a context of up to `capacity` positions; `ring`: kept in the ring of the window and a block where
    /// that is smaller. `half` is the rank's rows of the vocabulary, `tops` the pair's gathered candidates.
    /// `blocks`: the blocks its block buffers hold side by side (`seats.zig`); a pair's own hold one. `mine` is
    /// the rank's place in `tops`, `ranks.partial(tops, rank, 2 * blocks * (block - 1))`.
    pub fn init(gpa: Allocator, c: *const Config, capacity: usize, ring: bool, half: usize, rank: usize, mine: cuda.View, blocks: usize) (Error || Allocator.Error)!Rank {
        const shaped = ring_file.shape(c, capacity, ring);
        const cap = shaped.cap;
        const ringed = shaped.ring;
        const d = c.hidden;
        const n = blocks * c.block;
        const drafts = blocks * (c.block - 1);
        const kv = c.rankKvHeads();
        const heads = c.rankHeads();
        var self: Rank = undefined;
        self.c = c;
        self.cap = cap;
        self.ring = ringed;
        self.gpa = gpa;
        self.rank = rank;
        self.first_token = rank * half;
        const place_of: [*]u8 = @ptrCast(mine.ptr.?);
        self.top_values = .{ .view = .init(place_of, .f32, &.{ drafts, c.top_k }) };
        self.top_tokens = .{ .view = .init(place_of + drafts * c.top_k * @sizeOf(f32), .i32, &.{ drafts, c.top_k }) };
        self.k = try gpa.alloc(cuda.Buffer, c.layers);
        errdefer gpa.free(self.k);
        self.v = try gpa.alloc(cuda.Buffer, c.layers);
        errdefer gpa.free(self.v);
        var made: usize = 0;
        errdefer for (self.k[0..made], self.v[0..made]) |*k, *v| {
            k.deinit();
            v.deinit();
        };
        for (self.k, self.v) |*k, *v| {
            k.* = try .init(.bf16, &.{ kv, cap, kernels.head });
            errdefer k.deinit();
            v.* = try .init(.bf16, &.{ kv, cap, kernels.head });
            errdefer v.deinit();
            try k.zero();
            try v.zero();
            made += 1;
        }
        const shapes = .{
            .{ "pos", cuda.DType.i32, &[_]usize{1} },
            .{ "tap_in", cuda.DType.bf16, &[_]usize{ batch, c.tapWidth() } },
            .{ "projected", cuda.DType.bf16, &[_]usize{ batch, d } },
            .{ "ctx", cuda.DType.bf16, &[_]usize{ batch, d } },
            .{ "kv_rows", cuda.DType.bf16, &[_]usize{ @max(batch, n), 2 * kv * kernels.head } },
            .{ "ids", cuda.DType.i32, &[_]usize{n} },
            .{ "x", cuda.DType.bf16, &[_]usize{ n, d } },
            .{ "normed", cuda.DType.bf16, &[_]usize{ n, d } },
            .{ "mixed", cuda.DType.bf16, &[_]usize{ n, d } },
            .{ "branch", cuda.DType.bf16, &[_]usize{ n, d } },
            .{ "dyn", cuda.DType.bf16, &[_]usize{ n, c.kernelWidth() } },
            .{ "q_rows", cuda.DType.bf16, &[_]usize{ n, heads * kernels.head } },
            .{ "q", cuda.DType.bf16, &[_]usize{ blocks * heads, c.block, kernels.head } },
            .{ "attended", cuda.DType.bf16, &[_]usize{ n, heads * kernels.head } },
            .{ "parts", cuda.DType.f32, &[_]usize{kernels.attendParts(heads, c.block, c.seen(cap))} },
            .{ "gate_up", cuda.DType.bf16, &[_]usize{ n, 2 * c.rankWidth() } },
            .{ "act", cuda.DType.bf16, &[_]usize{ n, c.rankWidth() } },
            .{ "final", cuda.DType.bf16, &[_]usize{ drafts, d } },
            .{ "logits", cuda.DType.bf16, &[_]usize{ drafts, half } },
            .{ "selected", cuda.DType.bf16, &[_]usize{ drafts, c.selector_rank } },
            .{ "picks", cuda.DType.i32, &[_]usize{@max(1, kernels.topScratch(drafts, half, c.top_k))} },
        };
        comptime var done: usize = 0;
        errdefer inline for (shapes, 0..) |shape, at| {
            if (at < done) @field(self, shape[0]).deinit();
        };
        inline for (shapes) |shape| {
            @field(self, shape[0]) = try .init(shape[1], shape[2]);
            done += 1;
        }
        // every row of a block but the first is the mask
        const ids = try gpa.alloc(i32, n);
        defer gpa.free(ids);
        @memset(ids, @intCast(c.mask));
        try self.ids.upload(std.mem.sliceAsBytes(ids));
        try self.pos.zero();
        return self;
    }

    pub fn deinit(self: *Rank) void {
        for (self.k, self.v) |*k, *v| {
            k.deinit();
            v.deinit();
        }
        self.gpa.free(self.k);
        self.gpa.free(self.v);
        inline for (.{ "pos", "tap_in", "projected", "ctx", "kv_rows", "ids", "x", "normed", "mixed", "branch", "dyn", "q_rows", "q", "attended", "parts", "gate_up", "act", "final", "logits", "selected", "picks" }) |name| @field(self, name).deinit();
        self.* = undefined;
    }

    fn place(self: Rank) kernels.Place {
        return .{ .pos = self.pos.view, .ring = self.ring };
    }

    /// Before a launch: the position its first row stands at.
    pub fn stage(self: *Rank, position: usize) Error!void {
        const at: i32 = @intCast(position);
        return self.pos.uploadLaunched(0, std.mem.asBytes(&at));
    }

    /// Bytes of the context's K and V.
    pub fn contextBytes(self: Rank) u64 {
        var sum: u64 = 0;
        for (self.k, self.v) |k, v| sum += k.bytes() + v.bytes();
        return sum;
    }
};

/// `rows` kept rows' taps, in `r.tap_in`, become each layer's K and V at the staged position onward. The
/// caller moves the context's end.
pub fn context(w: Weights, r: *Rank, rows: usize) Error!void {
    std.debug.assert(rows >= 1 and rows <= batch);
    return contextOf(w, r, r.tap_in.rows(rows), .{ .projected = r.projected.rows(rows), .ctx = r.ctx.rows(rows), .kv_rows = r.kv_rows.rows(rows) });
}

/// What a context pass writes on its way to K and V, `taps`' rows of each.
const Through = struct { projected: cuda.View, ctx: cuda.View, kv_rows: cuda.View };

/// `taps` bf16 [rows, taps x dims] become each layer's K and V at the staged position onward, by way of
/// `through`. A row's bits are its own: in a pass of 64 rows or of a window's, the same K and V.
fn contextOf(w: Weights, r: *Rank, taps: cuda.View, through: Through) Error!void {
    const c = r.c;
    // the ring's slots of one pass are distinct
    std.debug.assert(taps.size[0] <= r.cap);
    try w.fc.forward(taps, through.projected);
    try glue.rmsnorm(through.projected, w.hidden_norm.view, through.ctx, @floatCast(c.eps));
    for (w.layers, r.k, r.v) |layer, k, v| {
        try layer.kv.forward(through.ctx, through.kv_rows);
        try kernels.prep(null, through.kv_rows, layer.q_norm.view, layer.k_norm.view, r.place(), null, k.view, v.view, @floatCast(c.eps), c.theta);
    }
}

/// A rank's buffers for a prompt's context pass (`wide` rows, or the ring's if fewer): the pair's, not the
/// rank's, whose block buffers the pool's seats size by `batch` (`seats.zig`).
pub const Wide = struct {
    rows: usize,
    projected: cuda.Buffer,
    ctx: cuda.Buffer,
    kv_rows: cuda.Buffer,

    pub fn init(r: Rank) Error!Wide {
        const c = r.c;
        const rows = @min(wide, r.cap);
        var projected: cuda.Buffer = try .init(.bf16, &.{ rows, c.hidden });
        errdefer projected.deinit();
        var ctx: cuda.Buffer = try .init(.bf16, &.{ rows, c.hidden });
        errdefer ctx.deinit();
        const kv_rows: cuda.Buffer = try .init(.bf16, &.{ rows, 2 * c.rankKvHeads() * kernels.head });
        return .{ .rows = rows, .projected = projected, .ctx = ctx, .kv_rows = kv_rows };
    }

    pub fn deinit(self: *Wide) void {
        self.projected.deinit();
        self.ctx.deinit();
        self.kv_rows.deinit();
        self.* = undefined;
    }

    fn through(self: Wide, rows: usize) Through {
        return .{ .projected = self.projected.rows(rows), .ctx = self.ctx.rows(rows), .kv_rows = self.kv_rows.rows(rows) };
    }
};

/// The block's ids into its rows: `embed` is the main model's table, bf16 [vocab, dims].
pub fn embed(r: *Rank, table: cuda.View) Error!void {
    return cuda.launched(cuda.kernels.zs_embed(&r.ids.view, &table, &r.x.view, 1));
}

/// Layer `index`'s attention over the block: its K and V written after the context's end, and the rank's
/// partial of `o` into `partial`, fp32 [block, dims].
pub fn attention(w: Weights, r: *Rank, index: usize, partial: cuda.View) Error!void {
    const c = r.c;
    const layer = w.layers[index];
    const eps: f32 = @floatCast(c.eps);
    try glue.rmsnorm(r.x.view, layer.in_norm.view, r.normed.view, eps);
    try layer.a_kp.forward(r.normed.view, r.dyn.view);
    try kernels.dconv(r.normed.view, r.dyn.view, layer.a_base.view, 0, null, r.mixed.view);
    try layer.q.forward(r.mixed.view, r.q_rows.view);
    try layer.kv.forward(r.mixed.view, r.kv_rows.rows(c.block));
    try kernels.prep(r.q_rows.view, r.kv_rows.rows(c.block), layer.q_norm.view, layer.k_norm.view, r.place(), r.q.view, r.k[index].view, r.v[index].view, eps, c.theta);
    try kernels.attend(r.q.view, r.k[index].view, r.v[index].view, r.place(), r.attended.view, r.parts.view, c.seen(r.cap), c.causal);
    try layer.o.forward(r.attended.view, partial);
}

/// The attention's branch from `gathered`, fp32 [world, block, dims], through the convolution's second branch
/// onto the rows; then the MLP's partial of `down` into `partial`.
pub fn mlp(w: Weights, r: *Rank, index: usize, gathered: cuda.View, partial: cuda.View) Error!void {
    const c = r.c;
    const layer = w.layers[index];
    try glue.branch(gathered, r.branch.view);
    try kernels.dconv(r.branch.view, r.dyn.view, layer.a_base.view, 1, r.x.view, r.x.view);
    try glue.rmsnorm(r.x.view, layer.post_norm.view, r.normed.view, @floatCast(c.eps));
    try layer.m_kp.forward(r.normed.view, r.dyn.view);
    try kernels.dconv(r.normed.view, r.dyn.view, layer.m_base.view, 0, null, r.mixed.view);
    try layer.gate_up.forward(r.mixed.view, r.gate_up.view);
    try glue.swiglu(r.gate_up.view, r.act.view, no_limit);
    try layer.down.forward(r.act.view, partial);
}

/// The MLP's branch from `gathered` onto the rows: the layer's end.
pub fn finish(w: Weights, r: *Rank, index: usize, gathered: cuda.View) Error!void {
    try glue.branch(gathered, r.branch.view);
    try kernels.dconv(r.branch.view, r.dyn.view, w.layers[index].m_base.view, 1, r.x.view, r.x.view);
}

/// After the last layer: the drafts' rows normed, the rank's logits of them from `m.head` (the main model's
/// rows of the vocabulary), their largest — a four-bit head's picks dotted again against the rows themselves
/// (`rescore`) — and the rows' projection for the chain.
pub fn head(w: Weights, r: *Rank, m: Main) Error!void {
    const c = r.c;
    try glue.rmsnorm(r.x.span(1, c.block - 1), w.norm.view, r.final.view, @floatCast(c.eps));
    try m.head.forward(r.final.view, r.logits.view);
    try kernels.top(r.logits.view, r.top_values.view, r.top_tokens.view, r.picks.view, r.first_token);
    try rescore(m, r.top_tokens.view, r.final.view, r.top_values.view, r.first_token);
    try w.projection.forward(r.final.view, r.selected.view);
}

/// The four-bit head's picks of a block's rows, dotted again against the main model's own bf16 rows
/// (`Main.rows`), their values replaced: the copy stays the proposer, the stored head the scorer. The order
/// the picks stand in is the four-bit head's; what merges them reads values. Nothing to do where the head was
/// read as stored: its logits are the bf16 rows'.
pub fn rescore(m: Main, tokens: cuda.View, final: cuda.View, values: cuda.View, first: usize) Error!void {
    switch (m.head) {
        .bf16, .mxfp8, .fp8, .fp8_block => {},
        .four => try kernels.rescore(tokens, m.rows, final, values, first),
    }
}

/// What the main model gives a rank's drafter: its embedding and its rows of the vocabulary.
pub const Main = struct {
    embed: cuda.View,
    /// The main model's own BF16 rows, or a four-bit copy of them made at load (TensorFold's `draft_head`):
    /// drafts only propose, and the head is half of a block's time read as stored (spec 067).
    head: weights_file.Linear,
    /// The rows `head` was made from, bf16 [vocab / world, dims] as the main checkpoint stores them: a
    /// four-bit head's picks are scored again here (`rescore`). Unused where the head is the stored one.
    rows: cuda.View,
};

/// A pair's context at one position, to return to: the rows a block there reads, the window back, each
/// rank's K and V a layer in position order. A context row is its taps' and its position's alone, so a pair
/// that restores it, of any capacity, flat or a ring, stands where the pair it was taken from stood.
pub fn KeptOf(comptime held: usize, comptime Target: type) type {
    return struct {
        const Self = @This();

        end: usize,
        rows: usize,
        /// bf16 [layers x (K, V) x KV heads x rows, 128] a rank.
        ranks: [held]cuda.Buffer,

        pub fn take(pair: *const PairOf(held, Target)) Error!Self {
            const c = pair.ranks[0].c;
            const rows = @min(pair.context_end, c.window orelse pair.context_end);
            const shape = [_]usize{ c.layers * 2 * c.rankKvHeads() * rows, kernels.head };
            var self: Self = .{ .end = pair.context_end, .rows = rows, .ranks = undefined };
            var done: usize = 0;
            errdefer for (self.ranks[0..done]) |*one| one.deinit();
            for (&self.ranks) |*one| {
                one.* = try .init(.bf16, &shape);
                done += 1;
            }
            try self.move(pair, .out);
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (&self.ranks) |*one| one.deinit();
            self.* = undefined;
        }

        /// `pair`'s context is this one's again, and ends where it ended.
        pub fn restore(self: Self, pair: *PairOf(held, Target)) Error!void {
            std.debug.assert(self.end <= pair.capacity);
            try self.move(pair, .in);
            pair.context_end = self.end;
        }

        /// The rows by position, between where a rank keeps them (its slot, the ring's wrap in two runs) and
        /// here. Waits before and after: a pair's graphs launch on a stream of their own.
        fn move(self: Self, pair: *const PairOf(held, Target), way: enum { out, in }) Error!void {
            try cuda.synchronize();
            for (pair.ranks, self.ranks) |rank, kept| {
                const kv = rank.c.rankKvHeads();
                std.debug.assert(self.rows <= rank.cap);
                const wrapped = ring_file.wrap(rank.cap, rank.ring, self.end, self.rows);
                const slot = wrapped.slot;
                const run = wrapped.run;
                var at: usize = 0;
                for (rank.k, rank.v) |k, v| for ([_]cuda.Buffer{ k, v }) |buffer| {
                    // [KV heads, cap, 128] read as rows of 128
                    const all: cuda.Buffer = .{ .view = .init(buffer.view.ptr.?, .bf16, &.{ kv * rank.cap, kernels.head }) };
                    for (0..kv) |h| {
                        for ([_][3]usize{ .{ 0, slot, run }, .{ run, 0, self.rows - run } }) |part| {
                            if (part[2] == 0) continue;
                            switch (way) {
                                .out => try kept.copyRows(at + part[0], all, h * rank.cap + part[1], part[2]),
                                .in => try all.copyRows(h * rank.cap + part[1], kept, at + part[0], part[2]),
                            }
                        }
                        at += self.rows;
                    }
                };
            }
            try cuda.synchronize();
        }
    };
}

/// Where `computeWatched` stops for its watcher.
pub const Point = enum {
    /// After a layer's attention and its gather: `attn` holds every rank's partial.
    attention,
    /// After a layer's end: `ffn` holds every rank's partial, and the block's rows are the layer's.
    layer,
};

/// The drafter over the ranks one process holds: both, as `loopback.zig` runs the main model's, or its own,
/// the other's partials and candidates brought by `gather`.
pub fn PairOf(comptime held: usize, comptime Target: type) type {
    comptime std.debug.assert(held == 1 or held == world);
    return struct {
        const Self = @This();

        /// The ranks held, rising: `ranks[i].rank` is which.
        weights: [held]Weights,
        ranks: [held]Rank,
        /// Each rank's buffers for a prompt's pass (`Wide`), where `Target.wide`.
        wides: [if (Target.wide) held else 0]Wide,
        /// fp32 [world, block, dims]: the attention's partials and the MLP's, a rank's at `rank * block * dims`.
        attn: cuda.Buffer,
        ffn: cuda.Buffer,
        /// Every rank's candidates of a block, 4 bytes a value [world, 2 (block - 1), top_k]: a rank's largest
        /// logits fp32 [block - 1, top_k], then their tokens int32 [block - 1, top_k] (`Rank.top_values`,
        /// `top_tokens`), gathered as one. TensorFold packs the ids as fp32 beside the values a row; here a
        /// rank's two tables are what `kernels.top` writes and what the host's merge reads, a rank at a time.
        tops: cuda.Buffer,
        /// What brings the other ranks' partials and candidates: nothing where they are all here.
        gather: Target.ranks.Gather = .loopback,
        /// This rank leads: an update and a block are told before they run (`Target.Op`, a model's `session/mirror.zig`).
        lead: ?*Target.lead.Lead = null,
        /// Positions of the context: the next kept row's.
        context_end: usize = 0,
        capacity: usize,
        /// With graphs, an update and a block are launched as captured ones: the same kernels on the same
        /// buffers, a block's gathers among them.
        graphs: ?*Graphs = null,

        /// Every rank in this process.
        pub fn init(gpa: Allocator, weights: [held]Weights, c: *const Config, capacity: usize, ring: bool, half: usize) (Error || Allocator.Error)!Self {
            comptime std.debug.assert(held == world);
            return initFrom(gpa, weights, c, capacity, ring, half, 0, .loopback);
        }

        /// Ranks `first` .. `first + held - 1`, their weights in that order, the others' through `gather`.
        pub fn initFrom(gpa: Allocator, weights: [held]Weights, c: *const Config, capacity: usize, ring: bool, half: usize, first: usize, gather: Target.ranks.Gather) (Error || Allocator.Error)!Self {
            std.debug.assert(first + held <= world and gather.serves(held, world));
            var self: Self = .{ .weights = weights, .ranks = undefined, .wides = undefined, .attn = undefined, .ffn = undefined, .tops = undefined, .gather = gather, .capacity = capacity };
            self.attn = try .init(.f32, &.{ world, c.block, c.hidden });
            errdefer self.attn.deinit();
            self.ffn = try .init(.f32, &.{ world, c.block, c.hidden });
            errdefer self.ffn.deinit();
            self.tops = try .init(.f32, &.{ world, 2 * (c.block - 1), c.top_k });
            errdefer self.tops.deinit();
            var done: usize = 0;
            errdefer for (self.ranks[0..done]) |*rank| rank.deinit();
            for (&self.ranks, first..) |*rank, which| {
                rank.* = try .init(gpa, c, capacity, ring, half, which, Target.ranks.partial(self.tops.view, which, 2 * (c.block - 1)), 1);
                done += 1;
            }
            if (Target.wide) {
                var wided: usize = 0;
                errdefer for (self.wides[0..wided]) |*one| one.deinit();
                for (&self.wides, self.ranks) |*one, rank| {
                    one.* = try .init(rank);
                    wided += 1;
                }
            }
            return self;
        }

        /// The ranks and the buffers; the weights are the caller's.
        pub fn deinit(self: *Self) void {
            self.attn.deinit();
            self.ffn.deinit();
            self.tops.deinit();
            for (&self.wides) |*one| one.deinit();
            for (&self.ranks) |*rank| rank.deinit();
            self.* = undefined;
        }

        pub fn reset(self: *Self) void {
            self.context_end = 0;
        }

        /// The context starts at `pos` with no rows before it, every row zeroed: a block there reads zeros
        /// where the positions before `pos` would stand, and no other sequence's rows, until the window back is
        /// absorbed. A mark without the context (a model session's `restoreLocal`) goes on from here.
        pub fn startAt(self: *Self, pos: usize) Error!void {
            std.debug.assert(pos <= self.capacity);
            // a pair's graphs launch on a stream of their own
            try cuda.synchronize();
            for (&self.ranks) |*rank| for (rank.k, rank.v) |k, v| {
                try k.zero();
                try v.zero();
            };
            try cuda.synchronize();
            self.context_end = pos;
        }

        /// The context ends at `end` again, told to the rank that follows as an update of no rows: the rows an
        /// update wrote past it are written again by the next (a model session's `flashTo`).
        pub fn setEnd(self: *Self, end: usize) Target.lead.Error!void {
            std.debug.assert(end <= self.capacity);
            if (self.lead) |lead| try lead.tell(@backingInt(Target.Op.flash_taps), &.{ @intCast(end), 0, 0 }, &.{});
            self.context_end = end;
        }

        /// `rank`'s rows of a gathered buffer, fp32 [block, dims]: `ranks.partial`'s place for a block.
        pub fn partial(gathered: cuda.Buffer, rank: usize) cuda.View {
            return Target.ranks.partial(gathered.view, rank, @intCast(gathered.view.size[1]));
        }

        /// Rows `first` onward, `total` of them, of each rank's taps, bf16 [rows, taps x dims] (the same bits
        /// on every rank), are the rows kept at the context's end onward (`Drafter.add_taps`). Nothing of an
        /// update crosses the ranks. Where `Target.wide`, a round's rows (a block's at most) are a captured
        /// graph where the pair has graphs, and a prompt's are read where they stand, a window's in one pass
        /// (spec 135 WS-02), whose graphs would be one a window's length and place; otherwise every update is
        /// `batch` rows copied in, captured where the pair has graphs.
        pub fn addTaps(self: *Self, taps: [held]cuda.Buffer, first: usize, total: usize) (Error || Allocator.Error || Target.lead.Error)!void {
            std.debug.assert(self.context_end + total <= self.capacity);
            if (self.lead) |lead| try lead.tell(@backingInt(Target.Op.flash_taps), &.{ @intCast(self.context_end), @intCast(first), @intCast(total) }, &.{});
            const graphs = if (!Target.wide or total <= self.ranks[0].c.block) self.graphs else null;
            const most = if (Target.wide and graphs == null) self.wides[0].rows else batch;
            var at: usize = 0;
            while (at < total) {
                const rows = @min(most, total - at);
                for (&self.ranks) |*rank| try rank.stage(self.context_end);
                if (graphs) |captured| try self.updateCaptured(captured, taps, first + at, rows) else if (Target.wide) try self.updateWide(taps, first + at, rows) else try self.update(taps, first + at, rows);
                self.context_end += rows;
                at += rows;
            }
        }

        /// A prompt's update: `rows` rows of the taps from row `from`, read in place, at the staged position.
        fn updateWide(self: *Self, taps: [held]cuda.Buffer, from: usize, rows: usize) Error!void {
            for (self.weights, &self.ranks, self.wides, taps) |w, *rank, one, source| {
                try contextOf(w, rank, source.span(from, rows), one.through(rows));
            }
        }

        /// An update's launches: `rows` rows of the taps from row `from`, at the staged position.
        fn update(self: *Self, taps: [held]cuda.Buffer, from: usize, rows: usize) Error!void {
            for (self.weights, &self.ranks, taps) |w, *rank, source| {
                try rank.tap_in.copyRows(0, source, from, rows);
                try context(w, rank, rows);
            }
        }

        /// The block after the context's end with `pending` first: each rank's `logits` and `selected` hold
        /// the drafts' rows after it, and `tops` every rank's candidates. Nothing of the context moves: the
        /// block's own K and V stand past its end, where the next update writes.
        pub fn block(self: *Self, main: [held]Main, pending: i32) (GatherError(Target) || Allocator.Error || Target.lead.Error)!void {
            if (self.lead) |lead| try lead.tell(@backingInt(Target.Op.flash_block), &.{ @intCast(self.context_end), @bitCast(pending) }, &.{});
            try self.stageBlock(pending);
            if (self.graphs) |graphs| try self.blockCaptured(graphs, main) else try self.compute(main);
        }

        /// What a block's launches read that no graph holds: the pending token and the context's end.
        pub fn stageBlock(self: *Self, pending: i32) Error!void {
            for (&self.ranks) |*rank| {
                try rank.ids.uploadLaunched(0, std.mem.asBytes(&pending));
                try rank.stage(self.context_end);
            }
        }

        /// A block's launches, from the staged ids and position.
        fn compute(self: *Self, main: [held]Main) GatherError(Target)!void {
            return self.computeWatched(main, {});
        }

        /// `compute`, stopping at every `Point` of every layer for `watch.at(point, layer)`, which may wait
        /// and read; `{}` watches nothing. Not inside a capture with a watcher that reads.
        pub fn computeWatched(self: *Self, main: [held]Main, watch: anytype) !void {
            std.debug.assert(self.gather.serves(held, world));
            for (&self.ranks, main) |*rank, m| try embed(rank, m.embed);
            for (0..self.weights[0].layers.len) |index| {
                for (self.weights, &self.ranks) |w, *rank| try attention(w, rank, index, partial(self.attn, rank.rank));
                try self.gather.run(self.attn.view);
                if (@TypeOf(watch) != void) try watch.at(Point.attention, index);
                for (self.weights, &self.ranks) |w, *rank| try mlp(w, rank, index, self.attn.view, partial(self.ffn, rank.rank));
                try self.gather.run(self.ffn.view);
                for (self.weights, &self.ranks) |w, *rank| try finish(w, rank, index, self.ffn.view);
                if (@TypeOf(watch) != void) try watch.at(Point.layer, index);
            }
            for (self.weights, &self.ranks, main) |w, *rank, m| try head(w, rank, m);
            try self.gather.run(self.tops.view);
        }

        fn updateCaptured(self: *Self, graphs: *Graphs, taps: [held]cuda.Buffer, from: usize, rows: usize) (Error || Allocator.Error)!void {
            const entry = try graphs.updates.getOrPut(graphs.gpa, .{ .rows = rows, .from = from, .taps = @intFromPtr(taps[0].view.ptr.?) });
            if (!entry.found_existing) {
                errdefer graphs.updates.removeByPtr(entry.key_ptr);
                try graphs.stream.begin();
                entry.value_ptr.* = try graphs.captured(self.update(taps, from, rows));
            }
            return entry.value_ptr.launch(graphs.stream);
        }

        fn blockCaptured(self: *Self, graphs: *Graphs, main: [held]Main) (GatherError(Target) || Allocator.Error)!void {
            if (graphs.whole == null) {
                try graphs.stream.begin();
                graphs.whole = try graphs.captured(self.compute(main));
                graphs.table = main[0].embed.ptr;
            }
            std.debug.assert(graphs.table == main[0].embed.ptr);
            return graphs.whole.?.launch(graphs.stream);
        }
    };
}

/// A pair's passes as captured graphs (`Drafter.capture`): one for the block, and one an update's (row count,
/// first row of the taps), captured when first run. A position is staged on the device before a launch, so a
/// graph holds none. They hold pointers into the pair, the taps and the main model: drop them first. Over
/// NCCL the block's graph holds its all-gathers, so both ranks launch it in step.
pub const Graphs = struct {
    stream: cuda.Stream,
    whole: ?cuda.Graph = null,
    /// What the block's graph was captured over.
    table: ?*anyopaque = null,
    updates: std.AutoHashMapUnmanaged(Key, cuda.Graph) = .empty,
    gpa: Allocator,

    const Key = struct { rows: usize, from: usize, taps: usize };

    pub fn init(gpa: Allocator) Error!Graphs {
        return .{ .stream = try .init(), .gpa = gpa };
    }

    pub fn deinit(self: *Graphs) void {
        if (self.whole) |*graph| graph.deinit();
        var graphs = self.updates.valueIterator();
        while (graphs.next()) |graph| graph.deinit();
        self.updates.deinit(self.gpa);
        self.stream.deinit();
        self.* = undefined;
    }

    /// The capture's end after `result`, whatever it is.
    fn captured(self: *Graphs, result: anytype) !cuda.Graph {
        result catch |err| {
            if (self.stream.end()) |graph| {
                var unused = graph;
                unused.deinit();
            } else |_| {}
            return err;
        };
        return self.stream.end();
    }
};
