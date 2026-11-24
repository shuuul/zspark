//! `zspark ranks-check`'s DFlash2 half (spec 045 WS-09): a random drafter beside the random GLM, the tests'
//! (`random_drafter.zig`), run over the ranks a process holds after the main model's windows. A round is an
//! update of the context from the last window's taps and a block pass; the first is launched phase by phase,
//! read at every gather, the rest go through the `draft.Drafter` as a decoder calls it, greedy and sampled,
//! and leave the drafts chosen. Everything the rank holds of a pass is read: both ranks' gathered partials of
//! every layer, the block's rows, the drafts' normed rows, its half of their logits, the projected rows, both
//! ranks' gathered candidates, its context's K and V, and on the host the merged candidates and the drafts.
//!
//! The gathered buffers are filled before every round, with other values on the two ways. The candidates'
//! fill is a small integer a word: a token of the vocabulary, so a table no gather brought reaches the
//! comparison and not an index past the codebooks.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const config_file = @import("../../../decode/dflash2/config.zig");
const drafter_file = @import("../dflash2/drafter.zig");
const drafter_testing = @import("random_drafter.zig");
const pass = @import("../dflash2/pass.zig");
const dflash2_pass = @import("../../../decode/dflash2/pass.zig");
const weights = @import("../../../decode/dflash2/weights.zig");
const held_file = @import("../layers/ranks_forward.zig");
const random_file = @import("random.zig");
const trace_file = @import("buffer_trace.zig");
const Trace = trace_file.Trace;

const world = config_file.world;
const fx = random_file.fx;
/// A block's rows: the pending token and seven drafts.
const block = 8;
/// Rounds through the `draft.Drafter` after the one launched phase by phase.
pub const rounds = 4;

/// The random drafter's tensors, its config and its codebooks, from a seed: the same on both ranks.
pub const Random = struct {
    c: config_file.Config,
    source: drafter_testing.Source,
    books: weights.Codebooks,

    /// For a main model tapped after `taps`.
    pub fn init(arena: Allocator, gpa: Allocator, seed: u64, taps: []const u32) !Random {
        var prng: std.Random.DefaultPrng = .init(seed ^ 0x67);
        var c = try drafter_testing.config(arena, fx.dims, fx.vocab, 5, taps);
        c.block = block;
        var source: drafter_testing.Source = .empty;
        try drafter_testing.put(arena, prng.random(), c, &source);
        const books: weights.Codebooks = try .load(gpa, source, &c);
        // random codebooks give an edge too small to move a pick: larger, as the tests', so the edges are held
        for (books.predecessor) |*value| value.* *= 12;
        return .{ .c = c, .source = source, .books = books };
    }

    pub fn deinit(self: *Random, gpa: Allocator) void {
        self.books.deinit(gpa);
    }
};

/// The drafter over `held` ranks beside a main model's.
pub fn Rig(comptime held: usize) type {
    return struct {
        const Self = @This();
        const H = held_file.Held(held);

        w: [held]weights.Weights,
        pair: pass.PairOf(held),
        d: drafter_file.DrafterOf(held),
        m: H.Model,
        b: *H.Buffers,
        /// Which of the held ranks is the one this process is.
        mine: usize,
        stale: []const u8,
        stale_tops: []const u32,

        /// In place: the drafter keeps the pair's address. `m` and `b` are the main model's, tapped as the
        /// drafter's config says; `rows` lends the model's scratch.
        pub fn init(self: *Self, arena: Allocator, gpa: Allocator, model: *const Random, m: H.Model, b: *H.Buffers, pool: *cuda.Scratch, capacity: usize, mine: usize) !void {
            const c = &model.c;
            const rows: cuda.Scratch.Rows = .{ .pool = pool, .count = dflash2_pass.batch };
            var done: usize = 0;
            errdefer for (self.w[0..done]) |*one| one.deinit();
            for (&self.w, m.ranks) |*one, rank| {
                one.* = try .load(arena, gpa, model.source, c, rank.rank, .mixed, rows);
                done += 1;
            }
            self.pair = try .initFrom(gpa, self.w, c, capacity, false, fx.vocab / world, m.ranks[0].rank, m.gather);
            errdefer self.pair.deinit();
            self.d = try .init(gpa, c, &self.pair, .{ .m = m, .main = b }, model.books, .{ .seed = 0, .temperature = 0 });
            self.d.cap = block - 1;
            self.d.confidence = 0;
            self.m = m;
            self.b = b;
            self.mine = mine;
            const stale = try arena.alloc(u8, self.pair.attn.bytes());
            @memset(stale, if (held == world) 0x3b else 0x3c);
            self.stale = stale;
            const stale_tops = try arena.alloc(u32, self.pair.tops.view.count());
            @memset(stale_tops, if (held == world) 1 else 2);
            self.stale_tops = stale_tops;
        }

        pub fn deinit(self: *Self) void {
            self.d.deinit();
            self.pair.deinit();
            for (&self.w) |*one| one.deinit();
        }

        fn spoil(self: *Self) !void {
            try cuda.synchronize();
            try self.pair.attn.upload(self.stale);
            try self.pair.ffn.upload(self.stale);
            try self.pair.tops.upload(std.mem.sliceAsBytes(self.stale_tops));
        }

        const Watch = struct {
            rig: *Self,
            trace: *Trace,

            pub fn at(self: *const Watch, point: dflash2_pass.Point, layer: usize) !void {
                switch (point) {
                    .attention => try self.trace.take(.drafter_attention, layer, self.rig.pair.attn.view),
                    .layer => {
                        try self.trace.take(.drafter_mlp, layer, self.rig.pair.ffn.view);
                        try self.trace.take(.drafter_rows, layer, self.rig.pair.ranks[self.rig.mine].x.view);
                    },
                }
            }
        };

        /// What a block pass left on this rank.
        fn ends(self: *Self, trace: *Trace) !void {
            const rank = self.pair.ranks[self.mine];
            trace.rows = block;
            try trace.take(.drafter_attention, null, self.pair.attn.view);
            try trace.take(.drafter_mlp, null, self.pair.ffn.view);
            try trace.take(.drafter_rows, null, rank.x.view);
            try trace.take(.drafter_final, null, rank.final.view);
            try trace.take(.drafter_logits, null, rank.logits.view);
            try trace.take(.drafter_selected, null, rank.selected.view);
            // a rank's rows of the table are its values', then its tokens'
            trace.rows = 2 * (block - 1);
            try trace.take(.drafter_tops, null, self.pair.tops.view);
            trace.rows = block;
            for (rank.k, rank.v) |k, v| {
                try trace.take(.drafter_k, null, k.view);
                try trace.take(.drafter_v, null, v.view);
            }
        }

        /// After the main model's windows, whose last ran `tokens`: `step` is the trace's first step here.
        /// The first round is launched and read at every gather, or with `graphs` one launch of each graph;
        /// the others are the drafter's own (`draft.Drafter.draft`), every other one sampled.
        pub fn run(self: *Self, trace: *Trace, step: usize, tokens: []const i32, graphs: ?*dflash2_pass.Graphs) !void {
            const pair = &self.pair;
            pair.reset();
            pair.graphs = graphs;
            // an empty context's rows, whatever a run before left past its end
            try cuda.synchronize();
            for (pair.ranks) |one| for (one.k, one.v) |k, v| {
                try k.zero();
                try v.zero();
            };
            self.d.full = false;
            var taps: [held]cuda.Buffer = undefined;
            for (&taps, self.b.ranks) |*one, rank| one.* = rank.taps.?;
            const mains = self.d.mains();
            const pending = tokens[tokens.len - 1];

            trace.step = step;
            trace.rows = tokens.len;
            try self.spoil();
            try pair.addTaps(taps, 0, tokens.len);
            const rank = &pair.ranks[self.mine];
            try trace.take(.drafter_context, null, rank.ctx.rows(tokens.len));
            try trace.take(.drafter_kv_rows, null, rank.kv_rows.rows(tokens.len));
            trace.rows = block;
            if (graphs == null) {
                try pair.stageBlock(pending);
                try pair.computeWatched(mains, &Watch{ .rig = self, .trace = trace });
            } else try pair.block(mains, pending);
            try self.ends(trace);

            var out: [block]i32 = undefined;
            const drafter = self.d.drafter();
            for (0..rounds) |round| {
                trace.step = step + 1 + round;
                trace.rows = block;
                self.d.settings = if (round % 2 == 0) .{ .seed = 0, .temperature = 0 } else .{ .seed = 7 + round, .temperature = 0.8 };
                // the drafter is asked for fewer drafts in the later rounds, as a reply's end asks
                const asked = block - 1 - round;
                try self.spoil();
                const n = try drafter.draft(drafter.context, self.b.ranks[0].final.rows(tokens.len), tokens, pair.context_end + tokens.len, null, out[0..asked]);
                try self.ends(trace);
                const top = self.d.c.top_k;
                trace.rows = asked;
                try trace.put(.drafter_candidates, std.mem.sliceAsBytes(self.d.tokens[0 .. asked * top]), @sizeOf(u32), top);
                try trace.put(.drafter_values, std.mem.sliceAsBytes(self.d.values[0 .. asked * top]), @sizeOf(f64), top);
                try trace.put(.drafter_drafts, std.mem.sliceAsBytes(out[0..n]), @sizeOf(i32), asked);
            }
            pair.graphs = null;
        }
    };
}
