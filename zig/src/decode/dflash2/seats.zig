//! DFlash2 drafting for several sequences at once (a model's `pool/hall.zig`, spec 068 WS-04): each slot's
//! context of the drafter (a seat: every layer's K and V and the position its launches stand at), and a round
//! that updates the contexts of the rows the main model kept and then runs every stream's block as one pass, so
//! a round has a block's eleven gathers whatever its streams. `pass.zig`'s phases are one block's; these are
//! the same kernels over the blocks side by side, with what reads a block as a whole (the convolutions, whose
//! rows read the row before, and the attention over a context) launched once a block. A block's bits are
//! those it has alone, so a stream's drafts beside others are its own (`pool/hall_test.zig` holds the counts).
//!
//! The passes are launched from the host: a graph holds a context's address. TensorFold gives no rule here:
//! 0.6.6's CUDA drafter is one stream's, and 1.0.2's host serves GLM's streams with the MTP layer alone.
//! `Target` is the model's (`pass.zig`'s header); two-rank replay is its mirror's, the loopback tests the
//! reference.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const drafter_file = @import("drafter.zig");
const kernels = @import("kernels.zig");
const pass = @import("pass.zig");
const Weights = @import("weights.zig").Weights;

const world = pass.world;
const glue = pass.glue;
const no_limit: f32 = 1e30;

/// A sequence's part of a round: `rows` rows of the main model's taps from row `from`, kept at positions
/// `at` onward of slot `slot`'s context; then, with `pending`, its block after them, that token first.
pub const Item = struct { slot: usize, at: usize, from: usize, rows: usize, pending: ?i32 };

/// The words a round of `items` items takes in a frame told to the other rank.
pub fn told(items: usize) usize {
    return 1 + 5 * items;
}

pub fn SeatsOf(comptime held: usize, comptime Target: type) type {
    return struct {
        const Self = @This();
        const Pair = pass.PairOf(held, Target);
        const ranks_file = Target.ranks;
        const lead_file = Target.lead;

        pub const Seat = struct { k: [held][]cuda.Buffer, v: [held][]cuda.Buffer, pos: [held]cuda.Buffer };

        gpa: Allocator,
        seats: []Seat,
        /// A rank's buffers for every seat's block side by side, and an update's rows.
        wide: [held]pass.Rank,
        /// fp32 [world, seats x block, dims], used tight for a round's blocks, and every rank's candidates of
        /// them, [world, 2 x seats x (block - 1), top_k].
        attn: cuda.Buffer,
        ffn: cuda.Buffer,
        tops: cuda.Buffer,
        /// The last pass's blocks: whose slot each is.
        order: []usize,
        blocks: usize = 0,
        /// `draft`'s round.
        items: []Item,

        /// `count` contexts shaped as `pair`'s own.
        pub fn init(gpa: Allocator, pair: *const Pair, count: usize) (cuda.Error || Allocator.Error)!Self {
            std.debug.assert(pair.graphs == null and count >= 1);
            const c = pair.ranks[0].c;
            var self: Self = .{ .gpa = gpa, .seats = &.{}, .wide = undefined, .attn = undefined, .ffn = undefined, .tops = undefined, .order = &.{}, .items = &.{} };
            self.order = try gpa.alloc(usize, count);
            errdefer gpa.free(self.order);
            self.items = try gpa.alloc(Item, count);
            errdefer gpa.free(self.items);
            self.attn = try .init(.f32, &.{ world, count * c.block, c.hidden });
            errdefer self.attn.deinit();
            self.ffn = try .init(.f32, &.{ world, count * c.block, c.hidden });
            errdefer self.ffn.deinit();
            self.tops = try .init(.f32, &.{ world, 2 * count * (c.block - 1), c.top_k });
            errdefer self.tops.deinit();
            var ranks: usize = 0;
            errdefer for (self.wide[0..ranks]) |*one| one.deinit();
            for (&self.wide, pair.ranks) |*one, rank| {
                // its own context is no sequence's: a row and a block
                one.* = try .init(gpa, c, 1, false, @intCast(rank.logits.view.size[1]), rank.rank, ranks_file.partial(self.tops.view, rank.rank, 2 * count * (c.block - 1)), count);
                if (c.window == null) {
                    // its attention is over a seat's context, `pair`'s cap of rows, not its own block's
                    const parts: cuda.Buffer = try .init(.f32, &.{kernels.attendParts(c.rankHeads(), c.block, rank.cap)});
                    one.parts.deinit();
                    one.parts = parts;
                }
                ranks += 1;
            }
            const seats = try gpa.alloc(Seat, count);
            errdefer gpa.free(seats);
            var done: usize = 0;
            errdefer for (seats[0..done]) |*one| drop(gpa, one, held);
            for (seats) |*one| {
                one.* = .{ .k = undefined, .v = undefined, .pos = undefined };
                var made: usize = 0;
                errdefer drop(gpa, one, made);
                for (&one.k, &one.v, &one.pos, pair.ranks) |*k, *v, *pos, rank| {
                    k.* = try like(gpa, rank.k);
                    errdefer free(gpa, k.*);
                    v.* = try like(gpa, rank.v);
                    errdefer free(gpa, v.*);
                    pos.* = try .init(.i32, &.{1});
                    made += 1;
                }
                done += 1;
            }
            self.seats = seats;
            return self;
        }

        fn like(gpa: Allocator, layers: []const cuda.Buffer) (cuda.Error || Allocator.Error)![]cuda.Buffer {
            const out = try gpa.alloc(cuda.Buffer, layers.len);
            errdefer gpa.free(out);
            var done: usize = 0;
            errdefer for (out[0..done]) |*one| one.deinit();
            for (out, layers) |*one, layer| {
                const size = layer.view.size;
                one.* = try .init(.bf16, &.{ @intCast(size[0]), @intCast(size[1]), @intCast(size[2]) });
                done += 1;
                try one.zero();
            }
            return out;
        }

        fn free(gpa: Allocator, layers: []cuda.Buffer) void {
            for (layers) |*one| one.deinit();
            gpa.free(layers);
        }

        fn drop(gpa: Allocator, one: *Seat, ranks: usize) void {
            for (one.k[0..ranks], one.v[0..ranks], one.pos[0..ranks]) |k, v, *pos| {
                free(gpa, k);
                free(gpa, v);
                pos.deinit();
            }
        }

        pub fn deinit(self: *Self) void {
            for (self.seats) |*one| drop(self.gpa, one, held);
            self.gpa.free(self.seats);
            for (&self.wide) |*one| one.deinit();
            self.attn.deinit();
            self.ffn.deinit();
            self.tops.deinit();
            self.gpa.free(self.order);
            self.gpa.free(self.items);
            self.* = undefined;
        }

        /// Seat `slot`'s context zeroed, as `pass.PairOf.startAt` zeroes a pair's: a sequence that goes on from
        /// a position past its first reads zeros where the positions before it would stand, and no other
        /// sequence's rows, until the window back is absorbed.
        pub fn clear(self: *Self, slot: usize) cuda.Error!void {
            const seat = self.seats[slot];
            for (seat.k, seat.v) |k, v| for (k, v) |one_k, one_v| {
                try one_k.zero();
                try one_v.zero();
            };
        }

        /// The device bytes a seat takes on a rank: its context, and its block of a pass's buffers.
        pub fn bytes(pair: *const Pair) usize {
            const rank = pair.ranks[0];
            var sum: usize = 4 + pair.attn.bytes() + pair.ffn.bytes() + pair.tops.bytes();
            for (rank.k, rank.v) |k, v| sum += k.bytes() + v.bytes();
            inline for (.{ "ids", "x", "normed", "mixed", "branch", "dyn", "q_rows", "q", "attended", "gate_up", "act", "final", "logits", "selected" }) |name| sum += @field(rank, name).bytes();
            return sum + rank.c.block * rank.kv_rows.bytes() / pass.batch;
        }

        fn stage(pos: cuda.Buffer, position: usize) cuda.Error!void {
            const at: i32 = @intCast(position);
            return pos.uploadLaunched(0, std.mem.asBytes(&at));
        }

        fn place(self: Self, pair: *const Pair, slot: usize, rank: usize) kernels.Place {
            return .{ .pos = self.seats[slot].pos[rank].view, .ring = pair.ranks[rank].ring };
        }

        /// A round: every item's rows into its seat's context, then the blocks of those that ask one as one
        /// pass, in the items' order (`source`). `taps` are the main model's of its last window. The slots are
        /// distinct. Returns the blocks run.
        pub fn round(self: *Self, pair: *Pair, taps: [held]cuda.Buffer, main: [held]pass.Main, items: []const Item) (pass.GatherError(Target) || Allocator.Error || lead_file.Error)!usize {
            std.debug.assert(items.len >= 1 and items.len <= self.seats.len and pair.gather.serves(held, world));
            if (pair.lead) |lead| {
                var words: [lead_file.Lead.most]u32 = undefined;
                const used = told(items.len);
                words[0] = @intCast(items.len);
                for (items, 0..) |item, index| words[1 + 5 * index ..][0..5].* = .{ @intCast(item.slot), @intCast(item.at), @intCast(item.from), @intCast(item.rows), if (item.pending) |token| @as(u32, @intCast(token)) + 1 else 0 };
                try lead.tell(@backingInt(Target.Op.flash_round), words[0..used], &.{});
            }
            const n = pair.ranks[0].c.block;
            var blocks: usize = 0;
            for (items, 0..) |item, index| {
                std.debug.assert(item.rows >= 1 and item.slot < self.seats.len and item.at + item.rows <= pair.capacity);
                for (items[0..index]) |before| std.debug.assert(before.slot != item.slot);
                const seat = self.seats[item.slot];
                // nothing of an update crosses the ranks
                var at: usize = 0;
                while (at < item.rows) {
                    const rows = @min(pass.batch, item.rows - at);
                    for (pair.weights, &self.wide, taps, 0..) |w, *r, tapped, rank| {
                        try stage(seat.pos[rank], item.at + at);
                        try r.tap_in.copyRows(0, tapped, item.from + at, rows);
                        try self.context(pair, w, r, item.slot, rank, rows);
                    }
                    at += rows;
                }
                for (pair.weights, 0..) |_, rank| try stage(seat.pos[rank], item.at + item.rows);
                const token = item.pending orelse continue;
                for (&self.wide) |*r| try r.ids.uploadLaunched(blocks * n * @sizeOf(i32), std.mem.asBytes(&token));
                self.order[blocks] = item.slot;
                blocks += 1;
            }
            self.blocks = blocks;
            if (blocks > 0) try self.compute(pair, main, blocks);
            return blocks;
        }

        /// `pass.context` on a seat: `rows` kept rows' taps, in `r.tap_in`, at the seat's staged position.
        fn context(self: Self, pair: *const Pair, w: Weights, r: *pass.Rank, slot: usize, rank: usize, rows: usize) cuda.Error!void {
            const c = r.c;
            const seat = self.seats[slot];
            const ctx = r.ctx.rows(rows);
            try w.fc.forward(r.tap_in.rows(rows), r.projected.rows(rows));
            try glue.rmsnorm(r.projected.rows(rows), w.hidden_norm.view, ctx, @floatCast(c.eps));
            for (w.layers, seat.k[rank], seat.v[rank]) |layer, k, v| {
                try layer.kv.forward(ctx, r.kv_rows.rows(rows));
                try kernels.prep(null, r.kv_rows.rows(rows), layer.q_norm.view, layer.k_norm.view, self.place(pair, slot, rank), null, k.view, v.view, @floatCast(c.eps), c.theta);
            }
        }

        /// The staged blocks through the layers, as `pass.PairOf.computeWatched` runs one.
        fn compute(self: *Self, pair: *Pair, main: [held]pass.Main, blocks: usize) pass.GatherError(Target)!void {
            const c = pair.ranks[0].c;
            const rows = blocks * c.block;
            const drafts = blocks * (c.block - 1);
            const attn = ranks_file.gathered(self.attn.view, rows);
            const ffn = ranks_file.gathered(self.ffn.view, rows);
            for (&self.wide, main) |*r, m| try cuda.launched(cuda.kernels.zs_embed(&r.ids.rows(rows), &m.embed, &r.x.rows(rows), 1));
            for (0..pair.weights[0].layers.len) |index| {
                for (pair.weights, &self.wide, 0..) |w, *r, rank| try self.attention(pair, w, r, rank, index, blocks, ranks_file.partial(self.attn.view, r.rank, rows));
                try pair.gather.run(attn);
                for (pair.weights, &self.wide) |w, *r| try mlp(w, r, index, blocks, attn, ranks_file.partial(self.ffn.view, r.rank, rows));
                try pair.gather.run(ffn);
                for (pair.weights, &self.wide) |w, *r| {
                    try glue.branch(ffn, r.branch.rows(rows));
                    try convolve(r, w.layers[index].m_base.view, blocks, r.branch, 1, r.x, r.x);
                }
            }
            const tops = ranks_file.gathered(self.tops.view, 2 * drafts);
            for (pair.weights, &self.wide, main) |w, *r, m| {
                for (0..blocks) |b| try glue.rmsnorm(r.x.span(b * c.block + 1, c.block - 1), w.norm.view, r.final.span(b * (c.block - 1), c.block - 1), @floatCast(c.eps));
                const final = r.final.rows(drafts);
                const logits = r.logits.rows(drafts);
                try m.head.forward(final, logits);
                const mine: [*]u8 = @ptrCast(ranks_file.partial(self.tops.view, r.rank, 2 * drafts).ptr.?);
                const values: cuda.View = .init(@ptrCast(mine), .f32, &.{ drafts, c.top_k });
                const tokens: cuda.View = .init(@ptrCast(mine + drafts * c.top_k * @sizeOf(f32)), .i32, &.{ drafts, c.top_k });
                try kernels.top(logits, values, tokens, r.picks.view, r.first_token);
                try pass.rescore(m, tokens, final, values, r.first_token);
                try w.projection.forward(final, r.selected.rows(drafts));
            }
            try pair.gather.run(tops);
        }

        /// A convolution over each block's rows: a row mixes with the row before it in its block alone.
        fn convolve(r: *pass.Rank, base: cuda.View, blocks: usize, x: cuda.Buffer, branch: u1, residual: ?cuda.Buffer, out: cuda.Buffer) cuda.Error!void {
            const n = r.c.block;
            for (0..blocks) |b| try kernels.dconv(x.span(b * n, n), r.dyn.span(b * n, n), base, branch, if (residual) |one| one.span(b * n, n) else null, out.span(b * n, n));
        }

        /// `pass.attention` over the blocks: each one's K and V written after its seat's context, and attended.
        fn attention(self: Self, pair: *const Pair, w: Weights, r: *pass.Rank, rank: usize, index: usize, blocks: usize, partial: cuda.View) cuda.Error!void {
            const c = r.c;
            const n = c.block;
            const rows = blocks * n;
            const layer = w.layers[index];
            const eps: f32 = @floatCast(c.eps);
            const heads = c.rankHeads();
            try glue.rmsnorm(r.x.rows(rows), layer.in_norm.view, r.normed.rows(rows), eps);
            try layer.a_kp.forward(r.normed.rows(rows), r.dyn.rows(rows));
            try convolve(r, layer.a_base.view, blocks, r.normed, 0, null, r.mixed);
            try layer.q.forward(r.mixed.rows(rows), r.q_rows.rows(rows));
            try layer.kv.forward(r.mixed.rows(rows), r.kv_rows.rows(rows));
            const q_all: [*]u8 = @ptrCast(r.q.view.ptr.?);
            for (self.order[0..blocks], 0..) |slot, b| {
                const seat = self.seats[slot];
                const at = self.place(pair, slot, rank);
                const q: cuda.View = .init(@ptrCast(q_all + b * heads * n * kernels.head * 2), .bf16, &.{ heads, n, kernels.head });
                const k = seat.k[rank][index].view;
                const v = seat.v[rank][index].view;
                try kernels.prep(r.q_rows.span(b * n, n), r.kv_rows.span(b * n, n), layer.q_norm.view, layer.k_norm.view, at, q, k, v, eps, c.theta);
                try kernels.attend(q, k, v, at, r.attended.span(b * n, n), r.parts.view, c.seen(@intCast(k.size[1])), c.causal);
            }
            try layer.o.forward(r.attended.rows(rows), partial);
        }

        /// `pass.mlp` over the blocks.
        fn mlp(w: Weights, r: *pass.Rank, index: usize, blocks: usize, gathered: cuda.View, partial: cuda.View) cuda.Error!void {
            const c = r.c;
            const rows = blocks * c.block;
            const layer = w.layers[index];
            try glue.branch(gathered, r.branch.rows(rows));
            try convolve(r, layer.a_base.view, blocks, r.branch, 1, r.x, r.x);
            try glue.rmsnorm(r.x.rows(rows), layer.post_norm.view, r.normed.rows(rows), @floatCast(c.eps));
            try layer.m_kp.forward(r.normed.rows(rows), r.dyn.rows(rows));
            try convolve(r, layer.m_base.view, blocks, r.normed, 0, null, r.mixed);
            try layer.gate_up.forward(r.mixed.rows(rows), r.gate_up.rows(rows));
            try glue.swiglu(r.gate_up.rows(rows), r.act.rows(rows), no_limit);
            try layer.down.forward(r.act.rows(rows), partial);
        }

        /// Where block `block` of the last pass left its candidates (`drafter.zig`'s `read`).
        pub fn source(self: *const Self, comptime D: type, block: usize) D.Source {
            const drafts = self.wide[0].c.block - 1;
            std.debug.assert(block < self.blocks);
            return .{ .tops = self.tops, .selected = self.wide[0].selected, .rows = self.blocks * drafts, .first = block * drafts };
        }

        /// Every job's drafts, as `d` drafts one stream's (`drafter.zig`'s `drafts`): the kept rows of all of
        /// them into their contexts and one pass of the blocks of those whose chain asks the drafter, then
        /// each chain from its block's candidates, by its job's own sampling and ends.
        pub fn draft(self: *Self, d: *drafter_file.DrafterOf(held, Target), jobs: []Target.Job) !void {
            const D = drafter_file.DrafterOf(held, Target);
            std.debug.assert(jobs.len <= self.seats.len);
            var used: usize = 0;
            var blocks: usize = 0;
            for (jobs) |*job| {
                job.count = 0;
                job.live = false;
                if (job.full.*) continue;
                // every position is absorbed in order, so a context's rows are the sequence's positions
                if (job.position > d.pair.capacity) {
                    job.full.* = true;
                    continue;
                }
                job.chain = try .begin(job.out[0..d.depthOf(job.out.len)], job.guide);
                // the block runs once a draft is the drafter's to choose: a chain of forced tokens costs none
                job.live = try D.advance(&job.chain, job.guide);
                self.items[used] = .{ .slot = job.slot, .at = job.position - job.kept.len, .from = job.from, .rows = job.kept.len, .pending = if (job.live) job.kept[job.kept.len - 1] else null };
                used += 1;
                job.count = job.chain.count;
                if (!job.live) continue;
                job.index = blocks;
                blocks += 1;
            }
            if (used == 0) return;
            _ = try self.round(d.pair, d.taps(), d.mains(), self.items[0..used]);
            for (jobs) |*job| {
                if (!job.live) continue;
                d.settings = job.settings;
                d.ends = job.ends;
                const made = try d.read(self.source(D, job.index), job.chain.out.len);
                try d.choices(&job.chain, job.guide, made, job.kept[job.kept.len - 1], job.position);
                job.count = job.chain.count;
            }
        }
    };
}
