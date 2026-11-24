//! The MTP layer drafting for several sequences at once (`hall.zig`, spec 068 WS-04): each slot's entries of
//! the layer (a seat), and a step that takes rows of several seats as one window, so a round's exchanges for
//! its drafts are a chain's depth and not the streams' count. `mtp.PairOf.step` is one sequence's; this is
//! the same phases with what reads a sequence (the latents written and attended) once a seat, as
//! `forward.attentionOf` runs the main model's DSA. A row's bits are those of its sequence alone, so a
//! stream's drafts beside others are the ones it has alone (`hall_test.zig` holds the counts).
//!
//! `draft` is the chains: every stream's rows kept absorbed in one step, then a step a draft of the chains
//! still running, each chosen as `mtp.HeadOf` chooses a session's (`choose`: the row's start on the device,
//! the whole row where that does not settle it, a row at a time). TensorFold 1.0.2's host does the same for
//! GLM on Metal (`families/glm/slots.zig`, `draftAll`: one absorb over every stream's rows, then a step a
//! level over the streams drafting on). It plans each stream's depth before and tells rank 1 the whole of it
//! once; here a chain ends by its drafts' confidence and its constraint, known a step at a time, so rank 1 is
//! told a step at a time, and the scored rows come first in a step where its deepest streams do.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const picking = @import("../../../decode/pick.zig");
const sampling = @import("../../../decode/sampling.zig");
const dsa = @import("../layers/dsa.zig");
const hc = @import("../layers/hc.zig");
const held_file = @import("../layers/ranks_forward.zig");
const layout = @import("../layers/layout.zig");
const model = @import("../layers/model.zig");
const mtp = @import("mtp.zig");
const Job = @import("../draft.zig").Job;
const Picked = picking.Picked;

const world = layout.world;

/// A sequence's rows in a step: `rows` of them as entries `at` onward of slot `slot`'s seat (0 starts the
/// sequence, whose first embedding is zero), each the hidden row `hidden` names and the token after it.
/// `ask`: the rank its last row's start is asked to; 0 scores nothing. The scored come first in a step.
pub const Item = struct {
    slot: usize,
    at: usize,
    rows: usize,
    hidden: Hidden,
    ask: u32,

    /// Rows `main` onward of the main model's final rows, or row `own` of the last step's hidden rows (the
    /// item's place in that step).
    pub const Hidden = union(enum) { main: usize, own: usize };
};

/// The words a step of `items` items takes in a frame told to the other rank, before its tokens.
pub fn told(items: usize) usize {
    return 1 + 5 * items + 2;
}

pub fn SeatsOf(comptime held: usize) type {
    return struct {
        const Self = @This();
        const main_model = held_file.Held(held);
        const Pair = mtp.PairOf(held);

        gpa: Allocator,
        capacity: usize,
        /// A rank's entries, a slot each: a latent a position absorbed, then one a chained draft.
        caches: [held][]dsa.Cache,
        /// A rank's last step: bf16 [slots, dims] each item's last hidden row, [slots, vocab / world] the scored
        /// rows' logits, u32 [slots, `Picked.wordsFor`] their starts.
        finals: [held]cuda.Buffer,
        logits: [held]cuda.Buffer,
        picked: [held]cuda.Buffer,
        /// With one rank here: every rank's starts, [world, slots, words], used tight for a step's scored rows.
        picks: ?cuda.Buffer = null,
        /// The rank each row of `picked` is asked to.
        asked: []u32,
        /// The last step's items, and the scored among them.
        ran: usize = 0,
        scored: usize = 0,
        /// `draft`'s steps.
        items: []Item,
        tokens: []i32,

        /// `seats` seats of `capacity` entries over `m`.
        pub fn init(gpa: Allocator, m: main_model.Model, seats: usize, capacity: usize) (cuda.Error || Allocator.Error)!Self {
            var self: Self = .{ .gpa = gpa, .capacity = capacity, .caches = undefined, .finals = undefined, .logits = undefined, .picked = undefined, .asked = &.{}, .items = &.{}, .tokens = &.{} };
            self.asked = try gpa.alloc(u32, seats);
            errdefer gpa.free(self.asked);
            @memset(self.asked, 1);
            self.items = try gpa.alloc(Item, seats);
            errdefer gpa.free(self.items);
            self.tokens = try gpa.alloc(i32, m.max_rows);
            errdefer gpa.free(self.tokens);
            const words = Picked.wordsFor(m.ranks[0].head.outputs());
            var ranks: usize = 0;
            errdefer for (0..ranks) |rank| self.drop(rank);
            for (m.ranks, 0..) |rank, index| {
                const caches = try gpa.alloc(dsa.Cache, seats);
                errdefer gpa.free(caches);
                var done: usize = 0;
                errdefer for (caches[0..done]) |*one| one.deinit();
                for (caches) |*one| {
                    one.* = try .holding(1, capacity, rank.indexDim());
                    done += 1;
                }
                var finals: cuda.Buffer = try .init(.bf16, &.{ seats, rank.dims });
                errdefer finals.deinit();
                var logits: cuda.Buffer = try .init(.bf16, &.{ seats, rank.head.outputs() });
                errdefer logits.deinit();
                var picked: cuda.Buffer = try .init(.u32, &.{ seats, words });
                errdefer picked.deinit();
                try picked.zero();
                for (0..seats) |row| try Picked.ask(.{ .view = picked.span(row, 1) }, 1);
                self.caches[index] = caches;
                self.finals[index] = finals;
                self.logits[index] = logits;
                self.picked[index] = picked;
                ranks += 1;
            }
            if (held < world) self.picks = try .init(.u32, &.{ world, seats, words });
            return self;
        }

        fn drop(self: *Self, rank: usize) void {
            for (self.caches[rank]) |*one| one.deinit();
            self.gpa.free(self.caches[rank]);
            self.finals[rank].deinit();
            self.logits[rank].deinit();
            self.picked[rank].deinit();
        }

        pub fn deinit(self: *Self) void {
            if (self.picks) |*one| one.deinit();
            for (0..held) |rank| self.drop(rank);
            self.gpa.free(self.asked);
            self.gpa.free(self.items);
            self.gpa.free(self.tokens);
            self.* = undefined;
        }

        pub fn count(self: Self) usize {
            return self.asked.len;
        }

        /// The device bytes a seat takes on rank `m`: its entries and its row of a step's buffers.
        pub fn bytes(m: model.Model, capacity: usize) usize {
            const words = Picked.wordsFor(m.head.outputs());
            return dsa.Cache.bytes(1, capacity, m.indexDim()) + m.dims * 2 + m.head.outputs() * 2 + words * 4 * (if (held < world) 1 + world else 1);
        }

        fn half(cat: cuda.View, dims: usize, first: usize, rows: usize, side: usize) cuda.View {
            const base: [*]u8 = @ptrCast(cat.ptr.?);
            var view: cuda.View = .init(@ptrCast(base + (first * 2 * dims + side * dims) * 2), .bf16, &.{ rows, dims });
            view.row_stride = @intCast(2 * dims);
            return view;
        }

        /// `items`' rows as one window over the layer, `tokens` theirs in order: each seat's entries `at`
        /// onward written and kept, the scored items' last rows scored (`picked`, `wholeRow`). The slots are
        /// distinct. `main` as `mtp.PairOf.step`'s: its final rows are read, its scratch computed in.
        pub fn step(self: *Self, pair: *Pair, m: main_model.Model, main: *main_model.Buffers, items: []const Item, tokens: []const i32) held_file.Error!void {
            const rows = tokens.len;
            var scored: usize = 0;
            var total: usize = 0;
            for (items, 0..) |item, index| {
                std.debug.assert(item.rows >= 1 and item.slot < self.count() and item.at + item.rows <= self.capacity and item.ask <= Picked.most);
                for (items[0..index]) |before| std.debug.assert(before.slot != item.slot);
                std.debug.assert(item.ask == 0 or scored == index);
                scored += @intFromBool(item.ask != 0);
                total += item.rows;
            }
            std.debug.assert(items.len >= 1 and items.len <= self.count() and total == rows and rows <= pair.b[0].rows);
            if (pair.lead) |lead| {
                var words: [lead_most]u32 = undefined;
                const used = told(items.len);
                std.debug.assert(1 + used + rows <= lead_most);
                words[0] = @intCast(items.len);
                for (items, 0..) |item, index| words[1 + 5 * index ..][0..5].* = .{
                    @intCast(item.slot), @intCast(item.at), @intCast(item.rows),
                    switch (item.hidden) {
                        .main => |from| @intCast(from << 1),
                        .own => |row| @intCast(row << 1 | 1),
                    },
                    item.ask,
                };
                words[used - 2 ..][0..2].* = held_file.words(try main_model.picks(main.*));
                try lead.tell(@backingInt(held_file.Op.mtp_steps), words[0..used], tokens);
            }
            // the picks in the main model's scratch are this step's from here
            main.window = rows;
            const attn = held_file.gathered(main.attn, rows);
            const ffn = held_file.gathered(main.ffn, rows);
            for (m.ranks, pair.weights, &pair.b, main.ranks, self.caches, self.finals) |rank, w, *b, scratch, caches, finals| {
                try b.ids.uploadLaunched(0, std.mem.sliceAsBytes(tokens));
                const me = b.me.rows(rows);
                try cuda.launched(cuda.kernels.zs_embed(&b.ids.rows(rows), &rank.embed.view, &me, 1));
                const cat = b.cat.rows(rows);
                var first: usize = 0;
                for (items) |item| {
                    if (item.at == 0) try cuda.zeroLaunched(b.me.span(first, 1));
                    first += item.rows;
                }
                try hc.rmsnorm(me, w.enorm.view, half(cat, rank.dims, 0, rows, 0), w.eps);
                first = 0;
                for (items) |item| {
                    const hidden = switch (item.hidden) {
                        .main => |from| scratch.final.span(from, item.rows),
                        .own => |row| finals.span(row, 1),
                    };
                    try hc.rmsnorm(hidden, w.hnorm.view, half(cat, rank.dims, first, item.rows, 1), w.eps);
                    first += item.rows;
                }
                const x = b.x.rows(rows);
                try w.eh.forward(cat, x);
                const normed = b.normed.rows(rows);
                try hc.rmsnorm(x, w.in_norm.view, normed, w.eps);
                const buffers = scratch.dsa.?;
                try w.attention.project(buffers, normed);
                try w.attention.indexProject(buffers, normed);
                first = 0;
                for (items) |item| {
                    const cache = &caches[item.slot];
                    if (cache.pos != item.at) try cache.setPos(item.at);
                    try w.attention.write(cache.*, 0, buffers, first, item.rows, item.at + item.rows);
                    try w.attention.indexRows(cache.*, 0, buffers, first, item.rows, item.at + item.rows);
                    first += item.rows;
                }
                try dsa.absorb(buffers.q.rows(rows), w.attention.kv_b.view, w.attention.key_dim, buffers.qa.rows(rows));
                first = 0;
                for (items) |item| {
                    try w.attention.attend(caches[item.slot], 0, buffers, first, item.rows, item.at + item.rows);
                    first += item.rows;
                }
                try w.attention.emit(buffers, rows, held_file.partial(main.attn, rank.rank, rows));
            }
            try m.gather.run(attn);
            for (m.ranks, pair.weights, &pair.b, main.ranks) |rank, w, *b, scratch|
                try mtp.moe(w, b, scratch.moe.?, rows, attn, held_file.partial(main.ffn, rank.rank, rows));
            try m.gather.run(ffn);
            for (self.asked[0..scored], items[0..scored], 0..) |*asked, item, row| {
                if (asked.* == item.ask) continue;
                for (self.picked) |picked| try Picked.ask(.{ .view = picked.span(row, 1) }, item.ask);
                asked.* = item.ask;
            }
            for (m.ranks, pair.weights, &pair.b, self.caches, self.finals, self.logits, self.picked, 0..) |rank, w, *b, caches, finals, logits, picked, index| {
                try hc.residualAdd(b.x.rows(rows), ffn);
                var first: usize = 0;
                for (items, 0..) |item, row| {
                    first += item.rows;
                    try hc.rmsnorm(b.x.span(first - 1, 1), w.norm.view, finals.span(row, 1), w.eps);
                    try caches[item.slot].setPos(item.at + item.rows);
                }
                if (scored == 0) continue;
                const head: mtp.Linear = if (pair.heads) |heads| heads[index] else rank.head;
                try head.forward(finals.rows(scored), logits.rows(scored));
                for (0..scored) |row| try picking.pick(logits.span(row, 1), picked.span(row, 1));
            }
            self.ran = items.len;
            self.scored = scored;
            if (scored == 0) return;
            switch (m.gather) {
                .loopback => {},
                .nccl => |comm| try comm.allGather(self.picked[0].rows(scored), self.gatheredPicks()),
            }
        }

        fn gatheredPicks(self: Self) cuda.View {
            const whole = self.picks.?.view;
            return .init(whole.ptr.?, .u32, &.{ world, self.scored, @intCast(whole.size[2]) });
        }

        /// What the last step found of its scored row `row` in each rank's half of the vocabulary, rank 0's
        /// first. Nothing is told: the step gathered them on every rank.
        fn pickedRow(self: Self, row: usize, out: *[world][Picked.words]u32) cuda.Error!void {
            std.debug.assert(row < self.scored);
            try cuda.synchronize();
            const words: usize = @intCast(self.picked[0].view.size[1]);
            for (out, 0..) |*raw, rank| {
                if (self.picks) |whole| {
                    try whole.downloadAt((rank * self.scored + row) * words * @sizeOf(u32), std.mem.sliceAsBytes(raw));
                } else try self.picked[rank].downloadAt(row * words * @sizeOf(u32), std.mem.sliceAsBytes(raw));
            }
        }

        /// The last step's scored row `row` over the vocabulary, wherever the ranks are.
        pub fn wholeRow(self: Self, pair: *Pair, main: main_model.Buffers, row: usize, out: []u16) held_file.Error!void {
            std.debug.assert(row < self.scored);
            if (pair.lead) |lead| try lead.tell(@backingInt(held_file.Op.mtp_row_of), &.{@intCast(row)}, &.{});
            if (held < world) return main_model.gatherRow(main, self.logits[0].span(row, 1), out);
            try cuda.synchronize();
            const part = out.len / world;
            for (self.logits, pair.weights) |logits, w| try logits.downloadAt(row * part * 2, std.mem.sliceAsBytes(out[w.rank * part ..][0..part]));
        }

        /// A job's scored row as `mtp.HeadOf.choose` reads it.
        const Row = struct {
            seats: *const Self,
            pair: *Pair,
            main: *main_model.Buffers,
            row: usize,

            pub fn picked(self: Row, out: *[world][Picked.words]u32) cuda.Error!void {
                return self.seats.pickedRow(self.row, out);
            }

            pub fn whole(self: Row, out: []u16) held_file.Error!void {
                return self.seats.wholeRow(self.pair, self.main.*, self.row, out);
            }
        };

        /// Every job's drafts, as `head` drafts one stream's (`mtp.HeadOf`'s `drafts`): the rows kept of all
        /// of them absorbed in one step, then a step a draft over the chains that go on. `head`'s sampling and
        /// ends are each job's own while its draft is chosen.
        pub fn draft(self: *Self, head: *mtp.HeadOf(held), jobs: []Job) !void {
            std.debug.assert(jobs.len <= self.count());
            var used: usize = 0;
            var rows: usize = 0;
            for (jobs) |*job| {
                job.count = 0;
                job.live = false;
                if (job.full.*) continue;
                // every position is absorbed in order, so a seat's entries are the sequence's positions
                if (job.position > self.capacity) {
                    job.full.* = true;
                    continue;
                }
                job.chain = try .begin(job.out, job.guide);
                // a job granted no row of the next window (`out` empty): live only so its kept rows absorb;
                // its closed chain scores nothing and halts below, no step of its own (TensorFold #1779)
                job.live = true;
            }
            // the scored first, as a step packs their rows
            for ([_]bool{ true, false }) |scored| for (jobs) |*job| {
                if (!job.live or job.chain.scores() != scored) continue;
                head.settings = job.settings;
                self.items[used] = .{ .slot = job.slot, .at = job.position - job.kept.len, .rows = job.kept.len, .hidden = .{ .main = job.from }, .ask = if (scored) head.rankAsked() else 0 };
                @memcpy(self.tokens[rows..][0..job.kept.len], job.kept);
                job.index = used;
                used += 1;
                rows += job.kept.len;
            };
            while (used > 0) {
                try self.step(head.pair, head.m, head.main, self.items[0..used], self.tokens[0..rows]);
                used = 0;
                for (jobs) |*job| {
                    if (!job.live) continue;
                    head.settings = job.settings;
                    head.ends = job.ends;
                    const chain = &job.chain;
                    while (job.live) switch (try chain.step(job.guide)) {
                        .halt => job.live = false,
                        .run => {
                            const at = job.position + chain.ran;
                            if (at == self.capacity) {
                                job.live = false;
                                break;
                            }
                            self.items[used] = .{ .slot = job.slot, .at = at, .rows = 1, .hidden = .{ .own = job.index }, .ask = head.rankAsked() };
                            self.tokens[used] = job.out[chain.ran];
                            break;
                        },
                        .choose => if (!try head.choose(Row{ .seats = self, .pair = head.pair, .main = head.main, .row = job.index }, job.position, job.guide, chain)) {
                            job.live = false;
                        },
                    };
                    job.count = chain.count;
                    if (!job.live) continue;
                    // its step is the next one, where its row is the `used`-th
                    chain.ran += 1;
                    job.index = used;
                    used += 1;
                }
                rows = used;
            }
        }
    };
}

const lead_most = @import("../../../ranks/lead.zig").Lead.most;
