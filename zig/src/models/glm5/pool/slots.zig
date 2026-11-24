//! Several sequences through GLM's forward in one window (spec 068 WS-04): what each keeps (`model.Seq`, a
//! slot a sequence) and a round that runs the rows of all of them as one window over the ranks held, so a
//! round's gathers do not grow with the sequences. TensorFold serves GLM's streams one forward a round the
//! same way (its host's lanes); the layout here is zspark's.
//!
//! A row's bits are those of its sequence alone (`forward.attentionOf`): the phases that read a row and
//! nothing else take the window whole, and the two that read a sequence (a KDA layer's chain, a DSA layer's
//! latents) run once a part on that sequence's state. `slots_test.zig` holds a sequence among others against
//! the same sequence in a `ranks_forward.zig` state of its own.
//!
//! The rows scored are packed first to last (`logitsRow`'s `at`), so the head is one product a round and, with
//! one rank here, its rows one gather.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const forms = @import("../../../decode/forms.zig");
const cache_file = @import("../../../decode/graphs.zig");
const dsa = @import("../layers/dsa.zig");
const hc = @import("../layers/hc.zig");
const held_file = @import("../layers/ranks_forward.zig");
const kda = @import("../layers/kda.zig");
const layout = @import("../layers/layout.zig");
const model = @import("../layers/model.zig");
const phases = @import("../layers/forward.zig");
const ranks_file = @import("../../../ranks/ranks.zig");
const nccl = @import("../../../ranks/nccl.zig");

pub const Error = held_file.Error;
pub const Score = phases.Score;
const Op = held_file.Op;
const world = layout.world;

/// A sequence's rows in a round, in the order the round's tokens come: `rows` of slot `slot`, scored as
/// `score` names.
pub const Part = struct {
    slot: usize,
    rows: usize,
    score: Score = .all,

    /// The rows of it the head scores.
    pub fn scored(self: Part) usize {
        return switch (self.score) {
            .all => self.rows,
            .last => 1,
            .none => 0,
        };
    }
};

/// The first row of a round that takes the quantized forms (spec 115, `decode/forms.zig` `formFrom`): the rows of
/// its parts before the first part of more than `forward.bf16_rows` rows.
pub fn formFrom(parts: []const Part) usize {
    return forms.formFrom(Part, parts, phases.bf16_rows);
}

/// The words a round's parts take in a frame told to the other rank, before its tokens.
pub fn told(parts: usize) usize {
    return 1 + 3 * parts + 2;
}

pub fn SlotsOf(comptime held: usize) type {
    return struct {
        const Self = @This();
        const H = held_file.Held(held);

        gpa: Allocator,
        /// A rank's sequences, a slot each.
        seqs: [held][]model.Seq,
        /// A rank's KDA window, which every sequence's chain runs in.
        windows: [held]?kda.State,
        /// A rank's latents, keys and gates, one store for every slot's extents. Null where the model has no indexer.
        stores: [held]?dsa.Store = @splat(null),
        capacity: usize,
        /// The last round: its parts, where each one's rows and scored rows start, and the rows scored.
        parts: []Part,
        spans: []phases.Span,
        starts: []usize,
        ran: usize = 0,
        scored: usize = 0,
        /// The most rows a round scores: what `whole` holds.
        most_scored: usize,
        /// With one rank here: every rank's scored rows, bf16 [world, most_scored, vocab / world], used tight
        /// for the rows of a round.
        whole: ?cuda.Buffer = null,

        /// `sequences` sequences of up to `capacity` positions over `m`; a round scores `most_scored` rows at most.
        pub fn init(gpa: Allocator, m: H.Model, sequences: usize, capacity: usize, most_scored: usize) !Self {
            std.debug.assert(sequences >= 1 and most_scored >= 1 and most_scored <= m.max_rows);
            var self: Self = .{ .gpa = gpa, .seqs = undefined, .windows = @splat(null), .capacity = capacity, .parts = &.{}, .spans = &.{}, .starts = &.{}, .most_scored = most_scored };
            self.parts = try gpa.alloc(Part, sequences);
            errdefer gpa.free(self.parts);
            self.spans = try gpa.alloc(phases.Span, sequences);
            errdefer gpa.free(self.spans);
            self.starts = try gpa.alloc(usize, sequences);
            errdefer gpa.free(self.starts);
            var ranks: usize = 0;
            errdefer for (self.seqs[0..ranks], self.windows[0..ranks], self.stores[0..ranks]) |seqs, *in, *store| {
                for (seqs) |*one| one.deinit();
                gpa.free(seqs);
                if (in.*) |*one| one.deinit();
                if (store.*) |*one| one.deinit();
            };
            const width = m.ranks[0].indexDim();
            for (m.ranks, &self.seqs, &self.windows, &self.stores) |rank, *seqs, *in, *store| {
                seqs.* = try gpa.alloc(model.Seq, sequences);
                errdefer gpa.free(seqs.*);
                var pool = try extents.Pool.init(gpa, .{ .extents = extents.wanted(capacity, sequences), .capacity = capacity, .slots = sequences });
                defer pool.deinit();
                if (width) |dim| store.* = try dsa.Store.init(rank.kv, rank.dsa_layers, pool.refs.len, dim);
                errdefer if (store.*) |*one| one.deinit();
                var done: usize = 0;
                errdefer for (seqs.*[0..done]) |*one| one.deinit();
                for (seqs.*) |*one| {
                    one.* = try model.Seq.bare(rank, capacity);
                    const chain = try pool.open();
                    try pool.grow(chain, capacity);
                    one.dsa = if (store.*) |*held_store|
                        try held_store.stream(capacity, chain.table.items)
                    else if (rank.dsa_layers > 0)
                        try dsa.Cache.holdingAs(rank.kv, rank.dsa_layers, capacity, null)
                    else
                        null;
                    done += 1;
                }
                in.* = try model.window(rank);
                ranks += 1;
            }
            errdefer if (self.whole) |*one| one.deinit();
            if (held < world) self.whole = try .init(.bf16, &.{ world, most_scored, m.ranks[0].head.outputs() });
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.whole) |*one| one.deinit();
            for (self.seqs, &self.windows, &self.stores) |seqs, *in, *store| {
                for (seqs) |*one| one.deinit();
                self.gpa.free(seqs);
                if (in.*) |*one| one.deinit();
                if (store.*) |*one| one.deinit();
            }
            self.gpa.free(self.parts);
            self.gpa.free(self.spans);
            self.gpa.free(self.starts);
            self.* = undefined;
        }

        /// The device bytes a slot takes on a rank, and what the slots share on it (the KDA window).
        pub fn slotBytes(m: model.Model, capacity: usize) usize {
            return model.Seq.bytes(m, capacity);
        }

        pub fn count(self: Self) usize {
            return self.seqs[0].len;
        }

        /// The positions slot `slot` has committed.
        pub fn pos(self: Self, slot: usize) usize {
            return self.seqs[0][slot].pos;
        }

        /// Slot `slot` is an empty sequence.
        pub fn reset(self: *Self, m: H.Model, slot: usize) Error!void {
            if (m.lead) |lead| try lead.tell(@backingInt(Op.slot_reset), &.{@intCast(slot)}, &.{});
            for (&self.seqs) |seqs| try seqs[slot].reset();
        }

        /// Runs `tokens` as one window: each part's rows, in order, at its slot's position onward, the slots
        /// distinct. Nothing is committed until `commit`.
        pub fn forward(self: *Self, m: H.Model, b: *H.Buffers, parts: []const Part, tokens: []const i32) Error!void {
            try self.stage(m, b, parts, tokens);
            try self.run(m, b, parts.len);
        }

        /// The round's telling and staging, every round: the parts into `spans`, the tokens into the window.
        fn stage(self: *Self, m: H.Model, b: *H.Buffers, parts: []const Part, tokens: []const i32) Error!void {
            std.debug.assert(parts.len >= 1 and parts.len <= self.parts.len and m.gather.serves(held, world));
            var rows: usize = 0;
            var scored: usize = 0;
            for (parts, 0..) |part, at| {
                try cuda.bound(part.rows >= 1 and part.slot < self.count() and self.pos(part.slot) + part.rows <= self.capacity, "glm5 pool: a part of {d} rows in slot {d} of {d} past a capacity of {d}", .{ part.rows, part.slot, self.count(), self.capacity });
                for (parts[0..at]) |before| std.debug.assert(before.slot != part.slot);
                self.parts[at] = part;
                self.spans[at] = .{ .slot = part.slot, .first = rows, .rows = part.rows, .prompt = part.score != .all };
                self.starts[at] = scored;
                rows += part.rows;
                scored += part.scored();
            }
            try cuda.bound(rows == tokens.len and rows <= m.max_rows and scored <= self.most_scored, "glm5 pool: a round of {d} rows ({d} tokens, at most {d}), {d} scored of {d}", .{ rows, tokens.len, m.max_rows, scored, self.most_scored });
            if (m.lead) |lead| {
                var words: [lead_most]u32 = undefined;
                const used = told(parts.len);
                std.debug.assert(1 + used + rows <= lead_most);
                words[0] = @intCast(parts.len);
                for (parts, 0..) |part, at| words[1 + 3 * at ..][0..3].* = .{ @intCast(part.slot), @intCast(part.rows), @backingInt(part.score) };
                words[used - 2 ..][0..2].* = held_file.words(try H.picks(b.*));
                try lead.tell(@backingInt(Op.slots_forward), words[0..used], tokens);
            }
            self.ran = parts.len;
            self.scored = scored;
            b.window = rows;
            try H.stage(b, tokens);
        }

        /// The staged round's compute: the embedding, the layers, the head over the rows scored. The parts
        /// and spans stand from `stage`.
        fn run(self: *Self, m: H.Model, b: *H.Buffers, taken: usize) Error!void {
            const parts = self.parts[0..taken];
            const rows = b.window;
            const scored = self.scored;
            const spans = self.spans[0..taken];
            // each part's form by its rows (spec 115): the parts of bf16 rows come first (`hall.zig` orders them),
            // the quantized from the first part of more than `bf16_rows`
            const from = formFrom(parts);
            for (m.ranks) |rank| phases.setForm(rank, from);
            defer for (m.ranks) |rank| phases.clearForm(rank);
            for (m.ranks, &b.ranks) |rank, *one| try phases.embed(rank, one, rows);
            // the partials bf16 where every row is quantized (FP4-Local, spec 116 WS-04); a round of both forms
            // keeps fp32, its quantized rows rounded as bf16 (`phases.partials`)
            const attn_in = held_file.partialsIn(m.ranks[0], b.attn, from);
            const ffn_in = held_file.partialsIn(m.ranks[0], b.ffn, from);
            const attn = ranks_file.gathered(attn_in, rows);
            const ffn = ranks_file.gathered(ffn_in, rows);
            // a prompt part's indexer rows at or past `index.split_from` split over the ranks, as
            // `ranks_forward.zig`'s prompt windows (spec 152 WS-10); the spans mark the parts
            const split: ?nccl.Comm = switch (m.gather) {
                .nccl => |comm| comm,
                .loopback => null,
            };
            if (H.splits(b.*, rows)) {
                for (0..m.ranks[0].layers.len) |index| {
                    try H.glue(m, b, rows, index, .attn, if (index == 0) null else ffn);
                    for (m.ranks, self.windows, self.seqs, &b.ranks) |rank, in, seqs, *one|
                        try phases.attendOfBody(rank, in, seqs, spans, one, index, rows, split);
                    try H.scattered(m, b, rows, attn_in, H.Emitted{ .m = m, .b = b, .index = index, .rows = rows, .chains = self.windows, .buffer = attn_in });
                    try H.glue(m, b, rows, index, .ffn, attn);
                    try H.feedScattered(m, b, index, rows, ffn_in);
                }
                try H.glueEnd(m, b, rows, ffn);
            } else for (0..m.ranks[0].layers.len) |index| {
                for (m.ranks, self.windows, self.seqs, &b.ranks) |rank, in, seqs, *one|
                    try phases.attentionOf(rank, in, seqs, spans, one, index, rows, ranks_file.partial(attn_in, rank.rank, rows), split);
                try m.gather.run(attn);
                for (m.ranks, &b.ranks) |rank, *one|
                    try phases.mlp(rank, one, index, rows, attn, ranks_file.partial(ffn_in, rank.rank, rows));
                try m.gather.run(ffn);
                for (&b.ranks) |*one| {
                    try phases.finish(one, rows, ffn);
                    try phases.tap(one, index, rows);
                }
            }
            for (m.ranks, &b.ranks) |rank, *one| {
                try phases.head(rank, one, rows, .none);
                if (scored == 0) continue;
                // the rows scored, side by side in the read-out's rows, which no phase reads again
                for (parts, spans, self.starts[0..taken]) |part, span, at| switch (part.score) {
                    .all => try hc.rmsnorm(one.hidden.span(span.first, span.rows), rank.norm.view, one.normed.span(at, span.rows), rank.eps),
                    .last => try hc.rmsnorm(one.hidden.span(span.first + span.rows - 1, 1), rank.norm.view, one.normed.span(at, 1), rank.eps),
                    .none => {},
                };
                try phases.headPlain(rank, one.normed.rows(scored), one.logits.rows(scored));
            }
            if (held < world and scored > 0) try m.gather.nccl.allGather(b.ranks[0].logits.rows(scored), self.gatheredRows());
        }

        /// The rounds as captured graphs, a graph a round's shape: its parts with each slot's side of the
        /// KDA states and its context bucket. Over NCCL a graph holds its all-gathers among the kernels, so
        /// both ranks launch the same graphs in step; a round is told and staged every time, outside it.
        pub const Graphs = struct {
            cache: cache_file.Cache(Key),
            pub const Key = struct { rows: usize, shape: u64 };
            pub fn init(gpa: Allocator) cuda.Error!Graphs {
                return .{ .cache = try .init(gpa) };
            }
            /// The graphs hold pointers into the slots, model and buffers they were captured with: drop them
            /// first.
            pub fn deinit(self: *Graphs) void {
                self.cache.deinit();
                self.* = undefined;
            }
            pub fn key(slots: *const SlotsOf(held), parts: []const Part, rows: usize) Key {
                var hashed: std.hash.Wyhash = .init(0);
                const many: u64 = parts.len;
                hashed.update(std.mem.asBytes(&many));
                for (parts) |part| {
                    const parity: u64 = if (slots.seqs[0][part.slot].kda) |one| @intCast(one.cur) else 0;
                    const bucket: u64 = phases.bucket(slots.capacity, slots.pos(part.slot) + part.rows);
                    // a chunked KDA part's chunks start at absolute positions, its first one's rows the phase's
                    const phase: u64 = if (kda.chunks(part.rows)) slots.pos(part.slot) % kda.chunk else 0;
                    for ([_]u64{ part.slot, part.rows, @backingInt(part.score), parity, bucket, phase }) |word| {
                        hashed.update(std.mem.asBytes(&word));
                    }
                }
                return .{ .rows = rows, .shape = hashed.final() };
            }
            /// `forward` through a graph: the first round of a shape captures it, the rest replay it.
            pub fn forward(self: *Graphs, slots: *SlotsOf(held), m: H.Model, b: *H.Buffers, parts: []const Part, tokens: []const i32) (Error || Allocator.Error)!void {
                try slots.stage(m, b, parts, tokens);
                const at = key(slots, parts, tokens.len);
                return self.cache.run(at, Self.run, .{ slots, m, b, parts.len });
            }
            /// Whether a round of these parts runs without a capture.
            pub fn holds(graphs: *const Graphs, slots: *const SlotsOf(held), parts: []const Part, rows: usize) bool {
                return graphs.cache.contains(key(slots, parts, rows));
            }
        };

        /// Every rank's scored rows of the last round: bf16 [world, scored, vocab / world].
        fn gatheredRows(self: Self) cuda.View {
            const whole = self.whole.?.view;
            return .init(whole.ptr.?, .bf16, &.{ world, self.scored, @intCast(whole.size[2]) });
        }

        /// Keeps the first `keeps[i]` rows of the last round's part i, of every part; 0 keeps none of it.
        pub fn commit(self: *Self, m: H.Model, keeps: []const usize) Error!void {
            std.debug.assert(keeps.len == self.ran);
            if (m.lead) |lead| {
                var words: [lead_most]u32 = undefined;
                words[0] = @intCast(keeps.len);
                for (keeps, words[1..][0..keeps.len]) |keep, *word| word.* = @intCast(keep);
                try lead.tell(@backingInt(Op.slots_commit), words[0 .. 1 + keeps.len], &.{});
            }
            for (self.spans[0..self.ran], keeps) |span, keep| {
                std.debug.assert(keep <= span.rows);
                if (keep == 0) continue;
                for (&self.seqs, self.windows) |seqs, in| try seqs[span.slot].commit(in, span.first, span.rows, keep);
            }
        }

        /// Where the last round's part `part` has its rows in the window, and its scored rows among the scored.
        pub fn first(self: Self, part: usize) usize {
            std.debug.assert(part < self.ran);
            return self.spans[part].first;
        }

        pub fn scoredAt(self: Self, part: usize) usize {
            std.debug.assert(part < self.ran);
            return self.starts[part];
        }

        /// Scored row `at` of the last round (`scoredAt` and the row within its part) as bf16 bits [vocab],
        /// rank 0's half of the vocabulary and then rank 1's. Nothing is told: the round gathered the rows.
        pub fn logitsRow(self: Self, b: H.Buffers, at: usize, out: []u16) Error!void {
            std.debug.assert(at < self.scored);
            const half = out.len / world;
            try cuda.synchronize();
            if (self.whole) |whole| {
                std.debug.assert(@as(usize, @intCast(whole.view.size[2])) == half);
                for (0..world) |rank| try whole.downloadAt((rank * self.scored + at) * half * 2, std.mem.sliceAsBytes(out[rank * half ..][0..half]));
                return;
            }
            for (b.ranks, b.first..) |one, rank| {
                std.debug.assert(@as(usize, @intCast(one.logits.view.size[1])) == half);
                try one.logits.downloadAt(at * half * 2, std.mem.sliceAsBytes(out[rank * half ..][0..half]));
            }
        }
    };
}

const lead_most = @import("../../../ranks/lead.zig").Lead.most;
