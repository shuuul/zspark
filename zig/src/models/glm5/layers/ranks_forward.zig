//! GLM's forward over the ranks one process holds (spec 045): both, their partials written where the other
//! reads them (`loopback.zig`, what the tests run and the reference for two devices), or its own one, the
//! other's brought by a gather over NCCL. The same phases in the same order either way (`forward.zig`); a
//! gathered buffer is used tight (`zig/src/ranks/ranks.zig`), so every reader gets the same view.
//!
//! `Held(n)`'s `Model`, `State`, `Buffers`, `Graphs` and functions are what `draft.Decoder` asks of a model
//! (`zig/src/decode/draft.zig`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const lead_file = @import("../../../ranks/lead.zig");
const ranks_file = @import("../../../ranks/ranks.zig");
const Config = @import("config.zig").Config;
const layout = @import("layout.zig");
const model = @import("model.zig");
const phases = @import("forward.zig");
const kda = @import("kda.zig");
const picking = @import("../../../decode/pick.zig");
const sampling = @import("../../../decode/sampling.zig");
const graphs_file = @import("../../../decode/graphs.zig");
const settle_file = @import("../../../decode/settle.zig");
const nccl = @import("../../../ranks/nccl.zig");

pub const Error = cuda.Error || ranks_file.Error;
const Failure = Error;

/// The fewest rows a window splits its glue at over the ranks (`Held.splits`): a prompt's chunks, not a
/// decoding round's windows, whose few rows gain nothing from two exchanges a site for one, as Flash Next's
/// (`qwen4/ranks_forward.zig`). A test lowers it.
pub var split_rows: usize = 256;
pub const Score = phases.Score;
const world = layout.world;

/// `rank`'s rows of a gathered buffer in a window of `rows` rows: fp32 [rows, dims] at `rank * rows * dims`.
pub fn partial(buffer: cuda.Buffer, rank: usize, rows: usize) cuda.View {
    return ranks_file.partial(buffer.view, rank, rows);
}

/// What a window of `rows` rows reads of a gathered buffer: fp32 [world, rows, dims].
pub fn gathered(buffer: cuda.Buffer, rows: usize) cuda.View {
    return ranks_file.gathered(buffer.view, rows);
}

/// A gathered buffer as a window whose quantized rows start at `from` writes its partials there: fp32, or bf16 in
/// FP4-Local's quantized windows (`phases.partials`, spec 116 WS-04).
pub fn partialsIn(rank: phases.Model, buffer: cuda.Buffer, from: usize) cuda.View {
    return ranks_file.as(buffer.view, phases.partials(rank, from));
}

/// What a window of `rows` rows (`compute`'s, of one form) reads of a gathered buffer, in its partials' dtype.
pub fn gatheredOf(rank: phases.Model, buffer: cuda.Buffer, rows: usize) cuda.View {
    return ranks_file.gathered(partialsIn(rank, buffer, phases.quantizedFrom(rows)), rows);
}

/// What the rank that leads tells the other before it runs it (`zig/src/ranks/lead.zig`), each with its arguments
/// and then its window's tokens; `mirror.zig` reads them back and runs them.
pub const Op = enum(u32) {
    /// `State.reset`.
    reset = 1,
    /// `forwardScoring`: the `Score`, then the digest of the picks before it (`picks`), low word first.
    forward,
    /// `Graphs.forward`: the digest.
    graph,
    /// `commit`: the rows kept.
    commit,
    /// `logitsRow`: the row.
    row,
    /// The MTP layer's `step`: whether its hidden rows are its own, the main model's first row, the rank its
    /// scored row's start is asked to (0: it does not score), the digest; `setLen`: the entries; `wholeRow`.
    mtp_step,
    mtp_len,
    mtp_row,
    /// Nothing follows.
    end,
    /// The DFlash2 drafter's passes (`dflash2/pass.zig`), each with the context's end before it: `addTaps`,
    /// the taps' first row and the rows; `block`, the pending token.
    flash_taps,
    flash_block,
    /// Several sequences in one window (`slots.zig`). `Slots.reset`: the slot. `Slots.forward`: the parts,
    /// then a slot, its rows and its `Score` for each, then the digest. `Slots.commit`: the parts, then the
    /// rows kept of each.
    slot_reset,
    slots_forward,
    slots_commit,
    /// `mtp_seats.zig`'s `step`: the items, then a slot, its first entry, its rows, its hidden rows (a row of
    /// the main model's twice, or one of the last step's twice and one) and the rank asked for each, then the
    /// digest. `wholeRow`: the scored row.
    mtp_steps,
    mtp_row_of,
    /// `decode/dflash2/seats.zig`'s `round`: the items, then a slot, its context's end, the taps' first row, the
    /// rows, and its block's pending token and one (0: no block) for each.
    flash_round,
    /// A snapshot of the committed state (`session/snapshot.zig`). `keep` takes one here; `back` restores
    /// the one at the position.
    keep,
    back,
    /// A state on disk (`serve --spill-gib`, `session/learnt.zig`; spec 108 WS-04), each rank its own half.
    /// `spill_tokens`: where the frame's tokens go in the sequence the next operation names (0 starts it
    /// again), then the tokens. `spill_open`: each rank hashes the state it holds and opens its store.
    /// `spill_write`: the mark at the position goes to disk, and whether the stop writes it. `spill_read`: the
    /// file of the tokens told through the position comes back as a mark there, both ranks agreeing that
    /// each holds its half, and where the prompt stood without it.
    spill_tokens,
    spill_open,
    spill_write,
    spill_read,
    /// `settle`: the window's first row and its rows.
    settle,
    /// A snapshot inside the window that has run and is not committed (`session/marks.zig`, spec 112 WS-06):
    /// the rows into it. `drop`: the snapshot at the position goes.
    keep_within,
    drop,
    /// The marks inside the next window, which runs chunked, whose states it writes as it runs (`kda.State.ask`,
    /// spec 116): how many, then the rows into it of each, rising. Each is then `keep_within`.
    keep_ahead,
};

/// A digest as an operation's two words, low first.
pub fn words(digest: u64) [2]u32 {
    return .{ @truncate(digest), @truncate(digest >> 32) };
}

fn told(lead: ?*lead_file.Lead, op: Op, arguments: []const u32, tokens: []const i32) lead_file.Error!void {
    if (lead) |to| try to.tell(@backingInt(op), arguments, tokens);
}

/// Where `computeWatched` stops for its watcher.
pub const Point = enum {
    /// After a layer's attention and its gather: `Buffers.attn` holds every rank's partial.
    attention,
    /// After a layer's end: `Buffers.ffn` holds every rank's partial, and the streams and the taps the layer's.
    layer,
};

/// The forward over `held` ranks in one process: all of them, or one.
pub fn Held(comptime held: usize) type {
    comptime std.debug.assert(held == 1 or held == world);
    return struct {
        const Self = @This();

        pub const Model = struct {
            /// The ranks held, rising: `ranks[i].rank` is which.
            ranks: [held]model.Model,
            max_rows: usize,
            /// What brings the other ranks' partials: nothing where they are all here.
            gather: ranks_file.Gather = .loopback,
            /// This rank leads: each operation below is told before it runs. Set before a state or buffers
            /// are made of the model, which keep it.
            lead: ?*lead_file.Lead = null,

            /// Every share of `source`; `arena` as `model.Model.load`'s.
            pub fn load(arena: Allocator, gpa: Allocator, source: anytype, c: *const Config, rows: cuda.Scratch.Rows) model.LoadError!Model {
                comptime std.debug.assert(held == world);
                return loadFrom(arena, gpa, source, c, 0, rows);
            }

            /// The shares of ranks `first` .. `first + held - 1`, and nothing of another's.
            pub fn loadFrom(arena: Allocator, gpa: Allocator, source: anytype, c: *const Config, first: usize, rows: cuda.Scratch.Rows) model.LoadError!Model {
                std.debug.assert(first + held <= world);
                var self: Model = .{ .ranks = undefined, .max_rows = rows.count };
                var done: usize = 0;
                errdefer for (self.ranks[0..done]) |*one| one.deinit();
                for (&self.ranks, first..) |*one, rank| {
                    one.* = try .load(arena, gpa, source, c, rank, rows);
                    done += 1;
                }
                return self;
            }

            pub fn deinit(self: *Model) void {
                for (&self.ranks) |*one| one.deinit();
                self.* = undefined;
            }
        };

        pub const State = struct {
            ranks: [held]model.State,
            /// Positions committed, each rank's too.
            pos: usize = 0,
            lead: ?*lead_file.Lead = null,

            /// For up to `capacity` positions.
            pub fn init(m: Model, capacity: usize) cuda.Error!State {
                var self: State = .{ .ranks = undefined, .lead = m.lead };
                var done: usize = 0;
                errdefer for (self.ranks[0..done]) |*one| one.deinit();
                for (&self.ranks, m.ranks) |*one, rank| {
                    one.* = try .init(rank, capacity);
                    done += 1;
                }
                return self;
            }

            pub fn deinit(self: *State) void {
                for (&self.ranks) |*one| one.deinit();
                self.* = undefined;
            }

            /// An empty sequence.
            pub fn reset(self: *State) Error!void {
                try told(self.lead, .reset, &.{}, &.{});
                for (&self.ranks) |*one| try one.reset();
                self.pos = 0;
            }
        };

        pub const Buffers = struct {
            ranks: [held]model.Buffers,
            /// fp32 [world, rows, dims], used tight: the attention's partials and the MLP's. Two, as a rank's
            /// MLP writes before the next rank has read the attention's.
            attn: cuda.Buffer,
            ffn: cuda.Buffer,
            rows: usize,
            capacity: usize,
            /// The rank `ranks[0]` is: where its half of the vocabulary lies.
            first: usize,
            /// What brings the other ranks' halves of a row of logits (`logitsRow`), and where: bf16
            /// [world, vocab / world]. Null where every rank is here.
            gather: ranks_file.Gather,
            row: ?cuda.Buffer = null,
            /// Where a window's rows are settled on the device (`settle`): u32 [world, rows, a pick's words over
            /// a half of the vocabulary], every slot asked to rank 1; the host's copy. Null where every rank is here.
            picks: ?cuda.Buffer = null,
            picked: []u32 = &.{},
            /// `picks`' copies, where every rank is not here.
            behind: ?*Behind = null,
            lead: ?*lead_file.Lead = null,
            /// The rows of the last window or MTP step: whose expert picks the MoE's buffers hold.
            window: usize = 0,
            /// Where a split window's scatters run beside the rank's own rows (`scattered`), over NCCL.
            beside: ?*ranks_file.Beside = null,

            pub fn init(m: Model, capacity: usize) (cuda.Error || Allocator.Error)!Buffers {
                var self: Buffers = .{ .ranks = undefined, .attn = undefined, .ffn = undefined, .rows = m.max_rows, .capacity = capacity, .first = m.ranks[0].rank, .gather = m.gather, .lead = m.lead };
                var done: usize = 0;
                errdefer for (self.ranks[0..done]) |*one| one.deinit();
                for (&self.ranks, m.ranks) |*one, rank| {
                    one.* = try .init(rank, capacity);
                    done += 1;
                }
                const shape = [_]usize{ world, m.max_rows, m.ranks[0].dims };
                self.attn = try .init(.f32, &shape);
                errdefer self.attn.deinit();
                self.ffn = try .init(.f32, &shape);
                errdefer self.ffn.deinit();
                if (held < world) {
                    const half = m.ranks[0].head.outputs();
                    self.row = try .init(.bf16, &.{ world, half });
                    errdefer self.row.?.deinit();
                    const slot = picking.Picked.wordsFor(half);
                    self.picked = try std.heap.page_allocator.alloc(u32, world * settled_rows * slot);
                    errdefer std.heap.page_allocator.free(self.picked);
                    // the kernel reads a slot's rank and never writes it, and the other rank's slots arrive asked
                    // alike, so a window of any rows finds its slots asked
                    @memset(self.picked, 0);
                    for (0..world * settled_rows) |at| self.picked[at * slot + picking.Picked.rank_word] = 1;
                    self.picks = try .init(.u32, &.{ world, settled_rows, slot });
                    errdefer self.picks.?.deinit();
                    try self.picks.?.upload(std.mem.sliceAsBytes(self.picked));
                    errdefer self.picks.?.deinit();
                    if (self.ranks[0].moe) |moe| self.behind = try .create(moe.picks.bytes());
                    errdefer if (self.behind) |one| one.destroy();
                    const beside = try std.heap.page_allocator.create(ranks_file.Beside);
                    errdefer std.heap.page_allocator.destroy(beside);
                    beside.* = try .init();
                    self.beside = beside;
                }
                return self;
            }

            pub fn deinit(self: *Buffers) void {
                if (self.row) |*row| row.deinit();
                if (self.picks) |*one| one.deinit();
                if (self.behind) |one| one.destroy();
                if (self.beside) |one| {
                    one.deinit();
                    std.heap.page_allocator.destroy(one);
                }
                if (self.picked.len > 0) std.heap.page_allocator.free(self.picked);
                self.attn.deinit();
                self.ffn.deinit();
                for (&self.ranks) |*one| one.deinit();
                self.* = undefined;
            }

            /// Every rank here keeps the drafter's taps from here on (`model.Buffers.setTaps`).
            pub fn setTaps(self: *Buffers, layers: []const u32) cuda.Error!void {
                for (&self.ranks) |*one| try one.setTaps(layers);
            }

            /// The widest window whose rows can all be scored.
            pub fn scoredRows(self: Buffers) usize {
                return self.rows;
            }
        };

        /// A digest of the expert picks the window before the last left (its last MoE's, or the MTP layer's
        /// after a step of it): 0 before any. Every rank routes from the same gathered rows, so every rank holds
        /// the same picks; the rank that leads sends this with each window and the follower fails the pair where
        /// its own differs (TensorFold's `PairOutOfStep`), one window after caches parted. Each call copies the
        /// picks as they stand, without waiting, and returns the digest of the last call's copy, waiting for that
        /// copy alone (spec 114 WS-02): a digest of the picks as they stand waited for every launch, so the rank
        /// that leads told each operation only once its GPU had drained, and the follower ran it only after its
        /// own had. Both ranks call it once an operation that routes (the leader telling it, the follower
        /// running it), so each digests the same windows. Where every rank is here it is never called.
        pub fn picks(b: Buffers) cuda.Error!u64 {
            const moe = b.ranks[0].moe orelse return 0;
            const behind = b.behind orelse return 0;
            const wide: usize = @intCast(moe.picks.view.size[1]);
            const last = behind.next ^ 1;
            var digest: u64 = 0;
            if (behind.windows[last] > 0) {
                const window = behind.windows[last];
                try behind.events[last].synchronize();
                var hash: std.hash.Wyhash = .init(window);
                hash.update(behind.copies[last].bytes[0 .. window * wide * @sizeOf(i32)]);
                digest = hash.final();
            }
            const slot = behind.next;
            behind.windows[slot] = b.window;
            if (b.window > 0) {
                try moe.picks.downloadLaunched(0, behind.copies[slot].bytes[0 .. b.window * wide * @sizeOf(i32)]);
                try behind.events[slot].record();
            }
            behind.next ^= 1;
            return digest;
        }

        /// `picks`' copies: two, the last call's and this one's, each with the window it holds and its event.
        pub const Behind = struct {
            copies: [2]cuda.HostBuffer,
            events: [2]cuda.Event,
            windows: [2]usize = .{ 0, 0 },
            next: u1 = 0,

            fn create(bytes: usize) (cuda.Error || Allocator.Error)!*Behind {
                const self = try std.heap.page_allocator.create(Behind);
                errdefer std.heap.page_allocator.destroy(self);
                self.* = .{ .copies = undefined, .events = undefined };
                self.copies[0] = try .init(bytes);
                errdefer self.copies[0].deinit();
                self.copies[1] = try .init(bytes);
                errdefer self.copies[1].deinit();
                self.events[0] = try .init();
                errdefer self.events[0].deinit();
                self.events[1] = try .init();
                return self;
            }

            fn destroy(self: *Behind) void {
                for (&self.copies, &self.events) |*copy, *event| {
                    copy.deinit();
                    event.deinit();
                }
                std.heap.page_allocator.destroy(self);
            }
        };

        pub fn stage(b: *Buffers, tokens: []const i32) (cuda.Error || nccl.Error)!void {
            for (&b.ranks) |*one| try phases.stageIds(one, tokens);
        }

        /// Whether a window of `rows` rows splits its glue by rows over the ranks (spec 107 WS-03, Flash Next's
        /// spec 106 WS-02): each rank writes back and reads out its own rows (`ranks_file.share`), receives
        /// only those rows of the other's partials (`Gather.scatter`), and the read-outs are then shared, bf16.
        /// Every kernel of the glue works a row at a time and the partials are summed rank 0 first as before,
        /// so the bits are the whole glue's. Not with a drafter's taps, which read every row's streams a layer.
        pub fn splits(b: Buffers, rows: usize) bool {
            return rows >= split_rows and b.ranks[0].tap_layers.len == 0;
        }

        pub const Side = enum { attn, ffn };

        /// A site's glue on each held rank's own rows: the write-back of `pending` (the branch before; null at
        /// the first), the read-out of layer `index`'s `side`, then every rank's rows of `normed` on every rank.
        pub fn glue(m: Model, b: *Buffers, rows: usize, index: usize, side: Side, pending: ?cuda.View) Error!void {
            for (m.ranks, &b.ranks) |rank, *one| {
                const own = ranks_file.share(rows, world, rank.rank);
                const count = own[1] - own[0];
                if (count == 0) continue;
                if (pending) |g| try phases.finishRows(one, own[0], count, g);
                const layer = rank.layers[index];
                try phases.readOutRows(rank, one, own[0], count, switch (side) {
                    .attn => layer.attn_site,
                    .ffn => layer.ffn_site,
                });
            }
            return shareRows(b, "normed", rows);
        }

        /// A split site's partials into `buffer` (`b.attn` or `b.ffn`) and scattered (spec 116 WS-03): each rank
        /// held makes the others' rows first (`made.part(held_rank, first, count)`), their scatter starts beside
        /// (`Gather.scatterBegin`), and the rank's own rows are made meanwhile; the next launches wait for the
        /// scatter. Each part's kernels work a row at a time, so the bits are the whole site's in one launch.
        pub fn scattered(m: Model, b: *Buffers, rows: usize, buffer: cuda.View, made: anytype) Error!void {
            for (m.ranks, 0..) |rank, i| {
                const own = ranks_file.share(rows, world, rank.rank);
                if (own[0] > 0) try made.part(i, 0, own[0]);
                if (own[1] < rows) try made.part(i, own[1], rows - own[1]);
            }
            try m.gather.scatterBegin(ranks_file.gathered(buffer, rows), b.beside);
            for (m.ranks, 0..) |rank, i| {
                const own = ranks_file.share(rows, world, rank.rank);
                if (own[1] > own[0]) try made.part(i, own[0], own[1] - own[0]);
            }
            try m.gather.scatterJoin(b.beside);
        }

        /// `scattered`'s parts of an attention: the out-projection of rows already attended (`phases.attendBody`
        /// or `attendOfBody`), `chains` each rank's KDA rows.
        pub const Emitted = struct {
            m: Model,
            b: *Buffers,
            index: usize,
            rows: usize,
            chains: [held]?kda.State,
            /// `b.attn` as `partialsIn` gives it
            buffer: cuda.View,

            pub fn part(self: Emitted, i: usize, first: usize, count: usize) Error!void {
                const rank = self.m.ranks[i];
                return phases.emitRows(rank, self.chains[i], &self.b.ranks[i], self.index, first, count, ranks_file.partial(self.buffer, rank.rank, self.rows));
            }
        };

        /// `scattered`'s parts of an MLP, after `phases.feedBody`.
        pub const Filled = struct {
            m: Model,
            b: *Buffers,
            index: usize,
            rows: usize,
            /// `b.ffn` as `partialsIn` gives it
            buffer: cuda.View,

            pub fn part(self: Filled, i: usize, first: usize, count: usize) Error!void {
                const rank = self.m.ranks[i];
                return phases.fillRows(rank, &self.b.ranks[i], self.index, self.rows, first, count, ranks_file.partial(self.buffer, rank.rank, self.rows));
            }
        };

        /// A split window's MLP of layer `index` after its glue: the bodies, then `scattered`.
        pub fn feedScattered(m: Model, b: *Buffers, index: usize, rows: usize, buffer: cuda.View) Error!void {
            for (m.ranks, &b.ranks) |rank, *one| try phases.feedBody(rank, one, index, rows);
            return scattered(m, b, rows, buffer, Filled{ .m = m, .b = b, .index = index, .rows = rows, .buffer = buffer });
        }

        /// The last layer's write-back on each rank's own rows, then every row's streams on every rank for the
        /// head (and the MTP layer, which reads the hidden rows after it).
        pub fn glueEnd(m: Model, b: *Buffers, rows: usize, pending: cuda.View) Error!void {
            for (m.ranks, &b.ranks) |rank, *one| {
                const own = ranks_file.share(rows, world, rank.rank);
                if (own[1] > own[0]) try phases.finishRows(one, own[0], own[1] - own[0], pending);
            }
            return shareRows(b, "x", rows);
        }

        /// Every rank's own rows of the buffer `name` of `b.ranks` on every rank held: over NCCL in place, in
        /// loopback a copy between the ranks' buffers.
        fn shareRows(b: *Buffers, comptime name: []const u8, rows: usize) Error!void {
            if (held == 1) return b.gather.shareRows(@field(b.ranks[0], name).rows(rows));
            for (0..held) |to| for (0..held) |from| {
                if (to == from) continue;
                const theirs = ranks_file.share(rows, world, b.first + from);
                if (theirs[1] > theirs[0]) try @field(b.ranks[to], name).copyRows(theirs[0], @field(b.ranks[from], name), theirs[0], theirs[1] - theirs[0]);
            };
        }

        /// The staged window of `rows` rows at the sequence's position, launched for `context` positions:
        /// each rank's logits in its buffers, and nothing committed.
        pub fn compute(m: Model, state: *State, b: *Buffers, rows: usize, context: usize, score: phases.Score) Error!void {
            return computeWatched(m, state, b, rows, context, score, {});
        }

        /// `compute`, stopping at every `Point` of every layer for `watch.at(point, layer)`, which may wait
        /// and read; `{}` watches nothing. Not inside a capture with a watcher that reads.
        pub fn computeWatched(m: Model, state: *State, b: *Buffers, rows: usize, context: usize, score: phases.Score, watch: anytype) !void {
            std.debug.assert(m.gather.serves(held, world));
            // the window's form by its rows (spec 115): a step's and a verify window's are a short prompt window's
            const from = phases.quantizedFrom(rows);
            for (m.ranks) |rank| phases.setForm(rank, from);
            defer for (m.ranks) |rank| phases.clearForm(rank);
            for (m.ranks, &b.ranks) |rank, *one| try phases.embed(rank, one, rows);
            const attn_in = partialsIn(m.ranks[0], b.attn, from);
            const ffn_in = partialsIn(m.ranks[0], b.ffn, from);
            const attn = ranks_file.gathered(attn_in, rows);
            const ffn = ranks_file.gathered(ffn_in, rows);
            // a window the head does not score every row of is a prompt's (`decoder.prompt`, the pool's
            // parts): over NCCL its indexer rows at or past `index.split_from` split across the ranks
            // (`dsa.indexRows`); a decode window, scored `.all`, and loopback never split
            const split: ?nccl.Comm = if (score != .all) switch (m.gather) {
                .nccl => |comm| comm,
                .loopback => null,
            } else null;
            if (@TypeOf(watch) == void and splits(b.*, rows)) {
                for (0..m.ranks[0].layers.len) |index| {
                    try glue(m, b, rows, index, .attn, if (index == 0) null else ffn);
                    var chains: [held]?kda.State = undefined;
                    for (m.ranks, &state.ranks, &b.ranks, &chains) |rank, *sequence, *one, *chain| {
                        try phases.attendBody(rank, sequence, one, index, rows, context, split);
                        chain.* = sequence.kda;
                    }
                    try scattered(m, b, rows, attn_in, Emitted{ .m = m, .b = b, .index = index, .rows = rows, .chains = chains, .buffer = attn_in });
                    try glue(m, b, rows, index, .ffn, attn);
                    try feedScattered(m, b, index, rows, ffn_in);
                }
                try glueEnd(m, b, rows, ffn);
                for (m.ranks, &b.ranks) |rank, *one| try phases.head(rank, one, rows, score);
                return;
            }
            for (0..m.ranks[0].layers.len) |index| {
                for (m.ranks, &state.ranks, &b.ranks) |rank, *sequence, *one|
                    try phases.attention(rank, sequence, one, index, rows, context, ranks_file.partial(attn_in, rank.rank, rows), split);
                try m.gather.run(attn);
                if (@TypeOf(watch) != void) try watch.at(Point.attention, index);
                for (m.ranks, &b.ranks) |rank, *one|
                    try phases.mlp(rank, one, index, rows, attn, ranks_file.partial(ffn_in, rank.rank, rows));
                try m.gather.run(ffn);
                for (&b.ranks) |*one| {
                    try phases.finish(one, rows, ffn);
                    try phases.tap(one, index, rows);
                }
                if (@TypeOf(watch) != void) try watch.at(Point.layer, index);
            }
            for (m.ranks, &b.ranks) |rank, *one| try phases.head(rank, one, rows, score);
        }

        /// Runs `tokens` as rows at positions `state.pos` onward and scores each. The committed state is
        /// unchanged until `commit`.
        pub fn forward(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
            return forwardScoring(m, state, b, tokens, .all);
        }

        /// `forward` scoring what `score` names.
        pub fn forwardScoring(m: Model, state: *State, b: *Buffers, tokens: []const i32, score: phases.Score) Error!void {
            try cuda.bound(tokens.len <= m.max_rows and state.pos + tokens.len <= b.capacity, "glm5 forward: {d} tokens (at most {d}) at {d} of a capacity of {d}", .{ tokens.len, m.max_rows, state.pos, b.capacity });
            if (m.lead != null) {
                const digest = words(try picks(b.*));
                try told(m.lead, .forward, &.{ @backingInt(score), digest[0], digest[1] }, tokens);
            }
            b.window = tokens.len;
            try stage(b, tokens);
            return compute(m, state, b, tokens.len, state.pos + tokens.len, score);
        }

        /// Keeps the first `keep` rows of the window `tokens` that the last forward ran.
        pub fn commit(m: Model, state: *State, b: Buffers, tokens: []const i32, keep: usize) Error!void {
            _ = b;
            try told(m.lead, .commit, &.{@intCast(keep)}, tokens);
            for (&state.ranks) |*one| try one.commit(tokens.len, keep);
            state.pos += keep;
        }

        /// Row `row`'s logits as bf16 bits [vocab], rank 0's half of the vocabulary and then rank 1's. A
        /// process that holds one rank gathers the other's half of the row (`gatherRow`).
        pub fn logitsRow(b: Buffers, row: usize, out: []u16) Error!void {
            try told(b.lead, .row, &.{@intCast(row)}, &.{});
            var logits: [held]cuda.Buffer = undefined;
            for (&logits, b.ranks) |*one, rank| one.* = rank.logits;
            return settle_file.logitsRow(world, held, logits, b.first, b.gather, b.row, row, out);
        }

        /// A row of logits over the vocabulary from this rank's half of it, `own` (bf16 [1, vocab / world]):
        /// the halves gathered, 155 KiB a rank for GLM's vocabulary, rank 0's first (`decode/settle.zig`). The
        /// buffers' `gather` is the model's when they were made.
        pub fn gatherRow(b: Buffers, own: cuda.View, out: []u16) Error!void {
            comptime std.debug.assert(held < world);
            std.debug.assert(own.count() * world == out.len);
            return settle_file.gatherRow(b.gather.nccl, own, b.row.?, out);
        }

        /// The most rows `settle` settles at once: a verify window's (`draft.forward_max_rows`).
        pub const settled_rows = 16;

        /// `draft.Chooser`'s rows settled on the device (spec 114 WS-02, as Flash Next's two ranks since spec
        /// 095, `decode/settle.zig`): one gather of some 4 KiB a row and a rank brings each rank's picks of its
        /// half, and the joined start chooses the row's token with nothing read back. A row the picks do not
        /// settle is read whole by `logitsRow`. Told first: the follower picks and gathers the same rows.
        pub fn settle(b: Buffers, row: usize, sampler: *sampling.Sampler, settings: sampling.Sampling, out: []?i32) Error!void {
            @memset(out, null);
            if (held == world or !settings.greedy() or out.len > settled_rows) return;
            try told(b.lead, .settle, &.{ @intCast(row), @intCast(out.len) }, &.{});
            std.debug.assert(row + out.len <= b.rows);
            return settle_file.settle(world, b.gather.nccl, b.ranks[0].logits, b.picks.?, b.picked, row, sampler, settings, out);
        }

        /// `settle`'s launches on every rank (`decode/settle.zig`'s `gatherPicks`): what the follower runs.
        pub fn gatherPicks(b: Buffers, row: usize, rows: usize) Error!void {
            comptime std.debug.assert(held < world);
            std.debug.assert(rows <= settled_rows and row + rows <= b.rows);
            return settle_file.gatherPicks(world, b.gather.nccl, b.ranks[0].logits, b.picks.?, null, row, rows);
        }

        /// The forward as captured graphs, a graph a (row count, side of the KDA states, context): `decode/graphs.zig`.
        pub const Graphs = graphs_file.Graphs(Graphed.Key, Graphed);

        const Graphed = struct {
            const Key = struct { rows: usize, parity: u1, context: usize };
            pub const Model = Self.Model;
            pub const State = Self.State;
            pub const Buffers = Self.Buffers;
            pub const Error = Failure;

            fn key(state: Self.State, b: Self.Buffers, rows: usize) Key {
                return .{ .rows = rows, .parity = state.ranks[0].parity(), .context = phases.bucket(b.capacity, state.pos + rows) };
            }

            pub fn enter(m: Self.Model, state: *Self.State, b: *Self.Buffers, tokens: []const i32) Failure!Key {
                if (m.lead != null) try told(m.lead, .graph, &words(try picks(b.*)), tokens);
                b.window = tokens.len;
                try stage(b, tokens);
                return key(state.*, b.*, tokens.len);
            }

            pub fn compute(m: Self.Model, state: *Self.State, b: *Self.Buffers, rows: usize, at: Key) Failure!void {
                return Self.compute(m, state, b, rows, at.context, .all);
            }

            pub fn ahead(state: Self.State, b: Self.Buffers, rows: usize) ?Key {
                return if (rows <= b.rows) key(state, b, rows) else null;
            }
        };
    };
}
