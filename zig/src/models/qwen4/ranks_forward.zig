//! Flash Next's forward over the ranks one process holds (spec 050): both, in loopback, or one with the other
//! rank's partial left at zero (`Model.alone`, the gate). A world of one does not come through here: it is
//! `forward.zig`, on the same kernels. Partials are fp32; the branch sum is rounded to bf16 once and written
//! back as `.branch`, the MoE sum as `.sum` (`partials.zig`, `hc.zig`). Fused write-backs stay on one rank.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const lead_file = @import("../../ranks/lead.zig");
const picking = @import("../../decode/pick.zig");
const sampling = @import("../../decode/sampling.zig");
const graphs_file = @import("../../decode/graphs.zig");
const settle_file = @import("../../decode/settle.zig");
const ranks_file = @import("../../ranks/ranks.zig");
const shelf_file = @import("../../decode/shelf.zig");
const hc = @import("blocks/hc.zig");
const moe = @import("blocks/moe.zig");
const ple = @import("blocks/ple.zig");
const forward_file = @import("forward.zig");
const model_file = @import("model.zig");
const partials = @import("partials.zig");
const share_file = @import("share.zig");
const taps_file = @import("taps.zig");

const Failure = forward_file.Error || ranks_file.Error || error{UnsupportedScheme};
pub const Score = forward_file.Score;
pub const Snapshot = forward_file.Snapshot;
const world: usize = share_file.world;

/// What rank 0 tells rank 1 before the operation runs (`session/mirror.zig` runs it). A window's tokens follow
/// the arguments. `pick` is the reply's tokens, chosen here and run by no kernel.
pub const Op = enum(u32) {
    forward = 1,
    graph,
    commit,
    row,
    pick,
    reset,
    end,
    /// Keep this rank's state as it stands, above every state held (`Shelf`). Both ranks are at the same
    /// position.
    keep,
    /// Restore the state kept at the position in the argument.
    back,
    /// A round of several streams (`pool/rounds_ranks.zig`'s `Desk`). The parts, then each part's slot and its
    /// rows. The tokens are the rows, in that order.
    streams,
    /// Commit that round: the parts, then how many rows each keeps, in the round's order. No tokens.
    streams_commit,
    /// A pass of the split MTP head (`mtp_ranks.zig`): the head's position before it, the streams it reads
    /// (0 the window's rows, else 1 + the row of the head's own), and its flags (1 scored, 2 launched and not a
    /// graph). The tokens are those that follow its rows.
    head,
    /// Keep the state after the first rows of the window this rank last ran and has not committed
    /// (`keepWithinLocal`): how many rows. The tokens are that window's.
    keep_within,
    /// The state kept at the position in the argument is held no more.
    drop,
    /// Where a stream's positions lie in the pools, as rank 0's extents now have them (`Desk.place`): the slot,
    /// the first entry that changed and the table's length. The tokens are the entries from that one on.
    table,
    /// A round of the pool's engine (`pool/hall_ranks.zig`): the score, the rows the main head scores, the
    /// parts, then each part's slot and its rows. The tokens are the rows, in that order.
    hall_round,
    /// The main head's half of the round's row in the first argument, into the row of logits in the second.
    hall_logits,
    /// Commit the engine's round: the parts, then how many rows each keeps. No tokens.
    hall_commit,
    /// The engine's slot in the argument begins again.
    hall_reset,
    /// A pass of the engine's split head (`hall_ranks.Link.tellHead`): the spans, then each one's slot, the
    /// streams it reads (0 the round's rows, 1 the head's own), the first of them, its rows, its position and
    /// whether it is scored. The tokens are the spans', in that order.
    hall_head,
    /// The engine's states (`pool/hall_keep.zig`), each by its id (`Keep.id`). Keep slot's state where it
    /// stands: the id, the slot and the head's rows it holds.
    hall_keep,
    /// The states the next round's prompt chunk takes within it: the slot, how many, then each one's id and
    /// position.
    hall_wanted,
    /// Those states' PLE part after the round: the slot and the chunk's first row. The tokens are the chunk's.
    hall_within,
    /// Those states held, their pooled keys taken: the slot.
    hall_hold,
    /// States held no more: how many, then their ids.
    hall_drop,
    /// The slot goes on from a state: the slot, the state's id, whether rows are copied, and from which extent
    /// to which and how many.
    hall_fork,
    /// The pool's graphs a lone rank's load captures, captured here too, launching nothing
    /// (`pool/hall_ranks.zig`'s `Link.captureFirst`): the rows and the context bucket. Rank 0 tells each
    /// after the join's barrier; no tokens.
    hall_capture,
    /// A reply's rounds from here on are rank 1's to run as rank 0 runs them (`session/pair.zig`'s `Rounds`,
    /// spec 095): `Rounds.told`'s arguments, then the tokens chosen after the last window's rows and the
    /// reply's end tokens. Nothing more is told until the reply's last round; what rank 1 cannot derive comes
    /// inside the stream, a few words a round.
    decode,
    /// A reply's prompt from here on is rank 1's to run as rank 0 runs it (`session/pair.zig`'s `Prompt`,
    /// spec 097): `Prompt.arguments`, then the states that go, the states the prompt's windows take within
    /// themselves, the prompt's next tokens and the reply's end tokens — in frames as wide as the channel's
    /// bound, as many as the prompt needs. Nothing more is told until the prompt's last window is committed;
    /// rank 1 runs its windows from the frames, the head drafting between them as rank 0's does.
    prompt,
    /// The states asked of the next window, wider than the rows each layer keeps, before it runs (`ask`): the
    /// window's rows, how many, then the rows into it of each, rising. Each is then `keep_within`.
    keep_ahead,
    /// DFlash2: a context update (`flash_taps`: the context's end, the taps' first row and the rows) and a
    /// block (`flash_block`: the pending token). Several sequences in one pass (`flash_round`).
    flash_taps,
    flash_block,
    flash_round,
    /// The engine's round settled on the device (`pool/hall_ranks.zig`'s `Link.settle`, spec 144 WS-03): the
    /// round's reply rows. The tokens are the rank each row is cut at, 0 where it is not settled.
    hall_settle,
};

pub fn told(lead: ?*lead_file.Lead, op: Op, arguments: []const u32, tokens: []const i32) lead_file.Error!void {
    if (lead) |to| try to.tell(@backingInt(op), arguments, tokens);
}

/// The forward over `held` ranks in one process.
pub fn Held(comptime held: usize) type {
    comptime std.debug.assert(held == 1 or held == world);
    return struct {
        const Self = @This();
        pub const Error = Failure;

        pub const Model = struct {
            ranks: [held]model_file.Model,
            /// `ranks[0]`'s rank in the world.
            first: usize = 0,
            max_rows: usize,
            gather: ranks_file.Gather = .loopback,
            /// Run these ranks only and leave every other partial at zero. The add still runs.
            alone: bool = false,
            /// Set before the buffers are made: they keep this pointer, and `logitsRow` tells through it.
            lead: ?*lead_file.Lead = null,

            pub fn deinit(self: *Model) void {
                for (&self.ranks) |*one| one.deinit();
            }
        };

        pub const State = struct {
            ranks: [held]forward_file.State,
            pos: usize = 0,

            pub fn init(gpa: Allocator, m: Model, rows: usize, capacity: usize) !State {
                return initKeeping(gpa, m, rows, rows, capacity);
            }

            pub fn initKeeping(gpa: Allocator, m: Model, rows: usize, scored: usize, capacity: usize) !State {
                var self: State = .{ .ranks = undefined };
                var done: usize = 0;
                errdefer for (self.ranks[0..done]) |*one| one.deinit();
                for (&self.ranks, 0..) |*one, i| {
                    one.* = try .initKeeping(gpa, m.ranks[i], rows, scored, capacity);
                    done += 1;
                }
                return self;
            }

            pub fn deinit(self: *State) void {
                for (&self.ranks) |*one| one.deinit();
            }
        };

        pub const Buffers = struct {
            ranks: [held]forward_file.Buffers,
            rows: usize,
            /// fp32 [world, rows, hidden]: each rank's branch partial, tight for the window.
            branch: cuda.Buffer,
            /// fp32 [world, rows, hidden]: each rank's folded MoE partial.
            moe_parts: cuda.Buffer,
            /// bf16 [rows, hidden]: the branch sum, rounded once.
            branch_bf16: cuda.Buffer,
            /// fp32 [rows, hidden]: the MoE sum before its write-back rounds it.
            moe_sum: cuda.Buffer,
            /// What brings the other rank's half of a row of logits (`logitsRow`), the model's when these were
            /// made, and where: bf16 [world, vocab / world]. Null where every rank is here.
            gather: ranks_file.Gather,
            row: ?cuda.Buffer = null,
            /// `Model.lead` as it was when these were made. `logitsRow` has no model to read it from.
            lead: ?*lead_file.Lead = null,
            /// Where a window's rows are settled on the device (`settle`), over NCCL: u32 [world, scored rows, a
            /// pick's words over this rank's columns], every slot asked to rank 1, the largest; the host's copy.
            picks: ?cuda.Buffer = null,
            picked: []u32 = &.{},
            /// `settle` settles rows: a pair sets it while both ranks run a reply's rounds (`session/pair.zig`).
            settling: bool = false,
            gpa: Allocator,

            pub fn init(gpa: Allocator, m: Model, rows: usize, capacity: usize) !Buffers {
                return initFor(gpa, m, rows, rows, capacity);
            }

            pub fn initFor(gpa: Allocator, m: Model, rows: usize, scored: usize, capacity: usize) !Buffers {
                var self: Buffers = undefined;
                self.rows = rows;
                self.gpa = gpa;
                self.gather = m.gather;
                self.lead = m.lead;
                self.row = null;
                self.picks = null;
                self.picked = &.{};
                self.settling = false;
                var done: usize = 0;
                errdefer for (self.ranks[0..done]) |*one| one.deinit();
                for (&self.ranks, 0..) |*one, i| {
                    one.* = try .initFor(gpa, m.ranks[i], m.ranks[i].layers, rows, scored, capacity);
                    done += 1;
                }
                const hidden = m.ranks[0].shape.hidden;
                self.branch = try .init(.f32, &.{ world, rows, hidden });
                errdefer self.branch.deinit();
                self.moe_parts = try .init(.f32, &.{ world, rows, hidden });
                errdefer self.moe_parts.deinit();
                self.branch_bf16 = try .init(.bf16, &.{ rows, hidden });
                errdefer self.branch_bf16.deinit();
                self.moe_sum = try .init(.f32, &.{ rows, hidden });
                errdefer self.moe_sum.deinit();
                if (m.gather != .loopback) {
                    self.row = try .init(.bf16, &.{ world, m.ranks[0].head.n });
                    errdefer self.row.?.deinit();
                    const slot = picking.Picked.wordsFor(m.ranks[0].head.n);
                    self.picked = try gpa.alloc(u32, world * scored * slot);
                    errdefer gpa.free(self.picked);
                    // The kernel reads a slot's rank and never writes it, and the other rank's slots arrive asked
                    // alike, so a window of any rows finds its slots asked.
                    @memset(self.picked, 0);
                    for (0..world * scored) |at| self.picked[at * slot + picking.Picked.rank_word] = 1;
                    self.picks = try .init(.u32, &.{ world, scored, slot });
                    try self.picks.?.upload(std.mem.sliceAsBytes(self.picked));
                }
                return self;
            }

            pub fn scoredRows(self: Buffers) usize {
                return self.ranks[0].scoredRows();
            }

            /// Every rank here keeps the drafter's taps from here on (`forward.Buffers.setTaps`).
            pub fn setTaps(self: *Buffers, layers: []const u32, mean: taps_file.Mean) cuda.Error!void {
                for (&self.ranks) |*one| try one.setTaps(layers, mean);
            }

            pub fn deinit(self: *Buffers) void {
                if (self.row) |*row| row.deinit();
                if (self.picks) |*picks| picks.deinit();
                self.gpa.free(self.picked);
                for (&self.ranks) |*one| one.deinit();
                self.branch.deinit();
                self.moe_parts.deinit();
                self.branch_bf16.deinit();
                self.moe_sum.deinit();
            }
        };

        /// The forward as captured graphs, keyed as one rank's (`forward.Graphs`: the rows, the DeltaNet states'
        /// side, the context's bucket): `decode/graphs.zig`.
        pub const Graphs = graphs_file.Graphs(Graphed.Key, Graphed);

        const Graphed = struct {
            const Key = struct { rows: usize, current: u1, context: usize };
            pub const Model = Self.Model;
            pub const State = Self.State;
            pub const Buffers = Self.Buffers;
            pub const Error = Failure;

            pub fn enter(m: Self.Model, state: *Self.State, b: *Self.Buffers, tokens: []const i32) Failure!Key {
                const rows = tokens.len;
                try told(m.lead, .graph, &.{}, tokens);
                try stageAll(m, state, b, tokens);
                const lead = state.ranks[0];
                const context = forward_file.Graphs.bucket(b.ranks[0], lead.pos + rows);
                return .{ .rows = rows, .current = if (lead.current.len > 0) lead.current[0] else 0, .context = context };
            }

            pub fn compute(m: Self.Model, state: *Self.State, b: *Self.Buffers, rows: usize, key: Key) Failure!void {
                return launch(m, state, b, rows, if (b.ranks[0].attention != null) key.context else null, .all, null);
            }

            /// As `forward.Graphs.holds` says of one rank.
            pub fn ahead(state: Self.State, b: Self.Buffers, rows: usize) ?Key {
                const lead = state.ranks[0];
                const own = b.ranks[0];
                if (rows > own.scoredRows() or forward_file.widerThanKept(lead, own, rows)) return null;
                const side: u1 = if (lead.current.len > 0) lead.current[0] ^ @intFromBool(forward_file.lands(lead, own, rows)) else 0;
                return .{ .rows = rows, .current = side, .context = forward_file.Graphs.bucket(own, lead.pos + rows) };
            }
        };

        pub fn forward(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
            return forwardScoring(m, state, b, tokens, .all);
        }

        pub fn forwardScoring(m: Model, state: *State, b: *Buffers, tokens: []const i32, score: Score) Error!void {
            const which: u32 = @intCast(@backingInt(score));
            try told(m.lead, .forward, &.{which}, tokens);
            try stageAll(m, state, b, tokens);
            return launch(m, state, b, tokens.len, null, score, null);
        }

        /// `forward.expect` on every rank held here: the window after the next one staged, whose PLE table rows
        /// are gathered on a thread while the device runs the one before it (spec 087; on two ranks spec 111).
        pub fn expect(m: Model, b: *Buffers, next: []const i32) void {
            const n = if (m.alone) 1 else held;
            for (b.ranks[0..n]) |*one| forward_file.expect(one, next);
        }

        /// Every rank's host part of a window (`forward.stage`): ids, the PLE's rows, a landing replay.
        pub fn stageAll(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
            const n = if (m.alone) 1 else held;
            for (0..n) |i| try forward_file.stage(m.ranks[i], &state.ranks[i], &b.ranks[i], tokens);
        }

        pub fn commit(m: Model, state: *State, b: Buffers, tokens: []const i32, keep: usize) Error!void {
            try told(m.lead, .commit, &.{@intCast(keep)}, tokens);
            const n = if (m.alone) 1 else held;
            for (0..n) |i| try forward_file.commit(m.ranks[i], &state.ranks[i], b.ranks[i], tokens, keep);
            state.pos = state.ranks[0].pos;
        }

        /// The states of one sequence kept on the ranks held here, a rank's own each, ascending by position:
        /// both ranks take and drop the same ones, rank 0 deciding (`decode/book.zig`). The attention
        /// rows are not in a state: they stay in the cache, and a row never reads one at or past its own
        /// position. A state dropped leaves its device buffers to the next one taken (`decode/shelf.zig`).
        pub const Shelf = shelf_file.Shelf(Kept, Kept);
        pub const Kept = [held]forward_file.Snapshot;

        /// The state as it stands, committed, into `shelf`. A position is held once, and a state of a sequence
        /// left is never found at a position run again (`dropLocal` first).
        pub fn keepLocal(shelf: *Shelf, state: *const State) (cuda.Error || Allocator.Error || error{OutOfStep})!void {
            if (!shelf.above(state.pos)) return error.OutOfStep;
            try shelf.held.ensureUnusedCapacity(shelf.gpa, 1);
            var one: Kept = undefined;
            if (shelf.lend()) |over| {
                one = over;
                errdefer for (&one) |*snap| snap.deinit();
                for (&one, state.ranks) |*snap, rank| try snap.retake(rank);
            } else {
                var done: usize = 0;
                errdefer for (one[0..done]) |*snap| snap.deinit();
                for (&one, state.ranks) |*snap, rank| {
                    snap.* = try forward_file.Snapshot.take(shelf.gpa, rank);
                    done += 1;
                }
            }
            shelf.held.appendAssumeCapacity(one);
        }

        /// The states asked of the next window wider than the rows each layer keeps, before it runs
        /// (`Snapshot.ahead`), as one rank's session asks them: each rank's own, lowest first, whose DeltaNet
        /// states and conv inputs the window's forward writes as it passes each layer (`State.wanted`), and
        /// `keepWithinLocal` finishes once it has run. So a pair keeps a layer's rows of a window as a session
        /// does, and not the window's whole (spec 111's follow-up). Their buffers are the shelf's spares, and
        /// go back there where they are not taken.
        pub const Asked = struct {
            ranks: [held]std.ArrayList(forward_file.Snapshot) = @splat(.empty),
            /// The window's rows they were asked of.
            rows: usize = 0,

            pub fn count(self: *const Asked) usize {
                return self.ranks[0].items.len;
            }

            pub fn deinit(self: *Asked, gpa: Allocator) void {
                for (&self.ranks) |*list| {
                    for (list.items) |*one| one.deinit();
                    list.deinit(gpa);
                }
            }
        };

        /// The most states one window is asked for: far above what a plan holds (`book.Book.due`, the budget
        /// `Options.snapshots` plus one, 9; no flag sets it).
        pub const most_asked = 64;

        /// What was asked and not taken goes to the shelf's spares.
        pub fn dropAsked(shelf: *Shelf, asked: *Asked) void {
            while (asked.count() > 0) {
                var one: Kept = undefined;
                for (&one, &asked.ranks) |*snap, *list| snap.* = list.pop().?;
                shelf.spare.give(shelf.gpa, one);
            }
            asked.rows = 0;
        }

        /// The states `into` rows into the window of `rows` rows about to run from the state's position (rising,
        /// each 1 to `rows`), asked of it: it must be wider than the rows each layer keeps.
        pub fn askLocal(shelf: *Shelf, asked: *Asked, state: *State, b: *const Buffers, rows: usize, into: []align(1) const u32) (cuda.Error || Allocator.Error || error{OutOfStep})!void {
            dropAsked(shelf, asked);
            if (into.len == 0 or into.len > most_asked or !forward_file.widerThanKept(state.ranks[0], b.ranks[0], rows)) return error.OutOfStep;
            var below: usize = 0;
            for (into) |word| {
                const keep: usize = word;
                if (keep <= below or keep > rows) return error.OutOfStep;
                below = keep;
            }
            for (&asked.ranks) |*list| try list.ensureUnusedCapacity(shelf.gpa, into.len);
            for (into) |word| {
                var one: Kept = undefined;
                if (shelf.lend()) |over| {
                    one = over;
                    for (&one, state.ranks) |*snap, rank| snap.aheadOver(rank, word);
                } else {
                    var done: usize = 0;
                    errdefer for (one[0..done]) |*snap| snap.deinit();
                    for (&one, state.ranks) |*snap, rank| {
                        snap.* = try .ahead(shelf.gpa, rank, word);
                        done += 1;
                    }
                }
                for (one, &asked.ranks) |snap, *list| list.appendAssumeCapacity(snap);
            }
            asked.rows = rows;
            for (&state.ranks, asked.ranks) |*rank, list| rank.wanted = list.items;
        }

        /// `askLocal`, after rank 1 has been told.
        pub fn ask(m: Model, shelf: *Shelf, asked: *Asked, state: *State, b: *const Buffers, rows: usize, into: []const u32) (Error || Allocator.Error || error{OutOfStep})!void {
            if (into.len > most_asked) return error.OutOfStep;
            var words: [2 + most_asked]u32 = undefined;
            words[0] = @intCast(rows);
            words[1] = @intCast(into.len);
            @memcpy(words[2..][0..into.len], into);
            try told(m.lead, .keep_ahead, words[0 .. 2 + into.len], &.{});
            try askLocal(shelf, asked, state, b, rows, into);
        }

        /// The state after the first `keep` rows of the window `tokens`, which has run from the state's
        /// position and is not committed, into `shelf`: the window is still committed whole. One asked of the
        /// window before it ran is finished (`Snapshot.finish`); a window no wider than the rows each layer
        /// keeps is replayed to it (`Snapshot.within`).
        pub fn keepWithinLocal(shelf: *Shelf, asked: *Asked, m: Model, state: *const State, b: *const Buffers, tokens: []const i32, keep: usize) (cuda.Error || Allocator.Error || error{OutOfStep})!void {
            std.debug.assert(!m.alone and keep >= 1 and keep <= tokens.len);
            if (!shelf.above(state.pos + keep)) return error.OutOfStep;
            try shelf.held.ensureUnusedCapacity(shelf.gpa, 1);
            var one: Kept = undefined;
            const found = for (asked.ranks[0].items, 0..) |snap, at| {
                if (snap.pos == state.pos + keep) break at;
            } else null;
            if (found) |at| {
                if (asked.rows != tokens.len) return error.OutOfStep;
                for (&one, &asked.ranks) |*snap, *list| snap.* = list.orderedRemove(at);
                errdefer for (&one) |*snap| snap.deinit();
                for (&one, 0..) |*snap, i| try snap.finish(state.ranks[i], b.ranks[i], tokens);
                shelf.held.appendAssumeCapacity(one);
                return;
            }
            if (forward_file.widerThanKept(state.ranks[0], b.ranks[0], tokens.len)) return error.OutOfStep;
            if (shelf.lend()) |over| {
                one = over;
                errdefer for (&one) |*snap| snap.deinit();
                for (&one, 0..) |*snap, i| try snap.again(m.ranks[i], state.ranks[i], b.ranks[i], tokens, keep);
            } else {
                var done: usize = 0;
                errdefer for (one[0..done]) |*snap| snap.deinit();
                for (&one, 0..) |*snap, i| {
                    snap.* = try forward_file.Snapshot.within(shelf.gpa, m.ranks[i], state.ranks[i], b.ranks[i], tokens, keep);
                    done += 1;
                }
            }
            shelf.held.appendAssumeCapacity(one);
        }

        /// The state at `pos` is held no more; held still where its buffers find no room among the spares.
        pub fn dropLocal(shelf: *Shelf, pos: usize) (Allocator.Error || error{OutOfStep})!void {
            const i = shelf.find(pos) orelse return error.OutOfStep;
            try shelf.spare.reserve(shelf.gpa);
            shelf.drop(i);
        }

        /// `state` back at the state held at `pos`.
        pub fn backLocal(shelf: *const Shelf, state: *State, pos: usize) (cuda.Error || error{OutOfStep})!void {
            const one = shelf.held.items[shelf.find(pos) orelse return error.OutOfStep];
            for (one, &state.ranks) |snap, *rank| try snap.restore(rank);
            state.pos = state.ranks[0].pos;
        }

        /// The state into `shelf`, after rank 1 has been told.
        pub fn hold(m: Model, state: *State, shelf: *Shelf) (Error || Allocator.Error || error{OutOfStep})!void {
            try told(m.lead, .keep, &.{}, &.{});
            try keepLocal(shelf, state);
        }

        /// `keepWithinLocal`, after rank 1 has been told the rows and the window.
        pub fn holdWithin(m: Model, state: *State, b: *const Buffers, shelf: *Shelf, asked: *Asked, tokens: []const i32, keep: usize) (Error || Allocator.Error || error{OutOfStep})!void {
            try told(m.lead, .keep_within, &.{@intCast(keep)}, tokens);
            try keepWithinLocal(shelf, asked, m, state, b, tokens, keep);
        }

        /// `dropLocal`, rank 1 told first.
        pub fn drop(m: Model, shelf: *Shelf, pos: usize) (Error || Allocator.Error || error{OutOfStep})!void {
            try told(m.lead, .drop, &.{@intCast(pos)}, &.{});
            try dropLocal(shelf, pos);
        }

        /// `state` back at the state held at `pos`. Rank 1 is told first.
        pub fn back(m: Model, state: *State, shelf: *const Shelf, pos: usize) (Error || error{OutOfStep})!void {
            try told(m.lead, .back, &.{@intCast(pos)}, &.{});
            try backLocal(shelf, state, pos);
        }

        /// The halves in rank order, rank 0's first. `out` is the whole vocabulary. A process that holds one
        /// rank gathers the other's half of the row over NCCL (496 KiB for Flash Next's vocabulary): every rank
        /// asks for the same rows in the same order, as each decodes the same reply (`decode/settle.zig`).
        pub fn logitsRow(b: Buffers, row: usize, out: []u16) Error!void {
            try told(b.lead, .row, &.{@intCast(row)}, &.{});
            var logits: [held]cuda.Buffer = undefined;
            for (&logits, b.ranks) |*one, rank| one.* = rank.logits;
            return settle_file.logitsRow(world, held, logits, 0, b.gather, b.row, row, out);
        }

        /// `draft.Chooser`'s rows settled on the device (spec 095, `decode/settle.zig`): where `b.settling`, over
        /// NCCL and for a greedy choice, the picks of each rank's half of rows `row` to `row + rows` are gathered
        /// in one all-gather (some 9 KiB a row for the 496 KiB a row `logitsRow` brings) and joined. A row they do
        /// not settle stays null and is read. Every rank settles the same rows from the same picks.
        pub fn settle(b: Buffers, row: usize, sampler: *sampling.Sampler, settings: sampling.Sampling, out: []?i32) Error!void {
            @memset(out, null);
            if (held == world or !b.settling or !settings.greedy()) return;
            const picks = b.picks.?;
            std.debug.assert(row + out.len <= picks.view.size[1]);
            return settle_file.settle(world, b.gather.nccl, b.ranks[0].logits, picks, b.picked, row, sampler, settings, out);
        }

        pub fn streams(b: Buffers, rows: usize) cuda.View {
            return b.ranks[0].h.rows(rows);
        }

        /// One window, launches only (`stageAll` first), so a graph can hold it. `bound` is the context the
        /// attention's launches cover, as `forward.compute`'s (null: exactly the window's end). `tap` sees rank 0
        /// after each full-width point, and that view is copied to the other rank before it is read again: a
        /// comparison that writes the reference over the view feeds both.
        pub fn launch(m: Model, state: *State, b: *Buffers, rows: usize, bound: ?usize, score: Score, tap: ?forward_file.Tap) Error!void {
            std.debug.assert(m.alone or m.gather.serves(held, world));
            // the states asked of a wide window are written by its launches, and asked of no other (`askLocal`)
            defer for (&state.ranks) |*one| {
                one.wanted = &.{};
            };
            if (m.gather == .nccl) std.debug.assert(m.gather.nccl.rank == m.first);
            const n = if (m.alone) 1 else held;
            // each rank's form from the window's rows, which every rank holds (`forward.setForm`)
            var was: [held]usize = undefined;
            for (0..n) |i| was[i] = forward_file.setForm(m.ranks[i], forward_file.quantizedFrom(rows));
            defer for (0..n) |i| {
                _ = forward_file.setForm(m.ranks[i], was[i]);
            };
            const shape = m.ranks[0].shape;
            const lead = &b.ranks[0];
            // spec 106 WS-02: the glue split by rows from the last PLE on (its conv reads the rows before)
            // not with a drafter's taps, which read every row's streams after a tap layer (GLM's `splits`, spec
            // 107 WS-03): a split window's ranks hold their own rows' streams alone (spec 126 WS-04)
            const splitting = splits(m, rows, tap) and b.ranks[0].tap_layers.len == 0;
            const split_from = splitFrom(m.ranks[0]);
            for (0..n) |i| {
                const h = b.ranks[i].h.rows(rows);
                try hc.embed(b.ranks[i].ids.rows(rows), m.ranks[i].embed.view, h, shape.streams);
            }
            try mirror(tap, .embed_h, 0, lead.h, b, rows);

            var pending: [held]hc.Pending = @splat(.none);
            var linear: [held]usize = @splat(0);
            var attending: [held]usize = @splat(0);
            var branches: usize = 0;
            for (m.ranks[0].layers, 0..) |layer0, index| {
                if (layer0.ple != null) {
                    for (0..n) |i| {
                        if (pending[i] != .none) {
                            try hc.writeback(b.ranks[i].h.rows(rows), b.ranks[i].pss.view, pending[i], shape.streams);
                            pending[i] = .none;
                        }
                    }
                    // the reference takes h_in before the branch, as one rank does
                    try mirror(tap, .h_in, index, lead.h, b, rows);
                    for (0..n) |i| {
                        const layer = m.ranks[i].layers[index];
                        try ple.forward(layer.ple.?.weights, b.ranks[i].ple[branches], state.ranks[i].ple[branches], b.ranks[i].h.rows(rows), rows);
                    }
                    try mirror(tap, .h_after_ple, index, lead.h, b, rows);
                    if (tap) |t| if (!t.passive) for (0..n) |i| try hc.writeback(b.ranks[i].h.rows(rows), b.ranks[i].pss.view, .none, shape.streams);
                    branches += 1;
                }
                var sites: [held]forward_file.Site = undefined;
                for (0..n) |i| {
                    const layer = m.ranks[i].layers[index];
                    sites[i] = switch (layer.block) {
                        .delta_net => .{ .delta_net = .{ .state = &state.ranks[i], .linear = linear[i] } },
                        .attention => .{ .attention = .{ .cache = state.ranks[i].caches[attending[i]], .position = state.ranks[i].position, .bound = bound } },
                    };
                    switch (layer.block) {
                        .delta_net => linear[i] += 1,
                        .attention => attending[i] += 1,
                    }
                }
                if (splitting and index >= split_from) {
                    try splitLayer(m, b, rows, index, &sites, &pending);
                    continue;
                }
                for (0..n) |i| try hc.writeback(b.ranks[i].h.rows(rows), b.ranks[i].pss.view, pending[i], shape.streams);
                if (layer0.ple == null) {
                    try mirror(tap, .h_in, index, lead.h, b, rows);
                    if (tap) |t| if (!t.passive) for (0..n) |i| try hc.writeback(b.ranks[i].h.rows(rows), b.ranks[i].pss.view, .none, shape.streams);
                }
                for (0..n) |i| try forward_file.readout(m.ranks[i].layers[index].attn_hc, b.ranks[i], rows, shape, b.ranks[i].inject_attn);
                try mirror(tap, .attn_mixed, index, lead.mixed, b, rows);
                try mirror(tap, .attn_inject, index, lead.inject_attn, b, rows);
                for (0..n) |i| {
                    const layer = m.ranks[i].layers[index];
                    const part = ranks_file.partial(b.branch.view, m.first + i, rows);
                    _ = try forward_file.runBlock(m.ranks[i], layer, index, sites[i], &b.ranks[i], rows, null, b.ranks[i].mixed.rows(rows), part);
                }
                // The gate's alone rank adds against a partial left at zero (`Model.alone`). Over NCCL the
                // all-gather overwrites every word of the gathered view: nothing is zeroed first.
                if (m.alone) try cuda.zeroLaunched(ranks_file.partial(b.branch.view, if (m.first == 0) 1 else 0, rows));
                if (m.gather != .loopback) try m.gather.run(ranks_file.gathered(b.branch.view, rows));
                try partials.roundSum(ranks_file.gathered(b.branch.view, rows), b.branch_bf16.rows(rows));
                try mirror(tap, .branch, index, b.branch_bf16, b, rows);
                for (0..n) |i| {
                    const landing: hc.Pending = .{ .branch = .{ .branch = b.branch_bf16.rows(rows), .inject = b.ranks[i].inject_attn.view } };
                    try hc.writeback(b.ranks[i].h.rows(rows), b.ranks[i].pss.view, landing, shape.streams);
                }
                try mirror(tap, .h_mid, index, lead.h, b, rows);
                if (tap) |t| if (!t.passive) for (0..n) |i| try hc.writeback(b.ranks[i].h.rows(rows), b.ranks[i].pss.view, .none, shape.streams);
                for (0..n) |i| {
                    const layer = m.ranks[i].layers[index];
                    try forward_file.readout(layer.mlp_hc, b.ranks[i], rows, shape, b.ranks[i].inject_mlp);
                }
                try mirror(tap, .mlp_mixed, index, lead.mixed, b, rows);
                try mirror(tap, .mlp_inject, index, lead.inject_mlp, b, rows);
                const slots = shape.top + 1;
                for (0..n) |i| {
                    const layer = m.ranks[i].layers[index];
                    try moe.forward(layer.moe, b.ranks[i].moe, b.ranks[i].mixed.rows(rows), rows, m.ranks[i].scratch.quantize_from);
                    try partials.fold(b.ranks[i].moe.y.rows(rows * slots), b.ranks[i].moe.shared_y.rows(rows), b.ranks[i].moe.weights.rows(rows), ranks_file.partial(b.moe_parts.view, m.first + i, rows));
                }
                if (tap) |t| {
                    try forward_file.observe(t, .moe_logits, index, lead.moe.logits.rows(rows));
                    try forward_file.observe(t, .moe_picks, index, lead.moe.picks.rows(rows));
                    try forward_file.observe(t, .moe_weights, index, lead.moe.weights.rows(rows));
                }
                if (m.alone) try cuda.zeroLaunched(ranks_file.partial(b.moe_parts.view, if (m.first == 0) 1 else 0, rows));
                const gathered_moe = ranks_file.gathered(b.moe_parts.view, rows);
                if (m.gather != .loopback) try m.gather.run(gathered_moe);
                for (0..n) |i| pending[i] = .{ .ranks_sum = .{ .gathered = gathered_moe, .inject = b.ranks[i].inject_mlp.view, .into = b.moe_sum.rows(rows) } };
                // a DFlash2 tap layer's streams, written back here and meaned on each rank, as one rank's
                // `compute` does: the same write-back the next layer would launch, so no bit of h moves
                for (0..n) |i| try forward_file.capture(&b.ranks[i], index, rows, &pending[i], shape.streams);
            }
            if (splitting) {
                for (0..n) |i| try Glued.of(b.ranks[i], rows, m.first + i).end(m.ranks[i].mixer, shape, pending[i]);
                // what the head and the MTP head read whole (`streams`)
                try shareGlued(m, b, "mixed", rows);
                try shareGlued(m, b, "h", rows);
                return head(m, b, rows, score);
            }
            for (0..n) |i| try hc.writeback(b.ranks[i].h.rows(rows), b.ranks[i].pss.view, pending[i], shape.streams);
            try mirror(tap, .final_h, m.ranks[0].layers.len, lead.h, b, rows);
            if (tap) |t| if (!t.passive) for (0..n) |i| try hc.writeback(b.ranks[i].h.rows(rows), b.ranks[i].pss.view, .none, shape.streams);
            for (0..n) |i| try forward_file.readout(m.ranks[i].mixer, b.ranks[i], rows, shape, null);
            try mirror(tap, .final_mixed, m.ranks[0].layers.len, lead.mixed, b, rows);
            return head(m, b, rows, score);
        }

        fn head(m: Model, b: *Buffers, rows: usize, score: Score) Error!void {
            const n = if (m.alone) 1 else held;
            switch (score) {
                .all => for (0..n) |i| try m.ranks[i].head.forward(b.ranks[i].mixed.rows(rows), b.ranks[i].logits.rows(rows)),
                .last => for (0..n) |i| try m.ranks[i].head.forward(b.ranks[i].mixed.span(rows - 1, 1), b.ranks[i].logits.rows(1)),
                .none => {},
            }
        }

        /// Whether a window of `rows` splits its glue by rows over the ranks (spec 106 WS-02): over NCCL or in
        /// loopback, every rank given rows, and nothing observing the full-width points.
        pub fn splits(m: Model, rows: usize, tap: ?forward_file.Tap) bool {
            if (m.alone or tap != null or rows < split_rows) return false;
            for (0..world) |rank| {
                const own = ownRows(rows, rank);
                if (own[1] == own[0]) return false;
            }
            return true;
        }

        /// One layer from its attention read-out on, the glue split (spec 106 WS-02): each rank writes back and
        /// reads out its own rows (`ranks.share`), shares its read-out's rows before each block, which reads every
        /// row, and after each block receives only its own rows of the other ranks' partials (`Gather.scatter`)
        /// and adds them rank 0 first. A row's bits are `launch`'s whole form's: the same kernels on fewer rows.
        /// `pending` is each rank's landing on its own rows, `.none` or `.sum`, and is left so.
        fn splitLayer(m: Model, b: *Buffers, rows: usize, index: usize, sites: *const [held]forward_file.Site, pending: *[held]hc.Pending) Error!void {
            const n = if (m.alone) 1 else held;
            const shape = m.ranks[0].shape;
            var glued: [held]Glued = undefined;
            for (0..n) |i| glued[i] = .of(b.ranks[i], rows, m.first + i);
            for (0..n) |i| try glued[i].before(m.ranks[i].layers[index].attn_hc, shape, pending[i]);
            try shareGlued(m, b, "mixed", rows);
            for (0..n) |i| {
                const layer = m.ranks[i].layers[index];
                const part = ranks_file.partial(b.branch.view, m.first + i, rows);
                _ = try forward_file.runBlock(m.ranks[i], layer, index, sites[i], &b.ranks[i], rows, null, b.ranks[i].mixed.rows(rows), part);
            }
            const branch = ranks_file.gathered(b.branch.view, rows);
            try m.gather.scatter(branch);
            for (0..n) |i| try glued[i].landed(branch, b.branch_bf16, m.ranks[i].layers[index].mlp_hc, shape);
            try shareGlued(m, b, "mixed", rows);
            const slots = shape.top + 1;
            for (0..n) |i| {
                const layer = m.ranks[i].layers[index];
                try moe.forward(layer.moe, b.ranks[i].moe, b.ranks[i].mixed.rows(rows), rows, m.ranks[i].scratch.quantize_from);
                try partials.fold(b.ranks[i].moe.y.rows(rows * slots), b.ranks[i].moe.shared_y.rows(rows), b.ranks[i].moe.weights.rows(rows), ranks_file.partial(b.moe_parts.view, m.first + i, rows));
            }
            const parts = ranks_file.gathered(b.moe_parts.view, rows);
            try m.gather.scatter(parts);
            for (0..n) |i| pending[i] = try glued[i].folded(parts, b.moe_sum);
        }

        /// Every rank's own rows of the buffer `name` of `b.ranks` brought to every rank held (spec 106 WS-02):
        /// over NCCL in place, in loopback a copy between the held ranks' buffers.
        fn shareGlued(m: Model, b: *Buffers, comptime name: []const u8, rows: usize) Error!void {
            if (held == 1) return m.gather.shareRows(@field(b.ranks[0], name).rows(rows));
            for (0..held) |to| for (0..held) |from| {
                if (to == from) continue;
                const theirs = ownRows(rows, m.first + from);
                try @field(b.ranks[to], name).copyRows(theirs[0], @field(b.ranks[from], name), theirs[0], theirs[1] - theirs[0]);
            };
        }
    };
}

/// The fewest rows a window splits its glue at over the ranks (spec 106 WS-02): a prompt's chunks, not a
/// decoding round's windows, whose few rows gain nothing from two exchanges a site for one. A test lowers it;
/// `ranks-check`'s windows (16 rows at most) stay under it, since it compares every rank's whole partials.
pub var split_rows: usize = 256;

/// The first layer whose glue splits: the last PLE's, whose convolution reads the rows before it (spec 106
/// WS-02), in a pair's window and a pool's round alike (spec 149 WS-02).
pub fn splitFrom(share: model_file.Model) usize {
    var from: usize = 0;
    for (share.layers, 0..) |layer, index| {
        if (layer.ple != null) from = index;
    }
    return from;
}

fn ownRows(rows: usize, rank: usize) [2]usize {
    return ranks_file.share(rows, world, rank);
}

/// One rank's part of a window's glue split by rows (spec 106 WS-02): its own rows of `b` (`ranks.share`),
/// written back and read out by the whole window's kernels. A pair's `splitLayer` runs it for each rank it
/// holds, a pool's round on two ranks for its own (`pool/rounds_parts.zig`, spec 149 WS-02); what the ranks
/// share or scatter between the steps is the caller's.
pub const Glued = struct {
    g: forward_file.Buffers,
    first: usize,
    count: usize,

    pub fn of(b: forward_file.Buffers, rows: usize, rank: usize) Glued {
        const own = ownRows(rows, rank);
        return .{ .g = glueBuffers(b, own[0], own[1] - own[0]), .first = own[0], .count = own[1] - own[0] };
    }

    /// Before a block: `pending`, this rank's landing on its own rows (`.none` or `.sum`), written back, and
    /// the block's read-out of them.
    pub fn before(self: Glued, connection: model_file.HyperConnection, shape: model_file.Shape, pending: hc.Pending) cuda.Error!void {
        try hc.writeback(self.g.h.view, self.g.pss.view, pending, shape.streams);
        try forward_file.readout(connection, self.g, self.count, shape, self.g.inject_attn);
    }

    /// After the block's partials were scattered (`branch`, fp32 [world, rows, hidden]): the own rows' sum, rank
    /// 0 first and rounded once into those of `branch_bf16`, written back, and the MLP's read-out of them.
    pub fn landed(self: Glued, branch: cuda.View, branch_bf16: cuda.Buffer, connection: model_file.HyperConnection, shape: model_file.Shape) cuda.Error!void {
        const own = branch_bf16.span(self.first, self.count);
        try ranks_file.sumRows(branch, self.first, own);
        try hc.writeback(self.g.h.view, self.g.pss.view, .{ .branch = .{ .branch = own, .inject = self.g.inject_attn.view } }, shape.streams);
        try forward_file.readout(connection, self.g, self.count, shape, self.g.inject_mlp);
    }

    /// After the MoE's partials were scattered (`parts`): the own rows' sum into those of `moe_sum`, the
    /// landing the next layer's `before` (or `end`) writes back.
    pub fn folded(self: Glued, parts: cuda.View, moe_sum: cuda.Buffer) cuda.Error!hc.Pending {
        const sum = moe_sum.span(self.first, self.count);
        try ranks_file.sumRows(parts, self.first, sum);
        return .{ .sum = .{ .sum = sum, .inject = self.g.inject_mlp.view } };
    }

    /// After the last layer: `pending` written back and the final read-out, the own rows.
    pub fn end(self: Glued, mixer: model_file.HyperConnection, shape: model_file.Shape, pending: hc.Pending) cuda.Error!void {
        try hc.writeback(self.g.h.view, self.g.pss.view, pending, shape.streams);
        try forward_file.readout(mixer, self.g, self.count, shape, null);
    }
};

/// `b` as the glue of rows [first, first + count) sees it: every buffer the write-back and the read-out
/// index by row starts at `first` (`forward.readout` reads from a buffer's row 0). Views only: never freed.
fn glueBuffers(b: forward_file.Buffers, first: usize, count: usize) forward_file.Buffers {
    var g = b;
    inline for (.{ "h", "pss", "normed", "down", "down_mixer", "act", "up", "inject_attn", "inject_mlp", "mixed" }) |name| {
        @field(g, name) = .{ .view = @field(b, name).span(first, count) };
    }
    g.rows = count;
    return g;
}

/// Rank 0's `point`, then the same bytes on the other rank when the tap may have written over them.
fn mirror(tap: ?forward_file.Tap, point: forward_file.Point, layer: usize, lead: cuda.Buffer, b: anytype, rows: usize) Failure!void {
    const t = tap orelse return;
    try forward_file.observe(t, point, layer, lead.rows(rows));
    if (b.ranks.len < 2 or point == .branch) return;
    const other = switch (point) {
        .embed_h, .h_after_ple, .h_in, .h_mid, .final_h => b.ranks[1].h,
        .attn_mixed, .mlp_mixed, .final_mixed => b.ranks[1].mixed,
        .attn_inject => b.ranks[1].inject_attn,
        .mlp_inject => b.ranks[1].inject_mlp,
        else => return,
    };
    try other.copyRows(0, lead, 0, rows);
}
