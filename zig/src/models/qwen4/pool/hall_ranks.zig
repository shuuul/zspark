//! The pool's `Engine` on one rank of two (spec 094 WS-03b): `serve --parallel N` over two ranks. Rank 0's
//! engine is the hall's, scheduler and all, over its share; before each operation on the device it tells rank 1
//! (`ranks_forward.zig`'s `Op`), whose engine holds the same slots and pools laid out alike, decides nothing and
//! runs what it is told (`follow`), refusing what does not fit what it holds. A round is `rounds.computeOver`
//! with the partials of `rounds_parts.zig`, so a row's bits are those of its stream alone on the two ranks; a
//! row of logits is gathered, half of the vocabulary a rank, as `ranks_forward.Held.logitsRow` gathers a lone
//! stream's. The extents are rank 0's: rank 1 is told the entries of a stream's table that changed.
//!
//! The MTP head drafts each stream as one rank's pool does. Whole (`ZSPARK_RANKS_HEAD=whole`), rank 0 holds it
//! beside its share and runs its passes alone (`mtp.over`). Split, each rank holds half of it: rank 0 tells each
//! pass (`Op.hall_head`), both run it, its attention's and MoE's partials meet as a round's do, and each rank
//! scores its part of the draft ids, gathered into the seats' rows (`scoreHead`). Rank 0 alone chooses the drafts.
//!
//! The states a stream keeps (`hall_keep.zig`) are rank 0's to place and let go, by its prefix store; rank 1
//! keeps its own half of each, by the id rank 0 gives it (`Keep.id`), told where it is taken (where a slot
//! stands, or within a prompt's chunk), when it is held, let go, and forked from.
//!
//! A round of replies replays a graph, as on one rank (`rounds.Graphs`, its all-gathers inside it); a round with
//! a prompt's chunk launches its kernels. One rank's engine holds no `Link` and runs as it did.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const channel_file = @import("../../../ranks/channel.zig");
const extents = @import("../../../decode/extents.zig");
const lead_file = @import("../../../ranks/lead.zig");
const shared_draft = @import("../../../decode/draft.zig");
const pick_file = @import("../../../decode/pick.zig");
const settle_file = @import("../../../decode/settle.zig");
const flash_follow = @import("../../../decode/dflash2/follow.zig");
const flash_pass = @import("../dflash2/pass.zig");
const forward = @import("../forward.zig");
const gdn = @import("../blocks/gdn.zig");
const held_file = @import("../ranks_forward.zig");
const share_file = @import("../share.zig");
const engine_file = @import("hall_engine.zig");
const head_file = @import("hall_head.zig");
const hall_seats = @import("hall_seats.zig");
const keep_file = @import("hall_keep.zig");
const ranks_file = @import("../../../ranks/ranks.zig");
const rounds = @import("rounds.zig");
const stage_file = @import("rounds_stage.zig");
const Span = hall_seats.Span;
const Engine = engine_file.Engine;

const H = held_file.Held(1);
const Op = held_file.Op;
pub const world = share_file.world;

pub const Error = H.Error || Allocator.Error || error{OutOfStep};

/// Words of a round's frame before its tokens: the score, the rows the main head scores, the parts, the graph,
/// and the prompt's tokens told after the round's (`rounds.askNext`).
const round_words = 5;
/// The states a prompt's chunk takes within it, told in one frame: one a row of the widest chunk (4,096, spec
/// 149 WS-06) fits `Lead.widest`.
const most_wanted = 4096;
/// Words a span of a head's pass takes (`tellHead`).
const span_words = 6;

/// What an engine holds to run on one rank of two: the model as the ranks see it, the partials' buffers,
/// and on rank 0 the tables rank 1 was told; on rank 1 the round it ran last and has not committed.
pub const Link = struct {
    /// `ranks[0]` is the engine's model, this rank's share; its `lead` tells rank 1, null on rank 1.
    hm: H.Model,
    /// fp32 [world, rows, hidden] each: the branch's partials and the MoE's.
    branch: cuda.Buffer,
    moe_parts: cuda.Buffer,
    /// bf16 [rows, hidden] and fp32 [rows, hidden]: the sums.
    branch_bf16: cuda.Buffer,
    moe_sum: cuda.Buffer,
    /// bf16 [world, vocab / world]: a gathered row of logits; null alone (`H.Model.alone`).
    row: ?cuda.Buffer,
    /// [slot][extents]: each stream's table as rank 1 holds it, and how many entries stand.
    tables: [][]i32,
    placed: []usize,
    /// Rank 1: the round run last and not committed, its parts and its tokens.
    open: []rounds.Segment,
    open_parts: usize = 0,
    /// The last committed round's layout, retained until its optional flash round consumes the taps.
    committed_parts: usize = 0,
    tokens: []i32,
    /// A split head's: bf16 [slots, a rank's most ids], this rank's part of a pass's scored rows, tight; and
    /// [world, slots * that] every rank's, gathered. Null where the head is whole or none.
    score_part: ?cuda.Buffer = null,
    score_all: ?cuda.Buffer = null,
    /// Rank 0: the ids of the states let go, told before its next state or round (`flush`).
    let_go: std.ArrayList(u32) = .empty,
    /// Rank 0: states let go and told so.
    told_gone: usize = 0,
    /// Rank 1: its half of the states rank 0 keeps, by id; those the open round's prompt chunk takes within
    /// it, until they are held; and buffers of states let go, for the next.
    held: std.AutoHashMapUnmanaged(u32, Kept) = .empty,
    pending: std.ArrayList(Kept) = .empty,
    spare: std.ArrayList(forward.Snapshot) = .empty,
    /// The engine this rank runs, `bind`'s. Null before it.
    engine: ?*Engine = null,
    /// Over NCCL: the words a round's operations are told in from its settle or its commit to its end
    /// (`heat`, `cool`; spec 144 WS-10), the head's passes among them. Null alone and over the loopback.
    said: ?ranks_file.Said = null,
    gpa: Allocator,

    /// A frame told in `said`'s words at most: a head's pass of 16 seats one row each (1 + 6 * 16 + 16 words),
    /// or 4 seats' kept windows; a wider one goes on the channel, the words saying so.
    pub const said_width = 256;

    /// A state rank 1 keeps: `hall_keep.Keep`'s device part.
    pub const Kept = struct {
        id: u32,
        pos: usize,
        head_len: usize,
        state: forward.Snapshot,
        pooled: cuda.Parts = .{},

        fn freePooled(self: *Kept) void {
            self.pooled.deinit(self.state.gpa);
        }
    };

    /// Ready buffers rank 1 holds at most.
    const spares = 8;

    /// For an engine of `slots` streams of `capacity` positions and rounds of `hm.max_rows` rows.
    pub fn init(gpa: Allocator, hm: H.Model, slots: usize, capacity: usize) !Link {
        const share = hm.ranks[0];
        const rows = hm.max_rows;
        const hidden = share.shape.hidden;
        var branch: cuda.Buffer = try .init(.f32, &.{ world, rows, hidden });
        errdefer branch.deinit();
        var moe_parts: cuda.Buffer = try .init(.f32, &.{ world, rows, hidden });
        errdefer moe_parts.deinit();
        var branch_bf16: cuda.Buffer = try .init(.bf16, &.{ rows, hidden });
        errdefer branch_bf16.deinit();
        var moe_sum: cuda.Buffer = try .init(.f32, &.{ rows, hidden });
        errdefer moe_sum.deinit();
        var row: ?cuda.Buffer = if (hm.gather != .loopback) try .init(.bf16, &.{ world, share.head.n }) else null;
        errdefer if (row) |*held| held.deinit();
        const entries = @max(extents.extentsFor(capacity), 1);
        const tables = try gpa.alloc([]i32, slots);
        errdefer gpa.free(tables);
        var made: usize = 0;
        errdefer for (tables[0..made]) |table| gpa.free(table);
        for (tables) |*table| {
            table.* = try gpa.alloc(i32, entries);
            @memset(table.*, 0);
            made += 1;
        }
        const placed = try gpa.alloc(usize, slots);
        errdefer gpa.free(placed);
        @memset(placed, 0);
        const open = try gpa.alloc(rounds.Segment, slots);
        errdefer gpa.free(open);
        const tokens = try gpa.alloc(i32, lead_file.Lead.widest);
        errdefer gpa.free(tokens);
        var score_part: ?cuda.Buffer = null;
        var score_all: ?cuda.Buffer = null;
        if (share.mtp) |head| if (head.split) |split| {
            score_part = try .init(.bf16, &.{ slots, split.widest() });
            errdefer score_part.?.deinit();
            score_all = try .init(.bf16, &.{ world, slots * split.widest() });
        };
        errdefer if (score_part) |*part| part.deinit();
        errdefer if (score_all) |*all| all.deinit();
        const said: ?ranks_file.Said = switch (hm.gather) {
            .nccl => |comm| try .init(comm, said_width),
            .loopback => null,
        };
        return .{ .hm = hm, .branch = branch, .moe_parts = moe_parts, .branch_bf16 = branch_bf16, .moe_sum = moe_sum, .row = row, .tables = tables, .placed = placed, .open = open, .tokens = tokens, .score_part = score_part, .score_all = score_all, .said = said, .gpa = gpa };
    }

    /// Frees what `init` made; the model is the caller's.
    pub fn deinit(self: *Link) void {
        const gpa = self.gpa;
        gpa.free(self.tokens);
        gpa.free(self.open);
        gpa.free(self.placed);
        for (self.tables) |table| gpa.free(table);
        gpa.free(self.tables);
        if (self.row) |*row| row.deinit();
        if (self.score_part) |*part| part.deinit();
        if (self.score_all) |*all| all.deinit();
        if (self.said) |*said| said.deinit();
        if (self.hm.lead) |lead| lead.over = null;
        self.let_go.deinit(gpa);
        var kept = self.held.valueIterator();
        while (kept.next()) |one| {
            one.freePooled();
            one.state.deinit();
        }
        self.held.deinit(gpa);
        for (self.pending.items) |*one| {
            one.freePooled();
            one.state.deinit();
        }
        self.pending.deinit(gpa);
        for (self.spare.items) |*state| state.deinit();
        self.spare.deinit(gpa);
        inline for (.{ "branch", "moe_parts", "branch_bf16", "moe_sum" }) |name| @field(self, name).deinit();
        self.* = undefined;
    }

    pub fn parts(self: *const Link) rounds.Parts {
        return .{ .branch = self.branch.view, .moe = self.moe_parts.view, .branch_bf16 = self.branch_bf16, .moe_sum = self.moe_sum, .gather = self.hm.gather, .rank = self.hm.first };
    }

    /// The engine's buffers as the ranks' forward names them: a view, nothing owned.
    pub fn buffers(self: *const Link, b: forward.Buffers) H.Buffers {
        return .{ .ranks = .{b}, .rows = self.hm.max_rows, .branch = self.branch, .moe_parts = self.moe_parts, .branch_bf16 = self.branch_bf16, .moe_sum = self.moe_sum, .gather = self.hm.gather, .row = self.row, .lead = self.hm.lead, .gpa = self.gpa };
    }

    /// Rank 1 is told slot `slot`'s table where it changed (`rounds_ranks.lay`); rank 0's own is set already.
    pub fn lay(self: *Link, slot: usize, table: []const i32) Error!void {
        const known = self.tables[slot][0..self.placed[slot]];
        var from: usize = 0;
        while (from < known.len and from < table.len and known[from] == table[from]) from += 1;
        if (from == known.len and from == table.len) return;
        if (table.len > self.tables[slot].len or 1 + 3 + table.len - from > lead_file.Lead.widest) return error.OutOfStep;
        try held_file.told(self.hm.lead, .table, &.{ @intCast(slot), @intCast(from), @intCast(table.len) }, table[from..]);
        @memcpy(self.tables[slot][from..table.len], table[from..]);
        self.placed[slot] = table.len;
    }

    /// Rank 1: entries `from ..` of slot `slot`'s table are `entries`, as `rounds_ranks.Desk.place`.
    fn place(self: *Link, engine: *Engine, slot: usize, from: usize, entries: []const i32) (cuda.Error || error{OutOfStep})!void {
        if (slot >= self.tables.len) return error.OutOfStep;
        const table = self.tables[slot];
        if (from > self.placed[slot] or from + entries.len > table.len) return error.OutOfStep;
        const count = engine.pool.refs.len;
        for (entries) |entry| if (entry < 0 or entry >= count) return error.OutOfStep;
        @memcpy(table[from..][0..entries.len], entries);
        self.placed[slot] = from + entries.len;
        if (entries.len > 0) try engine.slots[slot].table.device.uploadAt(from * @sizeOf(i32), std.mem.sliceAsBytes(entries));
    }

    /// A round of replies only as one rank's graph of `{rows, context}` runs it, scoring every row
    /// (`rounds.Graphs`), rank 1 told first (`tellRound`, graph `context + 1`): captured where it is not yet, on
    /// both ranks, and launched in step.
    pub fn launchGraph(self: *Link, engine: *Engine, rows: usize, context: usize) !void {
        try self.captureGraph(engine, rows, context);
        try engine.graphs.launch(rows, context);
    }

    /// Rank 1 told the round, before rank 0 stages it (spec 144 WS-10: the two stage at once): its streams'
    /// rows, how it scores, and its graph's context bucket and 1 (0: launched kernel by kernel, `compute`); and
    /// the first of `next`, the prompt's tokens after its chunk, as many as the frame holds, which both ranks
    /// ask ahead once the round is staged (`rounds.askNext`, spec 149 WS-02). Returns those told.
    pub fn tellRound(self: *Link, segments: []const rounds.Segment, score: forward.Score, head_rows: usize, graph: usize, next: []const i32) ![]const i32 {
        try self.flush();
        var words: [round_words + 2 * 64]u32 = undefined;
        if (segments.len > 64) return error.OutOfStep;
        words[0] = @intCast(@backingInt(score));
        words[1] = @intCast(head_rows);
        words[2] = @intCast(segments.len);
        words[3] = @intCast(graph);
        var rows: usize = 0;
        for (segments, 0..) |segment, at| {
            words[round_words + 2 * at] = @intCast(segment.slot);
            words[round_words + 2 * at + 1] = @intCast(segment.tokens.len);
            @memcpy(self.tokens[rows..][0..segment.tokens.len], segment.tokens);
            rows += segment.tokens.len;
        }
        const arguments = words[0 .. round_words + 2 * segments.len];
        if (1 + arguments.len + rows > lead_file.Lead.widest) return error.OutOfStep;
        const told = next[0..@min(next.len, lead_file.Lead.widest - (1 + arguments.len + rows))];
        words[4] = @intCast(told.len);
        @memcpy(self.tokens[rows..][0..told.len], told);
        try held_file.told(self.hm.lead, .hall_round, arguments, self.tokens[0 .. rows + told.len]);
        return told;
    }

    /// The round's launches over the share, rank 1 told first (`tellRound`, graph 0): the segments' rows
    /// through every layer, and `head_rows` of them through the main head (a rank's half of each row's logits).
    pub fn compute(self: *Link, engine: *Engine, segments: []const rounds.Segment, score: forward.Score, head_rows: usize) !void {
        const landing = self.parts();
        try rounds.computeOver(engine.m, engine.slots, &engine.shared, &engine.buffers, &engine.staged, segments, null, score, null, engine.saves, &landing);
        if (head_rows > 0) try engine.m.head.forward(engine.buffers.mixed.rows(head_rows), engine.buffers.logits.rows(head_rows));
    }

    /// The main head's half of row `from`'s logits into row `to`, rank 1 told first.
    pub fn scoreRow(self: *Link, engine: *Engine, from: usize, to: usize) Error!void {
        try held_file.told(self.hm.lead, .hall_logits, &.{ @intCast(from), @intCast(to) }, &.{});
        try engine.m.head.forward(engine.buffers.mixed.span(from, 1), engine.buffers.logits.span(to, 1));
    }

    /// Rank 1 told to settle the round's reply rows on the device with rank 0 (`Engine.settleRound`): the rows,
    /// and as tokens the rank each is cut at (`decode/settle.zig`'s `asksOf`). Both then launch the same picks
    /// and meet in their gather.
    pub fn settle(self: *Link, asks: []const u8) Error!void {
        for (self.tokens[0..asks.len], asks) |*word, ask| word.* = ask;
        try held_file.told(self.hm.lead, .hall_settle, &.{@intCast(asks.len)}, self.tokens[0..asks.len]);
        self.heat();
    }

    /// Commits the round as `rounds.commit` does, rank 1 told how many rows each segment keeps first.
    pub fn commit(self: *Link, engine: *Engine, segments: []const rounds.Segment, partial: bool) !void {
        var words: [1 + 64]u32 = undefined;
        words[0] = @intCast(segments.len);
        for (segments, 0..) |segment, at| words[1 + at] = @intCast(segment.keep);
        try held_file.told(self.hm.lead, .hall_commit, words[0 .. 1 + segments.len], &.{});
        self.heat();
        if (partial) try rounds.replayKept(engine.m, engine.slots, &engine.staged, segments, engine.saves);
        try rounds.commit(engine.m, engine.slots, &engine.buffers, &engine.staged, segments);
    }

    /// Rank 0, once it has told a round's settle or its commit: what it tells from here to the round's end
    /// (`cool`) goes in `said`'s words, where rank 1, which heats after the same two (`follow`), listens. Its
    /// host has waited for the device by then, so a word on the link is all a pass of the head waits for
    /// (spec 144 WS-07: rank 1 started each pass 113 to 293 us late on the channel). Before the settle the
    /// channel stays: rank 1 takes the settle's frame while its round's graph runs, where words would wait for
    /// the graph's end.
    fn heat(self: *Link) void {
        if (self.hm.lead) |lead| lead.heat();
    }

    /// Rank 0, at a round's end, its last operation told (`hall_engine.roundInner`): rank 1 back on the
    /// channel, where it may wait idle. Nothing where the round never heated.
    pub fn cool(self: *Link) void {
        if (self.hm.lead) |lead| lead.cool() catch {};
    }

    /// Slot `slot` begins again, rank 1 told first.
    pub fn reset(self: *Link, slot: usize) Error!void {
        try held_file.told(self.hm.lead, .hall_reset, &.{@intCast(slot)}, &.{});
    }

    /// `engine` runs as this rank of two. Its rounds of replies replay the graphs one rank's would, captured
    /// at the load (`Link.captureFirst`) or as they are first needed under `--lazy-graphs`, on both ranks,
    /// with the all-gathers among its kernels (spec 050's way). The rounds' `eager` is the load's
    /// `--lazy-graphs` (`EngineOptions`), as one rank's is.
    pub fn bind(self: *Link, engine: *Engine) void {
        engine.ranks = self;
        self.engine = engine;
        // `said` stays where the link does from here: the lead holds its address
        if (self.hm.lead) |lead| lead.over = if (self.said) |*said| said.over() else null;
    }

    /// The graphs a lone rank's load captures (`Engine.captureFirst`), captured here too: rank 0 tells each
    /// (`Op.hall_capture`) and both ranks capture in step, launching nothing. Rank 0, after the join's
    /// barrier, with rank 1 in `follow`.
    pub fn captureFirst(self: *Link) !void {
        const engine = self.engine.?;
        const first = forward.Graphs.bucket(engine.buffers, 1);
        // A graph replays through the short projection, which holds `kept` rows. A reply window fits in that;
        // a wider round launches directly.
        const limit = @min(engine.buffers.kept, gdn.wide_rows - 1);
        for (1..limit + 1) |rows| {
            try held_file.told(self.hm.lead, .hall_capture, &.{ @intCast(rows), @intCast(first) }, &.{});
            try self.captureGraph(engine, rows, first);
        }
    }

    fn captureGraph(self: *Link, engine: *Engine, rows: usize, context: usize) !void {
        const landing = self.parts();
        try engine.graphs.prepareOver(engine.m, engine.slots, &engine.shared, &engine.buffers, &engine.staged, rows, context, engine.saves, &landing);
    }

    /// Rank 0: the state of id `id` goes; rank 1 is told before the next state or round.
    pub fn dropped(self: *Link, id: u32) void {
        if (id == 0 or self.hm.lead == null) return;
        self.let_go.append(self.gpa, id) catch std.log.warn("a state let go was not told to rank 1: it holds it until the end", .{});
    }

    fn flush(self: *Link) Error!void {
        var words: [1 + 1024]u32 = undefined;
        while (self.let_go.items.len > 0) {
            const n = @min(self.let_go.items.len, 1024);
            words[0] = @intCast(n);
            @memcpy(words[1..][0..n], self.let_go.items[self.let_go.items.len - n ..]);
            try held_file.told(self.hm.lead, .hall_drop, words[0 .. 1 + n], &.{});
            self.told_gone += n;
            self.let_go.shrinkRetainingCapacity(self.let_go.items.len - n);
        }
    }

    /// Slot `slot`'s state where it stands is kept as `id`, its head's `head_len` rows (`hall_keep.remember`).
    pub fn keepTake(self: *Link, slot: usize, id: u32, head_len: usize) Error!void {
        try self.flush();
        try held_file.told(self.hm.lead, .hall_keep, &.{ id, @intCast(slot), @intCast(head_len) }, &.{});
    }

    /// The states the next round's chunk of slot `slot` takes within it (`hall_keep.before`).
    pub fn keepWanted(self: *Link, slot: usize, pending: []const keep_file.Keep) Error!void {
        try self.flush();
        var words: [2 + 2 * most_wanted]u32 = undefined;
        if (pending.len > most_wanted) return error.OutOfStep;
        words[0] = @intCast(slot);
        words[1] = @intCast(pending.len);
        for (pending, 0..) |keep, at| {
            words[2 + 2 * at] = keep.id;
            words[3 + 2 * at] = @intCast(keep.pos);
        }
        try held_file.told(self.hm.lead, .hall_wanted, words[0 .. 2 + 2 * pending.len], &.{});
    }

    /// Their PLE part, the round run (`hall_keep.within`).
    pub fn keepWithin(self: *Link, slot: usize, tokens: []const i32, row: usize) Error!void {
        try held_file.told(self.hm.lead, .hall_within, &.{ @intCast(slot), @intCast(row) }, tokens);
    }

    /// Their pooled keys, the head having taken the chunk in, and held (`hall_keep.after`).
    pub fn keepHold(self: *Link, slot: usize) Error!void {
        try held_file.told(self.hm.lead, .hall_hold, &.{@intCast(slot)}, &.{});
    }

    /// Slot `slot`, just reset, goes on from the state `id`, `copy` copied in the pools first (`applyFork`).
    pub fn keepFork(self: *Link, slot: usize, id: u32, copy: ?extents.Copy) Error!void {
        if (id == 0) return error.OutOfStep;
        try self.flush();
        const c = copy orelse extents.Copy{ .from = 0, .to = 0, .rows = 0 };
        try held_file.told(self.hm.lead, .hall_fork, &.{ @intCast(slot), id, @intFromBool(copy != null), @bitCast(c.from), @bitCast(c.to), @intCast(c.rows) }, &.{});
    }

    /// Rank 1: a state's buffers at `pos`, ready ones first.
    fn blank(self: *Link, engine: *Engine, pos: usize) !forward.Snapshot {
        if (self.spare.pop()) |held| {
            var state = held;
            state.pos = pos;
            return state;
        }
        return keep_file.fresh(engine, pos);
    }

    /// Rank 1: a state let go, its buffers ready for the next while there is room.
    fn release(self: *Link, kept: *Kept) void {
        kept.freePooled();
        if (self.spare.items.len < spares) {
            self.spare.append(self.gpa, kept.state) catch kept.state.deinit();
        } else kept.state.deinit();
    }

    /// Before a pass of a split head on rank 0: rank 1 is told its spans. Nothing where the head is whole.
    pub fn tellHead(self: *Link, engine: *Engine, spans: []const Span) Error!void {
        if (self.hm.lead == null or engine.m.mtp.?.split == null) return;
        if (spans.len > 64) return error.OutOfStep;
        var words: [1 + span_words * 64]u32 = undefined;
        words[0] = @intCast(spans.len);
        var rows: usize = 0;
        for (spans, 0..) |span, at| {
            const own = span.source.view.ptr != engine.buffers.h.view.ptr;
            const word = words[1 + span_words * at ..][0..span_words];
            word.* = .{ @intCast(span.slot), @intFromBool(own), @intCast(span.source_row), @intCast(span.n), @intCast(span.pos), @intFromBool(span.score) };
            if (rows + span.n > self.tokens.len) return error.OutOfStep;
            @memcpy(self.tokens[rows..][0..span.n], span.tokens);
            rows += span.n;
        }
        const arguments = words[0 .. 1 + span_words * spans.len];
        if (1 + arguments.len + rows > lead_file.Lead.widest) return error.OutOfStep;
        try held_file.told(self.hm.lead, .hall_head, arguments, self.tokens[0..rows]);
    }

    /// A split head's scores of rows `x`, one a seat of `seats`: each rank's part of the draft ids, gathered,
    /// laid in the seats' rows of `engine.scored` at that part's columns (`model.Mtp.Split.partOf`), so a row
    /// is the draft ids' in order as one rank's draft head scores them. A share alone (the tests) leaves the
    /// other rank's columns at zero.
    pub fn scoreHead(self: *Link, engine: *Engine, x: cuda.View, seats: []const usize) Error!void {
        const head = engine.m.mtp.?;
        const split = head.split.?;
        const own = head.draft.?;
        const n = seats.len;
        const widest = split.widest();
        const part = self.score_part.?;
        const all = self.score_all.?;
        try own.linear.forward(x, cuda.View.init(part.view.ptr.?, .bf16, &.{ n, own.ids.len }));
        const each = n * widest;
        const base: [*]u8 = @ptrCast(all.view.ptr.?);
        switch (self.hm.gather) {
            .loopback => {
                try cuda.zeroLaunched(cuda.View.init(base + (1 - self.hm.first) * each * 2, .bf16, &.{each}));
                try all.copyAt(self.hm.first * each * 2, part, each * 2);
            },
            .nccl => |comm| try comm.allGather(cuda.View.init(part.view.ptr.?, .bf16, &.{each}), cuda.View.init(all.view.ptr.?, .bf16, &.{world * each})),
        }
        const scored = engine.scored.?;
        const width: usize = @intCast(scored.view.size[1]);
        var column: usize = 0;
        for (0..world) |rank| {
            const count = @TypeOf(split).partOf(split.ids, rank).len;
            for (seats, 0..) |slot, i| {
                const from: cuda.Buffer = .{ .view = cuda.View.init(base + (rank * each + i * count) * 2, .bf16, &.{count}) };
                try scored.copyAt((slot * width + column) * 2, from, count * 2);
            }
            column += count;
        }
    }
};

/// `draft.Chooser`'s model on two ranks: a row of logits is gathered, rank 1 told (`Op.row`). A share alone
/// in its process (the tests) reads its own half and leaves the other at zero, as `rounds_ranks_test` does.
pub const namespace = struct {
    pub const Buffers = H.Buffers;
    pub const Error = H.Error;

    pub fn logitsRow(b: H.Buffers, row: usize, out: []u16) H.Error!void {
        if (b.row != null) return H.logitsRow(b, row, out);
        try held_file.told(b.lead, .row, &.{@intCast(row)}, &.{});
        const half: usize = @intCast(b.ranks[0].logits.view.size[1]);
        std.debug.assert(half * world == out.len);
        @memset(out, 0);
        try cuda.synchronize();
        // the share is alone: which rank's half it is does not move its columns
        try b.ranks[0].logits.downloadAt(row * half * 2, std.mem.sliceAsBytes(out[0..half]));
    }
};

pub const Chooser = shared_draft.Chooser(namespace);

/// Rank 1: runs what rank 0 tells on `engine`, whose link `link` is, until `end`. Returns how many operations
/// ran, the end among them.
pub fn follow(channel: *channel_file.Channel, engine: *Engine, link: *Link) !usize {
    std.debug.assert(link.hm.lead == null);
    const gpa = engine.gpa;
    const window = try gpa.alloc(i32, lead_file.Lead.widest);
    defer gpa.free(window);
    const row = try gpa.alloc(u16, engine.m.shape.vocab);
    defer gpa.free(row);
    var ran: usize = 0;
    // rank 0's frames come in `said`'s words from a round's settle or commit to its end (`Link.heat`)
    var hot = false;
    var heard: [Link.said_width]u32 = undefined;
    while (true) {
        const said: ?lead_file.Told = if (!hot) null else listen: {
            try link.said.?.hear(&heard);
            switch (lead_file.Heard.read(&heard) orelse return error.OutOfStep) {
                .told => |told| break :listen told,
                .channel => break :listen null,
                .cooled => {
                    hot = false;
                    break :listen null;
                },
            }
        };
        const told = said orelse switch (try channel.next()) {
            .request => |bytes| lead_file.Told.read(bytes) orelse return error.OutOfStep,
            .barrier => return error.OutOfStep,
        };
        const op = std.enums.fromInt(Op, told.kind) orelse return error.OutOfStep;
        ran += 1;
        const arguments: usize = switch (op) {
            .end => 0,
            .flash_round => flash_follow.arguments(flash_pass.Target, op, told.rest) orelse return error.OutOfStep,
            .row, .hall_reset, .hall_settle => 1,
            .hall_logits => 2,
            .table => 3,
            .hall_round => blk: {
                if (told.rest.len < round_words or told.rest[2] == 0 or told.rest[2] > engine.slots.len) return error.OutOfStep;
                break :blk round_words + 2 * @as(usize, told.rest[2]);
            },
            .hall_commit => blk: {
                if (told.rest.len < 1 or told.rest[0] != link.open_parts) return error.OutOfStep;
                break :blk 1 + @as(usize, told.rest[0]);
            },
            .hall_head => blk: {
                if (told.rest.len < 1 or told.rest[0] == 0 or told.rest[0] > engine.slots.len) return error.OutOfStep;
                break :blk 1 + span_words * @as(usize, told.rest[0]);
            },
            .hall_keep => 3,
            .hall_within => 2,
            .hall_hold => 1,
            .hall_fork => 6,
            .hall_capture => 2,
            .hall_wanted => blk: {
                if (told.rest.len < 2 or told.rest[1] == 0 or told.rest[1] > most_wanted) return error.OutOfStep;
                break :blk 2 + 2 * @as(usize, told.rest[1]);
            },
            .hall_drop => blk: {
                if (told.rest.len < 1 or told.rest[0] == 0 or told.rest[0] > 1024) return error.OutOfStep;
                break :blk 1 + @as(usize, told.rest[0]);
            },
            else => return error.OutOfStep,
        };
        const tokens = told.tokens(arguments, window) orelse return error.OutOfStep;
        const a = told.rest[0..arguments];
        if (op != .table and op != .hall_round and op != .hall_head and op != .hall_within and op != .hall_settle and tokens.len != 0) return error.OutOfStep;
        switch (op) {
            .end => return ran,
            .table => {
                if (a[2] < a[1] or a[2] - a[1] != tokens.len) return error.OutOfStep;
                try link.place(engine, a[0], a[1], tokens);
            },
            .hall_reset => {
                if (a[0] >= engine.slots.len or link.open_parts != 0) return error.OutOfStep;
                try engine.resetSlot(a[0]);
            },
            .hall_round => {
                if (link.open_parts != 0) return error.OutOfStep;
                link.committed_parts = 0;
                const score = std.enums.fromInt(forward.Score, a[0]) orelse return error.OutOfStep;
                const parts = a[2];
                var rows: usize = 0;
                for (0..parts) |at| {
                    const slot: usize = a[round_words + 2 * at];
                    const n: usize = a[round_words + 2 * at + 1];
                    if (slot >= engine.slots.len or n == 0 or rows + n > tokens.len) return error.OutOfStep;
                    for (link.open[0..at]) |before| if (before.slot == slot) return error.OutOfStep;
                    // the stream's rows lie where its table says: every extent to the window's end was told
                    const end = engine.slots[slot].pos + n;
                    if (end > engine.context or (engine.trunk.len > 0 and extents.extentsFor(end) > link.placed[slot])) return error.OutOfStep;
                    @memcpy(link.tokens[rows..][0..n], tokens[rows..][0..n]);
                    link.open[at] = .{ .slot = slot, .tokens = link.tokens[rows..][0..n], .keep = n };
                    rows += n;
                }
                const next = tokens[@min(rows, tokens.len)..];
                if (rows + a[4] != tokens.len or rows > engine.m.max_rows or a[1] > rows or a[1] > engine.buffers.scoredRows()) return error.OutOfStep;
                const segments = link.open[0..parts];
                // the states told wanted lie inside this round's chunk of their slot
                if (engine.staged.wanted.len > 0) {
                    const part = for (segments) |segment| {
                        if (segment.slot == engine.staged.wanted_slot) break segment;
                    } else return error.OutOfStep;
                    const end = engine.slots[part.slot].pos + part.tokens.len;
                    for (engine.staged.wanted) |state| if (state.pos >= end) return error.OutOfStep;
                }
                try rounds.stageAhead(gpa, engine.m, engine.slots, &engine.staged, &engine.buffers, segments, if (engine.ahead) |*ahead| ahead else null);
                rounds.askNext(engine.m, engine.slots, &engine.buffers, segments, next);
                link.open_parts = parts;
                if (a[3] == 0) try link.compute(engine, segments, score, a[1]) else {
                    // a graph is a round of replies, each narrower than the wide kernels, its rows all scored
                    const context: usize = a[3] - 1;
                    if (score != .all or a[1] != rows or rows > engine.buffers.kept or rows >= gdn.wide_rows or context != forward.Graphs.bucket(engine.buffers, @max(context, 1))) return error.OutOfStep;
                    for (segments) |segment| if (segment.tokens.len >= gdn.wide_rows) return error.OutOfStep;
                    try link.launchGraph(engine, rows, context);
                }
            },
            .hall_logits => {
                if (link.open_parts == 0 or a[0] >= engine.m.max_rows or a[1] >= engine.buffers.scoredRows()) return error.OutOfStep;
                try engine.m.head.forward(engine.buffers.mixed.span(a[0], 1), engine.buffers.logits.span(a[1], 1));
            },
            .row => {
                if (a[0] >= engine.buffers.scoredRows()) return error.OutOfStep;
                try namespace.logitsRow(link.buffers(engine.buffers), a[0], row);
            },
            .hall_settle => {
                const rows: usize = a[0];
                const picks = &engine.settling;
                // the open round's reply rows, each asked to a rank a pick can be: rank 0's `Engine.settleRound`
                if (link.open_parts == 0 or link.row == null or rows == 0 or rows != tokens.len or rows > picks.most() or rows > engine.buffers.scoredRows()) return error.OutOfStep;
                for (picks.asked[0..rows], tokens) |*ask, token| {
                    if (token < 0 or token > pick_file.Picked.most) return error.OutOfStep;
                    ask.* = @intCast(token);
                }
                try settle_file.launchRound(world, link.hm.gather.nccl, engine.buffers.logits, picks, rows);
                hot = link.said != null;
            },
            .hall_commit => {
                const segments = link.open[0..link.open_parts];
                var partial = false;
                for (segments, a[1..][0..segments.len]) |*segment, keep| {
                    if (keep == 0 or keep > segment.tokens.len) return error.OutOfStep;
                    if (keep < segment.tokens.len) {
                        if (segment.tokens.len >= gdn.wide_rows) return error.OutOfStep;
                        partial = true;
                    }
                    segment.keep = keep;
                }
                link.open_parts = 0;
                if (partial) try rounds.replayKept(engine.m, engine.slots, &engine.staged, segments, engine.saves);
                try rounds.commit(engine.m, engine.slots, &engine.buffers, &engine.staged, segments);
                link.committed_parts = segments.len;
                hot = link.said != null;
            },
            .hall_head => try followHead(engine, link, a, tokens),
            .flash_round => {
                if (link.open_parts != 0) return error.OutOfStep;
                const flash = engine.rank_flash orelse return error.OutOfStep;
                // Consume a subset of the just-committed segments at their original tap offsets,
                // never the rejected suffix or a previous round's rows.
                for (0..a[0]) |at| {
                    const item = a[1 + 5 * at ..][0..5];
                    var from: usize = 0;
                    var found = false;
                    for (link.open[0..link.committed_parts]) |segment| {
                        if (segment.slot == item[0]) {
                            if (item[2] != from or item[3] != segment.keep or @as(usize, item[1]) + item[3] != engine.slots[segment.slot].pos) return error.OutOfStep;
                            found = true;
                            break;
                        }
                        from += segment.tokens.len;
                    }
                    if (!found) return error.OutOfStep;
                }
                try flash_follow.follow(flash_pass.Target, op, a, tokens, &flash.flash.d, &flash.seats, engine.m.max_rows, engine.buffers.taps);
                link.committed_parts = 0;
            },
            .hall_keep, .hall_wanted, .hall_within, .hall_hold, .hall_drop, .hall_fork => try followKeep(op, engine, link, a, tokens),
            .hall_capture => {
                if (link.open_parts != 0) return error.OutOfStep;
                const rows: usize = a[0];
                // what `Engine.captureFirst` captures: a reply window's rows at the first bucket
                if (rows == 0 or rows >= gdn.wide_rows or rows > engine.buffers.kept) return error.OutOfStep;
                if (a[1] != forward.Graphs.bucket(engine.buffers, 1)) return error.OutOfStep;
                try link.captureGraph(engine, rows, a[1]);
            },
            else => unreachable,
        }
    }
}

/// Rank 1: what rank 0 told of its states, as `hall_keep.zig` does it there, checked against what this rank
/// holds.
fn followKeep(op: Op, engine: *Engine, link: *Link, a: []align(1) const u32, tokens: []const i32) !void {
    const gpa = link.gpa;
    switch (op) {
        .hall_keep => {
            const id = a[0];
            const slot: usize = a[1];
            if (id == 0 or slot >= engine.slots.len or link.held.contains(id) or link.open_parts != 0) return error.OutOfStep;
            const pos = engine.slots[slot].pos;
            // a head whole on rank 0 leaves this rank none, its rows none
            const head_len: usize = if (engine.m.mtp == null) 0 else a[2];
            if (a[2] > pos or engine.slots[slot].head_len < head_len) return error.OutOfStep;
            var kept: Link.Kept = .{ .id = id, .pos = pos, .head_len = head_len, .state = try link.blank(engine, pos) };
            errdefer link.release(&kept);
            try keep_file.takeState(engine, slot, &kept.state);
            kept.pooled = try keep_file.pooledBelow(engine, slot, pos);
            try link.held.put(gpa, id, kept);
        },
        .hall_wanted => {
            const slot: usize = a[0];
            if (slot >= engine.slots.len or link.pending.items.len != 0 or link.open_parts != 0 or engine.wanted.items.len != 0) return error.OutOfStep;
            const n: usize = a[1];
            for (0..n) |at| {
                const id = a[2 + 2 * at];
                const pos: usize = a[3 + 2 * at];
                if (id == 0 or link.held.contains(id) or pos <= engine.slots[slot].pos or pos > engine.context) return error.OutOfStep;
                for (link.pending.items) |before| if (before.id == id or before.pos >= pos) return error.OutOfStep;
                var state = try link.blank(engine, pos);
                link.pending.append(gpa, .{ .id = id, .pos = pos, .head_len = if (engine.m.mtp == null) 0 else pos, .state = state }) catch |err| {
                    state.deinit();
                    return err;
                };
            }
            try engine.wanted.ensureTotalCapacity(gpa, n);
            for (link.pending.items) |kept| engine.wanted.appendAssumeCapacity(kept.state);
            engine.staged.wanted = engine.wanted.items;
            engine.staged.wanted_slot = slot;
        },
        .hall_within => {
            const slot: usize = a[0];
            if (link.pending.items.len == 0 or link.open_parts == 0 or engine.staged.wanted_slot != slot) return error.OutOfStep;
            // the chunk is the open round's part of the slot, at the row told, and every state lies inside it
            const part = for (link.open[0..link.open_parts]) |segment| {
                if (segment.slot == slot) break segment;
            } else return error.OutOfStep;
            if (!std.mem.eql(i32, part.tokens, tokens) or a[1] + tokens.len > engine.m.max_rows) return error.OutOfStep;
            for (link.pending.items) |kept| if (kept.pos >= engine.slots[slot].pos + tokens.len) return error.OutOfStep;
            engine.staged.wanted = &.{};
            engine.wanted.clearRetainingCapacity();
            for (link.pending.items) |*kept| try keep_file.withinOne(engine, slot, &kept.state, tokens, a[1]);
        },
        .hall_hold => {
            const slot: usize = a[0];
            if (slot >= engine.slots.len or link.pending.items.len == 0 or link.open_parts != 0) return error.OutOfStep;
            while (link.pending.items.len > 0) {
                var kept = link.pending.orderedRemove(0);
                kept.pooled = keep_file.pooledBelow(engine, slot, kept.pos) catch |err| {
                    link.release(&kept);
                    return err;
                };
                link.held.put(gpa, kept.id, kept) catch |err| {
                    link.release(&kept);
                    return err;
                };
            }
        },
        .hall_drop => for (a[1..][0..a[0]]) |id| {
            var gone = link.held.fetchRemove(id) orelse return error.OutOfStep;
            link.release(&gone.value);
        },
        .hall_fork => {
            const slot: usize = a[0];
            const kept = link.held.get(a[1]) orelse return error.OutOfStep;
            if (slot >= engine.slots.len or link.open_parts != 0 or engine.slots[slot].pos != 0 or a[2] > 1) return error.OutOfStep;
            if (a[2] == 1) {
                const from: i32 = @bitCast(a[3]);
                const to: i32 = @bitCast(a[4]);
                const count = engine.pool.refs.len;
                if (from < 0 or to < 0 or from == to or from >= count or to >= count or a[5] > extents.positions) return error.OutOfStep;
                const copy: extents.Copy = .{ .from = from, .to = to, .rows = a[5] };
                for (engine.trunk) |pool| try pool.copy(copy);
                if (engine.head_pool) |pool| try pool.copy(copy);
            }
            try keep_file.restore(engine, slot, kept.state, kept.pooled, kept.pos, kept.head_len);
        },
        else => unreachable,
    }
}

/// Rank 1: a pass of the split head rank 0 told, its spans checked against what this rank holds.
fn followHead(engine: *Engine, link: *Link, a: []align(1) const u32, tokens: []const i32) !void {
    const head = engine.m.mtp orelse return error.OutOfStep;
    if (head.split == null or engine.hb == null or link.open_parts != 0) return error.OutOfStep;
    var spans: [64]Span = undefined;
    const count: usize = a[0];
    var rows: usize = 0;
    for (spans[0..count], 0..) |*span, at| {
        const word = a[1 + span_words * at ..][0..span_words];
        const slot: usize = word[0];
        const n: usize = word[3];
        const pos: usize = word[4];
        if (slot >= engine.slots.len or word[1] > 1 or word[5] > 1 or n == 0 or rows + n > tokens.len) return error.OutOfStep;
        for (spans[0..at]) |before| if (before.slot == slot) return error.OutOfStep;
        const source = if (word[1] == 0) engine.buffers.h else engine.held.?;
        if (word[2] + n > @as(usize, @intCast(source.view.size[0])) or pos + n > engine.context) return error.OutOfStep;
        // the head's rows lie where the stream's table says: every extent to the pass's end was told
        if (extents.extentsFor(pos + n) > link.placed[slot]) return error.OutOfStep;
        // rank 0's head stands where it says, after the drafts it dropped
        engine.slots[slot].head_len = pos;
        span.* = .{ .slot = slot, .source = source, .source_row = word[2], .n = n, .tokens = tokens[rows..][0..n], .pos = pos, .score = word[5] == 1 };
        rows += n;
    }
    if (rows != tokens.len or rows > engine.ids.len) return error.OutOfStep;
    try head_file.headPass(engine, spans[0..count], false);
}

/// One rank's chooser of a seat as the ranks' (`namespace`): the same sampler, settings, constraint and rows settled.
pub fn chooserOf(one: @import("../draft.zig").Chooser) Chooser {
    return .{ .sampler = one.sampler, .settings = one.settings, .eos = one.eos, .logits = one.logits, .masks = one.masks, .follow = one.follow, .settled = one.settled };
}
