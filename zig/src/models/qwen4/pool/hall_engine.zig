//! `Engine`, the shared pool of `hall.zig`: the seats, a round of their windows with a chunk of the oldest
//! prompt, the head's rows and their drafts, the forks and the states kept. `hall.zig`'s header says what a
//! round is; the seats' and the round's types are `hall_seats.zig`'s.

const std = @import("std");
const Allocator = std.mem.Allocator;
const copied = @import("../../../decode/copied.zig");
const recall = @import("../../../decode/recall.zig");
const cuda = @import("../../../cuda/cuda.zig");
const cache_file = @import("../../../decode/graphs.zig");
const extents = @import("../../../decode/extents.zig");
const masks_file = @import("../../../decode/masks.zig");
const prefix = @import("../../../decode/prefix.zig");
const sampling = @import("../../../decode/sampling.zig");
const sched = @import("../../../server/scheduler.zig");
const settle_file = @import("../../../decode/settle.zig");
const attention = @import("../blocks/attention.zig");
const forward = @import("../forward.zig");
const draft = @import("../draft.zig");
const flash_pass = @import("../dflash2/pass.zig");
const rig = @import("../dflash2/rig.zig");
const logprob = @import("../../../decode/logprob.zig");
const gdn = @import("../blocks/gdn.zig");
const hall_file = @import("hall.zig");
const open_file = @import("hall_open.zig");
const keep_file = @import("hall_keep.zig");
const warm_file = @import("hall_warm.zig");
const head_file = @import("hall_head.zig");
const chase_file = @import("hall_chase.zig");
const depth_file = @import("hall_depth.zig");
const hall_ranks = @import("hall_ranks.zig");
const hall_seats = @import("hall_seats.zig");
const marks = @import("../../../decode/marks.zig");
const model_file = @import("../model.zig");
const mtp = @import("../mtp.zig");
const rounds = @import("rounds.zig");
const stage_file = @import("rounds_stage.zig");
const Ask = hall_file.Ask;
const EngineOptions = hall_file.EngineOptions;
const HeadRound = hall_seats.HeadRound;
const Live = hall_seats.Live;
const Out = hall_file.Out;
const Report = hall_file.Report;
const Seat = hall_seats.Seat;
const Span = hall_seats.Span;
const address = hall_file.address;
const between = hall_file.between;
const window_max = masks_file.max_rows;

/// The pool's extents before a sequence is in them: each attention layer's arrays, the head's, and who names
/// which extent.
pub const Rooms = struct {
    trunk: []const attention.Pool,
    head_pool: ?attention.Pool,
    pool: *extents.Pool,
};

/// What `Engine.openAfter` calls with them.
pub const Between = struct {
    context: *anyopaque,
    call: *const fn (*anyopaque, Rooms) void,
};

pub const Engine = struct {
    gpa: Allocator,
    /// A clock for the requests' lines; null times nothing.
    io: ?std.Io = null,
    m: model_file.Model,
    eos: []const u32,
    context: usize,
    prompt_rows: usize,
    chunk: usize,
    depth: usize,
    /// A seat is offered copies of its context (`session.Options.copies`).
    copying: bool,
    /// `serve`'s tower and the slots' ropes, where it serves images (`EngineOptions.sight`).
    sight: ?*hall_file.Sight = null,
    /// `serve --recall`'s record (`session.Options.recall`), set when the pool opens: a seat's copies ask it.
    recall: ?*recall.Store = null,
    pool: extents.Pool,
    store: prefix.Store,
    trunk: []attention.Pool,
    head_pool: ?attention.Pool,
    slots: []rounds.Slot,
    seats: []?Seat,
    /// A seat's row of the main model's logits on the host, and of the head's: -inf but where the head scored.
    /// The first is its chooser's: the last row read whole, never a row settled on the device (`settleRound`).
    logits: [][]u16,
    head_rows: [][]u16,
    staged: rounds.Round,
    shared: rounds.Shared,
    buffers: forward.Buffers,
    saves: []gdn.Scratch,
    graphs: rounds.Graphs,
    /// The head's passes captured (`hall_head.zig`'s `headPass`), keyed as `graphs` is and launched on its
    /// stream; one rank's alone.
    head_graphs: ?cache_file.Cache(rounds.Graphs.Key) = null,
    /// The rounds launch their kernels and replay no graph (`EngineOptions.lazy_graphs`).
    eager: bool,
    hb: ?mtp.Buffers = null,
    head: ?HeadRound = null,
    /// DFlash2 in the head's place (`seatDrafter`): the session's drafter, taken when the pool opened, and a
    /// seat of its context for each slot.
    flash: ?*rig.Rig(2) = null,
    flash_seats: ?flash_pass.SeatsOf(2) = null,
    /// One split drafter share and a context per slot on this process of a TP2 pool.
    rank_flash: ?*rig.Pool = null,
    gathered: ?cuda.Buffer = null,
    held: ?cuda.Buffer = null,
    /// bf16 [slots, the head's logits]: the row the head last scored for each seat, a draft head's columns.
    scored: ?cuda.Buffer = null,
    /// bf16 [slots, hidden] and [slots, the head's logits]: a pass's rows to score where they do not lie
    /// together in the head's window, and their logits where their seats do not (`scoreRows`).
    score_in: ?cuda.Buffer = null,
    score_out: ?cuda.Buffer = null,
    /// A seat's `mtp.Picked` of its scored row, and the rank it is asked to.
    picked: []cuda.Buffer = &.{},
    asked: []Asked = &.{},
    columns: []u16 = &.{},
    whole: ?mtp.Normalizer = null,
    /// How a seat's draft is chosen and judged: `mtp.Head`'s rule, so a stream is drafted alike alone and in
    /// a round. A test turns `on_device` off to hold the two ways of choosing against each other.
    rule: mtp.Rule = .{},
    /// The round the seats' next drafts are priced in (`mtp.Paying.Round`); null prices each seat alone.
    priced: ?mtp.Paying.Round.Price = null,
    /// Drafts chosen each way, and the chained rows the head ran for them in so many passes.
    chosen: struct { device: usize = 0, host: usize = 0, rows: usize = 0, passes: usize = 0 } = .{},
    ids: []i32,
    sid_host: []i32,
    pos_host: []i32,
    seg_host: []i32,
    /// The states kept (`hall_keep.zig`), and those the round's prompt chunk takes within it.
    keeps: keep_file.Keeps,
    wanted: std.ArrayList(forward.Snapshot) = .empty,
    /// Where a prompt keeps its states: the session's `near` and `tail`.
    marking: marks.Options,
    seq: u64 = 0,
    sampler: sampling.Sampler,
    /// A reply that ended leaves its state for the next turn to fork from. Off for the test that holds the
    /// two against each other.
    keep_ends: bool = true,
    /// How many reply-only rounds have been logged for each live-stream count. Two: the first window, then
    /// one after the head has drafted.
    noted: [8]u8 = @splat(0),
    /// One rank of two (`hall_ranks.zig`): a round's launches go through it, rank 1 told first. Null on one
    /// rank, whose launches are as they were.
    ranks: ?*hall_ranks.Link = null,
    /// The device's words of a seat's window, for the seats that ask for log probabilities: made by the first.
    words: ?logprob.Scratch = null,
    /// Where a round's reply rows are settled on the device, every seat's at once (`decode/settle.zig`'s
    /// `launchRound`, spec 144 WS-03): one gather and one download a round in place of a row read back each.
    /// A test turns `settle_on_device` off to hold the two against each other; `settled` counts the rows.
    settling: settle_file.Picks,
    settle_on_device: bool = true,
    settled: usize = 0,
    /// Where a seat's chain is cut to `hall_depth.most_drafts` in a wide round (spec 144 WS-04): two ranks,
    /// unless a test says otherwise; `capped` counts the chains it cut.
    depth_cap: depth_file.Cap = .two_ranks,
    capped: usize = 0,
    /// The slots' PLE rows for their next round, gathered while the head runs (`rounds_stage.Ahead`, spec 144
    /// WS-10): made by the head's first pass, where the model has a PLE; null where it could not be.
    ahead: ?stage_file.Ahead = null,

    /// What a seat's `picked` is asked to, and for which sampling (`mtp_rule.Rule.rank`).
    pub const Asked = struct { rank: u32 = 0, settings: ?sampling.Sampling = null };

    pub fn open(gpa: Allocator, loaded: model_file.Model, eos: []const u32, options: EngineOptions) !Engine {
        return openAfter(gpa, loaded, eos, options, null);
    }

    /// `open`, with `between` called once the extents are made and before anything else is (`hall_open.zig`).
    pub const openAfter = open_file.openAfter;

    /// The prefix store names this engine's pool. The engine is returned by value, so this runs once it sits
    /// where it will stay: a move after it leaves the store pointing at the pool it left.
    pub fn bindPool(self: *Engine) void {
        self.store.pool = &self.pool;
    }

    pub fn readyStates(self: *Engine) void {
        keep_file.ready(self);
    }

    pub fn deinit(self: *Engine) void {
        if (self.ahead) |*ahead| ahead.deinit(self.gpa);
        self.wanted.deinit(self.gpa);
        self.keeps.deinit(self.gpa);
        for (self.seats, 0..) |_, index| self.clear(index);
        if (self.words) |*held| held.deinit();
        self.settling.deinit();
        if (self.whole) |*whole| whole.deinit(self.gpa);
        self.gpa.free(self.asked);
        for (self.picked) |*buffer| buffer.deinit();
        self.gpa.free(self.picked);
        if (self.scored) |*buffer| buffer.deinit();
        if (self.score_in) |*buffer| buffer.deinit();
        if (self.score_out) |*buffer| buffer.deinit();
        self.gpa.free(self.columns);
        if (self.held) |*buffer| buffer.deinit();
        if (self.gathered) |*buffer| buffer.deinit();
        if (self.head) |*head| head.deinit();
        if (self.flash_seats) |*seats| seats.deinit();
        if (self.rank_flash) |flash| flash.destroy();
        if (self.flash) |flash| {
            flash.deinit();
            self.gpa.destroy(flash);
        }
        if (self.hb) |*hb| hb.deinit();
        self.gpa.free(self.seg_host);
        self.gpa.free(self.pos_host);
        self.gpa.free(self.sid_host);
        self.gpa.free(self.ids);
        for (self.logits, self.head_rows) |row, other| {
            self.gpa.free(row);
            self.gpa.free(other);
        }
        self.gpa.free(self.head_rows);
        self.gpa.free(self.logits);
        self.gpa.free(self.seats);
        self.sampler.deinit(self.gpa);
        if (self.head_graphs) |*heads| heads.deinit();
        self.graphs.deinit();
        for (self.saves) |*scratch| scratch.deinit();
        self.gpa.free(self.saves);
        self.buffers.deinit();
        self.shared.deinit();
        self.staged.deinit();
        self.store.deinit();
        self.pool.deinit();
        for (self.slots) |*slot| slot.deinit();
        self.gpa.free(self.slots);
        if (self.head_pool) |*pool| pool.deinit();
        for (self.trunk) |*pool| pool.deinit();
        self.gpa.free(self.trunk);
        self.* = undefined;
    }

    /// DFlash2 drafts the pool's streams by `flash`, the session's drafter, owned here from now on (and freed
    /// here where this fails): its passes run eagerly, the blocks of a round's streams side by side over their
    /// seats (`SeatsOf.round`), from the taps each round writes, which the opening asked of the buffers
    /// (`session.Options.drafter`). Its graphs were captured over the session's taps, and go.
    pub fn seatDrafter(self: *Engine, flash: *rig.Rig(2)) !void {
        errdefer {
            flash.deinit();
            self.gpa.destroy(flash);
        }
        std.debug.assert(self.flash == null and self.hb == null and self.ranks == null);
        if (self.buffers.taps == null or !std.mem.eql(u32, self.buffers.tap_layers, flash.d.c.taps)) return error.Untapped;
        try flash.dropPasses();
        flash.d.target.main = &self.buffers;
        flash.d.target.m = self.m;
        self.flash_seats = try .init(self.gpa, &flash.pair, self.seats.len);
        self.flash = flash;
    }

    /// Attaches the already-reserved split drafter at the engine's permanent address. Ownership transfers
    /// only on success; both ranks do this before the leader can capture or tell a target round.
    pub fn seatRankDrafter(self: *Engine, flash: *rig.Pool) !void {
        std.debug.assert(self.flash == null and self.rank_flash == null and self.hb == null);
        const link = self.ranks orelse return error.Untapped;
        if (self.buffers.taps == null or !std.mem.eql(u32, self.buffers.tap_layers, flash.flash.pair.ranks[0].c.taps)) return error.Untapped;
        try flash.bind(link.hm, link.buffers(self.buffers));
        self.rank_flash = flash;
        self.copying = false;
    }

    pub fn captureFirst(self: *Engine) !void {
        const first = forward.Graphs.bucket(self.buffers, 1);
        // A graph replays through the short projection, which holds `kept` rows. A reply window fits in that;
        // a wider round launches directly.
        const limit = @min(self.buffers.kept, gdn.wide_rows - 1);
        for (1..limit + 1) |rows| {
            try self.graphs.prepare(self.m, self.slots, &self.shared, &self.buffers, &self.staged, rows, first, self.saves);
        }
    }

    pub fn bindHead(self: *Engine) !void {
        const ptrs = try self.gpa.alloc(i64, self.slots.len * 2);
        defer self.gpa.free(ptrs);
        const tables = try self.gpa.alloc(i64, self.slots.len);
        defer self.gpa.free(tables);
        const bases = try self.gpa.alloc(i64, self.slots.len * 3);
        defer self.gpa.free(bases);
        for (self.slots, 0..) |slot, s| {
            const cache = slot.head.?;
            tables[s] = address(slot.table.device.view);
            ptrs[s * 2] = tables[s];
            ptrs[s * 2 + 1] = address(cache.pooled.view);
            bases[s * 3] = address(cache.k.view);
            bases[s * 3 + 1] = address(cache.v.view);
            bases[s * 3 + 2] = address(cache.index_keys.view);
        }
        try self.head.?.ptrs.upload(std.mem.sliceAsBytes(ptrs));
        try self.head.?.tables.upload(std.mem.sliceAsBytes(tables));
        try self.head.?.bases.upload(std.mem.sliceAsBytes(bases));
    }

    pub fn resetSlot(self: *Engine, index: usize) !void {
        if (self.ranks) |link| try link.reset(index);
        const slot = &self.slots[index];
        slot.pos = 0;
        slot.head_len = 0;
        for (slot.recurrent) |pair| {
            try pair[0].zero();
            try pair[1].zero();
        }
        @memset(slot.current, 0);
        slot.behind = 0;
        for (slot.conv) |window| try window.zero();
        for (slot.ple) |*branch| try branch.reset();
        for (slot.caches) |cache| try cache.pooled.zero();
        if (slot.head) |cache| try cache.pooled.zero();
        if (slot.rope) |rope| try rope.clear();
        slot.image = null;
        // A new sequence, including one restored from a target prefix, owns no old drafter rows.
        if (self.flash_seats) |*seats| try seats.clear(index);
        if (self.rank_flash) |flash| try flash.seats.clear(index);
    }

    /// `ask.prompt` is copied; its last `sent` tokens are the reply's own, given before the stream was moved
    /// here or taken off, and the constraint is advanced over them. `ask.max_tokens` is what is left.
    pub fn begin(self: *Engine, index: usize, ask: Ask, sent: usize) !usize {
        const prompt = ask.prompt;
        const full = prompt.len;
        if (full < 1) return error.Never;
        // An image prompt (spec 019 WS-05) goes on from a text state at or below its first placeholder row,
        // as vLLM's prefix cache holds the blocks before an image by their tokens alone; none above it is
        // kept (`hall_keep.allows`), so no state's tokens stand for an image's rows.
        if (ask.image) |picture| {
            const rope = self.slots[index].rope orelse return error.InvalidImage;
            if (picture.rows.len == 0 or picture.first() >= full or picture.positions.len > rope.capacity) return error.InvalidImage;
        }
        const text = if (ask.image) |picture| prompt[0 .. picture.first() + 1] else prompt;
        const kept = if (ask.kept) |at| (if (at < text.len) at else null) else null;
        // what the prompt shares with the sequences held, before its own is one of them
        const shared = self.store.shared(prompt);
        const continued = keep_file.usable(self, text, kept);
        const source: ?extents.Source = if (continued) |found| .{ .chain = found.chain, .at = found.at } else null;
        switch (self.store.admission(full, source)) {
            .never => return error.Never,
            .wait => return error.Wait,
            .admit => {},
        }
        const cached = if (continued) |found| found.at else 0;
        const chain = if (continued) |found| chain: {
            // A fork inside an extent takes one for the rows it copies. Where none is free an idle chain's
            // tail goes, as a round's `reserve` lets one go, and never the chain forked from.
            const forked = while (true) {
                break self.store.fork(found.chain, found.at) catch |err| switch (err) {
                    error.Short => {
                        if (try self.store.evict(found.chain) == null) return error.Wait;
                        keep_file.sweep(self);
                        continue;
                    },
                    else => return err,
                };
            };
            try keep_file.applyFork(self, index, self.keeps.at(found.chain, found.at).?, forked);
            break :chain forked.chain;
        } else chain: {
            try self.resetSlot(index);
            break :chain try self.store.open(index);
        };
        // a fork's state is a text prompt's: the rows past it take the image's positions
        if (self.slots[index].rope) |rope| {
            if (ask.image) |picture| try rope.set(picture.positions, picture.delta) else try rope.clear();
        }
        self.slots[index].image = null;
        const owned = try self.gpa.dupe(i32, prompt);
        errdefer self.gpa.free(owned);
        const follow = try self.gpa.alloc(i32, self.m.max_rows);
        errdefer self.gpa.free(follow);
        const starts = try self.gpa.dupe(usize, ask.starts);
        errdefer self.gpa.free(starts);
        // two ranks give no rows (`hall_ranks.chooserOf`): `serve` advertises log probabilities on one alone
        std.debug.assert(ask.logprobs == null or self.ranks == null);
        comptime std.debug.assert(window_max <= draft.forward_max_rows);
        if (ask.logprobs != null and self.words == null) self.words = try .init(self.gpa, draft.forward_max_rows);
        const lp_rows = try self.gpa.alloc(logprob.Row, if (ask.logprobs != null) draft.forward_max_rows else 0);
        errdefer self.gpa.free(lp_rows);
        var masks: ?masks_file.Masks = null;
        if (ask.constraint) |held| {
            masks = try masks_file.Masks.init(held, ask.mask, ask.gate);
            try masks.?.advance(prompt[full - sent ..]);
        }
        const eos = ask.eos orelse self.eos;
        var copies: ?copied.Copied = null;
        if (self.copying and self.hb != null and masks == null) {
            copies = try .init(self.gpa, null, eos);
            errdefer copies.?.deinit();
            copies.?.store = self.recall;
            try copies.?.begin(prompt);
        }
        self.seq += 1;
        self.seats[index] = .{
            .chain = chain,
            .prompt = owned,
            .filled = cached,
            .seq = self.seq,
            .max_tokens = ask.max_tokens,
            .settings = ask.settings,
            .depth = ask.depth,
            .opening = ask.opening,
            .masks = masks,
            .copies = copies,
            .paying = if (self.m.mtp) |head| if (head.costs) |costs| .begin(costs) else null else null,
            .eos = eos,
            .follow = follow,
            .sent = sent,
            .cached = cached,
            .began = self.now(),
            .starts = starts,
            .kept = kept,
            .warm = ask.warm,
            .logprobs = ask.logprobs,
            .rows = lp_rows,
            .image = ask.image,
        };
        keep_file.plan(self, &self.seats[index].?, shared, ask.gate);
        return cached;
    }

    fn clear(self: *Engine, index: usize) void {
        if (self.seats[index]) |*seat| {
            if (seat.copies) |*copies| copies.deinit();
            self.gpa.free(seat.starts);
            self.gpa.free(seat.rows);
            self.gpa.free(seat.follow);
            self.gpa.free(seat.prompt);
            self.seats[index] = null;
        }
    }

    /// The slot's stream leaves. A reply that ended leaves its state behind as one a later prompt forks from
    /// (`hall_keep.remember`): the next turn of its conversation holds the prompt and the reply, as a session's
    /// live state serves it, and not the prompt's states alone. So does a cancelled reply whose connection
    /// took every token given, as a session's state holds what it gave; one that took fewer is cut there,
    /// below the slot's state, and keeps the states below the cut.
    pub fn retire(self: *Engine, index: usize, why: sched.Leave, kept: usize) ?Report {
        const seat = self.seats[index] orelse return null;
        const len = @min(seat.chain.len(), seat.prompt.len - seat.sent + kept);
        const whole = switch (why) {
            .ended => seat.phase == .done,
            .cancelled => seat.phase == .decode and seat.prompt.len - seat.sent + kept > seat.chain.len(),
            .failed => false,
        };
        // a warm pass's end is its one token, a guess at the next turn's: its prompt's states are what it left
        if (whole and self.keep_ends and !seat.warm) {
            self.trim(index);
            const slot = self.slots[index];
            // the head took in the reply's last window (`absorbAndChase`), so both stand at the chain's end
            if (slot.pos >= 1 and slot.pos == seat.chain.len() and (slot.head == null or slot.head_len == slot.pos)) {
                keep_file.remember(self, index, seat.last, .end) catch |err| std.log.err("keeping slot {d}'s state: {t}", .{ index, err });
            }
        }
        const result = switch (why) {
            .failed => self.store.release(seat.chain),
            .cancelled => self.store.retire(index, len),
            .ended => self.store.retire(index, seat.chain.len()),
        };
        result catch |err| std.log.err("retiring slot {d}: {t}", .{ index, err });
        if (why != .failed) if (seat.copies) |*copies| copies.end();
        var beside: usize = 0;
        for (self.seats, 0..) |other, i| beside += @intFromBool(other != null and i != index);
        var digest = seat.digest;
        const report: Report = .{
            .prompt = seat.prompt.len - seat.sent,
            .held = seat.cached,
            .prompt_seconds = between(seat.began, seat.first_at),
            .first = seat.first_at != null,
            .rounds = seat.rounds,
            .tokens = seat.given,
            .seconds = between(seat.first_at, seat.ended_at orelse self.now()),
            .accepted = seat.accepted,
            .drafted = seat.drafted,
            .drafts = digest.final(),
            .copies = if (seat.copies) |copies| copies.stats else .{},
            .levels = if (seat.paying) |paying| paying.counts else .{},
            .states = self.store.marks.items.len,
            .states_bytes = self.keeps.bytes(),
            .beside = beside,
        };
        self.clear(index);
        keep_file.sweep(self);
        // nobody waits for this reply now: the next prompt's states find their buffers made
        keep_file.ready(self);
        return report;
    }

    pub fn preempt(self: *Engine, index: usize) void {
        if (self.seats[index] == null) return;
        self.store.preempt(index) catch |err| std.log.err("preempting slot {d}: {t}", .{ index, err });
        self.clear(index);
    }

    pub fn round(self: *Engine, out: Out) sched.Outcome {
        return self.roundInner(out) catch |err| switch (err) {
            error.Short => .short,
            else => {
                std.log.err("shared round: {t}", .{err});
                return .failed;
            },
        };
    }

    /// Slot `index`'s image features where the chunk of `rows` from its position takes some (spec 019 WS-05):
    /// through the tower the first time, and again where another prompt's took the buffer since.
    fn see(self: *Engine, index: usize, rows: usize) !void {
        const seat = &self.seats[index].?;
        const picture = seat.image orelse return;
        const slot = &self.slots[index];
        const probe: forward.Image = .{ .features = undefined, .rows = picture.rows };
        if (!probe.touches(slot.pos, rows)) return;
        const sight = self.sight orelse return error.InvalidImage;
        slot.image = try sight.encode(self.gpa, picture, seat.seq);
    }

    /// The slot's reply ends, by its end token or at its length: what is said of it, then the end.
    fn ending(self: *Engine, out: Out, index: usize) void {
        const seat = &self.seats[index].?;
        out.tail(out.ctx, index, .{
            .end = if (seat.stop) seat.last else null,
            .rounds = seat.rounds,
            .drafted = seat.drafted,
            .accepted = seat.accepted,
            .prompt_seconds = if (seat.first_at != null) between(seat.began, seat.first_at) else null,
        });
        out.stop(index, if (seat.stop) .stop else .length);
    }

    fn roundInner(self: *Engine, out: Out) !sched.Outcome {
        warm_file.yieldSeats(self, out);
        var nreply: usize = 0;
        var fill: ?usize = null;
        for (self.seats, 0..) |maybe, i| {
            const seat = maybe orelse continue;
            if (seat.phase == .done) continue;
            if (seat.phase == .decode) {
                if (seat.stop or seat.made >= seat.max_tokens or self.slots[i].pos >= self.context) {
                    self.ending(out, i);
                    self.seats[i].?.phase = .done;
                    self.seats[i].?.ended_at = self.now();
                    continue;
                }
                nreply += 1;
            } else if (fill == null or seat.seq < self.seats[fill.?].?.seq) fill = i;
        }
        if (nreply == 0 and fill == null) return .ran;

        const replies = try self.gpa.alloc(rounds.Segment, nreply);
        defer self.gpa.free(replies);
        var n: usize = 0;
        var reply_rows: usize = 0;
        for (self.seats, 0..) |maybe, i| {
            if (maybe == null) continue;
            const seat = &self.seats[i].?;
            if (seat.phase != .decode or seat.stop or seat.made >= seat.max_tokens) continue;
            const room = seat.max_tokens - seat.made;
            var drafts = seat.drafts[0..@min(seat.draft_n, if (room > 0) room - 1 else 0)];
            if (seat.masks) |*masks| drafts = drafts[0..try masks.window(drafts)];
            seat.window[0] = seat.last;
            @memcpy(seat.window[1..][0..drafts.len], drafts);
            const tokens = seat.window[0 .. 1 + drafts.len];
            replies[n] = .{ .slot = i, .tokens = tokens, .keep = tokens.len };
            reply_rows += tokens.len;
            n += 1;
        }

        var prompt_slot: ?usize = null;
        var prompt_tokens: []const i32 = &.{};
        if (fill) |index| {
            const seat = &self.seats[index].?;
            const left = seat.prompt[seat.filled..];
            const budget = if (nreply > 0) self.chunk else self.prompt_rows;
            const room = self.m.max_rows -| reply_rows;
            const take = keep_file.chunk(self, seat, rounds.chunkRows(left.len, @min(budget, room)));
            if (take > 0) {
                prompt_slot = index;
                prompt_tokens = left[0..take];
            }
        }
        if (n == 0 and prompt_tokens.len == 0) return .ran;
        if (prompt_slot) |index| try self.see(index, prompt_tokens.len);
        const note = prompt_slot == null and n > 0 and n < self.noted.len and self.noted[n] < 2;
        if (note) self.noted[n] += 1;

        const segments = if (prompt_slot) |index|
            try rounds.promptBeside(self.gpa, replies[0..n], index, prompt_tokens, prompt_tokens.len)
        else
            replies[0..n];
        defer if (segments.ptr != replies.ptr) self.gpa.free(segments);

        const chains = try self.gpa.alloc(*extents.Chain, segments.len);
        defer self.gpa.free(chains);
        const uptos = try self.gpa.alloc(usize, segments.len);
        defer self.gpa.free(uptos);
        for (segments, chains, uptos) |segment, *chain, *upto| {
            chain.* = self.seats[segment.slot].?.chain;
            upto.* = self.slots[segment.slot].pos + segment.tokens.len;
            // and the rows the head chains after the round, from wherever the trunk then stands: they are
            // written through the same table before the next round grows it
            if (self.head_pool != null) upto.* = @max(upto.*, @min(upto.* + window_max - 1, self.context));
        }
        while (true) {
            rounds.reserve(&self.pool, chains, uptos) catch |err| switch (err) {
                error.Short => {
                    if (try self.store.evict(null)) |_| {
                        keep_file.sweep(self);
                        continue;
                    }
                    return error.Short;
                },
                else => return err,
            };
            break;
        }
        for (segments) |segment| {
            const table = self.seats[segment.slot].?.chain.table.items;
            try self.slots[segment.slot].table.set(table);
            if (self.ranks) |link| try link.lay(segment.slot, table);
        }
        errdefer keep_file.abandon(self);
        if (prompt_slot) |index| try keep_file.before(self, index, prompt_tokens);

        const reply_only = prompt_slot == null;
        const graphed = reply_only and !self.eager and reply_rows <= self.buffers.kept and reply_rows < gdn.wide_rows and self.graphable(segments);
        const context = if (graphed) self.bucket(uptos) else 0;
        const score: forward.Score = if (graphed) .all else if (n > 0) .none else if (prompt_slot != null and self.seats[prompt_slot.?].?.filled + prompt_tokens.len == self.seats[prompt_slot.?].?.prompt.len) .last else .none;
        const head_rows = if (graphed or n > 0) reply_rows else 0;
        // rank 1 is told the round before rank 0 stages it, so the two stage at once (spec 144 WS-10); what
        // this round tells after its settle goes over the link's words, rank 1 back on the channel at its end
        defer if (self.ranks) |link| link.cool();
        // the prompt's tokens after its chunk, whose table rows are gathered while this round runs (`askNext`)
        var next: []const i32 = &.{};
        if (prompt_slot) |index| {
            const seat = &self.seats[index].?;
            const after = seat.prompt[seat.filled + prompt_tokens.len ..];
            next = after[0..@min(after.len, self.prompt_rows)];
        }
        if (self.ranks) |link| next = try link.tellRound(segments, score, head_rows, if (graphed) context + 1 else 0, next);
        try rounds.stageAhead(self.gpa, self.m, self.slots, &self.staged, &self.buffers, segments, if (self.ahead) |*ahead| ahead else null);
        rounds.askNext(self.m, self.slots, &self.buffers, segments, next);

        if (graphed) {
            if (self.ranks) |link| try link.launchGraph(self, reply_rows, context) else {
                const key: rounds.Graphs.Key = .{ .rows = reply_rows, .context = context };
                if (!self.graphs.cache.contains(key)) {
                    try self.graphs.prepare(self.m, self.slots, &self.shared, &self.buffers, &self.staged, reply_rows, context, self.saves);
                }
                try self.graphs.launch(reply_rows, context);
            }
        } else {
            if (self.ranks) |link| try link.compute(self, segments, score, head_rows) else {
                try rounds.compute(self.m, self.slots, &self.shared, &self.buffers, &self.staged, segments, null, score, null, self.saves);
                if (n > 0) try self.m.head.forward(self.buffers.mixed.rows(reply_rows), self.buffers.logits.rows(reply_rows));
            }
        }

        // while the device runs a round of replies, the buffers a later round's states take (`hall_keep.refill`)
        if (reply_only) keep_file.refill(self);
        const settled = try self.settleRound(segments[0..n], reply_rows);
        var row: usize = 0;
        var partial: bool = false;
        // the seats whose round ends on the state after their reasoning's end
        var reasoned: u64 = 0;
        for (segments[0..n]) |*segment| {
            const seat = &self.seats[segment.slot].?;
            const tokens = segment.tokens;
            const pos = self.slots[segment.slot].pos;
            var choosing = self.chooser(segment.slot);
            if (settled) |all| choosing.settled = all[row..][0..tokens.len];
            const got: draft.Chooser.Accepted = if (self.ranks) |link| ranked: {
                const on = try hall_ranks.chooserOf(choosing).accept(link.buffers(self.buffers), row, pos, tokens, seat.follow);
                break :ranked .{ .keep = on.keep, .ended = on.ended };
            } else try choosing.accept(self.buffers, row, pos, tokens, seat.follow);
            const cut = keep_file.reasoning(seat, tokens, got.keep, seat.ahead);
            const keep = cut.keep;
            if (cut.reasoned) reasoned |= @as(u64, 1) << @intCast(segment.slot);
            self.keeps.took.cut += @intFromBool(keep != got.keep);
            // a round cut short ends on a draft it accepted, never on the reply's end
            if (got.ended and keep == got.keep) seat.stop = true;
            segment.keep = keep;
            seat.rounds += 1;
            seat.drafted += tokens.len - 1;
            seat.accepted += keep - 1;
            if (keep < tokens.len) partial = true;
            row += tokens.len;
        }

        var prompt_done: bool = false;
        if (prompt_slot) |index| {
            const seat = &self.seats[index].?;
            const completes = seat.filled + prompt_tokens.len == seat.prompt.len;
            if (completes) {
                const logit_row: usize = if (graphed or n == 0) 0 else reply_rows;
                const read = if (n > 0) reply_rows else logit_row;
                if (self.ranks) |link| {
                    if (n > 0) try link.scoreRow(self, reply_rows + prompt_tokens.len - 1, reply_rows);
                    try hall_ranks.namespace.logitsRow(link.buffers(self.buffers), read, self.logits[index]);
                } else {
                    if (n > 0) {
                        const last = self.buffers.mixed.span(reply_rows + prompt_tokens.len - 1, 1);
                        try self.m.head.forward(last, self.buffers.logits.span(reply_rows, 1));
                    }
                    try forward.logitsRow(self.buffers, read, self.logits[index]);
                    try self.chooser(index).launchFirst(self.buffers, read);
                }
                const token = try self.chooser(index).first(seat.prompt.len);
                if (try self.chooser(index).firstRow(token)) |one| seat.rows[0] = one;
                seat.last = token;
                seat.made = 1;
                seat.stop = std.mem.indexOfScalar(u32, seat.eos, @intCast(token)) != null;
                seat.first_at = self.now();
                prompt_done = true;
            }
        }
        if (prompt_slot) |index| try keep_file.within(self, index, prompt_tokens, reply_rows);
        if (self.ranks) |link| try link.commit(self, segments, partial) else {
            if (partial) try rounds.replayKept(self.m, self.slots, &self.staged, segments, self.saves);
            try rounds.commit(self.m, self.slots, &self.buffers, &self.staged, segments);
        }

        for (segments[0..n]) |segment| {
            const seat = &self.seats[segment.slot].?;
            const keep = segment.keep;
            if (seat.masks) |*masks| try masks.commit(keep, seat.follow[keep - 1]);
            self.pool.append(seat.chain, segment.tokens[0..keep]);
            const fresh = seat.ahead;
            if (fresh > keep) return error.DraftFailed;
            if (fresh > 0) for (seat.follow[0..fresh], seat.drafts[0..fresh]) |got, sent| if (got != sent) return error.DraftFailed;
            // The token that ends a reply is not one of its tokens: a lone stream's session gives its client
            // none (`session.Sending.take`), and the usage counts what was given.
            const upto = if (seat.stop) keep - 1 else keep;
            if (fresh > upto) return error.DraftFailed;
            // a reply that asks for log probabilities sends none ahead: its rows come with the window, the end's too
            if (seat.logprobs != null) out.sendRows(segment.slot, seat.rows[fresh..keep]);
            out.send(segment.slot, seat.follow[fresh..upto]);
            seat.given += upto - fresh;
            seat.ahead = 0;
            seat.draft_n = 0;
            seat.made += keep;
            seat.last = seat.follow[keep - 1];
            if (seat.last_chain) |last| if (seat.paying) |*paying| paying.done(keep, last);
            seat.last_chain = null;
            if (seat.copies) |*copies| try copies.after(seat.follow[0..keep], self.slots[segment.slot].pos);
            if (seat.stop or seat.made >= seat.max_tokens) {
                self.ending(out, segment.slot);
                seat.phase = .done;
                seat.ended_at = self.now();
            }
        }
        if (prompt_slot) |index| {
            const seat = &self.seats[index].?;
            self.pool.append(seat.chain, prompt_tokens);
            seat.filled += prompt_tokens.len;
            if (prompt_done) {
                const first = [_]i32{seat.last};
                if (seat.copies) |*copies| try copies.after(&first, seat.prompt.len);
                if (seat.logprobs != null) out.sendRows(index, seat.rows[0..1]);
                if (!seat.stop) {
                    out.send(index, &first);
                    seat.given += 1;
                }
                if (seat.stop or seat.made >= seat.max_tokens) {
                    self.ending(out, index);
                    seat.phase = .done;
                    seat.ended_at = self.now();
                } else seat.phase = .decode;
            }
        }

        if (self.hb != null) {
            try self.absorbAndChase(segments, n, prompt_slot, prompt_tokens, prompt_done, reasoned);
        } else {
            try self.keepRound(segments[0..n], prompt_slot, reasoned);
            if (self.flash != null or self.rank_flash != null) try self.flashRound(segments[0..n], prompt_slot, prompt_tokens, prompt_done);
        }
        if (note) {
            var kept: usize = 0;
            for (segments[0..n]) |segment| kept += segment.keep;
            std.log.info("decode round: {d} streams, {d} rows, kept {d}", .{ n, reply_rows, kept });
        }
        return .ran;
    }

    /// The round's reply rows `replies` (rows 0 to `rows` of the logits) settled on the device in one pass
    /// (`decode/settle.zig`, spec 144 WS-03): each row cut at the rank its seat's sampling needs at its own
    /// position, the picks gathered (two ranks, rank 1 told) and read back once, and each row's token chosen
    /// from its start, null where that is not `choose`'s (`readRound`). A seat under a constraint or asking for
    /// log probabilities reads its rows whole, as does a share alone in its process. Null where nothing is
    /// settled.
    fn settleRound(self: *Engine, replies: []const rounds.Segment, rows: usize) !?[]const ?i32 {
        const picks = &self.settling;
        if (!self.settle_on_device or rows == 0 or rows > picks.most()) return null;
        if (self.ranks) |link| if (link.row == null) return null;
        var at: usize = 0;
        for (replies) |segment| {
            const seat = &self.seats[segment.slot].?;
            const pos = self.slots[segment.slot].pos;
            const plain = seat.masks == null and seat.logprobs == null;
            for (picks.rows[at..][0..segment.tokens.len], 0..) |*one, i| {
                one.* = if (plain) .{ .settings = seat.settings, .position = pos + i + 1 } else null;
            }
            at += segment.tokens.len;
        }
        std.debug.assert(at == rows);
        if (!settle_file.asksOf(picks.rows[0..rows], self.m.head.n, self.m.shape.vocab, picks.asked[0..rows])) return null;
        if (self.ranks) |link| {
            try link.settle(picks.asked[0..rows]);
            try settle_file.launchRound(hall_ranks.world, link.hm.gather.nccl, self.buffers.logits, picks, rows);
            self.settled += try settle_file.readRound(hall_ranks.world, picks, rows, self.m.head.n, &self.sampler);
        } else {
            try settle_file.launchRound(1, {}, self.buffers.logits, picks, rows);
            self.settled += try settle_file.readRound(1, picks, rows, self.m.head.n, &self.sampler);
        }
        return picks.out[0..rows];
    }

    /// How seat `index`'s tokens are chosen (`draft.Chooser`): its own settings, constraint and logits row.
    fn chooser(self: *Engine, index: usize) draft.Chooser {
        const seat = &self.seats[index].?;
        return .{ .sampler = &self.sampler, .settings = seat.settings, .eos = seat.eos, .logits = self.logits[index], .masks = if (seat.masks) |*masks| masks else null, .logprobs = if (seat.logprobs) |k| .{ .k = k, .scratch = &self.words.?, .rows = seat.rows } else null };
    }

    fn now(self: *const Engine) ?std.Io.Timestamp {
        return if (self.io) |io| .now(io, .awake) else null;
    }

    fn graphable(self: *Engine, segments: []const rounds.Segment) bool {
        for (segments) |segment| if (segment.tokens.len >= gdn.wide_rows) return false;
        _ = self;
        return true;
    }

    fn bucket(self: *Engine, uptos: []const usize) usize {
        var end: usize = 1;
        for (uptos) |upto| end = @max(end, upto);
        return forward.Graphs.bucket(self.buffers, end);
    }

    // the round's head chase, `hall_chase.zig`
    pub const absorbAndChase = chase_file.absorbAndChase;
    pub const flashRound = chase_file.flashRound;
    pub const flashJob = chase_file.flashJob;
    pub const price = chase_file.price;
    pub const keepRound = chase_file.keepRound;
    pub const trim = chase_file.trim;
    pub const absorb = chase_file.absorb;
    pub const chase = chase_file.chase;
    pub const host = chase_file.host;
    pub const guide = chase_file.guide;
    pub const choose = chase_file.choose;

    pub fn publishAhead(self: *Engine, out: Out) void {
        for (self.seats, 0..) |maybe, i| {
            if (maybe == null) continue;
            const seat = &self.seats[i].?;
            if (seat.ahead == 0) continue;
            out.send(i, seat.drafts[0..seat.ahead]);
            seat.given += seat.ahead;
        }
    }
};
