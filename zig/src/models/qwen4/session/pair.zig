//! One rank of a Flash Next reply (spec 050 WS-05). Rank 0 samples, applies the constraint and decides when
//! the reply ends; before each device operation `ranks_forward.zig` tells rank 1, which runs the same window
//! and records the tokens rank 0 emits (`mirror.zig`).
//!
//! A pair keeps the sequence it ran (spec 094), as one rank's session does: a prompt is run from where it
//! parts from that sequence, the state put back to the highest one kept below there, the attention's rows
//! left where they are and the head rewound with it. Rank 0 decides which states are kept and dropped
//! (`book_file.zig`, one rank's positions) and tells rank 1 each. A reply's tokens are those an empty state
//! gives, whatever was decoded before.
//!
//! The MTP head is split as a layer is (`model.Mtp.split`, spec 090): each rank holds half of it and a head runs
//! its passes on both (`mtp_ranks.zig`). A reply's rounds, where its head is split and no constraint holds it,
//! are each rank's to run (`Rounds`, spec 095): rank 1 drafts and chooses as rank 0 does, from the same gathered
//! rows, and is told the reply's rounds once (`Op.decode`); what it cannot derive (where the reply ends, the
//! states kept, a window copied from the context) rank 0 says inside the stream, a few words a round
//! (`Words`), and rank 1 checks its own drafts and kept rows against them. Elsewhere rank 0 chooses alone. Under
//! `ZSPARK_RANKS_HEAD=whole` it is whole on rank 0, beside its share (`model.Mtp.whole`): it reads the final
//! streams, which rank 0 holds whole, and rank 1 loads and runs nothing for it and is told nothing of it.
//! Either way a draft is a token of the window rank 0 tells, and the window verifies it on both ranks.
//! DFlash2's taps and block join that stream (`Op.flash_*`): rank 1 consumes the leader's taps and proposes
//! nothing. `Of(2)` is both ranks in one process, nothing told: the tests' reference for `Pair` over NCCL.

const std = @import("std");
const Allocator = std.mem.Allocator;
const copied = @import("../../../decode/copied.zig");
const draft_file = @import("../../../decode/draft.zig");
const masks_file = @import("../../../decode/masks.zig");
const sampling = @import("../../../decode/sampling.zig");
const channel_file = @import("../../../ranks/channel.zig");
const lead_file = @import("../../../ranks/lead.zig");
const nccl = @import("../../../ranks/nccl.zig");
const request_file = @import("../../../server/request.zig");
const mtp = @import("../mtp.zig");
const mtp_ranks = @import("../mtp_ranks.zig");
const held_file = @import("../ranks_forward.zig");
const rounds_ranks = @import("../pool/rounds_ranks.zig");
const mirror_file = @import("mirror.zig");
const book_file = @import("../../../decode/book.zig");
const pair_prompt = @import("pair_prompt.zig");
const pair_rounds = @import("pair_rounds.zig");
const share_file = @import("../share.zig");
const forward_file = @import("../forward.zig");
const flash_file = @import("../dflash2/drafter.zig");
const flash_pass = @import("../dflash2/pass.zig");

const world: usize = share_file.world;

pub const Reply = struct {
    finish: request_file.Finish,
    /// Tokens emitted by this reply, the one that ended it among them.
    tokens: usize,
    /// That token, when `finish` is `.stop`.
    end: ?i32 = null,
    stats: draft_file.Stats,
    /// Rows copied from the context (`Drafts.copies`).
    copies: copied.Stats = .{},
    /// The prompt's tokens that were not run: the state held them.
    cached: usize = 0,
    /// The states taken in the prompt's windows and in the reply's.
    taken: usize = 0,
    /// The windows the prompt ran in, from where the state held it.
    windows: usize = 0,
};

/// Who drafts a reply's windows. All default: nobody, and every window is one row.
pub const Drafts = struct {
    /// The head's drafts a window at most; 0 leaves it out: it then lags the state, and the next prompt
    /// starts from an empty one, as on one rank.
    depth: usize = 0,
    /// The reply's first window's drafts at most (`session.openingFor`).
    opening: usize = draft_file.forward_max_rows,
    /// The head's drafts give way to what the context already holds (`decode/copied.zig`).
    copies: bool = false,
    /// A drafter in the head's place, asked for all a window carries: a check's.
    given: ?draft_file.Drafter = null,
};

/// What a prompt says of where its states are kept, as `session.ReplyOptions` has it. All default: by position
/// alone.
pub const Turn = struct {
    /// Where the prompt's messages start, ascending.
    starts: []const usize = &.{},
    /// `marks.Request.kept`: where the prefix other conversations share ends, of `starts`.
    kept: ?usize = null,
    /// The think block's closing tag where the prompt leaves the reply thinking: the state after it is kept.
    close: ?i32 = null,
};

/// What rank 0 tells a caller as the tokens come. `take` sees each token that does not end the reply; true ends
/// it, as a client that left or a stop string. `halt` is asked before a window, and true ends it with nothing
/// more emitted. `held` is told, before the prompt runs, how many of its tokens the state holds.
pub const Watch = struct {
    context: *anyopaque,
    take: *const fn (*anyopaque, i32) bool,
    halt: ?*const fn (*anyopaque) bool = null,
    held: ?*const fn (*anyopaque, usize) void = null,
};

/// Rank 0's share, or rank 1's, over NCCL.
pub const Pair = Of(1);

/// `held` ranks of the two in this process: one (`Pair`), or both in loopback.
pub fn Of(comptime held: usize) type {
    const H = held_file.Held(held);
    return struct {
        m: H.Model,
        state: H.State,
        b: H.Buffers,
        graphs: ?H.Graphs,
        /// Rank 0's, which the model and the buffers keep the address of; its channel is unset in loopback.
        lead: lead_file.Lead,
        logits: []u16,
        /// The tokens chosen after a window's rows, as wide as a prompt's window (`Rounds.kept_most` at most), on
        /// both ranks: a prompt's last window takes the rows that fit it, so a turn's new tokens are one window
        /// as on one Spark (`session.zig`'s `chosen`), not a window and then the last 16 (spec 110 WS-02).
        chosen: []i32,
        sampler: sampling.Sampler,
        /// Rank 0's, where its share came with one (`model.Mtp.whole`, or its half of a split one).
        head: ?mtp.Head,
        /// Where the head is split: this rank's half of it, rank 0's being `head`'s, and what joins the halves.
        split: ?mtp_ranks.Held(held).Split,
        copies: ?copied.Copied,
        /// The sequence the state and the head ran, and the token after it: a head put back at a position
        /// absorbed its last row with the token that followed, which the next prompt must hold still.
        seen: std.ArrayList(i32),
        /// `seen`, the state and the head agree: the last reply ended without an error and the head absorbed
        /// its windows. Otherwise the next prompt starts from an empty state.
        trusted: bool,
        /// Every reply's tokens, in order.
        tokens: std.ArrayList(i32),
        /// The states kept of `seen`'s sequence, at the positions rank 0 told.
        shelf: H.Shelf,
        /// The states asked of a prompt's window wider than the rows each layer keeps, before it runs.
        asked: H.Asked,
        /// Rank 0's account of them.
        book: book_file.Book,
        /// A round of several streams, when this rank follows one. Null for one reply.
        desk: ?*rounds_ranks.Desk = null,
        /// What rank 0 says inside the stream while both ranks run a reply's rounds; null in loopback.
        words: ?Words,
        /// Rank 1's: the reply's end tokens, as rank 0 told them with its rounds.
        ends: [Rounds.most_ends]u32 = undefined,
        /// Rank 1's: the reply windows it ran of its own, untold (`Rounds`).
        own_rounds: usize,
        /// Rank 1's: the prompt a telling is bringing, its frames appended until it is whole (`promptFollowed`
        /// runs it then).
        prompt_told: std.ArrayList(i32) = .empty,
        /// The DFlash2 drafter where rank 0 drafts by it: rank 1 follows its passes (`mirror.zig`).
        flash: ?*flash_file.DrafterOf(held) = null,
        /// Several sequences' DFlash2 seats where rank 0's hall tells a round.
        flash_seats: ?*flash_pass.SeatsOf(held) = null,
        rank: usize,
        capacity: usize,
        /// The ranks' model is this pair's, freed by `destroy`; false where the caller lent it
        /// (`createLoopbackBorrowing`).
        owns: bool,
        gpa: Allocator,

        const Self = @This();

        pub const Error = draft_file.Error(H) || lead_file.Error || mirror_file.Error;

        // A reply's rounds on both ranks (`pair_rounds.zig`) and a prompt told once (`pair_prompt.zig`).
        const Words = pair_rounds.Of(held).Words;
        const Rounds = pair_rounds.Of(held).Rounds;
        const decodeFollowed = pair_rounds.Of(held).decodeFollowed;
        const Prompt = pair_prompt.Of(held).Prompt;
        pub const PromptFrame = pair_prompt.Of(held).PromptFrame;
        pub const promptFrame = pair_prompt.Of(held).promptFrame;
        const toldPrompt = pair_prompt.Of(held).toldPrompt;
        pub const promptFollowed = pair_prompt.Of(held).promptFollowed;
        const promptTold = pair_prompt.Of(held).promptTold;

        /// On the heap: `m.lead` keeps `&lead`. One rank of two over NCCL.
        /// A window's rows scored as one rank's session scores them: a prompt's last, a round's 16 at most.
        pub fn create(gpa: Allocator, rank: usize, built: H.Model, comm: nccl.Comm, channel: *channel_file.Channel, rows: usize, capacity: usize, eager: bool) !*Self {
            return createScoring(gpa, rank, built, comm, channel, rows, @min(rows, draft_file.forward_max_rows), capacity, eager);
        }

        /// `create` with windows of `scored` rows or fewer scored whole: a check that reads every row's logits.
        pub fn createScoring(gpa: Allocator, rank: usize, built: H.Model, comm: nccl.Comm, channel: *channel_file.Channel, rows: usize, scored: usize, capacity: usize, eager: bool) !*Self {
            comptime std.debug.assert(held == 1);
            var owned = built;
            owned.gather = .{ .nccl = comm };
            return make(gpa, rank, owned, true, channel, rows, scored, capacity, eager);
        }

        /// Both ranks in this process: nothing is told and nothing gathered over a link. Every row scored.
        pub fn createLoopback(gpa: Allocator, built: H.Model, rows: usize, capacity: usize, eager: bool) !*Self {
            return createLoopbackScoring(gpa, built, rows, rows, capacity, eager);
        }

        /// `createLoopback` scoring `scored` rows of a window, as `create`: a window wider than the rows each
        /// layer keeps then asks for the states inside it before it runs (`askWithin`).
        pub fn createLoopbackScoring(gpa: Allocator, built: H.Model, rows: usize, scored: usize, capacity: usize, eager: bool) !*Self {
            comptime std.debug.assert(held > 1);
            return make(gpa, 0, built, true, null, rows, scored, capacity, eager);
        }

        /// `createLoopback` on ranks the caller keeps and frees after `destroy`: a check's pairs over one load, run
        /// one at a time. What a load holds on the device beyond its weights is its linears' scratch, written
        /// before it is read in each call; a pair's states, buffers and graphs are its own.
        pub fn createLoopbackBorrowing(gpa: Allocator, built: H.Model, rows: usize, capacity: usize, eager: bool) !*Self {
            comptime std.debug.assert(held > 1);
            return make(gpa, 0, built, false, null, rows, rows, capacity, eager);
        }

        fn make(gpa: Allocator, rank: usize, built: H.Model, owns: bool, channel: ?*channel_file.Channel, rows: usize, scored: usize, capacity: usize, eager: bool) !*Self {
            const self = gpa.create(Self) catch |err| {
                var owned = built;
                if (owns) owned.deinit();
                return err;
            };
            errdefer gpa.destroy(self);
            self.gpa = gpa;
            self.rank = rank;
            self.capacity = capacity;
            self.owns = owns;
            self.lead = .{ .channel = channel orelse undefined };
            self.m = built;
            errdefer if (owns) self.m.deinit();
            self.m.lead = if (rank == 0 and channel != null) &self.lead else null;
            self.m.max_rows = rows;
            // A window's rows scored and kept as one rank's session scores and keeps them: a prompt's window
            // of 2,048 rows scored whole was 0.47 GiB a rank on FP4-Local (spec 111), and each DeltaNet layer's
            // projection and scratch of it kept whole about 1 GiB more; the states inside a wider window are
            // asked of it before it runs (`H.Asked`).
            self.state = try .initKeeping(gpa, self.m, rows, scored, capacity);
            errdefer self.state.deinit();
            self.b = try .initFor(gpa, self.m, rows, scored, capacity);
            errdefer self.b.deinit();
            self.graphs = if (eager) null else try H.Graphs.init(gpa);
            errdefer if (self.graphs) |*g| g.deinit();
            const vocab = self.m.ranks[0].head.n * 2;
            self.logits = try gpa.alloc(u16, vocab);
            errdefer gpa.free(self.logits);
            self.sampler = try .init(gpa, vocab);
            errdefer self.sampler.deinit(gpa);
            self.chosen = try gpa.alloc(i32, @max(draft_file.forward_max_rows, @min(rows, Rounds.kept_most)));
            errdefer gpa.free(self.chosen);
            const halved = if (self.m.ranks[0].mtp) |half| half.split != null else false;
            // its own scratch: the share's is sized for half the heads. A split head's graphs are the split's.
            // rank 1's only where the head is split: it then drafts a reply's rounds as rank 0 does
            self.head = if ((rank == 0 or halved) and self.m.ranks[0].mtp != null) try .init(gpa, self.m.ranks[0], &self.sampler, .{ .seed = 0 }, .{ .rows = rows, .capacity = capacity, .graphs = !eager and !halved }) else null;
            errdefer if (self.head) |*h| h.deinit();
            self.split = null;
            if (halved) {
                self.split = @as(mtp_ranks.Held(held).Split, undefined);
                try self.split.?.init(gpa, self.m, &self.b, if (self.head) |*h| h else null, rows, capacity, !eager);
            }
            errdefer if (self.split) |*s| s.deinit();
            self.copies = null;
            self.seen = .empty;
            self.trusted = false;
            self.tokens = try .initCapacity(gpa, capacity);
            errdefer self.tokens.deinit(gpa);
            self.shelf = .{ .gpa = gpa };
            self.asked = .{};
            self.book = try .init(gpa, .{});
            errdefer self.book.deinit(gpa);
            self.desk = null;
            self.own_rounds = 0;
            self.prompt_told = .empty;
            self.flash = null;
            self.flash_seats = null;
            self.words = if (self.m.gather == .nccl) try .init(self.m.gather.nccl) else null;
            return self;
        }

        pub fn destroy(self: *Self) void {
            const gpa = self.gpa;
            if (self.words) |*w| w.said.deinit();
            self.book.deinit(gpa);
            H.dropAsked(&self.shelf, &self.asked);
            self.asked.deinit(gpa);
            self.shelf.deinit();
            self.prompt_told.deinit(gpa);
            self.tokens.deinit(gpa);
            self.seen.deinit(gpa);
            if (self.copies) |*c| c.deinit();
            if (self.split) |*s| s.deinit();
            if (self.head) |*h| h.deinit();
            self.sampler.deinit(gpa);
            gpa.free(self.chosen);
            gpa.free(self.logits);
            if (self.graphs) |*g| g.deinit();
            self.b.deinit();
            self.state.deinit();
            if (self.owns) self.m.deinit();
            gpa.destroy(self);
        }

        /// Where the states are kept from here on, in place of one rank's defaults. Before any is held.
        pub fn marking(self: *Self, options: book_file.Options) !void {
            std.debug.assert(self.book.count() == 0 and self.shelf.held.items.len == 0);
            const book: book_file.Book = try .init(self.gpa, options);
            self.book.deinit(self.gpa);
            self.book = book;
        }

        /// Rank 1, until rank 0 tells `end`.
        pub fn follow(self: *Self) !usize {
            comptime std.debug.assert(held == 1);
            std.debug.assert(self.rank != 0);
            return mirror_file.follow(self.lead.channel, .{
                .m = self.m,
                .state = &self.state,
                .b = &self.b,
                .graphs = if (self.graphs) |*g| g else null,
                .logits = self.logits,
                .capacity = self.capacity,
                .gpa = self.gpa,
                .tokens = &self.tokens,
                .shelf = &self.shelf,
                .asked = &self.asked,
                .desk = self.desk,
                .head = if (self.split) |*s| s else null,
                .rounds = if ((self.head != null and self.split != null) or self.flash != null) .{ .context = self, .run = decodeFollowed, .prompt = promptTold } else null,
                .flash = self.flash,
                .flash_seats = self.flash_seats,
            });
        }

        fn tell(self: *Self, op: held_file.Op, ids: []const i32) lead_file.Error!void {
            if (self.m.lead) |to| try to.tell(@backingInt(op), &.{}, ids);
        }

        /// The state as it stands is kept, told before it is taken here: both ranks then hold it at the same
        /// position. A check's; a reply keeps its own.
        pub fn keep(self: *Self) !void {
            std.debug.assert(self.rank == 0 and self.book.marked < self.book.marks.len);
            try H.hold(self.m, &self.state, &self.shelf);
            self.book.took(self.state.pos);
        }

        /// Both ranks back at the state kept at `pos`, and the head where it absorbed that far: one that did
        /// not stays where it is. What the state then holds is no reply's to go on from (`reset` first).
        pub fn back(self: *Self, pos: usize) !void {
            std.debug.assert(self.rank == 0);
            try H.back(self.m, &self.state, &self.shelf, pos);
            if (self.head) |*h| if (!h.full and h.state.position.host - h.state.drafted >= pos) try h.rewind(pos, h.settings);
            self.trusted = false;
        }

        /// An empty sequence, told before it runs here. The states kept stay until the next reply, which
        /// drops them.
        pub fn reset(self: *Self) !void {
            std.debug.assert(self.rank == 0);
            try self.empty(if (self.head) |h| h.settings else .{ .seed = 0 });
        }

        fn empty(self: *Self, settings: sampling.Sampling) !void {
            try self.tell(.reset, &.{});
            for (&self.state.ranks) |*one| try one.reset();
            self.state.pos = 0;
            if (self.head) |*h| try h.reset(settings);
            self.seen.clearRetainingCapacity();
            self.trusted = false;
        }

        /// Puts the state where `prompt` can go on from (`start`, as `book_file.Book.start` found it) and
        /// returns how many of `seen`'s tokens the prompt begins with. The states `drops` name go first — a
        /// copy of `start.drops`, which `Book.planned` writes over — and the head's entries below where the
        /// prompt finds it stay. Rank 1 is told each state dropped and where to go back to, or the whole of
        /// it in the prompt's frames (`Prompt`).
        fn @"resume"(self: *Self, start: book_file.Resumed, drops: []const usize, settings: sampling.Sampling) !usize {
            for (drops) |pos| try H.drop(self.m, &self.shelf, pos);
            switch (start.to) {
                .stay => if (self.head) |*h| try h.rewind(self.state.pos, settings),
                .back => |pos| {
                    try H.back(self.m, &self.state, &self.shelf, pos);
                    if (self.head) |*h| try h.rewind(pos, settings);
                },
                .reset => try self.empty(settings),
            }
            return start.shared;
        }

        /// Rank 1 is told that nothing follows.
        pub fn finish(self: *Self) !void {
            std.debug.assert(self.rank == 0);
            try self.tell(.end, &.{});
        }

        /// What a prompt's and a reply's windows keep of the states inside them: `draft.Decoder`'s Within and
        /// Kept, as `session_marks.zig`'s `Marks` and `Ends` are one rank's.
        pub const Keeping = struct {
            pair: *Self,
            /// The think block's closing tag while the reply reasons and the state after it is wanted.
            close: ?i32 = null,
            taken: usize = 0,
            /// The takes of the window just run, in order, where rank 1 runs the rounds and is not told them:
            /// rank 0 says them (`Rounds`). A window takes two at most, at the reasoning's end and the stride.
            /// A prompt told once (`Prompt`) runs its windows with the lead muted too, but its takes are in
            /// the frames, and only two fit here.
            said: bool = false,
            takes: [2]Take = undefined,
            recorded: usize = 0,
            /// The prompt and what plans its windows (`Ahead`).
            ahead: Ahead = .{},
            /// States are kept inside the prompt's windows: `window` takes them.
            inside: bool = true,

            fn take(keeping: *Keeping, tokens: []const i32, rows: usize, drop: ?usize) draft_file.Error(H)!void {
                const self = keeping.pair;
                if (self.lead.muted and keeping.said) {
                    keeping.takes[keeping.recorded] = .{ .drop = drop, .rows = rows };
                    keeping.recorded += 1;
                }
                try Take.run(.{ .drop = drop, .rows = rows }, self, tokens);
                keeping.taken += 1;
            }

            fn before(context: *anyopaque, run: []const i32) draft_file.Error(H)!void {
                const keeping: *Keeping = @ptrCast(@alignCast(context));
                keeping.ahead.before(keeping.pair, run);
                if (keeping.inside) try keeping.pair.askWithin(keeping.pair.book.due, run);
            }

            /// `Decoder.Within`: the window `run` has run from the state's position and is not committed.
            fn window(context: *anyopaque, run: []const i32, last: bool) draft_file.Error(H)!void {
                const keeping: *Keeping = @ptrCast(@alignCast(context));
                const self = keeping.pair;
                if (!keeping.inside) return;
                _ = last;
                const from = self.state.pos;
                for (self.book.due) |at| {
                    if (at <= from or at > from + run.len) continue;
                    try keeping.take(run, at - from, null);
                    self.book.took(at);
                }
                H.dropAsked(&self.shelf, &self.asked);
            }

            /// `Decoder.Kept`: the window `tokens` has run, and its first `rows` rows will be committed.
            pub fn kept(context: *anyopaque, tokens: []const i32, rows: usize) draft_file.Error(H)!void {
                const keeping: *Keeping = @ptrCast(@alignCast(context));
                const self = keeping.pair;
                const from = self.state.pos;
                if (keeping.close) |close| if (std.mem.indexOfScalar(i32, tokens[0..rows], close)) |row| {
                    keeping.close = null;
                    try keeping.take(tokens, row + 1, self.book.close(from + row + 1));
                };
                // after the reasoning's end, which is then the newest: the stride is counted from it
                if (self.book.stride(from, rows)) |one| try keeping.take(tokens, one.at - from, one.drop);
            }
        };

        /// A prompt's next window said to the buffers before this one runs (`H.expect`), its table rows gathered
        /// while the device computes, as one rank's `Marks.before` says them: the windows a pair's prompt ran
        /// had theirs gathered when staged, 50 to 78 ms of host time a 2,048-row window with the device idle
        /// (spec 111's trace). Both ranks: rank 0 in `reply`, rank 1 in `promptFollowed`.
        pub const Ahead = struct {
            prompt: []const i32 = &.{},
            rounds: ?*const draft_file.Decoder(H) = null,
            /// `chosen`'s length, which bounds the prompt's last window (`Decoder.promptRows`).
            room: usize = 0,

            pub fn before(self: Ahead, pair: *Self, run: []const i32) void {
                const rounds = self.rounds orelse return;
                const stop = pair.state.pos + run.len;
                if (stop < self.prompt.len) H.expect(pair.m, &pair.b, self.prompt[stop..][0..rounds.promptRows(stop, self.prompt.len, self.room)]);
            }
        };

        /// The states of `due` inside the window `run`, about to run from the state's position, asked of it
        /// where it is wider than the rows each layer keeps (`H.ask`): rank 1 told first where it is told per
        /// pass. A narrower window's are replayed once it has run (`Take`).
        pub fn askWithin(self: *Self, due: []const usize, run: []const i32) draft_file.Error(H)!void {
            H.dropAsked(&self.shelf, &self.asked);
            if (!forward_file.widerThanKept(self.state.ranks[0], self.b.ranks[0], run.len)) return;
            var words: [H.most_asked]u32 = undefined;
            const into = book_file.inside(due, self.state.pos, run.len, &words) orelse return Take.parted(error.OutOfStep);
            if (into.len == 0) return;
            H.ask(self.m, &self.shelf, &self.asked, &self.state, &self.b, run.len, into) catch |err| return Take.parted(err);
        }

        /// A state kept inside a window, the one before it at `drop` dropped first.
        pub const Take = struct {
            drop: ?usize,
            rows: usize,

            pub fn run(one: Take, self: *Self, tokens: []const i32) draft_file.Error(H)!void {
                if (one.drop) |pos| H.drop(self.m, &self.shelf, pos) catch |err| return parted(err);
                H.holdWithin(self.m, &self.state, &self.b, &self.shelf, &self.asked, tokens, one.rows) catch |err| return parted(err);
            }

            /// A state this rank cannot take or drop where rank 0 planned it: the plan and the shelf parted.
            pub fn parted(err: anyerror) draft_file.Error(H) {
                return switch (err) {
                    error.OutOfStep => blk: {
                        std.log.err("two ranks: a state is not where rank 0's account holds it", .{});
                        break :blk error.DraftFailed;
                    },
                    else => @errorCast(err),
                };
            }
        };

        pub fn outOfStep() draft_file.Error(H) {
            std.log.err("two ranks: rank 1 parted from rank 0's telling", .{});
            return error.DraftFailed;
        }

        /// Rank 0 decodes the reply to `prompt`: its tokens are those an empty state gives, whatever this pair
        /// decoded before, though the prompt is run only from where it parts from the sequence the state holds
        /// (`resume`). `hang_at` ends it once that many tokens have been emitted, as a client that left: rank 1
        /// is not told `end`, so the next reply can follow. `watch` sees the tokens as they are emitted; a stop
        /// there is the same end. `by` says who drafts: the reply's tokens are the serial ones whoever does.
        /// However a reply ends, the head absorbs the last window, to stand with the state for the next.
        pub fn reply(self: *Self, prompt: []const i32, settings: sampling.Sampling, eos: []const u32, by: Drafts, turn: Turn, masks: ?*masks_file.Masks, max_tokens: usize, hang_at: ?usize, watch: ?Watch) !Reply {
            std.debug.assert(self.rank == 0);
            std.debug.assert(prompt.len >= 1 and prompt.len + max_tokens <= self.capacity);
            const before = self.tokens.items.len;
            // a head that is not asked absorbs nothing, and its entries would lag the state
            const absorbing = self.head == null or (by.given == null and by.depth > 0);
            // DFlash2 in the head's place (`serve --drafter`, the pair's `flash`, rank 1 following its passes).
            const flashing = by.given == null and by.depth > 0 and self.head == null and self.flash != null;
            // The prompt's windows are rank 1's to run from one telling where a reply's rounds are (`Rounds`):
            // the head split over the ranks, or DFlash2 in its place (spec 135 WS-02), no constraint, no drafter
            // given. Rank 0 tells the prompt's frames below (`Prompt`) and runs its windows with the lead muted;
            // rank 1 runs them from the frames, its head or its DFlash2 drafter absorbing between them as this
            // rank's does: an update crosses nothing (`dflash2/pass.zig` `addTaps`).
            const once = held == 1 and self.m.lead != null and (self.split != null or flashing) and by.given == null and masks == null;
            defer if (once) {
                self.lead.muted = false;
                self.b.settling = false;
            };
            self.lead.muted = once;
            // Where the prompt goes on from, before the state is put there: the frames say it, and `planned`
            // writes over the account's scratch the drops came from (`Book.gone`), so they are kept.
            const start = self.book.start(self.seen.items, prompt, self.trusted, self.state.pos, turn.kept);
            const drops = try self.gpa.dupe(usize, start.drops);
            defer self.gpa.free(drops);
            const shared = try self.@"resume"(start, drops, settings);
            self.trusted = false;
            const cached = self.state.pos;
            if (watch) |w| if (w.held) |tell_held| tell_held(w.context, cached);
            var drafter: ?draft_file.Drafter = by.given;
            var copying = false;
            // DFlash2's context ends where the state does. The rows below are the last reply's, the same
            // sequence's up to where it was taken up; a session zeroes them past a restored state
            // (`PairOf.startAt`), which rank 1 would not be told, so here a restored state's are what stood
            // there. A prompt told once moves rank 1's end itself (`promptFollowed`).
            if (flashing) {
                const f = self.flash.?;
                f.settings = settings;
                f.ends = eos;
                f.full = false;
                if (f.pair.context_end != cached) try f.pair.setEnd(cached);
                drafter = f.drafter();
            }
            if (by.given == null and by.depth > 0 and self.head != null) {
                const h = &self.head.?;
                h.ends = eos;
                h.fresh();
                drafter = h.drafter();
                if (by.copies) {
                    if (self.copies == null) self.copies = try .init(self.gpa, drafter.?, &.{});
                    const copies = &self.copies.?;
                    copies.inner = drafter.?;
                    copies.stops = eos;
                    try copies.begin(prompt);
                    drafter = copies.drafter();
                    copying = true;
                }
            }
            // what the state holds of the sequence stays
            std.debug.assert(cached < prompt.len and (cached == 0 or self.seen.items.len > cached));
            self.seen.shrinkRetainingCapacity(cached);
            try self.seen.ensureTotalCapacity(self.gpa, self.capacity + 1);
            self.seen.appendSliceAssumeCapacity(prompt[cached..]);
            const chosen = self.chosen;
            var rounds: draft_file.Decoder(H) = .{
                .m = self.m,
                .state = &self.state,
                .b = &self.b,
                .graphs = if (self.graphs) |*g| g else null,
                .sampler = &self.sampler,
                .settings = settings,
                .eos = eos,
                .logits = self.logits,
                .masks = masks,
            };
            // States on the way, at one rank's positions (`marks.Planner`), each taken inside the window that
            // runs over it; only by a reply the head absorbs, whose state the next prompt can go on from.
            const inside = absorbing and self.book.options.marks.budget > 0;
            var keeping: Keeping = .{ .pair = self, .close = if (inside) turn.close else null, .inside = inside, .ahead = .{ .prompt = prompt, .rounds = &rounds, .room = chosen.len } };
            var planned: []const usize = &.{};
            if (inside) {
                var planner = self.book.plan(.{ .pos = cached, .len = prompt.len, .shared = shared, .starts = turn.starts, .kept = turn.kept });
                while (planner.from < prompt.len) self.book.window(&planner, rounds.promptRows(planner.from, prompt.len, chosen.len));
                planned = self.book.planned(&planner);
                for (planned) |pos| try H.drop(self.m, &self.shelf, pos);
            }
            rounds.within = .{ .context = &keeping, .window = Keeping.window, .before = Keeping.before };
            if (once) {
                // the frames go out with the lead clear (`tell` sends nothing muted); the prompt's windows
                // then run with it muted, rank 1 running them from the frames
                self.lead.muted = false;
                try self.toldPrompt(start, drops, planned, self.book.due, prompt, settings, eos, by.depth > 0);
                self.lead.muted = true;
            }
            var windows: usize = 0;
            var at = cached;
            while (at < prompt.len) : (windows += 1) at += rounds.promptRows(at, prompt.len, chosen.len);
            var kept = try rounds.prompt(drafter, prompt, chosen);
            rounds.within = null;
            self.lead.muted = false;
            if (inside) rounds.kept = .{ .context = &keeping, .window = Keeping.kept };
            self.seen.appendAssumeCapacity(chosen[kept - 1]);
            var fresh: usize = kept - 1;
            var drafts: [draft_file.forward_max_rows]i32 = undefined;
            var stopped: ?i32 = null;
            var first = true;
            // Rank 1 runs the rounds too where it can draft as this rank does: the head split, no constraint, no
            // drafter in the head's place (`Rounds`). The last window's absorbing pass is its own as well.
            // DFlash2's rounds stay told a pass (`Op.flash_*`).
            const derives = once and drafter != null and !flashing;
            const reason: request_file.Finish = reply: while (true) {
                if (halted(watch)) break :reply .cut;
                for (chosen[fresh..kept]) |next| {
                    const ending = try self.handed(next, eos, before, max_tokens, hang_at, watch, &stopped);
                    if (ending) |one| break :reply one;
                }
                if (halted(watch)) break :reply .cut;
                const want: usize = if (by.given != null or flashing) rounds.depth() else if (first) @min(by.depth, by.opening) else by.depth;
                first = false;
                if (derives) {
                    var led: Rounds = .{ .pair = self, .chosen = chosen, .copying = copying, .keeping = if (inside) &keeping else null, .lead = .{ .before = before, .max_tokens = max_tokens, .depth = by.depth, .eos = eos, .hang_at = hang_at, .watch = watch, .stopped = &stopped } };
                    break :reply try led.led(&rounds, drafter.?, &kept, @min(want, max_tokens - (self.tokens.items.len - before) - 1));
                }
                const n = if (drafter) |d| try rounds.draft(d, chosen[0..kept], @min(want, max_tokens - (self.tokens.items.len - before) - 1), &drafts) else 0;
                const sure = if (rounds.masks) |mask| mask.certain(drafts[0..n]) else 0;
                var ended: ?request_file.Finish = null;
                for (drafts[0..sure]) |next| {
                    ended = try self.handed(next, eos, before, max_tokens, hang_at, watch, &stopped);
                    if (ended != null) break;
                }
                const anchor = chosen[kept - 1];
                kept = try rounds.round(anchor, drafts[0..n], chosen);
                self.seen.appendSliceAssumeCapacity(chosen[0..kept]);
                if (kept <= sure or !std.mem.eql(i32, chosen[0..sure], drafts[0..sure])) return error.DraftFailed;
                if (ended) |one| break :reply one;
                fresh = sure;
            };
            // The head absorbs the last window, as it would before the next, whichever way the reply ended: it and
            // the state then stand at the same position (`session.zig` does the same on one rank).
            if (drafter) |d| _ = try rounds.draft(d, chosen[0..kept], 0, &drafts);
            std.debug.assert(self.seen.items.len == self.state.pos + 1);
            self.trusted = absorbing and (self.head == null or !self.head.?.full);
            return .{ .finish = reason, .tokens = self.tokens.items.len - before, .end = stopped, .stats = rounds.stats, .copies = if (copying) self.copies.?.stats else .{}, .cached = cached, .taken = keeping.taken, .windows = windows };
        }

        /// Emits `next` and says how the reply ends at it, or null while it goes on. A token that ends the reply is
        /// emitted and not handed to `watch`: the caller sends it as the reply's end.
        pub fn handed(self: *Self, next: i32, eos: []const u32, before: usize, max_tokens: usize, hang_at: ?usize, watch: ?Watch, stopped: *?i32) !?request_file.Finish {
            try self.emit(&.{next});
            if (std.mem.indexOfScalar(u32, eos, @intCast(next)) != null) {
                stopped.* = next;
                return .stop;
            }
            // the last token asked for is the client's too, as one rank sends it before it counts (`session.zig`)
            if (self.tokens.items.len - before == max_tokens) {
                if (watch) |w| _ = w.take(w.context, next);
                return .length;
            }
            if (hang_at) |n| if (self.tokens.items.len - before == n) return .cut;
            if (watch) |w| if (w.take(w.context, next)) return .cut;
            return null;
        }

        pub fn halted(watch: ?Watch) bool {
            const w = watch orelse return false;
            const halt = w.halt orelse return false;
            return halt(w.context);
        }

        fn emit(self: *Self, ids: []const i32) !void {
            try self.tokens.appendSlice(self.gpa, ids);
            try self.tell(.pick, ids);
        }
    };
}
