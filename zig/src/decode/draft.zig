//! Drafted decoding over any model, as TensorFold's `mtp_decode`: the main model runs the last kept token and
//! the drafts of what follows as one window, chooses a token after every row, and keeps the rows up to the
//! first draft that is not its own choice. A draft chooses nothing, so the reply is the serial one whatever
//! the drafts are; a right draft saves a pass over the weights.
//!
//! Nothing here names a model. `Decoder` and `Chooser` are generic over a model's namespace (their `M`, whose
//! contract is `Decoder`'s doc comment) at compile time, so a round calls the model's own functions directly;
//! `Drafter`, `Chain`, `Follow` and `Stats` need no model at all. A model instantiates them once
//! (`models/qwen4/draft.zig`) and its files name that.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");
const contract = @import("contract.zig");
const logprob = @import("logprob.zig");
const masks_file = @import("masks.zig");
const sampling = @import("sampling.zig");

/// Nanoseconds the host spent choosing the tokens of windows' rows (`accept`), spec 062's measure.
pub var choosing: u64 = 0;

/// Where drafts come from: what a model's draft source gives the decoder (Flash Next's MTP head, `mtp.Head`;
/// a second drafter beside it, as GLM's DFlash2; a test makes them up). Chosen when a reply starts, so this
/// is a pointer and a call a round, as it was; the main model's passes are not behind it.
pub const Drafter = struct {
    context: *anyopaque,
    /// The main model kept `tokens.len` rows. `streams` bf16 [rows, streams * hidden] holds their final residual
    /// streams, valid until the next forward; `tokens[i]` is the token chosen after row i, and the last of them
    /// sits at `position`. Writes drafts of the tokens after it, at most `out.len`, and returns how many.
    /// `guide` is the reply's constraint, where it has one: what is known of each draft before a model is asked
    /// (`masks.Guide`). A drafter may leave it unasked; with `out` empty it must.
    draft: *const fn (context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize,
};

pub const Stats = struct {
    /// Windows run.
    rounds: usize = 0,
    /// Drafts those windows carried, and how many of them were the main model's choice.
    drafted: usize = 0,
    accepted: usize = 0,
};

/// What a decoder over `M` can fail with.
pub fn Error(comptime M: type) type {
    return error{DraftFailed} || M.Error || sampling.Error || masks_file.Error || Allocator.Error;
}

/// A reply known beforehand, which the main model's choices are replaced by: two programs' drafts are then
/// counted on one text, each against the same tokens (spec 028; `zspark generate ... follow=FILE`). A
/// measurement's, never a reply's: the tokens are no longer the model's.
pub const Follow = struct {
    /// `tokens[i]` is the token at position `first + i`; a position past them keeps the model's choice.
    tokens: []const i32,
    first: usize,

    pub fn at(self: Follow, position: usize, own: i32) i32 {
        return if (position >= self.first and position - self.first < self.tokens.len) self.tokens[position - self.first] else own;
    }
};

/// What a reply that asks for log probabilities lends its chooser (spec 101 WS-02, `decode/logprob.zig`): how
/// many best tokens a row, the device's words, and where each chosen token's row goes, beside `chosen`.
pub const Logprobs = struct {
    k: u8,
    scratch: *logprob.Scratch,
    /// `rows[i]` is the row of the token chosen after row i of a window (`Chooser.accept`'s `chosen[i]`).
    rows: []logprob.Row,
};

/// How one stream's tokens are chosen from the main model's logits: what a session's decoder holds for its
/// one stream and a round of several streams for each of them. Of `M` (see `Decoder`) it reads `Buffers`,
/// `logitsRow` and `Error`, and for log probabilities `logitRows(Buffers, row, n) cuda.View` (bf16 [n, V], the
/// rows `logitsRow` copies), which a model without it does not give: its `Info.logprobs` stays false.
pub fn Chooser(comptime M: type) type {
    return struct {
        sampler: *sampling.Sampler,
        settings: sampling.Sampling,
        /// Tokens that end the reply.
        eos: []const u32,
        /// One row's logits on the host.
        logits: []u16,
        /// The reply's constraint, laid over each row's logits at the row's own place.
        masks: ?*masks_file.Masks = null,
        follow: ?Follow = null,
        /// The reply asks for log probabilities: each kept row's, for the token chosen after it.
        logprobs: ?Logprobs = null,
        /// The window's rows settled already, a round's beside others' (`decode/settle.zig`'s `readRound`, spec
        /// 144 WS-03): row i's token where not null, as `M.settle`'s, in its place. Asked under neither a
        /// constraint nor log probabilities.
        settled: ?[]const ?i32 = null,

        const Self = @This();
        /// The model gives its rows of logits on the device, which log probabilities are found from; for a
        /// model without them (a host test's) none of that is compiled, and `logprobs` is never set.
        const rowed = @hasDecl(M, "logitRows");

        /// What `accept` found: the rows kept, and whether the token chosen after the last of them ends the reply.
        pub const Accepted = struct { keep: usize, ended: bool };

        /// The rows of a verify window that are kept. `tokens` is the window, the stream's newest token and the
        /// drafts after it, which follows position `pos` of the sequence; its logits are rows `row` and on of
        /// `b`. A token is chosen after each row, into `chosen`, by the key of its own place in the sequence, as
        /// a serial step's is; a row is kept while the row before it chose its token and did not end the reply.
        /// This is the one place a draft is accepted, so a stream's kept rows are the same in a session's window
        /// and beside others in a round.
        ///
        /// A model may settle rows on the device first (optional `M.settle(b, row, sampler, settings, out:
        /// []?i32)`, Flash Next's two ranks, spec 095): a row it settles holds the token `choose` gives for the
        /// whole row and is not read back. It is not asked under a constraint, nor for log probabilities.
        ///
        /// With `logprobs`, the window's rows' words are launched before the first row is read back, and each
        /// kept row's goes to `logprobs.rows`, its chosen token's logit read from the row it was chosen from:
        /// after the constraint's mask, which leaves a token it allows as it was (a measurement's `follow` asks
        /// for none).
        pub fn accept(self: Self, b: M.Buffers, row: usize, pos: usize, tokens: []const i32, chosen: []i32) Error(M)!Accepted {
            var settled: [forward_max_rows]?i32 = @splat(null);
            if (self.masks == null and self.logprobs == null and tokens.len <= settled.len) {
                if (self.settled) |given| {
                    const n = @min(given.len, tokens.len);
                    @memcpy(settled[0..n], given[0..n]);
                } else if (@hasDecl(M, "settle")) try M.settle(b, row, self.sampler, self.settings, settled[0..tokens.len]);
            }
            var logits: [forward_max_rows]u16 = undefined;
            if (rowed) if (self.logprobs) |lp| {
                if (tokens.len > logits.len or tokens.len > lp.rows.len) return error.DraftFailed;
                try launchRows(lp, b, row, tokens.len);
            };
            var keep: usize = 0;
            var ended = false;
            while (keep < tokens.len) {
                const own = (if (keep < settled.len) settled[keep] else null) orelse own: {
                    try M.logitsRow(b, row + keep, self.logits);
                    if (self.masks) |masks| try masks.apply(keep, self.logits);
                    const before = cuda.monotonic();
                    const one = try self.sampler.choose(self.logits, pos + keep + 1, self.settings);
                    choosing += cuda.monotonic() - before;
                    break :own one;
                };
                const token = if (self.follow) |follow| follow.at(pos + keep + 1, own) else own;
                chosen[keep] = token;
                if (self.logprobs != null) logits[keep] = self.logits[@intCast(token)];
                keep += 1;
                ended = std.mem.indexOfScalar(u32, self.eos, @intCast(token)) != null;
                if (keep == tokens.len or token != tokens[keep] or ended) break;
            }
            if (rowed) if (self.logprobs) |lp| {
                lp.scratch.read(keep) catch return error.DraftFailed;
                for (chosen[0..keep], logits[0..keep], lp.rows[0..keep], 0..) |token, logit, *out, i| out.* = lp.scratch.row(i, token, logit, lp.k);
            };
            return .{ .keep = keep, .ended = ended };
        }

        /// For a reply that asks for log probabilities, launches the words of row `row` of `b`, the row `first`
        /// is to choose from: before anything is run or committed after it. `firstRow` reads them.
        pub fn launchFirst(self: Self, b: M.Buffers, row: usize) Error(M)!void {
            if (rowed) if (self.logprobs) |lp| try launchRows(lp, b, row, 1);
        }

        /// The row of `token`, chosen by `first` from the row `launchFirst` launched, whose logits `logits`
        /// holds: for a reply that asks for log probabilities, else null.
        pub fn firstRow(self: Self, token: i32) Error(M)!?logprob.Row {
            if (!rowed) return null;
            const lp = self.logprobs orelse return null;
            lp.scratch.read(1) catch return error.DraftFailed;
            return lp.scratch.row(0, token, self.logits[@intCast(token)], lp.k);
        }

        fn launchRows(lp: Logprobs, b: M.Buffers, row: usize, n: usize) Error(M)!void {
            lp.scratch.launch(M.logitRows(b, row, n), lp.k) catch return error.DraftFailed;
        }

        /// The reply's first token, from the logits of the prompt's last row, which `logits` holds: chosen at
        /// the constraint's first place (a window of one row, no drafts) by the key of position `len`, the
        /// prompt's length.
        pub fn first(self: Self, len: usize) Error(M)!i32 {
            if (self.masks) |masks| {
                _ = try masks.window(&.{});
                try masks.apply(0, self.logits);
            }
            var token = try self.sampler.choose(self.logits, len, self.settings);
            if (self.follow) |follow| token = follow.at(len, token);
            if (self.masks) |masks| try masks.commit(1, token);
            return token;
        }
    };
}

/// A stream's drafts asked of a drafter's seats (GLM's `mtp/mtp_seats.zig`, `decode/dflash2/seats.zig`, Flash Next's
/// `hall_engine.zig`):
/// the rows the main model kept of it and what its chain is chosen by, as `draft.Drafter.draft` takes them of
/// one stream.
pub const Job = struct {
    slot: usize,
    /// The row of the main model's last window its kept rows start at.
    from: usize,
    kept: []const i32,
    position: usize,
    settings: sampling.Sampling,
    ends: []const u32,
    guide: ?masks_file.Guide,
    out: []i32,
    /// The seat is full: nothing more is absorbed, and nothing drafted.
    full: *bool,
    /// The drafts written.
    count: usize = 0,
    chain: Chain = undefined,
    live: bool = false,
    /// Its row among the last step's.
    index: usize = 0,
};

/// One stream's chain of drafts while it is made: the tokens the reply's constraint forces, and between
/// them the ones a head chooses, each put to the constraint. `step` is the chain's rule, the same for a
/// session's head (one stream) and for a round of several streams (a chain a stream); whoever holds the
/// head runs it over the drafts `step` names and chooses with it by one rule (`mtp_rule.Rule`). The head runs a
/// draft only once a later one is its own to choose, so a run of forced tokens costs no head step: until
/// spec 057 a round ran each draft as soon as it was made and chose from the whole row, read back.
pub const Chain = struct {
    /// Where the drafts go; its length is the most the chain may hold.
    out: []i32,
    /// What the constraint says of the draft after `out[0..count]`; `.free` without one.
    next: masks_file.Next = .closed,
    count: usize = 0,
    /// Of `out[0..count]`, the drafts the head has run.
    ran: usize = 0,
    /// The drafts the head chose itself, and the product of the probabilities it gave them.
    own: usize = 0,
    product: f64 = 1,
    /// The newest draft is the head's, which the constraint has yet to take.
    asked: bool = false,
    /// The chain ends with its newest draft: one the head gave little, or one that ends the reply.
    ended: bool = false,
    /// Nothing more is drafted.
    halt: bool = false,
    /// The head's own choices in order, each with the probability it gave it: the drafts taken and, last, the
    /// one that ended the chain if it was not taken. How two programs' chains, or a session's and a round's,
    /// are set beside each other.
    trail: [forward_max_rows + 1]Considered = undefined,
    trailed: usize = 0,

    pub const Considered = struct { token: i32, p: f32 };

    /// What the chain needs before it can go on: nothing more (`halt`), the head run over `out[ran]`, after
    /// which the caller adds one to `ran` (`run`), or the head's choice of the next draft, given to `take`
    /// (`choose`).
    pub const Need = enum { halt, run, choose };

    /// A chain of at most `out.len` drafts, at its first place under `guide`.
    pub fn begin(out: []i32, guide: ?masks_file.Guide) !Chain {
        return .{ .out = out, .next = if (out.len == 0) .closed else if (guide) |g| try g.next(out[0..0], true) else .free };
    }

    /// Whether the next draft is the head's to choose, so that its logits are wanted.
    pub fn scores(self: Chain) bool {
        return self.next == .free or self.next == .among;
    }

    /// Takes the tokens the constraint forces until the chain needs the head or ends.
    pub fn step(self: *Chain, guide: ?masks_file.Guide) !Need {
        while (true) {
            if (self.halt) return .halt;
            switch (self.next) {
                .closed => self.halt = true,
                // only a draft of the head's own: the constraint's are ones it takes
                .refused => {
                    if (self.count > 0) self.count -= 1;
                    self.halt = true;
                },
                .token => |token| {
                    self.out[self.count] = token;
                    self.count += 1;
                    self.asked = false;
                    try self.refresh(guide);
                },
                // the head goes through the drafts it has not run, forced ones, to stand at this one
                .free, .among => return if (self.ran < self.count) .run else .choose,
            }
        }
    }

    /// The head's choice for the next draft: `token`, to which it gave the probability `p`. `low`: by the
    /// caller's rule that is little, and the chain stops before the draft unless it is the head's first,
    /// which is kept and ends it. `last`: the token ends the reply.
    pub fn take(self: *Chain, guide: ?masks_file.Guide, token: i32, p: f64, low: bool, last: bool) !void {
        if (self.trailed < self.trail.len) {
            self.trail[self.trailed] = .{ .token = token, .p = @floatCast(p) };
            self.trailed += 1;
        }
        if (low and self.own > 0) {
            self.halt = true;
            return;
        }
        self.out[self.count] = token;
        self.count += 1;
        self.own += 1;
        self.product *= p;
        // under a constraint the head's choice is put to it, the last of a full window's too
        self.asked = guide != null;
        self.ended = low or last;
        try self.refresh(guide);
    }

    fn refresh(self: *Chain, guide: ?masks_file.Guide) !void {
        const more = self.count < self.out.len and !self.ended;
        if (!more and !self.asked) {
            self.halt = true;
            return;
        }
        self.next = if (guide) |g| try g.next(self.out[0..self.count], more) else .free;
    }
};

/// A window's rows are held on the stack; windows are 16 rows at most (TensorFold's `MAX_DEPTH` 15).
pub const forward_max_rows = 16;

/// The rows of a prompt's window from position `pos` that may take up to `most` rows of the `left` before the
/// prompt's last window: all that is left where they fit, else the most that end at a multiple of `alignment`
/// (a model's `prompt_align`), or `most` where none does. So a prompt's rows cut into windows from any start
/// meet the same multiples (GLM's KDA chunks, spec 116 WS-06); an alignment of 1 cuts at `most`.
pub fn promptCut(pos: usize, most: usize, left: usize, alignment: usize) usize {
    if (left <= most) return left;
    const end = std.mem.alignBackward(usize, pos + most, alignment);
    return if (end > pos) end - pos else most;
}

/// One window of a prompt run from where `state` stands in `tokens`: as many rows as the buffers and the model
/// take, scored at its last row where it ends the prompt, then committed. A serial reply's prompt is these windows
/// until the state stands at its end (Qwen3-ASR's and Unlimited-OCR's greedy and served replies, spec 142
/// WS-04); `Decoder.prompt` plans its windows instead.
pub fn promptWindow(comptime M: type, m: M.Model, state: *M.State, b: *M.Buffers, tokens: []const i32) !void {
    const rows = @min(b.rows, m.max_rows, tokens.len - state.pos);
    const run = tokens[state.pos..][0..rows];
    try M.forwardScoring(m, state, b, run, if (state.pos + rows == tokens.len) .last else .none);
    try M.commit(m, state, b.*, run, rows);
}

/// `Decoder`'s contract as declarations, in its doc comment's order: a name added there is added here.
pub const owed = [_]contract.Owed{
    .{ .name = "Model", .is = .{ .type = .{ .fields = &.{"max_rows"} } } },
    .{ .name = "State", .is = .{ .type = .{ .fields = &.{"pos"} } } },
    .{ .name = "Buffers", .is = .{ .type = .{ .fields = &.{"rows"}, .decls = &.{"scoredRows"} } } },
    .{ .name = "Graphs", .is = .{ .type = .{ .decls = &.{ "forward", "holds" } } } },
    .{ .name = "Error", .is = .{ .type = .{} } },
    .{ .name = "forward", .is = .{ .function = 4 } },
    .{ .name = "forwardScoring", .is = .{ .function = 5 } },
    .{ .name = "commit", .is = .{ .function = 5 } },
    .{ .name = "logitsRow", .is = .{ .function = 3 } },
    .{ .name = "streams", .is = .{ .function = 2 } },
};

/// The drafted decoder of one stream over the model `M`: a prompt in windows (`advance`, `prompt`), then a
/// reply in verify windows (`round`), with drafts asked of a `Drafter` between them (`draft`).
///
/// `M` is a namespace (a struct of declarations; `models/qwen4/draft.zig`'s `namespace` is Flash Next's) that
/// declares all of:
///
/// - `Model`: the weights on the device, passed by value. Its field `max_rows: usize` is the most rows one
///   forward takes.
/// - Optionally `prompt_align: usize`: a prompt's windows before its last end at multiples of it (`promptCut`);
///   1 where not declared.
/// - `State`: one sequence's state, changed only by `commit`. Its field `pos: usize` is the number of tokens
///   committed, so the next window's first row stands at position `pos`.
/// - `Buffers`: what a forward writes and nothing keeps: the window's logits and final residual streams. Its
///   field `rows: usize` is the most rows a window of these buffers holds; `scoredRows() usize` the most rows
///   whose logits they hold (a reply's window is at most that, and at most `forward_max_rows`).
/// - `Graphs`: the forward as captured graphs. `forward(*Graphs, Model, *State, *Buffers, tokens) !void` is
///   `M.forward` by another launch, the same bits; `holds(*const Graphs, State, Buffers, rows) bool` says a
///   window of `rows` rows from this state runs without a capture, with every row scored.
/// - `Error`: the errors of the functions below.
/// - `forward(Model, *State, *Buffers, tokens: []const i32) Error!void`: runs `tokens` as one window that
///   follows the state, row i at position `pos + i`, and scores every row. It changes nothing a later
///   forward from the same state would read: a window can be run and never committed.
/// - `forwardScoring(Model, *State, *Buffers, tokens, score) Error!void`: the same window scoring what
///   `score` names, an enum with `none` (no logits) and `last` (the last row's logits into row 0).
/// - `commit(Model, *State, Buffers, tokens, keep: usize) !void`: makes the first `keep` rows (1 to
///   `tokens.len`) of the window just run part of the sequence; `pos` grows by `keep`.
/// - `logitsRow(Buffers, row: usize, out: []u16) !void`: waits for the device and copies a row's logits,
///   bf16 bits a token, to the host.
/// - `streams(Buffers, rows: usize) cuda.View`: the final residual streams of the window's first `rows`
///   rows, what a `Drafter` reads, valid until the next forward.
///
/// What must hold of them, since the decoder is exact only because of it:
///
/// - A row's bits never depend on the rows beside it: row i's logits and streams are a function of the
///   committed sequence and `tokens[0 .. i + 1]` alone, whatever follows in the window, however wide it is
///   and whichever launch ran it (`forward`, a graph, `forwardScoring`).
/// - A window equals its serial steps: after `commit(.., keep)` the state is, bit for bit, the state after
///   `keep` windows of one row each committed one by one, so the rows a round drops leave no trace.
/// - `logitsRow` of a row is what a serial step at that position would have read.
///
/// The decoder never looks inside a state or a snapshot; `within` and `kept` are where a session keeps one.
///
/// The declarations are held when `M` is given (`owed`, `contract.zig`); the rules are the tests' to hold.
pub fn Decoder(comptime M: type) type {
    contract.hold("decode/draft.zig's Decoder (its doc comment)", M, &owed);
    return struct {
        m: M.Model,
        state: *M.State,
        b: *M.Buffers,
        /// Null launches each kernel from the host.
        graphs: ?*M.Graphs,
        sampler: *sampling.Sampler,
        settings: sampling.Sampling,
        /// Tokens that end the reply: a row whose choice is one of them is the last kept.
        eos: []const u32,
        /// One row's logits on the host.
        logits: []u16,
        /// The reply's constraint: each row's logits are masked at the row's own place before its token is chosen,
        /// and the drafts it forbids are not run. Null: nothing is masked, cut or asked.
        masks: ?*masks_file.Masks = null,
        /// Called for each window of a prompt once it has run and before it is committed, so that a state inside
        /// it can be kept without the window being cut there.
        within: ?Within = null,
        /// Called for each window of a reply (`round`) once its kept rows are known and before they are committed,
        /// so that a state inside it can be kept the same way.
        kept: ?Kept = null,
        /// A measurement's known reply (`Follow`).
        follow: ?Follow = null,
        /// The reply asks for log probabilities: `rows` beside `round`'s and `prompt`'s `chosen`, as long.
        logprobs: ?Logprobs = null,
        stats: Stats = .{},
        /// Prompts whose last window was launched as a captured graph (`prompt`).
        graphed: usize = 0,

        const Self = @This();

        pub const Error = draft_file.Error(M);

        fn chooser(self: *const Self) Chooser(M) {
            return .{ .sampler = self.sampler, .settings = self.settings, .eos = self.eos, .logits = self.logits, .masks = self.masks, .follow = self.follow, .logprobs = self.logprobs };
        }

        pub const Within = struct {
            context: *anyopaque,
            /// `run` is the window's tokens, which follow the state's position; `last`: the prompt ends with it.
            window: *const fn (context: *anyopaque, run: []const i32, last: bool) Self.Error!void,
            /// Called before the window `run` runs: a window wider than the rows a model keeps a state for must
            /// be told which states inside it are wanted.
            before: ?*const fn (context: *anyopaque, run: []const i32) Self.Error!void = null,
            /// Called once the window `run` is launched and before anything waits for it: what the host has to
            /// do for `window` that needs no result (a snapshot's buffers allocated) is done while the device
            /// computes.
            launched: ?*const fn (context: *anyopaque, run: []const i32) Self.Error!void = null,
        };

        pub const Kept = struct {
            context: *anyopaque,
            /// `tokens` is the whole window, which follows the state's position; its first `keep` rows are kept.
            window: *const fn (context: *anyopaque, tokens: []const i32, keep: usize) Self.Error!void,
        };

        /// The most drafts a window can carry.
        pub fn depth(self: Self) usize {
            return @min(self.b.scoredRows(), self.m.max_rows, forward_max_rows) - 1;
        }

        /// Runs `last`, the newest token of the sequence and not yet committed, and `drafts` of the tokens after it
        /// as one window. Returns how many rows it kept; `chosen[i]` is the token chosen after kept row i, so the
        /// last of them is the sequence's newest and the next round's `last`. Every kept draft is `chosen[i - 1]`.
        /// Under a constraint the drafts it forbids are cut first, and `stats.drafted` counts the ones run.
        pub fn round(self: *Self, last: i32, given: []const i32, chosen: []i32) Self.Error!usize {
            std.debug.assert(given.len <= self.depth() and chosen.len > given.len);
            const drafts = if (self.masks) |masks| given[0..try masks.window(given)] else given;
            var window: [forward_max_rows]i32 = undefined;
            const rows = 1 + drafts.len;
            window[0] = last;
            @memcpy(window[1..rows], drafts);
            const tokens = window[0..rows];
            if (self.graphs) |graphs| try graphs.forward(self.m, self.state, self.b, tokens) else try M.forward(self.m, self.state, self.b, tokens);

            const keep = (try self.chooser().accept(self.b.*, 0, self.state.pos, tokens, chosen)).keep;
            if (self.kept) |kept| try kept.window(kept.context, tokens, keep);
            try M.commit(self.m, self.state, self.b.*, tokens, keep);
            if (self.masks) |masks| try masks.commit(keep, chosen[keep - 1]);
            self.stats.rounds += 1;
            self.stats.drafted += drafts.len;
            self.stats.accepted += keep - 1;
            return keep;
        }

        /// Runs the prompt `tokens` up to its position `to`, below its length, from where the state stands: windows
        /// kept whole, run eagerly, and absorbed by `drafter`. The state is then the sequence's at `to`, for a
        /// snapshot of it, and `prompt` runs the rest.
        pub fn advance(self: *Self, drafter: ?Drafter, tokens: []const i32, to: usize) Self.Error!void {
            std.debug.assert(self.state.pos <= to and to < tokens.len);
            // as wide as the buffers take: what a window costs grows far slower than its rows
            const window = @min(self.b.rows, self.m.max_rows);
            while (self.state.pos < to) {
                const rows = promptCut(self.state.pos, window, to - self.state.pos, prompt_align);
                const run = tokens[self.state.pos..][0..rows];
                if (self.within) |within| if (within.before) |before| try before(within.context, run);
                try M.forwardScoring(self.m, self.state, self.b, run, .none);
                if (self.within) |within| if (within.launched) |launched| try launched(within.context, run);
                if (self.within) |within| try within.window(within.context, run, false);
                try M.commit(self.m, self.state, self.b.*, run, rows);
                if (drafter) |d| _ = try self.ask(d, tokens[self.state.pos - rows + 1 ..][0..rows], chosen_none);
            }
        }

        const chosen_none: []i32 = &.{};

        /// The rows of the window `prompt` runs from position `pos` of a prompt `len` tokens long, when `chosen`
        /// holds `room` tokens: all that is left where the last window takes it, else as many as the buffers take
        /// of what comes before the last window. So whoever plans a prompt knows its windows before they run.
        pub fn promptRows(self: Self, pos: usize, len: usize, room: usize) usize {
            const wide = @min(self.b.rows, self.m.max_rows);
            const left = len - pos;
            const last = @min(wide, room);
            return if (left <= last) left else promptCut(pos, wide, left - last, prompt_align);
        }

        const prompt_align: usize = if (@hasDecl(M, "prompt_align")) M.prompt_align else 1;

        /// Runs the prompt `tokens` from where the state stands in it (an empty state: its first token): `advance`
        /// up to its last window, then that window, which is as wide as the buffers and `chosen` take, so a tail
        /// that fits one window is one window (a window's cost is mostly fixed: 16 rows take a quarter of what 512
        /// do). All are kept whole and run eagerly, prompts differing in length, but a last window of as many rows
        /// as a reply's window whose graph is captured already (`Graphs.holds`: a prompt sent again, a
        /// short new turn): it is launched as that graph, every row scored, for the same bits and without the time
        /// that lies between kernels launched one by one (3 ms of a seven-row window's 67 on NVFP4, spec 024's
        /// trace; spec 041). Nothing is captured for a prompt. Returns n, with `chosen[0..n]` the tokens after each row of the last window: the prompt's own, then
        /// the first token of the reply, as `round` leaves them. `drafter` absorbs every window before the last;
        /// `draft` hands it the last with `chosen[0..n]`.
        pub fn prompt(self: *Self, drafter: ?Drafter, tokens: []const i32, chosen: []i32) Self.Error!usize {
            std.debug.assert(self.state.pos < tokens.len and chosen.len >= 1);
            const window = @min(self.b.rows, self.m.max_rows, chosen.len);
            try self.advance(drafter, tokens, tokens.len - @min(window, tokens.len - self.state.pos));
            const run = tokens[self.state.pos..];
            const rows = run.len;
            if (self.within) |within| if (within.before) |before| try before(within.context, run);
            const graphed = if (self.graphs) |graphs| graphs.holds(self.state.*, self.b.*, rows) else false;
            if (graphed) {
                // the graph scores every row, so the last row's logits are the window's last
                try self.graphs.?.forward(self.m, self.state, self.b, run);
                self.graphed += 1;
            } else {
                try M.forwardScoring(self.m, self.state, self.b, run, .last);
            }
            if (self.within) |within| if (within.launched) |launched| try launched(within.context, run);
            try M.logitsRow(self.b.*, if (graphed) rows - 1 else 0, self.logits);
            try self.chooser().launchFirst(self.b.*, if (graphed) rows - 1 else 0);
            if (self.within) |within| try within.window(within.context, run, true);
            try M.commit(self.m, self.state, self.b.*, run, rows);
            @memcpy(chosen[0 .. rows - 1], run[1..]);
            chosen[rows - 1] = try self.chooser().first(tokens.len);
            if (try self.chooser().firstRow(chosen[rows - 1])) |row| self.logprobs.?.rows[rows - 1] = row;
            return rows;
        }

        fn ask(self: *Self, drafter: Drafter, chosen: []const i32, out: []i32) Self.Error!usize {
            const guide: ?masks_file.Guide = if (self.masks) |masks| masks.guide() else null;
            const count = drafter.draft(drafter.context, M.streams(self.b.*, chosen.len), chosen, self.state.pos, guide, out) catch |err| {
                std.log.err("drafting after position {d}: {t}", .{ self.state.pos, err });
                return error.DraftFailed;
            };
            std.debug.assert(count <= out.len);
            return count;
        }

        /// Asks `drafter` for at most `want` drafts after the `keep` rows the last window kept, whose choices are
        /// `chosen`. Call it after `round` or `prompt`. Under a constraint the drafter is handed its guide, so a
        /// token the constraint forces is drafted without a model.
        pub fn draft(self: *Self, drafter: Drafter, chosen: []const i32, want: usize, out: []i32) Self.Error!usize {
            return self.ask(drafter, chosen, out[0..@min(want, self.depth(), out.len)]);
        }
    };
}

const draft_file = @This();
