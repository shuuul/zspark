//! The loop guard and the thinking budget (spec 074 WS-02), as TensorFold 1.0.5 has them in its own engine
//! (`zig/src/core/lanes/stream.zig`: `thinkCut`, `startClose`, `detectLoop`, `commit`; `engine.zig`'s round),
//! made a reply's constraint: `Guard` is a `grammar.Constraint` around the one the reply already has, or
//! around none. `server/served.zig` makes one for a request that asks for either, and every decoder follows
//! it through `masks.zig` as it follows a grammar, so a session, a pool and both models have it from one file.
//!
//! What it does, each as TensorFold does:
//!
//! - **The budget** counts the reply's tokens from its first. While the think block is open, the token at
//!   index `budget - 1` is `close[0]` and the rest of `close` (the host's `"\n"`, `</think>`, `"\n\n"`) is
//!   forced after it. A `</think>` the model wrote earlier ends the budget.
//! - **The loop**: after each token while the block is open, the shortest period of 1 to 8 whose last 256
//!   tokens repeat, after 64 tokens of warm-in, is found by TensorFold's own `detectLoop` (the build's `lanes`
//!   module). The token that completes the cycle stays the reply's; the whole of `close` is forced after it.
//!   It fires once a reply: the block is closed by it, and `period` keeps what it found for the reply's
//!   `Stats.loop_period`.
//! - Where both would act at one index the budget's close replaces the token, so no cycle completes there.
//!
//! A forced token is a mask with one bit. A row's mask depends on the reply's tokens before the row and on
//! nothing else, so the row rule holds as for a grammar: a window's row and the serial step at its position
//! force the same token, a draft that is not the forced token is cut before its window runs (`Masks.walk`),
//! and the forced tokens are drafted without a model (`Masks.next`). Where nothing is forced and no grammar
//! has begun, `free` says so and `Masks` neither masks the row nor hands a drafter a mask.
//!
//! The place of a position is the count of reply tokens before it: `tokens[0..p]` are those tokens and `at[p]`
//! what they leave. `accept` at `p` writes `tokens[p]` and `at[p + 1]`. That is sound by how `Masks` asks: it
//! accepts forward from the newest token's place, walks another draft at a row only by walking that row
//! again (which writes it over), and commits along the rows it walked, so what lies before a place asked
//! about is always the tokens on its own path.
//!
//! With a grammar inside (a document, a required call), the grammar begins after `gate`, the token `Masks`
//! was given for it, as it does without a guard. TensorFold 1.0.5's engine enforces no grammar and gives no
//! reference here; vLLM and SGLang both force through the reasoning's end and start the grammar there. So
//! after the gate the grammar governs: what is left of `close` is forced where the grammar takes it (its
//! lead is what the template writes after `</think>`, the same blank line) and dropped where it does not.
//!
//! Departures from TensorFold, each because zspark's decoders end a reply themselves: a fire with an empty
//! `close` ends TensorFold's reply as stopped, and here such a request is not guarded (`Ask.of`: the host
//! never sends one); a fire on the reply's last token is its end by length in both.

const std = @import("std");
const Allocator = std.mem.Allocator;
const grammar = @import("../text/grammar.zig");
const lanes = @import("lanes");

const Constraint = grammar.Constraint;
const Position = grammar.Position;
const Error = grammar.Error;

/// The tokens a cycle is read from: no more of the reply's tail changes `detectLoop`'s answer.
const tail = lanes.stream.loop_max_period + lanes.stream.loop_min_run + lanes.stream.loop_warm_in;

/// What a request asks, in TensorFold's terms (`engine_api.Request`).
pub const Ask = struct {
    /// `think_budget`: the reply's tokens before the close, 0 for no limit.
    budget: u32,
    /// `loop_guard`.
    loop: bool,
    /// `think_close`: the tokens that close the block, one at least. The request's, as long as the reply.
    close: []const u32,
    /// `think_end`: `</think>`.
    end: ?u32,

    /// Null where the request asks for neither, or for what cannot be forced: no closing tokens.
    pub fn of(budget: u32, loop: bool, close: []const u32, end: ?u32) ?Ask {
        if (close.len == 0) return null;
        const guarded = loop and end != null;
        if (budget == 0 and !guarded) return null;
        return .{ .budget = budget, .loop = guarded, .close = close, .end = end };
    }
};

/// What the reply's tokens leave at a place.
const At = struct {
    /// The grammar's position, once `begun`.
    inner: Position = undefined,
    /// Tokens of `close` still to force: the next is `close[close.len - left]`.
    left: u32 = 0,
    /// The think block is open: TensorFold's `think_open`.
    open: bool,
    /// The grammar has begun: its gate was taken, or it has none.
    begun: bool,
    /// The cycle that fired, 0 while none has.
    period: u8 = 0,
};

pub const Guard = struct {
    ask: Ask,
    /// The reply's own constraint, the caller's to free; null for free text.
    inner: ?Constraint,
    words: usize,
    /// The token the grammar begins after, as `Masks.init` was given it; null: with the reply.
    gate: ?u32 = null,
    /// `tokens[p]`: the token last accepted at place `p`.
    tokens: []u32,
    /// `at[p]`: the place after `p` tokens. One more than `tokens`.
    at: []At,
    /// The place after the reply's newest token, as `Masks` last said (`reach`): what `period` reads.
    reached: usize = 0,

    /// A guard over a reply of `most` tokens at most, in `arena`: nothing is freed but with it. `words` is the
    /// mask's length, `inner`'s where there is one.
    pub fn init(arena: Allocator, ask: Ask, inner: ?Constraint, words: usize, most: usize) Allocator.Error!*Guard {
        const self = try arena.create(Guard);
        // a window walks its drafts past the reply's last token before it is cut
        const places = most + rows_ahead + 1;
        self.* = .{ .ask = ask, .inner = inner, .words = words, .tokens = try arena.alloc(u32, places - 1), .at = try arena.alloc(At, places) };
        self.at[0] = .{ .open = true, .begun = true };
        return self;
    }

    /// The most drafts a window walks (`masks.max_rows - 1`) and one for the row after them: `masks.zig` holds it
    /// to `max_rows`.
    pub const rows_ahead = 16;

    pub fn constraint(self: *Guard) Constraint {
        return .{ .context = self, .words = self.words, .table = &table };
    }

    /// The guard `held` is, where it is one.
    pub fn of(held: Constraint) ?*Guard {
        return if (held.table == &table) @ptrCast(@alignCast(held.context)) else null;
    }

    /// The cycle that closed the reply's think block, once a committed token has fired it.
    pub fn period(self: *const Guard) ?u32 {
        const found = self.at[self.reached].period;
        return if (found == 0) null else found;
    }

    /// `at` is the place after the reply's newest token.
    pub fn reach(self: *Guard, at: Position) void {
        self.reached = @intCast(@backingInt(at));
    }

    /// Nothing holds the token at `at`: none is forced and no grammar has begun.
    pub fn free(self: *Guard, at: Position) Error!bool {
        const p: usize = @intCast(@backingInt(at));
        if (self.at[p].begun and self.inner != null) return false;
        return try self.forced(p) == null;
    }

    const table: Constraint.Table = .{ .start = start, .fill = fill, .accept = accept, .ended = ended, .deinit = lent };

    /// The budget closes the block at `p`: TensorFold's `cutsNext`.
    fn cuts(self: *const Guard, p: usize) bool {
        return self.at[p].open and self.ask.budget > 0 and p + 1 >= self.ask.budget;
    }

    /// The token that must come at `p`, or null. After the gate a token the grammar does not take is not
    /// forced: the grammar's own lead stands there.
    fn forced(self: *Guard, p: usize) Error!?u32 {
        const from = self.at[p];
        const close = self.ask.close;
        const token = if (from.left > 0) close[close.len - from.left] else if (self.cuts(p)) close[0] else return null;
        if (from.begun) if (self.inner) |inner| {
            if (try inner.accept(from.inner, token) == null) return null;
        };
        return token;
    }

    fn start(context: *anyopaque) Error!Position {
        const self: *Guard = @ptrCast(@alignCast(context));
        self.at[0] = .{ .open = true, .begun = self.gate == null };
        if (self.inner) |inner| self.at[0].inner = try inner.start();
        self.reached = 0;
        return @fromBackingInt(0);
    }

    fn fill(context: *anyopaque, at: Position, mask: []u32) Error!void {
        const self: *Guard = @ptrCast(@alignCast(context));
        const p: usize = @intCast(@backingInt(at));
        if (try self.forced(p)) |token| {
            if (token / 32 >= mask.len) return error.ConstraintFailed;
            @memset(mask, 0);
            mask[token / 32] = @as(u32, 1) << @intCast(token % 32);
            return;
        }
        const from = self.at[p];
        if (from.begun) if (self.inner) |inner| return inner.fill(from.inner, mask);
        @memset(mask, std.math.maxInt(u32));
    }

    fn accept(context: *anyopaque, at: Position, token: u32) Error!?Position {
        const self: *Guard = @ptrCast(@alignCast(context));
        const p: usize = @intCast(@backingInt(at));
        if (p + 1 >= self.at.len) return error.ConstraintFailed;
        const from = self.at[p];
        const must = try self.forced(p);
        if (must) |one| if (token != one) return null;
        var to = from;
        // the close goes on, begins (the budget), or is given up where the grammar would not take its token
        if (from.left > 0) {
            to.left = if (must != null) from.left - 1 else 0;
        } else if (self.cuts(p)) {
            to.open = false;
            to.left = if (must != null) @intCast(self.ask.close.len - 1) else 0;
        }
        if (from.begun) {
            if (self.inner) |inner| to.inner = try inner.accept(from.inner, token) orelse return null;
        } else if (token == self.gate.?) {
            to.begun = true;
        }
        if (self.ask.end) |end| if (token == end) {
            to.open = false;
        };
        self.tokens[p] = token;
        if (self.ask.loop and to.open and to.period == 0) {
            if (lanes.stream.detectLoop(self.tokens[p + 1 -| tail .. p + 1])) |found| {
                to.period = @intCast(found);
                to.open = false;
                to.left = @intCast(self.ask.close.len);
            }
        }
        self.at[p + 1] = to;
        return @fromBackingInt(@as(u64, p + 1));
    }

    fn ended(context: *anyopaque, at: Position) bool {
        const self: *Guard = @ptrCast(@alignCast(context));
        const from = self.at[@intCast(@backingInt(at))];
        const inner = self.inner orelse return false;
        return from.begun and inner.ended(from.inner);
    }

    /// The arena's; `inner` is its maker's.
    fn lent(_: *anyopaque) void {}
};
