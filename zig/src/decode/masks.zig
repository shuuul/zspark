//! A reply's constraint as the decoder follows it (spec 017): where the grammar stands after the sequence's
//! newest token, where it stands at each row of a window, and each row's logits with what may not come there
//! at -inf. As TensorFold's `engine/grammar.py` `Constraint` (`window`, `mask`, `advance`), on the host.
//!
//! The rule: a row's mask is made at the position its own path reaches, the tokens before the row and nothing
//! else. Row 0 chooses after the newest token, row i after draft i - 1; the positions are values
//! (`text/grammar.zig`), so the drafts are walked once before the window and nothing is rolled back. A serial
//! step is a window of one row and takes the same position by the same tokens, so the masks are the same bits
//! and a drafted reply stays the serial one.
//!
//! Drafts the constraint forbids are not run, as in TensorFold: the window is cut before the first draft that
//! may not come at its row (the row before it, masked, could never choose it) and before a draft that ends the
//! reply (nothing is verified after the end).
//!
//! The constraint also drafts (spec 023), which TensorFold's does not: a drafter may ask, through `Guide`, what
//! is known of the draft after the ones it has. Where the mask there holds one token, that token is the draft
//! and no model is asked; where it holds several, the drafter is handed the mask and chooses among them; before
//! the gate it is told nothing. A draft still chooses nothing: the window runs every row and each row's token
//! is chosen under its own mask, so what a guide says changes which rows a window can keep and never a token.
//!
//! Where the mask meets the logits: on the host, in the row the sampler is about to read. The sampler is on the
//! host and reads one downloaded row at a time (`sampling.Sampler.choose`); a mask on the device would be an
//! upload a row and a kernel in the step's graph for bits the host then downloads anyway.
//!
//! With thinking on the constraint begins at the token after `</think>` (`gate`): until that token is taken no
//! row is masked and no draft is cut.
//!
//! A reply under a loop guard or a thinking budget has `guard.zig`'s `Guard` for its constraint, around its
//! grammar or around none (spec 074). It acts before the gate, so `init` hands it the gate and keeps none;
//! where it says a place is free (nothing forced, no grammar begun) the row is not masked and a drafter is
//! told nothing, as before a gate; and it is told the newest token's place, for the period it reports.

const std = @import("std");
const grammar = @import("../text/grammar.zig");
const guard_file = @import("guard.zig");

pub const Error = grammar.Error;

/// bf16 -inf: what a token that may not come scores.
pub const minus_infinity: u16 = 0xff80;

/// The most rows of a window: a model's `draft.forward_max_rows` (`models/qwen4/draft.zig`).
pub const max_rows = 16;

comptime {
    // a guard keeps a place for every row a window walks past the reply's last token
    std.debug.assert(guard_file.Guard.rows_ahead >= max_rows);
}

pub const Stats = struct {
    /// Rows whose logits were masked.
    masked: usize = 0,
    /// Drafts cut before their window ran.
    pruned: usize = 0,
    /// Drafts the constraint supplied: the one token a mask left.
    forced: usize = 0,
    /// Nanoseconds filling masks, walking drafts and writing -inf.
    nanoseconds: u64 = 0,
};

/// Where a row stands: before the gate no constraint applies, after it at `at`.
const Place = struct { open: bool, at: grammar.Position };

/// What a constraint knows of a draft before any model is asked.
pub const Next = union(enum) {
    /// The last of the drafts cannot be run: the constraint forbids it at its row, or it ends the reply.
    refused,
    /// The drafts stand and none may follow: the reply would end here, or a window holds no more.
    closed,
    /// Nothing is known: the constraint has not begun.
    free,
    /// The one token that may come: the draft.
    token: i32,
    /// The tokens that may come, as a mask; good until the guide, or its `Masks`, is asked again.
    among: []const u32,
};

/// What a drafter asks of a reply's constraint. `Masks.guide` is the one; a test makes them up.
pub const Guide = struct {
    context: *anyopaque,
    nextFn: *const fn (context: *anyopaque, drafts: []const i32, more: bool) Error!Next,

    /// What is known of the draft after `drafts`, the drafts so far of the tokens after the sequence's newest.
    /// Without `more` no draft is wanted after them, only whether they stand: `.refused` or `.closed`.
    pub fn next(self: Guide, drafts: []const i32, more: bool) Error!Next {
        return self.nextFn(self.context, drafts, more);
    }
};

pub const Masks = struct {
    constraint: grammar.Constraint,
    /// The token the constraint begins after, or null where it begins with the reply.
    gate: ?i32,
    /// `constraint` where it is a loop guard or a thinking budget: it holds the gate itself.
    guard: ?*guard_file.Guard = null,
    /// After the sequence's newest token.
    newest: Place,
    /// `rows[i]`: where window row i's token is chosen. `rows[0]` is `newest`, and `rows[i + 1]` the place after
    /// `walked[i]` at `rows[i]`: the drafts walked since the last `commit`, by `next` or by `window`.
    rows: [max_rows]Place = undefined,
    walked: [max_rows - 1]i32 = undefined,
    /// `forced[i]`: `walked[i]` is the one token that may come at `rows[i]`, as `next` found it.
    forced: [max_rows - 1]bool = undefined,
    /// How many of `rows` are made.
    set: usize = 0,
    /// One row's mask, `[constraint.words]`. The caller's, at least as long as the reply.
    mask: []u32,
    stats: Stats = .{},
    /// A clock for `stats.nanoseconds`; null leaves it 0.
    io: ?std.Io = null,

    /// Before a reply's first token. `mask` is scratch of `constraint.words` words; `gate` is `</think>` where
    /// the prompt leaves the reply thinking.
    pub fn init(constraint: grammar.Constraint, mask: []u32, gate: ?i32) Error!Masks {
        if (mask.len != constraint.words) return error.ConstraintFailed;
        if (guard_file.Guard.of(constraint)) |guard| {
            // it counts the reply from its first token, the reasoning too: the gate is its grammar's
            guard.gate = if (gate) |token| @intCast(token) else null;
            return .{ .constraint = constraint, .gate = null, .guard = guard, .mask = mask, .newest = .{ .open = true, .at = try constraint.start() } };
        }
        return .{ .constraint = constraint, .gate = gate, .mask = mask, .newest = .{ .open = gate == null, .at = try constraint.start() } };
    }

    /// A guard holds nothing at `place`: no token is forced there and its grammar has not begun.
    fn unheld(self: *Masks, place: Place) Error!bool {
        const guard = self.guard orelse return false;
        return guard.free(place.at);
    }

    /// The place after `token` at `from`, or null where the constraint forbids the token there.
    fn after(self: *Masks, from: Place, token: i32) Error!?Place {
        if (!from.open) return .{ .open = token == self.gate.?, .at = from.at };
        if (token < 0) return null;
        const taken = try self.constraint.accept(from.at, @intCast(token)) orelse return null;
        return .{ .open = true, .at = taken };
    }

    /// How many of `drafts` a window may run: up to the first that may not come at its row, or that ends the
    /// reply. `rows` then holds the place after each of them; a draft walked before is not walked again.
    fn walk(self: *Masks, drafts: []const i32) Error!usize {
        std.debug.assert(drafts.len < max_rows);
        if (self.set == 0) {
            self.rows[0] = self.newest;
            self.set = 1;
        }
        var kept: usize = 0;
        while (kept < drafts.len) : (kept += 1) {
            if (kept + 1 < self.set and self.walked[kept] == drafts[kept]) continue;
            // another draft than the one walked here: the places after it go
            self.set = kept + 1;
            const place = try self.after(self.rows[kept], drafts[kept]) orelse break;
            if (place.open and self.constraint.ended(place.at)) break;
            self.rows[kept + 1] = place;
            self.walked[kept] = drafts[kept];
            self.forced[kept] = false;
            self.set = kept + 2;
        }
        return kept;
    }

    /// Before a window of the newest token and `drafts`: how many of the drafts it runs. Row 0's place is the
    /// newest token's; each kept draft's row takes the place after it.
    pub fn window(self: *Masks, drafts: []const i32) Error!usize {
        const start = self.now();
        defer self.spent(start);
        const kept = try self.walk(drafts);
        self.set = kept + 1;
        self.stats.pruned += drafts.len - kept;
        return kept;
    }

    /// What is known of the draft after `drafts`, the drafts so far of the tokens after the newest: `Guide.next`.
    /// It depends on the newest token's place and on `drafts`, and on nothing a model computed.
    pub fn next(self: *Masks, drafts: []const i32, more: bool) Error!Next {
        const start = self.now();
        defer self.spent(start);
        if (try self.walk(drafts) < drafts.len) return .refused;
        if (!more or drafts.len == max_rows - 1) return .closed;
        const place = self.rows[drafts.len];
        if (!place.open) return .free;
        if (self.constraint.ended(place.at)) return .closed;
        if (try self.unheld(place)) return .free;
        try self.constraint.fill(place.at, self.mask);
        const token = only(self.mask) orelse return .{ .among = self.mask };
        const after_it = try self.constraint.accept(place.at, token) orelse return error.ConstraintFailed;
        // a token that ends the reply is no draft: nothing is verified after the end
        if (self.constraint.ended(after_it)) return .closed;
        self.rows[drafts.len + 1] = .{ .open = true, .at = after_it };
        self.walked[drafts.len] = @intCast(token);
        self.forced[drafts.len] = true;
        self.set = drafts.len + 2;
        self.stats.forced += 1;
        return .{ .token = @intCast(token) };
    }

    /// How many of `drafts`, from the first, are certain to be kept by the window that runs them: each the one
    /// token that may come at its row, given by `next` since the last `commit`. Their rows can choose nothing
    /// else, sampled or not, so their text may be sent before the window has run.
    pub fn certain(self: *const Masks, drafts: []const i32) usize {
        var n: usize = 0;
        while (n < drafts.len and n + 1 < self.set and self.forced[n] and self.walked[n] == drafts[n]) n += 1;
        return n;
    }

    pub fn guide(self: *Masks) Guide {
        return .{ .context = self, .nextFn = guideNext };
    }

    fn guideNext(context: *anyopaque, drafts: []const i32, more: bool) Error!Next {
        const self: *Masks = @ptrCast(@alignCast(context));
        return self.next(drafts, more);
    }

    /// Row `row`'s logits, bf16 over the vocabulary, with every token that may not come there at -inf.
    pub fn apply(self: *Masks, row: usize, logits: []u16) Error!void {
        std.debug.assert(row < self.set);
        const place = self.rows[row];
        if (!place.open) return;
        const start = self.now();
        defer self.spent(start);
        // nothing may follow the end: the caller ends a reply at its end token
        if (self.constraint.ended(place.at)) return error.ConstraintFailed;
        if (try self.unheld(place)) return;
        try self.constraint.fill(place.at, self.mask);
        if (logits.len > self.mask.len * 32 or logits.len + 31 < self.mask.len * 32) return error.ConstraintFailed;
        write(self.mask, logits);
        self.stats.masked += 1;
    }

    /// The window kept `keep` rows, and the last of them chose `token`: the sequence's newest. The rows before
    /// it chose their drafts, which `window` walked.
    pub fn commit(self: *Masks, keep: usize, token: i32) Error!void {
        std.debug.assert(keep >= 1 and keep <= self.set);
        const start = self.now();
        defer self.spent(start);
        // a token chosen under its row's mask is one the constraint takes
        self.newest = try self.after(self.rows[keep - 1], token) orelse return error.ConstraintFailed;
        if (self.guard) |guard| guard.reach(self.newest.at);
        self.set = 0;
    }

    /// The reply already holds `tokens`, chosen under this constraint by a decoder that is gone (a stream moved
    /// to another decoder, or taken off and admitted again: `server/scheduler.zig`'s `sent`): the place after
    /// them, as a window and a commit a token would leave it. Before the first window.
    pub fn advance(self: *Masks, tokens: []const i32) Error!void {
        std.debug.assert(self.set == 0);
        for (tokens) |token| self.newest = try self.after(self.newest, token) orelse return error.ConstraintFailed;
        if (self.guard) |guard| guard.reach(self.newest.at);
    }

    /// The reply's document is whole and its end token taken.
    pub fn ended(self: *const Masks) bool {
        return self.newest.open and self.constraint.ended(self.newest.at);
    }

    fn now(self: *const Masks) ?std.Io.Timestamp {
        return if (self.io) |io| .now(io, .awake) else null;
    }

    fn spent(self: *Masks, since: ?std.Io.Timestamp) void {
        if (since) |start| self.stats.nanoseconds += @intCast(start.untilNow(self.io.?, .awake).toNanoseconds());
    }
};

/// The token `mask` holds, where it holds exactly one.
pub fn only(mask: []const u32) ?u32 {
    var found: ?u32 = null;
    for (mask, 0..) |word, index| {
        if (word == 0) continue;
        if (found != null or word & (word - 1) != 0) return null;
        found = @intCast(index * 32 + @ctz(word));
    }
    return found;
}

/// `logits[t]` to -inf wherever `mask` lacks token `t`.
pub fn write(mask: []const u32, logits: []u16) void {
    for (mask, 0..) |word, index| {
        if (word == std.math.maxInt(u32)) continue;
        const from = index * 32;
        if (from >= logits.len) break;
        const row = logits[from..@min(from + 32, logits.len)];
        if (word == 0) {
            @memset(row, minus_infinity);
            continue;
        }
        for (row, 0..) |*logit, bit| {
            if (word >> @intCast(bit) & 1 == 0) logit.* = minus_infinity;
        }
    }
}
