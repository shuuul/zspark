//! Drafts copied from the context (spec 060): where a reply repeats what the prompt or the reply itself already
//! holds (a file edited, a call that echoes its argument), what followed the earlier occurrence is offered as
//! the window's drafts in place of the model's drafter's. A draft chooses nothing (`draft.zig`), so a copy that
//! is wrong costs its rows and no more.
//!
//! The search is TensorFold 1.0.2's `SuffixLookup` (`zig/src/core/lanes/proposer.zig`, the build's `lanes`
//! module): the longest earlier match of the context's tail, six tokens at least, and silence for sixteen
//! rounds after four offers of which no row was kept. zspark's own, and a departure from TensorFold, which
//! prices a copied row by a decayed hit rate: a copy is offered only where its first token is the inner
//! drafter's first draft. The inner drafter is asked every round as before, so a reply that repeats nothing
//! is drafted as it was (spec 060 WS-01: without the check a plain reply was modelled up to 7% slower).
//! A reply held to a constraint is not offered copies: the inner drafter drafts what the constraint allows.
//! Where `serve --recall` keeps a record of earlier requests (`recall.zig`, spec 106), a window the context
//! finds nothing for asks the record, and its copy is offered by the same check.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");
const draft = @import("draft.zig");
const masks_file = @import("masks.zig");
const lanes = @import("lanes");
const recall = @import("recall.zig");

/// A position whose token is not known (a state held from an earlier request, the prompt's first token before
/// `begin` was told it): no token's id, so nothing matches across it.
const unknown: u32 = 0xFFFF_FFFF;

/// What a lookup is made with, and whether a copy must begin as the inner drafter's drafts do. Changed by tests
/// alone, which set them back: a random model repeats no six tokens and its head agrees with no copy, so a
/// test that sets a session beside the pool lowers the match to a token and takes whatever is found.
pub var params: @FieldType(lanes.SuffixLookup, "params") = .{};
pub var agreeing: bool = true;

pub const Stats = struct {
    /// For a request's line in the log, after its drafts kept; nothing where no row was copied.
    pub fn line(self: Stats, buffer: []u8) []const u8 {
        if (self.rows == 0) return "";
        if (self.recalled_rows == 0) return std.fmt.bufPrint(buffer, ", {d} of {d} copied", .{ self.kept, self.rows }) catch "";
        return std.fmt.bufPrint(buffer, ", {d} of {d} copied, {d} of {d} recalled", .{ self.kept, self.rows, self.recalled_kept, self.recalled_rows }) catch "";
    }

    /// Rounds in which the lookup found a continuation, and those of them the inner drafter's first draft
    /// agreed with, whose window was the copy.
    found: usize = 0,
    offered: usize = 0,
    /// Copied rows those windows carried, and how many were the main model's choice.
    rows: usize = 0,
    kept: usize = 0,
    /// Of those, the windows, rows and kept rows that were copied from Recall's record.
    recalled: usize = 0,
    recalled_rows: usize = 0,
    recalled_kept: usize = 0,
};

pub const Copied = struct {
    gpa: Allocator,
    /// The model's drafter; null where the caller drafts itself and calls `after` and `offer`.
    inner: ?draft.Drafter,
    lookup: lanes.SuffixLookup,
    /// The sequence so far, a token a position: the prompt, then the reply. `context.items.len - 1` is the
    /// position of the newest token after a call to `draft`.
    context: std.ArrayList(u32) = .empty,
    /// Tokens that end a reply: a copy is cut after the first, as the head ends its chain there.
    stops: []const u32,
    /// Copied rows the last window was given, to be judged when the next call says how many rows were kept.
    pending: usize = 0,
    /// What the lookup found for the next window, asked ahead of the drafter (`ahead`), which `offer` takes.
    found: ?[]const u32 = null,
    stats: Stats = .{},
    /// Recall's record, set before `begin` where `serve --recall` keeps one; the reply's reader of it, made by
    /// `begin`: asked where the context's lookup finds nothing.
    store: ?*recall.Store = null,
    reader: ?recall.Reader = null,
    /// `found` is the record's, and so were the copied rows `pending` counts.
    found_recalled: bool = false,
    recalled: bool = false,
    /// Windows judged in this reply, the reader's clock; and the prompt's length, where the reply begins.
    rounds: u64 = 0,
    prompt_len: usize = 0,

    pub fn init(gpa: Allocator, inner: ?draft.Drafter, stops: []const u32) !Copied {
        return .{ .gpa = gpa, .inner = inner, .lookup = try .init(gpa, params), .stops = stops };
    }

    pub fn deinit(self: *Copied) void {
        if (self.reader) |*reader| reader.deinit();
        self.lookup.deinit();
        self.context.deinit(self.gpa);
        self.* = undefined;
    }

    /// A reply starts over `prompt`, whole: also the part a held state already ran, which no call to `draft`
    /// will show. The lookup keeps its index where the prompt goes on from all it has indexed (the next turn of
    /// a conversation) and starts again otherwise, so nothing of another sequence is offered.
    pub fn begin(self: *Copied, prompt: []const i32) !void {
        const indexed = self.lookup.indexed;
        const same = indexed <= prompt.len and indexed <= self.context.items.len and
            std.mem.eql(u32, self.context.items[0..indexed], @ptrCast(prompt[0..indexed]));
        if (!same) {
            self.lookup.deinit();
            self.lookup = try .init(self.gpa, params);
        } else {
            // only the silence and the judged rounds are the last reply's
            self.lookup.silent_for = 0;
            self.lookup.recent.clearRetainingCapacity();
        }
        self.context.clearRetainingCapacity();
        try self.context.ensureTotalCapacity(self.gpa, prompt.len);
        for (prompt) |token| self.context.appendAssumeCapacity(@bitCast(token));
        self.pending = 0;
        self.found = null;
        self.stats = .{};
        self.found_recalled = false;
        self.recalled = false;
        self.rounds = 0;
        self.prompt_len = prompt.len;
        if (self.reader) |*reader| reader.deinit();
        self.reader = null;
        if (self.store) |store| {
            store.readPrompt(@ptrCast(prompt), 0);
            self.reader = .{ .gpa = self.gpa, .shared = &store.shared };
        }
    }

    /// The reply ended (not failed): Recall's record, where there is one, takes it after its prompt's tail.
    pub fn end(self: *const Copied) void {
        const store = self.store orelse return;
        const context = self.context.items;
        if (context.len <= self.prompt_len) return;
        store.readReply(context[0..self.prompt_len], context[self.prompt_len..]);
    }

    pub fn drafter(self: *Copied) draft.Drafter {
        return .{ .context = self, .draft = draftFn };
    }

    /// `tokens` end at `position`: the context is made to end there with them, whatever it held past or short
    /// of their first position.
    fn follow(self: *Copied, tokens: []const i32, position: usize) !void {
        const from = position + 1 -| tokens.len;
        if (self.context.items.len > from) self.context.shrinkRetainingCapacity(from);
        try self.context.ensureTotalCapacity(self.gpa, position + 1);
        while (self.context.items.len < from) self.context.appendAssumeCapacity(unknown);
        for (tokens[tokens.len - (position + 1 - from) ..]) |token| self.context.appendAssumeCapacity(@bitCast(token));
    }

    /// A window was verified: `tokens` are what was chosen after its kept rows, the last at `position`. The
    /// copy it carried, if it did, is judged, and the context goes on to `position`. For a caller that drafts
    /// without a `Drafter` (a pool's round), before its `offer`.
    pub fn after(self: *Copied, tokens: []const i32, position: usize) !void {
        // a prompt's windows (`Decoder.advance`) are no round of the reply's: they set no bar
        const decoding = position >= self.prompt_len;
        if (decoding) self.rounds += 1;
        if (self.pending > 0) {
            // the window was the last token and the copied rows; a kept row past the first is a kept copy
            const kept = @min(tokens.len -| 1, self.pending);
            self.stats.kept += kept;
            if (self.recalled) {
                // a recalled copy is judged by the record's rates alone, not the context's silence
                self.stats.recalled_kept += kept;
                if (self.reader) |*reader| reader.observe(@intCast(self.pending), @intCast(kept), self.rounds);
            } else self.lookup.observe(@intCast(self.pending), @intCast(kept)) catch {};
            self.pending = 0;
            self.recalled = false;
        } else if (decoding) if (self.reader) |*reader| reader.noteHead(tokens.len); // the bar a recalled copy must beat

        try self.follow(tokens, position);
    }

    /// `out[0..drafted]` are the inner drafter's drafts for the window after the context's last token, one at
    /// least: the count of drafts `out` holds on return, the copy's where one is offered, else those.
    pub fn offer(self: *Copied, drafted: usize, out: []i32) !usize {
        const found = self.found orelse try self.find(out.len);
        self.found = null;
        if (found.len == 0) return drafted;
        self.stats.found += 1;
        const rows = copyOf(found, self.stops, out) orelse return drafted;
        self.stats.offered += 1;
        self.stats.rows += rows;
        self.pending = rows;
        self.recalled = self.found_recalled;
        if (self.recalled) {
            self.stats.recalled += 1;
            self.stats.recalled_rows += rows;
        }
        return rows;
    }

    /// The continuation the next window may copy, `most` tokens at most: the context's, else, where it found
    /// none, Recall's where its copy is expected to pay (TensorFold 1.0.12's `copyProposal`, the context first).
    fn find(self: *Copied, most: usize) ![]const u32 {
        self.found_recalled = false;
        const own = try self.lookup.propose(self.context.items, @intCast(most));
        if (own.len > 0) return own;
        const reader = if (self.reader) |*held| held else return own;
        const got = reader.propose(self.context.items, most, self.rounds);
        if (got.len < 2) return own;
        self.found_recalled = true;
        return got;
    }

    /// What `offer` does with what the lookup `found`, one token at least, where the inner drafter's drafts are
    /// in `out`: null, the drafts stand; else the copy's rows, written over them. Whoever knows what a lookup
    /// found and drafts as the inner drafter does offers the same copy (Flash Next's rank 1, spec 095).
    pub fn copyOf(found: []const u32, stops: []const u32, out: []i32) ?usize {
        if (agreeing and @as(i32, @bitCast(found[0])) != out[0]) return null;
        var rows: usize = 0;
        while (rows < found.len) {
            const token: i32 = @bitCast(found[rows]);
            if (found[rows] == unknown) break;
            out[rows] = token;
            rows += 1;
            if (std.mem.indexOfScalar(u32, stops, found[rows - 1]) != null) break;
        }
        // the copy alone is the window: past its first token the drafter's chain drafts another text
        return rows;
    }

    /// The lookup for the window after `tokens`, the last at `position`, asked now and not when the drafter
    /// has drafted: `draft` is then called with these and `len` drafts at most, and the inner drafter drafts
    /// one at least, without a constraint. The lookup is asked once a window as `draft` asks it, so the copies
    /// are those it would offer; what it found is returned, valid until the lookup is asked again.
    pub fn ahead(self: *Copied, tokens: []const i32, position: usize, len: usize) ![]const u32 {
        std.debug.assert(len > 0 and self.found == null);
        try self.after(tokens, position);
        const found = try self.find(len);
        self.found = found;
        return found;
    }

    fn draftFn(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Copied = @ptrCast(@alignCast(context));
        try self.after(tokens, position);
        const inner = self.inner.?;
        const drafted = try inner.draft(inner.context, streams, tokens, position, guide, out);
        // under a constraint the inner drafter drafts within it and a copy would not: the window is its own
        if (drafted == 0 or guide != null) {
            // `ahead` is asked only where neither holds
            std.debug.assert(self.found == null);
            return drafted;
        }
        return self.offer(drafted, out);
    }
};
