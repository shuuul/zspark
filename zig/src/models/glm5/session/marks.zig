//! Where GLM's session keeps its states (spec 112 WS-06) and goes on from them, the rules Flash Next's
//! (`decode/book.zig`'s `Book` over `decode/marks.zig`'s `Planner`, spec 112 WS-04), GLM's own as `Book.Options`
//! (`bookOptions`): a prompt's at its messages' starts, one token before where it parted from the sequence held,
//! two before its end, where a window that crossed a stride ends and where its last window starts, within
//! `Options.snapshots`; a reply's on a grid from the prompt's end and at its end, `snapshot.reply_extra` of them.
//! The Book reads the marks' positions before each question (`booked`). A prompt's mark inside a window is
//! taken there, the window not cut: between the window's forward and its commit, replayed from the committed
//! side into the mark's own buffers (`snapshot.takeWithin`); inside a window that runs KDA chunked it is moved
//! down to its chunk's start first (`snapped`, spec 116 WS-06) and asked before the window runs, which writes
//! the state there into the mark's buffers as it passes (`askAhead`). So a reply's windows, and its bits, are the same whatever it
//! keeps, and only a later prompt resumed from a new mark runs other windows (its rows' forms, spec 115).
//!
//! Rank 0 plans; rank 1 takes and drops what it is told, in order (`Op.keep`, `Op.keep_ahead`, `Op.keep_within`, `Op.drop`), and
//! thins nothing of its own. Every function here is the session's (`self`).

const std = @import("std");
const shared = @import("../../../decode/marks.zig");
const book_file = @import("../../../decode/book.zig");
const held_file = @import("../layers/ranks_forward.zig");
const kda = @import("../layers/kda.zig");
const cuda = @import("../../../cuda/cuda.zig");
const flash_pass = @import("../dflash2/pass.zig");
const snapshot = @import("snapshot.zig");

const Op = held_file.Op;

/// What the `Book` reads of the session's options (`options`, a session's `Options`): the `Planner`'s budget,
/// Flash Next's stride (512, which a prompt's windows of 4096 rows cross where no message starts in them),
/// `near` and `tail` (the generation prompt `<|assistant|><think>` is the two tokens a next turn's prompt
/// shares); a reply's stride `Options.stride`; and GLM's rules: a state at p holds no entry of token p
/// (`through`), a reply's mark at the prompt's position joins the prompt's where it stands at a message's start
/// too, the last window's start, a reply's grid and end.
pub fn bookOptions(options: anytype) book_file.Options {
    return .{
        .marks = .{ .budget = options.snapshots },
        .reply_stride = options.stride,
        .reply_marks = snapshot.reply_extra,
        .through = false,
        .joins_held = true,
        .start_joins = true,
        .last_window = true,
        .grid = true,
    };
}

/// The session's Book with the positions of the marks held, a reply's apart.
pub fn booked(self: anytype) !*book_file.Book {
    var replies: usize = 0;
    for (self.marks.held.items) |mark| replies += @intFromBool(mark.reply);
    const book = &self.book;
    try book.reserve(self.gpa, self.marks.held.items.len - replies, replies);
    book.clear();
    for (self.marks.held.items) |mark| book.put(mark.pos, mark.reply);
    return book;
}

/// The mark a prompt `len` tokens long that shares `same` with the sequence held goes back to: the latest
/// strictly before its end, at or before `same` (`Book.reach`, GLM's `through`). Null where none is held.
pub fn reached(self: anytype, len: usize, same: usize) !?usize {
    const book = try booked(self);
    const back = book.under(book.reach(same, len)) orelse return null;
    return back.pos;
}

/// The positions the prompt about to run from the state's position keeps, ascending, each above it and below
/// the prompt's end, which the caller frees; the prompt's marks held that go for them are dropped here. The
/// windows are `decoder`'s `promptRows` with `room` tokens to choose into, so the ones planned are the ones
/// that run. A reply's mark at the state's position that stands where the prompt parted or two before its end
/// joins the prompt's, as Flash Next's state on a reply's stride does (`session_resume.zig`'s `plan`), and so
/// does one at a message's start: a reply's end, where the next turn's message starts (the reply's end token,
/// `<|user|>`, is not run), which an edit of that message goes on from. Flash Next's keeps no state there.
/// A mark read from disk joins those held past the budget (`learnt.zig`): the Book makes room for it.
pub fn plan(self: anytype, decoder: anytype, room: usize, request: shared.Request) ![]usize {
    std.debug.assert(request.pos == self.state.pos);
    for (self.marks.held.items) |mark| std.debug.assert(mark.pos <= request.pos);
    const book = try booked(self);
    var planner = book.plan(request);
    if (book.joined) self.marks.held.items[self.marks.find(request.pos).?].reply = false;
    // the last window's start, where it is not the state's position (`Book.window`): the committed state, which
    // an edit of the prompt's last tokens goes on from in one window (its 16 rows, a graph's), where the
    // message's start would run two
    while (planner.from < request.len) book.window(&planner, decoder.promptRows(planner.from, request.len, room));
    for (book.planned(&planner)) |at| try drop(self, self.marks.find(at).?);
    return snapped(self.gpa, book.due, decoder, request, room);
}

/// `due` with each mark inside a window that runs KDA chunked moved down to its chunk's start, or the window's
/// (spec 116 WS-06; the coordinator's choice over carrying a chunk's rows, 88 MB a mark): the state there is
/// one the window carries, which it writes for the mark as it runs (`askAhead`), and a prompt resumed from it
/// runs the rest of that chunk again, the same bits. Two that meet are one, and one moved to the request's
/// start is dropped.
fn snapped(gpa: std.mem.Allocator, due: []const usize, decoder: anytype, request: shared.Request, room: usize) ![]usize {
    const out = try gpa.alloc(usize, due.len);
    var n: usize = 0;
    var from = request.pos;
    var next: usize = 0;
    while (next < due.len) {
        std.debug.assert(from < request.len);
        const rows = decoder.promptRows(from, request.len, room);
        while (next < due.len and due[next] < from + rows) : (next += 1) {
            const at = if (kda.chunks(rows)) @max(from, std.mem.alignBackward(usize, due[next], kda.chunk)) else due[next];
            // one moved down to the request's start is the state it starts from: none to take
            if (at == request.pos or (n > 0 and out[n - 1] == at)) continue;
            out[n] = at;
            n += 1;
        }
        from += rows;
    }
    return gpa.realloc(out, n);
}

/// The window of `rows` rows is about to run from the state's position: where it runs KDA chunked, the marks of
/// `planned` from `next` that stand inside it are asked of it, told to the rank that follows first, so that it
/// writes their states as it passes them (`kda.State.ask`); `takeIn` takes them once it has run.
pub fn askAhead(self: anytype, planned: []const usize, next: usize, rows: usize) !void {
    const start = self.state.pos;
    const chains = self.state.ranks[0].kda orelse return;
    if (!chains.chunkedFor(rows)) return;
    var words: [kda.most_asked]u32 = undefined;
    var n: usize = 0;
    for (planned[next..]) |at| {
        if (at >= start + rows) break;
        if (at == start) continue;
        words[n] = @intCast(at - start);
        n += 1;
    }
    if (n == 0) return;
    if (self.leads()) {
        var told: [1 + kda.most_asked]u32 = undefined;
        told[0] = @intCast(n);
        @memcpy(told[1..][0..n], words[0..n]);
        try self.lead.tell(@backingInt(Op.keep_ahead), told[0 .. 1 + n], &.{});
    }
    try askAheadLocal(self, words[0..n]);
}

/// The marks `rows` into the window about to run, each a blank mark (`blankMark`) whose KDA buffers every rank
/// held asks its state to write into; what an earlier window that did not finish left asked goes first.
pub fn askAheadLocal(self: anytype, rows: []align(1) const u32) !void {
    dropAhead(self);
    for (rows) |one| {
        var mark = try blankMark(self, self.state.pos + one);
        self.ahead.append(self.gpa, mark) catch |err| {
            mark.release(self.gpa, &self.marks.spare);
            return err;
        };
        for (&self.state.ranks, mark.ranks) |*rank, shot| if (rank.kda) |*chains| chains.ask(one, shot.kda.?);
    }
}

/// The marks asked ahead and not taken go, their buffers to the spares, and no state is asked of a window.
pub fn dropAhead(self: anytype) void {
    for (self.ahead.items) |*mark| mark.release(self.gpa, &self.marks.spare);
    self.ahead.clearRetainingCapacity();
    for (&self.state.ranks) |*rank| if (rank.kda) |*chains| {
        chains.asked_n = 0;
    };
}

/// The window of `rows` rows has run from the state's position and is not committed: the marks of `planned`
/// from `next.*` that stand in it are taken, the one at its first row as the committed state, the others by
/// replay.
pub fn takeIn(self: anytype, planned: []const usize, next: *usize, rows: usize) !void {
    const start = self.state.pos;
    while (next.* < planned.len and planned[next.*] < start + rows) : (next.* += 1) {
        const at = planned[next.*];
        std.debug.assert(at >= start);
        if (at == start) try keep(self, false) else try keepWithin(self, at - start);
    }
}

/// A mark of the committed state, told to the rank that follows: a reply's (`reply`) past `snapshot.reply_extra`
/// of them takes the oldest one's place (`Book.reply`). One held at this position is taken again. The DFlash2
/// drafter's context goes with it, brought to the mark first where the drafter has yet to absorb the last
/// round's rows (`flashTo`).
pub fn keep(self: anytype, reply: bool) !void {
    if (self.options.snapshots == 0) return;
    if (reply) if ((try booked(self)).reply(self.state.pos)) |gone| try drop(self, self.marks.find(gone).?);
    const end = try self.flashTo(self.state.pos, null);
    if (self.leads()) try self.lead.tell(@backingInt(Op.keep), &.{}, &.{});
    try keepLocal(self, reply);
    try self.flashBack(end);
}

pub fn keepLocal(self: anytype, reply: bool) !void {
    var mark = try self.takeMark();
    mark.reply = reply;
    if (self.marks.find(mark.pos)) |at| return self.marks.replace(at, mark);
    std.debug.assert(self.marks.above(mark.pos));
    self.marks.held.append(self.gpa, mark) catch |err| {
        mark.release(self.gpa, &self.marks.spare);
        return err;
    };
}

/// A prompt's mark after the first `rows` rows of the window that has run and is not committed, told to the
/// rank that follows between the window and its commit.
pub fn keepWithin(self: anytype, rows: usize) !void {
    const end = try self.flashTo(self.state.pos + rows, self.state.pos);
    if (self.leads()) try self.lead.tell(@backingInt(Op.keep_within), &.{@intCast(rows)}, &.{});
    try keepWithinLocal(self, rows);
    try self.flashBack(end);
}

/// The MTP layer's cache is cut at the mark, whose entries the window's absorb writes; the DFlash2 drafter's
/// context is taken over the window's first rows, which `keepWithin` brought it to (`flashAt`).
pub fn keepWithinLocal(self: anytype, rows: usize) !void {
    const pos = self.state.pos + rows;
    // one asked ahead holds its KDA states already (`askAhead`)
    var mark = for (self.ahead.items, 0..) |one, at| {
        if (one.pos == pos) break self.ahead.orderedRemove(at);
    } else try blankMark(self, pos);
    for (&mark.ranks, self.state.ranks) |*shot, rank| {
        const spare = shot.kda;
        shot.kda = null;
        // a spare that fails is freed by `takeWithin`
        shot.* = snapshot.takeWithin(rank, rows, spare) catch |err| {
            mark.release(self.gpa, &self.marks.spare);
            return err;
        };
    }
    mark.flash = flashAt(self, pos) catch |err| {
        mark.release(self.gpa, &self.marks.spare);
        return err;
    };
    std.debug.assert(self.marks.above(pos));
    self.marks.held.append(self.gpa, mark) catch |err| {
        mark.release(self.gpa, &self.marks.spare);
        return err;
    };
}

/// The mark at index `at` goes, here and on the rank that follows.
pub fn drop(self: anytype, at: usize) !void {
    const pos = self.marks.held.items[at].pos;
    if (self.leads()) try self.lead.tell(@backingInt(Op.drop), &.{@intCast(pos)}, &.{});
    self.marks.drop(at);
}

/// Rank 1: the mark at `pos` goes.
pub fn dropLocal(self: anytype, pos: usize) !void {
    self.marks.drop(self.marks.find(pos) orelse return error.NoSnapshot);
}

/// A mark at `pos` with no rows of its own yet: KDA buffers from `spare` or new, no drafter's context.
pub fn blankMark(self: anytype, pos: usize) !@TypeOf(self.*).Mark {
    var mark: @TypeOf(self.*).Mark = .{ .pos = pos, .ranks = undefined, .mtp = pos };
    var done: usize = 0;
    errdefer for (mark.ranks[0..done]) |*shot| shot.deinit();
    for (&mark.ranks, self.state.ranks) |*shot, rank| {
        shot.* = .{ .pos = pos };
        if (rank.kda) |chains| shot.kda = self.marks.lend() orelse try .init(chains.layers, chains.heads);
        done += 1;
    }
    return mark;
}

/// The marks above `pos` dropped, and the prompt's end past it.
pub fn dropAbove(self: anytype, pos: usize) void {
    self.marks.dropAbove(pos);
    if (self.prompt_end) |at| if (at > pos) {
        self.prompt_end = null;
    };
}

pub fn clearMarks(self: anytype) void {
    dropAhead(self);
    self.marks.clear();
    self.prompt_end = null;
}

/// A mark of the committed state as it stands, the drafter's context with it (`flashAt`).
pub fn takeMark(self: anytype) !@TypeOf(self.*).Mark {
    var mark: @TypeOf(self.*).Mark = .{ .pos = self.state.pos, .ranks = undefined, .mtp = self.state.pos };
    var done: usize = 0;
    errdefer for (mark.ranks[0..done]) |*shot| shot.deinit();
    for (&mark.ranks, self.state.ranks) |*shot, rank| {
        shot.* = try snapshot.takeRank(rank, self.marks.lend());
        done += 1;
    }
    mark.flash = try flashAt(self, mark.pos);
    return mark;
}

/// The DFlash2 drafter's context for a mark at `pos`, where it stands there: a mark holds it only so,
/// and a restore leaves the context where the state stands, as the drafter's next absorb asserts.
pub fn flashAt(self: anytype, pos: usize) !?flash_pass.KeptOf(@TypeOf(self.*).ranks_held) {
    if (self.options.drafter == null or pos == 0 or self.flash.pair.context_end != pos) return null;
    return try flash_pass.KeptOf(@TypeOf(self.*).ranks_held).take(&self.flash.pair);
}

/// Brings the DFlash2 drafter's context to `pos` for a mark there, on rank 0 before the mark is told:
/// where the drafter absorbs every window of this reply, the main model's last window ran from the
/// context's end and its taps still stand (a prompt's window not committed, a mark inside it; a
/// round's kept rows, a mark on the reply's grid), so the drafter has yet to absorb that window's first
/// rows. They are absorbed here and on the rank that follows, the mark is taken over them, and
/// `flashBack` puts the context's end back: the drafter's own absorb writes the same rows again (a row
/// is its taps' and its position's alone, `../dflash2/pass.zig`) before any block reads the context.
/// `from` is where that window ran from where the caller knows it. Returns the end to go back to, null
/// where nothing was absorbed.
pub fn flashTo(self: anytype, pos: usize, from: ?usize) !?usize {
    if (self.options.drafter == null or !self.in_step or self.flash.d.full) return null;
    const pair = &self.flash.pair;
    const absorbed = pair.context_end;
    if (absorbed >= pos or (from != null and from.? != absorbed)) return null;
    std.debug.assert(pos - absorbed <= self.b.rows);
    var taps: [@TypeOf(self.*).ranks_held]cuda.Buffer = undefined;
    for (&taps, self.b.ranks) |*one, rank| one.* = rank.taps.?;
    try pair.addTaps(taps, 0, pos - absorbed);
    return absorbed;
}

/// `flashTo`'s end again, told to the rank that follows.
pub fn flashBack(self: anytype, absorbed: ?usize) !void {
    if (absorbed) |at| try self.flash.pair.setEnd(at);
}

pub fn restoreLocal(self: anytype, pos: usize) !void {
    const mark = self.marks.held.items[self.marks.find(pos) orelse return error.NoSnapshot];
    dropAbove(self, pos);
    for (mark.ranks, &self.state.ranks) |shot, *rank| try snapshot.restoreRank(shot, rank);
    self.state.pos = pos;
    std.log.info("rank {d} resumed at {d}", .{ self.first, pos });
    for (&self.pair.caches) |*cache| try cache.cut(mark.mtp);
    self.pair.drafted = 0;
    self.head.full = false;
    if (self.options.drafter != null) {
        // a mark's context stands at the mark (`flashAt`); one without it starts empty there, as a file's
        if (mark.flash) |kept| {
            std.debug.assert(kept.end == pos);
            try kept.restore(&self.flash.pair);
        } else try self.flash.pair.startAt(pos);
        self.flash.d.full = false;
    }
}
