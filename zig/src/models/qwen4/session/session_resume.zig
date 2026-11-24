//! Where a session's prompt goes on from, and the states it keeps on the way (spec 104 WS-06 moved them out of
//! `session.zig`): `resume` puts the state where the prompt parts from the sequence before it, from the state,
//! a snapshot or a sequence parked (`parked.zig`); the snapshots planned, retired and kept on a reply's stride.
//! `Session` names each by alias.

const std = @import("std");
const cuda = @import("../../../cuda/cuda.zig");
const draft = @import("../draft.zig");
const forward = @import("../forward.zig");
const parked_file = @import("parked.zig");
const learnt = @import("learnt.zig");
const attention = @import("../blocks/attention.zig");
const sampling = @import("../../../decode/sampling.zig");
const chat_file = @import("chat.zig");
const options_file = @import("session_options.zig");
const Session = @import("session.zig").Session;
const Options = options_file.Options;

/// Puts the state where `prompt` can go on from, and returns how many of `known`'s tokens the prompt
/// begins with. A state at position p serves a prompt that agrees with `known` through token p, not only
/// below it: the head's entry for p - 1 was absorbed with that token. So the state as it stands serves
/// when the prompt holds all of `known`, else the highest snapshot the prompt agrees past, of those `plan`
/// kept and those on a reply's stride alike, else none. A sequence left earlier that the prompt holds the
/// whole of and that ends above all of these is put back instead (`lot`), and the one held is left in its
/// turn where the prompt parts inside its last prompt, `parked_file.worth` tokens or more below its end.
pub fn @"resume"(self: *Session, prompt: []const i32, settings: sampling.Sampling, kept: ?usize) !usize {
    var shared = if (self.trusted) std.mem.indexOfDiff(i32, self.known.items, prompt) orelse @min(self.known.items.len, prompt.len) else 0;
    self.book.from_stride = false;
    defer self.kept = kept;
    if (!self.trusted) self.lot.clear();
    var taken: ?usize = null;
    if (self.trusted and self.lot.most > 0 and self.m.attention != null) {
        const head_cache: ?attention.Cache = if (self.head) |h| h.state.cache else null;
        const whole = shared == self.known.items.len;
        // where the prompt starts from without the lot
        var start: usize = if (whole) self.state.pos else 0;
        if (!whole) for ([_][]const forward.Snapshot{ self.snapshots.items, self.strides.items }) |held| for (held) |one| {
            if (one.pos < shared) start = @max(start, one.pos);
        };
        const stands = if (self.head) |h| !h.full and h.state.position.host - h.state.drafted == self.state.pos else true;
        // Another conversation parts inside the last prompt; the next turn of this one holds that prompt
        // (but for its tail, as Qwen's template writes it) and parts in the reply, whose rows it runs again
        // from the prompt's mark: leaving the sequence at every turn would cost each one a state's copy and
        // fill the lot with ends no prompt asks for.
        const another = shared + self.options.tail < self.asked;
        if (another and stands and self.state.pos >= shared + parked_file.worth) self.leave(shared, head_cache);
        // on a failure the lot is empty, and the prompt starts as it would have
        self.lot.lower(self.state, head_cache, prompt) catch |err| std.log.warn("the sequences left are dropped: {t}", .{err});
        if (self.lot.best(prompt, start)) |at| {
            self.retireFrom(shared);
            self.trusted = false;
            var one = try self.lot.take(at, self.state, head_cache);
            defer self.lot.done(&one);
            taken = parked_file.common(one.tokens, prompt);
            self.known.clearRetainingCapacity();
            self.known.appendSliceAssumeCapacity(one.tokens);
            if (self.head) |*h| {
                try h.state.trim();
                try h.state.position.set(one.to);
                h.full = false;
            }
            self.kept = kept;
            self.adopt(&one, taken.?, prompt.len);
        }
    }
    if (taken) |theirs| shared = theirs else self.retireFrom(shared);
    if (learnt.recalled(self, prompt, kept, shared)) |theirs| shared = theirs;
    if (self.trusted and taken == null and self.state.pos < shared) {
        if (self.head) |*h| try h.rewind(self.state.pos, settings);
    } else if (self.booked().under(std.math.maxInt(usize))) |back| {
        // the newest held, of either kind (`decode/book.zig`)
        self.book.from_stride = back.stride;
        const snapshot = if (back.stride) self.strides.getLast() else self.snapshots.getLast();
        std.debug.assert(snapshot.pos == back.pos);
        try snapshot.restore(&self.state);
        if (self.head) |*h| try h.rewind(snapshot.pos, settings);
    } else {
        try self.state.reset();
        if (self.head) |*h| try h.reset(settings);
    }
    return shared;
}

/// Leaves the sequence held to the lot before a prompt that parts from it at `shared`, with the newest of
/// its states above there; the others are retired as ever.
pub fn leave(self: *Session, shared: usize, head_cache: ?attention.Cache) void {
    const one = self.lot.park(self.state, head_cache, self.m.attention.?.shape.ratio, self.known.items) catch |err| {
        return std.log.warn("a sequence of {d} tokens is not kept: {t}", .{ self.known.items.len, err });
    };
    // the state where the prefix other conversations share ends goes with it, whatever stands above
    var first: ?usize = null;
    if (self.kept) |pos| if (pos >= shared) for (self.snapshots.items, 0..) |held, i| if (held.pos == pos) {
        first = i;
    };
    while (one.states.items.len + @intFromBool(first != null) < parked_file.states) {
        const a: usize = if (self.snapshots.getLastOrNull()) |held| held.pos else 0;
        const b: usize = if (self.strides.getLastOrNull()) |held| held.pos else 0;
        if (@max(a, b) < shared or self.snapshots.items.len + self.strides.items.len == 0) break;
        self.lot.keep(one, if (a >= b and self.snapshots.items.len > 0) self.snapshots.pop().? else self.strides.pop().?);
        if (first != null and first.? >= self.snapshots.items.len) first = null;
    }
    if (first) |i| self.lot.keep(one, self.snapshots.orderedRemove(i));
    // among its states, not only the lowest: a stride's or a cut's below it inside the harness may go too
    if (self.kept) |pos| for (one.states.items) |held| if (held.pos == pos) {
        one.shared = pos;
    };
    one.asked = self.asked;
}

/// The states of a sequence taken up that serve a prompt sharing `shared` tokens with it join the
/// snapshots, within their budget (`thinned`, for a prompt `len` long).
pub fn adopt(self: *Session, one: *parked_file.Sequence, shared: usize, len: usize) void {
    while (one.states.items.len > 0 and one.states.items[0].pos < shared) {
        const state = one.states.orderedRemove(0);
        const held = self.snapshots.items;
        var at: usize = held.len;
        while (at > 0 and held[at - 1].pos >= state.pos) at -= 1;
        if (at < held.len and held[at].pos == state.pos) {
            self.lot.give(state);
            continue;
        }
        self.snapshots.insertAssumeCapacity(at, state);
        if (self.snapshots.items.len > self.options.snapshots) {
            self.lot.give(self.snapshots.orderedRemove(self.booked().over(len)));
        }
    }
    while (self.unused.items.len > 0 and self.snapshots.items.len + self.unused.items.len > self.options.snapshots) {
        self.lot.give(self.unused.pop().?);
    }
}

/// The states at or past `shared`, where a prompt parts from the sequence held, are no state of the prompt's.
pub fn retireFrom(self: *Session, shared: usize) void {
    // their buffers wait for the states this prompt takes
    while (self.snapshots.items.len > 0 and self.snapshots.items[self.snapshots.items.len - 1].pos >= shared) {
        self.retire(self.snapshots.pop().?);
    }
    // those on a reply's stride go too, and their buffers wait for the next reply's
    while (self.strides.items.len > 0 and self.strides.items[self.strides.items.len - 1].pos >= shared) {
        self.spare.appendAssumeCapacity(self.strides.pop().?);
    }
}

/// Which states the prompt about to run from the state's position will keep: `self.book.due`, ascending,
/// each above the state's position and below the prompt's end. The arithmetic is `marks.Planner`'s, which a
/// pool stream shares, by `decode/book.zig`'s `Book.plan`: where the messages start, one token before where
/// the prompt parted, `options.tail` before its end, and where a window that crossed a multiple of
/// `options.stride` ends; past `options.snapshots` one goes, by `thinned`: a snapshot held is dropped here, a
/// candidate is never taken.
///
/// The states on a reply's stride (`strides`) are not among them and take no place of theirs, but for
/// the one the state was restored from where it stands at the part point or the turn's mark (below).
pub fn plan(self: *Session, rounds: draft.Decoder, len: usize, shared: usize, starts: []const usize) void {
    const book = self.booked();
    var planner = book.plan(.{ .pos = self.state.pos, .len = len, .shared = shared, .starts = starts, .kept = self.kept });
    // The state gone on from is one on a reply's stride that stands at one of the prompt's two marks:
    // restored from lower, the prompt would have kept the state there among the snapshots, and it cannot
    // now, having no window over it. So that one joins them, as if taken first: a later reply writes over
    // the strides, and a prompt that parts here again would else start lower than it did before spec 036.
    if (book.joined) {
        // held, then counted as a position planned: one too many, and a lower one is freed below
        self.snapshots.appendAssumeCapacity(self.strides.pop().?);
        // it brings its buffers from the strides' budget, which takes an unused state's back
        if (self.unused.items.len > 0 and self.snapshots.items.len + self.unused.items.len > self.options.snapshots) self.spare.appendAssumeCapacity(self.unused.pop().?);
    }
    while (planner.from < len) book.window(&planner, rounds.promptRows(planner.from, len, self.chosen.len));
    for (book.planned(&planner)) |at| self.retire(self.snapshots.orderedRemove(indexAt(self.snapshots.items, at)));
    std.debug.assert(book.marked == self.snapshots.items.len and planner.count <= self.options.snapshots);
}

/// The index of the state at `pos` among `held`, which holds one there.
fn indexAt(held: []const forward.Snapshot, pos: usize) usize {
    for (held, 0..) |one, i| if (one.pos == pos) return i;
    unreachable;
}

/// A state taken out of `snapshots` that is no longer wanted: its buffers are kept for the next one taken
/// (`unused`) while the two lists are within `Options.snapshots` together, else freed.
pub fn retire(self: *Session, snapshot: forward.Snapshot) void {
    if (self.snapshots.items.len + self.unused.items.len < self.options.snapshots) self.unused.appendAssumeCapacity(snapshot) else self.lot.give(snapshot);
}

/// The buffers of the states the reply about to be decoded takes on its stride, made now where
/// `strides` and `spare` hold fewer than `Options.reply_marks` between them: a state that went gives
/// its own (`lot`), else new ones. Called once the prompt's last window is launched, so the device
/// computes meanwhile; made when a round's state was first due, 73 buffers were 6 to 7 ms of that round
/// with the device idle (spec 089 WS-00). `last` is where the reply's last token would stand: no more
/// are made than it can take, a stride past the newest state each (`Ends.window`; the states the window
/// is about to keep count), so a short reply's session holds none, and a long one's what its strides
/// would have left it.
pub fn readyStrides(self: *Session, last: usize) !void {
    var from = self.newest();
    for (self.ahead.items) |one| from = @max(from, one.pos);
    const most = @min(self.options.reply_marks, (last -| from) / self.options.reply_stride);
    while (self.strides.items.len + self.spare.items.len < most) {
        var blank: forward.Snapshot = self.lot.lend() orelse try .ahead(self.gpa, self.state, 1);
        errdefer blank.deinit();
        try blank.reserve(self.state);
        self.spare.appendAssumeCapacity(blank);
    }
}

/// Launches what the newest snapshots still wait for (`settling`). Before the next window on the state.
pub fn settle(self: *Session) cuda.Error!void {
    if (self.settling == 0) return;
    const held = self.snapshots.items;
    try forward.settle(self.state, held[held.len - self.settling ..], self.settle_from);
    self.settling = 0;
}

/// A reply that ended before `settle` ran: the snapshots that waited hold no state, and go.
pub fn unsettled(self: *Session) void {
    while (self.settling > 0) : (self.settling -= 1) self.retire(self.snapshots.pop().?);
}

/// Frees what a window that did not finish left asked for.
pub fn dropAhead(self: *Session) void {
    self.state.wanted = &.{};
    self.state.stashed = false;
    for (self.ahead.items) |*snapshot| snapshot.deinit();
    self.ahead.clearRetainingCapacity();
}

/// Keeps the state after the first `keep` rows of the window `tokens`, as `keepWithin`, among the states
/// on a reply's stride: with `Options.reply_marks` held the oldest goes, and its buffers take this one.
pub fn keepStride(self: *Session, tokens: []const i32, keep: usize) draft.Error!void {
    std.debug.assert(self.newest() < self.state.pos + keep);
    const full = self.strides.items.len == self.options.reply_marks;
    if (!full and self.spare.items.len + self.lot.spare.count() == 0) {
        self.strides.appendAssumeCapacity(try .within(self.gpa, self.m, self.state, self.buffers, tokens, keep));
        return;
    }
    // into buffers held already: the oldest's with `reply_marks` held (a state that went may wait in the
    // lot then too, and taking its buffers would hold one state more than the list has room for), else
    // a spare one's
    var oldest = if (full) self.strides.orderedRemove(0) else self.spare.pop() orelse self.lot.lend().?;
    oldest.again(self.m, self.state, self.buffers, tokens, keep) catch |err| {
        oldest.deinit();
        return err;
    };
    self.strides.appendAssumeCapacity(oldest);
}

/// Keeps the state after the first `keep` rows of the window `tokens`, which has run from the state's
/// position and is not committed. With the budget full one goes first, by `thinned`: never this one.
pub fn keepWithin(self: *Session, tokens: []const i32, keep: usize) draft.Error!void {
    const at = self.state.pos + keep;
    const held = self.snapshots.items.len;
    std.debug.assert(held == 0 or self.snapshots.items[held - 1].pos < at);
    // into the buffers of the one that goes (`Book.close`), or of one no longer wanted
    var over: ?forward.Snapshot = self.unused.pop();
    if (self.booked().close(at)) |drop| {
        std.debug.assert(over == null);
        over = self.snapshots.orderedRemove(indexAt(self.snapshots.items, drop));
    }
    if (over == null) over = self.lot.lend();
    if (over) |*snapshot| {
        snapshot.again(self.m, self.state, self.buffers, tokens, keep) catch |err| {
            snapshot.deinit();
            return err;
        };
        self.snapshots.appendAssumeCapacity(snapshot.*);
    } else {
        self.snapshots.appendAssumeCapacity(try .within(self.gpa, self.m, self.state, self.buffers, tokens, keep));
    }
}
