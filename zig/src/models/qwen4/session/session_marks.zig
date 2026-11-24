//! `Marks` and `Ends`: what a prompt's and a reply's windows keep of the states inside them, the session's
//! side of `draft.Decoder`'s Within and Kept and of the buffers' asks, which `session.zig` names as its own.

const std = @import("std");
const draft = @import("../draft.zig");
const forward = @import("../forward.zig");
const session_file = @import("session.zig");
const Session = session_file.Session;

/// What a prompt's windows keep of the states inside them.
pub const Marks = struct {
    session: *Session,
    /// Positions to keep the state at, ascending (`plan`): each is taken inside the window that runs over it.
    at: []const usize = &.{},
    /// How many of them the windows run so far have kept.
    taken: usize = 0,
    windows: usize = 0,
    /// The prompt's length: the states inside its last window wait for `Session.settle`.
    end: usize = 0,
    /// How many did.
    settled: usize = 0,
    /// The most tokens of the reply, where it takes states on its stride (their buffers are made behind
    /// the prompt's last window); 0: it takes none.
    reply: usize = 0,
    /// The prompt and what plans its windows (`Decoder.promptRows`, with `room`): the window after the one
    /// about to run is said to the buffers, which gather its table rows while this one runs.
    prompt: []const i32 = &.{},
    rounds: ?*const draft.Decoder = null,
    room: usize = 0,

    /// `draft.Decoder.Within`: the window `run` is about to run from the state's position. The states
    /// to keep inside it are listed, each over the buffers of one no longer wanted where there is one,
    /// and a window wider than the rows each layer keeps is told of them.
    pub fn before(context: *anyopaque, run: []const i32) draft.Error!void {
        const marks: *Marks = @ptrCast(@alignCast(context));
        const self = marks.session;
        self.dropAhead();
        const start = self.state.pos;
        const stop = start + run.len;
        if (marks.rounds) |rounds| if (stop < marks.prompt.len) {
            forward.expect(&self.buffers, marks.prompt[stop..][0..rounds.promptRows(stop, marks.prompt.len, marks.room)]);
        };
        for (marks.at) |at| {
            if (at <= start or at > stop) continue;
            if (self.unused.pop() orelse self.lot.lend()) |held| {
                var snapshot = held;
                snapshot.aheadOver(self.state, at - start);
                self.ahead.appendAssumeCapacity(snapshot);
            } else {
                var snapshot: forward.Snapshot = try .ahead(self.gpa, self.state, at - start);
                errdefer snapshot.deinit();
                self.ahead.appendAssumeCapacity(snapshot);
            }
        }
        if (forward.widerThanKept(self.state, self.buffers, run.len)) self.state.wanted = self.ahead.items;
        // the last window's: their replays after the reply's first token, not before it
        self.state.stashed = stop == marks.end and self.ahead.items.len > 0 and self.options.settle and forward.defers(self.state, self.buffers, run.len);
    }

    /// `draft.Decoder.Within`: the window `run` is launched. A wide one allocates what it keeps layer by
    /// layer as it is launched; a narrow one's states are taken once it has run (`window`), and their
    /// buffers are allocated here, while the device computes, where after it they were 6 to 9 ms a
    /// state that nothing overlapped (spec 036's figure for a state with new buffers).
    pub fn launched(context: *anyopaque, run: []const i32) draft.Error!void {
        const marks: *Marks = @ptrCast(@alignCast(context));
        const self = marks.session;
        if (marks.reply > 0 and self.state.pos + run.len == marks.end) try self.readyStrides(marks.end - 1 + marks.reply);
        if (forward.widerThanKept(self.state, self.buffers, run.len)) return;
        for (self.ahead.items) |snapshot| try snapshot.reserve(self.state);
    }

    /// `draft.Decoder.Within`: the window `run` has run from the state's position and is not committed.
    pub fn window(context: *anyopaque, run: []const i32, last: bool) draft.Error!void {
        const marks: *Marks = @ptrCast(@alignCast(context));
        const self = marks.session;
        _ = last;
        marks.windows += 1;
        const start = self.state.pos;
        if (forward.widerThanKept(self.state, self.buffers, run.len)) {
            // asked before the window ran, which wrote the DeltaNet's part of each
            self.state.wanted = &.{};
            for (self.ahead.items) |*snapshot| try snapshot.finish(self.state, self.buffers, run);
        } else {
            // each layer kept the window: the chain replayed to the row, into the buffers `launched` made
            for (self.ahead.items) |*snapshot| try snapshot.again(self.m, self.state, self.buffers, run, snapshot.pos - start);
        }
        for (self.ahead.items) |snapshot| {
            std.debug.assert(self.snapshots.items.len == 0 or self.snapshots.items[self.snapshots.items.len - 1].pos < snapshot.pos);
            self.snapshots.appendAssumeCapacity(snapshot);
            marks.taken += 1;
        }
        if (self.state.stashed) {
            self.state.stashed = false;
            self.settling = self.ahead.items.len;
            marks.settled += self.settling;
            self.settle_from = start;
        }
        self.ahead.clearRetainingCapacity();
    }
};

/// What a reply's windows keep of the states inside them (spec 035): the state after the token that closes
/// the reasoning. Whatever a later prompt writes otherwise in the reply's content (a stop string's text
/// gone, blanks trimmed where the length cut it, a call's value in the template's spelling, other tokens
/// for the same text, a reply the client hung up on), that prompt runs the content again and not the
/// reasoning before it, which is most of a reply that thinks. A prompt that leaves the reasoning out
/// parts before it, and is served by the marks of `plan`.
///
/// And the states on the reply's stride (spec 036): one each time the rows a window keeps reach
/// `Options.reply_stride` tokens past the newest state held, at that token, so that a prompt which parts
/// late in a long content, or in a long reasoning sent back cut, runs a stride of it and not all.
pub const Ends = struct {
    session: *Session,
    /// The think block's closing tag while the reply reasons and the state after it is wanted.
    close: ?i32,
    /// Tokens between a state and the next on the reply's stride; 0: none are taken.
    stride: usize,
    taken: usize = 0,

    /// `draft.Decoder.Kept`: the window `tokens` has run, and its first `keep` rows will be committed.
    pub fn window(context: *anyopaque, tokens: []const i32, keep: usize) draft.Error!void {
        const ends: *Ends = @ptrCast(@alignCast(context));
        const self = ends.session;
        if (ends.close) |close| if (std.mem.indexOfScalar(i32, tokens[0..keep], close)) |row| {
            ends.close = null;
            try self.keepWithin(tokens, row + 1);
            ends.taken += 1;
        };
        // After the reasoning's end, which is then the newest: the stride is counted from it.
        if (ends.stride == 0) return;
        const one = self.booked().stride(self.state.pos, keep) orelse return;
        try self.keepStride(tokens, one.at - self.state.pos);
        ends.taken += 1;
    }
};
