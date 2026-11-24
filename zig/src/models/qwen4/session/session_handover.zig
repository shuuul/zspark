//! What a session hands the pool that takes it over when a second stream opens it (`pool/hall_carry.zig`, spec
//! 054 WS-03): one call of the session's, `Session.handover` (spec 103 WS-03 moved it out of the pool's
//! file). `Session` names it by alias.
//!
//! What is lent, to be copied by the pool: the sequence (`known`, the position it stands at), the caches whose
//! rows go into the pool's extents, the head's, the state at the end, and the sequences it left (`parked.Lot`,
//! which stays the session's). What moves: its states, the snapshots of its prompts and those on its reply's
//! stride, each sequence's left, and the buffers it kept unused; each is the taker's once taken. Who frees on
//! an error: a state taken, the taker; one not taken stays the session's and goes with it (a sequence's left
//! as before: `learnt.flush` may still write it to disk). The session is then only to be destroyed.

const std = @import("std");
const cuda = @import("../../../cuda/cuda.zig");
const attention = @import("../blocks/attention.zig");
const forward = @import("../forward.zig");
const parked = @import("parked.zig");
const Session = @import("session.zig").Session;

pub const Handover = struct {
    held: *Session,
    /// The states' own allocator: a state frees its pooled keys with it.
    gpa: std.mem.Allocator,
    /// The position the sequence stands at, its tokens and one past them, its caches and the head's.
    pos: usize,
    known: []const i32,
    caches: []const attention.Cache,
    head: ?attention.Cache,
    /// Positions a pooled key of the shared attention stands for; 0: the model has none.
    ratio: usize,
    /// The state at the end, to be copied: `known[0..pos]` run.
    state: *forward.State,
    /// The sequences the session left, lent: their states are taken with `takeLeft` or given back.
    left: []parked.Sequence,
    /// Which of the two lists `next` reads: the prompts' snapshots, then the reply's strides.
    list: u2 = 0,

    /// The next of the session's own states, still the session's until `take` or `drop`.
    pub const Next = struct {
        pos: usize,
        /// On the reply's stride, not a prompt's snapshot.
        stride: bool,
        /// Where the prefix other conversations share ends (`marks.Request.kept`).
        shared: bool,
    };

    /// How many states the session holds of its own, its end not counted.
    pub fn states(self: *const Handover) usize {
        return self.held.snapshots.items.len + self.held.strides.items.len;
    }

    fn current(self: *Handover) ?*std.ArrayList(forward.Snapshot) {
        while (self.list < 2) : (self.list += 1) {
            const list = if (self.list == 0) &self.held.snapshots else &self.held.strides;
            if (list.items.len > 0) return list;
        }
        return null;
    }

    /// The first state of the snapshots, then of the strides, lowest first; null: none is left.
    pub fn next(self: *Handover) ?Next {
        const list = self.current() orelse return null;
        const at = list.items[0].pos;
        return .{ .pos = at, .stride = self.list == 1, .shared = self.held.kept == at };
    }

    /// The state `next` named, moved out: the caller's.
    pub fn take(self: *Handover) forward.Snapshot {
        return self.current().?.orderedRemove(0);
    }

    /// The state `next` named, freed.
    pub fn drop(self: *Handover) void {
        var gone = self.current().?.orderedRemove(0);
        gone.deinit();
    }

    /// A state's buffers the session kept unused, moved out: those of its snapshots, then of its strides.
    pub fn kept(self: *Handover) ?forward.Snapshot {
        return self.held.unused.pop() orelse self.held.spare.pop();
    }

    /// `kept`, then the lot's: every buffer the session holds no state in, moved out.
    pub fn buffer(self: *Handover) ?forward.Snapshot {
        return self.kept() orelse self.held.lot.lend();
    }

    /// The first state of `one`, a sequence of `left`, moved out: the caller's.
    pub fn takeLeft(_: *Handover, one: *parked.Sequence) forward.Snapshot {
        return one.states.orderedRemove(0);
    }

    /// The first state of `one` given back to the lot, for its buffers.
    pub fn giveLeft(self: *Handover, one: *parked.Sequence) void {
        self.held.lot.give(one.states.orderedRemove(0));
    }
};

/// What the session hands a pool of `capacity` positions over `caches` attention layers, with a head's pool
/// or not; null: nothing, for a session that is not `trusted`, holds no sequence, does not match the pool or
/// whose caches lie behind a table, or whose head stopped absorbing. The head's chained drafts are taken back
/// first, so it stands where the state does.
pub fn handover(self: *Session, capacity: usize, caches: usize, head_pool: bool) cuda.Error!?Handover {
    const pos = self.state.pos;
    if (!self.trusted or pos == 0 or self.known.items.len != pos + 1 or pos > capacity) return null;
    if (self.state.caches.len != caches or (self.head == null) == head_pool) return null;
    for (self.state.caches) |cache| if (cache.table != null) return null;
    if (self.head) |*head| {
        if (head.full or head.state.cache.table != null) return null;
        try head.state.trim();
        if (head.state.position.host != pos) return null;
    }
    return .{
        .held = self,
        .gpa = self.gpa,
        .pos = pos,
        .known = self.known.items,
        .caches = self.state.caches,
        .head = if (self.head) |*h| h.state.cache else null,
        .ratio = if (self.m.attention) |shared| shared.shape.ratio else 0,
        .state = &self.state,
        .left = self.lot.list.items,
    };
}
