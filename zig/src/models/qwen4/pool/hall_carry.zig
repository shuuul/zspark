//! What a lone stream's session hands the pool when a second stream opens it (spec 054, WS-03): the sequence
//! it holds becomes an idle chain of the pool, its cache rows copied into the chain's extents, and its states
//! become states of that chain, which the stream that was live and every later request fork from as from any
//! other (`hall_keep.usable`). So the live stream goes on from its row and not from its prompt's start, and
//! the turn after it finds what the session held.
//!
//! A session holds one sequence: `known`, the state at its end, the snapshots of its prompts and the states
//! on its reply's stride. A snapshot and a pool's state hold the same things (`forward.Snapshot`: one
//! recurrent side, the conv inputs, the PLE's tails and histories), so each is moved, not copied; the pooled
//! keys of the blocks below it are copied from the session's caches, where a complete block's key is what a
//! slot's is. The state at the end is copied into buffers the session kept unused where it has some, and is
//! a reply's end to the store. The session does not record what a snapshot was kept for: each is a `.window`
//! to the store, one of the chain's `prefix.prompt_marks`, which is the budget they shared in the session.
//!
//! The sequences the session left (`parked.Lot`, spec 064) come too, each an idle chain of its own with its
//! states (`parked`): a chain forked from the carried one at the extent below where the sequence parted, so
//! the extents below are named twice and not copied, then the rows above from the session's caches (below
//! where the sequence's own rows are held, the caches hold its rows still: `parked.zig`'s header) and from
//! the rows the lot held. Each is a sequence the session would have taken up again, so a conversation in
//! turn goes on from its states in the pool as in the session. Its chain is older than the carried one (by
//! the lot's order), as the session left it before: the store lets its states go first. A state below the
//! prompt whose reply ends it is a `.window`, one above a `.stride`, its end the `.end`: the lot knows which
//! (`Sequence.asked`), so the pool's disk writes it as the session would (`spill.pick`). One the extents do
//! not hold is not carried, and goes as before (to disk under `--spill-gib`).
//!
//! The rows are copied between `Engine.openAfter`'s extents and the rest of the pool, and the session is
//! freed there: the device holds the two caches for the copy alone, under what the pool holds once open.
//! What the session lends and what it hands over, and who frees what on an error, is its own call,
//! `Session.handover` (`session/session_handover.zig`): this file reads no field of the session's. Nothing is
//! carried from a session that is not `trusted`, whose head stopped absorbing, or whose caches lie behind a
//! table: the stream then runs from its prompt's start, as every one did before.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const attention = @import("../blocks/attention.zig");
const forward = @import("../forward.zig");
const hall_engine = @import("hall_engine.zig");
const keep_file = @import("hall_keep.zig");
const session = @import("../session/session.zig");
const parked = @import("../session/parked.zig");
const Engine = hall_engine.Engine;
const Keep = keep_file.Keep;

pub const Carried = struct {
    gpa: Allocator,
    /// The session's sequence in the pool, idle; null: nothing was carried.
    chain: ?*extents.Chain = null,
    /// Its states, and the buffers the session kept for its next ones.
    keeps: std.ArrayList(Keep) = .empty,
    buffers: std.ArrayList(forward.Snapshot) = .empty,
    /// Positions whose rows were copied, and how many states came with them.
    rows: usize = 0,
    states: usize = 0,
    /// The sequences the session left, carried: their chains, idle, and how many.
    left: std.ArrayList(*extents.Chain) = .empty,

    pub fn deinit(self: *Carried) void {
        for (self.keeps.items) |*keep| keep.deinit();
        self.keeps.deinit(self.gpa);
        for (self.buffers.items) |*state| state.deinit();
        self.buffers.deinit(self.gpa);
        self.left.deinit(self.gpa);
        self.* = undefined;
    }
};

/// Copies what `held` holds into `rooms` and takes its states, which it no longer has; `held` is then only to
/// be destroyed, once the device has run the copies (`cuda.synchronize`). An error leaves `carried` to be
/// given back with `drop`.
pub fn take(carried: *Carried, held: *session.Session, rooms: hall_engine.Rooms) !void {
    var hand = try held.handover(rooms.pool.capacity, rooms.trunk.len, rooms.head_pool != null) orelse return;
    const gpa = hand.gpa;
    const pos = hand.pos;
    const known = hand.known;
    const chain = try rooms.pool.open();
    carried.chain = chain;
    try rooms.pool.grow(chain, pos);
    rooms.pool.append(chain, known[0..pos]);
    rooms.pool.retire(chain);
    for (hand.caches, rooms.trunk) |cache, pool| try rows(chain, cache, pool, pos);
    const head = hand.head;
    if (head) |cache| try rows(chain, cache, rooms.head_pool.?, pos);
    carried.rows = pos;

    const ratio = hand.ratio;
    try carried.keeps.ensureTotalCapacity(carried.gpa, hand.states() + 1);
    while (hand.next()) |next| {
        const at = next.pos;
        if (at < 1 or at > pos or standing(carried, at) != null) {
            hand.drop();
            continue;
        }
        const pooled = try keep_file.pooledOf(gpa, hand.caches, head, if (ratio == 0) 0 else at / ratio);
        carried.keeps.appendAssumeCapacity(.{ .chain = chain, .pos = at, .head_len = at, .next = known[at], .kind = if (next.stride) .stride else .window, .target = at, .state = hand.take(), .pooled = pooled, .shared = next.shared });
    }
    // The state where the session stands is a reply's end, as a stream's that ended in the pool; one of its
    // states that stands there already is that one.
    if (standing(carried, pos)) |keep| {
        keep.kind = .end;
    } else {
        var state = try last(&hand);
        const pooled = keep_file.pooledOf(gpa, hand.caches, head, if (ratio == 0) 0 else pos / ratio) catch |err| {
            state.deinit();
            return err;
        };
        carried.keeps.appendAssumeCapacity(.{ .chain = chain, .pos = pos, .head_len = pos, .next = known[pos], .kind = .end, .target = pos, .state = state, .pooled = pooled });
    }
    for (hand.left, 0..) |*one, i| {
        const before = carried.keeps.items.len;
        if (!try left(carried, &hand, one, rooms)) continue;
        // older than the carried chain, in the lot's order: `left` grows with the lot's clock
        carried.left.items[carried.left.items.len - 1].used = i;
        std.debug.assert(carried.keeps.items.len > before);
    }
    carried.states = carried.keeps.items.len;
    while (hand.buffer()) |one| {
        var state = one;
        if (whole(state)) carried.buffers.append(carried.gpa, state) catch state.deinit() else state.deinit();
    }
}

/// What `take` left when it failed: the chain's extents are free again and nothing is carried.
pub fn drop(carried: *Carried, rooms: hall_engine.Rooms) void {
    for (carried.left.items) |chain| rooms.pool.release(chain);
    if (carried.chain) |chain| rooms.pool.release(chain);
    const gpa = carried.gpa;
    carried.deinit();
    carried.* = .{ .gpa = gpa };
}

/// The states join the pool's store, within its bounds, once the engine stands where it stays.
pub fn give(carried: *Carried, engine: *Engine) void {
    keep_file.adopt(engine, &carried.keeps, &carried.buffers);
}

fn standing(carried: *Carried, pos: usize) ?*Keep {
    for (carried.keeps.items) |*keep| if (keep.pos == pos) return keep;
    return null;
}

/// Every device buffer of `state` is made (one taken ahead of a window that did not run has none).
fn whole(state: forward.Snapshot) bool {
    for ([_][]cuda.Buffer{ state.recurrent, state.conv, state.tails }) |list| {
        for (list) |buffer| if (buffer.view.ptr == null) return false;
    }
    return true;
}

/// The state the session stands at, as a snapshot: into the buffers of one it kept unused, else new ones.
fn last(hand: *session.Handover) !forward.Snapshot {
    const source = hand.state.*;
    var state = hand.kept() orelse return forward.Snapshot.take(hand.gpa, source);
    if (!whole(state) or state.recurrent.len != source.recurrent.len or state.tails.len != source.ple.len) {
        state.deinit();
        return forward.Snapshot.take(hand.gpa, source);
    }
    errdefer state.deinit();
    state.pos = source.pos;
    for (state.recurrent, state.conv, source.conv, 0..) |recurrent, conv, window, linear| {
        try forward.committed(source, linear, recurrent);
        try conv.copyLaunched(window);
    }
    for (state.tails, source.ple) |tail, branch| try tail.copyLaunched(branch.tail);
    try keep_file.histories(&state, source.ple);
    return state;
}

/// A contiguous cache's rows below `pos` into `chain`'s extents of `pool`: one launch for each run of extents
/// that follow one another, which in a pool just made is the whole chain.
fn rows(chain: *const extents.Chain, cache: attention.Cache, pool: attention.Pool, pos: usize) cuda.Error!void {
    for (parked.Span.arrays(cache), arraysOf(pool)) |source, dest| {
        if (source) |buffer| try runs(chain, dest.?, buffer, 0, 0, pos);
    }
}

fn arraysOf(pool: attention.Pool) [5]?cuda.Buffer {
    return .{ pool.k, pool.v, pool.k_scale, pool.v_scale, pool.index_keys };
}

/// `source`'s rows from `at` on into `dest`'s rows of `chain`'s positions `[from, to)`: one launch for each
/// run of extents that follow one another.
fn runs(chain: *const extents.Chain, dest: cuda.Buffer, source: cuda.Buffer, at: usize, from: usize, to: usize) cuda.Error!void {
    const table = chain.table.items;
    var p = from;
    while (p < to) {
        var next = (p >> extents.shift) + 1;
        while (next < table.len and next * extents.positions < to and table[next] == table[next - 1] + 1) next += 1;
        const end = @min(to, next * extents.positions);
        try dest.copyRows(chain.row(p), source, at + (p - from), end - p);
        p = end;
    }
}

/// Carries `one`, a sequence the session left, into a chain of its own forked from the carried one, with its
/// states, which `one` no longer holds. False, with nothing changed: it is not carried (it shares less than
/// an extent's worth of the layout, or the extents do not hold it).
fn left(carried: *Carried, hand: *session.Handover, one: *parked.Sequence, rooms: hall_engine.Rooms) !bool {
    const ratio = hand.ratio;
    const head = hand.head;
    const main = carried.chain.?;
    const to = one.to;
    if (to < 1 or to > rooms.pool.capacity or one.tokens.len != to + 1 or one.states.items.len == 0) return false;
    if (one.spans.len != hand.caches.len or (one.head == null) != (head == null)) return false;
    // below `base` the sequence's rows are the carried chain's: the extents are named again
    var below = @min(one.from, main.len());
    for (one.spans) |span| below = @min(below, lowest(span, ratio));
    if (one.head) |span| below = @min(below, lowest(span, ratio));
    const base = below >> extents.shift << extents.shift;
    if (!std.mem.eql(i32, one.tokens[0..base], main.tokens.items[0..base])) return false;
    const forked = rooms.pool.fork(main, base) catch return false;
    std.debug.assert(forked.copy == null);
    const chain = forked.chain;
    rooms.pool.grow(chain, to) catch {
        rooms.pool.release(chain);
        return false;
    };
    rooms.pool.append(chain, one.tokens[base..to]);
    rooms.pool.retire(chain);
    carried.left.append(carried.gpa, chain) catch |err| {
        rooms.pool.release(chain);
        return err;
    };
    for (hand.caches, one.spans, rooms.trunk) |cache, span, pool| try spanRows(chain, cache, span, pool, base);
    if (head) |cache| try spanRows(chain, cache, one.head.?, rooms.head_pool.?, base);
    carried.rows += to - base;

    const gpa = hand.gpa;
    try carried.keeps.ensureUnusedCapacity(carried.gpa, one.states.items.len);
    while (one.states.items.len > 0) {
        const at = one.states.items[0].pos;
        if (at < 1 or at > to or !whole(one.states.items[0])) {
            hand.giveLeft(one);
            continue;
        }
        const pooled = try pooledLeft(gpa, hand.caches, head, one, if (ratio == 0) 0 else at / ratio);
        carried.keeps.appendAssumeCapacity(.{ .chain = chain, .pos = at, .head_len = at, .next = one.tokens[at], .kind = if (at == to) .end else if (at < one.asked) .window else .stride, .target = at, .state = hand.takeLeft(one), .pooled = pooled, .shared = one.shared == at });
    }
    return true;
}

/// The lowest position whose rows `span` holds rather than the caches.
fn lowest(span: parked.Span, ratio: usize) usize {
    var least: usize = span.to;
    for (span.rows) |held| if (held.held != null) {
        least = @min(least, held.from);
    };
    if (span.pooled.held != null) least = @min(least, span.pooled.from * ratio);
    return least;
}

/// A sequence's rows of one cache from `base` to its end: the cache's below where `span` holds them.
fn spanRows(chain: *const extents.Chain, cache: attention.Cache, span: parked.Span, pool: attention.Pool, base: usize) cuda.Error!void {
    for (parked.Span.arrays(cache), arraysOf(pool), span.rows) |source, dest, held| {
        const buffer = source orelse continue;
        const own = if (held.held != null) held.from else span.to;
        try runs(chain, dest.?, buffer, base, base, @min(own, span.to));
        if (held.held) |rows_held| try runs(chain, dest.?, rows_held, 0, own, own + held.count);
    }
}

/// `pooledOf` for a state of a sequence left: each cache's pooled keys below `blocks`, the session's below
/// where the sequence's own are held.
fn pooledLeft(gpa: std.mem.Allocator, caches: []const attention.Cache, head: ?attention.Cache, one: *const parked.Sequence, blocks: usize) !cuda.Parts {
    var pooled = try keep_file.pooledOf(gpa, caches, head, blocks);
    errdefer pooled.deinit(gpa);
    if (blocks == 0) return pooled;
    for (pooled.list, 0..) |buffer, i| {
        const span = if (i < caches.len) one.spans[i] else one.head.?;
        const held = span.pooled.held orelse continue;
        if (blocks <= span.pooled.from) continue;
        try buffer.copyRows(span.pooled.from, held, 0, @min(blocks, span.pooled.from + span.pooled.count) - span.pooled.from);
    }
    return pooled;
}
