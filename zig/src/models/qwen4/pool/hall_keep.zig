//! The states a stream of the pool keeps, and the forks that go on from them (spec 054). A stream keeps what
//! a session keeps of its one sequence. `marks.Planner` places the states of its prompt, a window each chunk:
//! where the messages start above the fork, `tail` before the end, below where the request parted from what
//! was held, where a chunk crossed the prompt's stride inside a message. A prompt's two newest chunk ends are
//! kept beside them (a stream taken off goes on from one). A reply keeps the state after the token that
//! closes its reasoning, one near each `reply_stride` tokens, and its end; so does a reply cut whose client
//! took all it was given. A chain holds `prefix.prompt_marks` of a prompt's kinds, thinned by the session's
//! rule, and `prefix.reply_marks` strides; the store holds `EngineOptions.states_bytes` of device memory and
//! `prefix.states_max` states, the least recently used idle chain's going first.
//!
//! A state is one recurrent side, the conv inputs and the PLE's tails (`forward.Snapshot`'s content), and the
//! pooled keys of the blocks below its position. The cache rows stay in the extents.
//!
//! A state inside a prompt's chunk is taken within the round (`within`, `rounds.Round.wanted`), as a session
//! takes one within a window: the chain replayed to the row. `Keeps.cut` ends the chunk at each one instead
//! and keeps the slot's state there, the form the tests hold the first against, bit for bit. A reply's stride
//! is the end of a round, where the slot stands: the first from which the next round could pass the stride,
//! so at most `round_rows - 1` positions below it and never above. The reasoning's end is a round kept
//! only up to that row, whose other rows run again; not where the rows after it were sent before the round
//! ran (a constraint's forced drafts), which are never run twice.
//!
//! An image prompt keeps none of these above its first placeholder row (`allows`, spec 019 WS-05).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const marks_file = @import("../../../decode/marks.zig");
const masks_file = @import("../../../decode/masks.zig");
const prefix = @import("../../../decode/prefix.zig");
const attention = @import("../blocks/attention.zig");
const forward = @import("../forward.zig");
const hall_engine = @import("hall_engine.zig");
const hall_seats = @import("hall_seats.zig");
const rounds = @import("rounds.zig");
const spill_file = @import("hall_spill.zig");
const state_windows = @import("../blocks/state.zig");
const Engine = hall_engine.Engine;
const Seat = hall_seats.Seat;

/// The rows a reply's round holds at most (`masks.max_rows`): how far below its stride a pool stream's state
/// may stand.
pub const round_rows = 16;

comptime {
    std.debug.assert(round_rows == masks_file.max_rows);
}

/// Whether a reply whose round has ended at `pos` keeps its state there for the stride that falls at `target`:
/// the first round's end from which the next round could pass the target, so the state stands at the target or
/// up to `round_rows - 1` positions below it and never above, and serves every prompt a state on the target
/// would (spec 054, F3: a round's end costs no replay).
pub fn strideDue(pos: usize, target: usize) bool {
    return pos + round_rows > target;
}

pub const Keep = struct {
    chain: *extents.Chain,
    pos: usize,
    head_len: usize,
    /// The token after `pos` the head took in with the last row: a prompt that goes on with another token
    /// would draft from a cache that is not its own, so it starts from a state below (`usable`), as a
    /// session's state serves a prompt that agrees past its position.
    next: i32,
    kind: prefix.Kind,
    /// Where the reply's next stride is counted from: `pos`, or for a stride's state the position it stands
    /// for, at or above it. So a reply taken up again keeps its strides where it would have uninterrupted.
    target: usize,
    /// The committed recurrent side, the conv inputs, the PLE's tails and histories at `pos`.
    state: forward.Snapshot,
    /// Each attention layer's pooled keys, then the head's: bf16 [pos / ratio, ..], the blocks wholly below
    /// `pos`, views of one allocation (`pooledOf`). Not allocated where there is none.
    pooled: cuda.Parts,
    /// It stands where a shared prefix ends (`Seat.kept`): the store lets it go last.
    shared: bool = false,
    /// What rank 1 of two knows it by (`hall_ranks.zig`); 0, a state rank 1 was never told of.
    id: u32 = 0,

    pub fn bytes(self: Keep) usize {
        return self.state.bytes() + self.pooled.bytes();
    }

    /// Frees the pooled keys and returns the state, whose buffers the next one may take.
    pub fn strip(self: *Keep) forward.Snapshot {
        self.pooled.deinit(self.state.gpa);
        return self.state;
    }

    pub fn deinit(self: *Keep) void {
        var state = self.strip();
        state.deinit();
    }
};

/// The pool's kept states: `list` is indexed by the store's marks, a chain and a position each.
pub const Keeps = struct {
    list: std.ArrayList(Keep) = .empty,
    /// Buffers ready for the next states, a state's each, not written: a state that went leaves its own here,
    /// and `ready` fills the list after a reply, so that a prompt's states cost no allocation before its
    /// first token (a state's buffers made anew were 6 ms each on a turn, and 50 to 100 ms each on NVFP4
    /// while the store grew past 111 GiB: spec 054). Inside `budget`.
    spare: std.ArrayList(forward.Snapshot) = .empty,
    /// The states a round's prompt chunk is taking within it, until the head has taken the chunk in.
    pending: std.ArrayList(Keep) = .empty,
    /// Device bytes `list` and `spare` hold at most.
    budget: usize,
    /// Tokens between a reply's states (`session.Options.reply_stride`); 0: none.
    reply_stride: usize,
    /// How many `spare` holds at most, and is filled up to after a reply: what a turn's prompt takes
    /// (`readyMost` lowers it where the budget is a few states).
    spares: usize = 4,
    /// `serve --spill-gib`'s store (`hall_spill.attach`): the states that go for room are written to it, and
    /// a prompt the pool holds less of is begun from it.
    spill: ?*spill_file.Store = null,
    /// `serve --learn`'s store, the session's when the pool opened (`hall.zig`'s `migrate`): a prompt whose
    /// shared prefix the pool holds less of is begun from its file (`hall_spill.learned`). Written by the lone
    /// session alone.
    learned: ?*spill_file.Store = null,
    /// Tests: a state inside a chunk ends the chunk (the form `within` is held against).
    cut: bool = false,
    /// A chunk's end that the plan does not name is kept too, the newest `prefix.reply_marks`. Tests turn it
    /// off for the serial reference, whose chunks are a row each.
    chunks: bool = true,
    /// States taken so far, by how: within a round, and where a slot stood; the rounds kept short for one;
    /// and the states' buffers made anew.
    took: struct { within: usize = 0, standing: usize = 0, cut: usize = 0, made: usize = 0 } = .{},
    /// The last `Keep.id` given.
    last_id: u32 = 0,

    /// An id for a new state, where two ranks keep it (`Keep.id`); 0 on one rank.
    fn newId(self: *Keeps, engine: *const Engine) u32 {
        if (engine.ranks == null) return 0;
        self.last_id += 1;
        return self.last_id;
    }

    pub fn deinit(self: *Keeps, gpa: Allocator) void {
        for (self.list.items) |*keep| keep.deinit();
        self.list.deinit(gpa);
        for (self.spare.items) |*state| state.deinit();
        self.spare.deinit(gpa);
        for (self.pending.items) |*keep| keep.deinit();
        self.pending.deinit(gpa);
    }

    pub fn bytes(self: *const Keeps) usize {
        var total: usize = 0;
        for (self.list.items) |keep| total += keep.bytes();
        for (self.spare.items) |state| total += state.bytes();
        for (self.pending.items) |keep| total += keep.bytes();
        return total;
    }

    pub fn at(self: *Keeps, chain: *const extents.Chain, pos: usize) ?*Keep {
        for (self.list.items) |*keep| if (keep.chain == chain and keep.pos == pos) return keep;
        return null;
    }
};

/// Frees the states whose mark the store let go.
pub fn sweep(engine: *Engine) void {
    const keeps = &engine.keeps;
    var i: usize = 0;
    while (i < keeps.list.items.len) {
        const keep = &keeps.list.items[i];
        const live = for (engine.store.marks.items) |mark| {
            if (mark.chain == keep.chain and mark.pos == keep.pos) break true;
        } else false;
        if (!live) {
            var dropped = keeps.list.swapRemove(i);
            if (engine.ranks) |link| link.dropped(dropped.id);
            release(engine, &dropped);
        } else i += 1;
    }
}

/// A state that goes: its buffers wait for the next one where there is room for them.
fn release(engine: *Engine, keep: *Keep) void {
    const keeps = &engine.keeps;
    var state = keep.strip();
    if (keeps.spare.items.len < readyMost(engine)) {
        keeps.spare.append(engine.gpa, state) catch state.deinit();
    } else state.deinit();
}

/// The state at `pos` on `chain` goes, mark and buffers.
fn drop(engine: *Engine, chain: *const extents.Chain, pos: usize) void {
    engine.store.drop(chain, pos);
    sweep(engine);
}

/// Bytes a state at `pos` takes: a slot's one side, conv inputs and tails, and the pooled rows below `pos`.
pub fn bytesAt(engine: *const Engine, pos: usize) usize {
    const slot = engine.slots[0];
    var total: usize = 0;
    for (slot.recurrent, slot.conv) |pair, conv| total += pair[0].bytes() + conv.bytes();
    for (slot.ple) |branch| total += branch.tail.bytes();
    const blocks = blocksBelow(engine, pos);
    for (slot.caches) |cache| total += blocks * rowBytes(cache.pooled);
    if (slot.head) |cache| total += blocks * rowBytes(cache.pooled);
    return total;
}

fn rowBytes(buffer: cuda.Buffer) usize {
    return buffer.bytes() / @as(usize, @intCast(buffer.view.size[0]));
}

fn blocksBelow(engine: *const Engine, pos: usize) usize {
    return if (engine.m.attention) |shared| pos / shared.shape.ratio else 0;
}

/// The states and the ready buffers back inside the budget: states go, by the store's order (an idle chain's
/// before a live one's, the least recently used chain first), each leaving its buffers ready while fewer
/// than `spares` are; with no state left the ready buffers go.
pub fn room(engine: *Engine) void {
    const keeps = &engine.keeps;
    while (keeps.bytes() > keeps.budget) {
        if (engine.store.marks.items.len > 0) {
            _ = engine.store.dropOne();
            sweep(engine);
        } else if (keeps.spare.pop()) |held| {
            var state = held;
            state.deinit();
        } else return;
    }
}

/// How many ready buffers the store holds at most: `spares`, and no more than an eighth of what the budget
/// holds, so that a budget of a few states keeps states.
fn readyMost(engine: *const Engine) usize {
    return @min(engine.keeps.spares, engine.keeps.budget / bytesAt(engine, 0) / 8);
}

/// Fills the ready buffers up to that many, with new ones while the budget has room and else with those of
/// the states that go by the store's order. After a reply, when no stream waits for a token of this one.
pub fn ready(engine: *Engine) void {
    const keeps = &engine.keeps;
    const each = bytesAt(engine, 0);
    const most = readyMost(engine);
    while (keeps.spare.items.len < most) {
        if (keeps.bytes() + each <= keeps.budget) {
            var state = fresh(engine, 0) catch return;
            keeps.spare.append(engine.gpa, state) catch return state.deinit();
        } else {
            if (engine.store.marks.items.len == 0) return;
            _ = engine.store.dropOne();
            sweep(engine);
        }
    }
}

/// The ready buffers filled up to half of `readyMost` with new ones while the budget has room, no state going
/// for them: while a round of replies runs on the device (`Engine.round`), so that their allocation overlaps
/// its work and a later round's state finds them made (spec 146 WS-04). Half, so that a state that goes
/// still leaves its buffers ready (`release`) and is not freed on the round's path while new ones are made.
/// Rank 1 keeps its own (`hall_ranks.Link`).
pub fn refill(engine: *Engine) void {
    const keeps = &engine.keeps;
    const most = (readyMost(engine) + 1) / 2;
    if (keeps.spare.items.len >= most) return;
    const each = bytesAt(engine, 0);
    var total = keeps.bytes();
    while (keeps.spare.items.len < most and total + each <= keeps.budget) : (total += each) {
        var state = fresh(engine, 0) catch return;
        keeps.spare.append(engine.gpa, state) catch return state.deinit();
    }
}

/// A state's buffers, not written: ready ones, or those of a state that goes for it where the budget is
/// full, or new ones.
pub fn blank(engine: *Engine, pos: usize) !forward.Snapshot {
    const keeps = &engine.keeps;
    while (true) {
        if (keeps.spare.pop()) |held| {
            var state = held;
            state.pos = pos;
            return state;
        }
        if (readyMost(engine) == 0 or keeps.bytes() + bytesAt(engine, pos) <= keeps.budget or engine.store.marks.items.len == 0) break;
        _ = engine.store.dropOne();
        sweep(engine);
    }
    return fresh(engine, pos);
}

/// A state's buffers made anew, not written: views of one allocation (`cuda.cut`), laid as a slot's.
pub fn fresh(engine: *Engine, pos: usize) !forward.Snapshot {
    const gpa = engine.gpa;
    const slot = engine.slots[0];
    engine.keeps.took.made += 1;
    var state: forward.Snapshot = .{ .pos = pos, .recurrent = &.{}, .conv = &.{}, .tails = &.{}, .history = &.{}, .gpa = gpa };
    errdefer state.deinit();
    state.recurrent = try forward.Snapshot.copies(gpa, slot.recurrent.len);
    state.conv = try forward.Snapshot.copies(gpa, slot.conv.len);
    state.tails = try forward.Snapshot.copies(gpa, slot.ple.len);
    for (state.recurrent, state.conv, slot.recurrent, slot.conv) |*recurrent, *conv, pair, source| {
        recurrent.* = shapedAs(pair[0]);
        conv.* = shapedAs(source);
    }
    for (state.tails, slot.ple) |*tail, branch| tail.* = shapedAs(branch.tail);
    state.block = try cuda.cut(&.{ state.recurrent, state.conv, state.tails });
    return state;
}

/// `source`'s type and shape, with no memory yet.
fn shapedAs(source: cuda.Buffer) cuda.Buffer {
    var shape: [6]usize = undefined;
    const ndim: usize = @intCast(source.view.ndim);
    for (shape[0..ndim], source.view.size[0..ndim]) |*dim, size| dim.* = @intCast(size);
    return .shaped(source.view.dtype, shape[0..ndim]);
}

/// The histories of `state` as `source`'s stand, a copy each.
pub fn histories(state: *forward.Snapshot, source: []const @import("../blocks/ple.zig").State) !void {
    const gpa = state.gpa;
    for (state.history) |h| gpa.free(h);
    gpa.free(state.history);
    state.history = &.{};
    const history = try gpa.alloc([]u32, source.len);
    for (history) |*h| h.* = &.{};
    state.history = history;
    for (history, source) |*h, branch| h.* = try gpa.dupe(u32, branch.history);
}

/// The pooled keys below `pos` of slot `index`, the trunk's layers then the head's: after the head has taken
/// in the rows up to `pos`.
pub fn pooledBelow(engine: *Engine, index: usize, pos: usize) !cuda.Parts {
    const slot = engine.slots[index];
    return pooledOf(engine.gpa, slot.caches, slot.head, blocksBelow(engine, pos));
}

/// The first `blocks` pooled keys of each of `caches`, then of `head`: a copy each, the copies views of one
/// allocation (`cuda.cut`). With no block, none has memory.
pub fn pooledOf(gpa: Allocator, caches: []const attention.Cache, head: ?attention.Cache, blocks: usize) !cuda.Parts {
    var pooled: cuda.Parts = .{ .list = try forward.Snapshot.copies(gpa, caches.len + @intFromBool(head != null)) };
    errdefer pooled.deinit(gpa);
    if (blocks == 0) return pooled;
    for (pooled.list, 0..) |*buffer, i| {
        const source = if (i < caches.len) caches[i].pooled else head.?.pooled;
        buffer.* = shapedAs(source);
        buffer.view.size[0] = @intCast(blocks);
    }
    pooled.block = try cuda.cut(&.{pooled.list});
    for (pooled.list, 0..) |buffer, i| try buffer.copyRows(0, if (i < caches.len) caches[i].pooled else head.?.pooled, 0, blocks);
    return pooled;
}

/// States made elsewhere join the store's, each with its mark, and `buffers` the ready ones while there is
/// room for them: a session's, when the pool opens (`hall_carry.zig`). Then the store's bounds hold, by its
/// own order: of one chain the lowest goes first.
pub fn adopt(engine: *Engine, list: *std.ArrayList(Keep), buffers: *std.ArrayList(forward.Snapshot)) void {
    const keeps = &engine.keeps;
    while (buffers.pop()) |held| {
        var state = held;
        if (keeps.spare.items.len < readyMost(engine)) {
            keeps.spare.append(engine.gpa, state) catch state.deinit();
        } else state.deinit();
    }
    while (list.pop()) |held| {
        var keep = held;
        keeps.list.append(engine.gpa, keep) catch {
            keep.deinit();
            continue;
        };
        // a mark the store could not take leaves a state `sweep` frees
        engine.store.note(keep.chain, keep.pos, keep.kind) catch {};
        if (keep.shared) engine.store.share(keep.chain, keep.pos);
    }
    sweep(engine);
    room(engine);
}

/// `keep` joins the states held, with its mark; then its chain is thinned and the store's bounds hold.
fn hold(engine: *Engine, seat: *Seat, keep: Keep) !void {
    var owned = keep;
    engine.keeps.list.append(engine.gpa, owned) catch |err| {
        owned.deinit();
        return err;
    };
    try engine.store.note(owned.chain, owned.pos, owned.kind);
    if (owned.shared) engine.store.share(owned.chain, owned.pos);
    if (owned.kind != .end) seat.newest = @max(seat.newest, owned.target);
    sweep(engine);
    // a prompt's kinds are thinned by its plan while it runs
    if (seat.phase != .fill or !owned.kind.thins()) thin(engine, seat, owned.kind);
    room(engine);
}

/// Keeps slot `index`'s state where it stands, for `kind`. `next` is the token after it, which the head took
/// in. A state held there already is that one, an `.end` if either is.
pub fn remember(engine: *Engine, index: usize, next: i32, kind: prefix.Kind) !void {
    const seat = &engine.seats[index].?;
    const slot = engine.slots[index];
    if (!allows(seat, slot.pos)) return;
    if (engine.keeps.at(seat.chain, slot.pos)) |held| {
        if (kind == .end and held.next == next) {
            held.kind = kind;
            for (engine.store.marks.items) |*mark| if (mark.chain == held.chain and mark.pos == held.pos) {
                mark.kind = kind;
            };
            thin(engine, seat, .stride);
        }
        return;
    }
    var state = try blank(engine, slot.pos);
    var given = false;
    errdefer if (!given) state.deinit();
    try takeState(engine, index, &state);
    const pooled = try pooledBelow(engine, index, slot.pos);
    given = true;
    engine.keeps.took.standing += 1;
    // the head's rows past the trunk's are drafts it chained, which `trim` takes back
    const target = if (kind == .stride) @max(seat.stride_at, slot.pos) else slot.pos;
    const head_len = slot.head_len - seat.drafted_rows;
    const id = engine.keeps.newId(engine);
    if (engine.ranks) |link| try link.keepTake(index, id, head_len);
    try hold(engine, seat, .{ .chain = seat.chain, .pos = slot.pos, .head_len = head_len, .next = next, .kind = kind, .target = target, .state = state, .pooled = pooled, .shared = seat.kept == slot.pos, .id = id });
}

/// Whether `seat` keeps a state at `pos`: an image prompt's none above its first placeholder row (spec 019
/// WS-05), as TensorFold 0.6.5 keeps no token-only state of one (decode.py:354, multi.py:284) and its kept
/// prompts are text only (state.py:361). A state at or below that row is a text prompt's to the bit.
pub fn allows(seat: *const Seat, pos: usize) bool {
    const picture = seat.image orelse return true;
    return pos <= picture.first();
}

/// Slot `index`'s committed state into `state`, a state's buffers: its side, conv inputs, tails and histories.
pub fn takeState(engine: *Engine, index: usize, state: *forward.Snapshot) !void {
    const slot = engine.slots[index];
    for (state.recurrent, state.conv, slot.conv, 0..) |recurrent, conv, source, linear| {
        try rounds.committed(slot, linear, recurrent);
        try conv.copyLaunched(source);
    }
    for (state.tails, slot.ple) |tail, branch| try tail.copyLaunched(branch.tail);
    try histories(state, slot.ple);
}

/// What `seat`'s chain holds of `kind`'s budget goes down to it: a reply's strides and a prompt's chunk ends
/// to the newest `prefix.reply_marks` each; the prompt's kinds to `prefix.prompt_marks`, by `decode/marks.zig`'s `thinned`
/// as a session's `keepWithin` (while the prompt runs its planner does that, `before`).
fn thin(engine: *Engine, seat: *Seat, kind: prefix.Kind) void {
    if (kind == .end) return;
    const chain = seat.chain;
    var list: [prefix.prompt_marks + 2]usize = undefined;
    while (true) {
        var n: usize = 0;
        for (engine.store.marks.items) |mark| {
            if (mark.chain != chain or n == list.len) continue;
            if (if (kind.thins()) !mark.kind.thins() else mark.kind != kind) continue;
            list[n] = mark.pos;
            n += 1;
        }
        if (n <= @as(usize, if (kind.thins()) prefix.prompt_marks else prefix.reply_marks)) return;
        std.mem.sort(usize, list[0..n], {}, std.sort.asc(usize));
        drop(engine, chain, if (kind.thins()) list[marks_file.thinned(list[0..n], list[n - 1] + 1, seat.kept)] else list[0]);
    }
}

/// The slot goes on from `keep`, on the chain `forked` made of its tokens.
pub fn applyFork(engine: *Engine, index: usize, keep: *Keep, forked: prefix.Fork) !void {
    // A fork at an extent's edge names whole extents and copies no row.
    if (forked.copy) |copy| {
        for (engine.trunk) |pool| try pool.copy(copy);
        if (engine.head_pool) |pool| try pool.copy(copy);
    }
    try engine.resetSlot(index);
    // rank 1 copies the rows too and goes on from its own state of the same id
    if (engine.ranks) |link| try link.keepFork(index, keep.id, forked.copy);
    try restore(engine, index, keep.state, keep.pooled, keep.pos, keep.head_len);
    const slot = &engine.slots[index];
    try slot.table.set(forked.chain.table.items);
    if (engine.ranks) |link| try link.lay(index, forked.chain.table.items);
    if (forked.copy != null) engine.store.copied();
    engine.store.place(index, forked.chain);
}

/// Slot `index`, just reset, stands at `pos` as `state` and `pooled` hold it, its head's `head_len` rows.
pub fn restore(engine: *Engine, index: usize, state: forward.Snapshot, pooled: cuda.Parts, pos: usize, head_len: usize) !void {
    const slot = &engine.slots[index];
    // the reset left side 0 the committed one
    for (slot.recurrent, slot.conv, state.recurrent, state.conv) |pair, conv, recurrent, saved| {
        try pair[0].copyFrom(recurrent);
        try conv.copyFrom(saved);
    }
    for (slot.ple, state.tails, state.history) |*branch, tail, history| {
        try branch.tail.copyFrom(tail);
        @memcpy(branch.history, history);
    }
    slot.pos = pos;
    const blocks = blocksBelow(engine, pos);
    if (blocks > 0) {
        for (slot.caches, pooled.list[0..slot.caches.len]) |cache, saved| try cache.pooled.copyRows(0, saved, 0, blocks);
        if (slot.head) |cache| try cache.pooled.copyRows(0, pooled.list[slot.caches.len], 0, blocks);
    }
    slot.head_len = head_len;
}

/// The longest kept state `prompt` goes on from, a state on disk that stands higher brought in first: the
/// shared prefix's that ends at `kept` (`hall_spill.learned`), then the longest that left
/// (`hall_spill.recall`), as a session reads them (`learnt.recalled`).
pub fn usable(engine: *Engine, prompt: []const i32, kept: ?usize) ?prefix.Hit {
    var found = standing(engine, prompt);
    if (spill_file.learned(engine, prompt, kept, if (found) |hit| hit.at else 0)) found = standing(engine, prompt);
    if (engine.keeps.spill == null or !spill_file.recall(engine, prompt, if (found) |hit| hit.at else 0)) return found;
    return standing(engine, prompt);
}

/// The longest state the pool keeps that `prompt` goes on from: its tokens are the prompt's, and the token
/// after them is the one the head took in (`Keep.next`).
fn standing(engine: *Engine, prompt: []const i32) ?prefix.Hit {
    var limit = prompt.len - 1;
    while (limit >= 1) {
        const hit = engine.store.find(prompt[0..limit]) orelse return null;
        if (engine.keeps.at(hit.chain, hit.at)) |keep| if (keep.next == prompt[hit.at]) return hit;
        limit = hit.at - 1;
    }
    return null;
}

/// Begins the plan of the states `seat`'s prompt keeps, as `Session.plan` begins a session's: it goes on from
/// `cached`, parted from what was held at `shared`, and its messages start at `seat.starts`. The states the
/// store holds along the prompt below the fork are told for the `near` rule alone: they are other chains',
/// and stay.
pub fn plan(engine: *const Engine, seat: *Seat, shared: usize, gate: ?i32) void {
    const prompt = seat.prompt;
    const context = seat.list.len - prefix.prompt_marks - 1;
    const held = engine.store.along(prompt, seat.cached, seat.list[0..context]).len;
    var options = engine.marking;
    options.budget = prefix.prompt_marks;
    seat.planner = .begin(options, .{ .pos = seat.cached, .len = prompt.len, .shared = shared, .starts = seat.starts, .kept = seat.kept }, &seat.list, held, held, &seat.dropped);
    // The newest state its sequence holds, from which a reply's stride is counted: as a session's, whose
    // live state is not one of its snapshots, so not a reply's end.
    seat.newest = 0;
    for (engine.keeps.list.items) |keep| {
        if (keep.kind == .end or keep.pos > seat.cached or keep.target <= seat.newest or keep.pos > keep.chain.len()) continue;
        if (std.mem.eql(i32, keep.chain.tokens.items[0..keep.pos], prompt[0..keep.pos])) seat.newest = keep.target;
    }
    // of the reply's own tokens sent before this admission, none closed the reasoning
    const sent = prompt[prompt.len - seat.sent ..];
    seat.close = if (gate) |close| (if (std.mem.indexOfScalar(i32, sent, close) == null) close else null) else null;
}

/// What the plan keeps `pos` for.
fn kindOf(engine: *const Engine, seat: *const Seat, pos: usize) prefix.Kind {
    const request = seat.planner.request;
    if (pos == request.len -| engine.marking.tail) return .tail;
    if (std.mem.indexOfScalar(usize, request.starts, pos) != null) return .message;
    return if (pos == request.shared -| 1) .parted else .window;
}

/// How many rows of its prompt `seat` takes this round, of `take`: with `Keeps.cut`, up to the next state it
/// keeps, found on a copy of its plan.
pub fn chunk(engine: *const Engine, seat: *const Seat, take: usize) usize {
    if (!engine.keeps.cut or take == 0) return take;
    var list = seat.list;
    var dropped = seat.dropped;
    var planner = seat.planner;
    planner.list = &list;
    planner.dropped = &dropped;
    planner.window(seat.filled + take);
    for (planner.taken()) |pos| if (pos > seat.filled) return pos - seat.filled;
    return take;
}

/// Before the round runs `tokens` of slot `index`'s prompt from where it is filled: the plan's window over
/// them. The states it keeps inside them are listed for `rounds.compute`, which writes their DeltaNet part;
/// one at their end is the slot's own state then (`after`); one the plan lets go that the chain holds goes.
pub fn before(engine: *Engine, index: usize, tokens: []const i32) !void {
    const keeps = &engine.keeps;
    const seat = &engine.seats[index].?;
    std.debug.assert(keeps.pending.items.len == 0 and engine.wanted.items.len == 0);
    const stop = seat.filled + tokens.len;
    const planner = &seat.planner;
    planner.window(stop);
    std.debug.assert(planner.drops().len == 0);
    const kept = planner.taken();
    var i: usize = 0;
    while (i < engine.store.marks.items.len) {
        const mark = engine.store.marks.items[i];
        if (mark.chain == seat.chain and mark.kind.thins() and mark.pos > seat.cached and std.mem.indexOfScalar(usize, kept, mark.pos) == null) {
            drop(engine, mark.chain, mark.pos);
            i = 0;
        } else i += 1;
    }
    seat.ending = null;
    for (kept) |pos| {
        if (pos <= seat.filled or !allows(seat, pos)) continue;
        if (pos == stop) {
            seat.ending = kindOf(engine, seat, pos);
            continue;
        }
        var state = try blank(engine, pos);
        keeps.pending.append(engine.gpa, .{ .chain = seat.chain, .pos = pos, .head_len = if (engine.m.mtp == null) 0 else pos, .next = seat.prompt[pos], .kind = kindOf(engine, seat, pos), .target = pos, .state = state, .pooled = .{}, .shared = seat.kept == pos, .id = keeps.newId(engine) }) catch |err| {
            state.deinit();
            return err;
        };
    }
    if (engine.ranks) |link| if (keeps.pending.items.len > 0) try link.keepWanted(index, keeps.pending.items);
    try engine.wanted.ensureTotalCapacity(engine.gpa, keeps.pending.items.len);
    for (keeps.pending.items) |keep| engine.wanted.appendAssumeCapacity(keep.state);
    engine.staged.wanted = engine.wanted.items;
    engine.staged.wanted_slot = index;
}

/// After the round's `compute` and before its `commit`: the PLE's part of the states taken within the chunk
/// `tokens`, whose rows begin at the round's row `row`: the tails slid to each one's row and the histories,
/// as `commit` leaves the slot's (`forward.Snapshot.finish`).
pub fn within(engine: *Engine, index: usize, tokens: []const i32, row: usize) !void {
    const keeps = &engine.keeps;
    engine.staged.wanted = &.{};
    engine.wanted.clearRetainingCapacity();
    if (engine.ranks) |link| if (keeps.pending.items.len > 0) try link.keepWithin(index, tokens, row);
    for (keeps.pending.items) |*keep| try withinOne(engine, index, &keep.state, tokens, row);
}

/// `within` for one state, at `state.pos` inside the chunk.
pub fn withinOne(engine: *Engine, index: usize, state: *forward.Snapshot, tokens: []const i32, row: usize) !void {
    const slot = engine.slots[index];
    const n = tokens.len;
    const kept = state.pos - slot.pos;
    try histories(state, slot.ple);
    for (state.tails, state.history, slot.ple, engine.buffers.ple) |tail, history, branch, buffers| {
        rounds.keepHistory(history, @ptrCast(tokens), kept);
        try tail.copyLaunched(branch.tail);
        if (tail.view.size[1] == 0) continue;
        const wide = buffers.shape.wide();
        try state_windows.shiftWindows(tail.view, .{ .view = buffers.nrow.span(row, n), .layer = 0, .row = wide, .rows = n }, kept);
    }
}

/// A round that failed before its states were held: they go.
pub fn abandon(engine: *Engine) void {
    engine.staged.wanted = &.{};
    engine.wanted.clearRetainingCapacity();
    for (engine.keeps.pending.items) |*keep| keep.deinit();
    engine.keeps.pending.clearRetainingCapacity();
}

/// After the round, the head having taken its rows in: what slot `index` keeps of them. Its prompt's chunk
/// (`filling`): the states taken within it get their pooled keys and are held, and the chunk's end is kept
/// where the prompt goes on past it. Its reply's window: the state after the reasoning's end where the round
/// was kept up to it (`reasoned`), and the round's end for the reply's stride.
pub fn after(engine: *Engine, index: usize, filling: bool, reasoned: bool) !void {
    const keeps = &engine.keeps;
    const seat = &engine.seats[index].?;
    const slot = engine.slots[index];
    if (filling) {
        if (engine.ranks) |link| if (keeps.pending.items.len > 0) try link.keepHold(index);
        // each leaves the list as it is held, so a failure frees the rest
        while (keeps.pending.items.len > 0) {
            var keep = keeps.pending.orderedRemove(0);
            keep.pooled = pooledBelow(engine, index, keep.pos) catch |err| {
                keep.deinit();
                return err;
            };
            keeps.took.within += 1;
            try hold(engine, seat, keep);
        }
        const done = seat.filled == seat.prompt.len;
        if (!done) {
            if (seat.ending != null or keeps.chunks) try remember(engine, index, seat.prompt[seat.filled], seat.ending orelse .chunk);
        } else if (keeps.reply_stride != 0) seat.stride_at = seat.newest + keeps.reply_stride;
        seat.ending = null;
        return;
    }
    if (seat.phase != .decode) return;
    if (reasoned) {
        try remember(engine, index, seat.last, .reasoning);
        // the stride is counted from it, as a session's
        if (keeps.reply_stride != 0) seat.stride_at = slot.pos + keeps.reply_stride;
    }
    if (keeps.reply_stride != 0 and strideDue(slot.pos, seat.stride_at)) {
        try remember(engine, index, seat.last, .stride);
        seat.stride_at = @max(seat.stride_at, slot.pos) + keeps.reply_stride;
    }
}

/// Where a reply's window `tokens`, of which the first `keep` rows were accepted, is kept up to for the state
/// after its reasoning's end, and whether that state is the round's end: the token that closes the reasoning
/// is one of the kept rows, and the round keeps the rows through it and runs the others again (spec 054, F3:
/// no second replay). `ahead` of the window's drafts were sent before it ran, and are never run again.
pub fn reasoning(seat: *Seat, tokens: []const i32, keep: usize, ahead: usize) struct { keep: usize, reasoned: bool } {
    const close = seat.close orelse return .{ .keep = keep, .reasoned = false };
    const row = std.mem.indexOfScalar(i32, tokens[0..keep], close) orelse return .{ .keep = keep, .reasoned = false };
    seat.close = null;
    if (row + 1 == keep) return .{ .keep = keep, .reasoned = true };
    if (ahead > row + 1) return .{ .keep = keep, .reasoned = false };
    return .{ .keep = row + 1, .reasoned = true };
}
