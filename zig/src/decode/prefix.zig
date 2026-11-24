//! The prefix store over streams (spec 018, step B7), on `extents.Pool`. A chain is a stream's while it
//! decodes and the store's after `retire` or `preempt`. Forks name the extents wholly below the fork and
//! return the pool's `Copy` for the partial one; `evict` and `release` refuse until `copied`, so that copy
//! runs on the device before anything can free its source. Snapshots are positions on a chain (the device
//! buffers are the caller's). Nothing here reaches the device. `zspark serve --parallel` above 1 calls it.
//!
//! Eviction skips a tail another chain names: shortening it would drop snapshots and free nothing. Admission's
//! length is the prompt (on a replay, the prompt and the tokens already sent). One round of room is left
//! beside that length for every stream already live. The store holds `Store.limit` snapshots at most, and a chain
//! `prompt_marks` of a prompt's kinds and `reply_marks` on a reply's stride (the caller thins them, spec 054).
//!
//! A snapshot where a shared prefix ends (a harness's system prompt and tools, `share`) is the last to go: every
//! conversation that begins with the prefix forks from it, and it lies on the chain of the first, which is the
//! least recently used long before the prefix is (TensorFold's `prompt_cache.victimBut` has the same order).

const std = @import("std");
const extents = @import("extents.zig");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Chain = extents.Chain;
const Copy = extents.Copy;
const Pool = extents.Pool;
const Source = extents.Source;
const Shrunk = extents.Shrunk;
const Verdict = extents.Verdict;

/// Prompt marks a lone session keeps (`session.Options.snapshots`).
pub const prompt_marks = 8;
/// Reply strides one slot keeps (`session.Options.reply_marks`).
pub const reply_marks = 2;
/// Positions a round reserves past where a stream stands: a reply's window, and after it the rows a head
/// chains its drafts on (`pool/hall_engine.zig`'s `roundInner`). The model is not imported here.
pub const round_positions = 16 + 15;

/// Snapshots the store holds at most, whatever their bytes (the caller bounds those, which the store does not
/// see): what the host's tables are sized for (spec 054, F2).
pub const states_max = 64;

/// What a snapshot was kept for.
pub const Kind = enum {
    /// Where a prompt's chunk ended: the newest of a prompt still running, which a stream taken off goes on
    /// from.
    chunk,
    /// Where a prompt's window ended that crossed its stride with no message's start in it.
    window,
    /// Where a message starts.
    message,
    /// `tail` tokens before a prompt's end.
    tail,
    /// One token before where a prompt parted from what was held.
    parted,
    /// After the token that closes a reply's reasoning.
    reasoning,
    /// On a reply's stride.
    stride,
    /// Where a reply ended.
    end,

    /// Whether it is one of a chain's `prompt_marks`: a reply's strides and a prompt's newest chunks are
    /// `reply_marks` each, and a reply's end is what a session's live state is.
    pub fn thins(self: Kind) bool {
        return self != .chunk and self != .stride and self != .end;
    }
};

pub const Error = extents.Error || error{
    /// A fork's `Copy` has not been run on the device. The source extent must not be freed yet.
    CopyPending,
};

/// A snapshot's place. The device buffers it names are the caller's.
pub const Mark = struct {
    chain: *Chain,
    pos: usize,
    kind: Kind = .chunk,
    /// Where a prefix other conversations begin with ends (`Store.share`): it goes after every snapshot that
    /// is not, and its chain is not shortened below it while another can be (spec 064).
    shared: bool = false,
    /// The caller wrote a snapshot of its chain at or above it to disk (`leaving`): it is not written again.
    spilt: bool = false,
};

/// Where a request's tokens were found, and the position to fork at: the highest snapshot at or under the
/// shared prefix.
pub const Hit = struct {
    chain: *Chain,
    at: usize,
};

pub const Fork = struct {
    chain: *Chain,
    copy: ?Copy,
};

pub const Store = struct {
    gpa: Allocator,
    pool: *Pool,
    /// Positions one live stream may append in a round. Admission leaves extents for that many.
    round: usize,
    slots: []?*Chain,
    marks: std.ArrayList(Mark) = .empty,
    /// Snapshots held at most.
    limit: usize = states_max,
    /// Set when `fork` or `revive` returned a `Copy`, until `copied`.
    pending: ?Copy = null,
    /// Told of the snapshots that go for room, while their rows are still held: before `dropOne` takes one
    /// and before `evict` lets an extent go (`spill.zig`: the caller may write them to disk). Not of those
    /// a caller drops itself, a chain cut or released.
    leaving: ?Leaving = null,

    pub const Going = union(enum) {
        one: Mark,
        /// Every snapshot of `chain` above `len`.
        past: struct { chain: *Chain, len: usize },
    };
    pub const Leaving = struct { context: *anyopaque, call: *const fn (*anyopaque, Going) void };

    pub fn init(gpa: Allocator, pool: *Pool, slots: usize, round: usize) Error!Store {
        const held = try gpa.alloc(?*Chain, slots);
        errdefer gpa.free(held);
        @memset(held, null);
        return .{ .gpa = gpa, .pool = pool, .round = round, .slots = held };
    }

    pub fn deinit(self: *Store) void {
        self.marks.deinit(self.gpa);
        self.gpa.free(self.slots);
        self.* = undefined;
    }

    pub fn bound(self: *const Store) usize {
        return self.limit;
    }

    /// A live chain with no tokens, in `slot`.
    pub fn open(self: *Store, slot: usize) Error!*Chain {
        try self.ready();
        assert(self.slots[slot] == null);
        const chain = try self.pool.open();
        self.slots[slot] = chain;
        return chain;
    }

    /// The live chain `fork` returned, now decoding in `slot`.
    pub fn place(self: *Store, slot: usize, chain: *Chain) void {
        assert(self.slots[slot] == null and chain.live);
        self.slots[slot] = chain;
    }

    /// A live chain of `source`'s first `at` tokens. The extents wholly below `at` are named again; a point
    /// inside an extent returns the rows to copy. The caller runs that copy on every layer, then `copied`,
    /// before the next `evict` or `release`. `source` stays as it was.
    pub fn fork(self: *Store, source: *Chain, at: usize) Error!Fork {
        try self.ready();
        const result = try self.pool.fork(source, at);
        if (result.copy) |copy| self.pending = copy;
        return .{ .chain = result.chain, .copy = result.copy };
    }

    /// `source` goes on in `slot` from its first `at` tokens. The tail is let go. A `Copy` is returned only
    /// when the extent `at` lies in has another name.
    pub fn revive(self: *Store, slot: usize, source: *Chain, at: usize) Error!?Copy {
        try self.ready();
        assert(self.slots[slot] == null);
        const copy = try self.pool.revive(source, at);
        if (copy) |held| self.pending = held;
        self.dropPast(source, at);
        self.slots[slot] = source;
        return copy;
    }

    /// The device has run the pending `Copy`.
    pub fn copied(self: *Store) void {
        self.pending = null;
    }

    /// One snapshot on `chain` at `pos` (a token the chain holds). Past `bound` one goes, by `dropOne`'s order.
    pub fn note(self: *Store, chain: *Chain, pos: usize, kind: Kind) Error!void {
        assert(pos >= 1 and pos <= chain.len());
        for (self.marks.items) |mark| if (mark.chain == chain and mark.pos == pos) return;
        try self.marks.append(self.gpa, .{ .chain = chain, .pos = pos, .kind = kind });
        while (self.marks.items.len > self.bound()) _ = self.dropOne();
    }

    /// The snapshot at `pos` on `chain` is a shared prefix's end from now on.
    pub fn share(self: *Store, chain: *const Chain, pos: usize) void {
        for (self.marks.items) |*mark| if (mark.chain == chain and mark.pos == pos) {
            mark.shared = true;
        };
    }

    /// The snapshot at `pos` on `chain` goes, where there is one.
    pub fn drop(self: *Store, chain: *const Chain, pos: usize) void {
        for (self.marks.items, 0..) |mark, i| if (mark.chain == chain and mark.pos == pos) {
            _ = self.marks.swapRemove(i);
            return;
        };
    }

    /// The highest snapshots of a prompt's kinds (`Kind.thins`) that `tokens` holds at or below `upto`, on
    /// whichever chain: their positions into `out`, ascending, each once. What a session's list of snapshots
    /// is to a prompt it plans.
    pub fn along(self: *const Store, tokens: []const i32, upto: usize, out: []usize) []usize {
        var n: usize = 0;
        for (self.marks.items) |mark| {
            if (!mark.kind.thins() or mark.pos > upto or mark.pos > tokens.len or mark.pos > mark.chain.len()) continue;
            if (std.mem.indexOfScalar(usize, out[0..n], mark.pos) != null) continue;
            if (n == out.len and (n == 0 or mark.pos < out[0])) continue;
            if (!std.mem.eql(i32, mark.chain.tokens.items[0..mark.pos], tokens[0..mark.pos])) continue;
            // the lowest goes when full
            if (n < out.len) {
                out[n] = mark.pos;
                n += 1;
            } else out[0] = mark.pos;
            std.mem.sort(usize, out[0..n], {}, std.sort.asc(usize));
        }
        return out[0..n];
    }

    /// How many tokens `tokens` shares with the chain that shares most: where a prompt parts from what is
    /// held, as a session's prompt parts from its one sequence.
    pub fn shared(self: *const Store, tokens: []const i32) usize {
        var most: usize = 0;
        for (self.pool.chains.items) |chain| {
            const held = chain.tokens.items;
            if (held.len <= most) continue;
            const same = std.mem.indexOfDiff(i32, held, tokens) orelse @min(held.len, tokens.len);
            most = @max(most, same);
        }
        return most;
    }

    /// The longest snapshot at or under a prefix of `tokens`.
    pub fn find(self: *const Store, tokens: []const i32) ?Hit {
        var best: ?Hit = null;
        for (self.marks.items) |mark| {
            if (mark.pos > tokens.len or mark.pos > mark.chain.len()) continue;
            if (!std.mem.eql(i32, mark.chain.tokens.items[0..mark.pos], tokens[0..mark.pos])) continue;
            const take = if (best) |held| mark.pos > held.at or (mark.pos == held.at and mark.chain.used > held.chain.used) else true;
            if (take) best = .{ .chain = mark.chain, .at = mark.pos };
        }
        return best;
    }

    /// The least recently used idle chain whose tail is its alone. A shared tail is left: dropping it frees
    /// nothing. Snapshots past the new end are dropped; one at the new end stays. Null when no private tail
    /// is idle. `spare` is the chain about to be forked from. A tail that holds a shared prefix's end is taken
    /// only when no other can be.
    pub fn evict(self: *Store, spare: ?*const Chain) Error!?Shrunk {
        try self.ready();
        var oldest: ?*Chain = null;
        var last: ?*Chain = null;
        for (self.pool.chains.items) |chain| {
            if (chain.live or chain == spare or chain.table.items.len == 0) continue;
            const tail = chain.table.items[chain.table.items.len - 1];
            if (self.pool.refs[@intCast(tail)] != 1) continue;
            const left = (chain.table.items.len - 1) << extents.shift;
            const holds = for (self.marks.items) |mark| {
                if (mark.shared and mark.chain == chain and mark.pos > left) break true;
            } else false;
            const best = if (holds) &last else &oldest;
            if (best.* == null or chain.used < best.*.?.used) best.* = chain;
        }
        const chain = oldest orelse last orelse return null;
        if (self.leaving) |told| told.call(told.context, .{ .past = .{ .chain = chain, .len = (chain.table.items.len - 1) << extents.shift } });
        const shrunk = self.pool.shrink(chain);
        self.dropPast(chain, shrunk.len);
        return shrunk;
    }

    /// The stream in `slot` stops. Its chain stays idle, with its tokens and snapshots, so a replay can fork
    /// what eviction has not taken.
    pub fn preempt(self: *Store, slot: usize) Error!void {
        try self.ready();
        const chain = self.slots[slot] orelse return;
        self.pool.retire(chain);
        self.slots[slot] = null;
    }

    /// The stream in `slot` ends. `kept` is how many of its tokens the chain holds after: a reply the
    /// connection finished keeps all of them, and a cancel keeps those the connection took.
    pub fn retire(self: *Store, slot: usize, kept: usize) Error!void {
        try self.ready();
        const chain = self.slots[slot] orelse return;
        self.pool.retire(chain);
        if (kept < chain.len()) {
            self.pool.trim(chain, kept);
            self.dropPast(chain, kept);
        }
        self.slots[slot] = null;
    }

    /// Every extent of `chain` is let go, and its snapshots with it.
    pub fn release(self: *Store, chain: *Chain) Error!void {
        try self.ready();
        self.forget(chain);
        for (self.slots) |*slot| {
            const held = slot.* orelse continue;
            if (held == chain) slot.* = null;
        }
        self.pool.release(chain);
    }

    /// `len` is the prompt, or on a replay the prompt plus the tokens already sent. A prompt past the context
    /// is `never`. One that fits is `wait` when a round of `round` positions on every stream already live, and
    /// on this request, would not fit beside it.
    pub fn admission(self: *const Store, len: usize, source: ?Source) Verdict {
        if (len > self.pool.capacity) return .never;
        const verdict = self.pool.admission(len, source);
        if (verdict != .admit) return verdict;
        const left = leftAfter(self.pool, len, source);
        return if (left >= self.roundExtents(len)) .admit else .wait;
    }

    fn ready(self: *const Store) Error!void {
        if (self.pending != null) return error.CopyPending;
    }

    /// Extents the next round needs after a prompt of `len` is placed: one window on that prompt, and one on
    /// every chain already live.
    fn roundExtents(self: *const Store, len: usize) usize {
        var need: usize = 0;
        const end = @min(len +| self.round, self.pool.capacity);
        const prompt = extents.extentsFor(len);
        const after = extents.extentsFor(end);
        if (after > prompt) need += after - prompt;
        for (self.pool.chains.items) |chain| {
            if (!chain.live) continue;
            const upto = @min(chain.len() +| self.round, self.pool.capacity);
            const want = extents.extentsFor(upto);
            if (want > chain.table.items.len) need += want - chain.table.items.len;
        }
        return need;
    }

    fn dropPast(self: *Store, chain: *Chain, len: usize) void {
        var i: usize = 0;
        while (i < self.marks.items.len) {
            const mark = self.marks.items[i];
            if (mark.chain == chain and mark.pos > len) {
                _ = self.marks.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }

    fn forget(self: *Store, chain: *Chain) void {
        var i: usize = 0;
        while (i < self.marks.items.len) {
            if (self.marks.items[i].chain == chain) {
                _ = self.marks.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }

    /// The snapshot that goes first: one that is no shared prefix's end before one that is, then an idle
    /// chain's before a live one's, the least recently used chain first and the lowest position on it. Null
    /// with none held.
    pub fn dropOne(self: *Store) ?Mark {
        if (self.marks.items.len == 0) return null;
        var best: usize = 0;
        for (self.marks.items[1..], 1..) |mark, index| {
            if (less(mark, self.marks.items[best])) best = index;
        }
        if (self.leaving) |told| told.call(told.context, .{ .one = self.marks.items[best] });
        return self.marks.swapRemove(best);
    }
};

/// Extents still free or held by idle chains once a prompt of `len` has taken its own. `Pool.admission` has
/// already answered `admit`, so this does not go below zero.
fn leftAfter(pool: *const Pool, len: usize, source: ?Source) usize {
    const want = extents.extentsFor(len);
    var room = pool.counts().free + pool.counts().idle;
    var kept: usize = 0;
    if (source) |s| {
        const table = s.chain.table.items;
        const partial = s.at % extents.positions != 0;
        kept = s.at >> extents.shift;
        if (s.in_place) {
            const copied = partial and pool.refs[@intCast(table[s.at >> extents.shift])] > 1;
            if (partial and !copied) kept += 1;
        }
        for (table[0..kept]) |e| room -= @intFromBool(pool.live[@intCast(e)] == 0);
    }
    const taken = want - kept;
    assert(room >= taken);
    return room - taken;
}

/// A shared prefix's end goes after every other snapshot. Then an idle chain's goes before a live one's, then
/// the least recently used chain, then the lowest position.
fn less(mark: Mark, than: Mark) bool {
    if (mark.shared != than.shared) return than.shared;
    const idle = !mark.chain.live;
    const than_idle = !than.chain.live;
    if (idle != than_idle) return idle;
    if (mark.chain.used != than.chain.used) return mark.chain.used < than.chain.used;
    return mark.pos < than.pos;
}
