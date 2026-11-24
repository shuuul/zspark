//! Sequences a session ran and left (spec 064): a session's state holds one sequence, and a prompt that parts
//! from it writes its own rows over the attention caches from where the two part. What the sequence left held
//! above that point is copied here before it is written over, with the state at its end and the newest of the
//! states the session held above that point, so a later prompt of that conversation starts where a held
//! turn's does, and two conversations in turn each run their new tokens alone.
//!
//! A row's bits never depend on the rows beside it, so a row a later prompt writes at a position whose tokens
//! before it are the parked sequence's is the parked sequence's row, bit for bit. A sequence therefore keeps
//! its rows from `from` up only, `from` being the least it has shared with any prompt since it was left: the
//! state's rows below `from` are its own still. `Lot.lower` keeps that true before each prompt runs.
//!
//! A prompt takes a sequence up when one of its states stands higher than any the session holds for it, and
//! the sequence then leaves the lot: its states below where the prompt parts from it become the session's.
//!
//! The lot counts its sequences (`most`) but for those that are the last to hold the state where a harness
//! ends (`Lot.last`, spec 108), which its bytes bound alone, 64 at most, as TensorFold 1.0.12's one store
//! bounds its states by `--prompt-cache-gib` and marks 64 shared cuts (`core/prompt_cache.zig`); vLLM and
//! SGLang count no conversations either. So a new conversation over any of more harnesses than `most` in
//! turn starts at its harness's end.

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("../blocks/attention.zig");
const cuda = @import("../../../cuda/cuda.zig");
const forward = @import("../forward.zig");
const shelf_file = @import("../../../decode/shelf.zig");

/// The bytes of one row of `array`.
fn rowBytes(array: cuda.Buffer) usize {
    return array.bytes() / @as(usize, @intCast(array.view.size[0]));
}

/// Where a store's regions start, as `cudaMalloc`'s buffers would.
const alignment = 256;

/// `count` rows shaped as `like`'s at byte `at.*` of `store` (aligned up), a view; `at` moves past them.
fn place(store: cuda.Buffer, at: *usize, like: cuda.Buffer, count: usize) cuda.Buffer {
    at.* = std.mem.alignForward(usize, at.*, alignment);
    var shape: [6]usize = undefined;
    const n: usize = @intCast(like.view.ndim);
    for (shape[0..n], like.view.size[0..n]) |*dim, stored| dim.* = @intCast(stored);
    shape[0] = count;
    const base: [*]u8 = @ptrCast(store.view.ptr.?);
    const rows: cuda.Buffer = .{ .view = .init(base + at.*, like.view.dtype, shape[0..n]) };
    at.* += count * rowBytes(like);
    std.debug.assert(at.* <= store.bytes());
    return rows;
}

/// The rows of one array of a cache from `from` on, held in the sequence's store (`Sequence.store`), each at
/// its own place there: a later lowering writes the rows below beside them and moves none.
const Rows = struct {
    from: usize,
    count: usize = 0,
    /// The rows held, a view of `region`'s.
    held: ?cuda.Buffer = null,
    /// The store's rows for this array, row 0 standing for row `first`; null before the sequence has a store.
    region: ?cuda.Buffer = null,
    first: usize = 0,

    /// The bytes a region over rows `[first, to)` of `like` takes, its alignment's included.
    fn bytesOver(like: cuda.Buffer, first: usize, to: usize) usize {
        return (to -| first) * rowBytes(like) + alignment;
    }

    /// Holds `source`'s rows `[to_from, self.from)` too, in their places; `to` is where the rows end. The
    /// region covers them (`Span.house`).
    fn lower(self: *Rows, source: cuda.Buffer, to_from: usize, to: usize) cuda.Error!void {
        std.debug.assert(to_from <= self.from and self.from + self.count <= @max(to, self.from));
        if (to <= to_from or to_from == self.from) {
            if (self.count == 0) self.from = @min(to_from, self.from);
            return;
        }
        const region = self.region.?;
        std.debug.assert(to_from >= self.first);
        const added = self.from - to_from;
        try region.copyRows(to_from - self.first, source, to_from, added);
        self.from = to_from;
        self.count += added;
        self.held = .{ .view = region.span(self.from - self.first, self.count) };
    }

    /// The rows over `[first, to)` of `like` laid at `at.*` of `store`, the rows held copied into them.
    fn house(self: *Rows, store: cuda.Buffer, at: *usize, like: cuda.Buffer, first: usize, to: usize) cuda.Error!void {
        const region = place(store, at, like, to -| first);
        if (self.held) |held| {
            std.debug.assert(self.from >= first);
            try region.copyRows(self.from - first, held, 0, self.count);
            self.held = .{ .view = region.span(self.from - first, self.count) };
        }
        self.region = region;
        self.first = first;
    }

    fn restore(self: Rows, dest: cuda.Buffer) cuda.Error!void {
        if (self.held) |held| try dest.copyRows(self.from, held, 0, self.count);
    }
};

/// One attention cache's rows of a sequence that ends at `to`, from where it was last shared.
pub const Span = struct {
    to: usize,
    ratio: usize,
    /// `k`, `v`, their scales and `index_keys`, a row a position.
    rows: [5]Rows,
    /// The pooled keys of the blocks complete below `to`, a row a block.
    pooled: Rows,

    pub fn arrays(cache: attention.Cache) [5]?cuda.Buffer {
        return .{ cache.k, cache.v, cache.k_scale, cache.v_scale, cache.index_keys };
    }

    /// Nothing held yet: the cache holds every row of the sequence.
    pub fn init(ratio: usize, to: usize) Span {
        return .{ .to = to, .ratio = ratio, .rows = @splat(.{ .from = to }), .pooled = .{ .from = to / ratio } };
    }

    /// The bytes `house` lays out for rows from `first` up of a sequence of `cache` ending at `to`.
    pub fn bytesOver(cache: attention.Cache, ratio: usize, first: usize, to: usize) usize {
        var total = Rows.bytesOver(cache.pooled, first / ratio, to / ratio);
        for (arrays(cache)) |array| if (array) |like| {
            total += Rows.bytesOver(like, first, to);
        };
        return total;
    }

    /// Its rows from `first` up get their places at `at.*` of `store`, what it holds copied there.
    pub fn house(self: *Span, cache: attention.Cache, store: cuda.Buffer, at: *usize, first: usize) cuda.Error!void {
        for (&self.rows, arrays(cache)) |*rows, array| if (array) |like| try rows.house(store, at, like, first, self.to);
        try self.pooled.house(store, at, cache.pooled, first / self.ratio, self.to / self.ratio);
    }

    /// The cache's rows from `from` on are about to be written over: they are held from there, in the places
    /// `house` gave them.
    pub fn lower(self: *Span, cache: attention.Cache, from: usize) cuda.Error!void {
        std.debug.assert(cache.table == null);
        for (&self.rows, arrays(cache)) |*rows, array| if (array) |source| try rows.lower(source, @min(from, rows.from), self.to);
        try self.pooled.lower(cache.pooled, @min(from / self.ratio, self.pooled.from), self.to / self.ratio);
    }

    pub fn restore(self: Span, cache: attention.Cache) cuda.Error!void {
        for (self.rows, arrays(cache)) |rows, array| if (array) |dest| try rows.restore(dest);
        try self.pooled.restore(cache.pooled);
    }
};

/// A sequence left: its tokens, states of it, and its caches' rows from `from` up.
pub const Sequence = struct {
    /// The session's `known` as it stood: `to` tokens and the one chosen after them.
    tokens: []i32,
    /// The state's rows below it are this sequence's still.
    from: usize,
    /// Where the sequence's state stood: its rows end there.
    to: usize,
    /// Its states, lowest first: the one at `to`, and the newest the session held above where it parted.
    states: std.ArrayList(forward.Snapshot),
    spans: []Span,
    head: ?Span,
    /// Where the spans' rows are held, every array of every span in one buffer (spec 101): over positions
    /// from `first` up to `to` (the head's from one lower). Made ahead of the prompt (`Lot.ready`) where it can be.
    store: ?cuda.Buffer = null,
    first: usize = 0,
    /// When it was left, by the lot's clock: the oldest goes first.
    left: u64,
    /// Where the prefix it shares with other conversations ends (`marks.Request.kept`), when it holds the
    /// state there: a new conversation over the same harness starts from it.
    shared: ?usize = null,
    /// How long the prompt was whose reply ends the sequence (`Session.asked`): its states below that stand
    /// in the prompt, where a later turn of the conversation starts (`spill.pick`).
    asked: usize = 0,

    /// Holds a state of the sequence below every one held, or hands it back where there is no room or it
    /// is none (`Lot.keep` takes its buffers).
    pub fn keep(self: *Sequence, state: forward.Snapshot) ?forward.Snapshot {
        if (self.states.items.len == states or state.pos >= self.states.items[0].pos) return state;
        self.states.insertAssumeCapacity(0, state);
        return null;
    }

    /// The highest state that serves a prompt sharing `shared` tokens with the sequence: one at a position
    /// below `shared` (the token after it is the prompt's too).
    pub fn serves(self: Sequence, shared: usize) ?usize {
        var found: ?usize = null;
        for (self.states.items) |one| if (one.pos < shared) {
            found = one.pos;
        };
        return found;
    }

    pub fn bytes(self: Sequence) usize {
        var total: usize = 0;
        for (self.states.items) |one| total += one.bytes();
        if (self.store) |store| total += store.bytes();
        return total;
    }

    pub fn deinit(self: *Sequence, gpa: Allocator) void {
        if (self.store) |*store| store.deinit();
        gpa.free(self.spans);
        for (self.states.items) |*one| one.deinit();
        self.states.deinit(gpa);
        gpa.free(self.tokens);
        self.* = undefined;
    }
};

/// A sequence is left where the prompt parts this many tokens below its end or more: nearer, the prompt runs
/// those tokens again from the session's own states as soon.
pub const worth = 16;
/// The states a sequence left holds at most, the one at its end among them.
pub const states = 4;
/// Sequences that are the last to hold their shared prefix's state (`Lot.last`) held at most, outside
/// `Lot.most`: the shared cuts TensorFold 1.0.12's store marks (`core/prompt_cache.zig` `SHARED_KEYS`).
pub const harnesses = 64;

pub fn common(a: []const i32, b: []const i32) usize {
    return std.mem.indexOfDiff(i32, a, b) orelse @min(a.len, b.len);
}

pub const Lot = struct {
    gpa: Allocator,
    /// Sequences held at most of those that are not `last` (`Options.parked`); 0: none is ever parked.
    most: usize,
    /// Device bytes they hold at most (`--prompt-cache-gib`): past it one goes, by `victim`.
    cap: usize = std.math.maxInt(usize),
    list: std.ArrayList(Sequence) = .empty,
    clock: u64 = 0,
    /// Sequences taken up again, and left, since the session began.
    taken: u64 = 0,
    parked: u64 = 0,
    /// Of those left, the ones whose end's state took the buffers of a state that went.
    reused: u64 = 0,
    /// The states of sequences that went, `states` at most between replies: their buffers are the next state's taken, the
    /// lot's or the session's (`lend`), which would else be made while a prompt waits (spec 066).
    spare: shelf_file.Spares(forward.Snapshot) = .{},
    /// Sequences that went and are not freed yet (`done`).
    late: std.ArrayList(Sequence) = .empty,
    /// A store for the rows of the sequence the session holds, made after a reply (`prepare`) or left by a
    /// sequence that went (`done`): the next prompt that leaves that sequence holds its rows there, where else
    /// it would make the buffer while the prompt waits, 23 ms for 580 MiB (spec 101). Idle device bytes
    /// within `cap`, as `spare`'s are.
    ready: ?cuda.Buffer = null,
    /// Stores a sequence moved out of, freed with `late`.
    stores: std.ArrayList(cuda.Buffer) = .empty,
    /// Stores a lowering took from `ready`, and those it made on the prompt's path.
    housed_ready: u64 = 0,
    housed_made: u64 = 0,
    /// Sequences `prepare` moved out of a whole store into one over their own rows.
    compacted: u64 = 0,
    /// Device buffers the lowerings made, read from `cuda.account`: none once `ready` holds the store.
    lowering_made: usize = 0,
    /// Told of a sequence that goes for room, by `victim`, while the caches and it still hold its rows
    /// (`learnt.zig` writes a state of it to disk); not of one a longer sequence replaces or a prompt takes.
    leaving: ?Leaving = null,

    pub const Leaving = struct { context: *anyopaque, call: *const fn (*anyopaque, *const Sequence) void };

    /// The sequence `victim` names goes.
    fn evict(self: *Lot, at: usize) void {
        if (self.leaving) |told| told.call(told.context, &self.list.items[at]);
        self.drop(at);
    }

    pub fn deinit(self: *Lot) void {
        self.clear();
        self.list.deinit(self.gpa);
        self.spare.deinit(self.gpa);
        self.late.deinit(self.gpa);
        self.stores.deinit(self.gpa);
        if (self.ready) |*store| store.deinit();
    }

    /// A state no longer wanted: kept for its buffers, `states` of them past the reply's end (`sweep`).
    pub fn give(self: *Lot, state: forward.Snapshot) void {
        var one = state;
        if (self.most == 0) return one.deinit();
        self.spare.give(self.gpa, one);
    }

    /// A state no longer wanted, for the caller to write over.
    pub fn lend(self: *Lot) ?forward.Snapshot {
        return self.spare.lend();
    }

    /// `Sequence.keep`, the state it does not hold given for its buffers.
    pub fn keep(self: *Lot, one: *Sequence, state: forward.Snapshot) void {
        if (one.keep(state)) |over| self.give(over);
    }

    /// A sequence that goes, out of the lot already: its states' buffers stay, and its rows are freed once
    /// the reply has ended (`sweep`): freeing a long sequence's is 13 to 20 ms a prompt would wait.
    pub fn done(self: *Lot, one: *Sequence) void {
        for (one.states.items) |state| self.give(state);
        one.states.clearRetainingCapacity();
        // a store over the whole of its sequence may be the next one's; the copies that read it come first in
        // the stream
        if (one.store) |store| if (one.first == 0 and self.most > 0) {
            one.store = null;
            self.spareStore(store);
        };
        self.late.append(self.gpa, one.*) catch one.deinit(self.gpa);
        one.* = undefined;
    }

    /// Frees the sequences that went since the last call, and the states past `states` kept for their
    /// buffers. After a reply, when no client waits.
    pub fn sweep(self: *Lot) void {
        for (self.late.items) |*one| one.deinit(self.gpa);
        self.late.clearRetainingCapacity();
        for (self.stores.items) |*store| store.deinit();
        self.stores.clearRetainingCapacity();
        self.spare.trim(states);
    }

    fn drop(self: *Lot, at: usize) void {
        var gone = self.list.orderedRemove(at);
        self.done(&gone);
    }

    /// The state is no longer any parked sequence's below its `from`: none can be taken up.
    pub fn clear(self: *Lot) void {
        for (self.list.items) |*one| one.deinit(self.gpa);
        self.list.clearRetainingCapacity();
        self.sweep();
        self.spare.trim(0);
    }

    /// Device bytes the states kept for their buffers hold, and the store kept for the next sequence's rows;
    /// `bytes` and `cap` are the sequences' alone.
    pub fn spareBytes(self: *const Lot) usize {
        const ready: usize = if (self.ready) |store| store.bytes() else 0;
        return ready + self.spare.bytes();
    }

    /// A store no sequence uses any more, freed after the reply: a copy may read it still, and `cudaFree`
    /// waits for the device.
    fn retire(self: *Lot, store: cuda.Buffer) void {
        var gone = store;
        self.stores.append(self.gpa, gone) catch {
            cuda.synchronize() catch {};
            gone.deinit();
        };
    }

    /// The bytes a store of the rows from `first` up of a sequence ending at `to` takes, over `caches` and the
    /// head's, whose rows start one lower (`lower`).
    fn storeBytes(caches: []const attention.Cache, head: ?attention.Cache, ratio: usize, first: usize, to: usize) usize {
        var total: usize = 0;
        for (caches) |cache| total += Span.bytesOver(cache, ratio, first, to);
        if (head) |cache| total += Span.bytesOver(cache, ratio, first -| 1, to);
        return total;
    }

    /// After a reply, before the next prompt is taken (a request queued meanwhile waits for it). A sequence
    /// lowered into a whole store that holds under a quarter of it (it parted near its end, as over a long
    /// harness) moves into one over its own rows, and the whole store is kept; then `ready` holds the rows of
    /// the sequence the state stands at the end of, a quarter more for the turns to come, where the lot has
    /// room for it beside its sequences. A failure leaves the next lowering to make its store.
    pub fn prepare(self: *Lot, state: forward.State, head: ?attention.Cache, ratio: usize) void {
        if (self.most == 0 or (state.caches.len == 0 and head == null)) return;
        for (self.list.items, 0..) |*one, at| if (one.store != null and one.first == 0 and one.from * 4 > one.to) {
            const store = cuda.Buffer.init(.u8, &.{storeBytes(state.caches, head, ratio, one.from, one.to)}) catch continue;
            const whole = one.store.?;
            one.store = null;
            self.lay(one, state, head, store, one.from) catch {
                // some of its rows went with the store that failed: it goes, told to no one
                self.drop(at);
                self.spareStore(whole);
                return;
            };
            self.compacted += 1;
            self.spareStore(whole);
        };
        if (state.pos < worth) return;
        const need = storeBytes(state.caches, head, ratio, 0, state.pos);
        if (self.ready) |store| if (store.bytes() >= need) return;
        const room = self.cap -| self.bytes();
        const want = @min(need + need / 4, room);
        if (want < need) return;
        if (self.ready) |store| self.retire(store);
        self.ready = null;
        self.ready = cuda.Buffer.init(.u8, &.{want}) catch return;
    }

    /// A store no sequence uses: `ready` where it is larger than the one waiting and fits beside the lot's
    /// sequences, else retired.
    fn spareStore(self: *Lot, store: cuda.Buffer) void {
        if (self.bytes() + store.bytes() <= self.cap and (self.ready == null or self.ready.?.bytes() < store.bytes())) {
            if (self.ready) |old| self.retire(old);
            self.ready = store;
        } else self.retire(store);
    }

    /// Gives `one` a store over its rows from `from` up, before they are lowered there: `ready` when it holds
    /// the whole sequence, else one made for the rows from `from` alone (`prepare` moves a sequence that holds
    /// little of a whole store into one of its own).
    fn house(self: *Lot, one: *Sequence, state: forward.State, head: ?attention.Cache, from: usize) !void {
        // a model without attention holds no rows
        const ratio = if (one.spans.len > 0) one.spans[0].ratio else if (one.head) |span| span.ratio else return;
        if (self.ready) |ready| if (ready.bytes() >= storeBytes(state.caches, head, ratio, 0, one.to)) {
            self.ready = null;
            self.housed_ready += 1;
            return self.lay(one, state, head, ready, 0);
        };
        self.housed_made += 1;
        try self.lay(one, state, head, try .init(.u8, &.{@max(storeBytes(state.caches, head, ratio, from, one.to), 1)}), from);
    }

    /// `one`'s rows from `first` up laid out in `store`, what it held copied there; the store it had retired.
    fn lay(self: *Lot, one: *Sequence, state: forward.State, head: ?attention.Cache, store: cuda.Buffer, first: usize) !void {
        var owned = store;
        errdefer owned.deinit();
        var at: usize = 0;
        for (one.spans, state.caches) |*span, cache| try span.house(cache, store, &at, first);
        if (one.head) |*span| try span.house(head.?, store, &at, first -| 1);
        if (one.store) |old| self.retire(old);
        one.store = store;
        one.first = first;
    }

    pub fn bytes(self: *const Lot) usize {
        var total: usize = 0;
        for (self.list.items) |one| total += one.bytes();
        return total;
    }

    /// Leaves the sequence `known` the state stands at the end of, and returns it for the caller to `keep`
    /// the session's states of it in. Its rows are all in the caches still: `lower` holds what the next
    /// prompt will write over. A parked sequence that `known` begins with goes, `known` serving every prompt
    /// it would. Past `most` of those that are not the last to hold their shared prefix's state, the one of
    /// them left longest ago goes; past `harnesses` of those that are, the one of those.
    pub fn park(self: *Lot, state: forward.State, head: ?attention.Cache, ratio: usize, known: []const i32) !*Sequence {
        std.debug.assert(self.most > 0 and known.len == state.pos + 1);
        var i: usize = 0;
        while (i < self.list.items.len) {
            if (std.mem.startsWith(i32, known, self.list.items[i].tokens)) self.drop(i) else i += 1;
        }
        self.makeRoom();
        try self.list.ensureUnusedCapacity(self.gpa, 1);
        const tokens = try self.gpa.dupe(i32, known);
        errdefer self.gpa.free(tokens);
        const spans = try self.gpa.alloc(Span, state.caches.len);
        errdefer self.gpa.free(spans);
        for (spans) |*span| span.* = .init(ratio, state.pos);
        var held: std.ArrayList(forward.Snapshot) = try .initCapacity(self.gpa, states);
        errdefer held.deinit(self.gpa);
        const lent = self.spare.count() > 0;
        // into the buffers of a state that went, where one waits
        var end: forward.Snapshot = self.lend() orelse try .take(self.gpa, state);
        errdefer end.deinit();
        if (lent) try end.retake(state);
        self.reused += @intFromBool(lent);
        held.appendAssumeCapacity(end);
        self.clock += 1;
        self.parked += 1;
        self.list.appendAssumeCapacity(.{ .tokens = tokens, .from = state.pos, .to = state.pos, .states = held, .spans = spans, .head = if (head != null) .init(ratio, state.pos) else null, .left = self.clock });
        return &self.list.items[self.list.items.len - 1];
    }

    /// Room for one more sequence: past `most` of those that are not `last` the oldest of them goes, and past
    /// `harnesses` of those that are the oldest of those.
    pub fn makeRoom(self: *Lot) void {
        while (self.list.items.len - self.spared() >= self.most) self.evict(self.oldest(false).?);
        while (self.spared() >= harnesses) self.evict(self.oldest(true).?);
    }

    /// The sequence that goes first: the one left longest ago of those that are not the last to hold a shared
    /// prefix's state, and where every one is, the one left longest ago (TensorFold's `victimBut`: a shared
    /// state goes when nothing else can). A harness's state so outlives the conversations of others.
    pub fn victim(self: *const Lot) ?usize {
        return self.oldest(false) orelse self.oldest(true);
    }

    /// Sequence `at` holds the state where its shared prefix ends, and no sequence left after it holds that one.
    pub fn last(self: *const Lot, at: usize) bool {
        const one = self.list.items[at];
        const pos = one.shared orelse return false;
        for (self.list.items) |other| {
            if (other.left > one.left and other.shared == pos and std.mem.eql(i32, other.tokens[0 .. pos + 1], one.tokens[0 .. pos + 1])) return false;
        }
        return true;
    }

    /// The sequences that are `last`: outside `most`, as TensorFold's store counts no conversations and
    /// bounds its states by bytes alone (`--prompt-cache-gib`, `cap`), `harnesses` of them at most.
    pub fn spared(self: *const Lot) usize {
        var total: usize = 0;
        for (0..self.list.items.len) |at| total += @intFromBool(self.last(at));
        return total;
    }

    /// The sequence left longest ago of those that are `last`, or of those that are not.
    fn oldest(self: *const Lot, kept: bool) ?usize {
        var found: ?usize = null;
        for (self.list.items, 0..) |one, at| {
            if (self.last(at) != kept) continue;
            if (found == null or one.left < self.list.items[found.?].left) found = at;
        }
        return found;
    }

    /// Before `prompt` runs on the state: each sequence holds its rows from where it parts from the prompt,
    /// which the prompt and its reply may write over; then the lot is within `cap`. On a failure the lot is
    /// emptied.
    pub fn lower(self: *Lot, state: forward.State, head: ?attention.Cache, prompt: []const i32) !void {
        errdefer self.clear();
        const made = cuda.account.made_ever;
        defer self.lowering_made += cuda.account.made_ever - made;
        defer {
            while (self.bytes() > self.cap) self.evict(self.victim() orelse break);
            if (self.ready) |store| if (self.bytes() + store.bytes() > self.cap) {
                self.retire(store);
                self.ready = null;
            };
        }
        for (self.list.items) |*one| {
            const from = @min(common(one.tokens, prompt), one.to);
            if (from >= one.from) continue;
            if (one.store == null or from < one.first) try self.house(one, state, head, from);
            for (one.spans, state.caches) |*span, cache| try span.lower(cache, from);
            // the head's entry for a position was absorbed with the token after it, which is the prompt's own
            // from where the two part: one entry lower
            if (one.head) |*span| try span.lower(head.?, from -| 1);
            one.from = from;
        }
        // No wait: the copies are launches on the default stream, which the prompt's launches and graphs (on
        // streams made with `cudaStreamCreate`) follow, so the rows are held before any is written over.
    }

    /// The sequence with the highest state that serves `prompt`, above `above`.
    pub fn best(self: *const Lot, prompt: []const i32, above: usize) ?usize {
        var found: ?usize = null;
        var highest = above;
        for (self.list.items, 0..) |one, at| {
            const pos = one.serves(common(one.tokens, prompt)) orelse continue;
            if (pos > highest) {
                found = at;
                highest = pos;
            }
        }
        return found;
    }

    /// Puts sequence `at`'s rows back in the state's caches and the head's, and takes it out of the lot: the
    /// caller owns it, restores one of its states and frees it. After `lower`, and after the sequence the
    /// state held was parked if it is wanted again.
    pub fn take(self: *Lot, at: usize, state: forward.State, head: ?attention.Cache) !Sequence {
        const one = self.list.items[at];
        for (one.spans, state.caches) |span, cache| try span.restore(cache);
        if (one.head) |span| try span.restore(head.?);
        self.taken += 1;
        return self.list.orderedRemove(at);
    }
};
