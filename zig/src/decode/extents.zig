//! The attention caches' extents on the host (spec 018, the design's step B1): which extent of the pool holds
//! which positions of which sequence, and nothing of the device. An extent is 512 positions of every layer's
//! cache rows; the pool has a fixed number of them, made at the start. A chain is a token sequence with the table
//! of the extents its rows lie in (position >> 9 to extent): a stream's while it decodes (live), the prefix
//! store's after its request ended (idle). Nothing here is Flash Next's own, so the file stands in `src/`: a
//! second model's caches are laid out by the same tables.
//!
//! What the pool keeps true, and `check` verifies:
//!
//! - An extent's count is the number of chains whose table names it; at zero it is free.
//! - An extent has one writer at most: the live chain whose committed length ends in it, or that reserved it for
//!   a window's rows. Such an extent is named by that chain alone. Every other name of an extent is a reader's,
//!   of rows that are never written again while the name stands.
//! - So a fork shares the extents wholly below its point by name and is told to copy the rows of the one the
//!   point lies in (`Copy`); an idle chain taken up again in place (`revive`) copies only where that extent
//!   has another name.
//! - Extents are taken back from idle chains only, least recently used chain first, from its tail (`evict`). An
//!   extent a live chain names is never freed by it.
//!
//! No call here evicts by itself: one that finds the pool short returns `error.Short` and has changed nothing,
//! and the caller evicts (it owns the snapshots a shortened chain loses), then takes a request off.
//!
//! Memory: a chain's tokens and table are allocated whole when it is made (an admission, never a round), so
//! `grow` and `append` allocate nothing.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

/// Positions an extent holds: the attend kernel's chunk of keys (`models/qwen4/blocks/attention.zig`'s `chunk`).
pub const positions = 512;
/// `position >> shift` is the index of its extent in a chain's table.
pub const shift = 9;

comptime {
    assert(1 << shift == positions);
}

/// Extents that hold `len` positions.
pub fn extentsFor(len: usize) usize {
    return (len + positions - 1) >> shift;
}

/// The fewest extents a pool serving `slots` streams of `capacity` positions may have: a whole context, and one
/// extent a slot. With it the oldest request can always finish once every other is taken off, and a fork's
/// one copied extent can always be found beside the chain it copies from.
pub fn lowerBound(capacity: usize, slots: usize) usize {
    return extentsFor(capacity) + slots;
}

/// The extents of a pool sized by what was asked (spec 082): `slots` streams of `capacity` positions each hold
/// their whole context at once, and no fewer than `lowerBound`.
pub fn wanted(capacity: usize, slots: usize) usize {
    return @max(lowerBound(capacity, slots), slots * extentsFor(capacity));
}

/// The most streams of `capacity` positions, `most` at most, whose `wanted` pool is within `places` extents;
/// 0 where one stream's is not.
pub fn slotsWithin(places: usize, capacity: usize, most: usize) usize {
    var slots = most;
    while (slots > 0 and wanted(capacity, slots) > places) slots -= 1;
    return slots;
}

/// The most positions a stream may hold for the `wanted` pool of `slots` streams to be within `places`
/// extents, a whole number of extents; 0 where none is.
pub fn capacityWithin(places: usize, slots: usize) usize {
    return @min(places -| slots, places / @max(slots, 1)) << shift;
}

pub const Error = error{
    /// No free extent: evict idle chains, then take a request off.
    Short,
    /// Past the positions a chain may hold.
    PastCapacity,
} || Allocator.Error;

/// Rows to copy on the device, in every layer, before the next call that frees an extent: the first `rows`
/// rows of extent `from` into extent `to`.
pub const Copy = struct {
    from: i32,
    to: i32,
    rows: usize,
};

pub const Chain = struct {
    /// The tokens whose cache rows the chain holds: token p's rows are at `row(p)`. Capacity: the pool's.
    tokens: std.ArrayList(i32),
    /// The extent of positions `[512 i, 512 i + 512)`, as the device's table takes it (int32). An idle chain
    /// has the extents of its tokens and no more; a live one may have reserved some past them.
    table: std.ArrayList(i32),
    /// A stream decodes on it: it may write, and none of its extents is taken back.
    live: bool,
    /// When it was last made, forked from, retired or revived, by the pool's clock.
    used: u64,
    /// Where the pool lists it.
    index: usize,

    pub fn len(self: *const Chain) usize {
        return self.tokens.items.len;
    }

    /// Positions its extents can hold.
    pub fn reserved(self: *const Chain) usize {
        return self.table.items.len << shift;
    }

    /// The pool's row of position `pos`: `stored_at` of `zig/kernels/common/attention.cuh`.
    pub fn row(self: *const Chain, pos: usize) usize {
        return (@as(usize, @intCast(self.table.items[pos >> shift])) << shift) + pos % positions;
    }
};

/// Extents by who names them; the three sum to the pool's.
pub const Counts = struct {
    free: usize,
    /// Named by a live chain at least.
    live: usize,
    /// Named by idle chains alone: what evicting every idle chain frees.
    idle: usize,
};

/// Where a request's chain starts from.
pub const Source = struct {
    chain: *const Chain,
    /// The tokens of `chain` the request begins with.
    at: usize,
    /// `revive` (the idle chain itself goes on, cut to `at`) and not `fork`.
    in_place: bool = false,
};

pub const Verdict = enum {
    /// The pool holds it once idle chains are evicted.
    admit,
    /// Not beside the live streams: it waits first in line.
    wait,
    /// More than a chain may hold or than the pool has: refused.
    never,
};

/// What `evict` took an extent from, and the tokens it has left.
pub const Shrunk = struct {
    chain: *Chain,
    len: usize,
};

pub const Pool = struct {
    gpa: Allocator,
    /// Positions a chain may hold.
    capacity: usize,
    /// Chains naming each extent: `[extents]`.
    refs: []u32,
    /// Live chains among them: `[extents]`.
    live: []u32,
    /// The free extents, `free[0..free_count]`, taken from the end: `[extents]`.
    free: []i32,
    free_count: usize,
    /// Extents a live chain names.
    live_count: usize = 0,
    chains: std.ArrayList(*Chain),
    clock: u64 = 0,

    pub const Options = struct {
        /// The pool's size (`--cache-tokens` as `extentsFor` of it).
        extents: usize,
        /// Positions a stream may hold.
        capacity: usize,
        /// Streams decoded at once.
        slots: usize,
    };

    pub const InitError = error{PoolTooSmall} || Allocator.Error;

    /// Refuses a pool under `lowerBound`.
    pub fn init(gpa: Allocator, options: Options) InitError!Pool {
        if (options.extents < lowerBound(options.capacity, options.slots)) return error.PoolTooSmall;
        const refs = try gpa.alloc(u32, options.extents);
        errdefer gpa.free(refs);
        const live = try gpa.alloc(u32, options.extents);
        errdefer gpa.free(live);
        const free = try gpa.alloc(i32, options.extents);
        errdefer gpa.free(free);
        @memset(refs, 0);
        @memset(live, 0);
        // taken from the end: the first extents asked for are 0, 1, 2
        for (free, 0..) |*e, i| e.* = @intCast(options.extents - 1 - i);
        return .{
            .gpa = gpa,
            .capacity = options.capacity,
            .refs = refs,
            .live = live,
            .free = free,
            .free_count = options.extents,
            .chains = .empty,
        };
    }

    /// Chains not released are freed with it.
    pub fn deinit(self: *Pool) void {
        for (self.chains.items) |chain| self.destroy(chain);
        self.chains.deinit(self.gpa);
        self.gpa.free(self.free);
        self.gpa.free(self.live);
        self.gpa.free(self.refs);
    }

    pub fn counts(self: *const Pool) Counts {
        return .{ .free = self.free_count, .live = self.live_count, .idle = self.refs.len - self.free_count - self.live_count };
    }

    /// A live chain without tokens or extents.
    pub fn open(self: *Pool) Error!*Chain {
        return self.create();
    }

    /// A live chain of `source`'s first `at` tokens. The extents wholly below `at` are `source`'s, named again;
    /// when `at` lies inside an extent the new chain gets one of its own there and the rows to copy into it are
    /// returned. `source` may be live or idle and is left as it was.
    pub fn fork(self: *Pool, source: *Chain, at: usize) Error!struct { chain: *Chain, copy: ?Copy } {
        assert(at <= source.len());
        const whole = at >> shift;
        const partial = at % positions;
        if (partial != 0 and self.free_count == 0) return error.Short;
        const chain = try self.create();
        for (source.table.items[0..whole]) |e| {
            self.name(e, true);
            chain.table.appendAssumeCapacity(e);
        }
        var copy: ?Copy = null;
        if (partial != 0) {
            const to = self.take();
            chain.table.appendAssumeCapacity(to);
            copy = .{ .from = source.table.items[whole], .to = to, .rows = partial };
        }
        chain.tokens.appendSliceAssumeCapacity(source.tokens.items[0..at]);
        source.used = self.tick();
        chain.used = self.tick();
        return .{ .chain = chain, .copy = copy };
    }

    /// An idle chain goes on as a stream's from its first `at` tokens; the tokens and extents past them are let
    /// go. Where `at` lies inside an extent that another chain names too, the chain gets one of its own there
    /// and the rows to copy are returned; where the extent is its alone, nothing is copied.
    pub fn revive(self: *Pool, chain: *Chain, at: usize) Error!?Copy {
        assert(!chain.live and at <= chain.len());
        const keep = extentsFor(at);
        const partial = at % positions;
        const table = &chain.table;
        const shared = partial != 0 and self.refs[@intCast(table.items[keep - 1])] > 1;
        if (shared and self.free_count == 0) {
            // its own tail may be what frees one
            for (table.items[keep..]) |e| {
                if (self.refs[@intCast(e)] == 1) break;
            } else return error.Short;
        }
        while (table.items.len > keep) self.drop(table.pop().?, false);
        chain.tokens.shrinkRetainingCapacity(at);
        for (table.items) |e| self.enliven(e);
        chain.live = true;
        chain.used = self.tick();
        if (!shared) return null;
        const from = table.items[keep - 1];
        const to = self.take();
        table.items[keep - 1] = to;
        self.drop(from, true);
        return .{ .from = from, .to = to, .rows = partial };
    }

    /// Extents for a live chain's positions below `upto`: a window's rows are written before any is committed.
    pub fn grow(self: *Pool, chain: *Chain, upto: usize) Error!void {
        assert(chain.live);
        if (upto > self.capacity) return error.PastCapacity;
        const want = extentsFor(upto);
        if (want <= chain.table.items.len) return;
        if (want - chain.table.items.len > self.free_count) return error.Short;
        while (chain.table.items.len < want) chain.table.appendAssumeCapacity(self.take());
    }

    /// The tokens a live chain committed; their rows are in extents `grow` gave.
    pub fn append(self: *Pool, chain: *Chain, tokens: []const i32) void {
        assert(chain.live and chain.len() + tokens.len <= chain.reserved());
        // the one writer of what it writes
        if (tokens.len > 0) for (chain.table.items[chain.len() >> shift .. extentsFor(chain.len() + tokens.len)]) |e| assert(self.refs[@intCast(e)] == 1);
        chain.tokens.appendSliceAssumeCapacity(tokens);
    }

    /// A stream's chain becomes the prefix store's: the extents reserved past its tokens are let go, and the
    /// rest may be evicted.
    pub fn retire(self: *Pool, chain: *Chain) void {
        assert(chain.live);
        const keep = extentsFor(chain.len());
        while (chain.table.items.len > keep) self.drop(chain.table.pop().?, true);
        for (chain.table.items) |e| self.deaden(e);
        chain.live = false;
        chain.used = self.tick();
    }

    /// Lets go of every extent of a chain, live (a request taken off) or idle, and frees it.
    pub fn release(self: *Pool, chain: *Chain) void {
        for (chain.table.items) |e| self.drop(e, chain.live);
        const last = self.chains.pop().?;
        if (last != chain) {
            self.chains.items[chain.index] = last;
            last.index = chain.index;
        }
        self.destroy(chain);
    }

    /// Drops the last extent of one idle chain and the tokens that extent held. The extent is free again
    /// unless another chain names it. `evict` is this on the least recently used chain; the prefix store calls
    /// it on the chain it chose, which is how a shared tail is left in place.
    pub fn shrink(self: *Pool, chain: *Chain) Shrunk {
        assert(!chain.live and chain.table.items.len != 0);
        self.drop(chain.table.pop().?, false);
        chain.tokens.shrinkRetainingCapacity(@min(chain.len(), chain.reserved()));
        return .{ .chain = chain, .len = chain.len() };
    }

    /// An idle chain keeps its first `len` tokens. Extents past those tokens are let go; the extent `len`
    /// lies in stays, and a later writer copies it when another chain names it.
    pub fn trim(self: *Pool, chain: *Chain, len: usize) void {
        assert(!chain.live and len <= chain.len());
        chain.tokens.shrinkRetainingCapacity(len);
        const keep = extentsFor(len);
        while (chain.table.items.len > keep) self.drop(chain.table.pop().?, false);
    }

    /// Takes the last extent of the idle chain least recently used, `spare` (the chain about to be forked from
    /// or revived) left alone, and says which chain and what it has left: its snapshots past that are the
    /// caller's to drop. The extent is free again unless another chain names it. Null when no idle chain but
    /// `spare` has an extent. A tail another chain names is still taken: the chain shortens and nothing is
    /// freed. The prefix store skips those tails instead of calling this.
    pub fn evict(self: *Pool, spare: ?*const Chain) ?Shrunk {
        var oldest: ?*Chain = null;
        for (self.chains.items) |chain| {
            if (chain.live or chain == spare or chain.table.items.len == 0) continue;
            if (oldest == null or chain.used < oldest.?.used) oldest = chain;
        }
        const chain = oldest orelse return null;
        return self.shrink(chain);
    }

    /// Whether a request of `len` positions that starts from `source` can be held beside the live chains. It
    /// can when `fork` or `revive` and then `grow` to `len` succeed after idle chains are evicted with `evict`
    /// (the source spared until it is forked or revived), and it cannot when they do not.
    pub fn admission(self: *const Pool, len: usize, source: ?Source) Verdict {
        const want = extentsFor(len);
        if (len > self.capacity or want > self.refs.len) return .never;
        const now = self.counts();
        // what evicting every idle chain would leave to take
        var room = now.free + now.idle;
        var kept: usize = 0;
        if (source) |s| {
            assert(s.at <= len and s.at <= s.chain.len() and !(s.in_place and s.chain.live));
            const table = s.chain.table.items;
            const partial = s.at % positions != 0;
            kept = s.at >> shift;
            // an extent of its own for the rows of the one `at` lies in
            var copied = partial;
            // idle extents that are not evicted while the copy waits for its extent
            var spared: usize = 0;
            if (s.in_place) {
                const last = table[s.at >> shift .. extentsFor(s.at)];
                copied = partial and self.refs[@intCast(last[0])] > 1;
                if (partial and !copied) kept += 1;
                for (table[0..kept]) |e| spared += @intFromBool(self.live[@intCast(e)] == 0);
                if (copied) spared += @intFromBool(self.live[@intCast(last[0])] == 0);
            } else {
                for (table) |e| spared += @intFromBool(self.live[@intCast(e)] == 0);
            }
            if (copied and room - spared == 0) return .wait;
            // the extents kept by name become a live chain's
            for (table[0..kept]) |e| room -= @intFromBool(self.live[@intCast(e)] == 0);
        }
        return if (want - kept <= room) .admit else .wait;
    }

    pub const Inconsistent = error{Inconsistent} || Allocator.Error;

    /// Counts every extent again from the chains' tables and holds the pool's own figures and the rules of this
    /// file's head against them. For tests and for a debug build's rounds.
    pub fn check(self: *const Pool, gpa: Allocator) Inconsistent!void {
        const n = self.refs.len;
        const refs = try gpa.alloc(u32, 2 * n);
        defer gpa.free(refs);
        @memset(refs, 0);
        const live = refs[n..];
        for (self.chains.items, 0..) |chain, i| {
            if (chain.index != i) return error.Inconsistent;
            const held = extentsFor(chain.len());
            const table = chain.table.items;
            if (chain.len() > self.capacity or table.len < held or (!chain.live and table.len != held)) return error.Inconsistent;
            for (table, 0..) |e, at| {
                if (e < 0 or e >= n) return error.Inconsistent;
                // a chain names an extent once
                if (std.mem.indexOfScalar(i32, table[0..at], e) != null) return error.Inconsistent;
                refs[@intCast(e)] += 1;
                live[@intCast(e)] += @intFromBool(chain.live);
            }
            // one writer: where a live chain writes, no other chain reads
            if (chain.live) for (table[chain.len() >> shift ..]) |e| {
                if (self.refs[@intCast(e)] != 1) return error.Inconsistent;
            };
        }
        if (!std.mem.eql(u32, refs[0..n], self.refs) or !std.mem.eql(u32, live, self.live)) return error.Inconsistent;
        var named: usize = 0;
        var by_live: usize = 0;
        for (refs[0..n], live) |r, l| {
            named += @intFromBool(r != 0);
            by_live += @intFromBool(l != 0);
        }
        if (self.free_count != n - named or self.live_count != by_live) return error.Inconsistent;
        // the free list is the extents no chain names, each once
        @memset(live, 0);
        for (self.free[0..self.free_count]) |e| {
            if (e < 0 or e >= n or refs[@intCast(e)] != 0 or live[@intCast(e)] != 0) return error.Inconsistent;
            live[@intCast(e)] = 1;
        }
    }

    fn tick(self: *Pool) u64 {
        self.clock += 1;
        return self.clock;
    }

    fn create(self: *Pool) Error!*Chain {
        const gpa = self.gpa;
        try self.chains.ensureUnusedCapacity(gpa, 1);
        const chain = try gpa.create(Chain);
        errdefer gpa.destroy(chain);
        var tokens: std.ArrayList(i32) = try .initCapacity(gpa, self.capacity);
        errdefer tokens.deinit(gpa);
        const table: std.ArrayList(i32) = try .initCapacity(gpa, extentsFor(self.capacity));
        chain.* = .{ .tokens = tokens, .table = table, .live = true, .used = self.tick(), .index = self.chains.items.len };
        self.chains.appendAssumeCapacity(chain);
        return chain;
    }

    fn destroy(self: *Pool, chain: *Chain) void {
        chain.tokens.deinit(self.gpa);
        chain.table.deinit(self.gpa);
        self.gpa.destroy(chain);
    }

    /// A free extent, named by one live chain.
    fn take(self: *Pool) i32 {
        self.free_count -= 1;
        const e = self.free[self.free_count];
        assert(self.refs[@intCast(e)] == 0);
        self.refs[@intCast(e)] = 1;
        self.live[@intCast(e)] = 1;
        self.live_count += 1;
        return e;
    }

    /// One more chain names `e`.
    fn name(self: *Pool, e: i32, live: bool) void {
        self.refs[@intCast(e)] += 1;
        if (live) self.enliven(e);
    }

    /// One chain fewer names `e`; the last one frees it.
    fn drop(self: *Pool, e: i32, live: bool) void {
        if (live) self.deaden(e);
        self.refs[@intCast(e)] -= 1;
        if (self.refs[@intCast(e)] != 0) return;
        self.free[self.free_count] = e;
        self.free_count += 1;
    }

    /// A chain that names `e` became live.
    fn enliven(self: *Pool, e: i32) void {
        self.live_count += @intFromBool(self.live[@intCast(e)] == 0);
        self.live[@intCast(e)] += 1;
    }

    /// A chain that names `e` is live no more.
    fn deaden(self: *Pool, e: i32) void {
        self.live[@intCast(e)] -= 1;
        self.live_count -= @intFromBool(self.live[@intCast(e)] == 0);
    }
};
