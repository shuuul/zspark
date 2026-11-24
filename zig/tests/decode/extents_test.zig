//! `extents.zig`'s tests.

const std = @import("std");
const extents_file = @import("../../src/decode/extents.zig");
const Allocator = std.mem.Allocator;
const Chain = extents_file.Chain;
const Copy = extents_file.Copy;
const Counts = extents_file.Counts;
const Error = extents_file.Error;
const Pool = extents_file.Pool;
const Shrunk = extents_file.Shrunk;
const Source = extents_file.Source;
const Verdict = extents_file.Verdict;
const extentsFor = extents_file.extentsFor;
const lowerBound = extents_file.lowerBound;
const positions = extents_file.positions;
const shift = extents_file.shift;

const testing = std.testing;

/// Test tokens: `n` of them counting from `first`.
fn counting(buffer: []i32, first: usize) []const i32 {
    for (buffer, first..) |*t, i| t.* = @intCast(i);
    return buffer;
}

/// Grows a live chain to `len` tokens that count from its length on.
fn fill(pool: *Pool, chain: *Chain, len: usize) Error!void {
    var buffer: [8 * positions]i32 = undefined;
    try pool.grow(chain, len);
    pool.append(chain, counting(buffer[0 .. len - chain.len()], chain.len()));
}

fn expectCounts(pool: *const Pool, free: usize, live: usize, idle: usize) !void {
    try pool.check(testing.allocator);
    try testing.expectEqual(Counts{ .free = free, .live = live, .idle = idle }, pool.counts());
}

test "a pool under a context and an extent a slot is refused" {
    try testing.expectEqual(@as(usize, 160 + 4), lowerBound(81920, 4));
    try testing.expectEqual(@as(usize, 3 + 2), lowerBound(2 * positions + 1, 2));
    try testing.expectError(error.PoolTooSmall, Pool.init(testing.allocator, .{ .extents = 163, .capacity = 81920, .slots = 4 }));
    var pool: Pool = try .init(testing.allocator, .{ .extents = 164, .capacity = 81920, .slots = 4 });
    defer pool.deinit();
    try expectCounts(&pool, 164, 0, 0);
}

test "extents are taken until none is free, each once, and all come back" {
    const gpa = testing.allocator;
    var pool: Pool = try .init(gpa, .{ .extents = 7, .capacity = 4 * positions, .slots = 3 });
    defer pool.deinit();
    const a = try pool.open();
    const b = try pool.open();
    try expectCounts(&pool, 7, 0, 0);
    // a position is an extent's first row, and the last of an extent does not ask for the next
    try pool.grow(a, 1);
    try testing.expectEqualSlices(i32, &.{0}, a.table.items);
    try pool.grow(a, positions);
    try testing.expectEqual(@as(usize, 1), a.table.items.len);
    try pool.grow(a, positions + 1);
    try pool.grow(b, 3 * positions);
    try pool.grow(a, 4 * positions);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 5, 6 }, a.table.items);
    try testing.expectEqualSlices(i32, &.{ 2, 3, 4 }, b.table.items);
    try expectCounts(&pool, 0, 7, 0);
    pool.append(a, &.{ 7, 8, 9 });
    try testing.expectEqual(@as(usize, 4 * positions), a.reserved());
    try testing.expectEqual(@as(usize, 2), a.row(2));
    // none left, and a refusal changes nothing
    try testing.expectError(error.Short, pool.grow(b, 3 * positions + 1));
    try testing.expectError(error.PastCapacity, pool.grow(a, 4 * positions + 1));
    try testing.expectEqual(@as(usize, 3), b.table.items.len);
    try expectCounts(&pool, 0, 7, 0);
    // a retired chain gives back what it reserved past its tokens
    pool.retire(a);
    try testing.expectEqualSlices(i32, &.{0}, a.table.items);
    try expectCounts(&pool, 3, 3, 1);
    pool.release(b);
    try expectCounts(&pool, 6, 0, 1);
    pool.release(a);
    try expectCounts(&pool, 7, 0, 0);
    // and the seven are there to take again, each once
    const c = try pool.open();
    try pool.grow(c, 4 * positions);
    const d = try pool.open();
    try pool.grow(d, 3 * positions);
    var seen: [7]bool = @splat(false);
    for ([_][]const i32{ c.table.items, d.table.items }) |table| for (table) |e| {
        try testing.expect(!seen[@intCast(e)]);
        seen[@intCast(e)] = true;
    };
    try testing.expectError(error.Short, pool.grow(d, 4 * positions));
    try expectCounts(&pool, 0, 7, 0);
}

test "a fork names the source's whole extents again: in either order of release an extent is freed with its last name" {
    const gpa = testing.allocator;
    for ([_]bool{ false, true }) |source_first| {
        var pool: Pool = try .init(gpa, .{ .extents = 8, .capacity = 6 * positions, .slots = 2 });
        defer pool.deinit();
        const a = try pool.open();
        try fill(&pool, a, 3 * positions + 100);
        const forked = try pool.fork(a, 2 * positions);
        const b = forked.chain;
        try testing.expect(forked.copy == null and b.live);
        try testing.expectEqualSlices(i32, a.table.items[0..2], b.table.items);
        try testing.expectEqualSlices(i32, a.tokens.items[0 .. 2 * positions], b.tokens.items);
        try testing.expectEqualSlices(u32, &.{ 2, 2, 1, 1, 0, 0, 0, 0 }, pool.refs);
        try expectCounts(&pool, 4, 4, 0);
        // the fork goes on in an extent of its own, the source in its own
        try fill(&pool, b, 2 * positions + 50);
        try fill(&pool, a, 3 * positions + 200);
        try testing.expectEqual(@as(i32, 4), b.table.items[2]);
        try expectCounts(&pool, 3, 5, 0);
        if (source_first) {
            pool.release(a);
            // the shared two stay, the fork's still
            try testing.expectEqualSlices(u32, &.{ 1, 1, 0, 0, 1, 0, 0, 0 }, pool.refs);
            try testing.expectEqualSlices(i32, &.{ 0, 1, 4 }, b.table.items);
            try expectCounts(&pool, 5, 3, 0);
            pool.release(b);
        } else {
            pool.release(b);
            try testing.expectEqualSlices(u32, &.{ 1, 1, 1, 1, 0, 0, 0, 0 }, pool.refs);
            try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, a.table.items);
            try expectCounts(&pool, 4, 4, 0);
            pool.release(a);
        }
        try testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, pool.refs);
        try expectCounts(&pool, 8, 0, 0);
    }
}

test "a fork inside an extent copies the rows below its point into an extent of its own; at an extent's end nothing is copied" {
    const gpa = testing.allocator;
    var pool: Pool = try .init(gpa, .{ .extents = 8, .capacity = 4 * positions, .slots = 4 });
    defer pool.deinit();
    const a = try pool.open();
    try fill(&pool, a, 2 * positions + 300);

    // inside the third extent: two by name, and 137 rows to copy
    const inside = try pool.fork(a, 2 * positions + 137);
    try testing.expectEqual(Copy{ .from = 2, .to = 3, .rows = 137 }, inside.copy.?);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 3 }, inside.chain.table.items);
    try testing.expectEqual(@as(usize, 2 * positions + 137), inside.chain.len());
    try testing.expectEqualSlices(u32, &.{ 2, 2, 1, 1, 0, 0, 0, 0 }, pool.refs);

    // one position before an extent's end is inside it, its end is not
    const before = try pool.fork(a, 2 * positions - 1);
    try testing.expectEqual(Copy{ .from = 1, .to = 4, .rows = positions - 1 }, before.copy.?);
    try testing.expectEqualSlices(i32, &.{ 0, 4 }, before.chain.table.items);
    const at_end = try pool.fork(a, 2 * positions);
    try testing.expect(at_end.copy == null);
    try testing.expectEqualSlices(i32, &.{ 0, 1 }, at_end.chain.table.items);
    try testing.expectEqualSlices(u32, &.{ 4, 3, 1, 1, 1, 0, 0, 0 }, pool.refs);

    // at the source's own end, inside its last extent: copied too, since the source writes on there
    const whole = try pool.fork(a, a.len());
    try testing.expectEqual(Copy{ .from = 2, .to = 5, .rows = 300 }, whole.copy.?);
    // and of nothing: an empty chain
    const none = try pool.fork(a, 0);
    try testing.expect(none.copy == null and none.chain.table.items.len == 0 and none.chain.len() == 0);
    try expectCounts(&pool, 2, 6, 0);

    // every chain writes where it alone reads
    for ([_]*Chain{ a, inside.chain, before.chain, whole.chain }) |chain| {
        try testing.expectEqual(@as(u32, 1), pool.refs[@intCast(chain.table.items[chain.len() >> shift])]);
    }

    // a copy needs an extent: with none free the fork is refused and nothing has changed
    try pool.grow(none.chain, 2 * positions);
    try expectCounts(&pool, 0, 8, 0);
    const chains = pool.chains.items.len;
    try testing.expectError(error.Short, pool.fork(a, 5));
    try testing.expectEqual(chains, pool.chains.items.len);
    try expectCounts(&pool, 0, 8, 0);
    // at an extent's end it needs none
    const free_fork = try pool.fork(a, positions);
    try testing.expect(free_fork.copy == null);
    try expectCounts(&pool, 0, 8, 0);
}

test "an idle chain taken up again writes in place where it alone names the extent, and in a copy where another does" {
    const gpa = testing.allocator;
    var pool: Pool = try .init(gpa, .{ .extents = 8, .capacity = 5 * positions, .slots = 3 });
    defer pool.deinit();
    const a = try pool.open();
    try fill(&pool, a, 3 * positions + 100);
    pool.retire(a);
    try expectCounts(&pool, 4, 0, 4);

    // cut inside its third extent, which is its alone: no copy, the fourth is let go
    try testing.expectEqual(@as(?Copy, null), try pool.revive(a, 2 * positions + 10));
    try testing.expect(a.live);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2 }, a.table.items);
    try testing.expectEqual(@as(usize, 2 * positions + 10), a.len());
    try expectCounts(&pool, 5, 3, 0);

    // b reads a's first three extents whole; a, idle, is then cut inside the second
    try fill(&pool, a, 3 * positions);
    const b = (try pool.fork(a, 3 * positions)).chain;
    pool.retire(a);
    try testing.expectEqualSlices(u32, &.{ 2, 2, 2, 0, 0, 0, 0, 0 }, pool.refs);
    const copy = (try pool.revive(a, positions + 77)).?;
    try testing.expectEqual(Copy{ .from = 1, .to = 3, .rows = 77 }, copy);
    try testing.expectEqualSlices(i32, &.{ 0, 3 }, a.table.items);
    // b's rows are where they were
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2 }, b.table.items);
    try testing.expectEqualSlices(u32, &.{ 2, 1, 1, 1, 0, 0, 0, 0 }, pool.refs);
    try expectCounts(&pool, 4, 4, 0);

    // cut at an extent's end, shared or not, copies nothing
    pool.retire(a);
    try testing.expectEqual(@as(?Copy, null), try pool.revive(a, positions));
    try testing.expectEqualSlices(i32, &.{0}, a.table.items);
    try expectCounts(&pool, 5, 3, 0);
}

test "a chain taken up inside an extent another names is refused when no extent is free and its tail frees none" {
    const gpa = testing.allocator;
    var pool: Pool = try .init(gpa, .{ .extents = 4, .capacity = 2 * positions, .slots = 2 });
    defer pool.deinit();
    const a = try pool.open();
    try fill(&pool, a, 2 * positions);
    const b = (try pool.fork(a, 2 * positions)).chain;
    const c = try pool.open();
    try pool.grow(c, 2 * positions);
    pool.retire(a);
    try expectCounts(&pool, 0, 4, 0);
    try testing.expectError(error.Short, pool.revive(a, positions + 1));
    try testing.expect(!a.live and a.len() == 2 * positions and a.table.items.len == 2);
    try expectCounts(&pool, 0, 4, 0);
    try testing.expectEqualSlices(i32, a.table.items, b.table.items);

    // with a tail of its own past the cut, that extent is let go first and takes the copy
    var other: Pool = try .init(gpa, .{ .extents = 5, .capacity = 3 * positions, .slots = 2 });
    defer other.deinit();
    const d = try other.open();
    try fill(&other, d, 3 * positions);
    const e = (try other.fork(d, 2 * positions)).chain;
    const f = try other.open();
    try other.grow(f, 2 * positions);
    other.retire(d);
    try expectCounts(&other, 0, 4, 1);
    try testing.expectEqual(Copy{ .from = 1, .to = 2, .rows = 1 }, (try other.revive(d, positions + 1)).?);
    try testing.expectEqualSlices(i32, &.{ 0, 2 }, d.table.items);
    try testing.expectEqualSlices(i32, &.{ 0, 1 }, e.table.items);
    try expectCounts(&other, 0, 5, 0);
}

test "idle chains give up extents least recently used first and from the tail; a live stream's are never taken" {
    const gpa = testing.allocator;
    var pool: Pool = try .init(gpa, .{ .extents = 12, .capacity = 4 * positions, .slots = 3 });
    defer pool.deinit();
    const a = try pool.open();
    const b = try pool.open();
    const c = try pool.open();
    try fill(&pool, a, 2 * positions + 7);
    try fill(&pool, b, positions + 9);
    try fill(&pool, c, 3 * positions);
    // nothing is idle: nothing to take
    try testing.expectEqual(@as(?Shrunk, null), pool.evict(null));
    // retired in the order b, a: b is the older, though a was made first
    pool.retire(b);
    pool.retire(a);
    try expectCounts(&pool, 4, 3, 5);

    // b's tail, then b's head, each cut to what its extents hold
    var got = pool.evict(null).?;
    try testing.expect(got.chain == b and got.len == positions);
    var first: [positions]i32 = undefined;
    try testing.expectEqualSlices(i32, counting(&first, 0), b.tokens.items);
    got = pool.evict(null).?;
    try testing.expect(got.chain == b and got.len == 0);
    // then a from its tail; c, live, is never asked
    got = pool.evict(null).?;
    try testing.expect(got.chain == a and got.len == 2 * positions);
    try testing.expectEqualSlices(i32, &.{ 0, 1 }, a.table.items);
    try expectCounts(&pool, 7, 3, 2);

    // a fork from a makes it the newest: d, retired before, goes first
    const d = try pool.open();
    try fill(&pool, d, positions);
    pool.retire(d);
    const e = (try pool.fork(a, positions)).chain;
    got = pool.evict(null).?;
    try testing.expect(got.chain == d and got.len == 0);
    // a's second extent is its alone and is freed; its first is e's too and is not, though a lets it go
    const free = pool.counts().free;
    got = pool.evict(null).?;
    try testing.expect(got.chain == a and got.len == positions);
    try testing.expectEqual(free + 1, pool.counts().free);
    got = pool.evict(null).?;
    try testing.expect(got.chain == a and got.len == 0);
    try testing.expectEqual(free + 1, pool.counts().free);
    try testing.expectEqualSlices(i32, &.{0}, e.table.items);
    try testing.expectEqual(@as(u32, 1), pool.refs[0]);
    // every idle chain is empty; the live ones hold what they held
    try testing.expectEqual(@as(?Shrunk, null), pool.evict(null));
    try testing.expectEqual(@as(usize, 3 * positions), c.len());
    try testing.expectEqual(@as(usize, 3), c.table.items.len);
    try expectCounts(&pool, 8, 4, 0);

    // the chain spared is passed over, whatever its age
    pool.retire(c);
    pool.retire(e);
    try testing.expectEqual(e, pool.evict(c).?.chain);
    try testing.expectEqual(@as(?Shrunk, null), pool.evict(c));
    try testing.expectEqual(c, pool.evict(null).?.chain);
}

test "a request is admitted when the pool holds it beside the live streams, waits when it does not, and is refused past the pool" {
    const gpa = testing.allocator;
    const capacity = 4 * positions;
    // the lower bound: 4 extents and one a slot
    var pool: Pool = try .init(gpa, .{ .extents = 6, .capacity = capacity, .slots = 2 });
    defer pool.deinit();
    // an empty pool takes a whole context, and nothing longer
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, null));
    try testing.expectEqual(Verdict.never, pool.admission(capacity + 1, null));
    try testing.expectEqual(Verdict.admit, pool.admission(0, null));

    const a = try pool.open();
    try fill(&pool, a, 3 * positions + 1);
    // beside a live stream of four extents, two are left
    try testing.expectEqual(Verdict.admit, pool.admission(2 * positions, null));
    try testing.expectEqual(Verdict.wait, pool.admission(2 * positions + 1, null));
    // from a's first two extents by name, two more: its whole context
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, .{ .chain = a, .at = 2 * positions }));
    // inside the second, that one is a copy: three more
    try testing.expectEqual(Verdict.wait, pool.admission(capacity, .{ .chain = a, .at = 2 * positions - 1 }));
    try testing.expectEqual(Verdict.admit, pool.admission(3 * positions, .{ .chain = a, .at = 2 * positions - 1 }));

    // idle, a's extents can be evicted: a new request counts them as room
    pool.retire(a);
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, null));
    // but not those it will read: forked at a's second extent's end, four extents are left for two more
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, .{ .chain = a, .at = 2 * positions }));
    const b = try pool.open();
    try fill(&pool, b, 2 * positions);
    // b live on two, a idle on four: a's first two by name and its other two evicted for two more
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, .{ .chain = a, .at = 2 * positions }));
    // inside a's second extent the copy needs an extent while a is spared, and b has the rest
    try testing.expectEqual(Verdict.wait, pool.admission(capacity, .{ .chain = a, .at = 2 * positions - 1 }));
    try testing.expectEqual(Verdict.admit, pool.admission(2 * positions, .{ .chain = a, .at = 2 * positions }));
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, .{ .chain = a, .at = 3 * positions + 1, .in_place = true }));
    // a fork inside a's last extent needs one for the copy while a is spared, and nothing is free
    try testing.expectEqual(Verdict.wait, pool.admission(3 * positions + 1, .{ .chain = a, .at = 3 * positions + 1 }));
    // in place it needs none: the extent is a's alone
    try testing.expectEqual(Verdict.admit, pool.admission(3 * positions + 1, .{ .chain = a, .at = 3 * positions + 1, .in_place = true }));
    try testing.expectError(error.Short, pool.fork(a, 3 * positions + 1));
    try testing.expectEqual(@as(?Shrunk, null), pool.evict(a));

    // with everything else taken off, the oldest has its whole context
    pool.release(a);
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, .{ .chain = b, .at = 2 * positions }));
    try fill(&pool, b, capacity);
    try expectCounts(&pool, 2, 4, 0);
    // and each other slot an extent to start in
    try testing.expectEqual(Verdict.admit, pool.admission(2 * positions, null));
    try testing.expectEqual(Verdict.admit, pool.admission(positions + 1, .{ .chain = b, .at = 1 }));

    // the extents a request reads by name are not room for its others: an idle chain of two, a live one of
    // three, one free; from the idle chain's end a third extent fits and a fourth does not
    var other: Pool = try .init(gpa, .{ .extents = 6, .capacity = capacity, .slots = 2 });
    defer other.deinit();
    const x = try other.open();
    try fill(&other, x, 2 * positions);
    other.retire(x);
    try fill(&other, try other.open(), 3 * positions);
    try expectCounts(&other, 1, 3, 2);
    try testing.expectEqual(Verdict.admit, other.admission(3 * positions, .{ .chain = x, .at = 2 * positions }));
    try testing.expectEqual(Verdict.wait, other.admission(3 * positions + 1, .{ .chain = x, .at = 2 * positions }));
    try testing.expectEqual(Verdict.wait, other.admission(3 * positions + 1, .{ .chain = x, .at = 2 * positions, .in_place = true }));
    try testing.expectEqual(Verdict.admit, other.admission(3 * positions, null));
    try testing.expectEqual(Verdict.wait, other.admission(3 * positions + 1, null));
}

test "the owner's defaults: four streams of 262,144 positions fill a pool of 1,048,576, and its lower bound is 516 extents" {
    const gpa = testing.allocator;
    const capacity = 262144;
    const slots = 4;
    try testing.expectEqual(@as(usize, 2048), extentsFor(slots * capacity));
    try testing.expectEqual(@as(usize, 516), lowerBound(capacity, slots));
    try testing.expectError(error.PoolTooSmall, Pool.init(gpa, .{ .extents = 515, .capacity = capacity, .slots = slots }));
    var pool: Pool = try .init(gpa, .{ .extents = 2048, .capacity = capacity, .slots = slots });
    defer pool.deinit();
    var streams: [slots]*Chain = undefined;
    for (&streams, 0..) |*stream, i| {
        try testing.expectEqual(Verdict.admit, pool.admission(capacity, null));
        stream.* = try pool.open();
        try pool.grow(stream.*, capacity);
        try testing.expectEqual(@as(usize, 512), stream.*.table.items.len);
        try expectCounts(&pool, 2048 - 512 * (i + 1), 512 * (i + 1), 0);
    }
    // the last position of the last stream is the pool's last row of that extent
    try testing.expectEqual(@as(usize, 2047 * positions + 511), streams[3].row(capacity - 1));
    try testing.expectEqual(Verdict.wait, pool.admission(1, null));
    try testing.expectEqual(Verdict.never, pool.admission(capacity + 1, null));
    // a stream ends with 1,000 tokens: what it reserved comes back, and a whole context fits again by evicting it
    pool.release(streams[0]);
    streams[0] = try pool.open();
    try pool.grow(streams[0], capacity);
    for (0..1000) |i| pool.append(streams[0], &.{@intCast(i)});
    pool.retire(streams[0]);
    try expectCounts(&pool, 510, 1536, 2);
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, null));
    // forked at its token 1,000: its first extent by name, the second copied and then evicted, 510 free
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, .{ .chain = streams[0], .at = 1000 }));
    try testing.expectEqual(Verdict.admit, pool.admission(capacity, .{ .chain = streams[0], .at = 1000, .in_place = true }));
    // at the lower bound the oldest stream has its whole context once the other three are off, and each of
    // them an extent to start in
    var least: Pool = try .init(gpa, .{ .extents = 516, .capacity = capacity, .slots = slots });
    defer least.deinit();
    const oldest = try least.open();
    try least.grow(oldest, capacity);
    for (0..slots - 1) |_| try least.grow(try least.open(), 1);
    try testing.expectEqual(Verdict.wait, least.admission(positions + 1, null));
    try testing.expectEqual(Verdict.admit, least.admission(positions, null));
    try expectCounts(&least, 1, 515, 0);
}

/// The pool beside a model of it that knows no extent: what each chain's rows must read, which chains share
/// which stretch, and the order of use. `store` stands for the device's rows.
const Sim = struct {
    gpa: Allocator,
    pool: Pool,
    random: std.Random,
    /// `[extents * positions]`: what was last written at each row.
    store: []u64,
    mirrors: std.ArrayList(Mirror),
    clock: u64 = 0,
    stretches: u32 = 0,
    /// `in_place`: copies a chain taken up again made.
    stats: struct { admitted: usize = 0, waited: usize = 0, copies: usize = 0, in_place: usize = 0, evicted: usize = 0, rounds: usize = 0 } = .{},

    const Mirror = struct {
        chain: *Chain,
        tokens: std.ArrayList(i32),
        /// What each of its positions' rows hold.
        rows: std.ArrayList(u64),
        /// A name for each extent's worth of positions it holds, equal between two chains where they share.
        stretch: std.ArrayList(u32),
        live: bool,
        used: u64,
    };

    fn deinit(self: *Sim) void {
        for (self.mirrors.items) |*m| self.free(m);
        self.mirrors.deinit(self.gpa);
        self.gpa.free(self.store);
        self.pool.deinit();
    }

    fn free(self: *Sim, m: *Mirror) void {
        m.tokens.deinit(self.gpa);
        m.rows.deinit(self.gpa);
        m.stretch.deinit(self.gpa);
    }

    fn stamp(self: *Sim) u64 {
        self.clock += 1;
        return self.clock;
    }

    fn fresh(self: *Sim) u32 {
        self.stretches += 1;
        return self.stretches;
    }

    fn add(self: *Sim, chain: *Chain) !*Mirror {
        try self.mirrors.append(self.gpa, .{ .chain = chain, .tokens = .empty, .rows = .empty, .stretch = .empty, .live = true, .used = self.stamp() });
        return &self.mirrors.items[self.mirrors.items.len - 1];
    }

    fn remove(self: *Sim, i: usize) void {
        var m = self.mirrors.swapRemove(i);
        self.pool.release(m.chain);
        self.free(&m);
    }

    /// The device's copy, and the model's.
    fn copied(self: *Sim, copy: ?Copy, m: *Mirror, at: usize) !void {
        const c = copy orelse return;
        try testing.expectEqual(at % positions, c.rows);
        const from: usize = @intCast(c.from);
        const to: usize = @intCast(c.to);
        @memcpy(self.store[to * positions ..][0..c.rows], self.store[from * positions ..][0..c.rows]);
        m.stretch.items[at >> shift] = self.fresh();
        self.stats.copies += 1;
    }

    /// One eviction, held against the model's choice: the idle chain used longest ago, its last extent.
    fn evict(self: *Sim, spare: ?*const Chain) !bool {
        var want: ?*Mirror = null;
        for (self.mirrors.items) |*m| {
            if (m.live or m.chain == spare or m.rows.items.len == 0) continue;
            if (want == null or m.used < want.?.used) want = m;
        }
        const got = self.pool.evict(spare);
        const m = want orelse {
            try testing.expectEqual(@as(?Shrunk, null), got);
            return false;
        };
        const len = (extentsFor(m.rows.items.len) - 1) * positions;
        try testing.expectEqual(m.chain, got.?.chain);
        try testing.expectEqual(len, got.?.len);
        m.rows.shrinkRetainingCapacity(len);
        m.tokens.shrinkRetainingCapacity(len);
        _ = m.stretch.pop();
        self.stats.evicted += 1;
        return true;
    }

    /// A live chain reserves to `upto`, evicting as a caller would; false when the pool cannot give it.
    fn reserve(self: *Sim, m: *Mirror, upto: usize) !bool {
        while (true) {
            if (self.pool.grow(m.chain, upto)) break else |err| try testing.expectEqual(error.Short, err);
            if (!try self.evict(null)) return false;
        }
        while (m.stretch.items.len < m.chain.table.items.len) try m.stretch.append(self.gpa, self.fresh());
        return true;
    }

    /// Rows a window wrote and did not keep: anywhere past the chain's tokens in what it reserved.
    fn scribble(self: *Sim, m: *Mirror) void {
        const chain = m.chain;
        if (chain.reserved() == chain.len()) return;
        for (0..8) |_| self.store[chain.row(self.random.intRangeLessThan(usize, chain.len(), chain.reserved()))] = self.random.int(u64);
    }

    /// Commits `count` tokens: their rows are written and the model remembers what with.
    fn commit(self: *Sim, m: *Mirror, count: usize) !void {
        const chain = m.chain;
        var buffer: [64]i32 = undefined;
        var left = count;
        while (left > 0) {
            const n = @min(left, buffer.len);
            for (buffer[0..n], 0..) |*t, i| {
                t.* = self.random.int(i32);
                const value = self.random.int(u64);
                self.store[chain.row(chain.len() + i)] = value;
                try m.rows.append(self.gpa, value);
            }
            self.pool.append(chain, buffer[0..n]);
            try m.tokens.appendSlice(self.gpa, buffer[0..n]);
            left -= n;
        }
    }

    /// A request: asked of `admission`, then tried as the caller would run it, and the two must agree.
    fn admit(self: *Sim) !void {
        const capacity = self.pool.capacity;
        var source: ?Source = null;
        var from: ?usize = null;
        if (self.mirrors.items.len > 0 and self.random.uintLessThan(u8, 4) != 0) {
            const i = self.random.uintLessThan(usize, self.mirrors.items.len);
            const m = &self.mirrors.items[i];
            // often at an extent's end or beside it
            var at = self.random.uintAtMost(usize, m.rows.items.len);
            if (self.random.boolean()) at = @min(m.rows.items.len, (at & ~@as(usize, positions - 1)) + self.random.uintLessThan(usize, 2));
            source = .{ .chain = m.chain, .at = at, .in_place = !m.live and self.random.boolean() };
            from = i;
        }
        const at = if (source) |s| s.at else 0;
        const len = if (self.random.uintLessThan(u8, 16) == 0) capacity else self.random.intRangeAtMost(usize, at, capacity);
        const verdict = self.pool.admission(len, source);
        try testing.expect(verdict != .never);
        try testing.expectEqual(Verdict.never, self.pool.admission(capacity + 1 + self.random.uintLessThan(usize, 9), null));
        // a refusal is tried too, now and then: trying it evicts every idle chain
        if (verdict == .wait and self.random.uintLessThan(u8, 3) != 0) return;

        const spare: ?*const Chain = if (source) |s| s.chain else null;
        var held: ?usize = null;
        if (source) |s| {
            const m = &self.mirrors.items[from.?];
            if (s.in_place) {
                const copy = while (true) {
                    if (self.pool.revive(@constCast(s.chain), at)) |copy| break copy else |err| try testing.expectEqual(error.Short, err);
                    if (!try self.evict(spare)) break error.Short;
                } else unreachable;
                if (copy) |c| {
                    m.rows.shrinkRetainingCapacity(at);
                    m.tokens.shrinkRetainingCapacity(at);
                    m.stretch.shrinkRetainingCapacity(extentsFor(at));
                    m.live = true;
                    m.used = self.stamp();
                    // a copy only where another chain shares the stretch
                    var others = false;
                    for (self.mirrors.items) |*o| {
                        if (at % positions != 0 and o != m and o.stretch.items.len > at >> shift and o.stretch.items[at >> shift] == m.stretch.items[at >> shift]) others = true;
                    }
                    try testing.expectEqual(others, c != null);
                    self.stats.in_place += @intFromBool(c != null);
                    try self.copied(c, m, at);
                    held = from.?;
                } else |_| {}
            } else {
                const forked = while (true) {
                    if (self.pool.fork(@constCast(s.chain), at)) |forked| break forked else |err| try testing.expectEqual(error.Short, err);
                    if (!try self.evict(spare)) break error.Short;
                } else unreachable;
                if (forked) |f| {
                    try testing.expectEqual(at % positions != 0, f.copy != null);
                    m.used = self.stamp();
                    const tokens = m.tokens.items[0..at];
                    const rows = m.rows.items[0..at];
                    const stretch = m.stretch.items[0..extentsFor(at)];
                    // `add` may move the mirrors: the slices are copied first
                    var n: Mirror = .{ .chain = f.chain, .tokens = .empty, .rows = .empty, .stretch = .empty, .live = true, .used = self.stamp() };
                    errdefer self.free(&n);
                    try n.tokens.appendSlice(self.gpa, tokens);
                    try n.rows.appendSlice(self.gpa, rows);
                    try n.stretch.appendSlice(self.gpa, stretch);
                    try self.mirrors.append(self.gpa, n);
                    held = self.mirrors.items.len - 1;
                    try self.copied(f.copy, &self.mirrors.items[held.?], at);
                } else |_| {}
            }
        } else {
            _ = try self.add(try self.pool.open());
            held = self.mirrors.items.len - 1;
        }
        const done = if (held) |i| try self.reserve(&self.mirrors.items[i], len) else false;
        try testing.expectEqual(verdict == .admit, done);
        if (done) {
            self.stats.admitted += 1;
            const m = &self.mirrors.items[held.?];
            // a prompt's windows, some rows of each written twice
            self.scribble(m);
            try self.commit(m, len - at);
        } else {
            self.stats.waited += 1;
            // every idle chain was evicted for it, and it still did not fit: taken off again
            if (held) |i| self.remove(i);
        }
    }

    /// A reply's round on a live chain: a window of rows, some kept.
    fn round(self: *Sim, m: *Mirror) !void {
        const len = m.rows.items.len;
        const rows = @min(self.random.intRangeAtMost(usize, 1, 16), self.pool.capacity - len);
        if (rows == 0 or !try self.reserve(m, len + rows)) return;
        self.scribble(m);
        try self.commit(m, self.random.uintAtMost(usize, rows));
        self.scribble(m);
        self.stats.rounds += 1;
    }

    /// Every chain reads what was written for it, the model's shares are the pool's, and so are its counts.
    fn verify(self: *Sim) !void {
        const pool = &self.pool;
        try pool.check(self.gpa);
        // an extent no chain names may be written by anyone next
        for (pool.refs, 0..) |r, e| {
            if (r == 0) for (self.store[e * positions ..][0..positions]) |*v| {
                v.* = self.random.int(u64);
            };
        }
        try testing.expectEqual(self.mirrors.items.len, pool.chains.items.len);
        var by_stretch: std.AutoHashMap(u32, i32) = .init(self.gpa);
        defer by_stretch.deinit();
        var by_extent: std.AutoHashMap(i32, u32) = .init(self.gpa);
        defer by_extent.deinit();
        var live: std.AutoHashMap(u32, void) = .init(self.gpa);
        defer live.deinit();
        for (self.mirrors.items) |*m| {
            const chain = m.chain;
            try testing.expectEqual(m.live, chain.live);
            try testing.expectEqualSlices(i32, m.tokens.items, chain.tokens.items);
            try testing.expectEqual(m.stretch.items.len, chain.table.items.len);
            for (m.rows.items, 0..) |want, pos| try testing.expectEqual(want, self.store[chain.row(pos)]);
            for (m.stretch.items, chain.table.items) |s, e| {
                const a = try by_stretch.getOrPut(s);
                if (a.found_existing) try testing.expectEqual(a.value_ptr.*, e) else a.value_ptr.* = e;
                const b = try by_extent.getOrPut(e);
                if (b.found_existing) try testing.expectEqual(b.value_ptr.*, s) else b.value_ptr.* = s;
                if (m.live) try live.put(s, {});
            }
        }
        const named = by_stretch.count();
        try testing.expectEqual(Counts{ .free = pool.refs.len - named, .live = live.count(), .idle = named - live.count() }, pool.counts());
    }

    fn step(self: *Sim, slots: usize) !void {
        var lives: usize = 0;
        for (self.mirrors.items) |m| lives += @intFromBool(m.live);
        const count = self.mirrors.items.len;
        switch (self.random.uintLessThan(u8, 16)) {
            0...3 => if (lives < slots and count < 3 * slots) try self.admit(),
            4...9 => if (count > 0) {
                const m = &self.mirrors.items[self.random.uintLessThan(usize, count)];
                if (m.live) try self.round(m);
            },
            10, 11 => if (count > 0) {
                const m = &self.mirrors.items[self.random.uintLessThan(usize, count)];
                if (!m.live) return;
                self.pool.retire(m.chain);
                m.live = false;
                m.used = self.stamp();
                m.stretch.shrinkRetainingCapacity(extentsFor(m.rows.items.len));
            },
            12, 13 => if (count > 0) self.remove(self.random.uintLessThan(usize, count)),
            else => _ = try self.evict(null),
        }
    }
};

test "admits, forks, rounds, retirements, releases and evictions at random: every chain reads its own rows, and the counts are a model's" {
    const gpa = testing.allocator;
    const cases = [_]struct { seed: u64, capacity: usize, slots: usize, over: usize }{
        .{ .seed = 18, .capacity = 4 * positions, .slots = 3, .over = 0 },
        .{ .seed = 181, .capacity = 3 * positions + 40, .slots = 2, .over = 0 },
        .{ .seed = 1801, .capacity = 4 * positions, .slots = 4, .over = 5 },
        .{ .seed = 18001, .capacity = 6 * positions, .slots = 2, .over = 3 },
    };
    var waited: usize = 0;
    var in_place: usize = 0;
    for (cases) |case| {
        var prng: std.Random.DefaultPrng = .init(case.seed);
        const extents = lowerBound(case.capacity, case.slots) + case.over;
        var sim: Sim = .{
            .gpa = gpa,
            .pool = try .init(gpa, .{ .extents = extents, .capacity = case.capacity, .slots = case.slots }),
            .random = prng.random(),
            .store = try gpa.alloc(u64, extents * positions),
            .mirrors = .empty,
        };
        defer sim.deinit();
        @memset(sim.store, 0);
        var at: usize = 0;
        errdefer std.debug.print("seed {d}, step {d}\n", .{ case.seed, at });
        while (at < 3000) : (at += 1) {
            try sim.step(case.slots);
            try sim.verify();
        }
        // the sequence met what it is for
        try testing.expect(sim.stats.admitted > 100 and sim.stats.copies > 50 and sim.stats.evicted > 50 and sim.stats.rounds > 300);
        waited += sim.stats.waited;
        in_place += sim.stats.in_place;
        // everything let go, the pool is whole
        while (sim.mirrors.items.len > 0) sim.remove(0);
        try sim.verify();
        try testing.expectEqual(extents, sim.pool.counts().free);
    }
    // requests that had to wait were tried too and did not fit, and chains taken up again made copies
    try testing.expect(waited > 20 and in_place >= 5);
}

test "a pool sized by what was asked: every stream's whole context, the lower bound where that is more, and what fits a number of extents" {
    const wanted = extents_file.wanted;
    const slotsWithin = extents_file.slotsWithin;
    const capacityWithin = extents_file.capacityWithin;
    // the defaults', and spec 080's four streams of 81920
    try testing.expectEqual(@as(usize, 1_048_576), wanted(262144, 4) * positions);
    try testing.expectEqual(@as(usize, 327_680), wanted(81920, 4) * positions);
    try testing.expectEqual(@as(usize, 2_097_152), wanted(262144, 8) * positions);
    // a context that is no whole number of extents takes the next
    try testing.expectEqual(@as(usize, 4 * 161), wanted(82000, 4));
    // one extent a stream is under the lower bound, and so is one stream
    try testing.expectEqual(lowerBound(64, 4), wanted(64, 4));
    try testing.expectEqual(lowerBound(512, 64), wanted(512, 64));
    try testing.expectEqual(lowerBound(262144, 1), wanted(262144, 1));
    try testing.expectEqual(@as(usize, 4), wanted(1024, 2));
    for ([_]usize{ 64, 256, 512, 513, 4096, 81920, 262144 }) |capacity| {
        for (1..65) |slots| {
            const places = wanted(capacity, slots);
            try testing.expect(places >= lowerBound(capacity, slots) and places >= slots * extentsFor(capacity));
            // what is said to fit fits, and one more does not
            try testing.expectEqual(slots, slotsWithin(places, capacity, 64));
            try testing.expectEqual(slots - 1, slotsWithin(places - 1, capacity, 64));
            const longest = capacityWithin(places, slots);
            try testing.expect(longest >= extentsFor(capacity) * positions and wanted(longest, slots) <= places);
            try testing.expect(wanted(longest + 1, slots) > places);
            const shorter = capacityWithin(places - 1, slots);
            try testing.expect(shorter < extentsFor(capacity) * positions);
            if (shorter > 0) try testing.expect(wanted(shorter, slots) <= places - 1);
        }
    }
    // what the defaults' pool holds of eight streams, and nothing where a stream's own extent is not there
    try testing.expectEqual(@as(usize, 4), slotsWithin(2048, 262144, 8));
    try testing.expectEqual(@as(usize, 131072), capacityWithin(2048, 8));
    try testing.expectEqual(@as(usize, 0), slotsWithin(1, 64, 64));
    try testing.expectEqual(@as(usize, 0), capacityWithin(4, 4));
}
