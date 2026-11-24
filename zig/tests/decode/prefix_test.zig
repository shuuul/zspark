//! `prefix.zig`'s tests.

const std = @import("std");
const extents = @import("../../src/decode/extents.zig");
const prefix = @import("../../src/decode/prefix.zig");
const Chain = extents.Chain;
const Copy = extents.Copy;
const Pool = extents.Pool;
const Shrunk = extents.Shrunk;
const Verdict = extents.Verdict;
const Store = prefix.Store;
const round_positions = prefix.round_positions;

const testing = std.testing;

fn write(pool: *Pool, chain: *Chain, n: usize) !void {
    try pool.grow(chain, n);
    var buf: [2048]i32 = undefined;
    for (buf[0..n], 0..) |*token, i| token.* = @intCast(i + 1);
    pool.append(chain, buf[0..n]);
}

fn has(store: *const Store, chain: *const Chain, pos: usize) bool {
    for (store.marks.items) |mark| if (mark.chain == chain and mark.pos == pos) return true;
    return false;
}

test "a fork inside an extent returns the copy and nothing frees the source until it has run" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, .{ .extents = 6, .capacity = 2048, .slots = 2 });
    defer pool.deinit();
    var store = try Store.init(gpa, &pool, 2, round_positions);
    defer store.deinit();
    const source = try store.open(0);
    try write(&pool, source, 649);
    const forked = try store.fork(source, 649);
    try testing.expect(forked.copy != null);
    try testing.expectEqual(@as(usize, 137), forked.copy.?.rows);
    try testing.expectEqual(source.table.items[0], forked.chain.table.items[0]);
    try testing.expect(source.table.items[1] != forked.chain.table.items[1]);
    try testing.expectEqual(@as(usize, 2), pool.refs[@intCast(source.table.items[0])]);
    try testing.expectEqual(@as(usize, 1), pool.refs[@intCast(source.table.items[1])]);
    try testing.expectEqual(@as(usize, 649), source.len());
    try testing.expectEqualSlices(i32, source.tokens.items[0..649], forked.chain.tokens.items);
    try testing.expectError(error.CopyPending, store.evict(null));
    try testing.expectError(error.CopyPending, store.release(source));
    store.copied();
    try testing.expectEqual(@as(?Shrunk, null), try store.evict(null));
    try pool.check(gpa);
}

test "snapshots past a shortened chain are dropped and one at the new end stays" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, .{ .extents = 3, .capacity = 1024, .slots = 1 });
    defer pool.deinit();
    var store = try Store.init(gpa, &pool, 1, round_positions);
    defer store.deinit();
    const chain = try store.open(0);
    try write(&pool, chain, 600);
    try store.note(chain, 100, .chunk);
    try store.note(chain, 400, .chunk);
    try store.note(chain, 512, .chunk);
    try store.note(chain, 600, .chunk);
    try store.preempt(0);
    const once = (try store.evict(null)).?;
    try testing.expectEqual(chain, once.chain);
    try testing.expectEqual(@as(usize, 512), once.len);
    try testing.expect(!has(&store, chain, 600));
    try testing.expect(has(&store, chain, 512));
    try testing.expect(has(&store, chain, 400));
    try testing.expect(has(&store, chain, 100));
    const twice = (try store.evict(null)).?;
    try testing.expectEqual(@as(usize, 0), twice.len);
    try testing.expectEqual(@as(usize, 0), store.marks.items.len);
    try pool.check(gpa);
}

test "eviction skips a shared tail and frees a private one" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, .{ .extents = 4, .capacity = 1024, .slots = 2 });
    defer pool.deinit();
    var store = try Store.init(gpa, &pool, 2, round_positions);
    defer store.deinit();
    const source = try store.open(0);
    try write(&pool, source, 512);
    try store.preempt(0);
    const forked = try store.fork(source, 512);
    try testing.expectEqual(@as(?Copy, null), forked.copy);
    store.place(0, forked.chain);
    const other = try store.open(1);
    try write(&pool, other, 512);
    try store.preempt(1);
    const before = pool.free_count;
    const shrunk = (try store.evict(null)).?;
    try testing.expectEqual(other, shrunk.chain);
    try testing.expectEqual(@as(usize, 0), other.len());
    try testing.expectEqual(@as(usize, 512), source.len());
    try testing.expectEqual(@as(usize, 2), pool.refs[@intCast(source.table.items[0])]);
    try testing.expect(pool.free_count == before + 1);
    // Nothing private is left: the shared tail stays, and the call frees nothing.
    try testing.expectEqual(@as(?Shrunk, null), try store.evict(null));
    try testing.expectEqual(before + 1, pool.free_count);
    try pool.check(gpa);
}

test "admission's length is the prompt, and a round's room is left beside it" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, .{ .extents = 3, .capacity = 1024, .slots = 1 });
    defer pool.deinit();
    var store = try Store.init(gpa, &pool, 1, round_positions);
    defer store.deinit();
    // 1020 fits. 1020 plus a round does not, and the round is not part of the length, so it is admitted.
    // 1025 is past the context.
    try testing.expectEqual(Verdict.admit, store.admission(1020, null));
    try testing.expectEqual(Verdict.admit, store.admission(600, null));
    try testing.expectEqual(Verdict.never, store.admission(1025, null));
    try testing.expectEqual(Verdict.never, pool.admission(1100, null));

    var tight = try Pool.init(gpa, .{ .extents = 4, .capacity = 1024, .slots = 2 });
    defer tight.deinit();
    var full = try Store.init(gpa, &tight, 2, round_positions);
    defer full.deinit();
    const a = try full.open(0);
    try write(&tight, a, 512);
    const b = try full.open(1);
    try write(&tight, b, 512);
    // The prompt of a whole context fits once the free extents are taken. Both streams sit on an extent
    // boundary, so a round needs an extent each and there is none left beside the prompt.
    try testing.expectEqual(Verdict.admit, tight.admission(1024, null));
    try testing.expectEqual(Verdict.wait, full.admission(1024, null));
    try testing.expectEqual(Verdict.wait, full.admission(100, null));
    try tight.check(gpa);
}

test "the snapshot bound drops an idle chain's mark before a live one's" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, .{ .extents = 3, .capacity = 1024, .slots = 1 });
    defer pool.deinit();
    var store = try Store.init(gpa, &pool, 1, round_positions);
    defer store.deinit();
    try testing.expectEqual(prefix.states_max, store.bound());
    store.limit = 10;
    const chain = try store.open(0);
    try write(&pool, chain, 32);
    var pos: usize = 1;
    while (pos <= 10) : (pos += 1) try store.note(chain, pos, .chunk);
    try testing.expectEqual(@as(usize, 10), store.marks.items.len);
    try store.note(chain, 11, .chunk);
    try testing.expectEqual(@as(usize, 10), store.marks.items.len);
    try testing.expect(!has(&store, chain, 1));
    try testing.expect(has(&store, chain, 11));

    var wide = try Pool.init(gpa, .{ .extents = 4, .capacity = 1024, .slots = 2 });
    defer wide.deinit();
    var two = try Store.init(gpa, &wide, 2, round_positions);
    defer two.deinit();
    two.limit = 12;
    const live = try two.open(0);
    try write(&wide, live, 32);
    pos = 1;
    while (pos <= 11) : (pos += 1) try two.note(live, pos, .chunk);
    const idle = try two.open(1);
    try write(&wide, idle, 8);
    try two.note(idle, 8, .chunk);
    try two.preempt(1);
    try testing.expectEqual(two.bound(), two.marks.items.len);
    // One more mark on the live chain. The idle chain's mark goes, and the live chain's lowest stays.
    try two.note(live, 12, .chunk);
    try testing.expectEqual(two.bound(), two.marks.items.len);
    try testing.expect(!has(&two, idle, 8));
    try testing.expect(has(&two, live, 1));
    try testing.expect(has(&two, live, 12));
}

test "a shared prefix's end outlives the bound's worth of other snapshots, and goes among its own kind by the usual order" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, .{ .extents = 32, .capacity = 2048, .slots = 2 });
    defer pool.deinit();
    var store = try Store.init(gpa, &pool, 2, round_positions);
    defer store.deinit();
    // the first conversation over a harness of 600 tokens, ended: the least recently used chain from here on
    const first = try store.open(0);
    try write(&pool, first, 900);
    try store.note(first, 600, .message);
    try store.note(first, 880, .tail);
    store.share(first, 600);
    try store.preempt(0);
    // other conversations, each on a chain of its own and idle in its turn: 64 snapshots and twice more
    var taken: usize = 0;
    while (taken < 3 * prefix.states_max) : (taken += 8) {
        const other = try store.open(1);
        try pool.grow(other, 100);
        pool.append(other, &@as([100]i32, @splat(@intCast(7000 + taken))));
        for (0..8) |i| try store.note(other, 10 + 10 * i, .message);
        try store.retire(1, other.len());
    }
    // a conversation that begins with the harness still finds its end
    const hit = store.find(first.tokens.items[0..700]).?;
    try testing.expectEqual(first, hit.chain);
    try testing.expectEqual(@as(usize, 600), hit.at);
    try testing.expectEqual(store.bound(), store.marks.items.len);
    try testing.expect(has(&store, first, 600));
    // its own later mark went as any idle chain's does
    try testing.expect(!has(&store, first, 880));

    // among shared ones the order is the usual one: with nothing else held the least recently used goes
    while (store.marks.items.len > 2) _ = store.dropOne();
    try testing.expect(has(&store, first, 600));
    const newest = store.marks.items[if (store.marks.items[0].chain == first) 1 else 0];
    store.share(newest.chain, newest.pos);
    const gone = store.dropOne().?;
    try testing.expectEqual(first, gone.chain);
    try testing.expect(gone.shared);
    try pool.check(gpa);
}

test "eviction leaves the tail a shared prefix ends in while another chain has one to give" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, .{ .extents = 8, .capacity = 2048, .slots = 2 });
    defer pool.deinit();
    var store = try Store.init(gpa, &pool, 2, round_positions);
    defer store.deinit();
    const harness = try store.open(0);
    try write(&pool, harness, 1100);
    try store.note(harness, 600, .message);
    store.share(harness, 600);
    try store.preempt(0);
    const other = try store.open(1);
    try write(&pool, other, 1100);
    try store.note(other, 600, .message);
    try store.preempt(1);
    // the harness's chain is the older; its third extent goes first, holding nothing shared
    try testing.expectEqual(harness, (try store.evict(null)).?.chain);
    try testing.expectEqual(@as(usize, 1024), harness.len());
    // its second holds the shared end: the other chain gives all it has before that one is taken
    for ([_]usize{ 1024, 512, 0 }) |left| {
        const shrunk = (try store.evict(null)).?;
        try testing.expectEqual(other, shrunk.chain);
        try testing.expectEqual(left, shrunk.len);
    }
    try testing.expect(has(&store, harness, 600) and !has(&store, other, 600));
    // nothing else is left to take
    try testing.expectEqual(harness, (try store.evict(null)).?.chain);
    try testing.expect(!has(&store, harness, 600));
    try pool.check(gpa);
}

test "a preempted chain stays idle and a replay finds the tokens it still holds" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, .{ .extents = 3, .capacity = 1024, .slots = 1 });
    defer pool.deinit();
    var store = try Store.init(gpa, &pool, 1, round_positions);
    defer store.deinit();
    const chain = try store.open(0);
    try write(&pool, chain, 12);
    try store.note(chain, 8, .chunk);
    try store.note(chain, 12, .chunk);
    try store.preempt(0);
    try testing.expect(store.slots[0] == null);
    try testing.expect(!chain.live);
    try testing.expectEqual(@as(usize, 12), chain.len());
    const whole = chain.tokens.items;
    const hit = store.find(whole).?;
    try testing.expectEqual(chain, hit.chain);
    try testing.expectEqual(@as(usize, 12), hit.at);
    const prompt = store.find(whole[0..8]).?;
    try testing.expectEqual(@as(usize, 8), prompt.at);
    var other: [12]i32 = undefined;
    @memset(&other, 9);
    try testing.expect(store.find(&other) == null);
    const forked = try store.fork(hit.chain, hit.at);
    try testing.expectEqualSlices(i32, whole, forked.chain.tokens.items);
    store.copied();

    // A cancel keeps the tokens the connection took, and the snapshot past that goes.
    const again = try store.open(0);
    try write(&pool, again, 12);
    try store.note(again, 8, .chunk);
    try store.note(again, 12, .chunk);
    try store.retire(0, 8);
    try testing.expect(!again.live);
    try testing.expectEqual(@as(usize, 8), again.len());
    try testing.expect(has(&store, again, 8));
    try testing.expect(!has(&store, again, 12));
    try pool.check(gpa);
}

/// What a store told its caller of.
const Told = struct {
    ones: [8]usize = undefined,
    n: usize = 0,
    past: ?usize = null,
    /// The chain still held the extent the snapshots stand in.
    held: bool = false,

    fn call(context: *anyopaque, what: Store.Going) void {
        const self: *Told = @ptrCast(@alignCast(context));
        switch (what) {
            .one => |mark| {
                self.ones[self.n] = mark.pos;
                self.n += 1;
            },
            .past => |cut| {
                self.past = cut.len;
                self.held = cut.chain.len() > cut.len;
            },
        }
    }
};

test "the caller is told of a snapshot the bound takes and of those an eviction takes, before the extent goes; not of a chain cut" {
    const gpa = testing.allocator;
    var pool = try Pool.init(gpa, .{ .extents = 3, .capacity = 1024, .slots = 1 });
    defer pool.deinit();
    var store = try Store.init(gpa, &pool, 1, round_positions);
    defer store.deinit();
    var told: Told = .{};
    store.leaving = .{ .context = &told, .call = Told.call };
    store.limit = 3;
    const chain = try store.open(0);
    try write(&pool, chain, 600);
    for ([_]usize{ 100, 300, 550 }) |pos| try store.note(chain, pos, .message);
    try testing.expectEqual(0, told.n);
    try store.note(chain, 590, .message);
    try testing.expectEqual(1, told.n);
    try testing.expect(!has(&store, chain, told.ones[0]));
    try store.preempt(0);
    // the second extent goes: the snapshots above 512 with it, told of while the chain holds 600
    try testing.expect(try store.evict(null) != null);
    try testing.expectEqual(512, told.past.?);
    try testing.expect(told.held);
    try testing.expect(!has(&store, chain, 550) and !has(&store, chain, 590));
    try testing.expectEqual(1, told.n);
    // a mark written is not written again, and the store keeps the word
    for (store.marks.items) |*mark| mark.spilt = true;
    try testing.expect(try store.evict(null) != null);
    try testing.expectEqual(0, told.past.?);
}
