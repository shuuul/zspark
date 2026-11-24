//! `spill_writer.zig`: a state handed to the writer is read from memory until its file stands, the same
//! bytes; what is listed twice, discarded, over the bound or failed; the stop.

const std = @import("std");
const testing = std.testing;
const Io = std.Io;
const learned = @import("../../src/decode/learned.zig");
const spill = @import("../../src/decode/spill.zig");
const spill_writer = @import("../../src/decode/spill_writer.zig");
const Store = learned.Store;
const Writer = spill_writer.Writer;

const fx = struct {
    const Sizes = struct { block: usize, queue: u64, warm: usize };

    /// The writer's sizes for a test, and what they were.
    fn sizes(new: Sizes) Sizes {
        const old: Sizes = .{ .block = spill.block, .queue = spill.queue_bytes, .warm = spill.warm_bytes };
        spill.block = new.block;
        spill.queue_bytes = new.queue;
        spill.warm_bytes = new.warm;
        return old;
    }

    fn tokensOf(buffer: []i32, seed: i32) []const i32 {
        for (buffer, 0..) |*token, i| token.* = seed * 1000 + @as(i32, @intCast(i % 997));
        return buffer;
    }

    /// A state's two parts, told from every other seed's.
    fn partsOf(buffer: []u8, seed: u8) [2][]const u8 {
        for (buffer, 0..) |*byte, i| byte.* = seed +% @as(u8, @truncate(i *% 31));
        return .{ buffer[0 .. buffer.len / 3], buffer[buffer.len / 3 ..] };
    }

    fn offer(store: *Store, tokens: []const i32, parts: []const []const u8, until: ?Io.Clock.Timestamp) !bool {
        var table: [8]learned.Part = undefined;
        for (parts, table[0..parts.len]) |part, *one| one.* = .{ .bytes = part.len };
        var taking = try store.writer.?.begin(tokens, table[0..parts.len], until) orelse return false;
        for (parts) |part| taking.copy(part);
        taking.commit();
        return true;
    }

    /// The state for `tokens` is held and is `parts`.
    fn expectHeld(store: *Store, tokens: []const i32, parts: []const []const u8) !void {
        const gpa = testing.allocator;
        var image = try store.find(gpa, tokens) orelse return error.TestExpectedState;
        defer image.close(gpa);
        try testing.expectEqual(parts.len, image.parts.len);
        for (parts) |part| {
            const got = try gpa.alloc(u8, part.len);
            defer gpa.free(got);
            try image.part(got);
            try testing.expectEqualSlices(u8, part, got);
        }
    }

    fn after(io: Io, ms: i64) Io.Clock.Timestamp {
        return .fromNow(io, .{ .raw = .fromMilliseconds(ms), .clock = .awake });
    }

    fn letGo(io: Io, writer: *Writer, ms: i64) void {
        io.sleep(.fromMilliseconds(ms), .awake) catch {};
        writer.hold(false);
    }

    fn files(io: Io, dir: Io.Dir, ending: []const u8) !usize {
        var n: usize = 0;
        var entries = dir.iterate();
        while (try entries.next(io)) |entry| n += @intFromBool(std.mem.endsWith(u8, entry.name, ending));
        return n;
    }
};

test "the spill writer: a state the writer still holds is read from its memory, never from a part; its file then gives the same bytes" {
    const io = testing.io;
    const old = fx.sizes(.{ .block = 4096, .queue = 1 << 20, .warm = 8192 });
    defer _ = fx.sizes(old);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 1 << 20);
    defer store.close();
    Writer.start(&store);
    Writer.start(&store);
    const writer = store.writer.?;

    var ids: [3000]i32 = undefined;
    var bytes: [9001]u8 = undefined;
    const tokens = fx.tokensOf(&ids, 1);
    const parts = fx.partsOf(&bytes, 1);
    const key = learned.keyOf(tokens);
    writer.hold(true);
    try testing.expect(try fx.offer(&store, tokens, &parts, null));
    try testing.expectEqual(1, store.written);
    // a part cut short stands under its name, as a write under way leaves one
    try store.dir.writeFile(io, .{ .sub_path = &learned.partOf(key), .data = "cut" });
    try testing.expect(!store.onDisk(key));
    try testing.expect(store.has(tokens));
    try testing.expectEqual(3000, store.longest(tokens, 3001));
    try testing.expectEqual(null, store.longest(tokens, 3000));
    try fx.expectHeld(&store, tokens, &parts);
    try testing.expectEqual(1, store.read);
    try testing.expectEqual(0, store.held.items.len);
    // other tokens, and a prefix, are no state of it
    try testing.expect(try store.find(testing.allocator, tokens[0..2999]) == null);

    writer.hold(false);
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    try testing.expect(store.onDisk(key));
    try testing.expectEqual(0, try fx.files(io, store.dir, ".part"));
    try testing.expectEqual(1, store.held.items.len);
    try testing.expectEqual(0, writer.jobs.items.len);
    try testing.expectEqual(3000, store.longest(tokens, 3001));
    try fx.expectHeld(&store, tokens, &parts);
    try testing.expectEqual(2, store.read);
    try testing.expectEqual(1, store.written);
    try testing.expectEqual(1, writer.done);

    // the next load's store reads the file
    var next: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 1 << 20);
    defer next.close();
    try fx.expectHeld(&next, tokens, &parts);
}

test "the spill writer: a part alone is no state: a prompt that asks for it is missed" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 1 << 20);
    defer store.close();
    Writer.start(&store);
    var ids: [40]i32 = undefined;
    const tokens = fx.tokensOf(&ids, 2);
    try store.dir.writeFile(io, .{ .sub_path = &learned.partOf(learned.keyOf(tokens)), .data = "zslearn\x02 and no more" });
    try testing.expect(!store.has(tokens));
    try testing.expectEqual(null, store.longest(tokens, 41));
    try testing.expect(try store.find(testing.allocator, tokens) == null);
}

test "the spill writer: a state listed or on disk is not taken again, and one discarded is not kept: listed, or written and not yet reaped" {
    const io = testing.io;
    const old = fx.sizes(.{ .block = 4096, .queue = 1 << 20, .warm = 0 });
    defer _ = fx.sizes(old);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 1 << 20);
    defer store.close();
    Writer.start(&store);
    const writer = store.writer.?;
    var ids: [3][600]i32 = undefined;
    var bytes: [3][5000]u8 = undefined;
    var tokens: [3][]const i32 = undefined;
    var parts: [3][2][]const u8 = undefined;
    for (0..3) |i| {
        tokens[i] = fx.tokensOf(&ids[i], @intCast(i + 1));
        parts[i] = fx.partsOf(&bytes[i], @intCast(i + 1));
    }

    writer.hold(true);
    try testing.expect(try fx.offer(&store, tokens[0], &parts[0], null));
    try testing.expect(!try fx.offer(&store, tokens[0], &parts[0], null));
    try testing.expect(try fx.offer(&store, tokens[1], &parts[1], null));
    try testing.expectEqual(2, store.written);
    store.discard(tokens[0]);
    try testing.expect(!store.has(tokens[0]));
    try testing.expectEqual(null, store.longest(tokens[0], 601));
    try testing.expect(try store.find(testing.allocator, tokens[0]) == null);
    writer.hold(false);
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    try testing.expect(!store.onDisk(learned.keyOf(tokens[0])));
    try testing.expect(store.onDisk(learned.keyOf(tokens[1])));
    try testing.expectEqual(1, writer.done);
    try testing.expectEqual(0, writer.failed);
    // on disk: not taken again
    try testing.expect(!try fx.offer(&store, tokens[1], &parts[1], null));
    try testing.expectEqual(2, store.written);

    // written, and discarded before the offering thread met it again: the file goes
    try testing.expect(try fx.offer(&store, tokens[2], &parts[2], null));
    while (!store.onDisk(learned.keyOf(tokens[2]))) try io.sleep(.fromMilliseconds(1), .awake);
    store.discard(tokens[2]);
    try testing.expect(!store.onDisk(learned.keyOf(tokens[2])));
    try testing.expect(!store.has(tokens[2]));
    try testing.expectEqual(1, store.held.items.len);
    try fx.expectHeld(&store, tokens[1], &parts[1]);
}

test "the spill writer: past the bound a state that leaves is not written; a stop waits for room, its time at most; one is always taken" {
    const io = testing.io;
    const old = fx.sizes(.{ .block = 4096, .queue = 3 * 4096, .warm = 4096 });
    defer _ = fx.sizes(old);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 1 << 20);
    defer store.close();
    Writer.start(&store);
    const writer = store.writer.?;
    var ids: [4][100]i32 = undefined;
    var bytes: [4][7000]u8 = undefined;
    var tokens: [4][]const i32 = undefined;
    var parts: [4][2][]const u8 = undefined;
    for (0..4) |i| {
        tokens[i] = fx.tokensOf(&ids[i], @intCast(i + 1));
        parts[i] = fx.partsOf(&bytes[i], @intCast(i + 1));
    }

    writer.hold(true);
    // two blocks each: the second has no room beside the first
    try testing.expect(try fx.offer(&store, tokens[0], &parts[0], null));
    try testing.expect(!try fx.offer(&store, tokens[1], &parts[1], null));
    try testing.expectEqual(1, writer.dropped);
    try testing.expectEqual(1, store.written);
    try testing.expect(!store.has(tokens[1]));
    // a stop's waits, and gives up at its time
    try testing.expect(!try fx.offer(&store, tokens[1], &parts[1], fx.after(io, 30)));
    try testing.expect(!writer.drain(fx.after(io, 20)));
    // and takes it once the first is written
    var later = try io.concurrent(fx.letGo, .{ io, writer, 20 });
    defer later.await(io);
    try testing.expect(try fx.offer(&store, tokens[1], &parts[1], fx.after(io, 10_000)));
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    try fx.expectHeld(&store, tokens[0], &parts[0]);
    try fx.expectHeld(&store, tokens[1], &parts[1]);

    // nothing listed: a state over the whole bound is taken, and its blocks beyond the bound's are not kept
    var big: [5 * 4096]u8 = undefined;
    const whole = fx.partsOf(&big, 9);
    try testing.expect(try fx.offer(&store, tokens[2], &whole, null));
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    try fx.expectHeld(&store, tokens[2], &whole);
    try testing.expect(writer.blocks <= 3);
    try testing.expectEqual(3, writer.done);
    try testing.expectEqual(1, writer.dropped);
}

test "the spill writer: the cap counts a state still to write: the oldest file goes as it is taken, and one over the cap is not" {
    const io = testing.io;
    const old = fx.sizes(.{ .block = 4096, .queue = 1 << 20, .warm = 0 });
    defer _ = fx.sizes(old);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a state is 24 + 400 + 32 + 5000 bytes: two and a half fit
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 13_500);
    defer store.close();
    Writer.start(&store);
    const writer = store.writer.?;
    var ids: [4][100]i32 = undefined;
    var bytes: [4][5000]u8 = undefined;
    var tokens: [4][]const i32 = undefined;
    var parts: [4][2][]const u8 = undefined;
    for (0..4) |i| {
        tokens[i] = fx.tokensOf(&ids[i], @intCast(i + 1));
        parts[i] = fx.partsOf(&bytes[i], @intCast(i + 1));
    }
    try testing.expect(try fx.offer(&store, tokens[0], &parts[0], null));
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    try testing.expect(try fx.offer(&store, tokens[1], &parts[1], null));
    try testing.expect(writer.drain(fx.after(io, 10_000)));

    writer.hold(true);
    try testing.expect(try fx.offer(&store, tokens[2], &parts[2], null));
    try testing.expect(!store.has(tokens[0]));
    try testing.expectEqual(1, try fx.files(io, store.dir, ".st"));
    try fx.expectHeld(&store, tokens[1], &parts[1]);
    try fx.expectHeld(&store, tokens[2], &parts[2]);
    var huge: [14_000]u8 = undefined;
    const over = fx.partsOf(&huge, 7);
    try testing.expect(!try fx.offer(&store, tokens[3], &over, null));
    try testing.expectEqual(0, writer.dropped);
    writer.hold(false);
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    try testing.expectEqual(2, try fx.files(io, store.dir, ".st"));
    try fx.expectHeld(&store, tokens[1], &parts[1]);
    try fx.expectHeld(&store, tokens[2], &parts[2]);
    try testing.expectEqual(3, store.written);
}

test "the spill writer: a cap under the queue's bytes bounds what is listed: a file on disk is not cleared for states that would not all stay" {
    const io = testing.io;
    const old = fx.sizes(.{ .block = 4096, .queue = 1 << 20, .warm = 0 });
    defer _ = fx.sizes(old);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a state is 5,432 bytes and two blocks: two and a half fit the cap, two lists of two blocks do not
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 13_500);
    defer store.close();
    Writer.start(&store);
    const writer = store.writer.?;
    var ids: [3][100]i32 = undefined;
    var bytes: [3][5000]u8 = undefined;
    var tokens: [3][]const i32 = undefined;
    var parts: [3][2][]const u8 = undefined;
    for (0..3) |i| {
        tokens[i] = fx.tokensOf(&ids[i], @intCast(i + 1));
        parts[i] = fx.partsOf(&bytes[i], @intCast(i + 1));
    }
    try testing.expect(try fx.offer(&store, tokens[0], &parts[0], null));
    try testing.expect(writer.drain(fx.after(io, 10_000)));

    writer.hold(true);
    try testing.expect(try fx.offer(&store, tokens[1], &parts[1], null));
    // a second one listed would keep 10,864 bytes of the cap and clear the file that is there
    try testing.expect(!try fx.offer(&store, tokens[2], &parts[2], null));
    try testing.expectEqual(1, writer.dropped);
    try testing.expectEqual(1, try fx.files(io, store.dir, ".st"));
    writer.hold(false);
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    try fx.expectHeld(&store, tokens[0], &parts[0]);
    try fx.expectHeld(&store, tokens[1], &parts[1]);
    try testing.expectEqual(2, store.written);
}

test "the spill writer: a file that fails costs that state alone, and a store closed with states listed leaves no part" {
    const io = testing.io;
    const old = fx.sizes(.{ .block = 4096, .queue = 1 << 20, .warm = 0 });
    defer _ = fx.sizes(old);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var ids: [3][100]i32 = undefined;
    var bytes: [3][9000]u8 = undefined;
    var tokens: [3][]const i32 = undefined;
    var parts: [3][2][]const u8 = undefined;
    for (0..3) |i| {
        tokens[i] = fx.tokensOf(&ids[i], @intCast(i + 1));
        parts[i] = fx.partsOf(&bytes[i], @intCast(i + 1));
    }
    {
        var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 1 << 20);
        defer store.close();
        Writer.start(&store);
        const writer = store.writer.?;
        writer.hold(true);
        // a directory stands where the first's part would
        try store.dir.createDir(io, &learned.partOf(learned.keyOf(tokens[0])), .default_dir);
        try testing.expect(try fx.offer(&store, tokens[0], &parts[0], null));
        try testing.expect(try fx.offer(&store, tokens[1], &parts[1], null));
        writer.hold(false);
        try testing.expect(writer.drain(fx.after(io, 10_000)));
        try testing.expectEqual(1, writer.failed);
        try testing.expectEqual(1, writer.done);
        try testing.expect(!store.has(tokens[0]));
        try testing.expectEqual(null, store.longest(tokens[0], 101));
        try fx.expectHeld(&store, tokens[1], &parts[1]);
        try store.dir.deleteDir(io, &learned.partOf(learned.keyOf(tokens[0])));

        writer.hold(true);
        try testing.expect(try fx.offer(&store, tokens[2], &parts[2], null));
        try testing.expect(store.has(tokens[2]));
    }
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 1 << 20);
    defer store.close();
    try testing.expectEqual(0, try fx.files(io, store.dir, ".part"));
    try testing.expectEqual(1, try fx.files(io, store.dir, ".st"));
    try testing.expect(!store.has(tokens[2]));
}

test "the spill writer: states offered while the writer writes: each is read whole at every moment, from memory or its file" {
    const io = testing.io;
    const gpa = testing.allocator;
    const old = fx.sizes(.{ .block = 4096, .queue = 1 << 20, .warm = 8192 });
    defer _ = fx.sizes(old);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(gpa, io, tmp.dir, 0xabc, 64 << 20);
    defer store.close();
    Writer.start(&store);
    const writer = store.writer.?;
    const count = 24;
    const ids = try gpa.alloc([200]i32, count);
    defer gpa.free(ids);
    const bytes = try gpa.alloc([30_000]u8, count);
    defer gpa.free(bytes);
    var parts: [count][2][]const u8 = undefined;
    for (0..count) |i| {
        _ = fx.tokensOf(&ids[i], @intCast(i + 1));
        parts[i] = fx.partsOf(&bytes[i], @intCast(i + 1));
        try testing.expect(try fx.offer(&store, &ids[i], &parts[i], null));
        for (0..i + 1) |j| try fx.expectHeld(&store, &ids[j], &parts[j]);
    }
    try testing.expect(writer.drain(fx.after(io, 30_000)));
    try testing.expectEqual(count, writer.done);
    try testing.expectEqual(count, store.held.items.len);
    try testing.expectEqual(count, try fx.files(io, store.dir, ".st"));
    for (0..count) |j| try fx.expectHeld(&store, &ids[j], &parts[j]);
}

/// A disk of a fixed free space, and a clock that stands.
const Disk = struct {
    var space: u64 = 0;
    fn free(_: std.posix.fd_t) ?u64 {
        return space;
    }
    fn clock() ?u64 {
        return 1000;
    }
};

test "the spill writer: a state that would leave the disk under its floor is not taken, and its bytes stay reserved only while it is listed" {
    const io = testing.io;
    const old = fx.sizes(.{ .block = 4096, .queue = 64 * 4096, .warm = 4096 });
    defer _ = fx.sizes(old);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 1 << 20);
    defer store.close();
    Writer.start(&store);
    const writer = store.writer.?;
    var ids: [3][100]i32 = undefined;
    var bytes: [3][7000]u8 = undefined;
    var tokens: [3][]const i32 = undefined;
    var parts: [3][2][]const u8 = undefined;
    for (0..3) |i| {
        tokens[i] = fx.tokensOf(&ids[i], @intCast(i + 1));
        parts[i] = fx.partsOf(&bytes[i], @intCast(i + 1));
    }
    // the disk above its floor by a file and a half: the first is taken, the second not while it is listed
    const file: u64 = learned.prefixBytes(100, 2) + 7000;
    Disk.space = 1 << 30;
    var floor: spill.Floor = .{ .floor = Disk.space - (file + file / 2), .free = Disk.free, .clock = Disk.clock };
    writer.floor = &floor;
    writer.hold(true);
    try testing.expect(try fx.offer(&store, tokens[0], &parts[0], null));
    try testing.expectEqual(file, floor.reserved);
    try testing.expect(!try fx.offer(&store, tokens[1], &parts[1], null));
    try testing.expect(!store.has(tokens[1]));
    writer.hold(false);
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    writer.reap();
    try testing.expectEqual(0, floor.reserved);
    try fx.expectHeld(&store, tokens[0], &parts[0]);
    // the disk freed space since the refusal: taken at once
    Disk.space += file;
    try testing.expect(try fx.offer(&store, tokens[1], &parts[1], null));
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    writer.reap();
    try fx.expectHeld(&store, tokens[1], &parts[1]);
    // no floor: taken as before
    writer.floor = null;
    try testing.expect(try fx.offer(&store, tokens[2], &parts[2], null));
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    try testing.expectEqual(0, floor.reserved);
}

test "the spill writer by block: two states over one prefix, listed together, are read from memory and leave its blocks once on disk" {
    const io = testing.io;
    const old = fx.sizes(.{ .block = 4096, .queue = 1 << 20, .warm = 0 });
    defer _ = fx.sizes(old);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabd, 1 << 22);
    defer store.close();
    Writer.start(&store);
    const writer = store.writer.?;
    var ids: [2][800]i32 = undefined;
    const first = fx.tokensOf(&ids[0], 1);
    const second = fx.tokensOf(&ids[1], 1);
    // they part at 700: of their three blocks each, 0 and 1 are the same
    for (ids[1][700..]) |*token| token.* +%= 3;
    var rows: [2][799 * 4]u8 = undefined;
    for (&rows, [_][]const i32{ first, second }) |*bytes, tokens| for (0..799) |i| {
        @memset(bytes[i * 4 ..][0..4], @truncate(@as(u32, @bitCast(tokens[i] +% tokens[i + 1]))));
    };
    const own = "a state's own";
    writer.hold(true);
    for ([_][]const i32{ first, second }, &rows) |tokens, *bytes| {
        const table = [_]learned.Part{ .{ .bytes = own.len }, .{ .bytes = bytes.len, .per = 1 } };
        var taking = (try writer.begin(tokens, &table, null)).?;
        taking.copy(own);
        taking.copy(bytes);
        taking.commit();
    }
    try fx.expectHeld(&store, first, &.{ own, &rows[0] });
    try fx.expectHeld(&store, second, &.{ own, &rows[1] });
    // nothing on disk yet, and the sweep a begin makes keeps what a listed state names
    try testing.expectEqual(0, try fx.files(io, store.dir, ".blk"));
    writer.hold(false);
    try testing.expect(writer.drain(fx.after(io, 10_000)));
    try testing.expectEqual(4, try fx.files(io, store.dir, ".blk"));
    try testing.expectEqual(2, try fx.files(io, store.dir, ".st"));
    try testing.expectEqual(2, store.held.items.len);
    try fx.expectHeld(&store, first, &.{ own, &rows[0] });
    try fx.expectHeld(&store, second, &.{ own, &rows[1] });
    var next: Store = try .open(testing.allocator, io, tmp.dir, 0xabd, 1 << 22);
    defer next.close();
    try fx.expectHeld(&next, second, &.{ own, &rows[1] });
    try testing.expectEqual(4, try fx.files(io, next.dir, ".blk"));
}
