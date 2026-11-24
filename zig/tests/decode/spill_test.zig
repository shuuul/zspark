//! `spill.zig`: which state of a sequence that leaves is written, and which file a prompt starts from.

const std = @import("std");
const testing = std.testing;
const learned = @import("../../src/decode/learned.zig");
const spill = @import("../../src/decode/spill.zig");

fn put(store: *learned.Store, tokens: []const i32) !void {
    try store.putHost(tokens, &.{"x"}, &.{});
}

test "the state written is the highest of the prompt's at the floor or above, and none where there is none" {
    try testing.expectEqual(2048, spill.floor);
    try testing.expectEqual(null, spill.pick(&.{}));
    // a reply's states and the prompt's below the floor are not written
    try testing.expectEqual(null, spill.pick(&.{ .{ .pos = 9000, .prompt = false }, .{ .pos = 2047, .prompt = true } }));
    try testing.expectEqual(1, spill.pick(&.{ .{ .pos = 2048, .prompt = true }, .{ .pos = 7000, .prompt = true }, .{ .pos = 9000, .prompt = false }, .{ .pos = 4000, .prompt = true } }));
    try testing.expectEqual(0, spill.pick(&.{.{ .pos = 2048, .prompt = true }}));
}

test "a prompt starts from the longest file it goes on from that stands worth a read above where it starts" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: learned.Store = try .open(testing.allocator, io, tmp.dir, 1, 1 << 20);
    defer store.close();
    var prompt: [4000]i32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(i * 31 % 1009);
    try testing.expectEqual(512, spill.worth);
    try testing.expectEqual(null, spill.serving(&store, &prompt, 0, prompt.len));
    // a state at 1000 and one at 3000: a file's tokens are the sequence through the token after its state
    try put(&store, prompt[0..1001]);
    try put(&store, prompt[0..3001]);
    try testing.expectEqual(3000, spill.serving(&store, &prompt, 0, prompt.len));
    try testing.expectEqual(3000, spill.serving(&store, &prompt, 2488, prompt.len));
    try testing.expectEqual(null, spill.serving(&store, &prompt, 2489, prompt.len));
    // the next one down, where the longest was not read
    try testing.expectEqual(1000, spill.serving(&store, &prompt, 0, 3001));
    try testing.expectEqual(null, spill.serving(&store, &prompt, 489, 3001));
    // a prompt that ends in the file's last token has none of its own to run from that state
    try testing.expectEqual(1000, spill.serving(&store, prompt[0..3001], 0, 3001));
    try testing.expectEqual(3000, spill.serving(&store, prompt[0..3002], 0, 3002));
}

const Fake = struct {
    var space: ?u64 = 110;
    var tick: ?u64 = 100;
    fn free(_: std.posix.fd_t) ?u64 {
        return space;
    }
    fn clock() ?u64 {
        return tick;
    }
};

test "the disk's floor, as TensorFold's: a write reserves above the floor and the writes before it; a refusal or a failure holds the next back 1 to 60 s unless the disk frees space; unknown space refuses" {
    Fake.space = 110;
    Fake.tick = 100;
    var floor: spill.Floor = .{ .floor = 100, .free = Fake.free, .clock = Fake.clock };
    try testing.expectEqual(4 << 30, (spill.Floor{}).floor);
    try testing.expectEqual(.ready, floor.reserve(0, 10));
    // the 10 reserved count against the next
    try testing.expectEqual(.refused, floor.reserve(0, 1));
    floor.finish(0, 10, false);
    try testing.expectEqual(0, floor.reserved);
    // held back while the disk has not freed space
    try testing.expectEqual(.quiet, floor.reserve(0, 10));
    Fake.space = 120;
    try testing.expectEqual(.ready, floor.reserve(0, 10));
    floor.finish(0, 10, true);
    try testing.expectEqual(0, floor.failures);
    // a write that was not begun gives its bytes back and holds nothing back
    try testing.expectEqual(.ready, floor.reserve(0, 10));
    floor.release(10);
    try testing.expectEqual(.ready, floor.reserve(0, 20));
    floor.finish(0, 20, true);
    Fake.space = null;
    try testing.expectEqual(.refused, floor.reserve(0, 1));
    Fake.tick = 200;
    Fake.space = std.math.maxInt(u64);
    try testing.expectEqual(.refused, floor.reserve(0, std.math.maxInt(u64)));
    for (0..20) |_| {
        Fake.tick.? += 100;
        _ = floor.reserve(0, std.math.maxInt(u64));
    }
    try testing.expectEqual(Fake.tick.? + 60, floor.retry_at);
    try testing.expectEqual(0, floor.reserved);
    // no clock: refused
    Fake.tick = null;
    try testing.expectEqual(.refused, floor.reserve(0, 1));
    // `admitted` without a floor writes, with one asks it
    try testing.expect(spill.admitted(null, 0, 1 << 40, 3000));
    Fake.tick = 10_000;
    Fake.space = 1000;
    var low: spill.Floor = .{ .floor = 990, .free = Fake.free, .clock = Fake.clock };
    try testing.expect(spill.admitted(&low, 0, 10, 3000));
    try testing.expect(!spill.admitted(&low, 0, 1, 3000));
}

test "the disk's floor reads the free space of the directory's own file system" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const free = spill.available(tmp.dir.handle) orelse return error.TestExpectedSpace;
    try testing.expect(free > 0);
    try testing.expectEqual(null, spill.available(-1));
}
