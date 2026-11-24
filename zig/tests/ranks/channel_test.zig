//! `channel.zig`'s tests, over a TCP connection on the loopback address: the frames as sent, the handshake and
//! each of its refusals by name, a frame cut short, a peer that closes. Host only.

const std = @import("std");
const testing = std.testing;
const net = std.Io.net;
const channel_file = @import("../../src/ranks/channel.zig");
const Channel = channel_file.Channel;
const Link = channel_file.Link;
const Named = channel_file.Named;

/// Both ends of one connection in this process: `zero` is rank 0's, which connected, `one` rank 1's.
const Ends = struct {
    links: [2]Link,
    zero: Channel,
    one: Channel,
    open: [2]bool,

    /// In place: a channel reads through its link.
    fn init(self: *Ends) !void {
        const io = testing.io;
        const any: net.IpAddress = .{ .ip4 = .loopback(0) };
        var server = try any.listen(io, .{});
        defer server.deinit(io);
        // the kernel completes the connection from the backlog, so one thread makes both ends
        const client = try server.socket.address.connect(io, .{ .mode = .stream });
        self.links[0].over(io, client);
        self.links[1].over(io, try server.accept(io));
        self.zero = self.links[0].channel(testing.allocator);
        self.one = self.links[1].channel(testing.allocator);
        self.open = .{ true, true };
    }

    fn close(self: *Ends, rank: usize) void {
        if (!self.open[rank]) return;
        self.links[rank].close();
        self.open[rank] = false;
    }

    /// Bytes as they are, from rank 0's end: what a frame is made of, or is not.
    fn raw(self: *Ends, bytes: []const u8) !void {
        try self.links[0].writer.interface.writeAll(bytes);
        try self.links[0].writer.interface.flush();
    }

    fn deinit(self: *Ends) void {
        self.zero.deinit();
        self.one.deinit();
        for (0..2) |rank| self.close(rank);
    }
};

const fx = struct {
    const same: channel_file.Identity = blk: {
        // three SHA-256 at compile time
        @setEvalBranchQuota(200_000);
        break :blk .{ .build = .of("2cdf223"), .checkpoint = .of("random, seed 45"), .options = .of("rows=80 capacity=256") };
    };
    const id: channel_file.Id = blk: {
        var bytes: channel_file.Id = undefined;
        for (&bytes, 0..) |*byte, index| byte.* = @intCast((index * 37 + 11) % 251);
        break :blk bytes;
    };
};

test "frames arrive as sent: a request's bytes, one longer than the link's buffers, an empty one, a barrier's counter" {
    var ends: Ends = undefined;
    try ends.init();
    defer ends.deinit();
    const long = try testing.allocator.alloc(u8, 20_000);
    defer testing.allocator.free(long);
    for (long, 0..) |*byte, index| byte.* = @truncate(index * 131 + (index >> 8));

    try ends.zero.request("ids and settings");
    try ends.zero.request(long);
    try ends.zero.request("");
    try testing.expectEqualStrings("ids and settings", (try ends.one.next()).request);
    try testing.expectEqualSlices(u8, long, (try ends.one.next()).request);
    try testing.expectEqual(@as(usize, 0), (try ends.one.next()).request.len);

    // a barrier: each rank sends its counter and takes the other's, in either order
    try ends.one.send(.barrier, &std.mem.toBytes(std.mem.nativeToLittle(u64, 7)));
    try ends.zero.barrier(7);
    try testing.expectEqual(@as(u64, 7), (try ends.one.next()).barrier);
}

test "the handshake: rank 1 holds rank 0's id and both pass when build, checkpoint and options are the same" {
    var ends: Ends = undefined;
    try ends.init();
    defer ends.deinit();
    var got: channel_file.Id = @splat(0);
    try ends.zero.offer(fx.same, &fx.id);
    try ends.one.answer(fx.same, &got);
    try ends.zero.conclude(fx.same);
    try testing.expectEqualSlices(u8, &fx.id, &got);
    // and the channel is at a frame's boundary after it
    try ends.one.request("next");
    try testing.expectEqualStrings("next", (try ends.zero.next()).request);
}

test "a hash that differs is refused by name on both ranks: the build, the checkpoint's headers, the options" {
    const cases = [_]struct { other: channel_file.Identity, err: channel_file.Error, names: []const []const u8 }{
        .{ .other = .{ .build = .of("609ca41 dirty"), .checkpoint = fx.same.checkpoint, .options = fx.same.options }, .err = error.BuildsDiffer, .names = &.{ "the builds differ", "2cdf223", "609ca41 dirty" } },
        .{ .other = .{ .build = fx.same.build, .checkpoint = .of("random, seed 46"), .options = fx.same.options }, .err = error.CheckpointsDiffer, .names = &.{ "the checkpoints' headers differ", "seed 45", "seed 46" } },
        .{ .other = .{ .build = fx.same.build, .checkpoint = fx.same.checkpoint, .options = .of("rows=64 capacity=256") }, .err = error.OptionsDiffer, .names = &.{ "the options differ", "rows=80", "rows=64" } },
        // the same label over other bytes: the digest is what is compared
        .{ .other = .{ .build = fx.same.build, .checkpoint = .{ .digest = channel_file.digest(&.{ "another", "header" }), .label = fx.same.checkpoint.label }, .options = fx.same.options }, .err = error.CheckpointsDiffer, .names = &.{"the checkpoints' headers differ"} },
    };
    for (cases) |case| {
        var ends: Ends = undefined;
        try ends.init();
        defer ends.deinit();
        var got: channel_file.Id = @splat(0);
        try ends.zero.offer(fx.same, &fx.id);
        // rank 1 refuses and says why; it takes no id from a rank it refuses
        try testing.expectError(case.err, ends.one.answer(case.other, &got));
        try testing.expectEqualSlices(u8, &@as(channel_file.Id, @splat(0)), &got);
        // rank 0 is told, in the same words
        try testing.expectError(error.PeerFailed, ends.zero.conclude(fx.same));
        try testing.expectEqualStrings(ends.one.said(), ends.zero.said());
        for (case.names) |name| try testing.expect(std.mem.indexOf(u8, ends.zero.said(), name) != null);
    }

    // rank 0 checks rank 1's hello too: a rank 1 that answered without checking is refused there
    var ends: Ends = undefined;
    try ends.init();
    defer ends.deinit();
    var got: channel_file.Id = @splat(0);
    try ends.zero.offer(fx.same, &fx.id);
    try ends.one.answer(fx.same, &got);
    try testing.expectError(error.OptionsDiffer, ends.zero.conclude(cases[2].other));
    try testing.expectError(error.PeerFailed, ends.one.next());
    try testing.expect(std.mem.indexOf(u8, ends.one.said(), "the options differ") != null);
}

test "two ranks of the same index, another protocol and bytes that are no hello are refused" {
    {
        var ends: Ends = undefined;
        try ends.init();
        defer ends.deinit();
        var got: channel_file.Id = undefined;
        // rank 0's hello met by a rank that also thinks it is rank 0
        try ends.zero.offer(fx.same, &fx.id);
        try ends.one.offer(fx.same, &fx.id);
        try testing.expectError(error.RanksCollide, ends.zero.conclude(fx.same));
        try testing.expectError(error.RanksCollide, ends.one.conclude(fx.same));
        _ = &got;
    }
    {
        var ends: Ends = undefined;
        try ends.init();
        defer ends.deinit();
        var got: channel_file.Id = undefined;
        var hello = [_]u8{ 14, 0, 0, 0, @backingInt(channel_file.Kind.hello) } ++ channel_file.magic.* ++ [_]u8{ 9, 0, 0, 0, 0, 2 } ++ [_]u8{ 0, 0, 0, 0 };
        try ends.raw(&hello);
        try testing.expectError(error.ProtocolsDiffer, ends.one.answer(fx.same, &got));
        hello[5] = 'x';
        try ends.raw(&hello);
        try testing.expectError(error.NotARank, ends.one.answer(fx.same, &got));
        // a request where a hello is due
        try ends.zero.request("early");
        try testing.expectError(error.UnexpectedFrame, ends.one.answer(fx.same, &got));
    }
}

test "a frame cut short is TruncatedFrame: inside its header, and inside its payload" {
    {
        var ends: Ends = undefined;
        try ends.init();
        defer ends.deinit();
        try ends.raw(&.{ 100, 0, 0 });
        ends.close(0);
        try testing.expectError(error.TruncatedFrame, ends.one.receive());
    }
    {
        var ends: Ends = undefined;
        try ends.init();
        defer ends.deinit();
        // a whole frame first, which arrives; then 100 bytes announced and 10 sent
        try ends.zero.request("whole");
        try ends.raw(&([_]u8{ 100, 0, 0, 0, @backingInt(channel_file.Kind.request) } ++ @as([10]u8, @splat(7))));
        ends.close(0);
        try testing.expectEqualStrings("whole", (try ends.one.next()).request);
        try testing.expectError(error.TruncatedFrame, ends.one.next());
    }
}

test "a length past the limit and a kind that is none are refused before anything is read for them" {
    var ends: Ends = undefined;
    try ends.init();
    defer ends.deinit();
    var header: [5]u8 = undefined;
    std.mem.writeInt(u32, header[0..4], channel_file.max_payload + 1, .little);
    header[4] = @backingInt(channel_file.Kind.request);
    try ends.raw(&header);
    try testing.expectError(error.FrameTooLarge, ends.one.receive());
    try ends.raw(&.{ 0, 0, 0, 0, 99 });
    try testing.expectError(error.UnknownFrame, ends.one.receive());
    try testing.expectError(error.FrameTooLarge, ends.zero.send(.request, @as([*]const u8, @ptrCast(&header))[0 .. channel_file.max_payload + 1]));
}

test "a peer that closes is PeerGone at the next frame, at a barrier, and under a write: no step waits" {
    {
        var ends: Ends = undefined;
        try ends.init();
        defer ends.deinit();
        // what it sent before closing still arrives
        try ends.zero.request("last words");
        ends.close(0);
        try testing.expectEqualStrings("last words", (try ends.one.next()).request);
        try testing.expectError(error.PeerGone, ends.one.next());
        try testing.expectError(error.PeerGone, ends.one.next());
    }
    {
        var ends: Ends = undefined;
        try ends.init();
        defer ends.deinit();
        ends.close(1);
        try testing.expectError(error.PeerGone, ends.zero.barrier(3));
        // a write to a closed peer fails once the reset has come back
        var failed = false;
        for (0..64) |_| ends.zero.request("into the void") catch |err| {
            try testing.expectEqual(error.PeerGone, err);
            failed = true;
            break;
        };
        try testing.expect(failed);
    }
    {
        var ends: Ends = undefined;
        try ends.init();
        defer ends.deinit();
        var got: channel_file.Id = undefined;
        // a rank that dies in the handshake
        try ends.zero.offer(fx.same, &fx.id);
        ends.close(1);
        try testing.expectError(error.PeerGone, ends.zero.conclude(fx.same));
        _ = &got;
    }
}

test "a failure is the peer's text, and a barrier at another step is OutOfStep on one rank and PeerFailed on the other" {
    var ends: Ends = undefined;
    try ends.init();
    defer ends.deinit();
    ends.one.fail("window 3, layer 2: the gathered partials differ");
    try testing.expectError(error.PeerFailed, ends.zero.next());
    try testing.expectEqualStrings("window 3, layer 2: the gathered partials differ", ends.zero.said());

    try ends.one.send(.barrier, &std.mem.toBytes(std.mem.nativeToLittle(u64, 5)));
    try testing.expectError(error.OutOfStep, ends.zero.barrier(4));
    try testing.expectEqual(@as(u64, 4), (try ends.one.next()).barrier);
    try testing.expectError(error.PeerFailed, ends.one.next());
    try testing.expect(std.mem.indexOf(u8, ends.one.said(), "barrier 4 here, 5 on the other rank") != null);
}

test "a digest tells lists that concatenate alike, and a label is cut to a line" {
    try testing.expect(!std.mem.eql(u8, &channel_file.digest(&.{ "ab", "c" }), &channel_file.digest(&.{ "a", "bc" })));
    try testing.expectEqualSlices(u8, &channel_file.digest(&.{"abc"}), &Named.of("abc").digest);
    const long: [400]u8 = @splat('o');
    try testing.expect(Named.of(&long).label.len < 256);
}
