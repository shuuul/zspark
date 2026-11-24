//! `lead.zig`'s test: what rank 0 tells of an operation is what rank 1 reads of it. Host only.

const std = @import("std");
const testing = std.testing;
const Channel = @import("../../src/ranks/channel.zig").Channel;
const lead_file = @import("../../src/ranks/lead.zig");

test "an operation's words come back as told" {
    var sent: std.Io.Writer.Allocating = .init(testing.allocator);
    defer sent.deinit();
    var nothing: std.Io.Reader = .fixed("");
    var channel: Channel = .init(testing.allocator, &nothing, &sent.writer);
    defer channel.deinit();
    var lead: lead_file.Lead = .{ .channel = &channel };
    try lead.tell(7, &.{ 3, 1 }, &.{ 5, -1, 154879 });
    try testing.expectEqual(@as(usize, 1), lead.told);

    var back: std.Io.Reader = .fixed(sent.written());
    var other: Channel = .init(testing.allocator, &back, &sent.writer);
    defer other.deinit();
    const told = lead_file.Told.read((try other.next()).request).?;
    try testing.expectEqual(@as(u32, 7), told.kind);
    try testing.expectEqual(@as(u32, 3), told.rest[0]);
    var out: [8]i32 = undefined;
    try testing.expectEqualSlices(i32, &.{ 5, -1, 154879 }, told.tokens(2, &out).?);
    // more tokens than a window holds, or fewer words than the arguments: not an operation
    var small: [2]i32 = undefined;
    try testing.expect(told.tokens(2, &small) == null);
    try testing.expect(told.tokens(6, &out) == null);
    try testing.expect(lead_file.Told.read("abc") == null);
}

/// What a hot lead's words reach: each saying kept whole, as the follower hears it.
const Heard = struct {
    said: [4][16]u32 = undefined,
    lens: [4]usize = undefined,
    n: usize = 0,

    fn say(context: *anyopaque, words: []const u32) anyerror!void {
        const self: *Heard = @ptrCast(@alignCast(context));
        @memcpy(self.said[self.n][0..words.len], words);
        self.lens[self.n] = words.len;
        self.n += 1;
    }

    fn of(self: *const Heard, at: usize) ?lead_file.Heard {
        // what the follower hears is the whole of the words, past those said
        var all: [16]u32 = @splat(0xdead);
        @memcpy(all[0..self.lens[at]], self.said[at][0..self.lens[at]]);
        return lead_file.Heard.read(&all);
    }
};

test "a hot lead's frames go in the words, a wide one on the channel, and it cools once" {
    var sent: std.Io.Writer.Allocating = .init(testing.allocator);
    defer sent.deinit();
    var nothing: std.Io.Reader = .fixed("");
    var channel: Channel = .init(testing.allocator, &nothing, &sent.writer);
    defer channel.deinit();
    var heard: Heard = .{};
    var lead: lead_file.Lead = .{ .channel = &channel, .over = .{ .context = &heard, .width = 16, .say = Heard.say } };

    // cold: the channel's
    try lead.tell(4, &.{9}, &.{});
    try testing.expectEqual(@as(usize, 0), heard.n);
    try lead.cool();
    try testing.expectEqual(@as(usize, 0), heard.n);

    lead.heat();
    try lead.tell(7, &.{ 3, 1 }, &.{ 5, -1 });
    const told = heard.of(0).?.told;
    try testing.expectEqual(@as(u32, 7), told.kind);
    try testing.expectEqual(@as(u32, 3), told.rest[0]);
    try testing.expectEqual(@as(u32, 1), told.rest[1]);
    var out: [4]i32 = undefined;
    try testing.expectEqualSlices(i32, &.{ 5, -1 }, told.tokens(2, &out).?);
    // as wide as the words: the channel's, the words saying so
    const wide: [15]i32 = @splat(2);
    try lead.tell(8, &.{}, &wide);
    try testing.expectEqual(@as(usize, 1), heard.lens[1]);
    try testing.expect(heard.of(1).? == .channel);
    try lead.cool();
    try testing.expect(heard.of(2).? == .cooled);
    try lead.cool();
    try testing.expectEqual(@as(usize, 3), heard.n);
    try testing.expectEqual(@as(usize, 3), lead.told);

    // the channel holds the cold frame and the wide one, in order
    var back: std.Io.Reader = .fixed(sent.written());
    var other: Channel = .init(testing.allocator, &back, &sent.writer);
    defer other.deinit();
    try testing.expectEqual(@as(u32, 4), lead_file.Told.read((try other.next()).request).?.kind);
    const far = lead_file.Told.read((try other.next()).request).?;
    try testing.expectEqual(@as(u32, 8), far.kind);
    try testing.expectEqual(@as(usize, 15), far.rest.len);
    // a count past the words heard: no frame
    try testing.expect(lead_file.Heard.read(&.{ 5, 1, 2 }) == null);
    // no words at all to say through: nothing heats
    var cold: lead_file.Lead = .{ .channel = &channel };
    cold.heat();
    try testing.expect(!cold.hot);
}
