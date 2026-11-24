//! `shelf.zig`'s tests, on the host: states held ascending, found and dropped by position, and their buffers
//! lent to the next taken, whole or as what a state leaves.

const std = @import("std");
const testing = std.testing;
const shelf_file = @import("../../src/decode/shelf.zig");

const fx = struct {
    /// A state whose buffers are a count of the frees that would run.
    const State = struct {
        pos: usize,
        freed: *usize,
        pub fn deinit(self: *State) void {
            self.freed.* += 1;
            self.* = undefined;
        }
        pub fn bytes(_: State) usize {
            return 100;
        }
    };

    /// A state that leaves one part's buffers when it goes and frees the rest, as GLM's mark.
    const Mark = struct {
        pos: usize,
        part: ?State,
        rest: *usize,
        pub fn deinit(self: *Mark) void {
            if (self.part) |*one| one.deinit();
            self.rest.* += 1;
        }
        pub fn release(self: *Mark, gpa: std.mem.Allocator, spare: *shelf_file.Spares(State)) void {
            if (self.part) |one| spare.give(gpa, one);
            self.part = null;
            self.deinit();
        }
    };
};

test "held states: found by position, dropped to the spares, lent newest first" {
    var freed: usize = 0;
    var shelf: shelf_file.Shelf(fx.State, fx.State) = .{ .gpa = testing.allocator };
    defer shelf.deinit();
    for ([_]usize{ 4, 9, 20 }) |pos| {
        try testing.expect(shelf.above(pos));
        try shelf.held.append(testing.allocator, .{ .pos = pos, .freed = &freed });
    }
    try testing.expect(!shelf.above(20) and !shelf.above(5));
    try testing.expectEqual(@as(?usize, 1), shelf.find(9));
    try testing.expectEqual(@as(?usize, null), shelf.find(10));
    var out: [3]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 4, 9, 20 }, shelf.positions(&out));
    shelf.drop(shelf.find(9).?);
    shelf.replace(shelf.find(20).?, .{ .pos = 20, .freed = &freed });
    try testing.expectEqual(@as(usize, 2), shelf.spare.count());
    try testing.expectEqual(@as(usize, 200), shelf.spare.bytes());
    shelf.dropAbove(4);
    try testing.expectEqual(@as(usize, 1), shelf.held.items.len);
    // the latest given lent first: the one at 20 that `dropAbove` left
    try testing.expectEqual(@as(usize, 20), shelf.lend().?.pos);
    shelf.clear();
    try testing.expectEqual(@as(usize, 0), shelf.held.items.len);
    try testing.expectEqual(@as(usize, 0), freed);
    shelf.spare.trim(1);
    try testing.expectEqual(@as(usize, 2), freed);
    try testing.expectEqual(@as(usize, 9), shelf.lend().?.pos);
}

test "a state that leaves a part: the part spared, the rest freed" {
    var parts: usize = 0;
    var rest: usize = 0;
    var shelf: shelf_file.Shelf(fx.Mark, fx.State) = .{ .gpa = testing.allocator };
    for ([_]usize{ 3, 8 }) |pos| try shelf.held.append(testing.allocator, .{ .pos = pos, .part = .{ .pos = pos, .freed = &parts }, .rest = &rest });
    try shelf.held.append(testing.allocator, .{ .pos = 12, .part = null, .rest = &rest });
    shelf.drop(0);
    try testing.expectEqual(@as(usize, 1), rest);
    try testing.expectEqual(@as(usize, 0), parts);
    try testing.expectEqual(@as(usize, 1), shelf.spare.count());
    shelf.clear();
    try testing.expectEqual(@as(usize, 3), rest);
    try testing.expectEqual(@as(usize, 2), shelf.spare.count());
    shelf.deinit();
    try testing.expectEqual(@as(usize, 2), parts);
}

test "arrays of states, a rank's each, held at rank 0's position" {
    var freed: usize = 0;
    var shelf: shelf_file.Shelf([2]fx.State, [2]fx.State) = .{ .gpa = testing.allocator };
    try shelf.held.append(testing.allocator, .{ .{ .pos = 5, .freed = &freed }, .{ .pos = 5, .freed = &freed } });
    try testing.expectEqual(@as(?usize, 0), shelf.find(5));
    try shelf.spare.reserve(testing.allocator);
    shelf.drop(0);
    try testing.expectEqual(@as(usize, 1), shelf.spare.count());
    shelf.deinit();
    try testing.expectEqual(@as(usize, 2), freed);
}
