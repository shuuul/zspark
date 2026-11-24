//! `grammar.zig`'s tests.

const std = @import("std");
const grammar = @import("../../src/text/grammar.zig");
const Constraint = grammar.Constraint;
const Error = grammar.Error;
const Led = grammar.Led;
const Position = grammar.Position;

/// A constraint that takes any token and keeps its contract to the letter: a position is good for the reply
/// it was given in, and `start` forgets the ones before.
const Strict = struct {
    reply: u32 = 0,

    const table: Constraint.Table = .{ .start = start, .fill = fill, .accept = accept, .ended = ended, .deinit = destroy };

    fn at(self: *const Strict, taken: u32) Position {
        return @fromBackingInt(@as(u64, self.reply) << 32 | taken);
    }

    fn check(self: *const Strict, position: Position) Error!u32 {
        if (@backingInt(position) >> 32 != self.reply) return error.ConstraintFailed;
        return @truncate(@backingInt(position));
    }

    fn start(context: *anyopaque) Error!Position {
        const self: *Strict = @ptrCast(@alignCast(context));
        self.reply += 1;
        return self.at(0);
    }

    fn fill(context: *anyopaque, position: Position, mask: []u32) Error!void {
        const self: *Strict = @ptrCast(@alignCast(context));
        _ = try self.check(position);
        @memset(mask, std.math.maxInt(u32));
    }

    fn accept(context: *anyopaque, position: Position, token: u32) Error!?Position {
        const self: *Strict = @ptrCast(@alignCast(context));
        _ = token;
        return self.at(try self.check(position) + 1);
    }

    fn ended(_: *anyopaque, _: Position) bool {
        return false;
    }

    fn destroy(_: *anyopaque) void {}
};

test "a led constraint starts its inner one once a reply: the lead walked again leaves the document's places good" {
    const gpa = std.testing.allocator;
    var strict: Strict = .{};
    const inner: Constraint = .{ .context = &strict, .words = 2, .table = &Strict.table };
    try std.testing.expectEqual(inner.context, (try Led.wrap(gpa, inner, &.{})).context);
    const led = try Led.wrap(gpa, inner, &.{ 7, 40 });
    defer led.deinit();
    var mask: [2]u32 = undefined;
    for (0..2) |_| {
        const first = try led.start();
        try led.fill(first, &mask);
        try std.testing.expectEqualSlices(u32, &.{ 1 << 7, 0 }, &mask);
        try std.testing.expectEqual(null, try led.accept(first, 40));
        const second = (try led.accept(first, 7)).?;
        try led.fill(second, &mask);
        try std.testing.expectEqualSlices(u32, &.{ 0, 1 << 8 }, &mask);
        try std.testing.expect(!led.ended(second));
        const opened = (try led.accept(second, 40)).?;
        const inside = (try led.accept(opened, 3)).?;
        // a window's drafts are walked from the row before them, so the lead's last token is taken again
        // after the document has begun: the same place, and the ones after it still the reply's
        try std.testing.expectEqual(opened, (try led.accept(second, 40)).?);
        try led.fill(inside, &mask);
        try std.testing.expectEqualSlices(u32, &.{ std.math.maxInt(u32), std.math.maxInt(u32) }, &mask);
        _ = (try led.accept(inside, 9)).?;
    }
    try std.testing.expectEqual(2, strict.reply);
}
