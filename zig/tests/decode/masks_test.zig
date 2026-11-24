//! `masks.zig`'s tests.

const std = @import("std");
const masks_file = @import("../../src/decode/masks.zig");
const grammar = @import("../../src/text/grammar.zig");
const Error = grammar.Error;
const Masks = masks_file.Masks;
const minus_infinity = masks_file.minus_infinity;
const only = masks_file.only;
const write = masks_file.write;

const testing = std.testing;

test "only: the token of a mask with one bit, and no other mask's" {
    var mask: [5]u32 = @splat(0);
    try testing.expectEqual(null, only(&mask));
    for ([_]u32{ 0, 31, 32, 95, 159 }) |token| {
        mask = @splat(0);
        mask[token / 32] = @as(u32, 1) << @intCast(token % 32);
        try testing.expectEqual(token, only(&mask).?);
        // a second bit, in the same word or another
        for ([_]u32{ 1, 33, 158 }) |other| {
            if (other == token) continue;
            var two = mask;
            two[other / 32] |= @as(u32, 1) << @intCast(other % 32);
            try testing.expectEqual(null, only(&two));
        }
    }
}

test "write: a token keeps its logit exactly where its bit is set" {
    var random: std.Random.DefaultPrng = .init(3);
    for ([_]usize{ 1, 31, 32, 33, 291, 320 }) |width| {
        const logits = try testing.allocator.alloc(u16, width);
        defer testing.allocator.free(logits);
        const mask = try testing.allocator.alloc(u32, (width + 31) / 32);
        defer testing.allocator.free(mask);
        for (0..8) |round| {
            for (logits, 0..) |*logit, token| logit.* = @intCast(0x3f00 + token);
            for (mask, 0..) |*word, index| word.* = switch ((round + index) % 4) {
                0 => 0,
                1 => std.math.maxInt(u32),
                else => random.random().int(u32),
            };
            write(mask, logits);
            for (logits, 0..) |logit, token| {
                const expected: u16 = if (grammar.allowed(mask, @intCast(token))) @intCast(0x3f00 + token) else minus_infinity;
                try testing.expectEqual(expected, logit);
            }
        }
    }
}

/// A constraint a test makes up: place i allows `allowed[i]`, and the reply ends after the last place.
const Scripted = struct {
    allowed: []const []const u32,

    const table: grammar.Constraint.Table = .{ .start = start, .fill = fill, .accept = accept, .ended = ended, .deinit = deinit };

    fn constraint(self: *Scripted) grammar.Constraint {
        return .{ .context = self, .words = 1, .table = &table };
    }

    fn start(_: *anyopaque) Error!grammar.Position {
        return @fromBackingInt(@intCast(0));
    }

    fn fill(context: *anyopaque, at: grammar.Position, mask: []u32) Error!void {
        const self: *Scripted = @ptrCast(@alignCast(context));
        mask[0] = 0;
        for (self.allowed[@backingInt(at)]) |token| mask[0] |= @as(u32, 1) << @intCast(token);
    }

    fn accept(context: *anyopaque, at: grammar.Position, token: u32) Error!?grammar.Position {
        const self: *Scripted = @ptrCast(@alignCast(context));
        return if (std.mem.indexOfScalar(u32, self.allowed[@backingInt(at)], token) != null) @fromBackingInt(@intCast(@backingInt(at) + 1)) else null;
    }

    fn ended(context: *anyopaque, at: grammar.Position) bool {
        const self: *Scripted = @ptrCast(@alignCast(context));
        return @backingInt(at) == self.allowed.len;
    }

    fn deinit(_: *anyopaque) void {}
};

test "certain: the drafts a constraint forced, from the first, and no draft a model chose or another walked" {
    var scripted: Scripted = .{ .allowed = &.{ &.{7}, &.{8}, &.{ 3, 4 }, &.{9}, &.{ 5, 6 }, &.{1} } };
    var mask: [1]u32 = undefined;
    var masks: Masks = try .init(scripted.constraint(), &mask, null);
    try testing.expectEqual(0, masks.certain(&.{ 7, 8 }));
    // as a drafter asks: two forced, one of two a model chooses, one forced after it
    try testing.expectEqual(7, (try masks.next(&.{}, true)).token);
    try testing.expectEqual(1, masks.certain(&.{ 7, 8 }));
    try testing.expectEqual(8, (try masks.next(&.{7}, true)).token);
    try testing.expect((try masks.next(&.{ 7, 8 }, true)) == .among);
    try testing.expectEqual(9, (try masks.next(&.{ 7, 8, 3 }, true)).token);
    // the forced draft after a chosen one is not certain: its row is only run if the chosen one was right
    try testing.expectEqual(2, masks.certain(&.{ 7, 8, 3, 9 }));
    try testing.expectEqual(2, masks.certain(&.{ 7, 8 }));
    try testing.expectEqual(1, masks.certain(&.{7}));
    try testing.expectEqual(0, masks.certain(&.{}));
    // other drafts than the ones it gave
    try testing.expectEqual(0, masks.certain(&.{ 8, 8 }));
    try testing.expectEqual(1, masks.certain(&.{ 7, 9 }));
    // the window keeps them, and its newest token starts again
    try testing.expectEqual(4, try masks.window(&.{ 7, 8, 3, 9 }));
    try testing.expectEqual(2, masks.certain(&.{ 7, 8, 3, 9 }));
    try masks.commit(3, 4);
    try testing.expectEqual(0, masks.certain(&.{9}));
    try testing.expectEqual(9, (try masks.next(&.{}, true)).token);
    try testing.expectEqual(1, masks.certain(&.{9}));
    // a draft a window walked without the guide having forced it: allowed, and not known to be the only one
    try masks.commit(1, 9);
    try testing.expectEqual(1, try masks.window(&.{5}));
    try testing.expectEqual(0, masks.certain(&.{5}));
    // walked as another draft, the place loses what `next` said of it
    var again: Masks = try .init(scripted.constraint(), &mask, null);
    try testing.expectEqual(7, (try again.next(&.{}, true)).token);
    try testing.expectEqual(0, try again.window(&.{2}));
    try testing.expectEqual(0, again.certain(&.{7}));
}

test "advance: the place after tokens a decoder that is gone chose is the one windows and commits leave, the gate included" {
    var scripted: Scripted = .{ .allowed = &.{ &.{7}, &.{8}, &.{ 3, 4 }, &.{9}, &.{ 5, 6 }, &.{1} } };
    var mask: [1]u32 = undefined;
    // the reply thinks: tokens before the gate (30) are free, and the constraint begins after it
    const sent = [_]i32{ 12, 13, 30, 7, 8, 4 };
    var stepped: Masks = try .init(scripted.constraint(), &mask, 30);
    for (sent) |token| {
        _ = try stepped.window(&.{});
        try stepped.commit(1, token);
    }
    var moved: Masks = try .init(scripted.constraint(), &mask, 30);
    try moved.advance(&sent);
    try testing.expectEqual(stepped.newest, moved.newest);
    try testing.expectEqual(9, (try moved.next(&.{}, true)).token);
    // without the advance the constraint would begin again at the reply's middle
    var fresh: Masks = try .init(scripted.constraint(), &mask, 30);
    try testing.expect((try fresh.next(&.{}, true)) == .free);
    // a token the constraint does not take there was not chosen under it
    var wrong: Masks = try .init(scripted.constraint(), &mask, null);
    try testing.expectError(error.ConstraintFailed, wrong.advance(&.{ 7, 9 }));
}
