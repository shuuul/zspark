//! `marks.zig`'s tests, on the host: which state is thinned when one stands at a shared prefix's end, and the
//! state kept for that prefix. Where a prompt's states are planned: Flash Next's `session_test.zig` and
//! `book_test.zig`, GLM's `session/marks_test.zig`.

const std = @import("std");
const testing = std.testing;
const marks = @import("../../src/decode/marks.zig");
const thinned = marks.thinned;
const keptOf = marks.keptOf;

test "the state at a shared prefix's end is never the one thinned" {
    var random: std.Random.DefaultPrng = .init(64);
    for (0..500) |_| {
        // a conversation that grows a turn at a time over a harness of `kept` tokens
        const kept = random.random().intRangeLessThan(usize, 512, 6000);
        var held: [9]usize = undefined;
        var n: usize = 0;
        var end = kept;
        var plain_lost = false;
        var plain: [9]usize = undefined;
        var m: usize = 0;
        for (0..80) |_| {
            held[n] = end;
            n += 1;
            plain[m] = end;
            m += 1;
            end += random.random().intRangeLessThan(usize, 40, 3000);
            if (n > 8) {
                const drop = thinned(held[0..n], end, kept);
                try std.testing.expect(held[drop] != kept and drop != n - 1);
                std.mem.copyForwards(usize, held[drop .. n - 1], held[drop + 1 .. n]);
                n -= 1;
            }
            if (m > 8) {
                const drop = thinned(plain[0..m], end, null);
                if (plain[drop] == kept) plain_lost = true;
                std.mem.copyForwards(usize, plain[drop .. m - 1], plain[drop + 1 .. m]);
                m -= 1;
            }
            try std.testing.expectEqual(kept, held[0]);
        }
        // what the rule is for: without it the harness's state goes as the conversation grows
        try std.testing.expect(plain_lost);
    }
    try std.testing.expectEqual(@as(usize, 0), thinned(&.{ 10, 20 }, 30, 10));
    try std.testing.expectEqual(thinned(&.{ 10, 20, 500, 900 }, 1000, null), thinned(&.{ 10, 20, 500, 900 }, 1000, 7));
}

test "marks thin toward the start" {
    const held = [_]usize{ 8, 16, 24, 32 };
    try testing.expectEqual(0, thinned(&held, 40, null));
    try testing.expectEqual(1, thinned(&held, 40, 8));
}

test "the state kept for a shared prefix stands at the first user message's start" {
    // system at 0, user at 4700, the reply's header at 4730; the probe parts three tokens into the user's
    try testing.expectEqual(@as(?usize, 4700), keptOf(&.{ 0, 4700, 4730 }, &.{ 2655, 4191, 4703 }));
    try testing.expectEqual(@as(?usize, null), keptOf(&.{ 0, 4700, 4730 }, &.{}));
    try testing.expectEqual(@as(?usize, null), keptOf(&.{0}, &.{600}));
    try testing.expectEqual(@as(?usize, 4700), keptOf(&.{ 0, 4700, 4703 }, &.{4703}));
}
