//! `hall_depth.zig`: where a seat's chain is cut by the round's width (spec 144 WS-04). Host only.

const std = @import("std");
const testing = std.testing;
const depth = @import("../../../../src/models/qwen4/pool/hall_depth.zig");

test "a seat's drafts are cut to four on two ranks while three seats or more decode, and nowhere else" {
    const default_depth = 15;
    for (1..9) |decoding| {
        const wide = decoding >= 3;
        // two ranks: the cap from three seats, none at one or two (a lone stream there is the pool's)
        try testing.expectEqual(@as(usize, if (wide) 4 else default_depth), depth.room(.two_ranks, true, decoding, default_depth));
        try testing.expectEqual(wide, depth.engages(.two_ranks, true, decoding));
        // one Spark: never
        try testing.expectEqual(@as(usize, default_depth), depth.room(.two_ranks, false, decoding, default_depth));
        try testing.expect(!depth.engages(.two_ranks, false, decoding));
        // a test's switches
        try testing.expectEqual(@as(usize, if (wide) 4 else default_depth), depth.room(.always, false, decoding, default_depth));
        try testing.expectEqual(@as(usize, default_depth), depth.room(.never, true, decoding, default_depth));
        // a chain asked for fewer than the cap keeps them, and one of none none
        for (0..5) |most| try testing.expectEqual(most, depth.room(.two_ranks, true, decoding, most));
    }
    try testing.expectEqual(@as(usize, 4), depth.most_drafts);
    try testing.expectEqual(@as(usize, 3), depth.from_seats);
}
