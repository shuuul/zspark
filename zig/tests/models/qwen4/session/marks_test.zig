//! `pool/hall_keep.zig`'s stride, on the host: where a reply's stride keeps its state among rounds of any length. Which
//! state is thinned: `tests/decode/marks_test.zig`.

const std = @import("std");
const hall_keep = @import("../../../../src/models/qwen4/pool/hall_keep.zig");
const round_rows = hall_keep.round_rows;
const strideDue = hall_keep.strideDue;

test "a stride's state stands at the target or at most fifteen positions below it, whatever the rounds keep" {
    var random: std.Random.DefaultPrng = .init(54);
    for (0..2000) |_| {
        const base = random.random().intRangeLessThan(usize, 0, 4000);
        var target = base + 256;
        var pos = base + random.random().intRangeLessThan(usize, 1, 40);
        var taken: usize = 0;
        while (pos < base + 1500) {
            if (strideDue(pos, target)) {
                try std.testing.expect(pos <= target and pos + round_rows > target);
                target += 256;
                taken += 1;
            }
            pos += random.random().intRangeAtMost(usize, 1, round_rows);
        }
        try std.testing.expect(taken >= 4);
    }
}
