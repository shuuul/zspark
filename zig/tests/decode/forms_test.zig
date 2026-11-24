//! `forms.zig`'s tests.

const std = @import("std");
const forms = @import("../../src/decode/forms.zig");
const draft = @import("../../src/decode/draft.zig");

const testing = std.testing;

test "a window takes one form: bf16 up to the bound, every row quantized above it" {
    try testing.expectEqual(@as(usize, 1), forms.quantizedFrom(1, 80));
    try testing.expectEqual(@as(usize, draft.forward_max_rows), forms.quantizedFrom(draft.forward_max_rows, 80));
    try testing.expectEqual(@as(usize, 80), forms.quantizedFrom(80, 80));
    try testing.expectEqual(@as(usize, 0), forms.quantizedFrom(81, 80));
    try testing.expectEqual(@as(usize, 0), forms.quantizedFrom(4096, 48));
    // a call outside a window quantizes no row
    try testing.expect(forms.none > 1 << 40);
}

test "a round's replies keep bf16 rows before its prompt chunk, whatever their count" {
    const Part = struct { rows: usize };
    const Segment = struct { tokens: []const i32 };
    // replies alone: none quantized, however many rows they come to
    const replies = [_]Part{ .{ .rows = 16 }, .{ .rows = 16 }, .{ .rows = 16 }, .{ .rows = 16 }, .{ .rows = 16 }, .{ .rows = 16 } };
    try testing.expectEqual(@as(usize, 96), forms.formFrom(Part, &replies, 80));
    // a prompt chunk after them: its rows on
    const beside = [_]Part{ .{ .rows = 5 }, .{ .rows = 1 }, .{ .rows = 512 } };
    try testing.expectEqual(@as(usize, 6), forms.formFrom(Part, &beside, 80));
    // a short last chunk is a short window's: bf16
    const short = [_]Part{ .{ .rows = 5 }, .{ .rows = 80 } };
    try testing.expectEqual(@as(usize, 85), forms.formFrom(Part, &short, 80));
    // two wide chunks: both quantized
    const wide = [_]Part{ .{ .rows = 3 }, .{ .rows = 200 }, .{ .rows = 300 } };
    try testing.expectEqual(@as(usize, 3), forms.formFrom(Part, &wide, 80));
    // segments count their tokens
    const tokens: [100]i32 = @splat(0);
    const segments = [_]Segment{ .{ .tokens = tokens[0..4] }, .{ .tokens = &tokens } };
    try testing.expectEqual(@as(usize, 4), forms.formFrom(Segment, &segments, 80));
    try testing.expectEqual(@as(usize, 104), forms.formFrom(Segment, &segments, 100));
}
