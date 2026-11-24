//! `block.zig`'s tests: where `cuda.cut` places the parts of one allocation.

const std = @import("std");
const block = @import("../../src/cuda/block.zig");
const testing = std.testing;

test "the parts of one allocation each begin where cudaMalloc would align it, after the part before it, and one of no byte takes none" {
    // a kept state's shapes in small: f32 states, bf16 conv inputs of odd widths, a PLE tail of no column
    const sizes = [_]usize{ 4 * 2 * 8 * 8, 2 * 3 * 7, 2 * 3 * 7, 0, 4 * 2 * 8 * 8, 2 * 5, 0 };
    var end: usize = 0;
    var last: usize = 0;
    for (sizes) |size| {
        const at = block.place(end, size);
        try testing.expect(at >= end);
        if (size > 0) {
            try testing.expectEqual(0, at % block.alignment);
            try testing.expect(at - end < block.alignment);
            try testing.expect(at >= last);
            last = at + size;
        } else try testing.expectEqual(end, at);
        end = at + size;
    }
    // 512, then 42 at 512, 42 at 768, nothing, 512 at 1024, 10 at 1536, nothing
    try testing.expectEqual(1546, end);
    try testing.expectEqual(0, block.place(0, 1));
    try testing.expectEqual(256, block.place(1, 1));
    try testing.expectEqual(256, block.place(256, 1));
}
