//! `preprocess.zig`'s host tests (spec 147 WS-02): the crop grid of sizes covering the aspect ratios and caps
//! the checkpoint's HF code (`modeling_unlimitedocr.py`'s `find_closest_aspect_ratio`, `dynamic_preprocess`)
//! meets, and the image tokens a grid's page holds.

const std = @import("std");
const testing = std.testing;
const pre = @import("../../../src/models/unlimited_ocr/preprocess.zig");
const requests = @import("../../../src/models/unlimited_ocr/requests.zig");

test "unlimited_ocr preprocess: gundam grids" {
    // `image_size` 640 on each side gives 1 x 1 (the global view alone)
    try testing.expectEqual(pre.Grid{ .w = 1, .h = 1 }, pre.gundamGrid(640, 640));
    try testing.expectEqual(pre.Grid{ .w = 1, .h = 1 }, pre.gundamGrid(640, 33));
    try testing.expectEqual(pre.Grid{ .w = 1, .h = 1 }, pre.gundamGrid(33, 640));
    // any larger picture is cut by `dynamic_preprocess`: 2 to `max_tiles` tiles, the grid
    // `find_closest_aspect_ratio` picks nearest the picture's aspect
    const sizes = [_][2]usize{
        .{ 641, 33 },    .{ 33, 641 },    .{ 1920, 1080 }, .{ 1080, 1920 },
        .{ 2479, 3508 }, .{ 1654, 2339 }, .{ 4200, 2970 }, .{ 2970, 4200 },
        .{ 100, 40000 }, .{ 40000, 100 },
    };
    for (sizes) |size| {
        const g = pre.gundamGrid(size[0], size[1]);
        const count = g.w * g.h;
        try testing.expect(count >= 2 and count <= pre.max_tiles and g.w <= pre.max_tiles and g.h <= pre.max_tiles);
    }
    // HF's area tie-break: 1 x 2 for a tall page, 2 x 2 for a 1280 square, 4 x 4 for a 2560 square.
    try testing.expectEqual(pre.Grid{ .w = 1, .h = 2 }, pre.gundamGrid(641, 1282));
    try testing.expectEqual(pre.Grid{ .w = 2, .h = 2 }, pre.gundamGrid(1280, 1280));
    try testing.expectEqual(pre.Grid{ .w = 4, .h = 4 }, pre.gundamGrid(2560, 2560));
}

test "unlimited_ocr preprocess: image tokens a page holds" {
    // `imageTokens` is a page's rows: the global view's 273 rows alone (1 x 1), plus the tiles' where the
    // picture is cut (10 queries a side, a 11th column the row's separator). The largest grid (1 x 16 tiles)
    // is `requests.page_tokens` (2,033).
    try testing.expectEqual(@as(usize, 273), pre.imageTokens(.{ .w = 1, .h = 1 }));
    try testing.expectEqual(pre.imageTokens(.{ .w = 1, .h = pre.max_tiles }), requests.page_tokens);
    try testing.expect(pre.imageTokens(.{ .w = 1, .h = 2 }) > 273);
    try testing.expect(pre.imageTokens(.{ .w = 4, .h = 4 }) > 273);
}
