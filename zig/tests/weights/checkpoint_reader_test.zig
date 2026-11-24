//! `weights/checkpoint_reader.zig`'s config.json readers: a refusal names the object it read, and a default stands
//! where a field is absent or of another kind.

const std = @import("std");
const testing = std.testing;
const reader = @import("../../src/weights/checkpoint_reader.zig");

test "a config's field readers: counts, defaults, and refusals naming the object read" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"n\": 3, \"f\": 0.5, \"neg\": -1, \"on\": true, \"s\": \"x\", \"o\": {}}", .{});
    const fields = root.object;
    var diagnostic: reader.Diagnostic = .{};
    const p: reader.Parser = .{ .arena = a, .diagnostic = &diagnostic };
    try testing.expectEqual(@as(usize, 3), try p.uint(fields, "n"));
    try testing.expectEqual(@as(f64, 3), p.numberOr(fields, "n", 9));
    try testing.expectEqual(@as(f64, 0.5), p.numberOr(fields, "f", 9));
    try testing.expectEqual(@as(f64, 9), p.numberOr(fields, "s", 9));
    try testing.expect(p.flagOr(fields, "on", false));
    try testing.expect(!p.flagOr(fields, "n", false));
    try testing.expectEqualStrings("x", p.stringOr(fields, "s", "y"));
    try testing.expectEqualStrings("y", p.stringOr(fields, "gone", "y"));
    _ = try p.object(fields, "o");
    try testing.expectError(error.InvalidConfig, p.uint(fields, "neg"));
    try testing.expectEqualStrings("config.json: neg is not a count", diagnostic.message);
    try testing.expectError(error.InvalidConfig, p.object(fields, "n"));
    try testing.expectEqualStrings("config.json: n is not an object", diagnostic.message);
    const inner: reader.Parser = .{ .arena = a, .diagnostic = &diagnostic, .where = "language_config" };
    try testing.expectError(error.InvalidConfig, inner.uint(fields, "gone"));
    try testing.expectEqualStrings("language_config has no gone", diagnostic.message);
}
