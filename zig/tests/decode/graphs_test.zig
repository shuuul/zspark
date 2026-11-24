//! Capture-cache ownership, key cardinality, replay and error cleanup on a device buffer.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../src/cuda/cuda.zig");
const cache_file = @import("../../src/decode/graphs.zig");

fn failCapture(buffer: cuda.Buffer) (cuda.Error || error{CaptureFailed})!void {
    try buffer.zeroLaunched();
    return error.CaptureFailed;
}

fn expectWords(buffer: cuda.Buffer, expected: []const u32) !void {
    try cuda.synchronize();
    var words: [4]u32 = undefined;
    try buffer.download(std.mem.sliceAsBytes(&words));
    try testing.expectEqualSlices(u32, expected, &words);
}

test "capture records without launching, repeated keys add no graph, and a failed capture leaves no entry" {
    var buffer: cuda.Buffer = try .init(.u32, &.{4});
    defer buffer.deinit();
    const initial = [_]u32{ 7, 11, 13, 17 };
    try buffer.upload(std.mem.sliceAsBytes(&initial));
    var cache: cache_file.Cache(u32) = try .init(testing.allocator);
    defer cache.deinit();
    for ([_]u32{ 2, 5, 2, 9, 5 }) |key| {
        _ = try cache.capture(key, cuda.Buffer.zeroLaunched, .{buffer});
        try expectWords(buffer, &initial);
    }
    try testing.expectEqual(3, cache.count());
    for ([_]u32{ 2, 5, 9 }) |key| try testing.expect(cache.contains(key));
    const captured = cache.capturing();
    try testing.expect(captured > 0);
    _ = try cache.capture(2, failCapture, .{buffer});
    try testing.expectEqual(captured, cache.capturing());
    try testing.expectError(error.CaptureFailed, cache.capture(7, failCapture, .{buffer}));
    try testing.expectEqual(3, cache.count());
    try testing.expect(!cache.contains(7));
    try testing.expectEqual(null, cuda.currentStream());
    try expectWords(buffer, &initial);
    try cache.run(7, cuda.Buffer.zeroLaunched, .{buffer});
    try testing.expectEqual(4, cache.count());
    try expectWords(buffer, &.{ 0, 0, 0, 0 });
}

test "a borrowed cache accounts capture on its owner's stream and leaves the stream alive" {
    var stream: cuda.Stream = try .init();
    defer stream.deinit();
    var buffer: cuda.Buffer = try .init(.u32, &.{4});
    defer buffer.deinit();
    var cache: cache_file.Cache(u32) = .initBorrowed(testing.allocator, &stream);
    try testing.expect(cache.stream() == &stream);
    _ = try cache.capture(1, cuda.Buffer.zeroLaunched, .{buffer});
    try testing.expect(stream.capturing > 0);
    try testing.expectEqual(stream.capturing, cache.capturing());
    cache.deinit();
    // A fresh graph on the same stream must still capture and replay after the borrower is destroyed.
    var again: cache_file.Cache(u32) = .initBorrowed(testing.allocator, &stream);
    defer again.deinit();
    const initial = [_]u32{ 1, 2, 3, 4 };
    try buffer.upload(std.mem.sliceAsBytes(&initial));
    try again.run(1, cuda.Buffer.zeroLaunched, .{buffer});
    try expectWords(buffer, &.{ 0, 0, 0, 0 });
}
