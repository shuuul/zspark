//! Captured-row alignment and fixed-set transcript metrics for Nemotron reference comparison.
const std = @import("std");
const testing = std.testing;
const reference = @import("../../../../src/models/nemotron/checks/reference.zig");
const safetensors = @import("../../../../src/weights/safetensors.zig");

test "reference: chunked prefill ids can be absent without accepting a rejected verify prefix" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const head = [_]u8{ 8, 0, 0, 0, 0, 0, 0, 0, '{', '}', ' ', ' ', ' ', ' ', ' ', ' ' };
    var file = try safetensors.File.fromHead(testing.allocator, &head, head.len);
    defer file.close();
    const map = try reference.rows(arena.allocator(), &file, &.{ 2, 2, 3 }, &.{ 0, 1, 2, 3, 4, 5, 6 }, &.{ 5, 88, 7 }, &.{ 1, 2, 3, 4 }, &.{ 5, 6, 7 });
    try testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3, 4, -1, -1 }, map.row_of);
    try testing.expectEqual(@as(usize, 2), map.unmapped);
}

test "reference: English WER shares the fixture scorer normalization and word boundaries" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const same = try reference.wer(a, "THE QUICK, BROWN FOX!", "the quick brown fox");
    try testing.expectEqual(@as(usize, 0), same.distance);
    try testing.expectEqual(@as(usize, 4), same.units);
    const quotes = try reference.wer(a, "HELLO", "“hello”");
    try testing.expectEqual(@as(usize, 0), quotes.distance);
    const one = try reference.wer(a, "hello", "goodbye");
    try testing.expectEqual(@as(usize, 1), one.distance);
    try testing.expectEqual(@as(usize, 1), one.units);
    try testing.expect(!one.codepoints);
    const edits = try reference.wer(a, "a b c d", "a x c d e");
    try testing.expectEqual(@as(usize, 2), edits.distance);
    try testing.expectEqual(@as(usize, 4), edits.units);
}

test "reference: Chinese CER removes whitespace but retains punctuation" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const error_rate = try reference.wer(arena.allocator(), "巴黎人", "巴 林人。");
    try testing.expectEqual(@as(usize, 2), error_rate.distance);
    try testing.expectEqual(@as(usize, 3), error_rate.units);
    try testing.expect(error_rate.codepoints);
}

test "reference: an empty reference counts every inserted word" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const error_rate = try reference.wer(arena.allocator(), "", "one two");
    try testing.expectEqual(@as(usize, 2), error_rate.distance);
    try testing.expectEqual(@as(f64, 2), error_rate.rate());
}

test "reference: orthogonal and matching rows retain their metric weights" {
    const theirs = [_]f32{ 1, 0, 0, 2 };
    const mine = [_]f32{ 0, 1, 0, 2 };
    const result = reference.compare(&mine, &theirs, 2, true);
    try testing.expectEqual(@as(usize, 1), result.picks_differing);
    try testing.expectApproxEqAbs(@as(f64, @sqrt(2.0)), result.rel_worst, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, @sqrt(0.4)), result.rel_all, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.8), result.cosine, 1e-12);
    try testing.expectEqual(@as(f64, 1), result.max_abs);
}
