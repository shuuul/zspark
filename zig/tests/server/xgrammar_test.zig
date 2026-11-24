//! `xgrammar.zig`'s tests: skipped where `libtfgrammar` cannot be loaded (a test binary finds it on the
//! loader's path alone, `LD_LIBRARY_PATH`; `zig build grammar` puts it in `zig-out/lib`).

const std = @import("std");
const api = @import("engine_api");
const grammar = @import("../../src/text/grammar.zig");
const xgrammar = @import("../../src/server/xgrammar.zig");
const testing = std.testing;

const tokenizer_json = @embedFile("fixtures/text/tiny_tokenizer.json");
/// The tiny tokenizer's ids are below 291; logits a row wider than xgrammar's, as a padded vocabulary is.
const vocab = 320;
const end = 281; // `<|end|>`

/// The library's paths are allocated once a process and kept (upstream's `load` is handed the server's arena).
var paths: std.heap.ArenaAllocator = .init(std.heap.page_allocator);

fn open() !api.grammar.Compiler {
    var err: [512]u8 = undefined;
    return api.grammar.Compiler.open(testing.io, paths.allocator(), tokenizer_json, vocab, &.{end}, &err) catch |e| switch (e) {
        error.NoGrammarLibrary => return error.SkipZigTest,
        else => return e,
    };
}

/// The mask after `tokens`, taken one after another by a constraint of its own.
fn straight(gpa: std.mem.Allocator, compiler: *api.grammar.Compiler, text: []const u8, tokens: []const u32, mask: []u32) !void {
    var err: [512]u8 = undefined;
    const made = try xgrammar.compile(gpa, testing.io, compiler, .regex, text, mask.len, &err);
    defer made.deinit();
    var at = try made.start();
    for (tokens) |token| at = (try made.accept(at, token)).?;
    try made.fill(at, mask);
}

test "a position is a value: masks along two branches and back are a straight walk's, and a refused token leaves the place" {
    const gpa = testing.allocator;
    var compiler = try open();
    defer compiler.close();
    const words = vocab / 32 + 1;
    const text = "(ab|ac|b)c*";
    var err: [512]u8 = undefined;
    const made = try xgrammar.compile(gpa, testing.io, &compiler, .regex, text, words, &err);
    defer made.deinit();
    // the byte-level ids of `a`, `b` and `c` in the tiny tokenizer
    var ids: [3]u32 = undefined;
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, tokenizer_json, .{});
        defer parsed.deinit();
        const table = parsed.value.object.get("model").?.object.get("vocab").?.object;
        for (&ids, "abc") |*id, ch| id.* = @intCast(table.get(&.{ch}).?.integer);
    }
    const mask = try gpa.alloc(u32, words);
    defer gpa.free(mask);
    const want = try gpa.alloc(u32, words);
    defer gpa.free(want);

    const start = try made.start();
    try made.fill(start, mask);
    try testing.expect(grammar.allowed(mask, ids[0]) and grammar.allowed(mask, ids[1]) and !grammar.allowed(mask, ids[2]));
    try testing.expect(!grammar.allowed(mask, end));
    try testing.expectEqual(null, try made.accept(start, ids[2]));

    const after_a = (try made.accept(start, ids[0])).?;
    const ab = (try made.accept(after_a, ids[1])).?;
    const ac = (try made.accept(after_a, ids[2])).?;
    const b = (try made.accept(start, ids[1])).?;
    for ([_]struct { grammar.Position, []const u32 }{
        .{ ab, &.{ ids[0], ids[1] } },
        .{ b, &.{ids[1]} },
        .{ after_a, &.{ids[0]} },
        .{ ac, &.{ ids[0], ids[2] } },
        .{ start, &.{} },
    }) |case| {
        try made.fill(case[0], mask);
        try straight(gpa, &compiler, text, case[1], want);
        try testing.expectEqualSlices(u32, want, mask);
    }
    // whole after `ab`: the end token may come, and nothing after it
    try made.fill(ab, mask);
    try testing.expect(grammar.allowed(mask, end) and grammar.allowed(mask, ids[2]));
    const done = (try made.accept(ab, end)).?;
    try testing.expect(made.ended(done) and !made.ended(ab));
    try made.fill(done, mask);
    for (mask) |w| try testing.expectEqual(0, w);
    try testing.expectEqual(null, try made.accept(done, ids[2]));
    // the bits past xgrammar's vocabulary are clear
    try made.fill(after_a, mask);
    for (vocab..words * 32) |id| try testing.expect(!grammar.allowed(mask, @intCast(id)));
}

test "a grammar xgrammar cannot compile says why" {
    const gpa = testing.allocator;
    var compiler = try open();
    defer compiler.close();
    var err: [512]u8 = undefined;
    try testing.expectError(error.Grammar, xgrammar.compile(gpa, testing.io, &compiler, .regex, "(", vocab / 32, &err));
    try testing.expect(std.mem.sliceTo(&err, 0).len > 0);
}
