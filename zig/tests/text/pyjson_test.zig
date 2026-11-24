//! `pyjson.zig`'s tests.

const std = @import("std");
const pyjson = @import("../../src/text/pyjson.zig");
const Value = std.json.Value;
const loads = pyjson.loads;
const max_depth = pyjson.max_depth;

const testing = std.testing;
const template = @import("../../src/text/template.zig");

test "strings parse as json.loads parses them, and print as json.dumps prints the result" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(Value, gpa, @embedFile("fixtures/text/message_cases.json"), .{});
    defer parsed.deinit();
    var diag: template.Diagnostic = .{};
    var t = try template.Template.parse(gpa, "{{ x | tojson }}", &diag);
    defer t.deinit();
    const cases = parsed.value.object.get("loads").?.array.items;
    try testing.expect(cases.len >= 80);
    var refused: usize = 0;
    for (cases) |entry| {
        const text = entry.object.get("text").?.string;
        errdefer std.debug.print("text: {s}\n", .{text});
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const result = loads(arena, text);
        switch (entry.object.get("dumps").?) {
            .null => {
                try testing.expectError(error.InvalidJson, result);
                refused += 1;
            },
            .string => |want| {
                var variables: std.json.ObjectMap = .empty;
                try variables.put(arena, "x", try result);
                try testing.expectEqualStrings(want, try t.renderAlloc(arena, .{ .object = variables }, &diag));
            },
            else => unreachable,
        }
    }
    try testing.expect(refused >= 30);
}

test "what Python keeps and this parser does not: lone surrogates, integers past i64, very deep nesting" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try testing.expectError(error.InvalidJson, loads(arena, "\"\\ud800\""));
    try testing.expectError(error.InvalidJson, loads(arena, "\"\\ud800\\u0041\""));
    try testing.expectError(error.InvalidJson, loads(arena, "\"\\udc00\""));
    try testing.expectEqualStrings("-123456789012345678901234567890", (try loads(arena, "-123456789012345678901234567890")).number_string);
    try testing.expectEqual(@as(i64, std.math.minInt(i64)), (try loads(arena, "-9223372036854775808")).integer);
    const deep = try arena.alloc(u8, 2 * (max_depth + 2));
    @memset(deep[0 .. deep.len / 2], '[');
    @memset(deep[deep.len / 2 ..], ']');
    try testing.expectError(error.InvalidJson, loads(arena, deep));
    try testing.expect((try loads(arena, deep[2 .. deep.len - 2])) == .array);
}
