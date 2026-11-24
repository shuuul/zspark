//! `messages.zig`'s tests.

const std = @import("std");
const messages_file = @import("../../src/text/messages.zig");
const template = @import("../../src/text/template.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const Effort = messages_file.Effort;
const LateSystem = messages_file.LateSystem;
const Levels = messages_file.Levels;
const Problem = messages_file.Problem;
const coerceEffort = messages_file.coerceEffort;
const effortLevels = messages_file.effortLevels;
const lateSystem = messages_file.lateSystem;
const prepare = messages_file.prepare;
const specials = messages_file.specials;
const ObjectMap = std.json.ObjectMap;
const context = messages_file.context;

const testing = std.testing;

fn expectSameJson(gpa: Allocator, expected: Value, actual: Value) !void {
    const want = try std.json.Stringify.valueAlloc(gpa, expected, .{});
    defer gpa.free(want);
    const got = try std.json.Stringify.valueAlloc(gpa, actual, .{});
    defer gpa.free(got);
    try testing.expectEqualStrings(want, got);
}

test "messages are normalized as TensorFold normalizes them, key order included" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(Value, gpa, @embedFile("fixtures/text/message_cases.json"), .{});
    defer parsed.deinit();
    const cases = parsed.value.object.get("messages").?.array.items;
    try testing.expect(cases.len >= 200);
    var refused: usize = 0;
    for (cases) |entry| {
        errdefer std.debug.print("case {s}, late_system {s}, allow_images {}\n", .{
            entry.object.get("name").?.string, entry.object.get("late_system").?.string, entry.object.get("allow_images").?.bool,
        });
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const late = std.meta.stringToEnum(LateSystem, entry.object.get("late_system").?.string).?;
        var problem: Problem = .{};
        const result = prepare(arena_state.allocator(), entry.object.get("messages").?, .{
            .late = late,
            .allow_images = entry.object.get("allow_images").?.bool,
        }, &problem);
        if (entry.object.get("error")) |message| {
            try testing.expectError(error.BadRequest, result);
            try testing.expectEqualStrings(message.string, problem.message);
            refused += 1;
        } else try expectSameJson(gpa, entry.object.get("expected").?, try result);
    }
    try testing.expect(refused >= 70);
}

test "effort levels are read from a template and efforts coerced as TensorFold coerces them" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(Value, gpa, @embedFile("fixtures/text/message_cases.json"), .{});
    defer parsed.deinit();
    for (parsed.value.object.get("levels").?.array.items) |entry| {
        errdefer std.debug.print("template {s}\n", .{entry.object.get("name").?.string});
        var want: Levels = .empty;
        for (entry.object.get("levels").?.array.items) |name| want.insert(std.meta.stringToEnum(Effort, name.string).?);
        try testing.expect(want.eql(effortLevels(entry.object.get("source").?.string)));
    }
    const efforts = parsed.value.object.get("efforts").?.array.items;
    try testing.expect(efforts.len >= 49);
    for (efforts) |entry| {
        var levels: Levels = .empty;
        for (entry.object.get("levels").?.array.items) |name| levels.insert(std.meta.stringToEnum(Effort, name.string).?);
        const effort = std.meta.stringToEnum(Effort, entry.object.get("effort").?.string).?;
        errdefer std.debug.print("effort {s}, {d} levels\n", .{ @tagName(effort), levels.count() });
        try testing.expectEqualStrings(entry.object.get("heard").?.string, @tagName(coerceEffort(effort, levels)));
    }
}

test "the late system role is what the template does with a later system message" {
    const gpa = testing.allocator;
    const keeps = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}";
    const drops = "{% for m in messages %}{% if m.role != 'system' or loop.first %}{{ m.content }}{% endif %}{% endfor %}";
    const raises = "{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('late') }}{% endif %}{% endfor %}";
    const names = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}";
    for ([_][]const u8{ keeps, drops, raises, names }, [_]LateSystem{ .system, .user, .user, .system }) |source, want| {
        var diag: template.Diagnostic = .{};
        var t = try template.Template.parse(gpa, source, &diag);
        defer t.deinit();
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const config = try std.json.parseFromSliceLeaky(Value, arena_state.allocator(),
            \\{"bos_token": null, "eos_token": {"content": "<e>", "lstrip": false}, "pad_token": "<p>", "model_max_length": 9}
        , .{});
        const special = try specials(arena_state.allocator(), config);
        try testing.expectEqual(@as(usize, 3), special.count());
        try testing.expectEqualStrings("<e>", special.get("eos_token").?.string);
        try testing.expectEqual(want, try lateSystem(gpa, &t, special));
    }
}

test "the template's variables: empty tools are none, effort only with thinking on" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const messages = try std.json.parseFromSliceLeaky(Value, arena, "[{\"role\": \"user\", \"content\": \"hi\"}]", .{});
    const none = try std.json.parseFromSliceLeaky(Value, arena, "[]", .{});
    var special: ObjectMap = .empty;
    try special.put(arena, "eos_token", .{ .string = "<e>" });

    const off = try context(arena, special, .{ .messages = messages, .tools = none, .enable_thinking = false, .reasoning_effort = .low });
    try expectSameJson(testing.allocator, try std.json.parseFromSliceLeaky(Value, arena,
        \\{"eos_token": "<e>", "messages": [{"role": "user", "content": "hi"}], "tools": null,
        \\ "add_generation_prompt": true, "enable_thinking": false}
    , .{}), off);

    const tools = try std.json.parseFromSliceLeaky(Value, arena, "[{\"type\": \"function\"}]", .{});
    const on = try context(arena, special, .{ .messages = messages, .tools = tools, .enable_thinking = true, .reasoning_effort = .xhigh });
    try testing.expectEqualStrings("xhigh", on.object.get("reasoning_effort").?.string);
    try testing.expect(on.object.get("tools").? == .array);
    const quiet = try context(arena, special, .{ .messages = messages, .enable_thinking = true, .reasoning_effort = .none });
    try testing.expect(quiet.object.get("reasoning_effort") == null);
}

test "tool-call arguments holding NaN and Infinity reach the template as Python prints them" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const messages = try std.json.parseFromSliceLeaky(Value, arena,
        \\[{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f",
        \\  "arguments": "{\"a\": NaN, \"b\": [Infinity, -Infinity], \"a\": 1e999}"}}]}]
    , .{});
    var problem: Problem = .{};
    const prepared = try prepare(arena, messages, .{ .late = .user }, &problem);
    var diag: template.Diagnostic = .{};
    var t = try template.Template.parse(gpa, "{{ messages[0].tool_calls[0].function.arguments | tojson }}", &diag);
    defer t.deinit();
    var variables: ObjectMap = .empty;
    try variables.put(arena, "messages", prepared);
    const text = try t.renderAlloc(gpa, .{ .object = variables }, &diag);
    defer gpa.free(text);
    // json.dumps(json.loads(arguments), ensure_ascii=False)
    try testing.expectEqualStrings("{\"a\": Infinity, \"b\": [Infinity, -Infinity]}", text);
}
