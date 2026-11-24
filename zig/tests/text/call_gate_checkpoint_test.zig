//! `call_gate.zig`'s tests against the checkpoint's template and tokenizer (ZSPARK_FLASH_NEXT_FP4_LOCAL): a required call
//! as the gate writes it is the next turn's prompt, and a call as the template writes it is tokens the gate
//! takes. The template's form on its own: `call_gate_walk_test.zig`.

const std = @import("std");
const testing = std.testing;
const call_gate = @import("../../src/text/call_gate.zig");
const grammar = @import("../../src/text/grammar.zig");
const json_schema = @import("../../src/text/json_schema.zig");
const tokenizer_file = @import("../../src/text/tokenizer.zig");
const Value = std.json.Value;
const Tokenizer = call_gate.Tokenizer;
const Tool = call_gate.Tool;
const compile = call_gate.compile;
const tx = @import("call_gate_walk_test.zig").tx;

test "the checkpoint's template: a required call after thinking, as the gate writes it, is the tokens the next turn's prompt has for it" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory; skipped without.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const frontend_file = @import("../../src/text/frontend.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try std.Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    var said: frontend_file.Diagnostic = .{};
    var frontend: frontend_file.Frontend = try .load(gpa, testing.io, dir, null, &said);
    defer frontend.deinit(gpa);
    const tokenizer = &frontend.tokenizer;
    const stops = [1]u32{tokenizer.addedId("<|im_end|>").?};
    const think_close = tokenizer.addedId("</think>").?;
    var engine: json_schema.Engine = try .init(gpa, tokenizer, tokenizer.count(), &stops);
    defer engine.deinit(gpa);
    var diagnostic: grammar.Diagnostic = .{};
    const mask = try gpa.alloc(u32, engine.interface().words);
    defer gpa.free(mask);

    const weather: Tool = .{ .name = "get_weather", .parameters = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"},\"units\":{\"type\":\"string\",\"enum\":[\"c\",\"f\"]}},\"required\":[\"city\",\"days\",\"units\"]}" };
    const clock: Tool = .{ .name = "get_time", .parameters = "{\"type\":\"object\",\"properties\":{\"timezone\":{\"type\":\"string\"},\"hours24\":{\"type\":\"boolean\"}},\"required\":[\"timezone\",\"hours24\"]}" };
    const offered = try std.fmt.allocPrint(arena, "[{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\",\"parameters\":{s}}}}},{{\"type\":\"function\",\"function\":{{\"name\":\"get_time\",\"parameters\":{s}}}}}]", .{ weather.parameters, clock.parameters });
    const tools = try std.json.parseFromSliceLeaky(Value, arena, offered, .{});
    const asked = "{\"role\":\"user\",\"content\":\"Weather in Lisbon for two days, and the time there?\"}";
    const one = "{\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Lisbon\\\",\\\"days\\\":2,\\\"units\\\":\\\"c\\\"}\"}}";
    const two = "{\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{\\\"timezone\\\":\\\"Europe/Lisbon\\\",\\\"hours24\\\":true}\"}}";
    const one_text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nLisbon\n</parameter>\n<parameter=days>\n2\n</parameter>\n<parameter=units>\nc\n</parameter>\n</function>\n</tool_call>";
    const two_text = "<tool_call>\n<function=get_time>\n<parameter=timezone>\nEurope/Lisbon\n</parameter>\n<parameter=hours24>\ntrue\n</parameter>\n</function>\n</tool_call>";
    const reasoning = "The user wants Lisbon's weather.\nI will call the tool.";

    // What the reply is when it thinks and when it does not; one call and two; any tool, and a named function
    // (which reaches the model as the one tool offered).
    const Case = struct { thinking: bool, calls: []const []const u8, texts: []const []const u8, offered: []const Tool, forced: usize };
    for ([_]Case{
        .{ .thinking = true, .calls = &.{one}, .texts = &.{one_text}, .offered = &.{ weather, clock }, .forced = 41 },
        .{ .thinking = true, .calls = &.{ one, two }, .texts = &.{ one_text, two_text }, .offered = &.{ weather, clock }, .forced = 73 },
        .{ .thinking = true, .calls = &.{one}, .texts = &.{one_text}, .offered = &.{weather}, .forced = 42 },
        .{ .thinking = false, .calls = &.{one}, .texts = &.{one_text}, .offered = &.{ weather, clock }, .forced = 40 },
        .{ .thinking = false, .calls = &.{ one, two }, .texts = &.{ one_text, two_text }, .offered = &.{ weather, clock }, .forced = 72 },
    }) |case| {
        errdefer std.debug.print("thinking {}, {d} calls, {d} tools\n", .{ case.thinking, case.calls.len, case.offered.len });
        const first = try std.json.parseFromSliceLeaky(Value, arena, "[" ++ asked ++ "]", .{});
        const prompt = try frontend.prompt(arena, .{ .messages = first, .tools = tools, .thinking = case.thinking }, &said);

        // The reply as it is decoded: the reasoning and its closing tag are the model's (here, the tokens of the
        // text a client is sent and sends back), and everything after it the gate's.
        var reply: std.ArrayList(u32) = .empty;
        if (case.thinking) {
            try tokenizer.encode(arena, reasoning ++ "\n", &reply);
            try reply.append(arena, think_close);
        }
        const gated = reply.items.len;
        const lead = if (case.thinking) frontend.call_lead else "";
        try tokenizer.encode(arena, try std.mem.concat(arena, u8, &.{ lead, try std.mem.join(arena, "\n", case.texts) }), &reply);
        const constraint = try compile(gpa, engine.interface(), tokenizer, &stops, case.offered, .{ .lead = lead }, &diagnostic);
        defer constraint.deinit();
        var at = try constraint.start();
        var forced: usize = 0;
        for (reply.items[gated..], gated..) |id, step| {
            errdefer std.debug.print("token {d}, '{s}'\n", .{ step, tokenizer.tokenBytes(id) });
            try constraint.fill(at, mask);
            var count: usize = 0;
            for (mask) |word| count += @popCount(word);
            forced += @intFromBool(count == 1);
            try testing.expect(grammar.allowed(mask, id));
            at = (try constraint.accept(at, id)).?;
        }
        try constraint.fill(at, mask);
        try testing.expect(grammar.allowed(mask, stops[0]));
        // the lead's tokens are among the forced: one more than the same call without thinking
        errdefer std.debug.print("{d} of {d} forced\n", .{ forced, reply.items.len - gated });
        try testing.expectEqual(case.forced, forced);
        try reply.append(arena, stops[0]);

        // The next turn: the same reply sent back with its reasoning and its calls, and the tools' results.
        var history: std.ArrayList(u8) = .empty;
        try history.print(arena, "[{s},{{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":{f},\"tool_calls\":[{s}]}}", .{
            asked, std.json.fmt(@as([]const u8, if (case.thinking) reasoning else ""), .{}), try std.mem.join(arena, ",", case.calls),
        });
        for (case.calls, 0..) |_, nth| try history.print(arena, ",{{\"role\":\"tool\",\"tool_call_id\":\"call_{c}\",\"content\":\"ok\"}}", .{@as(u8, 'a') + @as(u8, @intCast(nth))});
        try history.append(arena, ']');
        const second = try std.json.parseFromSliceLeaky(Value, arena, history.items, .{});
        const next = try frontend.prompt(arena, .{ .messages = second, .tools = tools, .thinking = case.thinking }, &said);
        // token for token: the first prompt, the reply, and only then what the request added
        const held = prompt.len + reply.items.len;
        try testing.expect(next.len > held);
        try testing.expectEqualSlices(u32, prompt, next[0..prompt.len]);
        try testing.expectEqualSlices(u32, reply.items, next[prompt.len..held]);

        // Without the lead the reply parts from the prompt right after the reasoning, as it did before spec 034.
        if (case.thinking) {
            const unled = try compile(gpa, engine.interface(), tokenizer, &stops, case.offered, .{}, &diagnostic);
            defer unled.deinit();
            try testing.expectEqual(null, try unled.accept(try unled.start(), next[prompt.len + gated]));
        }
    }
}

test "the checkpoint's tokenizer: a call as the template writes it is tokens the gate takes, most of them forced" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory; skipped without.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var dir = try std.Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    var said: tokenizer_file.Diagnostic = .{};
    var tokenizer = try Tokenizer.load(gpa, testing.io, dir, "tokenizer.json", &said);
    defer tokenizer.deinit(gpa);
    const stops = [1]u32{tokenizer.addedId("<|im_end|>").?};
    var engine: json_schema.Engine = try .init(gpa, &tokenizer, tokenizer.count(), &stops);
    defer engine.deinit(gpa);
    var diagnostic: grammar.Diagnostic = .{};
    const weather: Tool = .{ .name = "get_weather", .parameters = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"},\"units\":{\"type\":\"string\",\"enum\":[\"c\",\"f\"]},\"hours\":{\"type\":\"array\",\"items\":{\"type\":\"integer\"}}},\"required\":[\"city\",\"days\",\"units\"]}" };
    const clock: Tool = .{ .name = "get_time", .parameters = "{\"type\":\"object\",\"properties\":{\"timezone\":{\"type\":\"string\"},\"hours24\":{\"type\":\"boolean\"}},\"required\":[\"timezone\",\"hours24\"]}" };
    const constraint = try compile(gpa, engine.interface(), &tokenizer, &stops, &.{ weather, clock }, .{}, &diagnostic);
    defer constraint.deinit();
    for (tx.gateOf(constraint).tools.items) |tool| try testing.expect(tool.params != null);
    const mask = try gpa.alloc(u32, constraint.words);
    defer gpa.free(mask);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);

    // Each reply encoded whole, as the model's own tokens for that text fall. In two places the gate's tokens
    // are not the tokenizer's, which has one token where the gate has a frame's last and a value's first: an
    // empty string (two newlines) and a JSON value that ends with a bracket (`]` and a newline). That reply is
    // encoded in parts.
    const Case = struct { parts: []const []const u8, forced: usize };
    for ([_]Case{
        .{ .parts = &.{"<tool_call>\n<function=get_weather>\n<parameter=city>\nLisbon\n</parameter>\n<parameter=days>\n2\n</parameter>\n<parameter=units>\nc\n</parameter>\n</function>\n</tool_call>"}, .forced = 39 },
        .{ .parts = &.{"<tool_call>\n<function=get_time>\n<parameter=timezone>\nEurope/Lisbon.\n</parameter>\n<parameter=hours24>\ntrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nSão Paulo\nBrazil\n\n</parameter>\n<parameter=days>\n10\n</parameter>\n<parameter=units>\nf\n</parameter>\n</function>\n</tool_call>"}, .forced = 71 },
        .{ .parts = &.{ "<tool_call>\n<function=get_weather>\n<parameter=city>\n", "\n</parameter>\n<parameter=days>\n-1\n</parameter>\n<parameter=units>\nf\n</parameter>\n<parameter=hours>\n[6, 18]", "\n</parameter>\n</function>\n</tool_call>" }, .forced = 50 },
    }) |case| {
        ids.clearRetainingCapacity();
        for (case.parts) |part| try tokenizer.encode(gpa, part, &ids);
        var at = try constraint.start();
        var forced: usize = 0;
        for (ids.items, 0..) |id, step| {
            try constraint.fill(at, mask);
            var count: usize = 0;
            for (mask) |word| count += @popCount(word);
            forced += @intFromBool(count == 1);
            errdefer std.debug.print("token {d} of {d}, '{s}', in {s}\n", .{ step, ids.items.len, tokenizer.tokenBytes(id), case.parts[0] });
            try testing.expect(grammar.allowed(mask, id));
            at = (try constraint.accept(at, id)).?;
        }
        try constraint.fill(at, mask);
        try testing.expect(grammar.allowed(mask, stops[0]));
        errdefer std.debug.print("{d} of {d} tokens forced in {s}\n", .{ forced, ids.items.len, case.parts[0] });
        try testing.expectEqual(case.forced, forced);
    }
}
