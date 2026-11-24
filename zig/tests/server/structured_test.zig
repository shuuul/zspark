//! `structured.zig`'s tests.

const std = @import("std");
const call_gate = @import("../../src/text/call_gate.zig");
const grammar = @import("../../src/text/grammar.zig");
const json_schema = @import("../../src/text/json_schema.zig");
const request_file = @import("../../src/server/request.zig");
const structured = @import("../../src/server/structured.zig");
const text_fixtures = @import("../text/testing.zig");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const testing = std.testing;
const Bench = @import("structured_fixture.zig").Bench;
const Compiler = structured.Compiler;

const fx = struct {
    fn request(arena: Allocator, tools: []const u8, structure: request_file.Structure) !request_file.Request {
        return .{
            .tools = try std.json.parseFromSliceLeaky(Value, arena, tools, .{}),
            .structure = structure,
        };
    }
};

test "free text compiles nothing; a document is its schema's constraint; a refusal names the field and the keyword" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var bench: Bench = undefined;
    try bench.init(gpa);
    defer bench.deinit(gpa);
    const compiler = bench.compiler();
    var diagnostic: request_file.Diagnostic = .{};

    const before = json_schema.calls;
    try testing.expectEqual(null, try compiler.compile(gpa, arena, .{}, &diagnostic));
    try testing.expectEqual(before, json_schema.calls);

    const mask = try gpa.alloc(u32, compiler.engine.words);
    defer gpa.free(mask);
    for ([_]struct { request_file.Structure, u8 }{
        .{ .{ .document = .{ .spec = .json_object } }, '{' },
        .{ .{ .document = .{ .spec = .{ .json_schema = "{\"type\":\"array\",\"items\":{\"type\":\"boolean\"}}" } } }, '[' },
    }) |case| {
        const constraint = (try compiler.compile(gpa, arena, .{ .structure = case[0] }, &diagnostic)).?;
        defer constraint.deinit();
        try constraint.fill(try constraint.start(), mask);
        // every token the document may begin with begins with its bracket
        var count: usize = 0;
        for (0..bench.tokenizer.count()) |token| {
            if (!grammar.allowed(mask, @intCast(token))) continue;
            try testing.expectEqual(case[1], bench.tokenizer.tokenBytes(@intCast(token))[0]);
            count += 1;
        }
        try testing.expect(count > 0);
    }

    for ([_]struct { request_file.Structure, []const u8 }{
        .{ .{ .document = .{ .spec = .{ .json_schema = "{\"type\":\"string\",\"pattern\":\"a\"}" } } }, "response_format: the schema cannot be enforced: #: `pattern` is not read" },
        .{ .{ .document = .{ .spec = .{ .json_schema = "{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\",\"minimum\":0}}}" }, .field = "text.format" } }, "text.format: the schema cannot be enforced: #/n: `minimum` is not read" },
        .{ .{ .document = .{ .spec = .{ .json_schema = "{\"type\":\"teapot\"}" }, .field = "output_config.format" } }, "output_config.format: the schema cannot be enforced: #: `teapot` is not a type" },
    }) |case| {
        errdefer std.debug.print("said: {s}\n", .{diagnostic.message()});
        try testing.expectError(error.BadRequest, compiler.compile(gpa, arena, .{ .structure = case[0] }, &diagnostic));
        try testing.expectEqualStrings(case[1], diagnostic.message());
    }
}

test "a document after a lead: the lead's tokens forced one by one, then the document's masks; an unwritable lead left out" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var bench: Bench = undefined;
    try bench.init(gpa);
    defer bench.deinit(gpa);
    var diagnostic: request_file.Diagnostic = .{};
    const words = bench.engine.interface().words;
    const mask = try gpa.alloc(u32, words);
    defer gpa.free(mask);
    const expected = try gpa.alloc(u32, words);
    defer gpa.free(expected);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);

    const schema = "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"]}";
    const document = "{\"ok\":true}";
    for ([_]request_file.Structure{
        .{ .document = .{ .spec = .{ .json_schema = schema } } },
        .{ .document = .{ .spec = .json_object } },
    }) |structure| {
        const request: request_file.Request = .{ .structure = structure };
        const plain = (try bench.compiler().compile(gpa, arena, request, &diagnostic)).?;
        defer plain.deinit();
        for ([_][]const u8{ "\n\n", "\n", "\n\n\n" }) |lead| {
            var compiler = bench.compiler();
            compiler.lead = lead;
            const led = (try compiler.compile(gpa, arena, request, &diagnostic)).?;
            defer led.deinit();
            ids.clearRetainingCapacity();
            try bench.tokenizer.encode(gpa, lead, &ids);
            const leading = ids.items.len;
            try bench.tokenizer.encode(gpa, document, &ids);
            // two replies, one after the other: the constraint serves the next as it served the first
            for (0..2) |_| {
                var at = try led.start();
                var beside = try plain.start();
                var last_of_lead = at;
                for (ids.items, 0..) |id, step| {
                    if (step + 1 == leading) last_of_lead = at;
                    try led.fill(at, mask);
                    if (step < leading) {
                        // the lead: one token a mask, which the decoder drafts; no stop, and nothing of the document
                        var count: usize = 0;
                        for (mask) |word| count += @popCount(word);
                        try testing.expectEqual(1, count);
                        try testing.expect(grammar.allowed(mask, id));
                        try testing.expect(!led.ended(at));
                        try testing.expectEqual(null, try led.accept(at, ids.items[leading]));
                        try testing.expectEqual(null, try led.accept(at, bench.stops[0]));
                    } else {
                        // the document: the masks of the constraint without a lead, bit for bit
                        try plain.fill(beside, expected);
                        try testing.expectEqualSlices(u32, expected, mask);
                        beside = (try plain.accept(beside, id)).?;
                    }
                    const next = (try led.accept(at, id)).?;
                    // a position is a value: the same token from the same place again, as a draft walked twice
                    // is, gives a place whose mask is the same and leaves the first one good
                    const again = (try led.accept(at, id)).?;
                    try led.fill(next, mask);
                    try led.fill(again, expected);
                    try testing.expectEqualSlices(u32, mask, expected);
                    at = next;
                }
                // the lead's last token taken again once the document is under way, as a window's drafts walked
                // from an older row are: the places inside the document stay good
                _ = (try led.accept(last_of_lead, ids.items[leading - 1])).?;
                try led.fill(at, mask);
                try testing.expect(grammar.allowed(mask, bench.stops[0]));
                const end = (try led.accept(at, bench.stops[0])).?;
                try testing.expect(led.ended(end) and !led.ended(at));
            }
            // without the lead the document's first token is not taken, and the plain constraint takes no lead
            try testing.expectEqual(null, try led.accept(try led.start(), ids.items[leading]));
            try testing.expectEqual(null, try plain.accept(try plain.start(), ids.items[0]));
        }
        // A lead the tokenizer cannot write back (bytes that are not UTF-8, a token that ends the reply), or
        // none: the reply opens with the document.
        ids.clearRetainingCapacity();
        try bench.tokenizer.encode(gpa, document, &ids);
        for ([_][]const u8{ "\xff\n", "\n<|end|>\n", "" }) |lead| {
            var compiler = bench.compiler();
            compiler.lead = lead;
            const constraint = (try compiler.compile(gpa, arena, request, &diagnostic)).?;
            defer constraint.deinit();
            var at = try constraint.start();
            for (ids.items) |id| at = (try constraint.accept(at, id)).?;
        }
    }
    // a schema that is refused is refused with the lead too, and nothing is left held
    var compiler = bench.compiler();
    compiler.lead = "\n\n";
    try testing.expectError(error.BadRequest, compiler.compile(gpa, arena, .{ .structure = .{ .document = .{ .spec = .{ .json_schema = "{\"type\":\"teapot\"}" } } } }, &diagnostic));
}

test "the checkpoint's template: a document after thinking, as it is decoded, is the tokens the next turn's prompt has for it" {
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
    var diagnostic: request_file.Diagnostic = .{};
    const mask = try gpa.alloc(u32, engine.interface().words);
    defer gpa.free(mask);

    const schema = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"},\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"city\",\"days\",\"tags\"]}";
    const asked = "{\"role\":\"user\",\"content\":\"Plan two days in Lisbon, as JSON.\"}";
    const tools = try std.json.parseFromSliceLeaky(Value, arena, "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}]", .{});
    const reasoning = "The user wants a plan.\nI will answer with the document.";
    // What the reply is when it thinks and when it does not; a schema and any object; a document written
    // compactly and one with blanks inside it; with tools offered beside it.
    const Case = struct { thinking: bool, spec: grammar.Spec, document: []const u8, tools: ?Value = null };
    for ([_]Case{
        .{ .thinking = true, .spec = .{ .json_schema = schema }, .document = "{\"city\":\"Lisbon\",\"days\":2,\"tags\":[\"tram\",\"fado\"]}" },
        .{ .thinking = true, .spec = .{ .json_schema = schema }, .document = "{\n  \"city\": \"Lisbon\",\n  \"days\": 2,\n  \"tags\": []\n}" },
        .{ .thinking = true, .spec = .json_object, .document = "{\"plan\": [\"Alfama\", \"Belém\"]}" },
        .{ .thinking = true, .spec = .{ .json_schema = schema }, .document = "{\"city\":\"Lisbon\",\"days\":2,\"tags\":[]}", .tools = tools },
        .{ .thinking = false, .spec = .{ .json_schema = schema }, .document = "{\"city\":\"Lisbon\",\"days\":2,\"tags\":[\"tram\"]}" },
        .{ .thinking = false, .spec = .json_object, .document = "{\"plan\": []}", .tools = tools },
    }) |case| {
        errdefer std.debug.print("thinking {}, document {s}\n", .{ case.thinking, case.document });
        const first = try std.json.parseFromSliceLeaky(Value, arena, "[" ++ asked ++ "]", .{});
        const prompt = try frontend.prompt(arena, .{ .messages = first, .tools = case.tools, .thinking = case.thinking }, &said);

        // The reply as it is decoded: the reasoning and its closing tag are the model's (here, the tokens of the
        // text a client is sent and sends back), then the lead, the document and the end.
        var decoded: std.ArrayList(u32) = .empty;
        if (case.thinking) {
            try tokenizer.encode(arena, reasoning ++ "\n", &decoded);
            try decoded.append(arena, think_close);
        }
        const gated = decoded.items.len;
        const lead = if (case.thinking) frontend.content_lead else "";
        try tokenizer.encode(arena, lead, &decoded);
        const led = decoded.items.len;
        try tokenizer.encode(arena, case.document, &decoded);
        var compiler: Compiler = .{ .engine = engine.interface(), .tokenizer = tokenizer, .stops = &stops, .lead = lead };
        const request: request_file.Request = .{ .tools = case.tools, .structure = .{ .document = .{ .spec = case.spec } } };
        const constraint = (try compiler.compile(gpa, arena, request, &diagnostic)).?;
        defer constraint.deinit();
        var at = try constraint.start();
        for (decoded.items[gated..], gated..) |id, step| {
            errdefer std.debug.print("token {d}, '{s}'\n", .{ step, tokenizer.tokenBytes(id) });
            try constraint.fill(at, mask);
            try testing.expect(grammar.allowed(mask, id));
            if (step < led) {
                var count: usize = 0;
                for (mask) |word| count += @popCount(word);
                try testing.expectEqual(1, count);
            }
            at = (try constraint.accept(at, id)).?;
        }
        // the lead is the template's blank line, one token on this tokenizer
        try testing.expectEqual(@as(usize, @intFromBool(case.thinking)), led - gated);
        // the document ends the reply: the end token is allowed, and no blank after the document is
        try constraint.fill(at, mask);
        try testing.expect(grammar.allowed(mask, stops[0]));
        for ([_][]const u8{ "\n", " ", "\n\n" }) |blank| {
            var blanks: std.ArrayList(u32) = .empty;
            try tokenizer.encode(arena, blank, &blanks);
            try testing.expect(!grammar.allowed(mask, blanks.items[0]));
        }
        try decoded.append(arena, stops[0]);

        // The next turn: the same reply sent back with its reasoning, and another question.
        const history = try std.fmt.allocPrint(arena, "[{s},{{\"role\":\"assistant\",\"content\":{f},\"reasoning_content\":{f}}},{{\"role\":\"user\",\"content\":\"And Porto?\"}}]", .{
            asked, std.json.fmt(case.document, .{}), std.json.fmt(@as([]const u8, if (case.thinking) reasoning else ""), .{}),
        });
        const second = try std.json.parseFromSliceLeaky(Value, arena, history, .{});
        const next = try frontend.prompt(arena, .{ .messages = second, .tools = case.tools, .thinking = case.thinking }, &said);
        // token for token: the first prompt, the reply to its end token, and only then what the request added
        const held = prompt.len + decoded.items.len;
        try testing.expect(next.len > held);
        try testing.expectEqualSlices(u32, prompt, next[0..prompt.len]);
        try testing.expectEqualSlices(u32, decoded.items, next[prompt.len..held]);

        // Without the lead the reply parts from the prompt right after the reasoning, as it did before spec 035.
        if (case.thinking) {
            compiler.lead = "";
            const unled = (try compiler.compile(gpa, arena, request, &diagnostic)).?;
            defer unled.deinit();
            try testing.expectEqual(null, try unled.accept(try unled.start(), next[prompt.len + gated]));
        }
    }
}

test "a required call's tools: each one's parameters as an object's schema, or a refusal that names the tool" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var bench: Bench = undefined;
    try bench.init(gpa);
    defer bench.deinit(gpa);
    const compiler = bench.compiler();
    var diagnostic: request_file.Diagnostic = .{};
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);

    // In the JSON form, which holds every schema the engine takes: which arguments a tool's call takes, each
    // written whole after the call's opening.
    var as_json = compiler;
    as_json.gate.template_form = false;
    const Case = struct { tool: []const u8, taken: []const []const u8, refused: []const []const u8 };
    for ([_]Case{
        // the chat API's shape, the Messages API's, and a bare one
        .{ .tool = "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}},\"required\":[\"n\"]}}}", .taken = &.{"{\"n\":3}"}, .refused = &.{ "{}", "{\"n\":true}", "[3]" } },
        .{ .tool = "{\"name\":\"f\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}}}}", .taken = &.{ "{}", "{\"ok\":true}" }, .refused = &.{"{\"ok\":1}"} },
        // no parameters: no arguments
        .{ .tool = "{\"type\":\"function\",\"function\":{\"name\":\"f\"}}", .taken = &.{"{}"}, .refused = &.{ "{\"a\":1}", "1" } },
        .{ .tool = "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":null}}", .taken = &.{"{}"}, .refused = &.{"{\"a\":1}"} },
        // a schema that does not say its type is an object's
        .{ .tool = "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{}}}", .taken = &.{ "{}", "{\"a\":1}" }, .refused = &.{ "1", "[]", "\"a\"" } },
        .{ .tool = "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"null\"}},\"required\":[\"n\"]}}}", .taken = &.{"{\"n\":null}"}, .refused = &.{ "null", "{}" } },
        // `strict` and the annotations are not read
        .{ .tool = "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"strict\":false,\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"title\":\"T\",\"properties\":{\"n\":{\"const\":1,\"description\":\"one\"}},\"required\":[\"n\"],\"additionalProperties\":false}}}", .taken = &.{"{\"n\":1}"}, .refused = &.{ "{\"n\":2}", "{\"n\":1,\"m\":1}" } },
    }) |case| {
        const tools = try std.fmt.allocPrint(arena, "[{s}]", .{case.tool});
        const constraint = (try as_json.compile(gpa, arena, try fx.request(arena, tools, .{ .call = .{} }), &diagnostic)).?;
        defer constraint.deinit();
        for ([_][]const []const u8{ case.taken, case.refused }, [_]bool{ true, false }) |documents, expected| {
            for (documents) |document| {
                var at = try constraint.start();
                ids.clearRetainingCapacity();
                try bench.tokenizer.encode(gpa, "<tool_call>\n{\"name\": \"f\", \"arguments\":", &ids);
                for (ids.items) |id| at = (try constraint.accept(at, id)).?;
                ids.clearRetainingCapacity();
                try bench.tokenizer.encode(gpa, document, &ids);
                try bench.tokenizer.encode(gpa, "}\n</tool_call>", &ids);
                const taken = for (ids.items) |id| {
                    at = try constraint.accept(at, id) orelse break false;
                } else true;
                errdefer std.debug.print("tool {s}, arguments {s}\n", .{ case.tool, document });
                try testing.expectEqual(expected, taken);
            }
        }
    }

    // Two tools: a call's arguments are held to the schema of the tool it names, whichever comes first.
    const two = "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}},\"required\":[\"n\"],\"additionalProperties\":false}}}," ++
        "{\"type\":\"function\",\"function\":{\"name\":\"g\",\"parameters\":{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"],\"additionalProperties\":false}}}]";
    {
        const tools = two;
        const constraint = (try as_json.compile(gpa, arena, try fx.request(arena, tools, .{ .call = .{} }), &diagnostic)).?;
        defer constraint.deinit();
        for ([_]struct { []const u8, []const u8, bool }{
            .{ "f", "{\"n\":3}", true },
            .{ "f", "{\"ok\":true}", false },
            .{ "g", "{\"ok\":true}", true },
            .{ "g", "{\"n\":3}", false },
        }) |case| {
            var at = try constraint.start();
            ids.clearRetainingCapacity();
            try bench.tokenizer.encode(gpa, try std.fmt.allocPrint(arena, "<tool_call>\n{{\"name\": \"{s}\", \"arguments\":", .{case[0]}), &ids);
            for (ids.items) |id| at = (try constraint.accept(at, id)).?;
            ids.clearRetainingCapacity();
            try bench.tokenizer.encode(gpa, case[1], &ids);
            try bench.tokenizer.encode(gpa, "}\n</tool_call>", &ids);
            const taken = for (ids.items) |id| {
                at = try constraint.accept(at, id) orelse break false;
            } else true;
            errdefer std.debug.print("tool {s}, arguments {s}\n", .{ case[0], case[1] });
            try testing.expectEqual(case[2], taken);
        }
    }

    // As the server compiles it: in the template's form, each parameter held to its own schema, and what a
    // reply's parser reads back (`text/testing.zig`'s `gate.read`, written apart from the gate) is the call's
    // name and its arguments as JSON.
    {
        const constraint = (try compiler.compile(gpa, arena, try fx.request(arena, two, .{ .call = .{} }), &diagnostic)).?;
        defer constraint.deinit();
        const offered = [_]call_gate.Tool{
            .{ .name = "f", .parameters = "{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}}}" },
            .{ .name = "g", .parameters = "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}}}" },
        };
        for ([_]struct { []const u8, ?[]const u8 }{
            .{ "<tool_call>\n<function=f>\n<parameter=n>\n-3\n</parameter>\n</function>\n</tool_call>", "{\"n\":-3}" },
            .{ "<tool_call>\n<function=g>\n<parameter=ok>\ntrue\n</parameter>\n</function>\n</tool_call>", "{\"ok\":true}" },
            .{ "<tool_call>\n<function=f>\n<parameter=n>\ntrue", null },
            .{ "<tool_call>\n<function=f>\n<parameter=ok>", null },
            .{ "<tool_call>\n<function=g>\n</function>", null },
            // the JSON form is not taken where the template's is written
            .{ "<tool_call>\n{", null },
        }) |case| {
            var at = try constraint.start();
            ids.clearRetainingCapacity();
            try bench.tokenizer.encode(gpa, case[0], &ids);
            const taken = for (ids.items) |id| {
                at = try constraint.accept(at, id) orelse break false;
            } else true;
            errdefer std.debug.print("reply {s}\n", .{case[0]});
            try testing.expectEqual(case[1] != null, taken);
            const arguments = case[1] orelse continue;
            try testing.expect(try constraint.accept(at, bench.stops[0]) != null);
            const calls = try text_fixtures.gate.read(arena, case[0], &offered);
            try testing.expectEqual(1, calls.len);
            try testing.expectEqualStrings(case[0]["<tool_call>\n<function=".len..][0..1], calls[0].name);
            try testing.expectEqualStrings(arguments, calls[0].arguments);
        }
    }

    for ([_]struct { []const u8, []const u8 }{
        .{ "[{\"type\":\"function\",\"function\":{\"name\":\"find\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\",\"pattern\":\"^a\"}}}}}]", "tool 'find': its parameters: #/q: `pattern` is not read" },
        .{ "[{\"type\":\"function\",\"function\":{\"name\":\"ok\"}},{\"type\":\"function\",\"function\":{\"name\":\"find\",\"parameters\":{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\",\"maximum\":9}}}}}]", "tool 'find': its parameters: #/n: `maximum` is not read" },
        .{ "[{\"type\":\"function\",\"function\":{\"name\":\"find\",\"parameters\":{\"type\":\"array\"}}}]", "tool 'find': its parameters must be an object's schema (\"type\": \"object\")" },
        .{ "[{\"type\":\"function\",\"function\":{\"name\":\"find\",\"parameters\":{\"type\":[\"object\",\"null\"]}}}]", "tool 'find': its parameters must be an object's schema (\"type\": \"object\")" },
        .{ "[{\"type\":\"function\",\"function\":{\"name\":\"find\",\"parameters\":{\"anyOf\":[{\"type\":\"object\"},{\"type\":\"null\"}]}}}]", "tool 'find': its parameters must be an object's schema (\"type\": \"object\")" },
        .{ "[{\"type\":\"function\",\"function\":{\"name\":\"find\",\"parameters\":{\"$ref\":\"#/$defs/a\",\"$defs\":{\"a\":{\"type\":\"object\"}}}}}]", "tool 'find': its parameters must be an object's schema (\"type\": \"object\")" },
        .{ "[{\"type\":\"function\",\"function\":{\"name\":\"find\",\"parameters\":\"object\"}}]", "tool 'find': its parameters must be a JSON schema object" },
        .{ "[{\"type\":\"function\",\"function\":{\"name\":\"find\"}},{\"type\":\"function\",\"function\":{\"name\":\"Find\"}}]", "two tools are named 'Find': a required call could not say which it calls" },
        .{ "[]", "a required tool call needs a tool" },
    }) |case| {
        errdefer std.debug.print("said: {s}\n", .{diagnostic.message()});
        try testing.expectError(error.BadRequest, compiler.compile(gpa, arena, try fx.request(arena, case[0], .{ .call = .{} }), &diagnostic));
        const expected = try std.fmt.allocPrint(arena, "tool_choice: a required tool call cannot be enforced: {s}", .{case[1]});
        try testing.expectEqualStrings(expected, diagnostic.message());
    }
}
