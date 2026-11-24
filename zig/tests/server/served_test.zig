//! `served.zig`'s tests on the host: a model that decodes nothing behind TensorFold's server, asked over a
//! socket as a client asks it. What a model's replies are is `models/qwen4/served_test.zig`'s (the GPU).
//! The model and the server on the loopback are `served_fixture.zig`'s.

const std = @import("std");
const api = @import("engine_api");
const host = @import("server");
const masks_file = @import("../../src/decode/masks.zig");
const request_file = @import("../../src/server/request.zig");
const sched = @import("../../src/server/scheduler.zig");
const served_file = @import("../../src/server/served.zig");
const structured_fixture = @import("structured_fixture.zig");
const text_frontend = @import("../../src/text/frontend.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixture = @import("served_fixture.zig");
const Hi = fixture.Hi;
const Counting = fixture.Counting;
const Stand = fixture.Stand;
const sent = fixture.sent;
const expectRefused = fixture.expectRefused;
const jsonOf = fixture.jsonOf;
const bodyOf = fixture.bodyOf;
const user = fixture.user;
const Kept = fixture.Kept;

const unread = "{\"type\":\"string\",\"pattern\":\"a\"}";
const unread_said = "response_format: the schema cannot be enforced: #: `pattern` is not read";

test "a constraint the engine cannot enforce is a 400 before anything is sent or decoded: a schema's unread keyword, a regex, a choice and a grammar, on the three APIs, streamed and not" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.init(gpa);
    defer stand.deinit();

    const by = "{s}: structured output by {s} is not enforced by this engine without xgrammar (libtfgrammar): send a JSON schema";
    const chat = [_]struct { []const u8, []const u8 }{
        .{ "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"n\",\"schema\":" ++ unread ++ "}}", unread_said },
        .{ "\"guided_json\":" ++ unread, "guided_json" ++ unread_said["response_format".len..] },
        .{ "\"guided_regex\":\"a+\"", std.fmt.comptimePrint(by, .{ "guided_regex", "regex" }) },
        .{ "\"guided_choice\":[\"a\",\"b\"]", std.fmt.comptimePrint(by, .{ "guided_choice", "choice" }) },
        .{ "\"guided_grammar\":\"root ::= \\\"a\\\"\"", std.fmt.comptimePrint(by, .{ "guided_grammar", "grammar" }) },
        .{ "\"structured_outputs\":{\"regex\":\"a+\"}", std.fmt.comptimePrint(by, .{ "structured_outputs", "regex" }) },
    };
    for (chat) |case| for ([_][]const u8{ "", ",\"stream\":true" }) |stream| {
        const body = try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",{s},{s}{s}}}", .{ user, case[0], stream });
        const expected = try std.fmt.allocPrint(arena, "{{\"error\": {{\"message\": {f}, \"type\": \"invalid_request_error\"}}}}", .{std.json.fmt(case[1], .{})});
        try expectRefused(try stand.post(arena, "/v1/chat/completions", body), expected);
    };
    // Responses answers with the chat API's error body, Messages with its own
    for ([_][]const u8{ "", ",\"stream\":true" }) |stream| {
        const responses = try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",\"input\":\"x\",\"text\":{{\"format\":{{\"type\":\"json_schema\",\"name\":\"n\",\"schema\":{s}}}}}{s}}}", .{ unread, stream });
        try expectRefused(try stand.post(arena, "/v1/responses", responses), "{\"error\": {\"message\": \"text.format" ++ unread_said["response_format".len..] ++ "\", \"type\": \"invalid_request_error\"}}");
        const messages = try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",\"max_tokens\":8,{s},\"output_config\":{{\"format\":{{\"type\":\"json_schema\",\"schema\":{s}}}}}{s}}}", .{ user, unread, stream });
        try expectRefused(try stand.post(arena, "/v1/messages", messages), "{\"type\": \"error\", \"error\": {\"type\": \"invalid_request_error\", \"message\": \"output_config.format" ++ unread_said["response_format".len..] ++ "\"}}");
    }
    try testing.expectEqual(0, stand.served.behind.replies.load(.acquire));
    // the three kinds are refused before the grammar engine is asked; each schema was compiled once, and failed
    try testing.expectEqual(8, stand.counting.compiles.load(.acquire));
}

test "a constraint the engine enforces is served as before, compiled once a request: its reply is decoded under what the refusal compiled, streamed and not" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.init(gpa);
    defer stand.deinit();

    const format = "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"n\",\"schema\":{\"type\":\"string\"}}}";
    const whole = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ "," ++ format ++ "}");
    errdefer std.debug.print("the reply:\n{s}\n", .{whole});
    try testing.expect(std.mem.startsWith(u8, whole, "HTTP/1.1 200 OK\r\n"));
    try testing.expect(std.mem.indexOf(u8, whole, "\"content\": \"hi\"") != null);
    const streamed = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ "," ++ format ++ ",\"stream\":true}");
    errdefer std.debug.print("the stream:\n{s}\n", .{streamed});
    try testing.expect(std.mem.startsWith(u8, streamed, "HTTP/1.1 200 OK\r\n"));
    try testing.expect(std.mem.indexOf(u8, streamed, "Content-Type: text/event-stream\r\n") != null);
    try testing.expect(std.mem.endsWith(u8, streamed, "data: [DONE]\n\n"));
    // free text beside them: nothing is compiled for it
    const free = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ "}");
    try testing.expect(std.mem.startsWith(u8, free, "HTTP/1.1 200 OK\r\n"));
    try testing.expectEqual(3, stand.served.behind.replies.load(.acquire));
    try testing.expectEqual(2, stand.served.behind.constrained.load(.acquire));
    try testing.expectEqual(2, stand.counting.compiles.load(.acquire));
}

test "the engine's refusal is its constraint's: null for free text and for what it enforces, the words of a required call's tool it cannot hold; what compiled waits for its reply, the oldest gives way, and a request asked twice compiles once" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var bench: structured_fixture.Bench = undefined;
    try bench.init(gpa);
    defer bench.deinit(gpa);
    var frontend: text_frontend.Frontend = undefined;
    var device_bytes: usize = 0;
    var counting: Counting = undefined;
    var served: served_file.Served(Hi) = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = counting.compiler(&bench), .behind = .{}, .capacity = 64, .vocab = 256, .lanes = 1, .think = null, .device_bytes = &device_bytes };
    defer served.close();
    const engine = served.engine();
    var buf: [api.refusal_bytes]u8 = undefined;
    const prompt = [_]u32{ 1, 2, 3 };

    try testing.expectEqual(null, engine.refusal(&.{ .prompt = &prompt, .max_tokens = 4 }, &buf));
    try testing.expectEqual(0, counting.compiles.load(.acquire));
    try testing.expectEqual(null, engine.refusal(&.{ .prompt = &prompt, .max_tokens = 4, .structure = .{ .kind = .json } }, &buf));
    // the host prepares a request again before its reply (a compaction): what waits answers
    for (0..3) |_| try testing.expectEqual(null, engine.refusal(&.{ .prompt = &prompt, .max_tokens = 4, .structure = .{ .kind = .json_schema, .text = "{\"type\":\"string\"}" } }, &buf));
    try testing.expectEqual(2, counting.compiles.load(.acquire));
    // requests whose replies never come (their clients left): no more wait than there are places, the rest are freed
    for (0..12) |length| {
        const schema = try std.fmt.allocPrint(arena, "{{\"type\":\"object\",\"properties\":{{\"k{d}\":{{\"type\":\"string\"}}}}}}", .{length});
        try testing.expectEqual(null, engine.refusal(&.{ .prompt = &prompt, .max_tokens = 4, .structure = .{ .kind = .json_schema, .text = schema } }, &buf));
    }
    try testing.expectEqual(14, counting.compiles.load(.acquire));
    var waiting: usize = 0;
    for (served.ready) |slot| waiting += @intFromBool(slot != null);
    try testing.expectEqual(served.ready.len, waiting);
    // the newest are the ones kept: the first schema gave way and is compiled again
    try testing.expectEqual(null, engine.refusal(&.{ .prompt = &prompt, .max_tokens = 4, .structure = .{ .kind = .json_schema, .text = "{\"type\":\"object\",\"properties\":{\"k11\":{\"type\":\"string\"}}}" } }, &buf));
    try testing.expectEqual(14, counting.compiles.load(.acquire));
    try testing.expectEqual(null, engine.refusal(&.{ .prompt = &prompt, .max_tokens = 4, .structure = .{ .kind = .json_schema, .text = "{\"type\":\"string\"}" } }, &buf));
    try testing.expectEqual(15, counting.compiles.load(.acquire));
    try testing.expectEqualStrings(unread_said, (engine.refusal(&.{ .prompt = &prompt, .max_tokens = 4, .structure = .{ .kind = .json_schema, .text = unread } }, &buf)).?);
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"a\":" ++ unread ++ "}}}}]";
    const said = (engine.refusal(&.{ .prompt = &prompt, .max_tokens = 4, .tools_json = tools, .call = .{ .opener = 0 } }, &buf)).?;
    errdefer std.debug.print("said: {s}\n", .{said});
    try testing.expect(std.mem.startsWith(u8, said, "tool_choice: a required tool call cannot be enforced: "));
    try testing.expect(std.mem.indexOf(u8, said, "`pattern` is not read") != null);
    // a server started without a grammar engine holds no reply to anything
    served.compiler = null;
    try testing.expectEqualStrings("structured output is not available: this server was started without a grammar engine", (engine.refusal(&.{ .prompt = &prompt, .max_tokens = 4, .structure = .{ .kind = .json } }, &buf)).?);
}

test "a prompt's token outside the vocabulary is refused before anything else: free text too, the first such id named, and nothing compiled for it" {
    const gpa = testing.allocator;
    var bench: structured_fixture.Bench = undefined;
    try bench.init(gpa);
    defer bench.deinit(gpa);
    var frontend: text_frontend.Frontend = undefined;
    var device_bytes: usize = 0;
    var counting: Counting = undefined;
    var served: served_file.Served(Hi) = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = counting.compiler(&bench), .behind = .{}, .capacity = 64, .vocab = 256, .lanes = 1, .think = null, .device_bytes = &device_bytes };
    defer served.close();
    const engine = served.engine();
    var buf: [api.refusal_bytes]u8 = undefined;

    // the last id of the vocabulary is the model's
    try testing.expectEqual(null, engine.refusal(&.{ .prompt = &.{ 0, 255, 1 }, .max_tokens = 4 }, &buf));
    try testing.expectEqualStrings("prompt token 2 is 256, out of the model's vocabulary; valid range is [0, 256)", (engine.refusal(&.{ .prompt = &.{ 1, 2, 256 }, .max_tokens = 4 }, &buf)).?);
    // an id that is negative as the model reads it, and the first of two
    try testing.expectEqualStrings("prompt token 0 is 4294967295, out of the model's vocabulary; valid range is [0, 256)", (engine.refusal(&.{ .prompt = &.{ std.math.maxInt(u32), 1, 300 }, .max_tokens = 4 }, &buf)).?);
    try testing.expectEqualStrings("prompt token 1 is 2147483648, out of the model's vocabulary; valid range is [0, 256)", (engine.refusal(&.{ .prompt = &.{ 1, 1 << 31 }, .max_tokens = 4 }, &buf)).?);
    // before its constraint is read: nothing is compiled, nothing waits for a reply
    try testing.expect(engine.refusal(&.{ .prompt = &.{ 1, 256 }, .max_tokens = 4, .structure = .{ .kind = .json_schema, .text = "{\"type\":\"string\"}" } }, &buf) != null);
    try testing.expectEqual(0, counting.compiles.load(.acquire));
    for (served.ready) |slot| try testing.expectEqual(null, slot);
}

test "the zspark backend refuses multiple choices and cache salts before decoding" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.initWith(gpa, .{});
    defer stand.deinit();

    for ([_]struct { []const u8, []const u8 }{
        .{ "/v1/chat/completions", "{\"model\":\"hi\"," ++ user },
        .{ "/v1/completions", "{\"model\":\"hi\",\"prompt\":\"x\"" },
    }) |route| {
        const multiple = try std.mem.concat(arena, u8, &.{ route[1], ",\"n\":2}" });
        const salted = try std.mem.concat(arena, u8, &.{ route[1], ",\"cache_salt\":\"tenant\"}" });
        const choices = try stand.post(arena, route[0], multiple);
        const salt = try stand.post(arena, route[0], salted);
        try testing.expect(std.mem.startsWith(u8, choices, "HTTP/1.1 400"));
        try testing.expect(std.mem.indexOf(u8, choices, "n must be 1") != null);
        try testing.expect(std.mem.startsWith(u8, salt, "HTTP/1.1 400"));
        try testing.expect(std.mem.indexOf(u8, salt, "cache_salt is not supported") != null);
    }
    try testing.expectEqual(0, stand.served.behind.replies.load(.acquire));
}

const tool_f = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}}}}}]";

test "a request the context cannot hold is a 400 with context_length_exceeded before anything is sent, streamed as it is whole, on the four routes: a reply asked past it and a prompt that fills it; under --compact-at too, through compact.prepare" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // `openai.zig`'s `Run.stream` asks `compact.prepare` in place of `chat.prepare` when `--compact-at` is given
    for ([_]?host.server.CompactAt{ null, .auto }) |at| {
        var stand: Stand = undefined;
        try stand.initWith(gpa, .{ .compact_at = at });
        defer stand.deinit();

        // the stand's context is 64 tokens and a conversation renders to 6, a completion's prompt to its bytes
        const reply_of = "This server's maximum context length is 64 tokens, but the rendered prompt has {d} tokens and requests 100 reply tokens, which exceeds the context window. Reduce the prompt to at most 0 prompt tokens or request at most {d} reply tokens, including chat template and thinking tokens.";
        const over = std.fmt.comptimePrint(reply_of, .{ 6, 58 });
        const openai = "{{\"error\": {{\"message\": \"{s}\", \"type\": \"invalid_request_error\", \"param\": \"{s}\", \"code\": \"context_length_exceeded\"}}}}";
        const filled = "This server's maximum context length is 64 tokens, but the rendered prompt has 64 tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation.";
        const cases = [_]struct { []const u8, []const u8, []const u8 }{
            .{ "/v1/chat/completions", "{\"model\":\"hi\",\"max_tokens\":100," ++ user, std.fmt.comptimePrint(openai, .{ over, "messages" }) },
            // with tools a stream opened without the role chunk: the same refusal
            .{ "/v1/chat/completions", "{\"model\":\"hi\",\"max_tokens\":100," ++ user ++ tool_f, std.fmt.comptimePrint(openai, .{ over, "messages" }) },
            .{ "/v1/completions", "{\"model\":\"hi\",\"max_tokens\":100,\"prompt\":\"x\"", std.fmt.comptimePrint(openai, .{ std.fmt.comptimePrint(reply_of, .{ 1, 63 }), "prompt" }) },
            .{ "/v1/completions", "{\"model\":\"hi\",\"prompt\":\"" ++ &@as([64]u8, @splat('x')) ++ "\"", std.fmt.comptimePrint(openai, .{ filled, "prompt" }) },
            .{ "/v1/responses", "{\"model\":\"hi\",\"max_output_tokens\":100,\"input\":\"x\"", std.fmt.comptimePrint(openai, .{ over, "messages" }) },
            .{ "/v1/messages", "{\"model\":\"hi\",\"max_tokens\":100," ++ user, "{\"type\": \"error\", \"error\": {\"type\": \"invalid_request_error\", \"message\": \"" ++ over ++ "\"}}" },
        };
        for (cases) |case| for ([_][]const u8{ "", ",\"stream\":true" }) |stream| {
            try expectRefused(try stand.post(arena, case[0], try std.fmt.allocPrint(arena, "{s}{s}}}", .{ case[1], stream })), case[2]);
        };
        try testing.expectEqual(0, stand.served.behind.replies.load(.acquire));
    }
}

test "a call the token limit cuts is sent as a call, finish_reason tool_calls (vLLM's, spec 101): whole with the parameters closed before the cut, streamed with the cut one closed as it stands; on Messages tool_use, on Responses completed" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    // room for the second case's 70 tokens
    try stand.initWith(gpa, .{ .capacity = 128 });
    defer stand.deinit();

    const cases = [_]struct { says: []const u8, properties: []const u8, max: []const u8, whole: []const u8, streamed: []const u8 }{
        // 42 of the call's 45 tokens: cut inside its one value, which vLLM drops
        .{ .says = "<tool_call>\n<function=f>\n<parameter=x>\nabcdef", .properties = "{\"x\":{\"type\":\"string\"}}", .max = "42", .whole = "{}", .streamed = "{\"x\":\"abc\"}" },
        // cut inside its second value: the first is kept whole
        .{ .says = "<tool_call>\n<function=f>\n<parameter=x>\na\n</parameter>\n<parameter=y>\nbcdef", .properties = "{\"x\":{\"type\":\"string\"},\"y\":{\"type\":\"string\"}}", .max = "70", .whole = "{\"x\":\"a\"}", .streamed = "{\"x\":\"a\",\"y\":\"bc\"}" },
    };
    for (cases) |case| {
        stand.served.behind.says = case.says;
        // the tool in each API's form
        const schema = try std.fmt.allocPrint(arena, "{{\"type\":\"object\",\"properties\":{s}}}", .{case.properties});
        const asked = try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",{s},\"tools\":[{{\"type\":\"function\",\"function\":{{\"name\":\"f\",\"parameters\":{s}}}}}],\"max_tokens\":{s}", .{ user, schema, case.max });
        const messages_asked = try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",{s},\"tools\":[{{\"name\":\"f\",\"input_schema\":{s}}}],\"max_tokens\":{s}", .{ user, schema, case.max });
        const responses_asked = try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",\"input\":\"x\",\"tools\":[{{\"type\":\"function\",\"name\":\"f\",\"parameters\":{s}}}],\"max_output_tokens\":{s}", .{ schema, case.max });
        const whole = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/chat/completions", try std.fmt.allocPrint(arena, "{s}}}", .{asked})), "HTTP/1.1 200 OK\r\n"));
        const choice = whole.get("choices").?.array[0];
        try testing.expectEqualStrings("tool_calls", choice.get("finish_reason").?.string);
        try testing.expect(choice.get("message").?.get("content").? == .null);
        const call = choice.get("message").?.get("tool_calls").?.array;
        try testing.expectEqual(1, call.len);
        try testing.expectEqualStrings("f", call[0].get("function").?.get("name").?.string);
        try testing.expectEqualStrings(case.whole, call[0].get("function").?.get("arguments").?.string);

        const stream = try bodyOf(try stand.post(arena, "/v1/chat/completions", try std.fmt.allocPrint(arena, "{s},\"stream\":true}}", .{asked})), "HTTP/1.1 200 OK\r\n");
        errdefer std.debug.print("the cut call, streamed:\n{s}\n", .{stream});
        try testing.expectEqualStrings(case.streamed, try streamedArguments(arena, stream));
        // nothing of the markup is content, and the reason is the call's
        try testing.expect(std.mem.indexOf(u8, stream, "\"content\"") == null);
        try testing.expect(std.mem.indexOf(u8, stream, "\"finish_reason\": \"tool_calls\"") != null);
        try testing.expect(std.mem.indexOf(u8, stream, "\"finish_reason\": \"length\"") == null);

        // Messages: a tool_use block with the parameters as its input, stop_reason tool_use
        const messages = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/messages", try std.fmt.allocPrint(arena, "{s}}}", .{messages_asked})), "HTTP/1.1 200 OK\r\n"));
        try testing.expectEqualStrings("tool_use", messages.get("stop_reason").?.string);
        const block = messages.get("content").?.array;
        try testing.expectEqualStrings("tool_use", block[block.len - 1].get("type").?.string);
        try testing.expectEqualStrings(case.whole, try host.json.stringify(arena, block[block.len - 1].get("input").?, .{ .ascii = false, .compact = true }));
        const messages_stream = try bodyOf(try stand.post(arena, "/v1/messages", try std.fmt.allocPrint(arena, "{s},\"stream\":true}}", .{messages_asked})), "HTTP/1.1 200 OK\r\n");
        errdefer std.debug.print("Messages, streamed:\n{s}\n", .{messages_stream});
        try testing.expect(std.mem.indexOf(u8, messages_stream, "\"stop_reason\": \"tool_use\"") != null);
        try testing.expect(std.mem.indexOf(u8, messages_stream, "max_tokens\"") == null);

        // Responses: a function_call item, the response completed and not incomplete
        const responses = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/responses", try std.fmt.allocPrint(arena, "{s}}}", .{responses_asked})), "HTTP/1.1 200 OK\r\n"));
        try testing.expectEqualStrings("completed", responses.get("status").?.string);
        const items = responses.get("output").?.array;
        const item = items[items.len - 1];
        try testing.expectEqualStrings("function_call", item.get("type").?.string);
        try testing.expectEqualStrings(case.whole, item.get("arguments").?.string);
        const responses_stream = try bodyOf(try stand.post(arena, "/v1/responses", try std.fmt.allocPrint(arena, "{s},\"stream\":true}}", .{responses_asked})), "HTTP/1.1 200 OK\r\n");
        errdefer std.debug.print("Responses, streamed:\n{s}\n", .{responses_stream});
        try testing.expect(std.mem.indexOf(u8, responses_stream, "event: response.completed") != null);
        try testing.expect(std.mem.indexOf(u8, responses_stream, "\"status\": \"incomplete\"") == null);
    }
}

test "a reply that is bare JSON naming an offered tool is content, whole and streamed, and never a call (vLLM and SGLang read no bare call, spec 101)" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.init(gpa);
    defer stand.deinit();
    const asked = "{\"model\":\"hi\"," ++ user ++ tool_f;

    stand.served.behind.says = "{\"name\":\"f\",\"arguments\":{\"x\":\"a\"}}";
    const bare = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/chat/completions", asked ++ "}"), "HTTP/1.1 200 OK\r\n"));
    const choice = bare.get("choices").?.array[0];
    try testing.expectEqualStrings("stop", choice.get("finish_reason").?.string);
    try testing.expectEqualStrings(stand.served.behind.says, choice.get("message").?.get("content").?.string);
    try testing.expect(choice.get("message").?.get("tool_calls") == null);
    const bare_stream = try bodyOf(try stand.post(arena, "/v1/chat/completions", asked ++ ",\"stream\":true}"), "HTTP/1.1 200 OK\r\n");
    errdefer std.debug.print("the bare call, streamed:\n{s}\n", .{bare_stream});
    // its text goes out as content, a byte a delta here, and once
    try testing.expectEqual(stand.served.behind.says.len, std.mem.count(u8, bare_stream, "\"delta\": {\"content\": \""));
    try testing.expect(std.mem.indexOf(u8, bare_stream, "tool_calls") == null);
    try testing.expect(std.mem.indexOf(u8, bare_stream, "\"finish_reason\": \"stop\"") != null);
}

test "serve's defaults (spec 101): port 8888, and a reply that names no max_tokens takes what the context leaves, past TensorFold's 4,096, while --max-tokens still binds" {
    const defaults: host.cli.Args = .{};
    try testing.expectEqual(8888, defaults.port);
    try testing.expectEqual(0, defaults.max_tokens);
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var usage: host.cli.Usage = .{};
    try testing.expectEqual(10, (try host.cli.parse(arena, &.{ "/models/x", "--max-tokens", "10" }, &usage)).max_tokens);

    const says: []const u8 = &@as([4200]u8, @splat('a'));
    for ([_]struct { i64, []const u8, usize }{ .{ 0, "stop", says.len }, .{ 10, "length", 10 } }) |case| {
        var stand: Stand = undefined;
        try stand.initWith(testing.allocator, .{ .capacity = 4300, .default_max_tokens = case[0] });
        defer stand.deinit();
        stand.served.behind.says = says;
        for ([_][]const u8{ "}", ",\"max_tokens\":0}" }) |end| {
            const whole = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/chat/completions", try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",{s}{s}", .{ user, end })), "HTTP/1.1 200 OK\r\n"));
            const choice = whole.get("choices").?.array[0];
            try testing.expectEqualStrings(case[1], choice.get("finish_reason").?.string);
            try testing.expectEqual(case[2], choice.get("message").?.get("content").?.string.len);
        }
    }
}

/// A chat completion stream's call arguments, its deltas' fragments joined in order.
fn streamedArguments(arena: Allocator, stream: []const u8) ![]const u8 {
    var joined: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, stream, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: {")) continue;
        const event = try jsonOf(arena, line["data: ".len..]);
        const choices = event.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        const calls = choices.array[0].get("delta").?.get("tool_calls") orelse continue;
        for (calls.array) |call| if (call.get("function").?.get("arguments")) |fragment| try joined.appendSlice(arena, fragment.string);
    }
    return joined.items;
}

test "what TensorFold 1.0.4's and 1.0.5's host newly answers over this engine: a model's context in /models, /memory, Sliding Weights and logprobs refused, the pinned bytes in /metrics, and no turn prefilled behind a reply" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.init(gpa);
    defer stand.deinit();

    const models = try jsonOf(arena, try bodyOf(try stand.get(arena, "/v1/models"), "HTTP/1.1 200 OK\r\n"));
    const model = models.get("data").?.array[0];
    try testing.expectEqualStrings("hi", model.get("id").?.string);
    try testing.expectEqualStrings("model", model.get("object").?.string);
    try testing.expectEqual(@as(f64, 64), model.get("context_length").?.float64().?);
    try testing.expectEqual(@as(f64, 64), model.get("max_model_len").?.float64().?);
    const capabilities = model.get("capabilities").?;
    try testing.expect(capabilities.get("text_generation").?.bool);
    try testing.expect(capabilities.get("prompt_cache").?.bool);
    try testing.expect(!capabilities.get("logprobs").?.bool);
    const inputs = model.get("input_types").?.array;
    try testing.expectEqual(@as(usize, 1), inputs.len);
    try testing.expectEqualStrings("text", inputs[0].string);

    // a new route: the context, and the device's bytes as `Served.memory` counts them; no process footprint
    // on Linux and no prompt-cache plan, which the engine does not give
    for ([_][]const u8{ "/memory", "/v1/memory" }) |path| {
        const memory = try bodyOf(try stand.get(arena, path), "HTTP/1.1 200 OK\r\n");
        try testing.expectEqualStrings("{\"schema_version\": 1, \"context_window\": 64, \"context_fitted\": false, \"device\": {\"active_bytes\": 0, \"cache_bytes\": 0, \"peak_bytes\": 0, \"peak_scope\": \"since_start_or_health_reset\"}}", memory);
    }

    // Sliding Weights: the engine has no `learn`, so a lesson is refused before anything is asked of it
    const lesson = try bodyOf(try stand.post(arena, "/v1/slide/learn", "{\"text\":\"I like blue.\"}"), "HTTP/1.1 501 ");
    try testing.expect(std.mem.indexOf(u8, lesson, "this engine does not learn: serve its model with --slide") != null);
    const graph = try bodyOf(try stand.get(arena, "/v1/slide/graph"), "HTTP/1.1 200 OK\r\n");
    errdefer std.debug.print("/v1/slide/graph:\n{s}\n", .{graph});
    try testing.expect(std.mem.indexOf(u8, graph, "I like blue") == null);
    try testing.expectEqual(0, stand.served.behind.replies.load(.acquire));

    // 1.0.5: logprobs are an engine's to give (`Info.logprobs`), and this one gives none: the 400 of 1.0.2,
    // in its words, before anything is asked of the engine
    const logprobs = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"logprobs\":true,\"top_logprobs\":2}");
    try expectRefused(logprobs, "{\"error\": {\"message\": \"logprobs are not supported by this model or backend\", \"type\": \"invalid_request_error\"}}");
    try testing.expectEqual(0, stand.served.behind.replies.load(.acquire));

    // 1.0.5: pinned bytes and the kept-prompt counters, enabled by `Info.prompt_cache`.
    const metrics = try bodyOf(try stand.get(arena, "/metrics"), "HTTP/1.1 200 OK\r\n");
    errdefer std.debug.print("/metrics:\n{s}\n", .{metrics});
    try testing.expect(std.mem.indexOf(u8, metrics, "\ntensorfold:device_memory_bytes 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, metrics, "\ntensorfold:pinned_memory_bytes 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, metrics, "\ntensorfold:prefix_cache_queries_total 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, metrics, "\ntensorfold:prefix_cache_hits_total 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, metrics, "\ntensorfold:prompt_tokens_cached_total 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, metrics, "process_footprint") == null);

    // `Info.warm_turns` is left false: a reply is one request of the engine, whole or streamed, and the
    // host submits no prompt-only request behind it for the next turn
    for ([_][]const u8{ "", ",\"stream\":true" }) |stream| {
        const reply = try stand.post(arena, "/v1/chat/completions", try std.mem.concat(arena, u8, &.{ "{\"model\":\"hi\",", user, stream, "}" }));
        try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 OK\r\n"));
    }
    try testing.expectEqual(2, stand.served.behind.replies.load(.acquire));
}

test "cache observability: a second conversation turn hits the prefix, with retained states and startup budget in metrics and memory" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.initWith(testing.allocator, .{ .prefix_cache = true, .cache_capacity = .{ .budget_bytes = 4096, .minimum_state_bytes = 1024 }, .prompt_cache_plan = .{ .source = .explicit, .budget_bytes = 4096, .explicit_budget = true } });
    defer stand.deinit();
    stand.text.contents = true;
    const first = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ "}");
    try testing.expect(std.mem.startsWith(u8, first, "HTTP/1.1 200 OK\r\n"));
    const second = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"assistant\",\"content\":\"hi\"},{\"role\":\"user\",\"content\":\"y\"}]}");
    try testing.expect(std.mem.startsWith(u8, second, "HTTP/1.1 200 OK\r\n"));
    const metrics = try bodyOf(try stand.get(arena, "/metrics"), "HTTP/1.1 200 OK\r\n");
    errdefer std.debug.print("cache metrics:\n{s}\n", .{metrics});
    for ([_][]const u8{ "prefix_cache_queries_total 9", "prefix_cache_hits_total 5", "prompt_tokens_cached_total 5", "prompt_cache_states 1", "prompt_cache_bytes 0", "prompt_cache_budget_bytes 4096", "prompt_cache_minimum_state_bytes 1024", "prompt_cache_can_keep_state 1" }) |line| {
        try testing.expect(std.mem.indexOf(u8, metrics, try std.fmt.allocPrint(arena, "\ntensorfold:{s}\n", .{line})) != null);
    }
    const memory = try jsonOf(arena, try bodyOf(try stand.get(arena, "/memory"), "HTTP/1.1 200 OK\r\n"));
    const plan = memory.get("prompt_cache_plan").?;
    try testing.expectEqualStrings("4096", plan.get("budget_bytes").?.int);
    try testing.expect(plan.get("explicit_budget").?.bool);
}

test "prefill status distinguishes a session admission from queued work, generated tokens and warm passes" {
    var stand: Stand = undefined;
    try stand.init(testing.allocator);
    defer stand.deinit();
    const engine = stand.served.engine();
    const Live = @TypeOf(stand.served).Live;
    const request: api.Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 2 };
    var kept: Kept = .{};
    var live: Live = .{ .served = &stand.served, .id = 1, .request = &request, .sink = kept.sink() };
    try stand.served.live.put(testing.allocator, 1, &live);
    defer _ = stand.served.live.remove(1);
    var status: api.Status = .{};
    var streams: [1]u32 = undefined;
    engine.status(&status, &streams);
    try testing.expect(!status.fill_unknown);
    try testing.expect(live.enters());
    live.begins();
    engine.status(&status, &streams);
    try testing.expect(status.fill_unknown);
    _ = live.give(3);
    engine.status(&status, &streams);
    try testing.expect(!status.fill_unknown);
    live.leaves();
    const warm: api.Request = .{ .prompt = request.prompt, .max_tokens = 0, .background = true };
    live.request = &warm;
    try testing.expect(live.enters());
    defer live.leaves();
    engine.status(&status, &streams);
    try testing.expect(!status.fill_unknown);
}

test "queued_seconds measures submission through the session's admission, excluding its prompt" {
    var stand: Stand = undefined;
    try stand.init(testing.allocator);
    defer stand.deinit();
    const served = &stand.served;
    const engine = served.engine();
    served.busy = true;
    const request: api.Request = .{ .prompt = &.{ 1, 2 }, .max_tokens = 2 };
    var kept: Kept = .{};
    try engine.submit(7, &request, kept.sink());
    var waiting = false;
    for (0..400) |_| {
        served.mutex.lockUncancelable(served.io);
        waiting = served.first != null;
        served.mutex.unlock(served.io);
        if (waiting) break;
        try testing.io.sleep(.fromMilliseconds(1), .awake);
    }
    var status: api.Status = .{};
    engine.status(&status, &.{});
    served.mutex.lockUncancelable(served.io);
    served.busy = false;
    served.turned.broadcast(served.io);
    served.mutex.unlock(served.io);
    try kept.wait();
    try testing.expect(waiting);
    try testing.expect(!status.fill_unknown);
    try testing.expect(kept.queued_seconds > 0);
}

test "under warm_turns a chat reply is followed by the host's pass for the next turn, which reaches the model as a warm ask of one token and sends its client nothing; a completion has none" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.initWith(gpa, .{ .warm_turns = true, .capacity = 4096 });
    defer stand.deinit();
    const behind = &stand.served.behind;
    for ([_][]const u8{ "", ",\"stream\":true" }, 1..) |stream, nth| {
        const reply = try stand.post(arena, "/v1/chat/completions", try std.mem.concat(arena, u8, &.{ "{\"model\":\"hi\",", user, stream, "}" }));
        try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 OK\r\n"));
        try testing.expect(std.mem.indexOf(u8, reply, "hi") != null);
        // the pass is submitted after the reply's end: the host's thread, a moment later
        var spins: usize = 0;
        while (behind.warm.load(.acquire) < nth and spins < 400) : (spins += 1) try std.Io.sleep(testing.io, .fromMilliseconds(5), .awake);
        try testing.expectEqual(nth, behind.warm.load(.acquire));
        try testing.expectEqual(2 * nth, behind.replies.load(.acquire));
    }
    const completion = try stand.post(arena, "/v1/completions", "{\"model\":\"hi\",\"prompt\":\"hello\",\"max_tokens\":4}");
    try testing.expect(std.mem.startsWith(u8, completion, "HTTP/1.1 200 OK\r\n"));
    try std.Io.sleep(testing.io, .fromMilliseconds(50), .awake);
    try testing.expectEqual(2, behind.warm.load(.acquire));
    try testing.expectEqual(5, behind.replies.load(.acquire));
}

/// A model that thinks sixty-four tokens of its own and then one token over and over, and answers once
/// `</think>` stands: decoded a row a token under whatever `Live.constrained` lends, as a session decodes.
const Looper = struct {
    pub const Behind = struct {};
    const Live = served_file.Served(Looper).Live;
    const close = [_]u32{ 3, 4, 5 };
    const end: u32 = 4;
    const again: u32 = 7;
    const answer: u32 = 9;

    pub fn reply(live: *Live, arena: Allocator, ask: served_file.Ask, said: *request_file.Diagnostic) !served_file.Ended {
        const held = try live.constrained(arena, true, said);
        const words = live.served.compiler.?.engine.words;
        var masks: ?masks_file.Masks = if (held) |constraint| try .init(constraint, try arena.alloc(u32, words), end) else null;
        const logits = try arena.alloc(u16, words * 32);
        var answering = false;
        for (0..ask.max_tokens) |made| {
            const preferred: u32 = if (answering) answer else if (made < 64) @intCast(20 + made) else again;
            @memset(logits, 0x3f80);
            logits[preferred] = 0x4000;
            if (masks) |*under| {
                _ = try under.window(&.{});
                try under.apply(0, logits);
            }
            const token: u32 = if (logits[preferred] != masks_file.minus_infinity) preferred else @intCast(std.mem.indexOfNone(u16, logits, &.{masks_file.minus_infinity}).?);
            if (masks) |*under| try under.commit(1, @intCast(token));
            _ = live.give(@intCast(token));
            if (token == end) answering = true;
        }
        return .{ .reason = .length };
    }

    pub fn counts(_: *Behind) ?sched.Counts {
        return null;
    }

    pub fn closed(_: *Behind) void {}
};

test "a request's loop guard and thinking budget are the engine's: the close forced as TensorFold forces it, the period in the reply's stats, and a request that asks for neither decoded under nothing" {
    const gpa = testing.allocator;
    var bench: structured_fixture.Bench = undefined;
    try bench.init(gpa);
    defer bench.deinit(gpa);
    var frontend: text_frontend.Frontend = undefined;
    var device_bytes: usize = 0;
    var served: served_file.Served(Looper) = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = bench.compiler(), .behind = .{}, .capacity = 1024, .vocab = 256, .lanes = 1, .think = null, .device_bytes = &device_bytes };
    defer served.close();
    const engine = served.engine();
    // the host refuses `--loop-guard` to an engine that does not say this (`server/serve.zig`)
    try testing.expect(engine.info().loop_guard);
    const prompt = [_]u32{ 1, 2, 3 };

    // the cycle of one completes with the reply's token 321; the close follows it, and the answer
    var looped: Kept = .{};
    try engine.submit(1, &.{ .prompt = &prompt, .max_tokens = 330, .loop_guard = true, .think_close = &Looper.close, .think_end = Looper.end }, looped.sink());
    try looped.wait();
    try testing.expectEqual(api.Reason.length, looped.reason.?);
    try testing.expectEqual(330, looped.count);
    try testing.expectEqual(1, looped.period.?);
    for (looped.tokens[0..64], 0..) |token, index| try testing.expectEqual(20 + index, token);
    try testing.expectEqual(257, std.mem.count(u32, looped.tokens[0..321], &.{Looper.again}));
    try testing.expectEqualSlices(u32, &.{ 3, 4, 5, 9, 9, 9, 9, 9, 9 }, looped.tokens[321..330]);

    // a budget of nine, the guard beside it: the ninth token is the close's first
    var budgeted: Kept = .{};
    try engine.submit(2, &.{ .prompt = &prompt, .max_tokens = 16, .think_budget = 9, .loop_guard = true, .think_close = &Looper.close, .think_end = Looper.end }, budgeted.sink());
    try budgeted.wait();
    try testing.expectEqual(null, budgeted.period);
    try testing.expectEqualSlices(u32, &.{ 27, 3, 4, 5, 9, 9, 9, 9, 9 }, budgeted.tokens[7..16]);

    // neither: no constraint is lent, and the model's own tokens are the reply
    var plain: Kept = .{};
    try engine.submit(3, &.{ .prompt = &prompt, .max_tokens = 330 }, plain.sink());
    try plain.wait();
    try testing.expectEqual(null, plain.period);
    try testing.expectEqual(330 - 64, std.mem.count(u32, plain.tokens[0..330], &.{Looper.again}));
    // and the guard on, the request's reply not left thinking by the host (no closing tokens): the same
    var unasked: Kept = .{};
    try engine.submit(4, &.{ .prompt = &prompt, .max_tokens = 330, .loop_guard = true }, unasked.sink());
    try unasked.wait();
    try testing.expectEqualSlices(u32, plain.tokens[0..330], unasked.tokens[0..330]);
}

test "with xgrammar (libtfgrammar) a regex, a choice, a grammar and a required call are compiled by the host and decoded under xgrammar's masks; without it they are zspark's as before" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // where the library cannot be loaded (`xgrammar_test.zig` says how a test finds it) the first test covers this
    {
        var err: [512]u8 = undefined;
        var probe = api.grammar.Compiler.open(testing.io, arena, @embedFile("fixtures/text/tiny_tokenizer.json"), 320, &.{281}, &err) catch return error.SkipZigTest;
        probe.close();
    }
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "tokenizer.json", .data = @embedFile("fixtures/text/tiny_tokenizer.json") });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{\"vocab_size\": 320}" });
    var stand: Stand = undefined;
    try stand.initWith(gpa, .{ .model_dir = try std.fmt.allocPrint(arena, ".zig-cache/tmp/{s}", .{tmp.sub_path}) });
    defer stand.deinit();

    const asks = [_][]const u8{
        "\"guided_regex\":\"h[a-z]\"",
        "\"guided_choice\":[\"hi\",\"ho\"]",
        "\"guided_grammar\":\"root ::= \\\"hi\\\"\"",
        "\"guided_grammar\":\"root ::= \\\"hi\\\" | [" ++ &@as([129]u8, @splat('(')) ++ "]\"",
        "\"guided_grammar\":\"# " ++ &@as([129]u8, @splat('(')) ++ "\\nroot ::= \\\"hi\\\"\"",
        "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"n\",\"schema\":{\"type\":\"string\",\"pattern\":\"a\"}}}",
    };
    inline for (asks) |ask| {
        const whole = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ "," ++ ask ++ "}");
        _ = try bodyOf(whole, "HTTP/1.1 200 OK\r\n");
    }
    // a grammar xgrammar cannot compile is the host's 400, before the engine
    const bad = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"guided_regex\":\"(\"}");
    try testing.expect(std.mem.startsWith(u8, bad, "HTTP/1.1 400 Bad Request\r\n"));
    // every reply was held to a constraint, and zspark's own matcher compiled none of them
    try testing.expectEqual(asks.len, stand.served.behind.replies.load(.acquire));
    try testing.expectEqual(asks.len, stand.served.behind.constrained.load(.acquire));
    try testing.expectEqual(0, stand.counting.compiles.load(.acquire));
}

test "the no-repeat n-gram fields and max_page_tokens reach the engine's request: top-level, or under vllm_xargs, or refused" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.init(gpa);
    defer stand.deinit();
    const behind = &stand.served.behind;

    const top = "{\"model\":\"hi\"," ++ user ++ ",\"ngram_size\":35,\"window_size\":128,\"whitelist_token_ids\":[128821,128822]}";
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", top), "HTTP/1.1 200 OK\r\n");
    try testing.expectEqual(@as(?u32, 35), behind.no_repeat.size);
    try testing.expectEqual(@as(?u32, 128), behind.no_repeat.window);
    try testing.expectEqualSlices(u32, &.{ 128821, 128822 }, behind.no_repeat_whitelist[0..behind.no_repeat_whitelisted]);

    // under vllm_xargs; a top-level field wins over it
    const xargs = "{\"model\":\"hi\"," ++ user ++ ",\"window_size\":64,\"vllm_xargs\":{\"ngram_size\":7,\"window_size\":1024}}";
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", xargs), "HTTP/1.1 200 OK\r\n");
    try testing.expectEqual(@as(?u32, 7), behind.no_repeat.size);
    try testing.expectEqual(@as(?u32, 64), behind.no_repeat.window);
    try testing.expectEqual(@as(usize, 0), behind.no_repeat_whitelisted);

    // Unlimited-OCR-2's page cap (spec 147), the same ways; SGLang's `max(0, int(value))`
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"max_page_tokens\":300}"), "HTTP/1.1 200 OK\r\n");
    try testing.expectEqual(@as(?u32, 300), behind.max_page_tokens);
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"vllm_xargs\":{\"max_page_tokens\":-5}}"), "HTTP/1.1 200 OK\r\n");
    try testing.expectEqual(@as(?u32, 0), behind.max_page_tokens);
    const page_text = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"max_page_tokens\":\"x\"}");
    try testing.expect(std.mem.startsWith(u8, page_text, "HTTP/1.1 400 Bad Request\r\n"));

    // none named: none asked
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ "}"), "HTTP/1.1 200 OK\r\n");
    try testing.expectEqual(@as(?u32, null), behind.no_repeat.size);
    try testing.expectEqual(@as(?u32, null), behind.max_page_tokens);

    // vLLM's validation: strictly positive integers and a list of ids
    for ([_][]const u8{ "\"ngram_size\":0", "\"ngram_size\":-3", "\"ngram_size\":\"x\"", "\"window_size\":0", "\"whitelist_token_ids\":5", "\"whitelist_token_ids\":[-1]" }) |bad| {
        const whole = try stand.post(arena, "/v1/chat/completions", try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",{s},{s}}}", .{ user, bad }));
        try testing.expect(std.mem.startsWith(u8, whole, "HTTP/1.1 400 Bad Request\r\n"));
    }
}

test "SGLang's Unlimited-OCR-2 fields reach the engine's request: doc_split_options, images_config, mm_processor_kwargs, custom_params" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.init(gpa);
    defer stand.deinit();
    const behind = &stand.served.behind;
    const ok = "HTTP/1.1 200 OK\r\n";
    const head = "{\"model\":\"hi\"," ++ user;

    // doc_split_options: DocSplitOptions, nulls where unnamed; pydantic's refusals
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", head ++ ",\"doc_split_options\":{\"max_batch_pages\":10}}"), ok);
    try testing.expectEqual(api.DocSplit{ .max_batch_pages = 10 }, behind.doc_split);
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", head ++ ",\"doc_split_options\":{\"max_batch_pages\":null,\"overlap_pages\":0}}"), ok);
    try testing.expectEqual(api.DocSplit{ .overlap_pages = 0 }, behind.doc_split);
    for ([_][]const u8{ "{\"max_batch_pages\":0}", "{\"overlap_pages\":-1}", "{\"pages\":2}", "{\"max_batch_pages\":\"x\"}", "5" }) |bad| {
        const whole = try stand.post(arena, "/v1/chat/completions", try std.fmt.allocPrint(arena, "{s},\"doc_split_options\":{s}}}", .{ head, bad }));
        try testing.expect(std.mem.startsWith(u8, whole, "HTTP/1.1 400 Bad Request\r\n"));
    }

    // the image mode: images_config over mm_processor_kwargs; image_mode before crop_mode; an empty image_mode gundam
    const modes = [_]struct { []const u8, ?[]const u8 }{
        .{ "", null },
        .{ ",\"images_config\":{\"crop_mode\":false}", "base" },
        .{ ",\"images_config\":{\"crop_mode\":true}", "gundam" },
        .{ ",\"images_config\":{\"crop_mode\":null}", null },
        .{ ",\"images_config\":{\"image_mode\":\"Base\",\"crop_mode\":true}", "Base" },
        .{ ",\"images_config\":{\"image_mode\":null,\"crop_mode\":false}", "gundam" },
        .{ ",\"mm_processor_kwargs\":{\"crop_mode\":false}", "base" },
        .{ ",\"mm_processor_kwargs\":{\"image_mode\":\"tiny\"},\"images_config\":{\"crop_mode\":true}", "tiny" },
        .{ ",\"mm_processor_kwargs\":{\"crop_mode\":true},\"images_config\":{\"image_mode\":\"base\"}", "base" },
    };
    for (modes) |m| {
        _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", try std.fmt.allocPrint(arena, "{s}{s}}}", .{ head, m[0] })), ok);
        if (m[1]) |want| try testing.expectEqualStrings(want, behind.image_mode[0..behind.image_mode_len.?]) else try testing.expectEqual(@as(?usize, null), behind.image_mode_len);
    }
    try testing.expect(std.mem.startsWith(u8, try stand.post(arena, "/v1/chat/completions", head ++ ",\"images_config\":5}"), "HTTP/1.1 400 Bad Request\r\n"));

    // custom_params, with a custom_logit_processor named or not (SGLang lays its own): the rep_* fields as
    // int(value or 0), all zero where one is no integer; max_page_tokens lowers the cap of 20,000 and never raises
    // it; the body's own fields win
    const processor = ",\"custom_logit_processor\":\"p\"";
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", head ++ processor ++ ",\"custom_params\":{\"rep_min_pattern_size\":\"2\",\"rep_max_pattern_size\":8.9,\"rep_min_count\":3,\"max_page_tokens\":300,\"ngram_size\":20,\"window_size\":90}}"), ok);
    try testing.expectEqual(api.TailRepeat{ .min_size = 2, .max_size = 8, .min_count = 3 }, behind.tail_repeat);
    try testing.expectEqual(@as(?u32, 300), behind.max_page_tokens);
    try testing.expectEqual(@as(?u32, 20), behind.no_repeat.size);
    try testing.expectEqual(@as(?u32, 90), behind.no_repeat.window);
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", head ++ ",\"custom_params\":{\"rep_max_pattern_size\":8,\"rep_min_count\":3,\"max_page_tokens\":300}}"), ok);
    try testing.expectEqual(api.TailRepeat{ .max_size = 8, .min_count = 3 }, behind.tail_repeat);
    try testing.expectEqual(@as(?u32, 300), behind.max_page_tokens);
    try testing.expect(std.mem.startsWith(u8, try stand.post(arena, "/v1/chat/completions", head ++ ",\"custom_params\":5}"), "HTTP/1.1 400 Bad Request\r\n"));
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", head ++ processor ++ ",\"custom_params\":{\"rep_max_pattern_size\":\"x\",\"rep_min_count\":3,\"max_page_tokens\":90000}}"), ok);
    try testing.expectEqual(api.TailRepeat{}, behind.tail_repeat);
    try testing.expectEqual(@as(?u32, null), behind.max_page_tokens);
    _ = try bodyOf(try stand.post(arena, "/v1/chat/completions", head ++ ",\"max_page_tokens\":50" ++ processor ++ ",\"custom_params\":{\"max_page_tokens\":300}}"), ok);
    try testing.expectEqual(@as(?u32, 50), behind.max_page_tokens);
}

test "skip_special_tokens: absent or true leaves the special tokens out of the reply's text, false keeps them, streamed and not" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.initWith(gpa, .{ .special = 5 });
    defer stand.deinit();
    // `h`, the special byte 5 (Unlimited-OCR's layout tags are special tokens), `i`
    stand.served.behind.says = "h\x05i";

    const asks = [_]struct { []const u8, []const u8 }{
        .{ "", "\"content\": \"hi\"" },
        .{ ",\"skip_special_tokens\":true", "\"content\": \"hi\"" },
        .{ ",\"skip_special_tokens\":false,\"deadline_s\":60", "\"content\": \"h\\u0005i\"" },
    };
    for (asks) |ask| {
        const whole = try stand.post(arena, "/v1/chat/completions", try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",{s}{s}}}", .{ user, ask[0] }));
        errdefer std.debug.print("asked {s}, the reply:\n{s}\n", .{ ask[0], whole });
        try testing.expect(std.mem.startsWith(u8, whole, "HTTP/1.1 200 OK\r\n"));
        try testing.expect(std.mem.indexOf(u8, whole, ask[1]) != null);
        const streamed = try stand.post(arena, "/v1/chat/completions", try std.fmt.allocPrint(arena, "{{\"model\":\"hi\",{s}{s},\"stream\":true}}", .{ user, ask[0] }));
        errdefer std.debug.print("the stream:\n{s}\n", .{streamed});
        try testing.expect(std.mem.endsWith(u8, streamed, "data: [DONE]\n\n"));
        try testing.expectEqual(ask[1].len > "\"content\": \"hi\"".len, std.mem.indexOf(u8, streamed, "\\u0005") != null);
    }

    const refused = try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"skip_special_tokens\":false,\"deadline_s\":0,\"stream\":true}");
    try testing.expect(std.mem.startsWith(u8, refused, "HTTP/1.1 400 Bad Request\r\n"));
    try testing.expect(std.mem.indexOf(u8, refused, "deadline_s") != null);
}
