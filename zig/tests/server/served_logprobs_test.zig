//! `served.zig`'s logprobs on the host (spec 101 WS-02): `Hi`'s rows behind TensorFold's server, asked over
//! a socket as a client asks them, whole and streamed. Flash Next's own rows are
//! `models/qwen4/served_test.zig`'s (the GPU).

const std = @import("std");
const host = @import("server");
const testing = std.testing;
const fixture = @import("served_fixture.zig");
const Hi = fixture.Hi;
const Stand = fixture.Stand;
const jsonOf = fixture.jsonOf;
const bodyOf = fixture.bodyOf;

const user = "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]";

/// Each of `content`'s entries against the row `Hi` gave its token: OpenAI's `{token, logprob, bytes,
/// top_logprobs}`, as vLLM gives them, the text a byte a token.
fn expectEntries(content: []const host.json.Value, says: []const u8, k: u8) !void {
    try testing.expectEqual(says.len, content.len);
    for (content, says, 0..) |entry, byte, at| {
        const row = Hi.rowOf(byte, at, k);
        try testing.expectEqualStrings(&.{byte}, entry.get("token").?.string);
        try testing.expectEqual(@as(f64, row.logprob), entry.get("logprob").?.float64().?);
        const bytes = entry.get("bytes").?.array;
        try testing.expectEqual(1, bytes.len);
        try testing.expectEqual(@as(f64, @floatFromInt(byte)), bytes[0].float64().?);
        const top = entry.get("top_logprobs").?.array;
        try testing.expectEqual(k, top.len);
        for (top, row.ids[0..k], row.logprobs[0..k]) |alt, id, logprob| {
            try testing.expectEqualStrings(&.{@intCast(id)}, alt.get("token").?.string);
            try testing.expectEqual(@as(f64, logprob), alt.get("logprob").?.float64().?);
        }
    }
}

test "logprobs: each token's row, the reply's text a token an entry, whole and streamed, on chat and legacy completions; what the host refuses is a 400" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stand: Stand = undefined;
    try stand.initWith(gpa, .{ .logprobs = true });
    defer stand.deinit();

    // whole: `choices[0].logprobs.content`, the end token's row given and not in it
    const whole = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"logprobs\":true,\"top_logprobs\":2}"), "HTTP/1.1 200 OK\r\n"));
    const choice = whole.get("choices").?.array[0];
    try testing.expectEqualStrings("hi", choice.get("message").?.get("content").?.string);
    try testing.expectEqualStrings("stop", choice.get("finish_reason").?.string);
    try expectEntries(choice.get("logprobs").?.get("content").?.array, "hi", 2);
    // no top_logprobs: none a row
    const bare = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"logprobs\":true}"), "HTTP/1.1 200 OK\r\n"));
    try expectEntries(bare.get("choices").?.array[0].get("logprobs").?.get("content").?.array, "hi", 0);
    // without logprobs a reply has none
    const plain = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ "}"), "HTTP/1.1 200 OK\r\n"));
    try testing.expectEqual(null, plain.get("choices").?.array[0].field("logprobs"));

    // streamed: each content chunk carries the entries its text spells
    const stream = try bodyOf(try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"logprobs\":true,\"top_logprobs\":3,\"stream\":true}"), "HTTP/1.1 200 OK\r\n");
    errdefer std.debug.print("the stream:\n{s}\n", .{stream});
    var streamed: std.ArrayList(host.json.Value) = .empty;
    var lines = std.mem.splitScalar(u8, stream, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: {")) continue;
        const chunk = try jsonOf(arena, line["data: ".len..]);
        const choices = chunk.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        const lp = choices.array[0].field("logprobs") orelse continue;
        try streamed.appendSlice(arena, lp.get("content").?.array);
    }
    try expectEntries(streamed.items, "hi", 3);

    // legacy completions: `logprobs` is how many best, the chat form's rows in OpenAI's older shape
    const legacy = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/completions", "{\"model\":\"hi\",\"prompt\":\"x\",\"logprobs\":1}"), "HTTP/1.1 200 OK\r\n"));
    const lp = legacy.get("choices").?.array[0].get("logprobs").?;
    try testing.expectEqual(2, lp.get("tokens").?.array.len);
    try testing.expectEqual(@as(f64, -0.5), lp.get("token_logprobs").?.array[0].float64().?);
    try testing.expectEqual(@as(f64, -1.0), lp.get("token_logprobs").?.array[1].float64().?);

    // The top-logprob limit is enforced before the engine receives the request.
    try testing.expect(std.mem.startsWith(u8, try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"logprobs\":true,\"top_logprobs\":21}"), "HTTP/1.1 400 Bad Request\r\n"));
    const stopped = try jsonOf(arena, try bodyOf(try stand.post(arena, "/v1/chat/completions", "{\"model\":\"hi\"," ++ user ++ ",\"logprobs\":true,\"stop\":[\"i\"]}"), "HTTP/1.1 200 OK\r\n"));
    const stopped_choice = stopped.get("choices").?.array[0];
    try testing.expectEqualStrings("h", stopped_choice.get("message").?.get("content").?.string);
    try testing.expectEqualStrings("stop", stopped_choice.get("finish_reason").?.string);
    try expectEntries(stopped_choice.get("logprobs").?.get("content").?.array, "h", 0);
    try testing.expectEqual(6, stand.served.behind.replies.load(.acquire));
}
