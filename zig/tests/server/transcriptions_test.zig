//! The host's `POST /v1/audio/transcriptions` (zspark's patch, spec 138 WS-05: `vendor/.../server/transcriptions.zig`)
//! on the loopback, over an engine that hears every file as its length: the multipart form read, the answer and
//! every refusal in vLLM's shape and words. Every test's name says "transcription", for
//! `tools/check.sh server transcription`.

const std = @import("std");
const api = @import("engine_api");
const host = @import("server");
const fixture = @import("served_fixture.zig");
const audio = @import("../../src/audio/audio.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const form = host.transcriptions;

/// A transcription engine that hears `N bytes` in a file of N bytes, 2.2 s long, and refuses the language `xx`.
const Ear = struct {
    /// The last request's language (copied: the request's memory is the route's), and its token limit.
    language: [8]u8 = undefined,
    asked: ?usize = null,
    max_tokens: u32 = 0,
    decode_audio: bool = false,

    /// Chat with audio parts too (spec 150), as Qwen3-ASR's engine: a part of N bytes is N seconds and N rows,
    /// `bad` is undecodable; the reply is `ok`. The last chat's prompt and clips (`at`, `tokens`) are kept.
    chat: bool = false,
    prompt: [64]u32 = undefined,
    prompt_len: usize = 0,
    clips: [4][2]u32 = undefined,
    clip_count: usize = 0,

    fn engine(self: *Ear) api.Engine {
        return .{ .ctx = self, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .submit_transcription = transcribe } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const self: *Ear = @ptrCast(@alignCast(ctx));
        if (!self.chat) return .{ .name = "ear", .text_generation = false, .transcription = .{ .max_tokens = 4096 } };
        return .{
            .name = "ear",
            .context_window = 4096,
            .greedy_only = true,
            .transcription = .{ .max_tokens = 4096 },
            .audio = .{ .ctx = self, .prepare = prepare, .explain = explain, .audio_token = fixture.audio_token, .max_audios = 3, .max_seconds = 20, .max_tokens = 40, .max_bytes = if (self.decode_audio) 25 << 20 else 64 },
        };
    }

    fn prepare(ctx: *anyopaque, a: Allocator, bytes: []const u8) anyerror!api.PreparedAudio {
        const self: *Ear = @ptrCast(@alignCast(ctx));
        if (self.decode_audio) {
            const clip = try audio.clip(a, bytes, 20);
            return .{ .samples = clip.samples, .seconds = clip.seconds, .tokens = 1 };
        }
        if (std.mem.eql(u8, bytes, "bad")) return error.Invalid;
        return .{ .samples = &.{}, .seconds = @floatFromInt(bytes.len), .tokens = @intCast(bytes.len) };
    }

    fn explain(err: anyerror) ?[]const u8 {
        return if (err == error.Invalid) "Invalid or corrupted audio data." else null;
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const self: *Ear = @ptrCast(@alignCast(ctx));
        if (!self.chat) return error.Closed;
        @memcpy(self.prompt[0..r.prompt.len], r.prompt);
        self.prompt_len = r.prompt.len;
        for (r.audios, 0..) |clip, i| self.clips[i] = .{ clip.at, clip.tokens };
        self.clip_count = r.audios.len;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn transcribe(ctx: *anyopaque, id: api.Id, r: *const api.TranscriptionRequest, sink: api.Sink) api.TranscriptionSubmitError!void {
        const self: *Ear = @ptrCast(@alignCast(ctx));
        self.asked = null;
        if (r.language) |l| {
            @memcpy(self.language[0..l.len], l);
            self.asked = l.len;
        }
        self.max_tokens = r.max_tokens;
        const out = r.result;
        if (r.language) |l| if (std.mem.eql(u8, l, "xx")) {
            out.refused = r.gpa.dupe(u8, "Unsupported language: 'xx'.") catch return error.Busy;
            out.param = r.gpa.dupe(u8, "language") catch return error.Busy;
        };
        if (out.refused.len == 0) {
            if (self.decode_audio) {
                const clip = audio.clip(r.gpa, r.audio, 20) catch |err| {
                    out.refused = r.gpa.dupe(u8, audio.message(err)) catch return error.Busy;
                    sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
                    return;
                };
                defer clip.deinit(r.gpa);
                out.text = r.gpa.dupe(u8, "decoded") catch return error.Busy;
                out.seconds = clip.seconds;
            } else {
                out.text = std.fmt.allocPrint(r.gpa, "{d} bytes", .{r.audio.len}) catch return error.Busy;
                out.seconds = 2.2;
            }
            // as the engine's prompt and reply: the file's bytes and 3 tokens, the first stamped by `prefilled`
            out.prompt_tokens = @intCast(r.audio.len);
            out.tokens = 3;
            const prefilled: api.Event = .{ .prefilled = 0 };
            sink.event(sink.ctx, id, &prefilled);
        }
        const event: api.Event = .{ .finished = .{ .reason = .stop, .stats = .{ .prefill_seconds = 0.25 } } };
        sink.event(sink.ctx, id, &event);
    }
};

const Stand = struct {
    ear: Ear = .{},
    text: fixture.Bytes = .{},
    server: *host.Server,
    listener: host.listener.Listener,
    stop: std.atomic.Value(bool),
    accepting: std.Thread,
    port: u16,

    fn init(self: *Stand) !void {
        return self.initWith(false);
    }

    /// `chatting`: the engine answers chat with audio parts, and the text renders each part (`fixture.Bytes.audios`).
    fn initWith(self: *Stand, chatting: bool) !void {
        return self.initEar(.{ .chat = chatting });
    }

    fn initEar(self: *Stand, ear: Ear) !void {
        self.* = .{ .ear = ear, .text = .{ .contents = ear.chat, .audios = ear.chat }, .server = undefined, .listener = undefined, .stop = .init(false), .accepting = undefined, .port = 0 };
        self.server = try .init(testing.allocator, testing.io, self.ear.engine(), self.text.text(), .{ .served_name = "Qwen3-ASR-1.7B", .model_ids = &.{"Qwen3-ASR-1.7B"}, .enable_thinking = false, .keep_warm_s = 0 }, null);
        errdefer self.server.deinit();
        self.listener = try .open(.{ .ip4 = .loopback(0) });
        self.port = self.listener.port();
        self.accepting = try std.Thread.spawn(.{}, host.Server.serve, .{ self.server, self.listener, &self.stop });
    }

    fn deinit(self: *Stand) void {
        self.stop.store(true, .release);
        self.accepting.join();
        while (self.server.open_connections.load(.acquire) != 0) std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        self.listener.close();
        self.server.deinit();
    }

    /// The form of `fields` (name, value; `file` sent as a file part) posted; the whole reply.
    fn post(self: *Stand, a: Allocator, fields: []const [2][]const u8) ![]const u8 {
        var body: std.ArrayList(u8) = .empty;
        for (fields) |f| {
            if (std.mem.eql(u8, f[0], "file")) {
                try body.print(a, "--b0undary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n{s}\r\n", .{f[1]});
            } else try body.print(a, "--b0undary\r\nContent-Disposition: form-data; name=\"{s}\"\r\n\r\n{s}\r\n", .{ f[0], f[1] });
        }
        try body.appendSlice(a, "--b0undary--\r\n");
        return self.raw(a, "multipart/form-data; boundary=b0undary", body.items);
    }

    /// `messages` (JSON) and the request's `more` fields posted to /v1/chat/completions; the whole reply.
    fn chat(self: *Stand, a: Allocator, messages: []const u8, more: []const u8) ![]const u8 {
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"Qwen3-ASR-1.7B\",\"messages\":{s}{s}}}", .{ messages, more });
        return fixture.heard(a, try fixture.sent(self.port, try fixture.posted(a, "/v1/chat/completions", body)));
    }

    fn raw(self: *Stand, a: Allocator, content_type: []const u8, body: []const u8) ![]const u8 {
        const asked = try std.fmt.allocPrint(a, "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: {s}\r\nContent-Length: {d}\r\n\r\n{s}", .{ content_type, body.len, body });
        return fixture.heard(a, try fixture.sent(self.port, asked));
    }
};

/// The reply's status line is `status`; its error object's message, type, param and code are vLLM's.
fn expectError(a: Allocator, reply: []const u8, status: []const u8, message: []const u8, kind: []const u8, param: ?[]const u8) !void {
    const value = try fixture.jsonOf(a, try fixture.bodyOf(reply, status));
    const e = value.get("error").?;
    try testing.expectEqualStrings(message, e.get("message").?.string);
    try testing.expectEqualStrings(kind, e.get("type").?.string);
    if (param) |p| try testing.expectEqualStrings(p, e.get("param").?.string) else try testing.expect(e.get("param").? == .null);
    try testing.expectEqualStrings(status[9..12], e.get("code").?.int);
}

test "transcription: a multipart form's file is heard, json and text alike, and every refusal is vLLM's" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var stand: Stand = undefined;
    try stand.init();
    defer stand.deinit();

    for ([_][]const u8{ "json", "text" }) |format| {
        const reply = try stand.post(a, &.{ .{ "model", "Qwen3-ASR-1.7B" }, .{ "file", "RIFF0123" }, .{ "language", "zh" }, .{ "prompt", "Pod 42" }, .{ "response_format", format } });
        const value = try fixture.jsonOf(a, try fixture.bodyOf(reply, "HTTP/1.1 200 OK"));
        try testing.expectEqualStrings("8 bytes", value.get("text").?.string);
        try testing.expectEqualStrings("duration", value.get("usage").?.get("type").?.string);
        try testing.expectEqualStrings("3", value.get("usage").?.get("seconds").?.int);
        try testing.expectEqualStrings("zh", stand.ear.language[0..stand.ear.asked.?]);
        try testing.expectEqual(@as(u32, 4096), stand.ear.max_tokens);
    }
    // no model, no language, temperature 0 and fields vLLM ignores are taken
    {
        const reply = try stand.post(a, &.{ .{ "file", "x" }, .{ "temperature", "0.0" }, .{ "timestamp_granularities[]", "segment" }, .{ "max_completion_tokens", "12" } });
        _ = try fixture.bodyOf(reply, "HTTP/1.1 200 OK");
        try testing.expect(stand.ear.asked == null);
        try testing.expectEqual(@as(u32, 12), stand.ear.max_tokens);
    }
    try expectError(a, try stand.post(a, &.{.{ "model", "Qwen3-ASR-1.7B" }}), "HTTP/1.1 400 Bad Request", "Field required: file", "BadRequestError", "file");
    try expectError(a, try stand.post(a, &.{ .{ "model", "whisper-1" }, .{ "file", "x" } }), "HTTP/1.1 404 Not Found", "The model `whisper-1` does not exist. Valid aliases: ['Qwen3-ASR-1.7B'].", "NotFoundError", "model");
    try expectError(a, try stand.post(a, &.{ .{ "file", "x" }, .{ "response_format", "verbose_json" } }), "HTTP/1.1 400 Bad Request", "Currently do not support verbose_json for Qwen3-ASR-1.7B", "BadRequestError", "response_format");
    try expectError(a, try stand.post(a, &.{ .{ "file", "x" }, .{ "response_format", "srt" } }), "HTTP/1.1 400 Bad Request", "Currently only support response_format: `text`, `json`, `verbose_json` or `diarized_json`", "BadRequestError", "response_format");
    try expectError(a, try stand.post(a, &.{ .{ "file", "x" }, .{ "temperature", "0.7" } }), "HTTP/1.1 400 Bad Request", "Only greedy decoding is served: temperature must be 0", "BadRequestError", "temperature");
    try expectError(a, try stand.post(a, &.{ .{ "file", "x" }, .{ "stream", "true" } }), "HTTP/1.1 400 Bad Request", "Streaming is not supported for this model", "BadRequestError", "stream");
    try expectError(a, try stand.post(a, &.{ .{ "file", "x" }, .{ "language", "xx" } }), "HTTP/1.1 400 Bad Request", "Unsupported language: 'xx'.", "BadRequestError", "language");
    try expectError(a, try stand.raw(a, "application/json", "{\"file\": \"x\"}"), "HTTP/1.1 400 Bad Request", "Expected a multipart/form-data body", "BadRequestError", null);
    // past 25 MiB: vLLM's words, the size in MiB as Python prints a float
    const big = try a.alloc(u8, 25 * 1024 * 1024 + 512 * 1024);
    @memset(big, 'a');
    try expectError(a, try stand.post(a, &.{.{ "file", big }}), "HTTP/1.1 400 Bad Request", "Maximum file size exceeded (parameter=audio_filesize_mb, value=25.5)", "BadRequestError", "audio_filesize_mb");
}

test "transcription: M4A reaches the shared decoder from multipart and chat input_audio" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m4a = @embedFile("../fixtures/audio/stereo_aac.m4a");
    var stand: Stand = undefined;
    try stand.initEar(.{ .chat = true, .decode_audio = true });
    defer stand.deinit();

    const transcript = try stand.post(a, &.{.{ "file", m4a }});
    const result = try fixture.jsonOf(a, try fixture.bodyOf(transcript, "HTTP/1.1 200 OK"));
    try testing.expectEqualStrings("decoded", result.object.get("text").?.string);
    try testing.expectEqualStrings("1", result.object.get("usage").?.object.get("seconds").?.int);

    const encoder = std.base64.standard.Encoder;
    const encoded = encoder.encode(try a.alloc(u8, encoder.calcSize(m4a.len)), m4a);
    const messages = try std.fmt.allocPrint(a, "[{{\"role\":\"user\",\"content\":[{{\"type\":\"input_audio\",\"input_audio\":{{\"data\":\"{s}\",\"format\":\"m4a\"}}}}]}}]", .{encoded});
    const reply = try stand.chat(a, messages, "");
    _ = try fixture.bodyOf(reply, "HTTP/1.1 200 OK");
    try testing.expectEqual(@as(usize, 1), stand.ear.clip_count);
}

test "transcription: a finished transcription is counted in /metrics as a chat reply is, a refused one not" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var stand: Stand = undefined;
    try stand.init();
    defer stand.deinit();

    _ = try fixture.bodyOf(try stand.post(a, &.{.{ "file", "RIFF0123" }}), "HTTP/1.1 200 OK");
    _ = try fixture.bodyOf(try stand.post(a, &.{.{ "file", "RIFF01234567" }}), "HTTP/1.1 200 OK");
    // refused by the route and by the engine: no reply, nothing counted
    _ = try fixture.bodyOf(try stand.post(a, &.{ .{ "file", "x" }, .{ "stream", "true" } }), "HTTP/1.1 400 Bad Request");
    _ = try fixture.bodyOf(try stand.post(a, &.{ .{ "file", "x" }, .{ "language", "xx" } }), "HTTP/1.1 400 Bad Request");
    const m = &stand.server.metrics;
    try testing.expectEqual(@as(u64, 6), m.generation);
    try testing.expectEqual(@as(u64, 8 + 12), m.prompt);
    try testing.expectEqual(@as(u64, 2), m.latency.n);
    try testing.expectEqual(@as(u64, 2), m.ttft.n);
    try testing.expectEqual(@as(u64, 2), m.decode.n);
    try testing.expectEqual(@as(u64, 2), m.tpot.n);
    try testing.expectEqual(@as(u64, 2), m.prefill.n);
    try testing.expectApproxEqAbs(@as(f64, 0.5), m.prefill.total, 1e-9);
}

/// A chat refused before its reply: a 400 whose error says `message`.
fn expectRefusal(a: Allocator, reply: []const u8, message: []const u8) !void {
    const value = try fixture.jsonOf(a, try fixture.bodyOf(reply, "HTTP/1.1 400 Bad Request"));
    try testing.expectEqualStrings(message, value.get("error").?.get("message").?.string);
}

test "transcription: chat's audio parts, input_audio and audio_url, are each placeholder's rows, and every refusal is vLLM's" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var stand: Stand = undefined;
    try stand.initWith(true);
    defer stand.deinit();

    // 3 and 5 bytes: the template's `|ctx|hi` and a placeholder a part, each placeholder its audio's rows
    const two =
        \\[{"role":"system","content":"ctx"},{"role":"user","content":[{"type":"text","text":"hi"},
        \\{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}},
        \\{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,AAAAAAA="}}]}]
    ;
    {
        const value = try fixture.jsonOf(a, try fixture.bodyOf(try stand.chat(a, two, ""), "HTTP/1.1 200 OK"));
        try testing.expectEqualStrings("ok", value.get("choices").?.array[0].get("message").?.get("content").?.string);
        try testing.expectEqualStrings("15", value.get("usage").?.get("prompt_tokens").?.int);
        try testing.expectEqualStrings("2", value.get("usage").?.get("completion_tokens").?.int);
        const t = fixture.audio_token;
        try testing.expectEqualSlices(u32, &.{ '|', 'c', 't', 'x', '|', 'h', 'i', t, t, t, t, t, t, t, t }, stand.ear.prompt[0..stand.ear.prompt_len]);
        try testing.expectEqual(@as(usize, 2), stand.ear.clip_count);
        try testing.expectEqualSlices(u32, &.{ 7, 3, 10, 5 }, &.{ stand.ear.clips[0][0], stand.ear.clips[0][1], stand.ear.clips[1][0], stand.ear.clips[1][1] });
    }
    // streamed alike; a text-only chat is answered, as vLLM answers it for this model
    {
        const body = try fixture.bodyOf(try stand.chat(a, two, ",\"stream\":true"), "HTTP/1.1 200 OK");
        try testing.expect(std.mem.indexOf(u8, body, "data: [DONE]") != null);
        _ = try fixture.bodyOf(try stand.chat(a, "[{\"role\":\"user\",\"content\":\"hi\"}]", ""), "HTTP/1.1 200 OK");
        try testing.expectEqual(@as(usize, 0), stand.ear.clip_count);
    }
    const user = .{ "[{\"role\":\"user\",\"content\":[", "]}]" };
    const Case = struct { []const u8, []const u8 };
    // 3 bytes; 12 bytes; 72 bytes, past the engine's 64
    const sixteen = "AAAAAAAAAAAAAAAA";
    const small = "{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAA\",\"format\":\"wav\"}}";
    const long = "{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAAAAAAAAAAAAAA\",\"format\":\"wav\"}}";
    const big = "{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"" ++ sixteen ++ sixteen ++ sixteen ++ sixteen ++ sixteen ++ sixteen ++ "\",\"format\":\"wav\"}}";
    for ([_]Case{
        .{ "{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\"}}", "input_audio must contain a base64 data string" },
        .{ "{\"type\":\"audio_url\",\"audio_url\":{}}", "audio_url must contain a url string" },
        .{ "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}", "content parts must be text, audio_url or input_audio; image and video inputs are unsupported" },
        .{ "{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,@@@@\"}}", "invalid audio data URL encoding" },
        .{ "{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"YmFk\",\"format\":\"wav\"}}", "Invalid or corrupted audio data." },
        .{ "{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"http://example.com/a.wav\"}}", "audio requires data URLs or public HTTPS URLs" },
        .{ "{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"https://example.com/a.wav\"}}", "audio URLs are off on this server; send the audio as a data URL or input_audio, or start the server with --vision-urls" },
        .{ big, "Maximum file size exceeded" },
        .{ small ++ "," ++ small ++ "," ++ small ++ "," ++ small, "At most 3 audio(s) may be provided in one prompt." },
        .{ long ++ "," ++ long, "Audio exceeds maximum allowed duration: the request's audio is 24.0 s, at most 20 s are taken" },
    }) |case| {
        errdefer std.debug.print("the part: {s}\n", .{case[0]});
        try expectRefusal(a, try stand.chat(a, try std.mem.concat(a, u8, &.{ user[0], case[0], user[1] }), ""), case[1]);
    }
    try expectRefusal(a, try stand.chat(a, "[{\"role\":\"system\",\"content\":[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAA\",\"format\":\"wav\"}}]}]", ""), "audio parts are accepted in user, assistant and tool messages only");
    try expectRefusal(a, try stand.chat(a, two, ",\"temperature\":0.7"), "this engine decodes greedily only: send temperature 0 or leave it out");
}

test "transcription: a multipart form gives its fields, a file part's bytes whole; a malformed one is refused" {
    const a = testing.allocator;
    const body = "--XyZ\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nQwen3-ASR-1.7B\r\n" ++
        "--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n--\x00\xff\r\n" ++
        "--XyZ\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nzh\r\n" ++
        "--XyZ\r\nContent-Disposition: form-data; name=\"timestamp_granularities[]\"\r\n\r\nword\r\n--XyZ--\r\n";
    const got = try form.parse(a, "multipart/form-data; boundary=XyZ", body);
    try testing.expectEqualStrings("Qwen3-ASR-1.7B", got.model);
    try testing.expectEqualStrings("RIFF\r\n--\x00\xff", got.file.?);
    try testing.expectEqualStrings("zh", got.language.?);
    try testing.expectEqualStrings("json", got.response_format);
    try testing.expectEqualStrings("", got.prompt);

    try testing.expectError(error.NotMultipart, form.parse(a, "application/json", "{}"));
    try testing.expectError(error.NotMultipart, form.parse(a, "multipart/form-data", "--x\r\n"));
    try testing.expectError(error.Malformed, form.parse(a, "multipart/form-data; boundary=\"x\"", "--x\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm"));
    try testing.expectError(error.Malformed, form.parse(a, "multipart/form-data; boundary=x", "nothing here"));
    const empty = try form.parse(a, "Multipart/Form-Data; BOUNDARY=\"x\"", "--x--\r\n");
    try testing.expect(empty.file == null);
}
