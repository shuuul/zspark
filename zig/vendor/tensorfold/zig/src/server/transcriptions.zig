//! zspark: POST /v1/audio/transcriptions (spec 138), in vLLM's shape (``serve/speech_to_text``): a multipart form
//! in, ``{"text", "usage"}`` out, vLLM's error object and words. The engine decodes the file and writes the text.
//! Departures from vLLM: greedy only (a temperature above 0 is refused), no streaming, no 30 s chunking (the engine
//! takes the whole clip, as the Qwen3-ASR package does).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const ids = @import("ids.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

pub const path = "/v1/audio/transcriptions";

/// vLLM's ``VLLM_MAX_AUDIO_CLIP_FILESIZE_MB``, in MiB.
pub const max_file_mb = 25;

/// The form's fields this route reads; the rest are ignored, as FastAPI's form model ignores them.
pub const Form = struct {
    file: ?[]const u8 = null,
    model: []const u8 = "",
    language: ?[]const u8 = null,
    prompt: []const u8 = "",
    response_format: []const u8 = "json",
    temperature: []const u8 = "",
    stream: []const u8 = "",
    max_completion_tokens: []const u8 = "",
};

/// A refusal: vLLM's ``ErrorResponse`` (``{"error": {"message", "type", "param", "code"}}``).
const Refusal = struct { status: u16 = 400, message: []const u8, param: ?[]const u8 = null };

pub fn post(srv: *Server, c: *Conn, a: Allocator) !void {
    const info = srv.info.transcription orelse {
        _ = try http_body.read(c, a, http_body.limit);
        return routes.unknown(c, a);
    };
    const id = try ids.make(a, "trsc-", 32);
    const received = clock.nowNs(srv.io);
    const body = switch (try http_body.read(c, a, http_body.limit)) {
        .ok => |b| b,
        .refused => |m| return send(c, a, id, .{ .message = m }),
    };
    const form = parse(a, c.header("Content-Type") orelse "", body) catch |e| return send(c, a, id, .{ .message = switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.NotMultipart => "Expected a multipart/form-data body",
        error.Malformed => "Malformed multipart/form-data body",
    } });
    const request = check(srv, a, form, info) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Refused => return send(c, a, id, refused),
    };
    transcribe(srv, c, a, id, form, request, received);
}

threadlocal var refused: Refusal = .{ .message = "" };

fn refuse(r: Refusal) error{Refused} {
    refused = r;
    return error.Refused;
}

const Checked = struct { audio: []const u8, max_tokens: u32 };

/// Every refusal vLLM makes before the model runs, in its order and words.
fn check(srv: *Server, a: Allocator, form: Form, info: api.TranscriptionInfo) error{ Refused, OutOfMemory }!Checked {
    if (form.model.len > 0 and !listed(srv.config.model_ids, form.model)) {
        var names: std.ArrayList(u8) = .empty;
        try names.append(a, '[');
        for (srv.config.model_ids, 0..) |name, i| try names.print(a, "{s}'{s}'", .{ if (i > 0) ", " else "", name });
        try names.append(a, ']');
        return refuse(.{ .status = 404, .param = "model", .message = try std.fmt.allocPrint(a, "The model `{s}` does not exist. Valid aliases: {s}.", .{ form.model, names.items }) });
    }
    const audio = form.file orelse return refuse(.{ .param = "file", .message = "Field required: file" });
    if (truthy(form.stream)) return refuse(.{ .param = "stream", .message = "Streaming is not supported for this model" });
    const format = form.response_format;
    if (!std.mem.eql(u8, format, "json") and !std.mem.eql(u8, format, "text")) {
        if (std.mem.eql(u8, format, "verbose_json") or std.mem.eql(u8, format, "diarized_json"))
            return refuse(.{ .param = "response_format", .message = try std.fmt.allocPrint(a, "Currently do not support {s} for {s}", .{ format, srv.config.served_name }) });
        return refuse(.{ .param = "response_format", .message = "Currently only support response_format: `text`, `json`, `verbose_json` or `diarized_json`" });
    }
    if (form.temperature.len > 0) {
        const t = std.fmt.parseFloat(f64, std.mem.trim(u8, form.temperature, " ")) catch
            return refuse(.{ .param = "temperature", .message = "temperature must be a number" });
        if (t != 0) return refuse(.{ .param = "temperature", .message = "Only greedy decoding is served: temperature must be 0" });
    }
    var max_tokens = info.max_tokens;
    if (form.max_completion_tokens.len > 0) {
        const n = std.fmt.parseInt(u32, std.mem.trim(u8, form.max_completion_tokens, " "), 10) catch 0;
        if (n == 0) return refuse(.{ .param = "max_completion_tokens", .message = "max_completion_tokens must be a positive integer" });
        max_tokens = n;
    }
    if (audio.len > max_file_mb * 1024 * 1024) {
        const mb = @as(f64, @floatFromInt(audio.len)) / (1024 * 1024);
        return refuse(.{ .param = "audio_filesize_mb", .message = try std.fmt.allocPrint(a, "Maximum file size exceeded (parameter=audio_filesize_mb, value={d})", .{mb}) });
    }
    return .{ .audio = audio, .max_tokens = max_tokens };
}

fn truthy(v: []const u8) bool {
    const t = std.mem.trim(u8, v, " ");
    return std.ascii.eqlIgnoreCase(t, "true") or std.mem.eql(u8, t, "1") or std.ascii.eqlIgnoreCase(t, "yes") or std.ascii.eqlIgnoreCase(t, "on");
}

fn listed(names: []const []const u8, key: []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, name, key)) return true;
    return false;
}

fn transcribe(srv: *Server, c: *Conn, a: Allocator, id: []const u8, form: Form, checked: Checked, received: i96) void {
    var result: api.TranscriptionResult = .{};
    defer result.deinit(srv.gpa);
    const request: api.TranscriptionRequest = .{
        .audio = checked.audio,
        .language = form.language,
        .prompt = form.prompt,
        .max_tokens = checked.max_tokens,
        .result = &result,
        .gpa = srv.gpa,
    };
    var box: Mailbox = .{ .io = srv.io, .gpa = srv.gpa };
    defer box.deinit();
    const engine_id = srv.next_id.fetchAdd(1, .monotonic);
    srv.engine.submitTranscription(engine_id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => send(c, a, id, .{ .status = 503, .message = "the engine is busy; retry shortly" }),
        error.Closed => send(c, a, id, .{ .status = 500, .message = "the scheduler is closed" }),
        error.Unsupported => send(c, a, id, .{ .message = "this model does not support transcription" }),
    };
    // counted for /metrics as chat counts a reply (`Server.noteRequest`, run before `result` is freed), a client gone
    // or a failure included; the engine's `prefilled` stamps the first token, the finish the last
    var last: ?i96 = null;
    defer if (result.refused.len == 0) srv.noteRequest(result.prompt_tokens, box.cached orelse 0, result.tokens, box.stats.drafted, box.stats.accepted, box.stats.rounds, received, box.prefilled_ns, last, box.stats.prefill_seconds, if (box.stats.prefill_seconds != null) box.stats.queued_seconds else null);
    // the engine holds request and result until finished
    const settled = settle(srv, c, &box, engine_id);
    if (box.prefilled_ns != null) last = clock.nowNs(srv.io);
    if (!settled) {
        c.close = true;
        srv.metrics.disconnected(srv.io);
        log.line("transcription {s} client left after {d:.2}s", .{ id, clock.seconds(clock.nowNs(srv.io) - received) });
        return;
    }
    if (result.refused.len > 0) return send(c, a, id, .{ .message = result.refused, .param = if (result.param.len > 0) result.param else null });
    switch (box.reason) {
        .stop, .length => {},
        .cancelled => return send(c, a, id, .{ .status = 500, .message = "transcription was cancelled" }),
        .failed => return send(c, a, id, .{ .status = 500, .message = if (box.message.len > 0) box.message else "transcription failed" }),
        // since TensorFold 1.1.4 a speech reply's end (its silence guard), which no transcription gives
        .silence => return send(c, a, id, .{ .status = 500, .message = "the engine returned a speech-only finish reason" }),
    }
    reply(c, a, result) catch return;
    log.line("transcription {s} seconds_audio={d:.2} prompt={d} tokens={d} finish={s} seconds={d:.2}", .{ id, result.seconds, result.prompt_tokens, result.tokens, @tagName(box.reason), clock.seconds(clock.nowNs(srv.io) - received) });
}

/// vLLM's ``TranscriptionResponse``: the text and the audio's whole seconds, for ``json`` and ``text`` alike.
fn reply(c: *Conn, a: Allocator, result: api.TranscriptionResult) Allocator.Error!void {
    const o = try json.newObject(a);
    try o.put(a, "text", .{ .string = result.text });
    const usage = try json.newObject(a);
    try usage.put(a, "type", .{ .string = "duration" });
    try usage.put(a, "seconds", .{ .int = try std.fmt.allocPrint(a, "{d}", .{@as(u64, @intFromFloat(@ceil(@max(result.seconds, 0))))}) });
    try o.put(a, "usage", .{ .object = usage });
    sendJson(c, a, 200, .{ .object = o });
}

/// Waits for finished, cancelling if the client leaves; false: it left.
fn settle(srv: *Server, c: *Conn, box: *Mailbox, engine_id: api.Id) bool {
    while (true) {
        box.mutex.lockUncancelable(box.io);
        if (!box.finished) box.cond.waitTimeout(box.io, &box.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
        const finished = box.finished;
        box.mutex.unlock(box.io);
        if (finished) return true;
        if (c.peerGone()) {
            srv.engine.cancel(engine_id);
            box.mutex.lockUncancelable(box.io);
            defer box.mutex.unlock(box.io);
            while (!box.finished) box.cond.waitUncancelable(box.io, &box.mutex);
            return false;
        }
    }
}

fn send(c: *Conn, a: Allocator, id: []const u8, r: Refusal) void {
    var buf: [1400]u8 = undefined;
    log.line("{s}", .{log.refused(&buf, id, r.message)});
    const inner = json.newObject(a) catch return;
    inner.put(a, "message", .{ .string = r.message }) catch return;
    inner.put(a, "type", .{ .string = switch (r.status) {
        404 => "NotFoundError",
        500 => "InternalServerError",
        503 => "ServiceUnavailableError",
        else => "BadRequestError",
    } }) catch return;
    inner.put(a, "param", if (r.param) |p| .{ .string = p } else .null) catch return;
    inner.put(a, "code", .{ .int = std.fmt.allocPrint(a, "{d}", .{r.status}) catch return }) catch return;
    const o = json.newObject(a) catch return;
    o.put(a, "error", .{ .object = inner }) catch return;
    sendJson(c, a, r.status, .{ .object = o });
}

/// As FastAPI's ``JSONResponse`` writes it: UTF-8 as is, no spaces.
fn sendJson(c: *Conn, a: Allocator, code: u16, payload: json.Value) void {
    c.sendJson(code, json.stringify(a, payload, .{ .ascii = false, .compact = true }) catch return);
}

pub const ParseError = error{ NotMultipart, Malformed, OutOfMemory };

/// RFC 7578 form fields by name; a part's bytes are slices of ``body``, a later part of a name replaces an earlier.
pub fn parse(a: Allocator, content_type: []const u8, body: []const u8) ParseError!Form {
    _ = a;
    const boundary = param(content_type, "boundary") orelse return error.NotMultipart;
    if (!std.ascii.startsWithIgnoreCase(std.mem.trim(u8, content_type, " \t"), "multipart/form-data")) return error.NotMultipart;
    if (boundary.len == 0 or boundary.len > 70) return error.Malformed;
    var delim_buf: [76]u8 = undefined;
    const delim = std.fmt.bufPrint(&delim_buf, "--{s}", .{boundary}) catch return error.Malformed;
    var form: Form = .{};
    var at = std.mem.indexOf(u8, body, delim) orelse return error.Malformed;
    while (true) {
        at += delim.len;
        if (std.mem.startsWith(u8, body[at..], "--")) return form;
        if (!std.mem.startsWith(u8, body[at..], "\r\n")) return error.Malformed;
        at += 2;
        const head_end = std.mem.indexOfPos(u8, body, at, "\r\n\r\n") orelse return error.Malformed;
        const head = body[at..head_end];
        const start = head_end + 4;
        var close_buf: [78]u8 = undefined;
        const close = std.fmt.bufPrint(&close_buf, "\r\n{s}", .{delim}) catch unreachable;
        const end = std.mem.indexOfPos(u8, body, start, close) orelse return error.Malformed;
        const value = body[start..end];
        var name: ?[]const u8 = null;
        var file = false;
        var lines = std.mem.splitSequence(u8, head, "\r\n");
        while (lines.next()) |line| {
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            if (!std.ascii.eqlIgnoreCase(std.mem.trim(u8, line[0..colon], " \t"), "Content-Disposition")) continue;
            const v = line[colon + 1 ..];
            name = param(v, "name");
            file = param(v, "filename") != null;
        }
        if (name) |n| assign(&form, n, value, file);
        at = end + 2;
    }
}

fn assign(form: *Form, name: []const u8, value: []const u8, file: bool) void {
    if (std.mem.eql(u8, name, "file")) {
        form.file = value;
        return;
    }
    if (file) return;
    inline for (.{ "model", "prompt", "response_format", "temperature", "stream", "max_completion_tokens" }) |field|
        if (std.mem.eql(u8, name, field)) {
            @field(form, field) = value;
            return;
        };
    if (std.mem.eql(u8, name, "language")) form.language = value;
}

/// A header parameter's value (``key=value`` or ``key="value"``, the key case-blind), or null.
fn param(header: []const u8, key: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, header, ';');
    _ = it.next();
    while (it.next()) |raw| {
        const p = std.mem.trim(u8, raw, " \t");
        const eq = std.mem.indexOfScalar(u8, p, '=') orelse continue;
        if (!std.ascii.eqlIgnoreCase(std.mem.trim(u8, p[0..eq], " \t"), key)) continue;
        const v = std.mem.trim(u8, p[eq + 1 ..], " \t");
        if (v.len >= 2 and v[0] == '"' and v[v.len - 1] == '"') return v[1 .. v.len - 1];
        return v;
    }
    return null;
}
