//! Qwen3-ASR's engine (`served.zig`) behind TensorFold's host on the loopback, on each checkpoint
//! `forward_test.zig` loads (1.7B's and 0.6B's, its models shared): a clip posted to `/v1/audio/transcriptions` and
//! sent as a chat's `input_audio` and `audio_url` parts (spec 150). The chat's prompt is the checkpoint's template
//! as the host renders it, which writes the transcription route's prompt with an empty context; its reply is the
//! model's text as it is, whose transcript (`transcript.parse`) is the route's.

const std = @import("std");
const testing = std.testing;
const server = @import("server");
const cuda = @import("../../../src/cuda/cuda.zig");
const audio = @import("../../../src/audio/audio.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const text_tokenizer = @import("../../../src/text/tokenizer.zig");
const forward = @import("../../../src/models/qwen3_asr/forward.zig");
const model_file = @import("../../../src/models/qwen3_asr/model.zig");
const served = @import("../../../src/models/qwen3_asr/served.zig");
const tower_file = @import("../../../src/models/qwen3_asr/tower.zig");
const transcript = @import("../../../src/models/qwen3_asr/transcript.zig");
const fixture = @import("../../server/served_fixture.zig");
const loads = @import("forward_test.zig");

test "served: a chat's input_audio and audio_url parts, its prompt the template's, give the transcription route's transcript of the clip" {
    try loads.eachSubject(chatAsTranscription);
}

fn chatAsTranscription(s: *loads.Subject, m: model_file.Model) !void {
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var problem: []const u8 = "";
    const text = server.hf_text.HfText.load(gpa, io, s.path(), null, a, &problem) catch |err| {
        std.debug.print("{s}: {s}\n", .{ s.path(), problem });
        return err;
    };
    defer text.deinit();
    var diagnostic: text_tokenizer.Diagnostic = .{};
    var tok = try text_tokenizer.Tokenizer.over(gpa, &text.tok, &diagnostic);
    defer tok.deinit(gpa);
    var checkpoint: safetensors.Checkpoint = try .open(gpa, io, .cwd(), s.path());
    defer checkpoint.close();
    var tower = try s.tower(gpa, a, &checkpoint, 120);
    defer tower.deinit();
    const tables = try a.create(audio.mel.Tables);
    tables.init();
    const context = 8192;
    var state: forward.State = try .init(gpa, m.config, context);
    defer state.deinit();
    var buffers: forward.Buffers = try .init(m.config, 512, 1, context);
    defer buffers.deinit();
    var features: cuda.Buffer = try .init(.bf16, &.{ tower.max_chunks * tower_file.chunk_tokens, tower.shape.out_dim });
    defer features.deinit();
    const model = m;
    var asr: served.Asr = .{ .gpa = gpa, .io = io, .model = &model, .tower = &tower, .tables = tables, .tok = &tok, .state = &state, .buffers = &buffers, .features = features, .pad = tok.addedId(transcript.audio_pad).?, .context = context };
    try asr.start();
    defer asr.close();

    const srv: *server.Server = try .init(gpa, io, asr.engine(), text.text(), .{ .served_name = "asr", .model_ids = &.{"asr"}, .enable_thinking = false, .keep_warm_s = 0 }, null);
    defer srv.deinit();
    var listener: server.listener.Listener = try .open(.{ .ip4 = .loopback(0) });
    defer listener.close();
    var stop: std.atomic.Value(bool) = .init(false);
    const accepting = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, listener, &stop });
    defer {
        stop.store(true, .release);
        accepting.join();
        while (srv.open_connections.load(.acquire) != 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    const port = listener.port();

    const file = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/wer/clips/zh_m00.flac", .{s.fixtures()}), a, .limited(16 << 20));
    const form = try std.fmt.allocPrint(a, "--b0undary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.flac\"\r\nContent-Type: audio/flac\r\n\r\n{s}\r\n--b0undary--\r\n", .{file});
    const asked = try std.fmt.allocPrint(a, "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: multipart/form-data; boundary=b0undary\r\nContent-Length: {d}\r\n\r\n{s}", .{ form.len, form });
    const route = try fixture.jsonOf(a, try fixture.bodyOf(try fixture.heard(a, try fixture.sent(port, asked)), "HTTP/1.1 200 OK"));
    const want = route.get("text").?.string;

    // the route's prompt with an empty context: the template's, the clip's rows where its one placeholder is
    var ids: std.ArrayList(u32) = .empty;
    try tok.encode(a, try transcript.prompt(a, "", null), &ids);
    const clip = try audio.clip(a, file, served.max_seconds);
    const length = ids.items.len - 1 + audio.mel.tokens(audio.mel.frames(clip.samples.len));

    const data = try a.alloc(u8, std.base64.standard.Encoder.calcSize(file.len));
    _ = std.base64.standard.Encoder.encode(data, file);
    var said: ?[]const u8 = null;
    inline for ([_][]const u8{ "{{\"type\":\"input_audio\",\"input_audio\":{{\"data\":\"{s}\",\"format\":\"flac\"}}}}", "{{\"type\":\"audio_url\",\"audio_url\":{{\"url\":\"data:audio/flac;base64,{s}\"}}}}" }) |part_format| {
        const part = try std.fmt.allocPrint(a, part_format, .{data});
        const body = try std.mem.concat(a, u8, &.{ "{\"model\":\"asr\",\"max_tokens\":1024,\"messages\":[{\"role\":\"user\",\"content\":[", part, "]}]}" });
        const reply = try fixture.jsonOf(a, try fixture.bodyOf(try fixture.heard(a, try fixture.sent(port, try fixture.posted(a, "/v1/chat/completions", body))), "HTTP/1.1 200 OK"));
        const content = reply.get("choices").?.array[0].get("message").?.get("content").?.string;
        std.debug.print("qwen3_asr {s} chat: {s}\n", .{ s.name, content });
        try testing.expectEqualStrings(try std.fmt.allocPrint(a, "{d}", .{length}), reply.get("usage").?.get("prompt_tokens").?.int);
        try testing.expect(std.mem.indexOf(u8, content, "<asr_text>") != null);
        try testing.expectEqualStrings(want, (try transcript.parse(a, content, null)).text);
        if (said) |first| try testing.expectEqualStrings(first, content);
        said = content;
    }
}
