//! zspark: audio input on chat (spec 150): audio_url parts (``messages.zig`` makes an input_audio part one) to the
//! engine's decoded samples (``Info.audio``), and the template's placeholder for each to the audio's rows, as
//! ``images.zig`` does for images. vLLM is the reference (``entrypoints/chat_utils.py`` ``parse_audio``,
//! ``parse_input_audio``; ``multimodal/media/audio.py``): a data URL is decoded here, and an HTTPS URL fetched with
//! --vision-urls as an image's is, no declared type read. Its refusals keep vLLM's words where it has them.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const images = @import("images.zig");
const media_fetch = @import("media_fetch.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub const Expanded = struct { ids: []const u32, history_len: usize, audios: []const api.AudioClip };

/// The audio parts' URLs of normalized `messages`, in the order the template meets them.
fn urls(cx: *Cx, messages: Value) errors.Refused![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    if (messages != .array) return out.items;
    for (messages.array) |m| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |p| {
            const kind = p.get("type") orelse continue;
            if (kind != .string or !std.mem.eql(u8, kind.string, "audio_url")) continue;
            try out.append(cx.a, p.get("audio_url").?.get("url").?.string);
        }
    }
    return out.items;
}

/// An audio part's bytes, at most `limit`: a data URL's, or with --vision-urls a public HTTPS URL's before `deadline`.
pub fn sourceBytes(cx: *Cx, fetch: ?images.Fetch, url: []const u8, limit: usize, deadline: std.Io.Timestamp) errors.Refused![]const u8 {
    if (std.ascii.startsWithIgnoreCase(url, "data:")) {
        // vLLM's words past its limit (`AudioMediaIO.load_base64`, VLLM_MAX_AUDIO_CLIP_FILESIZE_MB)
        const comma = std.mem.indexOfScalar(u8, url, ',') orelse return cx.refuse("invalid audio data URL");
        if (url.len - comma - 1 > 4 * ((limit + 2) / 3)) return cx.refuse("Maximum file size exceeded");
        return images.dataBytesOf(cx, url, limit, "audio");
    }
    if (!std.mem.startsWith(u8, url, "https://")) return cx.refuse("audio requires data URLs or public HTTPS URLs");
    const f = fetch orelse return cx.refuse("audio URLs are off on this server; send the audio as a data URL or input_audio, or start the server with --vision-urls");
    if (url.len > media_fetch.max_url_chars) return cx.refuse("audio URL is too long");
    var failure: media_fetch.Failure = .{};
    // `image_media` is the list whose declared type is not read (zspark's patch of `media_fetch.answer`): the
    // engine decodes the bytes, as vLLM's connector reads no type
    const bytes = f.get(f.ctx, cx.a, f.io, url, limit, media_fetch.within(f.io, deadline, media_fetch.timeout_s), &media_fetch.image_media, &failure) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Media => return cx.refuse(failure.text),
    };
    if (bytes.len == 0) return cx.refuse("audio is empty or exceeds the encoded byte limit");
    return bytes;
}

/// The prompt with each audio placeholder (`AudioOffer.audio_token`, one a part as the template writes it)
/// expanded to its audio's rows, and the audios; null for a request with no audio part.
pub fn expand(offer: ?api.AudioOffer, fetch: ?images.Fetch, cx: *Cx, messages: Value, ids: []const u32, history_len: usize) errors.Refused!?Expanded {
    const o = offer orelse return null;
    const ps = try urls(cx, messages);
    if (ps.len == 0) return null;
    var marks: usize = 0;
    for (ids) |t| marks += @intFromBool(t == o.audio_token);
    if (marks != ps.len) return cx.fail(.server, "the chat template wrote {d} audio placeholders for {d} audio parts", .{ marks, ps.len });
    // vLLM's words (`multimodal/processing/context.py`) past `--limit-mm-per-prompt`
    if (ps.len > o.max_audios) return cx.refuse(try std.fmt.allocPrint(cx.a, "At most {d} audio(s) may be provided in one prompt.", .{o.max_audios}));
    const prepared = try cx.a.alloc(api.PreparedAudio, ps.len);
    var seconds: f64 = 0;
    var rows: usize = 0;
    const deadline: std.Io.Timestamp = if (fetch) |f| std.Io.Clock.awake.now(f.io).addDuration(.fromSeconds(media_fetch.total_s)) else .{ .nanoseconds = 0 };
    for (ps, prepared) |url, *p| {
        const bytes = try sourceBytes(cx, fetch, url, o.max_bytes, deadline);
        p.* = o.prepare(o.ctx, cx.a, bytes) catch |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            if (o.explain) |words| if (words(e)) |w| return cx.refuse(w);
            return cx.refuse("Invalid or unsupported audio file.");
        };
        seconds += p.seconds;
        rows += p.tokens;
    }
    if (seconds > o.max_seconds) return cx.refuse(try std.fmt.allocPrint(cx.a, "Audio exceeds maximum allowed duration: the request's audio is {d:.1} s, at most {d:.0} s are taken", .{ seconds, o.max_seconds }));
    if (rows > o.max_tokens) return cx.refuse(try std.fmt.allocPrint(cx.a, "the request's audio takes {d} rows, at most {d} are taken", .{ rows, o.max_tokens }));
    var out: std.ArrayList(u32) = .empty;
    var clips: std.ArrayList(api.AudioClip) = .empty;
    var hist = history_len;
    var k: usize = 0;
    for (ids, 0..) |t, src| {
        if (t != o.audio_token) {
            try out.append(cx.a, t);
            continue;
        }
        const p = prepared[k];
        k += 1;
        try clips.append(cx.a, .{ .samples = p.samples, .at = @intCast(out.items.len), .tokens = p.tokens });
        try out.appendNTimes(cx.a, t, p.tokens);
        if (src < history_len) hist += p.tokens - 1;
    }
    return .{ .ids = out.items, .history_len = hist, .audios = clips.items };
}
