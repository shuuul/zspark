//! Qwen3-ASR's text around a clip (spec 138 WS-05): the prompt a transcription asks with, the language it forces,
//! and the transcript read out of the reply. Host only. The behaviour reference is the `qwen-asr` 0.0.6 package
//! (spec 138's Decision): `Qwen3ASRModel._build_text_prompt` over the checkpoint's chat template, and
//! `utils.parse_asr_output` with `detect_and_fix_repetitions` (Apache-2.0, read 2026-10-25). What the route
//! takes from vLLM's (`model_executor/models/qwen3_asr.py`, `whisper_utils.py`, vLLM 3313825):
//!
//! - `language` as an ISO 639-1 code (what aimd and OpenAI's clients send), named by vLLM's table; the package's
//!   own names (`English`, any case) are taken too, and `yue` and `fil`, the model's Cantonese and Filipino,
//!   which vLLM's table lacks. Anything else is vLLM's 400, in its words.
//! - the request's `prompt` stripped of ChatML-like tokens and `<asr_text>` to a fixpoint
//!   (`_sanitize_transcription_user_text`; `audio/text.zig` since spec 151, shared with nemotron's prompt),
//!   so a client's context cannot close the system turn.
//!
//! Departures from vLLM, both the package's: the system turn is kept when the context is empty (vLLM drops it),
//! and the transcript is stripped and its repetitions folded (vLLM keeps the text after the last `<asr_text>`).

const std = @import("std");
const shared = @import("../../audio/text.zig");
const Allocator = std.mem.Allocator;

pub const asr_tag = "<asr_text>";
pub const audio_pad = "<|audio_pad|>";

/// `whisper_utils.py`'s `ISO639_1_SUPPORTED_LANGS` (vLLM's Qwen3-ASR `supported_languages`), then the model's two
/// languages it lacks.
pub const codes = [_][2][]const u8{
    .{ "af", "Afrikaans" },  .{ "ar", "Arabic" },     .{ "hy", "Armenian" },   .{ "az", "Azerbaijani" }, .{ "be", "Belarusian" },
    .{ "bs", "Bosnian" },    .{ "bg", "Bulgarian" },  .{ "ca", "Catalan" },    .{ "zh", "Chinese" },     .{ "hr", "Croatian" },
    .{ "cs", "Czech" },      .{ "da", "Danish" },     .{ "nl", "Dutch" },      .{ "en", "English" },     .{ "et", "Estonian" },
    .{ "fi", "Finnish" },    .{ "fr", "French" },     .{ "gl", "Galician" },   .{ "de", "German" },      .{ "el", "Greek" },
    .{ "he", "Hebrew" },     .{ "hi", "Hindi" },      .{ "hu", "Hungarian" },  .{ "is", "Icelandic" },   .{ "id", "Indonesian" },
    .{ "it", "Italian" },    .{ "ja", "Japanese" },   .{ "kn", "Kannada" },    .{ "kk", "Kazakh" },      .{ "ko", "Korean" },
    .{ "lv", "Latvian" },    .{ "lt", "Lithuanian" }, .{ "mk", "Macedonian" }, .{ "ms", "Malay" },       .{ "mr", "Marathi" },
    .{ "mi", "Maori" },      .{ "ne", "Nepali" },     .{ "no", "Norwegian" },  .{ "fa", "Persian" },     .{ "pl", "Polish" },
    .{ "pt", "Portuguese" }, .{ "ro", "Romanian" },   .{ "ru", "Russian" },    .{ "sr", "Serbian" },     .{ "sk", "Slovak" },
    .{ "sl", "Slovenian" },  .{ "es", "Spanish" },    .{ "sw", "Swahili" },    .{ "sv", "Swedish" },     .{ "tl", "Tagalog" },
    .{ "ta", "Tamil" },      .{ "th", "Thai" },       .{ "tr", "Turkish" },    .{ "uk", "Ukrainian" },   .{ "ur", "Urdu" },
    .{ "vi", "Vietnamese" }, .{ "cy", "Welsh" },      .{ "yue", "Cantonese" }, .{ "fil", "Filipino" },
};

/// The package's `SUPPORTED_LANGUAGES`.
pub const names = [_][]const u8{
    "Chinese", "English", "Cantonese", "Arabic",     "German",   "French",  "Spanish", "Portuguese", "Indonesian", "Italian",
    "Korean",  "Russian", "Thai",      "Vietnamese", "Japanese", "Turkish", "Hindi",   "Malay",      "Dutch",      "Swedish",
    "Danish",  "Finnish", "Polish",    "Czech",      "Filipino", "Persian", "Greek",   "Romanian",   "Hungarian",  "Macedonian",
};

/// The language name a request's `language` forces, or null where it forces none (absent or blank); a value
/// neither a code nor a name is `error.Unsupported` (`refusal` words it).
pub fn language(given: ?[]const u8) error{Unsupported}!?[]const u8 {
    const raw = given orelse return null;
    const s = std.mem.trim(u8, raw, " \t\r\n");
    if (s.len == 0) return null;
    for (codes) |pair| if (std.mem.eql(u8, pair[0], s)) return pair[1];
    // `normalize_language_name`: the first letter upper, the rest lower
    var buf: [32]u8 = undefined;
    if (s.len > buf.len) return error.Unsupported;
    for (s, 0..) |c, i| buf[i] = if (i == 0) std.ascii.toUpper(c) else std.ascii.toLower(c);
    for (names) |name| if (std.mem.eql(u8, name, buf[0..s.len])) return name;
    return error.Unsupported;
}

/// vLLM's words for a language it does not take (`SupportsTranscription.validate_language`), from `a`.
pub fn refusal(a: Allocator, given: []const u8) Allocator.Error![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    errdefer out.deinit();
    const w = &out.writer;
    w.writeAll("Unsupported language: '") catch return error.OutOfMemory;
    w.writeAll(given) catch return error.OutOfMemory;
    w.writeAll("'.  Must be one of [") catch return error.OutOfMemory;
    for (codes[0 .. codes.len - 2], 0..) |pair, i| w.print("{s}'{s}'", .{ if (i > 0) ", " else "", pair[0] }) catch return error.OutOfMemory;
    w.writeAll("].") catch return error.OutOfMemory;
    return out.toOwnedSlice();
}

/// `_sanitize_transcription_user_text`: `<|...|>` (no `|` inside) and `<asr_text>` removed until none is left.
pub fn sanitize(a: Allocator, text: []const u8) Allocator.Error![]u8 {
    return shared.stripTags(a, text, &.{asr_tag});
}

/// The package's prompt for one clip with one `<|audio_pad|>`, which the caller expands to the clip's tokens: the
/// chat template's system turn with `context` (always written, empty or not), the user's turn of the audio, the
/// assistant's header, and `language X<asr_text>` where a language is forced.
pub fn prompt(a: Allocator, context: []const u8, forced: ?[]const u8) Allocator.Error![]u8 {
    return std.fmt.allocPrint(a, "<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n<|audio_start|>" ++ audio_pad ++ "<|audio_end|><|im_end|>\n<|im_start|>assistant\n{s}{s}{s}", .{
        context,
        if (forced != null) "language " else "",
        forced orelse "",
        if (forced != null) asr_tag else "",
    });
}

/// What `parse_asr_output` gives: the language read (empty: none) and the transcript.
pub const Parsed = struct { language: []const u8, text: []const u8 };

/// `parse_asr_output(raw, forced)` over the reply's text (special tokens skipped, `<asr_text>` kept, as the
/// package decodes): stripped, repetitions folded, then the text after the first `<asr_text>` and the language
/// before it; under a forced language the whole text. Slices of `raw` or of `a`'s memory.
pub fn parse(a: Allocator, raw: []const u8, forced: ?[]const u8) Allocator.Error!Parsed {
    const s0 = strip(raw);
    if (s0.len == 0) return .{ .language = "", .text = "" };
    const s = try fixRepetitions(a, s0, 20);
    if (forced) |name| return .{ .language = name, .text = s };
    const at = std.mem.indexOf(u8, s, asr_tag) orelse return .{ .language = "", .text = strip(s) };
    const meta = s[0..at];
    const body = strip(s[at + asr_tag.len ..]);
    if (containsLower(meta, "language none")) return .{ .language = "", .text = body };
    var lines = std.mem.tokenizeAny(u8, meta, "\r\n");
    while (lines.next()) |raw_line| {
        const line = strip(raw_line);
        if (line.len == 0) continue;
        if (line.len >= 9 and std.ascii.eqlIgnoreCase(line[0..9], "language ")) {
            const value = strip(line[9..]);
            if (value.len == 0) break;
            const name = try a.alloc(u8, value.len);
            for (value, 0..) |c, i| name[i] = if (i == 0) std.ascii.toUpper(c) else std.ascii.toLower(c);
            return .{ .language = name, .text = body };
        }
    }
    return .{ .language = "", .text = body };
}

/// The seam between a long clip's pieces' transcripts (spec 142 WS-06), vLLM's `asr_inter_chunk_separator` over
/// `SupportsTranscription.no_space_languages` ({"ja", "zh"}): nothing where the request forces Chinese or
/// Japanese, one space otherwise, a request that names no language included (vLLM reads the request's language,
/// not the reply's, and Qwen3-ASR detects none apart).
pub fn separator(forced: ?[]const u8) []const u8 {
    const name = forced orelse return " ";
    return if (std.mem.eql(u8, name, "Chinese") or std.mem.eql(u8, name, "Japanese")) "" else " ";
}

/// The pieces' transcripts in order, `separator(forced)` between each two, an empty one's included (vLLM's
/// `separator.join(text_parts)`).
pub fn join(a: Allocator, texts: []const []const u8, forced: ?[]const u8) Allocator.Error![]u8 {
    return std.mem.join(a, separator(forced), texts);
}

/// Whether `needle` (lower case) is in `haystack` lowered (ASCII).
fn containsLower(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

/// Python's `str.strip()` for the whitespace a transcript may hold: ASCII's and 0x1c to 0x1f, NBSP and U+3000.
pub fn strip(s: []const u8) []const u8 {
    return shared.strip(s);
}

/// `detect_and_fix_repetitions(text, threshold)` over code points: a character repeated more than `threshold`
/// times kept once, then a pattern of 1 to 20 characters repeated `threshold` times or more kept once (the
/// search goes on after it, the first such pattern from each place).
pub fn fixRepetitions(a: Allocator, text: []const u8, threshold: usize) Allocator.Error![]const u8 {
    const points = try codePoints(a, text);
    defer a.free(points);
    var chars: std.ArrayList(u21) = .empty;
    defer chars.deinit(a);
    var i: usize = 0;
    while (i < points.len) {
        var count: usize = 1;
        while (i + count < points.len and points[i + count] == points[i]) count += 1;
        try chars.appendSlice(a, if (count > threshold) points[i..][0..1] else points[i..][0..count]);
        i += count;
    }
    var out: std.ArrayList(u21) = .empty;
    defer out.deinit(a);
    try patterns(a, chars.items, threshold, 20, &out);
    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(a);
    for (out.items) |c| {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(c, &buf) catch unreachable;
        try bytes.appendSlice(a, buf[0..n]);
    }
    return bytes.toOwnedSlice(a);
}

/// `fix_pattern_repeats(s, thresh, max_len)`, its recursion on what follows a folded pattern.
fn patterns(a: Allocator, s: []const u21, thresh: usize, max_len: usize, out: *std.ArrayList(u21)) Allocator.Error!void {
    const n = s.len;
    const least = thresh * 2;
    if (n < least) return out.appendSlice(a, s);
    var i: usize = 0;
    while (i <= n - least) {
        for (1..max_len + 1) |k| {
            if (i + k * thresh > n) break;
            const pattern = s[i..][0..k];
            var valid = true;
            for (1..thresh) |rep| {
                const from = i + rep * k;
                if (!std.mem.eql(u21, s[from..@min(n, from + k)], pattern)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;
            var end = i + thresh * k;
            while (end + k <= n and std.mem.eql(u21, s[end..][0..k], pattern)) end += k;
            try out.appendSlice(a, pattern);
            return patterns(a, s[end..], thresh, max_len, out);
        }
        try out.append(a, s[i]);
        i += 1;
    }
    try out.appendSlice(a, s[i..]);
}

fn codePoints(a: Allocator, text: []const u8) Allocator.Error![]u21 {
    var out: std.ArrayList(u21) = .empty;
    errdefer out.deinit(a);
    var it = (std.unicode.Utf8View.init(text) catch {
        // a reply's text is valid UTF-8 (decoding replaces what is not); bytes as their own points otherwise
        for (text) |b| try out.append(a, b);
        return out.toOwnedSlice(a);
    }).iterator();
    while (it.nextCodepoint()) |c| try out.append(a, c);
    return out.toOwnedSlice(a);
}
