//! Nemotron-3.5-Transcribe's text around a clip (spec 151 WS-08), host only: the prompt a transcription or a
//! chat asks with, the audio rows a clip's samples take, and the reply read as vLLM's `nemotron_v3` reasoning
//! parser reads it. The behaviour reference is NeMo Speech's vLLM plugin (`nemo/collections/speechlm2/vllm/salm/`
//! `audio.py`, the `nemotron3p5` prompt format of `collections/common/prompts/`) and the checkpoint's
//! `chat_template.jinja`: the prompt's ChatML text is what that template renders for a system-less request —
//! an empty system turn (the template defines `system_message` as `""` and always writes its turn), the
//! user's turn of the instruction and one `<|audio|>` a clip (the template's `render_content`: text parts
//! joined, a locator a part after a space), and the assistant's opener closed by `<think></think>`
//! (`enable_thinking` false, the transcription route's and the card's chat default for it).
//!
//! The plugin's numbers (`audio.py`): a clip's prompt rows are `_estimate_audio_tokens`, `mel.zig`'s
//! `rows`/`chunkSpans` — the featurizer's frames, feature-stacked 8x. This checkpoint mounts a
//! ParallelExpertEncoder (`pe_encoder_config`, `config.pe_encoder`), which owns the encoder's windowing, so
//! the plugin's estimate is one pass over the whole clip, not the 30 s `encoder_chunk_size_seconds` split a
//! plain encoder takes (`NeMoSpeechLMProcessingInfo`). A request's audio is at most 3,600 s
//! (`_DUMMY_AUDIO_MAX_DURATION_S`, the card's one hour; the plugin's own limit-mm hint). The locator is
//! `config.json`'s `audio_locator_tag` `<|audio|>` (added token 131,072), one a clip, each expanded to the
//! clip's rows (`PromptReplacement`).
//!
//! Departures from the plugin, all the route's own (the plugin serves chat only; `/v1/audio/transcriptions`
//! has no upstream reference here, so it keeps vLLM's transcription-route shape as `qwen3_asr/` does):
//! `language` is taken as an ISO 639-1 code or the card's English name, each from the card's 40-language
//! matrix, anything else refused in vLLM's words; a given `language` fills the instruction's `Language:`
//! field, a request's `prompt` replaces the instruction whole (sanitized of ChatML-like tags so it cannot
//! close a turn); the reply's text is the reasoning parser's content, stripped.

const std = @import("std");
const shared = @import("../../audio/text.zig");
const mel_file = @import("mel.zig");
const config_file = @import("config.zig");
const Allocator = std.mem.Allocator;

/// `config.json`'s `audio_locator_tag`; the checkpoint's tokenizer carries it as added token 131,072.
pub const audio_locator = "<|audio|>";
/// The longest clip taken: the plugin's `_DUMMY_AUDIO_MAX_DURATION_S`, the card's one hour.
pub const max_seconds = 3600;
/// `encoder_chunk_size_seconds`: the perception encoder's chunk span.
pub const chunk_seconds = 30.0;

/// A clip's prompt rows (`audio.py`'s `_estimate_audio_tokens`): `mel.zig`'s count — the featurizer's
/// `Params` from `config.perception` (`perception.zig`'s), stacking `config.perception.subsampling`. The
/// plugin runs one pass over the whole clip where the checkpoint mounts a ParallelExpertEncoder
/// (`config.pe_encoder`; PEE owns the encoder's windowing, `NeMoSpeechLMProcessingInfo`), and this
/// checkpoint does; without one the count is `mel.rows` over `config.chunk_seconds` spans, a tail under
/// the least chunk of two frames folded into the previous span.
pub fn clipRows(a: Allocator, c: *const config_file.Config, samples: usize) Allocator.Error!usize {
    const e = c.perception;
    const p: mel_file.Params = .{
        .n_fft = e.n_fft,
        .window = e.windowFrames(),
        .hop = @intFromFloat(@round(e.window_stride * @as(f64, @floatFromInt(e.sample_rate)))),
        .features = e.features,
        .rate = e.sample_rate,
    };
    if (c.pe_encoder) return @max(1, (p.frames(samples) + e.subsampling - 1) / e.subsampling);
    return mel_file.rows(a, p, c.chunk_seconds, e.subsampling, samples);
}

/// The card's verbatim transcription instruction; `Language:` takes the forced name or "as spoken".
pub fn instruction(a: Allocator, forced: ?[]const u8) Allocator.Error![]u8 {
    return std.fmt.allocPrint(a, "Audio transcription. Language: {s}. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.", .{forced orelse "as spoken"});
}

/// The card's 40 languages, each its ISO 639-1 code and the name the matrix and the instruction carry.
pub const codes = [_][2][]const u8{
    .{ "ar", "Arabic" },           .{ "bn", "Bengali" },    .{ "bg", "Bulgarian" },  .{ "hr", "Croatian" },
    .{ "cs", "Czech" },            .{ "da", "Danish" },     .{ "nl", "Dutch" },      .{ "en", "English" },
    .{ "et", "Estonian" },         .{ "fi", "Finnish" },    .{ "fr", "French" },     .{ "de", "German" },
    .{ "el", "Greek" },            .{ "he", "Hebrew" },     .{ "hi", "Hindi" },      .{ "hu", "Hungarian" },
    .{ "id", "Indonesian" },       .{ "it", "Italian" },    .{ "ja", "Japanese" },   .{ "ko", "Korean" },
    .{ "lv", "Latvian" },          .{ "lt", "Lithuanian" }, .{ "mt", "Maltese" },    .{ "mr", "Marathi" },
    .{ "no", "Norwegian" },        .{ "pl", "Polish" },     .{ "pt", "Portuguese" }, .{ "ro", "Romanian" },
    .{ "ru", "Russian" },          .{ "sk", "Slovak" },     .{ "sl", "Slovenian" },  .{ "es", "Spanish" },
    .{ "sv", "Swedish" },          .{ "tl", "Tagalog" },    .{ "ta", "Tamil" },      .{ "te", "Telugu" },
    .{ "th", "Thai" },             .{ "tr", "Turkish" },    .{ "uk", "Ukrainian" },  .{ "vi", "Vietnamese" },
    .{ "zh", "Chinese Mandarin" },
};

/// The language name a request's `language` forces, or null where it forces none (absent or blank); a value
/// neither a code nor a name is `error.Unsupported` (`refusal` words it).
pub fn language(given: ?[]const u8) error{Unsupported}!?[]const u8 {
    const raw = given orelse return null;
    const s = shared.strip(raw);
    if (s.len == 0) return null;
    for (codes) |pair| {
        if (std.mem.eql(u8, pair[0], s)) return pair[1];
        if (std.ascii.eqlIgnoreCase(pair[1], s)) return pair[1];
    }
    return error.Unsupported;
}

/// vLLM's words for a language it does not take (`SupportsTranscription.validate_language`).
pub fn refusal(a: Allocator, given: []const u8) Allocator.Error![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    errdefer out.deinit();
    const w = &out.writer;
    w.writeAll("Unsupported language: '") catch return error.OutOfMemory;
    w.writeAll(given) catch return error.OutOfMemory;
    w.writeAll("'.  Must be one of [") catch return error.OutOfMemory;
    for (codes, 0..) |pair, i| w.print("{s}'{s}'", .{ if (i > 0) ", " else "", pair[0] }) catch return error.OutOfMemory;
    w.writeAll("].") catch return error.OutOfMemory;
    return out.toOwnedSlice();
}

/// A `prompt` or `language`-free field's text made safe to write into a turn: `<|...|>` and the think and
/// tool tags the template would read, removed to a fixpoint (`audio/text.zig`'s `stripTags`).
pub fn sanitize(a: Allocator, text: []const u8) Allocator.Error![]u8 {
    return shared.stripTags(a, text, &.{ "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>" });
}

/// The transcription route's prompt for one clip with one `<|audio|>` (the caller expands it to the clip's
/// rows): the template's render of a system-less request — its empty system turn, the user's turn of the
/// instruction and the locator after a space, and the assistant's opener with thinking off (`<think></think>`).
/// `prompt_field` is the request's `prompt`, already sanitized; empty gives the card's verbatim instruction,
/// `forced` its `Language:`.
pub fn transcription(a: Allocator, prompt_field: []const u8, forced: ?[]const u8) Allocator.Error![]u8 {
    const text = if (shared.strip(prompt_field).len > 0) prompt_field else try instruction(a, forced);
    return std.fmt.allocPrint(a, "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{s} {s}<|im_end|>\n<|im_start|>assistant\n<think></think>", .{ text, audio_locator });
}

/// The ids' one `locator` expanded to `rows` placeholder rows (vLLM's `PromptReplacement`: the placeholder's
/// one id repeated per encoder token): the prompt's new ids and the audio rows' positions. `at` is the
/// locator's index in `ids`.
pub fn expand(a: Allocator, ids: []const u32, at: usize, rows_n: usize, locator: u32) Allocator.Error!struct { prompt: []i32, rows: []u32 } {
    const prompt = try a.alloc(i32, ids.len - 1 + rows_n);
    const rows_out = try a.alloc(u32, rows_n);
    for (ids[0..at], 0..) |id, i| prompt[i] = @intCast(id);
    for (rows_out, 0..) |*row, i| {
        row.* = @intCast(at + i);
        prompt[at + i] = @intCast(locator);
    }
    for (ids[at + 1 ..], 0..) |id, i| prompt[at + rows_n + i] = @intCast(id);
    return .{ .prompt = prompt, .rows = rows_out };
}

/// What `nemotron_v3`'s `extract_reasoning` gives: the reasoning between `<think>` and `</think>` and the
/// content after it; `force` (enable_thinking false or force_nonempty_content) with an empty content swaps
/// them. Slices of `raw`.
pub const Split = struct { reasoning: []const u8, content: []const u8 };

pub fn split(raw: []const u8, force: bool) Split {
    const think_end = "</think>";
    var content = raw;
    var reasoning: []const u8 = "";
    if (std.mem.indexOf(u8, raw, think_end)) |at| {
        reasoning = raw[0..at];
        if (std.mem.indexOf(u8, reasoning, "<think>")) |open| reasoning = reasoning[open + 7 ..];
        content = raw[at + think_end.len ..];
    }
    if (force and shared.strip(content).len == 0) {
        return .{ .reasoning = content, .content = reasoning };
    }
    return .{ .reasoning = reasoning, .content = content };
}

/// The reply's transcript text: the reasoning split's content, stripped (`strip`).
pub fn transcript(raw: []const u8) []const u8 {
    return shared.strip(split(raw, true).content);
}
