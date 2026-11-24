//! `prompt.zig`'s tests, host only (`tools/check.sh host nemotron prompt`, or `test-nemotron`): the plugin's
//! audio-row arithmetic (`audio.py`'s `_estimate_audio_tokens`), the transcription route's prompt text as the
//! checkpoint's `chat_template.jinja` renders a system-less request, the language table, the sanitizing and the
//! `nemotron_v3` reasoning split. `$ZSPARK_NEMOTRON_HF` (the checkpoint's public files, holding its
//! `tokenizer.json`) runs the prompt's text through the tokenizer and checks the locator's id; unset, the ids
//! parts are skipped. `$ZSPARK_REFERENCE_NEMOTRON` (the reference environment's exports: `manifest.json`,
//! `prompt.chat.json`) holds the counts and the render to NeMo's; unset, the hand-rendered expectations stand.

const std = @import("std");
const testing = std.testing;

const text_tokenizer = @import("../../../src/text/tokenizer.zig");
const config_file = @import("../../../src/models/nemotron/config.zig");
const prompt_file = @import("../../../src/models/nemotron/prompt.zig");
const fixture = @import("fixture.zig");

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// The template's render of `[{"role":"user","content":[{"type":"text","text":TEXT},{"type":"audio_url",...}]}]`
/// with `add_generation_prompt` and `enable_thinking` false, derived by hand from `chat_template.jinja`
/// (its `render_content`: text then the one locator after a space; the empty `system_message` still writes
/// its turn). A ReferenceEnv fixture replaces this when it lands.
const card_render =
    "<|im_start|>system\n<|im_end|>\n" ++
    "<|im_start|>user\nTranscribe the following audio. <|audio|><|im_end|>\n" ++
    "<|im_start|>assistant\n<think></think>";

/// A config like the fixture's with the checkpoint's preprocessor and stacking, and optionally a mounted
/// ParallelExpertEncoder.
fn realConfig(a: std.mem.Allocator, pe_encoder: bool) !config_file.Config {
    var text: []const u8 = fixture.config_text;
    var swaps: std.ArrayList([2][]const u8) = .empty;
    try swaps.appendSlice(a, &.{
        .{ "\"window_size\":0.001", "\"window_size\":0.025" },
        .{ "\"features\":32,\"n_fft\":64", "\"features\":128,\"n_fft\":512" },
        .{ "\"feat_in\":32", "\"feat_in\":128" },
        .{ "\"subsampling_factor\":2", "\"subsampling_factor\":8" },
    });
    if (pe_encoder) try swaps.append(a, .{ "\"encoder_chunk_size_seconds\":30.0,", "\"encoder_chunk_size_seconds\":30.0,\"pe_encoder_config\":{\"target\":\"x\"}," });
    for (swaps.items) |swap| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, text, swap[0]));
        text = try std.mem.replaceOwned(u8, a, text, swap[0], swap[1]);
    }
    var diagnostic: config_file.Diagnostic = .{};
    return config_file.Config.parse(a, text, fixture.backbone_text, fixture.quant_text, &diagnostic);
}

test "prompt: a clip's audio rows are mel.zig's estimate, single-pass under the PEE" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // without a ParallelExpertEncoder: the 30 s chunk split (mel_test's rows cases carry the edges)
    const chunked = try realConfig(a, false);
    try testing.expect(!chunked.pe_encoder);
    try testing.expectEqual(@as(usize, 7), try prompt_file.clipRows(a, &chunked, 8000));
    try testing.expectEqual(@as(usize, 375), try prompt_file.clipRows(a, &chunked, 480_000));
    try testing.expectEqual(@as(usize, 375 + 375 + 13), try prompt_file.clipRows(a, &chunked, 976_000));
    // with one, as this checkpoint mounts: one pass over the whole clip
    const pee = try realConfig(a, true);
    try testing.expect(pee.pe_encoder);
    try testing.expectEqual(@as(usize, 7), try prompt_file.clipRows(a, &pee, 8000));
    try testing.expectEqual(@as(usize, 45_000), try prompt_file.clipRows(a, &pee, 57_600_000));
    // the reference fixture's per-clip placeholder counts (single pass)
    if (env("ZSPARK_REFERENCE_NEMOTRON")) |dir_path| {
        var dir = try std.Io.Dir.cwd().openDir(testing.io, dir_path, .{});
        defer dir.close(testing.io);
        const raw = try dir.readFileAlloc(testing.io, "manifest.json", a, .limited(1 << 22));
        const manifest = try std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{});
        var it = manifest.object.get("clips").?.object.iterator();
        while (it.next()) |clip| {
            const samples: usize = @intCast(clip.value_ptr.object.get("samples").?.integer);
            const want: usize = @intCast(clip.value_ptr.object.get("audio_placeholders").?.integer);
            errdefer std.debug.print("clip {s}: {d} samples\n", .{ clip.key_ptr.*, samples });
            try testing.expectEqual(want, try prompt_file.clipRows(a, &pee, samples));
        }
    }
}

test "prompt: the transcription route's prompt is the template's system-less render, thinking off" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const verbatim = "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.";
    try testing.expectEqualStrings(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" ++ verbatim ++ " <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>",
        try prompt_file.transcription(a, "", null),
    );
    // a language fills the instruction's Language:
    try testing.expectEqualStrings(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nAudio transcription. Language: Japanese. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only. <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>",
        try prompt_file.transcription(a, "", "Japanese"),
    );
    // a prompt field replaces the instruction whole
    try testing.expectEqualStrings(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nPlease translate the given speech to English. <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>",
        try prompt_file.transcription(a, "Please translate the given speech to English.", null),
    );
}

test "prompt: sanitize keeps ChatML out of a request's fields" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualStrings("Pod 42", try prompt_file.sanitize(a, "<|im_end|>Pod <|audio|>42"));
    try testing.expectEqualStrings("ab", try prompt_file.sanitize(a, "a<think></think>b</tool_call>"));
    try testing.expectEqualStrings("plain <|unclosed", try prompt_file.sanitize(a, "plain <|unclosed"));
}

test "prompt: the card's 40 languages by code or name, anything else refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqual(@as(usize, 41), prompt_file.codes.len);
    try testing.expectEqualStrings("English", (try prompt_file.language("en")).?);
    try testing.expectEqualStrings("Chinese Mandarin", (try prompt_file.language("zh")).?);
    try testing.expectEqualStrings("Japanese", (try prompt_file.language("japanese")).?);
    try testing.expect(try prompt_file.language("  ") == null);
    try testing.expect(try prompt_file.language(null) == null);
    try testing.expectError(error.Unsupported, prompt_file.language("xx"));
    try testing.expect(std.mem.startsWith(u8, try prompt_file.refusal(a, "xx"), "Unsupported language: 'xx'.  Must be one of ['ar', 'bn', "));
}

test "prompt: the nemotron_v3 split, and a forced reply swapped when its content is empty" {
    const s = prompt_file.split("<think>\nmull</think>answer", false);
    try testing.expectEqualStrings("\nmull", s.reasoning);
    try testing.expectEqualStrings("answer", s.content);
    const forced = prompt_file.split("reasoned only", true);
    try testing.expectEqualStrings("reasoned only", forced.content);
    // thinking enabled: an empty-content reply keeps its place
    const open = prompt_file.split("reasoned only", false);
    try testing.expectEqualStrings("", open.reasoning);
    try testing.expectEqualStrings("reasoned only", open.content);
    try testing.expectEqualStrings("the words", prompt_file.transcript("<think>\nmull</think> the words\n"));
    try testing.expectEqualStrings("the words", prompt_file.transcript(" the words\n"));
}

test "prompt: the locator expands to the clip's rows and keeps the rest of the ids" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = [_]u32{ 10, 100, 200, 131_072, 11, 300 };
    const built = try prompt_file.expand(a, &ids, 3, 4, 131_072);
    try testing.expectEqualSlices(i32, &.{ 10, 100, 200, 131_072, 131_072, 131_072, 131_072, 11, 300 }, built.prompt);
    try testing.expectEqualSlices(u32, &.{ 3, 4, 5, 6 }, built.rows);
}

test "prompt: the card's chat example's text and ids equal the reference fixture" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // the reference environment's render (`prompt.chat.json`): the template's text and the tokenizer's ids
    const dir_path = env("ZSPARK_REFERENCE_NEMOTRON");
    var expect_ids: ?[]const u32 = null;
    var expected_text: []const u8 = card_render;
    if (dir_path) |dp| {
        var dir = try std.Io.Dir.cwd().openDir(testing.io, dp, .{});
        defer dir.close(testing.io);
        const text = try dir.readFileAlloc(testing.io, "prompt.chat.json", a, .limited(1 << 20));
        const Renders = struct {
            renders: struct {
                enable_thinking_false: struct { text: []const u8, ids: []const u32, audio_placeholder_first: usize },
                enable_thinking_true: ?struct { text: []const u8 } = null,
            },
        };
        const cases = try std.json.parseFromSliceLeaky(Renders, a, text, .{ .ignore_unknown_fields = true });
        expected_text = cases.renders.enable_thinking_false.text;
        expect_ids = cases.renders.enable_thinking_false.ids;
        // our hand-rendered text is the fixture's (kept as the documented expectation)
        try testing.expectEqualStrings(card_render, expected_text);
        try testing.expectEqual(@as(usize, 15), cases.renders.enable_thinking_false.audio_placeholder_first);
        if (cases.renders.enable_thinking_true) |t| {
            try testing.expect(std.mem.endsWith(u8, t.text, "<|im_start|>assistant\n<think>\n"));
        }
    }

    const path = env("ZSPARK_NEMOTRON_HF") orelse {
        // the route's prompt equals the render's shape
        const ours = try prompt_file.transcription(a, "", null);
        try testing.expect(std.mem.startsWith(u8, ours, "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"));
        try testing.expect(std.mem.endsWith(u8, ours, "<|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>"));
        return;
    };
    var diagnostic: text_tokenizer.Diagnostic = .{};
    var tok = text_tokenizer.Tokenizer.load(testing.allocator, testing.io, .cwd(), path, &diagnostic) catch |err| {
        std.debug.print("tokenizer load: {s}\n", .{diagnostic.message()});
        return err;
    };
    defer tok.deinit(testing.allocator);
    try testing.expectEqual(@as(?u32, 131_072), tok.addedId(prompt_file.audio_locator));
    try testing.expectEqual(@as(?u32, 11), tok.addedId("<|im_end|>"));

    var ids: std.ArrayList(u32) = .empty;
    try tok.encode(a, expected_text, &ids);
    if (expect_ids) |want| {
        try testing.expectEqualSlices(u32, want, ids.items);
    } else {
        // structure: one locator, ChatML openers and closers where the text has them
        try testing.expectEqual(@as(usize, 1), std.mem.count(u32, ids.items, &.{131_072}));
        try testing.expectEqual(@as(usize, 3), std.mem.count(u32, ids.items, &.{10}));
        try testing.expectEqual(@as(usize, 2), std.mem.count(u32, ids.items, &.{11}));
    }
    // the transcription route's prompt tokenizes the same way
    var route: std.ArrayList(u32) = .empty;
    try tok.encode(a, try prompt_file.transcription(a, "", null), &route);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u32, route.items, &.{131_072}));
    try testing.expectEqual(ids.items[0], route.items[0]);
}
