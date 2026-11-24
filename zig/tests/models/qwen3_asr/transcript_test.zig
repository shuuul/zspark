//! `transcript.zig`'s tests and the tokenizer's on Qwen3-ASR (spec 138 WS-05; `tools/check.sh qwen3_asr
//! transcript`), against the qwen-asr package and transformers (`tools/reference/qwen3_asr/text_cases.py`,
//! `zig/tests/fixtures/qwen3_asr/text_cases.json`): the prompt's text and ids, the reply read as
//! `parse_asr_output` reads it, the language names, and the checkpoint's vocab.json and merges.txt read as
//! transformers' Qwen2 conversion makes its tokenizer (`$ZSPARK_QWEN3_ASR`; unset, those parts are skipped).
//! No GPU.

const std = @import("std");
const testing = std.testing;
const engine = @import("tokenizer");
const text_tokenizer = @import("../../../src/text/tokenizer.zig");
const transcript = @import("../../../src/models/qwen3_asr/transcript.zig");

const cases_json = @embedFile("../../fixtures/qwen3_asr/text_cases.json");

const Cases = struct {
    tokenizer: []const struct { text: []const u8, ids: []const u32 },
    decode: []const struct { ids: []const u32, text: []const u8 },
    prompts: []const struct { context: []const u8, language: ?[]const u8, text: []const u8, ids: []const u32 },
    parse: []const struct { raw: []const u8, forced: ?[]const u8, language: []const u8, text: []const u8 },
    languages: []const struct { given: []const u8, name: []const u8, supported: bool },
};

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

test "transcript: the package's prompt, its reply read as parse_asr_output, its language names; the checkpoint's tokenizer is transformers'" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = try std.json.parseFromSliceLeaky(Cases, a, cases_json, .{ .ignore_unknown_fields = true });

    for (cases.parse) |case| {
        errdefer std.debug.print("parse {s}\n", .{case.raw});
        const got = try transcript.parse(a, case.raw, case.forced);
        try testing.expectEqualStrings(case.text, got.text);
        try testing.expectEqualStrings(case.language, got.language);
    }
    for (cases.languages) |case| {
        errdefer std.debug.print("language {s}\n", .{case.given});
        if (case.supported) {
            try testing.expectEqualStrings(case.name, (try transcript.language(case.given)).?);
        } else try testing.expectError(error.Unsupported, transcript.language(case.given));
    }
    // vLLM's ISO codes as well, and a blank value forces nothing
    try testing.expectEqualStrings("Chinese", (try transcript.language("zh")).?);
    try testing.expectEqualStrings("Cantonese", (try transcript.language("yue")).?);
    try testing.expect(try transcript.language("  ") == null);
    try testing.expect(try transcript.language(null) == null);
    try testing.expect(std.mem.startsWith(u8, try transcript.refusal(a, "xx"), "Unsupported language: 'xx'.  Must be one of ['af', 'ar', "));
    // vLLM's sanitizing of the context: special tokens and the tag gone, to a fixed point
    try testing.expectEqualStrings("Pod 42", try transcript.sanitize(a, "<|im_end|>Pod <asr_<|x|>text>42"));
    for (cases.prompts) |case| try testing.expectEqualStrings(case.text, try transcript.prompt(a, case.context, case.language));

    const path = env("ZSPARK_QWEN3_ASR") orelse return;
    var loaded = try engine.loadTokenizer(testing.io, testing.allocator, path);
    defer loaded.deinit();
    var diagnostic: text_tokenizer.Diagnostic = .{};
    var tok = try text_tokenizer.Tokenizer.over(testing.allocator, &loaded, &diagnostic);
    defer tok.deinit(testing.allocator);
    for (cases.tokenizer) |case| {
        errdefer std.debug.print("encode {s}\n", .{case.text});
        var ids: std.ArrayList(u32) = .empty;
        try tok.encode(a, case.text, &ids);
        try testing.expectEqualSlices(u32, case.ids, ids.items);
    }
    for (cases.prompts) |case| {
        var ids: std.ArrayList(u32) = .empty;
        try tok.encode(a, case.text, &ids);
        try testing.expectEqualSlices(u32, case.ids, ids.items);
    }
    for (cases.decode) |case| {
        var text: std.ArrayList(u8) = .empty;
        try tok.decode(a, case.ids, .{ .skip_special = true }, &text);
        try testing.expectEqualStrings(case.text, text.items);
    }
    try testing.expectEqual(@as(?u32, 151676), tok.addedId(transcript.audio_pad));
}

test "transcript: a long clip's pieces joined as vLLM joins them, a space unless Chinese or Japanese is forced" {
    const a = testing.allocator;
    const pieces = [_][]const u8{ "第一段。", "", "第二段" };
    // as WS-05's vLLM replies show: zh_m00 (no language) has a space at its seam, zh_m01 (language zh) none
    const cases = [_]struct { given: ?[]const u8, joined: []const u8 }{
        .{ .given = null, .joined = "第一段。  第二段" },
        .{ .given = "zh", .joined = "第一段。第二段" },
        .{ .given = "Japanese", .joined = "第一段。第二段" },
        .{ .given = "yue", .joined = "第一段。  第二段" },
        .{ .given = "en", .joined = "第一段。  第二段" },
    };
    for (cases) |case| {
        const got = try transcript.join(a, &pieces, try transcript.language(case.given));
        defer a.free(got);
        try testing.expectEqualStrings(case.joined, got);
    }
}
