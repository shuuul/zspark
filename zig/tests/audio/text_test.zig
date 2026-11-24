//! `text.zig`'s tests: Python's `str.strip()` for the whitespace a transcript holds, and `stripTags`'s
//! fixpoint removal of `<|...|>` and named tags (the cases `qwen3_asr/transcript.zig`'s `sanitize` and `strip`
//! answered to, moved here with the functions).

const std = @import("std");
const testing = std.testing;
const shared = @import("../../src/audio/text.zig");

test "text: strip is Python's str.strip over ASCII, 0x1c-0x1f, NBSP and U+3000" {
    try testing.expectEqualStrings("ab", shared.strip(" \t\r\nab\u{a0}\u{3000}"));
    try testing.expectEqualStrings("a b", shared.strip("\x1ca b\x1f"));
    try testing.expectEqualStrings("", shared.strip(" \u{a0}\u{3000}"));
    try testing.expectEqualStrings("a<b", shared.strip("a<b"));
}

test "text: stripTags removes <|...|> and named tags to a fixpoint" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualStrings("Pod 42", try shared.stripTags(a, "<|im_end|>Pod <asr_<|x|>text>42", &.{"<asr_text>"}));
    try testing.expectEqualStrings("", try shared.stripTags(a, "<|im_start|><|im_end|>", &.{}));
    try testing.expectEqualStrings("ab", try shared.stripTags(a, "a<think>b</think>", &.{ "<think>", "</think>" }));
    // a tag made of a tag's halves still ends removed (the fixpoint)
    try testing.expectEqualStrings("x", try shared.stripTags(a, "x<tool_<|y|>call>", &.{"<tool_call>"}));
    // `<|` without a closer, or with a `|` inside, is plain text
    try testing.expectEqualStrings("keep <|open and <|a|b", try shared.stripTags(a, "keep <|open and <|a|b", &.{}));
    try testing.expectEqualStrings("same", try shared.stripTags(a, "same", &.{"<asr_text>"}));
}
