//! Text helpers an ASR route's prompt and reply share (spec 151 WS-08): Python's `str.strip()` for the
//! whitespace a transcript may hold, and the removal of ChatML-like tags from a client's `prompt` field before
//! it is written into the model's turn — `qwen3_asr/transcript.zig`'s `strip` and `sanitize`, moved here when
//! Nemotron-3.5-Transcribe's transcription prompt needed the same two (the models' own tag lists stay with the
//! models). Host only, no other zspark layer.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Python's `str.strip()` for the whitespace a transcript may hold: ASCII's and 0x1c to 0x1f, NBSP and U+3000.
pub fn strip(s: []const u8) []const u8 {
    var start: usize = 0;
    var end = s.len;
    outer: while (start < end) {
        if (std.ascii.isWhitespace(s[start]) or (s[start] >= 0x1c and s[start] <= 0x1f)) {
            start += 1;
            continue;
        }
        for (wide_spaces) |wide| if (std.mem.startsWith(u8, s[start..end], wide)) {
            start += wide.len;
            continue :outer;
        };
        break;
    }
    outer: while (end > start) {
        if (std.ascii.isWhitespace(s[end - 1]) or (s[end - 1] >= 0x1c and s[end - 1] <= 0x1f)) {
            end -= 1;
            continue;
        }
        for (wide_spaces) |wide| if (std.mem.endsWith(u8, s[start..end], wide)) {
            end -= wide.len;
            continue :outer;
        };
        break;
    }
    return s[start..end];
}

const wide_spaces = [_][]const u8{ "\u{a0}", "\u{3000}" };

/// `text` with `<|...|>` (no `|` inside) and each of `tags` removed until none is left, so a client's
/// context cannot close a turn or plant a locator (vLLM's `_sanitize_transcription_user_text`, which removes
/// ChatML-like tokens to a fixpoint; the regex pass first, then the named tags as its `.replace` does).
/// The result is `a`'s.
pub fn stripTags(a: Allocator, text: []const u8, tags: []const []const u8) Allocator.Error![]u8 {
    var now = try a.dupe(u8, text);
    errdefer a.free(now);
    while (true) {
        var out: std.ArrayList(u8) = try .initCapacity(a, now.len);
        defer out.deinit(a);
        var i: usize = 0;
        while (i < now.len) {
            if (std.mem.startsWith(u8, now[i..], "<|")) {
                // `<\|[^|]+\|>`: at least one byte that is not `|`, then `|>`
                var j = i + 2;
                while (j < now.len and now[j] != '|') j += 1;
                if (j > i + 2 and j + 1 < now.len and now[j + 1] == '>') {
                    i = j + 2;
                    continue;
                }
            }
            try out.append(a, now[i]);
            i += 1;
        }
        var once: []u8 = try a.dupe(u8, out.items);
        for (tags) |tag| {
            const next = try std.mem.replaceOwned(u8, a, once, tag, "");
            a.free(once);
            once = next;
        }
        if (std.mem.eql(u8, once, now)) {
            a.free(once);
            return now;
        }
        a.free(now);
        now = once;
    }
}
