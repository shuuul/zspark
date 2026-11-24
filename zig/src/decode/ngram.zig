//! The sliding-window no-repeat n-gram rule (spec 137 WS-05), a reply's constraint that no model owns: over the
//! last `window` tokens of the sequence, every n-gram whose first n - 1 tokens equal the sequence's last n - 1
//! bans its last token at the row being chosen; `whitelist` tokens are never banned.
//!
//! As the checkpoint's `SlidingWindowNoRepeatNgramProcessor` (`modeling_unlimitedocr.py`) and vLLM's
//! `NoRepeatNGramLogitsProcessor` (`deepseek_ocr.py`), the ngram starts of which are `[max(0, len - window),
//! len - n]`. Two departures of vLLM's are not followed: it is handed the output ids alone where HF's sequence is
//! prompt and output (the caller of this file gives the sequence, whole or not), and at `size` 1 its prefix
//! `output_ids[-0:]` is the whole list and matches nothing, where HF bans every token of the window.
//!
//! The row rule: a row's banned set is made from that row's own sequence, the tokens before it and nothing else.
//! A verify window's row i is the sequence plus the first i drafts (`ahead`), so a serial step and a window give
//! the same set for the same row. Like `masks.zig` the rule acts on the host, on the row the sampler reads.

const std = @import("std");
const masks = @import("masks.zig");

pub const Rule = struct {
    /// n, 1 or more: a repeated run of this many tokens may not be completed.
    size: u32,
    /// How many of the sequence's last tokens the earlier n-grams start in, 1 or more.
    window: u32,
    /// Tokens never banned.
    whitelist: []const u32 = &.{},

    /// vLLM's recipe for Unlimited-OCR on an image; a PDF page takes window 1,024.
    pub const image: Rule = .{ .size = 35, .window = 128 };
    pub const page: Rule = .{ .size = 35, .window = 1024 };

    pub fn valid(self: Rule) bool {
        return self.size >= 1 and self.window >= 1;
    }
};

/// The sequence `base ++ ahead`, read without joining them.
const Sequence = struct {
    base: []const u32,
    ahead: []const u32,

    fn len(self: Sequence) usize {
        return self.base.len + self.ahead.len;
    }

    fn at(self: Sequence, i: usize) u32 {
        return if (i < self.base.len) self.base[i] else self.ahead[i - self.base.len];
    }
};

/// Calls `emit(token)` for each token the rule bans after `base ++ ahead`, once for each n-gram that bans it
/// (so a token may repeat), whitelisted tokens left out.
fn scan(rule: Rule, sequence: Sequence, context: anytype, comptime emit: fn (@TypeOf(context), u32) void) void {
    std.debug.assert(rule.valid());
    const n: usize = rule.size;
    const len = sequence.len();
    if (len < n) return;
    const start = len -| @as(usize, rule.window);
    const end = len - n + 1;
    if (end <= start) return;
    const prefix = n - 1;
    for (start..end) |from| {
        var same = true;
        for (0..prefix) |k| {
            if (sequence.at(from + k) != sequence.at(len - prefix + k)) {
                same = false;
                break;
            }
        }
        if (!same) continue;
        const token = sequence.at(from + prefix);
        if (std.mem.indexOfScalar(u32, rule.whitelist, token) == null) emit(context, token);
    }
}

/// Row logits' bf16 bits at -inf for each banned token (`masks.minus_infinity`); a banned token past the row is
/// not in the vocabulary and is left. Returns how many n-grams banned one (a token may count more than once).
pub fn apply(rule: Rule, base: []const u32, ahead: []const u32, logits: []u16) usize {
    const Hit = struct {
        logits: []u16,
        count: usize = 0,
        fn emit(self: *@This(), token: u32) void {
            self.count += 1;
            if (token < self.logits.len) self.logits[token] = masks.minus_infinity;
        }
    };
    var hit: Hit = .{ .logits = logits };
    scan(rule, .{ .base = base, .ahead = ahead }, &hit, Hit.emit);
    return hit.count;
}

/// The banned tokens, ascending and each once, appended to `out`.
pub fn banned(gpa: std.mem.Allocator, rule: Rule, base: []const u32, ahead: []const u32, out: *std.ArrayList(u32)) std.mem.Allocator.Error!void {
    const Hit = struct {
        gpa: std.mem.Allocator,
        out: *std.ArrayList(u32),
        failed: bool = false,
        fn emit(self: *@This(), token: u32) void {
            self.out.append(self.gpa, token) catch {
                self.failed = true;
            };
        }
    };
    const first = out.items.len;
    var hit: Hit = .{ .gpa = gpa, .out = out };
    scan(rule, .{ .base = base, .ahead = ahead }, &hit, Hit.emit);
    if (hit.failed) return error.OutOfMemory;
    const added = out.items[first..];
    std.mem.sort(u32, added, {}, std.sort.asc(u32));
    var kept: usize = 0;
    for (added, 0..) |token, i| {
        if (i > 0 and token == added[kept - 1]) continue;
        added[kept] = token;
        kept += 1;
    }
    out.shrinkRetainingCapacity(first + kept);
}
