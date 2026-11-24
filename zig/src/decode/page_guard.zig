//! Unlimited-OCR-2's PageCountGuard (spec 147 WS-06), a reply's constraint beside the no-repeat n-gram rule
//! (`ngram.zig`): a reply writes one `<PAGE>` for each the prompt holds, and a page is at most `cap` tokens.
//!
//! As the checkpoint's `_uocr_page_count_guard` (`modeling_unlimitedocr.py`, vLLM's `_PageCountGuard`), on the
//! reply's tokens alone, before the row is chosen and after the n-gram rule:
//!
//! - more markers than the prompt's: only the end token;
//! - a page of `cap` tokens or more (the tokens since the last marker, or since the reply began): only `<PAGE>`
//!   while markers are owed, else only the end token;
//! - fewer markers than the prompt's: the end token banned;
//! - a prompt with no marker, or a reply on its last page under the cap: the row as it stands.
//!
//! A cap of 0 is none (HF's `0 < max_page_tokens`). One departure: where the n-gram rule banned the one token the
//! guard leaves, HF's argmax over a row of -inf takes id 0; here that token is kept at 0, so the guard's token is
//! chosen, as SGLang's `DeepseekOCRNoRepeatNGramLogitProcessor` forces it (its guard runs first and skips the
//! n-gram rule). SGLang's `_page_guard` also bans `<PAGE>` on the last page; HF does not, and neither does this.
//!
//! The row rule: a row's action is made from the reply's tokens before it alone. The scan is incremental, as HF's
//! (`scanned`): a reply shorter than what was scanned is scanned again from its start.

const std = @import("std");
const masks = @import("masks.zig");

pub const Guard = struct {
    /// The prompt's `<PAGE>` count; 0 is no guard.
    expected: u32,
    /// A page's most tokens (`max_page_tokens`); 0 is no cap.
    cap: u32,
    /// `<PAGE>` and the end token.
    page: u32,
    eos: u32,
    /// The reply's markers, the tokens since its last (or since its start), and how much of it is counted.
    marks: u32 = 0,
    page_len: usize = 0,
    scanned: usize = 0,

    /// The checkpoint's default cap (`infer`'s `max_page_tokens`, SGLang's README request).
    pub const default_cap: u32 = 20000;

    pub fn init(prompt: anytype, page: u32, eos: u32, cap: u32) Guard {
        return .{ .expected = pagesIn(prompt, page), .cap = cap, .page = page, .eos = eos };
    }

    pub const Action = union(enum) {
        none,
        /// The end token may not be chosen.
        ban: u32,
        /// Only this token may be chosen.
        force: u32,
    };

    /// What the row after `reply` (the reply's tokens so far, without the prompt) may be.
    pub fn verdict(self: *Guard, reply: []const u32) Action {
        if (self.expected == 0) return .none;
        if (reply.len < self.scanned) self.* = .{ .expected = self.expected, .cap = self.cap, .page = self.page, .eos = self.eos };
        const tail = reply[self.scanned..];
        if (std.mem.lastIndexOfScalar(u32, tail, self.page)) |last| {
            self.marks += @intCast(std.mem.count(u32, tail, &.{self.page}));
            self.page_len = tail.len - 1 - last;
        } else self.page_len += tail.len;
        self.scanned = reply.len;
        if (self.marks > self.expected) return .{ .force = self.eos };
        if (self.cap > 0 and self.page_len >= self.cap) return .{ .force = if (self.marks < self.expected) self.page else self.eos };
        if (self.marks < self.expected) return .{ .ban = self.eos };
        return .none;
    }

    /// `verdict` laid on a row's logits, bf16 bits over the vocabulary.
    pub fn apply(self: *Guard, reply: []const u32, logits: []u16) Action {
        const action = self.verdict(reply);
        switch (action) {
            .none => {},
            .ban => |token| if (token < logits.len) {
                logits[token] = masks.minus_infinity;
            },
            .force => |token| if (token < logits.len) {
                const kept = logits[token];
                @memset(logits, masks.minus_infinity);
                // a token the n-gram rule banned (or a NaN) is kept at 0: the guard's token is chosen
                logits[token] = if ((kept & 0x7f80) == 0x7f80) 0 else kept;
            },
        }
        return action;
    }
};

/// How many of `ids` (u32 or i32) are `page`.
pub fn pagesIn(ids: anytype, page: u32) u32 {
    var n: u32 = 0;
    for (ids) |id| n += @intFromBool(id >= 0 and @as(u32, @intCast(id)) == page);
    return n;
}
