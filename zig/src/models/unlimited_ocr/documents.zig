//! Unlimited-OCR-2's long documents (spec 147 WS-08), host only: how a prompt of many pages is cut into parts of
//! consecutive pages, and how a part takes over the previous part's last page instead of reading it again. Ported
//! from SGLang's v2 wheel (Apache-2.0, read 2026-10-28): `entrypoints/openai/doc_split.py` (`plan_batches`,
//! `window_output_cap`), `serving_chat.py` (`_window_lookahead`, `_window_capacity`, `_window_request`) and
//! `session/uocr_doc_reuse.py`'s token-layout helpers (`parse_prompt`, `plan_tail_copy`, `text_tail`,
//! `_cut_at_extra_marker`, `_page_base`). `parts.zig` runs the parts on the device.
//!
//! - A document whose pages fit one state is one part: the request as it stands, the exact form. Else parts of
//!   `fits` pages (`plan`, SGLang's `(lo, hi, emit_from)` with no overlap): each part but the last reads the
//!   next part's first page too (its lookahead, `lookahead`), and ends on the `<PAGE>` past its own pages.
//! - A part after the first starts from the previous one's state (`TailCopy`): the `<PAGE>` of the previous
//!   part's last own page, that page's last 128 image rows from a 32-row block on, and the lookahead page whole
//!   are copied, keys and values, into a prompt with no BOS and no instruction; then the previous reply's last
//!   rows of its last page (`textTail`, at most 128) and a forced `<PAGE>`, which the reply goes on from. SGLang
//!   calls this approximate, held by OCR accuracy. A part that cannot (the previous one ran out of tokens, its
//!   text is too long for the part's budget) reads its pages again with the instruction, as the first does.
//! - The RoPE page of a row is its part's page plus `pageBase`: the document's page numbers, SGLang's
//!   `KEY_PAGE_BASE`. A part's own pages are what its guard counts (`ownPages`).

const std = @import("std");
const Allocator = std.mem.Allocator;

/// SGLang's `CONTEXT_RESERVE_TOKENS`.
pub const context_reserve = 64;
/// SGLang's `_PAGE_CAP_SLACK_TOKENS`: the markers around each page and the stop, over a part's page caps.
pub const page_cap_slack = 256;
/// SGLang's `TAIL_TOKENS`: the dense local window every query reads (128 keys).
pub const tail_tokens = 128;
/// SGLang's `CONTEXT_PAGES`: the pages a reused part's prompt carries but does not transcribe.
pub const context_pages = 1;
/// SGLang's `SGLANG_UOCR_DOC_LOOKAHEAD_PAGES` default.
pub const lookahead_pages = 1;
/// zspark's: the reply rows a page takes in the state beside its prompt rows, where the server picks a part's pages
/// by memory (SGLang counts the prompt's alone and lets a part's reply run out of context). Measured on a 30+ page
/// paper (spec 147 WS-08: Mamba, arXiv 2312.00752, 36 pages in base mode, 47,182 tokens, 1,311 a page).
pub const reply_rows_per_page = 1312;

/// SGLang's `_TOKENS_PER_PAGE`: a page's prompt rows by image mode, measured on the checkpoint; gundam where the
/// mode is unknown.
pub fn tokensPerPage(mode: []const u8) u32 {
    const table = [_]struct { []const u8, u32 }{ .{ "gundam", 1514 }, .{ "large", 422 }, .{ "base", 274 }, .{ "small", 112 }, .{ "tiny", 74 } };
    for (table) |entry| if (std.mem.eql(u8, entry[0], mode)) return entry[1];
    return 1514;
}

/// A prompt's pages: `prefix` rows before the first `<PAGE>`, a page's `<PAGE>` at `starts[i]` and its image rows to
/// `ends[i]` (exclusive), then the instruction from `instruction`. SGLang's `PromptPages`.
pub const Prompt = struct {
    prefix: usize,
    starts: []usize,
    ends: []usize,
    instruction: usize,

    pub fn deinit(self: Prompt, a: Allocator) void {
        a.free(self.starts);
        a.free(self.ends);
    }

    pub fn pages(self: Prompt) usize {
        return self.starts.len;
    }
};

/// SGLang's `parse_prompt`: `prefix <PAGE><image>... <PAGE><image>... instruction`, or null where the prompt is not
/// that (an image before the first marker, a marker without images, an image in the instruction).
pub fn parse(a: Allocator, ids: []const i32, page: i32, image: i32) Allocator.Error!?Prompt {
    var pos: usize = 0;
    while (pos < ids.len and ids[pos] != page) : (pos += 1) if (ids[pos] == image) return null;
    if (pos >= ids.len) return null;
    var starts: std.ArrayList(usize) = .empty;
    errdefer starts.deinit(a);
    var ends: std.ArrayList(usize) = .empty;
    errdefer ends.deinit(a);
    const prefix = pos;
    while (pos < ids.len and ids[pos] == page) {
        try starts.append(a, pos);
        pos += 1;
        const first = pos;
        while (pos < ids.len and ids[pos] == image) pos += 1;
        if (pos == first) {
            starts.deinit(a);
            ends.deinit(a);
            return null;
        }
        try ends.append(a, pos);
    }
    if (std.mem.indexOfScalar(i32, ids[pos..], image) != null) {
        starts.deinit(a);
        ends.deinit(a);
        return null;
    }
    return .{ .prefix = prefix, .starts = try starts.toOwnedSlice(a), .ends = try ends.toOwnedSlice(a), .instruction = pos };
}

/// A part of the document: its own pages `lo` .. `hi - 1`; SGLang's `(lo, hi, emit_from)`, `emit_from` `lo` with no
/// overlap.
pub const Part = struct { lo: u32, hi: u32 };

/// SGLang's `plan_batches` with `overlap_pages` 0: null where the document stays one part (one page, a budget of
/// none, or every page fits `context` and `max_pages` does not force a cut), else contiguous parts of `fits` pages,
/// `fits` the pages whose `per_page` rows fit `context` after the reserve, at most `max_pages`.
pub fn plan(a: Allocator, n_pages: u32, per_page: u32, context: u64, max_tokens: u64, max_pages: ?u32) Allocator.Error!?[]Part {
    if (n_pages <= 1 or context <= context_reserve or max_tokens == 0) return null;
    var fits: u64 = (context - context_reserve - 1) / per_page;
    if (max_pages) |m| fits = @min(fits, m);
    if (fits < 1) return null;
    const forced = if (max_pages) |m| n_pages > m else false;
    // SGLang's one part of every page where only the budget is clamped is the request as it stands
    if (fits >= n_pages and !forced) return null;
    var parts: std.ArrayList(Part) = .empty;
    errdefer parts.deinit(a);
    var lo: u32 = 0;
    while (lo < n_pages) {
        const hi: u32 = @intCast(@min(lo + fits, n_pages));
        try parts.append(a, .{ .lo = lo, .hi = hi });
        lo = hi;
    }
    return try parts.toOwnedSlice(a);
}

/// SGLang's `_window_lookahead`: the pages part `i` reads past its own, the next part's first ones; none for the last.
pub fn lookahead(parts: []const Part, i: usize) u32 {
    if (i + 1 >= parts.len) return 0;
    return @min(lookahead_pages, parts[parts.len - 1].hi - parts[i].hi);
}

/// SGLang's `window_output_cap`: the most a guarded part writes, null where a page has no cap.
pub fn outputCap(own_pages: u32, max_page_tokens: u32) ?u64 {
    if (max_page_tokens == 0) return null;
    return @as(u64, own_pages) * max_page_tokens + page_cap_slack;
}

/// SGLang's `_page_base`: what a row's part page adds to make the document's: `lo` for a part that reads its pages
/// (its first own page is its page 1), `lo - 1` for one that took over the previous part's last page (its page 1).
pub fn pageBase(part: Part, reused: bool) i32 {
    return @as(i32, @intCast(part.lo)) - @as(i32, @intFromBool(reused)) * context_pages;
}

/// The pages a part's guard expects (SGLang's `_page_guard` `expected`): the prompt's markers less the context page
/// a reused part carries and its lookahead.
pub fn ownPages(markers: u32, reused: bool, ahead: u32) u32 {
    return markers -| (@as(u32, @intFromBool(reused)) * context_pages + ahead);
}

/// The ids of a part that reads its pages: the prompt's prefix, pages `lo` .. `hi + ahead - 1` and its instruction,
/// as SGLang's `split_messages` and `_retag_markers` render it. In `a`.
pub fn partIds(a: Allocator, ids: []const i32, p: Prompt, part: Part, ahead: u32) Allocator.Error![]i32 {
    const last = part.hi + ahead - 1;
    const body = ids[p.starts[part.lo]..p.ends[last]];
    const tail = ids[p.instruction..];
    const out = try a.alloc(i32, p.prefix + body.len + tail.len);
    @memcpy(out[0..p.prefix], ids[0..p.prefix]);
    @memcpy(out[p.prefix..][0..body.len], body);
    @memcpy(out[p.prefix + body.len ..], tail);
    return out;
}

/// SGLang's `TailCopyPlan`: a part's rows copied from the previous part's state, and its prompt.
pub const TailCopy = struct {
    /// the old part's rows the new prompt's first `copy_len` rows take, in order: the marker's, then
    /// `start` .. `copy_end - 1`
    marker: usize,
    start: usize,
    copy_end: usize,
    /// the new prompt: the copied head, then the new part's ids from `kept_from` up to its instruction
    ids: []i32,
    copy_len: usize,
    kept_from: usize,
    old_pages: usize,
    new_pages: usize,
    /// the new part's leading pages copied whole (the old part's lookahead)
    copied_pages: usize,
    /// a one-page part leaves its copied page's last image row to its prompt (SGLang's extend needs a row)
    straddle: bool,

    pub fn deinit(self: TailCopy, a: Allocator) void {
        a.free(self.ids);
    }
};

/// SGLang's `plan_tail_copy` with `overlap` 0: `old` the previous part's prompt ids (`old_ahead` lookahead pages at
/// its end), `new` this part's as `partIds` makes them, `block` the KV page (32). Null where the layouts do not
/// allow it. `copy_ahead` SGLang's `SGLANG_UOCR_DOC_LOOKAHEAD_KV_REUSE` (on).
pub fn tailCopy(a: Allocator, old: []const i32, new: []const i32, page: i32, image: i32, block: usize, old_ahead: usize, copy_ahead: bool) Allocator.Error!?TailCopy {
    const op = try parse(a, old, page, image) orelse return null;
    defer op.deinit(a);
    const np = try parse(a, new, page, image) orelse return null;
    defer np.deinit(a);
    if (op.pages() <= old_ahead) return null;
    const p_old = op.pages() - old_ahead;
    const p_new = np.pages();
    if (p_old < 1 or p_new < 1) return null;
    const end = op.ends[p_old - 1];
    if (end < tail_tokens + end % block) return null;
    const start = end - tail_tokens - end % block;
    if (start <= op.starts[p_old - 1]) return null;
    var copy_end = end;
    var kept_from = np.prefix;
    var copied: usize = 0;
    var straddle = false;
    if (copy_ahead and old_ahead > 0) {
        const la_start = op.starts[p_old];
        const la_end = op.ends[p_old];
        if (la_start == end and std.mem.eql(i32, old[la_start..la_end], new[np.starts[0]..np.ends[0]])) {
            copy_end = la_end;
            kept_from = np.ends[0];
            copied = 1;
            if (p_new == 1) {
                copy_end = la_end - 1;
                kept_from = np.ends[0] - 1;
                straddle = true;
            }
        }
    }
    const copy_len = 1 + copy_end - start;
    const rest = new[kept_from..np.instruction];
    const ids = try a.alloc(i32, copy_len + rest.len);
    ids[0] = page;
    @memcpy(ids[1..copy_len], old[start..copy_end]);
    @memcpy(ids[copy_len..], rest);
    return .{
        .marker = op.starts[p_old - 1],
        .start = start,
        .copy_end = copy_end,
        .ids = ids,
        .copy_len = copy_len,
        .kept_from = kept_from,
        .old_pages = p_old,
        .new_pages = p_new,
        .copied_pages = copied,
        .straddle = straddle,
    };
}

/// SGLang's `text_tail`: the first of a reply's last at most `n` tokens after its last `<PAGE>`; null for an empty
/// reply.
pub fn textTail(reply: []const u32, n: usize, page: u32) ?usize {
    if (reply.len == 0 or n == 0) return null;
    const after = if (std.mem.lastIndexOfScalar(u32, reply, page)) |last| last + 1 else 0;
    return @max(after, reply.len -| n);
}

/// SGLang's `_cut_at_extra_marker`: a reply up to its `<PAGE>` past `pages` of them.
pub fn cutAtExtraMarker(reply: []const u32, page: u32, pages: usize) []const u32 {
    var seen: usize = 0;
    for (reply, 0..) |t, i| if (t == page) {
        seen += 1;
        if (seen > pages) return reply[0..i];
    };
    return reply;
}
