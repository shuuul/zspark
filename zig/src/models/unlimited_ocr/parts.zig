//! Unlimited-OCR-2's long documents on the device (spec 147 WS-08): a reply to a prompt of many pages, part after
//! part as `documents.zig` plans them, in one state. SGLang's v2 wheel runs its windows the same way
//! (`serving_chat.py` `_split_request`, `session/uocr_doc_reuse.py`), read 2026-10-28:
//!
//! - a part that reads its pages (the first, or one whose previous part did not end by a stop): its ids
//!   (`documents.partIds`) from an empty state, its rows' RoPE pages from the document's page `lo`;
//! - a part that takes over (`documents.TailCopy`): the previous part's marker of its last own page, that page's
//!   last 128 image rows from a block on and its lookahead page, keys, values and index keys, copied to the new
//!   prompt's first rows; the new pages read after them; then the previous reply's last text rows of its last
//!   page (`documents.textTail`) copied after the prompt, and a forced `<PAGE>` fed: the reply goes on from it.
//!   The copied rows keep the bits the previous part gave them; their blocks' k_bar is made again
//!   (`forward.summarize`). The client is given the forced `<PAGE>`, not the copied text (SGLang's
//!   `hide_forced_text`);
//! - a part with a page after its own, or one that took over, ends on the `<PAGE>` past its own pages, which is
//!   not given (`_cut_at_extra_marker`); its guard counts its own pages alone. The last part ends as a reply does.
//!
//! A part's budget is SGLang's `_window_capacity`: the reply's tokens left, the state's rows after the part's
//! prompt (zspark counts the rows the prompt takes, SGLang estimates them), and its pages' caps
//! (`documents.outputCap`). The reply ends `length` where a part ran out of it or a part was never run. Inside a
//! part every row is the state's as a serial step makes it: the row rule holds there. Across parts the copied rows
//! are SGLang's approximation (its words: held by OCR accuracy).

const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const cuda = @import("../../cuda/cuda.zig");
const decoding = @import("../../decode/draft.zig");
const page_guard = @import("../../decode/page_guard.zig");
const documents = @import("documents.zig");
const draft = @import("draft.zig");
const forward_file = @import("forward.zig");
const preprocess = @import("preprocess.zig");
const requests = @import("requests.zig");
const router_file = @import("router.zig");

const page: u32 = preprocess.page_token_id;
const image: u32 = preprocess.image_token_id;

/// The document: its prompt whole, parsed, and every page's image rows from the tower, page after page.
pub const Document = struct {
    ids: []const i32,
    prompt: documents.Prompt,
    /// bf16 [rows, hidden]
    features: cuda.Buffer,
    /// page i's rows are `features` rows `firsts[i]` .. `firsts[i + 1] - 1`
    firsts: []const usize,

    /// The document of `ids` whose pages' features are `features`, `firsts` in `a`; null where its pages are not
    /// `parse`'s or their rows are not `features`'.
    pub fn of(a: Allocator, ids: []const i32, features: cuda.Buffer, rows: []const usize) Allocator.Error!?Document {
        const p = try documents.parse(a, ids, @intCast(page), @intCast(image)) orelse return null;
        if (p.pages() != rows.len) return null;
        const firsts = try a.alloc(usize, rows.len + 1);
        firsts[0] = 0;
        for (rows, 0..) |r, i| {
            if (p.ends[i] - p.starts[i] - 1 != r) return null;
            firsts[i + 1] = firsts[i] + r;
        }
        if (firsts[rows.len] != @as(usize, @intCast(features.view.size[0]))) return null;
        return .{ .ids = ids, .prompt = p, .features = features, .firsts = firsts };
    }
};

/// What the caller does with the reply: choose a row's token, give a token to the client (true: the reply is
/// cut), say whether it was cancelled, and that the first part's prompt is in.
pub const Sink = struct {
    ctx: *anyopaque,
    choose: *const fn (ctx: *anyopaque, logits: []u16, position: usize) anyerror!i32,
    give: *const fn (ctx: *anyopaque, token: i32) bool,
    cancelled: *const fn (ctx: *anyopaque) bool,
    prefilled: *const fn (ctx: *anyopaque) void,
};

/// The model, its one state and buffers, a row's logits (bf16 bits over the vocabulary).
pub const Run = struct {
    model: forward_file.Model,
    state: *forward_file.State,
    buffers: *forward_file.Buffers,
    logits: []u16,
};

pub const Finish = enum { stop, length, cut, cancelled };

pub const Outcome = struct {
    finish: Finish = .stop,
    /// the end token a stop ended on, where it did
    end: ?u32 = null,
    /// the tokens the model chose (the forced ones not among them)
    made: usize = 0,
    parts: usize = 0,
    reused: usize = 0,
    /// when the first part's prompt was in
    prefilled_ns: u64 = 0,
};

pub const Error = error{
    /// a part's prompt alone fills the state
    PartTooLong,
};

/// A reply to `doc` in `parts` (a single part `{0, pages}`: the request as it stands), `request` its fields the
/// rules read (`requests.Rules.of`), at most `max_tokens` chosen tokens.
pub fn reply(gpa: Allocator, arena: Allocator, r: Run, doc: Document, parts: []const documents.Part, request: api.Request, max_tokens: usize, sink: Sink) !Outcome {
    const m = r.model;
    const state = r.state;
    const b = r.buffers;
    const eos = m.config.eos;
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    var outcome: Outcome = .{};
    var remaining = max_tokens;
    var all_stopped = true;
    var prev: ?Previous = null;
    for (parts, 0..) |part, i| {
        if (remaining == 0) {
            all_stopped = false;
            break;
        }
        const ahead = documents.lookahead(parts, i);
        const own = part.hi - part.lo;
        const ids = try documents.partIds(arena, doc.ids, doc.prompt, part, ahead);
        // the guard's cap where it is on (`requests.Rules.of`)
        const cap: u32 = if (request.eos.len == 0) 0 else request.max_page_tokens orelse page_guard.Guard.default_cap;
        var budget = remaining;
        if (documents.outputCap(own, cap)) |c| budget = @min(budget, c);

        // SGLang's `_build_reuse`: the previous part's tail and text where it ended by a stop and its text leaves
        // the part room
        var copy: ?documents.TailCopy = null;
        var text: []const u32 = &.{};
        if (prev) |p| if (p.stopped) {
            if (try documents.tailCopy(arena, p.ids, ids, @intCast(page), @intCast(image), router_file.block, p.ahead, true)) |t| {
                if (documents.textTail(p.out, documents.tail_tokens, page)) |s| if (p.out.len - s + 2 < budget) {
                    copy = t;
                    text = p.out[s..];
                };
            }
        };
        const prompt = if (copy) |t| t.ids else ids;
        const extra: usize = if (copy != null) text.len + 1 else 0;
        if (prompt.len + extra + 1 > state.capacity) return error.PartTooLong;
        budget = @min(budget, state.capacity - prompt.len - extra);
        var rules: requests.Rules = .of(request, prompt, eos);
        if (rules.guard) |*g| g.expected = own;

        // the previous part's rows this one takes, before the state is reset
        var saved: ?Saved = null;
        defer if (saved) |*s| s.deinit();
        if (copy) |t| {
            const p = prev.?;
            saved = try .init(state.*, t.copy_len + text.len);
            // SGLang's `_apply_segment`: the marker's row, then the copied image rows
            try saved.?.take(state.*, 0, t.marker, 1);
            try saved.?.take(state.*, 1, t.start, t.copy_end - t.start);
            try saved.?.take(state.*, t.copy_len, p.prompt_len + (p.out.len - text.len), text.len);
        }
        state.reset();
        const runs = try imageRuns(arena, doc, part, prompt, copy);
        if (runs.runs.len > 0) state.image = .{ .features = runs.features, .runs = runs.runs };
        try state.pagedFrom(prompt, documents.pageBase(part, copy != null));
        if (copy) |t| {
            try saved.?.give(state.*, 0, 0, t.copy_len);
            try forward_file.summarize(m, state.*, b, 0, t.copy_len);
            state.pos = t.copy_len;
        }
        while (state.pos < prompt.len) {
            if (sink.cancelled(sink.ctx)) return cancel(&outcome);
            try decoding.promptWindow(draft.namespace, m, state, b, prompt);
        }
        try graphs.prepare(m, state, b, 1, .last);
        if (i == 0) {
            outcome.prefilled_ns = cuda.monotonic();
            sink.prefilled(sink.ctx);
        }

        // the reply: `out` what the part wrote (SGLang's output_ids, the forced tokens first), `seen` what the
        // rules read
        var out: std.ArrayList(u32) = .empty;
        var seen: std.ArrayList(u32) = .empty;
        try seen.ensureTotalCapacity(arena, prompt.len + extra + budget);
        for (prompt) |id| seen.appendAssumeCapacity(@intCast(id));
        if (copy != null) {
            try saved.?.give(state.*, copy.?.copy_len, prompt.len, text.len);
            try forward_file.summarize(m, state.*, b, prompt.len, text.len);
            state.carried(text.len);
            try out.appendSlice(arena, text);
            try out.append(arena, page);
            seen.appendSliceAssumeCapacity(text);
            seen.appendAssumeCapacity(page);
            if (sink.give(sink.ctx, @intCast(page))) return cutAt(&outcome);
            const step = [_]i32{@intCast(page)};
            try graphs.forwardScoring(m, state, b, &step, .last);
            try forward_file.commit(m, state, b.*, &step, 1);
            outcome.reused += 1;
        }
        const on_marker = ahead > 0 or copy != null;
        const last = i + 1 == parts.len;
        var stopped = false;
        var made: usize = 0;
        while (made < budget) {
            try forward_file.logitsRow(b.*, 0, r.logits);
            rules.apply(seen.items, prompt.len, r.logits);
            const token: u32 = @intCast(try sink.choose(sink.ctx, r.logits, state.pos));
            made += 1;
            if (std.mem.indexOfScalar(u32, request.eos, token) != null) {
                stopped = true;
                if (last) outcome.end = token;
                break;
            }
            if (on_marker and token == page and std.mem.count(u32, out.items, &.{page}) >= own) {
                stopped = true;
                break;
            }
            try out.append(arena, token);
            seen.appendAssumeCapacity(token);
            if (sink.give(sink.ctx, @intCast(token))) {
                outcome.made += made;
                return cutAt(&outcome);
            }
            if (sink.cancelled(sink.ctx)) return cancel(&outcome);
            if (made == budget) break;
            const step = [_]i32{@intCast(token)};
            try graphs.forwardScoring(m, state, b, &step, .last);
            try forward_file.commit(m, state, b.*, &step, 1);
        }
        remaining -= made;
        outcome.made += made;
        outcome.parts += 1;
        if (!stopped) all_stopped = false;
        if (parts.len > 1) std.log.info("part {d} of {d}: pages {d} to {d}{s}, prompt {d} rows, {d} tokens, {s}", .{ i + 1, parts.len, part.lo + 1, part.hi, if (copy != null) " (took over the previous part's)" else "", prompt.len, made, if (stopped) "stopped" else "out of tokens" });
        prev = .{ .ids = prompt, .ahead = ahead, .out = out.items, .stopped = stopped, .prompt_len = prompt.len };
    }
    outcome.finish = if (all_stopped and outcome.parts == parts.len) .stop else .length;
    if (outcome.finish != .stop) outcome.end = null;
    return outcome;
}

/// What the next part may take over: the previous part's prompt as the state holds it, its lookahead pages, what
/// it wrote (the forced tokens first, no end token and no extra marker), whether it ended by a stop.
const Previous = struct { ids: []const i32, ahead: u32, out: []const u32, stopped: bool, prompt_len: usize };

fn cancel(outcome: *Outcome) Outcome {
    outcome.finish = .cancelled;
    outcome.end = null;
    return outcome.*;
}

fn cutAt(outcome: *Outcome) Outcome {
    outcome.finish = .cut;
    outcome.end = null;
    return outcome.*;
}

const Runs = struct { features: cuda.Buffer, runs: []const forward_file.Image.Run };

/// The pages of the part's prompt whose image rows are read, laid from the document's features: every page of a
/// part that reads its pages; after the copied ones of a part that took over (the page whose last row the copy
/// left, all of it, which `forward.place` lays from the state's position on).
fn imageRuns(a: Allocator, doc: Document, part: documents.Part, prompt: []const i32, copy: ?documents.TailCopy) !Runs {
    const p = try documents.parse(a, prompt, @intCast(page), @intCast(image)) orelse return error.Unpaged;
    // the prompt's page j is the document's page lo + j, or lo + j - 1 after a context page
    const shift: usize = if (copy != null) 1 else 0;
    const first: usize = if (copy) |t| (if (t.straddle) 1 else 1 + t.copied_pages) else 0;
    if (first >= p.pages()) return .{ .features = doc.features, .runs = &.{} };
    const runs = try a.alloc(forward_file.Image.Run, p.pages() - first);
    for (runs, first..) |*run, j| run.* = .{ .first = p.starts[j] + 1, .rows = p.ends[j] - p.starts[j] - 1 };
    const lo = part.lo + first - shift;
    const hi = part.lo + p.pages() - shift;
    const view = doc.features.span(doc.firsts[lo], doc.firsts[hi] - doc.firsts[lo]);
    return .{ .features = .{ .view = view }, .runs = runs };
}

/// Rows of every layer's keys, values and index keys held aside while the state is reset: a part's copied rows.
const Saved = struct {
    keys: cuda.Buffer,
    values: cuda.Buffer,
    idx_k: cuda.Buffer,
    rows: usize,

    fn init(state: forward_file.State, rows: usize) !Saved {
        const layers = state.keys.len;
        const k = state.keys[0].view;
        const d = &state.dash;
        const held = layers * @max(rows, 1);
        const shape = [_]usize{ held, @intCast(k.size[1]), @intCast(k.size[2]) };
        var keys: cuda.Buffer = try .init(k.dtype, &shape);
        errdefer keys.deinit();
        var values: cuda.Buffer = try .init(k.dtype, &shape);
        errdefer values.deinit();
        const idx_k: cuda.Buffer = try .init(d.idx_k[0].view.dtype, &.{ held, @intCast(d.idx_k[0].view.size[1]) });
        return .{ .keys = keys, .values = values, .idx_k = idx_k, .rows = @max(rows, 1) };
    }

    fn deinit(self: *Saved) void {
        self.keys.deinit();
        self.values.deinit();
        self.idx_k.deinit();
    }

    /// The state's rows `from` .. `from + count - 1` to the held rows `at` on, every layer.
    fn take(self: *Saved, state: forward_file.State, at: usize, from: usize, count: usize) !void {
        if (count == 0) return;
        const d = &state.dash;
        for (state.keys, state.values, d.idx_k, 0..) |k, v, x, l| {
            try self.keys.copyRows(l * self.rows + at, k, from, count);
            try self.values.copyRows(l * self.rows + at, v, from, count);
            try self.idx_k.copyRows(l * self.rows + at, x, from, count);
        }
    }

    /// The held rows `at` .. `at + count - 1` to the state's rows `to` on, every layer.
    fn give(self: *Saved, state: forward_file.State, at: usize, to: usize, count: usize) !void {
        if (count == 0) return;
        const d = &state.dash;
        for (state.keys, state.values, d.idx_k, 0..) |k, v, x, l| {
            try k.copyRows(to, self.keys, l * self.rows + at, count);
            try v.copyRows(to, self.values, l * self.rows + at, count);
            try x.copyRows(to, self.idx_k, l * self.rows + at, count);
        }
    }
};
