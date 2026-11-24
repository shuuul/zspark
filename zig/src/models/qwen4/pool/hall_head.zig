//! The head's pass of the pool (`hall_engine.zig`'s `Engine`): the MTP head over the seats' rows, each span
//! from its `pos`; the rows embedded, the head's attention layer and its selection run over them, the pass's
//! scored rows taken in one launch of the projection (`scoreRows`), a row read back (`fetch`). On rank 0 of
//! two with a split head, rank 1 is told each pass and runs it too.

const std = @import("std");
const cuda = @import("../../../cuda/cuda.zig");
const forms = @import("../../../decode/forms.zig");
const attention = @import("../blocks/attention.zig");
const forward = @import("../forward.zig");
const gdn = @import("../blocks/gdn.zig");
const hall_engine = @import("hall_engine.zig");
const hall_seats = @import("hall_seats.zig");
const hc = @import("../blocks/hc.zig");
const moe = @import("../blocks/moe.zig");
const mtp = @import("../mtp.zig");
const rounds = @import("rounds.zig");
const stage_file = @import("rounds_stage.zig");
const Engine = hall_engine.Engine;
const Span = hall_seats.Span;
const viewOf = attention.viewOf;

/// Scores rows `rows` of the head's window, one for each seat of `seats`, in one launch of the projection
/// (its weight is read once whatever the rows: 0.5 ms a launch on the draft head, spec 078): a seat's
/// logits into its row of `scored`, and what a draft's choice reads of them into its `picked`
/// (`mtp.pick`, a launch a row), so the host reads a kilobyte a draft and the row itself only where that
/// does not settle the choice. A row's bits are those of a launch of it alone. Rows or seats that do not
/// follow each other are gathered into `score_in` and placed from `score_out`.
pub fn scoreRows(self: *Engine, rows: []const usize, seats: []const usize) !void {
    const n = rows.len;
    std.debug.assert(n == seats.len);
    if (n == 0) return;
    // before a launch: a rank that changes waits for the device
    if (self.rule.on_device) for (seats) |slot| if (self.seats[slot] != null) try askRank(self, slot);
    const mixed = self.hb.?.layer.mixed;
    const scored = self.scored.?;
    var x = mixed.span(rows[0], 1);
    if (following(rows)) x = mixed.span(rows[0], n) else if (n > 1) {
        for (rows, 0..) |row, i| try self.score_in.?.copyRows(i, mixed, row, 1);
        x = self.score_in.?.rows(n);
    }
    // a split head (two ranks): each rank's part of the ids, laid in its columns of the seats' rows
    if (self.ranks) |link| if (self.m.mtp.?.split != null) {
        try link.scoreHead(self, x, seats);
        if (self.rule.on_device) for (seats) |slot| if (self.seats[slot] != null) try mtp.pick(scored.span(slot, 1), self.picked[slot].view);
        return;
    };
    const placed = following(seats);
    const out = if (placed) scored.span(seats[0], n) else self.score_out.?.rows(n);
    if (self.m.mtp.?.draft) |d| try d.linear.forward(x, out) else try self.m.head.forward(x, out);
    for (seats, 0..) |slot, i| {
        if (!placed) try scored.copyRows(slot, self.score_out.?, i, 1);
        if (self.rule.on_device) try mtp.pick(scored.span(slot, 1), self.picked[slot].view);
    }
}

fn following(list: []const usize) bool {
    for (list[1..], list[0 .. list.len - 1]) |next, one| if (next != one + 1) return false;
    return true;
}

/// Before seat `slot`'s row is scored: its `picked` asked to the rank its sampling needs.
fn askRank(self: *Engine, slot: usize) !void {
    const settings = self.seats[slot].?.settings;
    if (self.asked[slot].settings) |held| if (std.meta.eql(held, settings)) return;
    const rank = mtp.Rule.rank(settings, @intCast(self.scored.?.view.size[1]), self.m.shape.vocab);
    // a launch that reads the rank may still be running: the settings change between requests
    try cuda.synchronize();
    try mtp.Picked.ask(self.picked[slot], @max(rank, 1));
    self.asked[slot] = .{ .rank = rank, .settings = settings };
}

/// Seat `index`'s scored row read back into its host row. Returns the logits the head scored, in the
/// order of their ids: the whole row, or a draft head's columns (`mtp.Head.fetchLogits`).
pub fn fetch(self: *Engine, index: usize) ![]const u16 {
    try cuda.synchronize();
    const scored = self.scored.?;
    const width: usize = @intCast(scored.view.size[1]);
    const ids = self.m.mtp.?.scored() orelse {
        try scored.downloadAt(index * width * 2, std.mem.sliceAsBytes(self.head_rows[index]));
        return self.head_rows[index];
    };
    try scored.downloadAt(index * width * 2, std.mem.sliceAsBytes(self.columns));
    for (ids, self.columns) |id, bits| self.head_rows[index][id] = bits;
    return self.columns;
}

/// The head over `spans`, each from its `pos`. On rank 0 of two with a split head rank 1 is told first,
/// and runs it too (`hall_ranks.follow`).
pub fn headPass(self: *Engine, spans: []const Span, chained: bool) !void {
    if (spans.len == 0) return;
    if (self.ranks) |link| try link.tellHead(self, spans);
    var total: usize = 0;
    for (spans) |span| {
        std.debug.assert(span.n >= 1 and span.tokens.len == span.n);
        total += span.n;
    }
    std.debug.assert(total <= self.ids.len);
    var at: usize = 0;
    for (spans) |span| {
        try self.gathered.?.copyRows(at, span.source, span.source_row, span.n);
        for (span.tokens, 0..) |token, i| {
            self.ids[at + i] = token;
            self.sid_host[at + i] = @intCast(span.slot);
            self.pos_host[at + i] = @intCast(span.pos + i);
        }
        at += span.n;
    }
    try self.hb.?.layer.ids.uploadAt(0, std.mem.sliceAsBytes(self.ids[0..total]));
    try self.head.?.sid.uploadAt(0, std.mem.sliceAsBytes(self.sid_host[0..total]));
    try self.head.?.posr.uploadAt(0, std.mem.sliceAsBytes(self.pos_host[0..total]));
    @memset(self.seg_host, 0);
    at = 0;
    for (spans) |span| {
        const base = span.slot * 4;
        self.seg_host[base] = @intCast(at);
        self.seg_host[base + 1] = @intCast(span.n);
        self.seg_host[base + 2] = @intCast(span.n);
        self.seg_host[base + 3] = @intCast(span.pos);
        at += span.n;
    }
    try self.head.?.seg.upload(std.mem.sliceAsBytes(self.seg_host));
    if (graphs(self, total)) {
        var end: usize = 1;
        for (spans) |span| end = @max(end, span.pos + span.n);
        try launchGraph(self, total, forward.Graphs.bucket(self.buffers, end));
    } else {
        try embedHead(self, total);
        try headLayer(self, spans, total, null);
    }
    askAhead(self, spans);
    at = 0;
    var rows: [64]usize = undefined;
    var seats: [64]usize = undefined;
    var nscore: usize = 0;
    for (spans) |span| {
        const last = at + span.n - 1;
        if (span.score) {
            rows[nscore] = last;
            seats[nscore] = span.slot;
            nscore += 1;
        }
        try self.held.?.copyRows(span.slot, self.hb.?.layer.h, last, 1);
        self.slots[span.slot].head_len += span.n;
        if (chained) self.seats[span.slot].?.drafted_rows += span.n;
        at += span.n;
    }
    try scoreRows(self, rows[0..nscore], seats[0..nscore]);
}

/// The spans' slots' next windows asked of `Engine.ahead` while the pass runs (spec 144 WS-10), on each rank
/// that runs it: a span the head takes in from the round (its rows from `buffers.h`) ends on the token its next
/// window begins with; a chained one's token is the draft that window goes on with. A window known no further
/// than that, or not at all, is gathered when it is staged.
fn askAhead(self: *Engine, spans: []const Span) void {
    if (!forward.gather_ahead) return;
    if (self.ahead == null) {
        for (self.m.layers) |layer| {
            if (layer.ple != null) break;
        } else return;
        self.ahead = stage_file.Ahead.init(self.gpa, self.m, self.slots.len) catch return;
    }
    const ahead = &self.ahead.?;
    for (spans) |span| {
        if (span.source.view.ptr == self.buffers.h.view.ptr) {
            ahead.begin(self.m, self.slots, span.slot, span.tokens[span.n - 1]);
        } else for (span.tokens) |token| ahead.extend(self.m, self.slots, span.slot, token);
    }
}

/// Whether a pass of `rows` rows runs as a captured graph: one rank's (a split head tells rank 1 its passes,
/// and its partials meet over the link), not eager, and under `gdn.wide_rows` as the trunk's rounds are, so
/// that a prompt chunk's pass, whose rows are its own, captures nothing. A key is captured on its first pass,
/// as a round's past the first bucket is (`captureFirst` takes none), once a key inside a request.
fn graphs(self: *const Engine, rows: usize) bool {
    return self.ranks == null and !self.eager and rows < gdn.wide_rows;
}

/// The device's part of a pass of `rows` rows, from the embedding to the head's layer's last launch, as the
/// graph keyed by the rows and the bucket `context`, captured on its first use (the trunk's rounds'
/// `rounds.Graphs`, on their stream): everything the pass reads of its spans is in the tables uploaded before
/// it, and the selection reads the bucket's keys (`rounds.selectFixed`), as a captured round's does. A pass
/// launched kernel by kernel left 131 us of idle between them (spec 102 WS-02's trace).
fn launchGraph(self: *Engine, rows: usize, context: usize) !void {
    const key: rounds.Graphs.Key = .{ .rows = rows, .context = context };
    // The engine has reached its stable address before a head pass can borrow the trunk's stream.
    if (self.head_graphs == null) self.head_graphs = .initBorrowed(self.gpa, self.graphs.cache.stream());
    try self.head_graphs.?.run(key, captured, .{ self, rows, context });
}

fn captured(self: *Engine, rows: usize, context: usize) !void {
    try embedHead(self, rows);
    try headLayer(self, &.{}, rows, context);
}

fn embedHead(self: *Engine, rows: usize) !void {
    const s = self.m.shape;
    const head = self.m.mtp.?;
    const hb = &self.hb.?;
    const e = hb.e.rows(rows);
    try hc.embed(hb.layer.ids.rows(rows), self.m.embed.view, e, 1);
    try mtp.rmsnorm(e, head.norm_embedding.view, hb.e_normed.rows(rows), s.eps);
    try head.fc_embedding.forward(hb.e_normed.rows(rows), hb.eo.rows(rows));
    try mtp.rmsnorm(self.gathered.?.rows(rows), head.norm_hidden.view, hb.hn.rows(rows), s.eps);
    var each = hb.hn.rows(rows);
    each.size[0] = @intCast(rows * s.streams);
    each.size[1] = @intCast(s.hidden);
    try head.fc_hidden.forward(each, hb.hs.rows(rows * s.streams));
    try mtp.addStreams(hb.eo.rows(rows), hb.hs.rows(rows * s.streams), hb.layer.h.rows(rows), s.streams);
}

/// `bound` is the context bucket a captured pass was keyed by, as `rounds.compute`'s: the spans are then not
/// read. Null is the direct path: the exact end, a selection a span.
fn headLayer(self: *Engine, spans: []const Span, rows: usize, bound: ?usize) !void {
    // beside a share: the whole head's sizes on rank 0, or a split head's half, whose partials meet
    const m = mtp.over(self.m);
    const parts: ?rounds.Parts = if (self.ranks) |link| if (m.mtp.?.split != null) link.parts() else null else null;
    const head = m.mtp.?;
    const mtp_layer = head.layer;
    const weights = switch (mtp_layer.block) {
        .attention => |block| block,
        .delta_net => unreachable,
    };
    const shared = m.attention.?;
    const shape = m.shape;
    const b = &self.hb.?.layer;
    const h = b.h.rows(rows);
    // each write-back with the norm after it in one launch where the session's forward joins them (`hc.joins`)
    const none: hc.Pending = .none;
    if (!hc.joins(none, rows)) try hc.writeback(h, b.pss.view, none, shape.streams);
    try forward.readoutAfter(mtp_layer.attn_hc, b.*, rows, shape, b.inject_attn, if (hc.joins(none, rows)) none else null);
    const scratch = &b.attention.?;
    try scratch.poison();
    if (scratch.selection) |selection| {
        if (bound != null) try selection.sparse.zeroLaunched() else try selection.sparse.zero();
    }
    const mixed = b.mixed.rows(rows);
    const pa = scratch.pa.rows(rows);
    try weights.proj.forward(mixed, pa);
    var context: usize = bound orelse 1;
    var blocks: usize = 1;
    if (bound != null) blocks = @max(1, rows / shared.shape.ratio + 2);
    for (spans) |span| {
        const end = span.pos + span.n;
        if (end > context) context = end;
        blocks = @max(blocks, span.n / shared.shape.ratio + 2);
    }
    const pool = self.slots[0].head.?;
    // the queries as the head's cache is stored, which may be bf16 beside a coded trunk
    const mine = scratch.queries(pool);
    const hr = &self.head.?;
    try cuda.launched(cuda.kernels.zs_qwen4_attn_prep_rows(&hr.sid.rows(rows), &hr.tables.view, &hr.bases.view, &pa, &hr.posr.rows(rows), &weights.q_scale.view, &weights.k_scale.view, &weights.index_q_scale.view, &shared.rotary.inv_freq.view, &mine.q.rows(rows), &pool.k.view, &pool.v.view, &scratch.iq.rows(rows), &pool.index_keys.view, viewOf(&mine.scale), viewOf(&pool.k_scale), viewOf(&pool.v_scale), shared.shape.eps, @intCast(context), &self.staged.ropes.view, shared.shape.sections[1], shared.shape.sections[2]));
    try cuda.launched(cuda.kernels.zs_qwen4_attn_pool_slots(&hr.seg.view, &hr.ptrs.view, 2, 1, 0, &hr.bases.view, &pool.index_keys.view, &weights.index_k_scale.view, &shared.rotary.inv_freq.view, shared.shape.eps, @intCast(shared.shape.ratio), @intCast(blocks), &self.staged.ropes.view, shared.shape.sections[1], shared.shape.sections[2]));
    const listed = if (bound) |fixed| try rounds.selectFixed(self.slots, hr, scratch, shared.shape, rows, fixed, .head, if (self.graphs.gem) |*held| held else null) else try select(self, spans, scratch, shared.shape);
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(shared.shape.head_dim)));
    const lists: ?attention.Selection = if (listed) scratch.selection else null;
    const ids = if (lists) |*selection| &selection.ids.view else null;
    const nk = if (lists) |*selection| &selection.nk.view else null;
    const sparse = if (lists) |*selection| &selection.sparse.view else null;
    try cuda.launched(cuda.kernels.zs_attn_attend_rows(&hr.sid.rows(rows), &hr.tables.view, &hr.bases.view, &mine.q.rows(rows), &pool.k.view, &pool.v.view, &hr.posr.rows(rows), &scratch.partial_o.rows(rows), &scratch.partial_m.rows(rows), &scratch.partial_l.rows(rows), &scratch.attn.rows(rows), ids, nk, sparse, viewOf(&mine.scale), viewOf(&pool.k_scale), viewOf(&pool.v_scale), scale, @intCast(context)));
    try attention.gate(scratch.attn.rows(rows), pa, scratch.gated.rows(rows));
    var pending: hc.Pending = undefined;
    if (parts) |share| {
        try weights.o.forward(scratch.gated.rows(rows), share.part(rows));
        pending = try share.landed(rows, b.inject_attn.view);
    } else {
        try weights.o.forward(scratch.gated.rows(rows), scratch.branch.rows(rows));
        pending = .{ .branch = .{ .branch = scratch.branch.rows(rows), .inject = b.inject_attn.view } };
    }
    if (!hc.joins(pending, rows)) try hc.writeback(h, b.pss.view, pending, shape.streams);
    try forward.readoutAfter(mtp_layer.mlp_hc, b.*, rows, shape, b.inject_mlp, if (hc.joins(pending, rows)) pending else null);
    try moe.forward(mtp_layer.moe, b.moe, b.mixed.rows(rows), rows, forms.none);
    const mlp = if (parts) |share| try share.folded(b.moe, rows, shape.top + 1, b.inject_mlp.view) else moe.pending(b.moe, b.inject_mlp.view);
    try forward.finish(m, head.mixer, 1, b, rows, mlp, null);
}

fn select(self: *Engine, spans: []const Span, scratch: *attention.Scratch, shape: attention.Shape) !bool {
    const selection = if (scratch.selection) |*held| held else return false;
    var listed: bool = false;
    var at: usize = 0;
    for (spans) |span| {
        const end = span.pos + span.n;
        if (shape.selects(end)) {
            listed = true;
            const heads = shape.index_heads;
            const queries = cuda.View.init(scratch.iq.span(at, span.n).ptr.?, .bf16, &.{ span.n * heads, shape.index_dim });
            const pooled = self.slots[span.slot].head.?.pooled;
            const keys = cuda.View.init(pooled.view.ptr.?, .bf16, &.{ end / shape.ratio, shape.index_dim });
            const pos = self.head.?.posr.span(at, 1);
            try attention.select(shape, selection.*, queries, keys, pos, at, span.n, end, self.slots[span.slot].table.device.view);
        }
        at += span.n;
    }
    return listed;
}
