//! The MTP head as `decode/draft.zig`'s `Drafter` (`cuda_mtp.zig`): one attention layer and one MoE over
//! [rmsnorm(embed(tok)) | rmsnorm(hidden)] by `eh_proj`, the shared `lm_head` scored for the token after,
//! and the greedy draft its argmax — over the whole head, TensorFold's `draft_ids.txt` subset being
//! Lightning's vocabulary (ids 100 to 141 are Transcribe's `<spk:>`/`<ev:>`/`<sty:>` tokens, so the list
//! scores rows of the wrong tokens; `model.zig`).
//!
//! A `draft` call absorbs the window's kept rows (`tokens` the token after each, `streams` their final
//! residual rows) at the head's own position, then runs one level for each draft `out` asks: level 0
//! absorbs the kept rows only (their K/V written, nothing scored); level 1 absorbs them and drafts the
//! token after the last; level j > 1 drafts after draft j - 1, its row the last draft's hidden and token.
//! The head's KV cache is the sequence's own (`state.mtp_cache`), its position `state.mtp_pos`: a call's
//! absorbed rows advance it by `keep`, and each level's row is written at that position plus the drafts
//! before it, so a rolled-back window's rows are never read.
//!
//! `Head` holds the scratch the levels launch in, made once a model is loaded; the weights are the model's
//! `mtp` block and `draft` head, the state the sequence's own.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const config_file = @import("config.zig");
const attention_file = @import("attention.zig");
const moe_file = @import("moe.zig");
const model_file = @import("model.zig");
const state_file = @import("state.zig");
const glue = @import("glue.zig");
const draft_file = @import("draft.zig");
const masks_file = @import("../../decode/masks.zig");
const sampling = @import("../../decode/sampling.zig");
const ranks_forward = @import("ranks_forward.zig");
const exchange = @import("../../ranks/ranks.zig");
const ranks_file = @import("ranks.zig");

/// The drafts a window may carry beside the pending row (`draft.forward_max_rows - 1`).
pub const max_chain = draft_file.forward_max_rows - 1;

/// The MTP head's launch state: the model by value (its `mtp` block and `draft` head read) beside the
/// scratch a level's calls run in. `bind` re-reads the model after load (the draft count sizes the logits).
pub const Head = struct {
    m: model_file.Model,
    /// int32 [max_chain + 1]: level j's position word (the row's absolute position, meta[j]).
    meta: cuda.Buffer,
    /// int32 [max(rows, max_chain + 1)]: absorbed prompt tokens, then each chained level's draft.
    /// A prompt window can be wider than the verification window.
    ids: cuda.Buffer,
    /// bf16 [rows, hidden]: the kept rows' final residual streams, copied from `streams` (a level's `hin`).
    hin: cuda.Buffer,
    /// bf16 [1, hidden]: a chained draft's hidden (the level before's output row) for the level after.
    out: cuda.Buffer,
    /// bf16 [rows, hidden]: the tokens' embeddings.
    emb: cuda.Buffer,
    /// bf16 [rows, 2 * hidden] and fp32 [rows, 2 * hidden / 64]: the concat norms' product and sums.
    cat: cuda.Buffer,
    cxs: cuda.Buffer,
    /// bf16 [rows, hidden]: `eh_proj`'s product, the attention block's residual base.
    x: cuda.Buffer,
    /// bf16 [rows, hidden]: the attention's o_proj product (the residual's delta).
    delta: cuda.Buffer,
    /// bf16 [rows, hidden]: the add-RMSNorm's h (the residual's new row).
    hn: cuda.Buffer,
    /// bf16 [rows, hidden] and fp32 [rows, hidden / 64]: the norms' normed rows and group sums.
    y: cuda.Buffer,
    xs: cuda.Buffer,
    /// bf16 [1, hidden]: the last row's normed hidden, the draft head's input.
    row: cuda.Buffer,
    /// bf16 [1, draft_count]: the draft head's logits of the last row.
    logits: cuda.Buffer,
    /// The most rows an absorb takes (the window's buffers').
    rows: usize,
    /// The attention layer's and MoE's own scratch.
    asc: attention_file.Scratch,
    esc: moe_file.Scratch,

    /// The held ranks' partials when the head runs on a rank of two (`ranks_forward.Ranks`, set by
    /// `bindShared`): the attention's o_proj and the MoE's slots are this rank's half, met by the gather
    /// and summed into `delta`/`row` as the serial head's; the draft head's logits gather their halves.
    shared: ?*ranks_forward.Ranks = null,

    /// `rows` the most a window keeps; `capacity` the sequence's positions (the cache's own).
    /// `shapes` the mixers' shapes — `ranks.Shapes`'s halves on a rank of two, `Shape.of`'s whole
    /// otherwise.
    pub fn init(c: *const config_file.Config, rows: usize, capacity: usize) cuda.Error!Head {
        return initS(c, rows, capacity, .of(c.backbone, null));
    }

    /// `init` on `shapes`: the rank's attention and MoE scratch at its half of the heads and expert
    /// width (`ranks.Shapes.of` of a `Share`).
    pub fn initS(c: *const config_file.Config, rows: usize, capacity: usize, shapes: ranks_file.Shapes) cuda.Error!Head {
        const b = c.backbone;
        var h: Head = undefined;
        h.m = undefined; // `bind` gives it once the model is loaded
        h.shared = null;
        h.rows = rows;
        h.meta = try .init(.i32, &.{max_chain + 1});
        errdefer h.meta.deinit();
        h.ids = try .init(.i32, &.{@max(rows, max_chain + 1)});
        errdefer h.ids.deinit();
        h.hin = try .init(.bf16, &.{ rows, b.hidden });
        errdefer h.hin.deinit();
        h.out = try .init(.bf16, &.{ 1, b.hidden });
        errdefer h.out.deinit();
        h.emb = try .init(.bf16, &.{ rows, b.hidden });
        errdefer h.emb.deinit();
        h.cat = try .init(.bf16, &.{ rows, 2 * b.hidden });
        errdefer h.cat.deinit();
        h.cxs = try .init(.f32, &.{ rows, 2 * b.hidden / 64 });
        errdefer h.cxs.deinit();
        h.x = try .init(.bf16, &.{ rows, b.hidden });
        errdefer h.x.deinit();
        h.delta = try .init(.bf16, &.{ rows, b.hidden });
        errdefer h.delta.deinit();
        h.hn = try .init(.bf16, &.{ rows, b.hidden });
        errdefer h.hn.deinit();
        h.y = try .init(.bf16, &.{ rows, b.hidden });
        errdefer h.y.deinit();
        h.xs = try .init(.f32, &.{ rows, b.hidden / 64 });
        errdefer h.xs.deinit();
        h.row = try .init(.bf16, &.{ 1, b.hidden });
        errdefer h.row.deinit();
        h.logits = try .init(.bf16, &.{1});
        h.asc = try .init(shapes.attention, @max(rows, attention_file.window_rows), capacity);
        errdefer h.asc.deinit();
        h.esc = try .init(shapes.moe, @max(rows, attention_file.window_rows), false);
        return h;
    }

    /// `bind` on a rank of two: the head's partials gather through `shared` (`ranks_forward.Ranks` of
    /// the decode's buffers; its `parts` is at least `max_chain + 1` rows wide — the main buffers').
    pub fn bindShared(h: *Head, m: model_file.Model, shared: *ranks_forward.Ranks) (cuda.Error || error{NoMtpHead})!void {
        try h.bind(m);
        h.shared = shared;
    }

    /// The model's `mtp` block and `draft` head bound: the logits buffer made to the draft count.
    pub fn bind(h: *Head, m: model_file.Model) (cuda.Error || error{NoMtpHead})!void {
        if (m.mtp == null or m.draft == null) return error.NoMtpHead;
        h.m = m;
        if (h.logits.view.count() != m.draft.?.count) {
            h.logits.deinit();
            h.logits = try .init(.bf16, &.{ 1, m.draft.?.count });
        }
    }

    pub fn deinit(h: *Head) void {
        h.meta.deinit();
        h.ids.deinit();
        h.hin.deinit();
        h.out.deinit();
        h.emb.deinit();
        h.cat.deinit();
        h.cxs.deinit();
        h.x.deinit();
        h.delta.deinit();
        h.hn.deinit();
        h.y.deinit();
        h.xs.deinit();
        h.row.deinit();
        h.logits.deinit();
        h.asc.deinit();
        h.esc.deinit();
        h.* = undefined;
    }

    /// One level's forward over `rows` input rows (`hin`'s first `rows` and `ids`' `rows` from `id_at`):
    /// embed, concat-norms with the hidden rows, eh_proj, the attention block's norm, qkv, K/V at
    /// `meta[meta_i]`'s position (the first row's own, `pos` on the host) and attend, o_proj into `delta`.
    /// With `tail`, the last row alone through the MoE's norm and experts and the draft head's logits.
    pub fn forward(h: *Head, state: *state_file.State, id_at: usize, rows: usize, tail: bool, meta_i: usize, pos: usize) !void {
        const c = h.m.config.backbone;
        const w = h.m.mtp.?;
        const hid = h.hin.rows(rows);
        const x = h.x.rows(rows);
        const y = h.y.rows(rows);
        const xs = h.xs.rows(rows);
        const emb = h.emb.rows(rows);
        const cat = h.cat.rows(rows);
        const cxs = h.cxs.rows(rows);
        try cuda.launched(cuda.kernels.zs_embed(&h.ids.span(id_at, rows), &h.m.embed.view, &emb, 1));
        try glue.concatNorms(emb, hid, w.enorm.view, w.hnorm.view, cat, cxs, c.eps);
        try w.eh_proj.forward(cat, x);
        // the attention block: h = x (the residual base; the first norm's delta is none), y = rmsnorm(x)*norm
        try glue.addRmsnorm(x, null, w.attn_norm.view, h.hn.rows(rows), y, xs, c.eps);
        if (h.shared) |shared| {
            // a rank's attention block: o_proj's fp32 partial into `parts[rank]`, met and summed once —
            // the serial block's bf16 delta's bits (`ranks_forward`'s fold)
            const mine = exchange.partial(shared.parts.view, shared.rank, rows);
            if (rows == 1)
                try w.attn.window(state.mtp_cache.?, h.asc, y, h.meta.span(meta_i, 1), @intCast(pos + 1), mine)
            else
                try w.attn.prompt(state.mtp_cache.?, h.asc, y, h.meta.span(meta_i, 1), @intCast(pos + rows), mine);
            if (shared.gather != .loopback) try shared.gather.run(exchange.gathered(shared.parts.view, rows));
            try exchange.sumRows(exchange.gathered(shared.parts.view, rows), 0, h.delta.rows(rows));
        } else {
            if (rows == 1)
                try w.attn.window(state.mtp_cache.?, h.asc, y, h.meta.span(meta_i, 1), @intCast(pos + 1), h.delta.rows(1))
            else
                try w.attn.prompt(state.mtp_cache.?, h.asc, y, h.meta.span(meta_i, 1), @intCast(pos + rows), h.delta.rows(rows));
        }
        if (!tail) return;
        // the last row through the MoE and the draft head (level j's draft)
        const last = rows - 1;
        const xr = h.x.span(last, 1);
        const dr = h.delta.span(last, 1);
        try glue.addRmsnorm(xr, dr, w.moe_norm.view, h.hn.rows(1), y.span(0, 1), xs.span(0, 1), c.eps);
        try moe_file.window(w.moe, h.esc, y.span(0, 1));
        const s = w.moe.shape;
        const yk: cuda.View = .{ .ptr = h.esc.y.view.ptr, .dtype = .f32, .ndim = 3, .size = .{ 1, @intCast(s.slots()), @intCast(s.dims), 0, 0, 0 } };
        if (h.shared) |shared| {
            // the rank's slot rows and shared row folded into its partial, met, summed once
            try glue.moePartial(yk, h.esc.wts.rows(1), exchange.partial(shared.parts.view, shared.rank, 1), s.top);
            if (shared.gather != .loopback) try shared.gather.run(exchange.gathered(shared.parts.view, 1));
            try exchange.sumRows(exchange.gathered(shared.parts.view, 1), 0, h.delta.rows(1));
            try glue.addRmsnorm(h.hn.rows(1), h.delta.rows(1), w.final_norm.view, h.row.rows(1), y.span(0, 1), xs.span(0, 1), c.eps);
        } else {
            try glue.addMoeNorm(h.hn.rows(1), yk, h.esc.wts.rows(1), w.final_norm.view, h.row.rows(1), y.span(0, 1), xs.span(0, 1), c.eps, s.top);
        }
        try h.m.draft.?.linear.forward(h.row.rows(1), h.logits.rows(1));
    }

    /// Level j's draft token after its `forward`: the whole head's argmax (value down, id up on a tie —
    /// what `keyedGreedy`'s rank 0 is over every row the head scores, `draft` head being `lm_head` itself).
    /// `take` writes it into `ids[j]` for the next level's token.
    pub fn take(h: *Head, state: *state_file.State, j: usize) !i32 {
        _ = state;
        const count = h.m.draft.?.count;
        if (h.shared) |shared| {
            // this rank's half of the vocabulary into `row[rank]`, the halves met; the argmax's token id
            // is the global one's — `partial`'s slice offsets carry rank * half already
            const half: usize = @intCast(shared.row.view.size[2]);
            if (count != half or h.logits.view.count() != half) return error.UnexpectedTensor;
            try shared.row.copyAt(shared.rank * half * 2, h.logits, half * 2);
            if (shared.gather != .loopback) try shared.gather.run(.init(shared.row.view.ptr.?, .bf16, &.{ @intCast(shared.row.view.size[0]), 1, @intCast(half) }));
            const all = try std.heap.c_allocator.alloc(u16, count * ranks_file.world);
            defer std.heap.c_allocator.free(all);
            try cuda.synchronize();
            for (0..ranks_file.world) |rank| {
                var piece: cuda.Buffer = try .init(.bf16, &.{ 1, half });
                defer piece.deinit();
                try piece.copyColumns(0, exchange.partial(shared.row.view, rank, 1), 1);
                try piece.downloadAt(0, std.mem.sliceAsBytes(all[rank * half ..][0..half]));
            }
            const tok: [1]i32 = .{try sampling.argmax(all)};
            try h.ids.uploadAt(j * @sizeOf(i32), std.mem.sliceAsBytes(&tok));
            return tok[0];
        }
        const logits = try std.heap.c_allocator.alloc(u16, count);
        defer std.heap.c_allocator.free(logits);
        try cuda.synchronize();
        try h.logits.downloadAt(0, std.mem.sliceAsBytes(logits));
        const tok: [1]i32 = .{try sampling.argmax(logits)};
        try h.ids.uploadAt(j * @sizeOf(i32), std.mem.sliceAsBytes(&tok));
        return tok[0];
    }
};

/// The drafter itself: a `Head` plus the sequence it is bound to (`*State`: the head's cache and position
/// are its own fields), as `draft.Drafter` asks.
pub const Drafter = struct {
    head: *Head,
    state: *state_file.State,

    /// `draft.Drafter` over `self`.
    pub fn drafter(self: *Drafter) draft_file.Drafter {
        return .{ .context = self, .draft = run };
    }

    /// `draft`'s call: absorb the `tokens`' rows (the kept window's), then one level a draft of `out`.
    fn run(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
        _ = guide;
        _ = position;
        const self: *Drafter = @ptrCast(@alignCast(context));
        const h = self.head;
        const st = self.state;
        const keep = tokens.len;
        if (keep == 0 or keep > h.rows) return 0;
        // the kept rows' residual rows into `hin` and their tokens into `ids[0..keep]`
        try h.hin.copyRows(0, .{ .view = streams }, 0, keep);
        try h.ids.uploadAt(0, std.mem.sliceAsBytes(tokens));
        // the levels' position words: absorbed rows at the head's position, level j's chained row at the
        // position after them and the drafts before it (`meta[0]` also serves level 1, which re-runs the
        // kept rows whole)
        var metas: [max_chain + 1]i32 = undefined;
        metas[0] = @intCast(st.mtp_pos);
        for (1..metas.len) |j| metas[j] = @intCast(st.mtp_pos + keep + j - 1);
        try h.meta.upload(std.mem.sliceAsBytes(&metas));
        // level 0: absorb the rows whole (their K/V written at `meta[0]`, nothing scored)
        try h.forward(st, 0, keep, false, 0, st.mtp_pos);
        // level 1 drafts the token after the last kept row: the kept rows again, the last one scored
        const want = @min(out.len, max_chain);
        var made: usize = 0;
        while (made < want) : (made += 1) {
            if (made == 0) {
                try h.forward(st, 0, keep, true, 0, st.mtp_pos);
            } else {
                // chain: level j's row is draft j - 1's hidden and token (the draw wrote ids[j - 1]),
                // its K/V at its own position after the absorbed rows and the drafts before it
                try h.hin.copyRows(0, h.out, 0, 1);
                try h.forward(st, made - 1, 1, true, made, st.mtp_pos + keep + made - 1);
            }
            // keep the level's hidden for the next draft's row
            try h.out.copyRows(0, h.row, 0, 1);
            out[made] = try h.take(st, made);
        }
        st.mtp_pos += keep;
        return made;
    }
};
