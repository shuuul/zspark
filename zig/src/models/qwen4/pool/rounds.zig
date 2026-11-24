//! A round of several streams through `forward` (spec 018 WS-B3). `serve`'s pool calls this
//! once a second stream arrives (`hall_engine.zig`); one stream is `forward.compute`'s (spec 058). A row's bits do not depend on the rows beside it, whichever
//! stream they belong to, so a round equals each stream run alone.
//!
//! One upload a round carries `sid`, `posr`, `seg`, `ptrs` and `scored`. The DeltaNet chain and its replay,
//! the PLE convolution, the window shift and the indexer pooling launch once over the slots; a slot with no
//! rows returns at once. A segment of `gdn.wide_rows` or more, at most one, takes the wide kernels on its own
//! span. Prep and attend are one launch over the mixed rows. The selection GEMM stays one launch a stream
//! on the direct path, because a segment's rows are that stream's contiguous positions.
//!
//! A window of 2 to `gdn.wide_rows` - 1 rows writes no DeltaNet state (spec 144 WS-09), as a session's does
//! (`forward.stores`, specs 089 WS-03 and 096 WS-02): its rows wait in the slot's own scratch (`Slot.held`,
//! `Slot.behind`), and the slot's next chain replays them on the state it loads anyway. A slot's state at `pos`
//! is then `committed`'s; `stage` lands one whose window cannot go on from them (`forward.landsAt`).
//!
//! `Graphs` captures that round for `{rows, context}`. The DeltaNet side, the position and the slot are read
//! from the tables, so one set of graphs serves every slot. A segment of `gdn.wide_rows` or more is not
//! captured. `zs_b16_gemm` rejects zero rows, so the graph holds `slots` GEMMs of the round's row count and an
//! empty slot's pack returns at once.
//!
//! `Slot` is what design section 2 gives a stream, and `Shared` the one scratch. `forward.State` stays the
//! one-stream packing a session holds. The head's cache, where a slot has one, reads the trunk's table:
//! extent `e` is one index and the two lengths are `pos` and `head_len`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("../blocks/attention.zig");
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const forms = @import("../../../decode/forms.zig");
const forward = @import("../forward.zig");
const gdn = @import("../blocks/gdn.zig");
const hc = @import("../blocks/hc.zig");
const model = @import("../model.zig");
const moe = @import("../blocks/moe.zig");
const ple = @import("../blocks/ple.zig");
const state_windows = @import("../blocks/state.zig");

/// One stream's rows in the round: `tokens` at `slot.pos`, of which `commit` keeps the first `keep`.
pub const Segment = struct {
    slot: usize,
    tokens: []const i32,
    keep: usize,
};

/// Rows of the oldest filling prompt a round that already holds replies has room for, beside every slot's
/// window: `hall.rowsFor` sizes a round's buffers by it, and it is `EngineOptions.chunk`'s default.
/// A prompt with no reply beside it stays at `session.promptRows` (2048; the lone session's, before the pool opens, 4096). 512 is spec 018
/// WS-B8: WS-00's row past 64 is 0.74 to 1.16 ms, and spec 031's 512-row prompt window is 483 ms, against
/// 1592 ms at 2048. A chunk this wide is past `gdn.wide_rows`, so the round stays on direct launches.
/// `serve` takes what the replies leave of the model's rows, vLLM's budget (spec 101), or `--prompt-chunk N`
/// (spec 074, WS-04): the replies beside a prompt wait a round, which on the 3.05-bit pack is 66 ms and
/// 1.15 ms a row of the chunk, and the prompt pays the 66 ms once more for each round it is cut into.
pub const prompt_chunk: usize = 512;

/// How many of the `left` prompt rows this round takes. `budget` is `prompt_chunk` while replies are live.
pub fn chunkRows(left: usize, budget: usize) usize {
    return @min(left, budget);
}

/// The replies' windows, then one prompt chunk: the first `chunkRows(prompt.len, budget)` of `prompt`, kept
/// whole, on `prompt_slot`. A budget of 0, or a prompt with nothing left, is the replies alone.
pub fn promptBeside(gpa: Allocator, replies: []const Segment, prompt_slot: usize, prompt: []const i32, budget: usize) Allocator.Error![]Segment {
    const n = chunkRows(prompt.len, budget);
    const out = try gpa.alloc(Segment, replies.len + @intFromBool(n > 0));
    @memcpy(out[0..replies.len], replies);
    if (n > 0) out[replies.len] = .{ .slot = prompt_slot, .tokens = prompt[0..n], .keep = n };
    return out;
}

/// Where a layer's pointers sit in `ptrs`. Column 0 is the extent table, shared by every attention layer and
/// the head. Then each DeltaNet layer's state_in, state_out and conv, each PLE's tail, and each attention
/// layer's pooled keys.
const Columns = struct {
    deltas: usize,
    ples: usize,
    attentions: usize,

    pub fn stride(self: Columns) usize {
        return 1 + 3 * self.deltas + self.ples + self.attentions;
    }

    pub fn delta(self: Columns, layer: usize) usize {
        _ = self;
        return 1 + layer * 3;
    }

    pub fn ple(self: Columns, branch: usize) usize {
        return 1 + 3 * self.deltas + branch;
    }

    pub fn pooled(self: Columns, layer: usize) usize {
        return 1 + 3 * self.deltas + self.ples + layer;
    }

    fn of(m: model.Model) Columns {
        var self: Columns = .{ .deltas = 0, .ples = 0, .attentions = 0 };
        for (m.layers) |layer| {
            self.deltas += @intFromBool(layer.block == .delta_net);
            self.ples += @intFromBool(layer.ple != null);
            self.attentions += @intFromBool(layer.block == .attention);
        }
        return self;
    }
};

/// The round's row tables, uploaded once by `stage`.
pub const Round = struct {
    slots: usize,
    /// Rows the buffers hold. `used` is this round's.
    rows: usize,
    used: usize = 0,
    columns: Columns,
    /// int32 [rows]: the stream a row belongs to.
    sid: cuda.Buffer,
    /// int32 [rows]: the row's position.
    posr: cuda.Buffer,
    /// int32 [slots, 4]: first row, rows, kept rows, first position. A slot with no rows has rows 0.
    seg: cuda.Buffer,
    /// int64 [slots, stride]: device pointers, `Columns`.
    ptrs: cuda.Buffer,
    /// int32 [rows]: the rows the head scores. These tests score every row, 0 .. used - 1.
    scored: cuda.Buffer,
    /// int64 [slots]: each slot's extent table, what prep and attend read. The same addresses as `ptrs` column 0.
    tables: cuda.Buffer,
    /// int64 [slots, 3]: each slot's k, v and indexer keys. One pool repeats the pointer; identity tables do not.
    bases: cuda.Buffer,
    /// int64 [slots]: each slot's rotary table (`attention.Rope`, spec 019), 0 for a slot at its own
    /// positions; prep and pooling read it by the row's slot, so a captured round reads it too.
    ropes: cuda.Buffer,
    /// The features the staged round's image rows take (`forward.Buffers.image_pairs`), a slot's alone.
    image: ?cuda.View = null,
    /// bf16 [DeltaNet layers * (taps - 1), projection]: the last rows of a wide segment's projection, a layer
    /// each. A round with a wide segment writes every layer's projection into the one `projection_wide`, so
    /// what `commit` slides each layer's conv window over is kept here, and the rows before the wide segment
    /// in that layer's short projection (`compute`, `commit`; spec 051).
    tails: cuda.Buffer,
    /// This round's `compute` kept them: one wide segment, the last.
    tailed: bool = false,
    /// int64 [slots, 1 + 4 * DeltaNet layers]: each slot's `behind`, then each layer's `Slot.held` k, v, g and
    /// beta, which the chain reads (`stage` uploads it).
    held: cuda.Buffer,
    /// The states wanted inside this round's segment of slot `wanted_slot`, lowest first, each above the
    /// slot's position and below the segment's end (spec 054): `compute` writes each one's DeltaNet states
    /// and conv inputs, as `forward.State.wanted` has a session's window write them, and the caller takes the
    /// rest before `commit` (`hall_keep.zig`). The caller's; it empties the list after that `compute`, and a
    /// captured round has none.
    wanted: []const forward.Snapshot = &.{},
    wanted_slot: usize = 0,

    pub fn init(m: model.Model, slots: usize, rows: usize) !Round {
        const columns: Columns = .of(m);
        var sid: cuda.Buffer = try .init(.i32, &.{rows});
        errdefer sid.deinit();
        var posr: cuda.Buffer = try .init(.i32, &.{rows});
        errdefer posr.deinit();
        var seg: cuda.Buffer = try .init(.i32, &.{ slots, 4 });
        errdefer seg.deinit();
        var ptrs: cuda.Buffer = try .init(.i64, &.{ slots, columns.stride() });
        errdefer ptrs.deinit();
        var scored: cuda.Buffer = try .init(.i32, &.{rows});
        errdefer scored.deinit();
        var tables: cuda.Buffer = try .init(.i64, &.{slots});
        errdefer tables.deinit();
        // One layer's three pointers sit together, so a graph bakes that layer's address and `stage` fills them all.
        var bases: cuda.Buffer = try .init(.i64, &.{ @max(columns.attentions, 1) * slots, 3 });
        errdefer bases.deinit();
        var ropes: cuda.Buffer = try .init(.i64, &.{slots});
        errdefer ropes.deinit();
        try ropes.zero();
        var tails: cuda.Buffer = try .init(.bf16, &.{ @max(columns.deltas, 1) * (gdn.taps - 1), @max(m.shape.delta_net.projection(), 1) });
        errdefer tails.deinit();
        var held: cuda.Buffer = try .init(.i64, &.{ slots, 1 + 4 * columns.deltas });
        errdefer held.deinit();
        return .{
            .held = held,
            .tails = tails,
            .slots = slots,
            .rows = rows,
            .columns = columns,
            .sid = sid,
            .posr = posr,
            .seg = seg,
            .ptrs = ptrs,
            .scored = scored,
            .tables = tables,
            .bases = bases,
            .ropes = ropes,
        };
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(m: model.Model, slots: usize, rows: usize) usize {
        const of = cuda.Buffer.bytesOf;
        const columns: Columns = .of(m);
        return 3 * of(.i32, &.{rows}) + of(.i32, &.{ slots, 4 }) + of(.i64, &.{ slots, columns.stride() }) + 2 * of(.i64, &.{slots}) +
            of(.i64, &.{ @max(columns.attentions, 1) * slots, 3 }) +
            of(.bf16, &.{ @max(columns.deltas, 1) * (gdn.taps - 1), @max(m.shape.delta_net.projection(), 1) }) +
            of(.i64, &.{ slots, 1 + 4 * columns.deltas });
    }

    pub fn deinit(self: *Round) void {
        self.sid.deinit();
        self.posr.deinit();
        self.seg.deinit();
        self.ptrs.deinit();
        self.scored.deinit();
        self.tables.deinit();
        self.bases.deinit();
        self.ropes.deinit();
        self.tails.deinit();
        self.held.deinit();
        self.* = undefined;
    }
};

/// The scratch one round shares. Replay runs immediately after that layer's chain, so the next layer reuses it.
pub const Shared = struct {
    scratch: gdn.Scratch,

    pub fn init(rows: usize, shape: gdn.Shape) !Shared {
        return .{ .scratch = try .init(rows, shape) };
    }

    pub fn deinit(self: *Shared) void {
        self.scratch.deinit();
        self.* = undefined;
    }
};

/// One stream: its recurrent state, its conv windows, its PLE tails, and its caches behind one table. The
/// caches' arrays are a layer's pool, lent; `pooled` is the stream's own. `head`, where the slot has one, is
/// another pool behind the same table.
pub const Slot = struct {
    pos: usize = 0,
    /// Positions the head's cache has. The trunk's length is `pos`. One table, two lengths.
    head_len: usize = 0,
    /// With rows `behind` the committed side is not the state at `pos`: read that by `committed`.
    recurrent: [][2]cuda.Buffer,
    current: []u1,
    /// Each DeltaNet layer's replay scratch of `heldRows` rows: the rows its committed side does not hold yet
    /// at its front, `behind` of them, as `forward.State.scratch` and `behind` (spec 144 WS-09).
    held: []gdn.Scratch,
    behind: usize = 0,
    conv: []cuda.Buffer,
    ple: []ple.State,
    table: attention.Table,
    caches: []attention.Cache,
    head: ?attention.Cache = null,
    /// The rotary table of an image prompt's sequence (spec 019 WS-05), lent (`hall.Sight.ropes`); null
    /// where no image is served. Cleared, the slot's rows are at their own positions.
    rope: ?*attention.Rope = null,
    /// An image prompt's features while rows of its placeholders are still to run, as `forward.State.image`.
    image: ?forward.Image = null,
    gpa: Allocator,

    /// The address `Round.ropes` holds for the slot: 0 where its rows are at their own positions.
    pub fn ropeAddress(self: Slot) i64 {
        const rope = self.rope orelse return 0;
        return if (rope.length == 0 and rope.delta == 0) 0 else address(rope.view());
    }

    /// `pools` is one a attention layer, shared by every slot. `head_pool` is the head's cache, or null.
    pub fn open(gpa: Allocator, m: model.Model, pools: []const attention.Pool, head_pool: ?*const attention.Pool, capacity: usize) !Slot {
        const columns: Columns = .of(m);
        std.debug.assert(pools.len == columns.attentions);
        const dn = m.shape.delta_net;
        const recurrent = try gpa.alloc([2]cuda.Buffer, columns.deltas);
        errdefer gpa.free(recurrent);
        const current = try gpa.alloc(u1, columns.deltas);
        errdefer gpa.free(current);
        const conv = try gpa.alloc(cuda.Buffer, columns.deltas);
        errdefer gpa.free(conv);
        const held = try gpa.alloc(gdn.Scratch, columns.deltas);
        errdefer gpa.free(held);
        var holding: usize = 0;
        errdefer for (held[0..holding]) |*scratch| scratch.deinit();
        for (held) |*scratch| {
            scratch.* = try .init(heldRows(), dn);
            holding += 1;
        }
        for (recurrent, current, conv) |*pair, *side, *window| {
            for (pair) |*buffer| {
                buffer.* = try .init(.f32, &.{ dn.value_heads, gdn.value_dim, gdn.key_dim });
                try buffer.zero();
            }
            side.* = 0;
            window.* = try .init(.bf16, &.{ 1, gdn.taps - 1, dn.conv() });
            try window.zero();
        }
        const branches = try gpa.alloc(ple.State, columns.ples);
        errdefer gpa.free(branches);
        var made: usize = 0;
        errdefer for (branches[0..made]) |*branch| branch.deinit(gpa);
        var ple_at: usize = 0;
        for (m.layers) |layer| if (layer.ple) |branch| {
            branches[ple_at] = try .init(gpa, branch.weights.shape, branch.eos);
            made += 1;
            ple_at += 1;
        };
        var table: attention.Table = try .init(capacity);
        errdefer table.deinit();
        const caches = try gpa.alloc(attention.Cache, columns.attentions);
        errdefer gpa.free(caches);
        // The head's cache is the MTP layer's shape, which the model stores on `attention` even when the trunk
        // layers loaded here are all DeltaNet. A trunk layer uses that shape too.
        const shape = if (columns.attentions > 0) m.attention.?.shape else undefined;
        for (caches, pools) |*cache, pool| cache.* = try .over(shape, pool, table.device.view, capacity);
        const head: ?attention.Cache = if (head_pool) |pool| try .over(headShape(m), pool.*, table.device.view, capacity) else null;
        return .{
            .recurrent = recurrent,
            .current = current,
            .held = held,
            .conv = conv,
            .ple = branches,
            .table = table,
            .caches = caches,
            .head = head,
            .gpa = gpa,
        };
    }

    /// The device bytes `open` asks, with nothing made: the stream's states, its table and its pooled keys,
    /// a layer's and with `head` the head's. The caches' rows are the pools' and are not here.
    pub fn bytesFor(m: model.Model, capacity: usize, head: bool) usize {
        const of = cuda.Buffer.bytesOf;
        const columns: Columns = .of(m);
        const dn = m.shape.delta_net;
        var bytes = columns.deltas * (2 * of(.f32, &.{ dn.value_heads, gdn.value_dim, gdn.key_dim }) + of(.bf16, &.{ 1, gdn.taps - 1, dn.conv() }) + gdn.Scratch.bytesFor(heldRows(), dn));
        for (m.layers) |layer| if (layer.ple) |branch| {
            bytes += ple.State.bytesFor(branch.weights.shape);
        };
        bytes += attention.Table.bytesFor(capacity);
        if (columns.attentions > 0) bytes += columns.attentions * attention.Cache.bytesOver(m.attention.?.shape, capacity);
        if (head) bytes += attention.Cache.bytesOver(headShape(m), capacity);
        return bytes;
    }

    pub fn deinit(self: *Slot) void {
        if (self.head) |*cache| cache.deinit();
        for (self.caches) |*cache| cache.deinit();
        self.gpa.free(self.caches);
        self.table.deinit();
        for (self.ple) |*branch| branch.deinit(self.gpa);
        self.gpa.free(self.ple);
        for (self.conv) |*window| window.deinit();
        self.gpa.free(self.conv);
        for (self.recurrent) |*pair| {
            pair[0].deinit();
            pair[1].deinit();
        }
        self.gpa.free(self.recurrent);
        self.gpa.free(self.current);
        for (self.held) |*scratch| scratch.deinit();
        self.gpa.free(self.held);
        self.* = undefined;
    }
};

/// The rows a slot's DeltaNet layers hold behind (`Slot.held`): a window of drafts, at most `gdn.wide_rows` - 1,
/// fits after none, and 24 MB a slot on Flash Next's 36 layers.
pub fn heldRows() usize {
    return gdn.wide_rows;
}

/// Layer `linear`'s DeltaNet state at the slot's `pos` into `dest`, as a launch (`forward.committed`'s).
pub fn committed(slot: Slot, linear: usize, dest: cuda.Buffer) cuda.Error!void {
    return forward.committedOf(slot.recurrent[linear], slot.current[linear], slot.held[linear], slot.behind, dest);
}

/// The slot's committed sides brought to `pos`, as `forward.land` a session's.
pub fn land(slot: *Slot) cuda.Error!void {
    try forward.landStates(slot.recurrent, slot.current, slot.held, slot.behind);
    slot.behind = 0;
}

/// `rounds_stage.zig`'s `stage`, the slots that cannot go on from the rows they are behind by landed first and
/// `Round.held` uploaded after.
pub fn stage(gpa: Allocator, m: model.Model, slots: []Slot, round: *Round, b: *forward.Buffers, segments: []const Segment) !void {
    return stageAhead(gpa, m, slots, round, b, segments, null);
}

/// `stage` with the PLE rows the head's passes gathered ahead (`rounds_stage.Ahead`, spec 144 WS-10).
pub fn stageAhead(gpa: Allocator, m: model.Model, slots: []Slot, round: *Round, b: *forward.Buffers, segments: []const Segment, ahead: ?*stage_file.Ahead) !void {
    // every slot that lands this round in one launch (spec 146 WS-07), as `land` each
    var batch: forward.Lands = .{};
    for (segments) |segment| {
        if (segment.slot >= slots.len) return error.InvalidRound;
        const slot = &slots[segment.slot];
        if (forward.landsAt(slot.behind, segment.tokens.len, heldRows())) {
            try batch.add(slot.recurrent, slot.current, slot.held, slot.behind);
            slot.behind = 0;
        }
    }
    try batch.flush();
    try stage_file.stageAhead(gpa, m, slots, round, b, segments, ahead);
    const width = 1 + 4 * round.columns.deltas;
    const held = try gpa.alloc(i64, slots.len * width);
    defer gpa.free(held);
    for (slots, 0..) |slot, s| {
        const row = held[s * width ..][0..width];
        row[0] = @intCast(slot.behind);
        for (slot.held, 0..) |scratch, linear| {
            const at = row[1 + 4 * linear ..][0..4];
            at.* = .{ address(scratch.k.view), address(scratch.v.view), address(scratch.g.view), address(scratch.beta.view) };
        }
    }
    try round.held.upload(std.mem.sliceAsBytes(held));
}

/// Whether a window of `rows` rows leaves its state to the slot's next chain (`forward.stores`).
fn defers(rows: usize) bool {
    return !forward.stores(rows);
}

pub fn address(view: cuda.View) i64 {
    return @intCast(@intFromPtr(view.ptr.?));
}

/// The head's cache's shape: beside a share the whole head's sizes on rank 0 (`mtp.over`), else the model's
/// attention's (a split head's half on a share), which a test may lend a head without a head of its own.
fn headShape(m: model.Model) attention.Shape {
    if (m.mtp) |head| if (head.whole) |whole| return whole.attention.shape;
    return m.attention.?.shape;
}

const viewOf = attention.viewOf;

extern fn zs_qwen4_gdn_chain_slots(seg: *const cuda.View, ptrs: *const cuda.View, stride: i64, col: i64, skip: i64, held: ?*const cuda.View, hcol: i64, wide: i64, p: *const cuda.View, cw: *const cuda.View, a_log: *const cuda.View, dt_bias: *const cuda.View, norm: *const cuda.View, eps: f64, out: *const cuda.View, xs: *const cuda.View, k: *const cuda.View, v: *const cuda.View, g: *const cuda.View, beta: *const cuda.View) c_int;
extern fn zs_qwen4_gdn_replay_slots(seg: *const cuda.View, ptrs: *const cuda.View, stride: i64, col: i64, skip: i64, wide: i64, k: *const cuda.View, v: *const cuda.View, g: *const cuda.View, beta: *const cuda.View) c_int;
extern fn zs_qwen4_shift_slots(seg: *const cuda.View, ptrs: *const cuda.View, stride: i64, col: i64, fresh: *const cuda.View, fresh_row: i64, taps: i64, channels: i64) c_int;
extern fn zs_qwen4_shift_slots_each(seg: *const cuda.View, ptrs: *const cuda.View, stride: i64, col0: i64, col_step: i64, freshes: [*]const cuda.View, count: i64, fresh_row: i64, taps: i64, channels: i64) c_int;

/// The device's part of the round `stage` uploaded. One chain a DeltaNet layer, and a replay where a segment
/// that wrote its state kept part of its rows, before the next layer reuses the scratch. The wide kernels take at
/// most one segment.
///
/// `bound` is the context bucket a captured graph was keyed by. Null is the direct path: the exact end, one
/// GEMM a stream that selects, and the wide kernels for one long segment. Set, the launches do not depend on
/// how the rows split, so a graph can hold them. `gem` is that path's per-slot regions when there is more than
/// one slot. `saves`, when set, is one scratch a DeltaNet layer, so a keep decided after the logits can be
/// replayed from it. A graph holds no replay: its rounds are staged with every row kept, and their windows of
/// several rows defer (spec 144 WS-09).
pub fn compute(m: model.Model, slots: []Slot, shared: *Shared, b: *forward.Buffers, round: *Round, segments: []const Segment, bound: ?usize, score: forward.Score, gem: ?*Gem, saves: ?[]gdn.Scratch) !void {
    return computeOver(m, slots, shared, b, round, segments, bound, score, gem, saves, null);
}

/// `compute`, and with `parts` the round of a rank of two (`rounds_ranks.zig`): `m` is its share, a block
/// and the experts write this rank's fp32 partial, and what the next write-back takes is the ranks' sum.
/// Null is one rank: no launch more, none less and none in another place.
pub fn computeOver(m: model.Model, slots: []Slot, shared: *Shared, b: *forward.Buffers, round: *Round, segments: []const Segment, bound: ?usize, score: forward.Score, gem: ?*Gem, saves: ?[]gdn.Scratch, parts: ?*const Parts) !void {
    const rows = round.used;
    std.debug.assert(rows >= 1 and slots.len == round.slots);
    // the replies' rows a serial step's form, a prompt chunk's after them its window's (`forms.formFrom`); a
    // captured round (no segments) is replies' alone
    const from = if (segments.len == 0) rows else forms.formFrom(Segment, segments, forward.bf16_rows);
    const was = forward.setForm(m, from);
    defer _ = forward.setForm(m, was);
    const dn = m.shape.delta_net;
    std.debug.assert(dn.value_heads == 48 or dn.value_heads == 24);
    std.debug.assert(dn.key_heads * 3 == dn.value_heads);
    const shape = m.shape;
    const h = b.h.rows(rows);
    try hc.embed(b.ids.rows(rows), m.embed.view, h, shape.streams);
    // a prompt chunk's placeholder rows take its image's features, as a session's window's (`forward.compute`);
    // never in a graph, whose rounds are replies'
    if (bound == null and b.image_rows > 0) try forward.embedImage(round.image.?, b.image_pairs.rows(b.image_rows), h, shape.streams);

    // The first segment of at least `wide_rows` rows, or -1. Any other long segment stays on the slot kernel,
    // which is the same bits. A captured round never takes the wide kernels: their span is the segment's.
    var skip: i64 = -1;
    if (bound == null) {
        for (segments) |segment| {
            if (segment.tokens.len >= gdn.wide_rows and skip < 0) skip = @intCast(segment.slot);
        }
    }

    // a replay launch a layer only where a segment that writes its state keeps part of its rows (spec 144 WS-09)
    const replays = bound == null and partlyKept(segments, skip);

    // a prompt's chunk on two ranks splits its glue by rows from the last PLE on (spec 149 WS-02)
    const own: ?Glued = if (parts) |share| share.glued(b.*, rows, bound != null) else null;
    const split_from = held_file.splitFrom(m);
    var pending: hc.Pending = .none;
    var linear: usize = 0;
    var attending: usize = 0;
    var branch_at: usize = 0;
    const stride: i64 = @intCast(round.columns.stride());
    for (m.layers, 0..) |layer, index| {
        const split = if (own != null and index >= split_from) own else null;
        if (layer.ple) |branch| {
            if (pending != .none) {
                try hc.writeback(h, b.pss.view, pending, shape.streams);
                pending = .none;
            }
            try pleRows(branch.weights, b.ple[branch_at], h, round.sid.rows(rows), round.seg.view, round.ptrs.view, round.columns.stride(), round.columns.ple(branch_at), rows);
            branch_at += 1;
        }
        if (split) |glued| {
            try parts.?.glueIn(glued, b.*, rows, layer.attn_hc, shape, pending);
        } else {
            // the write-back with the norm after it in one launch where the session's forward joins them (`hc.joins`)
            if (!hc.joins(pending, rows)) try hc.writeback(h, b.pss.view, pending, shape.streams);
            try forward.readoutAfter(layer.attn_hc, b.*, rows, shape, b.inject_attn, if (hc.joins(pending, rows)) pending else null);
        }
        const mixed = b.mixed.rows(rows);
        switch (layer.block) {
            .delta_net => |block| {
                // A prompt's window is wider than the rows a reply keeps. That projection is `projection_wide`;
                // the short one is `kept` rows, and writing the window there is past its end.
                const projected = if (skip >= 0) (b.projection_wide orelse b.projections[linear]) else b.projections[linear];
                // The short projection is the pool's to size (`hall_open.shortRows`): a round past it is
                // refused here, in a fast build too, before a launch writes past the buffer (spec 086).
                if (rows > @as(usize, @intCast(projected.view.size[0]))) return error.RoundPastProjection;
                const projection = projected.rows(rows);
                try block.projection.forward(mixed, projection);
                const col: i64 = @intCast(round.columns.delta(linear));
                const weights = block.kernelWeights();
                // `saves`, when the caller kept one scratch a layer, is where a keep decided after the logits
                // is replayed from. The graph path already passed it; a direct round does too when it stages
                // every row as kept and replays the prefix once the sampler has chosen.
                const saved: gdn.Scratch = if (saves) |all| all[linear] else shared.scratch;
                try cuda.launched(zs_qwen4_gdn_chain_slots(&round.seg.view, &round.ptrs.view, stride, col, skip, &round.held.view, @intCast(1 + 4 * linear), @intCast(gdn.wide_rows), &projection, &weights.conv, &weights.a_log, &weights.dt_bias, &weights.norm, shape.eps, &b.delta_out.rows(rows), &b.delta_sums.view, &saved.k.view, &saved.v.view, &saved.g.view, &saved.beta.view));
                if (skip >= 0) {
                    const wide: usize = @intCast(skip);
                    var at: usize = 0;
                    for (segments) |segment| {
                        if (segment.slot == wide) {
                            const n = segment.tokens.len;
                            var scratch = shared.scratch;
                            scratch.k.view = shared.scratch.k.span(at, n);
                            scratch.v.view = shared.scratch.v.span(at, n);
                            scratch.g.view = shared.scratch.g.span(at, n);
                            scratch.beta.view = shared.scratch.beta.span(at, n);
                            const side = slots[wide].current[linear];
                            const proj = projected.span(at, n);
                            try gdn.chain(weights, .{
                                .projection = proj,
                                .conv_state = convView(slots[wide].conv[linear], dn.conv()),
                                .state_in = slots[wide].recurrent[linear][side].view,
                                .state_out = slots[wide].recurrent[linear][1 - side].view,
                                .out = b.delta_out.span(at, n),
                                .sums = b.delta_sums.span(at, n),
                                .scratch = &scratch,
                                .work = b.delta_q.span(at, n),
                            }, n, shape.eps);
                            if (b.projection_wide) |all| {
                                // The next layer writes its projection over this one's, and `commit` slides
                                // this layer's conv windows after the last layer has run: over the rows kept
                                // here. The wide segment is kept whole and is the round's last.
                                const taps = gdn.taps - 1;
                                std.debug.assert(at + n == rows and segment.keep == n and n >= taps and at <= b.kept);
                                try round.tails.copyRows(linear * taps, all, at + n - taps, taps);
                                if (at > 0) try b.projections[linear].copyRows(0, all, 0, at);
                                round.tailed = true;
                            }
                            break;
                        }
                        at += segment.tokens.len;
                    }
                }
                if (replays) try cuda.launched(zs_qwen4_gdn_replay_slots(&round.seg.view, &round.ptrs.view, stride, col, skip, @intCast(gdn.wide_rows), &saved.k.view, &saved.v.view, &saved.g.view, &saved.beta.view));
                if (round.wanted.len > 0) {
                    std.debug.assert(bound == null);
                    const wide = skip >= 0 and @as(usize, @intCast(skip)) == round.wanted_slot;
                    try keepWithin(slots, round, segments, linear, if (wide) shared.scratch else saved, projected, dn);
                }
                if (skip >= 0) {
                    const wide: usize = @intCast(skip);
                    var at: usize = 0;
                    for (segments) |segment| {
                        if (segment.slot == wide and segment.keep < segment.tokens.len) {
                            var scratch = shared.scratch;
                            scratch.k.view = shared.scratch.k.span(at, segment.keep);
                            scratch.v.view = shared.scratch.v.span(at, segment.keep);
                            scratch.g.view = shared.scratch.g.span(at, segment.keep);
                            scratch.beta.view = shared.scratch.beta.span(at, segment.keep);
                            const side = slots[wide].current[linear];
                            try gdn.replayThrough(.wide, slots[wide].recurrent[linear][side].view, scratch, segment.keep, slots[wide].recurrent[linear][1 - side].view);
                            break;
                        }
                        at += segment.tokens.len;
                    }
                }
                if (parts) |share| {
                    try block.out.forward(b.delta_out.rows(rows), share.part(rows));
                    pending = if (split) |glued| try landedSplit(share, glued, b.*, rows, layer.mlp_hc, shape) else try share.landed(rows, b.inject_attn.view);
                } else {
                    try block.out.forward(b.delta_out.rows(rows), b.branch.rows(rows));
                    pending = .{ .branch = .{ .branch = b.branch.rows(rows), .inject = b.inject_attn.view } };
                }
                linear += 1;
            },
            .attention => |weights| {
                const shared_attn = m.attention.?;
                const scratch = &b.attention.?;
                try scratch.poison();
                if (scratch.selection) |selection| {
                    if (bound != null) try selection.sparse.zeroLaunched() else try selection.sparse.zero();
                }
                const pa = scratch.pa.rows(rows);
                try weights.proj.forward(mixed, pa);
                var context: usize = bound orelse 0;
                if (bound == null) {
                    for (segments) |segment| {
                        const end = slots[segment.slot].pos + segment.tokens.len;
                        if (end > context) context = end;
                    }
                }
                const pool = slots[0].caches[attending];
                const bases = round.bases.span(attending * slots.len, slots.len);
                try cuda.launched(cuda.kernels.zs_qwen4_attn_prep_rows(&round.sid.rows(rows), &round.tables.view, &bases, &pa, &round.posr.rows(rows), &weights.q_scale.view, &weights.k_scale.view, &weights.index_q_scale.view, &shared_attn.rotary.inv_freq.view, &scratch.q.rows(rows), &pool.k.view, &pool.v.view, &scratch.iq.rows(rows), &pool.index_keys.view, viewOf(&scratch.q_scale), viewOf(&pool.k_scale), viewOf(&pool.v_scale), shared_attn.shape.eps, @intCast(context), &round.ropes.view, shared_attn.shape.sections[1], shared_attn.shape.sections[2]));
                var blocks: usize = 1;
                if (bound != null) {
                    blocks = @max(1, rows / shared_attn.shape.ratio + 2);
                } else {
                    for (segments) |segment| blocks = @max(blocks, segment.tokens.len / shared_attn.shape.ratio + 2);
                }
                try cuda.launched(cuda.kernels.zs_qwen4_attn_pool_slots(&round.seg.view, &round.ptrs.view, stride, @intCast(round.columns.pooled(attending)), 0, &bases, &pool.index_keys.view, &weights.index_k_scale.view, &shared_attn.rotary.inv_freq.view, shared_attn.shape.eps, @intCast(shared_attn.shape.ratio), @intCast(blocks), &round.ropes.view, shared_attn.shape.sections[1], shared_attn.shape.sections[2]));
                const listed = if (bound) |fixed| try selectFixed(slots, round, scratch, shared_attn.shape, rows, fixed, .{ .layer = attending }, gem) else try selectSegments(slots, round, scratch, shared_attn.shape, segments, attending);
                const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(shared_attn.shape.head_dim)));
                const lists: ?attention.Selection = if (listed) scratch.selection else null;
                const ids = if (lists) |*selection| &selection.ids.view else null;
                const nk = if (lists) |*selection| &selection.nk.view else null;
                const sparse = if (lists) |*selection| &selection.sparse.view else null;
                // a prompt's round: a row a block over its chunks, as the pair's window (spec 149 WS-02), the same bits
                if (attention.window_rows != 0 and rows >= attention.window_rows) {
                    try cuda.launched(cuda.kernels.zs_attn_attend_window_rows(&round.sid.rows(rows), &round.tables.view, &bases, &scratch.q.rows(rows), &pool.k.view, &pool.v.view, &round.posr.rows(rows), &scratch.attn.rows(rows), ids, nk, sparse, viewOf(&scratch.q_scale), viewOf(&pool.k_scale), viewOf(&pool.v_scale), scale, @intCast(context)));
                } else {
                    try cuda.launched(cuda.kernels.zs_attn_attend_rows(&round.sid.rows(rows), &round.tables.view, &bases, &scratch.q.rows(rows), &pool.k.view, &pool.v.view, &round.posr.rows(rows), &scratch.partial_o.rows(rows), &scratch.partial_m.rows(rows), &scratch.partial_l.rows(rows), &scratch.attn.rows(rows), ids, nk, sparse, viewOf(&scratch.q_scale), viewOf(&pool.k_scale), viewOf(&pool.v_scale), scale, @intCast(context)));
                }
                try attention.gate(scratch.attn.rows(rows), pa, scratch.gated.rows(rows));
                if (parts) |share| {
                    try weights.o.forward(scratch.gated.rows(rows), share.part(rows));
                    pending = if (split) |glued| try landedSplit(share, glued, b.*, rows, layer.mlp_hc, shape) else try share.landed(rows, b.inject_attn.view);
                } else {
                    try weights.o.forward(scratch.gated.rows(rows), scratch.branch.rows(rows));
                    pending = .{ .branch = .{ .branch = scratch.branch.rows(rows), .inject = b.inject_attn.view } };
                }
                attending += 1;
            },
        }
        if (split == null) {
            if (!hc.joins(pending, rows)) try hc.writeback(h, b.pss.view, pending, shape.streams);
            try forward.readoutAfter(layer.mlp_hc, b.*, rows, shape, b.inject_mlp, if (hc.joins(pending, rows)) pending else null);
        }
        try moe.forward(layer.moe, b.moe, b.mixed.rows(rows), rows, m.scratch.quantize_from);
        pending = if (split) |glued| try parts.?.foldedRows(glued, b.moe, rows, shape.top + 1) else if (parts) |share| try share.folded(b.moe, rows, shape.top + 1, b.inject_mlp.view) else moe.pending(b.moe, b.inject_mlp.view);
        // a DFlash2 tap layer's streams, every slot's rows, as a session's `compute` writes them (spec 126 WS-05):
        // the write-back the next layer would launch, so no bit of h moves; the two-rank round's sum is gathered
        try forward.capture(b, index, rows, &pending, shape.streams);
    }
    if (own) |glued| try parts.?.glueEnd(glued, b.*, rows, m.mixer, shape, pending) else try forward.finish(m, m.mixer, m.layers.len, b, rows, pending, null);
    switch (score) {
        .all => {
            std.debug.assert(rows <= b.scoredRows());
            try m.head.forward(b.mixed.rows(rows), b.logits.rows(rows));
        },
        .last => try m.head.forward(b.mixed.span(rows - 1, 1), b.logits.rows(1)),
        .none => {},
    }
}

/// A split layer's landing (`Parts.landedRows`): the MLP's read-out is already run on this rank's rows, so
/// nothing is left pending for the write-back after the block.
fn landedSplit(share: *const Parts, glued: Glued, b: forward.Buffers, rows: usize, connection: model.HyperConnection, shape: model.Shape) !hc.Pending {
    try share.landedRows(glued, b, rows, connection, shape);
    return .none;
}

/// Layer `linear`'s part of the states `round.wanted`: the chain that just ran replayed from the state the
/// segment started at to each one's row, lowest first and each from the one below it, and the conv inputs
/// slid to that row (`forward.layerForward` does the same for a session's wide window). `scratch` is what
/// that chain wrote, by the round's rows, and `projected` the projection it read. The kernels are the ones a
/// session's `Snapshot.within` runs, and the slot's state is only read. A window that deferred its state saved
/// its rows after the slot's `behind` in `Slot.held`: the replays start from the committed side over those.
fn keepWithin(slots: []const Slot, round: *const Round, segments: []const Segment, linear: usize, scratch: gdn.Scratch, projected: cuda.Buffer, dn: gdn.Shape) !void {
    var at: usize = 0;
    const segment = for (segments) |segment| {
        if (segment.slot == round.wanted_slot) break segment;
        at += segment.tokens.len;
    } else unreachable;
    const n = segment.tokens.len;
    const slot = slots[segment.slot];
    var own = scratch;
    var base: usize = 0;
    if (defers(n)) {
        own = slot.held[linear];
        base = slot.behind;
    } else {
        std.debug.assert(slot.behind == 0);
        own.k.view = scratch.k.span(at, n);
        own.v.view = scratch.v.span(at, n);
        own.g.view = scratch.g.span(at, n);
        own.beta.view = scratch.beta.span(at, n);
    }
    const fresh: state_windows.Fresh = .{ .view = projected.span(at, n), .layer = 0, .row = dn.projection(), .rows = n };
    var below: usize = 0;
    var from = slot.recurrent[linear][slot.current[linear]];
    for (round.wanted) |wanted| {
        const keep = wanted.pos - slot.pos;
        std.debug.assert(wanted.pos > slot.pos and keep < n and base + keep >= below and segment.keep == n);
        if (base + keep == below) {
            try wanted.recurrent[linear].copyLaunched(from);
        } else {
            try gdn.replayFrom(from.view, own, below, base + keep - below, wanted.recurrent[linear].view);
        }
        below = base + keep;
        from = wanted.recurrent[linear];
        try wanted.conv[linear].copyLaunched(slot.conv[linear]);
        try state_windows.shiftWindows(wanted.conv[linear].view, fresh, keep);
    }
}

pub fn keepHistory(history: []u32, tokens: []const u32, keep: usize) void {
    if (keep >= history.len) {
        @memcpy(history, tokens[keep - history.len .. keep]);
    } else {
        std.mem.copyForwards(u32, history[0 .. history.len - keep], history[keep..]);
        @memcpy(history[history.len - keep ..], tokens[0..keep]);
    }
}

/// Keeps each segment's prefix: the replay already ran in `compute`, so this flips the committed side of a
/// segment that wrote its state, or counts the rows a deferred one kept behind it (`Slot.behind`, as
/// `forward.commit`), slides the conv windows and the PLE tails, and advances `pos`.
pub fn commit(m: model.Model, slots: []Slot, b: *forward.Buffers, round: *Round, segments: []const Segment) !void {
    const stride: i64 = @intCast(round.columns.stride());
    var branch: usize = 0;
    for (m.layers) |layer| if (layer.ple) |held| {
        for (segments) |segment| {
            const tokens: []const u32 = @ptrCast(segment.tokens);
            keepHistory(slots[segment.slot].ple[branch].history, tokens, segment.keep);
        }
        const tail = held.weights.shape.tailRows();
        if (tail > 0) {
            const wide = held.weights.shape.wide();
            try cuda.launched(zs_qwen4_shift_slots(&round.seg.view, &round.ptrs.view, stride, @intCast(round.columns.ple(branch)), &b.ple[branch].nrow.view, @intCast(wide), @intCast(tail), @intCast(wide)));
        }
        branch += 1;
    };
    const dn = m.shape.delta_net;
    for (segments) |segment| {
        const slot = &slots[segment.slot];
        if (defers(segment.tokens.len)) {
            slot.behind += segment.keep;
            std.debug.assert(slot.behind <= heldRows());
        } else {
            for (slot.current) |*side| side.* = 1 - side.*;
            slot.behind = 0;
        }
    }
    // every layer's windows in one launch where each layer's rows are in its own projection (spec 144 WS-09)
    const deltas = round.columns.deltas;
    if (!round.tailed and deltas > 0 and deltas <= 64) {
        var freshes: [64]cuda.View = undefined;
        for (freshes[0..deltas], b.projections[0..deltas]) |*fresh, projection| fresh.* = projection.view;
        try cuda.launched(zs_qwen4_shift_slots_each(&round.seg.view, &round.ptrs.view, stride, @intCast(round.columns.delta(0) + 2), 3, &freshes, @intCast(deltas), @intCast(dn.projection()), @intCast(gdn.taps - 1), @intCast(dn.conv())));
    } else try shifts(m, b, round, segments, stride, dn);
    for (segments) |segment| {
        const slot = &slots[segment.slot];
        slot.pos += segment.keep;
        if (slot.image) |image| {
            if (slot.pos > image.rows[image.rows.len - 1]) slot.image = null;
        }
    }
}

/// `commit`'s conv windows a layer at a time: a round with a wide segment wrote every layer's projection into
/// `projection_wide`, and each layer's rows are put back there before its shift.
fn shifts(m: model.Model, b: *forward.Buffers, round: *Round, segments: []const Segment, stride: i64, dn: gdn.Shape) !void {
    // The chain wrote the wide segment into `projection_wide` when the round had one. The shift reads that
    // buffer, the same rows, or the short projection when every segment stayed under `wide_rows`.
    var wide_slot: i64 = -1;
    var before: usize = 0;
    var wide_end: usize = 0;
    for (segments) |segment| {
        if (segment.tokens.len >= gdn.wide_rows and wide_slot < 0) {
            wide_slot = @intCast(segment.slot);
            wide_end = before + segment.tokens.len;
        }
        if (wide_slot < 0) before += segment.tokens.len;
    }
    var linear: usize = 0;
    for (m.layers) |layer| if (layer.block == .delta_net) {
        const projected = if (wide_slot >= 0) (b.projection_wide orelse b.projections[linear]) else b.projections[linear];
        if (round.tailed) {
            // `projection_wide` holds the last layer's rows: this layer's are put back where the shift reads
            // them, the rows before the wide segment and that segment's last.
            const taps = gdn.taps - 1;
            if (before > 0) try projected.copyRows(0, b.projections[linear], 0, before);
            try projected.copyRows(wide_end - taps, round.tails, linear * taps, taps);
        }
        try cuda.launched(zs_qwen4_shift_slots(&round.seg.view, &round.ptrs.view, stride, @intCast(round.columns.delta(linear) + 2), &projected.view, @intCast(dn.projection()), @intCast(gdn.taps - 1), @intCast(dn.conv())));
        linear += 1;
    };
}

/// `stage`, `compute` and `commit` of one round.
pub fn run(gpa: Allocator, m: model.Model, slots: []Slot, shared: *Shared, round: *Round, b: *forward.Buffers, segments: []const Segment) !void {
    try stage(gpa, m, slots, round, b, segments);
    try compute(m, slots, shared, b, round, segments, null, .all, null, null);
    try commit(m, slots, b, round, segments);
}

/// Extents for every chain's positions below `uptos[i]`, or `error.Short` with nothing reserved and nothing
/// appended. A round calls this before `stage`: a window is not written, and so not committed, unless every
/// stream's rows fit. Evicting an idle chain, and which request is taken off, stay the scheduler's.
pub fn reserve(pool: *extents.Pool, chains: []const *extents.Chain, uptos: []const usize) extents.Error!void {
    std.debug.assert(chains.len == uptos.len);
    var need: usize = 0;
    for (chains, uptos) |chain, upto| {
        if (upto > pool.capacity) return error.PastCapacity;
        const want = extents.extentsFor(upto);
        if (want > chain.table.items.len) need += want - chain.table.items.len;
    }
    if (need > pool.free_count) return error.Short;
    for (chains, uptos) |chain, upto| try pool.grow(chain, upto);
}

/// Whether a segment other than slot `skip`'s writes its state and keeps part of its rows: one a replay serves.
fn partlyKept(segments: []const Segment, skip: i64) bool {
    for (segments) |segment| {
        if (@as(i64, @intCast(segment.slot)) != skip and !defers(segment.tokens.len) and segment.keep < segment.tokens.len) return true;
    }
    return false;
}

/// The device `seg` of `segments`, whose `keep` is how many rows each stream kept. `stage` may have uploaded
/// a longer keep; the replay and the commit read this one.
fn uploadKeeps(round: *Round, slots: []const Slot, segments: []const Segment) cuda.Error!void {
    var host: [64 * 4]i32 = undefined;
    @memset(&host, 0);
    std.debug.assert(slots.len <= 64);
    var row: usize = 0;
    for (segments) |segment| {
        const base = segment.slot * 4;
        host[base] = @intCast(row);
        host[base + 1] = @intCast(segment.tokens.len);
        host[base + 2] = @intCast(segment.keep);
        host[base + 3] = @intCast(slots[segment.slot].pos);
        row += segment.tokens.len;
    }
    try round.seg.upload(std.mem.sliceAsBytes(host[0 .. slots.len * 4]));
}

/// After a round staged with every row kept, replay each stream that kept fewer from `saves` (one scratch a
/// DeltaNet layer, what that layer's chain wrote). A stream that kept every row is left as the chain wrote it,
/// and so is one whose window deferred its state (spec 144 WS-09): `commit` counts its keep. Call it before
/// `commit`, which then slides the windows by `segments`' keeps.
pub fn replayKept(m: model.Model, slots: []const Slot, round: *Round, segments: []const Segment, saves: []const gdn.Scratch) !void {
    try uploadKeeps(round, slots, segments);
    if (!partlyKept(segments, -1)) return;
    const stride: i64 = @intCast(round.columns.stride());
    var linear: usize = 0;
    for (m.layers) |layer| if (layer.block == .delta_net) {
        const scratch = saves[linear];
        try cuda.launched(zs_qwen4_gdn_replay_slots(&round.seg.view, &round.ptrs.view, stride, @intCast(round.columns.delta(linear)), -1, @intCast(gdn.wide_rows), &scratch.k.view, &scratch.v.view, &scratch.g.view, &scratch.beta.view));
        linear += 1;
    };
}

const stage_file = @import("rounds_stage.zig");
const pleRows = stage_file.pleRows;
const convView = stage_file.convView;

const select_file = @import("rounds_select.zig");

pub const Gem = select_file.Gem;
const selectSegments = select_file.selectSegments;
pub const selectFixed = select_file.selectFixed;
pub const Keys = select_file.Keys;

pub const Graphs = @import("rounds_graphs.zig").Graphs;
pub const Parts = @import("rounds_parts.zig").Parts;
pub const askNext = stage_file.askNext;
const Glued = @import("rounds_parts.zig").Glued;
const held_file = @import("../ranks_forward.zig");
