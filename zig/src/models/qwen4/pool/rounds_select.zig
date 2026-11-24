//! The round's selection of the keys each row reads: each stream's own list on the direct path, the slots'
//! GEMMs and lists a captured round runs, and the regions they run from (`Gem`). `rounds.zig`'s `compute`
//! calls the two selects and names `Gem` as its own; the types it passes them are that file's.

const cuda = @import("../../../cuda/cuda.zig");
const attention = @import("../blocks/attention.zig");
const rounds = @import("rounds.zig");
const Round = rounds.Round;
const Segment = rounds.Segment;
const Slot = rounds.Slot;

extern fn zs_qwen4_attn_pack_queries(seg: *const cuda.View, iq: *const cuda.View, queries: *const cuda.View, heads: i64, dim: i64, gem_rows: i64) c_int;
extern fn zs_qwen4_attn_select_slots(dots: *const cuda.View, slot_stride: i64, seg: *const cuda.View, sid: *const cuda.View, posr: *const cuda.View, tables: *const cuda.View, scores: *const cuda.View, ids: *const cuda.View, nk: *const cuda.View, sparse: *const cuda.View, rows: i64, heads: i64, dim: i64, nbl: i64, ratio: i64, top: i64) c_int;

/// Each stream that selects, on the direct path. A segment wider than a round is tiles of `dots`; the list is that stream's own rows.
pub fn selectSegments(slots: []Slot, round: *Round, scratch: *attention.Scratch, shape: attention.Shape, segments: []const Segment, attending: usize) !bool {
    const selection = if (scratch.selection) |*held| held else return false;
    var listed = false;
    var at: usize = 0;
    for (segments) |segment| {
        const end = slots[segment.slot].pos + segment.tokens.len;
        if (shape.selects(end)) {
            listed = true;
            const heads = shape.index_heads;
            const queries = cuda.View.init(scratch.iq.span(at, segment.tokens.len).ptr.?, .bf16, &.{ segment.tokens.len * heads, shape.index_dim });
            const pooled = slots[segment.slot].caches[attending].pooled;
            const keys = cuda.View.init(pooled.view.ptr.?, .bf16, &.{ end / shape.ratio, shape.index_dim });
            const pos = round.posr.span(at, 1);
            try attention.select(shape, selection.*, queries, keys, pos, at, segment.tokens.len, end, slots[segment.slot].table.device.view);
        }
        at += segment.tokens.len;
    }
    return listed;
}

/// Per-slot query and product regions for a graph's GEMMs. A lone slot GEMMs the round's queries in place and
/// does not use one.
pub const Gem = struct {
    queries: cuda.Buffer,
    dots: cuda.Buffer,
    slots: usize,
    rows: usize,
    heads: usize,
    dim: usize,
    blocks: usize,

    pub fn init(slots: usize, rows: usize, shape: attention.Shape, capacity: usize) !Gem {
        const blocks = @max(capacity / shape.ratio, 1);
        var queries: cuda.Buffer = try .init(.bf16, &.{ slots * rows * shape.index_heads, shape.index_dim });
        errdefer queries.deinit();
        try queries.zero();
        const dots: cuda.Buffer = try .init(.f32, &.{slots * rows * shape.index_heads * blocks});
        return .{
            .queries = queries,
            .dots = dots,
            .slots = slots,
            .rows = rows,
            .heads = shape.index_heads,
            .dim = shape.index_dim,
            .blocks = blocks,
        };
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(slots: usize, rows: usize, shape: attention.Shape, capacity: usize) usize {
        const blocks = @max(capacity / shape.ratio, 1);
        return cuda.Buffer.bytesOf(.bf16, &.{ slots * rows * shape.index_heads, shape.index_dim }) + cuda.Buffer.bytesOf(.f32, &.{slots * rows * shape.index_heads * blocks});
    }

    pub fn deinit(self: *Gem) void {
        self.queries.deinit();
        self.dots.deinit();
        self.* = undefined;
    }

    fn stride(self: Gem) usize {
        return self.rows * self.heads * self.blocks;
    }
};

/// Whose pooled keys a selection reads: a trunk layer's (its index among the attention layers), or the MTP
/// head's own (`hall_head.zig`'s pass).
pub const Keys = union(enum) {
    layer: usize,
    head,

    fn of(self: Keys, slot: Slot) cuda.Buffer {
        return switch (self) {
            .layer => |attending| slot.caches[attending].pooled,
            .head => slot.head.?.pooled,
        };
    }
};

/// `slots` GEMMs of the round's row count, over the bucket's blocks, in one launch (`zs_b16_gemm_entries`, spec
/// 144 port 5). A slot with no rows in the round (`seg`'s count, read on the device) returns from the pack and from
/// its GEMM at once: a graph keyed {rows, context} keeps its one launch, and an empty slot reads no keys. A slot
/// that runs gets `zs_b16_gemm`'s bits for its region. `round` is the trunk's `Round` or the head's `HeadRound`:
/// the `seg`, `sid`, `posr` and `tables` its rows were staged into.
pub fn selectFixed(slots: []Slot, round: anytype, scratch: *attention.Scratch, shape: attention.Shape, rows: usize, context: usize, keys: Keys, gem: ?*Gem) !bool {
    const selection = if (scratch.selection) |*held| held else return false;
    if (!shape.selects(context)) return false;
    const nbl = context / shape.ratio;
    const heads = shape.index_heads;
    const dim = shape.index_dim;
    if (slots.len == 1) {
        const queries = cuda.View.init(scratch.iq.span(0, rows).ptr.?, .bf16, &.{ rows * heads, dim });
        const pooled = keys.of(slots[0]);
        const view = cuda.View.init(pooled.view.ptr.?, .bf16, &.{ nbl, dim });
        const pos = round.posr.span(0, 1);
        try attention.select(shape, selection.*, queries, view, pos, 0, rows, context, slots[0].table.device.view);
        return true;
    }
    const regions = gem.?;
    try cuda.launched(zs_qwen4_attn_pack_queries(&round.seg.view, &scratch.iq.view, &regions.queries.view, @intCast(heads), @intCast(dim), @intCast(regions.rows)));
    // 64: `rounds.zig`'s bound on a pool's slots and `zs_b16_gemm_entries`' on its entries
    var pooled: [64]cuda.View = undefined;
    for (slots, pooled[0..slots.len]) |slot, *view| view.* = .init(keys.of(slot).view.ptr.?, .bf16, &.{ nbl, dim });
    const x_stride = regions.rows * heads * dim;
    const queries = cuda.View.init(regions.queries.view.ptr.?, .bf16, &.{ slots.len * regions.rows * heads, dim });
    try cuda.launched(cuda.kernels.zs_b16_gemm_entries(&queries, @intCast(rows * heads), @intCast(x_stride), &pooled, @intCast(slots.len), &regions.dots.view, @intCast(regions.stride()), &round.seg.view, 1));
    var scores = selection.scores.rows(rows);
    var ids = selection.ids.rows(rows);
    var nk = selection.nk.rows(rows);
    var sparse = selection.sparse.rows(rows);
    try cuda.launched(zs_qwen4_attn_select_slots(&regions.dots.view, @intCast(regions.stride()), &round.seg.view, &round.sid.rows(rows), &round.posr.rows(rows), &round.tables.view, &scores, &ids, &nk, &sparse, @intCast(rows), @intCast(heads), @intCast(dim), @intCast(nbl), @intCast(shape.ratio), @intCast(shape.top())));
    return true;
}
