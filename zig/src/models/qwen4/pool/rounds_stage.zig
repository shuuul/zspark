//! `rounds.zig`'s round on the host, and the views its device part reads: `stage` gathers each segment's PLE
//! table rows and uploads the round's tables once; `pleRows` launches a PLE branch over the round's rows,
//! `convView` is a DeltaNet layer's convolution window. `rounds.zig` names the three as its own; the types
//! they take are that file's.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const forward = @import("../forward.zig");
const gdn = @import("../blocks/gdn.zig");
const model = @import("../model.zig");
const ple = @import("../blocks/ple.zig");
const masks_file = @import("../../../decode/masks.zig");
const rounds = @import("rounds.zig");
const Round = rounds.Round;
const Segment = rounds.Segment;
const Slot = rounds.Slot;
const address = rounds.address;

/// Gathers each segment's table rows and uploads `sid`, `posr`, `seg`, `ptrs`, `scored`, the slots' ropes and
/// an image prompt's pairs once.
pub fn stage(gpa: Allocator, m: model.Model, slots: []Slot, round: *Round, b: *forward.Buffers, segments: []const Segment) !void {
    return stageAhead(gpa, m, slots, round, b, segments, null);
}

/// `stage`, a segment's first rows taken from what `ahead` gathered for its slot where they are its own.
pub fn stageAhead(gpa: Allocator, m: model.Model, slots: []Slot, round: *Round, b: *forward.Buffers, segments: []const Segment, ahead: ?*Ahead) !void {
    // checked in every build (spec 103 WS-01): `seen`, `sid` and `seg` below are indexed by these, unchecked
    // in a fast build
    if (slots.len != round.slots or segments.len < 1) return error.InvalidRound;
    var total: usize = 0;
    for (segments) |segment| {
        if (segment.slot >= slots.len or segment.keep < 1 or segment.keep > segment.tokens.len) return error.InvalidRound;
        total += segment.tokens.len;
    }
    if (total < 1 or total > round.rows or total > b.rows or total > m.max_rows) return error.InvalidRound;
    round.used = total;
    round.tailed = false;

    const sid = try gpa.alloc(i32, total);
    defer gpa.free(sid);
    const posr = try gpa.alloc(i32, total);
    defer gpa.free(posr);
    const scored = try gpa.alloc(i32, total);
    defer gpa.free(scored);
    const ids = try gpa.alloc(i32, total);
    defer gpa.free(ids);
    const seg = try gpa.alloc(i32, slots.len * 4);
    defer gpa.free(seg);
    @memset(seg, 0);
    const ptrs = try gpa.alloc(i64, slots.len * round.columns.stride());
    defer gpa.free(ptrs);
    const tables = try gpa.alloc(i64, slots.len);
    defer gpa.free(tables);
    const seen = try gpa.alloc(bool, slots.len);
    defer gpa.free(seen);
    @memset(seen, false);

    var row: usize = 0;
    for (segments) |segment| {
        // a slot twice would write its `seg` row over its first segment's
        if (seen[segment.slot]) return error.InvalidRound;
        seen[segment.slot] = true;
        const n = segment.tokens.len;
        const base = segment.slot * 4;
        seg[base] = @intCast(row);
        seg[base + 1] = @intCast(n);
        seg[base + 2] = @intCast(segment.keep);
        seg[base + 3] = @intCast(slots[segment.slot].pos);
        for (segment.tokens, 0..) |token, i| {
            sid[row + i] = @intCast(segment.slot);
            posr[row + i] = @intCast(slots[segment.slot].pos + i);
            ids[row + i] = token;
        }
        row += n;
    }
    for (scored, 0..) |*entry, i| entry.* = @intCast(i);

    var branch: usize = 0;
    for (m.layers) |layer| if (layer.ple) |held| {
        const buffers = &b.ple[branch];
        const stride = held.table.stride();
        const heads = buffers.shape.heads;
        row = 0;
        for (segments) |segment| {
            const n = segment.tokens.len;
            const gathered = buffers.ids[row * heads ..][0 .. n * heads];
            const tokens: []const u32 = @ptrCast(segment.tokens);
            ple.ngramIds(held.ngram, slots[segment.slot].ple[branch].eos, slots[segment.slot].ple[branch].history, tokens, gathered);
            const host = buffers.host[row * heads * stride ..][0 .. gathered.len * stride];
            const asked = if (ahead) |one| one.take(segment.slot, branch, gathered) else &.{};
            // or a prompt's chunk asked of `b.ahead` as the round before was staged (`askNext`), where it holds
            // more: the head's pass over the chunk before asked the slot's `ahead` for its first token alone
            const next = b.ahead[branch].ready(gathered);
            const taken = if (next.len > asked.len) next else asked;
            @memcpy(host[0..taken.len], taken);
            try held.table.gather(gathered[taken.len / stride ..], host[taken.len..]);
            row += n;
        }
        branch += 1;
    };

    for (slots, 0..) |slot, s| {
        tables[s] = address(slot.table.device.view);
        const base = s * round.columns.stride();
        ptrs[base] = tables[s];
        for (slot.recurrent, slot.current, slot.conv, 0..) |pair, side, window, linear| {
            const col = round.columns.delta(linear);
            ptrs[base + col] = address(pair[side].view);
            ptrs[base + col + 1] = address(pair[1 - side].view);
            ptrs[base + col + 2] = address(window.view);
        }
        for (slot.ple, 0..) |state, p| ptrs[base + round.columns.ple(p)] = address(state.tail.view);
        for (slot.caches, 0..) |cache, a| ptrs[base + round.columns.pooled(a)] = address(cache.pooled.view);
    }

    try b.ids.uploadAt(0, std.mem.sliceAsBytes(ids));
    try round.sid.uploadAt(0, std.mem.sliceAsBytes(sid));
    try round.posr.uploadAt(0, std.mem.sliceAsBytes(posr));
    try round.seg.upload(std.mem.sliceAsBytes(seg));
    try round.ptrs.upload(std.mem.sliceAsBytes(ptrs));
    try round.scored.uploadAt(0, std.mem.sliceAsBytes(scored));
    try round.tables.upload(std.mem.sliceAsBytes(tables));
    const ropes = try gpa.alloc(i64, slots.len);
    defer gpa.free(ropes);
    for (slots, ropes) |slot, *rope| rope.* = slot.ropeAddress();
    try round.ropes.upload(std.mem.sliceAsBytes(ropes));
    // an image prompt's placeholder rows among the round's, a pair (round row, feature row) each
    b.image_rows = 0;
    round.image = null;
    row = 0;
    for (segments) |segment| {
        defer row += segment.tokens.len;
        const image = slots[segment.slot].image orelse continue;
        if (!image.touches(slots[segment.slot].pos, segment.tokens.len)) continue;
        if (round.image != null) return error.InvalidRound;
        round.image = image.features;
        var pairs: [256][2]i32 = undefined;
        var start: usize = 0;
        while (start < segment.tokens.len) : (start += pairs.len) {
            const at = image.within(slots[segment.slot].pos + start, @min(pairs.len, segment.tokens.len - start), &pairs);
            for (pairs[0..at]) |*pair| pair[0] += @intCast(row + start);
            if (at > 0) try b.image_pairs.uploadAt(b.image_rows * @sizeOf([2]i32), std.mem.sliceAsBytes(pairs[0..at]));
            b.image_rows += at;
        }
    }
    if (round.columns.attentions > 0) {
        const host_bases = try gpa.alloc(i64, round.columns.attentions * slots.len * 3);
        defer gpa.free(host_bases);
        for (slots, 0..) |slot, s| {
            for (slot.caches, 0..) |cache, a| {
                const at = (a * slots.len + s) * 3;
                host_bases[at] = address(cache.k.view);
                host_bases[at + 1] = address(cache.v.view);
                host_bases[at + 2] = address(cache.index_keys.view);
            }
        }
        try round.bases.upload(std.mem.sliceAsBytes(host_bases));
    }
    branch = 0;
    for (m.layers) |layer| if (layer.ple) |held| {
        try b.ple[branch].upload(total, held.table.stride());
        branch += 1;
    };
}

/// Once a round is staged: `next`, the prompt's tokens after its chunk, the round's last segment (`promptBeside`),
/// asked of `b.ahead` (`ple.Ahead`), whose thread gathers their table rows while the device runs this round, as
/// a lone stream's `forward.expect` does for its next window (spec 087, the pair's 111 WS-02; spec 149 WS-02).
/// The next `stageAhead` takes the rows of its tokens that were asked for and gathered; a chunk otherwise cut,
/// or whose rows are not gathered by then, is gathered there. A row's ids are its own token's and those before
/// it, so a row taken is the bytes a gather would read for it: the bits stay. Both ranks ask alike.
pub fn askNext(m: model.Model, slots: []const Slot, b: *forward.Buffers, segments: []const Segment, next: []const i32) void {
    if (!forward.gather_ahead or next.len == 0 or segments.len == 0) return;
    const last = segments[segments.len - 1];
    const n = @min(next.len, b.rows);
    var branch: usize = 0;
    for (m.layers) |layer| if (layer.ple) |held| {
        b.ahead[branch].start(held.ngram, held.table, slots[last.slot].ple[branch], @ptrCast(last.tokens), @ptrCast(next[0..n]));
        branch += 1;
    };
}

/// Each slot's PLE table rows for its next round, gathered on a thread of the slot's own while the head's passes
/// run (spec 144 WS-10, as a lone stream's `forward.ask`): the window known so far, the kept token and then each
/// draft a pass goes on from (`hall_head.headPass`, on both ranks); on rank 0, once the chase is over, the window
/// the round will run (`hall_chase.absorbAndChase`). `stageAhead` takes the rows of a segment's first tokens that
/// were asked for and the thread has gathered by then, never waiting on it, and gathers the rest. A row's ids are
/// its own token's and those before it, so rows taken are the bytes a gather would read for them: the bits stay.
/// A slot's thread starts on its first asking.
pub const Ahead = struct {
    /// [slots * branches], a slot's branches side by side.
    gathers: []ple.Ahead,
    /// [slots * rows]: each slot's window as asked.
    windows: []i32,
    asked: []usize,
    branches: usize,

    pub const rows = masks_file.max_rows;

    pub fn init(gpa: Allocator, m: model.Model, slots: usize) Allocator.Error!Ahead {
        var branches: usize = 0;
        for (m.layers) |layer| branches += @intFromBool(layer.ple != null);
        const gathers = try gpa.alloc(ple.Ahead, slots * branches);
        var made: usize = 0;
        errdefer {
            for (gathers[0..made]) |*one| one.deinit(gpa);
            gpa.free(gathers);
        }
        for (0..slots) |_| for (m.layers) |layer| if (layer.ple) |held| {
            gathers[made] = try .init(gpa, held.weights.shape, rows);
            made += 1;
        };
        const windows = try gpa.alloc(i32, slots * rows);
        errdefer gpa.free(windows);
        const asked = try gpa.alloc(usize, slots);
        @memset(asked, 0);
        return .{ .gathers = gathers, .windows = windows, .asked = asked, .branches = branches };
    }

    /// Once no thread is asked any more: each joins its own.
    pub fn deinit(self: *Ahead, gpa: Allocator) void {
        for (self.gathers) |*one| one.deinit(gpa);
        gpa.free(self.gathers);
        gpa.free(self.windows);
        gpa.free(self.asked);
        self.* = undefined;
    }

    /// Slot `slot`'s next window begins with `token`, the token after its last kept row.
    pub fn begin(self: *Ahead, m: model.Model, slots: []const Slot, slot: usize, token: i32) void {
        // asked here, not through `extend`, which goes on only from a window begun (spec 146 WS-06: a `begin`
        // that set the count to 0 and called `extend` asked nothing, so no pass gathered a row)
        self.windows[slot * rows] = token;
        self.asked[slot] = 1;
        self.start(m, slots, slot);
    }

    /// The draft `token` follows slot `slot`'s window as asked so far.
    pub fn extend(self: *Ahead, m: model.Model, slots: []const Slot, slot: usize, token: i32) void {
        const n = self.asked[slot];
        if (n == 0 or n >= rows) return;
        self.windows[slot * rows + n] = token;
        self.asked[slot] = n + 1;
        self.start(m, slots, slot);
    }

    /// Slot `slot`'s next window is `window`, as far as it is known.
    pub fn ask(self: *Ahead, m: model.Model, slots: []const Slot, slot: usize, window: []const i32) void {
        const n = @min(window.len, rows);
        if (n == 0) return;
        @memcpy(self.windows[slot * rows ..][0..n], window[0..n]);
        self.asked[slot] = n;
        self.start(m, slots, slot);
    }

    fn start(self: *Ahead, m: model.Model, slots: []const Slot, slot: usize) void {
        if (!forward.gather_ahead) return;
        const window: []const u32 = @ptrCast(self.windows[slot * rows ..][0..self.asked[slot]]);
        var branch: usize = 0;
        for (m.layers) |layer| if (layer.ple) |held| {
            self.gathers[slot * self.branches + branch].start(held.ngram, held.table, slots[slot].ple[branch], &.{}, window);
            branch += 1;
        };
    }

    /// The rows already gathered for the first of `ids`, slot `slot`'s branch `branch`'s (`ple.Ahead.ready`):
    /// the stage never sleeps on a thread, whose wake from the cores' deep idle states and back (cpuidle's LPI-3
    /// exits in 433 us on the GB10; `ranks/channel.zig` spins against the same) costs more than the few rows
    /// left; those are gathered inline.
    fn take(self: *Ahead, slot: usize, branch: usize, ids: []const u64) []const u16 {
        return self.gathers[slot * self.branches + branch].ready(ids);
    }
};

extern fn zs_qwen4_ple_conv_slots(gated: *const cuda.View, pss: *const cuda.View, norm_conv: *const cuda.View, conv: *const cuda.View, h: *const cuda.View, nrow: *const cuda.View, sid: *const cuda.View, seg: *const cuda.View, ptrs: *const cuda.View, stride: i64, col: i64, eps: f64, streams: i64, dilation: i64) c_int;

pub fn pleRows(weights: ple.Weights, buffers: ple.Buffers, h: cuda.View, sid: cuda.View, seg: cuda.View, ptrs: cuda.View, stride: usize, col: usize, rows: usize) !void {
    const shape = weights.shape;
    const staged = buffers.staged.rows(rows * shape.heads);
    var emb = cuda.View.init(staged.ptr.?, .bf16, &.{ rows, shape.embedding() });
    if (weights.scale != 1.0) {
        emb = buffers.emb.rows(rows);
        try cuda.launched(cuda.kernels.zs_qwen4_ple_embed(&staged, &emb, @intCast(shape.heads), weights.scale));
    }
    const keys = buffers.keys.rows(rows);
    const vals = buffers.vals.rows(rows);
    const gated = buffers.gated.rows(rows);
    const pss = buffers.pss.rows(rows);
    const nrow = buffers.nrow.rows(rows);
    try weights.key.forward(emb, keys);
    try weights.value.forward(emb, vals);
    try cuda.launched(cuda.kernels.zs_qwen4_ple_gate(&keys, &vals, &h, &weights.norm_key.view, &weights.norm_query.view, &gated, &pss, weights.eps, @intCast(shape.streams)));
    try cuda.launched(zs_qwen4_ple_conv_slots(&gated, &pss, &weights.norm_conv.view, &weights.conv.view, &h, &nrow, &sid, &seg, &ptrs, @intCast(stride), @intCast(col), weights.eps, @intCast(shape.streams), @intCast(shape.dilation)));
}

pub fn convView(window: cuda.Buffer, channels: usize) cuda.View {
    var view = window.view;
    view.ndim = 2;
    view.size[0] = gdn.taps - 1;
    view.size[1] = @intCast(channels);
    return view;
}
