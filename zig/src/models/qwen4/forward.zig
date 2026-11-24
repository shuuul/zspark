//! Flash Next's decode-path forward: a window of rows at consecutive positions of one sequence, each row's bits
//! the bits of the serial step. A block's result stays pending until the next hyper-connection writes it back.

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("blocks/attention.zig");
const bf16 = @import("../../weights/bf16.zig");
const cuda = @import("../../cuda/cuda.zig");
const forms = @import("../../decode/forms.zig");
const gdn = @import("blocks/gdn.zig");
const hc = @import("blocks/hc.zig");
const model = @import("model.zig");
const moe = @import("blocks/moe.zig");
const numeric = @import("../../weights/numeric.zig");
const placeholders = @import("../../decode/placeholders.zig");
const nvfp4 = @import("../../weights/nvfp4.zig");
const ple = @import("blocks/ple.zig");
const state_windows = @import("blocks/state.zig");
const forward_state = @import("forward_state.zig");
const taps = @import("taps.zig");

pub const Error = error{ RowOutOfRange, TapFailed } || cuda.Error;

// A window's scratch and a sequence's state (`forward_state.zig`).
pub const Buffers = forward_state.Buffers;
pub const keptRows = forward_state.keptRows;
pub const State = forward_state.State;
pub const Image = forward_state.Image;

/// Where a forward can be observed, in the order it reaches them; the names follow the reference exporter's.
pub const Point = enum {
    embed_h,
    h_after_ple,
    h_in,
    attn_mixed,
    attn_inject,
    projection,
    block_out,
    branch,
    h_mid,
    mlp_mixed,
    mlp_inject,
    moe_logits,
    moe_picks,
    moe_weights,
    moe_y,
    moe_shared,
    final_h,
    final_mixed,
    logits,
};

/// Called at each point with the rows the forward just produced there. It may read them or overwrite them, which
/// is how a comparison feeds every block the reference's input instead of this forward's own.
pub const Tap = struct {
    context: *anyopaque,
    at: *const fn (context: *anyopaque, point: Point, layer: usize, view: cuda.View) anyerror!void,
    /// The tap only reads (a timer): the streams' squares are not launched again after a point that hands it
    /// the streams, so the launches are `forward`'s own and nothing else.
    passive: bool = false,

    fn rewrites(tap: ?Tap) bool {
        return if (tap) |t| !t.passive else false;
    }
};

pub const Snapshot = @import("snapshot.zig").Snapshot;

/// A hyper-connection's read-out of the streams: `mixed` [rows, hidden], and its inject gates where it has them.
/// Five launches on every checkpoint (spec 028): the activation sums `down`'s K slices itself.
/// `rounds.zig` calls it for a round of several streams; the one-stream path is unchanged.
pub fn readout(connection: model.HyperConnection, b: Buffers, rows: usize, shape: model.Shape, inject: ?cuda.Buffer) cuda.Error!void {
    return readoutAfter(connection, b, rows, shape, inject, null);
}

/// `readout`, and before it the write-back of `pending` where one is given: one launch with the norm
/// (`hc.site`, the same bits), which a caller that holds `joins` leaves to this.
pub fn readoutAfter(connection: model.HyperConnection, b: Buffers, rows: usize, shape: model.Shape, inject: ?cuda.Buffer, pending: ?hc.Pending) cuda.Error!void {
    const h = b.h.rows(rows);
    const normed = b.normed.rows(rows);
    const down = (if (connection.inject) b.down else b.down_mixer).rows(rows);
    const act = b.act.rows(rows);
    const up = b.up.rows(rows);
    if (fusedFaces(connection, rows, shape)) |faces| {
        // a prompt's window: three launches, the norm formed in the down tile and in the up epilogue (spec 148 WS-03)
        if (pending) |landing| try hc.writeback(h, b.pss.view, landing, shape.streams);
        hc.fused_readouts += 1;
        try hc.downWindow(h, b.pss.view, connection.scale.view, faces.down.weight.view, faces.down.sk, act, if (inject) |buffer| buffer.view else null, shape.eps, shape.streams, shape.low);
        return hc.upMixStreams(act, faces.up.weight.view, h, b.pss.view, connection.scale.view, b.mixed.rows(rows), shape.eps, shape.streams);
    }
    if (pending) |landing| {
        try hc.site(h, b.pss.view, landing, connection.scale.view, normed, shape.eps, shape.streams);
    } else try hc.normed(h, b.pss.view, connection.scale.view, normed, shape.eps, shape.streams);
    const slices = try connection.down.slices(normed, down);
    try hc.activate(slices, act, if (inject) |buffer| buffer.view else null, shape.streams, shape.low);
    if (hc.window_rows != 0 and rows >= hc.window_rows and shape.streams == 4 and shape.hidden % 32 == 0 and shape.low % 32 == 0) window: {
        // a prompt's window: the up product never written, `mix` in the GEMM's epilogue (spec 095): the same bits
        const linear = connection.up.plain() orelse break :window;
        if (linear.sk != 1) break :window;
        hc.window_readouts += 1;
        return hc.upMixWindow(act, linear.weight.view, normed, b.mixed.rows(rows), shape.streams, 2);
    }
    try connection.up.forward(act, up);
    try hc.mix(up, normed, b.mixed.rows(rows), shape.streams);
}

/// The read-out's two bf16 matmuls where `hc.downWindow` and `hc.upMixStreams` take them at `rows` rows.
fn fusedFaces(connection: model.HyperConnection, rows: usize, shape: model.Shape) ?struct { down: bf16.Linear, up: bf16.Linear } {
    if (!hc.fused or hc.fused_rows == 0 or rows < hc.fused_rows) return null;
    if (shape.streams != 4 or shape.hidden % 256 != 0 or shape.low % 32 != 0) return null;
    const down = connection.down.plain() orelse return null;
    const up = connection.up.plain() orelse return null;
    if (down.n <= 160 or down.n > 336 or up.sk != 1) return null;
    return .{ .down = down, .up = up };
}

/// Runs `tokens` as rows at positions `state.pos` onward and leaves their logits in `b.logits`. The committed
/// state is unchanged until `commit`.
pub fn forward(m: model.Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
    return forwardTapped(m, state, b, tokens, null);
}

/// Which rows of a window the head scores. A row's logits are the same bits whichever is asked.
pub const Score = enum {
    /// Every row, into the same row of `b.logits`.
    all,
    /// The last row alone, into `b.logits`' first row.
    last,
    none,
};

/// `forward` for a prompt's rows, whose logits nobody reads but the last one's: `score` is `.last` or `.none`.
pub fn forwardScoring(m: model.Model, state: *State, b: *Buffers, tokens: []const i32, score: Score) Error!void {
    try stage(m, state, b, tokens);
    return compute(m, state, b, tokens.len, null, score, null);
}

pub fn observe(tap: ?Tap, point: Point, layer: usize, view: cuda.View) Error!void {
    const t = tap orelse return;
    t.at(t.context, point, layer, view) catch |err| {
        std.log.err("tap at {t} of layer {d}: {t}", .{ point, layer, err });
        return error.TapFailed;
    };
}

/// `forward` with `tap` called at each `Point`.
pub fn forwardTapped(m: model.Model, state: *State, b: *Buffers, tokens: []const i32, tap: ?Tap) Error!void {
    try stage(m, state, b, tokens);
    return compute(m, state, b, tokens.len, null, .all, tap);
}

/// Whether `expect` and `ask` are heeded. A test's, to hold a prompt whose rows are gathered ahead to one's that are not;
/// nothing else clears it.
pub var gather_ahead: bool = true;

/// `next` is the window that will follow the next one staged, when that one is kept whole: its table rows are
/// gathered on a thread of their own while the device runs the window before it (`ple.Ahead`, spec 087). The
/// next `stage` takes the word and forgets it; `next` must live until then. A window that comes otherwise
/// than expected is gathered when it is staged, as one nobody expected.
pub fn expect(b: *Buffers, next: []const i32) void {
    b.expected = if (gather_ahead) next else &.{};
}

/// `window` is the round that will be staged next, as far as it is known: the kept token once the round before
/// is accepted, then with each draft the head has picked and goes on from. It follows `state`'s committed
/// tokens. Its table rows are gathered while the head's passes run (`ple.Ahead`, spec 089): asked again with
/// more drafts, only the new rows are read, and `stage` takes the rows of the tokens it is given that were
/// asked for, the others gathered there.
pub fn ask(m: model.Model, state: *const State, b: *Buffers, window: []const i32) void {
    if (!gather_ahead or window.len > b.rows) return;
    var branches: usize = 0;
    for (m.layers) |layer| if (layer.ple) |branch| {
        b.ahead[branches].start(branch.ngram, branch.table, state.ple[branches], &.{}, @ptrCast(window));
        branches += 1;
    };
}

/// The host's part of a window: token ids, and each PLE's table rows.
pub fn stage(m: model.Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
    std.debug.assert(tokens.len >= 1 and tokens.len <= b.rows and tokens.len <= m.max_rows);
    if (lands(state.*, b.*, tokens.len)) try land(state);
    const next = b.expected;
    b.expected = &.{};
    // By now the device is idle: what committed the window before waited for it (the position's copy), so rows
    // gathered here are time nothing overlaps. A prompt's window has had its own gathered since the window
    // before it was staged (`expect`), a reply's round since its tokens were known (`ask`).
    var branches: usize = 0;
    for (m.layers) |layer| if (layer.ple) |branch| {
        try ple.gatherAfter(branch.ngram, branch.table, state.ple[branches], &b.ple[branches], &b.ahead[branches], @ptrCast(tokens));
        branches += 1;
    };
    try b.ids.uploadAt(0, std.mem.sliceAsBytes(tokens));
    b.image_rows = 0;
    if (state.image) |image| {
        // in parts of 256 rows, through the stack
        var pairs: [256][2]i32 = undefined;
        var start: usize = 0;
        while (start < tokens.len) : (start += pairs.len) {
            const at = image.within(state.pos + start, @min(pairs.len, tokens.len - start), &pairs);
            for (pairs[0..at]) |*pair| pair[0] += @intCast(start);
            if (at > 0) try b.image_pairs.uploadAt(b.image_rows * @sizeOf([2]i32), std.mem.sliceAsBytes(pairs[0..at]));
            b.image_rows += at;
        }
    }
    branches = 0;
    for (m.layers) |layer| if (layer.ple) |branch| {
        try b.ple[branches].upload(tokens.len, branch.table.stride());
        // the copies have returned, so the memory the rows came from is free for the next window's
        if (next.len > 0) b.ahead[branches].start(branch.ngram, branch.table, state.ple[branches], @ptrCast(tokens), @ptrCast(next));
        branches += 1;
    };
}

/// The card's forms (spec 143 WS-06, `decode/forms.zig`'s rule, where `model.prompt_form` is `.card`): a window of
/// more than this many rows is a prompt's, its MXFP8 linears on FP8 rows (`qmm_a8.cu`, vLLM b12x's quantizer) and
/// its routed experts W4A4 (the card's `B12X_W4A16_A4_PREFILL` path); a step, a verify window and a prompt window
/// of up to it keep bf16 rows. 80 is the card's largest CUDA graph (vLLM `config/vllm.py`: 8 sequences of five
/// tokens with MTP, twice): its batches of up to 80 tokens run the experts A16 under a capture and its MXFP8 plans
/// may tune A16 there. A test lowers it, never under `forward_max_rows`, or raises it past its windows (one form).
pub var bf16_rows: usize = card_bf16_rows;
pub const card_bf16_rows = 80;
comptime {
    forms.checkBound(card_bf16_rows);
}

/// The first row of a window of `rows` rows that takes the quantized forms: 0, or `rows` where none does.
pub fn quantizedFrom(rows: usize) usize {
    return forms.quantizedFrom(rows, bf16_rows);
}

/// Sets the form of `m`'s calls to a window whose quantized rows start at `from` (`quantizedFrom`, a round's
/// `forms.formFrom`) and returns the form it replaces, which the caller puts back (`forms.none` outside a window).
pub fn setForm(m: model.Model, from: usize) usize {
    const was = m.scratch.quantize_from;
    m.scratch.quantize_from = from;
    return was;
}

/// `compute` with no tap and no context bound. `stage` first: a capture records this and not the copies.
pub fn device(m: model.Model, state: *State, b: *Buffers, rows: usize, score: Score) Error!void {
    return compute(m, state, b, rows, null, score, null);
}

/// The device's part: launches only, reading the staged rows, so a graph can hold them. What differs from one
/// call to the next must be among a graph's keys: the rows, which DeltaNet state is the committed one, and
/// `bound`, the context the attention's launches cover (null: exactly the window's end).
pub fn compute(m: model.Model, state: *State, b: *Buffers, rows: usize, bound: ?usize, score: Score, tap: ?Tap) Error!void {
    const was = setForm(m, quantizedFrom(rows));
    defer _ = setForm(m, was);
    const shape = m.shape;
    const h = b.h.rows(rows);
    var branches: usize = 0;
    try hc.embed(b.ids.rows(rows), m.embed.view, h, shape.streams);
    // an image prompt's placeholder rows take the tower's features in every stream, before the first PLE
    // (TensorFold 0.6.5's `compute`, `features`); never in a graph (`forward_graphs.holds`)
    if (b.image_rows > 0) try embedImage(state.image.?.features, b.image_pairs.rows(b.image_rows), h, shape.streams);
    try observe(tap, .embed_h, 0, h);

    var pending: hc.Pending = .none;
    var linear: usize = 0;
    var attending: usize = 0;
    branches = 0;
    for (m.layers, 0..) |layer, index| {
        if (layer.ple) |branch| {
            // the n-gram branch adds to the streams as they stand, so what is pending lands first
            if (pending != .none) {
                try hc.writeback(h, b.pss.view, pending, shape.streams);
                pending = .none;
            }
            // the reference takes this layer's h_in here, before the branch
            try observe(tap, .h_in, index, h);
            try ple.forward(branch.weights, b.ple[branches], state.ple[branches], h, rows);
            try observe(tap, .h_after_ple, index, h);
            branches += 1;
        }
        const site: Site = switch (layer.block) {
            .delta_net => .{ .delta_net = .{ .state = state, .linear = linear } },
            .attention => .{ .attention = .{ .cache = state.caches[attending], .position = state.position, .bound = bound, .rope = state.rope.view() } },
        };
        switch (layer.block) {
            .delta_net => linear += 1,
            .attention => attending += 1,
        }
        pending = try layerForward(m, layer, index, site, b, rows, pending, layer.ple == null, tap);
        try capture(b, index, rows, &pending, shape.streams);
    }
    try finish(m, m.mixer, m.layers.len, b, rows, pending, tap);
    switch (score) {
        .all => {
            std.debug.assert(rows <= b.scoredRows());
            try m.head.forward(b.mixed.rows(rows), b.logits.rows(rows));
            try observe(tap, .logits, m.layers.len, b.logits.rows(rows));
        },
        .last => try m.head.forward(b.mixed.span(rows - 1, 1), b.logits.rows(1)),
        .none => {},
    }
}

/// What a layer's block reads and writes of a sequence.
pub const Site = union(enum) {
    /// The sequence's DeltaNet states, and which of them is this layer's.
    delta_net: struct { state: *State, linear: usize },
    /// The layer's cache, the position of the window's first row, the context its launches cover (null:
    /// exactly the window's end), and the sequence's rotary positions (`attention.Rope`; null: the rows' own,
    /// as on two ranks, which take no image yet).
    attention: struct { cache: attention.Cache, position: attention.Position, bound: ?usize, rope: ?cuda.View = null },
};

/// Whether a write-back goes with the read-out after it (`hc.joins`): not under a tap, which reads h between
/// the two and may write over it.
fn joins(tap: ?Tap, rows: usize, pending: hc.Pending) bool {
    return tap == null and hc.joins(pending, rows);
}

/// The block between the two read-outs. `partial` null: the out projection writes the bf16 branch (`b.branch`,
/// or the attention scratch's) and that view is returned. Set: it writes that fp32 [rows, hidden] instead, and
/// the caller sums the ranks before the write-back.
pub fn runBlock(m: model.Model, layer: model.Layer, index: usize, site: Site, b: *Buffers, rows: usize, tap: ?Tap, mixed: cuda.View, partial: ?cuda.View) Error!cuda.View {
    const shape = m.shape;
    const branch = partial orelse b.branch.rows(rows);
    switch (layer.block) {
        .delta_net => |dn| {
            const state = site.delta_net.state;
            const linear = site.delta_net.linear;
            // a window wider than the rows each layer keeps: one projection and one scratch for every layer
            const wide = widerThanKept(state.*, b.*, rows);
            std.debug.assert(wide or state.wanted.len == 0);
            const projection = if (wide) b.projection_wide.?.rows(rows) else b.projections[linear].rows(rows);
            try dn.projection.forward(mixed, projection);
            try observe(tap, .projection, index, projection);
            const current = state.current[linear];
            // the wanted states left to `settle`: the chain's replay scratch goes into the highest one's buffer
            var stash: gdn.Scratch = undefined;
            if (state.stashed) {
                std.debug.assert(wide and state.wanted.len > 0 and defers(state.*, b.*, rows));
                const holder = state.wanted[state.wanted.len - 1];
                try holder.reach(state.*, linear);
                stash = holder.stash(state.*, linear, rows);
            }
            var conv_state = state.conv[linear].view;
            conv_state.ndim = 2;
            conv_state.size[0] = gdn.taps - 1;
            conv_state.size[1] = @intCast(shape.delta_net.conv());
            try gdn.chain(dn.kernelWeights(), .{
                .projection = projection,
                .conv_state = conv_state,
                .state_in = state.recurrent[linear][current].view,
                .state_out = if (wide or stores(rows)) state.recurrent[linear][1 - current].view else null,
                .out = b.delta_out.rows(rows),
                .sums = b.delta_sums.view,
                .scratch = if (state.stashed) &stash else if (wide) &state.wide.? else &state.scratch[linear],
                .work = b.delta_q.view,
                .behind = if (wide or rows >= gdn.wide_rows) null else state.position.behindView(),
            }, rows, shape.eps);
            if (wide) {
                // What `commit` and `Snapshot.within` read of a kept window after it ran, taken here: the conv
                // inputs after the whole window, and each wanted row's state and conv inputs.
                const fresh: state_windows.Fresh = .{ .view = projection, .layer = 0, .row = shape.delta_net.projection(), .rows = rows };
                try state_windows.shiftWindows(state.conv_next[linear].view, fresh, rows);
                // Lowest first, each replayed on from the one below it: the rows between the two and not the
                // window's again (a replay of a 2048-row window to its end is 48 ms for the 36 layers).
                var below: usize = 0;
                var from = state.recurrent[linear][current];
                for (state.wanted) |wanted| {
                    const keep = wanted.pos - state.pos;
                    std.debug.assert(keep >= 1 and keep <= rows and keep >= below);
                    try wanted.reach(state.*, linear);
                    if (state.stashed) {
                        std.debug.assert(keep < rows);
                    } else if (keep < rows) {
                        if (keep == below) {
                            try wanted.recurrent[linear].copyLaunched(from);
                        } else {
                            try gdn.replayFrom(from.view, state.wide.?, below, keep - below, wanted.recurrent[linear].view);
                        }
                        below = keep;
                        from = wanted.recurrent[linear];
                    } else {
                        try wanted.recurrent[linear].copyLaunched(state.recurrent[linear][1 - current]);
                    }
                    try wanted.conv[linear].copyLaunched(state.conv[linear]);
                    try state_windows.shiftWindows(wanted.conv[linear].view, fresh, keep);
                }
            }
            try observe(tap, .block_out, index, b.delta_out.rows(rows));
            try dn.out.forward(b.delta_out.rows(rows), branch);
            return branch;
        },
        .attention => |weights| {
            const shared = m.attention.?;
            const scratch = b.attention.?;
            const at = site.attention;
            try scratch.poison();
            try attention.forward(shared.shape, weights, shared.rotary, at.cache, scratch, at.position, mixed, rows, at.bound, partial, at.rope);
            // the projection and the gated output are observed after the block: forcing them would not re-run it
            try observe(tap, .projection, index, scratch.pa.rows(rows));
            try observe(tap, .block_out, index, scratch.gated.rows(rows));
            return partial orelse scratch.branch.rows(rows);
        },
    }
}

/// After layer `index`'s MoE is pending: write it back (not through `hc.site`) and, when `index` is a DFlash2
/// tap layer, mean the four streams into that slot. Sets `pending` to `.none` so the next layer does not add
/// the MoE twice. No launch when `setTaps` was not called. Two ranks call it a rank at a time
/// (`ranks_forward.launch`), with the gathered partials pending (`.ranks_sum`).
pub fn capture(b: *Buffers, index: usize, rows: usize, pending: *hc.Pending, nstreams: usize) cuda.Error!void {
    const slot = taps.tapIndexOf(b.tap_layers, index) orelse return;
    const h = b.h.rows(rows);
    if (pending.* != .none) {
        try hc.writeback(h, b.pss.view, pending.*, nstreams);
        pending.* = .none;
    }
    try hc.streamMean(h, b.tap(slot, rows), nstreams, b.tap_mean);
}

/// One layer after its PLE: the write-back of what is pending, the block between its two read-outs, and the MoE,
/// whose slots it returns still pending. `h_in`: whether the tap takes the layer's input here.
pub fn layerForward(m: model.Model, layer: model.Layer, index: usize, site: Site, b: *Buffers, rows: usize, pending: hc.Pending, h_in: bool, tap: ?Tap) Error!hc.Pending {
    const shape = m.shape;
    const h = b.h.rows(rows);
    if (!joins(tap, rows, pending)) try hc.writeback(h, b.pss.view, pending, shape.streams);
    if (h_in) try observe(tap, .h_in, index, h);
    // an overwritten h needs its squares again
    if (Tap.rewrites(tap)) try hc.writeback(h, b.pss.view, .none, shape.streams);
    try readoutAfter(layer.attn_hc, b.*, rows, shape, b.inject_attn, if (joins(tap, rows, pending)) pending else null);
    try observe(tap, .attn_mixed, index, b.mixed.rows(rows));
    try observe(tap, .attn_inject, index, b.inject_attn.rows(rows));
    const mixed = b.mixed.rows(rows);
    const branch = try runBlock(m, layer, index, site, b, rows, tap, mixed, null);
    try observe(tap, .branch, index, branch);
    const landing: hc.Pending = .{ .branch = .{ .branch = branch, .inject = b.inject_attn.view } };
    if (!joins(tap, rows, landing)) try hc.writeback(h, b.pss.view, landing, shape.streams);
    try observe(tap, .h_mid, index, h);
    if (Tap.rewrites(tap)) try hc.writeback(h, b.pss.view, .none, shape.streams);
    try readoutAfter(layer.mlp_hc, b.*, rows, shape, b.inject_mlp, if (joins(tap, rows, landing)) landing else null);
    try observe(tap, .mlp_mixed, index, b.mixed.rows(rows));
    try observe(tap, .mlp_inject, index, b.inject_mlp.rows(rows));
    try moe.forward(layer.moe, b.moe, b.mixed.rows(rows), rows, m.scratch.quantize_from);
    try observe(tap, .moe_logits, index, b.moe.logits.rows(rows));
    try observe(tap, .moe_picks, index, b.moe.picks.rows(rows));
    try observe(tap, .moe_weights, index, b.moe.weights.rows(rows));
    try observe(tap, .moe_y, index, b.moe.y.rows(rows * (shape.top + 1)));
    try observe(tap, .moe_shared, index, b.moe.shared_y.rows(rows));
    return moe.pending(b.moe, b.inject_mlp.view);
}

/// The last write-back and `mixer`'s read-out: `b.h` holds the final streams and `b.mixed` what a head reads.
pub fn finish(m: model.Model, mixer: model.HyperConnection, layers: usize, b: *Buffers, rows: usize, pending: hc.Pending, tap: ?Tap) Error!void {
    const shape = m.shape;
    const h = b.h.rows(rows);
    if (!joins(tap, rows, pending)) try hc.writeback(h, b.pss.view, pending, shape.streams);
    try observe(tap, .final_h, layers, h);
    if (Tap.rewrites(tap)) try hc.writeback(h, b.pss.view, .none, shape.streams);
    try readoutAfter(mixer, b.*, rows, shape, null, if (joins(tap, rows, pending)) pending else null);
    try observe(tap, .final_mixed, layers, b.mixed.rows(rows));
}

pub const Graphs = @import("forward_graphs.zig").Graphs;

/// Whether a window of `rows` rows is wider than the rows each DeltaNet layer keeps of one: it then runs on
/// the scratch the layers share, is committed whole, and gives a state inside it only where one was asked
/// before it ran (`State.wanted`).
pub fn widerThanKept(state: State, b: Buffers, rows: usize) bool {
    return state.recurrent.len > 0 and rows > @min(state.kept, b.kept);
}

/// Whether a window of `rows` rows no wider than the rows kept writes the DeltaNet states after its rows (spec 096
/// WS-02). A row's window is kept whole, so its state is always the next one; a window of drafts is kept whole a
/// round in five, and its 3 MB a layer written for nothing were paid by the next GEMM (`out_proj`, 16 to 18 us
/// a layer, the dirty lines going back). Its rows wait in the replay scratch instead (`behind`), replayed by the
/// next chain on the state it loads anyway, as a window kept in part does: the same bits. From `gdn.wide_rows`
/// the wide kernel, which always writes it.
pub fn stores(rows: usize) bool {
    return rows == 1 or rows >= gdn.wide_rows;
}

/// The rows of a chain's replay scratch that a DeltaNet layer's state buffer holds (`Snapshot.stash`): 150 on
/// Flash Next.
pub fn stashRows(state: State) usize {
    const wide = state.wide orelse return 0;
    const nk: usize = @intCast(wide.k.view.size[1]);
    const nv: usize = @intCast(wide.v.view.size[1]);
    return state.recurrent[0][0].bytes() / (nk * gdn.key_dim * 4 + 2 * nv * 4 + nv * gdn.value_dim * 2);
}

/// Whether the states wanted inside the next window, of `rows` rows, can wait for `settle` (`State.stashed`):
/// the window is wider than the rows each layer keeps, its replay scratch fits a state's buffer, and the
/// shared scratch holds a state, where `settle` replays the highest one before it copies it over the scratch
/// in its buffer.
pub fn defers(state: State, b: Buffers, rows: usize) bool {
    if (!widerThanKept(state, b, rows) or rows > stashRows(state)) return false;
    return state.wide.?.k.bytes() >= state.recurrent[0][0].bytes();
}

/// The DeltaNet states of `wanted`, which a forward that found them `stashed` left out: each layer's chain
/// replayed from the state the window started at to each one's row, lowest first and each from the one
/// below it, as the forward does it. After the window's `commit` and before the next forward on `state`,
/// which writes over the state the window started at; `start` is the position the window ran from.
pub fn settle(state: State, wanted: []const Snapshot, start: usize) cuda.Error!void {
    if (wanted.len == 0) return;
    const rows = state.pos - start;
    const holder = wanted[wanted.len - 1];
    const wide = state.wide.?;
    for (state.recurrent, state.current, 0..) |pair, current, linear| {
        const scratch = holder.stash(state, linear, rows);
        // the shared scratch's k as one state, which nothing reads between two forwards
        var over = pair[0];
        over.view.ptr = wide.k.view.ptr;
        var below: usize = 0;
        var from = pair[1 - current];
        for (wanted, 0..) |one, index| {
            const keep = one.pos - start;
            std.debug.assert(keep >= 1 and keep < rows and keep >= below);
            const last = index + 1 == wanted.len;
            const out = if (last) over else one.recurrent[linear];
            if (keep == below) {
                try out.copyLaunched(from);
            } else {
                try gdn.replayFrom(from.view, scratch, below, keep - below, out.view);
            }
            if (last) try one.recurrent[linear].copyLaunched(over);
            below = keep;
            from = one.recurrent[linear];
        }
    }
}

/// Keeps the first `keep` rows of the window `tokens` that the last `forward` ran: all of them where the
/// window is wider than the rows each layer keeps (`widerThanKept`).
pub fn commit(m: model.Model, state: *State, b: Buffers, tokens: []const i32, keep: usize) cuda.Error!void {
    const rows = tokens.len;
    std.debug.assert(keep >= 1 and keep <= rows);
    for (state.ple, b.ple) |*branch, buffers| try ple.commit(branch, buffers, @ptrCast(tokens), keep);
    const dn = m.shape.delta_net;
    if (widerThanKept(state.*, b, rows)) {
        // kept whole: the window wrote the state after its rows, and its forward left the conv inputs
        std.debug.assert(keep == rows);
        for (state.current, state.conv, state.conv_next) |*current, conv, next| {
            current.* = 1 - current.*;
            try conv.copyLaunched(next);
        }
    } else {
        const each = state_windows.once and state.conv.len <= state_windows.most;
        for (state.current, state.conv, b.projections) |*current, conv, projection| {
            // a window that wrote the state after all its rows (`stores`) and is kept whole moves to the other
            // side; anything else leaves the committed side where it is, behind by the rows kept, which the next
            // chain replays (`behind`)
            if (keep == rows and stores(rows)) current.* = 1 - current.*;
            if (!each) try state_windows.shiftWindows(conv.view, .{ .view = projection.view, .layer = 0, .row = dn.projection(), .rows = rows }, keep);
        }
        if (each and state.conv.len > 0) {
            // every layer's window in one launch: the same copies
            var olds: [state_windows.most]cuda.View = undefined;
            var freshes: [state_windows.most]cuda.View = undefined;
            for (state.conv, b.projections, 0..) |conv, projection, i| {
                olds[i] = conv.view;
                freshes[i] = projection.view;
            }
            try state_windows.shiftEach(olds[0..state.conv.len], freshes[0..state.conv.len], dn.projection(), rows, keep);
        }
    }
    if (state.recurrent.len > 0 and !widerThanKept(state.*, b, rows)) state.behind = if (keep == rows and stores(rows)) 0 else state.behind + keep;
    try state.position.put(state.position.host + keep, state.behind);
    state.pos += keep;
    if (state.image) |image| {
        if (state.pos > image.rows[image.rows.len - 1]) state.image = null;
    }
}

/// Rows `pairs` int32 [K, 2] name of h bf16 [rows, streams * hidden] take features bf16 [N, hidden]'s rows, in
/// every stream (`decode/placeholders.zig`).
pub const embedImage = placeholders.embed;

/// Whether a window of `rows` rows cannot go on from the rows the states are behind by: one through the wide
/// kernels, or one whose rows the scratch has no room for after them. `stage` then replays them first.
pub fn lands(state: State, b: Buffers, rows: usize) bool {
    return (state.behind > 0 and widerThanKept(state, b, rows)) or landsAt(state.behind, rows, state.kept);
}

/// `lands` of states `behind` rows behind in a scratch of `kept` rows, for a window no wider than the rows
/// kept; a pool's slot asks it too (`rounds.stage`, spec 144 WS-09).
pub fn landsAt(behind: usize, rows: usize, kept: usize) bool {
    return behind > 0 and (rows >= gdn.wide_rows or behind + rows > kept);
}

/// The committed side of every DeltaNet state brought to `pos`: the rows it is behind by replayed, as `commit`
/// did for each window kept in part before spec 089. The same bits either way: a replay is the chain's update
/// a row at a time from the fp32 state the row before left.
pub fn land(state: *State) cuda.Error!void {
    if (state.behind == 0) return;
    try landStates(state.recurrent, state.current, state.scratch, state.behind);
    state.behind = 0;
    try state.position.put(state.position.host, 0);
}

/// `land`'s replays, for states held as a session's are, each layer's pair, its committed side and its scratch:
/// a pool's slot lands by it too (`rounds.land`).
pub fn landStates(recurrent: []const [2]cuda.Buffer, current: []u1, scratch: []const gdn.Scratch, behind: usize) cuda.Error!void {
    var batch: Lands = .{};
    try batch.add(recurrent, current, scratch, behind);
    try batch.flush();
}

/// Landings gathered into launches of `gdn.land` (spec 146 WS-07): every layer of every stream that lands, its
/// committed side flipped as its replay is added. `flush` before anything reads the states.
pub const Lands = struct {
    entries: [64]gdn.Landing = undefined,
    count: usize = 0,

    /// One stream's layers, `behind` rows behind, as `landStates` takes them; none when `behind` is 0.
    pub fn add(self: *Lands, recurrent: []const [2]cuda.Buffer, current: []u1, scratch: []const gdn.Scratch, behind: usize) cuda.Error!void {
        if (behind == 0) return;
        for (recurrent, current, scratch) |pair, *side, held| {
            if (self.count == self.entries.len) try self.flush();
            self.entries[self.count] = .of(pair[side.*].view, held, behind, pair[1 - side.*].view);
            self.count += 1;
            side.* = 1 - side.*;
        }
    }

    pub fn flush(self: *Lands) cuda.Error!void {
        defer self.count = 0;
        try gdn.land(self.entries[0..self.count]);
    }
};

/// Layer `linear`'s DeltaNet state at `pos` into `dest`, as a launch: the committed side, replayed over the
/// rows it is behind by.
pub fn committed(state: State, linear: usize, dest: cuda.Buffer) cuda.Error!void {
    return committedOf(state.recurrent[linear], state.current[linear], state.scratch[linear], state.behind, dest);
}

/// `committed` of one layer's pair, committed side and scratch, `behind` rows behind: a pool's slot's too
/// (`rounds.committed`).
pub fn committedOf(pair: [2]cuda.Buffer, current: u1, scratch: gdn.Scratch, behind: usize, dest: cuda.Buffer) cuda.Error!void {
    if (behind == 0) return dest.copyLaunched(pair[current]);
    return gdn.replay(pair[current].view, scratch, behind, dest.view);
}

/// Row `row`'s most likely token: the largest logit, the lower id among equals. Waits for the forward.
pub fn argmax(gpa: Allocator, b: Buffers, vocab: usize, row: usize) !i32 {
    return (try top(gpa, b, vocab, row)).token;
}

pub const Top = struct {
    token: i32,
    logit: f32,
    /// How far the runner-up's logit is below.
    margin: f32,
};

/// `argmax` with the winning logit and its margin over the runner-up.
pub fn top(gpa: Allocator, b: Buffers, vocab: usize, row: usize) !Top {
    const logits = try gpa.alloc(u16, vocab);
    defer gpa.free(logits);
    try cuda.synchronize();
    try downloadRow(b.logits, row, vocab, logits);
    var best: usize = 0;
    var second = -std.math.inf(f32);
    for (logits, 0..) |bits, id| {
        const v = numeric.f32FromBf16(bits);
        const leader = numeric.f32FromBf16(logits[best]);
        if (v > leader) {
            second = leader;
            best = id;
        } else if (id != best) second = @max(second, v);
    }
    const logit = numeric.f32FromBf16(logits[best]);
    return .{ .token = @intCast(best), .logit = logit, .margin = logit - second };
}

/// Row `row`'s logits as bf16 bits, for the host to choose a token from. `[vocab]`.
/// Rows of logits copied to the host by `logitsRow`, and the nanoseconds the copies took after the device
/// was waited for: what a choice made on the device would not pay (spec 062).
pub var rows_read: u64 = 0;
pub var reading: u64 = 0;

pub fn logitsRow(b: Buffers, row: usize, out: []u16) cuda.Error!void {
    try cuda.synchronize();
    const before = cuda.monotonic();
    defer {
        reading += cuda.monotonic() - before;
        rows_read += 1;
    }
    return downloadRow(b.logits, row, out.len, out);
}

pub fn downloadRow(buffer: cuda.Buffer, row: usize, width: usize, out: []u16) cuda.Error!void {
    return buffer.downloadAt(row * width * 2, std.mem.sliceAsBytes(out));
}
