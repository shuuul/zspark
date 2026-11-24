//! GLM-5.3-Flash's KDA (Kimi delta attention) over a rank's heads: the projections, the chain of a window's
//! rows over each head's 128 by 128 state, and what a sequence keeps between windows (the state, the conv's
//! three rows). TensorFold's `kda.py`, `forward.py` `kda_block`, `commit` and `State`; the chain is its
//! `kda.cu`, taken.
//!
//! A window's rows chain from the committed state and leave the state after the last row beside it; `commit`
//! keeps the first `keep` rows: all of them takes that state, fewer replays the saved rows. A window of up to
//! `recurrent_rows` rows (steps, verify windows) chains row by row, fp32: a row's output and the state after it
//! depend on the rows before it alone, so a window and serial steps give the same bits. A longer one, spec 115's
//! prompt form, runs chunked (`kda_chunk.cu`, spec 116 WS-06; FLA's `chunk_kda`, vLLM's and SGLang's prompt
//! form): chunks of `chunk` rows on the sequence's positions, so its rows' bits are the same however the prompt
//! is cut into windows at multiples of `chunk` (`decode/draft.zig`'s `prompt_align`); it is committed whole, and
//! a mark inside it, at a chunk's start, is asked before it runs (`State.ask`): the chunked chain writes the
//! state it holds there into the mark's buffers, no chunk run twice.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const Config = @import("config.zig").Config;
const layout = @import("layout.zig");
const share = @import("share.zig");
const face = @import("../../../weights/face.zig");
const Face = face.Face;
const Part = face.Part;

/// A head's key and value width, which the chain is written for.
pub const head_dim = 128;
/// Windows of up to this many rows run the recurrent chain, a step's bits; longer ones run chunked: spec 115's
/// forms, `forward.card_bf16_rows`.
pub const recurrent_rows = @import("forward.zig").card_bf16_rows;
/// A chunk's rows, on the sequence's positions (FLA's `BT`).
pub const chunk = 64;

/// The switch of the chunked chain, on: off, every window chains row by row, a step's bits (what the tests that
/// hold replies to serial steps take). Read when a `State` is made and by every window after it.
pub var chunked_windows = true;

/// Whether a window of `rows` rows runs chunked.
pub fn chunks(rows: usize) bool {
    return chunked_windows and rows > recurrent_rows;
}

/// The most states a chunked window is asked for (`State.ask`): a 4,096-row window's chunk starts past its first.
pub const most_asked = 64;

/// The chunks a window of up to `rows` rows may be cut into, wherever it starts.
fn chunksFor(rows: usize) usize {
    return rows / chunk + 2;
}
/// Rows the conv keeps of what came before: its taps less one.
pub const window = 3;

pub const LoadError = error{ MissingTensor, UnsupportedShape } || Allocator.Error || cuda.Error;

/// A layer's tensors, whole or a rank's shares: the linears as their checkpoint stores them (BF16, or MXFP8 in
/// FP4-Local), the rest bytes, bf16 but `a_log` and `dt_bias`, fp32.
pub const Parts = struct {
    q: Part,
    k: Part,
    v: Part,
    f_a: Part,
    g_a: Part,
    b: Part,
    f_b: Part,
    g_b: Part,
    o: Part,
    q_conv: []const u8,
    k_conv: []const u8,
    v_conv: []const u8,
    a_log: []const u8,
    dt_bias: []const u8,
    norm: []const u8,

    /// The checkpoint's leaf of each field, in the fields' order.
    pub const leaves = [_][]const u8{
        "q_proj.weight",   "k_proj.weight",   "v_proj.weight",   "f_a_proj.weight", "g_a_proj.weight",
        "b_proj.weight",   "f_b_proj.weight", "g_b_proj.weight", "o_proj.weight",   "q_conv1d.weight",
        "k_conv1d.weight", "v_conv1d.weight", "A_log",           "dt_bias",         "o_norm.weight",
    };
};

/// A KDA layer's weights for `heads` heads.
pub const Weights = struct {
    /// [q | k | v | f_a | g_a | b | zeros to a multiple of 8, dims]: one row a window's row, each run of one
    /// storage one product.
    proj: Face,
    /// [heads 128, 128]: the decay's and the output gate's second halves.
    f_b: Face,
    g_b: Face,
    /// [dims, heads 128]
    o: Face,
    /// bf16 [3 heads 128, 4]: q's, k's and v's depthwise taps.
    conv: cuda.Buffer,
    /// fp32 [heads] and [heads 128]
    a_log: cuda.Buffer,
    dt_bias: cuda.Buffer,
    /// bf16 [128]
    norm: cuda.Buffer,
    heads: usize,
    eps: f32,
    lower: f32,

    /// The conv's channels: q, k and v of every head.
    pub fn channels(self: Weights) usize {
        return 3 * self.heads * head_dim;
    }

    pub fn init(gpa: Allocator, parts: Parts, c: *const Config, rows: cuda.Scratch.Rows) LoadError!Weights {
        const dims = c.hidden;
        if (c.kda_head_dim != head_dim or c.conv_kernel != window + 1) return error.UnsupportedShape;
        const heads = parts.a_log.len / 4;
        const inner = heads * head_dim;
        if (heads == 0 or parts.a_log.len != heads * 4 or parts.dt_bias.len != inner * 4 or parts.norm.len != head_dim * 2) return error.UnsupportedShape;
        for ([_]Part{ parts.q, parts.k, parts.v }) |one| if (!share.shaped(one, inner, dims)) return error.UnsupportedShape;
        for ([_][]const u8{ parts.q_conv, parts.k_conv, parts.v_conv }) |one| if (one.len != inner * (window + 1) * 2) return error.UnsupportedShape;
        if (!share.shaped(parts.f_a, head_dim, dims) or !share.shaped(parts.g_a, head_dim, dims) or !share.shaped(parts.b, heads, dims)) return error.UnsupportedShape;
        if (!share.shaped(parts.f_b, inner, head_dim) or !share.shaped(parts.g_b, inner, head_dim) or !share.shaped(parts.o, dims, inner)) return error.UnsupportedShape;

        // rows of a whole number of 16 bytes, so that the f_a and g_a columns of a row are a matmul's input
        const outputs = 3 * inner + 2 * head_dim + heads;
        const pad = try gpa.alloc(u8, (std.mem.alignForward(usize, outputs, 8) - outputs) * dims * 2);
        defer gpa.free(pad);
        @memset(pad, 0);
        const stack = [_]Part{ parts.q, parts.k, parts.v, parts.f_a, parts.g_a, parts.b, .{ .bf16 = pad } };
        var proj = try share.faceOf(gpa, stack[0 .. stack.len - @intFromBool(pad.len == 0)], dims, rows);
        errdefer proj.deinit();
        var f_b = try share.faceOf(gpa, &.{parts.f_b}, head_dim, rows);
        errdefer f_b.deinit();
        var g_b = try share.faceOf(gpa, &.{parts.g_b}, head_dim, rows);
        errdefer g_b.deinit();
        var o = try share.faceOf(gpa, &.{parts.o}, inner, rows);
        errdefer o.deinit();
        var conv: cuda.Buffer = try .init(.bf16, &.{ 3 * inner, window + 1 });
        errdefer conv.deinit();
        for ([_][]const u8{ parts.q_conv, parts.k_conv, parts.v_conv }, 0..) |one, at| try conv.uploadAt(at * one.len, one);
        var a_log: cuda.Buffer = try .init(.f32, &.{heads});
        errdefer a_log.deinit();
        try a_log.upload(parts.a_log);
        var dt_bias: cuda.Buffer = try .init(.f32, &.{inner});
        errdefer dt_bias.deinit();
        try dt_bias.upload(parts.dt_bias);
        var norm: cuda.Buffer = try .init(.bf16, &.{head_dim});
        errdefer norm.deinit();
        try norm.upload(parts.norm);
        return .{ .proj = proj, .f_b = f_b, .g_b = g_b, .o = o, .conv = conv, .a_log = a_log, .dt_bias = dt_bias, .norm = norm, .heads = heads, .eps = @floatCast(c.eps), .lower = @floatCast(c.gate_lower_bound) };
    }

    /// Layer `layer` of `source` (anything with `get(name)`) as `rank` holds it: half of the heads.
    pub fn load(arena: Allocator, gpa: Allocator, source: anytype, c: *const Config, layer: usize, rank: usize, rows: cuda.Scratch.Rows) LoadError!Weights {
        var parts: Parts = undefined;
        inline for (comptime std.meta.fieldNames(Parts), Parts.leaves) |field, leaf| {
            const name = try std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.self_attn.{s}", .{ layer, leaf });
            if (comptime @FieldType(Parts, field) == Part) {
                @field(parts, field) = try share.partOf(arena, source, c, name[0 .. name.len - ".weight".len], rank);
            } else {
                const tensor = try share.get(arena, source, name, rank) orelse {
                    std.log.err("{s}: not in the checkpoint", .{name});
                    return error.MissingTensor;
                };
                const fp32 = comptime (std.mem.eql(u8, field, "a_log") or std.mem.eql(u8, field, "dt_bias"));
                if (tensor.dtype != if (fp32) .F32 else .BF16) return error.UnsupportedShape;
                @field(parts, field) = tensor.data;
            }
        }
        return init(gpa, parts, c, rows);
    }

    pub fn deinit(self: *Weights) void {
        self.proj.deinit();
        self.f_b.deinit();
        self.g_b.deinit();
        self.o.deinit();
        self.conv.deinit();
        self.a_log.deinit();
        self.dt_bias.deinit();
        self.norm.deinit();
        self.* = undefined;
    }
};

/// What `layers` KDA layers keep of a sequence, and of its last window of up to `rows` rows.
pub const State = struct {
    layers: usize,
    heads: usize,
    rows: usize,
    /// A projection row's width, the weights' `proj.n`.
    width: usize,
    /// bf16 [layers, 3, 3 heads 128]: the rows the conv reads before a window's first.
    conv: cuda.Buffer,
    /// fp32 [2, layers, heads, 128, 128]: the committed states at `cur`, the last window's beside them.
    rec: cuda.Buffer,
    cur: u1 = 0,
    /// bf16 [layers, rows, width]: the last window's projection rows, which the conv's next window is cut from.
    proj: cuda.Buffer,
    /// A recurrent window's first `saved` rows of every layer, for a replay: k and g fp32 [layers, saved, heads,
    /// 128], v bf16 the same, beta fp32 [layers, saved, heads]. A sequence's own state saves `recurrent_rows`, the
    /// most a window it does not commit whole has; a window of sequences (`initWindow`) all its rows.
    saved: usize,
    k: cuda.Buffer,
    v: cuda.Buffer,
    g: cuda.Buffer,
    beta: cuda.Buffer,
    /// bf16 [rows, heads 128], a layer at a time: the decay's rows.
    decay: cuda.Buffer,
    /// bf16 [rows, heads 128], a layer at a time: the output gate's rows, the chain's output.
    gate: cuda.Buffer,
    out: cuda.Buffer,
    /// The chunked chain's, with `rows` over `recurrent_rows`: a chunk and head's bf16 operands (`kda_chunk.cu`),
    /// their fp32 decays, and the read-out bf16 [rows, heads, 128].
    scratch: ?cuda.Buffer = null,
    last: ?cuda.Buffer = null,
    y: ?cuda.Buffer = null,
    /// The chain of every window, or null: chunked by its rows (`chunks`), which is what the model runs. Set in
    /// the tests that hold the two against each other.
    chunked: ?bool = null,
    /// The last window `forward` ran: where it started, its rows and whether it ran chunked.
    window: Window = .{},
    /// The marks the next chunked window `forward` runs writes the state of (`ask`), rising: the window's rows
    /// before each and its `Mark.rec`. `commit` empties them.
    asked_rows: [most_asked]u32 = undefined,
    asked_rec: [most_asked][*]f32 = undefined,
    asked_n: usize = 0,

    pub const Window = struct { pos: usize = 0, rows: usize = 0, chunked: bool = false };

    pub fn init(layers: usize, heads: usize, width: usize, rows: usize) cuda.Error!State {
        return make(layers, heads, width, rows, true);
    }

    /// A window alone, for sequences that are each a `Kept`: `project`, `chainKept` and `emit` run it and
    /// `Kept.commit` reads its saved rows. It keeps no sequence of its own.
    pub fn initWindow(layers: usize, heads: usize, width: usize, rows: usize) cuda.Error!State {
        return make(layers, heads, width, rows, false);
    }

    fn make(layers: usize, heads: usize, width: usize, rows: usize, keeps: bool) cuda.Error!State {
        std.debug.assert(layers >= 1 and heads >= 1 and rows >= 1 and width >= 3 * heads * head_dim + 2 * head_dim + heads);
        const inner = heads * head_dim;
        const saved = if (keeps and chunked_windows) @min(rows, recurrent_rows) else rows;
        var self: State = .{ .layers = layers, .heads = heads, .rows = rows, .width = width, .saved = saved, .conv = undefined, .rec = undefined, .proj = undefined, .k = undefined, .v = undefined, .g = undefined, .beta = undefined, .decay = undefined, .gate = undefined, .out = undefined };
        self.conv = try .init(.bf16, if (keeps) &.{ layers, window, 3 * inner } else &.{ 1, 1, 1 });
        errdefer self.conv.deinit();
        self.rec = try .init(.f32, if (keeps) &.{ 2, layers, heads, head_dim, head_dim } else &.{ 1, 1, 1, 1, 1 });
        errdefer self.rec.deinit();
        self.proj = try .init(.bf16, &.{ layers, rows, width });
        errdefer self.proj.deinit();
        self.k = try .init(.f32, &.{ layers, saved, heads, head_dim });
        errdefer self.k.deinit();
        self.v = try .init(.bf16, &.{ layers, saved, heads, head_dim });
        errdefer self.v.deinit();
        self.g = try .init(.f32, &.{ layers, saved, heads, head_dim });
        errdefer self.g.deinit();
        self.beta = try .init(.f32, &.{ layers, saved, heads });
        errdefer self.beta.deinit();
        self.decay = try .init(.bf16, &.{ rows, inner });
        errdefer self.decay.deinit();
        self.gate = try .init(.bf16, &.{ rows, inner });
        errdefer self.gate.deinit();
        self.out = try .init(.bf16, &.{ rows, inner });
        errdefer self.out.deinit();
        if (chunks(rows)) {
            const many = chunksFor(rows) * heads;
            self.scratch = try .init(.bf16, &.{ many, @intCast(zs_glm5_kda_chunk_elems()) });
            errdefer self.scratch.?.deinit();
            self.last = try .init(.f32, &.{ many, head_dim });
            errdefer self.last.?.deinit();
            self.y = try .init(.bf16, &.{ rows, heads, head_dim });
        }
        errdefer inline for (.{ "scratch", "last", "y" }) |name| if (@field(self, name)) |*one| one.deinit();
        try self.reset();
        return self;
    }

    pub fn deinit(self: *State) void {
        inline for (.{ "conv", "rec", "proj", "k", "v", "g", "beta", "decay", "gate", "out" }) |name| @field(self, name).deinit();
        inline for (.{ "scratch", "last", "y" }) |name| if (@field(self, name)) |*one| one.deinit();
        self.* = undefined;
    }

    /// An empty sequence: no state, a conv window of zeros.
    pub fn reset(self: *State) cuda.Error!void {
        try self.conv.zero();
        try self.rec.zero();
        self.cur = 0;
        self.asked_n = 0;
    }

    /// The next window `forward` runs, chunked (`chunkedFor`), writes into `mark`'s buffers the state after its
    /// first `rows` rows, a chunk's start inside it (`session/marks.zig` snaps a mark there; spec 116): the bits
    /// the window carries past it, so `Mark.convWithin` alone completes the mark once it has run. Asked rising.
    pub fn ask(self: *State, rows: usize, mark: Mark) void {
        std.debug.assert(mark.layers == self.layers and mark.heads == self.heads and rows >= 1);
        std.debug.assert(self.asked_n < most_asked and (self.asked_n == 0 or self.asked_rows[self.asked_n - 1] < rows));
        self.asked_rows[self.asked_n] = @intCast(rows);
        self.asked_rec[self.asked_n] = @ptrCast(@alignCast(mark.rec.view.ptr.?));
        self.asked_n += 1;
    }

    /// Whether the window that has run wrote the state after its first `rows` rows into `mark` (`ask`).
    pub fn wrote(self: State, rows: usize, mark: Mark) bool {
        if (!self.window.chunked) return false;
        for (self.asked_rows[0..self.asked_n], self.asked_rec[0..self.asked_n]) |r, rec| {
            if (r == rows and @intFromPtr(rec) == @intFromPtr(mark.rec.view.ptr.?)) return true;
        }
        return false;
    }

    /// `count` elements of `shape` from element `first` of a buffer.
    fn at(buffer: cuda.Buffer, first: usize, shape: []const usize) cuda.View {
        const base: [*]u8 = @ptrCast(buffer.view.ptr.?);
        return .init(@ptrCast(base + first * buffer.view.dtype.size()), buffer.view.dtype, shape);
    }

    /// Layer `index`'s state on side `side` of `rec`.
    fn recOf(self: State, side: u1, index: usize) cuda.View {
        const one = self.heads * head_dim * head_dim;
        return at(self.rec, (@as(usize, side) * self.layers + index) * one, &.{ self.heads, head_dim, head_dim });
    }

    /// The window x bf16 [rows, dims] at the sequence's position `pos` through KDA layer `index` of the sequence
    /// (`weights` that layer's): out fp32 [rows, dims], the rank's partial of the block. The committed state is
    /// read and not changed.
    pub fn forward(self: *State, weights: Weights, index: usize, pos: usize, x: cuda.View, out: cuda.View) cuda.Error!void {
        try self.forwardChain(weights, index, pos, x);
        try self.emit(weights, @intCast(x.size[0]), out);
    }

    /// `forward` but its last step (`emit`, or `emitRows` a part of the rows at a time, spec 116 WS-03).
    pub fn forwardChain(self: *State, weights: Weights, index: usize, pos: usize, x: cuda.View) cuda.Error!void {
        const rows: usize = @intCast(x.size[0]);
        const inner = self.heads * head_dim;
        self.window = .{ .pos = pos, .rows = rows, .chunked = self.chunkedFor(rows) };
        for (self.asked_rows[0..self.asked_n]) |one| std.debug.assert(self.window.chunked and one < rows and (pos + one) % chunk == 0);
        try self.project(weights, index, x);
        try self.chainAsked(weights, index, 0, rows, pos, at(self.conv, index * window * 3 * inner, &.{ window, 3 * inner }), self.recOf(self.cur, index), self.recOf(~self.cur, index), self.asked_n);
    }

    /// Whether a window of `rows` rows runs chunked.
    pub fn chunkedFor(self: State, rows: usize) bool {
        return self.chunked orelse chunks(rows);
    }

    /// `forward`'s first step over a window's rows, whichever sequences they are of: layer `index`'s projection
    /// rows, the decay's and the gate's. A row's are its own.
    pub fn project(self: State, weights: Weights, index: usize, x: cuda.View) cuda.Error!void {
        const rows: usize = @intCast(x.size[0]);
        std.debug.assert(index < self.layers and rows >= 1 and rows <= self.rows);
        std.debug.assert(weights.heads == self.heads and weights.proj.n == self.width);
        const inner = self.heads * head_dim;
        const p = at(self.proj, index * self.rows * self.width, &.{ rows, self.width });
        try weights.proj.forward(x, p);
        // the decay's and the gate's first halves are columns of the projection's rows
        var f_a = at(self.proj, index * self.rows * self.width + 3 * inner, &.{ rows, head_dim });
        f_a.row_stride = @intCast(self.width);
        var g_a = at(self.proj, index * self.rows * self.width + 3 * inner + head_dim, &.{ rows, head_dim });
        g_a.row_stride = @intCast(self.width);
        try weights.f_b.forward(f_a, self.decay.span(0, rows));
        try weights.g_b.forward(g_a, self.gate.span(0, rows));
    }

    /// The chain of the projected rows `first` .. `first + rows - 1`, one sequence's at its position `pos`: from
    /// its conv rows `conv` (bf16 [3, 3 heads 128]) and its state `from`, the state after the last row into `to`.
    /// A recurrent chain saves its rows for a replay where the saves hold them.
    pub fn chain(self: State, weights: Weights, index: usize, first: usize, rows: usize, pos: usize, conv: cuda.View, from: cuda.View, to: cuda.View) cuda.Error!void {
        return self.chainAsked(weights, index, first, rows, pos, conv, from, to, 0);
    }

    /// `chain`, a chunked one writing the first `asked` states asked (`ask`) on its way.
    fn chainAsked(self: State, weights: Weights, index: usize, first: usize, rows: usize, pos: usize, conv: cuda.View, from: cuda.View, to: cuda.View, asked: usize) cuda.Error!void {
        std.debug.assert(rows >= 1 and first + rows <= self.rows);
        const inner = self.heads * head_dim;
        const p = at(self.proj, (index * self.rows + first) * self.width, &.{ rows, self.width });
        const decay = self.decay.span(first, rows);
        const gate = self.gate.span(first, rows);
        const chained = self.out.span(first, rows);
        const b_off: i64 = @intCast(3 * inner + 2 * head_dim);
        if (self.chunkedFor(rows)) {
            std.debug.assert(self.scratch != null and rows <= self.rows);
            // a mark's layer `index`, as `Mark.committed`
            const one = self.heads * head_dim * head_dim;
            var rows_at: [most_asked]i64 = undefined;
            var into: [most_asked]cuda.View = undefined;
            for (self.asked_rows[0..asked], self.asked_rec[0..asked], rows_at[0..asked], into[0..asked]) |r, rec, *r_at, *v| {
                r_at.* = r;
                v.* = .init(@ptrCast(rec + index * one), .f32, &.{ self.heads, head_dim, head_dim });
            }
            return cuda.launched(zs_glm5_kda_chunked(&p, b_off, &decay, &gate, &conv, &weights.conv.view, &from, &weights.a_log.view, &weights.dt_bias.view, &weights.norm.view, weights.eps, weights.lower, @intCast(rows), @intCast(pos % chunk), &chained, &to, &self.scratch.?.view, &self.last.?.view, &self.y.?.view, @intCast(asked), &rows_at, &into));
        }
        std.debug.assert(asked == 0);
        const saved = (index * self.saved + first) * inner;
        const saves = first + rows <= self.saved;
        const k = at(self.k, if (saves) saved else 0, &.{ rows, self.heads, head_dim });
        const v = at(self.v, if (saves) saved else 0, &.{ rows, self.heads, head_dim });
        const g = at(self.g, if (saves) saved else 0, &.{ rows, self.heads, head_dim });
        const beta = at(self.beta, if (saves) (index * self.saved + first) * self.heads else 0, &.{ rows, self.heads });
        try cuda.launched(zs_glm5_kda_chain(&p, b_off, &decay, &gate, &conv, &weights.conv.view, &from, &weights.a_log.view, &weights.dt_bias.view, &weights.norm.view, weights.eps, weights.lower, @intCast(rows), &chained, &to, if (saves) &k else null, &v, &g, &beta));
    }

    /// `chain` for the sequence `kept` at its position `pos`: its committed state is read, the one beside it
    /// written.
    pub fn chainKept(self: State, weights: Weights, index: usize, kept: Kept, first: usize, rows: usize, pos: usize) cuda.Error!void {
        std.debug.assert(kept.layers == self.layers and kept.heads == self.heads);
        const inner = self.heads * head_dim;
        return self.chain(weights, index, first, rows, pos, at(kept.conv, index * window * 3 * inner, &.{ window, 3 * inner }), kept.recOf(kept.cur, index), kept.recOf(~kept.cur, index));
    }

    /// `forward`'s last step: the chained rows 0 .. `rows` - 1 into out fp32 [rows, dims].
    pub fn emit(self: State, weights: Weights, rows: usize, out: cuda.View) cuda.Error!void {
        std.debug.assert(out.dtype == .f32 or out.dtype == .bf16); // bf16: spec 116 WS-04
        return weights.o.forward(self.out.span(0, rows), out);
    }

    /// `emit` of the chained rows [first, first + count) alone, into those rows of out fp32 [rows, dims]: a
    /// row's bits are the whole's.
    pub fn emitRows(self: State, weights: Weights, first: usize, count: usize, out: cuda.View) cuda.Error!void {
        std.debug.assert(out.dtype == .f32 or out.dtype == .bf16); // bf16: spec 116 WS-04
        return weights.o.forward(self.out.span(first, count), out.span(first, count));
    }

    /// Keeps the first `keep` rows of the last window of `rows` rows, of every layer: the state after them
    /// becomes the committed one and the conv's window moves by `keep`. A window kept in part ran recurrent
    /// and saved its rows.
    pub fn commit(self: *State, rows: usize, keep: usize) cuda.Error!void {
        std.debug.assert(keep >= 1 and keep <= rows and rows <= self.rows);
        const inner = self.heads * head_dim;
        self.asked_n = 0;
        if (keep < rows) {
            std.debug.assert(!self.chunkedFor(rows) and rows <= self.saved);
            const one = self.heads * head_dim * head_dim;
            const from = at(self.rec, @as(usize, self.cur) * self.layers * one, &.{ self.layers, self.heads, head_dim, head_dim });
            const to = at(self.rec, @as(usize, ~self.cur) * self.layers * one, &.{ self.layers, self.heads, head_dim, head_dim });
            try cuda.launched(zs_glm5_kda_replay_layers(&from, @intCast(one), &self.k.view, &self.v.view, &self.g.view, &self.beta.view, @intCast(self.saved * inner), @intCast(self.saved * self.heads), @intCast(self.layers), @intCast(self.heads), @intCast(keep), &to));
        }
        self.cur = ~self.cur;
        try cuda.launched(zs_glm5_kda_conv_shift(&self.conv.view, &self.proj.view, @intCast(keep), @intCast(self.rows * self.width), @intCast(self.width)));
    }

    /// Layer `index`'s committed state, fp32 [heads, 128, 128].
    pub fn committed(self: State, index: usize) cuda.View {
        return self.recOf(self.cur, index);
    }
};

/// What `layers` KDA layers keep of a sequence and nothing of a window: the sequences that share a window
/// (`State.initWindow`) are each one. The same bits as a `State`'s own.
pub const Kept = struct {
    layers: usize,
    heads: usize,
    /// bf16 [layers, 3, 3 heads 128] and fp32 [2, layers, heads, 128, 128], as `State`'s.
    conv: cuda.Buffer,
    rec: cuda.Buffer,
    cur: u1 = 0,

    pub fn init(layers: usize, heads: usize) cuda.Error!Kept {
        var conv: cuda.Buffer = try .init(.bf16, &.{ layers, window, 3 * heads * head_dim });
        errdefer conv.deinit();
        var rec: cuda.Buffer = try .init(.f32, &.{ 2, layers, heads, head_dim, head_dim });
        errdefer rec.deinit();
        var self: Kept = .{ .layers = layers, .heads = heads, .conv = conv, .rec = rec };
        try self.reset();
        return self;
    }

    pub fn deinit(self: *Kept) void {
        self.conv.deinit();
        self.rec.deinit();
        self.* = undefined;
    }

    /// The device bytes of one.
    pub fn bytes(layers: usize, heads: usize) usize {
        return layers * heads * head_dim * (window * 3 * 2 + 2 * head_dim * 4);
    }

    /// An empty sequence.
    pub fn reset(self: *Kept) cuda.Error!void {
        try self.conv.zero();
        try self.rec.zero();
        self.cur = 0;
    }

    fn recOf(self: Kept, side: u1, index: usize) cuda.View {
        const one = self.heads * head_dim * head_dim;
        return State.at(self.rec, (@as(usize, side) * self.layers + index) * one, &.{ self.heads, head_dim, head_dim });
    }

    /// Layer `index`'s committed state, fp32 [heads, 128, 128].
    pub fn committed(self: Kept, index: usize) cuda.View {
        return self.recOf(self.cur, index);
    }

    /// `State.commit` for this sequence's rows `first` .. `first + rows - 1` of the window `in` ran last.
    pub fn commit(self: *Kept, in: State, first: usize, rows: usize, keep: usize) cuda.Error!void {
        std.debug.assert(keep >= 1 and keep <= rows and first + rows <= in.rows and in.layers == self.layers and in.heads == self.heads);
        const inner = self.heads * head_dim;
        const last = self.layers - 1;
        if (keep < rows) {
            std.debug.assert(!in.chunkedFor(rows) and first + rows <= in.saved);
            const one = self.heads * head_dim * head_dim;
            const from = State.at(self.rec, @as(usize, self.cur) * self.layers * one, &.{ self.layers, self.heads, head_dim, head_dim });
            const to = State.at(self.rec, @as(usize, ~self.cur) * self.layers * one, &.{ self.layers, self.heads, head_dim, head_dim });
            // a layer's saved rows from this sequence's first, a layer a stride of the window's saved rows
            const saved = [_]usize{(last * in.saved + keep) * inner};
            const k = State.at(in.k, first * inner, &saved);
            const v = State.at(in.v, first * inner, &saved);
            const g = State.at(in.g, first * inner, &saved);
            const beta = State.at(in.beta, first * self.heads, &.{(last * in.saved + keep) * self.heads});
            try cuda.launched(zs_glm5_kda_replay_layers(&from, @intCast(one), &k, &v, &g, &beta, @intCast(in.saved * inner), @intCast(in.saved * self.heads), @intCast(self.layers), @intCast(self.heads), @intCast(keep), &to));
        }
        self.cur = ~self.cur;
        const proj = State.at(in.proj, first * in.width, &.{(last * in.rows + keep) * in.width});
        try cuda.launched(zs_glm5_kda_conv_shift(&self.conv.view, &proj, @intCast(keep), @intCast(in.rows * in.width), @intCast(in.width)));
    }
};

/// A sequence's committed state alone, to return to (`session/snapshot.zig`'s marks): the conv window and the
/// side of `rec` a `State` reads, as Flash Next's snapshot keeps its live recurrent state alone. The other
/// side is what a window's chain or a commit's replay writes whole before anything reads it: a mark taken
/// between a window and its commit leaves that window's end, and a restore is followed by a window. So one
/// side gives a `State`'s bits, in half the bytes of a `Kept`.
pub const Mark = struct {
    layers: usize,
    heads: usize,
    /// bf16 [layers, 3, 3 heads 128] and fp32 [1, layers, heads, 128, 128]: `State`'s conv, one side of its `rec`.
    conv: cuda.Buffer,
    rec: cuda.Buffer,

    pub fn init(layers: usize, heads: usize) cuda.Error!Mark {
        var conv: cuda.Buffer = try .init(.bf16, &.{ layers, window, 3 * heads * head_dim });
        errdefer conv.deinit();
        const rec: cuda.Buffer = try .init(.f32, &.{ 1, layers, heads, head_dim, head_dim });
        return .{ .layers = layers, .heads = heads, .conv = conv, .rec = rec };
    }

    pub fn deinit(self: *Mark) void {
        self.conv.deinit();
        self.rec.deinit();
        self.* = undefined;
    }

    /// The device bytes of one.
    pub fn bytes(layers: usize, heads: usize) usize {
        return layers * heads * head_dim * (window * 3 * 2 + head_dim * 4);
    }

    /// The side of `state`'s `rec` that its next window reads.
    fn live(state: State) cuda.Buffer {
        std.debug.assert(state.rec.view.size[0] == 2);
        return .{ .view = state.rec.span(state.cur, 1) };
    }

    /// `state`'s committed state and conv window into this one's buffers.
    pub fn take(self: Mark, state: State) cuda.Error!void {
        std.debug.assert(state.layers == self.layers and state.heads == self.heads);
        try self.conv.copyFrom(state.conv);
        try self.rec.copyFrom(live(state));
    }

    /// The state after the first `rows` rows of `state`'s last window, which has run and is not committed,
    /// into this one's buffers, as `State.commit(.., rows)` would leave it: the saved rows replayed from the
    /// committed side, the conv window copied and moved by `rows`. Nothing of `state` is written, so the
    /// window's own commit is the same (spec 112 WS-06: a mark inside a prompt's window, the window not cut).
    /// A chunked window's states were written as it ran (`State.ask`); `convWithin` completes the mark.
    pub fn takeWithin(self: Mark, state: State, rows: usize) cuda.Error!void {
        std.debug.assert(state.layers == self.layers and state.heads == self.heads and rows >= 1 and rows <= state.saved and !state.window.chunked);
        const inner = state.heads * head_dim;
        const one = state.heads * head_dim * head_dim;
        const from = State.at(state.rec, @as(usize, state.cur) * state.layers * one, &.{ state.layers, state.heads, head_dim, head_dim });
        const to = State.at(self.rec, 0, &.{ self.layers, self.heads, head_dim, head_dim });
        try cuda.launched(zs_glm5_kda_replay_layers(&from, @intCast(one), &state.k.view, &state.v.view, &state.g.view, &state.beta.view, @intCast(state.saved * inner), @intCast(state.saved * state.heads), @intCast(state.layers), @intCast(state.heads), @intCast(rows), &to));
        try self.convWithin(state, rows);
    }

    /// The conv window after the first `rows` rows of `state`'s last window.
    pub fn convWithin(self: Mark, state: State, rows: usize) cuda.Error!void {
        try self.conv.copyFrom(state.conv);
        try cuda.launched(zs_glm5_kda_conv_shift(&self.conv.view, &state.proj.view, @intCast(rows), @intCast(state.rows * state.width), @intCast(state.width)));
    }

    /// `state` is this one again, in the side it reads: its `cur`, which a captured graph is bound to, kept.
    pub fn restore(self: Mark, state: State) cuda.Error!void {
        std.debug.assert(state.layers == self.layers and state.heads == self.heads);
        try state.conv.copyFrom(self.conv);
        try live(state).copyFrom(self.rec);
    }

    /// Layer `index`'s committed state, fp32 [heads, 128, 128].
    pub fn committed(self: Mark, index: usize) cuda.View {
        const one = self.heads * head_dim * head_dim;
        return State.at(self.rec, index * one, &.{ self.heads, head_dim, head_dim });
    }
};

/// The recurrent chain and the replay load and store the state four floats at a time
/// (`kda_quads.cu`, `ZSPARK_GLM_KDA_QUADS`, spec 073 WS-05): the same bits, off unless `main` reads the switch.
/// Said before a step's graph is captured.
pub fn setQuads(on: bool) void {
    zs_glm5_kda_quads(@intFromBool(on));
}

extern fn zs_glm5_kda_quads(on: c_int) void;
extern fn zs_glm5_kda_chain(p: *const cuda.View, b_off: i64, a: *const cuda.View, g: *const cuda.View, cs: *const cuda.View, cw: *const cuda.View, state_in: *const cuda.View, a_log: *const cuda.View, dt_bias: *const cuda.View, norm_w: *const cuda.View, eps: f64, lower: f64, rows: i64, out: *const cuda.View, state_out: ?*const cuda.View, k_save: ?*const cuda.View, v_save: ?*const cuda.View, g_save: ?*const cuda.View, b_save: ?*const cuda.View) c_int;
extern fn zs_glm5_kda_chunked(p: *const cuda.View, b_off: i64, a: *const cuda.View, g: ?*const cuda.View, cs: *const cuda.View, cw: *const cuda.View, state_in: *const cuda.View, a_log: *const cuda.View, dt_bias: *const cuda.View, norm_w: ?*const cuda.View, eps: f64, lower: f64, rows: i64, phase: i64, out: ?*const cuda.View, state_out: *const cuda.View, scratch: *const cuda.View, last: *const cuda.View, y_tmp: ?*const cuda.View, asked_n: i64, asked_rows: [*]const i64, asked_to: [*]const cuda.View) c_int;
extern fn zs_glm5_kda_chunk_elems() i64;
extern fn zs_glm5_kda_replay_layers(state_in: *const cuda.View, state_stride: i64, k_save: *const cuda.View, v_save: *const cuda.View, g_save: *const cuda.View, b_save: *const cuda.View, kv_stride: i64, b_stride: i64, layers: i64, heads: i64, rows: i64, state_out: *const cuda.View) c_int;
extern fn zs_glm5_kda_conv_shift(conv: *const cuda.View, proj: *const cuda.View, keep: i64, proj_layer: i64, proj_row: i64) c_int;
