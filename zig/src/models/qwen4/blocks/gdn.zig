//! Flash Next's Gated DeltaNet layer on the device: a chain of rows from a recurrent state, and the replay that
//! rebuilds the state after an accepted prefix of the chain with the same update routine, so its bits are the
//! serial steps'.

const cuda = @import("../../../cuda/cuda.zig");

pub const key_dim = 128;
pub const value_dim = 128;
pub const taps = 4;

/// A layer's head counts and the widths that follow from them.
pub const Shape = struct {
    key_heads: usize = 16,
    value_heads: usize = 48,

    /// Conv channels: q | k | v.
    pub fn conv(self: Shape) usize {
        return 2 * self.key_heads * key_dim + self.value_heads * value_dim;
    }

    /// A projection row: qkv | z | b | a.
    pub fn projection(self: Shape) usize {
        return self.conv() + self.value_heads * value_dim + 2 * self.value_heads;
    }

    pub fn out(self: Shape) usize {
        return self.value_heads * value_dim;
    }

    pub fn state(self: Shape) usize {
        return self.value_heads * value_dim * key_dim;
    }
};

/// What replaying a sequence's last window of up to `rows` rows needs.
pub const Scratch = struct {
    /// fp32 [rows, key heads, 128]
    k: cuda.Buffer,
    /// bf16 [rows, value heads, 128]
    v: cuda.Buffer,
    /// fp32 [rows, value heads]
    g: cuda.Buffer,
    /// fp32 [rows, value heads]
    beta: cuda.Buffer,

    pub fn init(rows: usize, shape: Shape) cuda.Error!Scratch {
        var k: cuda.Buffer = try .init(.f32, &.{ rows, shape.key_heads, key_dim });
        errdefer k.deinit();
        var v: cuda.Buffer = try .init(.bf16, &.{ rows, shape.value_heads, value_dim });
        errdefer v.deinit();
        var g: cuda.Buffer = try .init(.f32, &.{ rows, shape.value_heads });
        errdefer g.deinit();
        const beta: cuda.Buffer = try .init(.f32, &.{ rows, shape.value_heads });
        return .{ .k = k, .v = v, .g = g, .beta = beta };
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(rows: usize, shape: Shape) usize {
        const of = cuda.Buffer.bytesOf;
        return of(.f32, &.{ rows, shape.key_heads, key_dim }) + of(.bf16, &.{ rows, shape.value_heads, value_dim }) + 2 * of(.f32, &.{ rows, shape.value_heads });
    }

    pub fn deinit(self: *Scratch) void {
        self.k.deinit();
        self.v.deinit();
        self.g.deinit();
        self.beta.deinit();
        self.* = undefined;
    }
};

/// A layer's weights beside its projections.
pub const Weights = struct {
    /// bf16 [conv channels, 4]
    conv: cuda.View,
    /// fp32 [value heads]
    a_log: cuda.View,
    /// fp32 [value heads]
    dt_bias: cuda.View,
    /// bf16 [128]
    norm: cuda.View,
};

pub const Chain = struct {
    /// bf16 [rows, projection width]
    projection: cuda.View,
    /// bf16 [3, conv channels]: the conv inputs of the three rows before the chain
    conv_state: cuda.View,
    /// fp32 [value heads, 128, 128]
    state_in: cuda.View,
    /// The state after the last row, or null to leave it.
    state_out: ?cuda.View,
    /// bf16 [rows, value heads * 128]
    out: cuda.View,
    /// fp32 [rows, value heads * 4]: out's 32-group sums, which the kernel always writes
    sums: cuda.View,
    scratch: ?*const Scratch,
    /// fp32 [rows, key heads, 128]: scratch that lets a window of `wide_rows` rows or more run as a front end,
    /// a chain of the delta rule alone and a back end (`zig/kernels/qwen4/gdn_wide.cu`), with `scratch` and
    /// `state_out` given too. The bits are the same either way.
    work: ?cuda.View = null,
    /// i32 [1] on the device: the rows at the front of `scratch` that `state_in` does not hold yet (a window
    /// kept in part, its replay left to this chain, spec 089). The kernel replays them on the state it loaded
    /// and saves this chain's rows after them, so `scratch` holds that many rows and `rows` more; null is
    /// none. A window through the wide kernels has none.
    behind: ?cuda.View = null,
};

/// The rows from which a window runs through the wide kernels: below it the one launch of the chain kernel is
/// quicker than their three.
pub var wide_rows: usize = 32;

/// The chain and the replay below `wide_rows`, the pool's slots' too (spec 096), load and store the state in rows of four floats and not a float
/// at a time, and the chain and replay sum a row's products in butterflies (spec 143 WS-05), unless
/// `ZSPARK_GDN_QUADS` is 0 (spec 073): the same bits, the base kept for a comparison.
/// Said before a step's graph is captured.
pub fn setQuads(on: bool) void {
    zs_qwen4_gdn_quads(@intFromBool(on));
}

pub fn chain(weights: Weights, call: Chain, rows: usize, eps: f32) cuda.Error!void {
    const s = call.scratch;
    if (rows >= wide_rows and call.work != null and s != null and call.state_out != null) {
        return cuda.launched(zs_qwen4_gdn_chain_wide(&call.projection, &call.conv_state, &weights.conv, &call.state_in, &weights.a_log, &weights.dt_bias, &weights.norm, eps, @intCast(rows), &call.out, &call.sums, &call.state_out.?, &call.work.?, &s.?.k.view, &s.?.v.view, &s.?.g.view, &s.?.beta.view));
    }
    return cuda.launched(zs_qwen4_gdn_chain(&call.projection, &call.conv_state, &weights.conv, &call.state_in, &weights.a_log, &weights.dt_bias, &weights.norm, eps, @intCast(rows), &call.out, &call.sums, if (call.state_out) |*view| view else null, if (s) |x| &x.k.view else null, if (s) |x| &x.v.view else null, if (s) |x| &x.g.view else null, if (s) |x| &x.beta.view else null, if (call.behind) |*view| view else null));
}

/// The state after the first `rows` rows of the chain that filled `scratch`, from the state that chain started at.
pub fn replay(state_in: cuda.View, scratch: Scratch, rows: usize, state_out: cuda.View) cuda.Error!void {
    return replayThrough(if (rows >= wide_rows) .wide else .narrow, state_in, scratch, rows, state_out);
}

/// `replay` through the chain kernel's own layout or the wide chain's: the same bits.
pub fn replayThrough(kernel: enum { narrow, wide }, state_in: cuda.View, scratch: Scratch, rows: usize, state_out: cuda.View) cuda.Error!void {
    if (kernel == .wide) return cuda.launched(zs_qwen4_gdn_replay_wide(&state_in, &scratch.k.view, &scratch.v.view, &scratch.g.view, &scratch.beta.view, @intCast(rows), &state_out));
    return cuda.launched(zs_qwen4_gdn_replay(&state_in, &scratch.k.view, &scratch.v.view, &scratch.g.view, &scratch.beta.view, @intCast(rows), &state_out));
}

/// One DeltaNet state a landing brings to its position (`land`): `state_out` becomes the state after the first
/// `rows` rows of the chain that filled `k`, `v`, `g` and `beta` (a `Scratch`'s), from `state_in`, `replay`'s bits.
pub const Landing = extern struct {
    state_in: cuda.View,
    k: cuda.View,
    v: cuda.View,
    g: cuda.View,
    beta: cuda.View,
    state_out: cuda.View,
    rows: i64,

    pub fn of(state_in: cuda.View, scratch: Scratch, rows: usize, state_out: cuda.View) Landing {
        return .{ .state_in = state_in, .k = scratch.k.view, .v = scratch.v.view, .g = scratch.g.view, .beta = scratch.beta.view, .state_out = state_out, .rows = @intCast(rows) };
    }
};

/// `replay` of each of `entries`, which share no state: one launch for up to 64 of them, the heads of every entry
/// in one grid (spec 146 WS-07: a landing was a launch a layer, 36 in series), the same bits.
pub fn land(entries: []const Landing) cuda.Error!void {
    if (entries.len == 0) return;
    return cuda.launched(zs_qwen4_gdn_land(entries.ptr, @intCast(entries.len)));
}

/// `replay` gone on from a state already replayed: `state_in` is the state after the first `first` rows of the
/// chain that filled `scratch`, and `state_out` the state `rows` rows later. A replay is the chain's update a
/// row at a time, each row from the fp32 state the row before left, so the bits are those of one replay of
/// `first + rows` rows; what it saves is the rows below `first`, which is all but a few when two states are
/// wanted near the end of a wide window (spec 041).
pub fn replayFrom(state_in: cuda.View, scratch: Scratch, first: usize, rows: usize, state_out: cuda.View) cuda.Error!void {
    return replayFromThrough(if (rows >= wide_rows) .wide else .narrow, state_in, scratch, first, rows, state_out);
}

pub fn replayFromThrough(kernel: enum { narrow, wide }, state_in: cuda.View, scratch: Scratch, first: usize, rows: usize, state_out: cuda.View) cuda.Error!void {
    const k = scratch.k.span(first, rows);
    const v = scratch.v.span(first, rows);
    const g = scratch.g.span(first, rows);
    const beta = scratch.beta.span(first, rows);
    if (kernel == .wide) return cuda.launched(zs_qwen4_gdn_replay_wide(&state_in, &k, &v, &g, &beta, @intCast(rows), &state_out));
    return cuda.launched(zs_qwen4_gdn_replay(&state_in, &k, &v, &g, &beta, @intCast(rows), &state_out));
}

extern fn zs_qwen4_gdn_quads(on: c_int) void;
extern fn zs_qwen4_gdn_land(entries: [*]const Landing, count: i64) c_int;
extern fn zs_qwen4_gdn_chain(p: *const cuda.View, cs: *const cuda.View, cw: *const cuda.View, state_in: *const cuda.View, a_log: *const cuda.View, dt_bias: *const cuda.View, norm_w: *const cuda.View, eps: f64, rows: i64, out: *const cuda.View, xs: *const cuda.View, state_out: ?*const cuda.View, k_save: ?*const cuda.View, v_save: ?*const cuda.View, g_save: ?*const cuda.View, b_save: ?*const cuda.View, behind: ?*const cuda.View) c_int;
extern fn zs_qwen4_gdn_chain_wide(p: *const cuda.View, cs: *const cuda.View, cw: *const cuda.View, state_in: *const cuda.View, a_log: *const cuda.View, dt_bias: *const cuda.View, norm_w: *const cuda.View, eps: f64, rows: i64, out: *const cuda.View, xs: *const cuda.View, state_out: *const cuda.View, q: *const cuda.View, k: *const cuda.View, v: *const cuda.View, g: *const cuda.View, b: *const cuda.View) c_int;
extern fn zs_qwen4_gdn_replay_wide(state_in: *const cuda.View, k_save: *const cuda.View, v_save: *const cuda.View, g_save: *const cuda.View, b_save: *const cuda.View, rows: i64, state_out: *const cuda.View) c_int;
extern fn zs_qwen4_gdn_replay(state_in: *const cuda.View, k_save: *const cuda.View, v_save: *const cuda.View, g_save: *const cuda.View, b_save: *const cuda.View, rows: i64, state_out: *const cuda.View) c_int;
