//! The NemotronH Mamba-2 mixer between its two projections (`zig/kernels/nemotron/mamba.cu`, `mamba_wide.cu`):
//! in_proj's rows bf16 [rows, proj] = [z | x, B, C (the conv channels) | dt] -> the depthwise conv of 4 taps
//! and SiLU, the selective scan over an fp32 state [heads, head_dim, 128], y * silu(z), and the group RMSNorm
//! (`glue.groupRmsnorm`) -> bf16 [rows, xd] for out_proj. The projections are the caller's.
//!
//! TensorFold's state model (`families/nemotron/cuda_engine.zig`): the state lags a window. A window of up to
//! `window_rows` rows first replays the previous window's kept rows from the parity buffers (`meta`'s
//! prev_keep rows of buffer 1 - parity), committing the state through them, then runs its own rows into buffer
//! `parity`; rejected rows never touch the committed state, so a rollback costs nothing. `window` runs
//! TensorFold's step kernels; `prompt` runs a chunk of any rows (`mamba_wide.cu`) with the same bits, leaving
//! the state its rows' serial steps would. `Meta` is the device word row both read and the caller owns.

const std = @import("std");
const cuda = @import("../../cuda/cuda.zig");
const glue = @import("glue.zig");
const config = @import("config.zig");

/// The rows a window holds in the parity buffers (TensorFold's `state.max_rows`): a verify window, a step.
pub const window_rows = 16;

/// The mixer's dimensions; the defaults are Nemotron-3.5-Transcribe's.
pub const Shape = struct {
    xd: usize = 4096,
    heads: usize = 64,
    head_dim: usize = 64,
    groups: usize = 8,
    state: usize = 128,

    pub fn of(b: config.Backbone) Shape {
        return .{ .xd = b.mambaInner(), .heads = b.mamba_heads, .head_dim = b.mamba_head_dim, .groups = b.groups, .state = b.ssm_state };
    }

    /// The conv's channels: x, then each group's B, then each group's C.
    pub fn cd(s: Shape) usize {
        return s.xd + 2 * s.groups * s.state;
    }

    /// in_proj's outputs: z, the conv channels, a dt a head.
    pub fn proj(s: Shape) usize {
        return s.xd + s.cd() + s.heads;
    }
};

/// The window's meta row as the kernels read it (int32 [4]): {pos, parity, prev_keep, spare}.
pub const Meta = struct {
    pos: u32 = 0,
    parity: u32 = 0,
    prev_keep: u32 = 0,

    /// After a window whose first `keep` rows are kept: TensorFold's verify (parity flips) then commit(keep).
    pub fn afterWindow(m: Meta, keep: usize) Meta {
        return .{ .pos = m.pos + @as(u32, @intCast(keep)), .parity = m.parity ^ 1, .prev_keep = @intCast(keep) };
    }

    /// After a prompt chunk of `rows` (kept whole): its last row waits in the parity buffers as a step's would.
    pub fn afterPrompt(m: Meta, rows: usize) Meta {
        return .{ .pos = m.pos + @as(u32, @intCast(rows)), .parity = m.parity ^ 1, .prev_keep = 1 };
    }

    pub fn words(m: Meta) [4]i32 {
        return .{ @intCast(m.pos), @intCast(m.parity), @intCast(m.prev_keep), 0 };
    }

    /// Writes the row into `meta` (int32 [4]), as a launch: a graph's capture records it, a memcpy would
    /// fail it (`uploadLaunched`).
    pub fn set(m: Meta, meta: cuda.Buffer) cuda.Error!void {
        const w = m.words();
        return meta.uploadLaunched(0, std.mem.sliceAsBytes(&w));
    }
};

/// A Mamba layer's state, one allocation: what a sequence carries from one window to the next.
pub const State = struct {
    /// base bf16 [3, cd]: the last three committed conv inputs; raw, xc bf16 [2, window_rows, cd]: a window's
    /// inputs and conv outputs by parity; dt fp32 [2, window_rows, heads]; ssm fp32 [heads, head_dim, state].
    parts: [5]cuda.Buffer,
    block: cuda.Buffer,

    pub fn init(s: Shape) cuda.Error!State {
        var st: State = .{ .parts = .{
            .shaped(.bf16, &.{ 3, s.cd() }),
            .shaped(.bf16, &.{ 2, window_rows, s.cd() }),
            .shaped(.bf16, &.{ 2, window_rows, s.cd() }),
            .shaped(.f32, &.{ 2, window_rows, s.heads }),
            .shaped(.f32, &.{ s.heads, s.head_dim, s.state }),
        }, .block = undefined };
        st.block = try cuda.cut(&.{&st.parts});
        try st.block.zero();
        return st;
    }

    pub fn deinit(st: *State) void {
        st.block.deinit();
    }

    /// A new sequence: every part zero (meta's prev_keep 0 reads nothing of the parity buffers).
    pub fn zero(st: State) cuda.Error!void {
        return st.block.zero();
    }

    pub fn base(st: State) cuda.Buffer {
        return st.parts[0];
    }
    pub fn raw(st: State) cuda.Buffer {
        return st.parts[1];
    }
    pub fn xc(st: State) cuda.Buffer {
        return st.parts[2];
    }
    pub fn dt(st: State) cuda.Buffer {
        return st.parts[3];
    }
    pub fn ssm(st: State) cuda.Buffer {
        return st.parts[4];
    }
};

/// What a call lends, made once for the most rows a prompt chunk takes and shared by every Mamba layer: xw
/// bf16 [rows, cd] (a chunk's conv), y bf16 [rows, xd] (before the norm), xs fp32 [rows, xd / 64] (the norm's
/// group sums, which nothing reads).
pub const Scratch = struct {
    rows: usize,
    parts: [3]cuda.Buffer,
    block: cuda.Buffer,

    pub fn init(s: Shape, rows: usize) cuda.Error!Scratch {
        const r = @max(rows, window_rows);
        var sc: Scratch = .{ .rows = r, .parts = .{
            .shaped(.bf16, &.{ r, s.cd() }),
            .shaped(.bf16, &.{ r, s.xd }),
            .shaped(.f32, &.{ r, s.xd / 64 }),
        }, .block = undefined };
        sc.block = try cuda.cut(&.{&sc.parts});
        return sc;
    }

    pub fn deinit(sc: *Scratch) void {
        sc.block.deinit();
    }
};

/// A Mamba layer's own weights between its projections, widened to fp32 at load as TensorFold does.
pub const Mixer = struct {
    shape: Shape,
    /// a = -exp(A_log), d, dt_bias fp32 [heads]; conv_w fp32 [4, cd] (tap-major), conv_b fp32 [cd]; norm bf16 [xd].
    parts: [6]cuda.Buffer,
    block: cuda.Buffer,
    /// dt's clamp (`time_step_limit`, [0, inf) on Transcribe) and the group norm's epsilon.
    lo: f32,
    hi: f32,
    eps: f64,

    /// The checkpoint's bf16 tensors as stored: A_log, D, dt_bias [heads], conv1d.weight [cd, 1, 4],
    /// conv1d.bias [cd], norm.weight [xd].
    pub const Host = struct { a_log: []align(1) const u16, d: []align(1) const u16, dt_bias: []align(1) const u16, conv_w: []align(1) const u16, conv_b: []align(1) const u16, norm: []align(1) const u16 };

    pub fn init(gpa: std.mem.Allocator, s: Shape, host: Host, lo: f32, hi: f32, eps: f64) !Mixer {
        const cd = s.cd();
        if (host.a_log.len != s.heads or host.d.len != s.heads or host.dt_bias.len != s.heads or host.conv_w.len != 4 * cd or host.conv_b.len != cd or host.norm.len != s.xd) return error.UnexpectedTensor;
        var m: Mixer = .{ .shape = s, .parts = .{
            .shaped(.f32, &.{s.heads}),
            .shaped(.f32, &.{s.heads}),
            .shaped(.f32, &.{s.heads}),
            .shaped(.f32, &.{ 4, cd }),
            .shaped(.f32, &.{cd}),
            .shaped(.bf16, &.{s.xd}),
        }, .block = undefined, .lo = lo, .hi = hi, .eps = eps };
        m.block = try cuda.cut(&.{&m.parts});
        errdefer m.block.deinit();
        const wide = try gpa.alloc(f32, 4 * cd);
        defer gpa.free(wide);
        for (0..cd) |c| for (0..4) |t| {
            wide[t * cd + c] = widen(host.conv_w[c * 4 + t]);
        };
        try m.parts[3].upload(std.mem.sliceAsBytes(wide));
        for (host.conv_b, wide[0..cd]) |v, *w| w.* = widen(v);
        try m.parts[4].upload(std.mem.sliceAsBytes(wide[0..cd]));
        for (host.d, wide[0..s.heads]) |v, *w| w.* = widen(v);
        try m.parts[1].upload(std.mem.sliceAsBytes(wide[0..s.heads]));
        for (host.dt_bias, wide[0..s.heads]) |v, *w| w.* = widen(v);
        try m.parts[2].upload(std.mem.sliceAsBytes(wide[0..s.heads]));
        try m.parts[5].upload(std.mem.sliceAsBytes(host.norm));
        for (host.a_log, wide[0..s.heads]) |v, *w| w.* = widen(v);
        var alog = try cuda.Buffer.init(.f32, &.{s.heads});
        defer alog.deinit();
        try alog.upload(std.mem.sliceAsBytes(wide[0..s.heads]));
        try decay(alog.view, m.parts[0].view);
        try cuda.synchronize();
        return m;
    }

    pub fn deinit(m: *Mixer) void {
        m.block.deinit();
    }

    pub fn a(m: Mixer) cuda.View {
        return m.parts[0].view;
    }
    pub fn d(m: Mixer) cuda.View {
        return m.parts[1].view;
    }
    pub fn dtBias(m: Mixer) cuda.View {
        return m.parts[2].view;
    }
    pub fn convW(m: Mixer) cuda.View {
        return m.parts[3].view;
    }
    pub fn convB(m: Mixer) cuda.View {
        return m.parts[4].view;
    }
    pub fn norm(m: Mixer) cuda.View {
        return m.parts[5].view;
    }

    /// A window of 1 to `window_rows` rows (a step, a verify window): proj bf16 [rows, proj] -> out bf16
    /// [rows, xd]. TensorFold's step kernels; the caller then sets meta by `Meta.afterWindow(keep)`.
    pub fn window(m: Mixer, st: State, sc: Scratch, proj: cuda.View, meta: cuda.View, out: cuda.View) cuda.Error!void {
        const rows: usize = @intCast(proj.size[0]);
        try cuda.bound(rows >= 1 and rows <= window_rows, "mamba.window: {d} rows, 1 to {d}", .{ rows, window_rows });
        const y = sc.parts[1].span(0, rows);
        try conv(proj, st.base().view, st.raw().view, st.xc().view, m.convW(), m.convB(), meta, m.shape.xd);
        try scan(proj, st.xc().view, st.dt().view, st.ssm().view, m.a(), m.d(), m.dtBias(), meta, y, m.lo, m.hi, m.shape.groups);
        return glue.groupRmsnorm(y, m.norm(), out, sc.parts[2].span(0, rows), m.eps, m.shape.groups);
    }

    /// A prompt chunk of 1 to `sc.rows` rows, kept whole: as `window`, by the wide kernels; the caller then
    /// sets meta by `Meta.afterPrompt(rows)` and chunks a longer prompt so.
    pub fn prompt(m: Mixer, st: State, sc: Scratch, proj: cuda.View, meta: cuda.View, out: cuda.View) cuda.Error!void {
        const rows: usize = @intCast(proj.size[0]);
        try cuda.bound(rows >= 1 and rows <= sc.rows, "mamba.prompt: {d} rows, 1 to {d}", .{ rows, sc.rows });
        const xw = sc.parts[0].span(0, rows);
        const y = sc.parts[1].span(0, rows);
        try convWide(proj, st.base().view, st.raw().view, st.xc().view, xw, m.convW(), m.convB(), meta, m.shape.xd);
        try scanWide(proj, st.xc().view, xw, st.dt().view, st.ssm().view, m.a(), m.d(), m.dtBias(), meta, y, m.lo, m.hi, m.shape.groups);
        return glue.groupRmsnorm(y, m.norm(), out, sc.parts[2].span(0, rows), m.eps, m.shape.groups);
    }
};

fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// tf_nemo_conv: a window's conv (zs_nemotron_mamba_conv's comment has the shapes).
pub fn conv(proj: cuda.View, base: cuda.View, raw: cuda.View, xc: cuda.View, cw: cuda.View, cb: cuda.View, meta: cuda.View, xd: usize) cuda.Error!void {
    return cuda.launched(zs_nemotron_mamba_conv(&proj, &base, &raw, &xc, &cw, &cb, &meta, @intCast(xd)));
}

/// tf_nemo_scan: a window's scan into y bf16 [rows, xd] (before the group norm).
pub fn scan(proj: cuda.View, xc: cuda.View, dt: cuda.View, ssm: cuda.View, a: cuda.View, dsk: cuda.View, dtb: cuda.View, meta: cuda.View, y: cuda.View, lo: f32, hi: f32, groups: usize) cuda.Error!void {
    return cuda.launched(zs_nemotron_mamba_scan(&proj, &xc, &dt, &ssm, &a, &dsk, &dtb, &meta, &y, lo, hi, @intCast(groups)));
}

/// A prompt chunk's conv into xw bf16 [rows, cd], base committed through the row before the last.
pub fn convWide(proj: cuda.View, base: cuda.View, raw: cuda.View, xc: cuda.View, xw: cuda.View, cw: cuda.View, cb: cuda.View, meta: cuda.View, xd: usize) cuda.Error!void {
    return cuda.launched(zs_nemotron_mamba_conv_wide(&proj, &base, &raw, &xc, &xw, &cw, &cb, &meta, @intCast(xd)));
}

/// A prompt chunk's scan from xw, ssm committed through the row before the last.
pub fn scanWide(proj: cuda.View, xc: cuda.View, xw: cuda.View, dt: cuda.View, ssm: cuda.View, a: cuda.View, dsk: cuda.View, dtb: cuda.View, meta: cuda.View, y: cuda.View, lo: f32, hi: f32, groups: usize) cuda.Error!void {
    return cuda.launched(zs_nemotron_mamba_scan_wide(&proj, &xc, &xw, &dt, &ssm, &a, &dsk, &dtb, &meta, &y, lo, hi, @intCast(groups)));
}

/// a = -exp(alog), fp32, as TensorFold's load computes it.
pub fn decay(alog: cuda.View, a: cuda.View) cuda.Error!void {
    return cuda.launched(zs_nemotron_mamba_a(&alog, &a));
}

extern fn zs_nemotron_mamba_conv(p: *const cuda.View, base: *const cuda.View, raw: *const cuda.View, xc: *const cuda.View, cw: *const cuda.View, cb: *const cuda.View, meta: *const cuda.View, xd: i64) c_int;
extern fn zs_nemotron_mamba_scan(p: *const cuda.View, xc: *const cuda.View, dt: *const cuda.View, ssm: *const cuda.View, a: *const cuda.View, dsk: *const cuda.View, dtb: *const cuda.View, meta: *const cuda.View, y: *const cuda.View, lo: f64, hi: f64, groups: i64) c_int;
extern fn zs_nemotron_mamba_conv_wide(p: *const cuda.View, base: *const cuda.View, raw: *const cuda.View, xc: *const cuda.View, xw: *const cuda.View, cw: *const cuda.View, cb: *const cuda.View, meta: *const cuda.View, xd: i64) c_int;
extern fn zs_nemotron_mamba_scan_wide(p: *const cuda.View, xc: *const cuda.View, xw: *const cuda.View, dt: *const cuda.View, ssm: *const cuda.View, a: *const cuda.View, dsk: *const cuda.View, dtb: *const cuda.View, meta: *const cuda.View, y: *const cuda.View, lo: f64, hi: f64, groups: i64) c_int;
extern fn zs_nemotron_mamba_a(alog: *const cuda.View, a: *const cuda.View) c_int;
