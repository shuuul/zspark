//! `mamba.zig`'s tests at Transcribe's dimensions (64 heads of 64, 8 groups, state 128, 6,144 conv channels):
//! the step kernels against `glue_ref.zig`'s twins bit for bit; a window's rows against a wider window's and a
//! prompt's; a window, a window with rows rolled back, and prompts of 1 to 2,049 rows against their serial
//! steps, outputs and committed state bit for bit, and a step after each; inputs at softplus's and exp's ends.

const std = @import("std");
const cuda = @import("../../../src/cuda/cuda.zig");
const gpu = @import("../../cuda/testing.zig");
const mamba = @import("../../../src/models/nemotron/mamba.zig");
const ref = @import("../../../src/models/nemotron/glue_ref.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;
const Meta = mamba.Meta;
const fetch = gpu.fetch;
const tobf = ref.f32ToBf16;

const fx = struct {
    const shape: mamba.Shape = .{};
    const cd = shape.cd();
    const proj = shape.proj();
    const xd = shape.xd;
    const heads = shape.heads;
    const rmax = mamba.window_rows;
    const eps: f64 = 1e-5;
    const ref_shape: ref.Shape = .{ .proj = proj, .xd = xd, .cd = cd, .heads = heads, .dh = shape.head_dim, .groups = shape.groups, .rmax = rmax };
};

fn randbf16(gpa: Allocator, random: std.Random, n: usize, scale: f32, shift: f32) ![]u16 {
    const s = try gpa.alloc(u16, n);
    for (s) |*v| v.* = tobf(shift + random.floatNorm(f32) * scale);
    return s;
}

fn randf32(gpa: Allocator, random: std.Random, n: usize, scale: f32) ![]f32 {
    const s = try gpa.alloc(f32, n);
    for (s) |*v| v.* = random.floatNorm(f32) * scale;
    return s;
}

/// The checkpoint's bf16 tensors of one layer, drawn near the published ones' ranges.
const Weights = struct {
    a_log: []u16,
    d: []u16,
    dt_bias: []u16,
    conv_w: []u16,
    conv_b: []u16,
    norm: []u16,

    fn init(gpa: Allocator, random: std.Random) !Weights {
        return .{
            .a_log = try randbf16(gpa, random, fx.heads, 1.0, 1.0),
            .d = try randbf16(gpa, random, fx.heads, 0.5, 1.0),
            .dt_bias = try randbf16(gpa, random, fx.heads, 1.0, -2.0),
            .conv_w = try randbf16(gpa, random, 4 * fx.cd, 0.3, 0.0),
            .conv_b = try randbf16(gpa, random, fx.cd, 0.1, 0.0),
            .norm = try randbf16(gpa, random, fx.xd, 0.2, 1.0),
        };
    }

    fn deinit(w: Weights, gpa: Allocator) void {
        inline for (.{ w.a_log, w.d, w.dt_bias, w.conv_w, w.conv_b, w.norm }) |s| gpa.free(s);
    }

    fn mixer(w: Weights, gpa: Allocator) !mamba.Mixer {
        return .init(gpa, fx.shape, .{ .a_log = w.a_log, .d = w.d, .dt_bias = w.dt_bias, .conv_w = w.conv_w, .conv_b = w.conv_b, .norm = w.norm }, 0, std.math.inf(f32), fx.eps);
    }
};

/// in_proj's rows: z and the conv channels at `scale`, dt around `dt_shift`.
fn projRows(gpa: Allocator, random: std.Random, rows: usize, scale: f32, dt_shift: f32, dt_scale: f32) !cuda.Buffer {
    const host = try randbf16(gpa, random, rows * fx.proj, scale, 0.0);
    defer gpa.free(host);
    for (0..rows) |r| for (0..fx.heads) |h| {
        host[r * fx.proj + fx.xd + fx.cd + h] = tobf(dt_shift + random.floatNorm(f32) * dt_scale);
    };
    return gpu.device(u16, .bf16, &.{ rows, fx.proj }, host);
}

/// One sequence: a layer's state and the meta row it is run with.
const Seq = struct {
    st: mamba.State,
    meta: cuda.Buffer,
    m: Meta = .{},

    fn init() !Seq {
        var st: mamba.State = try .init(fx.shape);
        errdefer st.deinit();
        var meta: cuda.Buffer = try .init(.i32, &.{4});
        errdefer meta.deinit();
        return .{ .st = st, .meta = meta };
    }

    fn deinit(s: *Seq) void {
        s.st.deinit();
        s.meta.deinit();
    }

    /// The other sequence's state and meta.
    fn copy(s: *Seq, from: Seq) !void {
        try s.st.block.copyFrom(from.st.block);
        s.m = from.m;
    }

    fn window(s: *Seq, mx: mamba.Mixer, sc: mamba.Scratch, proj: cuda.View, out: cuda.View, keep: usize) !void {
        try s.m.set(s.meta);
        try mx.window(s.st, sc, proj, s.meta.view, out);
        s.m = s.m.afterWindow(keep);
    }

    fn prompt(s: *Seq, mx: mamba.Mixer, sc: mamba.Scratch, proj: cuda.View, out: cuda.View) !void {
        try s.m.set(s.meta);
        try mx.prompt(s.st, sc, proj, s.meta.view, out);
        s.m = s.m.afterPrompt(@intCast(proj.size[0]));
    }

    /// Rows [first, first + n) of proj as serial steps, each kept, into the same rows of out.
    fn serial(s: *Seq, mx: mamba.Mixer, sc: mamba.Scratch, proj: cuda.Buffer, first: usize, n: usize, out: cuda.Buffer) !void {
        for (first..first + n) |r| try s.window(mx, sc, proj.span(r, 1), out.span(r, 1), 1);
    }
};

fn expectSameBuffer(comptime T: type, gpa: Allocator, a: cuda.Buffer, b: cuda.Buffer) !void {
    const x = try fetch(T, gpa, a);
    defer gpa.free(x);
    const y = try fetch(T, gpa, b);
    defer gpa.free(y);
    try testing.expectEqualSlices(T, x, y);
}

fn expectSameRows(gpa: Allocator, a: cuda.Buffer, b: cuda.Buffer, first: usize, n: usize) !void {
    const x = try fetch(u16, gpa, a);
    defer gpa.free(x);
    const y = try fetch(u16, gpa, b);
    defer gpa.free(y);
    const w: usize = @intCast(a.view.size[1]);
    try testing.expectEqualSlices(u16, x[first * w ..][0 .. n * w], y[first * w ..][0 .. n * w]);
}

/// Both sequences committed through the same rows, with the same rows pending: the committed state, and the
/// pending rows' inputs, conv outputs and dt, bit for bit.
fn expectSameCommitted(gpa: Allocator, a: Seq, b: Seq) !void {
    try testing.expectEqual(a.m.pos, b.m.pos);
    try testing.expectEqual(a.m.prev_keep, b.m.prev_keep);
    try expectSameBuffer(f32, gpa, a.st.ssm(), b.st.ssm());
    try expectSameBuffer(u16, gpa, a.st.base(), b.st.base());
    const pk: usize = a.m.prev_keep;
    inline for (.{ .{ u16, "raw" }, .{ u16, "xc" }, .{ f32, "dt" } }) |c| {
        const T = c[0];
        const name = c[1];
        const x = try fetch(T, gpa, @field(mamba.State, name)(a.st));
        defer gpa.free(x);
        const y = try fetch(T, gpa, @field(mamba.State, name)(b.st));
        defer gpa.free(y);
        const w = x.len / (2 * fx.rmax);
        const pa: usize = 1 - a.m.parity;
        const pb: usize = 1 - b.m.parity;
        try testing.expectEqualSlices(T, x[pa * fx.rmax * w ..][0 .. pk * w], y[pb * fx.rmax * w ..][0 .. pk * w]);
    }
}

fn expectFinite(gpa: Allocator, out: cuda.Buffer) !void {
    const y = try fetch(u16, gpa, out);
    defer gpa.free(y);
    for (y) |v| try testing.expect(std.math.isFinite(ref.bf16ToF32(v)));
}

/// A state as a long sequence leaves it: random ssm and base, and `pk` kept rows pending in buffer 1 - parity.
fn randomState(gpa: Allocator, random: std.Random, s: *Seq, pk: usize, parity: u32) !void {
    const ssm = try randf32(gpa, random, fx.heads * fx.shape.head_dim * fx.shape.state, 0.5);
    defer gpa.free(ssm);
    try s.st.ssm().upload(std.mem.sliceAsBytes(ssm));
    const base = try randbf16(gpa, random, 3 * fx.cd, 1.0, 0.0);
    defer gpa.free(base);
    try s.st.base().upload(std.mem.sliceAsBytes(base));
    const raw = try randbf16(gpa, random, 2 * fx.rmax * fx.cd, 1.0, 0.0);
    defer gpa.free(raw);
    try s.st.raw().upload(std.mem.sliceAsBytes(raw));
    const xc = try randbf16(gpa, random, 2 * fx.rmax * fx.cd, 0.5, 0.0);
    defer gpa.free(xc);
    try s.st.xc().upload(std.mem.sliceAsBytes(xc));
    const dt = try gpa.alloc(f32, 2 * fx.rmax * fx.heads);
    defer gpa.free(dt);
    for (dt) |*v| v.* = random.float(f32) * 0.5;
    try s.st.dt().upload(std.mem.sliceAsBytes(dt));
    s.m = .{ .pos = 40, .parity = parity, .prev_keep = @intCast(pk) };
}

/// The step kernels and the norm against the host twins, from `seq`'s state (a window of `rows`).
fn checkWindowAgainstHost(gpa: Allocator, w: Weights, mx: mamba.Mixer, sc: mamba.Scratch, s: *Seq, proj: cuda.Buffer, rows: usize) !void {
    const base = try fetch(u16, gpa, s.st.base());
    defer gpa.free(base);
    const raw = try fetch(u16, gpa, s.st.raw());
    defer gpa.free(raw);
    const xc = try fetch(u16, gpa, s.st.xc());
    defer gpa.free(xc);
    const dt = try fetch(f32, gpa, s.st.dt());
    defer gpa.free(dt);
    const ssm = try fetch(f32, gpa, s.st.ssm());
    defer gpa.free(ssm);
    const p = try fetch(u16, gpa, proj);
    defer gpa.free(p);
    const parity: usize = s.m.parity;
    const pk: usize = s.m.prev_keep;

    var out: cuda.Buffer = try .init(.bf16, &.{ rows, fx.xd });
    defer out.deinit();
    try s.window(mx, sc, proj.view, out.view, rows);

    const cw = try fetch(f32, gpa, .{ .view = mx.convW() });
    defer gpa.free(cw);
    const cb = try fetch(f32, gpa, .{ .view = mx.convB() });
    defer gpa.free(cb);
    const a = try fetch(f32, gpa, .{ .view = mx.a() });
    defer gpa.free(a);
    const dsk = try fetch(f32, gpa, .{ .view = mx.d() });
    defer gpa.free(dsk);
    const dtb = try fetch(f32, gpa, .{ .view = mx.dtBias() });
    defer gpa.free(dtb);
    // the load: taps transposed, everything widened, a = -exp(A_log) within an ulp of the f64 exp
    for (0..fx.cd) |c| for (0..4) |t| try testing.expectEqual(ref.bf16ToF32(w.conv_w[c * 4 + t]), cw[t * fx.cd + c]);
    for (0..fx.heads) |h| {
        try testing.expectEqual(ref.bf16ToF32(w.d[h]), dsk[h]);
        try testing.expectEqual(ref.bf16ToF32(w.dt_bias[h]), dtb[h]);
        const want = -@exp(@as(f64, ref.bf16ToF32(w.a_log[h])));
        try testing.expect(@abs(@as(f64, a[h]) - want) <= @abs(want) * 0x1p-22);
    }

    ref.conv(p, base, raw, xc, cw, cb, parity, pk, rows, fx.ref_shape);
    const y = try gpa.alloc(u16, rows * fx.xd);
    defer gpa.free(y);
    ref.scan(p, xc, dt, ssm, a, dsk, dtb, y, parity, pk, rows, 0, std.math.inf(f32), fx.ref_shape);
    const normed = try gpa.alloc(u16, rows * fx.xd);
    defer gpa.free(normed);
    const xs = try gpa.alloc(f32, rows * (fx.xd / 64));
    defer gpa.free(xs);
    try ref.groupRmsnorm(gpa, y, w.norm, normed, xs, rows, fx.xd, fx.shape.groups, @floatCast(fx.eps));

    inline for (.{ .{ u16, base, "base" }, .{ u16, raw, "raw" }, .{ u16, xc, "xc" }, .{ f32, dt, "dt" }, .{ f32, ssm, "ssm" } }) |c| {
        const got = try fetch(c[0], gpa, @field(mamba.State, c[2])(s.st));
        defer gpa.free(got);
        try testing.expectEqualSlices(c[0], c[1], got);
    }
    const ys = try fetch(u16, gpa, .{ .view = sc.parts[1].span(0, rows) });
    defer gpa.free(ys);
    try testing.expectEqualSlices(u16, y, ys);
    const got = try fetch(u16, gpa, out);
    defer gpa.free(got);
    try testing.expectEqualSlices(u16, normed, got);
}

test "the window's conv, scan and norm are the host twins' bits, with kept rows replaying" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(151);
    const random = prng.random();
    const w = try Weights.init(gpa, random);
    defer w.deinit(gpa);
    var mx = try w.mixer(gpa);
    defer mx.deinit();
    var sc: mamba.Scratch = try .init(fx.shape, fx.rmax);
    defer sc.deinit();
    var s: Seq = try .init();
    defer s.deinit();
    for ([_][3]usize{ .{ 5, 0, 0 }, .{ 5, 3, 1 }, .{ 1, 1, 0 }, .{ 16, 16, 1 } }) |c| {
        try randomState(gpa, random, &s, c[1], @intCast(c[2]));
        var proj = try projRows(gpa, random, c[0], 1.0, 0.0, 2.0);
        defer proj.deinit();
        try checkWindowAgainstHost(gpa, w, mx, sc, &s, proj, c[0]);
    }
}

test "inputs at softplus's and exp's ends: the host's bits, every output finite, a prompt its steps" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(152);
    const random = prng.random();
    const w = try Weights.init(gpa, random);
    defer w.deinit(gpa);
    // A_log from -20 (a ~ -2e-9, no decay) to 20 (a ~ -5e8, da underflows to 0), dt_bias at both ends
    for (w.a_log, 0..) |*v, h| v.* = tobf(-20.0 + 40.0 * @as(f32, @floatFromInt(h)) / @as(f32, fx.heads - 1));
    for (w.dt_bias, 0..) |*v, h| v.* = tobf(if (h % 3 == 0) -30.0 else if (h % 3 == 1) 0.0 else 30.0);
    var mx = try w.mixer(gpa);
    defer mx.deinit();
    var sc: mamba.Scratch = try .init(fx.shape, 64);
    defer sc.deinit();
    var s: Seq = try .init();
    defer s.deinit();
    // dt at +-1e4 (softplus is v, or e^v underflows), at +-90 (exp's cut at 104 near), and a zero state
    for ([_][2]f32{ .{ 1e4, 0 }, .{ -1e4, 0 }, .{ 0, 90 }, .{ 0, 0 } }) |c| {
        if (c[0] == 0 and c[1] == 0) {
            try s.st.zero();
            s.m = .{};
        } else try randomState(gpa, random, &s, 2, 1);
        var proj = try projRows(gpa, random, 6, if (c[0] == 0 and c[1] == 0) 0.0 else 1.0, c[0], c[1]);
        defer proj.deinit();
        try checkWindowAgainstHost(gpa, w, mx, sc, &s, proj, 6);

        var a: Seq = try .init();
        defer a.deinit();
        var b: Seq = try .init();
        defer b.deinit();
        try a.copy(s);
        try b.copy(s);
        var more = try projRows(gpa, random, 40, 1.0, c[0], c[1] + 1.0);
        defer more.deinit();
        var oa: cuda.Buffer = try .init(.bf16, &.{ 40, fx.xd });
        defer oa.deinit();
        var ob: cuda.Buffer = try .init(.bf16, &.{ 40, fx.xd });
        defer ob.deinit();
        try a.prompt(mx, sc, more.view, oa.view);
        try b.serial(mx, sc, more, 0, 40, ob);
        try expectSameBuffer(u16, gpa, oa, ob);
        try expectSameCommitted(gpa, a, b);
        try expectFinite(gpa, oa);
        const ssm = try fetch(f32, gpa, a.st.ssm());
        defer gpa.free(ssm);
        for (ssm) |v| try testing.expect(std.math.isFinite(v));
    }
}

test "a window's rows are the first rows of a wider window's and of a prompt's, 1 to 16 rows" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(153);
    const random = prng.random();
    const w = try Weights.init(gpa, random);
    defer w.deinit(gpa);
    var mx = try w.mixer(gpa);
    defer mx.deinit();
    var sc: mamba.Scratch = try .init(fx.shape, fx.rmax);
    defer sc.deinit();
    var start: Seq = try .init();
    defer start.deinit();
    try randomState(gpa, random, &start, 4, 0);
    var proj = try projRows(gpa, random, fx.rmax, 1.0, 0.0, 2.0);
    defer proj.deinit();
    var wide: cuda.Buffer = try .init(.bf16, &.{ fx.rmax, fx.xd });
    defer wide.deinit();
    var s: Seq = try .init();
    defer s.deinit();
    try s.copy(start);
    try s.window(mx, sc, proj.view, wide.view, fx.rmax);
    var out: cuda.Buffer = try .init(.bf16, &.{ fx.rmax, fx.xd });
    defer out.deinit();
    for (1..fx.rmax + 1) |rows| {
        try s.copy(start);
        try s.window(mx, sc, proj.span(0, rows), out.span(0, rows), rows);
        try expectSameRows(gpa, wide, out, 0, rows);
        try s.copy(start);
        try s.prompt(mx, sc, proj.span(0, rows), out.span(0, rows));
        try expectSameRows(gpa, wide, out, 0, rows);
    }
}

test "a window is its serial steps: outputs, then the committed state and a step after it" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(154);
    const random = prng.random();
    const w = try Weights.init(gpa, random);
    defer w.deinit(gpa);
    var mx = try w.mixer(gpa);
    defer mx.deinit();
    var sc: mamba.Scratch = try .init(fx.shape, fx.rmax);
    defer sc.deinit();
    var start: Seq = try .init();
    defer start.deinit();
    try randomState(gpa, random, &start, 3, 1);
    var a: Seq = try .init();
    defer a.deinit();
    var b: Seq = try .init();
    defer b.deinit();
    for ([_]usize{ 1, 2, 5, 16 }) |rows| {
        var proj = try projRows(gpa, random, rows + 1, 1.0, 0.0, 2.0);
        defer proj.deinit();
        var oa: cuda.Buffer = try .init(.bf16, &.{ rows + 1, fx.xd });
        defer oa.deinit();
        var ob: cuda.Buffer = try .init(.bf16, &.{ rows + 1, fx.xd });
        defer ob.deinit();
        try a.copy(start);
        try b.copy(start);
        try a.window(mx, sc, proj.span(0, rows), oa.span(0, rows), rows);
        try b.serial(mx, sc, proj, 0, rows, ob);
        try expectSameRows(gpa, oa, ob, 0, rows);
        // the window's kept rows replay in the next step, the steps' last one in theirs
        try a.window(mx, sc, proj.span(rows, 1), oa.span(rows, 1), 1);
        try b.window(mx, sc, proj.span(rows, 1), ob.span(rows, 1), 1);
        try expectSameRows(gpa, oa, ob, rows, 1);
        try expectSameCommitted(gpa, a, b);
    }
}

test "a window's rejected rows leave no trace: keep 3 of 8, then 5 rows, is the kept tokens' serial run" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(155);
    const random = prng.random();
    const w = try Weights.init(gpa, random);
    defer w.deinit(gpa);
    var mx = try w.mixer(gpa);
    defer mx.deinit();
    var sc: mamba.Scratch = try .init(fx.shape, fx.rmax);
    defer sc.deinit();
    var a: Seq = try .init();
    defer a.deinit();
    try randomState(gpa, random, &a, 2, 0);
    var b: Seq = try .init();
    defer b.deinit();
    try b.copy(a);
    // rows 0..8 the drafted window, of which 3 kept; rows 8..13 the next window; row 13 a step
    var proj = try projRows(gpa, random, 14, 1.0, 0.0, 2.0);
    defer proj.deinit();
    var oa: cuda.Buffer = try .init(.bf16, &.{ 14, fx.xd });
    defer oa.deinit();
    var ob: cuda.Buffer = try .init(.bf16, &.{ 14, fx.xd });
    defer ob.deinit();
    try a.window(mx, sc, proj.span(0, 8), oa.span(0, 8), 3);
    try a.window(mx, sc, proj.span(8, 5), oa.span(8, 5), 5);
    try a.window(mx, sc, proj.span(13, 1), oa.span(13, 1), 1);
    try b.serial(mx, sc, proj, 0, 3, ob);
    try b.serial(mx, sc, proj, 8, 6, ob);
    try expectSameRows(gpa, oa, ob, 0, 3);
    try expectSameRows(gpa, oa, ob, 8, 6);
    try expectSameCommitted(gpa, a, b);
}

test "a prompt of 1, 17, 300 or 2,049 rows, whole or in chunks, is its serial steps, and so is a step after it" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(156);
    const random = prng.random();
    const w = try Weights.init(gpa, random);
    defer w.deinit(gpa);
    var mx = try w.mixer(gpa);
    defer mx.deinit();
    var sc: mamba.Scratch = try .init(fx.shape, 2049);
    defer sc.deinit();
    var start: Seq = try .init();
    defer start.deinit();
    var a: Seq = try .init();
    defer a.deinit();
    var b: Seq = try .init();
    defer b.deinit();
    // (rows, the chunk, the kept rows pending before it: 0 a new sequence, 5 after a verify window)
    for ([_][3]usize{ .{ 1, 1, 0 }, .{ 17, 17, 5 }, .{ 300, 300, 0 }, .{ 300, 100, 5 }, .{ 2049, 2049, 5 } }) |c| {
        const rows = c[0];
        if (c[2] == 0) {
            try start.st.zero();
            start.m = .{};
        } else try randomState(gpa, random, &start, c[2], 1);
        var proj = try projRows(gpa, random, rows + 1, 1.0, 0.0, 2.0);
        defer proj.deinit();
        var oa: cuda.Buffer = try .init(.bf16, &.{ rows + 1, fx.xd });
        defer oa.deinit();
        var ob: cuda.Buffer = try .init(.bf16, &.{ rows + 1, fx.xd });
        defer ob.deinit();
        try a.copy(start);
        try b.copy(start);
        var first: usize = 0;
        while (first < rows) : (first += c[1]) {
            const n = @min(c[1], rows - first);
            try a.prompt(mx, sc, proj.span(first, n), oa.span(first, n));
        }
        try b.serial(mx, sc, proj, 0, rows, ob);
        try expectSameRows(gpa, oa, ob, 0, rows);
        try expectSameCommitted(gpa, a, b);
        try a.window(mx, sc, proj.span(rows, 1), oa.span(rows, 1), 1);
        try b.window(mx, sc, proj.span(rows, 1), ob.span(rows, 1), 1);
        try expectSameRows(gpa, oa, ob, rows, 1);
        try expectSameCommitted(gpa, a, b);
    }
}
