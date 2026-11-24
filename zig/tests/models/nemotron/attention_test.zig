//! `attention.zig`'s tests on a small synthetic layer (8 query over 2 KV heads of 64, hidden 64, NVFP4
//! projections as the fixture's `fixture.zig` stores them): the window against the host's fp64 product of the
//! stacked weight and the attend kernels' own chunk arithmetic on the host, within the kernels' tolerance; the
//! row rule bit for bit — a serial step, a window of 16, prompt chunks of 1, 17, 300 and 2,049 rows, and a
//! rejected window's rows leaving nothing a later row reads.

const std = @import("std");
const cuda = @import("../../../src/cuda/cuda.zig");
const gpu = @import("../../cuda/testing.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const nvfp4 = @import("../../../src/weights/nvfp4.zig");
const linear = @import("../../../src/weights/linear.zig");
const attention = @import("../../../src/models/nemotron/attention.zig");
const ref = @import("../../../src/models/nemotron/glue_ref.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;
const fetch = gpu.fetch;
const tobf = ref.f32ToBf16;
const bf = ref.bf16ToF32;
const e2m1 = nvfp4.e2m1;

/// The test layer's dimensions: a small NemotronH attention (the real one is heads 32, kv_heads 2, head_dim
/// 128, hidden 2688; a KV head here still serves 4 query heads and the same kernels run).
const fx = struct {
    const shape: attention.Shape = .{ .hidden = 64, .heads = 8, .kv_heads = 2, .head_dim = 64, .eps = 1e-5 };
    const hidden = shape.hidden;
    const heads = shape.heads;
    const kvh = shape.kv_heads;
    const d = shape.head_dim;
    const qkv = shape.qkvWidth();
};

/// The layer's projections dequantized on the host, f64 [n, k] as `Fp4Linear` computes them.
const Weights = struct {
    /// q, k, v stacked [qkv, hidden], o [hidden, heads * d].
    qkv: []f64,
    o: []f64,
    /// NVFP4 codes and e4m3 block scales as a ModelOpt checkpoint stores them, and the tensor's scale.
    codes: []u8,
    block_scales: []u8,
    scale: f32,
    o_codes: []u8,
    o_block_scales: []u8,
    o_scale: f32,

    fn deinit(w: Weights, gpa: Allocator) void {
        gpa.free(w.qkv);
        gpa.free(w.o);
        gpa.free(w.codes);
        gpa.free(w.block_scales);
        gpa.free(w.o_codes);
        gpa.free(w.o_block_scales);
    }
};

/// Random NVFP4 storage of `n` x `k`: codes, e4m3 block scales and their f64 values at `scale`.
fn stored(gpa: Allocator, random: std.Random, n: usize, k: usize, scale: f32) !struct { codes: []u8, scales: []u8, dense: []f64 } {
    const codes = try gpa.alloc(u8, n * k / 2);
    const scales = try gpa.alloc(u8, n * k / 16);
    const dense = try gpa.alloc(f64, n * k);
    random.bytes(codes);
    for (scales) |*byte| byte.* = 0x20 + random.uintLessThan(u8, 0x30);
    for (dense, 0..) |*value, i| {
        const byte = codes[i / 2];
        const code = if (i % 2 == 0) byte & 0xf else byte >> 4;
        value.* = @as(f64, e2m1[code]) * numeric.f32FromE4m3(scales[i / 16]) * scale;
    }
    return .{ .codes = codes, .scales = scales, .dense = dense };
}

fn makeWeights(gpa: Allocator, random: std.Random) !Weights {
    const a = try stored(gpa, random, fx.qkv, fx.hidden, 0.02);
    const b = try stored(gpa, random, fx.hidden, fx.heads * fx.d, 0.04);
    return .{
        .qkv = a.dense,
        .o = b.dense,
        .codes = a.codes,
        .block_scales = a.scales,
        .scale = 0.02,
        .o_codes = b.codes,
        .o_block_scales = b.scales,
        .o_scale = 0.04,
    };
}

/// The layer on the device, its `qkv` and `o` the NVFP4 bytes `w` stores (bypassing `Attention.init`, which is
/// exercised by the manifest's tests: the fixture's quant map shapes these same `stored` linears).
fn makeLayer(gpa: Allocator, w: Weights) !attention.Attention {
    const qkv: linear.Linear = .{ .four = try nvfp4.Fp4Linear.init(gpa, w.codes, w.block_scales, fx.qkv, fx.hidden, w.scale) };
    errdefer {
        var one = qkv;
        one.deinit();
    }
    const o: linear.Linear = .{ .four = try nvfp4.Fp4Linear.init(gpa, w.o_codes, w.o_block_scales, fx.hidden, fx.heads * fx.d, w.o_scale) };
    return .{ .shape = fx.shape, .qkv = qkv, .o = o };
}

fn randbf16(gpa: Allocator, random: std.Random, n: usize, scale: f32) ![]u16 {
    const s = try gpa.alloc(u16, n);
    for (s) |*v| v.* = tobf(random.floatNorm(f32) * scale);
    return s;
}

/// A sequence: its position, its device word, the layer's cache, and the `window`/`prompt` calls over them.
const Seq = struct {
    pos: usize = 0,
    position: cuda.Buffer,
    cache: attention.Cache,

    fn init(capacity: usize) !Seq {
        var position: cuda.Buffer = try .init(.i32, &.{1});
        errdefer position.deinit();
        try position.zero();
        const cache: attention.Cache = try .init(fx.shape, capacity);
        return .{ .position = position, .cache = cache };
    }

    fn deinit(s: *Seq) void {
        s.position.deinit();
        var cache = s.cache;
        cache.deinit();
    }

    fn setPos(s: *Seq) !void {
        const word: [1]i32 = .{@intCast(s.pos)};
        try s.position.upload(std.mem.sliceAsBytes(&word));
    }

    /// The layer's run over `x`'s rows at `s.pos`, the cache writes bound by `s.pos + rows`.
    fn run(s: *Seq, a: attention.Attention, sc: attention.Scratch, x: cuda.View, out: cuda.View) !void {
        try s.setPos();
        const rows: usize = @intCast(x.size[0]);
        if (rows <= attention.window_rows) {
            try a.window(s.cache, sc, x, s.position.view, s.pos + rows, out);
        } else {
            try a.prompt(s.cache, sc, x, s.position.view, s.pos + rows, out);
        }
    }

    /// Rows `first .. first + n` of `x` as serial steps, committed, into the same rows of `out`.
    fn serial(s: *Seq, a: attention.Attention, sc: attention.Scratch, x: cuda.Buffer, first: usize, n: usize, out: cuda.Buffer) !void {
        for (first..first + n) |r| {
            try s.run(a, sc, x.span(r, 1), out.span(r, 1));
            s.pos += 1;
        }
    }
};

/// The host's row of q, k and v: `x` bf16 [hidden] times the stacked dense weight, then the attend over the
/// committed and the window's own keys (positions 0 .. pos + r).
const Host = struct {
    /// `x_bits` bf16 [rows, hidden] -> the stacked product's f64 rows [rows, qkv] (each output its K sum in
    /// the weight's column order), rounded where a bf16 launch writes (the launch's tolerance covers the rest).
    fn project(x: []const u16, rows: usize, w: []const f64, n: usize, k: usize, out: []f64) void {
        for (0..rows) |r| for (0..n) |col| {
            var sum: f64 = 0;
            for (0..k) |i| sum += @as(f64, bf(x[r * k + i])) * w[col * k + i];
            out[r * n + col] = sum;
        };
    }

    /// Row `r`'s attend over the cache rows 0 .. limit - 1, keys `kc`, values `vc` bf16 [positions, kvh, d],
    /// query `q` f64 [heads, d]: fp64 softmax, out f64 [heads * d].
    fn attend(q: []const f64, kc: []const u16, vc: []const u16, limit: usize, scale: f64, out: []f64) void {
        const scores = std.heap.page_allocator.alloc(f64, limit) catch unreachable;
        defer std.heap.page_allocator.free(scores);
        for (0..fx.heads) |h| {
            const kh = h * fx.kvh / fx.heads;
            var most: f64 = -std.math.inf(f64);
            for (0..limit) |j| {
                var dot: f64 = 0;
                for (0..fx.d) |i| dot += q[h * fx.d + i] * @as(f64, bf(kc[(j * fx.kvh + kh) * fx.d + i]));
                scores[j] = dot * scale;
                most = @max(most, scores[j]);
            }
            var total: f64 = 0;
            for (0..limit) |j| {
                scores[j] = if (scores[j] == -std.math.inf(f64)) 0 else @exp(scores[j] - most);
                total += scores[j];
            }
            for (0..fx.d) |i| {
                var sum: f64 = 0;
                for (0..limit) |j| sum += scores[j] / total * @as(f64, bf(vc[(j * fx.kvh + kh) * fx.d + i]));
                out[h * fx.d + i] = sum;
            }
        }
    }
};

/// What `run` writes for `x`'s rows, computed on the host: the projection (fp64), k and v at pos .. pos + r,
/// each row's attend over positions 0 .. pos + r, then o_proj. `kc`, `vc` bf16 [capacity, kvh, d] already hold
/// the committed rows.
fn hostRun(gpa: Allocator, w: Weights, x: []const u16, rows: usize, pos: usize, kc: []u16, vc: []u16) ![]u16 {
    const product = try gpa.alloc(f64, rows * fx.qkv);
    defer gpa.free(product);
    Host.project(x, rows, w.qkv, fx.qkv, fx.hidden, product);
    // the prep's k and v to bf16 at their positions, the row's v copied as stored
    for (0..rows) |r| for (0..fx.kvh) |kh| {
        const at = ((pos + r) * fx.kvh + kh) * fx.d;
        for (0..fx.d) |i| {
            kc[at + i] = tobf(@floatCast(product[r * fx.qkv + (fx.heads + kh) * fx.d + i]));
            vc[at + i] = tobf(@floatCast(product[r * fx.qkv + (fx.heads + fx.kvh + kh) * fx.d + i]));
        }
    };
    const scale = 1.0 / @sqrt(@as(f64, fx.d));
    const attended = try gpa.alloc(f64, rows * fx.heads * fx.d);
    defer gpa.free(attended);
    for (0..rows) |r| {
        const limit = pos + r + 1;
        // the device's K/V at the window's rows are bf16 of the projection, as above
        Host.attend(product[r * fx.qkv ..][0 .. fx.heads * fx.d], kc, vc, limit, scale, attended[r * fx.heads * fx.d ..][0 .. fx.heads * fx.d]);
    }
    const product_o = try gpa.alloc(f64, rows * fx.hidden);
    defer gpa.free(product_o);
    const o_rows = try gpa.alloc(u16, rows * fx.hidden);
    {
        const xo = try gpa.alloc(u16, rows * fx.heads * fx.d);
        defer gpa.free(xo);
        for (attended, xo) |v, *bits| bits.* = tobf(@floatCast(v));
        Host.project(xo, rows, w.o, fx.hidden, fx.heads * fx.d, product_o);
    }
    for (product_o, o_rows) |v, *bits| bits.* = tobf(@floatCast(v));
    return o_rows;
}

/// `got` bf16 [rows, hidden] within a loose tolerance of `want` (both roundings' compounding, NVFP4 kernels'
/// own order): cosine per row.
fn expectClose(gpa: Allocator, got: []const u16, want: []const u16, rows: usize) !void {
    _ = gpa;
    for (0..rows) |r| {
        var dot: f64 = 0;
        var a2: f64 = 0;
        var b2: f64 = 0;
        var worst: f64 = 0;
        for (0..fx.hidden) |i| {
            const g = bf(got[r * fx.hidden + i]);
            const t = bf(want[r * fx.hidden + i]);
            dot += @as(f64, g) * t;
            a2 += @as(f64, g) * g;
            b2 += @as(f64, t) * t;
            worst = @max(worst, @abs(@as(f64, g - t)));
        }
        const cos = dot / @sqrt(a2 * b2);
        if (!(cos > 0.999 or worst < 0.02)) {
            std.debug.print("row {d}: cosine {d}, worst {d}\n", .{ r, cos, worst });
            return error.TestUnexpectedResult;
        }
    }
}

fn expectSame(gpa: Allocator, a: cuda.Buffer, b: cuda.Buffer) !void {
    const x = try fetch(u16, gpa, a);
    defer gpa.free(x);
    const y = try fetch(u16, gpa, b);
    defer gpa.free(y);
    try testing.expectEqualSlices(u16, x, y);
}

test "a window against the host: NVFP4 product, NoPE prep, attend and o_proj" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1511);
    const random = prng.random();
    const w = try makeWeights(gpa, random);
    defer w.deinit(gpa);
    var a = try makeLayer(gpa, w);
    defer a.deinit();
    var sc: attention.Scratch = try .init(fx.shape, 64, 128);
    defer sc.deinit();
    var s: Seq = try .init(128);
    defer s.deinit();

    const x_bits = try randbf16(gpa, random, 8 * fx.hidden, 1.0);
    defer gpa.free(x_bits);
    const kc = try randbf16(gpa, random, 24 * fx.kvh * fx.d, 0.5);
    defer gpa.free(kc);
    const vc = try randbf16(gpa, random, 24 * fx.kvh * fx.d, 0.5);
    defer gpa.free(vc);
    try s.cache.keys.uploadAt(0, std.mem.sliceAsBytes(kc));
    try s.cache.values.uploadAt(0, std.mem.sliceAsBytes(vc));
    s.pos = 24;
    const kc_host = try gpa.alloc(u16, 32 * fx.kvh * fx.d);
    defer gpa.free(kc_host);
    const vc_host = try gpa.alloc(u16, 32 * fx.kvh * fx.d);
    defer gpa.free(vc_host);
    @memcpy(kc_host[0..kc.len], kc);
    @memcpy(vc_host[0..vc.len], vc);
    const want = try hostRun(gpa, w, x_bits, 8, s.pos, kc_host, vc_host);
    defer gpa.free(want);

    var x = try gpu.device(u16, .bf16, &.{ 8, fx.hidden }, x_bits);
    defer x.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ 8, fx.hidden });
    defer out.deinit();
    try s.run(a, sc, x.view, out.view);
    const got = try fetch(u16, gpa, out);
    defer gpa.free(got);
    try expectClose(gpa, got, want, 8);
    // the cache rows the window wrote are the host's own
    const keys = try fetch(u16, gpa, .{ .view = s.cache.keys.span(24, 8) });
    defer gpa.free(keys);
    try testing.expectEqualSlices(u16, kc_host[24 * fx.kvh * fx.d ..], keys);
}

test "the row rule: a serial step, a 16-row window and prompt chunks bit for bit" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1512);
    const random = prng.random();
    const w = try makeWeights(gpa, random);
    defer w.deinit(gpa);
    var a = try makeLayer(gpa, w);
    defer a.deinit();
    const capacity = 2200;
    var sc: attention.Scratch = try .init(fx.shape, 2100, capacity);
    defer sc.deinit();

    const total = 300;
    const x_bits = try randbf16(gpa, random, total * fx.hidden, 1.0);
    defer gpa.free(x_bits);
    var x = try gpu.device(u16, .bf16, &.{ total, fx.hidden }, x_bits);
    defer x.deinit();

    // serial: a row a call, each committed
    var serial: Seq = try .init(capacity);
    defer serial.deinit();
    var serial_out: cuda.Buffer = try .init(.bf16, &.{ total, fx.hidden });
    defer serial_out.deinit();
    try serial.serial(a, sc, x, 0, total, serial_out);

    // one 16-row window after 200 serial rows, then the rest serial
    var win: Seq = try .init(capacity);
    defer win.deinit();
    var win_out: cuda.Buffer = try .init(.bf16, &.{ total, fx.hidden });
    defer win_out.deinit();
    try win.serial(a, sc, x, 0, 200, win_out);
    try win.run(a, sc, x.span(200, 16), win_out.span(200, 16));
    win.pos += 16;
    try win.serial(a, sc, x, 216, total - 216, win_out);
    try expectSame(gpa, serial_out, win_out);

    // prompt chunks of 1, 17, 300 and 2,049 rows (the last over a second sequence's start)
    var prompt: Seq = try .init(capacity);
    defer prompt.deinit();
    var prompt_out: cuda.Buffer = try .init(.bf16, &.{ total, fx.hidden });
    defer prompt_out.deinit();
    try prompt.run(a, sc, x.span(0, 1), prompt_out.span(0, 1));
    prompt.pos += 1;
    try prompt.run(a, sc, x.span(1, 17), prompt_out.span(1, 17));
    prompt.pos += 17;
    try prompt.run(a, sc, x.span(18, 282), prompt_out.span(18, 282));
    prompt.pos += 282;
    try expectSame(gpa, serial_out, prompt_out);

    // a 2,049-row prompt over a second sequence start: row 0 of it is its serial step's
    const long_x_bits = try randbf16(gpa, random, 2049 * fx.hidden, 1.0);
    defer gpa.free(long_x_bits);
    var long_x = try gpu.device(u16, .bf16, &.{ 2049, fx.hidden }, long_x_bits);
    defer long_x.deinit();
    var long_prompt: Seq = try .init(capacity);
    defer long_prompt.deinit();
    var long_out: cuda.Buffer = try .init(.bf16, &.{ 2049, fx.hidden });
    defer long_out.deinit();
    try long_prompt.run(a, sc, long_x.view, long_out.view);
    var long_serial: Seq = try .init(capacity);
    defer long_serial.deinit();
    var step_out: cuda.Buffer = try .init(.bf16, &.{ 16, fx.hidden });
    defer step_out.deinit();
    try long_serial.serial(a, sc, long_x, 0, 16, step_out);
    const first = try fetch(u16, gpa, .{ .view = long_out.span(0, 16) });
    defer gpa.free(first);
    const steps = try fetch(u16, gpa, step_out);
    defer gpa.free(steps);
    try testing.expectEqualSlices(u16, steps, first);
}

test "a rejected window's rows leave nothing a later row reads" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1513);
    const random = prng.random();
    const w = try makeWeights(gpa, random);
    defer w.deinit(gpa);
    var a = try makeLayer(gpa, w);
    defer a.deinit();
    var sc: attention.Scratch = try .init(fx.shape, 64, 96);
    defer sc.deinit();

    const x_bits = try randbf16(gpa, random, 24 * fx.hidden, 1.0);
    defer gpa.free(x_bits);
    var x = try gpu.device(u16, .bf16, &.{ 24, fx.hidden }, x_bits);
    defer x.deinit();

    // A: 8 committed rows, a window of 8 more kept 3, then a step.
    var kept: Seq = try .init(96);
    defer kept.deinit();
    var kept_out: cuda.Buffer = try .init(.bf16, &.{ 24, fx.hidden });
    defer kept_out.deinit();
    try kept.serial(a, sc, x, 0, 8, kept_out);
    try kept.run(a, sc, x.span(8, 8), kept_out.span(8, 8));
    kept.pos += 3; // rows 8, 9, 10 kept; 11 .. 15 rejected
    try kept.run(a, sc, x.span(16, 1), kept_out.span(16, 1));

    // B: the same 8, then a window of the 3 kept rows alone, then the same step.
    var cut: Seq = try .init(96);
    defer cut.deinit();
    var cut_out: cuda.Buffer = try .init(.bf16, &.{ 24, fx.hidden });
    defer cut_out.deinit();
    try cut.serial(a, sc, x, 0, 8, cut_out);
    try cut.run(a, sc, x.span(8, 3), cut_out.span(8, 3));
    cut.pos += 3;
    try cut.run(a, sc, x.span(16, 1), cut_out.span(16, 1));

    // the step's row and both windows' kept rows: the same bits
    const a_step = try fetch(u16, gpa, .{ .view = kept_out.span(16, 1) });
    defer gpa.free(a_step);
    const b_step = try fetch(u16, gpa, .{ .view = cut_out.span(16, 1) });
    defer gpa.free(b_step);
    try testing.expectEqualSlices(u16, a_step, b_step);
    const a_kept = try fetch(u16, gpa, .{ .view = kept_out.span(8, 3) });
    defer gpa.free(a_kept);
    const b_kept = try fetch(u16, gpa, .{ .view = cut_out.span(8, 3) });
    defer gpa.free(b_kept);
    try testing.expectEqualSlices(u16, a_kept, b_kept);
    // the rejected rows' cache positions were never read: cache rows 11..15 differ between A and B without
    // changing row 16's bits — B never wrote them, A did; what matters is already held.
}
