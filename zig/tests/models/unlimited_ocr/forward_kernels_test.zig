//! The kernels `forward.zig` launches of its own (`zig/kernels/unlimited_ocr/`), on random inputs: RoPE's prep
//! (the shared `kernels/common/attention_prep.cu`) against fp64 on the host; the router's top 6 on inputs built
//! to tie; the MoE's sum and the residual's add with the model's roundings; the tower's linear with the residual
//! in its epilogue against the linear and then the add.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const forward_file = @import("../../../src/models/unlimited_ocr/forward.zig");
const experts_file = @import("../../../src/weights/experts.zig");
const glue = @import("../../../src/decode/glue.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const vw = @import("../../../src/models/unlimited_ocr/vision_weights.zig");

const D = 128;

fn of(bits: u16) f64 {
    return numeric.f32FromBf16(bits);
}

fn bf(v: f64) f64 {
    return numeric.roundBf16(@floatCast(v));
}

fn random16(random: std.Random, out: []u16, spread: f32) void {
    for (out) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32) * spread);
}

fn upload(dtype: cuda.DType, shape: []const usize, bytes: []const u8) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(dtype, shape);
    errdefer buffer.deinit();
    try buffer.upload(bytes);
    return buffer;
}

fn fetch(gpa: std.mem.Allocator, comptime T: type, view: cuda.View) ![]T {
    const out = try gpa.alloc(T, view.count());
    errdefer gpa.free(out);
    try cuda.synchronize();
    try cuda.download(view, std.mem.sliceAsBytes(out));
    return out;
}

test "RoPE's prep against the host: rotate-half with HF's bf16 roundings, the cache rows at their positions" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1371);
    const random = prng.random();
    const heads = 3;
    const rows = 5;
    const pos0 = 3000;
    const cap = pos0 + rows;
    const qkv = try gpa.alloc(u16, rows * 3 * heads * D);
    defer gpa.free(qkv);
    random16(random, qkv, 2.0);
    var inv: [D / 2]f32 = undefined;
    for (&inv, 0..) |*f, d| f.* = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * d)) / D);
    var qkv_b = try upload(.bf16, &.{ rows, 3 * heads * D }, std.mem.sliceAsBytes(qkv));
    defer qkv_b.deinit();
    var inv_b = try upload(.f32, &.{D / 2}, std.mem.sliceAsBytes(&inv));
    defer inv_b.deinit();
    var q: cuda.Buffer = try .init(.bf16, &.{ rows, heads, D });
    defer q.deinit();
    var kc: cuda.Buffer = try .init(.bf16, &.{ cap, heads, D });
    defer kc.deinit();
    var vc: cuda.Buffer = try .init(.bf16, &.{ cap, heads, D });
    defer vc.deinit();
    const words = [2]i32{ pos0, std.math.maxInt(i32) };
    var at = try upload(.i32, &.{2}, std.mem.sliceAsBytes(&words));
    defer at.deinit();
    try cuda.launched(cuda.kernels.zs_attn_prep(&qkv_b.view, null, null, &inv_b.view, &at.view, &q.view, &kc.view, &vc.view, 0));
    const got_q = try fetch(gpa, u16, q.view);
    defer gpa.free(got_q);
    const got_k = try fetch(gpa, u16, kc.span(pos0, rows));
    defer gpa.free(got_k);
    const got_v = try fetch(gpa, u16, vc.span(pos0, rows));
    defer gpa.free(got_v);
    var differ: usize = 0;
    for (0..rows) |r| for (0..2 * heads) |head| for (0..D / 2) |d| {
        const angle: f32 = @as(f32, @floatFromInt(pos0 + r)) * inv[d];
        const c = bf(@cos(angle));
        const s = bf(@sin(angle));
        const x = qkv[r * 3 * heads * D + head * D ..];
        const x1 = of(x[d]);
        const x2 = of(x[d + D / 2]);
        const want = [2]f64{ bf(bf(x1 * c) + bf(-x2 * s)), bf(bf(x2 * c) + bf(x1 * s)) };
        const out = if (head < heads) got_q[(r * heads + head) * D ..] else got_k[(r * heads + head - heads) * D ..];
        for (want, [2]u16{ out[d], out[d + D / 2] }) |w, g| {
            // cosf and sinf against the host's: a bf16 step apart where a value lies at a rounding's edge
            if (of(g) != w) differ += 1;
            try testing.expectApproxEqAbs(w, of(g), (@abs(x1) + @abs(x2)) / 64.0 + 1e-30);
        }
    };
    try testing.expect(differ * 100 <= rows * 2 * heads * D);
    for (0..rows) |r| try testing.expectEqualSlices(u16, qkv[r * 3 * heads * D + 2 * heads * D ..][0 .. heads * D], got_v[r * heads * D ..][0 .. heads * D]);
}

test "the router's top 6 of 64: largest softmax first, the lower id among equals, unnormalized" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1372);
    const random = prng.random();
    const rows = 9;
    const E = 64;
    const top = 6;
    var logits: [rows * E]u16 = undefined;
    random16(random, &logits, 0.5);
    // row 0 all equal; row 1 eight equal at the top; row 2 a tie across the sixth place
    for (logits[0..E]) |*l| l.* = numeric.bf16FromF32(0.5);
    for ([_]usize{ 3, 9, 17, 22, 40, 41, 50, 63 }) |e| logits[E + e] = numeric.bf16FromF32(9.0);
    for ([_]usize{ 1, 2, 3, 4, 5 }) |e| logits[2 * E + e] = numeric.bf16FromF32(8.0 - @as(f32, @floatFromInt(e)));
    for ([_]usize{ 60, 30, 10 }) |e| logits[2 * E + e] = numeric.bf16FromF32(2.5);
    var lb = try upload(.bf16, &.{ rows, E }, std.mem.sliceAsBytes(&logits));
    defer lb.deinit();
    var picks: cuda.Buffer = try .init(.i32, &.{ rows, top });
    defer picks.deinit();
    var weights: cuda.Buffer = try .init(.f32, &.{ rows, top });
    defer weights.deinit();
    try cuda.launched(forward_file.zs_unlimited_ocr_moe_route(&lb.view, &picks.view, &weights.view, null, null, null));
    const got_p = try fetch(gpa, i32, picks.view);
    defer gpa.free(got_p);
    const got_w = try fetch(gpa, f32, weights.view);
    defer gpa.free(got_w);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5 }, got_p[0..top]);
    try testing.expectEqualSlices(i32, &.{ 3, 9, 17, 22, 40, 41 }, got_p[top..][0..top]);
    try testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4, 5, 10 }, got_p[2 * top ..][0..top]);
    for (0..rows) |r| {
        const row = logits[r * E ..][0..E];
        var most: f64 = -std.math.inf(f64);
        for (row) |l| most = @max(most, of(l));
        var total: f64 = 0;
        for (row) |l| total += @exp(of(l) - most);
        // the host's picks: by logit down, id up
        var order: [E]usize = undefined;
        for (&order, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, &order, @as([]const u16, row), struct {
            fn before(l: []const u16, a: usize, b: usize) bool {
                return of(l[a]) > of(l[b]) or (of(l[a]) == of(l[b]) and a < b);
            }
        }.before);
        var want: [top]usize = order[0..top].*;
        std.mem.sort(usize, &want, {}, std.sort.asc(usize));
        for (want, 0..) |e, k| {
            try testing.expectEqual(@as(i32, @intCast(e)), got_p[r * top + k]);
            const p = @exp(of(row[e]) - most) / total;
            try testing.expectApproxEqRel(p, got_w[r * top + k], 1e-5);
        }
    }
    // one row's plan, written by the route, is `Plan.routeAs`'s on the same picks
    var by_route: experts_file.Plan = try .init(1, top, E);
    defer by_route.deinit();
    var by_plan: experts_file.Plan = try .init(1, top, E);
    defer by_plan.deinit();
    var one_p: cuda.Buffer = try .init(.i32, &.{ 1, top });
    defer one_p.deinit();
    var one_w: cuda.Buffer = try .init(.f32, &.{ 1, top });
    defer one_w.deinit();
    for (0..rows) |r| {
        const row = lb.view.span(r, 1);
        try cuda.launched(forward_file.zs_unlimited_ocr_moe_route(&row, &one_p.view, &one_w.view, &by_route.members.view, &by_route.items.view, &by_route.counts.view));
        try by_plan.routeAs(one_p.view, 1, .{ .item = 64, .tiles = 4 });
        inline for (.{ "members", "items", "counts" }) |name| {
            const want = try fetch(gpa, i32, @field(by_plan, name).view);
            defer gpa.free(want);
            const got = try fetch(gpa, i32, @field(by_route, name).view);
            defer gpa.free(got);
            const used = if (comptime std.mem.eql(u8, name, "members")) top else if (comptime std.mem.eql(u8, name, "items")) 3 * top else 2;
            try testing.expectEqualSlices(i32, want[0..used], got[0..used]);
        }
    }
}

test "the MoE's sum with the shared MLP and the residual, and the residual's add, as HF rounds them" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1373);
    const random = prng.random();
    const rows = 7;
    const top = 6;
    const H = 1280;
    const y = try gpa.alloc(u16, rows * top * H);
    defer gpa.free(y);
    random16(random, y, 1.0);
    var w: [rows * top]f32 = undefined;
    for (&w) |*v| v.* = random.float(f32) * 0.3;
    const shared = try gpa.alloc(u16, rows * H);
    defer gpa.free(shared);
    random16(random, shared, 1.0);
    const h = try gpa.alloc(u16, rows * H);
    defer gpa.free(h);
    random16(random, h, 4.0);
    var yb = try upload(.bf16, &.{ rows * top, H }, std.mem.sliceAsBytes(y));
    defer yb.deinit();
    var wb = try upload(.f32, &.{ rows, top }, std.mem.sliceAsBytes(&w));
    defer wb.deinit();
    var sb = try upload(.bf16, &.{ rows, H }, std.mem.sliceAsBytes(shared));
    defer sb.deinit();
    var hb = try upload(.bf16, &.{ rows, H }, std.mem.sliceAsBytes(h));
    defer hb.deinit();
    try cuda.launched(forward_file.zs_unlimited_ocr_moe_combine(&yb.view, &wb.view, &sb.view, &hb.view, null, null, 0));
    const got = try fetch(gpa, u16, hb.view);
    defer gpa.free(got);
    for (0..rows) |r| for (0..H) |d| {
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..top) |k| {
            sum += w[r * top + k] * of(y[(r * top + k) * H + d]);
            mag += @abs(w[r * top + k] * of(y[(r * top + k) * H + d]));
        }
        const want = bf(of(h[r * H + d]) + bf(bf(sum) + of(shared[r * H + d])));
        const scale = mag + @abs(of(shared[r * H + d])) + @abs(of(h[r * H + d]));
        try testing.expectApproxEqAbs(want, of(got[r * H + d]), scale / 128.0 + 1e-30);
    };
    // the add alone: one rounding of the fp32 sum, to the bit
    var xb = try upload(.bf16, &.{ rows, H }, std.mem.sliceAsBytes(shared));
    defer xb.deinit();
    try glue.residualAdd(hb.view, xb.view);
    const added = try fetch(gpa, u16, hb.view);
    defer gpa.free(added);
    for (got, shared, added) |a, x, s| try testing.expectEqual(numeric.bf16FromF32(numeric.f32FromBf16(a) + numeric.f32FromBf16(x)), s);

    // each with the norm after it: the sum's bits as alone, the norm's as common/glue.cu's `zs_rmsnorm` of it
    const norm_w = try gpa.alloc(u16, H);
    defer gpa.free(norm_w);
    random16(random, norm_w, 0.5);
    var nw = try upload(.bf16, &.{H}, std.mem.sliceAsBytes(norm_w));
    defer nw.deinit();
    var normed: cuda.Buffer = try .init(.bf16, &.{ rows, H });
    defer normed.deinit();
    var want: cuda.Buffer = try .init(.bf16, &.{ rows, H });
    defer want.deinit();
    const eps = 1e-6;
    for ([_]bool{ true, false }) |combine| {
        try hb.upload(std.mem.sliceAsBytes(h));
        if (combine) {
            try cuda.launched(forward_file.zs_unlimited_ocr_moe_combine(&yb.view, &wb.view, &sb.view, &hb.view, &nw.view, &normed.view, eps));
        } else try glue.residualNormBf16(hb.view, xb.view, nw.view, normed.view, eps);
        const sum = try fetch(gpa, u16, hb.view);
        defer gpa.free(sum);
        if (combine) try testing.expectEqualSlices(u16, got, sum) else {
            for (h, shared, sum) |a, x, s| try testing.expectEqual(numeric.bf16FromF32(numeric.f32FromBf16(a) + numeric.f32FromBf16(x)), s);
        }
        try glue.rmsnorm(hb.view, nw.view, want.view, eps);
        const mine = try fetch(gpa, u16, normed.view);
        defer gpa.free(mine);
        const theirs = try fetch(gpa, u16, want.view);
        defer gpa.free(theirs);
        try testing.expectEqualSlices(u16, theirs, mine);
    }
}

test "the tower's linear with the residual in its epilogue: the linear, then the add, to the bit" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1428);
    const random = prng.random();
    // SAM's lin2 and proj; rows that end inside a tile, and a few
    const Case = struct { m: usize, k: usize, n: usize };
    for ([_]Case{ .{ .m = 300, .k = 3072, .n = 768 }, .{ .m = 5, .k = 768, .n = 768 } }) |c| {
        const x = try gpa.alloc(u16, c.m * c.k);
        defer gpa.free(x);
        const w = try gpa.alloc(u16, c.n * c.k);
        defer gpa.free(w);
        const bias = try gpa.alloc(u16, c.n);
        defer gpa.free(bias);
        const h = try gpa.alloc(u16, c.m * c.n);
        defer gpa.free(h);
        random16(random, x, 1.0);
        random16(random, w, 0.03);
        random16(random, bias, 0.5);
        random16(random, h, 4.0);
        var xb = try upload(.bf16, &.{ c.m, c.k }, std.mem.sliceAsBytes(x));
        defer xb.deinit();
        var wb = try upload(.bf16, &.{ c.n, c.k }, std.mem.sliceAsBytes(w));
        defer wb.deinit();
        var bb = try upload(.bf16, &.{c.n}, std.mem.sliceAsBytes(bias));
        defer bb.deinit();
        var two = try upload(.bf16, &.{ c.m, c.n }, std.mem.sliceAsBytes(h));
        defer two.deinit();
        var one = try upload(.bf16, &.{ c.m, c.n }, std.mem.sliceAsBytes(h));
        defer one.deinit();
        var branch: cuda.Buffer = try .init(.bf16, &.{ c.m, c.n });
        defer branch.deinit();
        const linear: vw.Linear = .{ .weight = wb.view, .bias = bb.view };
        try linear.forward(xb.view, branch.view, .none);
        try glue.residualAdd(two.view, branch.view);
        try linear.residual(xb.view, one.view);
        const want = try fetch(gpa, u16, two.view);
        defer gpa.free(want);
        const got = try fetch(gpa, u16, one.view);
        defer gpa.free(got);
        try testing.expectEqualSlices(u16, want, got);
    }
}
