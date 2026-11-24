//! `glue.zig`'s tests: every launcher against `glue_ref.zig` on random tensors bit for bit, and a row alone
//! against the row among others. The kernels are TensorFold's; the host model is op for op theirs.

const std = @import("std");
const cuda = @import("../../../src/cuda/cuda.zig");
const gpu = @import("../../cuda/testing.zig");
const glue = @import("../../../src/models/nemotron/glue.zig");
const ref = @import("../../../src/models/nemotron/glue_ref.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;
const device = gpu.device;
const fetch = gpu.fetch;
const tobf = ref.f32ToBf16;

fn randbf16(gpa: Allocator, random: std.Random, n: usize, scale: f32) ![]u16 {
    const s = try gpa.alloc(u16, n);
    for (s) |*v| v.* = tobf(random.floatNorm(f32) * scale);
    return s;
}

fn randf32(gpa: Allocator, random: std.Random, n: usize, scale: f32) ![]f32 {
    const s = try gpa.alloc(f32, n);
    for (s) |*v| v.* = random.floatNorm(f32) * scale;
    return s;
}

test "add_rmsnorm h, y and xs are the host's bits, a row alone or among others" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(11);
    const random = prng.random();
    for ([_]usize{ 256, 2688 }) |d| {
        const rows = 5;
        const eps: f64 = 1e-5;
        const x = try randbf16(gpa, random, rows * d, 1.0);
        defer gpa.free(x);
        const r = try randbf16(gpa, random, rows * d, 1.0);
        defer gpa.free(r);
        const w = try randbf16(gpa, random, d, 1.0);
        defer gpa.free(w);

        var x_dev = try device(u16, .bf16, &.{ rows, d }, x);
        defer x_dev.deinit();
        var r_dev = try device(u16, .bf16, &.{ rows, d }, r);
        defer r_dev.deinit();
        var w_dev = try device(u16, .bf16, &.{d}, w);
        defer w_dev.deinit();
        var h_dev: cuda.Buffer = try .init(.bf16, &.{ rows, d });
        defer h_dev.deinit();
        var y_dev: cuda.Buffer = try .init(.bf16, &.{ rows, d });
        defer y_dev.deinit();
        var xs_dev: cuda.Buffer = try .init(.f32, &.{ rows, d / 64 });
        defer xs_dev.deinit();
        try glue.addRmsnorm(x_dev.view, r_dev.view, w_dev.view, h_dev.view, y_dev.view, xs_dev.view, eps);
        const h = try fetch(u16, gpa, h_dev);
        defer gpa.free(h);
        const y = try fetch(u16, gpa, y_dev);
        defer gpa.free(y);
        const xs = try fetch(f32, gpa, xs_dev);
        defer gpa.free(xs);

        const rh = try gpa.alloc(u16, rows * d);
        defer gpa.free(rh);
        const ry = try gpa.alloc(u16, rows * d);
        defer gpa.free(ry);
        const rxs = try gpa.alloc(f32, rows * (d / 64));
        defer gpa.free(rxs);
        try ref.addRmsnorm(gpa, x, r, w, rh, ry, rxs, rows, d, @floatCast(eps));
        try testing.expectEqualSlices(u16, rh, h);
        try testing.expectEqualSlices(u16, ry, y);
        try testing.expectEqualSlices(f32, rxs, xs);

        // a row alone on its own buffers is the row among the others
        var h1: cuda.Buffer = try .init(.bf16, &.{ 1, d });
        defer h1.deinit();
        var y1: cuda.Buffer = try .init(.bf16, &.{ 1, d });
        defer y1.deinit();
        var xs1: cuda.Buffer = try .init(.f32, &.{ 1, d / 64 });
        defer xs1.deinit();
        for (0..rows) |row| {
            try glue.addRmsnorm(x_dev.view.span(row, 1), r_dev.view.span(row, 1), w_dev.view, h1.view, y1.view, xs1.view, eps);
            const hs = try fetch(u16, gpa, h1);
            defer gpa.free(hs);
            const ys = try fetch(u16, gpa, y1);
            defer gpa.free(ys);
            const xss = try fetch(f32, gpa, xs1);
            defer gpa.free(xss);
            try testing.expectEqualSlices(u16, h[row * d ..][0..d], hs);
            try testing.expectEqualSlices(u16, y[row * d ..][0..d], ys);
            try testing.expectEqualSlices(f32, xs[row * (d / 64) ..][0 .. d / 64], xss);
        }
    }
}

test "add_rmsnorm without a residual is the plain norm" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(12);
    const random = prng.random();
    const rows = 3;
    const d = 512;
    const x = try randbf16(gpa, random, rows * d, 1.0);
    defer gpa.free(x);
    const w = try randbf16(gpa, random, d, 1.0);
    defer gpa.free(w);
    var x_dev = try device(u16, .bf16, &.{ rows, d }, x);
    defer x_dev.deinit();
    var w_dev = try device(u16, .bf16, &.{d}, w);
    defer w_dev.deinit();
    var h_dev: cuda.Buffer = try .init(.bf16, &.{ rows, d });
    defer h_dev.deinit();
    var y_dev: cuda.Buffer = try .init(.bf16, &.{ rows, d });
    defer y_dev.deinit();
    var xs_dev: cuda.Buffer = try .init(.f32, &.{ rows, d / 64 });
    defer xs_dev.deinit();
    try glue.addRmsnorm(x_dev.view, null, w_dev.view, h_dev.view, y_dev.view, xs_dev.view, 1e-5);
    const y = try fetch(u16, gpa, y_dev);
    defer gpa.free(y);
    const xs = try fetch(f32, gpa, xs_dev);
    defer gpa.free(xs);
    const ry = try gpa.alloc(u16, rows * d);
    defer gpa.free(ry);
    const rxs = try gpa.alloc(f32, rows * (d / 64));
    defer gpa.free(rxs);
    try ref.addRmsnorm(gpa, x, null, w, undefined, ry, rxs, rows, d, 1e-5);
    try testing.expectEqualSlices(u16, ry, y);
    try testing.expectEqualSlices(f32, rxs, xs);
}

test "add_moe_norm sums the routed slots by weight then the shared, fp32 and bf16 slots" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(13);
    const random = prng.random();
    const rows = 4;
    const d = 512;
    const nr = 6;
    const ns = 8;
    const x = try randbf16(gpa, random, rows * d, 1.0);
    defer gpa.free(x);
    const w = try randbf16(gpa, random, d, 1.0);
    defer gpa.free(w);
    const wt = try randf32(gpa, random, rows * ns, 0.5);
    defer gpa.free(wt);
    var x_dev = try device(u16, .bf16, &.{ rows, d }, x);
    defer x_dev.deinit();
    var w_dev = try device(u16, .bf16, &.{d}, w);
    defer w_dev.deinit();
    var wt_dev = try device(f32, .f32, &.{ rows, ns }, wt);
    defer wt_dev.deinit();

    const slots32 = try randf32(gpa, random, rows * ns * d, 0.5);
    defer gpa.free(slots32);
    const slots16 = try randbf16(gpa, random, rows * ns * d, 0.5);
    defer gpa.free(slots16);
    for ([_]bool{ true, false }) |f32_slots| {
        var yk_dev: cuda.Buffer = try .init(if (f32_slots) .f32 else .bf16, &.{ rows, ns, d });
        defer yk_dev.deinit();
        try yk_dev.upload(if (f32_slots) std.mem.sliceAsBytes(slots32) else std.mem.sliceAsBytes(slots16));
        var hn_dev: cuda.Buffer = try .init(.bf16, &.{ rows, d });
        defer hn_dev.deinit();
        var y_dev: cuda.Buffer = try .init(.bf16, &.{ rows, d });
        defer y_dev.deinit();
        var xs_dev: cuda.Buffer = try .init(.f32, &.{ rows, d / 64 });
        defer xs_dev.deinit();
        try glue.addMoeNorm(x_dev.view, yk_dev.view, wt_dev.view, w_dev.view, hn_dev.view, y_dev.view, xs_dev.view, 1e-5, nr);
        const hn = try fetch(u16, gpa, hn_dev);
        defer gpa.free(hn);
        const y = try fetch(u16, gpa, y_dev);
        defer gpa.free(y);
        const xs = try fetch(f32, gpa, xs_dev);
        defer gpa.free(xs);

        const rhn = try gpa.alloc(u16, rows * d);
        defer gpa.free(rhn);
        const ry = try gpa.alloc(u16, rows * d);
        defer gpa.free(ry);
        const rxs = try gpa.alloc(f32, rows * (d / 64));
        defer gpa.free(rxs);
        if (f32_slots) {
            try ref.addMoeNorm(gpa, x, slots32, null, wt, w, rhn, ry, rxs, rows, d, 1e-5, nr, ns);
        } else {
            try ref.addMoeNorm(gpa, x, null, slots16, wt, w, rhn, ry, rxs, rows, d, 1e-5, nr, ns);
        }
        try testing.expectEqualSlices(u16, rhn, hn);
        try testing.expectEqualSlices(u16, ry, y);
        try testing.expectEqualSlices(f32, rxs, xs);
    }
}

test "concat_norms norms the embedding and hidden halves side by side" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(14);
    const random = prng.random();
    const rows = 3;
    const d = 256;
    const e = try randbf16(gpa, random, rows * d, 1.0);
    defer gpa.free(e);
    const hd = try randbf16(gpa, random, rows * d, 1.0);
    defer gpa.free(hd);
    const we = try randbf16(gpa, random, d, 1.0);
    defer gpa.free(we);
    const wh = try randbf16(gpa, random, d, 1.0);
    defer gpa.free(wh);
    var e_dev = try device(u16, .bf16, &.{ rows, d }, e);
    defer e_dev.deinit();
    var hd_dev = try device(u16, .bf16, &.{ rows, d }, hd);
    defer hd_dev.deinit();
    var we_dev = try device(u16, .bf16, &.{d}, we);
    defer we_dev.deinit();
    var wh_dev = try device(u16, .bf16, &.{d}, wh);
    defer wh_dev.deinit();
    var out_dev: cuda.Buffer = try .init(.bf16, &.{ rows, 2 * d });
    defer out_dev.deinit();
    var xs_dev: cuda.Buffer = try .init(.f32, &.{ rows, 2 * d / 64 });
    defer xs_dev.deinit();
    try glue.concatNorms(e_dev.view, hd_dev.view, we_dev.view, wh_dev.view, out_dev.view, xs_dev.view, 1e-5);
    const out = try fetch(u16, gpa, out_dev);
    defer gpa.free(out);
    const xs = try fetch(f32, gpa, xs_dev);
    defer gpa.free(xs);
    const rout = try gpa.alloc(u16, rows * 2 * d);
    defer gpa.free(rout);
    const rxs = try gpa.alloc(f32, rows * (2 * d / 64));
    defer gpa.free(rxs);
    try ref.concatNorms(gpa, e, hd, we, wh, rout, rxs, rows, d, 1e-5);
    try testing.expectEqualSlices(u16, rout, out);
    try testing.expectEqualSlices(f32, rxs, xs);
}

test "group_rmsnorm norms each group on its own, a row alone among others" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(15);
    const random = prng.random();
    for ([_]struct { xd: usize, groups: usize }{ .{ .xd = 512, .groups = 8 }, .{ .xd = 4096, .groups = 8 } }) |cfg| {
        const rows = 3;
        const x = try randbf16(gpa, random, rows * cfg.xd, 1.0);
        defer gpa.free(x);
        const w = try randbf16(gpa, random, cfg.xd, 1.0);
        defer gpa.free(w);
        var x_dev = try device(u16, .bf16, &.{ rows, cfg.xd }, x);
        defer x_dev.deinit();
        var w_dev = try device(u16, .bf16, &.{cfg.xd}, w);
        defer w_dev.deinit();
        var out_dev: cuda.Buffer = try .init(.bf16, &.{ rows, cfg.xd });
        defer out_dev.deinit();
        var xs_dev: cuda.Buffer = try .init(.f32, &.{ rows, cfg.xd / 64 });
        defer xs_dev.deinit();
        try glue.groupRmsnorm(x_dev.view, w_dev.view, out_dev.view, xs_dev.view, 1e-5, cfg.groups);
        const out = try fetch(u16, gpa, out_dev);
        defer gpa.free(out);
        const xs = try fetch(f32, gpa, xs_dev);
        defer gpa.free(xs);
        const rout = try gpa.alloc(u16, rows * cfg.xd);
        defer gpa.free(rout);
        const rxs = try gpa.alloc(f32, rows * (cfg.xd / 64));
        defer gpa.free(rxs);
        try ref.groupRmsnorm(gpa, x, w, rout, rxs, rows, cfg.xd, cfg.groups, 1e-5);
        try testing.expectEqualSlices(u16, rout, out);
        try testing.expectEqualSlices(f32, rxs, xs);

        var out1: cuda.Buffer = try .init(.bf16, &.{ 1, cfg.xd });
        defer out1.deinit();
        var xs1: cuda.Buffer = try .init(.f32, &.{ 1, cfg.xd / 64 });
        defer xs1.deinit();
        for (0..rows) |row| {
            try glue.groupRmsnorm(x_dev.view.span(row, 1), w_dev.view, out1.view, xs1.view, 1e-5, cfg.groups);
            const os = try fetch(u16, gpa, out1);
            defer gpa.free(os);
            try testing.expectEqualSlices(u16, out[row * cfg.xd ..][0..cfg.xd], os);
        }
    }
}

test "router and topk: sliced logits, picks by score and bias, ties to the lower id" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(16);
    const random = prng.random();
    const rows = 5;
    const d = 2688;
    const e = 128;
    const top = 6;
    const ns = 8;
    const sk = glue.routerSlices(d);
    try testing.expectEqual(@as(usize, 6), sk);
    const x = try randbf16(gpa, random, rows * d, 1.0);
    defer gpa.free(x);
    const w = try randbf16(gpa, random, e * d, 1.0);
    defer gpa.free(w);
    // experts 10 and 42 route identically and share a bias: a tie the lower id wins
    @memcpy(w[42 * d ..][0..d], w[10 * d ..][0..d]);
    const bias = try randf32(gpa, random, e, 0.01);
    defer gpa.free(bias);
    bias[42] = bias[10];
    var x_dev = try device(u16, .bf16, &.{ rows, d }, x);
    defer x_dev.deinit();
    var w_dev = try device(u16, .bf16, &.{ e, d }, w);
    defer w_dev.deinit();
    var bias_dev = try device(f32, .f32, &.{e}, bias);
    defer bias_dev.deinit();
    var part_dev: cuda.Buffer = try .init(.f32, &.{ sk, rows, e });
    defer part_dev.deinit();
    var idx_dev: cuda.Buffer = try .init(.i32, &.{ rows, ns });
    defer idx_dev.deinit();
    var wt_dev: cuda.Buffer = try .init(.f32, &.{ rows, ns });
    defer wt_dev.deinit();
    try glue.route(x_dev.view, w_dev.view, bias_dev.view, part_dev.view, idx_dev.view, wt_dev.view, 2.5, top, ns, true);
    const part = try fetch(f32, gpa, part_dev);
    defer gpa.free(part);
    const idx = try fetch(i32, gpa, idx_dev);
    defer gpa.free(idx);
    const wt = try fetch(f32, gpa, wt_dev);
    defer gpa.free(wt);

    const rpart = try gpa.alloc(f32, sk * rows * e);
    defer gpa.free(rpart);
    ref.router(x, w, rpart, rows, d, e, sk);
    try testing.expectEqualSlices(f32, rpart, part);
    const ridx = try gpa.alloc(i32, rows * ns);
    defer gpa.free(ridx);
    const rwt = try gpa.alloc(f32, rows * ns);
    defer gpa.free(rwt);
    ref.topk(rpart, bias, ridx, rwt, rows, 2.5, e, sk, top, ns, true);
    try testing.expectEqualSlices(i32, ridx, idx);
    try testing.expectEqualSlices(f32, rwt, wt);
    // the crafted tie: wherever 42 is picked, 10 came first
    for (0..rows) |r| {
        const picks = idx[r * ns ..][0..ns];
        if (std.mem.indexOfScalar(i32, picks, 42)) |at| {
            const first = std.mem.indexOfScalar(i32, picks, 10).?;
            try testing.expect(first < at);
        }
        try testing.expectEqualSlices(i32, &.{ e, e + 1 }, picks[top..]);
        try testing.expectEqualSlices(f32, &.{ 1, 1 }, wt[r * ns + top ..][0..2]);
    }

    // a row alone: its slice of the logits, picks and weights are the row among others'
    var part1: cuda.Buffer = try .init(.f32, &.{ sk, 1, e });
    defer part1.deinit();
    var idx1: cuda.Buffer = try .init(.i32, &.{ 1, ns });
    defer idx1.deinit();
    var wt1: cuda.Buffer = try .init(.f32, &.{ 1, ns });
    defer wt1.deinit();
    for (0..rows) |r| {
        try glue.route(x_dev.view.span(r, 1), w_dev.view, bias_dev.view, part1.view, idx1.view, wt1.view, 2.5, top, ns, true);
        const ps = try fetch(f32, gpa, part1);
        defer gpa.free(ps);
        const is = try fetch(i32, gpa, idx1);
        defer gpa.free(is);
        const ws = try fetch(f32, gpa, wt1);
        defer gpa.free(ws);
        for (0..sk) |s| try testing.expectEqualSlices(f32, part[(s * rows + r) * e ..][0..e], ps[s * e ..][0..e]);
        try testing.expectEqualSlices(i32, idx[r * ns ..][0..ns], is);
        try testing.expectEqualSlices(f32, wt[r * ns ..][0..ns], ws);
    }
}

test "keyed greedy is the host's pick and share, ties included" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(17);
    const random = prng.random();
    const rows = 4;
    const count = 28;
    const vals = try randf32(gpa, random, rows * count, 2.0);
    defer gpa.free(vals);
    const ids = try gpa.alloc(i64, rows * count);
    defer gpa.free(ids);
    // ids above 2^16 but within the i32 the kernel returns (a vocabulary's ids are far below it)
    for (ids, 0..) |*v, i| v.* = @intCast((i * 2654435761) % 100003 + (i % 3) * (1 << 29));
    // ties: rows 1 and 3 repeat their best value under other ids
    vals[1 * count + 9] = vals[1 * count + 3];
    vals[3 * count + 27] = vals[3 * count + 0];
    var vals_dev = try device(f32, .f32, &.{ rows, count }, vals);
    defer vals_dev.deinit();
    var ids_dev = try device(i64, .i64, &.{ rows, count }, ids);
    defer ids_dev.deinit();
    var out_dev: cuda.Buffer = try .init(.i32, &.{rows});
    defer out_dev.deinit();
    var prob_dev: cuda.Buffer = try .init(.f32, &.{rows});
    defer prob_dev.deinit();
    try glue.keyedGreedy(vals_dev.view, ids_dev.view, out_dev.view, prob_dev.view, 20);
    const out = try fetch(i32, gpa, out_dev);
    defer gpa.free(out);
    const prob = try fetch(f32, gpa, prob_dev);
    defer gpa.free(prob);
    const rout = try gpa.alloc(i32, rows);
    defer gpa.free(rout);
    const rprob = try gpa.alloc(f32, rows);
    defer gpa.free(rprob);
    ref.keyedGreedy(vals, ids, rout, rprob, rows, count, 20);
    try testing.expectEqualSlices(i32, rout, out);
    // fp64 exp on the device may sit an ulp off the host's; the fp32 share agrees within 1e-6
    for (0..rows) |r| try testing.expectApproxEqRel(rprob[r], prob[r], 1e-6);

    var out1: cuda.Buffer = try .init(.i32, &.{1});
    defer out1.deinit();
    var prob1: cuda.Buffer = try .init(.f32, &.{1});
    defer prob1.deinit();
    for (0..rows) |r| {
        try glue.keyedGreedy(vals_dev.view.span(r, 1), ids_dev.view.span(r, 1), out1.view, prob1.view, 20);
        const os = try fetch(i32, gpa, out1);
        defer gpa.free(os);
        try testing.expectEqual(out[r], os[0]);
    }
}

test "serial feed writes the token and advances the meta on the device" {
    const gpa = testing.allocator;
    for ([_]i32{ 0, 1 }) |parity| {
        const sampled: i32 = 1042;
        var meta: [3]i32 = .{ 5, parity, 0 };
        var history: [16]i32 = @splat(-7);
        var sampled_dev = try device(i32, .i32, &.{1}, &.{sampled});
        defer sampled_dev.deinit();
        var ids_dev: cuda.Buffer = try .init(.i32, &.{1});
        defer ids_dev.deinit();
        var meta_dev = try device(i32, .i32, &.{3}, &meta);
        defer meta_dev.deinit();
        var history_dev = try device(i32, .i32, &.{16}, &history);
        defer history_dev.deinit();
        try glue.serialFeed(sampled_dev.view, ids_dev.view, meta_dev.view, history_dev.view);
        const ids = try fetch(i32, gpa, ids_dev);
        defer gpa.free(ids);
        const metas = try fetch(i32, gpa, meta_dev);
        defer gpa.free(metas);
        const hist = try fetch(i32, gpa, history_dev);
        defer gpa.free(hist);
        var rid: [1]i32 = .{-1};
        ref.serialFeed(&.{sampled}, &rid, &meta, &history);
        try testing.expectEqualSlices(i32, &rid, ids);
        try testing.expectEqualSlices(i32, &meta, metas);
        try testing.expectEqualSlices(i32, &history, hist);
    }
}

test "plan routed's members, items and counts are the host's, and launch after launch alike" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(18);
    const random = prng.random();
    for ([_]struct { rows: usize, tile: usize }{ .{ .rows = 8, .tile = 16 }, .{ .rows = 16, .tile = 64 } }) |cfg| {
        const slots = 8;
        const routed = 6;
        const e = 128;
        const picks = try gpa.alloc(i32, cfg.rows * slots);
        defer gpa.free(picks);
        for (picks) |*v| v.* = @intCast(random.uintLessThan(usize, e));
        var picks_dev = try device(i32, .i32, &.{ cfg.rows, slots }, picks);
        defer picks_dev.deinit();
        var members_dev: cuda.Buffer = try .init(.i32, &.{cfg.rows * routed});
        defer members_dev.deinit();
        var items_dev: cuda.Buffer = try .init(.i32, &.{3 * (e + cfg.rows * slots)});
        defer items_dev.deinit();
        var counts_dev: cuda.Buffer = try .init(.i32, &.{2});
        defer counts_dev.deinit();
        try glue.planRouted(picks_dev.view, members_dev.view, items_dev.view, counts_dev.view, cfg.rows, slots, routed, e, cfg.tile);
        const counts = try fetch(i32, gpa, counts_dev);
        defer gpa.free(counts);
        const members = try fetch(i32, gpa, members_dev);
        defer gpa.free(members);
        const items = try fetch(i32, gpa, items_dev);
        defer gpa.free(items);
        const nitems: usize = @intCast(counts[0]);
        const nmembers: usize = cfg.rows * routed;
        // the plan writes only its first counts[0] items and the members its experts hold
        const rmembers = try gpa.alloc(i32, nmembers);
        defer gpa.free(rmembers);
        const ritems = try gpa.alloc(i32, items.len);
        defer gpa.free(ritems);
        var rcounts: [2]i32 = undefined;
        ref.planRouted(picks, rmembers, ritems, &rcounts, cfg.rows, slots, routed, e, cfg.tile);
        try testing.expectEqualSlices(i32, &rcounts, counts);
        try testing.expectEqualSlices(i32, ritems[0 .. nitems * 3], items[0 .. nitems * 3]);
        // members are compact per expert: only the first sum over items' counts is written
        var used: usize = 0;
        for (0..nitems) |i| used += @intCast(items[i * 3 + 2]);
        try testing.expectEqualSlices(i32, rmembers[0..used], members[0..used]);

        var members2_dev: cuda.Buffer = try .init(.i32, &.{cfg.rows * routed});
        defer members2_dev.deinit();
        var items2_dev: cuda.Buffer = try .init(.i32, &.{3 * (e + cfg.rows * slots)});
        defer items2_dev.deinit();
        var counts2_dev: cuda.Buffer = try .init(.i32, &.{2});
        defer counts2_dev.deinit();
        try glue.planRouted(picks_dev.view, members2_dev.view, items2_dev.view, counts2_dev.view, cfg.rows, slots, routed, e, cfg.tile);
        const counts2 = try fetch(i32, gpa, counts2_dev);
        defer gpa.free(counts2);
        const members2 = try fetch(i32, gpa, members2_dev);
        defer gpa.free(members2);
        const items2 = try fetch(i32, gpa, items2_dev);
        defer gpa.free(items2);
        try testing.expectEqualSlices(i32, counts, counts2);
        try testing.expectEqualSlices(i32, members, members2);
        try testing.expectEqualSlices(i32, items[0 .. nitems * 3], items2[0 .. nitems * 3]);
    }
}

test "rest rows adds r x at the mapped rows, fp32 and bf16, a row alone or among others" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(19);
    const random = prng.random();
    const rows = 4;
    const n = 64;
    const k = 512;
    const x = try randbf16(gpa, random, rows * k, 1.0);
    defer gpa.free(x);
    const r = try randbf16(gpa, random, n * k, 1.0);
    defer gpa.free(r);
    var x_dev = try device(u16, .bf16, &.{ rows, k }, x);
    defer x_dev.deinit();
    var r_dev = try device(u16, .bf16, &.{ n, k }, r);
    defer r_dev.deinit();

    inline for (.{ true, false }) |f32_y| {
        const init32 = try randf32(gpa, random, rows * n, 1.0);
        defer gpa.free(init32);
        const init16 = try gpa.alloc(u16, rows * n);
        defer gpa.free(init16);
        for (init32, init16) |v, *b| b.* = tobf(v);
        var y_dev: cuda.Buffer = try .init(if (f32_y) .f32 else .bf16, &.{ rows, n });
        defer y_dev.deinit();
        try y_dev.upload(if (f32_y) std.mem.sliceAsBytes(init32) else std.mem.sliceAsBytes(init16));
        try glue.restRows(x_dev.view, r_dev.view, y_dev.view, rows, 1, 0);
        const got32 = if (f32_y) try fetch(f32, gpa, y_dev) else &.{};
        defer if (f32_y) gpa.free(got32);
        const got16 = if (f32_y) &.{} else try fetch(u16, gpa, y_dev);
        defer if (!f32_y) gpa.free(got16);

        const ry32 = try gpa.dupe(f32, init32);
        defer gpa.free(ry32);
        const ry16 = try gpa.dupe(u16, init16);
        defer gpa.free(ry16);
        if (f32_y) {
            ref.restRows(x, k, r, ry32, null, n, rows, 1, 0, n, k);
            try testing.expectEqualSlices(f32, ry32, got32);
        } else {
            ref.restRows(x, k, r, null, ry16, n, rows, 1, 0, n, k);
            try testing.expectEqualSlices(u16, ry16, got16);
        }

        // a row alone: its y update is its row of the batch's
        for (0..rows) |row| {
            var y1: cuda.Buffer = try .init(if (f32_y) .f32 else .bf16, &.{ 1, n });
            defer y1.deinit();
            try y1.upload(if (f32_y) std.mem.sliceAsBytes(init32[row * n ..][0..n]) else std.mem.sliceAsBytes(init16[row * n ..][0..n]));
            try glue.restRows(x_dev.view.span(row, 1), r_dev.view, y1.view, 1, 1, 0);
            if (f32_y) {
                const ys = try fetch(f32, gpa, y1);
                defer gpa.free(ys);
                try testing.expectEqualSlices(f32, got32[row * n ..][0..n], ys);
            } else {
                const ys = try fetch(u16, gpa, y1);
                defer gpa.free(ys);
                try testing.expectEqualSlices(u16, got16[row * n ..][0..n], ys);
            }
        }
    }
}
