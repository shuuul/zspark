//! `moe.zig`'s tests: the selection, SwiGLU and a rank's share against `glue_ref.zig`, rows built to tie, and a
//! row alone against the row among others.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const host = @import("../../../../src/models/glm5/layers/glue_ref.zig");
const glue_ref = @import("../../../../src/decode/glue_ref.zig");
const moe = @import("../../../../src/models/glm5/layers/moe.zig");
const shared = @import("../testing.zig");
const device = shared.device;
const fetch = shared.fetch;

/// GLM's router: 288 experts, 8 a token, the routed weights times 2.5.
const fx = struct {
    const experts = 288;
    const top = 8;
    const slots = top + 1;
    const scale = 2.5;
};

test "the selection is the host's picks and weights, from logits or from the router's slices, a row alone too" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(5);
    const random = prng.random();
    const rows = 7;
    const sk = 3;
    const part = try gpa.alloc(f32, sk * rows * fx.experts);
    defer gpa.free(part);
    shared.randomF32(random, part, 1.2);
    const bias = try gpa.alloc(f32, fx.experts);
    defer gpa.free(bias);
    shared.randomF32(random, bias, 0.2);
    const logits = try gpa.alloc(f32, rows * fx.experts);
    defer gpa.free(logits);
    for (logits, 0..) |*l, i| {
        l.* = part[i];
        for (1..sk) |s| l.* += part[s * rows * fx.experts + i];
    }

    var part_dev = try device(f32, .f32, &.{ sk, rows, fx.experts }, part);
    defer part_dev.deinit();
    var bias_dev = try device(f32, .f32, &.{fx.experts}, bias);
    defer bias_dev.deinit();
    var logits_dev: cuda.Buffer = try .init(.f32, &.{ rows, fx.experts });
    defer logits_dev.deinit();
    var picks: cuda.Buffer = try .init(.i32, &.{ rows, fx.slots });
    defer picks.deinit();
    var weights: cuda.Buffer = try .init(.f32, &.{ rows, fx.slots });
    defer weights.deinit();

    for ([_]bool{ true, false }) |norm| {
        try moe.select(logits_dev.view, .of(part_dev.view, sk, rows, fx.experts), bias_dev.view, picks.view, weights.view, fx.top, fx.scale, norm);
        const summed = try fetch(u32, gpa, logits_dev);
        defer gpa.free(summed);
        try testing.expectEqualSlices(u32, @ptrCast(logits), summed);
        const got_picks = try fetch(i32, gpa, picks);
        defer gpa.free(got_picks);
        const got_weights = try fetch(f32, gpa, weights);
        defer gpa.free(got_weights);
        var want_picks: [fx.slots]i32 = undefined;
        var want_weights: [fx.slots]f32 = undefined;
        for (0..rows) |r| {
            host.select(logits[r * fx.experts ..][0..fx.experts], bias, &want_picks, &want_weights, fx.scale, norm);
            try testing.expectEqualSlices(i32, &want_picks, got_picks[r * fx.slots ..][0..fx.slots]);
            for (want_weights, got_weights[r * fx.slots ..][0..fx.slots]) |w, g| try testing.expectApproxEqRel(w, g, 1e-5);
            try testing.expectEqual(@as(f32, 1), got_weights[r * fx.slots + fx.top]);
        }

        // from the logits as they are, and a row alone: the same bits
        var picks1: cuda.Buffer = try .init(.i32, &.{ 1, fx.slots });
        defer picks1.deinit();
        var weights1: cuda.Buffer = try .init(.f32, &.{ 1, fx.slots });
        defer weights1.deinit();
        for (0..rows) |r| {
            try moe.select(logits_dev.span(r, 1), null, bias_dev.view, picks1.view, weights1.view, fx.top, fx.scale, norm);
            const one_picks = try fetch(i32, gpa, picks1);
            defer gpa.free(one_picks);
            try testing.expectEqualSlices(i32, got_picks[r * fx.slots ..][0..fx.slots], one_picks);
            const one_weights = try fetch(u32, gpa, weights1);
            defer gpa.free(one_weights);
            try testing.expectEqualSlices(u32, @ptrCast(got_weights[r * fx.slots ..][0..fx.slots]), one_weights);
        }
    }
}

test "equal scores go to the lower id, across warps and within one, and a bias decides before the logit" {
    const gpa = testing.allocator;
    const rows = 4;
    const logits = try gpa.alloc(f32, rows * fx.experts);
    defer gpa.free(logits);
    const bias = try gpa.alloc(f32, fx.experts);
    defer gpa.free(bias);
    // the bias lifts three ids, two of them equally: under equal logits those first, the lower id before
    @memset(bias, 0);
    bias[270] = 0.5;
    bias[130] = 0.5;
    bias[31] = 0.25;
    // row 0: every logit equal, so the lifted three, then the lowest ids
    @memset(logits[0..fx.experts], 0.25);
    // row 1: twelve equal at the top, spread over warps, among lower ones
    @memset(logits[fx.experts..][0..fx.experts], -3);
    const tied = [_]usize{ 287, 256, 255, 224, 100, 97, 96, 64, 63, 33, 32, 5 };
    for (tied) |e| logits[fx.experts + e] = 1.5;
    // row 2: nine in one warp equal, one above them in another
    @memset(logits[2 * fx.experts ..][0..fx.experts], -1);
    for (40..49) |e| logits[2 * fx.experts + e] = 2;
    logits[2 * fx.experts + 200] = 4;
    // row 3: a high logit (a score under 1) stays behind the two the bias lifts to 1, ahead of the third
    @memset(logits[3 * fx.experts ..][0..fx.experts], 0);
    logits[3 * fx.experts + 7] = 6;

    var logits_dev = try device(f32, .f32, &.{ rows, fx.experts }, logits);
    defer logits_dev.deinit();
    var bias_dev = try device(f32, .f32, &.{fx.experts}, bias);
    defer bias_dev.deinit();
    var picks: cuda.Buffer = try .init(.i32, &.{ rows, fx.slots });
    defer picks.deinit();
    var weights: cuda.Buffer = try .init(.f32, &.{ rows, fx.slots });
    defer weights.deinit();
    try moe.select(logits_dev.view, null, bias_dev.view, picks.view, weights.view, fx.top, fx.scale, true);
    const got = try fetch(i32, gpa, picks);
    defer gpa.free(got);
    const got_weights = try fetch(f32, gpa, weights);
    defer gpa.free(got_weights);
    try testing.expectEqualSlices(i32, &.{ 130, 270, 31, 0, 1, 2, 3, 4, fx.experts }, got[0..fx.slots]);
    try testing.expectEqualSlices(i32, &.{ 5, 32, 33, 63, 64, 96, 97, 100, fx.experts }, got[fx.slots..][0..fx.slots]);
    try testing.expectEqualSlices(i32, &.{ 200, 40, 41, 42, 43, 44, 45, 46, fx.experts }, got[2 * fx.slots ..][0..fx.slots]);
    try testing.expectEqualSlices(i32, &.{ 130, 270, 7, 31, 0, 1, 2, 3, fx.experts }, got[3 * fx.slots ..][0..fx.slots]);
    // equal scores share the weight: an eighth of the scale each, whatever the bias chose
    for ([_]usize{ 0, 1 }) |r| for (got_weights[r * fx.slots ..][0..fx.top]) |w| try testing.expectApproxEqRel(@as(f32, fx.scale / 8.0), w, 1e-6);
    var want_picks: [fx.slots]i32 = undefined;
    var want_weights: [fx.slots]f32 = undefined;
    for (0..rows) |r| {
        host.select(logits[r * fx.experts ..][0..fx.experts], bias, &want_picks, &want_weights, fx.scale, true);
        try testing.expectEqualSlices(i32, &want_picks, got[r * fx.slots ..][0..fx.slots]);
    }
}

test "the selection with a NaN among a row's logits picks experts that exist, and the rows beside it keep their bits" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(81);
    const random = prng.random();
    const rows = 4;
    const logits = try gpa.alloc(f32, rows * fx.experts);
    defer gpa.free(logits);
    shared.randomF32(random, logits, 2.0);
    const bias = try gpa.alloc(f32, fx.experts);
    defer gpa.free(bias);
    shared.randomF32(random, bias, 0.2);
    const nan = std.math.nan(f32);
    // row 1: a few NaNs, one in the first warp; row 2: every logit; rows 0 and 3 have none
    for ([_]usize{ 3, 40, 41, 200, 287 }) |e| logits[1 * fx.experts + e] = nan;
    @memset(logits[2 * fx.experts ..][0..fx.experts], nan);

    var bias_dev = try device(f32, .f32, &.{fx.experts}, bias);
    defer bias_dev.deinit();
    var picks: cuda.Buffer = try .init(.i32, &.{ rows, fx.slots });
    defer picks.deinit();
    var weights: cuda.Buffer = try .init(.f32, &.{ rows, fx.slots });
    defer weights.deinit();
    // The launch is repeated over other launches' shared memory: what a place never written would show.
    for (0..8) |_| {
        var logits_dev = try device(f32, .f32, &.{ rows, fx.experts }, logits);
        defer logits_dev.deinit();
        try moe.select(logits_dev.view, null, bias_dev.view, picks.view, weights.view, fx.top, fx.scale, true);
        const got = try fetch(i32, gpa, picks);
        defer gpa.free(got);
        for (0..rows) |r| {
            for (got[r * fx.slots ..][0..fx.top]) |e| try testing.expect(e >= 0 and e < fx.experts);
            try testing.expectEqual(@as(i32, fx.experts), got[r * fx.slots + fx.top]);
        }
        // every candidate of the all-NaN row counts place 0: the places after it are what they started as
        try testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4, 5, 6, 7 }, got[2 * fx.slots + 1 ..][0 .. fx.top - 1]);
    }
    const got_picks = try fetch(i32, gpa, picks);
    defer gpa.free(got_picks);
    const got_weights = try fetch(u32, gpa, weights);
    defer gpa.free(got_weights);
    for ([_]usize{ 0, 3 }) |r| {
        var one_logits = try device(f32, .f32, &.{ 1, fx.experts }, logits[r * fx.experts ..][0..fx.experts]);
        defer one_logits.deinit();
        var one_picks: cuda.Buffer = try .init(.i32, &.{ 1, fx.slots });
        defer one_picks.deinit();
        var one_weights: cuda.Buffer = try .init(.f32, &.{ 1, fx.slots });
        defer one_weights.deinit();
        try moe.select(one_logits.view, null, bias_dev.view, one_picks.view, one_weights.view, fx.top, fx.scale, true);
        const alone = try fetch(i32, gpa, one_picks);
        defer gpa.free(alone);
        const alone_weights = try fetch(u32, gpa, one_weights);
        defer gpa.free(alone_weights);
        try testing.expectEqualSlices(i32, alone, got_picks[r * fx.slots ..][0..fx.slots]);
        try testing.expectEqualSlices(u32, alone_weights, got_weights[r * fx.slots ..][0..fx.slots]);
    }
}

test "SwiGLU is the host's within a rounding and holds to its limit; a share is the host's bits; rows alone too" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(9);
    const random = prng.random();
    const rows = 5;
    const width = 1024;
    const limit = 10.0;
    const gu = try gpa.alloc(u16, rows * 2 * width);
    defer gpa.free(gu);
    // a spread that passes the limit on both sides
    shared.randomBf16(random, gu, 6.0);
    var gu_dev = try device(u16, .bf16, &.{ rows, 2 * width }, gu);
    defer gu_dev.deinit();
    var act: cuda.Buffer = try .init(.bf16, &.{ rows, width });
    defer act.deinit();
    try moe.swiglu(gu_dev.view, act.view, limit);
    const got = try fetch(u16, gpa, act);
    defer gpa.free(got);
    const want = try gpa.alloc(u16, width);
    defer gpa.free(want);
    var act1: cuda.Buffer = try .init(.bf16, &.{ 1, width });
    defer act1.deinit();
    var limited: usize = 0;
    for (0..rows) |r| {
        glue_ref.swiglu(gu[r * 2 * width ..][0 .. 2 * width], want, limit);
        for (want, got[r * width ..][0..width], gu[r * 2 * width ..][0..width], gu[r * 2 * width + width ..][0..width]) |w, g, gate, up| {
            try shared.expectClose(w, g);
            try testing.expect(@abs(numeric.f32FromBf16(g)) <= limit * limit);
            limited += @intFromBool(numeric.f32FromBf16(gate) > limit or @abs(numeric.f32FromBf16(up)) > limit);
        }
        try moe.swiglu(gu_dev.span(r, 1), act1.view, limit);
        const one = try fetch(u16, gpa, act1);
        defer gpa.free(one);
        try testing.expectEqualSlices(u16, got[r * width ..][0..width], one);
    }
    try testing.expect(limited > 100);

    const dims = 512;
    const y = try gpa.alloc(f32, rows * fx.slots * dims);
    defer gpa.free(y);
    shared.randomF32(random, y, 2.0);
    const weights = try gpa.alloc(f32, rows * fx.slots);
    defer gpa.free(weights);
    for (weights) |*w| w.* = random.float(f32);
    var y_dev = try device(f32, .f32, &.{ rows * fx.slots, dims }, y);
    defer y_dev.deinit();
    var weights_dev = try device(f32, .f32, &.{ rows, fx.slots }, weights);
    defer weights_dev.deinit();
    var out: cuda.Buffer = try .init(.f32, &.{ rows, dims });
    defer out.deinit();
    try moe.combine(y_dev.view, null, weights_dev.view, out.view);
    const share = try fetch(u32, gpa, out);
    defer gpa.free(share);
    const want_share = try gpa.alloc(f32, dims);
    defer gpa.free(want_share);
    var out1: cuda.Buffer = try .init(.f32, &.{ 1, dims });
    defer out1.deinit();
    for (0..rows) |r| {
        glue_ref.combine(y[r * fx.slots * dims ..][0 .. fx.slots * dims], weights[r * fx.slots ..][0..fx.slots], want_share);
        try testing.expectEqualSlices(u32, @ptrCast(want_share), share[r * dims ..][0..dims]);
        try moe.combine(y_dev.span(r * fx.slots, fx.slots), null, weights_dev.span(r, 1), out1.view);
        const one = try fetch(u32, gpa, out1);
        defer gpa.free(one);
        try testing.expectEqualSlices(u32, share[r * dims ..][0..dims], one);
    }

    // the shared expert's rows in a buffer of their own: the same bits, and y's last slot is not read
    const last = try gpa.alloc(f32, rows * dims);
    defer gpa.free(last);
    for (0..rows) |r| {
        const slot = y[(r * fx.slots + fx.slots - 1) * dims ..][0..dims];
        @memcpy(last[r * dims ..][0..dims], slot);
        @memset(slot, 1e30);
    }
    try y_dev.upload(std.mem.sliceAsBytes(y));
    var last_dev = try device(f32, .f32, &.{ rows, dims }, last);
    defer last_dev.deinit();
    try moe.combine(y_dev.view, last_dev.view, weights_dev.view, out.view);
    const apart = try fetch(u32, gpa, out);
    defer gpa.free(apart);
    try testing.expectEqualSlices(u32, share, apart);

    // a bf16 out is the fp32 sum rounded once (spec 116 WS-04: a quantized window's partials)
    var half: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
    defer half.deinit();
    try moe.combine(y_dev.view, last_dev.view, weights_dev.view, half.view);
    const halves = try fetch(u16, gpa, half);
    defer gpa.free(halves);
    for (share, halves) |wide, narrow| try testing.expectEqual(numeric.bf16FromF32(@bitCast(wide)), narrow);
}
