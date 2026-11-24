//! `moe.zig`'s tests: the selection against the host with ties, and the whole MoE against the host and a row alone.

const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const experts = @import("../../../../src/weights/experts.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const moe = @import("../../../../src/models/qwen4/blocks/moe.zig");
const gpu = @import("../../../cuda/testing.zig");
const bf = numeric.roundBf16;
const fetch = gpu.fetch;
const Allocator = std.mem.Allocator;
const Buffers = moe.Buffers;
const Weights = moe.Weights;
const forward = moe.forward;
const zs_qwen4_moe_topk = moe.zs_qwen4_moe_topk;

const testing = std.testing;

test "top-k takes the largest logits, the lower id among equals, and the shared expert last" {
    const gpa = testing.allocator;
    const routed = 512;
    const top = 10;
    const rows = 4;
    var prng: std.Random.DefaultPrng = .init(9);
    const random = prng.random();
    const logits = try gpa.alloc(f32, rows * (routed + 1));
    defer gpa.free(logits);
    for (logits) |*v| v.* = random.floatNorm(f32) * 3.0;
    // row 1: equal logits at the top, among them the last expert; row 2: every logit equal
    for ([_]usize{ 500, 17, 511, 300 }) |e| logits[1 * (routed + 1) + e] = 20.0;
    @memset(logits[2 * (routed + 1) ..][0..routed], 1.5);

    var logits_dev: cuda.Buffer = try .init(.f32, &.{ rows, routed + 1 });
    defer logits_dev.deinit();
    try logits_dev.upload(std.mem.sliceAsBytes(logits));
    var picks_dev: cuda.Buffer = try .init(.i32, &.{ rows, top + 1 });
    defer picks_dev.deinit();
    var weights_dev: cuda.Buffer = try .init(.f32, &.{ rows, top + 1 });
    defer weights_dev.deinit();
    try cuda.launched(zs_qwen4_moe_topk(&logits_dev.view, &picks_dev.view, &weights_dev.view, top, null));
    const picks = try fetch(i32, gpa, picks_dev);
    defer gpa.free(picks);
    const weights = try fetch(f32, gpa, weights_dev);
    defer gpa.free(weights);

    for (0..rows) |r| {
        const row = logits[r * (routed + 1) ..][0 .. routed + 1];
        var taken: [routed]bool = @splat(false);
        var want: [top]usize = undefined;
        var exps: [top]f32 = undefined;
        var total: f32 = 0;
        for (0..top) |k| {
            var best: usize = routed;
            for (0..routed) |e| {
                if (!taken[e] and (best == routed or row[e] > row[best])) best = e;
            }
            taken[best] = true;
            want[k] = best;
            exps[k] = @exp(row[best] - row[want[0]]);
            total += exps[k];
        }
        for (0..top) |k| {
            try testing.expectEqual(@as(i32, @intCast(want[k])), picks[r * (top + 1) + k]);
            try testing.expectApproxEqRel(exps[k] / total, weights[r * (top + 1) + k], 1.0 / 64.0);
        }
        try testing.expectEqual(@as(i32, routed), picks[r * (top + 1) + top]);
        const gate = bf(row[routed]);
        try testing.expectApproxEqRel(1.0 / (1.0 + @exp(-gate)), weights[r * (top + 1) + top], 1.0 / 64.0);
    }
    try testing.expectEqualSlices(i32, &.{ 17, 300, 500, 511 }, picks[1 * (top + 1) ..][0..4]);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, picks[2 * (top + 1) ..][0..top]);
}

test "top-k with a NaN among a row's logits picks experts that exist, and the rows beside it keep their bits" {
    const gpa = testing.allocator;
    const routed = 512;
    const top = 10;
    const rows = 4;
    var prng: std.Random.DefaultPrng = .init(81);
    const random = prng.random();
    const logits = try gpa.alloc(f32, rows * (routed + 1));
    defer gpa.free(logits);
    for (logits) |*v| v.* = random.floatNorm(f32) * 3.0;
    const nan = std.math.nan(f32);
    // row 1: a few NaNs, one of them in the first warp; row 2: every logit; rows 0 and 3 have none
    for ([_]usize{ 3, 40, 41, 300, 511 }) |e| logits[1 * (routed + 1) + e] = nan;
    @memset(logits[2 * (routed + 1) ..][0 .. routed + 1], nan);

    var picks_dev: cuda.Buffer = try .init(.i32, &.{ rows, top + 1 });
    defer picks_dev.deinit();
    var weights_dev: cuda.Buffer = try .init(.f32, &.{ rows, top + 1 });
    defer weights_dev.deinit();
    // The launch is repeated over other launches' shared memory: what a place never written would show.
    for (0..8) |_| {
        var logits_dev: cuda.Buffer = try .init(.f32, &.{ rows, routed + 1 });
        defer logits_dev.deinit();
        try logits_dev.upload(std.mem.sliceAsBytes(logits));
        try cuda.launched(zs_qwen4_moe_topk(&logits_dev.view, &picks_dev.view, &weights_dev.view, top, null));
        const picks = try fetch(i32, gpa, picks_dev);
        defer gpa.free(picks);
        for (0..rows) |r| {
            for (picks[r * (top + 1) ..][0..top]) |e| try testing.expect(e >= 0 and e < routed);
            try testing.expectEqual(@as(i32, routed), picks[r * (top + 1) + top]);
        }
        // every candidate of the all-NaN row counts place 0, so places 1 and after are what they started as:
        // without that start they are whatever the block's shared memory held, in range or not
        try testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, picks[2 * (top + 1) + 1 ..][0 .. top - 1]);
    }
    const picks = try fetch(i32, gpa, picks_dev);
    defer gpa.free(picks);
    const weights = try fetch(f32, gpa, weights_dev);
    defer gpa.free(weights);

    for ([_]usize{ 0, 3 }) |r| {
        var one_logits: cuda.Buffer = try .init(.f32, &.{ 1, routed + 1 });
        defer one_logits.deinit();
        try one_logits.upload(std.mem.sliceAsBytes(logits[r * (routed + 1) ..][0 .. routed + 1]));
        var one_picks: cuda.Buffer = try .init(.i32, &.{ 1, top + 1 });
        defer one_picks.deinit();
        var one_weights: cuda.Buffer = try .init(.f32, &.{ 1, top + 1 });
        defer one_weights.deinit();
        try cuda.launched(zs_qwen4_moe_topk(&one_logits.view, &one_picks.view, &one_weights.view, top, null));
        const alone = try fetch(i32, gpa, one_picks);
        defer gpa.free(alone);
        const alone_weights = try fetch(f32, gpa, one_weights);
        defer gpa.free(alone_weights);
        try testing.expectEqualSlices(i32, alone, picks[r * (top + 1) ..][0 .. top + 1]);
        try testing.expectEqualSlices(u32, @ptrCast(alone_weights), @ptrCast(weights[r * (top + 1) ..][0 .. top + 1]));
    }
}

test "the selection over a router's unsummed slices: the reduction's logits, then the same picks and weights" {
    // Spec 028: the selection adds the router's K slices as it reads them. Slices built so that their sums tie
    // (rows 1 and 2) and so that the order of the adds shows in the others' last bits.
    const gpa = testing.allocator;
    const top = 10;
    const rows = 5;
    var prng: std.Random.DefaultPrng = .init(28);
    const random = prng.random();
    for ([_][2]usize{ .{ 512, 8 }, .{ 512, 2 }, .{ 37, 4 }, .{ 100, 32 } }) |shape| {
        const routed, const sk = shape;
        const n = routed + 1;
        const part = try gpa.alloc(f32, sk * rows * n);
        defer gpa.free(part);
        for (part) |*v| v.* = random.floatNorm(f32) * 3.0 * @exp(random.floatNorm(f32));
        // small integers sum exactly in any order: equal logits, among them the last expert
        for (0..sk) |s| {
            for (0..routed) |e| part[(s * rows + 1) * n + e] = @floatFromInt((e * 7 + s) % 3);
            for ([_]usize{ routed - 1, 17, 3, 30 }) |e| part[(s * rows + 1) * n + e] = 5.0;
            for (0..routed) |e| part[(s * rows + 2) * n + e] = 0.25;
        }
        const summed = try gpa.alloc(f32, rows * n);
        defer gpa.free(summed);
        for (summed, 0..) |*total, i| {
            total.* = part[i];
            for (1..sk) |s| total.* += part[s * rows * n + i];
        }

        var logits_dev: cuda.Buffer = try .init(.f32, &.{ rows, n });
        defer logits_dev.deinit();
        try logits_dev.upload(std.mem.sliceAsBytes(summed));
        var part_dev: cuda.Buffer = try .init(.f32, &.{ sk, rows, n });
        defer part_dev.deinit();
        try part_dev.upload(std.mem.sliceAsBytes(part));
        var picks_dev: cuda.Buffer = try .init(.i32, &.{ rows, top + 1 });
        defer picks_dev.deinit();
        var weights_dev: cuda.Buffer = try .init(.f32, &.{ rows, top + 1 });
        defer weights_dev.deinit();
        try cuda.launched(zs_qwen4_moe_topk(&logits_dev.view, &picks_dev.view, &weights_dev.view, top, null));
        const want_picks = try fetch(i32, gpa, picks_dev);
        defer gpa.free(want_picks);
        const want_weights = try fetch(u32, gpa, weights_dev);
        defer gpa.free(want_weights);
        try testing.expectEqualSlices(i32, &.{ 3, 17, 30, @intCast(routed - 1) }, want_picks[1 * (top + 1) ..][0..4]);

        try logits_dev.zero();
        try picks_dev.zero();
        try weights_dev.zero();
        try cuda.launched(zs_qwen4_moe_topk(&logits_dev.view, &picks_dev.view, &weights_dev.view, top, &part_dev.view));
        const got_picks = try fetch(i32, gpa, picks_dev);
        defer gpa.free(got_picks);
        const got_weights = try fetch(u32, gpa, weights_dev);
        defer gpa.free(got_weights);
        const got_logits = try fetch(u32, gpa, logits_dev);
        defer gpa.free(got_logits);
        try testing.expectEqualSlices(i32, want_picks, got_picks);
        try testing.expectEqualSlices(u32, want_weights, got_weights);
        try testing.expectEqualSlices(u32, @ptrCast(summed), got_logits);
    }
}

test "equal logits go to the lower id wherever the reduction meets them" {
    // The reduction halves 1,024 slots ten times. After the first halving a lower slot can hold a higher id than
    // the slot it meets next, so a tie must compare ids, not slots. Tied sets spread over every distance exercise
    // each level; a first version that let the lower slot win got row 1 of the test above wrong.
    const gpa = testing.allocator;
    const top = 6;
    var prng: std.Random.DefaultPrng = .init(77);
    const random = prng.random();
    for ([_]usize{ 7, 512, 1000, 1023 }) |routed| {
        const rows = 64;
        const logits = try gpa.alloc(f32, rows * (routed + 1));
        defer gpa.free(logits);
        const want = try gpa.alloc(i32, rows * top);
        defer gpa.free(want);
        for (0..rows) |r| {
            const row = logits[r * (routed + 1) ..][0 .. routed + 1];
            for (row) |*v| v.* = -random.float(f32) - 1.0;
            // more tied experts than picks, at ids chosen anywhere, in two tiers so order within and across counts
            const tied = top + 1 + random.uintLessThan(usize, 5);
            for (0..tied) |i| row[random.uintLessThan(usize, routed)] = if (i % 2 == 0) 4.0 else 3.0;
            var taken = try gpa.alloc(bool, routed);
            defer gpa.free(taken);
            @memset(taken, false);
            for (0..top) |k| {
                var best: usize = routed;
                for (0..routed) |e| {
                    if (!taken[e] and (best == routed or row[e] > row[best])) best = e;
                }
                taken[best] = true;
                want[r * top + k] = @intCast(best);
            }
        }
        var logits_dev: cuda.Buffer = try .init(.f32, &.{ rows, routed + 1 });
        defer logits_dev.deinit();
        try logits_dev.upload(std.mem.sliceAsBytes(logits));
        var picks_dev: cuda.Buffer = try .init(.i32, &.{ rows, top + 1 });
        defer picks_dev.deinit();
        var weights_dev: cuda.Buffer = try .init(.f32, &.{ rows, top + 1 });
        defer weights_dev.deinit();
        try cuda.launched(zs_qwen4_moe_topk(&logits_dev.view, &picks_dev.view, &weights_dev.view, top, null));
        const picks = try fetch(i32, gpa, picks_dev);
        defer gpa.free(picks);
        for (0..rows) |r| try testing.expectEqualSlices(i32, want[r * top ..][0..top], picks[r * (top + 1) ..][0..top]);
    }
}

const fx = struct {
    const routed = 24;
    const top = 3;
    const slots = top + 1;
    const width = 64;
    const dims = 192;
    const max_rows = 9;
};

const Dense = struct {
    bits: []u16,

    fn init(gpa: Allocator, random: std.Random, n: usize, k: usize, spread: f32) !Dense {
        const bits = try gpa.alloc(u16, n * k);
        for (bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32) * spread);
        return .{ .bits = bits };
    }

    fn row(self: Dense, r: usize, k: usize) []const u16 {
        return self.bits[r * k ..][0..k];
    }
};

fn dot(w: []const u16, x: []const u16) f64 {
    var sum: f64 = 0;
    for (w, x) |a, b| sum += @as(f64, numeric.f32FromBf16(a)) * numeric.f32FromBf16(b);
    return sum;
}

fn randomMatrix(gpa: Allocator, random: std.Random, n: usize, k: usize) !experts.Matrix {
    const codes = try gpa.alloc(u8, fx.routed * n * k / 2);
    random.bytes(codes);
    const block_scales = try gpa.alloc(u8, fx.routed * n * k / 16);
    for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
    const scales = try gpa.alloc(f32, fx.routed);
    for (scales) |*scale| scale.* = 0.01 + 0.02 * random.float(f32);
    return .{ .codes = try experts.Matrix.parts(gpa, codes, fx.routed), .block_scales = try experts.Matrix.parts(gpa, block_scales, fx.routed), .scales = scales };
}

test "the MoE's shared expert and slots match the host, and a row keeps its bits among others" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(21);
    const random = prng.random();

    const router: Dense = try .init(gpa, random, fx.routed + 1, fx.dims, 0.3);
    const gate_up: Dense = try .init(gpa, random, 2 * fx.width, fx.dims, 0.1);
    const down: Dense = try .init(gpa, random, fx.dims, fx.width, 0.1);
    const x: Dense = try .init(gpa, random, fx.max_rows, fx.dims, 1.0);
    var staging: experts.Staging = .{};
    defer staging.deinit(gpa);
    var scratch: cuda.Scratch = .init(gpa);
    defer scratch.deinit();
    var weights: Weights = .{
        .router = try .init(std.mem.sliceAsBytes(router.bits), fx.routed + 1, fx.dims, scratch.rows(fx.max_rows)),
        .stored = .{ .nvfp4 = .{
            .routed = .{ .nvfp4 = try .init(gpa, &staging, fx.routed, fx.width, fx.dims, try randomMatrix(gpa, random, fx.width, fx.dims), try randomMatrix(gpa, random, fx.width, fx.dims), try randomMatrix(gpa, random, fx.dims, fx.width)) },
            .shared_gate_up = try .init(std.mem.sliceAsBytes(gate_up.bits), 2 * fx.width, fx.dims, scratch.rows(fx.max_rows)),
            .shared_down = try .init(std.mem.sliceAsBytes(down.bits), fx.dims, fx.width, scratch.rows(fx.max_rows)),
        } },
    };
    defer weights.deinit();

    var first_y: []f32 = &.{};
    var first_shared: []f32 = &.{};
    var first_weights: []f32 = &.{};
    for ([_]usize{ 1, fx.max_rows }) |rows| {
        var buffers: Buffers = try .init(rows, fx.top, fx.routed, fx.width, fx.dims, false);
        defer buffers.deinit();
        var x_dev: cuda.Buffer = try .init(.bf16, &.{ rows, fx.dims });
        defer x_dev.deinit();
        try x_dev.upload(std.mem.sliceAsBytes(x.bits[0 .. rows * fx.dims]));
        try forward(weights, buffers, x_dev.view, rows, rows);
        const y = try fetch(f32, gpa, buffers.y);
        const shared_y = try fetch(f32, gpa, buffers.shared_y);
        const slot_weights = try fetch(f32, gpa, buffers.weights);
        const picks = try fetch(i32, gpa, buffers.picks);
        const logits = try fetch(f32, gpa, buffers.logits);

        for (0..rows) |r| {
            const xr = x.row(r, fx.dims);
            // the router's logits, and picks that are the largest of them
            var lowest_pick = std.math.inf(f32);
            for (0..fx.top) |k| lowest_pick = @min(lowest_pick, logits[r * (fx.routed + 1) + @as(usize, @intCast(picks[r * fx.slots + k]))]);
            for (0..fx.routed + 1) |e| {
                const want = dot(router.row(e, fx.dims), xr);
                const got = logits[r * (fx.routed + 1) + e];
                try testing.expectApproxEqAbs(want, @as(f64, got), 1e-3);
                var picked = false;
                for (0..fx.top) |k| picked = picked or picks[r * fx.slots + k] == @as(i32, @intCast(e));
                if (e < fx.routed and !picked) try testing.expect(got <= lowest_pick);
            }
            // the shared expert in the last slot
            var act: [fx.width]u16 = undefined;
            for (&act, 0..) |*a, i| {
                const g = bf(@floatCast(dot(gate_up.row(i, fx.dims), xr)));
                const u = bf(@floatCast(dot(gate_up.row(fx.width + i, fx.dims), xr)));
                a.* = numeric.bf16FromF32(bf(g / (1.0 + @exp(-g))) * u);
            }
            for (0..fx.dims) |d| {
                const want = dot(down.row(d, fx.width), &act);
                try testing.expectApproxEqAbs(want, @as(f64, shared_y[r * fx.dims + d]), @abs(want) / 16.0 + 2e-2);
            }
        }
        if (rows == 1) {
            first_y = y;
            first_shared = shared_y;
            first_weights = slot_weights;
        } else {
            try testing.expectEqualSlices(u32, @ptrCast(first_y[0 .. fx.top * fx.dims]), @ptrCast(y[0 .. fx.top * fx.dims]));
            try testing.expectEqualSlices(u32, @ptrCast(first_shared), @ptrCast(shared_y[0..fx.dims]));
            try testing.expectEqualSlices(u32, @ptrCast(first_weights), @ptrCast(slot_weights[0..fx.slots]));
        }
    }
}
