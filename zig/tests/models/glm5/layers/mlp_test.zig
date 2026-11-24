//! `mlp.zig`'s tests on random weights: each rank's partial against the whole matrices on the host in fp64, the
//! two ranks' picks against each other and the host's, and a row alone against the row among others.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const nvfp4 = @import("../../../../src/weights/nvfp4.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const config_file = @import("../../../../src/models/glm5/layers/config.zig");
const fixtures = @import("../../../../src/models/glm5/checks/fixtures.zig");
const host = @import("../../../../src/models/glm5/layers/glue_ref.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const mlp = @import("../../../../src/models/glm5/layers/mlp.zig");
const shared = @import("../testing.zig");
const device = shared.device;
const fetch = shared.fetch;
const Source = fixtures.Source;

const fx = struct {
    const dims = 256;
    const intermediate = 512;
    const moe_width = 256;
    const experts = 12;
    const top = 3;
    const limit = 3.0;

    /// The fixture's config at sizes an NVFP4 expert and a GEMM take.
    fn config(arena: Allocator) !config_file.Config {
        var diagnostic: config_file.Diagnostic = .{};
        var c = try config_file.Config.parse(arena, fixtures.config, &diagnostic);
        c.hidden = dims;
        c.intermediate = intermediate;
        c.moe_width = moe_width;
        c.experts = experts;
        c.top_k = top;
        c.swiglu_limit = limit;
        return c;
    }

    /// A random bf16 [n, k] matrix under `name`; values near `spread`.
    fn putMatrix(arena: Allocator, random: std.Random, source: *Source, name: []const u8, n: usize, k: usize, spread: f32) !void {
        const bits = try arena.alloc(u16, n * k);
        shared.randomBf16(random, bits, spread);
        try source.put(arena, name, .{ .dtype = .BF16, .shape = try arena.dupe(u64, &.{ n, k }), .data = std.mem.sliceAsBytes(bits) });
    }

    /// The three matrices of a BF16 MLP under `prefix`, the gate and the up large enough for the clip to bite.
    fn putDense(arena: Allocator, random: std.Random, source: *Source, prefix: []const u8, width: usize) !void {
        try putMatrix(arena, random, source, try std.mem.concat(arena, u8, &.{ prefix, "gate_proj.weight" }), width, dims, 0.15);
        try putMatrix(arena, random, source, try std.mem.concat(arena, u8, &.{ prefix, "up_proj.weight" }), width, dims, 0.15);
        try putMatrix(arena, random, source, try std.mem.concat(arena, u8, &.{ prefix, "down_proj.weight" }), dims, width, 0.1);
    }

    fn value(source: Source, name: []const u8, at: usize) f64 {
        const bits: []align(1) const u16 = @ptrCast(source.get(name).?.data);
        return numeric.f32FromBf16(bits[at]);
    }

    fn bf(v: f64) f64 {
        return numeric.roundBf16(@floatCast(v));
    }

    /// SwiGLU as the kernels round it: the gate and the up bf16, silu bf16, the product bf16.
    fn act(gate: f64, up: f64) f64 {
        const g = @min(bf(gate), limit);
        const u = @min(@max(bf(up), -limit), limit);
        return bf(bf(g / (1.0 + @exp(-g))) * u);
    }

    /// out += weight * the BF16 MLP under `prefix` over `rank`'s half of the inner width, from whole matrices.
    fn addDense(arena: Allocator, source: Source, prefix: []const u8, width: usize, rank: usize, x: []const f64, weight: f64, out: []f64) !void {
        const gate = try std.mem.concat(arena, u8, &.{ prefix, "gate_proj.weight" });
        const up = try std.mem.concat(arena, u8, &.{ prefix, "up_proj.weight" });
        const down = try std.mem.concat(arena, u8, &.{ prefix, "down_proj.weight" });
        const half = width / layout.world;
        for (rank * half..(rank + 1) * half) |j| {
            var g: f64 = 0;
            var u: f64 = 0;
            for (x, 0..) |xv, i| {
                g += xv * value(source, gate, j * dims + i);
                u += xv * value(source, up, j * dims + i);
            }
            const a = act(g, u);
            for (out, 0..) |*o, d| o.* += weight * a * value(source, down, d * width + j);
        }
    }

    /// ||got - want|| / ||want||
    fn distance(want: []const f64, got: []const f32) f64 {
        var off: f64 = 0;
        var size: f64 = 0;
        for (want, got) |w, g| {
            off += (g - w) * (g - w);
            size += w * w;
        }
        return @sqrt(off / size);
    }
};

/// What the split may change of a dense MLP, measured on the device: both ranks' partials added against the MLP of
/// whole matrices (whose K slices differ). Returns the distance.
fn againstWhole(arena: Allocator, source: anytype, c: *const config_file.Config, dense_layer: usize, x: cuda.View) !f64 {
    const gpa = testing.allocator;
    const rows: usize = @intCast(x.size[0]);
    const dims = c.hidden;
    var pool: cuda.Scratch = .init(gpa);
    defer pool.deinit();
    const scratch_rows: cuda.Scratch.Rows = .{ .pool = &pool, .count = rows };
    var out: cuda.Buffer = try .init(.f32, &.{ rows, dims });
    defer out.deinit();
    var distance: f64 = 0;

    const p = try std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.mlp.", .{dense_layer});
    const sum = try arena.alloc(f64, rows * dims);
    @memset(sum, 0);
    for (0..layout.world) |rank| {
        var dense: mlp.Dense = try .load(arena, source, "{s}", .{p}, c, rank, scratch_rows);
        defer dense.deinit();
        var buffers: mlp.Dense.Buffers = try .init(dense.width, rows);
        defer buffers.deinit();
        try dense.run(buffers, x, out.view);
        const got = try fetch(f32, gpa, out);
        defer gpa.free(got);
        for (sum, got) |*total, v| total.* += v;
    }
    var parts: [3][]const u8 = undefined;
    for (&parts, [_][]const u8{ "gate_proj.weight", "up_proj.weight", "down_proj.weight" }) |*part, leaf| {
        part.* = source.get(try std.mem.concat(arena, u8, &.{ p, leaf })).?.data;
    }
    var dense: mlp.Dense = try .init(parts[0], parts[1], parts[2], dims, @floatCast(c.swiglu_limit), scratch_rows);
    defer dense.deinit();
    try testing.expectEqual(c.intermediate, dense.width);
    var buffers: mlp.Dense.Buffers = try .init(dense.width, rows);
    defer buffers.deinit();
    try dense.run(buffers, x, out.view);
    const got = try fetch(f32, gpa, out);
    defer gpa.free(got);
    for (0..rows) |r| distance = @max(distance, fx.distance(sum[r * dims ..][0..dims], got[r * dims ..][0..dims]));
    return distance;
}

test "a dense MLP's partial is its rank's half of the whole product, a row alone has the window's bits" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(11);
    const random = prng.random();
    const c = try fx.config(arena);
    const prefix = layout.prefix ++ "layers.0.mlp.";
    var source: Source = .empty;
    try fx.putDense(arena, random, &source, prefix, fx.intermediate);
    const rows = 5;
    const x = try arena.alloc(u16, rows * fx.dims);
    shared.randomBf16(random, x, 1.5);
    var x_dev = try device(u16, .bf16, &.{ rows, fx.dims }, x);
    defer x_dev.deinit();
    var out: cuda.Buffer = try .init(.f32, &.{ rows, fx.dims });
    defer out.deinit();
    var out1: cuda.Buffer = try .init(.f32, &.{ 1, fx.dims });
    defer out1.deinit();
    var pool: cuda.Scratch = .init(testing.allocator);
    defer pool.deinit();
    const x_row = try arena.alloc(f64, fx.dims);
    const want = try arena.alloc(f64, fx.dims);
    var worst: f64 = 0;
    for (0..layout.world) |rank| {
        var dense: mlp.Dense = try .load(arena, source, prefix, .{}, &c, rank, .{ .pool = &pool, .count = rows });
        defer dense.deinit();
        try testing.expectEqual(@as(usize, fx.intermediate / 2), dense.width);
        var buffers: mlp.Dense.Buffers = try .init(dense.width, rows);
        defer buffers.deinit();
        try dense.run(buffers, x_dev.view, out.view);
        const got = try fetch(f32, testing.allocator, out);
        defer testing.allocator.free(got);
        for (0..rows) |r| {
            for (x_row, x[r * fx.dims ..][0..fx.dims]) |*v, bits| v.* = numeric.f32FromBf16(bits);
            @memset(want, 0);
            try fx.addDense(arena, source, prefix, fx.intermediate, rank, x_row, 1, want);
            worst = @max(worst, fx.distance(want, got[r * fx.dims ..][0..fx.dims]));
            try dense.run(buffers, x_dev.span(r, 1), out1.view);
            const one = try fetch(u32, testing.allocator, out1);
            defer testing.allocator.free(one);
            try testing.expectEqualSlices(u32, @ptrCast(got[r * fx.dims ..][0..fx.dims]), one);
        }
    }
    // fp32 sums against fp64 ones, and now and then a bf16 rounding that falls the other way: 1.4e-4 here
    try testing.expect(worst > 0 and worst < 5e-4);
}

test "a MoE's partial is its rank's half of the picked experts and the shared one; both ranks pick alike; a row alone" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(12);
    const random = prng.random();
    const c = try fx.config(arena);
    const layer = 3;
    const p = std.fmt.comptimePrint(layout.prefix ++ "layers.{d}.mlp.", .{layer});
    var source: Source = .empty;
    try fx.putMatrix(arena, random, &source, p ++ "gate.weight", fx.experts, fx.dims, 0.1);
    const bias = try arena.alloc(f32, fx.experts);
    shared.randomF32(random, bias, 0.05);
    try source.put(arena, p ++ "gate.e_score_correction_bias", .{ .dtype = .F32, .shape = try arena.dupe(u64, &.{fx.experts}), .data = std.mem.sliceAsBytes(bias) });
    try fx.putDense(arena, random, &source, p ++ "shared_experts.", fx.moe_width);
    // [experts][dims x width], [experts][dims x width], [experts][width x dims]: W row-major, input first
    // the experts NVFP4 with an input scale, as FP4-Local's layers 3 to 44 (`fixtures.config`'s map: W4A4)
    var dense: [3][][]f64 = undefined;
    for (&dense, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |*list, leaf, m| {
        list.* = try arena.alloc([]f64, fx.experts);
        for (list.*, 0..) |*w, e| {
            const prefix = try std.fmt.allocPrint(arena, p ++ "experts.{d}.{s}", .{ e, leaf });
            const k: usize = if (m == 2) fx.moe_width else fx.dims;
            const n: usize = if (m == 2) fx.dims else fx.moe_width;
            try fixtures.putNvfp4(arena, random, &source, prefix, k, n, if (m == 2) 0.1 else 1.0 / @sqrt(@as(f64, @floatFromInt(k))));
            // stored [n, k]: the lower input in a byte's low nibble, a scale a group of 16 inputs
            const codes = source.get(try std.mem.concat(arena, u8, &.{ prefix, ".weight" })).?.data;
            const scales = source.get(try std.mem.concat(arena, u8, &.{ prefix, ".weight_scale" })).?.data;
            const global: f32 = @bitCast(source.get(try std.mem.concat(arena, u8, &.{ prefix, ".weight_scale_2" })).?.data[0..4].*);
            w.* = try arena.alloc(f64, k * n);
            for (0..n) |o| for (0..k) |i| {
                const code = codes[(o * k + i) / 2] >> @intCast(4 * (i % 2)) & 0xf;
                w.*[i * n + o] = @as(f64, nvfp4.e2m1[code]) * numeric.f32FromE4m3(scales[o * (k / 16) + i / 16]) * global;
            };
        }
    }

    const rows = 6;
    const slots = fx.top + 1;
    const x = try arena.alloc(u16, rows * fx.dims);
    shared.randomBf16(random, x, 1.5);
    var x_dev = try device(u16, .bf16, &.{ rows, fx.dims }, x);
    defer x_dev.deinit();
    var out: cuda.Buffer = try .init(.f32, &.{ rows, fx.dims });
    defer out.deinit();
    var out1: cuda.Buffer = try .init(.f32, &.{ 1, fx.dims });
    defer out1.deinit();
    var pool: cuda.Scratch = .init(testing.allocator);
    defer pool.deinit();
    const x_row = try arena.alloc(f64, fx.dims);
    const want = try arena.alloc(f64, fx.dims);
    var want_picks: [slots]i32 = undefined;
    var want_weights: [slots]f32 = undefined;
    var first_picks: []i32 = &.{};
    var worst: f64 = 0;
    for (0..layout.world) |rank| {
        var moe: mlp.Moe = try .load(arena, testing.allocator, source, &c, layer, rank, .{ .pool = &pool, .count = rows });
        defer moe.deinit();
        try testing.expectEqual(@as(usize, fx.moe_width / 2), moe.routed.width);
        var buffers: mlp.Moe.Buffers = try .init(moe, rows);
        defer buffers.deinit();
        var alone: mlp.Moe.Buffers = try .init(moe, 1);
        defer alone.deinit();
        // every row before `from` = rows A16: the host's bf16 rows on the stored weights
        try moe.forward(buffers, x_dev.view, out.view, rows);
        const got = try fetch(f32, testing.allocator, out);
        defer testing.allocator.free(got);
        const logits = try fetch(f32, testing.allocator, buffers.logits);
        defer testing.allocator.free(logits);
        const picks = try fetch(i32, arena, buffers.picks);
        const weights = try fetch(f32, testing.allocator, buffers.weights);
        defer testing.allocator.free(weights);
        // the router is whole on both ranks: the same picks, bit for bit
        if (rank == 0) first_picks = picks else try testing.expectEqualSlices(i32, first_picks, picks);
        for (0..rows) |r| {
            host.select(logits[r * fx.experts ..][0..fx.experts], bias, &want_picks, &want_weights, @floatCast(c.routed_scale), true);
            try testing.expectEqualSlices(i32, &want_picks, picks[r * slots ..][0..slots]);
            try testing.expectEqualSlices(u32, @ptrCast(&want_weights), @ptrCast(weights[r * slots ..][0..slots]));
            try testing.expectEqual(@as(i32, fx.experts), want_picks[fx.top]);

            for (x_row, x[r * fx.dims ..][0..fx.dims]) |*v, bits| v.* = numeric.f32FromBf16(bits);
            @memset(want, 0);
            const half = fx.moe_width / layout.world;
            for (want_picks[0..fx.top], want_weights[0..fx.top]) |pick, weight| {
                const e: usize = @intCast(pick);
                for (rank * half..(rank + 1) * half) |j| {
                    var g: f64 = 0;
                    var u: f64 = 0;
                    for (x_row, 0..) |xv, i| {
                        g += xv * dense[0][e][i * fx.moe_width + j];
                        u += xv * dense[1][e][i * fx.moe_width + j];
                    }
                    const a = fx.act(g, u);
                    for (want, 0..) |*o, d| o.* += weight * a * dense[2][e][j * fx.dims + d];
                }
            }
            try fx.addDense(arena, source, p ++ "shared_experts.", fx.moe_width, rank, x_row, 1, want);
            worst = @max(worst, fx.distance(want, got[r * fx.dims ..][0..fx.dims]));

            try moe.forward(alone, x_dev.span(r, 1), out1.view, 1);
            const one = try fetch(u32, testing.allocator, out1);
            defer testing.allocator.free(one);
            try testing.expectEqualSlices(u32, @ptrCast(got[r * fx.dims ..][0..fx.dims]), one);
        }
        // W4A4, every row quantized: a row alone has the window's bits too
        try moe.forward(buffers, x_dev.view, out.view, 0);
        const quantized = try fetch(u32, testing.allocator, out);
        defer testing.allocator.free(quantized);
        try testing.expect(!std.mem.eql(u32, quantized, @ptrCast(got)));
        for (0..rows) |r| {
            try moe.forward(alone, x_dev.span(r, 1), out1.view, 0);
            const one = try fetch(u32, testing.allocator, out1);
            defer testing.allocator.free(one);
            try testing.expectEqualSlices(u32, quantized[r * fx.dims ..][0..fx.dims], one);
        }
    }
    try fx.putDense(arena, random, &source, layout.prefix ++ "layers.0.mlp.", fx.intermediate);
    // the cut changes the order of fp32 sums and nothing else
    try testing.expect(try againstWhole(arena, source, &c, 0, x_dev.view) < 2e-6);
    // the device sums in fp32 and rounds the gate, the up and their product to bf16 as the host does, now and
    // then the other way: 2.7e-3 here
    try testing.expect(worst > 0 and worst < 4e-3);
}
