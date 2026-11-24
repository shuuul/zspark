//! `experts.zig`'s tests.

const std = @import("std");
const cuda = @import("../../src/cuda/cuda.zig");
const experts_file = @import("../../src/weights/experts.zig");
const numeric = @import("../../src/weights/numeric.zig");
const Allocator = std.mem.Allocator;
const Experts = experts_file.Experts;
const Matrix = experts_file.Matrix;
const Pass = experts_file.Pass;
const Plan = experts_file.Plan;
const Staging = experts_file.Staging;
const nvfp4 = @import("../../src/weights/nvfp4.zig");

const TestMatrix = struct {
    codes: []u8,
    block_scales: []u8,
    scales: []f32,
    /// [experts, n, k] as the host dequantizes it
    dense: []f64,

    fn init(gpa: Allocator, random: std.Random, experts: usize, n: usize, k: usize) !TestMatrix {
        const codes = try gpa.alloc(u8, experts * n * k / 2);
        errdefer gpa.free(codes);
        random.bytes(codes);
        const block_scales = try gpa.alloc(u8, experts * n * k / 16);
        errdefer gpa.free(block_scales);
        for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
        const scales = try gpa.alloc(f32, experts);
        errdefer gpa.free(scales);
        for (scales) |*scale| scale.* = 0.01 + 0.02 * random.float(f32);
        const dense = try gpa.alloc(f64, experts * n * k);
        for (dense, 0..) |*value, i| {
            const byte = codes[i / 2];
            const code = if (i % 2 == 0) byte & 0xf else byte >> 4;
            value.* = @as(f64, nvfp4.e2m1[code]) * numeric.f32FromE4m3(block_scales[i / 16]) * scales[i / (n * k)];
        }
        return .{ .codes = codes, .block_scales = block_scales, .scales = scales, .dense = dense };
    }

    fn deinit(self: TestMatrix, gpa: Allocator) void {
        gpa.free(self.codes);
        gpa.free(self.block_scales);
        gpa.free(self.scales);
        gpa.free(self.dense);
    }

    fn matrix(self: TestMatrix, arena: Allocator) !Matrix {
        const count = self.scales.len;
        return .{ .codes = try Matrix.parts(arena, self.codes, count), .block_scales = try Matrix.parts(arena, self.block_scales, count), .scales = self.scales };
    }
};

const Dot = struct { value: f64, magnitude: f64 };

fn dot(weights: []const f64, x_bits: []const u16) Dot {
    var result: Dot = .{ .value = 0, .magnitude = 0 };
    for (weights, x_bits) |w, bits| {
        const term = w * numeric.f32FromBf16(bits);
        result.value += term;
        result.magnitude += @abs(term);
    }
    return result;
}

test "routed experts match the host and keep a pair's bits whatever shares its call" {
    const gpa = std.testing.allocator;
    const count = 24;
    const width = 64;
    const dims = 160;
    const slots = 3;
    var prng: std.Random.DefaultPrng = .init(7);
    const random = prng.random();

    const gate: TestMatrix = try .init(gpa, random, count, width, dims);
    defer gate.deinit(gpa);
    const up: TestMatrix = try .init(gpa, random, count, width, dims);
    defer up.deinit(gpa);
    const down: TestMatrix = try .init(gpa, random, count, dims, width);
    defer down.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var staging: Staging = .{};
    defer staging.deinit(gpa);
    var layer: Experts = try .init(gpa, &staging, count, width, dims, try gate.matrix(arena.allocator()), try up.matrix(arena.allocator()), try down.matrix(arena.allocator()));
    defer layer.deinit();

    // one-block plans, an item past 16 pairs, and a plan wide enough to rank in blocks
    const row_counts = [_]usize{ 1, 2, 9, 40, 400 };
    const max_rows = row_counts[row_counts.len - 1];
    const x_bits = try gpa.alloc(u16, max_rows * dims);
    defer gpa.free(x_bits);
    for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    const picks = try gpa.alloc(i32, max_rows * slots);
    defer gpa.free(picks);
    for (0..max_rows) |row| {
        // distinct experts a row, few enough overall that some expert takes more than a tile of pairs
        const first = random.uintLessThan(usize, count);
        for (0..slots) |slot| picks[row * slots + slot] = @intCast((first + slot * 5) % count);
    }

    const act_bits = try gpa.alloc(u16, max_rows * slots * width);
    defer gpa.free(act_bits);
    const out = try gpa.alloc(f32, max_rows * slots * dims);
    defer gpa.free(out);
    const first_act = try gpa.alloc(u16, slots * width);
    defer gpa.free(first_act);
    const first_out = try gpa.alloc(f32, slots * dims);
    defer gpa.free(first_out);

    for (row_counts, 0..) |rows, round| {
        const pairs = rows * slots;
        var plan: Plan = try .init(rows, slots, count);
        defer plan.deinit();
        var x_dev: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
        defer x_dev.deinit();
        try x_dev.upload(std.mem.sliceAsBytes(x_bits[0 .. rows * dims]));
        var picks_dev: cuda.Buffer = try .init(.i32, &.{ rows, slots });
        defer picks_dev.deinit();
        try picks_dev.upload(std.mem.sliceAsBytes(picks[0..pairs]));
        var act_dev: cuda.Buffer = try .init(.bf16, &.{ pairs, width });
        defer act_dev.deinit();
        var out_dev: cuda.Buffer = try .init(.f32, &.{ pairs, dims });
        defer out_dev.deinit();

        try plan.route(picks_dev.view, rows);
        try layer.gateUp(x_dev.view, plan, act_dev.view, rows, null);
        try layer.downProject(act_dev.view, plan, out_dev.view, rows, null);
        try cuda.synchronize();
        try act_dev.download(std.mem.sliceAsBytes(act_bits[0 .. pairs * width]));
        try out_dev.download(std.mem.sliceAsBytes(out[0 .. pairs * dims]));

        for (0..pairs) |pair| {
            const e: usize = @intCast(picks[pair]);
            const x_row = x_bits[pair / slots * dims ..][0..dims];
            for (0..width) |col| {
                const g = dot(gate.dense[(e * width + col) * dims ..][0..dims], x_row);
                const u = dot(up.dense[(e * width + col) * dims ..][0..dims], x_row);
                const want = g.value / (1.0 + @exp(-g.value)) * u.value;
                // three roundings to bf16 on the way, and silu's slope of at most 1.1 on the first
                const tolerance = 0.02 * @abs(g.value * u.value) + 1e-4 * (g.magnitude * @abs(u.value) + u.magnitude * @abs(g.value)) + 1e-6;
                try std.testing.expectApproxEqAbs(want, @as(f64, numeric.f32FromBf16(act_bits[pair * width + col])), tolerance);
            }
            // down is checked from the activation the device produced, so its tolerance is fp32's alone
            const act_row = act_bits[pair * width ..][0..width];
            for (0..dims) |col| {
                const d = dot(down.dense[(e * dims + col) * width ..][0..width], act_row);
                try std.testing.expectApproxEqAbs(d.value, @as(f64, out[pair * dims + col]), d.magnitude * 1e-5 + 1e-9);
            }
        }
        if (round == 0) {
            @memcpy(first_act, act_bits[0 .. slots * width]);
            @memcpy(first_out, out[0 .. slots * dims]);
        } else {
            try std.testing.expectEqualSlices(u16, first_act, act_bits[0 .. slots * width]);
            try std.testing.expectEqualSlices(u32, @ptrCast(first_out), @ptrCast(out[0 .. slots * dims]));
        }
    }
}

test "a wide window's experts keep TensorFold's bits, read once for 32 or 64 pairs or again for every 16" {
    const gpa = std.testing.allocator;
    const count = 24;
    const width = 64;
    const dims = 160;
    const slots = 3;
    const rows = 512;
    const pairs = rows * slots;
    var prng: std.Random.DefaultPrng = .init(21);
    const random = prng.random();

    const gate: TestMatrix = try .init(gpa, random, count, width, dims);
    defer gate.deinit(gpa);
    const up: TestMatrix = try .init(gpa, random, count, width, dims);
    defer up.deinit(gpa);
    const down: TestMatrix = try .init(gpa, random, count, dims, width);
    defer down.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var staging: Staging = .{};
    defer staging.deinit(gpa);
    var layer: Experts = try .init(gpa, &staging, count, width, dims, try gate.matrix(arena.allocator()), try up.matrix(arena.allocator()), try down.matrix(arena.allocator()));
    defer layer.deinit();

    const x_bits = try gpa.alloc(u16, rows * dims);
    defer gpa.free(x_bits);
    for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    // Expert 0 takes 301 rows (9 items of 32 pairs and one of 13, a lone tile), expert 1 the next 49 (an item of 32
    // and one of 17: a full tile and a tile of one pair), expert 2 the next 16, expert 3 one; the rest fall anywhere.
    const picks = try gpa.alloc(i32, pairs);
    defer gpa.free(picks);
    var held: [count]usize = @splat(0);
    for (0..rows) |row| {
        const first: usize = if (row < 301) 0 else if (row < 350) 1 else if (row < 366) 2 else if (row < 367) 3 else 4 + random.uintLessThan(usize, count - 4);
        for (0..slots) |slot| {
            const e = if (slot == 0) first else 4 + (first + slot * 7) % (count - 4);
            picks[row * slots + slot] = @intCast(e);
            held[e] += 1;
        }
    }
    try std.testing.expectEqual(301, held[0]);
    try std.testing.expectEqual(49, held[1]);
    try std.testing.expectEqual(16, held[2]);
    try std.testing.expectEqual(1, held[3]);

    var plan: Plan = try .init(rows, slots, count);
    defer plan.deinit();
    var x_dev: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
    defer x_dev.deinit();
    try x_dev.upload(std.mem.sliceAsBytes(x_bits));
    var picks_dev: cuda.Buffer = try .init(.i32, &.{ rows, slots });
    defer picks_dev.deinit();
    try picks_dev.upload(std.mem.sliceAsBytes(picks));
    var act_dev: cuda.Buffer = try .init(.bf16, &.{ pairs, width });
    defer act_dev.deinit();
    var out_dev: cuda.Buffer = try .init(.f32, &.{ pairs, dims });
    defer out_dev.deinit();

    const act_bits = try gpa.alloc(u16, pairs * width);
    defer gpa.free(act_bits);
    const out = try gpa.alloc(u32, pairs * dims);
    defer gpa.free(out);
    const want_act = try gpa.alloc(u16, pairs * width);
    defer gpa.free(want_act);
    const want_out = try gpa.alloc(u32, pairs * dims);
    defer gpa.free(want_out);

    // TensorFold's form first, then the window's own and the wide kernel as it would run; then the wide kernel
    // under items of 16 pairs, where every pass is a lone tile, and TensorFold's under items of 32, which it
    // reads twice.
    // Then an item of 64 pairs, the wide kernel's two passes of it by one warp; last experts_tile.cu's four tiles
    // under items of 64, 32 and 16 pairs, the rows past an item's pairs left stale in its shared memory.
    const passes = [_]?Pass{ .narrow, null, .wide, .{ .item = 16, .tiles = 2 }, .{ .item = 32, .tiles = 1 }, .{ .item = 64, .tiles = 2 }, .{ .item = 64, .tiles = 4 }, .{ .item = 32, .tiles = 4 }, .{ .item = 16, .tiles = 4 } };
    for (passes, 0..) |pass, round| {
        // what an earlier pass wrote must not stand in for this one's
        try act_dev.upload(std.mem.sliceAsBytes(act_bits));
        @memset(out, 0x7fc00000);
        try out_dev.upload(std.mem.sliceAsBytes(out));
        if (pass) |as| {
            try plan.routeAs(picks_dev.view, rows, as);
            try layer.gateUpAs(x_dev.view, plan, act_dev.view, rows, null, as);
            try layer.downProjectAs(act_dev.view, plan, out_dev.view, rows, null, as);
        } else {
            try plan.route(picks_dev.view, rows);
            try layer.gateUp(x_dev.view, plan, act_dev.view, rows, null);
            try layer.downProject(act_dev.view, plan, out_dev.view, rows, null);
        }
        try cuda.synchronize();
        try act_dev.download(std.mem.sliceAsBytes(act_bits));
        try out_dev.download(std.mem.sliceAsBytes(out));
        if (round == 0) {
            @memcpy(want_act, act_bits);
            @memcpy(want_out, out);
            @memset(act_bits, 0x7fc0);
            continue;
        }
        for (0..pairs) |pair| {
            try std.testing.expectEqualSlices(u16, want_act[pair * width ..][0..width], act_bits[pair * width ..][0..width]);
            try std.testing.expectEqualSlices(u32, want_out[pair * dims ..][0..dims], out[pair * dims ..][0..dims]);
        }
        @memset(act_bits, 0x7fc0);
    }

    // and the bits are the host's numbers: every pair of the crowded experts, and each 7th of the rest
    for (0..pairs) |pair| {
        const e: usize = @intCast(picks[pair]);
        if (e > 3 and pair % 7 != 0) continue;
        const x_row = x_bits[pair / slots * dims ..][0..dims];
        const act_row = want_act[pair * width ..][0..width];
        for (0..width) |col| {
            const g = dot(gate.dense[(e * width + col) * dims ..][0..dims], x_row);
            const u = dot(up.dense[(e * width + col) * dims ..][0..dims], x_row);
            const want = g.value / (1.0 + @exp(-g.value)) * u.value;
            const tolerance = 0.02 * @abs(g.value * u.value) + 1e-4 * (g.magnitude * @abs(u.value) + u.magnitude * @abs(g.value)) + 1e-6;
            try std.testing.expectApproxEqAbs(want, @as(f64, numeric.f32FromBf16(act_row[col])), tolerance);
        }
        for (0..dims) |col| {
            const d = dot(down.dense[(e * dims + col) * width ..][0..width], act_row);
            try std.testing.expectApproxEqAbs(d.value, @as(f64, @as(f32, @bitCast(want_out[pair * dims + col]))), d.magnitude * 1e-5 + 1e-9);
        }
    }
}

test "the experts' units in whole rounds keep the stride's bits, a row alone as among others" {
    const gpa = std.testing.allocator;
    // 20 units an expert either way: 96 experts are 1920, three rounds of gate and up's 768 warps and two of
    // down's 960; 48 are two rounds and one
    const count = 96;
    const width = 640;
    const dims = 640;
    const slots = 4;
    const rows = 24;
    var prng: std.Random.DefaultPrng = .init(73);
    const random = prng.random();

    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var matrices: [3]Matrix = undefined;
    for (&matrices) |*matrix| {
        const codes = try arena.allocator().alloc(u8, count * width * dims / 2);
        random.bytes(codes);
        const block_scales = try arena.allocator().alloc(u8, count * width * dims / 16);
        for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
        const scales = try arena.allocator().alloc(f32, count);
        for (scales) |*scale| scale.* = 0.01 + 0.02 * random.float(f32);
        matrix.* = .{ .codes = try Matrix.parts(arena.allocator(), codes, count), .block_scales = try Matrix.parts(arena.allocator(), block_scales, count), .scales = scales };
    }
    var staging: Staging = .{};
    defer staging.deinit(gpa);
    var layer: Experts = try .init(gpa, &staging, count, width, dims, matrices[0], matrices[1], matrices[2]);
    defer layer.deinit();

    const x_bits = try gpa.alloc(u16, rows * dims);
    defer gpa.free(x_bits);
    for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    // every pair its own expert, in no order
    const picks = try gpa.alloc(i32, rows * slots);
    defer gpa.free(picks);
    for (picks, 0..) |*pick, pair| pick.* = @intCast(pair * 37 % count);

    const Run = struct {
        act: []u16,
        out: []u32,

        fn of(allocator: Allocator, experts: Experts, x: []const u16, chosen: []const i32, m: usize) !@This() {
            const pairs = m * slots;
            var plan: Plan = try .init(m, slots, count);
            defer plan.deinit();
            var x_dev: cuda.Buffer = try .init(.bf16, &.{ m, dims });
            defer x_dev.deinit();
            try x_dev.upload(std.mem.sliceAsBytes(x));
            var picks_dev: cuda.Buffer = try .init(.i32, &.{ m, slots });
            defer picks_dev.deinit();
            try picks_dev.upload(std.mem.sliceAsBytes(chosen));
            var act_dev: cuda.Buffer = try .init(.bf16, &.{ pairs, width });
            defer act_dev.deinit();
            var out_dev: cuda.Buffer = try .init(.f32, &.{ pairs, dims });
            defer out_dev.deinit();
            try plan.route(picks_dev.view, m);
            try experts.gateUp(x_dev.view, plan, act_dev.view, m, null);
            try experts.downProject(act_dev.view, plan, out_dev.view, m, null);
            const act = try allocator.alloc(u16, pairs * width);
            try act_dev.download(std.mem.sliceAsBytes(act));
            const out = try allocator.alloc(u32, pairs * dims);
            try out_dev.download(std.mem.sliceAsBytes(out));
            return .{ .act = act, .out = out };
        }
    };

    const before = experts_file.even_units;
    defer experts_file.even_units = before;
    experts_file.even_units = false;
    const base: Run = try .of(arena.allocator(), layer, x_bits, picks, rows);
    var nonzero: usize = 0;
    for (base.out) |bits| nonzero += @intFromBool(bits << 1 != 0);
    try std.testing.expect(nonzero > base.out.len / 2);
    experts_file.even_units = true;
    for ([_]usize{ rows, 12, 10 }) |m| {
        const even: Run = try .of(arena.allocator(), layer, x_bits[0 .. m * dims], picks[0 .. m * slots], m);
        try std.testing.expectEqualSlices(u16, base.act[0 .. m * slots * width], even.act);
        try std.testing.expectEqualSlices(u32, base.out[0 .. m * slots * dims], even.out);
    }
    for ([_]usize{ 0, 7, 23 }) |row| {
        const alone: Run = try .of(arena.allocator(), layer, x_bits[row * dims ..][0..dims], picks[row * slots ..][0..slots], 1);
        try std.testing.expectEqualSlices(u16, base.act[row * slots * width ..][0 .. slots * width], alone.act);
        try std.testing.expectEqualSlices(u32, base.out[row * slots * dims ..][0 .. slots * dims], alone.out);
    }
    // spec 152 WS-09: the half kernel is on only where every half unit fits its resident warps. 640 wide is 20
    // column blocks a gate and up, and here every pair is its own expert: four rows are 16 items, 640 half units
    // on a 768-warp grid, so this call takes the half kernel while the runs above take the classic one's. A row
    // alone is the half path's too; the equality is a pair's bits held across either.
    {
        const half: Run = try .of(arena.allocator(), layer, x_bits[0 .. 4 * dims], picks[0 .. 4 * slots], 4);
        try std.testing.expectEqualSlices(u16, base.act[0 .. 4 * slots * width], half.act);
        try std.testing.expectEqualSlices(u32, base.out[0 .. 4 * slots * dims], half.out);
    }
}

/// x's group `group` of 16 dequantized as `numeric.nvfp4Group` codes it, without the global scale's division.
fn quantizedGroup(x_row: []const u16, group: usize, gs: f32, out: *[16]f64) struct { codes: u64, scale: u8 } {
    var values: [16]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = numeric.f32FromBf16(x_row[group * 16 + i]);
    const q = numeric.nvfp4Group(values, gs);
    const sf: f64 = numeric.f32FromE4m3(q.scale);
    for (out, 0..) |*v, i| v.* = nvfp4.e2m1[@intCast((q.codes >> @intCast(4 * i)) & 0xf)] * sf;
    return .{ .codes = q.codes, .scale = q.scale };
}

test "W4A4 experts quantize as the host, down's rows from the card's SwiGLU, and keep a pair's bits by any kernel or call" {
    const gpa = std.testing.allocator;
    const count = 24;
    const width = 128;
    const dims = 192;
    const slots = 3;
    var prng: std.Random.DefaultPrng = .init(11);
    const random = prng.random();

    const gate: TestMatrix = try .init(gpa, random, count, width, dims);
    defer gate.deinit(gpa);
    const up: TestMatrix = try .init(gpa, random, count, width, dims);
    defer up.deinit(gpa);
    const down: TestMatrix = try .init(gpa, random, count, dims, width);
    defer down.deinit(gpa);
    // input scales near a checkpoint's (|x| up to ~4 over 6 * 448); gate and up quantize by the layer's widest
    // (spec 115, the card's `a1.amin()`), down by its expert's, some of whose every scale saturates at 448 or
    // underflows to zero
    var inputs: [3][count]f32 = undefined;
    for (0..count) |e| {
        const base: f32 = 0.0015 * (0.5 + random.float(f32));
        inputs[0][e] = base;
        inputs[1][e] = if (e % 3 == 0) base * 0.75 else base;
        inputs[2][e] = switch (e % 6) {
            4 => 1e-6,
            5 => 1e3,
            else => 0.004 * (0.5 + random.float(f32)),
        };
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var matrices = [3]Matrix{ try gate.matrix(arena.allocator()), try up.matrix(arena.allocator()), try down.matrix(arena.allocator()) };
    for (&matrices, &inputs) |*m, *one| m.input_scales = one;
    var staging: Staging = .{};
    defer staging.deinit(gpa);
    var layer: Experts = try .initWith(gpa, &staging, count, width, dims, matrices[0], matrices[1], matrices[2], .nvfp4);
    defer layer.deinit();
    layer.limit = 10;

    // up to 1,200 rows: 150 pairs an expert, past an item of 128
    const row_counts = [_]usize{ 1, 2, 9, 40, 1200 };
    const max_rows = row_counts[row_counts.len - 1];
    const x_bits = try gpa.alloc(u16, max_rows * dims);
    defer gpa.free(x_bits);
    for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    // a row of zeros and a row of one large input among small ones
    @memset(x_bits[1 * dims ..][0..dims], 0);
    for (x_bits[2 * dims ..][0..dims], 0..) |*bits, i| bits.* = numeric.bf16FromF32(if (i == 5) 300 else 0.01 * random.floatNorm(f32));
    const picks = try gpa.alloc(i32, max_rows * slots);
    defer gpa.free(picks);
    for (0..max_rows) |row| {
        const first = random.uintLessThan(usize, count);
        for (0..slots) |slot| picks[row * slots + slot] = @intCast((first + slot * 5) % count);
    }

    const act_bits = try gpa.alloc(u16, max_rows * slots * width);
    defer gpa.free(act_bits);
    const out = try gpa.alloc(f32, max_rows * slots * dims);
    defer gpa.free(out);
    const first_act = try gpa.alloc(u16, slots * width);
    defer gpa.free(first_act);
    const first_out = try gpa.alloc(f32, slots * dims);
    defer gpa.free(first_out);
    const first_down = try gpa.alloc(u8, slots * (width / 2 + width / 16));
    defer gpa.free(first_down);
    const other_act = try gpa.alloc(u16, max_rows * slots * width);
    defer gpa.free(other_act);
    const other_out = try gpa.alloc(f32, max_rows * slots * dims);
    defer gpa.free(other_out);
    const codes = try gpa.alloc(u8, max_rows * slots * dims / 2);
    defer gpa.free(codes);
    const scales = try gpa.alloc(u8, max_rows * slots * dims / 16);
    defer gpa.free(scales);
    const members = try gpa.alloc(i32, max_rows * slots);
    defer gpa.free(members);
    const form_members = try gpa.alloc(i32, max_rows * slots);
    defer gpa.free(form_members);
    // down's rows by pair (not by plan position, which a form's items may order otherwise): codes, then scales
    const down_row = width / 2 + width / 16;
    const down_q = try gpa.alloc(u8, max_rows * slots * down_row);
    defer gpa.free(down_q);
    const other_down_q = try gpa.alloc(u8, max_rows * slots * down_row);
    defer gpa.free(other_down_q);
    const plan_codes = try gpa.alloc(u8, max_rows * slots * width / 2);
    defer gpa.free(plan_codes);
    const plan_scales = try gpa.alloc(u8, max_rows * slots * width / 16);
    defer gpa.free(plan_scales);
    var gs_up: [count]f32 = undefined;
    var gs_down: [count]f32 = undefined;
    var widest: f32 = 0;
    for (0..count) |e| widest = @max(widest, inputs[0][e], inputs[1][e]);
    for (0..count) |e| {
        gs_up[e] = 1.0 / widest;
        gs_down[e] = 1.0 / inputs[2][e];
    }

    var checked_round = false;
    for (row_counts, 0..) |rows, round| {
        const pairs = rows * slots;
        var plan: Plan = try .init(rows, slots, count);
        defer plan.deinit();
        var rows_q: experts_file.Quantized = try .init(pairs, dims, width);
        defer rows_q.deinit();
        var x_dev: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
        defer x_dev.deinit();
        try x_dev.upload(std.mem.sliceAsBytes(x_bits[0 .. rows * dims]));
        var picks_dev: cuda.Buffer = try .init(.i32, &.{ rows, slots });
        defer picks_dev.deinit();
        try picks_dev.upload(std.mem.sliceAsBytes(picks[0..pairs]));
        var act_dev: cuda.Buffer = try .init(.bf16, &.{ pairs, width });
        defer act_dev.deinit();
        var out_dev: cuda.Buffer = try .init(.f32, &.{ pairs, dims });
        defer out_dev.deinit();

        // narrow, items of 16, then the tile kernel over items of 64 and over items of 16, its 128 over items of
        // 128 and of 64, and narrow over items of 128: the same bits
        for ([_]Pass{ .narrow, .tile4, .{ .item = 16, .tiles = 4 }, experts_file.a4_tile8, .{ .item = 64, .tiles = 8 }, .{ .item = 128, .tiles = 1 } }, 0..) |pass, form| {
            try plan.routeAs(picks_dev.view, rows, pass);
            try layer.gateUpQuantizedAs(x_dev.view, plan, rows_q, act_dev.view, rows, null, pass);
            try cuda.synchronize();
            if (form == 0) {
                try rows_q.codes.download(codes[0 .. pairs * dims / 2]);
                try rows_q.scales.download(scales[0 .. pairs * dims / 16]);
                try plan.members.download(std.mem.sliceAsBytes(members[0..pairs]));
            }
            try plan.members.download(std.mem.sliceAsBytes(form_members[0..pairs]));
            try rows_q.down_codes.download(plan_codes[0 .. pairs * width / 2]);
            try rows_q.down_scales.download(plan_scales[0 .. pairs * width / 16]);
            const downs = if (form == 0) down_q else other_down_q;
            for (0..pairs) |at| {
                const pair: usize = @intCast(form_members[at]);
                @memcpy(downs[pair * down_row ..][0 .. width / 2], plan_codes[at * width / 2 ..][0 .. width / 2]);
                @memcpy(downs[pair * down_row + width / 2 ..][0 .. width / 16], plan_scales[at * width / 16 ..][0 .. width / 16]);
            }
            try layer.downProjectQuantizedAs(plan, rows_q, out_dev.view, rows, null, pass);
            try cuda.synchronize();
            const acts = if (form == 0) act_bits else other_act;
            const outs = if (form == 0) out else other_out;
            try act_dev.download(std.mem.sliceAsBytes(acts[0 .. pairs * width]));
            try out_dev.download(std.mem.sliceAsBytes(outs[0 .. pairs * dims]));
            if (form != 0) {
                try std.testing.expectEqualSlices(u16, act_bits[0 .. pairs * width], other_act[0 .. pairs * width]);
                try std.testing.expectEqualSlices(u8, down_q[0 .. pairs * down_row], other_down_q[0 .. pairs * down_row]);
                try std.testing.expectEqualSlices(u32, @ptrCast(out[0 .. pairs * dims]), @ptrCast(other_out[0 .. pairs * dims]));
            }
        }

        // down's rows are the host's quantization of the values the epilogue quantized, to the bit
        for (0..pairs) |pair| {
            const e: usize = @intCast(picks[pair]);
            const act_row = act_bits[pair * width ..][0..width];
            for (0..width / 16) |group| {
                var unused: [16]f64 = undefined;
                const want = quantizedGroup(act_row, group, gs_down[e], &unused);
                try std.testing.expectEqual(want.scale, down_q[pair * down_row + width / 2 + group]);
                try std.testing.expectEqual(want.codes, std.mem.readInt(u64, down_q[pair * down_row + group * 8 ..][0..8], .little));
            }
        }

        // the quantized rows, by plan position, are the host's to the bit
        for (0..pairs) |at| {
            const pair: usize = @intCast(members[at]);
            const e: usize = @intCast(picks[pair]);
            const x_row = x_bits[pair / slots * dims ..][0..dims];
            for (0..dims / 16) |group| {
                var unused: [16]f64 = undefined;
                const want = quantizedGroup(x_row, group, gs_up[e], &unused);
                try std.testing.expectEqual(want.scale, scales[at * dims / 16 + group]);
                try std.testing.expectEqual(want.codes, std.mem.readInt(u64, codes[at * dims / 2 + group * 8 ..][0..8], .little));
            }
        }

        // the sums against an f64 product of the quantized operands; once, on the widest call
        if (rows == max_rows) {
            checked_round = true;
            const xq = try gpa.alloc(f64, dims);
            defer gpa.free(xq);
            const aq = try gpa.alloc(f64, width);
            defer gpa.free(aq);
            // down's input by two host models: one bf16 rounding of the SwiGLU (the A16 form's) and the card's A4
            // prompt form's (`prefill_a4.py`: g and u rounded to bf16 and clipped, silu and the product each
            // rounded), each against the device's values and, quantized, its groups
            const model_rows = try gpa.alloc(u16, 2 * width);
            defer gpa.free(model_rows);
            var same_values: [2]usize = @splat(0);
            var same_groups: [2]usize = @splat(0);
            for (0..pairs) |pair| {
                const e: usize = @intCast(picks[pair]);
                const x_row = x_bits[pair / slots * dims ..][0..dims];
                for (0..dims / 16) |group| _ = quantizedGroup(x_row, group, gs_up[e], xq[group * 16 ..][0..16]);
                for (0..width) |col| {
                    var g: f64 = 0;
                    var u: f64 = 0;
                    var magnitude: f64 = 0;
                    for (0..dims) |i| {
                        const gw = gate.dense[(e * width + col) * dims + i];
                        const uw = up.dense[(e * width + col) * dims + i];
                        g += gw * xq[i];
                        u += uw * xq[i];
                        magnitude += @abs(gw * xq[i]) + @abs(uw * xq[i]);
                    }
                    g /= gs_up[e];
                    u /= gs_up[e];
                    magnitude /= gs_up[e];
                    const gc = @min(g, 10.0);
                    const uc = std.math.clamp(u, -10.0, 10.0);
                    // in fp32 and clipped, rounded to bf16 at the card's three points (each half an ulp: 2^-8)
                    const want = gc / (1.0 + @exp(-gc)) * uc;
                    const got: f64 = numeric.f32FromBf16(act_bits[pair * width + col]);
                    try std.testing.expect(std.math.isFinite(got));
                    const tolerance = 0.012 * @abs(want) + 2e-5 * magnitude * (1 + @abs(uc) + @abs(gc)) + 1e-6;
                    try std.testing.expectApproxEqAbs(want, got, tolerance);
                    model_rows[col] = numeric.bf16FromF32(@floatCast(want));
                    const gv = numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(g)));
                    const uv = numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(u)));
                    const gl = @min(gv, 10.0);
                    const ul = std.math.clamp(uv, -10.0, 10.0);
                    const silu = numeric.f32FromBf16(numeric.bf16FromF32(gl / (1.0 + @exp(-gl))));
                    model_rows[width + col] = numeric.bf16FromF32(silu * ul);
                    for (0..2) |m| same_values[m] += @intFromBool(model_rows[m * width + col] == act_bits[pair * width + col]);
                }
                for (0..2) |m| {
                    for (0..width / 16) |group| {
                        var unused: [16]f64 = undefined;
                        const want = quantizedGroup(model_rows[m * width ..][0..width], group, gs_down[e], &unused);
                        same_groups[m] += @intFromBool(want.scale == down_q[pair * down_row + width / 2 + group] and
                            want.codes == std.mem.readInt(u64, down_q[pair * down_row + group * 8 ..][0..8], .little));
                    }
                }
                const act_row = act_bits[pair * width ..][0..width];
                for (0..width / 16) |group| _ = quantizedGroup(act_row, group, gs_down[e], aq[group * 16 ..][0..16]);
                for (0..dims) |col| {
                    var d: f64 = 0;
                    var magnitude: f64 = 0;
                    for (0..width) |i| {
                        const term = down.dense[(e * dims + col) * width + i] * aq[i];
                        d += term;
                        magnitude += @abs(term);
                    }
                    const got: f64 = out[pair * dims + col];
                    try std.testing.expect(std.math.isFinite(got));
                    // stored bf16 as the card's FC2 (half an ulp, at most 2^-8 of the value), in fp32
                    try std.testing.expectEqual(0, @as(u32, @bitCast(out[pair * dims + col])) & 0xffff);
                    try std.testing.expectApproxEqAbs(d / gs_down[e], got, (magnitude * 1e-5 + 1e-9) / gs_down[e] + @abs(d / gs_down[e]) * 0x1p-8);
                }
            }
            const values: f64 = @floatFromInt(pairs * width);
            const groups: f64 = @floatFromInt(pairs * width / 16);
            const v = [2]f64{ @as(f64, @floatFromInt(same_values[0])) / values, @as(f64, @floatFromInt(same_values[1])) / values };
            const gr = [2]f64{ @as(f64, @floatFromInt(same_groups[0])) / groups, @as(f64, @floatFromInt(same_groups[1])) / groups };
            std.debug.print("A4DOWN down's input equal to one rounding's host model {d:.4} of values, {d:.4} of groups; to the card's {d:.4}, {d:.4}\n", .{ v[0], gr[0], v[1], gr[1] });
            try std.testing.expect(v[1] > 0.98 and v[1] > v[0] and gr[1] >= gr[0]);
        }

        // the first row's pairs alone as among others
        if (round == 0) {
            @memcpy(first_act, act_bits[0 .. slots * width]);
            @memcpy(first_down, down_q[0 .. slots * down_row]);
            @memcpy(first_out, out[0 .. slots * dims]);
        } else {
            try std.testing.expectEqualSlices(u16, first_act, act_bits[0 .. slots * width]);
            try std.testing.expectEqualSlices(u8, first_down, down_q[0 .. slots * down_row]);
            try std.testing.expectEqualSlices(u32, @ptrCast(first_out), @ptrCast(out[0 .. slots * dims]));
        }
    }
    try std.testing.expect(checked_round);
}

test "A16 on the W4A4 pack matches the host and keeps a pair's bits whatever shares its call" {
    const gpa = std.testing.allocator;
    const count = 24;
    const width = 128;
    const dims = 192;
    const slots = 3;
    var prng: std.Random.DefaultPrng = .init(13);
    const random = prng.random();

    const gate: TestMatrix = try .init(gpa, random, count, width, dims);
    defer gate.deinit(gpa);
    const up: TestMatrix = try .init(gpa, random, count, width, dims);
    defer up.deinit(gpa);
    const down: TestMatrix = try .init(gpa, random, count, dims, width);
    defer down.deinit(gpa);
    var inputs: [3][count]f32 = undefined;
    for (&inputs) |*one| for (one) |*scale| {
        scale.* = 0.002 * (0.5 + random.float(f32));
    };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var matrices = [3]Matrix{ try gate.matrix(arena.allocator()), try up.matrix(arena.allocator()), try down.matrix(arena.allocator()) };
    for (&matrices, &inputs) |*m, *one| m.input_scales = one;
    var staging: Staging = .{};
    defer staging.deinit(gpa);
    var layer: Experts = try .initWith(gpa, &staging, count, width, dims, matrices[0], matrices[1], matrices[2], .nvfp4);
    defer layer.deinit();
    layer.limit = 10;

    const row_counts = [_]usize{ 1, 9, 40 };
    const max_rows = row_counts[row_counts.len - 1];
    const x_bits = try gpa.alloc(u16, max_rows * dims);
    defer gpa.free(x_bits);
    for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    @memset(x_bits[1 * dims ..][0..dims], 0);
    const picks = try gpa.alloc(i32, max_rows * slots);
    defer gpa.free(picks);
    for (0..max_rows) |row| {
        const first = random.uintLessThan(usize, count);
        for (0..slots) |slot| picks[row * slots + slot] = @intCast((first + slot * 5) % count);
    }
    const act = try gpa.alloc(u16, max_rows * slots * width);
    defer gpa.free(act);
    const out = try gpa.alloc(f32, max_rows * slots * dims);
    defer gpa.free(out);
    var first_act: [slots * width]u16 = undefined;
    var first_out: [slots * dims]f32 = undefined;

    for (row_counts, 0..) |rows, round| {
        const pairs = rows * slots;
        var plan: Plan = try .init(rows, slots, count);
        defer plan.deinit();
        var x_dev: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
        defer x_dev.deinit();
        try x_dev.upload(std.mem.sliceAsBytes(x_bits[0 .. rows * dims]));
        var picks_dev: cuda.Buffer = try .init(.i32, &.{ rows, slots });
        defer picks_dev.deinit();
        try picks_dev.upload(std.mem.sliceAsBytes(picks[0..pairs]));
        var act_dev: cuda.Buffer = try .init(.bf16, &.{ pairs, width });
        defer act_dev.deinit();
        var out_dev: cuda.Buffer = try .init(.f32, &.{ pairs, dims });
        defer out_dev.deinit();
        try plan.routeAs(picks_dev.view, rows, .narrow);
        try layer.gateUpA16(x_dev.view, plan, act_dev.view, rows, null);
        try layer.downProjectA16(act_dev.view, plan, out_dev.view, rows, null);
        try cuda.synchronize();
        try act_dev.download(std.mem.sliceAsBytes(act[0 .. pairs * width]));
        try out_dev.download(std.mem.sliceAsBytes(out[0 .. pairs * dims]));

        // gate and up on the bf16 rows, the SwiGLU in fp32 rounded once; down on those rows
        var same: usize = 0;
        for (0..pairs) |pair| {
            const e: usize = @intCast(picks[pair]);
            const x_row = x_bits[pair / slots * dims ..][0..dims];
            for (0..width) |col| {
                const g = dot(gate.dense[(e * width + col) * dims ..][0..dims], x_row);
                const u = dot(up.dense[(e * width + col) * dims ..][0..dims], x_row);
                const gc = @min(g.value, 10.0);
                const uc = std.math.clamp(u.value, -10.0, 10.0);
                const want = gc / (1.0 + @exp(-gc)) * uc;
                const got: f64 = numeric.f32FromBf16(act[pair * width + col]);
                try std.testing.expect(std.math.isFinite(got));
                const tolerance = 0.004 * @abs(want) + 1e-5 * (g.magnitude + u.magnitude) * (1 + @abs(uc) + @abs(gc)) + 1e-6;
                try std.testing.expectApproxEqAbs(want, got, tolerance);
                same += @intFromBool(numeric.bf16FromF32(@floatCast(want)) == act[pair * width + col]);
            }
            for (0..dims) |col| {
                const d = dot(down.dense[(e * dims + col) * width ..][0..width], act[pair * width ..][0..width]);
                try std.testing.expectApproxEqAbs(d.value, out[pair * dims + col], d.magnitude * 1e-5 + 1e-9);
            }
        }
        try std.testing.expect(@as(f64, @floatFromInt(same)) > 0.95 * @as(f64, @floatFromInt(pairs * width)));

        // the first row's pairs alone as among others
        if (round == 0) {
            @memcpy(&first_act, act[0 .. slots * width]);
            @memcpy(&first_out, out[0 .. slots * dims]);
        } else {
            try std.testing.expectEqualSlices(u16, &first_act, act[0 .. slots * width]);
            try std.testing.expectEqualSlices(u32, @ptrCast(&first_out), @ptrCast(out[0 .. slots * dims]));
        }
    }
}

test "relu squared experts (one up matrix) match the host and keep a pair's bits by any pass or call" {
    const gpa = std.testing.allocator;
    const count = 24;
    const width = 64;
    const dims = 160;
    const slots = 4;
    var prng: std.Random.DefaultPrng = .init(151);
    const random = prng.random();

    const up: TestMatrix = try .init(gpa, random, count, width, dims);
    defer up.deinit(gpa);
    const down: TestMatrix = try .init(gpa, random, count, dims, width);
    defer down.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var staging: Staging = .{};
    defer staging.deinit(gpa);
    var layer: Experts = try .initRelu2(gpa, &staging, count, width, dims, try up.matrix(arena.allocator()), try down.matrix(arena.allocator()));
    defer layer.deinit();
    try std.testing.expect(!layer.gated);

    const row_counts = [_]usize{ 1, 2, 16, 17, 300 };
    const max_rows = row_counts[row_counts.len - 1];
    const x_bits = try gpa.alloc(u16, max_rows * dims);
    defer gpa.free(x_bits);
    for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    const picks = try gpa.alloc(i32, max_rows * slots);
    defer gpa.free(picks);
    for (0..max_rows) |row| {
        const first = random.uintLessThan(usize, count);
        for (0..slots) |slot| picks[row * slots + slot] = @intCast((first + slot * 5) % count);
    }
    const act_bits = try gpa.alloc(u16, max_rows * slots * width);
    defer gpa.free(act_bits);
    const out = try gpa.alloc(f32, max_rows * slots * dims);
    defer gpa.free(out);
    const first_act = try gpa.alloc(u16, slots * width);
    defer gpa.free(first_act);
    const first_out = try gpa.alloc(f32, slots * dims);
    defer gpa.free(first_out);

    var first = true;
    for (row_counts) |rows| for ([_]Pass{ .narrow, .wide, .tile4 }) |pass| {
        const pairs = rows * slots;
        var plan: Plan = try .init(rows, slots, count);
        defer plan.deinit();
        var x_dev: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
        defer x_dev.deinit();
        try x_dev.upload(std.mem.sliceAsBytes(x_bits[0 .. rows * dims]));
        var picks_dev: cuda.Buffer = try .init(.i32, &.{ rows, slots });
        defer picks_dev.deinit();
        try picks_dev.upload(std.mem.sliceAsBytes(picks[0..pairs]));
        var act_dev: cuda.Buffer = try .init(.bf16, &.{ pairs, width });
        defer act_dev.deinit();
        var out_dev: cuda.Buffer = try .init(.f32, &.{ pairs, dims });
        defer out_dev.deinit();

        try plan.routeAs(picks_dev.view, rows, pass);
        try layer.gateUpAs(x_dev.view, plan, act_dev.view, rows, null, pass);
        try layer.downProjectAs(act_dev.view, plan, out_dev.view, rows, null, pass);
        try cuda.synchronize();
        try act_dev.download(std.mem.sliceAsBytes(act_bits[0 .. pairs * width]));
        try out_dev.download(std.mem.sliceAsBytes(out[0 .. pairs * dims]));

        for (0..pairs) |pair| {
            const e: usize = @intCast(picks[pair]);
            const x_row = x_bits[pair / slots * dims ..][0..dims];
            for (0..width) |col| {
                const u = dot(up.dense[(e * width + col) * dims ..][0..dims], x_row);
                const r = @max(u.value, 0);
                // bf16(u) then bf16 of its square: two roundings, and fp32's sum error doubled by the square
                const tolerance = 0.012 * r * r + 2 * (@abs(u.value) + 1) * u.magnitude * 1e-5 + 1e-9;
                try std.testing.expectApproxEqAbs(r * r, @as(f64, numeric.f32FromBf16(act_bits[pair * width + col])), tolerance);
                if (u.value < -u.magnitude * 1e-5) try std.testing.expectEqual(@as(u16, 0), act_bits[pair * width + col] & 0x7fff);
            }
            const act_row = act_bits[pair * width ..][0..width];
            for (0..dims) |col| {
                const d = dot(down.dense[(e * dims + col) * width ..][0..width], act_row);
                try std.testing.expectApproxEqAbs(d.value, @as(f64, out[pair * dims + col]), d.magnitude * 1e-5 + 1e-9);
            }
        }
        if (first) {
            @memcpy(first_act, act_bits[0 .. slots * width]);
            @memcpy(first_out, out[0 .. slots * dims]);
            first = false;
        } else {
            try std.testing.expectEqualSlices(u16, first_act, act_bits[0 .. slots * width]);
            try std.testing.expectEqualSlices(u32, @ptrCast(first_out), @ptrCast(out[0 .. slots * dims]));
        }
    };
}

test "W4A4 relu2 experts match the host, zero padded K, and keep row bits across kernels" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const count = 3;
    const dims = 128;
    const width = 96; // TP2's 928-wide down projection has the same half-k64 tail.
    const padded = 128;
    const rows = 129;
    const slots = 2;
    const pairs = rows * slots;
    var prng = std.Random.DefaultPrng.init(15104);
    const random = prng.random();
    const up = try TestMatrix.init(gpa, random, count, width, dims);
    defer up.deinit(gpa);
    const down = try TestMatrix.init(gpa, random, count, dims, width);
    defer down.deinit(gpa);
    var up_matrix = try up.matrix(a);
    var down_matrix = try down.matrix(a);
    up_matrix.input_scales = &.{ 0.01, 0.02, 0.03 };
    down_matrix.input_scales = &.{ 0.003, 0.007, 0.011 };
    var staging: Staging = .{};
    defer staging.deinit(gpa);
    var layer = try Experts.initRelu2Quantized(gpa, &staging, count, width, dims, up_matrix, down_matrix);
    defer layer.deinit();
    const x = try a.alloc(u16, rows * dims);
    for (x) |*v| v.* = numeric.bf16FromF32(random.floatNorm(f32));
    const picks = try a.alloc(i32, pairs);
    for (picks, 0..) |*v, i| v.* = @intCast((i / slots + i % slots) % count);
    var x_dev = try cuda.Buffer.init(.bf16, &.{ rows, dims });
    defer x_dev.deinit();
    try x_dev.upload(std.mem.sliceAsBytes(x));
    var picks_dev = try cuda.Buffer.init(.i32, &.{ rows, slots });
    defer picks_dev.deinit();
    try picks_dev.upload(std.mem.sliceAsBytes(picks));
    var act_dev = try cuda.Buffer.init(.bf16, &.{ pairs, width });
    defer act_dev.deinit();
    var out_dev = try cuda.Buffer.init(.f32, &.{ pairs, dims });
    defer out_dev.deinit();
    var plan = try Plan.init(rows, slots, count);
    defer plan.deinit();
    var q = try experts_file.Quantized.init(pairs, dims, width);
    defer q.deinit();
    const act = try a.alloc(u16, pairs * width);
    const out = try a.alloc(f32, pairs * dims);
    const baseline = try a.alloc(f32, pairs * dims);
    const baseline_act = try a.alloc(u16, pairs * width);
    const members = try a.alloc(i32, pairs);
    const down_codes = try a.alloc(u8, pairs * padded / 2);
    const down_scales = try a.alloc(u8, pairs * padded / 16);
    for ([_]Pass{ .narrow, .tile4, experts_file.a4_tile8 }, 0..) |pass, form| {
        try plan.routeAs(picks_dev.view, rows, pass);
        try layer.gateUpQuantizedAs(x_dev.view, plan, q, act_dev.view, rows, null, pass);
        try layer.downProjectQuantizedAs(plan, q, out_dev.view, rows, null, pass);
        try act_dev.download(std.mem.sliceAsBytes(act));
        try out_dev.download(std.mem.sliceAsBytes(out));
        if (form == 0) {
            @memcpy(baseline, out);
            @memcpy(baseline_act, act);
        } else {
            try std.testing.expectEqualSlices(f32, baseline, out);
            try std.testing.expectEqualSlices(u16, baseline_act, act);
        }
        try plan.members.download(std.mem.sliceAsBytes(members));
        try q.down_codes.download(down_codes);
        try q.down_scales.download(down_scales);
        for (members, 0..) |member, at| {
            const pair: usize = @intCast(member);
            const e: usize = @intCast(picks[pair]);
            const act_row = act[pair * width ..][0..width];
            for (0..width / 16) |group| {
                var unused: [16]f64 = undefined;
                const want = quantizedGroup(act_row, group, 1.0 / down_matrix.input_scales[e], &unused);
                try std.testing.expectEqual(want.scale, down_scales[at * padded / 16 + group]);
                try std.testing.expectEqual(want.codes, std.mem.readInt(u64, down_codes[at * padded / 2 + group * 8 ..][0..8], .little));
            }
            for (down_codes[at * padded / 2 + width / 2 ..][0 .. (padded - width) / 2]) |v| try std.testing.expectEqual(@as(u8, 0), v);
            for (down_scales[at * padded / 16 + width / 16 ..][0 .. (padded - width) / 16]) |v| try std.testing.expectEqual(@as(u8, 0), v);
        }
    }
    for (0..pairs) |pair| {
        const e: usize = @intCast(picks[pair]);
        const gs_up = 1.0 / up_matrix.input_scales[e];
        const gs_down = 1.0 / down_matrix.input_scales[e];
        var xq: [dims]f64 = undefined;
        var aq: [width]f64 = undefined;
        for (0..dims / 16) |group| _ = quantizedGroup(x[pair / slots * dims ..][0..dims], group, gs_up, xq[group * 16 ..][0..16]);
        for (0..width) |col| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (0..dims) |k| {
                const term = up.dense[(e * width + col) * dims + k] * xq[k] / gs_up;
                sum += term;
                magnitude += @abs(term);
            }
            const r = @max(0, numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(sum))));
            const want = numeric.f32FromBf16(numeric.bf16FromF32(r * r));
            try std.testing.expectApproxEqAbs(@as(f64, want), @as(f64, numeric.f32FromBf16(act[pair * width + col])), 0.01 * @abs(want) + 2e-5 * magnitude * (1 + @abs(sum)) + 1e-9);
        }
        for (0..width / 16) |group| _ = quantizedGroup(act[pair * width ..][0..width], group, gs_down, aq[group * 16 ..][0..16]);
        for (0..dims) |col| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (0..width) |k| {
                const term = down.dense[(e * dims + col) * width + k] * aq[k] / gs_down;
                sum += term;
                magnitude += @abs(term);
            }
            try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(out[pair * dims + col])) & 0xffff);
            try std.testing.expectApproxEqAbs(sum, @as(f64, out[pair * dims + col]), magnitude * 1e-5 + @abs(sum) * 0x1p-8 + 1e-9);
        }
    }
    try plan.routeAs(picks_dev.span(37, 1), 1, .narrow);
    try layer.gateUpQuantizedAs(x_dev.span(37, 1), plan, q, act_dev.view, 1, null, .narrow);
    try layer.downProjectQuantizedAs(plan, q, out_dev.rows(slots), 1, null, .narrow);
    const alone = try a.alloc(f32, slots * dims);
    try cuda.download(out_dev.rows(slots), std.mem.sliceAsBytes(alone));
    try std.testing.expectEqualSlices(f32, baseline[37 * slots * dims ..][0 .. slots * dims], alone);
}
