//! The NVFP4 kernels the published checkpoint launches (`nvfp4/experts.cu`: the routed experts' gate and up, then
//! down; `nvfp4/qmmf.cu`: the draft head) alone at Flash Next's shapes, on random weights: what a profiler is
//! given (spec 073 WS-04); the EXL3 pack's twin retired with it (tag `retired-checkpoints-2026-10-21`). No checkpoint and no model.
//!
//! `ZSPARK_NVFP4_PROFILE` names a case (`experts:4:27`, `experts:1:6:hot`, `experts:8:40:even`, `head:1`, `lane`, `lanewide`, `glmlinears`, `glmdecode`, `glma8`), a set
//! (`rounds`: the distinct experts a round of 1 to 16 rows routes, as counted in the pack's rounds; `sweep`: 8 and
//! 16 rows over 30 to 100 distinct experts, where the units pass the grid's warps; `head`) or `all`. Unset, the
//! test is a smoke on 16 experts and a head of 12,800 rows (one K slice, as the draft head), one pass, for a tier.
//!
//! An experts case is ROWS rows of ten slots routed to DISTINCT experts of 512 (a launch's time follows the
//! experts it reads, not its rows): the plan, gate and up, then down, as a layer launches them, each launch on
//! the experts after the last one's, so none is in L2 (24 MB) when it is read. `hot` routes every row of every
//! launch to the same six, which L2 holds (11.1 MB for gate and up with 5.5 for down): what a warp's chain over
//! K takes without DRAM, against `experts:1:6` (the set `chain`). `even` is the launch with
//! `experts.even_units`, which the wide kernel of the set `window` (a prompt's windows of 768 to 2048 rows) does not read.
//! `tile` and `wide` force `Pass.tile4` and `Pass.wide` (`experts_tile.cu`, `experts_wide.cu`) whatever the rows.
//! The set `skew` is a window as prose routes it (400 experts, some with hundreds of pairs); `again` gives an
//! expert's 64 pairs to one warp, twice over its weights (spec 087 WS-04: slower than two warps at once). 64 MB are allocated after the weights (spec 071 WS-10).
//! The set `glm` is GLM FP4-Local's decoding rows and prompt windows (with `ZSPARK_NVFP4_SHAPE=glm`), `glm:a4` the
//! same in W4A4, `glm:a4narrow`, `glm:a4tile` and `glm:a4tile8` its kernels forced (spec 114 WS-03).
//!
//!   ZSPARK_NVFP4_PROFILE=experts:1:10 ncu --kernel-name regex:nvfp4_expert --launch-skip 32 --launch-count 1 TEST
//!   ZSPARK_NVFP4_PROFILE=rounds nsys profile --trace=cuda --sample=none --cpuctxsw=none TEST
//!
//! Each case prints its launches a pass and the weight bytes one launch reads.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../src/cuda/cuda.zig");
const experts_file = @import("../../src/weights/experts.zig");
const numeric = @import("../../src/weights/numeric.zig");
const nvfp4 = @import("../../src/weights/nvfp4.zig");
const profile_fixture = @import("../weights/profile_fixture.zig");
const nowNs = profile_fixture.nowNs;
const expectRan = profile_fixture.expectRan;

/// `a4`, `a4narrow`, `a4tile` and `a4tile8` (128 pairs a reading): the W4A4 form (spec 114 WS-03: `experts_quant.cu`, then `experts_a4.cu` or
/// `experts_tile_a4.cu`, down's rows made in gate and up's epilogue since WS-07), by `Activations.passOf` or forced, the call's quantization included.
/// `a16`: bf16 rows on the W4A4 pack (`gateUpA16`, `experts_a4.cu`'s A16 kernel), a step's form there (spec 143 WS-06).
const Form = enum { base, hot, even, again, tile, wide, a4, a4narrow, a4tile, a4tile8, a16 };

/// Flash Next's routed experts (`config.json`): 512 of 640 by 2560, ten a token. `ZSPARK_NVFP4_SHAPE=glm`: one
/// rank's of GLM-5.3-Flash FP4-Local, 288 of 1024 (half of 2048) by 4096, eight a token (spec 107 WS-03).
var width: usize = 640;
var dims: usize = 2560;
var slots: usize = 10;
var model_experts: usize = 512;
const most_experts = 512;
/// The draft vocabulary's rows of the head (`models/qwen4/draft_vocab.txt`).
const head_rows = 79591;

const Routed = struct { rows: usize, distinct: usize };

/// Distinct experts a round routes by its rows (spec 073 WS-02, counted in the pack's rounds).
const round_cases = [_]Routed{
    .{ .rows = 1, .distinct = 10 }, .{ .rows = 2, .distinct = 17 }, .{ .rows = 3, .distinct = 22 },
    .{ .rows = 4, .distinct = 27 }, .{ .rows = 8, .distinct = 46 }, .{ .rows = 16, .distinct = 72 },
};
/// Around the grid's warps: gate and up have 768 (38.4 experts of 20 units), down 960 (12 of 80).
const sweep_cases = [_]Routed{
    .{ .rows = 8, .distinct = 30 },  .{ .rows = 8, .distinct = 36 },  .{ .rows = 8, .distinct = 38 },
    .{ .rows = 8, .distinct = 40 },  .{ .rows = 8, .distinct = 42 },  .{ .rows = 8, .distinct = 44 },
    .{ .rows = 8, .distinct = 48 },  .{ .rows = 8, .distinct = 50 },  .{ .rows = 8, .distinct = 56 },
    .{ .rows = 8, .distinct = 64 },  .{ .rows = 16, .distinct = 60 }, .{ .rows = 16, .distinct = 76 },
    .{ .rows = 16, .distinct = 80 }, .{ .rows = 16, .distinct = 90 }, .{ .rows = 16, .distinct = 100 },
};
/// One row on the experts `hot` reads, past L2: against `hot`, what DRAM adds to a warp's chain over K.
/// A prompt's windows, wide from `Pass.wide_rows` (`experts_wide.cu`): every expert routed, 2048 rows a served
/// window's (spec 073 WS-09: its units in whole rounds were no faster at 768 to 1536 rows and 1 to 3% slower at 2048,
/// some 35 rounds on the grid, so the kernel keeps its stride).
const window_cases = [_]Routed{
    .{ .rows = 768, .distinct = 512 },  .{ .rows = 1024, .distinct = 512 }, .{ .rows = 1536, .distinct = 512 },
    .{ .rows = 2048, .distinct = 512 }, .{ .rows = 2048, .distinct = 300 }, .{ .rows = 1000, .distinct = 400 },
    .{ .rows = 4096, .distinct = 512 },
};
/// A window as prose routes it (spec 087 WS-00's tally on the pack: 353 experts of 512 at 1,024 rows, some with
/// hundreds of pairs): `distinct` 0 is `routeSkewed`.
const skew_cases = [_]Routed{ .{ .rows = 4096, .distinct = 0 }, .{ .rows = 2048, .distinct = 0 }, .{ .rows = 1024, .distinct = 0 }, .{ .rows = 512, .distinct = 0 } };
const chain_cases = [_]Routed{.{ .rows = 1, .distinct = hot_experts }};
/// GLM FP4-Local's (`ZSPARK_NVFP4_SHAPE=glm`, spec 114 WS-03): decoding's 1 to 6 rows of eight slots, about the
/// experts they route, and a prompt's windows of 512 and 2048 rows, every expert routed or as prose skews them.
const glm_cases = [_]Routed{
    .{ .rows = 1, .distinct = 8 },     .{ .rows = 2, .distinct = 15 },     .{ .rows = 3, .distinct = 21 },
    .{ .rows = 4, .distinct = 27 },    .{ .rows = 5, .distinct = 32 },     .{ .rows = 6, .distinct = 37 },
    .{ .rows = 512, .distinct = 288 }, .{ .rows = 2048, .distinct = 288 }, .{ .rows = 512, .distinct = 0 },
    .{ .rows = 2048, .distinct = 0 },
};
/// Experts a `hot` launch reads: 11.1 MB of gate and up and 5.5 of down, which L2 holds together.
const hot_experts = 6;
const head_cases = [_]usize{ 1, 2, 4, 16 };

const rounds = 9;
const launches = 16;

fn randomRows(gpa: Allocator, random: std.Random, rows: usize, k: usize) !cuda.Buffer {
    const host = try gpa.alloc(u16, rows * k);
    defer gpa.free(host);
    for (host) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    var x: cuda.Buffer = try .init(.bf16, &.{ rows, k });
    errdefer x.deinit();
    try x.upload(std.mem.sliceAsBytes(host));
    return x;
}

/// One expert's random projection, which every expert of the layer then is: the bytes differ by address only,
/// which is all a timing asks.
const OneMatrix = struct {
    codes: []u8,
    block_scales: []u8,
    scales: []f32,
    code_parts: [][]const u8,
    scale_parts: [][]const u8,

    fn init(gpa: Allocator, random: std.Random, count: usize, n: usize, k: usize) !OneMatrix {
        const codes = try gpa.alloc(u8, n * k / 2);
        errdefer gpa.free(codes);
        random.bytes(codes);
        const block_scales = try gpa.alloc(u8, n * k / 16);
        errdefer gpa.free(block_scales);
        for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
        const scales = try gpa.alloc(f32, count);
        errdefer gpa.free(scales);
        for (scales) |*scale| scale.* = 0.01 + 0.02 * random.float(f32);
        const code_parts = try gpa.alloc([]const u8, count);
        errdefer gpa.free(code_parts);
        @memset(code_parts, codes);
        const scale_parts = try gpa.alloc([]const u8, count);
        @memset(scale_parts, block_scales);
        return .{ .codes = codes, .block_scales = block_scales, .scales = scales, .code_parts = code_parts, .scale_parts = scale_parts };
    }

    fn deinit(self: OneMatrix, gpa: Allocator) void {
        gpa.free(self.codes);
        gpa.free(self.block_scales);
        gpa.free(self.scales);
        gpa.free(self.code_parts);
        gpa.free(self.scale_parts);
    }

    fn matrix(self: OneMatrix) experts_file.Matrix {
        return .{ .codes = self.code_parts, .block_scales = self.scale_parts, .scales = self.scales };
    }
};

fn makeLayer(gpa: Allocator, random: std.Random, count: usize) !experts_file.Experts {
    return makeLayerAs(gpa, random, count, .bf16);
}

fn makeLayerAs(gpa: Allocator, random: std.Random, count: usize, activations: experts_file.Activations) !experts_file.Experts {
    const gate: OneMatrix = try .init(gpa, random, count, width, dims);
    defer gate.deinit(gpa);
    const up: OneMatrix = try .init(gpa, random, count, width, dims);
    defer up.deinit(gpa);
    const down: OneMatrix = try .init(gpa, random, count, dims, width);
    defer down.deinit(gpa);
    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    // activation scales of a checkpoint's size: |x| of a few over 6 * 448
    const inputs = try gpa.alloc(f32, count);
    defer gpa.free(inputs);
    @memset(inputs, 0.0015);
    var matrices = [3]experts_file.Matrix{ gate.matrix(), up.matrix(), down.matrix() };
    for (&matrices) |*m| m.input_scales = inputs;
    return .initWith(gpa, &staging, count, width, dims, matrices[0], matrices[1], matrices[2], activations);
}

/// Launch `at`'s picks: row r takes ten experts in a row of the launch's `distinct`, the rows' tens spread so
/// that together they are all of them.
fn route(picks: []i32, case: Routed, first: usize, count: usize) void {
    std.debug.assert(case.distinct <= count and (case.distinct <= slots or case.distinct - slots <= slots * (case.rows - 1)));
    for (0..case.rows) |row| {
        // fewer experts than slots: a row's pairs share them
        const shift = if (case.rows == 1 or case.distinct <= slots) 0 else row * (case.distinct - slots) / (case.rows - 1);
        for (0..slots) |slot| picks[row * slots + slot] = @intCast((first + shift + slot % case.distinct) % count);
    }
}

/// A row's ten distinct experts drawn with a skew (an expert's weight falls as the cube of a uniform draw over
/// 400 of them), each launch on other experts; prints the experts touched and the readings at 32 and 64 pairs.
fn routeSkewed(picks: []i32, rows: usize, first: usize, count: usize, say: bool) void {
    var prng: std.Random.DefaultPrng = .init(87 + first);
    const random = prng.random();
    var held: [most_experts]usize = @splat(0);
    for (0..rows) |row| {
        for (0..slots) |slot| {
            pick: while (true) {
                const u = random.float(f64);
                const e = (first + @as(usize, @intFromFloat(400.0 * u * u * u))) % count;
                for (picks[row * slots ..][0..slot]) |other| if (other == @as(i32, @intCast(e))) continue :pick;
                picks[row * slots + slot] = @intCast(e);
                held[e] += 1;
                break;
            }
        }
    }
    if (!say) return;
    var distinct: usize = 0;
    var by32: usize = 0;
    var by64: usize = 0;
    var by128: usize = 0;
    var most: usize = 0;
    for (held[0..count]) |pairs| {
        distinct += @intFromBool(pairs > 0);
        by32 += (pairs + 31) / 32;
        by64 += (pairs + 63) / 64;
        by128 += (pairs + 127) / 128;
        most = @max(most, pairs);
    }
    std.debug.print("PROFILE skew {d} rows: {d} experts, {d} readings of 32 pairs, {d} of 64, {d} of 128, the most pairs {d}\n", .{ rows, distinct, by32, by64, by128, most });
}

fn runExperts(gpa: Allocator, random: std.Random, layer: experts_file.Experts, case: Routed, form: Form, full: bool) !void {
    const routed: Routed = if (form == .hot) .{ .rows = case.rows, .distinct = hot_experts } else case;
    const passes: usize = if (full) launches else 1;
    const host_picks = try gpa.alloc(i32, case.rows * slots);
    defer gpa.free(host_picks);
    const picks = try gpa.alloc(cuda.Buffer, passes);
    defer gpa.free(picks);
    var made: usize = 0;
    defer for (picks[0..made]) |*buffer| buffer.deinit();
    for (picks, 0..) |*buffer, at| {
        if (routed.distinct == 0) routeSkewed(host_picks, routed.rows, at * 97, layer.count, at == 0) else route(host_picks, routed, if (form == .hot) 0 else at * routed.distinct, layer.count);
        buffer.* = try .init(.i32, &.{ case.rows, slots });
        made += 1;
        try buffer.upload(std.mem.sliceAsBytes(host_picks));
    }
    var plan: experts_file.Plan = try .init(case.rows, slots, layer.count);
    defer plan.deinit();
    var x = try randomRows(gpa, random, case.rows, dims);
    defer x.deinit();
    var act: cuda.Buffer = try .init(.bf16, &.{ case.rows * slots, width });
    defer act.deinit();
    var out: cuda.Buffer = try .init(.f32, &.{ case.rows * slots, dims });
    defer out.deinit();
    // 64 MB that nothing reads, after the weights: the pages the kernel zeroed last are then these
    var tail: ?cuda.Buffer = if (full) try .init(.u8, &.{64 << 20}) else null;
    defer if (tail) |*buffer| buffer.deinit();
    const a4 = form == .a4 or form == .a4narrow or form == .a4tile or form == .a4tile8;
    const a16 = form == .a16;
    var rows_q: ?experts_file.Quantized = if (a4) try .init(case.rows * slots, dims, width) else null;
    defer if (rows_q) |*one| one.deinit();

    const before = experts_file.even_units;
    defer experts_file.even_units = before;
    experts_file.even_units = form == .even;
    var times: [rounds]u64 = undefined;
    for (if (full) times[0..] else times[0..1]) |*time| {
        try cuda.synchronize();
        const t0 = nowNs();
        for (picks) |buffer| {
            // `again`: an expert's second 32 pairs by the warp that read its first, straight after (spec 087 WS-04).
            if (a16) {
                try plan.routeAs(buffer.view, case.rows, .narrow);
                try layer.gateUpA16(x.view, plan, act.view, case.rows, null);
                try layer.downProjectA16(act.view, plan, out.view, case.rows, null);
                continue;
            }
            if (a4) {
                const pass: experts_file.Pass = switch (form) {
                    .a4narrow => .narrow,
                    .a4tile => .tile4,
                    .a4tile8 => experts_file.a4_tile8,
                    else => experts_file.Activations.passOf(case.rows),
                };
                try plan.routeAs(buffer.view, case.rows, pass);
                try layer.gateUpQuantizedAs(x.view, plan, rows_q.?, null, case.rows, null, pass);
                try layer.downProjectQuantizedAs(plan, rows_q.?, out.view, case.rows, null, pass);
                continue;
            }
            const pass: experts_file.Pass = if (form == .again) .{ .item = 64, .tiles = 2 } else if (form == .tile) .tile4 else if (form == .wide) .wide else .of(case.rows);
            try plan.routeAs(buffer.view, case.rows, pass);
            try layer.gateUpAs(x.view, plan, act.view, case.rows, null, pass);
            try layer.downProjectAs(act.view, plan, out.view, case.rows, null, pass);
        }
        try cuda.synchronize();
        time.* = nowNs() - t0;
    }
    if (!a4) try expectRan(gpa, u16, act.view);
    try expectRan(gpa, f32, out.view);
    if (!full) return;
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    // an expert's blocks: 576 B a (32 columns, 32 inputs), gate and up, then down
    const expert_bytes = (width / 32) * (dims / 32) * 576;
    std.debug.print("PROFILE experts:{d}:{d}:{t}: {d} launches a pass, {d} weight bytes a gate+up launch, {d} a down; plan, gate+up and down by the loop {d:.1} us median\n", .{
        case.rows,                          case.distinct,                  form,                                                                               passes,
        2 * expert_bytes * routed.distinct, expert_bytes * routed.distinct, @as(f64, @floatFromInt(times[rounds / 2])) / 1e3 / @as(f64, @floatFromInt(passes)),
    });
}

fn runHead(gpa: Allocator, random: std.Random, n: usize, rows: usize, full: bool) !void {
    const codes = try gpa.alloc(u8, n * dims / 2);
    defer gpa.free(codes);
    random.bytes(codes);
    const block_scales = try gpa.alloc(u8, n * dims / 16);
    defer gpa.free(block_scales);
    for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
    var linear: nvfp4.Fp4Linear = try .init(gpa, codes, block_scales, n, dims, 0.02);
    defer linear.deinit();
    var x = try randomRows(gpa, random, rows, dims);
    defer x.deinit();
    var out: cuda.Buffer = try .init(.f32, &.{ rows, n });
    defer out.deinit();
    var tail: ?cuda.Buffer = if (full) try .init(.u8, &.{64 << 20}) else null;
    defer if (tail) |*buffer| buffer.deinit();
    var times: [rounds]u64 = undefined;
    for (if (full) times[0..] else times[0..1]) |*time| {
        try cuda.synchronize();
        const t0 = nowNs();
        for (0..if (full) launches else 1) |_| try linear.forward(x.view, out.view);
        try cuda.synchronize();
        time.* = nowNs() - t0;
    }
    try expectRan(gpa, f32, out.view);
    if (!full) return;
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    std.debug.print("PROFILE head:{d}:base: {d} launches a pass, {d} weight bytes a launch; a launch by the loop {d:.1} us median\n", .{
        rows, launches, linear.npad * dims / 2 + linear.npad * dims / 16, @as(f64, @floatFromInt(times[rounds / 2])) / 1e3 / launches,
    });
}

/// A lane matmul split in K at the decoding rows (`lane`, spec 105 WS-03): FP4-Local's DeltaNet out_proj (MXFP8,
/// 2560 x 6144, 40 tiles of 8 slices) and the same shape in NVFP4, by the cluster and by one block a tile summing
/// its slices (`nvfp4.lane_fused_tiles`). Four copies of the weight take turns, so none is in L2 when read; the two
/// forms' outputs are compared bit for bit.
const lane_shape = .{ .n = 2560, .k = 6144 };
const lane_rows = [_]usize{ 1, 2, 4, 8, 16, 24 };
const lane_copies = 4;

fn LaneOf(comptime T: type) type {
    return struct {
        copies: [lane_copies]T,

        fn init(gpa: Allocator, random: std.Random) !@This() {
            const n = lane_shape.n;
            const k = lane_shape.k;
            var self: @This() = undefined;
            var made: usize = 0;
            errdefer for (self.copies[0..made]) |*one| one.deinit();
            for (&self.copies) |*one| {
                if (T == nvfp4.Mx8Linear) {
                    const codes = try gpa.alloc(u8, n * k);
                    defer gpa.free(codes);
                    random.bytes(codes);
                    // no NaN (S.1111.111)
                    for (codes) |*byte| if (byte.* & 0x7F == 0x7F) {
                        byte.* ^= 1;
                    };
                    const exponents = try gpa.alloc(u8, n * k / 32);
                    defer gpa.free(exponents);
                    for (exponents) |*byte| byte.* = 0x70 + random.uintLessThan(u8, 0x0C);
                    one.* = try .init(gpa, codes, exponents, n, k);
                } else {
                    const codes = try gpa.alloc(u8, n * k / 2);
                    defer gpa.free(codes);
                    random.bytes(codes);
                    const block_scales = try gpa.alloc(u8, n * k / 16);
                    defer gpa.free(block_scales);
                    for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
                    one.* = try .init(gpa, codes, block_scales, n, k, 0.02);
                }
                made += 1;
            }
            return self;
        }

        fn deinit(self: *@This()) void {
            for (&self.copies) |*one| one.deinit();
        }
    };
}

fn runLane(gpa: Allocator, random: std.Random, comptime T: type, label: []const u8) !void {
    const Lane = LaneOf(T);
    var lane: Lane = try .init(gpa, random);
    defer lane.deinit();
    const saved = nvfp4.lane_fused_tiles;
    defer nvfp4.lane_fused_tiles = saved;
    for (lane_rows) |rows| {
        var x = try randomRows(gpa, random, rows, lane_shape.k);
        defer x.deinit();
        var outs: [2]cuda.Buffer = undefined;
        for (&outs) |*out| out.* = try .init(.bf16, &.{ rows, lane_shape.n });
        defer for (&outs) |*out| out.deinit();
        var medians: [2]f64 = undefined;
        for (0..2) |form| {
            nvfp4.lane_fused_tiles = if (form == 0) 0 else 40;
            var times: [rounds]u64 = undefined;
            for (&times) |*time| {
                try cuda.synchronize();
                const t0 = nowNs();
                for (0..launches) |i| try lane.copies[i % lane_copies].forward(x.view, outs[form].view);
                try cuda.synchronize();
                time.* = nowNs() - t0;
            }
            std.mem.sort(u64, &times, {}, std.sort.asc(u64));
            medians[form] = @as(f64, @floatFromInt(times[rounds / 2])) / 1e3 / launches;
        }
        // both forms' last launch was on the same copy
        const a = try gpa.alloc(u16, rows * lane_shape.n);
        defer gpa.free(a);
        const b = try gpa.alloc(u16, rows * lane_shape.n);
        defer gpa.free(b);
        try outs[0].download(std.mem.sliceAsBytes(a));
        try outs[1].download(std.mem.sliceAsBytes(b));
        try std.testing.expectEqualSlices(u16, a, b);
        std.debug.print("PROFILE lane:{s}:{d}: {d}x{d}, cluster {d:.1} us, a block a tile {d:.1} us a launch by the loop (median of {d}), the same bits\n", .{ label, rows, lane_shape.n, lane_shape.k, medians[0], medians[1], rounds });
    }
}

/// FP4-Local's MXFP8 linears at a prompt's windows (spec 105 WS-03): the DeltaNet and attention input projections,
/// their output ones, the shared expert's gate and up and its down. The 64-row tile against the wide one, compared
/// bit for bit.
const wide_shapes = [_]struct { n: usize, k: usize }{ .{ .n = 16512, .k = 2560 }, .{ .n = 13952, .k = 2560 }, .{ .n = 2560, .k = 6144 }, .{ .n = 1280, .k = 2560 }, .{ .n = 2560, .k = 640 } };
const wide_rows = [_]usize{ 256, 512, 1024, 2048 };

fn runLaneWide(gpa: Allocator, random: std.Random) !void {
    const saved = nvfp4.lane_wide_rows;
    defer nvfp4.lane_wide_rows = saved;
    for (wide_shapes) |shape| {
        const codes = try gpa.alloc(u8, shape.n * shape.k);
        defer gpa.free(codes);
        random.bytes(codes);
        for (codes) |*byte| if (byte.* & 0x7F == 0x7F) {
            byte.* ^= 1;
        };
        const exponents = try gpa.alloc(u8, shape.n * shape.k / 32);
        defer gpa.free(exponents);
        for (exponents) |*byte| byte.* = 0x70 + random.uintLessThan(u8, 0x0C);
        var linear: nvfp4.Mx8Linear = try .init(gpa, codes, exponents, shape.n, shape.k);
        defer linear.deinit();
        for (wide_rows) |rows| {
            var x = try randomRows(gpa, random, rows, shape.k);
            defer x.deinit();
            var outs: [2]cuda.Buffer = undefined;
            for (&outs) |*out| out.* = try .init(.bf16, &.{ rows, shape.n });
            defer for (&outs) |*out| out.deinit();
            var medians: [2]f64 = undefined;
            for (0..2) |form| {
                nvfp4.lane_wide_rows = if (form == 0) 0 else 1;
                var times: [rounds]u64 = undefined;
                for (&times) |*time| {
                    try cuda.synchronize();
                    const t0 = nowNs();
                    for (0..launches) |_| try linear.forward(x.view, outs[form].view);
                    try cuda.synchronize();
                    time.* = nowNs() - t0;
                }
                std.mem.sort(u64, &times, {}, std.sort.asc(u64));
                medians[form] = @as(f64, @floatFromInt(times[rounds / 2])) / 1e3 / launches;
            }
            const a = try gpa.alloc(u16, rows * shape.n);
            defer gpa.free(a);
            const b = try gpa.alloc(u16, rows * shape.n);
            defer gpa.free(b);
            try outs[0].download(std.mem.sliceAsBytes(a));
            try outs[1].download(std.mem.sliceAsBytes(b));
            try std.testing.expectEqualSlices(u16, a, b);
            const flops = 2.0 * @as(f64, @floatFromInt(rows * shape.n * shape.k));
            std.debug.print("PROFILE lanewide:{d}x{d}:{d}: 64-row tile {d:.1} us ({d:.1} TFLOPS), wide {d:.1} us ({d:.1}), the same bits\n", .{ shape.n, shape.k, rows, medians[0], flops / medians[0] / 1e6, medians[1], flops / medians[1] / 1e6 });
        }
    }
}

/// GLM FP4-Local's lane-matmul linears on a rank at a prompt's window (spec 107 WS-03, `glmlinears`): MXFP8 KDA
/// q, k and v stacked, its o_proj and the dense MLP's down; block FP8 the shared expert's gate and up and down and
/// MLA's q_b and o_proj. Each prints its time, its TFLOPS and a hash of its output, to hold a form's bits against
/// another build's.
/// A linear's outputs and inputs.
const LinearShape = struct { n: usize, k: usize };
const glm_mx8 = [_]LinearShape{ .{ .n = 12288, .k = 4096 }, .{ .n = 4096, .k = 4096 }, .{ .n = 4096, .k = 6144 } };
/// Flash Next FP4-Local's MXFP8 linears (spec 143 WS-06): DeltaNet's qkv, z and out, attention's q (gated), o, the
/// shared expert's gate and up, and down (`ZSPARK_NVFP4_PROFILE=flasha8`).
const flash_mx8 = [_]LinearShape{ .{ .n = 10240, .k = 2560 }, .{ .n = 6144, .k = 2560 }, .{ .n = 2560, .k = 6144 }, .{ .n = 12288, .k = 2560 }, .{ .n = 1280, .k = 2560 }, .{ .n = 2560, .k = 640 } };
const glm_fp8 = [_]LinearShape{ .{ .n = 2048, .k = 4096 }, .{ .n = 4096, .k = 1024 }, .{ .n = 8192, .k = 1536 }, .{ .n = 4096, .k = 8192 } };

fn timeLinear(gpa: Allocator, linear: anytype, label: []const u8, n: usize, k: usize, x: cuda.Buffer, rows: usize) !void {
    var out: cuda.Buffer = try .init(.bf16, &.{ rows, n });
    defer out.deinit();
    var times: [rounds]u64 = undefined;
    for (&times) |*time| {
        try cuda.synchronize();
        const t0 = nowNs();
        for (0..launches) |_| try linear.forward(x.view, out.view);
        try cuda.synchronize();
        time.* = nowNs() - t0;
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    const us = @as(f64, @floatFromInt(times[rounds / 2])) / 1e3 / launches;
    const host = try gpa.alloc(u16, rows * n);
    defer gpa.free(host);
    try out.download(std.mem.sliceAsBytes(host));
    var hash: u64 = 0xcbf29ce484222325;
    for (host) |bits| hash = (hash ^ bits) *% 0x100000001b3;
    const flops = 2.0 * @as(f64, @floatFromInt(rows * n * k));
    std.debug.print("PROFILE glmlinears:{s}:{d}x{d}:{d}: {d:.1} us ({d:.1} TFLOPS), output {x:0>16}\n", .{ label, n, k, rows, us, flops / us / 1e6, hash });
}

fn runGlmLinears(gpa: Allocator, random: std.Random) !void {
    const rows = 2048;
    for (glm_mx8) |shape| {
        const codes = try gpa.alloc(u8, shape.n * shape.k);
        defer gpa.free(codes);
        random.bytes(codes);
        for (codes) |*byte| if (byte.* & 0x7F == 0x7F) {
            byte.* ^= 1;
        };
        const exponents = try gpa.alloc(u8, shape.n * shape.k / 32);
        defer gpa.free(exponents);
        for (exponents) |*byte| byte.* = 0x70 + random.uintLessThan(u8, 0x0C);
        var linear: nvfp4.Mx8Linear = try .init(gpa, codes, exponents, shape.n, shape.k);
        defer linear.deinit();
        var x = try randomRows(gpa, random, rows, shape.k);
        defer x.deinit();
        try timeLinear(gpa, linear, "mxfp8", shape.n, shape.k, x, rows);
    }
    for (glm_fp8) |shape| {
        const codes = try gpa.alloc(u8, shape.n * shape.k);
        defer gpa.free(codes);
        random.bytes(codes);
        for (codes) |*byte| if (byte.* & 0x7F == 0x7F) {
            byte.* ^= 1;
        };
        const scales = try gpa.alloc(f32, (shape.n + 127) / 128 * (shape.k / 128));
        defer gpa.free(scales);
        for (scales) |*scale| scale.* = 1e-3 * (1.0 + random.float(f32));
        var linear: nvfp4.Fp8BlockLinear = try .init(gpa, codes, scales, shape.n, shape.k, .of(shape.k));
        defer linear.deinit();
        var x = try randomRows(gpa, random, rows, shape.k);
        defer x.deinit();
        try timeLinear(gpa, linear, "fp8", shape.n, shape.k, x, rows);
    }
}

/// Block FP8's decode windows at GLM FP4-Local's shapes on a rank (spec 114 WS-02, `glmdecode`): `qmmf_tile.cu`
/// at 1, 2 and 6 rows in each form of K slices (a cluster, or summed in the block), a copy of the weight a launch, against the bytes' time at
/// 220 GB/s. A form's output hash holds its bits against the others' of the same slices.
fn runGlmDecode(gpa: Allocator, random: std.Random) !void {
    for (glm_fp8) |shape| {
        const codes = try gpa.alloc(u8, shape.n * shape.k);
        defer gpa.free(codes);
        random.bytes(codes);
        for (codes) |*byte| if (byte.* & 0x7F == 0x7F) {
            byte.* ^= 1;
        };
        const scales = try gpa.alloc(f32, (shape.n + 127) / 128 * (shape.k / 128));
        defer gpa.free(scales);
        for (scales) |*scale| scale.* = 1e-3 * (1.0 + random.float(f32));
        // a copy a launch, past L2's 24 MB, as a round reads each layer's weight once
        var copies: [launches]nvfp4.Fp8BlockLinear = undefined;
        for (&copies, 0..) |*copy, i| {
            errdefer for (copies[0..i]) |*made| made.deinit();
            copy.* = try .init(gpa, codes, scales, shape.n, shape.k, .of(shape.k));
        }
        defer for (&copies) |*copy| copy.deinit();
        const bus = @as(f64, @floatFromInt(shape.n * shape.k)) / 220e3;
        for ([_]usize{ 1, 2, 6 }) |rows| {
            var x = try randomRows(gpa, random, rows, shape.k);
            defer x.deinit();
            var out: cuda.Buffer = try .init(.bf16, &.{ rows, shape.n });
            defer out.deinit();
            const groups = shape.k / 64;
            for ([_]usize{ 1, 2, 4, 8 }) |sk| for ([_]bool{ false, true }) |fuse| {
                if (groups % sk != 0 or (fuse and sk == 1)) continue;
                var times: [rounds]u64 = undefined;
                for (&times) |*time| {
                    try cuda.synchronize();
                    const t0 = nowNs();
                    for (&copies) |*linear| try cuda.launched(cuda.kernels.zs_nvfp4_qmmf_tile(&x.view, &linear.w8.view, &linear.tile_scales.?.view, &out.view, @intCast(shape.n), @intCast(sk), @intCast(linear.npad), @intFromBool(fuse), 0));
                    try cuda.synchronize();
                    time.* = nowNs() - t0;
                }
                std.mem.sort(u64, &times, {}, std.sort.asc(u64));
                const us = @as(f64, @floatFromInt(times[rounds / 2])) / 1e3 / launches;
                const host = try gpa.alloc(u16, rows * shape.n);
                defer gpa.free(host);
                try out.download(std.mem.sliceAsBytes(host));
                var hash: u64 = 0xcbf29ce484222325;
                for (host) |bits| hash = (hash ^ bits) *% 0x100000001b3;
                const now = nvfp4.splitK(shape.n, shape.k) == sk and (sk > 1 and (shape.n + 63) / 64 >= nvfp4.fused_tiles) == fuse;
                std.debug.print("PROFILE glmdecode:{d}x{d}:{d}: {d} slices {s}: {d:.1} us (bus {d:.1}){s}, output {x:0>16}\n", .{ shape.n, shape.k, rows, sk, if (fuse) "in the block" else "by a cluster", us, bus, if (now) " (now)" else "", hash });
            };
        }
    }
}

/// GLM FP4-Local's quantized linears with their rows quantized as vLLM's (spec 114 WS-04, `glma8`): `qmm_a8.cu`
/// against the bf16 rows' lane matmul at the shapes of `glmlinears`, a copy of the weight a launch (past L2) at 1,
/// 2 and 6 rows and at 512, 2,048 and 4,096. Each prints us, TFLOPS and the weight's GB/s.
fn runGlmA8(gpa: Allocator, random: std.Random, flash: bool) !void {
    var scratch: cuda.Scratch = .init(gpa);
    defer scratch.deinit();
    inline for (.{ "mxfp8", "fp8" }) |label| {
        const mx = comptime std.mem.eql(u8, label, "mxfp8");
        const Linear = if (mx) nvfp4.Mx8Linear else nvfp4.Fp8BlockLinear;
        const shapes: []const LinearShape = if (flash) (if (mx) &flash_mx8 else &.{}) else if (mx) &glm_mx8 else &glm_fp8;
        for (shapes) |shape| {
            const codes = try gpa.alloc(u8, shape.n * shape.k);
            defer gpa.free(codes);
            random.bytes(codes);
            for (codes) |*byte| if (byte.* & 0x7F == 0x7F) {
                byte.* ^= 1;
            };
            const exponents = try gpa.alloc(u8, shape.n * shape.k / 32);
            defer gpa.free(exponents);
            for (exponents) |*byte| byte.* = 0x70 + random.uintLessThan(u8, 0x0C);
            const scales = try gpa.alloc(f32, (shape.n + 127) / 128 * (shape.k / 128));
            defer gpa.free(scales);
            for (scales) |*scale| scale.* = 1e-3 * (1.0 + random.float(f32));
            var copies: [launches]Linear = undefined;
            for (&copies, 0..) |*copy, i| {
                errdefer for (copies[0..i]) |*made| made.deinit();
                copy.* = if (mx) try .init(gpa, codes, exponents, shape.n, shape.k) else try .init(gpa, codes, scales, shape.n, shape.k, .of(shape.k));
                _ = try copy.quantizeActivations(scratch.rows(4096));
            }
            defer for (&copies) |*copy| copy.deinit();
            const a8 = copies[0].a8.?;
            for (if (flash) &[_]usize{ 1, 49, 512, 2048, 4096 } else &[_]usize{ 1, 2, 6, 512, 2048, 4096 }) |rows| {
                var x = try randomRows(gpa, random, rows, shape.k);
                defer x.deinit();
                var out: cuda.Buffer = try .init(.bf16, &.{ rows, shape.n });
                defer out.deinit();
                var medians: [2]f64 = .{ 0, 0 };
                for (&medians, 0..) |*median, form| {
                    for (&copies) |*copy| copy.a8 = if (form == 0) null else a8;
                    var times: [rounds]u64 = undefined;
                    for (&times) |*time| {
                        try cuda.synchronize();
                        const t0 = nowNs();
                        for (&copies) |*linear| try linear.forward(x.view, out.view);
                        try cuda.synchronize();
                        time.* = nowNs() - t0;
                    }
                    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
                    median.* = @as(f64, @floatFromInt(times[rounds / 2])) / 1e3 / launches;
                }
                for (&copies) |*copy| copy.a8 = a8;
                const flops = 2.0 * @as(f64, @floatFromInt(rows * shape.n * shape.k));
                const bytes = @as(f64, @floatFromInt(shape.n * shape.k));
                std.debug.print("PROFILE glma8:{s}:{d}x{d}:{d}: bf16 rows {d:.1} us ({d:.1} TFLOPS, {d:.0} GB/s), fp8 rows {d:.1} us ({d:.1} TFLOPS, {d:.0} GB/s)\n", .{ label, shape.n, shape.k, rows, medians[0], flops / medians[0] / 1e6, bytes / medians[0] / 1e3, medians[1], flops / medians[1] / 1e6, bytes / medians[1] / 1e3 });
            }
        }
    }
}

/// Whether `asked` takes the experts case: its name with or without the form, its set, or `all`.
fn takes(asked: []const u8, set: []const u8, case: Routed, form: Form) bool {
    var buffer: [64]u8 = undefined;
    const full_name = std.fmt.bufPrint(&buffer, "experts:{d}:{d}:{t}", .{ case.rows, case.distinct, form }) catch unreachable;
    if (std.mem.eql(u8, asked, full_name)) return true;
    if (form == .base and std.mem.eql(u8, asked, full_name[0 .. full_name.len - ":base".len])) return true;
    // a set's `even` and `hot` forms by the set's name and the form's: `rounds:even`
    var set_buffer: [64]u8 = undefined;
    const set_name = std.fmt.bufPrint(&set_buffer, "{s}:{t}", .{ set, form }) catch unreachable;
    if (std.mem.eql(u8, asked, set_name)) return true;
    return form == .base and (std.mem.eql(u8, asked, set) or std.mem.eql(u8, asked, "all"));
}

test "PROFILE the NVFP4 experts and the draft head launched alone at Flash Next's shapes" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(73);
    const random = prng.random();
    const asked: ?[]const u8 = if (std.c.getenv("ZSPARK_NVFP4_PROFILE")) |text| std.mem.span(text) else null;
    var ran: usize = 0;
    if (std.c.getenv("ZSPARK_NVFP4_SHAPE")) |text| if (std.mem.eql(u8, std.mem.span(text), "glm")) {
        width = 1024;
        dims = 4096;
        slots = 8;
        model_experts = 288;
    };
    defer {
        width = 640;
        dims = 2560;
        slots = 10;
        model_experts = 512;
    }

    const name = asked orelse {
        var layer = try makeLayer(gpa, random, 16);
        defer layer.deinit();
        for ([_]Form{ .base, .even }) |form| try runExperts(gpa, random, layer, .{ .rows = 2, .distinct = 12 }, form, false);
        return runHead(gpa, random, 12800, 2, false);
    };
    if (std.mem.eql(u8, name, "lane") or std.mem.eql(u8, name, "all")) {
        ran += 1;
        try runLane(gpa, random, nvfp4.Mx8Linear, "mxfp8");
        try runLane(gpa, random, nvfp4.Fp4Linear, "nvfp4");
    }
    if (std.mem.eql(u8, name, "lanewide")) {
        ran += 1;
        try runLaneWide(gpa, random);
    }
    if (std.mem.eql(u8, name, "glmlinears")) {
        ran += 1;
        try runGlmLinears(gpa, random);
    }
    if (std.mem.eql(u8, name, "glmdecode")) {
        ran += 1;
        try runGlmDecode(gpa, random);
    }
    if (std.mem.eql(u8, name, "glma8") or std.mem.eql(u8, name, "flasha8")) {
        ran += 1;
        try runGlmA8(gpa, random, std.mem.eql(u8, name, "flasha8"));
    }
    if (!std.mem.startsWith(u8, name, "head") and !std.mem.eql(u8, name, "lane") and !std.mem.eql(u8, name, "lanewide") and !std.mem.eql(u8, name, "glmlinears") and !std.mem.eql(u8, name, "glmdecode") and !std.mem.eql(u8, name, "glma8") and !std.mem.eql(u8, name, "flasha8")) {
        var layer: ?experts_file.Experts = null;
        defer if (layer) |*made| made.deinit();
        var layer_a4: ?experts_file.Experts = null;
        defer if (layer_a4) |*made| made.deinit();
        inline for (.{ .{ "rounds", &round_cases }, .{ "sweep", &sweep_cases }, .{ "chain", &chain_cases }, .{ "window", &window_cases }, .{ "skew", &skew_cases }, .{ "glm", &glm_cases } }) |entry| {
            for (entry[1]) |case| for (std.enums.values(Form)) |form| {
                if (!takes(name, entry[0], case, form)) continue;
                ran += 1;
                if (form == .a4 or form == .a4narrow or form == .a4tile or form == .a4tile8 or form == .a16) {
                    if (layer_a4 == null) layer_a4 = try makeLayerAs(gpa, random, model_experts, .nvfp4);
                    try runExperts(gpa, random, layer_a4.?, case, form, true);
                    continue;
                }
                if (layer == null) layer = try makeLayer(gpa, random, model_experts);
                try runExperts(gpa, random, layer.?, case, form, true);
            };
        }
    }
    for (head_cases) |rows| {
        var buffer: [32]u8 = undefined;
        const case_name = std.fmt.bufPrint(&buffer, "head:{d}", .{rows}) catch unreachable;
        if (!std.mem.eql(u8, name, case_name) and !std.mem.eql(u8, name, "head") and !std.mem.eql(u8, name, "all")) continue;
        ran += 1;
        try runHead(gpa, random, head_rows, rows, true);
    }
    // a name that is no case's is a mistake, not a pass
    try std.testing.expect(ran > 0);
}
