//! The bf16 GEMM (`bf16/gemm.cu`: `zs_b16_gemm`) alone at the shapes Flash Next and GLM launch it at, on random
//! weights: what a profiler is given (spec 073 WS-04). No checkpoint and no model. The retired EXL3 packs' fp16
//! GEMM and its set are in tag retired-checkpoints-2026-10-21.
//!
//! `ZSPARK_BF16_PROFILE` names a case (`down:2`, `up:2:hot`, `stack:4:tiles`, ...), a set (`nvfp4`: the retired NVFP4
//! checkpoint's bf16 side layers, head and read-outs; `glm`: the retired GLM EXL3 pack's, a rank's; `fp4local`: GLM FP4-Local's; `prompt`: the windows of 24 to 2048 rows, `prompt:tiles` their other form) or `all`. Unset, the test is a smoke of the small
//! cases, one copy and one pass, for a tier. A named case's weights are `copies` buffers taken in turn, so that
//! none is in L2 (24 MB) when it is read, with `Tail` allocated after them; a `hot` case launches on one copy,
//! which L2 then holds if it fits: what the kernel takes without DRAM. The launches are `rounds` passes; a
//! trace takes the medians of a kernel's own duration, a profiler skips the first pass.
//!
//!   ZSPARK_BF16_PROFILE=up:2 ncu --kernel-name regex:Kernel2 --launch-skip 40 --launch-count 1 TEST
//!   ZSPARK_BF16_PROFILE=fp4local nsys profile --trace=cuda --sample=none --cpuctxsw=none TEST
//!
//! Each case prints its launches a pass and the weight bytes one launch reads.

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("../../src/weights/bf16.zig");
const cuda = @import("../../src/cuda/cuda.zig");
const numeric = @import("../../src/weights/numeric.zig");
const profile_fixture = @import("../weights/profile_fixture.zig");
const nowNs = profile_fixture.nowNs;
const expectRan = profile_fixture.expectRan;

const Element = enum { b16 };

/// How a launch is asked. `sum`: `forward` into bf16, the slices summed by `b16_reduce` where the weight splits
/// K. `slices`: the K slices left in fp32 for the kernel that reads them next (the read-out's down projection,
/// the router's gate).
const Ask = enum { sum, slices };

/// `base`: the GEMM on the narrow tile, or the wide one for a window. `hot`: the same on one copy of the weight.
/// `tiles`: on the tile `bf16.tiles_by_shape` picks.
const Form = enum { base, hot, tiles };

const Shape = struct {
    name: []const u8,
    element: Element,
    n: usize,
    k: usize,
    ask: Ask = .sum,
    /// Left out of the smoke and of a set unless named: the head is 1.27 GB.
    heavy: bool = false,
};

/// The retired NVFP4 checkpoint's bf16 linears (its recipe in tag retired-checkpoints-2026-10-21; the grids of
/// spec 070's trace): DeltaNet's stacked `in_proj` (129 column tiles), attention's stacked q, k and v (109), `out_proj`
/// and `o_proj` (20 tiles by 4 slices), the shared expert's gate and up (10 by 8) and down (20 by 2), the
/// read-outs, the router's gate and the head.
const nvfp4 = [_]Shape{
    .{ .name = "stack", .element = .b16, .n = 16512, .k = 2560 },
    .{ .name = "qkv", .element = .b16, .n = 13952, .k = 2560 },
    .{ .name = "out", .element = .b16, .n = 2560, .k = 6144 },
    .{ .name = "wide", .element = .b16, .n = 1280, .k = 2560, .ask = .slices },
    .{ .name = "back", .element = .b16, .n = 2560, .k = 1280 },
    .{ .name = "bdown", .element = .b16, .n = 320, .k = 10240, .ask = .slices },
    .{ .name = "bup", .element = .b16, .n = 10240, .k = 320 },
    .{ .name = "head", .element = .b16, .n = 248320, .k = 2560, .heavy = true },
};

/// The retired GLM EXL3 pack's bf16 linears as a rank holds them (its recipe in tag retired-checkpoints-2026-10-21;
/// the grids of spec 073 WS-05's trace): KDA's stacked projection (1,572 column tiles of 8, 103 MB) and its `o` (128 of 32), the dense
/// MLP's gate and up (1,536 of 8) and down, DSA's `q_b`. None fits L2, so none has a `hot` case.
const glm = [_]Shape{
    .{ .name = "kproj", .element = .b16, .n = 12576, .k = 4096 },
    .{ .name = "ko", .element = .b16, .n = 4096, .k = 4096 },
    .{ .name = "dgu", .element = .b16, .n = 12288, .k = 4096 },
    .{ .name = "ddown", .element = .b16, .n = 4096, .k = 6144 },
    .{ .name = "qb", .element = .b16, .n = 8192, .k = 1536 },
};

/// FP4-Local's bf16 linears as a rank holds them (spec 114 WS-02; the grids of WS-02's decode trace): the
/// router's gate and KDA's stacked f_a, g_a and b (both 288 x 4096 in 32 slices), KDA's f_b and g_b (4096 x 128
/// in 2), the indexer's wk and gate (128 x 4096), its weights_proj (32 x 4096) and its wq_b (4096 x 1536 in 4).
const fp4_local = [_]Shape{
    .{ .name = "router", .element = .b16, .n = 288, .k = 4096, .ask = .slices },
    .{ .name = "kab", .element = .b16, .n = 288, .k = 4096 },
    .{ .name = "fb", .element = .b16, .n = 4096, .k = 128 },
    .{ .name = "iwk", .element = .b16, .n = 128, .k = 4096 },
    .{ .name = "iwp", .element = .b16, .n = 32, .k = 4096 },
    .{ .name = "iwq", .element = .b16, .n = 4096, .k = 1536 },
};

const decode_rows = [_]usize{ 1, 2, 3, 4, 6, 8, 16 };
/// 24 and 32: a pooled round's or a verify window's rows above the 16-row tiles; from 1,024 a window's GEMM
/// sums a split inside itself (`bf16.sums_from`).
const prompt_rows = [_]usize{ 24, 32, 64, 512, 1024, 2048 };
const rounds = 9;
const l2 = 24 << 20;

const Case = struct { shape: Shape, rows: usize, form: Form };

/// Finite values near 2^-7 of either sign: random mantissas under a fixed exponent.
fn randomWeight(random: std.Random, element: Element, bits: []u16) void {
    random.bytes(std.mem.sliceAsBytes(bits));
    const keep: u16, const exponent: u16 = switch (element) {
        .b16 => .{ 0x807f, 0x78 << 7 },
    };
    for (bits) |*value| value.* = (value.* & keep) | exponent;
}

fn randomRows(gpa: Allocator, random: std.Random, element: Element, rows: usize, k: usize) !cuda.Buffer {
    const host = try gpa.alloc(u16, rows * k);
    defer gpa.free(host);
    for (host) |*bits| {
        const value = random.floatNorm(f32);
        bits.* = switch (element) {
            .b16 => numeric.bf16FromF32(value),
        };
    }
    var x: cuda.Buffer = try .init(.bf16, &.{ rows, k });
    errdefer x.deinit();
    try x.upload(std.mem.sliceAsBytes(host));
    return x;
}

const Linear = union(Element) {
    b16: bf16.Linear,

    fn init(element: Element, weight: []const u8, n: usize, k: usize, rows: cuda.Scratch.Rows) !Linear {
        return switch (element) {
            .b16 => .{ .b16 = try .init(weight, n, k, rows) },
        };
    }

    fn deinit(self: *Linear) void {
        switch (self.*) {
            inline else => |*linear| linear.deinit(),
        }
    }

    /// What the launch left for a reader: `out`, or the slices `slices` gave.
    fn launch(self: Linear, ask: Ask, x: cuda.View, out: cuda.View) !cuda.View {
        switch (self) {
            inline else => |linear| switch (ask) {
                .sum => {
                    try linear.forward(x, out);
                    return out;
                },
                .slices => return (try linear.slices(x, out)).view,
            },
        }
    }
};

fn run(gpa: Allocator, random: std.Random, case: Case, full: bool) !void {
    const shape = case.shape;
    const bytes = shape.n * shape.k * 2;
    const host = try gpa.alloc(u16, shape.n * shape.k);
    defer gpa.free(host);
    randomWeight(random, shape.element, host);
    // copies past L2 several times over; a weight larger than L2 is past it in two
    const copies: usize = if (!full) 1 else if (bytes > 4 * l2) 1 else if (bytes > l2) 2 else (128 << 20) / bytes;
    var scratch: cuda.Scratch = .init(gpa);
    defer scratch.deinit();
    const linears = try gpa.alloc(Linear, copies);
    defer gpa.free(linears);
    var made: usize = 0;
    defer for (linears[0..made]) |*linear| linear.deinit();
    for (linears) |*linear| {
        linear.* = try .init(shape.element, std.mem.sliceAsBytes(host), shape.n, shape.k, scratch.rows(case.rows));
        made += 1;
    }
    var x = try randomRows(gpa, random, shape.element, case.rows, shape.k);
    defer x.deinit();
    var out: cuda.Buffer = try .init(if (shape.ask == .slices) .f32 else .bf16, &.{ case.rows, shape.n });
    defer out.deinit();
    // 64 MB that nothing reads, after the weights: the pages the kernel zeroed last are then these (spec 071 WS-10)
    var tail: ?cuda.Buffer = if (full) try .init(.u8, &.{64 << 20}) else null;
    defer if (tail) |*buffer| buffer.deinit();

    const before = bf16.tiles_by_shape;
    defer bf16.tiles_by_shape = before;
    bf16.tiles_by_shape = case.form == .tiles;
    // a pass is at least 16 launches, so that a small weight's medians are of as many as a large one's
    const launches = if (full) @max(copies, 16) else 1;
    var times: [rounds]u64 = undefined;
    var left = out.view;
    for (if (full) times[0..] else times[0..1]) |*time| {
        try cuda.synchronize();
        const t0 = nowNs();
        for (0..launches) |at| left = try linears[if (case.form == .hot) 0 else at % copies].launch(shape.ask, x.view, out.view);
        try cuda.synchronize();
        time.* = nowNs() - t0;
    }
    switch (shape.ask) {
        .sum => try expectRan(gpa, u16, out.view),
        // the slices themselves where a round's weight splits K, else `out`
        .slices => try expectRan(gpa, f32, left),
    }
    if (!full) return;
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    std.debug.print("PROFILE {s}:{d}:{t}: {d} launches a pass, {d} weight bytes a launch; a launch by the loop {d:.1} us median\n", .{
        shape.name, case.rows, case.form, launches, bytes, @as(f64, @floatFromInt(times[rounds / 2])) / 1e3 / @as(f64, @floatFromInt(launches)),
    });
}

/// Whether `asked` takes this case: its name with or without the form (`up:2` is `up:2:base`), a set, or `all`.
fn takes(asked: []const u8, set: []const u8, case: Case, prompt: bool) bool {
    var buffer: [64]u8 = undefined;
    const full_name = std.fmt.bufPrint(&buffer, "{s}:{d}:{t}", .{ case.shape.name, case.rows, case.form }) catch unreachable;
    if (std.mem.eql(u8, asked, full_name)) return true;
    if (case.form == .base and std.mem.eql(u8, asked, full_name[0 .. full_name.len - ":base".len])) return true;
    if (case.shape.heavy) return false;
    if (std.mem.eql(u8, asked, "prompt")) return prompt and case.form == .base;
    if (std.mem.eql(u8, asked, "prompt:tiles")) return prompt and case.form == .tiles;
    if (prompt) return false;
    return std.mem.eql(u8, asked, set) or std.mem.eql(u8, asked, "all");
}

test "PROFILE the half-precision GEMMs launched alone at Flash Next's shapes, a few rows and a prompt's window" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(73);
    const random = prng.random();
    const asked: ?[]const u8 = if (std.c.getenv("ZSPARK_BF16_PROFILE")) |text| std.mem.span(text) else null;
    var ran: usize = 0;
    inline for (.{ .{ "nvfp4", &nvfp4 }, .{ "glm", &glm }, .{ "fp4local", &fp4_local } }) |entry| {
        for (entry[1]) |shape| {
            for (decode_rows ++ prompt_rows) |rows| {
                const prompt = rows > 16;
                for (std.enums.values(Form)) |form| {
                    // the tiles are a decoding launch's: 16 rows at most
                    // from L2 only what L2 holds
                    if (form == .hot and shape.n * shape.k * 2 > l2 * 3 / 4) continue;
                    const case: Case = .{ .shape = shape, .rows = rows, .form = form };
                    if (asked) |name| {
                        if (!takes(name, entry[0], case, prompt)) continue;
                    } else if (shape.n * shape.k > 4 << 20 or rows != 2 or form != .base) continue;
                    ran += 1;
                    try run(gpa, random, case, asked != null);
                }
            }
        }
    }
    // a name that is no case's is a mistake, not a pass
    try std.testing.expect(ran > 0);
}
