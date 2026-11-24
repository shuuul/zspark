//! Spec 158 WS-02: the kernel contract's registrations and planted controls.
//!
//! Registered: `zs_b16_gemm` (`bf16/gemm.cu`, CUTLASS) and `zs_rmsnorm` (`common/glue.cu`). The controls in
//! `zig/tests/kernels/contract/controls.cu` each break one gate on purpose; a control that passes its gate
//! fails this file.

const std = @import("std");
const contract = @import("contract.zig");
const cuda = @import("../../src/cuda/cuda.zig");
const numeric = @import("../../src/weights/numeric.zig");
const bf16 = @import("../../src/weights/bf16.zig");

/// Sentinel rows before and after the output: the kernel must not reach them.
const GUARD: usize = 2;
const SENTINEL_F32: u32 = 0x5a5a_5a5a;
const SENTINEL_BF16: u16 = 0x5a5a;

extern fn zs_rmsnorm(x: *const cuda.View, w: *const cuda.View, out: *const cuda.View, eps: f64) c_int;

extern fn zs_contract_max(x: *const cuda.View, out: *const cuda.View) c_int;
extern fn zs_contract_sum(x: *const cuda.View, out: *const cuda.View) c_int;
extern fn zs_contract_alias(x: *const cuda.View, out: *const cuda.View) c_int;
extern fn zs_contract_tail(x: *const cuda.View, out: *const cuda.View) c_int;
extern fn zs_contract_half(x: *const cuda.View, out: *const cuda.View) c_int;

fn cast(comptime T: type, c: contract.Case) *T {
    return @ptrCast(@alignCast(c.ctx));
}

fn sentinelF32(bits: u32) bool {
    return bits == SENTINEL_F32;
}

fn randomBf16(rng: std.Random, scale: f32) u16 {
    return numeric.bf16FromF32(rng.floatNorm(f32) * scale);
}

/// NaN, +Inf and -Inf at known places: a row each, and one row with both infinities.
fn placeNonFinite(bits: []u16, rows: usize, cols: usize) void {
    if (rows == 0 or cols == 0) return;
    bits[0] = 0x7fc0;
    if (rows > 1) bits[cols + cols / 2] = 0x7f80;
    if (rows > 2) bits[2 * cols + cols - 1] = 0xff80;
    if (rows > 3) {
        bits[3 * cols] = 0x7f80;
        bits[3 * cols + cols - 1] = 0xff80;
    }
}

/// Gate 4's integers: an odd permutation of 1..255 whose running sum f16 cannot hold (64 terms land 40 off,
/// 300 land 148 off) and fp32 holds exactly; a replay in bf16 misses it as well.
fn oddValue(at: usize) f32 {
    const index = (at * 41 + at * at * 3) % 128;
    return @floatFromInt(2 * index + 1);
}

/// Gate 4's integers for a row sum of squares: -8..8, whose squares f32 sums exactly up to the norm's 4,096
/// columns (2,62,144 < 2^24) and f16 does not.
fn integerBits(at: usize, span: i32) i32 {
    return @as(i32, @intCast(at % @as(usize, @intCast(2 * span + 1)))) - span;
}

// ------------------------------------------------------------------------------------------------
// zs_b16_gemm

const gemm_shapes = [_]contract.Shape{
    .{ .rows = 1, .n = 256, .k = 64 },
    .{ .rows = 17, .n = 256, .k = 64 },
    .{ .rows = 40, .n = 200, .k = 64 },
    .{ .rows = 40, .n = 512, .k = 128 },
};

const gemm_spec: contract.Spec = .{
    .name = "zs_b16_gemm",
    .shapes = &gemm_shapes,
    .kind = .f32,
    .depth = 2,
    // cuobjdump -res-usage of the linked test binary: the 16x128x64 tile 72 registers and 73,728 + 1,024 B, the
    // 128x128x64 one 234 and 98,304 + 1,024 B; these shapes take both, so the worst is declared (a shape that
    // takes the slim or the deep tile declares its own).
    .launch = .{ .symbol = "zs_b16_gemm", .threads = 128, .registers = 234, .bound = 128, .shared = 99_328 },
    .make = gemmMake,
};

const Gemm = struct {
    gpa: std.mem.Allocator,
    shape: contract.Shape,
    x: cuda.Buffer,
    w: cuda.Buffer,
    out: cuda.Buffer,
    x_bits: []u16,
    w_bits: []u16,
    out_bits: []u32,

    fn view(self: *Gemm) cuda.View {
        return self.out.span(GUARD / 2, self.shape.rows);
    }
};

fn gemmMake(gpa: std.mem.Allocator, shape: contract.Shape) !contract.Case {
    const self = try gpa.create(Gemm);
    errdefer gpa.destroy(self);
    var x: cuda.Buffer = try .init(.bf16, &.{ shape.rows, shape.k });
    errdefer x.deinit();
    var w: cuda.Buffer = try .init(.bf16, &.{ shape.n, shape.k });
    errdefer w.deinit();
    var out: cuda.Buffer = try .init(.f32, &.{ shape.rows + GUARD, shape.n });
    errdefer out.deinit();
    const x_bits = try gpa.alloc(u16, shape.rows * shape.k);
    errdefer gpa.free(x_bits);
    const w_bits = try gpa.alloc(u16, shape.n * shape.k);
    errdefer gpa.free(w_bits);
    const out_bits = try gpa.alloc(u32, (shape.rows + GUARD) * shape.n);
    errdefer gpa.free(out_bits);
    @memset(out_bits, SENTINEL_F32);
    try out.upload(std.mem.sliceAsBytes(out_bits));
    self.* = .{ .gpa = gpa, .shape = shape, .x = x, .w = w, .out = out, .x_bits = x_bits, .w_bits = w_bits, .out_bits = out_bits };
    return .{ .ctx = self, .v = &gemm_vtable };
}

fn gemmDeinit(c: contract.Case, gpa: std.mem.Allocator) void {
    const self = cast(Gemm, c);
    self.x.deinit();
    self.w.deinit();
    self.out.deinit();
    gpa.free(self.x_bits);
    gpa.free(self.w_bits);
    gpa.free(self.out_bits);
    gpa.destroy(self);
}

fn gemmFill(c: contract.Case, fill: contract.Fill, seed: u64) void {
    const self = cast(Gemm, c);
    var prng: std.Random.DefaultPrng = .init(seed);
    const rng = prng.random();
    for (self.x_bits) |*b| b.* = randomBf16(rng, 1.0);
    for (self.w_bits) |*b| b.* = randomBf16(rng, 0.05);
    switch (fill) {
        .random => {},
        .nonfinite => placeNonFinite(self.x_bits, self.shape.rows, self.shape.k),
        .integers => {
            for (self.x_bits, 0..) |*b, i| b.* = numeric.bf16FromF32(oddValue(i));
            for (self.w_bits) |*b| b.* = numeric.bf16FromF32(1.0);
        },
    }
    self.x.upload(std.mem.sliceAsBytes(self.x_bits)) catch unreachable;
    self.w.upload(std.mem.sliceAsBytes(self.w_bits)) catch unreachable;
}

fn gemmRun(c: contract.Case) !void {
    const self = cast(Gemm, c);
    var out = self.view();
    try cuda.launched(cuda.kernels.zs_b16_gemm(&self.x.view, &self.w.view, &out, null, 1, 0));
    try cuda.synchronize();
}

fn gemmOutLen(c: contract.Case) usize {
    const self = cast(Gemm, c);
    return self.shape.rows * self.shape.n;
}

fn gemmOutput(c: contract.Case, dest: []u8) !void {
    const self = cast(Gemm, c);
    try self.out.download(std.mem.sliceAsBytes(self.out_bits));
    const n = self.shape.n;
    for (0..self.shape.rows) |r| {
        @memcpy(dest[r * n * 4 ..][0 .. n * 4], std.mem.sliceAsBytes(self.out_bits[(r + 1) * n ..][0..n]));
    }
}

fn gemmDot(self: *Gemm, r: usize, col: usize, acc: contract.Acc) f64 {
    const k = self.shape.k;
    const x = self.x_bits[r * k ..][0..k];
    const w = self.w_bits[col * k ..][0..k];
    switch (acc) {
        .f64 => {
            var sum: f64 = 0;
            for (x, w) |a, b| sum += @as(f64, numeric.f32FromBf16(a)) * @as(f64, numeric.f32FromBf16(b));
            return sum;
        },
        .f16 => {
            var sum: f16 = 0;
            for (x, w) |a, b| {
                const term: f32 = numeric.f32FromBf16(a) * numeric.f32FromBf16(b);
                sum += @floatCast(term);
            }
            return @floatCast(sum);
        },
        .bf16 => {
            var sum: f32 = 0;
            for (x, w) |a, b| {
                const term: f32 = numeric.f32FromBf16(a) * numeric.f32FromBf16(b);
                sum = numeric.f32FromBf16(numeric.bf16FromF32(sum + term));
            }
            return sum;
        },
    }
}

fn gemmWant(c: contract.Case, gpa: std.mem.Allocator, out: []f64, acc: contract.Acc) !void {
    _ = gpa;
    const self = cast(Gemm, c);
    for (0..self.shape.rows) |r| for (0..self.shape.n) |col| {
        out[r * self.shape.n + col] = gemmDot(self, r, col, acc);
    };
}

fn gemmMagnitude(c: contract.Case, gpa: std.mem.Allocator, out: []f64) !void {
    _ = gpa;
    const self = cast(Gemm, c);
    const k = self.shape.k;
    for (0..self.shape.rows) |r| for (0..self.shape.n) |col| {
        var mag: f64 = 0;
        for (self.x_bits[r * k ..][0..k], self.w_bits[col * k ..][0..k]) |a, b| {
            mag += @abs(@as(f64, numeric.f32FromBf16(a)) * @as(f64, numeric.f32FromBf16(b)));
        }
        out[r * self.shape.n + col] = mag;
    };
}

fn gemmInputs(c: contract.Case, gpa: std.mem.Allocator, dest: []u8) !void {
    _ = gpa;
    const self = cast(Gemm, c);
    const x = std.mem.sliceAsBytes(self.x_bits);
    @memcpy(dest[0..x.len], x);
    @memcpy(dest[x.len..][0..std.mem.sliceAsBytes(self.w_bits).len], std.mem.sliceAsBytes(self.w_bits));
}

fn gemmInputLen(c: contract.Case) usize {
    const self = cast(Gemm, c);
    return (self.x_bits.len + self.w_bits.len) * 2;
}

fn gemmViews(c: contract.Case, out: []cuda.View) usize {
    const self = cast(Gemm, c);
    out[0] = self.x.view;
    out[1] = self.w.view;
    out[2] = self.view();
    return 3;
}

fn gemmGuards(c: contract.Case) !void {
    const self = cast(Gemm, c);
    try self.out.download(std.mem.sliceAsBytes(self.out_bits));
    const n = self.shape.n;
    for (self.out_bits[0..n]) |bits| if (!sentinelF32(bits)) return error.OutputOverran;
    for (self.out_bits[(self.shape.rows + 1) * n ..][0..n]) |bits| if (!sentinelF32(bits)) return error.OutputOverran;
}

const gemm_vtable: contract.Case.VTable = .{
    .deinit = gemmDeinit,
    .fill = gemmFill,
    .run = gemmRun,
    .outLen = gemmOutLen,
    .output = gemmOutput,
    .want = gemmWant,
    .magnitude = gemmMagnitude,
    .inputs = gemmInputs,
    .inputLen = gemmInputLen,
    .views = gemmViews,
    .checkGuards = gemmGuards,
};

// ------------------------------------------------------------------------------------------------
// zs_rmsnorm

const norm_shapes = [_]contract.Shape{
    .{ .rows = 1, .n = 128, .k = 0 },
    .{ .rows = 3, .n = 256, .k = 0 },
    .{ .rows = 17, .n = 300, .k = 0 },
    .{ .rows = 40, .n = 4096, .k = 0 },
};

const norm_eps: f64 = 1e-6;

const norm_spec: contract.Spec = .{
    .name = "zs_rmsnorm",
    .shapes = &norm_shapes,
    .kind = .bf16,
    .depth = 2,
    // cuobjdump -res-usage: 22 registers, 16,384 B of squares + 1,024 + 4 B.
    .launch = .{ .symbol = "zs_rmsnorm", .threads = 256, .registers = 22, .bound = 256, .shared = 17_412 },
    .make = normMake,
};

const Norm = struct {
    gpa: std.mem.Allocator,
    shape: contract.Shape,
    x: cuda.Buffer,
    w: cuda.Buffer,
    out: cuda.Buffer,
    x_bits: []u16,
    w_bits: []u16,
    out_bits: []u16,

    fn view(self: *Norm) cuda.View {
        return self.out.span(GUARD / 2, self.shape.rows);
    }
};

fn normMake(gpa: std.mem.Allocator, shape: contract.Shape) !contract.Case {
    const self = try gpa.create(Norm);
    errdefer gpa.destroy(self);
    var x: cuda.Buffer = try .init(.bf16, &.{ shape.rows, shape.n });
    errdefer x.deinit();
    var w: cuda.Buffer = try .init(.bf16, &.{shape.n});
    errdefer w.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ shape.rows + GUARD, shape.n });
    errdefer out.deinit();
    const x_bits = try gpa.alloc(u16, shape.rows * shape.n);
    errdefer gpa.free(x_bits);
    const w_bits = try gpa.alloc(u16, shape.n);
    errdefer gpa.free(w_bits);
    const out_bits = try gpa.alloc(u16, (shape.rows + GUARD) * shape.n);
    errdefer gpa.free(out_bits);
    @memset(out_bits, SENTINEL_BF16);
    try out.upload(std.mem.sliceAsBytes(out_bits));
    self.* = .{ .gpa = gpa, .shape = shape, .x = x, .w = w, .out = out, .x_bits = x_bits, .w_bits = w_bits, .out_bits = out_bits };
    return .{ .ctx = self, .v = &norm_vtable };
}

fn normDeinit(c: contract.Case, gpa: std.mem.Allocator) void {
    const self = cast(Norm, c);
    self.x.deinit();
    self.w.deinit();
    self.out.deinit();
    gpa.free(self.x_bits);
    gpa.free(self.w_bits);
    gpa.free(self.out_bits);
    gpa.destroy(self);
}

fn normFill(c: contract.Case, fill: contract.Fill, seed: u64) void {
    const self = cast(Norm, c);
    var prng: std.Random.DefaultPrng = .init(seed);
    const rng = prng.random();
    for (self.x_bits) |*b| b.* = randomBf16(rng, 1.0);
    for (self.w_bits) |*b| b.* = randomBf16(rng, 0.5);
    switch (fill) {
        .random => {},
        .nonfinite => placeNonFinite(self.x_bits, self.shape.rows, self.shape.n),
        .integers => {
            for (self.x_bits, 0..) |*b, i| b.* = numeric.bf16FromF32(@floatFromInt(integerBits(i, 8)));
            for (self.w_bits) |*b| b.* = numeric.bf16FromF32(1.0);
        },
    }
    self.x.upload(std.mem.sliceAsBytes(self.x_bits)) catch unreachable;
    self.w.upload(std.mem.sliceAsBytes(self.w_bits)) catch unreachable;
}

fn normRun(c: contract.Case) !void {
    const self = cast(Norm, c);
    var out = self.view();
    try cuda.launched(zs_rmsnorm(&self.x.view, &self.w.view, &out, norm_eps));
    try cuda.synchronize();
}

fn normOutLen(c: contract.Case) usize {
    const self = cast(Norm, c);
    return self.shape.rows * self.shape.n;
}

fn normOutput(c: contract.Case, dest: []u8) !void {
    const self = cast(Norm, c);
    try self.out.download(std.mem.sliceAsBytes(self.out_bits));
    const n = self.shape.n;
    for (0..self.shape.rows) |r| {
        @memcpy(dest[r * n * 2 ..][0 .. n * 2], std.mem.sliceAsBytes(self.out_bits[(r + 1) * n ..][0..n]));
    }
}

fn normWant(c: contract.Case, gpa: std.mem.Allocator, out: []f64, acc: contract.Acc) !void {
    _ = gpa;
    const self = cast(Norm, c);
    const n = self.shape.n;
    for (0..self.shape.rows) |r| {
        const x = self.x_bits[r * n ..][0..n];
        const sum = switch (acc) {
            .f64 => blk: {
                var s: f64 = 0;
                for (x) |b| {
                    const v = @as(f64, numeric.f32FromBf16(b));
                    s += v * v;
                }
                break :blk s;
            },
            .f16 => blk: {
                var s: f16 = 0;
                for (x) |b| {
                    const v: f16 = @floatCast(numeric.f32FromBf16(b));
                    s += v * v;
                }
                break :blk @as(f64, @floatCast(s));
            },
            .bf16 => blk: {
                var s: f32 = 0;
                for (x) |b| {
                    const v = numeric.f32FromBf16(b);
                    s = numeric.f32FromBf16(numeric.bf16FromF32(s + v * v));
                }
                break :blk @as(f64, s);
            },
        };
        const inverse = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(n)) + norm_eps);
        for (0..n) |d| {
            const xv = @as(f64, numeric.f32FromBf16(x[d]));
            const rounded = numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(xv * inverse)));
            out[r * n + d] = numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(rounded * @as(f64, numeric.f32FromBf16(self.w_bits[d])))));
        }
    }
}

fn normMagnitude(c: contract.Case, gpa: std.mem.Allocator, out: []f64) !void {
    _ = gpa;
    const self = cast(Norm, c);
    const n = self.shape.n;
    for (0..self.shape.rows) |r| {
        const x = self.x_bits[r * n ..][0..n];
        var sum: f64 = 0;
        for (x) |b| {
            const v = @as(f64, numeric.f32FromBf16(b));
            sum += v * v;
        }
        const inverse = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(n)) + norm_eps);
        // In the output's units: the sum's relative error reaches the output scaled by the output itself.
        for (0..n) |d| {
            const xv = @as(f64, numeric.f32FromBf16(x[d]));
            const rounded = numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(xv * inverse)));
            const value = numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(rounded * @as(f64, numeric.f32FromBf16(self.w_bits[d])))));
            out[r * n + d] = @abs(value);
        }
    }
}

fn normInputs(c: contract.Case, gpa: std.mem.Allocator, dest: []u8) !void {
    _ = gpa;
    const self = cast(Norm, c);
    const x = std.mem.sliceAsBytes(self.x_bits);
    @memcpy(dest[0..x.len], x);
    @memcpy(dest[x.len..][0..std.mem.sliceAsBytes(self.w_bits).len], std.mem.sliceAsBytes(self.w_bits));
}

fn normInputLen(c: contract.Case) usize {
    const self = cast(Norm, c);
    return (self.x_bits.len + self.w_bits.len) * 2;
}

fn normViews(c: contract.Case, out: []cuda.View) usize {
    const self = cast(Norm, c);
    out[0] = self.x.view;
    out[1] = self.w.view;
    out[2] = self.view();
    return 3;
}

fn normGuards(c: contract.Case) !void {
    const self = cast(Norm, c);
    try self.out.download(std.mem.sliceAsBytes(self.out_bits));
    const n = self.shape.n;
    for (self.out_bits[0..n]) |bits| if (bits != SENTINEL_BF16) return error.OutputOverran;
    for (self.out_bits[(self.shape.rows + 1) * n ..][0..n]) |bits| if (bits != SENTINEL_BF16) return error.OutputOverran;
}

const norm_vtable: contract.Case.VTable = .{
    .deinit = normDeinit,
    .fill = normFill,
    .run = normRun,
    .outLen = normOutLen,
    .output = normOutput,
    .want = normWant,
    .magnitude = normMagnitude,
    .inputs = normInputs,
    .inputLen = normInputLen,
    .views = normViews,
    .checkGuards = normGuards,
};

// ------------------------------------------------------------------------------------------------
// The controls: one planted break a gate, and only that gate's registration.

const control_shapes = [_]contract.Shape{
    .{ .rows = 4, .n = 256, .k = 0 },
    .{ .rows = 4, .n = 300, .k = 0 },
};

const Control = enum { max, sum, alias, tail, half };

const Rows = struct {
    gpa: std.mem.Allocator,
    shape: contract.Shape,
    which: Control,
    x: cuda.Buffer,
    out: cuda.Buffer,
    x_bits: []u16,
    out_bits: []u32,

    fn view(self: *Rows) cuda.View {
        return self.out.span(GUARD / 2, self.shape.rows);
    }
};

fn rowsMake(which: Control) *const fn (gpa: std.mem.Allocator, shape: contract.Shape) anyerror!contract.Case {
    return switch (which) {
        .max => rowsMakeMax,
        .sum => rowsMakeSum,
        .alias => rowsMakeAlias,
        .tail => rowsMakeTail,
        .half => rowsMakeHalf,
    };
}

fn rowsMakeInner(gpa: std.mem.Allocator, shape: contract.Shape, which: Control) !contract.Case {
    const self = try gpa.create(Rows);
    errdefer gpa.destroy(self);
    var x: cuda.Buffer = try .init(.bf16, &.{ shape.rows, shape.n });
    errdefer x.deinit();
    var out: cuda.Buffer = try .init(.f32, &.{ shape.rows + GUARD, 1 });
    errdefer out.deinit();
    const x_bits = try gpa.alloc(u16, shape.rows * shape.n);
    errdefer gpa.free(x_bits);
    const out_bits = try gpa.alloc(u32, shape.rows + GUARD);
    errdefer gpa.free(out_bits);
    @memset(out_bits, SENTINEL_F32);
    try out.upload(std.mem.sliceAsBytes(out_bits));
    self.* = .{ .gpa = gpa, .shape = shape, .which = which, .x = x, .out = out, .x_bits = x_bits, .out_bits = out_bits };
    return .{ .ctx = self, .v = &rows_vtable };
}

fn rowsMakeMax(gpa: std.mem.Allocator, shape: contract.Shape) !contract.Case {
    return rowsMakeInner(gpa, shape, .max);
}
fn rowsMakeSum(gpa: std.mem.Allocator, shape: contract.Shape) !contract.Case {
    return rowsMakeInner(gpa, shape, .sum);
}
fn rowsMakeAlias(gpa: std.mem.Allocator, shape: contract.Shape) !contract.Case {
    return rowsMakeInner(gpa, shape, .alias);
}
fn rowsMakeTail(gpa: std.mem.Allocator, shape: contract.Shape) !contract.Case {
    return rowsMakeInner(gpa, shape, .tail);
}
fn rowsMakeHalf(gpa: std.mem.Allocator, shape: contract.Shape) !contract.Case {
    return rowsMakeInner(gpa, shape, .half);
}

fn rowsDeinit(c: contract.Case, gpa: std.mem.Allocator) void {
    const self = cast(Rows, c);
    self.x.deinit();
    self.out.deinit();
    gpa.free(self.x_bits);
    gpa.free(self.out_bits);
    gpa.destroy(self);
}

fn rowsFill(c: contract.Case, fill: contract.Fill, seed: u64) void {
    const self = cast(Rows, c);
    var prng: std.Random.DefaultPrng = .init(seed);
    const rng = prng.random();
    for (self.x_bits) |*b| b.* = randomBf16(rng, 1.0);
    switch (fill) {
        .random => {},
        .nonfinite => placeNonFinite(self.x_bits, self.shape.rows, self.shape.n),
        .integers => for (self.x_bits, 0..) |*b, i| {
            b.* = numeric.bf16FromF32(oddValue(i));
        },
    }
    self.x.upload(std.mem.sliceAsBytes(self.x_bits)) catch unreachable;
}

fn rowsRun(c: contract.Case) !void {
    const self = cast(Rows, c);
    var out = self.view();
    const status = switch (self.which) {
        .max => zs_contract_max(&self.x.view, &out),
        .sum => zs_contract_sum(&self.x.view, &out),
        .alias => zs_contract_alias(&self.x.view, &out),
        .tail => zs_contract_tail(&self.x.view, &out),
        .half => zs_contract_half(&self.x.view, &out),
    };
    try cuda.launched(status);
    try cuda.synchronize();
}

fn rowsOutLen(c: contract.Case) usize {
    const self = cast(Rows, c);
    return self.shape.rows;
}

fn rowsOutput(c: contract.Case, dest: []u8) !void {
    const self = cast(Rows, c);
    try self.out.download(std.mem.sliceAsBytes(self.out_bits));
    @memcpy(dest[0 .. self.shape.rows * 4], std.mem.sliceAsBytes(self.out_bits[1..][0..self.shape.rows]));
}

fn rowsWant(c: contract.Case, gpa: std.mem.Allocator, out: []f64, acc: contract.Acc) !void {
    _ = gpa;
    const self = cast(Rows, c);
    const n = self.shape.n;
    for (0..self.shape.rows) |r| {
        const row = self.x_bits[r * n ..][0..n];
        out[r] = switch (self.which) {
            .max => blk: {
                var any_nan = false;
                var best: f64 = -std.math.inf(f64);
                for (row) |b| {
                    const v = @as(f64, numeric.f32FromBf16(b));
                    if (std.math.isNan(v)) any_nan = true;
                    best = @max(best, v);
                }
                break :blk if (any_nan) std.math.nan(f64) else best;
            },
            else => blk: {
                switch (acc) {
                    .f64 => {
                        var s: f64 = 0;
                        for (row) |b| s += @as(f64, numeric.f32FromBf16(b));
                        break :blk s;
                    },
                    .f16 => {
                        var s: f16 = 0;
                        for (row) |b| s += @floatCast(numeric.f32FromBf16(b));
                        break :blk @floatCast(s);
                    },
                    .bf16 => {
                        var s: f32 = 0;
                        for (row) |b| s = numeric.f32FromBf16(numeric.bf16FromF32(s + numeric.f32FromBf16(b)));
                        break :blk @as(f64, s);
                    },
                }
            },
        };
    }
}

fn rowsMagnitude(c: contract.Case, gpa: std.mem.Allocator, out: []f64) !void {
    _ = gpa;
    const self = cast(Rows, c);
    const n = self.shape.n;
    for (0..self.shape.rows) |r| {
        var mag: f64 = 0;
        for (self.x_bits[r * n ..][0..n]) |b| mag += @abs(@as(f64, numeric.f32FromBf16(b)));
        out[r] = mag;
    }
}

fn rowsInputs(c: contract.Case, gpa: std.mem.Allocator, dest: []u8) !void {
    _ = gpa;
    const self = cast(Rows, c);
    @memcpy(dest, std.mem.sliceAsBytes(self.x_bits));
}

fn rowsInputLen(c: contract.Case) usize {
    const self = cast(Rows, c);
    return self.x_bits.len * 2;
}

fn rowsViews(c: contract.Case, out: []cuda.View) usize {
    const self = cast(Rows, c);
    out[0] = self.x.view;
    out[1] = self.view();
    return 2;
}

fn rowsGuards(c: contract.Case) !void {
    const self = cast(Rows, c);
    try self.out.download(std.mem.sliceAsBytes(self.out_bits));
    if (!sentinelF32(self.out_bits[0])) return error.OutputOverran;
    if (!sentinelF32(self.out_bits[self.shape.rows + 1])) return error.OutputOverran;
}

const rows_vtable: contract.Case.VTable = .{
    .deinit = rowsDeinit,
    .fill = rowsFill,
    .run = rowsRun,
    .outLen = rowsOutLen,
    .output = rowsOutput,
    .want = rowsWant,
    .magnitude = rowsMagnitude,
    .inputs = rowsInputs,
    .inputLen = rowsInputLen,
    .views = rowsViews,
    .checkGuards = rowsGuards,
};

fn controlSpec(name: []const u8, symbol: []const u8, which: Control) contract.Spec {
    return .{
        .name = name,
        .shapes = &control_shapes,
        .kind = .f32,
        .depth = 1,
        .launch = .{ .symbol = symbol, .threads = 1, .registers = 32, .bound = 1, .shared = 0 },
        .make = rowsMake(which),
    };
}

// ------------------------------------------------------------------------------------------------
// The tests

test "contract.zs_b16_gemm passes gates 1 to 5" {
    try contract.runAll(std.testing.allocator, gemm_spec);
}

test "contract.zs_rmsnorm passes gates 1 to 5" {
    try contract.runAll(std.testing.allocator, norm_spec);
}

test "contract.control max by fmax drops NaN, reported on gate 1" {
    const spec = controlSpec("control_max", "zs_contract_max", .max);
    try std.testing.expectError(contract.Error.NonFiniteClass, contract.runGate(std.testing.allocator, spec, .nonfinite));
}

test "contract.control writes its input, reported on gate 2" {
    const spec = controlSpec("control_alias", "zs_contract_alias", .alias);
    try std.testing.expectError(contract.Error.RepeatBytes, contract.runGate(std.testing.allocator, spec, .repeat));
}

test "contract.control drops the ragged tail, reported on gate 3" {
    const spec = controlSpec("control_tail", "zs_contract_tail", .tail);
    try std.testing.expectError(contract.Error.OutsideTolerance, contract.runGate(std.testing.allocator, spec, .shapes));
}

test "contract.control sums in bf16, reported on gate 4" {
    const spec = controlSpec("control_half", "zs_contract_half", .half);
    try std.testing.expectError(contract.Error.OutsideTolerance, contract.runGate(std.testing.allocator, spec, .fp32));
}

test "contract.a launch past the GB10's limits is refused by gate 5" {
    var spec = controlSpec("control_launch", "zs_contract_sum", .sum);
    spec.launch = .{ .symbol = "zs_contract_sum", .threads = 1024, .registers = 128, .bound = 1024, .shared = 0 };
    try std.testing.expectError(contract.Error.LaunchBudget, contract.runGate(std.testing.allocator, spec, .launch));
}
