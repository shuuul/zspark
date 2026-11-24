//! The kernel contract (spec 158 WS-02): five gates a registered kernel must pass, and the f64 reference each
//! one is judged against. Read `docs/kernels.md`, "The kernel contract", before registering a kernel.
//!
//! Gate 1 — non-finite: NaN and both infinities go into the inputs at known places; every output's class
//! (finite, NaN, +Inf, -Inf) must equal the f64 reference's. A kernel with a stated rule does not register
//! here yet.
//! Gate 2 — repeat and aliasing: five runs on the same inputs give the same bytes, the inputs are unchanged,
//! the guards an adapter placed are intact, and no output overlaps an input unless the spec says in place.
//! Gate 3 — shapes: every shape the adapter declares (its served shapes and their tile edges), each output
//! within `|want| * ulp/2 + gamma(depth) * magnitude`.
//! Gate 4 — fp32 sums: inputs whose terms are integers and whose fp32 partials are exact; the kernel must
//! stay inside gate 3's bound, while the same sum replayed in f16 and in bf16 must miss the fp32 bound by 100
//! times (the output's own step is not the measure: a narrow output's rounding would hide an f16 sum).
//! Gate 5 — the launch: threads, registers, shared memory and blocks an SM against the GB10's limits
//! (`cuda-gb10`: 1,536 threads and 65,536 registers an SM, 102,400 B of shared memory an SM, 101,376 B a
//! block opted in, 24 blocks an SM).

const std = @import("std");
const cuda = @import("../../src/cuda/cuda.zig");
const numeric = @import("../../src/weights/numeric.zig");

pub const Error = error{
    NonFiniteClass,
    RepeatBytes,
    InputChanged,
    GuardBroken,
    Aliased,
    OutsideTolerance,
    HostReplayDidNotMiss,
    LaunchBudget,
    ShapeMissing,
};

/// What a shape means for the adapter; the harness only passes it through.
pub const Shape = struct { rows: usize, n: usize, k: usize };

/// How the harness fills a case's inputs before a gate runs.
pub const Fill = enum {
    /// Values the reference and the kernel can both take, from a seeded PRNG.
    random,
    /// `random`, then NaN, +Inf, -Inf in the first rows at known columns.
    nonfinite,
    /// Gate 4's integers: terms whose fp32 partials are exact and whose f16/bf16 replay misses the fp32
    /// bound by 100 times.
    integers,
};

/// The accumulator a replay sums in (gate 4).
pub const Acc = enum { f64, f16, bf16 };

/// An output's storage.
pub const Kind = enum {
    f32,
    bf16,

    pub fn size(self: Kind) usize {
        return if (self == .f32) 4 else 2;
    }
};

pub const Nan = enum { propagate, rule };

/// A launch as the kernel declares it; gate 5 checks it against the chip.
pub const Launch = struct {
    symbol: []const u8,
    threads: usize,
    registers: usize,
    /// `__launch_bounds__`'s first argument.
    bound: usize,
    shared: usize = 0,
};

/// One registered kernel.
pub const Spec = struct {
    name: []const u8,
    shapes: []const Shape,
    kind: Kind,
    /// Roundings between the sums and the output: the depth gate 3's gamma charges.
    depth: usize,
    nan: Nan = .propagate,
    in_place: bool = false,
    launch: Launch,
    make: *const fn (gpa: std.mem.Allocator, shape: Shape) anyerror!Case,
};

/// A shaped instance of a kernel: its buffers, its inputs and its reference.
pub const Case = struct {
    ctx: *anyopaque,
    v: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (c: Case, gpa: std.mem.Allocator) void,
        /// Writes the inputs for a gate and uploads them.
        fill: *const fn (c: Case, fill: Fill, seed: u64) void,
        /// Launches the kernel and waits.
        run: *const fn (c: Case) anyerror!void,
        /// Elements the kernel writes.
        outLen: *const fn (c: Case) usize,
        /// The output's bytes, raw: `kind.size()` a value.
        output: *const fn (c: Case, dest: []u8) anyerror!void,
        /// The kernel's formula on the current inputs. `acc` is the f64 reference for `.f64`, and a replay
        /// that sums the same terms in f16 or bf16 for the other two.
        want: *const fn (c: Case, gpa: std.mem.Allocator, out: []f64, acc: Acc) anyerror!void,
        /// The magnitude of the terms each output sums (the same length as the output).
        magnitude: *const fn (c: Case, gpa: std.mem.Allocator, out: []f64) anyerror!void,
        /// Every input's bytes, concatenated, for the unchanged check.
        inputs: *const fn (c: Case, gpa: std.mem.Allocator, dest: []u8) anyerror!void,
        /// How many bytes `inputs` writes.
        inputLen: *const fn (c: Case) usize,
        /// The input and output views, for the overlap check.
        views: *const fn (c: Case, out: []cuda.View) usize,
        /// The guard bytes the adapter placed around the output.
        checkGuards: *const fn (c: Case) anyerror!void,
    };
};

pub const Gate = enum { nonfinite, repeat, shapes, fp32, launch };

/// The five gates, in order.
pub fn runAll(gpa: std.mem.Allocator, spec: Spec) !void {
    try runGate(gpa, spec, .nonfinite);
    try runGate(gpa, spec, .repeat);
    try runGate(gpa, spec, .shapes);
    try runGate(gpa, spec, .fp32);
    try runGate(gpa, spec, .launch);
}

pub fn runGate(gpa: std.mem.Allocator, spec: Spec, gate: Gate) !void {
    switch (gate) {
        .launch => return gateLaunch(spec),
        .nonfinite => return gateNonFinite(gpa, spec),
        .repeat => return gateRepeat(gpa, spec),
        .shapes => return gateShapes(gpa, spec),
        .fp32 => return gateFp32(gpa, spec),
    }
}

fn outBytes(gpa: std.mem.Allocator, c: Case, kind: Kind) ![]u8 {
    const bytes = try gpa.alloc(u8, c.v.outLen(c) * kind.size());
    errdefer gpa.free(bytes);
    return bytes;
}

fn readValue(bytes: []const u8, i: usize, kind: Kind) f64 {
    return switch (kind) {
        .f32 => @as(f64, @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little))))),
        .bf16 => numeric.f32FromBf16(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)),
    };
}

/// Rounds `x` to the output's storage, as the kernel's last write does.
fn roundTo(x: f64, kind: Kind) f64 {
    return switch (kind) {
        .f32 => @as(f64, @floatCast(@as(f32, @floatCast(x)))),
        .bf16 => numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(x))),
    };
}

const Class = enum { finite, nan, inf, ninf };

fn classOfF64(x: f64) Class {
    if (std.math.isNan(x)) return .nan;
    if (std.math.isInf(x)) return if (x > 0) .inf else .ninf;
    return .finite;
}

fn classOfBits(bytes: []const u8, i: usize, kind: Kind) Class {
    return switch (kind) {
        .f32 => blk: {
            const bits = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
            const exp = (bits >> 23) & 0xff;
            const mant = bits & 0x7f_ffff;
            if (exp == 0xff) break :blk if (mant != 0) .nan else if (bits >> 31 != 0) .ninf else .inf;
            break :blk .finite;
        },
        .bf16 => blk: {
            const bits = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
            const exp = (bits >> 7) & 0xff;
            const mant = bits & 0x7f;
            if (exp == 0xff) break :blk if (mant != 0) .nan else if (bits >> 15 != 0) .ninf else .inf;
            break :blk .finite;
        },
    };
}

/// `|want| * half an ulp + gamma(depth) * magnitude`, the bound gate 3 and 4 use.
fn tolerance(kind: Kind, want: f64, magnitude: f64, depth: usize) f64 {
    const u: f64 = if (kind == .f32) 0x1p-24 else 0x1p-8;
    const n: f64 = @floatFromInt(@max(depth, 1));
    const gamma = n * u / (1 - n * u);
    return @abs(want) * u / 2 + gamma * magnitude;
}

fn gateNonFinite(gpa: std.mem.Allocator, spec: Spec) !void {
    if (spec.nan == .rule) return;
    for (spec.shapes) |shape| {
        var c = try spec.make(gpa, shape);
        defer c.v.deinit(c, gpa);
        c.v.fill(c, .nonfinite, 1);
        try c.v.run(c);
        const len = c.v.outLen(c);
        const want = try gpa.alloc(f64, len);
        defer gpa.free(want);
        try c.v.want(c, gpa, want, .f64);
        const got = try outBytes(gpa, c, spec.kind);
        defer gpa.free(got);
        try c.v.output(c, got);
        for (want, 0..) |w, i| {
            const got_class = classOfBits(got, i, spec.kind);
            const want_class = classOfF64(w);
            if (got_class != want_class) {
                std.debug.print("contract {s} gate 1: element {d} of {d}: kernel {s}, reference {s}\n", .{
                    spec.name, i, len, @tagName(got_class), @tagName(want_class),
                });
                return Error.NonFiniteClass;
            }
        }
    }
}

fn gateRepeat(gpa: std.mem.Allocator, spec: Spec) !void {
    for (spec.shapes) |shape| {
        var c = try spec.make(gpa, shape);
        defer c.v.deinit(c, gpa);
        c.v.fill(c, .random, 2);
        const before = try gpa.alloc(u8, c.v.inputLen(c));
        defer gpa.free(before);
        try c.v.inputs(c, gpa, before);
        const first = try outBytes(gpa, c, spec.kind);
        defer gpa.free(first);
        try c.v.run(c);
        try c.v.output(c, first);
        try checkOverlap(c, spec);
        for (1..5) |_| {
            const next = try outBytes(gpa, c, spec.kind);
            defer gpa.free(next);
            try c.v.run(c);
            try c.v.output(c, next);
            if (!std.mem.eql(u8, first, next)) {
                std.debug.print("contract {s} gate 2: run {d} differs from the first\n", .{ spec.name, 1 });
                return Error.RepeatBytes;
            }
        }
        const after = try gpa.alloc(u8, before.len);
        defer gpa.free(after);
        try c.v.inputs(c, gpa, after);
        if (!std.mem.eql(u8, before, after)) {
            std.debug.print("contract {s} gate 2: the inputs changed\n", .{spec.name});
            return Error.InputChanged;
        }
        c.v.checkGuards(c) catch |e| {
            std.debug.print("contract {s} gate 2: guard bytes: {s}\n", .{ spec.name, @errorName(e) });
            return Error.GuardBroken;
        };
    }
}

fn gateShapes(gpa: std.mem.Allocator, spec: Spec) !void {
    for (spec.shapes) |shape| {
        var c = try spec.make(gpa, shape);
        defer c.v.deinit(c, gpa);
        c.v.fill(c, .random, 3);
        try c.v.run(c);
        try check(gpa, c, spec, shape, .shapes);
    }
}

fn gateFp32(gpa: std.mem.Allocator, spec: Spec) !void {
    for (spec.shapes) |shape| {
        var c = try spec.make(gpa, shape);
        defer c.v.deinit(c, gpa);
        c.v.fill(c, .integers, 4);
        try c.v.run(c);
        try check(gpa, c, spec, shape, .fp32);
        const len = c.v.outLen(c);
        const exact = try gpa.alloc(f64, len);
        defer gpa.free(exact);
        try c.v.want(c, gpa, exact, .f64);
        const magnitude = try gpa.alloc(f64, len);
        defer gpa.free(magnitude);
        try c.v.magnitude(c, gpa, magnitude);
        inline for ([_]Acc{ .f16, .bf16 }) |acc| {
            const replay = try gpa.alloc(f64, len);
            defer gpa.free(replay);
            try c.v.want(c, gpa, replay, acc);
            // The replay is judged against the fp32 sum's own bound, not the output's: a narrower output's
            // rounding would otherwise hide an f16 accumulator (the rmsnorm's bf16 step is larger than an
            // f16 sum's error at its shapes). The worst element over the shape must miss it by 100 times.
            var worst: f64 = 0;
            var worst_at: usize = 0;
            for (exact, replay, magnitude, 0..) |ex, rp, mag, i| {
                if (!std.math.isFinite(ex)) continue;
                if (!std.math.isFinite(rp)) {
                    worst = std.math.inf(f64);
                    worst_at = i;
                    break;
                }
                const missed = @abs(rp - ex);
                const bound = tolerance(.f32, ex, mag, spec.depth);
                const ratio = if (bound > 0) missed / bound else if (missed > 0) std.math.inf(f64) else 0;
                if (ratio > worst) {
                    worst = ratio;
                    worst_at = i;
                }
            }
            if (worst < 100) {
                std.debug.print("contract {s} gate fp32: shape rows {d} n {d} k {d}: the {s} replay misses by {e} times the fp32 bound at element {d}, under 100\n", .{
                    spec.name, shape.rows, shape.n, shape.k, @tagName(acc), worst, worst_at,
                });
                return Error.HostReplayDidNotMiss;
            }
        }
    }
}

fn check(gpa: std.mem.Allocator, c: Case, spec: Spec, shape: Shape, gate: Gate) !void {
    const len = c.v.outLen(c);
    const want = try gpa.alloc(f64, len);
    defer gpa.free(want);
    try c.v.want(c, gpa, want, .f64);
    const magnitude = try gpa.alloc(f64, len);
    defer gpa.free(magnitude);
    try c.v.magnitude(c, gpa, magnitude);
    const got = try outBytes(gpa, c, spec.kind);
    defer gpa.free(got);
    try c.v.output(c, got);
    for (want, magnitude, 0..) |w, mag, i| {
        const got_value = readValue(got, i, spec.kind);
        const bound = tolerance(spec.kind, w, mag, spec.depth);
        if (!(std.math.isNan(w) and std.math.isNan(got_value)) and
            !(std.math.isNan(w) == std.math.isNan(got_value) and @abs(got_value - w) <= bound))
        {
            std.debug.print("contract {s} gate {s}: shape rows {d} n {d} k {d}, element {d}: want {e}, got {e}, bound {e}\n", .{
                spec.name, @tagName(gate), shape.rows, shape.n, shape.k, i, w, got_value, bound,
            });
            return Error.OutsideTolerance;
        }
    }
}

fn checkOverlap(c: Case, spec: Spec) !void {
    var views: [8]cuda.View = undefined;
    const count = c.v.views(c, &views);
    var ranges: [8]struct { at: usize, len: usize } = undefined;
    for (views[0..count], 0..) |view, i| {
        const rows: usize = @intCast(view.size[0]);
        var inner: usize = 1;
        for (view.size[1..@intCast(view.ndim)]) |d| inner *= @intCast(d);
        const row = if (view.row_stride != 0) @as(usize, @intCast(view.row_stride)) else inner;
        ranges[i] = .{ .at = @intFromPtr(view.ptr.?), .len = rows * row * view.dtype.size() };
    }
    for (ranges[0..count], 0..) |a, i| {
        for (ranges[0..count], 0..) |b, j| {
            if (i >= j) continue;
            const disjoint = a.at + a.len <= b.at or b.at + b.len <= a.at;
            if (disjoint) continue;
            // The output is the last view; an input overlapping it is in place only when declared.
            const touches_output = i == count - 1 or j == count - 1;
            if (touches_output and !spec.in_place) {
                std.debug.print("contract {s} gate 2: views {d} and {d} overlap\n", .{ spec.name, i, j });
                return Error.Aliased;
            }
        }
    }
}

/// Gate 5: the GB10's limits (`cuda-gb10`).
pub fn gateLaunch(spec: Spec) !void {
    const l = spec.launch;
    if (l.threads == 0 or l.threads > 1024) return budget(spec, "threads a block");
    if (l.bound < l.threads) return budget(spec, "threads over __launch_bounds__");
    if (l.registers == 0 or l.registers > 255) return budget(spec, "registers a thread");
    if (l.threads * l.registers > 65_536) return budget(spec, "registers a block");
    if (l.shared > 101_376) return budget(spec, "shared memory a block");
    if (l.shared + 1024 > 102_400) return budget(spec, "shared memory an SM's block");
    const by_threads = 1_536 / l.threads;
    const by_regs = 65_536 / (l.threads * l.registers);
    const by_shared = 102_400 / (l.shared + 1024);
    if (@min(@min(by_threads, by_regs), @min(by_shared, 24)) < 1) return budget(spec, "a block that does not fit");
}

fn budget(spec: Spec, what: []const u8) Error {
    std.debug.print("contract {s} gate 5: {s} ({s})\n", .{ spec.name, what, spec.launch.symbol });
    return Error.LaunchBudget;
}
