//! What the opt-in `PROFILE` tests of glm5 and qwen4 share: the monotonic clock in nanoseconds, the checks that a
//! launch ran on what it was given, the random fill of a buffer and the marker `nsys` finds a case by. Their
//! `Set` (which buffers a case holds) and the cases stay in each.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../src/cuda/cuda.zig");
const numeric = @import("../../src/weights/numeric.zig");

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Every value finite and some not zero: the launch ran on what it was given.
pub fn expectRan(gpa: Allocator, comptime T: type, view: cuda.View) !void {
    const host = try gpa.alloc(T, view.count());
    defer gpa.free(host);
    try cuda.download(view, std.mem.sliceAsBytes(host));
    var nonzero: usize = 0;
    for (host) |value| {
        const v: f32 = if (T == f32) value else numeric.f32FromBf16(value);
        try std.testing.expect(std.math.isFinite(v));
        nonzero += @intFromBool(v != 0);
    }
    try std.testing.expect(nonzero > host.len / 2);
}

/// Every value of a case's result finite: the launch ran on what it was given.
pub fn expectFinite(gpa: Allocator, view: cuda.View) !void {
    try cuda.synchronize();
    const bytes = try gpa.alloc(u8, view.count() * view.dtype.size());
    defer gpa.free(bytes);
    try cuda.download(view, bytes);
    switch (view.dtype) {
        .bf16 => for (std.mem.bytesAsSlice(u16, bytes)) |bits| try std.testing.expect(std.math.isFinite(numeric.f32FromBf16(bits))),
        .f32 => for (std.mem.bytesAsSlice(f32, bytes)) |x| try std.testing.expect(std.math.isFinite(x)),
        else => {},
    }
}

/// What a buffer starts as.
pub const Fill = enum {
    zero,
    /// Standard normal values.
    normal,
    /// A tenth of that: weights, a state.
    small,
    /// Near one: a norm's scale.
    unit,
};

/// A megabyte of each kind of random value, made once: a buffer's fill is a walk over it from a random place.
pub const Noise = struct {
    random: std.Random,
    normal: []f32,

    pub const count = 1 << 18;

    pub fn init(gpa: Allocator, random: std.Random) !Noise {
        const normal = try gpa.alloc(f32, count);
        for (normal) |*x| x.* = random.floatNorm(f32);
        return .{ .random = random, .normal = normal };
    }

    pub fn deinit(self: *Noise, gpa: Allocator) void {
        gpa.free(self.normal);
    }

    pub fn value(self: Noise, fill: Fill, at: usize) f32 {
        const x = self.normal[at % count];
        return switch (fill) {
            .zero => 0,
            .normal => x,
            .small => 0.1 * x,
            .unit => 1.0 + 0.05 * x,
        };
    }
};

/// A case's launches lie between two `cudaMemset`s of this buffer, of `mark + 2 i` and `mark + 2 i + 1` bytes.
pub const Marker = struct {
    buffer: cuda.Buffer,
    mark: usize,

    pub fn at(self: Marker, index: usize, end: bool) !void {
        try cuda.synchronize();
        var view = self.buffer.view;
        view.size[0] = @intCast(self.mark + 2 * index + @intFromBool(end));
        try cuda.zeroLaunched(view);
        try cuda.synchronize();
    }
};
