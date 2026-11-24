//! What `hc_test.zig` calls: random bf16 values, a buffer made from host values and back, the fixture's shape and
//! its random inputs.

const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric_fixture = @import("../../../weights/numeric_fixture.zig");
const gpu = @import("../../../cuda/testing.zig");

pub const randomBf16 = numeric_fixture.randomBf16;

pub const device = gpu.device;

pub const fetch = gpu.fetch;

/// The fixture's shape.
pub const fx = struct {
    pub const dims = 512;
    pub const streams = 4;
    pub const wide = streams * dims;
    pub const low = 64;
    pub const slots = 3;
    pub const eps = 1e-6;
};

pub const Inputs = struct {
    h: []u16,
    scale: []f32,
    down: []f32,
    up: []u16,
    branch: []u16,
    y: []f32,
    shared: []f32,
    weights: []f32,

    pub fn init(gpa: std.mem.Allocator, rows: usize) !Inputs {
        var prng: std.Random.DefaultPrng = .init(11);
        const random = prng.random();
        const self: Inputs = .{
            .h = try gpa.alloc(u16, rows * fx.wide),
            .scale = try gpa.alloc(f32, fx.wide),
            .down = try gpa.alloc(f32, rows * (fx.low + fx.streams)),
            .up = try gpa.alloc(u16, rows * fx.wide),
            .branch = try gpa.alloc(u16, rows * fx.dims),
            .y = try gpa.alloc(f32, rows * fx.slots * fx.dims),
            .shared = try gpa.alloc(f32, rows * fx.dims),
            .weights = try gpa.alloc(f32, rows * fx.slots),
        };
        randomBf16(random, self.h, 2.0);
        for (self.scale) |*v| v.* = 0.5 + random.float(f32);
        for (self.down) |*v| v.* = random.floatNorm(f32) * 6.0;
        randomBf16(random, self.up, 2.0);
        randomBf16(random, self.branch, 1.0);
        for (self.y, 0..) |*v, i| {
            // the last slot is the shared expert's, which comes from its own buffer: NaN here must not be read
            v.* = if (i / fx.dims % fx.slots == fx.slots - 1) std.math.nan(f32) else random.floatNorm(f32);
        }
        for (self.shared) |*v| v.* = random.floatNorm(f32);
        for (self.weights) |*v| v.* = random.float(f32);
        return self;
    }

    pub fn deinit(self: Inputs, gpa: std.mem.Allocator) void {
        inline for (comptime std.meta.fieldNames(Inputs)) |name| gpa.free(@field(self, name));
    }
};
