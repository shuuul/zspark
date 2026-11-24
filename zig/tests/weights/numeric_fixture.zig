//! What tests on the host's side of bf16 share: the logistic function and random bf16
//! rows. Nothing here reaches the device, so the host tier may import it.

const std = @import("std");
const numeric = @import("../../src/weights/numeric.zig");

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub fn randomBf16(random: std.Random, out: []u16, spread: f32) void {
    for (out) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32) * spread);
}
