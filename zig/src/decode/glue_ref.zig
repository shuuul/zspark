//! What the models' host glue (`glue_ref.zig` of glm5) holds alike: the bf16 reads, a four-way
//! fold, the ranks' branch sum and the expert products. The `hcPost` of each stays in its model, since glm5 fuses
//! (`@mulAdd`). No device, no allocation.

const numeric = @import("../weights/numeric.zig");

pub fn value(bits: u16) f32 {
    return numeric.f32FromBf16(bits);
}

pub fn fold4(a: f32, b: f32, c: f32, d: f32) f32 {
    return (a + c) + (b + d);
}

/// The ranks' partials of element `d`, rank 0 first, rounded to bf16.
pub fn branchOf(gathered: []const []const f32, d: usize) f32 {
    var acc = gathered[0][d];
    for (gathered[1..]) |rank| acc += rank[d];
    return numeric.roundBf16(acc);
}

/// gu bf16 [2 * width] (gate | up) -> act bf16 [width], clamped at `limit` (TensorFold's `swiglu_limit`).
pub fn swiglu(gu: []const u16, act: []u16, limit: f32) void {
    const width = act.len;
    for (act, gu[0..width], gu[width..]) |*a, g, u| {
        const gate = @min(value(g), limit);
        const up = @min(@max(value(u), -limit), limit);
        a.* = numeric.bf16FromF32(numeric.roundBf16(gate / (1.0 + @exp(-gate))) * up);
    }
}

/// y fp32 [slots, dims], weights [slots] -> out fp32 [dims], a rank's share of the block.
pub fn combine(y: []const f32, weights: []const f32, out: []f32) void {
    const dims = out.len;
    for (out, 0..) |*o, d| {
        var acc: f32 = 0;
        for (weights, 0..) |w, k| acc = @mulAdd(f32, y[k * dims + d], w, acc);
        o.* = acc;
    }
}
