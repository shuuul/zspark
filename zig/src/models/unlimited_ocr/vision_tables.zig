//! Unlimited-OCR's vision tables resampled to a view's grid, on the host at load (spec 137 WS-03): SAM's 64 x 64
//! position table to a 640 crop's 40 x 40 and CLIP's 16 x 16 to its 10 x 10 (`get_abs_pos_sam`, `get_abs_pos`:
//! `F.interpolate(mode='bicubic', antialias=True)` in fp32, then bf16), and the global blocks' 127-row relative
//! position tables to 79 (`get_rel_pos`: `F.interpolate(mode='linear')` in fp32, then bf16). The arithmetic is
//! PyTorch's CUDA kernels' (`upsample_gen2d_aa_out_frame` with `bicubic_filter`, a = -0.5, its weights' filter in
//! double; `upsample_linear1d_out_frame`), as the reference ran them on the GPU, products contracted to FMA as
//! nvcc does.

const std = @import("std");
const Allocator = std.mem.Allocator;
const numeric = @import("../../weights/numeric.zig");

/// PyTorch's `bicubic_filter` for the antialiased path (a = -0.5), in double as its literals make it.
fn cubic(x_in: f32) f32 {
    const a: f64 = -0.5;
    var x: f64 = x_in;
    if (x < 0) x = -x;
    if (x < 1.0) return @floatCast(((a + 2.0) * x - (a + 3.0)) * x * x + 1);
    if (x < 2.0) return @floatCast((((x - 5) * x + 8) * x - 4) * a);
    return 0;
}

/// One output index's taps: its first input, how many, their normalized weights (`_compute_weights_span` and
/// `_compute_weights`).
const Taps = struct {
    first: usize,
    count: usize,
    weights: [16]f32,

    fn of(i: usize, in_size: usize, out_size: usize) Taps {
        const scale: f32 = @as(f32, @floatFromInt(in_size)) / @as(f32, @floatFromInt(out_size));
        const support: f32 = if (scale >= 1.0) 2.0 * scale else 2.0;
        const center: f32 = scale * (@as(f32, @floatFromInt(i)) + 0.5);
        const invscale: f32 = if (scale >= 1.0) @floatCast(1.0 / @as(f64, scale)) else 1.0;
        const lo: i64 = @intFromFloat(center - support + 0.5);
        const first: usize = @intCast(@max(lo, 0));
        const hi: i64 = @intFromFloat(center + support + 0.5);
        const count: usize = @as(usize, @intCast(@min(hi, @as(i64, @intCast(in_size))))) - first;
        var taps: Taps = .{ .first = first, .count = count, .weights = @splat(0) };
        std.debug.assert(count <= taps.weights.len);
        var total: f32 = 0;
        for (0..count) |j| {
            const w = cubic((@as(f32, @floatFromInt(j + first)) - center + 0.5) * invscale);
            taps.weights[j] = w;
            total += w;
        }
        if (total != 0) {
            for (taps.weights[0..count]) |*w| w.* /= total;
        }
        return taps;
    }
};

/// `src` fp32 [in_side, in_side, channels] (rows, columns, channels last) resampled to [out_side, out_side,
/// channels] by PyTorch's antialiased bicubic: each output a sum over its rows' taps of the sums over their
/// columns' taps, from the first tap, in fp32.
pub fn bicubicAntialias(gpa: Allocator, src: []const f32, in_side: usize, out_side: usize, channels: usize) Allocator.Error![]f32 {
    std.debug.assert(src.len == in_side * in_side * channels);
    const out = try gpa.alloc(f32, out_side * out_side * channels);
    for (0..out_side) |oy| {
        const ty = Taps.of(oy, in_side, out_side);
        for (0..out_side) |ox| {
            const tx = Taps.of(ox, in_side, out_side);
            for (0..channels) |c| {
                var total: f32 = 0;
                for (0..ty.count) |y| {
                    const row = (ty.first + y) * in_side;
                    var line: f32 = src[(row + tx.first) * channels + c] * tx.weights[0];
                    for (1..tx.count) |x| line = @mulAdd(f32, src[(row + tx.first + x) * channels + c], tx.weights[x], line);
                    total = if (y == 0) line * ty.weights[0] else @mulAdd(f32, line, ty.weights[y], total);
                }
                out[(oy * out_side + ox) * channels + c] = total;
            }
        }
    }
    return out;
}

/// `src` fp32 [in_len, channels] resampled along its rows to [out_len, channels] by PyTorch's linear
/// interpolation (align_corners false, the source index clamped at 0).
pub fn linear(gpa: Allocator, src: []const f32, in_len: usize, out_len: usize, channels: usize) Allocator.Error![]f32 {
    std.debug.assert(src.len == in_len * channels);
    const out = try gpa.alloc(f32, out_len * channels);
    const scale: f32 = @as(f32, @floatFromInt(in_len)) / @as(f32, @floatFromInt(out_len));
    for (0..out_len) |i| {
        const raw = @mulAdd(f32, scale, @as(f32, @floatFromInt(i)) + 0.5, -0.5);
        const at: f32 = if (raw < 0) 0 else raw;
        const h: usize = @intFromFloat(at);
        const step: usize = if (h < in_len - 1) 1 else 0;
        const l1 = at - @as(f32, @floatFromInt(h));
        const l0 = 1.0 - l1;
        for (0..channels) |c| out[i * channels + c] = @mulAdd(f32, l1, src[(h + step) * channels + c], l0 * src[h * channels + c]);
    }
    return out;
}

/// bf16 bits as fp32.
pub fn widen(gpa: Allocator, bits: []const u8) Allocator.Error![]f32 {
    const out = try gpa.alloc(f32, bits.len / 2);
    for (out, 0..) |*v, i| v.* = numeric.f32FromBf16(std.mem.readInt(u16, bits[2 * i ..][0..2], .little));
    return out;
}

/// fp32 rounded to bf16 bits, as `.to(bfloat16)`.
pub fn narrow(gpa: Allocator, values: []const f32) Allocator.Error![]u16 {
    const out = try gpa.alloc(u16, values.len);
    for (values, out) |v, *b| b.* = numeric.bf16FromF32(v);
    return out;
}
