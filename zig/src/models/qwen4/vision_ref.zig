//! Host arithmetic the vision tower's kernels are held against (`vision.zig`'s tests): the attention by image
//! and the biased linear with its GELU (a quantized weight's values given in bf16), in f64 from the same bf16 inputs. Near, not to the bit: the kernels sum
//! in fp32 in their own order.

const std = @import("std");
const numeric = @import("../../weights/numeric.zig");
const vision = @import("vision.zig");

fn value(bits: u16) f64 {
    return numeric.f32FromBf16(bits);
}

/// out f64 [P, 1152] = each segment's attention over its own patches from qkv bf16 [P, 3456]; `segments` are
/// the frames' patch counts in order.
pub fn attention(qkv: []const u16, segments: []const usize, out: []f64, scores: []f64) void {
    const width = 3 * vision.hidden;
    const scale = 1.0 / @sqrt(@as(f64, vision.head_dim));
    var start: usize = 0;
    for (segments) |n| {
        for (0..vision.heads) |head| {
            for (start..start + n) |q| {
                var top: f64 = -std.math.inf(f64);
                for (start..start + n, 0..) |k, j| {
                    var dot: f64 = 0;
                    for (0..vision.head_dim) |d| dot += value(qkv[q * width + head * vision.head_dim + d]) * value(qkv[k * width + vision.hidden + head * vision.head_dim + d]);
                    scores[j] = dot * scale;
                    top = @max(top, scores[j]);
                }
                var sum: f64 = 0;
                for (scores[0..n]) |*s| {
                    s.* = @exp(s.* - top);
                    sum += s.*;
                }
                for (0..vision.head_dim) |d| {
                    var acc: f64 = 0;
                    for (start..start + n, 0..) |k, j| acc += scores[j] * value(qkv[k * width + 2 * vision.hidden + head * vision.head_dim + d]);
                    out[q * vision.hidden + head * vision.head_dim + d] = acc / sum;
                }
            }
        }
        start += n;
    }
}

/// out f64 (m, n) = act(bf(x (m, k) W^T + bias)), the activation of the bf16-rounded sum as `gemm_bias.cu`.
pub fn linear(x: []const u16, w: []const u16, bias: []const u16, m: usize, n: usize, k: usize, act: vision.Act, out: []f64) void {
    linearScaled(x, w, 1, bias, m, n, k, act, out);
}

/// `linear` with the product times `scale` before the bias, as `prompt_bias.cu` takes a quantized weight's
/// tensor scale; `w` the weight's values without it (exact in bf16: a code times its block scale).
pub fn linearScaled(x: []const u16, w: []const u16, scale: f64, bias: []const u16, m: usize, n: usize, k: usize, act: vision.Act, out: []f64) void {
    for (0..m) |r| {
        for (0..n) |c| {
            var acc: f64 = 0;
            for (0..k) |i| acc += value(x[r * k + i]) * value(w[c * k + i]);
            const v: f64 = numeric.f32FromBf16(numeric.bf16FromF32(@floatCast(acc * scale + value(bias[c]))));
            out[r * n + c] = switch (act) {
                .none => v,
                .gelu_tanh => 0.5 * v * (1 + std.math.tanh(@sqrt(2.0 / std.math.pi) * (v + 0.044715 * v * v * v))),
                .gelu_erf => 0.5 * v * (1 + erf(v / @sqrt(2.0))),
            };
        }
    }
}

/// erf to f64's precision by its Taylor series near 0 and its continued fraction beyond (Abramowitz and
/// Stegun 7.1.5 and 7.1.14): what a test of GELU needs, 1e-15.
pub fn erf(x: f64) f64 {
    const a = @abs(x);
    const sign: f64 = if (x < 0) -1 else 1;
    if (a < 2.5) {
        var term = a;
        var sum = a;
        var n: f64 = 0;
        while (n < 200) : (n += 1) {
            term *= -a * a / (n + 1);
            const add = term / (2 * n + 3);
            sum += add;
            if (@abs(add) < 1e-18) break;
        }
        return sign * sum * 2 / @sqrt(std.math.pi);
    }
    // erfc(a) = exp(-a^2) / sqrt(pi) * 1 / (a + 1/2 / (a + 1 / (a + 3/2 / (a + ...))))
    var f: f64 = a;
    var i: f64 = 60;
    while (i >= 1) : (i -= 1) f = a + (i / 2) / f;
    return sign * (1 - @exp(-a * a) / @sqrt(std.math.pi) / f);
}
