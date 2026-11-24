//! The storage formats of weights and activations, on the host.

const std = @import("std");

/// Round to nearest even, as `__float2bfloat16_rn`.
pub fn bf16FromF32(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if (std.math.isNan(value)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

pub fn f32FromBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// `value` rounded to bf16 and back.
pub fn roundBf16(value: f32) f32 {
    return f32FromBf16(bf16FromF32(value));
}

/// An e4m3 byte (sign, 4 exponent bits biased by 7, 3 mantissa bits); 0x7f and 0xff are NaN.
pub fn f32FromE4m3(byte: u8) f32 {
    const sign: f32 = if (byte & 0x80 != 0) -1.0 else 1.0;
    const exponent: i32 = (byte >> 3) & 0xf;
    const mantissa: f32 = @floatFromInt(byte & 7);
    if (exponent == 15 and byte & 7 == 7) return std.math.nan(f32);
    if (exponent == 0) return sign * std.math.ldexp(mantissa / 8.0, -6);
    return sign * std.math.ldexp(1.0 + mantissa / 8.0, exponent - 7);
}

/// Round to nearest even, as PyTorch's cast to `float8_e4m3fn` (c10's `fp8e4m3fn_from_fp32_value`): 480 and
/// above, and NaN, give NaN (0x7f under the sign).
pub fn e4m3FromF32(value: f32) u8 {
    const bits: u32 = @bitCast(value);
    const sign: u8 = @intCast((bits >> 24) & 0x80);
    const magnitude = bits & 0x7fffffff;
    if (magnitude >= 1087 << 20) return sign | 0x7f;
    if (magnitude < 121 << 23) {
        // below 2^-6, the smallest normal: adding 2^14 rounds to the subnormal step of 2^-9
        const lifted: u32 = @bitCast(@as(f32, @bitCast(magnitude)) + 16384.0);
        return sign | @as(u8, @intCast(lifted - (141 << 23)));
    }
    const odd = (magnitude >> 20) & 1;
    const rounded = magnitude -% (120 << 23) + 0x7ffff + odd;
    return sign | @as(u8, @truncate(rounded >> 20));
}

/// One NVFP4 group of 16 activations as vLLM's b12x quantizes it (`quantize_block_fp4`), the device's
/// `experts_a4.cuh` to the bit: `gs` the global scale (1 / input_scale), the scale e4m3(min(max|x| * gs / 6, 448))
/// to nearest even, each value the e2m1 nearest x / (scale / gs) by `quantize_and_pack_16`'s thresholds, ties to
/// even; a scale of zero leaves the codes zero. Code i is bits 4i .. 4i + 3.
pub fn nvfp4Group(values: [16]f32, gs: f32) struct { codes: u64, scale: u8 } {
    var most: f32 = 0;
    for (values) |v| most = @max(most, @abs(v));
    const raw: f32 = @min(most * gs / 6.0, 448.0);
    const scale = e4m3FromF32(raw);
    const sf = f32FromE4m3(scale);
    if (sf == 0 or gs == 0) return .{ .codes = 0, .scale = scale };
    const vs: f32 = sf / gs;
    const k = [7]f32{ 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0 };
    var t: [7]f32 = undefined;
    for (&t, k) |*one, f| one.* = vs * f;
    var codes: u64 = 0;
    for (values, 0..) |v, i| {
        const a = @abs(v);
        var n: u64 = 0;
        if (a > t[0] and a < t[1]) n = 1 else if (a >= t[1] and a <= t[2]) n = 2 else if (a > t[2] and a < t[3]) n = 3 else if (a >= t[3] and a <= t[4]) n = 4 else if (a > t[4] and a < t[5]) n = 5 else if (a >= t[5] and a <= t[6]) n = 6 else if (a > t[6]) n = 7;
        if (n != 0 and v < 0) n |= 8;
        codes |= n << @intCast(4 * i);
    }
    return .{ .codes = codes, .scale = scale };
}

/// A ue8m0 scale byte's value, 2^(byte - 127) (MXFP8 weights, the MX block scales).
pub fn f32FromUe8m0(byte: u8) f32 {
    return std.math.ldexp(@as(f32, 1), @as(i32, byte) - 127);
}
