//! `numeric.zig`'s tests.

const std = @import("std");
const numeric = @import("../../src/weights/numeric.zig");
const bf16FromF32 = numeric.bf16FromF32;
const e4m3FromF32 = numeric.e4m3FromF32;
const f32FromBf16 = numeric.f32FromBf16;
const f32FromE4m3 = numeric.f32FromE4m3;

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16FromF32(1.0));
    try std.testing.expectEqual(@as(f32, 1.0), f32FromBf16(0x3f80));
    // 1 + 2^-8 is halfway between two bf16 values and goes to the even one; just above it goes up
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16FromF32(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16FromF32(@bitCast(@as(u32, 0x3f808001))));
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16FromF32(@bitCast(@as(u32, 0x3f818000))));
}

test "e4m3 values" {
    try std.testing.expectEqual(@as(f32, 1.0), f32FromE4m3(0x38));
    try std.testing.expectEqual(@as(f32, 448.0), f32FromE4m3(0x7e));
    try std.testing.expectEqual(@as(f32, -1.5), f32FromE4m3(0xbc));
    try std.testing.expectEqual(@as(f32, 0.001953125), f32FromE4m3(0x01));
    try std.testing.expect(std.math.isNan(f32FromE4m3(0x7f)));
}

test "e4m3 rounds to nearest even, through the subnormals, and every byte comes back" {
    for (0..256) |byte| {
        const code: u8 = @intCast(byte);
        if (code & 0x7f == 0x7f) continue;
        try std.testing.expectEqual(code, e4m3FromF32(f32FromE4m3(code)));
    }
    // halfway between 1 (0x38) and 1.125 (0x39) goes to the even one, as does halfway between 1.125 and 1.25
    try std.testing.expectEqual(@as(u8, 0x38), e4m3FromF32(1.0625));
    try std.testing.expectEqual(@as(u8, 0x3a), e4m3FromF32(1.1875));
    try std.testing.expectEqual(@as(u8, 0x39), e4m3FromF32(1.0626));
    // subnormals step by 2^-9: half a step goes to zero (even), one and a half to two
    try std.testing.expectEqual(@as(u8, 0x00), e4m3FromF32(0.0009765625));
    try std.testing.expectEqual(@as(u8, 0x01), e4m3FromF32(0.00098));
    try std.testing.expectEqual(@as(u8, 0x02), e4m3FromF32(0.0029296875));
    // the last subnormal rounds up into the first normal
    try std.testing.expectEqual(@as(u8, 0x08), e4m3FromF32(0.0155));
    try std.testing.expectEqual(@as(u8, 0x7e), e4m3FromF32(448.0));
    try std.testing.expectEqual(@as(u8, 0x7e), e4m3FromF32(463.9));
    try std.testing.expectEqual(@as(u8, 0xfe), e4m3FromF32(-448.0));
    try std.testing.expectEqual(@as(u8, 0x80), e4m3FromF32(-0.0));
    try std.testing.expectEqual(@as(u8, 0x7f), e4m3FromF32(480.0));
}
