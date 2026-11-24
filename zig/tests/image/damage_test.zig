//! Damaged pictures: every case of `fixtures/image/cases.json` cut short at many lengths and with bytes flipped,
//! each decoded and either refused or decoded, never a crash (a safe build checks every access, the C ones
//! under the sanitizer `build.zig` leaves on) and never a leak.

const std = @import("std");
const image = @import("../../src/image/image.zig");

const Case = struct { name: []const u8, format: []const u8, bytes: []const u8 };

fn survive(bytes: []const u8) !void {
    var got = image.decode(std.testing.allocator, bytes) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return,
    };
    got.deinit(std.testing.allocator);
}

test "damaged pictures are refused or decoded, never a crash: cut short anywhere, bytes flipped" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(struct { cases: []Case }, gpa, @embedFile("fixtures/image/cases.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var prng: std.Random.DefaultPrng = .init(0x019_07);
    const random = prng.random();
    var tried: usize = 0;
    for (parsed.value.cases) |c| {
        const d = std.base64.standard.Decoder;
        const bytes = try gpa.alloc(u8, try d.calcSizeForSlice(c.bytes));
        defer gpa.free(bytes);
        try d.decode(bytes, c.bytes);
        if (bytes.len == 0) continue;
        errdefer std.debug.print("case {s}\n", .{c.name});
        // every length up to 512 bytes, then 256 lengths spread over the rest
        const step = @max(1, (bytes.len -| 512) / 256);
        var n: usize = 0;
        while (n < bytes.len) : (n += if (n < 512) 1 else step) {
            try survive(bytes[0..n]);
            tried += 1;
        }
        const damaged = try gpa.dupe(u8, bytes);
        defer gpa.free(damaged);
        for (0..64) |_| {
            @memcpy(damaged, bytes);
            for (0..1 + random.uintLessThan(usize, 4)) |_| {
                const at = random.uintLessThan(usize, bytes.len);
                damaged[at] = if (random.boolean()) random.int(u8) else damaged[at] ^ (@as(u8, 1) << random.int(u3));
            }
            try survive(damaged);
            tried += 1;
        }
    }
    try std.testing.expect(tried > 10_000);
}
