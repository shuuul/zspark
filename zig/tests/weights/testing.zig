//! Fixtures two or more of `zig/src/weights/`'s test files share, a struct a fixture with its imports inside.
//! `linears`: a linear on the device held against its dense weight (`checkLinear`), for the tests of `nvfp4.zig`,
//! `bf16.zig` and `face.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

pub const linears = struct {
    const cuda = @import("../../src/cuda/cuda.zig");
    const numeric = @import("../../src/weights/numeric.zig");

    pub const Path = union(enum) { decode, prompt: u3 };

    /// `linear` against `dense`, its weight [n, k] as the host dequantizes it: each row count matches the fp64 product,
    /// and the first row keeps its bits whatever rows share its launch.
    pub fn checkLinear(gpa: Allocator, linear: anytype, dense: []const f64, rows: []const usize, path: Path) !void {
        const n = linear.n;
        const k = linear.k;
        var prng: std.Random.DefaultPrng = .init(n * 131 + k);
        const random = prng.random();

        const max_rows = std.mem.max(usize, rows);
        const x_bits = try gpa.alloc(u16, max_rows * k);
        defer gpa.free(x_bits);
        for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
        const first = try gpa.alloc(f32, n);
        defer gpa.free(first);
        const got = try gpa.alloc(f32, max_rows * n);
        defer gpa.free(got);

        for (rows, 0..) |m, round| {
            var x: cuda.Buffer = try .init(.bf16, &.{ m, k });
            defer x.deinit();
            try x.upload(std.mem.sliceAsBytes(x_bits[0 .. m * k]));
            var out: cuda.Buffer = try .init(.f32, &.{ m, n });
            defer out.deinit();
            switch (path) {
                .decode => try linear.forward(x.view, out.view),
                .prompt => |tile| try linear.prefill(x.view, out.view, tile),
            }
            try cuda.synchronize();
            try out.download(std.mem.sliceAsBytes(got[0 .. m * n]));

            for (0..m) |row| for (0..n) |col| {
                var want: f64 = 0;
                var magnitude: f64 = 0;
                for (0..k) |i| {
                    const term = @as(f64, numeric.f32FromBf16(x_bits[row * k + i])) * dense[col * k + i];
                    want += term;
                    magnitude += @abs(term);
                }
                try std.testing.expectApproxEqAbs(want, @as(f64, got[row * n + col]), magnitude * 1e-5);
            };
            if (round == 0) @memcpy(first, got[0..n]) else try std.testing.expectEqualSlices(u32, @ptrCast(first), @ptrCast(got[0..n]));
        }
    }
};
