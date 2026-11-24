//! `cuda.zig`'s tests.

const std = @import("std");
const cuda = @import("../../src/cuda/cuda.zig");
const Buffer = cuda.Buffer;
const Scratch = cuda.Scratch;
const probe = cuda.probe;
/// The elements the probe's kernel sums (`zig/kernels/zs/probe.cu`).
const probe_n = 4096;

test "a pool lends one part buffer to every shape that fits in it, and never across two roles" {
    var pool: Scratch = .init(std.testing.allocator);
    defer pool.deinit();
    const part = try pool.lend(.part, .f32, &.{ 2, 8, 64 });
    const again = try pool.lend(.part, .f32, &.{ 2, 8, 64 });
    try std.testing.expectEqual(part.view.ptr, again.view.ptr);
    // the same shape in another role is another buffer
    const staging = try pool.lend(.staging, .f32, &.{ 2, 8, 64 });
    try std.testing.expect(staging.view.ptr != part.view.ptr);
    // the same bytes in another shape, and a smaller shape, are that one part buffer
    const other = try pool.lend(.part, .f32, &.{ 1, 16, 64 });
    try std.testing.expectEqual(part.view.ptr, other.view.ptr);
    const smaller = try pool.lend(.part, .f32, &.{ 1, 8, 64 });
    try std.testing.expectEqual(part.view.ptr, smaller.view.ptr);
    // a larger shape does not fit, and the next request takes the smallest buffer that does
    const larger = try pool.lend(.part, .f32, &.{ 4, 8, 64 });
    try std.testing.expect(larger.view.ptr != part.view.ptr);
    const mid = try pool.lend(.part, .f32, &.{ 3, 8, 64 });
    try std.testing.expectEqual(larger.view.ptr, mid.view.ptr);
    const wider = try pool.lend(.staging, .f32, &.{ 4, 8, 64 });
    try std.testing.expect(wider.view.ptr != staging.view.ptr and wider.view.ptr != larger.view.ptr);
    try std.testing.expectEqual(part.bytes() + staging.bytes() + larger.bytes() + wider.bytes(), pool.bytes());

    Scratch.shared = false;
    defer Scratch.shared = true;
    const own = try pool.lend(.part, .f32, &.{ 2, 8, 64 });
    try std.testing.expect(own.view.ptr != part.view.ptr);
}

test "probe launches a kernel" {
    const info = try probe();
    try std.testing.expectEqual(@as(u64, probe_n * (probe_n - 1) / 2), info.kernel_sum);
}

test "buffer round trip" {
    var buffer: Buffer = try .init(.f32, &.{ 3, 5 });
    defer buffer.deinit();
    var values: [15]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @floatFromInt(i);
    try buffer.upload(std.mem.sliceAsBytes(&values));
    var back: [15]f32 = @splat(0);
    try buffer.download(std.mem.sliceAsBytes(&back));
    try std.testing.expectEqualSlices(f32, &values, &back);
}

test "a download goes through the page-locked staging whole: from an offset, across pieces of the staging's size, and nothing past its end" {
    // spec 132: a copy from the device straight into pageable memory sometimes missed it on the Spark; every
    // synchronous download now lands in `cuda.staging` first. Its pieces and offsets are what this holds; the
    // miss itself showed in the rank rounds' test, 10 runs of 60 before and none after.
    const gpa = std.testing.allocator;
    const n = 2 * cuda.staging.most + 4093;
    var source: Buffer = try .init(.u8, &.{n});
    defer source.deinit();
    const want = try gpa.alloc(u8, n);
    defer gpa.free(want);
    for (want, 0..) |*byte, i| byte.* = @truncate(i *% 131 +% 7 | 1);
    try source.upload(want);
    const got = try gpa.alloc(u8, n + 64);
    defer gpa.free(got);
    for ([_][2]usize{ .{ 0, n }, .{ 3, 1 }, .{ 0, 0 }, .{ 4093, cuda.staging.most + 1 }, .{ 17, n - 17 } }) |case| {
        const offset, const len = case;
        @memset(got, 0xaa);
        try source.downloadAt(offset, got[0..len]);
        try std.testing.expectEqualSlices(u8, want[offset..][0..len], got[0..len]);
        try std.testing.expect(std.mem.indexOfNone(u8, got[len..], &.{0xaa}) == null);
    }
    @memset(got, 0xaa);
    try source.download(got[0..n]);
    try std.testing.expectEqualSlices(u8, want, got[0..n]);
    try cuda.download(source.view, got[0..n]);
    try std.testing.expectEqualSlices(u8, want, got[0..n]);
}

test "what the device can still give is the system's estimate where it has one, and no less than what the device says is free" {
    const free = (try cuda.memory()).free;
    // Linux has the estimate, and a read that gave nothing would leave the device's count alone
    const system = cuda.systemAvailable(std.testing.io).?;
    try std.testing.expect(system > 1 << 30);
    const left = cuda.available(std.testing.io);
    // the page cache is given back on demand and the device's own count leaves it out
    try std.testing.expect(left >= free -| (64 << 20) and left >= system -| (256 << 20));
}

test "the quarantine holds freed buffers back and names one written through a stale pointer" {
    const quarantine = cuda.quarantine;
    quarantine.on = true;
    defer {
        quarantine.on = false;
        quarantine.found = null;
    }
    const before = quarantine.checked;
    // a buffer freed and left alone reads back as it was filled
    var clean: Buffer = try .init(.f32, &.{ 4, 64 });
    clean.deinit();
    try quarantine.drain();
    try std.testing.expectEqual(before + 1, quarantine.checked);
    // a copy of the view kept past the free, written through: the bytes the device now gives to someone else
    var buffer: Buffer = try .init(.bf16, &.{ 9, 640 });
    const stale = buffer;
    buffer.deinit();
    try stale.zero();
    try std.testing.expectError(error.Overwritten, quarantine.drain());
    try std.testing.expectEqual(before + 2, quarantine.checked);
    // the finding names the buffer and the byte `zero` left, for the panic that meets the error to print
    const found = quarantine.found.?;
    try std.testing.expectEqual(stale.view.ptr, found.held.view.ptr);
    try std.testing.expectEqual(cuda.DType.bf16, found.held.view.dtype);
    try std.testing.expectEqualSlices(i64, &.{ 9, 640 }, found.held.view.size[0..@intCast(found.held.view.ndim)]);
    try std.testing.expectEqual(@as(usize, 11520), found.held.bytes);
    try std.testing.expectEqual(@as(usize, 0), found.at);
    try std.testing.expectEqual(@as(u8, 0), found.byte);
}
