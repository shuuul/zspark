//! `drive.zig`'s tests, on the host: a file of the test's own in place of a checkpoint's.

const std = @import("std");
const linux = std.os.linux;
const testing = std.testing;
const Drive = @import("../../src/weights/drive.zig").Drive;

const fx = struct {
    const blocks = 64;
    /// A period and a linger short enough for a test: what is counted is reads, never a time.
    const quick: Drive.Options = .{ .period_ns = 2 * std.time.ns_per_ms, .linger_ns = 40 * std.time.ns_per_ms };

    /// A file of `blocks` blocks in `tmp` opened with O_DIRECT; the test is skipped where the directory's
    /// filesystem has no direct reads (a tmpfs).
    fn direct(tmp: *testing.TmpDir) !linux.fd_t {
        const bytes: [blocks * Drive.block]u8 = @splat(0x5a);
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "rows", .data = &bytes });
        const fd: isize = @bitCast(linux.openat(tmp.dir.handle, "rows", .{ .ACCMODE = .RDONLY, .DIRECT = true, .CLOEXEC = true }, 0));
        if (fd < 0) return error.SkipZigTest;
        return @intCast(fd);
    }

    fn sleep(ms: u64) void {
        const time: linux.timespec = .{ .sec = 0, .nsec = @intCast(ms * std.time.ns_per_ms) };
        _ = linux.nanosleep(&time, null);
    }
};

test "the drive is read while rows are asked for, and not before or after" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const fd = try fx.direct(&tmp);
    defer _ = linux.close(fd);
    const drive = Drive.open(testing.allocator, fd, fx.blocks * Drive.block, fx.quick) orelse return error.TestUnexpectedResult;
    defer drive.close(testing.allocator);

    // nothing asked for yet: no read
    fx.sleep(30);
    try testing.expectEqual(@as(u64, 0), drive.reads.load(.monotonic));

    // rows asked for every 5 ms for 100: a read a period, 50 or so; a loaded machine may give fewer
    for (0..20) |_| {
        drive.touch();
        fx.sleep(5);
    }
    const during = drive.reads.load(.monotonic);
    try testing.expect(during >= 10);

    // past the linger the reads end, and stay ended
    fx.sleep(150);
    const after = drive.reads.load(.monotonic);
    try testing.expect(after >= during);
    fx.sleep(100);
    try testing.expectEqual(after, drive.reads.load(.monotonic));

    // and begin again with the next row, from the reader's sleep
    drive.touch();
    fx.sleep(30);
    const again = drive.reads.load(.monotonic);
    try testing.expect(again > after);
    // one touch keeps it reading for the linger and no longer
    fx.sleep(150);
    const ended = drive.reads.load(.monotonic);
    fx.sleep(100);
    try testing.expectEqual(ended, drive.reads.load(.monotonic));
}

test "the drive's reader ends with its owner, asleep or reading" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const fd = try fx.direct(&tmp);
    defer _ = linux.close(fd);
    // never touched: closed in its first sleep
    const idle = Drive.open(testing.allocator, fd, fx.blocks * Drive.block, .{}) orelse return error.TestUnexpectedResult;
    idle.close(testing.allocator);
    // closed between two reads 30 s apart: the close does not wait a period out
    const busy = Drive.open(testing.allocator, fd, fx.blocks * Drive.block, .{ .period_ns = 30 * std.time.ns_per_s }) orelse return error.TestUnexpectedResult;
    busy.touch();
    while (busy.reads.load(.monotonic) == 0) fx.sleep(1);
    busy.close(testing.allocator);
}

test "no drive is kept awake without a direct descriptor or a whole block" {
    try testing.expectEqual(null, Drive.open(testing.allocator, -1, 1 << 20, .{}));
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const fd = try fx.direct(&tmp);
    defer _ = linux.close(fd);
    try testing.expectEqual(null, Drive.open(testing.allocator, fd, Drive.block - 1, .{}));
}

test "a descriptor that cannot be read ends the reader, and a touch after it is harmless" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    _ = linux.close(try fx.direct(&tmp));
    // open for writing alone: every read of it is refused
    const fd: isize = @bitCast(linux.openat(tmp.dir.handle, "rows", .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, 0));
    try testing.expect(fd >= 0);
    defer _ = linux.close(@intCast(fd));
    const drive = Drive.open(testing.allocator, @intCast(fd), fx.blocks * Drive.block, fx.quick) orelse return error.TestUnexpectedResult;
    defer drive.close(testing.allocator);
    for (0..5) |_| {
        drive.touch();
        fx.sleep(5);
    }
    try testing.expectEqual(@as(u64, 0), drive.reads.load(.monotonic));
}
