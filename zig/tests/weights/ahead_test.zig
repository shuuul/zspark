//! `ahead.zig`'s tests, on the host: a source of the test's own, whose gather the test holds at a gate.

const std = @import("std");
const linux = std.os.linux;
const testing = std.testing;
const ahead_file = @import("../../src/weights/ahead.zig");

const fx = struct {
    const stride = 2;
    /// Ids from here have no row.
    const rows = 1000;

    /// Shut, a gather waits at it before it reads a row.
    const Gate = struct {
        open: std.atomic.Value(bool) = .init(true),
    };

    /// Row `id` is `[id, 3 * id]`, truncated.
    const Source = struct {
        gate: *Gate,

        pub fn stride(_: Source) usize {
            return fx.stride;
        }

        pub fn gather(self: Source, ids: []const u64, out: []u16) error{RowOutOfRange}!void {
            while (!self.gate.open.load(.acquire)) {
                const time: linux.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
                _ = linux.nanosleep(&time, null);
            }
            for (ids, 0..) |id, i| {
                if (id >= rows) return error.RowOutOfRange;
                out[i * fx.stride] = @truncate(id);
                out[i * fx.stride + 1] = @truncate(3 * id);
            }
        }
    };

    fn expectRows(ids: []const u64, got: []const u16) !void {
        try testing.expectEqual(ids.len * stride, got.len);
        for (ids, 0..) |id, i| {
            try testing.expectEqual(@as(u16, @truncate(id)), got[i * stride]);
            try testing.expectEqual(@as(u16, @truncate(3 * id)), got[i * stride + 1]);
        }
    }
};

const Ahead = ahead_file.Ahead(fx.Source);

test "ready hands over the rows already gathered without waiting, the rest left to the caller" {
    const gpa = testing.allocator;
    var gate: fx.Gate = .{};
    var ahead: Ahead = try .init(gpa, 8, fx.stride);
    defer ahead.deinit(gpa);
    const ids = [_]u64{ 11, 12, 13, 14, 15, 16 };

    // no thread, nothing asked: nothing ready
    try testing.expectEqual(0, ahead.ready(&ids).len);

    ahead.adopt(.{ .gate = &gate });
    ahead.ask(0, ids[0..3]);
    try fx.expectRows(ids[0..3], ahead.take(ids[0..3]));

    // the next rows asked while the gather is held: the first three are ready, the others not yet
    gate.open.store(false, .release);
    ahead.ask(3, ids[3..6]);
    try testing.expectEqual(6, ahead.held());
    try fx.expectRows(ids[0..3], ahead.ready(&ids));
    try fx.expectRows(ids[0..2], ahead.ready(ids[0..2]));
    // an id not yet gathered is not compared; one among those gathered is
    var other = ids;
    other[4] ^= 1;
    try fx.expectRows(ids[0..3], ahead.ready(&other));
    other = ids;
    other[1] ^= 1;
    try testing.expectEqual(0, ahead.ready(&other).len);

    gate.open.store(true, .release);
    try fx.expectRows(&ids, ahead.take(&ids));
    try fx.expectRows(&ids, ahead.ready(&ids));

    // what was held is gone with a new asking, though the thread has not yet answered it
    gate.open.store(false, .release);
    ahead.drop();
    ahead.ask(0, ids[0..2]);
    try testing.expectEqual(0, ahead.ready(&ids).len);
    gate.open.store(true, .release);
    try fx.expectRows(ids[0..2], ahead.take(&ids));

    // a row out of range: nothing is ready, the caller's gather says why
    const bad = [_]u64{ 21, fx.rows };
    ahead.drop();
    ahead.ask(0, &bad);
    try testing.expectEqual(0, ahead.take(&bad).len);
    try testing.expectEqual(0, ahead.ready(&bad).len);
}
