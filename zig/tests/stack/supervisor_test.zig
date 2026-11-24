//! The supervisor with fake children: shell commands for `serve`, an in-test listener for a child's /health.

const std = @import("std");
const testing = std.testing;
const linux = std.os.linux;
const supervisor = @import("../../src/stack/supervisor.zig");
const plan_file = @import("../../src/stack/plan.zig");
const Child = supervisor.Child;

/// A loopback listener that answers every request 200, as a loaded `serve`'s /health does.
const Health = struct {
    fd: i32,
    port: u16,
    stop: std.atomic.Value(bool) = .init(false),
    thread: std.Thread = undefined,

    fn start(self: *Health) !void {
        const rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
        try testing.expectEqual(linux.E.SUCCESS, linux.errno(rc));
        self.fd = @intCast(rc);
        var in: linux.sockaddr.in = .{ .port = 0, .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
        try testing.expectEqual(linux.E.SUCCESS, linux.errno(linux.bind(self.fd, @ptrCast(&in), @sizeOf(linux.sockaddr.in))));
        try testing.expectEqual(linux.E.SUCCESS, linux.errno(linux.listen(self.fd, 16)));
        var len: linux.socklen_t = @sizeOf(linux.sockaddr.in);
        _ = linux.getsockname(self.fd, @ptrCast(&in), &len);
        self.port = std.mem.bigToNative(u16, in.port);
        self.thread = try std.Thread.spawn(.{}, serve, .{self});
    }

    fn serve(self: *Health) void {
        while (!self.stop.load(.acquire)) {
            var fds = [_]linux.pollfd{.{ .fd = self.fd, .events = linux.POLL.IN, .revents = 0 }};
            if (linux.poll(&fds, 1, 50) != 1) continue;
            const rc = linux.accept(self.fd, null, null);
            if (linux.errno(rc) != .SUCCESS) continue;
            const conn: i32 = @intCast(rc);
            var buf: [512]u8 = undefined;
            _ = linux.read(conn, &buf, buf.len);
            const reply = "HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";
            _ = linux.write(conn, reply, reply.len);
            _ = linux.close(conn);
        }
    }

    fn end(self: *Health) void {
        self.stop.store(true, .release);
        self.thread.join();
        _ = linux.close(self.fd);
    }

    fn address(self: *Health) std.Io.net.IpAddress {
        return .{ .ip4 = .loopback(self.port) };
    }
};

/// A run's log, written to a file of the test's directory and read back after.
const Logged = struct {
    tmp: testing.TmpDir,
    file: std.Io.File,

    fn open() !Logged {
        var tmp = testing.tmpDir(.{});
        const file = try tmp.dir.createFile(testing.io, "log", .{ .read = true });
        return .{ .tmp = tmp, .file = file };
    }

    fn text(self: *Logged) ![]const u8 {
        return try self.tmp.dir.readFileAlloc(testing.io, "log", testing.allocator, .limited(1 << 20));
    }

    fn close(self: *Logged) void {
        self.file.close(testing.io);
        self.tmp.cleanup();
    }
};

fn sh(script: []const u8) []const []const u8 {
    return testing.allocator.dupe([]const u8, &.{ "sh", "-c", script }) catch @panic("oom");
}

fn expectBefore(log: []const u8, first: []const u8, second: []const u8) !void {
    const a = std.mem.indexOf(u8, log, first) orelse return error.TestExpectedFirst;
    const b = std.mem.indexOf(u8, log, second) orelse return error.TestExpectedSecond;
    try testing.expect(a < b);
}

test "waves wait on health, lines carry their child's name, and a child's end stops the rest naming its exit" {
    var health: Health = .{ .fd = 0, .port = 0 };
    try health.start();
    defer health.end();
    var logged = try Logged.open();
    defer logged.close();
    const a_argv = sh("echo loaded; echo second line >&2; exec sleep 30");
    defer testing.allocator.free(a_argv);
    const b_argv = sh("echo b here; sleep 0.5; exit 3");
    defer testing.allocator.free(b_argv);
    const first = [_]Child{.{ .name = "a", .argv = a_argv, .health = health.address() }};
    const second = [_]Child{.{ .name = "b", .argv = b_argv }};
    var stop: std.atomic.Value(u32) = .init(0);
    const outcome = supervisor.run(.{
        .io = testing.io,
        .gpa = testing.allocator,
        .waves = &.{ &first, &second },
        .log = logged.file.handle,
        .stop = &stop,
        .poll_ms = 20,
        .stop_timeout_ms = 5000,
    });
    try testing.expectEqualStrings("b", outcome.ended.name);
    try testing.expectEqual(supervisor.Term{ .exited = 3 }, outcome.ended.term);
    try testing.expectEqual(@as(u8, 1), outcome.code());
    const log = try logged.text();
    defer testing.allocator.free(log);
    try testing.expect(std.mem.indexOf(u8, log, "[a] loaded\n") != null);
    try testing.expect(std.mem.indexOf(u8, log, "[a] second line\n") != null);
    try testing.expect(std.mem.indexOf(u8, log, "[b] b here\n") != null);
    try expectBefore(log, "[stack] a is up", "[stack] b started");
    try testing.expect(std.mem.indexOf(u8, log, "[stack] b exited with code 3: stopping the rest") != null);
    try testing.expect(std.mem.indexOf(u8, log, "did not end") == null); // `a` ended on its SIGTERM
}

test "a signal stops every child, and a child that ignores SIGTERM is killed after the timeout" {
    var logged = try Logged.open();
    defer logged.close();
    const plain = sh("exec sleep 30");
    defer testing.allocator.free(plain);
    const stubborn = sh("trap '' TERM; echo stubborn; sleep 30");
    defer testing.allocator.free(stubborn);
    const wave = [_]Child{ .{ .name = "plain", .argv = plain }, .{ .name = "stubborn", .argv = stubborn } };
    var stop: std.atomic.Value(u32) = .init(0);
    const Signaller = struct {
        fn later(flag: *std.atomic.Value(u32)) void {
            testing.io.sleep(.fromMilliseconds(300), .awake) catch {};
            flag.store(@backingInt(std.posix.SIG.INT), .release);
        }
    };
    const thread = try std.Thread.spawn(.{}, Signaller.later, .{&stop});
    const outcome = supervisor.run(.{
        .io = testing.io,
        .gpa = testing.allocator,
        .waves = &.{&wave},
        .log = logged.file.handle,
        .stop = &stop,
        .poll_ms = 20,
        .stop_timeout_ms = 400,
    });
    thread.join();
    try testing.expectEqual(@as(u32, @backingInt(std.posix.SIG.INT)), outcome.signal);
    try testing.expectEqual(@as(u8, 0), outcome.code());
    const log = try logged.text();
    defer testing.allocator.free(log);
    try testing.expect(std.mem.indexOf(u8, log, "[stack] all 2 up") != null);
    try testing.expect(std.mem.indexOf(u8, log, "[stack] stubborn did not end in 0 s: killed") != null);
    try testing.expect(std.mem.indexOf(u8, log, "[stack] plain did not end") == null);
}

test "a child that never answers /health stops the stack" {
    var logged = try Logged.open();
    defer logged.close();
    const argv = sh("exec sleep 30");
    defer testing.allocator.free(argv);
    // a port nothing listens on: the probe's connect is refused
    var health: Health = .{ .fd = 0, .port = 0 };
    try health.start();
    const port = health.port;
    health.end();
    const wave = [_]Child{.{ .name = "silent", .argv = argv, .health = .{ .ip4 = .loopback(port) } }};
    var stop: std.atomic.Value(u32) = .init(0);
    const outcome = supervisor.run(.{
        .io = testing.io,
        .gpa = testing.allocator,
        .waves = &.{&wave},
        .log = logged.file.handle,
        .stop = &stop,
        .poll_ms = 20,
        .health_timeout_ms = 300,
    });
    try testing.expectEqualStrings("silent", outcome.unanswered);
}

test "a child on the other Spark's shell ends when its stdin ends, its status handed back" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // the line ssh would run there, run here by `sh -c` in its place
    const ssh = try plan_file.remote(arena, "198.51.100.2", &.{"GREETING=it's here"}, &.{ "sh", "-c", "trap 'echo told; exit 7' TERM; echo $GREETING; while :; do sleep 0.05; done" });
    try testing.expectEqualStrings("ssh", ssh[0]);
    try testing.expectEqualStrings("198.51.100.2", ssh[3]);
    const argv = [_][]const u8{ "sh", "-c", ssh[4] };
    var logged = try Logged.open();
    defer logged.close();
    const wave = [_]Child{.{ .name = "far", .argv = &argv, .remote = true }};
    var stop: std.atomic.Value(u32) = .init(0);
    const Signaller = struct {
        fn later(flag: *std.atomic.Value(u32)) void {
            testing.io.sleep(.fromMilliseconds(400), .awake) catch {};
            flag.store(@backingInt(std.posix.SIG.TERM), .release);
        }
    };
    const thread = try std.Thread.spawn(.{}, Signaller.later, .{&stop});
    const outcome = supervisor.run(.{
        .io = testing.io,
        .gpa = testing.allocator,
        .waves = &.{&wave},
        .log = logged.file.handle,
        .stop = &stop,
        .poll_ms = 20,
        .stop_timeout_ms = 5000,
    });
    thread.join();
    try testing.expect(outcome == .signal);
    const log = try logged.text();
    defer testing.allocator.free(log);
    try testing.expect(std.mem.indexOf(u8, log, "[far] it's here\n") != null);
    try testing.expect(std.mem.indexOf(u8, log, "[far] told\n") != null);
    try testing.expect(std.mem.indexOf(u8, log, "did not end") == null);
}
