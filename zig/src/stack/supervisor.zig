//! A stack's processes kept together (spec 139): children started in waves, each wave waited on until every
//! child of it answers `GET /health`, then the next; each child's lines on the stack's stderr behind its name;
//! SIGINT, SIGTERM or SIGHUP to the stack stops them all, the last started first; a child that ends on its own
//! stops the rest, its exit named. As SGLang's `sglang_router.launch_server` keeps its workers and router: each
//! child in a process group of its own, so the terminal's Ctrl-C reaches the stack alone and the stack ends
//! them. A child on the other Spark runs under ssh and is stopped by closing its stdin, which its remote shell
//! waits on (`plan.zig`'s `remote`): an ssh that dies, the stack killed outright included, ends it the same way.
//! Linux only, as the Sparks are; host only, nothing of the device.

const std = @import("std");
const linux = std.os.linux;
const posix = std.posix;
const net = std.Io.net;
const Io = std.Io;
const Allocator = std.mem.Allocator;

pub const Child = struct {
    /// what its lines and its exit are named by: its checkpoint's name, or `router`
    name: []const u8,
    argv: []const []const u8,
    /// up once this address's `GET /health` answers 200; null: up once started (a rank 1, which serves nothing)
    health: ?net.IpAddress = null,
    /// on the other Spark, under ssh: stopped by closing its stdin, never by a signal to the ssh
    remote: bool = false,
    /// its environment in place of the stack's
    environ: ?*const std.process.Environ.Map = null,
};

/// Why the stack ended.
pub const Outcome = union(enum) {
    /// a signal to the stack: every child was stopped
    signal: u32,
    /// a child ended on its own (`term` its wait status), before its wave was up or after
    ended: struct { name: []const u8, term: Term },
    /// a child never answered within `health_timeout_ms`
    unanswered: []const u8,
    /// a child could not be started
    unstarted: []const u8,

    /// The stack's exit code: 0 for a stop asked by a signal, 1 otherwise.
    pub fn code(self: Outcome) u8 {
        return if (self == .signal) 0 else 1;
    }
};

pub const Term = union(enum) {
    exited: u8,
    signal: u32,

    pub fn of(status: u32) Term {
        if (linux.W.IFEXITED(status)) return .{ .exited = linux.W.EXITSTATUS(status) };
        return .{ .signal = @backingInt(linux.W.TERMSIG(status)) };
    }

    pub fn format(self: Term, w: *Io.Writer) Io.Writer.Error!void {
        switch (self) {
            .exited => |c| try w.print("exited with code {d}", .{c}),
            .signal => |s| try w.print("was killed by signal {d}", .{s}),
        }
    }
};

/// Before the first wave (an empty one) and after each is up: a line the stack prints of it (the memory left on each Spark), written into `buf`.
pub const AfterWave = struct {
    context: *anyopaque,
    call: *const fn (context: *anyopaque, wave: []const Child, buf: []u8) []const u8,
};

pub const Config = struct {
    io: Io,
    gpa: Allocator,
    waves: []const []const Child,
    /// where the children's lines and the stack's own go, each behind its name
    log: posix.fd_t = 2,
    /// a signal's number stored here stops the stack (`catchSignals` points the handler at it)
    stop: *std.atomic.Value(u32),
    poll_ms: u32 = 250,
    /// the longest a child may take to answer /health after it starts: a load
    health_timeout_ms: u64 = 30 * 60 * 1000,
    /// the longest the children have to end after they are told, before SIGKILL
    stop_timeout_ms: u64 = 60 * 1000,
    after_wave: ?AfterWave = null,
};

/// The signal the stack was sent, for `catchSignals`' handler; zero while none.
pub var signalled: std.atomic.Value(u32) = .init(0);

fn onSignal(sig: posix.SIG) callconv(.c) void {
    signalled.store(@backingInt(sig), .release);
}

/// SIGINT, SIGTERM and SIGHUP stored into `signalled` instead of ending the process; reads restart.
pub fn catchSignals() void {
    const action: posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = posix.sigemptyset(), .flags = posix.SA.RESTART };
    posix.sigaction(.INT, &action, null);
    posix.sigaction(.TERM, &action, null);
    posix.sigaction(.HUP, &action, null);
}

const Live = struct {
    child: *const Child,
    pid: linux.pid_t,
    stdin: ?posix.fd_t,
    reader: std.Thread,
    ended: ?Term = null,
    up: bool = false,
};

const Log = struct {
    fd: posix.fd_t,
    io: Io,
    mutex: Io.Mutex = .init,

    /// One line, behind `name`.
    fn line(self: *Log, name: []const u8, text: []const u8) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        writeAll(self.fd, "[");
        writeAll(self.fd, name);
        writeAll(self.fd, "] ");
        writeAll(self.fd, text);
        writeAll(self.fd, "\n");
    }

    fn print(self: *Log, comptime format: []const u8, args: anytype) void {
        var buf: [1024]u8 = undefined;
        self.line("stack", std.fmt.bufPrint(&buf, format, args) catch format);
    }
};

fn writeAll(fd: posix.fd_t, bytes: []const u8) void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = linux.write(fd, bytes[at..].ptr, bytes.len - at);
        if (linux.errno(rc) == .INTR) continue;
        if (linux.errno(rc) != .SUCCESS or rc == 0) return;
        at += rc;
    }
}

/// A child's pipe read to its end, each line written behind its name; a line past the buffer goes in parts.
fn relay(log: *Log, name: []const u8, fd: posix.fd_t) void {
    defer _ = linux.close(fd);
    var buf: [4096]u8 = undefined;
    var held: usize = 0;
    while (true) {
        if (held == buf.len) {
            log.line(name, buf[0..held]);
            held = 0;
        }
        const rc = linux.read(fd, buf[held..].ptr, buf.len - held);
        if (linux.errno(rc) == .INTR) continue;
        if (linux.errno(rc) != .SUCCESS or rc == 0) break;
        held += rc;
        var from: usize = 0;
        while (std.mem.indexOfScalarPos(u8, buf[0..held], from, '\n')) |end| {
            log.line(name, std.mem.trimEnd(u8, buf[from..end], "\r"));
            from = end + 1;
        }
        std.mem.copyForwards(u8, buf[0 .. held - from], buf[from..held]);
        held -= from;
    }
    if (held > 0) log.line(name, buf[0..held]);
}

/// The stack run until a signal or a child's end, every child stopped before it returns.
pub fn run(config: Config) Outcome {
    var log: Log = .{ .fd = config.log, .io = config.io };
    var lives: std.ArrayList(Live) = .empty;
    defer lives.deinit(config.gpa);
    const outcome = bring(config, &log, &lives);
    switch (outcome) {
        .signal => |sig| log.print("signal {d}: stopping every child", .{sig}),
        .ended => |e| log.print("{s} {f}: stopping the rest", .{ e.name, e.term }),
        .unanswered => |name| log.print("{s} did not answer /health in {d} s: stopping the rest", .{ name, config.health_timeout_ms / 1000 }),
        .unstarted => |name| log.print("{s} could not be started: stopping the rest", .{name}),
    }
    stopAll(config, &log, lives.items);
    for (lives.items) |*live| live.reader.join();
    log.print("stopped", .{});
    return outcome;
}

/// The waves started and waited on, then the stack watched; returns at the first signal or end.
fn bring(config: Config, log: *Log, lives: *std.ArrayList(Live)) Outcome {
    if (config.after_wave) |after| {
        var buf: [512]u8 = undefined;
        log.line("stack", after.call(after.context, &.{}, &buf)); // before any: what the first sizes from
    }
    for (config.waves) |wave| {
        const first = lives.items.len;
        for (wave) |*child| {
            const live = start(config, log, child) catch |err| {
                log.print("{s}: {t}", .{ child.name, err });
                return .{ .unstarted = child.name };
            };
            lives.append(config.gpa, live) catch return .{ .unstarted = child.name };
            log.print("{s} started (pid {d})", .{ child.name, live.pid });
        }
        const began = now();
        while (true) {
            if (watch(config, lives.items)) |outcome| return outcome;
            var all = true;
            for (lives.items[first..]) |*live| {
                if (live.up) continue;
                if (live.child.health) |address| live.up = healthy(address) else live.up = true;
                if (live.up) log.print("{s} is up after {d} s", .{ live.child.name, (now() - began) / 1000 }) else all = false;
            }
            if (all) break;
            for (lives.items[first..]) |live| {
                if (!live.up and now() - began > config.health_timeout_ms) return .{ .unanswered = live.child.name };
            }
            config.io.sleep(.fromMilliseconds(config.poll_ms), .awake) catch {};
        }
        if (config.after_wave) |after| {
            var buf: [512]u8 = undefined;
            log.line("stack", after.call(after.context, wave, &buf));
        }
    }
    log.print("all {d} up", .{lives.items.len});
    while (true) {
        if (watch(config, lives.items)) |outcome| return outcome;
        config.io.sleep(.fromMilliseconds(config.poll_ms), .awake) catch {};
    }
}

/// A signal, or the first child found ended; null while neither.
fn watch(config: Config, lives: []Live) ?Outcome {
    const sig = config.stop.load(.acquire);
    if (sig != 0) return .{ .signal = sig };
    reap(lives);
    for (lives) |live| if (live.ended) |term| return .{ .ended = .{ .name = live.child.name, .term = term } };
    return null;
}

/// Each child that has ended, its status taken.
fn reap(lives: []Live) void {
    for (lives) |*live| {
        if (live.ended != null) continue;
        var status: i32 = 0;
        const rc = linux.waitpid(live.pid, &status, linux.W.NOHANG);
        if (linux.errno(rc) == .SUCCESS and rc == @as(usize, @intCast(live.pid))) live.ended = .of(@bitCast(status));
        if (linux.errno(rc) == .CHILD) live.ended = .{ .exited = 255 }; // reaped elsewhere: never ours to see
    }
}

fn start(config: Config, log: *Log, child: *const Child) !Live {
    var pipe: [2]i32 = undefined;
    if (linux.errno(linux.pipe2(&pipe, .{ .CLOEXEC = true })) != .SUCCESS) return error.SystemResources;
    const output: Io.File = .{ .handle = pipe[1], .flags = .{ .nonblocking = false } };
    const spawned = std.process.spawn(config.io, .{
        .argv = child.argv,
        .environ_map = child.environ,
        .stdin = if (child.remote) .pipe else .ignore,
        .stdout = .{ .file = output },
        .stderr = .{ .file = output },
        // its own group: the terminal's Ctrl-C reaches the stack alone, which ends the children itself
        .pgid = 0,
    }) catch |err| {
        _ = linux.close(pipe[0]);
        _ = linux.close(pipe[1]);
        return err;
    };
    _ = linux.close(pipe[1]);
    const reader = std.Thread.spawn(.{}, relay, .{ log, child.name, pipe[0] }) catch |err| {
        _ = linux.close(pipe[0]);
        return err;
    };
    return .{
        .child = child,
        .pid = spawned.id.?,
        .stdin = if (spawned.stdin) |file| file.handle else null,
        .reader = reader,
    };
}

/// Every child told to end, the last started first, then waited on; what is left after the timeout killed.
fn stopAll(config: Config, log: *Log, lives: []Live) void {
    reap(lives);
    var i = lives.len;
    while (i > 0) {
        i -= 1;
        const live = &lives[i];
        if (live.ended != null) continue;
        if (live.child.remote) {
            // the remote shell's watcher reads its stdin's end and ends its child
            if (live.stdin) |fd| _ = linux.close(fd);
            live.stdin = null;
        } else {
            _ = linux.kill(-live.pid, .TERM);
        }
    }
    const began = now();
    while (now() - began < config.stop_timeout_ms) {
        reap(lives);
        var left = false;
        for (lives) |live| left = left or live.ended == null;
        if (!left) break;
        config.io.sleep(.fromMilliseconds(config.poll_ms), .awake) catch {};
    }
    for (lives) |*live| {
        if (live.stdin) |fd| _ = linux.close(fd);
        live.stdin = null;
        if (live.ended != null) continue;
        log.print("{s} did not end in {d} s: killed", .{ live.child.name, config.stop_timeout_ms / 1000 });
        _ = linux.kill(-live.pid, .KILL);
        var status: i32 = 0;
        _ = linux.waitpid(live.pid, &status, 0);
        live.ended = .of(@bitCast(status));
    }
}

/// Milliseconds of the monotonic clock.
fn now() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1000 + @as(u64, @intCast(ts.nsec)) / 1_000_000;
}

/// Whether `address` answers `GET /health` with 200 within a second. The router's `http.zig` has the like,
/// but no layer imports the router.
pub fn healthy(address: net.IpAddress) bool {
    const fd = connect(address, 1000) catch return false;
    defer _ = linux.close(fd);
    // the address itself as the host: a `serve` bound to the loopback refuses any other name (403)
    const ip = address.ip4;
    var request_buf: [128]u8 = undefined;
    const request = std.fmt.bufPrint(&request_buf, "GET /health HTTP/1.1\r\nhost: {d}.{d}.{d}.{d}:{d}\r\nconnection: close\r\n\r\n", .{ ip.bytes[0], ip.bytes[1], ip.bytes[2], ip.bytes[3], ip.port }) catch return false;
    writeAll(fd, request);
    var buf: [32]u8 = undefined;
    var held: usize = 0;
    while (held < 12) {
        var fds = [_]linux.pollfd{.{ .fd = fd, .events = linux.POLL.IN, .revents = 0 }};
        if (linux.poll(&fds, 1, 1000) != 1) return false;
        const rc = linux.read(fd, buf[held..].ptr, buf.len - held);
        if (linux.errno(rc) != .SUCCESS or rc == 0) return false;
        held += rc;
    }
    return std.mem.startsWith(u8, buf[0..held], "HTTP/1.") and std.mem.eql(u8, buf[8..12], " 200");
}

/// A socket connected to `address`, given up after `timeout_ms`.
fn connect(address: net.IpAddress, timeout_ms: i32) !posix.fd_t {
    const ip4 = switch (address) {
        .ip4 => |a| a,
        .ip6 => return error.Unsupported,
    };
    const rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC, 0);
    if (linux.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.fd_t = @intCast(rc);
    errdefer _ = linux.close(fd);
    var in: linux.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, ip4.port), .addr = @bitCast(ip4.bytes) };
    switch (linux.errno(linux.connect(fd, @ptrCast(&in), @sizeOf(linux.sockaddr.in)))) {
        .SUCCESS, .INPROGRESS => {},
        else => return error.ConnectFailed,
    }
    var fds = [_]linux.pollfd{.{ .fd = fd, .events = linux.POLL.OUT, .revents = 0 }};
    if (linux.poll(&fds, 1, timeout_ms) != 1) return error.ConnectTimedOut;
    var err: i32 = 0;
    var len: linux.socklen_t = @sizeOf(i32);
    _ = linux.getsockopt(fd, linux.SOL.SOCKET, linux.SO.ERROR, @ptrCast(&err), &len);
    if (err != 0) return error.ConnectFailed;
    return fd;
}
