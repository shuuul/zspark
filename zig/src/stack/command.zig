//! `zspark stack` (spec 139): `plan.zig`'s children found their binaries and their link settings, then run by
//! `supervisor.zig` until a signal or a child's end. Between waves it says what memory each Spark has left,
//! the figure the next model sizes itself from. `link-env` is the NCCL settings' reading alone, which the
//! stack runs on the other Spark over ssh, so each side reads its own netdev and RoCE device.

const std = @import("std");
const plan_file = @import("plan.zig");
const supervisor = @import("supervisor.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub fn run(init: std.process.Init, stdout: *Io.Writer, words: []const []const u8) !u8 {
    const arena = init.arena.allocator();
    const io = init.io;
    if (words.len == 0 or std.mem.eql(u8, words[0], "-h") or std.mem.eql(u8, words[0], "--help")) {
        try stdout.writeAll(plan_file.usage);
        return if (words.len == 0) 2 else 0;
    }
    if (std.mem.eql(u8, words[0], "link-env")) {
        if (words.len != 2) {
            try stdout.writeAll(plan_file.usage);
            return 2;
        }
        const env = linkEnv(arena, io, words[1]) catch |err| {
            std.log.err("link-env {s}: {t}", .{ words[1], err });
            return 1;
        };
        for (env, 0..) |word, i| try stdout.print("{s}{s}", .{ if (i > 0) " " else "", word });
        try stdout.writeAll("\n");
        return 0;
    }
    var said: []const u8 = "";
    const options = plan_file.parse(words, &said) orelse {
        std.log.err("stack: {s}", .{said});
        try stdout.writeAll(plan_file.usage);
        return 2;
    };

    const zspark = try std.process.executablePathAlloc(io, arena);
    const bin_dir = std.fs.path.dirname(zspark) orelse ".";
    var own = try init.minimal.environ.createMap(arena);
    var bin: plan_file.Binaries = .{
        .environ = &own,
        .zspark = zspark,
        .router = try std.fs.path.join(arena, &.{ bin_dir, "zspark-router" }),
    };
    std.Io.Dir.cwd().access(io, bin.router, .{}) catch {
        std.log.err("stack: no {s} beside zspark: zig build", .{bin.router});
        return 2;
    };
    if (options.peer) |peer| {
        if (options.asr) |dir| {
            bin.asr_split = asrSplit(arena, io, dir, peer) catch |err| {
                std.log.err("stack: ASR config at {s}: {t}", .{ dir, err });
                return 1;
            };
        }
        // each side's NCCL settings, read on its own Spark: here directly, there by this command over ssh
        const here = linkEnv(arena, io, peer) catch |err| {
            std.log.err("stack: this Spark's link to {s}: {t}", .{ peer, err });
            return 1;
        };
        const mine = sourceAddress(arena, io, peer) catch |err| {
            std.log.err("stack: no route to {s}: {t}", .{ peer, err });
            return 1;
        };
        const peer_zspark = options.peer_zspark orelse zspark;
        const there = runText(arena, io, &.{ "ssh", "-o", "BatchMode=yes", peer, try std.fmt.allocPrint(arena, "{s} stack link-env {s}", .{ peer_zspark, mine }) }) catch |err| {
            std.log.err("stack: the link's settings on {s} ({s} stack link-env): {t}", .{ peer, peer_zspark, err });
            return 1;
        };
        var environ = try init.minimal.environ.createMap(arena);
        for (here) |word| {
            const eq = std.mem.indexOfScalar(u8, word, '=').?;
            try environ.put(word[0..eq], word[eq + 1 ..]);
        }
        bin.environ_here = &environ;
        var theirs: std.ArrayList([]const u8) = .empty;
        var split = std.mem.tokenizeAny(u8, there, " \n");
        while (split.next()) |word| try theirs.append(arena, word);
        for (plan_file.shared_variables) |name| {
            if (init.minimal.environ.getPosix(name)) |value| try theirs.append(arena, try std.fmt.allocPrint(arena, "{s}={s}", .{ name, value }));
        }
        bin.environ_there = theirs.items;
        std.log.info("stack: rank 0's link settings: {s}", .{try std.mem.join(arena, " ", here)});
        std.log.info("stack: rank 1's on {s}: {s}", .{ peer, try std.mem.join(arena, " ", theirs.items) });
    }
    const plan = try plan_file.build(arena, options, bin);
    if (plan.chosen_parallel) |n| {
        const beside = if (plan.ocr_name != null and plan.asr_name != null) "OCR and ASR" else if (plan.ocr_name != null) "OCR" else "ASR";
        if (options.peer != null)
            std.log.info("stack: {s} at --parallel {d} on both ranks, not two ranks' serve's 1: the stack serves several clients; -- --parallel N chooses", .{ plan.flash_name, n })
        else
            std.log.info("stack: {s} at --parallel {d} of its full context beside {s} on one Spark: serve's 4 does not fit beside them, or leaves its prompt cache nothing (spec 139 WS-03); -- --parallel N chooses", .{ plan.flash_name, n, beside });
    }
    if (plan.mps_dir) |dir| {
        // before any child: a child that comes up before its daemon listens runs time-sliced, with no error
        const line = try plan_file.mpsLine(arena, dir, true);
        if (plan.mps_here) _ = runText(arena, io, &.{ "sh", "-c", line }) catch {
            std.log.err("stack: no MPS daemon here ({s}); --mps 0 runs without one", .{dir});
            return 1;
        };
        if (plan.mps_there) _ = runText(arena, io, &.{ "ssh", "-o", "BatchMode=yes", options.peer.?, line }) catch {
            std.log.err("stack: no MPS daemon on {s} ({s}); --mps 0 runs without one", .{ options.peer.?, dir });
            if (plan.mps_here) mpsQuit(arena, io, dir, null);
            return 1;
        };
        std.log.info("stack: MPS ({s}{s}{s}): the small models at {d}% of the SMs, {s} at all of them; a fault in one client ends its Spark's others", .{
            dir,
            if (plan.mps_here) " here" else "",
            if (plan.mps_there) " and there" else "",
            options.mps,
            plan.flash_name,
        });
    }
    defer if (plan.mps_dir) |dir| {
        if (plan.mps_here) mpsQuit(arena, io, dir, null);
        if (plan.mps_there) mpsQuit(arena, io, dir, options.peer.?);
    };
    var memory: Memory = .{ .arena = arena, .io = io, .peer = options.peer };
    supervisor.signalled.store(0, .release);
    supervisor.catchSignals();
    const outcome = supervisor.run(.{
        .io = io,
        .gpa = init.gpa,
        .waves = plan.waves,
        .stop = &supervisor.signalled,
        .health_timeout_ms = options.health_s * 1000,
        .stop_timeout_ms = options.stop_s * 1000,
        .after_wave = .{ .context = &memory, .call = Memory.say },
    });
    return outcome.code();
}

/// Nemotron's ASR spans both Sparks; a single-rank ASR may exist only on the peer.
fn asrSplit(arena: Allocator, io: Io, dir: []const u8, peer: []const u8) !bool {
    const path = try std.fs.path.join(arena, &.{ dir, "config.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 24)) catch |err| switch (err) {
        error.FileNotFound => remote: {
            const command = try std.fmt.allocPrint(arena, "cat -- {s}", .{try plan_file.quote(arena, path)});
            const result = try std.process.run(arena, io, .{ .argv = &.{ "ssh", "-o", "BatchMode=yes", peer, command }, .stdout_limit = .limited(1 << 24), .stderr_limit = .limited(1 << 16) });
            if (!result.term.success()) {
                std.log.err("stack: reading ASR config on {s}: {s}", .{ peer, result.stderr });
                return error.CommandFailed;
            }
            break :remote result.stdout;
        },
        else => return err,
    };
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
    if (root != .object) return error.InvalidConfig;
    const kind = root.object.get("model_type") orelse return error.InvalidConfig;
    if (kind != .string) return error.InvalidConfig;
    return std.mem.eql(u8, kind.string, "nemo_speechlm");
}

/// What memory each Spark has left after a wave: `MemAvailable`, the device's share on GB10's one memory.
const Memory = struct {
    arena: Allocator,
    io: Io,
    peer: ?[]const u8,

    fn say(context: *anyopaque, wave: []const supervisor.Child, buf: []u8) []const u8 {
        const self: *Memory = @ptrCast(@alignCast(context));
        const read = [_][]const u8{ "grep", "MemAvailable", "/proc/meminfo" };
        const here = gib(runText(self.arena, self.io, &read) catch "");
        var there: f64 = 0;
        if (self.peer) |peer| there = gib(runText(self.arena, self.io, &.{ "ssh", "-o", "BatchMode=yes", peer, "grep MemAvailable /proc/meminfo" }) catch "");
        var names: std.ArrayList(u8) = .empty;
        if (wave.len == 0) names.appendSlice(self.arena, "nothing started") catch {};
        for (wave, 0..) |child, i| {
            if (i > 0) names.appendSlice(self.arena, ", ") catch {};
            names.appendSlice(self.arena, child.name) catch {};
        }
        if (self.peer) |peer| {
            return std.fmt.bufPrint(buf, "after {s}: {d:.1} GiB available here, {d:.1} GiB on {s}", .{ names.items, here, there, peer }) catch "";
        }
        return std.fmt.bufPrint(buf, "after {s}: {d:.1} GiB available", .{ names.items, here }) catch "";
    }

    /// `MemAvailable:   N kB` in GiB; 0 where the line is not there.
    fn gib(line: []const u8) f64 {
        var words = std.mem.tokenizeAny(u8, line, " \t\n");
        _ = words.next();
        const kib = std.fmt.parseInt(u64, words.next() orelse return 0, 10) catch return 0;
        return @as(f64, @floatFromInt(kib)) / (1024 * 1024);
    }
};

/// The stack's MPS daemon on this Spark (`peer` null) or the other, quit once its children have ended.
fn mpsQuit(arena: Allocator, io: Io, dir: []const u8, peer: ?[]const u8) void {
    const line = plan_file.mpsLine(arena, dir, false) catch return;
    _ = if (peer) |p| runText(arena, io, &.{ "ssh", "-o", "BatchMode=yes", p, line }) catch return else runText(arena, io, &.{ "sh", "-c", line }) catch return;
}

/// A command's stdout, which must end with status 0.
fn runText(arena: Allocator, io: Io, argv: []const []const u8) ![]const u8 {
    const result = try std.process.run(arena, io, .{ .argv = argv, .stdout_limit = .limited(1 << 16), .stderr_limit = .limited(1 << 16) });
    if (!result.term.success()) {
        std.log.err("{s}: {f}: {s}", .{ argv[0], result.term, std.mem.trim(u8, result.stderr, " \n") });
        return error.CommandFailed;
    }
    return result.stdout;
}

/// The route to `peer`: its netdev and this Spark's source address on it.
const Route = struct { dev: []const u8, src: []const u8 };

fn route(arena: Allocator, io: Io, peer: []const u8) !Route {
    const text = try runText(arena, io, &.{ "ip", "-o", "route", "get", peer });
    return .{ .dev = wordAfter(text, "dev") orelse return error.NoRoute, .src = wordAfter(text, "src") orelse return error.NoRoute };
}

fn sourceAddress(arena: Allocator, io: Io, peer: []const u8) ![]const u8 {
    return (try route(arena, io, peer)).src;
}

/// The word after `key` among `text`'s words.
pub fn wordAfter(text: []const u8, key: []const u8) ?[]const u8 {
    var words = std.mem.tokenizeAny(u8, text, " \t\n");
    while (words.next()) |word| {
        if (std.mem.eql(u8, word, key)) return words.next();
    }
    return null;
}

/// NCCL's settings for the link to `peer`, as tools/two_ranks.sh's `link_env` reads them on its machine: the
/// route's netdev, its RoCE device, the index of that device's RoCE v2 GID of the netdev's IPv4 address, every
/// other RoCE device with an IPv4 RoCE v2 GID at that index (the second rail), and 4 channels.
pub fn linkEnv(arena: Allocator, io: Io, peer: []const u8) ![]const []const u8 {
    const found = try route(arena, io, peer);
    const cwd = std.Io.Dir.cwd();
    const hca = blk: {
        var dir = cwd.openDir(io, try std.fmt.allocPrint(arena, "/sys/class/net/{s}/device/infiniband", .{found.dev}), .{ .iterate = true }) catch return error.NoRoceDevice;
        defer dir.close(io);
        var names: std.ArrayList([]const u8) = .empty;
        var it = dir.iterate();
        while (try it.next(io)) |entry| try names.append(arena, try arena.dupe(u8, entry.name));
        if (names.items.len == 0) return error.NoRoceDevice;
        std.mem.sort([]const u8, names.items, {}, lessThan);
        break :blk names.items[0];
    };
    var octets: [4]u8 = undefined;
    var parts = std.mem.splitScalar(u8, found.src, '.');
    for (&octets) |*o| o.* = std.fmt.parseInt(u8, parts.next() orelse return error.NoRoute, 10) catch return error.NoRoute;
    const gid_text = try std.fmt.allocPrint(arena, "0000:0000:0000:0000:0000:ffff:{x:0>2}{x:0>2}:{x:0>2}{x:0>2}", .{ octets[0], octets[1], octets[2], octets[3] });
    const gid = blk: {
        const types = try std.fmt.allocPrint(arena, "/sys/class/infiniband/{s}/ports/1/gid_attrs/types", .{hca});
        var dir = cwd.openDir(io, types, .{ .iterate = true }) catch return error.NoGid;
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (!std.mem.eql(u8, sysText(arena, io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ types, entry.name })), "RoCE v2")) continue;
            const value = sysText(arena, io, try std.fmt.allocPrint(arena, "/sys/class/infiniband/{s}/ports/1/gids/{s}", .{ hca, entry.name }));
            if (std.mem.eql(u8, value, gid_text)) break :blk try arena.dupe(u8, entry.name);
        }
        return error.NoGid;
    };
    var hcas: std.ArrayList(u8) = .empty;
    try hcas.appendSlice(arena, hca);
    {
        var dir = try cwd.openDir(io, "/sys/class/infiniband", .{ .iterate = true });
        defer dir.close(io);
        var names: std.ArrayList([]const u8) = .empty;
        var it = dir.iterate();
        while (try it.next(io)) |entry| try names.append(arena, try arena.dupe(u8, entry.name));
        std.mem.sort([]const u8, names.items, {}, lessThan);
        for (names.items) |name| {
            if (std.mem.eql(u8, name, hca)) continue;
            if (!std.mem.eql(u8, sysText(arena, io, try std.fmt.allocPrint(arena, "/sys/class/infiniband/{s}/ports/1/gid_attrs/types/{s}", .{ name, gid })), "RoCE v2")) continue;
            const value = sysText(arena, io, try std.fmt.allocPrint(arena, "/sys/class/infiniband/{s}/ports/1/gids/{s}", .{ name, gid }));
            if (!std.mem.startsWith(u8, value, "0000:0000:0000:0000:0000:ffff:")) continue;
            try hcas.append(arena, ',');
            try hcas.appendSlice(arena, name);
        }
    }
    return try arena.dupe([]const u8, &.{
        try std.fmt.allocPrint(arena, "NCCL_SOCKET_IFNAME={s}", .{found.dev}),
        try std.fmt.allocPrint(arena, "NCCL_IB_HCA={s}", .{hcas.items}),
        try std.fmt.allocPrint(arena, "NCCL_IB_GID_INDEX={s}", .{gid}),
        "NCCL_MIN_NCHANNELS=4",
        "NCCL_MAX_NCHANNELS=4",
    });
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// A sysfs file's text, its newline cut; empty where it cannot be read. Read as a stream: sysfs says size 4096.
fn sysText(arena: Allocator, io: Io, path: []const u8) []const u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return "";
    defer file.close(io);
    const buf = arena.alloc(u8, 256) catch return "";
    var reader = file.readerStreaming(io, &.{});
    const len = reader.interface.readSliceShort(buf) catch return "";
    return std.mem.trimEnd(u8, buf[0..len], "\n");
}
