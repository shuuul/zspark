//! `command.zig`'s tests.

const std = @import("std");
const host = @import("server");
const command = @import("../../src/server/command.zig");
const fixture = @import("served_fixture.zig");
const testing = std.testing;
const posix = std.posix;
const builtin = @import("builtin");

test "the host's flags through TensorFold 1.2.7 that zspark does not serve are named, each alone, and no other flag is" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var usage: host.cli.Usage = .{};
    const cases = [_]struct { []const []const u8, ?[]const u8 }{
        .{ &.{ "m", "--slide" }, "--slide" },
        .{ &.{ "m", "--slide-graph", "graph.json" }, "--slide-graph" },
        // each model's own, parted before the host's words (Flash Next's since spec 126)
        .{ &.{ "m", "--drafter", "dir" }, null },
        .{ &.{ "m", "--drafter-bits", "0" }, "--drafter-bits" },
        .{ &.{ "m", "--chat-template", "t.jinja" }, "--chat-template" },
        .{ &.{ "m", "--learn-min-free-gib", "1.5" }, "--learn-min-free-gib" },
        // since 1.0.12
        .{ &.{ "m", "--learn-conversations" }, "--learn-conversations" },
        .{ &.{ "m", "--learn-daily-gib", "8" }, "--learn-daily-gib" },
        .{ &.{ "m", "--recall" }, "--recall" },
        .{ &.{ "m", "--recall-dir", "d" }, "--recall-dir" },
        .{ &.{ "m", "--recall-gib", "2" }, "--recall-gib" },
        .{ &.{ "m", "--recall-read-only" }, "--recall-read-only" },
        .{ &.{ "m", "--record", "rows" }, "--record" },
        .{ &.{ "m", "--precision", "checkpoint" }, "--precision" },
        .{ &.{ "m", "--cortheon" }, "--cortheon" },
        .{ &.{ "m", "--vision" }, "--vision" },
        .{ &.{ "m", "--vision", "--vision-urls" }, "--vision" },
        .{ &.{ "m", "--vision", "--vision-max-images", "2" }, "--vision-max-images" },
        .{ &.{ "m", "--vision", "--vision-image-tokens", "1024" }, "--vision-image-tokens" },
        .{ &.{ "m", "--vision", "--vision-video-tokens", "1024" }, "--vision-video-tokens" },
        // since 1.1.1 (spec 113)
        .{ &.{ "m", "--vision-max-videos", "4" }, "--vision-max-videos" },
        .{ &.{ "m", "--kv-cache", "fp8" }, "--kv-cache" },
        .{ &.{ "m", "--kv-cache", "bf16", "--prompt-seed", "--allowed-host", "spark.lan", "--allowed-origin", "http://spark.lan:8888" }, null },
        // since 1.1.4 (spec 141): the speech server's, the idle unload, int8 prompt activations
        .{ &.{ "m", "--voices", "voices" }, "--voices" },
        .{ &.{ "m", "--default-voice", "narrator" }, "--default-voice" },
        .{ &.{ "m", "--allowed-controls", "laugh" }, "--allowed-controls" },
        .{ &.{ "m", "--max-frames", "64" }, "--max-frames" },
        .{ &.{ "m", "--silence-seconds", "2", "--silence-floor-dbfs", "-50" }, "--silence-seconds" },
        .{ &.{ "m", "--idle-unload", "600" }, "--idle-unload" },
        .{ &.{ "m", "--prompt-precision", "int8" }, "--prompt-precision" },
        .{ &.{ "m", "--prompt-precision", "bf16", "--idle-unload", "0", "--thinking-reserve", "512" }, null },
        // since 1.1.5 (spec 145): a looped model's passes
        .{ &.{ "m", "--loops", "2" }, "--loops" },
        // since 1.2.1 (spec 152): host-native flags zspark does not read; the FSR ones are macOS Metal's
        .{ &.{ "m", "--name-priority", "x=background" }, "--name-priority" },
        .{ &.{ "m", "--loop-guard-stop" }, "--loop-guard-stop" },
        .{ &.{ "m", "--rope-yarn", "2" }, "--rope-yarn" },
        .{ &.{ "m", "--decode-share", "0.5" }, "--decode-share" },
        // `--ssd-experts`, `--ple-on-ssd` and `--fsr-route-trace` are macOS-only in the host's flag table and
        // `--spill-gib` is zspark's own, parted before the host's words: the parser refuses all four itself.
        // the admission flags are the host's own 429s, off by default, and served
        .{ &.{ "m", "--max-inflight", "8", "--max-inflight-per-key", "2", "--requests-per-second", "10", "--request-burst", "4" }, null },
        // since 1.2.7: shared Startup resolves the thinking-off set; these are served host flags
        .{ &.{ "m", "--instruct-sampling", "none" }, null },
        .{ &.{ "m", "--instruct-sampling=card" }, null },
        .{ &.{ "m", "--instruct-sampling", "temperature=0.7,top_p=0.8,top_k=20,presence_penalty=0", "--temperature", "0.5" }, null },
        // the host's defaults: not told from the flag left out
        .{ &.{ "m", "--drafter-bits", "4", "--learn-min-free-gib", "4" }, null },
        .{ &.{"m"}, null },
        .{ &.{ "m", "--parallel", "4", "--context", "4096", "--no-drafts", "--learn", "--prompt-cache-gib", "2", "--port", "18888" }, null },
        // spec 019 WS-04: the host fetches image URLs
        .{ &.{ "m", "--vision-urls" }, null },
        .{ &.{ "m", "--cache-salt-per-key" }, "--cache-salt-per-key" },
        // the host opens its Unix listener directly from Args; the engine needs no socket-specific setup
        .{ &.{ "m", "--unix-socket", "zspark.sock" }, null },
    };
    for (cases) |case| {
        const args = try host.cli.parse(arena, case[0], &usage);
        const named = command.unserved(args, false);
        if (case[1]) |flag| try testing.expectEqualStrings(flag, named.?) else try testing.expect(named == null);
    }
    // A model that only decodes greedily cannot serve an instruct sampling set; turning it off is accepted.
    for ([_][]const u8{ "card", "temperature=0.7" }) |text| {
        const args = try host.cli.parse(arena, &.{ "m", "--instruct-sampling", text }, &usage);
        try testing.expectEqualStrings("--instruct-sampling", command.unserved(args, true).?);
    }
    const off = try host.cli.parse(arena, &.{ "m", "--instruct-sampling", "none" }, &usage);
    try testing.expect(command.unserved(off, true) == null);
    try testing.expect(command.unserved(.{}, true) == null);
}

test "a sampled request that names no seed draws a fresh one, as TensorFold 1.1.1's; --prompt-seed keys it to the prompt" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const fresh = (try command.hostArgs(arena_state.allocator(), "m", &.{ "--port", "18888" })).?;
    try testing.expect(!fresh.prompt_seed);
    const keyed = (try command.hostArgs(arena_state.allocator(), "m", &.{ "--port", "18888", "--prompt-seed" })).?;
    try testing.expect(keyed.prompt_seed);
}

test "--max-queued is taken out of the host's words, in both of the host's forms; not given nothing is bounded" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var usage: host.cli.Usage = .{};

    const none = (try command.bounded(arena, &.{ "--parallel", "4", "--port", "18888" })).?;
    try testing.expectEqual(null, none.max_queued);
    try testing.expectEqual(4, none.rest.len);
    const spaced = (try command.bounded(arena, &.{ "--parallel", "4", "--max-queued", "16", "--port", "18888" })).?;
    try testing.expectEqual(16, spaced.max_queued.?);
    const joined = (try command.bounded(arena, &.{ "--max-queued=0", "--parallel", "4", "--port", "18888" })).?;
    try testing.expectEqual(0, joined.max_queued.?);
    for ([_]command.Bounded{ spaced, joined }) |each| {
        try testing.expectEqual(4, each.rest.len);
        for (each.rest, [_][]const u8{ "--parallel", "4", "--port", "18888" }) |got, want| try testing.expectEqualStrings(want, got);
        // what is left is the host's, and the host reads it
        _ = try host.cli.parse(arena, try std.mem.concat(arena, []const u8, &.{ &.{"m"}, each.rest }), &usage);
    }
    // the host does not know the flag: left in its words it is a usage error
    try testing.expectError(error.Usage, host.cli.parse(arena, &.{ "m", "--max-queued", "16" }, &usage));
}

// A refusal is logged at err, which fails a test: only what is accepted is tested here, and the refusals'
// words by the diff of every command's text before and after (spec 136 WS-04).
test "the pair a command asks for: neither is one process, both need NCCL (spec 136 WS-04)" {
    const Place = struct {
        rank: ?usize = null,
        at: ?u16 = null,
        pub fn complete(self: @This()) bool {
            return self.rank != null and self.at != null;
        }
    };
    try testing.expectEqual(command.Ranked.alone, command.ranked("serve", Place{}, true, .flags));
    try testing.expectEqual(command.Ranked.paired, command.ranked("serve", Place{ .rank = 0, .at = 9000 }, true, .flags));
}

test "the bounds of --parallel and --context, and --spill-gib's three flags (spec 136 WS-04)" {
    var args: host.cli.Args = .{};
    try testing.expectEqual(4, command.parallelWithin(args, 4, 0, 64).?);
    try testing.expectEqual(1, command.parallelWithin(args, 1, 1, 64).?);
    args.parallel = "64";
    try testing.expectEqual(64, command.parallelWithin(args, 4, 1, 64).?);
    try testing.expectEqual(null, try command.askedContext(.{}, 64));
    args.context = 4096;
    try testing.expectEqual(4096, (try command.askedContext(args, 64)).?);

    try testing.expect(command.checkSpill(0, 4, null));
    try testing.expect(command.checkSpill(8, 0, "dir"));
}

/// A run of `startup_serve`: the shared startup, listener and tool-id key live in a process of their own.
/// The build supplies its executable path through `startup_fixture`; `stop` sends SIGTERM and expects 0.
const Drive = struct {
    child: std.process.Child = undefined,
    env: std.process.Environ.Map = undefined,

    /// `startup_serve serve argv...` under `environ` (the map replaces the child's whole environment;
    /// `argv[0]` is the model directory the flags follow), the bound port once its `PORT` line is read (0 on
    /// a Unix socket).
    fn begin(self: *Drive, a: std.mem.Allocator, environ: []const struct { []const u8, []const u8 }, argv: []const []const u8) !u16 {
        const exe = @import("startup_fixture").exe;
        self.env = .init(a);
        errdefer self.env.deinit();
        try self.env.put("TENSORFOLD_NO_LIVE", "1");
        try self.env.put("TENSORFOLD_WIRE", "spark");
        for (environ) |e| try self.env.put(e[0], e[1]);
        var words: std.ArrayList([]const u8) = .empty;
        try words.append(a, exe);
        try words.append(a, "serve");
        for (argv) |w| try words.append(a, w);
        self.child = try std.process.spawn(testing.io, .{ .argv = words.items, .environ_map = &self.env, .stdout = .ignore, .stderr = .pipe });
        errdefer self.child.kill(testing.io);
        return self.waitPort();
    }

    /// The driver's `PORT` line on its stderr, whatever logs stand before it; fails when it exits first.
    fn waitPort(self: *Drive) !u16 {
        const file = self.child.stderr orelse return error.NoPort;
        var seen: [4096]u8 = undefined;
        var at: usize = 0;
        while (at < seen.len) {
            var ready = [_]posix.pollfd{.{ .fd = file.handle, .events = posix.POLL.IN, .revents = 0 }};
            if (try posix.poll(&ready, 10_000) == 0) return error.StartupTimeout;
            const got = try posix.read(file.handle, seen[at..]);
            if (got == 0) break;
            at += got;
            if (std.mem.indexOf(u8, seen[0..at], "PORT ")) |p| {
                const rest = seen[p + 5 .. at];
                const end = std.mem.indexOfScalar(u8, rest, '\n') orelse continue;
                return try std.fmt.parseInt(u16, rest[0..end], 10);
            }
        }
        std.debug.print("startup fixture did not listen: {s}\n", .{seen[0..at]});
        return error.NoPort;
    }

    /// SIGTERM, as `zspark serve` ends: the drain, the exit status (0 expected).
    fn stop(self: *Drive) !u8 {
        try posix.kill(self.child.id orelse return error.NoChild, .TERM);
        const term = try self.child.wait(testing.io);
        return switch (term) {
            .exited => |code| code,
            else => error.Stopped,
        };
    }
};

/// A chat request the driver answers; the reply's body, arena-kept.
fn chat(a: std.mem.Allocator, port: u16, extra: []const u8) ![]const u8 {
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}]{s}}}", .{extra});
    const reply = try fixture.heard(a, try fixture.sent(port, try fixture.posted(a, "/v1/chat/completions", body)));
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200"));
    const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.NoReply;
    return reply[at + 4 ..];
}

/// Writes `body` under `name` in `tmp`, with the mode a key file must have; returns the path it got.
fn put(a: std.mem.Allocator, tmp: std.testing.TmpDir, name: []const u8, body: []const u8, mode: u32) ![]const u8 {
    try tmp.dir.writeFile(testing.io, .{ .sub_path = name, .data = body });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name });
    if (builtin.os.tag != .windows) {
        const z = try a.dupeSentinel(u8, path, 0);
        _ = std.c.fchmodat(posix.AT.FDCWD, z, mode, 0);
    }
    return path;
}

/// The tool id a reply's first call carries; fails when no call came back.
fn firstToolId(a: std.mem.Allocator, got: []const u8) ![]const u8 {
    const parsed = try host.json.parse(a, got);
    try testing.expect(parsed == .ok);
    const choices = parsed.ok.get("choices") orelse return error.NoReply;
    const calls = ((choices.items() orelse return error.NoReply)[0].get("message") orelse return error.NoReply).get("tool_calls") orelse return error.NoReply;
    return a.dupe(u8, (calls.items() orelse return error.NoReply)[0].strField("id") orelse return error.NoReply);
}

test "TENSORFOLD_TOOL_ID_KEY_FILE refuses an unfit file before anything listens" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const secret = "zspark startup first test key, not a secret";
    for ([_]struct { []const u8, u32 }{ .{ "", 0 }, .{ "tiny", 0o600 }, .{ secret, 0o640 } }) |c| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        var arena2: std.heap.ArenaAllocator = .init(testing.allocator);
        defer arena2.deinit();
        var env: std.process.Environ.Map = .init(testing.allocator);
        defer env.deinit();
        const path = if (c[0].len == 0)
            try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/missing", .{tmp.sub_path})
        else
            try put(a, tmp, "key", c[0], c[1]);
        try env.put("TENSORFOLD_TOOL_ID_KEY_FILE", path);
        try testing.expectEqual(@as(?command.Startup, null), try command.Startup.open(arena2.allocator(), &env));
    }
}

test "a configured tool id key keeps a call's id across server processes" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const secret = "zspark startup first test key, not a secret";
    const other = "zspark startup other test key, not a secret";
    const call_xml = "<tool_call>\n<function=lookup>\n<parameter=q>\na\n</parameter>\n</function>\n</tool_call>\n";
    const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}}]";

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const first_path = try put(a, tmp, "key", secret, 0o600);
    const second_path = try put(a, tmp, "key2", other, 0o600);

    // Each run is its own process on the same scripted reply: the same key keeps the call's id; a different
    // key or no key changes it.
    var ids: [3][]const u8 = @splat("");
    for ([_]?[]const u8{ first_path, second_path, null }, 0..) |key, i| {
        var drive: Drive = .{};
        const environ: []const struct { []const u8, []const u8 } = if (key) |p|
            &.{.{ "TENSORFOLD_TOOL_ID_KEY_FILE", p }}
        else
            &.{};
        const port = try drive.begin(a, environ, &.{ "--says", call_xml, ".", "--port", "0" });
        defer drive.env.deinit();
        defer drive.child.kill(testing.io);
        const got = try chat(a, port, tools);
        ids[i] = try firstToolId(a, got);
        try testing.expectEqual(@as(u8, 0), try drive.stop());
    }
    try testing.expect(!std.mem.eql(u8, ids[0], ids[1]));
    try testing.expect(!std.mem.eql(u8, ids[0], ids[2]));

    // Restarted on the first key, the id is the same bytes again: the file, not the process, is the key.
    var again: Drive = .{};
    const port = try again.begin(a, &.{.{ "TENSORFOLD_TOOL_ID_KEY_FILE", first_path }}, &.{ "--says", call_xml, ".", "--port", "0" });
    defer again.env.deinit();
    defer again.child.kill(testing.io);
    const got = try chat(a, port, tools);
    const id = try firstToolId(a, got);
    try testing.expectEqual(@as(u8, 0), try again.stop());
    try testing.expectEqualStrings(ids[0], id);
}

test "--instruct-sampling's pairs are refused by the host's parser when they name an out-of-range field" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    var usage: host.cli.Usage = .{};
    try testing.expectError(error.Usage, host.cli.parse(arena_state.allocator(), &.{ "m", "--instruct-sampling", "min_p=2" }, &usage));
    try testing.expectError(error.Usage, host.cli.parse(arena_state.allocator(), &.{ "m", "--instruct-sampling", "bogus=1" }, &usage));
}

test "the Unix listener serves, a second bind is refused, and SIGTERM removes the socket" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    if (builtin.os.tag != .windows) {
        const dir = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}, 0);
        _ = std.c.fchmodat(posix.AT.FDCWD, dir, 0o700, 0);
    }
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/s.sock", .{tmp.sub_path}, 0);

    var drive: Drive = .{};
    _ = try drive.begin(a, &.{}, &.{ ".", "--unix-socket", path });
    defer drive.env.deinit();
    defer drive.child.kill(testing.io);

    // The socket answers a chat request.
    const req = try fixture.posted(a, "/v1/chat/completions", "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}");
    var un: std.c.sockaddr.un = .{ .path = undefined };
    @memset(&un.path, 0);
    @memcpy(un.path[0..path.len], path);
    const fd = std.c.socket(std.c.AF.UNIX, std.c.SOCK.STREAM, 0);
    try testing.expect(fd >= 0);
    errdefer _ = posix.system.close(fd);
    const timeout: posix.timeval = .{ .sec = 10, .usec = 0 };
    try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.RCVTIMEO, std.mem.asBytes(&timeout));
    try testing.expectEqual(@as(c_int, 0), std.c.connect(fd, @ptrCast(&un), @sizeOf(std.c.sockaddr.un)));
    var at: usize = 0;
    while (at < req.len) {
        const wrote = posix.system.write(fd, req[at..].ptr, req.len - at);
        if (posix.errno(wrote) != .SUCCESS) return error.WriteFailed;
        at += @intCast(wrote);
    }
    const got = try fixture.heard(a, fd);
    try testing.expect(std.mem.startsWith(u8, got, "HTTP/1.1 200"));

    // A second bind on the live socket is refused.
    const again = host.listener.Listener.openUnix(path) catch null;
    try testing.expectEqual(@as(?host.listener.Listener, null), again);

    // SIGTERM drains, joins, and the socket file is gone.
    try testing.expectEqual(@as(u8, 0), try drive.stop());
    try testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(testing.io, path, .{}));
}
