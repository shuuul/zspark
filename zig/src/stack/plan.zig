//! What `zspark stack` starts: Flash Next, OCR and ASR behind `zspark-router`, the small models first,
//! Flash Next after them, the router last. With a peer, Flash Next spans both Sparks; Qwen3-ASR runs
//! on the peer, while Nemotron SpeechLM spans both Sparks on a separate communicator. Its ASR rank
//! pair starts together and reaches health before Flash Next starts. Nothing here starts a process.

const std = @import("std");
const supervisor = @import("supervisor.zig");
const net = std.Io.net;
const Allocator = std.mem.Allocator;
const Child = supervisor.Child;

pub const usage =
    \\usage: zspark stack FLASH_DIR [--ocr DIR] [--asr DIR] [--peer ADDR] [FLAG..] [-- SERVE_FLAG..]
    \\       zspark stack link-env PEER_ADDR
    \\
    \\  Flash Next, Unlimited-OCR-2 and Qwen3-ASR behind one address (spec 139): each a `zspark serve` of its
    \\  own on a loopback port, `zspark-router` over them on --listen. OCR, then ASR, each at --parallel 1 and
    \\  each waited on until its /health answers; then Flash Next, sizing itself from what the device has left;
    \\  then the router (`--default` Flash Next, `--audio` ASR). Each child's --name is its checkpoint's name:
    \\  the directory's, or under a Hugging Face cache's `models--ORG--NAME/snapshots/HASH` the NAME. Each
    \\  child's lines come out behind its name (--ocr-name, --asr-name give a small model's). SIGINT or SIGTERM stops all of them, the router first; a child
    \\  that ends on its own stops the rest, its exit said, and the stack exits 1.
    \\
    \\  --peer ADDR: Flash Next spans two ranks, rank 1 there, rank 0 here; OCR runs here. A Qwen3-ASR
    \\  checkpoint runs on the peer. Nemotron SpeechLM ASR spans both Sparks: rank 1 there and rank 0
    \\  here on port AT+1, then Flash Next's pair on AT. Its config is read to select placement, so
    \\  --asr DIR must name the same checkpoint on both Sparks for Nemotron. Each rank reads its own
    \\  NCCL link settings (`link-env`); remote children end when their ssh stdin closes.
    \\
    \\  --ocr DIR, --asr DIR    the small models; either may be left out
    \\  --ocr-name, --asr-name NAME  the id a small model is served and routed under, in place of its
    \\                          checkpoint's name (vLLM's --served-model-name): a build kept under its base's id
    \\  --listen HOST:PORT      the router's address (127.0.0.1:8888)
    \\  --port N                the children's first port: OCR N, ASR N+1, Flash Next N+2 (18101)
    \\  --peer-dir DIR          Flash Next's directory on the other Spark (FLASH_DIR)
    \\  --peer-zspark PATH      zspark on the other Spark (this one's path)
    \\  --at N                  Flash Next rank 1's link port (29551); two-rank ASR uses N+1
    \\  --health-s N            the longest a child's load may take before it answers /health (1800)
    \\  --stop-s N              the longest the children have to end once told, before SIGKILL (60)
    \\  --mps PCT               the small models' share of the SMs, in percent, under a CUDA MPS control daemon
    \\                          of the stack's own on each Spark that runs one (/tmp/zspark-mps-PORT, started
    \\                          before the children and quit after them); Flash Next keeps them all. 0: no
    \\                          daemon, the processes time-sliced, each about half the GPU beside another (25).
    \\                          The small models also run at MPS's below-normal client priority
    \\  -- SERVE_FLAG..         Flash Next's own `serve` flags, on both ranks (--parallel, --context, --name..);
    \\                          its --parallel is the router's limit for it (OCR's and ASR's are 1)
    \\
    \\  Flash Next's --parallel, where the flags after -- give none: on one Spark 2 beside OCR and ASR, 3
    \\  beside one of them, at its full --context (serve's 4 does not fit beside them, spec 139 WS-03);
    \\  serve's own alone (4); 4 on two Sparks (serve's two-rank default is 1).
    \\
    \\  link-env PEER_ADDR      print this Spark's NCCL settings for the link to PEER_ADDR, `K=V` a word
    \\
;

pub const Options = struct {
    flash: []const u8,
    ocr: ?[]const u8 = null,
    asr: ?[]const u8 = null,
    ocr_name: ?[]const u8 = null,
    asr_name: ?[]const u8 = null,
    peer: ?[]const u8 = null,
    peer_dir: ?[]const u8 = null,
    peer_zspark: ?[]const u8 = null,
    listen: []const u8 = "127.0.0.1:8888",
    port: u16 = 18101,
    at: u16 = 29551,
    health_s: u64 = 1800,
    stop_s: u64 = 60,
    /// the small models' CUDA_MPS_ACTIVE_THREAD_PERCENTAGE under the stack's MPS daemon; 0 no daemon
    mps: u8 = default_mps,
    serve_flags: []const []const u8 = &.{},
};

/// The options a command line gives; `said` the reason when it gives none.
pub fn parse(words: []const []const u8, said: *[]const u8) ?Options {
    var options: Options = .{ .flash = "" };
    var flash: ?[]const u8 = null;
    var i: usize = 0;
    while (i < words.len) : (i += 1) {
        const word = words[i];
        if (std.mem.eql(u8, word, "--")) {
            options.serve_flags = words[i + 1 ..];
            break;
        }
        if (!std.mem.startsWith(u8, word, "--")) {
            if (flash != null) {
                said.* = "one FLASH_DIR; Flash Next's flags go after --";
                return null;
            }
            flash = word;
            continue;
        }
        i += 1;
        if (i == words.len) {
            said.* = "a flag was left without its value";
            return null;
        }
        const value = words[i];
        if (std.mem.eql(u8, word, "--ocr")) {
            options.ocr = value;
        } else if (std.mem.eql(u8, word, "--asr")) {
            options.asr = value;
        } else if (std.mem.eql(u8, word, "--ocr-name")) {
            options.ocr_name = value;
        } else if (std.mem.eql(u8, word, "--asr-name")) {
            options.asr_name = value;
        } else if (std.mem.eql(u8, word, "--peer")) {
            options.peer = value;
        } else if (std.mem.eql(u8, word, "--peer-dir")) {
            options.peer_dir = value;
        } else if (std.mem.eql(u8, word, "--peer-zspark")) {
            options.peer_zspark = value;
        } else if (std.mem.eql(u8, word, "--listen")) {
            options.listen = value;
        } else if (std.mem.eql(u8, word, "--port") or std.mem.eql(u8, word, "--at")) {
            const n = std.fmt.parseInt(u16, value, 10) catch {
                said.* = "a port is a number below 65536";
                return null;
            };
            if (word[2] == 'p') options.port = n else options.at = n;
        } else if (std.mem.eql(u8, word, "--health-s") or std.mem.eql(u8, word, "--stop-s")) {
            const n = std.fmt.parseInt(u64, value, 10) catch {
                said.* = "seconds are a number";
                return null;
            };
            if (word[2] == 'h') options.health_s = n else options.stop_s = n;
        } else if (std.mem.eql(u8, word, "--mps")) {
            options.mps = std.fmt.parseInt(u8, value, 10) catch 101;
            if (options.mps > 100) {
                said.* = "--mps is a percentage, 0 to 100";
                return null;
            }
        } else {
            said.* = "a flag the stack does not take; Flash Next's own go after --";
            return null;
        }
    }
    options.flash = flash orelse {
        said.* = "no FLASH_DIR";
        return null;
    };
    if (parseAddress(options.listen) == null) {
        said.* = "--listen is HOST:PORT, the host an IPv4 address";
        return null;
    }
    if (options.peer) |peer| if ((net.Ip4Address.parse(peer, 0) catch null) == null) {
        said.* = "--peer is the other Spark's IPv4 link address";
        return null;
    };
    return options;
}

/// A checkpoint's name from its directory: the last part, or a Hugging Face cache's repository name where the
/// directory is `models--ORG--NAME/snapshots/HASH`.
pub fn modelName(dir: []const u8) []const u8 {
    const trimmed = std.mem.trimEnd(u8, dir, "/");
    var parts = std.mem.splitBackwardsScalar(u8, trimmed, '/');
    const last = parts.next() orelse return trimmed;
    const snapshots = parts.next() orelse return last;
    const repository = parts.next() orelse return last;
    if (!std.mem.eql(u8, snapshots, "snapshots") or !std.mem.startsWith(u8, repository, "models--")) return last;
    const cut = std.mem.lastIndexOf(u8, repository, "--").?;
    return if (cut + 2 < repository.len) repository[cut + 2 ..] else last;
}

/// The value `flag` takes in `flags`, the last given; null when it is not there.
pub fn flagValue(flags: []const []const u8, flag: []const u8) ?[]const u8 {
    var found: ?[]const u8 = null;
    for (flags, 0..) |word, i| {
        if (std.mem.eql(u8, word, flag) and i + 1 < flags.len) found = flags[i + 1];
    }
    return found;
}

/// `HOST:PORT`, the host an IPv4 literal.
pub fn parseAddress(text: []const u8) ?net.IpAddress {
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return null;
    const port = std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return null;
    return net.IpAddress.parse(text[0..colon], port) catch null;
}

/// `text` as one word of a POSIX shell: single quotes, a quote inside closed, escaped and opened again.
pub fn quote(arena: Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(arena, '\'');
    for (text) |c| {
        if (c == '\'') try out.appendSlice(arena, "'\\''") else try out.append(arena, c);
    }
    try out.append(arena, '\'');
    return out.items;
}

/// The ssh argv that runs `argv` on `peer` with `environ` (`K=V` words) before it, under a shell that ends it
/// when its stdin ends (kept as fd 3, since a background job's stdin is /dev/null in a shell without job control): the stack's stop closes the pipe, and an ssh that dies closes it too, so no child is
/// left on the other Spark. The shell exits with the child's status, which ssh hands back.
pub fn remote(arena: Allocator, peer: []const u8, environ: []const []const u8, argv: []const []const u8) ![]const []const u8 {
    var command: std.ArrayList(u8) = .empty;
    if (environ.len > 0) try command.appendSlice(arena, "env ");
    for (environ) |word| {
        try command.appendSlice(arena, try quote(arena, word));
        try command.append(arena, ' ');
    }
    for (argv, 0..) |word, i| {
        if (i > 0) try command.append(arena, ' ');
        try command.appendSlice(arena, try quote(arena, word));
    }
    const script = try std.fmt.allocPrint(arena,
        \\exec 3<&0
        \\{s} </dev/null 3<&- & p=$!
        \\(cat <&3 >/dev/null; kill -TERM $p) >/dev/null 2>&1 & w=$!
        \\exec 3<&-
        \\wait $p; s=$?; kill $w 2>/dev/null; exit $s
    , .{command.items});
    const line = try std.fmt.allocPrint(arena, "exec sh -c {s}", .{try quote(arena, script)});
    return try arena.dupe([]const u8, &.{ "ssh", "-o", "BatchMode=yes", peer, line });
}

/// Where the binaries are and what each side's environment adds for the ranks.
pub const Binaries = struct {
    zspark: []const u8,
    router: []const u8,
    /// the stack's own environment, what a child given variables of its own starts from (empty where null)
    environ: ?*const std.process.Environ.Map = null,
    /// this Spark's NCCL settings for the link, rank 0's environment (two Sparks only)
    environ_here: ?*const std.process.Environ.Map = null,
    /// the other Spark's, `K=V` words before rank 1's command, with what both ranks must read alike
    environ_there: []const []const u8 = &.{},
    /// The ASR checkpoint is Nemotron SpeechLM: with a peer it spans both Sparks, before Flash Next.
    asr_split: bool = false,
};

/// The small models' share of the SMs under the stack's MPS daemon (spec 139 WS-06, measured 2026-10-26 on one
/// Spark and two: Flash Next's recipe has the table). Time slicing gives each process about half the GPU beside another
/// whatever its size, Flash Next's decoding 22 tok/s of 53 beside OCR's pages; under MPS the three run at once,
/// the small models held to this share of the SMs at a lower priority: Flash Next 43 beside OCR, OCR's pages
/// then 4x their time alone where time slicing made them 2x. Flash Next first, as the owner orders the work.
pub const default_mps: u8 = 25;

/// Where a stack's MPS daemon keeps its pipes and logs on each Spark: its own, so only the stack's children
/// (which are told it) are its clients; every other CUDA process on the Spark runs as it did. By the children's
/// first port, so two stacks on a Spark keep apart.
pub fn mpsDirectory(arena: Allocator, port: u16) ![]const u8 {
    return std.fmt.allocPrint(arena, "/tmp/zspark-mps-{d}", .{port});
}

/// The variables that make a child a client of the daemon in `dir`, `K=V` a word. A small model's (`percentage`
/// its share of the SMs) also takes MPS's below-normal client priority (1), so Flash Next's kernels go first
/// where both wait: 25% alone left Flash Next 39 tok/s beside OCR, 25% and priority 1 43. Flash Next's (null):
/// all the SMs at the normal priority.
pub fn mpsWords(arena: Allocator, dir: []const u8, percentage: ?u8) ![]const []const u8 {
    var words: std.ArrayList([]const u8) = .empty;
    try words.append(arena, try std.fmt.allocPrint(arena, "CUDA_MPS_PIPE_DIRECTORY={s}/pipe", .{dir}));
    try words.append(arena, try std.fmt.allocPrint(arena, "CUDA_MPS_LOG_DIRECTORY={s}/log", .{dir}));
    if (percentage) |p| {
        try words.append(arena, try std.fmt.allocPrint(arena, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={d}", .{p}));
        try words.append(arena, "CUDA_MPS_CLIENT_PRIORITY=1");
    }
    return words.items;
}

/// The shell line that starts the daemon in `dir` (`-d` returns once it listens; its descriptors to /dev/null, or
/// the reader of the line's output, here or ssh, would wait on the daemon), a daemon a dead stack left there quit first; and the one that quits it, its servers forced after 20 s. As the user, no root: the daemon
/// and its server run as the stack's user.
pub fn mpsLine(arena: Allocator, dir: []const u8, start: bool) ![]const u8 {
    const env = try std.fmt.allocPrint(arena, "export CUDA_MPS_PIPE_DIRECTORY={0s}/pipe CUDA_MPS_LOG_DIRECTORY={0s}/log; ", .{dir});
    const quit = "echo 'quit -t 20' | timeout 60 nvidia-cuda-mps-control";
    if (!start) return std.fmt.allocPrint(arena, "{s}{s}", .{ env, quit });
    return std.fmt.allocPrint(arena, "{s}mkdir -p {1s}/pipe {1s}/log && {{ {2s} >/dev/null 2>&1; nvidia-cuda-mps-control -d </dev/null >/dev/null 2>&1; }}", .{ env, dir, quit });
}

/// `base` with `words` (`K=V`) put over it: a child's own environment.
fn withWords(arena: Allocator, base: ?*const std.process.Environ.Map, words: []const []const u8) !*const std.process.Environ.Map {
    const map = try arena.create(std.process.Environ.Map);
    map.* = if (base) |b| try b.clone(arena) else .init(arena);
    for (words) |word| {
        const eq = std.mem.indexOfScalar(u8, word, '=').?;
        try map.put(word[0..eq], word[eq + 1 ..]);
    }
    return map;
}

/// Flash Next's `--parallel` where its flags give none: on one Spark beside the small models, the most that
/// fits at its full context with its weights and session (81.6 GiB) after them (spec 139 WS-03, measured
/// 2026-10-26: 17.5 GiB left beside both, 24.5 beside OCR, 26.0 beside ASR, against 25.9 for 4 streams and
/// 19.1 for 3, before the states). Lowered before `--context`, by the spec's decision; ASR alone takes 3 as
/// OCR does, since 4 would leave the states 0.15 GiB of their 6.66. On two Sparks 4, not `serve`'s two-rank 1:
/// the stack serves several clients, and each rank's bf16 pool of 4 leaves 37.6 and 39.4 GiB free beside OCR and ASR
/// (2026-10-26; FP8's 43.5 and 45.7; 70.3 tok/s alone, 127.5 over 4 streams against 77.4 and 70.6 at 1). Null: `serve`'s own, 4.
pub fn defaultParallel(options: Options) ?u8 {
    if (options.peer != null) return 4;
    const small = @as(u8, @intFromBool(options.ocr != null)) + @intFromBool(options.asr != null);
    return switch (small) {
        0 => null,
        1 => 3,
        else => 2,
    };
}

pub const Plan = struct {
    waves: []const []const Child,
    /// Flash Next's `--parallel` the stack chose (`defaultParallel`); null where the flags gave one or none was needed.
    chosen_parallel: ?u8 = null,
    flash_name: []const u8,
    ocr_name: ?[]const u8,
    asr_name: ?[]const u8,
    /// the MPS daemon's directory (`mpsDirectory`) where one runs: here when a small model runs here, there
    /// (`--peer`) when ASR runs there
    mps_dir: ?[]const u8 = null,
    mps_here: bool = false,
    mps_there: bool = false,
};

/// The children and their waves: one Spark, [OCR] [ASR] [Flash Next] [router]; two Sparks, [OCR, ASR there]
/// [rank 1 there, rank 0] [router]. An absent small model is left out with its wave.
pub fn build(arena: Allocator, options: Options, bin: Binaries) !Plan {
    const flash_name = flagValue(options.serve_flags, "--name") orelse modelName(options.flash);
    const chosen: ?u8 = if (flagValue(options.serve_flags, "--parallel") == null) defaultParallel(options) else null;
    const chosen_text: ?[]const u8 = if (chosen) |n| try std.fmt.allocPrint(arena, "{d}", .{n}) else null;
    // the router's limit for Flash Next: what its serve runs, its own default where nothing names one
    const parallel = flagValue(options.serve_flags, "--parallel") orelse chosen_text orelse "4";
    const loopback = "127.0.0.1";
    const asr_split = bin.asr_split and options.peer != null;
    const mps_here = options.mps > 0 and (options.ocr != null or (options.asr != null and options.peer == null and !bin.asr_split));
    // a split ASR is not a small model: it holds its own share and is left at full SM priority
    const mps_there = options.mps > 0 and options.peer != null and options.asr != null and !asr_split;
    const mps_dir: ?[]const u8 = if (mps_here or mps_there) try mpsDirectory(arena, options.port) else null;
    // a small model's words and Flash Next's: the same daemon, the small ones held to their share
    const small_words: []const []const u8 = if (mps_dir) |d| try mpsWords(arena, d, options.mps) else &.{};
    const flash_words: []const []const u8 = if (mps_dir) |d| try mpsWords(arena, d, null) else &.{};
    var waves: std.ArrayList([]const Child) = .empty;
    var small: std.ArrayList(Child) = .empty;
    var asr_pair: ?[2]Child = null;
    var router_argv: std.ArrayList([]const u8) = .empty;
    try router_argv.appendSlice(arena, &.{ bin.router, "--listen", options.listen, "--default", flash_name, "--parallel", parallel });
    var copies: std.ArrayList([]const u8) = .empty;
    try copies.append(arena, try std.fmt.allocPrint(arena, "{s}={s}:{d}", .{ flash_name, loopback, options.port + 2 }));

    var ocr_name: ?[]const u8 = null;
    if (options.ocr) |dir| {
        const name = options.ocr_name orelse modelName(dir);
        ocr_name = name;
        const port = options.port;
        try small.append(arena, .{
            .name = name,
            .argv = try serveArgv(arena, bin.zspark, dir, loopback, port, name, &.{ "--parallel", "1" }),
            .health = .{ .ip4 = try .parse(loopback, port) },
            .environ = if (mps_here) try withWords(arena, bin.environ, small_words) else null,
        });
        try copies.append(arena, try std.fmt.allocPrint(arena, "{s}={s}:{d}", .{ name, loopback, port }));
        try router_argv.appendSlice(arena, &.{ "--parallel-of", try std.fmt.allocPrint(arena, "{s}=1", .{name}) });
    }
    var asr_name: ?[]const u8 = null;
    if (options.asr) |dir| {
        const name = options.asr_name orelse modelName(dir);
        asr_name = name;
        const port = options.port + 1;
        const host = if (asr_split) loopback else options.peer orelse loopback;
        if (asr_split) {
            if (options.at == std.math.maxInt(u16)) return error.InvalidPort;
            const at = try std.fmt.allocPrint(arena, "{s}:{d}", .{ options.peer.?, options.at + 1 });
            const rank1 = try serveArgv(arena, options.peer_zspark orelse bin.zspark, dir, loopback, port, name, &.{ "--rank", "1", "--at", at });
            const rank0 = try serveArgv(arena, bin.zspark, dir, loopback, port, name, &.{ "--rank", "0", "--at", at });
            asr_pair = .{
                .{
                    .name = try std.fmt.allocPrint(arena, "{s} rank 1", .{name}),
                    .argv = try remote(arena, options.peer.?, bin.environ_there, rank1),
                    .remote = true,
                },
                .{
                    .name = try std.fmt.allocPrint(arena, "{s} rank 0", .{name}),
                    .argv = rank0,
                    .health = .{ .ip4 = try .parse(loopback, port) },
                    .environ = bin.environ_here,
                },
            };
        } else {
            const argv = try serveArgv(arena, if (options.peer != null) options.peer_zspark orelse bin.zspark else bin.zspark, dir, host, port, name, &.{ "--parallel", "1" });
            try small.append(arena, .{
                .name = name,
                .argv = if (options.peer) |peer| try remote(arena, peer, small_words, argv) else argv,
                .health = .{ .ip4 = try .parse(host, port) },
                .remote = options.peer != null,
                .environ = if (options.peer == null and mps_here) try withWords(arena, bin.environ, small_words) else null,
            });
        }
        try copies.append(arena, try std.fmt.allocPrint(arena, "{s}={s}:{d}", .{ name, host, port }));
        try router_argv.appendSlice(arena, &.{ "--audio", name, "--parallel-of", try std.fmt.allocPrint(arena, "{s}=1", .{name}) });
    }
    if (options.peer == null) {
        // one Spark: each small model waited on alone, as each sizes itself from what the device has
        for (small.items) |child| try waves.append(arena, try arena.dupe(Child, &.{child}));
    } else if (small.items.len > 0) {
        // two Sparks: one small model a Spark, so the two load at once
        try waves.append(arena, small.items);
    }
    if (asr_pair) |pair| try waves.append(arena, try arena.dupe(Child, &pair));

    const flash_port = options.port + 2;
    var rank0_flags: std.ArrayList([]const u8) = .empty;
    if (flagValue(options.serve_flags, "--name") == null) try rank0_flags.appendSlice(arena, &.{ "--name", flash_name });
    if (chosen_text) |n| try rank0_flags.appendSlice(arena, &.{ "--parallel", n });
    if (options.peer) |peer| {
        const at = try std.fmt.allocPrint(arena, "{s}:{d}", .{ peer, options.at });
        var rank1: std.ArrayList([]const u8) = .empty;
        try rank1.appendSlice(arena, &.{ options.peer_zspark orelse bin.zspark, "serve", options.peer_dir orelse options.flash, "--rank", "1", "--at", at });
        try rank1.appendSlice(arena, rank0_flags.items);
        try rank1.appendSlice(arena, options.serve_flags);
        try rank0_flags.appendSlice(arena, &.{ "--rank", "0", "--at", at });
        try rank0_flags.appendSlice(arena, options.serve_flags);
        const rank1_name = try std.fmt.allocPrint(arena, "{s} rank 1", .{flash_name});
        const rank0_name = try std.fmt.allocPrint(arena, "{s} rank 0", .{flash_name});
        // rank 1 first: it listens, and rank 0 tries its address for a minute
        try waves.append(arena, try arena.dupe(Child, &.{
            .{ .name = rank1_name, .argv = try remote(arena, peer, if (mps_there) try std.mem.concat(arena, []const u8, &.{ bin.environ_there, flash_words }) else bin.environ_there, rank1.items), .remote = true },
            .{
                .name = rank0_name,
                .argv = try serveArgv(arena, bin.zspark, options.flash, loopback, flash_port, null, rank0_flags.items),
                .health = .{ .ip4 = try .parse(loopback, flash_port) },
                .environ = if (mps_here) try withWords(arena, bin.environ_here orelse bin.environ, flash_words) else bin.environ_here,
            },
        }));
    } else {
        try rank0_flags.appendSlice(arena, options.serve_flags);
        try waves.append(arena, try arena.dupe(Child, &.{.{
            .name = flash_name,
            .argv = try serveArgv(arena, bin.zspark, options.flash, loopback, flash_port, null, rank0_flags.items),
            .health = .{ .ip4 = try .parse(loopback, flash_port) },
            .environ = if (mps_here) try withWords(arena, bin.environ, flash_words) else null,
        }}));
    }

    try router_argv.appendSlice(arena, copies.items);
    try waves.append(arena, try arena.dupe(Child, &.{.{
        .name = "router",
        .argv = router_argv.items,
        .health = parseAddress(options.listen).?,
    }}));
    return .{
        .waves = waves.items,
        .chosen_parallel = chosen,
        .flash_name = flash_name,
        .ocr_name = ocr_name,
        .asr_name = asr_name,
        .mps_dir = mps_dir,
        .mps_here = mps_here,
        .mps_there = mps_there,
    };
}

/// `zspark serve DIR --host HOST --port PORT [--name NAME] FLAG..`.
fn serveArgv(arena: Allocator, zspark: []const u8, dir: []const u8, host: []const u8, port: u16, name: ?[]const u8, flags: []const []const u8) ![]const []const u8 {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(arena, &.{ zspark, "serve", dir, "--host", host, "--port", try std.fmt.allocPrint(arena, "{d}", .{port}) });
    if (name) |n| try argv.appendSlice(arena, &.{ "--name", n });
    try argv.appendSlice(arena, flags);
    return argv.items;
}

/// The ZSPARK_* variables both ranks must read alike, handed to rank 1 as this shell has them
/// (tools/two_ranks.sh's list).
pub const shared_variables = [_][]const u8{ "ZSPARK_RANKS_HEAD", "ZSPARK_DRAFTER_STORAGE", "ZSPARK_QUARANTINE", "ZSPARK_LOAD_PHASES" };
