//! `zspark generate DIR [rank=R at=ADDR:PORT] [OPTION..] ID..`: a prompt's ids through GLM-5.3-Flash and a reply
//! decoded, timed (spec 045 WS-04). With `rank=` and `at=` this process is one rank of two: it loads its share,
//! joins the other through the control channel and NCCL (`zig/src/ranks/join.zig`) and decodes with the other's
//! partials and half of each chosen row's logits gathered. Without them both ranks are in this process
//! (`loopback.zig`), which a Spark holds only for the first few layers (`layers=L`): the reference a pair is
//! held to. One decoder either way (`draft.Namespace`), drafted by the MTP layer (`mtp.HeadOf`) unless `serial`,
//! or by the DFlash2 drafter of `drafter=DIR` (`dflash2/rig.zig`), which the join's text names. `mtp` has the
//! MTP layer draft a reply though a drafter is loaded, so one load records both. `chains` verifies no draft:
//! every round is one row and the drafter's chain from it is printed with the probability it gave each pick,
//! a line a position (`chain at P: 0 drafts, 0 kept; TOKEN:P ..`, DFlash2's `TOKEN:P:NOISY`), which `tools/reference/draft_policy.py`
//! plays every stop rule over (spec 067 WS-12). The sampling's words and `mtp` after a `then` are the prompts'
//! from there on, so one load records greedy and sampled replies by both drafters. So are DFlash2's rule's:
//! `cap=N`, `confidence=F` and `noisy` (the product of the kit's noise-aware confidences), for one load to time
//! several rules on the same replies.
//!
//! Of a pair rank 0 leads and rank 1 follows (`zig/src/ranks/lead.zig`, `mirror.zig`): rank 0 decodes and tells each
//! operation before it runs it, rank 1 runs what it is told and chooses nothing, and each watches the
//! connection for the other's end. With `alike` both decode instead: they choose from the same gathered row
//! with the same key, each prints its reply, and they compare it through the channel. `then` between two prompts decodes both
//! on one load (`session.zig`), each from an empty sequence, or with `onward` from what the caches hold where
//! the prompt begins with it.

const std = @import("std");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const sampling = @import("../../../decode/sampling.zig");
const glm5_layout = @import("../layers/layout.zig");
const rig = @import("../dflash2/rig.zig");
const session_file = @import("../session/session.zig");
const dsa = @import("../layers/dsa.zig");
const snapshot_file = @import("../session/snapshot.zig");
const common = @import("common.zig");
const world = glm5_layout.world;

pub const usage = "generate DIR [rank=R at=ADDR:PORT] [max=N] [layers=L] [rows=N] [temperature=T] [seed=N] [top_k=K] [top_p=P] [eos=no] [eager] [serial] [confidence=F] [drafter=DIR] [cap=N] [noisy[=no]] [mtp] [chains] [alike] [onward] [kv=bf16|fp8] ID.. [then ID..]";

const Options = struct {
    /// The reply's tokens at most.
    max: usize = 16,
    /// The first layers alone, to try the command; null is the model.
    layers: ?usize = null,
    /// A prompt window's rows at most.
    rows: usize = 16,
    /// Greedy unless a temperature is given.
    settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 },
    /// Whether an end token ends the reply.
    ends: bool = true,
    /// Every window launched, none through a captured graph.
    eager: bool = false,
    /// Both ranks of a pair decode, neither follows.
    alike: bool = false,
    /// No drafts: a round is one row.
    serial: bool = false,
    /// A prompt after `then` that begins with what the caches hold runs its rest alone, as a served turn does.
    onward: bool = false,
    /// A chain of drafts stops below this product of their probabilities; null is the drafter's own default.
    confidence: ?f32 = null,
    /// The DFlash2 drafter's directory: it drafts in the MTP layer's place.
    drafter: ?[]const u8 = null,
    /// The drafter's linears BF16 as the file has them, not NVFP4 made at load.
    drafter_stored: bool = false,
    /// The MTP layer drafts though a drafter is loaded.
    mtp: bool = false,
    /// DFlash2's drafts a round at most, and whether its chain stops by the noise-aware confidences.
    cap: ?usize = null,
    noisy: ?bool = null,
    /// A line a position with the drafter's whole chain; no draft is verified.
    chains: bool = false,
    /// The DSA caches' latents (`kv=bf16|fp8`); null is the checkpoint's (`dsa.Storage.stored`).
    kv: ?dsa.Storage = null,

    /// Whether `word` was an option; one that cannot be read is an error.
    fn take(self: *Options, word: []const u8) !bool {
        if (cli.value(word, "max=")) |text| {
            self.max = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "layers=")) |text| {
            self.layers = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "rows=")) |text| {
            self.rows = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "temperature=")) |text| {
            self.settings.temperature = try std.fmt.parseFloat(f64, text);
        } else if (cli.value(word, "seed=")) |text| {
            self.settings.seed = try std.fmt.parseInt(u64, text, 10);
        } else if (cli.value(word, "top_k=")) |text| {
            self.settings.top_k = try std.fmt.parseInt(u32, text, 10);
        } else if (cli.value(word, "top_p=")) |text| {
            self.settings.top_p = try std.fmt.parseFloat(f64, text);
        } else if (std.mem.eql(u8, word, "eos=no")) {
            self.ends = false;
        } else if (std.mem.eql(u8, word, "eager")) {
            self.eager = true;
        } else if (std.mem.eql(u8, word, "alike")) {
            self.alike = true;
        } else if (std.mem.eql(u8, word, "serial")) {
            self.serial = true;
        } else if (cli.value(word, "cap=")) |text| {
            self.cap = try std.fmt.parseInt(usize, text, 10);
        } else if (std.mem.eql(u8, word, "noisy")) {
            self.noisy = true;
        } else if (std.mem.eql(u8, word, "noisy=no")) {
            self.noisy = false;
        } else if (std.mem.eql(u8, word, "mtp")) {
            self.mtp = true;
        } else if (std.mem.eql(u8, word, "chains")) {
            self.chains = true;
        } else if (std.mem.eql(u8, word, "onward")) {
            self.onward = true;
        } else if (cli.value(word, "kv=")) |text| {
            self.kv = dsa.Storage.parse(text) orelse return error.InvalidCharacter;
        } else if (cli.value(word, "confidence=")) |text| {
            self.confidence = try std.fmt.parseFloat(f32, text);
        } else if (cli.value(word, "drafter=")) |text| {
            self.drafter = text;
        } else if (std.mem.eql(u8, word, "drafter_stored")) {
            self.drafter_stored = true;
        } else return false;
        return true;
    }
};

/// How one prompt is decoded: the words before it.
const Way = struct { settings: sampling.Sampling, mtp: bool, cap: ?usize, confidence: ?f32, noisy: ?bool };

/// `chains`' lines of one reply, written once it is whole.
const Record = struct {
    lines: std.Io.Writer.Allocating,

    fn chain(context: *anyopaque, position: usize, trail: []const session_file.Chains.Trail, noisy: []const f64) anyerror!void {
        const self: *Record = @ptrCast(@alignCast(context));
        try self.lines.writer.print("chain at {d}: 0 drafts, 0 kept;", .{position});
        for (trail, 0..) |one, i| {
            try self.lines.writer.print(" {d}:{d:.4}", .{ one.token, one.p });
            if (i < noisy.len) try self.lines.writer.print(":{d:.4}", .{noisy[i]});
        }
        try self.lines.writer.writeByte('\n');
    }
};

fn milliseconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    const arena = init.arena.allocator();
    var place: join_file.Place = .{};
    var options: Options = .{};
    var prompt: std.ArrayList(i32) = .empty;
    // where each prompt after the first begins, and how the one before it is decoded
    var cuts: std.ArrayList(usize) = .empty;
    var ways: std.ArrayList(Way) = .empty;
    // in any order: `tools/two_ranks.sh` writes a rank's place after the command's own words
    for (words[1..]) |word| {
        if (place.take(word)) continue;
        const taken = options.take(word) catch {
            std.log.err("generate: '{s}' cannot be read", .{word});
            return 2;
        };
        if (taken) continue;
        if (std.mem.eql(u8, word, "then")) {
            try cuts.append(arena, prompt.items.len);
            try ways.append(arena, .{ .settings = options.settings, .mtp = options.mtp, .cap = options.cap, .confidence = options.confidence, .noisy = options.noisy });
            continue;
        }
        try prompt.append(arena, std.fmt.parseInt(i32, word, 10) catch {
            std.log.err("generate: '{s}' is no option and no token id", .{word});
            return 2;
        });
    }
    const paired = place.rank != null or place.at != null;
    try cuts.append(arena, prompt.items.len);
    try ways.append(arena, .{ .settings = options.settings, .mtp = options.mtp, .cap = options.cap, .confidence = options.confidence, .noisy = options.noisy });
    var from: usize = 0;
    var empty = false;
    const prompts = try arena.alloc([]const i32, cuts.items.len);
    for (prompts, cuts.items) |*one, to| {
        one.* = prompt.items[from..to];
        from = to;
        empty = empty or one.len == 0;
    }
    if (empty or options.max == 0 or options.rows == 0 or (paired and !place.complete())) {
        try stdout.writeAll("usage: zspark " ++ usage ++ "\n");
        return 2;
    }
    for (ways.items) |way| way.settings.validate() catch |err| {
        std.log.err("generate: {t}", .{err});
        return 2;
    };
    if (paired and !nccl.linked) {
        std.log.err("generate: this build has no NCCL; build with -Dnccl=PREFIX (/usr for the apt packages)", .{});
        return 1;
    }
    var opened: common.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.enter(init.io, words[0], .raise) or !try opened.configure(init) or !try opened.resolve(init)) return 1;
    const layers = options.layers orelse opened.config.layers;
    if (layers == 0 or layers > opened.config.layers) {
        std.log.err("generate: the model has {d} layers", .{opened.config.layers});
        return 1;
    }
    for (prompt.items) |id| if (id < 0 or id >= opened.config.vocab) {
        std.log.err("generate: {d} is no token of a vocabulary of {d}", .{ id, opened.config.vocab });
        return 1;
    };
    // the drafter's directory, held to the model before anything loads
    var drafting: rig.Opened = .{};
    defer drafting.close(init.io);
    if (options.drafter) |path| if (!try rig.enter(init, &drafting, path, &opened.config, layers)) return 1;
    const drafter: ?*const rig.Opened = if (options.drafter != null) &drafting else null;
    return if (paired)
        decode(1, init, stdout, &opened, drafter, place, options, layers, prompts, ways.items)
    else
        decode(world, init, stdout, &opened, drafter, place, options, layers, prompts, ways.items);
}

fn decode(comptime held: usize, init: std.process.Init, stdout: *std.Io.Writer, opened: *common.Opened, drafter: ?*const rig.Opened, place: join_file.Place, options: Options, layers: usize, prompts: []const []const i32, ways: []const Way) !u8 {
    const Session = session_file.SessionOf(held);
    const arena = init.arena.allocator();
    const c = &opened.config;
    const s = options.settings;
    var longest: usize = 0;
    var key: u64 = 0;
    for (prompts, ways) |prompt, way| {
        longest = @max(longest, prompt.len);
        key = std.hash.Wyhash.hash(key +% prompt.len, std.mem.sliceAsBytes(prompt));
        key = std.hash.Wyhash.hash(key +% @intFromBool(way.mtp), std.mem.asBytes(&[_]f64{ way.settings.temperature, way.settings.top_p, @floatFromInt(way.settings.top_k), @floatFromInt(way.settings.seed) }));
    }
    // what changes the reply and is not the session's, alike on both ranks
    const what = try std.fmt.allocPrint(arena, "generate max={d} t={d} k={d} p={d} seed={d} ends={} prompts={d}:{x}", .{ options.max, s.temperature, s.top_k, s.top_p, s.seed, options.ends, prompts.len, key });
    const session = try Session.create(init.gpa, arena, init.io, &opened.checkpoint.?, c, place, .{
        .layers = layers,
        .rows = options.rows,
        .capacity = longest + options.max,
        .eager = options.eager,
        .serial = options.serial,
        .alike = options.alike,
        .confidence = options.confidence,
        .cap = options.cap,
        .drafter = drafter,
        .drafter_stored = options.drafter_stored,
        .snapshots = snapshot_file.held_default,
        .stride = snapshot_file.stride_default,
        .kv = options.kv orelse .stored(c),
    }, what);
    defer session.destroy();
    const first = session.first;
    const channel = &session.joined.channel;
    const gib = @as(f64, @floatFromInt(cuda.account.bytes)) / (1024.0 * 1024.0 * 1024.0);
    const seconds = milliseconds(session.load_ns) / 1e3;
    if (Session.paired) {
        try stdout.print("rank {d} of {d}: zspark {s}{s}, NCCL {d}, {d} of {d} layers, its share {d:.2} GiB in {d:.1} s\n", .{ first, world, cli.stamp.commit, if (cli.stamp.dirty) " dirty" else "", try nccl.version(), layers, c.layers, gib, seconds });
    } else try stdout.print("both ranks in this process: {d} of {d} layers, {d:.2} GiB in {d:.1} s\n", .{ layers, c.layers, gib, seconds });
    if (drafter) |d| if (!options.serial) try stdout.print("drafted by DFlash2, its taps after layers {any}, a context of {d} rows{s}\n", .{ d.config.taps, session.flash.pair.ranks[0].cap, if (session.flash.pair.ranks[0].ring) ", a ring" else "" });
    try stdout.flush();
    session.meet() catch |err| return channel.gone(first, @errorCast(err));

    if (session.follows()) {
        const ran = session.follow() catch |err| {
            if (err == error.PeerFailed or err == error.PeerGone) return channel.gone(first, @errorCast(err));
            std.log.err("rank {d} following: {t}", .{ first, err });
            return 1;
        };
        try stdout.print("rank {d}: followed rank 0's {d} operations\n", .{ first, ran });
        return 0;
    }

    const out = try arena.alloc(i32, options.max);
    const eos: []const u32 = if (options.ends) c.eos else &.{};
    var record: Record = .{ .lines = .init(arena) };
    for (prompts, ways) |prompt, way| {
        record.lines.clearRetainingCapacity();
        if (way.cap) |cap| session.head.cap = cap;
        if (drafter != null) {
            const d = &session.flash.d;
            if (way.cap) |cap| d.cap = cap;
            if (way.confidence) |confidence| d.confidence = confidence;
            if (way.noisy) |noisy| d.noisy = noisy;
        }
        const done = try session.reply(prompt, way.settings, .{ .eos = eos, .fresh = !options.onward, .mtp = way.mtp, .chains = if (options.chains) .{ .context = &record, .chain = Record.chain } else null }, out);
        const made = done.made;
        try stdout.print("prompt: {d} tokens, {d} of them held, the first token of the reply after {d:.1} ms\nreply:", .{ prompt.len, done.cached, milliseconds(done.first_ns) });
        for (out[0..made]) |token| try stdout.print(" {d}", .{token});
        try stdout.print("\n{d} tokens", .{made});
        if (made > 1) try stdout.print(", {d:.2} a second after the first ({d:.2} ms a token); {d} rounds, {d} of {d} drafts kept", .{ @as(f64, @floatFromInt(made - 1)) * 1e3 / milliseconds(done.rest_ns), milliseconds(done.rest_ns) / @as(f64, @floatFromInt(made - 1)), done.stats.rounds, done.stats.accepted, done.stats.drafted });
        if (drafter != null and !way.mtp and !options.serial) try stdout.print("; DFlash2 cap {d}, {s} {d}", .{ session.flash.d.cap, if (session.flash.d.noisy) "noise-aware product" else "product", session.flash.d.confidence });
        try stdout.writeAll("\n");
        if (options.chains) try stdout.print("chains by {s}, temperature {d}\n", .{ if (drafter == null or way.mtp) "mtp" else "dflash2", way.settings.temperature });
        try stdout.writeAll(record.lines.written());
        try stdout.flush();
        if (Session.paired and options.alike) {
            // the reply's ids as a barrier's counter: another reply on the other rank is out of step, on both
            const reply = std.hash.Wyhash.hash(made, std.mem.sliceAsBytes(out[0..made]));
            channel.barrier(reply | 1 << 63) catch |err| {
                if (err == error.OutOfStep) std.log.err("rank {d}: the other rank made another reply", .{first});
                return channel.gone(first, err);
            };
            try stdout.print("rank {d}: the other rank made the same reply\n", .{first});
        }
    }
    if (session.leads()) {
        const told = session.end() catch |err| return channel.gone(first, @errorCast(err));
        try stdout.print("rank {d}: rank 1 followed the same {d} operations\n", .{ first, told });
    }
    return 0;
}
