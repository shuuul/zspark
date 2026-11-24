//! `zspark generate DIR N rank=R at=ADDR:PORT [bf16|fp8|fp8-nvfp4] [eager] [serial] [depth=D] [temperature=X]
//! [seed=N] ID..`: Flash Next a rank a Spark (spec 050 WS-05). Rank 0 samples and rank 1 follows
//! (`session/pair.zig`). The MTP head is rank 0's, whole beside its share, or half a rank (spec 090), and
//! drafts unless `serial`; the tokens are the same either way. The attention caches are bf16 unless a storage
//! is named, as one rank's `generate` (spec 094); the head's own stay bf16. The ranks' token lists are held
//! alike at the end, by a barrier on their digest.

const std = @import("std");
const cli = @import("../../../cli.zig");
const sampling = @import("../../../decode/sampling.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const qwen4_attention = @import("../blocks/attention.zig");
const qwen4_layout = @import("../layout.zig");
const qwen4_model = @import("../model.zig");
const held_file = @import("../ranks_forward.zig");
const pair_file = @import("../session/pair.zig");
const common = @import("common.zig");

const H = held_file.Held(1);
const window = 16;
/// The caches hold the prompt and the reply together.
const context = 4096;

pub fn generate(init: std.process.Init, stdout: *std.Io.Writer, opened: *common.Opened, place: join_file.Place, count: usize, words: []const []const u8) !u8 {
    if (!nccl.linked) {
        std.log.err("generate: this build has no NCCL; build with -Dnccl=PREFIX (/usr for the apt packages)", .{});
        return 1;
    }
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    var eager = false;
    var serial = false;
    var depth: usize = common.draft_depth;
    var temperature: f64 = 0;
    var seed: u64 = 1;
    var named: ?qwen4_attention.Storage = null;
    var listed: std.ArrayList([]const u8) = .empty;
    for (words) |word| {
        if (qwen4_attention.Storage.parse(word)) |storage| {
            named = storage;
        } else if (std.mem.eql(u8, word, "eager")) {
            eager = true;
        } else if (std.mem.eql(u8, word, "serial")) {
            serial = true;
        } else if (cli.value(word, "depth=")) |text| {
            depth = std.fmt.parseInt(usize, text, 10) catch return error.InvalidArgument;
            if (depth >= window) return error.InvalidArgument;
        } else if (cli.value(word, "temperature=")) |text| {
            temperature = std.fmt.parseFloat(f64, text) catch return error.InvalidArgument;
        } else if (cli.value(word, "seed=")) |text| {
            seed = std.fmt.parseInt(u64, text, 10) catch return error.InvalidArgument;
        } else try listed.append(arena, word);
    }
    const prompt = try cli.ids(arena, listed.items);
    if (count == 0 or prompt.len == 0 or prompt.len + count > context) return error.InvalidArgument;
    var settings: sampling.Sampling = .{ .seed = seed, .temperature = temperature };
    settings.validate() catch |err| {
        std.log.err("sampling: {t}", .{err});
        return 2;
    };
    const rank = place.rank.?;
    const config = opened.config;

    const drafting = common.headOnRank(rank, !serial and depth > 0) orelse return 2;
    // what changes the reply: alike on both ranks, or neither starts. Who drafts changes no token; a split
    // head is computed by both.
    const caches = qwen4_layout.storageFor(config, 1, 2, named);
    const options = try std.fmt.allocPrint(arena, "generate {d} {s} t={d} seed={d} prompt {x}{s}{s}", .{ count, if (eager) "eager" else "graphs", temperature, seed, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(prompt)), common.cachesAlike(caches), common.headAlike(!serial and depth > 0) });
    var joined: join_file.Joined = undefined;
    try joined.open(io, gpa, place, try join_file.headers(arena, &opened.checkpoint.?), options);
    defer joined.close();
    const channel = &joined.channel;

    const load_start: std.Io.Timestamp = .now(io, .awake);
    const built: H.Model = .{
        .ranks = .{try qwen4_model.Model.loadWith(gpa, config, opened.layout, .{ .layers = config.layers, .max_rows = window, .mtp = drafting, .draft_vocab = if (drafting) try qwen4_model.DraftHead.defaultIds(arena) else null, .share = .of(@intCast(rank)), .head_split = qwen4_model.ranks_head == .split, .kv = caches.trunk, .head_kv = caches.head })},
        .first = rank,
        .max_rows = window,
    };
    const pair = pair_file.Pair.create(gpa, rank, built, joined.comm, channel, window, context, eager) catch |err| return channel.failWith(rank, err);
    defer pair.destroy();
    channel.barrier(1) catch |err| return channel.gone(rank, err);
    const load_s = @as(f64, @floatFromInt(load_start.untilNow(io, .awake).toNanoseconds())) / 1e9;
    try stdout.print("rank {d}: loaded its share in {d:.1} s\n", .{ rank, load_s });
    try stdout.flush();

    if (rank != 0) {
        const ran = pair.follow() catch |err| return channel.failWith(rank, err);
        try stdout.print("rank {d}: followed rank 0's {d} operations, {d} tokens, {d} reply windows run of its own\n", .{ rank, ran, pair.tokens.items.len, pair.own_rounds });
    } else {
        const start: std.Io.Timestamp = .now(io, .awake);
        const done = pair.reply(prompt, settings, &.{}, .{ .depth = if (pair.head == null) 0 else depth }, .{}, null, count, null, null) catch |err| return channel.failWith(rank, err);
        const seconds = @as(f64, @floatFromInt(start.untilNow(io, .awake).toNanoseconds())) / 1e9;
        const made = pair.tokens.items;
        try stdout.print("{s}:", .{if (settings.greedy()) "greedy" else "sampled"});
        for (made) |token| try stdout.print(" {d}", .{token});
        var drafts_buffer: [96]u8 = undefined;
        const drafts_text = if (pair.head == null) "serial" else try std.fmt.bufPrint(&drafts_buffer, "{d} rounds, {d} of {d} drafts kept", .{ done.stats.rounds, done.stats.accepted, done.stats.drafted });
        try stdout.print("\n{d} tokens in {d:.2} s: {d:.1} tok/s (two ranks, {s} caches, {s}, {s})\n", .{ done.tokens, seconds, @as(f64, @floatFromInt(done.tokens)) / seconds, caches.trunk.name(), drafts_text, if (eager) "one launch a kernel" else "a captured graph a step" });
        pair.finish() catch |err| return channel.failWith(rank, err);
    }
    try stdout.flush();
    channel.barrier(std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(pair.tokens.items))) catch |err| return channel.gone(rank, err);
    try stdout.print("rank {d}: the other rank holds the same reply\n", .{rank});
    return 0;
}
