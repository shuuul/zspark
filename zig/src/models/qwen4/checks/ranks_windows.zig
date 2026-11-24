//! `ranks-check ... windows`: a prompt's windows on two ranks as wide as one Spark's (spec 111). Both ranks
//! run one prompt of more rows than a window three ways, in lockstep over NCCL with nothing told: windows of
//! `rows` (2,048, `session.promptRows`, a pair's unless asked), of `narrow` (512, two ranks' until spec 111), and serial
//! steps. Every row's streams and this rank's half of its logits, and after the prompt every DeltaNet state,
//! conv window and attention cache's committed rows, are the same bits on the three (the row rule): the two
//! window widths in the command's form (the card's, quantized), the serial steps against windows of `rows` in one
//! form (bf16, spec 143 WS-06). The
//! windows' row counts take the experts' wide and tiled paths (`experts.Pass`) that narrower ones do not.
//!
//! Then the frames that carry such windows, in the bf16 form: rank 0 leads replies through a pair of `rows` rows and rank 1
//! follows. The prompt from an empty state runs in two windows, the second turn's new tokens, more than
//! `Rounds.kept_most` was before spec 111 (933), in one, and each drafted reply is the undrafted one.

const std = @import("std");
const units = @import("../../../cuda/units.zig");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const sampling = @import("../../../decode/sampling.zig");
const channel_file = @import("../../../ranks/channel.zig");
const nccl = @import("../../../ranks/nccl.zig");
const model = @import("../model.zig");
const forward_file = @import("../forward.zig");
const held_file = @import("../ranks_forward.zig");
const pair_file = @import("../session/pair.zig");

const H = held_file.Held(1);
/// The windows' rows, and the pair's.
pub const rows = 2048;
const narrow = 512;
/// The prompt: a window of `rows` and a few rows after it.
const prompt_len = rows + 37;
/// A turn's new tokens: more than a pair's last window kept before spec 111.
const turn_len = 1100;
const made = 12;
const room = 4096;

/// What one way leaves: a digest a row of its streams and of its logits, and of the state after the prompt.
const Way = struct {
    streams: []u64,
    logits: []u64,
    state: u64,
};

/// The prompt in windows of `width` rows from an empty state, every row scored.
fn play(gpa: Allocator, arena: Allocator, pair: *pair_file.Pair, prompt: []const i32, width: usize) !Way {
    var state: H.State = try .init(gpa, pair.m, rows, room);
    defer state.deinit();
    // every row scored: the pair's own buffers score 16 rows, as one rank's session's
    var b: H.Buffers = try .init(gpa, pair.m, rows, room);
    defer b.deinit();
    const way: Way = .{ .streams = try arena.alloc(u64, prompt.len), .logits = try arena.alloc(u64, prompt.len), .state = 0 };
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    var at: usize = 0;
    while (at < prompt.len) {
        const ids = prompt[at..@min(at + width, prompt.len)];
        try H.forward(pair.m, &state, &b, ids);
        const own = b.ranks[0];
        try rowDigests(gpa, &bytes, own.h.rows(ids.len), ids.len, way.streams[at..]);
        try rowDigests(gpa, &bytes, own.logits.rows(ids.len), ids.len, way.logits[at..]);
        try H.commit(pair.m, &state, b, ids, ids.len);
        at += ids.len;
    }
    var hasher = std.hash.Wyhash.init(state.pos);
    const sequence = state.ranks[0];
    for (sequence.recurrent, sequence.current) |sides, side| try mix(gpa, &bytes, &hasher, sides[side].view);
    for (sequence.conv) |window| try mix(gpa, &bytes, &hasher, window.view);
    for (sequence.caches) |cache| {
        try mix(gpa, &bytes, &hasher, cache.k.rows(at));
        try mix(gpa, &bytes, &hasher, cache.v.rows(at));
        if (cache.k_scale) |scale| try mix(gpa, &bytes, &hasher, scale.rows(at));
        if (cache.v_scale) |scale| try mix(gpa, &bytes, &hasher, scale.rows(at));
        try mix(gpa, &bytes, &hasher, cache.index_keys.rows(at));
    }
    var out = way;
    out.state = hasher.final();
    return out;
}

fn rowDigests(gpa: Allocator, bytes: *std.ArrayList(u8), view: cuda.View, count: usize, out: []u64) !void {
    try bytes.resize(gpa, view.count() * view.dtype.size());
    try cuda.download(view, bytes.items);
    const each = bytes.items.len / count;
    for (out[0..count], 0..) |*digest, row| digest.* = std.hash.Wyhash.hash(0, bytes.items[row * each ..][0..each]);
}

fn mix(gpa: Allocator, bytes: *std.ArrayList(u8), hasher: *std.hash.Wyhash, view: cuda.View) !void {
    try bytes.resize(gpa, view.count() * view.dtype.size());
    try cuda.download(view, bytes.items);
    hasher.update(bytes.items);
}

/// Where `got` parts from `want`, or null.
fn parts(arena: Allocator, want: Way, got: Way, name: []const u8) !?[]const u8 {
    for (want.streams, got.streams, 0..) |a, b, row| if (a != b) return try std.fmt.allocPrint(arena, "{s}: row {d}'s streams differ from windows of {d}", .{ name, row, rows });
    for (want.logits, got.logits, 0..) |a, b, row| if (a != b) return try std.fmt.allocPrint(arena, "{s}: row {d}'s logits differ from windows of {d}", .{ name, row, rows });
    if (want.state != got.state) return try std.fmt.allocPrint(arena, "{s}: the state after the prompt differs from windows of {d}", .{ name, rows });
    return null;
}

/// Rank 0: the replies through the pair (the file's header). Null, or what failed.
fn replies(gpa: Allocator, stdout: *std.Io.Writer, pair: *pair_file.Pair, prompt: []const i32, more: []const i32) !?[]const u8 {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const depth: usize = if (pair.head != null) 4 else 0;
    var lines: [4]usize = undefined;
    // the reply from an empty state, undrafted, then drafted
    var first: [2][]const i32 = undefined;
    for (&first, [_]usize{ 0, depth }, 0..) |*one, drafts, i| {
        try pair.reset();
        const before = pair.tokens.items.len;
        const done = try pair.reply(prompt, greedy, &.{}, .{ .depth = drafts }, .{}, null, made, null, null);
        if (done.cached != 0 or done.windows != 2) {
            std.log.err("rank 0: a prompt of {d} tokens from an empty state: {d} held, {d} windows; 0 and 2 expected", .{ prompt.len, done.cached, done.windows });
            return "a prompt of more rows than a window did not run in two windows";
        }
        one.* = try arena.dupe(i32, pair.tokens.items[before..]);
        lines[i] = done.windows;
    }
    if (!std.mem.eql(i32, first[0], first[1])) return "the drafted reply to the prompt is not the undrafted one";
    // the next turn on the state the drafted reply left, and from an empty state undrafted
    const next = try std.mem.concat(arena, i32, &.{ prompt, first[1], more });
    const on_before = pair.tokens.items.len;
    const on = try pair.reply(next, greedy, &.{}, .{ .depth = depth }, .{ .starts = &.{ 0, prompt.len + first[1].len } }, null, made, null, null);
    const turned = try arena.dupe(i32, pair.tokens.items[on_before..]);
    if (on.windows != 1 or next.len - on.cached < more.len) {
        std.log.err("rank 0: a turn of {d} new tokens: {d} held, in {d} windows; one expected", .{ more.len, on.cached, on.windows });
        return "a turn's new tokens did not run in one window";
    }
    lines[2] = on.windows;
    try pair.reset();
    const alone_before = pair.tokens.items.len;
    const alone = try pair.reply(next, greedy, &.{}, .{}, .{}, null, made, null, null);
    lines[3] = alone.windows;
    if (!std.mem.eql(i32, turned, pair.tokens.items[alone_before..])) return "the turn's reply on the state held is not the one from an empty state";
    try stdout.print("rank 0: a prompt of {d} tokens in {d} windows, drafted the undrafted reply; a turn of {d} new tokens in {d} window ({d} held), the reply from an empty state's ({d} windows)\n", .{ prompt.len, lines[1], next.len - on.cached, lines[2], on.cached, lines[3] });
    return null;
}

pub fn run(gpa: Allocator, arena: Allocator, stdout: *std.Io.Writer, channel: *channel_file.Channel, comm: nccl.Comm, rank: usize, seed: u64, vocab: usize, shares: [1]model.Model) !u8 {
    const pair = try pair_file.Pair.create(gpa, rank, .{ .ranks = shares, .first = rank, .max_rows = rows }, comm, channel, rows, room, false);
    defer pair.destroy();
    var prng: std.Random.DefaultPrng = .init(seed +% 111);
    const prompt = try arena.alloc(i32, prompt_len);
    for (prompt) |*id| id.* = prng.random().intRangeLessThan(i32, 0, @intCast(vocab));
    const more = try arena.alloc(i32, turn_len);
    for (more) |*id| id.* = prng.random().intRangeLessThan(i32, 0, @intCast(vocab));

    // the three ways in lockstep: rank 0 tells nothing, rank 1 runs the same calls
    pair.lead.muted = true;
    const started = cuda.monotonic();
    const wide = play(gpa, arena, pair, prompt, rows) catch |err| return channel.failWith(rank, err);
    const narrowed = play(gpa, arena, pair, prompt, narrow) catch |err| return channel.failWith(rank, err);
    // a serial step is bf16 and a window of `rows` the card's form (spec 143 WS-06): they meet in one form, as
    // `forward_checkpoint_test`'s first layers do
    const bound = forward_file.bf16_rows;
    forward_file.bf16_rows = std.math.maxInt(usize);
    const one_form = play(gpa, arena, pair, prompt, rows) catch |err| return channel.failWith(rank, err);
    const serial = play(gpa, arena, pair, prompt, 1) catch |err| return channel.failWith(rank, err);
    forward_file.bf16_rows = bound;
    pair.lead.muted = false;
    var failure = try parts(arena, wide, narrowed, "windows of 512");
    if (failure == null) failure = try parts(arena, one_form, serial, "serial steps");
    if (failure) |text| {
        std.log.err("rank {d}: {s}", .{ rank, text });
        channel.fail(text);
        return 1;
    }
    try stdout.print("rank {d}: {d} rows in windows of {d}, of {d} and serial steps, every row's streams and logits and the state the same bits ({d:.1} s)\n", .{ rank, prompt.len, rows, narrow, units.seconds(cuda.monotonic() - started) });
    try stdout.flush();
    // a rank's state is its own half: the barrier holds the position alike
    channel.barrier(prompt.len) catch |err| return channel.gone(rank, err);

    // a held state's rows ran in other windows than an empty state's, so in the card's form not as its rows
    // (the 37 after the first window are under the bound): the two replies meet in one form, bf16 (spec 143 WS-08)
    forward_file.bf16_rows = std.math.maxInt(usize);
    defer forward_file.bf16_rows = bound;
    if (rank == 0) {
        if (replies(gpa, stdout, pair, prompt, more) catch |err| return channel.failWith(rank, err)) |text| {
            std.log.err("rank 0: {s}", .{text});
            channel.fail(text);
            return 1;
        }
        pair.finish() catch |err| return channel.failWith(rank, err);
    } else {
        _ = pair.follow() catch |err| return channel.failWith(rank, err);
        try stdout.print("rank 1: followed, {d} tokens\n", .{pair.tokens.items.len});
    }
    try stdout.flush();
    channel.barrier(std.hash.Wyhash.hash(2, std.mem.sliceAsBytes(pair.tokens.items))) catch |err| return channel.gone(rank, err);
    try stdout.print("rank {d}: the other rank holds the same {d} tokens\n", .{ rank, pair.tokens.items.len });
    return 0;
}
