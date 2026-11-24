//! `ranks-check ... session`: rank 0 leads two replies and rank 1 follows (spec 050 WS-05). The first reply
//! drafts, under a constraint that allows one token, and stops as a client hanging up. The second is sampled,
//! one token a step. Both ranks' token lists are the picks rank 0 emitted; a barrier on their digest holds
//! them the same.
//!
//! Where rank 0's share came with an MTP head (spec 090; the random model's is random, `random.zig`), the head
//! then drafts: a reply its constraint forces, whose drafts are all kept, and replies set against the serial
//! ones token for token, greedy and sampled, a turn that goes on from the reply before it without a reset, and
//! one after a reply that hung. A random head agrees with its trunk by chance alone, so only the forced reply's
//! drafts are sure to be kept.
//!
//! Then what a pair keeps of the sequence it ran (spec 094): a conversation's prompts, each run from where it
//! parts from the one before (sent again, the next turn, a message edited, the turn as it was, after a reply
//! that hung, after one nobody drafted, a turn wider than a reply's window, which runs in one window as on one
//! rank, spec 110), each reply set against the one from an empty state. The barrier's
//! digest holds the positions of the states each rank is left with.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const draft_file = @import("../../../decode/draft.zig");
const masks_file = @import("../../../decode/masks.zig");
const sampling = @import("../../../decode/sampling.zig");
const channel_file = @import("../../../ranks/channel.zig");
const nccl = @import("../../../ranks/nccl.zig");
const grammar = @import("../../../text/grammar.zig");
const model = @import("../model.zig");
const held_file = @import("../ranks_forward.zig");
const pair_file = @import("../session/pair.zig");
const book_file = @import("../../../decode/book.zig");

const H = held_file.Held(1);
/// Wider than a reply's window (`draft.forward_max_rows`): a turn of more new tokens than that is one window.
pub const rows = 40;
const room = 128;
/// The only token the first reply's constraint allows.
const forced_token: u32 = 4;

const Forced = struct {
    fn start(_: *anyopaque) grammar.Error!grammar.Position {
        return @fromBackingInt(@intCast(0));
    }

    fn fill(_: *anyopaque, _: grammar.Position, mask: []u32) grammar.Error!void {
        @memset(mask, 0);
        mask[forced_token / 32] |= @as(u32, 1) << @truncate(forced_token % 32);
    }

    fn accept(_: *anyopaque, _: grammar.Position, token: u32) grammar.Error!?grammar.Position {
        if (token != forced_token) return null;
        return @fromBackingInt(@intCast(1));
    }

    fn ended(_: *anyopaque, _: grammar.Position) bool {
        return false;
    }

    fn deinit(_: *anyopaque) void {}
};

fn propose(_: *anyopaque, _: cuda.View, _: []const i32, _: usize, _: ?masks_file.Guide, out: []i32) anyerror!usize {
    const n = @min(out.len, 2);
    for (out[0..n]) |*id| id.* = forced_token;
    return n;
}

const drafter: draft_file.Drafter = .{ .context = undefined, .draft = propose };

const depth = draft_file.forward_max_rows - 1;
const made = 12;

const Told = struct { drafted: usize = 0, kept: usize = 0, rounds: usize = 0 };

/// The reply to `prompt` one row a window from an empty sequence, then drafted by the head: from an empty
/// sequence too, or where `on` from where the state stands, which `prompt` must go on from. The same tokens,
/// or what differs.
fn same(gpa: Allocator, pair: *pair_file.Pair, prompt: []const i32, settings: sampling.Sampling, on: bool, hang_at: ?usize, told: *Told) !?[]const u8 {
    // drafted first where it goes on: the serial reply resets
    var replies: [2][]i32 = undefined;
    var filled: usize = 0;
    defer for (replies[0..filled]) |one| gpa.free(one);
    for ([2]bool{ on, !on }) |drafting| {
        if (!(on and drafting)) try pair.reset();
        const before = pair.tokens.items.len;
        const done = try pair.reply(prompt, settings, &.{}, .{ .depth = if (drafting) depth else 0 }, .{}, null, made, if (drafting) hang_at else null, null);
        if (drafting) {
            if (pair.head.?.state.position.host != pair.state.pos) return "the head does not stand where the state does after its reply";
            told.drafted += done.stats.drafted;
            told.kept += done.stats.accepted;
            told.rounds += done.stats.rounds;
        } else if (done.stats.drafted != 0) return "the serial reply drafted";
        replies[filled] = try gpa.dupe(i32, pair.tokens.items[before..]);
        filled += 1;
    }
    const drafted = replies[if (on) 0 else 1];
    const serial = replies[if (on) 1 else 0];
    // a reply that hung is the serial one's first tokens
    if (drafted.len > serial.len or !std.mem.eql(i32, drafted, serial[0..drafted.len])) return "the drafted reply is not the serial one";
    if (hang_at == null and drafted.len != serial.len) return "the drafted reply is not as long as the serial one";
    return null;
}

/// The head's replies against the serial ones (the file's header). Leaves the state after a serial reply.
fn drafts(gpa: Allocator, stdout: *std.Io.Writer, pair: *pair_file.Pair, masks: *masks_file.Masks, prompt: []const i32) !?[]const u8 {
    try pair.reset();
    const start = pair.tokens.items.len;
    const forced = try pair.reply(prompt, .{ .seed = 3, .temperature = 0 }, &.{}, .{ .depth = depth }, .{}, masks, 8, null, null);
    if (forced.finish != .length or forced.tokens != 8 or forced.stats.accepted == 0) return "the head's forced reply was not 8 tokens with a draft kept";
    for (pair.tokens.items[start..]) |id| if (id != forced_token) return "a token escaped the constraint under the head";

    var told: Told = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const sampled: sampling.Sampling = .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95 };
    if (try same(gpa, pair, prompt, greedy, false, null, &told)) |text| return text;
    if (try same(gpa, pair, prompt, sampled, false, null, &told)) |text| return text;

    // a second turn: the head stands where the reply before it ended
    var turn: std.ArrayList(i32) = .empty;
    defer turn.deinit(gpa);
    for ([_]?usize{ null, 3 }) |hang_at| {
        try pair.reset();
        _ = try pair.reply(prompt, sampled, &.{}, .{ .depth = depth }, .{}, null, made, hang_at, null);
        // what the state ran, and the token after it: a reply that hung emitted fewer than it kept
        turn.clearRetainingCapacity();
        try turn.appendSlice(gpa, pair.seen.items);
        try turn.appendSlice(gpa, &.{ 7, 9 });
        if (try same(gpa, pair, turn.items, sampled, true, null, &told)) |text| return text;
    }
    if (try same(gpa, pair, prompt, greedy, false, 5, &told)) |text| return text;
    try stdout.print("rank 0: the head's forced reply {d} tokens, {d} drafts kept; 5 drafted replies are the serial ones, {d} of {d} drafts kept in {d} rounds\n", .{ forced.tokens, forced.stats.accepted, told.kept, told.drafted, told.rounds });
    return null;
}

/// The states every few tokens: a short prompt takes several, inside its windows and at their ends.
const close_marks: book_file.Options = .{ .marks = .{ .budget = 8, .stride = 16, .near = 2, .tail = 2 }, .reply_stride = 5, .reply_marks = 2 };

/// The reply to `prompt` from an empty state, drafted by the head.
fn alone(arena: Allocator, pair: *pair_file.Pair, prompt: []const i32, settings: sampling.Sampling) ![]const i32 {
    try pair.reset();
    const before = pair.tokens.items.len;
    const done = try pair.reply(prompt, settings, &.{}, .{ .depth = depth }, .{}, null, made, null, null);
    if (done.cached != 0) return error.DraftFailed;
    return arena.dupe(i32, pair.tokens.items[before..]);
}

const Step = struct {
    what: []const u8,
    prompt: []const i32,
    starts: []const usize,
    /// The reply from an empty state.
    want: []const i32,
    /// How many of the prompt's tokens the state must hold, at least and at most.
    low: usize,
    high: usize,
    hang_at: ?usize = null,
    drafted: bool = true,
    /// The windows the prompt runs in, where the step says (spec 110 WS-02).
    windows: ?usize = null,
};

/// A conversation on the states the pair keeps (the file's header), greedy and sampled.
fn turns(gpa: Allocator, stdout: *std.Io.Writer, pair: *pair_file.Pair) !?[]const u8 {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const first = [_]i32{ 1, 2, 3, 5, 211, 40, 7, 99, 150, 23, 301, 12, 64, 8, 77, 250, 31, 5, 190, 6, 44, 120, 9, 280, 17, 3, 61 };
    const starts = [_]usize{ 0, 6, 19 };
    const more = [_]i32{ 9, 301, 44, 17, 250 };
    const extra = [_]i32{ 33, 8, 160, 72, 5, 219, 14, 90, 301, 6, 47, 128, 3, 260, 55, 11, 199, 20, 81, 7 };
    const edit = 9;
    const tail = close_marks.marks.tail;
    var held: usize = 0;
    var taken: usize = 0;
    var ran: usize = 0;
    for ([_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95 } }) |settings| {
        // the replies from an empty state first: each empties what the pair holds
        const one = try alone(arena, pair, &first, settings);
        const next = try std.mem.concat(arena, i32, &.{ &first, one, &more });
        const next_starts = [_]usize{ 0, 6, 19, first.len + one.len };
        const two = try alone(arena, pair, next, settings);
        const edited = try arena.dupe(i32, next);
        edited[edit] = 319;
        const three = try alone(arena, pair, edited, settings);
        const after = try std.mem.concat(arena, i32, &.{ next, two[0..5], &more });
        const four = try alone(arena, pair, after, settings);
        // a turn of more new tokens than a reply's window: one window, as on one rank
        const longer = try std.mem.concat(arena, i32, &.{ after, four, &extra });
        const five = try alone(arena, pair, longer, settings);
        try pair.reset();
        const steps = [_]Step{
            .{ .what = "the first prompt", .prompt = &first, .starts = &starts, .want = one, .low = 0, .high = 0 },
            .{ .what = "the prompt sent again", .prompt = &first, .starts = &starts, .want = one, .low = first.len - tail, .high = first.len - tail },
            .{ .what = "the next turn", .prompt = next, .starts = &next_starts, .want = two, .low = first.len + one.len - 1, .high = first.len + one.len - 1 },
            .{ .what = "a message edited", .prompt = edited, .starts = &next_starts, .want = three, .low = 1, .high = edit - 1 },
            .{ .what = "the turn as it was", .prompt = next, .starts = &next_starts, .want = two, .low = edit - 1, .high = edit - 1 },
            .{ .what = "a reply that hung", .prompt = next, .starts = &next_starts, .want = two, .low = 1, .high = next.len - 1, .hang_at = 5 },
            .{ .what = "the turn after it", .prompt = after, .starts = &next_starts, .want = four, .low = next.len - tail, .high = next.len + 4 },
            .{ .what = "a turn of 20 new tokens", .prompt = longer, .starts = &next_starts, .want = five, .low = after.len + four.len - 1, .high = after.len + four.len - 1, .windows = 1 },
            .{ .what = "a reply nobody drafted", .prompt = &first, .starts = &starts, .want = one, .low = 1, .high = first.len - 1, .drafted = false },
            .{ .what = "the prompt after it", .prompt = &first, .starts = &starts, .want = one, .low = 0, .high = 0 },
        };
        for (steps) |step| {
            const before = pair.tokens.items.len;
            const done = try pair.reply(step.prompt, settings, &.{}, .{ .depth = if (step.drafted) depth else 0 }, .{ .starts = step.starts }, null, made, step.hang_at, null);
            const got = pair.tokens.items[before..];
            if (got.len != (step.hang_at orelse step.want.len) or !std.mem.eql(i32, got, step.want[0..got.len])) {
                std.log.err("rank 0: {s}: the reply on the states held is not the one from an empty state", .{step.what});
                return "a reply on the states held is not the one from an empty state";
            }
            if (done.cached < step.low or done.cached > step.high) {
                std.log.err("rank 0: {s}: {d} of the prompt's {d} tokens held, {d} to {d} expected", .{ step.what, done.cached, step.prompt.len, step.low, step.high });
                return "a prompt went on from another position than one rank's";
            }
            if (step.windows) |want| if (done.windows != want) {
                std.log.err("rank 0: {s}: {d} new tokens in {d} windows, {d} expected", .{ step.what, step.prompt.len - done.cached, done.windows, want });
                return "a turn ran in more windows than one rank's";
            };
            if (step.drafted and pair.head.?.state.position.host != pair.state.pos) return "the head does not stand where the state does after a turn";
            if (pair.shelf.held.items.len != pair.book.count()) return "rank 0 holds other states than its account";
            held += done.cached;
            taken += done.taken;
            ran += step.prompt.len;
        }
    }
    try stdout.print("rank 0: 20 turns on the states held are the replies from an empty state: {d} of {d} prompt tokens held, {d} states taken\n", .{ held, ran, taken });
    return null;
}

/// The positions of the states this rank holds, folded into `seed`.
fn holding(pair: *const pair_file.Pair, seed: u64) u64 {
    var digest = seed;
    for (pair.shelf.held.items) |one| digest = std.hash.Wyhash.hash(digest, std.mem.asBytes(&@as(u64, one[0].pos)));
    return digest;
}

pub fn run(gpa: Allocator, stdout: *std.Io.Writer, channel: *channel_file.Channel, comm: nccl.Comm, rank: usize, shares: [1]model.Model) !u8 {
    const pair = try pair_file.Pair.create(gpa, rank, .{ .ranks = shares, .first = rank, .max_rows = rows }, comm, channel, rows, room, false);
    defer pair.destroy();
    const prompt = [_]i32{ 1, 2, 3, 5 };
    if (rank == 0) {
        try pair.marking(close_marks);
        var held: Forced = .{};
        const words = (pair.logits.len + 31) / 32;
        const mask = try gpa.alloc(u32, words);
        defer gpa.free(mask);
        const table: grammar.Constraint.Table = .{
            .start = Forced.start,
            .fill = Forced.fill,
            .accept = Forced.accept,
            .ended = Forced.ended,
            .deinit = Forced.deinit,
        };
        var masks: masks_file.Masks = try .init(.{ .context = &held, .words = words, .table = &table }, mask, null);
        const hung = pair.reply(&prompt, .{ .seed = 3, .temperature = 0 }, &.{}, .{ .given = drafter }, .{}, &masks, 8, 4, null) catch |err| return channel.failWith(rank, err);
        if (hung.finish != .cut or hung.tokens != 4 or hung.stats.accepted == 0) {
            channel.fail("the hung reply was not 4 forced tokens with a draft kept");
            return 1;
        }
        for (pair.tokens.items) |id| {
            if (id != forced_token) {
                channel.fail("a token escaped the constraint");
                return 1;
            }
        }
        pair.reset() catch |err| return channel.failWith(rank, err);
        _ = pair.reply(&prompt, .{ .seed = 11, .temperature = 0.7, .top_k = 8 }, &.{}, .{}, .{}, null, 5, null, null) catch |err| return channel.failWith(rank, err);
        if (pair.head != null) {
            var again: masks_file.Masks = try .init(.{ .context = &held, .words = words, .table = &table }, mask, null);
            if (drafts(gpa, stdout, pair, &again, &prompt) catch |err| return channel.failWith(rank, err)) |text| {
                std.log.err("rank 0: {s}", .{text});
                channel.fail(text);
                return 1;
            }
            if (turns(gpa, stdout, pair) catch |err| return channel.failWith(rank, err)) |text| {
                std.log.err("rank 0: {s}", .{text});
                channel.fail(text);
                return 1;
            }
        } else try stdout.print("rank 0: no MTP head beside the share, nothing drafted by one\n", .{});
        pair.finish() catch |err| return channel.failWith(rank, err);
        try stdout.print("rank 0: hung reply {d} tokens, {d} drafts kept; sampled 5 more; {d} tokens in all\n", .{ hung.tokens, hung.stats.accepted, pair.tokens.items.len });
    } else {
        _ = pair.follow() catch |err| return channel.failWith(rank, err);
        try stdout.print("rank 1: followed, {d} tokens, {d} reply windows run of its own\n", .{ pair.tokens.items.len, pair.own_rounds });
    }
    try stdout.flush();
    channel.barrier(holding(pair, std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(pair.tokens.items)))) catch |err| {
        if (err == error.PeerFailed) std.log.err("rank {d}: the other rank failed: {s}", .{ rank, channel.said() });
        return 1;
    };
    try stdout.print("rank {d}: the other rank holds the same tokens, and {d} states at the same positions\n", .{ rank, pair.shelf.held.items.len });
    return 0;
}
