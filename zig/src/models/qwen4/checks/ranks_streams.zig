//! `ranks-check ... streams`: four streams packed into rounds (`pool/rounds_ranks.zig`), every row against
//! the row its stream has alone on the two ranks, in bits (spec 094 WS-03a; spec 050 WS-08 held DeltaNet
//! layers alone to their tokens). Rank 0 runs the extents, samples and tells; rank 1 runs what it is told.
//!
//! The streams are unlike on purpose. Two prompts pass 512 positions, so their tables name a second extent
//! taken after the other streams' first (no table is the identity) and, under `kv=`, their rows read keys the
//! indexer chose. One prompt ends in the first round and decodes beside the others' chunks. Stream 0's chunk
//! is `gdn.wide_rows` or more, the others' short. Each stream once runs three rows and keeps one, as a
//! refused draft would leave it.
//!
//! A window or round of `split_at` rows or more splits its glue by rows over the ranks (spec 149 WS-02): the
//! first rounds, prompts' chunks with a reply beside them from the second on, split; the later ones, fewer rows,
//! keep the whole glue, and so do the decoded steps alone; the lone windows of 64 rows split as a pair's do.

const std = @import("std");
const Allocator = std.mem.Allocator;
const extents = @import("../../../decode/extents.zig");
const sampling = @import("../../../decode/sampling.zig");
const channel_file = @import("../../../ranks/channel.zig");
const nccl = @import("../../../ranks/nccl.zig");
const model = @import("../model.zig");
const rounds = @import("../pool/rounds.zig");
const rounds_ranks = @import("../pool/rounds_ranks.zig");
const held_file = @import("../ranks_forward.zig");
const pair_file = @import("../session/pair.zig");

const H = held_file.Held(1);
const Desk = rounds_ranks.Desk;

pub const streams = 4;
/// Tokens a stream decodes after its prompt.
pub const made = 6;
/// A round's rows at most, and a lone window's.
pub const rows = 64;
/// Positions a stream may hold: three extents.
pub const room = 1536;
/// The pool's extents: fewer than the streams' rooms together, more than they reach.
pub const count = extents.lowerBound(room, streams) + 1;
const lengths = [streams]usize{ 600, 45, 5, 530 };
/// A prompt's rows a round: stream 0's take the wide kernels.
const chunks = [streams]usize{ 40, 8, 8, 8 };
/// The decoded row that runs with two rows after it, of which none is kept.
const drafted_at = 2;
/// The rows from which the glue splits in this check, both ranks alike (`ranks_forward.split_rows`, 256 served).
pub const split_at = 32;

/// What a run left: each stream's tokens, and a hash of every kept row's logits, in the stream's order.
pub const Run = struct {
    tokens: [streams][made]i32 = undefined,
    logits: [streams]std.ArrayList(u64) = @splat(.empty),

    pub const Parting = struct { stream: usize, row: ?usize };

    pub fn deinit(self: *Run, gpa: Allocator) void {
        for (&self.logits) |*one| one.deinit(gpa);
    }

    /// Null, or where the two part: a row's logits, or with no row the tokens or the rows' number.
    pub fn differs(self: *const Run, other: *const Run) ?Parting {
        for (0..streams) |stream| {
            const a = self.logits[stream].items;
            const b = other.logits[stream].items;
            if (a.len != b.len) return .{ .stream = stream, .row = null };
            for (a, b, 0..) |left, right, row| if (left != right) return .{ .stream = stream, .row = row };
            if (!std.mem.eql(i32, &self.tokens[stream], &other.tokens[stream])) return .{ .stream = stream, .row = null };
        }
        return null;
    }
};

/// The four prompts of a seed, ids every Flash Next vocabulary has. The arena's.
pub fn prompts(arena: Allocator, seed: u64) ![streams][]const i32 {
    var prng: std.Random.DefaultPrng = .init(seed +% 94);
    var out: [streams][]const i32 = undefined;
    for (&out, lengths) |*prompt, n| {
        const ids = try arena.alloc(i32, n);
        for (ids) |*id| id.* = prng.random().intRangeLessThan(i32, 1, 300);
        prompt.* = ids;
    }
    return out;
}

/// What rank 0 reads a row of logits into and how, and where it lists the tokens it told.
pub const Taking = struct {
    row: []u16,
    told: *std.ArrayList(i32),
    /// `H.logitsRow`, both ranks' halves; a test of one share alone reads that half.
    read: *const fn (H.Buffers, usize, []u16) H.Error!void = H.logitsRow,
    gpa: Allocator,

    fn take(self: Taking, m: H.Model, b: H.Buffers, row: usize, into: *std.ArrayList(u64), pick: bool) !i32 {
        try self.read(b, row, self.row);
        try into.append(self.gpa, std.hash.Wyhash.hash(94, std.mem.sliceAsBytes(self.row)));
        if (!pick) return 0;
        const token = try sampling.argmax(self.row);
        try held_file.told(m.lead, .pick, &.{}, &.{token});
        try self.told.append(self.gpa, token);
        return token;
    }
};

/// Each stream alone, from an empty state: its prompt in windows of `rows`, then a token a step.
pub fn alone(m: H.Model, state: *H.State, b: *H.Buffers, taking: Taking, asked: [streams][]const i32, out: *Run) !void {
    for (asked, 0..) |prompt, stream| {
        try held_file.told(m.lead, .reset, &.{}, &.{});
        for (&state.ranks) |*one| try one.reset();
        state.pos = 0;
        var at: usize = 0;
        var token: i32 = 0;
        while (at < prompt.len) {
            const window = prompt[at..@min(at + rows, prompt.len)];
            try H.forward(m, state, b, window);
            for (0..window.len) |row| token = try taking.take(m, b.*, row, &out.logits[stream], at + row + 1 == prompt.len);
            try H.commit(m, state, b.*, window, window.len);
            at += window.len;
        }
        out.tokens[stream][0] = token;
        for (1..made) |step| {
            const fed = [1]i32{token};
            try H.forward(m, state, b, &fed);
            token = try taking.take(m, b.*, 0, &out.logits[stream], true);
            try H.commit(m, state, b.*, &fed, 1);
            out.tokens[stream][step] = token;
        }
    }
}

/// The same streams in rounds, from empty slots. `pool` is rank 0's: each round reserves its streams'
/// extents, and the tables that changed are laid (and told) before the round.
pub fn together(gpa: Allocator, m: H.Model, desk: *Desk, b: *H.Buffers, pool: *extents.Pool, taking: Taking, asked: [streams][]const i32, out: *Run) !void {
    var chains: [streams]*extents.Chain = undefined;
    for (&chains) |*chain| chain.* = try pool.open();
    var fed: [streams]i32 = undefined;
    // rows of the prompt run, then tokens decoded
    var at: [streams]usize = @splat(0);
    var done: [streams]usize = @splat(0);
    const window = try gpa.alloc(i32, rows);
    defer gpa.free(window);
    while (true) {
        var parts: [streams]rounds_ranks.Part = undefined;
        var keeps: [streams]usize = undefined;
        var live: [streams]*extents.Chain = undefined;
        var uptos: [streams]usize = undefined;
        var n: usize = 0;
        var used: usize = 0;
        for (asked, 0..) |prompt, stream| {
            if (done[stream] == made) continue;
            var part: []i32 = undefined;
            if (at[stream] < prompt.len) {
                const take = @min(prompt.len - at[stream], chunks[stream]);
                part = window[used..][0..take];
                @memcpy(part, prompt[at[stream]..][0..take]);
                keeps[n] = take;
            } else {
                // a decoded token, once with two rows after it that are not kept
                const extra: usize = if (done[stream] == drafted_at) 2 else 0;
                part = window[used..][0 .. 1 + extra];
                part[0] = fed[stream];
                for (part[1..], 1..) |*id, i| id.* = @mod(fed[stream] + @as(i32, @intCast(i * 7)), 300);
                keeps[n] = 1;
            }
            parts[n] = .{ .slot = stream, .rows = part.len };
            live[n] = chains[stream];
            uptos[n] = desk.slots[stream].pos + part.len;
            used += part.len;
            n += 1;
        }
        if (n == 0) break;
        try rounds.reserve(pool, live[0..n], uptos[0..n]);
        for (parts[0..n]) |part| try rounds_ranks.lay(m, desk, part.slot, chains[part.slot].table.items);
        try rounds_ranks.shared(m, desk, b, parts[0..n], window[0..used]);
        var row: usize = 0;
        for (parts[0..n], keeps[0..n]) |part, keep| {
            const stream = part.slot;
            for (0..keep) |i| {
                const last = at[stream] + i + 1 >= asked[stream].len;
                const token = try taking.take(m, b.*, row + i, &out.logits[stream], last);
                if (last) {
                    fed[stream] = token;
                    out.tokens[stream][done[stream]] = token;
                    done[stream] += 1;
                }
            }
            pool.append(chains[stream], window[row..][0..keep]);
            at[stream] += keep;
            row += part.rows;
        }
        try rounds_ranks.commit(m, desk, b, keeps[0..n]);
    }
    try pool.check(gpa);
}

pub fn run(gpa: Allocator, arena: Allocator, stdout: *std.Io.Writer, channel: *channel_file.Channel, comm: nccl.Comm, rank: usize, seed: u64, shares: [1]model.Model) !u8 {
    const saved = held_file.split_rows;
    defer held_file.split_rows = saved;
    held_file.split_rows = split_at;
    const pair = try pair_file.Pair.createScoring(gpa, rank, .{ .ranks = shares, .first = rank, .max_rows = rows }, comm, channel, rows, rows, room, true);
    defer pair.destroy();
    var desk: Desk = try .init(gpa, pair.m.ranks[0], streams, rows, room, count);
    defer desk.deinit();
    pair.desk = &desk;

    if (rank == 0) {
        const asked = try prompts(arena, seed);
        const taking: Taking = .{ .row = pair.logits, .told = &pair.tokens, .gpa = gpa };
        var lone: Run = .{};
        defer lone.deinit(gpa);
        alone(pair.m, &pair.state, &pair.b, taking, asked, &lone) catch |err| return channel.failWith(rank, err);
        var pool: extents.Pool = try .init(gpa, .{ .extents = count, .capacity = room, .slots = streams });
        defer pool.deinit();
        var packed_run: Run = .{};
        defer packed_run.deinit(gpa);
        together(gpa, pair.m, &desk, &pair.b, &pool, taking, asked, &packed_run) catch |err| return channel.failWith(rank, err);
        if (packed_run.differs(&lone)) |where| {
            std.log.err("rank 0: stream {d} among the others parts from the stream alone at row {?d} (null: its tokens)", .{ where.stream, where.row });
            channel.fail("a stream among the others is not the stream alone");
            return 1;
        }
        pair.finish() catch |err| return channel.failWith(rank, err);
        var kept: usize = 0;
        for (lone.logits) |one| kept += one.items.len;
        try stdout.print("rank 0: four streams in rounds, each of {d} rows the row its stream has alone, in bits, the glue split from {d} rows; tables {any} {any} {any} {any}\n", .{ kept, split_at, desk.tables[0][0..desk.placed[0]], desk.tables[1][0..desk.placed[1]], desk.tables[2][0..desk.placed[2]], desk.tables[3][0..desk.placed[3]] });
    } else {
        const ran = pair.follow() catch |err| return channel.failWith(rank, err);
        try stdout.print("rank 1: followed the four streams, {d} operations; tables {any} {any} {any} {any}\n", .{ ran, desk.tables[0][0..desk.placed[0]], desk.tables[1][0..desk.placed[1]], desk.tables[2][0..desk.placed[2]], desk.tables[3][0..desk.placed[3]] });
    }
    try stdout.flush();
    // the tokens rank 0 chose, and where this rank's streams stand
    var hash = std.hash.Wyhash.init(1);
    hash.update(std.mem.sliceAsBytes(pair.tokens.items));
    for (desk.slots, desk.tables, desk.placed) |slot, table, placed| {
        hash.update(std.mem.asBytes(&slot.pos));
        hash.update(std.mem.sliceAsBytes(table[0..placed]));
    }
    channel.barrier(hash.final()) catch |err| {
        if (err == error.PeerFailed) std.log.err("rank {d}: the other rank failed: {s}", .{ rank, channel.said() });
        return 1;
    };
    return 0;
}
