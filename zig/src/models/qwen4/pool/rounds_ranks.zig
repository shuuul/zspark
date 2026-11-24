//! A round of several streams on one rank of two (spec 094 WS-03a): `rounds.zig`'s `stage`, `compute` and
//! `commit` over this rank's share, its slots and its pools of extents, with the partials of `rounds_parts.zig`.
//! Rank 0 decides and tells before it runs (`ranks_forward.zig`'s `Op.table`, `streams`, `streams_commit`);
//! rank 1 holds the same slots and the same pools, laid out alike, runs what it is told (`session/mirror.zig`)
//! and refuses what does not fit what it holds. The extents' allocator is rank 0's alone
//! (`decode/extents.zig`): a rank is told the entries of a stream's table that changed, as one rank's device is.
//!
//! A row's bits are the bits its stream has alone on the two ranks (`ranks_forward.zig`'s `launch`): each
//! rank's rows are `rounds.zig`'s, which are one rank's lone ones, and a row's partials are added by that
//! row. No drafting head, no kept state inside a round, no fork and no graph here yet (WS-03b).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const lead_file = @import("../../../ranks/lead.zig");
const attention = @import("../blocks/attention.zig");
const gdn = @import("../blocks/gdn.zig");
const model = @import("../model.zig");
const held_file = @import("../ranks_forward.zig");
const rounds = @import("rounds.zig");

const H = held_file.Held(1);
const Op = held_file.Op;

pub const Error = H.Error || Allocator.Error || error{ OutOfStep, RoundPastProjection, InvalidRound };

/// One stream's rows in a round.
pub const Part = struct { slot: usize, rows: usize };

/// What one rank holds for its rounds: the streams' slots, a pool of extents an attention layer, in this
/// rank's shape (half the KV heads, the index keys whole), and the round's tables and scratch.
pub const Desk = struct {
    /// The streams a round's frame names at most.
    pub const most = 8;

    slots: []rounds.Slot,
    /// One an attention layer of the share, `count` extents each: the same extent index in every layer, and
    /// on both ranks.
    pools: []attention.Pool,
    count: usize,
    /// [slot][capacity / 512]: the table as told, and how many of its entries stand.
    tables: [][]i32,
    placed: []usize,
    round: rounds.Round,
    scratch: rounds.Shared,
    /// One a DeltaNet layer: what its chain wrote, for a commit that keeps fewer rows than ran.
    saves: []gdn.Scratch,
    /// The round run last and not committed, and its rows' tokens.
    open: [most]Part = undefined,
    open_parts: usize = 0,
    tokens: []i32,
    capacity: usize,
    gpa: Allocator,

    /// `streams` slots of up to `capacity` positions, rounds of up to `rows` rows, over `count` extents.
    pub fn init(gpa: Allocator, m: model.Model, streams: usize, rows: usize, capacity: usize, count: usize) !Desk {
        std.debug.assert(streams >= 1 and count >= 1);
        var attending: usize = 0;
        var deltas: usize = 0;
        for (m.layers) |layer| switch (layer.block) {
            .attention => attending += 1,
            .delta_net => deltas += 1,
        };
        const pools = try gpa.alloc(attention.Pool, attending);
        errdefer gpa.free(pools);
        var pooled: usize = 0;
        errdefer for (pools[0..pooled]) |*pool| pool.deinit();
        for (pools) |*pool| {
            pool.* = try .init(m.attention.?.shape, count);
            pooled += 1;
        }
        const slots = try gpa.alloc(rounds.Slot, streams);
        errdefer gpa.free(slots);
        var opened: usize = 0;
        errdefer for (slots[0..opened]) |*slot| slot.deinit();
        for (slots) |*slot| {
            slot.* = try .open(gpa, m, pools, null, capacity);
            opened += 1;
        }
        const entries = @max(extents.extentsFor(capacity), 1);
        const tables = try gpa.alloc([]i32, streams);
        errdefer gpa.free(tables);
        var tabled: usize = 0;
        errdefer for (tables[0..tabled]) |table| gpa.free(table);
        for (tables) |*table| {
            table.* = try gpa.alloc(i32, entries);
            @memset(table.*, 0);
            tabled += 1;
        }
        const placed = try gpa.alloc(usize, streams);
        errdefer gpa.free(placed);
        @memset(placed, 0);
        var round: rounds.Round = try .init(m, streams, rows);
        errdefer round.deinit();
        var scratch: rounds.Shared = try .init(rows, m.shape.delta_net);
        errdefer scratch.deinit();
        const saves = try gpa.alloc(gdn.Scratch, deltas);
        errdefer gpa.free(saves);
        var saved: usize = 0;
        errdefer for (saves[0..saved]) |*one| one.deinit();
        for (saves) |*one| {
            one.* = try .init(rows, m.shape.delta_net);
            saved += 1;
        }
        const tokens = try gpa.alloc(i32, rows);
        return .{ .slots = slots, .pools = pools, .count = count, .tables = tables, .placed = placed, .round = round, .scratch = scratch, .saves = saves, .tokens = tokens, .capacity = capacity, .gpa = gpa };
    }

    pub fn deinit(self: *Desk) void {
        const gpa = self.gpa;
        gpa.free(self.tokens);
        for (self.saves) |*one| one.deinit();
        gpa.free(self.saves);
        self.scratch.deinit();
        self.round.deinit();
        gpa.free(self.placed);
        for (self.tables) |table| gpa.free(table);
        gpa.free(self.tables);
        for (self.slots) |*slot| slot.deinit();
        gpa.free(self.slots);
        for (self.pools) |*pool| pool.deinit();
        gpa.free(self.pools);
        self.* = undefined;
    }

    /// Entries `from ..` of slot `slot`'s table are `entries`, and the table is `from + entries.len` long:
    /// what a rank is told. An entry past the pool, or past the table, is refused with nothing changed.
    pub fn place(self: *Desk, slot: usize, from: usize, entries: []const i32) (cuda.Error || error{OutOfStep})!void {
        if (slot >= self.slots.len) return error.OutOfStep;
        const table = self.tables[slot];
        if (from > self.placed[slot] or from + entries.len > table.len) return error.OutOfStep;
        for (entries) |entry| if (entry < 0 or entry >= self.count) return error.OutOfStep;
        @memcpy(table[from..][0..entries.len], entries);
        self.placed[slot] = from + entries.len;
        if (entries.len > 0) try self.slots[slot].table.device.uploadAt(from * @sizeOf(i32), std.mem.sliceAsBytes(entries));
    }
};

/// Slot `slot`'s table is now `table` (a chain's, after rank 0's extents grew or cut it): rank 1 is told the
/// entries from the first that changed, and nothing where none did.
pub fn lay(m: H.Model, desk: *Desk, slot: usize, table: []const i32) Error!void {
    if (slot >= desk.slots.len) return error.OutOfStep;
    const known = desk.tables[slot][0..desk.placed[slot]];
    var from: usize = 0;
    while (from < known.len and from < table.len and known[from] == table[from]) from += 1;
    if (from == known.len and from == table.len) return;
    if (3 + table.len - from + 1 > lead_file.Lead.most) return error.OutOfStep;
    try held_file.told(m.lead, .table, &.{ @intCast(slot), @intCast(from), @intCast(table.len) }, table[from..]);
    try desk.place(slot, from, table[from..]);
}

/// One round: each part's rows are one stream's, at its slot's position, through every layer of the share.
/// Rank 1 is told before the launches. Every row is staged as kept; `commit` says how many are.
pub fn shared(m: H.Model, desk: *Desk, b: *H.Buffers, parts: []const Part, tokens: []const i32) Error!void {
    if (parts.len == 0 or parts.len > Desk.most or parts.len > desk.slots.len) return error.OutOfStep;
    const share = m.ranks[0];
    var words: [1 + 2 * Desk.most]u32 = undefined;
    words[0] = @intCast(parts.len);
    var segments: [Desk.most]rounds.Segment = undefined;
    var rows: usize = 0;
    for (parts, 0..) |part, at| {
        if (part.slot >= desk.slots.len or part.rows == 0 or rows + part.rows > tokens.len) return error.OutOfStep;
        for (parts[0..at]) |before| if (before.slot == part.slot) return error.OutOfStep;
        const end = desk.slots[part.slot].pos + part.rows;
        // the stream's rows lie where its table says: every extent to the window's end has been told
        if (end > desk.capacity or (desk.pools.len > 0 and extents.extentsFor(end) > desk.placed[part.slot])) return error.OutOfStep;
        words[1 + 2 * at] = @intCast(part.slot);
        words[2 + 2 * at] = @intCast(part.rows);
        segments[at] = .{ .slot = part.slot, .tokens = tokens[rows..][0..part.rows], .keep = part.rows };
        rows += part.rows;
    }
    if (rows != tokens.len or rows > desk.round.rows or rows > b.rows or rows > m.max_rows or rows > b.scoredRows()) return error.OutOfStep;
    if (1 + 1 + 2 * parts.len + rows > lead_file.Lead.most) return error.OutOfStep;
    try held_file.told(m.lead, .streams, words[0 .. 1 + 2 * parts.len], tokens);
    desk.open_parts = 0;
    const own = &b.ranks[0];
    const landing: rounds.Parts = .{ .branch = b.branch.view, .moe = b.moe_parts.view, .branch_bf16 = b.branch_bf16, .moe_sum = b.moe_sum, .gather = m.gather, .rank = m.first };
    try rounds.stage(desk.gpa, share, desk.slots, &desk.round, own, segments[0..parts.len]);
    try rounds.computeOver(share, desk.slots, &desk.scratch, own, &desk.round, segments[0..parts.len], null, .all, null, desk.saves, &landing);
    @memcpy(desk.open[0..parts.len], parts);
    @memcpy(desk.tokens[0..rows], tokens);
    desk.open_parts = parts.len;
}

/// Keeps `keeps[i]` rows of the last round's part `i`. Rank 1 is told first. A stream that keeps fewer rows
/// than it ran is replayed from what its chains saved; a segment of `gdn.wide_rows` rows or more is kept whole.
pub fn commit(m: H.Model, desk: *Desk, b: *H.Buffers, keeps: []const usize) Error!void {
    if (desk.open_parts == 0 or keeps.len != desk.open_parts) return error.OutOfStep;
    var words: [1 + Desk.most]u32 = undefined;
    words[0] = @intCast(keeps.len);
    var segments: [Desk.most]rounds.Segment = undefined;
    var rows: usize = 0;
    var short = false;
    for (desk.open[0..keeps.len], keeps, 0..) |part, keep, at| {
        if (keep == 0 or keep > part.rows) return error.OutOfStep;
        if (keep < part.rows) {
            if (part.rows >= gdn.wide_rows) return error.OutOfStep;
            short = true;
        }
        words[1 + at] = @intCast(keep);
        segments[at] = .{ .slot = part.slot, .tokens = desk.tokens[rows..][0..part.rows], .keep = keep };
        rows += part.rows;
    }
    try held_file.told(m.lead, .streams_commit, words[0 .. 1 + keeps.len], &.{});
    desk.open_parts = 0;
    const share = m.ranks[0];
    if (short) try rounds.replayKept(share, desk.slots, &desk.round, segments[0..keeps.len], desk.saves);
    try rounds.commit(share, desk.slots, &b.ranks[0], &desk.round, segments[0..keeps.len]);
}
