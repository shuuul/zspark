//! `ranks-check ... hall`: the pool's engine on two ranks (`pool/hall_ranks.zig`, spec 094 WS-03b). Rank 0's
//! engine serves `ranks_streams.zig`'s four prompts, each alone and then the four together, and tells; rank 1's
//! engine follows. Each stream together gives the tokens it gives alone and leaves its slot, on rank 0, as it
//! leaves it alone, bit for bit; drafted by the MTP head where the shares hold it (whole on rank 0, or split),
//! with as many drafts and as many kept where the head pays no costs (`mtp_rule.Paying`: a round's price is its
//! seats', so a stream among others may draft otherwise, its tokens the same) and none was cut by the round's
//! width (`pool/hall_depth.zig`: four together on two ranks chain at most 4). The four together go on from the
//! states their runs alone kept (`cached`, as `begin` gives it), forked on both ranks. `hall_ranks_test.zig`
//! holds one share alone in a process to the same.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const sampling = @import("../../../decode/sampling.zig");
const channel_file = @import("../../../ranks/channel.zig");
const lead_file = @import("../../../ranks/lead.zig");
const nccl = @import("../../../ranks/nccl.zig");
const request_file = @import("../../../server/request.zig");
const attention = @import("../blocks/attention.zig");
const hall_file = @import("../pool/hall.zig");
const hall_engine = @import("../pool/hall_engine.zig");
const hall_ranks = @import("../pool/hall_ranks.zig");
const model = @import("../model.zig");
const forward = @import("../forward.zig");
const mtp = @import("../mtp.zig");
const mtp_ranks = @import("../mtp_ranks.zig");
const rounds = @import("../pool/rounds.zig");
const held_file = @import("../ranks_forward.zig");
const streams_file = @import("ranks_streams.zig");
const rig_file = @import("../dflash2/rig.zig");
const Engine = hall_engine.Engine;

const H = held_file.Held(1);
const streams = streams_file.streams;
pub const rows = streams_file.rows;
const room = streams_file.room;
/// Tokens a stream's reply has.
pub const made = 9;

pub const greedy: sampling.Sampling = .{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };

/// What a stream left: its tokens, a hash of its slot's state as its reply ended, its drafts, and the rows
/// of its prompt a kept state gave it.
pub const Left = struct { tokens: [made]i32 = @splat(0), state: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, cached: usize = 0 };

fn hashBuffer(gpa: Allocator, hasher: *std.hash.Wyhash, buffer: cuda.Buffer, from: usize, count: usize) !void {
    const width = buffer.bytes() / @as(usize, @intCast(buffer.view.size[0]));
    const bytes = try gpa.alloc(u8, width * count);
    defer gpa.free(bytes);
    try buffer.downloadAt(from * width, bytes);
    hasher.update(bytes);
}

/// Slot `index`'s state: its committed DeltaNet side, conv windows and PLE tails, every cache row it holds
/// where its table puts it, and the pooled keys.
pub fn slotHash(engine: *Engine, index: usize) !u64 {
    const gpa = engine.gpa;
    try cuda.synchronize();
    var hasher: std.hash.Wyhash = .init(94);
    const slot = engine.slots[index];
    // the state at `pos`, the rows a deferred window left behind replayed (`rounds.committed`, spec 144 WS-09)
    for (slot.recurrent, 0..) |pair, linear| {
        var at = try forward.Snapshot.likeOf(pair[0]);
        defer at.deinit();
        try rounds.committed(slot, linear, at);
        try hashBuffer(gpa, &hasher, at, 0, @intCast(at.view.size[0]));
    }
    for (slot.conv) |window| try hashBuffer(gpa, &hasher, window, 0, @intCast(window.view.size[0]));
    for (slot.ple) |branch| try hashBuffer(gpa, &hasher, branch.tail, 0, @intCast(branch.tail.view.size[0]));
    const table = engine.seats[index].?.chain.table.items;
    for (slot.caches) |cache| {
        var at: usize = 0;
        while (at < slot.pos) : (at += attention.chunk) {
            const n = @min(attention.chunk, slot.pos - at);
            const place = @as(usize, @intCast(table[at / attention.chunk])) * attention.chunk;
            try hashBuffer(gpa, &hasher, cache.k, place, n);
            try hashBuffer(gpa, &hasher, cache.v, place, n);
            try hashBuffer(gpa, &hasher, cache.index_keys, place, n);
            if (cache.k_scale) |scale| try hashBuffer(gpa, &hasher, scale, place, n);
            if (cache.v_scale) |scale| try hashBuffer(gpa, &hasher, scale, place, n);
        }
        const blocks = slot.pos / engine.m.attention.?.shape.ratio;
        if (blocks > 0) try hashBuffer(gpa, &hasher, cache.pooled, 0, blocks);
    }
    hasher.update(std.mem.asBytes(&slot.pos));
    return hasher.final();
}

const Collect = struct {
    gpa: Allocator,
    lists: *[streams]std.ArrayList(i32),
    ended: *[streams]bool,
    out: *[streams]Left,

    fn tail(ctx: *anyopaque, slot: usize, said: @import("../../../server/scheduler.zig").Tail) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.out[slot].drafted = said.drafted;
        self.out[slot].accepted = said.accepted;
    }

    fn emit(ctx: *anyopaque, slot: usize, tokens: []const i32) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.lists[slot].appendSlice(self.gpa, tokens) catch @panic("out of memory");
    }

    fn end(ctx: *anyopaque, slot: usize, _: request_file.Finish) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
    }

    fn fail(ctx: *anyopaque, slot: usize, message: []const u8) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
        std.log.err("slot {d} failed: {s}", .{ slot, message });
    }
};

/// Serves `which` of the prompts together, each in the slot of its index, until every reply has ended; each
/// slot's state is hashed before it is retired.
pub const Plan = struct {
    lengths: [streams]usize = @splat(made),
    depth: ?usize = null,
    /// Replace proposed prefixes with the serial oracle, then reject one known depth. The verifier,
    /// rollback and tap absorption must repair the suffix on both ranks before the next round.
    reject: ?*const [streams]Left = null,
};

pub fn serve(engine: *Engine, asked: [streams][]const i32, which: []const usize, out: *[streams]Left) !void {
    return servePlan(engine, asked, which, out, .{});
}

pub fn servePlan(engine: *Engine, asked: [streams][]const i32, which: []const usize, out: *[streams]Left, plan: Plan) !void {
    const gpa = engine.gpa;
    var lists: [streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&lists) |*list| list.deinit(gpa);
    var ended: [streams]bool = @splat(true);
    for (which) |stream| {
        ended[stream] = false;
        const depth = plan.depth orelse engine.depth;
        out[stream].cached = try engine.begin(stream, .{ .prompt = asked[stream], .settings = greedy, .max_tokens = plan.lengths[stream], .depth = depth, .opening = depth }, 0);
    }
    var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended, .out = out };
    const to: hall_file.Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail, .tail = Collect.tail };
    var retired: [streams]bool = @splat(false);
    var spins: usize = 0;
    while (true) : (spins += 1) {
        if (spins == 400) return error.Stuck;
        if (plan.reject) |oracle| for (which) |stream| {
            const seat = &(engine.seats[stream] orelse continue);
            if (seat.phase != .decode or seat.draft_n == 0) continue;
            const count = @min(seat.draft_n, plan.lengths[stream] - seat.made - 1);
            if (count == 0) continue;
            const live = &engine.seats[stream].?;
            @memcpy(live.drafts[0..count], oracle[stream].tokens[seat.made..][0..count]);
            const at = @min(count - 1, stream * 2);
            live.drafts[at] = @mod(live.drafts[at] + 1, @as(i32, @intCast(engine.m.shape.vocab)));
            live.draft_n = count;
        };
        switch (engine.round(to)) {
            .ran => engine.publishAhead(to),
            .short => return error.Short,
            .failed => return error.Failed,
        }
        var all = true;
        for (which) |stream| {
            if (!ended[stream]) {
                all = false;
                continue;
            }
            if (retired[stream]) continue;
            out[stream].state = try slotHash(engine, stream);
            if (lists[stream].items.len != plan.lengths[stream]) return error.Short;
            @memcpy(out[stream].tokens[0..plan.lengths[stream]], lists[stream].items);
            _ = engine.retire(stream, .ended, plan.lengths[stream]);
            retired[stream] = true;
        }
        if (all) return;
    }
}

/// An engine of the four streams over `m`'s share, and its link, both ranks' alike. The load's captures are
/// told through the link (`Link.captureFirst`, as `serve`'s are), the rounds replay graphs as one rank's do.
pub fn open(gpa: Allocator, m: H.Model, link: *hall_ranks.Link) !Engine {
    return openDrafting(gpa, m, link, null);
}

pub fn openDrafting(gpa: Allocator, m: H.Model, link: *hall_ranks.Link, opened: ?*const rig_file.Opened) !Engine {
    const places = extents.lowerBound(room, streams) + 1;
    var engine: Engine = try .open(gpa, m.ranks[0], &.{}, .{
        .slots = streams,
        .session = .{ .capacity = room, .prompt_rows = 32, .drafter = opened, .copies = opened == null },
        .pool = places * extents.positions,
        .chunk = 8,
        .lazy_graphs = false,
        .capture_at_open = false,
    });
    errdefer engine.deinit();
    link.* = try .init(gpa, m, streams, room);
    return engine;
}

/// Each stream alone, then the four together: null where they agree, else the stream that parts.
pub fn compare(engine: *Engine, asked: [streams][]const i32) !?usize {
    var counted: [3]u64 = undefined;
    return compareCounting(engine, asked, &counted);
}

/// `compare`, the drafts the four carried together and kept, and the prompt rows kept states gave them, into
/// `counted`.
pub fn compareCounting(engine: *Engine, asked: [streams][]const i32, counted: *[3]u64) !?usize {
    var alone: [streams]Left = @splat(.{});
    for (0..streams) |stream| try serve(engine, asked, &.{stream}, &alone);
    var together: [streams]Left = @splat(.{});
    try serve(engine, asked, &.{ 0, 1, 2, 3 }, &together);
    const priced = if (engine.m.mtp) |head| head.costs != null else false;
    // four together on two ranks cut a chain at `hall_depth.most_drafts` (spec 144 WS-04): the drafts may
    // differ from a stream's alone, its tokens and state never
    const cut = engine.capped > 0;
    counted.* = .{ 0, 0, 0 };
    for (together) |got| counted.* = .{ counted[0] + got.drafted, counted[1] + got.accepted, counted[2] + got.cached };
    for (together, alone, 0..) |got, want, stream| {
        // alone, each began afresh; together, from the state its run alone kept
        if (want.cached != 0 or got.cached == 0) return stream;
        if (!std.mem.eql(i32, &got.tokens, &want.tokens) or got.state != want.state) return stream;
        if (!priced and !cut and (got.drafted != want.drafted or got.accepted != want.accepted)) return stream;
    }
    // the streams' states are their own, so the hash holds them
    if (together[0].state == together[3].state) return 0;
    return null;
}

/// The real two-process flash regression: serial tokens/state, unequal replies together, known rejected
/// prefixes on successive rounds, host/device choice, cancellation with drafts outstanding, and slot reuse.
/// Natural proposals are exercised before the controlled rejection; a serial-only attachment fails.
pub fn compareFlash(engine: *Engine, asked: [streams][]const i32, counted: *[3]u64) !?usize {
    const flash = engine.rank_flash orelse return error.OutOfStep;
    const lengths = [streams]usize{ made, made - 1, made - 2, made - 4 };
    var serial: [streams]Left = @splat(.{});
    for (0..streams) |stream| try servePlan(engine, asked, &.{stream}, &serial, .{ .lengths = lengths, .depth = 0 });
    counted.* = .{ 0, 0, 0 };
    for ([_]bool{ true, false }) |device| {
        engine.settle_on_device = device;
        var together: [streams]Left = @splat(.{});
        try servePlan(engine, asked, &.{ 0, 1, 2, 3 }, &together, .{ .lengths = lengths });
        for (together, serial, 0..) |got, want, stream| {
            if (!std.mem.eql(i32, got.tokens[0..lengths[stream]], want.tokens[0..lengths[stream]]) or got.state != want.state) return stream;
            counted.* = .{ counted[0] + got.drafted, counted[1] + got.accepted, counted[2] + got.cached };
        }
        var rejected: [streams]Left = @splat(.{});
        try servePlan(engine, asked, &.{ 0, 1, 2, 3 }, &rejected, .{ .lengths = lengths, .reject = &serial });
        var dropped: u64 = 0;
        for (rejected, serial, 0..) |got, want, stream| {
            if (!std.mem.eql(i32, got.tokens[0..lengths[stream]], want.tokens[0..lengths[stream]]) or got.state != want.state) return stream;
            dropped += got.drafted - got.accepted;
        }
        if (dropped == 0) return error.NoRejection;
    }
    if (counted[0] == 0 or flash.flash.d.passes == 0) return error.NoDrafts;
    // Cancel after a proposal, before verifying it; an unrelated prompt reuses that same slot.
    _ = try engine.begin(0, .{ .prompt = asked[3], .settings = greedy, .max_tokens = made, .depth = engine.depth, .opening = engine.depth }, 0);
    var lists: [streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&lists) |*list| list.deinit(engine.gpa);
    var ended: [streams]bool = @splat(false);
    var cancelled: [streams]Left = @splat(.{});
    var collect: Collect = .{ .gpa = engine.gpa, .lists = &lists, .ended = &ended, .out = &cancelled };
    const out: hall_file.Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail, .tail = Collect.tail };
    var spins: usize = 0;
    while (engine.seats[0].?.draft_n == 0) : (spins += 1) {
        if (spins == 400 or ended[0]) return error.NoDrafts;
        if (engine.round(out) != .ran) return error.Failed;
    }
    _ = engine.retire(0, .cancelled, 1);
    var reused: [streams]Left = @splat(.{});
    try servePlan(engine, asked, &.{ 0, 1, 2, 3 }, &reused, .{ .lengths = lengths });
    for (reused, serial, 0..) |got, want, stream| {
        if (!std.mem.eql(i32, got.tokens[0..lengths[stream]], want.tokens[0..lengths[stream]]) or got.state != want.state) return stream;
    }
    return null;
}

/// One kept state as this rank holds it: what both ranks know (`id`, `pos`, `head_len`) and a hash of the
/// bytes of its half.
const KeptHash = struct { id: u32, pos: usize, head_len: usize, bytes: u64 };

fn byId(_: void, a: KeptHash, b: KeptHash) bool {
    return a.id < b.id;
}

/// A state half's device bytes and histories, however each rank's share holds it.
fn hashKeptState(gpa: Allocator, hasher: *std.hash.Wyhash, state: forward.Snapshot, pooled: cuda.Parts) !void {
    for ([_][]cuda.Buffer{ state.recurrent, state.conv, state.tails }) |list| for (list) |buffer| {
        if (buffer.view.ptr != null) try hashBuffer(gpa, hasher, buffer, 0, @intCast(buffer.view.size[0]));
    };
    for (state.history) |h| hasher.update(std.mem.sliceAsBytes(h));
    for (pooled.list) |buffer| {
        if (buffer.view.ptr != null) try hashBuffer(gpa, hasher, buffer, 0, @intCast(buffer.view.size[0]));
    }
}

/// Every kept state this rank holds, in id order. Rank 0's are `engine.keeps`'s, told of where `Keep.id`
/// stands; rank 1's the link's `held`. Each half is compared with an independent replay below, not only
/// through rank 0's bits.
fn keptHashes(gpa: Allocator, engine: *Engine, link: *hall_ranks.Link, rank: usize) ![]KeptHash {
    var out: std.ArrayList(KeptHash) = .empty;
    errdefer out.deinit(gpa);
    if (rank == 0) {
        for (engine.keeps.list.items) |keep| {
            if (keep.id == 0) continue;
            var hasher: std.hash.Wyhash = .init(421);
            try hashKeptState(gpa, &hasher, keep.state, keep.pooled);
            try out.append(gpa, .{ .id = keep.id, .pos = keep.pos, .head_len = keep.head_len, .bytes = hasher.final() });
        }
    } else {
        var it = link.held.iterator();
        while (it.next()) |entry| {
            const kept = entry.value_ptr.*;
            var hasher: std.hash.Wyhash = .init(421);
            try hashKeptState(gpa, &hasher, kept.state, kept.pooled);
            try out.append(gpa, .{ .id = kept.id, .pos = kept.pos, .head_len = kept.head_len, .bytes = hasher.final() });
        }
    }
    std.mem.sort(KeptHash, out.items, {}, byId);
    return out.toOwnedSlice(gpa);
}

/// Replay each prefix from zero, one row at a time, without the pool, forks, graphs or within-window
/// snapshots. Both ranks compute together, but compare only with their own half's retained bytes. The
/// head consumes the next token alongside each trunk row, including `Keep.next` at the prefix's end.
fn checkKept(gpa: Allocator, hm: H.Model, engine: *Engine, lead: *lead_file.Lead, rank: usize, kept: []const KeptHash) !void {
    var m = hm;
    m.lead = null;
    var state: H.State = try .init(gpa, m, 1, room);
    defer state.deinit();
    var b: H.Buffers = try .init(gpa, m, 1, room);
    defer b.deinit();
    const split_head = if (m.ranks[0].mtp) |head| head.split != null else false;
    var split: mtp_ranks.Held(1).Split = undefined;
    if (split_head) try split.init(gpa, m, &b, null, 1, room, false);
    defer if (split_head) split.deinit();
    const whole_head = !split_head and engine.slots[0].head != null;
    var head_state: mtp.State = undefined;
    var head_b: mtp.Buffers = undefined;
    if (whole_head) head_state = try .init(m.ranks[0], room);
    defer if (whole_head) head_state.deinit();
    if (whole_head) head_b = try .init(gpa, m.ranks[0], 1, room);
    defer if (whole_head) head_b.deinit();
    var window: [room]i32 = undefined;
    for (kept) |want| {
        var next: i32 = undefined;
        const tokens = if (rank == 0) blk: {
            const original = for (engine.keeps.list.items) |one| {
                if (one.id == want.id) break one;
            } else return error.OutOfStep;
            next = original.next;
            const prefix = original.chain.tokens.items[0..want.pos];
            try lead.tell(want.id, &.{@bitCast(next)}, prefix);
            break :blk prefix;
        } else blk: {
            const told = switch (try lead.channel.next()) {
                .request => |bytes| lead_file.Told.read(bytes) orelse return error.OutOfStep,
                .barrier => return error.OutOfStep,
            };
            if (told.kind != want.id or told.rest.len != 1 + want.pos) return error.OutOfStep;
            next = @bitCast(told.rest[0]);
            break :blk told.tokens(1, &window) orelse return error.OutOfStep;
        };
        try state.ranks[0].reset();
        state.pos = 0;
        if (split_head) try split.state[0].position.set(0);
        if (whole_head) try head_state.position.set(0);
        for (tokens, 0..) |token, at| {
            try H.forwardScoring(m, &state, &b, &.{token}, .none);
            try H.commit(m, &state, b, &.{token}, 1);
            if (at < want.head_len) {
                const shifted = if (at + 1 < tokens.len) tokens[at + 1] else next;
                if (split_head) try split.follow(at, null, &.{shifted}, false, true);
                if (whole_head) {
                    try mtp.run(m.ranks[0], &head_state, &head_b, b.ranks[0].h.view, &.{shifted}, false);
                    try head_state.position.advance(1);
                }
            }
        }
        var snap = try forward.Snapshot.take(gpa, state.ranks[0]);
        defer snap.deinit();
        const head_cache = if (split_head) split.state[0].cache else if (whole_head) head_state.cache else null;
        const blocks = if (m.ranks[0].attention) |shared| want.pos / shared.shape.ratio else 0;
        var pooled = try @import("../pool/hall_keep.zig").pooledOf(gpa, state.ranks[0].caches, head_cache, blocks);
        defer pooled.deinit(gpa);
        try cuda.synchronize();
        var hasher: std.hash.Wyhash = .init(421);
        try hashKeptState(gpa, &hasher, snap, pooled);
        if (hasher.final() != want.bytes) {
            std.log.err("rank {d}: kept state {d} at {d} differs from its independent serial prefix", .{ rank, want.id, want.pos });
            return error.StateDiffers;
        }
        // A mismatch must reach the peer before it starts another prefix's NCCL gathers.
        try lead.channel.barrier(want.id);
    }
}

pub fn run(gpa: Allocator, arena: Allocator, stdout: *std.Io.Writer, channel: *channel_file.Channel, comm: nccl.Comm, rank: usize, seed: u64, shares: [1]model.Model, opened: ?*const rig_file.Opened) !u8 {
    // a stream's 32-row rounds alone split their glue by rows (spec 149 WS-02), on both ranks alike; the four
    // together, a chunk of 8 each, from 32 rows; what the engine keeps, commits and chases after them is checked
    const saved = held_file.split_rows;
    defer held_file.split_rows = saved;
    held_file.split_rows = streams_file.split_at;
    var lead: lead_file.Lead = .{ .channel = channel };
    var m: H.Model = .{ .ranks = shares, .first = rank, .max_rows = rows, .gather = .{ .nccl = comm }, .lead = if (rank == 0) &lead else null };
    defer m.deinit();
    var link: hall_ranks.Link = undefined;
    var engine = try openDrafting(gpa, m, &link, opened);
    defer engine.deinit();
    defer link.deinit();
    engine.bindPool();
    link.bind(&engine);
    if (opened) |source| {
        const flash = try rig_file.Pool.create(arena, gpa, source, m, room, streams, rig_file.drafter_stored);
        errdefer if (engine.rank_flash == null) flash.destroy();
        try engine.seatRankDrafter(flash);
        // Exercise full proposed windows, independent of the trained confidence of a narrowed target.
        for (flash.flash.books.predecessor) |*value| value.* *= 12;
        flash.flash.d.confidence = 0;
        flash.flash.d.cap = flash.flash.d.c.block - 1;
    }
    if (rank == 0) {
        // the load's captures, rank 1 told each after it (`Link.captureFirst`), as `serve`'s load does
        link.captureFirst() catch |err| return channel.failWith(rank, err);
        const asked = try streams_file.prompts(arena, seed);
        var counted: [3]u64 = undefined;
        const parted = (if (opened != null) compareFlash(&engine, asked, &counted) else compareCounting(&engine, asked, &counted)) catch |err| return channel.failWith(rank, err);
        held_file.told(&lead, .end, &.{}, &.{}) catch |err| return channel.failWith(rank, err);
        if (parted) |stream| {
            std.log.err("rank 0: stream {d} among the others is not the stream alone", .{stream});
            channel.fail("a stream of the pool among the others is not the stream alone");
            return 1;
        }
        const head = if (opened != null) "DFlash2" else if (m.ranks[0].mtp) |held| (if (held.split != null) "split" else "whole") else "none";
        try stdout.print("rank 0: the pool's four streams together, each its tokens and its slot's state alone, in bits; the head {s}, {d} drafts together, {d} kept, {d} chains cut by the round's width; {d} prompt rows from kept states ({d} taken within a chunk, {d} where a slot stood, {d} let go); {d} graphs; {d} operations told\n", .{ head, counted[0], counted[1], engine.capped, counted[2], engine.keeps.took.within, engine.keeps.took.standing, link.told_gone, engine.graphs.cache.count(), lead.told });
        if (opened != null) try stdout.writeAll("DFlash2: unequal lengths and serial token/state parity, rejected draft prefixes, host/device chooser, cancel before verification and slot reuse passed\n");
    } else {
        const ran = hall_ranks.follow(channel, &engine, &link) catch |err| return channel.failWith(rank, err);
        try stdout.print("rank 1: followed the pool's streams, {d} operations; {d} graphs\n", .{ ran, engine.graphs.cache.count() });
    }
    try stdout.flush();
    // where this rank's slots stand, the tables told, and its kept states: the two ranks' kept sets are
    // held to agree id for id, each state at the same position and, for a split head, head length. A whole
    // head exists only on rank 0: rank 1 correctly keeps zero head rows there.
    var hash = std.hash.Wyhash.init(1);
    for (engine.slots, link.tables, link.placed) |slot, table, placed| {
        hash.update(std.mem.asBytes(&slot.pos));
        hash.update(std.mem.sliceAsBytes(table[0..placed]));
    }
    const kept = keptHashes(gpa, &engine, &link, rank) catch |err| return channel.failWith(rank, err);
    defer gpa.free(kept);
    for (kept) |one| {
        hash.update(std.mem.asBytes(&one.id));
        hash.update(std.mem.asBytes(&one.pos));
        if (model.ranks_head == .split) hash.update(std.mem.asBytes(&one.head_len));
    }
    var bytes_hash: std.hash.Wyhash = .init(421);
    for (kept) |one| bytes_hash.update(std.mem.asBytes(&one.bytes));
    try stdout.print("rank {d}: {d} states kept, their halves {x}\n", .{ rank, kept.len, bytes_hash.final() });
    channel.barrier(hash.final()) catch |err| {
        if (err == error.PeerFailed) std.log.err("rank {d}: the other rank failed: {s}", .{ rank, channel.said() });
        return channel.failWith(rank, err);
    };
    checkKept(gpa, m, &engine, &lead, rank, kept) catch |err| return channel.failWith(rank, err);
    try stdout.print("rank {d}: all {d} kept halves match independent serial prefixes in bits (recurrent, conv, PLE histories/tails, trunk/head pooled keys)\n", .{ rank, kept.len });
    try stdout.flush();
    channel.barrier(kept.len) catch |err| return channel.failWith(rank, err);
    return 0;
}
