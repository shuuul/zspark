//! `ranks-check random states`: both ranks keep a snapshot at the same position, run past it, restore it,
//! and run that prefix again from empty. Each rank's state and its snapshot are then held byte for byte to
//! the same rank's share through the loopback (no NCCL, no mirror), which is the independent baseline: the
//! two ranks hold different halves of the weights, so no hash of theirs is compared with the other's. A
//! barrier holds the position the same on both.

const std = @import("std");
const Allocator = std.mem.Allocator;
const channel_file = @import("../../../ranks/channel.zig");
const check = @import("../../../ranks/check.zig");
const nccl = @import("../../../ranks/nccl.zig");
const cuda = @import("../../../cuda/cuda.zig");
const lead_file = @import("../../../ranks/lead.zig");
const config_file = @import("../layers/config.zig");
const held_file = @import("../layers/ranks_forward.zig");
const kda = @import("../layers/kda.zig");
const model_file = @import("../layers/model.zig");
const mtp = @import("../mtp/mtp.zig");
const mirror = @import("../session/mirror.zig");
const snapshot = @import("../session/snapshot.zig");
const fx = @import("random.zig").fx;

const H = held_file.Held(1);
const capacity = 64;

const Book = struct {
    gpa: Allocator,
    shots: std.ArrayList(snapshot.Shot) = .empty,
    state: *H.State,

    fn deinit(self: *Book) void {
        for (self.shots.items) |*shot| shot.deinit();
        self.shots.deinit(self.gpa);
    }

    fn keep(context: *anyopaque) anyerror!void {
        const self: *Book = @ptrCast(@alignCast(context));
        var shot = try snapshot.takeRank(self.state.ranks[0], null);
        self.shots.append(self.gpa, shot) catch |err| {
            shot.deinit();
            return err;
        };
    }

    fn back(context: *anyopaque, pos: usize) anyerror!void {
        const self: *Book = @ptrCast(@alignCast(context));
        for (self.shots.items) |shot| if (shot.pos == pos) {
            try snapshot.restoreRank(shot, &self.state.ranks[0]);
            self.state.pos = pos;
            return;
        };
        return error.NoSnapshot;
    }
};

pub fn run(gpa: Allocator, arena: Allocator, stdout: *std.Io.Writer, channel: *channel_file.Channel, comm: nccl.Comm, rank: usize, source: anytype, c: *const config_file.Config, rows: cuda.Scratch.Rows, tokens: []const i32) !u8 {
    var lead: lead_file.Lead = .{ .channel = channel };
    var m: H.Model = try .loadFrom(arena, gpa, source, c, rank, rows);
    errdefer m.deinit();
    m.gather = .{ .nccl = comm };
    if (rank == 0) m.lead = &lead;
    var state: H.State = try .init(m, capacity);
    errdefer state.deinit();
    var b: H.Buffers = try .init(m, capacity);
    errdefer b.deinit();
    var pair: mtp.PairOf(1) = try .init(arena, gpa, source, c, m, capacity, rows);
    errdefer pair.deinit();
    var book: Book = .{ .gpa = gpa, .state = &state };
    defer book.deinit();
    const logits = try arena.alloc(u16, fx.vocab);
    const prompt = tokens[0..4];
    const spoil = tokens[4..5];

    if (rank == 0) {
        H.forwardScoring(m, &state, &b, prompt, .none) catch |err| return channel.failWith(rank, err);
        H.commit(m, &state, b, prompt, prompt.len) catch |err| return channel.failWith(rank, err);
        lead.tell(@backingInt(held_file.Op.keep), &.{}, &.{}) catch |err| return channel.failWith(rank, err);
        Book.keep(&book) catch |err| return channel.failWith(rank, err);
        const kept = digest(gpa, state.ranks[0]) catch |err| return channel.failWith(rank, err);
        H.forwardScoring(m, &state, &b, spoil, .none) catch |err| return channel.failWith(rank, err);
        H.commit(m, &state, b, spoil, 1) catch |err| return channel.failWith(rank, err);
        lead.tell(@backingInt(held_file.Op.back), &.{prompt.len}, &.{}) catch |err| return channel.failWith(rank, err);
        Book.back(&book, prompt.len) catch |err| return channel.failWith(rank, err);
        const restored = digest(gpa, state.ranks[0]) catch |err| return channel.failWith(rank, err);
        if (book.shots.items.len != 1 or book.shots.items[0].pos != prompt.len or kept != restored or state.pos != prompt.len) {
            channel.fail("the restored state is not the snapshot");
            return 1;
        }
        state.reset() catch |err| return channel.failWith(rank, err);
        H.forwardScoring(m, &state, &b, prompt, .none) catch |err| return channel.failWith(rank, err);
        H.commit(m, &state, b, prompt, prompt.len) catch |err| return channel.failWith(rank, err);
        const again = digest(gpa, state.ranks[0]) catch |err| return channel.failWith(rank, err);
        if (again != kept) {
            channel.fail("the cold run is not the snapshot");
            return 1;
        }
        lead.tell(@backingInt(held_file.Op.end), &.{}, &.{}) catch |err| return channel.failWith(rank, err);
        try stdout.print("rank 0: snapshot at {d}, restored, then a cold run of the prefix\n", .{prompt.len});
    } else {
        const followed: mirror.Followed = .{ .m = m, .state = &state, .b = &b, .graphs = null, .pair = &pair, .logits = logits, .keep = Book.keep, .back = Book.back, .snap = &book };
        _ = mirror.follow(channel, followed) catch |err| return channel.failWith(rank, err);
        try stdout.print("rank 1: followed, {d} snapshot(s)\n", .{book.shots.items.len});
    }
    try stdout.flush();
    if (book.shots.items.len != 1 or book.shots.items[0].pos != prompt.len or state.pos != prompt.len) {
        channel.fail("the snapshot is not at the prefix");
        return 1;
    }
    const live = digest(gpa, state.ranks[0]) catch |err| return channel.failWith(rank, err);
    const taken = digestShot(gpa, book.shots.items[0]) catch |err| return channel.failWith(rank, err);
    if (live != taken) {
        channel.fail("the cold run is not the snapshot");
        return 1;
    }
    // this rank's own share through the loopback, no NCCL and no mirror: the independent baseline the
    // snapshot's bytes are held to (the two ranks' shares are different halves, so no cross-rank hash)
    var loop_m: held_file.Held(2).Model = try .loadFrom(arena, gpa, source, c, 0, rows);
    defer loop_m.deinit();
    var loop_state: held_file.Held(2).State = try .init(loop_m, capacity);
    defer loop_state.deinit();
    var loop_b: held_file.Held(2).Buffers = try .init(loop_m, capacity);
    defer loop_b.deinit();
    held_file.Held(2).forwardScoring(loop_m, &loop_state, &loop_b, prompt, .none) catch |err| return channel.failWith(rank, err);
    held_file.Held(2).commit(loop_m, &loop_state, loop_b, prompt, prompt.len) catch |err| return channel.failWith(rank, err);
    const base = digest(gpa, loop_state.ranks[rank]) catch |err| return channel.failWith(rank, err);
    if (live != base or taken != base) {
        channel.fail("this rank's state or its snapshot is not the loopback's same-rank bits");
        return 1;
    }
    if (check.atPosition(channel, rank, prompt.len)) |code| return code;
    try stdout.print("rank {d}: the other rank holds position {d}, and this rank's snapshot, its restore and its cold run are the loopback's bits for this rank's share\n", .{ rank, prompt.len });
    try stdout.flush();
    pair.deinit();
    b.deinit();
    state.deinit();
    m.deinit();
    return 0;
}

/// The shot's KDA state, hashed as `digest` hashes a live one: the copy, not the cache it was cut from.
fn digestShot(gpa: Allocator, shot: snapshot.Shot) !u64 {
    var hasher = std.hash.Wyhash.init(shot.pos);
    const kept = shot.kda orelse return hasher.final();
    const one = kept.heads * kda.head_dim * kda.head_dim;
    const bytes = try gpa.alloc(u8, kept.layers * one * 4);
    defer gpa.free(bytes);
    for (0..kept.layers) |index| try cuda.download(kept.committed(index), bytes[index * one * 4 ..][0 .. one * 4]);
    hasher.update(bytes);
    return hasher.final();
}

fn digest(gpa: Allocator, state: model_file.State) !u64 {
    var hasher = std.hash.Wyhash.init(state.pos);
    const chains = state.kda orelse return hasher.final();
    const one = chains.heads * kda.head_dim * kda.head_dim;
    const bytes = try gpa.alloc(u8, chains.layers * one * 4);
    defer gpa.free(bytes);
    for (0..chains.layers) |index| try cuda.download(chains.committed(index), bytes[index * one * 4 ..][0 .. one * 4]);
    hasher.update(bytes);
    return hasher.final();
}
