//! `ranks-check ... states`: both ranks keep a snapshot at the same position, run past it, restore it, and
//! run that prefix again from empty (spec 050 WS-06). A rank's recurrent state is its own heads, so those
//! bytes are compared with that rank's snapshot; a barrier holds the position the same on both.

const std = @import("std");
const Allocator = std.mem.Allocator;
const channel_file = @import("../../../ranks/channel.zig");
const check = @import("../../../ranks/check.zig");
const cuda = @import("../../../cuda/cuda.zig");
const nccl = @import("../../../ranks/nccl.zig");
const model = @import("../model.zig");
const forward_file = @import("../forward.zig");
const held_file = @import("../ranks_forward.zig");
const pair_file = @import("../session/pair.zig");

const H = held_file.Held(1);
const rows = 16;
const room = 64;

pub fn run(gpa: Allocator, stdout: *std.Io.Writer, channel: *channel_file.Channel, comm: nccl.Comm, rank: usize, shares: [1]model.Model) !u8 {
    const pair = try pair_file.Pair.create(gpa, rank, .{ .ranks = shares, .first = rank, .max_rows = rows }, comm, channel, rows, room, true);
    defer pair.destroy();
    const prompt = [_]i32{ 1, 2, 3, 5 };
    const spoil = [_]i32{9};
    if (rank == 0) {
        H.forwardScoring(pair.m, &pair.state, &pair.b, &prompt, .last) catch |err| return channel.failWith(rank, err);
        H.commit(pair.m, &pair.state, pair.b, &prompt, prompt.len) catch |err| return channel.failWith(rank, err);
        pair.keep() catch |err| return channel.failWith(rank, err);
        const kept = digestState(gpa, pair.state.ranks[0]) catch |err| return channel.failWith(rank, err);
        H.forwardScoring(pair.m, &pair.state, &pair.b, &spoil, .none) catch |err| return channel.failWith(rank, err);
        H.commit(pair.m, &pair.state, pair.b, &spoil, 1) catch |err| return channel.failWith(rank, err);
        pair.back(prompt.len) catch |err| return channel.failWith(rank, err);
        const restored = digestState(gpa, pair.state.ranks[0]) catch |err| return channel.failWith(rank, err);
        if (pair.shelf.held.items.len != 1 or pair.shelf.held.items[0][0].pos != prompt.len or kept != restored) {
            std.log.err("rank 0: the restored state is not the snapshot: {d} held, at {d}, {x} kept and {x} restored", .{ pair.shelf.held.items.len, if (pair.shelf.held.items.len > 0) pair.shelf.held.items[0][0].pos else 0, kept, restored });
            channel.fail("the restored state is not the snapshot");
            return 1;
        }
        pair.reset() catch |err| return channel.failWith(rank, err);
        H.forwardScoring(pair.m, &pair.state, &pair.b, &prompt, .last) catch |err| return channel.failWith(rank, err);
        H.commit(pair.m, &pair.state, pair.b, &prompt, prompt.len) catch |err| return channel.failWith(rank, err);
        pair.finish() catch |err| return channel.failWith(rank, err);
        try stdout.print("rank 0: snapshot at {d}, restored, then a cold run of the prefix\n", .{prompt.len});
    } else {
        _ = pair.follow() catch |err| return channel.failWith(rank, err);
        try stdout.print("rank 1: followed, {d} snapshot(s)\n", .{pair.shelf.held.items.len});
    }
    try stdout.flush();
    if (pair.shelf.held.items.len != 1 or pair.shelf.held.items[0][0].pos != prompt.len or pair.state.pos != prompt.len) {
        channel.fail("the snapshot is not at the prefix");
        return 1;
    }
    const live = digestState(gpa, pair.state.ranks[0]) catch |err| return channel.failWith(rank, err);
    const taken = digestSnap(gpa, pair.shelf.held.items[0][0]) catch |err| return channel.failWith(rank, err);
    if (live != taken) {
        channel.fail("the cold run is not the snapshot");
        return 1;
    }
    if (check.atPosition(channel, rank, prompt.len)) |code| return code;
    try stdout.print("rank {d}: the other rank holds position {d}, and this rank's cold run matches its snapshot\n", .{ rank, prompt.len });
    return 0;
}

/// The state committed at `state.pos`, as a snapshot of it holds it: a state `behind` rows has its DeltaNet
/// side where the rows before them left it, and those rows are replayed (`forward.committed`), not read.
fn digestState(gpa: Allocator, state: forward_file.State) !u64 {
    var snap: held_file.Snapshot = try .take(gpa, state);
    defer snap.deinit();
    return digestSnap(gpa, snap);
}

fn digestSnap(gpa: Allocator, snap: held_file.Snapshot) !u64 {
    var hasher = std.hash.Wyhash.init(snap.pos);
    for (snap.recurrent) |buffer| try mix(&hasher, gpa, buffer);
    for (snap.conv) |buffer| try mix(&hasher, gpa, buffer);
    return hasher.final();
}

fn mix(hasher: *std.hash.Wyhash, gpa: Allocator, buffer: cuda.Buffer) !void {
    const bytes = try gpa.alloc(u8, buffer.bytes());
    defer gpa.free(bytes);
    try buffer.download(bytes);
    hasher.update(bytes);
}
