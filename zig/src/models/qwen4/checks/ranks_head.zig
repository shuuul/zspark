//! `ranks-check ... session` with the head split (the default), before the session: the split MTP head's passes
//! (spec 090's scheme B) with both halves in this process, then this rank's half alone with the other's
//! partials through NCCL, compared bit for bit. Both ranks run the same script, no lead: a window absorbed and
//! scored, two chained levels, the chain dropped under the next window, a row launched; all of it launched,
//! then through captured graphs.

const std = @import("std");
const Allocator = std.mem.Allocator;
const channel_file = @import("../../../ranks/channel.zig");
const cuda = @import("../../../cuda/cuda.zig");
const nccl = @import("../../../ranks/nccl.zig");
const model = @import("../model.zig");
const mtp_ranks = @import("../mtp_ranks.zig");
const held_file = @import("../ranks_forward.zig");
const share_file = @import("../share.zig");

const world = share_file.world;
const rows = 4;
const room = 64;

/// What a script left, in order: each buffer read's bytes end at a mark.
const Taken = struct {
    bytes: std.ArrayList(u8) = .empty,
    marks: std.ArrayList(usize) = .empty,
    gpa: Allocator,

    fn deinit(self: *Taken) void {
        self.bytes.deinit(self.gpa);
        self.marks.deinit(self.gpa);
    }

    /// The first `count` bytes of `buffer`, once what was launched has run.
    fn take(self: *Taken, buffer: cuda.Buffer, count: usize) !void {
        try cuda.synchronize();
        try buffer.downloadAt(0, try self.bytes.addManyAsSlice(self.gpa, count));
        try self.marks.append(self.gpa, self.bytes.items.len);
    }

    /// The first read that is other bytes, null where every one is the same.
    fn differs(self: Taken, other: Taken) ?usize {
        if (std.mem.eql(u8, self.bytes.items, other.bytes.items) and std.mem.eql(usize, self.marks.items, other.marks.items)) return null;
        var from: usize = 0;
        for (self.marks.items, 0..) |end, index| {
            if (index >= other.marks.items.len or other.marks.items[index] != end or !std.mem.eql(u8, self.bytes.items[from..end], other.bytes.items[from..end])) return index;
            from = end;
        }
        return self.marks.items.len;
    }
};

/// The script over `held` ranks, `mine` of them this process's: its streams after each pass, and each
/// scored row as gathered, which every rank holds; rank 0 what its choice reads of it.
fn play(comptime held: usize, gpa: Allocator, m: held_file.Held(held).Model, mine: usize, graphed: bool, taken: *Taken) !void {
    const H = held_file.Held(held);
    var state: H.State = try .init(gpa, m, rows, room);
    defer state.deinit();
    var b: H.Buffers = try .init(gpa, m, rows, room);
    defer b.deinit();
    var split: mtp_ranks.Held(held).Split = undefined;
    try split.init(gpa, m, &b, null, rows, room, graphed);
    defer split.deinit();
    const own = split.b[mine];
    const row = split.b[0].draft_logits.?;
    const wide = own.layer.h.bytes() / @as(usize, @intCast(own.layer.h.view.size[0]));
    const first = m.first + mine == 0;

    const window = [_]i32{ 9, 41, 300, 2 };
    try H.forward(m, &state, &b, &window);
    try H.commit(m, &state, b, &window, window.len);
    try split.follow(0, null, &.{ 41, 300, 2, 77 }, true, false);
    try taken.take(own.layer.h, window.len * wide);
    try taken.take(row, row.bytes());
    if (first) try taken.take(own.picked, own.picked.bytes());
    // two levels of a chain, each reading the row the pass before left
    for ([_]i32{ 5, 250 }, [_]usize{ window.len - 1, 0 }, 0..) |token, at, level| {
        try split.follow(window.len + level, at, &.{token}, true, false);
        try taken.take(own.layer.h, wide);
        try taken.take(row, row.bytes());
        if (first) try taken.take(own.picked, own.picked.bytes());
    }
    // the chain was not kept: the next window is absorbed over its entries, on every rank
    const next = [_]i32{ 77, 8 };
    try H.forward(m, &state, &b, &next);
    try H.commit(m, &state, b, &next, next.len);
    try split.follow(window.len, null, &.{ 8, 19 }, false, false);
    try taken.take(own.layer.h, next.len * wide);
    // a row as a reply's last window is absorbed and scored, launched whatever the graphs
    try H.forward(m, &state, &b, &.{19});
    try H.commit(m, &state, b, &.{19}, 1);
    try split.follow(window.len + next.len, null, &.{3}, true, true);
    try taken.take(own.layer.h, wide);
    try taken.take(row, row.bytes());
    if (first) try taken.take(own.picked, own.picked.bytes());
    for (split.state) |one| std.debug.assert(one.position.host == window.len + next.len + 1);
}

/// `both`: every rank's half, in this process; `own`: this rank's. Takes both. 0, or 1 after telling the other
/// rank what differed.
pub fn run(gpa: Allocator, stdout: *std.Io.Writer, channel: *channel_file.Channel, comm: nccl.Comm, rank: usize, both: [world]model.Model, own: [1]model.Model) !u8 {
    var loopback: [2]Taken = .{ .{ .gpa = gpa }, .{ .gpa = gpa } };
    defer for (&loopback) |*one| one.deinit();
    var own_m: held_file.Held(1).Model = .{ .ranks = own, .first = rank, .max_rows = rows, .gather = .{ .nccl = comm } };
    defer own_m.deinit();
    {
        var m: held_file.Held(world).Model = .{ .ranks = both, .max_rows = rows };
        defer m.deinit();
        for (&loopback, [_]bool{ false, true }) |*taken, graphed| try play(world, gpa, m, rank, graphed, taken);
    }
    var failure: ?[]const u8 = null;
    var buffer: [200]u8 = undefined;
    if (loopback[0].differs(loopback[1])) |index| failure = try std.fmt.bufPrint(&buffer, "rank {d}: the split head's loopback itself: read {d} through graphs is not the launched one", .{ rank, index });
    try stdout.print("loopback: the split head's passes launched, {d} buffers read, {d} bytes; through graphs, {s}\n", .{ loopback[0].marks.items.len, loopback[0].bytes.items.len, if (failure == null) "the same" else "DIFFERENT" });
    try stdout.flush();
    // both ranks run every pass whatever they find: one that stopped would leave the other inside a gather
    for (loopback, [_]bool{ false, true }, [_][]const u8{ "launched", "through captured graphs" }, 0..) |want, graphed, how, index| {
        channel.barrier(900 + index) catch |err| return channel.gone(rank, err);
        var got: Taken = .{ .gpa = gpa };
        defer got.deinit();
        try play(1, gpa, own_m, 0, graphed, &got);
        if (want.differs(got)) |at| {
            try stdout.print("over NCCL, the split head {s}: DIFFERS at read {d}\n", .{ how, at });
            if (failure == null) failure = try std.fmt.bufPrint(&buffer, "rank {d}: the split head {s}: read {d} over NCCL is not the loopback's", .{ rank, how, at });
        } else try stdout.print("over NCCL, the split head {s}: {d} buffers, {d} bytes, the loopback's bits\n", .{ how, got.marks.items.len, got.bytes.items.len });
        try stdout.flush();
    }
    if (failure) |text| {
        channel.fail(text);
        return 1;
    }
    channel.barrier(902) catch |err| return channel.gone(rank, err);
    return 0;
}
