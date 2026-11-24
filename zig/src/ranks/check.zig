//! What every model's `zspark ranks-check` does alike between its runs and at its end (spec 136 WS-05): both
//! ranks meet at a numbered barrier before either holds a run to the loopback's, and at the end a rank that found
//! a difference tells the other and fails, or both meet once more and pass. Both ranks run every step whatever
//! they find: one that stopped would leave the other inside a gather.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");
const channel_file = @import("channel.zig");

/// This rank's run over NCCL has been launched: once it has run and both ranks meet at `barrier`, it is held to
/// the loopback's, `want` (a `trace.Trace`; `Trace.report`, the first difference kept in `failure`). Null to go
/// on; otherwise the exit status, the channel having ended.
pub fn held(arena: Allocator, stdout: *std.Io.Writer, channel: *channel_file.Channel, rank: usize, barrier: u64, failure: *?[]const u8, comptime how: []const u8, want: anytype, got: @TypeOf(want)) !?u8 {
    try cuda.synchronize();
    channel.barrier(barrier) catch |err| return channel.gone(rank, err);
    try @TypeOf(want.*).report(arena, stdout, failure, how, want, got);
    return null;
}

/// The check's end: `failure` told the other rank and printed, 1; else both ranks meet at `barrier` and this one
/// says every bit over NCCL is the loopback's, `also` after it, 0.
pub fn end(arena: Allocator, stdout: *std.Io.Writer, channel: *channel_file.Channel, rank: usize, failure: ?[]const u8, barrier: u64, comptime also: []const u8) !u8 {
    if (failure) |text| {
        channel.fail(try std.fmt.allocPrint(arena, "rank {d}: {s}", .{ rank, text }));
        try stdout.print("rank {d}: FAILED: {s}\n", .{ rank, text });
        return 1;
    }
    channel.barrier(barrier) catch |err| return channel.gone(rank, err);
    try stdout.print("rank {d}: every bit over NCCL is the loopback's" ++ also ++ "\n", .{rank});
    return 0;
}

/// A states check's end (`ranks-check ... states`): both ranks at position `pos`, or 1 where the barrier
/// finds them apart. Null to go on.
pub fn atPosition(channel: *channel_file.Channel, rank: usize, pos: usize) ?u8 {
    channel.barrier(pos) catch |err| {
        std.log.err("rank {d}: the positions differ: {t} {s}", .{ rank, err, channel.said() });
        return 1;
    };
    return null;
}
