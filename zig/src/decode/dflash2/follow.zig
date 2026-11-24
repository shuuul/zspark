//! Rank 1's side of the drafter's passes (spec 136 WS-05, from both models' `session/mirror.zig`): the
//! `flash_taps`, `flash_block` and `flash_round` a leading rank tells (`pass.zig`'s `addTaps` and `block`,
//! `seats.zig`'s `round`), read back and run on this rank's share. This rank consumes the leader's taps and runs
//! the block; it proposes nothing itself. `Target` is the model's (`pass.zig`'s header).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const drafter_file = @import("drafter.zig");
const pass = @import("pass.zig");
const seats_file = @import("seats.zig");

/// The words of arguments an operation told takes, `rest` the words that follow its kind; null where `op` is
/// none of the drafter's or its arguments do not fit.
pub fn arguments(comptime Target: type, op: Target.Op, rest: []align(1) const u32) ?usize {
    return switch (op) {
        .flash_block => 2,
        .flash_taps => 3,
        .flash_round => if (rest.len >= 1 and rest[0] <= Target.lead.Lead.most) seats_file.told(rest[0]) else null,
        else => null,
    };
}

pub fn Error(comptime Target: type) type {
    return error{OutOfStep} || pass.GatherError(Target) || Allocator.Error || Target.lead.Error;
}

/// Runs the drafter's operation `op` with its arguments `a` and the tokens told after them (none). `drafter`
/// and `seats` are what this rank holds of the drafter (null: rank 0 tells none of it), `rows` the main
/// model's rows and `taps` its taps on this rank. A frame this rank cannot run is `OutOfStep`.
pub fn follow(
    comptime Target: type,
    op: Target.Op,
    a: []align(1) const u32,
    tokens: []const i32,
    drafter: ?*drafter_file.DrafterOf(1, Target),
    seats_held: ?*seats_file.SeatsOf(1, Target),
    rows: usize,
    taps: ?cuda.Buffer,
) Error(Target)!void {
    switch (op) {
        .flash_taps, .flash_block => {
            const d = drafter orelse return error.OutOfStep;
            if (tokens.len != 0 or a[0] > d.pair.capacity) return error.OutOfStep;
            d.pair.context_end = a[0];
            if (op == .flash_block) return d.pair.block(d.mains(), @bitCast(a[1]));
            if (a[1] + a[2] > rows or a[0] + a[2] > d.pair.capacity) return error.OutOfStep;
            try d.pair.addTaps(.{taps orelse return error.OutOfStep}, a[1], a[2]);
        },
        .flash_round => {
            const seats = seats_held orelse return error.OutOfStep;
            const d = drafter orelse return error.OutOfStep;
            if (tokens.len != 0 or a[0] == 0 or a[0] > seats.seats.len) return error.OutOfStep;
            const items = seats.items[0..a[0]];
            for (items, 0..) |*item, at| {
                const words = a[1 + 5 * at ..][0..5];
                if (words[0] >= seats.seats.len or words[3] == 0 or words[4] > std.math.maxInt(i32)) return error.OutOfStep;
                if (@as(usize, words[2]) + words[3] > rows or @as(usize, words[1]) + words[3] > d.pair.capacity) return error.OutOfStep;
                for (items[0..at]) |before| if (before.slot == words[0]) return error.OutOfStep;
                item.* = .{ .slot = words[0], .at = words[1], .from = words[2], .rows = words[3], .pending = if (words[4] == 0) null else @intCast(words[4] - 1) };
            }
            _ = try seats.round(d.pair, .{taps orelse return error.OutOfStep}, d.mains(), items);
        },
        else => return error.OutOfStep,
    }
}
