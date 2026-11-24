//! The check of a head pass's scoring (`hall_head.zig` `scoreRows`) that `hall_score_test.zig` runs on the
//! random model and `checkpoint_prefix_test.zig` on a checkpoint (spec 133 WS-02 moved it here): the seats of
//! a pass scored in one launch against each scored alone, bit for bit.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const hall_head = @import("../../../../src/models/qwen4/pool/hall_head.zig");
const hall_fixture = @import("hall_fixture.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const mtp = @import("../../../../src/models/qwen4/mtp.zig");
const Engine = hall_engine.Engine;
const Picked = mtp.Picked;

pub const seats_most = 8;
/// The rows of the head's window the test fills.
pub const window = 32;

/// A seat's scored row and what the device found in it.
pub const Scored = struct {
    row: []u16,
    raw: [Picked.words]u32,

    fn read(gpa: Allocator, engine: *Engine, slot: usize) !Scored {
        const width: usize = @intCast(engine.scored.?.view.size[1]);
        const row = try gpa.alloc(u16, width);
        errdefer gpa.free(row);
        try engine.scored.?.downloadAt(slot * width * 2, std.mem.sliceAsBytes(row));
        var raw: [Picked.words]u32 = undefined;
        try engine.picked[slot].downloadAt(0, std.mem.sliceAsBytes(&raw));
        return .{ .row = row, .raw = raw };
    }

    /// The row's bits, and the words as the host reads them: the kernel leaves its candidates in the order
    /// its threads came.
    fn expectEqual(want: Scored, got: Scored) !void {
        try testing.expectEqualSlices(u16, want.row, got.row);
        const a: Picked = .read(&want.raw);
        const b: Picked = .read(&got.raw);
        try testing.expectEqualSlices(u32, want.raw[0..8], got.raw[0..8]);
        try testing.expectEqual(a.total, b.total);
        try testing.expectEqual(a.count, b.count);
        try testing.expectEqualSlices(u16, a.keys[0..a.count], b.keys[0..b.count]);
        try testing.expectEqualSlices(u32, a.columns[0..a.count], b.columns[0..b.count]);
    }
};

/// Over `m`'s head, the draft head where it has one: passes of 1, 2, 4 and 8 seats as a chained pass lays
/// them (rows and seats one after another), as a round's first does (the rows apart) and with the seats
/// apart, each seat's row and `picked` against the row scored alone; a seat no pass scored keeps its row.
pub fn scoredAlike(gpa: Allocator, m: model_file.Model) !void {
    const context: usize = 256;
    var engine: Engine = try .open(gpa, m, &.{}, .{
        .slots = seats_most,
        .session = .{ .capacity = context, .prompt_rows = 32 },
        .pool = extents.lowerBound(context, seats_most) * extents.positions,
        .chunk = 32,
        .lazy_graphs = true,
    });
    defer engine.deinit();
    engine.bindPool();
    // seats whose samplings ask the device to different ranks
    const samplings = [_]sampling.Sampling{
        hall_fixture.greedy,
        .{ .seed = 11, .temperature = 0.9, .top_k = 20, .top_p = 1, .min_p = 0 },
        .{ .seed = 6, .temperature = 1.0, .top_k = 0, .top_p = 0.9, .min_p = 0 },
    };
    for (0..seats_most) |slot| {
        _ = try engine.begin(slot, .{ .prompt = &.{ 40, 41, @intCast(42 + slot) }, .settings = samplings[slot % samplings.len], .max_tokens = 9, .depth = engine.depth, .opening = engine.depth }, 0);
    }
    const mixed = engine.hb.?.layer.mixed;
    const hidden = m.shape.hidden;
    try testing.expect(@as(usize, @intCast(mixed.view.size[0])) >= window);
    var prng: std.Random.DefaultPrng = .init(78);
    const random = prng.random();
    const rows_host = try gpa.alloc(u16, window * hidden);
    defer gpa.free(rows_host);
    // rows of unlike sizes: a row's sums must not lean on its neighbour's
    for (0..window) |r| {
        const spread: f32 = if (r % 3 == 0) 4.0 else 0.5;
        for (rows_host[r * hidden ..][0..hidden]) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32) * spread);
    }
    try mixed.uploadAt(0, std.mem.sliceAsBytes(rows_host));

    const Lay = enum { chained, rows_apart, seats_apart };
    const apart = [seats_most]usize{ 6, 1, 4, 3, 7, 0, 5, 2 };
    for ([_]usize{ 1, 2, 4, 8 }) |n| for ([_]Lay{ .chained, .rows_apart, .seats_apart }) |lay| {
        errdefer std.debug.print("{d} seats, {t}\n", .{ n, lay });
        var rows: [seats_most]usize = undefined;
        var seats: [seats_most]usize = undefined;
        for (0..n) |i| {
            rows[i] = if (lay == .rows_apart) 3 * i + 2 else i + n;
            seats[i] = if (lay == .seats_apart) apart[i] else i;
        }
        var alone: [seats_most]Scored = undefined;
        var made: usize = 0;
        defer for (alone[0..made]) |one| gpa.free(one.row);
        for (0..n) |i| {
            try hall_head.scoreRows(&engine, rows[i .. i + 1], seats[i .. i + 1]);
            alone[i] = try .read(gpa, &engine, seats[i]);
            made += 1;
        }
        // every seat left with another row's logits, so a pass that wrote nothing is seen
        var other: [seats_most]Scored = undefined;
        var others: usize = 0;
        defer for (other[0..others]) |one| gpa.free(one.row);
        for (0..seats_most) |slot| {
            try hall_head.scoreRows(&engine, &.{window - 1 - slot}, &.{slot});
            other[slot] = try .read(gpa, &engine, slot);
            others += 1;
        }
        try hall_head.scoreRows(&engine, rows[0..n], seats[0..n]);
        var in_pass: [seats_most]bool = @splat(false);
        for (0..n) |i| {
            const got: Scored = try .read(gpa, &engine, seats[i]);
            defer gpa.free(got.row);
            try alone[i].expectEqual(got);
            try testing.expect(!std.mem.eql(u16, other[seats[i]].row, got.row));
            in_pass[seats[i]] = true;
        }
        for (0..seats_most) |slot| {
            if (in_pass[slot]) continue;
            const got: Scored = try .read(gpa, &engine, slot);
            defer gpa.free(got.row);
            try other[slot].expectEqual(got);
        }
    };
}
