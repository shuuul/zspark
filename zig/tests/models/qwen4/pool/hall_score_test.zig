//! The pool's scoring of a head pass (`hall_head.zig` `scoreRows`): the seats of a pass scored in one launch
//! of the projection against each scored alone, bit for bit, on the random model (`hall_score_fixture.zig`'s check;
//! on a checkpoint, `checkpoint_prefix_test.zig`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const fixtures = @import("../testing.zig");
const testing = std.testing;
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const hall_head = @import("../../../../src/models/qwen4/pool/hall_head.zig");
const hall_fixture = @import("hall_fixture.zig");
const mtp = @import("../../../../src/models/qwen4/mtp.zig");
const Engine = hall_engine.Engine;
const Picked = mtp.Picked;

const score_fixture = @import("hall_score_fixture.zig");
const seats_most = score_fixture.seats_most;
const window = score_fixture.window;
const Scored = score_fixture.Scored;
const scoredAlike = score_fixture.scoredAlike;

test "a pass's seats scored in one launch are each the row scored alone, bit for bit, at 1, 2, 4 and 8 seats, by the draft head and by the model's" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    for ([_]bool{ true, false }) |own| {
        const ids: ?[]const u32 = if (own) ids: {
            const half = try arena.alloc(u32, 160);
            for (half, 0..) |*id, i| id.* = @intCast(2 * i);
            break :ids half;
        } else null;
        var m = try fixtures.random_model.testModelRows(gpa, arena, ids, seats_most * 16 + 32);
        defer m.deinit();
        try scoredAlike(gpa, m);
    }
}
