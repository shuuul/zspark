//! `hall_ranks.zig`'s tests: the pool's engine on one rank of two, the share alone in this process (the other
//! rank's partial left at zero, the add still run, as `ranks_forward.zig`'s `Model.alone`; a row of logits
//! this rank's half and zeros), each rank in turn. Four streams served together leave each slot as the same
//! stream served alone leaves it, bit for bit, and give the same tokens (`checks/ranks_hall.zig`): the random
//! shares with an attention layer in each storage. Both ranks meeting over NCCL, rank 1 following, is
//! `ranks-check ... hall`'s. The MTP head none, whole on rank 0 (rank 1 none) or split, drafting each stream:
//! the drafts as many and as many kept (the random head pays no costs) where the depth cap is off. Together, each goes on from the states
//! its run alone kept. The rounds of replies launched kernel by kernel, and replaying graphs.

const std = @import("std");
const testing = std.testing;
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const random = @import("../../../../src/models/qwen4/checks/random.zig");
const check = @import("../../../../src/models/qwen4/checks/ranks_hall.zig");
const streams_check = @import("../../../../src/models/qwen4/checks/ranks_streams.zig");
const hall_ranks = @import("../../../../src/models/qwen4/pool/hall_ranks.zig");
const ranks = @import("../../../../src/models/qwen4/ranks_forward.zig");
const One = ranks.Held(1);

const storages = [_]?attention.Storage{ null, .bf16, .fp8, .fp8_nvfp4 };

test "the pool on a rank of two, the random shares with an attention layer in each storage and the head none, whole or split: four streams together leave each slot as the stream alone does, and give its tokens and drafts, going on from kept states" {
    const gpa = testing.allocator;
    for ([_]random.Head{ .none, .whole, .split }) |head| for (storages) |kv| for (0..2) |rank| for ([_]bool{ false, true }) |eager| {
        errdefer std.debug.print("head {t}, kv {?t}, rank {d}, eager {}\n", .{ head, kv, rank, eager });
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const built = try random.sharesOf(1, gpa, arena, 42, rank, check.rows, .{ .head = head, .caches = kv });
        var m: One.Model = .{ .ranks = built, .first = rank, .max_rows = check.rows, .alone = true };
        defer m.deinit();
        var link: hall_ranks.Link = undefined;
        var engine = try check.open(gpa, m, &link);
        defer engine.deinit();
        defer link.deinit();
        engine.bindPool();
        link.bind(&engine);
        engine.eager = eager;
        // the rounds replaying graphs take the depth cap of two ranks (spec 144 WS-04: the four together chain
        // at most 4), the eager rounds none, so that their drafts are held to the streams' alone
        engine.depth_cap = if (eager) .never else .two_ranks;
        // rank 0's load captures the first graphs where it was not told to leave them
        if (!eager and rank == 0) try link.captureFirst();
        const at_load = engine.graphs.cache.count();
        var counted: [3]u64 = undefined;
        try testing.expectEqual(null, try check.compareCounting(&engine, try streams_check.prompts(arena, 50), &counted));
        // a head drafts: whole on rank 0 alone
        const drafting = head == .split or (head == .whole and rank == 0);
        try testing.expect((counted[0] > 0) == drafting);
        // nothing cut where the cap is off or no head chains (how many rounds of three seats the four make
        // depends on the states kept: `four seats each sampled` holds a cut where it surely engages)
        if (eager or !drafting) try testing.expectEqual(0, engine.capped);
        // together, each stream went on from the states its run alone kept
        try testing.expect(counted[2] > 0);
        // the load's graphs stood before any round ran, and the rounds of replies replayed graphs, captured
        // as they were first needed where the load left them
        try testing.expect((at_load > 0) == (!eager and rank == 0));
        try testing.expect((engine.graphs.cache.count() > 0) != eager);
    };
}
