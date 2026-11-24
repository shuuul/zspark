//! `slots.zig`'s tests on the random GLM in loopback (`testing.zig`'s `Fixture`): sequences that share a
//! window against each of them alone in a `ranks_forward.zig` state of its own, bit for bit: every scored row's logits,
//! and what each keeps when it ends (the KDA states, the conv's rows, the latents).

const std = @import("std");
const testing = std.testing;
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const random_file = @import("../../../../src/models/glm5/checks/random.zig");
const shared = @import("../testing.zig");
const slots_file = @import("../../../../src/models/glm5/pool/slots.zig");
const held_file = @import("../../../../src/models/glm5/layers/ranks_forward.zig");
const forward = @import("../../../../src/models/glm5/layers/forward.zig");
const slots_fixture = @import("slots_fixture.zig");

const fx = random_file.fx;
const Slots = slots_file.SlotsOf(layout.world);
const Run = slots_fixture.Run;
const Sequence = slots_fixture.Sequence;
const alone = slots_fixture.alone;
const among = slots_fixture.among;
const keptRows = slots_fixture.keptRows;

test "sequences that share a window give each one's bits alone: logits, KDA states, conv rows and latents, windows cut short and the wide chain among them" {
    try shareWindows(false);
}

test "the glue split by rows over the ranks in a pool's rounds gives each sequence's bits alone (spec 107 WS-03)" {
    const saved = held_file.split_rows;
    defer held_file.split_rows = saved;
    held_file.split_rows = 17;
    try shareWindows(false);
    // bf16 partials in the quantized windows (FP4-Local's, spec 116 WS-04): a round of both forms rounds its
    // quantized rows' fp32 partials, so a sequence among others is its bits alone
    try shareWindows(true);
}

fn shareWindows(bf16_partials: bool) !void {
    // without the card's forms, every window one form, a step's (bf16 rows): the rounds take their parts in any order
    const rows_was = forward.bf16_rows;
    if (!bf16_partials) forward.bf16_rows = fx.max_rows;
    defer forward.bf16_rows = rows_was;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var fixture: shared.Fixture = undefined;
    try fixture.init(0x5107, 400);
    defer fixture.deinit();
    const capacity = 96;
    var loop = try fixture.loop(capacity);
    defer loop.deinit();
    for (&loop.m.ranks) |*rank| rank.bf16_partials = bf16_partials;
    var slots: Slots = try .init(testing.allocator, loop.m, 4, capacity, fx.max_rows);
    defer slots.deinit();

    // a prompt in chunks and steps of one; drafted windows cut short; a window past the wide chain's 64 rows
    // beside small ones; a sequence that takes the slot another has left
    const plans = [_][]const [2]usize{
        &.{ .{ 9, 9 }, .{ 1, 1 }, .{ 1, 1 }, .{ 4, 2 }, .{ 3, 3 }, .{ 2, 1 }, .{ 1, 1 } },
        &.{ .{ 5, 5 }, .{ 6, 1 }, .{ 6, 4 }, .{ 3, 3 }, .{ 1, 1 }, .{ 5, 2 }, .{ 2, 2 }, .{ 7, 7 }, .{ 1, 1 } },
        &.{ .{ 2, 2 }, .{ 64, 64 }, .{ 4, 3 }, .{ 1, 1 } },
        &.{ .{ 1, 1 }, .{ 1, 1 }, .{ 8, 1 }, .{ 2, 2 }, .{ 1, 1 } },
        &.{ .{ 3, 3 }, .{ 2, 1 }, .{ 2, 2 }, .{ 1, 1 } },
    };
    const slot_of = [_]usize{ 2, 0, 3, 1, 2 };
    const starts = [_]usize{ 0, 0, 1, 2, 8 };
    const scores = [_]slots_file.Score{ .all, .last, .all, .none, .last };
    var all: [plans.len]Sequence = undefined;
    var want: [plans.len]Run = undefined;
    var from: usize = 0;
    for (&all, &want, plans, slot_of, starts, scores) |*one, *run, plan, slot, start, score| {
        // a window cut short reads ids past those kept: the next kept ones or the ids after them, in both
        const rows = keptRows(plan) + 16;
        one.* = .{ .tokens = fixture.tokens[from..][0..rows], .plan = plan, .slot = slot, .start = start, .score = score };
        from += rows;
        run.* = try alone(arena, &loop, fx.vocab, one.*);
    }
    const rounds = try slots_fixture.amongBy(arena, &loop, &slots, fx.vocab, &all, &want, bf16_partials);
    try testing.expect(rounds >= 12);
    // the sequences are not each other's: two that start alike part
    try testing.expect(!std.mem.eql(u16, want[0].logits[0..fx.vocab], want[1].logits[0..fx.vocab]));
}

test "a slot alone is the model's forward: one sequence through a pool of one, every row count's kernels" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var fixture: shared.Fixture = undefined;
    try fixture.init(0x5108, 200);
    defer fixture.deinit();
    var loop = try fixture.loop(160);
    defer loop.deinit();
    var slots: Slots = try .init(testing.allocator, loop.m, 1, 160, fx.max_rows);
    defer slots.deinit();
    const plan: []const [2]usize = &.{ .{ 80, 80 }, .{ 1, 1 }, .{ 16, 5 }, .{ 33, 33 }, .{ 17, 2 }, .{ 2, 2 } };
    const one: Sequence = .{ .tokens = fixture.tokens[0 .. keptRows(plan) + 20], .plan = plan, .slot = 0 };
    const want = try alone(arena, &loop, fx.vocab, one);
    try testing.expectEqual(plan.len, try among(arena, &loop, &slots, fx.vocab, &.{one}, &.{want}));
}

test "pool graph keys preserve slot, rows, scoring, parity, bucket and chunk phase, and a repeated round adds no graph" {
    const gpa = testing.allocator;
    var fixture: shared.Fixture = undefined;
    try fixture.init(0x5156, 80);
    defer fixture.deinit();
    var loop = try fixture.loop(160);
    defer loop.deinit();
    var slots: Slots = try .init(gpa, loop.m, 2, 160, fx.max_rows);
    defer slots.deinit();
    var graphs: Slots.Graphs = try .init(gpa);
    defer graphs.deinit();
    const part = [_]slots_file.Part{.{ .slot = 0, .rows = 1 }};
    const base = Slots.Graphs.key(&slots, &part, 1);
    const want = try gpa.alloc(u16, fx.vocab);
    defer gpa.free(want);
    const got = try gpa.alloc(u16, fx.vocab);
    defer gpa.free(got);
    try slots.forward(loop.m, &loop.b, &part, fixture.tokens[0..1]);
    try slots.logitsRow(loop.b, 0, want);
    for (0..2) |_| {
        try graphs.forward(&slots, loop.m, &loop.b, &part, fixture.tokens[0..1]);
        try slots.logitsRow(loop.b, 0, got);
        try testing.expectEqualSlices(u16, want, got);
        try testing.expectEqual(1, graphs.cache.count());
        try testing.expect(graphs.cache.contains(base));
        try testing.expect(graphs.holds(&slots, &part, 1));
    }
    try testing.expect(base.shape != Slots.Graphs.key(&slots, &.{.{ .slot = 1, .rows = 1 }}, 1).shape);
    try testing.expect(base.shape != Slots.Graphs.key(&slots, &.{.{ .slot = 0, .rows = 2 }}, 2).shape);
    try testing.expect(base.shape != Slots.Graphs.key(&slots, &.{.{ .slot = 0, .rows = 1, .score = .none }}, 1).shape);
    slots.seqs[0][0].kda.?.cur ^= 1;
    try testing.expect(base.shape != Slots.Graphs.key(&slots, &part, 1).shape);
    slots.seqs[0][0].kda.?.cur ^= 1;
    // Only key prediction reads these synthetic positions; the allocated storage stays at 160 rows.
    slots.capacity = 16384;
    const shallow = Slots.Graphs.key(&slots, &part, 1);
    slots.seqs[0][0].pos = 8192;
    try testing.expect(shallow.shape != Slots.Graphs.key(&slots, &part, 1).shape);
    slots.seqs[0][0].pos = 0;
    const chunk = [_]slots_file.Part{.{ .slot = 0, .rows = 49 }};
    const aligned = Slots.Graphs.key(&slots, &chunk, 49);
    slots.seqs[0][0].pos = 1;
    try testing.expect(aligned.shape != Slots.Graphs.key(&slots, &chunk, 49).shape);
    slots.seqs[0][0].pos = 0;
    slots.capacity = 160;
    try testing.expectEqual(base, Slots.Graphs.key(&slots, &part, 1));
}
