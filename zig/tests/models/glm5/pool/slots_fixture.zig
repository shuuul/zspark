//! What the tests of sequences sharing a window share (`slots_test.zig` on the random model,
//! `hall_pack_test.zig` on the pack): a sequence alone in a `ranks_forward.zig` state of its own (`alone`), and the
//! sequences through a `Slots` held to those bit for bit (`among`): every scored row's logits, and what each
//! keeps when it ends (the KDA states, the conv's rows, the latents).

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const dsa = @import("../../../../src/models/glm5/layers/dsa.zig");
const kda = @import("../../../../src/models/glm5/layers/kda.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const loopback = @import("../../../../src/models/glm5/layers/loopback.zig");
const slots_file = @import("../../../../src/models/glm5/pool/slots.zig");
const phases = @import("../../../../src/models/glm5/layers/forward.zig");

const world = layout.world;
const Slots = slots_file.SlotsOf(world);
const Part = slots_file.Part;

/// What a sequence leaves: its rows' logits, bf16 [total, vocab], and each rank's committed KDA states, conv
/// rows and latents.
pub const Run = struct {
    logits: []u16,
    states: [world][]u32,
    convs: [world][]u16,
    latents: [world][]u16,
};

/// A sequence in a test: its ids, its windows as (rows, rows kept), the slot it takes and the round it starts.
pub const Sequence = struct {
    tokens: []const i32,
    plan: []const [2]usize,
    slot: usize,
    start: usize = 0,
    /// How its windows are scored among the others: a window whose rows are all kept may score its last alone,
    /// or none.
    score: slots_file.Score = .all,
};

fn kept(arena: Allocator, chains: anytype, conv: cuda.Buffer, cache: dsa.Cache, positions: usize, rank: usize, run: *Run) !void {
    const one = chains.heads * kda.head_dim * kda.head_dim;
    run.states[rank] = try arena.alloc(u32, chains.layers * one);
    for (0..chains.layers) |index| try cuda.download(chains.committed(index), std.mem.sliceAsBytes(run.states[rank][index * one ..][0..one]));
    run.convs[rank] = try arena.alloc(u16, conv.view.count());
    try conv.download(std.mem.sliceAsBytes(run.convs[rank]));
    run.latents[rank] = try arena.alloc(u16, cache.layers * positions * dsa.latent);
    const ids = try logicalExtents(arena, cache);
    for (0..cache.layers) |index| {
        const plane = cache.layer(index);
        const base: [*]u8 = @ptrCast(plane.ptr.?);
        var pos: usize = 0;
        while (pos < positions) {
            const physical = if (ids.len == 0) pos else @as(usize, @intCast(ids[pos >> 9])) * dsa.chunk + (pos & (dsa.chunk - 1));
            const n = @min(dsa.chunk - (pos & (dsa.chunk - 1)), positions - pos);
            const view = cuda.View.init(@ptrCast(base + physical * dsa.latent * 2), .bf16, &.{ n, dsa.latent });
            const at = index * positions * dsa.latent + pos * dsa.latent;
            try cuda.download(view, std.mem.sliceAsBytes(run.latents[rank][at..][0 .. n * dsa.latent]));
            pos += n;
        }
    }
}

fn logicalExtents(arena: Allocator, cache: dsa.Cache) ![]const i32 {
    const table = cache.extents() orelse return &.{};
    const ids = try arena.alloc(i32, table.count());
    try cuda.download(table, std.mem.sliceAsBytes(ids));
    return ids;
}

/// The sequence alone, from an empty state: `ranks_forward.zig`'s forward and commit.
pub fn alone(arena: Allocator, loop: anytype, vocab: usize, one: Sequence) !Run {
    const half = vocab / world;
    var run: Run = undefined;
    run.logits = try arena.alloc(u16, keptRows(one.plan) * vocab);
    try loop.state.reset();
    var at: usize = 0;
    for (one.plan) |step| {
        const window = one.tokens[at..][0..step[0]];
        try loopback.forward(loop.m, &loop.state, &loop.b, window);
        try cuda.synchronize();
        for (loop.b.ranks, 0..) |b, rank| for (0..step[1]) |r|
            try b.logits.downloadAt(r * half * 2, std.mem.sliceAsBytes(run.logits[(at + r) * vocab + rank * half ..][0..half]));
        try loopback.commit(loop.m, &loop.state, loop.b, window, step[1]);
        at += step[1];
    }
    try testing.expectEqual(keptRows(one.plan), at);
    try cuda.synchronize();
    for (loop.state.ranks, 0..) |state, rank| try kept(arena, state.kda.?, state.kda.?.conv, state.dsa.?, at, rank, &run);
    return run;
}

/// The sequences through `slots`, a round a step of each that has one left, the parts' order turning with the
/// round; each held to `want` as it goes and when it ends. Returns the rounds run.
pub fn among(arena: Allocator, loop: anytype, slots: *Slots, vocab: usize, all: []const Sequence, want: []const Run) !usize {
    return amongBy(arena, loop, slots, vocab, all, want, false);
}

/// `among` with `forms_first`: a round's parts of up to `forward.bf16_rows` rows before the others, as `hall.zig`
/// orders them, so that each part takes its own rows' form (spec 115) and a sequence's bits are its own alone.
pub fn amongBy(arena: Allocator, loop: anytype, slots: *Slots, vocab: usize, all: []const Sequence, want: []const Run, forms_first: bool) !usize {
    const steps = try arena.alloc(usize, all.len);
    const ats = try arena.alloc(usize, all.len);
    @memset(steps, 0);
    @memset(ats, 0);
    const parts = try arena.alloc(Part, all.len);
    const whose = try arena.alloc(usize, all.len);
    const keeps = try arena.alloc(usize, all.len);
    const tokens = try arena.alloc(i32, loop.m.max_rows);
    const row = try arena.alloc(u16, vocab);
    var round: usize = 0;
    while (true) : (round += 1) {
        var count: usize = 0;
        var rows: usize = 0;
        var left = false;
        for (0..@as(usize, if (forms_first) 2 else 1)) |pass| for (0..all.len) |turn| {
            const index = (turn + round) % all.len;
            const one = all[index];
            if (steps[index] == one.plan.len) continue;
            left = true;
            if (round < one.start) continue;
            const step = one.plan[steps[index]];
            if (forms_first and (step[0] > phases.bf16_rows) != (pass == 1)) continue;
            if (steps[index] == 0) try slots.reset(loop.m, one.slot);
            try testing.expectEqual(ats[index], slots.pos(one.slot));
            parts[count] = .{ .slot = one.slot, .rows = step[0], .score = if (step[0] == step[1]) one.score else .all };
            @memcpy(tokens[rows..][0..step[0]], one.tokens[ats[index]..][0..step[0]]);
            whose[count] = index;
            keeps[count] = step[1];
            rows += step[0];
            count += 1;
        };
        if (!left) return round;
        if (count == 0) continue;
        try slots.forward(loop.m, &loop.b, parts[0..count], tokens[0..rows]);
        for (parts[0..count], whose[0..count], keeps[0..count], 0..) |part, index, keep, at| {
            const scored = slots.scoredAt(at);
            const from: usize = if (part.score == .last) keep - 1 else 0;
            for (from..if (part.score == .none) from else keep) |r| {
                try slots.logitsRow(loop.b, scored + r - from, row);
                try testing.expectEqualSlices(u16, want[index].logits[(ats[index] + r) * vocab ..][0..vocab], row);
            }
        }
        try slots.commit(loop.m, keeps[0..count]);
        for (whose[0..count], keeps[0..count]) |index, keep| {
            ats[index] += keep;
            steps[index] += 1;
            if (steps[index] < all[index].plan.len) continue;
            // the sequence has ended: what its slot keeps is what it keeps alone
            try cuda.synchronize();
            var got: Run = undefined;
            for (slots.seqs, 0..) |seqs, rank| {
                const seq = seqs[all[index].slot];
                try kept(arena, seq.kda.?, seq.kda.?.conv, seq.dsa.?, ats[index], rank, &got);
                try testing.expectEqualSlices(u32, want[index].states[rank], got.states[rank]);
                try testing.expectEqualSlices(u16, want[index].convs[rank], got.convs[rank]);
                try testing.expectEqualSlices(u16, want[index].latents[rank], got.latents[rank]);
            }
        }
    }
}

pub fn keptRows(plan: []const [2]usize) usize {
    var sum: usize = 0;
    for (plan) |step| sum += step[1];
    return sum;
}
