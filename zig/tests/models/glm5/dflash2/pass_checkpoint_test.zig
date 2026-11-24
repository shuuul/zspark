//! `pass.zig`'s test on the published drafter (spec 104 WS-06 moved it out of `pass_test.zig`): on made taps,
//! past its window, a block the same bits flat, in the ring and as a graph, and what its passes cost.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const bf16 = @import("../../../../src/weights/bf16.zig");
const safetensors = @import("../../../../src/weights/safetensors.zig");
const glm5_testing = @import("../testing.zig");
const config_file = @import("../../../../src/decode/dflash2/config.zig");
const kernels = @import("../../../../src/decode/dflash2/kernels.zig");
const pass = @import("../../../../src/models/glm5/dflash2/pass.zig");
const dflash2_pass = @import("../../../../src/decode/dflash2/pass.zig");
const weights = @import("../../../../src/decode/dflash2/weights.zig");
const pass_fixture = @import("pass_fixture.zig");
const drafts = pass_fixture.drafts;
const value = pass_fixture.value;

const world = config_file.world;

fn median(values: []f64) f64 {
    std.mem.sort(f64, values, {}, std.sort.asc(f64));
    return values[values.len / 2];
}

test "DFlash2 pass: the published drafter on made taps, past its window: a block is the same bits flat, in the ring and as a graph; what its passes cost" {
    // Runs with ZSPARK_GLM_DRAFTER set to the drafter's directory; skipped without. Both ranks' shares with the
    // four-bit copies made in memory (0.8 GiB), a table and a head of the main model's shapes holding made
    // values (2.5 GiB), no main model. The times are printed, not held: a measurement takes the machine alone.
    const path = testing.environ.getPosix("ZSPARK_GLM_DRAFTER") orelse return error.SkipZigTest;
    const io = testing.io;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.read(arena, io, dir, &diagnostic);
    var checkpoint: safetensors.Checkpoint = try .open(gpa, io, .cwd(), path);
    defer checkpoint.close();
    var pool: cuda.Scratch = .init(gpa);
    defer pool.deinit();
    const rows: cuda.Scratch.Rows = .{ .pool = &pool, .count = dflash2_pass.batch };
    var w: [world]weights.Weights = undefined;
    w[0] = try .load(arena, gpa, &checkpoint, &c, 0, .mixed, rows);
    defer w[0].deinit();
    w[1] = try .load(arena, gpa, &checkpoint, &c, 1, .mixed, rows);
    defer w[1].deinit();

    var prng: std.Random.DefaultPrng = .init(2067);
    const random = prng.random();
    const d = c.hidden;
    const part = c.vocab / world;
    // a head whose rows repeat 64 made ones, and a table with the two rows a block reads
    const some = try arena.alloc(u16, 64 * d);
    glm5_testing.randomBf16(random, some, 1.0 / @sqrt(@as(f32, @floatFromInt(d))));
    const head_host = try gpa.alloc(u16, part * d);
    defer gpa.free(head_host);
    var at: usize = 0;
    while (at < head_host.len) : (at += some.len) @memcpy(head_host[at..][0..@min(some.len, head_host.len - at)], some[0..@min(some.len, head_host.len - at)]);
    var heads: [world]bf16.Linear = undefined;
    heads[0] = try .init(std.mem.sliceAsBytes(head_host), part, d, rows);
    defer heads[0].deinit();
    heads[1] = try .init(std.mem.sliceAsBytes(head_host), part, d, rows);
    defer heads[1].deinit();
    var table: cuda.Buffer = try .init(.bf16, &.{ c.vocab, d });
    defer table.deinit();
    try table.zero();
    const pending = 1234;
    const row = try arena.alloc(u16, d);
    for ([_]usize{ pending, pending + 1, c.mask }) |token| {
        glm5_testing.randomBf16(random, row, 1.0);
        try table.uploadAt(token * d * 2, std.mem.sliceAsBytes(row));
    }
    const main: [world]dflash2_pass.Main = .{ .{ .embed = table.view, .head = .{ .bf16 = heads[0] }, .rows = heads[0].weight.view }, .{ .embed = table.view, .head = .{ .bf16 = heads[1] }, .rows = heads[1].weight.view } };

    const total = c.ringRows() + 200;
    const taps_host = try gpa.alloc(u16, total * c.tapWidth());
    defer gpa.free(taps_host);
    glm5_testing.randomBf16(random, taps_host, 1.0);
    var taps: [world]cuda.Buffer = undefined;
    taps[0] = try glm5_testing.device(u16, .bf16, &.{ total, c.tapWidth() }, taps_host);
    defer taps[0].deinit();
    taps[1] = try glm5_testing.device(u16, .bf16, &.{ total, c.tapWidth() }, taps_host);
    defer taps[1].deinit();

    var flat: pass.Pair = try .init(gpa, w, &c, total, false, part);
    defer flat.deinit();
    var ring: pass.Pair = try .init(gpa, w, &c, total, true, part);
    defer ring.deinit();
    var graphs: dflash2_pass.Graphs = try .init(gpa);
    defer graphs.deinit();
    ring.graphs = &graphs;
    try testing.expect(ring.ranks[0].ring and ring.ranks[0].cap == c.ringRows() and !flat.ranks[0].ring);
    try flat.addTaps(taps, 0, total);
    // the ring's rows in other batches, its last rows as rounds keep them
    try ring.addTaps(taps, 0, 37);
    try ring.addTaps(taps, 37, total - 37 - 12);
    for ([_]usize{ 1, 4, 7 }) |keep| try ring.addTaps(taps, ring.context_end, keep);
    try testing.expectEqual(total, ring.context_end);
    for ([_]i32{ pending, pending + 1, pending }) |token| {
        try flat.block(main, token);
        try ring.block(main, token);
        try cuda.synchronize();
        const want = try drafts(arena, flat);
        try testing.expectEqualSlices(u16, want, try drafts(arena, ring));
        // made rows, not zeros or NaN
        var finite: usize = 0;
        var distinct: usize = 0;
        for (want[0..1000], want[1..1001]) |a, b| {
            finite += @intFromBool(std.math.isFinite(value(a)));
            distinct += @intFromBool(a != b);
        }
        try testing.expect(finite == 1000 and distinct > 900);
    }
    // the ring's context kept (21 MiB a rank) and restored into a pair of its own: the same block
    {
        var kept: pass.Kept = try .take(&ring);
        defer kept.deinit();
        try testing.expectEqual(c.window.?, kept.rows);
        var fresh: pass.Pair = try .init(gpa, w, &c, total, false, part);
        defer fresh.deinit();
        try kept.restore(&fresh);
        try fresh.block(main, pending);
        try testing.expectEqualSlices(u16, try drafts(arena, flat), try drafts(arena, fresh));
    }

    // what a round pays: a block and an update of the rows it kept, eager and as graphs, and the host's
    // candidates of seven positions from both ranks' logits
    const trials = 30;
    var times: [trials]f64 = undefined;
    const Timed = struct {
        fn ms(start: std.Io.Timestamp, clock: std.Io) f64 {
            return @as(f64, @floatFromInt(start.durationTo(.now(clock, .awake)).toNanoseconds())) / 1e6;
        }
    };
    const end = ring.context_end;
    std.debug.print("DFlash2 at its shapes, both ranks in one process, context {d} in a ring of {d} rows; medians of {d}, ms\n", .{ end, c.ringRows(), trials });
    for ([_]bool{ false, true }) |graphed| {
        ring.graphs = if (graphed) &graphs else null;
        for (&times) |*one| {
            const start: std.Io.Timestamp = .now(io, .awake);
            try ring.block(main, pending);
            try cuda.synchronize();
            one.* = Timed.ms(start, io);
        }
        std.debug.print("  a block pass, {s}: {d:.2}\n", .{ if (graphed) "graph" else "eager", median(&times) });
        for ([_]usize{ 1, 4, 7 }) |keep| {
            for (&times) |*one| {
                ring.context_end = end - keep;
                const start: std.Io.Timestamp = .now(io, .awake);
                try ring.addTaps(taps, total - keep, keep);
                try cuda.synchronize();
                one.* = Timed.ms(start, io);
            }
            std.debug.print("  an update of {d} rows, {s}: {d:.2}\n", .{ keep, if (graphed) "graph" else "eager", median(&times) });
        }
    }
    // where a block's time goes: each phase of both ranks waited for
    var phases: [4]f64 = @splat(0);
    for (0..trials) |_| {
        for (&ring.ranks) |*rank| try rank.stage(end);
        for (&ring.ranks, main) |*rank, m| try dflash2_pass.embed(rank, m.embed);
        try cuda.synchronize();
        for (0..c.layers) |index| {
            var start: std.Io.Timestamp = .now(io, .awake);
            for (w, &ring.ranks, 0..) |one, *rank, r| try dflash2_pass.attention(one, rank, index, pass.Pair.partial(ring.attn, r));
            try cuda.synchronize();
            phases[0] += Timed.ms(start, io);
            start = .now(io, .awake);
            for (w, &ring.ranks, 0..) |one, *rank, r| try dflash2_pass.mlp(one, rank, index, ring.attn.view, pass.Pair.partial(ring.ffn, r));
            try cuda.synchronize();
            phases[1] += Timed.ms(start, io);
            start = .now(io, .awake);
            for (w, &ring.ranks) |one, *rank| try dflash2_pass.finish(one, rank, index, ring.ffn.view);
            try cuda.synchronize();
            phases[2] += Timed.ms(start, io);
        }
        const start: std.Io.Timestamp = .now(io, .awake);
        for (w, &ring.ranks, main) |one, *rank, m| try dflash2_pass.head(one, rank, m);
        try cuda.synchronize();
        phases[3] += Timed.ms(start, io);
    }
    std.debug.print("  of a block, means: the layers' attention {d:.2}, their MLP {d:.2}, their last convolution {d:.2}, the head and the projection {d:.2}\n", .{ phases[0] / trials, phases[1] / trials, phases[2] / trials, phases[3] / trials });
    // the head in four bits, made in memory as a loader makes it: what a block costs then
    {
        const made_start: std.Io.Timestamp = .now(io, .awake);
        var four: weights.Linear = try weights.draftHead(arena, gpa, std.mem.sliceAsBytes(head_host), d, rows);
        defer four.deinit();
        const made = Timed.ms(made_start, io);
        const light: [world]dflash2_pass.Main = .{ .{ .embed = table.view, .head = four, .rows = heads[0].weight.view }, .{ .embed = table.view, .head = four, .rows = heads[1].weight.view } };
        try testing.expect(four.bytes() * 3 < heads[0].weight.bytes());
        const stored = try glm5_testing.fetch(u16, arena, ring.ranks[0].logits);
        for (&times) |*one| {
            const start: std.Io.Timestamp = .now(io, .awake);
            for (w, &ring.ranks, light) |one_w, *rank, m| try dflash2_pass.head(one_w, rank, m);
            try cuda.synchronize();
            one.* = Timed.ms(start, io);
        }
        // four bits of made rows: near the stored head's logits, not their bits
        const near = try glm5_testing.fetch(u16, arena, ring.ranks[0].logits);
        var off: f64 = 0;
        var size: f64 = 0;
        for (stored, near) |a, b| {
            off += @abs(@as(f64, value(a)) - value(b));
            size += @abs(@as(f64, value(a)));
        }
        try testing.expect(off < 0.2 * size and !std.mem.eql(u16, stored, near));
        // its picks' values are the stored head's at the four-bit head's tokens, near: the dots are its
        // products summed in another order, beside its bf16 rounding
        const picked_values = try glm5_testing.fetch(f32, arena, ring.ranks[0].top_values);
        const picked_tokens = try glm5_testing.fetch(i32, arena, ring.ranks[0].top_tokens);
        var again: f64 = 0;
        var again_size: f64 = 0;
        for (picked_values, 0..) |got, place| {
            const want: f64 = value(stored[(place / c.top_k) * part + @as(usize, @intCast(picked_tokens[place]))]);
            again += @abs(got - want);
            again_size += @abs(want);
        }
        try testing.expect(again < 0.01 * again_size);
        std.debug.print("  the head in four bits ({d:.0} MiB a rank, made in {d:.0} ms; logits off by {d:.3} of their size): the head and the projection {d:.2}\n", .{ @as(f64, @floatFromInt(four.bytes())) / (1024.0 * 1024.0), made, off / size, median(&times) });
        var lighter = ring;
        lighter.graphs = null;
        for (&times) |*one| {
            const start: std.Io.Timestamp = .now(io, .awake);
            try lighter.block(light, pending);
            try cuda.synchronize();
            one.* = Timed.ms(start, io);
        }
        std.debug.print("  a block pass with it, eager: {d:.2}\n", .{median(&times)});
    }
    const logits = try arena.alloc(u16, part);
    var values: [world * 16]f32 = undefined;
    var tokens: [world * 16]i32 = undefined;
    var merged_tokens: [16]u32 = undefined;
    var merged_values: [16]f64 = undefined;
    const chain = @import("../../../../src/decode/dflash2/chain.zig");
    for (&times) |*one| {
        const start: std.Io.Timestamp = .now(io, .awake);
        for (0..c.block - 1) |position| {
            var halves: [world]chain.Half = undefined;
            for (&halves, ring.ranks, 0..) |*half_of, rank, r| {
                try rank.logits.downloadAt(position * part * 2, std.mem.sliceAsBytes(logits));
                chain.largest(logits, r * part, values[r * 16 ..][0..16], tokens[r * 16 ..][0..16]);
                half_of.* = .{ .values = values[r * 16 ..][0..16], .tokens = tokens[r * 16 ..][0..16] };
            }
            chain.merge(&halves, &merged_tokens, &merged_values);
        }
        one.* = Timed.ms(start, io);
    }
    std.debug.print("  seven positions' candidates on the host from the logits read back: {d:.2}\n", .{median(&times)});
    // as the drafter has them: each rank's largest picked on the device, read back and merged
    for (&times) |*one| {
        const start: std.Io.Timestamp = .now(io, .awake);
        for (&ring.ranks) |*rank| try kernels.top(rank.logits.view, rank.top_values.view, rank.top_tokens.view, rank.picks.view, rank.first_token);
        try cuda.synchronize();
        one.* = Timed.ms(start, io);
    }
    const picked = median(&times);
    var all_values: [world * 7 * 16]f32 = undefined;
    var all_tokens: [world * 7 * 16]i32 = undefined;
    try testing.expect(c.block == 8 and c.top_k == 16);
    for (&times) |*one| {
        const start: std.Io.Timestamp = .now(io, .awake);
        for (ring.ranks, 0..) |rank, r| {
            try rank.top_values.downloadAt(0, std.mem.sliceAsBytes(all_values[r * 112 ..][0..112]));
            try rank.top_tokens.downloadAt(0, std.mem.sliceAsBytes(all_tokens[r * 112 ..][0..112]));
        }
        for (0..c.block - 1) |position| {
            var halves: [world]chain.Half = undefined;
            for (&halves, 0..) |*half_of, r| half_of.* = .{ .values = all_values[r * 112 + position * 16 ..][0..16], .tokens = all_tokens[r * 112 + position * 16 ..][0..16] };
            chain.merge(&halves, &merged_tokens, &merged_values);
        }
        one.* = Timed.ms(start, io);
    }
    std.debug.print("  picked on the device: the kernel, both ranks {d:.3}; read back and merged {d:.3}\n", .{ picked, median(&times) });
    // and they are the host's of the same logits
    {
        for (ring.ranks, 0..) |rank, r| {
            try rank.logits.downloadAt(0, std.mem.sliceAsBytes(logits));
            chain.largest(logits, r * part, values[0..16], tokens[0..16]);
            try testing.expectEqualSlices(i32, tokens[0..16], all_tokens[r * 112 ..][0..16]);
            try testing.expectEqualSlices(u32, @ptrCast(values[0..16]), @ptrCast(all_values[r * 112 ..][0..16]));
        }
    }
}
