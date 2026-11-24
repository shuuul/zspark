//! `attention.zig`'s selection tests (spec 104 WS-06 moved them out of `attention_test.zig`): the best blocks
//! and the unfinished one, a window's selection at once against the tiles', and its profile.

const std = @import("std");
const testing = std.testing;
const fixtures = @import("../testing.zig");
const fx = fixtures.attending.fx;
const fetch = fixtures.attending.fetch;
const randomBf16 = fixtures.attending.randomBf16;
const hostList = fixtures.attending.hostList;
const cuda = @import("../../../../src/cuda/cuda.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const Position = attention.Position;
const Selection = attention.Selection;
const pick = attention.pick;
const uploaded = attention.uploaded;

test "the selection takes the best blocks, the lower block among equal scores, and the unfinished block" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(33);
    const random = prng.random();
    const heads = 3;
    const dim = 64;
    const Case = struct { ratio: usize, top: usize, pos0: usize, rows: usize };
    const cases = [_]Case{
        // rows on both sides of the first sparse row (end 644), tails of 0 to 3 positions
        .{ .ratio = 4, .top = 160, .pos0 = 640, .rows = 9 },
        // more blocks than the kernel has threads: runs of 3 blocks a thread
        .{ .ratio = 4, .top = 5, .pos0 = 12000, .rows = 6 },
        // no tail, and a single block to take
        .{ .ratio = 1, .top = 1, .pos0 = 1500, .rows = 6 },
        // more blocks to take than threads
        .{ .ratio = 3, .top = 1100, .pos0 = 7500, .rows = 7 },
        // complete blocks across the kernel's 1,024 threads: one block a thread, then two, ragged
        .{ .ratio = 4, .top = 5, .pos0 = 4080, .rows = 24 },
    };
    var ids: std.ArrayList(i32) = .empty;
    var position: Position = try .init();
    defer position.deinit();
    for (cases) |case| {
        var shape = fx.small;
        shape.ratio = case.ratio;
        shape.budget = case.top * case.ratio;
        const context = case.pos0 + case.rows;
        const blocks = context / case.ratio;
        var selection: Selection = try .init(shape, case.rows, context + 40);
        defer selection.deinit();
        // Products built to tie: row 0 draws from four values (scores from a handful), row 1 is all equal, row 2
        // all negative (every score zero), row 3 has runs of equal scores at three levels, row 4 runs of seven
        // equal scores (across the runs of blocks a thread counts, whatever the case), the rest are continuous
        // with tied pairs planted at random blocks.
        const dots = try gpa.alloc(f32, case.rows * heads * blocks);
        for (0..case.rows) |r| for (0..heads) |h| for (0..blocks) |b| {
            const value: f32 = switch (r) {
                0 => @as(f32, @floatFromInt(random.uintLessThan(u8, 4))) - 1.0,
                1 => 0.75,
                2 => -1.0 - random.float(f32),
                3 => if (b % 7 == 3) 2.0 else if (b % 5 == 0) 1.0 else @as(f32, @floatFromInt(b % 3)) * 0.125,
                4 => @as(f32, @floatFromInt(b / 7)) * 0.25,
                else => random.floatNorm(f32),
            };
            dots[(r * heads + h) * blocks + b] = value;
        };
        for (5..case.rows) |r| for (0..40) |_| {
            const a = random.uintLessThan(usize, blocks);
            const b = random.uintLessThan(usize, blocks);
            for (0..heads) |h| dots[(r * heads + h) * blocks + b] = dots[(r * heads + h) * blocks + a];
        };
        var dots_dev = try uploaded(f32, .f32, &.{ case.rows * heads, blocks }, dots);
        defer dots_dev.deinit();
        try position.set(case.pos0);
        try pick(shape, selection, dots_dev.view, position.device.view, 0, case.rows, heads, dim, null);
        const got_ids = try fetch(i32, gpa, selection.ids);
        const got_nk = try fetch(i32, gpa, selection.nk);
        const got_sparse = try fetch(i32, gpa, selection.sparse);
        const got_scores = try fetch(f32, gpa, selection.scores);

        const scores = try gpa.alloc(f32, blocks);
        var sparse_rows: usize = 0;
        for (0..case.rows) |r| {
            const end = case.pos0 + r + 1;
            for (scores, 0..) |*s, b| {
                var total: f32 = 0;
                for (0..heads) |h| total = total + @max(dots[(r * heads + h) * blocks + b], 0.0);
                s.* = total / @sqrt(@as(f32, dim));
            }
            const sparse = try hostList(gpa, scores, end, case.ratio, case.top, &ids);
            try testing.expectEqual(@as(i32, @intFromBool(sparse)), got_sparse[r]);
            if (!sparse) {
                try testing.expectEqual(@as(i32, @intCast(end)), got_nk[r]);
                continue;
            }
            sparse_rows += 1;
            try testing.expectEqualSlices(u32, @ptrCast(scores[0 .. end / case.ratio]), @ptrCast(got_scores[r * selection.blocks ..][0 .. end / case.ratio]));
            try testing.expectEqual(@as(i32, @intCast(ids.items.len)), got_nk[r]);
            try testing.expectEqualSlices(i32, ids.items, got_ids[r * shape.listWidth() ..][0..ids.items.len]);
        }
        try testing.expectEqual(if (case.top == 160) @as(usize, 6) else case.rows, sparse_rows);
    }
}

test "a window's selection at once gives the tiles' scores and lists, a row alone as among others" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(95);
    const random = prng.random();
    const Case = struct { heads: usize, dim: usize, budget: usize, pos0: usize, rows: usize };
    const cases = [_]Case{
        // the test shape across the first sparse row (end 644), several row tiles, ragged
        .{ .heads = 2, .dim = 64, .budget = 640, .pos0 = 600, .rows = 300 },
        // Flash Next's indexer (4 heads of 128, 512 blocks): across its first sparse row, one row tile
        .{ .heads = 4, .dim = 128, .budget = 2048, .pos0 = 2040, .rows = 64 },
        // a long context, blocks ragged against the kernel's 64, rows ragged against its 64 and the tiles' 32
        .{ .heads = 4, .dim = 128, .budget = 2048, .pos0 = 16000, .rows = 201 },
        // a prompt window's shape (2,048 rows, 16k context), where the selection's block a row grinds the most
        .{ .heads = 4, .dim = 128, .budget = 2048, .pos0 = 16000, .rows = 2048 },
        .{ .heads = 2, .dim = 64, .budget = 640, .pos0 = 5003, .rows = 33 },
        // every row dense: nothing scored
        .{ .heads = 4, .dim = 128, .budget = 2048, .pos0 = 0, .rows = 40 },
    };
    var position: Position = try .init();
    defer position.deinit();
    const kept = attention.whole_rows;
    defer attention.whole_rows = kept;
    for (cases) |case| {
        var shape = fx.small;
        shape.index_heads = case.heads;
        shape.index_dim = case.dim;
        shape.budget = case.budget;
        const context = case.pos0 + case.rows;
        const blocks = context / shape.ratio;
        var selection: Selection = try .init(shape, case.rows, context + 40);
        defer selection.deinit();
        var iq = try uploaded(u16, .bf16, &.{ case.rows, case.heads, case.dim }, try randomBf16(gpa, random, case.rows * case.heads * case.dim, 1.0));
        defer iq.deinit();
        var pooled = try uploaded(u16, .bf16, &.{ selection.blocks, case.dim }, try randomBf16(gpa, random, selection.blocks * case.dim, 1.0));
        defer pooled.deinit();

        const Lists = struct { ids: []i32, nk: []i32, sparse: []i32, scores: []f32 };
        var lists: [2]Lists = undefined;
        for (0..2) |whole| {
            attention.whole_rows = if (whole == 1) 1 else 0;
            for ([_]*cuda.Buffer{ &selection.ids, &selection.nk, &selection.sparse, &selection.scores }) |buffer| try buffer.zero();
            try position.set(case.pos0);
            const before = attention.whole_selections;
            try attention.select(shape, selection, iq.view, pooled.view, position.device.view, 0, case.rows, context, null);
            try testing.expectEqual(before + whole, attention.whole_selections);
            lists[whole] = .{ .ids = try fetch(i32, gpa, selection.ids), .nk = try fetch(i32, gpa, selection.nk), .sparse = try fetch(i32, gpa, selection.sparse), .scores = try fetch(f32, gpa, selection.scores) };
        }
        try testing.expectEqualSlices(i32, lists[0].ids, lists[1].ids);
        try testing.expectEqualSlices(i32, lists[0].nk, lists[1].nk);
        try testing.expectEqualSlices(i32, lists[0].sparse, lists[1].sparse);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(lists[0].scores), std.mem.sliceAsBytes(lists[1].scores));
        const all = lists[1];

        // behind a table of scattered extents, as a pool's round selects (zs_qwen4_attn_select_window_table, spec
        // 149 WS-02): the same lists, each key the pool's row of its position
        {
            const count = (context + 511) / 512;
            const places = try gpa.alloc(i32, 2 * count + 1);
            for (places, 0..) |*place, i| place.* = @intCast(i);
            random.shuffle(i32, places);
            var table = try uploaded(i32, .i32, &.{count}, places[0..count]);
            defer table.deinit();
            attention.whole_rows = 1;
            for ([_]*cuda.Buffer{ &selection.ids, &selection.nk, &selection.sparse, &selection.scores }) |buffer| try buffer.zero();
            try position.set(case.pos0);
            const before = attention.whole_selections;
            try attention.select(shape, selection, iq.view, pooled.view, position.device.view, 0, case.rows, context, table.view);
            try testing.expectEqual(before + 1, attention.whole_selections);
            try testing.expectEqualSlices(i32, all.nk, try fetch(i32, gpa, selection.nk));
            try testing.expectEqualSlices(i32, all.sparse, try fetch(i32, gpa, selection.sparse));
            const ids = try fetch(i32, gpa, selection.ids);
            const width = shape.listWidth();
            for (0..case.rows) |r| {
                if (all.sparse[r] == 0) continue;
                const n: usize = @intCast(all.nk[r]);
                for (all.ids[r * width ..][0..n], ids[r * width ..][0..n]) |pos, stored| {
                    const at: usize = @intCast(pos);
                    try testing.expectEqual(@as(i32, @intCast(@as(usize, @intCast(places[at / 512])) * 512 + at % 512)), stored);
                }
            }
        }
        var any: usize = 0;
        for (all.sparse[0..case.rows]) |s| any += @intCast(s);
        try testing.expect((any > 0) == (blocks > shape.top()));

        // rows alone, each a window of its own at its position, by the tiles
        attention.whole_rows = 0;
        const width = shape.listWidth();
        for ([_]usize{ 0, case.rows / 2, case.rows - 1 }) |r| {
            if ((case.pos0 + r + 1) / shape.ratio == 0) continue;
            for ([_]*cuda.Buffer{ &selection.ids, &selection.nk, &selection.sparse }) |buffer| try buffer.zero();
            try position.set(case.pos0 + r);
            const one: cuda.View = .init(@as([*]u8, @ptrCast(iq.view.ptr.?)) + r * case.heads * case.dim * 2, .bf16, &.{ 1, case.heads, case.dim });
            try attention.select(shape, selection, one, pooled.view, position.device.view, 0, 1, case.pos0 + r + 1, null);
            const ids = try fetch(i32, gpa, selection.ids);
            const nk = try fetch(i32, gpa, selection.nk);
            const sparse = try fetch(i32, gpa, selection.sparse);
            try testing.expectEqual(all.nk[r], nk[0]);
            try testing.expectEqual(all.sparse[r], sparse[0]);
            const n: usize = @intCast(nk[0]);
            if (sparse[0] == 1) try testing.expectEqualSlices(i32, all.ids[r * width ..][0..n], ids[0..n]);
        }
    }
}

test "PROFILE a window's selection by tiles or at once" {
    // ZSPARK_SELECT_PROFILE=FORM:ROWS:CONTEXT, FORM tiles or whole: Flash Next's indexer (4 heads of 128, ratio 4,
    // 512 blocks) on random rows, `passes` launches of the form for a profiler. Unset, skipped.
    const asked = std.c.getenv("ZSPARK_SELECT_PROFILE") orelse return error.SkipZigTest;
    var it = std.mem.splitScalar(u8, std.mem.span(asked), ':');
    const form = it.next() orelse return error.InvalidArgument;
    const rows = try std.fmt.parseInt(usize, it.next() orelse return error.InvalidArgument, 10);
    const context = try std.fmt.parseInt(usize, it.next() orelse return error.InvalidArgument, 10);
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(96);
    const random = prng.random();
    var shape = fx.small;
    shape.index_heads = 4;
    shape.index_dim = 128;
    shape.budget = 2048;
    var selection: Selection = try .init(shape, rows, context);
    defer selection.deinit();
    var iq = try uploaded(u16, .bf16, &.{ rows, 4, 128 }, try randomBf16(gpa, random, rows * 4 * 128, 1.0));
    defer iq.deinit();
    var pooled = try uploaded(u16, .bf16, &.{ selection.blocks, 128 }, try randomBf16(gpa, random, selection.blocks * 128, 1.0));
    defer pooled.deinit();
    var position: Position = try .init();
    defer position.deinit();
    try position.set(context - rows);
    const kept = attention.whole_rows;
    defer attention.whole_rows = kept;
    attention.whole_rows = if (std.mem.eql(u8, form, "whole")) 1 else 0;
    for (0..4) |_| try attention.select(shape, selection, iq.view, pooled.view, position.device.view, 0, rows, context, null);
    try cuda.synchronize();
    std.debug.print("PROFILE select {s} at {d} rows, context {d}\n", .{ form, rows, context });
}
