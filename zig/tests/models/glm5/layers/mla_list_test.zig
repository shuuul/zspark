//! Sparse attention over a list (`mla.cu`'s chunk kernel, a listed row's branch): a window against the host,
//! a row alone, and the same lists through a swapped extent table. A row with count 0 is a dense row.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const dsa = @import("../../../../src/models/glm5/layers/dsa.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const shared = @import("../testing.zig");

const latent = dsa.latent;
const heads = 4;
const rows = 6;
const width = 700;
const extent = 512;
const extents = 2;
const scale: f64 = 0.0625;

fn scatter(dst: []u16, table: []const i32, logical: []const u16) void {
    @memset(dst, 0);
    const positions = logical.len / latent;
    for (0..positions) |pos| {
        const physical: usize = @intCast(table[pos / extent]);
        const at = (physical * extent + pos % extent) * latent;
        @memcpy(dst[at..][0..latent], logical[pos * latent ..][0..latent]);
    }
}

test "sparse attention over a list is the serial rows, through a swapped extent table" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(46);
    const random = prng.random();

    const positions = 1000;
    const logical = try arena.alloc(u16, positions * latent);
    const qa = try arena.alloc(u16, rows * heads * latent);
    shared.randomBf16(random, logical, 0.3);
    shared.randomBf16(random, qa, 0.3);
    const table = [_]i32{ 1, 0 };
    const physical = try arena.alloc(u16, extents * extent * latent);
    scatter(physical, &table, logical);

    var tokens: [rows * width]i32 = @splat(-1);
    // within a tile, across tiles, across a chunk of 512
    const counts = [_]i32{ 0, 5, 700, 1, 33, 513 };
    for (0..rows) |r| {
        var used: [positions]bool = undefined;
        @memset(&used, false);
        for (0..@as(usize, @intCast(counts[r]))) |t| {
            var tok: usize = (r * 17 + t * 29) % positions;
            while (used[tok]) tok = (tok + 1) % positions;
            used[tok] = true;
            tokens[r * width + t] = @intCast(tok);
        }
    }

    var cache = try shared.device(u16, .bf16, &.{ extents * extent, latent }, physical);
    defer cache.deinit();
    var qa_dev = try shared.device(u16, .bf16, &.{ rows, heads, latent }, qa);
    defer qa_dev.deinit();
    var table_dev = try shared.device(i32, .i32, &.{extents}, &table);
    defer table_dev.deinit();
    var tok_dev = try shared.device(i32, .i32, &.{ rows, width }, &tokens);
    defer tok_dev.deinit();
    var count_dev = try shared.device(i32, .i32, &.{rows}, &counts);
    defer count_dev.deinit();
    var out = try cuda.Buffer.init(.bf16, &.{ rows, heads, latent });
    defer out.deinit();
    var po = try cuda.Buffer.init(.f32, &.{ rows, extents, heads, latent });
    defer po.deinit();
    var pm = try cuda.Buffer.init(.f32, &.{ rows, extents, heads });
    defer pm.deinit();
    var pl = try cuda.Buffer.init(.f32, &.{ rows, extents, heads });
    defer pl.deinit();
    const parts = [3]cuda.View{ po.view, pm.view, pl.view };
    const at: i32 = positions - rows;
    var pos = try shared.device(i32, .i32, &.{1}, &.{at});
    defer pos.deinit();

    try dsa.attendRows(qa_dev.view, cache.view, pos.view, &table_dev.view, parts, out.view, .{ tok_dev.view, count_dev.view }, scale, positions);
    const got = try shared.fetch(u16, arena, out);
    // the window form (spec 116 WS-01) through the table: the same bits
    dsa.setWindow(1);
    defer dsa.setWindow(dsa.window_rows);
    try dsa.attendRows(qa_dev.view, cache.view, pos.view, &table_dev.view, parts, out.view, .{ tok_dev.view, count_dev.view }, scale, positions);
    try testing.expectEqualSlices(u16, got, try shared.fetch(u16, arena, out));

    // a listed row against the host: the list's latents in list order, fp64
    for (0..rows) |r| {
        const n: usize = @intCast(counts[r]);
        if (n == 0) continue;
        for (0..heads) |h| {
            const q = qa[(r * heads + h) * latent ..][0..latent];
            var score: [width]f64 = undefined;
            var top: f64 = -std.math.inf(f64);
            for (0..n) |t| {
                const key = logical[@as(usize, @intCast(tokens[r * width + t])) * latent ..][0..latent];
                var dot: f64 = 0;
                for (q, key) |a, b| dot += @as(f64, numeric.f32FromBf16(a)) * numeric.f32FromBf16(b);
                score[t] = dot * scale;
                top = @max(top, score[t]);
            }
            var sum: f64 = 0;
            for (score[0..n]) |*one| {
                one.* = @exp(one.* - top);
                sum += one.*;
            }
            for (0..latent) |d| {
                var want: f64 = 0;
                var mass: f64 = 0;
                for (0..n) |t| {
                    const v: f64 = numeric.f32FromBf16(logical[@as(usize, @intCast(tokens[r * width + t])) * latent + d]);
                    want += score[t] / sum * v;
                    mass += score[t] / sum * @abs(v);
                }
                const have: f64 = numeric.f32FromBf16(got[(r * heads + h) * latent + d]);
                // bf16(p) on the tensor cores, then the result's own rounding
                try testing.expect(@abs(have - want) <= 0.02 * mass + 1e-6);
            }
        }
    }

    // a row alone at its position is the row in the window, a listed row and a dense one alike
    var one_qa = try cuda.Buffer.init(.bf16, &.{ 1, heads, latent });
    defer one_qa.deinit();
    var one_tok = try cuda.Buffer.init(.i32, &.{ 1, width });
    defer one_tok.deinit();
    var one_count = try cuda.Buffer.init(.i32, &.{1});
    defer one_count.deinit();
    var one_out = try cuda.Buffer.init(.bf16, &.{ 1, heads, latent });
    defer one_out.deinit();
    var one_pos = try cuda.Buffer.init(.i32, &.{1});
    defer one_pos.deinit();
    for (0..rows) |r| {
        const here: i32 = at + @as(i32, @intCast(r));
        try one_qa.upload(std.mem.sliceAsBytes(qa[r * heads * latent ..][0 .. heads * latent]));
        try one_tok.upload(std.mem.sliceAsBytes(tokens[r * width ..][0..width]));
        try one_count.upload(std.mem.sliceAsBytes(counts[r..][0..1]));
        try one_pos.upload(std.mem.asBytes(&here));
        try dsa.attendRows(one_qa.view, cache.view, one_pos.view, &table_dev.view, parts, one_out.view, .{ one_tok.view, one_count.view }, scale, positions);
        const alone = try shared.fetch(u16, arena, one_out);
        try testing.expectEqualSlices(u16, got[r * heads * latent ..][0 .. heads * latent], alone);
        if (counts[r] != 0) continue;
        // a count of 0 is the dense row: the same launch without lists
        try dsa.attendRows(one_qa.view, cache.view, one_pos.view, &table_dev.view, parts, one_out.view, null, scale, positions);
        try testing.expectEqualSlices(u16, alone, try shared.fetch(u16, arena, one_out));
    }

    // the same lists on an identity table, the latents in order, give the swapped table's rows
    const identity = [_]i32{ 0, 1 };
    const plain = try arena.alloc(u16, extents * extent * latent);
    scatter(plain, &identity, logical);
    var plain_dev = try shared.device(u16, .bf16, &.{ extents * extent, latent }, plain);
    defer plain_dev.deinit();
    var id_dev = try shared.device(i32, .i32, &.{extents}, &identity);
    defer id_dev.deinit();
    try dsa.attendRows(qa_dev.view, plain_dev.view, pos.view, &id_dev.view, parts, out.view, .{ tok_dev.view, count_dev.view }, scale, positions);
    const ordered = try shared.fetch(u16, arena, out);
    try testing.expectEqualSlices(u16, got, ordered);
}
