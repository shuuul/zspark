//! `logprob.zig`'s tests (needs the GPU): a row's words against fp64 on the host, and the same bits whichever
//! rows are launched beside it.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../src/cuda/cuda.zig");
const numeric = @import("../../src/weights/numeric.zig");
const logprob = @import("../../src/decode/logprob.zig");

const words = logprob.words;

/// The words of `logits`' rows launched together, bf16 [rows, V] on the device.
fn launched(gpa: std.mem.Allocator, logits: cuda.View, k: u8) ![]u32 {
    const rows: usize = @intCast(logits.size[0]);
    var scratch: logprob.Scratch = try .init(gpa, rows);
    defer scratch.deinit();
    try scratch.launch(logits, k);
    try scratch.read(rows);
    return gpa.dupe(u32, scratch.host);
}

/// Whether id `a` comes before id `b` in a row's best: its logit higher, the lower id among equals.
fn before(row: []const u16, a: u32, b: u32) bool {
    const x = numeric.f32FromBf16(row[a]);
    const y = numeric.f32FromBf16(row[b]);
    return if (x != y) x > y else a < b;
}

test "a row's log-sum-exp and its best k: fp64's, logits descending and ties to the lower id, the same bits alone, in fours and in sixteen, at every k" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // Flash Next's vocabulary, and one short of a block's lanes
    for ([_]usize{ 151_936, 1023 }) |vocab| {
        const rows = 16;
        var prng: std.Random.DefaultPrng = .init(vocab);
        const random = prng.random();
        const host = try arena.alloc(u16, rows * vocab);
        for (host) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32) * 3);
        // ties among the best: row 3's highest three are equal, row 5 is flat but for one, row 7 is all one value
        for ([_]u32{ 900, 7, 5 }) |id| host[3 * vocab + id] = numeric.bf16FromF32(40);
        @memset(host[5 * vocab ..][0..vocab], numeric.bf16FromF32(-1));
        host[5 * vocab + vocab - 1] = numeric.bf16FromF32(2);
        @memset(host[7 * vocab ..][0..vocab], numeric.bf16FromF32(0.5));
        var logits: cuda.Buffer = try .init(.bf16, &.{ rows, vocab });
        defer logits.deinit();
        try logits.upload(std.mem.sliceAsBytes(host));

        const most = logprob.Row.most;
        const all = try launched(gpa, logits.view, most);
        defer gpa.free(all);
        for (0..rows) |r| {
            const row = host[r * vocab ..][0..vocab];
            const w = all[r * words ..][0..words];
            // fp64 on the host
            var top: f64 = -std.math.inf(f64);
            for (row) |bits| top = @max(top, numeric.f32FromBf16(bits));
            var sum: f64 = 0;
            for (row) |bits| sum += @exp(@as(f64, numeric.f32FromBf16(bits)) - top);
            const lse = top + @log(sum);
            const got: f32 = @bitCast(w[0]);
            try testing.expectApproxEqAbs(lse, @as(f64, got), 1e-5 * @max(1, @abs(lse)));
            // the best, in order: each before the next, and none outside before the last
            const ids = w[2..][0..most];
            for (ids, 0..) |id, j| {
                try testing.expect(id < vocab);
                try testing.expectEqual(numeric.f32FromBf16(row[id]), @as(f32, @bitCast(w[2 + most + j])));
                if (j > 0) try testing.expect(before(row, ids[j - 1], id));
            }
            var outside: usize = 0;
            for (0..vocab) |id| {
                if (std.mem.indexOfScalar(u32, ids, @intCast(id)) != null) continue;
                if (before(row, @intCast(id), ids[most - 1])) outside += 1;
            }
            try testing.expectEqual(0, outside);
            if (r == 3) try testing.expectEqualSlices(u32, &.{ 5, 7, 900 }, ids[0..3]);
            if (r == 7) for (ids, 0..) |id, j| try testing.expectEqual(j, id);
            // the row as `of` gives it: a logit less the log-sum-exp
            const one = logprob.of(w, 9, row[9], 4);
            try testing.expectEqual(numeric.f32FromBf16(row[9]) - got, one.logprob);
            try testing.expectEqual(numeric.f32FromBf16(row[ids[0]]) - got, one.logprobs[0]);
        }
        // the row rule: a row's words are its own, whoever is launched beside it
        for (0..rows) |r| {
            const alone = try launched(gpa, logits.span(r, 1), most);
            defer gpa.free(alone);
            try testing.expectEqualSlices(u32, all[r * words ..][0..words], alone);
        }
        for (0..rows / 4) |q| {
            const four = try launched(gpa, logits.span(q * 4, 4), most);
            defer gpa.free(four);
            try testing.expectEqualSlices(u32, all[q * 4 * words ..][0 .. 4 * words], four);
        }
        // fewer best: the log-sum-exp is the same bits, and the best the first of the twenty
        for ([_]u8{ 0, 1, 5 }) |k| {
            const fewer = try launched(gpa, logits.view, k);
            defer gpa.free(fewer);
            for (0..rows) |r| {
                const w = all[r * words ..][0..words];
                const v = fewer[r * words ..][0..words];
                try testing.expectEqual(w[0], v[0]);
                try testing.expectEqualSlices(u32, w[2..][0..k], v[2..][0..k]);
                try testing.expectEqualSlices(u32, w[2 + most ..][0..k], v[2 + most ..][0..k]);
            }
        }
    }
}
