//! `mtp.zig`'s tests of a draft's choice: a token's probability, the head's normalizer, what the device finds
//! in a scored row against the row read back, and where a chain ends; and a round's rows picked at once and
//! settled from their starts (`decode/settle.zig`, spec 144 WS-03). The head's rows: `mtp_test.zig`.

const std = @import("std");
const fixtures = @import("testing.zig");
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const pick_file = @import("../../../src/decode/pick.zig");
const settle = @import("../../../src/decode/settle.zig");
const draft = @import("../../../src/models/qwen4/draft.zig");
const mtp_file = @import("../../../src/models/qwen4/mtp.zig");
const Head = mtp_file.Head;
const Normalizer = mtp_file.Normalizer;
const Picked = mtp_file.Picked;
const minus_infinity = mtp_file.minus_infinity;
const normalizer = mtp_file.normalizer;
const pick = mtp_file.pick;
const probability = mtp_file.probability;
const testing = std.testing;
const heads = @import("mtp_fixture.zig");
const device = heads.device;
const fetch = fixtures.attending.fetch;
const randomBf16 = heads.randomBf16;
const testSubset = heads.testSubset;

test "probability: a token's share at temperature 1" {
    var logits: [5]u16 = undefined;
    for (&logits, [_]f32{ 1.0, 3.0, -40.0, 3.0, 0.5 }) |*bits, v| bits.* = numeric.bf16FromF32(v);
    var sum: f64 = 0;
    for ([_]f64{ 1.0, 3.0, -40.0, 3.0, 0.5 }) |v| sum += @exp(v);
    try testing.expectApproxEqRel(@as(f32, @floatCast(@exp(3.0) / sum)), probability(&logits, 1), 1e-5);
    try testing.expectApproxEqRel(@as(f32, @floatCast(@exp(0.5) / sum)), probability(&logits, 4), 1e-5);
    try testing.expect(probability(&logits, 2) < 1e-15);
}

test "the head's normalizer is the plain one to the bit, over a draft head's columns too" {
    const gpa = testing.allocator;
    var whole: Normalizer = try .init(gpa);
    defer whole.deinit(gpa);
    var prng: std.Random.DefaultPrng = .init(28);
    const random = prng.random();
    const vocab = 4096;
    const row = try gpa.alloc(u16, vocab);
    defer gpa.free(row);
    const columns = try gpa.alloc(u16, vocab);
    defer gpa.free(columns);
    // Rows one after another through one table, each with another largest logit: a term kept from the row
    // before would be of the wrong largest. Their spread puts logits on both sides of the 24 below it, and few
    // patterns a row make most terms ones read back.
    for (0..64) |round| {
        const spread: f32 = if (round % 3 == 0) 30.0 else 4.0;
        const shift = random.floatNorm(f32) * 10.0;
        var scored: usize = 0;
        for (row) |*bits| {
            // a draft head scores some of the vocabulary; the rest of the row is -inf
            if (random.uintLessThan(u8, 3) == 0) {
                bits.* = minus_infinity;
                continue;
            }
            bits.* = numeric.bf16FromF32(@round(random.floatNorm(f32) * spread) / 8.0 + shift);
            columns[scored] = bits.*;
            scored += 1;
        }
        const want: u64 = @bitCast(normalizer(row));
        try testing.expectEqual(want, @as(u64, @bitCast(whole.of(row))));
        try testing.expectEqual(want, @as(u64, @bitCast(whole.of(columns[0..scored]))));
    }
    // the row counter wrapping starts the table over rather than meeting an old stamp
    whole.row = std.math.maxInt(u32);
    try testing.expectEqual(@as(u64, @bitCast(normalizer(row))), @as(u64, @bitCast(whole.of(row))));
    try testing.expectEqual(@as(u32, 1), whole.row);
}

fn keyBefore(row: []const u16, a: u32, b: u32) bool {
    const x = sampling.keyOf(row[a]);
    const y = sampling.keyOf(row[b]);
    return x > y or (x == y and a < b);
}

test "what the device finds in a scored row: the sampler's order of its start, its extremes and the normalizer's sum" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(2028);
    const random = prng.random();
    const Kind = enum { coarse, fine, zeros, holes, flat, nothing, nan, minus_nan, infinite };
    var cuts_among_equals: usize = 0;
    var overflowed: usize = 0;
    var compared: usize = 0;
    // a draft head's columns and the vocabulary; sizes around the ranks asked to and the blocks' spans
    for ([_]usize{ 1, 2, 63, 64, 65, 213, 1000, 1023, 1024, 1025, 4097, 79591, 248320 }) |n| {
        const row = try gpa.alloc(u16, n);
        defer gpa.free(row);
        const order = try gpa.alloc(u32, n);
        defer gpa.free(order);
        var out: cuda.Buffer = try .init(.u32, &.{Picked.wordsFor(n)});
        defer out.deinit();
        for (std.enums.values(Kind)) |kind| {
            // the two long rows are sorted on the host for the first kinds alone
            if (n > 5000 and @backingInt(kind) > @backingInt(Kind.holes)) continue;
            for (row) |*bits| {
                const v = random.floatNorm(f32);
                bits.* = numeric.bf16FromF32(switch (kind) {
                    // few values, so a cut is among equals; -0 and +0 both, apart in this order
                    .coarse => @round(v * 2.0) / 2.0 * @as(f32, if (random.boolean()) 1.0 else -1.0),
                    .fine, .nan, .minus_nan, .infinite => v * 6.0,
                    .zeros => if (random.uintLessThan(u8, 16) == 0) @as(f32, 0.0) else if (random.boolean()) -0.0 else -@abs(v),
                    .holes => if (random.uintLessThan(u8, 8) != 0) -std.math.inf(f32) else @round(v * 30.0),
                    .flat => 1.25,
                    .nothing => -std.math.inf(f32),
                });
            }
            const at = random.uintLessThan(usize, n);
            switch (kind) {
                .nan => row[at] = 0x7fc0,
                .minus_nan => row[at] = 0xffc1,
                .infinite => row[at] = 0x7f80,
                else => {},
            }
            var row_dev = try device(u16, .bf16, &.{ 1, n }, row);
            defer row_dev.deinit();
            for (order, 0..) |*id, i| id.* = @intCast(i);
            std.mem.sort(u32, order, @as([]const u16, row), keyBefore);

            for ([_]u32{ 1, 28, Picked.most }) |rank| {
                try Picked.ask(out, rank);
                try pick(row_dev.view, out.view);
                const raw = try fetch(u32, gpa, out);
                defer gpa.free(raw);
                const got: Picked = .read(raw[0..Picked.words]);

                // the cut is the key of that rank, or the row's last; everything at or above it is wanted
                const cut = sampling.keyOf(row[order[@min(rank, n) - 1]]);
                var total: usize = 0;
                while (total < n and sampling.keyOf(row[order[total]]) >= cut) total += 1;
                try testing.expectEqual(total, got.total);
                try testing.expectEqual(sampling.keyOf(row[order[0]]), got.largest);
                try testing.expectEqual(sampling.keyOf(row[order[n - 1]]), got.smallest);
                if (total > Picked.slots) {
                    try testing.expectEqual(@as(usize, 0), got.count);
                    overflowed += 1;
                } else {
                    try testing.expectEqual(total, got.count);
                    try testing.expectEqualSlices(u32, order[0..total], got.columns[0..total]);
                    for (order[0..total], got.keys[0..total]) |id, key| try testing.expectEqual(sampling.keyOf(row[id]), key);
                    compared += 1;
                    if (total > @min(rank, n)) cuts_among_equals += 1;
                }

                // a row of holes alone, or with a NaN or +inf on top, has no sum
                if (got.largest >= 0x0080 and got.largest <= 0xff7f) {
                    const largest = numeric.f32FromBf16(row[order[0]]);
                    var sum: f64 = 0;
                    for (row) |bits| {
                        const v = numeric.f32FromBf16(bits);
                        if (v > largest - 24) sum += @exp(@as(f64, v - largest));
                    }
                    try testing.expectApproxEqRel(sum, got.sum, 1e-5);
                } else try testing.expectEqual(@as(f64, 0), got.sum);

                // the same row again: the same answer, whatever order the threads ran in
                try pick(row_dev.view, out.view);
                const again_raw = try fetch(u32, gpa, out);
                defer gpa.free(again_raw);
                const again: Picked = .read(again_raw[0..Picked.words]);
                try testing.expectEqual(@as(u64, @bitCast(got.sum)), @as(u64, @bitCast(again.sum)));
                try testing.expectEqual(got.total, again.total);
                try testing.expectEqualSlices(u16, got.keys[0..got.count], again.keys[0..again.count]);
                try testing.expectEqualSlices(u32, got.columns[0..got.count], again.columns[0..again.count]);
            }
        }
    }
    try testing.expect(cuts_among_equals > 20 and overflowed > 5 and compared > 150);
}

test "a draft chosen from what the device found is the draft chosen from the row read back, and a chain ends where its floors say" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const subset = try testSubset(arena.allocator(), 320);
    var prng: std.Random.DefaultPrng = .init(77);
    const random = prng.random();
    const rows = 3;
    const depth = 5;
    const tokens = [rows]i32{ 4, 250, 9 };
    var on_device: usize = 0;
    var chains: [depth + 1]usize = @splat(0);
    var by_chain: [depth + 1]usize = @splat(0);
    var by_landing: [depth + 1]usize = @splat(0);
    for ([_]?[]const u32{ null, subset }) |draft_vocab| {
        var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), draft_vocab);
        defer m.deinit();
        var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
        defer sampler.deinit(gpa);
        const host = try arena.allocator().alloc(u16, rows * m.shape.wide());
        for ([_]sampling.Sampling{
            .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 },
            .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 },
            .{ .seed = 5, .temperature = 0.7, .top_k = 20, .top_p = 0.8, .min_p = 0.01 },
            // the kernel's candidates do not reach this top-k: the host's; a nucleus is the device's where
            // the row's start bounds its cut
            .{ .seed = 5, .temperature = 1.0, .top_k = 60, .top_p = 1, .min_p = 0 },
            .{ .seed = 6, .temperature = 1.0, .top_k = 0, .top_p = 0.9, .min_p = 0 },
        }, 0..) |settings, which| {
            const served = which < 3;
            for (0..2) |_| {
                randomBf16(random, host, 1.5);
                var streams = try device(u16, .bf16, &.{ rows, m.shape.wide() }, host);
                defer streams.deinit();
                // a random head gives its drafts a few hundredths: thresholds among them, so that chains end
                // at every length
                const Floors = struct { confidence: f32, chain: f32 = 0, landing: f64 = 0 };
                for ([_]Floors{
                    .{ .confidence = 0 },
                    .{ .confidence = 0.004 },
                    .{ .confidence = 0.012 },
                    .{ .confidence = 0.03 },
                    .{ .confidence = 0.7 },
                    // a chain of such drafts is a few hundredths to the power of its length
                    .{ .confidence = 0, .chain = 1e-4 },
                    .{ .confidence = 0, .chain = 1e-6 },
                    .{ .confidence = 0.004, .chain = 1e-8 },
                    // by the square and by a fitted calibration (`Paying.Landing`) both, one that says less
                    // of such a draft than its square: `landing` is the row's cost, so that chains end at
                    // every length
                    .{ .confidence = 0, .landing = 1e-3 },
                    .{ .confidence = 0, .landing = 1e-8 },
                    .{ .confidence = 0, .landing = 1e-11 },
                    .{ .confidence = 0, .landing = 1e-14 },
                    .{ .confidence = 0, .landing = 1e-18 },
                }) |floors| {
                    const confidence = floors.confidence;
                    var want: [depth]i32 = undefined;
                    var want_count: usize = 0;
                    const Way = enum { host, device, margin };
                    for ([_]Way{ .host, .device, .margin }) |way| for ([_]bool{ true, false }) |graphed| {
                        var head: Head = try .init(gpa, m, &sampler, settings, .{ .rows = m.max_rows, .capacity = 64, .graphs = graphed });
                        defer head.deinit();
                        head.confidence = confidence;
                        head.chain = floors.chain;
                        if (floors.landing > 0) {
                            head.costs = .{ .row = floors.landing, .level = floors.landing / 4, .landing = .{ .slope = 2, .levels = @splat(-0.5) } };
                            head.fresh();
                        }
                        head.on_device = way != .host;
                        // every probability within the margin of the threshold: the host decides each draft
                        if (way == .margin) head.margin = 1e9;
                        var out: [depth]i32 = undefined;
                        const d = head.drafter();
                        const n = try d.draft(d.context, streams.view, &tokens, rows, null, &out);
                        if (way == .host and graphed) {
                            want = out;
                            want_count = n;
                            chains[n] += 1;
                            if (floors.chain > 0) by_chain[n] += 1;
                            if (floors.landing > 0) by_landing[n] += 1;
                            // the rule itself, from the probabilities the head gave: every draft kept left the
                            // chain at its floors, but the first; the one that ended it did not
                            var through: f64 = 1;
                            var landed: f64 = 1;
                            for (head.trail[0..head.trailed], 0..) |considered, i| {
                                through *= considered.p;
                                if (floors.landing > 0) landed *= head.costs.?.landing.?.of(considered.p, i);
                                const first: mtp_file.Paying = if (floors.landing > 0) .begin(head.costs.?) else undefined;
                                const short = floors.landing > 0 and i > 0 and (first.pays(landed).low or through < first.floors()[0]);
                                const under = (confidence > 0 and considered.p < confidence) or (floors.chain > 0 and through < floors.chain) or short;
                                if (i + 1 < n) try testing.expect(!under);
                                if (i >= n) try testing.expect(under and i == n and n > 0);
                            }
                            if (confidence > 0 or floors.chain > 0 or floors.landing > 0) try testing.expect(head.trailed >= n and head.trailed <= n + 1);
                        }
                        try testing.expectEqual(want_count, n);
                        try testing.expectEqualSlices(i32, want[0..n], out[0..n]);
                        const all_device = way == .device or (way == .margin and confidence == 0 and floors.chain == 0 and floors.landing == 0);
                        if (served and all_device) {
                            try testing.expectEqual(@as(usize, 0), head.chosen.host);
                            try testing.expect(head.chosen.device >= n);
                            on_device += head.chosen.device;
                            // under a calibration a draft that no margin moves across a threshold stays the device's
                        } else if (!(way == .margin and floors.landing > 0)) try testing.expectEqual(@as(usize, 0), head.chosen.device);
                    };
                }
            }
        }
    }
    try testing.expect(on_device > 250);
    // chains that ended at their first draft, in between, and that filled the window
    try testing.expect(chains[1] > 0 and chains[depth] > 0 and chains[2] + chains[3] + chains[4] > 0);
    try testing.expect(by_chain[1] > 0 and by_chain[depth] > 0 and by_chain[2] + by_chain[3] + by_chain[4] > 0);
    try testing.expect(by_landing[1] > 0 and by_landing[depth] > 0 and by_landing[2] + by_landing[3] + by_landing[4] > 0);
}

test "a chain ends with a draft that ends the reply" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var prng: std.Random.DefaultPrng = .init(78);
    const random = prng.random();
    const rows = 3;
    const depth = 5;
    const tokens = [rows]i32{ 4, 250, 9 };
    var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), null);
    defer m.deinit();
    var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
    defer sampler.deinit(gpa);
    const host = try arena.allocator().alloc(u16, rows * m.shape.wide());
    const settings: sampling.Sampling = .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 };
    var cut: usize = 0;
    for (0..4) |_| {
        randomBf16(random, host, 1.5);
        var streams = try device(u16, .bf16, &.{ rows, m.shape.wide() }, host);
        defer streams.deinit();
        var whole: [depth]i32 = undefined;
        for ([_]?usize{ null, 0, 2, depth - 1 }) |end| {
            var head: Head = try .init(gpa, m, &sampler, settings, .{ .rows = m.max_rows, .capacity = 64 });
            defer head.deinit();
            head.confidence = 0;
            head.chain = 0;
            const ends = [1]u32{if (end) |at| @intCast(whole[at]) else 0};
            if (end != null) head.ends = &ends;
            var out: [depth]i32 = undefined;
            const d = head.drafter();
            const n = try d.draft(d.context, streams.view, &tokens, rows, null, &out);
            const at = end orelse {
                try testing.expectEqual(depth, n);
                whole = out;
                continue;
            };
            // the chain is the same up to the first draft that is the end token, which is its last
            const first = std.mem.indexOfScalar(i32, &whole, whole[at]).?;
            try testing.expectEqual(first + 1, n);
            try testing.expectEqualSlices(i32, whole[0..n], out[0..n]);
            cut += @intFromBool(n < depth);
        }
    }
    try testing.expect(cut >= 4);
}

test "a round's rows settled on the device in one pass are the tokens choose gives each whole row, on built ties, one rank or two halves" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(144);
    const random = prng.random();
    // every sampling option the server takes (seed, temperature, top_k, top_p, min_p): those a pick's start can
    // settle, and those read whole (a top_k past `Picked.most`, no cut, a nucleus over a temperature of 1)
    const settings = [_]sampling.Sampling{
        .{ .seed = 1, .temperature = 0 },
        .{ .seed = 2, .temperature = 0.7, .top_k = 20, .top_p = 1.0 },
        .{ .seed = 3, .temperature = 0.6 },
        .{ .seed = 4, .temperature = 1.0, .top_k = 0, .top_p = 0.9 },
        .{ .seed = 5, .temperature = 0.5, .top_k = 0, .top_p = 0.8, .min_p = 0.05 },
        .{ .seed = 6, .temperature = 1.5, .top_k = 40, .top_p = 1.0 },
        .{ .seed = 7, .temperature = 0.8, .top_k = 50, .top_p = 0.9, .min_p = 0.1 },
        .{ .seed = 8, .temperature = 0.7, .top_k = 60 },
        .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 10, .temperature = 1.5, .top_k = 0, .top_p = 0.9 },
    };
    const Kind = enum { fine, coarse, zeros, holes, flat, nan };
    const kinds = std.enums.values(Kind);
    const rows = kinds.len * settings.len;
    var settled: usize = 0;
    var compared: usize = 0;
    var sure: usize = 0;
    for ([_]usize{ 4097, 79591 }) |half| {
        const n = 2 * half;
        const host = try gpa.alloc(u16, rows * n);
        defer gpa.free(host);
        // each row's two halves, as two ranks hold them: [part, rows, half]
        const halves = try gpa.alloc(u16, 2 * rows * half);
        defer gpa.free(halves);
        for (kinds, 0..) |kind, k| for (0..settings.len) |s| {
            const i = k * settings.len + s;
            const row = host[i * n ..][0..n];
            for (row) |*bits| {
                const v = random.floatNorm(f32);
                bits.* = numeric.bf16FromF32(switch (kind) {
                    .fine, .nan => v * 6.0,
                    // few values: ties within each half and across the seam
                    .coarse => @round(v * 2.0) / 2.0,
                    .zeros => if (random.uintLessThan(u8, 16) == 0) @as(f32, 0.0) else if (random.boolean()) -0.0 else -@abs(v),
                    .holes => if (random.uintLessThan(u8, 8) != 0) -std.math.inf(f32) else @round(v * 30.0),
                    .flat => 1.25,
                });
            }
            if (kind == .nan) row[random.uintLessThan(usize, n)] = 0x7fc0;
            for (0..2) |part| @memcpy(halves[(part * rows + i) * half ..][0..half], row[part * half ..][0..half]);
        };
        var sampler: sampling.Sampler = try .init(gpa, n);
        defer sampler.deinit(gpa);
        var whole = try device(u16, .bf16, &.{ rows, n }, host);
        defer whole.deinit();
        var first = try device(u16, .bf16, &.{ rows, half }, halves[0 .. rows * half]);
        defer first.deinit();
        var second = try device(u16, .bf16, &.{ rows, half }, halves[rows * half ..]);
        defer second.deinit();
        for ([_]usize{ 1, 2 }) |world| {
            const columns = n / world;
            var picks: settle.Picks = try .init(gpa, rows, columns);
            defer picks.deinit();
            for (picks.rows[0..rows], 0..) |*one, i| one.* = .{ .settings = settings[i % settings.len], .position = 7 + 3 * i };
            try testing.expect(settle.asksOf(picks.rows[0..rows], columns, n, picks.asked[0..rows]));
            const slot: usize = @intCast(picks.words.view.size[2]);
            const got = if (world == 1) got: {
                try settle.launchRound(1, {}, whole, &picks, rows);
                break :got try settle.readRound(1, &picks, rows, columns, &sampler);
            } else got: {
                // two ranks' halves, each picked as its rank's `launchRound` does, into its part of the words
                try settle.launchRound(1, {}, first, &picks, rows);
                const base: [*]u8 = @ptrCast(picks.words.view.ptr.?);
                const asks: cuda.View = .init(picks.asks.view.ptr.?, .u8, &.{rows});
                try pick_file.pickRows(second.view, .init(@ptrCast(base + rows * slot * 4), .u32, &.{ rows, slot }), asks);
                break :got try settle.readRound(2, &picks, rows, columns, &sampler);
            };
            settled += got;
            for (picks.out[0..rows], picks.rows[0..rows], picks.asked[0..rows], 0..) |token, one, ask, i| {
                const row = host[i * n ..][0..n];
                if (ask == 0) try testing.expectEqual(@as(?i32, null), token);
                const chosen = sampler.choose(row, one.?.position, one.?.settings) catch {
                    try testing.expectEqual(@as(?i32, null), token);
                    continue;
                };
                if (token) |t| {
                    try testing.expectEqual(chosen, t);
                    compared += 1;
                }
                // a row of finite values far from zero settles a greedy choice and top-k's (a nucleus's may
                // need more than its start)
                const rule = one.?.settings;
                if (i / settings.len == @backingInt(Kind.fine) and ask != 0 and (rule.greedy() or rule.top_k != 0)) {
                    try testing.expect(token != null);
                    sure += 1;
                }
            }
            // a pick of many rows is each row's pick alone: the last part's rows against `pick` of each
            var one: cuda.Buffer = try .init(.u32, &.{Picked.wordsFor(columns)});
            defer one.deinit();
            for (0..rows) |i| {
                try Picked.ask(one, @max(picks.asked[i], 1));
                try pick(if (world == 1) whole.span(i, 1) else second.span(i, 1), one.view);
                const raw = try fetch(u32, gpa, one);
                defer gpa.free(raw);
                const alone: Picked = .read(raw[0..Picked.words]);
                const many: Picked = .read(picks.host[((world - 1) * rows + i) * slot ..][0..Picked.words]);
                try testing.expectEqual(alone.total, many.total);
                try testing.expectEqual(alone.largest, many.largest);
                try testing.expectEqual(alone.smallest, many.smallest);
                try testing.expectEqual(@as(u64, @bitCast(alone.sum)), @as(u64, @bitCast(many.sum)));
                try testing.expectEqualSlices(u16, alone.keys[0..alone.count], many.keys[0..many.count]);
                try testing.expectEqualSlices(u32, alone.columns[0..alone.count], many.columns[0..many.count]);
            }
        }
    }
    // five rules a fine row settles, two row lengths, one rank and two
    try testing.expect(settled == compared and sure == 20 and settled >= sure);
}
