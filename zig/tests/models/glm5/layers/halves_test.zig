//! A rank's half of the head, chosen from on the device (spec 045 WS-03): each half of a row of GLM's
//! vocabulary picked apart (`zig/src/decode/pick.zig`), the halves' picks joined, and the token chosen from the
//! joined start, against the row read whole.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const picking = @import("../../../../src/decode/pick.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const shared = @import("../testing.zig");
const Picked = picking.Picked;

fn keyBefore(row: []const u16, a: u32, b: u32) bool {
    const x = sampling.keyOf(row[a]);
    const y = sampling.keyOf(row[b]);
    return x > y or (x == y and a < b);
}

test "the halves' picks joined are the row's start, and the token chosen from them the whole row's" {
    const gpa = testing.allocator;
    const vocab = 154880;
    const half = vocab / layout.world;
    var prng: std.Random.DefaultPrng = .init(4503);
    const random = prng.random();
    const row = try gpa.alloc(u16, vocab);
    defer gpa.free(row);
    const order = try gpa.alloc(u32, vocab);
    defer gpa.free(order);
    var sampler: sampling.Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    var outs: [layout.world]cuda.Buffer = undefined;
    for (&outs) |*out| out.* = try .init(.u32, &.{Picked.wordsFor(half)});
    defer for (&outs) |*out| out.deinit();

    const Kind = enum { peaked, coarse, zeros, holes, lower_empty, upper_empty };
    var settled: usize = 0;
    var nucleus: usize = 0;
    var across: usize = 0;
    var refused: usize = 0;
    for (0..18) |round| {
        const kind = std.enums.values(Kind)[round % 6];
        for (row, 0..) |*bits, id| {
            const v = random.floatNorm(f32);
            bits.* = numeric.bf16FromF32(switch (kind) {
                // a head's row: a few tokens well above the rest
                .peaked => if (random.uintLessThan(u32, 4000) == 0) 12.0 + 3.0 * v else 2.0 * v,
                // few values, so every cut is among equals of both halves
                .coarse => @round(v * 2.0) / 2.0,
                .zeros => if (random.uintLessThan(u8, 16) == 0) @as(f32, 0.0) else if (random.boolean()) -0.0 else -@abs(v),
                .holes => if (random.uintLessThan(u32, 500) != 0) -std.math.inf(f32) else @round(v * 30.0),
                .lower_empty => if (id < half) -std.math.inf(f32) else 6.0 * v,
                .upper_empty => if (id >= half) -std.math.inf(f32) else 6.0 * v,
            });
        }
        var row_dev = try shared.device(u16, .bf16, &.{ 1, vocab }, row);
        defer row_dev.deinit();
        for (order, 0..) |*id, i| id.* = @intCast(i);
        std.mem.sort(u32, order, @as([]const u16, row), keyBefore);
        const largest: f64 = numeric.f32FromBf16(row[order[0]]);
        var plain: f64 = 0;
        for (row) |bits| {
            const v: f64 = numeric.f32FromBf16(bits);
            if (v > largest - 24) plain += @exp(v - largest);
        }

        for ([_]u32{ 1, 28, Picked.most }) |rank| {
            var picked: [layout.world]Picked = undefined;
            for (&outs, &picked, 0..) |out, *one, index| {
                const base: [*]u8 = @ptrCast(row_dev.view.ptr.?);
                try Picked.ask(out, rank);
                try picking.pick(.init(@ptrCast(base + index * half * 2), .bf16, &.{ 1, half }), out.view);
                const raw = try shared.fetch(u32, gpa, out);
                defer gpa.free(raw);
                one.* = .read(raw[0..Picked.words]);
            }
            var scratch: picking.Joined = .{};
            const joined = picking.joined(&picked, half, &scratch) orelse {
                try testing.expect(picked[0].count == 0 or picked[1].count == 0);
                refused += 1;
                continue;
            };
            try testing.expect(joined.keys.len >= @min(rank, vocab));
            try testing.expectEqualSlices(u32, order[0..joined.ids.len], joined.ids);
            for (joined.keys, joined.ids) |key, id| try testing.expectEqual(sampling.keyOf(row[id]), key);
            try testing.expectEqual(sampling.keyOf(row[order[vocab - 1]]), joined.smallest);
            try testing.expect(joined.sum.? >= plain * (1 - 1.1e-5) and joined.sum.? <= plain * (1 + 1.1e-5) + vocab * @exp(-24.0));
            across += @intFromBool(joined.keys.len > 1 and joined.keys[0] == joined.keys[1] and (joined.ids[0] < half) != (joined.ids[1] < half));

            const seed = random.int(u64);
            const position = random.int(u32);
            for ([_]sampling.Sampling{
                .{ .seed = seed, .temperature = 0 },
                .{ .seed = seed },
                .{ .seed = seed, .temperature = 0.7, .top_k = 40, .top_p = 0.8, .min_p = 0.02 },
                .{ .seed = seed, .temperature = 0.8, .top_k = 0, .top_p = 0.9 },
                .{ .seed = seed, .temperature = 1.0, .top_k = 0, .top_p = 0.5 },
            }) |settings| {
                const whole = try sampler.choose(row, position, settings);
                const led = sampler.chooseLeading(joined, position, settings) orelse continue;
                try testing.expectEqual(whole, led);
                settled += 1;
                nucleus += @intFromBool(settings.top_k == 0 and !settings.greedy());
            }
        }
    }
    testing.expect(settled > 60 and nucleus > 5 and across > 3 and refused > 0) catch |err| {
        std.debug.print("{d} settled, {d} by a nucleus, {d} tied across the halves, {d} refused\n", .{ settled, nucleus, across, refused });
        return err;
    };
}
