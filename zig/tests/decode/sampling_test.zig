//! `sampling.zig`'s tests: the expected picks are TensorFold's own, exported by `tools/reference/sampling_cases.py`.

const std = @import("std");
const numeric = @import("../../src/weights/numeric.zig");
const sampling = @import("../../src/decode/sampling.zig");
const Allocator = std.mem.Allocator;
const Sampler = sampling.Sampler;
const Sampling = sampling.Sampling;
const argmax = sampling.argmax;
const keyOf = sampling.keyOf;
const lowest_finite = sampling.lowest_finite;
const margin = sampling.margin;
const massNeeded = sampling.massNeeded;
const massOf = sampling.massOf;
const minus_zero = sampling.minus_zero;
const pairwiseSum = sampling.pairwiseSum;
const plus_zero = sampling.plus_zero;
const seedFor = sampling.seedFor;
const seedFromInt = sampling.seedFromInt;
const uniform = sampling.uniform;

const testing = std.testing;

const fx = struct {
    const cases = @embedFile("fixtures/sampling_cases.bin");
    /// The file the fixture's real rows index: `logits`, BF16 [23, 248320], from `flash_next_export.py`.
    const reference = ".cache/zspark/reference/primes.serial.f32.safetensors";

    const Reader = struct {
        bytes: []const u8,
        at: usize = 0,

        fn int(self: *Reader, comptime T: type) T {
            const size = @divExact(@typeInfo(T).int.bits, 8);
            defer self.at += size;
            return std.mem.readInt(T, self.bytes[self.at..][0..size], .little);
        }

        fn float(self: *Reader) f64 {
            return @bitCast(self.int(u64));
        }
    };

    const Row = struct { offset: usize, vocab: usize };

    const Fixture = struct {
        reader: Reader,
        rows: []Row,
        settings: []Sampling,

        fn open(gpa: Allocator) !Fixture {
            var reader: Reader = .{ .bytes = cases };
            try testing.expectEqualStrings("ZSSAMPL1", cases[0..8]);
            reader.at = 8;
            const rows = try gpa.alloc(Row, reader.int(u32));
            errdefer gpa.free(rows);
            for (rows) |*row| {
                row.vocab = reader.int(u32);
                row.offset = reader.at;
                reader.at += 2 * row.vocab;
            }
            const settings = try gpa.alloc(Sampling, reader.int(u32));
            for (settings) |*s| {
                const temperature = reader.float();
                const top_p = reader.float();
                const min_p = reader.float();
                const top_k = reader.int(u32);
                _ = reader.int(u32);
                s.* = .{ .seed = 0, .temperature = temperature, .top_k = top_k, .top_p = top_p, .min_p = min_p };
            }
            return .{ .reader = reader, .rows = rows, .settings = settings };
        }

        fn close(self: *Fixture, gpa: Allocator) void {
            gpa.free(self.rows);
            gpa.free(self.settings);
        }

        fn logitsOf(self: Fixture, gpa: Allocator, index: usize) ![]u16 {
            const found = self.rows[index];
            const out = try gpa.alloc(u16, found.vocab);
            for (out, 0..) |*bits, i| bits.* = std.mem.readInt(u16, cases[found.offset + 2 * i ..][0..2], .little);
            return out;
        }
    };

    const real_row = 1 << 31;

    /// Runs the fixture's cases on its own rows (`real` false) or on the reference file's rows.
    fn check(gpa: Allocator, real: ?[]const u16, vocab: usize) !usize {
        var fixture: Fixture = try .open(gpa);
        defer fixture.close(gpa);
        var widest = vocab;
        for (fixture.rows) |r| widest = @max(widest, r.vocab);
        var sampler: Sampler = try .init(gpa, widest);
        defer sampler.deinit(gpa);

        var reader = fixture.reader;
        var checked: usize = 0;
        var wrong: usize = 0;
        var held: ?usize = null;
        var logits: []u16 = &.{};
        defer gpa.free(logits);
        for (0..reader.int(u32)) |_| {
            const row_index = reader.int(u32);
            var s = fixture.settings[reader.int(u32)];
            s.seed = reader.int(u64);
            const position = reader.int(u64);
            const expected = reader.int(u32);
            if ((row_index & real_row != 0) != (real != null)) continue;
            const row: []const u16 = if (real) |all| all[(row_index & ~@as(u32, real_row)) * vocab ..][0..vocab] else own: {
                if (held != row_index) {
                    gpa.free(logits);
                    logits = &.{};
                    logits = try fixture.logitsOf(gpa, row_index);
                    held = row_index;
                }
                break :own logits;
            };
            try s.validate();
            const got = try sampler.choose(row, position, s);
            checked += 1;
            if (got != expected) {
                wrong += 1;
                std.debug.print("row {d} temperature {d} top_k {d} top_p {d} min_p {d} seed {d} position {d}: {d}, TensorFold {d}\n", .{
                    row_index & ~@as(u32, real_row), s.temperature, s.top_k, s.top_p, s.min_p, s.seed, position, got, expected,
                });
            }
        }
        try testing.expectEqual(0, wrong);
        return checked;
    }

    /// The top-k rule as `sample_rows` and `choose_rows` state it: the candidates in `torch.topk`'s order, those
    /// sorted by value and id, cut, score.
    fn plainTopK(gpa: Allocator, logits: []const u16, position: u64, s: Sampling) !i32 {
        const order = try sorted(gpa, logits, .bits);
        defer gpa.free(order);
        std.mem.sort(u32, order[0..@min(@as(usize, s.top_k) + margin, logits.len)], logits, byValue);
        const k: usize = @min(s.top_k, logits.len);
        const scaled = try gpa.alloc(f64, k);
        defer gpa.free(scaled);
        const probs = try gpa.alloc(f64, k);
        defer gpa.free(probs);
        const temperature = @max(s.temperature, 1e-6);
        for (scaled, order[0..k]) |*v, id| v.* = @as(f64, numeric.f32FromBf16(logits[id])) / temperature;
        var kept = k;
        if (s.cuts()) {
            for (probs, scaled) |*p, v| p.* = @exp(v - scaled[0]);
            const sum = pairwiseSum(probs);
            var running: f64 = 0.0;
            var below: usize = 0;
            for (probs) |p| {
                running += p / sum;
                if (running < s.top_p) below += 1;
            }
            kept = @min(kept, below + 1);
        }
        if (s.min_p > 0.0) {
            var above: usize = 0;
            for (scaled[0..kept]) |v| above += @intFromBool(v >= scaled[0] + @log(s.min_p));
            kept = above;
        }
        return best(scaled[0..kept], order, position, s);
    }

    /// The whole-row rule as `nucleus_rows` and `_draw` state it, a mass per token.
    fn plainNucleus(gpa: Allocator, logits: []const u16, position: u64, s: Sampling) !i32 {
        const order = try sorted(gpa, logits, .value);
        defer gpa.free(order);
        const scaled = try gpa.alloc(f64, logits.len);
        defer gpa.free(scaled);
        const reciprocal = 1.0 / @max(s.temperature, 1e-6);
        for (scaled, order) |*v, id| v.* = @as(f64, numeric.f32FromBf16(logits[id])) * reciprocal;
        var kept = logits.len;
        if (s.cuts()) {
            var total: u64 = 0;
            for (scaled) |v| total += massOf(v, scaled[0]);
            const need = massNeeded(s.top_p, total);
            var running: u64 = 0;
            var below: usize = 0;
            for (scaled) |v| {
                running += massOf(v, scaled[0]);
                if (running < need) below += 1;
            }
            kept = @min(kept, below + 1);
        }
        if (s.min_p > 0.0) {
            var above: usize = 0;
            for (scaled) |v| above += @intFromBool(v >= scaled[0] + @log(s.min_p));
            kept = @min(kept, above);
        }
        return best(scaled[0..kept], order, position, s);
    }

    /// Every id, by value down then id up, or by `torch.topk`'s order: the same but +0 ahead of -0.
    fn sorted(gpa: Allocator, logits: []const u16, comptime by: enum { value, bits }) ![]u32 {
        const order = try gpa.alloc(u32, logits.len);
        for (order, 0..) |*id, i| id.* = @intCast(i);
        std.mem.sort(u32, order, logits, if (by == .value) byValue else byBits);
        return order;
    }

    fn byValue(row: []const u16, a: u32, b: u32) bool {
        const x = numeric.f32FromBf16(row[a]);
        const y = numeric.f32FromBf16(row[b]);
        return x > y or (x == y and a < b);
    }

    fn byBits(row: []const u16, a: u32, b: u32) bool {
        const x = numeric.f32FromBf16(row[a]);
        const y = numeric.f32FromBf16(row[b]);
        if (x != y) return x > y;
        if (row[a] != row[b]) return row[a] == 0; // both are zeros: the one without the sign bit first
        return a < b;
    }

    fn best(scaled: []const f64, order: []const u32, position: u64, s: Sampling) i32 {
        var winner: usize = 0;
        var highest = -std.math.inf(f64);
        for (scaled, 0..) |v, i| {
            const value = v - @log(-@log(uniform(s.seed, position, order[i])));
            if (value > highest) {
                highest = value;
                winner = i;
            }
        }
        return @intCast(order[winner]);
    }

    fn bf16(values: []const f32, out: []u16) void {
        for (out, values) |*bits, v| bits.* = numeric.bf16FromF32(v);
    }
};

test "every fixture case picks TensorFold's token" {
    // 15 synthetic rows (random at three scales, peaked, flat, zeros of both signs, ties at the top, at the top-k
    // cut and at the top-p cut, forty tied across the top-k cut, masked, a nucleus wider than 1,024, a row
    // narrower than top-k) by 73 settings by two (seed, position) keys
    const checked = try fx.check(testing.allocator, null, 0);
    try testing.expectEqual(2190, checked);
}

test "the real logits' fixture cases pick TensorFold's token" {
    // Rows 0, 11 and 22 of the `primes` reference logits, 248,320 wide, by the 73 settings. Skipped unless
    // ~/.cache/zspark/reference/primes.serial.f32.safetensors exists (tools/reference/AGENTS.md).
    const gpa = testing.allocator;
    const io = testing.io;
    const home = testing.environ.getPosix("HOME") orelse return error.SkipZigTest;
    const path = try std.fs.path.join(gpa, &.{ home, fx.reference });
    defer gpa.free(path);
    var file = @import("../../src/weights/safetensors.zig").File.open(gpa, io, .cwd(), path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.close();
    const tensor = file.get("logits") orelse return error.SkipZigTest;
    try testing.expectEqual(.BF16, tensor.dtype);
    const vocab: usize = @intCast(tensor.shape[1]);
    const logits = try gpa.alloc(u16, @intCast(tensor.count()));
    defer gpa.free(logits);
    @memcpy(std.mem.sliceAsBytes(logits), tensor.data);
    try testing.expectEqual(219, try fx.check(gpa, logits, vocab));
}

test "the keyed draws and the prompt's seed are TensorFold's" {
    const gpa = testing.allocator;
    var fixture: fx.Fixture = try .open(gpa);
    defer fixture.close(gpa);
    var reader = fixture.reader;
    const cases: usize = reader.int(u32);
    reader.at += 28 * cases; // past the cases
    const draws = reader.int(u32);
    try testing.expectEqual(240, draws);
    for (0..draws) |_| {
        const seed = reader.int(u64);
        const position = reader.int(u64);
        const token = reader.int(u64);
        try testing.expectEqual(reader.int(u64), @as(u64, @bitCast(uniform(seed, position, token))));
    }
    const seeds = reader.int(u32);
    try testing.expectEqual(20, seeds);
    for (0..seeds) |_| {
        const salt = reader.int(i64);
        const tokens = try gpa.alloc(i32, reader.int(u32));
        defer gpa.free(tokens);
        for (tokens) |*token| token.* = @intCast(reader.int(u32));
        try testing.expectEqual(reader.int(u64), seedFor(tokens, salt));
    }
    try testing.expectEqual(reader.bytes.len, reader.at);

    // a negative seed is its low 64 bits, and a draw sits strictly inside (0, 1)
    try testing.expectEqual(@as(u64, 0xFFFFFFFFFFFFFFFB), seedFromInt(-5));
    for (0..1000) |token| {
        const u = uniform(7, 3, token);
        try testing.expect(u > 0.0 and u < 1.0);
    }
}

test "temperature 0 is the argmax, the lower id among equals" {
    const gpa = testing.allocator;
    var sampler: Sampler = try .init(gpa, 200);
    defer sampler.deinit(gpa);
    var values: [200]f32 = undefined;
    var prng: std.Random.DefaultPrng = .init(11);
    for (&values) |*v| v.* = prng.random().floatNorm(f32) * 3.0;
    var logits: [200]u16 = undefined;
    // the largest at three ids, in each part of argmax's loop: its windows of 32 and the tail
    for ([_][3]usize{ .{ 150, 40, 199 }, .{ 199, 197, 198 }, .{ 31, 32, 64 }, .{ 0, 1, 2 } }) |ties| {
        var tied = values;
        for (ties) |id| tied[id] = 40.0;
        fx.bf16(&tied, &logits);
        const lowest: i32 = @intCast(@min(ties[0], ties[1], ties[2]));
        try testing.expectEqual(lowest, try argmax(&logits));
        for ([_]f64{ 0.0, -1.0 }) |temperature| {
            for (0..8) |position| {
                const s: Sampling = .{ .seed = position * 977, .temperature = temperature };
                try testing.expectEqual(lowest, try sampler.choose(&logits, position, s));
            }
        }
    }
    // all negative but two zeros of different signs: they are equal, so the lower id wins whichever comes first
    for ([_][2]f32{ .{ 0.0, -0.0 }, .{ -0.0, 0.0 } }) |zeros| {
        for (&values) |*v| v.* = -@abs(v.*) - 0.5;
        values[90] = zeros[0];
        values[120] = zeros[1];
        fx.bf16(&values, &logits);
        try testing.expectEqual(90, try argmax(&logits));
        // and the sampling rules see them as one value too: only those two survive a top-k of 2
        var seen: [2]bool = .{ false, false };
        for (0..64) |position| {
            const got = try sampler.choose(&logits, position, .{ .seed = 5, .top_k = 2, .top_p = 1.0 });
            try testing.expect(got == 90 or got == 120);
            seen[@intFromBool(got == 120)] = true;
        }
        try testing.expect(seen[0] and seen[1]);
    }
}

test "the counting sort's picks equal a plain sort's" {
    // Against the host: the rules written out over a full sort of the row, on random rows whose values are
    // coarse enough to tie everywhere (steps of 1/4), so every cut lands among equals somewhere.
    const gpa = testing.allocator;
    var sampler: Sampler = try .init(gpa, 700);
    defer sampler.deinit(gpa);
    var prng: std.Random.DefaultPrng = .init(404);
    const random = prng.random();
    var values: [700]f32 = undefined;
    var logits: [700]u16 = undefined;
    var differing: usize = 0;
    // rows whose start settled the choice, and of those the sampled ones with a -0 among the candidates
    var led_settled: usize = 0;
    var led_zero: usize = 0;
    var led_nucleus: usize = 0;
    defer testing.expect(led_settled > 400 and led_zero > 20 and led_nucleus > 20) catch std.debug.panic("too few rows settled by their start: {d} {d} {d}", .{ led_settled, led_zero, led_nucleus });
    for (0..400) |round| {
        const vocab = random.intRangeAtMost(usize, 1, 700);
        const spread: f32 = ([_]f32{ 0.3, 1.0, 4.0 })[round % 3];
        for (values[0..vocab]) |*v| v.* = @round(random.floatNorm(f32) * spread * 4.0) / 4.0;
        // every other round nothing is above zero, and the zeros have either sign: `torch.topk` takes the +0s
        // first, the other rules take them by id
        if (round % 2 == 0) for (values[0..vocab]) |*v| {
            v.* = @min(v.*, 0.0);
        };
        for (values[0..vocab]) |*v| {
            if (v.* == 0.0) v.* = if (random.boolean()) 0.0 else -0.0;
        }
        if (round % 5 == 0) for (values[0..vocab]) |*v| {
            if (random.boolean()) v.* = -std.math.inf(f32);
        };
        values[random.uintLessThan(usize, vocab)] = -1.0; // something finite
        fx.bf16(values[0..vocab], logits[0..vocab]);
        const s: Sampling = .{
            .seed = random.int(u64),
            .temperature = ([_]f64{ 0.3, 1.0, 1.7, 1e-9 })[random.uintLessThan(usize, 4)],
            .top_k = ([_]u32{ 0, 0, 1, 7, 20, 100, 5000 })[random.uintLessThan(usize, 7)],
            .top_p = ([_]f64{ 1.0, 0.95, 0.5, 0.01, 0.0 })[random.uintLessThan(usize, 5)],
            .min_p = ([_]f64{ 0.0, 0.05, 0.6, 1.0 })[random.uintLessThan(usize, 4)],
        };
        const position = random.int(u32);
        const row = logits[0..vocab];

        // A row known by its start (`chooseLeading`): the same token, or none where the start cannot say.
        {
            const order = try fx.sorted(gpa, row, .bits);
            defer gpa.free(order);
            var keys: [64]u16 = undefined;
            var ids: [64]u32 = undefined;
            var smallest: u16 = 0xffff;
            for (row) |bits| smallest = @min(smallest, keyOf(bits));
            var plain_sum: f64 = 0;
            const largest: f64 = numeric.f32FromBf16(row[order[0]]);
            for (row) |bits| {
                const v: f64 = numeric.f32FromBf16(bits);
                if (v > largest - 24) plain_sum += @exp(v - largest);
            }
            for ([_]usize{ 64, 28, 1 }) |most| {
                const n = @min(most, vocab);
                for (order[0..n], keys[0..n], ids[0..n]) |id, *key, *out| {
                    key.* = keyOf(row[id]);
                    out.* = id;
                }
                for ([_]Sampling{ s, .{ .seed = s.seed, .temperature = 0, .top_k = s.top_k, .top_p = s.top_p, .min_p = s.min_p } }) |each| {
                    const whole = sampler.choose(row, position, each) catch unreachable;
                    // the row's sum as a kernel gives it: fp32, a little off either way
                    const off: f64 = ([_]f64{ 1, 1 + 1e-5, 1 - 1e-5 })[position % 3];
                    const led = sampler.chooseLeading(.{ .keys = keys[0..n], .ids = ids[0..n], .smallest = smallest, .len = vocab, .sum = plain_sum * off }, position, each);
                    if (led) |token| try testing.expectEqual(whole, token);
                    if (each.top_k == 0 and !each.greedy()) {
                        // a nucleus is settled where the start bounds the cut, which no rule here foretells
                        led_nucleus += @intFromBool(led != null);
                        if (!each.cuts() or each.temperature > 1) try testing.expectEqual(null, led);
                        continue;
                    }
                    const zero_top = keys[0] == plus_zero or keys[0] == minus_zero;
                    const candidates = @min(@as(usize, each.top_k) + margin, vocab);
                    const settled = if (each.greedy()) !zero_top else each.top_k != 0 and candidates <= n and keys[candidates - 1] >= lowest_finite;
                    try testing.expectEqual(settled, led != null);
                    led_settled += @intFromBool(led != null);
                    led_zero += @intFromBool(led != null and !each.greedy() and std.mem.findScalar(u16, keys[0..candidates], minus_zero) != null);
                }
            }
        }
        const plain = if (s.top_k != 0) try fx.plainTopK(gpa, row, position, s) else try fx.plainNucleus(gpa, row, position, s);
        const got = try sampler.choose(row, position, s);
        try testing.expectEqual(plain, got);
        try testing.expectEqual(got, try sampler.choose(row, position, s)); // the scratch carries nothing over
        differing += @intFromBool(got != try argmax(row));
    }
    try testing.expect(differing > 100);
}

test "a row's start joined from its parts' is the row's, and settles the same token" {
    // GLM's ranks score half the vocabulary each (spec 045): each part's start as a pick kernel asked to a rank
    // leaves it (every token at or above the part's token of that rank), on rows coarse enough to tie across
    // the parts and at every cut.
    const gpa = testing.allocator;
    const most = 700;
    const slots = 128;
    var sampler: Sampler = try .init(gpa, most);
    defer sampler.deinit(gpa);
    var prng: std.Random.DefaultPrng = .init(45);
    const random = prng.random();
    var values: [most]f32 = undefined;
    var logits: [most]u16 = undefined;
    var settled: usize = 0;
    var nucleus: usize = 0;
    var across: usize = 0;
    var refused: usize = 0;
    defer testing.expect(settled > 600 and nucleus > 20 and across > 50 and refused > 5) catch std.debug.panic("too few rows: {d} settled, {d} by a nucleus, {d} tied across the parts, {d} refused", .{ settled, nucleus, across, refused });
    for (0..600) |round| {
        const vocab = random.intRangeAtMost(usize, 2, most);
        const spread: f32 = ([_]f32{ 0.3, 1.0, 4.0 })[round % 3];
        for (values[0..vocab]) |*v| v.* = @round(random.floatNorm(f32) * spread * 4.0) / 4.0;
        if (round % 2 == 0) for (values[0..vocab]) |*v| {
            v.* = @min(v.*, 0.0);
        };
        for (values[0..vocab]) |*v| {
            if (v.* == 0.0) v.* = if (random.boolean()) 0.0 else -0.0;
        }
        // a part with nothing finite, and rows with holes
        if (round % 5 == 0) for (values[0..vocab]) |*v| {
            if (random.boolean()) v.* = -std.math.inf(f32);
        };
        if (round % 11 == 0) @memset(values[0 .. vocab / 2], -std.math.inf(f32));
        values[vocab - 1 - random.uintLessThan(usize, vocab - vocab / 2)] = -1.0; // something finite
        fx.bf16(values[0..vocab], logits[0..vocab]);
        const row = logits[0..vocab];
        const s: Sampling = .{
            .seed = random.int(u64),
            .temperature = ([_]f64{ 0.3, 1.0, 1.7, 1e-9 })[random.uintLessThan(usize, 4)],
            .top_k = ([_]u32{ 0, 0, 1, 7, 20, 100 })[random.uintLessThan(usize, 6)],
            .top_p = ([_]f64{ 1.0, 0.95, 0.5, 0.01, 0.0 })[random.uintLessThan(usize, 5)],
            .min_p = ([_]f64{ 0.0, 0.05, 0.6, 1.0 })[random.uintLessThan(usize, 4)],
        };
        const position = random.int(u32);
        const order = try fx.sorted(gpa, row, .bits);
        defer gpa.free(order);
        var plain: f64 = 0;
        const largest: f64 = numeric.f32FromBf16(row[order[0]]);
        for (row) |bits| {
            const v: f64 = numeric.f32FromBf16(bits);
            if (v > largest - 24) plain += @exp(v - largest);
        }

        const halves = [_][2]usize{ .{ 0, vocab / 2 }, .{ vocab / 2, vocab } };
        for ([_]usize{ 64, 28, 1 }) |asked| {
            var part_keys: [2][slots]u16 = undefined;
            var part_ids: [2][slots]u32 = undefined;
            var parts: [2]sampling.Leading = undefined;
            for (halves, &parts, &part_keys, &part_ids, 0..) |half, *part, *keys, *ids, index| {
                const mine = row[half[0]..half[1]];
                const local = try fx.sorted(gpa, mine, .bits);
                defer gpa.free(local);
                const cut = keyOf(mine[local[@min(asked, mine.len) - 1]]);
                var total: usize = 0;
                while (total < mine.len and keyOf(mine[local[total]]) >= cut) total += 1;
                // more than the kernel's slots at or above the cut: it gives none
                const n = if (total > slots) 0 else total;
                for (local[0..n], keys[0..n], ids[0..n]) |column, *key, *id| {
                    key.* = keyOf(mine[column]);
                    id.* = @intCast(half[0] + column);
                }
                var smallest: u16 = 0xffff;
                for (mine) |bits| smallest = @min(smallest, keyOf(bits));
                const top: f64 = numeric.f32FromBf16(mine[local[0]]);
                var sum: f64 = 0;
                for (mine) |bits| {
                    const v: f64 = numeric.f32FromBf16(bits);
                    if (v > top - 24) sum += @exp(v - top);
                }
                // the part's sum as a kernel gives it: fp32, a little off either way; not a number where the part has
                // nothing finite
                const off: f64 = ([_]f64{ 1, 1 + 1e-5, 1 - 1e-5 })[(position + index) % 3];
                part.* = .{ .keys = keys[0..n], .ids = ids[0..n], .smallest = smallest, .len = mine.len, .sum = if (std.math.isFinite(top)) sum * off else std.math.nan(f64) };
            }
            var keys: [2 * slots]u16 = undefined;
            var ids: [2 * slots]u32 = undefined;
            const joined = sampling.joinLeading(&parts, &keys, &ids) orelse {
                try testing.expect(parts[0].keys.len == 0 or parts[1].keys.len == 0);
                refused += 1;
                continue;
            };
            // the row's own start, to the token
            try testing.expect(joined.keys.len >= @min(parts[0].keys.len, parts[1].keys.len));
            try testing.expectEqualSlices(u32, order[0..joined.ids.len], joined.ids);
            for (joined.keys, joined.ids) |key, id| try testing.expectEqual(keyOf(row[id]), key);
            try testing.expectEqual(vocab, joined.len);
            try testing.expectEqual(keyOf(row[order[vocab - 1]]), joined.smallest);
            try testing.expect(joined.sum.? >= plain * (1 - 1.1e-5) and joined.sum.? <= plain * (1 + 1.1e-5) + @as(f64, @floatFromInt(vocab)) * @exp(-24.0));
            across += @intFromBool(joined.keys.len > 1 and joined.keys[0] == joined.keys[1] and (joined.ids[0] < vocab / 2) != (joined.ids[1] < vocab / 2));

            for ([_]Sampling{ s, .{ .seed = s.seed, .temperature = 0, .top_k = s.top_k, .top_p = s.top_p, .min_p = s.min_p } }) |each| {
                const whole = sampler.choose(row, position, each) catch unreachable;
                const led = sampler.chooseLeading(joined, position, each) orelse continue;
                try testing.expectEqual(whole, led);
                settled += 1;
                nucleus += @intFromBool(each.top_k == 0 and !each.greedy());
            }
        }
    }
}

test "recorded rows scored in halves: the joined start settles TensorFold's settings as the whole row does" {
    // Rows 0, 11 and 22 of the `primes` reference logits by the fixture's 73 settings, each half's start as a
    // pick kernel asked to the settings' rank leaves it. Skipped without the reference, as the test above.
    const gpa = testing.allocator;
    const io = testing.io;
    const home = testing.environ.getPosix("HOME") orelse return error.SkipZigTest;
    const path = try std.fs.path.join(gpa, &.{ home, fx.reference });
    defer gpa.free(path);
    var file = @import("../../src/weights/safetensors.zig").File.open(gpa, io, .cwd(), path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.close();
    const tensor = file.get("logits") orelse return error.SkipZigTest;
    const vocab: usize = @intCast(tensor.shape[1]);
    const half = vocab / 2;
    const row = try gpa.alloc(u16, vocab);
    defer gpa.free(row);
    var fixture: fx.Fixture = try .open(gpa);
    defer fixture.close(gpa);
    var sampler: Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    const slots = 128;
    var settled: usize = 0;
    var nucleus: usize = 0;
    var read: usize = 0;
    for ([_]usize{ 0, 11, 22 }) |index| {
        @memcpy(std.mem.sliceAsBytes(row), tensor.data[index * vocab * 2 ..][0 .. vocab * 2]);
        var orders: [2][]u32 = undefined;
        for (&orders, 0..) |*order, part| order.* = try fx.sorted(gpa, row[part * half ..][0..half], .bits);
        defer for (orders) |order| gpa.free(order);
        for (fixture.settings, 0..) |base, case| {
            var s = base;
            s.seed = 1000 + case;
            const position: u64 = 17 * case + index;
            const whole = try sampler.choose(row, position, s);
            const asked = sampling.leadingNeeded(s, vocab) orelse {
                read += 1;
                continue;
            };
            var part_keys: [2][slots]u16 = undefined;
            var part_ids: [2][slots]u32 = undefined;
            var parts: [2]sampling.Leading = undefined;
            var given = true;
            for (&parts, orders, &part_keys, &part_ids, 0..) |*part, order, *keys, *ids, at| {
                const mine = row[at * half ..][0..half];
                const cut = keyOf(mine[order[@min(asked, half) - 1]]);
                var total: usize = 0;
                while (total < half and keyOf(mine[order[total]]) >= cut) total += 1;
                given = given and total <= slots;
                const n = if (total > slots) 0 else total;
                for (order[0..n], keys[0..n], ids[0..n]) |column, *key, *id| {
                    key.* = keyOf(mine[column]);
                    id.* = @intCast(at * half + column);
                }
                const top: f64 = numeric.f32FromBf16(mine[order[0]]);
                var sum: f64 = 0;
                for (mine) |bits| {
                    const v: f64 = numeric.f32FromBf16(bits);
                    if (v > top - 24) sum += @exp(v - top);
                }
                part.* = .{ .keys = keys[0..n], .ids = ids[0..n], .smallest = keyOf(mine[order[half - 1]]), .len = half, .sum = sum };
            }
            var keys: [2 * slots]u16 = undefined;
            var ids: [2 * slots]u32 = undefined;
            const joined = sampling.joinLeading(&parts, &keys, &ids);
            try testing.expectEqual(given, joined != null);
            const led = sampler.chooseLeading(joined orelse continue, position, s) orelse {
                read += 1;
                continue;
            };
            try testing.expectEqual(whole, led);
            settled += 1;
            nucleus += @intFromBool(s.top_k == 0 and !s.greedy());
        }
    }
    // most of TensorFold's cases are settled by the halves' starts; the rest read the row
    testing.expect(settled > 3 * 73 / 2 and nucleus > 0 and settled + read == 3 * 73) catch |err| {
        std.debug.print("{d} settled, {d} by a nucleus, {d} read\n", .{ settled, nucleus, read });
        return err;
    };
}

test "over many positions a row's picks follow its probabilities" {
    // 50,000 positions under one seed. A token of probability p is picked n times with standard deviation
    // sqrt(50000 p (1 - p)); the bound is five of those, under 0.012 in frequency for any p.
    const gpa = testing.allocator;
    var sampler: Sampler = try .init(gpa, 6);
    defer sampler.deinit(gpa);
    const values = [6]f32{ 3.0, 2.0, 1.5, 1.0, 1.25, 3.0 };
    var logits: [6]u16 = undefined;
    fx.bf16(&values, &logits);
    const positions = 50_000;

    const Case = struct { s: Sampling, kept: []const usize };
    const everything = [_]usize{ 0, 1, 2, 3, 4, 5 };
    for ([_]Case{
        .{ .s = .{ .seed = 1, .top_k = 0, .top_p = 1.0 }, .kept = &everything }, // the whole-row rule
        .{ .s = .{ .seed = 2, .top_k = 6, .top_p = 1.0, .temperature = 0.7 }, .kept = &everything }, // top-k's
        // top-k 4 keeps ids 0, 5, 1, 2 with probabilities 0.386, 0.386, 0.142, 0.086: 0.9 is reached at the third
        .{ .s = .{ .seed = 3, .top_k = 4, .top_p = 0.9 }, .kept = &.{ 0, 5, 1 } },
        // of the whole row's mass ids 0 and 5 hold 0.345 each and id 1 0.127: 0.75 is reached at id 1
        .{ .s = .{ .seed = 4, .top_k = 0, .top_p = 0.75 }, .kept = &.{ 0, 5, 1 } },
        // within ln(0.3) = -1.2 of the top: ids 0, 5 and 1
        .{ .s = .{ .seed = 5, .top_k = 0, .top_p = 1.0, .min_p = 0.3 }, .kept = &.{ 0, 5, 1 } },
    }) |case| {
        var counts: [6]usize = @splat(0);
        var first: [64]i32 = undefined;
        for (0..positions) |position| {
            const got = try sampler.choose(&logits, position, case.s);
            if (position < first.len) first[position] = got;
            counts[@intCast(got)] += 1;
        }
        var expected: [6]f64 = @splat(0.0);
        var sum: f64 = 0.0;
        for (case.kept) |id| {
            expected[id] = @exp(@as(f64, values[id]) / case.s.temperature);
            sum += expected[id];
        }
        for (counts, expected) |count, weight| {
            const p = weight / sum;
            const n: f64 = positions;
            const deviation = @abs(@as(f64, @floatFromInt(count)) - n * p);
            try testing.expect(deviation <= 5.0 * @sqrt(n * p * (1.0 - p)));
        }
        // the same key and position give the same token again, and another seed gives another sequence
        var other = case.s;
        other.seed += 100;
        var moved = false;
        for (first, 0..) |token, position| {
            try testing.expectEqual(token, try sampler.choose(&logits, position, case.s));
            moved = moved or token != try sampler.choose(&logits, position, other);
        }
        try testing.expect(moved);
    }
}

test "the mass the nucleus needs is the exact ceiling" {
    try testing.expectEqual(1, massNeeded(0.5, 2));
    try testing.expectEqual(2, massNeeded(0.5, 3));
    // 0.1 as a double is a little above 1/10, so 10 units need 2 of them, though 0.1 * 10 rounds to 1 in fp64
    try testing.expect(@as(f64, 0.1) * 10.0 == 1.0);
    try testing.expectEqual(2, massNeeded(0.1, 10));
    // and 0.95 is a little below 19/20
    try testing.expectEqual(19, massNeeded(0.95, 20));
    // against 128-bit arithmetic at the largest total a 248,320-token row can have
    const total: u64 = 248_320 << 40;
    const p: f64 = 0.95;
    const mantissa: u128 = @intFromFloat(p * 0x1p53);
    try testing.expectEqual(@as(u64, @intCast((mantissa * total + (1 << 53) - 1) >> 53)), massNeeded(p, total));
    try testing.expectEqual(1, massNeeded(std.math.floatTrueMin(f64), total));
    try testing.expectEqual(1, massNeeded(0x1p-200, total));
}

test "numpy's order of additions" {
    var values: [300]f64 = undefined;
    for (&values, 0..) |*v, i| v.* = 1.0 / @as(f64, @floatFromInt(i + 1));
    // np.sum(1 / np.arange(1, n + 1)) for n = 7, 100, 131 and 300, as bits: one running sum, eight of them,
    // and two depths of halving
    try testing.expectEqual(@as(u64, 0x4004be2be2be2be2), @as(u64, @bitCast(pairwiseSum(values[0..7]))));
    try testing.expectEqual(@as(u64, 0x4014bfdfe4591243), @as(u64, @bitCast(pairwiseSum(values[0..100]))));
    try testing.expectEqual(@as(u64, 0x4015d32ca0840eb8), @as(u64, @bitCast(pairwiseSum(values[0..131]))));
    try testing.expectEqual(@as(u64, 0x40192172a3e6965a), @as(u64, @bitCast(pairwiseSum(values[0..300]))));
}

test "settings are validated and rows without a finite top are refused" {
    try (Sampling{ .seed = 0 }).validate();
    try (Sampling{ .seed = 0, .temperature = 0.0, .top_k = 0, .top_p = 1.0, .min_p = 1.0 }).validate();
    try testing.expectError(error.InvalidTemperature, (Sampling{ .seed = 0, .temperature = std.math.nan(f64) }).validate());
    try testing.expectError(error.InvalidTopP, (Sampling{ .seed = 0, .top_p = std.math.inf(f64) }).validate());
    try testing.expectError(error.InvalidMinP, (Sampling{ .seed = 0, .min_p = 1.5 }).validate());
    try testing.expectError(error.InvalidMinP, (Sampling{ .seed = 0, .min_p = -0.1 }).validate());
    try testing.expectError(error.InvalidMinP, (Sampling{ .seed = 0, .min_p = std.math.nan(f64) }).validate());
    try testing.expect((Sampling{ .seed = 0, .temperature = 0.0 }).greedy());
    try testing.expect(!(Sampling{ .seed = 0 }).greedy());

    const gpa = testing.allocator;
    var sampler: Sampler = try .init(gpa, 40);
    defer sampler.deinit(gpa);
    var values: [40]f32 = @splat(1.0);
    var logits: [40]u16 = undefined;
    const inf = std.math.inf(f32);
    for ([_]f32{ std.math.nan(f32), -std.math.nan(f32), inf }) |bad| {
        values[37] = bad;
        fx.bf16(&values, &logits);
        for ([_]u32{ 0, 20 }) |top_k| {
            try testing.expectError(error.NonFiniteLogits, sampler.choose(&logits, 0, .{ .seed = 0, .top_k = top_k }));
        }
    }
    try testing.expectError(error.NonFiniteLogits, argmax(&logits));
    values = @splat(-inf);
    fx.bf16(&values, &logits);
    try testing.expectError(error.NonFiniteLogits, sampler.choose(&logits, 0, .{ .seed = 0 }));
    try testing.expectError(error.NonFiniteLogits, argmax(&logits));
    values[3] = -2.0; // one finite logit among masked ones is always the pick
    fx.bf16(&values, &logits);
    for ([_]u32{ 0, 20 }) |top_k| {
        try testing.expectEqual(3, try sampler.choose(&logits, 9, .{ .seed = 0, .top_k = top_k, .top_p = 1.0 }));
    }
    try testing.expectEqual(3, try argmax(&logits));
}

test "a temperature below 0 is greedy and a top_p outside 0 to 1 cuts nothing, as TensorFold decodes them" {
    // What `sampling_for` and `choose_rows` do with what `parse_numbers` lets through (spec 056): `temp <= 0`
    // is no sampling at all, and the nucleus is cut only when `0.0 < top_p < 1.0`.
    const gpa = testing.allocator;
    var sampler: Sampler = try .init(gpa, 40);
    defer sampler.deinit(gpa);
    var values: [40]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @floatFromInt(@mod(i * 7, 11));
    var logits: [40]u16 = undefined;
    fx.bf16(&values, &logits);

    for ([_]f64{ -1.0, -1e-9, -0.0, -1e300 }) |temperature| {
        const settings: Sampling = .{ .seed = 3, .temperature = temperature };
        try settings.validate();
        try testing.expect(settings.greedy());
        for (0..8) |position| try testing.expectEqual(try argmax(&logits), try sampler.choose(&logits, position, settings));
    }
    for ([_]f64{ 2.5, 100.0, 1e300 }) |temperature| {
        const settings: Sampling = .{ .seed = 3, .temperature = temperature };
        try settings.validate();
        try testing.expect(!settings.greedy());
        _ = try sampler.choose(&logits, 0, settings);
    }
    // each value outside picks what top_p 1 picks, with top-k on and off
    for ([_]f64{ 0.0, -0.5, -1e300, 1.5, 2.0, 1e300 }) |top_p| {
        for ([_]u32{ 0, 20 }) |top_k| {
            const settings: Sampling = .{ .seed = 3, .temperature = 1.5, .top_k = top_k, .top_p = top_p };
            try settings.validate();
            try testing.expect(!settings.cuts());
            var open = settings;
            open.top_p = 1.0;
            for (0..32) |position| try testing.expectEqual(try sampler.choose(&logits, position, open), try sampler.choose(&logits, position, settings));
        }
    }
}

test "a checkpoint's generation config gives the defaults, and none is greedy" {
    const gpa = std.testing.allocator;
    const cases = [_]struct { []const u8, Sampling }{
        .{ "{\"do_sample\": true, \"temperature\": 1.0, \"top_k\": 20, \"top_p\": 0.95, \"eos_token_id\": [1, 2]}", .{ .seed = 7 } },
        .{ "{\"do_sample\": true}", .{ .seed = 7 } },
        .{ "{\"do_sample\": false, \"temperature\": 0.7, \"top_k\": 40}", .{ .seed = 7, .temperature = 0, .top_k = 40 } },
        .{ "{\"temperature\": 0.6, \"top_k\": 0, \"top_p\": 1, \"min_p\": 0.05}", .{ .seed = 7, .temperature = 0.6, .top_k = 0, .top_p = 1, .min_p = 0.05 } },
        .{ "{\"temperature\": null, \"top_k\": -3}", .{ .seed = 7, .temperature = 0, .top_k = 0 } },
        .{ "{}", .{ .seed = 7, .temperature = 0 } },
        .{ "[]", .{ .seed = 7, .temperature = 0 } },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, case[0], .{});
        defer parsed.deinit();
        try std.testing.expectEqual(case[1], Sampling.fromGenerationConfig(parsed.value, 7));
    }
    try std.testing.expect(Sampling.fromGenerationConfig(null, 7).greedy());
}
