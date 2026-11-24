//! `mtp.zig`'s tests of a draft priced by the round its seat shares (`Paying.Round`, spec 077): one seat is
//! priced as alone to the bit, several raise the floors, rows beside lower a row's price; and of the
//! levels a reply counts (`Paying.Counts`). No device: the rule is the host's. The rule of a stream alone: `mtp_test.zig`.

const std = @import("std");
const mtp_file = @import("../../../src/models/qwen4/mtp.zig");
const masks = @import("../../../src/decode/masks.zig");
const Paying = mtp_file.Paying;
const Round = Paying.Round;
const Rule = mtp_file.Rule;
const draft = @import("../../../src/decode/draft.zig");
const testing = std.testing;

const history = struct {
    /// A reply's rounds as `Paying.done` is told of them: the tokens a round gave and its chain, some rounds
    /// keeping all, some none, some whose head said more than was kept.
    fn round(random: std.Random) struct { tokens: usize, chain: Paying.Told } {
        const drafts = random.intRangeAtMost(usize, 1, masks.max_rows - 1);
        const scored = @min(drafts + random.intRangeAtMost(usize, 0, 1), masks.max_rows - 1);
        const tokens = 1 + random.intRangeAtMost(usize, 0, drafts);
        return .{ .tokens = tokens, .chain = .{ .drafts = drafts, .scored = scored, .said = if (drafts > 1 and random.boolean()) random.float(f64) * @as(f64, @floatFromInt(drafts - 1)) else 0 } };
    }

    /// A stream after `rounds` rounds of prose: two tokens of three rows, the head saying more than is kept.
    fn prose(costs: Paying.Costs, rounds: usize) Paying {
        var seat: Paying = .begin(costs);
        for (0..rounds) |_| seat.done(2, .{ .drafts = 3, .scored = 3, .said = 1.2 });
        return seat;
    }
};

/// A storage with no round of several seats, as NVFP4.
const untabled: Paying.Costs = .{ .row = 0.065, .level = 0.032 };

/// A storage with a round of several seats and bands of rows: the retired EXL3 pack's figures (tag
/// retired-checkpoints-2026-10-21), kept as the rule's case.
const tabled: Paying.Costs = .{ .row = 0.098, .level = 0.043, .shared = 0.82, .rows = &.{
    .{ .upto = 32, .of_row = 1 },
    .{ .upto = 64, .of_row = 0.6 },
    .{ .upto = std.math.maxInt(usize), .of_row = 0.46 },
} };

/// A fitted calibration: the retired 3.05-bit pack's (`tools/reference/lands_fit.py`, 68 replies).
const fitted_calibration: Paying.Landing = .{ .slope = 0.543, .levels = .{ -0.24, -0.57, -0.67, -0.84, -0.85, -1.10 } };

fn bits(floors: [2]f64) [2]u64 {
    return .{ @bitCast(floors[0]), @bitCast(floors[1]) };
}

test "paying by the round: one seat alone is priced as without it, the same floors to the bit over a reply's rounds" {
    for ([_]Paying.Costs{ tabled, .{ .row = 0.065, .level = 0.032, .shared = 0.9, .rows = &.{.{ .upto = masks.max_rows, .of_row = 1 }} } }) |costs| {
        var prng: std.Random.DefaultPrng = .init(77);
        const random = prng.random();
        var seat: Paying = .begin(costs);
        for (0..400) |_| {
            var round: Round = .{};
            round.add(seat);
            // whatever the lone seat's window held, one row to the most a window takes
            for (1..masks.max_rows + 1) |rows| {
                const priced = seat.within(round.price(costs, rows));
                try testing.expect(priced.round != null);
                try testing.expectEqual(bits(seat.floors()), bits(priced.floors()));
                // and so the same draft is low or closing
                const through = random.float(f64);
                try testing.expectEqual((Rule{ .paying = seat }).little(through, through, 1), (Rule{ .paying = priced }).little(through, through, 1));
            }
            const next = history.round(random);
            seat.done(next.tokens, next.chain);
        }
    }
    // no seat decodes: nothing is priced, and a seat given no round is alone
    try testing.expectEqual(@as(?Round.Price, null), (Round{}).price(tabled, 4));
    const alone = history.prose(tabled, 8);
    try testing.expectEqual(bits(alone.floors()), bits(alone.within(null).floors()));
}

test "paying by the round: four seats of one history hold a draft to more than one of them alone does, by what the round pays once" {
    const costs = tabled;
    for ([_]usize{ 0, 1, 8, 60 }) |rounds| {
        const seat = history.prose(costs, rounds);
        var round: Round = .{};
        var before = seat.floors();
        for (1..5) |seats| {
            round.add(seat);
            // the seats' windows, under the first band of rows: the row is the lone one's
            const price = round.price(costs, 3 * seats).?;
            try testing.expectEqual(@as(f64, 1), price.of_row);
            const floors = seat.within(price).floors();
            if (seats > 1) try testing.expect(floors[0] > before[0] and floors[1] > before[1]);
            before = floors;
            // the rate is the seats' tokens over their costs less the shared part of each seat past the first
            const n: f64 = @floatFromInt(seats);
            try testing.expectApproxEqRel(n * seat.tokens / (n * seat.cost - costs.shared * (n - 1)), price.rate, 1e-12);
        }
        // four seats' rate is 1.6 to 2.7 times a lone seat's, never the four times of a round that costs
        // one seat's whatever it holds
        const four = round.price(costs, 12).?.rate / seat.rate();
        try testing.expect(four > 1.6 and four < 2.7);
        // a draft between the lone floor and the round's is verified by a seat alone and not in the round
        const lone = seat.floors();
        const shared = seat.within(round.price(costs, 12)).floors();
        const through = (lone[0] + shared[0]) / 2;
        try testing.expect(!(Rule{ .paying = seat }).little(0.9, through, 1).low);
        try testing.expect((Rule{ .paying = seat.within(round.price(costs, 12)) }).little(0.9, through, 1).low);
        // the trust is the seat's own: a seat whose head is kept less than it says pays more in the same round
        var doubted = seat;
        for (0..40) |_| doubted.done(1, .{ .drafts = 4, .scored = 4, .said = 2 });
        const price = round.price(costs, 12);
        var trusted = doubted;
        trusted.got = 1;
        trusted.said = 1;
        try testing.expect(doubted.within(price).floors()[0] > trusted.within(price).floors()[0]);
    }
    // a storage with no round of several seats prices each of four seats alone, to the bit
    const seat = history.prose(untabled, 20);
    var round: Round = .{};
    for (0..4) |_| round.add(seat);
    try testing.expectEqual(@as(?Round.Price, null), round.price(untabled, 400));
    try testing.expectEqual(@as(?Round.Price, null), round.price(Paying.published, 400));
    try testing.expectEqual(bits(seat.floors()), bits(seat.within(round.price(untabled, 400)).floors()));
}

test "paying by the round: a prompt chunk's rows beside the replies lower a row's price, and a chain then runs deeper" {
    const costs = tabled;
    const seat = history.prose(costs, 20);
    var round: Round = .{};
    for (0..3) |_| round.add(seat);
    // three replies of 12 rows: alone, beside 16 rows of a prompt, beside 64 and beside 512
    const alone = seat.within(round.price(costs, 12)).floors();
    const beside16 = seat.within(round.price(costs, 12 + 16)).floors();
    const beside64 = seat.within(round.price(costs, 12 + 64)).floors();
    const beside512 = seat.within(round.price(costs, 12 + 512)).floors();
    // under 32 rows a row is a row (spec 075: 2.1 to 2.5 ms in every group of rounds)
    try testing.expectEqual(bits(alone), bits(beside16));
    try testing.expect(beside64[0] < alone[0] and beside64[1] < alone[1]);
    try testing.expectEqual(bits(beside64), bits(beside512));
    // the bands fall and never rise, whatever the rows
    var last: f64 = 1;
    for (1..700) |rows| {
        const of_row = costs.rowAt(rows);
        try testing.expect(of_row <= last and of_row > 0);
        last = of_row;
    }
    try testing.expectEqual(@as(f64, 1), costs.rowAt(masks.max_rows));
    try testing.expectEqual(@as(f64, 1), untabled.rowAt(4096));
    // a level of the head is not a row: only the row's part of the second floor falls
    const rate = round.price(costs, 76).?.rate;
    try testing.expectApproxEqRel(@sqrt(rate / std.math.clamp(seat.got / seat.said, 0.25, 1) * (costs.row * costs.rowAt(76) + costs.level)), beside64[1], 1e-12);
    // a draft the round of replies alone would not verify is verified beside the chunk
    const through = (alone[0] + beside64[0]) / 2;
    try testing.expect((Rule{ .paying = seat.within(round.price(costs, 12)) }).little(0.9, through, 1).low);
    try testing.expect(!(Rule{ .paying = seat.within(round.price(costs, 76)) }).little(0.9, through, 1).low);
}

/// A chain of the head's own drafts as a round is told of it: each of `ps` taken, then one more scored and
/// dropped where `dropped`.
fn toldOf(ps: []const f64, dropped: bool) !Paying.Told {
    var out: [draft.forward_max_rows]i32 = undefined;
    var chain: draft.Chain = try .begin(&out, null);
    for (ps) |p| try chain.take(null, 7, p, false, false);
    if (dropped) try chain.take(null, 7, 0.1, true, false);
    return Paying.told(chain);
}

test "a reply's levels are counted by the depth the chain stood at: run, taken, kept; a dropped draft is run and not taken, a forced token counts nothing" {
    var seat: Paying = .begin(Paying.published);
    const chain = try toldOf(&.{ 0.9, 0.9, 0.9, 0.9, 0.9 }, false);
    for (0..200) |_| seat.done(6, chain);
    // then the main model keeps two of the five: the levels at depths 2, 3 and 4 gave drafts it refused
    for (0..200) |_| seat.done(3, chain);
    try testing.expectEqual([Paying.deep]usize{ 400, 400, 400, 400 }, seat.counts.passes);
    try testing.expectEqual([Paying.deep]usize{ 400, 400, 400, 400 }, seat.counts.taken);
    try testing.expectEqual([Paying.deep]usize{ 400, 200, 200, 200 }, seat.counts.kept);
    // two drafts taken and a third the rule dropped: the level at depth 2 ran and gave nothing
    seat.done(3, try toldOf(&.{ 0.9, 0.9 }, true));
    try testing.expectEqual([Paying.deep]usize{ 401, 401, 400, 400 }, seat.counts.passes);
    try testing.expectEqual([Paying.deep]usize{ 401, 400, 400, 400 }, seat.counts.taken);
    try testing.expectEqual([Paying.deep]usize{ 401, 200, 200, 200 }, seat.counts.kept);
    // past the third draft the depths are one
    const ps: [15]f64 = @splat(0.99);
    seat.done(16, try toldOf(&ps, false));
    try testing.expectEqual([Paying.deep]usize{ 402, 402, 401, 411 }, seat.counts.passes);
    // a chain a constraint forced a token into says nothing of depths
    var forced = chain;
    forced.drafts += 1;
    const held = seat;
    seat.done(3, forced);
    try testing.expectEqual(held.counts.passes, seat.counts.passes);
    // the counts change no floor
    var twin: Paying = .begin(Paying.published);
    twin.done(6, chain);
    var plain: Paying = .begin(Paying.published);
    plain.done(6, .{ .drafts = chain.drafts, .scored = chain.scored, .said = chain.said });
    try testing.expectEqual(bits(plain.floors()), bits(twin.floors()));
}

test "a fitted calibration: the replay's numbers, the product paying untrusted, and the square's floors and what a round is told as without it" {
    const fitted = fitted_calibration;
    // sigma(0.543 logit p + b of the level), the level's last from five on
    try testing.expectApproxEqAbs(@as(f64, 1.0) / (1.0 + @exp(-(0.543 * @log(@as(f64, 9.0)) - 0.24))), fitted.of(0.9, 0), 1e-12);
    try testing.expectEqual(fitted.of(0.7, 5), fitted.of(0.7, 12));
    try testing.expect(fitted.of(0.7, 0) > fitted.of(0.7, 1) and fitted.of(0.99, 3) > fitted.of(0.9, 3));
    // a sure draft is said to land less often than the head says, a doubtful one more
    try testing.expect(fitted.of(0.999, 1) < 0.999 and fitted.of(0.2, 0) > 0.2);
    try testing.expectEqual(fitted.of(1, 0), fitted.of(1 - 1e-6, 0));
    const trail = [_]draft.Chain.Considered{ .{ .token = 1, .p = 0.9 }, .{ .token = 2, .p = 0.8 } };
    try testing.expectApproxEqAbs(fitted.of(trail[0].p, 0) * fitted.of(trail[1].p, 1) * fitted.of(0.7, 2), fitted.through(&trail, 0.7), 1e-12);
    try testing.expect(fitted.through(&trail, 0.69) < fitted.through(&trail, 0.7));

    // what a round is told the chain said is the squares, a calibration or none: the trust is the square's
    try testing.expectApproxEqAbs(@as(f64, 0.72 * 0.72 + 0.504 * 0.504), (try toldOf(&.{ 0.9, 0.8, 0.7 }, false)).said, 1e-6);

    // the thresholds are the round's rate times a row, and a row and a level; the trust does not move them
    var costs = tabled;
    costs.landing = fitted;
    var seat: Paying = .begin(costs);
    try testing.expect(seat.landing != null and Paying.begin(tabled).landing == null);
    const row = seat.rate() * seat.row;
    const level = seat.rate() * (seat.row + seat.level);
    try testing.expectEqual(@TypeOf(seat.pays(0)){ .low = false, .closing = false }, seat.pays(level * 1.01));
    try testing.expectEqual(@TypeOf(seat.pays(0)){ .low = false, .closing = true }, seat.pays((row + level) / 2));
    try testing.expectEqual(@TypeOf(seat.pays(0)){ .low = true, .closing = true }, seat.pays(row * 0.99));
    var doubted = seat;
    doubted.got = 0.01;
    try testing.expect(!doubted.pays(row * 1.01).low and doubted.pays(row * 0.99).low);
    // a round of several seats prices it as it prices the square
    var round: Round = .{};
    for (0..3) |_| round.add(seat);
    const shared = seat.within(round.price(costs, 12));
    try testing.expect(shared.pays(row * 1.01).low and !seat.pays(row * 1.01).low);
    // the floors of the probability squared do not read the calibration
    try testing.expectEqual(bits(Paying.begin(tabled).floors()), bits(seat.floors()));
}
