//! `hall.zig`'s tests on the random GLM in loopback: replies decoded several at once, each held token for
//! token to the same reply by serial steps of the model alone (`serialReply`: no window, no draft, no
//! other stream), greedy and sampled, drafted by the MTP layer or not, with streams that come and go, an end
//! token, and a prompt that goes on from what a slot holds.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const flash_file = @import("../../../../src/models/glm5/dflash2/drafter.zig");
const flash_pass = @import("../../../../src/models/glm5/dflash2/pass.zig");
const dflash2_pass = @import("../../../../src/decode/dflash2/pass.zig");
const flash_testing = @import("../../../../src/models/glm5/checks/random_drafter.zig");
const flash_weights = @import("../../../../src/decode/dflash2/weights.zig");
const draft_file = @import("../../../../src/models/glm5/draft.zig");
const hall_file = @import("../../../../src/models/glm5/pool/hall.zig");
const hall_fixture = @import("hall_fixture.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const mtp = @import("../../../../src/models/glm5/mtp/mtp.zig");
const random_file = @import("../../../../src/models/glm5/checks/random.zig");
const shared = @import("../testing.zig");

const fx = random_file.fx;
const Case = hall_fixture.Case;
const Served = hall_fixture.Served;
const serve = hall_fixture.serve;
const Fixture = hall_fixture.Fixture;
const greedy = hall_fixture.greedy;
const sampled = hall_fixture.sampled;
const Hall = hall_file.HallOf(layout.world);

test "replies decoded several at once are each the serial reply of the model alone: greedy and sampled, drafted by the MTP layer or not, streams coming and going" {
    // held to serial steps: one form, a step's (`testing.oneForm`)
    const forms = shared.oneForm();
    defer forms.restore();
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const capacity = 160;
    var f: Fixture = undefined;
    try f.init(0x4a11, 400, capacity);
    defer f.deinit();
    const ids = f.fixture.tokens;
    const none: []const u32 = &.{};
    // more requests than slots; prompts of one chunk and of several beside replies; a prompt of one token
    const cases = [_]Case{
        .{ .request = .{ .prompt = ids[0..23], .settings = greedy, .eos = none, .max_tokens = 30 } },
        .{ .request = .{ .prompt = ids[30..37], .settings = sampled(11), .eos = none, .max_tokens = 41 } },
        .{ .request = .{ .prompt = ids[40..41], .settings = sampled(12), .eos = none, .max_tokens = 9 }, .start = 1 },
        .{ .request = .{ .prompt = ids[50..163], .settings = greedy, .eos = none, .max_tokens = 17 }, .start = 3 },
        .{ .request = .{ .prompt = ids[170..185], .settings = sampled(13), .eos = none, .max_tokens = 1 }, .start = 3 },
        .{ .request = .{ .prompt = ids[190..250], .settings = sampled(14), .eos = none, .max_tokens = 33, .drafts = false }, .start = 4 },
        .{ .request = .{ .prompt = ids[260..269], .settings = greedy, .eos = none, .max_tokens = 25 }, .start = 9 },
    };
    var want: [cases.len][]i32 = undefined;
    for (&want, cases) |*serial, case| serial.* = try shared.serialReply(arena, f.loop.m, fx.vocab, case.request.prompt, case.request.max_tokens, case.request.settings);

    for ([_]?f32{ null, 0, mtp.Head.default_confidence }) |confidence| {
        if (confidence) |value| f.head.confidence = value;
        for ([_]usize{ 1, 3, 4 }) |slots| {
            var hall = try f.hall(slots, capacity, confidence != null);
            defer hall.deinit();
            // every round through a graph a round's shape, captured at its first sight: the same replies
            // out of them (spec 099 WS-03)
            hall.graphs = try .init(gpa);
            const served = try serve(arena, &hall, &cases);
            var drafted: usize = 0;
            var accepted: usize = 0;
            for (want, served.tokens, served.ended, cases) |serial, got, ended, case| {
                try testing.expectEqualSlices(i32, serial, got.items);
                try testing.expectEqual(hall_file.Finish.length, ended.?.finish);
                try testing.expectEqual(case.request.max_tokens - 1, ended.?.stats.rounds + ended.?.stats.accepted);
                drafted += ended.?.stats.drafted;
                accepted += ended.?.stats.accepted;
            }
            try testing.expectEqual(@min(slots, 4), served.beside);
            try testing.expectEqual(@as(usize, 0), hall.live());
            try testing.expect(hall.graphs.?.cache.count() > 0);
            if (confidence == null) try testing.expectEqual(@as(usize, 0), drafted);
            // whole chains where no probability ends them, and some of a random model's drafts kept
            if (confidence != null and confidence.? == 0) try testing.expect(drafted > 100 and accepted > 0);
            // streams that share a round take fewer rounds than their windows
            if (slots > 1) try testing.expect(hall.rounds < served.rounds + 1 and hall.rounds + 20 < windows(served));
        }
    }

    // a stream's drafts beside others are its drafts alone: windows as deep, each reply's rounds, drafts and
    // drafts kept the same from a hall of one slot and of four
    var alone: [cases.len]draft_file.Stats = undefined;
    for ([_]f32{ 0, mtp.Head.default_confidence }) |confidence| {
        f.head.confidence = confidence;
        for ([_]usize{ 1, 4 }) |slots| {
            var hall = try f.hall(slots, capacity, true);
            defer hall.deinit();
            hall.depth = 3;
            const served = try serve(arena, &hall, &cases);
            for (served.ended, &alone) |ended, *one| {
                if (slots == 1) one.* = ended.?.stats;
                try testing.expectEqual(one.*, ended.?.stats);
            }
        }
    }
}

test "replies drafted by DFlash2 several at once, each from its slot's own context, flat and in the ring, are each the serial reply" {
    // held to serial steps: one form, a step's (`testing.oneForm`)
    const forms = shared.oneForm();
    defer forms.restore();
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const capacity = 160;
    // `dflash2/drafter_test.zig`'s random drafter over the random model, tapped
    var fixture: shared.Fixture = undefined;
    try fixture.init(0x4a13, 300);
    defer fixture.deinit();
    const inner = fixture.arena_state.allocator();
    var c = try flash_testing.config(inner, fx.dims, fx.vocab, 5, &.{ 1, 3 });
    c.block = 8;
    var source: flash_testing.Source = .empty;
    var prng: std.Random.DefaultPrng = .init(0x4a14);
    try flash_testing.put(inner, prng.random(), c, &source);
    var loop = try fixture.loop(capacity);
    defer loop.deinit();
    try loop.b.setTaps(c.taps);
    const rows: cuda.Scratch.Rows = .{ .pool = &fixture.pool, .count = dflash2_pass.batch };
    var w: [layout.world]flash_weights.Weights = undefined;
    w[0] = try .load(inner, gpa, source, &c, 0, .mixed, rows);
    defer w[0].deinit();
    w[1] = try .load(inner, gpa, source, &c, 1, .mixed, rows);
    defer w[1].deinit();
    var books: flash_weights.Codebooks = try .load(gpa, source, &c);
    defer books.deinit(gpa);
    for (books.predecessor) |*value| value.* *= 12;
    var sampler: sampling.Sampler = try .init(gpa, fx.vocab);
    defer sampler.deinit(gpa);
    const logits = try arena.alloc(u16, fx.vocab);

    const ids = fixture.tokens;
    const none: []const u32 = &.{};
    const cases = [_]Case{
        .{ .request = .{ .prompt = ids[0..23], .settings = greedy, .eos = none, .max_tokens = 30 } },
        .{ .request = .{ .prompt = ids[30..37], .settings = sampled(51), .eos = none, .max_tokens = 41 } },
        .{ .request = .{ .prompt = ids[40..41], .settings = sampled(52), .eos = none, .max_tokens = 9 }, .start = 1 },
        .{ .request = .{ .prompt = ids[50..163], .settings = greedy, .eos = none, .max_tokens = 17 }, .start = 3 },
        .{ .request = .{ .prompt = ids[190..250], .settings = sampled(54), .eos = none, .max_tokens = 33, .drafts = false }, .start = 4 },
        .{ .request = .{ .prompt = ids[260..269], .settings = greedy, .eos = none, .max_tokens = 25 }, .start = 9 },
    };
    var want: [cases.len][]i32 = undefined;
    for (&want, cases) |*serial, case| serial.* = try shared.serialReply(arena, loop.m, fx.vocab, case.request.prompt, case.request.max_tokens, case.request.settings);

    // a stream's drafts alone, in a hall of one slot: what it is given beside others
    var alone: [2][cases.len]draft_file.Stats = undefined;
    for ([_]bool{ false, true }) |ring| for ([_]usize{ 1, 3 }) |slots| {
        var pair: flash_pass.Pair = try .init(gpa, w, &c, capacity, ring, fx.vocab / layout.world);
        defer pair.deinit();
        var d: flash_file.Drafter = try .init(gpa, &c, &pair, .{ .m = loop.m, .main = &loop.b }, books, greedy);
        defer d.deinit();
        d.confidence = if (ring) 0 else flash_file.Drafter.default_confidence;
        var hall: Hall = try .init(gpa, loop.m, &loop.b, slots, capacity, fx.vocab, &sampler, logits, .{ .flash = &d });
        defer hall.deinit();
        // windows as deep whatever the slots, so that a stream's rounds are the same rounds
        hall.depth = 3;
        const served = try serve(arena, &hall, &cases);
        var drafted: usize = 0;
        for (want, served.tokens, served.ended, &alone[@intFromBool(ring)]) |serial, got, ended, *one| {
            try testing.expectEqualSlices(i32, serial, got.items);
            drafted += ended.?.stats.drafted;
            if (slots == 1) one.* = ended.?.stats;
            try testing.expectEqual(one.*, ended.?.stats);
        }
        try testing.expectEqual(slots, served.beside);
        try testing.expect(drafted > 20 and d.passes > 20);

        // a second turn goes on from its slot's context beside a new prompt
        const turns = [_]Case{
            .{ .request = .{ .prompt = try std.mem.concat(arena, i32, &.{ cases[5].request.prompt, served.tokens[5].items, ids[280..286] }), .settings = sampled(61), .eos = none, .max_tokens = 12 } },
            .{ .request = .{ .prompt = ids[100..131], .settings = sampled(62), .eos = none, .max_tokens = 12 } },
        };
        const second = try serve(arena, &hall, &turns);
        for (turns, second.tokens) |turn, got| {
            const serial = try shared.serialReply(arena, loop.m, fx.vocab, turn.request.prompt, turn.request.max_tokens, turn.request.settings);
            try testing.expectEqualSlices(i32, serial, got.items);
        }
        try testing.expectEqual(cases[5].request.prompt.len + served.tokens[5].items.len - 1, second.admitted[0].?.cached);
    };
}

fn windows(served: Served) usize {
    var sum: usize = 0;
    for (served.ended) |ended| sum += ended.?.stats.rounds;
    return sum;
}

test "a reply ends at its end token beside others; a prompt that goes on from what a slot holds runs the rest alone and gives the serial reply" {
    // held to serial steps: one form, a step's (`testing.oneForm`)
    const forms = shared.oneForm();
    defer forms.restore();
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const capacity = 160;
    var f: Fixture = undefined;
    try f.init(0x4a12, 300, capacity);
    defer f.deinit();
    const ids = f.fixture.tokens;
    const none: []const u32 = &.{};
    f.head.confidence = 0;

    const first = [_]Case{
        .{ .request = .{ .prompt = ids[0..19], .settings = sampled(21), .eos = none, .max_tokens = 24 } },
        .{ .request = .{ .prompt = ids[20..31], .settings = greedy, .eos = none, .max_tokens = 40 } },
    };
    var serial: [2][]i32 = undefined;
    for (&serial, first) |*one, case| one.* = try shared.serialReply(arena, f.loop.m, fx.vocab, case.request.prompt, case.request.max_tokens, case.request.settings);
    // the first reply ends at the token its tenth is, where that comes first
    const end = [_]u32{@intCast(serial[0][9])};
    const stops = std.mem.indexOfScalar(i32, serial[0], serial[0][9]).? + 1;

    for ([_]bool{ false, true }) |drafted| {
        var hall = try f.hall(3, capacity, drafted);
        defer hall.deinit();
        var cases = first;
        cases[0].request.eos = &end;
        const served = try serve(arena, &hall, &cases);
        try testing.expectEqualSlices(i32, serial[0][0..stops], served.tokens[0].items);
        try testing.expectEqual(hall_file.Finish.stop, served.ended[0].?.finish);
        try testing.expectEqualSlices(i32, serial[1], served.tokens[1].items);

        // the second turn of each: the prompt, the reply but its last token's row, and more; one of them
        // sampled otherwise, one in a slot that holds nothing of it
        var turns: [3]Case = undefined;
        var again: [3][]i32 = undefined;
        for (turns[0..2], served.tokens[0..2], first, 0..) |*turn, reply, case, at| {
            const prompt = try std.mem.concat(arena, i32, &.{ case.request.prompt, reply.items, ids[200 + 20 * at ..][0..7] });
            turn.* = .{ .request = .{ .prompt = prompt, .settings = sampled(31 + at), .eos = none, .max_tokens = 15 } };
        }
        turns[2] = .{ .request = .{ .prompt = ids[100..140], .settings = sampled(41), .eos = none, .max_tokens = 15 } };
        for (&again, turns) |*one, turn| one.* = try shared.serialReply(arena, f.loop.m, fx.vocab, turn.request.prompt, turn.request.max_tokens, turn.request.settings);
        const second = try serve(arena, &hall, &turns);
        for (again, second.tokens) |one, got| try testing.expectEqualSlices(i32, one, got.items);
        // each found its slot: the caches held the prompt and the reply but its last token
        for (second.admitted[0..2], served.admitted[0..2], first, served.tokens[0..2]) |now, before, case, reply| {
            try testing.expectEqual(before.?.slot, now.?.slot);
            try testing.expectEqual(case.request.prompt.len + reply.items.len - 1, now.?.cached);
        }
        try testing.expectEqual(@as(usize, 0), second.admitted[2].?.cached);
        // a reply that does not draft leaves the MTP layer behind: the next drafted prompt starts again
        if (drafted) {
            const plain = [_]Case{.{ .request = .{ .prompt = try std.mem.concat(arena, i32, &.{ turns[2].request.prompt, second.tokens[2].items }), .settings = greedy, .eos = none, .max_tokens = 4, .drafts = false } }};
            const third = try serve(arena, &hall, &plain);
            try testing.expectEqual(turns[2].request.prompt.len + 14, third.admitted[0].?.cached);
            const more = [_]Case{.{ .request = .{ .prompt = try std.mem.concat(arena, i32, &.{ plain[0].request.prompt, third.tokens[0].items }), .settings = greedy, .eos = none, .max_tokens = 6 } }};
            const fourth = try serve(arena, &hall, &more);
            try testing.expectEqual(@as(usize, 0), fourth.admitted[0].?.cached);
            const whole = try shared.serialReply(arena, f.loop.m, fx.vocab, more[0].request.prompt, 6, greedy);
            try testing.expectEqualSlices(i32, whole, fourth.tokens[0].items);
        }
    }
}

test "a sampled request's first token is the sampler's draw over the prompt's last row, fresh and from what a slot holds (TensorFold #892)" {
    // TensorFold's GLM read a prompt's draw from a rule row it had not armed, so the first token fell back to
    // greedy. Here each first token is held to the sampler run directly (`serialReply`), and some seeds' to a
    // token other than greedy's, so that a greedy draw cannot pass
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const capacity = 96;
    var f: Fixture = undefined;
    try f.init(0x892, 200, capacity);
    defer f.deinit();
    const ids = f.fixture.tokens;
    const none: []const u32 = &.{};
    var cases: [4]Case = undefined;
    for (&cases, 0..) |*case, i| case.* = .{ .request = .{ .prompt = ids[20 * i ..][0 .. 9 + 3 * i], .settings = sampled(51 + i), .eos = none, .max_tokens = 3 } };
    for ([_]bool{ false, true }) |drafted| {
        var hall = try f.hall(cases.len, capacity, drafted);
        defer hall.deinit();
        const served = try serve(arena, &hall, &cases);
        var turns: [cases.len]Case = undefined;
        for (&turns, cases, served.tokens, 0..) |*turn, case, reply, i| {
            const prompt = try std.mem.concat(arena, i32, &.{ case.request.prompt, reply.items, ids[100 + 10 * i ..][0..5] });
            turn.* = .{ .request = .{ .prompt = prompt, .settings = sampled(61 + i), .eos = none, .max_tokens = 2 } };
        }
        const second = try serve(arena, &hall, &turns);
        for ([_][]const Case{ &cases, &turns }, [_]Served{ served, second }) |asked, got| {
            var unlike: usize = 0;
            for (asked, got.tokens) |case, reply| {
                const drawn = try shared.serialReply(arena, f.loop.m, fx.vocab, case.request.prompt, 1, case.request.settings);
                const argmax = try shared.serialReply(arena, f.loop.m, fx.vocab, case.request.prompt, 1, greedy);
                try testing.expectEqual(drawn[0], reply.items[0]);
                unlike += @intFromBool(drawn[0] != argmax[0]);
            }
            try testing.expect(unlike > 0);
        }
        // the second turns went on from their slots' caches, where a stale rule would show
        for (second.admitted, cases, served.tokens) |now, case, reply| try testing.expectEqual(case.request.prompt.len + reply.items.len - 1, now.?.cached);
    }
}
