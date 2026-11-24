//! The pool's drafts against a session's (`hall_engine.zig`): a seat's head chains, the rule that ends them
//! and the copies offered are a lone stream's.

const std = @import("std");
const fixtures = @import("../testing.zig");
const testing = std.testing;
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const grammar = @import("../../../../src/text/grammar.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const draft = @import("../../../../src/models/qwen4/draft.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const hall_depth = @import("../../../../src/models/qwen4/pool/hall_depth.zig");
const hall_fixture = @import("hall_fixture.zig");
const session = @import("../../../../src/models/qwen4/session/session.zig");
const Engine = hall_engine.Engine;
const Ask = hall_file.Ask;
const Core = hall_file.Core;
const Out = hall_file.Out;
const engineFor = hall_fixture.engineFor;
const greedy = hall_fixture.greedy;
const Collect = hall_fixture.Collect;

/// Slot 0's reply to `ask` beside a greedy one in slot 1: its tokens, the caller's, and what the pool reports
/// of it.
fn besideAnother(engine: *Engine, ask: Ask, tokens: *std.ArrayList(i32)) !hall_file.Report {
    const gpa = engine.gpa;
    _ = try engine.begin(0, ask, 0);
    _ = try engine.begin(1, .{ .prompt = &.{ 40, 41, 42 }, .settings = greedy, .max_tokens = 9, .depth = engine.depth, .opening = engine.depth }, 0);
    var lists = [2]std.ArrayList(i32){ .empty, .empty };
    defer lists[1].deinit(gpa);
    errdefer lists[0].deinit(gpa);
    var ended = [2]bool{ false, false };
    var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended };
    const out: Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail };
    var left = false;
    var spins: usize = 0;
    while (!(ended[0] and ended[1])) : (spins += 1) {
        if (spins == 200) return error.Stuck;
        switch (engine.round(out)) {
            .ran => engine.publishAhead(out),
            .short => return error.Short,
            .failed => return error.Failed,
        }
        if (ended[1] and !left) {
            _ = engine.retire(1, .ended, lists[1].items.len);
            left = true;
        }
    }
    // what the seat's copies were found in is the prompt and the reply, a token a position
    if (engine.seats[0].?.copies) |copies| {
        const context: []const i32 = @ptrCast(copies.context.items);
        try testing.expectEqualSlices(i32, ask.prompt, context[0..ask.prompt.len]);
        try testing.expectEqualSlices(i32, lists[0].items, context[ask.prompt.len..][0..lists[0].items.len]);
    }
    tokens.* = lists[0];
    return engine.retire(0, .ended, lists[0].items.len).?;
}

test "a stream is drafted in the pool as its session drafts it: the same drafts and as many kept, whatever the floor or the costs it pays by, the sampling and the draft head's vocabulary, from the device's choice as from the row read back" {
    const mtp = @import("../../../../src/models/qwen4/mtp.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const prompt = [_]i32{ 5, 17, 200, 3, 3, 91 };
    const count = 24;
    const samplings = [_]sampling.Sampling{
        greedy,
        .{ .seed = 11, .temperature = 0.9, .top_k = 20, .top_p = 1, .min_p = 0 },
        // a nucleus is settled from the row's start where that bounds its cut, else from the whole row
        .{ .seed = 6, .temperature = 1.0, .top_k = 0, .top_p = 0.9, .min_p = 0 },
    };
    var chained: usize = 0;
    var paid: usize = 0;
    for ([_]bool{ false, true }) |own| {
        // a draft head over every other token
        const ids: ?[]const u32 = if (own) ids: {
            const half = try arena.alloc(u32, 160);
            for (half, 0..) |*id, i| id.* = @intCast(2 * i);
            break :ids half;
        } else null;
        var m = try fixtures.random_model.testModelRows(gpa, arena, ids, 48);
        defer m.deinit();
        // a random head gives a draft a few hundredths: the served floor keeps a chain's first, the others
        // let it run on. The costs are small enough that a chain's second draft pays in some rounds and not
        // in others as the stream's rate moves (`mtp.Paying`); in the last a level costs so much more than
        // a row that a chain ends at a draft a further one would have followed.
        // The last judges a chain by a fitted calibration (`Paying.Landing`), which gives such a draft a tenth.
        const Case = struct { floor: f32, costs: ?mtp.Paying.Costs = null };
        for ([_]Case{ .{ .floor = mtp.Head.default_chain }, .{ .floor = 1e-4 }, .{ .floor = 0 }, .{ .floor = 0, .costs = .{ .row = 2e-7, .level = 2e-6 } }, .{ .floor = 0, .costs = .{ .row = 2e-10, .level = 2e-5 } }, .{ .floor = 0, .costs = .{ .row = 2e-7, .level = 2e-6, .landing = .{ .slope = 0.543, .levels = .{ -0.24, -0.57, -0.67, -0.84, -0.85, -1.10 } } } }, .{ .floor = 0, .costs = .{ .row = 1e-8, .level = 1e-9, .landing = .{ .slope = 1, .levels = @splat(-6) } } } }) |case| {
            const floor = case.floor;
            m.mtp.?.costs = case.costs;
            const Want = struct { tokens: [count]i32, drafts: u64, stats: draft.Stats };
            var want: [samplings.len]Want = undefined;
            {
                const lone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = 256, .prompt_rows = 32 });
                defer lone.destroy();
                lone.head.?.chain = floor;
                for (samplings, &want) |settings, *w| {
                    var unasked: u8 = 0;
                    const reply = try lone.reply(&prompt, settings, .{ .max_tokens = count, .specials = true, .record = &w.tokens }, .{ .context = &unasked, .writeFn = Core.unasked });
                    try testing.expectEqual(count, reply.tokens);
                    w.drafts = reply.drafts;
                    w.stats = reply.stats;
                    chained += reply.stats.drafted -| reply.stats.rounds;
                    if (case.costs != null) paid += reply.stats.drafted -| reply.stats.rounds;
                }
            }
            for ([_]bool{ true, false }) |on_device| {
                var engine = try engineFor(gpa, m, 2, 32);
                defer engine.deinit();
                engine.bindPool();
                engine.rule.chain = floor;
                engine.rule.on_device = on_device;
                // one seat serves each sampling in turn: what the device is asked follows the request
                for (samplings, want) |settings, w| {
                    var got: std.ArrayList(i32) = .empty;
                    defer got.deinit(gpa);
                    const report = try besideAnother(&engine, .{ .prompt = &prompt, .settings = settings, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, &got);
                    try testing.expectEqualSlices(i32, &w.tokens, got.items);
                    try testing.expectEqual(w.drafts, report.drafts);
                    try testing.expectEqual(w.stats.drafted, report.drafted);
                    try testing.expectEqual(w.stats.accepted, report.accepted);
                }
                if (on_device) try testing.expect(engine.chosen.device > engine.chosen.host) else try testing.expectEqual(0, engine.chosen.device);
                // the head's passes ran as captured graphs, keyed by their bucket, against the session's eager head
                try testing.expect(engine.head_graphs.?.count() > 0);
                try testing.expect(engine.head_graphs.?.stream() == engine.graphs.cache.stream());
            }
        }
    }
    // chains ran past their first draft, so the floors were compared and not one draft a window
    try testing.expect(chained > 0 and paid > 0);
}

test "four seats each sampled its own way in one round are each its session's reply, their rows settled on the device in one pass or each read back (spec 144 WS-03), their chains cut to four drafts or not (WS-04)" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var vocab = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer vocab.deinit(gpa);
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 128);
    defer m.deinit();
    const count = 24;
    const prompts = [4][]const i32{ &.{ 5, 17, 200, 3 }, &.{ 9, 9, 41 }, &.{ 77, 12, 150, 8, 2 }, &.{ 33, 60 } };
    const samplings = [4]sampling.Sampling{
        greedy,
        .{ .seed = 11, .temperature = 0.9, .top_k = 20, .top_p = 1, .min_p = 0 },
        .{ .seed = 6, .temperature = 1.0, .top_k = 0, .top_p = 0.9, .min_p = 0.05 },
        // more leading tokens than a pick holds: its rows are read whole beside the others' settled
        .{ .seed = 3, .temperature = 0.7, .top_k = 60, .top_p = 0.8, .min_p = 0 },
    };
    var want: [4][count]i32 = undefined;
    {
        const lone: *session.Session = try .create(gpa, testing.io, m, &vocab, &.{}, .{ .capacity = 256, .prompt_rows = 32 });
        defer lone.destroy();
        for (prompts, samplings, &want) |prompt, settings, *w| {
            var unasked: u8 = 0;
            const reply = try lone.reply(prompt, settings, .{ .max_tokens = count, .specials = true, .record = w }, .{ .context = &unasked, .writeFn = Core.unasked });
            try testing.expectEqual(count, reply.tokens);
        }
    }
    // the third: the depth cap of two ranks (`hall_depth`) held on this one, the chains let run to their room,
    // so that it cuts them while three seats or more decode (spec 144 WS-04)
    const Way = struct { on_device: bool, cap: hall_depth.Cap = .two_ranks };
    for ([_]Way{ .{ .on_device = true }, .{ .on_device = false }, .{ .on_device = true, .cap = .always } }) |way| {
        const on_device = way.on_device;
        var engine = try engineFor(gpa, m, 4, 32);
        defer engine.deinit();
        engine.bindPool();
        engine.settle_on_device = on_device;
        engine.depth_cap = way.cap;
        if (way.cap == .always) engine.rule.chain = 0;
        var lists: [4]std.ArrayList(i32) = @splat(.empty);
        defer for (&lists) |*list| list.deinit(gpa);
        var ended: [4]bool = @splat(false);
        var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended };
        const out: Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail };
        for (prompts, samplings, 0..) |prompt, settings, i| {
            _ = try engine.begin(i, .{ .prompt = prompt, .settings = settings, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, 0);
        }
        var spins: usize = 0;
        while (!std.mem.allEqual(bool, &ended, true)) : (spins += 1) {
            if (spins == 200) return error.Stuck;
            switch (engine.round(out)) {
                .ran => engine.publishAhead(out),
                .short => return error.Short,
                .failed => return error.Failed,
            }
            var decoding: usize = 0;
            for (engine.seats) |maybe| if (maybe) |seat| if (seat.phase == .decode) {
                decoding += 1;
            };
            if (way.cap == .always and decoding >= hall_depth.from_seats) for (engine.seats) |maybe| if (maybe) |seat| if (seat.phase == .decode) {
                try testing.expect(seat.draft_n <= hall_depth.most_drafts);
            };
        }
        for (want, lists, 0..) |w, got, i| {
            try testing.expectEqualSlices(i32, &w, got.items);
            _ = engine.retire(i, .ended, count);
        }
        if (on_device) try testing.expect(engine.settled > count) else try testing.expectEqual(0, engine.settled);
        // one rank: the cap of two ranks cuts nothing; held here, it cut chains the floor of 0 let run on
        try testing.expectEqual(way.cap == .always, engine.capped > 0);
    }
}

test "a head's chain past the last extent a round reserved is the session's: a prompt that ends on an extent's last position, and one whose reply crosses it" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var m = try fixtures.random_model.testModelRows(gpa, arena, null, 48);
    defer m.deinit();
    const context = 1024;
    const count = 40;
    var long: [extents.positions]i32 = undefined;
    for (&long, 0..) |*id, i| id.* = @intCast((i * 7 + i / 13) % 300 + 3);
    var chained: usize = 0;
    // the head chains its drafts from where the trunk stands, rows the round before did not reserve: at an
    // extent's end they are positions of the next one
    for ([_]usize{ extents.positions, extents.positions - 9 }) |len| {
        const prompt = long[0..len];
        var want: [count]i32 = undefined;
        const lone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = context, .prompt_rows = 32 });
        defer lone.destroy();
        // no floor: a random head's chain runs on
        lone.head.?.chain = 0;
        var unasked: u8 = 0;
        const reply = try lone.reply(prompt, greedy, .{ .max_tokens = count, .specials = true, .record = &want }, .{ .context = &unasked, .writeFn = Core.unasked });
        try testing.expectEqual(count, reply.tokens);
        chained += reply.stats.drafted -| reply.stats.rounds;

        const places = extents.lowerBound(context, 2);
        var engine = try Engine.open(gpa, m, &.{}, .{ .slots = 2, .session = .{ .capacity = context, .prompt_rows = 32 }, .pool = places * extents.positions, .chunk = 32, .lazy_graphs = false });
        defer engine.deinit();
        engine.bindPool();
        engine.rule.chain = 0;
        var got: std.ArrayList(i32) = .empty;
        defer got.deinit(gpa);
        const report = try besideAnother(&engine, .{ .prompt = prompt, .settings = greedy, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, &got);
        try testing.expectEqualSlices(i32, &want, got.items);
        try testing.expectEqual(reply.drafts, report.drafts);
        try testing.expectEqual(reply.stats.drafted, report.drafted);
        try testing.expectEqual(reply.stats.accepted, report.accepted);
    }
    try testing.expect(chained > 0);
}

test "priced by the round its seats share, a stream's reply in the pool is its session's tokens and its drafts need not be" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const prompt = [_]i32{ 5, 17, 200, 3, 3, 91 };
    const count = 24;
    var m = try fixtures.random_model.testModelRows(gpa, arena, null, 48);
    defer m.deinit();
    // costs under which a random head's second draft pays in some rounds and not in others
    m.mtp.?.costs = .{ .row = 2e-7, .level = 2e-6 };
    const settings: sampling.Sampling = .{ .seed = 11, .temperature = 0.9, .top_k = 20, .top_p = 1, .min_p = 0 };
    var want: [count]i32 = undefined;
    {
        const lone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = 256, .prompt_rows = 32 });
        defer lone.destroy();
        lone.head.?.chain = 0;
        var unasked: u8 = 0;
        const reply = try lone.reply(&prompt, settings, .{ .max_tokens = count, .specials = true, .record = &want }, .{ .context = &unasked, .writeFn = Core.unasked });
        try testing.expectEqual(count, reply.tokens);
    }
    var drafted: [2]u64 = undefined;
    // a storage with no round of several seats, its seats priced alone, and one whose round of two seats
    // pays most of a seat's round once, its floors not a lone seat's: the tokens are the session's under both
    for ([_]f64{ 0, 0.82 }, &drafted) |shared, *rows| {
        m.mtp.?.costs.?.shared = shared;
        var engine = try engineFor(gpa, m, 2, 32);
        defer engine.deinit();
        engine.bindPool();
        engine.rule.chain = 0;
        var got: std.ArrayList(i32) = .empty;
        defer got.deinit(gpa);
        const report = try besideAnother(&engine, .{ .prompt = &prompt, .settings = settings, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, &got);
        try testing.expectEqualSlices(i32, &want, got.items);
        rows.* = report.drafted;
    }
    // the round's rate is above a seat's own while the other seat decodes: a draft a seat alone verifies is
    // not drafted there (23 rows and 22)
    try testing.expect(drafted[1] < drafted[0]);
}

test "in the pool the head runs no row for the drafts a constraint forces" {
    const json_schema = @import("../../../../src/text/json_schema.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const eos = [1]u32{tokens.addedId("<|end|>").?};
    var matcher: json_schema.Engine = try .init(gpa, &tokens, m.shape.vocab, &eos);
    defer matcher.deinit(gpa);
    var schema_diagnostic: grammar.Diagnostic = .{};
    // every token of the document is the only one its mask leaves
    const constraint = try matcher.interface().compile(gpa, .{ .json_schema = "{\"const\":[true,true,true]}" }, &schema_diagnostic);
    defer constraint.deinit();
    const mask = try gpa.alloc(u32, constraint.words);
    defer gpa.free(mask);
    const prompt = [_]i32{ 5, 17, 200, 3, 3, 91 };
    const expected = try fixtures.serial.serialConstrained(gpa, m, &prompt, 32, greedy, constraint, null, &eos);
    defer gpa.free(expected);
    try testing.expect(expected.len > 12 and expected[expected.len - 1] == eos[0]);

    const places = extents.lowerBound(256, 2);
    var engine = try Engine.open(gpa, m, &eos, .{ .slots = 2, .session = .{ .capacity = 256, .prompt_rows = 32 }, .pool = places * extents.positions, .chunk = 32, .lazy_graphs = false });
    defer engine.deinit();
    engine.bindPool();
    var got: std.ArrayList(i32) = .empty;
    defer got.deinit(gpa);
    const report = try besideAnother(&engine, .{ .prompt = &prompt, .settings = greedy, .max_tokens = 32, .depth = engine.depth, .opening = engine.depth, .constraint = constraint, .mask = mask }, &got);
    try testing.expectEqualSlices(i32, expected[0 .. expected.len - 1], got.items);
    // all but the first were drafts, and the forced ones cost the head nothing: the stream beside it has the
    // served floor, under which a random head's chain is its first draft, chosen from the row the trunk left
    try testing.expect(report.drafted == got.items.len - 1 and report.accepted == report.drafted);
    try testing.expectEqual(0, engine.chosen.rows);
    try testing.expect(engine.chosen.device + engine.chosen.host <= 9);
}

test "a stream is offered in the pool the copies its session offers it: as many found, offered and kept, the same drafts, and the reply that no copy drafts" {
    const copied = @import("../../../../src/decode/copied.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var m = try fixtures.random_model.testModelRows(gpa, arena, null, 48);
    defer m.deinit();
    // A random model repeats one token here and there and its head agrees with nothing: a match of one token
    // is a copy, offered whatever the head drafts. The model keeps few of its rows, so the lookup falls
    // silent and speaks again within the reply.
    copied.params = .{ .ngram = 1, .min_match = 1 };
    copied.agreeing = false;
    defer copied.params = .{};
    defer copied.agreeing = true;
    var prompt: [40]i32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i % 10) * 7 + 3);
    const count = 120;
    var plain: [count]i32 = undefined;
    var want: [count]i32 = undefined;
    var drafts: u64 = undefined;
    var stats: draft.Stats = undefined;
    var copies: copied.Stats = undefined;
    for ([_]bool{ false, true }) |on| {
        const lone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = 256, .prompt_rows = 32, .copies = on });
        defer lone.destroy();
        var unasked: u8 = 0;
        const reply = try lone.reply(&prompt, greedy, .{ .max_tokens = count, .specials = true, .record = if (on) &want else &plain }, .{ .context = &unasked, .writeFn = Core.unasked });
        try testing.expectEqual(count, reply.tokens);
        if (!on) continue;
        drafts = reply.drafts;
        stats = reply.stats;
        copies = lone.copies.?.stats;
    }
    try testing.expectEqualSlices(i32, &plain, &want);
    try testing.expect(copies.offered >= 8 and copies.found == copies.offered and copies.rows > copies.offered);

    for ([_]bool{ true, false }) |on| {
        var engine = try engineFor(gpa, m, 2, 32);
        defer engine.deinit();
        engine.bindPool();
        engine.copying = on;
        var got: std.ArrayList(i32) = .empty;
        defer got.deinit(gpa);
        const report = try besideAnother(&engine, .{ .prompt = &prompt, .settings = greedy, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, &got);
        try testing.expectEqualSlices(i32, &want, got.items);
        if (!on) {
            try testing.expectEqual(copied.Stats{}, report.copies);
            continue;
        }
        try testing.expectEqual(copies, report.copies);
        try testing.expectEqual(drafts, report.drafts);
        try testing.expectEqual(stats.drafted, report.drafted);
        try testing.expectEqual(stats.accepted, report.accepted);
    }
}

test "Recall: a reply drafted from the record of earlier requests is its reply without it, greedy and seeded, the record holding the same reply or another, in the session and in the pool" {
    const copied = @import("../../../../src/decode/copied.zig");
    const recall = @import("../../../../src/decode/recall.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var m = try fixtures.random_model.testModelRows(gpa, arena, null, 48);
    defer m.deinit();
    // a random model's head agrees with no copy: a recalled copy is offered whatever the head drafts
    copied.agreeing = false;
    defer copied.agreeing = true;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var stores: [2]*recall.Store = undefined;
    for (&stores, 0..) |*store, i| {
        const dir = try std.fmt.allocPrint(arena, ".zig-cache/tmp/{s}/recall{d}", .{ tmp.sub_path, i });
        var problem: []const u8 = "";
        store.* = (try recall.open(gpa, testing.io, arena, .{ .digest = @splat(9), .vocab = @intCast(m.shape.vocab), .name = "random" }, .{ .dir = dir }, null, &problem)).?;
    }
    defer for (stores) |store| store.close();
    var prompt: [40]i32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i * 37 + 11) % 200);
    const count = 120;
    const seeded: sampling.Sampling = .{ .seed = 11, .temperature = 0.9, .top_k = 20, .top_p = 1, .min_p = 0 };
    // greedy twice, then seeded twice: the second greedy reply finds the first in the record, the first seeded
    // one finds the greedy reply (another text), the second seeded one its own
    const order = [_]sampling.Sampling{ greedy, greedy, seeded, seeded };
    var plain: [order.len][count]i32 = undefined;
    var got: [order.len][count]i32 = undefined;
    var recalled: [order.len]copied.Stats = undefined;
    for ([_]?*recall.Store{ null, stores[0] }) |store| {
        const lone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = 256, .prompt_rows = 32, .recall = store });
        defer lone.destroy();
        for (order, 0..) |settings, i| {
            var unasked: u8 = 0;
            const reply = try lone.reply(&prompt, settings, .{ .max_tokens = count, .specials = true, .record = if (store == null) &plain[i] else &got[i] }, .{ .context = &unasked, .writeFn = Core.unasked });
            try testing.expectEqual(count, reply.tokens);
            if (store == null) try testing.expectEqual(0, reply.copies.recalled) else recalled[i] = reply.copies;
        }
    }
    for (plain, got) |want, have| try testing.expectEqualSlices(i32, &want, &have);
    // nothing to recall the first time; the same reply again lands its copies
    try testing.expectEqual(0, recalled[0].recalled);
    try testing.expect(recalled[1].recalled_rows > 0 and recalled[1].recalled_kept > recalled[1].recalled_rows / 2);
    try testing.expect(recalled[2].recalled_rows + recalled[3].recalled_rows > 0);

    // the pool: a seat's copies ask the record as the session's do
    var engine = try engineFor(gpa, m, 2, 32);
    defer engine.deinit();
    engine.bindPool();
    engine.recall = stores[1];
    for (0..2) |again| {
        var reply: std.ArrayList(i32) = .empty;
        defer reply.deinit(gpa);
        const report = try besideAnother(&engine, .{ .prompt = &prompt, .settings = greedy, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, &reply);
        try testing.expectEqualSlices(i32, &plain[0], reply.items);
        if (again == 0) try testing.expectEqual(0, report.copies.recalled) else try testing.expect(report.copies.recalled_kept > 0);
    }
}

/// A random stored-BF16 DFlash2 drafter for `fixtures.random_model.testModel`'s shapes, opened from memory, its
/// taps after layers 0 and 2.
const RandomDrafter = struct {
    const rig = @import("../../../../src/models/qwen4/dflash2/rig.zig");
    const safetensors = @import("../../../../src/weights/safetensors.zig");
    const drafter_testing = @import("../../../../src/models/qwen4/checks/random_drafter.zig");

    opened: rig.Opened,
    tensors: std.StringArrayHashMapUnmanaged(safetensors.Tensor) = .empty,

    fn init(self: *RandomDrafter, arena: std.mem.Allocator) !void {
        const fx = fixtures.random_model.fx;
        var c = try drafter_testing.config(arena, fx.shape.hidden, fx.shape.vocab, fx.layers, &.{ 0, 2 });
        c.block = 8;
        var source: drafter_testing.Source = .empty;
        var prng: std.Random.DefaultPrng = .init(468);
        try drafter_testing.put(arena, prng.random(), c, &source);
        var it = source.iterator();
        while (it.next()) |entry| try self.tensors.put(arena, entry.key_ptr.*, entry.value_ptr.*);
        self.opened = .{ .path = "the tests' random drafter", .config = c, .checkpoint = .{ .shards = &.{}, .tensors = self.tensors, .gpa = testing.allocator } };
    }

    /// A random drafter's chains end at once: its codebook's weight and no confidence floor make it draft a
    /// block's worth, as `dflash2/drafter_test.zig` does.
    fn force(flash: *rig.Rig(2)) void {
        for (flash.books.predecessor) |*value| value.* *= 12;
        flash.d.confidence = 0;
        flash.d.cap = flash.d.c.block - 1;
    }
};

test "DFlash2 drafts a stream in the pool as its session drafts it: the same drafts and as many kept, beside another, and two streams through the core's opening are each the serial reply (serve --drafter)" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var m = try fixtures.random_model.testModelOfRows(gpa, arena, 48);
    defer m.deinit();
    var random_drafter: RandomDrafter = .{ .opened = undefined };
    try random_drafter.init(arena);
    const opened = &random_drafter.opened;
    const count = 24;
    const samplings = [_]sampling.Sampling{ greedy, .{ .seed = 11, .temperature = 0.9, .top_k = 20, .top_p = 1, .min_p = 0 } };
    // prompts that share no prefix: each reply's drafter context starts at position 0, in the session as in the pool
    const prompts = [_][]const i32{ &.{ 5, 17, 200, 3, 3, 91 }, &.{ 6, 17, 200, 3, 3, 91, 44 } };
    const Want = struct { tokens: [count]i32, drafts: u64, stats: draft.Stats };
    var want: [samplings.len]Want = undefined;
    const lone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = 256, .prompt_rows = 32, .drafter = opened, .copies = false });
    {
        errdefer lone.destroy();
        RandomDrafter.force(lone.flash.?);
        for (samplings, prompts, &want) |settings, prompt, *w| {
            var unasked: u8 = 0;
            const reply = try lone.reply(prompt, settings, .{ .max_tokens = count, .specials = true, .record = &w.tokens }, .{ .context = &unasked, .writeFn = Core.unasked });
            try testing.expectEqual(count, reply.tokens);
            w.drafts = reply.drafts;
            w.stats = reply.stats;
            try testing.expect(reply.stats.drafted > reply.stats.rounds);
        }
    }
    // the pool takes the session's drafter as a core's opening does (`Core.migrate`)
    const flash = lone.takeFlash().?;
    lone.destroy();
    {
        var engine = try hall_fixture.engineDrafting(gpa, m, 2, 32, opened);
        defer engine.deinit();
        engine.bindPool();
        try engine.seatDrafter(flash);
        for (samplings, prompts, want) |settings, prompt, w| {
            var got: std.ArrayList(i32) = .empty;
            defer got.deinit(gpa);
            const report = try besideAnother(&engine, .{ .prompt = prompt, .settings = settings, .max_tokens = count, .depth = session.default_depth, .opening = session.default_depth }, &got);
            try testing.expectEqualSlices(i32, &w.tokens, got.items);
            try testing.expectEqual(w.drafts, report.drafts);
            try testing.expectEqual(w.stats.drafted, report.drafted);
            try testing.expectEqual(w.stats.accepted, report.accepted);
        }
    }

    // through a core: its session holds the drafter, and two streams at once open the pool, which seats it
    const core = try hall_fixture.coreDrafting(gpa, m, &tokens, &.{}, 4, 256, opened);
    defer core.destroy();
    RandomDrafter.force(core.session.?.flash.?);
    try hall_fixture.twoDrafted(gpa, core, m, prompts, samplings, count);
}

test "DFlash2 four seats keep serial tokens and state through unequal lengths, rejected prefixes, cancellation and unrelated slot reuse" {
    const check = @import("../../../../src/models/qwen4/checks/ranks_hall.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var m = try fixtures.random_model.testModelOfRows(gpa, arena, 64);
    defer m.deinit();
    var random_drafter: RandomDrafter = .{ .opened = undefined };
    try random_drafter.init(arena);
    const lone = try session.Session.create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = 256, .prompt_rows = 32, .drafter = &random_drafter.opened, .copies = false });
    RandomDrafter.force(lone.flash.?);
    const flash = lone.takeFlash().?;
    lone.destroy();
    var engine = try hall_fixture.engineDrafting(gpa, m, 4, 32, &random_drafter.opened);
    defer engine.deinit();
    engine.bindPool();
    try engine.seatDrafter(flash);
    const prompts = [4][]const i32{ &.{ 5, 17, 200, 3, 3, 91 }, &.{ 6, 11, 30 }, &.{ 7, 9, 90, 14 }, &.{ 8, 17, 100, 1, 2, 3, 4 } };
    const lengths = [4]usize{ 9, 8, 7, 5 };
    var serial: [4]check.Left = @splat(.{});
    for (0..4) |stream| {
        try check.servePlan(&engine, prompts, &.{stream}, &serial, .{ .lengths = lengths, .depth = 0 });
        const expected = try fixtures.serial.serialReply(gpa, m, prompts[stream], lengths[stream], greedy);
        defer gpa.free(expected);
        try testing.expectEqualSlices(i32, expected, serial[stream].tokens[0..lengths[stream]]);
    }
    // A proposal that its client cancels before verification must not enter a later sequence's context.
    _ = try engine.begin(0, .{ .prompt = prompts[3], .settings = greedy, .max_tokens = 9, .depth = 7, .opening = 7 }, 0);
    var lists: [4]std.ArrayList(i32) = @splat(.empty);
    defer for (&lists) |*list| list.deinit(gpa);
    var ended: [4]bool = @splat(false);
    var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended };
    const out: Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail };
    try testing.expectEqual(.ran, engine.round(out));
    try testing.expect(engine.seats[0].?.draft_n > 0);
    _ = engine.retire(0, .cancelled, 1);
    var drafted: u64 = 0;
    var rejected: u64 = 0;
    for ([_]bool{ false, true }) |spoil| {
        var together: [4]check.Left = @splat(.{});
        try check.servePlan(&engine, prompts, &.{ 0, 1, 2, 3 }, &together, .{ .lengths = lengths, .reject = if (spoil) &serial else null });
        for (together, serial, 0..) |got, want, stream| {
            try testing.expectEqualSlices(i32, want.tokens[0..lengths[stream]], got.tokens[0..lengths[stream]]);
            try testing.expectEqual(want.state, got.state);
            drafted += got.drafted;
            if (spoil) rejected += got.drafted - got.accepted;
        }
    }
    try testing.expect(drafted > 0 and rejected > 0 and flash.d.passes > 0);
}

test "a seat the round grants no draft row keeps its context and drafts again where it is granted one: its reply the session's, no head pass for it (spec 152 WS-08, TensorFold #1779)" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var m = try fixtures.random_model.testModelRows(gpa, arena, null, 48);
    defer m.deinit();
    const prompt = [_]i32{ 5, 17, 200, 3, 3, 91 };
    const count = 24;
    var want: [count]i32 = undefined;
    {
        const lone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = 256, .prompt_rows = 32 });
        defer lone.destroy();
        var unasked: u8 = 0;
        const reply = try lone.reply(&prompt, greedy, .{ .max_tokens = count, .specials = true, .record = &want }, .{ .context = &unasked, .writeFn = Core.unasked });
        try testing.expectEqual(count, reply.tokens);
    }

    var engine = try engineFor(gpa, m, 2, 32);
    defer engine.deinit();
    engine.bindPool();
    var lists = [2]std.ArrayList(i32){ .empty, .empty };
    defer for (&lists) |*list| list.deinit(gpa);
    var ended = [2]bool{ false, false };
    var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended };
    const out: Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail };
    // alone in the pool at depth 0: every round grants the seat no row, so the head chases nothing for it,
    // its kept rows absorbed as another seat's are (`spans`), and its reply is the session's to the token
    _ = try engine.begin(0, .{ .prompt = &prompt, .settings = greedy, .max_tokens = count, .depth = 0, .opening = 0 }, 0);
    var spins: usize = 0;
    while (!ended[0]) : (spins += 1) {
        if (spins == 200) return error.Stuck;
        switch (engine.round(out)) {
            .ran => engine.publishAhead(out),
            .short => return error.Short,
            .failed => return error.Failed,
        }
    }
    const report = engine.retire(0, .ended, lists[0].items.len).?;
    try testing.expectEqualSlices(i32, &want, lists[0].items);
    // never a chain begun, never a row run or a choice made for it: nothing drafted or kept of a draft
    try testing.expectEqual(@as(usize, 0), report.drafted);
    try testing.expectEqual(@as(usize, 0), report.accepted);
    try testing.expectEqual(@as(usize, 0), engine.chosen.rows);
    try testing.expectEqual(@as(usize, 0), engine.chosen.passes);
    try testing.expectEqual(@as(usize, 0), engine.chosen.device + engine.chosen.host);

    // granted depth again, the slot drafts on the state the skipped rounds kept absorbing
    const prompt2 = [_]i32{ 40, 41, 42 };
    const count2 = 9;
    var want2: [count2]i32 = undefined;
    {
        const lone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = 256, .prompt_rows = 32 });
        defer lone.destroy();
        var unasked: u8 = 0;
        const reply = try lone.reply(&prompt2, greedy, .{ .max_tokens = count2, .specials = true, .record = &want2 }, .{ .context = &unasked, .writeFn = Core.unasked });
        try testing.expectEqual(count2, reply.tokens);
    }
    lists[0].clearRetainingCapacity();
    ended[0] = false;
    _ = try engine.begin(0, .{ .prompt = &prompt2, .settings = greedy, .max_tokens = count2, .depth = engine.depth, .opening = engine.depth }, 0);
    spins = 0;
    while (!ended[0]) : (spins += 1) {
        if (spins == 200) return error.Stuck;
        switch (engine.round(out)) {
            .ran => engine.publishAhead(out),
            .short => return error.Short,
            .failed => return error.Failed,
        }
    }
    // its reply the session's to the token, and drafts proposed for it again: a chain of its own was begun
    // each round, where the rounds at depth 0 began none and the seat's reply came undrafted
    const granted = engine.retire(0, .ended, lists[0].items.len).?;
    try testing.expectEqualSlices(i32, &want2, lists[0].items);
    try testing.expect(granted.drafted > 0);
}
