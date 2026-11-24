//! `session.zig`'s tests of a reply: the serial one whatever the session decoded before, from a kept state,
//! in wide windows, with the scratch poisoned; the opening window; the graphs captured before the first reply
//! and between turns.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const Counting = fixtures.sessions.Counting;
const deepest = fixtures.sessions.deepest;
const cuda = @import("../../../../src/cuda/cuda.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const request_file = @import("../../../../src/server/request.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const draft = @import("../../../../src/models/qwen4/draft.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const mtp = @import("../../../../src/models/qwen4/mtp.zig");
const session_file = @import("../../../../src/models/qwen4/session/session.zig");
const Options = session_file.Options;
const ReplyOptions = session_file.ReplyOptions;
const Session = session_file.Session;
const default_depth = session_file.default_depth;
const default_opening_depth = session_file.default_opening_depth;
const openingFor = session_file.openingFor;

test "a reply drafts as deep as a window takes unless told, served or not" {
    // `serve` once took its depth from the struct while `chat` asked for another
    try testing.expectEqual(15, default_depth);
    try testing.expectEqual(default_depth, (ReplyOptions{ .max_tokens = 0 }).depth);
    try testing.expect(default_depth < draft.forward_max_rows);
    // the head's rule beside it: the chain's probability, not TensorFold's one draft's
    try testing.expectEqual(0.65, mtp.Head.default_chain);
}

test "a session's replies are the serial ones, whatever it decoded before" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), null);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = m.max_rows, .capacity = 256 });
    defer session.destroy();

    const count = 24;
    const prompts = [_][]const i32{ &.{ 5, 17, 200, 3, 3, 91 }, &.{ 7, 7, 250, 1 }, &.{ 5, 17, 200, 3, 3, 91 } };
    const samplings = [_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 11, .temperature = 0.9 }, .{ .seed = 0, .temperature = 0 } };
    for (prompts, samplings) |prompt, settings| {
        const expected = try fixtures.serial.serialReply(gpa, m, prompt, count, settings);
        defer gpa.free(expected);
        var record: [count]i32 = undefined;
        var counting: Counting = .{};
        const reply = try session.reply(prompt, settings, .{ .max_tokens = count, .depth = deepest, .record = &record }, counting.sink());
        try testing.expectEqual(request_file.Finish.length, reply.finish);
        try testing.expectEqual(count, reply.tokens);
        try testing.expectEqualSlices(i32, expected, &record);
    }

    // a sink that asks for the end gets it, and the next reply is whole again
    var counting: Counting = .{ .limit = 1 };
    var record: [count]i32 = undefined;
    const cut = try session.reply(prompts[0], samplings[0], .{ .max_tokens = count, .depth = deepest, .specials = true, .record = &record }, counting.sink());
    if (counting.pieces > 0) {
        try testing.expectEqual(request_file.Finish.cut, cut.finish);
        try testing.expect(cut.tokens < count);
    }
}

test "a prompt run from a kept state gives the reply an empty state gives" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), null);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = m.max_rows, .capacity = 256, .snapshots = 2, .stride = 0 });
    defer session.destroy();
    // one that starts every prompt from an empty state: the head's drafts must be the same too
    const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = m.max_rows, .capacity = 256, .snapshots = 0 });
    defer fresh.destroy();

    const count = 12;
    var random: std.Random.DefaultPrng = .init(8);
    var first: [40]i32 = undefined;
    for (&first) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var record: [count]i32 = undefined;
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const sampled: sampling.Sampling = .{ .seed = 5, .temperature = 0.8 };

    // What each turn's prompt is made of, and the tokens it must find held.
    var prompt: std.ArrayList(i32) = .empty;
    defer prompt.deinit(gpa);
    const Turn = struct { keep: usize, reply: bool, add: []const i32, settings: sampling.Sampling, cached: usize };
    const turns = [_]Turn{
        // nothing held
        .{ .keep = 0, .reply = false, .add = &first, .settings = greedy, .cached = 0 },
        // the prompt, the whole reply and more: the state as the reply left it
        .{ .keep = 40, .reply = true, .add = &.{ 9, 8, 7, 6, 5 }, .settings = sampled, .cached = 40 + count - 1 },
        // the same prompt again: the snapshot before its end
        .{ .keep = 40 + count + 5, .reply = false, .add = &.{}, .settings = greedy, .cached = 40 + count + 5 - 2 },
        // parting inside the first prompt's tail, as a template that rewrites the last turn does
        .{ .keep = 39, .reply = false, .add = &.{ 77, 78, 79 }, .settings = sampled, .cached = 38 },
        // parting early: below every snapshot
        .{ .keep = 20, .reply = false, .add = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .settings = greedy, .cached = 0 },
        // and that prompt's own tail is held for the one after
        .{ .keep = 28, .reply = false, .add = &.{ 100, 101 }, .settings = greedy, .cached = 26 },
        // parting at a snapshot's own position: the head absorbed the token there, so the one below serves
        .{ .keep = 28, .reply = false, .add = &.{ 102, 101, 7 }, .settings = sampled, .cached = 19 },
    };
    for (turns) |turn| {
        prompt.shrinkRetainingCapacity(turn.keep);
        if (turn.reply) try prompt.appendSlice(gpa, &record);
        try prompt.appendSlice(gpa, turn.add);
        const expected = try fixtures.serial.serialReply(gpa, m, prompt.items, count, turn.settings);
        defer gpa.free(expected);
        const reply = try session.reply(prompt.items, turn.settings, .{ .max_tokens = count, .depth = deepest, .record = &record }, counting.sink());
        try testing.expectEqual(turn.cached, reply.cached);
        try testing.expectEqualSlices(i32, expected, &record);
        fresh.trusted = false;
        var again: [count]i32 = undefined;
        const whole = try fresh.reply(prompt.items, turn.settings, .{ .max_tokens = count, .depth = deepest, .record = &again }, counting.sink());
        try testing.expectEqual(0, whole.cached);
        try testing.expectEqual(whole.drafts, reply.drafts);
        try testing.expectEqual(whole.stats.accepted, reply.stats.accepted);
        try testing.expect(session.snapshots.items.len <= 2);
    }

    // snapshots along a prompt: one that parts early in it starts from the last below
    const strided: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = m.max_rows, .capacity = 256, .snapshots = 3, .stride = 16 });
    defer strided.destroy();
    _ = try strided.reply(&first, greedy, .{ .max_tokens = count, .depth = deepest, .record = &record }, counting.sink());
    try testing.expectEqual(3, strided.snapshots.items.len);
    try testing.expect(strided.snapshotBytes() > 0);
    const edited = first[0..25] ++ [_]i32{ 60, 61, 62 };
    const expected = try fixtures.serial.serialReply(gpa, m, edited, count, sampled);
    defer gpa.free(expected);
    const part = try strided.reply(edited, sampled, .{ .max_tokens = count, .depth = deepest, .record = &record }, counting.sink());
    try testing.expectEqual(16, part.cached);
    try testing.expectEqualSlices(i32, expected, &record);

    // a reply without drafts leaves the head behind, and the next prompt starts over
    _ = try session.reply(prompt.items, greedy, .{ .max_tokens = count, .depth = 0, .record = &record }, counting.sink());
    const after = try session.reply(prompt.items, greedy, .{ .max_tokens = count, .depth = deepest, .record = &record }, counting.sink());
    try testing.expectEqual(0, after.cached);
}

test "a prompt run in wide windows gives the reply and the drafts one-row steps give" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // the same weights twice: windows of up to 96 rows, and of 4
    var wide = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 96);
    defer wide.deinit();
    var narrow = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), null);
    defer narrow.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const count = 12;
    var random: std.Random.DefaultPrng = .init(21);
    var prompt: [210]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var counting: Counting = .{};
    const settings: sampling.Sampling = .{ .seed = 3, .temperature = 0.7 };

    const expected = try fixtures.serial.serialReply(gpa, narrow, &prompt, count, settings);
    defer gpa.free(expected);
    const stepwise: *Session = try .create(gpa, testing.io, narrow, &tokens, &.{}, .{ .window = narrow.max_rows, .capacity = 256, .snapshots = 0 });
    defer stepwise.destroy();
    var record: [count]i32 = undefined;
    const narrow_reply = try stepwise.reply(&prompt, settings, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
    try testing.expectEqualSlices(i32, expected, &record);

    // whole windows of 96 rows, and with snapshots where a window crossed a multiple of 40 positions
    for ([_]usize{ 0, 40 }) |stride| {
        const session: *Session = try .create(gpa, testing.io, wide, &tokens, &.{}, .{ .window = 4, .capacity = 256, .stride = stride });
        defer session.destroy();
        try testing.expectEqual(96, session.buffers.rows);
        const reply = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
        try testing.expectEqualSlices(i32, expected, &record);
        try testing.expectEqual(narrow_reply.drafts, reply.drafts);
        // and a later turn from the wide windows' snapshot
        var next: [230]i32 = undefined;
        @memcpy(next[0..209], prompt[0..209]);
        for (next[209..], 0..) |*token, i| token.* = @intCast(30 + i);
        const after = try fixtures.serial.serialReply(gpa, narrow, &next, count, settings);
        defer gpa.free(after);
        const turn = try session.reply(&next, settings, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
        try testing.expectEqual(208, turn.cached);
        try testing.expectEqualSlices(i32, after, &record);
    }
}

test "with the linears' scratch poisoned between their calls, a reply and its drafts are what they were" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 96);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const count = 24;
    var random: std.Random.DefaultPrng = .init(22);
    var prompt: [210]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var counting: Counting = .{};

    // A session captures its graphs as it goes, so each side has its own: the poisoned one's hold the fills.
    // The prompt runs in windows of 96 rows, the reply in verify windows the head drafts for.
    defer cuda.Scratch.poisoned = false;
    for ([_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 5, .temperature = 0.8 } }) |settings| {
        var records: [2][count]i32 = undefined;
        var drafts: [2]usize = undefined;
        for ([_]bool{ false, true }, &records, &drafts) |poisoned, *record, *drafted| {
            cuda.Scratch.poisoned = poisoned;
            const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256 });
            defer session.destroy();
            const reply = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = 3, .record = record }, counting.sink());
            try testing.expectEqual(count, reply.tokens);
            drafted.* = reply.drafts;
            // the head's layer computes in the main model's attention scratch, and frees none of it
            try testing.expect(session.head.?.b.layer.attention_lent and !session.buffers.attention_lent);
            try testing.expectEqual(session.buffers.attention.?.pa.view.ptr, session.head.?.b.layer.attention.?.pa.view.ptr);
        }
        try testing.expectEqualSlices(i32, &records[0], &records[1]);
        try testing.expectEqual(drafts[0], drafts[1]);
        try testing.expect(drafts[0] > 0);
    }
}

/// What a reply reads or writes of a session, as bytes: both sides of every DeltaNet state and which is the
/// committed one, the conv inputs, the PLE tails, the main model's streams, read-out and logits, the head's
/// streams and logits, and both positions.
fn stateBytes(gpa: Allocator, session: *const Session) ![]u8 {
    try cuda.synchronize();
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    const state = session.state;
    for (state.recurrent, state.current, state.conv) |pair, current, conv| {
        for ([_]cuda.Buffer{ pair[0], pair[1], conv }) |buffer| try buffer.download(try out.addManyAsSlice(gpa, buffer.bytes()));
        try out.append(gpa, current);
    }
    for (state.ple) |branch| try branch.tail.download(try out.addManyAsSlice(gpa, branch.tail.bytes()));
    const b = session.buffers;
    for ([_]cuda.Buffer{ b.h, b.mixed, b.logits, b.ids }) |buffer| try buffer.download(try out.addManyAsSlice(gpa, buffer.bytes()));
    try out.appendSlice(gpa, std.mem.asBytes(&state.pos));
    try out.appendSlice(gpa, std.mem.asBytes(&state.position.host));
    if (session.head) |head| {
        for ([_]cuda.Buffer{ head.b.layer.h, head.b.layer.mixed, head.b.layer.logits, head.b.layer.ids }) |buffer| try buffer.download(try out.addManyAsSlice(gpa, buffer.bytes()));
        try out.appendSlice(gpa, std.mem.asBytes(&head.state.position.host));
    }
    return out.toOwnedSlice(gpa);
}

test "a prompt's windows and a reply's rounds whose table rows are gathered ahead give the reply, the drafts and the state of ones gathered window by window" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // the model with a PLE, in windows of 40 rows
    var m = try fixtures.tiny_model.tinyModel(gpa, arena.allocator());
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const count = 12;
    var random: std.Random.DefaultPrng = .init(87);
    var prompt: [213]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var counting: Counting = .{};
    const settings: sampling.Sampling = .{ .seed = 3, .temperature = 0.7 };
    defer forward.gather_ahead = true;

    var record: [2][count]i32 = undefined;
    var after: [2][count]i32 = undefined;
    var drafts: [2]usize = undefined;
    var states: [2][]u8 = .{ &.{}, &.{} };
    defer for (states) |bytes| gpa.free(bytes);
    // with states kept inside the windows and with none
    for ([_]usize{ 0, 4 }) |snapshots| {
        for ([_]bool{ false, true }, 0..) |ahead, side| {
            forward.gather_ahead = ahead;
            const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 512, .snapshots = snapshots });
            defer session.destroy();
            const reply = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = 3, .record = &record[side] }, counting.sink());
            drafts[side] = reply.drafts;
            gpa.free(states[side]);
            states[side] = &.{};
            states[side] = try stateBytes(gpa, session);
            // every window after the prompt's first had its rows before its turn, or none had
            var windows: usize = 0;
            var pos: usize = 0;
            while (pos < prompt.len) : (windows += 1) {
                const left = prompt.len - pos;
                const last = @min(session.buffers.rows, session.chosen.len);
                pos += if (left <= last) left else @min(session.buffers.rows, left - last);
            }
            try testing.expect(windows >= 5 and session.buffers.ahead.len == 1);
            // every window after the prompt's first, and every round of the reply (its kept token at the least)
            try testing.expect(reply.stats.rounds >= 2);
            try testing.expectEqual(if (ahead) windows - 1 + reply.stats.rounds else 0, session.buffers.ahead[0].served);
            // and a later turn, from the state those windows left
            var next: [260]i32 = undefined;
            @memcpy(next[0..prompt.len], &prompt);
            for (next[prompt.len..], 0..) |*token, i| token.* = @intCast(30 + i);
            _ = try session.reply(&next, settings, .{ .max_tokens = count, .depth = 3, .record = &after[side] }, counting.sink());
        }
        try testing.expectEqualSlices(i32, &record[0], &record[1]);
        try testing.expectEqual(drafts[0], drafts[1]);
        try testing.expectEqualSlices(u8, states[0], states[1]);
        try testing.expectEqualSlices(i32, &after[0], &after[1]);
    }
}

test "a reply offered tools opens with a window of six drafts at most, every later window and every other reply at the depth" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    // which requests are capped: tools offered and the call free
    try testing.expectEqual(default_opening_depth, openingFor(true, false, default_depth));
    try testing.expectEqual(3, openingFor(true, false, 3));
    try testing.expectEqual(0, openingFor(true, false, 0));
    try testing.expectEqual(default_depth, openingFor(false, false, default_depth));
    try testing.expectEqual(default_depth, openingFor(true, true, default_depth));
    try testing.expectEqual(default_depth, openingFor(false, true, default_depth));
    try testing.expectEqual(default_depth, (ReplyOptions{ .max_tokens = 0 }).opening);
    try testing.expect(default_opening_depth < default_depth);

    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 96);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const count = 40;
    var counting: Counting = .{};
    var random: std.Random.DefaultPrng = .init(41);
    var prompt: [30]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);

    const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = draft.forward_max_rows, .capacity = 256 });
    defer session.destroy();
    // a random model's chains end at their first draft under the default rule; with no floor they go to the
    // depth, so a window's drafts are what it was allowed
    session.head.?.confidence = 0;
    session.head.?.chain = 0;
    try session.prepare(deepest, 1);
    const held = session.captured().count;
    for ([_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 5, .temperature = 0.8 } }) |settings| {
        var serial: [count]i32 = undefined;
        _ = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = 0, .record = &serial }, counting.sink());
        var record: [count]i32 = undefined;
        const whole = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = deepest, .record = &record }, counting.sink());
        try testing.expectEqualSlices(i32, &serial, &record);
        try testing.expectEqual(deepest, whole.opening);
        const capped = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = deepest, .opening = default_opening_depth, .record = &record }, counting.sink());
        try testing.expectEqualSlices(i32, &serial, &record);
        try testing.expectEqual(default_opening_depth, capped.opening);
        const shallow = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = default_opening_depth, .record = &record }, counting.sink());
        try testing.expectEqualSlices(i32, &serial, &record);
        try testing.expectEqual(default_opening_depth, shallow.opening);
        // The windows after the first are the depth's: more drafts than a reply held to six throughout, and
        // the first window apart, the reply drafted at the depth. (A random model keeps next to no draft, so
        // the three replies are the same number of windows or nearly.)
        try testing.expect(capped.stats.drafted > shallow.stats.drafted + (capped.stats.rounds - 1) * (deepest - default_opening_depth) / 2);
        try testing.expect(capped.stats.drafted < whole.stats.drafted or capped.stats.rounds > whole.stats.rounds);
        // an opening above the depth asks for no more than the depth
        const over = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = 3, .opening = default_opening_depth, .record = &record }, counting.sink());
        try testing.expectEqualSlices(i32, &serial, &record);
        try testing.expectEqual(3, over.opening);
    }
    // a window of seven rows was captured with the rest: the cap adds no graph
    try testing.expectEqual(held, session.captured().count);
}

test "graphs captured before the first reply: no reply captures another, and its tokens and drafts are a session's that captures as it goes" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // prompts in windows of up to 96 rows, replies in windows of up to 16
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 96);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const window = draft.forward_max_rows;
    const options: Options = .{ .window = window, .capacity = 256, .stride = 40 };
    const count = 20;
    var counting: Counting = .{};
    var random: std.Random.DefaultPrng = .init(33);
    var first: [70]i32 = undefined;
    for (&first) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const sampled: sampling.Sampling = .{ .seed = 9, .temperature = 0.8 };

    // The head's graphs do not follow the depth; the step's are one for each window size and state side.
    const shallow: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, options);
    defer shallow.destroy();
    try shallow.prepare(3, 1);
    try testing.expectEqual(2 * 4 + 3 * window, shallow.captured().count);
    try testing.expectEqual(shallow.graphs.?.cache.capturing() + shallow.head.?.graphs.?.cache.capturing(), shallow.captured().nanoseconds);
    try testing.expect(shallow.captured().nanoseconds > 0);
    try shallow.prepare(3, 1);
    try testing.expectEqual(2 * 4 + 3 * window, shallow.captured().count);
    // a session that launches each kernel holds no graph
    const eager: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = window, .capacity = 256, .eager = true });
    defer eager.destroy();
    try eager.prepare(window - 1, 1);
    try testing.expectEqual(0, eager.captured().count);
    // and one without a head decodes one row a step: a graph for each state side, whatever depth is named
    var plain = try fixtures.random_model.testModel(gpa, arena.allocator());
    defer plain.deinit();
    const serial: *Session = try .create(gpa, testing.io, plain, &tokens, &.{}, .{ .window = plain.max_rows, .capacity = 256 });
    defer serial.destroy();
    try serial.prepare(window - 1, 1);
    try testing.expectEqual(2, serial.captured().count);
    var alone: [count]i32 = undefined;
    _ = try serial.reply(&first, greedy, .{ .max_tokens = count, .depth = deepest, .record = &alone }, counting.sink());
    try testing.expectEqual(2, serial.captured().count);

    // `confidence` 0.7 ends a random model's chains at their first draft; 0 chains to the depth, so windows
    // of every size are met
    for ([_]f32{ 0.7, 0 }) |confidence| {
        const prepared: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, options);
        defer prepared.destroy();
        const lazy: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, options);
        defer lazy.destroy();
        prepared.head.?.confidence = confidence;
        lazy.head.?.confidence = confidence;
        prepared.head.?.chain = 0;
        lazy.head.?.chain = 0;
        const fresh = try stateBytes(gpa, prepared);
        defer gpa.free(fresh);
        try prepared.prepare(window - 1, 1);
        const held = prepared.captured().count;
        try testing.expectEqual(2 * window + 3 * window, held);
        // nothing was launched: every buffer is as `create` left it
        const after = try stateBytes(gpa, prepared);
        defer gpa.free(after);
        try testing.expectEqualSlices(u8, fresh, after);

        // a prompt of several windows, the next turn from the state it left, a short prompt from an empty one
        var prompt: std.ArrayList(i32) = .empty;
        defer prompt.deinit(gpa);
        var record: [count]i32 = undefined;
        var again: [count]i32 = undefined;
        const Turn = struct { keep: usize, reply: bool, add: []const i32, settings: sampling.Sampling, depth: usize };
        for ([_]Turn{
            .{ .keep = 0, .reply = false, .add = &first, .settings = greedy, .depth = window - 1 },
            .{ .keep = 70, .reply = true, .add = &.{ 9, 8, 7 }, .settings = sampled, .depth = 6 },
            .{ .keep = 0, .reply = false, .add = &.{ 5, 17, 200 }, .settings = sampled, .depth = window - 1 },
            .{ .keep = 3, .reply = true, .add = first[0..37], .settings = greedy, .depth = 1 },
        }) |turn| {
            prompt.shrinkRetainingCapacity(turn.keep);
            if (turn.reply) try prompt.appendSlice(gpa, &record);
            try prompt.appendSlice(gpa, turn.add);
            const one = try prepared.reply(prompt.items, turn.settings, .{ .max_tokens = count, .depth = turn.depth, .record = &record }, counting.sink());
            const other = try lazy.reply(prompt.items, turn.settings, .{ .max_tokens = count, .depth = turn.depth, .record = &again }, counting.sink());
            try testing.expectEqualSlices(i32, &again, &record);
            try testing.expectEqual(other.drafts, one.drafts);
            try testing.expectEqual(other.stats, one.stats);
            try testing.expectEqual(other.cached, one.cached);
            try testing.expectEqual(held, prepared.captured().count);
        }
        // the replies did run on graphs, and on fewer than were prepared
        try testing.expect(lazy.captured().count > 0 and lazy.captured().count < held);
        if (confidence == 0) try testing.expect(lazy.graphs.?.cache.count() > 2 * 7);
    }
}

test "a prompt's short last window launched as the graph held for its rows gives the reply and the drafts of one launched kernel by kernel" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 96);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const window = draft.forward_max_rows;
    const count = 9;
    var counting: Counting = .{};
    var random: std.Random.DefaultPrng = .init(41);
    var first: [70]i32 = undefined;
    for (&first) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const sampled: sampling.Sampling = .{ .seed = 9, .temperature = 0.8 };

    {
        const prepared: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = window, .capacity = 256, .stride = 0 });
        defer prepared.destroy();
        try prepared.prepare(window - 1, 1);
        const held = prepared.captured().count;
        const eager: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = window, .capacity = 256, .stride = 0, .eager = true });
        defer eager.destroy();

        var prompt: std.ArrayList(i32) = .empty;
        defer prompt.deinit(gpa);
        var record: [count]i32 = undefined;
        var again: [count]i32 = undefined;
        // `rows`: the rows the prompt runs, from the state or the snapshot it goes on from
        const Turn = struct { keep: usize, reply: bool, add: []const i32, settings: sampling.Sampling, rows: usize };
        for ([_]Turn{
            // first met: one wide window, launched kernel by kernel
            .{ .keep = 0, .reply = false, .add = &first, .settings = greedy, .rows = 70 },
            // sent again: the two rows after the mark before its end, on the graph of two rows
            .{ .keep = 70, .reply = false, .add = &.{}, .settings = sampled, .rows = 2 },
            // and again: one row, from the mark the second took where it parted (no message's start is near)
            .{ .keep = 70, .reply = false, .add = &.{}, .settings = greedy, .rows = 1 },
            // a short turn after the reply, from the state it left
            .{ .keep = 70, .reply = true, .add = &.{ 9, 8, 7, 6, 5 }, .settings = sampled, .rows = 6 },
            // a last window of as many rows as a reply's widest
            .{ .keep = 70 + count + 4, .reply = false, .add = first[0..15], .settings = greedy, .rows = 16 },
            // one row more is no reply's window: launched kernel by kernel
            .{ .keep = 70 + count + 4, .reply = false, .add = first[20..36], .settings = greedy, .rows = 17 },
            // a short prompt from an empty state
            .{ .keep = 0, .reply = false, .add = &.{ 5, 17, 200 }, .settings = sampled, .rows = 3 },
        }) |turn| {
            prompt.shrinkRetainingCapacity(turn.keep);
            if (turn.reply) try prompt.appendSlice(gpa, &record);
            try prompt.appendSlice(gpa, turn.add);
            const expected = try fixtures.serial.serialReply(gpa, m, prompt.items, count, turn.settings);
            defer gpa.free(expected);
            const one = try prepared.reply(prompt.items, turn.settings, .{ .max_tokens = count, .depth = deepest, .record = &record }, counting.sink());
            const other = try eager.reply(prompt.items, turn.settings, .{ .max_tokens = count, .depth = deepest, .record = &again }, counting.sink());
            try testing.expectEqualSlices(i32, expected, &record);
            try testing.expectEqualSlices(i32, &again, &record);
            try testing.expectEqual(other.drafts, one.drafts);
            try testing.expectEqual(other.stats, one.stats);
            try testing.expectEqual(other.cached, one.cached);
            try testing.expectEqual(prompt.items.len - turn.rows, one.cached);
            try testing.expectEqual(1, one.windows);
            try testing.expectEqual(turn.rows <= window, one.graphed);
            try testing.expect(!other.graphed);
            // a prompt captures nothing
            try testing.expectEqual(held, prepared.captured().count);
        }
    }

    // A session whose graphs are captured as its windows meet them launches a prompt's tail kernel by kernel
    // until a reply's window of those rows has been captured, and captures none for it.
    const lazy: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = window, .capacity = 256, .stride = 0 });
    defer lazy.destroy();
    var record: [count]i32 = undefined;
    const met = try lazy.reply(first[0..5], greedy, .{ .max_tokens = count, .depth = 0, .record = &record }, counting.sink());
    try testing.expect(!met.graphed);
    // the reply's one-row steps captured theirs, a side each
    try testing.expectEqual(2, lazy.graphs.?.cache.count());
    // the same prompt again, its five rows from an empty state (a reply without drafts leaves the head
    // behind): a window of five rows was never captured, and is not now
    const resent = try lazy.reply(first[0..5], greedy, .{ .max_tokens = count, .depth = 0, .record = &record }, counting.sink());
    try testing.expect(!resent.graphed);
    try testing.expectEqual(2, lazy.graphs.?.cache.count());
}

test "graphs captured between two turns leave every buffer and the state's side as they were, and the next reply is the same" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 96);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const window = draft.forward_max_rows;
    // two buckets of context: 8192, then the capacity
    const options: Options = .{ .window = window, .capacity = 9000, .stride = 0 };
    const count = 9;
    var counting: Counting = .{};
    const settings: sampling.Sampling = .{ .seed = 4, .temperature = 0.6 };

    const lazy: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, options);
    defer lazy.destroy();
    try testing.expectEqual(8192, forward.Graphs.bucketAfter(lazy.buffers, null).?);
    try testing.expectEqual(9000, forward.Graphs.bucketAfter(lazy.buffers, 8192).?);
    try testing.expectEqual(null, forward.Graphs.bucketAfter(lazy.buffers, 9000));

    // Each turn's prompt is the one before it, its reply and one token more, so every turn but the first goes
    // on from the state as it stands. The graphs are captured before turn `at`; the committed DeltaNet state
    // must be met on either side there.
    const turns = 4;
    var sides: [2]bool = .{ false, false };
    var expected: [turns][count]i32 = undefined;
    var drafts: [turns]u64 = undefined;
    for (0..turns + 1) |at| {
        const session: *Session = if (at == 0) lazy else try .create(gpa, testing.io, m, &tokens, &.{}, options);
        defer if (at != 0) session.destroy();
        var prompt: std.ArrayList(i32) = .empty;
        defer prompt.deinit(gpa);
        try prompt.appendSlice(gpa, &.{ 5, 17, 200, 3, 3, 91, 40 });
        for (0..turns) |turn| {
            if (at == turn + 1) {
                const before = try stateBytes(gpa, session);
                defer gpa.free(before);
                sides[session.state.current[0]] = true;
                try session.prepare(window - 1, 2);
                try testing.expectEqual(2 * (2 * window + 3 * window), session.captured().count);
                const after = try stateBytes(gpa, session);
                defer gpa.free(after);
                try testing.expectEqualSlices(u8, before, after);
            }
            var record: [count]i32 = undefined;
            const reply = try session.reply(prompt.items, settings, .{ .max_tokens = count, .depth = deepest, .record = &record }, counting.sink());
            try testing.expectEqual(if (turn == 0) 0 else prompt.items.len - 2, reply.cached);
            if (at == 0) {
                expected[turn] = record;
                drafts[turn] = reply.drafts;
            } else {
                try testing.expectEqualSlices(i32, &expected[turn], &record);
                try testing.expectEqual(drafts[turn], reply.drafts);
            }
            try prompt.appendSlice(gpa, &record);
            try prompt.append(gpa, @intCast(60 + turn));
        }
    }
    try testing.expect(sides[0] and sides[1]);
}
