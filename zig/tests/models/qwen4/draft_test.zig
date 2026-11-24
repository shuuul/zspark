//! `draft.zig`'s tests: drafted decoding against serial steps, with drafts a test makes up from the serial reply
//! (right ones, wrong ones at chosen places, none); and under a constraint (spec 017), the same decoder, each
//! row's logits masked at the row's own place. The serial references other files' tests read too (`serialReply`,
//! `serialConstrained`) are `testing.zig`'s `serial`. The drafted reply on a checkpoint's first layers runs on
//! the loads other checks share (`draft_fixture.checkpointDrafted`, spec 133 WS-08).

const std = @import("std");
const fixtures = @import("testing.zig");
const cuda = @import("../../../src/cuda/cuda.zig");
const grammar = @import("../../../src/text/grammar.zig");
const json_schema = @import("../../../src/text/json_schema.zig");
const text_fixtures = @import("../../text/testing.zig");
const masks_file = @import("../../../src/decode/masks.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const tokenizer_file = @import("../../../src/text/tokenizer.zig");
const draft_file = @import("../../../src/models/qwen4/draft.zig");
const forward = @import("../../../src/models/qwen4/forward.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const Decoder = draft_file.Decoder;
const Drafter = draft_file.Drafter;
const forward_max_rows = draft_file.forward_max_rows;
const serialConstrained = fixtures.serial.serialConstrained;
const serialReply = fixtures.serial.serialReply;

const draft_fixture = @import("../../decode/draft_fixture.zig");
const never = draft_fixture.never;
const always = draft_fixture.always;
const everyThird = draft_fixture.everyThird;
const everyFifth = draft_fixture.everyFifth;
// the drafted replies, which the checkpoint's shared loads run too (`checkpoint_prefix_test.zig`,
// `forward_checkpoint_test.zig`)
const drafted_fixture = @import("draft_fixture.zig");
const Oracle = drafted_fixture.Oracle;
const replyWith = drafted_fixture.replyWith;
const expectDrafted = drafted_fixture.expectDrafted;

test "drafted decoding gives the serial reply, greedy and sampled, whatever the drafts" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModel(gpa, arena.allocator());
    defer m.deinit();
    const prompt = [_]i32{ 5, 300, 17, 17, 0, 251 };
    try expectDrafted(gpa, m, &prompt, 40, .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 });
    try expectDrafted(gpa, m, &prompt, 40, .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 });
}

test "a followed reply is the text given, its rounds verified against it, then the model's own" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModel(gpa, arena.allocator());
    defer m.deinit();
    const prompt = [_]i32{ 5, 300, 17, 17, 0, 251 };
    const count = 40;
    const none: []const u32 = &.{};
    const depth = m.max_rows - 1;
    for ([_]sampling.Sampling{
        .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 },
        .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 },
    }) |settings| {
        const serial = try serialReply(gpa, m, &prompt, count, settings);
        defer gpa.free(serial);
        // a text that is not the model's: its reply with one token in three changed, 25 tokens of it
        var text: [25]i32 = undefined;
        var changed: usize = 0;
        for (&text, serial[0..text.len], 0..) |*token, own, i| {
            token.* = if (i % 3 == 1) @mod(own + 7, 300) else own;
            changed += @intFromBool(token.* != own);
        }
        try testing.expect(changed >= text.len / 3);
        // after the text the model goes on by itself: the serial reply to the prompt and the text
        const whole = try gpa.alloc(i32, prompt.len + text.len);
        defer gpa.free(whole);
        @memcpy(whole[0..prompt.len], &prompt);
        @memcpy(whole[prompt.len..], &text);
        const after = try serialReply(gpa, m, whole, count - text.len, settings);
        defer gpa.free(after);
        const want = try gpa.alloc(i32, count);
        defer gpa.free(want);
        @memcpy(want[0..text.len], &text);
        @memcpy(want[text.len..], after);

        const follow: draft_file.Follow = .{ .tokens = &text, .first = prompt.len };
        for ([_]?*const fn (usize) bool{ null, never, everyThird }) |wrong| for ([_]bool{ true, false }) |graphed| {
            // the drafts are the followed text's, as a head that had read it would give them
            var oracle: Oracle = .{ .reply = want, .first = prompt.len, .count = depth, .wrong = wrong orelse never };
            const got = try replyWith(gpa, m, &prompt, count, settings, none, if (wrong != null) oracle.drafter() else null, graphed, null, follow);
            defer gpa.free(got.tokens);
            try testing.expectEqualSlices(i32, want, got.tokens);
            // a draft is kept where it is the text's token, not where it is the model's
            if (wrong == &never) try testing.expectEqual(got.stats.drafted, got.stats.accepted);
            if (wrong == &everyThird) try testing.expect(got.stats.accepted > 0 and got.stats.accepted < got.stats.drafted);
        };
        // following the model's own reply changes nothing
        const own = try replyWith(gpa, m, &prompt, count, settings, none, null, true, null, .{ .tokens = serial, .first = prompt.len });
        defer gpa.free(own.tokens);
        try testing.expectEqualSlices(i32, serial, own.tokens);

        // `Decoder.prompt` follows too: the reply's first token is the text's where the model's is another
        var state: forward.State = try .init(gpa, m, m.max_rows, 256);
        defer state.deinit();
        var b: forward.Buffers = try .init(gpa, m, m.max_rows, 256);
        defer b.deinit();
        var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
        defer sampler.deinit(gpa);
        const logits = try gpa.alloc(u16, m.shape.vocab);
        defer gpa.free(logits);
        const other = [1]i32{@mod(serial[0] + 7, 300)};
        var decoder: Decoder = .{ .m = m, .state = &state, .b = &b, .graphs = null, .sampler = &sampler, .settings = settings, .eos = none, .logits = logits, .follow = .{ .tokens = &other, .first = prompt.len } };
        var chosen: [forward_max_rows]i32 = undefined;
        const rows = try decoder.prompt(null, &prompt, &chosen);
        try testing.expectEqual(other[0], chosen[rows - 1]);
    }
}

const Constrained = struct {
    /// The reply ended with its end token, and how many of the replies' rounds cut a draft.
    ended: bool,
    pruned: usize,
    accepted: usize,
};

/// Under `schema`: the decoder's reply is the serial reference's, with no drafts and with an oracle's drafts,
/// right, wrong and mixed, eager and through graphs; and the document it wrote is the schema's.
fn expectConstrained(gpa: Allocator, m: model.Model, tokens: *const tokenizer_file.Tokenizer, engine: grammar.Engine, schema: grammar.Spec, prompt: []const i32, count: usize, settings: sampling.Sampling, gate: ?i32, eos: []const u32) !Constrained {
    var diagnostic: grammar.Diagnostic = .{};
    const constraint = engine.compile(gpa, schema, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message()});
        return err;
    };
    defer constraint.deinit();
    const mask = try gpa.alloc(u32, constraint.words);
    defer gpa.free(mask);

    const serial = try serialConstrained(gpa, m, prompt, count, settings, constraint, gate, eos);
    defer gpa.free(serial);
    const ended = std.mem.indexOfScalar(u32, eos, @intCast(serial[serial.len - 1])) != null;
    errdefer std.debug.print("serial reply: {any}\n", .{serial});

    // what was written after the gate is the schema's document: whole when the reply ended, else a beginning
    const from = if (gate) |token| if (std.mem.indexOfScalar(i32, serial, token)) |at| at + 1 else serial.len else 0;
    const document = try textOf(gpa, tokens, serial[from .. serial.len - @intFromBool(ended)]);
    defer gpa.free(document);
    errdefer std.debug.print("document: {s}\n", .{document});
    if (ended) {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const text = switch (schema) {
            .json_object => "{\"type\":\"object\"}",
            .json_schema => |text| text,
        };
        try testing.expect(try text_fixtures.validates(arena.allocator(), text, document));
    } else try testing.expectEqual(count, serial.len);
    // before the gate nothing was asked of the reply: it is the reply without a constraint
    if (gate != null) {
        const free = try serialReply(gpa, m, prompt, from, settings);
        defer gpa.free(free);
        try testing.expectEqualSlices(i32, free[0..from], serial[0..from]);
    }

    var result: Constrained = .{ .ended = ended, .pruned = 0, .accepted = 0 };
    const depth = m.max_rows - 1;
    const Case = struct { wrong: ?*const fn (usize) bool, count: usize, graphed: bool };
    for ([_]Case{
        .{ .wrong = null, .count = 0, .graphed = false },
        .{ .wrong = never, .count = depth, .graphed = true },
        .{ .wrong = never, .count = 1, .graphed = false },
        .{ .wrong = always, .count = depth, .graphed = true },
        .{ .wrong = everyThird, .count = depth, .graphed = true },
        .{ .wrong = everyFifth, .count = depth, .graphed = false },
        .{ .wrong = everyFifth, .count = 2, .graphed = true },
    }) |case| {
        var masks: masks_file.Masks = try .init(constraint, mask, gate);
        var oracle: Oracle = .{ .reply = serial, .first = prompt.len, .count = case.count, .wrong = case.wrong orelse never };
        const drafted = try replyWith(gpa, m, prompt, count, settings, eos, if (case.wrong != null) oracle.drafter() else null, case.graphed, &masks, null);
        defer gpa.free(drafted.tokens);
        try testing.expectEqualSlices(i32, serial, drafted.tokens);
        try testing.expectEqual(ended, masks.ended());
        // every row that chose a token after the gate was masked, and no other
        try testing.expect(masks.stats.masked >= serial.len - from);
        if (case.wrong == null) {
            try testing.expectEqual(serial.len - 1, drafted.stats.rounds);
            try testing.expectEqual(serial.len - from, masks.stats.masked);
        } else if (case.wrong.? == &never) {
            // right drafts are the constraint's own tokens: none is cut but the end token, and all are kept
            try testing.expect(masks.stats.pruned <= 1);
            try testing.expectEqual(drafted.stats.drafted, drafted.stats.accepted);
            try testing.expect(drafted.stats.rounds < serial.len - 1 or serial.len <= 2);
        } else if (case.wrong.? == &always) {
            try testing.expectEqual(0, drafted.stats.accepted);
            try testing.expectEqual(serial.len - 1, drafted.stats.rounds);
        }
        if (case.wrong != null and case.wrong.? != &never) result.pruned += masks.stats.pruned;
        result.accepted += drafted.stats.accepted;
    }
    return result;
}

const constrained = struct {
    /// Schemas a model of random weights finishes: no string it may write on in for ever.
    const flags =
        \\{"type":"object","properties":{"ok":{"type":"boolean"},"kind":{"enum":["a","ab","abc",12,null]},"n":{"type":["integer","null"]},"list":{"type":"array","items":{"type":"boolean"},"maxItems":3}},"required":["ok","kind"]}
    ;
    const tree =
        \\{"$defs":{"node":{"type":"object","properties":{"leaf":{"type":"boolean"},"next":{"anyOf":[{"$ref":"#/$defs/node"},{"type":"null"}]}},"required":["leaf","next"]}},"$ref":"#/$defs/node"}
    ;
    /// With strings, which a model of random weights may never close: its reply may be cut at `count`.
    const person =
        \\{"$defs":{"Kind":{"enum":["cat","dog"]}},"type":"object","properties":{"name":{"type":"string"},"kind":{"$ref":"#/$defs/Kind"},"age":{"anyOf":[{"type":"integer"},{"type":"null"}]}},"required":["name","kind"]}
    ;
    const union_ =
        \\{"oneOf":[{"type":"array","items":{"enum":["x","y"]},"minItems":1,"maxItems":4},{"type":"object","properties":{"a":{"const":"the the"}},"required":["a"]},{"type":"null"},{"type":"boolean"}]}
    ;
};

test "under a constraint drafted decoding still gives the serial reply, greedy and sampled, and the reply is the schema's" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModel(gpa, arena.allocator());
    defer m.deinit();
    var diagnostic: tokenizer_file.Diagnostic = .{};
    var tokens = try tokenizer_file.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const eos = [1]u32{tokens.addedId("<|end|>").?};
    var engine: json_schema.Engine = try .init(gpa, &tokens, m.shape.vocab, &eos);
    defer engine.deinit(gpa);

    const prompt = [_]i32{ 5, 300, 17, 17, 0, 251 };
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
    const count = 96;
    var ended: usize = 0;
    var pruned: usize = 0;
    var accepted: usize = 0;
    var cases: usize = 0;
    const specs = [_]grammar.Spec{ .{ .json_schema = constrained.flags }, .{ .json_schema = constrained.tree }, .{ .json_schema = constrained.union_ }, .{ .json_schema = constrained.person }, .json_object };
    for (specs) |spec| {
        for ([_]sampling.Sampling{
            greedy,
            .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 },
            .{ .seed = 12, .temperature = 1.0, .top_k = 0, .top_p = 1, .min_p = 0 },
            .{ .seed = 13, .temperature = 0.7, .top_k = 5, .top_p = 1, .min_p = 0 },
        }) |settings| {
            const got = try expectConstrained(gpa, m, &tokens, engine.interface(), spec, &prompt, count, settings, null, &eos);
            ended += @intFromBool(got.ended);
            pruned += got.pruned;
            accepted += got.accepted;
            cases += 1;
        }
    }
    // the replies that ended were whole documents, and most did; wrong drafts were cut, right ones kept
    try testing.expect(ended * 2 > cases);
    try testing.expect(pruned > 0 and accepted > 0);

    // thinking: the constraint begins after a token the free reply writes on its way
    const free = try serialReply(gpa, m, &prompt, 12, greedy);
    defer gpa.free(free);
    var gated: usize = 0;
    for ([_]usize{ 0, 4, 11 }) |at| {
        const gate = free[at];
        if (std.mem.indexOfScalar(i32, free, gate).? != at) continue;
        for ([_]grammar.Spec{ .{ .json_schema = constrained.flags }, .{ .json_schema = constrained.union_ } }) |spec| {
            _ = try expectConstrained(gpa, m, &tokens, engine.interface(), spec, &prompt, count, greedy, gate, &eos);
        }
        gated += 1;
    }
    try testing.expect(gated >= 2);
}

// Tests of the constraint's own drafts (spec 023): a drafter with no model behind it, which drafts the tokens
// the constraint forces and stops at the first it does not.

const Forced = struct {
    /// Drafts a call gives, at most.
    count: usize,

    fn drafter(self: *Forced) Drafter {
        return .{ .context = self, .draft = draft };
    }

    fn draft(context: *anyopaque, _: cuda.View, _: []const i32, _: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Forced = @ptrCast(@alignCast(context));
        var n: usize = 0;
        while (n < @min(out.len, self.count)) : (n += 1) {
            switch (try guide.?.next(out[0..n], true)) {
                .token => |token| out[n] = token,
                // a draft the constraint gave is one it takes
                .refused => return error.TestUnexpectedResult,
                .closed, .free, .among => break,
            }
        }
        return n;
    }
};

/// How many drafts `Forced` gives over the reply `serial`, at most `depth` a round, every one of them kept:
/// counted from the constraint alone, the reply walked from its start and each mask's tokens counted one by one.
fn forcedDrafts(gpa: Allocator, constraint: grammar.Constraint, serial: []const i32, count: usize, depth: usize, gate: ?i32) !usize {
    const mask = try gpa.alloc(u32, constraint.words);
    defer gpa.free(mask);
    // forced[i]: `serial[i]` is the one token that may come there, and the reply goes on after it
    const forced = try gpa.alloc(bool, serial.len);
    defer gpa.free(forced);
    var open = gate == null;
    var at = try constraint.start();
    for (serial, forced) |token, *is| {
        is.* = false;
        if (!open) {
            open = token == gate.?;
            continue;
        }
        try constraint.fill(at, mask);
        var allowed: usize = 0;
        for (0..constraint.words * 32) |id| allowed += @intFromBool(grammar.allowed(mask, @intCast(id)));
        at = (try constraint.accept(at, @intCast(token))).?;
        is.* = allowed == 1 and !constraint.ended(at);
    }
    var total: usize = 0;
    var made: usize = 1;
    while (made < serial.len) {
        var n: usize = 0;
        while (n < @min(depth, count - made - 1) and made + n < serial.len and forced[made + n]) n += 1;
        total += n;
        made += n + 1;
    }
    return total;
}

test "the constraint drafts the tokens it forces: each is kept, a run is drafted whole, and the reply is the serial one" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModel(gpa, arena.allocator());
    defer m.deinit();
    var diagnostic: tokenizer_file.Diagnostic = .{};
    var tokens = try tokenizer_file.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const eos = [1]u32{tokens.addedId("<|end|>").?};
    var engine: json_schema.Engine = try .init(gpa, &tokens, m.shape.vocab, &eos);
    defer engine.deinit(gpa);

    const prompt = [_]i32{ 5, 300, 17, 17, 0, 251 };
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
    const free = try serialReply(gpa, m, &prompt, 12, greedy);
    defer gpa.free(free);
    try testing.expectEqual(4, std.mem.indexOfScalar(i32, free, free[4]).?);
    const count = 96;
    const depth = m.max_rows - 1;
    var drafted: usize = 0;
    var gated: usize = 0;
    // replies with a run of forced tokens longer than a window of one draft, of two, holds
    var split = [2]usize{ 0, 0 };
    for ([_]grammar.Spec{ .{ .json_schema = constrained.flags }, .{ .json_schema = constrained.tree }, .{ .json_schema = constrained.union_ }, .{ .json_schema = constrained.person } }) |spec| {
        var said: grammar.Diagnostic = .{};
        const constraint = try engine.interface().compile(gpa, spec, &said);
        defer constraint.deinit();
        const mask = try gpa.alloc(u32, constraint.words);
        defer gpa.free(mask);
        for ([_]sampling.Sampling{
            greedy,
            .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 },
            .{ .seed = 12, .temperature = 1.0, .top_k = 0, .top_p = 1, .min_p = 0 },
        }, 0..) |settings, nth| {
            // thinking with the greedy reply, whose fifth token is the gate: nothing is forced before it
            for ([_]?i32{ null, free[4] }) |gate| {
                if (gate != null and nth != 0) continue;
                const serial = try serialConstrained(gpa, m, &prompt, count, settings, constraint, gate, &eos);
                defer gpa.free(serial);
                errdefer std.debug.print("serial reply: {any}\n", .{serial});
                for ([_]usize{ depth, 2, 1 }) |most| for ([_]bool{ true, false }) |graphed| {
                    var masks: masks_file.Masks = try .init(constraint, mask, gate);
                    var forced: Forced = .{ .count = most };
                    const got = try replyWith(gpa, m, &prompt, count, settings, &eos, forced.drafter(), graphed, &masks, null);
                    defer gpa.free(got.tokens);
                    try testing.expectEqualSlices(i32, serial, got.tokens);
                    // the row before a forced draft can choose nothing else, sampled or not
                    try testing.expectEqual(got.stats.drafted, got.stats.accepted);
                    try testing.expectEqual(0, masks.stats.pruned);
                    try testing.expectEqual(got.stats.drafted, masks.stats.forced);
                    try testing.expectEqual(try forcedDrafts(gpa, constraint, serial, count, most, gate), got.stats.drafted);
                    try testing.expectEqual(serial.len - 1, got.stats.rounds + got.stats.accepted);
                    drafted += got.stats.drafted;
                    if (gate != null) gated += got.stats.drafted;
                };
                const whole = try forcedDrafts(gpa, constraint, serial, count, depth, gate);
                for (&split, [_]usize{ 1, 2 }) |*n, most| n.* += @intFromBool(try forcedDrafts(gpa, constraint, serial, count, most, gate) < whole);
            }
        }
    }
    // tokens were forced, after a gate too, and runs of them were longer than a shallow window
    try testing.expect(drafted > 0 and gated > 0);
    try testing.expect(split[0] > 0 and split[1] > 0);
}

/// The text of `tokens`.
fn textOf(gpa: Allocator, tokens: *const tokenizer_file.Tokenizer, ids: []const i32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    const unsigned = try gpa.alloc(u32, ids.len);
    defer gpa.free(unsigned);
    for (unsigned, ids) |*to, from| to.* = @intCast(from);
    try tokens.decode(gpa, unsigned, .{}, &out);
    return out.toOwnedSlice(gpa);
}
