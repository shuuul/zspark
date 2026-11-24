//! `session.zig`'s tests of what a request asks: a prompt counted, a reply under a schema, a required tool
//! call, and the served forms of both.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const text_fixtures = @import("../../../text/testing.zig");
const Counting = fixtures.sessions.Counting;
const deepest = fixtures.sessions.deepest;
const masks_file = @import("../../../../src/decode/masks.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const request_file = @import("../../../../src/server/request.zig");
const structured = @import("../../../../src/server/structured.zig");
const grammar = @import("../../../../src/text/grammar.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const draft = @import("../../../../src/models/qwen4/draft.zig");
const session_file = @import("../../../../src/models/qwen4/session/session.zig");
const Session = session_file.Session;

/// Keeps what it is sent, the reasoning apart from the content.
const Collecting = struct {
    gpa: Allocator,
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    failed: bool = false,

    fn deinit(self: *Collecting) void {
        self.reasoning.deinit(self.gpa);
        self.content.deinit(self.gpa);
    }

    fn sink(self: *Collecting) request_file.Sink {
        return .{ .context = self, .writeFn = write };
    }

    fn write(context: *anyopaque, kind: request_file.Kind, text: []const u8) request_file.Sink.Signal {
        const self: *Collecting = @ptrCast(@alignCast(context));
        const to = switch (kind) {
            .reasoning => &self.reasoning,
            .content => &self.content,
        };
        to.appendSlice(self.gpa, text) catch {
            self.failed = true;
        };
        return .more;
    }
};

const constrained_schemas = [_][]const u8{
    \\{"type":"object","properties":{"ok":{"type":"boolean"},"kind":{"enum":["a","ab","abc",12,null]},"list":{"type":"array","items":{"type":"boolean"},"maxItems":3}},"required":["ok","kind"]}
    ,
    \\{"oneOf":[{"type":"array","items":{"enum":["x","y"]},"minItems":1,"maxItems":4},{"type":"object","properties":{"a":{"const":"the the"}},"required":["a"]},{"type":"null"},{"type":"boolean"}]}
    ,
    \\{"$defs":{"node":{"type":"object","properties":{"leaf":{"type":"boolean"},"next":{"anyOf":[{"$ref":"#/$defs/node"},{"type":"null"}]}},"required":["leaf","next"]}},"$ref":"#/$defs/node"}
};

test "a session's reply under a constraint is the serial one, the head drafting, and is the schema's document" {
    const json_schema = @import("../../../../src/text/json_schema.zig");
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), null);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const eos = [1]u32{tokens.addedId("<|end|>").?};
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &eos, .{ .window = m.max_rows, .capacity = 256 });
    defer session.destroy();
    var engine: json_schema.Engine = try .init(gpa, &tokens, m.shape.vocab, &eos);
    defer engine.deinit(gpa);

    const count = 96;
    const prompt = [_]i32{ 5, 17, 200, 3, 3, 91 };
    const samplings = [_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 11, .temperature = 0.9 }, .{ .seed = 12, .temperature = 1.4, .top_k = 30 } };
    // The tiny tokenizer has no think block: a token of the prompt stands for `<think>`, and one the free
    // reply writes for `</think>`.
    const free = try fixtures.serial.serialReply(gpa, m, &prompt, 8, samplings[0]);
    defer gpa.free(free);
    const close = free[5];
    try testing.expect(std.mem.indexOfScalar(i32, free, close).? == 5);

    var stopped: usize = 0;
    var masked: usize = 0;
    var forced: usize = 0;
    var kept: usize = 0;
    var ahead: usize = 0;
    var cuts: usize = 0;
    for (constrained_schemas) |schema| {
        var schema_diagnostic: grammar.Diagnostic = .{};
        const constraint = try engine.interface().compile(gpa, .{ .json_schema = schema }, &schema_diagnostic);
        defer constraint.deinit();
        for (samplings, 0..) |settings, nth| {
            // thinking with the greedy reply alone, whose sixth token is the gate
            for ([_]bool{ false, true }) |thinking| {
                if (thinking and nth != 0) continue;
                session.think = if (thinking) .{ .open = prompt[2], .close = close } else null;
                defer session.think = null;
                const gate: ?i32 = if (thinking) close else null;
                const expected = try fixtures.serial.serialConstrained(gpa, m, &prompt, count, settings, constraint, gate, &eos);
                defer gpa.free(expected);
                const ended = expected[expected.len - 1] == eos[0];
                // each depth of drafts, and none
                for ([_]usize{ m.max_rows - 1, 1, 0 }) |depth| {
                    var record: [count]i32 = undefined;
                    var collecting: Collecting = .{ .gpa = gpa };
                    defer collecting.deinit();
                    const reply = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = depth, .record = &record, .constraint = constraint }, collecting.sink());
                    try testing.expect(!collecting.failed);
                    try testing.expectEqualSlices(i32, expected[0 .. expected.len - @intFromBool(ended)], record[0..reply.tokens]);
                    try testing.expectEqual(if (ended) request_file.Finish.stop else request_file.Finish.length, reply.finish);
                    if (thinking) {
                        // what came before the gate is the free reply's, sent as reasoning; the constraint began after
                        try testing.expectEqualSlices(i32, free[0..6], record[0..6]);
                        try testing.expectEqual(6, reply.reasoning_tokens);
                        try testing.expect(collecting.reasoning.items.len > 0);
                        try testing.expectEqual(expected.len - 6, reply.masks.masked);
                    } else {
                        try testing.expectEqual(0, collecting.reasoning.items.len);
                        // a row is masked when its token is chosen, and only then
                        try testing.expectEqual(expected.len, reply.masks.masked);
                    }
                    // the head drafts under the constraint's guide: no draft of its is one the constraint cuts
                    try testing.expectEqual(0, reply.masks.pruned);
                    if (depth == 0) try testing.expectEqual(0, reply.masks.forced);
                    try testing.expect(reply.masks.forced <= reply.stats.drafted);
                    // the forced drafts that open a window were sent before it ran, and no other token was
                    try testing.expect(reply.ahead <= reply.masks.forced);
                    if (depth == 1) try testing.expectEqual(reply.masks.forced, reply.ahead);
                    ahead += reply.ahead;
                    if (reply.ahead > 0 and depth > 1) {
                        // A sink that stops at each of its first pieces: the reply is cut there, with the tokens
                        // sent so far, and the state holds them all, so the same prompt again gives the reply whole.
                        for (1..6) |limit| {
                            var short: [count]i32 = undefined;
                            var counting: Counting = .{ .limit = limit };
                            const cut = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = depth, .record = &short, .constraint = constraint }, counting.sink());
                            if (cut.finish != .cut) break;
                            try testing.expectEqualSlices(i32, record[0..cut.tokens], short[0..cut.tokens]);
                            try testing.expect(session.known.items.len >= prompt.len + cut.tokens);
                            try testing.expectEqualSlices(i32, record[0..cut.tokens], session.known.items[prompt.len..][0..cut.tokens]);
                            cuts += 1;
                        }
                        var whole: [count]i32 = undefined;
                        var counting: Counting = .{};
                        const after = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = depth, .record = &whole, .constraint = constraint }, counting.sink());
                        try testing.expectEqualSlices(i32, record[0..reply.tokens], whole[0..after.tokens]);
                    }
                    if (ended) {
                        errdefer std.debug.print("content: {s}\n", .{collecting.content.items});
                        try testing.expect(try text_fixtures.validates(arena.allocator(), schema, collecting.content.items));
                        stopped += 1;
                    }
                    masked += reply.masks.masked;
                    forced += reply.masks.forced;
                    kept += reply.stats.accepted;
                    try testing.expect(reply.masks.nanoseconds > 0);
                }
            }
        }
    }
    // most replies were whole documents; the constraint drafted the tokens it forces, and drafts were kept
    try testing.expect(stopped >= 18);
    try testing.expect(masked > 0 and forced > 0 and kept > 0);
    try testing.expect(ahead > 0 and cuts > 0);
}

test "without a constraint a reply asks the matcher nothing and is the reply it was" {
    const json_schema = @import("../../../../src/text/json_schema.zig");
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelDrafting(gpa, arena.allocator(), null);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const eos = [1]u32{tokens.addedId("<|end|>").?};
    const count = 24;
    const prompt = [_]i32{ 5, 17, 200, 3, 3, 91 };
    const samplings = [_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 11, .temperature = 0.9 } };

    // a session that never saw a constraint: its replies and the drafts they were given
    var plain: [samplings.len]struct { tokens: [count]i32, made: usize, drafts: u64, stats: draft.Stats } = undefined;
    {
        const session: *Session = try .create(gpa, testing.io, m, &tokens, &eos, .{ .window = m.max_rows, .capacity = 256 });
        defer session.destroy();
        for (samplings, &plain) |settings, *kept| {
            var counting: Counting = .{};
            const reply = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = deepest, .record = &kept.tokens }, counting.sink());
            kept.made = reply.tokens;
            kept.drafts = reply.drafts;
            kept.stats = reply.stats;
            const serial = try fixtures.serial.serialReply(gpa, m, &prompt, count, settings);
            defer gpa.free(serial);
            try testing.expectEqualSlices(i32, serial[0..reply.tokens], kept.tokens[0..reply.tokens]);
        }
    }

    // a session beside an engine, between replies under a constraint
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &eos, .{ .window = m.max_rows, .capacity = 256 });
    defer session.destroy();
    var engine: json_schema.Engine = try .init(gpa, &tokens, m.shape.vocab, &eos);
    defer engine.deinit(gpa);
    var schema_diagnostic: grammar.Diagnostic = .{};
    const constraint = try engine.interface().compile(gpa, .{ .json_schema = constrained_schemas[0] }, &schema_diagnostic);
    defer constraint.deinit();
    for (0..2) |_| {
        var counting: Counting = .{};
        const before = json_schema.calls;
        const under = try session.reply(&prompt, samplings[0], .{ .max_tokens = count, .depth = deepest, .constraint = constraint }, counting.sink());
        // the counter counts: a constrained reply asks at every row
        try testing.expect(json_schema.calls >= before + under.masks.masked);
        try testing.expect(under.masks.masked > 0);
        for (samplings, plain) |settings, kept| {
            var record: [count]i32 = undefined;
            const asked = json_schema.calls;
            const reply = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = deepest, .record = &record }, counting.sink());
            try testing.expectEqual(asked, json_schema.calls);
            try testing.expectEqual(masks_file.Stats{}, reply.masks);
            try testing.expectEqual(kept.made, reply.tokens);
            try testing.expectEqualSlices(i32, kept.tokens[0..kept.made], record[0..reply.tokens]);
            // the same drafts were given and the same kept: nothing was cut
            try testing.expectEqual(kept.drafts, reply.drafts);
            try testing.expectEqual(kept.stats.drafted, reply.stats.drafted);
            try testing.expectEqual(kept.stats.accepted, reply.stats.accepted);
        }
    }
}

test "a required tool call: drafted is serial, and every reply is calls of the tools offered with arguments their schemas take" {
    const json_schema = @import("../../../../src/text/json_schema.zig");
    const call_gate = @import("../../../../src/text/call_gate.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var m = try fixtures.random_model.testModelDrafting(gpa, arena, null);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const eos = [1]u32{tokens.addedId("<|end|>").?};
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &eos, .{ .window = m.max_rows, .capacity = 256 });
    defer session.destroy();
    var engine: json_schema.Engine = try .init(gpa, &tokens, m.shape.vocab, &eos);
    defer engine.deinit(gpa);
    // what `served.zig` compiles a request's structure with
    const compiler: structured.Compiler = .{ .engine = engine.interface(), .tokenizer = &tokens, .stops = &eos };

    // Schemas no document of one of which is the other's, so a call held to the wrong tool's would not validate.
    const add = "{\"type\":\"object\",\"properties\":{\"a\":{\"enum\":[1,2]},\"b\":{\"type\":\"boolean\"}},\"required\":[\"a\",\"b\"],\"additionalProperties\":false}";
    const sub = "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"],\"additionalProperties\":false}";
    const both = "[{\"type\":\"function\",\"function\":{\"name\":\"add\",\"parameters\":" ++ add ++ "}},{\"type\":\"function\",\"function\":{\"name\":\"sub\",\"parameters\":" ++ sub ++ "}}]";
    // a named function reaches the engine as the one tool offered
    const named = "[{\"type\":\"function\",\"function\":{\"name\":\"sub\",\"parameters\":" ++ sub ++ "}}]";
    // The gate writes `sub` and `note` in the template's form, `note`'s text as a raw string that ends where its
    // closer is written, and `add`, whose `a` says no type, as JSON.
    const note = "{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"}},\"required\":[\"text\",\"n\"],\"additionalProperties\":false}";
    const noted = "[{\"type\":\"function\",\"function\":{\"name\":\"note\",\"parameters\":" ++ note ++ "}}]";
    // the same tools as the test's own reader of a reply's calls takes them (`text/testing.zig`'s `gate.read`)
    const offered = [_]call_gate.Tool{ .{ .name = "add", .parameters = add }, .{ .name = "sub", .parameters = sub }, .{ .name = "note", .parameters = note } };
    const Case = struct { tools: []const u8, single: bool };
    const cases = [_]Case{ .{ .tools = both, .single = false }, .{ .tools = both, .single = true }, .{ .tools = named, .single = false }, .{ .tools = noted, .single = true } };

    const count = 240;
    const prompt = [_]i32{ 5, 17, 200, 3, 3, 91 };
    const samplings = [_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 11, .temperature = 0.9 }, .{ .seed = 12, .temperature = 1.4, .top_k = 30 }, .{ .seed = 13, .temperature = 2.5 } };
    const free = try fixtures.serial.serialReply(gpa, m, &prompt, 8, samplings[0]);
    defer gpa.free(free);
    const close = free[5];
    try testing.expect(std.mem.indexOfScalar(i32, free, close).? == 5);

    var stopped: usize = 0;
    var forced: usize = 0;
    var drafted: usize = 0;
    var kept: usize = 0;
    var called_add = false;
    var called_sub = false;
    var raw: usize = 0;
    var replies: usize = 0;
    var led_replies: usize = 0;
    const lead = "\n\n";
    for (cases) |case| {
        const tools = try std.json.parseFromSliceLeaky(std.json.Value, arena, case.tools, .{});
        var said: request_file.Diagnostic = .{};
        const plain = (try compiler.compile(gpa, arena, .{ .tools = tools, .structure = .{ .call = .{ .single = case.single } } }, &said)).?;
        defer plain.deinit();
        // after thinking, as `served.zig` compiles it: with the text a template writes before the call
        var leading = compiler;
        leading.gate.lead = lead;
        const led = (try leading.compile(gpa, arena, .{ .tools = tools, .structure = .{ .call = .{ .single = case.single } } }, &said)).?;
        defer led.deinit();
        for (samplings, 0..) |settings, nth| {
            for ([_]bool{ false, true }) |thinking| {
                if (thinking and nth != 0) continue;
                session.think = if (thinking) .{ .open = prompt[2], .close = close } else null;
                defer session.think = null;
                const gate: ?i32 = if (thinking) close else null;
                const constraint = if (thinking) led else plain;
                const expected = try fixtures.serial.serialConstrained(gpa, m, &prompt, count, settings, constraint, gate, &eos);
                defer gpa.free(expected);
                const ended = expected[expected.len - 1] == eos[0];
                for ([_]usize{ m.max_rows - 1, 1, 0 }) |depth| {
                    var record: [count]i32 = undefined;
                    var collecting: Collecting = .{ .gpa = gpa };
                    defer collecting.deinit();
                    const reply = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = depth, .record = &record, .constraint = constraint }, collecting.sink());
                    try testing.expect(!collecting.failed);
                    try testing.expectEqualSlices(i32, expected[0 .. expected.len - @intFromBool(ended)], record[0..reply.tokens]);
                    try testing.expectEqual(if (ended) request_file.Finish.stop else request_file.Finish.length, reply.finish);
                    // the reasoning is free and the call begins after it, after the lead: its tokens are in the
                    // reply, and the text a client is sent begins with the call all the same
                    if (thinking) {
                        try testing.expectEqualSlices(i32, free[0..6], record[0..6]);
                        var written: std.ArrayList(u8) = .empty;
                        try tokens.decode(arena, @ptrCast(record[6..reply.tokens]), .{}, &written);
                        const begins = lead ++ "<tool_call>\n";
                        try testing.expectEqualStrings(begins, written.items[0..@min(begins.len, written.items.len)]);
                        led_replies += 1;
                    }
                    try testing.expectEqual(expected.len - @as(usize, if (thinking) 6 else 0), reply.masks.masked);
                    // the gate's frame is drafted by the gate and the head chooses among what it allows: none cut
                    try testing.expectEqual(0, reply.masks.pruned);
                    if (depth == 0) try testing.expectEqual(0, reply.masks.forced);
                    forced += reply.masks.forced;
                    drafted += reply.stats.drafted;
                    kept += reply.stats.accepted;
                    // Whole or cut by the length, the reply begins as a call: nothing comes before it.
                    errdefer std.debug.print("content: {s}\n", .{collecting.content.items});
                    // in the template's form (`sub`) or, for a tool it cannot hold (`add`), as JSON
                    const opening = "<tool_call>\n";
                    try testing.expectEqualStrings(opening[0..@min(opening.len, collecting.content.items.len)], collecting.content.items[0..@min(opening.len, collecting.content.items.len)]);
                    replies += 1;
                    // a raw string was decoded: text went on after its frame
                    const framed = "<parameter=text>\n";
                    if (std.mem.indexOf(u8, collecting.content.items, framed)) |at| raw += @intFromBool(collecting.content.items.len > at + framed.len + 8);
                    if (!ended) continue;
                    stopped += 1;
                    // as a reply's parser reads it: calls and no content
                    const calls = try text_fixtures.gate.read(arena, collecting.content.items, &offered);
                    try testing.expect(calls.len != 0);
                    if (case.single) try testing.expectEqual(1, calls.len);
                    for (calls) |call| {
                        errdefer std.debug.print("call {s} {s}\n", .{ call.name, call.arguments });
                        const is_add = std.mem.eql(u8, call.name, "add");
                        const is_note = case.tools.ptr == noted.ptr;
                        if (!is_add) try testing.expectEqualStrings(if (is_note) "note" else "sub", call.name);
                        if (case.tools.ptr == named.ptr) try testing.expectEqualStrings("sub", call.name);
                        try testing.expect(try text_fixtures.validates(arena, if (is_add) add else if (is_note) note else sub, call.arguments));
                        if (is_add) called_add = true else if (!is_note) called_sub = true;
                    }
                }
            }
        }
    }
    // Measured: 45 of the 60 replies whole, the 45 of `add` and `sub`. This model ends its reply after one call
    // every time; a second call is walked on the host (`text/call_gate.zig`). The 15 of
    // `note` are in its raw string when the length cuts them: this model does not write the closer.
    try testing.expect(replies == 60 and stopped >= 30 and led_replies == 12);
    // Measured: 1851 of the 3866 drafts were the gate's, and 2044 were kept.
    try testing.expect(forced > 0 and kept >= forced and kept < drafted);
    try testing.expect(called_add and called_sub and raw >= 10);
}
