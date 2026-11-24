//! `call_gate.zig`'s tests of the JSON form and of the gate under `grammar.zig`'s interface.

const std = @import("std");
const fixtures = @import("testing.zig");
const fx = fixtures.gate;
const call_gate = @import("../../src/text/call_gate.zig");
const grammar = @import("../../src/text/grammar.zig");
const Tool = call_gate.Tool;
const compile = call_gate.compile;

// -- tests -------------------------------------------------------------------------------------------------------

const testing = std.testing;

test "in the JSON form a gated reply is calls and nothing else: an offered name, arguments its schema takes" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    var prng: std.Random.DefaultPrng = .init(0x5eed_017);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);

    const sets = [_][]const Tool{
        &.{fx.add},
        &.{ fx.add, fx.lookup, fx.nothing },
        &.{ fx.abc, fx.a, fx.ab },
        &.{ fx.quoted, fx.lookup },
    };
    for (sets) |tools| {
        for ([_]bool{ false, true }) |single| {
            const constraint = try bench.gate(tools, .{ .single = single, .template_form = false }, &diagnostic);
            defer constraint.deinit();
            const called = try arena.alloc(usize, tools.len);
            @memset(called, 0);
            var several: usize = 0;
            var ended: usize = 0;
            for (0..120) |_| {
                if (!try fx.walk(gpa, constraint, bench.engine.vocabulary.width, bench.stops[0], prng.random(), 60, 600, &ids, &bench.tokenizer)) continue;
                ended += 1;
                text.clearRetainingCapacity();
                try bench.tokenizer.decode(gpa, ids.items, .{ .skip_special = true }, &text);
                errdefer std.debug.print("reply: {s}\n", .{text.items});
                const calls = try fx.calls(arena, text.items);
                try testing.expect(calls.len >= 1);
                if (single) try testing.expectEqual(1, calls.len);
                several += @intFromBool(calls.len > 1);
                for (calls) |call| {
                    const tool = for (tools, 0..) |tool, index| {
                        if (std.mem.eql(u8, tool.name, call.name)) break index;
                    } else return error.TestUnexpectedResult;
                    called[tool] += 1;
                    try testing.expect(try fixtures.validates(arena, tools[tool].parameters, call.arguments));
                }
            }
            // nearly every walk ends, every tool is called, and without `single` some replies call twice
            try testing.expect(ended >= 100);
            for (called) |count| try testing.expect(count > 0);
            try testing.expectEqual(single, several == 0);
        }
    }
}

test "the gate keeps the interface's contract: a token is taken exactly where its bit is set, positions are values, a reply follows a reply" {
    const gpa = testing.allocator;
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    const constraint = try bench.gate(&.{ fx.ab, fx.lookup, fx.a }, .{}, &diagnostic);
    defer constraint.deinit();
    const words = constraint.words;
    const width = bench.engine.vocabulary.width;
    const stop = bench.stops[0];
    const masks = try gpa.alloc(u32, 2 * words);
    defer gpa.free(masks);
    const got = masks[0..words];
    const again = masks[words..];

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var between: usize = 0;
    const reply = "<tool_call>\n{\"name\": \"lookup\", \"arguments\":{\"kind\": \"ab\", \"tags\": [\"x\"]}}\n</tool_call>\n" ++
        "<tool_call>\n{\"name\": \"a\", \"arguments\":{\"x\":1}}\n</tool_call>";
    // cut at the places the gate cuts it: the openings, the arguments and the closes are encoded apart
    for ([_][]const u8{
        "<tool_call>\n{\"name\": \"lookup\", \"arguments\":", "{\"kind\": \"ab\", \"tags\": [\"x\"]}", "}\n</tool_call>",
        "\n<tool_call>\n{\"name\": \"a\", \"arguments\":",    "{\"x\":1}",                             "}\n</tool_call>",
    }, 0..) |part, index| {
        // between the two calls the reply may end
        if (index == 3) between = ids.items.len;
        try bench.tokenizer.encode(gpa, part, &ids);
    }
    var back: std.ArrayList(u8) = .empty;
    defer back.deinit(gpa);
    try bench.tokenizer.decode(gpa, ids.items, .{}, &back);
    try testing.expectEqualStrings(reply, back.items);

    const kept = try gpa.alloc(grammar.Position, ids.items.len + 1);
    defer gpa.free(kept);
    const hashes = try gpa.alloc(u64, ids.items.len + 1);
    defer gpa.free(hashes);
    // twice: the constraint serves one reply after another
    for (0..2) |round| {
        var at = try constraint.start();
        for (ids.items, 0..) |id, step| {
            kept[step] = at;
            try constraint.fill(at, got);
            const hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(got));
            if (round == 1) try testing.expectEqual(hashes[step], hash);
            hashes[step] = hash;
            try testing.expect(grammar.allowed(got, id));
            try testing.expectEqual(step == between, grammar.allowed(got, stop));
            try testing.expect(!constraint.ended(at));
            // a draft tree from here: every token tried, none followed
            for (0..width + 64) |token| {
                const next = try constraint.accept(at, @intCast(token));
                try testing.expectEqual(token < width and grammar.allowed(got, @intCast(token)), next != null);
            }
            try constraint.fill(at, again);
            try testing.expectEqualSlices(u32, got, again);
            at = (try constraint.accept(at, id)).?;
        }
        // after the calls: the end or another call, and after the end nothing
        try constraint.fill(at, got);
        try testing.expect(grammar.allowed(got, stop));
        var count: usize = 0;
        for (got) |word| count += @popCount(word);
        try testing.expectEqual(2, count);
        const end = (try constraint.accept(at, stop)).?;
        try testing.expect(constraint.ended(end) and !constraint.ended(at));
        try constraint.fill(end, got);
        try testing.expect(std.mem.allEqual(u32, got, 0));
        try testing.expectEqual(null, try constraint.accept(end, stop));
        // the positions on the way still give their masks
        for (kept[0..ids.items.len], hashes[0..ids.items.len]) |position, hash| {
            try constraint.fill(position, got);
            try testing.expectEqual(hash, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(got)));
        }
    }
    // a position of the reply before is not this reply's
    _ = try constraint.start();
    try testing.expectError(error.ConstraintFailed, constraint.fill(kept[ids.items.len - 1], got));
}

test "what is forced is one token a mask; a named function is that function; the tool decides the schema" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    const mask = try gpa.alloc(u32, bench.engine.interface().words);
    defer gpa.free(mask);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);

    // one tool: every token up to the arguments is the only one allowed
    {
        const constraint = try bench.gate(&.{fx.ab}, .{ .single = true }, &diagnostic);
        defer constraint.deinit();
        try bench.tokenizer.encode(gpa, "<tool_call>\n{\"name\": \"ab\", \"arguments\":", &ids);
        var at = try constraint.start();
        for (ids.items) |id| {
            try constraint.fill(at, mask);
            var count: usize = 0;
            for (mask) |word| count += @popCount(word);
            try testing.expectEqual(1, count);
            at = (try constraint.accept(at, id)).?;
        }
        // `ab`'s schema, not another tool's: x is 2
        ids.clearRetainingCapacity();
        try bench.tokenizer.encode(gpa, "{\"x\":2}", &ids);
        for (ids.items) |id| at = (try constraint.accept(at, id)).?;
        ids.clearRetainingCapacity();
        try bench.tokenizer.encode(gpa, "}\n</tool_call>", &ids);
        for (ids.items) |id| {
            try constraint.fill(at, mask);
            var count: usize = 0;
            for (mask) |word| count += @popCount(word);
            try testing.expectEqual(1, count);
            at = (try constraint.accept(at, id)).?;
        }
        // one call was asked for: the end alone
        try constraint.fill(at, mask);
        var count: usize = 0;
        for (mask) |word| count += @popCount(word);
        try testing.expectEqual(1, count);
        try testing.expect(grammar.allowed(mask, bench.stops[0]));
    }

    // three tools whose names go on from one another, each with its own schema: after each name only its own
    // arguments are taken
    const tools = [_]Tool{ fx.a, fx.abc, fx.ab };
    const constraint = try bench.gate(&tools, .{}, &diagnostic);
    defer constraint.deinit();
    for (tools, [_][]const u8{ "{\"x\":1}", "{\"x\":3}", "{\"x\":2}" }) |tool, arguments| {
        for ([_][]const u8{ "{\"x\":1}", "{\"x\":2}", "{\"x\":3}" }) |tried| {
            const head = try std.fmt.allocPrint(arena_state.allocator(), "<tool_call>\n{{\"name\": \"{s}\", \"arguments\":", .{tool.name});
            ids.clearRetainingCapacity();
            try bench.tokenizer.encode(gpa, head, &ids);
            var at = try constraint.start();
            for (ids.items) |id| at = (try constraint.accept(at, id)).?;
            ids.clearRetainingCapacity();
            try bench.tokenizer.encode(gpa, tried, &ids);
            const taken = for (ids.items) |id| {
                at = try constraint.accept(at, id) orelse break false;
            } else true;
            try testing.expectEqual(std.mem.eql(u8, tried, arguments), taken);
        }
    }
}

test "what a gate cannot hold is refused, and says which tool" {
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    const Case = struct { tools: []const Tool, err: grammar.CompileError, message: []const u8 };
    for ([_]Case{
        .{ .tools = &.{}, .err = error.InvalidSchema, .message = "a required tool call needs a tool" },
        .{ .tools = &.{ fx.add, fx.lookup, .{ .name = "ADD", .parameters = "{}" } }, .err = error.InvalidSchema, .message = "two tools are named 'ADD': a required call could not say which it calls" },
        .{ .tools = &.{ fx.add, .{ .name = "find", .parameters = "{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\",\"pattern\":\"^a\"}}}" } }, .err = error.UnsupportedSchema, .message = "tool 'find': its parameters: #/q: `pattern` is not read" },
        .{ .tools = &.{.{ .name = "find", .parameters = "{\"type\":\"object\"" }}, .err = error.InvalidSchema, .message = "tool 'find': its parameters: the schema is not JSON: UnexpectedEndOfInput" },
        // a name that holds a special token's text, or the end token's, would not come back out of the reply
        .{ .tools = &.{.{ .name = "a<l>b", .parameters = "{}" }}, .err = error.InvalidSchema, .message = "tool 'a<l>b': this model's tokenizer cannot write a call to that name" },
        .{ .tools = &.{ fx.add, .{ .name = "<|end|>", .parameters = "{}" } }, .err = error.InvalidSchema, .message = "tool '<|end|>': this model's tokenizer cannot write a call to that name" },
        .{ .tools = &.{.{ .name = "caf\xff", .parameters = "{}" }}, .err = error.InvalidSchema, .message = "a tool's name is not UTF-8" },
    }) |case| {
        errdefer std.debug.print("said: {s}\n", .{diagnostic.message()});
        try testing.expectError(case.err, bench.gate(case.tools, .{}, &diagnostic));
        try testing.expectEqualStrings(case.message, diagnostic.message());
    }
    // without an end token there is no reply to end
    try testing.expectError(error.InvalidSchema, compile(testing.allocator, bench.engine.interface(), &bench.tokenizer, &.{}, &.{fx.add}, .{}, &diagnostic));
}

test "past its bound of states the gate fails instead of growing, and the next reply starts clean" {
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    const constraint = try bench.gate(&.{fx.lookup}, .{ .max_states = 24 }, &diagnostic);
    defer constraint.deinit();
    var prng: std.Random.DefaultPrng = .init(7);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(testing.allocator);
    try testing.expectError(error.ConstraintFailed, fx.walk(testing.allocator, constraint, bench.engine.vocabulary.width, bench.stops[0], prng.random(), 1000, 1000, &ids, &bench.tokenizer));
    try testing.expect(ids.items.len < 24);
    // the reply after it begins as any other
    const mask = try testing.allocator.alloc(u32, constraint.words);
    defer testing.allocator.free(mask);
    try constraint.fill(try constraint.start(), mask);
    var count: usize = 0;
    for (mask) |word| count += @popCount(word);
    try testing.expectEqual(1, count);
}
