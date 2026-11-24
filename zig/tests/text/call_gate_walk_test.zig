//! `call_gate.zig`'s tests of the template's form: the text automaton, which form a tool's call is written
//! in, replies walked through the masks, and the lead. The checkpoint's template and tokenizer:
//! `call_gate_checkpoint_test.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("testing.zig");
const fx = fixtures.gate;
const call_gate = @import("../../src/text/call_gate.zig");
const grammar = @import("../../src/text/grammar.zig");
const Value = std.json.Value;
const Gate = call_gate.Gate;
const Strings = call_gate.Strings;
const Tokenizer = call_gate.Tokenizer;
const Tool = call_gate.Tool;
const automaton = call_gate.automaton;

// -- tests of the template's form --------------------------------------------------------------------------------

pub const tx = struct {
    const sum: Tool = fx.add;
    const note: Tool = .{ .name = "note", .parameters = "{\"type\":\"object\",\"properties\":{\"kind\":{\"type\":\"string\",\"enum\":[\"a\",\"ab\",\"abc\"]},\"body\":{\"type\":\"string\",\"description\":\"any text\"},\"tags\":{\"type\":\"array\",\"items\":{\"enum\":[\"x\",\"y\"]},\"maxItems\":3},\"deep\":{\"type\":\"object\",\"properties\":{\"s\":{\"type\":\"string\"}}},\"n\":{\"type\":\"null\"},\"mode\":{\"type\":\"string\",\"const\":\"on\"}},\"required\":[\"kind\",\"body\"]}" };

    /// A raw string's end, and how much of it `ids` ends with.
    const closer = "\n</parameter>\n";

    pub fn gateOf(constraint: grammar.Constraint) *Gate {
        return @ptrCast(@alignCast(constraint.context));
    }

    /// As `fx.walk`, and once the reply is `hurry` tokens long a raw string is ended too: the tokens that
    /// write its closer are taken where they may be.
    fn walk(gpa: Allocator, constraint: grammar.Constraint, vocabulary: usize, stop: u32, random: std.Random, hurry: usize, limit: usize, ids: *std.ArrayList(u32), tokenizer: *const Tokenizer) !bool {
        const mask = try gpa.alloc(u32, constraint.words);
        defer gpa.free(mask);
        var allowed: std.ArrayList(u32) = .empty;
        defer allowed.deinit(gpa);
        ids.clearRetainingCapacity();
        var at = try constraint.start();
        // how much of the closer the raw string being hurried ends with
        var closing: usize = 0;
        while (ids.items.len < limit) {
            try constraint.fill(at, mask);
            allowed.clearRetainingCapacity();
            for (0..vocabulary) |token| {
                if (grammar.allowed(mask, @intCast(token))) try allowed.append(gpa, @intCast(token));
            }
            try testing.expect(allowed.items.len != 0);
            var token = allowed.items[random.uintLessThan(usize, allowed.items.len)];
            const phase = gateOf(constraint).states.items[@intCast(@backingInt(at))].phase;
            if (phase != .raw) closing = 0;
            if (ids.items.len >= hurry) {
                if (phase == .raw) {
                    for (allowed.items) |candidate| {
                        const bytes = tokenizer.tokenBytes(candidate);
                        if (std.mem.startsWith(u8, closer[closing..], bytes)) token = candidate;
                    }
                    const bytes = tokenizer.tokenBytes(token);
                    closing = if (std.mem.startsWith(u8, closer[closing..], bytes)) closing + bytes.len else 0;
                } else for (allowed.items) |candidate| {
                    const bytes = tokenizer.tokenBytes(candidate);
                    if (bytes.len == 1 and std.mem.indexOfScalar(u8, "\"}]", bytes[0]) != null) token = candidate;
                }
            }
            at = (try constraint.accept(at, token)).?;
            if (token == stop) {
                try testing.expect(constraint.ended(at));
                return true;
            }
            try testing.expect(!constraint.ended(at));
            try ids.append(gpa, token);
        }
        return false;
    }
};

test "the text automaton: a raw string goes on exactly while it holds no closer, and ends with the one after a newline" {
    // What the automaton must say, from the text alone.
    const spec = struct {
        fn holds(haystack: []const u8, needle: []const u8) bool {
            return fx.findFolded(haystack, needle) != null;
        }
        /// The longest beginning of `bytes` that is whole characters, where the rest begins one; null where
        /// no UTF-8 text begins with `bytes`.
        fn characters(bytes: []const u8) ?usize {
            var at: usize = 0;
            while (at < bytes.len) {
                const length = std.unicode.utf8ByteSequenceLength(bytes[at]) catch return null;
                if (at + length > bytes.len) {
                    // a character cut short: some ending must make it one
                    var whole: [4]u8 = @splat(0x80);
                    @memcpy(whole[0 .. bytes.len - at], bytes[at..]);
                    for ([_]u8{ 0x80, 0x90, 0xa0, 0xbf }) |filler| {
                        if (bytes.len - at == 1) whole[1] = filler;
                        if (std.unicode.utf8ValidateSlice(whole[0..length])) return at;
                    }
                    return null;
                }
                if (!std.unicode.utf8ValidateSlice(bytes[at..][0..length])) return null;
                at += length;
            }
            return at;
        }
        /// `dead`, `done`, or 0 for a value that goes on.
        fn of(bytes: []const u8) automaton.Tail {
            const end = "\n</parameter>\n";
            for (0..bytes.len + 1) |length| {
                const head = bytes[0..length];
                if (!std.mem.endsWith(u8, head, end[0 .. end.len - 1])) continue;
                // the first closer: it ends the value if it is this one, in lower case after a newline
                if (holds(head[0 .. head.len - 1], "</parameter>")) return automaton.dead;
                if (holds(head, "</tool_call>") or characters(head) != head.len) return automaton.dead;
                if (length == bytes.len) return 0;
                return if (length + 1 == bytes.len and bytes[length] == '\n') automaton.done else automaton.dead;
            }
            if (holds(bytes, "</parameter>") or holds(bytes, "</tool_call>") or characters(bytes) == null) return automaton.dead;
            return 0;
        }
    };
    const pieces = [_][]const u8{ "<", "/", "</", "parameter", "PARAMETER", "Parameter", "param", "eter", ">", "\n", "\n\n", "tool_call", "TOOL_CALL", "tool", "_call", "x", " ", "p", "t", "é", "\xc3", "\xa9", "日", "\xe6", "\x97", "\xa5", "\xf0\x9f", "\x98\x80", "\xff", "\xc0", "\xed\xa0", "\n</parameter>", "</parameter>", "\n</parameter>\n", "</tool_call>" };
    var prng: std.Random.DefaultPrng = .init(0x023);
    const random = prng.random();
    var buffer: [128]u8 = undefined;
    var seen = [3]usize{ 0, 0, 0 };
    for (0..200_000) |_| {
        var length: usize = 0;
        for (0..random.uintLessThan(usize, 9)) |_| {
            const piece = pieces[random.uintLessThan(usize, pieces.len)];
            @memcpy(buffer[length..][0..piece.len], piece);
            length += piece.len;
        }
        const bytes = buffer[0..length];
        const expected = spec.of(bytes);
        const got = automaton.walk(automaton.plain, bytes);
        errdefer std.debug.print("text {any} gives {d}, expected {d}\n", .{ bytes, got, expected });
        switch (expected) {
            automaton.dead => {
                try testing.expectEqual(automaton.dead, got);
                seen[0] += 1;
            },
            automaton.done => {
                try testing.expectEqual(automaton.done, got);
                seen[1] += 1;
            },
            else => {
                try testing.expect(got != automaton.dead and got != automaton.done);
                // a JSON value may stand there unless the text ends a raw string
                try testing.expectEqual(!std.mem.endsWith(u8, bytes, "\n</parameter>"), automaton.inside(got));
                seen[2] += 1;
            },
        }
    }
    for (seen) |count| try testing.expect(count > 1000);
    // any string without a closer is written, the empty one and one of newlines too
    for ([_][]const u8{ "", "\n", "a\n\n", "<parameter=x>", "</parameter", "</function>", "é日😀" }) |value| try testing.expect(automaton.writable(value));
    for ([_][]const u8{ "</parameter>", "a</PARAMETER>b", "</Tool_Call>", "\xff", "\xc3" }) |value| try testing.expect(!automaton.writable(value));
}

test "which form a tool's call is written in" {
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    const Case = struct { name: []const u8 = "f", parameters: []const u8, template: bool };
    for ([_]Case{
        .{ .parameters = fx.add.parameters, .template = true },
        .{ .parameters = tx.note.parameters, .template = true },
        .{ .parameters = fx.nothing.parameters, .template = true },
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\",\"enum\":[1,2]}},\"additionalProperties\":false,\"title\":\"T\"}", .template = true },
        // a value the parser would hand over as a string: no type of its own, or several
        .{ .parameters = fx.lookup.parameters, .template = false },
        .{ .parameters = fx.a.parameters, .template = false },
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":[\"string\",\"null\"]}}}", .template = false },
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}}}", .template = false },
        // a reference, which a part of the schema compiled alone would not find
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"array\",\"items\":{\"$ref\":\"#/$defs/n\"}}},\"$defs\":{\"n\":{\"type\":\"null\"}}}", .template = false },
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"$ref\":\"#/properties/b\"},\"b\":{\"type\":\"null\"}}}", .template = false },
        // and one the part would find in itself, where the whole schema was meant
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"array\",\"items\":{\"$ref\":\"#\"}}}}", .template = false },
        // any keys and values: there is no parameter to name
        .{ .parameters = "{\"type\":\"object\"}", .template = false },
        // a string no raw text can be: it holds a closer
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\",\"enum\":[\"x\",\"a</Parameter>\"]}}}", .template = false },
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\",\"const\":\"</tool_call>\"}}}", .template = false },
        // names the parser's `[^>\s]+` would cut
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a b\":{\"type\":\"string\"}}}", .template = false },
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"a>b\":{\"type\":\"string\"}}}", .template = false },
        .{ .parameters = "{\"type\":\"object\",\"properties\":{\"\":{\"type\":\"string\"}}}", .template = false },
        .{ .name = "say \"hi\"\\now", .parameters = fx.add.parameters, .template = false },
        .{ .name = "a>b", .parameters = fx.add.parameters, .template = false },
    }) |case| {
        errdefer std.debug.print("{s}: {s}\n", .{ case.name, case.parameters });
        for ([_]bool{ true, false }) |asked| {
            const constraint = try bench.gate(&.{.{ .name = case.name, .parameters = case.parameters }}, .{ .template_form = asked }, &diagnostic);
            defer constraint.deinit();
            try testing.expectEqual(asked and case.template, tx.gateOf(constraint).tools.items[0].params != null);
        }
    }
}

test "in the template's form a gated reply is calls and nothing else: an offered name, arguments its schema takes" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    var prng: std.Random.DefaultPrng = .init(0x5eed_023);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(gpa);
    // one set of masks for every gate, as a server keeps them, and a gate's own
    var strings: Strings = try .init(gpa, &bench.tokenizer, bench.engine.interface().words, &bench.stops);
    defer strings.deinit();

    const sets = [_][]const Tool{
        &.{tx.sum},
        &.{tx.note},
        &.{ tx.sum, tx.note, fx.nothing },
        // both forms offered together
        &.{ fx.lookup, tx.note, fx.abc, tx.sum, fx.quoted },
    };
    var raw_values: usize = 0;
    var multiline: usize = 0;
    for (sets, 0..) |tools, nth| {
        for ([_]bool{ false, true }) |single| {
            const constraint = try bench.gate(tools, .{ .single = single, .strings = if (nth % 2 == 0) &strings else null }, &diagnostic);
            defer constraint.deinit();
            const called = try arena.alloc(usize, tools.len);
            @memset(called, 0);
            var several: usize = 0;
            var ended: usize = 0;
            for (0..120) |_| {
                if (!try tx.walk(gpa, constraint, bench.engine.vocabulary.width, bench.stops[0], prng.random(), 120, 1200, &ids, &bench.tokenizer)) continue;
                ended += 1;
                reply.clearRetainingCapacity();
                try bench.tokenizer.decode(gpa, ids.items, .{ .skip_special = true }, &reply);
                errdefer std.debug.print("reply: {s}\n", .{reply.items});
                const calls = try fx.read(arena, reply.items, tools);
                try testing.expect(calls.len >= 1);
                if (single) try testing.expectEqual(1, calls.len);
                several += @intFromBool(calls.len > 1);
                for (calls) |call| {
                    const tool = for (tools, 0..) |tool, index| {
                        if (std.mem.eql(u8, tool.name, call.name)) break index;
                    } else return error.TestUnexpectedResult;
                    called[tool] += 1;
                    errdefer std.debug.print("arguments: {s}\n", .{call.arguments});
                    try testing.expect(try fixtures.validates(arena, tools[tool].parameters, call.arguments));
                    if (std.mem.eql(u8, call.name, "note")) {
                        const body = (try std.json.parseFromSliceLeaky(Value, arena, call.arguments, .{})).object.get("body").?.string;
                        raw_values += 1;
                        multiline += @intFromBool(std.mem.indexOfScalar(u8, body, '\n') != null);
                    }
                }
            }
            try testing.expect(ended >= 100);
            for (called) |count| try testing.expect(count > 0);
            try testing.expectEqual(single, several == 0);
        }
    }
    // raw strings were written, some over several lines
    errdefer std.debug.print("{d} raw strings, {d} of several lines\n", .{ raw_values, multiline });
    try testing.expect(raw_values > 100 and multiline > 10);
}

test "the template's form keeps the interface's contract, and its frames are one token a mask" {
    const gpa = testing.allocator;
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    const constraint = try bench.gate(&.{ tx.note, fx.ab, tx.sum }, .{}, &diagnostic);
    defer constraint.deinit();
    const words = constraint.words;
    const width = bench.engine.vocabulary.width;
    const stop = bench.stops[0];
    const masks = try gpa.alloc(u32, 2 * words);
    defer gpa.free(masks);
    const got = masks[0..words];
    const again = masks[words..];

    // Cut where the gate cuts it: a frame, a value, a closer. `forced`: every token of the part is the only
    // one its mask allows.
    const Part = struct { written: []const u8, forced: bool = false };
    const parts = [_]Part{
        .{ .written = "<tool_call>\n<function=note>\n" },
        .{ .written = "<parameter=kind>\n", .forced = true },
        .{ .written = "ab\n</parameter>\n" },
        .{ .written = "<parameter=body>\n", .forced = true },
        // a raw string: quotes, a tag, lines, a closer's beginning, and its end however the tokens fall
        .{ .written = "say \"hi\" <x>\n\nto </param and </tool\n</parameter>\n" },
        .{ .written = "<parameter=tags>\n" },
        .{ .written = "[\"x\", \"y\"]" },
        .{ .written = "\n</parameter>\n", .forced = true },
        .{ .written = "<parameter=n>\n" },
        .{ .written = "null" },
        .{ .written = "\n</parameter>\n", .forced = true },
        .{ .written = "</function>\n</tool_call>" },
        .{ .written = "\n<tool_call>\n<function=add>\n" },
        .{ .written = "<parameter=a>\n", .forced = true },
        .{ .written = "-12" },
        // a number may go on, so its closer's first token is not the only one
        .{ .written = "\n</parameter>\n" },
        .{ .written = "<parameter=b>\n", .forced = true },
        .{ .written = "0" },
        .{ .written = "\n</parameter>\n", .forced = true },
        .{ .written = "</function>\n</tool_call>" },
    };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var forced: std.ArrayList(bool) = .empty;
    defer forced.deinit(gpa);
    var between: usize = 0;
    for (parts, 0..) |part, index| {
        if (index == 12) between = ids.items.len;
        const from = ids.items.len;
        try bench.tokenizer.encode(gpa, part.written, &ids);
        try forced.appendNTimes(gpa, part.forced, ids.items.len - from);
    }

    const kept = try gpa.alloc(grammar.Position, ids.items.len + 1);
    defer gpa.free(kept);
    const hashes = try gpa.alloc(u64, ids.items.len + 1);
    defer gpa.free(hashes);
    for (0..2) |round| {
        var at = try constraint.start();
        for (ids.items, forced.items, 0..) |id, one, step| {
            kept[step] = at;
            try constraint.fill(at, got);
            const hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(got));
            if (round == 1) try testing.expectEqual(hashes[step], hash);
            hashes[step] = hash;
            errdefer std.debug.print("at token {d}, '{s}'\n", .{ step, bench.tokenizer.tokenBytes(id) });
            try testing.expect(grammar.allowed(got, id));
            try testing.expectEqual(step == between, grammar.allowed(got, stop));
            try testing.expect(!constraint.ended(at));
            var count: usize = 0;
            for (got) |word| count += @popCount(word);
            if (one) try testing.expectEqual(1, count);
            // a draft tree from here: every token tried, none followed
            for (0..width + 64) |token| {
                const next = try constraint.accept(at, @intCast(token));
                try testing.expectEqual(token < width and grammar.allowed(got, @intCast(token)), next != null);
            }
            try constraint.fill(at, again);
            try testing.expectEqualSlices(u32, got, again);
            at = (try constraint.accept(at, id)).?;
        }
        try constraint.fill(at, got);
        try testing.expect(grammar.allowed(got, stop));
        const end = (try constraint.accept(at, stop)).?;
        try testing.expect(constraint.ended(end) and !constraint.ended(at));
        for (kept[0..ids.items.len], hashes[0..ids.items.len]) |position, hash| {
            try constraint.fill(position, got);
            try testing.expectEqual(hash, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(got)));
        }
    }
}

test "what a value may not hold: a closer in a raw string or in JSON's, a wrong type, a key out of order" {
    const gpa = testing.allocator;
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    const open = "<tool_call>\n<function=note>\n<parameter=kind>\na\n</parameter>\n<parameter=body>\n";
    const Case = struct { form: bool = true, tool: Tool = tx.note, written: []const u8, taken: bool };
    for ([_]Case{
        .{ .written = open ++ "x\n</parameter>\n</function>\n</tool_call>", .taken = true },
        // an empty string; `|` is where the gate's tokens part, which the tokenizer's would not
        .{ .written = open ++ "|\n</parameter>\n<parameter=n>\nnull\n</parameter>\n</function>\n</tool_call>", .taken = true },
        .{ .written = open ++ "x</PARAMETER", .taken = true },
        .{ .written = open ++ "x</PARAMETER>", .taken = false },
        // the closer without the newline the parser drops would end the value all the same
        .{ .written = open ++ "x</parameter>", .taken = false },
        .{ .written = open ++ "x\n</parameter>x", .taken = false },
        .{ .written = open ++ "x </tool_call", .taken = true },
        .{ .written = open ++ "x </tool_call>", .taken = false },
        .{ .written = open ++ "x\n</parameter>\n<parameter=tags>\n[\"x\"]\n</parameter>\n<parameter=deep>\n{\"s\": \"</tool_call\"}", .taken = true },
        .{ .written = open ++ "x\n</parameter>\n<parameter=deep>\n{\"s\": \"</tool_call>", .taken = false },
        .{ .written = open ++ "x\n</parameter>\n<parameter=deep>\n{\"s\": \"</Parameter>", .taken = false },
        .{ .written = open ++ "x\n</parameter>\n<parameter=deep>\n{\"s\": \"<\\/parameter>\"}\n</parameter>\n", .taken = true },
        // each value by its own property's schema
        .{ .written = open ++ "x\n</parameter>\n<parameter=n>\n1", .taken = false },
        .{ .written = open ++ "x\n</parameter>\n<parameter=tags>\n[\"a", .taken = false },
        .{ .written = open ++ "x\n</parameter>\n<parameter=mode>\noff", .taken = false },
        .{ .written = open ++ "x\n</parameter>\n<parameter=mode>\non\n</parameter>\n</function>\n</tool_call>", .taken = true },
        // the schema's order, the required ones, no other key
        .{ .written = "<tool_call>\n<function=note>\n<parameter=body>\n", .taken = false },
        .{ .written = "<tool_call>\n<function=note>\n<parameter=kind>\nabc\n</parameter>\n</function>", .taken = false },
        .{ .written = open ++ "x\n</parameter>\n<parameter=n>\nnull\n</parameter>\n<parameter=tags>\n", .taken = false },
        .{ .written = open ++ "x\n</parameter>\n<parameter=other>\n", .taken = false },
        .{ .written = "<tool_call>\n<function=note>\n<parameter=kind>\nabcd", .taken = false },
        .{ .written = "<tool_call>\n<function=lookup>\n", .taken = false },
        // the JSON form keeps the call's end out of its strings too
        .{ .form = false, .written = "<tool_call>\n{\"name\": \"note\", \"arguments\":{\"kind\": \"a\", \"body\": \"</tool_call\"}", .taken = true },
        .{ .form = false, .written = "<tool_call>\n{\"name\": \"note\", \"arguments\":{\"kind\": \"a\", \"body\": \"</tool_call>", .taken = false },
    }) |case| {
        const constraint = try bench.gate(&.{case.tool}, .{ .template_form = case.form }, &diagnostic);
        defer constraint.deinit();
        ids.clearRetainingCapacity();
        var parts = std.mem.splitScalar(u8, case.written, '|');
        while (parts.next()) |part| try bench.tokenizer.encode(gpa, part, &ids);
        const mask = try gpa.alloc(u32, constraint.words);
        defer gpa.free(mask);
        var at = try constraint.start();
        const taken = for (ids.items) |id| {
            // the mask says what `accept` does
            try constraint.fill(at, mask);
            const next = try constraint.accept(at, id);
            try testing.expectEqual(grammar.allowed(mask, id), next != null);
            at = next orelse break false;
        } else true;
        errdefer {
            std.debug.print("{s}\n", .{case.written});
            for (ids.items) |id| std.debug.print("[{s}]", .{bench.tokenizer.tokenBytes(id)});
        }
        try testing.expectEqual(case.taken, taken);
    }
}

test "a lead comes before the reply's first call and no other, each of its tokens forced; one that cannot be written is left out" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var bench: fx.Bench = undefined;
    try bench.init();
    defer bench.deinit();
    var diagnostic: grammar.Diagnostic = .{};
    const mask = try gpa.alloc(u32, bench.engine.interface().words);
    defer gpa.free(mask);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);

    const plain = try bench.gate(&.{fx.ab}, .{}, &diagnostic);
    defer plain.deinit();
    const opening = "<tool_call>\n{\"name\": \"ab\", \"arguments\":";
    const rest = "}\n</tool_call>";
    for ([_][]const u8{ "\n\n", "\n", "\n\n\n" }) |lead| {
        const constraint = try bench.gate(&.{fx.ab}, .{ .lead = lead }, &diagnostic);
        defer constraint.deinit();
        // one tool: the lead and the opening are the tokenizer's tokens for them, the only one a mask allows
        ids.clearRetainingCapacity();
        try bench.tokenizer.encode(gpa, try std.mem.concat(arena, u8, &.{ lead, opening }), &ids);
        var at = try constraint.start();
        for (ids.items) |id| {
            try constraint.fill(at, mask);
            var count: usize = 0;
            for (mask) |word| count += @popCount(word);
            try testing.expectEqual(1, count);
            at = (try constraint.accept(at, id)).?;
        }
        // a gate without the lead takes none of it, and this one no call without it
        try testing.expectEqual(null, try plain.accept(try plain.start(), ids.items[0]));
        const first = ids.items[0];
        ids.clearRetainingCapacity();
        try bench.tokenizer.encode(gpa, opening, &ids);
        try testing.expect(first != ids.items[0]);
        try testing.expectEqual(null, try constraint.accept(try constraint.start(), ids.items[0]));

        // the call after the first is written after the text between calls, not after the lead
        for ([_]struct { []const u8, bool }{ .{ "\n", true }, .{ lead, lead.len == 1 }, .{ "", false } }) |second| {
            at = try constraint.start();
            const said = try std.mem.concat(arena, u8, &.{ lead, opening, "{\"x\":2}", rest });
            const more = try std.mem.concat(arena, u8, &.{ second[0], opening, "{\"x\":2}", rest });
            const taken = for ([_][]const u8{ said, more }) |part| {
                ids.clearRetainingCapacity();
                try bench.tokenizer.encode(gpa, part, &ids);
                const whole = for (ids.items) |id| {
                    at = try constraint.accept(at, id) orelse break false;
                } else true;
                if (!whole) break false;
            } else true;
            try testing.expectEqual(second[1], taken);
        }

        // whatever the masks let through is the lead, then calls and nothing else
        var random: std.Random.DefaultPrng = .init(34);
        const several = try bench.gate(&.{ fx.ab, fx.abc, fx.nothing }, .{ .lead = lead, .template_form = false }, &diagnostic);
        defer several.deinit();
        var ended: usize = 0;
        for (0..40) |_| {
            if (!try fx.walk(gpa, several, bench.tokenizer.count(), bench.stops[0], random.random(), 12, 400, &ids, &bench.tokenizer)) continue;
            ended += 1;
            text.clearRetainingCapacity();
            try bench.tokenizer.decode(gpa, ids.items, .{}, &text);
            try testing.expect(std.mem.startsWith(u8, text.items, lead));
            try testing.expect((try fx.calls(arena, text.items[lead.len..])).len != 0);
        }
        try testing.expect(ended > 20);
    }

    // A lead the tokenizer cannot write back (bytes that are not UTF-8, a token that ends the reply) is left
    // out: the reply opens with the call.
    ids.clearRetainingCapacity();
    try bench.tokenizer.encode(gpa, opening, &ids);
    for ([_][]const u8{ "\xff\n", "\n<|end|>\n", "" }) |lead| {
        const constraint = try bench.gate(&.{fx.ab}, .{ .lead = lead }, &diagnostic);
        defer constraint.deinit();
        var at = try constraint.start();
        for (ids.items) |id| at = (try constraint.accept(at, id)).?;
    }
}
