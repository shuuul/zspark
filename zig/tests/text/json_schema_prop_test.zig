//! `json_schema.zig`'s property tests: seeded documents and schemas, judged by `Validator` and `std.json`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const Value = std.json.Value;
const fixtures = @import("testing.zig");
const fx = fixtures.schemas;
const grammar_file = @import("../../src/text/grammar.zig");
const json_schema = @import("../../src/text/json_schema.zig");
const tokenizer_file = @import("../../src/text/tokenizer.zig");
const Diagnostic = json_schema.Diagnostic;
const Engine = json_schema.Engine;
const Tokenizer = json_schema.Tokenizer;
const Validator = @import("json_schema_validator.zig").Validator;
const prop = @import("json_schema_prop_fixture.zig").prop;

test "properties: what the masks take is a valid document, a valid document is taken however it is cut, no position is a dead end, and a mask is its prefix's alone" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    var bench: prop.Bench = undefined;
    try bench.init(gpa, &tokenizer, &.{tokenizer.addedId("<|end|>").?});
    defer bench.deinit();
    const counts = try bench.run(60, 0x5eed_017, false);
    // the test did what it says: replies drawn and ended, documents written and cut, masks compared
    try testing.expectEqual(60 * prop.schemas().len, counts.walks);
    try testing.expect(counts.documents * 10 >= counts.walks * 9);
    try testing.expect(counts.walk_tokens > 20 * counts.walks and counts.document_tokens > 20 * counts.documents);
    try testing.expect(counts.compared > counts.document_tokens / 4);
    // and with a key held to its own tokens: the replies drawn are documents still, none stops short, and a
    // document in the tokenizer's tokens is taken
    const own = try bench.run(60, 0x5eed_035, true);
    try testing.expectEqual(60 * prop.schemas().len, own.walks);
    try testing.expect(own.documents * 10 >= own.walks * 9);
    try testing.expect(own.compared >= own.document_tokens);
}

test "the same properties on the published checkpoint's vocabulary" {
    // Needs ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory, for its tokenizer.json; no GPU and no weights.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var diagnostic: tokenizer_file.Diagnostic = .{};
    var tokenizer = try Tokenizer.load(gpa, testing.io, .cwd(), path, &diagnostic);
    defer tokenizer.deinit(gpa);
    var bench: prop.Bench = undefined;
    try bench.init(gpa, &tokenizer, &.{ tokenizer.addedId("<|im_end|>").?, tokenizer.addedId("<|endoftext|>").? });
    defer bench.deinit();
    const counts = try bench.run(6, 0x5eed_017, false);
    try testing.expectEqual(6 * prop.schemas().len, counts.walks);
    // every unit of a string can be written with this vocabulary, so every document is
    try testing.expectEqual(counts.walks, counts.documents);
    const own = try bench.run(6, 0x5eed_035, true);
    try testing.expectEqual(6 * prop.schemas().len, own.walks);
    try testing.expectEqual(own.walks, own.documents);
}

test "a property's name is taken in the tokenizer's own tokens: one key to come is one token a mask, and the document is the prompt's tokens" {
    // Needs ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory, for its tokenizer.json; no GPU and no weights.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var tokenizer_diagnostic: tokenizer_file.Diagnostic = .{};
    var tokenizer = try Tokenizer.load(gpa, testing.io, .cwd(), path, &tokenizer_diagnostic);
    defer tokenizer.deinit(gpa);
    const stop: u32 = tokenizer.addedId("<|im_end|>").?;
    var engine: Engine = try .init(gpa, &tokenizer, tokenizer.count(), &.{stop});
    defer engine.deinit(gpa);
    const words = engine.vocabulary.words();
    const mask = try gpa.alloc(u32, words);
    defer gpa.free(mask);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var diagnostic: Diagnostic = .{};
    const one = struct {
        fn token(t: *const Tokenizer, allocator: Allocator, text: []const u8) !u32 {
            var out: std.ArrayList(u32) = .empty;
            defer out.deinit(allocator);
            try t.encode(allocator, text, &out);
            try testing.expectEqual(1, out.items.len);
            return out.items[0];
        }
        fn count(bits: []const u32) usize {
            var total: usize = 0;
            for (bits) |word| total += @popCount(word);
            return total;
        }
    };

    // One key to come: after its quote the mask is the name's token alone, however the quote was written,
    // and the place after the name is the machine's own (the quote's token depends on what follows it).
    const answer = "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"],\"additionalProperties\":false}";
    const constraint = try engine.interface().compile(gpa, .{ .json_schema = answer }, &diagnostic);
    defer constraint.deinit();
    const name = try one.token(&tokenizer, gpa, "answer");
    for ([_][]const []const u8{ &.{"{\""}, &.{ "{", "\"" }, &.{ "{", "\n", " \"" } }) |opening| {
        var at = try constraint.start();
        for (opening) |piece| {
            try constraint.fill(at, mask);
            try testing.expect(one.count(mask) > 1);
            at = (try constraint.accept(at, try one.token(&tokenizer, gpa, piece))).?;
        }
        try constraint.fill(at, mask);
        try testing.expectEqual(1, one.count(mask));
        try testing.expect(grammar_file.allowed(mask, name));
        // the name in pieces is not taken, nor its first letters with the name's own token after them
        try testing.expectEqual(null, try constraint.accept(at, try one.token(&tokenizer, gpa, "an")));
        try testing.expectEqual(null, try constraint.accept(at, try one.token(&tokenizer, gpa, "a")));
        at = (try constraint.accept(at, name)).?;
        try constraint.fill(at, mask);
        try testing.expect(grammar_file.allowed(mask, try one.token(&tokenizer, gpa, "\"")));
        try testing.expect(grammar_file.allowed(mask, try one.token(&tokenizer, gpa, "\":")));
    }

    // Several keys to come: each one's first token and nothing else; a name of several tokens goes on with
    // its next; a name that begins inside the quote's token, and one the tokenizer cannot give alike in
    // every surrounding, are the machine's.
    const several = "{\"type\":\"object\",\"properties\":{\"tool_call_id\":{\"type\":\"integer\"},\"why\":{\"type\":\"string\"},\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"]}";
    const optional = try engine.interface().compile(gpa, .{ .json_schema = several }, &diagnostic);
    defer optional.deinit();
    ids.clearRetainingCapacity();
    try tokenizer.encode(gpa, "tool_call_id", &ids);
    try testing.expect(ids.items.len > 1);
    var at = (try optional.accept(try optional.start(), try one.token(&tokenizer, gpa, "{\""))).?;
    try optional.fill(at, mask);
    try testing.expectEqual(3, one.count(mask));
    for ([_]u32{ ids.items[0], name, try one.token(&tokenizer, gpa, "why") }) |id| try testing.expect(grammar_file.allowed(mask, id));
    for (ids.items) |id| {
        try optional.fill(at, mask);
        try testing.expectEqual(grammar_file.allowed(mask, id), true);
        at = (try optional.accept(at, id)).?;
        try optional.fill(at, mask);
        if (id != ids.items[ids.items.len - 1]) try testing.expectEqual(1, one.count(mask));
    }

    // Every document below is taken in the tokens the tokenizer gives its text, which are the tokens the
    // next turn's prompt holds for it: names with marks at their ends, blanks, escapes, another script.
    const odd = "{\"type\":\"object\",\"properties\":{\"$ref\":{\"type\":\"integer\"},\"_id\":{\"type\":\"integer\"},\"a.b\":{\"type\":\"integer\"},\"my key\":{\"type\":\"integer\"}," ++
        "\"caf\\u00e9\":{\"type\":\"integer\"},\"\\u540d\\u524d\":{\"type\":\"integer\"},\"q\\\"q\":{\"type\":\"integer\"},\"x\":{\"type\":\"object\",\"properties\":{\"inner_name\":{\"type\":\"integer\"}}},\"-\":{\"type\":\"integer\"},\"\":{\"type\":\"integer\"}}}";
    const names = try engine.interface().compile(gpa, .{ .json_schema = odd }, &diagnostic);
    defer names.deinit();
    for ([_][]const u8{
        "{\"$ref\":1,\"_id\":2,\"a.b\":3,\"my key\":4,\"caf\u{e9}\":5,\"\u{540d}\u{524d}\":6,\"q\\\"q\":7,\"x\":{\"inner_name\":8},\"-\":9,\"\":10}",
        "{\"$ref\": 1, \"_id\": 2, \"a.b\": 3, \"my key\": 4, \"caf\u{e9}\": 5, \"\u{540d}\u{524d}\": 6, \"q\\\"q\": 7, \"x\": {\"inner_name\": 8}, \"-\": 9, \"\": 10}",
        "{\n  \"$ref\": 1,\n  \"_id\": 2,\n  \"a.b\": 3,\n  \"my key\": 4,\n  \"x\": {\n    \"inner_name\": 8\n  },\n  \"\": 10\n}",
        "{ \"_id\" : 2 , \"x\" : { \"inner_name\" : 8 } , \"-\" : 9 }",
        "{\"my key\":4}",
        "{}",
    }) |document| {
        errdefer std.debug.print("document: {s}\n", .{document});
        ids.clearRetainingCapacity();
        try tokenizer.encode(gpa, document, &ids);
        at = try names.start();
        for (ids.items) |id| {
            try names.fill(at, mask);
            try testing.expect(one.count(mask) > 0);
            try testing.expect(grammar_file.allowed(mask, id));
            at = (try names.accept(at, id)).?;
        }
        try names.fill(at, mask);
        try testing.expect(grammar_file.allowed(mask, stop));
    }

    // Where the tokenizer has no one token for a place, the place is the machine's. A name's last mark goes
    // with the closing quote before a blank (`$"`) and alone before a colon (`$`, then `":`); and `="` is one
    // token everywhere, which takes the quote, whose own token depends on what follows.
    for ([_][2][]const u8{ .{ "a$", "$\"" }, .{ "a=", "=\"" } }) |case| {
        const text = try std.fmt.allocPrint(gpa, "{{\"type\":\"object\",\"properties\":{{\"{s}\":{{\"type\":\"integer\"}}}},\"required\":[\"{s}\"]}}", .{ case[0], case[0] });
        defer gpa.free(text);
        const marked = try engine.interface().compile(gpa, .{ .json_schema = text }, &diagnostic);
        defer marked.deinit();
        at = (try marked.accept(try marked.start(), try one.token(&tokenizer, gpa, "{\""))).?;
        try marked.fill(at, mask);
        try testing.expectEqual(1, one.count(mask));
        at = (try marked.accept(at, try one.token(&tokenizer, gpa, "a"))).?;
        try marked.fill(at, mask);
        try testing.expect(grammar_file.allowed(mask, try one.token(&tokenizer, gpa, case[0][1..])));
        try testing.expect(grammar_file.allowed(mask, try one.token(&tokenizer, gpa, case[1])));
        for (0..3) |which| {
            const document = switch (which) {
                0 => try std.fmt.allocPrint(gpa, "{{\"{s}\":1}}", .{case[0]}),
                1 => try std.fmt.allocPrint(gpa, "{{ \"{s}\" : 1 }}", .{case[0]}),
                else => try std.fmt.allocPrint(gpa, "{{\n\"{s}\"\n: 1}}", .{case[0]}),
            };
            defer gpa.free(document);
            ids.clearRetainingCapacity();
            try tokenizer.encode(gpa, document, &ids);
            at = try marked.start();
            for (ids.items) |id| at = (try marked.accept(at, id)).?;
        }
    }
}

test "a schema changed at random compiles or is refused, and what it compiles to still takes only valid documents" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    var bench: prop.Bench = undefined;
    try bench.init(gpa, &tokenizer, &.{tokenizer.addedId("<|end|>").?});
    defer bench.deinit();
    bench.loud = false;
    bench.limit = 400;
    var prng: std.Random.DefaultPrng = .init(0xbad_5c4e3a);
    const random = prng.random();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    // What is put into a schema: keywords with values of the form they have, and of forms they do not.
    const Put = struct { name: []const u8, value: []const u8 };
    const puts = [_]Put{
        .{ .name = "type", .value = "\"object\"" },                                    .{ .name = "type", .value = "\"array\"" },
        .{ .name = "type", .value = "\"string\"" },                                    .{ .name = "type", .value = "[\"integer\",\"null\"]" },
        .{ .name = "type", .value = "5" },                                             .{ .name = "type", .value = "\"nothing\"" },
        .{ .name = "required", .value = "[\"a\"]" },                                   .{ .name = "required", .value = "[]" },
        .{ .name = "required", .value = "\"a\"" },                                     .{ .name = "minItems", .value = "2" },
        .{ .name = "minItems", .value = "-1" },                                        .{ .name = "maxItems", .value = "1" },
        .{ .name = "maxItems", .value = "0" },                                         .{ .name = "enum", .value = "[1,\"a\",null]" },
        .{ .name = "enum", .value = "[]" },                                            .{ .name = "const", .value = "null" },
        .{ .name = "const", .value = "{\"a\":[1]}" },                                  .{ .name = "items", .value = "{}" },
        .{ .name = "items", .value = "{\"type\":\"boolean\"}" },                       .{ .name = "items", .value = "false" },
        .{ .name = "properties", .value = "{}" },                                      .{ .name = "properties", .value = "{\"a\":{\"type\":\"null\"}}" },
        .{ .name = "additionalProperties", .value = "false" },                         .{ .name = "additionalProperties", .value = "{\"type\":\"integer\"}" },
        .{ .name = "additionalProperties", .value = "true" },                          .{ .name = "$ref", .value = "\"#\"" },
        .{ .name = "$ref", .value = "\"#/properties/a\"" },                            .{ .name = "$ref", .value = "\"#/$defs/Kind\"" },
        .{ .name = "anyOf", .value = "[{\"type\":\"null\"},{\"type\":\"boolean\"}]" }, .{ .name = "anyOf", .value = "[{},{}]" },
        .{ .name = "oneOf", .value = "[{\"type\":\"string\"}]" },                      .{ .name = "pattern", .value = "\"a\"" },
        .{ .name = "description", .value = "\"x\"" },                                  .{ .name = "title", .value = "1" },
    };
    const junk = [_][]const u8{ "null", "true", "5", "\"a\"", "[]", "{}", "[{}]", "{\"type\":\"string\"}", "[\"a\"]", "1.5" };
    const Objects = struct {
        fn collect(arena: Allocator, value: *Value, out: *std.ArrayList(*std.json.ObjectMap)) Allocator.Error!void {
            switch (value.*) {
                .object => |*object| {
                    try out.append(arena, object);
                    for (object.values()) |*under| try collect(arena, under, out);
                },
                .array => |*array| for (array.items) |*under| try collect(arena, under, out),
                else => {},
            }
        }
    };
    var compiled: usize = 0;
    var refused: usize = 0;
    var ended: usize = 0;
    const all = prop.schemas();
    for (0..4000) |_| {
        const arena = arena_state.allocator();
        // one to three changes to the schema's JSON: a key gone, a keyword put, a value replaced
        var changed = try std.json.parseFromSliceLeaky(Value, arena, all[random.uintLessThan(usize, all.len)], .{});
        for (0..1 + random.uintLessThan(usize, 3)) |_| {
            var objects: std.ArrayList(*std.json.ObjectMap) = .empty;
            try Objects.collect(arena, &changed, &objects);
            const object = objects.items[random.uintLessThan(usize, objects.items.len)];
            switch (random.uintLessThan(u32, 4)) {
                0 => if (object.count() > 0) object.orderedRemoveAt(random.uintLessThan(usize, object.count())),
                1 => if (object.count() > 0) {
                    object.values()[random.uintLessThan(usize, object.count())] = try std.json.parseFromSliceLeaky(Value, arena, junk[random.uintLessThan(usize, junk.len)], .{});
                },
                else => {
                    const put = puts[random.uintLessThan(usize, puts.len)];
                    try object.put(arena, put.name, try std.json.parseFromSliceLeaky(Value, arena, put.value, .{}));
                },
            }
        }
        text.clearRetainingCapacity();
        try text.print(gpa, "{f}", .{std.json.fmt(changed, .{})});
        // and now and then a change to its bytes: cut short, a byte replaced, a run doubled
        if (random.uintLessThan(u32, 5) == 0) {
            const at = random.uintLessThan(usize, text.items.len);
            switch (random.uintLessThan(u32, 3)) {
                0 => text.shrinkRetainingCapacity(at),
                1 => text.items[at] = random.int(u8),
                else => {
                    const copy = try arena.dupe(u8, text.items[at..@min(text.items.len, at + random.uintLessThan(usize, 40))]);
                    try text.insertSlice(gpa, at, copy);
                },
            }
        }
        errdefer std.debug.print("schema: {s}\n", .{text.items});
        var diagnostic: Diagnostic = .{};
        const constraint = bench.engine.interface().compile(gpa, .{ .json_schema = text.items }, &diagnostic) catch |err| {
            try testing.expect(err == error.InvalidSchema or err == error.UnsupportedSchema);
            try testing.expect(diagnostic.message().len > 0);
            refused += 1;
            continue;
        };
        defer constraint.deinit();
        compiled += 1;
        const schema = try std.json.parseFromSliceLeaky(Value, arena, text.items, .{});
        // A changed literal may hold a byte the tiny tokenizer has no token for, and a changed schema may ask
        // for a document without end: those walks stop where they stand. Every other one is held to the
        // properties at each step and ends valid.
        if (bench.walk(constraint, schema, random, 0.2, 30)) |_| {
            ended += 1;
        } else |err| switch (err) {
            error.WalkDidNotEnd => {},
            error.TestUnexpectedResult => if (bench.count(bench.mask) != 0 or grammar_file.allowed(bench.mask, bench.stops[0])) return err,
            else => return err,
        }
        _ = arena_state.reset(.retain_capacity);
    }
    // both happened often, and most of what compiled was walked to its end
    try testing.expect(compiled > 400 and refused > 400);
    try testing.expect(ended * 10 > compiled * 8);
}

test "a long reply adds two frames a byte at most, and past the bound the constraint fails instead of growing" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    const stop = tokenizer.addedId("<|end|>").?;
    var bench: prop.Bench = undefined;
    try bench.init(gpa, &tokenizer, &.{stop});
    defer bench.deinit();
    var prng: std.Random.DefaultPrng = .init(7);
    const random = prng.random();

    // 20000 items of an array without bounds, then of one whose bound is past them
    var document: std.ArrayList(u8) = .empty;
    defer document.deinit(gpa);
    try document.appendSlice(gpa, "[");
    for (0..20000) |at| try document.print(gpa, "{s}{{\"a\": {d}, \"b\": \"t\\n{d}\"}}", .{ if (at == 0) "" else ", ", at, at % 7 });
    try document.appendSlice(gpa, "]");
    try testing.expect(try bench.tokens.split(gpa, document.items, .longest, random, &bench.ids));
    const ids = try gpa.dupe(u32, bench.ids.items);
    defer gpa.free(ids);

    for ([_][]const u8{
        "{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"},\"b\":{\"type\":\"string\"}},\"required\":[\"a\",\"b\"]}}",
        "{\"type\":\"array\",\"maxItems\":30000,\"items\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"},\"b\":{\"type\":\"string\"}},\"required\":[\"a\",\"b\"]}}",
        "{\"type\":\"array\"}",
    }, [_]bool{ false, true, false }) |schema, counted| {
        var diagnostic: Diagnostic = .{};
        const constraint = try bench.engine.interface().compile(gpa, .{ .json_schema = schema }, &diagnostic);
        defer constraint.deinit();
        const made: *Engine.Compiled = @ptrCast(@alignCast(constraint.context));
        // twice: the second reply starts from the first's frames gone
        for (0..2) |_| {
            var at = try constraint.start();
            try testing.expectEqual(1, made.matcher.machine.frames.items.len);
            var walked: usize = 0;
            for (ids, 0..) |id, step| {
                const before = made.matcher.machine.frames.items.len;
                try constraint.fill(at, bench.mask);
                // a mask adds no frame
                try testing.expectEqual(before, made.matcher.machine.frames.items.len);
                try testing.expect(grammar_file.allowed(bench.mask, id));
                // a draft that is not kept, now and then: its frames stay, and are counted
                if (step % 5 == 0) {
                    const other = ids[random.uintLessThan(usize, ids.len)];
                    _ = try constraint.accept(at, other);
                    walked += tokenizer.tokenBytes(other).len;
                }
                at = (try constraint.accept(at, id)).?;
                walked += tokenizer.tokenBytes(id).len;
                try testing.expect(made.matcher.machine.frames.items.len <= 1 + 2 * walked);
            }
            try testing.expect(constraint.ended((try constraint.accept(at, stop)).?));
            try testing.expect(walked > document.items.len);
            // What the masks keep: an array without bounds is one position however long it grows; one with a
            // bound keeps a few masks an item, each a handful of tokens.
            const cache = made.grammar.cache.count();
            if (counted) try testing.expect(cache > 20000 and cache <= 4 * 20000 + 64) else try testing.expect(cache <= 64);
        }
    }

    // past the bound: the constraint fails where the frames pass it, and stays failed until the next reply
    bench.engine.options.max_frames = 5000;
    var diagnostic: Diagnostic = .{};
    const constraint = try bench.engine.interface().compile(gpa, .{ .json_schema = "{\"type\":\"array\"}" }, &diagnostic);
    defer constraint.deinit();
    const made: *Engine.Compiled = @ptrCast(@alignCast(constraint.context));
    for (0..2) |_| {
        var at = try constraint.start();
        const failed_at = for (ids, 0..) |id, step| {
            try testing.expect(made.matcher.machine.frames.items.len <= 5000 + 2 * bench.engine.vocabulary.longest + 2);
            if (constraint.fill(at, bench.mask)) |_| {} else |err| {
                try testing.expectEqual(error.ConstraintFailed, err);
                break step;
            }
            at = (try constraint.accept(at, id)).?;
        } else return error.TestExpectedError;
        try testing.expect(failed_at > 100 and failed_at < ids.len);
        try testing.expect(made.matcher.machine.frames.items.len > 5000);
        try testing.expectError(error.ConstraintFailed, constraint.accept(at, ids[failed_at]));
        try testing.expectError(error.ConstraintFailed, constraint.fill(at, bench.mask));
    }
}
