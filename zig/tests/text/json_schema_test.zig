//! `json_schema.zig`'s tests on the fixture's schemas: the masks token by token, the interface, the refusals
//! and the bounds.

const std = @import("std");
const fixtures = @import("testing.zig");
const fx = fixtures.schemas;
const grammar_file = @import("../../src/text/grammar.zig");
const json_builder = @import("../../src/text/json_builder.zig");
const json_schema = @import("../../src/text/json_schema.zig");
const tokenizer_file = @import("../../src/text/tokenizer.zig");
const annotations = json_builder.annotations;
const definitions = json_builder.definitions;
const keywords = json_builder.keywords;
const Diagnostic = json_schema.Diagnostic;
const Engine = json_schema.Engine;
const Grammar = json_schema.Grammar;
const Matcher = json_schema.Matcher;
const Options = json_schema.Options;
const SchemaError = json_schema.SchemaError;
const Tokenizer = json_schema.Tokenizer;
const Vocabulary = json_schema.Vocabulary;
const allowed = json_schema.allowed;
const max_depth = json_schema.max_depth;

// -- tests -------------------------------------------------------------------------------------------------------

const testing = std.testing;

test "documents of each schema, token by token: the mask takes them, equals the uncached mask, and ends at a stop" {
    var tokenizer = try fx.tiny(testing.allocator);
    defer tokenizer.deinit(testing.allocator);
    // the tiny tokenizer has no added token to stop at: an ordinary token stands for one
    try fx.check(&tokenizer, &.{@intCast(tokenizer.count() - 1)}, 1);
}

test "the same on the published checkpoint's vocabulary" {
    // Needs ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory, for its tokenizer.json; no GPU and no weights.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    var diagnostic: tokenizer_file.Diagnostic = .{};
    var tokenizer = try Tokenizer.load(testing.allocator, testing.io, .cwd(), path, &diagnostic);
    defer tokenizer.deinit(testing.allocator);
    try fx.check(&tokenizer, &.{ tokenizer.addedId("<|im_end|>").?, tokenizer.addedId("<|endoftext|>").? }, 7);
}

test "special tokens, stops and ids past the tokenizer are never in a mask before the end" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    const stop: u32 = @intCast(tokenizer.count() - 1);
    var vocabulary: Vocabulary = try .init(gpa, &tokenizer, tokenizer.count() + 40, &.{stop});
    defer vocabulary.deinit(gpa);
    var diagnostic: Diagnostic = .{};
    var grammar: Grammar = try .init(gpa, &vocabulary, fx.anything, .{}, &diagnostic);
    defer grammar.deinit();
    var matcher: Matcher = try .init(&grammar);
    defer matcher.deinit();
    const mask = try gpa.alloc(u32, vocabulary.words());
    defer gpa.free(mask);
    try testing.expect(try matcher.acceptBytes("{\"key\": \"inside a string almost anything goes"));
    try matcher.fill(mask);
    var count: usize = 0;
    for (0..vocabulary.width) |index| {
        const id: u32 = @intCast(index);
        if (!allowed(mask, id)) continue;
        count += 1;
        try testing.expect(id < tokenizer.count() and id != stop and !tokenizer.isSpecial(id));
    }
    try testing.expect(count > vocabulary.order.len / 2);
    // a string's inside is the same in every schema: kept with the vocabulary
    try testing.expectEqual(0, grammar.cache.count());
    try testing.expectEqual(1, vocabulary.common.count());
    try testing.expect(grammar.cacheBytes().own == 0 and grammar.cacheBytes().common > 0);
}

test "what a schema says that is not read is refused by name" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    var vocabulary: Vocabulary = try .init(gpa, &tokenizer, tokenizer.count(), &.{0});
    defer vocabulary.deinit(gpa);
    const Refusal = struct { schema: []const u8, err: SchemaError, message: []const u8 };
    const refusals = [_]Refusal{
        .{ .schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"$ref\":\"#/$defs/a\"}}}", .err = error.InvalidSchema, .message = "#/a: `$ref` names `#/$defs/a`, which the schema does not have" },
        .{ .schema = "{\"$ref\":\"https://example.com/a.json\"}", .err = error.UnsupportedSchema, .message = "#: `$ref` to `https://example.com/a.json` is not read: only a place in the schema itself (`#/...`)" },
        .{ .schema = "{\"$ref\":\"#/$defs/a\",\"type\":\"object\",\"$defs\":{\"a\":{}}}", .err = error.UnsupportedSchema, .message = "#: `type` beside `$ref` is not read" },
        .{ .schema = "{\"$ref\":\"#/$defs/a\",\"$defs\":{\"a\":{\"$ref\":\"#/$defs/b\"},\"b\":{\"$ref\":\"#/$defs/a\"}}}", .err = error.InvalidSchema, .message = "a `$ref` leads only to itself" },
        .{ .schema = "{\"$defs\":{\"a\":{\"anyOf\":[{\"$ref\":\"#/$defs/a\"},{\"type\":\"null\"}]}},\"type\":\"array\",\"items\":{\"$ref\":\"#/$defs/a\"}}", .err = error.UnsupportedSchema, .message = "#/$defs/a: `anyOf` has itself as an alternative" },
        .{ .schema = "{\"anyOf\":[{\"type\":\"string\"},{\"enum\":[\"a\",1]}]}", .err = error.UnsupportedSchema, .message = "#: `anyOf` has alternatives that both begin with `\"`: they are told apart by their first byte alone" },
        .{ .schema = "{\"type\":\"object\",\"properties\":{\"u\":{\"oneOf\":[{\"type\":\"object\",\"properties\":{\"a\":{}}},{\"type\":\"object\",\"properties\":{\"b\":{}}}]}}}", .err = error.UnsupportedSchema, .message = "#/u: `oneOf` has alternatives that both begin with `{`: they are told apart by their first byte alone" },
        .{ .schema = "{\"oneOf\":[{\"type\":\"integer\"},{\"type\":\"number\"}]}", .err = error.UnsupportedSchema, .message = "#: `oneOf` has alternatives that both begin with `-`: they are told apart by their first byte alone" },
        .{ .schema = "{\"anyOf\":[{\"type\":\"string\"}],\"type\":\"string\"}", .err = error.UnsupportedSchema, .message = "#: `type` beside `anyOf` is not read" },
        .{ .schema = "{\"anyOf\":[]}", .err = error.InvalidSchema, .message = "#: `anyOf` is a list of at least one schema" },
        .{ .schema = "{\"allOf\":[{\"type\":\"string\"}]}", .err = error.UnsupportedSchema, .message = "#: `allOf` is not read" },
        .{ .schema = "{\"type\":\"string\",\"pattern\":\"^a+$\"}", .err = error.UnsupportedSchema, .message = "#: `pattern` is not read" },
        .{ .schema = "{\"type\":\"array\",\"items\":{\"type\":\"string\",\"format\":\"date\"}}", .err = error.UnsupportedSchema, .message = "#/items: `format` is not read" },
        .{ .schema = "{\"type\":\"integer\",\"minimum\":0}", .err = error.UnsupportedSchema, .message = "#: `minimum` is not read" },
        .{ .schema = "{\"type\":\"object\",\"properties\":{\"a\":{}},\"additionalProperties\":true}", .err = error.UnsupportedSchema, .message = "#: `additionalProperties` beside `properties` is not read" },
        .{ .schema = "{\"type\":\"object\",\"properties\":{\"a\":{}},\"required\":[\"b\"]}", .err = error.UnsupportedSchema, .message = "#: `required` names `b`, which `properties` does not" },
        .{ .schema = "{\"type\":\"tuple\"}", .err = error.InvalidSchema, .message = "#: `tuple` is not a type" },
        .{ .schema = "{\"type\":", .err = error.InvalidSchema, .message = "the schema is not JSON: UnexpectedEndOfInput" },
        .{ .schema = "[]", .err = error.InvalidSchema, .message = "#: a schema is an object, true or false" },
        .{ .schema = "false", .err = error.UnsupportedSchema, .message = "#: the schema `false` takes no value" },
    };
    for (refusals) |refusal| {
        var diagnostic: Diagnostic = .{};
        try testing.expectError(refusal.err, Grammar.init(gpa, &vocabulary, refusal.schema, .{}, &diagnostic));
        try testing.expectEqualStrings(refusal.message, diagnostic.message());
    }
    // a definition nothing names is not read, whatever it says
    var diagnostic: Diagnostic = .{};
    var grammar: Grammar = try .init(gpa, &vocabulary, "{\"type\":[\"string\",\"null\"],\"$defs\":{\"unused\":{\"pattern\":\"x\"}}}", .{}, &diagnostic);
    grammar.deinit();
}

test "through the interface: positions are values, a mask is its position's alone, and a reply follows a reply" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    const stop: u32 = tokenizer.addedId("<|end|>").?;
    var engine: Engine = try .init(gpa, &tokenizer, tokenizer.count() + 29, &.{stop});
    defer engine.deinit(gpa);
    // the masks are held to the matcher's own here; a key's own tokens have their test below
    engine.options.key_tokens = false;
    const interface = engine.interface();
    var diagnostic: Diagnostic = .{};
    try testing.expectError(error.UnsupportedSchema, interface.compile(gpa, .{ .json_schema = "{\"type\":\"string\",\"pattern\":\"a\"}" }, &diagnostic));
    try testing.expectEqualStrings("#: `pattern` is not read", diagnostic.message());

    const words = interface.words;
    const masks = try gpa.alloc(u32, 4 * words);
    defer gpa.free(masks);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);

    const Case = struct { spec: grammar_file.Spec, document: []const u8 };
    for ([_]Case{
        .{ .spec = .json_object, .document = "{\"a\": [1, {\"b\": null}], \"c\": \"d\"}" },
        .{ .spec = .{ .json_schema = fx.pydantic }, .document = "{\"name\":\"Ada\",\"age\":36,\"address\":{\"street\":\"x\",\"post\":null},\"kind\":\"work\",\"previous\":null}" },
        .{ .spec = .{ .json_schema = fx.tree }, .document = "{\"value\": 1, \"children\": [{\"value\": 2}], \"next\": null}" },
    }) |case| {
        const constraint = try interface.compile(gpa, case.spec, &diagnostic);
        defer constraint.deinit();
        try testing.expectEqual(words, constraint.words);
        ids.clearRetainingCapacity();
        try tokenizer.encode(gpa, case.document, &ids);

        // the same schema followed by the matcher's own stack, token by token: what each position's mask must be
        var plain: Grammar = try .init(gpa, &engine.vocabulary, switch (case.spec) {
            .json_object => fx.anything,
            .json_schema => |schema| schema,
        }, .{}, &diagnostic);
        defer plain.deinit();

        // twice: a constraint serves one reply after another
        for (0..2) |_| {
            var stack: Matcher = try .init(&plain);
            defer stack.deinit();
            var at = try constraint.start();
            // every position on the way, kept: each is asked again at the end
            var kept: [256]grammar_file.Position = undefined;
            const expected = masks[0..words];
            const got = masks[words..][0..words];
            for (ids.items, 0..) |id, step| {
                kept[step] = at;
                try stack.fill(expected);
                try constraint.fill(at, got);
                try testing.expectEqualSlices(u32, expected, got);
                try testing.expect(!constraint.ended(at));

                // a draft tree from here: every token of the vocabulary tried, none of them followed. Each is
                // taken exactly where its bit is set, and the mask here is afterwards what it was.
                var taken: usize = 0;
                for (0..engine.vocabulary.width) |token| {
                    const next = try constraint.accept(at, @intCast(token));
                    try testing.expectEqual(grammar_file.allowed(got, @intCast(token)), next != null);
                    taken += @intFromBool(next != null);
                }
                try testing.expect(taken > 0);
                try constraint.fill(at, masks[2 * words ..][0..words]);
                try testing.expectEqualSlices(u32, expected, masks[2 * words ..][0..words]);

                try testing.expect(try stack.accept(id));
                at = (try constraint.accept(at, id)).?;
            }
            // the document is whole: a stop and nothing else, and after the stop nothing
            try constraint.fill(at, got);
            var count: usize = 0;
            for (got) |word| count += @popCount(word);
            try testing.expectEqual(1, count);
            const end = (try constraint.accept(at, stop)).?;
            try testing.expect(constraint.ended(end) and !constraint.ended(at));
            try constraint.fill(end, got);
            try testing.expect(std.mem.allEqual(u32, got, 0));
            try testing.expectEqual(null, try constraint.accept(end, stop));

            // the positions passed on the way still give their masks, after every token tried or taken since
            var replay: Matcher = try .init(&plain);
            defer replay.deinit();
            for (ids.items, 0..) |id, step| {
                try replay.fill(expected);
                try constraint.fill(kept[step], got);
                try testing.expectEqualSlices(u32, expected, got);
                try testing.expect(try replay.accept(id));
            }
        }
    }
}

// -- refusals and bounds -----------------------------------------------------------------------------------------

test "every keyword of JSON Schema that is not read is refused by its name, wherever it stands" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    var vocabulary: Vocabulary = try .init(gpa, &tokenizer, tokenizer.count(), &.{0});
    defer vocabulary.deinit(gpa);
    // Drafts 4 to 2020-12 and OpenAPI's additions, each with a value of the form it has; and a word no draft has.
    const Keyword = struct { name: []const u8, value: []const u8 };
    const unread = [_]Keyword{
        .{ .name = "allOf", .value = "[{\"type\":\"string\"}]" },
        .{ .name = "not", .value = "{\"type\":\"null\"}" },
        .{ .name = "if", .value = "{\"type\":\"string\"}" },
        .{ .name = "then", .value = "{\"type\":\"string\"}" },
        .{ .name = "else", .value = "{\"type\":\"string\"}" },
        .{ .name = "dependentSchemas", .value = "{\"a\":{}}" },
        .{ .name = "dependentRequired", .value = "{\"a\":[\"b\"]}" },
        .{ .name = "dependencies", .value = "{\"a\":[\"b\"]}" },
        .{ .name = "prefixItems", .value = "[{\"type\":\"string\"}]" },
        .{ .name = "additionalItems", .value = "false" },
        .{ .name = "contains", .value = "{\"type\":\"string\"}" },
        .{ .name = "minContains", .value = "1" },
        .{ .name = "maxContains", .value = "2" },
        .{ .name = "uniqueItems", .value = "true" },
        .{ .name = "unevaluatedItems", .value = "false" },
        .{ .name = "unevaluatedProperties", .value = "false" },
        .{ .name = "patternProperties", .value = "{\"^a\":{}}" },
        .{ .name = "propertyNames", .value = "{\"type\":\"string\"}" },
        .{ .name = "minProperties", .value = "1" },
        .{ .name = "maxProperties", .value = "2" },
        .{ .name = "pattern", .value = "\"^a+$\"" },
        .{ .name = "format", .value = "\"date-time\"" },
        .{ .name = "minLength", .value = "1" },
        .{ .name = "maxLength", .value = "8" },
        .{ .name = "minimum", .value = "0" },
        .{ .name = "maximum", .value = "10" },
        .{ .name = "exclusiveMinimum", .value = "0" },
        .{ .name = "exclusiveMaximum", .value = "10" },
        .{ .name = "multipleOf", .value = "2" },
        .{ .name = "contentEncoding", .value = "\"base64\"" },
        .{ .name = "contentMediaType", .value = "\"image/png\"" },
        .{ .name = "contentSchema", .value = "{}" },
        .{ .name = "$anchor", .value = "\"a\"" },
        .{ .name = "$dynamicRef", .value = "\"#a\"" },
        .{ .name = "$dynamicAnchor", .value = "\"a\"" },
        .{ .name = "$recursiveRef", .value = "\"#\"" },
        .{ .name = "$recursiveAnchor", .value = "true" },
        .{ .name = "$vocabulary", .value = "{}" },
        .{ .name = "nullable", .value = "true" },
        .{ .name = "discriminator", .value = "{\"propertyName\":\"kind\"}" },
        .{ .name = "x-order", .value = "3" },
        .{ .name = "Type", .value = "\"string\"" },
    };
    // Where a schema stands: `K` is the keyword and its value, the path what the refusal names.
    const Place = struct { before: []const u8, after: []const u8, path: []const u8 };
    const places = [_]Place{
        .{ .before = "{", .after = "}", .path = "#" },
        .{ .before = "{\"type\":\"string\",", .after = "}", .path = "#" },
        .{ .before = "{\"type\":\"object\",\"properties\":{\"p\":{\"type\":\"integer\",", .after = "}}}", .path = "#/p" },
        .{ .before = "{\"type\":\"array\",\"items\":{", .after = "}}", .path = "#/items" },
        .{ .before = "{\"type\":\"object\",\"additionalProperties\":{", .after = "}}", .path = "#/additionalProperties" },
        .{ .before = "{\"anyOf\":[{\"type\":\"null\"},{\"type\":\"string\",", .after = "}]}", .path = "#/anyOf/1" },
        .{ .before = "{\"oneOf\":[{", .after = "}]}", .path = "#/oneOf/0" },
        .{ .before = "{\"$defs\":{\"d\":{", .after = "}},\"type\":\"array\",\"items\":{\"$ref\":\"#/$defs/d\"}}", .path = "#/$defs/d" },
        .{ .before = "{\"$ref\":\"#/$defs/d\",\"$defs\":{\"d\":{}},", .after = "}", .path = "#" },
        .{ .before = "{\"enum\":[1],", .after = "}", .path = "#" },
        .{ .before = "{\"type\":[\"object\",\"null\"],\"properties\":{\"a\":{}},", .after = "}", .path = "#" },
    };
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(gpa);
    for (unread) |keyword| {
        // the validator's own list must not know the keyword either, or it would not be an independent judge
        for (keywords ++ annotations ++ definitions) |known| try testing.expect(!std.mem.eql(u8, known, keyword.name));
        for (places) |place| {
            text.clearRetainingCapacity();
            try text.print(gpa, "{s}\"{s}\":{s}{s}", .{ place.before, keyword.name, keyword.value, place.after });
            expected.clearRetainingCapacity();
            try expected.print(gpa, "{s}: `{s}`", .{ place.path, keyword.name });
            errdefer std.debug.print("schema: {s}\n", .{text.items});
            var diagnostic: Diagnostic = .{};
            try testing.expectError(error.UnsupportedSchema, Grammar.init(gpa, &vocabulary, text.items, .{}, &diagnostic));
            // "... is not read", or "... beside `$ref` is not read"
            try testing.expect(std.mem.startsWith(u8, diagnostic.message(), expected.items));
            try testing.expect(std.mem.endsWith(u8, diagnostic.message(), "is not read"));
        }
    }

    // Keywords that are read, where what they ask could not be held: refused too, not dropped.
    const Refusal = struct { schema: []const u8, err: SchemaError, message: []const u8 };
    for ([_]Refusal{
        .{ .schema = "{\"required\":[\"a\"]}", .err = error.UnsupportedSchema, .message = "#: `required` names keys that `properties` does not" },
        .{ .schema = "{\"minItems\":2,\"properties\":{\"a\":{}}}", .err = error.UnsupportedSchema, .message = "#: `minItems` beside `properties` without a `type` is not read" },
        .{ .schema = "{\"items\":{},\"additionalProperties\":false}", .err = error.UnsupportedSchema, .message = "#: `items` beside `additionalProperties` without a `type` is not read" },
        .{ .schema = "{\"enum\":[\"a\"],\"items\":{}}", .err = error.UnsupportedSchema, .message = "#: `items` beside `enum` is not read" },
        .{ .schema = "{\"const\":1,\"enum\":[1]}", .err = error.UnsupportedSchema, .message = "#: `const` beside `enum` is not read" },
        .{ .schema = "{\"const\":{\"a\":1},\"properties\":{\"a\":{}}}", .err = error.UnsupportedSchema, .message = "#: `properties` beside `const` is not read" },
        .{ .schema = "{\"type\":\"string\",\"enum\":[\"a\",1]}", .err = error.UnsupportedSchema, .message = "#: `enum` has a value its `type` does not take" },
        .{ .schema = "{\"type\":[\"integer\",\"null\"],\"const\":1.5}", .err = error.UnsupportedSchema, .message = "#: `const` has a value its `type` does not take" },
        .{ .schema = "{\"type\":\"letter\",\"enum\":[\"a\"]}", .err = error.InvalidSchema, .message = "#: `letter` is not a type" },
        .{ .schema = "{\"oneOf\":[{\"type\":\"array\"},{\"additionalProperties\":false}]}", .err = error.UnsupportedSchema, .message = "#: `oneOf` has an alternative without a `type`, which values of other kinds pass too" },
        .{ .schema = "{\"type\":\"array\",\"items\":{\"oneOf\":[{\"type\":\"null\"},{\"anyOf\":[{\"$ref\":\"#/$defs/a\"},{\"type\":\"string\"}]}]},\"$defs\":{\"a\":{\"minItems\":1}}}", .err = error.UnsupportedSchema, .message = "#/items: `oneOf` has an alternative without a `type`, which values of other kinds pass too" },
    }) |refusal| {
        var diagnostic: Diagnostic = .{};
        try testing.expectError(refusal.err, Grammar.init(gpa, &vocabulary, refusal.schema, .{}, &diagnostic));
        try testing.expectEqualStrings(refusal.message, diagnostic.message());
    }
    // and without a `type` the keywords still bind: two items at least, not any value
    var diagnostic: Diagnostic = .{};
    var grammar: Grammar = try .init(gpa, &vocabulary, "{\"minItems\":2}", .{}, &diagnostic);
    defer grammar.deinit();
    var matcher: Matcher = try .init(&grammar);
    defer matcher.deinit();
    try testing.expect(!try matcher.acceptBytes("1"));
    try testing.expect(try matcher.acceptBytes("[1"));
    try testing.expect(!try matcher.acceptBytes("]"));
    try testing.expect(try matcher.acceptBytes(",{}]"));
    try testing.expect(matcher.complete());
}

test "a schema that is malformed, too deep or too large is an error with its bound, never a crash" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    var vocabulary: Vocabulary = try .init(gpa, &tokenizer, tokenizer.count(), &.{0});
    defer vocabulary.deinit(gpa);

    // each is an error of one kind or the other, with words
    for ([_][]const u8{
        "",                                                          " ",                                              "nul",
        "5",                                                         "null",                                           "\"string\"",
        "{\"type\":5}",                                              "{\"type\":[]}",                                  "{\"type\":[1]}",
        "{\"type\":[\"string\",\"str\"]}",                           "{\"type\":null}",                                "{\"type\":{}}",
        "{\"properties\":[]}",                                       "{\"properties\":5}",                             "{\"properties\":{\"a\":5}}",
        "{\"properties\":{\"a\":[]}}",                               "{\"properties\":{\"a\":{}},\"required\":\"a\"}", "{\"properties\":{\"a\":{}},\"required\":[1]}",
        "{\"properties\":{\"a\":{}},\"required\":{}}",               "{\"enum\":{}}",                                  "{\"enum\":[]}",
        "{\"enum\":5}",                                              "{\"items\":[{}]}",                               "{\"items\":5}",
        "{\"type\":\"array\",\"minItems\":-1}",                      "{\"type\":\"array\",\"minItems\":1.5}",          "{\"type\":\"array\",\"minItems\":\"1\"}",
        "{\"type\":\"array\",\"maxItems\":1e99}",                    "{\"type\":\"array\",\"maxItems\":65537}",        "{\"type\":\"array\",\"minItems\":3,\"maxItems\":2}",
        "{\"type\":\"array\",\"maxItems\":99999999999999999999999}", "{\"$ref\":5}",                                   "{\"$ref\":\"\"}",
        "{\"$ref\":\"#/\"}",                                         "{\"$ref\":\"#a\"}",                              "{\"$ref\":\"#/$defs\"}",
        "{\"$ref\":\"#/type\",\"type\":\"string\"}",                 "{\"$ref\":\"#/enum/5\",\"enum\":[1]}",           "{\"$ref\":\"#/$defs/a/b\",\"$defs\":{\"a\":5}}",
        "{\"$ref\":\"#\"}",                                          "{\"anyOf\":{}}",                                 "{\"anyOf\":[5]}",
        "{\"oneOf\":[]}",                                            "{\"oneOf\":\"a\"}",                              "{\"additionalProperties\":5}",
        "{\"type\":\"object\",\"additionalProperties\":[]}",         "{\"type\":\"string\"",                           "{\"type\":\"string\"}}",
        "{\"type\":\"string\"} x",                                   "{\"type\":\"str\\ud800ing\"}",                   "{\"type\":\"\xff\"}",
        "{\"a\x00\":1}",                                             "{type:\"string\"}",                              "{'type':'string'}",
        "{\"type\":\"string\",}",                                    "[{\"type\":\"string\"}]",                        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"object\",\"properties\":{\"b\":false}}}}",
    }) |schema| {
        errdefer std.debug.print("schema: {s}\n", .{schema});
        var diagnostic: Diagnostic = .{};
        if (Grammar.init(gpa, &vocabulary, schema, .{}, &diagnostic)) |made| {
            var grammar = made;
            grammar.deinit();
            return error.TestExpectedError;
        } else |err| {
            try testing.expect(err == error.InvalidSchema or err == error.UnsupportedSchema);
            try testing.expect(diagnostic.message().len > 0);
        }
    }

    // the bounds, each named with its number
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    const Bound = struct { options: Options = .{}, message: []const u8 };
    const refused = struct {
        fn refused(v: *Vocabulary, schema: []const u8, bound: Bound) !void {
            var diagnostic: Diagnostic = .{};
            try testing.expectError(error.UnsupportedSchema, Grammar.init(testing.allocator, v, schema, bound.options, &diagnostic));
            try testing.expectEqualStrings(bound.message, diagnostic.message());
        }
    }.refused;
    // schemas inside schemas: 64 deep is read, 65 is not
    for ([_]usize{ max_depth, max_depth + 1 }) |deep| {
        text.clearRetainingCapacity();
        for (0..deep) |_| try text.appendSlice(gpa, "{\"type\":\"array\",\"items\":");
        try text.appendSlice(gpa, "{\"type\":\"null\"}");
        try text.appendNTimes(gpa, '}', deep);
        var diagnostic: Diagnostic = .{};
        if (deep == max_depth) {
            var grammar: Grammar = try .init(gpa, &vocabulary, text.items, .{}, &diagnostic);
            grammar.deinit();
        } else {
            try testing.expectError(error.UnsupportedSchema, Grammar.init(gpa, &vocabulary, text.items, .{}, &diagnostic));
            // a long path is named by its end
            try testing.expect(std.mem.startsWith(u8, diagnostic.message(), ".../items/items"));
            try testing.expect(std.mem.endsWith(u8, diagnostic.message(), "/items: nested deeper than 64 schemas"));
        }
    }
    // a chain of `$ref`s is schemas inside schemas too
    text.clearRetainingCapacity();
    try text.appendSlice(gpa, "{\"$ref\":\"#/$defs/d0\",\"$defs\":{");
    for (0..80) |at| try text.print(gpa, "\"d{d}\":{{\"$ref\":\"#/$defs/d{d}\"}},", .{ at, at + 1 });
    try text.appendSlice(gpa, "\"d80\":{\"type\":\"null\"}}}");
    {
        var diagnostic: Diagnostic = .{};
        try testing.expectError(error.UnsupportedSchema, Grammar.init(gpa, &vocabulary, text.items, .{}, &diagnostic));
        try testing.expectEqualStrings("#/$defs/d64: nested deeper than 64 schemas", diagnostic.message());
    }
    // the schema's JSON, whatever it nests: an `enum` value a hundred thousand arrays deep
    text.clearRetainingCapacity();
    try text.appendSlice(gpa, "{\"enum\":[");
    try text.appendNTimes(gpa, '[', 100_000);
    try text.appendNTimes(gpa, ']', 100_000);
    try text.appendSlice(gpa, "]}");
    try refused(&vocabulary, text.items, .{ .message = "the schema's JSON nests deeper than 256" });
    // brackets inside a string are not nesting
    text.clearRetainingCapacity();
    try text.appendSlice(gpa, "{\"enum\":[\"\\\"");
    try text.appendNTimes(gpa, '[', 600);
    try text.appendSlice(gpa, "\"]}");
    {
        var diagnostic: Diagnostic = .{};
        var grammar: Grammar = try .init(gpa, &vocabulary, text.items, .{}, &diagnostic);
        grammar.deinit();
    }
    // its size
    text.clearRetainingCapacity();
    try text.appendSlice(gpa, "{\"type\":\"string\",\"description\":\"");
    try text.appendNTimes(gpa, 'a', 1 << 20);
    try text.appendSlice(gpa, "\"}");
    try refused(&vocabulary, text.items, .{ .message = "the schema is 1048610 bytes: more than the 1048576 read" });
    try refused(&vocabulary, "{\"type\":\"string\"}", .{ .options = .{ .max_schema_bytes = 16 }, .message = "the schema is 17 bytes: more than the 16 read" });
    // a literal and a property's name
    text.clearRetainingCapacity();
    try text.appendSlice(gpa, "{\"enum\":[\"");
    try text.appendNTimes(gpa, 'a', 1023);
    try text.appendSlice(gpa, "\"]}");
    try refused(&vocabulary, text.items, .{ .message = "#: `enum` has a value of 1025 bytes: more than the 1024 read" });
    text.clearRetainingCapacity();
    try text.appendSlice(gpa, "{\"type\":\"object\",\"properties\":{\"");
    try text.appendNTimes(gpa, 'a', 1023);
    try text.appendSlice(gpa, "\":{}}}");
    try refused(&vocabulary, text.items, .{ .message = "#: a property's name is 1025 bytes: more than the 1024 read" });
}
