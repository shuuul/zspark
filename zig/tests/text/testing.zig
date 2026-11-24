//! Fixtures two or more of `zig/src/text/`'s test files share. `schemas`: the tiny tokenizer and the schemas with
//! their documents, for `json_schema_test.zig`, `json_schema_prop_test.zig` and `json_schema_oracle_test.zig`.
//! `gate`: the tools, a tokenizer with its engine (`Bench`) and a reply drawn from a gate's masks (`walk`), for
//! `call_gate_test.zig` and `call_gate_walk_test.zig`, and a reply's calls read back (`read`), which the tests of
//! `zig/src/server/` and of a model's constrained replies hold a call to. A fixture's imports stand inside it, so that one's
//! names are never another's locals. `validates`: a document held against a schema, which tests here, in
//! `zig/src/server/` and in the models read.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const json_schema = @import("../../src/text/json_schema.zig");
const json_schema_validator = @import("json_schema_validator.zig");
const tokenizer_file = @import("../../src/text/tokenizer.zig");

/// Whether the text `document` is JSON and what the schema text `schema` asks. `arena` holds both while it
/// runs. The document's repeated keys keep their last value, as JSON's readers do.
pub fn validates(arena: Allocator, schema: []const u8, document: []const u8) !bool {
    const asked = try std.json.parseFromSliceLeaky(std.json.Value, arena, schema, .{});
    const given = std.json.parseFromSliceLeaky(std.json.Value, arena, document, .{ .duplicate_field_behavior = .use_last }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return false,
    };
    return (json_schema_validator.Validator{ .root = asked }).valid(asked, given);
}

pub const schemas = struct {
    const Diagnostic = json_schema.Diagnostic;
    const Grammar = json_schema.Grammar;
    const Matcher = json_schema.Matcher;
    const Tokenizer = json_schema.Tokenizer;
    const Vocabulary = json_schema.Vocabulary;
    const allowed = json_schema.allowed;

    const tiny_tokenizer = @embedFile("fixtures/text/tiny_tokenizer.json");

    const flat =
        \\{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"score":{"type":"number"},"active":{"type":"boolean"},"note":{"type":"null"}},"required":["name","age","score","active"]}
    ;
    const nested =
        \\{"type":"object","properties":{"id":{"type":"integer"},"customer":{"type":"object","properties":{"name":{"type":"string"},"email":{"type":"string"},"tier":{"enum":["free","pro","enterprise"]}},"required":["name","tier"]},"items":{"type":"array","items":{"type":"object","properties":{"sku":{"type":"string"},"qty":{"type":"integer"},"price":{"type":"number"},"tags":{"type":"array","items":{"type":"string"}},"status":{"type":"string","enum":["new","shipped","returned"]}},"required":["sku","qty","status"]}},"gift":{"type":"boolean"},"coupon":{"type":["string","null"]}},"required":["id","customer","items"]}
    ;
    /// A `get_weather` tool: nothing required.
    const tool =
        \\{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"units":{"type":"string","enum":["c","f"]}}}
    ;
    pub const anything =
        \\{"type": "object"}
    ;
    /// As Pydantic 2 writes a model: a nested model and an enum through `$ref`, `Optional[...]` as `anyOf` with
    /// null, `default` and `title` everywhere, `type` after `properties`.
    pub const pydantic =
        \\{"$defs":{"Address":{"properties":{"street":{"title":"Street","type":"string"},"post":{"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"title":"Post"}},"required":["street"],"title":"Address","type":"object"},"Kind":{"enum":["home","work"],"title":"Kind","type":"string"}},"properties":{"name":{"title":"Name","type":"string"},"age":{"anyOf":[{"type":"integer"},{"type":"null"}],"default":null,"title":"Age"},"address":{"$ref":"#/$defs/Address"},"kind":{"$ref":"#/$defs/Kind","default":"home"},"previous":{"anyOf":[{"$ref":"#/$defs/Address"},{"type":"null"}],"default":null},"scores":{"items":{"anyOf":[{"type":"number"},{"type":"integer"},{"type":"boolean"}]},"title":"Scores","type":"array"}},"required":["name","address"],"title":"Person","type":"object"}
    ;
    /// A schema that names itself: a tree, by `$ref` in `items` and in an `anyOf` with null; `definitions` and
    /// an escaped pointer.
    pub const tree =
        \\{"definitions":{"a/node":{"type":"object","properties":{"value":{"type":"integer"},"children":{"type":"array","items":{"$ref":"#/definitions/a~1node"}},"next":{"oneOf":[{"$ref":"#/definitions/a~1node"},{"type":"null"}]}},"required":["value"]}},"$ref":"#/definitions/a~1node"}
    ;

    const Case = struct { schema: []const u8, good: []const []const u8, bad: []const []const u8 };
    /// Each bad document is refused at its last byte and no sooner.
    pub const cases = [_]Case{
        .{
            .schema = flat,
            .good = &.{
                \\{"name": "Ada Lovelace", "age": 36, "score": 9.75, "active": true}
                ,
                \\{"name":"李雷 \"the\" tester\n","age":-7,"score":-1.5e-3,"active":false,"note":null}
                ,
                "{\n  \"name\": \"caf\\u00e9 \\\\ path/to\",\n  \"age\": 0,\n  \"score\": 12,\n  \"active\": true,\n  \"note\": null\n}",
                // surrogates as escapes: a pair, and the code points on either side of them
                \\{"name":"\ud83d\ude00 \uD7FF\ue000\uDBFF\uDFFF","age":1,"score":1,"active":true}
                ,
            },
            .bad = &.{
                " ",
                \\{"a
                ,
                \\{"name": 1
                ,
                \\{"name": "x", "s
                ,
                \\{"name": "x", "age": 01
                ,
                \\{"name": "x", "age": -0
                ,
                \\{"name": "x", "age": 1.
                ,
                \\{"name": "x", "age": 1, "score": 1.e
                ,
                \\{"name": "x", "age": 1, "score": 1}
                ,
                \\{"name": "x", "age": 1, "score": 1, "active": true, "na
                ,
                \\{"name": "x", "age": 1, "score": 1, "active": true,}
                ,
                "{\"name\": \"x\", \"age\": 1, \"score\": 1, \"active\": true} ",
                "{\"name\": \"\t",
                "{\"name\": \"\\x",
                "{\"name\": \"\\u12g",
                // a high surrogate's escape with no low one after it, and a low one's alone
                \\{"name": "\ud83dx
                ,
                \\{"name": "\ud83d\n
                ,
                \\{"name": "\ud83d\u0
                ,
                \\{"name": "\uD83D\ud8
                ,
                \\{"name": "\udc
                ,
                \\{"name": "\uDF
                ,
                "{\"name\": \"\xc0",
                "{\"name\": \"\xed\xa0",
                "{\"name\": \"\xf4\x90",
                "{" ++ @as([33]u8, @splat(' ')),
            },
        },
        .{
            .schema = nested,
            .good = &.{
                \\{"id": 1042, "customer": {"name": "Grace Hopper", "email": "grace@example.com", "tier": "pro"}, "items": [{"sku": "A-100", "qty": 2, "price": 19.99, "tags": ["blue", "large"], "status": "new"}, {"sku": "B-7", "qty": 1, "status": "shipped"}], "gift": false, "coupon": null}
                ,
                \\{"id":7,"customer":{"name":"x","tier":"enterprise"},"items":[]}
                ,
                "{\n  \"id\": 3,\n  \"customer\": {\n    \"name\": \"Zoë\",\n    \"tier\": \"free\"\n  },\n  \"items\": [\n    {\n      \"sku\": \"q\",\n      \"qty\": 10,\n      \"tags\": [],\n      \"status\": \"returned\"\n    }\n  ],\n  \"coupon\": \"SAVE10\"\n}",
            },
            .bad = &.{
                \\{"id":7,"customer":{"name":"x","tier":"prox
                ,
                \\{"id":7,"customer":{"name":"x","tier":"pro"},"items":[,
                ,
                \\{"id":7,"customer":{"name":"x","tier":"pro"},"items":[{"sku":"a","qty":1,"status":"new"},]
                ,
                \\{"id":7,"customer":{"name":"x","tier":"pro"},"items":[],"coupon":t
            },
        },
        .{
            .schema = tool,
            .good = &.{
                \\{"city": "San Francisco", "days": 3, "units": "c"}
                ,
                \\{"days": 14}
                ,
                \\{}
                ,
                \\{"city":"東京","units":"f"}
            },
            .bad = &.{
                \\{"units": "c",
                ,
                \\{"days": 1, "c
                ,
                \\{"units": "k
            },
        },
        .{
            .schema = anything,
            .good = &.{
                \\{"answer": 42, "list": [1, 2, {"a": "b"}], "nested": {"deep": {"deeper": [[], {}]}}, "text": "line\nbreak", "n": [-0.0, 1E+2, true, false, null]}
                ,
                \\{}
            },
            .bad = &.{
                "[",
                \\{"a": tx
                ,
                \\{"a": [1 2
                ,
                \\{"a": {"b": 1]
                ,
                \\{1
            },
        },
        .{
            // a literal that another goes on from, a bounded array, a number as the whole document's last value
            .schema =
            \\{"type":"object","properties":{"e":{"enum":[1,12,"1"]},"p":{"type":"array","items":{"type":["integer","number","null"]},"minItems":2,"maxItems":3},"o":{"type":"object","additionalProperties":{"type":"boolean"}}},"required":["e","p"]}
            ,
            .good = &.{
                \\{"e":1,"p":[1,null]}
                ,
                \\{"e": 12 , "p": [1.5, 2, 3], "o": {"k": true, "j": false}}
                ,
                \\{"e":"1","p":[0,0],"o":{}}
            },
            .bad = &.{
                \\{"e":13
                ,
                \\{"e":1,"p":[1]
                ,
                \\{"e":1,"p":[1,2,3,
                ,
                \\{"e":1,"p":[1,2],"o":{"k":1
            },
        },
        .{
            .schema = pydantic,
            .good = &.{
                \\{"name": "Ada", "address": {"street": "12 St James's Square"}}
                ,
                \\{"name":"Ada","age":36,"address":{"street":"x","post":null},"kind":"work","previous":{"street":"y","post":"W1"},"scores":[1,2.5,true,-3e2]}
                ,
                "{\n  \"name\": \"Zoë\",\n  \"age\": null,\n  \"address\": {\n    \"street\": \"\"\n  },\n  \"kind\": \"home\",\n  \"previous\": null,\n  \"scores\": []\n}",
            },
            .bad = &.{
                \\{"name":"a","age":"
                ,
                \\{"name":"a","age":t
                ,
                \\{"name":"a","address":{"x
                ,
                \\{"name":"a","address":{"street":"x","post":1
                ,
                \\{"name":"a","address":{"street":"x"},"kind":"homes
                ,
                \\{"name":"a","address":{"street":"x"},"previous":{"x
                ,
                \\{"name":"a","address":{"street":"x"},"previous":nul}
                ,
                \\{"name":"a","address":{"street":"x"},"scores":["
            },
        },
        .{
            .schema = tree,
            .good = &.{
                \\{"value":1}
                ,
                \\{"value": 1, "children": [{"value": 2, "children": [{"value": 3}, {"value": 4, "next": null}]}, {"value": 5}], "next": {"value": 6, "next": {"value": 7, "children": []}}}
            },
            .bad = &.{
                \\{"value":1,"children":[1
                ,
                \\{"value":1,"children":[{"value":2,"children":[{"n
                ,
                \\{"value":1,"next":{"value":2,"next":{"value":3,"next":[
                ,
                \\{"value":1,"next":{"value":2,"next":{"c
            },
        },
        .{
            // a union at the root, one in a union, and a `$ref` to the whole schema
            .schema =
            \\{"oneOf":[{"type":"array","items":{"$ref":"#"},"maxItems":2},{"anyOf":[{"const":"leaf"},{"type":"null"}]},{"type":"integer"}]}
            ,
            .good = &.{
                \\[[1, "leaf"], [null, []]]
                ,
                \\"leaf"
                ,
                \\[]
            },
            .bad = &.{
                \\[[1,"leaf",
                ,
                \\"lead
                ,
                \\[1.
                ,
                \\{
            },
        },
    };

    pub fn tiny(gpa: Allocator) !Tokenizer {
        var diagnostic: tokenizer_file.Diagnostic = .{};
        return Tokenizer.parse(gpa, tiny_tokenizer, &diagnostic);
    }

    /// Every good document, cut into `tokenizer`'s tokens: each token is in the mask before it and taken, the
    /// mask is the one made without the cache, and at the end only a stop may come. Every bad document, as
    /// bytes: taken up to its last byte, refused there.
    pub fn check(tokenizer: *const Tokenizer, stops: []const u32, every: usize) !void {
        const gpa = testing.allocator;
        var vocabulary: Vocabulary = try .init(gpa, tokenizer, tokenizer.count() + 40, stops);
        defer vocabulary.deinit(gpa);
        const mask = try gpa.alloc(u32, vocabulary.words());
        defer gpa.free(mask);
        const slow = try gpa.alloc(u32, vocabulary.words());
        defer gpa.free(slow);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);

        for (cases) |case| {
            var diagnostic: Diagnostic = .{};
            var grammar = Grammar.init(gpa, &vocabulary, case.schema, .{}, &diagnostic) catch |err| {
                std.debug.print("{s}\n", .{diagnostic.message()});
                return err;
            };
            defer grammar.deinit();
            for (case.good) |document| {
                errdefer std.debug.print("document: {s}\n", .{document});
                // the validator, written apart from the machine, takes what the machine takes
                var arena: std.heap.ArenaAllocator = .init(gpa);
                defer arena.deinit();
                try testing.expect(try validates(arena.allocator(), case.schema, document));
                ids.clearRetainingCapacity();
                try tokenizer.encode(gpa, document, &ids);
                var matcher: Matcher = try .init(&grammar);
                defer matcher.deinit();
                for (ids.items, 0..) |id, at| {
                    errdefer std.debug.print("token {d}: {d} `{s}`\n", .{ at, id, tokenizer.tokenBytes(id) });
                    try matcher.fill(mask);
                    if (at % every == 0) {
                        try matcher.fillSlowly(slow);
                        try testing.expectEqualSlices(u32, slow, mask);
                    }
                    try testing.expect(allowed(mask, id));
                    for (stops) |stop| try testing.expect(!allowed(mask, stop));
                    try testing.expect(!matcher.complete());
                    try testing.expect(try matcher.accept(id));
                }
                try testing.expect(matcher.complete() and !matcher.finished());
                try matcher.fill(mask);
                var count: usize = 0;
                for (mask) |word| count += @popCount(word);
                try testing.expectEqual(stops.len, count);
                for (stops) |stop| try testing.expect(allowed(mask, stop));

                // a step back gives the mask there was
                try matcher.fillSlowly(slow);
                matcher.rollback(@min(3, ids.items.len));
                try matcher.fill(mask);
                try testing.expect(!std.mem.eql(u32, slow, mask));
                for (ids.items[ids.items.len - @min(3, ids.items.len) ..]) |id| try testing.expect(try matcher.accept(id));
                try testing.expect(try matcher.accept(stops[0]));
                try testing.expect(matcher.finished());
                try matcher.fill(mask);
                try testing.expect(std.mem.allEqual(u32, mask, 0));
                try testing.expect(!try matcher.accept(stops[0]));
            }
            for (case.bad) |document| {
                errdefer std.debug.print("refused document: {s}\n", .{document});
                var matcher: Matcher = try .init(&grammar);
                defer matcher.deinit();
                try testing.expect(try matcher.acceptBytes(document[0 .. document.len - 1]));
                try testing.expect(!try matcher.acceptBytes(document[document.len - 1 ..]));
            }
        }
    }
};

pub const gate = struct {
    const call_gate = @import("../../src/text/call_gate.zig");
    const grammar = @import("../../src/text/grammar.zig");
    const Options = call_gate.Options;
    const Tokenizer = call_gate.Tokenizer;
    const Tool = call_gate.Tool;
    const compile = call_gate.compile;

    const tiny_tokenizer = @embedFile("fixtures/text/tiny_tokenizer.json");

    pub const add: Tool = .{ .name = "add", .parameters = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"},\"b\":{\"type\":\"integer\"},\"exact\":{\"type\":\"boolean\"}},\"required\":[\"a\",\"b\"]}" };
    pub const lookup: Tool = .{ .name = "lookup", .parameters = "{\"type\":\"object\",\"properties\":{\"kind\":{\"enum\":[\"a\",\"ab\",null]},\"tags\":{\"type\":\"array\",\"items\":{\"enum\":[\"x\",\"y\"]},\"maxItems\":3},\"note\":{\"type\":\"string\"}},\"required\":[\"kind\",\"tags\"]}" };
    pub const nothing: Tool = .{ .name = "noop", .parameters = "{\"type\":\"object\",\"properties\":{}}" };
    // names that go on from one another, and one that needs escaping
    pub const a: Tool = .{ .name = "a", .parameters = "{\"type\":\"object\",\"properties\":{\"x\":{\"const\":1}},\"required\":[\"x\"]}" };
    pub const ab: Tool = .{ .name = "ab", .parameters = "{\"type\":\"object\",\"properties\":{\"x\":{\"const\":2}},\"required\":[\"x\"]}" };
    pub const abc: Tool = .{ .name = "abc", .parameters = "{\"type\":\"object\",\"properties\":{\"x\":{\"const\":3}},\"required\":[\"x\"]}" };
    pub const quoted: Tool = .{ .name = "say \"hi\"\\now", .parameters = "{\"type\":\"object\",\"properties\":{\"x\":{\"const\":4}},\"required\":[\"x\"]}" };

    pub const Bench = struct {
        tokenizer: Tokenizer,
        engine: json_schema.Engine,
        stops: [1]u32,

        pub fn init(self: *Bench) !void {
            var diagnostic: tokenizer_file.Diagnostic = .{};
            self.tokenizer = try Tokenizer.parse(testing.allocator, tiny_tokenizer, &diagnostic);
            errdefer self.tokenizer.deinit(testing.allocator);
            self.stops = .{self.tokenizer.addedId("<|end|>").?};
            self.engine = try .init(testing.allocator, &self.tokenizer, self.tokenizer.count() + 29, &self.stops);
        }

        pub fn deinit(self: *Bench) void {
            self.engine.deinit(testing.allocator);
            self.tokenizer.deinit(testing.allocator);
        }

        pub fn gate(self: *Bench, tools: []const Tool, options: Options, diagnostic: *grammar.Diagnostic) !grammar.Constraint {
            return compile(testing.allocator, self.engine.interface(), &self.tokenizer, &self.stops, tools, options, diagnostic);
        }
    };

    /// Where `needle` first is in `haystack`, case aside.
    pub fn findFolded(haystack: []const u8, needle: []const u8) ?usize {
        if (needle.len > haystack.len) return null;
        for (0..haystack.len - needle.len + 1) |at| {
            if (std.ascii.eqlIgnoreCase(haystack[at..][0..needle.len], needle)) return at;
        }
        return null;
    }

    /// One call of a reply's text, as the server's parser would find it.
    pub const Call = struct { name: []const u8, arguments: []const u8 };

    /// The calls of `text`, which must be calls in the gate's form and nothing else.
    pub fn calls(arena: Allocator, text: []const u8) ![]Call {
        var out: std.ArrayList(Call) = .empty;
        var rest = text;
        while (rest.len != 0) {
            if (out.items.len != 0) {
                try testing.expect(std.mem.startsWith(u8, rest, "\n"));
                rest = rest[1..];
            }
            try testing.expect(std.mem.startsWith(u8, rest, "<tool_call>\n"));
            const end = std.mem.indexOf(u8, rest, "\n</tool_call>") orelse return error.TestUnexpectedResult;
            const body = rest["<tool_call>\n".len..end];
            rest = rest[end + "\n</tool_call>".len ..];
            const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{ .duplicate_field_behavior = .@"error" });
            try testing.expectEqual(2, parsed.object.count());
            try testing.expectEqualStrings("name", parsed.object.keys()[0]);
            try testing.expectEqualStrings("arguments", parsed.object.keys()[1]);
            try testing.expect(parsed.object.get("arguments").? == .object);
            // the arguments' own text: after `"arguments":`, before the call's last `}`
            const marker = ", \"arguments\":";
            const from = std.mem.indexOf(u8, body, marker).? + marker.len;
            try out.append(arena, .{ .name = parsed.object.get("name").?.string, .arguments = body[from .. body.len - 1] });
        }
        return out.items;
    }

    /// The calls of `reply`, which must be calls and nothing else, each in the JSON form or in the template's,
    /// read by the rules a reply's parser reads them by and written here apart from any: the first
    /// `</tool_call>` in any case ends a call, the first `</parameter>` a value, one newline either side of a
    /// value is the frame's, and a value is its text for a string and JSON for another type.
    pub fn read(arena: Allocator, reply: []const u8, tools: []const Tool) ![]Call {
        var out: std.ArrayList(Call) = .empty;
        var rest = reply;
        while (rest.len != 0) {
            if (out.items.len != 0) {
                try testing.expect(std.mem.startsWith(u8, rest, "\n"));
                rest = rest[1..];
            }
            try testing.expect(std.mem.startsWith(u8, rest, "<tool_call>\n"));
            const end = findFolded(rest, "</tool_call>") orelse return error.TestUnexpectedResult;
            const block = rest["<tool_call>\n".len..end];
            if (block[0] == '{') {
                const one = try calls(arena, rest[0 .. end + "</tool_call>".len]);
                try out.append(arena, one[0]);
                rest = rest[end + "</tool_call>".len ..];
                continue;
            }
            rest = rest[end + "</tool_call>".len ..];
            try testing.expect(std.mem.startsWith(u8, block, "<function="));
            const name_end = std.mem.indexOfScalar(u8, block, '>').?;
            const name = block["<function=".len..name_end];
            const tool = for (tools) |tool| {
                if (std.mem.eql(u8, tool.name, name)) break tool;
            } else return error.TestUnexpectedResult;
            const properties = (try std.json.parseFromSliceLeaky(std.json.Value, arena, tool.parameters, .{})).object.get("properties").?.object;
            var body = block[name_end + 1 ..];
            try testing.expect(std.mem.startsWith(u8, body, "\n"));
            body = body[1..];
            var arguments: std.ArrayList(u8) = .empty;
            try arguments.append(arena, '{');
            while (!std.mem.eql(u8, body, "</function>\n")) {
                try testing.expect(std.mem.startsWith(u8, body, "<parameter="));
                const key_end = std.mem.indexOfScalar(u8, body, '>').?;
                const key = body["<parameter=".len..key_end];
                try testing.expectEqual('\n', body[key_end + 1]);
                const from = key_end + 2;
                const to = from + (findFolded(body[from..], "</parameter>") orelse return error.TestUnexpectedResult);
                try testing.expect(to > from and body[to - 1] == '\n');
                const written = body[from .. to - 1];
                body = body[to + "</parameter>".len ..];
                try testing.expect(std.mem.startsWith(u8, body, "\n"));
                body = body[1..];
                if (arguments.items.len != 1) try arguments.append(arena, ',');
                try arguments.appendSlice(arena, try std.json.Stringify.valueAlloc(arena, key, .{}));
                try arguments.append(arena, ':');
                const kind = properties.get(key).?.object.get("type").?.string;
                if (std.mem.eql(u8, kind, "string")) {
                    try testing.expect(std.unicode.utf8ValidateSlice(written));
                    try arguments.appendSlice(arena, try std.json.Stringify.valueAlloc(arena, written, .{}));
                } else try arguments.appendSlice(arena, written);
            }
            try arguments.append(arena, '}');
            try out.append(arena, .{ .name = name, .arguments = arguments.items });
        }
        return out.items;
    }

    /// A reply drawn from the gate's masks: at each position one of the tokens allowed, by `random`. A closing
    /// quote or bracket is taken when it may be once the reply is `hurry` tokens long, so that a
    /// string does not run for ever. Null when the reply did not end within `limit` tokens.
    pub fn walk(gpa: Allocator, constraint: grammar.Constraint, vocabulary: usize, stop: u32, random: std.Random, hurry: usize, limit: usize, ids: *std.ArrayList(u32), tokenizer: *const Tokenizer) !bool {
        const mask = try gpa.alloc(u32, constraint.words);
        defer gpa.free(mask);
        var allowed: std.ArrayList(u32) = .empty;
        defer allowed.deinit(gpa);
        ids.clearRetainingCapacity();
        var at = try constraint.start();
        while (ids.items.len < limit) {
            try constraint.fill(at, mask);
            allowed.clearRetainingCapacity();
            for (0..vocabulary) |token| {
                if (grammar.allowed(mask, @intCast(token))) try allowed.append(gpa, @intCast(token));
            }
            // never a dead end
            try testing.expect(allowed.items.len != 0);
            var token = allowed.items[random.uintLessThan(usize, allowed.items.len)];
            if (ids.items.len >= hurry) {
                for (allowed.items) |candidate| {
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
