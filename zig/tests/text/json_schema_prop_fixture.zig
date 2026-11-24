//! `json_schema_prop_test.zig`'s fixture: documents written from a schema's JSON (`Writer`), a vocabulary to
//! cut them by (`Tokens`), and the bench that walks a constraint and judges what it took (`Bench`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const Value = std.json.Value;
const fixtures = @import("testing.zig");
const fx = fixtures.schemas;
const grammar_file = @import("../../src/text/grammar.zig");
const json_schema = @import("../../src/text/json_schema.zig");
const Diagnostic = json_schema.Diagnostic;
const Engine = json_schema.Engine;
const Tokenizer = json_schema.Tokenizer;
const Validator = @import("json_schema_validator.zig").Validator;

// -- properties --------------------------------------------------------------------------------------------------
//
// Seeded and deterministic. What judges the machine here was written apart from it: `Validator` and `std.json`
// say what a document is, and `Writer` makes documents from a schema's JSON.

pub const prop = struct {
    /// Schemas beside `fx.cases`': what each stresses is in its comment. Their literals use only letters the tiny
    /// tokenizer has tokens for.
    const more = [_][]const u8{
        // empty objects and arrays, and the object that takes no key
        \\{"type":"object","properties":{"none":{"type":"object","additionalProperties":false},"nothing":{"type":"array","maxItems":0},"open":{"type":"object"},"list":{"type":"array"},"props":{"type":"object","properties":{}}},"required":["none","nothing"]}
        ,
        // enums whose values go on from one another, as strings, as numbers and across kinds
        \\{"type":"array","items":{"enum":["a","ab","abc","b","",1,12,123,1.5,-1,true,null,"true","nul",[1,2],{"a":"b"}]},"minItems":1,"maxItems":6}
        ,
        // escapes and characters past ASCII in an enum's value and in a property's name
        \\{"type":"object","properties":{"Zo\u00eb":{"enum":["\u6771\u4eac","a\"b","a\\b","\n","/"]},"a\"b":{"type":"string"},"\u6771":{"const":"\u00eb"}},"required":["Zo\u00eb"]}
        ,
        // numbers, each kind alone and together
        \\{"type":"object","properties":{"i":{"type":"integer"},"n":{"type":"number"},"both":{"type":["integer","number"]},"list":{"type":"array","items":{"type":"number"},"minItems":1},"or":{"anyOf":[{"type":"integer"},{"type":"string"},{"type":"null"}]}},"required":["i","n"]}
        ,
        // nesting: objects in arrays in objects, bounded at each level
        \\{"type":"object","properties":{"a":{"type":"array","items":{"type":"array","items":{"type":"object","properties":{"b":{"type":"array","items":{"type":"object","properties":{"c":{"type":"array","items":{"type":"boolean"},"maxItems":2}},"required":["c"]},"minItems":1,"maxItems":2}},"required":["b"]},"maxItems":2},"minItems":1,"maxItems":2}},"required":["a"]}
        ,
        // a tool's arguments as the server's tests give them: one required, the others not
        \\{"type":"object","properties":{"path":{"type":"string","description":"the file"},"lines":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2},"mode":{"type":"string","enum":["read","write","append"]},"force":{"type":"boolean","default":false}},"required":["path"],"additionalProperties":false}
        ,
        // a bare value of each kind
        \\{"type":"string"}
        ,
        \\{"type":"number"}
        ,
        \\{"type":["boolean","null","integer"]}
        ,
        \\{}
        ,
        \\{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":3}
        ,
        // a map of maps, and keywords that say what the value is without a `type`
        \\{"additionalProperties":{"additionalProperties":{"type":["integer","null"]}}}
        ,
        \\{"items":{"properties":{"a":{"maxItems":1}},"required":["a"]},"minItems":1}
    };

    pub fn schemas() [fx.cases.len + more.len][]const u8 {
        var all: [fx.cases.len + more.len][]const u8 = undefined;
        for (fx.cases, 0..) |case, at| all[at] = case.schema;
        for (more, 0..) |schema, at| all[fx.cases.len + at] = schema;
        return all;
    }

    /// A tokenizer's tokens as the tests draw them.
    const Tokens = struct {
        tokenizer: *const Tokenizer,
        arena: std.heap.ArenaAllocator,
        /// The ids a document may hold whose bytes begin with each byte.
        by_first: [256][]u32,
        /// How eager each id is to end a document, 1 the most (`walk`); 0 for an id no document holds.
        eager: []u8,

        fn init(gpa: Allocator, tokenizer: *const Tokenizer, stops: []const u32) !Tokens {
            var arena: std.heap.ArenaAllocator = .init(gpa);
            errdefer arena.deinit();
            var lists: [256]std.ArrayList(u32) = @splat(.empty);
            const eager = try arena.allocator().alloc(u8, tokenizer.count());
            for (0..tokenizer.count()) |index| {
                const id: u32 = @intCast(index);
                const bytes = tokenizer.tokenBytes(id);
                eager[index] = 0;
                if (bytes.len == 0 or tokenizer.isSpecial(id) or std.mem.findScalar(u32, stops, id) != null) continue;
                try lists[bytes[0]].append(arena.allocator(), id);
                eager[index] = if (bytes.len > 1) (if (std.mem.findAny(u8, bytes, " \t\r\n") == null) 5 else 6) else switch (bytes[0]) {
                    '"', '}', ']' => 1,
                    ':' => 2,
                    ',' => 3,
                    ' ', '\t', '\r', '\n' => 6,
                    else => 4,
                };
            }
            var by_first: [256][]u32 = undefined;
            for (&by_first, &lists) |*to, *from| to.* = from.items;
            return .{ .tokenizer = tokenizer, .arena = arena, .by_first = by_first, .eager = eager };
        }

        fn deinit(self: *Tokens) void {
            self.arena.deinit();
        }

        const How = enum { longest, shortest, random };

        /// `text` cut into tokens, each the longest that fits, the shortest, or any of them. False where no
        /// token begins the rest.
        pub fn split(self: *const Tokens, gpa: Allocator, text: []const u8, how: How, random: std.Random, out: *std.ArrayList(u32)) !bool {
            out.clearRetainingCapacity();
            var rest = text;
            while (rest.len > 0) {
                var chosen: ?u32 = null;
                var fits: usize = 0;
                for (self.by_first[rest[0]]) |id| {
                    const bytes = self.tokenizer.tokenBytes(id);
                    if (!std.mem.startsWith(u8, rest, bytes)) continue;
                    fits += 1;
                    const held = if (chosen) |c| self.tokenizer.tokenBytes(c).len else 0;
                    const better = switch (how) {
                        .longest => bytes.len > held,
                        .shortest => held == 0 or bytes.len < held,
                        .random => random.uintLessThan(usize, fits) == 0,
                    };
                    if (better) chosen = id;
                }
                const id = chosen orelse return false;
                try out.append(gpa, id);
                rest = rest[self.tokenizer.tokenBytes(id).len..];
            }
            return true;
        }
    };

    /// Documents of a schema, made from the schema's JSON by JSON Schema's own words and this file's four rules
    /// (its head comment): keys in the schema's order, blanks where they may go, values written as JSON writes
    /// them. It knows nothing of nodes, frames or tokens.
    const Writer = struct {
        arena: Allocator,
        random: std.Random,
        root: Value,
        /// What a string is made of: `units[i]` is a character or an escape, as JSON text.
        units: []const []const u8,
        out: std.ArrayList(u8) = .empty,

        const Error = error{TooDeep} || Allocator.Error;

        const all_units = [_][]const u8{
            "a",      "b",       "c",       "e",        "t",        "n",        "r",         "u",              "0",              "1",      "9",        "-",         ".",
            " ",      ",",       ":",       "{",        "}",        "[",        "]",         "'",              "\\n",            "\\t",    "\\\"",     "\\\\",      "\\/",
            "\\b",    "\\f",     "\\r",     "\\u00e9",  "\\u0000",  "\\uD7FF",  "\\ue000",   "\\ud83d\\ude00", "\\uDBFF\\uDFFF", "\u{e9}", "\u{6771}", "\u{1F600}", "\u{7f}",
            "\u{80}", "\u{7ff}", "\u{800}", "\u{d7ff}", "\u{e000}", "\u{ffff}", "\u{10000}", "\u{10ffff}",
        };

        /// The units `tokens` can write.
        fn unitsFor(arena: Allocator, tokens: *const Tokens) ![]const []const u8 {
            var kept: std.ArrayList([]const u8) = .empty;
            var ids: std.ArrayList(u32) = .empty;
            var none_random: std.Random.DefaultPrng = .init(0);
            for (all_units) |unit| {
                if (try tokens.split(arena, unit, .longest, none_random.random(), &ids)) try kept.append(arena, unit);
            }
            return kept.items;
        }

        fn put(self: *Writer, text: []const u8) Allocator.Error!void {
            try self.out.appendSlice(self.arena, text);
        }

        /// A place blanks may go: mostly none, sometimes a few, now and then as many as there may be.
        fn gap(self: *Writer) Allocator.Error!void {
            const count: usize = switch (self.random.uintLessThan(u32, 40)) {
                0 => 32,
                1...8 => 1 + self.random.uintLessThan(usize, 3),
                else => 0,
            };
            for (0..count) |_| try self.out.append(self.arena, " \n\t\r"[self.random.uintLessThan(usize, 4)]);
        }

        fn string(self: *Writer) Allocator.Error!void {
            try self.put("\"");
            if (self.units.len > 0) {
                for (0..self.random.uintLessThan(usize, 7)) |_| try self.put(self.units[self.random.uintLessThan(usize, self.units.len)]);
            }
            try self.put("\"");
        }

        fn number(self: *Writer, whole: bool) Allocator.Error!void {
            const zero = self.random.uintLessThan(u32, 5) == 0;
            // an integer has no `-0`
            if (self.random.uintLessThan(u32, 3) == 0 and !(whole and zero)) try self.put("-");
            if (zero) try self.put("0") else {
                try self.out.append(self.arena, "123456789"[self.random.uintLessThan(usize, 9)]);
                for (0..self.random.uintLessThan(usize, 6)) |_| try self.out.append(self.arena, '0' + self.random.uintLessThan(u8, 10));
            }
            if (whole) return;
            if (self.random.boolean()) {
                try self.put(".");
                for (0..1 + self.random.uintLessThan(usize, 3)) |_| try self.out.append(self.arena, '0' + self.random.uintLessThan(u8, 10));
            }
            if (self.random.uintLessThan(u32, 3) == 0) {
                try self.out.append(self.arena, "eE"[self.random.uintLessThan(usize, 2)]);
                switch (self.random.uintLessThan(u32, 3)) {
                    0 => try self.put("+"),
                    1 => try self.put("-"),
                    else => {},
                }
                for (0..1 + self.random.uintLessThan(usize, 2)) |_| try self.out.append(self.arena, '0' + self.random.uintLessThan(u8, 10));
            }
        }

        /// A value no schema asks anything of.
        fn anything(self: *Writer, depth: usize) Error!void {
            switch (self.random.uintLessThan(u32, if (depth > 3) 6 else 8)) {
                0 => try self.string(),
                1 => try self.number(false),
                2 => try self.number(true),
                3 => try self.put("true"),
                4 => try self.put("false"),
                5 => try self.put("null"),
                6 => try self.array(.{ .bool = true }, 0, 2, depth),
                else => try self.map(.{ .bool = true }, depth),
            }
        }

        fn array(self: *Writer, items: Value, least: usize, most: usize, depth: usize) Error!void {
            const count = if (depth > 5) least else @min(most, least + self.random.uintLessThan(usize, 4));
            try self.put("[");
            try self.gap();
            for (0..count) |at| {
                if (at > 0) {
                    try self.put(",");
                    try self.gap();
                }
                try self.value(items, depth + 1);
                try self.gap();
            }
            try self.put("]");
        }

        /// An object of any keys, each value of `values`.
        fn map(self: *Writer, values: Value, depth: usize) Error!void {
            const count = if (depth > 5) 0 else self.random.uintLessThan(usize, 3);
            try self.put("{");
            try self.gap();
            for (0..count) |at| {
                if (at > 0) {
                    try self.put(",");
                    try self.gap();
                }
                try self.string();
                try self.gap();
                try self.put(":");
                try self.gap();
                try self.value(values, depth + 1);
                try self.gap();
            }
            try self.put("}");
        }

        fn compact(self: *Writer, given: Value) Allocator.Error!void {
            try self.put(try std.fmt.allocPrint(self.arena, "{f}", .{std.json.fmt(given, .{})}));
        }

        fn value(self: *Writer, schema: Value, depth: usize) Error!void {
            if (depth > 40) return error.TooDeep;
            const object = switch (schema) {
                .object => |object| object,
                else => return self.anything(depth),
            };
            if (object.get("$ref")) |pointer| {
                return self.value((Validator{ .root = self.root }).target(pointer.string).?, depth + 1);
            }
            for ([_][]const u8{ "anyOf", "oneOf" }) |word| {
                const branches = (object.get(word) orelse continue).array.items;
                return self.value(branches[self.random.uintLessThan(usize, branches.len)], depth + 1);
            }
            if (object.get("enum")) |values| return self.compact(values.array.items[self.random.uintLessThan(usize, values.array.items.len)]);
            if (object.get("const")) |only| return self.compact(only);
            const Type = enum { string, number, integer, boolean, null, object, array };
            const of: Type = if (object.get("type")) |named| switch (named) {
                .string => |name| std.meta.stringToEnum(Type, name).?,
                .array => |names| std.meta.stringToEnum(Type, names.items[self.random.uintLessThan(usize, names.items.len)].string).?,
                else => unreachable,
            } else if (object.get("properties") != null or object.get("additionalProperties") != null)
                .object
            else if (object.get("items") != null or object.get("minItems") != null or object.get("maxItems") != null)
                .array
            else
                return self.anything(depth);
            switch (of) {
                .string => try self.string(),
                .number => try self.number(self.random.uintLessThan(u32, 4) == 0),
                .integer => try self.number(true),
                .boolean => try self.put(if (self.random.boolean()) "true" else "false"),
                .null => try self.put("null"),
                .array => {
                    const least: usize = if (object.get("minItems")) |n| @intCast(n.integer) else 0;
                    const most: usize = if (object.get("maxItems")) |n| @intCast(n.integer) else std.math.maxInt(usize);
                    try self.array(object.get("items") orelse .{ .bool = true }, least, most, depth);
                },
                .object => {
                    const listed: ?std.json.ObjectMap = if (object.get("properties")) |properties| properties.object else null;
                    if (listed == null or listed.?.count() == 0) {
                        const others = object.get("additionalProperties") orelse Value{ .bool = listed == null };
                        if (others == .bool and !others.bool) {
                            try self.put("{");
                            try self.gap();
                            return self.put("}");
                        }
                        return self.map(others, depth);
                    }
                    try self.put("{");
                    try self.gap();
                    var written: usize = 0;
                    for (listed.?.keys(), listed.?.values()) |key, under| {
                        const required = if (object.get("required")) |names| for (names.array.items) |name| {
                            if (std.mem.eql(u8, name.string, key)) break true;
                        } else false else false;
                        if (!required and (depth > 5 or self.random.boolean())) continue;
                        if (written > 0) {
                            try self.put(",");
                            try self.gap();
                        }
                        written += 1;
                        try self.compact(.{ .string = key });
                        try self.gap();
                        try self.put(":");
                        try self.gap();
                        try self.value(under, depth + 1);
                        try self.gap();
                    }
                    try self.put("}");
                },
            }
        }
    };

    /// A vocabulary, its engine and what the tests draw from, for one tokenizer.
    pub const Bench = struct {
        gpa: Allocator,
        tokens: Tokens,
        engine: Engine,
        stops: []const u32,
        /// Scratch masks, `words` each.
        mask: []u32,
        other: []u32,
        ids: std.ArrayList(u32) = .empty,
        text: std.ArrayList(u8) = .empty,
        /// Whether a walk that fails says where it stood.
        loud: bool = true,
        /// The most tokens a walk goes before it is given up.
        limit: usize = 4000,

        pub fn init(self: *Bench, gpa: Allocator, tokenizer: *const Tokenizer, stops: []const u32) !void {
            self.* = .{ .gpa = gpa, .tokens = try .init(gpa, tokenizer, stops), .engine = try .init(gpa, tokenizer, tokenizer.count() + 29, stops), .stops = stops, .mask = &.{}, .other = &.{} };
            const words = self.engine.vocabulary.words();
            self.mask = try gpa.alloc(u32, words);
            self.other = try gpa.alloc(u32, words);
        }

        pub fn deinit(self: *Bench) void {
            self.gpa.free(self.mask);
            self.gpa.free(self.other);
            self.ids.deinit(self.gpa);
            self.text.deinit(self.gpa);
            self.engine.deinit(self.gpa);
            self.tokens.deinit();
        }

        /// Whether `text` is a whole document of the schema: JSON as `std.json` reads it, and valid.
        fn whole(gpa: Allocator, schema: Value, text: []const u8) !bool {
            var arena: std.heap.ArenaAllocator = .init(gpa);
            defer arena.deinit();
            const given = std.json.parseFromSliceLeaky(Value, arena.allocator(), text, .{ .duplicate_field_behavior = .use_last }) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return false,
            };
            return (Validator{ .root = schema }).valid(schema, given);
        }

        /// The tokens `mask` allows, the stops aside.
        pub fn count(self: *const Bench, mask: []const u32) usize {
            var total: usize = 0;
            for (mask) |word| total += @popCount(word);
            for (self.stops) |stop| total -= @intFromBool(grammar_file.allowed(mask, stop));
            return total;
        }

        /// One reply drawn from the masks themselves: at each step a token the mask allows, any of them while
        /// `wander` says so and then the one most eager to end. Every step is held to the properties: the mask
        /// is not empty, the stop is allowed exactly where the text so far is a whole valid document, what the
        /// mask allows is taken and what it does not is refused, and the reply that ends is valid.
        /// Returns the reply's length in tokens.
        pub fn walk(self: *Bench, constraint: grammar_file.Constraint, schema: Value, random: std.Random, wander: f32, budget: usize) !usize {
            const tokenizer = self.tokens.tokenizer;
            self.text.clearRetainingCapacity();
            self.ids.clearRetainingCapacity();
            errdefer if (self.loud) std.debug.print("text so far: {s}\nids: {any}\n", .{ self.text.items, self.ids.items });
            var at = try constraint.start();
            var steps: usize = 0;
            while (true) : (steps += 1) {
                if (steps > self.limit) return error.WalkDidNotEnd;
                try constraint.fill(at, self.mask);
                const may_stop = grammar_file.allowed(self.mask, self.stops[0]);
                for (self.stops) |stop| try testing.expectEqual(may_stop, grammar_file.allowed(self.mask, stop));
                try testing.expectEqual(try whole(self.gpa, schema, self.text.items), may_stop);
                const open = self.count(self.mask);
                // no dead end: something may come, or the document may end
                try testing.expect(open > 0 or may_stop);

                // what the mask does not allow is not taken: a few of them, the specials and the ids past the tokenizer among them
                for (0..6) |nth| {
                    const token: u32 = switch (nth) {
                        0 => @intCast(self.engine.vocabulary.width - 1),
                        1 => @intCast(tokenizer.count()),
                        else => random.uintLessThan(u32, @intCast(self.engine.vocabulary.width)),
                    };
                    const next = try constraint.accept(at, token);
                    try testing.expectEqual(grammar_file.allowed(self.mask, token), next != null);
                }

                const wandering = steps < budget and random.float(f32) < wander;
                if (may_stop and (open == 0 or !wandering or random.uintLessThan(u32, 4) == 0)) {
                    const end = (try constraint.accept(at, self.stops[random.uintLessThan(usize, self.stops.len)])).?;
                    try testing.expect(constraint.ended(end));
                    try constraint.fill(end, self.mask);
                    try testing.expect(std.mem.allEqual(u32, self.mask, 0));
                    try testing.expectEqual(null, try constraint.accept(end, self.stops[0]));
                    try testing.expect(std.unicode.utf8ValidateSlice(self.text.items));
                    return steps;
                }
                // the nth allowed token, the stops aside: of them all, or of the most eager kind there is
                var best: u8 = 255;
                var of_best: usize = 0;
                var word_at: usize = 0;
                if (!wandering) for (self.mask, 0..) |word, index| {
                    var bits = word;
                    while (bits != 0) : (bits &= bits - 1) {
                        const id = index * 32 + @ctz(bits);
                        const kind = self.tokens.eager[id];
                        if (kind == 0 or kind > best) continue;
                        if (kind < best) of_best = 0;
                        best = kind;
                        of_best += 1;
                    }
                };
                var nth = random.uintLessThan(usize, if (wandering) open else of_best);
                const chosen: u32 = found: while (true) : (word_at += 1) {
                    var bits = self.mask[word_at];
                    while (bits != 0) : (bits &= bits - 1) {
                        const id: u32 = @intCast(word_at * 32 + @ctz(bits));
                        if (self.tokens.eager[id] == 0) continue;
                        if (!wandering and self.tokens.eager[id] != best) continue;
                        if (nth == 0) break :found id;
                        nth -= 1;
                    }
                };
                at = (try constraint.accept(at, chosen)).?;
                try self.ids.append(self.gpa, chosen);
                try self.text.appendSlice(self.gpa, tokenizer.tokenBytes(chosen));
            }
        }

        /// A document of the schema, as `Writer` makes it, that the tokenizer can write; null after a few tries.
        fn written(self: *Bench, arena: Allocator, schema: Value, units: []const []const u8, random: std.Random) !?[]const u8 {
            for (0..20) |_| {
                var writer: Writer = .{ .arena = arena, .random = random, .root = schema, .units = units };
                writer.value(schema, 0) catch |err| switch (err) {
                    error.TooDeep => continue,
                    error.OutOfMemory => return error.OutOfMemory,
                };
                if (try self.tokens.split(self.gpa, writer.out.items, .longest, random, &self.ids)) return writer.out.items;
            }
            return null;
        }

        const Counts = struct { walks: usize = 0, walk_tokens: usize = 0, documents: usize = 0, document_tokens: usize = 0, unwritten: usize = 0, compared: usize = 0 };

        /// The four properties over every schema, `rounds` times each. With `own` the engine holds a key to
        /// the tokenizer's own tokens (`Options.key_tokens`): a document is then followed in the tokens the
        /// tokenizer gives it, which must all be taken, and is not cut otherwise.
        pub fn run(self: *Bench, rounds: usize, seed: u64, own: bool) !Counts {
            const gpa = self.gpa;
            self.engine.options.key_tokens = own;
            var counts: Counts = .{};
            var arena_state: std.heap.ArenaAllocator = .init(gpa);
            defer arena_state.deinit();
            const words = self.engine.vocabulary.words();
            var prng: std.Random.DefaultPrng = .init(seed);
            const random = prng.random();
            var units_arena: std.heap.ArenaAllocator = .init(gpa);
            defer units_arena.deinit();
            const units = try Writer.unitsFor(units_arena.allocator(), &self.tokens);
            // the masks along a document, by the byte each stands at
            var along: std.ArrayList(u32) = .empty;
            defer along.deinit(gpa);
            var offsets: std.ArrayList(usize) = .empty;
            defer offsets.deinit(gpa);
            var places: std.ArrayList(grammar_file.Position) = .empty;
            defer places.deinit(gpa);
            var second: std.ArrayList(u32) = .empty;
            defer second.deinit(gpa);

            for (schemas(), 0..) |text, which| {
                errdefer std.debug.print("schema {d}: {s}\n", .{ which, text });
                const arena = arena_state.allocator();
                const schema = try std.json.parseFromSliceLeaky(Value, arena, text, .{});
                var diagnostic: Diagnostic = .{};
                const constraint = self.engine.interface().compile(gpa, .{ .json_schema = text }, &diagnostic) catch |err| {
                    std.debug.print("{s}\n", .{diagnostic.message()});
                    return err;
                };
                defer constraint.deinit();
                // a second constraint of the same schema: what the first is compared to shares nothing with it
                const reference = try self.engine.interface().compile(gpa, .{ .json_schema = text }, &diagnostic);
                defer reference.deinit();

                // (a) soundness and (c) no dead ends, on replies drawn from the masks
                for (0..rounds) |round| {
                    const wander: f32 = ([_]f32{ 0, 0.2, 0.6, 0.95 })[round % 4];
                    counts.walk_tokens += try self.walk(constraint, schema, random, wander, 20 + random.uintLessThan(usize, 120));
                    counts.walks += 1;
                }

                // (b) completeness and (d) a mask is its prefix's alone, on documents written from the schema
                for (0..rounds) |_| {
                    const document = try self.written(arena, schema, units, random) orelse {
                        counts.unwritten += 1;
                        continue;
                    };
                    errdefer std.debug.print("document: {s}\n", .{document});
                    try testing.expect(try whole(gpa, schema, document));
                    counts.documents += 1;

                    // the reference: the document's longest tokens, one after another from a fresh start
                    try testing.expect(try self.tokens.split(gpa, document, .longest, random, &self.ids));
                    if (own) {
                        self.ids.clearRetainingCapacity();
                        try self.tokens.tokenizer.encode(gpa, document, &self.ids);
                    }
                    along.clearRetainingCapacity();
                    offsets.clearRetainingCapacity();
                    var at = try reference.start();
                    var offset: usize = 0;
                    for (self.ids.items) |id| {
                        try reference.fill(at, self.mask);
                        try along.appendSlice(gpa, self.mask);
                        try offsets.append(gpa, offset);
                        try testing.expect(grammar_file.allowed(self.mask, id));
                        try testing.expectEqual(try whole(gpa, schema, document[0..offset]), grammar_file.allowed(self.mask, self.stops[0]));
                        at = (try reference.accept(at, id)).?;
                        offset += self.tokens.tokenizer.tokenBytes(id).len;
                    }
                    try reference.fill(at, self.mask);
                    try along.appendSlice(gpa, self.mask);
                    try offsets.append(gpa, offset);
                    try testing.expect(grammar_file.allowed(self.mask, self.stops[0]));
                    counts.document_tokens += self.ids.items.len;

                    // the same tokens with a draft tree at every position: tokens tried and not kept, some of
                    // them followed a few tokens deep and then left. The mask at each position is the reference's.
                    places.clearRetainingCapacity();
                    at = try constraint.start();
                    for (self.ids.items, 0..) |id, step| {
                        try places.append(gpa, at);
                        for (0..4) |_| {
                            var draft = at;
                            for (0..1 + random.uintLessThan(usize, 4)) |_| {
                                try constraint.fill(draft, self.other);
                                const open = self.count(self.other);
                                // mostly a token the mask allows there, sometimes any token at all
                                var token = random.uintLessThan(u32, @intCast(self.engine.vocabulary.width));
                                if (open > 0 and random.uintLessThan(u32, 4) != 0) {
                                    var nth = random.uintLessThan(usize, open);
                                    token = found: for (self.other, 0..) |word, index| {
                                        var bits = word;
                                        while (bits != 0) : (bits &= bits - 1) {
                                            const candidate: u32 = @intCast(index * 32 + @ctz(bits));
                                            if (std.mem.findScalar(u32, self.stops, candidate) != null) continue;
                                            if (nth == 0) break :found candidate;
                                            nth -= 1;
                                        }
                                    } else unreachable;
                                }
                                draft = try constraint.accept(draft, token) orelse break;
                            }
                        }
                        try constraint.fill(at, self.other);
                        try testing.expectEqualSlices(u32, along.items[step * words ..][0..words], self.other);
                        at = (try constraint.accept(at, id)).?;
                        counts.compared += 1;
                    }
                    try places.append(gpa, at);
                    // every position passed is still its prefix's, after all that was tried since
                    for (places.items, 0..) |place, step| {
                        try constraint.fill(place, self.other);
                        try testing.expectEqualSlices(u32, along.items[step * words ..][0..words], self.other);
                    }

                    // other cuts of the same text: taken at every step, and where a cut falls on a byte the
                    // reference stood at, the mask is the reference's
                    for ([_]Tokens.How{ .shortest, .random, .random }) |how| {
                        if (own) break;
                        if (!try self.tokens.split(gpa, document, how, random, &second)) continue;
                        at = try constraint.start();
                        offset = 0;
                        for (second.items, 0..) |id, step| {
                            errdefer std.debug.print("{t} cut, token {d}: {d} at byte {d}\n", .{ how, step, id, offset });
                            try constraint.fill(at, self.other);
                            try testing.expect(grammar_file.allowed(self.other, id));
                            try testing.expect(self.count(self.other) > 0);
                            if (std.mem.findScalar(usize, offsets.items, offset)) |same| {
                                try testing.expectEqualSlices(u32, along.items[same * words ..][0..words], self.other);
                                counts.compared += 1;
                            }
                            at = (try constraint.accept(at, id)).?;
                            offset += self.tokens.tokenizer.tokenBytes(id).len;
                        }
                        try constraint.fill(at, self.other);
                        try testing.expectEqualSlices(u32, along.items[(offsets.items.len - 1) * words ..][0..words], self.other);
                        try testing.expect(constraint.ended((try constraint.accept(at, self.stops[0])).?));
                        counts.document_tokens += second.items.len;
                    }
                }
                _ = arena_state.reset(.retain_capacity);
            }
            return counts;
        }
    };
};
