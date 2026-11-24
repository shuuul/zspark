//! A JSON schema's text into the tree of nodes the matcher walks (`Schema`, `Node`, `Builder`, `parseSchema`):
//! what `json_schema.zig`'s header lists as read is read here, and a keyword that is not is refused by its
//! name.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const json_schema = @import("json_schema.zig");
const Diagnostic = json_schema.Diagnostic;
const Options = json_schema.Options;
const SchemaError = json_schema.SchemaError;
const max_depth = json_schema.max_depth;
const none = json_schema.none;

// -- the schema --------------------------------------------------------------------------------------------------

const Kind = enum(u8) {
    any,
    string,
    integer,
    number,
    literals,
    object,
    array,
    choice,
    /// A `$ref` while the schema is built: `first` is the node it names. None is left once it is built.
    ref,
};

const Node = struct {
    kind: Kind,
    /// literals: its trie's root. object: its first property in `properties`. array: its items' node.
    /// choice: its first alternative in `alternatives`.
    first: u32 = none,
    /// object: how many properties. choice: how many alternatives. array: the least items.
    count: u32 = 0,
    /// object: the node of the value under any key, for an object without properties (`none`: no key at all).
    /// array: the most items (`none`: no bound).
    other: u32 = none,
};

const Property = struct {
    node: u32,
    required: bool,
    /// The trie of the keys that may come when this property is the earliest one not yet passed: its own
    /// and those after it up to the next required one. A key's value is its property's place in the object.
    keys: u32,
};

/// A node of a trie of byte strings. A string may end where another goes on (`1` and `12` in an enum).
const Trie = struct {
    /// Its children are `edge_bytes[edges..][0..fan]` leading to `edge_nodes[...]`.
    edges: u32,
    fan: u32,
    /// What the string ending here stands for, or `none` when none ends here.
    value: u32,
};

/// The nodes every schema has, at fixed places.
pub const fixed = struct {
    const any = 0;
    pub const string = 1;
    pub const number = 2;
    const integer = 3;
    /// `{"type": "object"}`: any keys, any values.
    pub const object = 4;
    /// `{"type": "array"}`: any items.
    pub const array = 5;
    const boolean = 6;
    const null_ = 7;
    /// `true`, `false` and `null`, for a value of any type.
    pub const words = 8;
};

pub const Schema = struct {
    arena: std.heap.ArenaAllocator,
    nodes: []Node,
    properties: []Property,
    alternatives: []u32,
    tries: []Trie,
    edge_bytes: []u8,
    edge_nodes: []u32,
    root: u32,
    blanks: u8,

    pub fn child(self: *const Schema, trie: u32, byte: u8) ?u32 {
        const node = self.tries[trie];
        const at = std.mem.findScalar(u8, self.edge_bytes[node.edges..][0..node.fan], byte) orelse return null;
        return self.edge_nodes[node.edges + at];
    }

    /// Whether a value of `node` can begin with `byte`.
    pub fn begins(self: *const Schema, node: u32, byte: u8) bool {
        const n = self.nodes[node];
        return switch (n.kind) {
            .any => switch (byte) {
                '"', '-', '0'...'9', 't', 'f', 'n', '[', '{' => true,
                else => false,
            },
            .string => byte == '"',
            .integer, .number => byte == '-' or (byte >= '0' and byte <= '9'),
            .literals => self.child(n.first, byte) != null,
            .object => byte == '{',
            .array => byte == '[',
            .choice => for (self.alternatives[n.first..][0..n.count]) |alternative| {
                if (self.begins(alternative, byte)) break true;
            } else false,
            .ref => unreachable,
        };
    }

    /// Whether the object `node` still owes a required property at or after place `from`.
    pub fn owes(self: *const Schema, node: Node, from: u32) bool {
        if (from >= node.count) return false;
        for (self.properties[node.first + from ..][0 .. node.count - from]) |property| {
            if (property.required) return true;
        }
        return false;
    }
};

pub const annotations = [_][]const u8{ "description", "title", "default", "examples", "$schema", "$id", "$comment", "deprecated", "readOnly", "writeOnly" };
pub const keywords = [_][]const u8{ "type", "properties", "required", "items", "enum", "const", "minItems", "maxItems", "additionalProperties", "$ref", "anyOf", "oneOf" };
/// Where a schema keeps the schemas its `$ref`s name: read when one is named, not otherwise.
pub const definitions = [_][]const u8{ "$defs", "definitions" };

const Builder = struct {
    arena: Allocator,
    diagnostic: *Diagnostic,
    nodes: std.ArrayList(Node) = .empty,
    properties: std.ArrayList(Property) = .empty,
    alternatives: std.ArrayList(u32) = .empty,
    tries: std.ArrayList(Trie) = .empty,
    edge_bytes: std.ArrayList(u8) = .empty,
    edge_nodes: std.ArrayList(u32) = .empty,
    /// The whole schema, for a `$ref` to find its target in.
    root: Value = .null,
    /// A `$ref`'s pointer to its node, put there before the target is compiled: a schema that names itself
    /// finds the node it is being compiled into.
    refs: std.StringHashMapUnmanaged(u32) = .empty,
    /// Where each choice came from, for the refusal of one whose alternatives begin alike.
    origins: std.ArrayList(Origin) = .empty,
    /// The nodes of schemas without a `type` whose keywords said what they are (`kindless`).
    narrowed: std.AutoHashMapUnmanaged(u32, void) = .empty,
    options: Options = .{},

    const Word = struct { bytes: []const u8, value: u32 };
    const Origin = struct { node: u32, path: []const u8, keyword: []const u8 };

    fn node(self: *Builder, made: Node) Allocator.Error!u32 {
        try self.nodes.append(self.arena, made);
        return @intCast(self.nodes.items.len - 1);
    }

    /// The trie of `words`, which this sorts; the same bytes twice keep the first value.
    fn trie(self: *Builder, words: []Word) Allocator.Error!u32 {
        std.mem.sort(Word, words, {}, struct {
            fn lessThan(_: void, left: Word, right: Word) bool {
                return std.mem.lessThan(u8, left.bytes, right.bytes);
            }
        }.lessThan);
        return self.trieAt(words, 0);
    }

    /// The trie of sorted `words` that share their first `depth` bytes.
    fn trieAt(self: *Builder, words: []const Word, depth: usize) Allocator.Error!u32 {
        var rest = words;
        var value: u32 = none;
        while (rest.len > 0 and rest[0].bytes.len == depth) : (rest = rest[1..]) {
            if (value == none) value = rest[0].value;
        }
        var fan: u32 = 0;
        for (rest, 0..) |word, at| {
            if (at == 0 or rest[at - 1].bytes[depth] != word.bytes[depth]) fan += 1;
        }
        const made: u32 = @intCast(self.tries.items.len);
        const edges: u32 = @intCast(self.edge_bytes.items.len);
        try self.tries.append(self.arena, .{ .edges = edges, .fan = fan, .value = value });
        _ = try self.edge_bytes.addManyAsSlice(self.arena, fan);
        _ = try self.edge_nodes.addManyAsSlice(self.arena, fan);
        var edge = edges;
        var start: usize = 0;
        while (start < rest.len) : (edge += 1) {
            const byte = rest[start].bytes[depth];
            var stop = start + 1;
            while (stop < rest.len and rest[stop].bytes[depth] == byte) stop += 1;
            const under = try self.trieAt(rest[start..stop], depth + 1);
            self.edge_bytes.items[edge] = byte;
            self.edge_nodes.items[edge] = under;
            start = stop;
        }
        return made;
    }

    fn literals(self: *Builder, texts: []const []const u8) Allocator.Error!u32 {
        const words = try self.arena.alloc(Word, texts.len);
        for (words, texts) |*word, text| word.* = .{ .bytes = text, .value = 0 };
        return self.node(.{ .kind = .literals, .first = try self.trie(words) });
    }

    fn fixedNodes(self: *Builder) Allocator.Error!void {
        std.debug.assert(try self.node(.{ .kind = .any }) == fixed.any);
        std.debug.assert(try self.node(.{ .kind = .string }) == fixed.string);
        std.debug.assert(try self.node(.{ .kind = .number }) == fixed.number);
        std.debug.assert(try self.node(.{ .kind = .integer }) == fixed.integer);
        std.debug.assert(try self.node(.{ .kind = .object, .other = fixed.any }) == fixed.object);
        std.debug.assert(try self.node(.{ .kind = .array, .first = fixed.any }) == fixed.array);
        std.debug.assert(try self.literals(&.{ "true", "false" }) == fixed.boolean);
        std.debug.assert(try self.literals(&.{"null"}) == fixed.null_);
        std.debug.assert(try self.literals(&.{ "true", "false", "null" }) == fixed.words);
    }

    /// A value as JSON text with nothing between its tokens: an enum's value, a property's key.
    fn compact(self: *Builder, value: Value) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(self.arena, "{f}", .{std.json.fmt(value, .{})});
    }

    /// The path of a schema inside another, for a refusal to name: its end, where it is long, since a refusal
    /// has room for a few hundred bytes and says why after it says where.
    fn below(self: *Builder, comptime format: []const u8, args: anytype) Allocator.Error![]const u8 {
        const whole = try std.fmt.allocPrint(self.arena, format, args);
        const most = 120;
        if (whole.len <= most) return whole;
        var from = whole.len - most;
        // not from the middle of a character
        while (from < whole.len and whole[from] & 0xc0 == 0x80) from += 1;
        return std.fmt.allocPrint(self.arena, "...{s}", .{whole[from..]});
    }

    fn compile(self: *Builder, schema: Value, path: []const u8, depth: usize) SchemaError!u32 {
        if (depth > max_depth) return self.diagnostic.fail(error.UnsupportedSchema, "{s}: nested deeper than {d} schemas", .{ path, max_depth });
        const object = switch (schema) {
            .bool => |anything| {
                if (anything) return fixed.any;
                return self.diagnostic.fail(error.UnsupportedSchema, "{s}: the schema `false` takes no value", .{path});
            },
            .object => |object| object,
            else => return self.diagnostic.fail(error.InvalidSchema, "{s}: a schema is an object, true or false", .{path}),
        };
        keys: for (object.keys()) |key| {
            for (keywords ++ annotations ++ definitions) |known| {
                if (std.mem.eql(u8, key, known)) continue :keys;
            }
            return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `{s}` is not read", .{ path, key });
        }
        // these stand for the whole schema: what else it says would have to hold too, and is not merged
        for ([_][]const u8{ "$ref", "anyOf", "oneOf" }) |whole| {
            const value = object.get(whole) orelse continue;
            others: for (object.keys()) |key| {
                if (std.mem.eql(u8, key, whole)) continue;
                for (annotations ++ definitions) |known| {
                    if (std.mem.eql(u8, key, known)) continue :others;
                }
                return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `{s}` beside `{s}` is not read", .{ path, key, whole });
            }
            if (std.mem.eql(u8, whole, "$ref")) return self.reference(value, path, depth);
            return self.branches(value, whole, path, depth);
        }
        // `enum` and `const` name the values themselves. A `type` beside them must take each one; what else
        // the schema says would have to hold of them too, and is not checked.
        for ([_][]const u8{ "enum", "const" }) |whole| {
            const given = object.get(whole) orelse continue;
            others: for (object.keys()) |key| {
                if (std.mem.eql(u8, key, whole) or std.mem.eql(u8, key, "type")) continue;
                for (annotations ++ definitions) |known| {
                    if (std.mem.eql(u8, key, known)) continue :others;
                }
                return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `{s}` beside `{s}` is not read", .{ path, key, whole });
            }
            const values: []const Value = if (std.mem.eql(u8, whole, "const")) &.{given} else switch (given) {
                .array => |list| list.items,
                else => &.{},
            };
            if (values.len == 0) return self.diagnostic.fail(error.InvalidSchema, "{s}: `enum` is a list of at least one value", .{path});
            const texts = try self.arena.alloc([]const u8, values.len);
            for (texts, values) |*text, value| {
                if (object.get("type")) |named| {
                    if (!try self.takes(named, value, path)) {
                        return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `{s}` has a value its `type` does not take", .{ path, whole });
                    }
                }
                text.* = try self.compact(value);
                if (text.len > self.options.max_literal) {
                    return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `{s}` has a value of {d} bytes: more than the {d} read", .{ path, whole, text.len, self.options.max_literal });
                }
            }
            return self.literals(texts);
        }

        const named = object.get("type") orelse {
            // Without a `type` the keywords say what the value is: an object's, an array's. Both at once would
            // be a value that is either, each held to its own keywords.
            const object_words = [_][]const u8{ "properties", "required", "additionalProperties" };
            const array_words = [_][]const u8{ "items", "minItems", "maxItems" };
            const of_object = for (object_words) |word| {
                if (object.get(word) != null) break word;
            } else null;
            const of_array = for (array_words) |word| {
                if (object.get(word) != null) break word;
            } else null;
            if (of_object != null and of_array != null) {
                return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `{s}` beside `{s}` without a `type` is not read", .{ path, of_array.?, of_object.? });
            }
            if (of_object != null) return self.kindless(try self.objectNode(object, path, depth));
            if (of_array != null) return self.kindless(try self.arrayNode(object, path, depth));
            return fixed.any;
        };
        switch (named) {
            .string => |name| return self.typed(name, object, path, depth),
            .array => |names| {
                if (names.items.len == 0) return self.diagnostic.fail(error.InvalidSchema, "{s}: `type` is an empty list", .{path});
                var made: std.ArrayList(u32) = .empty;
                var number = false;
                for (names.items) |name| number = number or (name == .string and std.mem.eql(u8, name.string, "number"));
                for (names.items) |name| {
                    if (name != .string) return self.diagnostic.fail(error.InvalidSchema, "{s}: `type` lists a value that is not a name", .{path});
                    // every integer is a number
                    if (number and std.mem.eql(u8, name.string, "integer")) continue;
                    try made.append(self.arena, try self.typed(name.string, object, path, depth));
                }
                return self.choice(made.items, path, "type");
            },
            else => return self.diagnostic.fail(error.InvalidSchema, "{s}: `type` is a name or a list of names", .{path}),
        }
    }

    /// The node of a schema with no `type`, made as the object or the array its keywords describe. JSON Schema
    /// lets every value of another kind pass such a schema; the masks take the described kind alone, which is
    /// fewer documents and all of them valid, except under `oneOf` (`checkChoices`). The node is a copy, so
    /// that it is known from the same schema with its `type`.
    fn kindless(self: *Builder, id: u32) Allocator.Error!u32 {
        const copy = try self.node(self.nodes.items[id]);
        try self.narrowed.put(self.arena, copy, {});
        return copy;
    }

    /// Whether `type` (a name or a list of names) takes `value`: for an `enum` beside it.
    fn takes(self: *Builder, named: Value, value: Value, path: []const u8) SchemaError!bool {
        const names: []const Value = switch (named) {
            .string => &.{named},
            .array => |list| list.items,
            else => return self.diagnostic.fail(error.InvalidSchema, "{s}: `type` is a name or a list of names", .{path}),
        };
        var taken = false;
        for (names) |name| {
            if (name != .string) return self.diagnostic.fail(error.InvalidSchema, "{s}: `type` lists a value that is not a name", .{path});
            const of = std.meta.stringToEnum(enum { string, number, integer, boolean, null, object, array }, name.string) orelse {
                return self.diagnostic.fail(error.InvalidSchema, "{s}: `{s}` is not a type", .{ path, name.string });
            };
            taken = taken or switch (of) {
                .string => value == .string,
                .number => value == .integer or value == .float or value == .number_string,
                .integer => value == .integer or (value == .float and @floor(value.float) == value.float),
                .boolean => value == .bool,
                .null => value == .null,
                .object => value == .object,
                .array => value == .array,
            };
        }
        return taken;
    }

    /// One of `alternatives`, which `checkChoices` holds to different first bytes.
    fn choice(self: *Builder, alternatives: []const u32, path: []const u8, keyword: []const u8) Allocator.Error!u32 {
        if (alternatives.len == 1) return alternatives[0];
        const first: u32 = @intCast(self.alternatives.items.len);
        try self.alternatives.appendSlice(self.arena, alternatives);
        const made = try self.node(.{ .kind = .choice, .first = first, .count = @intCast(alternatives.len) });
        try self.origins.append(self.arena, .{ .node = made, .path = path, .keyword = keyword });
        return made;
    }

    /// `anyOf` or `oneOf`: one of its branches. Branches that begin with different bytes cannot both take a
    /// document, so the two keywords ask the same.
    fn branches(self: *Builder, value: Value, keyword: []const u8, path: []const u8, depth: usize) SchemaError!u32 {
        if (value != .array or value.array.items.len == 0) {
            return self.diagnostic.fail(error.InvalidSchema, "{s}: `{s}` is a list of at least one schema", .{ path, keyword });
        }
        var made: std.ArrayList(u32) = .empty;
        for (value.array.items, 0..) |branch, at| {
            const under = try self.compile(branch, try self.below("{s}/{s}/{d}", .{ path, keyword, at }), depth + 1);
            try made.append(self.arena, under);
        }
        if (std.mem.eql(u8, keyword, "anyOf") and std.mem.findScalar(u32, made.items, fixed.number) != null) {
            // every integer is a number
            while (std.mem.findScalar(u32, made.items, fixed.integer)) |at| _ = made.orderedRemove(at);
        }
        return self.choice(made.items, path, keyword);
    }

    /// A `$ref`: the node of the schema it points to, compiled once however often it is named.
    fn reference(self: *Builder, value: Value, path: []const u8, depth: usize) SchemaError!u32 {
        if (value != .string) return self.diagnostic.fail(error.InvalidSchema, "{s}: `$ref` is a string", .{path});
        const pointer = value.string;
        if (self.refs.get(pointer)) |made| return made;
        if (pointer.len == 0 or pointer[0] != '#') {
            return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `$ref` to `{s}` is not read: only a place in the schema itself (`#/...`)", .{ path, pointer });
        }
        const target = try self.find(pointer) orelse {
            return self.diagnostic.fail(error.InvalidSchema, "{s}: `$ref` names `{s}`, which the schema does not have", .{ path, pointer });
        };
        const placeholder = try self.node(.{ .kind = .ref });
        try self.refs.put(self.arena, pointer, placeholder);
        const made = try self.compile(target, try self.below("{s}", .{pointer}), depth + 1);
        self.nodes.items[placeholder].first = made;
        return placeholder;
    }

    /// The value a JSON pointer (RFC 6901, after its `#`) names in the schema.
    fn find(self: *Builder, pointer: []const u8) Allocator.Error!?Value {
        var at = self.root;
        if (pointer.len == 1) return at;
        if (pointer[1] != '/') return null;
        var parts = std.mem.splitScalar(u8, pointer[2..], '/');
        while (parts.next()) |escaped| {
            const tilde = try std.mem.replaceOwned(u8, self.arena, escaped, "~1", "/");
            const part = try std.mem.replaceOwned(u8, self.arena, tilde, "~0", "~");
            at = switch (at) {
                .object => |object| object.get(part) orelse return null,
                .array => |array| array: {
                    const index = std.fmt.parseInt(usize, part, 10) catch return null;
                    if (index >= array.items.len) return null;
                    break :array array.items[index];
                },
                else => return null,
            };
        }
        return at;
    }

    /// The node `id` stands for, through any `$ref`s.
    fn resolved(self: *Builder, id: u32) SchemaError!u32 {
        var at = id;
        var hops: usize = 0;
        while (at != none and self.nodes.items[at].kind == .ref) : (hops += 1) {
            if (hops > self.nodes.items.len) return self.diagnostic.fail(error.InvalidSchema, "a `$ref` leads only to itself", .{});
            at = self.nodes.items[at].first;
        }
        return at;
    }

    /// Writes every `$ref` out of the schema: whatever named one names its target.
    fn resolve(self: *Builder, root: u32) SchemaError!u32 {
        for (self.properties.items) |*property| property.node = try self.resolved(property.node);
        for (self.alternatives.items) |*alternative| alternative.* = try self.resolved(alternative.*);
        for (self.nodes.items) |*made| switch (made.kind) {
            .array => made.first = try self.resolved(made.first),
            .object => made.other = try self.resolved(made.other),
            else => {},
        };
        return self.resolved(root);
    }

    fn typed(self: *Builder, name: []const u8, object: std.json.ObjectMap, path: []const u8, depth: usize) SchemaError!u32 {
        const names = [_][]const u8{ "string", "number", "integer", "boolean", "null", "object", "array" };
        const which = for (names, 0..) |known, at| {
            if (std.mem.eql(u8, name, known)) break at;
        } else return self.diagnostic.fail(error.InvalidSchema, "{s}: `{s}` is not a type", .{ path, name });
        return switch (which) {
            0 => fixed.string,
            1 => fixed.number,
            2 => fixed.integer,
            3 => fixed.boolean,
            4 => fixed.null_,
            5 => self.objectNode(object, path, depth),
            6 => self.arrayNode(object, path, depth),
            else => unreachable,
        };
    }

    fn objectNode(self: *Builder, object: std.json.ObjectMap, path: []const u8, depth: usize) SchemaError!u32 {
        const others = object.get("additionalProperties");
        const listed: ?std.json.ObjectMap = if (object.get("properties")) |properties| switch (properties) {
            .object => |map| if (map.count() == 0) null else map,
            else => return self.diagnostic.fail(error.InvalidSchema, "{s}: `properties` is an object", .{path}),
        } else null;
        const required: []const Value = if (object.get("required")) |list| switch (list) {
            .array => |array| array.items,
            else => return self.diagnostic.fail(error.InvalidSchema, "{s}: `required` is a list of names", .{path}),
        } else &.{};

        const properties = listed orelse {
            if (required.len != 0) return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `required` names keys that `properties` does not", .{path});
            const other: u32 = if (others) |value| switch (value) {
                .bool => |any| if (any) fixed.any else none,
                else => try self.compile(value, try self.below("{s}/additionalProperties", .{path}), depth + 1),
                // `properties` that lists nothing still says no other key is taken: `{}` alone, as xgrammar reads
                // it. With no `properties` at all the object is open.
            } else if (object.get("properties") != null) none else fixed.any;
            if (other == fixed.any) return fixed.object;
            return self.node(.{ .kind = .object, .other = other });
        };
        if (others) |value| {
            if (value != .bool or value.bool) {
                return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `additionalProperties` beside `properties` is not read", .{path});
            }
        }
        const count: u32 = @intCast(properties.count());
        const first: u32 = @intCast(self.properties.items.len);
        _ = try self.properties.addManyAsSlice(self.arena, count);
        const made = try self.node(.{ .kind = .object, .first = first, .count = count });
        const keys = try self.arena.alloc([]const u8, count);
        const musts = try self.arena.alloc(bool, count);
        @memset(musts, false);
        for (required) |name| {
            if (name != .string) return self.diagnostic.fail(error.InvalidSchema, "{s}: `required` is a list of names", .{path});
            const at = properties.getIndex(name.string) orelse {
                return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `required` names `{s}`, which `properties` does not", .{ path, name.string });
            };
            musts[at] = true;
        }
        for (properties.keys(), properties.values(), 0..) |key, value, at| {
            keys[at] = try self.compact(.{ .string = key });
            if (keys[at].len > self.options.max_literal) {
                return self.diagnostic.fail(error.UnsupportedSchema, "{s}: a property's name is {d} bytes: more than the {d} read", .{ path, keys[at].len, self.options.max_literal });
            }
            const under = try self.compile(value, try self.below("{s}/{s}", .{ path, key }), depth + 1);
            self.properties.items[first + at] = .{ .node = under, .required = musts[at], .keys = none };
        }
        for (0..count) |from| {
            var words: std.ArrayList(Builder.Word) = .empty;
            for (from..count) |at| {
                try words.append(self.arena, .{ .bytes = keys[at], .value = @intCast(at) });
                if (musts[at]) break;
            }
            self.properties.items[first + from].keys = try self.trie(words.items);
        }
        return made;
    }

    fn arrayNode(self: *Builder, object: std.json.ObjectMap, path: []const u8, depth: usize) SchemaError!u32 {
        const items: u32 = if (object.get("items")) |value|
            try self.compile(value, try self.below("{s}/items", .{path}), depth + 1)
        else
            fixed.any;
        const least = try self.bound(object, "minItems", path) orelse 0;
        const most = try self.bound(object, "maxItems", path) orelse none;
        if (most < least) return self.diagnostic.fail(error.InvalidSchema, "{s}: `maxItems` is less than `minItems`", .{path});
        if (items == fixed.any and least == 0 and most == none) return fixed.array;
        return self.node(.{ .kind = .array, .first = items, .count = least, .other = most });
    }

    fn bound(self: *Builder, object: std.json.ObjectMap, key: []const u8, path: []const u8) SchemaError!?u32 {
        const value = object.get(key) orelse return null;
        if (value != .integer or value.integer < 0 or value.integer > 1 << 16) {
            return self.diagnostic.fail(error.InvalidSchema, "{s}: `{s}` is a count from 0 to 65536", .{ path, key });
        }
        return @intCast(value.integer);
    }

    /// A choice's alternatives must differ in their first byte: the position stays one stack. And no choice may
    /// be its own alternative, straight or through others: such a value has no first byte to tell it by.
    fn checkChoices(self: *Builder, schema: *const Schema) SchemaError!void {
        const seen = try self.arena.alloc(Seen, schema.nodes.len);
        @memset(seen, .no);
        for (self.origins.items) |origin| try self.acyclic(schema, origin, origin.node, seen);
        // `oneOf` asks that no other alternative pass. Alternatives that begin with different bytes are values
        // of different kinds, and each passes its own alone, unless one has no `type`: then values of every
        // other kind pass it too, and a document the masks take would pass two.
        for (self.origins.items) |origin| {
            if (!std.mem.eql(u8, origin.keyword, "oneOf")) continue;
            if (self.anyKindless(schema, origin.node)) {
                return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `oneOf` has an alternative without a `type`, which values of other kinds pass too", .{origin.path});
            }
        }
        for (self.origins.items) |origin| {
            const made = schema.nodes[origin.node];
            const alternatives = schema.alternatives[made.first..][0..made.count];
            for (0..256) |byte| {
                var takers: usize = 0;
                for (alternatives) |alternative| takers += @intFromBool(schema.begins(alternative, @intCast(byte)));
                if (takers > 1) {
                    return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `{s}` has alternatives that both begin with `{c}`: they are told apart by their first byte alone", .{ origin.path, origin.keyword, @as(u8, @intCast(byte)) });
                }
            }
        }
    }

    /// Whether `at`, or an alternative of it through any choices, is a schema without a `type`. No choice is
    /// its own alternative here (`acyclic`).
    fn anyKindless(self: *Builder, schema: *const Schema, at: u32) bool {
        if (self.narrowed.contains(at)) return true;
        const made = schema.nodes[at];
        if (made.kind != .choice) return false;
        for (schema.alternatives[made.first..][0..made.count]) |alternative| {
            if (self.anyKindless(schema, alternative)) return true;
        }
        return false;
    }

    const Seen = enum { no, inside, yes };

    fn acyclic(self: *Builder, schema: *const Schema, origin: Origin, at: u32, seen: []Seen) SchemaError!void {
        const made = schema.nodes[at];
        if (made.kind != .choice or seen[at] == .yes) return;
        if (seen[at] == .inside) {
            return self.diagnostic.fail(error.UnsupportedSchema, "{s}: `{s}` has itself as an alternative", .{ origin.path, origin.keyword });
        }
        seen[at] = .inside;
        for (schema.alternatives[made.first..][0..made.count]) |alternative| try self.acyclic(schema, origin, alternative, seen);
        seen[at] = .yes;
    }
};

pub fn parseSchema(gpa: Allocator, text: []const u8, options: Options, diagnostic: *Diagnostic) SchemaError!Schema {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena_state.deinit();
    const arena = arena_state.allocator();
    if (text.len > options.max_schema_bytes) {
        return diagnostic.fail(error.UnsupportedSchema, "the schema is {d} bytes: more than the {d} read", .{ text.len, options.max_schema_bytes });
    }
    // What reads the schema after this follows its nesting: the bound comes first.
    var nesting: usize = 0;
    var in_string = false;
    var escaped = false;
    for (text) |byte| {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (byte == '\\') {
                escaped = true;
            } else if (byte == '"') in_string = false;
            continue;
        }
        switch (byte) {
            '"' => in_string = true,
            '{', '[' => {
                nesting += 1;
                if (nesting > options.max_nesting) {
                    return diagnostic.fail(error.UnsupportedSchema, "the schema's JSON nests deeper than {d}", .{options.max_nesting});
                }
            },
            '}', ']' => nesting -|= 1,
            else => {},
        }
    }
    const value = std.json.parseFromSliceLeaky(Value, arena, text, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return diagnostic.fail(error.InvalidSchema, "the schema is not JSON: {s}", .{@errorName(err)}),
    };
    var builder: Builder = .{ .arena = arena, .diagnostic = diagnostic, .root = value, .options = options };
    try builder.fixedNodes();
    const root = try builder.resolve(try builder.compile(value, "#", 0));
    var schema: Schema = .{
        .arena = undefined,
        .nodes = builder.nodes.items,
        .properties = builder.properties.items,
        .alternatives = builder.alternatives.items,
        .tries = builder.tries.items,
        .edge_bytes = builder.edge_bytes.items,
        .edge_nodes = builder.edge_nodes.items,
        .root = root,
        .blanks = options.blanks,
    };
    try builder.checkChoices(&schema);
    schema.arena = arena_state;
    return schema;
}
