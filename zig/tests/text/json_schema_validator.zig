//! `Validator` (spec 104 WS-06 moved it out of `json_schema.zig`, spec 121 out of `zig/src/`): a second reading
//! of a schema, written apart from the masks, for the tests. Host only.

const std = @import("std");
const Value = std.json.Value;

/// Whether a document is what a schema asks, by the keywords `json_schema.zig` reads: written apart from the machine,
/// from the schema's JSON and the document's, as the reference the tests hold the masks to. Not on any decode
/// path. It asks what the masks ask and JSON Schema does not: an object with `properties` has no other key.
pub const Validator = struct {
    /// The whole schema: where a `$ref` is looked up.
    root: Value,

    pub fn valid(self: Validator, schema: Value, document: Value) bool {
        return self.check(schema, document, 0);
    }

    fn check(self: Validator, schema: Value, document: Value, depth: usize) bool {
        // a `$ref` chain that never reaches a schema
        if (depth > 256) return false;
        const object = switch (schema) {
            .bool => |anything| return anything,
            .object => |object| object,
            else => return false,
        };
        if (object.get("$ref")) |pointer| return self.check(self.target(pointer.string) orelse return false, document, depth + 1);
        if (object.get("anyOf")) |branches| {
            for (branches.array.items) |branch| {
                if (self.check(branch, document, depth + 1)) return true;
            }
            return false;
        }
        if (object.get("oneOf")) |branches| {
            var count: usize = 0;
            for (branches.array.items) |branch| count += @intFromBool(self.check(branch, document, depth + 1));
            return count == 1;
        }
        if (object.get("enum")) |values| {
            for (values.array.items) |value| {
                if (same(value, document)) return true;
            }
            return false;
        }
        if (object.get("const")) |value| return same(value, document);
        if (object.get("type")) |named| switch (named) {
            .string => |name| if (!is(name, document)) return false,
            .array => |names| {
                for (names.items) |name| {
                    if (is(name.string, document)) break;
                } else return false;
            },
            else => return false,
        };
        switch (document) {
            .object => |members| {
                const listed: ?std.json.ObjectMap = if (object.get("properties")) |properties| properties.object else null;
                if (listed != null and listed.?.count() > 0) {
                    for (members.keys(), members.values()) |key, value| {
                        if (!self.check(listed.?.get(key) orelse return false, value, depth + 1)) return false;
                    }
                    if (object.get("required")) |required| for (required.array.items) |name| {
                        if (!members.contains(name.string)) return false;
                    };
                } else if (object.get("additionalProperties")) |others| {
                    for (members.values()) |value| {
                        if (!self.check(others, value, depth + 1)) return false;
                    }
                }
            },
            .array => |items| {
                if (object.get("minItems")) |least| if (items.items.len < least.integer) return false;
                if (object.get("maxItems")) |most| if (items.items.len > most.integer) return false;
                if (object.get("items")) |each| for (items.items) |item| {
                    if (!self.check(each, item, depth + 1)) return false;
                };
            },
            else => {},
        }
        return true;
    }

    fn is(name: []const u8, document: Value) bool {
        const whole = switch (document) {
            .integer => true,
            // an integer past 64 bits is kept as its text
            .number_string => |text| std.mem.findAny(u8, text, ".eE") == null,
            else => false,
        };
        if (std.mem.eql(u8, name, "integer")) return whole;
        if (std.mem.eql(u8, name, "number")) return whole or document == .float or document == .number_string;
        if (std.mem.eql(u8, name, "string")) return document == .string;
        if (std.mem.eql(u8, name, "boolean")) return document == .bool;
        if (std.mem.eql(u8, name, "null")) return document == .null;
        if (std.mem.eql(u8, name, "object")) return document == .object;
        if (std.mem.eql(u8, name, "array")) return document == .array;
        return false;
    }

    /// Two JSON values that are the same value.
    fn same(left: Value, right: Value) bool {
        switch (left) {
            .null => return right == .null,
            .bool => |value| return right == .bool and right.bool == value,
            .integer => |value| return switch (right) {
                .integer => |other| other == value,
                .float => |other| other == @as(f64, @floatFromInt(value)),
                else => false,
            },
            .float => |value| return switch (right) {
                .integer => |other| value == @as(f64, @floatFromInt(other)),
                .float => |other| other == value,
                else => false,
            },
            .number_string => |value| return right == .number_string and std.mem.eql(u8, value, right.number_string),
            .string => |value| return right == .string and std.mem.eql(u8, value, right.string),
            .array => |items| {
                if (right != .array or right.array.items.len != items.items.len) return false;
                for (items.items, right.array.items) |a, b| {
                    if (!same(a, b)) return false;
                }
                return true;
            },
            .object => |members| {
                if (right != .object or right.object.count() != members.count()) return false;
                for (members.keys(), members.values()) |key, value| {
                    if (!same(value, right.object.get(key) orelse return false)) return false;
                }
                return true;
            },
        }
    }

    /// What a pointer of `#`, `/` and names leads to (RFC 6901's `~0` and `~1` in a name).
    pub fn target(self: Validator, pointer: []const u8) ?Value {
        var at = self.root;
        var rest = pointer;
        if (rest.len == 0 or rest[0] != '#') return null;
        rest = rest[1..];
        var name: [256]u8 = undefined;
        while (rest.len > 0) {
            if (rest[0] != '/') return null;
            const end = std.mem.findScalarPos(u8, rest, 1, '/') orelse rest.len;
            var len: usize = 0;
            var i: usize = 1;
            while (i < end) : (i += 1) {
                if (len == name.len) return null;
                var byte = rest[i];
                if (byte == '~' and i + 1 < end) {
                    i += 1;
                    byte = if (rest[i] == '1') '/' else '~';
                }
                name[len] = byte;
                len += 1;
            }
            at = switch (at) {
                .object => |object| object.get(name[0..len]) orelse return null,
                .array => |items| items.items[std.fmt.parseInt(usize, name[0..len], 10) catch return null],
                else => return null,
            };
            rest = rest[end..];
        }
        return at;
    }
};
