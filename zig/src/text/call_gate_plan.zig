//! `call_gate.zig`'s reading of a tool's schema in the template's form (spec 104 WS-06 moved it out of
//! `call_gate.zig`): which parameters a call is written with, and which schemas the form cannot hold. Host only.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const automaton = @import("call_gate_strings.zig").automaton;
const template = @import("call_gate.zig").template;

/// One parameter as the template's form holds it.
pub const Plan = struct {
    key: []const u8,
    required: bool,
    value: union(enum) {
        /// Any string.
        raw,
        /// One of these strings.
        choice: []const []const u8,
        /// A document of this schema.
        json: []const u8,
    },
};

/// The annotations `json_schema.zig` reads past.
const annotations = [_][]const u8{ "description", "title", "default", "examples", "$schema", "$id", "$comment", "deprecated", "readOnly", "writeOnly" };

fn annotation(key: []const u8) bool {
    for (annotations) |known| {
        if (std.mem.eql(u8, known, key)) return true;
    }
    return false;
}

/// Whether the reply's parser reads `name` back from `<function=NAME>` or `<parameter=NAME>`: its `[^>\s]+`,
/// kept here to the printable ASCII.
pub fn nameable(name: []const u8) bool {
    if (name.len == 0) return false;
    for (name) |byte| {
        if (byte <= ' ' or byte >= 0x7f or byte == '>') return false;
    }
    return true;
}

fn refers(value: Value) bool {
    switch (value) {
        .object => |object| {
            if (object.contains("$ref")) return true;
            for (object.values()) |inner| {
                if (refers(inner)) return true;
            }
        },
        .array => |array| for (array.items) |inner| {
            if (refers(inner)) return true;
        },
        else => {},
    }
    return false;
}

/// `parameters`, a tool's schema the engine has taken, as the parameters of the template's form; null where the
/// form cannot hold a call to it (the file's header says which schemas). The types are the ones
/// the reply's parser reads a value as (TensorFold's `decode_parameter`).
pub fn planOf(arena: Allocator, parameters: []const u8) Allocator.Error!?[]Plan {
    const parsed = std.json.parseFromSliceLeaky(Value, arena, parameters, .{}) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        else => return null,
    };
    if (parsed != .object) return null;
    const schema = parsed.object;
    var properties: ?std.json.ObjectMap = null;
    var required: []const Value = &.{};
    for (schema.keys(), schema.values()) |key, value| {
        if (std.mem.eql(u8, key, "type")) {
            if (value != .string or !std.mem.eql(u8, value.string, "object")) return null;
        } else if (std.mem.eql(u8, key, "properties")) {
            if (value != .object) return null;
            properties = value.object;
        } else if (std.mem.eql(u8, key, "required")) {
            if (value != .array) return null;
            required = value.array.items;
        } else if (std.mem.eql(u8, key, "additionalProperties")) {
            // beside `properties` the engine takes no other key, whatever this says
            if (value != .bool) return null;
        } else if (!annotation(key)) return null;
    }
    const listed = properties orelse return null;
    if (!schema.contains("type") or listed.count() > template.max_parameters) return null;
    const plans = try arena.alloc(Plan, listed.count());
    for (plans, listed.keys(), listed.values()) |*plan, key, value| {
        if (!nameable(key) or value != .object) return null;
        const property = value.object;
        const named = property.get("type") orelse return null;
        if (named != .string) return null;
        plan.* = .{ .key = key, .required = false, .value = .raw };
        for (required) |name| {
            if (name != .string) return null;
            if (std.mem.eql(u8, name.string, key)) plan.required = true;
        }
        if (std.mem.eql(u8, named.string, "string")) {
            var values: ?[]const Value = null;
            for (property.keys(), property.values()) |keyword, given| {
                if (std.mem.eql(u8, keyword, "type") or annotation(keyword)) continue;
                if (values != null) return null;
                if (std.mem.eql(u8, keyword, "enum") and given == .array) {
                    values = given.array.items;
                } else if (std.mem.eql(u8, keyword, "const")) {
                    values = try arena.dupe(Value, &.{given});
                } else return null;
            }
            const among = values orelse continue;
            if (among.len == 0) return null;
            const texts = try arena.alloc([]const u8, among.len);
            for (texts, among) |*written, one| {
                if (one != .string or !automaton.writable(one.string)) return null;
                written.* = one.string;
            }
            plan.value = .{ .choice = texts };
            continue;
        }
        const kinds = [_][]const u8{ "integer", "number", "boolean", "null", "array", "object" };
        for (kinds) |kind| {
            if (std.mem.eql(u8, kind, named.string)) break;
        } else return null;
        if (refers(value)) return null;
        plan.value = .{ .json = try std.json.Stringify.valueAlloc(arena, value, .{}) };
    }
    return plans;
}
