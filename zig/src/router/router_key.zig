//! The router's key of a request (spec 088; spec 104 WS-06 moved it out of `router.zig`): the API a target is,
//! the hash of a body's opening (the system, the tools and the first user message of chat completions and
//! Messages; the instructions, tools and first user input of Responses), and the fields a Responses
//! conversation is followed by.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Api = enum { chat, messages, responses, completions, none };

/// The API a target is one of, by its last path element; `none` when the router keys nothing about it.
pub fn apiOf(target: []const u8) Api {
    if (std.mem.endsWith(u8, target, "/chat/completions")) return .chat;
    if (std.mem.endsWith(u8, target, "/responses")) return .responses;
    if (std.mem.endsWith(u8, target, "/messages")) return .messages;
    if (std.mem.endsWith(u8, target, "/completions")) return .completions;
    return .none;
}

/// The key of a body's opening, or null when the body is not one of the three APIs' JSON object.
pub fn openingKey(arena: Allocator, api: Api, value: std.json.Value) ?u64 {
    if (value != .object) return null;
    switch (api) {
        .none => return null,
        .completions => {
            var h = std.hash.Wyhash.init(0);
            h.update("completions\x00");
            const prompt = value.object.get("prompt") orelse return null;
            canonicalHash(&h, prompt);
            return h.final();
        },
        else => {
            var h = std.hash.Wyhash.init(0);
            h.update(@tagName(api));
            h.update("\x00system\x00");
            h.update(systemOf(arena, api, value));
            h.update("\x00tools\x00");
            if (value.object.get("tools")) |tools| canonicalHash(&h, tools) else h.update("none");
            h.update("\x00user\x00");
            switch (firstUserOf(arena, api, value)) {
                .text => |text| h.update(text),
                .whole => |input| canonicalHash(&h, input),
                .none => h.update("none"),
            }
            return h.final();
        },
    }
}

const UserPart = union(enum) { text: []const u8, whole: std.json.Value, none };

fn systemOf(arena: Allocator, api: Api, value: std.json.Value) []const u8 {
    switch (api) {
        .chat => {
            var out: std.ArrayList(u8) = .empty;
            const messages = value.object.get("messages") orelse return "";
            switch (messages) {
                .array => |items| for (items.items) |item| {
                    if (item != .object) continue;
                    const role = item.object.get("role") orelse continue;
                    if (role != .string or !std.mem.eql(u8, role.string, "system")) continue;
                    if (out.items.len > 0) out.append(arena, '\n') catch return "";
                    out.appendSlice(arena, textOf(arena, item.object.get("content"))) catch return "";
                },
                else => {},
            }
            return out.items;
        },
        .messages => return textOf(arena, value.object.get("system")),
        .responses => return textOf(arena, value.object.get("instructions")),
        else => return "",
    }
}

fn firstUserOf(arena: Allocator, api: Api, value: std.json.Value) UserPart {
    switch (api) {
        .chat, .messages => {
            const messages = value.object.get("messages") orelse return .none;
            switch (messages) {
                .array => |items| for (items.items) |item| {
                    if (item != .object) continue;
                    const role = item.object.get("role") orelse continue;
                    if (role != .string or !std.mem.eql(u8, role.string, "user")) continue;
                    return .{ .text = textOf(arena, item.object.get("content")) };
                },
                else => {},
            }
            return .none;
        },
        .responses => {
            const input = value.object.get("input") orelse return .none;
            switch (input) {
                .string => |s| return .{ .text = s },
                .array => |items| for (items.items) |item| {
                    if (item != .object) continue;
                    const role = item.object.get("role") orelse continue;
                    if (role != .string or !std.mem.eql(u8, role.string, "user")) continue;
                    return .{ .text = textOf(arena, item.object.get("content")) };
                },
                else => {},
            }
            return .{ .whole = input };
        },
        else => return .none,
    }
}

/// The text of a content field: a string, or the text parts of the array it is, joined with newlines.
fn textOf(arena: Allocator, maybe: ?std.json.Value) []const u8 {
    const value = maybe orelse return "";
    switch (value) {
        .string => |s| return s,
        .array => |items| {
            var out: std.ArrayList(u8) = .empty;
            for (items.items) |item| {
                const text: ?[]const u8 = switch (item) {
                    .string => |s| s,
                    .object => |obj| switch (obj.get("text") orelse continue) {
                        .string => |s| s,
                        else => null,
                    },
                    else => null,
                };
                if (text) |t| {
                    if (out.items.len > 0) out.append(arena, '\n') catch return "";
                    out.appendSlice(arena, t) catch return "";
                }
            }
            return out.items;
        },
        else => return "",
    }
}

/// A value's hash as its shape and every leaf, in document order: the same JSON, the same hash.
fn canonicalHash(h: *std.hash.Wyhash, value: std.json.Value) void {
    var buf: [32]u8 = undefined;
    switch (value) {
        .null => h.update("n"),
        .bool => |b| h.update(if (b) "t" else "f"),
        .integer => |i| {
            h.update("i");
            h.update(std.fmt.bufPrint(&buf, "{d}", .{i}) catch "?");
        },
        .float => |f| {
            h.update("f");
            h.update(std.fmt.bufPrint(&buf, "{d}", .{f}) catch "?");
        },
        .number_string => |s| {
            h.update("#");
            h.update(s);
        },
        .string => |s| {
            h.update("s");
            h.update(std.fmt.bufPrint(&buf, "{d}:", .{s.len}) catch "?");
            h.update(s);
        },
        .array => |items| {
            h.update("a");
            h.update(std.fmt.bufPrint(&buf, "{d}", .{items.items.len}) catch "?");
            for (items.items) |item| canonicalHash(h, item);
        },
        .object => |obj| {
            h.update("o");
            h.update(std.fmt.bufPrint(&buf, "{d}", .{obj.count()}) catch "?");
            for (obj.keys(), obj.values()) |key, item| {
                h.update("k");
                h.update(std.fmt.bufPrint(&buf, "{d}:", .{key.len}) catch "?");
                h.update(key);
                canonicalHash(h, item);
            }
        },
    }
}

pub fn stringField(value: std.json.Value, name: []const u8) ?[]const u8 {
    if (value != .object) return null;
    const got = value.object.get(name) orelse return null;
    return switch (got) {
        .string => |s| s,
        else => null,
    };
}

/// The id of a target like `/v1/responses/resp_abc` or `/v1/responses/resp_abc/input`, after its last
/// `responses/` and before the next `/` or `?`.
pub fn responseIdOfTarget(target: []const u8) ?[]const u8 {
    const at = std.mem.indexOf(u8, target, "responses/") orelse return null;
    const rest = target[at + "responses/".len ..];
    const queried = if (std.mem.indexOfScalar(u8, rest, '?')) |q| rest[0..q] else rest;
    const path = if (std.mem.indexOfScalar(u8, queried, '/')) |s| queried[0..s] else queried;
    return if (path.len == 0) null else path;
}
