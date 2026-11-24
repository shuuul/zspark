//! What TensorFold does to a request's messages before the chat template sees them, and the names it gives the
//! reasoning effort. Ported from TensorFold 0.6.5: `server/messages.py` (`normalize_messages`,
//! `_normalize_tool_call_arguments`, `late_system_role`), `server/request_options.py` (`effort_levels`,
//! `coerce_effort`) and `cuda/chat_template.py` (`ChatTemplate.render`'s variables).
//!
//! Images and videos: with `allow_images`, a user or tool message whose content has visual parts keeps its parts
//! as they are, for the template to render their placeholders. Loading and encoding them is not done here.
//!
//! Tool-call arguments are parsed by `pyjson.zig`, Python's `json.loads`, so `NaN` and `Infinity` are read as
//! TensorFold reads them; its file lists what still differs.
//!
//! Everything returned is allocated in the caller's arena and may share strings and subtrees with the input.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const pyjson = @import("pyjson.zig");
const template = @import("template.zig");

pub const Error = error{ OutOfMemory, BadRequest };

/// Why a request was refused: TensorFold's `RequestError` text.
pub const Problem = struct { message: []const u8 = "" };

/// The role a system message after the first turn is given.
pub const LateSystem = enum { system, user };

const media = [_][]const u8{ "image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url" };
const roles = [_][]const u8{ "system", "developer", "user", "assistant", "tool" };
const visual = [_][]const u8{ "image_url", "image", "video_url", "video" };

pub const Options = struct {
    late: LateSystem,
    /// Whether the model takes images (and videos, where its frontend encodes them).
    allow_images: bool = false,
};

fn isVisual(part: Value) bool {
    if (part != .object) return false;
    const kind = part.object.get("type");
    for (visual) |name| if (isString(kind, name)) return true;
    return false;
}

fn refuse(problem: *Problem, message: []const u8) Error {
    problem.message = message;
    return error.BadRequest;
}

/// Python's truth value of a JSON value.
fn truthy(v: Value) bool {
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .number_string => true,
        .string => |s| s.len != 0,
        .array => |a| a.items.len != 0,
        .object => |o| o.count() != 0,
    };
}

fn hasMedia(o: ObjectMap) bool {
    for (media) |key| if (o.get(key)) |v| if (truthy(v)) return true;
    return false;
}

fn isString(v: ?Value, want: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, want);
}

/// `message` with `key` set, at the key's place when it is there already.
fn with(arena: Allocator, message: ObjectMap, key: []const u8, value: Value) Allocator.Error!ObjectMap {
    var copy = try message.clone(arena);
    try copy.put(arena, key, value);
    return copy;
}

/// The leading system and developer messages merged into one system message, later ones given `late` as their
/// role so the tokens before them stay as they were, and every content turned into a string.
pub fn normalize(arena: Allocator, messages: Value, options: Options, problem: *Problem) Error!Value {
    const late = options.late;
    if (messages != .array or messages.array.items.len == 0) return refuse(problem, "messages must be a non-empty list");
    var out: std.json.Array = .init(arena);
    var instructions: std.ArrayList(ObjectMap) = .empty;
    for (messages.array.items) |entry| {
        if (entry != .object) return refuse(problem, "each message must be an object");
        const message = entry.object;
        const role = message.get("role") orelse Value.null;
        const known = for (roles) |r| {
            if (isString(role, r)) break true;
        } else false;
        if (!known) return refuse(problem, "message role must be system, developer, user, assistant or tool");
        if (hasMedia(message)) {
            return refuse(problem, "this server accepts text only; image, audio and video inputs are unsupported");
        }
        var item = message;
        const content = message.get("content") orelse Value.null;
        if (options.allow_images and content == .array and for (content.array.items) |part| {
            if (isVisual(part)) break true;
        } else false) {
            if (!isString(role, "user") and !isString(role, "tool")) {
                return refuse(problem, "images and videos are supported only in user and tool messages");
            }
            for (content.array.items) |part| {
                const text = part == .object and isString(part.object.get("type"), "text");
                if (!text and !isVisual(part)) {
                    return refuse(problem, "image messages may contain text and image_url (or video_url) parts only");
                }
                if (text and (part.object.get("text") orelse Value.null) != .string) {
                    return refuse(problem, "a text content part must contain a text string");
                }
            }
            try out.append(entry);
            continue;
        }
        switch (content) {
            .string => {},
            .null => item = try with(arena, message, "content", .{ .string = "" }),
            .array => |parts| {
                var text: std.ArrayList(u8) = .empty;
                for (parts.items) |part| {
                    if (part != .object or !isString(part.object.get("type"), "text") or hasMedia(part.object)) {
                        return refuse(problem, "this server accepts text parts only; image, audio and video inputs are unsupported");
                    }
                    const piece = part.object.get("text") orelse Value.null;
                    if (piece != .string) return refuse(problem, "a text content part must contain a text string");
                    try text.appendSlice(arena, piece.string);
                }
                item = try with(arena, message, "content", .{ .string = text.items });
            },
            else => return refuse(problem, "message content must be text or an array of text parts"),
        }
        if (isString(role, "system") or isString(role, "developer")) {
            if (out.items.len == 0) {
                try instructions.append(arena, item);
                continue;
            }
            if (!isString(role, @tagName(late))) item = try with(arena, item, "role", .{ .string = @tagName(late) });
        }
        try out.append(.{ .object = item });
    }
    if (instructions.items.len != 0) {
        var joined: std.ArrayList(u8) = .empty;
        for (instructions.items, 0..) |instruction, i| {
            if (i != 0) try joined.appendSlice(arena, "\n\n");
            try joined.appendSlice(arena, instruction.get("content").?.string);
        }
        var first = try with(arena, instructions.items[0], "role", .{ .string = "system" });
        try first.put(arena, "content", .{ .string = joined.items });
        try out.insert(0, .{ .object = first });
    }
    return .{ .array = out };
}

/// Tool-call arguments as mappings, which is what templates iterate: a JSON string is parsed, and anything that
/// is not a mapping afterwards goes under `_invalid_arguments` as it came.
pub fn toolCallArguments(arena: Allocator, messages: Value) Allocator.Error!Value {
    var out: std.json.Array = .init(arena);
    for (messages.array.items) |entry| {
        const calls: Value = if (entry == .object) entry.object.get("tool_calls") orelse Value.null else Value.null;
        if (calls != .array or calls.array.items.len == 0) {
            try out.append(entry);
            continue;
        }
        var new_calls: std.json.Array = .init(arena);
        var touched = false;
        for (calls.array.items) |call| {
            const function: Value = if (call == .object) call.object.get("function") orelse Value.null else Value.null;
            const arguments: ?Value = if (function == .object) function.object.get("arguments") else null;
            if (arguments == null or arguments.? == .object) {
                try new_calls.append(call);
                continue;
            }
            var parsed = arguments.?;
            if (parsed == .string) {
                parsed = pyjson.loads(arena, parsed.string) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.InvalidJson => Value.null,
                };
            }
            if (parsed != .object) {
                var wrapped: ObjectMap = .empty;
                try wrapped.put(arena, "_invalid_arguments", arguments.?);
                parsed = .{ .object = wrapped };
            }
            const new_function = try with(arena, function.object, "arguments", parsed);
            try new_calls.append(.{ .object = try with(arena, call.object, "function", .{ .object = new_function }) });
            touched = true;
        }
        try out.append(if (touched) .{ .object = try with(arena, entry.object, "tool_calls", .{ .array = new_calls }) } else entry);
    }
    return .{ .array = out };
}

/// The messages as the template is given them: `normalize`, then `toolCallArguments`.
pub fn prepare(arena: Allocator, messages: Value, options: Options, problem: *Problem) Error!Value {
    return toolCallArguments(arena, try normalize(arena, messages, options, problem));
}

/// The tokenizer's special tokens a template may name, from `tokenizer_config.json`: each present key of
/// `bos_token`, `eos_token`, `pad_token`, `unk_token`, as its `content` when it is an object.
pub fn specials(arena: Allocator, tokenizer_config: Value) Allocator.Error!ObjectMap {
    var out: ObjectMap = .empty;
    if (tokenizer_config != .object) return out;
    for ([_][]const u8{ "bos_token", "eos_token", "pad_token", "unk_token" }) |key| {
        const v = tokenizer_config.object.get(key) orelse continue;
        try out.put(arena, key, if (v == .object) v.object.get("content") orelse Value.null else v);
    }
    return out;
}

const probe = "tensorfold-late-system-probe";
const probe_messages =
    \\[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"},
    \\ {"role": "system", "content": "tensorfold-late-system-probe"}, {"role": "user", "content": "v"}]
;

/// `system` when the template keeps a system message that follows the first turn in place, else `user`; a
/// template that refuses such a conversation gives `user`.
pub fn lateSystem(gpa: Allocator, t: *const template.Template, special: ObjectMap) Allocator.Error!LateSystem {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var variables = try special.clone(arena);
    const messages = std.json.parseFromSliceLeaky(Value, arena, probe_messages, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => unreachable,
    };
    try variables.put(arena, "messages", messages);
    try variables.put(arena, "add_generation_prompt", .{ .bool = false });
    var diag: template.Diagnostic = .{};
    const text = t.renderAlloc(arena, .{ .object = variables }, &diag) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return .user,
    };
    return if (std.mem.indexOf(u8, text, probe) != null) .system else .user;
}

/// A request's reasoning effort. `none` is not on the ladder: it is thinking switched off.
pub const Effort = enum {
    none,
    minimal,
    low,
    medium,
    high,
    xhigh,
    max,

    /// The ladder's levels, lowest first.
    const ladder = [_]Effort{ .minimal, .low, .medium, .high, .xhigh, .max };

    fn rank(e: Effort) usize {
        return std.mem.indexOfScalar(Effort, &ladder, e).?;
    }
};

pub const Levels = std.EnumSet(Effort);

/// The efforts a chat template names, as quoted strings: Qwen3.8's low, medium and xhigh; GLM-5.3's low, high
/// and max.
pub fn effortLevels(source: []const u8) Levels {
    var levels: Levels = .empty;
    for (source, 0..) |c, i| {
        if (c != '\'' and c != '"') continue;
        const rest = source[i + 1 ..];
        for (Effort.ladder) |e| {
            const name = @tagName(e);
            if (rest.len > name.len and std.mem.startsWith(u8, rest, name) and
                (rest[name.len] == '\'' or rest[name.len] == '"')) levels.insert(e);
        }
    }
    return levels;
}

/// The effort the template hears: the nearest level it names, ties going higher. `none` and `xhigh` stay, and a
/// template that names none hears low, medium or xhigh.
pub fn coerceEffort(effort: Effort, levels: Levels) Effort {
    if (levels.count() == 0) return switch (effort) {
        .high, .max => .xhigh,
        .minimal => .low,
        else => effort,
    };
    if (effort == .none or effort == .xhigh or levels.contains(effort)) return effort;
    const want = effort.rank();
    var best: Effort = undefined;
    var best_distance: usize = std.math.maxInt(usize);
    // Highest first, so of two levels equally far the higher one is kept.
    var i = Effort.ladder.len;
    while (i > 0) {
        i -= 1;
        const level = Effort.ladder[i];
        if (!levels.contains(level)) continue;
        const distance = if (i > want) i - want else want - i;
        if (distance < best_distance) {
            best = level;
            best_distance = distance;
        }
    }
    return best;
}

pub const Request = struct {
    /// As `prepare` returns them.
    messages: Value,
    /// The request's tools; an empty list is none.
    tools: ?Value = null,
    enable_thinking: bool,
    /// Already in the template's names (`coerceEffort`); passed only with thinking on and never as `none`.
    reasoning_effort: ?Effort = null,
};

/// The variables `ChatTemplate.render` hands the template.
pub fn context(arena: Allocator, special: ObjectMap, request: Request) Allocator.Error!Value {
    var out = try special.clone(arena);
    try out.put(arena, "messages", request.messages);
    const tools = request.tools orelse Value.null;
    try out.put(arena, "tools", if (truthy(tools)) tools else Value.null);
    try out.put(arena, "add_generation_prompt", .{ .bool = true });
    try out.put(arena, "enable_thinking", .{ .bool = request.enable_thinking });
    if (request.reasoning_effort) |effort| {
        if (request.enable_thinking and effort != .none) try out.put(arena, "reasoning_effort", .{ .string = @tagName(effort) });
    }
    return .{ .object = out };
}
