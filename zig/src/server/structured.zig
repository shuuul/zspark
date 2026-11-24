//! A request's `request_file.Structure` as the constraint its reply is decoded under (spec 017): a document's
//! schema through the grammar engine, a required call through the call gate over the request's tools. Host
//! only. An engine's grammars share its vocabulary's caches, which their masks write: two constraints must
//! not be asked at the same time, and nothing outside decoding (parsing a request, `count_tokens`) asks one.
//! A request is compiled on its connection's thread when the host asks whether it is refused
//! (`Served.refusal`, `served.zig`), else on the request's own: beside other compiles and beside masks, which
//! it may be, since compiling reads the vocabulary and writes a constraint of its own. A session's request is
//! masked in the session's turn (`served.zig` `Live.enters`); under shared rounds (`scheduler.zig`) every mask is asked on the one
//! decoding thread.
//!
//! What a client gets wrong here is a 400 that names it: a keyword the matcher does not read, a schema that is
//! not one, a tool whose `parameters` are not an object's schema.
//!
//! A tool's `parameters` under a required call, where TensorFold's gate reads none of it:
//!
//! - absent or null: the call takes no arguments, `{}` alone (OpenAI: "omitting parameters defines a function
//!   with an empty parameter list");
//! - a schema that does not say its `type` (and is not a `$ref`, an `anyOf`, a `oneOf`, an `enum` or a
//!   `const`): an object's, as a call's arguments always are, so `"type": "object"` is added to it;
//! - a schema of another type, or one of those forms: refused. The arguments have to be a JSON object for the
//!   reply's parser and for every API's tool call.
//! - a keyword the matcher does not read: refused, with the tool's name. With `tool_choice` "auto" nothing
//!   reads the schema and the same tools are served as before.
//!
//! The gate writes the call in the chat template's form where the tool's schema can be held in it and as JSON
//! where not (`text/call_gate.zig` says which); the host's parser reads both, so a client is sent the same call.
//!
//! A tool's name is read as TensorFold 0.6.5's `tool_spec_name` reads it (`toolName`), but that a name which
//! is a list or an object is no name, where Python prints its `repr`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const request_file = @import("request.zig");
const grammar = @import("../text/grammar.zig");
const call_gate = @import("../text/call_gate.zig");
const template = @import("../text/template.zig");

/// A call without parameters takes `{}`.
const no_parameters = "{\"type\":\"object\",\"properties\":{}}";

/// What a schema that leaves out `type` may hold and still be read as an object's.
const typed_otherwise = [_][]const u8{ "$ref", "anyOf", "oneOf", "allOf", "enum", "const" };

pub const Compiler = struct {
    /// The grammar engine made for the model's tokenizer, logits width and end tokens.
    engine: grammar.Engine,
    tokenizer: *const call_gate.Tokenizer,
    /// The tokens that end a reply: the engine's.
    stops: []const u32,
    gate: call_gate.Options = .{},
    /// Text before a document: what the chat template writes between a reply's reasoning and its content
    /// (`Frontend.content_lead`), for a reply whose constraint begins after `</think>` (spec 035). Its tokens
    /// are forced before the document's first (`grammar.Led`), so the reply is the tokens the next turn's
    /// prompt renders it as. Empty: the reply begins with the document. A lead the tokenizer cannot write
    /// back is left out. A call's lead is `gate.lead`.
    lead: []const u8 = "",

    /// The constraint `request` asks for, or null for free text; the caller frees it (`deinit`) after the
    /// reply. `arena` holds what is made on the way. A refusal is `BadRequest` with `diagnostic` saying which
    /// field or tool, and what in it.
    /// `made` after the tokens of `written`, each forced (`grammar.Led`): a document after the content lead,
    /// and a structure xgrammar compiled (`xgrammar.zig`) after its lead. A lead the tokenizer cannot write back
    /// is left out, as the call gate leaves it. `made` is the result's to free.
    pub fn led(self: Compiler, gpa: Allocator, arena: Allocator, made: grammar.Constraint, written: []const u8) Allocator.Error!grammar.Constraint {
        var lead: std.ArrayList(u32) = .empty;
        call_gate.tokensOf(arena, self.tokenizer, self.stops, self.engine.words, written, &lead) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            error.Unwritable => lead.clearRetainingCapacity(),
        };
        return grammar.Led.wrap(gpa, made, lead.items);
    }

    pub fn compile(self: Compiler, gpa: Allocator, arena: Allocator, request: request_file.Request, diagnostic: *request_file.Diagnostic) request_file.Error!?grammar.Constraint {
        const structure = request.structure orelse return null;
        var said: grammar.Diagnostic = .{};
        switch (structure) {
            .document => |document| {
                const made = self.engine.compile(gpa, document.spec, &said) catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.InvalidSchema, error.UnsupportedSchema => {
                        diagnostic.set("{s}: the schema cannot be enforced: {s}", .{ document.field, said.message() });
                        return error.BadRequest;
                    },
                };
                errdefer made.deinit();
                return try self.led(gpa, arena, made, self.lead);
            },
            .call => |call| {
                const offered: []const Value = switch (request.tools orelse Value.null) {
                    .array => |array| array.items,
                    else => &.{},
                };
                const tools = try arena.alloc(call_gate.Tool, offered.len);
                for (tools, offered) |*tool, given| {
                    const name = try toolName(arena, given);
                    tool.* = .{ .name = name, .parameters = try parametersOf(arena, name, given, diagnostic) };
                }
                var options = self.gate;
                options.single = call.single;
                return call_gate.compile(gpa, self.engine, self.tokenizer, self.stops, tools, options, &said) catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.InvalidSchema, error.UnsupportedSchema => {
                        diagnostic.set("tool_choice: a required tool call cannot be enforced: {s}", .{said.message()});
                        return error.BadRequest;
                    },
                };
            },
        }
    }
};

/// `tool_spec_name`: the name under `function`, or beside it when there is no function object, as
/// `str(name or "").strip()`.
fn toolName(arena: Allocator, tool: Value) Allocator.Error![]const u8 {
    if (tool != .object) return "";
    const function = tool.object.get("function") orelse .null;
    const named = (if (function == .object) function.object else tool.object).get("name") orelse return "";
    var out: std.ArrayList(u8) = .empty;
    // `bool(name)` false is no name; `str(name)` otherwise
    const text: []const u8 = switch (named) {
        .null => "",
        .bool => |b| if (b) "True" else "",
        .integer => |i| if (i == 0) "" else text: {
            try out.print(arena, "{d}", .{i});
            break :text out.items;
        },
        .float => |f| if (f == 0) "" else text: {
            try template.appendFloat(arena, &out, f, "nan", "inf");
            break :text out.items;
        },
        .number_string, .string => |s| s,
        // a departure: Python prints the `repr` of a list or an object
        .array, .object => "",
    };
    return template.strip(text, null, true, true);
}

/// The schema a required call's arguments are held to, as JSON text.
fn parametersOf(arena: Allocator, name: []const u8, tool: Value, diagnostic: *request_file.Diagnostic) request_file.Error![]const u8 {
    const given: Value = given: {
        if (tool != .object) break :given .null;
        const inner = tool.object.get("function") orelse Value.null;
        const holder = if (inner == .object) inner.object else tool.object;
        // an empty object is a schema too (any object), where the reply's parser reads it as no schema
        break :given holder.get("parameters") orelse holder.get("input_schema") orelse .null;
    };
    var schema = switch (given) {
        .null => return no_parameters,
        .object => |object| object,
        else => {
            diagnostic.set("tool_choice: a required tool call cannot be enforced: tool '{s}': its parameters must be a JSON schema object", .{name});
            return error.BadRequest;
        },
    };
    const an_object = if (schema.get("type")) |kind|
        kind == .string and std.mem.eql(u8, kind.string, "object")
    else for (typed_otherwise) |key| {
        if (schema.contains(key)) break false;
    } else true;
    if (!an_object) {
        diagnostic.set("tool_choice: a required tool call cannot be enforced: tool '{s}': its parameters must be an object's schema (\"type\": \"object\")", .{name});
        return error.BadRequest;
    }
    if (!schema.contains("type")) {
        schema = try schema.clone(arena);
        try schema.put(arena, "type", .{ .string = "object" });
    }
    return std.json.Stringify.valueAlloc(arena, Value{ .object = schema }, .{});
}
