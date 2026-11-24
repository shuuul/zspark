//! A checkpoint's text side in one place: its tokenizer, chat template and generation config, and the way from
//! a request's messages to the prompt's ids, as TensorFold's `ChatTemplate.render` and the tokenizer take it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Value = std.json.Value;
const messages = @import("messages.zig");
const template = @import("template.zig");
const tokenizer = @import("tokenizer.zig");
/// TensorFold's tokenizer, which `load` may borrow.
pub const Loaded = tokenizer.Loaded;

pub const Diagnostic = struct {
    buffer: [640]u8 = undefined,
    len: usize = 0,

    pub fn message(d: *const Diagnostic) []const u8 {
        return d.buffer[0..d.len];
    }

    fn set(d: *Diagnostic, comptime format: []const u8, args: anytype) void {
        var w: Io.Writer = .fixed(&d.buffer);
        w.print(format, args) catch {};
        d.len = w.buffered().len;
    }
};

pub const Request = struct {
    /// The request's `messages`, as it sent them.
    messages: Value,
    tools: ?Value = null,
    thinking: bool = true,
    /// As the request names it; the template hears the nearest level it knows. `none` switches thinking off.
    effort: ?messages.Effort = null,
    allow_images: bool = false,
    /// The reply's header after the messages (`add_generation_prompt`): every prompt's; off for a probe alone.
    generation: bool = true,
};

pub const Error = error{ OutOfMemory, BadRequest, TemplateFailed };

pub const Frontend = struct {
    /// Holds the parsed JSON files and `specials`.
    arena: std.heap.ArenaAllocator,
    tokenizer: tokenizer.Tokenizer,
    template: template.Template,
    specials: std.json.ObjectMap,
    late: messages.LateSystem,
    levels: messages.Levels,
    /// `generation_config.json`, when the checkpoint has one.
    generation: ?Value,
    /// The special tokens the template opens a message with, the reply's header included, ascending: one where
    /// every message opens with the same (`<|im_start|>`), a role's own where they differ (GLM's `<|user|>` and
    /// `<|assistant|>`), none where the probes find neither. A prompt's messages start where one stands
    /// (`starts`). Held by `arena`.
    openers: []const u32 = &.{},
    /// What the template writes between a reply's reasoning and its first tool call (spec 034): the text from
    /// `</think>` to `<tool_call>` in a rendered assistant message that has both and no content (Qwen's two
    /// newlines). A required call after thinking is written with it (`call_gate.Options.lead`), so the next
    /// turn's prompt holds the reply as it was decoded. Empty where the template writes nothing there or the
    /// probe finds no such place; held by `arena`.
    call_lead: []const u8 = "",
    /// What the template writes between a reply's reasoning and its content (spec 035): the text from
    /// `</think>` to the content in a rendered assistant message that has both (Qwen's two newlines, the same
    /// as before a call). A reply held to a JSON document after thinking is written with it
    /// (`server/structured.zig` `Compiler.lead`). Empty as `call_lead` is; held by `arena`.
    content_lead: []const u8 = "",

    /// Reads `tokenizer.json`, `tokenizer_config.json`, the template (`chat_template.jinja`, else the config's
    /// `chat_template`) and `generation_config.json` from a checkpoint's directory. `shared` is the directory's
    /// `tokenizer.json` as something else already read it and frees it after this front end (`serve`'s host):
    /// the file is then not read again.
    pub fn load(gpa: Allocator, io: Io, dir: Io.Dir, shared: ?*const tokenizer.Loaded, diagnostic: *Diagnostic) !Frontend {
        return loadWith(gpa, io, dir, shared, null, diagnostic);
    }

    /// `load` with `own` the template where it is not null (Unlimited-OCR's, whose checkpoint has none): the
    /// directory's is then not read.
    pub fn loadWith(gpa: Allocator, io: Io, dir: Io.Dir, shared: ?*const tokenizer.Loaded, own: ?[]const u8, diagnostic: *Diagnostic) !Frontend {
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena_state.deinit();
        const arena = arena_state.allocator();
        const limit: Io.Limit = .limited(1 << 24);

        const config_text = dir.readFileAlloc(io, "tokenizer_config.json", arena, limit) catch |err| {
            diagnostic.set("tokenizer_config.json: {t}", .{err});
            return err;
        };
        const config = std.json.parseFromSliceLeaky(Value, arena, config_text, .{}) catch |err| {
            diagnostic.set("tokenizer_config.json: {t}", .{err});
            return err;
        };
        const source = if (own) |given| given else dir.readFileAlloc(io, "chat_template.jinja", arena, limit) catch |err| switch (err) {
            error.FileNotFound => source: {
                const inline_template: Value = if (config == .object) config.object.get("chat_template") orelse .null else .null;
                if (inline_template != .string) {
                    diagnostic.set("no chat_template.jinja, and tokenizer_config.json has no chat_template", .{});
                    return error.NoChatTemplate;
                }
                break :source inline_template.string;
            },
            else => return err,
        };
        const generation: ?Value = generation: {
            const text = dir.readFileAlloc(io, "generation_config.json", arena, limit) catch |err| switch (err) {
                error.FileNotFound => break :generation null,
                else => return err,
            };
            break :generation std.json.parseFromSliceLeaky(Value, arena, text, .{}) catch |err| {
                diagnostic.set("generation_config.json: {t}", .{err});
                return err;
            };
        };

        var template_diagnostic: template.Diagnostic = .{};
        var parsed = template.Template.parse(gpa, source, &template_diagnostic) catch |err| {
            diagnostic.set("chat template: {s}", .{template_diagnostic.message()});
            return err;
        };
        errdefer parsed.deinit();
        var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
        var tokens = (if (shared) |loaded| tokenizer.Tokenizer.over(gpa, loaded, &tokenizer_diagnostic) else tokenizer.Tokenizer.load(gpa, io, dir, "tokenizer.json", &tokenizer_diagnostic)) catch |err| {
            diagnostic.set("tokenizer.json: {s}", .{if (tokenizer_diagnostic.len > 0) tokenizer_diagnostic.message() else @errorName(err)});
            return err;
        };
        errdefer tokens.deinit(gpa);

        const specials = try messages.specials(arena, config);
        var self: Frontend = .{
            .arena = arena_state,
            .tokenizer = tokens,
            .template = parsed,
            .specials = specials,
            .late = try messages.lateSystem(gpa, &parsed, specials),
            .levels = messages.effortLevels(source),
            .generation = generation,
        };
        self.openers = if (try self.probeOpener(gpa)) |one| try arena.dupe(u32, &.{one}) else try self.probeOpeners(arena);
        self.call_lead = try self.probeCallLead(arena);
        self.content_lead = try self.probeContentLead(arena);
        // the probes allocated in the arena after `self` took its copy of it
        self.arena = arena_state;
        return self;
    }

    /// The reasoning's closing tag and a call's opening tag, as the decoder (`models/qwen4/session/session.zig`) and the
    /// call gate (`call_gate.zig`) know them.
    const think_close = "</think>";
    const call_open = "<tool_call>";

    /// `call_lead`, found by rendering the turn after a call: a question, a reply that reasoned and called a
    /// tool without a word of content, and the tool's result, with thinking on. The lead is what stands between
    /// the tag that closes that reasoning and the tag that opens the call. TensorFold's `call_format`
    /// (`engine/call_gate.py`) reads a call's form off a rendered message the same way, from the opener on.
    /// Kept only when it is newlines and nothing else, which `Session.reply` drops after `</think>` so that a
    /// client is sent the same content with it; a template the probe cannot render, one that leaves the
    /// reasoning or the call out, or one with other text there has none. Allocated in `arena`.
    pub fn probeCallLead(self: *const Frontend, arena: Allocator) Allocator.Error![]const u8 {
        const said =
            \\[{"role":"user","content":"Alpha"},
            \\ {"role":"assistant","content":"","reasoning_content":"Beta","tool_calls":[
            \\   {"id":"call_0","type":"function","function":{"name":"gamma","arguments":"{\"delta\":\"Epsilon\"}"}}]},
            \\ {"role":"tool","tool_call_id":"call_0","content":"Zeta"}]
        ;
        const list = std.json.parseFromSliceLeaky(Value, arena, said, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => unreachable,
        };
        var diagnostic: Diagnostic = .{};
        const text = self.render(arena, .{ .messages = list }, &diagnostic) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return "",
        };
        // the reply's reasoning, the tag that closes it, the call's opening and its name, in that order
        const reasoning = std.mem.lastIndexOf(u8, text, "Beta") orelse return "";
        const close = std.mem.indexOfPos(u8, text, reasoning, think_close) orelse return "";
        const from = close + think_close.len;
        const open = std.mem.indexOfPos(u8, text, from, call_open) orelse return "";
        const name = std.mem.indexOfPos(u8, text, open, "gamma") orelse return "";
        // the opening found is that call's own: no other stands before the name
        if (std.mem.indexOfPos(u8, text, open + call_open.len, call_open)) |other| {
            if (other < name) return "";
        }
        const lead = text[from..open];
        if (std.mem.indexOfNone(u8, lead, "\n") != null) return "";
        return lead;
    }

    /// `content_lead`, found by rendering the turn after a reply that reasoned and answered: a question, that
    /// reply, and the next question, with thinking on. The lead is what stands between the tag that closes
    /// the reasoning and the content. The content is a JSON document, as the replies this lead is forced
    /// before are: a template may write other text before other content. Kept under `probeCallLead`'s rule:
    /// newlines and nothing else, or none. A template that drops the reasoning of a turn already answered
    /// renders no `</think>` here and has no lead: the content then follows another prefix whatever is forced.
    pub fn probeContentLead(self: *const Frontend, arena: Allocator) Allocator.Error![]const u8 {
        const said =
            \\[{"role":"user","content":"Alpha"},
            \\ {"role":"assistant","content":"{\"gamma\":\"Epsilon\"}","reasoning_content":"Beta"},
            \\ {"role":"user","content":"Zeta"}]
        ;
        const list = std.json.parseFromSliceLeaky(Value, arena, said, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => unreachable,
        };
        var diagnostic: Diagnostic = .{};
        const text = self.render(arena, .{ .messages = list }, &diagnostic) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return "",
        };
        // the reply's reasoning, the tag that closes it and the content, in that order
        const reasoning = std.mem.lastIndexOf(u8, text, "Beta") orelse return "";
        const close = std.mem.indexOfPos(u8, text, reasoning, think_close) orelse return "";
        const from = close + think_close.len;
        const content = std.mem.indexOfPos(u8, text, from, "{\"gamma\":\"Epsilon\"}") orelse return "";
        const lead = text[from..content];
        if (std.mem.indexOfNone(u8, lead, "\n") != null) return "";
        return lead;
    }

    /// The token every message opens with, found by rendering one message and three: the prompt's first token,
    /// when it is special and each message more adds it once. TensorFold's `message_markers` asks the same of a
    /// template (`engine/prefill_plan.py`); it probes thinking on as well and keeps the assistant's header,
    /// which nothing here reads. A template the probe cannot render has no opener.
    pub fn probeOpener(self: *const Frontend, gpa: Allocator) Allocator.Error!?u32 {
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var counts: [2]usize = undefined;
        var first: ?u32 = null;
        for (&counts, [_]usize{ 1, 3 }) |*count, said| {
            var list: std.json.Array = .init(arena);
            const roles = [_][]const u8{ "user", "assistant", "user" };
            const words = [_][]const u8{ "Alpha", "Beta", "Gamma" };
            for (roles[0..said], words[0..said]) |role, content| {
                var message: std.json.ObjectMap = .empty;
                try message.put(arena, "role", .{ .string = role });
                try message.put(arena, "content", .{ .string = content });
                try list.append(.{ .object = message });
            }
            var diagnostic: Diagnostic = .{};
            const ids = self.prompt(arena, .{ .messages = .{ .array = list }, .thinking = false }, &diagnostic) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return null,
            };
            if (ids.len == 0 or !self.tokenizer.isSpecial(ids[0])) return null;
            if (first != null and first.? != ids[0]) return null;
            first = ids[0];
            count.* = std.mem.count(u32, ids, &.{ids[0]});
        }
        // a message each and the reply's; two messages more add two
        return if (counts[0] >= 2 and counts[1] == counts[0] + 2) first else null;
    }

    /// The openers where no one token opens every message (`probeOpener` found none): the special token each
    /// role's messages open with, as TensorFold's `markers` finds them (`server/chunk_plan.zig`, its
    /// `openers`). A conversation of a user's and an assistant's messages in turn is rendered a message at a
    /// time without the reply's header, and once and three messages with it: what a render adds to the one
    /// before is that message (the reply's header an assistant's). A role whose pieces all open with one special
    /// token gives it. A render that is not the one before it extended (a template that writes an earlier reply
    /// otherwise once another follows) gives no piece: TensorFold looks there for the next opener, which no
    /// template served here needs. Allocated in `arena`.
    pub fn probeOpeners(self: *const Frontend, arena: Allocator) Allocator.Error![]const u32 {
        const roles = [_][]const u8{ "user", "assistant", "user", "assistant" };
        const words = [_][]const u8{ "Alpha", "Beta", "Gamma", "Delta" };
        var talk: std.json.Array = .init(arena);
        for (roles, words) |role, content| {
            var message: std.json.ObjectMap = .empty;
            try message.put(arena, "role", .{ .string = role });
            try message.put(arena, "content", .{ .string = content });
            try talk.append(.{ .object = message });
        }
        // r[k]: the first k + 1 messages; g[k] with the reply's header
        var r: [4][]const u32 = undefined;
        var g: [4]?[]const u32 = @splat(null);
        for (&r, 0..) |*out, k| {
            var some: std.json.Array = .init(arena);
            try some.appendSlice(talk.items[0 .. k + 1]);
            var diagnostic: Diagnostic = .{};
            out.* = self.prompt(arena, .{ .messages = .{ .array = some }, .thinking = false, .generation = false }, &diagnostic) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return &.{},
            };
            if (k % 2 == 1) continue;
            g[k] = self.prompt(arena, .{ .messages = .{ .array = some }, .thinking = false }, &diagnostic) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return &.{},
            };
        }
        // a piece each: what `after` adds to `before`, of `role` (0 the user's, 1 the assistant's)
        const Pair = struct { before: []const u32, after: []const u32, role: u1 };
        const pairs = [_]Pair{
            .{ .before = r[0], .after = r[1], .role = 1 },   .{ .before = r[1], .after = r[2], .role = 0 },
            .{ .before = r[2], .after = r[3], .role = 1 },   .{ .before = r[0], .after = g[0].?, .role = 1 },
            .{ .before = r[2], .after = g[2].?, .role = 1 },
        };
        var out: std.ArrayList(u32) = .empty;
        for ([_]u1{ 0, 1 }) |role| {
            var first: ?u32 = null;
            var alike = true;
            for (pairs) |pair| {
                if (pair.role != role) continue;
                const extends = pair.after.len > pair.before.len and std.mem.eql(u32, pair.after[0..pair.before.len], pair.before);
                if (!extends) continue;
                const opens = pair.after[pair.before.len];
                if (first != null and first.? != opens) alike = false;
                first = opens;
            }
            const one = first orelse continue;
            if (!alike or !self.tokenizer.isSpecial(one) or std.mem.indexOfScalar(u32, out.items, one) != null) continue;
            try out.append(arena, one);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.items;
    }

    /// Where the messages of a prompt's `ids` start, ascending: the positions of the template's openers. The
    /// last is the reply's own header. Empty for a template without one.
    pub fn starts(self: *const Frontend, arena: Allocator, ids: []const i32) Allocator.Error![]usize {
        var out: std.ArrayList(usize) = .empty;
        if (self.openers.len == 0) return out.items;
        for (ids, 0..) |id, at| {
            if (id >= 0 and std.mem.indexOfScalar(u32, self.openers, @intCast(id)) != null) try out.append(arena, at);
        }
        return out.items;
    }

    pub fn deinit(self: *Frontend, gpa: Allocator) void {
        self.tokenizer.deinit(gpa);
        self.template.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    /// The prompt's text, ending where the reply begins. On `error.BadRequest` and `error.TemplateFailed` the
    /// diagnostic says why.
    pub fn render(self: *const Frontend, arena: Allocator, request: Request, diagnostic: *Diagnostic) Error![]u8 {
        var problem: messages.Problem = .{};
        const prepared = messages.prepare(arena, request.messages, .{
            .late = self.late,
            .allow_images = request.allow_images,
        }, &problem) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.BadRequest => {
                diagnostic.set("{s}", .{problem.message});
                return error.BadRequest;
            },
        };
        const thinking = request.thinking and request.effort != .none;
        var variables = try messages.context(arena, self.specials, .{
            .messages = prepared,
            .tools = request.tools,
            .enable_thinking = thinking,
            .reasoning_effort = if (request.effort) |effort| messages.coerceEffort(effort, self.levels) else null,
        });
        if (!request.generation) try variables.object.put(arena, "add_generation_prompt", .{ .bool = false });
        var template_diagnostic: template.Diagnostic = .{};
        return self.template.renderAlloc(arena, variables, &template_diagnostic) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TemplateRaised => {
                diagnostic.set("{s}", .{template_diagnostic.message()});
                return error.BadRequest;
            },
            error.TemplateRuntime => {
                diagnostic.set("chat template: {s}", .{template_diagnostic.message()});
                return error.TemplateFailed;
            },
        };
    }

    /// `render`, then the tokenizer: the prompt's ids, with no token added around them.
    pub fn prompt(self: *const Frontend, arena: Allocator, request: Request, diagnostic: *Diagnostic) Error![]u32 {
        const text = try self.render(arena, request, diagnostic);
        var ids: std.ArrayList(u32) = .empty;
        self.tokenizer.encode(arena, text, &ids) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.InvalidUtf8 => {
                diagnostic.set("the rendered prompt is not UTF-8", .{});
                return error.BadRequest;
            },
        };
        return ids.items;
    }
};
