//! A reply that must call a tool (spec 017): `tool_choice` "required" or a named function as a constraint
//! behind `grammar.zig`'s interface, so the decoder masks it as it masks a schema and a drafted reply stays the
//! serial one. Host only.
//!
//! The reply is one call or more, and nothing else. A call is written in the form the chat template tells the
//! model and writes its history in (spec 023), where the tool's schema can be held in it:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! and otherwise as JSON, which a reply's parser reads as well:
//!
//! ```text
//! <tool_call>
//! {"name": "NAME", "arguments":{...}}
//! </tool_call>
//! ```
//!
//! What is forced: everything but the values. The opening tag, the name and each parameter's frame are the
//! tokens the tokenizer cuts that text into, each the one token its mask allows; where texts part (several
//! tools at the name, an optional parameter or the next one, the call's end) the mask holds one token for each
//! that still fits, so the model chooses by choosing the token. A mask of one token is a draft the decoder
//! takes without asking a model (`masks.zig`), which is what makes a required call as fast as a free
//! one: in the template's form a call is mostly frame.
//!
//! The template's form. The parameters come in the schema's order, the optional ones may be left out and no
//! other key is written, as the JSON form's objects go (`json_schema.zig`). A value is held to its property's
//! schema by the way the reply's parser types it (TensorFold's `decode_parameter`):
//!
//! - `"type": "string"`: the text itself, not quoted and not escaped, over as many lines as it likes. It ends
//!   with a newline, `</parameter>` and a newline, of which the parser drops the one newline before the closer:
//!   every string can be written, one that ends with a newline too. What it may not hold is `</parameter>` or
//!   `</tool_call>` in any case, where the parser would end the value or the call (`Strings`). The mask is over
//!   bytes, so the model's own tokens for the closer are taken however they fall. With `enum` or `const`, the
//!   values written so and no other.
//! - `integer`, `number`, `boolean`, `null`, `array`, `object`: the property's schema as a JSON document
//!   (the engine handed in), then the closer's tokens. Its strings may not hold the two closers unescaped
//!   either.
//!
//! A frame's tokens are the tokenizer's for the frame alone. Where the tokenizer would join a frame's end to a
//! value's (an empty string's two newlines, a `]` and the newline after it) the gate has the two tokens apart;
//! everywhere else a call is the tokens the model would write freely, which is why the form was chosen: under
//! it a required call was the free call's own tokens on most seeds, where the JSON form's arguments drifted
//! (spec 023 has the replies and the speeds).
//!
//! A tool is written as JSON when its schema says something the template's form cannot hold to it: a property
//! without one `type` of its own (`anyOf`, a list of types, an `enum` of numbers: the parser would hand the
//! client a string), a `$ref`, a string with a keyword beside `enum` and `const`, a schema without
//! `properties`, a name or key the parser's `[^>\s]+` would not read. Tools of both forms may be offered
//! together: the forms part at the token after `<tool_call>`. In the JSON form the arguments are the tool's
//! `parameters` compiled as one schema, and `</tool_call>` unescaped is kept out of them as well.
//!
//! After a call the mask holds the end tokens and, unless one call was asked for, the first token of another.
//!
//! Thinking is before the gate: the decoder begins a constraint at the token after `</think>`
//! (`masks.zig`), so the reasoning is free. What follows it is the first call's opening tag, after
//! `Options.lead`: the text the chat template writes between a message's reasoning and its call (Qwen's two
//! newlines), which `frontend.zig` reads off the template at load and the caller hands in for a reply that
//! opens thinking (spec 034). Its tokens are forced like a frame's, so they are drafted, and the reply is then
//! the tokens the next turn's prompt renders it as: the tool result's turn runs nothing again. Without a
//! lead (a reply that does not think: the prompt already ends with the template's text; a template that writes
//! nothing there) the first token is the opening tag.
//!
//! TensorFold's `engine/call_gate.py` `CallGate` is the reference, and this departs from it:
//!
//! - TensorFold lets the reply run and repairs it: the first token that is not blank and not the opener is cut,
//!   the opener and the template's lead (`\n<function=`) are spliced in and decoding starts again from there;
//!   a name that leaves the offered ones is completed. Here nothing is cut or run again: the same tokens are
//!   the only ones their masks allow.
//! - TensorFold's gate is done at the name, and the arguments are the model's own, held to nothing. Here they
//!   are held to the tool's schema, in the template's form where it can be and in JSON where it cannot.
//! - Blank tokens before the opener are not the model's to write: none is allowed but the lead's after
//!   thinking, which is forced (TensorFold lets any blank token pass and forces none, so after `</think>` its
//!   reply has whatever blank lines the model wrote), and `<think>` cannot be opened again in the answer.
//! - After the first call TensorFold constrains nothing. Here every call of the reply is gated, and the reply
//!   ends after its calls.
//!
//! A position is an index into states that are only added, as the matcher's positions are: taking a token gives
//! a new one and the old one stays good.
//!
//! The text a value may not hold (`automaton`, `Strings`) is `call_gate_strings.zig`, a tool's schema read for the
//! template's form (`planOf`) `call_gate_plan.zig`. The tests are `call_gate_test.zig` (the JSON form, the
//! interface) and `call_gate_walk_test.zig` (the template's form).

const std = @import("std");
const grammar = @import("grammar.zig");
const tokenizer_file = @import("tokenizer.zig");
const call_gate_plan = @import("call_gate_plan.zig");
const call_gate_strings = @import("call_gate_strings.zig");
const Allocator = std.mem.Allocator;

pub const Tokenizer = tokenizer_file.Tokenizer;

/// One offered tool.
pub const Tool = struct {
    name: []const u8,
    /// The JSON schema of its arguments, as text: an object's.
    parameters: []const u8,
};

/// The text around a call's name and arguments in the JSON form.
pub const Format = struct {
    /// Before the name, which is written as a JSON string.
    open: []const u8 = "<tool_call>\n{\"name\": ",
    /// Between the name and the arguments. No blank after it: a schema's document begins at once.
    middle: []const u8 = ", \"arguments\":",
    /// After the arguments.
    close: []const u8 = "}\n</tool_call>",
    /// Between one call and the next, in either form.
    between: []const u8 = "\n",
};

/// The template's form, as `chat_template.jinja` writes a call of the history. `Strings` knows the closer too.
pub const template = struct {
    const open = "<tool_call>\n<function=";
    const opened = ">\n";
    const parameter = "<parameter=";
    pub const parameter_close = "\n</parameter>\n";
    const close = "</function>\n</tool_call>";
    /// The most parameters a tool written in it has: a frame's tree is made for each place a call can be at.
    pub const max_parameters = 128;
};

pub const Options = struct {
    /// One call and the end (`parallel_tool_calls: false`); otherwise one or more.
    single: bool = false,
    format: Format = .{},
    /// Text before the reply's first call, in either form: what the chat template writes between a reply's
    /// reasoning and its call, for a reply whose constraint begins after `</think>` (`Frontend.call_lead`).
    /// Empty: the reply begins with the opening tag. A lead the tokenizer cannot write is left out.
    lead: []const u8 = "",
    /// Write a call in the template's form where the tool's schema can be held in it. False: always as JSON.
    template_form: bool = true,
    /// What a raw string's tokens are, kept from one gate to the next by whoever keeps the engine; it must be
    /// made for the same tokenizer, width and stops. Null: the gate makes its own.
    strings: ?*Strings = null,
    /// The most states a reply's positions may take: past it the constraint fails, and the reply with it. A
    /// token taken adds one, a draft that is walked and not kept included.
    max_states: usize = 1 << 20,
};

const none = std.math.maxInt(u32);

// -- text a call's parser must not meet ------------------------------------------------------------------------

pub const automaton = call_gate_strings.automaton;
pub const Strings = call_gate_strings.Strings;

// -- the gate -------------------------------------------------------------------------------------------------

/// Token sequences that part where their texts do, as a tree: a node's edges are the tokens that may come
/// there, and a leaf is one sequence whole. The openings of calls, one a tool; a parameter's frames; a string's
/// `enum`.
const Tree = struct {
    nodes: std.ArrayList(Node) = .empty,

    const Edge = struct { token: u32, child: u32 };
    const Node = struct {
        edges: std.ArrayList(Edge) = .empty,
        /// What the sequence that ends here is: a tool, a parameter.
        leaf: u32 = none,
    };

    fn deinit(self: *Tree, gpa: Allocator) void {
        for (self.nodes.items) |*node| node.edges.deinit(gpa);
        self.nodes.deinit(gpa);
    }

    fn step(self: *const Tree, node: u32, token: u32) ?u32 {
        for (self.nodes.items[node].edges.items) |edge| {
            if (edge.token == token) return edge.child;
        }
        return null;
    }

    /// Adds `tokens` as the sequence `leaf`. False where it is another's, or begins or goes on from one: the
    /// two could not be told apart.
    fn add(self: *Tree, gpa: Allocator, tokens: []const u32, leaf: u32) Allocator.Error!bool {
        if (self.nodes.items.len == 0) try self.nodes.append(gpa, .{});
        var node: u32 = 0;
        for (tokens) |token| {
            if (self.nodes.items[node].leaf != none) return false;
            node = self.step(node, token) orelse made: {
                const child: u32 = @intCast(self.nodes.items.len);
                try self.nodes.append(gpa, .{});
                try self.nodes.items[node].edges.append(gpa, .{ .token = token, .child = child });
                break :made child;
            };
        }
        const end = &self.nodes.items[node];
        if (end.leaf != none or end.edges.items.len != 0 or node == 0) return false;
        end.leaf = leaf;
        return true;
    }
};

const Phase = enum(u8) {
    /// In a call's opening: `node` of the first call's tree.
    first,
    /// The same in a later call's, which begins with the text between calls.
    later,
    /// The JSON form: in `tool`'s arguments, at `sub`, the text at `node` of `automaton`'s states.
    arguments,
    /// In the JSON form's close: `node` of its tokens are taken.
    close,
    /// The template's form, between values: at `node` of the frames that may come before parameter `param`
    /// (`param`'s own and the ones after it up to a required one, or the call's end).
    frame,
    /// In parameter `param`'s raw string, at `node` of `automaton`'s states.
    raw,
    /// In its `enum`: `node` of the values' tree.
    choice,
    /// In its JSON value, at `sub`, the text at `node` of `automaton`'s states.
    value,
    /// In the closer after a JSON value: `node` of its tokens are taken.
    value_close,
    /// After a call: the end, or another call.
    after,
    /// An end token was taken.
    ended,
};

const State = struct {
    phase: Phase,
    tool: u32 = none,
    param: u32 = 0,
    node: u32 = 0,
    sub: grammar.Position = undefined,
};

/// How a parameter's value is written in the template's form.
const Param = struct {
    kind: enum { raw, choice, json },
    /// A `choice`'s tree among `Gate.trees`; a `json`'s constraint among `Gate.subs`.
    index: u32 = none,
};

/// How a tool's call is written.
const Written = struct {
    /// The JSON form: the arguments' constraint among `Gate.subs`.
    sub: u32,
    /// The template's form, or null for the JSON form: the parameters in order, and the first of the
    /// `params.len + 1` trees of frames among `Gate.trees`, one for each parameter that may come next.
    params: ?[]Param = null,
    frames: u32 = none,
};

/// A compiled gate and the reply it follows, on the heap.
pub const Gate = struct {
    gpa: Allocator,
    words: usize,
    single: bool,
    max_states: usize,
    /// The tokens that end a reply.
    stops: []u32,
    tools: std.ArrayList(Written) = .empty,
    /// The constraints of JSON written inside calls, and where each one's documents start in this reply.
    subs: std.ArrayList(grammar.Constraint) = .empty,
    origins: []grammar.Position,
    first: Tree = .{},
    later: Tree = .{},
    trees: std.ArrayList(Tree) = .empty,
    /// The JSON form's close, and the closer after a JSON value in the template's form.
    close: []u32,
    value_close: []u32,
    strings: *Strings,
    owns_strings: bool = false,
    states: std.ArrayList(State) = .empty,

    fn destroy(self: *Gate) void {
        const gpa = self.gpa;
        for (self.subs.items) |sub| sub.deinit();
        self.subs.deinit(gpa);
        for (self.tools.items) |tool| if (tool.params) |params| gpa.free(params);
        self.tools.deinit(gpa);
        for (self.trees.items) |*tree| tree.deinit(gpa);
        self.trees.deinit(gpa);
        gpa.free(self.origins);
        gpa.free(self.stops);
        gpa.free(self.close);
        gpa.free(self.value_close);
        self.first.deinit(gpa);
        self.later.deinit(gpa);
        self.states.deinit(gpa);
        if (self.owns_strings) {
            self.strings.deinit();
            gpa.destroy(self.strings);
        }
        gpa.destroy(self);
    }

    fn push(self: *Gate, state: State) grammar.Error!grammar.Position {
        if (self.states.items.len >= self.max_states) return error.ConstraintFailed;
        try self.states.append(self.gpa, state);
        return @fromBackingInt(self.states.items.len - 1);
    }

    fn at(self: *const Gate, position: grammar.Position) grammar.Error!State {
        const index = @backingInt(position);
        if (index >= self.states.items.len) return error.ConstraintFailed;
        return self.states.items[@intCast(index)];
    }

    fn isStop(self: *const Gate, token: u32) bool {
        return std.mem.indexOfScalar(u32, self.stops, token) != null;
    }

    /// Whether the JSON of constraint `sub` is a whole document at `position`: the constraint takes an end
    /// there.
    fn whole(self: *const Gate, sub: u32, position: grammar.Position) grammar.Error!bool {
        return try self.subs.items[sub].accept(position, self.stops[0]) != null;
    }

    /// The state after a call's opening tree was left at `node`.
    fn opened(self: *const Gate, phase: Phase, tree: *const Tree, node: u32) State {
        const tool = tree.nodes.items[node].leaf;
        if (tool == none) return .{ .phase = phase, .node = node };
        const written = self.tools.items[tool];
        if (written.params != null) return .{ .phase = .frame, .tool = tool };
        return .{ .phase = .arguments, .tool = tool, .node = automaton.plain, .sub = self.origins[written.sub] };
    }

    /// The tree a state of the template's form walks: the frames before `param`, or `param`'s values.
    fn treeOf(self: *const Gate, state: State) *const Tree {
        const written = self.tools.items[state.tool];
        return &self.trees.items[if (state.phase == .frame) written.frames + state.param else written.params.?[state.param].index];
    }

    /// The constraint a state in JSON is held to: the tool's arguments, or `param`'s value.
    fn subOf(self: *const Gate, state: State) u32 {
        const written = self.tools.items[state.tool];
        return if (state.phase == .arguments) written.sub else written.params.?[state.param].index;
    }

    /// The state after the frames before `state.param` were left at `node`: in them still, in a parameter's
    /// value, or after the call.
    fn framed(self: *const Gate, state: State, node: u32) State {
        const params = self.tools.items[state.tool].params.?;
        const leaf = self.treeOf(state).nodes.items[node].leaf;
        if (leaf == none) return .{ .phase = .frame, .tool = state.tool, .param = state.param, .node = node };
        if (leaf == params.len) return .{ .phase = .after };
        const param = params[leaf];
        return switch (param.kind) {
            .raw => .{ .phase = .raw, .tool = state.tool, .param = leaf, .node = automaton.plain },
            .choice => .{ .phase = .choice, .tool = state.tool, .param = leaf },
            .json => .{ .phase = .value, .tool = state.tool, .param = leaf, .node = automaton.plain, .sub = self.origins[param.index] },
        };
    }

    /// The state after parameter `state.param`'s value and its closer: before the frames that may follow.
    fn valued(state: State) State {
        return .{ .phase = .frame, .tool = state.tool, .param = state.param + 1 };
    }

    fn start(context: *anyopaque) grammar.Error!grammar.Position {
        const self: *Gate = @ptrCast(@alignCast(context));
        self.states.clearRetainingCapacity();
        for (self.subs.items, self.origins) |sub, *origin| origin.* = try sub.start();
        return self.push(.{ .phase = .first });
    }

    fn set(mask: []u32, token: u32) void {
        mask[token / 32] |= @as(u32, 1) << @intCast(token % 32);
    }

    fn fill(context: *anyopaque, position: grammar.Position, mask: []u32) grammar.Error!void {
        const self: *Gate = @ptrCast(@alignCast(context));
        const state = try self.at(position);
        switch (state.phase) {
            .first, .later, .frame, .choice => {
                @memset(mask, 0);
                const tree = switch (state.phase) {
                    .first => &self.first,
                    .later => &self.later,
                    else => self.treeOf(state),
                };
                for (tree.nodes.items[state.node].edges.items) |edge| set(mask, edge.token);
            },
            .arguments, .value => {
                try self.subs.items[self.subOf(state)].fill(state.sub, mask);
                // the schema's end is the call's close or the value's, not the reply's end
                var complete = false;
                for (self.stops) |stop| complete = complete or grammar.allowed(mask, stop);
                // no closer the reply's parser looks for, written out in a string
                for (mask, try self.strings.mask(@intCast(state.node), true)) |*word, kept| word.* &= kept;
                if (complete) set(mask, if (state.phase == .arguments) self.close[0] else self.value_close[0]);
            },
            .close, .value_close => {
                @memset(mask, 0);
                set(mask, if (state.phase == .close) self.close[state.node] else self.value_close[state.node]);
            },
            .raw => @memcpy(mask, try self.strings.mask(@intCast(state.node), false)),
            .after => {
                @memset(mask, 0);
                for (self.stops) |stop| set(mask, stop);
                if (!self.single) for (self.later.nodes.items[0].edges.items) |edge| set(mask, edge.token);
            },
            .ended => @memset(mask, 0),
        }
    }

    fn accept(context: *anyopaque, position: grammar.Position, token: u32) grammar.Error!?grammar.Position {
        const self: *Gate = @ptrCast(@alignCast(context));
        const state = try self.at(position);
        if (token / 32 >= self.words) return null;
        switch (state.phase) {
            .first, .later => {
                const tree = if (state.phase == .first) &self.first else &self.later;
                const child = tree.step(state.node, token) orelse return null;
                return try self.push(self.opened(state.phase, tree, child));
            },
            .arguments, .value => {
                if (self.isStop(token)) return null;
                const sub = self.subOf(state);
                // a token the JSON takes and the text does not is refused, unless it begins the close
                if (try self.subs.items[sub].accept(state.sub, token)) |next| taken: {
                    const written = self.strings.after(@intCast(state.node), token) orelse break :taken;
                    if (!automaton.inside(written)) break :taken;
                    return try self.push(.{ .phase = state.phase, .tool = state.tool, .param = state.param, .node = written, .sub = next });
                }
                const close = if (state.phase == .arguments) self.close else self.value_close;
                if (token != close[0] or !try self.whole(sub, state.sub)) return null;
                return try self.push(self.closing(state, 1));
            },
            .close, .value_close => {
                const close = if (state.phase == .close) self.close else self.value_close;
                if (token != close[state.node]) return null;
                return try self.push(self.closing(state, state.node + 1));
            },
            .frame => {
                const child = self.treeOf(state).step(state.node, token) orelse return null;
                return try self.push(self.framed(state, child));
            },
            .choice => {
                const tree = self.treeOf(state);
                const child = tree.step(state.node, token) orelse return null;
                if (tree.nodes.items[child].leaf != none) return try self.push(valued(state));
                return try self.push(.{ .phase = .choice, .tool = state.tool, .param = state.param, .node = child });
            },
            .raw => {
                const next = self.strings.after(@intCast(state.node), token) orelse return null;
                if (next == automaton.done) return try self.push(valued(state));
                return try self.push(.{ .phase = .raw, .tool = state.tool, .param = state.param, .node = next });
            },
            .after => {
                if (self.isStop(token)) return try self.push(.{ .phase = .ended });
                if (self.single) return null;
                const child = self.later.step(0, token) orelse return null;
                return try self.push(self.opened(.later, &self.later, child));
            },
            .ended => return null,
        }
    }

    /// The state with `taken` tokens taken of the close `state` is in or begins: the JSON form's, or a value's.
    fn closing(self: *const Gate, state: State, taken: u32) State {
        const json_form = state.phase == .arguments or state.phase == .close;
        const close = if (json_form) self.close else self.value_close;
        if (taken == close.len) return if (json_form) .{ .phase = .after } else valued(state);
        return .{ .phase = if (json_form) .close else .value_close, .tool = state.tool, .param = state.param, .node = taken };
    }

    fn ended(context: *anyopaque, position: grammar.Position) bool {
        const self: *Gate = @ptrCast(@alignCast(context));
        const state = self.at(position) catch return false;
        return state.phase == .ended;
    }

    fn deinit(context: *anyopaque) void {
        const self: *Gate = @ptrCast(@alignCast(context));
        self.destroy();
    }

    const table: grammar.Constraint.Table = .{ .start = start, .fill = fill, .accept = accept, .ended = ended, .deinit = deinit };
};

/// `written` as the tokens that write it, or a refusal where they would not: text the tokenizer normalizes, or
/// that holds a token the reply's text leaves out, or one past the mask.
pub fn tokensOf(gpa: Allocator, tokenizer: *const Tokenizer, stops: []const u32, words: usize, written: []const u8, out: *std.ArrayList(u32)) (Allocator.Error || error{Unwritable})!void {
    out.clearRetainingCapacity();
    tokenizer.encode(gpa, written, out) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.InvalidUtf8 => return error.Unwritable,
    };
    for (out.items) |token| {
        if (std.mem.indexOfScalar(u32, stops, token) != null or token / 32 >= words) return error.Unwritable;
    }
    var back: std.ArrayList(u8) = .empty;
    defer back.deinit(gpa);
    try tokenizer.decode(gpa, out.items, .{ .skip_special = true }, &back);
    if (!std.mem.eql(u8, back.items, written)) return error.Unwritable;
}

// -- a tool's schema in the template's form ---------------------------------------------------------------------

const Plan = call_gate_plan.Plan;
const planOf = call_gate_plan.planOf;
const nameable = call_gate_plan.nameable;

/// The constraint of a reply that calls one of `tools`: `engine` compiles their `parameters`, `tokenizer` is
/// the one `engine` was made for, `stops` the tokens that end a reply (one at least). The constraint must not
/// outlive `engine` or `options.strings`; it serves one reply at a time, and like the engine's own it must not
/// be asked while another of the engine's constraints is.
///
/// Refused, with `diagnostic` saying why: no tool, two tools named alike (case aside, as the reply's parser
/// folds it), a name the tokenizer cannot write back, and whatever `engine` refuses in a tool's `parameters`.
pub fn compile(
    gpa: Allocator,
    engine: grammar.Engine,
    tokenizer: *const Tokenizer,
    stops: []const u32,
    tools: []const Tool,
    options: Options,
    diagnostic: *grammar.Diagnostic,
) grammar.CompileError!grammar.Constraint {
    if (tools.len == 0) return diagnostic.fail(error.InvalidSchema, "a required tool call needs a tool", .{});
    if (stops.len == 0) return diagnostic.fail(error.InvalidSchema, "the model has no token that ends a reply", .{});
    for (tools, 0..) |tool, index| {
        if (!std.unicode.utf8ValidateSlice(tool.name)) return diagnostic.fail(error.InvalidSchema, "a tool's name is not UTF-8", .{});
        for (tools[0..index]) |other| {
            if (std.ascii.eqlIgnoreCase(tool.name, other.name)) {
                return diagnostic.fail(error.InvalidSchema, "two tools are named '{s}': a required call could not say which it calls", .{tool.name});
            }
        }
    }

    const self = try gpa.create(Gate);
    self.* = .{
        .gpa = gpa,
        .words = engine.words,
        .single = options.single,
        .max_states = options.max_states,
        .stops = &.{},
        .origins = &.{},
        .close = &.{},
        .value_close = &.{},
        .strings = undefined,
    };
    // until the gate has its strings there are none of its own to free
    if (options.strings) |strings| {
        std.debug.assert(strings.words == engine.words and strings.tokenizer == tokenizer);
        self.strings = strings;
    } else {
        const strings = gpa.create(Strings) catch |err| {
            gpa.destroy(self);
            return err;
        };
        strings.* = Strings.init(gpa, tokenizer, engine.words, stops) catch |err| {
            gpa.destroy(strings);
            gpa.destroy(self);
            return err;
        };
        self.strings = strings;
        self.owns_strings = true;
    }
    errdefer self.destroy();
    self.stops = try gpa.dupe(u32, stops);

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    const format = options.format;
    const unwritable = "this model's tokenizer cannot write a tool call's end";
    tokensOf(gpa, tokenizer, stops, engine.words, format.close, &tokens) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.Unwritable => return diagnostic.fail(error.InvalidSchema, unwritable, .{}),
    };
    self.close = try gpa.dupe(u32, tokens.items);
    // without the template's closer every tool is written as JSON
    var template_form = options.template_form;
    if (tokensOf(gpa, tokenizer, stops, engine.words, template.parameter_close, &tokens)) |_| {
        self.value_close = try gpa.dupe(u32, tokens.items);
    } else |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.Unwritable => template_form = false,
    }
    // a lead that could not be written back is no lead: the reply opens as it does without one
    const lead = if (tokensOf(gpa, tokenizer, stops, engine.words, options.lead, &tokens)) |_| options.lead else |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.Unwritable => "",
    };

    for (tools, 0..) |tool, index| {
        // the whole schema first, in either form: what the engine refuses is refused with its place in it
        const whole = engine.compile(gpa, .{ .json_schema = tool.parameters }, diagnostic) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            else => |e| {
                const inner = diagnostic.*;
                return diagnostic.fail(e, "tool '{s}': its parameters: {s}", .{ tool.name, inner.message() });
            },
        };
        {
            errdefer whole.deinit();
            try self.subs.append(gpa, whole);
        }
        try self.tools.append(gpa, .{ .sub = @intCast(self.subs.items.len - 1) });
        const written = &self.tools.items[index];
        if (template_form and nameable(tool.name)) {
            if (try planOf(arena, tool.parameters)) |plans| try framesOf(self, engine, tokenizer, plans, written, &tokens);
        }

        const name = try std.json.Stringify.valueAlloc(arena, tool.name, .{});
        for ([_]struct { *Tree, []const u8 }{ .{ &self.first, lead }, .{ &self.later, format.between } }) |entry| {
            const opening = if (written.params != null)
                try std.mem.concat(arena, u8, &.{ entry[1], template.open, tool.name, template.opened })
            else
                try std.mem.concat(arena, u8, &.{ entry[1], format.open, name, format.middle });
            tokensOf(gpa, tokenizer, stops, engine.words, opening, &tokens) catch |err| switch (err) {
                error.OutOfMemory => |e| return e,
                error.Unwritable => return diagnostic.fail(error.InvalidSchema, "tool '{s}': this model's tokenizer cannot write a call to that name", .{tool.name}),
            };
            if (!try entry[0].add(gpa, tokens.items, @intCast(index))) {
                return diagnostic.fail(error.InvalidSchema, "tool '{s}': a call to it cannot be told from a call to another tool", .{tool.name});
            }
        }
    }
    self.origins = try gpa.alloc(grammar.Position, self.subs.items.len);
    return .{ .context = self, .words = engine.words, .table = &Gate.table };
}

/// Makes `written` the template's form of `plans`: the frames' trees, the values' trees and constraints. Where
/// the tokenizer cannot write a frame or a value, or two frames could not be told apart, nothing is kept and
/// `written` stays in the JSON form.
fn framesOf(self: *Gate, engine: grammar.Engine, tokenizer: *const Tokenizer, plans: []const Plan, written: *Written, tokens: *std.ArrayList(u32)) grammar.CompileError!void {
    const gpa = self.gpa;
    const trees = self.trees.items.len;
    const subs = self.subs.items.len;
    const params = try gpa.alloc(Param, plans.len);
    var kept = false;
    defer if (!kept) {
        gpa.free(params);
        for (self.trees.items[trees..]) |*tree| tree.deinit(gpa);
        self.trees.shrinkRetainingCapacity(trees);
        for (self.subs.items[subs..]) |sub| sub.deinit();
        self.subs.shrinkRetainingCapacity(subs);
    };
    var scratch: std.ArrayList(u8) = .empty;
    defer scratch.deinit(gpa);

    // before each parameter and after the last: the frames that may come there
    try self.trees.ensureUnusedCapacity(gpa, plans.len + 1);
    for (0..plans.len + 1) |from| {
        self.trees.appendAssumeCapacity(.{});
        const tree = &self.trees.items[self.trees.items.len - 1];
        const ends = for (plans[from..], from..) |plan, at| {
            scratch.clearRetainingCapacity();
            try scratch.appendSlice(gpa, template.parameter);
            try scratch.appendSlice(gpa, plan.key);
            try scratch.appendSlice(gpa, template.opened);
            tokensOf(gpa, tokenizer, self.stops, self.words, scratch.items, tokens) catch |err| switch (err) {
                error.OutOfMemory => |e| return e,
                error.Unwritable => return,
            };
            if (!try tree.add(gpa, tokens.items, @intCast(at))) return;
            if (plan.required) break false;
        } else true;
        if (!ends) continue;
        tokensOf(gpa, tokenizer, self.stops, self.words, template.close, tokens) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            error.Unwritable => return,
        };
        if (!try tree.add(gpa, tokens.items, @intCast(plans.len))) return;
    }

    for (plans, params) |plan, *param| switch (plan.value) {
        .raw => param.* = .{ .kind = .raw },
        .choice => |values| {
            try self.trees.append(gpa, .{});
            const tree = &self.trees.items[self.trees.items.len - 1];
            for (values) |value| {
                scratch.clearRetainingCapacity();
                try scratch.appendSlice(gpa, value);
                try scratch.appendSlice(gpa, template.parameter_close);
                tokensOf(gpa, tokenizer, self.stops, self.words, scratch.items, tokens) catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.Unwritable => return,
                };
                // a value listed twice is one value
                if (!try tree.add(gpa, tokens.items, 0) and !listedBefore(values, value)) return;
            }
            param.* = .{ .kind = .choice, .index = @intCast(self.trees.items.len - 1) };
        },
        .json => |schema| {
            // the engine took the whole schema, and this is a part of it without a reference out of itself
            var said: grammar.Diagnostic = .{};
            const sub = engine.compile(gpa, .{ .json_schema = schema }, &said) catch |err| switch (err) {
                error.OutOfMemory => |e| return e,
                else => return,
            };
            {
                errdefer sub.deinit();
                try self.subs.append(gpa, sub);
            }
            param.* = .{ .kind = .json, .index = @intCast(self.subs.items.len - 1) };
        },
    };
    written.params = params;
    written.frames = @intCast(trees);
    kept = true;
}

/// Whether `value` is among `values` before its own place.
fn listedBefore(values: []const []const u8, value: []const u8) bool {
    for (values) |other| {
        if (other.ptr == value.ptr) return false;
        if (std.mem.eql(u8, other, value)) return true;
    }
    return false;
}
