//! What the decoder asks of a grammar engine, and nothing of the engine itself (spec 017): a request's
//! structured output compiled for a vocabulary, a mask over the vocabulary at a position, and the position after
//! a token. `json_schema.zig` is the one engine behind it; xgrammar's `GrammarMatcher` would be another, with
//! `rollback` where this has values.
//!
//! A `Position` is a value. Taking a token gives a new position and leaves the old one as it was, so the tokens
//! a window drafts are walked one after another from the position of the sequence's newest token, each row
//! keeps the position its mask is made at, and a draft that is not kept costs nothing to forget. A mask depends
//! on its position alone: on the tokens before the row, never on the rows beside it.
//!
//! A mask is `Constraint.words` 32-bit words; token `t` is bit `t % 32` of word `t / 32` (xgrammar's layout),
//! set where the token may come next. The tokens that end a reply are set where the document is whole, and
//! nowhere else; after one of them nothing is.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// What a request asks its reply to be.
pub const Spec = union(enum) {
    /// `response_format` `json_object`: any JSON object (OpenAI's contract: not an array, not a scalar).
    json_object,
    /// `response_format` `json_schema`: the schema's JSON text.
    json_schema: []const u8,
};

/// A spec that is malformed, or that says what the engine does not read. Both are the client's (a 400).
pub const CompileError = error{ InvalidSchema, UnsupportedSchema } || Allocator.Error;

/// A constraint could not go on: the engine failed, or was given a position that is not its own. The reply ends
/// with it; the session and the server go on.
pub const Error = error{ConstraintFailed} || Allocator.Error;

/// Why a spec was refused, in words for the client.
pub const Diagnostic = struct {
    buffer: [320]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Diagnostic) []const u8 {
        return self.buffer[0..self.len];
    }

    pub fn fail(self: *Diagnostic, err: CompileError, comptime format: []const u8, args: anytype) CompileError {
        var writer: std.Io.Writer = .fixed(&self.buffer);
        writer.print(format, args) catch {};
        self.len = writer.end;
        // a message cut where the buffer ends is not cut inside a character
        for (0..3) |_| {
            if (self.len == 0 or std.unicode.utf8ValidateSlice(self.buffer[0..self.len])) break;
            self.len -= 1;
        }
        return err;
    }
};

/// A place in a constraint's documents: after the tokens taken to reach it. The engine's own, opaque here.
pub const Position = enum(u64) { _ };

/// A grammar engine with a tokenizer's vocabulary in it, made once for a model.
pub const Engine = struct {
    context: *anyopaque,
    /// The mask's length in words: the logits' width over 32, rounded up.
    words: usize,
    compileFn: *const fn (context: *anyopaque, gpa: Allocator, spec: Spec, diagnostic: *Diagnostic) CompileError!Constraint,

    /// `spec` for one reply at a time, or the refusal with `diagnostic` saying why. The constraint must not
    /// outlive the engine, and two constraints of one engine must not be asked (`start`, `fill`, `accept`) at
    /// the same time: their masks share the engine's caches. Compiling and freeing write nothing shared, so
    /// they may run beside one another and beside a constraint being asked: the server compiles on a
    /// connection's task and asks on its one decoding thread (`server/scheduler.zig`).
    pub fn compile(self: Engine, gpa: Allocator, spec: Spec, diagnostic: *Diagnostic) CompileError!Constraint {
        return self.compileFn(self.context, gpa, spec, diagnostic);
    }
};

/// A compiled spec following one reply.
pub const Constraint = struct {
    context: *anyopaque,
    words: usize,
    table: *const Table,

    pub const Table = struct {
        start: *const fn (context: *anyopaque) Error!Position,
        fill: *const fn (context: *anyopaque, at: Position, mask: []u32) Error!void,
        accept: *const fn (context: *anyopaque, at: Position, token: u32) Error!?Position,
        ended: *const fn (context: *anyopaque, at: Position) bool,
        deinit: *const fn (context: *anyopaque) void,
    };

    /// The position before a reply's first token. Every position given before is forgotten: call it once a
    /// reply, and the constraint serves one reply after another.
    pub fn start(self: Constraint) Error!Position {
        return self.table.start(self.context);
    }

    /// Sets `mask` (`words` long) to the tokens that may come at `at`.
    pub fn fill(self: Constraint, at: Position, mask: []u32) Error!void {
        std.debug.assert(mask.len == self.words);
        return self.table.fill(self.context, at, mask);
    }

    /// The position after `token` at `at`, or null where the token may not come: exactly where its bit in
    /// `fill`'s mask is clear. `at` stays good.
    pub fn accept(self: Constraint, at: Position, token: u32) Error!?Position {
        return self.table.accept(self.context, at, token);
    }

    /// A token that ends the reply was taken on the way to `at`: nothing follows.
    pub fn ended(self: Constraint, at: Position) bool {
        return self.table.ended(self.context, at);
    }

    pub fn deinit(self: Constraint) void {
        self.table.deinit(self.context);
    }
};

/// A constraint that begins with tokens of its own, each the only one allowed at its place, and then is `inner`
/// from its start (spec 035): a document after the text a chat template writes between a reply's reasoning and
/// its content, so that the reply is decoded as the next turn's prompt renders it. A mask with one bit is a
/// draft the decoder supplies (`masks.zig` `Guide`), so the lead's tokens are drafted, not chosen.
///
/// Its positions are `inner`'s, and the `lead.len` highest values of a `Position` for the places inside the
/// lead: an engine's positions stay below them (`json_schema.zig`'s are 33 bits wide), and one that does not
/// fails the reply at its start.
pub const Led = struct {
    gpa: Allocator,
    inner: Constraint,
    /// The tokens before `inner`'s first; owned.
    lead: []const u32,
    /// `inner`'s start for the reply being followed.
    first: Position = undefined,

    /// `inner` after `lead`. The result owns `inner` and frees it with itself; on an error `inner` is the
    /// caller's still. An empty lead is `inner` itself.
    pub fn wrap(gpa: Allocator, inner: Constraint, lead: []const u32) Allocator.Error!Constraint {
        if (lead.len == 0) return inner;
        const self = try gpa.create(Led);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .inner = inner, .lead = try gpa.dupe(u32, lead) };
        return .{ .context = self, .words = inner.words, .table = &table };
    }

    const table: Constraint.Table = .{ .start = start, .fill = fill, .accept = accept, .ended = ended, .deinit = destroy };

    /// The place before lead token `index`.
    fn before(index: usize) Position {
        return @fromBackingInt(std.math.maxInt(u64) - @as(u64, index));
    }

    /// Which lead token `at` stands before, or null for a position of `inner`'s.
    fn within(self: *const Led, at: Position) ?usize {
        const back = std.math.maxInt(u64) - @backingInt(at);
        return if (back < self.lead.len) @intCast(back) else null;
    }

    fn start(context: *anyopaque) Error!Position {
        const self: *Led = @ptrCast(@alignCast(context));
        // `inner` forgets its positions at its start, so it starts here, once a reply, and not where the
        // lead's last token is taken: a draft walked twice takes that token twice.
        self.first = try self.inner.start();
        if (self.within(self.first) != null) return error.ConstraintFailed;
        return before(0);
    }

    fn fill(context: *anyopaque, at: Position, mask: []u32) Error!void {
        const self: *Led = @ptrCast(@alignCast(context));
        const index = self.within(at) orelse return self.inner.fill(at, mask);
        const token = self.lead[index];
        if (token / 32 >= mask.len) return error.ConstraintFailed;
        @memset(mask, 0);
        mask[token / 32] = @as(u32, 1) << @intCast(token % 32);
    }

    fn accept(context: *anyopaque, at: Position, token: u32) Error!?Position {
        const self: *Led = @ptrCast(@alignCast(context));
        const index = self.within(at) orelse return self.inner.accept(at, token);
        if (token != self.lead[index]) return null;
        return if (index + 1 < self.lead.len) before(index + 1) else self.first;
    }

    fn ended(context: *anyopaque, at: Position) bool {
        const self: *Led = @ptrCast(@alignCast(context));
        return self.within(at) == null and self.inner.ended(at);
    }

    fn destroy(context: *anyopaque) void {
        const self: *Led = @ptrCast(@alignCast(context));
        const gpa = self.gpa;
        self.inner.deinit();
        gpa.free(self.lead);
        gpa.destroy(self);
    }
};

/// Whether `mask` has token `id`.
pub fn allowed(mask: []const u32, id: u32) bool {
    return mask[id / 32] >> @intCast(id % 32) & 1 != 0;
}
