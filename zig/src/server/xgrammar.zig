//! A structure the host compiled through xgrammar (TensorFold 1.0.12's `libtfgrammar`, dlopened by its
//! `lanes/grammar.zig`; spec 106) as a `text/grammar.zig` constraint, so the model decodes under it as under
//! zspark's own matcher: a regex, a choice, an EBNF grammar, a structural tag (a required call), and a JSON
//! schema or object, once the library is there. Without it the host hands zspark's matcher the schema and
//! the call instead (`zig/vendor/tensorfold/PATCHES.md`).
//!
//! xgrammar's matcher has one place, which it leaves by taking a token and goes back to by `rollback`; a
//! `grammar.Position` is a value that stays good after a token is taken from it. Here a position is a node of
//! a tree of the tokens taken from the start: a node is its parent's position and one token. Asked at a node,
//! the matcher is rolled back to where its place and the node's path part and then takes the node's tokens
//! from there; a window's drafts, walked one after another from the newest committed token, cost a rollback
//! of a row or two each. A mask depends on the node's path alone, so a row's mask is its own.

const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const grammar = @import("../text/grammar.zig");

/// The library's kinds, which are the request's in the same order.
pub fn kindOf(kind: @FieldType(api.Structure, "kind")) api.grammar.Kind {
    return std.meta.stringToEnum(api.grammar.Kind, @tagName(kind)).?;
}

/// A token taken from `parent`'s position; node 0 is the start and has none.
const Node = struct {
    parent: u32,
    token: u32,
    /// A token that ends the reply was taken on the way here (xgrammar's `IsTerminated`).
    ended: bool,
};

const Walk = struct {
    gpa: Allocator,
    rules: api.grammar.Rules,
    /// The decoder's mask length; xgrammar's own (`own`) may differ by the vocabulary's padding.
    words: usize,
    own: []u32,
    nodes: std.ArrayList(Node) = .empty,
    /// The matcher's place: the nodes from the start's first token to its newest.
    place: std.ArrayList(u32) = .empty,
    path: std.ArrayList(u32) = .empty,

    const table: grammar.Constraint.Table = .{ .start = start, .fill = fill, .accept = accept, .ended = ended, .deinit = deinit };

    fn of(context: *anyopaque) *Walk {
        return @ptrCast(@alignCast(context));
    }

    fn start(context: *anyopaque) grammar.Error!grammar.Position {
        const self = of(context);
        self.rules.vtable.rollback(self.rules.ptr, self.place.items.len) catch return error.ConstraintFailed;
        self.place.clearRetainingCapacity();
        self.nodes.clearRetainingCapacity();
        try self.nodes.append(self.gpa, .{ .parent = 0, .token = 0, .ended = false });
        return @fromBackingInt(@intCast(0));
    }

    /// The node `at` names, or a failure where it is not one of this reply's.
    fn node(self: *const Walk, at: grammar.Position) grammar.Error!u32 {
        const index = @backingInt(at);
        if (index >= self.nodes.items.len) return error.ConstraintFailed;
        return @intCast(index);
    }

    /// The matcher at node `to`: back to where its place and `to`'s path part, then on along `to`'s path.
    fn reach(self: *Walk, to: u32) grammar.Error!void {
        const place = self.place.items;
        if ((place.len == 0 and to == 0) or (place.len > 0 and place[place.len - 1] == to)) return;
        self.path.clearRetainingCapacity();
        var at = to;
        while (at != 0) : (at = self.nodes.items[at].parent) try self.path.append(self.gpa, at);
        std.mem.reverse(u32, self.path.items);
        const path = self.path.items;
        var shared: usize = 0;
        while (shared < @min(path.len, place.len) and path[shared] == place[shared]) shared += 1;
        self.rules.vtable.rollback(self.rules.ptr, place.len - shared) catch return error.ConstraintFailed;
        self.place.shrinkRetainingCapacity(shared);
        for (path[shared..]) |each| {
            // every node was taken once from its parent, so it is taken again
            const taken = self.rules.vtable.accept(self.rules.ptr, self.nodes.items[each].token) catch return error.ConstraintFailed;
            if (!taken) return error.ConstraintFailed;
            try self.place.append(self.gpa, each);
        }
    }

    fn fill(context: *anyopaque, at: grammar.Position, mask: []u32) grammar.Error!void {
        const self = of(context);
        const index = try self.node(at);
        @memset(mask, 0);
        if (self.nodes.items[index].ended) return;
        try self.reach(index);
        self.rules.vtable.fill(self.rules.ptr, self.own) catch return error.ConstraintFailed;
        const n = @min(mask.len, self.own.len);
        @memcpy(mask[0..n], self.own[0..n]);
    }

    fn accept(context: *anyopaque, at: grammar.Position, token: u32) grammar.Error!?grammar.Position {
        const self = of(context);
        const index = try self.node(at);
        if (self.nodes.items[index].ended or token >= self.words * 32) return null;
        try self.reach(index);
        const taken = self.rules.vtable.accept(self.rules.ptr, token) catch return error.ConstraintFailed;
        if (!taken) return null;
        const next: u32 = @intCast(self.nodes.items.len);
        try self.nodes.append(self.gpa, .{ .parent = index, .token = token, .ended = self.rules.vtable.terminated(self.rules.ptr) });
        try self.place.append(self.gpa, next);
        return @fromBackingInt(@intCast(next));
    }

    fn ended(context: *anyopaque, at: grammar.Position) bool {
        const self = of(context);
        const index = self.node(at) catch return true;
        return self.nodes.items[index].ended;
    }

    fn deinit(context: *anyopaque) void {
        const self = of(context);
        self.rules.vtable.free(self.rules.ptr);
        self.nodes.deinit(self.gpa);
        self.place.deinit(self.gpa);
        self.path.deinit(self.gpa);
        self.gpa.free(self.own);
        self.gpa.destroy(self);
    }
};

/// `text` of `kind` through `compiler` (a cache hit: the host compiled it already) as a constraint of masks
/// `words` long, or `error.Grammar` with xgrammar's words in `said`.
pub fn compile(gpa: Allocator, io: std.Io, compiler: *api.grammar.Compiler, kind: api.grammar.Kind, text: []const u8, words: usize, said: *[512]u8) (Allocator.Error || error{Grammar})!grammar.Constraint {
    const compiled = compiler.compile(io, kind, text, said) catch return error.Grammar;
    defer compiled.free();
    const matcher = compiled.matcher() catch return error.Grammar;
    errdefer matcher.free();
    const own = try gpa.alloc(u32, compiler.words);
    errdefer gpa.free(own);
    const walk = try gpa.create(Walk);
    walk.* = .{ .gpa = gpa, .rules = matcher.rules(), .words = words, .own = own };
    return .{ .context = walk, .words = words, .table = &Walk.table };
}
