//! A JSON schema as a mask over a tokenizer's vocabulary: which tokens may follow the tokens chosen so far.
//! Spec 017's engine, measured against xgrammar (WS-01) and written to give xgrammar's masks on the schemas it
//! reads. The decoder calls it through `grammar.zig`'s interface, which `Engine` here implements.
//!
//! The documents accepted are the ones xgrammar's `compile_json_schema` accepts in strict mode with any
//! whitespace, as TensorFold calls it (`engine/grammar.py`):
//!
//! 1. An object's properties come in the schema's order, the optional ones may be left out, and no other key
//!    is taken, also when `properties` lists nothing. An object without `properties` takes any keys and values.
//! 2. Up to `Options.blanks` of space, tab, newline and return go after `{`, `[`, `,` and `:` and before `}`,
//!    `]`, `,` and `:`; none before the document or after it.
//! 3. A string is `"`, then any character from U+0020 on but `"` and `\`, or `\` and one of `"\/bfnrt` or `u`
//!    and four hex digits, then `"`. An integer is `0` or `-?[1-9][0-9]*`; a number has JSON's form.
//! 4. `enum` and `const` are their values written compactly, and nothing else.
//!
//! What is read: `type` (a name or a list of names), `properties`, `required`, `items`, `enum`, `const`,
//! `minItems`, `maxItems`, `additionalProperties` (beside no `properties`), `$ref` to a place in the schema
//! itself (`#/$defs/Name`, any JSON pointer; recursion included) with `$defs` and `definitions`, `anyOf` and
//! `oneOf` whose branches all begin with different bytes (a string, a number, an object, an array, `true`,
//! `false`, `null`: what `Optional[...]` writes), and the annotations. Any other keyword (`allOf`, `pattern`,
//! `format`, `minimum`, ...) is refused with its name, and so are branches that begin alike: never approximated.
//!
//! How a mask is made. The schema becomes a tree of nodes, and a document's position a stack of frames, one a
//! value that is open; every byte has one way to go on, so a position is one stack. Frames are never changed,
//! only added, each naming the frame under it: a position is an index, a step back is the index before, and
//! many tokens can be tried from one position. A mask is filled in two parts. What a token does while it stays
//! inside the top frame depends on that frame alone, so it is worked out once for the frame's contents and kept
//! (`Grammar.cache`, and `Vocabulary.common` for the frames every schema has): the tokens taken, and the tokens
//! that close the frame with bytes left over. Only those are run again, from the frame below, each time; where
//! the frame may end before a token (a number), the frame below's own mask stands for them. The first part
//! walks the vocabulary in byte order, so tokens that share a prefix share its steps, and a prefix that is
//! refused refuses all of them.
//!
//! This file holds the vocabulary, the masks (`Grammar`, `Matcher`) and `Engine`; the schema's tree and its
//! `Builder` are `json_builder.zig`, the frames and `Machine` `json_machine.zig`. The tests are
//! `json_schema_test.zig`, `json_schema_prop_test.zig` and `json_schema_oracle_test.zig`; `Validator`, the second
//! reading of a schema they hold the masks to, is `zig/tests/text/json_schema_validator.zig`.

const std = @import("std");
const builtin = @import("builtin");
const grammar_file = @import("grammar.zig");
const tokenizer_file = @import("tokenizer.zig");
const Allocator = std.mem.Allocator;
const json_builder = @import("json_builder.zig");
const json_machine = @import("json_machine.zig");
const Schema = json_builder.Schema;
const fixed = json_builder.fixed;
const parseSchema = json_builder.parseSchema;
const Frame = json_machine.Frame;
const Machine = json_machine.Machine;
const Phase = json_machine.Phase;
const done = json_machine.done;
const outside = json_machine.outside;
const rejected = json_machine.rejected;
const unknown = json_machine.unknown;

pub const Tokenizer = tokenizer_file.Tokenizer;

pub const none = std.math.maxInt(u32);

/// Why a schema was refused.
pub const Diagnostic = grammar_file.Diagnostic;
pub const SchemaError = grammar_file.CompileError;

pub const Options = struct {
    /// The most blank characters at one place between a document's tokens (TensorFold's `BLANKS`).
    blanks: u8 = 32,
    /// The longest schema text read, in bytes: a longer one is refused.
    max_schema_bytes: usize = 1 << 20,
    /// The deepest the schema's own JSON may nest its objects and arrays. Schemas inside schemas stop at
    /// `max_depth` before it; this one bounds a deep `enum` value too.
    max_nesting: usize = 256,
    /// The longest `enum` or `const` value and the longest property name, in bytes of their JSON text.
    max_literal: usize = 1024,
    /// The most frames a reply's positions may take (`Engine`): past it the constraint fails, and the reply
    /// with it. A token adds at most two frames a byte, a draft that is walked and not kept included, so the
    /// default holds a reply of some hundred thousand tokens in 80 MiB.
    max_frames: usize = 1 << 22,
    /// A property's name is taken in the tokenizer's own tokens for it and no others (`Engine.keyTokens`), so
    /// that a reply is the tokens the next turn's prompt has for its text. False: every way of cutting a
    /// name's bytes into tokens is taken, which is xgrammar's mask to the bit.
    key_tokens: bool = true,
};

/// The deepest schemas nest inside schemas (`properties`, `items`, `anyOf`, a `$ref`'s target).
pub const max_depth = 64;

// -- the vocabulary ----------------------------------------------------------------------------------------------

/// A tokenizer's tokens in byte order, made once for a tokenizer.
pub const Vocabulary = struct {
    tokenizer: *const Tokenizer,
    /// The mask's width in tokens: the logits' width, which may be more than the tokenizer's count.
    width: usize,
    /// The ids a document may be made of, sorted by their bytes: not the special tokens, the stops, or an id
    /// without bytes.
    order: []u32,
    /// How many leading bytes `order[i]` has in common with `order[i - 1]`. `[order.len]`, the first 0.
    shared: []u16,
    /// The ids that end a reply: allowed where the document is complete, and nowhere else.
    stops: []u32,
    /// The longest token of `order` in bytes.
    longest: usize,
    /// `order[firsts[b]..firsts[b + 1]]` are the tokens whose first byte is `b`.
    firsts: [257]u32,
    /// What the tokens do in the frames every schema has (a string, a number, a value of any type): made by
    /// the first grammar that comes to one and read by all of them, so its grammars must not fill at the same
    /// time.
    common: Cache = .empty,
    gpa: Allocator,

    pub fn init(gpa: Allocator, tokenizer: *const Tokenizer, width: usize, stops: []const u32) Allocator.Error!Vocabulary {
        std.debug.assert(width >= tokenizer.count());
        var order: std.ArrayList(u32) = .empty;
        errdefer order.deinit(gpa);
        var longest: usize = 0;
        for (0..tokenizer.count()) |index| {
            const id: u32 = @intCast(index);
            const bytes = tokenizer.tokenBytes(id);
            if (bytes.len == 0 or tokenizer.isSpecial(id) or std.mem.findScalar(u32, stops, id) != null) continue;
            longest = @max(longest, bytes.len);
            try order.append(gpa, id);
        }
        std.debug.assert(longest <= std.math.maxInt(u16));
        const ByBytes = struct {
            tokenizer: *const Tokenizer,
            fn lessThan(self: @This(), left: u32, right: u32) bool {
                return switch (std.mem.order(u8, self.tokenizer.tokenBytes(left), self.tokenizer.tokenBytes(right))) {
                    .lt => true,
                    .gt => false,
                    .eq => left < right,
                };
            }
        };
        std.mem.sortUnstable(u32, order.items, ByBytes{ .tokenizer = tokenizer }, ByBytes.lessThan);

        const shared = try gpa.alloc(u16, order.items.len);
        errdefer gpa.free(shared);
        for (shared, 0..) |*count, at| {
            count.* = if (at == 0) 0 else @intCast(std.mem.indexOfDiff(u8, tokenizer.tokenBytes(order.items[at - 1]), tokenizer.tokenBytes(order.items[at])) orelse
                tokenizer.tokenBytes(order.items[at]).len);
        }
        var firsts: [257]u32 = undefined;
        var reached: u32 = 0;
        for (&firsts, 0..) |*first, byte| {
            while (reached < order.items.len and tokenizer.tokenBytes(order.items[reached])[0] < byte) reached += 1;
            first.* = reached;
        }
        const kept_stops = try gpa.dupe(u32, stops);
        errdefer gpa.free(kept_stops);
        return .{
            .gpa = gpa,
            .firsts = firsts,
            .tokenizer = tokenizer,
            .width = width,
            .order = try order.toOwnedSlice(gpa),
            .shared = shared,
            .stops = kept_stops,
            .longest = longest,
        };
    }

    pub fn deinit(self: *Vocabulary, gpa: Allocator) void {
        freeCache(gpa, &self.common);
        gpa.free(self.order);
        gpa.free(self.shared);
        gpa.free(self.stops);
        self.* = undefined;
    }

    /// A mask's length in 32-bit words: token `t` is bit `t % 32` of word `t / 32` (xgrammar's layout).
    pub fn words(self: *const Vocabulary) usize {
        return (self.width + 31) / 32;
    }

    fn isStop(self: *const Vocabulary, id: u32) bool {
        return std.mem.findScalar(u32, self.stops, id) != null;
    }
};

/// A top frame's contents, and the most blanks its schema lets a place have.
const Key = struct { node: u32, phase: Phase, blanks: u8, limit: u8, a: u32, b: u32 };

/// What the tokens do from a top frame with the contents of a key, whatever is under it.
const Entry = struct {
    /// The tokens taken without leaving the frame, or leaving it with their last byte: ids, or when `dense`,
    /// a whole mask.
    taken: []u32,
    dense: bool,
    /// The tokens the frame ends before, as a mask: taken where the frame below takes them. Empty for none.
    through: []u32,
    /// The tokens whose bytes from `at` on are for the frame below.
    pending: []Pending,
};
const Pending = struct { id: u32, at: u16 };
const Cache = std.AutoHashMapUnmanaged(Key, Entry);

fn freeCache(gpa: Allocator, cache: *Cache) void {
    var entries = cache.valueIterator();
    while (entries.next()) |entry| {
        gpa.free(entry.taken);
        gpa.free(entry.through);
        gpa.free(entry.pending);
    }
    cache.deinit(gpa);
}

fn bytesOf(cache: *const Cache) usize {
    var total: usize = 0;
    var entries = cache.valueIterator();
    while (entries.next()) |entry| total += (entry.taken.len + entry.through.len) * 4 + entry.pending.len * @sizeOf(Pending);
    return total;
}

fn setBit(mask: []u32, id: u32) void {
    mask[id / 32] |= @as(u32, 1) << @intCast(id % 32);
}

pub fn allowed(mask: []const u32, id: u32) bool {
    return mask[id / 32] >> @intCast(id % 32) & 1 != 0;
}

// -- masks -------------------------------------------------------------------------------------------------------

/// A schema compiled for a vocabulary, with what its masks have in common. One for a schema; its matchers
/// share it and must not fill at the same time.
pub const Grammar = struct {
    gpa: Allocator,
    schema: Schema,
    vocabulary: *Vocabulary,
    /// The frames of this schema's own nodes; the ones every schema has are in `Vocabulary.common`.
    cache: Cache = .empty,
    /// Walk scratch: the position after each byte of the token at hand, and the frames there were. `[longest + 1]`.
    states: []u32,
    marks: []u32,

    pub fn init(gpa: Allocator, vocabulary: *Vocabulary, schema: []const u8, options: Options, diagnostic: *Diagnostic) SchemaError!Grammar {
        var parsed = try parseSchema(gpa, schema, options, diagnostic);
        errdefer parsed.arena.deinit();
        const states = try gpa.alloc(u32, vocabulary.longest + 1);
        errdefer gpa.free(states);
        return .{ .gpa = gpa, .schema = parsed, .vocabulary = vocabulary, .states = states, .marks = try gpa.alloc(u32, vocabulary.longest + 1) };
    }

    pub fn deinit(self: *Grammar) void {
        freeCache(self.gpa, &self.cache);
        self.gpa.free(self.states);
        self.gpa.free(self.marks);
        self.schema.arena.deinit();
        self.* = undefined;
    }

    /// The bytes this schema's own cache holds, and the bytes of the one every schema shares.
    pub fn cacheBytes(self: *const Grammar) struct { own: usize, common: usize } {
        return .{ .own = bytesOf(&self.cache), .common = bytesOf(&self.vocabulary.common) };
    }

    fn lookup(self: *Grammar, machine: *Machine, top: u32) Allocator.Error!Entry {
        const frame = machine.frames.items[top];
        // a fixed node's frames do the same in every schema that allows as many blanks
        const common = frame.node <= fixed.words;
        const gpa = if (common) self.vocabulary.gpa else self.gpa;
        const cache = if (common) &self.vocabulary.common else &self.cache;
        const key: Key = .{ .node = frame.node, .phase = frame.phase, .blanks = frame.blanks, .limit = self.schema.blanks, .a = frame.a, .b = frame.b };
        if (cache.get(key)) |found| return found;
        try cache.ensureUnusedCapacity(gpa, 1);
        const made = try self.walkVocabulary(gpa, machine, frame);
        cache.putAssumeCapacity(key, made);
        return made;
    }

    fn walkVocabulary(self: *Grammar, gpa: Allocator, machine: *Machine, frame: Frame) Allocator.Error!Entry {
        const vocabulary = self.vocabulary;
        const tokenizer = vocabulary.tokenizer;
        const mark = machine.frames.items.len;
        defer machine.frames.shrinkRetainingCapacity(mark);
        var alone = frame;
        alone.parent = outside;

        var taken: std.ArrayList(u32) = .empty;
        defer taken.deinit(gpa);
        var pending: std.ArrayList(Pending) = .empty;
        defer pending.deinit(gpa);
        const words = vocabulary.words();
        var through: []u32 = &.{};
        errdefer gpa.free(through);

        self.states[0] = machine.put(alone);
        self.marks[0] = @intCast(machine.frames.items.len);
        // `states[0..walked + 1]` are the positions along the last token walked; `fate` is why its walk
        // stopped at `walked` bytes
        var walked: usize = 0;
        var fate: enum { whole, refused, below } = .whole;
        var at: usize = 0;
        while (at < vocabulary.order.len) : (at += 1) {
            const id = vocabulary.order[at];
            const shared = vocabulary.shared[at];
            if (shared <= walked) {
                const bytes = tokenizer.tokenBytes(id);
                walked = shared;
                fate = .whole;
                machine.frames.shrinkRetainingCapacity(self.marks[walked]);
                while (walked < bytes.len) {
                    const next = if (self.states[walked] == outside) unknown else machine.step(self.states[walked], bytes[walked]);
                    if (next == rejected) {
                        fate = .refused;
                        break;
                    }
                    if (next == unknown) {
                        fate = .below;
                        break;
                    }
                    walked += 1;
                    self.states[walked] = next;
                    self.marks[walked] = @intCast(machine.frames.items.len);
                }
                if (fate == .refused and walked == 0) {
                    // no token with this first byte is taken
                    at = vocabulary.firsts[@as(usize, bytes[0]) + 1] - 1;
                    continue;
                }
            }
            // a token that shares more than the bytes walked goes the way of the one before it
            switch (fate) {
                .whole => try taken.append(gpa, id),
                .refused => {},
                .below => if (walked == 0) {
                    if (through.len == 0) {
                        through = try gpa.alloc(u32, words);
                        @memset(through, 0);
                    }
                    setBit(through, id);
                } else try pending.append(gpa, .{ .id = id, .at = @intCast(walked) }),
            }
        }

        const dense = taken.items.len > words;
        const kept = try gpa.alloc(u32, if (dense) words else taken.items.len);
        errdefer gpa.free(kept);
        if (dense) {
            @memset(kept, 0);
            for (taken.items) |id| setBit(kept, id);
        } else @memcpy(kept, taken.items);
        return .{ .taken = kept, .dense = dense, .through = through, .pending = try pending.toOwnedSlice(gpa) };
    }
};

/// Masks filled and tokens asked of every matcher since the program began, counted in test builds alone: by it
/// a test shows that a reply without a constraint asks nothing.
pub var calls: usize = 0;

/// One reply's place in a grammar's documents: the mask for the next token, and the tokens taken.
pub const Matcher = struct {
    grammar: *Grammar,
    machine: Machine,
    /// The position before any token, then after each token taken.
    tops: std.ArrayList(u32) = .empty,
    /// The frames there were at each of `tops`.
    marks: std.ArrayList(u32) = .empty,
    /// A stop token was taken: nothing follows.
    ended: bool = false,

    pub fn init(grammar: *Grammar) Allocator.Error!Matcher {
        var self: Matcher = .{ .grammar = grammar, .machine = .{ .schema = &grammar.schema } };
        errdefer self.deinit();
        try self.machine.frames.ensureUnusedCapacity(grammar.gpa, 1);
        const top = self.machine.put(.{ .parent = done, .node = grammar.schema.root, .phase = .start });
        try self.tops.append(grammar.gpa, top);
        try self.marks.append(grammar.gpa, 1);
        return self;
    }

    pub fn deinit(self: *Matcher) void {
        self.machine.frames.deinit(self.grammar.gpa);
        self.tops.deinit(self.grammar.gpa);
        self.marks.deinit(self.grammar.gpa);
        self.* = undefined;
    }

    /// A place in the grammar's documents, as a value: the top frame, and whether a stop token was taken on
    /// the way. Frames are never changed, so a position stays good while frames are only added (`fillAt`,
    /// `acceptAt`), until `reset` or `rollback`.
    pub const Position = struct { top: u32, ended: bool = false };

    /// The position before any token.
    pub fn origin(self: *const Matcher) Position {
        return .{ .top = self.tops.items[0] };
    }

    /// Forgets every token taken and every position given.
    pub fn reset(self: *Matcher) void {
        self.rollback(self.tops.items.len - 1);
        self.ended = false;
    }

    fn position(self: *const Matcher) Position {
        return .{ .top = self.tops.items[self.tops.items.len - 1], .ended = self.ended };
    }

    /// The document is whole at `at`, or could be: a stop token may come.
    pub fn completeAt(self: *const Matcher, at: Position) bool {
        if (at.ended) return false;
        if (at.top == done) return true;
        return self.machine.frames.items[at.top].parent == done and self.machine.mayEnd(at.top);
    }

    /// The document is whole, or could be: a stop token may come.
    pub fn complete(self: *const Matcher) bool {
        return self.completeAt(self.position());
    }

    /// A stop token was taken.
    pub fn finished(self: *const Matcher) bool {
        return self.ended;
    }

    fn reserve(self: *Matcher, bytes: usize) Allocator.Error!void {
        try self.machine.frames.ensureUnusedCapacity(self.grammar.gpa, 2 * bytes + 2);
    }

    /// Sets `mask` (`Vocabulary.words` long) to the tokens that may come next. The first fill at a kind of
    /// place walks the vocabulary and allocates; the fills after it do neither.
    pub fn fill(self: *Matcher, mask: []u32) Allocator.Error!void {
        return self.fillAt(self.position(), mask);
    }

    /// `fill` at a position: the mask is the position's alone, whatever was taken or filled since.
    pub fn fillAt(self: *Matcher, at: Position, mask: []u32) Allocator.Error!void {
        const vocabulary = self.grammar.vocabulary;
        std.debug.assert(mask.len == vocabulary.words());
        if (builtin.is_test) calls += 1;
        if (at.ended) return @memset(mask, 0);
        try self.reserve(vocabulary.longest);
        try self.fillFrom(at.top, mask);
        if (self.completeAt(at)) {
            for (vocabulary.stops) |id| setBit(mask, id);
        }
    }

    /// Sets `mask` to the tokens taken at position `at`, the stops aside.
    fn fillFrom(self: *Matcher, at: u32, mask: []u32) Allocator.Error!void {
        if (at == done) return @memset(mask, 0);
        const entry = try self.grammar.lookup(&self.machine, at);
        const below = self.machine.frames.items[at].parent;
        if (entry.through.len != 0) {
            // the frame may end here: of the tokens it does not begin, the ones the frame below takes
            try self.fillFrom(below, mask);
            for (mask, entry.through) |*word, through| word.* &= through;
            if (entry.dense) {
                for (mask, entry.taken) |*word, taken| word.* |= taken;
            } else for (entry.taken) |id| setBit(mask, id);
        } else if (entry.dense) {
            @memcpy(mask, entry.taken);
        } else {
            @memset(mask, 0);
            for (entry.taken) |id| setBit(mask, id);
        }
        const tokenizer = self.grammar.vocabulary.tokenizer;
        const mark = self.machine.frames.items.len;
        for (entry.pending) |pending| {
            if (self.machine.walk(below, tokenizer.tokenBytes(pending.id)[pending.at..]) == rejected) continue;
            setBit(mask, pending.id);
            self.machine.frames.shrinkRetainingCapacity(mark);
        }
    }

    /// The same mask with every token walked from the position itself and nothing kept: what `fill` must equal.
    pub fn fillSlowly(self: *Matcher, mask: []u32) Allocator.Error!void {
        const vocabulary = self.grammar.vocabulary;
        @memset(mask, 0);
        if (self.ended) return;
        if (self.complete()) {
            for (vocabulary.stops) |id| setBit(mask, id);
        }
        try self.reserve(vocabulary.longest);
        const mark = self.machine.frames.items.len;
        for (vocabulary.order) |id| {
            if (self.machine.walk(self.position().top, vocabulary.tokenizer.tokenBytes(id)) == rejected) continue;
            setBit(mask, id);
            self.machine.frames.shrinkRetainingCapacity(mark);
        }
    }

    /// Takes `token` if it may come next, and says whether it did.
    pub fn accept(self: *Matcher, token: u32) Allocator.Error!bool {
        const next = try self.acceptAt(self.position(), token) orelse return false;
        try self.tops.append(self.grammar.gpa, next.top);
        try self.marks.append(self.grammar.gpa, @intCast(self.machine.frames.items.len));
        self.ended = next.ended;
        return true;
    }

    /// The position after `token` at `at`, or null where it may not come. `at` stays good: the frames the
    /// token makes are added above every frame there is.
    pub fn acceptAt(self: *Matcher, at: Position, token: u32) Allocator.Error!?Position {
        const vocabulary = self.grammar.vocabulary;
        if (builtin.is_test) calls += 1;
        if (at.ended) return null;
        if (vocabulary.isStop(token)) return if (self.completeAt(at)) .{ .top = at.top, .ended = true } else null;
        const tokenizer = vocabulary.tokenizer;
        if (token >= tokenizer.count()) return null;
        const bytes = tokenizer.tokenBytes(token);
        if (bytes.len == 0 or tokenizer.isSpecial(token)) return null;
        try self.reserve(bytes.len);
        const next = self.machine.walk(at.top, bytes);
        return if (next == rejected) null else .{ .top = next };
    }

    /// Takes bytes that are not a token's: a forced prefix, a test's document.
    pub fn acceptBytes(self: *Matcher, bytes: []const u8) Allocator.Error!bool {
        if (self.ended) return false;
        try self.reserve(bytes.len);
        const next = self.machine.walk(self.position().top, bytes);
        if (next == rejected) return false;
        try self.tops.append(self.grammar.gpa, next);
        try self.marks.append(self.grammar.gpa, @intCast(self.machine.frames.items.len));
        return true;
    }

    /// Forgets the last `count` tokens taken.
    pub fn rollback(self: *Matcher, count: usize) void {
        std.debug.assert(count < self.tops.items.len);
        self.tops.shrinkRetainingCapacity(self.tops.items.len - count);
        self.marks.shrinkRetainingCapacity(self.marks.items.len - count);
        self.machine.frames.shrinkRetainingCapacity(self.marks.items[self.marks.items.len - 1]);
        if (count > 0) self.ended = false;
    }
};

// -- the engine --------------------------------------------------------------------------------------------------

/// The matcher as `grammar.zig`'s engine: a tokenizer's vocabulary, and the constraints compiled for it. One for
/// a model; its constraints follow one reply at a time and must not be asked at the same time.
pub const Engine = struct {
    vocabulary: Vocabulary,
    options: Options = .{},

    /// `width` is the logits' width and `stops` the tokens that end a reply; `tokenizer` outlives the engine.
    pub fn init(gpa: Allocator, tokenizer: *const Tokenizer, width: usize, stops: []const u32) Allocator.Error!Engine {
        return .{ .vocabulary = try .init(gpa, tokenizer, width, stops) };
    }

    pub fn deinit(self: *Engine, gpa: Allocator) void {
        self.vocabulary.deinit(gpa);
        self.* = undefined;
    }

    /// Call it on the engine where it will stay: the interface and its constraints point at it.
    pub fn interface(self: *Engine) grammar_file.Engine {
        return .{ .context = self, .words = self.vocabulary.words(), .compileFn = compile };
    }

    /// TensorFold's `OBJECT`: any JSON object, not an array.
    const object_schema = "{\"type\": \"object\"}";

    /// A grammar and the matcher that follows a reply through it, together on the heap.
    pub const Compiled = struct {
        gpa: Allocator,
        grammar: Grammar,
        matcher: Matcher,
        max_frames: usize,
        key_tokens: bool,
    };

    fn compile(context: *anyopaque, gpa: Allocator, spec: grammar_file.Spec, diagnostic: *Diagnostic) SchemaError!grammar_file.Constraint {
        const self: *Engine = @ptrCast(@alignCast(context));
        const text = switch (spec) {
            .json_object => object_schema,
            .json_schema => |schema| schema,
        };
        const made = try gpa.create(Compiled);
        errdefer gpa.destroy(made);
        made.gpa = gpa;
        made.max_frames = self.options.max_frames;
        made.key_tokens = self.options.key_tokens;
        made.grammar = try .init(gpa, &self.vocabulary, text, self.options, diagnostic);
        errdefer made.grammar.deinit();
        made.matcher = try .init(&made.grammar);
        return .{ .context = made, .words = self.vocabulary.words(), .table = &table };
    }

    const table: grammar_file.Constraint.Table = .{ .start = start, .fill = fill, .accept = accept, .ended = ended, .deinit = destroy };

    // a position's 64 bits: the top frame, then whether a stop was taken
    fn pack(at: Matcher.Position) grammar_file.Position {
        return @fromBackingInt(@intCast(@as(u64, at.top) | @as(u64, @intFromBool(at.ended)) << 32));
    }

    fn unpack(at: grammar_file.Position) Matcher.Position {
        const bits = @backingInt(at);
        return .{ .top = @truncate(bits), .ended = bits >> 32 != 0 };
    }

    /// A position that is not this reply's (one from before `start`) is refused, not followed; and so is any
    /// position once the reply's frames pass `Options.max_frames`.
    fn own(made: *const Compiled, at: Matcher.Position) grammar_file.Error!void {
        const frames = made.matcher.machine.frames.items.len;
        if (at.top != done and at.top >= frames) return error.ConstraintFailed;
        if (frames > made.max_frames) return error.ConstraintFailed;
    }

    fn start(context: *anyopaque) grammar_file.Error!grammar_file.Position {
        const made: *Compiled = @ptrCast(@alignCast(context));
        made.matcher.reset();
        return pack(made.matcher.origin());
    }

    fn fill(context: *anyopaque, at: grammar_file.Position, mask: []u32) grammar_file.Error!void {
        const made: *Compiled = @ptrCast(@alignCast(context));
        try own(made, unpack(at));
        try made.matcher.fillAt(unpack(at), mask);
        var keys: [max_keys]u32 = undefined;
        if (try keyTokens(made, unpack(at).top, &keys)) |tokens| {
            @memset(mask, 0);
            for (tokens) |id| setBit(mask, id);
        }
    }

    fn accept(context: *anyopaque, at: grammar_file.Position, token: u32) grammar_file.Error!?grammar_file.Position {
        const made: *Compiled = @ptrCast(@alignCast(context));
        try own(made, unpack(at));
        var keys: [max_keys]u32 = undefined;
        if (try keyTokens(made, unpack(at).top, &keys)) |tokens| {
            if (std.mem.indexOfScalar(u32, tokens, token) == null) return null;
        }
        return pack(try made.matcher.acceptAt(unpack(at), token) orelse return null);
    }

    /// The most keys that may come at one place for the place to be narrowed (`keyTokens`).
    const max_keys = 64;

    /// What stands around a key when its tokens are asked of the tokenizer: after `{` and before `:`, between
    /// blanks, and on a line of its own. A token the three agree on is the key's own, whatever is around it.
    const key_contexts = [_][2][]const u8{ .{ "{", ":" }, .{ " ", " :" }, .{ "\n", "\n" } };

    /// Inside a property's name the text is the schema's, so the tokens are held to the tokenizer's own for
    /// it (spec 035): the ones at `top`, in `out`, or null where the place is as the machine has it. A reply's
    /// document goes back into the next turn's prompt as text, and that prompt holds the tokenizer's tokens
    /// for it; a model that is made to write a key it did not choose spells it out in pieces otherwise
    /// (`answer` as three tokens), and the turn after runs the reply again. With one key to come the mask is
    /// one token, which the decoder drafts (`masks.zig` `Guide`).
    ///
    /// Narrowed: a place inside a key, after its opening quote, where every key that may still come has a
    /// token of its own starting there and ending before its closing quote. That token is found by encoding
    /// the key in `key_contexts`: the quotes' own tokens depend on what is around them (`{"`, ` "`, `":`),
    /// so a place inside one of those, or inside a token the model split otherwise, is left alone. The
    /// machine takes every such token (it is the key's bytes), so a narrowed mask is never empty, and the
    /// tokenizer's own tokens for a document are never refused by it.
    fn keyTokens(made: *Compiled, top: u32, out: *[max_keys]u32) Allocator.Error!?[]const u32 {
        if (top == done or !made.key_tokens) return null;
        const frame = made.matcher.machine.frames.items[top];
        if (frame.phase != .key) return null;
        const schema = &made.grammar.schema;
        const root = schema.properties[schema.nodes[frame.node].first + frame.a].keys;
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(made.gpa);
        if (!try pathTo(schema, made.gpa, root, frame.b, &text)) return null;
        var walk: KeyWalk = .{ .made = made, .text = &text, .at = text.items.len, .out = out };
        return if (try walk.under(frame.b)) out[0..walk.count] else null;
    }

    /// Appends the bytes that lead from `from` to `target` in a trie; false when `target` is not under it.
    fn pathTo(schema: *const Schema, gpa: Allocator, from: u32, target: u32, text: *std.ArrayList(u8)) Allocator.Error!bool {
        if (from == target) return true;
        const node = schema.tries[from];
        for (schema.edge_bytes[node.edges..][0..node.fan], schema.edge_nodes[node.edges..][0..node.fan]) |byte, under| {
            try text.append(gpa, byte);
            if (try pathTo(schema, gpa, under, target, text)) return true;
            _ = text.pop();
        }
        return false;
    }

    /// Every key under a place in a trie, each asked for its token at `at` bytes into its text.
    const KeyWalk = struct {
        made: *Compiled,
        /// The key's text down to the node being walked.
        text: *std.ArrayList(u8),
        at: usize,
        out: *[max_keys]u32,
        count: usize = 0,

        /// False when a key under `node` has no token of its own at the place, or the keys are too many.
        fn under(self: *KeyWalk, node: u32) Allocator.Error!bool {
            const schema = &self.made.grammar.schema;
            const trie = schema.tries[node];
            if (trie.value != none) {
                const own_token = try self.tokenHere() orelse return false;
                if (std.mem.indexOfScalar(u32, self.out[0..self.count], own_token) == null) {
                    if (self.count == self.out.len) return false;
                    self.out[self.count] = own_token;
                    self.count += 1;
                }
            }
            for (schema.edge_bytes[trie.edges..][0..trie.fan], schema.edge_nodes[trie.edges..][0..trie.fan]) |byte, below| {
                try self.text.append(self.made.gpa, byte);
                defer _ = self.text.pop();
                if (!try self.under(below)) return false;
            }
            return true;
        }

        /// The token that starts `at` bytes into the key `text` and ends before its closing quote, the same
        /// in every context; null when there is none.
        fn tokenHere(self: *KeyWalk) Allocator.Error!?u32 {
            const gpa = self.made.gpa;
            const key = self.text.items;
            const tokenizer = self.made.grammar.vocabulary.tokenizer;
            var found: ?u32 = null;
            var whole: std.ArrayList(u8) = .empty;
            defer whole.deinit(gpa);
            var ids: std.ArrayList(u32) = .empty;
            defer ids.deinit(gpa);
            for (key_contexts) |context| {
                whole.clearRetainingCapacity();
                try whole.appendSlice(gpa, context[0]);
                try whole.appendSlice(gpa, key);
                try whole.appendSlice(gpa, context[1]);
                ids.clearRetainingCapacity();
                tokenizer.encode(gpa, whole.items, &ids) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => return null,
                };
                const from = context[0].len + self.at;
                var offset: usize = 0;
                const here: u32 = for (ids.items) |id| {
                    const bytes = tokenizer.tokenBytes(id);
                    // a tokenizer that rewrites its text gives no offsets to go by
                    if (!std.mem.startsWith(u8, whole.items[offset..], bytes)) return null;
                    if (offset == from) break if (offset + bytes.len < context[0].len + key.len) id else return null;
                    offset += bytes.len;
                    if (offset > from) return null;
                } else return null;
                if (found != null and found.? != here) return null;
                found = here;
            }
            return found;
        }
    };

    fn ended(_: *anyopaque, at: grammar_file.Position) bool {
        return unpack(at).ended;
    }

    fn destroy(context: *anyopaque) void {
        const made: *Compiled = @ptrCast(@alignCast(context));
        made.matcher.deinit();
        made.grammar.deinit();
        made.gpa.destroy(made);
    }
};
