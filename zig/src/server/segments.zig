//! TensorFold's `model_text.Text` with the ids of a prompt's segments kept between requests (spec 072 WS-03).
//!
//! TensorFold's host renders a request's whole conversation and encodes the whole text, three times a chat
//! request (`server/prompt.zig`: the prompt, the prompt without its generation suffix for `history_len`, the
//! probe for the shared system prefix; since 1.0.4 also four renders of one short message, `replySpans`,
//! whose segments are answered here after a server's first request). The template is a fraction of a millisecond; an encode of 17,000
//! tokens is 5 to 10 ms, so a turn that holds a long conversation waited 10 to 20 ms for text it had encoded
//! the turn before. `Segments` stands between the host and its `hf_text` and answers `encode` from what it
//! kept. Nothing else of the text changes, and the ids are the tokenizer's own to the id.
//!
//! Why a segment's ids are its own. TensorFold's `Tokenizer.encode` first parts the text at the added tokens
//! that are matched before normalization (`splitAdded` over its `raw` trie, leftmost and longest), then
//! normalizes, pre-tokenizes and merges each text between two of them alone. A segment here is one such added
//! token and the text up to the next one (the text before the first is a segment too), so encoding the
//! segments apart and joining their ids is what `encode` does, given what `init` and `encode` check:
//!
//! - no added token strips the space beside it (`lstrip`, `rstrip`), which would move a segment's edge into
//!   its neighbour: with one that does, nothing is kept and every `encode` is the host's;
//! - the cuts are found as the tokenizer finds them, by the same leftmost-longest walk over the same tokens
//!   (`Trie`), so a cut is never inside an added token or at one the tokenizer would not match there;
//! - a segment after the first begins with its added token, so its text never stands at the text's offset
//!   0, which the tokenizer treats apart (`first`); the first segment stands at 0 both ways;
//! - the text is UTF-8: one that is not is repaired as a whole by the tokenizer, and goes to it whole.
//!
//! A segment is found by its bytes, compared in full, never by a hash alone. So an edited earlier turn, a
//! changed system prompt or a shorter history need no rule: a segment whose bytes were seen is answered, any
//! other is encoded by the host's tokenizer and kept. `add_special` (a completion's text) is the host's alone.
//!
//! Since TensorFold 1.1.4 (#1117) a chat's prompt is encoded with its request bytes masked (`encode_masked`,
//! `Tokenizer.encodeMasked`): the tokenizer matches added tokens in the template's runs alone, between the
//! request's bytes, leftmost and longest, then encodes each text between two of them alone, refusing one that
//! holds request bytes and gives an added id. So the cuts are found here by the same walk over the same runs,
//! and a segment is its added token and the text up to the next as before, with the request's bytes inside it:
//! a segment encoded alone, masked by its own request bytes, has the ids the whole text gives it, and what it
//! refuses the whole text refuses. A segment with request bytes is kept by its bytes and where they lie, apart
//! from a template's segment of the same bytes; one without is the same entry `encode` keeps.
//!
//! What is kept: a segment's bytes, its ids and an entry, at most `limit` bytes for a server
//! (`default_bytes`), the least recently used leaving first. A conversation of 17,000 tokens is about 125 kB.
//! One mutex, held while the kept ids are copied out and while new ones are put in; the encodes run outside
//! it, so two requests that miss the same segment both encode it and one copy is kept.
//!
//! `zig/tests/server/segments_test.zig` holds the ids against the tokenizer's on made texts and on the
//! checkpoint's own template; `tools/held_turn_bench.zig` times a turn with and without.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const host = @import("server");
const tokenizer = @import("tokenizer");
const Text = host.model_text.Text;
const Error = host.model_text.Error;
const Rendered = host.model_text.Rendered;
const Span = host.model_text.Span;

/// What a server keeps unless told: 32 MiB, the segments of about four million tokens.
pub const default_bytes: usize = 32 << 20;

/// What `Segments` did so far.
pub const Counts = struct {
    /// Segments answered from what was kept, and segments the tokenizer encoded.
    hits: u64 = 0,
    misses: u64 = 0,
    /// Texts handed to the host whole: `add_special`, not UTF-8, or nothing may be kept.
    passed: u64 = 0,
    /// Kept now: segments and their bytes (text, ids and the entry).
    kept: usize = 0,
    bytes: usize = 0,
    dropped: u64 = 0,
};

/// The added tokens matched before normalization, for the walk the tokenizer's own trie makes: at a byte, the
/// longest token that starts there.
const Trie = struct {
    const Node = struct { ends: bool = false, kids: std.ArrayList(Kid) = .empty };
    const Kid = struct { byte: u8, node: u32 };

    first: [256]u32 = @splat(0),
    nodes: std.ArrayList(Node) = .empty,

    fn deinit(self: *Trie, gpa: Allocator) void {
        for (self.nodes.items) |*node| node.kids.deinit(gpa);
        self.nodes.deinit(gpa);
    }

    fn insert(self: *Trie, gpa: Allocator, token: []const u8) Allocator.Error!void {
        if (token.len == 0) return;
        if (self.nodes.items.len == 0) try self.nodes.append(gpa, .{});
        var at: u32 = 0;
        for (token, 0..) |byte, i| {
            const next: u32 = if (i == 0) self.first[byte] else self.kid(at, byte);
            if (next != 0) {
                at = next;
                continue;
            }
            try self.nodes.append(gpa, .{});
            const made: u32 = @intCast(self.nodes.items.len - 1);
            if (i == 0) self.first[byte] = made else try self.nodes.items[at].kids.append(gpa, .{ .byte = byte, .node = made });
            at = made;
        }
        self.nodes.items[at].ends = true;
    }

    fn kid(self: *const Trie, at: u32, byte: u8) u32 {
        for (self.nodes.items[at].kids.items) |each| {
            if (each.byte == byte) return each.node;
        }
        return 0;
    }

    /// The end of the longest token that starts at `from`, or null.
    fn match(self: *const Trie, text: []const u8, from: usize) ?usize {
        var at = self.first[text[from]];
        if (at == 0) return null;
        var best: ?usize = null;
        var end = from + 1;
        while (true) {
            if (self.nodes.items[at].ends) best = end;
            if (end == text.len) break;
            at = self.kid(at, text[end]);
            if (at == 0) break;
            end += 1;
        }
        return best;
    }
};

pub const Segments = struct {
    gpa: Allocator,
    io: Io,
    /// The host's text, which every entry but `encode` goes to as it came.
    inner: Text,
    trie: Trie = .{},
    /// False where an added token strips its neighbours or the limit is nothing: every text is the host's.
    keeps: bool,
    limit: usize,
    mutex: Io.Mutex = .init,
    /// By a segment's bytes, which are the entry's own.
    kept: std.StringHashMapUnmanaged(*Entry) = .empty,
    /// Least recently used first.
    order: std.DoublyLinkedList = .{},
    counts: Counts = .{},

    const Entry = struct {
        node: std.DoublyLinkedList.Node = .{},
        text: []u8,
        ids: []u32,

        /// What an entry is charged: its bytes, its ids, itself and its place in the table.
        fn cost(text_len: usize, ids_len: usize) usize {
            return text_len + ids_len * @sizeOf(u32) + @sizeOf(Entry) + 4 * @sizeOf(usize);
        }
    };

    /// Over `inner`, whose encodes are `loaded`'s (the host's `hf_text` and its tokenizer). `limit` bytes are
    /// kept at most; 0 keeps nothing.
    pub fn init(gpa: Allocator, io: Io, inner: Text, loaded: *const tokenizer.Tokenizer, limit: usize) Allocator.Error!Segments {
        var self: Segments = .{ .gpa = gpa, .io = io, .inner = inner, .keeps = limit > 0, .limit = limit };
        errdefer self.trie.deinit(gpa);
        for (loaded.added) |token| {
            if (token.lstrip or token.rstrip) self.keeps = false;
            // a normalized one is matched inside a segment's text, after normalization: it parts nothing here
            if (!token.normalized) try self.trie.insert(gpa, token.content);
        }
        return self;
    }

    pub fn deinit(self: *Segments) void {
        while (self.order.popFirst()) |node| self.free(@fieldParentPtr("node", node));
        self.kept.deinit(self.gpa);
        self.trie.deinit(self.gpa);
        self.* = undefined;
    }

    /// The host's text with `encode` answered here.
    pub fn text(self: *Segments) Text {
        return .{ .ctx = self, .vtable = &.{
            .encode = encodeFn,
            .decode = decodeFn,
            .token_id = tokenIdFn,
            .token_string = tokenStringFn,
            .vocab_size = vocabSizeFn,
            .eos_ids = eosIdsFn,
            .render = renderFn,
            .template_source = templateSourceFn,
            .special = specialFn,
            .added = addedFn,
            .token_bytes = tokenBytesFn,
            .encode_plain = encodePlainFn,
            .encode_masked = encodeMaskedFn,
        } };
    }

    pub fn account(self: *Segments) Counts {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.counts;
    }

    /// Where `text`'s segments start: 0, then every added token the tokenizer matches before normalization in
    /// the text's runs between the request's bytes (`spans`, sorted and apart; none: the text is one run).
    fn starts(self: *const Segments, a: Allocator, s: []const u8, spans: []const Span) Allocator.Error![]const usize {
        var out: std.ArrayList(usize) = .empty;
        try out.append(a, 0);
        var from: usize = 0;
        for (0..spans.len + 1) |i| {
            const run = s[0..if (i < spans.len) spans[i][0] else s.len];
            var at = from;
            while (at < run.len) {
                if (self.trie.first[run[at]] != 0) if (self.trie.match(run, at)) |end| {
                    if (at > 0) try out.append(a, at);
                    at = end;
                    continue;
                };
                at += 1;
            }
            if (i < spans.len) from = spans[i][1];
        }
        return out.items;
    }

    /// `text`'s ids as `inner.encode` gives them, in `a`.
    pub fn encode(self: *Segments, a: Allocator, s: []const u8, add_special: bool) Error![]u32 {
        const io = self.io;
        if (add_special or !self.keeps or s.len == 0 or !std.unicode.utf8ValidateSlice(s)) {
            self.mutex.lockUncancelable(io);
            self.counts.passed += 1;
            self.mutex.unlock(io);
            return self.inner.encode(a, s, add_special);
        }
        var problem: []const u8 = "";
        return std.mem.concat(a, u32, try self.segmentIds(a, s, try self.starts(a, s, &.{}), &.{}, &problem));
    }

    /// Each segment's ids, from what was kept or the host's tokenizer (masked by its own request bytes), in `a`.
    fn segmentIds(self: *Segments, a: Allocator, s: []const u8, cuts: []const usize, spans: []const Span, problem: *[]const u8) Error![]const []const u32 {
        const io = self.io;
        const parts = try a.alloc([]const u32, cuts.len);
        const missed = try a.alloc(bool, cuts.len);
        // a segment's request bytes, from its start, and the key it is kept by
        const local = try a.alloc([]const Span, cuts.len);
        const keys = try a.alloc([]const u8, cuts.len);
        var next: usize = 0;
        for (cuts, local, keys, 0..) |from, *own, *key, i| {
            const to = if (i + 1 < cuts.len) cuts[i + 1] else s.len;
            var inside: std.ArrayList(Span) = .empty;
            while (next < spans.len and spans[next][1] <= from) next += 1;
            var j = next;
            while (j < spans.len and spans[j][0] < to) : (j += 1) try inside.append(a, .{ @max(from, spans[j][0]) - from, @min(to, spans[j][1]) - from });
            own.* = inside.items;
            key.* = try maskedKey(a, s[from..to], own.*);
        }
        var misses: usize = 0;
        {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            for (keys, parts, missed) |segment, *part, *miss| {
                if (self.kept.get(segment)) |entry| {
                    part.* = try a.dupe(u32, entry.ids);
                    self.order.remove(&entry.node);
                    self.order.append(&entry.node);
                    miss.* = false;
                } else {
                    miss.* = true;
                    misses += 1;
                }
            }
            self.counts.hits += cuts.len - misses;
            self.counts.misses += misses;
        }
        if (misses > 0) {
            for (cuts, local, parts, missed, 0..) |from, own, *part, miss, i| {
                const segment = s[from..if (i + 1 < cuts.len) cuts[i + 1] else s.len];
                if (miss) part.* = if (own.len == 0) try self.inner.encode(a, segment, false) else try self.inner.encodeMasked(a, segment, own, problem);
            }
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            for (keys, parts, missed) |key, part, miss| {
                // a segment that is not kept for want of memory is encoded again the next time
                if (miss) self.keep(key, part) catch {};
            }
        }
        return parts;
    }

    /// `text`'s ids as `inner.encodeMasked` gives them, in `a`: `spans` are the request's bytes.
    pub fn encodeMasked(self: *Segments, a: Allocator, s: []const u8, spans: []const Span, problem: *[]const u8) Error![]u32 {
        if (spans.len == 0) return self.encode(a, s, false);
        if (!self.keeps or !ordered(spans, s.len) or !std.unicode.utf8ValidateSlice(s)) {
            self.mutex.lockUncancelable(self.io);
            self.counts.passed += 1;
            self.mutex.unlock(self.io);
            return self.inner.encodeMasked(a, s, spans, problem);
        }
        return std.mem.concat(a, u32, try self.segmentIds(a, s, try self.starts(a, s, spans), spans, problem));
    }

    /// A segment's key: its bytes where it holds no request byte (`encode`'s entry), else a byte no UTF-8 text
    /// begins with, its request ranges and its bytes.
    fn maskedKey(a: Allocator, segment: []const u8, own: []const Span) Allocator.Error![]const u8 {
        if (own.len == 0) return segment;
        const key = try a.alloc(u8, 1 + 4 + own.len * 8 + segment.len);
        key[0] = 0xff;
        std.mem.writeInt(u32, key[1..5], @intCast(own.len), .little);
        for (own, 0..) |p, i| {
            std.mem.writeInt(u32, key[5 + i * 8 ..][0..4], @intCast(p[0]), .little);
            std.mem.writeInt(u32, key[9 + i * 8 ..][0..4], @intCast(p[1]), .little);
        }
        @memcpy(key[5 + own.len * 8 ..], segment);
        return key;
    }

    /// The spans sorted, apart and inside the text, as the tokenizer asks (else it refuses them in its words).
    fn ordered(spans: []const Span, len: usize) bool {
        var previous: usize = 0;
        for (spans) |p| {
            if (p[0] < previous or p[0] >= p[1] or p[1] > len or p[1] > std.math.maxInt(u32)) return false;
            previous = p[1];
        }
        return true;
    }

    /// Under the mutex: `segment` with `ids` kept, and the least recently used dropped down to the limit.
    fn keep(self: *Segments, segment: []const u8, ids: []const u32) Allocator.Error!void {
        const cost = Entry.cost(segment.len, ids.len);
        if (cost > self.limit or self.kept.contains(segment)) return;
        const entry = try self.gpa.create(Entry);
        errdefer self.gpa.destroy(entry);
        const own_text = try self.gpa.dupe(u8, segment);
        errdefer self.gpa.free(own_text);
        const own_ids = try self.gpa.dupe(u32, ids);
        errdefer self.gpa.free(own_ids);
        entry.* = .{ .text = own_text, .ids = own_ids };
        try self.kept.put(self.gpa, own_text, entry);
        self.order.append(&entry.node);
        self.counts.kept += 1;
        self.counts.bytes += cost;
        while (self.counts.bytes > self.limit) {
            const oldest: *Entry = @fieldParentPtr("node", self.order.popFirst().?);
            _ = self.kept.remove(oldest.text);
            self.counts.kept -= 1;
            self.counts.bytes -= Entry.cost(oldest.text.len, oldest.ids.len);
            self.counts.dropped += 1;
            self.free(oldest);
        }
    }

    fn free(self: *Segments, entry: *Entry) void {
        self.gpa.free(entry.text);
        self.gpa.free(entry.ids);
        self.gpa.destroy(entry);
    }

    fn of(ctx: *anyopaque) *Segments {
        return @ptrCast(@alignCast(ctx));
    }
    fn encodeFn(ctx: *anyopaque, a: Allocator, s: []const u8, add_special: bool) Error![]u32 {
        return of(ctx).encode(a, s, add_special);
    }
    fn decodeFn(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        return of(ctx).inner.decode(a, ids);
    }
    fn tokenIdFn(ctx: *anyopaque, piece: []const u8) ?u32 {
        return of(ctx).inner.tokenId(piece);
    }
    fn tokenStringFn(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return of(ctx).inner.tokenString(a, id);
    }
    fn vocabSizeFn(ctx: *anyopaque) u32 {
        return of(ctx).inner.vocabSize();
    }
    fn eosIdsFn(ctx: *anyopaque) []const u32 {
        return of(ctx).inner.eosIds();
    }
    fn renderFn(ctx: *anyopaque, a: Allocator, messages: host.json.Value, options: host.model_text.RenderOptions, problem: *[]const u8) Error!Rendered {
        return of(ctx).inner.render(a, messages, options, problem);
    }
    fn templateSourceFn(ctx: *anyopaque) []const u8 {
        return of(ctx).inner.templateSource();
    }
    fn specialFn(ctx: *anyopaque, id: u32) bool {
        return of(ctx).inner.isSpecial(id);
    }
    /// The host's since TensorFold 1.1.4: its `--vision` check that every image marker is an added token.
    fn addedFn(ctx: *anyopaque, id: u32) bool {
        return of(ctx).inner.isAdded(id);
    }
    fn encodeMaskedFn(ctx: *anyopaque, a: Allocator, s: []const u8, spans: []const Span, problem: *[]const u8) Error![]u32 {
        return of(ctx).encodeMasked(a, s, spans, problem);
    }
    /// The host's since TensorFold 1.0.5, for a reply's logprobs: null where the inner text cannot say.
    fn tokenBytesFn(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error!?[]u8 {
        return of(ctx).inner.tokenBytes(a, id);
    }
    /// The host's since TensorFold 1.0.12, for an image's markers (which zspark does not serve): the inner text's.
    fn encodePlainFn(ctx: *anyopaque, a: Allocator, s: []const u8) Error![]u32 {
        return of(ctx).inner.encodePlain(a, s);
    }
};
