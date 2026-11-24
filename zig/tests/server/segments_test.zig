//! `segments.zig`'s tests: the ids `Segments` gives are the tokenizer's own, on texts made at random over a
//! tiny tokenizer and over the checkpoints', and on a conversation rendered by the checkpoint's template turn
//! after turn, as TensorFold's host renders one (`Text.renderIds`).

const std = @import("std");
const host = @import("server");
const tokenizer = @import("tokenizer");
const segments_file = @import("../../src/server/segments.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const Segments = segments_file.Segments;
const Text = host.model_text.Text;

const tiny_tokenizer = @embedFile("fixtures/text/tiny_tokenizer.json");

/// A tokenizer alone behind `Text`: `add_special` puts `lead` first, and nothing renders.
const Plain = struct {
    loaded: *const tokenizer.Tokenizer,
    encodes: usize = 0,

    const lead: u32 = 4_000_000;

    fn of(ctx: *anyopaque) *Plain {
        return @ptrCast(@alignCast(ctx));
    }
    fn encode(ctx: *anyopaque, a: Allocator, s: []const u8, add_special: bool) host.model_text.Error![]u32 {
        const self = of(ctx);
        self.encodes += 1;
        const ids = self.loaded.encode(a, s) catch |err| return if (err == error.OutOfMemory) error.OutOfMemory else error.Template;
        return if (add_special) std.mem.concat(a, u32, &.{ &.{lead}, ids }) else ids;
    }
    fn decode(_: *anyopaque, a: Allocator, _: []const u32) Allocator.Error![]u8 {
        return a.dupe(u8, "");
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, _: u32) Allocator.Error![]u8 {
        return a.dupe(u8, "");
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 0;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn encodeMasked(ctx: *anyopaque, a: Allocator, s: []const u8, spans: []const host.model_text.Span, problem: *[]const u8) host.model_text.Error![]u32 {
        const self = of(ctx);
        self.encodes += 1;
        return self.loaded.encodeMasked(a, s, spans) catch |err| {
            if (err == error.OutOfMemory) return error.OutOfMemory;
            problem.* = @errorName(err);
            return error.Template;
        };
    }
    fn added(ctx: *anyopaque, id: u32) bool {
        return of(ctx).loaded.isAdded(id);
    }
    fn render(_: *anyopaque, _: Allocator, _: host.json.Value, _: host.model_text.RenderOptions, _: *[]const u8) host.model_text.Error!host.model_text.Rendered {
        return error.Template;
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
    fn text(self: *Plain) Text {
        return .{ .ctx = self, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .added = added, .encode_masked = encodeMasked } };
    }
};

/// The tiny tokenizer; `strips` false with every added token's `lstrip` and `rstrip` taken away. Its added
/// tokens overlap on purpose: `<t>` and `<t>x`, `<w>` and `<w>!`, `>>` and ` >>`.
fn tiny(arena: Allocator, strips: bool) !tokenizer.Tokenizer {
    if (strips) return tokenizer.parse(arena, tiny_tokenizer);
    const left = try std.mem.replaceOwned(u8, arena, tiny_tokenizer, "\"lstrip\":true", "\"lstrip\":false");
    return tokenizer.parse(arena, try std.mem.replaceOwned(u8, arena, left, "\"rstrip\":true", "\"rstrip\":false"));
}

/// Texts of the pieces a boundary could be wrong at: every added token, each cut short, space and newlines
/// beside them, characters NFC composes (alone at a segment's start too), and now and then bytes that are not
/// UTF-8.
const Maker = struct {
    pieces: std.ArrayList([]const u8) = .empty,

    const always = [_][]const u8{ " ", "  ", "\n", "\n\n", "a", "The cat", " sat", "x", "!", "<", ">", "<|", "|>", "é", "e\u{0301}", "\u{0301}", "'s", "12", "你好", "🙂", "A\u{030A}ngstro\u{0308}m", "user\n", "\t" };
    const broken = [_][]const u8{ "\xff", "\xc3", "\xe4\xbd" };

    fn init(arena: Allocator, loaded: *const tokenizer.Tokenizer) !Maker {
        var self: Maker = .{};
        try self.pieces.appendSlice(arena, &always);
        for (loaded.added) |token| {
            // each added token three times as likely as another piece, then its head and its tail
            for (0..3) |_| try self.pieces.append(arena, token.content);
            if (token.content.len > 1) {
                try self.pieces.append(arena, token.content[0 .. token.content.len - 1]);
                try self.pieces.append(arena, token.content[1..]);
            }
        }
        return self;
    }

    fn make(self: *const Maker, a: Allocator, random: std.Random, most: usize) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (0..random.uintLessThan(usize, most + 1)) |_| {
            if (random.uintLessThan(usize, 400) == 0) {
                try out.appendSlice(a, broken[random.uintLessThan(usize, broken.len)]);
            } else try out.appendSlice(a, self.pieces.items[random.uintLessThan(usize, self.pieces.items.len)]);
        }
        return out.items;
    }
};

/// `rounds` texts made at random through `segments` and through `plain`: how many differed. Texts come again
/// (one in four is an earlier text, or one with a tail added), as a conversation's do.
fn differing(gpa: Allocator, segments: *Segments, plain: Text, maker: *const Maker, seed: u64, rounds: usize, most: usize) !usize {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    var earlier: std.ArrayList([]const u8) = .empty;
    var differed: usize = 0;
    for (0..rounds) |_| {
        var made = try maker.make(arena, random, most);
        if (earlier.items.len > 0 and random.uintLessThan(usize, 4) == 0) {
            const before = earlier.items[random.uintLessThan(usize, earlier.items.len)];
            made = try std.mem.concat(arena, u8, &.{ before, if (random.boolean()) made else "" });
        }
        if (earlier.items.len < 64) try earlier.append(arena, made) else earlier.items[random.uintLessThan(usize, 64)] = made;
        const want = try plain.encode(arena, made, false);
        const got = try segments.encode(arena, made, false);
        if (!std.mem.eql(u32, want, got)) differed += 1;
    }
    return differed;
}

/// `differing` with the request's bytes masked (TensorFold 1.1.4's `encode_masked`): each text's ranges drawn
/// at random, apart, now and then none; a text the tokenizer refuses (request bytes that give an added id) is
/// refused through `segments` too. How many differed, and how many were refused.
fn differingMasked(gpa: Allocator, segments: *Segments, plain: Text, maker: *const Maker, seed: u64, rounds: usize, most: usize) !struct { usize, usize } {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    var earlier: std.ArrayList(struct { []const u8, []const host.model_text.Span }) = .empty;
    var differed: usize = 0;
    var refused: usize = 0;
    for (0..rounds) |_| {
        var made = try maker.make(arena, random, most);
        var spans: std.ArrayList(host.model_text.Span) = .empty;
        if (earlier.items.len > 0 and random.uintLessThan(usize, 4) == 0) {
            // an earlier text, its ranges kept, and a tail of request bytes now and then: a conversation's next turn
            const before = earlier.items[random.uintLessThan(usize, earlier.items.len)];
            try spans.appendSlice(arena, before[1]);
            if (random.boolean() and made.len > 0) try spans.append(arena, .{ before[0].len, before[0].len + made.len });
            made = try std.mem.concat(arena, u8, &.{ before[0], made });
        } else {
            var at: usize = 0;
            while (made.len > 0 and at < made.len and random.uintLessThan(usize, 3) != 0) {
                const start = at + random.uintLessThan(usize, made.len - at);
                const end = start + 1 + random.uintLessThan(usize, made.len - start);
                try spans.append(arena, .{ start, end });
                at = end;
            }
        }
        if (earlier.items.len < 64) try earlier.append(arena, .{ made, spans.items }) else earlier.items[random.uintLessThan(usize, 64)] = .{ made, spans.items };
        var problem: []const u8 = "";
        const want = plain.encodeMasked(arena, made, spans.items, &problem) catch |err| {
            refused += 1;
            if (segments.text().encodeMasked(arena, made, spans.items, &problem)) |_| {
                differed += 1;
            } else |got| if (got != err) {
                differed += 1;
            }
            continue;
        };
        const got = segments.text().encodeMasked(arena, made, spans.items, &problem) catch {
            differed += 1;
            continue;
        };
        if (!std.mem.eql(u32, want, got)) differed += 1;
    }
    return .{ differed, refused };
}

test "the ids of texts made at random are the tokenizer's, kept or not, with little room and with much" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var loaded = try tiny(arena, false);
    var plain: Plain = .{ .loaded = &loaded };
    const maker: Maker = try .init(arena, &loaded);
    for ([_]usize{ segments_file.default_bytes, 2000 }) |limit| {
        var segments: Segments = try .init(gpa, testing.io, plain.text(), &loaded, limit);
        defer segments.deinit();
        try testing.expect(segments.keeps);
        try testing.expectEqual(@as(usize, 0), try differing(gpa, &segments, plain.text(), &maker, 0x5e9_0072, 4000, 24));
        const counts = segments.account();
        // the fast path was the one walked: most segments were answered, and the small table turned over
        try testing.expect(counts.hits > 0 and (counts.hits > counts.misses or limit == 2000));
        try testing.expect(counts.bytes <= limit);
        try testing.expect((counts.dropped > 0) == (limit == 2000));
        // and masked, as a chat's prompt is since TensorFold 1.1.4 (an added token's spelling cut by request
        // bytes, request bytes that spell one): the same ids, and what the tokenizer refuses is refused
        const masked = try differingMasked(gpa, &segments, plain.text(), &maker, 0x5e9_0141, 4000, 24);
        try testing.expectEqual(@as(usize, 0), masked[0]);
        try testing.expect(masked[1] < 4000);
    }
}

test "request bytes that the tokenizer encodes to an added id are refused through the segments as by the tokenizer" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // `aaaa`, the vocabulary's 263 by its merges, made an added token too: in a request's bytes it is refused
    const left = try std.mem.replaceOwned(u8, arena, tiny_tokenizer, "\"lstrip\":true", "\"lstrip\":false");
    const unstripped = try std.mem.replaceOwned(u8, arena, left, "\"rstrip\":true", "\"rstrip\":false");
    const json = try std.mem.replaceOwned(u8, arena, unstripped, "\"added_tokens\":[", "\"added_tokens\":[{\"id\": 263, \"content\": \"aaaa\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": false},");
    var loaded = try tokenizer.parse(arena, json);
    var plain: Plain = .{ .loaded = &loaded };
    const maker: Maker = try .init(arena, &loaded);
    var segments: Segments = try .init(gpa, testing.io, plain.text(), &loaded, segments_file.default_bytes);
    defer segments.deinit();
    try testing.expect(segments.keeps);
    const masked = try differingMasked(gpa, &segments, plain.text(), &maker, 0x141_aaaa, 4000, 24);
    try testing.expectEqual(@as(usize, 0), masked[0]);
    try testing.expect(masked[1] > 0 and masked[1] < 4000);
}

test "a tokenizer whose added token strips its neighbours keeps nothing; made to keep, its ids part" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var loaded = try tiny(arena, true);
    var plain: Plain = .{ .loaded = &loaded };
    const maker: Maker = try .init(arena, &loaded);
    var segments: Segments = try .init(gpa, testing.io, plain.text(), &loaded, segments_file.default_bytes);
    defer segments.deinit();
    try testing.expect(!segments.keeps);
    try testing.expectEqual(@as(usize, 0), try differing(gpa, &segments, plain.text(), &maker, 7, 1000, 24));
    try testing.expectEqual(@as(usize, 0), segments.account().kept);
    try testing.expectEqual(@as(u64, 1000), segments.account().passed);
    // the mutation: the same texts with the check off are not the tokenizer's, so the test above can fail
    segments.keeps = true;
    try testing.expect(try differing(gpa, &segments, plain.text(), &maker, 7, 1000, 24) > 0);
}

test "a text seen again costs no encode, an edit costs its segment alone, and the oldest leaves first" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var loaded = try tiny(arena, false);
    var plain: Plain = .{ .loaded = &loaded };
    var segments: Segments = try .init(gpa, testing.io, plain.text(), &loaded, segments_file.default_bytes);
    defer segments.deinit();
    const text = segments.text();

    const first = "<|end|>The cat<|end|> sat<|end|>12";
    const want = try loaded.encode(arena, first);
    try testing.expectEqualSlices(u32, want, try text.encode(arena, first, false));
    try testing.expectEqual(@as(usize, 3), plain.encodes);
    try testing.expectEqualSlices(u32, want, try text.encode(arena, first, false));
    try testing.expectEqual(@as(usize, 3), plain.encodes);
    // a turn more: the three segments held, one encoded
    const second = first ++ "<|end|>a";
    try testing.expectEqualSlices(u32, try loaded.encode(arena, second), try text.encode(arena, second, false));
    try testing.expectEqual(@as(usize, 4), plain.encodes);
    // the middle one edited, and the history shorter
    const edited = "<|end|>The cat<|end|> sat!<|end|>12";
    try testing.expectEqualSlices(u32, try loaded.encode(arena, edited), try text.encode(arena, edited, false));
    try testing.expectEqual(@as(usize, 5), plain.encodes);
    try testing.expectEqualSlices(u32, try loaded.encode(arena, "<|end|>The cat<|end|>12"), try text.encode(arena, "<|end|>The cat<|end|>12", false));
    try testing.expectEqual(@as(usize, 5), plain.encodes);
    // text before the first added token is a segment of its own, and stands at the text's start both ways
    const led = " sat<|end|> sat";
    try testing.expectEqualSlices(u32, try loaded.encode(arena, led), try text.encode(arena, led, false));
    try testing.expectEqual(@as(usize, 6), plain.encodes);
    // the longest added token is the cut, as in the tokenizer: `<t>x`, not `<t>` and an `x`
    const longest = "a<t>xa<t>a";
    try testing.expectEqualSlices(u32, try loaded.encode(arena, longest), try text.encode(arena, longest, false));

    // a completion's text and bytes that are not UTF-8 are the host's whole
    const passed = segments.account().passed;
    const special = try text.encode(arena, first, true);
    try testing.expectEqual(Plain.lead, special[0]);
    try testing.expectEqualSlices(u32, want, special[1..]);
    try testing.expectEqualSlices(u32, try loaded.encode(arena, "<|end|>a\xff<|end|>b"), try text.encode(arena, "<|end|>a\xff<|end|>b", false));
    try testing.expectEqual(passed + 2, segments.account().passed);

    // room for two of these segments and not for three: the one used longest ago is the one that went
    var small: Segments = try .init(gpa, testing.io, plain.text(), &loaded, segments_file.default_bytes);
    defer small.deinit();
    const a = "<|end|>" ++ @as([40]u8, @splat('a')) ++ "1";
    const b = "<|end|>" ++ @as([40]u8, @splat('a')) ++ "2";
    const c = "<|end|>" ++ @as([40]u8, @splat('a')) ++ "3";
    _ = try small.encode(arena, a, false);
    small.limit = small.account().bytes * 5 / 2;
    _ = try small.encode(arena, b, false);
    _ = try small.encode(arena, a, false);
    _ = try small.encode(arena, c, false);
    try testing.expect(small.account().bytes <= small.limit);
    try testing.expectEqual(@as(u64, 1), small.account().dropped);
    const before = plain.encodes;
    _ = try small.encode(arena, a, false);
    try testing.expectEqual(before, plain.encodes);
    _ = try small.encode(arena, b, false);
    try testing.expectEqual(before + 1, plain.encodes);
}

test "several requests at once take the same ids" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var loaded = try tiny(arena, false);
    var plain: Plain = .{ .loaded = &loaded };
    const maker: Maker = try .init(arena, &loaded);
    // little room, so that entries leave while another thread reads
    var segments: Segments = try .init(gpa, testing.io, plain.text(), &loaded, 4000);
    defer segments.deinit();
    const Worker = struct {
        fn run(shared: *Segments, loaded_: *const tokenizer.Tokenizer, maker_: *const Maker, seed: u64, differed: *usize) void {
            // a tokenizer's text of its own: `Plain` counts its encodes without a lock
            var own: Plain = .{ .loaded = loaded_ };
            differed.* = differing(std.heap.page_allocator, shared, own.text(), maker_, seed, 1500, 24) catch 1 << 20;
        }
    };
    var differed: [4]usize = @splat(0);
    var threads: [4]std.Thread = undefined;
    // the same seed twice: two threads miss the same segments at the same time
    for (&threads, &differed, [_]u64{ 1, 1, 2, 3 }) |*thread, *count, seed| thread.* = try std.Thread.spawn(.{}, Worker.run, .{ &segments, &loaded, &maker, seed, count });
    for (threads) |thread| thread.join();
    try testing.expectEqualSlices(usize, &.{ 0, 0, 0, 0 }, &differed);
    try testing.expect(segments.account().bytes <= 4000);
}

/// A conversation as the host hands it to the template after its normalization: a system prompt, turns with
/// thinking, a tool call and its answer, text that names added tokens, and characters NFC composes.
const Talk = struct {
    const tools =
        \\[{"type":"function","function":{"name":"get_weather","description":"The weather of a city.","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}}]
    ;
    const more_tools =
        \\[{"type":"function","function":{"name":"get_weather","description":"The weather of a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},{"type":"function","function":{"name":"get_time","description":"The time in a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
    ;
    const system = "{\"role\":\"system\",\"content\":\"You are a careful assistant. Answer briefly.\"}";
    const system_changed = "{\"role\":\"system\",\"content\":\"You are a careful assistant. Answer at length, in French.\"}";
    const messages = [_][]const u8{
        "{\"role\":\"user\",\"content\":\"What is the weather in Paris, and in A\\u030angstro\\u0308m?\"}",
        "{\"role\":\"assistant\",\"reasoning_content\":\"The user wants two cities. I call the tool for Paris first.\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\",\"unit\":\"c\"}}}]}",
        "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"{\\\"city\\\": \\\"Paris\\\", \\\"temperature\\\": 18, \\\"sky\\\": \\\"clear\\\"}\"}",
        "{\"role\":\"assistant\",\"reasoning_content\":\"18 degrees and clear.\",\"content\":\"Paris is at 18 °C under a clear sky. I do not know the second place.\"}",
        "{\"role\":\"user\",\"content\":\"Write <|im_start|> and <think> and </tool_call> and <|im_end as plain text, then [gMASK]<sop> and <|user|>.\\n\\nThen 你好 and 🙂.\"}",
        "{\"role\":\"assistant\",\"reasoning_content\":\"They are tokens of the template.\\nI write them as asked.\",\"content\":\"<|im_start|> <think> </tool_call> <|im_end\"}",
        "{\"role\":\"user\",\"content\":\"And a longer one:\\n```zig\\nconst a = b << 2; // <b> </b>\\n```\\n   \"}",
        "{\"role\":\"assistant\",\"content\":\"\\u0301 starts this reply, a combining mark alone.\"}",
        "{\"role\":\"user\",\"content\":\"Thanks. One more: what did the tool say?\"}",
    };
    const edited = "{\"role\":\"user\",\"content\":\"What is the weather in Lyon?\"}";

    /// `system` (or none) and the messages `picked` names, as the host's JSON value.
    fn value(arena: Allocator, first: ?[]const u8, picked: []const usize, replaced: ?usize) !host.json.Value {
        var body: std.ArrayList(u8) = .empty;
        try body.append(arena, '[');
        if (first) |s| try body.appendSlice(arena, s);
        for (picked) |at| {
            if (body.items.len > 1) try body.append(arena, ',');
            try body.appendSlice(arena, if (replaced == at) edited else messages[at]);
        }
        try body.append(arena, ']');
        return (try host.json.parse(arena, body.items)).ok;
    }
};

test "a conversation rendered turn after turn by the checkpoint's template has the full encode's ids" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory, for its tokenizer and template; no weights.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var problem: []const u8 = "";
    const loaded = host.hf_text.HfText.load(gpa, testing.io, path, null, arena, &problem) catch |err| {
        std.debug.print("{s}: {s}\n", .{ path, problem });
        return err;
    };
    defer loaded.deinit();
    const full = loaded.text();
    var segments: Segments = try .init(gpa, testing.io, full, &loaded.tok, segments_file.default_bytes);
    defer segments.deinit();
    try testing.expect(segments.keeps);
    const fast = segments.text();
    const tools = (try host.json.parse(arena, Talk.tools)).ok;
    const more_tools = (try host.json.parse(arena, Talk.more_tools)).ok;

    const Step = struct { name: []const u8, system: ?[]const u8 = Talk.system, picked: []const usize, replaced: ?usize = null, tools: u2 = 1 };
    const steps = [_]Step{
        .{ .name = "first met", .picked = &.{0} },
        .{ .name = "the tool's answer", .picked = &.{ 0, 1, 2 } },
        .{ .name = "a second question", .picked = &.{ 0, 1, 2, 3, 4 } },
        .{ .name = "a third", .picked = &.{ 0, 1, 2, 3, 4, 5, 6 } },
        .{ .name = "a fourth", .picked = &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 } },
        .{ .name = "the same again", .picked = &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 } },
        .{ .name = "the system prompt changed", .system = Talk.system_changed, .picked = &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 } },
        .{ .name = "no system prompt", .system = null, .picked = &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 } },
        .{ .name = "an earlier turn edited", .picked = &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }, .replaced = 0 },
        .{ .name = "the history shorter", .picked = &.{ 0, 3, 4, 5, 8 } },
        .{ .name = "shorter still, no tools", .picked = &.{ 4, 5, 8 }, .tools = 0 },
        .{ .name = "another tool offered", .picked = &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }, .tools = 2 },
        .{ .name = "back to the first", .picked = &.{ 0, 1, 2, 3, 4 } },
    };
    var renders: usize = 0;
    for (steps) |step| {
        const messages = try Talk.value(arena, step.system, step.picked, step.replaced);
        const before = segments.account();
        // what `prompt.zig` asks for one request: the prompt and its history, thinking as the request says
        for ([_]bool{ true, false }) |thinking| for ([_]bool{ true, false }) |generation| for ([_]?[]const u8{ null, "low" }) |effort| {
            const options: host.model_text.RenderOptions = .{
                .tools = switch (step.tools) {
                    0 => null,
                    1 => tools,
                    else => more_tools,
                },
                .add_generation_prompt = generation,
                .enable_thinking = thinking,
                .reasoning_effort = if (thinking) effort else null,
            };
            const want = full.renderIds(arena, messages, options, &problem) catch |err| {
                std.debug.print("{s}: {s}\n", .{ step.name, problem });
                return err;
            };
            const got = try fast.renderIds(arena, messages, options, &problem);
            testing.expectEqualSlices(u32, want, got) catch |err| {
                std.debug.print("{s}: thinking {}, generation {}\n", .{ step.name, thinking, generation });
                return err;
            };
            try testing.expect(want.len > 20);
            renders += 1;
        };
        const after = segments.account();
        // every step's eight texts share their messages: most segments are answered, from the first step on
        try testing.expect(after.hits - before.hits > after.misses - before.misses);
        if (std.mem.eql(u8, step.name, "the same again") or std.mem.eql(u8, step.name, "back to the first")) try testing.expectEqual(before.misses, after.misses);
    }
    try testing.expectEqual(steps.len * 8, renders);
    try testing.expectEqual(@as(u64, 0), segments.account().passed);
}

test "texts made at random over the checkpoints' tokenizers have the full encode's ids" {
    const gpa = testing.allocator;
    var ran = false;
    for ([_][]const u8{ "ZSPARK_FLASH_NEXT_FP4_LOCAL", "ZSPARK_GLM_FP4_LOCAL" }) |name| {
        const path = testing.environ.getPosix(name) orelse continue;
        ran = true;
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var loaded = try tokenizer.loadTokenizer(testing.io, gpa, path);
        defer loaded.deinit();
        var plain: Plain = .{ .loaded = &loaded };
        const maker: Maker = try .init(arena, &loaded);
        var segments: Segments = try .init(gpa, testing.io, plain.text(), &loaded, 1 << 20);
        defer segments.deinit();
        // both checkpoints' added tokens strip nothing: the memo is on for both
        try testing.expect(segments.keeps);
        try testing.expectEqual(@as(usize, 0), try differing(gpa, &segments, plain.text(), &maker, 0x72, 3000, 40));
        try testing.expectEqual(@as(usize, 0), (try differingMasked(gpa, &segments, plain.text(), &maker, 0x141, 3000, 40))[0]);
        try testing.expect(segments.account().hits > segments.account().misses);
    }
    if (!ran) return error.SkipZigTest;
}
