//! `tokenizer.zig`'s tests: the expected values are the `tokenizers` library's, written by tools/reference/tokenizer_cases.py.

const std = @import("std");
const tokenizer_file = @import("../../src/text/tokenizer.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const json = std.json;
const Diagnostic = tokenizer_file.Diagnostic;
const ParseError = tokenizer_file.ParseError;
const StreamDecoder = tokenizer_file.StreamDecoder;
const Tokenizer = tokenizer_file.Tokenizer;
const Utf8Lossy = tokenizer_file.Utf8Lossy;
const DecodeOptions = tokenizer_file.DecodeOptions;

const testing = std.testing;

const fx = struct {
    const tiny_tokenizer = @embedFile("fixtures/text/tiny_tokenizer.json");
    const tiny_tokenizer_pairs = @embedFile("fixtures/text/tiny_tokenizer_pairs.json");
    const tiny_cases = @embedFile("fixtures/text/tiny_cases.json");
    const cases = @embedFile("fixtures/text/tokenizer_cases.json");

    const Case = struct {
        text: []const u8,
        ids: []const u32,
        /// What the ids decode to, where that is not `text`.
        decoded: ?[]const u8 = null,
    };
    const Decode = struct {
        ids: []const u32,
        text: []const u8,
        /// The text without special tokens, where that is not `text`.
        skipped: ?[]const u8 = null,
    };
    const Nfc = struct { text: []const u8, nfc: []const u8 };
    const Pieces = struct { text: []const u8, pieces: []const []const u8 };
    const Fixture = struct {
        cases: []const Case,
        decode: []const Decode,
        nfc: []const Nfc = &.{},
        nfc_unchanged: []const []const u8 = &.{},
        pieces: []const Pieces = &.{},
    };

    fn read(arena: Allocator, text: []const u8) !Fixture {
        return json.parseFromSliceLeaky(Fixture, arena, text, .{ .ignore_unknown_fields = true });
    }

    fn tiny(gpa: Allocator) !Tokenizer {
        var diagnostic: Diagnostic = .{};
        return Tokenizer.parse(gpa, tiny_tokenizer, &diagnostic) catch |err| {
            std.debug.print("tiny: {t} {s}\n", .{ err, diagnostic.message() });
            return err;
        };
    }

    /// Decodes `ids` whole and as a stream, checks that the stream's pieces are whole characters and add up
    /// to the whole, and returns the text.
    fn decodeBoth(arena: Allocator, tokenizer: *const Tokenizer, ids: []const u32, options: DecodeOptions) ![]const u8 {
        var whole: std.ArrayList(u8) = .empty;
        try tokenizer.decode(arena, ids, options, &whole);
        var stream: StreamDecoder = try .init(arena, tokenizer, options);
        var streamed: std.ArrayList(u8) = .empty;
        for (ids) |id| {
            const piece = stream.next(id);
            try testing.expect(std.unicode.utf8ValidateSlice(piece));
            try streamed.appendSlice(arena, piece);
        }
        try streamed.appendSlice(arena, stream.finish());
        try testing.expectEqualStrings(whole.items, streamed.items);
        return whole.items;
    }

    /// Every case of a fixture against a tokenizer: ids, the text back, and the stream. Reports the first
    /// cases that differ and fails if any did.
    fn check(arena: Allocator, tokenizer: *const Tokenizer, fixture: Fixture) !void {
        var wrong: usize = 0;
        var ids: std.ArrayList(u32) = .empty;
        for (fixture.cases) |case| {
            ids.clearRetainingCapacity();
            try tokenizer.encode(arena, case.text, &ids);
            const text = try decodeBoth(arena, tokenizer, case.ids, .{});
            if (std.mem.eql(u32, case.ids, ids.items) and std.mem.eql(u8, case.decoded orelse case.text, text)) continue;
            wrong += 1;
            if (wrong <= 5) std.debug.print("\"{f}\"\n  library {any}\n  zspark  {any}\n  decoded \"{f}\"\n", .{ std.zig.fmtString(case.text), case.ids, ids.items, std.zig.fmtString(text) });
        }
        for (fixture.decode) |case| {
            const shown = try decodeBoth(arena, tokenizer, case.ids, .{});
            const skipped = try decodeBoth(arena, tokenizer, case.ids, .{ .skip_special = true });
            if (std.mem.eql(u8, case.text, shown) and std.mem.eql(u8, case.skipped orelse case.text, skipped)) continue;
            wrong += 1;
            if (wrong <= 5) std.debug.print("{any}\n  library \"{f}\"\n  zspark  \"{f}\"\n  skipping \"{f}\"\n", .{ case.ids, std.zig.fmtString(case.text), std.zig.fmtString(shown), std.zig.fmtString(skipped) });
        }
        if (wrong != 0) {
            std.debug.print("{d} of {d} cases differ from the library\n", .{ wrong, fixture.cases.len + fixture.decode.len });
            return error.TestExpectedEqual;
        }
    }
};

test "a tiny tokenizer: ids, decoding and the stream equal the library's" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fixture = try fx.read(arena, fx.tiny_cases);
    try testing.expect(fixture.cases.len > 1000 and fixture.decode.len > 100);

    var tokenizer = try fx.tiny(testing.allocator);
    defer tokenizer.deinit(testing.allocator);
    try fx.check(arena, &tokenizer, fixture);

    // the same tokenizer with its merges written as pairs
    var diagnostic: Diagnostic = .{};
    var pairs = try Tokenizer.parse(testing.allocator, fx.tiny_tokenizer_pairs, &diagnostic);
    defer pairs.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, tokenizer.bytes, pairs.bytes);
    try fx.check(arena, &pairs, fixture);
}

test "the published checkpoint's tokenizer: ids, decoding and the stream equal the library's" {
    // Needs ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory, for its 12.8 MB tokenizer.json.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fixture = try fx.read(arena, fx.cases);
    try testing.expect(fixture.cases.len > 1000 and fixture.decode.len > 100);

    var diagnostic: Diagnostic = .{};
    var tokenizer = Tokenizer.load(testing.allocator, testing.io, .cwd(), path, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message()});
        return err;
    };
    defer tokenizer.deinit(testing.allocator);
    try testing.expectEqual(248077, tokenizer.count());
    try testing.expectEqual(33, tokenizer.engine.added.len);
    try testing.expectEqual(248046, tokenizer.addedId("<|im_end|>").?);
    try testing.expect(tokenizer.isSpecial(248046) and !tokenizer.isSpecial(248068));
    try fx.check(arena, &tokenizer, fixture);

    // the file itself is read the same as its directory
    const file = try std.fs.path.join(arena, &.{ path, "tokenizer.json" });
    var again = try Tokenizer.load(testing.allocator, testing.io, .cwd(), file, &diagnostic);
    defer again.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, tokenizer.starts, again.starts);
}

test "the GLM-5.3-Flash pack's tokenizer: its stops are special, and the corpus comes back whole and as a stream" {
    // Runs with ZSPARK_GLM_FP4_LOCAL set; skipped without. The ids are TensorFold's tokenizer's own (no library
    // wrote a fixture for this one): what is checked is what zspark keeps of it, each token's bytes and the
    // stream. The tokenizer has no normalizer, so text encoded and decoded is the text.
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fixture = try fx.read(arena, fx.cases);

    var diagnostic: Diagnostic = .{};
    var tokenizer = Tokenizer.load(testing.allocator, testing.io, .cwd(), path, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{diagnostic.message()});
        return err;
    };
    defer tokenizer.deinit(testing.allocator);
    // 154,820 of the model and 36 added; the config's 154,880 rows leave 24 unnamed
    try testing.expectEqual(154856, tokenizer.count());
    try testing.expectEqual(154820, tokenizer.addedId("<|endoftext|>").?);
    try testing.expectEqual(154827, tokenizer.addedId("<|user|>").?);
    try testing.expectEqual(154829, tokenizer.addedId("<|observation|>").?);
    for ([_]u32{ 154820, 154827, 154829 }) |stop| try testing.expect(tokenizer.isSpecial(stop));

    var ids: std.ArrayList(u32) = .empty;
    var wrong: usize = 0;
    for (fixture.cases) |case| {
        ids.clearRetainingCapacity();
        try tokenizer.encode(arena, case.text, &ids);
        const text = try fx.decodeBoth(arena, &tokenizer, ids.items, .{});
        if (std.mem.eql(u8, case.text, text)) continue;
        wrong += 1;
        if (wrong <= 5) std.debug.print("\"{f}\"\n  back \"{f}\"\n", .{ std.zig.fmtString(case.text), std.zig.fmtString(text) });
    }
    try testing.expectEqual(0, wrong);
}

test "more random strings than the repository holds, when ZSPARK_TOKENIZER_MORE names their directory" {
    // tools/reference/tokenizer_cases.py --more N --more-dir DIR writes them.
    const path = testing.environ.getPosix("ZSPARK_TOKENIZER_MORE") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    {
        const fixture = try fx.read(arena, try dir.readFileAlloc(testing.io, "tiny_cases_more.json", arena, .limited(1 << 30)));
        var tokenizer = try fx.tiny(gpa);
        defer tokenizer.deinit(gpa);
        try fx.check(arena, &tokenizer, fixture);
    }
    const checkpoint = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return;
    const fixture = try fx.read(arena, try dir.readFileAlloc(testing.io, "tokenizer_cases_more.json", arena, .limited(1 << 30)));
    var diagnostic: Diagnostic = .{};
    var tokenizer = try Tokenizer.load(gpa, testing.io, .cwd(), checkpoint, &diagnostic);
    defer tokenizer.deinit(gpa);
    try fx.check(arena, &tokenizer, fixture);
}

test "the stream holds a character back until its last byte" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    // 中 is three bytes and two tokens here (its first two bytes merge); é is one token
    try tokenizer.encode(gpa, "a中é", &ids);
    try testing.expectEqual(4, ids.items.len);
    var stream: StreamDecoder = try .init(gpa, &tokenizer, .{ .skip_special = true });
    defer stream.deinit(gpa);
    try testing.expectEqualStrings("a", stream.next(ids.items[0]));
    try testing.expectEqualStrings("", stream.next(ids.items[1]));
    try testing.expectEqualStrings("中", stream.next(ids.items[2]));
    try testing.expectEqualStrings("é", stream.next(ids.items[3]));
    try testing.expectEqualStrings("", stream.finish());
    // cut short, the held bytes become one replacement character, and a special token is not shown
    try testing.expectEqualStrings("", stream.next(ids.items[1]));
    try testing.expectEqualStrings("", stream.next(tokenizer.addedId("<|end|>").?));
    try testing.expectEqualStrings("\u{FFFD}", stream.finish());
    try testing.expectEqualStrings("", stream.next(ids.items[1]));
    try testing.expectEqualStrings("\u{FFFD}a", stream.next(ids.items[0]));
    try testing.expectEqualStrings("<t>", stream.next(tokenizer.addedId("<t>").?));
}

test "bytes that are not UTF-8 decode as Rust's from_utf8_lossy writes them" {
    const cases = [_][2][]const u8{
        .{ "a\xffb", "a\u{FFFD}b" },
        .{ "\xc3", "\u{FFFD}" },
        .{ "\xc3\xa9\xa9", "é\u{FFFD}" },
        .{ "\xc0\xaf", "\u{FFFD}\u{FFFD}" },
        .{ "\xe4\xb8a", "\u{FFFD}a" },
        .{ "\xe0\x80\x80", "\u{FFFD}\u{FFFD}\u{FFFD}" },
        .{ "\xed\xa0\x80", "\u{FFFD}\u{FFFD}\u{FFFD}" },
        .{ "\xf0\x9f\x98", "\u{FFFD}" },
        .{ "\xf0\x9f\x98\xf0\x9f\x98\x80", "\u{FFFD}😀" },
        .{ "\xf4\x90\x80\x80", "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}" },
        .{ "\xf5\x80", "\u{FFFD}\u{FFFD}" },
        .{ "\xe4\xc3\xa9", "\u{FFFD}é" },
    };
    for (cases) |case| {
        // whole, and a byte at a time
        var utf8: Utf8Lossy = .{};
        var out: [64]u8 = undefined;
        var len = utf8.feed(case[0], &out);
        len += utf8.finish(out[len..]);
        try testing.expectEqualStrings(case[1], out[0..len]);
        len = 0;
        for (case[0]) |byte| {
            const n = utf8.feed(&.{byte}, out[len..]);
            try testing.expect(n <= Utf8Lossy.bound(1));
            try testing.expect(std.unicode.utf8ValidateSlice(out[len..][0..n]));
            len += n;
        }
        len += utf8.finish(out[len..]);
        try testing.expectEqualStrings(case[1], out[0..len]);
    }
}

test "text that is not UTF-8 is refused" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try testing.expectError(error.InvalidUtf8, tokenizer.encode(gpa, "ab\xff", &ids));
    try testing.expectError(error.InvalidUtf8, tokenizer.encode(gpa, "\xed\xa0\x80", &ids));
    try testing.expectEqual(0, ids.items.len);
}

fn expectRefused(expected: ParseError, needle: []const u8, replacement: []const u8, message: []const u8) !void {
    const gpa = testing.allocator;
    try testing.expect(std.mem.indexOf(u8, fx.tiny_tokenizer, needle) != null);
    const text = try std.mem.replaceOwned(u8, gpa, fx.tiny_tokenizer, needle, replacement);
    defer gpa.free(text);
    var diagnostic: Diagnostic = .{};
    if (Tokenizer.parse(gpa, text, &diagnostic)) |tokenizer| {
        var loaded = tokenizer;
        loaded.deinit(gpa);
        return error.TestUnexpectedResult;
    } else |err| try testing.expectEqual(expected, err);
    if (std.mem.indexOf(u8, diagnostic.message(), message) == null) {
        std.debug.print("message: {s}\n", .{diagnostic.message()});
        return error.TestUnexpectedResult;
    }
}

test "a tokenizer.json whose tokens are not bytes, or that TensorFold's tokenizer does not read, is refused" {
    const unsupported = error.UnsupportedTokenizer;
    try expectRefused(unsupported, "\"decoder\":{\"type\":\"ByteLevel\"", "\"decoder\":{\"type\":\"Fuse\"", "only ByteLevel is read");
    try expectRefused(unsupported, "\"vocab\":{", "\"vocab\":[{", "TensorFold's tokenizer does not read it");
}
