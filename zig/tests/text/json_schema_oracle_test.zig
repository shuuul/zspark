//! `json_schema.zig`'s masks against xgrammar's, recorded in `zig/tests/fixtures/text/grammar_masks*.txt`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("testing.zig");
const fx = fixtures.schemas;
const json_schema = @import("../../src/text/json_schema.zig");
const tokenizer_file = @import("../../src/text/tokenizer.zig");
const Diagnostic = json_schema.Diagnostic;
const Grammar = json_schema.Grammar;
const Matcher = json_schema.Matcher;
const Options = json_schema.Options;
const Tokenizer = json_schema.Tokenizer;
const Vocabulary = json_schema.Vocabulary;

/// xgrammar's masks over a corpus of schemas and token sequences (`tools/reference/grammar_masks.zig` writes the
/// files and says what each line is), and this matcher held to them.
const oracle = struct {
    const tiny_masks = @embedFile("fixtures/text/grammar_masks.txt");
    const checkpoint_masks = @embedFile("fixtures/text/grammar_masks_checkpoint.txt");

    /// Whether `text`, a JSON document from its first byte, holds what xgrammar takes in a string and this
    /// matcher refuses by intent: a surrogate encoded as UTF-8 (ED, then A0 to BF), or an escaped surrogate
    /// without its partner (a high one that no escaped low one follows, a low one that follows no high one).
    /// A scan of the text that knows nothing of the machine; what is cut short at the end decides nothing.
    fn surrogate(text: []const u8) bool {
        var in_string = false;
        // the escape before was a high surrogate
        var high = false;
        var at: usize = 0;
        while (at < text.len) {
            const byte = text[at];
            if (byte == 0xed and at + 1 < text.len and text[at + 1] >= 0xa0) return true;
            if (!in_string) {
                in_string = byte == '"';
                at += 1;
                continue;
            }
            if (high) {
                const forms = [_][]const u8{ "\\", "u", "dD", "cdefCDEF" };
                for (forms, 0..) |form, ahead| {
                    if (at + ahead >= text.len) return false;
                    if (std.mem.findScalar(u8, form, text[at + ahead]) == null) return true;
                }
                high = false;
                at += 6;
                continue;
            }
            if (byte == '\\') {
                if (at + 1 >= text.len) return false;
                if (text[at + 1] != 'u') {
                    at += 2;
                    continue;
                }
                if (at + 3 < text.len and (text[at + 2] == 'd' or text[at + 2] == 'D')) {
                    if (std.mem.findScalar(u8, "cdefCDEF", text[at + 3]) != null) return true;
                    high = std.mem.findScalar(u8, "89abAB", text[at + 3]) != null;
                }
                at += 6;
                continue;
            }
            if (byte == '"') in_string = false;
            at += 1;
        }
        return false;
    }

    /// FNV-1a over the words' bytes, least significant first, with only the bits of `keep` when it is given:
    /// what a checkpoint mask is written as.
    fn hash(words: []const u32, keep: ?[]const u32) u64 {
        var value: u64 = 0xcbf29ce484222325;
        for (words, 0..) |word, at| {
            const kept = if (keep) |k| word & k[at] else word;
            for (0..4) |byte| {
                value ^= (kept >> @intCast(8 * byte)) & 0xff;
                value *%= 0x100000001b3;
            }
        }
        return value;
    }

    const Counts = struct {
        schemas: usize = 0,
        walks: usize = 0,
        /// Masks held against xgrammar's, and those of them that were xgrammar's bit for bit.
        steps: usize = 0,
        equal: usize = 0,
        /// Tokens xgrammar's mask holds and this matcher's does not, each one a surrogate.
        surrogates: usize = 0,
        /// And those that are not, under a schema this matcher reads more narrowly.
        narrowed: usize = 0,
        /// Checkpoint masks that could not be held to a hash: the text before them ends in an escape or in ED.
        unheld: usize = 0,
        /// Documents xgrammar takes, refused here at a surrogate; and special tokens refused.
        refused: usize = 0,
        specials: usize = 0,
        ended: usize = 0,
    };

    fn numbers(gpa: Allocator, line: []const u8, out: *std.ArrayList(u32)) !void {
        out.clearRetainingCapacity();
        var parts = std.mem.tokenizeScalar(u8, line, ' ');
        while (parts.next()) |part| try out.append(gpa, try std.fmt.parseInt(u32, part, 10));
    }

    /// Every walk of `file` through this matcher on `tokenizer`. Before each token the mask is xgrammar's, but
    /// for tokens that make a surrogate; the token is taken; an `only` walk is refused at a surrogate and a
    /// `special` one at its special token.
    fn check(gpa: Allocator, tokenizer: *const Tokenizer, file: []const u8) !Counts {
        var counts: Counts = .{};
        var table: std.ArrayList([]const u8) = .empty;
        defer table.deinit(gpa);
        var lines = std.mem.splitScalar(u8, file, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "mask ")) try table.append(gpa, line[5..]);
        }
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        var marks: std.ArrayList(u32) = .empty;
        defer marks.deinit(gpa);
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);

        var vocabulary: Vocabulary = undefined;
        var have_vocabulary = false;
        defer if (have_vocabulary) vocabulary.deinit(gpa);
        var grammar: Grammar = undefined;
        var have_grammar = false;
        defer if (have_grammar) grammar.deinit();
        var whole = true;
        var stops: [4]u32 = undefined;
        var stop_count: usize = 0;
        var theirs: []u32 = &.{};
        defer gpa.free(theirs);
        var ours: []u32 = &.{};
        defer gpa.free(ours);
        var plain: []u32 = &.{};
        defer gpa.free(plain);
        var name: []const u8 = "";
        // the schema is one this matcher reads more narrowly than xgrammar: its masks are within xgrammar's
        var narrowed = false;

        lines = std.mem.splitScalar(u8, file, '\n');
        while (lines.next()) |line| {
            if (line.len == 0 or line[0] == '#' or std.mem.startsWith(u8, line, "mask ")) continue;
            if (std.mem.startsWith(u8, line, "xgrammar ")) {
                // the blanks xgrammar was told to allow are this matcher's
                var parts = std.mem.tokenizeScalar(u8, line, ' ');
                _ = parts.next();
                _ = parts.next();
                try testing.expectEqualStrings("whitespace", parts.next().?);
                try testing.expectEqual((Options{}).blanks, try std.fmt.parseInt(u8, parts.next().?, 10));
            } else if (std.mem.startsWith(u8, line, "vocabulary ")) {
                var parts = std.mem.tokenizeScalar(u8, line, ' ');
                _ = parts.next();
                whole = std.mem.eql(u8, parts.next().?, "tiny");
                // another tokenizer than the one the file was written on: nothing to hold it to
                if (try std.fmt.parseInt(usize, parts.next().?, 10) != tokenizer.count()) return error.SkipZigTest;
                const width = try std.fmt.parseInt(usize, parts.next().?, 10);
                while (parts.next()) |part| : (stop_count += 1) stops[stop_count] = try std.fmt.parseInt(u32, part, 10);
                vocabulary = try .init(gpa, tokenizer, width, stops[0..stop_count]);
                have_vocabulary = true;
                theirs = try gpa.alloc(u32, vocabulary.words());
                ours = try gpa.alloc(u32, vocabulary.words());
                plain = try gpa.alloc(u32, vocabulary.words());
                @memset(plain, 0);
                for (0..tokenizer.count()) |index| {
                    const bytes = tokenizer.tokenBytes(@intCast(index));
                    if (bytes.len == 0 or tokenizer.isSpecial(@intCast(index))) continue;
                    if (std.mem.findAny(u8, bytes, "\xed\\") == null) plain[index / 32] |= @as(u32, 1) << @intCast(index % 32);
                }
                for (stops[0..stop_count]) |stop| plain[stop / 32] |= @as(u32, 1) << @intCast(stop % 32);
            } else if (std.mem.startsWith(u8, line, "schema ")) {
                var parts = std.mem.tokenizeScalar(u8, line[7..], ' ');
                name = parts.next().?;
                narrowed = parts.next() != null;
                if (have_grammar) grammar.deinit();
                have_grammar = false;
                var diagnostic: Diagnostic = .{};
                grammar = Grammar.init(gpa, &vocabulary, lines.next().?, .{}, &diagnostic) catch |err| {
                    std.debug.print("{s}: {s}\n", .{ name, diagnostic.message() });
                    return err;
                };
                have_grammar = true;
                counts.schemas += 1;
            } else if (std.mem.startsWith(u8, line, "walk ")) {
                var parts = std.mem.tokenizeScalar(u8, line, ' ');
                _ = parts.next();
                const kind = parts.next().?;
                const ended = parts.next().?[0] == '1';
                const only = std.mem.eql(u8, kind, "only");
                const special = std.mem.eql(u8, kind, "special");
                try numbers(gpa, lines.next().?[1..], &ids);
                if (!special) try numbers(gpa, lines.next().?[1..], &marks);
                if (!special) try testing.expectEqual(ids.items.len, marks.items.len);
                counts.walks += 1;
                var matcher: Matcher = try .init(&grammar);
                defer matcher.deinit();
                text.clearRetainingCapacity();
                var refused = false;
                for (ids.items, 0..) |id, step| {
                    errdefer std.debug.print("{s}, a {s} walk, token {d} ({d} `{f}`) after: {f}\n", .{ name, kind, step, id, std.zig.fmtString(tokenizer.tokenBytes(id)), std.zig.fmtString(text.items) });
                    const before = text.items.len;
                    if (!special) {
                        try matcher.fill(ours);
                        counts.steps += 1;
                        const mask = table.items[marks.items[step]];
                        if (whole) {
                            for (theirs, 0..) |*word, at| word.* = try std.fmt.parseInt(u32, mask[at * 8 ..][0..8], 16);
                            if (std.mem.eql(u32, theirs, ours)) counts.equal += 1;
                            for (theirs, ours, 0..) |their_word, our_word, word| {
                                // nothing is taken here that xgrammar refuses
                                try testing.expectEqual(@as(u32, 0), our_word & ~their_word);
                                var more = their_word & ~our_word;
                                while (more != 0) : (more &= more - 1) {
                                    const token: u32 = @intCast(word * 32 + @ctz(more));
                                    errdefer std.debug.print("xgrammar alone takes {d} `{f}`\n", .{ token, std.zig.fmtString(tokenizer.tokenBytes(token)) });
                                    try testing.expect(token < tokenizer.count());
                                    try text.appendSlice(gpa, tokenizer.tokenBytes(token));
                                    defer text.shrinkRetainingCapacity(before);
                                    if (surrogate(text.items)) {
                                        counts.surrogates += 1;
                                    } else {
                                        try testing.expect(narrowed);
                                        counts.narrowed += 1;
                                    }
                                }
                            }
                        } else {
                            var parts_of = std.mem.tokenizeScalar(u8, mask, ' ');
                            _ = parts_of.next();
                            const all = try std.fmt.parseInt(u64, parts_of.next().?, 16);
                            const kept = try std.fmt.parseInt(u64, parts_of.next().?, 16);
                            if (hash(ours, null) == all) {
                                counts.equal += 1;
                            } else if (narrowed) {
                                counts.narrowed += 1;
                            } else if (std.mem.findScalar(u8, text.items[text.items.len -| 11..], '\\') != null or (text.items.len > 0 and text.items[text.items.len - 1] == 0xed)) {
                                counts.unheld += 1;
                            } else {
                                // no token without a backslash or ED makes a surrogate here
                                try testing.expectEqual(kept, hash(ours, plain));
                            }
                        }
                    }
                    const taken = try matcher.accept(id);
                    if (std.mem.findScalar(u32, stops[0..stop_count], id) == null) try text.appendSlice(gpa, tokenizer.tokenBytes(id));
                    if (taken) continue;
                    refused = true;
                    if (only) {
                        try testing.expect(surrogate(text.items) and !surrogate(text.items[0..before]));
                        counts.refused += 1;
                    } else {
                        try testing.expect(special and step == ids.items.len - 1 and tokenizer.isSpecial(id));
                        counts.specials += 1;
                    }
                    break;
                }
                try testing.expectEqual(only or special, refused);
                if (!refused) try testing.expectEqual(ended, matcher.finished());
                if (!refused and ended) counts.ended += 1;
            } else return error.UnknownLine;
        }
        return counts;
    }
};

test "the masks are xgrammar's over a corpus of schemas, but for the surrogates and special tokens it takes" {
    const gpa = testing.allocator;
    var tokenizer = try fx.tiny(gpa);
    defer tokenizer.deinit(gpa);
    const counts = try oracle.check(gpa, &tokenizer, oracle.tiny_masks);
    try testing.expect(counts.schemas >= 30 and counts.walks >= 600 and counts.steps >= 19000);
    // nearly every mask is xgrammar's bit for bit; the others part from it by tokens that make a surrogate,
    // or under the one schema read more narrowly, and both were met
    try testing.expect(counts.equal * 100 >= counts.steps * 98);
    try testing.expect(counts.surrogates > 1000 and counts.narrowed > 1000);
    try testing.expect(counts.refused >= 7);
    try testing.expectEqual(1, counts.specials);
    try testing.expect(counts.ended * 10 >= counts.walks * 9);
}

test "xgrammar's masks on the published checkpoint's vocabulary, held by their hashes" {
    // Needs ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory, for its tokenizer.json; no GPU and no weights.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var diagnostic: tokenizer_file.Diagnostic = .{};
    var tokenizer = try Tokenizer.load(gpa, testing.io, .cwd(), path, &diagnostic);
    defer tokenizer.deinit(gpa);
    const counts = try oracle.check(gpa, &tokenizer, oracle.checkpoint_masks);
    try testing.expect(counts.schemas >= 30 and counts.walks >= 300 and counts.steps >= 5000);
    // a hash says equal or not: the masks not held are the few after an open escape or ED, where the tiny
    // vocabulary's test names every differing token
    try testing.expect(counts.equal * 100 >= counts.steps * 95);
    try testing.expect(counts.unheld * 50 <= counts.steps);
    try testing.expect(counts.refused >= 7);
    try testing.expectEqual(1, counts.specials);
}
