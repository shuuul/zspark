//! A checkpoint's tokenizer for what zspark's engine does with one: a token's bytes for the grammar's masks
//! and a reply's text, the ids of a short text (a property's name, a call's frame, `chat`'s prompt). The
//! tokenizer itself is TensorFold's (its `tokenizer` module, a dependency of the build, spec 059): the one
//! `zspark serve`'s host encodes a request with, so the two cannot part, and `serve` reads tokenizer.json
//! once (`over`). Until spec 059 WS-06 this file was a tokenizer of zspark's own for the one configuration
//! Flash Next ships.
//!
//! What this file adds to TensorFold's: every token's bytes in one table, which TensorFold keeps as the
//! vocabulary's strings and decodes through its decoder chain. The table is made at load by undoing the
//! ByteLevel alphabet (a byte is the character of its value where that prints, U+0100 and up in order where
//! not), so a tokenizer.json whose decoder is not ByteLevel is refused with what it has. An added token's
//! bytes are its content.
//!
//! Decoding concatenates the tokens' bytes and replaces what is not UTF-8 with U+FFFD, as `ByteLevel` decoding
//! does. An added token with `single_word` is refused by TensorFold's tokenizer.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const engine = @import("tokenizer");

/// TensorFold's tokenizer, which `Tokenizer` stands on and may borrow (`over`).
pub const Loaded = engine.Tokenizer;

/// Why a tokenizer.json was refused.
pub const Diagnostic = struct {
    buffer: [320]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Diagnostic) []const u8 {
        return self.buffer[0..self.len];
    }

    fn fail(self: *Diagnostic, err: ParseError, comptime format: []const u8, args: anytype) ParseError {
        var writer: Io.Writer = .fixed(&self.buffer);
        writer.print(format, args) catch {};
        self.len = writer.end;
        return err;
    }
};

pub const ParseError = error{ InvalidTokenizer, UnsupportedTokenizer } || Allocator.Error;
pub const EncodeError = error{InvalidUtf8} || Allocator.Error;

pub const DecodeOptions = struct {
    /// Leave out the added tokens marked `special` (`<|im_end|>` and the like; `<think>` is not one of them).
    skip_special: bool = false,
};

/// The byte a character of a ByteLevel vocabulary stands for, or null: the library's `bytes_char` undone.
fn byteOf(point: u21) ?u8 {
    const printable = struct {
        fn is(byte: u16) bool {
            return (byte >= 0x21 and byte <= 0x7E) or (byte >= 0xA1 and byte <= 0xAC) or (byte >= 0xAE and byte <= 0xFF);
        }
    };
    if (point < 0x100) return if (printable.is(@intCast(point))) @intCast(point) else null;
    // the bytes that do not print take U+0100, U+0101, ... in the order of their values
    var next: u21 = 0x100;
    for (0..256) |byte| {
        if (printable.is(@intCast(byte))) continue;
        if (next == point) return @intCast(byte);
        next += 1;
    }
    return null;
}

pub const Tokenizer = struct {
    /// TensorFold's tokenizer: this one's when `owned`, else a host's that outlives it.
    engine: *const engine.Tokenizer,
    owned: bool,
    /// Every token's bytes end to end in id order: what the ByteLevel decoder writes for the token.
    bytes: []u8,
    /// Token `id` is `bytes[starts[id]..starts[id + 1]]`, empty for an id without a token. `[count + 1]`.
    starts: []u32,
    /// Whether id is an added token marked special. `[count]`.
    special: std.DynamicBitSetUnmanaged,
    /// The longest token in bytes.
    longest: usize,

    /// Reads `sub_path` if it is a file, `sub_path/tokenizer.json` if it is a directory.
    pub fn load(gpa: Allocator, io: Io, dir: Io.Dir, sub_path: []const u8, diagnostic: *Diagnostic) !Tokenizer {
        const limit: Io.Limit = .limited(1 << 28);
        const text = text: {
            var inner = dir.openDir(io, sub_path, .{}) catch |err| switch (err) {
                error.NotDir => break :text try dir.readFileAlloc(io, sub_path, gpa, limit),
                else => return err,
            };
            defer inner.close(io);
            break :text try inner.readFileAlloc(io, "tokenizer.json", gpa, limit);
        };
        defer gpa.free(text);
        return parse(gpa, text, diagnostic);
    }

    /// `text` is a tokenizer.json; nothing of it is kept.
    pub fn parse(gpa: Allocator, text: []const u8, diagnostic: *Diagnostic) ParseError!Tokenizer {
        const made = try gpa.create(engine.Tokenizer);
        errdefer gpa.destroy(made);
        made.* = engine.parse(gpa, text) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return diagnostic.fail(error.UnsupportedTokenizer, "TensorFold's tokenizer does not read it: {t}", .{err}),
        };
        errdefer made.deinit();
        var self = try over(gpa, made, diagnostic);
        self.owned = true;
        return self;
    }

    /// Over a tokenizer something else loaded and frees after this one: `serve`'s host's.
    pub fn over(gpa: Allocator, loaded: *const engine.Tokenizer, diagnostic: *Diagnostic) ParseError!Tokenizer {
        const decoder = loaded.decoder orelse return diagnostic.fail(error.UnsupportedTokenizer, "no decoder: only ByteLevel is read", .{});
        if (decoder != .byte_level) return diagnostic.fail(error.UnsupportedTokenizer, "decoder {t}: only ByteLevel is read", .{decoder});
        const count_ids = loaded.id_to_token.tokens.items.len;
        var special: std.DynamicBitSetUnmanaged = try .initEmpty(gpa, count_ids);
        errdefer special.deinit(gpa);
        for (loaded.added) |token| {
            if (token.id >= count_ids) continue;
            if (token.special) special.set(token.id);
        }
        var bytes: std.ArrayList(u8) = .empty;
        errdefer bytes.deinit(gpa);
        const starts = try gpa.alloc(u32, count_ids + 1);
        errdefer gpa.free(starts);
        var longest: usize = 0;
        for (0..count_ids) |id| {
            starts[id] = @intCast(bytes.items.len);
            const token = loaded.id_to_token.get(@intCast(id)) orelse continue;
            // What the ByteLevel decoder writes: the token's characters as bytes, or the token as it is when
            // one of them is no byte's (an added token's content, as a rule).
            const from = bytes.items.len;
            var view = std.unicode.Utf8View.init(token) catch return diagnostic.fail(error.InvalidTokenizer, "token {d} is not UTF-8", .{id});
            var points = view.iterator();
            while (points.nextCodepoint()) |point| {
                try bytes.append(gpa, byteOf(point) orelse {
                    bytes.shrinkRetainingCapacity(from);
                    try bytes.appendSlice(gpa, token);
                    break;
                });
            }
            longest = @max(longest, bytes.items.len - starts[id]);
        }
        starts[count_ids] = @intCast(bytes.items.len);
        return .{ .engine = loaded, .owned = false, .bytes = try bytes.toOwnedSlice(gpa), .starts = starts, .special = special, .longest = longest };
    }

    pub fn deinit(self: *Tokenizer, gpa: Allocator) void {
        gpa.free(self.bytes);
        gpa.free(self.starts);
        self.special.deinit(gpa);
        if (self.owned) {
            const made: *engine.Tokenizer = @constCast(self.engine);
            made.deinit();
            gpa.destroy(made);
        }
        self.* = undefined;
    }

    /// One past the largest id.
    pub fn count(self: *const Tokenizer) usize {
        return self.starts.len - 1;
    }

    /// The bytes token `id` decodes to: possibly part of a character, empty for an id without a token.
    pub fn tokenBytes(self: *const Tokenizer, id: u32) []const u8 {
        if (id >= self.count()) return "";
        return self.bytes[self.starts[id]..self.starts[id + 1]];
    }

    pub fn isSpecial(self: *const Tokenizer, id: u32) bool {
        return id < self.count() and self.special.isSet(id);
    }

    /// The id of the added token whose text is `content` (`<|im_end|>`), if there is one.
    pub fn addedId(self: *const Tokenizer, content: []const u8) ?u32 {
        for (self.engine.added) |candidate| {
            if (std.mem.eql(u8, candidate.content, content)) return candidate.id;
        }
        return null;
    }

    /// Appends the ids of `text` to `out`. No token is added at either end: a chat template writes them as text.
    /// Text that is not UTF-8 is refused (the library cannot be handed any).
    pub fn encode(self: *const Tokenizer, gpa: Allocator, text: []const u8, out: *std.ArrayList(u32)) EncodeError!void {
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
        const ids = self.engine.encode(gpa, text) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            // what is left of the engine's errors is a tokenizer.json's, met at load
            else => return error.InvalidUtf8,
        };
        defer gpa.free(ids);
        try out.appendSlice(gpa, ids);
    }

    /// Appends the text of `ids` to `out`: always UTF-8, with U+FFFD where the tokens' bytes are not. An id
    /// without a token adds nothing.
    pub fn decode(self: *const Tokenizer, gpa: Allocator, ids: []const u32, options: DecodeOptions, out: *std.ArrayList(u8)) Allocator.Error!void {
        var utf8: Utf8Lossy = .{};
        for (ids) |id| {
            if (options.skip_special and self.isSpecial(id)) continue;
            const bytes = self.tokenBytes(id);
            try out.ensureUnusedCapacity(gpa, Utf8Lossy.bound(bytes.len));
            out.items.len += utf8.feed(bytes, out.unusedCapacitySlice());
        }
        try out.ensureUnusedCapacity(gpa, Utf8Lossy.bound(0));
        out.items.len += utf8.finish(out.unusedCapacitySlice());
    }
};

/// Decodes ids one at a time into text that is always whole characters: the bytes of a character split across
/// tokens are held until the character is complete (or cannot be, when U+FFFD stands for them, as in `decode`).
/// The pieces together, with `finish`'s, are exactly what `Tokenizer.decode` gives for the same ids.
pub const StreamDecoder = struct {
    tokenizer: *const Tokenizer,
    options: DecodeOptions,
    utf8: Utf8Lossy = .{},
    /// What `next` returns a slice of: room for the longest token with every byte replaced. `[bound(longest)]`.
    buffer: []u8,

    pub fn init(gpa: Allocator, tokenizer: *const Tokenizer, options: DecodeOptions) Allocator.Error!StreamDecoder {
        return .{ .tokenizer = tokenizer, .options = options, .buffer = try gpa.alloc(u8, Utf8Lossy.bound(tokenizer.longest)) };
    }

    pub fn deinit(self: *StreamDecoder, gpa: Allocator) void {
        gpa.free(self.buffer);
        self.* = undefined;
    }

    /// The text that `id` completes, often empty; valid until the next call.
    pub fn next(self: *StreamDecoder, id: u32) []const u8 {
        if (self.options.skip_special and self.tokenizer.isSpecial(id)) return "";
        return self.buffer[0..self.utf8.feed(self.tokenizer.tokenBytes(id), self.buffer)];
    }

    /// U+FFFD if a character was left incomplete, else nothing. The decoder is then ready for new ids.
    pub fn finish(self: *StreamDecoder) []const u8 {
        return self.buffer[0..self.utf8.finish(self.buffer)];
    }
};

// Bytes to text.

/// Bytes in, UTF-8 out, with U+FFFD for each maximal run that cannot begin a character, as Rust's
/// `String::from_utf8_lossy` writes it. The bytes of a character not yet complete are held.
pub const Utf8Lossy = struct {
    held: [3]u8 = undefined,
    held_len: u8 = 0,
    /// How many more bytes the held character needs; 0 when nothing is held.
    needed: u8 = 0,
    /// The range its next byte must be in: narrower than 0x80..0xBF after E0, ED, F0 and F4.
    lower: u8 = 0x80,
    upper: u8 = 0xBF,

    const replacement = "\u{FFFD}";

    /// The most bytes `feed` writes for `len` bytes in, or `finish` for 0: every byte, the held ones
    /// included, replaced.
    pub fn bound(len: usize) usize {
        return replacement.len * (len + 3);
    }

    /// Writes what `bytes` completes to `out` (`bound(bytes.len)` long at least) and returns its length.
    pub fn feed(self: *Utf8Lossy, bytes: []const u8, out: []u8) usize {
        var len: usize = 0;
        for (bytes) |byte| {
            if (self.needed != 0) {
                if (byte >= self.lower and byte <= self.upper) {
                    self.lower = 0x80;
                    self.upper = 0xBF;
                    self.needed -= 1;
                    if (self.needed != 0) {
                        self.held[self.held_len] = byte;
                        self.held_len += 1;
                        continue;
                    }
                    @memcpy(out[len..][0..self.held_len], self.held[0..self.held_len]);
                    out[len + self.held_len] = byte;
                    len += self.held_len + 1;
                    self.held_len = 0;
                    continue;
                }
                // The held bytes cannot be completed: one replacement for them, and this byte starts afresh.
                len += self.finish(out[len..]);
            }
            switch (byte) {
                0x00...0x7F => {
                    out[len] = byte;
                    len += 1;
                },
                0xC2...0xDF => self.hold(byte, 1, 0x80, 0xBF),
                0xE0 => self.hold(byte, 2, 0xA0, 0xBF),
                0xE1...0xEC, 0xEE...0xEF => self.hold(byte, 2, 0x80, 0xBF),
                0xED => self.hold(byte, 2, 0x80, 0x9F),
                0xF0 => self.hold(byte, 3, 0x90, 0xBF),
                0xF1...0xF3 => self.hold(byte, 3, 0x80, 0xBF),
                0xF4 => self.hold(byte, 3, 0x80, 0x8F),
                else => {
                    @memcpy(out[len..][0..replacement.len], replacement);
                    len += replacement.len;
                },
            }
        }
        return len;
    }

    fn hold(self: *Utf8Lossy, byte: u8, needed: u8, lower: u8, upper: u8) void {
        self.held[0] = byte;
        self.held_len = 1;
        self.needed = needed;
        self.lower = lower;
        self.upper = upper;
    }

    /// Ends the text: a replacement for a held, incomplete character. Returns the length written.
    pub fn finish(self: *Utf8Lossy, out: []u8) usize {
        if (self.needed == 0) return 0;
        self.* = .{};
        @memcpy(out[0..replacement.len], replacement);
        return replacement.len;
    }
};
