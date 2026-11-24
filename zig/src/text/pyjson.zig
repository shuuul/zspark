//! `json.loads` as Python reads a string, into a `std.json.Value`. TensorFold parses tool-call arguments with it,
//! and it accepts what `std.json` refuses: `NaN`, `Infinity` and `-Infinity`, a float too large for fp64 (it
//! becomes an infinity), and a key given twice (the last value, at the first one's place).
//!
//! Departures: an escaped surrogate with no partner (`"\ud800"`) is refused, because the result is UTF-8 and has
//! no way to hold one; Python keeps it. An integer outside i64 becomes a `.number_string`, as `std.json` gives.
//! Nesting deeper than `max_depth` is refused; Python raises `RecursionError` somewhat later.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Error = error{ OutOfMemory, InvalidJson };

pub const max_depth = 512;

/// The value `text` holds. Everything is allocated in `arena`; nothing points into `text`.
pub fn loads(arena: Allocator, text: []const u8) Error!Value {
    var p: Parser = .{ .arena = arena, .text = text };
    p.skipSpace();
    const value = try p.value(0);
    p.skipSpace();
    if (p.at != text.len) return error.InvalidJson;
    return value;
}

const Parser = struct {
    arena: Allocator,
    text: []const u8,
    at: usize = 0,

    fn skipSpace(p: *Parser) void {
        while (p.at < p.text.len) : (p.at += 1) switch (p.text[p.at]) {
            ' ', '\t', '\n', '\r' => {},
            else => return,
        };
    }

    fn take(p: *Parser, word: []const u8) bool {
        if (!std.mem.startsWith(u8, p.text[p.at..], word)) return false;
        p.at += word.len;
        return true;
    }

    fn value(p: *Parser, depth: usize) Error!Value {
        if (p.at == p.text.len or depth > max_depth) return error.InvalidJson;
        switch (p.text[p.at]) {
            '"' => return .{ .string = try p.string() },
            '{' => {
                p.at += 1;
                var object: std.json.ObjectMap = .empty;
                p.skipSpace();
                if (p.take("}")) return .{ .object = object };
                while (true) {
                    p.skipSpace();
                    if (p.at == p.text.len or p.text[p.at] != '"') return error.InvalidJson;
                    const key = try p.string();
                    p.skipSpace();
                    if (!p.take(":")) return error.InvalidJson;
                    p.skipSpace();
                    try object.put(p.arena, key, try p.value(depth + 1));
                    p.skipSpace();
                    if (p.take("}")) return .{ .object = object };
                    if (!p.take(",")) return error.InvalidJson;
                }
            },
            '[' => {
                p.at += 1;
                var array: std.json.Array = .init(p.arena);
                p.skipSpace();
                if (p.take("]")) return .{ .array = array };
                while (true) {
                    p.skipSpace();
                    try array.append(try p.value(depth + 1));
                    p.skipSpace();
                    if (p.take("]")) return .{ .array = array };
                    if (!p.take(",")) return error.InvalidJson;
                }
            },
            'n' => return if (p.take("null")) .null else error.InvalidJson,
            't' => return if (p.take("true")) .{ .bool = true } else error.InvalidJson,
            'f' => return if (p.take("false")) .{ .bool = false } else error.InvalidJson,
            'N' => return if (p.take("NaN")) .{ .float = std.math.nan(f64) } else error.InvalidJson,
            'I' => return if (p.take("Infinity")) .{ .float = std.math.inf(f64) } else error.InvalidJson,
            else => {
                if (p.take("-Infinity")) return .{ .float = -std.math.inf(f64) };
                return p.number();
            },
        }
    }

    fn digits(p: *Parser) usize {
        const start = p.at;
        while (p.at < p.text.len and std.ascii.isDigit(p.text[p.at])) p.at += 1;
        return p.at - start;
    }

    /// `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`; a fraction or exponent that does not complete is left unread, as
    /// Python's scanner leaves it, and fails as trailing text.
    fn number(p: *Parser) Error!Value {
        const start = p.at;
        _ = p.take("-");
        if (!p.take("0") and (p.at == p.text.len or p.text[p.at] == '0' or p.digits() == 0)) return error.InvalidJson;
        var float = false;
        var mark = p.at;
        if (p.take(".")) {
            if (p.digits() != 0) float = true else p.at = mark;
        }
        mark = p.at;
        if (p.take("e") or p.take("E")) {
            if (!p.take("-")) _ = p.take("+");
            if (p.digits() != 0) float = true else p.at = mark;
        }
        const text = p.text[start..p.at];
        if (float) return .{ .float = std.fmt.parseFloat(f64, text) catch return error.InvalidJson };
        const integer = std.fmt.parseInt(i64, text, 10) catch return .{ .number_string = try p.arena.dupe(u8, text) };
        return .{ .integer = integer };
    }

    fn hex4(p: *Parser) Error!u16 {
        if (p.text.len - p.at < 4) return error.InvalidJson;
        for (p.text[p.at..][0..4]) |c| if (!std.ascii.isHex(c)) return error.InvalidJson;
        defer p.at += 4;
        return std.fmt.parseInt(u16, p.text[p.at..][0..4], 16) catch unreachable;
    }

    fn string(p: *Parser) Error![]const u8 {
        p.at += 1;
        var out: std.ArrayList(u8) = .empty;
        while (true) {
            if (p.at == p.text.len) return error.InvalidJson;
            const c = p.text[p.at];
            p.at += 1;
            switch (c) {
                '"' => return out.items,
                0...0x1f => return error.InvalidJson,
                '\\' => {
                    if (p.at == p.text.len) return error.InvalidJson;
                    const escape = p.text[p.at];
                    p.at += 1;
                    switch (escape) {
                        '"', '\\', '/' => try out.append(p.arena, escape),
                        'b' => try out.append(p.arena, 0x08),
                        'f' => try out.append(p.arena, 0x0c),
                        'n' => try out.append(p.arena, '\n'),
                        'r' => try out.append(p.arena, '\r'),
                        't' => try out.append(p.arena, '\t'),
                        'u' => {
                            var point: u21 = try p.hex4();
                            if (point >= 0xdc00 and point < 0xe000) return error.InvalidJson;
                            if (point >= 0xd800 and point < 0xdc00) {
                                if (!p.take("\\u")) return error.InvalidJson;
                                const low = try p.hex4();
                                if (low < 0xdc00 or low >= 0xe000) return error.InvalidJson;
                                point = 0x10000 + ((point - 0xd800) << 10) + (low - 0xdc00);
                            }
                            var buffer: [4]u8 = undefined;
                            const n = std.unicode.utf8Encode(point, &buffer) catch return error.InvalidJson;
                            try out.appendSlice(p.arena, buffer[0..n]);
                        },
                        else => return error.InvalidJson,
                    }
                },
                else => try out.append(p.arena, c),
            }
        }
    }
};
