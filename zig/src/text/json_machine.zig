//! A document's position under a schema's tree: the frames, one a value that is open, and `Machine`, which
//! takes a byte from a position. `json_schema.zig`'s header says how a mask is made of them.

const std = @import("std");
const json_builder = @import("json_builder.zig");
const json_schema = @import("json_schema.zig");
const Schema = json_builder.Schema;
const fixed = json_builder.fixed;
const none = json_schema.none;

// -- positions ---------------------------------------------------------------------------------------------------

/// Where a frame is in its value.
pub const Phase = enum(u8) {
    /// Before the value's first byte.
    start,
    // a string
    body,
    escape,
    /// `a` hex digits are still to come; `b` is what the digits so far say of a surrogate (`hex_*`).
    hex,
    /// After the escape of a high surrogate: the `\` of the low one's escape must come, then its `u`.
    pair,
    pair_u,
    /// `a` continuation bytes are still to come, the next one from `b & 0xff` to `b >> 8`.
    utf8,
    // an integer, or a number's integer part
    minus,
    zero,
    digits,
    // a number
    point,
    fraction,
    exponent,
    exponent_sign,
    exponent_digits,
    /// Literals: at node `a` of their trie.
    word,
    // an object: `a` is the earliest property not yet passed. `first` is after `{`, `next` after a comma, `key`
    // inside a key at node `b` of its trie, `colon` and `value` before those of property `b` (`none`: any key),
    // `after` behind a value.
    first,
    key,
    colon,
    value,
    after,
    next,
    // an array: `a` is how many items it has, counted as far as its bounds need.
    item_first,
    item_after,
    item_next,
};

/// An open value. Frames are only ever added: one that moves on is written again above the same parent.
pub const Frame = struct {
    /// The frame of the value this one is inside, `done` for the document, `outside` when it is not known.
    parent: u32,
    node: u32,
    phase: Phase,
    /// The blank characters taken at this place.
    blanks: u8 = 0,
    a: u32 = 0,
    b: u32 = 0,
};

// A `\u` escape's digits so far (`Frame.b` in `.hex`). U+D800 to U+DBFF is a high surrogate and must be followed
// by the escape of a low one, U+DC00 to U+DFFF, which may not come alone: JSON's grammar takes either alone, and
// a reader that decodes the string (RFC 8259, section 8.2: "unpredictable") may refuse the document.
const hex_first = 0;
/// The first digit was `d`.
const hex_d = 1;
/// `d8` to `db`: a high surrogate.
const hex_high = 2;
/// Neither kind of surrogate.
const hex_plain = 3;
/// The escape after a high surrogate's: its digits must be `dc00` to `dfff`.
const hex_low_first = 4;
const hex_low_d = 5;

/// The document is whole: nothing but a stop follows.
pub const done = none;
/// Below the frame a cache entry is made for: what follows depends on the frames that are not there.
pub const outside = none - 1;
/// What `Machine.step` returns for a byte that cannot follow.
pub const rejected = none - 2;
/// What `Machine.step` returns for a byte that reached `outside` and was not taken.
pub const unknown = none - 3;

pub const Machine = struct {
    schema: *const Schema,
    frames: std.ArrayList(Frame) = .empty,

    /// Adds a frame. The caller has reserved room: two frames a byte at most.
    pub fn put(self: *Machine, frame: Frame) u32 {
        self.frames.appendAssumeCapacity(frame);
        return @intCast(self.frames.items.len - 1);
    }

    fn blank(self: *Machine, frame: Frame) u32 {
        if (frame.blanks >= self.schema.blanks) return rejected;
        var more = frame;
        more.blanks += 1;
        return self.put(more);
    }

    fn isBlank(byte: u8) bool {
        return byte == ' ' or byte == '\n' or byte == '\r' or byte == '\t';
    }

    /// The position after `byte` at position `from`: a frame, `done`, `outside` (the byte closed the frame the
    /// walk started in), `rejected`, or `unknown` (the byte is for a frame below the one the walk started in).
    pub fn step(self: *Machine, from: u32, byte: u8) u32 {
        var state = from;
        while (true) {
            if (state == done) return rejected;
            if (state == outside) return unknown;
            const frame = self.frames.items[state];
            const node = self.schema.nodes[frame.node];
            const digit = byte >= '0' and byte <= '9';
            switch (frame.phase) {
                .start => return self.enter(frame.parent, frame.node, byte),

                .body => switch (byte) {
                    '"' => return frame.parent,
                    '\\' => return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .escape }),
                    0...0x1f => return rejected,
                    0x20...0x21, 0x23...0x5b, 0x5d...0x7f => return state,
                    // well-formed UTF-8 (Unicode's table 3-7): no overlong form, no surrogate, nothing past U+10FFFF
                    0xc2...0xdf => return self.utf8(frame, 1, 0x80, 0xbf),
                    0xe0 => return self.utf8(frame, 2, 0xa0, 0xbf),
                    0xe1...0xec, 0xee...0xef => return self.utf8(frame, 2, 0x80, 0xbf),
                    0xed => return self.utf8(frame, 2, 0x80, 0x9f),
                    0xf0 => return self.utf8(frame, 3, 0x90, 0xbf),
                    0xf1...0xf3 => return self.utf8(frame, 3, 0x80, 0xbf),
                    0xf4 => return self.utf8(frame, 3, 0x80, 0x8f),
                    else => return rejected,
                },
                .utf8 => {
                    if (byte < (frame.b & 0xff) or byte > (frame.b >> 8)) return rejected;
                    if (frame.a == 1) return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .body });
                    return self.utf8(frame, frame.a - 1, 0x80, 0xbf);
                },
                .escape => switch (byte) {
                    '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .body }),
                    'u' => return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .hex, .a = 4, .b = hex_first }),
                    else => return rejected,
                },
                .hex => {
                    if (!std.ascii.isHex(byte)) return rejected;
                    const lower = byte | 0x20;
                    const kind: u32 = switch (frame.b) {
                        hex_first => if (lower == 'd') hex_d else hex_plain,
                        hex_d => switch (lower) {
                            '8', '9', 'a', 'b' => hex_high,
                            // a low surrogate with no high one before it
                            'c'...'f' => return rejected,
                            else => hex_plain,
                        },
                        hex_low_first => if (lower == 'd') hex_low_d else return rejected,
                        hex_low_d => switch (lower) {
                            'c'...'f' => hex_plain,
                            else => return rejected,
                        },
                        else => frame.b,
                    };
                    if (frame.a == 1) return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = if (kind == hex_high) .pair else .body });
                    return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .hex, .a = frame.a - 1, .b = kind });
                },
                .pair => return if (byte == '\\') self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .pair_u }) else rejected,
                .pair_u => return if (byte == 'u') self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .hex, .a = 4, .b = hex_low_first }) else rejected,

                .minus => {
                    if (!digit) return rejected;
                    // an integer has no `-0` (xgrammar's `basic_integer`); a number has
                    if (byte == '0' and node.kind == .integer) return rejected;
                    return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = if (byte == '0') .zero else .digits });
                },
                .zero, .digits => {
                    if (digit and frame.phase == .digits) return state;
                    if (node.kind == .number and byte == '.') return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .point });
                    if (node.kind == .number and (byte == 'e' or byte == 'E')) return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .exponent });
                    state = frame.parent;
                },
                .point => return if (digit) self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .fraction }) else rejected,
                .fraction => {
                    if (digit) return state;
                    if (byte == 'e' or byte == 'E') return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .exponent });
                    state = frame.parent;
                },
                .exponent => {
                    if (digit) return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .exponent_digits });
                    if (byte == '+' or byte == '-') return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .exponent_sign });
                    return rejected;
                },
                .exponent_sign => return if (digit) self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .exponent_digits }) else rejected,
                .exponent_digits => {
                    if (digit) return state;
                    state = frame.parent;
                },

                .word => {
                    if (self.schema.child(frame.a, byte)) |under| return self.word(frame.parent, frame.node, under);
                    if (self.schema.tries[frame.a].value == none) return rejected;
                    state = frame.parent;
                },

                .first, .next => {
                    if (isBlank(byte)) return self.blank(frame);
                    if (byte == '}') {
                        if (frame.phase == .next or self.schema.owes(node, frame.a)) return rejected;
                        return frame.parent;
                    }
                    if (byte != '"') return rejected;
                    if (node.count == 0) {
                        if (node.other == none) return rejected;
                        const colon = self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .colon, .b = none });
                        return self.put(.{ .parent = colon, .node = fixed.string, .phase = .body });
                    }
                    if (frame.a >= node.count) return rejected;
                    const keys = self.schema.properties[node.first + frame.a].keys;
                    return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .key, .a = frame.a, .b = self.schema.child(keys, byte).? });
                },
                .key => {
                    const under = self.schema.child(frame.b, byte) orelse return rejected;
                    const property = self.schema.tries[under].value;
                    if (property == none) return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .key, .a = frame.a, .b = under });
                    return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .colon, .a = property + 1, .b = property });
                },
                .colon => {
                    if (isBlank(byte)) return self.blank(frame);
                    if (byte != ':') return rejected;
                    return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .value, .a = frame.a, .b = frame.b });
                },
                .value => {
                    if (isBlank(byte)) return self.blank(frame);
                    const under = if (frame.b == none) node.other else self.schema.properties[node.first + frame.b].node;
                    if (!self.schema.begins(under, byte)) return rejected;
                    const after = self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .after, .a = frame.a });
                    return self.enter(after, under, byte);
                },
                .after => {
                    if (isBlank(byte)) return self.blank(frame);
                    if (byte == ',') {
                        if (if (node.count == 0) node.other == none else frame.a >= node.count) return rejected;
                        return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .next, .a = frame.a });
                    }
                    if (byte == '}' and !self.schema.owes(node, frame.a)) return frame.parent;
                    return rejected;
                },

                .item_first, .item_next => {
                    if (isBlank(byte)) return self.blank(frame);
                    if (byte == ']' and frame.phase == .item_first) return if (node.count == 0) frame.parent else rejected;
                    if (frame.a >= node.other or !self.schema.begins(node.first, byte)) return rejected;
                    // the count stops where the bounds stop asking, so an unbounded array stays one position
                    const counted = if (node.other == none) @min(frame.a + 1, node.count) else frame.a + 1;
                    const after = self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .item_after, .a = counted });
                    return self.enter(after, node.first, byte);
                },
                .item_after => {
                    if (isBlank(byte)) return self.blank(frame);
                    if (byte == ',') {
                        if (frame.a >= node.other) return rejected;
                        return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .item_next, .a = frame.a });
                    }
                    if (byte == ']' and frame.a >= node.count) return frame.parent;
                    return rejected;
                },
            }
        }
    }

    fn utf8(self: *Machine, frame: Frame, left: u32, low: u32, high: u32) u32 {
        return self.put(.{ .parent = frame.parent, .node = frame.node, .phase = .utf8, .a = left, .b = low | high << 8 });
    }

    fn word(self: *Machine, parent: u32, node: u32, at: u32) u32 {
        const trie = self.schema.tries[at];
        if (trie.fan == 0) return parent;
        return self.put(.{ .parent = parent, .node = node, .phase = .word, .a = at });
    }

    /// The position after `byte` as the first byte of a value of `node` inside `parent`.
    fn enter(self: *Machine, parent: u32, node: u32, byte: u8) u32 {
        const n = self.schema.nodes[node];
        switch (n.kind) {
            .any => return switch (byte) {
                '"' => self.enter(parent, fixed.string, byte),
                '-', '0'...'9' => self.enter(parent, fixed.number, byte),
                't', 'f', 'n' => self.enter(parent, fixed.words, byte),
                '[' => self.enter(parent, fixed.array, byte),
                '{' => self.enter(parent, fixed.object, byte),
                else => rejected,
            },
            .string => return if (byte == '"') self.put(.{ .parent = parent, .node = node, .phase = .body }) else rejected,
            .integer, .number => return switch (byte) {
                '-' => self.put(.{ .parent = parent, .node = node, .phase = .minus }),
                '0' => self.put(.{ .parent = parent, .node = node, .phase = .zero }),
                '1'...'9' => self.put(.{ .parent = parent, .node = node, .phase = .digits }),
                else => rejected,
            },
            .literals => return self.word(parent, node, self.schema.child(n.first, byte) orelse return rejected),
            .object => return if (byte == '{') self.put(.{ .parent = parent, .node = node, .phase = .first }) else rejected,
            .array => return if (byte == '[') self.put(.{ .parent = parent, .node = node, .phase = .item_first }) else rejected,
            .choice => {
                for (self.schema.alternatives[n.first..][0..n.count]) |alternative| {
                    if (self.schema.begins(alternative, byte)) return self.enter(parent, alternative, byte);
                }
                return rejected;
            },
            .ref => unreachable,
        }
    }

    /// Whether the value of the frame at `state` may end before the next byte: a number, a literal that
    /// another goes on from.
    pub fn mayEnd(self: *const Machine, state: u32) bool {
        const frame = self.frames.items[state];
        return switch (frame.phase) {
            .zero, .digits, .fraction, .exponent_digits => true,
            .word => self.schema.tries[frame.a].value != none,
            else => false,
        };
    }

    /// The position after `bytes` from `from`, the frames it made left in place; `rejected` leaves none.
    pub fn walk(self: *Machine, from: u32, bytes: []const u8) u32 {
        const mark = self.frames.items.len;
        var state = from;
        for (bytes) |byte| {
            state = self.step(state, byte);
            if (state == rejected) {
                self.frames.shrinkRetainingCapacity(mark);
                return rejected;
            }
        }
        return state;
    }
};
