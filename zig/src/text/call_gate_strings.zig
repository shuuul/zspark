//! `call_gate.zig`'s text a call's parser must not meet (spec 104 WS-06 moved it out of `call_gate.zig`): the
//! bytes of a value written inside a call as an automaton, and what each token does at its states (`Strings`).
//! Host only.

const std = @import("std");
const tokenizer_file = @import("tokenizer.zig");
const Allocator = std.mem.Allocator;
const Tokenizer = tokenizer_file.Tokenizer;
const template = @import("call_gate.zig").template;

/// The bytes of a value written inside a call, as an automaton: where a string written raw may go on, where it
/// ends, and what no value may hold. A state is what the text so far ends with.
///
/// A raw string is its bytes, then `\n</parameter>\n`. The bytes are UTF-8 and hold neither `</parameter>` nor
/// `</tool_call>` in any case (the parser looks for both without regard to case); the closer that ends the
/// value is the first, in lower case, right after a newline. A value written as JSON goes through the same
/// states and may reach neither closer at all.
pub const automaton = struct {
    pub const Tail = u8;
    /// In the value, after a byte that is not a newline; and after a newline.
    pub const plain: Tail = 0;
    const line: Tail = 1;
    /// Inside a character: one, two or three bytes to come, and the four leads whose next byte is narrower.
    const need1: Tail = 2;
    const need2: Tail = 3;
    const need3: Tail = 4;
    const lead_e0: Tail = 5;
    const lead_ed: Tail = 6;
    const lead_f0: Tail = 7;
    const lead_f4: Tail = 8;
    /// After the value's `\n</parameter>`: one newline to come. And after it: the value is over.
    const closed: Tail = 9;
    pub const done: Tail = 10;
    /// `partial + exact * 20 + place`: the text ends with the beginning of a closer. `place` 0 is `<`, 1 `</`,
    /// 2 to 10 `</p` to `</parameter`, 11 to 19 `</t` to `</tool_call`. `exact`: in lower case and after a
    /// newline, so that `>` would end the value.
    const partial: Tail = 11;
    pub const count = partial + 40;
    pub const dead: Tail = 255;

    const closer = "</parameter>";
    const envelope = "</tool_call>";

    fn step(state: Tail, byte: u8) Tail {
        switch (state) {
            done => return dead,
            closed => return if (byte == '\n') done else dead,
            need1 => return if (byte & 0xc0 == 0x80) plain else dead,
            need2 => return if (byte & 0xc0 == 0x80) need1 else dead,
            need3 => return if (byte & 0xc0 == 0x80) need2 else dead,
            lead_e0 => return if (byte >= 0xa0 and byte <= 0xbf) need1 else dead,
            lead_ed => return if (byte >= 0x80 and byte <= 0x9f) need1 else dead,
            lead_f0 => return if (byte >= 0x90 and byte <= 0xbf) need2 else dead,
            lead_f4 => return if (byte >= 0x80 and byte <= 0x8f) need2 else dead,
            else => {},
        }
        if (state >= partial) {
            const exact = state - partial >= 20;
            const place = (state - partial) % 20;
            // how much of a closer the text ends with, and of which
            const matched: usize = if (place <= 10) place + 1 else place - 8;
            const lower = std.ascii.toLower(byte);
            if (place <= 10 and closer[matched] == lower) {
                if (matched + 1 == closer.len) return if (exact) closed else dead;
                return partial + @as(Tail, if (exact and byte == lower) 20 else 0) + @as(Tail, @intCast(matched));
            }
            if ((place <= 1 or place >= 11) and envelope[matched] == lower) {
                if (matched + 1 == envelope.len) return dead;
                return partial + @as(Tail, @intCast(matched + 9));
            }
        }
        // no closer goes on: the byte begins one, or is the value's
        return switch (byte) {
            '<' => partial + @as(Tail, if (state == line) 20 else 0),
            '\n' => line,
            0...'\n' - 1, '\n' + 1...'<' - 1, '<' + 1...0x7f => plain,
            0x80...0xc1, 0xf5...0xff => dead,
            0xc2...0xdf => need1,
            0xe0 => lead_e0,
            0xe1...0xec, 0xee, 0xef => need2,
            0xed => lead_ed,
            0xf0 => lead_f0,
            0xf1...0xf3 => need3,
            0xf4 => lead_f4,
        };
    }

    const table: [count][256]Tail = made: {
        @setEvalBranchQuota(1 << 21);
        var out: [count][256]Tail = undefined;
        for (&out, 0..) |*row, state| {
            for (row, 0..) |*next, byte| next.* = step(@intCast(state), @intCast(byte));
        }
        break :made out;
    };

    /// The state after `bytes` from `from`; `dead` where they may not come.
    pub fn walk(from: Tail, bytes: []const u8) Tail {
        var state = from;
        for (bytes) |byte| {
            if (state == dead) return dead;
            state = table[state][byte];
        }
        return state;
    }

    /// Whether a value written as JSON may stand at `state`: it has reached no closer.
    pub fn inside(state: Tail) bool {
        return state != dead and state != closed and state != done;
    }

    /// Whether a string can be written raw: it holds no closer, and is UTF-8.
    pub fn writable(value: []const u8) bool {
        return walk(walk(plain, value), template.parameter_close) == done;
    }
};

/// What each token does at each of `automaton`'s states, as masks made when first asked for: a raw string's tokens,
/// and the tokens a JSON value may take. Made for one tokenizer; not to be asked from two threads.
pub const Strings = struct {
    gpa: Allocator,
    tokenizer: *const Tokenizer,
    words: usize,
    stops: []u32,
    /// `[2][automaton.count]`, each `words` long or empty until made: a raw string's, then a JSON value's.
    masks: [2][automaton.count][]u32 = @splat(@splat(&.{})),

    pub fn init(gpa: Allocator, tokenizer: *const Tokenizer, words: usize, stops: []const u32) Allocator.Error!Strings {
        return .{ .gpa = gpa, .tokenizer = tokenizer, .words = words, .stops = try gpa.dupe(u32, stops) };
    }

    pub fn deinit(self: *Strings) void {
        for (&self.masks) |*family| {
            for (family) |made| self.gpa.free(made);
        }
        self.gpa.free(self.stops);
        self.* = undefined;
    }

    /// The state after `token` in a raw string at `state`, or null where it may not come: a token without
    /// bytes, a special one, one that ends a reply, bytes the value may not hold or that go past its end.
    pub fn after(self: *const Strings, state: automaton.Tail, token: u32) ?automaton.Tail {
        if (token / 32 >= self.words or self.tokenizer.isSpecial(token)) return null;
        if (std.mem.indexOfScalar(u32, self.stops, token) != null) return null;
        const bytes = self.tokenizer.tokenBytes(token);
        if (bytes.len == 0) return null;
        const next = automaton.walk(state, bytes);
        return if (next == automaton.dead) null else next;
    }

    /// The tokens a raw string takes at `state`; with `json`, the tokens a JSON value may take there.
    pub fn mask(self: *Strings, state: automaton.Tail, json: bool) Allocator.Error![]const u32 {
        const kept = &self.masks[@intFromBool(json)][state];
        if (kept.len != 0) return kept.*;
        const made = try self.gpa.alloc(u32, self.words);
        @memset(made, 0);
        for (0..@min(self.tokenizer.count(), self.words * 32)) |index| {
            const token: u32 = @intCast(index);
            const next = self.after(state, token) orelse continue;
            if (json and !automaton.inside(next)) continue;
            made[token / 32] |= @as(u32, 1) << @intCast(token % 32);
        }
        kept.* = made;
        return made;
    }
};
