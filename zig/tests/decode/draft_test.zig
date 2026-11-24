//! `draft.zig`'s tests on the host: `Decoder` over a model that is not Flash Next and has no device. Its state
//! is a hash of the committed tokens and a row's logits are made from the hash after that row, so it keeps
//! the two rules `Decoder` asks of a model by construction, and the tests hold the decoder's own steps (its
//! windows, what it keeps, what it hands a drafter) to one-row steps written without it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../src/cuda/cuda.zig");
const draft = @import("../../src/decode/draft.zig");
const numeric = @import("../../src/weights/numeric.zig");
const sampling = @import("../../src/decode/sampling.zig");

/// The model: nothing of it is on a device, and `cuda.zig` is read for the type `View` alone.
const toy = struct {
    pub const Model = struct { max_rows: usize, vocab: usize, salt: u64 };

    pub const State = struct { pos: usize = 0, hash: u64 = 0 };

    /// Values of a row's streams: the hash after the row, 16 bits a value.
    pub const width = 4;

    pub const Buffers = struct {
        rows: usize,
        scored: usize,
        vocab: usize,
        /// [scored, vocab] bf16 bits.
        logits: []u16,
        /// The hash after each row of the window, and the same as [rows, width] for a drafter.
        hashes: []u64,
        h: []u16,
        /// Rows a forward scored, over every forward.
        scored_rows: usize = 0,

        pub fn init(gpa: Allocator, m: Model, rows: usize, scored: usize) !Buffers {
            const logits = try gpa.alloc(u16, scored * m.vocab);
            errdefer gpa.free(logits);
            const hashes = try gpa.alloc(u64, rows);
            errdefer gpa.free(hashes);
            return .{ .rows = rows, .scored = scored, .vocab = m.vocab, .logits = logits, .hashes = hashes, .h = try gpa.alloc(u16, rows * width) };
        }

        pub fn deinit(self: *Buffers, gpa: Allocator) void {
            gpa.free(self.h);
            gpa.free(self.hashes);
            gpa.free(self.logits);
        }

        pub fn scoredRows(self: Buffers) usize {
            return self.scored;
        }
    };

    pub const Error = error{RowOutOfRange};
    pub const Score = enum { all, last, none };

    fn mix(hash: u64, token: i32, salt: u64) u64 {
        var x = hash ^ (@as(u64, @as(u32, @bitCast(token))) +% salt) *% 0x9e3779b97f4a7c15;
        x ^= x >> 29;
        x *%= 0xbf58476d1ce4e5b9;
        return x ^ (x >> 32);
    }

    /// A row's logits from the hash after it: distinct finite values, so a choice has no tie.
    fn score(m: Model, hash: u64, out: []u16) void {
        var random: std.Random.DefaultPrng = .init(hash);
        for (out, 0..) |*bits, token| {
            const value = random.random().float(f32) * 8.0 - 4.0 + @as(f32, @floatFromInt(token)) * 1e-3;
            bits.* = numeric.bf16FromF32(value);
        }
        _ = m;
    }

    pub fn forwardScoring(m: Model, state: *State, b: *Buffers, tokens: []const i32, which: Score) Error!void {
        if (tokens.len > b.rows or tokens.len > m.max_rows) return error.RowOutOfRange;
        if (which == .all and tokens.len > b.scored) return error.RowOutOfRange;
        // what a window leaves is its own alone: rows no forward scored read as NaN
        @memset(b.logits, 0x7fc0);
        var hash = state.hash;
        for (tokens, 0..) |token, row| {
            hash = mix(hash, token, m.salt);
            b.hashes[row] = hash;
            for (b.h[row * width ..][0..width], 0..) |*value, part| value.* = @truncate(hash >> @intCast(16 * part));
            switch (which) {
                .all => score(m, hash, b.logits[row * m.vocab ..][0..m.vocab]),
                .last => if (row + 1 == tokens.len) score(m, hash, b.logits[0..m.vocab]),
                .none => {},
            }
            b.scored_rows += @intFromBool(which == .all or (which == .last and row + 1 == tokens.len));
        }
    }

    pub fn forward(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
        return forwardScoring(m, state, b, tokens, .all);
    }

    pub fn commit(m: Model, state: *State, b: Buffers, tokens: []const i32, keep: usize) Error!void {
        _ = m;
        std.debug.assert(keep >= 1 and keep <= tokens.len);
        state.hash = b.hashes[keep - 1];
        state.pos += keep;
    }

    pub fn logitsRow(b: Buffers, row: usize, out: []u16) Error!void {
        if (row >= b.scored) return error.RowOutOfRange;
        @memcpy(out, b.logits[row * b.vocab ..][0..out.len]);
    }

    pub fn streams(b: Buffers, rows: usize) cuda.View {
        return .init(@ptrCast(b.h.ptr), .bf16, &.{ rows, width });
    }

    /// The forward by another launch: the windows it has run before are the ones it holds.
    pub const Graphs = struct {
        captured: [draft.forward_max_rows + 1]bool = @splat(false),
        launches: usize = 0,

        pub fn forward(self: *Graphs, m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
            self.captured[tokens.len] = true;
            self.launches += 1;
            return toy.forward(m, state, b, tokens);
        }

        pub fn holds(self: *const Graphs, state: State, b: Buffers, rows: usize) bool {
            _ = state;
            return rows <= b.scored and rows < self.captured.len and self.captured[rows];
        }
    };
};

const Decoder = draft.Decoder(toy);

/// `count` tokens after `prompt` by one-row steps, written without the decoder; `hashes[i]` is the state after
/// position i of the whole sequence.
fn serial(gpa: Allocator, m: toy.Model, prompt: []const i32, count: usize, settings: sampling.Sampling, hashes: []u64) ![]i32 {
    var state: toy.State = .{};
    var b: toy.Buffers = try .init(gpa, m, 1, 1);
    defer b.deinit(gpa);
    var sampler: sampling.Sampler = try .init(gpa, m.vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, m.vocab);
    defer gpa.free(logits);
    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    for (prompt) |token| {
        try toy.forward(m, &state, &b, &.{token});
        try toy.commit(m, &state, b, &.{token}, 1);
        hashes[state.pos - 1] = state.hash;
    }
    for (out, 0..) |*token, made| {
        try toy.logitsRow(b, 0, logits);
        token.* = try sampler.choose(logits, prompt.len + made, settings);
        try toy.forward(m, &state, &b, &.{token.*});
        try toy.commit(m, &state, b, &.{token.*}, 1);
        hashes[state.pos - 1] = state.hash;
    }
    return out;
}

/// A drafter that knows the reply: `count` drafts a call, the one at a place `wrong` names spoiled. It holds
/// what it is handed to the serial steps: the rows' streams are the states after those rows.
const Oracle = struct {
    reply: []const i32,
    first: usize,
    count: usize,
    wrong: usize,
    hashes: []const u64,
    calls: usize = 0,
    rows: usize = 0,
    faults: usize = 0,

    fn propose(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?@import("../../src/decode/masks.zig").Guide, out: []i32) anyerror!usize {
        const self: *Oracle = @ptrCast(@alignCast(context));
        std.debug.assert(guide == null);
        self.calls += 1;
        self.rows += tokens.len;
        const h: [*]const u16 = @ptrCast(@alignCast(streams.ptr.?));
        if (streams.size[0] != tokens.len) self.faults += 1;
        // row i stands at `position - tokens.len + i`
        for (0..tokens.len) |row| {
            const at = position - tokens.len + row;
            for (0..toy.width) |part| {
                if (h[row * toy.width + part] != @as(u16, @truncate(self.hashes[at] >> @intCast(16 * part)))) self.faults += 1;
            }
        }
        var made: usize = 0;
        while (made < @min(self.count, out.len)) : (made += 1) {
            const at = position + 1 + made;
            if (at < self.first or at - self.first >= self.reply.len) break;
            const right = self.reply[at - self.first];
            out[made] = if (self.wrong > 0 and at % self.wrong == 0) right ^ 1 else right;
        }
        return made;
    }

    fn drafter(self: *Oracle) draft.Drafter {
        return .{ .context = self, .draft = propose };
    }
};

/// The windows a prompt ran in, as `Decoder.Within` is told them.
const Windows = struct {
    rows: [64]usize = undefined,
    len: usize = 0,
    lasts: usize = 0,

    fn window(context: *anyopaque, run: []const i32, last: bool) Decoder.Error!void {
        const self: *Windows = @ptrCast(@alignCast(context));
        self.rows[self.len] = run.len;
        self.len += 1;
        self.lasts += @intFromBool(last);
    }
};

const Setup = struct { rows: usize, scored: usize, depth: usize, wrong: usize, graphed: bool };

/// The reply to `prompt` by the decoder: the prompt in windows, then rounds with the oracle's drafts.
fn drafted(gpa: Allocator, m: toy.Model, prompt: []const i32, expected: []const i32, hashes: []const u64, settings: sampling.Sampling, setup: Setup) !draft.Stats {
    var state: toy.State = .{};
    var b: toy.Buffers = try .init(gpa, m, setup.rows, setup.scored);
    defer b.deinit(gpa);
    var graphs: toy.Graphs = .{};
    var sampler: sampling.Sampler = try .init(gpa, m.vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, m.vocab);
    defer gpa.free(logits);
    var windows: Windows = .{};
    var decoder: Decoder = .{
        .m = m,
        .state = &state,
        .b = &b,
        .graphs = if (setup.graphed) &graphs else null,
        .sampler = &sampler,
        .settings = settings,
        .eos = &.{},
        .logits = logits,
        .within = .{ .context = &windows, .window = Windows.window },
    };
    try testing.expectEqual(@min(setup.scored, m.max_rows, draft.forward_max_rows) - 1, decoder.depth());
    var oracle: Oracle = .{ .reply = expected, .first = prompt.len, .count = setup.depth, .wrong = setup.wrong, .hashes = hashes };

    const out = try gpa.alloc(i32, expected.len);
    defer gpa.free(out);
    var chosen: [draft.forward_max_rows]i32 = undefined;
    var drafts: [draft.forward_max_rows]i32 = undefined;
    // the windows the prompt will run in, known before it runs
    var planned: [64]usize = undefined;
    var plans: usize = 0;
    var at: usize = 0;
    while (at < prompt.len) : (plans += 1) {
        planned[plans] = decoder.promptRows(at, prompt.len, chosen.len);
        at += planned[plans];
    }
    const n = try decoder.prompt(oracle.drafter(), prompt, &chosen);
    try testing.expectEqualSlices(usize, planned[0..plans], windows.rows[0..windows.len]);
    try testing.expectEqual(@as(usize, 1), windows.lasts);
    try testing.expectEqual(prompt.len, state.pos);
    try testing.expectEqualSlices(i32, prompt[prompt.len - n + 1 ..], chosen[0 .. n - 1]);
    // every window before the last was absorbed, none scored but the last's last row
    try testing.expectEqual(plans - 1, oracle.calls);
    try testing.expectEqual(prompt.len - n, oracle.rows);
    try testing.expectEqual(@as(usize, 1), b.scored_rows);

    out[0] = chosen[n - 1];
    var made: usize = 1;
    var kept: []const i32 = chosen[0..n];
    var held: [draft.forward_max_rows]i32 = undefined;
    while (made < out.len) {
        @memcpy(held[0..kept.len], kept);
        const count = try decoder.draft(oracle.drafter(), held[0..kept.len], out.len - made - 1, &drafts);
        const keep = try decoder.round(out[made - 1], drafts[0..count], &chosen);
        @memcpy(out[made..][0..keep], chosen[0..keep]);
        made += keep;
        kept = chosen[0..keep];
    }
    try testing.expectEqualSlices(i32, expected, out);
    try testing.expectEqual(prompt.len + expected.len - 1, state.pos);
    try testing.expectEqual(hashes[state.pos - 1], state.hash);
    try testing.expectEqual(@as(usize, 0), oracle.faults);
    try testing.expectEqual(expected.len - 1, decoder.stats.rounds + decoder.stats.accepted);
    if (setup.depth == 0) try testing.expectEqual(expected.len - 1, decoder.stats.rounds);
    if (setup.depth > 0 and setup.wrong == 0) try testing.expectEqual(decoder.stats.drafted, decoder.stats.accepted);
    if (setup.depth > 0 and setup.wrong > 1) try testing.expect(decoder.stats.accepted > 0 and decoder.stats.accepted <= decoder.stats.drafted);
    if (setup.graphed) try testing.expectEqual(decoder.stats.rounds, graphs.launches - decoder.graphed) else try testing.expectEqual(@as(usize, 0), graphs.launches);
    return decoder.stats;
}

test "a decoder over a second model: drafted equals serial at every depth, greedy and sampled, whatever the drafts" {
    const gpa = testing.allocator;
    const m: toy.Model = .{ .max_rows = 8, .vocab = 97, .salt = 11 };
    var prompt: [21]i32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast((i * 37 + 5) % m.vocab);
    const count = 48;
    for ([_]sampling.Sampling{
        .{ .seed = 3, .temperature = 0.0 },
        .{ .seed = 3, .temperature = 0.8, .top_k = 20, .top_p = 0.9 },
    }) |settings| {
        const hashes = try gpa.alloc(u64, prompt.len + count);
        defer gpa.free(hashes);
        const expected = try serial(gpa, m, &prompt, count, settings, hashes);
        defer gpa.free(expected);
        // the reply is not one token over and over, or a spoiled draft could be right
        var distinct: usize = 0;
        for (expected[1..], expected[0 .. count - 1]) |a, b| distinct += @intFromBool(a != b);
        try testing.expect(distinct > count / 2);
        var dropped: usize = 0;
        for (0..m.max_rows) |depth| {
            // no draft wrong, every draft wrong, some
            for ([_]usize{ 0, 1, 3, 5 }) |wrong| {
                for ([_]bool{ false, true }) |graphed| {
                    const stats = try drafted(gpa, m, &prompt, expected, hashes, settings, .{ .rows = 8, .scored = 8, .depth = depth, .wrong = wrong, .graphed = graphed });
                    if (wrong > 1) dropped += stats.drafted - stats.accepted;
                    if (wrong == 1) try testing.expectEqual(@as(usize, 0), stats.accepted);
                }
            }
        }
        // some drafts were run and not kept: a round's dropped rows left nothing behind
        try testing.expect(dropped > 0);
    }
}

test "a decoder over a second model: a prompt's windows are promptRows', and a last window its graph holds is launched as it" {
    const gpa = testing.allocator;
    const m: toy.Model = .{ .max_rows = 16, .vocab = 61, .salt = 5 };
    const settings: sampling.Sampling = .{ .seed = 9, .temperature = 0.0 };
    var prompt: [40]i32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast((i * 13 + 2) % m.vocab);
    const count = 12;
    // buffers narrower and wider than the model's rows, more rows than are scored, a prompt of one window and
    // of many, of one token
    for ([_]usize{ 1, 2, 5, 16, 17, 33, 40 }) |len| {
        const hashes = try gpa.alloc(u64, len + count);
        defer gpa.free(hashes);
        const expected = try serial(gpa, m, prompt[0..len], count, settings, hashes);
        defer gpa.free(expected);
        for ([_][2]usize{ .{ 4, 4 }, .{ 16, 16 }, .{ 12, 3 }, .{ 32, 16 } }) |shape| {
            _ = try drafted(gpa, m, prompt[0..len], expected, hashes, settings, .{ .rows = shape[0], .scored = shape[1], .depth = 3, .wrong = 4, .graphed = false });
        }
    }

    // a prompt whose last window is as wide as a window run before on the graphs is launched as that graph, and
    // its last row's logits are read where the graph leaves them
    const len = 19;
    const hashes = try gpa.alloc(u64, len + count);
    defer gpa.free(hashes);
    const expected = try serial(gpa, m, prompt[0..len], count, settings, hashes);
    defer gpa.free(expected);
    var state: toy.State = .{};
    var b: toy.Buffers = try .init(gpa, m, 16, 16);
    defer b.deinit(gpa);
    var graphs: toy.Graphs = .{};
    graphs.captured[3] = true;
    var sampler: sampling.Sampler = try .init(gpa, m.vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, m.vocab);
    defer gpa.free(logits);
    var decoder: Decoder = .{ .m = m, .state = &state, .b = &b, .graphs = &graphs, .sampler = &sampler, .settings = settings, .eos = &.{}, .logits = logits };
    var chosen: [draft.forward_max_rows]i32 = undefined;
    try decoder.advance(null, prompt[0..len], len - 3);
    try testing.expectEqual(len - 3, state.pos);
    try testing.expectEqual(hashes[len - 4], state.hash);
    try testing.expectEqual(@as(usize, 3), try decoder.prompt(null, prompt[0..len], &chosen));
    try testing.expectEqual(@as(usize, 1), decoder.graphed);
    try testing.expectEqual(@as(usize, 1), graphs.launches);
    try testing.expectEqual(expected[0], chosen[2]);
    // a reply that ends: the row that chooses an end token is the last kept, whatever was drafted after it
    decoder.eos = &.{@intCast(expected[2])};
    const keep = try decoder.round(expected[0], expected[1..6], &chosen);
    const first_end = std.mem.indexOfScalar(i32, expected[1..], expected[2]).? + 1;
    try testing.expectEqual(first_end, keep);
    try testing.expectEqualSlices(i32, expected[1..][0..keep], chosen[0..keep]);
    try testing.expectEqual(len + keep, state.pos);
}
