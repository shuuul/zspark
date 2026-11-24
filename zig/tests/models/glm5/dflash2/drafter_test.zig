//! `drafter.zig`'s tests on a random GLM and a random drafter beside it: replies drafted by DFlash2 are the
//! serial replies; the context a reply leaves, its rows kept several a round and through the ring, proposes
//! what a context fed row by row proposes; and a constraint's forced tokens and masks in a chain.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const cuda = @import("../../../../src/cuda/cuda.zig");
const masks_file = @import("../../../../src/decode/masks.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const draft_file = @import("../../../../src/models/glm5/draft.zig");
const loopback = @import("../../../../src/models/glm5/layers/loopback.zig");
const glm5_testing = @import("../testing.zig");
const config_file = @import("../../../../src/decode/dflash2/config.zig");
const drafter_file = @import("../../../../src/models/glm5/dflash2/drafter.zig");
const drafter_testing = @import("../../../../src/models/glm5/checks/random_drafter.zig");
const pass = @import("../../../../src/models/glm5/dflash2/pass.zig");
const dflash2_pass = @import("../../../../src/decode/dflash2/pass.zig");
const weights = @import("../../../../src/decode/dflash2/weights.zig");

const world = config_file.world;
const fx = @import("../../../../src/models/glm5/checks/random.zig").fx;
const block = 8;

/// A random GLM in loopback, tapped, and a random drafter of its shapes.
const Both = struct {
    fixture: glm5_testing.Fixture,
    loop: glm5_testing.Loop,
    c: config_file.Config,
    w: [world]weights.Weights,
    books: weights.Codebooks,
    capacity: usize,

    fn init(self: *Both, total: usize, capacity: usize) !void {
        try self.fixture.init(467, total);
        errdefer self.fixture.deinit();
        const arena = self.fixture.arena_state.allocator();
        self.capacity = capacity;
        self.c = try drafter_testing.config(arena, fx.dims, fx.vocab, 5, &.{ 1, 3 });
        self.c.block = block;
        var source: drafter_testing.Source = .empty;
        var prng: std.Random.DefaultPrng = .init(468);
        try drafter_testing.put(arena, prng.random(), self.c, &source);
        self.loop = try self.fixture.loop(capacity);
        errdefer self.loop.deinit();
        try self.loop.b.setTaps(self.c.taps);
        const rows: cuda.Scratch.Rows = .{ .pool = &self.fixture.pool, .count = dflash2_pass.batch };
        self.w[0] = try .load(arena, testing.allocator, source, &self.c, 0, .mixed, rows);
        errdefer self.w[0].deinit();
        self.w[1] = try .load(arena, testing.allocator, source, &self.c, 1, .mixed, rows);
        errdefer self.w[1].deinit();
        self.books = try .load(testing.allocator, source, &self.c);
        // random codebooks give an edge too small to move a pick: larger, so that the chain's edges are held
        for (self.books.predecessor) |*value| value.* *= 12;
    }

    fn deinit(self: *Both) void {
        self.books.deinit(testing.allocator);
        for (&self.w) |*one| one.deinit();
        self.loop.deinit();
        self.fixture.deinit();
    }

    fn pair(self: *Both, ring: bool) !pass.Pair {
        return .init(testing.allocator, self.w, &self.c, self.capacity, ring, fx.vocab / world);
    }

    fn taps(self: *Both) [world]cuda.Buffer {
        return .{ self.loop.b.ranks[0].taps.?, self.loop.b.ranks[1].taps.? };
    }
};

/// What a block pass left on both ranks: the logits, the projected rows and the block's rows.
fn left(arena: Allocator, pair: pass.Pair) ![]u16 {
    var all: std.ArrayList(u16) = .empty;
    for (pair.ranks) |rank| for ([_]cuda.Buffer{ rank.logits, rank.selected, rank.x }) |buffer| try all.appendSlice(arena, try glm5_testing.fetch(u16, arena, buffer));
    return all.items;
}

test "replies drafted by DFlash2 are the serial replies: greedy and sampled, at every cap, at any confidence and the noise-aware one, graphed or not, flat and in the ring" {
    const gpa = testing.allocator;
    const prompt_len = 21;
    const count = 40;
    var both: Both = undefined;
    try both.init(prompt_len, prompt_len + count + 1);
    defer both.deinit();
    const prompt = both.fixture.tokens;
    const none: []const u32 = &.{};

    for ([_]sampling.Sampling{ .{ .seed = 3, .temperature = 0 }, .{ .seed = 5, .temperature = 0.8 } }) |settings| {
        const serial = try glm5_testing.serialReply(gpa, both.loop.m, fx.vocab, prompt, count, settings);
        defer gpa.free(serial);
        var graphs: loopback.Graphs = try .init(gpa);
        defer graphs.deinit();
        var drafted: usize = 0;
        for (1..block) |cap| for ([_]bool{ false, true }) |ring| {
            var pair = try both.pair(ring);
            defer pair.deinit();
            // the drafter's own passes as graphs, at every other cap
            var passes: dflash2_pass.Graphs = try .init(gpa);
            defer passes.deinit();
            if (cap % 2 == 1) pair.graphs = &passes;
            var d: drafter_file.Drafter = try .init(gpa, &both.c, &pair, .{ .m = both.loop.m, .main = &both.loop.b }, both.books, settings);
            defer d.deinit();
            d.cap = cap;
            d.confidence = if (cap % 2 == 0) 0 else drafter_file.Drafter.default_confidence;
            // the kit's noise-aware product, at the caps of 5 and more: only where a chain stops is another
            d.noisy = cap >= 5;
            const got = try glm5_testing.reply(gpa, &both.loop, prompt, count, settings, none, d.drafter(), if (cap % 3 == 0) &graphs else null);
            defer gpa.free(got.tokens);
            try testing.expectEqualSlices(i32, serial, got.tokens);
            // every round but the last, which has no token left to draft, was drafted for; without a confidence to the cap
            try testing.expect(d.passes + 1 >= got.stats.rounds and got.stats.drafted + 1 >= got.stats.rounds);
            if (d.confidence == 0) try testing.expect(got.stats.drafted > (got.stats.rounds - 2) * @min(cap, 2));
            drafted += got.stats.drafted;
        };
        try testing.expect(drafted > 0);
    }
}

/// DFlash2 asked every round, so that it absorbs the kept rows, and its drafts replaced every other round by
/// the reply's own: rounds then keep several rows.
const Mixed = struct {
    inner: draft_file.Drafter,
    reply: []const i32,
    first: usize,
    calls: usize = 0,

    fn drafter(self: *Mixed) draft_file.Drafter {
        return .{ .context = self, .draft = draft };
    }

    fn draft(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Mixed = @ptrCast(@alignCast(context));
        const own = try self.inner.draft(self.inner.context, streams, tokens, position, guide, out);
        if (position < self.first) return own;
        self.calls += 1;
        if (self.calls % 2 == 0) return own;
        var n: usize = 0;
        while (n < @min(out.len, 1 + self.calls % 6) and position + 1 + n - self.first < self.reply.len) : (n += 1) out[n] = self.reply[position + 1 + n - self.first];
        return n;
    }
};

test "DFlash2: the context a drafted reply leaves, its rows kept several a round and through the ring's wrap, proposes what a context fed row by row proposes" {
    // held to serial steps: one form, a step's (`testing.oneForm`)
    const forms = glm5_testing.oneForm();
    defer forms.restore();
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const prompt_len = 150;
    const count = 90;
    var both: Both = undefined;
    try both.init(prompt_len, prompt_len + count + 1);
    defer both.deinit();
    const prompt = both.fixture.tokens;
    const settings: sampling.Sampling = .{ .seed = 9, .temperature = 0 };
    const serial = try glm5_testing.serialReply(gpa, both.loop.m, fx.vocab, prompt, count, settings);
    defer gpa.free(serial);

    var hot = try both.pair(true);
    defer hot.deinit();
    try testing.expect(hot.ranks[0].ring and prompt_len + count > hot.ranks[0].cap + 16);
    // its updates as graphs, one a row count: the rounds keep 1 to 7 rows from the window's first
    var passes: dflash2_pass.Graphs = try .init(gpa);
    defer passes.deinit();
    hot.graphs = &passes;
    var d: drafter_file.Drafter = try .init(gpa, &both.c, &hot, .{ .m = both.loop.m, .main = &both.loop.b }, both.books, settings);
    defer d.deinit();
    var mixed: Mixed = .{ .inner = d.drafter(), .reply = serial, .first = prompt_len };
    const got = try glm5_testing.reply(gpa, &both.loop, prompt, count, settings, &.{}, mixed.drafter(), null);
    defer gpa.free(got.tokens);
    try testing.expectEqualSlices(i32, serial, got.tokens);
    try testing.expect(got.stats.accepted > count / 3 and got.stats.rounds < count - count / 3);
    try testing.expect(passes.updates.count() >= 6);

    // the sequence, and the same rows one by one into a flat context
    const end = hot.context_end;
    try testing.expect(end > prompt_len + count - fx.max_rows and end < prompt_len + count);
    const sequence = try arena.alloc(i32, prompt_len + count);
    @memcpy(sequence[0..prompt_len], prompt);
    @memcpy(sequence[prompt_len..], serial);
    var cold = try both.pair(false);
    defer cold.deinit();
    try both.loop.state.reset();
    for (sequence[0..end]) |token| {
        try loopback.forward(both.loop.m, &both.loop.state, &both.loop.b, &.{token});
        try loopback.commit(both.loop.m, &both.loop.state, both.loop.b, &.{token}, 1);
        try cold.addTaps(both.taps(), 0, 1);
    }
    try testing.expectEqual(end, cold.context_end);
    var main: [world]dflash2_pass.Main = undefined;
    for (&main, both.loop.m.ranks) |*one, rank| one.* = .{ .embed = rank.embed.view, .head = rank.head, .rows = rank.head.bf16.weight.view };
    try hot.block(main, sequence[end]);
    try cold.block(main, sequence[end]);
    try cuda.synchronize();
    try testing.expectEqualSlices(u16, try left(arena, cold), try left(arena, hot));
}

/// A constraint made up: the first draft among `mask`, the second forced, the third free, then no more.
const Made = struct {
    mask: []const u32,
    forced: i32,
    asked: usize = 0,

    fn guide(self: *Made) masks_file.Guide {
        return .{ .context = self, .nextFn = next };
    }

    fn next(context: *anyopaque, drafts: []const i32, more: bool) masks_file.Error!masks_file.Next {
        const self: *Made = @ptrCast(@alignCast(context));
        self.asked += 1;
        if (!more) return .closed;
        return switch (drafts.len) {
            0 => .{ .among = self.mask },
            1 => .{ .token = self.forced },
            2 => .free,
            else => .closed,
        };
    }
};

/// Every draft forced.
const Forced = struct {
    fn next(_: *anyopaque, drafts: []const i32, more: bool) masks_file.Error!masks_file.Next {
        if (!more or drafts.len == 3) return .closed;
        return .{ .token = @intCast(10 + drafts.len) };
    }
};

test "DFlash2: under a constraint a draft is among its mask, a forced token is the draft and takes a place of the block, and a chain of forced tokens runs no pass" {
    const gpa = testing.allocator;
    const rows = 12;
    var both: Both = undefined;
    try both.init(rows + 1, 64);
    defer both.deinit();
    var pair = try both.pair(false);
    defer pair.deinit();
    const settings: sampling.Sampling = .{ .seed = 1, .temperature = 0 };
    var d: drafter_file.Drafter = try .init(gpa, &both.c, &pair, .{ .m = both.loop.m, .main = &both.loop.b }, both.books, settings);
    defer d.deinit();
    d.confidence = 0;
    d.cap = block - 1;
    const tokens = both.fixture.tokens;
    const run = d.drafter();
    try both.loop.state.reset();
    try loopback.forward(both.loop.m, &both.loop.state, &both.loop.b, tokens[0..rows]);
    try loopback.commit(both.loop.m, &both.loop.state, both.loop.b, tokens[0..rows], rows);
    const streams = draft_file.namespace.streams(both.loop.b, rows);

    // unconstrained first, on a copy of the context: the candidates the masked pick is among
    var out: [block]i32 = undefined;
    const free = try run.draft(run.context, streams, tokens[1 .. rows + 1], rows, null, &out);
    try testing.expectEqual(@as(usize, block - 1), free);
    try testing.expectEqual(@as(usize, 1), d.passes);
    const first_free = out[0];

    // the same rows again from an empty context, under the constraint; the mask leaves the second candidate
    // of the first position and tokens that are no candidates
    d.reset(settings);
    d.passes = 0;
    // pick a candidate of position 0 that is not the unconstrained pick
    try pair.addTaps(both.taps(), 0, rows);
    const c = try d.candidates(tokens[rows], 3);
    var other: u32 = undefined;
    for (c.tokens[0..c.top]) |token| if (token != first_free) {
        other = token;
        break;
    };
    var mask: [fx.vocab / 32]u32 = @splat(0);
    mask[other / 32] |= @as(u32, 1) << @intCast(other % 32);
    d.reset(settings);
    d.passes = 0;
    var constraint: Made = .{ .mask = &mask, .forced = @mod(first_free + 1, fx.vocab) };
    const n = try run.draft(run.context, streams, tokens[1 .. rows + 1], rows, constraint.guide(), &out);
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqual(@as(i32, @intCast(other)), out[0]);
    try testing.expectEqual(constraint.forced, out[1]);
    try testing.expectEqual(@as(usize, 1), d.passes);
    // the third is the block's third position's pick after the forced token, not after the free chain's second
    const want = @import("../../../../src/decode/dflash2/chain.zig").pick(d.books, c, 2, @intCast(constraint.forced), rows + 1, null, null, false).?;
    try testing.expectEqual(@as(i32, @intCast(want.token)), out[2]);

    d.reset(settings);
    d.passes = 0;
    const all = try run.draft(run.context, streams, tokens[1 .. rows + 1], rows, .{ .context = &constraint, .nextFn = Forced.next }, &out);
    try testing.expectEqual(@as(usize, 3), all);
    try testing.expectEqualSlices(i32, &.{ 10, 11, 12 }, out[0..3]);
    try testing.expectEqual(@as(usize, 0), d.passes);
    try testing.expectEqual(@as(usize, rows), pair.context_end);
}

test "DFlash2: the drafter's drafts are its steps by hand: the kept rows' taps, a block after the newest token, each position's best of the whole vocabulary, the chain keyed by the drafts' places" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const chain = @import("../../../../src/decode/dflash2/chain.zig");
    const numeric = @import("../../../../src/weights/numeric.zig");
    const rows = 9;
    var both: Both = undefined;
    try both.init(rows + 1, 64);
    defer both.deinit();
    const tokens = both.fixture.tokens;
    try both.loop.state.reset();
    try loopback.forward(both.loop.m, &both.loop.state, &both.loop.b, tokens[0..rows]);
    try loopback.commit(both.loop.m, &both.loop.state, both.loop.b, tokens[0..rows], rows);
    const streams = draft_file.namespace.streams(both.loop.b, rows);
    const kept = tokens[1 .. rows + 1];
    const pending = kept[rows - 1];

    // by hand: the block, and a position's candidates by a sort of every token of both ranks
    var by_hand = try both.pair(false);
    defer by_hand.deinit();
    try by_hand.addTaps(both.taps(), 0, rows);
    var main: [world]dflash2_pass.Main = undefined;
    for (&main, both.loop.m.ranks) |*one, rank| one.* = .{ .embed = rank.embed.view, .head = rank.head, .rows = rank.head.bf16.weight.view };
    try by_hand.block(main, pending);
    try cuda.synchronize();
    const top = both.c.top_k;
    const half = fx.vocab / world;
    const want_tokens = try arena.alloc(u32, (block - 1) * top);
    const want_values = try arena.alloc(f64, (block - 1) * top);
    const logits: [world][]u16 = .{ try glm5_testing.fetch(u16, arena, by_hand.ranks[0].logits), try glm5_testing.fetch(u16, arena, by_hand.ranks[1].logits) };
    for (0..block - 1) |d| {
        var order: [fx.vocab]u32 = undefined;
        var row: [fx.vocab]f32 = undefined;
        for (&order, &row, 0..) |*one, *value, token| {
            one.* = @intCast(token);
            value.* = numeric.f32FromBf16(logits[token / half][d * half + token % half]);
        }
        const Context = struct {
            row: []const f32,
            fn before(self: @This(), a: u32, b: u32) bool {
                return self.row[a] > self.row[b] or (self.row[a] == self.row[b] and a < b);
            }
        };
        std.mem.sort(u32, &order, Context{ .row = &row }, Context.before);
        for (want_tokens[d * top ..][0..top], want_values[d * top ..][0..top], order[0..top]) |*token, *value, best| {
            token.* = best;
            value.* = row[best];
        }
    }
    const selected = try glm5_testing.fetch(u16, arena, by_hand.ranks[1].selected);
    const projected = try arena.alloc(f64, selected.len);
    for (projected, selected) |*wide, bits| wide.* = numeric.f32FromBf16(bits);
    const books: chain.Books = .{ .predecessor = both.books.predecessor, .successor = both.books.successor, .rank = both.books.rank };

    var pair = try both.pair(false);
    defer pair.deinit();
    var d: drafter_file.Drafter = try .init(gpa, &both.c, &pair, .{ .m = both.loop.m, .main = &both.loop.b }, both.books, .{ .seed = 0, .temperature = 0 });
    defer d.deinit();
    const run = d.drafter();
    var edged: usize = 0;
    var lengths: std.AutoHashMapUnmanaged(usize, void) = .empty;
    for ([_]sampling.Sampling{ .{ .seed = 21, .temperature = 0 }, .{ .seed = 22, .temperature = 1.0 }, .{ .seed = 23, .temperature = 0.6 } }) |settings| {
        for ([_]f64{ 0, 0.3, 0.9 }) |confidence| for ([_]usize{ 2, 5, block - 1 }) |cap| {
            d.reset(settings);
            d.confidence = confidence;
            d.cap = cap;
            var out: [block]i32 = undefined;
            const n = try run.draft(run.context, streams, kept, rows, null, &out);
            const c: chain.Candidates = .{ .tokens = want_tokens[0 .. cap * top], .values = want_values[0 .. cap * top], .projected = projected[0 .. cap * books.rank], .top = top };
            var want: [block]u32 = undefined;
            const noise: ?chain.Noise = if (settings.temperature > 0) .{ .seed = settings.seed, .temperature = settings.temperature } else null;
            const count = chain.choose(books, c, @intCast(pending), rows + 1, noise, confidence, d.noisy, &want);
            try testing.expectEqual(count, n);
            for (want[0..count], out[0..n]) |a, b| try testing.expectEqual(@as(i32, @intCast(a)), b);
            if (confidence == 0) try testing.expectEqual(cap, n);
            // and the edge is in it: without the codebooks some chain is another
            const flat = try arena.alloc(f32, books.predecessor.len);
            @memset(flat, 0);
            var plain: [block]u32 = undefined;
            const plain_count = chain.choose(.{ .predecessor = flat, .successor = flat, .rank = books.rank }, c, @intCast(pending), rows + 1, noise, confidence, d.noisy, &plain);
            edged += @intFromBool(plain_count != count or !std.mem.eql(u32, plain[0..count], want[0..count]));
            try lengths.put(arena, n, {});
            try testing.expectEqual(@as(usize, rows), pair.context_end);
        };
    }
    // the cases differ: a confidence cuts some chains, and a chain without the codebooks is another. The
    // noise is the hand chain's draw, matched above.
    try testing.expect(lengths.count() >= 4 and edged >= 7);

    // past the capacity nothing is absorbed and nothing drafted, from then on
    d.reset(.{ .seed = 0, .temperature = 0 });
    pair.capacity = rows - 1;
    var out: [block]i32 = undefined;
    try testing.expectEqual(@as(usize, 0), try run.draft(run.context, streams, kept, rows, null, &out));
    try testing.expect(d.full and pair.context_end == 0);
}
