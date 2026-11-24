//! Two-rank DFlash2 in one process (`Held(2)`): the repo's exact reference for NCCL. Drafted replies match
//! the two ranks' serial tokens, greedy and fixed-seed sampled, the target and the drafter's passes graphed
//! or not; rejection at first/middle/last depth; a captured block's bits, and a context taken, overwritten and
//! restored; EOS mid-draft; usage; a cancel mid-draft; seats beside others, each a lone pair's bits (spec 126
//! WS-01). Rank 1 does not propose: loopback runs both shares' ops in one process, so the mirror's flash
//! replay is `ranks-check ... drafter=`'s (N1), and its `flash_round` is not exercised here.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const cuda = @import("../../../../src/cuda/cuda.zig");
const draft = @import("../../../../src/decode/draft.zig");
const masks_file = @import("../../../../src/decode/masks.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const random = @import("../../../../src/models/qwen4/checks/random.zig");
const drafter_testing = @import("../../../../src/models/qwen4/checks/random_drafter.zig");
const config_file = @import("../../../../src/decode/dflash2/config.zig");
const drafter_file = @import("../../../../src/models/qwen4/dflash2/drafter.zig");
const pass = @import("../../../../src/models/qwen4/dflash2/pass.zig");
const dflash2_pass = @import("../../../../src/decode/dflash2/pass.zig");
const weights = @import("../../../../src/decode/dflash2/weights.zig");
const ranks = @import("../../../../src/models/qwen4/ranks_forward.zig");
const forward_file = @import("../../../../src/models/qwen4/forward.zig");
const taps_file = @import("../../../../src/models/qwen4/taps.zig");
const pair_file = @import("../../../../src/models/qwen4/session/pair.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const seats_file = @import("../../../../src/decode/dflash2/seats.zig");

const borrowed_fixture = @import("borrowed_head_fixture.zig");

const Pair = ranks.Held(2);
const Both = pair_file.Of(2);
const world = config_file.world;
const fx = random.fx;
const block = 8;
/// Wide enough that a round can take the drafter's block (`depth` is `max_rows - 1`).
const rows_cap = 16;

const ns = struct {
    pub const Model = Pair.Model;
    pub const State = Pair.State;
    pub const Buffers = Pair.Buffers;
    pub const Graphs = Pair.Graphs;
    pub const Error = Pair.Error;
    pub const forward = Pair.forward;
    pub const forwardScoring = Pair.forwardScoring;
    pub const commit = Pair.commit;
    pub const logitsRow = Pair.logitsRow;
    pub const streams = Pair.streams;
};

const Decoder = draft.Decoder(ns);

const HeldBoth = struct {
    arena_state: std.heap.ArenaAllocator,
    held: Pair.Model,
    c: config_file.Config,
    w: [world]weights.Weights,
    books: weights.Codebooks,
    capacity: usize,

    fn init(self: *HeldBoth, capacity: usize) !void {
        return self.initStored(capacity, .stored);
    }

    /// The drafter's linears as `storage` says (`weights.Storage`).
    fn initStored(self: *HeldBoth, capacity: usize, storage: weights.Storage) !void {
        self.arena_state = .init(testing.allocator);
        errdefer self.arena_state.deinit();
        const arena = self.arena_state.allocator();
        self.capacity = capacity;
        const built = try random.shares(2, testing.allocator, arena, 42, 0, rows_cap);
        self.held = .{ .ranks = built, .max_rows = rows_cap };
        errdefer self.held.deinit();
        self.c = try drafter_testing.config(arena, fx.shape.hidden, fx.shape.vocab, fx.layers, &.{ 0, 2 });
        self.c.block = block;
        var source: drafter_testing.Source = .empty;
        var prng: std.Random.DefaultPrng = .init(468);
        try drafter_testing.put(arena, prng.random(), self.c, &source);
        const rows: cuda.Scratch.Rows = .{ .pool = self.held.ranks[0].scratch, .count = dflash2_pass.wide };
        self.w[0] = try .load(arena, testing.allocator, source, &self.c, 0, storage, rows);
        errdefer self.w[0].deinit();
        self.w[1] = try .load(arena, testing.allocator, source, &self.c, 1, storage, rows);
        errdefer self.w[1].deinit();
        self.books = try .load(testing.allocator, source, &self.c);
        for (self.books.predecessor) |*value| value.* *= 12;
    }

    fn deinit(self: *HeldBoth) void {
        self.books.deinit(testing.allocator);
        for (&self.w) |*one| one.deinit();
        self.held.deinit();
        self.arena_state.deinit();
    }

    fn pair(self: *HeldBoth, ring: bool) !pass.PairOf(world) {
        return .init(testing.allocator, self.w, &self.c, self.capacity, ring, fx.shape.vocab / world);
    }
};

const Reply = struct { tokens: []i32, stats: draft.Stats };

/// `count` tokens after `prompt` by one-row steps, as `testing.serial.serialReply` on one rank. Ends on `eos`.
fn serialHeld(gpa: Allocator, held: Pair.Model, prompt: []const i32, count: usize, settings: sampling.Sampling, eos: []const u32) ![]i32 {
    const vocab = held.ranks[0].shape.vocab;
    var state: Pair.State = try .init(gpa, held, held.max_rows, 64);
    defer state.deinit();
    var b: Pair.Buffers = try .init(gpa, held, held.max_rows, 64);
    defer b.deinit();
    var sampler: sampling.Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, vocab);
    defer gpa.free(logits);
    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    for (prompt) |token| {
        try Pair.forward(held, &state, &b, &.{token});
        try Pair.commit(held, &state, b, &.{token}, 1);
    }
    var made: usize = 0;
    while (made < count) {
        try Pair.logitsRow(b, 0, logits);
        out[made] = try sampler.choose(logits, prompt.len + made, settings);
        made += 1;
        if (std.mem.indexOfScalar(u32, eos, @intCast(out[made - 1])) != null) break;
        const step = [1]i32{out[made - 1]};
        try Pair.forward(held, &state, &b, &step);
        try Pair.commit(held, &state, b, &step, 1);
    }
    return gpa.realloc(out, made);
}

fn draftedHeld(gpa: Allocator, both: *HeldBoth, flash: *pass.PairOf(world), prompt: []const i32, count: usize, settings: sampling.Sampling, graphed: bool, eos: []const u32, spoil: ?*Spoil) !Reply {
    return draftedHeldCompared(gpa, both, flash, prompt, count, settings, graphed, eos, spoil, null);
}

fn draftedHeldCompared(gpa: Allocator, both: *HeldBoth, flash: *pass.PairOf(world), prompt: []const i32, count: usize, settings: sampling.Sampling, graphed: bool, eos: []const u32, spoil: ?*Spoil, evidence: ?*borrowed_fixture.Evidence) !Reply {
    var state: Pair.State = try .init(gpa, both.held, both.held.max_rows, both.capacity);
    defer state.deinit();
    var b: Pair.Buffers = try .init(gpa, both.held, both.held.max_rows, both.capacity);
    defer b.deinit();
    try b.setTaps(both.c.taps, .f32_acc);
    var graphs: Pair.Graphs = try .init(gpa);
    defer graphs.deinit();
    // graphed: the drafter's passes too, over the taps of `b`, dropped with it
    var passes: dflash2_pass.Graphs = try .init(gpa);
    defer passes.deinit();
    if (graphed) flash.graphs = &passes;
    defer flash.graphs = null;
    var sampler: sampling.Sampler = try .init(gpa, both.held.ranks[0].shape.vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, both.held.ranks[0].shape.vocab);
    defer gpa.free(logits);
    var d: drafter_file.DrafterOf(world) = try .init(gpa, &both.c, flash, try .heldBy(gpa, both.held, &b), both.books, settings);
    defer d.deinit();
    if (evidence) |record| {
        for (d.mains(), both.held.ranks) |main, rank| {
            const stored = rank.head.runs[0].linear.bf16.weight;
            try testing.expectEqual(stored.view.ptr, main.rows.ptr);
            try testing.expectEqual(stored.view.ptr, main.head.bf16.weight.view.ptr);
            try testing.expectEqual(stored.bytes(), main.head.bf16.weight.bytes());
        }
        try record.install(&d, both.arena_state.allocator(), gpa);
    }
    d.confidence = 0;
    d.cap = both.c.block - 1;
    d.ends = eos;
    var decoder: Decoder = .{ .m = both.held, .state = &state, .b = &b, .graphs = if (graphed) &graphs else null, .sampler = &sampler, .settings = settings, .eos = eos, .logits = logits };
    const run: draft.Drafter = if (spoil) |s| blk: {
        s.inner = d.drafter();
        break :blk s.drafter();
    } else d.drafter();
    var chosen: [draft.forward_max_rows]i32 = undefined;
    var drafts: [draft.forward_max_rows]i32 = undefined;
    var kept = try decoder.prompt(run, prompt, &chosen);
    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    var made: usize = 1;
    out[0] = chosen[kept - 1];
    if (std.mem.indexOfScalar(u32, eos, @intCast(out[0])) != null) return .{ .tokens = try gpa.realloc(out, 1), .stats = decoder.stats };
    while (made < count) {
        const n = try decoder.draft(run, chosen[0..kept], count - made - 1, &drafts);
        if (evidence) |record| try record.capture(&d);
        kept = try decoder.round(out[made - 1], drafts[0..n], &chosen);
        @memcpy(out[made..][0..kept], chosen[0..kept]);
        made += kept;
        if (std.mem.indexOfScalar(u32, eos, @intCast(out[made - 1])) != null) break;
    }
    return .{ .tokens = try gpa.realloc(out, made), .stats = decoder.stats };
}

/// Target ranks and the drafter's context stand at the same position after a reply.
fn expectRanksClean(pair: *Both, flash: *pass.PairOf(world)) !void {
    try testing.expectEqual(pair.state.ranks[0].pos, pair.state.ranks[1].pos);
    try testing.expectEqual(pair.state.pos, pair.state.ranks[0].pos);
    try testing.expectEqual(pair.state.pos + 1, pair.seen.items.len);
    try testing.expectEqual(pair.state.pos, flash.context_end);
}

/// Replaces the draft at depth `at` with a wrong token so the verifier must reject it.
const Spoil = struct {
    inner: draft.Drafter = undefined,
    at: usize,
    vocab: i32,

    fn drafter(self: *Spoil) draft.Drafter {
        return .{ .context = self, .draft = draftFn };
    }

    fn draftFn(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Spoil = @ptrCast(@alignCast(context));
        const n = try self.inner.draft(self.inner.context, streams, tokens, position, guide, out);
        if (n > self.at) out[self.at] = @mod(out[self.at] + 7, self.vocab);
        return n;
    }
};

test "two ranks in loopback: DFlash2 drafts are the serial tokens, greedy and sampled, graphed or not, the drafter's passes too" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 12;
    var both: HeldBoth = undefined;
    try both.init(64);
    defer both.deinit();
    var flash = try both.pair(false);
    defer flash.deinit();

    var drafted: usize = 0;
    for ([_]sampling.Sampling{ .{ .seed = 3, .temperature = 0 }, .{ .seed = 5, .temperature = 0.8 } }) |settings| {
        const serial = try serialHeld(gpa, both.held, &prompt, count, settings, &.{});
        defer gpa.free(serial);
        // captured, the target's rounds and the drafter's passes draft what eager ones do, to the draft
        var eager: draft.Stats = undefined;
        for ([_]bool{ false, true }) |graphed| {
            flash.reset();
            const got = try draftedHeld(gpa, &both, &flash, &prompt, count, settings, graphed, &.{}, null);
            defer gpa.free(got.tokens);
            try testing.expectEqualSlices(i32, serial, got.tokens);
            try testing.expect(got.stats.drafted > 0);
            if (graphed) try testing.expectEqual(eager, got.stats) else eager = got.stats;
            drafted += got.stats.drafted;
        }
    }
    try testing.expect(drafted > 0);
}

test "two ranks in loopback: the verifier rejects a wrong draft at first, middle and last depth; tokens stay serial" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 10;
    var both: HeldBoth = undefined;
    try both.init(64);
    defer both.deinit();
    var flash = try both.pair(false);
    defer flash.deinit();
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const serial = try serialHeld(gpa, both.held, &prompt, count, settings, &.{});
    defer gpa.free(serial);
    const last = block - 2;
    for ([_]usize{ 0, last / 2, last }) |at| {
        flash.reset();
        var spoil: Spoil = .{ .at = at, .vocab = @intCast(fx.shape.vocab) };
        const got = try draftedHeld(gpa, &both, &flash, &prompt, count, settings, false, &.{}, &spoil);
        defer gpa.free(got.tokens);
        try testing.expectEqualSlices(i32, serial, got.tokens);
        try testing.expect(got.stats.accepted < got.stats.drafted);
    }
}

test "two ranks in loopback: a captured pair proposes the eager bits; a context taken, overwritten by another sequence and restored proposes them again" {
    // spec 126 WS-01: no test set a pair's graphs (`dflash2_pass.Graphs`), and the restore was held to its end alone
    const gpa = testing.allocator;
    var both: HeldBoth = undefined;
    try both.init(96);
    defer both.deinit();
    const c = both.c;
    var b: Pair.Buffers = try .init(gpa, both.held, both.held.max_rows, both.capacity);
    defer b.deinit();
    try b.setTaps(c.taps, .f32_acc);
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    // two sequences' taps, each its own buffer as each window's are: a graph is keyed on the rows' address
    var taps: [2]cuda.Buffer = undefined;
    var made: usize = 0;
    defer for (taps[0..made]) |*one| one.deinit();
    const lengths = [_]usize{ 70, 21 };
    var prng: std.Random.DefaultPrng = .init(1262);
    for (&taps, lengths) |*one, n| {
        const host = try gpa.alloc(u16, n * c.tapWidth());
        defer gpa.free(host);
        for (host) |*bits| bits.* = numeric.bf16FromF32(prng.random().floatNorm(f32));
        one.* = try .init(.bf16, &.{ n, c.tapWidth() });
        made += 1;
        try one.upload(std.mem.sliceAsBytes(host));
    }
    const a = [world]cuda.Buffer{ taps[0], taps[0] };
    const other = [world]cuda.Buffer{ taps[1], taps[1] };

    var cold = try both.pair(false);
    defer cold.deinit();
    var d: drafter_file.DrafterOf(world) = try .init(gpa, &both.c, &cold, try .heldBy(gpa, both.held, &b), both.books, settings);
    defer d.deinit();
    const mains = d.mains();
    const Bits = struct {
        fn of(pair: *pass.PairOf(world), cc: config_file.Config, main: [world]dflash2_pass.Main, pending: i32) ![]u8 {
            try pair.block(main, pending);
            try cuda.synchronize();
            return blockBits(testing.allocator, cc, pair.tops, .{ &pair.ranks[0], &pair.ranks[1] }, 1, 0);
        }
    };
    // the eager pair: a's 70 rows in one pass, then its block; and b's alone
    try cold.addTaps(a, 0, lengths[0]);
    const want = try Bits.of(&cold, c, mains, 17);
    defer gpa.free(want);
    try testing.expect(std.mem.indexOfNone(u8, want, &.{0}) != null);
    var fresh = try both.pair(false);
    defer fresh.deinit();
    try fresh.addTaps(other, 0, lengths[1]);
    const want_other = try Bits.of(&fresh, c, mains, 5);
    defer gpa.free(want_other);

    var graphs: dflash2_pass.Graphs = try .init(gpa);
    defer graphs.deinit();
    var hot = try both.pair(false);
    defer hot.deinit();
    hot.graphs = &graphs;
    defer hot.graphs = null;
    // a prompt's 62 rows in one pass, then a round's 8 as a captured update (spec 135 WS-02)
    try hot.addTaps(a, 0, lengths[0] - block);
    try hot.addTaps(a, lengths[0] - block, block);
    const captured = try Bits.of(&hot, c, mains, 17);
    defer gpa.free(captured);
    try testing.expectEqualSlices(u8, want, captured);
    try testing.expect(graphs.whole != null and graphs.updates.count() == 1);
    // replayed, not captured again: the same bits
    const replayed = try Bits.of(&hot, c, mains, 17);
    defer gpa.free(replayed);
    try testing.expectEqualSlices(u8, want, replayed);

    // taken, then another sequence from the start over the same graphs: its own bits
    const kept = try pass.KeptOf(world).take(&hot);
    defer {
        var copy = kept;
        copy.deinit();
    }
    try testing.expectEqual(lengths[0], kept.end);
    hot.reset();
    try hot.addTaps(other, 0, lengths[1]);
    const between = try Bits.of(&hot, c, mains, 5);
    defer gpa.free(between);
    try testing.expectEqualSlices(u8, want_other, between);
    // restored: the first sequence's bits again, and its context the eager pair's
    try kept.restore(&hot);
    try testing.expectEqual(lengths[0], hot.context_end);
    const again = try Bits.of(&hot, c, mains, 17);
    defer gpa.free(again);
    try testing.expectEqualSlices(u8, want, again);
    const cold_context = try contextBits(gpa, .{ cold.ranks[0].k, cold.ranks[1].k }, .{ cold.ranks[0].v, cold.ranks[1].v });
    defer gpa.free(cold_context);
    const hot_context = try contextBits(gpa, .{ hot.ranks[0].k, hot.ranks[1].k }, .{ hot.ranks[0].v, hot.ranks[1].v });
    defer gpa.free(hot_context);
    // the window back is what a restore brings and a block reads; before it, and past the end (each pair's
    // last block's K and V), the rows are no one's
    const row = 128 * 2;
    const back = lengths[0] - c.window.?;
    const layer_bytes = cold.ranks[0].k[0].bytes();
    const cap = cold.ranks[0].cap;
    var at: usize = 0;
    while (at < cold_context.len) : (at += layer_bytes) {
        const heads = layer_bytes / (cap * row);
        for (0..heads) |h| {
            const first = at + (h * cap + back) * row;
            try testing.expectEqualSlices(u8, cold_context[first..][0 .. c.window.? * row], hot_context[first..][0 .. c.window.? * row]);
        }
    }
}

test "two ranks in loopback: EOS mid-draft stops serial and drafted alike" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 12;
    var both: HeldBoth = undefined;
    try both.init(64);
    defer both.deinit();
    var flash = try both.pair(false);
    defer flash.deinit();
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const full = try serialHeld(gpa, both.held, &prompt, count, settings, &.{});
    defer gpa.free(full);
    try testing.expect(full.len >= 4);
    const eos = [_]u32{@intCast(full[2])};
    const serial = try serialHeld(gpa, both.held, &prompt, count, settings, &eos);
    defer gpa.free(serial);
    try testing.expectEqualSlices(i32, full[0..3], serial);
    flash.reset();
    const got = try draftedHeld(gpa, &both, &flash, &prompt, count, settings, false, &eos, null);
    defer gpa.free(got.tokens);
    try testing.expectEqualSlices(i32, serial, got.tokens);
    try testing.expect(got.stats.drafted > 0);
}

test "two ranks in loopback: pair.reply with a given DFlash2 drafter is the serial reply" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 8;
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };

    var both: HeldBoth = undefined;
    try both.init(64);
    defer both.deinit();
    const serial = try serialHeld(gpa, both.held, &prompt, count, settings, &.{});
    defer gpa.free(serial);

    const built = try random.shares(2, gpa, arena, 42, 0, rows_cap);
    const pair = try Both.createLoopback(gpa, .{ .ranks = built, .max_rows = rows_cap }, rows_cap, 64, true);
    defer pair.destroy();
    try pair.b.setTaps(both.c.taps, .f32_acc);
    var flash = try both.pair(false);
    defer flash.deinit();
    var d: drafter_file.DrafterOf(world) = try .init(gpa, &both.c, &flash, try .heldBy(gpa, pair.m, &pair.b), both.books, settings);
    defer d.deinit();
    d.confidence = 0;
    d.cap = both.c.block - 1;
    const got = try pair.reply(&prompt, settings, &.{}, .{ .given = d.drafter() }, .{}, null, count, null, null);
    try testing.expectEqual(count, got.tokens);
    try testing.expectEqualSlices(i32, serial, pair.tokens.items);
    try testing.expect(got.stats.drafted > 0);
}

test "two ranks in loopback: DFlash2 usage is the serial reply's" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 8;
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };

    var both: HeldBoth = undefined;
    try both.init(64);
    defer both.deinit();
    const serial_ids = try serialHeld(gpa, both.held, &prompt, count, settings, &.{});
    defer gpa.free(serial_ids);

    const built = try random.shares(2, gpa, arena, 42, 0, rows_cap);
    const pair = try Both.createLoopback(gpa, .{ .ranks = built, .max_rows = rows_cap }, rows_cap, 64, true);
    defer pair.destroy();
    try pair.b.setTaps(both.c.taps, .f32_acc);
    var flash = try both.pair(false);
    defer flash.deinit();
    var d: drafter_file.DrafterOf(world) = try .init(gpa, &both.c, &flash, try .heldBy(gpa, pair.m, &pair.b), both.books, settings);
    defer d.deinit();
    d.confidence = 0;
    d.cap = both.c.block - 1;

    const serial = try pair.reply(&prompt, settings, &.{}, .{}, .{}, null, count, null, null);
    const serial_tok = try gpa.dupe(i32, pair.tokens.items);
    defer gpa.free(serial_tok);
    try testing.expectEqualSlices(i32, serial_ids, serial_tok);
    try pair.reset();
    d.reset(settings);
    d.confidence = 0;
    d.cap = both.c.block - 1;
    const before = pair.tokens.items.len;
    const drafted = try pair.reply(&prompt, settings, &.{}, .{ .given = d.drafter() }, .{}, null, count, null, null);
    try testing.expectEqual(serial.tokens, drafted.tokens);
    try testing.expectEqual(serial.cached, drafted.cached);
    try testing.expectEqual(serial.windows, drafted.windows);
    try testing.expectEqual(serial.taken, drafted.taken);
    try testing.expectEqual(serial.finish, drafted.finish);
    try testing.expectEqual(count, drafted.tokens);
    try testing.expectEqual(@as(usize, 0), drafted.cached);
    try testing.expectEqualSlices(i32, serial_tok, pair.tokens.items[before..]);
    try testing.expect(drafted.stats.drafted > 0);
    try expectRanksClean(pair, &flash);
}

test "two ranks in loopback: a DFlash2 reply cancelled mid-draft is a prefix of serial and the next request is clean" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 12;
    const hang_at: usize = 3;
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };

    var both: HeldBoth = undefined;
    try both.init(64);
    defer both.deinit();
    const serial = try serialHeld(gpa, both.held, &prompt, count, settings, &.{});
    defer gpa.free(serial);
    try testing.expect(serial.len > hang_at);

    const built = try random.shares(2, gpa, arena, 42, 0, rows_cap);
    const pair = try Both.createLoopback(gpa, .{ .ranks = built, .max_rows = rows_cap }, rows_cap, 64, true);
    defer pair.destroy();
    try pair.b.setTaps(both.c.taps, .f32_acc);
    var flash = try both.pair(false);
    defer flash.deinit();
    var d: drafter_file.DrafterOf(world) = try .init(gpa, &both.c, &flash, try .heldBy(gpa, pair.m, &pair.b), both.books, settings);
    defer d.deinit();
    d.confidence = 0;
    d.cap = both.c.block - 1;

    const cut = try pair.reply(&prompt, settings, &.{}, .{ .given = d.drafter() }, .{}, null, count, hang_at, null);
    try testing.expectEqual(.cut, cut.finish);
    try testing.expectEqual(hang_at, cut.tokens);
    try testing.expectEqualSlices(i32, serial[0..hang_at], pair.tokens.items);
    try testing.expect(cut.stats.drafted > 0);
    if (flash.context_end != pair.state.pos) {
        if (pair.state.pos == 0) flash.reset() else try flash.startAt(pair.state.pos);
    }
    try expectRanksClean(pair, &flash);

    var turn: std.ArrayList(i32) = .empty;
    defer turn.deinit(gpa);
    try turn.appendSlice(gpa, pair.seen.items);
    try turn.appendSlice(gpa, &.{ 7, 9 });
    const want = try serialHeld(gpa, both.held, turn.items, count, settings, &.{});
    defer gpa.free(want);
    const before = pair.tokens.items.len;
    const again = try pair.reply(turn.items, settings, &.{}, .{ .given = d.drafter() }, .{}, null, count, null, null);
    try testing.expectEqual(count, again.tokens);
    try testing.expectEqualSlices(i32, want, pair.tokens.items[before..]);
    try expectRanksClean(pair, &flash);
}

/// Each tap layer's completed streams as the watched forward hands them: `h_in` of the layer after it, or the
/// final streams after the last.
const Streams = struct {
    taps: []const u32,
    layers: usize,
    seen: [2][]u16,

    fn tap(self: *Streams) forward_file.Tap {
        return .{ .context = self, .at = at, .passive = true };
    }

    fn at(context: *anyopaque, point: forward_file.Point, layer: usize, view: cuda.View) anyerror!void {
        const self: *Streams = @ptrCast(@alignCast(context));
        for (self.taps, 0..) |one, slot| {
            const after = one + 1;
            const here = if (after < self.layers) point == .h_in and layer == after else point == .final_h and layer == self.layers;
            if (here) try cuda.download(view, std.mem.sliceAsBytes(self.seen[slot]));
        }
    }
};

test "two ranks in loopback: each rank's DFlash2 taps are its streams' mean after the tap layers, split rows or not" {
    // spec 126 WS-04: `Held.launch` wrote no taps, so the drafter read a buffer nothing had filled
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    // wide enough that the window splits its rows over the ranks where nothing keeps it whole (`ranks_test.zig`)
    const rows_max = 48;
    const built = try random.sharesOf(2, gpa, arena_state.allocator(), 42, 0, rows_max, .{ .caches = .bf16 });
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_max };
    defer held.deinit();
    const capacity = 64;
    const rows = 33;
    var prompt: [rows]i32 = undefined;
    var prng: std.Random.DefaultPrng = .init(126);
    for (&prompt) |*id| id.* = prng.random().intRangeLessThan(i32, 1, 300);
    const shape = held.ranks[0].shape;
    const hidden = shape.hidden;
    const nstreams = shape.streams;
    const taps = [_]u32{ 0, 2 };
    const wide = taps.len * hidden;

    // the reference: a watched forward, which never splits its rows
    var seen: [2][]u16 = undefined;
    for (&seen) |*one| one.* = try gpa.alloc(u16, rows * nstreams * hidden);
    defer for (seen) |one| gpa.free(one);
    var streams: Streams = .{ .taps = &taps, .layers = held.ranks[0].layers.len, .seen = seen };
    var state: Pair.State = try .init(gpa, held, rows_max, capacity);
    defer state.deinit();
    var b: Pair.Buffers = try .init(gpa, held, rows_max, capacity);
    defer b.deinit();
    try b.setTaps(&taps, .f32_acc);
    try Pair.stageAll(held, &state, &b, &prompt);
    try Pair.launch(held, &state, &b, rows, null, .all, streams.tap());
    const want = try gpa.alloc(u16, rows * wide);
    defer gpa.free(want);
    const mean = try gpa.alloc(u16, rows * hidden);
    defer gpa.free(mean);
    for (seen, 0..) |one, slot| {
        taps_file.streamMeanRows(one, mean, rows, nstreams, .f32_acc);
        for (0..rows) |r| @memcpy(want[r * wide + slot * hidden ..][0..hidden], mean[r * hidden ..][0..hidden]);
    }
    try testing.expect(std.mem.indexOfNone(u16, want, &.{ 0, 0x8000 }) != null);
    const got = try gpa.alloc(u16, rows * wide);
    defer gpa.free(got);
    for (b.ranks) |rank| {
        try cuda.download(rank.taps.?.rows(rows), std.mem.sliceAsBytes(got));
        try testing.expectEqualSlices(u16, want, got);
    }

    // a window past the split's threshold keeps its rows whole while taps are kept, and has the same taps
    const saved = ranks.split_rows;
    defer ranks.split_rows = saved;
    ranks.split_rows = 17;
    try testing.expect(Pair.splits(held, rows, null));
    var split_state: Pair.State = try .init(gpa, held, rows_max, capacity);
    defer split_state.deinit();
    var split_b: Pair.Buffers = try .init(gpa, held, rows_max, capacity);
    defer split_b.deinit();
    try split_b.setTaps(&taps, .f32_acc);
    try Pair.forward(held, &split_state, &split_b, &prompt);
    for (split_b.ranks) |rank| {
        try cuda.download(rank.taps.?.rows(rows), std.mem.sliceAsBytes(got));
        try testing.expectEqualSlices(u16, want, got);
    }
}

test "two ranks in loopback: a prompt's taps in a window's passes leave the context a round's updates leave, byte for byte, stored and mixed, flat and in the ring" {
    // spec 135 WS-02: a prompt's rows were 64 an update; now a pass a window's (up to the ring's rows)
    const gpa = testing.allocator;
    const capacity = 420;
    for ([_]weights.Storage{ .stored, .mixed }) |storage| {
        var both: HeldBoth = undefined;
        try both.initStored(capacity, storage);
        defer both.deinit();
        const c = both.c;
        for ([_]bool{ false, true }) |ring| {
            // in the ring (192 rows here) past twice its rows, so the passes wrap; flat in one pass
            const n: usize = if (ring) 400 else 190;
            const host = try gpa.alloc(u16, n * c.tapWidth());
            defer gpa.free(host);
            var prng: std.Random.DefaultPrng = .init(1350);
            for (host) |*bits| bits.* = numeric.bf16FromF32(prng.random().floatNorm(f32));
            var taps: cuda.Buffer = try .init(.bf16, &.{ n, c.tapWidth() });
            defer taps.deinit();
            try taps.upload(std.mem.sliceAsBytes(host));
            const both_taps = [world]cuda.Buffer{ taps, taps };

            var wide = try both.pair(ring);
            defer wide.deinit();
            try testing.expectEqual(ring, wide.ranks[0].ring);
            try wide.addTaps(both_taps, 0, n);
            try testing.expect(n > wide.wides[0].rows or !ring);

            // a round's updates, a block's rows each, as captured graphs
            var graphs: dflash2_pass.Graphs = try .init(gpa);
            defer graphs.deinit();
            var rounds = try both.pair(ring);
            defer rounds.deinit();
            rounds.graphs = &graphs;
            defer rounds.graphs = null;
            var at: usize = 0;
            while (at < n) : (at += block) try rounds.addTaps(both_taps, at, @min(block, n - at));
            try cuda.synchronize();
            try testing.expect(graphs.updates.count() > 0);
            try testing.expectEqual(n, wide.context_end);
            try testing.expectEqual(n, rounds.context_end);

            const wide_bits = try contextBits(gpa, .{ wide.ranks[0].k, wide.ranks[1].k }, .{ wide.ranks[0].v, wide.ranks[1].v });
            defer gpa.free(wide_bits);
            const round_bits = try contextBits(gpa, .{ rounds.ranks[0].k, rounds.ranks[1].k }, .{ rounds.ranks[0].v, rounds.ranks[1].v });
            defer gpa.free(round_bits);
            try testing.expect(std.mem.indexOfNone(u8, wide_bits, &.{0}) != null);
            try testing.expectEqualSlices(u8, round_bits, wide_bits);
        }
    }
}

/// A block's every bit as the pass left it, on both ranks: their largest logits and tokens, the logits and
/// the projected rows, block `at` of `blocks` side by side.
fn blockBits(gpa: Allocator, c: config_file.Config, tops: cuda.Buffer, ranks_of: [world]*const dflash2_pass.Rank, blocks: usize, at: usize) ![]u8 {
    const drafts = c.block - 1;
    const rows = blocks * drafts;
    const table = drafts * c.top_k * 4;
    const logits = @as(usize, @intCast(ranks_of[0].logits.view.size[1])) * drafts * 2;
    const selected = c.selector_rank * drafts * 2;
    const out = try gpa.alloc(u8, world * (2 * table + logits + selected));
    errdefer gpa.free(out);
    var used: usize = 0;
    for (ranks_of, 0..) |r, rank| {
        const mine = rank * 2 * rows * c.top_k * 4;
        try tops.downloadAt(mine + at * table, out[used..][0..table]);
        used += table;
        try tops.downloadAt(mine + rows * c.top_k * 4 + at * table, out[used..][0..table]);
        used += table;
        try r.logits.downloadAt(at * logits, out[used..][0..logits]);
        used += logits;
        try r.selected.downloadAt(at * selected, out[used..][0..selected]);
        used += selected;
    }
    return out;
}

/// Each layer's K and V of a context, both ranks, end to end.
fn contextBits(gpa: Allocator, k: [world][]const cuda.Buffer, v: [world][]const cuda.Buffer) ![]u8 {
    var size: usize = 0;
    for (k, v) |ks, vs| for (ks, vs) |one, other| {
        size += one.bytes() + other.bytes();
    };
    const out = try gpa.alloc(u8, size);
    errdefer gpa.free(out);
    var used: usize = 0;
    for (k, v) |ks, vs| for (ks, vs) |one, other| for ([_]cuda.Buffer{ one, other }) |buffer| {
        try buffer.download(out[used..][0..buffer.bytes()]);
        used += buffer.bytes();
    };
    return out;
}

test "two ranks in loopback: DFlash2 seats beside others are each the lone pair's bits, flat and in the ring" {
    // spec 126 WS-01: `seats.zig` (the pool's drafter, `Engine.seatDrafter`) had no test
    const gpa = testing.allocator;
    // past the ring of the window, a block and a tile (192 rows here)
    const capacity = 200;
    var both: HeldBoth = undefined;
    try both.init(capacity);
    defer both.deinit();
    const c = both.c;
    const width = c.tapWidth();
    // each sequence's taps at random: a seat reads the main model's rows, whatever made them
    const lengths = [_]usize{ 8, 196, 13, 9 };
    var taps: [lengths.len]cuda.Buffer = undefined;
    var made: usize = 0;
    defer for (taps[0..made]) |*one| one.deinit();
    var prng: std.Random.DefaultPrng = .init(1261);
    for (&taps, lengths) |*one, n| {
        const host = try gpa.alloc(u16, n * width);
        defer gpa.free(host);
        for (host) |*bits| bits.* = numeric.bf16FromF32(prng.random().floatNorm(f32));
        one.* = try .init(.bf16, &.{ n, width });
        made += 1;
        try one.upload(std.mem.sliceAsBytes(host));
    }
    var b: Pair.Buffers = try .init(gpa, both.held, both.held.max_rows, capacity);
    defer b.deinit();
    try b.setTaps(c.taps, .f32_acc);
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };

    // rounds of a slot's rows from its start and the token of its block, none where it only absorbs; a slot
    // sits the second round out, the order changes, a seat's rows pass a batch of 64 and wrap the ring
    const Part = struct { slot: usize, from: usize, rows: usize, pending: ?i32 };
    const rounds = [_][]const Part{
        &.{ .{ .slot = 0, .from = 0, .rows = 5, .pending = 11 }, .{ .slot = 1, .from = 0, .rows = 70, .pending = 23 }, .{ .slot = 2, .from = 0, .rows = 1, .pending = null }, .{ .slot = 3, .from = 0, .rows = 9, .pending = 4 } },
        &.{ .{ .slot = 2, .from = 1, .rows = 12, .pending = 40 }, .{ .slot = 0, .from = 5, .rows = 3, .pending = 7 }, .{ .slot = 1, .from = 70, .rows = 126, .pending = 99 } },
    };
    var compared: usize = 0;
    for ([_]bool{ false, true }) |ring| {
        var base = try both.pair(ring);
        defer base.deinit();
        var d: drafter_file.DrafterOf(world) = try .init(gpa, &both.c, &base, try .heldBy(gpa, both.held, &b), both.books, settings);
        defer d.deinit();
        const mains = d.mains();
        var seats: pass.SeatsOf(world) = try .init(gpa, &base, lengths.len);
        defer seats.deinit();
        var lone: [lengths.len]pass.PairOf(world) = undefined;
        var opened: usize = 0;
        defer for (lone[0..opened]) |*one| one.deinit();
        for (&lone) |*one| {
            one.* = try both.pair(ring);
            opened += 1;
        }
        try testing.expectEqual(ring, lone[1].ranks[0].ring);

        for (rounds) |parts| {
            // the round's rows side by side, as the pool's last window holds them
            var total: usize = 0;
            for (parts) |part| total += part.rows;
            var joined: cuda.Buffer = try .init(.bf16, &.{ total, width });
            defer joined.deinit();
            var items: [lengths.len]seats_file.Item = undefined;
            var at: usize = 0;
            for (parts, items[0..parts.len]) |part, *item| {
                try joined.copyRows(at, taps[part.slot], part.from, part.rows);
                item.* = .{ .slot = part.slot, .at = part.from, .from = at, .rows = part.rows, .pending = part.pending };
                at += part.rows;
            }
            const blocks = try seats.round(&base, .{ joined, joined }, mains, items[0..parts.len]);
            try cuda.synchronize();
            const seat_bits = try gpa.alloc([]u8, blocks);
            var kept: usize = 0;
            defer {
                for (seat_bits[0..kept]) |one| gpa.free(one);
                gpa.free(seat_bits);
            }
            for (seat_bits) |*one| {
                one.* = try blockBits(gpa, c, seats.tops, .{ &seats.wide[0], &seats.wide[1] }, blocks, kept);
                kept += 1;
            }

            // each slot alone, in its own pair: the same rows, then its block
            var block_index: usize = 0;
            for (parts) |part| {
                const one = &lone[part.slot];
                try testing.expectEqual(part.from, one.context_end);
                try one.addTaps(.{ taps[part.slot], taps[part.slot] }, part.from, part.rows);
                if (part.pending) |token| {
                    try one.block(mains, token);
                    try cuda.synchronize();
                    try testing.expectEqual(part.slot, seats.order[block_index]);
                    const alone = try blockBits(gpa, c, one.tops, .{ &one.ranks[0], &one.ranks[1] }, 1, 0);
                    defer gpa.free(alone);
                    try testing.expect(std.mem.indexOfNone(u8, alone, &.{0}) != null);
                    try testing.expectEqualSlices(u8, alone, seat_bits[block_index]);
                    block_index += 1;
                    compared += 1;
                }
                const seat = seats.seats[part.slot];
                const want = try contextBits(gpa, .{ one.ranks[0].k, one.ranks[1].k }, .{ one.ranks[0].v, one.ranks[1].v });
                defer gpa.free(want);
                const got = try contextBits(gpa, seat.k, seat.v);
                defer gpa.free(got);
                try testing.expectEqualSlices(u8, want, got);
            }
            try testing.expectEqual(blocks, block_index);
        }
    }
    try testing.expectEqual(@as(usize, 12), compared);
}

test "two ranks in loopback: DFlash2 borrowed head equality stored and mixed, eager and captured, teardown then main decode" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const settings: sampling.Sampling = .{ .seed = 5, .temperature = 0.8 };
    for ([_]weights.Storage{ .stored, .mixed }) |storage| {
        var both: HeldBoth = undefined;
        try both.initStored(64, storage);
        defer both.deinit();
        const serial = try serialHeld(gpa, both.held, &prompt, 12, settings, &.{});
        defer gpa.free(serial);
        var eager: [32]u8 = undefined;
        for ([_]bool{ false, true }) |graphed| {
            var want: [32]u8 = undefined;
            for ([_]bool{ true, false }) |legacy| {
                var pair = try both.pair(false);
                defer pair.deinit();
                var evidence: borrowed_fixture.Evidence = .{ .legacy = legacy, .mixed = storage == .mixed };
                defer evidence.deinit();
                const reply = try draftedHeldCompared(gpa, &both, &pair, &prompt, 12, settings, graphed, &.{}, null, &evidence);
                defer gpa.free(reply.tokens);
                try testing.expectEqualSlices(i32, serial, reply.tokens);
                try testing.expect(evidence.blocks > 0);
                const digest = evidence.digest();
                if (legacy) want = digest else {
                    try testing.expectEqualSlices(u8, &want, &digest);
                    if (graphed) try testing.expectEqualSlices(u8, &eager, &digest) else eager = digest;
                    std.debug.print("spec-155 equality held2 storage={t} captured={} blocks={d} sha256={x}\n", .{ storage, graphed, evidence.blocks, digest });
                }
            }
        }
        const continued = try serialHeld(gpa, both.held, &prompt, 12, settings, &.{});
        defer gpa.free(continued);
        try testing.expectEqualSlices(i32, serial, continued);
    }
}

test "two ranks in loopback: DFlash2 borrowed head rig ownership and partial target failure" {
    const rig_file = @import("../../../../src/models/qwen4/dflash2/rig.zig");
    const gpa = testing.allocator;
    for ([_]weights.Storage{ .stored, .mixed }) |storage| {
        var both: HeldBoth = undefined;
        try both.initStored(64, storage);
        defer both.arena_state.deinit();
        defer both.held.deinit();
        const prompt = [_]i32{ 5, 17, 200, 3 };
        const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
        const serial = try serialHeld(gpa, both.held, &prompt, 8, settings, &.{});
        defer gpa.free(serial);
        var b: Pair.Buffers = try .init(gpa, both.held, both.held.max_rows, both.capacity);
        defer b.deinit();
        // Rank zero is borrowed successfully before rank one's unsupported format rejects init.
        var invalid = both.held;
        invalid.ranks[1].head.runs[0].linear = .{ .mxfp8 = undefined };
        const before = cuda.account.bytes;
        var rejected_b: Pair.Buffers = try .init(gpa, both.held, both.held.max_rows, both.capacity);
        const rejected_heads = try rig_file.draftHeadsHeld(world, both.arena_state.allocator(), gpa, both.held, storage == .stored, both.held.ranks[0].scratch.rows(dflash2_pass.batch));
        var rejected: rig_file.Rig(world) = undefined;
        rejected.w = both.w;
        rejected.books = both.books;
        rejected.gpa = gpa;
        var opened: rig_file.Opened = .{ .config = both.c };
        try testing.expectError(error.UnsupportedShape, rejected.initHeld(gpa, &opened, rejected_heads, null, invalid, &rejected_b, 64, settings, true));
        rejected_b.deinit();
        try testing.expectEqual(before, cuda.account.bytes);
        var flash: rig_file.Rig(world) = undefined;
        flash.w = both.w;
        flash.books = both.books;
        flash.gpa = gpa;
        const heads = try rig_file.draftHeadsHeld(world, both.arena_state.allocator(), gpa, both.held, storage == .stored, both.held.ranks[0].scratch.rows(dflash2_pass.batch));
        try testing.expectEqual(storage == .stored, heads == null);
        try flash.initHeld(gpa, &opened, heads, null, both.held, &b, 64, settings, false);
        for (flash.d.mains(), both.held.ranks) |main, rank| try testing.expectEqual(rank.head.runs[0].linear.bf16.weight.view.ptr, main.rows.ptr);
        flash.deinit();
        const continued = try serialHeld(gpa, both.held, &prompt, 8, settings, &.{});
        defer gpa.free(continued);
        try testing.expectEqualSlices(i32, serial, continued);
    }
}
