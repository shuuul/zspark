//! `drafter.zig`'s tests on a random Qwen target and a random stored-BF16 DFlash2 drafter: proposals come
//! out, the verifier accepts/rejects, drafted==serial on greedy and fixed-seed sampled paths, rejection at
//! first/middle/last depth, prefix restore, and the mask token path.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const cuda = @import("../../../../src/cuda/cuda.zig");
const masks_file = @import("../../../../src/decode/masks.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const draft_file = @import("../../../../src/models/qwen4/draft.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const session_file = @import("../../../../src/models/qwen4/session/session.zig");
const fixtures = @import("../testing.zig");
const chain_file = @import("../../../../src/decode/dflash2/chain.zig");
const config_file = @import("../../../../src/decode/dflash2/config.zig");
const rig = @import("../../../../src/models/qwen4/dflash2/rig.zig");
const drafter_file = @import("../../../../src/models/qwen4/dflash2/drafter.zig");
const drafter_testing = @import("../../../../src/models/qwen4/checks/random_drafter.zig");
const pass = @import("../../../../src/models/qwen4/dflash2/pass.zig");
const dflash2_pass = @import("../../../../src/decode/dflash2/pass.zig");
const weights = @import("../../../../src/decode/dflash2/weights.zig");

const borrowed_fixture = @import("borrowed_head_fixture.zig");

const world = config_file.world;
const fx = fixtures.random_model.fx;
const block = 8;
const Decoder = draft_file.Decoder;
const Drafter = draft_file.Drafter;
const Counting = fixtures.sessions.Counting;
const serialReply = fixtures.serial.serialReply;

const Both = struct {
    arena_state: std.heap.ArenaAllocator,
    m: model_file.Model,
    c: config_file.Config,
    w: [world]weights.Weights,
    books: weights.Codebooks,
    capacity: usize,

    fn init(self: *Both, capacity: usize) !void {
        return self.initStored(capacity, .stored);
    }

    fn initStored(self: *Both, capacity: usize, storage: weights.Storage) !void {
        self.arena_state = .init(testing.allocator);
        errdefer self.arena_state.deinit();
        const arena = self.arena_state.allocator();
        self.capacity = capacity;
        self.m = try fixtures.random_model.testModel(testing.allocator, arena);
        errdefer self.m.deinit();
        self.c = try drafter_testing.config(arena, fx.shape.hidden, fx.shape.vocab, fx.layers, &.{ 0, 2 });
        self.c.block = block;
        var source: drafter_testing.Source = .empty;
        var prng: std.Random.DefaultPrng = .init(468);
        try drafter_testing.put(arena, prng.random(), self.c, &source);
        const rows: cuda.Scratch.Rows = .{ .pool = self.m.scratch, .count = dflash2_pass.wide };
        self.w[0] = try .load(arena, testing.allocator, source, &self.c, 0, storage, rows);
        errdefer self.w[0].deinit();
        self.w[1] = try .load(arena, testing.allocator, source, &self.c, 1, storage, rows);
        errdefer self.w[1].deinit();
        self.books = try .load(testing.allocator, source, &self.c);
        for (self.books.predecessor) |*value| value.* *= 12;
    }

    fn deinit(self: *Both) void {
        self.books.deinit(testing.allocator);
        for (&self.w) |*one| one.deinit();
        self.m.deinit();
        self.arena_state.deinit();
    }

    fn pair(self: *Both, ring: bool) !pass.PairOf(world) {
        return .init(testing.allocator, self.w, &self.c, self.capacity, ring, fx.shape.vocab / world);
    }
};

const Reply = struct { tokens: []i32, stats: draft_file.Stats };

fn draftedReply(gpa: Allocator, both: *Both, pair: *pass.PairOf(world), prompt: []const i32, count: usize, settings: sampling.Sampling, graphed: bool, spoil: ?*Spoil) !Reply {
    return draftedReplyCompared(gpa, both, pair, prompt, count, settings, graphed, spoil, null);
}

fn draftedReplyCompared(gpa: Allocator, both: *Both, pair: *pass.PairOf(world), prompt: []const i32, count: usize, settings: sampling.Sampling, graphed: bool, spoil: ?*Spoil, evidence: ?*borrowed_fixture.Evidence) !Reply {
    var state: forward.State = try .init(gpa, both.m, both.m.max_rows, both.capacity);
    defer state.deinit();
    var b: forward.Buffers = try .init(gpa, both.m, both.m.max_rows, both.capacity);
    defer b.deinit();
    try b.setTaps(both.c.taps, .f32_acc);
    var graphs: forward.Graphs = try .init(gpa);
    defer graphs.deinit();
    var sampler: sampling.Sampler = try .init(gpa, both.m.shape.vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, both.m.shape.vocab);
    defer gpa.free(logits);
    var d: drafter_file.Drafter = try .init(gpa, &both.c, pair, try .one(gpa, both.m, &b), both.books, settings);
    defer d.deinit();
    if (evidence) |record| try record.install(&d, both.arena_state.allocator(), gpa);
    d.confidence = 0;
    d.cap = both.c.block - 1;
    var passes: dflash2_pass.Graphs = try .init(gpa);
    defer passes.deinit();
    if (graphed) pair.graphs = &passes;
    defer pair.graphs = null;
    var decoder: Decoder = .{ .m = both.m, .state = &state, .b = &b, .graphs = if (graphed) &graphs else null, .sampler = &sampler, .settings = settings, .eos = &.{}, .logits = logits };
    const run: Drafter = if (spoil) |s| blk: {
        s.inner = d.drafter();
        break :blk s.drafter();
    } else d.drafter();
    var chosen: [draft_file.forward_max_rows]i32 = undefined;
    var drafts: [draft_file.forward_max_rows]i32 = undefined;
    var kept = try decoder.prompt(run, prompt, &chosen);
    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    var made: usize = 1;
    out[0] = chosen[kept - 1];
    while (made < count) {
        const n = try decoder.draft(run, chosen[0..kept], count - made - 1, &drafts);
        if (evidence) |record| try record.capture(&d);
        kept = try decoder.round(out[made - 1], drafts[0..n], &chosen);
        @memcpy(out[made..][0..kept], chosen[0..kept]);
        made += kept;
    }
    return .{ .tokens = out, .stats = decoder.stats };
}

/// Replaces the draft at depth `at` with a wrong token so the verifier must reject it.
const Spoil = struct {
    inner: Drafter = undefined,
    at: usize,
    vocab: i32,

    fn drafter(self: *Spoil) Drafter {
        return .{ .context = self, .draft = draft };
    }

    fn draft(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Spoil = @ptrCast(@alignCast(context));
        const n = try self.inner.draft(self.inner.context, streams, tokens, position, guide, out);
        if (n > self.at) out[self.at] = @mod(out[self.at] + 7, self.vocab);
        return n;
    }
};

test "replies drafted by DFlash2 are the serial replies: greedy and sampled, graphed or not" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 12;
    var both: Both = undefined;
    try both.init(64);
    defer both.deinit();
    var pair = try both.pair(false);
    defer pair.deinit();

    var drafted: usize = 0;
    for ([_]sampling.Sampling{ .{ .seed = 3, .temperature = 0 }, .{ .seed = 5, .temperature = 0.8 } }) |settings| {
        const serial = try serialReply(gpa, both.m, &prompt, count, settings);
        defer gpa.free(serial);
        for ([_]bool{ false, true }) |graphed| {
            pair.reset();
            const got = try draftedReply(gpa, &both, &pair, &prompt, count, settings, graphed, null);
            defer gpa.free(got.tokens);
            try testing.expectEqualSlices(i32, serial, got.tokens);
            try testing.expect(got.stats.drafted > 0);
            drafted += got.stats.drafted;
        }
    }
    try testing.expect(drafted > 0);
}

test "DFlash2: the verifier rejects a wrong draft at first, middle and last depth; tokens stay serial" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 10;
    var both: Both = undefined;
    try both.init(64);
    defer both.deinit();
    var pair = try both.pair(false);
    defer pair.deinit();
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const serial = try serialReply(gpa, both.m, &prompt, count, settings);
    defer gpa.free(serial);
    const last = block - 2;
    for ([_]usize{ 0, last / 2, last }) |at| {
        pair.reset();
        var spoil: Spoil = .{ .at = at, .vocab = @intCast(fx.shape.vocab) };
        const got = try draftedReply(gpa, &both, &pair, &prompt, count, settings, false, &spoil);
        defer gpa.free(got.tokens);
        try testing.expectEqualSlices(i32, serial, got.tokens);
        try testing.expect(got.stats.accepted < got.stats.drafted);
    }
}

test "DFlash2: a session's drafted replies match serial" {
    const gpa = testing.allocator;
    var both: Both = undefined;
    try both.init(256);
    defer both.deinit();
    var pair = try both.pair(false);
    defer pair.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("../../../fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const session: *session_file.Session = try .create(gpa, testing.io, both.m, &tokens, &.{}, .{ .window = both.m.max_rows, .capacity = 256, .snapshots = 0, .copies = false });
    defer session.destroy();
    try session.buffers.setTaps(both.c.taps, .f32_acc);
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    var d: drafter_file.Drafter = try .init(gpa, &both.c, &pair, try .one(gpa, both.m, &session.buffers), both.books, settings);
    defer d.deinit();
    d.confidence = 0;
    d.cap = both.c.block - 1;
    session.drafting = d.drafter();
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const count = 8;
    const serial = try serialReply(gpa, both.m, &prompt, count, settings);
    defer gpa.free(serial);
    var record: [count]i32 = undefined;
    var counting: Counting = .{};
    const reply = try session.reply(&prompt, settings, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
    try testing.expectEqual(count, reply.tokens);
    try testing.expectEqualSlices(i32, serial, record[0..reply.tokens]);
}

test "DFlash2: Kept restores a pair's prefix; a cold feed of the same rows stands there" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3 };
    var both: Both = undefined;
    try both.init(64);
    defer both.deinit();
    var hot = try both.pair(false);
    defer hot.deinit();
    var state: forward.State = try .init(gpa, both.m, both.m.max_rows, both.capacity);
    defer state.deinit();
    var b: forward.Buffers = try .init(gpa, both.m, both.m.max_rows, both.capacity);
    defer b.deinit();
    try b.setTaps(both.c.taps, .f32_acc);
    try forward.forward(both.m, &state, &b, &prompt);
    try forward.commit(both.m, &state, b, &prompt, prompt.len);
    var taps: [world]cuda.Buffer = undefined;
    for (&taps) |*one| one.* = b.taps.?;
    try hot.addTaps(taps, 0, prompt.len);
    const kept = try pass.KeptOf(world).take(&hot);
    defer {
        var copy = kept;
        copy.deinit();
    }
    try testing.expectEqual(prompt.len, kept.end);
    hot.reset();
    try kept.restore(&hot);
    try testing.expectEqual(prompt.len, hot.context_end);
    var cold = try both.pair(false);
    defer cold.deinit();
    try cold.addTaps(taps, 0, prompt.len);
    try testing.expectEqual(cold.context_end, hot.context_end);
}

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

const Forced = struct {
    fn next(_: *anyopaque, drafts: []const i32, more: bool) masks_file.Error!masks_file.Next {
        if (!more or drafts.len == 3) return .closed;
        return .{ .token = @intCast(10 + drafts.len) };
    }
};

test "DFlash2: the mask token is among a constraint's mask and a forced token takes a place of the block" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3 };
    var both: Both = undefined;
    try both.init(64);
    defer both.deinit();
    var pair = try both.pair(false);
    defer pair.deinit();
    var state: forward.State = try .init(gpa, both.m, both.m.max_rows, both.capacity);
    defer state.deinit();
    var b: forward.Buffers = try .init(gpa, both.m, both.m.max_rows, both.capacity);
    defer b.deinit();
    try b.setTaps(both.c.taps, .f32_acc);
    const settings: sampling.Sampling = .{ .seed = 1, .temperature = 0 };
    var d: drafter_file.Drafter = try .init(gpa, &both.c, &pair, try .one(gpa, both.m, &b), both.books, settings);
    defer d.deinit();
    d.confidence = 0;
    d.cap = both.c.block - 1;
    try forward.forward(both.m, &state, &b, &prompt);
    try forward.commit(both.m, &state, b, &prompt, prompt.len);
    const streams = draft_file.namespace.streams(b, prompt.len);
    const run = d.drafter();
    var out: [block]i32 = undefined;
    const free = try run.draft(run.context, streams, &prompt, prompt.len, null, &out);
    try testing.expectEqual(@as(usize, block - 1), free);
    try testing.expectEqual(@as(usize, 1), d.passes);
    const first_free = out[0];

    d.reset(settings);
    d.passes = 0;
    try pair.addTaps(.{ b.taps.?, b.taps.? }, 0, prompt.len);
    const c = try d.candidates(prompt[prompt.len - 1], 3);
    var other: u32 = undefined;
    for (c.tokens[0..c.top]) |token| if (token != first_free) {
        other = token;
        break;
    };
    var mask: [(fx.shape.vocab + 31) / 32]u32 = @splat(0);
    mask[other / 32] |= @as(u32, 1) << @intCast(other % 32);
    d.reset(settings);
    d.passes = 0;
    var constraint: Made = .{ .mask = &mask, .forced = @mod(first_free + 1, @as(i32, @intCast(fx.shape.vocab))) };
    const n = try run.draft(run.context, streams, &prompt, prompt.len, constraint.guide(), &out);
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqual(@as(i32, @intCast(other)), out[0]);
    try testing.expectEqual(constraint.forced, out[1]);
    try testing.expectEqual(@as(usize, 1), d.passes);
    const want = chain_file.pick(d.books, c, 2, @intCast(constraint.forced), prompt.len + 1, null, null, false).?;
    try testing.expectEqual(@as(i32, @intCast(want.token)), out[2]);

    d.reset(settings);
    d.passes = 0;
    const all = try run.draft(run.context, streams, &prompt, prompt.len, .{ .context = &constraint, .nextFn = Forced.next }, &out);
    try testing.expectEqual(@as(usize, 3), all);
    try testing.expectEqualSlices(i32, &.{ 10, 11, 12 }, out[0..3]);
    try testing.expectEqual(@as(usize, 0), d.passes);
}

test "the pinned DFlash2 checkpoint loads as stored BF16" {
    const gpa = testing.allocator;
    const home = testing.environ.getPosix("HOME") orelse return error.SkipZigTest;
    const path = try std.fmt.allocPrint(gpa, "{s}/models/Qwen3.8-Flash-Next-DFlash2", .{home});
    defer gpa.free(path);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: config_file.Diagnostic = .{};
    var dir = std.Io.Dir.cwd().openDir(testing.io, path, .{}) catch return error.SkipZigTest;
    defer dir.close(testing.io);
    const c = try config_file.Config.readStored(arena, testing.io, dir, &diagnostic);
    try testing.expectEqual(rig.published.hidden, c.hidden);
    try testing.expectEqualSlices(u32, &rig.published.taps, c.taps);
    var checkpoint = try @import("../../../../src/weights/safetensors.zig").Checkpoint.open(gpa, testing.io, .cwd(), path);
    defer checkpoint.close();
    try testing.expectEqual(@as(usize, 81), checkpoint.tensors.count());
}

test "DFlash2 borrowed head equality: stored and mixed, eager and captured, teardown then main decode" {
    const gpa = testing.allocator;
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const settings: sampling.Sampling = .{ .seed = 5, .temperature = 0.8 };
    for ([_]weights.Storage{ .stored, .mixed }) |storage| {
        var both: Both = undefined;
        try both.initStored(64, storage);
        defer both.deinit();
        const serial = try serialReply(gpa, both.m, &prompt, 12, settings);
        defer gpa.free(serial);
        var eager: [32]u8 = undefined;
        for ([_]bool{ false, true }) |graphed| {
            var want: [32]u8 = undefined;
            for ([_]bool{ true, false }) |legacy| {
                var pair = try both.pair(false);
                defer pair.deinit();
                var evidence: borrowed_fixture.Evidence = .{ .legacy = legacy, .mixed = storage == .mixed };
                defer evidence.deinit();
                const reply = try draftedReplyCompared(gpa, &both, &pair, &prompt, 12, settings, graphed, null, &evidence);
                defer gpa.free(reply.tokens);
                try testing.expectEqualSlices(i32, serial, reply.tokens);
                try testing.expect(evidence.blocks > 0);
                const digest = evidence.digest();
                if (legacy) want = digest else {
                    try testing.expectEqualSlices(u8, &want, &digest);
                    if (graphed) try testing.expectEqualSlices(u8, &eager, &digest) else eager = digest;
                    std.debug.print("spec-155 equality one storage={t} captured={} blocks={d} sha256={x}\n", .{ storage, graphed, evidence.blocks, digest });
                }
            }
        }
        const continued = try serialReply(gpa, both.m, &prompt, 12, settings);
        defer gpa.free(continued);
        try testing.expectEqualSlices(i32, serial, continued);
    }
}

// Production shapes exercise allocation accounting; the small shape deliberately changes splitK.
test "DFlash2 borrowed head accounting: production spans and tiny vocabulary metadata" {
    const bf16 = @import("../../../../src/weights/bf16.zig");
    const numeric = @import("../../../../src/weights/numeric.zig");
    const gpa = testing.allocator;
    for ([_]struct { n: usize, k: usize }{ .{ .n = 8192, .k = 256 }, .{ .n = rig.published.vocab, .k = rig.published.hidden } }) |shape| {
        var pool: cuda.Scratch = .init(gpa);
        defer pool.deinit();
        const bytes = try gpa.alloc(u16, shape.n * shape.k);
        defer gpa.free(bytes);
        for (bytes, 0..) |*value, i| value.* = numeric.bf16FromF32(@as(f32, @floatFromInt(i % 19)) / 32 - 0.25);
        var linear = try bf16.Linear.init(std.mem.sliceAsBytes(bytes), shape.n, shape.k, pool.rows(1));
        defer linear.deinit();
        var m: model_file.Model = undefined;
        m.shape = fx.shape;
        m.shape.vocab = shape.n;
        m.shape.hidden = shape.k;
        m.scratch = &pool;
        m.embed = linear.weight;
        m.head = .{ .runs = undefined, .count = 1, .n = shape.n, .k = shape.k, .max_rows = 1, .staging = null, .scratch = &pool };
        m.head.runs[0] = .{ .linear = .{ .bf16 = linear }, .at = 0 };
        var b: forward.Buffers = undefined;
        const half = shape.n / world;
        // Warm only the same partials that a newly initialized half would use.
        _ = try drafter_file.borrowHead(linear, 0, half, pool.rows(dflash2_pass.batch));
        const baseline = cuda.account.bytes;
        const made = cuda.account.made_ever;
        const tracing = cuda.account.trace;
        cuda.account.trace = true;
        defer cuda.account.trace = tracing;
        std.debug.print("spec-155 borrow begin n={d} bytes={d} allocations={d}\n", .{ shape.n, baseline, made });
        var target = try drafter_file.Of(world).one(gpa, m, &b);
        const mains = target.mains(null, null);
        try testing.expectEqual(baseline, cuda.account.bytes);
        try testing.expectEqual(made, cuda.account.made_ever);
        for (mains, 0..) |main, r| {
            const borrowed = main.head.bf16;
            const expected = linear.weight.span(r * half, half);
            try testing.expectEqual(expected.ptr, main.rows.ptr);
            try testing.expectEqual(expected.ptr, borrowed.weight.view.ptr);
            std.debug.print("spec-155 borrow rank={d} model_span={x} rows={x} proposer={x} bytes={d}\n", .{ r, @intFromPtr(expected.ptr), @intFromPtr(main.rows.ptr), @intFromPtr(borrowed.weight.view.ptr), borrowed.weight.bytes() });
            try testing.expectEqual(half * shape.k * 2, borrowed.weight.bytes());
            try testing.expectEqual(half, borrowed.n);
            try testing.expectEqual(shape.k, borrowed.k);
            try testing.expectEqual(bf16.splitK(half, shape.k), borrowed.sk);
            try testing.expectEqual(dflash2_pass.batch, borrowed.max_rows);
            try testing.expectEqual(&pool, borrowed.scratch);
            if (borrowed.sk > 1) try testing.expect(borrowed.part.?.bytes() >= borrowed.sk * dflash2_pass.batch * half * 4);
        }
        // An explicitly widened rescore view still wins, independent of the stored proposer.
        const swapped = target.mains(null, .{ mains[1].rows, mains[0].rows });
        try testing.expectEqual(mains[1].rows.ptr, swapped[0].rows.ptr);
        target.deinit();
        try testing.expectEqual(baseline, cuda.account.bytes);
        std.debug.print("spec-155 borrow end n={d} bytes={d} allocations={d}\n", .{ shape.n, cuda.account.bytes, cuda.account.made_ever });
        if (shape.n == 8192) try testing.expect(linear.sk != mains[0].head.bf16.sk);
        for ([_]bool{ false, true }) |stored| {
            var copies: [4]weights.Linear = undefined;
            const count: usize = if (stored) 4 else 2;
            for (copies[0..count], 0..) |*copy, r| {
                const offset = (r % world) * half * shape.k;
                copy.* = try .init(gpa, gpa, &.{std.mem.sliceAsBytes(bytes[offset..][0 .. half * shape.k])}, shape.k, .bf16, pool.rows(dflash2_pass.batch));
            }
            if (!stored and shape.n == 8192) try expectBorrowedProducts(mains[0].head, copies[0]);
            const before = cuda.account.bytes;
            for (copies[0..count]) |*copy| copy.deinit();
            const after = cuda.account.bytes;
            try testing.expectEqual(count * half * shape.k * 2, before - after);
            try testing.expectEqual(baseline, after);
            std.debug.print("spec-155 accounting n={d} k={d} stored={} before={d} after={d} saved={d} half={d} borrowed_allocations=0\n", .{ shape.n, shape.k, stored, before, after, before - after, half * shape.k * 2 });
        }
        if (shape.n == rig.published.vocab) inline for (.{ 1, 2 }) |held| {
            const H = @import("../../../../src/models/qwen4/ranks_forward.zig").Held(held);
            const first: usize = if (held == 1) 1 else 0;
            var shares: H.Model = .{ .ranks = undefined, .max_rows = 1, .first = first };
            for (&shares.ranks, 0..) |*share, rank| {
                share.* = m;
                share.head.n = half;
                share.head.runs[0].linear.bf16.weight = .{ .view = linear.weight.span((first + rank) * half, half) };
                share.head.runs[0].linear.bf16.n = half;
            }
            var held_b: H.Buffers = undefined;
            const held_before = cuda.account.bytes;
            const held_made = cuda.account.made_ever;
            var held_target = try drafter_file.Of(held).heldBy(gpa, shares, &held_b);
            for (held_target.mains(null, null), shares.ranks, 0..) |main, share, rank| {
                const own = share.head.runs[0].linear.bf16.weight;
                try testing.expectEqual(own.view.ptr, main.rows.ptr);
                try testing.expectEqual(own.view.ptr, main.head.bf16.weight.view.ptr);
                try testing.expectEqual(own.bytes(), main.head.bf16.weight.bytes());
                std.debug.print("spec-155 borrow held={d} rank={d} model_span={x} rows={x} proposer={x} bytes={d}\n", .{ held, first + rank, @intFromPtr(own.view.ptr), @intFromPtr(main.rows.ptr), @intFromPtr(main.head.bf16.weight.view.ptr), own.bytes() });
            }
            held_target.deinit();
            try testing.expectEqual(held_before, cuda.account.bytes);
            try testing.expectEqual(held_made, cuda.account.made_ever);
            for ([_]bool{ false, true }) |stored| {
                var copies: [4]weights.Linear = undefined;
                const count: usize = held * @as(usize, if (stored) 2 else 1);
                for (copies[0..count], 0..) |*copy, r| {
                    const offset = (first + r % held) * half * shape.k;
                    copy.* = try .init(gpa, gpa, &.{std.mem.sliceAsBytes(bytes[offset..][0 .. half * shape.k])}, shape.k, .bf16, pool.rows(dflash2_pass.batch));
                }
                const before = cuda.account.bytes;
                for (copies[0..count]) |*copy| copy.deinit();
                try testing.expectEqual(count * half * shape.k * 2, before - cuda.account.bytes);
                std.debug.print("spec-155 accounting held={d} stored={} before={d} after={d} saved={d} per_rank={d} borrowed_allocations=0\n", .{ held, stored, before, cuda.account.bytes, before - cuda.account.bytes, (before - cuda.account.bytes) / held });
            }
        };
        // Calls wider than the original head's one-row capacity use the sliced plan's own capacity.
        var x = try cuda.Buffer.init(.bf16, &.{ 7, shape.k });
        defer x.deinit();
        try x.zero();
        var out = try cuda.Buffer.init(.bf16, &.{ 7, half });
        defer out.deinit();
        try mains[0].head.forward(x.view, out.view);
        try cuda.synchronize();
    }
}

test "DFlash2 borrowed head ownership: partial finish failures retain model and free four-bit heads" {
    const gpa = testing.allocator;
    var both: Both = undefined;
    try both.init(64);
    defer both.deinit();
    var pair = try both.pair(false);
    defer pair.deinit();
    var b: forward.Buffers = try .init(gpa, both.m, both.m.max_rows, both.capacity);
    defer b.deinit();
    try b.setTaps(both.c.taps, .f32_acc);
    const prompt = [_]i32{ 5, 17, 200, 3 };
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const serial = try serialReply(gpa, both.m, &prompt, 8, settings);
    defer gpa.free(serial);
    for ([_]bool{ true, false }) |stored| {
        for (0..6) |fail_at| {
            const before = cuda.account.bytes;
            const heads = try rig.draftHeads(world, both.arena_state.allocator(), gpa, both.m, stored, both.m.scratch.rows(dflash2_pass.batch));
            try testing.expectEqual(stored, heads == null);
            const target = try drafter_file.Of(world).one(gpa, both.m, &b);
            var failing: testing.FailingAllocator = .init(gpa, .{ .fail_index = fail_at });
            var flash: rig.Rig(world) = undefined;
            flash.pair = pair;
            flash.books = both.books;
            try testing.expectError(error.OutOfMemory, flash.finish(failing.allocator(), &both.c, target, heads, null, settings));
            try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            try testing.expectEqual(before, cuda.account.bytes);
        }
        // Fail before finish, after supplied four-bit heads already exist.
        const before = cuda.account.bytes;
        var fresh: forward.Buffers = try .init(gpa, both.m, both.m.max_rows, both.capacity);
        const heads = try rig.draftHeads(world, both.arena_state.allocator(), gpa, both.m, stored, both.m.scratch.rows(dflash2_pass.batch));
        var failing: testing.FailingAllocator = .init(gpa, .{ .fail_index = 0 });
        var flash: rig.Rig(world) = undefined;
        flash.w = both.w;
        var opened: rig.Opened = .{ .config = both.c };
        try testing.expectError(error.OutOfMemory, flash.init(failing.allocator(), &opened, heads, null, both.m, &fresh, 64, settings, true));
        fresh.deinit();
        try testing.expectEqual(before, cuda.account.bytes);
    }
    const continued = try serialReply(gpa, both.m, &prompt, 8, settings);
    defer gpa.free(continued);
    try testing.expectEqualSlices(i32, serial, continued);
}

test "DFlash2 borrowed head rig teardown: stored and mixed leave main decoding alive" {
    const gpa = testing.allocator;
    for ([_]weights.Storage{ .stored, .mixed }) |storage| {
        var both: Both = undefined;
        try both.initStored(64, storage);
        // The rig takes the weights/books; the main model and arena stay here.
        defer both.arena_state.deinit();
        defer both.m.deinit();
        const prompt = [_]i32{ 5, 17, 200, 3 };
        const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
        const serial = try serialReply(gpa, both.m, &prompt, 8, settings);
        defer gpa.free(serial);
        var b: forward.Buffers = try .init(gpa, both.m, both.m.max_rows, both.capacity);
        defer b.deinit();
        var flash: rig.Rig(world) = undefined;
        flash.w = both.w;
        flash.books = both.books;
        flash.gpa = gpa;
        var opened: rig.Opened = .{ .config = both.c };
        const heads = try rig.draftHeads(world, both.arena_state.allocator(), gpa, both.m, storage == .stored, both.m.scratch.rows(dflash2_pass.batch));
        try flash.init(gpa, &opened, heads, null, both.m, &b, 64, settings, false);
        const main = flash.d.mains();
        try testing.expectEqual(both.m.head.runs[0].linear.bf16.weight.view.ptr, main[0].rows.ptr);
        flash.deinit();
        const continued = try serialReply(gpa, both.m, &prompt, 8, settings);
        defer gpa.free(continued);
        try testing.expectEqualSlices(i32, serial, continued);
    }
}

fn expectBorrowedProducts(borrowed: weights.Linear, copied: weights.Linear) !void {
    const gpa = testing.allocator;
    const k = borrowed.bf16.k;
    const n = borrowed.bf16.n;
    const host = try gpa.alloc(u16, 7 * k);
    defer gpa.free(host);
    for (host, 0..) |*value, i| value.* = @import("../../../../src/weights/numeric.zig").bf16FromF32(@as(f32, @floatFromInt(i % 23)) / 32 - 0.25);
    var x = try cuda.Buffer.init(.bf16, &.{ 7, k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(host));
    var got = try cuda.Buffer.init(.bf16, &.{ 7, n });
    defer got.deinit();
    var want = try cuda.Buffer.init(.bf16, &.{ 7, n });
    defer want.deinit();
    try copied.forward(x.view, want.view);
    try borrowed.forward(x.view, got.view);
    try cuda.synchronize();
    const expected = try gpa.alloc(u16, 7 * n);
    defer gpa.free(expected);
    const actual = try gpa.alloc(u16, 7 * n);
    defer gpa.free(actual);
    try want.download(std.mem.sliceAsBytes(expected));
    try got.download(std.mem.sliceAsBytes(actual));
    try testing.expectEqualSlices(u16, expected, actual);
    for (0..7) |r| try borrowed.forward(x.span(r, 1), got.span(r, 1));
    try cuda.synchronize();
    try got.download(std.mem.sliceAsBytes(actual));
    try testing.expectEqualSlices(u16, expected, actual);
}

test "DFlash2 borrowed head allocation measurement: original constructors on production dimensions" {
    const gpa = testing.allocator;
    const n = rig.published.vocab;
    const k = rig.published.hidden;
    const half = n / world;
    var pool: cuda.Scratch = .init(gpa);
    defer pool.deinit();
    const bytes = try gpa.alloc(u8, n * k * 2);
    defer gpa.free(bytes);
    @memset(bytes, 0);
    var m: model_file.Model = undefined;
    m.shape = fx.shape;
    m.shape.vocab = n;
    m.shape.hidden = k;
    m.scratch = &pool;
    m.head = try @import("../../../../src/weights/face.zig").Face.init(bytes, n, k, pool.rows(dflash2_pass.batch));
    defer m.head.deinit();
    m.embed = m.head.runs[0].linear.bf16.weight;
    const traced = cuda.account.trace;
    cuda.account.trace = true;
    defer cuda.account.trace = traced;
    for ([_]bool{ false, true }) |stored| {
        const before = cuda.account.bytes;
        const made = cuda.account.made_ever;
        std.debug.print("spec-155 live begin kind=one stored={} bytes={d} allocations={d}\n", .{ stored, before, made });
        var supplied: ?[world]weights.Linear = if (stored) try rig.draftHeads(world, gpa, gpa, m, true, pool.rows(dflash2_pass.batch)) else null;
        var b: forward.Buffers = undefined;
        var target = try drafter_file.Of(world).one(gpa, m, &b);
        for (target.mains(supplied, null), 0..) |main, rank| {
            const expected = m.head.runs[0].linear.bf16.weight.span(rank * half, half);
            std.debug.print("spec-155 live pointer kind=one rank={d} stored={} model={x} rows={x} proposer={x} span={d}\n", .{ rank, stored, @intFromPtr(expected.ptr), @intFromPtr(main.rows.ptr), @intFromPtr(main.head.bf16.weight.view.ptr), half * k * 2 });
        }
        std.debug.print("spec-155 live loaded kind=one stored={} bytes={d} added={d} allocations={d}\n", .{ stored, cuda.account.bytes, cuda.account.bytes - before, cuda.account.made_ever - made });
        target.deinit();
        if (supplied) |*heads| for (heads) |*head| head.deinit();
        try testing.expectEqual(before, cuda.account.bytes);
    }
    inline for (.{ 1, 2 }) |held| {
        const H = @import("../../../../src/models/qwen4/ranks_forward.zig").Held(held);
        const first: usize = if (held == 1) 1 else 0;
        var shares: H.Model = .{ .ranks = undefined, .max_rows = dflash2_pass.batch, .first = first };
        for (&shares.ranks, 0..) |*share, rank| {
            share.* = m;
            share.head.n = half;
            share.head.runs[0].linear.bf16.weight = .{ .view = m.head.runs[0].linear.bf16.weight.span((first + rank) * half, half) };
            share.head.runs[0].linear.bf16.n = half;
        }
        for ([_]bool{ false, true }) |stored| {
            const before = cuda.account.bytes;
            const made = cuda.account.made_ever;
            std.debug.print("spec-155 live begin kind=held{d} stored={} bytes={d} allocations={d}\n", .{ held, stored, before, made });
            var supplied: ?[held]weights.Linear = if (stored) try rig.draftHeadsHeld(held, gpa, gpa, shares, true, pool.rows(dflash2_pass.batch)) else null;
            var b: H.Buffers = undefined;
            var target = try drafter_file.Of(held).heldBy(gpa, shares, &b);
            for (target.mains(supplied, null), shares.ranks, 0..) |main, share, rank| {
                std.debug.print("spec-155 live pointer kind=held{d} rank={d} stored={} model={x} rows={x} proposer={x} span={d}\n", .{ held, first + rank, stored, @intFromPtr(share.head.runs[0].linear.bf16.weight.view.ptr), @intFromPtr(main.rows.ptr), @intFromPtr(main.head.bf16.weight.view.ptr), half * k * 2 });
            }
            std.debug.print("spec-155 live loaded kind=held{d} stored={} bytes={d} added={d} allocations={d}\n", .{ held, stored, cuda.account.bytes, cuda.account.bytes - before, cuda.account.made_ever - made });
            target.deinit();
            if (supplied) |*heads| for (heads) |*head| head.deinit();
            try testing.expectEqual(before, cuda.account.bytes);
        }
    }
}
