//! `mtp.zig`'s tests on the random GLM of `random.zig`: a first step phase by phase (the projection
//! against fp64, each norm and residual the host's bits, each branch the MTP layer's weights loaded alone, the
//! logits the head's); rows absorbed at once and one by one, a chained draft and `trim`; the head's drafts
//! against the steps by hand; replies drafted by it against serial steps; and its drafts chosen from the
//! halves' starts against those chosen from whole rows, under the main head and a four-bit copy of it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const masks_file = @import("../../../../src/decode/masks.zig");
const decode_draft = @import("../../../../src/decode/draft.zig");
const rig = @import("../../../../src/models/glm5/dflash2/rig.zig");
const draft_file = @import("../../../../src/models/glm5/draft.zig");
const dsa = @import("../../../../src/models/glm5/layers/dsa.zig");
const host = @import("../../../../src/models/glm5/layers/glue_ref.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const loopback = @import("../../../../src/models/glm5/layers/loopback.zig");
const mlp = @import("../../../../src/models/glm5/layers/mlp.zig");
const mtp_file = @import("../../../../src/models/glm5/mtp/mtp.zig");
const shared = @import("../testing.zig");
const Fixture = shared.Fixture;
const fx = @import("../../../../src/models/glm5/checks/random.zig").fx;
const Pair = mtp_file.Pair;
const Head = mtp_file.Head;
const world = layout.world;

const mtp_layer = layout.prefix ++ "layers.5.";

fn val(bits: u16) f64 {
    return numeric.f32FromBf16(bits);
}

/// A tensor of the fixture's as a host slice.
fn tensor(comptime T: type, arena: Allocator, fixture: *Fixture, name: []const u8) ![]T {
    const bytes = fixture.source.get(name).?.data;
    const values = try arena.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(values), bytes);
    return values;
}

/// The first `rows` rows of a buffer of rows `width` wide.
fn fetchRows(comptime T: type, arena: Allocator, buffer: cuda.Buffer, rows: usize, width: usize) ![]T {
    return (try shared.fetch(T, arena, buffer))[0 .. rows * width];
}

/// The latents a cache holds, bf16 [len, 512].
fn latents(arena: Allocator, cache: dsa.Cache) ![]u16 {
    return fetchRows(u16, arena, cache.rows, cache.pos, dsa.latent);
}

/// The main model's first `rows` ids as one window, every final row left in its buffers.
fn runMain(loop: *shared.Loop, tokens: []const i32) !void {
    try loop.state.reset();
    try loopback.forwardScoring(loop.m, &loop.state, &loop.b, tokens, .none);
}

test "a first step phase by phase: the projection of both norms, the first embedding zero, each branch the layer's own, the head's logits" {
    const rows = 3;
    const capacity = 16;
    const gpa = testing.allocator;
    var fixture: Fixture = undefined;
    try fixture.init(71, rows + 1);
    defer fixture.deinit();
    const arena = fixture.arena_state.allocator();
    const c = &fixture.c;
    try testing.expectEqual(@as(usize, 5), c.layers);
    var loop = try fixture.loop(capacity);
    defer loop.deinit();
    var pair: Pair = try .init(arena, gpa, fixture.source, c, loop.m, capacity, fixture.rows());
    defer pair.deinit();
    try runMain(&loop, fixture.tokens[0..rows]);
    const next = fixture.tokens[1..];
    const eps: f32 = @floatCast(c.eps);

    const table = try tensor(u16, arena, &fixture, layout.prefix ++ "embed_tokens.weight");
    const eh = try tensor(u16, arena, &fixture, mtp_layer ++ "eh_proj.weight");
    const lm_head = try tensor(u16, arena, &fixture, "lm_head.weight");
    const enorm = try tensor(u16, arena, &fixture, mtp_layer ++ "enorm.weight");
    const hnorm = try tensor(u16, arena, &fixture, mtp_layer ++ "hnorm.weight");
    const in_norm = try tensor(u16, arena, &fixture, mtp_layer ++ "input_layernorm.weight");
    const post_norm = try tensor(u16, arena, &fixture, mtp_layer ++ "post_attention_layernorm.weight");
    const head_norm = try tensor(u16, arena, &fixture, mtp_layer ++ "shared_head.norm.weight");
    const scratch = try arena.alloc(f32, fx.dims);
    const want_row = try arena.alloc(u16, fx.dims);
    const cat = try arena.alloc(u16, 2 * fx.dims);
    // the host's stream, the device's bits once the first phase's are taken
    var x: []u16 = undefined;
    var alone: cuda.Buffer = try .init(.f32, &.{ rows, fx.dims });
    defer alone.deinit();
    const partialOf = struct {
        fn of(a: Allocator, gathered: cuda.Buffer, rank: usize) ![]f32 {
            const all = try shared.fetch(f32, a, gathered);
            return all[rank * rows * fx.dims ..][0 .. rows * fx.dims];
        }
    }.of;

    for (loop.m.ranks, pair.weights, pair.caches, &pair.b, loop.b.ranks, 0..) |m, w, cache, *b, main, rank| {
        try b.ids.uploadAt(0, std.mem.sliceAsBytes(next[0..rows]));
        try mtp_file.attention(m, w, cache, b, main.dsa.?, main.final.rows(rows), true, loopback.partial(loop.b.attn, rank, rows));
        const finals = try fetchRows(u16, arena, main.final, rows, fx.dims);
        const got = try fetchRows(u16, arena, b.x, rows, fx.dims);
        const normed = try fetchRows(u16, arena, b.normed, rows, fx.dims);
        for (0..rows) |r| {
            const id: usize = @intCast(next[r]);
            const me = try arena.dupe(u16, table[id * fx.dims ..][0..fx.dims]);
            if (r == 0) @memset(me, 0);
            host.rmsnorm(me, .{ .bf16 = enorm }, cat[0..fx.dims], eps, scratch);
            host.rmsnorm(finals[r * fx.dims ..][0..fx.dims], .{ .bf16 = hnorm }, cat[fx.dims..], eps, scratch);
            if (r == 0) for (cat[0..fx.dims]) |bits| try testing.expectEqual(@as(f64, 0), val(bits));
            var size: f64 = 0;
            for (got[r * fx.dims ..][0..fx.dims], 0..) |bits, d| {
                var sum: f64 = 0;
                var mags: f64 = 0;
                for (cat, eh[d * 2 * fx.dims ..][0 .. 2 * fx.dims]) |in, weight| {
                    sum += val(in) * val(weight);
                    mags += @abs(val(in) * val(weight));
                }
                size += sum * sum;
                try testing.expectApproxEqAbs(sum, val(bits), @abs(sum) / 128.0 + mags * 1e-5);
            }
            try testing.expect(@sqrt(size / fx.dims) > 0.3);
            host.rmsnorm(got[r * fx.dims ..][0..fx.dims], .{ .bf16 = in_norm }, want_row, eps, scratch);
            try testing.expectEqualSlices(u16, want_row, normed[r * fx.dims ..][0..fx.dims]);
        }
        // both ranks hold the projection whole
        if (rank == 0) x = got else try testing.expectEqualSlices(u16, x, got);

        var weights: dsa.Weights = try .load(arena, fixture.source, mtp_layer ++ "self_attn.", c, rank, fixture.rows());
        defer weights.deinit();
        var fresh: dsa.Cache = try weights.cacheFor(1, capacity);
        defer fresh.deinit();
        var buffers: dsa.Buffers = try .init(weights, capacity, fx.max_rows);
        defer buffers.deinit();
        try weights.forward(fresh, 0, buffers, b.normed.rows(rows), alone.view);
        try testing.expectEqualSlices(f32, try shared.fetch(f32, arena, alone), try partialOf(arena, loop.b.attn, rank));
    }

    const residual = struct {
        fn add(a: Allocator, stream: []u16, gathered: cuda.Buffer) !void {
            var ranks: [world][]const f32 = undefined;
            for (0..rows) |r| {
                for (&ranks, 0..) |*one, rank| one.* = (try partialOf(a, gathered, rank))[r * fx.dims ..][0..fx.dims];
                host.residualAdd(stream[r * fx.dims ..][0..fx.dims], &ranks);
            }
        }
    }.add;
    try residual(arena, x, loop.b.attn);
    for (pair.weights, &pair.b, loop.b.ranks, 0..) |w, *b, main, rank| {
        try mtp_file.moe(w, b, main.moe.?, rows, loopback.gathered(loop.b.attn, rows), loopback.partial(loop.b.ffn, rank, rows));
        try testing.expectEqualSlices(u16, x, try fetchRows(u16, arena, b.x, rows, fx.dims));
        const normed = try fetchRows(u16, arena, b.normed, rows, fx.dims);
        for (0..rows) |r| {
            host.rmsnorm(x[r * fx.dims ..][0..fx.dims], .{ .bf16 = post_norm }, want_row, eps, scratch);
            try testing.expectEqualSlices(u16, want_row, normed[r * fx.dims ..][0..fx.dims]);
        }
        var weights: mlp.Moe = try .load(arena, gpa, fixture.source, c, c.layers, rank, fixture.rows());
        defer weights.deinit();
        var buffers: mlp.Moe.Buffers = try .init(weights, fx.max_rows);
        defer buffers.deinit();
        try weights.forward(buffers, b.normed.rows(rows), alone.view, 0);
        try testing.expectEqualSlices(f32, try shared.fetch(f32, arena, alone), try partialOf(arena, loop.b.ffn, rank));
    }

    try residual(arena, x, loop.b.ffn);
    host.rmsnorm(x[(rows - 1) * fx.dims ..][0..fx.dims], .{ .bf16 = head_norm }, want_row, eps, scratch);
    const half = fx.vocab / world;
    var phased: [world][]u16 = undefined;
    for (loop.m.ranks, pair.weights, &pair.b, 0..) |m, w, *b, rank| {
        try mtp_file.finish(w, b, rows, loopback.gathered(loop.b.ffn, rows));
        try mtp_file.score(m.head, b);
        try testing.expectEqualSlices(u16, x, try fetchRows(u16, arena, b.x, rows, fx.dims));
        try testing.expectEqualSlices(u16, want_row, try shared.fetch(u16, arena, b.final));
        const logits = try shared.fetch(u16, arena, b.logits);
        const want = try arena.alloc(f64, half);
        var size: f64 = 0;
        for (want, 0..) |*one, j| {
            one.* = 0;
            for (want_row, 0..) |f, d| one.* += val(f) * val(lm_head[(rank * half + j) * fx.dims + d]);
            size += one.* * one.*;
        }
        const rms = @sqrt(size / half);
        try testing.expect(rms > 0.1);
        for (want, logits) |one, g| try testing.expectApproxEqAbs(one, val(g), 1e-3 * rms + @abs(one) / 128.0);
        phased[rank] = logits;
    }

    // `step` is these phases, and leaves the entries to `setLen`
    try pair.step(loop.m, &loop.b, .{ .main = 0 }, next[0..rows], true);
    try testing.expectEqual(@as(usize, 0), pair.len());
    for (pair.b, phased) |b, want| try testing.expectEqualSlices(u16, want, try shared.fetch(u16, arena, b.logits));
    const joined = try arena.alloc(u16, fx.vocab);
    try pair.logitsRow(joined);
    try testing.expectEqualSlices(u16, phased[0], joined[0..half]);
    try testing.expectEqualSlices(u16, phased[1], joined[half..]);
}

test "rows absorbed at once and one by one leave the same entries; a chained draft reads the layer's own row; trim drops it" {
    const rows = 7;
    const capacity = 16;
    const gpa = testing.allocator;
    var fixture: Fixture = undefined;
    try fixture.init(73, rows + 2);
    defer fixture.deinit();
    const arena = fixture.arena_state.allocator();
    var loop = try fixture.loop(capacity);
    defer loop.deinit();
    var pair: Pair = try .init(arena, gpa, fixture.source, &fixture.c, loop.m, capacity, fixture.rows());
    defer pair.deinit();
    try runMain(&loop, fixture.tokens[0..rows]);
    const next = fixture.tokens[1..];
    const logits = try arena.alloc(u16, fx.vocab);
    const again = try arena.alloc(u16, fx.vocab);

    try pair.step(loop.m, &loop.b, .{ .main = 0 }, next[0..rows], true);
    try pair.setLen(rows);
    try pair.logitsRow(logits);
    const whole = try latents(arena, pair.caches[0]);
    const final = try shared.fetch(u16, arena, pair.b[0].final);
    // the latents and the hidden row are whole on each rank
    try testing.expectEqualSlices(u16, whole, try latents(arena, pair.caches[1]));
    try testing.expectEqualSlices(u16, final, try shared.fetch(u16, arena, pair.b[1].final));
    var size: f64 = 0;
    for (whole) |bits| size += val(bits) * val(bits);
    try testing.expect(size > 1);

    try pair.setLen(0);
    for (0..rows) |r| {
        try pair.step(loop.m, &loop.b, .{ .main = r }, next[r..][0..1], r == rows - 1);
        try pair.setLen(r + 1);
    }
    try pair.logitsRow(again);
    try testing.expectEqualSlices(u16, logits, again);
    try testing.expectEqualSlices(u16, whole, try latents(arena, pair.caches[0]));
    try testing.expectEqualSlices(u16, final, try shared.fetch(u16, arena, pair.b[0].final));
    // and in two steps that cut the rows unevenly
    try pair.setLen(0);
    try pair.step(loop.m, &loop.b, .{ .main = 0 }, next[0..2], false);
    try pair.setLen(2);
    try pair.step(loop.m, &loop.b, .{ .main = 2 }, next[2..rows], true);
    try pair.setLen(rows);
    try pair.logitsRow(again);
    try testing.expectEqualSlices(u16, logits, again);
    try testing.expectEqualSlices(u16, whole, try latents(arena, pair.caches[0]));

    // a chained draft: the layer's own last row as the hidden row, an entry after the absorbed ones
    const drafted = [1]i32{next[rows]};
    try pair.step(loop.m, &loop.b, .own, &drafted, true);
    try pair.setLen(rows + 1);
    pair.drafted += 1;
    try pair.logitsRow(logits);
    const chained = try latents(arena, pair.caches[0]);
    try testing.expectEqualSlices(u16, whole, chained[0..whole.len]);
    try testing.expect(!std.mem.eql(u16, logits, again));
    // the same step with that row where the main model's rows are
    try pair.trim();
    try testing.expectEqual(@as(usize, rows), pair.len());
    try testing.expectEqual(@as(usize, 0), pair.drafted);
    for (loop.b.ranks) |main| try main.final.uploadAt(3 * fx.dims * 2, std.mem.sliceAsBytes(final));
    try pair.step(loop.m, &loop.b, .{ .main = 3 }, &drafted, true);
    try pair.setLen(rows + 1);
    try pair.logitsRow(again);
    try testing.expectEqualSlices(u16, logits, again);
    try testing.expectEqualSlices(u16, chained, try latents(arena, pair.caches[0]));
}

const Drafted = struct {
    fixture: Fixture,
    loop: shared.Loop,
    pair: Pair,
    sampler: sampling.Sampler,
    head: Head,

    fn init(self: *Drafted, seed: u64, prompt: usize, capacity: usize, settings: sampling.Sampling) !void {
        const gpa = testing.allocator;
        try self.fixture.init(seed, prompt);
        errdefer self.fixture.deinit();
        self.loop = try self.fixture.loop(capacity);
        errdefer self.loop.deinit();
        self.pair = try .init(self.fixture.arena_state.allocator(), gpa, self.fixture.source, &self.fixture.c, self.loop.m, capacity, self.fixture.rows());
        errdefer self.pair.deinit();
        self.sampler = try .init(gpa, fx.vocab);
        errdefer self.sampler.deinit(gpa);
        self.head = try .init(gpa, self.loop.m, &self.pair, &self.loop.b, &self.sampler, settings, fx.vocab);
    }

    fn deinit(self: *Drafted) void {
        self.head.deinit();
        self.sampler.deinit(testing.allocator);
        self.pair.deinit();
        self.loop.deinit();
        self.fixture.deinit();
    }
};

test "the head's drafts are its steps by hand: the kept rows absorbed, each draft chosen by the key of its place from the layer's own row" {
    const prompt = 9;
    const depth = 5;
    const capacity = 32;
    const settings: sampling.Sampling = .{ .seed = 5, .temperature = 1.3, .top_k = 0, .top_p = 1, .min_p = 0 };
    var d: Drafted = undefined;
    try d.init(79, prompt + 1, capacity, settings);
    defer d.deinit();
    const arena = d.fixture.arena_state.allocator();
    d.head.confidence = 0;
    try d.head.reset(settings);
    // the tokens chosen after the prompt's rows: the prompt's own, and a first token of the reply
    const chosen = d.fixture.tokens[1..];
    try runMain(&d.loop, d.fixture.tokens[0..prompt]);
    var out: [depth]i32 = undefined;
    const drafter = d.head.drafter();
    try testing.expectEqual(@as(usize, depth), try drafter.draft(drafter.context, draft_file.namespace.streams(d.loop.b, prompt), chosen, prompt, null, &out));
    try testing.expectEqual(@as(usize, prompt + depth - 1), d.pair.len());
    try testing.expectEqual(@as(usize, depth - 1), d.pair.drafted);

    var by_hand: Pair = try .init(arena, testing.allocator, d.fixture.source, &d.fixture.c, d.loop.m, capacity, d.fixture.rows());
    defer by_hand.deinit();
    const logits = try arena.alloc(u16, fx.vocab);
    var want: [depth]i32 = undefined;
    try by_hand.step(d.loop.m, &d.loop.b, .{ .main = 0 }, chosen, true);
    try by_hand.setLen(prompt);
    for (&want, 0..) |*token, n| {
        if (n > 0) {
            try by_hand.step(d.loop.m, &d.loop.b, .own, want[n - 1 ..][0..1], true);
            try by_hand.setLen(prompt + n);
        }
        try by_hand.logitsRow(logits);
        token.* = try d.sampler.choose(logits, prompt + 1 + n, settings);
    }
    try testing.expectEqualSlices(i32, &want, &out);
    // sampled drafts that a key of another place would not give
    var distinct: usize = 0;
    for (want[1..], want[0 .. depth - 1]) |a, b| distinct += @intFromBool(a != b);
    try testing.expect(distinct >= 2);
    // the next call drops the chained entries and absorbs after the kept rows
    try d.loop.state.reset();
    try loopback.forwardScoring(d.loop.m, &d.loop.state, &d.loop.b, d.fixture.tokens[0..prompt], .none);
    try loopback.commit(d.loop.m, &d.loop.state, d.loop.b, d.fixture.tokens[0..prompt], prompt);
    const one = [1]i32{chosen[prompt - 1]};
    try loopback.forwardScoring(d.loop.m, &d.loop.state, &d.loop.b, &one, .none);
    try testing.expectEqual(@as(usize, 1), try drafter.draft(drafter.context, draft_file.namespace.streams(d.loop.b, 1), want[0..1], prompt + 1, null, out[0..1]));
    try testing.expectEqual(@as(usize, prompt + 1), d.pair.len());
    try testing.expectEqual(@as(usize, 0), d.pair.drafted);
}

test "replies drafted by the MTP layer are the serial replies, greedy and sampled, graphed or not, at any confidence" {
    const gpa = testing.allocator;
    const count = 40;
    const prompt = 23;
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
    var d: Drafted = undefined;
    try d.init(83, prompt, prompt + count + draft_file.forward_max_rows, greedy);
    defer d.deinit();
    var graphs: loopback.Graphs = try .init(gpa);
    defer graphs.deinit();
    const none: []const u32 = &.{};
    const every = try d.fixture.arena_state.allocator().alloc(u32, fx.vocab);
    for (every, 0..) |*token, id| token.* = @intCast(id);
    for ([_]sampling.Sampling{ greedy, .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 } }) |settings| {
        const serial = try shared.serialReply(gpa, d.loop.m, fx.vocab, d.fixture.tokens, count, settings);
        defer gpa.free(serial);
        var drafted: [4]usize = undefined;
        const Case = struct { confidence: f32, graphed: bool, ends: []const u32 };
        for ([_]Case{
            .{ .confidence = 0, .graphed = false, .ends = none },
            .{ .confidence = 0, .graphed = true, .ends = none },
            .{ .confidence = 2, .graphed = true, .ends = none },
            .{ .confidence = 0, .graphed = false, .ends = every },
        }, &drafted) |case, *ran| {
            d.head.confidence = case.confidence;
            d.head.ends = case.ends;
            try d.head.reset(settings);
            const got = try shared.reply(gpa, &d.loop, d.fixture.tokens, count, settings, none, d.head.drafter(), if (case.graphed) &graphs else null);
            defer gpa.free(got.tokens);
            try testing.expectEqualSlices(i32, serial, got.tokens);
            try testing.expectEqual(count - 1, got.stats.rounds + got.stats.accepted);
            try testing.expect(!d.head.full);
            ran.* = got.stats.drafted;
            // a chain ends with a draft no probability carries, or that ends the reply: one a round
            if (case.confidence > 1 or case.ends.len > 0) try testing.expect(got.stats.drafted <= got.stats.rounds and got.stats.drafted + 2 >= got.stats.rounds);
        }
        // the same drafts through a graph, and whole chains where nothing ends them
        try testing.expectEqual(drafted[0], drafted[1]);
        try testing.expect(drafted[0] > 3 * drafted[2]);
    }

    // a cap cuts the chains to it, the replies still the serial ones (spec 099 WS-04). The chains run long
    // here — confidence 0 and no end tokens, which an earlier case leaves set and `reset` keeps — so the
    // bound is one the uncapped run refuses: without the cap the rounds carry fifteen drafts
    d.head.confidence = 0;
    d.head.ends = none;
    var uncapped: usize = 0;
    for ([_]usize{ mtp_file.Head.default_cap, 1 }) |cap| {
        d.head.cap = cap;
        try d.head.reset(greedy);
        const serial = try shared.serialReply(gpa, d.loop.m, fx.vocab, d.fixture.tokens, count, greedy);
        defer gpa.free(serial);
        const got = try shared.reply(gpa, &d.loop, d.fixture.tokens, count, greedy, none, d.head.drafter(), null);
        defer gpa.free(got.tokens);
        try testing.expectEqualSlices(i32, serial, got.tokens);
        if (cap == 1) {
            try testing.expect(got.stats.drafted <= got.stats.rounds);
            try testing.expect(got.stats.drafted + 4 * got.stats.rounds < uncapped);
        } else uncapped = got.stats.drafted;
    }
    d.head.cap = mtp_file.Head.default_cap;
    d.head.confidence = mtp_file.Head.default_confidence;
}

/// A drafter that keeps every call's drafts of another's, each call's count before them.
const Recorder = struct {
    inner: decode_draft.Drafter,
    all: *std.ArrayList(i32),

    fn drafter(self: *Recorder) decode_draft.Drafter {
        return .{ .context = self, .draft = drafts };
    }

    fn drafts(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Recorder = @ptrCast(@alignCast(context));
        const count = try self.inner.draft(self.inner.context, streams, tokens, position, guide, out);
        try self.all.append(testing.allocator, @intCast(count));
        try self.all.appendSlice(testing.allocator, out[0..count]);
        return count;
    }
};

test "drafts chosen from the halves' starts are the whole rows', under any sampling and confidence, by the main head and by a four-bit copy; the replies are the serial replies" {
    const gpa = testing.allocator;
    const count = 40;
    const prompt = 23;
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
    var d: Drafted = undefined;
    try d.init(89, prompt, prompt + count + draft_file.forward_max_rows, greedy);
    defer d.deinit();
    const arena = d.fixture.arena_state.allocator();
    var heads = try rig.draftHeads(world, arena, gpa, d.loop.m, d.fixture.rows(), null);
    defer for (&heads) |*one| one.deinit();
    const none: []const u32 = &.{};
    const Case = struct { settings: sampling.Sampling, device: bool };
    const cases = [_]Case{
        .{ .settings = greedy, .device = true },
        .{ .settings = .{ .seed = 11, .temperature = 0.7, .top_k = 20, .top_p = 0.95, .min_p = 0 }, .device = true },
        .{ .settings = .{ .seed = 12, .temperature = 1.5, .top_k = 12, .top_p = 1, .min_p = 0 }, .device = true },
        .{ .settings = .{ .seed = 13, .temperature = 0.9, .top_k = 0, .top_p = 0.9, .min_p = 0 }, .device = false },
        // a nucleus asks for more of a start than half of this vocabulary; with no cut the whole row is the sampler's
        .{ .settings = .{ .seed = 14, .temperature = 1.3, .top_k = 0, .top_p = 1, .min_p = 0 }, .device = false },
    };
    var replies: [2][cases.len][]i32 = undefined;
    var made: usize = 0;
    defer for (0..made) |at| gpa.free(replies[at / cases.len][at % cases.len]);
    for ([_]?[world]mtp_file.Linear{ null, heads }, 0..) |scored_by, which| {
        d.pair.heads = scored_by;
        for (cases, 0..) |case, index| {
            const serial = try shared.serialReply(gpa, d.loop.m, fx.vocab, d.fixture.tokens, count, case.settings);
            replies[which][index] = serial;
            made += 1;
            for ([_]f32{ 0, Head.default_confidence, 1e-4 }) |confidence| {
                var chains: [2]std.ArrayList(i32) = .{ .empty, .empty };
                defer for (&chains) |*chain| chain.deinit(gpa);
                for ([_]bool{ true, false }, &chains) |on_device, *chain| {
                    d.head.on_device = on_device;
                    d.head.confidence = confidence;
                    d.head.chosen = .{};
                    try d.head.reset(case.settings);
                    var recorder: Recorder = .{ .inner = d.head.drafter(), .all = chain };
                    const got = try shared.reply(gpa, &d.loop, d.fixture.tokens, count, case.settings, none, recorder.drafter(), null);
                    defer gpa.free(got.tokens);
                    try testing.expectEqualSlices(i32, serial, got.tokens);
                    if (!on_device) try testing.expectEqual(@as(usize, 0), d.head.chosen.device);
                    // the start settles a greedy choice and a top-k one whatever the row; a nucleus where its cut is within it
                    if (on_device and case.device) try testing.expect(d.head.chosen.device > 9 * d.head.chosen.host);
                    if (on_device and case.settings.top_k == 0 and case.settings.top_p == 1 and !case.settings.greedy()) try testing.expectEqual(@as(usize, 0), d.head.chosen.device);
                }
                try testing.expectEqualSlices(i32, chains[1].items, chains[0].items);
            }
        }
    }
    // the four-bit rows are other logits: other drafts, and the same replies
    for (replies[0], replies[1]) |own, four| try testing.expectEqualSlices(i32, own, four);
}

test "a seat granted no row absorbs its kept rows without a scored item, and drafts where it is granted one (spec 152 WS-08, TensorFold #1779)" {
    const mtp_seats = @import("../../../../src/models/glm5/mtp/mtp_seats.zig");
    const rows = 7;
    const capacity = 16;
    const gpa = testing.allocator;
    var fixture: Fixture = undefined;
    try fixture.init(73, rows + 2);
    defer fixture.deinit();
    const arena = fixture.arena_state.allocator();
    var loop = try fixture.loop(capacity);
    defer loop.deinit();
    var pair: Pair = try .init(arena, gpa, fixture.source, &fixture.c, loop.m, capacity, fixture.rows());
    defer pair.deinit();
    var sampler: sampling.Sampler = try .init(gpa, fx.vocab);
    defer sampler.deinit(gpa);
    var head: Head = try .init(gpa, loop.m, &pair, &loop.b, &sampler, .{ .seed = 0, .temperature = 0 }, fx.vocab);
    defer head.deinit();
    var seats: mtp_seats.SeatsOf(world) = try .init(gpa, loop.m, 2, capacity);
    defer seats.deinit();
    try runMain(&loop, fixture.tokens[0..rows]);
    const next = fixture.tokens[1..];
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    var full = [_]bool{ false, false };
    var granted: [4]i32 = undefined;
    var jobs = [_]draft_file.Job{
        .{ .slot = 0, .from = 0, .kept = next[0..3], .position = 3, .settings = greedy, .ends = &.{}, .guide = null, .out = &.{}, .full = &full[0] },
        .{ .slot = 1, .from = 3, .kept = next[3..rows], .position = rows, .settings = greedy, .ends = &.{}, .guide = null, .out = &granted, .full = &full[1] },
    };
    try seats.draft(&head, &jobs);
    // the zero-grant job proposed nothing; both seats' kept rows stand absorbed at its position
    try testing.expectEqual(@as(usize, 0), jobs[0].count);
    for (seats.caches) |caches| try testing.expectEqual(@as(usize, 3), caches[0].pos);
    for (seats.caches) |caches| try testing.expectEqual(@as(usize, rows), caches[1].pos);
    try testing.expect(jobs[1].count > 0);

    // a later round that grants slot 0 rows drafts from the state the skipped round kept
    full[0] = false;
    var later: [4]i32 = undefined;
    var again = [_]draft_file.Job{
        .{ .slot = 0, .from = 3, .kept = next[3..6], .position = 6, .settings = greedy, .ends = &.{}, .guide = null, .out = &later, .full = &full[0] },
    };
    try seats.draft(&head, &again);
    try testing.expect(again[0].count > 0);
    for (seats.caches) |caches| try testing.expectEqual(@as(usize, 6), caches[0].pos);
}
