//! Flash Next's MTP head split over the two ranks (spec 090's scheme B), both in one process: the random model's
//! head cut in two drafts a reply that is the two ranks' serial one; a row of a pass has the bits it has alone,
//! on both ranks alike; and a checkpoint's head loads as two halves. The checkpoint's first four layers on two
//! ranks are loaded here once for each form of the head and also run `ranks_test.zig`'s and
//! `pool/rounds_ranks_test.zig`'s checks on them (spec 133 WS-08: their fixtures).

const std = @import("std");
const testing = std.testing;
const fixtures = @import("testing.zig");
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const draft = @import("../../../src/decode/draft.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const mtp_file = @import("../../../src/models/qwen4/mtp.zig");
const mtp_ranks = @import("../../../src/models/qwen4/mtp_ranks.zig");
const ranks = @import("../../../src/models/qwen4/ranks_forward.zig");
const random = @import("../../../src/models/qwen4/checks/random.zig");
const attention = @import("../../../src/models/qwen4/blocks/attention.zig");
const streams_check = @import("../../../src/models/qwen4/checks/ranks_streams.zig");
const qwen4_config = @import("../../../src/models/qwen4/config.zig");
const layout = @import("../../../src/models/qwen4/layout.zig");
const ranks_fixture = @import("ranks_fixture.zig");
const rounds_ranks_fixture = @import("pool/rounds_ranks_fixture.zig");

const Allocator = std.mem.Allocator;
const Pair = ranks.Held(2);
const Halves = mtp_ranks.Held(2);

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
const rows_cap = 4;

const Drafted = struct { tokens: []i32, stats: draft.Stats, device: usize, host: usize };

/// `count` tokens after `prompt` on both ranks in loopback: one row a round, or in rounds the split head drafts
/// for (`chain` its floor, 0 filling every window), its passes through graphs with `graphed`. The trunk is
/// launched. `on_device` false reads every scored row back.
fn replyOn(gpa: Allocator, held: Pair.Model, prompt: []const i32, count: usize, settings: sampling.Sampling, drafting: bool, graphed: bool, chain: f32, on_device: bool, capacity: usize) !Drafted {
    const rows = held.max_rows;
    const vocab = held.ranks[0].shape.vocab;
    var state: Pair.State = try .init(gpa, held, rows, capacity);
    defer state.deinit();
    var b: Pair.Buffers = try .init(gpa, held, rows, capacity);
    defer b.deinit();
    var sampler: sampling.Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, vocab);
    defer gpa.free(logits);
    var decoder: Decoder = .{ .m = held, .state = &state, .b = &b, .graphs = null, .sampler = &sampler, .settings = settings, .eos = &.{}, .logits = logits };
    var head: ?mtp_file.Head = if (drafting) try .init(gpa, held.ranks[0], &sampler, settings, .{ .rows = rows, .capacity = capacity, .graphs = false }) else null;
    defer if (head) |*h| h.deinit();
    var split: Halves.Split = undefined;
    if (head) |*h| {
        h.chain = chain;
        h.on_device = on_device;
        try split.init(gpa, held, &b, h, rows, capacity, graphed);
    }
    defer if (head != null) split.deinit();
    const drafter: ?draft.Drafter = if (head) |*h| h.drafter() else null;

    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    var chosen: [draft.forward_max_rows]i32 = undefined;
    var drafts: [draft.forward_max_rows]i32 = undefined;
    var kept = try decoder.prompt(drafter, prompt, &chosen);
    out[0] = chosen[kept - 1];
    var made: usize = 1;
    while (made < count) {
        const n = if (drafter) |d| try decoder.draft(d, chosen[0..kept], count - made - 1, &drafts) else 0;
        kept = try decoder.round(chosen[kept - 1], drafts[0..n], &chosen);
        @memcpy(out[made..][0..kept], chosen[0..kept]);
        made += kept;
    }
    // the reply's last window absorbed, as a session leaves the head: both halves stand where the state does
    if (drafter) |d| _ = try decoder.draft(d, chosen[0..kept], 0, &drafts);
    if (head) |h| {
        try testing.expectEqual(state.pos, h.state.position.host);
        try testing.expectEqual(state.pos, split.state[1].position.host);
    }
    return .{ .tokens = out, .stats = decoder.stats, .device = if (head) |h| h.chosen.device else 0, .host = if (head) |h| h.chosen.host else 0 };
}

const reply_settings = [_]sampling.Sampling{
    .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 },
    .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 },
};

test "the random model's MTP head split over two ranks: each rank holds its half, and the ids in two parts" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const built = try random.sharesSplit(2, gpa, arena_state.allocator(), 42, 0, rows_cap);
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_cap };
    defer held.deinit();
    var ids: usize = 0;
    for (held.ranks, 0..) |one, rank| {
        const head = one.mtp.?;
        try testing.expect(head.whole == null and head.split != null);
        // the head's layer is of the share's own sizes
        try testing.expectEqual(random.fx.expert_width / 2, mtp_file.over(one).shape.width);
        // half of a head of twice the fixture's heads
        try testing.expectEqual(random.fx.head_attention.heads, one.attention.?.shape.heads);
        try testing.expectEqual(random.fx.head_attention.kv_heads, one.attention.?.shape.kv_heads);
        try testing.expectEqualSlices(u32, model.Mtp.Split.partOf(head.split.?.ids, rank), head.draft.?.ids);
        try testing.expectEqualSlices(u32, held.ranks[0].mtp.?.split.?.ids, head.scored().?);
        ids += head.draft.?.ids.len;
    }
    // an odd count: rank 0's part is an id longer, as `np.array_split` leaves it
    const all = held.ranks[0].mtp.?.split.?.ids;
    try testing.expectEqual(all.len, ids);
    try testing.expect(all.len % 2 == 1);
    try testing.expectEqual(held.ranks[1].mtp.?.draft.?.ids.len + 1, held.ranks[0].mtp.?.draft.?.ids.len);
    try testing.expectEqual(held.ranks[0].mtp.?.draft.?.ids.len, held.ranks[0].mtp.?.split.?.widest());
    try testing.expectEqualSlices(u32, &.{ 1, 2, 4 }, model.Mtp.Split.partOf(&.{ 1, 2, 4, 5, 7 }, 0));
    try testing.expectEqualSlices(u32, &.{ 5, 7 }, model.Mtp.Split.partOf(&.{ 1, 2, 4, 5, 7 }, 1));
}

test "the MTP head split over the random model's two ranks: the drafted reply is the two ranks' serial one, greedy and sampled" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const built = try random.sharesSplit(2, gpa, arena_state.allocator(), 42, 0, rows_cap);
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_cap };
    defer held.deinit();
    // longer than a window, so the head absorbs a prompt window before the last
    const prompt = [_]i32{ 5, 300, 17, 17, 0, 251 };
    const count = 40;
    for (reply_settings) |settings| {
        const serial = try replyOn(gpa, held, &prompt, count, settings, false, false, 0, true, 64);
        defer gpa.free(serial.tokens);
        var chosen: [2]?Drafted = @splat(null);
        defer for (chosen) |one| if (one) |kept| gpa.free(kept.tokens);
        for ([_]bool{ true, false }) |graphed| for ([_]f32{ 0, mtp_file.Head.default_chain }) |chain| for ([_]bool{ true, false }) |on_device| {
            const got = try replyOn(gpa, held, &prompt, count, settings, true, graphed, chain, on_device, 64);
            errdefer gpa.free(got.tokens);
            try testing.expectEqualSlices(i32, serial.tokens, got.tokens);
            try testing.expectEqual(count - 1, got.stats.rounds + got.stats.accepted);
            // a head of random weights is rarely right; without a floor it still fills every window
            if (chain == 0) try testing.expect(got.stats.drafted > got.stats.rounds);
            if (!on_device) try testing.expectEqual(@as(usize, 0), got.device);
            // The drafts are the same through graphs and launched, and chosen on the device as read back:
            // the gathered row's start is the row's.
            if (chain != 0) {
                gpa.free(got.tokens);
                continue;
            }
            const slot = &chosen[@intFromBool(on_device)];
            if (slot.*) |first| {
                try testing.expectEqual(first.stats.drafted, got.stats.drafted);
                try testing.expectEqual(first.stats.accepted, got.stats.accepted);
                gpa.free(got.tokens);
            } else slot.* = got;
        };
        try testing.expect(chosen[1].?.device > 0);
        try testing.expectEqual(chosen[0].?.stats.drafted, chosen[1].?.stats.drafted);
        try testing.expectEqual(chosen[0].?.stats.accepted, chosen[1].?.stats.accepted);
    }
}

/// The relative rms of two bf16 arrays' difference, over the first's.
fn apart(a_bytes: []const u8, b_bytes: []const u8) f32 {
    const a: []align(1) const u16 = std.mem.bytesAsSlice(u16, a_bytes);
    const b: []align(1) const u16 = std.mem.bytesAsSlice(u16, b_bytes);
    var diff: f64 = 0;
    var norm: f64 = 0;
    for (a, b) |x_bits, y_bits| {
        const x = numeric.f32FromBf16(x_bits);
        const y = numeric.f32FromBf16(y_bits);
        diff += (x - y) * (x - y);
        norm += x * x;
    }
    return @floatCast(@sqrt(diff / norm));
}

fn rowsOf(gpa: Allocator, buffer: cuda.Buffer, first: usize, count: usize) ![]u8 {
    const width = buffer.bytes() / @as(usize, @intCast(buffer.view.size[0]));
    const out = try gpa.alloc(u8, count * width);
    errdefer gpa.free(out);
    try cuda.synchronize();
    try buffer.downloadAt(first * width, out);
    return out;
}

test "a split head's rows in one pass have the bits they have a pass each, on both ranks alike, and the scored row is the ranks' parts in order" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const built = try random.sharesSplit(2, gpa, arena_state.allocator(), 7, 0, rows_cap);
    var held: Pair.Model = .{ .ranks = built, .max_rows = rows_cap };
    defer held.deinit();
    const capacity = 32;
    const window = [_]i32{ 9, 41, 300, 2 };
    const next = [_]i32{ 41, 300, 2, 77 };
    const scored = held.ranks[0].mtp.?.scored().?.len;

    // one pass of the window's rows, launched and through a graph
    var together: [2][2][]u8 = undefined;
    var rows_scored: [2][]u8 = undefined;
    for ([_]bool{ false, true }, 0..) |graphed, way| {
        var state: Pair.State = try .init(gpa, held, rows_cap, capacity);
        defer state.deinit();
        var b: Pair.Buffers = try .init(gpa, held, rows_cap, capacity);
        defer b.deinit();
        var split: Halves.Split = undefined;
        try split.init(gpa, held, &b, null, rows_cap, capacity, graphed);
        defer split.deinit();
        try Pair.forward(held, &state, &b, &window);
        try Pair.commit(held, &state, b, &window, window.len);
        try split.follow(0, null, &next, true, false);
        for (0..2) |rank| together[way][rank] = try rowsOf(gpa, split.b[rank].layer.h, 0, window.len);
        rows_scored[way] = try rowsOf(gpa, split.b[0].draft_logits.?, 0, 1);
        try testing.expectEqual(window.len, split.state[0].position.host);
        try testing.expectEqual(window.len, split.state[1].position.host);
        if (graphed) {
            const cache = &split.graphs.?.cache;
            try testing.expectEqual(1, cache.count());
            try testing.expect(cache.contains(.{ .rows = window.len, .context = capacity, .own = 0, .score = true }));
            try split.follow(0, null, &next, true, false);
            try testing.expectEqual(1, cache.count());
            for ([_]?usize{ null, 0, 1 }, 0..) |own, index| {
                try split.follow(0, own, &.{5}, true, false);
                try testing.expectEqual(index + 2, cache.count());
                try testing.expect(cache.contains(.{ .rows = 1, .context = capacity, .own = index, .score = true }));
                try split.follow(0, own, &.{5}, true, false);
                try testing.expectEqual(index + 2, cache.count());
            }
        }
    }
    defer for (together) |way| for (way) |bytes| gpa.free(bytes);
    defer for (rows_scored) |bytes| gpa.free(bytes);
    // the streams are whole on each rank, so the ranks' are the same bytes; a graph's are the launches'
    try testing.expectEqualSlices(u8, together[0][0], together[0][1]);
    try testing.expectEqualSlices(u8, together[0][0], together[1][0]);
    try testing.expectEqualSlices(u8, together[0][1], together[1][1]);
    try testing.expectEqualSlices(u8, rows_scored[0][0 .. scored * 2], rows_scored[1][0 .. scored * 2]);

    // a pass a row, the trunk a row a window
    var state: Pair.State = try .init(gpa, held, rows_cap, capacity);
    defer state.deinit();
    var b: Pair.Buffers = try .init(gpa, held, rows_cap, capacity);
    defer b.deinit();
    var split: Halves.Split = undefined;
    try split.init(gpa, held, &b, null, rows_cap, capacity, false);
    defer split.deinit();
    const width = together[0][0].len / window.len;
    for (window, next, 0..) |token, after, row| {
        try Pair.forward(held, &state, &b, &.{token});
        try Pair.commit(held, &state, b, &.{token}, 1);
        try split.follow(row, null, &.{after}, true, false);
        for (0..2) |rank| {
            const alone = try rowsOf(gpa, split.b[rank].layer.h, 0, 1);
            defer gpa.free(alone);
            try testing.expectEqualSlices(u8, together[0][rank][row * width ..][0..width], alone);
        }
    }
    const last = try rowsOf(gpa, split.b[0].draft_logits.?, 0, 1);
    defer gpa.free(last);
    try testing.expectEqualSlices(u8, rows_scored[0][0 .. scored * 2], last[0 .. scored * 2]);

    // each rank's part of the row is its own draft head's product of the last row's read-out
    const widest = held.ranks[0].mtp.?.split.?.widest();
    for (0..2) |rank| {
        const part = held.ranks[rank].mtp.?.draft.?;
        var out: cuda.Buffer = try .init(.bf16, &.{ 1, part.ids.len });
        defer out.deinit();
        try part.linear.forward(split.b[rank].layer.mixed.span(0, 1), out.view);
        const own = try rowsOf(gpa, out, 0, 1);
        defer gpa.free(own);
        try testing.expectEqualSlices(u8, own, last[rank * widest * 2 ..][0 .. part.ids.len * 2]);
    }
    // a chained pass reads the row the last one left, and a pass told a lower position stands there first
    try split.follow(window.len, 0, &.{5}, true, false);
    try testing.expectEqual(window.len + 1, split.state[1].position.host);
    try split.follow(2, null, &.{3}, false, false);
    try testing.expectEqual(@as(usize, 3), split.state[0].position.host);
    try testing.expectEqual(@as(usize, 3), split.state[1].position.host);
}

/// The checkpoint's first four layers on two ranks, both given the same options, the MTP head whole beside
/// rank 0's share or `split` in halves: the draft vocabulary as each check's own load had it, and the rows of
/// the widest check, a rank's rounds (`checks/ranks_streams.zig`). Narrowed for the others (`narrowedPair`).
fn loadRanks(gpa: Allocator, config: qwen4_config.Config, found: layout.Layout, ids: []const u32, split: bool) !Pair.Model {
    var ranks_m: [2]model.Model = undefined;
    var made: usize = 0;
    errdefer for (ranks_m[0..made]) |*one| one.deinit();
    for (&ranks_m, 0..) |*one, rank| {
        one.* = try .loadWith(gpa, config, found, .{ .layers = 4, .max_rows = streams_check.rows, .mtp = true, .draft_vocab = ids, .share = .of(@intCast(rank)), .head_split = split });
        made += 1;
    }
    return .{ .ranks = ranks_m, .max_rows = streams_check.rows };
}

/// `held` in windows of `rows` rows: each share as `checkpoints.narrowed` has it, the same weights.
fn narrowedPair(held: Pair.Model, rows: usize) Pair.Model {
    var narrow = held;
    for (&narrow.ranks) |*one| one.* = fixtures.checkpoints.narrowed(one.*, rows);
    narrow.max_rows = rows;
    return narrow;
}

// The checks on a checkpoint's first four layers on two ranks (spec 133 WS-08): one load for each form of the
// MTP head, the checks of `ranks_test.zig` and `pool/rounds_ranks_test.zig` on the whole head's and this file's
// on the split head's, each in the windows it was written for. Rank 0's share with the whole head beside it
// is also the whole head the split one is held against.
test "a checkpoint's first four layers on two ranks, one load a form of the MTP head: whole beside rank 0's share the drafted reply is the serial one and a rank's rounds on its share, caches in each storage, give each stream's rows alone, split the halves are the whole head's function and the drafted reply is the serial one" {
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    const window = 16;
    const ids = try model.DraftHead.defaultIds(arena);

    var whole = try loadRanks(gpa, config, found, ids, false);
    defer whole.deinit();
    {
        errdefer std.debug.print("the head whole beside rank 0's share: the drafted reply\n", .{});
        try ranks_fixture.wholeHeadDrafted(gpa, narrowedPair(whole, window), config, ids);
    }
    // a rank's rounds on each share alone, its caches stored as each case's own load had them
    // (`checkpoints.storing`), the trunk alone (`.mtp = false`)
    const cases = [_]struct { rank: usize, kv: attention.Storage }{ .{ .rank = 0, .kv = .bf16 }, .{ .rank = 1, .kv = .fp8 }, .{ .rank = 0, .kv = .fp8_nvfp4 } };
    for (cases) |case| {
        errdefer std.debug.print("rank {d}'s share alone, caches {t}: a rank's rounds\n", .{ case.rank, case.kv });
        var share = fixtures.checkpoints.storing(whole.ranks[case.rank], case.kv, case.kv);
        share.mtp = null;
        try rounds_ranks_fixture.checkpointPacked(gpa, rounds_ranks_fixture.alone(share, case.rank));
    }
    var held = try loadRanks(gpa, config, found, ids, true);
    defer held.deinit();
    errdefer std.debug.print("the head split over the ranks\n", .{});
    try splitHead(gpa, narrowedPair(held, window), fixtures.checkpoints.narrowed(whole.ranks[0], window), config, ids);
}

/// The MTP head split over the two ranks in `held`: each rank holds its half and its part of the ids `ids`; the
/// halves are the whole head's function, held against `whole`, rank 0's share with the head whole beside it,
/// on the same streams; and the drafted reply is the serial one.
fn splitHead(gpa: Allocator, held: Pair.Model, whole: model.Model, config: qwen4_config.Config, ids: []const u32) !void {
    const window = held.max_rows;
    for (held.ranks, 0..) |one, rank| {
        const head = one.mtp.?;
        try testing.expect(head.whole == null and head.split != null);
        try testing.expectEqual(config.moe_width / 2, mtp_file.over(one).shape.width);
        try testing.expectEqual(config.heads / 2, mtp_file.over(one).attention.?.shape.heads);
        try testing.expectEqualSlices(u32, model.Mtp.Split.partOf(ids, rank), head.draft.?.ids);
        try testing.expectEqualSlices(u32, ids, head.scored().?);
    }
    try testing.expectEqual(ids.len, held.ranks[0].mtp.?.draft.?.ids.len + held.ranks[1].mtp.?.draft.?.ids.len);

    const prompt = [_]i32{ 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198 };
    const count = 24;

    // The halves are the whole head's function: beside rank 0's whole head on the same final streams, a
    // pass's read-out and scored row differ by the rounding of a sum of two partials, no more.
    {
        try testing.expect(whole.mtp.?.whole != null);
        var state: Pair.State = try .init(gpa, held, window, 64);
        defer state.deinit();
        var b: Pair.Buffers = try .init(gpa, held, window, 64);
        defer b.deinit();
        var split: Halves.Split = undefined;
        try split.init(gpa, held, &b, null, window, 64, false);
        defer split.deinit();
        var one_state: mtp_file.State = try .init(whole, 64);
        defer one_state.deinit();
        var one_b: mtp_file.Buffers = try .init(gpa, whole, window, 64);
        defer one_b.deinit();
        const n = prompt.len - 1;
        try Pair.forward(held, &state, &b, prompt[0..n]);
        try Pair.commit(held, &state, b, prompt[0..n], n);
        try mtp_file.run(whole, &one_state, &one_b, b.ranks[0].h.rows(n), prompt[1..], true);
        try split.follow(0, null, prompt[1..], true, false);
        const hidden: usize = config.hidden;
        const mixed = [2][]u8{ try rowsOf(gpa, one_b.layer.mixed, 0, n), try rowsOf(gpa, split.b[0].layer.mixed, 0, n) };
        defer for (mixed) |bytes| gpa.free(bytes);
        const picks = [2][]u8{ try rowsOf(gpa, one_b.layer.moe.picks, 0, n), try rowsOf(gpa, split.b[0].layer.moe.picks, 0, n) };
        defer for (picks) |bytes| gpa.free(bytes);
        const slots = picks[0].len / n / 4;
        var alike: usize = 0;
        var worst: f32 = 0;
        var rerouted: f32 = 0;
        for (0..n) |r| {
            // A row whose router had two experts within the rounding takes another under it: a different
            // function of that row, so only rows routed alike are held near.
            var routed: [2][32]i32 = undefined;
            for (&routed, picks) |*one, bytes| {
                @memcpy(std.mem.sliceAsBytes(one[0..slots]), bytes[r * slots * 4 ..][0 .. slots * 4]);
                std.mem.sort(i32, one[0..slots], {}, std.sort.asc(i32));
            }
            const rms = apart(mixed[0][r * hidden * 2 ..][0 .. hidden * 2], mixed[1][r * hidden * 2 ..][0 .. hidden * 2]);
            if (std.mem.eql(i32, routed[0][0..slots], routed[1][0..slots])) {
                alike += 1;
                worst = @max(worst, rms);
            } else rerouted = @max(rerouted, rms);
        }
        std.debug.print("the split head beside the whole one, four layers: {d} of {d} rows routed alike, their read-out within {d:.5} relative rms; another expert, {d:.5}\n", .{ alike, n, worst, rerouted });
        try testing.expect(alike * 4 >= n * 3);
        try testing.expect(worst < 0.02);
    }
    for (reply_settings) |settings| {
        const serial = try replyOn(gpa, held, &prompt, count, settings, false, false, 0, true, 4096);
        defer gpa.free(serial.tokens);
        const got = try replyOn(gpa, held, &prompt, count, settings, true, true, 0, true, 4096);
        defer gpa.free(got.tokens);
        std.debug.print("two ranks, four layers, the head split: {d} rounds, {d} of {d} drafts kept\n", .{ got.stats.rounds, got.stats.accepted, got.stats.drafted });
        try testing.expectEqualSlices(i32, serial.tokens, got.tokens);
        try testing.expectEqual(count - 1, got.stats.rounds + got.stats.accepted);
    }
}
