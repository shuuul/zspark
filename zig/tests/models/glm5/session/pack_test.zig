//! `pack.zig`'s test, on FP4-Local (`ZSPARK_GLM_FP4_LOCAL`, each linear in its stored format, spec 107 WS-02): both ranks' shares of its first four layers (three KDA with dense MLPs,
//! a DSA with a MoE) and of its MTP layer in loopback, at the model's own sizes. A window of bf16 rows is its serial
//! steps, a quantized one's row the same in windows of two sizes (spec 115), and a reply drafted by the MTP layer the
//! serial reply; FP4-Local's logits on rows quantized as vLLM's against
//! bf16 rows' (spec 114 WS-04).

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const safetensors = @import("../../../../src/weights/safetensors.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const config_file = @import("../../../../src/models/glm5/layers/config.zig");
const draft_file = @import("../../../../src/models/glm5/draft.zig");
const dsa = @import("../../../../src/models/glm5/layers/dsa.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const loopback = @import("../../../../src/models/glm5/layers/loopback.zig");
const mtp_file = @import("../../../../src/models/glm5/mtp/mtp.zig");
const pack = @import("../../../../src/models/glm5/session/pack.zig");
const rig = @import("../../../../src/models/glm5/dflash2/rig.zig");
const shared = @import("../testing.zig");
const nvfp4 = @import("../../../../src/weights/nvfp4.zig");
const forward = @import("../../../../src/models/glm5/layers/forward.zig");
const profile_fixture = @import("../../../weights/profile_fixture.zig");
const nowNs = profile_fixture.nowNs;

const world = layout.world;

/// `primes`' prompt and the first 31 tokens of vLLM's reply (`ZSPARK_REFERENCE_GLM_FP4_LOCAL`'s primes.json).
const primes_tokens = [_]i32{ 154822, 154824, 154827, 852, 279, 1156, 4236, 10245, 5109, 11, 18630, 553, 75994, 13, 154828, 154841, 154842, 17, 11, 220, 18, 11, 220, 20, 11, 220, 22, 11, 220, 98965, 154827, 3838, 374, 279, 7772, 10245, 1372, 2686, 1091, 220, 99457, 271, 785, 7772, 10245, 1372, 2686, 1091 };

// The two checks on the first layers' load of rows quantized as vLLM's (the default) share it (spec 133); the bf16
// rows' linears are a load of their own, since `nvfp4.quantized_activations` is read as a linear loads.
test "FP4-Local's first layers in loopback, one load: a window is its serial steps, the MTP layer's drafts change no reply, and rows quantized as vLLM's against bf16 rows' logits" {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    const tokens = primes_tokens;
    const total = tokens.len;
    // the same windows of 24 (the quantize kernel's path) on each load: FP8 rows on `firstLayers`' load, then bf16
    // rows (spec 114's before) on one of their own
    var logits: [2][]u16 = undefined;
    for (&logits) |*out| out.* = try arena.alloc(u16, total * c.vocab);
    try firstLayers(gpa, arena, &checkpoint, &c, &tokens, logits[1]);
    defer nvfp4.quantized_activations = true;
    nvfp4.quantized_activations = false;
    try windowLogits(gpa, &checkpoint, &c, &tokens, logits[0]);
    // a row's relative error, its top token, the KL divergence of the quantized rows' softmax from bf16's, and
    // the change of the next token's loss
    var worst: f64 = 0;
    var mean: f64 = 0;
    var kl: f64 = 0;
    var loss: f64 = 0;
    var same: usize = 0;
    for (0..total) |r| {
        const a = logits[0][r * c.vocab ..][0..c.vocab];
        const b = logits[1][r * c.vocab ..][0..c.vocab];
        var diff: f64 = 0;
        var size: f64 = 0;
        var top: [2]usize = .{ 0, 0 };
        var most: [2]f64 = .{ -std.math.inf(f64), -std.math.inf(f64) };
        for (a, b, 0..) |x, y, i| {
            const fx: f64 = numeric.f32FromBf16(x);
            const fy: f64 = numeric.f32FromBf16(y);
            diff += (fx - fy) * (fx - fy);
            size += fx * fx;
            if (fx > most[0]) most[0] = fx;
            if (fx == most[0]) top[0] = i;
            if (fy > most[1]) most[1] = fy;
            if (fy == most[1]) top[1] = i;
        }
        var sums: [2]f64 = .{ 0, 0 };
        for (a, b) |x, y| {
            sums[0] += @exp(@as(f64, numeric.f32FromBf16(x)) - most[0]);
            sums[1] += @exp(@as(f64, numeric.f32FromBf16(y)) - most[1]);
        }
        for (a, b) |x, y| {
            const la = @as(f64, numeric.f32FromBf16(x)) - most[0] - @log(sums[0]);
            const lb = @as(f64, numeric.f32FromBf16(y)) - most[1] - @log(sums[1]);
            kl += @exp(la) * (la - lb);
        }
        if (r + 1 < total) {
            const next: usize = @intCast(tokens[r + 1]);
            const la = @as(f64, numeric.f32FromBf16(a[next])) - most[0] - @log(sums[0]);
            const lb = @as(f64, numeric.f32FromBf16(b[next])) - most[1] - @log(sums[1]);
            loss += @abs(la - lb) / @as(f64, total - 1);
        }
        const relative = @sqrt(diff / size);
        worst = @max(worst, relative);
        mean += relative / total;
        same += @intFromBool(top[0] == top[1]);
    }
    kl /= total;
    std.debug.print("FP4-Local first 4 layers and head, {d} rows: logits' relative RMS error mean {d:.4}, worst {d:.4}; top token kept {d} of {d}; KL(bf16 rows || fp8 rows) {e:.3} nats a row; the next token's loss {d:.4} nats off on average\n", .{ total, mean, worst, same, total, kl, loss });
    // 2026-10-19 (spec 114 WS-04): mean 0.033, worst 0.063, top token 45 of 48, KL 1.6e-3, the next token's loss
    // 0.045 off; what the score text costs is the two ranks' (`check.sh ranks glm-fp4-local`)
    try testing.expect(mean < 0.06);
    try testing.expect(kl < 0.01);
}

/// The logits of `tokens` in windows of 24 through the first four layers of both ranks in loopback, loaded here.
fn windowLogits(gpa: std.mem.Allocator, checkpoint: *safetensors.Checkpoint, c: *const config_file.Config, tokens: []const i32, out: []u16) !void {
    const rows = 24;
    var pool: cuda.Scratch = .init(gpa);
    defer pool.deinit();
    var first = try pack.load(gpa, checkpoint, c, 4, 0, .{ .pool = &pool, .count = rows });
    defer first.mtp.deinit();
    var second = pack.load(gpa, checkpoint, c, 4, 1, .{ .pool = &pool, .count = rows }) catch |err| {
        first.m.deinit();
        return err;
    };
    defer second.mtp.deinit();
    var m: loopback.Model = .{ .ranks = .{ first.m, second.m }, .max_rows = rows };
    defer m.deinit();
    try windowsOn(m, c.vocab, tokens, out);
}

/// The logits of `tokens` in windows of `m.max_rows` (24) through `m`, on a state of their own. Windows of 24 take
/// the quantized forms only past a bound of 16 (spec 115), lowered here.
fn windowsOn(m: loopback.Model, vocab: usize, tokens: []const i32, out: []u16) !void {
    const rows = m.max_rows;
    const was = forward.bf16_rows;
    forward.bf16_rows = draft_file.forward_max_rows;
    defer forward.bf16_rows = was;
    var loop: shared.Loop = undefined;
    loop.m = m;
    loop.state = try .init(loop.m, tokens.len);
    defer loop.state.deinit();
    loop.b = try .init(loop.m, tokens.len);
    defer loop.b.deinit();
    try loop.state.reset();
    var at: usize = 0;
    while (at < tokens.len) : (at += rows) {
        const window = tokens[at..][0..rows];
        try loopback.forward(loop.m, &loop.state, &loop.b, window);
        for (0..rows) |r| try loopback.logitsRow(loop.b, r, out[(at + r) * vocab ..][0..vocab]);
        try loopback.commit(loop.m, &loop.state, loop.b, window, rows);
    }
}

/// A window is its serial steps, the MTP layer's drafts change no reply; then `primes`' logits in windows of 24 on
/// the same load into `primes_out`.
fn firstLayers(gpa: std.mem.Allocator, arena: std.mem.Allocator, checkpoint: *safetensors.Checkpoint, c: *const config_file.Config, primes: []const i32, primes_out: []u16) !void {
    const layers = 4;
    // past `nvfp4.a8_tile_rows`, so that FP4-Local's quantized rows take both forms
    const rows = 24;
    const capacity = 64;

    var pool: cuda.Scratch = .init(gpa);
    defer pool.deinit();
    const first = try pack.load(gpa, checkpoint, c, layers, 0, .{ .pool = &pool, .count = rows });
    const second = pack.load(gpa, checkpoint, c, layers, 1, .{ .pool = &pool, .count = rows }) catch |err| {
        var loaded = first;
        loaded.deinit();
        return err;
    };
    // FP4-Local's routed experts are W4A4 (NVFP4), its MTP layer's bf16-activation (W4A16_NVFP4), as vLLM runs them
    for ([_]pack.Loaded{ first, second }) |one| {
        try testing.expectEqual(.nvfp4, one.m.layers[layers - 1].mlp.moe.routed.activations);
        try testing.expectEqual(.bf16, one.mtp.moe.routed.activations);
    }
    var loop: shared.Loop = undefined;
    loop.m = .{ .ranks = .{ first.m, second.m }, .max_rows = rows };
    var pair: mtp_file.Pair = undefined;
    pair.weights = .{ first.mtp, second.mtp };
    pair.drafted = 0;
    pair.lead = null;
    defer {
        for (&pair.weights) |*w| w.deinit();
        loop.m.deinit();
    }

    loop.state = try .init(loop.m, capacity);
    defer loop.state.deinit();
    loop.b = try .init(loop.m, capacity);
    defer loop.b.deinit();
    for (&pair.caches, &pair.weights) |*cache, w| cache.* = try w.attention.cacheFor(1, capacity);
    defer for (&pair.caches) |*cache| cache.deinit();
    for (&pair.b, loop.m.ranks) |*b, m| b.* = try mtp_file.Buffers.init(m, rows);
    defer for (&pair.b) |*b| b.deinit();

    const total = 24;
    var tokens: [total]i32 = undefined;
    for (&tokens, 0..) |*id, at| id.* = @intCast(1000 + 37 * at * at);
    const serial = try arena.alloc(u16, total * c.vocab);
    const row = try arena.alloc(u16, c.vocab);
    try loop.state.reset();
    for (tokens, 0..) |token, at| {
        try loopback.forward(loop.m, &loop.state, &loop.b, &.{token});
        try loopback.logitsRow(loop.b, 0, serial[at * c.vocab ..][0..c.vocab]);
        try loopback.commit(loop.m, &loop.state, loop.b, &.{token}, 1);
    }
    // logits of a model: finite, and a row's not its neighbour's
    var size: f64 = 0;
    for (serial) |bits| {
        const value: f64 = numeric.f32FromBf16(bits);
        try testing.expect(std.math.isFinite(value));
        size += value * value;
    }
    try testing.expect(size > 1);
    try testing.expect(!std.mem.eql(u16, serial[0..c.vocab], serial[c.vocab..][0..c.vocab]));
    // windows of 16 and 8, of 5 with 3 kept and then the rest, and of 24
    for ([_][]const [2]usize{ &.{ .{ 16, 16 }, .{ 8, 8 } }, &.{ .{ 5, 3 }, .{ 16, 16 }, .{ 5, 5 } }, &.{.{ 24, 24 }} }) |plan| {
        try loop.state.reset();
        var at: usize = 0;
        for (plan) |step| {
            const window = tokens[at..][0..step[0]];
            try loopback.forward(loop.m, &loop.state, &loop.b, window);
            for (0..step[1]) |r| {
                try loopback.logitsRow(loop.b, r, row);
                try testing.expectEqualSlices(u16, serial[(at + r) * c.vocab ..][0..c.vocab], row);
            }
            try loopback.commit(loop.m, &loop.state, loop.b, window, step[1]);
            at += step[1];
        }
        try testing.expectEqual(@as(usize, total), at);
    }
    // windows past the bf16 forms' bound (lowered to 16 here, spec 115) take the quantized forms: a row's bits the
    // same in a window of 24 and of 20, FP4-Local's not a serial step's
    {
        const was = forward.bf16_rows;
        forward.bf16_rows = draft_file.forward_max_rows;
        defer forward.bf16_rows = was;
        const wide = try arena.alloc(u16, total * c.vocab);
        try loop.state.reset();
        try loopback.forward(loop.m, &loop.state, &loop.b, &tokens);
        for (0..total) |r| try loopback.logitsRow(loop.b, r, wide[r * c.vocab ..][0..c.vocab]);
        const narrow = 20;
        try loop.state.reset();
        try loopback.forward(loop.m, &loop.state, &loop.b, tokens[0..narrow]);
        var serial_rows: usize = 0;
        for (0..narrow) |r| {
            try loopback.logitsRow(loop.b, r, row);
            try testing.expectEqualSlices(u16, wide[r * c.vocab ..][0..c.vocab], row);
            serial_rows += @intFromBool(std.mem.eql(u16, serial[r * c.vocab ..][0..c.vocab], row));
        }
        try testing.expect(serial_rows < narrow);
    }
    // a call of both forms (a hall's round of a reply and a long prompt's part): rows before `from` a bf16 call's
    // bits, the rest a quantized call's, layer 3's MoE with its shared expert
    {
        const mlp_file = @import("../../../../src/models/glm5/layers/mlp.zig");
        const moe = first.m.layers[layers - 1].mlp.moe;
        var prng: std.Random.DefaultPrng = .init(115);
        const x_host = try arena.alloc(u16, rows * c.hidden);
        shared.randomBf16(prng.random(), x_host, 1.0);
        var x_dev = try shared.device(u16, .bf16, &.{ rows, c.hidden }, x_host);
        defer x_dev.deinit();
        var out: cuda.Buffer = try .init(.f32, &.{ rows, c.hidden });
        defer out.deinit();
        var buffers: mlp_file.Moe.Buffers = try .init(moe, rows);
        defer buffers.deinit();
        defer pool.quantize_from = 0;
        var whole: [2][]u32 = undefined;
        for (&whole, [_]usize{ rows, 0 }) |*one, from| {
            pool.quantize_from = from;
            try moe.forward(buffers, x_dev.view, out.view, from);
            one.* = try shared.fetch(u32, arena, out);
        }
        try testing.expect(!std.mem.eql(u32, whole[0], whole[1]));
        for ([_]usize{ 1, 10, 23 }) |from| {
            pool.quantize_from = from;
            try moe.forward(buffers, x_dev.view, out.view, from);
            const got = try shared.fetch(u32, arena, out);
            try testing.expectEqualSlices(u32, whole[0][0 .. from * c.hidden], got[0 .. from * c.hidden]);
            try testing.expectEqualSlices(u32, whole[1][from * c.hidden ..], got[from * c.hidden ..]);
        }
    }

    const count = 12;
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
    var sampler: sampling.Sampler = try .init(gpa, c.vocab);
    defer sampler.deinit(gpa);
    var head: mtp_file.Head = try .init(gpa, loop.m, &pair, &loop.b, &sampler, greedy, c.vocab);
    defer head.deinit();
    head.confidence = 0;
    // as a session drafts: scored by a four-bit copy of each rank's rows of the vocabulary; an MXFP8 head as stored
    var heads: ?[world]mtp_file.Linear = if (loop.m.ranks[0].head == .bf16) try rig.draftHeads(world, arena, gpa, loop.m, .{ .pool = &pool, .count = 1 }, null) else null;
    defer if (heads) |*held| for (held) |*one| one.deinit();
    const nucleus: sampling.Sampling = .{ .seed = 4, .temperature = 0.8, .top_k = 0, .top_p = 0.9, .min_p = 0 };
    for ([_]sampling.Sampling{ greedy, .{ .seed = 3, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0 }, nucleus }) |settings| {
        const want = try shared.serialReply(gpa, loop.m, c.vocab, tokens[0..9], count, settings);
        defer gpa.free(want);
        const scorers = [_]?[world]mtp_file.Linear{ null, heads };
        for (scorers[0..if (heads == null) 1 else 2]) |scored_by| {
            pair.heads = scored_by;
            // each draft from the halves' starts, then from its row read back: the same chains
            var ran: [2]shared.Reply = undefined;
            for ([_]bool{ true, false }, &ran) |on_device, *got| {
                head.on_device = on_device;
                head.chosen = .{};
                try head.reset(settings);
                got.* = try shared.reply(gpa, &loop, tokens[0..9], count, settings, &.{}, head.drafter(), null);
                defer gpa.free(got.tokens);
                try testing.expectEqualSlices(i32, want, got.tokens);
                try testing.expect(got.stats.drafted > got.stats.rounds);
                // the start settles a greedy choice and a top-k one; a nucleus where its cut lies within it
                if (on_device and (settings.greedy() or settings.top_k != 0)) try testing.expectEqual(@as(usize, 0), head.chosen.host);
            }
            try testing.expectEqual(ran[1].stats.drafted, ran[0].stats.drafted);
            try testing.expectEqual(ran[1].stats.accepted, ran[0].stats.accepted);
        }
    }
    try windowsOn(loop.m, c.vocab, primes, primes_out);
}

test "FP4-Local's layer 3: W4A4 routed experts against A16 on its own inputs of a real prompt" {
    // Runs with ZSPARK_GLM_FP4_LOCAL set and ZSPARK_GLM_A4_ERROR naming a reference case (`score.json`: 4,096 of
    // its prompt's tokens, FP4-Local's served window since spec 114 WS-08, 2,048 of a shorter prompt;
    // `sky-long.json`: its prompt and first reply, 256); skipped without.
    // Four layers of both ranks in loopback and an A16 copy of layer 3's experts of rank 0: some 15 GiB. Prints the
    // error of each pair's expert output and of each row's routed sum, the groups whose scale saturates, the
    // routing, and each form's time on these picks (spec 114 WS-03; held alone, the machine lock whole); and the
    // window's form against 16 pairs a reading, bit for bit.
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const case_path = testing.environ.getPosix("ZSPARK_GLM_A4_ERROR") orelse return error.SkipZigTest;
    const experts = @import("../../../../src/weights/experts.zig");
    const mlp = @import("../../../../src/models/glm5/layers/mlp.zig");
    const gpa = testing.allocator;
    const io = testing.io;
    const layers = 4;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const text = try std.Io.Dir.cwd().readFileAlloc(io, case_path, arena, .limited(64 << 20));
    const Case = struct { prompt: []const i32, runs: []const struct { tokens: []const i32 = &.{} } };
    const case = try std.json.parseFromSliceLeaky(Case, arena, text, .{ .ignore_unknown_fields = true });
    const rows: usize = if (case.prompt.len >= 4096) 4096 else if (case.prompt.len >= 2048) 2048 else 256;
    const tokens = try arena.alloc(i32, rows);
    for (tokens, 0..) |*id, at| id.* = if (at < case.prompt.len) case.prompt[at] else case.runs[0].tokens[at - case.prompt.len];

    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    var pool: cuda.Scratch = .init(gpa);
    defer pool.deinit();
    const first = try pack.load(gpa, &checkpoint, &c, layers, 0, .{ .pool = &pool, .count = rows });
    const second = pack.load(gpa, &checkpoint, &c, layers, 1, .{ .pool = &pool, .count = rows }) catch |err| {
        var loaded = first;
        loaded.deinit();
        return err;
    };
    var loop: shared.Loop = undefined;
    loop.m = .{ .ranks = .{ first.m, second.m }, .max_rows = rows };
    var mtps = [_]mtp_file.Weights{ first.mtp, second.mtp };
    defer {
        for (&mtps) |*w| w.deinit();
        loop.m.deinit();
    }
    loop.state = try .init(loop.m, 2 * rows);
    defer loop.state.deinit();
    loop.b = try .init(loop.m, 2 * rows);
    defer loop.b.deinit();
    try loop.state.reset();
    try loopback.forward(loop.m, &loop.state, &loop.b, tokens);

    // rank 0's last read-out is layer 3's MoE input, its picks and weights the layer's
    const b = loop.b.ranks[0];
    const moe_b = b.moe.?;
    const x = b.normed.rows(rows);
    const a4 = first.m.layers[layers - 1].mlp.moe.routed;
    try testing.expectEqual(experts.Activations.nvfp4, a4.activations);
    var staging: experts.Staging = .{};
    defer staging.deinit(gpa);
    var a16 = try mlp.Moe.nvfp4Experts(arena, gpa, &staging, &checkpoint, &c, layers - 1, 0, .bf16);
    defer a16.deinit();

    const slots = moe_b.slots;
    const pairs = rows * slots;
    const picks_view = moe_b.picks.view;
    var plan: experts.Plan = try .init(rows, slots, a4.count + 1);
    defer plan.deinit();
    var act: cuda.Buffer = try .init(.bf16, &.{ pairs, a4.width });
    defer act.deinit();
    var rows_q: experts.Quantized = try .init(pairs, a4.dims, a4.width);
    defer rows_q.deinit();
    var y4: cuda.Buffer = try .init(.f32, &.{ pairs, a4.dims });
    defer y4.deinit();
    var y16: cuda.Buffer = try .init(.f32, &.{ pairs, a4.dims });
    defer y16.deinit();
    const pass = experts.Activations.passOf(rows);
    try plan.routeAs(picks_view, rows, pass);
    try a4.gateUpQuantizedAs(x, plan, rows_q, act.view, rows, a4.count, pass);
    // down's input as W4A4 makes it (the values its epilogue quantized), for its saturated groups
    try cuda.synchronize();
    const act4 = try arena.alloc(u16, pairs * a4.width);
    try act.download(std.mem.sliceAsBytes(act4));
    try a4.downProjectQuantizedAs(plan, rows_q, y4.view, rows, a4.count, pass);
    try plan.route(picks_view, rows);
    try a16.gateUp(x, plan, act.view, rows, a16.count);
    try a16.downProject(act.view, plan, y16.view, rows, a16.count);
    try cuda.synchronize();

    // the row rule at the layer's own shapes and inputs: 16 pairs a reading give the window kernel's bits
    {
        var y_narrow: cuda.Buffer = try .init(.f32, &.{ pairs, a4.dims });
        defer y_narrow.deinit();
        try plan.routeAs(picks_view, rows, .narrow);
        try a4.gateUpQuantizedAs(x, plan, rows_q, null, rows, a4.count, .narrow);
        try a4.downProjectQuantizedAs(plan, rows_q, y_narrow.view, rows, a4.count, .narrow);
        try cuda.synchronize();
        const got = try arena.alloc(u32, pairs * a4.dims);
        try y_narrow.download(std.mem.sliceAsBytes(got));
        const want = try arena.alloc(u32, pairs * a4.dims);
        try y4.download(std.mem.sliceAsBytes(want));
        const ids = try arena.alloc(i32, pairs);
        try moe_b.picks.download(std.mem.sliceAsBytes(ids));
        for (0..pairs) |p| {
            if (ids[p] >= a4.count) continue;
            try testing.expectEqualSlices(u32, want[p * a4.dims ..][0..a4.dims], got[p * a4.dims ..][0..a4.dims]);
        }
    }

    const picks = try arena.alloc(i32, pairs);
    try moe_b.picks.download(std.mem.sliceAsBytes(picks));
    const weights = try arena.alloc(f32, pairs);
    try moe_b.weights.download(std.mem.sliceAsBytes(weights));
    const h4 = try arena.alloc(f32, pairs * a4.dims);
    try y4.download(std.mem.sliceAsBytes(h4));
    const h16 = try arena.alloc(f32, pairs * a4.dims);
    try y16.download(std.mem.sliceAsBytes(h16));

    const dims = a4.dims;
    var pair_sum: f64 = 0;
    var pair_most: f64 = 0;
    var pair_count: usize = 0;
    var row_sum: f64 = 0;
    var row_most: f64 = 0;
    var held: [512]usize = @splat(0);
    const r4 = try arena.alloc(f64, dims);
    const r16 = try arena.alloc(f64, dims);
    for (0..rows) |row| {
        @memset(r4, 0);
        @memset(r16, 0);
        for (0..slots) |slot| {
            const p = row * slots + slot;
            const e: usize = @intCast(picks[p]);
            if (e >= a4.count) continue;
            held[e] += 1;
            var diff: f64 = 0;
            var size: f64 = 0;
            for (0..dims) |i| {
                const v4: f64 = h4[p * dims + i];
                const v16: f64 = h16[p * dims + i];
                try testing.expect(std.math.isFinite(v4));
                diff += (v4 - v16) * (v4 - v16);
                size += v16 * v16;
                r4[i] += weights[p] * v4;
                r16[i] += weights[p] * v16;
            }
            const rel = @sqrt(diff / size);
            pair_sum += rel;
            pair_most = @max(pair_most, rel);
            pair_count += 1;
        }
        var diff: f64 = 0;
        var size: f64 = 0;
        for (r4, r16) |v4, v16| {
            diff += (v4 - v16) * (v4 - v16);
            size += v16 * v16;
        }
        const rel = @sqrt(diff / size);
        row_sum += rel;
        row_most = @max(row_most, rel);
    }
    // groups of 16 inputs whose scale saturates at 448 (amax * gs / 6 past it), gate and up's
    const x_host = try arena.alloc(u16, rows * dims);
    try cuda.download(x, std.mem.sliceAsBytes(x_host));
    const gs = try arena.alloc(f32, a4.count);
    try a4.up_gs.?.download(std.mem.sliceAsBytes(gs));
    var saturated: usize = 0;
    var groups: usize = 0;
    for (0..pairs) |p| {
        const e: usize = @intCast(picks[p]);
        if (e >= a4.count) continue;
        for (0..dims / 16) |g| {
            var amax: f32 = 0;
            for (x_host[(p / slots) * dims + g * 16 ..][0..16]) |bits| amax = @max(amax, @abs(numeric.f32FromBf16(bits)));
            saturated += @intFromBool(amax * gs[e] / 6 > 448);
            groups += 1;
        }
    }
    const gs_down = try arena.alloc(f32, a4.count);
    try a4.down_gs.?.download(std.mem.sliceAsBytes(gs_down));
    var saturated_down: usize = 0;
    var most_ratio: f32 = 0;
    for (0..pairs) |p| {
        const e: usize = @intCast(picks[p]);
        if (e >= a4.count) continue;
        for (0..a4.width / 16) |g| {
            var amax: f32 = 0;
            for (act4[p * a4.width + g * 16 ..][0..16]) |bits| amax = @max(amax, @abs(numeric.f32FromBf16(bits)));
            saturated_down += @intFromBool(amax * gs_down[e] / 6 > 448);
            most_ratio = @max(most_ratio, amax * gs_down[e] / 6 / 448);
        }
    }
    std.debug.print("A4ERROR groups saturated: gate and up's {d} of {d}, down's {d} of {d} (the largest at {d:.3} of 448)\n", .{ saturated, groups, saturated_down, pairs / slots * (slots - 1) * (a4.width / 16), most_ratio });

    // each form's call on these picks, the layer's weights past L2 between calls (2 GB a form)
    var times: [2][7]u64 = undefined;
    for (0..7) |round| {
        for (0..2) |form| {
            try cuda.synchronize();
            const t0 = nowNs();
            if (form == 0) {
                try plan.route(picks_view, rows);
                try a16.gateUp(x, plan, act.view, rows, a16.count);
                try a16.downProject(act.view, plan, y16.view, rows, a16.count);
            } else {
                try plan.routeAs(picks_view, rows, pass);
                try a4.gateUpQuantizedAs(x, plan, rows_q, null, rows, a4.count, pass);
                try a4.downProjectQuantizedAs(plan, rows_q, y4.view, rows, a4.count, pass);
            }
            try cuda.synchronize();
            times[form][round] = nowNs() - t0;
        }
    }
    for (&times) |*one| std.mem.sort(u64, one, {}, std.sort.asc(u64));
    std.debug.print("A4ERROR time on these picks, median of 7: A16 {d:.1} us, W4A4 {d:.1} us\n", .{ @as(f64, @floatFromInt(times[0][3])) / 1e3, @as(f64, @floatFromInt(times[1][3])) / 1e3 });
    var distinct: usize = 0;
    var most: usize = 0;
    var by64: usize = 0;
    var by128: usize = 0;
    for (held[0..a4.count]) |n| {
        distinct += @intFromBool(n > 0);
        most = @max(most, n);
        by64 += (n + 63) / 64;
        by128 += (n + 127) / 128;
    }
    std.debug.print("A4ERROR layer 3, {d} rows, rank 0: a pair's expert output, relative L2 to A16 mean {d:.4} max {d:.4} ({d} pairs); a row's routed sum mean {d:.4} max {d:.4}; {d} experts routed, the most pairs {d}, {d} readings of 64, {d} of 128\n", .{ rows, pair_sum / @as(f64, @floatFromInt(pair_count)), pair_most, pair_count, row_sum / @as(f64, @floatFromInt(rows)), row_most, distinct, most, by64, by128 });
    try testing.expect(row_sum / @as(f64, @floatFromInt(rows)) < 0.25);
}

test "FP4-Local's parts as staged for both ranks: every range a rank reads is the file's bytes, no GPU" {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    for (0..world) |rank| {
        // the head, a DSA layer with routed experts, a KDA one, the MTP layer: the reader fills one slot while the
        // other is read here
        var each: pack.Each = .{ .gpa = gpa, .checkpoint = &checkpoint, .c = &c, .rank = rank, .first = 3, .count = 2, .with_mtp = true, .arena = .init(gpa) };
        defer each.deinit();
        const view = each.view();
        var staged: usize = 0;
        var tensors: usize = 0;
        for ([_]@import("../../../../src/models/glm5/layers/model.zig").Model.Part{ .top, .{ .layer = 3 }, .{ .layer = 4 }, .{ .layer = c.layers } }) |part| {
            _ = try each.begin(part);
            var name: [64]u8 = undefined;
            const under = switch (part) {
                .top => null,
                .layer => |index| try std.fmt.bufPrint(&name, layout.prefix ++ "layers.{d}.", .{index}),
            };
            var it = checkpoint.tensors.iterator();
            while (it.next()) |entry| {
                const key = entry.key_ptr.*;
                if (under) |prefix| {
                    if (!std.mem.startsWith(u8, key, prefix)) continue;
                } else if (!std.mem.eql(u8, key, layout.prefix ++ "embed_tokens.weight") and !std.mem.eql(u8, key, "lm_head.weight")) continue;
                const data = entry.value_ptr.data;
                const range = pack.rangeOf(key, data, rank, false) orelse continue;
                const got = view.get(key).?.data;
                try testing.expectEqual(data.len, got.len);
                const offset = @intFromPtr(range.ptr) - @intFromPtr(data.ptr);
                try testing.expectEqualSlices(u8, range, got[offset..][0..range.len]);
                staged += @intFromBool(got.ptr != data.ptr);
                tensors += 1;
            }
            each.end(part);
        }
        std.debug.print("rank {d}: {d} tensors read as the file does, {d} of them staged\n", .{ rank, tensors, staged });
        if (checkpoint.shards[0].direct >= 0) try testing.expectEqual(tensors, staged);
    }
}

test "FP4-Local's parts over a wire: rank 1's, from rank 0's reads, are the file's bytes, and it reads no file, no GPU" {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    // rank 1's checkpoint: rank 0's headers alone, whose tensors' bytes no read reaches
    const heads = try arena.alloc([]const u8, checkpoint.shards.len);
    const lengths = try arena.alloc(u64, checkpoint.shards.len);
    for (checkpoint.shards, heads, lengths) |shard, *head, *length| {
        head.* = shard.head;
        length.* = shard.map.len;
    }
    var shell: safetensors.Checkpoint = try .fromHeads(gpa, heads, lengths);
    defer shell.close();
    const Part = @import("../../../../src/models/glm5/layers/model.zig").Model.Part;
    const parts = [_]Part{ .top, .{ .layer = 3 }, .{ .layer = c.layers } };
    var queue: Queue = .{};
    defer queue.deinit();
    // rank 0 walks the parts, reading rank 1's ranges beside its own and sending them; then rank 1 walks them
    var sending: pack.Each = .init(gpa, &checkpoint, &c, 0, 3, 1, true);
    sending.swapping = true;
    sending.wire = queue.wire();
    defer sending.deinit();
    var mine: Staged = .{};
    for (parts) |part| {
        _ = try sending.begin(part);
        try mine.check(&checkpoint, null, sending.view(), part, 0, c.layers);
        sending.end(part);
    }
    var receiving: pack.Each = .init(gpa, &shell, &c, 1, 3, 1, true);
    receiving.swapping = true;
    receiving.wire = queue.wire();
    defer receiving.deinit();
    var theirs: Staged = .{};
    for (parts) |part| {
        _ = try receiving.begin(part);
        try theirs.check(&checkpoint, &shell, receiving.view(), part, 1, c.layers);
        receiving.end(part);
    }
    try testing.expectEqual(queue.sent.items.len, queue.taken);
    std.debug.print("rank 0: {d} tensors, {d} bytes; rank 1: {d} tensors, {d} bytes of three parts, the file's from rank 0's reads\n", .{ mine.tensors, mine.bytes, theirs.tensors, theirs.bytes });
}

/// What a rank's view of a part holds: every range it reads, the file's bytes, staged.
const Staged = struct {
    tensors: usize = 0,
    bytes: usize = 0,

    fn check(self: *Staged, checkpoint: *const safetensors.Checkpoint, shell: ?*const safetensors.Checkpoint, view: pack.View, part: @import("../../../../src/models/glm5/layers/model.zig").Model.Part, rank: usize, layers: usize) !void {
        var name: [64]u8 = undefined;
        const under = switch (part) {
            .top => null,
            .layer => |index| try std.fmt.bufPrint(&name, layout.prefix ++ "layers.{d}.", .{index}),
        };
        const halved = switch (part) {
            .top => false,
            .layer => |index| index < layers,
        };
        var it = checkpoint.tensors.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            if (under) |prefix| {
                if (!std.mem.startsWith(u8, key, prefix)) continue;
            } else if (!std.mem.startsWith(u8, key, "lm_head.") and !std.mem.startsWith(u8, key, layout.prefix ++ "embed_tokens.") and !std.mem.startsWith(u8, key, layout.prefix ++ "norm.")) continue;
            const data = entry.value_ptr.data;
            const range = pack.rangeOf(key, data, rank, halved) orelse continue;
            const got = view.get(key).?.data;
            try testing.expectEqual(data.len, got.len);
            // staged: not the file's mapping, nor the shell's unreadable one
            try testing.expect(checkpoint.fileOf(got) == null);
            if (shell) |held| try testing.expect(held.fileOf(got) == null);
            const offset = @intFromPtr(range.ptr) - @intFromPtr(data.ptr);
            try testing.expectEqualSlices(u8, range, got[offset..][0..range.len]);
            self.tensors += 1;
            self.bytes += range.len;
        }
    }
};

/// `pack.Wire` in one process: what rank 0 sends, queued for rank 1 to receive after.
const Queue = struct {
    sent: std.ArrayList([]u8) = .empty,
    taken: usize = 0,

    fn wire(self: *Queue) pack.Wire {
        return .{ .context = self, .send = send, .flush = flush, .receive = receive };
    }

    fn deinit(self: *Queue) void {
        for (self.sent.items) |bytes| testing.allocator.free(bytes);
        self.sent.deinit(testing.allocator);
    }

    fn send(context: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *Queue = @ptrCast(@alignCast(context));
        try self.sent.append(testing.allocator, try testing.allocator.dupe(u8, bytes));
    }

    fn flush(_: *anyopaque, _: [*]const u8) anyerror!void {}

    fn receive(context: *anyopaque, into: []u8) anyerror!void {
        const self: *Queue = @ptrCast(@alignCast(context));
        if (self.taken == self.sent.items.len) return error.NothingSent;
        const bytes = self.sent.items[self.taken];
        self.taken += 1;
        if (bytes.len != into.len) return error.SizesDiffer;
        @memcpy(into, bytes);
    }
};

test "FP4-Local's experts placed by the device from a staged part are the host's packed bytes, both ranks" {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const mlp = @import("../../../../src/models/glm5/layers/mlp.zig");
    const experts = @import("../../../../src/weights/experts.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    // the first and the last layer of routed experts, down read whole, and halved with its other inputs traded
    for ([_]usize{ 3, c.layers - 1 }) |layer| for ([_]bool{ false, true }) |swapping| for (0..world) |rank| {
        var each: pack.Each = .init(gpa, &checkpoint, &c, rank, layer, 1, false);
        each.swapping = swapping;
        defer each.deinit();
        const part_arena = try each.begin(.{ .layer = layer });
        var host_staging: experts.Staging = .{};
        defer host_staging.deinit(gpa);
        var host = try mlp.Moe.nvfp4Experts(part_arena, gpa, &host_staging, &checkpoint, &c, layer, rank, null);
        defer host.deinit();
        var device_staging: experts.Staging = .{ .place = true };
        defer device_staging.deinit(gpa);
        // the peer's side of the trade from the checkpoint, and this rank's checked against it
        var peer: Peer = .{ .checkpoint = &checkpoint, .layer = layer, .rank = rank, .experts = c.experts };
        if (swapping) device_staging.swap = .{ .rank = rank, .context = &peer, .trade = Peer.trade };
        var placed = try mlp.Moe.nvfp4Experts(part_arena, gpa, &device_staging, each.view(), &c, layer, rank, null);
        defer placed.deinit();
        try device_staging.settle();
        each.end(.{ .layer = layer });
        try testing.expect(device_staging.device != null);
        try testing.expectEqual(swapping, peer.traded);
        try testing.expect(peer.sent_right);
        try testing.expectEqual(host.activations, placed.activations);
        try testing.expectEqual(experts.Activations.nvfp4, placed.activations);
        const pairs = [_][2]cuda.Buffer{ .{ host.up, placed.up }, .{ host.down, placed.down }, .{ host.up_scale, placed.up_scale }, .{ host.down_scale, placed.down_scale }, .{ host.up_gs.?, placed.up_gs.? }, .{ host.down_gs.?, placed.down_gs.? }, .{ host.up_ws2.?, placed.up_ws2.? }, .{ host.down_ws2.?, placed.down_ws2.? } };
        var bytes: usize = 0;
        for (pairs) |pair| {
            try testing.expectEqual(pair[0].bytes(), pair[1].bytes());
            const want = try gpa.alloc(u8, pair[0].bytes());
            defer gpa.free(want);
            const got = try gpa.alloc(u8, pair[1].bytes());
            defer gpa.free(got);
            try pair[0].download(want);
            try pair[1].download(got);
            try testing.expect(std.mem.eql(u8, want, got));
            bytes += want.len;
        }
        std.debug.print("layer {d} rank {d}{s}: {d} bytes placed by the device as the host packs them\n", .{ layer, rank, if (swapping) ", down traded" else "", bytes });
    };
}

/// The other rank of `experts.Swap` in a test: what it would send, read from the checkpoint.
const Peer = struct {
    checkpoint: *const safetensors.Checkpoint,
    layer: usize,
    rank: usize,
    experts: usize,
    traded: bool = false,
    sent_right: bool = true,

    fn trade(context: *const anyopaque, send: cuda.View, recv: cuda.View) cuda.Error!void {
        const self: *Peer = @ptrCast(@alignCast(@constCast(context)));
        self.traded = true;
        const gpa = testing.allocator;
        const sent = gpa.alloc(u8, send.count()) catch return error.CudaFailure;
        defer gpa.free(sent);
        try cuda.synchronize();
        try cuda.download(send, sent);
        const given = gpa.alloc(u8, recv.count()) catch return error.CudaFailure;
        defer gpa.free(given);
        const half: usize = @intCast(send.size[1]);
        const bytes: usize = @intCast(send.size[2]);
        const other = 1 - self.rank;
        for (0..self.experts) |e| {
            var name: [128]u8 = undefined;
            const weight = self.checkpoint.get(std.fmt.bufPrint(&name, layout.prefix ++ "layers.{d}.mlp.experts.{d}.down_proj.weight", .{ self.layer, e }) catch unreachable).?.data;
            const row = 2 * bytes;
            for (0..half) |r| {
                const at = (e * half + r) * bytes;
                if (!std.mem.eql(u8, sent[at..][0..bytes], weight[(self.rank * half + r) * row + other * bytes ..][0..bytes])) self.sent_right = false;
                @memcpy(given[at..][0..bytes], weight[(other * half + r) * row + self.rank * bytes ..][0..bytes]);
            }
        }
        const into: cuda.Buffer = .{ .view = recv };
        try into.upload(given);
    }
};
