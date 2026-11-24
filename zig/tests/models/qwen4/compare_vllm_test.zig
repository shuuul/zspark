//! A checkpoint no TensorFold reads, against the vLLM container its card names (spec 105): the whole model's
//! greedy replies, serial and drafted, and the score text's losses, held to the files `vllm_reference.py`
//! fetched (`tools/reference/vllm_reference.sh`); then, on the same load, DFlash2's completed-layer taps against
//! the card's vLLM's own (spec 125 Z1, `tools/reference/flash_next_taps_vllm.sh`), and the stored DFlash2 drafter
//! conditioned on either's taps along the same rows (spec 126 WS-02). The full tier's alone: it loads the whole
//! model.

const std = @import("std");
const cuda = @import("../../../src/cuda/cuda.zig");
const forward = @import("../../../src/models/qwen4/forward.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const compare_fixture = @import("compare_fixture.zig");
const Bench = compare_fixture.Bench;
const commonPrefix = compare_fixture.commonPrefix;
const testing = std.testing;
const fixtures = @import("testing.zig");

const whole_tier = @import("tier").whole;
const safetensors = @import("../../../src/weights/safetensors.zig");
const qwen4_config = @import("../../../src/models/qwen4/config.zig");
const rig = @import("../../../src/models/qwen4/dflash2/rig.zig");
const pass = @import("../../../src/models/qwen4/dflash2/pass.zig");
const dflash2_pass = @import("../../../src/decode/dflash2/pass.zig");
const flash_config = @import("../../../src/decode/dflash2/config.zig");
const flash_layout = @import("../../../src/decode/dflash2/layout.zig");

const numeric = @import("../../../src/weights/numeric.zig");
const vllm = @import("../../../src/models/qwen4/checks/vllm_reference.zig");
const cases = vllm.cases;
const tie = vllm.tie;

test "a mixed-precision checkpoint's whole model against the card's vLLM: greedy replies drafted, each its own serial reply, and the score text" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory and ZSPARK_REFERENCE_FP4_LOCAL to
    // what vllm_reference.sh wrote for it (and taps/, flash_next_taps_vllm.sh's, for Z1); skipped without either,
    // and outside the full tier.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const reference_path = testing.environ.getPosix("ZSPARK_REFERENCE_FP4_LOCAL") orelse return error.SkipZigTest;
    if (!whole_tier) return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    var references = try std.Io.Dir.cwd().openDir(io, reference_path, .{});
    defer references.close(io);

    const rows = 32;
    // the card's prompt form (spec 143 WS-06): prompt windows of more than `forward.bf16_rows` quantized, as the
    // score text's windows of 2048 below; steps, verify windows and the windows of 16 keep bf16 rows
    const form = model.prompt_form;
    model.prompt_form = .card;
    defer model.prompt_form = form;
    var m: model.Model = try .loadWith(gpa, config, found, .{ .layers = config.layers, .max_rows = prompt_window });
    defer m.deinit();
    // the states `Bench` makes are `Bench.capacity`, and a graph's context bucket is the buffers'
    var b: forward.Buffers = try .init(gpa, m, rows, Bench.capacity);
    defer b.deinit();
    var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
    defer sampler.deinit(gpa);
    var wide: cuda.Buffer = try .init(.f32, &.{ 1, m.shape.vocab });
    defer wide.deinit();
    const bench: Bench = .{ .m = m, .b = &b, .sampler = &sampler, .logits = try arena.alloc(u16, m.shape.vocab), .wide = &wide, .gpa = gpa };
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };

    for (cases) |case| {
        var name_buffer: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buffer, "{s}.json", .{case.name});
        const text = try references.readFileAlloc(io, name, arena, .limited(1 << 26));
        const theirs = try vllm.reply(arena, text);
        const expected = theirs.runs[0].tokens;
        try testing.expect(theirs.prompt.len <= rows and expected.len >= case.least);
        // vLLM's two runs agree, or its own reply is not one to hold zspark to
        for (theirs.runs[1..]) |run| try testing.expectEqualSlices(i32, expected, run.tokens);

        // zspark's own reply drafted as `zspark chat` decodes, then its serial choice at each of its rows: the same
        const reply = try arena.alloc(i32, expected.len);
        const stats = try bench.drafted(theirs.prompt, reply, greedy);
        var label_buffer: [96]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buffer, "vLLM {s}", .{case.name});
        const own = try bench.forced(label, theirs.prompt, reply, greedy);
        try testing.expectEqual(reply.len, own.agree);

        // then along vLLM's reply: zspark's reply leaves it where the serial choice after vLLM's tokens does
        const serial = try bench.forced(label, theirs.prompt, expected, greedy);
        const same = commonPrefix(reply, expected);
        try testing.expectEqual(serial.first orelse reply.len, same);
        std.debug.print("{s}: the first {d} of {d} tokens are vLLM's; after its tokens {d} of {d} choices are zspark's own; drafted: {d} steps, {d} of {d} drafts kept\n", .{ label, same, reply.len, serial.agree, reply.len, stats.rounds, stats.accepted, stats.drafted });
        try testing.expect(same >= case.least);
        if (serial.first) |at| {
            // vLLM's own row there: its token and zspark's are a tie within `tie`
            const want = vllm.logprobOf(theirs, at, expected[at]) orelse return error.TestUnexpectedResult;
            const mine = vllm.logprobOf(theirs, at, reply[at]) orelse return error.TestUnexpectedResult;
            std.debug.print("{s}: at token {d} vLLM's {d} at {d:.4}, zspark's {d} at {d:.4}\n", .{ label, at, expected[at], want, reply[at], mine });
            try testing.expect(want - mine <= tie);
        }
    }

    // the score text in windows of 16, as `zspark score` runs it, against vLLM's prompt logprobs
    const text = try references.readFileAlloc(io, "score.json", arena, .limited(1 << 26));
    const scored = try vllm.scored(arena, text);
    try testing.expect(scored.runs.len >= 2);
    const tokens = scored.prompt;
    const window = 16;
    var state: forward.State = try .init(gpa, m, window, tokens.len);
    defer state.deinit();
    var buffers: forward.Buffers = try .init(gpa, m, window, tokens.len);
    defer buffers.deinit();
    const logits = bench.logits;
    var sums: vllm.Sums = .{};
    var at: usize = 0;
    while (at < tokens.len) {
        const count = @min(window, tokens.len - at);
        try forward.forward(m, &state, &buffers, tokens[at..][0..count]);
        for (0..count) |r| {
            const next = at + r + 1;
            if (next >= tokens.len) break;
            try forward.logitsRow(buffers, r, logits);
            sums.add(scored, logits, next);
        }
        try forward.commit(m, &state, buffers, tokens[at..][0..count], count);
        at += count;
    }
    var line: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&line);
    try sums.print(&w);
    std.debug.print("{s}", .{w.buffered()});
    // within `nll_apart` of vLLM, and as near vLLM as vLLM is to itself
    try testing.expect(sums.within(vllm.nll_apart));
    try promptForms(gpa, arena, m, references, scored);

    try tapsAgainstVllm(gpa, arena, m, &config, reference_path, rows);
}

/// The score text's prompt windows: as `session.promptRows` (a round's; one rank's lone stream 4,096 since spec 148), each quantized in the card's form.
const prompt_window = 2048;

/// Spec 143 WS-06's gate: the score text in prompt windows of 2048, in the card's form and in one bf16 form (the
/// bound raised past the window), against the reference's A16 prefill (`score.json`) and, where `score-a4.json` is
/// beside it (`vllm_reference.sh` with `DOCKER_EXTRA="-e B12X_W4A16_A4_PREFILL=1"`), the card's own A4 prefill. The
/// card's form's nll is within `nll_apart` of both references', and a token's mean |card - m| is no larger than the
/// A4 reference's mean |A4 - m|, m the token's mean logprob over every A16 run `score.json` holds (four: vLLM's A16
/// prefill differs run to run by 0.21 a token, its A4 not at all). The owner's rule, 2026-10-26 (spec 143 WS-08),
/// no statistical margin; the distances from each run and the paired excess a run are printed only.
fn promptForms(gpa: std.mem.Allocator, arena: std.mem.Allocator, m: model.Model, references: std.Io.Dir, a16: vllm.Scored) !void {
    const io = testing.io;
    const text = references.readFileAlloc(io, "score-a4.json", arena, .limited(1 << 26)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    const a4: ?vllm.Scored = if (text) |one| try vllm.scored(arena, one) else null;
    const tokens = a16.prompt;
    if (a4) |one| try testing.expectEqualSlices(i32, tokens, one.prompt);
    // the two references' forms apart: a token's |A4 - A16| from the A4's first run to each A16 run
    var across: [4]f64 = @splat(0);
    var count: usize = 0;
    if (a4) |one| for (1..tokens.len) |next| {
        for (across[0..a16.runs.len], a16.runs) |*sum, run| sum.* += @abs(one.runs[0].positions[next].?.logprob.? - run.positions[next].?.logprob.?);
        count += 1;
    };
    if (count > 0) for (&across) |*sum| {
        sum.* /= @floatFromInt(count);
    };
    var state: forward.State = try .init(gpa, m, prompt_window, tokens.len);
    defer state.deinit();
    var buffers: forward.Buffers = try .init(gpa, m, prompt_window, tokens.len);
    defer buffers.deinit();
    const logits = try arena.alloc(u16, m.shape.vocab);
    var by_form: [2][2]vllm.Sums = .{ .{ .{}, .{} }, .{ .{}, .{} } };
    var farther: [4]Paired = @splat(.{});
    // from the A16 runs' mean: the card's |card - m| and the A4's |A4 - m|, summed
    var from_mean: [2]f64 = @splat(0);
    std.debug.assert(a16.runs.len >= 2 and a16.runs.len <= farther.len);
    for ([_][]const u8{ "card", "bf16" }, 0..) |name, f| {
        const bound = forward.bf16_rows;
        if (f == 1) forward.bf16_rows = std.math.maxInt(usize);
        defer forward.bf16_rows = bound;
        try state.reset();
        var at: usize = 0;
        while (at < tokens.len) {
            const n = @min(prompt_window, tokens.len - at);
            try forward.forward(m, &state, &buffers, tokens[at..][0..n]);
            for (0..n) |r| {
                const next = at + r + 1;
                if (next >= tokens.len) break;
                try forward.logitsRow(buffers, r, logits);
                by_form[f][0].add(a16, logits, next);
                if (a4) |one| by_form[f][1].add(one, logits, next);
                // the card's form: how much farther from each A16 run a token is than the A4 reference's
                if (a4 != null and f == 0) {
                    const mine = -lossOf(logits, tokens[next]);
                    const theirs = a4.?.runs[0].positions[next].?.logprob.?;
                    var m_lp: f64 = 0;
                    for (a16.runs, farther[0..a16.runs.len]) |run, *one| {
                        const want = run.positions[next].?.logprob.?;
                        one.add(@abs(mine - want) - @abs(theirs - want));
                        m_lp += want;
                    }
                    m_lp /= @floatFromInt(a16.runs.len);
                    from_mean[0] += @abs(mine - m_lp);
                    from_mean[1] += @abs(theirs - m_lp);
                }
            }
            try forward.commit(m, &state, buffers, tokens[at..][0..n], n);
            at += n;
        }
        for ([_][]const u8{ "A16", "A4" }, by_form[f]) |reference, sums| {
            if (sums.count == 0) continue;
            var line: [256]u8 = undefined;
            var w: std.Io.Writer = .fixed(&line);
            try sums.print(&w);
            std.debug.print("windows of {d}, {s} form, against vLLM's {s} prefill: {s}", .{ prompt_window, name, reference, w.buffered() });
        }
    }
    if (a4 == null) return;
    const card = by_form[0];
    const n: f64 = @floatFromInt(card[0].count);
    for (card) |sums| try testing.expect(@abs(sums.zspark - sums.vllm[0]) / n <= vllm.nll_apart);
    for (farther[0..a16.runs.len], across[0..a16.runs.len], 0..) |one, from, run| {
        std.debug.print("A16 run {d}: the card's form {d:.4} from it, vLLM's A4 {d:.4}; a token's excess {d:.4}, a standard error {d:.4}\n", .{ run, from + one.mean(), from, one.mean(), one.err() });
    }
    const card_mean = from_mean[0] / n;
    const a4_mean = from_mean[1] / n;
    std.debug.print("from the mean of vLLM's {d} A16 runs: the card's form {d:.4}, vLLM's A4 {d:.4}\n", .{ a16.runs.len, card_mean, a4_mean });
    // no farther from the A16 runs' mean than vLLM's own A4, plainly
    try testing.expect(card_mean <= a4_mean);
}

/// A paired difference's mean and its standard error.
const Paired = struct {
    sum: f64 = 0,
    squares: f64 = 0,
    count: usize = 0,

    fn add(self: *Paired, d: f64) void {
        self.sum += d;
        self.squares += d * d;
        self.count += 1;
    }

    fn mean(self: Paired) f64 {
        return self.sum / @as(f64, @floatFromInt(self.count));
    }

    fn err(self: Paired) f64 {
        const n: f64 = @floatFromInt(self.count);
        return @sqrt(@max(self.squares / n - self.mean() * self.mean(), 0) / n);
    }
};

/// The loss of `token` under a bf16 row of logits, as `vllm.Sums.add` takes it.
fn lossOf(logits: []const u16, token: i32) f64 {
    var highest = -std.math.inf(f32);
    for (logits) |bits| highest = @max(highest, numeric.f32FromBf16(bits));
    var total: f64 = 0;
    for (logits) |bits| total += @exp(@as(f64, numeric.f32FromBf16(bits) - highest));
    return @log(total) - @as(f64, numeric.f32FromBf16(logits[@intCast(token)]) - highest);
}

/// A tap's figures over a range of rows against vLLM's: |zspark - vLLM|'s norm over vLLM's, its largest over
/// vLLM's largest, the worst row's norm ratio, and the elements whose bits agree.
const Figures = struct {
    rel: f64,
    max: f64,
    worst_row: f64,
    equal: f64,

    fn of(mine: []const u16, theirs: []const u16, hidden: usize, from: usize, to: usize) Figures {
        var dd: f64 = 0;
        var rr: f64 = 0;
        var dmax: f64 = 0;
        var rmax: f64 = 0;
        var worst: f64 = 0;
        var same: usize = 0;
        for (from..to) |row| {
            var row_dd: f64 = 0;
            var row_rr: f64 = 0;
            for (mine[row * hidden ..][0..hidden], theirs[row * hidden ..][0..hidden]) |a, b| {
                const x: f64 = @as(f32, @bitCast(@as(u32, a) << 16));
                const y: f64 = @as(f32, @bitCast(@as(u32, b) << 16));
                row_dd += (x - y) * (x - y);
                row_rr += y * y;
                dmax = @max(dmax, @abs(x - y));
                rmax = @max(rmax, @abs(y));
                same += @intFromBool(a == b);
            }
            dd += row_dd;
            rr += row_rr;
            worst = @max(worst, @sqrt(row_dd / @max(row_rr, 1e-30)));
        }
        return .{ .rel = @sqrt(dd / rr), .max = dmax / rmax, .worst_row = worst, .equal = @as(f64, @floatFromInt(same)) / @as(f64, @floatFromInt((to - from) * hidden)) };
    }
};

fn number(value: std.json.Value) f64 {
    return switch (value) {
        .float => |x| x,
        .integer => |x| @floatFromInt(x),
        else => std.math.nan(f64),
    };
}

/// Spec 125 Z1, on the first test's load: DFlash2's completed-layer taps, each case's r1 rows teacher-forced in
/// windows (the row rule makes their shape free), against the card's vLLM's in `REFERENCE/taps/` (what
/// flash_next_taps_vllm.sh assembled), within the bound its MANIFEST.json holds. Nothing where there is no taps/.
/// Then, where `~/models/Qwen3.8-Flash-Next-DFlash2` is, the drafter on both sources' taps (`Flash.walk`).
fn tapsAgainstVllm(gpa: std.mem.Allocator, arena: std.mem.Allocator, m: model.Model, config: *const qwen4_config.Config, reference_path: []const u8, window: usize) !void {
    const io = testing.io;
    const taps_path = try std.fmt.allocPrint(arena, "{s}/taps", .{reference_path});
    var taps_dir = std.Io.Dir.cwd().openDir(io, taps_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer taps_dir.close(io);
    const manifest = try std.json.parseFromSliceLeaky(std.json.Value, arena, try taps_dir.readFileAlloc(io, "MANIFEST.json", arena, .limited(1 << 24)), .{});

    const layers = [_]u32{ 5, 15, 25, 35, 45 };
    // the cases vLLM prefilled again: their bounds are its decode/prefill spread
    const held_cases = [_][]const u8{ "sky-long", "primes" };
    const hidden = m.shape.hidden;
    var past = false;
    const flash: ?*Flash = try Flash.open(gpa, arena, m, config, window);
    defer if (flash) |f| f.close(gpa);
    for ([_][]const u8{ "sky-long", "primes", "prose" }) |name| {
        const file_path = try std.fmt.allocPrint(arena, "{s}/{s}.safetensors", .{ taps_path, name });
        var file: safetensors.File = try .open(gpa, io, .cwd(), file_path);
        defer file.close();
        const ids_tensor = file.get("ids") orelse return error.TestUnexpectedResult;
        try testing.expectEqual(safetensors.DType.I32, ids_tensor.dtype);
        const ids = try arena.alloc(i32, ids_tensor.data.len / 4);
        @memcpy(std.mem.sliceAsBytes(ids), ids_tensor.data);
        const rows = ids.len;
        const spreads = manifest.object.get("spreads").?.object.get(name).?.object;
        const prompt = manifest.object.get("requests").?.object.get(name).?.object.get("prompt").?.array.items.len;

        // the rows teacher-forced in windows (the row rule makes their shape free), the taps of each read back
        var state: forward.State = try .init(gpa, m, window, rows);
        defer state.deinit();
        var b: forward.Buffers = try .init(gpa, m, window, rows);
        defer b.deinit();
        try b.setTaps(&layers, .f32_acc);
        const width = layers.len * hidden;
        const held = try arena.alloc(u16, b.rows * width);
        var mine: [layers.len][]u16 = undefined;
        for (&mine) |*one| one.* = try arena.alloc(u16, rows * hidden);
        var at: usize = 0;
        while (at < rows) {
            const count = @min(window, rows - at);
            try forward.forward(m, &state, &b, ids[at..][0..count]);
            try cuda.download(b.taps.?.view, std.mem.sliceAsBytes(held));
            for (0..count) |r| {
                for (0..layers.len) |slot| @memcpy(mine[slot][(at + r) * hidden ..][0..hidden], held[r * width + slot * hidden ..][0..hidden]);
            }
            try forward.commit(m, &state, b, ids[at..][0..count], count);
            at += count;
        }

        var theirs_all: [layers.len][]u16 = undefined;
        for (layers, 0..) |layer, slot| {
            var tensor_name: [32]u8 = undefined;
            const theirs_tensor = file.get(try std.fmt.bufPrint(&tensor_name, "mean.{d}", .{layer})) orelse return error.TestUnexpectedResult;
            try testing.expectEqual(rows * hidden * 2, theirs_tensor.data.len);
            const theirs = try arena.alloc(u16, rows * hidden);
            @memcpy(std.mem.sliceAsBytes(theirs), theirs_tensor.data);
            theirs_all[slot] = theirs;
            // twice vLLM's own spread at this layer (its runs', its decoded rows' against them prefilled), or 2^-7
            var layer_name: [8]u8 = undefined;
            const bound = spreads.get(try std.fmt.bufPrint(&layer_name, "{d}", .{layer})).?.object.get("bound").?.object;
            const bound_rel = number(bound.get("rel").?);
            const bound_max = number(bound.get("max").?);
            const all: Figures = .of(mine[slot], theirs, hidden, 0, rows);
            const head: Figures = .of(mine[slot], theirs, hidden, 0, prompt);
            // the reply's rows, those vLLM decoded (none where a case asks for one token)
            const tail: ?Figures = if (rows > prompt) .of(mine[slot], theirs, hidden, prompt, rows) else null;
            std.debug.print("Z1 {s} L{d}: {d} rows ({d} decoded), rel {e:.3} (prompt {e:.3}, reply {e:.3}), max {e:.3}, worst row {e:.3}, bits equal {d:.4}; bound rel {e:.3} max {e:.3}\n", .{ name, layer, rows, rows - prompt, all.rel, head.rel, if (tail) |t| t.rel else 0, all.max, all.worst_row, all.equal, bound_rel, bound_max });
            if (std.mem.eql(u8, name, "primes") and rows > prompt + 15) {
                // vLLM's primes answer ends at its 15th token; the rest is a loop of turn tokens under ignore_eos
                const answer: Figures = .of(mine[slot], theirs, hidden, prompt, prompt + 15);
                const loop: Figures = .of(mine[slot], theirs, hidden, prompt + 15, rows);
                std.debug.print("Z1 primes L{d}: the answer's rows rel {e:.3} max {e:.3}, the loop's rel {e:.3} max {e:.3}\n", .{ layer, answer.rel, answer.max, loop.rel, loop.max });
            }
            // A case vLLM did not prefill again (prose; the coordinator, 2026-10-25): its prompt rows are printed by
            // blocks of 256 and not held, since vLLM's own two prefills of them differ (r1 != r2) and no reference
            // holds them tighter than that spread, which a second boot would not settle; its decoded rows are held
            // to the short cases' bound at this layer, the larger of sky-long's and primes' in each figure.
            const layer_spreads = spreads.get(layer_name[0..std.fmt.count("{d}", .{layer})]).?.object;
            if (layer_spreads.get("prefill") == null or layer_spreads.get("prefill").? == .null) {
                var from: usize = 0;
                while (from < prompt) : (from += 256) {
                    const block: Figures = .of(mine[slot], theirs, hidden, from, @min(prompt, from + 256));
                    std.debug.print("Z1 {s} L{d} prompt rows {d}..{d}: rel {e:.3} max {e:.3}\n", .{ name, layer, from, @min(prompt, from + 256), block.rel, block.max });
                }
                var short_rel: f64 = 0;
                var short_max: f64 = 0;
                for (held_cases) |held_case| {
                    const short = manifest.object.get("spreads").?.object.get(held_case).?.object.get(layer_name[0..std.fmt.count("{d}", .{layer})]).?.object.get("bound").?.object;
                    short_rel = @max(short_rel, number(short.get("rel").?));
                    short_max = @max(short_max, number(short.get("max").?));
                }
                if (tail) |t| {
                    std.debug.print("Z1 {s} L{d}: its decoded rows rel {e:.3} max {e:.3}, the short cases' bound rel {e:.3} max {e:.3}\n", .{ name, layer, t.rel, t.max, short_rel, short_max });
                    if (t.rel > short_rel or t.max > short_max) past = true;
                } else std.debug.print("Z1 {s} L{d}: no decoded rows to hold (a reply of one token)\n", .{ name, layer });
                continue;
            }
            if (all.rel > bound_rel or all.max > bound_max) past = true;
        }
        if (flash) |f| {
            if (rows > prompt + 1) {
                const own = try f.walk(arena, &mine, ids, prompt);
                const vllms = try f.walk(arena, &theirs_all, ids, prompt);
                report(name, rows, own, vllms);
                // the unwritten taps of WS-04 kept about one token a round; N2's working drafter 1.8 to 2.9
                if (own.perRound() < 1.25) past = true;
            }
        }
    }
    try testing.expect(!past);
}

/// The stored DFlash2 drafter beside the target's load (spec 126 WS-02, 125 Z1's drafter half): its own
/// buffers, whose taps are written from the host, so zspark's taps and vLLM's take one path to the context.
const Flash = struct {
    opened: rig.Opened,
    b: forward.Buffers,
    flash: rig.Rig(2),
    out: [rig.published.block - 1]i32 = undefined,
    host: []u16,
    hidden: usize,

    const capacity = 512;
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };

    fn open(gpa: std.mem.Allocator, arena: std.mem.Allocator, m: model.Model, config: *const qwen4_config.Config, window: usize) !?*Flash {
        const io = testing.io;
        const home = testing.environ.getPosix("HOME") orelse return null;
        const path = try std.fmt.allocPrint(arena, "{s}/models/Qwen3.8-Flash-Next-DFlash2", .{home});
        const dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch {
            std.debug.print("DFlash2 on the taps: no drafter at {s}, not run\n", .{path});
            return null;
        };
        const self = try gpa.create(Flash);
        errdefer gpa.destroy(self);
        self.opened = .{ .path = path, .dir = dir };
        errdefer self.opened.close(io);
        self.opened.config = try flash_config.Config.readStored(arena, io, dir, &self.opened.diagnostic);
        self.opened.checkpoint = try .open(gpa, io, .cwd(), path);
        self.opened.layout = try flash_layout.Layout.init(arena, &self.opened.config, &self.opened.checkpoint.?, &self.opened.diagnostic);
        try testing.expect(try rig.fit(arena, &self.opened, config, config.layers));
        self.b = try .init(gpa, m, window, capacity);
        errdefer self.b.deinit();
        var source: rig.Source = .{ .checkpoint = &self.opened.checkpoint.? };
        defer source.deinit(gpa);
        try self.flash.load(arena, gpa, &self.opened, &source, .{ 0, 1 }, m.scratch, true, null);
        errdefer self.flash.unload();
        const heads = try rig.draftHeads(2, arena, gpa, m, true, .{ .pool = m.scratch, .count = dflash2_pass.batch });
        try self.flash.init(gpa, &self.opened, heads, null, m, &self.b, capacity, greedy, true);
        self.hidden = m.shape.hidden;
        self.host = try arena.alloc(u16, window * self.opened.config.taps.len * m.shape.hidden);
        return self;
    }

    fn close(self: *Flash, gpa: std.mem.Allocator) void {
        self.flash.deinit();
        self.b.deinit();
        self.opened.close(testing.io);
        gpa.destroy(self);
    }

    /// The rows `from..to` of `taps` (a tap layer's [rows, hidden] each) into the buffers' taps, from row 0.
    fn put(self: *Flash, taps: []const []u16, from: usize, to: usize) !void {
        const hidden = self.hidden;
        const width = taps.len * hidden;
        std.debug.assert(to - from <= self.b.rows);
        for (from..to, 0..) |row, r| {
            for (taps, 0..) |one, slot| @memcpy(self.host[r * width + slot * hidden ..][0..hidden], one[row * hidden ..][0..hidden]);
        }
        try self.b.taps.?.uploadAt(0, std.mem.sliceAsBytes(self.host[0 .. (to - from) * width]));
    }

    /// Along `ids` from `prompt`, at every position p: the context holds rows 0..p-1's taps of `taps`, the
    /// pending token is ids[p], and the drafts (the stored chain) are kept while they are ids[p + 1..]'s.
    fn walk(self: *Flash, arena: std.mem.Allocator, taps: []const []u16, ids: []const i32, prompt: usize) !Walk {
        const d = &self.flash.d;
        d.reset(greedy);
        const drafter = d.drafter();
        const positions = ids.len - 1 - prompt;
        const made: Walk = .{ .drafted = try arena.alloc(u8, positions), .kept = try arena.alloc(u8, positions), .drafts = try arena.alloc([rig.published.block - 1]i32, positions), .prompt = prompt };
        var kept_end: usize = 0;
        for (prompt..ids.len - 1, 0..) |p, at| {
            try self.put(taps, kept_end, p);
            const room = @min(self.out.len, ids.len - 1 - p);
            @memset(&self.out, -1);
            const count = try drafter.draft(drafter.context, self.b.taps.?.view, ids[kept_end + 1 .. p + 1], p, null, self.out[0..room]);
            kept_end = p;
            var kept: usize = 0;
            while (kept < count and self.out[kept] == ids[p + 1 + kept]) kept += 1;
            made.drafted[at] = @intCast(count);
            made.kept[at] = @intCast(kept);
            made.drafts[at] = self.out;
        }
        return made;
    }
};

/// A source's drafts at each decoded position: how many, how many kept, which.
const Walk = struct {
    drafted: []u8,
    kept: []u8,
    drafts: [][rig.published.block - 1]i32,
    prompt: usize,

    const Rounds = struct { rounds: usize = 0, drafted: usize = 0, kept: usize = 0, tokens: usize = 0 };

    /// The rounds a decode along the rows would run from `first` to `end` positions past the prompt: each
    /// round's kept drafts and the target's token after them.
    fn rounds(self: Walk, first: usize, end: usize) Rounds {
        var r: Rounds = .{};
        var at = first;
        while (at < end) {
            r.rounds += 1;
            r.drafted += self.drafted[at];
            r.kept += self.kept[at];
            r.tokens += @as(usize, self.kept[at]) + 1;
            at += @as(usize, self.kept[at]) + 1;
        }
        return r;
    }

    fn perRound(self: Walk) f64 {
        const r = self.rounds(0, self.kept.len);
        return @as(f64, @floatFromInt(r.tokens)) / @as(f64, @floatFromInt(r.rounds));
    }
};

fn report(name: []const u8, rows: usize, own: Walk, theirs: Walk) void {
    const parts = [_]struct { []const u8, usize, usize }{
        .{ "all", 0, own.kept.len },
        // vLLM's primes answer ends at its 15th token; the rest is a loop of turn tokens under ignore_eos
        .{ "answer", 0, @min(15, own.kept.len) },
        .{ "loop", @min(15, own.kept.len), own.kept.len },
    };
    for (parts, 0..) |part, index| {
        if (index > 0 and !std.mem.eql(u8, name, "primes")) break;
        const label, const from, const to = part;
        if (to <= from) continue;
        var first_own: usize = 0;
        var first_theirs: usize = 0;
        var kept_own: usize = 0;
        var kept_theirs: usize = 0;
        var equal: usize = 0;
        for (from..to) |at| {
            first_own += @intFromBool(own.kept[at] > 0);
            first_theirs += @intFromBool(theirs.kept[at] > 0);
            kept_own += own.kept[at];
            kept_theirs += theirs.kept[at];
            equal += @intFromBool(own.drafted[at] == theirs.drafted[at] and std.mem.eql(i32, own.drafts[at][0..own.drafted[at]], theirs.drafts[at][0..theirs.drafted[at]]));
        }
        const n: f64 = @floatFromInt(to - from);
        const a = own.rounds(from, to);
        const b = theirs.rounds(from, to);
        std.debug.print("DFlash2 {s} {s} ({d} rows, {d} decoded positions): zspark's taps {d} of {d} kept in {d} rounds, {d:.2} a round; vLLM's {d} of {d} in {d}, {d:.2} a round\n", .{ name, label, rows, to - from, a.kept, a.drafted, a.rounds, @as(f64, @floatFromInt(a.tokens)) / @as(f64, @floatFromInt(a.rounds)), b.kept, b.drafted, b.rounds, @as(f64, @floatFromInt(b.tokens)) / @as(f64, @floatFromInt(b.rounds)) });
        std.debug.print("DFlash2 {s} {s}, every position: kept {d:.3} (zspark's) / {d:.3} (vLLM's), the first draft kept {d:.3} / {d:.3}, the same drafts {d:.3}\n", .{ name, label, @as(f64, @floatFromInt(kept_own)) / n, @as(f64, @floatFromInt(kept_theirs)) / n, @as(f64, @floatFromInt(first_own)) / n, @as(f64, @floatFromInt(first_theirs)) / n, @as(f64, @floatFromInt(equal)) / n });
    }
}
