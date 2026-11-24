//! Unlimited-OCR-2's decoder (DASH, `forward.zig`'s v2 layers, spec 147 WS-05) on its checkpoint
//! (`$ZSPARK_UNLIMITED_OCR_2`) against the references of its own HF code with Triton off
//! (`$ZSPARK_REFERENCE_UNLIMITED_OCR_2`, `tools/reference/unlimited_ocr/export_v2.py`), both cases, on one load:
//!
//! - the prompt: each layer from HF's input to it (`prefill.embeds`, `prefill.layers`), its rotated q
//!   (`prefill.q`, page2d RoPE) and its output; then carried, the last row's logits (`prefill.logits_last`);
//! - the reply fed HF's tokens in windows of 16: each row's argmax against HF's token, the detailed calls' logits
//!   (`detail.logits`);
//! - the row rule on `two_pages` (its reply's second `<PAGE>` at step 514): the prompt in one window against chunks
//!   of 1 to 100 rows (both attend kernels, blocks cut anywhere) and serial rows; the reply as serial steps against
//!   verify windows of 1 to 16 rows, one holding the `<PAGE>` at its fourth row, with drafts rejected after their
//!   first row (`<PAGE>`s among them): final rows, logits and every cache (keys, values, index keys, k_bar) equal to
//!   the bit;
//! - both cases' greedy replies with graphs, HF's n-gram rule over the reply alone and its PageCountGuard
//!   (`modeling_unlimitedocr.py` `_uocr_no_repeat_ngram`, `_uocr_page_count_guard`), against HF's `tokens`: equal
//!   up to the first step where the two part, which must be a near tie of zspark's own logits;
//! - a long document's parts (`parts_test.zig`).
//!
//! HF's logits are bf16 values (`lm_head` in bf16, then `.float()`): ties are exact there (`pick.gap` 0 at 1 and
//! 9 steps), so a step where two tokens lie a bf16 step or two apart may go either way on any sum order. The
//! figures are printed.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const requests = @import("../../../src/models/unlimited_ocr/requests.zig");
const forward = @import("../../../src/models/unlimited_ocr/forward.zig");
const model_file = @import("../../../src/models/unlimited_ocr/model.zig");
const router_file = @import("../../../src/models/unlimited_ocr/router.zig");
const ocr_testing = @import("testing.zig");

const Match = ocr_testing.Match;

const rows_max = 1024;
const scored_max = 16;
const page_token = 128827;
const eos = 1;

const floor = struct {
    /// A layer from HF's input, the prompt's rows whole (cosine)
    const layer = 0.999;
    /// The rotated q from HF's input: the qkv product's sum order alone
    const q = 0.99999;
    const logits = 0.999;
};

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

test "unlimited_ocr dash forward: rejected capacity cleans up before DASH exists" {
    const config_file = @import("../../../src/models/unlimited_ocr/config.zig");
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: config_file.Diagnostic = .{};
    var config = try config_file.Config.parse(arena.allocator(), @embedFile("fixtures/unlimited_ocr/config_dash.json"), &diagnostic);
    // Zero layers exercise the state constructor without allocating a device buffer.
    config.layers = 0;
    try testing.expectError(error.WindowTooWide, forward.State.init(testing.allocator, config, router_file.block - 1));
}

test "unlimited_ocr dash forward: the fixture's layers, the row rule and both greedy replies, on one load" {
    const path = env("ZSPARK_UNLIMITED_OCR_2") orelse return error.SkipZigTest;
    if (env("ZSPARK_REFERENCE_UNLIMITED_OCR_2") == null) return error.SkipZigTest;
    const start = cuda.monotonic();
    var m = try model_file.open(testing.allocator, testing.io, path, rows_max, scored_max);
    defer m.deinit();
    try cuda.synchronize();
    std.debug.print("unlimited_ocr dash: loaded in {d:.0} ms\n", .{@as(f64, @floatFromInt(cuda.monotonic() - start)) / 1e6});
    var failed: ?anyerror = null;
    inline for (.{ .{ "fixture", fixture }, .{ "row rule", rowRule }, .{ "greedy", greedy }, .{ "parts", @import("parts_test.zig").document } }) |part| {
        part[1](m) catch |err| {
            std.debug.print("unlimited_ocr dash {s}: {t}\n", .{ part[0], err });
            failed = failed orelse err;
        };
    }
    if (failed) |err| return err;
}

const Case = struct {
    name: []const u8,
    file: safetensors.File,
    ids: []const i32,
    tokens: []const i32,
    /// bf16 [S, hidden]: HF's prompt embeddings, the image's rows laid in
    embeds: cuda.Buffer,

    fn open(gpa: std.mem.Allocator, name: []const u8) !Case {
        var buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/{s}.safetensors", .{ env("ZSPARK_REFERENCE_UNLIMITED_OCR_2").?, name });
        var file: safetensors.File = try .open(gpa, testing.io, .cwd(), path);
        errdefer file.close();
        const e = file.get("prefill.embeds").?;
        var embeds: cuda.Buffer = try .init(.bf16, &.{ @intCast(e.shape[0]), @intCast(e.shape[1]) });
        errdefer embeds.deinit();
        try embeds.upload(e.data);
        return .{
            .name = name,
            .file = file,
            .ids = @alignCast(std.mem.bytesAsSlice(i32, file.get("input_ids").?.data)),
            .tokens = @alignCast(std.mem.bytesAsSlice(i32, file.get("tokens").?.data)),
            .embeds = embeds,
        };
    }

    fn close(self: *Case) void {
        self.embeds.deinit();
        self.file.close();
    }

    fn of(self: Case, comptime T: type, name: []const u8) []const T {
        return @alignCast(std.mem.bytesAsSlice(T, self.file.get(name).?.data));
    }

    /// Layer l's part of a [L, ...] tensor.
    fn layer(self: Case, comptime T: type, name: []const u8, l: usize) []const T {
        const all = self.of(T, name);
        const n = all.len / @as(usize, @intCast(self.file.get(name).?.shape[0]));
        return all[l * n ..][0..n];
    }

    /// A new sequence: the prompt's every row HF's embedding, its coordinates from the prompt whole.
    fn start(self: Case, state: *forward.State) !void {
        state.reset();
        state.image = .{ .first = 0, .features = self.embeds };
        try state.paged(self.ids);
    }
};

/// The token's logit less the row's largest: how far a miss is, in zspark's own logits.
fn behind(logits: []const u16, token: usize) f32 {
    const top = numeric.f32FromBf16(logits[@intCast(sampling.argmax(logits) catch unreachable)]);
    return top - numeric.f32FromBf16(logits[token]);
}

/// Two bf16 steps of the row's largest logit: a near tie.
fn nearTie(logits: []const u16, token: usize) bool {
    const top = numeric.f32FromBf16(logits[@intCast(sampling.argmax(logits) catch unreachable)]);
    return behind(logits, token) <= @abs(top) / 64.0;
}

/// The prompt layer by layer from HF's rows, then carried, then the reply fed HF's tokens; both cases.
fn fixture(m: model_file.Model) !void {
    const gpa = testing.allocator;
    const c = m.config;
    var state: forward.State = try .init(gpa, c, 4096);
    defer state.deinit();
    var b: forward.Buffers = try .init(c, rows_max, scored_max);
    defer b.deinit();
    const logits = try gpa.alloc(u16, scored_max * c.vocab);
    defer gpa.free(logits);
    var bad = false;
    for ([_][]const u8{ "one_page", "two_pages" }) |name| {
        var case: Case = try .open(gpa, name);
        defer case.close();
        const n = case.ids.len;
        const q_width = c.heads * c.head_dim;
        const h = try gpa.alloc(u16, n * c.hidden);
        defer gpa.free(h);
        const q = try gpa.alloc(u16, n * q_width);
        defer gpa.free(q);

        // each layer from HF's input
        try case.start(&state);
        try forward.embed(m, state, &b, case.ids);
        try cuda.download(b.h.rows(n), std.mem.sliceAsBytes(h));
        try testing.expectEqualSlices(u16, case.of(u16, "prefill.embeds"), h);
        var q_match: Match = .{};
        var layer_match: Match = .{};
        std.debug.print("unlimited_ocr dash {s} prompt ({d} rows), a layer from HF's input: q, output (worst row/whole/max abs):", .{ name, n });
        for (0..c.layers) |i| {
            if (i > 0) try b.h.uploadAt(0, std.mem.sliceAsBytes(case.layer(u16, "prefill.layers", i - 1)));
            try forward.layer(m, i, state, &b, n);
            try cuda.download(b.q.rows(n), std.mem.sliceAsBytes(q));
            try cuda.download(b.h.rows(n), std.mem.sliceAsBytes(h));
            const qm: Match = .of(q, u16, case.layer(u16, "prefill.q", i), q_width);
            const lm: Match = .of(h, u16, case.layer(u16, "prefill.layers", i), c.hidden);
            std.debug.print(" L{d} {d:.6}/{d:.6} {d:.5}/{d:.6}/{d:.3}", .{ i, qm.worst, qm.whole, lm.worst, lm.whole, lm.max_abs });
            q_match.merge(qm);
            layer_match.merge(lm);
        }
        std.debug.print("\n", .{});
        if (q_match.whole < floor.q or layer_match.whole < floor.layer) bad = true;

        // carried: the prompt in one window, then the reply fed HF's tokens in windows of 16
        try case.start(&state);
        try forward.forwardScoring(m, &state, &b, case.ids, .last);
        try forward.commit(m, &state, b, case.ids, n);
        try forward.logitsRow(b, 0, logits[0..c.vocab]);
        const first: Match = .of(logits[0..c.vocab], f32, case.of(f32, "prefill.logits_last"), c.vocab);
        const gaps = case.of(f32, "pick.gap");
        const calls = case.of(i32, "detail.calls");
        const detail = case.of(f32, "detail.logits");
        var details: Match = .{};
        var misses: usize = 0;
        var far: usize = 0;
        var worst_behind: f32 = 0;
        const Miss = struct {
            fn add(row: []const u16, k: usize, want: i32, gap: f32, count: *usize, far_count: *usize, worst: *f32, label: []const u8) void {
                const mine: usize = @intCast(sampling.argmax(row) catch unreachable);
                if (mine == @as(usize, @intCast(want))) return;
                count.* += 1;
                const by = behind(row, @intCast(want));
                worst.* = @max(worst.*, by);
                if (!nearTie(row, @intCast(want))) far_count.* += 1;
                std.debug.print("  {s} pick {d}: zspark {d}, HF {d}, behind by {d:.4} in zspark's logits; HF's gap {d:.4}\n", .{ label, k, mine, want, by, gap });
            }
        };
        Miss.add(logits[0..c.vocab], 0, case.tokens[0], gaps[0], &misses, &far, &worst_behind, name);
        var s: usize = 0;
        while (s < case.tokens.len) {
            const rows = @min(scored_max, case.tokens.len - s);
            const run = case.tokens[s..][0..rows];
            try forward.forwardScoring(m, &state, &b, run, .all);
            try forward.commit(m, &state, b, run, rows);
            try cuda.download(b.logits.rows(rows), std.mem.sliceAsBytes(logits[0 .. rows * c.vocab]));
            for (0..rows) |j| {
                const k = s + j + 1;
                const row = logits[j * c.vocab ..][0..c.vocab];
                // pick k chose tokens[k], or EOS after the last
                const want: i32 = if (k < case.tokens.len) case.tokens[k] else eos;
                Miss.add(row, k, want, gaps[k], &misses, &far, &worst_behind, name);
                // call c fed tokens[c - 1]
                for (calls, 0..) |call, d| {
                    if (@as(usize, @intCast(call)) != s + j + 1) continue;
                    const ref = detail[d * c.vocab ..][0..c.vocab];
                    if (std.math.isNan(ref[0])) continue;
                    details.merge(.of(row, f32, ref, c.vocab));
                }
            }
            s += rows;
        }
        std.debug.print("unlimited_ocr dash {s}: prompt logits {d:.6}/{d:.4}; reply fed HF's {d} tokens: {d} argmax misses ({d} beyond a near tie, the worst {d:.4} behind); detailed calls' logits {d:.6}/{d:.6}/{d:.4}\n", .{ name, first.whole, first.max_abs, case.tokens.len, misses, far, worst_behind, details.worst, details.whole, details.max_abs });
        if (first.whole < floor.logits or details.worst < floor.logits or far > 0) bad = true;
    }
    try testing.expect(!bad);
}

/// Every cache of a state's first `rows` rows: keys, values and index keys, and k_bar of the blocks they complete.
fn caches(gpa: std.mem.Allocator, state: forward.State, rows: usize) ![]u16 {
    const d = &state.dash;
    const kv = rows * state.keys[0].view.count() / @as(usize, @intCast(state.keys[0].view.size[0]));
    const idx = rows * router_file.dim;
    const kbar = rows / router_file.block * router_file.dim * 2;
    const each = 2 * kv + idx + kbar;
    const out = try gpa.alloc(u16, state.keys.len * each);
    errdefer gpa.free(out);
    try cuda.synchronize();
    for (0..state.keys.len) |i| {
        const at = out[i * each ..][0..each];
        try state.keys[i].downloadAt(0, std.mem.sliceAsBytes(at[0..kv]));
        try state.values[i].downloadAt(0, std.mem.sliceAsBytes(at[kv..][0..kv]));
        try d.idx_k[i].downloadAt(0, std.mem.sliceAsBytes(at[2 * kv ..][0..idx]));
        try d.kbar[i].downloadAt(0, std.mem.sliceAsBytes(at[2 * kv + idx ..][0..kbar]));
    }
    return out;
}

/// two_pages: the prompt whole against chunks and serial rows, the reply's serial steps against verify windows and
/// rejected drafts, to the bit.
fn rowRule(m: model_file.Model) !void {
    const gpa = testing.allocator;
    const c = m.config;
    var case: Case = try .open(gpa, "two_pages");
    defer case.close();
    const n = case.ids.len;
    // the reply's second <PAGE> is step 514
    const marker = std.mem.indexOfScalarPos(i32, case.tokens, 1, page_token).?;
    const reply = case.tokens[0 .. marker + 12];
    const R = reply.len;
    var b: forward.Buffers = try .init(c, rows_max, scored_max);
    defer b.deinit();
    const logits = try gpa.alloc(u16, scored_max * c.vocab);
    defer gpa.free(logits);

    // A: the prompt in one window, then the reply's rows as serial steps
    var a: forward.State = try .init(gpa, c, n + R + 16);
    defer a.deinit();
    try case.start(&a);
    try forward.forwardScoring(m, &a, &b, case.ids, .last);
    try forward.commit(m, &a, b, case.ids, n);
    const whole_last = try gpa.alloc(u16, c.vocab);
    defer gpa.free(whole_last);
    try forward.logitsRow(b, 0, whole_last);
    const serial_final = try gpa.alloc(u16, R * c.hidden);
    defer gpa.free(serial_final);
    // the logits of the steps around the marker
    const around = marker - 3;
    const serial_logits = try gpa.alloc(u16, 9 * c.vocab);
    defer gpa.free(serial_logits);
    for (reply, 0..) |token, s| {
        try forward.forwardScoring(m, &a, &b, &.{token}, .last);
        try forward.commit(m, &a, b, &.{token}, 1);
        try cuda.download(b.final.rows(1), std.mem.sliceAsBytes(serial_final[s * c.hidden ..][0..c.hidden]));
        if (s >= around and s < around + 9) try forward.logitsRow(b, 0, serial_logits[(s - around) * c.vocab ..][0..c.vocab]);
    }
    const a_caches = try caches(gpa, a, n + R);
    defer gpa.free(a_caches);

    // B: the prompt in chunks across the blocks and both attend kernels, its last 5 rows serial
    var v: forward.State = try .init(gpa, c, n + R + 16);
    defer v.deinit();
    try case.start(&v);
    const chunks = [_]usize{ 1, 7, 8, 9, 31, 33, 64, 100, 17 };
    var i: usize = 0;
    while (v.pos < n - 5) : (i += 1) {
        const rows = @min(chunks[i % chunks.len], n - 5 - v.pos);
        const run = case.ids[v.pos..][0..rows];
        try forward.forwardScoring(m, &v, &b, run, .none);
        try forward.commit(m, &v, b, run, rows);
    }
    while (v.pos < n) {
        const run = case.ids[v.pos..][0..1];
        try forward.forwardScoring(m, &v, &b, run, .last);
        try forward.commit(m, &v, b, run, 1);
    }
    try forward.logitsRow(b, 0, logits[0..c.vocab]);
    try testing.expectEqualSlices(u16, whole_last, logits[0..c.vocab]);

    // the reply in verify windows; every third a draft of junk (two <PAGE>s in it) rejected after its first row
    const windows = [_]usize{ 16, 3, 9, 1, 12, 8, 5, 16, 2, 11 };
    const junk = [_]i32{ page_token, 100, 200, page_token, 300 };
    const final = try gpa.alloc(u16, scored_max * c.hidden);
    defer gpa.free(final);
    var s: usize = 0;
    var drafts: usize = 0;
    i = 0;
    while (s < R) : (i += 1) {
        if (i % 3 == 2 and s != around) {
            var run: [6]i32 = undefined;
            run[0] = reply[s];
            @memcpy(run[1..], &junk);
            try forward.forwardScoring(m, &v, &b, &run, .all);
            try forward.commit(m, &v, b, &run, 1);
            try cuda.download(b.final.rows(1), std.mem.sliceAsBytes(final[0..c.hidden]));
            try testing.expectEqualSlices(u16, serial_final[s * c.hidden ..][0..c.hidden], final[0..c.hidden]);
            s += 1;
            drafts += 1;
            continue;
        }
        // the marker at a window's fourth row
        var rows = @min(windows[i % windows.len], R - s);
        if (s < around) rows = @min(rows, around - s) else if (s == around) rows = 9;
        const run = reply[s..][0..rows];
        try forward.forwardScoring(m, &v, &b, run, .all);
        try forward.commit(m, &v, b, run, rows);
        try cuda.download(b.final.rows(rows), std.mem.sliceAsBytes(final[0 .. rows * c.hidden]));
        try testing.expectEqualSlices(u16, serial_final[s * c.hidden ..][0 .. rows * c.hidden], final[0 .. rows * c.hidden]);
        if (s == around) for (0..rows) |j| {
            try forward.logitsRow(b, j, logits[0..c.vocab]);
            try testing.expectEqualSlices(u16, serial_logits[j * c.vocab ..][0..c.vocab], logits[0..c.vocab]);
        };
        s += rows;
    }
    const v_caches = try caches(gpa, v, n + R);
    defer gpa.free(v_caches);
    try testing.expectEqualSlices(u16, a_caches, v_caches);
    std.debug.print("unlimited_ocr dash row_rule two_pages: prompt of {d} whole = chunks of 1 to 100 + 5 serial; {d} reply rows serial = verify windows of 1 to 16 (the <PAGE> at step {d} a window's fourth row) and {d} drafts rejected after a row (final rows, logits, keys, values, index keys, k_bar)\n", .{ n, R, marker, drafts });
}

/// Both cases' greedy replies as `infer` makes them: graphs a step, HF's n-gram rule and guard as `serve` lays
/// them (`requests.Rules` of a request that asks neither n-gram field).
fn greedy(m: model_file.Model) !void {
    const gpa = testing.allocator;
    const c = m.config;
    var b: forward.Buffers = try .init(c, rows_max, 1);
    defer b.deinit();
    const logits = try gpa.alloc(u16, c.vocab);
    defer gpa.free(logits);
    const raw = try gpa.alloc(u16, c.vocab);
    defer gpa.free(raw);
    const parted = try gpa.alloc(u16, c.vocab);
    defer gpa.free(parted);
    var bad = false;
    for ([_]struct { []const u8, u32 }{ .{ "one_page", 128 }, .{ "two_pages", 1024 } }) |run| {
        var case: Case = try .open(gpa, run[0]);
        defer case.close();
        var state: forward.State = try .init(gpa, c, 4096);
        defer state.deinit();
        // HF's n-gram rule (35 over `run[1]`) and guard (`max_page_tokens` 20,000: `infer`'s default, the export's)
        var rules: requests.Rules = .of(.{ .prompt = &.{}, .max_tokens = 0, .eos = &.{eos} }, case.ids, eos);
        try testing.expectEqual([2]u32{ 35, run[1] }, [2]u32{ rules.ngram.?.size, rules.ngram.?.window });
        try testing.expectEqual(@as(u32, 20000), rules.guard.?.cap);
        var seen: std.ArrayList(u32) = .empty;
        defer seen.deinit(gpa);
        for (case.ids) |id| try seen.append(gpa, @intCast(id));
        var gen: std.ArrayList(u32) = .empty;
        defer gen.deinit(gpa);
        var graphs: forward.Graphs = try .init(gpa);
        defer graphs.deinit();
        const start = cuda.monotonic();
        try case.start(&state);
        while (state.pos < case.ids.len) {
            const rows = @min(b.rows, case.ids.len - state.pos);
            const window = case.ids[state.pos..][0..rows];
            try forward.forwardScoring(m, &state, &b, window, if (state.pos + rows == case.ids.len) .last else .none);
            try forward.commit(m, &state, b, window, rows);
        }
        try graphs.prepare(m, &state, &b, 1, .last);
        var first: @TypeOf(start) = 0;
        var parted_at: ?usize = null;
        while (gen.items.len < case.tokens.len + 64) {
            try forward.logitsRow(b, 0, logits);
            if (first == 0) first = cuda.monotonic();
            @memcpy(raw, logits);
            rules.apply(seen.items, case.ids.len, logits);
            const token: u32 = @intCast(try sampling.argmax(logits));
            const k = gen.items.len;
            const want: u32 = if (k < case.tokens.len) @intCast(case.tokens[k]) else eos;
            if (parted_at == null and token != want) {
                parted_at = k;
                @memcpy(parted, raw);
            }
            if (token == eos) break;
            try gen.append(gpa, token);
            try seen.append(gpa, token);
            const step = [_]i32{@intCast(token)};
            try graphs.forwardScoring(m, &state, &b, &step, .last);
            try forward.commit(m, &state, b, &step, 1);
        }
        const end = cuda.monotonic();
        const ref = try gpa.alloc(u32, case.tokens.len);
        defer gpa.free(ref);
        for (ref, case.tokens) |*r, t| r.* = @intCast(t);
        const distance = try ocr_testing.editDistance(gpa, gen.items, ref);
        const steps = gen.items.len;
        std.debug.print("unlimited_ocr dash greedy {s}: {d} tokens (HF's {d}), first difference {?d}, edit distance {d}; prompt {d} rows {d:.1} ms, {d:.1} tokens a second after the first\n", .{
            run[0],                                       case.tokens.len,                                                               gen.items.len, parted_at, distance, case.ids.len,
            @as(f64, @floatFromInt(first - start)) / 1e6, @as(f64, @floatFromInt(steps)) / (@as(f64, @floatFromInt(end - first)) / 1e9),
        });
        if (parted_at) |k| {
            const want: usize = if (k < case.tokens.len) @intCast(case.tokens[k]) else eos;
            const mine: usize = @intCast(try sampling.argmax(parted));
            const hf_raw = case.of(i32, "pick.argmax")[k];
            std.debug.print("  at {d}: zspark's raw argmax {d}, HF's token {d} (HF's raw argmax {d}, its top-2 gap {d:.4}); HF's token {d:.4} behind in zspark's raw logits\n", .{ k, mine, want, hf_raw, case.of(f32, "pick.gap")[k], behind(parted, want) });
            if (!nearTie(parted, want)) bad = true;
        }
        // past a near tie the replies part; they read the same pages
        if (distance * 10 > case.tokens.len) bad = true;
    }
    try testing.expect(!bad);
}
