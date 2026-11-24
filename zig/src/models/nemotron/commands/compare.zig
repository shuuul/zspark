//! `zspark compare DIR REF.safetensors... [ref_enc] [pcm=FILE] [truth=FILE] [steps=N]`: the clips'
//! reference runs (`tools/reference/nemotron_transcribe/export_model.py`'s fixtures: `enc.out`,
//! `hidden.Lxx` + `residual.Lxx` per layer over every forward, `forward.positions`/`num_rows`,
//! `prompt.ids`, `audio.placeholder_rows`, `logits.stepNN`, `tokens.greedy`) against the native
//! forward at stage level — the perception encoder's rows, then the residual stream after each of
//! the 52 blocks for the prefill and the first `steps` teacher-forced serial steps, then the logits
//! and top-1 picks. Several REFs share the one loaded model and perception, a clip each in turn.
//!
//! A clip's side files are found by REF's stem: `NAME.mel.safetensors` (its `pcm.16k` f32 samples)
//! in REF's directory or the one above it (`fixtures/model/en_5s.safetensors` -> `fixtures/
//! en_5s.mel.safetensors`), `NAME.transcript.txt` beside REF (the reference's own transcript,
//! decoded; WER reports against it). `pcm=FILE` and `truth=FILE` (a ground-truth transcript for a
//! second WER line) are taken with a single REF only.
//!
//! The encoder's rows are compared before they are injected, so an encoder's error is told from the
//! backbone's; `ref_enc` instead feeds the fixture's `enc.out` as the audio features, isolating the
//! backbone on the reference's own perception output.
//!
//! The serial replay is honest about the reference's speculative forwards: an MTP verify window's
//! rows past its accepted count were computed but never committed, so their positions map out
//! (`checks/reference.zig`'s `rows`), and each `logits.step{f}` is compared at the committed
//! position it scored — never against a neighbouring step, and a position two forwards produced rows
//! for keeps the first.

const std = @import("std");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const numeric = @import("../../../weights/numeric.zig");
const sampling = @import("../../../decode/sampling.zig");
const placeholders = @import("../../../decode/placeholders.zig");
const text_tokenizer = @import("../../../text/tokenizer.zig");
const config_file = @import("../config.zig");
const layout_file = @import("../layout.zig");
const model_file = @import("../model.zig");
const state_file = @import("../state.zig");
const forward_file = @import("../forward.zig");
const draft_file = @import("../draft.zig");
const perception_file = @import("../perception.zig");
const reference = @import("../checks/reference.zig");
const experts_file = @import("../../../weights/experts.zig");

pub const usage = "compare DIR REF.safetensors... [ref_enc] [pcm=FILE] [truth=FILE] [steps=N]";

const Opened = cli.Opened(config_file.Config, layout_file.Layout, config_file.Diagnostic);

/// The clip's 16 kHz f32 samples: `pcm=`'s safetensors' `pcm.16k`, or the file read raw as f32 when
/// it is not a safetensors.
fn samplesOf(init: std.process.Init, path: []const u8) !?[]f32 {
    const arena = init.arena.allocator();
    if (safetensors.File.open(init.gpa, init.io, .cwd(), path)) |*held| {
        var file = held.*;
        defer file.close();
        const t = file.get("pcm.16k") orelse return null;
        if (t.dtype != .F32) return error.UnexpectedDtype;
        const bits = std.mem.bytesAsSlice(u32, t.data);
        const out = try arena.alloc(f32, bits.len);
        for (out, bits) |*v, b| v.* = @bitCast(b);
        return out;
    } else |err| switch (err) {
        error.FileNotFound => return null,
        error.InvalidSafetensors => {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, path, arena, .unlimited);
            if (bytes.len % 4 != 0) return error.UnexpectedShape;
            const bits = std.mem.bytesAsSlice(u32, bytes);
            const out = try arena.alloc(f32, bits.len);
            for (out, bits) |*v, b| v.* = @bitCast(b);
            return out;
        },
        else => return err,
    }
}

fn melPath(io: std.Io, arena: std.mem.Allocator, ref_path: []const u8) !?[]const u8 {
    const stem = if (std.mem.endsWith(u8, ref_path, ".safetensors")) ref_path[0 .. ref_path.len - ".safetensors".len] else ref_path;
    const direct = try std.fmt.allocPrint(arena, "{s}.mel.safetensors", .{stem});
    if (std.Io.Dir.cwd().access(io, direct, .{})) |_| return direct else |_| {}
    // fixtures/model/en_5s.safetensors -> fixtures/en_5s.mel.safetensors
    const dir = std.fs.path.dirname(ref_path) orelse return null;
    const parent = std.fs.path.dirname(dir) orelse return null;
    const base = std.fs.path.basename(stem);
    const up = try std.fmt.allocPrint(arena, "{s}/{s}.mel.safetensors", .{ parent, base });
    if (std.Io.Dir.cwd().access(io, up, .{})) |_| return up else |_| {}
    return null;
}

/// `NAME.safetensors` -> `NAME.transcript.txt` beside REF, or null.
fn transcriptPath(io: std.Io, arena: std.mem.Allocator, ref_path: []const u8) !?[]const u8 {
    const stem = if (std.mem.endsWith(u8, ref_path, ".safetensors")) ref_path[0 .. ref_path.len - ".safetensors".len] else ref_path;
    const path = try std.fmt.allocPrint(arena, "{s}.transcript.txt", .{stem});
    if (std.Io.Dir.cwd().access(io, path, .{})) |_| return path else |_| {}
    return null;
}

/// `mine` bf16 rows downloaded from `view`, compared with `theirs` f32 rows of the same width.
fn against(theirs: []const f32, view: cuda.View, arena: std.mem.Allocator) !reference.Stats {
    const mine = try arena.alloc(u16, view.count());
    try cuda.synchronize();
    try cuda.download(view, std.mem.sliceAsBytes(mine));
    return reference.compareBf16(mine, theirs, @intCast(view.size[0]), false, arena);
}

/// A clip's tensors; completed layer streams stay in the mapping and are read a row at a time.
const Fixture = struct {
    file: safetensors.File,
    prompt: []i32,
    placeholder_rows: []i32,
    num_rows: []i32,
    positions: []i32,
    greedy: []i32,
    map: reference.Rows,
    layers: []reference.LayerRef,
    enc: safetensors.Tensor,

    fn open(init: std.process.Init, arena: std.mem.Allocator, path: []const u8, layers_n: usize, hidden: usize) !?Fixture {
        var file = safetensors.File.open(init.gpa, init.io, .cwd(), path) catch |err| {
            std.log.err("{s}: {t} — REF is one fixture file export_model.py wrote, NAME.safetensors", .{ path, err });
            return null;
        };
        errdefer file.close();
        const prompt = reference.ints(arena, &file, "prompt.ids") catch return missing(&file, path, "prompt.ids");
        const rows = reference.ints(arena, &file, "audio.placeholder_rows") catch return missing(&file, path, "audio.placeholder_rows");
        const num_rows = reference.ints(arena, &file, "forward.num_rows") catch return missing(&file, path, "forward.num_rows");
        const positions = reference.ints(arena, &file, "forward.positions") catch return missing(&file, path, "forward.positions");
        const greedy = reference.ints(arena, &file, "tokens.greedy") catch return missing(&file, path, "tokens.greedy");
        if (num_rows.len == 0 or prompt.len == 0) {
            std.log.err("{s}: empty prompt or forward capture", .{path});
            file.close();
            return null;
        }
        if (positions.len < prompt.len) {
            std.log.err("{s}: {d} positions cannot cover the {d}-row prompt", .{ path, positions.len, prompt.len });
            file.close();
            return null;
        }
        for (positions[0..prompt.len], 0..) |p, i| if (p != i) {
            std.log.err("{s}: prefill row {d} sits at position {d}; complete chunked-prefill capture is required", .{ path, i, p });
            file.close();
            return null;
        };
        const enc = file.get("enc.out") orelse return missing(&file, path, "enc.out");
        if (enc.dtype != .BF16 or enc.shape.len != 2 or enc.shape[1] != hidden or enc.shape[0] != rows.len) {
            std.log.err("{s}: enc.out {any} is not bf16 [{d}, {d}]", .{ path, enc.shape, rows.len, hidden });
            file.close();
            return null;
        }
        // `forward.input_ids` can be absent entirely (every forward on inputs_embeds) or miss the
        // prefill's; `rows` accepts either, prompt standing in for forward 0.
        const input_ids = reference.ints(arena, &file, "forward.input_ids") catch &.{};
        const map = reference.rows(arena, &file, num_rows, positions, input_ids, prompt, greedy) catch {
            std.log.err("{s}: forward.input_ids/positions do not line up with num_rows", .{path});
            file.close();
            return null;
        };
        const layers = try arena.alloc(reference.LayerRef, layers_n);
        for (layers, 0..) |*t, layer| {
            t.* = reference.layerTensors(&file, layer) catch return missing(&file, path, "hidden.Lxx/residual.Lxx");
            if (t.hidden.shape[0] != positions.len or t.hidden.shape[1] != hidden) {
                std.log.err("{s}: layer {d} does not cover [{d}, {d}] captured values", .{ path, layer, positions.len, hidden });
                file.close();
                return null;
            }
        }
        return .{ .file = file, .prompt = prompt, .placeholder_rows = rows, .num_rows = num_rows, .positions = positions, .greedy = greedy, .map = map, .layers = layers, .enc = enc };
    }

    fn missing(file: *safetensors.File, path: []const u8, tensor: []const u8) ?Fixture {
        std.log.err("{s}: no {s} — the fixture predates the exporter's replay capture", .{ path, tensor });
        file.close();
        return null;
    }
};

/// Per-layer aggregates over the committed rows a serial replay produced: summed squared
/// differences and reference squares (the all-rows rel rms), dot products and magnitudes (the
/// cosine), the worst row's rel rms and the widest single-element miss.
const Acc = struct { diff: f64 = 0, ref: f64 = 0, dot: f64 = 0, mine: f64 = 0, max_abs: f64 = 0, worst_rel: f64 = 0, worst_row: usize = 0, n: usize = 0 };

/// Folds the taps of `rows_n` window rows starting at committed position `pos` into `acc`: `taps`
/// is the taps buffer's download — a row's `acc.len` layers side by side of `width` bf16 values
/// (`setTaps`' layout), layer l's slice compared with the reference's completed row at the
/// committed position (a speculative row's position maps out and is skipped).
fn foldRows(acc: []Acc, layers: []const reference.LayerRef, map: reference.Rows, taps: []const u16, rows_n: usize, pos: usize, width: usize, scratch: []f32) void {
    const stride = acc.len * width;
    for (0..rows_n) |r| {
        const position = pos + r;
        if (position >= map.row_of.len) break;
        const ref_row = map.row_of[position];
        if (ref_row < 0) continue;
        for (acc, layers, 0..) |*a, all, layer| {
            reference.completedRow(all, @intCast(ref_row), scratch);
            const mine = taps[r * stride + layer * width ..][0..width];
            var diff: f64 = 0;
            var ref_sq: f64 = 0;
            for (mine, scratch) |b, y| {
                const x = numeric.f32FromBf16(b);
                const d = x - y;
                diff += d * d;
                ref_sq += y * y;
                a.dot += x * y;
                a.mine += x * x;
                a.max_abs = @max(a.max_abs, @abs(d));
            }
            a.diff += diff;
            a.ref += ref_sq;
            const rel = if (ref_sq > 0) @sqrt(diff / ref_sq) else @sqrt(diff);
            if (rel >= a.worst_rel) {
                a.worst_rel = rel;
                a.worst_row = position;
            }
            a.n += 1;
        }
    }
}

/// One clip through the loaded model: the encoder's rows against `enc.out`, then the serial replay
/// with them (or the fixture's own, `ref_enc`) over the placeholder rows, taps folded per layer,
/// logits and tokens compared, the reply decoded and the WER lines printed.
fn clip(init: std.process.Init, stdout: *std.Io.Writer, ref_path: []const u8, fixture: *Fixture, m: model_file.Model, perception: *perception_file.Perception, state: *state_file.State, buffers: *state_file.Buffers, nlayers: usize, hidden: usize, pcm: ?[]const f32, ref_enc: bool, truth: ?[]const u8, steps: usize, tokenizer: ?*text_tokenizer.Tokenizer) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const c = &m.config;
    const b_cfg = c.backbone;
    try stdout.print("== {s}: {d} prompt rows, {d} audio rows, {d} greedy tokens, {d} forwards\n", .{ ref_path, fixture.prompt.len, fixture.placeholder_rows.len, fixture.greedy.len, fixture.num_rows.len });

    // The audio features the forward injects: the native encoder's by default (compared with
    // enc.out first), or the fixture's own under ref_enc.
    var enc_mine: ?cuda.Buffer = null;
    defer if (enc_mine) |*buf| buf.deinit();
    var features: cuda.View = undefined;
    if (pcm) |pcm_samples| {
        enc_mine = try .init(.bf16, &.{ fixture.enc.shape[0], hidden });
        try perception.encode(arena, pcm_samples, enc_mine.?.view);
        const theirs = try arena.alloc(f32, @intCast(fixture.enc.count()));
        for (theirs, std.mem.bytesAsSlice(u16, fixture.enc.data)) |*v, bits| v.* = numeric.f32FromBf16(bits);
        const s = try against(theirs, enc_mine.?.view, arena);
        try stdout.print("encoder: {d} rows x {d}, rel rms {e:.3}, worst row {d} {e:.3}, max abs {e:.3}, cosine {d:.6}\n", .{ fixture.enc.shape[0], hidden, s.rel_all, s.worst_row, s.rel_worst, s.max_abs, s.cosine });
    } else {
        try stdout.writeAll("encoder: no pcm (NAME.mel.safetensors unfound); enc.out not compared\n");
    }
    if (ref_enc) {
        if (enc_mine == null) enc_mine = try .init(.bf16, &.{ fixture.enc.shape[0], hidden });
        try cuda.upload(enc_mine.?.view, fixture.enc.data);
        try stdout.writeAll("audio features: the reference's enc.out (ref_enc)\n");
    } else if (enc_mine == null) {
        std.log.err("{s}: no pcm and no ref_enc: the audio rows have no features", .{ref_path});
        return 1;
    }
    features = enc_mine.?.view;

    try state.reset();
    const audio_rows_host = try arena.alloc(u32, fixture.placeholder_rows.len);
    for (audio_rows_host, fixture.placeholder_rows) |*r, p| r.* = @intCast(p);
    state.audio = .{ .features = features, .rows = audio_rows_host };

    const taps_host = try arena.alloc(u16, buffers.taps.?.bytes() / 2);
    const logits_host = try gpa.alloc(u16, b_cfg.vocab);
    defer gpa.free(logits_host);
    const layer_acc = try arena.alloc(Acc, nlayers);
    @memset(layer_acc, .{});

    const row_scratch = try arena.alloc(f32, hidden);
    const chunk_rows = @min(buffers.rows, m.max_rows);
    var done: usize = 0;
    while (done < fixture.prompt.len) {
        const n = @min(chunk_rows, fixture.prompt.len - done);
        const part = fixture.prompt[done..][0..n];
        const score: forward_file.Score = if (done + n == fixture.prompt.len) .last else .none;
        try forward_file.forwardScoring(m, state, buffers, part, score);
        try cuda.download(buffers.taps.?.rows(n), std.mem.sliceAsBytes(taps_host[0 .. n * nlayers * hidden]));
        if (score == .last) try forward_file.logitsRow(buffers.*, 0, logits_host);
        try forward_file.commit(m, state, buffers.*, part, n);
        foldRows(layer_acc, fixture.layers, fixture.map, taps_host, n, done, hidden, row_scratch);
        done += n;
    }

    // `logits.step{f}` compared where its scored committed position lands in the replay.
    var mine_reply: std.ArrayList(i32) = .empty;
    defer mine_reply.deinit(gpa);
    var logits_rel_sum: f64 = 0;
    var logits_rel_n: usize = 0;
    var logits_top1_diff: usize = 0;
    const compareLogits = struct {
        fn at(stdout2: *std.Io.Writer, arena2: std.mem.Allocator, file: *const safetensors.File, map: reference.Rows, num_rows: []const i32, mine: []const u16, pos: usize, vocab: usize, sum: *f64, n: *usize, differing: *usize) !void {
            for (map.scored_pos, map.scored_f, map.scored_r) |sp, f, r| {
                if (sp != pos) continue;
                const theirs = try reference.logits(arena2, file, f, @intCast(num_rows[f]), r, vocab);
                const widened = try arena2.alloc(f32, vocab);
                for (widened, mine) |*v, bits| v.* = numeric.f32FromBf16(bits);
                const s = reference.compare(widened, theirs, 1, true);
                sum.* += s.rel_all;
                n.* += 1;
                differing.* += s.picks_differing;
                try stdout2.print("logits pos {d} (step {d:0>2} row {d}): rel rms {e:.3}, top1 {s}\n", .{ pos, f, r, s.rel_all, if (s.picks_differing == 0) "same" else "DIFFERS" });
            }
        }
    }.at;
    try compareLogits(stdout, arena, &fixture.file, fixture.map, fixture.num_rows, logits_host, fixture.prompt.len - 1, b_cfg.vocab, &logits_rel_sum, &logits_rel_n, &logits_top1_diff);
    const first_pick = try sampling.argmax(logits_host);

    try mine_reply.append(gpa, first_pick);
    // Teacher-forced serial steps: step s feeds the reference's greedy[s] and the step's logits
    // predict greedy[s + 1]; `logits.step{f}` compares where its scored committed row sits.
    const take = @min(steps, fixture.greedy.len);
    for (0..take) |s| {
        const pos = state.pos;
        if (pos >= fixture.map.row_of.len) break;
        const tok: [1]i32 = .{fixture.greedy[s]};
        try forward_file.forwardScoring(m, state, buffers, &tok, .all);
        try cuda.synchronize();
        try cuda.download(buffers.taps.?.rows(1), std.mem.sliceAsBytes(taps_host[0 .. nlayers * hidden]));
        try forward_file.logitsRow(buffers.*, 0, logits_host);
        foldRows(layer_acc, fixture.layers, fixture.map, taps_host, 1, pos, hidden, row_scratch);
        try forward_file.commit(m, state, buffers.*, &tok, 1);
        const pick = try sampling.argmax(logits_host);
        if (s + 1 < fixture.greedy.len) try mine_reply.append(gpa, pick);
        try compareLogits(stdout, arena, &fixture.file, fixture.map, fixture.num_rows, logits_host, pos, b_cfg.vocab, &logits_rel_sum, &logits_rel_n, &logits_top1_diff);
    }

    try stdout.print("layers (bf16 hidden+residual of the committed rows):\n", .{});
    for (layer_acc, 0..) |a, layer| {
        if (a.n == 0) continue;
        const rel = if (a.ref > 0) @sqrt(a.diff / a.ref) else @sqrt(a.diff);
        const cosine = if (a.mine > 0 and a.ref > 0) a.dot / @sqrt(a.mine * a.ref) else 1;
        try stdout.print("  L{d:0>2} rows {d:>4} rel rms {e:>9.2} worst {e:>9.2} (pos {d}) max abs {e:>9.2} cos {d:.6}\n", .{ layer, a.n, rel, a.worst_rel, a.worst_row, a.max_abs, cosine });
    }
    if (fixture.map.unmapped > 0) try stdout.print("{d} positions have no committed captured row; their layers are skipped\n", .{fixture.map.unmapped});
    var same: usize = 0;
    for (mine_reply.items, 0..) |t, i| {
        if (i < fixture.greedy.len and t == fixture.greedy[i]) same += 1;
    }
    try stdout.print("top-1: {d}/{d} tokens match; {d} logits compared of {d} forwards, mean rel rms {e:.3}, {d} picks differ\n", .{ same, mine_reply.items.len, logits_rel_n, fixture.num_rows.len, if (logits_rel_n > 0) logits_rel_sum / @as(f64, @floatFromInt(logits_rel_n)) else 0, logits_top1_diff });
    if (fixture.file.get("logits.step00")) |t| {
        if (t.shape.len == 2 and t.shape[1] != b_cfg.vocab) try stdout.print("note: the fixture's logits carry {d} columns to the checkpoint's {d} — the tokenizer's added columns; the first {d} are compared\n", .{ t.shape[1], b_cfg.vocab, b_cfg.vocab });
    }

    if (tokenizer) |tokenizer_| {
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        const ids_u32 = try arena.alloc(u32, mine_reply.items.len);
        for (ids_u32, mine_reply.items) |*id, t| id.* = @intCast(@max(0, t));
        try tokenizer_.decode(gpa, ids_u32, .{ .skip_special = true }, &text);
        try stdout.print("teacher-forced diagnostic text (not an autoregressive reply): {s}\n", .{text.items});
    }

    // Score a real reply: every new token conditions the next step, up to EOS or the actual context.
    try state.reset();
    state.audio = .{ .features = features, .rows = audio_rows_host };
    done = 0;
    var free_pick: i32 = -1;
    while (done < fixture.prompt.len) {
        const n = @min(chunk_rows, fixture.prompt.len - done);
        const part = fixture.prompt[done..][0..n];
        const score: forward_file.Score = if (done + n == fixture.prompt.len) .last else .none;
        try forward_file.forwardScoring(m, state, buffers, part, score);
        if (score == .last) {
            try forward_file.logitsRow(buffers.*, 0, logits_host);
            free_pick = try sampling.argmax(logits_host);
        }
        try forward_file.commit(m, state, buffers.*, part, n);
        done += n;
    }
    var free: std.ArrayList(u32) = .empty;
    defer free.deinit(gpa);
    var ended = false;
    const room = state.capacity - fixture.prompt.len;
    while (free.items.len < room) {
        const token: u32 = @intCast(free_pick);
        try free.append(gpa, token);
        if (std.mem.indexOfScalar(u32, c.eos, token) != null) {
            ended = true;
            break;
        }
        if (free.items.len == room) break;
        const next = [_]i32{free_pick};
        try forward_file.forwardScoring(m, state, buffers, &next, .all);
        try forward_file.logitsRow(buffers.*, 0, logits_host);
        try forward_file.commit(m, state, buffers.*, &next, 1);
        free_pick = try sampling.argmax(logits_host);
    }
    try stdout.print("free greedy: {d} tokens, finish {s}\n", .{ free.items.len, if (ended) "eos" else "length" });
    if (tokenizer) |tokenizer_| {
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        try tokenizer_.decode(gpa, free.items, .{ .skip_special = true }, &text);
        try stdout.print("reply (free greedy): {s}\n", .{text.items});
        if (try transcriptPath(init.io, arena, ref_path)) |p| {
            const ref_text = std.Io.Dir.cwd().readFileAlloc(init.io, p, arena, .unlimited) catch |err| {
                std.log.err("{s}: {t}", .{ p, err });
                return 1;
            };
            const w = try reference.wer(arena, std.mem.trim(u8, ref_text, " \t\r\n"), std.mem.trim(u8, text.items, " \t\r\n"));
            try stdout.print("{s} vs reference transcript: {d:.3} ({d}/{d} {s})\n", .{ if (w.codepoints) "CER" else "WER", w.rate(), w.distance, w.units, if (w.codepoints) "codepoints" else "words" });
        }
        if (truth) |truth_text| {
            const w = try reference.wer(arena, std.mem.trim(u8, truth_text, " \t\r\n"), std.mem.trim(u8, text.items, " \t\r\n"));
            try stdout.print("{s} vs ground truth: {d:.3} ({d}/{d} {s})\n", .{ if (w.codepoints) "CER" else "WER", w.rate(), w.distance, w.units, if (w.codepoints) "codepoints" else "words" });
        }
    }
    return 0;
}

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    var pcm_path: ?[]const u8 = null;
    var truth_path: ?[]const u8 = null;
    var steps: usize = 32;
    var ref_enc = false;
    var refs: std.ArrayList([]const u8) = .empty;
    for (words[1..]) |word| {
        if (cli.value(word, "pcm=")) |p| {
            pcm_path = p;
        } else if (cli.value(word, "truth=")) |p| {
            truth_path = p;
        } else if (cli.value(word, "steps=")) |n| {
            steps = try std.fmt.parseInt(usize, n, 10);
        } else if (std.mem.eql(u8, word, "ref_enc")) {
            ref_enc = true;
        } else if (std.mem.startsWith(u8, word, "--")) {
            try stdout.print("usage: zspark {s}\n", .{usage});
            return 2;
        } else {
            try refs.append(arena, word);
        }
    }
    if (refs.items.len == 0 or (refs.items.len > 1 and (pcm_path != null or truth_path != null))) {
        try stdout.print("usage: zspark {s}\n(pcm= and truth= go with a single REF)\n", .{usage});
        return 2;
    }
    var opened: Opened = .{};
    defer opened.close(init.io);
    if (!try opened.open(init, words[0], .raise)) return 1;
    const c = opened.config;
    const b_cfg = c.backbone;
    const nlayers = b_cfg.types.len;
    const hidden = b_cfg.hidden;

    // Every clip's fixture and PCM on the host first, so the perception is sized for the longest
    // and a bad fixture fails before the weights are loaded.
    const fixtures = try arena.alloc(Fixture, refs.items.len);
    const pcms = try arena.alloc(?[]f32, refs.items.len);
    var fixtures_n: usize = 0;
    defer for (fixtures[0..fixtures_n]) |*f| f.file.close();
    var max_prompt: usize = 0;
    const capacity = @import("../served.zig").default_context;
    for (refs.items, 0..) |path, i| {
        fixtures[i] = (try Fixture.open(init, arena, path, nlayers, hidden)) orelse return 1;
        fixtures_n += 1;
        max_prompt = @max(max_prompt, fixtures[i].prompt.len);
        if (fixtures[i].prompt.len >= capacity) {
            std.log.err("{s}: prompt exhausts the card's {d}-token context", .{ path, capacity });
            return 1;
        }
        pcms[i] = if (pcm_path) |p|
            try samplesOf(init, p)
        else if (try melPath(init.io, arena, path)) |p|
            try samplesOf(init, p)
        else
            null;
    }
    const window_rows = @min(@max(max_prompt, draft_file.forward_max_rows), 2048);

    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    var scratch = try cuda.Scratch.create(gpa);
    defer scratch.destroy();
    var m = try model_file.Model.load(arena, gpa, &c, &opened.checkpoint.?, &opened.diagnostic, &staging, scratch, window_rows, window_rows);
    defer m.deinit();
    try staging.settle();
    // Long clips use the same bounded groups as serving; the encoder iterates over all windows.
    var perception = try perception_file.Perception.load(gpa, arena, &c, &opened.checkpoint.?, 8);
    defer perception.deinit();
    var state: state_file.State = try .init(gpa, &c, capacity);
    defer state.deinit();
    var buffers: state_file.Buffers = try .init(&c, window_rows, draft_file.forward_max_rows, capacity);
    defer buffers.deinit();
    const tap_layers = try arena.alloc(u32, nlayers);
    for (tap_layers, 0..) |*t, i| t.* = @intCast(i);
    try buffers.setTaps(tap_layers);
    var tokenizer: ?text_tokenizer.Tokenizer = null;
    var tok_diag: text_tokenizer.Diagnostic = .{};
    if (text_tokenizer.Tokenizer.load(gpa, init.io, opened.dir.?, "tokenizer.json", &tok_diag)) |tok| {
        tokenizer = tok;
    } else |_| {}
    defer if (tokenizer) |*tok| tok.deinit(gpa);
    const truth = if (truth_path) |p| std.Io.Dir.cwd().readFileAlloc(init.io, p, arena, .unlimited) catch |err| {
        std.log.err("{s}: {t}", .{ p, err });
        return 1;
    } else null;

    for (fixtures, pcms, refs.items) |*fixture, pcm, path| {
        // a clip's audio rows must match its fixture's before the forward takes them
        if (pcm) |pcm_samples| {
            const audio_rows = try perception.rows(arena, pcm_samples.len);
            if (audio_rows != fixture.enc.shape[0]) {
                std.log.err("{s}: the clip encodes to {d} rows, the reference's enc.out holds {d}", .{ path, audio_rows, fixture.enc.shape[0] });
                return 1;
            }
        }
        const code = try clip(init, stdout, path, fixture, m, &perception, &state, &buffers, nlayers, hidden, pcm, ref_enc, truth, steps, if (tokenizer) |*t| t else null);
        if (code != 0) return code;
    }
    return 0;
}
