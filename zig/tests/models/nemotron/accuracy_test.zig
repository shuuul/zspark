//! WS-05's accuracy gate on the real checkpoint (`zspark compare`'s stages, held, not reported): with
//! `$ZSPARK_NEMOTRON_NVFP4` (the NVFP4 checkpoint's directory) and `$ZSPARK_REFERENCE_NEMOTRON` (the WS-04
//! fixtures' root) set, every clip the root `manifest.json`'s `clips` declare — `model/NAME.safetensors`
//! by name; a sibling file the manifest does not declare is not a clip — is replayed: the clip's `pcm.16k`
//! through `perception.zig` against `enc.out`, the prompt then the first serial steps through the
//! backbone's taps against `hidden.L*`/`residual.L*` and `logits.step*`, the free greedy reply against
//! `transcript.txt` —
//! and the per-stage cosine floors, the top-1 agreement and the WER floor are held. Skipped without the
//! env vars; the fixtures' own transcript stands for ground truth (its WER against the manifest's text is
//! `wer/refs.jsonl`'s, recorded in the spec entry).

const std = @import("std");
const cuda = @import("../../../src/cuda/cuda.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const config_file = @import("../../../src/models/nemotron/config.zig");
const layout_file = @import("../../../src/models/nemotron/layout.zig");
const model_file = @import("../../../src/models/nemotron/model.zig");
const state_file = @import("../../../src/models/nemotron/state.zig");
const forward_file = @import("../../../src/models/nemotron/forward.zig");
const draft_file = @import("../../../src/models/nemotron/draft.zig");
const perception_file = @import("../../../src/models/nemotron/perception.zig");
const served_file = @import("../../../src/models/nemotron/served.zig");
const reference = @import("../../../src/models/nemotron/checks/reference.zig");
const experts_file = @import("../../../src/weights/experts.zig");
const text_tokenizer = @import("../../../src/text/tokenizer.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// The floors the WS-05 run measured on the nine fixtures (`scratch-151/ws05/`), with headroom: the
/// encoder's cosine (worst 0.9937 on zh_600s), every backbone layer's over the committed rows (worst
/// 0.9441, zh_140s L51 — the Chinese clips run looser than English's), and the reply's WER against the
/// fixture's transcript (worst real 0.042 on zh_600s; en_3500s's 0.257 is the 65,536-token context's
/// truncation of a 21,840-token reply, not a decode error — the bound has to hold it too).
const enc_cosine_floor = 0.99;
const layer_cosine_floor = 0.94;
const wer_floor = 0.30;

/// A clip's fixture tensors as `commands/compare.zig` reads them (its `Fixture`), host side.
const Clip = struct {
    file: safetensors.File,
    name: []const u8,
    prompt: []i32,
    placeholder_rows: []i32,
    num_rows: []i32,
    positions: []i32,
    greedy: []i32,
    map: reference.Rows,
    layers: []reference.LayerRef,
    enc: safetensors.Tensor,
    transcript: []const u8,
};

fn openClip(gpa: Allocator, arena: Allocator, io: std.Io, model_dir: std.Io.Dir, path: []const u8, layers_n: usize, hidden: usize) !Clip {
    var file = try safetensors.File.open(gpa, io, model_dir, path);
    errdefer file.close();
    const stem = path[0 .. path.len - ".safetensors".len];
    const prompt = try reference.ints(arena, &file, "prompt.ids");
    const rows = try reference.ints(arena, &file, "audio.placeholder_rows");
    const num_rows = try reference.ints(arena, &file, "forward.num_rows");
    const positions = try reference.ints(arena, &file, "forward.positions");
    const greedy = try reference.ints(arena, &file, "tokens.greedy");
    const input_ids = reference.ints(arena, &file, "forward.input_ids") catch &.{};
    const map = try reference.rows(arena, &file, num_rows, positions, input_ids, prompt, greedy);
    const enc = file.get("enc.out") orelse return error.MissingTensor;
    try testing.expect(enc.dtype == .BF16 and enc.shape.len == 2 and enc.shape[0] == rows.len and enc.shape[1] == hidden);
    const layers = try arena.alloc(reference.LayerRef, layers_n);
    for (layers, 0..) |*t, layer| {
        t.* = try reference.layerTensors(&file, layer);
        try testing.expectEqual(positions.len, @as(usize, @intCast(t.hidden.shape[0])));
        try testing.expectEqual(hidden, @as(usize, @intCast(t.hidden.shape[1])));
    }
    var name_buf: [128]u8 = undefined;
    const transcript_path = try std.fmt.bufPrint(&name_buf, "{s}.transcript.txt", .{stem});
    const transcript = try model_dir.readFileAlloc(io, transcript_path, arena, .limited(1 << 20));
    return .{ .file = file, .name = stem, .prompt = prompt, .placeholder_rows = rows, .num_rows = num_rows, .positions = positions, .greedy = greedy, .map = map, .layers = layers, .enc = enc, .transcript = transcript };
}

/// A clip's `pcm.16k` f32 samples: `NAME.mel.safetensors` beside the fixtures root (`model/`'s parent).
fn pcmOf(arena: Allocator, io: std.Io, fixtures: std.Io.Dir, stem: []const u8) ![]f32 {
    var name_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&name_buf, "{s}.mel.safetensors", .{stem});
    var file = try safetensors.File.open(arena, io, fixtures, path);
    defer file.close();
    const t = file.get("pcm.16k") orelse return error.MissingTensor;
    const bits = std.mem.bytesAsSlice(u32, t.data);
    const out = try arena.alloc(f32, bits.len);
    for (out, bits) |*v, b| v.* = @bitCast(b);
    return out;
}

/// The clip fixture paths `manifest.json`'s `clips` declare, `model/NAME.safetensors` in ascending name
/// order. The manifest is the fixtures' clip set (WS-04 writes one entry a clip): a safetensors beside the
/// declared ones — an exporter or debug run's diagnostic — is not replayed, and a clip whose file is
/// missing still fails.
fn clipFiles(arena: Allocator, io: std.Io, fixtures: std.Io.Dir, model_dir: std.Io.Dir) ![]const []const u8 {
    const raw = try fixtures.readFileAlloc(io, "manifest.json", arena, .limited(1 << 22));
    const manifest = try std.json.parseFromSliceLeaky(std.json.Value, arena, raw, .{});
    const declared = manifest.object.get("clips").?.object;
    const files = try arena.alloc([]const u8, declared.count());
    for (files, declared.keys()) |*file, name| {
        file.* = try std.fmt.allocPrint(arena, "{s}.safetensors", .{name});
        try model_dir.access(io, file.*, .{});
    }
    std.mem.sort([]const u8, files, {}, struct {
        fn before(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.before);
    return files;
}

/// `acc`'s per-layer sums folded over `n` window rows from committed `pos` (`compare.zig`'s foldRows):
/// taps buffer row r's layers side by side of `width` bf16 against the reference's completed row at the
/// position, unmapped (speculative) positions skipped.
const Acc = struct { diff: f64 = 0, ref: f64 = 0, dot: f64 = 0, mine: f64 = 0, n: usize = 0 };

fn fold(acc: []Acc, layers: []const reference.LayerRef, map: reference.Rows, taps: []const u16, rows_n: usize, pos: usize, width: usize, scratch: []f32) void {
    const stride = acc.len * width;
    for (0..rows_n) |r| {
        const position = pos + r;
        if (position >= map.row_of.len) break;
        const ref_row = map.row_of[position];
        if (ref_row < 0) continue;
        for (acc, layers, 0..) |*a, all, layer| {
            reference.completedRow(all, @intCast(ref_row), scratch);
            for (taps[r * stride + layer * width ..][0..width], scratch) |b, y| {
                const x = numeric.f32FromBf16(b);
                const d = x - y;
                a.diff += d * d;
                a.ref += y * y;
                a.dot += x * y;
                a.mine += x * x;
            }
            a.n += 1;
        }
    }
}

test "the real checkpoint's per-stage cosines against the WS-04 fixtures, and the WER floor" {
    const ck_path = env("ZSPARK_NEMOTRON_NVFP4") orelse return error.SkipZigTest;
    const fix_path = env("ZSPARK_REFERENCE_NEMOTRON") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // the checkpoint's config, manifest and tensors (the commands' `Opened` steps, by hand)
    var ck_dir = try std.Io.Dir.cwd().openDir(io, ck_path, .{});
    defer ck_dir.close(io);
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.read(arena, io, ck_dir, &diagnostic);
    var checkpoint: safetensors.Checkpoint = try .open(gpa, io, .cwd(), ck_path);
    defer checkpoint.close();
    _ = try layout_file.Layout.init(arena, &c, &checkpoint, &diagnostic);
    const nlayers = c.backbone.types.len;
    const hidden = c.backbone.hidden;

    // the fixtures: the root manifest's `clips` each as `model/NAME.safetensors`, its pcm and transcript
    var fixtures_dir = try std.Io.Dir.cwd().openDir(io, fix_path, .{});
    defer fixtures_dir.close(io);
    var model_dir = try fixtures_dir.openDir(io, "model", .{});
    defer model_dir.close(io);
    var clips: std.ArrayList(Clip) = .empty;
    var pcms: std.ArrayList([]f32) = .empty;
    defer for (clips.items) |*clip| clip.file.close();
    var max_prompt: usize = 0;
    for (try clipFiles(arena, io, fixtures_dir, model_dir)) |path| {
        const clip = try openClip(gpa, arena, io, model_dir, path, nlayers, hidden);
        try clips.append(arena, clip);
        try pcms.append(arena, try pcmOf(arena, io, fixtures_dir, clip.name));
        max_prompt = @max(max_prompt, clip.prompt.len);
    }
    try testing.expect(clips.items.len > 0);

    const capacity = served_file.default_context;
    const window_rows = @min(@max(max_prompt, draft_file.forward_max_rows), 2048);
    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    var scratch_pool = try cuda.Scratch.create(gpa);
    defer scratch_pool.destroy();
    var m = try model_file.Model.load(arena, gpa, &c, &checkpoint, &diagnostic, &staging, scratch_pool, window_rows, window_rows);
    defer m.deinit();
    try staging.settle();
    var perception = try perception_file.Perception.load(gpa, arena, &c, &checkpoint, 8);
    defer perception.deinit();
    var state: state_file.State = try .init(gpa, &c, capacity);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&c, window_rows, draft_file.forward_max_rows, capacity);
    defer b.deinit();
    const tap_layers = try arena.alloc(u32, nlayers);
    for (tap_layers, 0..) |*t, i| t.* = @intCast(i);
    try b.setTaps(tap_layers);
    var tokenizer: ?text_tokenizer.Tokenizer = null;
    var tok_diag: text_tokenizer.Diagnostic = .{};
    if (text_tokenizer.Tokenizer.load(gpa, io, ck_dir, "tokenizer.json", &tok_diag)) |tok| tokenizer = tok else |_| {}
    defer if (tokenizer) |*tok| tok.deinit(gpa);
    try testing.expect(tokenizer != null);

    const taps_host = try arena.alloc(u16, b.taps.?.bytes() / 2);
    const logits_host = try gpa.alloc(u16, c.backbone.vocab);
    defer gpa.free(logits_host);
    const row_scratch = try arena.alloc(f32, hidden);
    var worst_layer_cosine: f64 = 1;
    var worst_enc_cosine: f64 = 1;
    var worst_wer: f64 = 0;
    var token_same: usize = 0;
    var token_total: usize = 0;

    for (clips.items, pcms.items) |clip, pcm| {
        // the encoder's rows, then the audio features the forward injects
        const audio_n = try perception.rows(arena, pcm.len);
        try testing.expectEqual(@as(usize, @intCast(clip.enc.shape[0])), audio_n);
        var enc_buf: cuda.Buffer = try .init(.bf16, &.{ audio_n, hidden });
        defer enc_buf.deinit();
        try perception.encode(arena, pcm, enc_buf.view);
        const enc_mine = try arena.alloc(u16, enc_buf.view.count());
        try cuda.synchronize();
        try cuda.download(enc_buf.view, std.mem.sliceAsBytes(enc_mine));
        const enc_theirs = try arena.alloc(f32, @intCast(clip.enc.count()));
        for (enc_theirs, std.mem.bytesAsSlice(u16, clip.enc.data)) |*v, bits| v.* = numeric.f32FromBf16(bits);
        const enc_stats = try reference.compareBf16(enc_mine, enc_theirs, audio_n, false, arena);
        worst_enc_cosine = @min(worst_enc_cosine, enc_stats.cosine);
        std.debug.print("{s}: encoder cosine {d:.6} rel rms {e:.3}\n", .{ clip.name, enc_stats.cosine, enc_stats.rel_all });

        try state.reset();
        const audio_rows_host = try arena.alloc(u32, clip.placeholder_rows.len);
        for (audio_rows_host, clip.placeholder_rows) |*r, p| r.* = @intCast(p);
        state.audio = .{ .features = enc_buf.view, .rows = audio_rows_host };
        const layer_acc = try arena.alloc(Acc, nlayers);
        @memset(layer_acc, .{});

        // the prompt in chunks (the last scored), then `take` teacher-forced serial steps
        const chunk_rows = @min(b.rows, m.max_rows);
        var done: usize = 0;
        while (done < clip.prompt.len) {
            const n = @min(chunk_rows, clip.prompt.len - done);
            const part = clip.prompt[done..][0..n];
            const score: forward_file.Score = if (done + n == clip.prompt.len) .last else .none;
            try forward_file.forwardScoring(m, &state, &b, part, score);
            try cuda.download(b.taps.?.rows(n), std.mem.sliceAsBytes(taps_host[0 .. n * nlayers * hidden]));
            if (score == .last) try forward_file.logitsRow(b, 0, logits_host);
            try forward_file.commit(m, &state, b, part, n);
            fold(layer_acc, clip.layers, clip.map, taps_host, n, done, hidden, row_scratch);
            done += n;
        }
        // the prefill's last row's pick opens the reply; compare the logits it scored
        var reply: std.ArrayList(i32) = .empty;
        defer reply.deinit(gpa);
        var differing: usize = 0;
        var compared: usize = 0;
        for (clip.map.scored_pos, clip.map.scored_f, clip.map.scored_r) |sp, f, r| {
            if (sp != clip.prompt.len - 1) continue;
            const theirs = try reference.logits(arena, &clip.file, f, @intCast(clip.num_rows[f]), r, c.backbone.vocab);
            const widened = try arena.alloc(f32, c.backbone.vocab);
            for (widened, logits_host) |*v, bits| v.* = numeric.f32FromBf16(bits);
            const s = reference.compare(widened, theirs, 1, true);
            differing += s.picks_differing;
            compared += 1;
        }
        try reply.append(gpa, try sampling.argmax(logits_host));
        const take = @min(8, clip.greedy.len);
        for (0..take) |s| {
            const pos = state.pos;
            if (pos >= clip.map.row_of.len) break;
            const tok: [1]i32 = .{clip.greedy[s]};
            try forward_file.forwardScoring(m, &state, &b, &tok, .all);
            try cuda.synchronize();
            try cuda.download(b.taps.?.rows(1), std.mem.sliceAsBytes(taps_host[0 .. nlayers * hidden]));
            try forward_file.logitsRow(b, 0, logits_host);
            fold(layer_acc, clip.layers, clip.map, taps_host, 1, pos, hidden, row_scratch);
            try forward_file.commit(m, &state, b, &tok, 1);
            if (s + 1 < clip.greedy.len) try reply.append(gpa, try sampling.argmax(logits_host));
            for (clip.map.scored_pos, clip.map.scored_f, clip.map.scored_r) |sp, f, r| {
                if (sp != pos) continue;
                const theirs = try reference.logits(arena, &clip.file, f, @intCast(clip.num_rows[f]), r, c.backbone.vocab);
                const widened = try arena.alloc(f32, c.backbone.vocab);
                for (widened, logits_host) |*v, bits| v.* = numeric.f32FromBf16(bits);
                const st = reference.compare(widened, theirs, 1, true);
                differing += st.picks_differing;
                compared += 1;
            }
        }
        for (layer_acc, 0..) |a, layer| {
            if (a.n == 0) continue;
            const cosine = if (a.mine > 0 and a.ref > 0) a.dot / @sqrt(a.mine * a.ref) else 1;
            const rel = if (a.ref > 0) @sqrt(a.diff / a.ref) else @sqrt(a.diff);
            if (cosine < worst_layer_cosine) worst_layer_cosine = cosine;
            std.debug.print("{s} L{d:0>2} rows {d} rel rms {e:.2} cos {d:.6}\n", .{ clip.name, layer, a.n, rel, cosine });
        }
        var same: usize = 0;
        for (reply.items, 0..) |t, i| {
            if (i < clip.greedy.len and t == clip.greedy[i]) same += 1;
        }
        token_same += same;
        token_total += reply.items.len;
        std.debug.print("{s}: top-1 {d}/{d} match, {d} logits compared, {d} picks differ\n", .{ clip.name, same, reply.items.len, compared, differing });
        // two of the nine clips (zh_140s, zh_600s) see one speculative row's argmax differ; hold the bound
        try testing.expect(differing <= 1);

        // the free greedy reply, its text held to the fixture's transcript
        try state.reset();
        state.audio = .{ .features = enc_buf.view, .rows = audio_rows_host };
        done = 0;
        var free_pick: i32 = -1;
        while (done < clip.prompt.len) {
            const n = @min(chunk_rows, clip.prompt.len - done);
            const part = clip.prompt[done..][0..n];
            const score: forward_file.Score = if (done + n == clip.prompt.len) .last else .none;
            try forward_file.forwardScoring(m, &state, &b, part, score);
            if (score == .last) {
                try forward_file.logitsRow(b, 0, logits_host);
                free_pick = try sampling.argmax(logits_host);
            }
            try forward_file.commit(m, &state, b, part, n);
            done += n;
        }
        var free: std.ArrayList(u32) = .empty;
        defer free.deinit(gpa);
        const room = @min(state.capacity - clip.prompt.len, clip.greedy.len + 64);
        while (free.items.len < room) {
            const token: u32 = @intCast(free_pick);
            try free.append(gpa, token);
            if (std.mem.indexOfScalar(u32, c.eos, token) != null) break;
            const next = [_]i32{free_pick};
            try forward_file.forwardScoring(m, &state, &b, &next, .all);
            try forward_file.logitsRow(b, 0, logits_host);
            try forward_file.commit(m, &state, b, &next, 1);
            free_pick = try sampling.argmax(logits_host);
        }
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(gpa);
        try tokenizer.?.decode(gpa, free.items, .{ .skip_special = true }, &text);
        const w = try reference.wer(arena, std.mem.trim(u8, clip.transcript, " \t\r\n"), std.mem.trim(u8, text.items, " \t\r\n"));
        worst_wer = @max(worst_wer, w.rate());
        std.debug.print("{s}: free greedy {d} tokens, {s} {d:.3} vs the reference transcript\n", .{ clip.name, free.items.len, if (w.codepoints) "CER" else "WER", w.rate() });
    }

    std.debug.print("floors: worst encoder cosine {d:.6}, worst layer cosine {d:.6}, worst WER {d:.3}, top-1 {d}/{d}\n", .{ worst_enc_cosine, worst_layer_cosine, worst_wer, token_same, token_total });
    try testing.expect(worst_enc_cosine >= enc_cosine_floor);
    try testing.expect(worst_layer_cosine >= layer_cosine_floor);
    try testing.expect(worst_wer <= wer_floor);
}

test "manifest.json's clips are the fixture set: a sibling safetensors ignored, a missing or broken one an error" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "manifest.json", .data = "{\"clips\":{\"zh_140s\":{},\"en_5s\":{}}}" });
    var model_dir = try tmp.dir.createDirPathOpen(io, "model", .{});
    defer model_dir.close(io);
    // a diagnostic beside the declared fixtures (spark-b's first-layer-host.safetensors, spec 163): not a clip
    try model_dir.writeFile(io, .{ .sub_path = "first-layer-host.safetensors", .data = "diagnostic" });
    try model_dir.writeFile(io, .{ .sub_path = "en_5s.safetensors", .data = "fixture" });
    try model_dir.writeFile(io, .{ .sub_path = "zh_140s.safetensors", .data = "fixture" });
    const files = try clipFiles(arena, io, tmp.dir, model_dir);
    try testing.expectEqualStrings("en_5s.safetensors,zh_140s.safetensors", try std.mem.join(arena, ",", files));

    // a declared clip whose file is absent fails the selection; a broken one fails at open
    try model_dir.deleteFile(io, "zh_140s.safetensors");
    try testing.expectError(error.FileNotFound, clipFiles(arena, io, tmp.dir, model_dir));
    try testing.expectError(error.InvalidSafetensors, safetensors.File.open(testing.allocator, io, model_dir, "en_5s.safetensors"));

    // a manifest that does not parse fails too
    try tmp.dir.writeFile(io, .{ .sub_path = "manifest.json", .data = "not json" });
    try testing.expectError(error.SyntaxError, clipFiles(arena, io, tmp.dir, model_dir));
}
