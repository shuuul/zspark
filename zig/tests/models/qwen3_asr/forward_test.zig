//! Qwen3-ASR's language model on its checkpoint (`$ZSPARK_QWEN3_ASR`) against the `qwen-asr` package's
//! transformers backend (`$ZSPARK_REFERENCE_QWEN3_ASR`, `tools/reference/qwen3_asr/export.py`, spec 138 WS-01), on
//! every fixture scenario (5 s, 30 s free, forced and with context, 140 s and 600 s, English and Chinese), the
//! audio embeddings the reference's (`enc.proj2`):
//!
//! - the prompt's embeddings, the audio rows laid over the pads, equal the reference's to the bit;
//! - every layer from the reference's own input (`isolated`, prompts of up to `rows_max` rows) and the prompt
//!   carried through all layers, each layer's rows at a cosine of `floor` or more (the last 32 rows of a longer
//!   prompt, as the reference keeps them), the final norm too, then the first logits and the greedy token;
//! - the first 32 decode steps carried, each fed the reference's token: every layer's row, the logits, the token;
//! - the whole greedy reply to its end token, against `tokens.greedy` token by token and the reference's
//!   `raw_text` (decoded here from `vocab.json`; the checkpoint ships no tokenizer.json); a reply may leave the
//!   reference's at a near tie alone (`tie_steps`), the long clips do, its character error rate reported;
//! - the whole pipeline, the fixture's mel through zspark's audio tower (`tower.zig`): the embeddings' worst row
//!   against `enc.proj2`, reported; with `ZSPARK_QWEN3_ASR_TOWER_REPLIES` set, on to the greedy reply against
//!   `tokens.greedy`, how many tokens agree, reported (the tower is near the reference's, not equal);
//! - the row rule: the prompt in one window, in windows of 100 and of 7 rows, its last rows serial, launched and as
//!   captured graphs (`forward.Graphs`); a verify window against serial steps; the same bits;
//! - a long clip's pieces (spec 142 WS-06, `pieces.zig`): cut, through the tower, decoded at once in fewer lanes
//!   than pieces, each piece's tokens those of its greedy reply alone;
//! - a W4A16_NVFP4 build (`$ZSPARK_QWEN3_ASR_NVFP4`, spec 140): its linears in the storage its config names, its
//!   logits near those of the same build with every NVFP4 weight held as bf16 of its value, and both row rules.
//!
//! Every test runs on each checkpoint whose variables are set (`Subject`): 1.7B's, and 0.6B's with its own fixtures
//! (`$ZSPARK_QWEN3_ASR_0_6B`, `$ZSPARK_REFERENCE_QWEN3_ASR_0_6B`, `$ZSPARK_QWEN3_ASR_0_6B_NVFP4`; spec 140 WS-09),
//! held to the same bounds.
//!
//! One load of each model (4 GB and 2 GB, seconds) serves every test here, so they stay in the area rather than the
//! whole-model tier. The reference attends with PyTorch's sdpa in bf16; zspark's attention keeps the softmax and
//! its sums in fp32 with bf16 probabilities against the values (common/attention.cu), and its norms take
//! `rsqrtf` where torch takes `rsqrt`: near, not bit-equal, so the layers are held by cosine.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const draft = @import("../../../src/models/qwen3_asr/draft.zig");
const forward = @import("../../../src/models/qwen3_asr/forward.zig");
const model_file = @import("../../../src/models/qwen3_asr/model.zig");
const config_file = @import("../../../src/models/qwen3_asr/config.zig");
const layout_file = @import("../../../src/models/qwen3_asr/layout.zig");
const modelopt = @import("../../../src/weights/modelopt.zig");
const requantize = @import("../../../src/weights/requantize.zig");
const tower_file = @import("../../../src/models/qwen3_asr/tower.zig");
const pieces_file = @import("../../../src/models/qwen3_asr/pieces.zig");
const audio = @import("../../../src/audio/audio.zig");

comptime {
    _ = draft.Decoder;
}

/// A layer's, the final norm's and the logits' cosine against the reference's, the worst row's, each layer from
/// the reference's own input. Position 0, the attention sink, is held apart: its last layer's output carries
/// massive activations (an input of 10,688 at 5 s) and lands at 0.9989 with every earlier layer at 0.99998.
const floor = 0.999;
const sink_floor = 0.998;
/// Carried through the layers, a row's bf16 differences compound: the worst prompt row reaches 0.956 at layer 21
/// (zh_30s, a row past 400) with every layer alone at 0.99997, the logits still at 0.9995 and every token the
/// reference's. Carried prompts and steps are held on their logits and tokens, their layers only against a gross
/// fault.
const carried_floor = 0.9;
const carried_logits = 0.99;
/// A greedy reply may leave the reference's at a near tie alone: its token's logit within this many bf16 steps of
/// the largest (the long clips' first divergences are 0 to 1 steps, the 0.999 of the logits).
const tie_steps = 2;
const rows_max = 1024;
const scored_max = 16;
/// The longest scenario's prompt and reply (en_600s: 7862 and 2131).
const capacity = 10240;

const scenarios = [_][]const u8{
    "en_5s.free",    "zh_5s.free",     "en_30s.free",  "en_30s.forced", "en_30s.context", "zh_30s.free",
    "zh_30s.forced", "zh_30s.context", "en_140s.free", "zh_140s.free",  "en_600s.free",   "zh_600s.free",
};

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// A checkpoint and its own fixtures: 1.7B's and 0.6B's (spec 140 WS-09), each with the variables that name its bf16
/// base, its fixtures (`export.py` run on that base) and its W4A16_NVFP4 build. A test runs on every subject whose
/// base and fixtures are set, and skips when none is; the models are loaded once and never let go (the process
/// ends after the tests).
pub const Subject = struct {
    name: []const u8,
    checkpoint: [*:0]const u8,
    reference: [*:0]const u8,
    nvfp4: [*:0]const u8,
    /// the NVFP4 build's carried logits against its dequantized weights'
    nvfp4_logits: f64,
    loaded: ?model_file.Model = null,
    nvfp4_loaded: ?[2]model_file.Model = null,

    pub fn path(self: Subject) []const u8 {
        return env(self.checkpoint).?;
    }

    pub fn fixtures(self: Subject) []const u8 {
        return env(self.reference).?;
    }

    fn model(self: *Subject) !?model_file.Model {
        if (self.loaded) |m| return m;
        const path_ = env(self.checkpoint) orelse return null;
        if (env(self.reference) == null) return null;
        const start = cuda.monotonic();
        self.loaded = try model_file.open(std.heap.page_allocator, testing.io, path_, rows_max, scored_max);
        try cuda.synchronize();
        std.debug.print("qwen3_asr {s}: loaded in {d:.0} ms, device {d} MB\n", .{ self.name, @as(f64, @floatFromInt(cuda.monotonic() - start)) / 1e6, cuda.account.bytes >> 20 });
        return self.loaded;
    }

    /// Its audio tower, of the shape its config.json gives.
    pub fn tower(self: Subject, gpa: std.mem.Allocator, arena: std.mem.Allocator, checkpoint: *const safetensors.Checkpoint, max_chunks: usize) !tower_file.Tower {
        var diagnostic: tower_file.Diagnostic = .{};
        const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, try std.fmt.allocPrint(arena, "{s}/config.json", .{self.path()}), arena, .limited(1 << 20));
        const shape = try tower_file.checkConfig(arena, text, &diagnostic);
        return tower_file.Tower.load(gpa, arena, checkpoint, shape, max_chunks, tower_file.conv_chunks_default, &diagnostic) catch |err| {
            std.debug.print("audio tower: {s}\n", .{diagnostic.message});
            return err;
        };
    }
};

var subjects = [_]Subject{
    .{ .name = "1.7B", .checkpoint = "ZSPARK_QWEN3_ASR", .reference = "ZSPARK_REFERENCE_QWEN3_ASR", .nvfp4 = "ZSPARK_QWEN3_ASR_NVFP4", .nvfp4_logits = 0.995 },
    .{ .name = "0.6B", .checkpoint = "ZSPARK_QWEN3_ASR_0_6B", .reference = "ZSPARK_REFERENCE_QWEN3_ASR_0_6B", .nvfp4 = "ZSPARK_QWEN3_ASR_0_6B_NVFP4", .nvfp4_logits = 0.99 },
};

/// `body` on every subject that is set; skipped when none is.
pub fn eachSubject(comptime body: fn (*Subject, model_file.Model) anyerror!void) !void {
    var ran = false;
    for (&subjects) |*s| {
        const m = try s.model() orelse continue;
        ran = true;
        std.debug.print("qwen3_asr: the {s} checkpoint\n", .{s.name});
        errdefer std.debug.print("qwen3_asr {s}: failed\n", .{s.name});
        try body(s, m);
    }
    if (!ran) return error.SkipZigTest;
}

const Case = struct {
    file: safetensors.File,
    ids: []const i32,
    tokens: []const i32,
    rows: []u32,
    features: cuda.Buffer,
    gpa: std.mem.Allocator,

    fn open(gpa: std.mem.Allocator, subject: *const Subject, name: []const u8) !Case {
        const dir = subject.fixtures();
        var buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/{s}.safetensors", .{ dir, name });
        var file: safetensors.File = try .open(gpa, testing.io, .cwd(), path);
        errdefer file.close();
        const enc = file.get("enc.proj2").?;
        const count: usize = @intCast(enc.shape[0]);
        const first: usize = @intCast(@as([]const i32, @alignCast(std.mem.bytesAsSlice(i32, file.get("prompt.audio_pad_first").?.data)))[0]);
        const rows = try gpa.alloc(u32, count);
        errdefer gpa.free(rows);
        for (rows, 0..) |*row, i| row.* = @intCast(first + i);
        var features: cuda.Buffer = try .init(.bf16, &.{ count, @intCast(enc.shape[1]) });
        errdefer features.deinit();
        try features.upload(enc.data);
        return .{
            .file = file,
            .ids = @alignCast(std.mem.bytesAsSlice(i32, file.get("prompt.ids").?.data)),
            .tokens = @alignCast(std.mem.bytesAsSlice(i32, file.get("tokens.greedy").?.data)),
            .rows = rows,
            .features = features,
            .gpa = gpa,
        };
    }

    fn close(self: *Case) void {
        self.features.deinit();
        self.gpa.free(self.rows);
        self.file.close();
    }

    fn bf16s(self: Case, name: []const u8) []const u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, self.file.get(name).?.data));
    }

    fn f32s(self: Case, name: []const u8) []const f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, self.file.get(name).?.data));
    }

    fn start(self: Case, state: *forward.State) !void {
        try state.reset();
        state.audio = .{ .features = self.features.view, .rows = self.rows };
    }
};

/// The worst row's cosine and the largest absolute difference of bf16 rows against the reference's, bf16 or f32.
const Match = struct {
    worst: f64 = 1,
    max_abs: f64 = 0,
    /// the worst row
    at: usize = 0,

    fn of(mine: []const u16, comptime T: type, want: []const T, width: usize) Match {
        var out: Match = .{};
        for (0..mine.len / width) |row| {
            var dot: f64 = 0;
            var aa: f64 = 0;
            var bb: f64 = 0;
            for (mine[row * width ..][0..width], want[row * width ..][0..width]) |a_bits, b_raw| {
                const a: f64 = numeric.f32FromBf16(a_bits);
                const b: f64 = if (T == u16) numeric.f32FromBf16(b_raw) else b_raw;
                dot += a * b;
                aa += a * a;
                bb += b * b;
                out.max_abs = @max(out.max_abs, @abs(a - b));
            }
            const cos = dot / @sqrt(aa * bb + 1e-300);
            if (cos < out.worst) {
                out.worst = cos;
                out.at = row;
            }
        }
        return out;
    }

    fn merge(self: *Match, other: Match) void {
        if (other.worst < self.worst) self.at = other.at;
        self.worst = @min(self.worst, other.worst);
        self.max_abs = @max(self.max_abs, other.max_abs);
    }
};

/// What a scenario's comparison found: a Match a layer (0 .. 27 the layers' outputs, 28 the final norm), the
/// logits', and the greedy tokens missed.
const Stage = struct {
    layers: [29]Match = @splat(.{}),
    logits: Match = .{},
    misses: usize = 0,
    rows: usize = 0,
    /// position 0's last layer, apart (`sink_floor`), where the layers' Matches leave it out
    sink: ?Match = null,

    fn print(self: Stage, case: []const u8, how: []const u8) void {
        var worst: Match = .{};
        var at: usize = 0;
        for (self.layers, 0..) |l, i| if (l.worst < worst.worst) {
            worst = l;
            at = i;
        };
        std.debug.print("qwen3_asr {s} {s} ({d} rows): worst layer {d} {d:.5} (max abs {d:.3}), final norm {d:.5}, logits {d:.5} (max abs {d:.3}), argmax misses {d}\n", .{
            case, how, self.rows, at, worst.worst, worst.max_abs, self.layers[28].worst, self.logits.worst, self.logits.max_abs, self.misses,
        });
        std.debug.print("  layers:", .{});
        for (self.layers) |l| std.debug.print(" {d:.5}@{d}", .{ l.worst, l.at });
        if (self.sink) |s| std.debug.print("\n  position 0's final norm (apart): {d:.5} (max abs {d:.3})", .{ s.worst, s.max_abs });
        std.debug.print("\n", .{});
    }

    /// Whether every layer, the logits and the tokens are held: isolated layers at `floor`, carried ones at
    /// `carried_floor` with their logits at `carried_logits`.
    fn held(self: Stage, isolated: bool) bool {
        const layer_floor: f64 = if (isolated) floor else carried_floor;
        for (self.layers) |l| if (l.worst < layer_floor) return false;
        if (self.sink) |s| if (s.worst < sink_floor) return false;
        const logits_floor: f64 = if (isolated) floor else carried_logits;
        return self.logits.worst >= logits_floor and self.misses == 0;
    }
};

fn rowsOf(gpa: std.mem.Allocator, view: cuda.View) ![]u16 {
    const out = try gpa.alloc(u16, view.count());
    errdefer gpa.free(out);
    try cuda.download(view, std.mem.sliceAsBytes(out));
    return out;
}

fn argmax(bits: []const u16) usize {
    return @intCast(sampling.argmax(bits) catch unreachable);
}

/// The prompt through the layers one at a time in its last window, the earlier windows run whole: each layer's
/// rows from `from` on (the reference's kept rows) against `lm.prefill.hidden.NN`, then the logits of the last row.
fn carried(gpa: std.mem.Allocator, m: model_file.Model, state: *forward.State, b: *forward.Buffers, case: Case, from: usize, stage: *Stage) !void {
    const n = case.ids.len;
    const c = m.config;
    // the reference's rows in the last window: all of a prompt it keeps whole
    const last = if (from == 0) n else @min(n, 512);
    while (state.pos < n - last) {
        const rows = @min(512, n - last - state.pos);
        const run = case.ids[state.pos..][0..rows];
        try forward.forwardScoring(m, state, b, run, .none);
        try forward.commit(m, state, b.*, run, rows);
    }
    const base = state.pos;
    std.debug.assert(base <= from);
    const run = case.ids[base..];
    try forward.embed(m, state.*, b, run);
    var name: [32]u8 = undefined;
    for (0..c.layers) |i| {
        try forward.layer(m, i, state.*, b, run.len, n);
        if (i + 1 == c.layers) break;
        const mine = try rowsOf(gpa, b.h.span(from - base, n - from));
        defer gpa.free(mine);
        const want = case.bf16s(try std.fmt.bufPrint(&name, "lm.prefill.hidden.{d:0>2}", .{i + 1}));
        stage.layers[i].merge(.of(mine, u16, want[(from - (n - want.len / c.hidden)) * c.hidden ..], c.hidden));
    }
    try forward.finish(m, b, run.len, .last);
    const final = try rowsOf(gpa, b.final.span(from - base, n - from));
    defer gpa.free(final);
    const want = case.bf16s("lm.prefill.hidden.28");
    stage.layers[c.layers - 1].merge(.of(final, u16, want[(from - (n - want.len / c.hidden)) * c.hidden ..], c.hidden));
    stage.layers[28].merge(.of(final, u16, want[(from - (n - want.len / c.hidden)) * c.hidden ..], c.hidden));
    try forward.commit(m, state, b.*, run, run.len);
    stage.rows += n - from;
}

test "the prompt's embeddings, every layer of it and the first 32 steps against the reference's" {
    try eachSubject(againstReference);
}

fn againstReference(s: *Subject, m: model_file.Model) !void {
    const gpa = testing.allocator;
    const c = m.config;
    var state: forward.State = try .init(gpa, c, capacity);
    defer state.deinit();
    var b: forward.Buffers = try .init(c, rows_max, scored_max, capacity);
    defer b.deinit();
    const logits = try gpa.alloc(u16, c.vocab);
    defer gpa.free(logits);
    var name: [32]u8 = undefined;
    var failed: usize = 0;
    for (scenarios) |scenario| {
        var case: Case = try .open(gpa, s, scenario);
        defer case.close();
        const n = case.ids.len;
        const from: usize = if (n > rows_max) n - 32 else 0;

        // the embeddings, the audio rows laid over the pads: the reference's bits
        try case.start(&state);
        if (n <= rows_max) {
            try forward.embed(m, state, &b, case.ids);
            const mine = try rowsOf(gpa, b.h.rows(n));
            defer gpa.free(mine);
            try testing.expectEqualSlices(u16, case.bf16s("lm.prefill.hidden.00"), mine);

            // each layer from the reference's input
            var isolated: Stage = .{ .rows = n };
            for (0..c.layers) |i| {
                try b.h.uploadAt(0, std.mem.sliceAsBytes(case.bf16s(try std.fmt.bufPrint(&name, "lm.prefill.hidden.{d:0>2}", .{i}))));
                try forward.layer(m, i, state, &b, n, n);
                if (i + 1 < c.layers) {
                    const out = try rowsOf(gpa, b.h.rows(n));
                    defer gpa.free(out);
                    isolated.layers[i] = .of(out, u16, case.bf16s(try std.fmt.bufPrint(&name, "lm.prefill.hidden.{d:0>2}", .{i + 1})), c.hidden);
                } else {
                    // the final norm on the last layer's output: position 0 apart
                    try forward.finish(m, &b, n, .last);
                    const out = try rowsOf(gpa, b.final.rows(n));
                    defer gpa.free(out);
                    const want = case.bf16s("lm.prefill.hidden.28");
                    isolated.layers[i] = .of(out[c.hidden..], u16, want[c.hidden..], c.hidden);
                    isolated.layers[i].at += 1;
                    isolated.layers[28] = isolated.layers[i];
                    isolated.sink = .of(out[0..c.hidden], u16, want[0..c.hidden], c.hidden);
                    try forward.logitsRow(b, 0, logits);
                    isolated.logits = .of(logits, f32, case.f32s("logits.first33")[0..c.vocab], c.vocab);
                    if (argmax(logits) != case.tokens[0]) isolated.misses += 1;
                }
            }
            isolated.print(scenario, "prompt isolated");
            if (!isolated.held(true)) failed += 1;
        }

        // carried through all layers, then the steps fed the reference's tokens
        try case.start(&state);
        var prompt: Stage = .{};
        try carried(gpa, m, &state, &b, case, from, &prompt);
        try forward.logitsRow(b, 0, logits);
        prompt.logits = .of(logits, f32, case.f32s("logits.first33")[0..c.vocab], c.vocab);
        if (argmax(logits) != case.tokens[0]) prompt.misses += 1;
        prompt.print(scenario, "prompt carried");
        if (!prompt.held(false)) failed += 1;

        const hidden = case.bf16s("lm.decode.hidden");
        const steps = hidden.len / (29 * c.hidden);
        const first33 = case.f32s("logits.first33");
        var decode: Stage = .{ .rows = steps };
        for (0..steps) |k| {
            const step = [_]i32{case.tokens[k]};
            try forward.embed(m, state, &b, &step);
            const want = hidden[k * 29 * c.hidden ..][0 .. 29 * c.hidden];
            for (0..c.layers) |i| {
                try forward.layer(m, i, state, &b, 1, state.pos + 1);
                if (i + 1 == c.layers) break;
                const row = try rowsOf(gpa, b.h.rows(1));
                defer gpa.free(row);
                decode.layers[i].merge(.of(row, u16, want[(i + 1) * c.hidden ..][0..c.hidden], c.hidden));
            }
            try forward.finish(m, &b, 1, .last);
            const final = try rowsOf(gpa, b.final.rows(1));
            defer gpa.free(final);
            decode.layers[27].merge(.of(final, u16, want[28 * c.hidden ..][0..c.hidden], c.hidden));
            decode.layers[28].merge(.of(final, u16, want[28 * c.hidden ..][0..c.hidden], c.hidden));
            try forward.logitsRow(b, 0, logits);
            if ((k + 2) * c.vocab <= first33.len) decode.logits.merge(.of(logits, f32, first33[(k + 1) * c.vocab ..][0..c.vocab], c.vocab));
            if (argmax(logits) != case.tokens[k + 1]) decode.misses += 1;
            try forward.commit(m, &state, b, &step, 1);
        }
        decode.print(scenario, "steps");
        if (!decode.held(false)) failed += 1;
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

/// Token ids to text as the reference's `skip_special_tokens=True` decode: vocab.json's byte-level strings, the
/// added tokens of tokenizer_config.json (special ones left out).
const Vocabulary = struct {
    arena: std.heap.ArenaAllocator,
    /// a token's bytes, by id
    bytes: [][]const u8,
    special: []bool,

    fn load(gpa: std.mem.Allocator, dir: []const u8, size: usize) !Vocabulary {
        var self: Vocabulary = .{ .arena = .init(gpa), .bytes = undefined, .special = undefined };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.bytes = try a.alloc([]const u8, size);
        @memset(self.bytes, "");
        self.special = try a.alloc(bool, size);
        @memset(self.special, false);
        // GPT-2's bytes_to_unicode undone: a printable byte is its own character, the others U+0100 on in order
        var byte_of: [512]u8 = undefined;
        var next: u16 = 256;
        for (0..256) |i| {
            const byte: u8 = @intCast(i);
            const printable = (byte >= 33 and byte <= 126) or (byte >= 161 and byte <= 172) or byte >= 174;
            if (printable) byte_of[byte] = byte else {
                byte_of[next] = byte;
                next += 1;
            }
        }
        var buf: [512]u8 = undefined;
        const vocab_text = try std.Io.Dir.cwd().readFileAlloc(testing.io, try std.fmt.bufPrint(&buf, "{s}/vocab.json", .{dir}), a, .limited(1 << 26));
        const vocab = try std.json.parseFromSliceLeaky(std.json.Value, a, vocab_text, .{});
        var it = vocab.object.iterator();
        while (it.next()) |entry| {
            const id: usize = @intCast(entry.value_ptr.integer);
            var out: std.ArrayList(u8) = .empty;
            var view = try std.unicode.Utf8View.init(entry.key_ptr.*);
            var points = view.iterator();
            while (points.nextCodepoint()) |point| try out.append(a, byte_of[point]);
            self.bytes[id] = out.items;
        }
        const config_text = try std.Io.Dir.cwd().readFileAlloc(testing.io, try std.fmt.bufPrint(&buf, "{s}/tokenizer_config.json", .{dir}), a, .limited(1 << 26));
        const config = try std.json.parseFromSliceLeaky(std.json.Value, a, config_text, .{});
        var added = config.object.get("added_tokens_decoder").?.object.iterator();
        while (added.next()) |entry| {
            const id = try std.fmt.parseInt(usize, entry.key_ptr.*, 10);
            self.bytes[id] = entry.value_ptr.object.get("content").?.string;
            self.special[id] = entry.value_ptr.object.get("special").?.bool;
        }
        return self;
    }

    fn deinit(self: *Vocabulary) void {
        self.arena.deinit();
    }

    fn decode(self: Vocabulary, gpa: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (!self.special[id]) try out.appendSlice(gpa, self.bytes[id]);
        return out.toOwnedSlice(gpa);
    }
};

fn rawText(gpa: std.mem.Allocator, s: *const Subject, scenario: []const u8) ![]u8 {
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/{s}.json", .{ s.fixtures(), scenario });
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    return gpa.dupe(u8, parsed.value.object.get("raw_text").?.string);
}

/// The character error rate of `text` against `want`: the edit distance over code points, over `want`'s count.
fn charErrors(gpa: std.mem.Allocator, text: []const u8, want: []const u8) !f64 {
    const a = try codePoints(gpa, text);
    defer gpa.free(a);
    const w = try codePoints(gpa, want);
    defer gpa.free(w);
    const row = try gpa.alloc(usize, w.len + 1);
    defer gpa.free(row);
    for (row, 0..) |*r, j| r.* = j;
    for (a, 1..) |ca, i| {
        var diagonal = row[0];
        row[0] = i;
        for (w, 1..) |cw, j| {
            const above = row[j];
            row[j] = @min(@min(above + 1, row[j - 1] + 1), diagonal + @intFromBool(ca != cw));
            diagonal = above;
        }
    }
    return @as(f64, @floatFromInt(row[w.len])) / @as(f64, @floatFromInt(@max(w.len, 1)));
}

fn codePoints(gpa: std.mem.Allocator, text: []const u8) ![]u21 {
    var out: std.ArrayList(u21) = .empty;
    errdefer out.deinit(gpa);
    var it = (try std.unicode.Utf8View.init(text)).iterator();
    while (it.nextCodepoint()) |cp| try out.append(gpa, cp);
    return out.toOwnedSlice(gpa);
}

/// At a reply's first divergence from the reference's: the prompt and the reply before it run again (in windows:
/// the row rule gives the steps' bits), then the largest logit, the reference token's and their gap, to tell a
/// near tie from a fault: the gap in bf16 steps of the largest logit.
fn divergence(gpa: std.mem.Allocator, m: model_file.Model, state: *forward.State, b: *forward.Buffers, case: Case, reply: []const u32, at: usize, logits: []u16) !f64 {
    const tokens = try gpa.alloc(i32, case.ids.len + at);
    defer gpa.free(tokens);
    @memcpy(tokens[0..case.ids.len], case.ids);
    for (tokens[case.ids.len..], reply[0..at]) |*t, r| t.* = @intCast(r);
    try case.start(state);
    while (state.pos < tokens.len) {
        const rows = @min(b.rows, tokens.len - state.pos);
        const run = tokens[state.pos..][0..rows];
        try forward.forwardScoring(m, state, b, run, if (state.pos + rows == tokens.len) .last else .none);
        try forward.commit(m, state, b.*, run, rows);
    }
    try forward.logitsRow(b.*, 0, logits);
    const mine = argmax(logits);
    const want: usize = @intCast(case.tokens[at]);
    const top = numeric.f32FromBf16(logits[mine]);
    const theirs = numeric.f32FromBf16(logits[want]);
    const step = std.math.ldexp(@as(f32, 1), std.math.ilogb(@abs(top)) - 7);
    const steps = (top - theirs) / step;
    std.debug.print("  at {d} (position {d}): mine {d} logit {d:.3}, the reference's {d} logit {d:.3}, gap {d:.3} ({d:.0} bf16 steps)\n", .{ at, tokens.len, mine, top, want, theirs, top - theirs, steps });
    return steps;
}

test "the greedy reply to its end token is the reference's, token by token and as text" {
    try eachSubject(greedyReplies);
}

fn greedyReplies(s: *Subject, m: model_file.Model) !void {
    const gpa = testing.allocator;
    const c = m.config;
    var vocabulary: Vocabulary = try .load(gpa, s.path(), c.vocab);
    defer vocabulary.deinit();
    var state: forward.State = try .init(gpa, c, capacity);
    defer state.deinit();
    var b: forward.Buffers = try .init(c, 512, 1, capacity);
    defer b.deinit();
    const logits = try gpa.alloc(u16, c.vocab);
    defer gpa.free(logits);
    var failed: usize = 0;
    for (scenarios) |scenario| {
        var case: Case = try .open(gpa, s, scenario);
        defer case.close();
        try case.start(&state);
        var reply: std.ArrayList(u32) = .empty;
        defer reply.deinit(gpa);
        var timing: draft.Timing = .{};
        const start = cuda.monotonic();
        try draft.greedy(gpa, m, &state, &b, case.ids, 4096, logits, &reply, &timing);
        const end = cuda.monotonic();
        var first: ?usize = null;
        for (case.tokens, 0..) |want, i| {
            if (i >= reply.items.len or reply.items[i] != want) {
                first = i;
                break;
            }
        }
        if (first == null and reply.items.len != case.tokens.len) first = case.tokens.len;
        const text = try vocabulary.decode(gpa, reply.items);
        defer gpa.free(text);
        const want = try rawText(gpa, s, scenario);
        defer gpa.free(want);
        const steps = reply.items.len -| 1;
        std.debug.print("qwen3_asr {s}: {d} prompt rows {d:.1} ms, {d} tokens (reference {d}), {d:.1} tokens a second, first divergence {?d}, text {s}\n", .{
            scenario,                                                                                                   case.ids.len, @as(f64, @floatFromInt(timing.first - start)) / 1e6,     reply.items.len, case.tokens.len,
            if (steps > 0) @as(f64, @floatFromInt(steps)) / (@as(f64, @floatFromInt(end - timing.first)) / 1e9) else 0, first,        if (std.mem.eql(u8, text, want)) "equal" else "differs",
        });
        if (!std.mem.eql(u8, text, want)) std.debug.print("  character error rate against the reference's text {d:.4}%\n", .{100 * try charErrors(gpa, text, want)});
        // a divergence passes as a near tie: the reference's token within `tie_steps` of the largest logit
        const tied = if (first) |at| at < reply.items.len and at < case.tokens.len and
            try divergence(gpa, m, &state, &b, case, reply.items, at, logits) <= tie_steps else true;
        if (!tied or (first == null and !std.mem.eql(u8, text, want))) failed += 1;
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

// The coordinator's question (spec 138): on the 600 s clips the tower has rows far from the reference's (cosine as
// low as 0.23 on 14 to 20 of 7.8k); whether they change a transcript. Reported, not held: the tower's own bound is
// tower_test.zig's, the transcripts' the WER set's (`serve_wer.py`). The replies are 5,048 greedy tokens, 100 s of
// the 104 s this test took (2026-10-26), and answered the question as the same figures each run: they decode
// only with `ZSPARK_QWEN3_ASR_TOWER_REPLIES` set (spec 133 WS-05); unset, every clip's embeddings are reported.
test "the fixture's mel through zspark's audio tower to the greedy reply, its agreement with the reference's" {
    try eachSubject(throughTower);
}

fn throughTower(s: *Subject, m: model_file.Model) !void {
    const replies = env("ZSPARK_QWEN3_ASR_TOWER_REPLIES") != null;
    const gpa = testing.allocator;
    const c = m.config;
    const path = s.path();
    var checkpoint: safetensors.Checkpoint = try .open(gpa, testing.io, .cwd(), path);
    defer checkpoint.close();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var tower = try s.tower(gpa, arena.allocator(), &checkpoint, tower_file.max_chunks_default);
    defer tower.deinit();
    var vocabulary: Vocabulary = try .load(gpa, path, c.vocab);
    defer vocabulary.deinit();
    var state: forward.State = try .init(gpa, c, capacity);
    defer state.deinit();
    var b: forward.Buffers = try .init(c, 512, 1, capacity);
    defer b.deinit();
    const logits = try gpa.alloc(u16, c.vocab);
    defer gpa.free(logits);
    for (scenarios) |scenario| {
        var case: Case = try .open(gpa, s, scenario);
        defer case.close();
        const mel = case.file.get("mel.bf16").?;
        const frames: usize = @intCast(mel.shape[1]);
        const n = tower_file.tokens(frames);
        try testing.expectEqual(case.rows.len, n);
        const features = try gpa.alloc(u16, mel.data.len / 2);
        defer gpa.free(features);
        @memcpy(std.mem.sliceAsBytes(features), mel.data);
        try tower.encode(gpa, features, frames, case.features.rows(n));
        const embeddings = try rowsOf(gpa, case.features.rows(n));
        defer gpa.free(embeddings);
        const against: Match = .of(embeddings, u16, case.bf16s("enc.proj2"), c.hidden);
        if (!replies) {
            std.debug.print("qwen3_asr {s} through the tower: embeddings' worst row {d:.4}@{d}\n", .{ scenario, against.worst, against.at });
            continue;
        }

        try case.start(&state);
        var reply: std.ArrayList(u32) = .empty;
        defer reply.deinit(gpa);
        var timing: draft.Timing = .{};
        try draft.greedy(gpa, m, &state, &b, case.ids, 4096, logits, &reply, &timing);
        var agree: usize = 0;
        while (agree < @min(reply.items.len, case.tokens.len) and reply.items[agree] == case.tokens[agree]) agree += 1;
        var equal: usize = 0;
        for (reply.items[0..@min(reply.items.len, case.tokens.len)], case.tokens[0..@min(reply.items.len, case.tokens.len)]) |a, w| equal += @intFromBool(a == w);
        const text = try vocabulary.decode(gpa, reply.items);
        defer gpa.free(text);
        const want = try rawText(gpa, s, scenario);
        defer gpa.free(want);
        std.debug.print("qwen3_asr {s} through the tower: embeddings' worst row {d:.4}@{d}; {d} tokens (reference {d}), the first {d} the reference's, {d} equal by position, text {s}\n", .{
            scenario, against.worst, against.at, reply.items.len, case.tokens.len, agree, equal, if (std.mem.eql(u8, text, want)) "equal" else "differs",
        });
        if (!std.mem.eql(u8, text, want)) std.debug.print("  character error rate against the reference's text {d:.4}%\n", .{100 * try charErrors(gpa, text, want)});
        if (agree < @min(reply.items.len, case.tokens.len)) _ = try divergence(gpa, m, &state, &b, case, reply.items, agree, logits);
    }
}

test "row rule: a prompt in one window, in windows of 100 and of 7, its last rows serial (also as graphs); a verify window and its steps" {
    try eachSubject(rowRule);
}

/// The first row rule test's body, on `m`: the bf16 base's, or a W4A16_NVFP4 build's (`Subject.nvfp4Models`).
fn rowRule(s: *Subject, m: model_file.Model) !void {
    const gpa = testing.allocator;
    const c = m.config;
    var state: forward.State = try .init(gpa, c, capacity);
    defer state.deinit();
    var b: forward.Buffers = try .init(c, rows_max, scored_max, capacity);
    defer b.deinit();
    var case: Case = try .open(gpa, s, "en_30s.free");
    defer case.close();
    const n = case.ids.len;
    const reply = case.tokens[0..8];

    // the final rows of the whole prompt, then the logits of a window of the reply's first 8 tokens after it
    const Run = struct { final: []u16, verify: []u16 };
    const runs = [_]struct { width: usize, serial: usize, graphed: bool = false }{ .{ .width = n, .serial = 0 }, .{ .width = 100, .serial = 0 }, .{ .width = 7, .serial = 0 }, .{ .width = 512, .serial = 12 }, .{ .width = 512, .serial = 12, .graphed = true } };
    var graphs: forward.Graphs = try .init(gpa);
    defer graphs.deinit();
    var got: [runs.len]Run = undefined;
    for (runs, &got) |shape, *out| {
        try case.start(&state);
        out.final = try gpa.alloc(u16, n * c.hidden);
        while (state.pos < n) {
            const left = n - state.pos;
            const rows = if (left <= shape.serial) 1 else @min(shape.width, left - shape.serial);
            const run = case.ids[state.pos..][0..rows];
            if (shape.graphed and rows == 1) try graphs.forward(m, &state, &b, run) else try forward.forwardScoring(m, &state, &b, run, .none);
            try cuda.download(b.final.rows(rows), std.mem.sliceAsBytes(out.final[state.pos * c.hidden ..][0 .. rows * c.hidden]));
            try forward.commit(m, &state, b, run, rows);
        }
        // a verify window of the reply's tokens, every row scored; then (the 4th run) the same as serial steps
        out.verify = try gpa.alloc(u16, reply.len * c.vocab);
        if (shape.serial == 0) {
            try forward.forward(m, &state, &b, reply);
            try cuda.download(b.logits.rows(reply.len), std.mem.sliceAsBytes(out.verify));
        } else for (reply, 0..) |token, i| {
            const step = [_]i32{token};
            if (shape.graphed) try graphs.forward(m, &state, &b, &step) else try forward.forwardScoring(m, &state, &b, &step, .last);
            try forward.logitsRow(b, 0, out.verify[i * c.vocab ..][0..c.vocab]);
            try forward.commit(m, &state, b, &step, 1);
        }
    }
    defer for (got) |run| {
        gpa.free(run.final);
        gpa.free(run.verify);
    };
    for (got[1..]) |run| {
        try testing.expectEqualSlices(u16, got[0].final, run.final);
        try testing.expectEqualSlices(u16, got[0].verify, run.verify);
    }
}

test "row rule, a long prompt: its rows' bits in windows of 1,000, of 100 and of 7 (groups of rows against a row's chunks)" {
    try eachSubject(rowRuleLong);
}

/// The second row rule test's body, on `m`.
fn rowRuleLong(s: *Subject, m: model_file.Model) !void {
    const gpa = testing.allocator;
    const c = m.config;
    var state: forward.State = try .init(gpa, c, capacity);
    defer state.deinit();
    var b: forward.Buffers = try .init(c, rows_max, scored_max, capacity);
    defer b.deinit();
    var case: Case = try .open(gpa, s, "en_140s.free");
    defer case.close();
    const n = case.ids.len;
    // 1,869 rows: four chunks of keys at the last, windows that end inside groups of rows
    const widths = [_]usize{ 1000, 100, 7 };
    var got: [widths.len][]u16 = undefined;
    for (widths, &got) |width, *out| {
        try case.start(&state);
        out.* = try gpa.alloc(u16, n * c.hidden);
        while (state.pos < n) {
            const rows = @min(width, n - state.pos);
            const run = case.ids[state.pos..][0..rows];
            try forward.forwardScoring(m, &state, &b, run, .none);
            try cuda.download(b.final.rows(rows), std.mem.sliceAsBytes(out.*[state.pos * c.hidden ..][0 .. rows * c.hidden]));
            try forward.commit(m, &state, b, run, rows);
        }
    }
    defer for (got) |out| gpa.free(out);
    for (got[1..]) |out| try testing.expectEqualSlices(u16, got[0], out);
}

test "pieces: a long clip cut as vLLM cuts it, its pieces decoded at once in fewer lanes, each piece's tokens its own greedy reply's" {
    try eachSubject(piecesAtOnce);
}

fn piecesAtOnce(s: *Subject, m: model_file.Model) !void {
    const gpa = testing.allocator;
    const c = m.config;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var checkpoint: safetensors.Checkpoint = try .open(gpa, testing.io, .cwd(), s.path());
    defer checkpoint.close();
    var tower = try s.tower(gpa, a, &checkpoint, 64);
    defer tower.deinit();
    const tables = try a.create(audio.mel.Tables);
    tables.init();

    // en_x01 of the WER set, 145 s: five pieces, in three lanes; the prompt en_30s.free's around its audio rows
    const clip_path = try std.fmt.allocPrint(a, "{s}/wer/clips/en_x01.flac", .{s.fixtures()});
    const clip = try audio.clip(a, try std.Io.Dir.cwd().readFileAlloc(testing.io, clip_path, a, .limited(64 << 20)), 1200);
    const ends = try audio.split.cuts(a, clip.samples, audio.rate, 30, 1, 1600);
    try testing.expectEqual(@as(usize, 5), ends.len);
    var template: Case = try .open(gpa, s, "en_30s.free");
    defer template.close();
    const first_pad = template.rows[0];
    const after = template.ids[first_pad + template.rows.len ..];
    var features: cuda.Buffer = try .init(.bf16, &.{ ends.len * 30 * tower_file.chunk_tokens, c.hidden });
    defer features.deinit();
    const list = try a.alloc(pieces_file.Piece, ends.len);
    var begin: usize = 0;
    var offset: usize = 0;
    for (list, ends) |*piece, end| {
        const part = clip.samples[begin..end];
        begin = end;
        const frames = audio.mel.frames(part.len);
        const n = tower_file.tokens(frames);
        const mel = try audio.mel.features(a, tables, part);
        const bits = try a.alloc(u16, mel.len);
        for (bits, mel) |*bit, x| bit.* = numeric.bf16FromF32(x);
        try tower.encode(a, bits, frames, features.span(offset, n));
        const prompt = try a.alloc(i32, first_pad + n + after.len);
        const rows = try a.alloc(u32, n);
        @memcpy(prompt[0..first_pad], template.ids[0..first_pad]);
        for (rows, 0..) |*row, i| {
            row.* = @intCast(first_pad + i);
            prompt[first_pad + i] = template.ids[first_pad];
        }
        @memcpy(prompt[first_pad + n ..], after);
        piece.* = .{ .prompt = prompt, .audio = .{ .features = features.span(offset, n), .rows = rows }, .limit = 400 };
        offset += n;
    }

    var lanes: pieces_file.Lanes = try .init(gpa, c, 3, 2048);
    defer lanes.deinit();
    var b: forward.Buffers = try .init(c, 512, 3, capacity);
    defer b.deinit();
    const start = cuda.monotonic();
    try pieces_file.greedy(a, m, lanes, &b, list, null);
    const at_once = cuda.monotonic() - start;

    var state: forward.State = try .init(gpa, c, 2048);
    defer state.deinit();
    const logits = try a.alloc(u16, c.vocab);
    var tokens: usize = 0;
    const alone_start = cuda.monotonic();
    for (list, 0..) |piece, i| {
        errdefer std.debug.print("piece {d}\n", .{i});
        try state.reset();
        state.audio = piece.audio;
        var reply: std.ArrayList(u32) = .empty;
        try draft.greedy(a, m, &state, &b, piece.prompt, piece.limit, logits, &reply, null);
        try testing.expectEqualSlices(u32, reply.items, piece.reply.items);
        tokens += reply.items.len;
    }
    const alone = cuda.monotonic() - alone_start;
    std.debug.print("qwen3_asr pieces: {d} pieces, {d} tokens, at once in 3 lanes {d:.0} ms, one by one (graphed steps) {d:.0} ms\n", .{ list.len, tokens, @as(f64, @floatFromInt(at_once)) / 1e6, @as(f64, @floatFromInt(alone)) / 1e6 });
}

/// A subject's W4A16_NVFP4 build (`$ZSPARK_QWEN3_ASR_NVFP4`, `$ZSPARK_QWEN3_ASR_0_6B_NVFP4`) as stored and as bf16
/// of its values, loaded once; it reads its base's fixtures.
fn nvfp4Models(s: *Subject) !?[2]model_file.Model {
    if (s.nvfp4_loaded) |both| return both;
    const path = env(s.nvfp4) orelse return null;
    if (env(s.reference) == null) return null;
    const start = cuda.monotonic();
    const stored = try model_file.open(std.heap.page_allocator, testing.io, path, rows_max, scored_max);
    try cuda.synchronize();
    std.debug.print("qwen3_asr {s} NVFP4: loaded in {d:.0} ms, device {d} MB\n", .{ s.name, @as(f64, @floatFromInt(cuda.monotonic() - start)) / 1e6, cuda.account.bytes >> 20 });
    s.nvfp4_loaded = .{ stored, try dequantized(std.heap.page_allocator, path) };
    return s.nvfp4_loaded;
}

/// The build at `path` with every NVFP4 linear held as bf16 of its weights' values (code x block scale x
/// `weight_scale_2`, rounded to bf16), the rest as `model_file.open` loads it: the stored form's reference.
fn dequantized(gpa: std.mem.Allocator, path: []const u8) !model_file.Model {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try std.Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    var diagnostic: config_file.Diagnostic = .{};
    const config = try config_file.Config.read(arena, testing.io, dir, &diagnostic);
    const quant = try config_file.Config.quantization(arena, testing.io, dir, &diagnostic);
    var checkpoint: safetensors.Checkpoint = try .open(gpa, testing.io, .cwd(), path);
    defer checkpoint.close();
    const layout: layout_file.Layout = try .init(arena, &config, &checkpoint, quant, &diagnostic);
    var m: model_file.Model = try .load(gpa, &checkpoint, layout, config, rows_max, scored_max, &diagnostic);
    const rows = m.pool.rows(rows_max);
    for (m.layers, layout.layers) |*l, t| {
        const groups = .{ .{ "qkv", &[_]modelopt.Linear{ t.q, t.k, t.v } }, .{ "o", &[_]modelopt.Linear{t.o} }, .{ "gu", &[_]modelopt.Linear{ t.gate, t.up } }, .{ "down", &[_]modelopt.Linear{t.down} } };
        inline for (groups) |group| {
            const parts = group[1];
            if (parts[0].scheme == .nvfp4) {
                var weights: [3][]const u8 = undefined;
                for (parts, weights[0..parts.len]) |part, *w| {
                    const values = try arena.alloc(f32, @intCast(part.rows * part.cols));
                    const scale: f32 = @bitCast(std.mem.readInt(u32, part.scale_2.?.data[0..4], .little));
                    requantize.dequantize(part.weight.data, part.scale.?.data, scale, values);
                    const bits = try arena.alloc(u16, values.len);
                    for (bits, values) |*b, v| b.* = numeric.bf16FromF32(v);
                    w.* = std.mem.sliceAsBytes(bits);
                }
                @field(l, group[0]).deinit();
                @field(l, group[0]) = .{ .bf16 = try .initStacked(weights[0..parts.len], @intCast(parts[0].cols), rows) };
            }
        }
    }
    return m;
}

// The stored build's logits against the dequantized one's, carried through the prompt and 32 steps: the lane
// matmul (bf16 MMAs of exact E2M1 weights, a block's products scaled in fp32, `weight_scale_2` on the sum) against
// bf16 GEMMs of the weights' values rounded to bf16, a rounding of every weight the stored form does not have.
// Near, not equal: a layer alone from the same input at 0.99994 or more (held at `floor`), carried logits at
// 0.9977 (en_30s), 0.9981 (zh_30s), 0.9992 (en_5s), every argmax the same (spec 140 WS-03, 2026-10-28). 0.6B's
// build: a layer alone 0.99992 or more, logits 0.9978, 0.9954, 0.9941 and every argmax the same, so its bound is
// the subject's `nvfp4_logits` (spec 140 WS-09).

test "NVFP4 W4A16: the linears its config names stored as NVFP4, its logits near its dequantized weights', both row rules" {
    var ran = false;
    for (&subjects) |*s| {
        const both = try nvfp4Models(s) orelse continue;
        ran = true;
        std.debug.print("qwen3_asr: the {s} NVFP4 build\n", .{s.name});
        try nvfp4Against(s, both[0], both[1]);
    }
    if (!ran) return error.SkipZigTest;
}

fn nvfp4Against(s: *Subject, m: model_file.Model, ref: model_file.Model) !void {
    const gpa = testing.allocator;
    const c = m.config;
    // the build's quantization_config excludes layers 0, 1, 26 and 27; the others' seven linears are NVFP4
    var four: usize = 0;
    for (m.layers, ref.layers) |l, r| {
        inline for (.{ "qkv", "o", "gu", "down" }) |name| {
            four += @intFromBool(@field(l, name) == .four);
            try testing.expect(@field(r, name) == .bf16);
        }
    }
    const kept = [_]usize{ 0, 1, c.layers - 2, c.layers - 1 };
    for (kept) |i| try testing.expect(m.layers[i].qkv == .bf16 and m.layers[i].down == .bf16);
    try testing.expectEqual(4 * (c.layers - kept.len), four);

    var states: [2]forward.State = .{ try .init(gpa, c, capacity), try .init(gpa, c, capacity) };
    defer for (&states) |*state| state.deinit();
    var b: forward.Buffers = try .init(c, rows_max, scored_max, capacity);
    defer b.deinit();
    const logits: [2][]u16 = .{ try gpa.alloc(u16, c.vocab), try gpa.alloc(u16, c.vocab) };
    defer for (logits) |l| gpa.free(l);
    for ([_][]const u8{ "en_30s.free", "zh_30s.free", "en_5s.free" }) |scenario| {
        var case: Case = try .open(gpa, s, scenario);
        defer case.close();
        // each layer alone from the dequantized model's input to it: where the two forms part
        const n = case.ids.len;
        var isolated: Match = .{};
        for (&states) |*state| try case.start(state);
        try forward.embed(ref, states[1], &b, case.ids);
        var input = try rowsOf(gpa, b.h.rows(n));
        defer gpa.free(input);
        for (0..c.layers) |i| {
            var outs: [2][]u16 = undefined;
            for ([_]model_file.Model{ m, ref }, states, &outs) |each, state, *out| {
                try b.h.uploadAt(0, std.mem.sliceAsBytes(input));
                try forward.layer(each, i, state, &b, n, n);
                out.* = try rowsOf(gpa, b.h.rows(n));
            }
            defer gpa.free(outs[0]);
            // position 0, the attention sink, apart as in the reference's test
            isolated.merge(.of(outs[0][c.hidden..], u16, outs[1][c.hidden..], c.hidden));
            gpa.free(input);
            input = outs[1];
        }
        var match: Match = .{};
        var misses: usize = 0;
        var ties: usize = 0;
        const steps = @min(32, case.tokens.len);
        for (0..steps + 1) |step| {
            for ([_]model_file.Model{ m, ref }, &states, logits) |each, *state, out| {
                if (step == 0) {
                    try case.start(state);
                    try forward.forwardScoring(each, state, &b, case.ids, .last);
                    try forward.commit(each, state, b, case.ids, case.ids.len);
                } else {
                    const token = [_]i32{case.tokens[step - 1]};
                    try forward.forwardScoring(each, state, &b, &token, .last);
                    try forward.commit(each, state, b, &token, 1);
                }
                try forward.logitsRow(b, 0, out);
            }
            match.merge(.of(logits[0], u16, logits[1], c.vocab));
            const got = argmax(logits[0]);
            const want = argmax(logits[1]);
            if (got != want) {
                // a near tie: the stored pick within `tie_steps` bf16 steps of the dequantized one's best
                const top = numeric.f32FromBf16(logits[1][want]);
                const gap = top - numeric.f32FromBf16(logits[1][got]);
                if (gap <= @as(f32, tie_steps) * top / 128) ties += 1 else misses += 1;
            }
        }
        std.debug.print("qwen3_asr NVFP4 {s}: a layer alone {d:.6} (max abs {d:.3}); the prompt's last row and {d} steps, logits {d:.6} (max abs {d:.3}) against the dequantized weights', argmax {d} near ties, {d} misses\n", .{ scenario, isolated.worst, isolated.max_abs, steps, match.worst, match.max_abs, ties, misses });
        try testing.expect(isolated.worst >= floor and match.worst >= s.nvfp4_logits and misses == 0);
    }
    try rowRule(s, m);
    try rowRuleLong(s, m);
}
