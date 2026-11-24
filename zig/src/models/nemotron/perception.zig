//! Nemotron-3.5-Transcribe's perception on the device (spec 151 WS-03): a clip's 16 kHz samples to the bf16
//! [rows, output_dim] embeddings that replace a prompt's audio placeholder rows. The checkpoint mounts a
//! ParallelExpertEncoder (`config.pe_encoder`; `parallel_expert_encoder.py`, commit 774bc30): NeMo's
//! AudioPerceptionModule computes the whole clip's log-mel once (`mel.zig`), the PEE normalizes it
//! `per_feature` over the valid frames (`asr_normalize_type`, `normalize_batch`), windows it in encoded-row
//! spans — `online_inference` over `chunk_feat_len` cores of `pe_core` rows (`online_inference_length`,
//! 500) plus `pe_left`/`pe_right` context rows (50) a side, each window its own sequence of the encoder's
//! batch with its key rows masked at `ceil(length / subsampling)` and only its core rows kept — and fuses
//! `asr_norm` before the IdentityConnector and `perception.proj` (`_fuse_diar_and_asr` minus the speaker
//! infusion, which needs the Sortformer diarization expert — out of scope, see AGENTS.md). With
//! `pe_encoder` unset NeMo instead runs `encoder_chunking.encode_audio_with_optional_chunking`: the mel in
//! `chunk_seconds` sample chunks, each its own segment.
//!
//! The encoder's stack (TransformerEncoder): FeatureStacking of `subsampling` frames a row (zeros past a
//! window's frames, `pre_encode.proj` without a bias), `embed_norm`, the layers' pre-norm blocks (x +
//! attn(norm1 x), x + ffn(norm2 x); qk LayerNorm and RoPE on the prep kernel `perception.cu` — a row's RoPE
//! position its index in its window — the attention `perception_attention.cu`, exact GELU in the FFN),
//! `final_norm`, `asr_norm` under the PEE, `perception.proj` with its bias. A window's rows never depend on
//! another window's.
//!
//! The FP8 linears (w_qkv, out_proj, the FFN's) run bf16 activations through `weights/nvfp4.zig`'s Fp8Linear
//! and `prompt_bias.cu`'s FP8 mode — more precise than the card's static input_scale (its vLLM quantizes the
//! activations; W8A8 would go through `quantizeActivations`/`qmm_a8.cu` beside `prefill` when taken).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const safetensors = @import("../../weights/safetensors.zig");
const nvfp4 = @import("../../weights/nvfp4.zig");
const bf16 = @import("../../weights/bf16.zig");
const config_file = @import("config.zig");
const layout_file = @import("layout.zig");
const mel_file = @import("mel.zig");
const speaker_file = @import("speaker.zig");
const prompt_file = @import("prompt.zig");

pub const Error = cuda.Error || Allocator.Error || error{ MissingTensor, UnsupportedShape };

extern fn zs_nemotron_perception_prep(qkv: *const cuda.View, offsets: *const cuda.View, qw: *const cuda.View, qb: *const cuda.View, kw: *const cuda.View, kb: *const cuda.View, inv: *const cuda.View, heads: i64, rot: i64, eps: f64) c_int;
extern fn zs_nemotron_perception_attention(qkv: *const cuda.View, tiles: *const cuda.View, out: *const cuda.View) c_int;

pub const eps = 1e-5;

/// A LayerNorm's weight and bias, bf16 [d_model] or [head_dim].
pub const Norm = struct {
    weight: cuda.Buffer,
    bias: cuda.Buffer,

    pub fn deinit(self: *Norm) void {
        self.weight.deinit();
        self.bias.deinit();
    }
};

/// An encoder layer's weights (`perception.encoder.asr_encoder.layers.N.`).
pub const Layer = struct {
    norm1: Norm,
    norm2: Norm,
    q_norm: Norm,
    k_norm: Norm,
    w_qkv: nvfp4.Fp8Linear,
    out_proj: nvfp4.Fp8Linear,
    out_bias: cuda.Buffer,
    ffn0: nvfp4.Fp8Linear,
    ffn0_bias: cuda.Buffer,
    ffn3: nvfp4.Fp8Linear,
    ffn3_bias: cuda.Buffer,

    fn deinit(self: *Layer) void {
        self.norm1.deinit();
        self.norm2.deinit();
        self.q_norm.deinit();
        self.k_norm.deinit();
        self.w_qkv.deinit();
        self.out_proj.deinit();
        self.out_bias.deinit();
        self.ffn0.deinit();
        self.ffn0_bias.deinit();
        self.ffn3.deinit();
        self.ffn3_bias.deinit();
    }
};

fn view(buffer: cuda.Buffer, shape: []const usize) cuda.View {
    var bytes: usize = 2;
    for (shape) |dim| bytes *= dim;
    std.debug.assert(bytes <= buffer.bytes());
    return .init(buffer.view.ptr.?, .bf16, shape);
}

fn at(base: cuda.View, offset_elements: usize, shape: []const usize) cuda.View {
    return .init(@ptrCast(@as([*]u8, @ptrCast(base.ptr.?)) + offset_elements * 2), .bf16, shape);
}

/// norm: y = LayerNorm(h) bf16; with `branch`, h = bf(h + branch) first, written back (`zs_layernorm`).
fn norm(h: cuda.View, branch: ?cuda.View, by: ?*const Norm, y: ?cuda.View) cuda.Error!void {
    const b: ?*const cuda.View = if (branch) |*v| v else null;
    const w: ?*const cuda.View = if (by) |n| &n.weight.view else null;
    const bi: ?*const cuda.View = if (by) |n| &n.bias.view else null;
    const yv: ?*const cuda.View = if (y) |*v| v else null;
    return cuda.launched(cuda.kernels.zs_layernorm(&h, b, w, bi, yv, eps));
}

fn tensor(checkpoint: *const safetensors.Checkpoint, name: []const u8) !safetensors.Tensor {
    return checkpoint.get(name) orelse error.MissingTensor;
}

fn uploadTensor(checkpoint: *const safetensors.Checkpoint, name: []const u8, shape: []const usize) !cuda.Buffer {
    const t = try tensor(checkpoint, name);
    std.debug.assert(t.data.len == blk: {
        var c: usize = 2;
        for (shape) |d| c *= d;
        break :blk c;
    });
    var buffer: cuda.Buffer = try .init(.bf16, shape);
    errdefer buffer.deinit();
    try buffer.upload(t.data);
    return buffer;
}

fn normOf(checkpoint: *const safetensors.Checkpoint, name: []const u8, width: usize) !Norm {
    var buffer: [128]u8 = undefined;
    return .{
        .weight = try uploadTensor(checkpoint, std.fmt.bufPrint(&buffer, "{s}.weight", .{name}) catch unreachable, &.{width}),
        .bias = try uploadTensor(checkpoint, std.fmt.bufPrint(&buffer, "{s}.bias", .{name}) catch unreachable, &.{width}),
    };
}

fn scaleOf(checkpoint: *const safetensors.Checkpoint, name: []const u8) !f64 {
    const t = try tensor(checkpoint, name);
    std.debug.assert(t.data.len == 4);
    return @as(f32, @bitCast(std.mem.bytesToValue(u32, t.data[0..4])));
}

/// A per-tensor FP8 linear as ModelOpt stores it: e4m3 [n, k], an F32 tensor scale, an input scale not read
/// (activations stay bf16).
fn fp8Of(gpa: Allocator, checkpoint: *const safetensors.Checkpoint, name: []const u8, n: usize, k: usize) !nvfp4.Fp8Linear {
    var buffer: [160]u8 = undefined;
    const w = try tensor(checkpoint, std.fmt.bufPrint(&buffer, "{s}.weight", .{name}) catch unreachable);
    std.debug.assert(w.data.len == n * k);
    return nvfp4.Fp8Linear.init(gpa, w.data, n, k, try scaleOf(checkpoint, std.fmt.bufPrint(&buffer, "{s}.weight_scale", .{name}) catch unreachable));
}

/// The perception module: weights, the host tables (fb, window in f32; inv_freq), and scratch for at most
/// `max_chunks` encoder chunks a call.
pub const Perception = struct {
    gpa: Allocator,
    params: mel_file.Params,
    /// `encoder_chunk_size_seconds` in samples (the non-PEE path's chunk span).
    chunk_samples: usize,
    /// The ParallelExpertEncoder's online windowing in mel frames (`config.pe_core`/`pe_left`/`pe_right`
    /// x `subsampling`); `pe` false keeps the `chunk_seconds` sample chunks.
    pe: bool,
    pe_core: usize,
    pe_left: usize,
    pe_right: usize,
    /// `asr_normalize_type` ("per_feature" on the whole clip's mel; `config.pe_norm`).
    pe_norm: bool,
    subsampling: usize,
    stacked: usize,
    d_model: usize,
    heads: usize,
    ff: usize,
    /// RoPE's dims of a head (`rotary_fraction` x head_dim), even.
    rot: usize,
    /// Rows the largest segment gives: a PEE window's `ceil((core + left + right) / sub)`, or the largest
    /// 30 s chunk's (`mel.frames(max_span) / subsampling`, rounded up).
    rows_per_chunk: usize,
    max_chunks: usize,
    fb: []f32,
    window: []f32,
    inv: cuda.Buffer,
    pre_encode: bf16.Linear,
    embed_norm: Norm,
    final_norm: Norm,
    /// The PEE's fusion LayerNorm over the ASR states (`perception.encoder.asr_norm`). Absent where
    /// the checkpoint mounts no PEE.
    asr_norm: ?Norm,
    /// The Sortformer speaker encoder and the clip's activity it scores (spec 151 WS-11): mounted
    /// where `config.diar` parses and `perception.encoder.diarization_model.*` tensors exist. `act`
    /// is bf16 [max_rows, speakers] of a clip's aligned, thresholded-or-not activity rows.
    speaker: ?speaker_file.Speaker,
    act: ?cuda.Buffer,
    proj_weight: cuda.Buffer,
    proj_bias: cuda.Buffer,
    output_dim: usize,
    enc: []Layer,
    pool: *cuda.Scratch,
    mel_scratch: mel_file.Scratch,
    /// bf16 [max_rows, stacked]: the uploaded stacked mel of a call's chunks.
    input: cuda.Buffer,
    x: cuda.Buffer,
    y: cuda.Buffer,
    qkv: cuda.Buffer,
    attn: cuda.Buffer,
    ffn: cuda.Buffer,
    branch: cuda.Buffer,
    tiles: cuda.Buffer,
    offsets: cuda.Buffer,
    /// bf16 [max_rows, output_dim]: a PEE group's projected rows before the kept core rows' compaction
    /// (`zs_embed_rows`), and the kernel's i32 (out row, batch row) pairs. Absent where the checkpoint
    /// mounts no PEE (the 30 s chunk path writes `out` straight).
    proj_tmp: ?cuda.Buffer,
    pairs: ?cuda.Buffer,

    /// A clip's audio rows — the plugin's single pass `ceil(frames / subsampling)` where the checkpoint
    /// mounts the PEE (it owns the windowing), `mel.rows` over `chunk_seconds` spans otherwise; the same
    /// count `prompt.zig`'s `clipRows` gives.
    pub fn rows(self: *const Perception, arena: Allocator, samples: usize) Allocator.Error!usize {
        if (self.pe) return @max(1, (self.params.frames(samples) + self.subsampling - 1) / self.subsampling);
        return mel_file.rows(arena, self.params, @as(f64, @floatFromInt(self.chunk_samples)) / @as(f64, @floatFromInt(self.params.rate)), self.subsampling, samples);
    }

    pub fn load(gpa: Allocator, arena: Allocator, c: *const config_file.Config, checkpoint: *const safetensors.Checkpoint, max_chunks: usize) Error!Perception {
        const e = c.perception;
        const params: mel_file.Params = .{
            .n_fft = e.n_fft,
            .window = e.windowFrames(),
            .hop = @intFromFloat(@round(e.window_stride * @as(f64, @floatFromInt(e.sample_rate)))),
            .features = e.features,
            .rate = e.sample_rate,
        };
        const chunk_samples = @max(1, @as(usize, @intFromFloat(@round(c.chunk_seconds * @as(f64, @floatFromInt(e.sample_rate))))));
        // a segment's rows: the PEE's context-extended window (`pe_core + pe_left + pe_right` mel frames)
        // or the largest chunk's (a full chunk plus a folded tail of at most min_chunk - 1 samples)
        const pe_core = c.pe_core * e.subsampling;
        const pe_left = c.pe_left * e.subsampling;
        const pe_right = c.pe_right * e.subsampling;
        var min_chunk = params.hop;
        for (0..16) |_| {
            if (params.frames(min_chunk) >= 2) break;
            min_chunk += params.hop;
        }
        const max_span = if (c.pe_encoder) pe_core + pe_left + pe_right else chunk_samples + min_chunk - 1;
        const rows_per_chunk = if (c.pe_encoder) (max_span + e.subsampling - 1) / e.subsampling else (params.frames(max_span) + e.subsampling - 1) / e.subsampling;
        const max_rows = max_chunks * rows_per_chunk;
        const rot = @as(usize, @intFromFloat(e.rotary_fraction * @as(f64, @floatFromInt(e.headDim()))));
        var self: Perception = undefined;
        self.gpa = gpa;
        self.params = params;
        self.chunk_samples = chunk_samples;
        self.pe = c.pe_encoder;
        self.pe_core = pe_core;
        self.pe_left = pe_left;
        self.pe_right = pe_right;
        self.pe_norm = c.pe_norm;
        self.subsampling = e.subsampling;
        self.stacked = e.stacked();
        self.d_model = e.d_model;
        self.heads = e.heads;
        self.ff = e.ffWidth();
        self.rot = rot;
        self.rows_per_chunk = rows_per_chunk;
        self.max_chunks = max_chunks;
        self.output_dim = e.output_dim;

        // the featurizer's stored tables in f32 (bf16 as stored, read once)
        const fb_t = try tensor(checkpoint, "perception.preprocessor.featurizer.fb");
        const win_t = try tensor(checkpoint, "perception.preprocessor.featurizer.window");
        const fb = try arena.alloc(f32, params.features * (params.n_fft / 2 + 1));
        const window = try arena.alloc(f32, params.window);
        for (fb, 0..) |*v, i| v.* = f32Of(std.mem.bytesAsSlice(u16, fb_t.data)[i]);
        for (window, 0..) |*v, i| v.* = f32Of(std.mem.bytesAsSlice(u16, win_t.data)[i]);
        self.fb = fb;
        self.window = window;
        self.mel_scratch = try .init(arena, params, max_span);
        // RoPE's inverse frequencies in fp32 (RotaryPositionalEncoding, arange(0, rot, 2) / rot)
        {
            const host = try gpa.alloc(f32, rot / 2);
            defer gpa.free(host);
            for (host, 0..) |*v, i| v.* = @floatCast(1.0 / std.math.pow(f64, e.rope_base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rot))));
            self.inv = try .init(.f32, &.{rot / 2});
            try self.inv.upload(std.mem.sliceAsBytes(host));
        }
        self.pool = try cuda.Scratch.create(gpa);
        const pool_rows = self.pool.rows(max_rows);
        const pre_t = try tensor(checkpoint, "perception.encoder.asr_encoder.pre_encode.proj.weight");
        self.pre_encode = try bf16.Linear.init(pre_t.data, e.d_model, self.stacked, pool_rows);
        self.embed_norm = try normOf(checkpoint, "perception.encoder.asr_encoder.embed_norm", e.d_model);
        self.final_norm = try normOf(checkpoint, "perception.encoder.asr_encoder.final_norm", e.d_model);
        self.asr_norm = if (c.pe_encoder) try normOf(checkpoint, "perception.encoder.asr_norm", e.d_model) else null;
        self.proj_weight = try uploadTensor(checkpoint, "perception.proj.weight", &.{ e.output_dim, e.d_model });
        self.proj_bias = try uploadTensor(checkpoint, "perception.proj.bias", &.{e.output_dim});
        self.enc = try gpa.alloc(Layer, e.layers);
        var name: [128]u8 = undefined;
        var leaf: [160]u8 = undefined;
        for (self.enc, 0..) |*layer, i| {
            const p = std.fmt.bufPrint(&name, "{s}{d}", .{ layout_file.encoder_prefix, i }) catch unreachable;
            layer.norm1 = try normOf(checkpoint, cat(&leaf, p, ".norm1"), e.d_model);
            layer.norm2 = try normOf(checkpoint, cat(&leaf, p, ".norm2"), e.d_model);
            layer.q_norm = try normOf(checkpoint, cat(&leaf, p, ".attn.q_norm"), e.headDim());
            layer.k_norm = try normOf(checkpoint, cat(&leaf, p, ".attn.k_norm"), e.headDim());
            layer.w_qkv = try fp8Of(gpa, checkpoint, cat(&leaf, p, ".attn.w_qkv"), 3 * e.d_model, e.d_model);
            layer.out_proj = try fp8Of(gpa, checkpoint, cat(&leaf, p, ".attn.out_proj"), e.d_model, e.d_model);
            layer.out_bias = try uploadTensor(checkpoint, cat(&leaf, p, ".attn.out_proj.bias"), &.{e.d_model});
            layer.ffn0 = try fp8Of(gpa, checkpoint, cat(&leaf, p, ".ffn.net.0"), e.ffWidth(), e.d_model);
            layer.ffn0_bias = try uploadTensor(checkpoint, cat(&leaf, p, ".ffn.net.0.bias"), &.{e.ffWidth()});
            layer.ffn3 = try fp8Of(gpa, checkpoint, cat(&leaf, p, ".ffn.net.3"), e.d_model, e.ffWidth());
            layer.ffn3_bias = try uploadTensor(checkpoint, cat(&leaf, p, ".ffn.net.3.bias"), &.{e.d_model});
        }
        self.input = try .init(.bf16, &.{ max_rows, self.stacked });
        self.x = try .init(.bf16, &.{ max_rows, e.d_model });
        self.y = try .init(.bf16, &.{ max_rows, e.d_model });
        self.qkv = try .init(.bf16, &.{ max_rows, 3 * e.d_model });
        self.attn = try .init(.bf16, &.{ max_rows, e.d_model });
        self.ffn = try .init(.bf16, &.{ max_rows, e.ffWidth() });
        self.branch = try .init(.bf16, &.{ max_rows, e.d_model });
        self.tiles = try .init(.i32, &.{ max_rows, 4 });
        self.offsets = try .init(.i32, &.{max_chunks + 1});
        self.proj_tmp = if (c.pe_encoder) try .init(.bf16, &.{ max_rows, e.output_dim }) else null;
        self.pairs = if (c.pe_encoder) try .init(.i32, &.{ max_rows, 2 }) else null;
        // the speaker expert: a window's stacked rows bound the packed segment (`max_window`); the
        // activity holds the whole clip the route accepts — `encodePe`'s `encoded_len` is the clip's
        // full row count, which a 3,600 s file reaches past `max_chunks * rows_per_chunk`
        if (c.diar != null and c.pe_encoder) {
            const longest: usize = @intFromFloat(prompt_file.max_seconds * @as(f64, @floatFromInt(e.sample_rate)));
            const clip_rows = @max(max_chunks * rows_per_chunk, @max(1, (params.frames(longest) + e.subsampling - 1) / e.subsampling));
            self.speaker = try speaker_file.Speaker.load(gpa, arena, c, checkpoint, e.d_model, rows_per_chunk);
            self.act = try .init(.bf16, &.{ clip_rows, c.diar.?.speakers });
        } else {
            self.speaker = null;
            self.act = null;
        }
        return self;
    }

    /// `samples` f32 at 16 kHz -> `out` bf16 [rows(samples), output_dim]. Under the PEE the whole clip's
    /// mel, normalized `per_feature`, windows `max_chunks` of them an encoder call (`_forward_online_impl`);
    /// else the `chunk_seconds` spans, `max_chunks` a call. A window's bits never depend on its group.
    pub fn encode(self: *Perception, arena: Allocator, samples: []const f32, out: cuda.View) Error!void {
        if (self.pe) return self.encodePe(arena, samples, out);
        const spans = try mel_file.chunkSpans(arena, self.params, self.chunkSeconds(), samples.len);
        var at_row: usize = 0;
        var rest = spans;
        while (rest.len > 0) {
            const take = @min(rest.len, self.max_chunks);
            const slice = at(out, at_row * self.output_dim, &.{ @as(usize, @intCast(out.size[0])) - at_row, self.output_dim });
            at_row += try self.encodeGroup(arena, samples, rest[0..take], slice);
            rest = rest[take..];
        }
    }

    fn chunkSeconds(self: *const Perception) f64 {
        return @as(f64, @floatFromInt(self.chunk_samples)) / @as(f64, @floatFromInt(self.params.rate));
    }

    /// The PEE's `forward` under `online_inference`: the clip's mel once (`p.frames` valid frames; the
    /// STFT's `stftFrames` padded width is the windows' tiling), `normalize_batch` per feature over them,
    /// then `ceil(T / pe_core)` windows of `pe_core` core frames with `pe_left`/`pe_right` context each
    /// (`_forward_online_impl`'s windows; a window's length clamps at the clip's valid frames). Each
    /// window's stacked rows are one attention segment masked at `ceil(length / sub)` keys; only its core
    /// rows are kept, compacted into `out` by `zs_embed_rows`. NeMo's `encoded_len` (the clip's row count)
    /// is `ceil(frames / sub)`.
    fn encodePe(self: *Perception, arena: Allocator, samples: []const f32, out: cuda.View) Error!void {
        const p = self.params;
        const sub = self.subsampling;
        const valid = p.frames(samples.len);
        const width = p.stftFrames(samples.len); // the STFT's padded width the windows tile
        // the clip's encoded length: `encoded_len = ceil(valid / sub)`, the kept rows' cap
        const encoded_len = @max(1, (valid + sub - 1) / sub);
        // host: the whole clip's mel [valid, features], `per_feature` over `valid` (masked frames zeroed)
        const mel_rows = try arena.alloc(f32, width * p.features);
        @memset(mel_rows, 0);
        var scratch = self.mel_scratch;
        scratch.signal = try arena.alloc(f32, @max(samples.len, 1));
        mel_file.features(p, &scratch, self.fb, self.window, samples, mel_rows[0 .. valid * p.features]);
        // NeMo normalizes ASR and diarization independently. Sortformer's `NA` takes the raw mel,
        // including the masked STFT tail: `_forward_online_impl` windows over the tensor's full width.
        const diar_rows = if (self.speaker != null and self.pe_norm)
            try arena.dupe(f32, mel_rows)
        else
            mel_rows;
        if (self.pe_norm) mel_file.normalizePerFeature(p, mel_rows, valid);
        // the windows over `width`: [start, end) core + [context_start, context_end) context; a window's
        // kept core rows stop at `encoded_len` (NeMo's `emb[:emblen]` tail cut)
        const num_windows = @max(1, (width + self.pe_core - 1) / self.pe_core);
        var windows = try arena.alloc([6]usize, num_windows);
        var out_row_total: usize = 0;
        for (windows, 0..) |*w, i| {
            const start = i * self.pe_core;
            const end = @min(start + self.pe_core, width);
            const cs = if (start >= self.pe_left) start - self.pe_left else 0;
            const ce = @min(end + self.pe_right, width);
            const core_rows = @min(encoded_len, (end + sub - 1) / sub) - (start + sub - 1) / sub;
            w.* = .{ cs, ce, core_rows, (start - cs) / sub, out_row_total, start - cs };
            out_row_total += core_rows;
        }
        // a window's (cs, ce, core_rows, left_rows, out_row, left_off): `cs..ce` frames stacked,
        // `key_len` keys, rows [left_rows, + core_rows) kept into out at out_row; `left_off` the raw
        // left context in mel frames (the speaker's `round(left_off / sub)` reads it).
        // The speaker expert's streaming state carries window to window across groups: its activity
        // over the clip's `encoded_len` rows runs once here (`_forward_online_impl`'s Sortformer
        // loop), the windows' kept spans fused after `runEncoder`'s `final_norm`.
        if (self.speaker) |*spk|
            try spk.activity(arena, diar_rows, p.features, valid, windows, encoded_len, self.act.?.rows(encoded_len));
        var w0: usize = 0;
        while (w0 < windows.len) {
            const take = @min(windows.len - w0, self.max_chunks);
            var batch_rows: usize = 0;
            for (windows[w0..][0..take]) |w| batch_rows += (w[1] - w[0] + sub - 1) / sub;
            const rows_bf = try arena.alloc(u16, batch_rows * self.stacked);
            @memset(rows_bf, 0);
            const offs = try arena.alloc(i32, take + 1);
            const tiles = try arena.alloc([4]i32, batch_rows / 128 + take + 1);
            const kept = try arena.alloc([2]i32, batch_rows);
            offs[0] = 0;
            var row0: usize = 0;
            var tile_count: usize = 0;
            var kept_count: usize = 0;
            for (windows[w0..][0..take], 0..) |w, i| {
                const cs, const ce, const core_rows, const left_drop, const out_row = w[0..5].*;
                const seg_rows = (ce - cs + sub - 1) / sub;
                const key_len = @min(seg_rows, (@min(ce - cs, if (cs < valid) valid - cs else 0) + sub - 1) / sub);
                for (cs..ce) |f| {
                    const r = row0 + (f - cs) / sub;
                    const within = (f - cs) % sub;
                    for (0..p.features) |m|
                        rows_bf[r * self.stacked + within * p.features + m] = bf16Of(mel_rows[f * p.features + m]);
                }
                offs[i + 1] = @intCast(row0 + seg_rows);
                var q: usize = 0;
                while (q < seg_rows) : (q += 128) {
                    tiles[tile_count] = .{ @intCast(row0 + q), @intCast(row0), @intCast(seg_rows), @intCast(key_len) };
                    tile_count += 1;
                }
                for (0..core_rows) |k| {
                    kept[kept_count] = .{ @intCast(out_row + k), @intCast(row0 + left_drop + k) };
                    kept_count += 1;
                }
                row0 += seg_rows;
            }
            try self.runEncoder(arena, rows_bf[0 .. row0 * self.stacked], offs[0 .. take + 1], tiles[0..tile_count], row0);
            const row_base = try arena.alloc(usize, take);
            {
                var rb: usize = 0;
                for (windows[w0..][0..take], 0..) |w, i| {
                    row_base[i] = rb;
                    rb += (w[1] - w[0] + sub - 1) / sub;
                }
            }
            try self.project(arena, windows[w0..][0..take], row_base, row0, view(self.proj_tmp.?, &.{ row0, self.output_dim }));
            var pairs_view = self.pairs.?.view;
            pairs_view.size[0] = @intCast(kept_count);
            try cuda.upload(pairs_view, std.mem.sliceAsBytes(kept[0..kept_count]));
            try cuda.launched(cuda.kernels.zs_embed_rows(&self.proj_tmp.?.view, &pairs_view, &out, 1));
            w0 += take;
        }
    }

    /// One group of chunk spans of `samples`: host mel and stacking into `input`, the encoder's layers over the
    /// batch's rows, `perception.proj` into `out` from its first row. Returns the group's row count.
    fn encodeGroup(self: *Perception, arena: Allocator, samples: []const f32, spans: []const [2]usize, out: cuda.View) Error!usize {
        const p = self.params;
        const sub = self.subsampling;
        // host: each chunk's mel [frames, feat] stacked into bf16 rows [rows, stacked], zeros past its frames
        var total: usize = 0;
        for (spans) |span| total += (p.frames(span[1] - span[0]) + sub - 1) / sub;
        const rows_bf = try arena.alloc(u16, total * self.stacked);
        @memset(rows_bf, 0);
        const frames = try arena.alloc(f32, self.rows_per_chunk * sub * p.features);
        const offs = try arena.alloc(i32, spans.len + 1);
        const tiles = try arena.alloc([4]i32, total / 128 + spans.len + 1);
        offs[0] = 0;
        var row0: usize = 0;
        var tile_count: usize = 0;
        for (spans, 0..) |span, i| {
            const n_samples = span[1] - span[0];
            const t_len = p.frames(n_samples);
            const chunk_rows = (t_len + sub - 1) / sub;
            mel_file.features(p, &self.mel_scratch, self.fb, self.window, samples[span[0]..span[1]], frames[0 .. t_len * p.features]);
            // FeatureStacking: row r holds frames r*sub .. r*sub+sub-1's features in order, zeros past t_len
            for (0..t_len) |f| {
                const r = row0 + f / sub;
                const within = f % sub;
                for (0..p.features) |m|
                    rows_bf[r * self.stacked + within * p.features + m] = bf16Of(frames[f * p.features + m]);
            }
            offs[i + 1] = @intCast(row0 + chunk_rows);
            // attention tiles: blocks of 128 queries of this segment, all its rows keys
            var q: usize = 0;
            while (q < chunk_rows) : (q += 128) {
                tiles[tile_count] = .{ @intCast(row0 + q), @intCast(row0), @intCast(chunk_rows), 0 };
                tile_count += 1;
            }
            row0 += chunk_rows;
        }
        try self.runEncoder(arena, rows_bf, offs, tiles[0..tile_count], row0);
        const x = view(self.x, &.{ row0, self.d_model });
        if (self.asr_norm) |*nrm| try norm(x, null, nrm, x);
        try cuda.launched(cuda.kernels.zs_b16_gemm_bias(&x, &self.proj_weight.view, &self.proj_bias.view, &at(out, 0, &.{ row0, self.output_dim }), 0));
        return row0;
    }

    /// `rows_bf` [n, stacked] stacked mel rows through `input`, the encoder's layers over the `offs`
    /// segments (attention tiles' key lengths honored) and `final_norm`; `x` [n, d_model] holds the
    /// states on return. `project` then maps them (the PEE's `asr_norm` or the speaker infusion's
    /// `Speaker.fuse` applied to `x` first by the caller) into `proj` [n, output_dim].
    fn runEncoder(self: *Perception, arena: Allocator, rows_bf: []const u16, offs: []const i32, tiles: []const [4]i32, n: usize) Error!void {
        _ = arena;
        const d = self.d_model;
        try self.input.uploadAt(0, std.mem.sliceAsBytes(rows_bf));
        var offsets_view = self.offsets.view;
        offsets_view.size[0] = @intCast(offs.len);
        try cuda.upload(offsets_view, std.mem.sliceAsBytes(offs));
        var tiles_view = self.tiles.view;
        tiles_view.size[0] = @intCast(tiles.len);
        try cuda.upload(tiles_view, std.mem.sliceAsBytes(tiles));
        const input = view(self.input, &.{ n, self.stacked });
        const x = view(self.x, &.{ n, d });
        const y = view(self.y, &.{ n, d });
        const qkv = view(self.qkv, &.{ n, 3 * d });
        const attn = view(self.attn, &.{ n, d });
        const ffn = view(self.ffn, &.{ n, self.ff });
        const branch = view(self.branch, &.{ n, d });
        try self.pre_encode.forward(input, x);
        try norm(x, null, &self.embed_norm, x);
        for (self.enc) |*layer| {
            try norm(x, null, &layer.norm1, y);
            try layer.w_qkv.prefill(y, qkv, 0);
            try cuda.launched(zs_nemotron_perception_prep(&qkv, &offsets_view, &layer.q_norm.weight.view, &layer.q_norm.bias.view, &layer.k_norm.weight.view, &layer.k_norm.bias.view, &self.inv.view, @intCast(self.heads), @intCast(self.rot), eps));
            try cuda.launched(zs_nemotron_perception_attention(&qkv, &tiles_view, &attn));
            try layer.out_proj.forwardBiased(attn, layer.out_bias.view, branch, .none);
            try norm(x, branch, &layer.norm2, y);
            try layer.ffn0.forwardBiased(y, layer.ffn0_bias.view, ffn, .gelu_erf);
            try layer.ffn3.forwardBiased(ffn, layer.ffn3_bias.view, branch, .none);
            try norm(x, branch, null, null);
        }
        try norm(x, null, &self.final_norm, x);
    }

    /// `x`'s first `n` rows of `d_model` through the PEE's fusion and `perception.proj` into `proj`
    /// [n, output_dim]: `asr_norm` alone, or `Speaker.fuse` (asr_norm + the thresholded activity's
    /// infusion) where the speaker expert is mounted. `acts` the clip's activity rows for the batch's
    /// kept rows — the windows' (row0 + left_drop, out_row, core_rows) spans fused one at a time.
    fn project(self: *Perception, arena: Allocator, windows: []const [6]usize, row_base: []const usize, n: usize, proj: cuda.View) Error!void {
        const d = self.d_model;
        const x = view(self.x, &.{ n, d });
        if (self.speaker) |*spk| {
            const s = spk.speakers;
            for (windows, row_base) |w, rb| {
                const core_rows = w[2];
                const left_drop = w[3];
                const out_row = w[4];
                if (core_rows == 0) continue;
                const xs = self.x.span(rb + left_drop, core_rows);
                const as = self.act.?.span(out_row, core_rows);
                try spk.fuse(xs, as, &self.asr_norm.?, self.x.span(rb + left_drop, core_rows));
            }
            _ = arena;
            _ = s;
        } else if (self.asr_norm) |*nrm| {
            try norm(x, null, nrm, x);
        }
        try cuda.launched(cuda.kernels.zs_b16_gemm_bias(&x, &self.proj_weight.view, &self.proj_bias.view, &proj, 0));
    }

    pub fn deinit(self: *Perception) void {
        self.inv.deinit();
        self.pre_encode.deinit();
        self.embed_norm.deinit();
        self.final_norm.deinit();
        if (self.asr_norm) |*nrm| nrm.deinit();
        self.proj_weight.deinit();
        self.proj_bias.deinit();
        for (self.enc) |*layer| layer.deinit();
        self.gpa.free(self.enc);
        self.pool.destroy();
        for ([_]*cuda.Buffer{ &self.input, &self.x, &self.y, &self.qkv, &self.attn, &self.ffn, &self.branch, &self.tiles, &self.offsets }) |buffer| buffer.deinit();
        if (self.proj_tmp) |*b| b.deinit();
        if (self.pairs) |*b| b.deinit();
        if (self.speaker) |*spk| spk.deinit();
        if (self.act) |*b| b.deinit();
        self.* = undefined;
    }
};

fn cat(buffer: []u8, prefix: []const u8, leaf: []const u8) []const u8 {
    return std.fmt.bufPrint(buffer, "{s}{s}", .{ prefix, leaf }) catch unreachable;
}

/// A bf16 bit pattern's f32 value.
pub fn f32Of(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// `x` rounded to bf16's bits (round to nearest even, as the hardware casts).
pub fn bf16Of(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    const rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
    return @intCast(rounded & 0xFFFF);
}
