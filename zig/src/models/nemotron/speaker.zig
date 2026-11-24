//! The PEE's Sortformer diarization expert on the device (spec 151 WS-11): the clip's raw mel
//! (`diar_normalize_type=NA`), separate from the ASR windows' per-feature normalization, through
//! NeMo's streaming Sortformer to the speaker activity `_fuse_diar_and_asr` reads, and the infusion
//! it adds before `perception.proj` (`parallel_expert_encoder.py`,
//! `sortformer_diar_models.py`, `sortformer_modules.py`, commit 774bc30 — AGENTS.md has the module
//! order and every threshold).
//!
//! A window of the PEE's `online_inference_length` cores extended by its left/right context — in mel
//! frames, `subsampling` 8 a row — is a streaming step: FeatureStacking of the window's frames (the
//! stacking grid anchored at `context_start`, a multiple of the factor), the pre-encoded rows
//! concatenated after the speaker cache and FIFO, the encoder stack (`bypass_pre_encode`),
//! `encoder_proj` into `tf_d_model`, `forward_infer` — no post-encoder transformer on this bundle
//! (`transformer_encoder` is null), so the hidden states go straight to the subpixel upsample conv
//! (x8, k = 3, pad 1, inputs past the packed_n length masked first) and the speaker heads' ReLU /
//! sigmoid chain — preds masked at the packed_n length, then `streaming_update`'s FIFO/spkcache
//! bookkeeping on the host. `high_resolution`: the chunk's own preds are the native-rate rows at
//! (spkcache + fifo + lc_enc) x `upsample_factor`, and `downsample_preds` x
//! `output_subsampling_factor` lands them on the ASR row grid, clamped to the window's kept core rows.
//!
//! All bf16 as stored — the quant map names nothing under `diarization_model`. The layer stack shares
//! `perception.zig`'s kernels (the Sortformer layers are BF16 with no qk norms: `speaker.cu`'s `prep`
//! rotates q/k alone, `zs_b16_gemm_bias` runs the linears, `perception_attention.cu` the one-segment
//! attention — the packed_n rows are one masked segment a step, every row's bits its own). The spkcache
//! update — `_compress_spkcache`'s log scores, -inf masks, boosted top-k in frame order, disabled
//! slots the learnable silence embedding — is host code: once a clip, no kernel earns it.
//!
//! Not bit-faithful in structure: attention is `perception_attention.cu`'s (near PyTorch's
//! FlexAttention, as the ASR path's); the spkcache/FIFO bookkeeping, the pops' order and the
//! silence profile are NeMo's exactly; `spk_perm` never permutes at eval so the prediction order is
//! the model's. The last window's `update_state=false` skips `streaming_update` as NeMo's does.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const safetensors = @import("../../weights/safetensors.zig");
const bf16 = @import("../../weights/bf16.zig");
const config_file = @import("config.zig");
const layout_file = @import("layout.zig");
const perception_file = @import("perception.zig");

pub const Error = cuda.Error || Allocator.Error || error{ MissingTensor, UnsupportedShape };

extern fn zs_nemotron_speaker_prep(qkv: *const cuda.View, offsets: *const cuda.View, inv: *const cuda.View, heads: i64, rot: i64) c_int;
extern fn zs_nemotron_speaker_conv_input(emb: *const cuda.View, columns: *const cuda.View, gate: i64) c_int;
extern fn zs_nemotron_speaker_relu(x: *const cuda.View) c_int;
extern fn zs_nemotron_speaker_sigmoid(logits: *const cuda.View, preds: *const cuda.View, gate: i64) c_int;
extern fn zs_nemotron_speaker_fuse(x: *const cuda.View, act: *const cuda.View, aw: *const cuda.View, ab: *const cuda.View, nw: *const cuda.View, nb: *const cuda.View, kernel: *const cuda.View, y: *const cuda.View, scale: f64, eps: f64) c_int;
extern fn zs_nemotron_perception_attention(qkv: *const cuda.View, tiles: *const cuda.View, out: *const cuda.View) c_int;

const eps = perception_file.eps;
const Norm = perception_file.Norm;

fn view(buffer: cuda.Buffer, shape: []const usize) cuda.View {
    var bytes: usize = 2;
    for (shape) |dim| bytes *= dim;
    std.debug.assert(bytes <= buffer.bytes());
    return .init(buffer.view.ptr.?, .bf16, shape);
}

fn norm(h: cuda.View, by: *const Norm, y: cuda.View) cuda.Error!void {
    return cuda.launched(cuda.kernels.zs_layernorm(&h, null, &by.weight.view, &by.bias.view, &y, eps));
}

fn tensor(checkpoint: *const safetensors.Checkpoint, name: []const u8) !safetensors.Tensor {
    return checkpoint.get(name) orelse error.MissingTensor;
}

fn uploadTensor(checkpoint: *const safetensors.Checkpoint, name: []const u8, shape: []const usize) !cuda.Buffer {
    const t = try tensor(checkpoint, name);
    std.debug.assert(t.data.len == blk: {
        var cnt: usize = 2;
        for (shape) |d| cnt *= d;
        break :blk cnt;
    });
    var buffer: cuda.Buffer = try .init(.bf16, shape);
    errdefer buffer.deinit();
    try buffer.upload(t.data);
    return buffer;
}

fn uploadZeros(shape: []const usize) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(.bf16, shape);
    errdefer buffer.deinit();
    try buffer.zero();
    return buffer;
}

fn normOf(checkpoint: *const safetensors.Checkpoint, name: []const u8, width: usize) !Norm {
    var buffer: [192]u8 = undefined;
    return .{
        .weight = try uploadTensor(checkpoint, std.fmt.bufPrint(&buffer, "{s}.weight", .{name}) catch unreachable, &.{width}),
        .bias = try uploadTensor(checkpoint, std.fmt.bufPrint(&buffer, "{s}.bias", .{name}) catch unreachable, &.{width}),
    };
}

fn cat(buffer: []u8, prefix: []const u8, leaf: []const u8) []const u8 {
    return std.fmt.bufPrint(buffer, "{s}{s}", .{ prefix, leaf }) catch unreachable;
}

/// A bf16 [n, k] weight and its [n] bias (or the shared zero row), on `zs_b16_gemm_bias`.
const BiasLinear = struct {
    weight: cuda.Buffer,
    bias: ?cuda.Buffer,
    n: usize,
    k: usize,
    zero: cuda.Buffer,

    fn init(checkpoint: *const safetensors.Checkpoint, name: []const u8, n: usize, k: usize, biased: bool) !BiasLinear {
        var leaf: [224]u8 = undefined;
        var self: BiasLinear = undefined;
        self.n = n;
        self.k = k;
        self.weight = try uploadTensor(checkpoint, cat(&leaf, name, ".weight"), &.{ n, k });
        errdefer self.weight.deinit();
        self.bias = if (biased) try uploadTensor(checkpoint, cat(&leaf, name, ".bias"), &.{n}) else null;
        errdefer if (self.bias) |*b| b.deinit();
        self.zero = if (biased) undefined else try uploadZeros(&.{n});
        return self;
    }

    fn deinit(self: *BiasLinear) void {
        self.weight.deinit();
        if (self.bias) |*b| b.deinit() else self.zero.deinit();
    }

    /// out (m, n) = act(x (m, k) W^T + bias), bf16; `act` is `zs_b16_gemm_bias`'s epilogue id
    /// (0 none, 2 erf-GELU — `prompt_bias.cu`'s codes).
    fn forward(self: *const BiasLinear, x: cuda.View, out: cuda.View, act: i64) cuda.Error!void {
        const bias: *const cuda.View = if (self.bias) |*b| &b.view else &self.zero.view;
        return cuda.launched(cuda.kernels.zs_b16_gemm_bias(&x, &self.weight.view, bias, &out, act));
    }
};

/// A Sortformer encoder layer's weights (`diarization_model.encoder.layers.N.`): norms, w_qkv with
/// no bias, out_proj and the FFN with theirs (`qk_norm: false` — no q/k norms).
const DiarLayer = struct {
    norm1: Norm,
    norm2: Norm,
    w_qkv: BiasLinear,
    out_proj: BiasLinear,
    ffn0: BiasLinear,
    ffn3: BiasLinear,

    fn deinit(self: *DiarLayer) void {
        self.norm1.deinit();
        self.norm2.deinit();
        self.w_qkv.deinit();
        self.out_proj.deinit();
        self.ffn0.deinit();
        self.ffn3.deinit();
    }
};

/// The Sortformer expert: weights, the RoPE table, the sinusoid `diar_kernel`, the streaming state
/// and scratch for a step's packed_n rows.
pub const Speaker = struct {
    gpa: Allocator,
    d_model: usize,
    heads: usize,
    ff: usize,
    layers: usize,
    subsampling: usize,
    stacked: usize,
    speakers: usize,
    tf: usize,
    /// `upsample_factor` = subsampling (high_resolution): native rows an encoder row.
    up: usize,
    /// Effective PEE output factor / native factor: the ASR subsampling (8 on this checkpoint).
    out_factor: usize,
    /// The streaming knobs the PEE writes (`_init_streaming_diar`), in encoder rows.
    fifo_len: usize,
    spkcache_len: usize,
    update_period: usize,
    /// `_compress_spkcache`'s constants (`sortformer_modules` on this bundle).
    sil_frames_per_spk: usize,
    pred_score_threshold: f64,
    scores_boost_latest: f64,
    sil_threshold: f64,
    strong_boost_rate: f64,
    weak_boost_rate: f64,
    min_pos_scores_rate: f64,
    /// `_speaker_features`' `speaker_activity_threshold` and `spk_kernel_scale`.
    activity_threshold: f64,
    spk_kernel_scale: f64,
    /// Stacked mel rows the largest window gives (`ceil((core + left + right) / sub)`), and the
    /// largest packed_n segment a step can run (spkcache + fifo + a window).
    max_window: usize,
    max_rows: usize,
    // weights
    pre_encode: bf16.Linear,
    embed_norm: Norm,
    final_norm: Norm,
    enc: []DiarLayer,
    encoder_proj: BiasLinear,
    upsample_w: cuda.Buffer, // bf16 [tf*up, tf*3], stored Conv1d weights flattened without reordering
    upsample_b: cuda.Buffer, // bf16 [tf*up]
    first_hidden: BiasLinear,
    single_hidden: BiasLinear,
    learnable_sil: []f32, // [d_model] host, `_compress_spkcache`'s disabled slots
    diar_norm: Norm,
    /// The sinusoid `diar_kernel` [speakers, asr_d_model] (`_build_sinusoid_position_encoding`, bf16).
    diar_kernel: cuda.Buffer,
    inv: cuda.Buffer,
    pool: *cuda.Scratch,
    // scratch: the packed_n segment's rows
    input: cuda.Buffer, // [max_rows, stacked] the packed_n stacked rows (state's rows placeholder)
    x: cuda.Buffer, // [max_rows, d_model]
    y: cuda.Buffer,
    qkv: cuda.Buffer, // [max_rows, 3*d_model]
    attn: cuda.Buffer,
    ffn: cuda.Buffer, // [max_rows, ff]
    branch: cuda.Buffer,
    proj: cuda.Buffer, // [max_rows, tf] encoder_proj's
    up_input: cuda.Buffer, // bf16 [max_rows, tf*3], reusable full-channel Conv1d input
    up_buf: cuda.Buffer, // [max_rows*up, tf] the upsampled hidden
    hid: cuda.Buffer, // [max_rows*up, tf] first_hidden's (ReLU'd)
    logits: cuda.Buffer, // [max_rows*up, speakers]
    preds: cuda.Buffer, // [max_rows*up, speakers] sigmoid masked
    tiles: cuda.Buffer,
    offsets: cuda.Buffer,
    // streaming state (`StreamingSortformerState`, synchronous): device rows and host preds
    spkcache: cuda.Buffer, // [spkcache_len, d_model]
    spk_preds: []f32, // [spkcache_len, speakers]
    spk_n: usize,
    spk_compressed: bool,
    fifo: cuda.Buffer, // [fifo_len + max_window, d_model]
    fifo_preds: []f32, // [fifo_len + max_window, speakers]
    fifo_n: usize,
    mean_sil: []f32, // [d_model]
    n_sil: usize,

    /// `max_window_rows` bounds a window's stacked rows; `asr_d` is `diar_kernel`'s width.
    pub fn load(gpa: Allocator, arena: Allocator, c: *const config_file.Config, checkpoint: *const safetensors.Checkpoint, asr_d: usize, max_window_rows: usize) Error!Speaker {
        const d = c.diar orelse return error.UnsupportedShape;
        const max_rows = d.spkcache_len + d.fifo_len + max_window_rows;
        var self: Speaker = undefined;
        self.gpa = gpa;
        self.d_model = d.d_model;
        self.heads = d.heads;
        self.ff = d.ffWidth();
        self.layers = d.layers;
        self.subsampling = d.subsampling;
        self.stacked = d.stacked();
        self.speakers = d.speakers;
        self.tf = d.tf_d_model;
        self.up = d.subsampling;
        self.out_factor = d.output_factor; // native factor is 1 under high_resolution
        self.fifo_len = d.fifo_len;
        self.spkcache_len = d.spkcache_len;
        self.update_period = d.update_period;
        self.sil_frames_per_spk = 1; // the bundle's `spkcache_sil_frames_per_spk`
        self.pred_score_threshold = 0.25;
        self.scores_boost_latest = 0.05;
        self.sil_threshold = 0.2;
        self.strong_boost_rate = 0.75;
        self.weak_boost_rate = 1.5;
        self.min_pos_scores_rate = 0.5;
        self.activity_threshold = 0.5;
        self.spk_kernel_scale = 1.0;
        self.max_window = max_window_rows;
        self.max_rows = max_rows;
        self.pool = try cuda.Scratch.create(gpa);
        const rot = @as(usize, @intFromFloat(d.rotary_fraction * @as(f64, @floatFromInt(d.headDim()))));
        std.debug.assert(rot == d.headDim());
        {
            const host = try gpa.alloc(f32, rot / 2);
            defer gpa.free(host);
            for (host, 0..) |*v, i| v.* = @floatCast(1.0 / std.math.pow(f64, d.rope_base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rot))));
            self.inv = try .init(.f32, &.{rot / 2});
            try self.inv.upload(std.mem.sliceAsBytes(host));
        }
        const prefix = "perception.encoder.diarization_model.";
        var name: [224]u8 = undefined;
        var leaf: [224]u8 = undefined;
        const pre_t = try tensor(checkpoint, cat(&name, prefix, "encoder.pre_encode.proj.weight"));
        self.pre_encode = try bf16.Linear.init(pre_t.data, d.d_model, d.stacked(), self.pool.rows(max_rows));
        self.embed_norm = try normOf(checkpoint, cat(&name, prefix, "encoder.embed_norm"), d.d_model);
        self.final_norm = try normOf(checkpoint, cat(&name, prefix, "encoder.final_norm"), d.d_model);
        self.enc = try gpa.alloc(DiarLayer, d.layers);
        for (self.enc, 0..) |*layer, i| {
            const p = std.fmt.bufPrint(&name, "{s}{d}", .{ layout_file.diar_prefix, i }) catch unreachable;
            layer.norm1 = try normOf(checkpoint, cat(&leaf, p, ".norm1"), d.d_model);
            layer.norm2 = try normOf(checkpoint, cat(&leaf, p, ".norm2"), d.d_model);
            layer.w_qkv = try BiasLinear.init(checkpoint, cat(&leaf, p, ".attn.w_qkv"), 3 * d.d_model, d.d_model, false);
            layer.out_proj = try BiasLinear.init(checkpoint, cat(&leaf, p, ".attn.out_proj"), d.d_model, d.d_model, true);
            layer.ffn0 = try BiasLinear.init(checkpoint, cat(&leaf, p, ".ffn.net.0"), d.ffWidth(), d.d_model, true);
            layer.ffn3 = try BiasLinear.init(checkpoint, cat(&leaf, p, ".ffn.net.3"), d.d_model, d.ffWidth(), true);
        }
        self.encoder_proj = try BiasLinear.init(checkpoint, cat(&name, prefix, "sortformer_modules.encoder_proj"), d.tf_d_model, d.d_model, true);
        self.upsample_w = try uploadTensor(checkpoint, cat(&name, prefix, "sortformer_modules.subpixel_upsample.weight"), &.{ d.tf_d_model * self.up, d.tf_d_model, 3 });
        self.upsample_b = try uploadTensor(checkpoint, cat(&name, prefix, "sortformer_modules.subpixel_upsample.bias"), &.{d.tf_d_model * self.up});
        self.first_hidden = try BiasLinear.init(checkpoint, cat(&name, prefix, "sortformer_modules.first_hidden_to_hidden"), d.tf_d_model, d.tf_d_model, true);
        self.single_hidden = try BiasLinear.init(checkpoint, cat(&name, prefix, "sortformer_modules.single_hidden_to_spks"), d.speakers, d.tf_d_model, true);
        self.diar_norm = try normOf(checkpoint, "perception.encoder.diar_norm", d.speakers);
        {
            const sil_t = try tensor(checkpoint, cat(&name, prefix, "sortformer_modules.learnable_sil_emb"));
            std.debug.assert(sil_t.data.len == d.d_model * 2);
            const sil = try arena.alloc(f32, d.d_model);
            for (sil, std.mem.bytesAsSlice(u16, sil_t.data)) |*v, b| v.* = perception_file.f32Of(b);
            self.learnable_sil = sil;
            // `diar_kernel` is `_build_sinusoid_position_encoding` row s: dims 2i sin(s*10000^(-2i/D)),
            // 2i+1 cos — computed once, bf16 as the buffer casts.
            const kernel = try arena.alloc(u16, d.speakers * asr_d);
            const log_step = -@log(@as(f64, 10000.0)) / @as(f64, @floatFromInt(asr_d));
            for (0..asr_d / 2) |i| {
                const frequency = @exp(@as(f64, @floatFromInt(2 * i)) * log_step);
                for (0..d.speakers) |sp| {
                    const angle = @as(f64, @floatFromInt(sp)) * frequency;
                    kernel[sp * asr_d + 2 * i] = perception_file.bf16Of(@floatCast(@sin(angle)));
                    kernel[sp * asr_d + 2 * i + 1] = perception_file.bf16Of(@floatCast(@cos(angle)));
                }
            }
            self.diar_kernel = try .init(.bf16, &.{ d.speakers, asr_d });
            try self.diar_kernel.upload(std.mem.sliceAsBytes(kernel));
        }
        self.input = try .init(.bf16, &.{ max_rows, self.stacked });
        self.x = try .init(.bf16, &.{ max_rows, d.d_model });
        self.y = try .init(.bf16, &.{ max_rows, d.d_model });
        self.qkv = try .init(.bf16, &.{ max_rows, 3 * d.d_model });
        self.attn = try .init(.bf16, &.{ max_rows, d.d_model });
        self.ffn = try .init(.bf16, &.{ max_rows, self.ff });
        self.branch = try .init(.bf16, &.{ max_rows, d.d_model });
        self.proj = try .init(.bf16, &.{ max_rows, self.tf });
        self.up_input = try .init(.bf16, &.{ max_rows, 3 * self.tf });
        self.up_buf = try .init(.bf16, &.{ max_rows * self.up, self.tf });
        self.hid = try .init(.bf16, &.{ max_rows * self.up, self.tf });
        self.logits = try .init(.bf16, &.{ max_rows * self.up, d.speakers });
        self.preds = try .init(.bf16, &.{ max_rows * self.up, d.speakers });
        self.tiles = try .init(.i32, &.{ max_rows, 4 });
        self.offsets = try .init(.i32, &.{2});
        self.spkcache = try .init(.bf16, &.{ d.spkcache_len, d.d_model });
        self.spk_preds = try gpa.alloc(f32, d.spkcache_len * d.speakers);
        self.spk_n = 0;
        self.spk_compressed = false;
        self.fifo = try .init(.bf16, &.{ d.fifo_len + max_window_rows, d.d_model });
        self.fifo_preds = try gpa.alloc(f32, (d.fifo_len + max_window_rows) * d.speakers);
        self.fifo_n = 0;
        self.mean_sil = try gpa.alloc(f32, d.d_model);
        @memset(self.mean_sil, 0);
        self.n_sil = 0;
        return self;
    }

    /// `_compress_spkcache` on the host: the kept `spkcache_len` frames of `embs` [frames, d_model]
    /// bf16 and `preds` [frames, speakers] f32, in frame order, disabled slots `learnable_sil` and
    /// zeros (`_get_log_pred_scores`, `_disable_low_scores`, the two `_boost_topk_scores`, the +inf
    /// silence slots, `_get_topk_indices`' speaker-major flatten, `_gather_spkcache_and_preds`).
    fn compressSpkcache(self: *Speaker, embs: []const u16, preds: []const f32, frames: usize) Error!void {
        const s = self.speakers;
        const dm = self.d_model;
        const gpa = self.gpa;
        const per_spk = self.spkcache_len / s - self.sil_frames_per_spk;
        const strong = @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(per_spk)) * self.strong_boost_rate)));
        const weak = @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(per_spk)) * self.weak_boost_rate)));
        const min_pos = @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(per_spk)) * self.min_pos_scores_rate)));
        const scores = try gpa.alloc(f64, frames * s);
        defer gpa.free(scores);
        for (0..frames) |f| {
            var log1_sum: f64 = 0;
            for (0..s) |sp| log1_sum += @log(@max(self.pred_score_threshold, 1.0 - preds[f * s + sp]));
            for (0..s) |sp| {
                const p = preds[f * s + sp];
                scores[f * s + sp] = @log(@max(self.pred_score_threshold, p)) - @log(@max(self.pred_score_threshold, 1 - p)) + log1_sum - @log(0.5);
            }
        }
        for (0..s) |sp| {
            var pos: usize = 0;
            for (0..frames) |f| if (scores[f * s + sp] > 0) {
                pos += 1;
            };
            for (0..frames) |f| {
                const speech = preds[f * s + sp] > 0.5;
                if (!speech or (scores[f * s + sp] <= 0 and pos >= min_pos)) scores[f * s + sp] = -std.math.inf(f64);
            }
        }
        if (self.scores_boost_latest > 0) for (@min(self.spkcache_len, frames)..frames) |f| {
            for (0..s) |sp| scores[f * s + sp] += self.scores_boost_latest;
        };
        // `_boost_topk_scores`: the n highest scores of the speaker each -scale * log(0.5)
        const top_idx = try gpa.alloc(usize, frames);
        defer gpa.free(top_idx);
        for (0..s) |sp| {
            for ([_]struct { n: usize, scale: f64 }{ .{ .n = strong, .scale = 2 }, .{ .n = weak, .scale = 1 } }) |boost| {
                if (boost.n == 0) continue;
                for (top_idx, 0..) |*v, i| v.* = i;
                const Ctx = struct { scores: []f64, s: usize, sp: usize };
                std.mem.sort(usize, top_idx, Ctx{ .scores = scores, .s = s, .sp = sp }, struct {
                    fn lt(ctx: Ctx, a: usize, b: usize) bool {
                        return ctx.scores[a * ctx.s + ctx.sp] > ctx.scores[b * ctx.s + ctx.sp];
                    }
                }.lt);
                for (0..@min(boost.n, frames)) |i| scores[top_idx[i] * s + sp] -= boost.scale * @log(0.5);
            }
        }
        // `_get_topk_indices`: the spkcache_len largest of the speaker-major flattened scores (the
        // appended +inf silence slots take their speaker's trailing positions); disabled a -inf or a
        // silence slot, then kept in frame order.
        const per_spk_len = frames + self.sil_frames_per_spk;
        const flat = try gpa.alloc(f64, s * per_spk_len);
        defer gpa.free(flat);
        for (0..s) |sp| {
            for (0..frames) |f| flat[sp * per_spk_len + f] = scores[f * s + sp];
            for (0..self.sil_frames_per_spk) |e| flat[sp * per_spk_len + frames + e] = std.math.inf(f64);
        }
        const order = try gpa.alloc(usize, flat.len);
        defer gpa.free(order);
        for (order, 0..) |*v, i| v.* = i;
        std.mem.sort(usize, order, flat, struct {
            fn lt(sc: []const f64, a: usize, b: usize) bool {
                return sc[a] > sc[b];
            }
        }.lt);
        const top = order[0..self.spkcache_len];
        std.mem.sort(usize, top, {}, std.sort.asc(usize));
        const embs_h = try gpa.alloc(u16, self.spkcache_len * dm);
        defer gpa.free(embs_h);
        for (top, 0..) |flat_idx, i| {
            const f = flat_idx % per_spk_len;
            const disabled = flat[flat_idx] == -std.math.inf(f64) or f >= frames;
            if (disabled) {
                for (0..dm) |j| embs_h[i * dm + j] = perception_file.bf16Of(@floatCast(self.learnable_sil[j]));
                for (0..s) |sp| self.spk_preds[i * s + sp] = 0;
            } else {
                @memcpy(embs_h[i * dm ..][0..dm], embs[f * dm ..][0..dm]);
                for (0..s) |sp| self.spk_preds[i * s + sp] = preds[f * s + sp];
            }
        }
        try self.spkcache.upload(std.mem.sliceAsBytes(embs_h));
        self.spk_n = self.spkcache_len;
        self.spk_compressed = true;
    }

    /// `streaming_update` after a window's preds: `chunk_embs` the window's pre-encoded rows (host
    /// bf16 [rows, d_model]), `lc_enc`/`core` its context and kept rows in encoder frames, `down`
    /// the step's preds over the whole packed_n segment after the `downsample_preds` the step applies
    /// (f32 [packed_n, speakers] — the sigmoid masked output NeMo downsamples before the update). The
    /// FIFO takes the chunk's core rows, the pop moves `update_period` (or more to fit) rows into
    /// the cache, and `_compress_spkcache` runs when the cache overruns.
    fn updateState(self: *Speaker, chunk_embs: []const u16, lc_enc: usize, core: usize, down: []const f32, spk_before: usize, fifo_before: usize) Error!void {
        const s = self.speakers;
        const dm = self.d_model;
        const gpa = self.gpa;
        const chunk_start = spk_before + fifo_before + lc_enc;
        // the FIFO's preds are this step's over [cache | fifo], then the chunk's core appended
        @memcpy(self.fifo_preds[0 .. fifo_before * s], down[spk_before * s .. (spk_before + fifo_before) * s]);
        @memcpy(self.fifo_preds[fifo_before * s ..][0 .. core * s], down[chunk_start * s .. (chunk_start + core) * s]);
        const fifo_embs = try gpa.alloc(u16, (fifo_before + core) * dm);
        defer gpa.free(fifo_embs);
        if (fifo_before > 0) try self.fifo.downloadAt(0, std.mem.sliceAsBytes(fifo_embs)[0 .. fifo_before * dm * 2]);
        @memcpy(fifo_embs[fifo_before * dm ..][0 .. core * dm], chunk_embs[lc_enc * dm .. (lc_enc + core) * dm]);
        var fifo_n = fifo_before + core;
        if (fifo_n > self.fifo_len) {
            const pop = @min(@max(self.update_period, core + fifo_before - self.fifo_len), fifo_n);
            const cache_new = self.spk_n + pop;
            const cache_embs = try gpa.alloc(u16, cache_new * dm);
            defer gpa.free(cache_embs);
            if (self.spk_n > 0) try self.spkcache.downloadAt(0, std.mem.sliceAsBytes(cache_embs)[0 .. self.spk_n * dm * 2]);
            @memcpy(cache_embs[self.spk_n * dm ..][0 .. pop * dm], fifo_embs[0 .. pop * dm]);
            const cache_preds = try gpa.alloc(f32, cache_new * s);
            defer gpa.free(cache_preds);
            // pre-compression the cache rows' fresh preds are this step's own; after it, the kept
            // `spkcache_preds` (NeMo keeps the compressed cache's, which `spk_perm` never reorders
            // at eval)
            if (self.spk_compressed) {
                @memcpy(cache_preds[0 .. self.spk_n * s], self.spk_preds[0 .. self.spk_n * s]);
            } else {
                @memcpy(cache_preds[0 .. self.spk_n * s], down[0 .. self.spk_n * s]);
            }
            @memcpy(cache_preds[self.spk_n * s ..][0 .. pop * s], self.fifo_preds[0 .. pop * s]);
            // `_get_silence_profile` over the popped rows (`use_learnable_sil_emb` leaves it unread;
            // kept anyway — NeMo maintains it only when the flag is off, and it is on here)
            std.mem.copyForwards(u16, fifo_embs[0 .. (fifo_n - pop) * dm], fifo_embs[pop * dm .. fifo_n * dm]);
            std.mem.copyForwards(f32, self.fifo_preds[0 .. (fifo_n - pop) * s], self.fifo_preds[pop * s .. fifo_n * s]);
            fifo_n -= pop;
            if (cache_new > self.spkcache_len) {
                try self.compressSpkcache(cache_embs, cache_preds, cache_new);
            } else {
                if (cache_new > 0) try self.spkcache.uploadAt(0, std.mem.sliceAsBytes(cache_embs));
                @memcpy(self.spk_preds[0 .. cache_new * s], cache_preds);
                self.spk_n = cache_new;
            }
        }
        if (fifo_n > 0) try self.fifo.uploadAt(0, std.mem.sliceAsBytes(fifo_embs[0 .. fifo_n * dm]));
        self.fifo_n = fifo_n;
    }

    /// One streaming step on the window `mel[cs..ce)` (the clip's normed mel rows, f32): stack the
    /// window's frames, pack [spkcache | fifo | chunk] pre-encoded rows, run the encoder,
    /// encoder_proj and `forward_infer`, and write the chunk's native preds (the downsampled ones
    /// `_forward_online_impl` fuses) into `preds` f32 [core*up/out_factor? -> core_rows, speakers].
    /// `lc_enc`/`rc_enc` are the encoder-row offsets NeMo's `forward_streaming_step` computes from
    /// the window's left/right offsets; `last` skips the state update (`update_state=false`).
    fn step(self: *Speaker, arena: Allocator, mel: []const f32, features: usize, cs: usize, ce: usize, valid: usize, lc_enc: usize, rc_enc: usize, last: bool, preds: []f32) Error!void {
        const d = self.d_model;
        const sub = self.subsampling;
        const s = self.speakers;
        const frames = ce - cs;
        const rows = (frames + sub - 1) / sub;
        // `chunk_pre_encode_length`: the window's valid frames clamped to [cs, ce), stacked
        const valid_in_window = if (cs < valid) @min(frames, valid - cs) else 0;
        const gate_rows = @min(rows, (valid_in_window + sub - 1) / sub);
        const packed_n = self.spk_n + self.fifo_n + rows;
        // Only the new chunk goes through pre_encode. Cache and FIFO rows already hold embeddings
        // and are copied into the packed encoder input afterward.
        const packed_stacked = try arena.alloc(u16, rows * self.stacked);
        @memset(packed_stacked, 0);
        for (0..frames) |f| {
            for (0..features) |m|
                packed_stacked[(f / sub) * self.stacked + (f % sub) * features + m] = perception_file.bf16Of(mel[(cs + f) * features + m]);
        }
        try self.input.uploadAt(0, std.mem.sliceAsBytes(packed_stacked[0 .. rows * self.stacked]));
        const chunk_in = view(self.input, &.{ rows, self.stacked });
        const chunk_out = self.x.span(self.spk_n + self.fifo_n, rows);
        try self.pre_encode.forward(chunk_in, chunk_out);
        // streaming_update needs these pre-encoder embeddings after the stack overwrites x.
        const chunk_embs: ?[]u16 = if (!last) blk: {
            const embs = try arena.alloc(u16, rows * d);
            try cuda.download(chunk_out, std.mem.sliceAsBytes(embs));
            break :blk embs;
        } else null;
        // the packed_n `bypass_pre_encode` input: [cache rows | fifo rows | chunk rows]
        if (self.spk_n > 0) try self.x.copyRows(0, self.spkcache, 0, self.spk_n);
        if (self.fifo_n > 0) try self.x.copyRows(self.spk_n, self.fifo, 0, self.fifo_n);
        // the encoder over the single packed_n segment; keys masked at the packed_n valid length
        const key_len = self.spk_n + self.fifo_n + gate_rows;
        const x = view(self.x, &.{ packed_n, d });
        const y = view(self.y, &.{ packed_n, d });
        const qkv = view(self.qkv, &.{ packed_n, 3 * d });
        const attn = view(self.attn, &.{ packed_n, d });
        const ffn = view(self.ffn, &.{ packed_n, self.ff });
        const branch = view(self.branch, &.{ packed_n, d });
        const offs_h = [_]i32{ 0, @intCast(packed_n) };
        var offsets_view = self.offsets.view;
        try cuda.upload(offsets_view, std.mem.sliceAsBytes(&offs_h));
        var tiles_view = self.tiles.view;
        const tile_n = (packed_n + 127) / 128;
        tiles_view.size[0] = @intCast(tile_n);
        const tiles_h = try arena.alloc([4]i32, tile_n);
        for (tiles_h, 0..) |*t, i| t.* = .{ @intCast(i * 128), 0, @intCast(packed_n), @intCast(key_len) };
        try cuda.upload(tiles_view, std.mem.sliceAsBytes(tiles_h));
        try norm(x, &self.embed_norm, x);
        for (self.enc) |*layer| {
            try norm(x, &layer.norm1, y);
            try layer.w_qkv.forward(y, qkv, 0);
            try cuda.launched(zs_nemotron_speaker_prep(&qkv, &offsets_view, &self.inv.view, @intCast(self.heads), @intCast(d / self.heads)));
            try cuda.launched(zs_nemotron_perception_attention(&qkv, &tiles_view, &attn));
            try layer.out_proj.forward(attn, branch, 0);
            try cuda.launched(cuda.kernels.zs_layernorm(&x, &branch, null, null, null, eps));
            try norm(x, &layer.norm2, y);
            try layer.ffn0.forward(y, ffn, 2);
            try layer.ffn3.forward(ffn, branch, 0);
            try cuda.launched(cuda.kernels.zs_layernorm(&x, &branch, null, null, null, eps));
        }
        try norm(x, &self.final_norm, x);
        try self.encoder_proj.forward(x, view(self.proj, &.{ packed_n, self.tf }), 0);
        // `forward_infer` (no post-encoder transformer on this bundle): the subpixel upsample conv
        // over inputs past `key_len` masked (`masked_fill`), ReLU -> first_hidden -> ReLU ->
        // single_hidden, then sigmoid masked at the packed_n length (`apply_mask_to_preds`).
        const conv_input = view(self.up_input, &.{ packed_n, 3 * self.tf });
        const conv_weight = view(self.upsample_w, &.{ self.up * self.tf, 3 * self.tf });
        const conv_out = view(self.up_buf, &.{ packed_n, self.up * self.tf });
        const conv_emb = view(self.proj, &.{ packed_n, self.tf });
        try cuda.launched(zs_nemotron_speaker_conv_input(&conv_emb, &conv_input, @intCast(key_len)));
        try cuda.launched(cuda.kernels.zs_b16_gemm_bias(&conv_input, &conv_weight, &self.upsample_b.view, &conv_out, 0));
        const up_view = view(self.up_buf, &.{ packed_n * self.up, self.tf });
        try cuda.launched(zs_nemotron_speaker_relu(&up_view));
        const up_n = packed_n * self.up;
        try self.first_hidden.forward(view(self.up_buf, &.{ up_n, self.tf }), view(self.hid, &.{ up_n, self.tf }), 0);
        try cuda.launched(zs_nemotron_speaker_relu(&self.hid.view));
        try self.single_hidden.forward(view(self.hid, &.{ up_n, self.tf }), view(self.logits, &.{ up_n, s }), 0);
        try cuda.launched(zs_nemotron_speaker_sigmoid(&self.logits.view, &self.preds.view, @intCast(key_len * self.up)));
        // host: the packed_n preds and the chunk's native rows at (spkcache + fifo + lc_enc) * up
        const got = try arena.alloc(u16, up_n * s);
        try cuda.download(view(self.preds, &.{ up_n, s }), std.mem.sliceAsBytes(got));
        const packed_preds = try arena.alloc(f32, up_n * s);
        for (packed_preds, got) |*v, b| v.* = perception_file.f32Of(b);
        const core = rows - lc_enc - rc_enc;
        const start = (self.spk_n + self.fifo_n + lc_enc) * self.up;
        @memcpy(preds[0 .. core * self.up * s], packed_preds[start * s .. (start + core * self.up) * s]);
        // `streaming_update` (skipped on the last window): the FIFO takes the chunk's core rows'
        // pre-encoded embeddings and the packed_n preds it reads, downsampled x `up` back to the
        // packed_n grid (NeMo's `downsample_preds(spkcache_fifo_chunk_preds, upsample_factor)`)
        if (!last) {
            const down = try arena.alloc(f32, packed_n * s);
            for (0..packed_n) |r| {
                for (0..s) |sp| {
                    var sum: f64 = 0;
                    for (0..self.up) |e| sum += packed_preds[(r * self.up + e) * s + sp];
                    down[r * s + sp] = @floatCast(sum / @as(f64, @floatFromInt(self.up)));
                }
            }
            try self.updateState(chunk_embs.?, lc_enc, core, down, self.spk_n, self.fifo_n);
        }
    }

    /// The PEE's online loop: `mel` is raw [frames, features], including masked STFT tail frames,
    /// with `valid` valid frames. `windows` (cs, ce, core_rows,
    /// left_drop, out_row) each, and `act_out` bf16 [encoded_len, speakers] the activity the fusion
    /// reads — `downsample_preds` x `output_subsampling_factor` then `_align_diar_frames` per window.
    pub fn activity(self: *Speaker, arena: Allocator, mel: []const f32, features: usize, valid: usize, windows: []const [6]usize, encoded_len: usize, act_out: cuda.View) Error!void {
        self.spk_n = 0;
        self.fifo_n = 0;
        self.spk_compressed = false;
        self.n_sil = 0;
        @memset(self.mean_sil, 0);
        const s = self.speakers;
        const sub = self.subsampling;
        const native = try arena.alloc(f32, self.max_window * self.up * s);
        const act = try arena.alloc(u16, encoded_len * s);
        @memset(act, 0);
        for (windows, 0..) |w, i| {
            const cs, const ce, const core_rows, const left_off, const out_row = .{ w[0], w[1], w[2], w[5], w[4] };
            const rows = (ce - cs + sub - 1) / sub;
            const lc_enc = (left_off + sub / 2) / sub; // `round(left_offset / sub)`
            const rc_enc = rows - lc_enc - core_rows; // `ceil(right_offset / sub)` folded: rows = lc_enc + core + ceil(right/sub)
            try self.step(arena, mel, features, cs, ce, valid, lc_enc, rc_enc, i + 1 == windows.len, native);
            // `downsample_preds` x `output_factor` (avg-pool ceil_mode, pad frames excluded by the
            // ceil's partial window — `count_include_pad=False` pools the whole frames it gets):
            // then `_align_diar_frames` to the window's core rows on the ASR grid.
            const native_rows = rows - lc_enc - rc_enc;
            const native_frames = native_rows * self.up;
            const down_rows = (native_frames + self.out_factor - 1) / self.out_factor;
            for (0..@min(core_rows, down_rows)) |r| {
                for (0..s) |sp| {
                    var sum: f64 = 0;
                    var n: usize = 0;
                    for (0..self.out_factor) |e| {
                        const f = r * self.out_factor + e;
                        if (f < native_frames) {
                            sum += native[f * s + sp];
                            n += 1;
                        }
                    }
                    act[(out_row + r) * s + sp] = perception_file.bf16Of(@floatCast(sum / @as(f64, @floatFromInt(@max(1, n)))));
                }
            }
            // `_align_diar_frames` would pad the last row where preds ran short; the clip's native
            // preds never run short of its kept cores — zeros stand where a window's tail was
            // masked.
        }
        try cuda.upload(act_out, std.mem.sliceAsBytes(act));
    }

    /// `_fuse_diar_and_asr` on `states` bf16 [rows, asr_d]: y = asr_norm(x) + spk_kernel_scale *
    /// diar_norm(act > `activity_threshold`) @ diar_kernel (`zs_nemotron_speaker_fuse`).
    pub fn fuse(self: *const Speaker, states: cuda.View, act: cuda.View, asr_norm: *const Norm, y: cuda.View) Error!void {
        return cuda.launched(zs_nemotron_speaker_fuse(&states, &act, &asr_norm.weight.view, &asr_norm.bias.view, &self.diar_norm.weight.view, &self.diar_norm.bias.view, &self.diar_kernel.view, &y, self.spk_kernel_scale, eps));
    }

    pub fn deinit(self: *Speaker) void {
        self.inv.deinit();
        self.pre_encode.deinit();
        self.embed_norm.deinit();
        self.final_norm.deinit();
        for (self.enc) |*layer| layer.deinit();
        self.gpa.free(self.enc);
        self.encoder_proj.deinit();
        self.upsample_w.deinit();
        self.upsample_b.deinit();
        self.first_hidden.deinit();
        self.single_hidden.deinit();
        self.diar_norm.deinit();
        self.diar_kernel.deinit();
        self.pool.destroy();
        self.spkcache.deinit();
        self.fifo.deinit();
        self.gpa.free(self.spk_preds);
        self.gpa.free(self.fifo_preds);
        self.gpa.free(self.mean_sil);
        for ([_]*cuda.Buffer{ &self.input, &self.x, &self.y, &self.qkv, &self.attn, &self.ffn, &self.branch, &self.proj, &self.up_input, &self.up_buf, &self.hid, &self.logits, &self.preds, &self.tiles, &self.offsets }) |b| b.deinit();
        self.* = undefined;
    }
};
