//! Qwen3-ASR's audio tower (spec 138 WS-03; 0.6B's shape spec 140 WS-09): a clip's log-mel features to the audio embeddings that stand in
//! for its `<|audio_pad|>` tokens, from the checkpoint's `thinker.audio_tower.` tensors, BF16 as stored. The
//! qwen-asr 0.0.6 package's `Qwen3ASRAudioEncoder` (transformers backend, modeling_qwen3_asr.py 523-737) with the
//! attention in 104-token blocks as FlashAttention-2 runs it (the spec's decision; the shipped SDPA path ignores
//! the blocks):
//!
//! - the mel, bf16 [128, T], cut in chunks of 100 frames, the tail zero-padded to 100 (value 0) and its padded
//!   outputs dropped after the convolutions (the package pads the tail to its longest chunk, 100 when the clip has
//!   two chunks or more; a clip under 100 frames is padded here too, where the package runs it unpadded, and
//!   its rows past the valid length are zeroed after conv1 and conv2, so they read as the package's padding);
//! - three Conv2d k3 s2 p1 with bias and exact GELU, 1 -> 480 -> 480 -> 480, (128, 100) -> (16, 13), as patches and
//!   GEMMs (`kernels/qwen3_asr/tower.cu`); `conv_out` 7680 -> d_model without bias on (time, channel x frequency);
//! - the sinusoid's rows 0..12 added a chunk; the first `audio.mel.tokens(T)` rows kept;
//! - `depth` pre-LN layers: LayerNorm (eps 1e-5), q, k and v as one GEMM, block-diagonal attention of heads of 64
//!   over blocks of 104 tokens (`tower_attention.cu`), out_proj, the residual, LayerNorm, fc1 with exact GELU,
//!   fc2, the residual;
//! - `ln_post`, `proj1` with exact GELU, `proj2` to the language model's width (`out_dim`).
//!
//! The widths and the depth are `audio_config`'s (`Shape`): 1.7B's d_model 1,024 (16 heads), 24 layers, FFN 4,096,
//! output 2,048; 0.6B's 896 (14 heads), 18 layers, FFN 3,584, output 1,024. The attention kernel is instantiated for
//! those two widths; the convolutions, the chunks and the blocks are the same in both.
//!
//! Every step depends on its chunk (the convolutions, the GEMMs' rows, the norms) or its block (the attention)
//! alone, so a block's embeddings have the same bits whatever audio stands beside it (the row rule; the test).
//! The convolutions run in groups of `conv_chunks` chunks, whose patches are the largest buffer.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const numeric = @import("../../weights/numeric.zig");
const safetensors = @import("../../weights/safetensors.zig");
const mel = @import("../../audio/mel.zig");

/// The widths and depth `audio_config` gives (`checkConfig`).
pub const Shape = struct {
    d_model: usize,
    heads: usize,
    ffn: usize,
    depth: usize,
    out_dim: usize,

    pub const qwen3_asr_1_7b: Shape = .{ .d_model = 1024, .heads = 16, .ffn = 4096, .depth = 24, .out_dim = 2048 };
    pub const qwen3_asr_0_6b: Shape = .{ .d_model = 896, .heads = 14, .ffn = 3584, .depth = 18, .out_dim = 1024 };
};

/// The widths `tower_attention.cu` is instantiated for.
pub const attention_widths = [_]usize{ 1024, 896 };
pub const head_dim = 64;
pub const bins = mel.bins;
pub const channels = 480;
pub const out_dim = 2048;
/// Mel frames a chunk (`n_window * 2`) and the tokens it gives.
pub const chunk_frames = 100;
pub const chunk_tokens = 13;
/// Tokens an attention block: `13 * (n_window_infer / 100)`.
pub const block_tokens = 104;
/// The third convolution's frequencies: 128 halved thrice.
pub const freqs = 16;
pub const conv_in = channels * freqs;
pub const eps = 1e-5;
/// conv1's K, 9 taps padded to the GEMM's 8s.
pub const conv1_k = 16;
pub const conv_k = 9 * channels;
/// Chunks a group of the convolutions by default: 64 chunks' patches are 442 MB.
pub const conv_chunks_default = 64;
/// Chunks a call by default: 1,200 s, the route's cap.
pub const max_chunks_default = 1200;

pub const Error = error{UnsupportedAudio} || cuda.Error || Allocator.Error;

/// Why a checkpoint was refused, in words for the user.
pub const Diagnostic = @import("../../weights/checkpoint_reader.zig").Diagnostic;

/// The tower's shape from audio_config; the fields the kernels take as constants checked, anything else refused in
/// words.
pub fn checkConfig(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) (error{ UnsupportedAudio, InvalidConfig } || Allocator.Error)!Shape {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch {
        diagnostic.set(arena, "config.json is not JSON", .{});
        return error.InvalidConfig;
    };
    const thinker = if (root == .object) root.object.get("thinker_config") else null;
    const audio = if (thinker) |t| (if (t == .object) t.object.get("audio_config") else null) else null;
    const config = audio orelse {
        diagnostic.set(arena, "config.json has no thinker_config.audio_config: this checkpoint has no audio tower", .{});
        return error.UnsupportedAudio;
    };
    if (config != .object) {
        diagnostic.set(arena, "config.json's audio_config is not an object", .{});
        return error.InvalidConfig;
    }
    const object = config.object;
    const expected = [_]struct { []const u8, i64 }{
        .{ "num_mel_bins", bins },         .{ "downsample_hidden_size", channels },
        .{ "n_window", chunk_frames / 2 }, .{ "n_window_infer", block_tokens / chunk_tokens * chunk_frames },
    };
    for (expected) |entry| {
        const value = object.get(entry[0]) orelse {
            diagnostic.set(arena, "audio_config has no {s}", .{entry[0]});
            return error.InvalidConfig;
        };
        if (value != .integer or value.integer != entry[1]) {
            diagnostic.set(arena, "audio_config.{s} is not {d}: zspark's tower reads Qwen3-ASR-1.7B's and 0.6B's chunks and blocks", .{ entry[0], entry[1] });
            return error.UnsupportedAudio;
        }
    }
    var widths: [5]usize = undefined;
    for ([_][]const u8{ "d_model", "encoder_attention_heads", "encoder_layers", "encoder_ffn_dim", "output_dim" }, &widths) |name, *out| {
        const value = object.get(name) orelse {
            diagnostic.set(arena, "audio_config has no {s}", .{name});
            return error.InvalidConfig;
        };
        if (value != .integer or value.integer <= 0 or value.integer > 1 << 20) {
            diagnostic.set(arena, "audio_config.{s} is not a positive integer", .{name});
            return error.InvalidConfig;
        }
        out.* = @intCast(value.integer);
    }
    const shape: Shape = .{ .d_model = widths[0], .heads = widths[1], .depth = widths[2], .ffn = widths[3], .out_dim = widths[4] };
    if (shape.heads * head_dim != shape.d_model or std.mem.indexOfScalar(usize, &attention_widths, shape.d_model) == null) {
        diagnostic.set(arena, "audio_config: d_model {d} in {d} heads; zspark's tower attention takes heads of 64 at a width of 1024 or 896 (Qwen3-ASR-1.7B, 0.6B)", .{ shape.d_model, shape.heads });
        return error.UnsupportedAudio;
    }
    if (shape.ffn % 8 != 0 or shape.out_dim % 8 != 0) {
        diagnostic.set(arena, "audio_config: encoder_ffn_dim {d} or output_dim {d} not in whole 8s", .{ shape.ffn, shape.out_dim });
        return error.UnsupportedAudio;
    }
    if (object.get("activation_function")) |value| {
        if (value != .string or !std.mem.eql(u8, value.string, "gelu")) {
            diagnostic.set(arena, "audio_config.activation_function is not gelu", .{});
            return error.UnsupportedAudio;
        }
    }
    if (object.get("scale_embedding")) |value| {
        if (value == .bool and value.bool) {
            diagnostic.set(arena, "audio_config.scale_embedding is true: the tower is read without it", .{});
            return error.UnsupportedAudio;
        }
    }
    return shape;
}

/// The epilogue's activation (`bf16/gemm_bias.cu`).
pub const Act = enum(i64) { none = 0, gelu_erf = 2 };

extern fn zs_qwen3_asr_tower_patches(x: *const cuda.View, col: *const cuda.View, time_major: i64) c_int;
extern fn zs_qwen3_asr_tower_bias_gelu(x: *const cuda.View, bias: *const cuda.View) c_int;
extern fn zs_qwen3_asr_tower_positions(x: *const cuda.View, table: *const cuda.View) c_int;
extern fn zs_qwen3_asr_tower_attention(qkv: *const cuda.View, tiles: *const cuda.View, out: *const cuda.View) c_int;

/// A BF16 linear with its bias: weight [n, k], bias [n].
pub const Linear = struct {
    weight: cuda.Buffer,
    bias: cuda.Buffer,

    /// out (m, n) = act(x (m, k) W^T + bias), the bias added to the fp32 sum before one rounding; a row's bits are
    /// its own (`gemm_bias.cu`: one tile, no K split).
    pub fn forward(self: Linear, x: cuda.View, out: cuda.View, act: Act) cuda.Error!void {
        return gemm(x, self.weight.view, self.bias.view, out, act);
    }

    fn deinit(self: *Linear) void {
        self.weight.deinit();
        self.bias.deinit();
    }
};

pub fn gemm(x: cuda.View, w: cuda.View, bias: cuda.View, out: cuda.View, act: Act) cuda.Error!void {
    return cuda.launched(cuda.kernels.zs_b16_gemm_bias(&x, &w, &bias, &out, @backingInt(act)));
}

/// A LayerNorm's weight and bias, bf16 [d_model].
pub const Norm = struct {
    weight: cuda.Buffer,
    bias: cuda.Buffer,

    fn deinit(self: *Norm) void {
        self.weight.deinit();
        self.bias.deinit();
    }
};

/// h = bf(h + branch) when `branch`, then y = LayerNorm(h) when `y` (`common/layernorm.cu`).
pub fn norm(h: cuda.View, branch: ?cuda.View, by: ?Norm, y: ?cuda.View) cuda.Error!void {
    const b: ?*const cuda.View = if (branch) |*v| v else null;
    const w: ?*const cuda.View = if (by) |*n| &n.weight.view else null;
    const bi: ?*const cuda.View = if (by) |*n| &n.bias.view else null;
    const yv: ?*const cuda.View = if (y) |*v| v else null;
    return cuda.launched(cuda.kernels.zs_layernorm(&h, b, w, bi, yv, eps));
}

pub const Layer = struct {
    norm1: Norm,
    /// q, k and v stacked: weight [3 d_model, d_model], bias [3 d_model]
    qkv: Linear,
    out: Linear,
    norm2: Norm,
    fc1: Linear,
    fc2: Linear,

    fn deinit(self: *Layer) void {
        self.norm1.deinit();
        self.qkv.deinit();
        self.out.deinit();
        self.norm2.deinit();
        self.fc1.deinit();
        self.fc2.deinit();
    }
};

/// The attention's tile size in queries: a block of 104 tokens is one tile.
pub const attention_tile = 128;

/// The tokens of a clip of `frames` mel frames: `_get_feat_extract_output_lengths`.
pub fn tokens(frames: usize) usize {
    return mel.tokens(frames);
}

/// The chunks of a clip of `frames` mel frames, the tail included.
pub fn chunkCount(frames: usize) usize {
    return (frames + chunk_frames - 1) / chunk_frames;
}

/// The attention's tiles for `n` tokens: blocks of 104 from the first, the last the remainder (`cu_seqlens`),
/// each cut in tiles of 128 queries from its start: {first query, block start, block length, 0}.
pub fn tileCount(n: usize) usize {
    var count: usize = 0;
    var start: usize = 0;
    while (start < n) : (start += block_tokens) count += (@as(usize, @min(block_tokens, n - start)) + attention_tile - 1) / attention_tile;
    return count;
}

pub fn writeTiles(n: usize, tiles: [][4]i32) void {
    var next: usize = 0;
    var start: usize = 0;
    while (start < n) : (start += block_tokens) {
        const len: usize = @min(block_tokens, n - start);
        var q: usize = 0;
        while (q < len) : (q += attention_tile) {
            tiles[next] = .{ @intCast(start + q), @intCast(start), @intCast(len), 0 };
            next += 1;
        }
    }
}

/// `SinusoidsPositionEmbedding(1500, d_model)`'s rows 0..12 as bf16 into `out` [13, d_model]: sin of
/// t / 10000^(j / (half - 1)) for j < half = d_model / 2, then cos, computed in f32 as PyTorch does (the increment
/// rounded to f32, `exp`, the product), then rounded.
pub fn positionTable(d_model: usize, out: []u16) void {
    std.debug.assert(out.len == chunk_tokens * d_model);
    const half = d_model / 2;
    const increment: f32 = @floatCast(@log(@as(f64, 10000.0)) / @as(f64, @floatFromInt(half - 1)));
    for (0..chunk_tokens) |t| {
        for (0..half) |j| {
            const inv: f32 = @exp(-increment * @as(f32, @floatFromInt(j)));
            const scaled: f32 = @as(f32, @floatFromInt(t)) * inv;
            out[t * d_model + j] = numeric.bf16FromF32(@sin(scaled));
            out[t * d_model + half + j] = numeric.bf16FromF32(@cos(scaled));
        }
    }
}

fn view(buffer: cuda.Buffer, shape: []const usize) cuda.View {
    var count: usize = 2;
    for (shape) |dim| count *= dim;
    std.debug.assert(count <= buffer.bytes());
    return .init(buffer.view.ptr.?, .bf16, shape);
}

fn at(base: cuda.View, offset_elements: usize, shape: []const usize) cuda.View {
    return .init(@ptrCast(@as([*]u8, @ptrCast(base.ptr.?)) + offset_elements * 2), .bf16, shape);
}

pub const Tower = struct {
    conv1: Linear,
    conv2: Linear,
    conv3: Linear,
    /// [d_model, 7680], its columns reordered from PyTorch's (channel, frequency) to (frequency, channel)
    conv_out: cuda.Buffer,
    zero_bias: cuda.Buffer,
    /// bf16 [13, d_model] (`positionTable`)
    table: cuda.Buffer,
    shape: Shape,
    /// `shape.depth` of them, `gpa`'s
    layers: []Layer,
    gpa: Allocator,
    ln_post: Norm,
    proj1: Linear,
    proj2: Linear,
    max_chunks: usize,
    conv_chunks: usize,

    /// The clip's chunks, bf16 [max_chunks, 128, 100], the tail padded with zeros.
    input: cuda.Buffer,
    /// A conv group's patches: [conv_chunks * 800, 4320] at most.
    col: cuda.Buffer,
    /// A conv group's activations, channels last: [g, 64, 50, 480], [g, 32, 25, 480], [g, 13, 16, 480].
    a1: cuda.Buffer,
    a2: cuda.Buffer,
    a3: cuda.Buffer,
    /// The run's rows, `max_chunks * 13` each: the residual stream, a norm's output (and the attention's), the
    /// wide one (qkv, fc1, proj1), a branch (out_proj, fc2).
    h: cuda.Buffer,
    y: cuda.Buffer,
    wide: cuda.Buffer,
    branch: cuda.Buffer,
    tiles: cuda.Buffer,
    /// The current clip (`begin`): its chunks, tokens and tiles.
    chunks: usize = 0,
    tokens: usize = 0,
    tile_count: usize = 0,
    /// A lone chunk under 100 frames: its frames (0 otherwise), whose convolutions' rows past them are zeroed.
    short: usize = 0,
    /// `branch` holds a residual not yet added to `h`.
    pending: bool = false,

    /// Uploads the tower of `shape` (`checkConfig`) from `checkpoint`'s `thinker.audio_tower.` tensors (BF16, the
    /// shapes checked), for clips of at most `max_chunks` chunks, the convolutions `conv_chunks` chunks at a time.
    pub fn load(gpa: Allocator, arena: Allocator, checkpoint: *const safetensors.Checkpoint, shape: Shape, max_chunks: usize, conv_chunks: usize, diagnostic: *Diagnostic) Error!Tower {
        std.debug.assert(max_chunks >= 1 and conv_chunks >= 1);
        try check(arena, checkpoint, shape, diagnostic);
        const d_model = shape.d_model;
        const ffn = shape.ffn;
        var ranges: std.ArrayList([]const u8) = .empty;
        defer ranges.deinit(gpa);
        var names = Names{ .shape = shape };
        while (names.next()) |entry| try ranges.append(gpa, checkpoint.get(entry.name).?.data);
        var held = try checkpoint.hold(gpa, ranges.items);
        defer held.release();

        var self: Tower = undefined;
        self.shape = shape;
        self.gpa = gpa;
        self.max_chunks = max_chunks;
        self.conv_chunks = conv_chunks;
        self.chunks = 0;
        self.tokens = 0;
        self.tile_count = 0;
        self.pending = false;
        var made: Made = .{ .gpa = gpa };
        errdefer made.free();
        self.layers = try gpa.alloc(Layer, shape.depth);
        errdefer gpa.free(self.layers);

        // conv1 [480, 1, 3, 3] -> [480, 16], the taps then zeros
        {
            const w = try gpa.alloc(u16, channels * conv1_k);
            defer gpa.free(w);
            @memset(w, 0);
            const stored = bits(checkpoint, "thinker.audio_tower.conv2d1.weight");
            for (0..channels) |o| @memcpy(w[o * conv1_k ..][0..9], stored[o * 9 ..][0..9]);
            self.conv1 = .{ .weight = try made.upload(&.{ channels, conv1_k }, std.mem.sliceAsBytes(w)), .bias = try tensor(checkpoint, "thinker.audio_tower.conv2d1.bias", &.{channels}, &made) };
        }
        // conv2 and conv3 [480, 480, 3, 3] -> [480, (kf, kt, c)]
        inline for (.{ "2", "3" }, 0..) |which, i| {
            const w = try gpa.alloc(u16, channels * conv_k);
            defer gpa.free(w);
            const stored = bits(checkpoint, "thinker.audio_tower.conv2d" ++ which ++ ".weight");
            for (0..channels) |o| {
                for (0..channels) |c| {
                    for (0..9) |tap| w[o * conv_k + tap * channels + c] = stored[(o * channels + c) * 9 + tap];
                }
            }
            const linear: Linear = .{ .weight = try made.upload(&.{ channels, conv_k }, std.mem.sliceAsBytes(w)), .bias = try tensor(checkpoint, "thinker.audio_tower.conv2d" ++ which ++ ".bias", &.{channels}, &made) };
            if (i == 0) self.conv2 = linear else self.conv3 = linear;
        }
        // conv_out [1024, (c, f)] -> [1024, (f, c)]
        {
            const w = try gpa.alloc(u16, d_model * conv_in);
            defer gpa.free(w);
            const stored = bits(checkpoint, "thinker.audio_tower.conv_out.weight");
            for (0..d_model) |o| {
                for (0..channels) |c| {
                    for (0..freqs) |f| w[o * conv_in + f * channels + c] = stored[o * conv_in + c * freqs + f];
                }
            }
            self.conv_out = try made.upload(&.{ d_model, conv_in }, std.mem.sliceAsBytes(w));
        }
        self.zero_bias = try made.buffer(&.{d_model});
        try self.zero_bias.zero();
        {
            const t = try gpa.alloc(u16, chunk_tokens * d_model);
            defer gpa.free(t);
            positionTable(d_model, t);
            self.table = try made.upload(&.{ chunk_tokens, d_model }, std.mem.sliceAsBytes(t));
        }
        for (self.layers, 0..) |*each, i| {
            var buffer: [64]u8 = undefined;
            const p = std.fmt.bufPrint(&buffer, "thinker.audio_tower.layers.{d}.", .{i}) catch unreachable;
            each.norm1 = try normOf(checkpoint, p, "self_attn_layer_norm", d_model, &made);
            each.qkv = .{ .weight = try made.buffer(&.{ 3 * d_model, d_model }), .bias = try made.buffer(&.{3 * d_model}) };
            for ([_][]const u8{ "q_proj", "k_proj", "v_proj" }, 0..) |part, j| {
                var nb: [96]u8 = undefined;
                try each.qkv.weight.uploadAt(j * d_model * d_model * 2, checkpoint.get(std.fmt.bufPrint(&nb, "{s}self_attn.{s}.weight", .{ p, part }) catch unreachable).?.data);
                try each.qkv.bias.uploadAt(j * d_model * 2, checkpoint.get(std.fmt.bufPrint(&nb, "{s}self_attn.{s}.bias", .{ p, part }) catch unreachable).?.data);
            }
            each.out = try linearOf(checkpoint, p, "self_attn.out_proj", d_model, d_model, &made);
            each.norm2 = try normOf(checkpoint, p, "final_layer_norm", d_model, &made);
            each.fc1 = try linearOf(checkpoint, p, "fc1", ffn, d_model, &made);
            each.fc2 = try linearOf(checkpoint, p, "fc2", d_model, ffn, &made);
        }
        self.ln_post = try normOf(checkpoint, "thinker.audio_tower.", "ln_post", d_model, &made);
        self.proj1 = try linearOf(checkpoint, "thinker.audio_tower.", "proj1", d_model, d_model, &made);
        self.proj2 = try linearOf(checkpoint, "thinker.audio_tower.", "proj2", shape.out_dim, d_model, &made);

        const g = conv_chunks;
        const rows = max_chunks * chunk_tokens;
        self.input = try made.buffer(&.{ max_chunks, bins, chunk_frames });
        self.col = try made.buffer(&.{ g * 32 * 25, conv_k });
        self.a1 = try made.buffer(&.{ g * 64 * 50, channels });
        self.a2 = try made.buffer(&.{ g * 32 * 25, channels });
        self.a3 = try made.buffer(&.{ g * freqs * chunk_tokens, channels });
        self.h = try made.buffer(&.{ rows, d_model });
        self.y = try made.buffer(&.{ rows, d_model });
        self.wide = try made.buffer(&.{ rows, ffn });
        self.branch = try made.buffer(&.{ rows, d_model });
        const tile_buffer: cuda.Buffer = try .init(.i32, &.{ tileCount(rows), 4 });
        try made.add(tile_buffer);
        self.tiles = tile_buffer;
        made.list.deinit(gpa);
        return self;
    }

    pub fn deinit(self: *Tower) void {
        self.conv1.deinit();
        self.conv2.deinit();
        self.conv3.deinit();
        for (self.layers) |*each| each.deinit();
        self.gpa.free(self.layers);
        self.ln_post.deinit();
        self.proj1.deinit();
        self.proj2.deinit();
        for ([_]*cuda.Buffer{ &self.conv_out, &self.zero_bias, &self.table, &self.input, &self.col, &self.a1, &self.a2, &self.a3, &self.h, &self.y, &self.wide, &self.branch, &self.tiles }) |buffer| buffer.deinit();
        self.* = undefined;
    }

    /// The embeddings of a clip: `features` its mel, bf16 bits [128, frames] bin-major (the extractor's f32
    /// rounded to bf16, as the package feeds the tower), into `out` bf16 [tokens(frames), out_dim]. `gpa` holds the
    /// chunks and tiles for the call.
    pub fn encode(self: *Tower, gpa: Allocator, features: []const u16, frames: usize, out: cuda.View) Error!void {
        try self.begin(gpa, features, frames);
        try self.convs();
        try self.positions();
        for (0..self.shape.depth) |i| try self.layer(i);
        try self.post(out);
    }

    /// Starts a clip: its chunks uploaded (the tail padded with zeros to 100 frames) and its attention tiles.
    pub fn begin(self: *Tower, gpa: Allocator, features: []const u16, frames: usize) Error!void {
        const chunks = chunkCount(frames);
        if (frames == 0 or chunks > self.max_chunks) return error.UnsupportedAudio;
        try cuda.bound(features.len == bins * frames, "audio tower: {d} features for {d} frames", .{ features.len, frames });
        const chunked = try gpa.alloc(u16, chunks * bins * chunk_frames);
        defer gpa.free(chunked);
        @memset(chunked, 0);
        for (0..chunks) |c| {
            const first = c * chunk_frames;
            const len = @min(chunk_frames, frames - first);
            for (0..bins) |b| @memcpy(chunked[(c * bins + b) * chunk_frames ..][0..len], features[b * frames + first ..][0..len]);
        }
        try self.input.uploadAt(0, std.mem.sliceAsBytes(chunked));
        const n = tokens(frames);
        const count = tileCount(n);
        const tiles = try gpa.alloc([4]i32, count);
        defer gpa.free(tiles);
        writeTiles(n, tiles);
        try self.tiles.uploadAt(0, std.mem.sliceAsBytes(tiles));
        self.chunks = chunks;
        self.tokens = n;
        self.tile_count = count;
        self.short = if (chunks == 1 and frames < chunk_frames) frames else 0;
        self.pending = false;
    }

    /// The convolutions and `conv_out` over the clip's chunks, `conv_chunks` at a time, into `h`'s first
    /// `chunks * 13` rows (the tail's padded rows last). `a1`, `a2` and `a3` keep the last group's activations.
    pub fn convs(self: *Tower) Error!void {
        var first: usize = 0;
        while (first < self.chunks) : (first += self.conv_chunks) {
            const g = @min(self.conv_chunks, self.chunks - first);
            const x = at(self.input.view, first * bins * chunk_frames, &.{ g, bins, chunk_frames, 1 });
            try self.conv(x, view(self.col, &.{ g * 64 * 50, conv1_k }), self.conv1, view(self.a1, &.{ g * 64 * 50, channels }), false);
            const l1 = (self.short + 1) / 2;
            if (self.short > 0) try zeroPast(self.a1, 64, 50, l1);
            try self.conv(view(self.a1, &.{ g, 64, 50, channels }), view(self.col, &.{ g * 32 * 25, conv_k }), self.conv2, view(self.a2, &.{ g * 32 * 25, channels }), false);
            if (self.short > 0) try zeroPast(self.a2, 32, 25, (l1 + 1) / 2);
            try self.conv(view(self.a2, &.{ g, 32, 25, channels }), view(self.col, &.{ g * chunk_tokens * freqs, conv_k }), self.conv3, view(self.a3, &.{ g * chunk_tokens * freqs, channels }), true);
            const rows = g * chunk_tokens;
            try gemm(view(self.a3, &.{ rows, conv_in }), self.conv_out.view, self.zero_bias.view, self.h.span(first * chunk_tokens, rows), .none);
        }
        self.pending = false;
    }

    /// The rows of the first chunk's activations (frequency, time, channel) at a time `valid` or later set to zero:
    /// what the next convolution's padding reads where the package runs a short clip unpadded.
    fn zeroPast(a: cuda.Buffer, f_n: usize, t_n: usize, valid: usize) Error!void {
        if (valid >= t_n) return;
        for (0..f_n) |f| try cuda.zeroLaunched(at(a.view, (f * t_n + valid) * channels, &.{(t_n - valid) * channels}));
    }

    fn conv(self: *Tower, x: cuda.View, col: cuda.View, by: Linear, out: cuda.View, time_major: bool) Error!void {
        try cuda.launched(zs_qwen3_asr_tower_patches(&x, &col, @intFromBool(time_major)));
        // the product rounded, then its bias and GELU (PyTorch's cuDNN path: the bias is a second op)
        try gemm(col, by.weight.view, at(self.zero_bias.view, 0, &.{channels}), out, .none);
        try cuda.launched(zs_qwen3_asr_tower_bias_gelu(&out, &by.bias.view));
    }

    /// The sinusoid's rows added to `h`'s rows, a chunk's 13 from 0.
    pub fn positions(self: *Tower) Error!void {
        const x = self.h.rows(self.chunks * chunk_tokens);
        try cuda.launched(zs_qwen3_asr_tower_positions(&x, &self.table.view));
    }

    /// Layer `i` on the clip's tokens; its FFN's residual is left in `branch` for the next norm (`pending`).
    pub fn layer(self: *Tower, i: usize) Error!void {
        const n = self.tokens;
        const l = self.layers[i];
        const h = self.h.rows(n);
        const y = self.y.rows(n);
        const branch = self.branch.rows(n);
        const qkv = view(self.wide, &.{ n, 3 * self.shape.d_model });
        try norm(h, if (self.pending) branch else null, l.norm1, y);
        try l.qkv.forward(y, qkv, .none);
        const tiles = self.tiles.rows(self.tile_count);
        try cuda.launched(zs_qwen3_asr_tower_attention(&qkv, &tiles, &y));
        try l.out.forward(y, branch, .none);
        try norm(h, branch, l.norm2, y);
        const up = view(self.wide, &.{ n, self.shape.ffn });
        try l.fc1.forward(y, up, .gelu_erf);
        try l.fc2.forward(up, branch, .none);
        self.pending = true;
    }

    /// Adds a pending residual to `h`: the residual stream after the last layer run, for a reader of `h`.
    pub fn settle(self: *Tower) Error!void {
        if (!self.pending) return;
        try norm(self.h.rows(self.tokens), self.branch.rows(self.tokens), null, null);
        self.pending = false;
    }

    /// `ln_post` into `y`, `proj1` and GELU into `wide`, `proj2` into `out` bf16 [tokens, out_dim].
    pub fn post(self: *Tower, out: cuda.View) Error!void {
        const n = self.tokens;
        try cuda.bound(out.dtype == .bf16 and out.count() == n * self.shape.out_dim, "audio tower: embeddings for {d} tokens", .{n});
        try norm(self.h.rows(n), if (self.pending) self.branch.rows(n) else null, self.ln_post, self.y.rows(n));
        self.pending = false;
        const up = view(self.wide, &.{ n, self.shape.d_model });
        try self.proj1.forward(self.y.rows(n), up, .gelu_erf);
        try self.proj2.forward(up, out, .none);
    }

    /// The tower's tensors: names and shapes, all BF16.
    const Names = struct {
        shape: Shape,
        index: usize = 0,
        buffer: [96]u8 = undefined,
        part: [64]u8 = undefined,

        const Entry = struct { name: []const u8, shape: [4]u64, ndim: usize };
        /// A dimension: a number, or the shape's width it stands for.
        const Dim = union(enum) { n: u64, d_model, ffn, out_dim };
        const fixed = [_]struct { []const u8, []const Dim }{
            .{ "conv2d1.weight", &.{ .{ .n = channels }, .{ .n = 1 }, .{ .n = 3 }, .{ .n = 3 } } },
            .{ "conv2d1.bias", &.{.{ .n = channels }} },
            .{ "conv2d2.weight", &.{ .{ .n = channels }, .{ .n = channels }, .{ .n = 3 }, .{ .n = 3 } } },
            .{ "conv2d2.bias", &.{.{ .n = channels }} },
            .{ "conv2d3.weight", &.{ .{ .n = channels }, .{ .n = channels }, .{ .n = 3 }, .{ .n = 3 } } },
            .{ "conv2d3.bias", &.{.{ .n = channels }} },
            .{ "conv_out.weight", &.{ .d_model, .{ .n = conv_in } } },
            .{ "ln_post.weight", &.{.d_model} },
            .{ "ln_post.bias", &.{.d_model} },
            .{ "proj1.weight", &.{ .d_model, .d_model } },
            .{ "proj1.bias", &.{.d_model} },
            .{ "proj2.weight", &.{ .out_dim, .d_model } },
            .{ "proj2.bias", &.{.out_dim} },
        };
        const per_layer = [_]struct { []const u8, []const Dim }{
            .{ "self_attn_layer_norm.weight", &.{.d_model} },
            .{ "self_attn_layer_norm.bias", &.{.d_model} },
            .{ "self_attn.q_proj.weight", &.{ .d_model, .d_model } },
            .{ "self_attn.q_proj.bias", &.{.d_model} },
            .{ "self_attn.k_proj.weight", &.{ .d_model, .d_model } },
            .{ "self_attn.k_proj.bias", &.{.d_model} },
            .{ "self_attn.v_proj.weight", &.{ .d_model, .d_model } },
            .{ "self_attn.v_proj.bias", &.{.d_model} },
            .{ "self_attn.out_proj.weight", &.{ .d_model, .d_model } },
            .{ "self_attn.out_proj.bias", &.{.d_model} },
            .{ "final_layer_norm.weight", &.{.d_model} },
            .{ "final_layer_norm.bias", &.{.d_model} },
            .{ "fc1.weight", &.{ .ffn, .d_model } },
            .{ "fc1.bias", &.{.ffn} },
            .{ "fc2.weight", &.{ .d_model, .ffn } },
            .{ "fc2.bias", &.{.d_model} },
        };

        fn count(self: Names) usize {
            return fixed.len + self.shape.depth * per_layer.len;
        }

        fn next(self: *Names) ?Entry {
            const i = self.index;
            if (i >= self.count()) return null;
            self.index += 1;
            const part, const dims = if (i < fixed.len) .{ fixed[i][0], fixed[i][1] } else blk: {
                const j = i - fixed.len;
                const each = per_layer[j % per_layer.len];
                break :blk .{ std.fmt.bufPrint(&self.part, "layers.{d}.{s}", .{ j / per_layer.len, each[0] }) catch unreachable, each[1] };
            };
            const name = std.fmt.bufPrint(&self.buffer, "thinker.audio_tower.{s}", .{part}) catch unreachable;
            var entry: Entry = .{ .name = name, .shape = @splat(0), .ndim = dims.len };
            for (dims, entry.shape[0..dims.len]) |dim, *out| out.* = switch (dim) {
                .n => |v| v,
                .d_model => self.shape.d_model,
                .ffn => self.shape.ffn,
                .out_dim => self.shape.out_dim,
            };
            return entry;
        }
    };

    fn check(arena: Allocator, checkpoint: *const safetensors.Checkpoint, shape: Shape, diagnostic: *Diagnostic) error{UnsupportedAudio}!void {
        var names = Names{ .shape = shape };
        while (names.next()) |entry| {
            const found = checkpoint.get(entry.name) orelse {
                diagnostic.set(arena, "the audio tower lacks {s}: no audio can be read with this checkpoint", .{entry.name});
                return error.UnsupportedAudio;
            };
            if (found.dtype != .BF16) {
                diagnostic.set(arena, "the audio tower's {s} is {t}, not BF16: zspark runs the tower as stored, BF16", .{ entry.name, found.dtype });
                return error.UnsupportedAudio;
            }
            if (!std.mem.eql(u64, found.shape, entry.shape[0..entry.ndim])) {
                diagnostic.set(arena, "the audio tower's {s} has shape {any}, not {any}", .{ entry.name, found.shape, entry.shape[0..entry.ndim] });
                return error.UnsupportedAudio;
            }
        }
    }

    /// What a load has made so far, freed in reverse if a later step fails.
    const Made = struct {
        gpa: Allocator,
        list: std.ArrayList(cuda.Buffer) = .empty,

        fn add(self: *Made, made: cuda.Buffer) Allocator.Error!void {
            errdefer {
                var lost = made;
                lost.deinit();
            }
            try self.list.append(self.gpa, made);
        }

        fn buffer(self: *Made, shape: []const usize) (cuda.Error || Allocator.Error)!cuda.Buffer {
            const made: cuda.Buffer = try .init(.bf16, shape);
            try self.add(made);
            return made;
        }

        fn upload(self: *Made, shape: []const usize, data: []const u8) (cuda.Error || Allocator.Error)!cuda.Buffer {
            const made = try self.buffer(shape);
            try made.upload(data);
            return made;
        }

        fn free(self: *Made) void {
            var i = self.list.items.len;
            while (i > 0) {
                i -= 1;
                self.list.items[i].deinit();
            }
            self.list.deinit(self.gpa);
        }
    };

    /// A stored BF16 tensor's bits (checked by `check`).
    fn bits(checkpoint: *const safetensors.Checkpoint, name: []const u8) []align(1) const u16 {
        return std.mem.bytesAsSlice(u16, checkpoint.get(name).?.data);
    }

    fn tensor(checkpoint: *const safetensors.Checkpoint, name: []const u8, shape: []const usize, made: *Made) (cuda.Error || Allocator.Error)!cuda.Buffer {
        return made.upload(shape, checkpoint.get(name).?.data);
    }

    fn linearOf(checkpoint: *const safetensors.Checkpoint, p: []const u8, part: []const u8, n: usize, k: usize, made: *Made) (cuda.Error || Allocator.Error)!Linear {
        var w: [96]u8 = undefined;
        var b: [96]u8 = undefined;
        return .{
            .weight = try tensor(checkpoint, std.fmt.bufPrint(&w, "{s}{s}.weight", .{ p, part }) catch unreachable, &.{ n, k }, made),
            .bias = try tensor(checkpoint, std.fmt.bufPrint(&b, "{s}{s}.bias", .{ p, part }) catch unreachable, &.{n}, made),
        };
    }

    fn normOf(checkpoint: *const safetensors.Checkpoint, p: []const u8, part: []const u8, d_model: usize, made: *Made) (cuda.Error || Allocator.Error)!Norm {
        var w: [96]u8 = undefined;
        var b: [96]u8 = undefined;
        return .{
            .weight = try tensor(checkpoint, std.fmt.bufPrint(&w, "{s}{s}.weight", .{ p, part }) catch unreachable, &.{d_model}, made),
            .bias = try tensor(checkpoint, std.fmt.bufPrint(&b, "{s}{s}.bias", .{ p, part }) catch unreachable, &.{d_model}, made),
        };
    }
};
