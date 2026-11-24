//! Qwen3-ASR's config.json (`model_type` `qwen3_asr`): the language model's shape from `thinker_config`'s
//! `text_config` (a Qwen3 decoder: GQA with q and k norms, a SwiGLU MLP, the embedding tied to the head) and the
//! prompt's audio ids from `thinker_config`. The audio tower's part is WS-03's. Host only.

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const modelopt = @import("../../weights/modelopt.zig");
const reader = @import("../../weights/checkpoint_reader.zig");
const Parser = reader.Parser;

/// Why a config was refused, in the model's own words.
pub const Diagnostic = reader.Diagnostic;

pub const Error = reader.Error;

/// The name config.json's `model_type` carries.
pub const model_types = [_][]const u8{"qwen3_asr"};

/// The language model.
pub const Config = struct {
    hidden: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    eps: f32,
    rope_theta: f64,
    max_positions: usize,
    /// `<|audio_pad|>`: a prompt's rows the audio embeddings take.
    audio_pad: u32,
    audio_start: u32,
    audio_end: u32,
    /// `<|im_end|>` and `<|endoftext|>`: generation_config.json's `eos_token_id`, which config.json omits.
    eos: [2]u32 = .{ 151645, 151643 },

    pub fn read(arena: Allocator, io: Io, dir: Io.Dir, diagnostic: *Diagnostic) (Error || Io.Dir.ReadFileAllocError)!Config {
        const text = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 26));
        return parse(arena, text, diagnostic);
    }

    pub fn parse(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) Error!Config {
        const p: Parser = .{ .arena = arena, .diagnostic = diagnostic };
        const root = json.parseFromSliceLeaky(json.Value, arena, text, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return p.fail(error.InvalidConfig, "config.json is not valid JSON ({t})", .{err}),
        };
        if (root != .object) return p.fail(error.InvalidConfig, "config.json is not a JSON object", .{});
        const model_type = if (root.object.get("model_type")) |v| (if (v == .string) v.string else "") else "";
        if (!std.mem.eql(u8, model_type, model_types[0]))
            return p.fail(error.UnsupportedModel, "config.json describes model_type \"{s}\", not Qwen3-ASR ({s})", .{ model_type, model_types[0] });
        const thinker = try p.object(root.object, "thinker_config");
        const t = try p.object(thinker, "text_config");
        if (!p.flagOr(t, "tie_word_embeddings", false)) return p.fail(error.UnsupportedModel, "text_config: the embedding is not tied to the head", .{});
        if (p.flagOr(t, "attention_bias", false)) return p.fail(error.UnsupportedModel, "text_config: attention_bias is set", .{});
        const act = p.stringOr(t, "hidden_act", "silu");
        if (!std.mem.eql(u8, act, "silu")) return p.fail(error.UnsupportedModel, "text_config: hidden_act \"{s}\", not silu", .{act});
        const c: Config = .{
            .hidden = try p.uint(t, "hidden_size"),
            .layers = try p.uint(t, "num_hidden_layers"),
            .vocab = try p.uint(t, "vocab_size"),
            .heads = try p.uint(t, "num_attention_heads"),
            .kv_heads = try p.uint(t, "num_key_value_heads"),
            .head_dim = try p.uint(t, "head_dim"),
            .intermediate = try p.uint(t, "intermediate_size"),
            .eps = @floatCast(p.numberOr(t, "rms_norm_eps", 1e-6)),
            .rope_theta = p.numberOr(t, "rope_theta", 1e6),
            .max_positions = try p.uint(t, "max_position_embeddings"),
            .audio_pad = @intCast(try p.uint(thinker, "audio_token_id")),
            .audio_start = @intCast(try p.uint(thinker, "audio_start_token_id")),
            .audio_end = @intCast(try p.uint(thinker, "audio_end_token_id")),
        };
        if (c.layers == 0 or c.vocab == 0 or c.heads == 0 or c.kv_heads == 0 or c.heads % c.kv_heads != 0 or c.heads / c.kv_heads > 16)
            return p.fail(error.UnsupportedModel, "text_config: {d} heads over {d} KV heads", .{ c.heads, c.kv_heads });
        if (c.head_dim != 64 and c.head_dim != 128 and c.head_dim != 256) return p.fail(error.UnsupportedModel, "text_config: head_dim {d}", .{c.head_dim});
        if (c.hidden % 64 != 0 or c.hidden > 4096 or c.intermediate % 64 != 0) return p.fail(error.UnsupportedModel, "text_config: widths not multiples of 64, or hidden above 4096", .{});
        return c;
    }

    /// config.json's `quantization_config`, the language model's linears' storage: null for the bf16 base; ModelOpt's
    /// W4A16_NVFP4 (spec 140: `shuuul/Qwen3-ASR-*-NVFP4-W4A16`, the excluded modules bf16), read as vLLM's
    /// `ModelOptQuantConfigBase.from_config` reads it (config.json's, as vLLM takes it before hf_quant_config.json).
    /// Anything else is refused: another method, NVFP4 with quantized activations (W4A4, spec 140's non-goal), a
    /// map of `quantized_layers`, or a group other than 16. Its slices are `arena`'s.
    pub fn quantization(arena: Allocator, io: Io, dir: Io.Dir, diagnostic: *Diagnostic) (Error || Io.Dir.ReadFileAllocError)!?modelopt.Uniform {
        const text = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 26));
        return parseQuantization(arena, text, diagnostic);
    }

    pub fn parseQuantization(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) Error!?modelopt.Uniform {
        const p: Parser = .{ .arena = arena, .diagnostic = diagnostic };
        const root = json.parseFromSliceLeaky(json.Value, arena, text, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return p.fail(error.InvalidConfig, "config.json is not valid JSON ({t})", .{err}),
        };
        if (root != .object) return p.fail(error.InvalidConfig, "config.json is not a JSON object", .{});
        const found = root.object.get("quantization_config") orelse return null;
        if (found != .object) return p.fail(error.InvalidConfig, "config.json: quantization_config is not an object", .{});
        const method = p.stringOr(found.object, "quant_method", "");
        if (!std.mem.eql(u8, method, "modelopt"))
            return p.fail(error.UnsupportedModel, "quantization_config: quant_method \"{s}\"; zspark reads Qwen3-ASR as bf16 or as ModelOpt's W4A16_NVFP4", .{method});
        const q = modelopt.uniform(arena, found.object) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => p.fail(error.UnsupportedModel, "quantization_config: {t}; zspark reads ModelOpt's W4A16_NVFP4 here, one quant_algo and its excluded modules", .{err}),
        };
        if (q.layer.algo != .nvfp4 or !q.layer.a16 or (q.layer.group_size orelse modelopt.nvfp4_block) != modelopt.nvfp4_block)
            return p.fail(error.UnsupportedModel, "quantization_config: {t} in groups of {?d}{s}; zspark reads W4A16_NVFP4 in groups of 16 here", .{ q.layer.algo, q.layer.group_size, if (q.layer.a16) "" else " with quantized activations" });
        return q;
    }

    /// The stacked q, k and v projection's width.
    pub fn qkvWidth(self: Config) usize {
        return (self.heads + 2 * self.kv_heads) * self.head_dim;
    }

    pub fn isEos(self: Config, token: u32) bool {
        return token == self.eos[0] or token == self.eos[1];
    }
};
