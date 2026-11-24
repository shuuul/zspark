//! Unlimited-OCR-2's config.json (`model_type` `unlimited-ocr`, spec 147): the language model's shape from its
//! `language_config` (a DeepseekV2 decoder with plain MHA, `use_mla` false), and the DeepseekV2 defaults it omits
//! (the checkpoint's `configuration_deepseek_v2.py`: `rms_norm_eps` 1e-6, `scoring_func` softmax,
//! `norm_topk_prob` false, `routed_scaling_factor` 1.0). The vision tower's part is WS-03's. Host only.
//!
//! A top-level `sparse_attention_config` names the model: its attention is DASH (`Dash`), its RoPE page2d, read as
//! its HF code reads them (`dash_attention.py` `DashAttention.__init__`, `modeling_deepseekv2.py`
//! `configure_dash_inference`, `_sparse_base_impl.py`'s indexer and main rope). A config without it is Unlimited-
//! OCR v1's, retired (spec 170), and refused. There `rope_theta` (500,000) and the yarn entries are dead and are
//! not read; `sliding_window_size` is not read either: a row's window is `local_blocks * block` (128, the value
//! the checkpoints give it).

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const reader = @import("../../weights/checkpoint_reader.zig");
const Parser = reader.Parser;

/// Why a config was refused, in the model's own words.
pub const Diagnostic = reader.Diagnostic;

pub const Error = reader.Error;

/// The name config.json's `model_type` carries.
pub const model_types = [_][]const u8{"unlimited-ocr"};

/// The language model. Every attention head is 128 wide (hidden / heads).
pub const Config = struct {
    hidden: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    head_dim: usize,
    /// The first dense layer's MLP width.
    intermediate: usize,
    /// A routed expert's width; the shared MLP is `shared` of them fused.
    expert_width: usize,
    experts: usize,
    shared: usize,
    top: usize,
    /// Layers 0 .. dense_layers - 1 are dense (`first_k_dense_replace`).
    dense_layers: usize,
    eps: f32,
    bos: u32,
    eos: u32,
    max_positions: usize,
    /// Unlimited-OCR-2's attention, DASH (`sparse_attention_config`).
    dash: Dash,

    pub fn read(arena: Allocator, io: Io, dir: Io.Dir, diagnostic: *Diagnostic) (Error || Io.Dir.ReadFileAllocError)!Config {
        const text = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 26));
        return parse(arena, text, diagnostic);
    }

    pub fn parse(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) Error!Config {
        const p: Parser = .{ .arena = arena, .diagnostic = diagnostic, .where = "language_config" };
        const root = json.parseFromSliceLeaky(json.Value, arena, text, .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return p.fail(error.InvalidConfig, "config.json is not valid JSON ({t})", .{err}),
        };
        if (root != .object) return p.fail(error.InvalidConfig, "config.json is not a JSON object", .{});
        const model_type = if (root.object.get("model_type")) |v| (if (v == .string) v.string else "") else "";
        if (!std.mem.eql(u8, model_type, model_types[0]))
            return p.fail(error.UnsupportedModel, "config.json describes model_type \"{s}\", not Unlimited-OCR ({s})", .{ model_type, model_types[0] });
        // Unlimited-OCR-2 is told by this object alone; a config without it is the retired v1 (spec 170)
        const sparse = root.object.get("sparse_attention_config") orelse json.Value.null;
        const q: Parser = .{ .arena = arena, .diagnostic = diagnostic, .where = "sparse_attention_config" };
        if (sparse == .null) return p.fail(error.UnsupportedModel, "config.json has no sparse_attention_config: Unlimited-OCR v1 is retired; zspark serves Unlimited-OCR-2", .{});
        if (sparse != .object) return q.fail(error.InvalidConfig, "config.json: sparse_attention_config is not an object", .{});
        const value = root.object.get("language_config") orelse return p.fail(error.InvalidConfig, "config.json has no language_config", .{});
        if (value != .object) return p.fail(error.InvalidConfig, "config.json: language_config is not an object", .{});
        const t = value.object;
        if (p.flagOr(t, "use_mla", true)) return p.fail(error.UnsupportedModel, "language_config: use_mla is set; zspark's Unlimited-OCR is plain MHA", .{});
        const method = p.stringOr(t, "topk_method", "greedy");
        if (!std.mem.eql(u8, method, "greedy")) return p.fail(error.UnsupportedModel, "language_config: topk_method \"{s}\", not greedy", .{method});
        if (p.flagOr(t, "norm_topk_prob", false)) return p.fail(error.UnsupportedModel, "language_config: norm_topk_prob is set", .{});
        const scoring = p.stringOr(t, "scoring_func", "softmax");
        if (!std.mem.eql(u8, scoring, "softmax")) return p.fail(error.UnsupportedModel, "language_config: scoring_func \"{s}\", not softmax", .{scoring});
        if (p.numberOr(t, "routed_scaling_factor", 1.0) != 1.0) return p.fail(error.UnsupportedModel, "language_config: routed_scaling_factor is not 1", .{});
        const config: Config = .{
            .hidden = try p.uint(t, "hidden_size"),
            .layers = try p.uint(t, "num_hidden_layers"),
            .vocab = try p.uint(t, "vocab_size"),
            .heads = try p.uint(t, "num_attention_heads"),
            .head_dim = 0,
            .intermediate = try p.uint(t, "intermediate_size"),
            .expert_width = try p.uint(t, "moe_intermediate_size"),
            .experts = try p.uint(t, "n_routed_experts"),
            .shared = try p.uint(t, "n_shared_experts"),
            .top = try p.uint(t, "num_experts_per_tok"),
            .dense_layers = try p.uint(t, "first_k_dense_replace"),
            .eps = @floatCast(p.numberOr(t, "rms_norm_eps", 1e-6)),
            .bos = @intCast(try p.uint(t, "bos_token_id")),
            .eos = @intCast(try p.uint(t, "eos_token_id")),
            .max_positions = try p.uint(t, "max_position_embeddings"),
            .dash = undefined,
        };
        var c = config;
        if (c.heads == 0 or c.hidden % c.heads != 0) return p.fail(error.InvalidConfig, "language_config: hidden_size is not a multiple of the heads", .{});
        c.head_dim = c.hidden / c.heads;
        const kv = try p.uint(t, "num_key_value_heads");
        if (kv != c.heads) return p.fail(error.UnsupportedModel, "language_config: {d} key heads for {d} query heads; zspark's Unlimited-OCR is MHA", .{ kv, c.heads });
        if (c.head_dim != 128) return p.fail(error.UnsupportedModel, "language_config: heads of {d}, not 128", .{c.head_dim});
        if (c.top == 0 or c.top > 16 or c.experts == 0 or c.experts > 256 or c.top > c.experts) return p.fail(error.UnsupportedModel, "language_config: {d} of {d} experts", .{ c.top, c.experts });
        if (c.dense_layers > c.layers) return p.fail(error.InvalidConfig, "language_config: first_k_dense_replace out of range", .{});
        if (c.hidden % 64 != 0 or c.expert_width % 64 != 0 or c.intermediate % 64 != 0) return p.fail(error.UnsupportedModel, "language_config: widths not multiples of 64", .{});
        c.dash = try Dash.parse(q, sparse.object, c.layers, c.head_dim);
        return c;
    }

    /// The fused shared MLP's width.
    pub fn sharedWidth(self: Config) usize {
        return self.shared * self.expert_width;
    }
};

/// Unlimited-OCR-2's attention, DASH (SR-SWA): a row sees its own `local_blocks * block` keys, and up to
/// `read_blocks` remote blocks its layer's router picks for it (the heads share the pick), each of their keys
/// biased by `(log(max(p, floor)) - mean) / sigma`, p the block's 1.5-entmax weight among the top `candidates`.
pub const Dash = struct {
    /// Keys a block (`sparse_block_size`, 32).
    block: usize,
    /// The local window in blocks (`sparse_local_block`, 4): 128 keys, unbiased.
    local_blocks: usize,
    /// Remote blocks a row reads (`dash_topk_blocks`, 12): the entmax's first candidates.
    read_blocks: usize,
    /// The block scores' top the entmax takes (`sparse_topk_blocks`, 16). HF's inference takes its `infer`
    /// argument `dash_topm` (16; `modeling_unlimitedocr.py`, vLLM's `DASH_INFER_TOPM`) there, not a config key:
    /// zspark reads `sparse_topk_blocks`, the same 16.
    candidates: usize,
    /// The router's one index head's width (`sparse_index_dim`, 128).
    index_dim: usize,
    /// A layer's entmax sharpness (`dash_scale_per_layer`, else `dash_scaling_factor`): the entmax takes the
    /// block score `idx_q . k_bar / sqrt(index_dim)` times `scale / sqrt(index_dim)`.
    scale: []const f64,
    /// A layer's floor under an entmax weight before its log (`dash_floor_per_layer`, else `dash_prior_floor`).
    floor: []const f64,
    /// The bias's divisor (`dash_sigma`, 1).
    sigma: f64,
    /// How a block is summed into one key (`dash_stage0`): `head_cls`'s softmax over the block's index keys.
    stage0: Stage0,
    /// How blocks are picked (`dash_router`): entmax over the top `candidates`, the first `read_blocks` read.
    routing: Routing,
    /// `dash_image_only`, as HF's router keeps it.
    image_only: bool,
    /// `image_bucket_constraint`: a text page's rows pick among its own image page's blocks.
    image_buckets: bool,
    /// `<image>` (`image_token_id`, 128815) and `<PAGE>` (`page_token_id`, 128827).
    image_token: u32,
    page_token: u32,
    /// `per_page_image_tokens` (273).
    page_image_tokens: usize,
    /// The main attention's and the router's RoPE, one table (`sparse_main_rope_*` and `sparse_rope_*` agree).
    rope: Page2d,

    pub const Stage0 = enum { head_cls };
    pub const Routing = enum { entmax_topk_perlayer };

    fn parse(p: Parser, s: json.ObjectMap, layers: usize, head_dim: usize) Error!Dash {
        // HF builds a layer's router only with both (`modeling_deepseekv2.py` `_patched_init`)
        const stage = p.stringOr(s, "training_stage", "warmup");
        if (!p.flagOr(s, "use_sparse_attention", false) or !std.mem.eql(u8, stage, "dash"))
            return p.fail(error.UnsupportedModel, "sparse_attention_config: training_stage \"{s}\" or use_sparse_attention off; zspark reads Unlimited-OCR-2's DASH alone", .{stage});
        try everyLayer(p, s, "sparse_attention_freq", layers);
        if (s.get("sparse_inconsistency_freq") != null) try everyLayer(p, s, "sparse_inconsistency_freq", layers);
        if (try p.uint(s, "sparse_num_index_heads") != 1) return p.fail(error.UnsupportedModel, "sparse_attention_config: sparse_num_index_heads is not 1", .{});
        const stage0_name = p.stringOr(s, "dash_stage0", "indexer");
        const stage0 = std.meta.stringToEnum(Stage0, stage0_name) orelse
            return p.fail(error.UnsupportedModel, "sparse_attention_config: dash_stage0 \"{s}\", not head_cls", .{stage0_name});
        const routing_name = p.stringOr(s, "dash_router", "");
        const routing = std.meta.stringToEnum(Routing, routing_name) orelse
            return p.fail(error.UnsupportedModel, "sparse_attention_config: dash_router \"{s}\", not entmax_topk_perlayer", .{routing_name});
        const d: Dash = .{
            .block = try p.uint(s, "sparse_block_size"),
            .local_blocks = try p.uint(s, "sparse_local_block"),
            .read_blocks = try p.uint(s, "dash_topk_blocks"),
            .candidates = try p.uint(s, "sparse_topk_blocks"),
            .index_dim = try p.uint(s, "sparse_index_dim"),
            .scale = try perLayer(p, s, "dash_scale_per_layer", "dash_scaling_factor", 2.0, layers),
            .floor = try perLayer(p, s, "dash_floor_per_layer", "dash_prior_floor", 1e-2, layers),
            .sigma = p.numberOr(s, "dash_sigma", 1.0),
            .stage0 = stage0,
            .routing = routing,
            .image_only = p.flagOr(s, "dash_image_only", true),
            .image_buckets = p.flagOr(s, "image_bucket_constraint", false),
            .image_token = @intCast(try p.uint(s, "image_token_id")),
            .page_token = @intCast(try p.uint(s, "page_token_id")),
            .page_image_tokens = try p.uint(s, "per_page_image_tokens"),
            .rope = try Page2d.parse(p, s),
        };
        if (d.block == 0 or d.local_blocks == 0 or d.read_blocks == 0 or d.read_blocks > d.candidates)
            return p.fail(error.InvalidConfig, "sparse_attention_config: blocks of {d}, {d} local, {d} read of {d} candidates", .{ d.block, d.local_blocks, d.read_blocks, d.candidates });
        if (!(d.sigma > 0)) return p.fail(error.InvalidConfig, "sparse_attention_config: dash_sigma is not positive", .{});
        // the router's RoPE and the main attention's are one table only at one width (page2d's pairs are of it)
        if (d.index_dim != head_dim) return p.fail(error.UnsupportedModel, "sparse_attention_config: sparse_index_dim {d}, not the heads' {d}", .{ d.index_dim, head_dim });
        if (d.rope.page_pairs == 0 or d.rope.page_pairs >= d.index_dim / 2)
            return p.fail(error.UnsupportedModel, "sparse_attention_config: sparse_rope_page_pairs {d} outside 1 to {d}, where HF falls back to 1d RoPE on rope_theta", .{ d.rope.page_pairs, d.index_dim / 2 - 1 });
        return d;
    }
};

/// page2d RoPE (`_sparse_base_impl.py` `DeepseekV2IndexerRotary2D`): of a head's `dim / 2` pairs the first
/// `n_tok = dim / 2 - page_pairs` turn with the row's count within its page at `tok_theta^(-i / n_tok)`
/// ("resample"), the last `page_pairs` with its page's index (mod `page_mod`) at `page_theta^(-i / page_pairs)`.
pub const Page2d = struct {
    /// `sparse_rope_tok_theta` (10,000).
    tok_theta: f64,
    /// `sparse_rope_page_theta` (256).
    page_theta: f64,
    /// `sparse_rope_page_pairs` (1).
    page_pairs: usize,
    /// `sparse_rope_page_mod` (2): the page axis turns with the page's index mod this; 0 for none.
    page_mod: usize,
    /// `sparse_main_rope_tok_freq_mode` and `sparse_rope_tok_freq_mode`.
    tok_freq: TokFreq,

    pub const TokFreq = enum { resample };

    fn parse(p: Parser, s: json.ObjectMap) Error!Page2d {
        // a missing key takes HF's default, the same for the main attention's rope and the router's
        for ([_][]const u8{ "sparse_main_rope_mode", "sparse_rope_mode" }) |key| {
            const mode = p.stringOr(s, key, "1d");
            if (!std.mem.eql(u8, mode, "page2d")) return p.fail(error.UnsupportedModel, "sparse_attention_config: {s} \"{s}\", not page2d", .{ key, mode });
        }
        for ([_][2][]const u8{ .{ "sparse_main_rope_tok_freq_mode", "truncate" }, .{ "sparse_rope_tok_freq_mode", "resample" } }) |pair| {
            const mode = p.stringOr(s, pair[0], pair[1]);
            if (std.meta.stringToEnum(TokFreq, mode) == null) return p.fail(error.UnsupportedModel, "sparse_attention_config: {s} \"{s}\", not resample", .{ pair[0], mode });
        }
        if (p.numberOr(s, "sparse_rope_page_span", 0) != 0) return p.fail(error.UnsupportedModel, "sparse_attention_config: sparse_rope_page_span is not 0", .{});
        // without it HF's page2d takes the token pairs' base from rope_theta, which is dead on DASH
        const tok = s.get("sparse_rope_tok_theta") orelse
            return p.fail(error.UnsupportedModel, "sparse_attention_config has no sparse_rope_tok_theta: HF would take rope_theta, which zspark does not read on DASH", .{});
        if (tok != .integer and tok != .float) return p.fail(error.InvalidConfig, "sparse_attention_config: sparse_rope_tok_theta is not a number", .{});
        const r: Page2d = .{
            .tok_theta = p.numberOr(s, "sparse_rope_tok_theta", 0),
            .page_theta = p.numberOr(s, "sparse_rope_page_theta", 256.0),
            .page_pairs = if (s.get("sparse_rope_page_pairs") != null) try p.uint(s, "sparse_rope_page_pairs") else 8,
            .page_mod = if (s.get("sparse_rope_page_mod") != null) try p.uint(s, "sparse_rope_page_mod") else 0,
            .tok_freq = .resample,
        };
        if (!(r.tok_theta > 1 and r.page_theta > 1)) return p.fail(error.InvalidConfig, "sparse_attention_config: a page2d base is not above 1", .{});
        return r;
    }
};

/// `key`, a flag a layer, 1 on every one of `layers`.
fn everyLayer(p: Parser, s: json.ObjectMap, key: []const u8, layers: usize) Error!void {
    const value = s.get(key) orelse return p.fail(error.InvalidConfig, "sparse_attention_config has no {s}", .{key});
    if (value != .array or value.array.items.len != layers)
        return p.fail(error.InvalidConfig, "sparse_attention_config: {s} is not a list of {d} layers", .{ key, layers });
    for (value.array.items, 0..) |each, i| {
        if (each != .integer or each.integer != 1)
            return p.fail(error.UnsupportedModel, "sparse_attention_config: {s}[{d}] is not 1; zspark runs DASH on every layer, each with its own router", .{ key, i });
    }
}

/// HF's `_per_layer_val`: the list's value at the layer, else the scalar's (with DASH on every layer, a layer's
/// slot is its index).
fn perLayer(p: Parser, s: json.ObjectMap, key: []const u8, scalar: []const u8, default: f64, layers: usize) Error![]const f64 {
    const out = try p.arena.alloc(f64, layers);
    const list = s.get(key) orelse json.Value.null;
    if (list != .array or list.array.items.len == 0) {
        @memset(out, p.numberOr(s, scalar, default));
        return out;
    }
    if (list.array.items.len != layers) return p.fail(error.InvalidConfig, "sparse_attention_config: {s} has {d} values for {d} layers", .{ key, list.array.items.len, layers });
    for (out, list.array.items) |*o, each| o.* = switch (each) {
        .integer => |v| @floatFromInt(v),
        .float => |v| v,
        else => return p.fail(error.InvalidConfig, "sparse_attention_config: {s} holds a value that is not a number", .{key}),
    };
    return out;
}
