//! A random GLM under the checkpoint's names, no file: five layers of both attentions and both MLPs and the MTP
//! layer after them, at sizes the kernels take, stored as FP4-Local stores its experts: NVFP4 with plain scales and
//! an `input_scale` (W4A4), the MTP layer's W4A16_NVFP4, every other linear BF16 and the norms FP4-Local's F32. The tests' model (`testing.zig`'s
//! `Fixture`) and `zspark ranks-check random`'s, where both ranks make it from one seed.

const std = @import("std");
const Allocator = std.mem.Allocator;
const numeric = @import("../../../weights/numeric.zig");
const config_file = @import("../layers/config.zig");
// a four-layer `config.json` whose sizes `fx.config` replaces, its quantization map for these layers
// (`quantOf`), and a random NVFP4 linear under a name (`putNvfp4`)
const fixtures = @import("fixtures.zig");
const dsa_file = @import("../layers/dsa.zig");
const host = @import("../layers/glue_ref.zig");
const kda_file = @import("../layers/kda.zig");
const layout = @import("../layers/layout.zig");

/// Tensors by name, as a checkpoint answers `get`.
pub const Source = fixtures.Source;

pub const fx = struct {
    pub const dims = 512;
    pub const vocab = 64;
    pub const kda_heads = 4;
    pub const heads = 4;
    pub const key_dim = 64;
    pub const value_dim = 64;
    pub const q_rank = 64;
    pub const intermediate = 1024;
    pub const moe_width = 256;
    pub const experts = 12;
    pub const top = 3;
    pub const kinds = [_]config_file.LayerType{ .linear_attention, .linear_attention, .deepseek_sparse_attention, .linear_attention, .deepseek_sparse_attention };
    pub const mlps = [_]config_file.MlpType{ .dense, .sparse, .sparse, .dense, .sparse };
    /// The most rows of a window: past the 64 from which KDA's chain is the wide one.
    pub const max_rows = 80;

    /// The fixture's config with this model's quantization map: its MoE layers' experts NVFP4, the MTP's W4A16.
    pub const text = fixtures.config_head ++ fixtures.config_rest ++ fixtures.quantOf(&.{ "1", "2", "4" }, "5");

    pub fn config(arena: Allocator) !config_file.Config {
        var diagnostic: config_file.Diagnostic = .{};
        var c = try config_file.Config.parse(arena, text, &diagnostic);
        c.hidden = dims;
        c.vocab = vocab;
        c.layers = kinds.len;
        c.layer_types = &kinds;
        c.mlp_types = &mlps;
        c.kda_heads = kda_heads;
        c.heads = heads;
        c.key_dim = key_dim;
        c.value_dim = value_dim;
        c.q_rank = q_rank;
        c.intermediate = intermediate;
        c.moe_width = moe_width;
        c.experts = experts;
        c.top_k = top;
        c.swiglu_limit = 3.0;
        return c;
    }

    const Putter = struct {
        arena: Allocator,
        random: std.Random,
        source: *Source,
        /// What the names below are under.
        under: []const u8 = "",

        fn name(self: Putter, leaf: []const u8) ![]const u8 {
            return std.mem.concat(self.arena, u8, &.{ self.under, leaf });
        }

        fn bf16(self: Putter, leaf: []const u8, shape: []const u64, spread: f32, mean: f32) !void {
            var count: usize = 1;
            for (shape) |dim| count *= @intCast(dim);
            const values = try self.arena.alloc(u16, count);
            for (values) |*one| one.* = numeric.bf16FromF32(mean + self.random.floatNorm(f32) * spread);
            try self.source.put(self.arena, try self.name(leaf), .{ .dtype = .BF16, .shape = try self.arena.dupe(u64, shape), .data = std.mem.sliceAsBytes(values) });
        }

        fn f32s(self: Putter, leaf: []const u8, count: usize, spread: f32, mean: f32) !void {
            const values = try self.arena.alloc(f32, count);
            for (values) |*one| one.* = mean + self.random.floatNorm(f32) * spread;
            try self.source.put(self.arena, try self.name(leaf), .{ .dtype = .F32, .shape = try self.arena.dupe(u64, &.{count}), .data = std.mem.sliceAsBytes(values) });
        }

        fn kda(p: Putter) !void {
            const inner = kda_heads * kda_file.head_dim;
            for ([_][]const u8{ "q", "k", "v" }) |one| {
                try p.bf16(try std.fmt.allocPrint(p.arena, "self_attn.{s}_proj.weight", .{one}), &.{ inner, dims }, 0.06, 0);
                try p.bf16(try std.fmt.allocPrint(p.arena, "self_attn.{s}_conv1d.weight", .{one}), &.{ inner, 1, kda_file.window + 1 }, 0.6, 0);
            }
            try p.bf16("self_attn.f_a_proj.weight", &.{ kda_file.head_dim, dims }, 0.06, 0);
            try p.bf16("self_attn.g_a_proj.weight", &.{ kda_file.head_dim, dims }, 0.06, 0);
            try p.bf16("self_attn.f_b_proj.weight", &.{ inner, kda_file.head_dim }, 0.1, 0);
            try p.bf16("self_attn.g_b_proj.weight", &.{ inner, kda_file.head_dim }, 0.1, 0);
            try p.bf16("self_attn.b_proj.weight", &.{ kda_heads, dims }, 0.06, 0);
            try p.bf16("self_attn.o_proj.weight", &.{ dims, inner }, 0.05, 0);
            try p.bf16("self_attn.o_norm.weight", &.{kda_file.head_dim}, 0.2, 1);
            try p.f32s("self_attn.A_log", kda_heads, 0.5, 0);
            try p.f32s("self_attn.dt_bias", inner, 0.5, 0);
        }

        fn dsa(p: Putter) !void {
            try p.bf16("self_attn.q_a_proj.weight", &.{ q_rank, dims }, 0.06, 0);
            try p.bf16("self_attn.kv_a_proj_with_mqa.weight", &.{ dsa_file.latent, dims }, 0.06, 0);
            try p.bf16("self_attn.q_a_layernorm.weight", &.{q_rank}, 0.2, 1);
            try p.bf16("self_attn.kv_a_layernorm.weight", &.{dsa_file.latent}, 0.2, 1);
            try p.bf16("self_attn.q_b_proj.weight", &.{ heads * key_dim, q_rank }, 0.1, 0);
            try p.bf16("self_attn.kv_b_proj.weight", &.{ heads * (key_dim + value_dim), dsa_file.latent }, 0.1, 0);
            try p.bf16("self_attn.o_proj.weight", &.{ dims, heads * value_dim }, 0.05, 0);
            // the fixture config's index sizes, which config() leaves: dense past 11 tokens
            try p.bf16("self_attn.indexer.wk.weight", &.{ 16, dims }, 0.06, 0);
            try p.bf16("self_attn.indexer.k_norm.weight", &.{16}, 0.2, 1);
            try p.bf16("self_attn.indexer.k_norm.bias", &.{16}, 0.05, 0);
            try p.bf16("self_attn.indexer.wq_b.weight", &.{ 32, q_rank }, 0.1, 0);
            try p.bf16("self_attn.indexer.weights_proj.weight", &.{ 2, dims }, 0.1, 0);
            try p.bf16("self_attn.indexer.index_kpool_compress_gate", &.{ 16, dims }, 0.06, 0);
            try p.bf16("self_attn.indexer.index_kpool_compress_ape", &.{ 4, 16 }, 0.1, 0);
        }

        fn moe(p: Putter) !void {
            try p.bf16("mlp.gate.weight", &.{ experts, dims }, 0.1, 0);
            try p.f32s("mlp.gate.e_score_correction_bias", experts, 0.05, 0);
            try p.dense("mlp.shared_experts.", moe_width);
            for (0..experts) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |leaf, m| {
                const k: usize = if (m == 2) moe_width else dims;
                const n: usize = if (m == 2) dims else moe_width;
                const prefix = try p.name(try std.fmt.allocPrint(p.arena, "mlp.experts.{d}.{s}", .{ e, leaf }));
                try fixtures.putNvfp4(p.arena, p.random, p.source, prefix, k, n, if (m == 2) 0.05 else 1.0 / @sqrt(@as(f64, @floatFromInt(k))));
            };
        }

        fn dense(self: Putter, leaf: []const u8, width: usize) !void {
            try self.bf16(try std.mem.concat(self.arena, u8, &.{ leaf, "gate_proj.weight" }), &.{ width, dims }, 0.1, 0);
            try self.bf16(try std.mem.concat(self.arena, u8, &.{ leaf, "up_proj.weight" }), &.{ width, dims }, 0.1, 0);
            try self.bf16(try std.mem.concat(self.arena, u8, &.{ leaf, "down_proj.weight" }), &.{ dims, width }, 0.05, 0);
        }
    };

    /// A whole random model under the checkpoint's names, its MTP layer after the last.
    pub fn putModel(arena: Allocator, random: std.Random, source: *Source) !void {
        const top_level: Putter = .{ .arena = arena, .random = random, .source = source };
        try top_level.bf16(layout.prefix ++ "embed_tokens.weight", &.{ vocab, dims }, 1.0, 0);
        try top_level.f32s(layout.prefix ++ "norm.weight", dims, 0.2, 1);
        try top_level.bf16("lm_head.weight", &.{ vocab, dims }, 0.05, 0);
        for (kinds, mlps, 0..) |kind, block, layer| {
            var p = top_level;
            p.under = try std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.", .{layer});
            try p.bf16("input_layernorm.weight", &.{dims}, 0.2, 1);
            try p.f32s("post_attention_layernorm.weight", dims, 0.2, 1);
            for ([_][]const u8{ "attn", "ffn" }) |site| {
                try p.bf16(try std.fmt.allocPrint(arena, "hc_{s}_fn", .{site}), &.{ host.mix, host.streams * dims }, 0.02, 0);
                try p.f32s(try std.fmt.allocPrint(arena, "hc_{s}_base", .{site}), host.mix, 0.5, 0);
                try p.f32s(try std.fmt.allocPrint(arena, "hc_{s}_scale", .{site}), 3, 0.2, 0.8);
            }
            switch (kind) {
                .linear_attention => try p.kda(),
                .deepseek_sparse_attention => try p.dsa(),
            }
            switch (block) {
                .dense => try p.dense("mlp.", intermediate),
                .sparse => try p.moe(),
            }
        }
        // the MTP layer after them: a DSA and a MoE with plain residuals, its projection and norms
        var p = top_level;
        p.under = try std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.", .{kinds.len});
        for ([_][]const u8{ "input_layernorm", "post_attention_layernorm", "enorm", "hnorm", "shared_head.norm" }) |norm|
            try p.bf16(try std.fmt.allocPrint(arena, "{s}.weight", .{norm}), &.{dims}, 0.2, 1);
        try p.bf16("eh_proj.weight", &.{ dims, 2 * dims }, 0.04, 0);
        try p.dsa();
        try p.moe();
    }
};
