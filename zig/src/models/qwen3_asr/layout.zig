//! Qwen3-ASR's language model's tensors in its checkpoint (`thinker.model.*`, `thinker.lm_head`), by name, each
//! checked against the config's shape (`config.zig`). The norms and the head are bf16; a layer's seven linears are
//! bf16 or, in a W4A16_NVFP4 build (spec 140), as its `quantization_config` says (`modelopt.read`: codes, e4m3
//! block scales, `weight_scale_2`). `thinker.model.embed_tokens` is the head's bytes again (tied, and equal to
//! the bit: spec 138 WS-01), so the head alone is loaded and serves as the embedding too; ModelOpt's export
//! writes only `embed_tokens` (the same bytes, spec 140 WS-03), which is then the head. The audio tower's tensors
//! are `tower.zig`'s. Host only.

const std = @import("std");
const Allocator = std.mem.Allocator;
const config_file = @import("config.zig");
const modelopt = @import("../../weights/modelopt.zig");
const reader = @import("../../weights/checkpoint_reader.zig");
const safetensors = @import("../../weights/safetensors.zig");

const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;
const Finder = reader.Finder;
const Tensor = safetensors.Tensor;

pub const Error = reader.FindError || modelopt.Error;

pub const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    q: modelopt.Linear,
    k: modelopt.Linear,
    v: modelopt.Linear,
    o: modelopt.Linear,
    /// [head_dim]: a head's norm, the same for every head
    q_norm: Tensor,
    k_norm: Tensor,
    gate: modelopt.Linear,
    up: modelopt.Linear,
    down: modelopt.Linear,

    /// The linears' tensors' bytes, what a load reads.
    pub fn ranges(self: Layer, out: *[4 * 7][]const u8) [][]const u8 {
        var count: usize = 0;
        inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }) |name| {
            const linear = @field(self, name);
            out[count] = linear.weight.data;
            count += 1;
            inline for (.{ "scale", "scale_2", "input_scale" }) |part| if (@field(linear, part)) |t| {
                out[count] = t.data;
                count += 1;
            };
        }
        return out[0..count];
    }
};

pub const Layout = struct {
    norm: Tensor,
    /// [vocab, hidden]: the head, and the embedding's table
    head: Tensor,
    layers: []const Layer,

    /// `quant`: `Config.quantization`'s, null for the bf16 base.
    pub fn init(arena: Allocator, config: *const Config, checkpoint: *const safetensors.Checkpoint, quant: ?modelopt.Uniform, diagnostic: *Diagnostic) Error!Layout {
        const f: Finder = .{ .arena = arena, .checkpoint = checkpoint, .diagnostic = diagnostic };
        const l: Linears = .{ .arena = arena, .checkpoint = checkpoint, .quant = quant, .diagnostic = diagnostic };
        const c = config.*;
        const q = c.heads * c.head_dim;
        const kv = c.kv_heads * c.head_dim;
        const layers = try arena.alloc(Layer, c.layers);
        for (layers, 0..) |*layer, i| {
            const at = "thinker.model.layers.";
            layer.* = .{
                .input_norm = try f.get(&.{c.hidden}, "{s}{d}.input_layernorm.weight", .{ at, i }),
                .post_norm = try f.get(&.{c.hidden}, "{s}{d}.post_attention_layernorm.weight", .{ at, i }),
                .q = try l.get(q, c.hidden, "{s}{d}.self_attn.q_proj", .{ at, i }),
                .k = try l.get(kv, c.hidden, "{s}{d}.self_attn.k_proj", .{ at, i }),
                .v = try l.get(kv, c.hidden, "{s}{d}.self_attn.v_proj", .{ at, i }),
                .o = try l.get(c.hidden, q, "{s}{d}.self_attn.o_proj", .{ at, i }),
                .q_norm = try f.get(&.{c.head_dim}, "{s}{d}.self_attn.q_norm.weight", .{ at, i }),
                .k_norm = try f.get(&.{c.head_dim}, "{s}{d}.self_attn.k_norm.weight", .{ at, i }),
                .gate = try l.get(c.intermediate, c.hidden, "{s}{d}.mlp.gate_proj", .{ at, i }),
                .up = try l.get(c.intermediate, c.hidden, "{s}{d}.mlp.up_proj", .{ at, i }),
                .down = try l.get(c.hidden, c.intermediate, "{s}{d}.mlp.down_proj", .{ at, i }),
            };
        }
        return .{
            .norm = try f.get(&.{c.hidden}, "thinker.model.norm.weight", .{}),
            .head = try f.get(&.{ c.vocab, c.hidden }, "{s}", .{if (checkpoint.get("thinker.lm_head.weight") == null) "thinker.model.embed_tokens.weight" else "thinker.lm_head.weight"}),
            .layers = layers,
        };
    }
};

/// A layer's linears by module name, each as `quant` stores it (`modelopt.read` checks its parts).
const Linears = struct {
    arena: Allocator,
    checkpoint: *const safetensors.Checkpoint,
    quant: ?modelopt.Uniform,
    diagnostic: *Diagnostic,

    fn get(self: Linears, rows: usize, cols: usize, comptime format: []const u8, args: anytype) Error!modelopt.Linear {
        const module = try std.fmt.allocPrint(self.arena, format, args);
        const layer = if (self.quant) |q| q.of(module) else null;
        return modelopt.read(self.arena, self.checkpoint, module, layer, rows, cols, self.diagnostic);
    }
};
