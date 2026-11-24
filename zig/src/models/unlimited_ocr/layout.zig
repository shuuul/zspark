//! Unlimited-OCR's language model's tensors in its checkpoint, by name, each checked against the config's shape
//! (`config.zig`). Every tensor is bf16, the checkpoint's own storage. The vision tower's, the projector's and
//! `model.image_newline` / `model.view_seperator` are WS-03's (`vision*.zig`). Unlimited-OCR-2 (`config.dash`,
//! spec 147) stores each layer's DASH router; what it stores and no forward reads is `unused`. Host only.

const std = @import("std");
const Allocator = std.mem.Allocator;
const config_file = @import("config.zig");
const reader = @import("../../weights/checkpoint_reader.zig");
const safetensors = @import("../../weights/safetensors.zig");

const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;
const Finder = reader.Finder;
const Tensor = safetensors.Tensor;

pub const Error = reader.FindError;

pub const Dense = struct {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
};

pub const Experts = struct {
    /// [64, hidden]: the router's weight (`mlp.gate.weight`)
    router: Tensor,
    /// An expert's `gate_proj`, `up_proj` [width, hidden] and `down_proj` [hidden, width], expert by expert.
    gate: []const Tensor,
    up: []const Tensor,
    down: []const Tensor,
    /// `mlp.shared_experts`, the shared experts fused into one MLP of `shared * width`.
    shared: Dense,
};

/// A layer's DASH router (`self_attn.dash_router`, `dash_attention.py` `DashAttention`): one index head of
/// `index_dim`.
pub const Router = struct {
    /// [index_dim]: the summary query a block's keys are softmaxed against (`head_cls`)
    head_cls: Tensor,
    /// [index_dim, hidden] each
    q: Tensor,
    k: Tensor,
    /// [index_dim] each, Gemma's RMSNorm (`x * (1 + w)`)
    q_norm: Tensor,
    k_norm: Tensor,
};

pub const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    mlp: union(enum) { dense: Dense, experts: Experts },
    router: Router,
};

/// What Unlimited-OCR-2 stores and nothing reads, by name: a layer's `self_attn.indexer.*` (the DSA indexer its
/// training's KL loss used; HF builds it but DASH's inference reads `dash_router` alone) and the vision tower's
/// `position_ids` (I64 [1, 257], 0 to 256: the rows of the position table `vision_clip.zig` reads whole). A check
/// on the tensors no loader reads takes these as read.
pub fn unused(name: []const u8) bool {
    if (std.mem.eql(u8, name, "model.vision_model.embeddings.position_ids")) return true;
    const at = "model.layers.";
    if (!std.mem.startsWith(u8, name, at)) return false;
    const dot = std.mem.indexOfScalarPos(u8, name, at.len, '.') orelse return false;
    _ = std.fmt.parseUnsigned(usize, name[at.len..dot], 10) catch return false;
    const rest = name[dot + 1 ..];
    for ([_][]const u8{ "q_proj", "k_proj", "q_norm", "k_norm" }) |part| {
        var buf: [64]u8 = undefined;
        const want = std.fmt.bufPrint(&buf, "self_attn.indexer.{s}.weight", .{part}) catch unreachable;
        if (std.mem.eql(u8, rest, want)) return true;
    }
    return false;
}

pub const Layout = struct {
    embed: Tensor,
    norm: Tensor,
    head: Tensor,
    layers: []const Layer,

    pub fn init(arena: Allocator, config: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic) Error!Layout {
        const f: Finder = .{ .arena = arena, .checkpoint = checkpoint, .diagnostic = diagnostic };
        const c = config.*;
        const layers = try arena.alloc(Layer, c.layers);
        for (layers, 0..) |*layer, i| {
            const at = "model.layers.";
            layer.* = .{
                .input_norm = try f.get(&.{c.hidden}, "{s}{d}.input_layernorm.weight", .{ at, i }),
                .post_norm = try f.get(&.{c.hidden}, "{s}{d}.post_attention_layernorm.weight", .{ at, i }),
                .q = try f.get(&.{ c.hidden, c.hidden }, "{s}{d}.self_attn.q_proj.weight", .{ at, i }),
                .k = try f.get(&.{ c.hidden, c.hidden }, "{s}{d}.self_attn.k_proj.weight", .{ at, i }),
                .v = try f.get(&.{ c.hidden, c.hidden }, "{s}{d}.self_attn.v_proj.weight", .{ at, i }),
                .o = try f.get(&.{ c.hidden, c.hidden }, "{s}{d}.self_attn.o_proj.weight", .{ at, i }),
                .mlp = undefined,
                .router = undefined,
            };
            {
                const n = c.dash.index_dim;
                const r = "self_attn.dash_router";
                layer.router = .{
                    .head_cls = try f.get(&.{n}, "{s}{d}.{s}.head_cls", .{ at, i, r }),
                    .q = try f.get(&.{ n, c.hidden }, "{s}{d}.{s}.q_proj.weight", .{ at, i, r }),
                    .k = try f.get(&.{ n, c.hidden }, "{s}{d}.{s}.k_proj.weight", .{ at, i, r }),
                    .q_norm = try f.get(&.{n}, "{s}{d}.{s}.q_norm.weight", .{ at, i, r }),
                    .k_norm = try f.get(&.{n}, "{s}{d}.{s}.k_norm.weight", .{ at, i, r }),
                };
            }
            if (i < c.dense_layers) {
                layer.mlp = .{ .dense = try dense(f, c.intermediate, c.hidden, "{s}{d}.mlp", .{ at, i }) };
                continue;
            }
            const gate = try arena.alloc(Tensor, c.experts);
            const up = try arena.alloc(Tensor, c.experts);
            const down = try arena.alloc(Tensor, c.experts);
            for (gate, up, down, 0..) |*g, *u, *d, e| {
                g.* = try f.get(&.{ c.expert_width, c.hidden }, "{s}{d}.mlp.experts.{d}.gate_proj.weight", .{ at, i, e });
                u.* = try f.get(&.{ c.expert_width, c.hidden }, "{s}{d}.mlp.experts.{d}.up_proj.weight", .{ at, i, e });
                d.* = try f.get(&.{ c.hidden, c.expert_width }, "{s}{d}.mlp.experts.{d}.down_proj.weight", .{ at, i, e });
            }
            layer.mlp = .{ .experts = .{
                .router = try f.get(&.{ c.experts, c.hidden }, "{s}{d}.mlp.gate.weight", .{ at, i }),
                .gate = gate,
                .up = up,
                .down = down,
                .shared = try dense(f, c.sharedWidth(), c.hidden, "{s}{d}.mlp.shared_experts", .{ at, i }),
            } };
        }
        return .{
            .embed = try f.get(&.{ c.vocab, c.hidden }, "model.embed_tokens.weight", .{}),
            .norm = try f.get(&.{c.hidden}, "model.norm.weight", .{}),
            .head = try f.get(&.{ c.vocab, c.hidden }, "lm_head.weight", .{}),
            .layers = layers,
        };
    }
};

/// The MLP of `format`'s gate, up and down projections, `width` wide.
fn dense(f: Finder, width: usize, hidden: usize, comptime format: []const u8, args: anytype) Error!Dense {
    return .{
        .gate = try f.get(&.{ width, hidden }, format ++ ".gate_proj.weight", args),
        .up = try f.get(&.{ width, hidden }, format ++ ".up_proj.weight", args),
        .down = try f.get(&.{ hidden, width }, format ++ ".down_proj.weight", args),
    };
}
