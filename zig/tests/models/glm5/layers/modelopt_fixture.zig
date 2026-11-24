//! A ModelOpt MIXED_PRECISION checkpoint of the four-layer config (`checks/fixtures.zig`), as FP4-Local stores
//! GLM-5.3-Flash at the sizes the fixture allows: MXFP8 beside the experts where the inputs are whole blocks of
//! 32 (FP8_PB_WO needs 128, which no fixture linear has), the routed experts NVFP4 with csf_32x2_v1 scales and
//! an `input_scale`, the MTP layer's plain (W4A16_NVFP4), a vision entry; every other linear BF16. What
//! `layout_test.zig` and `share_test.zig` read.

const std = @import("std");
const safetensors = @import("../../../../src/weights/safetensors.zig");
const modelopt = @import("../../../../src/weights/modelopt.zig");
const csf_blocks = @import("../../../../src/weights/csf_blocks.zig");
const fixtures = @import("../../../../src/models/glm5/checks/fixtures.zig");
const config_file = @import("../../../../src/models/glm5/layers/config.zig");
const layout_file = @import("../../../../src/models/glm5/layers/layout.zig");

const mx = "{\"quant_algo\":\"MXFP8\",\"weight_block_size\":[1,32]}";
const mx_mtp = "{\"group_size\":32,\"quant_algo\":\"MXFP8\"}";

fn kda(comptime layer: []const u8) []const u8 {
    const p = "\"model.language_model.layers." ++ layer ++ ".";
    return p ++ "self_attn.q_proj\":" ++ mx ++ "," ++ p ++ "self_attn.k_proj\":" ++ mx ++ "," ++ p ++ "self_attn.v_proj\":" ++ mx ++
        "," ++ p ++ "self_attn.o_proj\":" ++ mx ++ "," ++ p ++ "mlp.gate_proj\":" ++ mx ++ "," ++ p ++ "mlp.up_proj\":" ++ mx ++
        "," ++ p ++ "mlp.down_proj\":" ++ mx ++ ",";
}

pub const quantized_layers = "\"lm_head\":" ++ mx ++ "," ++ kda("0") ++ kda("1") ++ kda("2") ++
    "\"model.language_model.layers.3.self_attn.q_a_proj\":" ++ mx ++ "," ++
    "\"model.language_model.layers.3.self_attn.kv_a_proj_with_mqa\":" ++ mx ++ "," ++
    "\"model.language_model.layers.3.self_attn.o_proj\":" ++ mx ++ "," ++
    "\"model.language_model.layers.3.mlp.shared_experts.gate_proj\":" ++ mx ++ "," ++
    "\"model.language_model.layers.3.mlp.shared_experts.up_proj\":" ++ mx ++ "," ++
    "\"model.language_model.layers.3.mlp.shared_experts.down_proj\":" ++ mx ++ "," ++
    "\"model.language_model.layers.3.mlp.experts\":{\"group_size\":16,\"quant_algo\":\"NVFP4\",\"weight_scale_encoding\":\"csf_32x2_v1\"}," ++
    "\"model.language_model.layers.4.self_attn.kv_b_proj\":" ++ mx_mtp ++ "," ++
    "\"model.language_model.layers.4.mlp.experts\":{\"group_size\":16,\"quant_algo\":\"W4A16_NVFP4\"}," ++
    "\"model.visual.blocks.0.attn.qkv\":" ++ mx_mtp;

pub const config_quant = ",\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"MIXED_PRECISION\",\"exclude_modules\":[],\"quantized_layers\":{" ++
    quantized_layers ++ "}}}";
pub const config = fixtures.config_head ++ fixtures.config_rest ++ config_quant;

/// Linears by storage, those with csf scales, and the map's language-model entries, as `config` makes them.
pub const mxfp8 = 29;
pub const nvfp4 = 36;
pub const csf32 = 18;
pub const bf16 = 33;
pub const mapped = 31;

fn put(arena: std.mem.Allocator, checkpoint: *safetensors.Checkpoint, random: std.Random, name: []const u8, dtype: safetensors.DType, shape: []const u64, zero: bool) !void {
    var tensor: safetensors.Tensor = .{ .dtype = dtype, .shape = try arena.dupe(u64, shape), .data = &.{} };
    const data = try arena.alloc(u8, @intCast(tensor.count() * dtype.size()));
    if (zero) @memset(data, 0) else random.bytes(data);
    tensor.data = data;
    try checkpoint.tensors.put(arena, try arena.dupe(u8, name), tensor);
}

/// The linear `module` of [rows, cols] as `layer` stores it; csf streams of zeros (a valid plane of bytes equal
/// to the bases), every other part random.
fn putLinear(arena: std.mem.Allocator, checkpoint: *safetensors.Checkpoint, random: std.Random, module: []const u8, layer: ?modelopt.QuantizedLayer, rows: u64, cols: u64) !void {
    var name: [256]u8 = undefined;
    const at = struct {
        fn of(buffer: []u8, m: []const u8, leaf: []const u8) []const u8 {
            return std.fmt.bufPrint(buffer, "{s}.{s}", .{ m, leaf }) catch unreachable;
        }
    }.of;
    switch (modelopt.Scheme.of(layer).?) {
        .bf16 => try put(arena, checkpoint, random, at(&name, module, "weight"), .BF16, &.{ rows, cols }, false),
        .mxfp8 => {
            try put(arena, checkpoint, random, at(&name, module, "weight"), .F8_E4M3, &.{ rows, cols }, false);
            try put(arena, checkpoint, random, at(&name, module, "weight_scale"), .U8, &.{ rows, cols / 32 }, false);
        },
        .fp8_block, .fp8 => unreachable,
        .nvfp4 => {
            try put(arena, checkpoint, random, at(&name, module, "weight"), .U8, &.{ rows, cols / 2 }, false);
            try put(arena, checkpoint, random, at(&name, module, "weight_scale_2"), .F32, &.{}, false);
            const plane: csf_blocks.Shape = .{ .rows = rows, .cols = cols / 16 };
            if (layer.?.scale_encoding == .csf32) {
                const streams = [_]struct { []const u8, safetensors.DType, u64 }{
                    .{ csf_blocks.bases_suffix, .U8, plane.rows },
                    .{ csf_blocks.codes_suffix, .U8, plane.codeBytes() },
                    .{ csf_blocks.page_bases_suffix, .U32, plane.pages() },
                    .{ csf_blocks.offsets_suffix, .U16, plane.blocks() + plane.pages() },
                    .{ csf_blocks.exceptions_suffix, .U16, 0 },
                };
                for (streams) |stream| {
                    var leaf: [64]u8 = undefined;
                    try put(arena, checkpoint, random, at(&name, module, std.fmt.bufPrint(&leaf, "weight_scale{s}", .{stream[0]}) catch unreachable), stream[1], &.{stream[2]}, true);
                }
                try put(arena, checkpoint, random, at(&name, module, "input_scale"), .F32, &.{}, false);
            } else try put(arena, checkpoint, random, at(&name, module, "weight_scale"), .F8_E4M3, &.{ plane.rows, plane.cols }, false);
        },
    }
}

/// A checkpoint of every tensor `c` (a ModelOpt config) asks for, its linears as the map stores them.
pub fn synthetic(arena: std.mem.Allocator, c: config_file.Config, seed: u64) !safetensors.Checkpoint {
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    var checkpoint: safetensors.Checkpoint = .{ .shards = &.{}, .tensors = .empty, .gpa = arena };
    var name: [192]u8 = undefined;
    for (layout_file.kinds) |kind| {
        if (kind.where == .top) {
            try putKind(arena, &checkpoint, random, c, kind, kind.leaf);
            continue;
        }
        for (0..c.layers + c.mtp_layers) |layer| {
            if (!layout_file.Layout.holds(c, kind.where, layer)) continue;
            if (kind.where != .expert) {
                try putKind(arena, &checkpoint, random, c, kind, std.fmt.bufPrint(&name, layout_file.prefix ++ "layers.{d}.{s}", .{ layer, kind.leaf }) catch unreachable);
            } else for (0..c.experts) |expert| {
                try putKind(arena, &checkpoint, random, c, kind, std.fmt.bufPrint(&name, layout_file.prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ layer, expert, kind.leaf }) catch unreachable);
            }
        }
    }
    return checkpoint;
}

fn putKind(arena: std.mem.Allocator, checkpoint: *safetensors.Checkpoint, random: std.Random, c: config_file.Config, kind: layout_file.Kind, name: []const u8) !void {
    var dims: [3]u64 = undefined;
    const shape = kind.shape(c, &dims);
    if (!kind.linear) return put(arena, checkpoint, random, name, kind.dtype, shape, false);
    const module = name[0 .. name.len - ".weight".len];
    try putLinear(arena, checkpoint, random, module, c.quant.find(module), shape[0], shape[1]);
}
