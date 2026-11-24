//! A random DFlash2 drafter as a source of tensors, shared by the models' checks and the tests of
//! `decode/dflash2/` (spec 154 WS-02): the per-model `config.json` text stays with its model
//! (`models/*/checks/random_drafter.zig`), and the reading of it, the heads a model asks for and the tensors
//! every kind wants live here once. Host only: no device, no checkpoint.

const std = @import("std");
const Allocator = std.mem.Allocator;
const safetensors = @import("../../weights/safetensors.zig");
const numeric = @import("../../weights/numeric.zig");
const config_file = @import("config.zig");
const layout = @import("layout.zig");

pub const Source = std.StringHashMapUnmanaged(safetensors.Tensor);

/// Which entry of the parser reads the fixture's `config.json`, as the model's own schema is read:
/// `stored` a `sliding_window` written as-is (Qwen), `plain` one written exclusive (GLM), `full` a null one
/// (Nemotron, the drafter's layers all full attention).
pub const Reader = enum { stored, plain, full };

/// What the fixture leaves to its model: the query heads are `hidden / head_dim` times `heads_scale`
/// (Nemotron's published drafter has 1,024 over hidden 512), and `kv_divisor` 0 keeps the fixture's
/// key/value heads (Nemotron's GQA), else `heads / kv_divisor` (Qwen's one a head, GLM's two a pair).
pub const Shape = struct {
    heads_scale: usize = 1,
    kv_divisor: usize = 0,
};

/// The fixture's config for a main model of `hidden`, `vocab` and `layers`, tapped after `taps`.
pub fn config(arena: Allocator, reader: Reader, fixture: []const u8, shape: Shape, hidden: usize, vocab: usize, layers: usize, taps: []const u32) !config_file.Config {
    var diagnostic: config_file.Diagnostic = .{};
    var c = switch (reader) {
        .stored => try config_file.Config.parseStored(arena, fixture, &diagnostic),
        .plain => try config_file.Config.parse(arena, fixture, &diagnostic),
        .full => try config_file.Config.parseFull(arena, fixture, &diagnostic),
    };
    c.hidden = hidden;
    c.heads = shape.heads_scale * (hidden / c.head_dim);
    if (shape.kv_divisor != 0) c.kv_heads = c.heads / shape.kv_divisor;
    c.vocab = vocab;
    c.mask = @intCast(vocab - 3);
    c.top_k = @min(c.top_k, vocab / 4);
    c.target_layers = layers;
    c.taps = try arena.dupe(u32, taps);
    return c;
}

/// Every tensor `c` asks for, random: weights of a spread that keeps a row's size through a layer, norms and
/// base kernels about one.
pub fn put(arena: Allocator, random: std.Random, c: config_file.Config, source: *Source) !void {
    for (layout.kinds) |kind| {
        var dims: [3]u64 = undefined;
        const shape = try arena.dupe(u64, kind.shape(c, &dims));
        var count: usize = 1;
        for (shape) |dim| count *= @intCast(dim);
        const vector = shape.len == 1 or shape.len == 3;
        const spread: f32 = if (vector) 0.1 else 1.0 / @sqrt(@as(f32, @floatFromInt(shape[shape.len - 1])));
        for (0..if (kind.layer) c.layers else 1) |layer| {
            const values = try arena.alloc(u16, count);
            for (values) |*one| one.* = numeric.bf16FromF32((if (vector) @as(f32, 1.0) else 0.0) + random.floatNorm(f32) * spread);
            const name = if (kind.layer) try std.fmt.allocPrint(arena, "layers.{d}.{s}", .{ layer, kind.leaf }) else kind.leaf;
            try source.put(arena, name, .{ .dtype = .BF16, .shape = shape, .data = std.mem.sliceAsBytes(values) });
        }
    }
}
