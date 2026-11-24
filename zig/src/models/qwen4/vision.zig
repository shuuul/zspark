//! Flash Next's vision tower (spec 019 WS-02): an image's patches to the features that stand in for its
//! placeholder tokens, from the checkpoint's `model.visual.` tensors as stored. transformers 5.18's
//! `Qwen3_5VisionModel` as TensorFold 0.6.5 runs it: the patches' linear (Conv3d) and the learned position
//! table resampled to the grid; 27 blocks of LayerNorm, qkv, the 2D rope, attention by whole image, proj,
//! LayerNorm, fc1, GELU (tanh), fc2, each a residual; the merger's LayerNorm, four patches as one row, fc1,
//! exact GELU, fc2 to the language model's 2,560.
//!
//! The kernels: `kernels/bf16/gemm_bias.cu` (a BF16 linear, its bias and GELU in the epilogue),
//! `kernels/nvfp4/prompt_bias.cu` (an MXFP8 or NVFP4 linear with bf16 activations, the same epilogue),
//! `kernels/qwen4/vision.cu` (positions, rope, the residual add and LayerNorm) and `vision_attention.cu`. A
//! residual is added by the next norm's launch (`pending`). Images go through in runs of whole images of at
//! most `max_patches` patches, TensorFold's `MAX_PATCHES`; an image never attends to another, and a row's bits
//! are the same alone, beside other images or in another order (the row rule).
//!
//! Each linear is read in the format its tensors have (`Format.of`): the retired NVFP4 checkpoint stores the
//! whole tower BF16 (spec 019 WS-02); a ModelOpt mixed-precision one (spec 019 WS-08) stores the attention's
//! qkv and proj MXFP8 and the MLP's and merger's linears NVFP4 (`W4A16_NVFP4`), the rest BF16, and runs them as
//! stored (W8A16, W4A16: no activation is quantized). TensorFold 0.6.5 takes only an unquantized tower
//! (qwen_cuda.py:105), so a quantized tower is held to the BF16 one by the spread measured in WS-08. An EXL3
//! pack's tower (trellises, F16) is refused (the spec's decision of 2026-10-13).

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const cuda = @import("../../cuda/cuda.zig");
const numeric = @import("../../weights/numeric.zig");
const safetensors = @import("../../weights/safetensors.zig");
const nvfp4 = @import("../../weights/nvfp4.zig");
const qwen4_config = @import("config.zig");
const Diagnostic = qwen4_config.Diagnostic;

/// The tower's shape, which the kernels are built for: config.json's `vision_config` must say the same.
pub const hidden = 1152;
pub const heads = 16;
pub const head_dim = 72;
pub const intermediate = 4304;
pub const out_hidden = 2560;
pub const patch_values = 3 * 2 * 16 * 16;
pub const merge_unit = 4;
pub const merged = hidden * merge_unit;
pub const positions = 48 * 48;
pub const eps = 1e-6;
/// TensorFold's `rotary_frequencies` default; the published config names no theta.
pub const rope_theta = 10000.0;
/// TensorFold's `MAX_PATCHES`: the most patches a tower call takes, and so the largest image.
pub const max_patches_default = 16384;

pub const Error = error{UnsupportedVision} || cuda.Error || Allocator.Error;

/// An image's grid of patches (`image_grid_thw`): frames, then rows and columns of 16-pixel patches.
pub const Grid = struct {
    t: usize,
    h: usize,
    w: usize,

    pub fn patches(self: Grid) usize {
        return self.t * self.h * self.w;
    }
};

/// The vision_config fields the kernels take as constants, checked; anything else refused in words.
pub fn checkConfig(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) (error{ UnsupportedVision, InvalidConfig } || Allocator.Error)!void {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch {
        diagnostic.set(arena, "config.json is not JSON", .{});
        return error.InvalidConfig;
    };
    const vision = if (root == .object) root.object.get("vision_config") else null;
    const config = (vision orelse {
        diagnostic.set(arena, "config.json has no vision_config: this checkpoint has no vision tower", .{});
        return error.UnsupportedVision;
    });
    if (config != .object) {
        diagnostic.set(arena, "config.json's vision_config is not an object", .{});
        return error.InvalidConfig;
    }
    const object = config.object;
    if (object.get("model_type")) |value| {
        const name = if (value == .string) value.string else "";
        // the published NVFP4 directory's is `qwen4_exp_vision` (decision 2026-10-13); TensorFold 0.6.5 reads
        // `qwen3_5`, its packs' `qwen4_exp`
        const known = [_][]const u8{ "qwen4_exp_vision", "qwen4_exp", "qwen3_5", "qwen3_5_vision" };
        for (known) |each| {
            if (std.mem.eql(u8, name, each)) break;
        } else {
            diagnostic.set(arena, "vision_config.model_type \"{s}\" is not Flash Next's tower ({s}, {s} or {s})", .{ name, known[0], known[1], known[2] });
            return error.UnsupportedVision;
        }
    }
    const expected = [_]struct { []const u8, i64 }{
        .{ "depth", 27 },                          .{ "hidden_size", hidden },         .{ "num_heads", heads },
        .{ "intermediate_size", intermediate },    .{ "out_hidden_size", out_hidden }, .{ "patch_size", 16 },
        .{ "temporal_patch_size", 2 },             .{ "spatial_merge_size", 2 },       .{ "in_channels", 3 },
        .{ "num_position_embeddings", positions },
    };
    for (expected) |entry| {
        const value = object.get(entry[0]) orelse {
            diagnostic.set(arena, "vision_config has no {s}", .{entry[0]});
            return error.InvalidConfig;
        };
        if (value != .integer or value.integer != entry[1]) {
            diagnostic.set(arena, "vision_config.{s} is not {d}: zspark's tower kernels are built for Flash Next's shape", .{ entry[0], entry[1] });
            return error.UnsupportedVision;
        }
    }
    if (object.get("hidden_act")) |value| {
        if (value != .string or !std.mem.eql(u8, value.string, "gelu_pytorch_tanh")) {
            diagnostic.set(arena, "vision_config.hidden_act is not gelu_pytorch_tanh", .{});
            return error.UnsupportedVision;
        }
    }
    if (object.get("deepstack_visual_indexes")) |value| {
        if (value == .array and value.array.items.len != 0) {
            diagnostic.set(arena, "vision_config.deepstack_visual_indexes is not empty: deepstack features are not taken", .{});
            return error.UnsupportedVision;
        }
    }
}

/// The epilogue's activation (`gemm_bias.cu`).
pub const Act = enum(i64) { none = 0, gelu_tanh = 1, gelu_erf = 2 };

extern fn zs_qwen4_vision_pos(embed: *const cuda.View, bias: *const cuda.View, table: *const cuda.View, places: *const cuda.View, out: *const cuda.View) c_int;
extern fn zs_qwen4_vision_rope(qkv: *const cuda.View, places: *const cuda.View, theta: f64) c_int;
extern fn zs_qwen4_vision_norm(h: *const cuda.View, branch: ?*const cuda.View, weight: ?*const cuda.View, bias: ?*const cuda.View, y: ?*const cuda.View, eps: f64) c_int;
extern fn zs_qwen4_vision_attention(qkv: *const cuda.View, tiles: *const cuda.View, out: *const cuda.View) c_int;

/// A linear with its bias, its weight as the checkpoint stores it: BF16 [n, k] (`bf16/gemm_bias.cu`), or MXFP8
/// or NVFP4 with bf16 activations (`nvfp4/prompt_bias.cu`, W8A16 and W4A16). Bias bf16 [width].
pub const Linear = struct {
    weight: Weight,
    /// bf16 [width]: the stored [n], then zeros to the columns a quantized linear writes (`width`)
    bias: cuda.Buffer,

    pub const Weight = union(enum) {
        /// [n, k] as stored
        bf16: cuda.Buffer,
        /// e4m3 codes and a ue8m0 exponent a 32 inputs (`nvfp4.Mx8Linear`'s layout)
        mxfp8: nvfp4.Mx8Linear,
        /// e2m1 codes, an e4m3 scale a 16 inputs and the tensor's fp32 scale (`nvfp4.Fp4Linear`'s layout); a K
        /// that is not a multiple of 64 (fc2's 4,304) padded with zero codes and zero scales
        nvfp4: nvfp4.Fp4Linear,
    };

    /// out (m, width) = act(x (m, k) W^T + bias), bf16; a row's bits are its own (one tile, no K split). A
    /// quantized linear may write columns past its n up to its padding (the bias's width): exact zeros.
    pub fn forward(self: Linear, x: cuda.View, out: cuda.View, act: Act) cuda.Error!void {
        const a: nvfp4.Activation = @fromBackingInt(@intCast(@backingInt(act)));
        return switch (self.weight) {
            .bf16 => |w| cuda.launched(cuda.kernels.zs_b16_gemm_bias(&x, &w.view, &self.bias.view, &out, @backingInt(act))),
            .mxfp8 => |w| w.forwardBiased(x, self.bias.view, out, a),
            .nvfp4 => |w| w.forwardBiased(x, self.bias.view, out, a),
        };
    }

    pub fn deinit(self: *Linear) void {
        switch (self.weight) {
            inline else => |*w| w.deinit(),
        }
        self.bias.deinit();
    }
};

/// How a linear's weight is stored, read from its tensors' dtypes and shapes (not from a name or a config).
pub const Format = enum {
    bf16,
    mxfp8,
    nvfp4,

    /// The format of `prefix`'s weight [n, k]: BF16 [n, k]; MXFP8, F8_E4M3 [n, k] with U8 `weight_scale` [n,
    /// k/32]; NVFP4, U8 [n, k/2] with F8_E4M3 `weight_scale` [n, k/16] and an F32 `weight_scale_2`. Anything
    /// else is refused, with what was found in `diagnostic`.
    pub fn of(arena: Allocator, checkpoint: *const safetensors.Checkpoint, prefix: []const u8, n: usize, k: usize, diagnostic: *Diagnostic) error{UnsupportedVision}!Format {
        var buffers: [3][128]u8 = undefined;
        const names = tensorNames(&buffers, prefix);
        const weight = checkpoint.get(names[0]) orelse {
            diagnostic.set(arena, "the vision tower lacks {s}: no image can be read with this checkpoint", .{names[0]});
            return error.UnsupportedVision;
        };
        const scale = checkpoint.get(names[1]);
        const scale_2 = checkpoint.get(names[2]);
        const shaped = struct {
            fn is(t: safetensors.Tensor, dims: []const u64) bool {
                return std.mem.eql(u64, t.shape, dims);
            }
        }.is;
        if (weight.dtype == .BF16 and shaped(weight, &.{ n, k }) and scale == null) return .bf16;
        if (weight.dtype == .F8_E4M3 and shaped(weight, &.{ n, k }) and k % 32 == 0) {
            if (scale) |s| if (s.dtype == .U8 and shaped(s, &.{ n, k / 32 }) and scale_2 == null) return .mxfp8;
        }
        if (weight.dtype == .U8 and shaped(weight, &.{ n, k / 2 }) and k % 16 == 0) {
            if (scale) |s| if (scale_2) |s2| if (s.dtype == .F8_E4M3 and shaped(s, &.{ n, k / 16 }) and s2.dtype == .F32 and s2.count() == 1) return .nvfp4;
        }
        diagnostic.set(arena, "the vision tower's {s} is {t} {any}{s}: zspark reads a tower linear as BF16 [{d}, {d}], MXFP8 (e4m3 with ue8m0 scales a 32) or NVFP4 (e2m1 with e4m3 scales a 16 and an fp32 scale), as stored", .{ names[0], weight.dtype, weight.shape, if (scale != null) " with a weight_scale" else "", n, k });
        return error.UnsupportedVision;
    }
};

/// `prefix`'s `.weight`, `.weight_scale` and `.weight_scale_2`.
fn tensorNames(buffers: *[3][128]u8, prefix: []const u8) [3][]const u8 {
    return .{
        std.fmt.bufPrint(&buffers[0], "{s}.weight", .{prefix}) catch unreachable,
        std.fmt.bufPrint(&buffers[1], "{s}.weight_scale", .{prefix}) catch unreachable,
        std.fmt.bufPrint(&buffers[2], "{s}.weight_scale_2", .{prefix}) catch unreachable,
    };
}

/// A LayerNorm's weight and bias, bf16 [1152].
pub const Norm = struct {
    weight: cuda.Buffer,
    bias: cuda.Buffer,

    fn deinit(self: *Norm) void {
        self.weight.deinit();
        self.bias.deinit();
    }
};

pub const Block = struct {
    norm1: Norm,
    qkv: Linear,
    proj: Linear,
    norm2: Norm,
    fc1: Linear,
    fc2: Linear,
    /// The columns fc1 writes and fc2 reads: 4,304, or 4,352 before a quantized fc2 (the last 48 zero)
    inner: usize,
};

pub const depth = 27;

/// h = bf(h + branch) when `branch`, then y = LayerNorm(h) when `y`.
pub fn norm(h: cuda.View, branch: ?cuda.View, by: ?Norm, y: ?cuda.View) cuda.Error!void {
    const b: ?*const cuda.View = if (branch) |*v| v else null;
    const w: ?*const cuda.View = if (by) |*n| &n.weight.view else null;
    const bi: ?*const cuda.View = if (by) |*n| &n.bias.view else null;
    const yv: ?*const cuda.View = if (y) |*v| v else null;
    return cuda.launched(zs_qwen4_vision_norm(&h, b, w, bi, yv, eps));
}

pub fn rope(qkv: cuda.View, places: cuda.View) cuda.Error!void {
    return cuda.launched(zs_qwen4_vision_rope(&qkv, &places, rope_theta));
}

pub fn attention(qkv: cuda.View, tiles: cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_qwen4_vision_attention(&qkv, &tiles, &out));
}

/// A query tile's row of the attention's table: {first query, image start, image length, 0}.
pub const attention_tile = 128;

/// Each patch's place {row, column, height, width} in its frame, in merge-block order (2 x 2 blocks of the
/// grid, row-major within and between), `get_vision_position_ids` and the interpolation's indices.
pub fn writePlaces(grids: []const Grid, places: [][4]i32) void {
    var at: usize = 0;
    for (grids) |grid| {
        const blocks_w = grid.w / 2;
        for (0..grid.t) |_| {
            for (0..grid.h * grid.w) |within| {
                const in_col = within % 2;
                const in_row = (within / 2) % 2;
                const block_col = (within / 4) % blocks_w;
                const block_row = within / (4 * blocks_w);
                places[at] = .{ @intCast(block_row * 2 + in_row), @intCast(block_col * 2 + in_col), @intCast(grid.h), @intCast(grid.w) };
                at += 1;
            }
        }
    }
}

/// The attention's tiles for `grids`: each frame its own segment (`cu_seqlens`), cut from its start.
pub fn tileCount(grids: []const Grid) usize {
    var count: usize = 0;
    for (grids) |grid| count += grid.t * ((grid.h * grid.w + attention_tile - 1) / attention_tile);
    return count;
}

pub fn writeTiles(grids: []const Grid, tiles: [][4]i32) void {
    var start: usize = 0;
    var at: usize = 0;
    for (grids) |grid| {
        const n = grid.h * grid.w;
        for (0..grid.t) |_| {
            var q: usize = 0;
            while (q < n) : (q += attention_tile) {
                tiles[at] = .{ @intCast(start + q), @intCast(start), @intCast(n), 0 };
                at += 1;
            }
            start += n;
        }
    }
}

fn view2(buffer: cuda.Buffer, dtype: cuda.DType, rows: usize, cols: usize) cuda.View {
    std.debug.assert(rows * cols * dtype.size() <= buffer.bytes());
    return .init(buffer.view.ptr.?, dtype, &.{ rows, cols });
}

pub const Tower = struct {
    patch: Linear,
    table: cuda.Buffer,
    blocks: [depth]Block,
    merger_norm: Norm,
    merger_fc1: Linear,
    merger_fc2: Linear,
    max_patches: usize,

    /// The run's buffers, `max_patches` rows each: the residual stream, a norm's output (and the attention's),
    /// the wide one (qkv, fc1, the merger's fc1), a branch (the patches' embedding, proj, fc2).
    pixels: cuda.Buffer,
    /// zeros [1152]: the patches' product takes its bias after rounding (`embedBf16`)
    zero_bias: cuda.Buffer,
    h: cuda.Buffer,
    y: cuda.Buffer,
    wide: cuda.Buffer,
    branch: cuda.Buffer,
    places: cuda.Buffer,
    tiles: cuda.Buffer,
    /// Patches in the current run (`begin`).
    patches: usize = 0,
    tile_count: usize = 0,
    /// `branch` holds a residual not yet added to `h`.
    pending: bool = false,

    /// Uploads the tower from `checkpoint`'s `model.visual.` tensors as stored: each linear BF16, MXFP8 or
    /// NVFP4 by what its tensors are (`Format.of`), everything else BF16 (norms, biases, the patches' Conv3d,
    /// the position table). An EXL3 pack's tower (trellises, F16) is refused with `diagnostic` set.
    pub fn load(gpa: Allocator, arena: Allocator, checkpoint: *const safetensors.Checkpoint, max_patches: usize, diagnostic: *Diagnostic) Error!Tower {
        try check(arena, checkpoint, diagnostic);
        var ranges: std.ArrayList([]const u8) = .empty;
        defer ranges.deinit(gpa);
        var names = Names{};
        var it = names.iterator();
        while (it.next()) |entry| try ranges.append(gpa, checkpoint.get(entry.name).?.data);
        var sites = Sites{};
        while (sites.next()) |site| {
            var buffers: [3][128]u8 = undefined;
            for (tensorNames(&buffers, site.prefix)) |name| if (checkpoint.get(name)) |found| try ranges.append(gpa, found.data);
        }
        var held = try checkpoint.hold(gpa, ranges.items);
        defer held.release();

        var self: Tower = undefined;
        self.max_patches = max_patches;
        self.patches = 0;
        self.tile_count = 0;
        self.pending = false;
        var made: Made = .{};
        errdefer made.free();
        self.patch = .{
            .weight = .{ .bf16 = try tensor(checkpoint, "model.visual.patch_embed.proj.weight", &.{ hidden, patch_values }, &made) },
            .bias = try tensor(checkpoint, "model.visual.patch_embed.proj.bias", &.{hidden}, &made),
        };
        self.table = try tensor(checkpoint, "model.visual.pos_embed.weight", &.{ positions, hidden }, &made);
        var widest: usize = intermediate;
        for (&self.blocks, 0..) |*each, i| {
            var buffer: [64]u8 = undefined;
            const at = std.fmt.bufPrint(&buffer, "model.visual.blocks.{d}.", .{i}) catch unreachable;
            each.norm1 = try normOf(checkpoint, at, "norm1", &made);
            each.qkv = try linearAt(gpa, arena, checkpoint, at, "attn.qkv", 3 * hidden, hidden, 3 * hidden, &made, diagnostic);
            each.proj = try linearAt(gpa, arena, checkpoint, at, "attn.proj", hidden, hidden, hidden, &made, diagnostic);
            each.norm2 = try normOf(checkpoint, at, "norm2", &made);
            // a quantized fc2 reads K in 64s: fc1 then writes that many columns, the padded ones zero
            const fc2_format = Format.of(arena, checkpoint, std.fmt.bufPrint(&buffer, "model.visual.blocks.{d}.mlp.linear_fc2", .{i}) catch unreachable, hidden, intermediate, diagnostic) catch unreachable;
            each.inner = if (fc2_format == .bf16) intermediate else padded(intermediate);
            widest = @max(widest, each.inner);
            const at_again = std.fmt.bufPrint(&buffer, "model.visual.blocks.{d}.", .{i}) catch unreachable;
            each.fc1 = try linearAt(gpa, arena, checkpoint, at_again, "mlp.linear_fc1", intermediate, hidden, each.inner, &made, diagnostic);
            each.fc2 = try linearAt(gpa, arena, checkpoint, at_again, "mlp.linear_fc2", hidden, intermediate, hidden, &made, diagnostic);
        }
        self.merger_norm = try normOf(checkpoint, "model.visual.", "merger.norm", &made);
        self.merger_fc1 = try linearAt(gpa, arena, checkpoint, "model.visual.", "merger.linear_fc1", merged, merged, merged, &made, diagnostic);
        self.merger_fc2 = try linearAt(gpa, arena, checkpoint, "model.visual.", "merger.linear_fc2", out_hidden, merged, out_hidden, &made, diagnostic);

        self.pixels = try made.buffer(.bf16, &.{ max_patches, patch_values });
        self.zero_bias = try made.buffer(.bf16, &.{hidden});
        try self.zero_bias.zero();
        self.h = try made.buffer(.bf16, &.{ max_patches, hidden });
        self.y = try made.buffer(.bf16, &.{ max_patches, hidden });
        self.wide = try made.buffer(.bf16, &.{ max_patches, widest });
        self.branch = try made.buffer(.bf16, &.{ max_patches, hidden });
        self.places = try made.buffer(.i32, &.{ max_patches, 4 });
        // a frame of n patches has ceil(n / 128) tiles; frames of at least 4 patches (2 x 2 merge blocks)
        self.tiles = try made.buffer(.i32, &.{ max_patches / attention_tile + max_patches / 4 + 1, 4 });
        return self;
    }

    pub fn deinit(self: *Tower) void {
        self.patch.deinit();
        self.table.deinit();
        for (&self.blocks) |*each| {
            each.norm1.deinit();
            each.qkv.deinit();
            each.proj.deinit();
            each.norm2.deinit();
            each.fc1.deinit();
            each.fc2.deinit();
        }
        self.merger_norm.deinit();
        self.merger_fc1.deinit();
        self.merger_fc2.deinit();
        for ([_]*cuda.Buffer{ &self.pixels, &self.zero_bias, &self.h, &self.y, &self.wide, &self.branch, &self.places, &self.tiles }) |buffer| buffer.deinit();
        self.* = undefined;
    }

    /// Rows of the features `grids` give: a quarter of their patches.
    pub fn featureRows(grids: []const Grid) usize {
        var count: usize = 0;
        for (grids) |grid| count += grid.patches() / merge_unit;
        return count;
    }

    /// The features of every image of `grids`, `pixels` their `pixel_values` (f32 [patches, 1536], cast to
    /// bf16 as TensorFold does) into `out` bf16 [patches / 4, 2560]; in runs of whole images of at most
    /// `max_patches`. `gpa` holds a run's tables for the call.
    pub fn encode(self: *Tower, gpa: Allocator, pixels: []const f32, grids: []const Grid, out: cuda.View) Error!void {
        var total: usize = 0;
        for (grids) |grid| {
            if (grid.patches() > self.max_patches or grid.h % 2 != 0 or grid.w % 2 != 0 or grid.patches() == 0) return error.UnsupportedVision;
            total += grid.patches();
        }
        try cuda.bound(pixels.len == total * patch_values, "vision: {d} pixel values for {d} patches", .{ pixels.len, total });
        try cuda.bound(out.dtype == .bf16 and out.count() == total / merge_unit * out_hidden, "vision: features for {d} rows", .{total / merge_unit});
        var first: usize = 0;
        var at: usize = 0;
        var row: usize = 0;
        while (first < grids.len) {
            var last = first;
            var patches: usize = 0;
            while (last < grids.len and patches + grids[last].patches() <= self.max_patches) : (last += 1) patches += grids[last].patches();
            const run = grids[first..last];
            try self.begin(gpa, run);
            try self.embed(gpa, pixels[at * patch_values ..][0 .. patches * patch_values]);
            for (0..depth) |i| try self.block(i);
            var piece = out;
            piece.ptr = @ptrCast(@as([*]u8, @ptrCast(out.ptr.?)) + row * out_hidden * 2);
            piece.ndim = 2;
            piece.size = @splat(0);
            piece.size[0] = @intCast(patches / merge_unit);
            piece.size[1] = out_hidden;
            try self.merge(piece);
            at += patches;
            row += patches / merge_unit;
            first = last;
        }
    }

    /// Starts a run over `grids` (at most `max_patches` patches): their places and attention tiles uploaded.
    pub fn begin(self: *Tower, gpa: Allocator, grids: []const Grid) Error!void {
        var patches: usize = 0;
        for (grids) |grid| patches += grid.patches();
        try cuda.bound(patches >= 1 and patches <= self.max_patches, "vision: a run of {d} patches", .{patches});
        const places = try gpa.alloc([4]i32, patches);
        defer gpa.free(places);
        writePlaces(grids, places);
        try self.places.uploadAt(0, std.mem.sliceAsBytes(places));
        const count = tileCount(grids);
        try cuda.bound(count * 4 * 4 <= self.tiles.bytes(), "vision: {d} attention tiles", .{count});
        const tiles = try gpa.alloc([4]i32, count);
        defer gpa.free(tiles);
        writeTiles(grids, tiles);
        try self.tiles.uploadAt(0, std.mem.sliceAsBytes(tiles));
        self.patches = patches;
        self.tile_count = count;
        self.pending = false;
    }

    /// The run's patches embedded: `pixel_values` rounded to bf16, the patches' linear into `branch` (its
    /// product rounded, then its bias), then `h` = it plus the resampled position table.
    pub fn embed(self: *Tower, gpa: Allocator, pixels: []const f32) Error!void {
        try cuda.bound(pixels.len == self.patches * patch_values, "vision: {d} pixel values", .{pixels.len});
        const bits = try gpa.alloc(u16, pixels.len);
        defer gpa.free(bits);
        for (pixels, bits) |value, *bit| bit.* = numeric.bf16FromF32(value);
        try self.pixels.uploadAt(0, std.mem.sliceAsBytes(bits));
        try self.embedBf16();
    }

    /// `embed` from the pixels already in `pixels` as bf16.
    pub fn embedBf16(self: *Tower) Error!void {
        const p = self.patches;
        // the product rounded, then its bias added and rounded again by the positions' kernel, as Conv3d
        try cuda.launched(cuda.kernels.zs_b16_gemm_bias(&self.pixels.rows(p), &self.patch.weight.bf16.view, &self.zero_bias.view, &self.branch.rows(p), @backingInt(Act.none)));
        try cuda.launched(zs_qwen4_vision_pos(&self.branch.rows(p), &self.patch.bias.view, &self.table.view, &self.places.rows(p), &self.h.rows(p)));
        self.pending = false;
    }

    /// Block `i` on the run's `h`; its MLP's residual is left in `branch` for the next norm (`pending`).
    pub fn block(self: *Tower, i: usize) Error!void {
        const p = self.patches;
        const b = self.blocks[i];
        const h = self.h.rows(p);
        const y = self.y.rows(p);
        const branch = self.branch.rows(p);
        const qkv = view2(self.wide, .bf16, p, 3 * hidden);
        try norm(h, if (self.pending) branch else null, b.norm1, y);
        try b.qkv.forward(y, qkv, .none);
        try rope(qkv, self.places.rows(p));
        try attention(qkv, self.tiles.rows(self.tile_count), y);
        try b.proj.forward(y, branch, .none);
        try norm(h, branch, b.norm2, y);
        const up = view2(self.wide, .bf16, p, b.inner);
        try b.fc1.forward(y, up, .gelu_tanh);
        try b.fc2.forward(up, branch, .none);
        self.pending = true;
    }

    /// Adds a pending residual to `h`: the residual stream after the last block run, for a reader of `h`.
    pub fn settle(self: *Tower) Error!void {
        if (!self.pending) return;
        try norm(self.h.rows(self.patches), self.branch.rows(self.patches), null, null);
        self.pending = false;
    }

    /// The merger on the run's `h`: its LayerNorm into `y`, four patches a row, fc1 and exact GELU into `wide`,
    /// fc2 into `out` bf16 [patches / 4, 2560].
    pub fn merge(self: *Tower, out: cuda.View) Error!void {
        const p = self.patches;
        try norm(self.h.rows(p), if (self.pending) self.branch.rows(p) else null, self.merger_norm, self.y.rows(p));
        self.pending = false;
        const rows = p / merge_unit;
        const x = view2(self.y, .bf16, rows, merged);
        const up = view2(self.wide, .bf16, rows, merged);
        try self.merger_fc1.forward(x, up, .gelu_erf);
        try self.merger_fc2.forward(up, out, .none);
    }

    /// The tower's tensors that are BF16 in every checkpoint it takes: names and shapes (the linears' weights
    /// are `Sites`, by format).
    const Names = struct {
        fn iterator(_: *Names) Iterator {
            return .{};
        }
        const Entry = struct { name: []const u8, shape: [5]u64, ndim: usize };
        const Iterator = struct {
            index: usize = 0,
            buffer: [96]u8 = undefined,

            fn next(self: *Iterator) ?Entry {
                const fixed = [_]struct { []const u8, []const u64 }{
                    .{ "model.visual.patch_embed.proj.weight", &.{ hidden, 3, 2, 16, 16 } },
                    .{ "model.visual.patch_embed.proj.bias", &.{hidden} },
                    .{ "model.visual.pos_embed.weight", &.{ positions, hidden } },
                    .{ "model.visual.merger.norm.weight", &.{hidden} },
                    .{ "model.visual.merger.norm.bias", &.{hidden} },
                    .{ "model.visual.merger.linear_fc1.bias", &.{merged} },
                    .{ "model.visual.merger.linear_fc2.bias", &.{out_hidden} },
                };
                const per_block = [_]struct { []const u8, []const u64 }{
                    .{ "norm1.weight", &.{hidden} },              .{ "norm1.bias", &.{hidden} },
                    .{ "norm2.weight", &.{hidden} },              .{ "norm2.bias", &.{hidden} },
                    .{ "attn.qkv.bias", &.{3 * hidden} },         .{ "attn.proj.bias", &.{hidden} },
                    .{ "mlp.linear_fc1.bias", &.{intermediate} }, .{ "mlp.linear_fc2.bias", &.{hidden} },
                };
                const i = self.index;
                if (i >= fixed.len + depth * per_block.len) return null;
                self.index += 1;
                const name, const shape = if (i < fixed.len) .{ fixed[i][0], fixed[i][1] } else blk: {
                    const j = i - fixed.len;
                    const part = per_block[j % per_block.len];
                    const name = std.fmt.bufPrint(&self.buffer, "model.visual.blocks.{d}.{s}", .{ j / per_block.len, part[0] }) catch unreachable;
                    break :blk .{ name, part[1] };
                };
                var entry: Entry = .{ .name = name, .shape = @splat(0), .ndim = shape.len };
                @memcpy(entry.shape[0..shape.len], shape);
                return entry;
            }
        };
    };

    /// The tower's linears (a prefix and its weight's [n, k]): four a block, then the merger's two.
    const Sites = struct {
        index: usize = 0,
        buffer: [96]u8 = undefined,

        const Site = struct { prefix: []const u8, n: usize, k: usize };

        fn next(self: *Sites) ?Site {
            const per_block = [_]Site{
                .{ .prefix = "attn.qkv", .n = 3 * hidden, .k = hidden },
                .{ .prefix = "attn.proj", .n = hidden, .k = hidden },
                .{ .prefix = "mlp.linear_fc1", .n = intermediate, .k = hidden },
                .{ .prefix = "mlp.linear_fc2", .n = hidden, .k = intermediate },
            };
            const merger = [_]Site{
                .{ .prefix = "model.visual.merger.linear_fc1", .n = merged, .k = merged },
                .{ .prefix = "model.visual.merger.linear_fc2", .n = out_hidden, .k = merged },
            };
            const i = self.index;
            if (i >= depth * per_block.len + merger.len) return null;
            self.index += 1;
            if (i >= depth * per_block.len) return merger[i - depth * per_block.len];
            var site = per_block[i % per_block.len];
            site.prefix = std.fmt.bufPrint(&self.buffer, "model.visual.blocks.{d}.{s}", .{ i / per_block.len, site.prefix }) catch unreachable;
            return site;
        }
    };

    /// The tensors the NVFP4 checkpoint's BF16 tower stores: 9 of its own and 12 a block, 333.
    pub const tensor_count = 9 + depth * 12;

    fn check(arena: Allocator, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic) error{UnsupportedVision}!void {
        if (checkpoint.get("model.visual.blocks.0.attn.proj.trellis") != null or checkpoint.get("model.visual.blocks.0.mlp.linear_fc1.trellis") != null) {
            diagnostic.set(arena, "this checkpoint's vision tower is quantized as EXL3 (trellises, F16 norms): zspark runs the tower from BF16, MXFP8 or NVFP4 linears as stored; images are refused on a pack", .{});
            return error.UnsupportedVision;
        }
        var names = Names{};
        var it = names.iterator();
        while (it.next()) |entry| {
            const found = checkpoint.get(entry.name) orelse {
                diagnostic.set(arena, "the vision tower lacks {s}: no image can be read with this checkpoint", .{entry.name});
                return error.UnsupportedVision;
            };
            if (found.dtype != .BF16) {
                diagnostic.set(arena, "the vision tower's {s} is {t}, not BF16: zspark reads the tower's norms, biases, Conv3d and position table as BF16, as stored", .{ entry.name, found.dtype });
                return error.UnsupportedVision;
            }
            if (!std.mem.eql(u64, found.shape, entry.shape[0..entry.ndim])) {
                diagnostic.set(arena, "the vision tower's {s} has shape {any}, not {any}", .{ entry.name, found.shape, entry.shape[0..entry.ndim] });
                return error.UnsupportedVision;
            }
        }
        var sites = Sites{};
        var fc1: Format = .bf16;
        while (sites.next()) |site| {
            const format = try Format.of(arena, checkpoint, site.prefix, site.n, site.k, diagnostic);
            // fc1 writes the columns a quantized fc2 reads past 4,304 as zeros: a BF16 fc1 writes 4,304 alone
            if (std.mem.endsWith(u8, site.prefix, "mlp.linear_fc1")) fc1 = format;
            if (std.mem.endsWith(u8, site.prefix, "mlp.linear_fc2") and format != .bf16 and fc1 == .bf16 and site.k != padded(site.k)) {
                diagnostic.set(arena, "the vision tower's {s} is {t} with a BF16 fc1: zspark takes a quantized fc2 of K {d} only after a quantized fc1", .{ site.prefix, format, site.k });
                return error.UnsupportedVision;
            }
        }
    }

    /// What a load has made so far, freed in reverse if a later step fails.
    const Made = struct {
        list: [tensor_count + 8]Owned = undefined,
        count: usize = 0,

        const Owned = union(enum) { buffer: cuda.Buffer, mxfp8: nvfp4.Mx8Linear, nvfp4: nvfp4.Fp4Linear };

        fn buffer(self: *Made, dtype: cuda.DType, shape: []const usize) cuda.Error!cuda.Buffer {
            const made: cuda.Buffer = try .init(dtype, shape);
            self.list[self.count] = .{ .buffer = made };
            self.count += 1;
            return made;
        }

        fn add(self: *Made, owned: Owned) void {
            self.list[self.count] = owned;
            self.count += 1;
        }

        fn free(self: *Made) void {
            var i = self.count;
            while (i > 0) {
                i -= 1;
                switch (self.list[i]) {
                    inline else => |*each| each.deinit(),
                }
            }
        }
    };

    fn tensor(checkpoint: *const safetensors.Checkpoint, name: []const u8, shape: []const usize, made: *Made) cuda.Error!cuda.Buffer {
        const found = checkpoint.get(name).?;
        const buffer = try made.buffer(.bf16, shape);
        try buffer.upload(found.data);
        return buffer;
    }

    /// The linear at `at ++ part`, its weight [n, k] as stored (`Format.of`), writing `width` columns: n, or
    /// past it up to the padding a quantized weight has (zero codes, a zero bias).
    fn linearAt(gpa: Allocator, arena: Allocator, checkpoint: *const safetensors.Checkpoint, at: []const u8, part: []const u8, n: usize, k: usize, width: usize, made: *Made, diagnostic: *Diagnostic) Error!Linear {
        var buffer: [96]u8 = undefined;
        const prefix = std.fmt.bufPrint(&buffer, "{s}{s}", .{ at, part }) catch unreachable;
        var buffers: [3][128]u8 = undefined;
        const names = tensorNames(&buffers, prefix);
        const format = try Format.of(arena, checkpoint, prefix, n, k, diagnostic);
        var b: [96]u8 = undefined;
        const bias_name = std.fmt.bufPrint(&b, "{s}.bias", .{prefix}) catch unreachable;
        const weight: Linear.Weight = switch (format) {
            .bf16 => blk: {
                std.debug.assert(width == n);
                break :blk .{ .bf16 = try tensor(checkpoint, names[0], &.{ n, k }, made) };
            },
            .mxfp8 => blk: {
                const kp = padded(k);
                const codes = try padRows(gpa, checkpoint.get(names[0]).?.data, n, k, kp, 0);
                defer gpa.free(codes);
                const exponents = try padRows(gpa, checkpoint.get(names[1]).?.data, n, k / nvfp4.mx8_block, kp / nvfp4.mx8_block, 127);
                defer gpa.free(exponents);
                const linear = nvfp4.Mx8Linear.init(gpa, codes, exponents, n, kp) catch |err| switch (err) {
                    error.UnsupportedShape => return error.UnsupportedVision,
                    else => |e| return e,
                };
                made.add(.{ .mxfp8 = linear });
                break :blk .{ .mxfp8 = linear };
            },
            .nvfp4 => blk: {
                const kp = padded(k);
                const codes = try padRows(gpa, checkpoint.get(names[0]).?.data, n, k / 2, kp / 2, 0);
                defer gpa.free(codes);
                const scales = try padRows(gpa, checkpoint.get(names[1]).?.data, n, k / 16, kp / 16, 0);
                defer gpa.free(scales);
                const scale: f32 = @bitCast(std.mem.readInt(u32, checkpoint.get(names[2]).?.data[0..4], .little));
                const linear = nvfp4.Fp4Linear.init(gpa, codes, scales, n, kp, scale) catch |err| switch (err) {
                    error.UnsupportedShape => return error.UnsupportedVision,
                    else => |e| return e,
                };
                made.add(.{ .nvfp4 = linear });
                break :blk .{ .nvfp4 = linear };
            },
        };
        const bias = try made.buffer(.bf16, &.{width});
        if (width > n) try bias.zero();
        try bias.uploadAt(0, checkpoint.get(bias_name).?.data);
        return .{ .weight = weight, .bias = bias };
    }

    /// Rows of `width` bytes from `data` [rows, width] widened to `wider` with `fill`.
    fn padRows(gpa: Allocator, data: []const u8, rows: usize, width: usize, wider: usize, fill: u8) Allocator.Error![]u8 {
        std.debug.assert(data.len == rows * width);
        const out = try gpa.alloc(u8, rows * wider);
        @memset(out, fill);
        for (0..rows) |r| @memcpy(out[r * wider ..][0..width], data[r * width ..][0..width]);
        return out;
    }

    fn normOf(checkpoint: *const safetensors.Checkpoint, at: []const u8, part: []const u8, made: *Made) cuda.Error!Norm {
        var w: [96]u8 = undefined;
        var b: [96]u8 = undefined;
        return .{
            .weight = try tensor(checkpoint, std.fmt.bufPrint(&w, "{s}{s}.weight", .{ at, part }) catch unreachable, &.{hidden}, made),
            .bias = try tensor(checkpoint, std.fmt.bufPrint(&b, "{s}{s}.bias", .{ at, part }) catch unreachable, &.{hidden}, made),
        };
    }
};

/// K rounded up to the quantized kernel's groups of 64 inputs.
fn padded(k: usize) usize {
    return (k + 63) / 64 * 64;
}
