//! Unlimited-OCR's CLIP-L on SAM's features (spec 137 WS-03; the checkpoint's `deepencoder.py`, `VitModel` with
//! `CLIPVisionEmbeddings` given `patch_embeds`): the class token and SAM's features plus the position table (its
//! 16 x 16 grid resampled to a crop's 10 x 10), `pre_layrnorm`, then 24 pre-LN layers of 16 heads of 64 with
//! quick_gelu, the residual stream fp32 as autocast leaves it. Then the projector: the concat of CLIP's rows past
//! the class token and SAM's features, a linear with its bias to the language model's 1280.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const vw = @import("vision_weights.zig");
const tables = @import("vision_tables.zig");

const Error = vw.Error;
const Linear = vw.Linear;
const Norm = vw.Norm;
const Size = vw.Size;
const shaped = vw.shaped;

pub const hidden = 1024;
pub const depth = 24;
pub const inner = 4096;
pub const lm_hidden = 1280;
/// the stored position table's grid (CLIP-L at 224, patches of 14)
const stored_grid = 16;
const eps = 1e-5;

const Layer = struct {
    norm1: Norm,
    qkv: Linear,
    out: Linear,
    norm2: Norm,
    fc1: Linear,
    fc2: Linear,
};

pub const Clip = struct {
    cls: cuda.Buffer,
    /// bf16 [1 + 256, 1024] as stored, and a crop's [1 + 100, 1024]
    pos_global: cuda.View,
    pos_crop: cuda.View,
    pre: Norm,
    layers: [depth]Layer,
    projector: Linear,

    pub fn load(gpa: Allocator, source: vw.Source, made: *vw.Made) Error!Clip {
        var self: Clip = undefined;
        self.cls = try source.upload(made, "model.vision_model.embeddings.class_embedding", &.{hidden});
        const positions = 1 + stored_grid * stored_grid;
        const pos = try source.upload(made, "model.vision_model.embeddings.position_embedding.weight", &.{ positions, hidden });
        self.pos_global = pos.view;
        self.pos_crop = try resampled(gpa, source, made, Size.crop.out());
        self.pre = try source.norm(made, "model.vision_model.pre_layrnorm", hidden);
        for (&self.layers, 0..) |*l, i| {
            var buffer: [96]u8 = undefined;
            l.norm1 = try source.norm(made, at(&buffer, i, "layer_norm1"), hidden);
            l.qkv = try source.linear(made, at(&buffer, i, "self_attn.qkv_proj"), 3 * hidden, hidden);
            l.out = try source.linear(made, at(&buffer, i, "self_attn.out_proj"), hidden, hidden);
            l.norm2 = try source.norm(made, at(&buffer, i, "layer_norm2"), hidden);
            l.fc1 = try source.linear(made, at(&buffer, i, "mlp.fc1"), inner, hidden);
            l.fc2 = try source.linear(made, at(&buffer, i, "mlp.fc2"), hidden, inner);
        }
        self.projector = try source.linear(made, "model.projector.layers", lm_hidden, 2 * hidden);
        return self;
    }

    /// `get_abs_pos`: the class row as stored, the 16 x 16 grid by PyTorch's antialiased bicubic to `grid` in
    /// fp32, then bf16.
    fn resampled(gpa: Allocator, source: vw.Source, made: *vw.Made, grid: usize) Error!cuda.View {
        const positions = 1 + stored_grid * stored_grid;
        const stored = (try source.get("model.vision_model.embeddings.position_embedding.weight", &.{ positions, hidden })).data;
        const wide = try tables.widen(gpa, stored[2 * hidden ..]);
        defer gpa.free(wide);
        const resized = try tables.bicubicAntialias(gpa, wide, stored_grid, grid, hidden);
        defer gpa.free(resized);
        const bits = try tables.narrow(gpa, resized);
        defer gpa.free(bits);
        const buffer = try made.buffer(.bf16, &.{ 1 + grid * grid, hidden });
        try buffer.uploadAt(0, stored[0 .. 2 * hidden]);
        try buffer.uploadAt(2 * hidden, std.mem.sliceAsBytes(bits));
        return buffer.view;
    }

    pub fn posOf(self: *const Clip, size: Size) cuda.View {
        return if (size.side == Size.global.side) self.pos_global else self.pos_crop;
    }

    /// CLIP over `views` views of `size` from SAM's features in `work.sam` to `work.clip`, fp32 [views * (1 +
    /// out tokens), 1024]. `segments`: int32 tiles, a view one segment of 1 + out tokens rows.
    pub fn forward(self: *const Clip, work: vw.Work, segments: cuda.View, size: Size, views: usize) Error!void {
        const og = size.outTokens();
        const rows = views * (1 + og);
        const h = shaped(work.clip, .f32, &.{ rows, hidden });
        try vw.clipEmbed(shaped(work.sam, .bf16, &.{ views * og, hidden }), self.cls.view, self.posOf(size), self.pre, h, eps);
        const y = shaped(work.y, .bf16, &.{ rows, hidden });
        const branch = shaped(work.branch, .bf16, &.{ rows, hidden });
        var pending = false;
        for (&self.layers) |*l| {
            try vw.norm(h, .{ .branch = if (pending) branch else null, .by = l.norm1, .y = y, .eps = eps });
            const qkv = shaped(work.wide, .bf16, &.{ rows, 3 * hidden });
            try l.qkv.forward(y, qkv, .none);
            try vw.attention(qkv, null, segments, null, y);
            try l.out.forward(y, branch, .none);
            try vw.norm(h, .{ .branch = branch, .by = l.norm2, .y = y, .eps = eps });
            const wide = shaped(work.wide, .bf16, &.{ rows, inner });
            try l.fc1.forward(y, wide, .quick_gelu);
            try l.fc2.forward(wide, branch, .none);
            pending = true;
        }
        try vw.norm(h, .{ .branch = branch });
    }

    /// The projector over `views` views: x = bf16(cat(CLIP's rows past the class token, SAM's features)) into
    /// `work.wide`, then out bf16 [views * out tokens, 1280] into `work.branch`.
    pub fn project(self: *const Clip, work: vw.Work, size: Size, views: usize) Error!cuda.View {
        const og = size.outTokens();
        const x = shaped(work.wide, .bf16, &.{ views * og, 2 * hidden });
        try vw.concat(shaped(work.clip, .f32, &.{ views * (1 + og), hidden }), shaped(work.sam, .bf16, &.{ views * og, hidden }), x);
        return self.projectFrom(work, x);
    }

    /// The projector's linear alone on x bf16 [rows, 2048] (a test gives it the reference's concat).
    pub fn projectFrom(self: *const Clip, work: vw.Work, x: cuda.View) Error!cuda.View {
        const out = shaped(work.branch, .bf16, &.{ @intCast(x.size[0]), lm_hidden });
        try self.projector.forward(x, out, .none);
        return out;
    }
};

fn at(buffer: []u8, layer: usize, what: []const u8) []const u8 {
    return std.fmt.bufPrint(buffer, "model.vision_model.transformer.layers.{d}.{s}", .{ layer, what }) catch unreachable;
}
