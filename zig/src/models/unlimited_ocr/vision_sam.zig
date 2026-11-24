//! Unlimited-OCR's SAM ViT-B (spec 137 WS-03; the checkpoint's `deepencoder.py`, `ImageEncoderViT` with its
//! neck and `net_2`, `net_3`): the 16 x 16 patches as a GEMM over their columns, the position table, 12 blocks of
//! 12 heads (14 x 14 windows but blocks 2, 5, 8, 11, which attend over the whole view, each with decomposed
//! relative positions), then a 1 x 1 conv, LayerNorm2d, a 3 x 3 conv, LayerNorm2d and two 3 x 3 convs of stride 2
//! to 1024 channels at a sixteenth of the patches. Views of one size run together, a row's bits its own.

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

pub const hidden = 768;
pub const heads = 12;
pub const depth = 12;
pub const inner = 3072;
pub const neck_width = 256;
pub const features = 1024;
/// the blocks that attend over the whole view (`global_attn_indexes`)
pub const global_blocks = [_]usize{ 2, 5, 8, 11 };
/// the stored position table's side (a 1024 view's)
const stored_grid = 64;

fn at(buffer: []u8, block: usize, what: []const u8) []const u8 {
    return std.fmt.bufPrint(buffer, "model.sam_model.blocks.{d}.{s}", .{ block, what }) catch unreachable;
}

pub fn isGlobal(block: usize) bool {
    return std.mem.indexOfScalar(usize, &global_blocks, block) != null;
}

const Block = struct {
    norm1: Norm,
    qkv: Linear,
    proj: Linear,
    norm2: Norm,
    lin1: Linear,
    lin2: Linear,
    /// a window block's tables, bf16 [27, 64]; a global block's as stored [127, 64] (a 1024 view's)
    rel_h: cuda.Buffer,
    rel_w: cuda.Buffer,
};

/// A size's tables: its position table and its global blocks' relative-position tables, resampled where the
/// size is not the stored one's.
pub const SizeTables = struct {
    pos: cuda.View,
    rel_h: [global_blocks.len]cuda.View,
    rel_w: [global_blocks.len]cuda.View,
};

pub const Sam = struct {
    patch: Linear,
    patch_bias: cuda.Buffer,
    blocks: [depth]Block,
    neck0: Linear,
    neck1: Norm,
    neck2: Linear,
    neck3: Norm,
    net2: Linear,
    net3: Linear,
    global: SizeTables,
    crop: SizeTables,

    pub fn load(gpa: Allocator, source: vw.Source, made: *vw.Made, zeros: cuda.View) Error!Sam {
        var self: Sam = undefined;
        const patch = try source.upload(made, "model.sam_model.patch_embed.proj.weight", &.{ hidden, 3, 16, 16 });
        self.patch = .{ .weight = shaped(patch, .bf16, &.{ hidden, hidden }), .bias = zeros.span(0, hidden) };
        self.patch_bias = try source.upload(made, "model.sam_model.patch_embed.proj.bias", &.{hidden});
        const pos = try source.upload(made, "model.sam_model.pos_embed", &.{ 1, stored_grid, stored_grid, hidden });
        self.global.pos = shaped(pos, .bf16, &.{ stored_grid * stored_grid, hidden });
        for (&self.blocks, 0..) |*b, i| {
            var buffer: [96]u8 = undefined;
            b.norm1 = try source.norm(made, at(&buffer, i, "norm1"), hidden);
            b.qkv = try source.linear(made, at(&buffer, i, "attn.qkv"), 3 * hidden, hidden);
            b.proj = try source.linear(made, at(&buffer, i, "attn.proj"), hidden, hidden);
            b.norm2 = try source.norm(made, at(&buffer, i, "norm2"), hidden);
            b.lin1 = try source.linear(made, at(&buffer, i, "mlp.lin1"), inner, hidden);
            b.lin2 = try source.linear(made, at(&buffer, i, "mlp.lin2"), hidden, inner);
            const span: usize = if (isGlobal(i)) 2 * stored_grid - 1 else 2 * 14 - 1;
            b.rel_h = try source.upload(made, at(&buffer, i, "attn.rel_pos_h"), &.{ span, 64 });
            b.rel_w = try source.upload(made, at(&buffer, i, "attn.rel_pos_w"), &.{ span, 64 });
        }
        const neck0 = try source.upload(made, "model.sam_model.neck.0.weight", &.{ neck_width, hidden, 1, 1 });
        self.neck0 = .{ .weight = shaped(neck0, .bf16, &.{ neck_width, hidden }), .bias = zeros.span(0, neck_width) };
        self.neck1 = try source.norm(made, "model.sam_model.neck.1", neck_width);
        self.neck2 = try source.conv3(gpa, made, "model.sam_model.neck.2.weight", neck_width, neck_width, zeros);
        self.neck3 = try source.norm(made, "model.sam_model.neck.3", neck_width);
        self.net2 = try source.conv3(gpa, made, "model.sam_model.net_2.weight", 512, neck_width, zeros);
        self.net3 = try source.conv3(gpa, made, "model.sam_model.net_3.weight", features, 512, zeros);

        for (global_blocks, 0..) |block, g| {
            self.global.rel_h[g] = self.blocks[block].rel_h.view;
            self.global.rel_w[g] = self.blocks[block].rel_w.view;
        }
        self.crop = try resampled(gpa, source, made, Size.crop);
        return self;
    }

    /// The tables of a view whose grid is not the stored 64: the position table by PyTorch's antialiased bicubic
    /// (`get_abs_pos_sam`), each global block's relative positions by its linear interpolation (`get_rel_pos`),
    /// both in fp32 from the stored bf16, then rounded to bf16.
    fn resampled(gpa: Allocator, source: vw.Source, made: *vw.Made, size: Size) Error!SizeTables {
        const g = size.grid();
        var out: SizeTables = undefined;
        {
            const stored = (try source.get("model.sam_model.pos_embed", &.{ 1, stored_grid, stored_grid, hidden })).data;
            const wide = try tables.widen(gpa, stored);
            defer gpa.free(wide);
            const resized = try tables.bicubicAntialias(gpa, wide, stored_grid, g, hidden);
            defer gpa.free(resized);
            const bits = try tables.narrow(gpa, resized);
            defer gpa.free(bits);
            const buffer = try made.buffer(.bf16, &.{ g * g, hidden });
            try buffer.upload(std.mem.sliceAsBytes(bits));
            out.pos = buffer.view;
        }
        for (global_blocks, 0..) |block, i| {
            var name: [96]u8 = undefined;
            inline for (.{ "rel_pos_h", "rel_pos_w" }, 0..) |what, which| {
                const full = std.fmt.bufPrint(&name, "model.sam_model.blocks.{d}.attn.{s}", .{ block, what }) catch unreachable;
                const stored = (try source.get(full, &.{ 2 * stored_grid - 1, 64 })).data;
                const wide = try tables.widen(gpa, stored);
                defer gpa.free(wide);
                const resized = try tables.linear(gpa, wide, 2 * stored_grid - 1, 2 * g - 1, 64);
                defer gpa.free(resized);
                const bits = try tables.narrow(gpa, resized);
                defer gpa.free(bits);
                const buffer = try made.buffer(.bf16, &.{ 2 * g - 1, 64 });
                try buffer.upload(std.mem.sliceAsBytes(bits));
                if (which == 0) out.rel_h[i] = buffer.view else out.rel_w[i] = buffer.view;
            }
        }
        return out;
    }

    pub fn tablesOf(self: *const Sam, size: Size) *const SizeTables {
        return if (size.side == Size.global.side) &self.global else &self.crop;
    }

    /// `views` views of `size` in `work.pixels` to SAM's features in `work.sam`, bf16 [views * out tokens, 1024]
    /// (channels last, the order of `flatten(2).permute(0, 2, 1)`). `layout` holds the windows' maps and the
    /// attention's tiles for `views` views of the size.
    pub fn forward(self: *const Sam, work: vw.Work, layout: Layout, size: Size, views: usize) Error!void {
        const g = size.grid();
        const n = views * size.tokens();
        const rw = views * size.windowRows();
        const t = self.tablesOf(size);

        const h = shaped(work.h, .bf16, &.{ n, hidden });
        {
            const cols = shaped(work.wide, .bf16, &.{ n, hidden });
            try vw.patches(shaped(work.pixels, .bf16, &.{ views, 3, size.side, size.side }), cols);
            try self.patch.forward(cols, h, .none);
            try vw.samPos(h, self.patch_bias.view, t.pos);
        }
        const flat_y = shaped(work.y, .bf16, &.{ n, hidden });
        var g_index: usize = 0;
        for (&self.blocks, 0..) |*b, i| {
            // every block's linears run on the tokens alone; a window block's attention reads them through
            // `to_window`, its padding (`window_partition`'s zero rows) qkv's bias, and stores the tokens' rows.
            // proj and lin2 add their output to h in their epilogue (spec 142 WS-08), the bits of a linear and
            // then the add.
            const global = isGlobal(i);
            try vw.norm(h, .{ .by = b.norm1, .y = flat_y });
            const qkv = shaped(work.wide, .bf16, &.{ n, 3 * hidden });
            try b.qkv.forward(flat_y, qkv, .none);
            if (global) {
                const rel = shaped(work.rel, .bf16, &.{ heads, n, 2 * g });
                try vw.relpos(qkv, null, t.rel_h[g_index], t.rel_w[g_index], rel, size.tokens(), g);
                try vw.attention(qkv, null, layout.tiles_global, rel, flat_y);
                g_index += 1;
            } else {
                const rel = shaped(work.rel, .bf16, &.{ heads, rw, 28 });
                try vw.relpos(qkv, layout.to_window, b.rel_h.view, b.rel_w.view, rel, 196, 14);
                try vw.attention(qkv, .{ .map = layout.to_window, .pad = b.qkv.bias }, layout.tiles_window, rel, flat_y);
            }
            try b.proj.residual(flat_y, h);
            try vw.norm(h, .{ .by = b.norm2, .y = flat_y });
            const wide = shaped(work.wide, .bf16, &.{ n, inner });
            try b.lin1.forward(flat_y, wide, .gelu_erf);
            try b.lin2.residual(wide, h);
        }

        // the neck, channels last: a 1 x 1 conv, LayerNorm2d, a 3 x 3 conv, LayerNorm2d; then net_2 and net_3
        const n0 = shaped(work.y, .bf16, &.{ n, neck_width });
        const y0 = shaped(work.branch, .bf16, &.{ n, neck_width });
        try self.neck0.forward(h, n0, .none);
        try vw.norm(n0, .{ .by = self.neck1, .y = y0, .quirk = true });
        const cols1 = shaped(work.wide, .bf16, &.{ n, 9 * neck_width });
        try vw.im2col(y0, cols1, views, g, g, 1);
        try self.neck2.forward(cols1, n0, .none);
        try vw.norm(n0, .{ .by = self.neck3, .y = y0, .quirk = true });
        const half = g / 2;
        const cols2 = shaped(work.wide, .bf16, &.{ views * half * half, 9 * neck_width });
        try vw.im2col(y0, cols2, views, g, g, 2);
        const n2 = shaped(work.y, .bf16, &.{ views * half * half, 512 });
        try self.net2.forward(cols2, n2, .none);
        const cols3 = shaped(work.wide, .bf16, &.{ views * size.outTokens(), 9 * 512 });
        try vw.im2col(n2, cols3, views, half, half, 2);
        try self.net3.forward(cols3, shaped(work.sam, .bf16, &.{ views * size.outTokens(), features }), .none);
    }
};

/// A size's windows and tiles for some views (prefixes of the tables built for the run's most views).
pub const Layout = struct {
    /// int32 [views * window rows]: a window row's token, -1 for padding
    to_window: cuda.View,
    /// int32 [views * windows * 2, 4]: each 196-row window in two tiles of 128
    tiles_window: cuda.View,
    /// int32 [views * ceil(tokens / 128), 4]: each view one segment
    tiles_global: cuda.View,
};

/// The host's maps for `views` views of `size`, in the order of `window_partition`: window (b, wy, wx), its
/// rows (ly, lx), the padded positions past the grid -1.
pub fn windowMaps(gpa: Allocator, size: Size, views: usize) Allocator.Error![]i32 {
    const g = size.grid();
    const nw = size.windows();
    const to = try gpa.alloc(i32, views * size.windowRows());
    var row: usize = 0;
    for (0..views) |b| for (0..nw) |wy| for (0..nw) |wx| for (0..14) |ly| for (0..14) |lx| {
        const py = wy * 14 + ly;
        const px = wx * 14 + lx;
        if (py < g and px < g) {
            const token = b * g * g + py * g + px;
            to[row] = @intCast(token);
        } else to[row] = -1;
        row += 1;
    };
    return to;
}

/// Tiles of 128 queries over `segments` segments of `len` rows each, `width` wide (0: no bias), in order.
pub fn tiles(gpa: Allocator, segments: usize, len: usize, width: usize) Allocator.Error![]i32 {
    const per = (len + 127) / 128;
    const out = try gpa.alloc(i32, segments * per * 4);
    for (0..segments) |s| for (0..per) |k| {
        const slot = (s * per + k) * 4;
        out[slot] = @intCast(s * len + k * 128);
        out[slot + 1] = @intCast(s * len);
        out[slot + 2] = @intCast(len);
        out[slot + 3] = @intCast(width);
    };
    return out;
}
