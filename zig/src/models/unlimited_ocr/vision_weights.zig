//! What Unlimited-OCR's SAM and CLIP share (spec 137 WS-03): the launchers of `kernels/unlimited_ocr/vision*.cu`
//! and `kernels/bf16/gemm_bias.cu`, a BF16 linear with its bias, a LayerNorm's weight and bias, and the upload of
//! a checkpoint tensor as stored (BF16, its shape checked, a refusal in words otherwise).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const safetensors = @import("../../weights/safetensors.zig");

pub const Error = error{UnsupportedVision} || cuda.Error || Allocator.Error;

/// What a refusal found, in words.
pub const Diagnostic = @import("../../weights/checkpoint_reader.zig").Diagnostic;

/// The epilogue's activation (`gemm_bias.cu`).
pub const Act = enum(i64) { none = 0, gelu_tanh = 1, gelu_erf = 2, quick_gelu = 3 };

extern fn zs_b16_gemm_bias_residual(x: *const cuda.View, w: *const cuda.View, bias: *const cuda.View, h: *const cuda.View) c_int;
extern fn zs_unlimited_ocr_vision_patches(pixels: *const cuda.View, cols: *const cuda.View) c_int;
extern fn zs_unlimited_ocr_vision_im2col(x: *const cuda.View, cols: *const cuda.View, batch: i64, h: i64, w: i64, stride: i64) c_int;
extern fn zs_unlimited_ocr_vision_sam_pos(h: *const cuda.View, bias: *const cuda.View, pos: *const cuda.View) c_int;
extern fn zs_unlimited_ocr_vision_norm(h: *const cuda.View, branch: ?*const cuda.View, weight: ?*const cuda.View, bias: ?*const cuda.View, y: ?*const cuda.View, eps: f64, quirk: i64) c_int;
extern fn zs_unlimited_ocr_vision_clip_embed(sam: *const cuda.View, cls: *const cuda.View, pos: *const cuda.View, weight: *const cuda.View, bias: *const cuda.View, h: *const cuda.View, eps: f64) c_int;
extern fn zs_unlimited_ocr_vision_concat(clip: *const cuda.View, sam: *const cuda.View, x: *const cuda.View) c_int;
extern fn zs_unlimited_ocr_vision_place(src: ?*const cuda.View, from: *const cuda.View, to: *const cuda.View, newline: *const cuda.View, separator: *const cuda.View, out: *const cuda.View) c_int;
extern fn zs_unlimited_ocr_vision_attention(qkv: *const cuda.View, map: ?*const cuda.View, pad: ?*const cuda.View, tiles: *const cuda.View, rel: ?*const cuda.View, out: *const cuda.View) c_int;
extern fn zs_unlimited_ocr_vision_relpos(qkv: *const cuda.View, map: ?*const cuda.View, table_h: *const cuda.View, table_w: *const cuda.View, rel: *const cuda.View, len: i64, width: i64) c_int;

fn opt(v: *const ?cuda.View) ?*const cuda.View {
    return if (v.*) |*x| x else null;
}

pub fn patches(pixels: cuda.View, cols: cuda.View) cuda.Error!void {
    return cuda.launched(zs_unlimited_ocr_vision_patches(&pixels, &cols));
}

pub fn im2col(x: cuda.View, cols: cuda.View, batch: usize, h: usize, w: usize, stride: usize) cuda.Error!void {
    return cuda.launched(zs_unlimited_ocr_vision_im2col(&x, &cols, @intCast(batch), @intCast(h), @intCast(w), @intCast(stride)));
}

pub fn samPos(h: cuda.View, bias: cuda.View, pos: cuda.View) cuda.Error!void {
    return cuda.launched(zs_unlimited_ocr_vision_sam_pos(&h, &bias, &pos));
}

/// h += branch when `branch`, then y = LayerNorm(h) when `by` and `y`, a row a token (`vision.cu`).
pub const NormArgs = struct {
    branch: ?cuda.View = null,
    by: ?Norm = null,
    y: ?cuda.View = null,
    eps: f64 = 1e-6,
    /// the neck's LayerNorm2d
    quirk: bool = false,
};

pub fn norm(h: cuda.View, args: NormArgs) cuda.Error!void {
    const w: ?cuda.View = if (args.by) |n| n.weight.view else null;
    const b: ?cuda.View = if (args.by) |n| n.bias.view else null;
    return cuda.launched(zs_unlimited_ocr_vision_norm(&h, opt(&args.branch), opt(&w), opt(&b), opt(&args.y), args.eps, @intFromBool(args.quirk)));
}

pub fn clipEmbed(sam: cuda.View, cls: cuda.View, pos: cuda.View, by: Norm, h: cuda.View, eps: f64) cuda.Error!void {
    return cuda.launched(zs_unlimited_ocr_vision_clip_embed(&sam, &cls, &pos, &by.weight.view, &by.bias.view, &h, eps));
}

pub fn concat(clip: cuda.View, sam: cuda.View, x: cuda.View) cuda.Error!void {
    return cuda.launched(zs_unlimited_ocr_vision_concat(&clip, &sam, &x));
}

pub fn place(src: ?cuda.View, from: cuda.View, to: cuda.View, newline: cuda.View, separator: cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_unlimited_ocr_vision_place(opt(&src), &from, &to, &newline, &separator, &out));
}

/// A window run's rows: `map` int32 [P], a window row's token or -1 for padding, whose q, k and v are `pad`
/// (qkv's bias, what a zero row gives).
pub const Windows = struct { map: cuda.View, pad: cuda.View };

/// `out` [N, hidden] = each segment's attention; the segments are qkv's rows, or `windows`' rows of it.
pub fn attention(qkv: cuda.View, windows: ?Windows, tiles: cuda.View, rel: ?cuda.View, out: cuda.View) cuda.Error!void {
    const map: ?cuda.View = if (windows) |w| w.map else null;
    const pad: ?cuda.View = if (windows) |w| w.pad else null;
    return cuda.launched(zs_unlimited_ocr_vision_attention(&qkv, opt(&map), opt(&pad), &tiles, opt(&rel), &out));
}

pub fn relpos(qkv: cuda.View, map: ?cuda.View, table_h: cuda.View, table_w: cuda.View, rel: cuda.View, len: usize, width: usize) cuda.Error!void {
    return cuda.launched(zs_unlimited_ocr_vision_relpos(&qkv, opt(&map), &table_h, &table_w, &rel, @intCast(len), @intCast(width)));
}

/// A BF16 linear [n, k] as stored and its bias [n] (a conv's zeros where it has none).
pub const Linear = struct {
    weight: cuda.View,
    bias: cuda.View,

    /// out (m, n) = act(x (m, k) W^T + bias), bf16; one tile for every row count, no K split: a row's bits are
    /// its own.
    pub fn forward(self: Linear, x: cuda.View, out: cuda.View, act: Act) cuda.Error!void {
        return cuda.launched(cuda.kernels.zs_b16_gemm_bias(&x, &self.weight, &self.bias, &out, @backingInt(act)));
    }

    /// h (m, n) = bf(h + bf(x (m, k) W^T + bias)) in place: `forward`, then the residual add, in its epilogue
    /// (spec 142 WS-08); the same bits as the two.
    pub fn residual(self: Linear, x: cuda.View, h: cuda.View) cuda.Error!void {
        return cuda.launched(zs_b16_gemm_bias_residual(&x, &self.weight, &self.bias, &h));
    }
};

/// A LayerNorm's weight and bias, bf16 [width].
pub const Norm = struct {
    weight: cuda.Buffer,
    bias: cuda.Buffer,
};

/// A view of `buffer`'s first bytes as `shape` of `dtype`; past the buffer it panics in every build.
pub fn shaped(buffer: cuda.Buffer, dtype: cuda.DType, shape: []const usize) cuda.View {
    var n: usize = dtype.size();
    for (shape) |d| n *= d;
    if (n > buffer.bytes()) std.debug.panic("vision: a view of {any} {t} in a buffer of {d} bytes", .{ shape, dtype, buffer.bytes() });
    return .init(buffer.view.ptr.?, dtype, shape);
}

/// The buffers a load has made, freed together.
pub const Made = struct {
    list: std.ArrayList(cuda.Buffer) = .empty,
    gpa: Allocator,

    pub fn buffer(self: *Made, dtype: cuda.DType, shape: []const usize) Error!cuda.Buffer {
        try self.list.ensureUnusedCapacity(self.gpa, 1);
        const made: cuda.Buffer = try .init(dtype, shape);
        self.list.appendAssumeCapacity(made);
        return made;
    }

    pub fn deinit(self: *Made) void {
        var i = self.list.items.len;
        while (i > 0) {
            i -= 1;
            self.list.items[i].deinit();
        }
        self.list.deinit(self.gpa);
    }
};

/// What a load reads: the checkpoint, the arena of its refusals' words.
pub const Source = struct {
    checkpoint: *const safetensors.Checkpoint,
    arena: Allocator,
    diagnostic: *Diagnostic,

    /// `name`'s tensor, BF16 of `shape`, or a refusal.
    pub fn get(self: Source, name: []const u8, shape: []const u64) error{UnsupportedVision}!safetensors.Tensor {
        const found = self.checkpoint.get(name) orelse {
            self.diagnostic.set(self.arena, "the vision tower lacks {s}: no image can be read with this checkpoint", .{name});
            return error.UnsupportedVision;
        };
        if (found.dtype != .BF16 or !std.mem.eql(u64, found.shape, shape)) {
            self.diagnostic.set(self.arena, "the vision tower's {s} is {t} {any}: zspark reads it as BF16 {any}, as Unlimited-OCR stores it", .{ name, found.dtype, found.shape, shape });
            return error.UnsupportedVision;
        }
        return found;
    }

    /// `name`'s tensor uploaded as stored into a buffer of `shape`.
    pub fn upload(self: Source, made: *Made, name: []const u8, shape: []const u64) Error!cuda.Buffer {
        const found = try self.get(name, shape);
        var dims: [6]usize = undefined;
        for (shape, 0..) |d, i| dims[i] = @intCast(d);
        const buffer = try made.buffer(.bf16, dims[0..shape.len]);
        try buffer.upload(found.data);
        return buffer;
    }

    /// `prefix.weight` [n, k] and `prefix.bias` [n].
    pub fn linear(self: Source, made: *Made, prefix: []const u8, n: usize, k: usize) Error!Linear {
        var a: [128]u8 = undefined;
        var b: [128]u8 = undefined;
        const weight = try self.upload(made, std.fmt.bufPrint(&a, "{s}.weight", .{prefix}) catch unreachable, &.{ n, k });
        const bias = try self.upload(made, std.fmt.bufPrint(&b, "{s}.bias", .{prefix}) catch unreachable, &.{n});
        return .{ .weight = weight.view, .bias = bias.view };
    }

    /// `prefix.weight` and `prefix.bias`, [width] each.
    pub fn norm(self: Source, made: *Made, prefix: []const u8, width: usize) Error!Norm {
        var a: [128]u8 = undefined;
        var b: [128]u8 = undefined;
        return .{
            .weight = try self.upload(made, std.fmt.bufPrint(&a, "{s}.weight", .{prefix}) catch unreachable, &.{width}),
            .bias = try self.upload(made, std.fmt.bufPrint(&b, "{s}.bias", .{prefix}) catch unreachable, &.{width}),
        };
    }

    /// A 3 x 3 conv's weight [out, in, 3, 3] as [out, 9 * in], each output's columns in the order (ky, kx, in)
    /// that `im2col` writes.
    pub fn conv3(self: Source, gpa: Allocator, made: *Made, name: []const u8, out: usize, in: usize, zeros: cuda.View) Error!Linear {
        const found = try self.get(name, &.{ out, in, 3, 3 });
        const permuted = try gpa.alloc(u16, out * in * 9);
        defer gpa.free(permuted);
        for (0..out) |o| for (0..in) |c| for (0..9) |tap| {
            const from = (o * in + c) * 9 + tap;
            permuted[o * in * 9 + tap * in + c] = std.mem.readInt(u16, found.data[2 * from ..][0..2], .little);
        };
        const weight = try made.buffer(.bf16, &.{ out, 9 * in });
        try weight.upload(std.mem.sliceAsBytes(permuted));
        return .{ .weight = weight.view, .bias = zeros.span(0, out) };
    }
};

/// A view's size: the 1024 global view or a 640 crop (`infer`'s `base_size` and `image_size`).
pub const Size = struct {
    /// pixels a side
    side: usize,

    pub const global: Size = .{ .side = 1024 };
    pub const crop: Size = .{ .side = 640 };

    /// SAM's patches a side (16 pixels each)
    pub fn grid(self: Size) usize {
        return self.side / 16;
    }
    pub fn tokens(self: Size) usize {
        return self.grid() * self.grid();
    }
    /// 14 x 14 windows a side, the last padded
    pub fn windows(self: Size) usize {
        return (self.grid() + 13) / 14;
    }
    pub fn windowRows(self: Size) usize {
        return self.windows() * self.windows() * 196;
    }
    /// the neck's output a side (two stride-2 convs): CLIP's patches and the projector's rows
    pub fn out(self: Size) usize {
        return self.grid() / 4;
    }
    pub fn outTokens(self: Size) usize {
        return self.out() * self.out();
    }
};

/// A run's buffers, shared by SAM, CLIP and the projector, each sized for the run's widest use.
pub const Work = struct {
    /// pixels bf16 [views, 3, S, S]
    pixels: cuda.Buffer,
    /// SAM's residual stream, bf16 [tokens, 768]
    h: cuda.Buffer,
    /// a norm's output, the attention's; the neck's products (bf16, a row a token)
    y: cuda.Buffer,
    /// CLIP's branch before its add (SAM's proj and lin2 add theirs in their epilogue); the neck's norms'
    /// outputs; the projector's rows
    branch: cuda.Buffer,
    /// the patches' and the convs' columns, qkv, the MLPs' inner rows, the concat
    wide: cuda.Buffer,
    /// the attention's relative-position terms, bf16 [heads, rows, kh + W] (a window block's rows its padded ones)
    rel: cuda.Buffer,
    /// SAM's features, bf16 [views * out tokens, 1024]
    sam: cuda.Buffer,
    /// CLIP's residual stream, fp32 [views * (1 + out tokens), 1024]
    clip: cuda.Buffer,
};
