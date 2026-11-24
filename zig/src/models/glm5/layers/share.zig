//! A rank's share of a tensor, as TensorFold's `split.py` cuts it: half of the rows, of the columns, or all of
//! it. Host only: bytes in, bytes out.
//!
//! A ModelOpt linear (FP4-Local) is read as its map stores it and cut by its kind's split part by part
//! (`modeloptLinear`, `weights/modelopt.zig`): its scales by whole blocks, csf_32x2_v1 streams whole with the
//! rank's window. A linear's share is a face's part in its stored format (`partOf`), which `faceOf` puts on the
//! device: what every GLM linear but the routed experts and the router runs on.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const face = @import("../../../weights/face.zig");
const numeric = @import("../../../weights/numeric.zig");
const modelopt = @import("../../../weights/modelopt.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const weights_share = @import("../../../weights/share.zig");
const layout = @import("layout.zig");
const Config = @import("config.zig").Config;
const Split = layout.Split;
const world = layout.world;

/// `tensor` as `rank` holds it. Half of the first axis is a slice of the tensor's own bytes; half of the second
/// is not contiguous and is copied into `arena`, as its shape is.
pub fn take(arena: Allocator, tensor: safetensors.Tensor, split: Split, rank: usize) Allocator.Error!safetensors.Tensor {
    return weights_share.take(arena, tensor, split, rank, world);
}

/// The tensor `name` of `source` (anything with `get(name) ?safetensors.Tensor`) as `rank` holds it, by the
/// split `layout.rule` gives the name; null where the source has none.
pub fn get(arena: Allocator, source: anytype, name: []const u8, rank: usize) Allocator.Error!?safetensors.Tensor {
    const tensor = source.get(name) orelse return null;
    return try take(arena, tensor, layout.rule(name).?, rank);
}

/// The ModelOpt linear `module` (`MODULE.weight` and its parts) of `source` as `rank` holds it: read as `c`'s map
/// stores it, at the shape its kind makes, and cut by the kind's split.
pub fn modeloptLinear(arena: Allocator, source: anytype, c: *const Config, module: []const u8, rank: usize, diagnostic: anytype) modelopt.Error!modelopt.Linear {
    const stored = try modeloptStoredWith(arena, source, c, module, diagnostic);
    return modelopt.cut(arena, stored.linear, stored.split, rank, world, diagnostic);
}

/// A ModelOpt linear as stored, uncut, and the split its kind gives it.
pub const Stored = struct { linear: modelopt.Linear, split: Split };

/// `modeloptLinear` before the cut, its refusals logged: for a reader that takes a rank's part where it lies
/// (`mlp.zig`'s experts placed by the device, which read down's columns at the stored row's stride).
pub fn modeloptStored(arena: Allocator, source: anytype, c: *const Config, module: []const u8) PartError!Stored {
    return modeloptStoredWith(arena, source, c, module, Logged{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.MissingTensor => error.MissingTensor,
        error.UnexpectedTensor, error.Indivisible => error.UnsupportedShape,
    };
}

fn modeloptStoredWith(arena: Allocator, source: anytype, c: *const Config, module: []const u8, diagnostic: anytype) modelopt.Error!Stored {
    var name: [256]u8 = undefined;
    const weight = std.fmt.bufPrint(&name, "{s}.weight", .{module}) catch return error.MissingTensor;
    const kind = layout.kindIn(c.*, weight) orelse {
        diagnostic.set(arena, "{s} is no linear of GLM-5.3-Flash's layout", .{module});
        return error.MissingTensor;
    };
    if (!kind.linear) {
        diagnostic.set(arena, "{s} is not read as a ModelOpt linear", .{module});
        return error.UnexpectedTensor;
    }
    var dims: [3]u64 = undefined;
    const shape = kind.shape(c.*, &dims);
    return .{ .linear = try modelopt.read(arena, source, module, c.quant.find(module), shape[0], shape[1], diagnostic), .split = kind.split };
}

/// What a load refuses of a ModelOpt linear (`modelopt.read`'s diagnostic), logged: a load has no other place for it.
pub const Logged = struct {
    pub fn set(_: Logged, _: Allocator, comptime format: []const u8, args: anytype) void {
        std.log.err(format, args);
    }
};

pub const PartError = error{ MissingTensor, UnsupportedShape } || Allocator.Error;

/// `modeloptLinear`, its refusals logged and named as a load's.
pub fn modeloptOf(arena: Allocator, source: anytype, c: *const Config, module: []const u8, rank: usize) PartError!modelopt.Linear {
    return modeloptLinear(arena, source, c, module, rank, Logged{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.MissingTensor => error.MissingTensor,
        error.UnexpectedTensor, error.Indivisible => error.UnsupportedShape,
    };
}

/// The linear `module` (`MODULE.weight`) of `source` as `rank` holds it, as a face's part in the storage the
/// checkpoint gives it: BF16 (a linear the map does not name), MXFP8 and FP8_PB_WO read as stored (FP4-Local).
pub fn partOf(arena: Allocator, source: anytype, c: *const Config, module: []const u8, rank: usize) PartError!face.Part {
    const linear_ = try modeloptOf(arena, source, c, module, rank);
    return switch (linear_.scheme) {
        .bf16 => .{ .bf16 = linear_.weight.data },
        .mxfp8 => .{ .mxfp8 = .{ .codes = linear_.weight.data, .exponents = linear_.scale.?.data, .n = linear_.rows } },
        .fp8_block => .{ .fp8_block = .{ .codes = linear_.weight.data, .block_scales = .{ .data = linear_.scale.?.data }, .n = linear_.rows } },
        .nvfp4, .fp8 => {
            std.log.err("{s} is stored {t}, which zspark reads for routed experts alone", .{ module, linear_.scheme });
            return error.UnsupportedShape;
        },
    };
}

/// `module` as bf16 [rows, cols] bytes, for a weight a kernel reads as bf16 itself (MLA's `kv_b_proj`): BF16 as
/// stored; MXFP8 written out in bf16, which holds every weight of it exactly (an e4m3 code, three bits of
/// mantissa, times a power of two), and refused where one would round. Nothing is quantized.
pub fn bf16Weight(arena: Allocator, source: anytype, c: *const Config, module: []const u8, rank: usize) PartError![]const u8 {
    switch (try partOf(arena, source, c, module, rank)) {
        .bf16 => |bytes| return bytes,
        .mxfp8 => |rows| {
            const k = rows.codes.len / rows.n;
            const out = try arena.alloc(u8, rows.codes.len * 2);
            for (rows.codes, 0..) |code, at| {
                const exponent = rows.exponents[at / k * (k / modelopt.mxfp8_block) + at % k / modelopt.mxfp8_block];
                const value = numeric.f32FromE4m3(code) * numeric.f32FromUe8m0(exponent);
                const bits: u32 = @bitCast(value);
                if (exponent == 255 or !std.math.isFinite(value) or bits & 0xffff != 0 or std.math.ldexp(value, 127 - @as(i32, exponent)) != numeric.f32FromE4m3(code)) {
                    std.log.err("{s}: an MXFP8 weight is not exact in bf16", .{module});
                    return error.UnsupportedShape;
                }
                std.mem.writeInt(u16, out[at * 2 ..][0..2], @intCast(bits >> 16), .little);
            }
            return out;
        },
        else => {
            std.log.err("{s}: zspark reads this linear as bf16, and its storage is not exact there", .{module});
            return error.UnsupportedShape;
        },
    }
}

/// Whether `one` is [n, k].
pub fn shaped(one: face.Part, n: usize, k: usize) bool {
    return switch (one) {
        .bf16 => |bytes| bytes.len == n * k * 2,
        .mxfp8 => |rows| rows.n == n and rows.codes.len == n * k,
        .fp8_block => |rows| rows.n == n and rows.codes.len == n * k,
    };
}

/// `parts` of `k` inputs, in column order, as one face on the device (`face.Face.initParts`: a BF16 face's bits
/// are `bf16.Linear.initStacked`'s). `scratch` holds what a run is packed in while it is. Its MXFP8 and block-FP8
/// runs take their rows quantized as the card's vLLM quantizes them, in every phase (spec 114's 2026-10-19
/// decision, `nvfp4.Activations`).
pub fn faceOf(scratch: Allocator, parts: []const face.Part, k: usize, rows: cuda.Scratch.Rows) (PartError || cuda.Error)!face.Face {
    var made = face.Face.initParts(scratch, parts, k, rows) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedShape => error.UnsupportedShape,
        else => error.CudaFailure,
    };
    errdefer made.deinit();
    const kept = made.quantizeActivations() catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.CudaFailure,
    };
    if (kept > 0) std.log.debug("a face of {d} inputs keeps bf16 rows on {d} quantized runs", .{ k, kept });
    return made;
}
