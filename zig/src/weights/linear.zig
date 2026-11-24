//! A linear held as its checkpoint stores it (BF16, MXFP8, NVFP4 or per-tensor FP8 read as stored) or in four
//! bits made of BF16 at load (`requantize.zig`'s NVFP4, 4.5 bits a weight), in memory only: what a drafter's large
//! linears and its copy of a model's head are held in (`models/glm5/`'s MTP layer and DFlash2 drafter), GLM's head
//! (MXFP8 in FP4-Local), Qwen3-ASR's language model as its W4A16_NVFP4 builds store it (`stored`, spec 140), and
//! Nemotron's per-tensor FP8 Mamba projections (spec 151). One type for all, so that a caller's forward does not
//! say which.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");
const bf16 = @import("bf16.zig");
const nvfp4 = @import("nvfp4.zig");
const modelopt = @import("modelopt.zig");
const requantize = @import("requantize.zig");

pub const Error = error{UnsupportedShape} || Allocator.Error || cuda.Error;

pub const Storage = enum {
    /// As stored.
    bf16,
    /// NVFP4 made at load, with an e4m3 scale a group of 16.
    four,
};

/// A linear in either storage. Both give x bf16 [rows, k] -> out [rows, n], bf16 or fp32 by
/// `out`'s dtype, a row's bits its own.
pub const Linear = union(enum) {
    bf16: bf16.Linear,
    /// NVFP4, made at load (`init`) or read as stored (`stored`): bf16 rows by default;
    /// `quantizeActivations` applies checkpoint NVFP4 activation rounding before the lane matmul.
    four: nvfp4.Fp4Linear,
    /// As a ModelOpt checkpoint stores it (`nvfp4.Mx8Linear`); made by its caller, never by `init`.
    mxfp8: nvfp4.Mx8Linear,
    /// Per-tensor FP8 as ModelOpt stores it (e4m3 and one `weight_scale`, `nvfp4.Fp8Linear`): bf16 rows into the
    /// lane matmul, made by `stored`, never by `init`.
    fp8: nvfp4.Fp8Linear,
    /// Stacked FP8 parts with distinct scales, or static FP8 activations (`storedStaticFp8`).
    fp8_block: nvfp4.Fp8BlockLinear,

    /// `parts` are bf16 [rows, k] of one input, stacked in order. In four bits the stack has one scale.
    pub fn init(arena: Allocator, gpa: Allocator, parts: []const []const u8, k: usize, storage: Storage, rows: cuda.Scratch.Rows) Error!Linear {
        switch (storage) {
            .bf16 => return .{ .bf16 = bf16.Linear.initStacked(parts, k, rows) catch |err| return refused(err) },
            .four => {
                const weight = if (parts.len == 1) parts[0] else try std.mem.concat(arena, u8, parts);
                const n = weight.len / (k * 2);
                if (n == 0 or n * k * 2 != weight.len) return error.UnsupportedShape;
                const ids = try arena.alloc(u32, n);
                for (ids, 0..) |*id, at| id.* = @intCast(at);
                var quantized: requantize.Rows = try .init(gpa, .{ .bf16 = weight }, n, k, ids);
                defer quantized.deinit(gpa);
                return .{ .four = nvfp4.Fp4Linear.init(gpa, quantized.codes, quantized.block_scales, n, k, quantized.scale) catch |err| return refused(err) };
            },
        }
    }

    /// `parts`, ModelOpt linears of one input as `modelopt.read` gives them, stacked in order: BF16 parts as
    /// `bf16.Linear.initStacked` stacks them (its bits), NVFP4 parts as one `nvfp4.Fp4Linear` of their codes and
    /// block scales row after row, read as stored. Per-tensor FP8 parts share the scalar fast path when
    /// their scales agree; otherwise each part's scalar is retained by the block-scaled FP8 path.
    /// NVFP4 takes one `weight_scale_2` per stack and refuses differing scales.
    pub fn stored(arena: Allocator, gpa: Allocator, parts: []const modelopt.Linear, rows: cuda.Scratch.Rows, diagnostic: anytype) Error!Linear {
        std.debug.assert(parts.len > 0);
        const k: usize = @intCast(parts[0].cols);
        for (parts[1..]) |part| if (part.scheme != parts[0].scheme or part.cols != parts[0].cols) {
            diagnostic.set(arena, "a stack of {t} and {t} linears, or of {d} and {d} inputs: zspark stacks one storage of one input", .{ parts[0].scheme, part.scheme, parts[0].cols, part.cols });
            return error.UnsupportedShape;
        };
        const weights = try arena.alloc([]const u8, parts.len);
        switch (parts[0].scheme) {
            .bf16 => for (weights, parts) |*b, part| {
                b.* = part.weight.data;
            },
            .nvfp4 => {
                const scale = scaleOf(parts[0]);
                for (parts[1..]) |part| if (scaleOf(part) != scale) {
                    diagnostic.set(arena, "a stack's NVFP4 parts have weight_scale_2 {e} and {e}: zspark runs one a stack, as stored", .{ scale, scaleOf(part) });
                    return error.UnsupportedShape;
                };
                const codes = try arena.alloc([]const u8, parts.len);
                const scales = try arena.alloc([]const u8, parts.len);
                var n: usize = 0;
                for (parts, 0..) |part, i| {
                    codes[i] = part.weight.data;
                    scales[i] = try blockScales(arena, part);
                    n += @intCast(part.rows);
                }
                const joined_codes = if (parts.len == 1) codes[0] else try std.mem.concat(arena, u8, codes);
                const joined_scales = if (parts.len == 1) scales[0] else try std.mem.concat(arena, u8, scales);
                return .{ .four = nvfp4.Fp4Linear.init(gpa, joined_codes, joined_scales, n, k, scale) catch |err| return refused(err) };
            },
            .fp8 => {
                const scale = scaleOf(parts[0]);
                for (parts[1..]) |part| if (scaleOf(part) != scale) {
                    const stacked = try arena.alloc(nvfp4.Rows, parts.len);
                    var most: usize = 0;
                    for (parts, stacked) |one, *into| {
                        const n: usize = @intCast(one.rows);
                        most = @max(most, n);
                        into.* = .{ .codes = one.weight.data, .block_scales = .{ .data = one.scale_2.?.data }, .n = n };
                    }
                    // Blocks restart at each part, so one output block spans its scalar's whole tensor.
                    return .{ .fp8_block = nvfp4.Fp8BlockLinear.initStacked(gpa, stacked, k, .{ .outputs = most, .inputs = k }) catch |err| return refused(err) };
                };
                var n: usize = 0;
                for (parts, weights) |part, *codes| {
                    codes.* = part.weight.data;
                    n += @intCast(part.rows);
                }
                const joined = if (parts.len == 1) weights[0] else try std.mem.concat(arena, u8, weights);
                if (joined.len != n * k) return error.UnsupportedShape;
                return .{ .fp8 = nvfp4.Fp8Linear.init(gpa, joined, n, k, scale) catch |err| return refused(err) };
            },
            else => |scheme| {
                diagnostic.set(arena, "a {t} linear: zspark stacks BF16, NVFP4 and per-tensor FP8 ones here", .{scheme});
                return error.UnsupportedShape;
            },
        }
        return .{ .bf16 = bf16.Linear.initStacked(weights, k, rows) catch |err| return refused(err) };
    }

    /// A per-tensor FP8 projection with its checkpoint's activation scale, in every window.
    pub fn storedStaticFp8(gpa: Allocator, part: modelopt.Linear, rows: cuda.Scratch.Rows) Error!Linear {
        if (part.scheme != .fp8 or part.input_scale == null) return error.UnsupportedShape;
        const n: usize = @intCast(part.rows);
        const k: usize = @intCast(part.cols);
        const scale: f32 = @bitCast(std.mem.readInt(u32, part.input_scale.?.data[0..4], .little));
        var projection = nvfp4.Fp8BlockLinear.init(gpa, part.weight.data, &.{scaleOf(part)}, n, k, .{ .outputs = n, .inputs = k }) catch |err| return refused(err);
        errdefer projection.deinit();
        projection.quantizeStaticActivations(rows, scale) catch |err| return refused(err);
        return .{ .fp8_block = projection };
    }

    /// An NVFP4 linear's block scales, e4m3 [rows, cols / 16]; a csf stream that does not decode is no shape here.
    fn blockScales(arena: Allocator, part: modelopt.Linear) Error![]const u8 {
        return part.blockScales(arena) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.UnsupportedShape,
        };
    }

    /// An NVFP4 linear's `weight_scale_2`, an FP8 one's `weight_scale`.
    fn scaleOf(part: modelopt.Linear) f32 {
        return @bitCast(std.mem.readInt(u32, part.scale_2.?.data[0..4], .little));
    }

    fn refused(err: anyerror) Error {
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.UnsupportedShape => error.UnsupportedShape,
            else => error.CudaFailure,
        };
    }

    pub fn deinit(self: *Linear) void {
        switch (self.*) {
            inline else => |*linear| linear.deinit(),
        }
    }

    pub fn outputs(self: Linear) usize {
        return switch (self) {
            inline else => |linear| linear.n,
        };
    }

    pub fn forward(self: Linear, x: cuda.View, out: cuda.View) cuda.Error!void {
        return switch (self) {
            inline else => |linear| linear.forward(x, out),
        };
    }

    /// What `forward` into fp32 `out` would write, as K slices for the kernel that reads it next to sum
    /// (`cuda.Slices`): BF16's own, else `out` itself, written (`face.Face.slices`).
    pub fn slices(self: Linear, x: cuda.View, out: cuda.View) cuda.Error!cuda.Slices {
        if (self == .bf16) return self.bf16.slices(x, out);
        try self.forward(x, out);
        return .one(out);
    }

    /// Device bytes of the weight.
    pub fn bytes(self: Linear) u64 {
        return switch (self) {
            .bf16 => |linear| linear.weight.bytes(),
            .four => |linear| linear.words.bytes() + linear.block_scales.bytes(),
            .mxfp8 => |linear| linear.w8.bytes() + linear.exponents.bytes(),
            .fp8 => |linear| linear.w8.bytes(),
            .fp8_block => |linear| linear.w8.bytes() + linear.column_scales.bytes() + if (linear.tile_scales) |scales| scales.bytes() else @as(u64, 0),
        };
    }
};
