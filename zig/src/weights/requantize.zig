//! Experts a checkpoint stores as BF16 or FP8, quantized on the host to the NVFP4 the experts kernel reads. The
//! MTP layer's experts come so (BF16 stacked; 128x128-block FP8 in the retired NVFP4 checkpoint); they only draft, so what they
//! lose changes which drafts are offered and never a token. The arithmetic is TensorFold's, operation for
//! operation in fp32 (cuda/nvfp4/experts.py `_quantize`, after families/qwen4_exp/cuda/weights.py `weight_bf16`),
//! so the codes and scales are its own: `tools/reference/requantize_cases.py` exports the fixture that holds
//! this to the bit.

const std = @import("std");
const Allocator = std.mem.Allocator;
const experts = @import("experts.zig");
const numeric = @import("numeric.zig");
const nvfp4 = @import("nvfp4.zig");

/// Inputs sharing one e4m3 scale.
pub const group = 16;

/// One expert's projection [n, k] row-major, as a checkpoint stores it where it is not NVFP4.
pub const Weight = union(enum) {
    /// bf16 values
    bf16: []const u8,
    /// e4m3 bytes with a scale per 128x128 block (`weight_scale_inv`)
    fp8_block: nvfp4.Rows,
    /// e4m3 bytes with one scale for the tensor or one per output (`weight_scale`)
    fp8: Scaled,
    /// fp32 values [n, k], taken as they are: rows decoded from another format (an EXL3 head's)
    f32: []const f32,

    pub const Scaled = struct {
        codes: []const u8,
        /// one value, or n
        scales: nvfp4.BlockScales,
    };

    /// The weight as TensorFold reads it before it quantizes: bf16 values, here widened to fp32. FP8 is its
    /// code times its scale in fp32, rounded to bf16.
    pub fn values(self: Weight, n: usize, k: usize, out: []f32) void {
        std.debug.assert(out.len == n * k);
        for (0..n) |r| self.row(n, k, r, out[r * k ..][0..k]);
    }

    /// Row `r` of `values`.
    pub fn row(self: Weight, n: usize, k: usize, r: usize, out: []f32) void {
        std.debug.assert(out.len == k and r < n);
        switch (self) {
            .bf16 => |bytes| {
                std.debug.assert(bytes.len == n * k * 2);
                for (out, 0..) |*value, i| value.* = numeric.f32FromBf16(std.mem.readInt(u16, bytes[(r * k + i) * 2 ..][0..2], .little));
            },
            .fp8_block => |rows| {
                const block: nvfp4.Fp8BlockLinear.Block = .of(k);
                std.debug.assert(rows.codes.len == n * k and k % block.inputs == 0);
                std.debug.assert(rows.block_scales.len() == (n + block.outputs - 1) / block.outputs * (k / block.inputs));
                for (0..k / block.inputs) |column| {
                    const scale = rows.block_scales.at(r / block.outputs * (k / block.inputs) + column);
                    const at = column * block.inputs;
                    for (out[at..][0..block.inputs], rows.codes[r * k + at ..][0..block.inputs]) |*value, code| value.* = numeric.roundBf16(e4m3[code] * scale);
                }
            },
            .f32 => |all| {
                std.debug.assert(all.len == n * k);
                @memcpy(out, all[r * k ..][0..k]);
            },
            .fp8 => |scaled| {
                const count = scaled.scales.len();
                std.debug.assert(scaled.codes.len == n * k and (count == 1 or count == n));
                const scale = scaled.scales.at(if (count == 1) 0 else r);
                for (out, scaled.codes[r * k ..][0..k]) |*value, code| value.* = numeric.roundBf16(e4m3[code] * scale);
            },
        }
    }
};

const e4m3: [256]f32 = blk: {
    @setEvalBranchQuota(20000);
    var table: [256]f32 = undefined;
    for (&table, 0..) |*value, code| value.* = numeric.f32FromE4m3(code);
    break :blk table;
};

/// The magnitudes of the E2M1 codes 0 to 7; a negative value takes its magnitude's code plus 8.
const magnitudes: @Vector(8, f32) = nvfp4.e2m1[0..8].*;

/// One expert's `values` [n, k] to NVFP4 by ModelOpt's recipe as TensorFold computes it on the device: `codes`
/// [n, k/2], two E2M1 codes a byte with the lower input in the low nibble; `block_scales` e4m3 [n, k/16]; the
/// result is the expert's scale (`weight_scale_2`). A weight is code x block scale x expert scale.
///
/// - The expert's scale puts its largest magnitude at 6 x 448: largest x fl(1 / 2688), at least 1e-30.
///   PyTorch divides a device tensor by a Python scalar as a product with the scalar's fp32 reciprocal, so
///   this and the 1/6 below are products, not divisions.
/// - A block's scale puts its largest magnitude at code 6: (largest x fl(1 / 6)) / expert scale, at most 448,
///   rounded to e4m3 to nearest even. A block far below the expert's largest gets a subnormal scale, or zero.
/// - A value's code is the first of the eight magnitudes nearest value / (block scale x expert scale), each
///   distance taken in fp32: a value halfway between two codes takes the smaller. Its sign bit is set where
///   the quotient is below zero, so a negative value that rounds to nothing is -0 (code 8). In a block whose
///   scale rounded to zero every code is 0.
pub fn quantize(values: []const f32, n: usize, k: usize, codes: []u8, block_scales: []u8) f32 {
    std.debug.assert(k % group == 0 and values.len == n * k and codes.len == n * k / 2 and block_scales.len == n * k / group);
    const expert = scaleOf(largestOf(values));
    quantizeScaled(values, n, k, expert, codes, block_scales);
    return expert;
}

fn largestOf(values: []const f32) f32 {
    var largest: f32 = 0;
    for (values) |value| largest = @max(largest, @abs(value));
    return largest;
}

/// The scale of a tensor whose largest magnitude is `largest`.
fn scaleOf(largest: f32) f32 {
    const range: f32 = 6.0 * 448.0;
    return @max(largest * (1.0 / range), tiny);
}

/// `quantize`'s blocks and codes under a tensor scale already known: `values` may be some rows of the tensor.
fn quantizeScaled(values: []const f32, n: usize, k: usize, expert: f32, codes: []u8, block_scales: []u8) void {
    std.debug.assert(k % group == 0 and values.len == n * k and codes.len == n * k / 2 and block_scales.len == n * k / group);
    const sixth: f32 = 1.0 / 6.0;
    for (0..n * k / group) |b| {
        const block: @Vector(group, f32) = values[b * group ..][0..group].*;
        const scale = numeric.e4m3FromF32(@min(@reduce(.Max, @abs(block)) * sixth / expert, 448.0));
        block_scales[b] = scale;
        const step = e4m3[scale] * expert;
        if (!(step > 0)) {
            @memset(codes[b * group / 2 ..][0 .. group / 2], 0);
            continue;
        }
        const v = block / @as(@Vector(group, f32), @splat(@max(step, tiny)));
        const magnitude = @abs(v);
        var best: @Vector(group, f32) = magnitude; // the distance to code 0
        var code: @Vector(group, u8) = @splat(0);
        for (1..8) |candidate| {
            const distance = @abs(magnitude - @as(@Vector(group, f32), @splat(magnitudes[candidate])));
            const nearer = distance < best;
            best = @select(f32, nearer, distance, best);
            code = @select(u8, nearer, @as(@Vector(group, u8), @splat(@intCast(candidate))), code);
        }
        code |= @select(u8, v < @as(@Vector(group, f32), @splat(0)), @as(@Vector(group, u8), @splat(8)), @as(@Vector(group, u8), @splat(0)));
        const nibbles: [group]u8 = code;
        for (codes[b * group / 2 ..][0 .. group / 2], 0..) |*byte, i| byte.* = nibbles[2 * i] | nibbles[2 * i + 1] << 4;
    }
}

/// TensorFold's `clamp_min(1e-30)`, on the expert's scale and on a step.
pub const tiny: f32 = 1e-30;

/// The weight [n, k] that NVFP4 codes stand for, in fp32 as the kernel forms it (code x block scale, x expert
/// scale).
pub fn dequantize(codes: []const u8, block_scales: []const u8, expert: f32, out: []f32) void {
    std.debug.assert(codes.len * 2 == out.len and block_scales.len * group == out.len);
    for (out, 0..) |*value, i| {
        const byte = codes[i / 2];
        const code = if (i % 2 == 0) byte & 0xf else byte >> 4;
        value.* = nvfp4.e2m1[code] * e4m3[block_scales[i / group]] * expert;
    }
}

/// One projection of every expert, quantized: what `experts.Experts.init` takes, and the memory behind it.
pub const Quantized = struct {
    /// [experts, n, k/2]
    codes: []u8,
    /// e4m3 [experts, n, k/16]
    block_scales: []u8,
    /// [experts]
    scales: []f32,
    code_parts: []const []const u8,
    scale_parts: []const []const u8,

    /// `weights[e]` is expert e's [n, k]. Experts quantize in parallel, each on its own, so the result does not
    /// depend on the threads.
    pub fn init(gpa: Allocator, weights: []const Weight, n: usize, k: usize) (Allocator.Error || error{UnsupportedShape})!Quantized {
        if (k % group != 0 or weights.len == 0) return error.UnsupportedShape;
        const count = weights.len;
        const codes = try gpa.alloc(u8, count * n * k / 2);
        errdefer gpa.free(codes);
        const block_scales = try gpa.alloc(u8, count * n * k / group);
        errdefer gpa.free(block_scales);
        const scales = try gpa.alloc(f32, count);
        errdefer gpa.free(scales);
        const code_parts = try experts.Matrix.parts(gpa, codes, count);
        errdefer gpa.free(code_parts);
        const scale_parts = try experts.Matrix.parts(gpa, block_scales, count);
        errdefer gpa.free(scale_parts);
        const self: Quantized = .{ .codes = codes, .block_scales = block_scales, .scales = scales, .code_parts = code_parts, .scale_parts = scale_parts };

        const workers = @min(count, std.Thread.getCpuCount() catch 1, max_threads);
        // each worker's expert as fp32 values
        const scratch = try gpa.alloc(f32, workers * n * k);
        defer gpa.free(scratch);
        var threads: [max_threads]std.Thread = undefined;
        var spawned: usize = 0;
        for (0..workers - 1) |w| {
            threads[spawned] = std.Thread.spawn(.{}, share, .{ self, weights, n, k, scratch[(w + 1) * n * k ..][0 .. n * k], w + 1, workers }) catch break;
            spawned += 1;
        }
        // the shares no thread took, this thread's own included
        share(self, weights, n, k, scratch[0 .. n * k], 0, workers);
        for (spawned + 1..workers) |w| share(self, weights, n, k, scratch[0 .. n * k], w, workers);
        for (threads[0..spawned]) |thread| thread.join();
        return self;
    }

    const max_threads = 64;

    /// Experts `first`, `first + of`, ...
    fn share(self: Quantized, weights: []const Weight, n: usize, k: usize, scratch: []f32, first: usize, of: usize) void {
        var e = first;
        while (e < weights.len) : (e += of) {
            weights[e].values(n, k, scratch);
            self.scales[e] = quantize(scratch, n, k, self.codes[e * n * k / 2 ..][0 .. n * k / 2], self.block_scales[e * n * k / group ..][0 .. n * k / group]);
        }
    }

    pub fn deinit(self: *Quantized, gpa: Allocator) void {
        gpa.free(self.codes);
        gpa.free(self.block_scales);
        gpa.free(self.scales);
        gpa.free(self.code_parts);
        gpa.free(self.scale_parts);
        self.* = undefined;
    }

    pub fn matrix(self: Quantized) experts.Matrix {
        return .{ .codes = self.code_parts, .block_scales = self.scale_parts, .scales = self.scales };
    }
};

/// Chosen rows of one projection as one NVFP4 tensor: what `nvfp4.Fp4Linear.init` takes. The draft head is the
/// main head's rows of the draft vocabulary, so. TensorFold quantizes those rows to MLX's affine 4-bit, which
/// zspark has no kernel for; like the experts above, they only draft.
pub const Rows = struct {
    /// [ids.len, k/2]
    codes: []u8,
    /// e4m3 [ids.len, k/16]
    block_scales: []u8,
    /// The tensor's scale: the largest magnitude among the chosen rows at 6 x 448.
    scale: f32,

    /// Row i of the result is row `ids[i]` of `weight` [n, k]. The rows quantize in parallel under one scale,
    /// so the result does not depend on the threads.
    pub fn init(gpa: Allocator, weight: Weight, n: usize, k: usize, ids: []const u32) (Allocator.Error || error{UnsupportedShape})!Rows {
        if (k % group != 0 or ids.len == 0) return error.UnsupportedShape;
        for (ids) |id| if (id >= n) return error.UnsupportedShape;
        const codes = try gpa.alloc(u8, ids.len * k / 2);
        errdefer gpa.free(codes);
        const block_scales = try gpa.alloc(u8, ids.len * k / group);
        errdefer gpa.free(block_scales);
        const workers = @min(ids.len, std.Thread.getCpuCount() catch 1, max_threads);
        // each worker's row as fp32 values
        const scratch = try gpa.alloc(f32, workers * k);
        defer gpa.free(scratch);
        var largest: [max_threads]f32 = @splat(0);
        var job: Job = .{ .weight = weight, .n = n, .k = k, .ids = ids, .codes = codes, .block_scales = block_scales, .scratch = scratch, .largest = largest[0..workers], .scale = null };
        job.run();
        var most: f32 = 0;
        for (largest[0..workers]) |value| most = @max(most, value);
        job.scale = scaleOf(most);
        job.run();
        return .{ .codes = codes, .block_scales = block_scales, .scale = job.scale.? };
    }

    pub fn deinit(self: *Rows, gpa: Allocator) void {
        gpa.free(self.codes);
        gpa.free(self.block_scales);
        self.* = undefined;
    }

    const max_threads = 64;

    const Job = struct {
        weight: Weight,
        n: usize,
        k: usize,
        ids: []const u32,
        codes: []u8,
        block_scales: []u8,
        scratch: []f32,
        /// Each worker's largest magnitude: the first pass's result.
        largest: []f32,
        /// Null on the first pass, which only measures.
        scale: ?f32,

        /// One pass over the rows, worker w taking rows w, w + workers, ...
        fn run(job: *const Job) void {
            const workers = job.largest.len;
            var threads: [max_threads]std.Thread = undefined;
            var spawned: usize = 0;
            for (1..workers) |w| {
                threads[spawned] = std.Thread.spawn(.{}, share, .{ job, w, w }) catch break;
                spawned += 1;
            }
            // the shares no thread took, this thread's own included
            job.share(0, 0);
            for (spawned + 1..workers) |w| job.share(w, 0);
            for (threads[0..spawned]) |thread| thread.join();
        }

        /// Worker `first`'s rows, with scratch `slot`'s.
        fn share(job: *const Job, first: usize, slot: usize) void {
            const k = job.k;
            const values = job.scratch[slot * k ..][0..k];
            var i = first;
            while (i < job.ids.len) : (i += job.largest.len) {
                job.weight.row(job.n, k, job.ids[i], values);
                if (job.scale) |scale| {
                    quantizeScaled(values, 1, k, scale, job.codes[i * k / 2 ..][0 .. k / 2], job.block_scales[i * k / group ..][0 .. k / group]);
                } else {
                    job.largest[first] = @max(job.largest[first], largestOf(values));
                }
            }
        }
    };
};
