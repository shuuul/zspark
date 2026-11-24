//! A face: several linears of one input run as one projection writing one output row (DeltaNet's qkv, z, b and a;
//! attention's q, k, v and the indexer's). A checkpoint may store a face's linears differently, block FP8 or MXFP8
//! beside BF16, so the face runs each storage on its own kernel and writes each result into its columns of the row.
//! After TensorFold's face builder (families/qwen4_exp/cuda/weights.py `face`) and `Concat`
//! (cuda/nvfp4/linear.py).

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("bf16.zig");
const cuda = @import("../cuda/cuda.zig");
const nvfp4 = @import("nvfp4.zig");

/// One linear of a face as the checkpoint stores it, [rows, k] row-major.
pub const Part = union(enum) {
    /// bf16 values
    bf16: []const u8,
    /// e4m3 bytes and a scale per block (`nvfp4.Fp8BlockLinear.Block.of`), read as stored
    fp8_block: nvfp4.Rows,
    /// e4m3 bytes and a ue8m0 exponent a 32 inputs, read as stored
    mxfp8: nvfp4.Mx8Rows,
};

/// The parts one face takes.
pub const max_parts = 8;

pub const Face = struct {
    /// Neighbouring parts of one storage, stacked into one matmul; `runs[0..count]` in column order.
    runs: [max_parts]Run,
    count: usize,
    /// Outputs: the row's width.
    n: usize,
    k: usize,
    max_rows: usize,
    /// [max_rows, the widest run] elements of 4 bytes: where a run's rows land before they are placed in their
    /// columns. Null with one run, which writes the output itself. The pool's, as its runs' scratch is.
    staging: ?cuda.Buffer,
    scratch: *cuda.Scratch,

    pub const Linear = union(enum) {
        bf16: bf16.Linear,
        fp8_block: nvfp4.Fp8BlockLinear,
        mxfp8: nvfp4.Mx8Linear,

        fn outputs(self: Linear) usize {
            return switch (self) {
                inline else => |linear| linear.n,
            };
        }

        fn forward(self: Linear, x: cuda.View, out: cuda.View) cuda.Error!void {
            return switch (self) {
                inline else => |linear| linear.forward(x, out),
            };
        }

        fn deinit(self: *Linear) void {
            switch (self.*) {
                inline else => |*linear| linear.deinit(),
            }
        }
    };

    pub const Run = struct {
        linear: Linear,
        /// Its first column in the row.
        at: usize,
    };

    /// One BF16 weight [n, k], the checkpoint's bytes: `bf16.Linear.init`'s arguments and bits.
    pub fn init(weight: []const u8, n: usize, k: usize, rows: cuda.Scratch.Rows) !Face {
        var face: Face = .{ .runs = undefined, .count = 1, .n = n, .k = k, .max_rows = rows.count, .staging = null, .scratch = rows.pool };
        face.runs[0] = .{ .linear = .{ .bf16 = try .init(weight, n, k, rows) }, .at = 0 };
        return face;
    }

    /// Several BF16 weights [rows, k] stacked in order: `bf16.Linear.initStacked`'s arguments and bits.
    pub fn initStacked(weights: []const []const u8, k: usize, rows: cuda.Scratch.Rows) !Face {
        var face: Face = .{ .runs = undefined, .count = 1, .n = 0, .k = k, .max_rows = rows.count, .staging = null, .scratch = rows.pool };
        face.runs[0] = .{ .linear = .{ .bf16 = try .initStacked(weights, k, rows) }, .at = 0 };
        face.n = face.runs[0].linear.outputs();
        return face;
    }

    /// `parts` in column order, each [rows, k]. Neighbours of one storage become one matmul, as in TensorFold:
    /// BF16 parts alone give `bf16.Linear.initStacked`'s bits. `gpa` holds a block-FP8 run while it is packed.
    pub fn initParts(gpa: Allocator, parts: []const Part, k: usize, rows: cuda.Scratch.Rows) !Face {
        if (parts.len == 0 or parts.len > max_parts) return error.UnsupportedShape;
        var face: Face = .{ .runs = undefined, .count = 0, .n = 0, .k = k, .max_rows = rows.count, .staging = null, .scratch = rows.pool };
        errdefer face.deinit();
        var widest: usize = 0;
        var first: usize = 0;
        while (first < parts.len) {
            var end = first + 1;
            while (end < parts.len and std.meta.activeTag(parts[end]) == std.meta.activeTag(parts[first])) end += 1;
            const linear: Linear = switch (parts[first]) {
                .bf16 => blk: {
                    var weights: [max_parts][]const u8 = undefined;
                    for (parts[first..end], 0..) |part, i| weights[i] = part.bf16;
                    break :blk .{ .bf16 = try .initStacked(weights[0 .. end - first], k, rows) };
                },
                .fp8_block => blk: {
                    var stacked: [max_parts]nvfp4.Rows = undefined;
                    for (parts[first..end], 0..) |part, i| stacked[i] = part.fp8_block;
                    break :blk .{ .fp8_block = try .initStacked(gpa, stacked[0 .. end - first], k, .of(k)) };
                },
                .mxfp8 => blk: {
                    var stacked: [max_parts]nvfp4.Mx8Rows = undefined;
                    for (parts[first..end], 0..) |part, i| stacked[i] = part.mxfp8;
                    break :blk .{ .mxfp8 = try .initStacked(gpa, stacked[0 .. end - first], k) };
                },
            };
            face.runs[face.count] = .{ .linear = linear, .at = face.n };
            face.count += 1;
            face.n += linear.outputs();
            widest = @max(widest, linear.outputs());
            first = end;
        }
        if (face.count > 1) face.staging = try rows.pool.lend(.staging, .u32, &.{ rows.count, widest });
        return face;
    }

    /// Its MXFP8 and block-FP8 runs take their rows quantized as vLLM's from now on (`nvfp4.Activations`), at up
    /// to the face's rows; how many of them cannot and keep bf16 rows.
    pub fn quantizeActivations(self: *Face) !usize {
        const rows = self.scratch.rows(self.max_rows);
        var kept: usize = 0;
        for (self.runs[0..self.count]) |*run| switch (run.linear) {
            .fp8_block => |*linear| kept += @intFromBool(!try linear.quantizeActivations(rows)),
            .mxfp8 => |*linear| kept += @intFromBool(!try linear.quantizeActivations(rows)),
            else => {},
        };
        return kept;
    }

    /// Where its runs quantize a call's rows, the columns from `column` on (inside one MXFP8 run that quantizes,
    /// a whole tile of it) keep bf16 rows: `nvfp4.Mx8Linear.keepBf16From`.
    pub fn keepBf16From(self: *Face, column: usize) !void {
        for (self.runs[0..self.count]) |*run| switch (run.linear) {
            .mxfp8 => |*linear| if (column >= run.at and column < run.at + linear.n and linear.a8 != null)
                return linear.keepBf16From(self.scratch.rows(self.max_rows), column - run.at),
            else => {},
        };
        return error.UnsupportedShape;
    }

    pub fn deinit(self: *Face) void {
        for (self.runs[0..self.count]) |*run| run.linear.deinit();
        self.* = undefined;
    }

    /// The face's bf16 matmul where it is one: what a kernel that reads the weight itself takes (`hc_window.cu`).
    pub fn plain(self: Face) ?bf16.Linear {
        if (self.count == 1 and self.runs[0].linear == .bf16) return self.runs[0].linear.bf16;
        return null;
    }

    /// What `forward` into fp32 `out` would write, as K slices for the kernel that reads it next to sum
    /// (`cuda.Slices`): the matmul's own where the face is one bf16 matmul, else `out` itself, written.
    pub fn slices(self: Face, x: cuda.View, out: cuda.View) cuda.Error!cuda.Slices {
        if (self.count == 1 and self.runs[0].linear == .bf16) return self.runs[0].linear.bf16.slices(x, out);
        try self.forward(x, out);
        return .one(out);
    }

    /// out (m, n) = x (m, k) bf16 times the stacked weight, bf16 or fp32 by `out`'s dtype. A run's rows come from
    /// its own kernel, whose bits do not depend on the row count, and reach their columns by a copy: a row's bits
    /// are the same alone and among others.
    pub fn forward(self: Face, x: cuda.View, out: cuda.View) cuda.Error!void {
        if (self.count == 1) return self.runs[0].linear.forward(x, out);
        const m: usize = @intCast(x.size[0]);
        std.debug.assert(m <= self.max_rows and out.ndim == 2 and out.size[1] == self.n and (out.row_stride == 0 or out.row_stride == out.size[1]));
        const width = out.dtype.size();
        std.debug.assert(width == 2 or width == 4);
        for (self.runs[0..self.count]) |run| {
            const outputs = run.linear.outputs();
            // one row is contiguous whatever its columns, so the kernel writes them itself; the matmuls want
            // their output on a 16-byte boundary
            const base: [*]u8 = @ptrCast(out.ptr.?);
            if ((run.at * width) % 16 == 0 and m == 1) {
                try run.linear.forward(x, .init(base + run.at * width, out.dtype, &.{ 1, outputs }));
                continue;
            }
            const staged: cuda.View = .init(self.staging.?.view.ptr.?, out.dtype, &.{ m, outputs });
            try run.linear.forward(x, staged);
            try cuda.launched(cuda.kernels.zs_columns_place(&staged, &out, @intCast(run.at)));
        }
    }

    /// Prompt rows take `forward`: its bits.
    pub fn prefill(self: Face, x: cuda.View, out: cuda.View, _: u3) cuda.Error!void {
        return self.forward(x, out);
    }
};
