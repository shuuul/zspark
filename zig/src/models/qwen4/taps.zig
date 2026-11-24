//! DFlash2's completed-layer taps: the mean over Flash Next's four residual streams, five layers concatenated.
//! CUDA stays out of this file so the host tests run without a GPU; `hc.streamMean` writes the same bits on the
//! device. Spec 122, the pinned contract in `~/.cache/zspark/scratch-122/tap_contract.txt`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const numeric = @import("../../weights/numeric.zig");

/// The pinned drafter's `dflash_config.target_layer_ids`: completed zero-based layers, rising.
pub const layers = [_]u32{ 5, 15, 25, 35, 45 };

/// Flash Next's hidden width and residual stream count; the drafter's `fc` is [hidden, layers.len * hidden].
pub const hidden: usize = 2560;
pub const streams: usize = 4;
pub const concat_width: usize = layers.len * hidden;

/// How the four streams are reduced to one row of `hidden`.
pub const Mean = enum(u8) {
    /// GLM: f32 acc of the four bf16 values in stream order, divide by 4, round to nearest even. Default until
    /// a fixture from the SGLang capture says otherwise (spec 122: torch.mean on bf16 may differ).
    f32_acc = 0,
    /// Round to bf16 after each add in stream order, then divide in f32 and round. Differs from `f32_acc`
    /// where a large stream swallows a small one at bf16's ULP.
    bf16 = 1,
};

/// The slot of `layer` among `layers`, or null (an off-by-one is not a tap).
pub fn tapIndex(layer: usize) ?usize {
    return tapIndexOf(&layers, layer);
}

pub fn tapIndexOf(at: []const u32, layer: usize) ?usize {
    for (at, 0..) |one, i| if (one == layer) return i;
    return null;
}

/// x bf16 [`nstreams` * dims] -> out bf16 [dims], one row. Stream 0 is the first `dims` values.
pub fn streamMean(x: []const u16, out: []u16, nstreams: usize, mean: Mean) void {
    std.debug.assert(nstreams >= 1 and x.len == out.len * nstreams);
    const dims = out.len;
    const n: f32 = @floatFromInt(nstreams);
    switch (mean) {
        .f32_acc => {
            for (out, 0..) |*o, d| {
                var acc = numeric.f32FromBf16(x[d]);
                for (1..nstreams) |s| acc += numeric.f32FromBf16(x[s * dims + d]);
                o.* = numeric.bf16FromF32(acc / n);
            }
        },
        .bf16 => {
            for (out, 0..) |*o, d| {
                var acc = x[d];
                for (1..nstreams) |s| {
                    acc = numeric.bf16FromF32(numeric.f32FromBf16(acc) + numeric.f32FromBf16(x[s * dims + d]));
                }
                o.* = numeric.bf16FromF32(numeric.f32FromBf16(acc) / n);
            }
        },
    }
}

/// x bf16 [rows, `nstreams` * dims] -> out bf16 [rows, dims]. A row's mean is that row alone, never a reduction
/// across rows.
pub fn streamMeanRows(x: []const u16, out: []u16, rows: usize, nstreams: usize, mean: Mean) void {
    std.debug.assert(rows >= 1 and out.len % rows == 0);
    const dims = out.len / rows;
    std.debug.assert(x.len == rows * nstreams * dims);
    for (0..rows) |r| {
        streamMean(
            x[r * nstreams * dims ..][0 .. nstreams * dims],
            out[r * dims ..][0..dims],
            nstreams,
            mean,
        );
    }
}

/// `n` slots of [rows, dims] packed as [rows, n * dims], each row's slots side by side (the drafter's `fc` input).
pub fn concat(slots: []const []const u16, rows: usize, dims: usize, out: []u16) void {
    const n = slots.len;
    std.debug.assert(n >= 1 and out.len == rows * n * dims);
    for (slots, 0..) |slot, s| {
        std.debug.assert(slot.len == rows * dims);
        for (0..rows) |r| {
            @memcpy(out[r * n * dims + s * dims ..][0..dims], slot[r * dims ..][0..dims]);
        }
    }
}

/// Host capture of completed-layer means: `values` is bf16 [rows, layers.len * hidden], reserved at init.
pub const Taps = struct {
    layers: []const u32,
    hidden: usize,
    streams: usize,
    rows: usize,
    mean: Mean,
    values: []u16,
    gpa: Allocator,

    pub fn init(
        gpa: Allocator,
        tap_layers: []const u32,
        hidden_dim: usize,
        nstreams: usize,
        rows: usize,
        mean: Mean,
    ) Allocator.Error!Taps {
        std.debug.assert(tap_layers.len >= 1 and hidden_dim >= 1 and nstreams >= 1 and rows >= 1);
        const values = try gpa.alloc(u16, rows * tap_layers.len * hidden_dim);
        @memset(values, 0);
        return .{
            .layers = tap_layers,
            .hidden = hidden_dim,
            .streams = nstreams,
            .rows = rows,
            .mean = mean,
            .values = values,
            .gpa = gpa,
        };
    }

    pub fn deinit(self: *Taps) void {
        self.gpa.free(self.values);
        self.* = undefined;
    }

    pub fn tapIndex(self: Taps, layer: usize) ?usize {
        return tapIndexOf(self.layers, layer);
    }

    /// Mean the completed residual `h` bf16 [rows, streams * hidden] into this layer's slot. False if `layer`
    /// is not a tap (off-by-one leaves `values` unchanged).
    pub fn capture(self: *Taps, layer: usize, h: []const u16) bool {
        const at = self.tapIndex(layer) orelse return false;
        const wide = self.layers.len * self.hidden;
        std.debug.assert(h.len == self.rows * self.streams * self.hidden);
        for (0..self.rows) |r| {
            streamMean(
                h[r * self.streams * self.hidden ..][0 .. self.streams * self.hidden],
                self.values[r * wide + at * self.hidden ..][0..self.hidden],
                self.streams,
                self.mean,
            );
        }
        return true;
    }

    /// Slot `index`'s columns of `row`: [hidden].
    pub fn slot(self: Taps, index: usize, row: usize) []u16 {
        const wide = self.layers.len * self.hidden;
        return self.values[row * wide + index * self.hidden ..][0..self.hidden];
    }
};
