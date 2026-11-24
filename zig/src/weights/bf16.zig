//! BF16 linears: the weights a checkpoint keeps unquantized, on a matmul where a row's bits never depend on the
//! other rows of the call.

const std = @import("std");
const cuda = @import("../cuda/cuda.zig");

/// K slices for an (n, k) weight: a power of two up to 32 with whole blocks of 64 a slice, by the shape alone.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const blocks = k / 64;
    var sk: usize = 1;
    while (sk < 32 and tiles * sk < 160 and blocks % (sk * 2) == 0 and blocks / (sk * 2) >= 1) sk *= 2;
    return sk;
}

/// A [n, k] BF16 weight as the checkpoint stores it.
pub const Linear = struct {
    weight: cuda.Buffer,
    /// fp32 [sk, max_rows, n] when the weight splits K: the pool's (`cuda.Scratch`), dead once its sum is read
    part: ?cuda.Buffer,
    scratch: *cuda.Scratch,
    n: usize,
    k: usize,
    sk: usize,
    max_rows: usize,

    /// `weight` bf16 [n, k] row-major, the checkpoint's bytes; calls take up to `rows.count` rows.
    pub fn init(weight: []const u8, n: usize, k: usize, rows: cuda.Scratch.Rows) !Linear {
        std.debug.assert(weight.len == n * k * 2);
        return initStacked(&.{weight}, k, rows);
    }

    /// Several [rows, k] weights of one input stacked in order into one [n, k], each uploaded where it lands.
    pub fn initStacked(parts: []const []const u8, k: usize, rows: cuda.Scratch.Rows) !Linear {
        if (k % 64 != 0) return error.UnsupportedShape;
        var total: usize = 0;
        for (parts) |part| total += part.len;
        const n = total / (k * 2);
        std.debug.assert(n * k * 2 == total);
        const sk = splitK(n, k);
        var buffer: cuda.Buffer = try .init(.bf16, &.{ n, k });
        errdefer buffer.deinit();
        var at: usize = 0;
        for (parts) |part| {
            try buffer.uploadAt(at, part);
            at += part.len;
        }
        const part: ?cuda.Buffer = if (sk > 1) try rows.pool.lend(.part, .f32, &.{ sk, rows.count, n }) else null;
        return .{ .weight = buffer, .part = part, .scratch = rows.pool, .n = n, .k = k, .sk = sk, .max_rows = rows.count };
    }

    pub fn deinit(self: *Linear) void {
        self.weight.deinit();
        self.* = undefined;
    }

    /// out (m, n) = x (m, k) bf16 times the weight, bf16 or fp32 by `out`'s dtype, on the CUTLASS GEMM with the tile
    /// the row count picks. Tiles never change bits.
    pub fn forward(self: Linear, x: cuda.View, out: cuda.View) cuda.Error!void {
        return self.gemm(x, out, .auto);
    }

    /// The product on CUTLASS's GEMM, which reads the weight once for all the rows, on the tile asked. Its bits are
    /// the hand-written kernel's it replaced (`zig/tests/kernels/bf16/matmul.cu`, which the tests hold it against).
    pub fn gemm(self: Linear, x: cuda.View, out: cuda.View, tile: Tile) cuda.Error!void {
        std.debug.assert(x.size[0] <= self.max_rows);
        try self.scratch.enter(x);
        const part: ?*const cuda.View = if (self.part) |*part| &part.view else null;
        return cuda.launched(cuda.kernels.zs_b16_gemm(&x, &self.weight.view, &out, part, @intCast(self.sk), tile.of(@intCast(x.size[0]), self.sk, self.n, self.k)));
    }

    /// `forward` into fp32 `out` [m, n] without the reduction: the K slices, for a kernel that sums them as it
    /// reads them (`cuda.Slices`). A weight of one slice writes `out`, which is then the slice, and so does a
    /// window from `sums_from` rows, whose GEMM sums its slices in the block: the same bits.
    pub fn slices(self: Linear, x: cuda.View, out: cuda.View) cuda.Error!cuda.Slices {
        const m: usize = @intCast(x.size[0]);
        std.debug.assert(m <= self.max_rows and out.dtype == .f32);
        const part = if (self.part != null and cuda.Slices.folded and m < sums_from) self.part.?.view else {
            try self.forward(x, out);
            return .one(out);
        };
        try self.scratch.enter(x);
        try cuda.launched(cuda.kernels.zs_b16_gemm(&x, &self.weight.view, null, &part, @intCast(self.sk), Tile.auto.of(m, self.sk, self.n, self.k)));
        return .of(part, self.sk, m, self.n);
    }

    /// Prompt rows take the GEMM; its bits are `forward`'s.
    pub fn prefill(self: Linear, x: cuda.View, out: cuda.View, _: u3) cuda.Error!void {
        return self.forward(x, out);
    }
};

/// The GEMM's threadblock tile: 16 rows (`narrow`, `slim`, `deep`) or 128 (`wide`, `half`); `auto` by the row count.
pub const Tile = enum(i64) {
    auto = 0,
    narrow = 1,
    wide = 2,
    /// 32 columns a block, four blocks an SM: a weight split over K (spec 073).
    slim = 3,
    /// 8 columns a block and eight stages on their way: a weight of one K slice (spec 073).
    deep = 4,
    /// `wide` with stages of 32 values of K, two blocks an SM: a window (spec 073).
    half = 5,

    /// The tile a call of `m` rows on an [n, k] weight of `sk` K slices launches on when it asks `self`. For
    /// `auto` with `tiles_by_shape`: up to 16 rows the one the weight's slices pick, a narrow input (`deep_from`)
    /// kept on the base's; a window takes `half` where it is
    /// not slower than `wide` (`gemm.cu`): from 2048 rows, from 1024 but for a weight in 32 slices, from 256 a
    /// weight of one slice that L2 holds. Tiles never change bits.
    pub fn of(self: Tile, m: usize, sk: usize, n: usize, k: usize) i64 {
        if (self != .auto or !tiles_by_shape) return @backingInt(self);
        if (m <= 16) return @backingInt(if (sk > 1) Tile.slim else if (k >= deep_from) Tile.deep else Tile.auto);
        const held = sk == 1 and n * k * 2 <= 24 << 20;
        return @backingInt(if (m >= 2048 or (m >= 1024 and sk < 32) or (m >= 256 and held)) Tile.half else Tile.auto);
    }
};

/// From this many rows a call's GEMM sums a weight's K slices inside itself (CUTLASS's serial split-K on the
/// 128-row tiles, `gemm.cu`'s `SUMS_FROM`): each slice its own fp32 sum, added from slice 0 in slice order, so
/// `b16_reduce`'s bits, with the sum kept in L2 instead of every slice written (spec 095 WS-03). Under it the
/// split is written whole for its reduction or its reader: the serial chain is as long at 64 rows as at 2,048.
pub const sums_from: usize = 1024;

/// A half-precision GEMM on the tile its rows and its weight pick (`Tile.of`, `gemm.cu`) instead of the narrow
/// tile up to 16 rows and the wide one above: the same bits, 1 to 5% off a decoding launch alone and 5 to 18% off
/// a window's; with the experts' whole rounds 2.5% of the published NVFP4 checkpoint's decode tokens a second
/// and 1.4% of a cold prompt (spec 073 WS-04). `ZSPARK_GEMM_TILES=0` (`main` reads it) keeps the base's tiles.
pub var tiles_by_shape: bool = true;

/// The K from which a weight of one slice takes the deep tile up to 16 rows. Under it the eight stages have
/// nothing to hold: Flash Next's read-out up (320 to 10,240) was 34.2 us on the base's tile and 35.6 on the
/// deep one in a served round of the NVFP4 checkpoint, 35.7 and 36.2 on the pack, though faster alone (spec
/// 073 WS-06b). The wide products (2,560 and up) gain 3 to 6%. `ZSPARK_GEMM_TILES=2` sets 0, the deep tile on
/// every such weight, for a comparison.
pub var deep_from: usize = 1024;

/// Routed experts on bf16 weights (`zig/kernels/bf16/experts.cu`, spec 137): a call's pairs grouped by
/// `experts.Plan` in items of `item` pairs, each pair's sums its own row's against its expert's weight.
pub const Experts = struct {
    /// Pairs an item of the plan holds (`experts.Plan.routeAs` with this tile).
    pub const item = 64;

    /// Items a plan of `pairs` in items of `item` can hold: one per used expert, one per `item` pairs past its first.
    pub fn maxItems(pairs: usize, experts: usize) usize {
        return @min(pairs, experts) + pairs / item;
    }

    /// x bf16 [rows, hidden] (pair p = row * top + slot), gu bf16 [experts, 2 width, hidden] (gate rows, then up
    /// rows) -> act bf16 [pairs, width] in the plan's order: bf16(bf16(silu(bf16 gate)) * bf16 up).
    pub fn up(x: cuda.View, gu: cuda.View, members: cuda.View, items: cuda.View, counts: cuda.View, top: usize, act: cuda.View) cuda.Error!void {
        const pairs: usize = @as(usize, @intCast(x.size[0])) * top;
        return cuda.launched(zs_bf16_experts_up(&x, &gu, &members, &items, &counts, @intCast(top), @intCast(maxItems(pairs, @intCast(gu.size[0]))), &act));
    }

    /// act bf16 [pairs, width] in the plan's order, down bf16 [experts, hidden, width] -> y bf16 [pairs, hidden]
    /// at each pair.
    pub fn down(act: cuda.View, w: cuda.View, members: cuda.View, items: cuda.View, counts: cuda.View, y: cuda.View) cuda.Error!void {
        const pairs: usize = @intCast(y.size[0]);
        return cuda.launched(zs_bf16_experts_down(&act, &w, &members, &items, &counts, @intCast(maxItems(pairs, @intCast(w.size[0]))), &y));
    }
};

extern fn zs_bf16_experts_up(x: *const cuda.View, gu: *const cuda.View, members: *const cuda.View, items: *const cuda.View, counts: *const cuda.View, top: i64, max_items: i64, act: *const cuda.View) c_int;
extern fn zs_bf16_experts_down(act: *const cuda.View, down: *const cuda.View, members: *const cuda.View, items: *const cuda.View, counts: *const cuda.View, max_items: i64, y: *const cuda.View) c_int;
