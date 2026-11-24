//! NVFP4 and FP8 linears: weights packed into the lane matmul's order and the launches that read them.
//! The layouts are TensorFold's (cuda/nvfp4/linear.py).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");
const numeric = @import("numeric.zig");

pub const Mode = enum(i64) { fp4 = 0, fp8 = 1, mxfp8 = 2, fp8_grouped = 3 };

/// K slices for an (n, k) weight: a function of the shape alone, never of the row count, so a row's bits
/// stay the same in every verify window.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// Tiles from which a lane matmul of up to 32 rows split in K sums a tile's slices in one block (`qmmf.cu`'s
/// tiles of -16 and -32 rows) instead of a cluster: the same bits. 0 never. Spec 105 WS-03, alone on random
/// weights (2560 x 6144, 40 tiles of 8 slices, 1 to 16 rows): MXFP8 82 to 85 us by the cluster, 69.4 to 70.0 so;
/// NVFP4 55 to 57.5, 45.3 to 49.3; at 24 rows 97 to 100 against 72, 77 against 54.
pub var lane_fused_tiles: usize = 40;

/// The row tile for `m` rows; tiles never change bits.
pub fn rowTile(m: usize) usize {
    std.debug.assert(m >= 1);
    return if (m <= 16) 16 else if (m <= 32) 32 else 64;
}

/// Rows from which each block sums its tile's K slices itself.
pub const fused_rows = 256;

/// Rows from which an MXFP8 or block-FP8 lane matmul runs on the 128-row tile, its K slices summed in the block: the same bits;
/// 0 never. A warp's weights decoded once for 128 rows instead of 64 (spec 105 WS-03, `ZSPARK_NVFP4_PROFILE=lanewide`,
/// FP4-Local's shapes at 2,048 rows, us for the 64-row tile's): 2,723 for 3,109 (16,512 x 2,560), 2,319 for 2,645
/// (13,952 x 2,560), 1,255 for 1,540 (2,560 x 6,144), 275 for 326 (1,280 x 2,560), 125 for 141 (2,560 x 640); at 256
/// rows all but 1,280 x 2,560 (50.5 for 49.4) less. Block FP8 the same from spec 107 WS-03 (`glmlinears`, GLM
/// FP4-Local's shapes on a rank at 2,048 rows, us for the 64-row tile's): 646.5 for 710.7 (2,048 x 4,096), 351.1
/// for 363.7 (4,096 x 1,024), 960.0 for 1,076.1 (8,192 x 1,536), 2,454.4 for 2,695.3 (4,096 x 8,192).
pub var lane_wide_rows: usize = 256;

/// Windows of up to this many rows (16 at most) run a block-FP8 linear on `qmmf_tile.cu`: one scale a tile and
/// group instead of 64, and from `fused_tiles` a block sums its K slices instead of a cluster. The lane matmul's
/// bits. 0 never. `ZSPARK_FP8_TILES` sets it.
/// On since spec 070: the default layout's prose 48.2 to 50.1 tokens a second, code 92.1 to 94.7, the same replies.
pub var tile_rows: usize = 16;

/// Tiles from which `qmmf_tile.cu` sums a tile's K slices in one block: fewer blocks than these leave SMs idle
/// (20 tiles: 20.3 us fused, 19.6 by the cluster; 40 tiles of 6,144 inputs: 72.7 against 86.2). GLM FP4-Local's
/// shared expert's gate and up, 32 tiles of 8 slices on a rank (spec 114 WS-02, `glmdecode`, a copy a launch):
/// 39.5 us at 1 and 6 rows in the block, 49.0 by the cluster, the same bits.
pub const fused_tiles = 32;

/// e4m3 bytes [n, k] row-major into the FP8 GEMM's fragment order [npad/64][k/64][8][32][2][8], padded
/// outputs zero. `out` holds npad * k bytes.
pub fn fragmentOrder(codes: []const u8, n: usize, k: usize, npad: usize, out: []u8) void {
    std.debug.assert(codes.len == n * k);
    fragmentOrderStacked(&.{.{ .codes = codes, .block_scales = .{ .data = &.{} }, .n = n }}, k, npad, out);
}

/// `fragmentOrder` over several [rows, k] weights of one input stacked in order, without joining them first.
pub fn fragmentOrderStacked(parts: []const Rows, k: usize, npad: usize, out: []u8) void {
    std.debug.assert(out.len == npad * k and k % 64 == 0 and npad % 64 == 0);
    for (parts) |part| std.debug.assert(part.codes.len == part.n * k);
    const tiles = npad / 64;
    // a tile's bytes are its own, so tiles pack in parallel once there is enough to pay for the threads
    const workers = if (out.len < (4 << 20)) 1 else @min(tiles, std.Thread.getCpuCount() catch 1);
    var threads: [64]std.Thread = undefined;
    var spawned: usize = 0;
    for (0..@min(workers, threads.len) - 1) |w| {
        threads[spawned] = std.Thread.spawn(.{}, fragmentShare, .{ parts, k, tiles, out, w + 1, workers }) catch break;
        spawned += 1;
    }
    // the shares no thread took, this thread's own included
    fragmentShare(parts, k, tiles, out, 0, workers);
    for (spawned + 1..workers) |w| fragmentShare(parts, k, tiles, out, w, workers);
    for (threads[0..spawned]) |thread| thread.join();
}

/// Tiles `share`, `share + of`, ... of `fragmentOrderStacked`.
fn fragmentShare(parts: []const Rows, k: usize, tiles: usize, out: []u8, share: usize, of: usize) void {
    const inputs = fragmentInputs();
    var tile = share;
    while (tile < tiles) : (tile += of) {
        var rows: [64]?[*]const u8 = undefined;
        for (&rows, 0..) |*row, column| {
            row.* = if (locate(parts, tile * 64 + column)) |found| parts[found.part].codes[found.row * k ..].ptr else null;
        }
        var at = tile * 64 * k;
        for (0..k / 64) |group| for (0..8) |block| for (0..32) |lane| {
            const row = rows[block * 8 + lane / 4];
            for (inputs[lane % 4]) |by_byte| for (by_byte) |input| {
                out[at] = if (row) |codes| codes[group * 64 + input] else 0;
                at += 1;
            };
        };
    }
}

/// The part holding output `output` of a stack and the output's row in it; null in the padding.
/// Where a lane (by its index modulo 4), half and byte read within a group of 64 inputs, in fragment order.
fn fragmentInputs() [4][2][8]u8 {
    var inputs: [4][2][8]u8 = undefined;
    for (&inputs, 0..) |*by_half, lane| for (by_half, 0..) |*by_byte, half| for (by_byte, 0..) |*input, byte| {
        const p = 16 * (byte / 4) + 4 * lane + (byte % 4);
        const kin = 16 * (p / 16) + 2 * ((p % 16) / 4) + (p % 2) + 8 * ((p % 4) / 2);
        input.* = @intCast(half * 32 + kin);
    };
    return inputs;
}

fn locate(parts: []const Rows, output: usize) ?struct { part: usize, row: usize } {
    var first: usize = 0;
    for (parts, 0..) |part, index| {
        if (output < first + part.n) return .{ .part = index, .row = output - first };
        first += part.n;
    }
    return null;
}

fn paddedOutputs(n: usize) usize {
    return (n + 127) / 128 * 128;
}

/// bm: the row tile, 16, 32 or 64; 0 a tile of 64 rows whose block sums its K slices itself, -16 and -32 the
/// same at 16 and 32 rows, 128 (mode 2 only) a tile of 128 rows that does. Every form the same bits.
extern fn zs_nvfp4_qmmf(x: *const cuda.View, w: *const cuda.View, bs: ?*const cuda.View, scale: f64, out: *const cuda.View, part: ?*const cuda.View, mode: i64, n: i64, sk: i64, npad: i64, bm: i64, f32: c_int) c_int;
extern fn zs_nvfp4_prompt16(x: *const cuda.View, w: *const cuda.View, bs: ?*const cuda.View, scale: f64, out: *const cuda.View, mode: i64, n: i64, npad: i64, tile: i64, f32: c_int) c_int;
extern fn zs_nvfp4_linear_quant(x: *const cuda.View, out: *const cuda.View, gs: f64) c_int;

/// The launch every lane-matmul linear shares: K slices and the row tile follow from the shapes.
fn laneMatmul(mode: Mode, w: cuda.View, bs: ?*const cuda.View, scale: f64, n: usize, k: usize, npad: usize, x: cuda.View, out: cuda.View) cuda.Error!void {
    const m: usize = @intCast(x.size[0]);
    const sk = splitK(n, k);
    const tile: i64 = if ((mode == .mxfp8 or mode == .fp8_grouped) and lane_wide_rows > 0 and m >= lane_wide_rows) 128 else if (sk > 1 and m >= fused_rows) 0 else @intCast(rowTile(m));
    // a block sums its tile's slices where the tiles fill the SMs alone: the cluster's bits
    const own = sk > 1 and tile > 0 and tile < 64 and lane_fused_tiles > 0 and (n + 63) / 64 >= lane_fused_tiles;
    return cuda.launched(zs_nvfp4_qmmf(&x, &w, bs, scale, &out, null, @backingInt(mode), @intCast(n), @intCast(sk), @intCast(npad), if (own) -tile else tile, @intFromBool(out.dtype == .f32)));
}

fn prompt(mode: Mode, w: cuda.View, bs: ?*const cuda.View, scale: f64, n: usize, npad: usize, x: cuda.View, out: cuda.View, tile: u3) cuda.Error!void {
    return cuda.launched(zs_nvfp4_prompt16(&x, &w, bs, scale, &out, @backingInt(mode), @intCast(n), @intCast(npad), tile, @intFromBool(out.dtype == .f32)));
}

/// Activations quantized as the card's vLLM quantizes them before an MXFP8 or block-FP8 linear, products on FP8
/// tensor cores (`qmm_a8.cu`, spec 114 WS-04): a mode a model sets on its linears (`quantizeActivations`), GLM
/// FP4-Local's; a linear without it keeps bf16 rows and the lane matmul's bits (Flash Next's). An MXFP8 linear
/// quantizes a call's rows from its pool's `quantize_from` on (spec 115: the card runs a step's and a verify
/// window's MXFP8 rows bf16); a block-FP8 one every row, as the card does in every phase.
pub const Activations = struct {
    /// u8 [rows * k * 33 / 32]: rows past `a8_tile_rows` quantized and their scales, the pool's part; null
    /// where no call takes more
    work: ?cuda.Buffer,
    pool: *cuda.Scratch,
    /// Zero keeps dynamic groups; positive uses a checkpoint's per-tensor activation scale.
    input_scale: f32 = 0,

    fn of(rows: cuda.Scratch.Rows, k: usize) !Activations {
        return .{ .work = try rows.pool.lend(.part, .u8, &.{rows.count * k / 32 * 33}), .pool = rows.pool };
    }
};

/// Windows of up to this many rows quantize their activations in the matmul's blocks, where the weight is on its
/// way (`qmm_a8.cu`'s bm 16); more rows through `quant_a8_kernel` first. The same bytes, scales and bits.
pub const a8_tile_rows = 16;

/// Whether `quantizeActivations` takes a linear's rows to FP8; a test sets it false before a load to hold the bf16
/// rows' path beside it.
pub var quantized_activations = true;

/// Whether a linear of `k` inputs can take quantized activations: whole stages of 128 inputs.
fn a8Fits(k: usize) bool {
    return k % 128 == 0;
}

/// out (m, n) = quant(x) (m, k) times the weight on FP8 tensor cores; K slices by shape, a row tile by the rows,
/// neither changing a bit.
fn a8Matmul(mode: Mode, a8: Activations, w: cuda.View, bs: cuda.View, n: usize, k: usize, npad: usize, x: cuda.View, out: cuda.View) cuda.Error!void {
    const m: usize = @intCast(x.size[0]);
    try a8.pool.enter(x);
    var sk = splitK(n, k);
    // The lane matmul splits 64-input groups; FP8 tensor cores consume whole 128-input stages.
    while (k / 128 % sk != 0) sk /= 2;
    const bm: i64 = if (m <= a8_tile_rows) 16 else 64;
    // a block sums its tile's slices where the tiles fill the SMs alone (`fused_tiles`): the cluster's bits
    const fuse = sk == 1 or (n + 63) / 64 >= fused_tiles;
    const work: ?cuda.View = if (a8.work) |*w8buf| w8buf.view else null;
    return cuda.launched(cuda.kernels.zs_nvfp4_qmm_a8(&x, &w, &bs, &out, if (work) |*view| view else null, @backingInt(mode), @intCast(n), @intCast(sk), @intCast(npad), bm, @intFromBool(fuse), @intFromBool(out.dtype == .f32), a8.input_scale));
}

extern fn zs_nvfp4_prompt_bias(x: *const cuda.View, w: *const cuda.View, bs: ?*const cuda.View, scale: f64, bias: *const cuda.View, out: *const cuda.View, mode: i64, n: i64, npad: i64, tile: i64, act: i64) c_int;

/// The activation `forwardBiased` applies to the bias-added sum rounded to bf16 (`prompt_bias.cu`).
pub const Activation = enum(i64) { none = 0, gelu_tanh = 1, gelu_erf = 2 };

/// `prompt_bias.cu`'s tile for `m` rows (0: 128 x 128 on 8 warps, 3 stages; 4: on 4 warps, 2 stages; 5: 64 x
/// 128 on 2 warps): by time alone, a tile changes no bit. `null` takes tile 4; a test sets one.
pub var biased_tile: ?u3 = null;

/// Tile 4 at every count of rows: two blocks an SM where tile 0 is one. A mixed-precision checkpoint's vision tower,
/// wall time a pass (spec 019 WS-08, 2026-10-18): 4,096 patches 101 ms against 112 for tile 0 and 110 for tile
/// 5, 1,024 patches 22.2 against 24.6 and 25.0, 256 patches 9.3 against 9.5 and 9.8 (within the passes' spread).
pub fn biasedTile(m: usize) u3 {
    if (biased_tile) |tile| return tile;
    _ = m;
    return 4;
}

/// out (m, w) = act(x (m, k) W^T * scale + bias), bf16, for many rows (`prompt_bias.cu`): one chain over K and no
/// split, so a row's bits are its own; w = out's columns, from the weight's n up to npad, the padded ones zero
/// where the bias is.
fn biased(mode: Mode, w: cuda.View, bs: ?*const cuda.View, scale: f64, npad: usize, bias: cuda.View, x: cuda.View, out: cuda.View, act: Activation) cuda.Error!void {
    const m: usize = @intCast(x.size[0]);
    return cuda.launched(zs_nvfp4_prompt_bias(&x, &w, bs, scale, &bias, &out, @backingInt(mode), out.size[1], @intCast(npad), biasedTile(m), @backingInt(act)));
}

fn uploaded(gpa: Allocator, comptime T: type, dtype: cuda.DType, shape: []const usize, fill: anytype) !cuda.Buffer {
    var count: usize = 1;
    for (shape) |dim| count *= dim;
    const host = try gpa.alloc(T, count);
    defer gpa.free(host);
    fill.write(host);
    var buffer: cuda.Buffer = try .init(dtype, shape);
    errdefer buffer.deinit();
    try buffer.upload(std.mem.sliceAsBytes(host));
    return buffer;
}

/// An FP8 projection: e4m3 bytes in fragment order and one scale for the tensor.
pub const Fp8Linear = struct {
    w8: cuda.Buffer,
    scale: f64,
    n: usize,
    k: usize,
    npad: usize,

    /// `codes` e4m3 [n, k] row-major with the tensor's scale, as ModelOpt stores per-tensor FP8.
    pub fn init(gpa: Allocator, codes: []const u8, n: usize, k: usize, scale: f64) !Fp8Linear {
        if (k % 64 != 0) return error.UnsupportedShape;
        const npad = paddedOutputs(n);
        const Fill = struct {
            codes: []const u8,
            n: usize,
            k: usize,
            npad: usize,
            fn write(f: @This(), out: []u8) void {
                fragmentOrder(f.codes, f.n, f.k, f.npad, out);
            }
        };
        const w8 = try uploaded(gpa, u8, .u8, &.{npad * k}, Fill{ .codes = codes, .n = n, .k = k, .npad = npad });
        return .{ .w8 = w8, .scale = scale, .n = n, .k = k, .npad = npad };
    }

    pub fn deinit(self: *Fp8Linear) void {
        self.w8.deinit();
        self.* = undefined;
    }

    /// out (m, n) = x (m, k) bf16 times the weight, bf16 or fp32 by `out`'s dtype.
    pub fn forward(self: Fp8Linear, x: cuda.View, out: cuda.View) cuda.Error!void {
        return laneMatmul(.fp8, self.w8.view, null, self.scale, self.n, self.k, self.npad, x, out);
    }

    /// Prompt rows: out (m, n) = x (m, k) bf16 times the weight in one fp32 chain over K, so a row's bits depend
    /// neither on its chunk nor on `tile` (0 to 5, the block shape).
    pub fn prefill(self: Fp8Linear, x: cuda.View, out: cuda.View, tile: u3) cuda.Error!void {
        return prompt(.fp8, self.w8.view, null, self.scale, self.n, self.npad, x, out, tile);
    }

    /// out (m, w) = act(x W^T * scale + bias) for many rows, the bias bf16 [w] (`biased`), as `Fp4Linear`'s.
    pub fn forwardBiased(self: Fp8Linear, x: cuda.View, bias: cuda.View, out: cuda.View, act: Activation) cuda.Error!void {
        return biased(.fp8, self.w8.view, null, self.scale, self.npad, bias, x, out, act);
    }
};

/// The value of an E2M1 code: sign, two exponent bits, one mantissa bit.
pub const e2m1 = [16]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };

/// Nibble slot p of a lane's word holds input 32 v + 2 (lane % 4) + nibble_offsets[p] of its group of 64.
const nibble_offsets = [8]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };

/// An NVFP4 projection: E2M1 codes tiled for the lane matmul, an e4m3 scale per 16 inputs, one scale for the tensor.
pub const Fp4Linear = struct {
    /// int32 [npad/64, k/64, 8, 32, 2]
    words: cuda.Buffer,
    /// e4m3 [npad/64, k/64, 64, 4]: a 64-column tile's block scales together, like its words
    block_scales: cuda.Buffer,
    scale: f64,
    n: usize,
    k: usize,
    npad: usize,
    /// Normalized NVFP4 activations widened exactly to bf16; the global factor is in `scale`.
    a4: ?struct { work: cuda.Buffer, pool: *cuda.Scratch, gs: f32 } = null,

    /// `codes` [n, k/2], two E2M1 codes a byte with the lower input in the low nibble; `block_scales` e4m3
    /// [n, k/16]; `scale` the tensor's factor (ModelOpt's weight_scale_2).
    pub fn init(gpa: Allocator, codes: []const u8, block_scales: []const u8, n: usize, k: usize, scale: f64) !Fp4Linear {
        if (k % 64 != 0) return error.UnsupportedShape;
        std.debug.assert(codes.len == n * k / 2 and block_scales.len == n * k / 16);
        const npad = paddedOutputs(n);
        const Source = struct {
            codes: []const u8,
            block_scales: []const u8,
            n: usize,
            k: usize,
            npad: usize,

            fn code(f: @This(), output: usize, input: usize) u32 {
                if (output >= f.n) return 0;
                const byte = f.codes[output * (f.k / 2) + input / 2];
                return if (input % 2 == 0) byte & 0xf else byte >> 4;
            }
        };
        const source: Source = .{ .codes = codes, .block_scales = block_scales, .n = n, .k = k, .npad = npad };
        const Words = struct {
            source: Source,
            fn write(f: @This(), out: []u32) void {
                const src = f.source;
                var at: usize = 0;
                for (0..src.npad / 64) |tile| for (0..src.k / 64) |group| for (0..8) |block| for (0..32) |lane_index| for (0..2) |half| {
                    const output = tile * 64 + block * 8 + lane_index / 4;
                    var word: u32 = 0;
                    for (nibble_offsets, 0..) |offset, p| {
                        const input = group * 64 + 32 * half + 2 * (lane_index % 4) + offset;
                        word |= src.code(output, input) << @intCast(4 * p);
                    }
                    out[at] = word;
                    at += 1;
                };
            }
        };
        const Scales = struct {
            source: Source,
            fn write(f: @This(), out: []u8) void {
                const src = f.source;
                var at: usize = 0;
                for (0..src.npad / 64) |tile| for (0..src.k / 64) |group| for (0..64) |column| for (0..4) |quarter| {
                    const output = tile * 64 + column;
                    out[at] = if (output < src.n) src.block_scales[output * (src.k / 16) + group * 4 + quarter] else 0;
                    at += 1;
                };
            }
        };
        var words = try uploaded(gpa, u32, .i32, &.{ npad / 64, k / 64, 8, 32, 2 }, Words{ .source = source });
        errdefer words.deinit();
        const scales = try uploaded(gpa, u8, .u8, &.{ npad / 64, k / 64, 64, 4 }, Scales{ .source = source });
        return .{ .words = words, .block_scales = scales, .scale = scale, .n = n, .k = k, .npad = npad };
    }

    pub fn deinit(self: *Fp4Linear) void {
        self.words.deinit();
        self.block_scales.deinit();
        self.* = undefined;
    }

    /// The checkpoint's NVFP4 activation rounding in every window. No weight is requantized.
    pub fn quantizeActivations(self: *Fp4Linear, rows: cuda.Scratch.Rows, input_scale: f32) !void {
        std.debug.assert(self.a4 == null);
        if (!(input_scale > 0) or !std.math.isFinite(input_scale)) return error.UnsupportedShape;
        const gs: f32 = 1.0 / input_scale;
        if (!std.math.isFinite(gs)) return error.UnsupportedShape;
        const work = try rows.pool.lend(.part, .bf16, &.{ rows.count, self.k });
        self.a4 = .{ .work = work, .pool = rows.pool, .gs = gs };
        self.scale = @as(f32, @floatCast(self.scale)) / gs;
    }

    fn inputRows(self: Fp4Linear, x: cuda.View) cuda.Error!cuda.View {
        const a4 = self.a4 orelse return x;
        try a4.pool.enter(x);
        const normalized: cuda.View = .init(a4.work.view.ptr.?, .bf16, &.{ @intCast(x.size[0]), self.k });
        try cuda.launched(zs_nvfp4_linear_quant(&x, &normalized, a4.gs));
        return normalized;
    }

    /// out (m, n) = x (m, k) bf16 times the weight: W4A16, every weight exact in bf16.
    pub fn forward(self: Fp4Linear, x: cuda.View, out: cuda.View) cuda.Error!void {
        return laneMatmul(.fp4, self.words.view, &self.block_scales.view, self.scale, self.n, self.k, self.npad, try self.inputRows(x), out);
    }

    pub fn prefill(self: Fp4Linear, x: cuda.View, out: cuda.View, tile: u3) cuda.Error!void {
        return prompt(.fp4, self.words.view, &self.block_scales.view, self.scale, self.n, self.npad, try self.inputRows(x), out, tile);
    }

    /// out (m, w) = act(x W^T + bias) for many rows, the bias bf16 [w] (`biased`).
    pub fn forwardBiased(self: Fp4Linear, x: cuda.View, bias: cuda.View, out: cuda.View, act: Activation) cuda.Error!void {
        return biased(.fp4, self.words.view, &self.block_scales.view, self.scale, self.npad, bias, try self.inputRows(x), out, act);
    }
};

/// Block scales as a checkpoint stores them (`weight_scale_inv`): little-endian fp32, or bf16 where an export keeps
/// them so. They may sit unaligned in a mapping.
pub const BlockScales = struct {
    data: []const u8,
    bf16: bool = false,

    pub fn of(values: []const f32) BlockScales {
        return .{ .data = std.mem.sliceAsBytes(values) };
    }

    pub fn len(self: BlockScales) usize {
        return self.data.len / @as(usize, if (self.bf16) 2 else 4);
    }

    pub fn at(self: BlockScales, index: usize) f32 {
        if (self.bf16) return numeric.f32FromBf16(std.mem.readInt(u16, self.data[index * 2 ..][0..2], .little));
        return @bitCast(std.mem.readInt(u32, self.data[index * 4 ..][0..4], .little));
    }
};

/// One MXFP8 weight of a stack.
pub const Mx8Rows = struct {
    /// e4m3 [n, k] row-major
    codes: []const u8,
    /// ue8m0 [n, k / 32]: each byte e scales its 32 inputs' codes by 2^(e - 127)
    exponents: []const u8,
    n: usize,
};

/// Inputs sharing one MXFP8 exponent.
pub const mx8_block = 32;

/// An MXFP8 projection (ModelOpt MXFP8): e4m3 bytes in the FP8 GEMM's fragment order and a ue8m0 exponent per
/// (output, 32 inputs), read as stored by the lane matmul's mode 2 with bf16 rows (W8A16): a block's products
/// summed from zero on the tensor cores, then added into the row's sum times its power of two. TensorFold's
/// cuda/nvfp4/linear.py `Mx8Linear`. Prompt rows take the same launch: TensorFold's `prefill` takes `prompt.cu`
/// where the exponents fold into bf16, one chain over K whose bits are not a step's, so zspark does not.
/// An exponent byte 0 multiplies by 0 (2^-127 lies under fp32's normals) and 255 by infinity, as the kernel reads them.
pub const Mx8Linear = struct {
    /// u8 [npad/64][k/64][8][32][2][8]
    w8: cuda.Buffer,
    /// u8 [npad/64, k/64, 64, 2]: a tile's exponents together, two a column a group; padded outputs 127
    exponents: cuda.Buffer,
    n: usize,
    k: usize,
    npad: usize,
    /// Set by `quantizeActivations`: rows quantized to MXFP8 and FP8 products
    a8: ?Activations = null,
    /// Set by `keepBf16From`: the first output column that keeps bf16 rows where `a8` quantizes the rest, and
    /// where those columns' rows land before they are placed (the pool's part, [rows, n - from] of 4 bytes)
    bf16_from: ?struct { column: usize, staging: cuda.Buffer } = null,

    /// `codes` e4m3 [n, k] and `exponents` ue8m0 [n, k/32], as a checkpoint stores them.
    pub fn init(gpa: Allocator, codes: []const u8, exponents: []const u8, n: usize, k: usize) !Mx8Linear {
        return initStacked(gpa, &.{.{ .codes = codes, .exponents = exponents, .n = n }}, k);
    }

    /// Several MXFP8 weights of one input stacked in order into one [n, k], as TensorFold's `Mx8Linear.stack`:
    /// the parts' outputs in order, one padding at the stack's end.
    pub fn initStacked(gpa: Allocator, parts: []const Mx8Rows, k: usize) !Mx8Linear {
        if (k % 64 != 0) return error.UnsupportedShape;
        const rows = try gpa.alloc(Rows, parts.len);
        defer gpa.free(rows);
        var n: usize = 0;
        for (parts, rows) |part, *row| {
            if (part.codes.len != part.n * k or part.exponents.len != part.n * (k / mx8_block)) return error.UnsupportedShape;
            row.* = .{ .codes = part.codes, .block_scales = .{ .data = &.{} }, .n = part.n };
            n += part.n;
        }
        const npad = paddedOutputs(n);
        const Codes = struct {
            rows: []const Rows,
            k: usize,
            npad: usize,
            fn write(f: @This(), out: []u8) void {
                fragmentOrderStacked(f.rows, f.k, f.npad, out);
            }
        };
        const Exponents = struct {
            parts: []const Mx8Rows,
            rows: []const Rows,
            k: usize,
            npad: usize,
            fn write(f: @This(), out: []u8) void {
                const per_row = f.k / mx8_block;
                var at: usize = 0;
                for (0..f.npad / 64) |tile| for (0..f.k / 64) |group| for (0..64) |column| for (0..2) |half| {
                    out[at] = if (locate(f.rows, tile * 64 + column)) |found|
                        f.parts[found.part].exponents[found.row * per_row + group * 2 + half]
                    else
                        127;
                    at += 1;
                };
            }
        };
        var w8 = try uploaded(gpa, u8, .u8, &.{npad * k}, Codes{ .rows = rows, .k = k, .npad = npad });
        errdefer w8.deinit();
        const exponents = try uploaded(gpa, u8, .u8, &.{ npad / 64, k / 64, 64, 2 }, Exponents{ .parts = parts, .rows = rows, .k = k, .npad = npad });
        return .{ .w8 = w8, .exponents = exponents, .n = n, .k = k, .npad = npad };
    }

    /// The weight as bf16 [n, k] row-major, read back from the device (`fragmentOrderStacked` undone), each
    /// value rounded to bf16, exact where it is one (`forward`'s premise, every checkpoint's weight so far): what
    /// a drafter's four-bit copy of an MXFP8 head is made of (GLM's `draftHeads`, spec 107 WS-04).
    pub fn bf16Rows(self: Mx8Linear, gpa: Allocator) ![]u8 {
        const k = self.k;
        const w8 = try gpa.alloc(u8, self.npad * k);
        defer gpa.free(w8);
        const exponents = try gpa.alloc(u8, self.npad / 64 * (k / 64) * 128);
        defer gpa.free(exponents);
        // the pages touched before the device writes them: a copy into fresh pages ran at 0.13 GB/s now and
        // then (3.7 s for 473 MiB, a GLM head's) against 54 GB/s into touched ones (`cuda-gb10`'s loading.md)
        @memset(w8, 0);
        @memset(exponents, 0);
        try self.w8.download(w8);
        try self.exponents.download(exponents);
        const out = try gpa.alloc(u8, self.n * k * 2);
        errdefer gpa.free(out);
        // a tile of 64 outputs is its own bytes: the tiles on threads (one thread took a second of a load's end)
        const tiles = self.npad / 64;
        const workers = @min(tiles, std.Thread.getCpuCount() catch 1, 32);
        var threads: [32]std.Thread = undefined;
        var spawned: usize = 0;
        for (1..workers) |w| {
            threads[spawned] = std.Thread.spawn(.{}, widenTiles, .{ self, w8, exponents, out, w, workers }) catch break;
            spawned += 1;
        }
        // the shares no thread took, this thread's own included
        widenTiles(self, w8, exponents, out, 0, workers);
        for (spawned + 1..workers) |w| widenTiles(self, w8, exponents, out, w, workers);
        for (threads[0..spawned]) |thread| thread.join();
        return out;
    }

    /// Tiles `share`, `share + of`, ... of `bf16Rows`.
    fn widenTiles(self: Mx8Linear, w8: []const u8, exponents: []const u8, out: []u8, share: usize, of: usize) void {
        const k = self.k;
        const inputs = fragmentInputs();
        var tile = share;
        while (tile < self.npad / 64) : (tile += of) {
            var at = tile * (k / 64) * 4096;
            for (0..k / 64) |group| for (0..8) |block| for (0..32) |lane| {
                const column = block * 8 + lane / 4;
                const output = tile * 64 + column;
                for (inputs[lane % 4]) |by_byte| for (by_byte) |input| {
                    defer at += 1;
                    if (output >= self.n) continue;
                    const exponent = exponents[((tile * (k / 64) + group) * 64 + column) * 2 + input / 32];
                    const value = numeric.f32FromE4m3(w8[at]) * numeric.f32FromUe8m0(exponent);
                    std.mem.writeInt(u16, out[(output * k + group * 64 + input) * 2 ..][0..2], numeric.bf16FromF32(value), .little);
                };
            };
        }
    }

    pub fn deinit(self: *Mx8Linear) void {
        self.w8.deinit();
        self.exponents.deinit();
        self.* = undefined;
    }

    /// From now on its rows are quantized as vLLM's b12x quantizes them (e4m3 under an e8m0 exponent a 32 inputs,
    /// `qmm_a8.cu`), every call of up to `rows.count` rows; false where `k` does not allow it, and the linear
    /// keeps bf16 rows.
    pub fn quantizeActivations(self: *Mx8Linear, rows: cuda.Scratch.Rows) !bool {
        if (!quantized_activations or !a8Fits(self.k)) return false;
        self.a8 = try .of(rows, self.k);
        return true;
    }

    /// Where `a8` quantizes a call's rows, its columns from `column` on (a whole 64-column tile, the weight's
    /// tiles after it) keep bf16 rows: the lane matmul of those tiles alone, placed over the quantized product's.
    /// What the card's vLLM runs where a stacked part is its own linear, raced A16 against quantized by time
    /// (spec 143 WS-08: DeltaNet's b and a, `in_proj_ba`, 96 outputs).
    pub fn keepBf16From(self: *Mx8Linear, rows: cuda.Scratch.Rows, column: usize) !void {
        std.debug.assert(self.a8 != null and column % 64 == 0 and column < self.n);
        const staging = try rows.pool.lend(.part, .u32, &.{ rows.count, self.n - column });
        self.bf16_from = .{ .column = column, .staging = staging };
    }

    /// out (m, n) = x (m, k) bf16 times the weight, bf16 or fp32 by `out`'s dtype; every weight exact in bf16. With
    /// `a8`, x's rows from the pool's `quantize_from` quantized to MXFP8 first, the rows before as bf16 rows (spec
    /// 115): each row's bits those of its form alone.
    pub fn forward(self: Mx8Linear, x: cuda.View, out: cuda.View) cuda.Error!void {
        const m: usize = @intCast(x.size[0]);
        if (self.a8) |a8| {
            const from = @min(a8.pool.quantize_from, m);
            if (from < m) {
                if (from > 0) try laneMatmul(.mxfp8, self.w8.view, &self.exponents.view, 1.0, self.n, self.k, self.npad, x.span(0, from), out.span(0, from));
                const rows = m - from;
                try a8Matmul(.mxfp8, a8, self.w8.view, self.exponents.view, self.n, self.k, self.npad, x.span(from, rows), out.span(from, rows));
                const kept = self.bf16_from orelse return;
                // the tiles from `column` on: the weight's [npad/64][k/64] tile order, a tile 64 * k bytes and
                // its exponents 128 * k / 64
                const tile = kept.column / 64;
                const w8: [*]u8 = @ptrCast(self.w8.view.ptr.?);
                const exponents: [*]u8 = @ptrCast(self.exponents.view.ptr.?);
                const tiles = self.npad / 64 - tile;
                const w: cuda.View = .init(w8 + tile * 64 * self.k, .u8, &.{tiles * 64 * self.k});
                const bs: cuda.View = .init(exponents + tile * (self.k / 64) * 128, .u8, &.{ tiles, self.k / 64, 64, 2 });
                const staged: cuda.View = .init(kept.staging.view.ptr.?, out.dtype, &.{ rows, self.n - kept.column });
                try laneMatmul(.mxfp8, w, &bs, 1.0, self.n - kept.column, self.k, tiles * 64, x.span(from, rows), staged);
                const into = out.span(from, rows);
                return cuda.launched(cuda.kernels.zs_columns_place(&staged, &into, @intCast(kept.column)));
            }
        }
        return laneMatmul(.mxfp8, self.w8.view, &self.exponents.view, 1.0, self.n, self.k, self.npad, x, out);
    }

    /// out (m, w) = act(x W^T + bias) for many rows, the bias bf16 [w] (`biased`): prompt.cu's chain, not the lane
    /// matmul's, so not a step's bits; for rows that have no step (the vision tower).
    pub fn forwardBiased(self: Mx8Linear, x: cuda.View, bias: cuda.View, out: cuda.View, act: Activation) cuda.Error!void {
        return biased(.mxfp8, self.w8.view, &self.exponents.view, 1.0, self.npad, bias, x, out, act);
    }

    /// Prompt rows take `forward`: its bits.
    pub fn prefill(self: Mx8Linear, x: cuda.View, out: cuda.View, _: u3) cuda.Error!void {
        return self.forward(x, out);
    }
};

/// One block-FP8 weight of a stack.
pub const Rows = struct {
    /// e4m3 [n, k] row-major
    codes: []const u8,
    /// [ceil(n / block.outputs), k / block.inputs]: each multiplies its block's codes
    block_scales: BlockScales,
    n: usize,
};

/// A block-scaled FP8 projection (ModelOpt FP8_PB_WO): e4m3 bytes in fragment order and an fp32 scale per
/// (output, 64 inputs), spread from the checkpoint's scale per block.
pub const Fp8BlockLinear = struct {
    w8: cuda.Buffer,
    /// fp32 [npad/64, k/64, 64]: a tile's column scales together
    column_scales: cuda.Buffer,
    /// fp32 [npad/64, k/64]: a tile's one scale a group, where every tile has one
    tile_scales: ?cuda.Buffer = null,
    n: usize,
    k: usize,
    npad: usize,
    /// Set by `quantizeActivations`: rows quantized to FP8 a token's 128 inputs and FP8 products
    a8: ?Activations = null,

    pub const Block = struct {
        outputs: usize = 128,
        inputs: usize = 128,

        /// The blocks of a checkpoint's linear of `k` inputs: 128 outputs by 128 inputs, or by 64 where the
        /// inputs are not a multiple of 128 (a hyper-connection's `up` has 320).
        pub fn of(k: usize) Block {
            return .{ .inputs = if (k % 128 == 0) 128 else 64 };
        }
    };

    /// `codes` e4m3 [n, k] row-major; `block_scales` fp32 [ceil(n / block.outputs), k / block.inputs], the
    /// checkpoint's weight_scale_inv, which multiplies its block's codes.
    pub fn init(gpa: Allocator, codes: []const u8, block_scales: []const f32, n: usize, k: usize, block: Block) !Fp8BlockLinear {
        return initStacked(gpa, &.{.{ .codes = codes, .block_scales = .of(block_scales), .n = n }}, k, block);
    }

    /// Several block-FP8 weights of one input stacked in order into one [n, k], so one matmul serves them all, as
    /// TensorFold's `from_rows` over joined rows. A part keeps its own blocks: its block rows count from its
    /// first output, not from the stack's.
    pub fn initStacked(gpa: Allocator, parts: []const Rows, k: usize, block: Block) !Fp8BlockLinear {
        if (k % 64 != 0 or block.inputs % 64 != 0 or k % block.inputs != 0) return error.UnsupportedShape;
        var n: usize = 0;
        for (parts) |part| {
            const block_rows = (part.n + block.outputs - 1) / block.outputs;
            if (part.codes.len != part.n * k or part.block_scales.len() != block_rows * (k / block.inputs)) return error.UnsupportedShape;
            n += part.n;
        }
        const npad = paddedOutputs(n);
        const Codes = struct {
            parts: []const Rows,
            k: usize,
            npad: usize,
            fn write(f: @This(), out: []u8) void {
                fragmentOrderStacked(f.parts, f.k, f.npad, out);
            }
        };
        const Scales = struct {
            parts: []const Rows,
            k: usize,
            npad: usize,
            block: Block,
            fn write(f: @This(), out: []f32) void {
                var at: usize = 0;
                for (0..f.npad / 64) |tile| for (0..f.k / 64) |group| for (0..64) |column| {
                    out[at] = if (locate(f.parts, tile * 64 + column)) |found|
                        f.parts[found.part].block_scales.at(found.row / f.block.outputs * (f.k / f.block.inputs) + group * 64 / f.block.inputs)
                    else
                        1.0;
                    at += 1;
                };
            }
        };
        var w8 = try uploaded(gpa, u8, .u8, &.{npad * k}, Codes{ .parts = parts, .k = k, .npad = npad });
        errdefer w8.deinit();
        var column_scales = try uploaded(gpa, f32, .f32, &.{ npad / 64, k / 64, 64 }, Scales{ .parts = parts, .k = k, .npad = npad, .block = block });
        errdefer column_scales.deinit();
        // made whatever `tile_rows` says: `quantizeActivations` reads them too
        const tile_scales = try tileScales(gpa, parts, k, npad, block);
        return .{ .w8 = w8, .column_scales = column_scales, .tile_scales = tile_scales, .n = n, .k = k, .npad = npad };
    }

    /// A tile's scale a group [npad/64, k/64], or null where some tile's outputs do not share one (parts of a
    /// stack meeting inside a tile). Padded outputs have zero codes and take their tile's scale.
    fn tileScales(gpa: Allocator, parts: []const Rows, k: usize, npad: usize, block: Block) !?cuda.Buffer {
        const groups = k / 64;
        const host = try gpa.alloc(f32, npad / 64 * groups);
        defer gpa.free(host);
        for (0..npad / 64) |tile| for (0..groups) |group| {
            var scale: ?f32 = null;
            for (0..64) |column| {
                const found = locate(parts, tile * 64 + column) orelse break;
                const value = parts[found.part].block_scales.at(found.row / block.outputs * (k / block.inputs) + group * 64 / block.inputs);
                if (scale) |seen| {
                    if (@as(u32, @bitCast(seen)) != @as(u32, @bitCast(value))) return null;
                } else scale = value;
            }
            host[tile * groups + group] = scale orelse 1.0;
        };
        var buffer: cuda.Buffer = try .init(.f32, &.{ npad / 64, groups });
        errdefer buffer.deinit();
        try buffer.upload(std.mem.sliceAsBytes(host));
        return buffer;
    }

    pub fn deinit(self: *Fp8BlockLinear) void {
        self.w8.deinit();
        self.column_scales.deinit();
        if (self.tile_scales) |*scales| scales.deinit();
        self.* = undefined;
    }

    /// From now on its rows are quantized as vLLM quantizes them (e4m3 under an fp32 scale a token's 128 inputs,
    /// `qmm_a8.cu`), every call of up to `rows.count` rows; false where the shape does not allow it (inputs not
    /// whole blocks of 128, or a tile whose outputs have no one scale), and the linear keeps bf16 rows.
    pub fn quantizeActivations(self: *Fp8BlockLinear, rows: cuda.Scratch.Rows) !bool {
        if (!quantized_activations or !a8Fits(self.k) or self.tile_scales == null) return false;
        self.a8 = try .of(rows, self.k);
        return true;
    }

    /// Per-tensor FP8: stored activation scale in every window, using the same shared scratch.
    pub fn quantizeStaticActivations(self: *Fp8BlockLinear, rows: cuda.Scratch.Rows, scale: f32) !void {
        if (!a8Fits(self.k) or self.tile_scales == null or !std.math.isFinite(scale) or scale <= 0) return error.UnsupportedShape;
        self.a8 = try .of(rows, self.k);
        self.a8.?.input_scale = scale;
    }

    /// out (m, n) = x (m, k) bf16 times the weight; prompt rows take this lane matmul too. With `a8`, x quantized to
    /// FP8 first.
    pub fn forward(self: Fp8BlockLinear, x: cuda.View, out: cuda.View) cuda.Error!void {
        if (self.a8) |a8| return a8Matmul(.fp8_grouped, a8, self.w8.view, self.tile_scales.?.view, self.n, self.k, self.npad, x, out);
        if (self.tile_scales) |scales| if (x.size[0] <= tile_rows) {
            const sk = splitK(self.n, self.k);
            const fuse = sk > 1 and (self.n + 63) / 64 >= fused_tiles;
            return cuda.launched(cuda.kernels.zs_nvfp4_qmmf_tile(&x, &self.w8.view, &scales.view, &out, @intCast(self.n), @intCast(sk), @intCast(self.npad), @intFromBool(fuse), @intFromBool(out.dtype == .f32)));
        };
        return laneMatmul(.fp8_grouped, self.w8.view, &self.column_scales.view, 1.0, self.n, self.k, self.npad, x, out);
    }

    pub fn prefill(self: Fp8BlockLinear, x: cuda.View, out: cuda.View, _: u3) cuda.Error!void {
        return self.forward(x, out);
    }
};
