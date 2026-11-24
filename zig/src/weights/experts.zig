//! Routed NVFP4 experts: pairs (row, slot) grouped by expert into a plan, then each pair's SwiGLU (or relu squared
//! of one up matrix, `initRelu2`) and down projection. A pair's bits never depend on the other pairs of the call.
//! Layouts are TensorFold's (cuda/experts.py, cuda/nvfp4/experts.py).
//!
//! `Activations.nvfp4` (spec 114 WS-03, zspark's own) packs the weights for the block-scaled FP4 mma (`packA4`)
//! and runs a call in two forms (spec 115), as the card's vLLM runs FP4-Local's experts: W4A4 on a prompt's rows
//! (`gateUpQuantizedAs`: each pair's row quantized to NVFP4 with the layer's gate/up `input_scale`,
//! `zig/kernels/nvfp4/experts_quant.cu`, the products on the FP4 mma, `experts_a4.cu`, `experts_tile_a4.cu`, down's
//! rows quantized in gate and up's epilogue at the card's rounding points), and A16 on a step's and a verify
//! window's (`gateUpA16`: bf16 rows times the same pack decoded, `experts_a4.cu` `nvfp4_expert_a16_kernel`). A model
//! chooses it by its checkpoint's map (GLM FP4-Local's `NVFP4` experts); `bf16` is TensorFold's, every other
//! caller's.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");
const loading = @import("loading.zig");
const nvfp4 = @import("nvfp4.zig");
const experts_pack = @import("experts_pack.zig");
pub const Staging = experts_pack.Staging;
pub const Swap = experts_pack.Swap;
pub const Matrix = experts_pack.Matrix;
const cols = experts_pack.cols;
const block_words = experts_pack.block_words;
const pack = experts_pack.pack;
const packA4 = experts_pack.packA4;

/// Pairs an item holds at most in a narrow call, and the pairs one tile of an expert kernel's pass holds.
pub const tile = 16;

/// How a call's pairs meet an expert's weights. It never changes a bit: a pair's sums are its own, whatever
/// shares its item or its pass.
pub const Pass = struct {
    /// Pairs an item holds at most (the plan's tile).
    item: i64,
    /// Tiles of 16 pairs that one reading of an expert's weights serves: 1 is TensorFold's kernel
    /// (`zig/kernels/nvfp4/experts.cu`), 2 and 4 zspark's (`experts_wide.cu`, `experts_tile.cu`).
    tiles: i64,

    /// A step, a verify window, a short chunk: TensorFold's form, which reads an expert's weights again for
    /// every 16 of its pairs. Up to 16 rows no expert has more.
    pub const narrow: Pass = .{ .item = tile, .tiles = 1 };
    /// A wide window, where an expert has 17 to 21 pairs on average and some have hundreds: 32 pairs a reading.
    /// The kernel holds more registers a thread, so fewer warps have loads in flight; it pays from about 256
    /// rows (spec 021: 7 to 8% of a 512-row window's routed experts, nothing at 128 rows, 3% slower at 16).
    pub const wide: Pass = .{ .item = 2 * tile, .tiles = 2 };
    /// Rows from which a call is wide, or null: every call is TensorFold's. The owner's bar is 2% of a window
    /// (spec 032), and the wide kernel clears it from 768 rows: measured in the model on two prompts, the
    /// machine held alone, it takes 13 to 21 ms off a window of 768 rows (2.1 to 3.3%), 32 to 40 of 1024 (3.9 to
    /// 4.9%) and 70 of 2048 (4.7%). Below that it is faster too, but by less: 0.4 to 1.2% at 128 rows, 1.1 to
    /// 1.6% at 256, 1.2 to 1.8% at 384, 1.5 to 1.9% at 512 (a cold prompt of 3500 tokens 1.8%), 1.5 to 2.3% at
    /// 576 to 704. A session's prompt windows are 2048 rows or more (`session.promptRows`, spec
    /// 031; one rank's lone stream 4096, spec 148), so a prompt's full windows are wide and its last window by its rows; 256 is where to set it if 1
    /// to 2% is wanted. The tests run the kernel whatever stands here.
    pub const wide_rows: ?usize = 768;

    /// A prompt's window of `tile_rows` or more: 64 pairs a reading, the rows staged once for a block's 8 warps and
    /// not by each warp into its registers (spec 105 WS-03). Launched alone at 2048 rows it takes 23% off the
    /// wide kernel's plan, gate and up and down with every expert routed (7.73 ms against 10.04) and 17% with a
    /// prose's skew (7.58 against 9.22); 7% at 1024 rows of skew, and within 1% either way with every expert
    /// routed at 768 to 1536 rows, where an expert's 15 to 30 pairs fit the wide kernel's 32.
    pub const tile4: Pass = .{ .item = 4 * tile, .tiles = 4 };
    /// Rows from which a call is `tile4`, or null: never.
    pub var tile_rows: ?usize = 1024;

    pub fn of(rows: usize) Pass {
        if (tile_rows) |from| {
            if (rows >= from) return tile4;
        }
        if (wide_rows) |from| {
            if (rows >= from) return wide;
        }
        return narrow;
    }
};
/// A narrow call's units (an item's 32-column blocks) over the grid's warps in whole rounds
/// (`zig/kernels/nvfp4/experts.cu`, EVEN) and not TensorFold's stride, which leaves the units past the grid's
/// warps to a few of them: the same bits, a launch whose units pass the warps by a little up to 12% shorter
/// (spec 073). `ZSPARK_EVEN_EXPERTS=0` (`main` reads it) keeps the stride.
pub var even_units: bool = true;

/// What the products take of the activations.
pub const Activations = enum {
    /// bf16 rows times the weights decoded to bf16 (TensorFold's; W4A16).
    bf16,
    /// the weights packed for the FP4 mma: rows quantized to NVFP4, W4A4 (`gateUpQuantizedAs`), or bf16 rows on the
    /// same pack, A16 (`gateUpA16`), by the caller's form.
    nvfp4,

    /// The pass of a W4A4 call of `rows` rows: `experts_a4.cu`'s pairs 16 at a time, `experts_tile_a4.cu`'s 64
    /// under one reading of an expert's weights from `a4_tile_rows`, or its 128 from `a4_tile8_rows`. None changes
    /// a bit.
    pub fn passOf(rows: usize) Pass {
        if (a4_tile8_rows) |from| {
            if (rows >= from) return a4_tile8;
        }
        if (a4_tile_rows) |from| {
            if (rows >= from) return .tile4;
        }
        return .narrow;
    }
};

/// W4A4's items of 128 pairs, each under one reading of its expert's weights (`experts_tile_a4.cu`, RH 2).
pub const a4_tile8: Pass = .{ .item = 8 * tile, .tiles = 8 };
/// Rows from which a W4A4 call is `Pass.tile4`, or null: never.
pub var a4_tile_rows: ?usize = 256;
/// Rows from which a W4A4 call is `a4_tile8`, or null: never.
pub var a4_tile8_rows: ?usize = 1024;

/// The products' kernel for a W4A4 pass: 1 (16 pairs a reading), 4 (up to 64) or 8 (up to 128).
fn a4Tiles(pass: Pass) i64 {
    return switch (pass.tiles) {
        1 => 1,
        8 => 8,
        else => 4,
    };
}

/// A W4A4 call's rows, the plan's pairs quantized to NVFP4, row i the plan's member i: gate and up's
/// (`experts_quant.cu`), and down's, which gate and up's epilogue writes while the first are still read.
pub const Quantized = struct {
    /// u8 [pairs, dims / 2]: two e2m1 codes a byte, the lower input in the low nibble
    codes: cuda.Buffer,
    /// u8 [pairs, dims / 16]: e4m3
    scales: cuda.Buffer,
    /// u8 [pairs, width / 2] and [pairs, width / 16]: down's, as `codes` and `scales`
    down_codes: cuda.Buffer,
    down_scales: cuda.Buffer,

    /// For calls of up to `pairs` pairs (a plan's rows times slots) of experts `dims` by `width`.
    pub fn init(pairs: usize, dims: usize, width: usize) cuda.Error!Quantized {
        var codes: cuda.Buffer = try .init(.u8, &.{ pairs, dims / 2 });
        errdefer codes.deinit();
        var scales: cuda.Buffer = try .init(.u8, &.{ pairs, dims / 16 });
        errdefer scales.deinit();
        const padded_width = std.mem.alignForward(usize, width, 64);
        var down_codes: cuda.Buffer = try .init(.u8, &.{ pairs, padded_width / 2 });
        errdefer down_codes.deinit();
        var down_scales: cuda.Buffer = try .init(.u8, &.{ pairs, padded_width / 16 });
        errdefer down_scales.deinit();
        if (padded_width != width) {
            try down_codes.zero();
            try down_scales.zero();
        }
        return .{ .codes = codes, .scales = scales, .down_codes = down_codes, .down_scales = down_scales };
    }

    pub fn deinit(self: *Quantized) void {
        self.codes.deinit();
        self.scales.deinit();
        self.down_codes.deinit();
        self.down_scales.deinit();
        self.* = undefined;
    }
};

extern fn zs_experts_plan(picks: *const cuda.View, pairs: i64, experts: i64, tile: i64, members: *const cuda.View, items: *const cuda.View, counts: *const cuda.View, rank: *const cuda.View, hist: *const cuda.View) c_int;
extern fn zs_nvfp4_experts(epi: i64, tiles: i64, even: i64, x: *const cuda.View, x_stride: i64, slots: i64, w: *const cuda.View, scale: *const cuda.View, kg: i64, nb: i64, items: *const cuda.View, counts: *const cuda.View, members: *const cuda.View, out: *const cuda.View, n: i64, limit: f64, skip: i64, max_units: i64) c_int;
extern fn zs_nvfp4_csf_decode(planes: *const cuda.View, rows: i64, cols: i64, wrow: i64, wrows: i64, wcol: i64, wcols: i64, out: *const cuda.View, failed: *const cuda.View, mark: i64) c_int;
extern fn zs_nvfp4_experts_pack_a16(sources: *const cuda.View, n: i64, k: i64, split: i64, code_stride: i64, high_stride: i64, scale_stride: i64, out: *const cuda.View, per: i64, m: i64) c_int;
extern fn zs_nvfp4_experts_pack_a4(sources: *const cuda.View, n: i64, k: i64, split: i64, code_stride: i64, high_stride: i64, scale_stride: i64, out: *const cuda.View, per: i64, m: i64) c_int;
extern fn zs_nvfp4_rows_gather(sources: *const cuda.View, rows: i64, width: i64, stride: i64, out: *const cuda.View) c_int;
extern fn zs_nvfp4_experts_quant(x: *const cuda.View, x_stride: i64, slots: i64, gs: *const cuda.View, items: *const cuda.View, counts: *const cuda.View, members: *const cuda.View, codes: *const cuda.View, scales: *const cuda.View, k: i64, skip: i64, max_items: i64) c_int;
extern fn zs_nvfp4_experts_a16(epi: i64, even: i64, x: *const cuda.View, x_stride: i64, slots: i64, w: *const cuda.View, scale: *const cuda.View, kg: i64, nb: i64, items: *const cuda.View, counts: *const cuda.View, members: *const cuda.View, out: *const cuda.View, n: i64, limit: f64, skip: i64, max_units: i64) c_int;
extern fn zs_nvfp4_experts_a4(epi: i64, tiles: i64, even: i64, codes: *const cuda.View, scales: *const cuda.View, w: *const cuda.View, alpha: *const cuda.View, kg: i64, nb: i64, items: *const cuda.View, counts: *const cuda.View, members: *const cuda.View, out: ?*const cuda.View, n: i64, limit: f64, skip: i64, max_units: i64, down_codes: ?*const cuda.View, down_scales: ?*const cuda.View, down_gs: ?*const cuda.View) c_int;

/// Pairs the one-block plan takes; wider plans rank in blocks of this many.
const small_plan = 1024;

/// Items a plan of `pairs` can hold: one per used expert, plus one per `tile` pairs past its first.
pub fn maxItems(pairs: usize, experts: usize) usize {
    return @min(pairs, experts) + pairs / tile;
}

/// Scratch grouping pairs by expert: pair p = row * slots + slot.
pub const Plan = struct {
    rows: usize,
    slots: usize,
    experts: usize,
    /// int32 [pairs]: pairs in expert order, pair order within an expert
    members: cuda.Buffer,
    /// int32 [max items, 3]: expert, first member, count
    items: cuda.Buffer,
    /// int32 [4]: items, distinct experts, then the units `experts_tile_a4.cu` has handed out and the blocks it has
    /// ended, zero between its launches (it leaves them so)
    counts: cuda.Buffer,
    rank: cuda.Buffer,
    hist: cuda.Buffer,

    pub fn init(rows: usize, slots: usize, experts: usize) cuda.Error!Plan {
        const pairs = rows * slots;
        const wide = pairs > small_plan;
        var members: cuda.Buffer = try .init(.i32, &.{pairs});
        errdefer members.deinit();
        var items: cuda.Buffer = try .init(.i32, &.{ maxItems(pairs, experts), 3 });
        errdefer items.deinit();
        var counts: cuda.Buffer = try .init(.i32, &.{4});
        errdefer counts.deinit();
        try counts.zero();
        var rank: cuda.Buffer = try .init(.i32, &.{if (wide) pairs else 1});
        errdefer rank.deinit();
        const hist: cuda.Buffer = try .init(.i32, &.{if (wide) (pairs + 1023) / 1024 * experts else 1});
        return .{ .rows = rows, .slots = slots, .experts = experts, .members = members, .items = items, .counts = counts, .rank = rank, .hist = hist };
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(rows: usize, slots: usize, experts: usize) usize {
        const of = cuda.Buffer.bytesOf;
        const pairs = rows * slots;
        const wide = pairs > small_plan;
        return of(.i32, &.{pairs}) + of(.i32, &.{ maxItems(pairs, experts), 3 }) + of(.i32, &.{4}) + of(.i32, &.{if (wide) pairs else 1}) +
            of(.i32, &.{if (wide) (pairs + 1023) / 1024 * experts else 1});
    }

    pub fn deinit(self: *Plan) void {
        self.members.deinit();
        self.items.deinit();
        self.counts.deinit();
        self.rank.deinit();
        self.hist.deinit();
        self.* = undefined;
    }

    /// `picks` int32 [rows, slots]: each pair's expert, for `rows` up to the plan's.
    pub fn route(self: Plan, picks: cuda.View, rows: usize) cuda.Error!void {
        return self.routeAs(picks, rows, .of(rows));
    }

    /// `route` in items of `pass`'s size, whatever the rows.
    pub fn routeAs(self: Plan, picks: cuda.View, rows: usize, pass: Pass) cuda.Error!void {
        std.debug.assert(rows <= self.rows);
        return cuda.launched(zs_experts_plan(&picks, @intCast(rows * self.slots), @intCast(self.experts), pass.item, &self.members.view, &self.items.view, &self.counts.view, &self.rank.view, &self.hist.view));
    }
};

/// One layer's routed experts: gate and up (SwiGLU), and down; or, made by `initRelu2`, up alone with relu
/// squared (Nemotron's, spec 151), and down.
pub const Experts = struct {
    /// bf16: int32 [experts, width/32, dims/32, 2, 144], gate then up (`pack`), [experts, width/32, dims/32, 1, 144]
    /// where not `gated`; nvfp4: int32 [experts, width/32, dims/64, 2, 288] (`packA4`)
    up: cuda.Buffer,
    /// bf16: int32 [experts, dims/32, width/32, 1, 144]; nvfp4: int32 [experts, dims/32, width/64, 1, 288]
    down: cuda.Buffer,
    /// fp32 [experts, 2] ([experts, 1] where not `gated`): weight_scale_2; nvfp4: alpha, weight_scale_2 / gs
    /// (b12x's runtime alphas)
    up_scale: cuda.Buffer,
    /// fp32 [experts, 1], as `up_scale`
    down_scale: cuda.Buffer,
    /// nvfp4: fp32 [experts] each, gs = 1 / input_scale: gate and up's the layer's, 1 / the largest gate or up
    /// input_scale of its experts (the card's A4 prompt path, `fused_moe/b12x.py` `a1_gscale = a1.amin()`), down's
    /// each expert's
    up_gs: ?cuda.Buffer = null,
    down_gs: ?cuda.Buffer = null,
    /// nvfp4: fp32 [experts, 2] and [experts, 1], weight_scale_2: the A16 form's factors
    up_ws2: ?cuda.Buffer = null,
    down_ws2: ?cuda.Buffer = null,
    activations: Activations = .bf16,
    count: usize,
    /// The intermediate width.
    width: usize,
    dims: usize,
    /// SwiGLU clip; 0: none.
    limit: f32 = 0,
    /// Gate and up with SwiGLU, else one up matrix with relu(bf16(acc))^2 (`common/experts.cuh`'s epilogue 1,
    /// TensorFold's); both A16 and NVFP4 activations support it.
    gated: bool = true,

    /// `staging` holds the packed blocks until they are uploaded; a loader passes the same one for every layer.
    pub fn init(gpa: Allocator, staging: *Staging, count: usize, width: usize, dims: usize, gate: Matrix, up_matrix: Matrix, down_matrix: Matrix) !Experts {
        return initWith(gpa, staging, count, width, dims, gate, up_matrix, down_matrix, .bf16);
    }

    /// `init` with the activations' form: `.nvfp4` reads each matrix's `input_scales` and refuses one that is not
    /// finite and positive (vLLM's `scalar` check), or inputs not in steps of 64.
    pub fn initWith(gpa: Allocator, staging: *Staging, count: usize, width: usize, dims: usize, gate: Matrix, up_matrix: Matrix, down_matrix: Matrix, activations: Activations) !Experts {
        const a4 = activations == .nvfp4;
        try checkShape(count, width, dims, .{ gate.input_scales, up_matrix.input_scales, down_matrix.input_scales }, activations);
        const up_blocks = try Staging.take(&staging.up, gpa, count * (width / cols) * (dims / 32) * 2 * block_words);
        const down_blocks = try Staging.take(&staging.down, gpa, count * (dims / cols) * (width / 32) * block_words);
        const packing = loading.now();
        if (a4) {
            packA4(gate, count, width, dims, up_blocks, 2, 0);
            packA4(up_matrix, count, width, dims, up_blocks, 2, 1);
            packA4(down_matrix, count, dims, width, down_blocks, 1, 0);
        } else {
            pack(gate, count, width, dims, up_blocks, 2, 0);
            pack(up_matrix, count, width, dims, up_blocks, 2, 1);
            pack(down_matrix, count, dims, width, down_blocks, 1, 0);
        }
        loading.since(&loading.host.pack_ns, packing);
        const words: usize = if (a4) 2 * block_words else block_words;
        const steps: usize = if (a4) 64 else 32;
        var up: cuda.Buffer = try .init(.i32, &.{ count, width / cols, dims / steps, 2, words });
        errdefer up.deinit();
        try up.upload(std.mem.sliceAsBytes(up_blocks));
        var down: cuda.Buffer = try .init(.i32, &.{ count, dims / cols, width / steps, 1, words });
        errdefer down.deinit();
        try down.upload(std.mem.sliceAsBytes(down_blocks));
        return finish(gpa, up, down, count, width, dims, .{ gate.scales, up_matrix.scales, down_matrix.scales }, .{ gate.input_scales, up_matrix.input_scales, down_matrix.input_scales }, activations, true);
    }

    /// Experts of one up matrix [width, dims] and relu squared, then down [dims, width], each expert's
    /// `weight_scale_2` its factor: TensorFold's non-gated `relu2` experts (NemotronH), W4A16.
    /// `initRelu2Quantized` takes the same stored weights and NVFP4 activations on tensor cores.
    pub fn initRelu2(gpa: Allocator, staging: *Staging, count: usize, width: usize, dims: usize, up_matrix: Matrix, down_matrix: Matrix) !Experts {
        try checkShape(count, width, dims, .{ &.{}, &.{}, &.{} }, .bf16);
        if (up_matrix.scales.len != count or down_matrix.scales.len != count) return error.UnsupportedShape;
        const up_blocks = try Staging.take(&staging.up, gpa, count * (width / cols) * (dims / 32) * block_words);
        const down_blocks = try Staging.take(&staging.down, gpa, count * (dims / cols) * (width / 32) * block_words);
        const packing = loading.now();
        pack(up_matrix, count, width, dims, up_blocks, 1, 0);
        pack(down_matrix, count, dims, width, down_blocks, 1, 0);
        loading.since(&loading.host.pack_ns, packing);
        var up: cuda.Buffer = try .init(.i32, &.{ count, width / cols, dims / 32, 1, block_words });
        errdefer up.deinit();
        try up.upload(std.mem.sliceAsBytes(up_blocks));
        var down: cuda.Buffer = try .init(.i32, &.{ count, dims / cols, width / 32, 1, block_words });
        errdefer down.deinit();
        try down.upload(std.mem.sliceAsBytes(down_blocks));
        var up_scale: cuda.Buffer = try .init(.f32, &.{ count, 1 });
        errdefer up_scale.deinit();
        try up_scale.upload(std.mem.sliceAsBytes(up_matrix.scales));
        var down_scale: cuda.Buffer = try .init(.f32, &.{ count, 1 });
        errdefer down_scale.deinit();
        try down_scale.upload(std.mem.sliceAsBytes(down_matrix.scales));
        return .{ .up = up, .down = down, .up_scale = up_scale, .down_scale = down_scale, .count = count, .width = width, .dims = dims, .gated = false };
    }

    /// NVFP4 relu2 in every window. Input scales are already grouped by the caller: routed experts
    /// share their layer scale, while folded shared experts retain their own linear's scale.
    pub fn initRelu2Quantized(gpa: Allocator, staging: *Staging, count: usize, width: usize, dims: usize, up_matrix: Matrix, down_matrix: Matrix) !Experts {
        if (width % cols != 0 or dims % 64 != 0) return error.UnsupportedShape;
        for ([_][]const f32{ up_matrix.input_scales, down_matrix.input_scales }) |inputs| {
            if (inputs.len != count) return error.UnsupportedShape;
            for (inputs) |one| if (!std.math.isFinite(one) or !(one > 0)) return error.UnsupportedShape;
        }
        if (up_matrix.scales.len != count or down_matrix.scales.len != count) return error.UnsupportedShape;
        const down_steps = (width + 63) / 64;
        const up_blocks = try Staging.take(&staging.up, gpa, count * (width / cols) * (dims / 64) * 2 * block_words);
        const down_blocks = try Staging.take(&staging.down, gpa, count * (dims / cols) * down_steps * 2 * block_words);
        const packing = loading.now();
        packA4(up_matrix, count, width, dims, up_blocks, 1, 0);
        packA4(down_matrix, count, dims, width, down_blocks, 1, 0);
        loading.since(&loading.host.pack_ns, packing);
        const held_blocks = blk: {
            var up: cuda.Buffer = try .init(.i32, &.{ count, width / cols, dims / 64, 1, 2 * block_words });
            errdefer up.deinit();
            try up.upload(std.mem.sliceAsBytes(up_blocks));
            var down: cuda.Buffer = try .init(.i32, &.{ count, dims / cols, down_steps, 1, 2 * block_words });
            errdefer down.deinit();
            try down.upload(std.mem.sliceAsBytes(down_blocks));
            break :blk .{ up, down };
        };
        return finish(gpa, held_blocks[0], held_blocks[1], count, width, dims, .{ up_matrix.scales, up_matrix.scales, down_matrix.scales }, .{ up_matrix.input_scales, up_matrix.input_scales, down_matrix.input_scales }, .nvfp4, false);
    }

    fn checkShape(count: usize, width: usize, dims: usize, inputs: [3][]const f32, activations: Activations) error{UnsupportedShape}!void {
        if (width % cols != 0 or dims % cols != 0) return error.UnsupportedShape;
        if (activations != .nvfp4) return;
        if (width % 64 != 0 or dims % 64 != 0) return error.UnsupportedShape;
        for (inputs) |matrix| {
            if (matrix.len != count) return error.UnsupportedShape;
            for (matrix) |one| if (!std.math.isFinite(one) or !(one > 0)) return error.UnsupportedShape;
        }
    }

    /// One projection of every expert where the device reads it: host memory, through the host's page tables
    /// (the GB10's `pageableMemoryAccess`), held unchanged until `Staging.settle` or the next layer's
    /// `initPlaced` returns. An expert's [n, k] of the stored matrix: its rank's rows of gate and up, its
    /// columns of down, which `code_stride` steps over; under `Staging.swap`, down's of this rank's half of the
    /// outputs alone.
    pub const Placed = struct {
        /// each expert's first code byte, `code_stride` bytes a row: two E2M1 codes a byte as `Matrix.codes`
        codes: []const u64,
        code_stride: usize,
        scales: Scales,
        /// as `Matrix.scales` and `Matrix.input_scales`
        factors: []const f32,
        input_scales: []const f32,

        pub const Scales = union(enum) {
            /// e4m3 as stored: each expert's first byte of the [n, k/16] it holds, `stride` bytes a row
            plain: struct { at: []const u64, stride: usize },
            /// csf_32x2_v1 planes of `rows` by `cols` (whole tiles), each expert's bases, codes, page_bases, offsets,
            /// exceptions and its records, the window [row, rows) x [col, cols) decoded on the device
            csf: struct { planes: []const [6]u64, rows: usize, cols: usize, window: [4]usize },
        };
    };

    /// `initWith`'s experts, packed by the device (`zig/kernels/nvfp4/experts_pack.cu`) from where `Placed` says
    /// the bytes are, into the bytes `packA4` (`.nvfp4`) or `pack` (`.bf16`) writes: the host's packing and the
    /// copy of its blocks are off the load's path. A csf plane that does not decode is found by the next `settle`.
    pub fn initPlaced(gpa: Allocator, staging: *Staging, layer: usize, count: usize, width: usize, dims: usize, gate: Placed, up_matrix: Placed, down_matrix: Placed, activations: Activations) !Experts {
        const matrices = [3]Placed{ gate, up_matrix, down_matrix };
        const a4 = activations == .nvfp4;
        try checkShape(count, width, dims, .{ gate.input_scales, up_matrix.input_scales, down_matrix.input_scales }, activations);
        // the layer before is done with the tables, the scales and its sources
        try staging.settle();
        var widest: usize = 1;
        for (matrices) |matrix| {
            if (matrix.codes.len != count or matrix.factors.len != count) return error.UnsupportedShape;
            switch (matrix.scales) {
                .plain => |plain| if (plain.at.len != count) return error.UnsupportedShape,
                .csf => |csf| {
                    if (csf.planes.len != count) return error.UnsupportedShape;
                    widest = @max(widest, count * csf.window[1] * csf.window[3]);
                },
            }
        }
        // a matrix's table: its csf planes [count, 6], its sources [count, 3], what `Swap` gathers [count]
        const fields = 10;
        if (staging.device == null) {
            var tables: cuda.Buffer = try .init(.i64, &.{3 * count * fields});
            errdefer tables.deinit();
            var scales: cuda.Buffer = try .init(.u8, &.{widest});
            errdefer scales.deinit();
            var failed: cuda.Buffer = try .init(.u32, &.{1});
            errdefer failed.deinit();
            try failed.zero();
            staging.device = .{ .tables = tables, .scales = scales, .failed = failed };
        }
        const device = &staging.device.?;
        device.layer = layer;
        if (device.tables.bytes() < 3 * count * fields * 8) {
            device.tables.deinit();
            device.tables = try .init(.i64, &.{3 * count * fields});
        }
        if (device.scales.bytes() < widest) {
            device.scales.deinit();
            device.scales = try .init(.u8, &.{widest});
        }
        // down's half of the outputs each rank reads, and the inputs of it that it trades
        const half = dims / 2;
        const traded = count * half * (width / 2);
        if (staging.swap != null) {
            if (dims % 64 != 0) return error.UnsupportedShape;
            for ([_]*?cuda.Buffer{ &device.send, &device.recv }) |one| {
                if (one.* != null and one.*.?.bytes() < traded) {
                    one.*.?.deinit();
                    one.* = null;
                }
                if (one.* == null) one.* = try .init(.u8, &.{traded});
            }
        }
        const host = try gpa.alloc(u64, 3 * count * fields);
        defer gpa.free(host);
        @memset(host, 0);
        const scratch = @intFromPtr(device.scales.view.ptr.?);
        var strides: [3][3]usize = undefined;
        for (matrices, 0..) |matrix, m| {
            const planes = host[m * count * fields ..][0 .. count * 6];
            const sources = host[m * count * fields + count * 6 ..][0 .. count * 3];
            const gathered = host[m * count * fields + count * 9 ..][0..count];
            const n = if (m < 2) width else dims;
            // the outputs below `split` from the first address, `code_stride` apart; the rest from the second
            var split = n;
            var low_stride = matrix.code_stride;
            var high_stride = matrix.code_stride;
            const swap = if (m == 2) staging.swap else null;
            if (swap) |one| {
                split = half;
                if (one.rank == 0) high_stride = width / 2 else low_stride = width / 2;
            }
            strides[m] = .{ split, low_stride, high_stride };
            for (0..count) |e| {
                const mine = matrix.codes[e];
                if (swap) |one| {
                    // this rank's outputs and its inputs, where it read them; the peer's inputs of them beside
                    const peer = 1 - one.rank;
                    gathered[e] = mine - one.rank * (width / 2) + peer * (width / 2);
                    const theirs = @intFromPtr(device.recv.?.view.ptr.?) + e * half * (width / 2);
                    sources[3 * e] = if (one.rank == 0) mine else theirs;
                    sources[3 * e + 1] = if (one.rank == 0) theirs else mine;
                } else {
                    sources[3 * e] = mine;
                    sources[3 * e + 1] = mine;
                }
                sources[3 * e + 2] = switch (matrix.scales) {
                    .plain => |plain| plain.at[e],
                    .csf => |csf| scratch + e * csf.window[1] * csf.window[3],
                };
                if (matrix.scales == .csf) @memcpy(planes[6 * e ..][0..6], &matrix.scales.csf.planes[e]);
            }
        }
        try device.tables.uploadAt(0, std.mem.sliceAsBytes(host));

        const words: usize = if (a4) 2 * block_words else block_words;
        const steps: usize = if (a4) 64 else 32;
        var up: cuda.Buffer = try .init(.i32, &.{ count, width / cols, dims / steps, 2, words });
        errdefer up.deinit();
        var down: cuda.Buffer = try .init(.i32, &.{ count, dims / cols, width / steps, 1, words });
        errdefer down.deinit();
        const table: [*]u8 = @ptrCast(device.tables.view.ptr.?);
        for (matrices, 0..) |matrix, m| {
            const at = table + m * count * fields * 8;
            const planes: cuda.View = .init(@ptrCast(at), .i64, &.{ count, 6 });
            const sources: cuda.View = .init(@ptrCast(at + count * 48), .i64, &.{ count, 3 });
            const n = if (m < 2) width else dims;
            const k = if (m < 2) dims else width;
            if (m == 2) if (staging.swap) |swap| {
                const gathered: cuda.View = .init(@ptrCast(at + count * 72), .i64, &.{count});
                const send: cuda.View = .init(device.send.?.view.ptr.?, .u8, &.{ count, half, width / 2 });
                const recv: cuda.View = .init(device.recv.?.view.ptr.?, .u8, &.{ count, half, width / 2 });
                try cuda.launched(zs_nvfp4_rows_gather(&gathered, @intCast(half), @intCast(width / 2), @intCast(matrix.code_stride), &send));
                try swap.trade(swap.context, send, recv);
            };
            var scale_stride: usize = undefined;
            switch (matrix.scales) {
                .plain => |plain| scale_stride = plain.stride,
                .csf => |csf| {
                    const w = csf.window;
                    if (w[1] != n or w[3] != k / 16) return error.UnsupportedShape;
                    const out: cuda.View = .init(device.scales.view.ptr.?, .u8, &.{ count, w[1], w[3] });
                    try cuda.launched(zs_nvfp4_csf_decode(&planes, @intCast(csf.rows), @intCast(csf.cols), @intCast(w[0]), @intCast(w[1]), @intCast(w[2]), @intCast(w[3]), &out, &device.failed.view, @intCast(m + 1)));
                    scale_stride = w[3];
                },
            }
            const target = if (m < 2) up else down;
            const stride = strides[m];
            const place = if (a4) &zs_nvfp4_experts_pack_a4 else &zs_nvfp4_experts_pack_a16;
            try cuda.launched(place(&sources, @intCast(n), @intCast(k), @intCast(stride[0]), @intCast(stride[1]), @intCast(stride[2]), @intCast(scale_stride), &target.view, if (m < 2) 2 else 1, if (m < 2) @intCast(m) else 0));
        }
        return finish(gpa, up, down, count, width, dims, .{ gate.factors, up_matrix.factors, down_matrix.factors }, .{ gate.input_scales, up_matrix.input_scales, down_matrix.input_scales }, activations, true);
    }

    /// The experts of the blocks `up` and `down` (which it takes, and frees where it fails) with their factors.
    fn finish(gpa: Allocator, up_blocks: cuda.Buffer, down_blocks: cuda.Buffer, count: usize, width: usize, dims: usize, factors: [3][]const f32, inputs: [3][]const f32, activations: Activations, gated: bool) !Experts {
        var up = up_blocks;
        errdefer up.deinit();
        var down = down_blocks;
        errdefer down.deinit();
        const a4 = activations == .nvfp4;
        const gate: struct { scales: []const f32, input_scales: []const f32 } = .{ .scales = factors[0], .input_scales = inputs[0] };
        const up_matrix: @TypeOf(gate) = .{ .scales = factors[1], .input_scales = inputs[1] };
        const down_matrix: @TypeOf(gate) = .{ .scales = factors[2], .input_scales = inputs[2] };
        const matrices: usize = if (gated) 2 else 1;
        const up_scales = try gpa.alloc(f32, count * matrices);
        defer gpa.free(up_scales);
        const down_scales = try gpa.alloc(f32, count);
        defer gpa.free(down_scales);
        const up_gs = try gpa.alloc(f32, count);
        defer gpa.free(up_gs);
        const down_gs = try gpa.alloc(f32, count);
        defer gpa.free(down_gs);
        // the card's: a1_gscale = 1 / the layer's largest gate or up input_scale (`a1.amin()` of the reciprocals),
        // a2_gscale each expert's, alpha = weight_scale_2 / gscale, each in fp32
        var widest: f32 = 0;
        if (a4 and gated) for (gate.input_scales, up_matrix.input_scales) |g, u| {
            widest = @max(widest, @max(g, u));
        };
        for (0..count) |e| {
            if (a4) {
                up_gs[e] = 1.0 / (if (gated) widest else up_matrix.input_scales[e]);
                down_gs[e] = 1.0 / down_matrix.input_scales[e];
                up_scales[matrices * e] = gate.scales[e] / up_gs[e];
                if (gated) up_scales[matrices * e + 1] = up_matrix.scales[e] / up_gs[e];
                down_scales[e] = down_matrix.scales[e] / down_gs[e];
            } else {
                up_scales[matrices * e] = gate.scales[e];
                if (gated) up_scales[matrices * e + 1] = up_matrix.scales[e];
                down_scales[e] = down_matrix.scales[e];
            }
        }

        var up_scale: cuda.Buffer = try .init(.f32, &.{ count, matrices });
        errdefer up_scale.deinit();
        try up_scale.upload(std.mem.sliceAsBytes(up_scales));
        var down_scale: cuda.Buffer = try .init(.f32, &.{ count, 1 });
        errdefer down_scale.deinit();
        try down_scale.upload(std.mem.sliceAsBytes(down_scales));
        var self: Experts = .{ .up = up, .down = down, .up_scale = up_scale, .down_scale = down_scale, .activations = activations, .count = count, .width = width, .dims = dims, .gated = gated };
        if (a4) {
            self.up_gs = try .init(.f32, &.{count});
            errdefer self.up_gs.?.deinit();
            try self.up_gs.?.upload(std.mem.sliceAsBytes(up_gs));
            self.down_gs = try .init(.f32, &.{count});
            errdefer self.down_gs.?.deinit();
            try self.down_gs.?.upload(std.mem.sliceAsBytes(down_gs));
            if (gated) {
                for (0..count) |e| {
                    up_scales[matrices * e] = gate.scales[e];
                    up_scales[matrices * e + 1] = up_matrix.scales[e];
                    down_scales[e] = down_matrix.scales[e];
                }
                self.up_ws2 = try .init(.f32, &.{ count, matrices });
                errdefer self.up_ws2.?.deinit();
                try self.up_ws2.?.upload(std.mem.sliceAsBytes(up_scales));
                self.down_ws2 = try .init(.f32, &.{ count, 1 });
                errdefer self.down_ws2.?.deinit();
                try self.down_ws2.?.upload(std.mem.sliceAsBytes(down_scales));
            }
        }
        return self;
    }

    pub fn deinit(self: *Experts) void {
        self.up.deinit();
        self.down.deinit();
        self.up_scale.deinit();
        self.down_scale.deinit();
        if (self.up_gs) |*one| one.deinit();
        if (self.down_gs) |*one| one.deinit();
        if (self.up_ws2) |*one| one.deinit();
        if (self.down_ws2) |*one| one.deinit();
        self.* = undefined;
    }

    /// x bf16 [rows, dims]'s routed pairs by `picks` [rows, slots] -> y [rows * slots, dims], fp32 or bf16, `act`
    /// bf16 [rows * slots, width] their SwiGLU between; pairs of expert `skip` are left untouched. A `.bf16` pack
    /// runs on its kernels by the rows' pass; a W4A4 pack (`.nvfp4`) takes the rows before `from` A16 and the
    /// rows from it W4A4 through `quantized`, each row's bits its form's alone (`decode/forms.zig`: the cards'
    /// forms by window kind, GLM's spec 115 and Flash Next's spec 143 WS-06; moved here from GLM's `mlp.zig`).
    pub fn run(self: Experts, plan: Plan, picks: cuda.View, x: cuda.View, act: cuda.View, quantized: ?Quantized, y: cuda.View, rows: usize, from: usize, skip: ?usize) cuda.Error!void {
        const slots = plan.slots;
        switch (self.activations) {
            .bf16 => {
                try plan.route(picks, rows);
                try self.gateUp(x, plan, act, rows, skip);
                try self.downProject(act, plan, y, rows, skip);
            },
            .nvfp4 => {
                const plain = @min(from, rows);
                if (plain > 0) {
                    const a16 = act.span(0, plain * slots);
                    try plan.routeAs(picks.span(0, plain), plain, .narrow);
                    try self.gateUpA16(x.span(0, plain), plan, a16, plain, skip);
                    try self.downProjectA16(a16, plan, y.span(0, plain * slots), plain, skip);
                }
                const four = rows - plain;
                if (four > 0) {
                    const pass = Activations.passOf(four);
                    const qy = y.span(plain * slots, four * slots);
                    try plan.routeAs(picks.span(plain, four), four, pass);
                    try self.gateUpQuantizedAs(x.span(plain, four), plan, quantized.?, null, four, skip, pass);
                    try self.downProjectQuantizedAs(plan, quantized.?, qy, four, skip, pass);
                }
            },
        }
    }

    /// W4A4: x [rows, dims] bf16 quantized into `rows_q` by pair, then each routed pair's SwiGLU, quantized into
    /// `rows_q`'s down rows; pairs of expert `skip` are left untouched. `act`, a check's: null, or bf16
    /// [rows * slots, width] that takes the SwiGLU's values as they were quantized. `pass`'s tiles a reading,
    /// whatever the rows and the plan's items (up to 64 pairs).
    pub fn gateUpQuantizedAs(self: Experts, x: cuda.View, plan: Plan, rows_q: Quantized, act: ?cuda.View, rows: usize, skip: ?usize, pass: Pass) cuda.Error!void {
        std.debug.assert(self.activations == .nvfp4);
        const blocks = self.width / cols;
        const x_stride: i64 = if (x.row_stride != 0) x.row_stride else x.size[1];
        const items = maxItems(rows * plan.slots, plan.experts);
        try cuda.launched(zs_nvfp4_experts_quant(&x, x_stride, @intCast(plan.slots), &self.up_gs.?.view, &plan.items.view, &plan.counts.view, &plan.members.view, &rows_q.codes.view, &rows_q.scales.view, @intCast(self.dims), skipId(skip), @intCast(items)));
        const out: ?*const cuda.View = if (act) |*one| one else null;
        return cuda.launched(zs_nvfp4_experts_a4(if (self.gated) 4 else 7, a4Tiles(pass), @intFromBool(even_units), &rows_q.codes.view, &rows_q.scales.view, &self.up.view, &self.up_scale.view, @intCast(self.dims / 64), @intCast(blocks), &plan.items.view, &plan.counts.view, &plan.members.view, out, @intCast(self.width), self.limit, skipId(skip), @intCast(items * blocks), &rows_q.down_codes.view, &rows_q.down_scales.view, &self.down_gs.?.view));
    }

    /// W4A4: `rows_q`'s down rows, as `gateUpQuantizedAs` left them, -> out [rows * slots, dims], fp32 or bf16 by
    /// `out`'s dtype, with `pass`'s tiles a reading. Each pair's values are rounded to bf16 either way: the card's
    /// A4 FC2 stores each route bf16 before its fp32 top-k sum (b12x `prefill_a4.py`, the `fc2` launch's bf16 `y`
    /// and `compile_w4a16_topk_sum(element_dtype="bf16")`; spec 143 WS-08).
    pub fn downProjectQuantizedAs(self: Experts, plan: Plan, rows_q: Quantized, out: cuda.View, rows: usize, skip: ?usize, pass: Pass) cuda.Error!void {
        std.debug.assert(self.activations == .nvfp4);
        const blocks = self.dims / cols;
        const epilogue: i64 = if (out.dtype == .f32) 6 else 3;
        const items = maxItems(rows * plan.slots, plan.experts);
        return cuda.launched(zs_nvfp4_experts_a4(epilogue, a4Tiles(pass), @intFromBool(even_units), &rows_q.down_codes.view, &rows_q.down_scales.view, &self.down.view, &self.down_scale.view, @intCast((self.width + 63) / 64), @intCast(blocks), &plan.items.view, &plan.counts.view, &plan.members.view, &out, @intCast(self.dims), 0, skipId(skip), @intCast(items * blocks), null, null, null));
    }

    /// A16 on the W4A4 pack (spec 115): x [rows, dims] bf16 -> act [rows * slots, width] bf16, each routed pair's
    /// SwiGLU in fp32 rounded once; pairs of expert `skip` are left untouched. The plan's items of 16 (`Pass.narrow`).
    pub fn gateUpA16(self: Experts, x: cuda.View, plan: Plan, act: cuda.View, rows: usize, skip: ?usize) cuda.Error!void {
        std.debug.assert(self.activations == .nvfp4);
        const blocks = self.width / cols;
        const x_stride: i64 = if (x.row_stride != 0) x.row_stride else x.size[1];
        return cuda.launched(zs_nvfp4_experts_a16(5, @intFromBool(even_units), &x, x_stride, @intCast(plan.slots), &self.up.view, &self.up_ws2.?.view, @intCast(self.dims / 64), @intCast(blocks), &plan.items.view, &plan.counts.view, &plan.members.view, &act, @intCast(self.width), self.limit, skipId(skip), @intCast(maxItems(rows * plan.slots, plan.experts) * blocks)));
    }

    /// A16 on the W4A4 pack: act [rows * slots, width] bf16, as `gateUpA16` left it -> out [rows * slots, dims],
    /// fp32 or bf16 by `out`'s dtype.
    pub fn downProjectA16(self: Experts, act: cuda.View, plan: Plan, out: cuda.View, rows: usize, skip: ?usize) cuda.Error!void {
        std.debug.assert(self.activations == .nvfp4);
        const blocks = self.dims / cols;
        const epilogue: i64 = if (out.dtype == .f32) 0 else 3;
        const x_stride: i64 = if (act.row_stride != 0) act.row_stride else act.size[1];
        return cuda.launched(zs_nvfp4_experts_a16(epilogue, @intFromBool(even_units), &act, x_stride, 0, &self.down.view, &self.down_ws2.?.view, @intCast(self.width / 64), @intCast(blocks), &plan.items.view, &plan.counts.view, &plan.members.view, &out, @intCast(self.dims), 0, skipId(skip), @intCast(maxItems(rows * plan.slots, plan.experts) * blocks)));
    }

    /// x [rows, dims] bf16 -> out [rows * slots, width] bf16, each routed pair's SwiGLU (relu squared where not
    /// `gated`); pairs of expert `skip` are left untouched.
    pub fn gateUp(self: Experts, x: cuda.View, plan: Plan, out: cuda.View, rows: usize, skip: ?usize) cuda.Error!void {
        return self.gateUpAs(x, plan, out, rows, skip, .of(rows));
    }

    /// `gateUp` with `pass`'s tiles a reading, whatever the rows and whatever the plan's items.
    pub fn gateUpAs(self: Experts, x: cuda.View, plan: Plan, out: cuda.View, rows: usize, skip: ?usize, pass: Pass) cuda.Error!void {
        std.debug.assert(self.activations == .bf16);
        const blocks = self.width / cols;
        const x_stride: i64 = if (x.row_stride != 0) x.row_stride else x.size[1];
        const epilogue: i64 = if (self.gated) 2 else 1;
        return cuda.launched(zs_nvfp4_experts(epilogue, pass.tiles, @intFromBool(even_units), &x, x_stride, @intCast(plan.slots), &self.up.view, &self.up_scale.view, @intCast(self.dims / 32), @intCast(blocks), &plan.items.view, &plan.counts.view, &plan.members.view, &out, @intCast(self.width), self.limit, skipId(skip), @intCast(maxItems(rows * plan.slots, plan.experts) * blocks)));
    }

    /// act [rows * slots, width] bf16 -> out [rows * slots, dims], fp32 or bf16 by `out`'s dtype.
    pub fn downProject(self: Experts, act: cuda.View, plan: Plan, out: cuda.View, rows: usize, skip: ?usize) cuda.Error!void {
        return self.downProjectAs(act, plan, out, rows, skip, .of(rows));
    }

    /// `downProject` with `pass`'s tiles a reading.
    pub fn downProjectAs(self: Experts, act: cuda.View, plan: Plan, out: cuda.View, rows: usize, skip: ?usize, pass: Pass) cuda.Error!void {
        std.debug.assert(self.activations == .bf16);
        const blocks = self.dims / cols;
        const epilogue: i64 = if (out.dtype == .f32) 0 else 3;
        const x_stride: i64 = if (act.row_stride != 0) act.row_stride else act.size[1];
        return cuda.launched(zs_nvfp4_experts(epilogue, pass.tiles, @intFromBool(even_units), &act, x_stride, 0, &self.down.view, &self.down_scale.view, @intCast(self.width / 32), @intCast(blocks), &plan.items.view, &plan.counts.view, &plan.members.view, &out, @intCast(self.dims), 0, skipId(skip), @intCast(maxItems(rows * plan.slots, plan.experts) * blocks)));
    }

    fn skipId(skip: ?usize) i64 {
        return if (skip) |id| @intCast(id) else -1;
    }
};
