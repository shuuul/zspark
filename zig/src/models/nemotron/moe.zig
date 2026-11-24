//! A NemotronH MoE layer (`mixer.*`): the router (bf16 gate + fp32 `e_score_correction_bias`, sigmoid scores,
//! `glue.zig`'s sliced router and top-k), the `experts` routed experts of relu squared and the one shared expert
//! folded in as TensorFold's two pseudo-experts (ids `routed` and `routed + 1`: the shared up projection's row
//! halves, the shared down projection's column halves, each of `moe_width` = `shared_width` / 2), then the row's
//! delta through `glue.addMoeNorm` — the routed slots weighted, the two shared added — and its RMSNorm into the
//! rows the next projection reads (TensorFold `families/nemotron/cuda_forward.zig` `experts` and `norm`).
//!
//! MoE is stateless row by row, so a serial step, a verify window and a prompt chunk run the same launches:
//! `window` (up to 16 rows) and `prompt` (a chunk of any rows) are one `run`; the experts' own pass
//! (`weights/experts.zig` `Pass.of`) picks the kernel by rows, never a pair's bits. A row's weights are the
//! top-k's: `top` routed experts by sigmoid score plus bias (lower id on a tie), prob / sum * `scale`, then the
//! two shared slots at weight 1 (`glue.topk` writes ids `routed`, `routed + 1`). Rejected rows past `pos`
//! never reach the layer.
//!
//! Where it departs from TensorFold: zspark plans every slot in one call, including the shared halves,
//! and keeps slots' yk fp32 in prompt chunks. The backbone takes the checkpoint's NVFP4 activation
//! scales in every window (W4A4), with bf16 relu2 and separate routed/shared scales as vLLM does;
//! MTP's drafter-only requantized experts retain A16. Each form leaves a row's bits its own.

const std = @import("std");
const cuda = @import("../../cuda/cuda.zig");
const experts_file = @import("../../weights/experts.zig");
const requantize = @import("../../weights/requantize.zig");
const modelopt = @import("../../weights/modelopt.zig");
const layout_file = @import("layout.zig");
const config_file = @import("config.zig");
const glue = @import("glue.zig");
const ranks_file = @import("ranks.zig");

const Allocator = std.mem.Allocator;
const Experts = experts_file.Experts;
const Matrix = experts_file.Matrix;
const Diagnostic = config_file.Diagnostic;

pub const window_rows = 16;

pub const Error = modelopt.Error || error{ UnsupportedShape, InvalidCsf } || Allocator.Error || cuda.Error;

/// A layer's shape and routing; the defaults are Nemotron-3.5-Transcribe's.
pub const Shape = struct {
    dims: usize = 2688,
    /// The routed experts.
    routed: usize = 128,
    /// A routed expert's intermediate width; the shared pseudo-experts' too (`shared_width` = 2 * `width`).
    width: usize = 1856,
    /// Picked routed experts a row; the two shared slots follow (`slots` a row, up to 8).
    top: usize = 6,
    /// `routed_scaling_factor` (2.5).
    scale: f64 = 2.5,
    /// `norm_topk_prob`: the picks' probs normalized over their sum before the scale.
    norm: bool = true,
    /// The post-norm's RMS epsilon.
    eps: f64 = 1e-5,

    pub fn of(b: config_file.Backbone) Shape {
        std.debug.assert(b.shared_width == 2 * b.moe_width);
        return .{ .dims = b.hidden, .routed = b.experts, .width = b.moe_width, .top = b.top_k, .scale = b.routed_scale, .norm = b.norm_topk, .eps = b.eps };
    }

    /// The weights' experts: the routed ones, then the shared expert's two halves.
    pub fn count(s: Shape) usize {
        return s.routed + 2;
    }

    /// The slots top-k writes a row: `top` routed, then the two shared halves' ids.
    pub fn slots(s: Shape) usize {
        return s.top + 2;
    }
};

/// A layer's weights as `init` takes them: the checkpoint tensors read (router and bias bytes, each
/// projection's `Matrix`), so a test can stage a synthetic checkpoint without `load`'s names.
pub const Host = struct {
    /// `mixer.gate.weight` bf16 [routed, dims]
    router: []align(1) const u16,
    /// `mixer.gate.e_score_correction_bias` fp32 [routed]
    bias: []align(1) const f32,
    /// The routed experts' `up_proj` [width, dims] and `down_proj` [dims, width] a piece, NVFP4 by `Matrix`.
    up: Matrix,
    down: Matrix,
    /// `mixer.shared_experts.up_proj` [2 * width, dims] and `down_proj` [dims, 2 * width].
    shared_up: Matrix,
    shared_down: Matrix,
    activations: experts_file.Activations = .bf16,
};

/// A MoE layer's weights on the device. `staging` holds the packed blocks until the upload; a loader passes
/// the same one for every layer.
pub const Moe = struct {
    shape: Shape,
    /// bf16 [routed, dims]: the gate as stored.
    router: cuda.Buffer,
    /// fp32 [routed]
    bias: cuda.Buffer,
    /// `routed + 2` relu-squared experts: backbone W4A4, MTP A16; shared halves last.
    experts: Experts,

    /// The layer `prefix` holds (`backbone.layers.N.mixer` or `llm.mtp.layers.N.mixer`), `source` anything
    /// with `get(name) ?Tensor`; `quant` (`config.Quant.find`) gives a module's storage.
    pub fn load(arena: Allocator, gpa: Allocator, staging: *experts_file.Staging, source: anytype, s: Shape, prefix: []const u8, quant: config_file.Quant, diagnostic: *Diagnostic, share: ?ranks_file.Share) Error!Moe {
        const full = share != null;
        // `s` is the rank's shape (its `width` is the stored one's half at two ranks); `full_width` the
        // stored expert's. The shared expert stores two pseudo-experts of `full_width`.
        const full_width: usize = if (full) s.width * ranks_file.world else s.width;
        var name: [512]u8 = undefined;
        const at = struct {
            fn of(buffer: []u8, p: []const u8, leaf: []const u8) []const u8 {
                return std.fmt.bufPrint(buffer, "{s}.{s}", .{ p, leaf }) catch unreachable;
            }
        }.of;
        const router_name = at(&name, prefix, "gate.weight");
        const router_tensor = source.get(router_name) orelse {
            diagnostic.set(arena, "the checkpoint has no {s}", .{router_name});
            return error.MissingTensor;
        };
        if (router_tensor.dtype != .BF16 or !std.mem.eql(u64, router_tensor.shape, &.{ s.routed, s.dims })) {
            diagnostic.set(arena, "{s} is {t} {any}; [{d}, {d}] bf16", .{ router_name, router_tensor.dtype, router_tensor.shape, s.routed, s.dims });
            return error.UnexpectedTensor;
        }
        const bias_name = at(&name, prefix, "gate.e_score_correction_bias");
        const bias_tensor = source.get(bias_name) orelse {
            diagnostic.set(arena, "the checkpoint has no {s}", .{bias_name});
            return error.MissingTensor;
        };
        if (bias_tensor.dtype != .F32 or bias_tensor.count() != s.routed) {
            diagnostic.set(arena, "{s} is {t} {any}; [{d}] fp32", .{ bias_name, bias_tensor.dtype, bias_tensor.shape, s.routed });
            return error.UnexpectedTensor;
        }

        // a projection's routed experts: NVFP4 as stored, or FP8 requantized on the host — the MTP layer's
        // experts come FP8 and only draft, so what the conversion loses changes offers, never a token
        // (`requantize`; the root AGENTS.md's quantization exception for a drafter). A backbone expert
        // stored FP8 is refused outright: requantizing one would move real logits, not just drafts.
        const drafter_only = std.mem.startsWith(u8, prefix, layout_file.mtp_prefix);
        var quantized: [4]?requantize.Quantized = @splat(null);
        defer for (&quantized) |*owned| if (owned.*) |*one| one.deinit(gpa);
        const Pick = enum { up, down, shared_up, shared_down };
        var matrices: [4]Matrix = undefined;
        for (&matrices, &quantized, [_]Pick{ .up, .down, .shared_up, .shared_down }) |*matrix, *owned, pick| {
            const shared = pick == .shared_up or pick == .shared_down;
            // up projections are [expert width, hidden], down [hidden, expert width]; the shared expert's
            // width is the two pseudo-experts' (2 * width)
            const n: u64 = if (pick == .up) full_width else if (pick == .down) s.dims else if (pick == .shared_up) 2 * full_width else s.dims;
            const k: u64 = if (pick == .up or pick == .shared_up) s.dims else if (pick == .down) full_width else 2 * full_width;
            const rows = n;
            var module: [512]u8 = undefined;
            if (shared) {
                const leaf = if (pick == .shared_up) "up_proj" else "down_proj";
                const shared_module = std.fmt.bufPrint(&module, "{s}.shared_experts.{s}", .{ prefix, leaf }) catch unreachable;
                matrix.* = try sharedMatrix(arena, gpa, owned, source, shared_module, quant.find(shared_module), rows, k, full_width, share, pick == .shared_up, drafter_only, diagnostic);
            } else {
                const leaf = if (pick == .up) "up_proj" else "down_proj";
                const codes = try arena.alloc([]const u8, s.routed);
                const scales = try arena.alloc([]const u8, s.routed);
                const factors = try arena.alloc(f32, s.routed);
                const inputs = try arena.alloc(f32, if (drafter_only) 0 else s.routed);
                const weights = try arena.alloc(requantize.Weight, s.routed);
                var stored: usize = 0;
                for (0..s.routed) |e| {
                    const one = std.fmt.bufPrint(&module, "{s}.experts.{d}.{s}", .{ prefix, e, leaf }) catch unreachable;
                    var linear = try modelopt.read(arena, source, one, quant.find(one), n, k, diagnostic);
                    if (share) |sh| {
                        // each rank keeps its half of every expert's width: up's rows, down's input columns
                        linear = try modelopt.cut(arena, linear, if (pick == .up) .row else .col, sh.rank, ranks_file.world, diagnostic);
                    }
                    switch (linear.scheme) {
                        .nvfp4 => {
                            codes[e] = linear.weight.data;
                            scales[e] = try linear.blockScales(arena);
                            factors[e] = @bitCast(std.mem.readInt(u32, linear.scale_2.?.data[0..4], .little));
                            if (!drafter_only) inputs[e] = try activationScale(arena, linear, one, diagnostic);
                            stored += 1;
                        },
                        .fp8 => {
                            if (!drafter_only) {
                                diagnostic.set(arena, "{s} is stored FP8; only the MTP block's experts are requantized on the host", .{one});
                                return error.UnexpectedTensor;
                            }
                            weights[e] = .{ .fp8 = .{ .codes = linear.weight.data, .scales = .of(&.{@as(f32, @bitCast(std.mem.readInt(u32, linear.scale_2.?.data[0..4], .little)))}) } };
                        },
                        else => {
                            diagnostic.set(arena, "{s} is stored {t}; zspark reads the experts NVFP4 or FP8", .{ one, linear.scheme });
                            return error.UnexpectedTensor;
                        },
                    }
                }
                if (stored == s.routed) {
                    matrix.* = .{ .codes = codes, .block_scales = scales, .scales = factors, .input_scales = inputs };
                } else if (stored == 0) {
                    const cut_rows = if (pick == .up) s.width else s.dims;
                    const cut_cols = if (pick == .up) s.dims else s.width;
                    owned.* = requantize.Quantized.init(gpa, weights, cut_rows, cut_cols) catch |err| return switch (err) {
                        error.OutOfMemory => error.OutOfMemory,
                        else => error.UnsupportedShape,
                    };
                    matrix.* = owned.*.?.matrix();
                } else return error.UnexpectedTensor;
            }
        }
        const host: Host = .{
            .router = std.mem.bytesAsSlice(u16, router_tensor.data),
            .bias = std.mem.bytesAsSlice(f32, bias_tensor.data),
            .up = matrices[0],
            .down = matrices[1],
            .shared_up = matrices[2],
            .shared_down = matrices[3],
            .activations = if (drafter_only) .bf16 else .nvfp4,
        };
        return init(gpa, staging, s, host);
    }

    /// One projection's `Matrix`: NVFP4 as stored, or FP8 requantized into `owned` (a shared projection is one
    /// tensor of every pseudo-expert, still one `Quantized` of one weight). `drafter` says the layer is the
    /// MTP block's, the only place FP8 experts are taken (see `load`).
    fn expertMatrix(arena: Allocator, gpa: Allocator, owned: *?requantize.Quantized, source: anytype, module: []const u8, layer: ?modelopt.QuantizedLayer, rows: u64, cols: u64, drafter: bool, diagnostic: *Diagnostic) Error!Matrix {
        const linear = try modelopt.read(arena, source, module, layer, rows, cols, diagnostic);
        return matrixOf(arena, gpa, owned, linear, module, rows, cols, drafter, diagnostic);
    }

    /// The shared expert's one projection as a rank holds it: read whole, then the rank's quarter — its
    /// pseudo-experts' width halves (`up` by rows of the stored [2 * width, dims], `down` by the same
    /// columns), which `init` then splits into two experts of the rank's `width` / 2... `s.width`. `up`
    /// selects rows [w0, w1) ∪ [w + w0, w + w1) of the stored width w.
    fn sharedMatrix(arena: Allocator, gpa: Allocator, owned: *?requantize.Quantized, source: anytype, module: []const u8, layer: ?modelopt.QuantizedLayer, rows: u64, cols: u64, full_width: usize, share: ?ranks_file.Share, is_up: bool, drafter: bool, diagnostic: *Diagnostic) Error!Matrix {
        var linear = try modelopt.read(arena, source, module, layer, rows, cols, diagnostic);
        var out_rows = rows;
        var out_cols = cols;
        if (share) |sh| {
            const w = full_width;
            const half = w / ranks_file.world;
            const lo = sh.rank * half;
            if (is_up) {
                // rows [lo, hi) of each pseudo-expert's [w] block: the rank's two pseudo-experts of half
                const idx = try arena.alloc(usize, 2 * half);
                for (0..half) |i| {
                    idx[i] = lo + i;
                    idx[half + i] = w + lo + i;
                }
                linear = try ranks_file.linearRows(arena, linear, idx, diagnostic);
                out_rows = 2 * half;
            } else {
                // the same columns of the stored [dims, 2 w]: block units of the scheme's group
                const block: usize = switch (linear.scheme) {
                    .nvfp4 => modelopt.nvfp4_block,
                    .mxfp8 => modelopt.mxfp8_block,
                    .fp8_block => modelopt.fp8_block,
                    else => 1,
                };
                if (w % (block * ranks_file.world) != 0) {
                    diagnostic.set(arena, "{s}: the shared down's width {d} does not quarter on blocks of {d}", .{ module, w, block });
                    return error.Indivisible;
                }
                const each = w / block;
                const part = half / block;
                const first = sh.rank * part;
                const idx = try arena.alloc(usize, 2 * part);
                for (0..part) |i| {
                    idx[i] = first + i;
                    idx[part + i] = each + first + i;
                }
                linear = try ranks_file.linearCols(arena, linear, idx, block, diagnostic);
                out_cols = 2 * half;
            }
            if (is_up) std.debug.assert(out_rows == 2 * half) else std.debug.assert(out_cols == 2 * half);
        }
        return matrixOf(arena, gpa, owned, linear, module, out_rows, out_cols, drafter, diagnostic);
    }

    /// `linear` (already the rank's cut where two ranks run) as a `Matrix`: NVFP4 as stored, or FP8
    /// requantized into `owned` under the MTP prefix alone.
    fn matrixOf(arena: Allocator, gpa: Allocator, owned: *?requantize.Quantized, linear: modelopt.Linear, module: []const u8, rows: u64, cols: u64, drafter: bool, diagnostic: *Diagnostic) Error!Matrix {
        switch (linear.scheme) {
            .nvfp4 => {
                const codes = try arena.alloc([]const u8, 1);
                const scales = try arena.alloc([]const u8, 1);
                const factors = try arena.alloc(f32, 1);
                codes[0] = linear.weight.data;
                scales[0] = try linear.blockScales(arena);
                factors[0] = @bitCast(std.mem.readInt(u32, linear.scale_2.?.data[0..4], .little));
                const inputs = try arena.alloc(f32, if (drafter) 0 else 1);
                if (!drafter) inputs[0] = try activationScale(arena, linear, module, diagnostic);
                return .{ .codes = codes, .block_scales = scales, .scales = factors, .input_scales = inputs };
            },
            .fp8 => {
                if (!drafter) {
                    diagnostic.set(arena, "{s} is stored FP8; only the MTP block's experts are requantized on the host", .{module});
                    return error.UnexpectedTensor;
                }
                const scale: f32 = @bitCast(std.mem.readInt(u32, linear.scale_2.?.data[0..4], .little));
                const weight: requantize.Weight = .{ .fp8 = .{ .codes = linear.weight.data, .scales = .of(&.{scale}) } };
                owned.* = requantize.Quantized.init(gpa, &.{weight}, rows, cols) catch |err| return switch (err) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.UnsupportedShape,
                };
                return owned.*.?.matrix();
            },
            else => {
                diagnostic.set(arena, "{s} is stored {t}; zspark reads the experts NVFP4 or FP8", .{ module, linear.scheme });
                return error.UnexpectedTensor;
            },
        }
    }

    fn activationScale(arena: Allocator, linear: modelopt.Linear, module: []const u8, diagnostic: *Diagnostic) Error!f32 {
        const t = linear.input_scale orelse {
            diagnostic.set(arena, "{s}: the NVFP4 backbone expert has no input_scale", .{module});
            return error.MissingTensor;
        };
        return @bitCast(std.mem.readInt(u32, t.data[0..4], .little));
    }

    /// `host`'s weights packed and uploaded: the shared expert's halves appended to the routed experts'
    /// matrices (up's rows, down's columns) as TensorFold's `weights.fold_shared` makes experts E and E + 1.
    pub fn init(gpa: Allocator, staging: *experts_file.Staging, s: Shape, host: Host) Error!Moe {
        if (host.router.len != s.routed * s.dims or host.bias.len != s.routed) return error.UnexpectedTensor;
        const e = s.count();
        if (host.up.codes.len != e - 2 or host.down.codes.len != e - 2) return error.UnexpectedTensor;
        for ([_]Matrix{ host.up, host.down, host.shared_up, host.shared_down }) |m| {
            if (m.block_scales.len != m.codes.len or m.scales.len != m.codes.len) return error.UnexpectedTensor;
        }
        const codes = try gpa.alloc([]const u8, e);
        defer gpa.free(codes);
        const scales = try gpa.alloc([]const u8, e);
        defer gpa.free(scales);
        const down_codes = try gpa.alloc([]const u8, e);
        defer gpa.free(down_codes);
        const down_scales = try gpa.alloc([]const u8, e);
        defer gpa.free(down_scales);
        const up_factors = try gpa.alloc(f32, e);
        defer gpa.free(up_factors);
        const down_factors = try gpa.alloc(f32, e);
        defer gpa.free(down_factors);
        const quantized = host.activations == .nvfp4;
        const up_inputs = try gpa.alloc(f32, if (quantized) e else 0);
        defer gpa.free(up_inputs);
        const down_inputs = try gpa.alloc(f32, if (quantized) e else 0);
        defer gpa.free(down_inputs);
        if (quantized) {
            for ([_]Matrix{ host.up, host.down }) |m|
                if (m.input_scales.len != s.routed) return error.UnexpectedTensor;
            if (host.shared_up.input_scales.len != 1 or host.shared_down.input_scales.len != 1) return error.UnexpectedTensor;
            var routed_input: f32 = 0;
            for (host.up.input_scales) |one| routed_input = @max(routed_input, one);
            @memset(up_inputs[0..s.routed], routed_input);
            @memcpy(down_inputs[0..s.routed], host.down.input_scales);
            @memset(up_inputs[s.routed..], host.shared_up.input_scales[0]);
            @memset(down_inputs[s.routed..], host.shared_down.input_scales[0]);
        }
        for (0..s.routed) |i| {
            codes[i] = host.up.codes[i];
            scales[i] = host.up.block_scales[i];
            up_factors[i] = host.up.scales[i];
            down_codes[i] = host.down.codes[i];
            down_scales[i] = host.down.block_scales[i];
            down_factors[i] = host.down.scales[i];
        }
        // the shared expert's halves as experts routed and routed + 1: up's row halves (slices of the stored
        // bytes), down's column halves (copies, kept until `initRelu2` has packed and uploaded them)
        const up_halves = try split(host.shared_up, 2 * s.width, s.dims, true, gpa, codes[e - 2 ..], scales[e - 2 ..]);
        defer for (up_halves) |p| {
            if (p[0]) |b| gpa.free(b);
            if (p[1]) |b| gpa.free(b);
        };
        const down_halves = try split(host.shared_down, s.dims, 2 * s.width, false, gpa, down_codes[e - 2 ..], down_scales[e - 2 ..]);
        defer for (down_halves) |p| {
            if (p[0]) |b| gpa.free(b);
            if (p[1]) |b| gpa.free(b);
        };
        up_factors[e - 2] = host.shared_up.scales[0];
        up_factors[e - 1] = host.shared_up.scales[0];
        down_factors[e - 2] = host.shared_down.scales[0];
        down_factors[e - 1] = host.shared_down.scales[0];

        var router: cuda.Buffer = try .init(.bf16, &.{ s.routed, s.dims });
        errdefer router.deinit();
        try router.upload(std.mem.sliceAsBytes(host.router));
        var bias: cuda.Buffer = try .init(.f32, &.{s.routed});
        errdefer bias.deinit();
        try bias.upload(std.mem.sliceAsBytes(host.bias));
        const up: Matrix = .{ .codes = codes, .block_scales = scales, .scales = up_factors, .input_scales = up_inputs };
        const down: Matrix = .{ .codes = down_codes, .block_scales = down_scales, .scales = down_factors, .input_scales = down_inputs };
        const experts = if (quantized)
            try Experts.initRelu2Quantized(gpa, staging, e, s.width, s.dims, up, down)
        else
            try Experts.initRelu2(gpa, staging, e, s.width, s.dims, up, down);
        return .{ .shape = s, .router = router, .bias = bias, .experts = experts };
    }

    /// `matrix` [rows, cols] into two experts' codes and block scales: `rows_split` cuts the rows in half
    /// (slices of the stored bytes), else the input columns (each row's halves copied into `gpa`). The pairs
    /// returned own the copies for the caller to free.
    fn split(matrix: Matrix, rows: usize, cols: usize, rows_split: bool, gpa: Allocator, codes: [][]const u8, scales: [][]const u8) ![2][2]?[]const u8 {
        var owned: [2][2]?[]const u8 = .{ .{ null, null }, .{ null, null } };
        if (matrix.codes.len != 1 or matrix.scales.len != 1) return error.UnexpectedTensor;
        if (matrix.codes[0].len != rows * cols / 2 or matrix.block_scales[0].len != rows * cols / 16) return error.UnexpectedTensor;
        if (rows_split) {
            const half = rows / 2;
            codes[0] = matrix.codes[0][0 .. half * cols / 2];
            codes[1] = matrix.codes[0][half * cols / 2 ..];
            scales[0] = matrix.block_scales[0][0 .. half * cols / 16];
            scales[1] = matrix.block_scales[0][half * cols / 16 ..];
        } else {
            // the column halves: codes [rows, cols / 2] -> [rows, cols / 4] a half, scales [rows, cols / 16] ->
            // [rows, cols / 32] (whole blocks of 16 a half: cols is even in blocks here, 3712 = 2 * 1856)
            if (cols % 32 != 0) return error.UnexpectedTensor;
            const code_row = cols / 2;
            const scale_row = cols / 16;
            for (0..2) |half| {
                const c = try gpa.alloc(u8, rows * code_row / 2);
                errdefer gpa.free(c);
                const sc = try gpa.alloc(u8, rows * scale_row / 2);
                errdefer gpa.free(sc);
                for (0..rows) |r| {
                    @memcpy(c[r * code_row / 2 ..][0 .. code_row / 2], matrix.codes[0][r * code_row + half * code_row / 2 ..][0 .. code_row / 2]);
                    @memcpy(sc[r * scale_row / 2 ..][0 .. scale_row / 2], matrix.block_scales[0][r * scale_row + half * scale_row / 2 ..][0 .. scale_row / 2]);
                }
                codes[half] = c;
                scales[half] = sc;
                owned[half] = .{ c, sc };
            }
        }
        return owned;
    }

    pub fn deinit(m: *Moe) void {
        m.router.deinit();
        m.bias.deinit();
        m.experts.deinit();
        m.* = undefined;
    }
};

/// What a call lends, made once for the most rows a prompt chunk takes and shared by every MoE layer.
pub const Scratch = struct {
    rows: usize,
    /// fp32 [SK, rows, routed]: the router's slices (`glue.routerSlices`).
    part: cuda.Buffer,
    /// int32 and fp32 [rows, slots]: each row's picks (routed ids, then `routed`, `routed + 1`) and weights.
    picks: cuda.Buffer,
    wts: cuda.Buffer,
    plan: experts_file.Plan,
    /// A16 only: bf16 [rows * slots, width]. A4 quantizes in the epilogue without this intermediate.
    act: ?cuda.Buffer,
    /// fp32 [rows * slots, dims]: each slot's output row; `combine` reads it [rows, slots, dims].
    y: cuda.Buffer,
    /// fp32 [rows, dims / 64]: the norm's group sums (nothing reads them; `addMoeNorm` writes them).
    xs: cuda.Buffer,
    quantized: ?experts_file.Quantized,

    pub fn init(s: Shape, rows: usize, quantized: bool) cuda.Error!Scratch {
        const slots = s.slots();
        var sc: Scratch = undefined;
        sc.rows = rows;
        sc.part = try .init(.f32, &.{ glue.routerSlices(s.dims), rows, s.routed });
        errdefer sc.part.deinit();
        sc.picks = try .init(.i32, &.{ rows, slots });
        errdefer sc.picks.deinit();
        sc.wts = try .init(.f32, &.{ rows, slots });
        errdefer sc.wts.deinit();
        sc.plan = try .init(rows, slots, s.count());
        errdefer sc.plan.deinit();
        sc.act = if (quantized) null else try .init(.bf16, &.{ rows * slots, s.width });
        errdefer if (sc.act) |*act| act.deinit();
        sc.y = try .init(.f32, &.{ rows * slots, s.dims });
        errdefer sc.y.deinit();
        sc.xs = try .init(.f32, &.{ rows, s.dims / 64 });
        errdefer sc.xs.deinit();
        sc.quantized = if (quantized) try .init(rows * slots, s.dims, s.width) else null;
        return sc;
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(s: Shape, rows: usize, quantized: bool) usize {
        const of = cuda.Buffer.bytesOf;
        const slots = s.slots();
        const padded_width = std.mem.alignForward(usize, s.width, 64);
        const q = if (quantized) of(.u8, &.{ rows * slots, s.dims / 2 }) + of(.u8, &.{ rows * slots, s.dims / 16 }) +
            of(.u8, &.{ rows * slots, padded_width / 2 }) + of(.u8, &.{ rows * slots, padded_width / 16 }) else 0;
        return of(.f32, &.{ glue.routerSlices(s.dims), rows, s.routed }) + of(.i32, &.{ rows, slots }) + of(.f32, &.{ rows, slots }) +
            experts_file.Plan.bytesFor(rows, slots, s.count()) + (if (quantized) 0 else of(.bf16, &.{ rows * slots, s.width })) + of(.f32, &.{ rows * slots, s.dims }) + of(.f32, &.{ rows, s.dims / 64 }) + q;
    }

    pub fn deinit(sc: *Scratch) void {
        sc.part.deinit();
        sc.picks.deinit();
        sc.wts.deinit();
        sc.plan.deinit();
        if (sc.act) |*act| act.deinit();
        sc.y.deinit();
        sc.xs.deinit();
        if (sc.quantized) |*q| q.deinit();
        sc.* = undefined;
    }
};

/// A window of 1 to `window_rows` rows (a step, a verify window): x bf16 [rows, dims], the block-norm's
/// output -> each row's slots in `sc.y` (fp32 [rows * slots, dims]) and `sc.wts`. `combine` adds them to the
/// residual.
pub fn window(m: Moe, sc: Scratch, x: cuda.View) cuda.Error!void {
    const rows: usize = @intCast(x.size[0]);
    try cuda.bound(rows >= 1 and rows <= window_rows, "moe.window: {d} rows, 1 to {d}", .{ rows, window_rows });
    return run(m, sc, x);
}

/// A prompt chunk of 1 to `sc.rows` rows: `window`'s launches exactly (MoE is stateless row by row); the
/// experts' plan and kernels pass by `rows` (`Experts.Pass.of`), a pair's bits unchanged.
pub fn prompt(m: Moe, sc: Scratch, x: cuda.View) cuda.Error!void {
    const rows: usize = @intCast(x.size[0]);
    try cuda.bound(rows >= 1 and rows <= sc.rows, "moe.prompt: {d} rows, 1 to {d}", .{ rows, sc.rows });
    return run(m, sc, x);
}

/// `window`/`prompt`'s common path: route every row, plan all `slots` picks by expert (the shared halves'
/// fixed ids included; TensorFold's decode plans them aside on a second stream), then up (relu²) and down.
pub fn run(m: Moe, sc: Scratch, x: cuda.View) cuda.Error!void {
    const s = m.shape;
    const rows: usize = @intCast(x.size[0]);
    const slots = s.slots();
    const picks = sc.picks.rows(rows);
    const wts = sc.wts.rows(rows);
    // part is slice-major [SK, sc.rows, E]; a call's is [SK, rows, E]
    var sliced = sc.part.view;
    sliced.size[1] = @intCast(rows);
    try glue.route(x, m.router.view, m.bias.view, sliced, picks, wts, s.scale, s.top, slots, s.norm);
    const y = sc.y.rows(rows * slots);
    if (m.experts.activations == .nvfp4) {
        const pass = experts_file.Activations.passOf(rows);
        try sc.plan.routeAs(picks, rows, pass);
        try m.experts.gateUpQuantizedAs(x, sc.plan, sc.quantized.?, null, rows, null, pass);
        try m.experts.downProjectQuantizedAs(sc.plan, sc.quantized.?, y, rows, null, pass);
    } else {
        const act = sc.act.?.rows(rows * slots);
        const pass = experts_file.Pass.of(rows);
        try sc.plan.routeAs(picks, rows, pass);
        try m.experts.gateUpAs(x, sc.plan, act, rows, null, pass);
        try m.experts.downProjectAs(act, sc.plan, y, rows, null, pass);
    }
}

/// The layer's end (`cuda_forward.zig` `norm` with the MoE delta): `sc.y`'s slots and `sc.wts` fold into the
/// residual — delta = bf16(routed slots weighted in slot order + the two shared rows), h = bf16(x + delta),
/// out = bf16(rmsnorm(h) * w) and xs its group sums. w is the next block's (or the final) norm, bf16 [dims].
pub fn combine(m: Moe, sc: Scratch, x: cuda.View, w: cuda.View, h: cuda.View, out: cuda.View, rows: usize) cuda.Error!void {
    const s = m.shape;
    const yk: cuda.View = .{ .ptr = sc.y.view.ptr, .dtype = .f32, .ndim = 3, .size = .{ @intCast(rows), @intCast(s.slots()), @intCast(s.dims), 0, 0, 0 } };
    const wts = sc.wts.rows(rows);
    const xs = sc.xs.rows(rows);
    return glue.addMoeNorm(x, yk, wts, w, h, out, xs, s.eps, s.top);
}
