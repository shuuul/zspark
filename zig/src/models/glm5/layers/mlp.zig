//! GLM-5.3-Flash's MLPs over a rank's share: the dense MLP of the first layers, and the MoE (a BF16 router, the
//! routed experts NVFP4 on the grouped kernel of `weights/experts.zig`, the shared expert a
//! dense MLP). A dense MLP's linears run as stored (BF16, or MXFP8 and FP8_PB_WO in FP4-Local) on their faces. Each ends in this rank's fp32 partial of
//! the block, [rows, dims]: the ranks' partials are added rank 0 first by the write-back (`hc.post`).
//! TensorFold's `forward.py` `mlp_block` and `moe_block`, `weights.py` `mlp` and `moe`.
//!
//! A rank holds half of an inner width: half of the gate's and the up's outputs and the same half of the down's
//! inputs. The activation between them is element-wise, so the two ranks' partials add to the whole product.
//! The router is whole on both ranks, which therefore pick the same experts.

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("../../../weights/bf16.zig");
const cuda = @import("../../../cuda/cuda.zig");
const experts = @import("../../../weights/experts.zig");
const face = @import("../../../weights/face.zig");
const Config = @import("config.zig").Config;
const layout = @import("layout.zig");
const moe = @import("moe.zig");
const share = @import("share.zig");

pub const LoadError = error{ MissingTensor, UnsupportedShape, Unsupported } || Allocator.Error || cuda.Error;

/// The first `rows` rows of a buffer of rows.
/// A gated MLP: a rank's half of the inner width, each linear as stored.
pub const Dense = struct {
    /// [2 width, dims]: the rank's gate rows, then its up rows.
    gate_up: face.Face,
    /// [dims, width]: the rank's columns.
    down: face.Face,
    /// The rank's share of the inner width.
    width: usize,
    /// SwiGLU's clip: the gate from above, the up on both sides.
    limit: f32,

    /// `gate` and `up` bf16 [width, dims] and `down` bf16 [dims, width], a rank's shares as `share.take` gives
    /// them.
    pub fn init(gate: []const u8, up: []const u8, down: []const u8, dims: usize, limit: f32, rows: cuda.Scratch.Rows) LoadError!Dense {
        if (gate.len % (dims * 2) != 0) return error.UnsupportedShape;
        return initParts(std.heap.page_allocator, .{ .bf16 = gate }, .{ .bf16 = up }, .{ .bf16 = down }, dims, gate.len / (dims * 2), limit, rows);
    }

    /// `gate` and `up` [width, dims] and `down` [dims, width] as stored; `scratch` holds what a linear is
    /// packed in while it is.
    pub fn initParts(scratch: Allocator, gate: face.Part, up: face.Part, down: face.Part, dims: usize, width: usize, limit: f32, rows: cuda.Scratch.Rows) LoadError!Dense {
        if (width == 0 or !share.shaped(gate, width, dims) or !share.shaped(up, width, dims) or !share.shaped(down, dims, width)) return error.UnsupportedShape;
        var gate_up = try share.faceOf(scratch, &.{ gate, up }, dims, rows);
        errdefer gate_up.deinit();
        const down_linear = try share.faceOf(scratch, &.{down}, width, rows);
        return .{ .gate_up = gate_up, .down = down_linear, .width = width, .limit = limit };
    }

    /// The MLP under `prefix` (`...layers.N.mlp.` or `...mlp.shared_experts.`) as `rank` holds it.
    pub fn load(arena: Allocator, source: anytype, comptime format: []const u8, args: anytype, c: *const Config, rank: usize, rows: cuda.Scratch.Rows) LoadError!Dense {
        const prefix = try std.fmt.allocPrint(arena, format, args);
        var parts: [3]face.Part = undefined;
        for (&parts, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |*part, leaf|
            part.* = try share.partOf(arena, source, c, try std.mem.concat(arena, u8, &.{ prefix, leaf }), rank);
        const width = switch (parts[0]) {
            .bf16 => |bytes| bytes.len / (c.hidden * 2),
            .mxfp8 => |stored| stored.n,
            .fp8_block => |stored| stored.n,
        };
        return initParts(arena, parts[0], parts[1], parts[2], c.hidden, width, @floatCast(c.swiglu_limit), rows);
    }

    pub fn deinit(self: *Dense) void {
        self.gate_up.deinit();
        self.down.deinit();
        self.* = undefined;
    }

    /// x bf16 [rows, dims] -> out fp32 or bf16 [rows, dims], the rank's partial. `gate_up` bf16 [rows, 2 width] and
    /// `act` bf16 [rows, width] are scratch.
    pub fn forward(self: Dense, x: cuda.View, gate_up: cuda.View, act: cuda.View, out: cuda.View) cuda.Error!void {
        std.debug.assert(out.dtype == .f32 or out.dtype == .bf16);
        try self.gate_up.forward(x, gate_up);
        try moe.swiglu(gate_up, act, self.limit);
        try self.down.forward(act, out);
    }

    /// Scratch for calls of up to `rows` rows.
    pub const Buffers = struct {
        gate_up: cuda.Buffer,
        act: cuda.Buffer,

        pub fn init(width: usize, rows: usize) cuda.Error!Buffers {
            var gate_up: cuda.Buffer = try .init(.bf16, &.{ rows, 2 * width });
            errdefer gate_up.deinit();
            return .{ .gate_up = gate_up, .act = try .init(.bf16, &.{ rows, width }) };
        }

        pub fn deinit(self: *Buffers) void {
            self.gate_up.deinit();
            self.act.deinit();
            self.* = undefined;
        }
    };

    pub fn run(self: Dense, buffers: Buffers, x: cuda.View, out: cuda.View) cuda.Error!void {
        const rows: usize = @intCast(x.size[0]);
        return self.forward(x, buffers.gate_up.rows(rows), buffers.act.rows(rows), out);
    }
};

/// A layer's MoE as a rank holds it.
pub const Moe = struct {
    /// [experts, dims], whole.
    router: bf16.Linear,
    /// fp32 [experts]: added to a score for the choice alone.
    bias: cuda.Buffer,
    /// The routed experts: gate and up dims -> width, down width -> dims, `width` the rank's half. NVFP4 on the
    /// grouped kernel, whose SwiGLU clips as `zs_swiglu` does.
    routed: experts.Experts,
    shared: Dense,
    top: usize,
    scale: f32,
    norm: bool,

    /// Layer `layer`'s MoE of `source` for `rank`. `gpa` stages the experts' scales; the weights are uploaded
    /// from `arena`'s copies and the checkpoint's bytes, so the caller holds them and frees `arena` after.
    pub fn load(arena: Allocator, gpa: Allocator, source: anytype, c: *const Config, layer: usize, rank: usize, rows: cuda.Scratch.Rows) LoadError!Moe {
        var staging: experts.Staging = .{ .place = placesOn(@TypeOf(source)) };
        defer staging.deinit(gpa);
        var loaded = try loadWith(arena, gpa, &staging, source, c, layer, rank, rows);
        errdefer loaded.deinit();
        try staging.settle();
        return loaded;
    }

    /// Whether `Source` holds a part's bytes until the device has placed its experts (`Staging.place`): a
    /// type that says so (`session/pack.zig`'s `View`).
    pub fn placesOn(comptime Source: type) bool {
        return @typeInfo(Source) == .@"struct" and @hasDecl(Source, "staged");
    }

    /// `load`, NVFP4 experts packed in `staging`, which a loader passes for every layer.
    pub fn loadWith(arena: Allocator, gpa: Allocator, staging: *experts.Staging, source: anytype, c: *const Config, layer: usize, rank: usize, rows: cuda.Scratch.Rows) LoadError!Moe {
        const p = layout.prefix ++ "layers.{d}.mlp.";
        const router_name = try std.fmt.allocPrint(arena, p ++ "gate.weight", .{layer});
        const router_tensor = source.get(router_name) orelse return missing(router_name);
        const bias_name = try std.fmt.allocPrint(arena, p ++ "gate.e_score_correction_bias", .{layer});
        const bias_tensor = source.get(bias_name) orelse return missing(bias_name);
        if (router_tensor.dtype != .BF16 or router_tensor.count() != c.experts * c.hidden) return error.UnsupportedShape;
        if (bias_tensor.dtype != .F32 or bias_tensor.count() != c.experts) return error.UnsupportedShape;

        var router: bf16.Linear = try .init(router_tensor.data, c.experts, c.hidden, rows);
        errdefer router.deinit();
        var bias: cuda.Buffer = try .init(.f32, &.{c.experts});
        errdefer bias.deinit();
        try bias.upload(bias_tensor.data);
        var shared: Dense = try .load(arena, source, p ++ "shared_experts.", .{layer}, c, rank, rows);
        errdefer shared.deinit();

        var routed = try nvfp4Experts(arena, gpa, staging, source, c, layer, rank, null);
        errdefer routed.deinit();
        if (routed.dims != c.hidden or routed.width != c.moe_width / layout.world or routed.width != shared.width) return error.UnsupportedShape;
        return .{ .router = router, .bias = bias, .routed = routed, .shared = shared, .top = c.top_k, .scale = @floatCast(c.routed_scale), .norm = c.norm_topk };
    }

    /// FP4-Local's experts (NVFP4, csf_32x2_v1 scales; the MTP layer's W4A16 with plain ones) as `rank` holds
    /// them: its rows of gate and up, its columns of down, the block scales of its window decoded into `arena`.
    /// An `NVFP4` layer's activations are quantized with each expert's `input_scale` (W4A4), a `W4A16_NVFP4`
    /// layer's stay bf16, as vLLM's image default runs the checkpoint (spec 114 WS-03); `as`, where given, instead
    /// (a test's A16 copy of a W4A4 layer).
    pub fn nvfp4Experts(arena: Allocator, gpa: Allocator, staging: *experts.Staging, source: anytype, c: *const Config, layer: usize, rank: usize, as: ?experts.Activations) LoadError!experts.Experts {
        if (staging.place and as == null) {
            if (try placedExperts(arena, gpa, staging, source, c, layer, rank)) |routed| return routed;
        }
        if (staging.swap != null) {
            // the source holds half of down's outputs, which only the device's placing reads
            std.log.err("layer {d}: its routed experts are not W4A4 with csf or plain scales, which a load of halved down projections needs", .{layer});
            return error.UnsupportedShape;
        }
        var matrices: [3]experts.Matrix = undefined;
        var reads: Reads(@TypeOf(source)) = .{ .source = source, .c = c, .layer = layer, .rank = rank, .gpa = gpa };
        for (&matrices, &reads.out) |*matrix, *out| {
            out.* = .{ .codes = try arena.alloc([]const u8, c.experts), .scales = try arena.alloc([]const u8, c.experts), .factors = try arena.alloc(f32, c.experts), .inputs = try arena.alloc(f32, c.experts) };
            matrix.* = .{ .codes = out.codes, .block_scales = out.scales, .scales = out.factors, .input_scales = out.inputs };
        }
        // the cuts and the scales' decoding into each worker's own arena, kept until the experts are uploaded
        const workers = @min(c.experts, max_readers, std.Thread.getCpuCount() catch 1);
        const arenas = try staging.arenasOf(gpa, workers);
        var threads: [max_readers]std.Thread = undefined;
        var spawned: usize = 0;
        for (1..workers) |w| {
            threads[spawned] = std.Thread.spawn(.{}, Reads(@TypeOf(source)).part, .{ &reads, arenas[w].allocator(), w, workers }) catch break;
            spawned += 1;
        }
        // the shares no thread took, this thread's own included
        reads.part(arenas[0].allocator(), 0, workers);
        for (spawned + 1..workers) |w| reads.part(arenas[w].allocator(), w, workers);
        for (threads[0..spawned]) |thread| thread.join();
        if (reads.failed.load(.monotonic)) |err| return err;
        const activations: experts.Activations = if (reads.a16.load(.monotonic)) .bf16 else .nvfp4;
        if (activations == .nvfp4 and reads.bare.load(.monotonic)) {
            // the layer's NVFP4 experts quantize their activations with it, unless one of them is W4A16
            std.log.err("layer {d}: an NVFP4 expert stores no input_scale; zspark quantizes its activations with it", .{layer});
            return error.UnsupportedShape;
        }
        var routed = experts.Experts.initWith(gpa, staging, c.experts, c.moe_width / layout.world, c.hidden, matrices[0], matrices[1], matrices[2], as orelse activations) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.UnsupportedShape => return error.UnsupportedShape,
            else => return error.CudaFailure,
        };
        routed.limit = @floatCast(c.swiglu_limit);
        return routed;
    }

    /// `nvfp4Experts` where `staging.place` is set and the layer is W4A4: the device decodes the csf scales and packs
    /// the blocks from the bytes where the source holds them (`experts.Experts.initPlaced`), the same bytes as the
    /// host's. Null where the layer is another case (a W4A16 expert, an expert without its input_scale, scales
    /// stored two ways), which the host reads.
    fn placedExperts(arena: Allocator, gpa: Allocator, staging: *experts.Staging, source: anytype, c: *const Config, layer: usize, rank: usize) LoadError!?experts.Experts {
        const Placed = experts.Experts.Placed;
        var matrices: [3]Placed = undefined;
        for (&matrices, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |*matrix, leaf| {
            const codes = try arena.alloc(u64, c.experts);
            const factors = try arena.alloc(f32, c.experts);
            const inputs = try arena.alloc(f32, c.experts);
            var plain: []u64 = &.{};
            var planes: [][6]u64 = &.{};
            var code_stride: usize = 0;
            var scale_stride: usize = 0;
            var plane: [2]usize = undefined;
            var window: [4]usize = undefined;
            for (0..c.experts) |e| {
                var name: [256]u8 = undefined;
                const module = std.fmt.bufPrint(&name, layout.prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ layer, e, leaf }) catch return error.MissingTensor;
                const stored = try share.modeloptStored(arena, source, c, module);
                const linear = stored.linear;
                if (linear.scheme != .nvfp4) {
                    std.log.err("{s} is stored {t}; zspark reads routed experts NVFP4", .{ module, linear.scheme });
                    return error.UnsupportedShape;
                }
                if (c.quant.find(module)) |declared| if (declared.a16) return null;
                const input = linear.input_scale orelse return null;
                inputs[e] = @bitCast(std.mem.readInt(u32, input.data[0..4], .little));
                factors[e] = @bitCast(std.mem.readInt(u32, linear.scale_2.?.data[0..4], .little));
                const rows: usize = @intCast(linear.rows);
                const inputs_n: usize = @intCast(linear.cols);
                code_stride = inputs_n / 2;
                // the rank's [n, k]: its rows of gate and up, its columns of down, where they lie
                const row: usize, const col: usize = switch (stored.split) {
                    .row => if (rows % layout.world == 0) .{ rank * rows / layout.world, 0 } else return null,
                    .col => if (inputs_n % (16 * layout.world) == 0) .{ 0, rank * inputs_n / layout.world } else return null,
                    else => return null,
                };
                const n = if (stored.split == .row) rows / layout.world else rows;
                const k = if (stored.split == .col) inputs_n / layout.world else inputs_n;
                // under a swap, down's outputs of this rank's half alone (`experts.Swap`)
                const first = if (staging.swap != null and stored.split == .col) rank * rows / layout.world else row;
                codes[e] = @intFromPtr(linear.weight.data.ptr) + first * code_stride + col / 2;
                if (linear.csf) |csf| {
                    if (e == 0) planes = try arena.alloc([6]u64, c.experts) else if (planes.len == 0) return null;
                    const s = csf.streams;
                    if (csf.shape.rows % 128 != 0 or csf.shape.cols % 4 != 0 or s.exceptions.len % 2 != 0) return null;
                    plane = .{ @intCast(csf.shape.rows), @intCast(csf.shape.cols) };
                    window = .{ row, n, col / 16, k / 16 };
                    planes[e] = .{ @intFromPtr(s.bases.ptr), @intFromPtr(s.codes.ptr), @intFromPtr(s.page_bases.ptr), @intFromPtr(s.offsets.ptr), @intFromPtr(s.exceptions.ptr), s.exceptions.len / 2 };
                } else {
                    if (e == 0) plain = try arena.alloc(u64, c.experts) else if (plain.len == 0) return null;
                    scale_stride = inputs_n / 16;
                    plain[e] = @intFromPtr(linear.scale.?.data.ptr) + row * scale_stride + col / 16;
                }
            }
            matrix.* = .{ .codes = codes, .code_stride = code_stride, .factors = factors, .input_scales = inputs, .scales = if (planes.len != 0)
                .{ .csf = .{ .planes = planes, .rows = plane[0], .cols = plane[1], .window = window } }
            else
                .{ .plain = .{ .at = plain, .stride = scale_stride } } };
        }
        var routed = experts.Experts.initPlaced(gpa, staging, layer, c.experts, c.moe_width / layout.world, c.hidden, matrices[0], matrices[1], matrices[2], .nvfp4) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.UnsupportedShape => return error.UnsupportedShape,
            else => return error.CudaFailure,
        };
        routed.limit = @floatCast(c.swiglu_limit);
        return routed;
    }

    /// The threads that read a layer's experts at most: each expert's cut (down's columns copied) and block
    /// scales (csf decoded byte by byte) are its own, and on one thread they were most of the load (16.5 s of
    /// 46.6 on a rank's share of FP4-Local in a safe build, 2026-10-20).
    const max_readers = 16;

    /// `nvfp4Experts`'s reading of experts `w`, `w + of`, ... of the three projections, on any thread: the bytes
    /// are each expert's alone, the same whichever thread reads them.
    fn Reads(comptime Source: type) type {
        return struct {
            source: Source,
            c: *const Config,
            layer: usize,
            rank: usize,
            gpa: Allocator,
            out: [3]Out = undefined,
            /// a `W4A16_NVFP4` expert: the layer's activations stay bf16
            a16: std.atomic.Value(bool) = .init(false),
            /// an expert neither W4A16 nor storing an input_scale
            bare: std.atomic.Value(bool) = .init(false),
            failed: Failed = .{},

            const Out = struct { codes: [][]const u8, scales: [][]const u8, factors: []f32, inputs: []f32 };
            const leaves = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

            fn part(self: *@This(), arena: Allocator, w: usize, of: usize) void {
                var e = w;
                while (e < self.c.experts) : (e += of) {
                    for (leaves, &self.out) |leaf, out| self.one(arena, leaf, out, e) catch |err| return self.failed.store(err);
                    if (self.failed.load(.monotonic) != null) return;
                }
            }

            fn one(self: *@This(), arena: Allocator, leaf: []const u8, out: Out, e: usize) LoadError!void {
                var name: [256]u8 = undefined;
                const module = std.fmt.bufPrint(&name, layout.prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ self.layer, e, leaf }) catch return error.MissingTensor;
                const linear = try share.modeloptOf(arena, self.source, self.c, module, self.rank);
                if (linear.scheme != .nvfp4) {
                    std.log.err("{s} is stored {t}; zspark reads routed experts NVFP4", .{ module, linear.scheme });
                    return error.UnsupportedShape;
                }
                var a16 = false;
                if (self.c.quant.find(module)) |declared| a16 = declared.a16;
                if (a16) self.a16.store(true, .monotonic);
                out.inputs[e] = 0;
                if (linear.input_scale) |stored| {
                    out.inputs[e] = @bitCast(std.mem.readInt(u32, stored.data[0..4], .little));
                } else if (!a16) {
                    self.bare.store(true, .monotonic);
                }
                out.codes[e] = linear.weight.data;
                out.scales[e] = linear.blockScales(arena) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.InvalidCsf => {
                        std.log.err("{s}: its csf_32x2_v1 scales do not decode", .{module});
                        return error.UnsupportedShape;
                    },
                };
                out.factors[e] = @bitCast(std.mem.readInt(u32, linear.scale_2.?.data[0..4], .little));
            }
        };
    }

    /// The first error of the threads that read experts.
    const Failed = struct {
        code: std.atomic.Value(u8) = .init(0),

        const errors = [_]LoadError{ error.OutOfMemory, error.MissingTensor, error.UnsupportedShape };

        fn store(self: *Failed, err: LoadError) void {
            const code: u8 = for (errors, 1..) |one, i| {
                if (one == err) break @intCast(i);
            } else 3;
            _ = self.code.cmpxchgStrong(0, code, .monotonic, .monotonic);
        }

        fn load(self: *Failed, comptime order: std.builtin.AtomicOrder) ?LoadError {
            const code = self.code.load(order);
            return if (code == 0) null else errors[code - 1];
        }
    };

    fn missing(name: []const u8) LoadError {
        std.log.err("{s}: not in the checkpoint", .{name});
        return error.MissingTensor;
    }

    pub fn deinit(self: *Moe) void {
        self.router.deinit();
        self.bias.deinit();
        self.routed.deinit();
        self.shared.deinit();
        self.* = undefined;
    }

    /// Scratch for calls of up to `rows` rows; `picks` and `weights` are also what a test reads.
    pub const Buffers = struct {
        rows: usize,
        slots: usize,
        /// fp32 [rows, experts]
        logits: cuda.Buffer,
        /// int32 and fp32 [rows, top + 1]: the routed picks, then the shared expert (id `experts`, weight 1)
        picks: cuda.Buffer,
        weights: cuda.Buffer,
        /// fp32 [rows * (top + 1), dims]: the routed slots; a row's last slot is the shared expert's, unwritten
        y: cuda.Buffer,
        /// fp32 [rows, dims]: the shared expert's rows
        shared_y: cuda.Buffer,
        shared: Dense.Buffers,
        /// The routed experts' own: the plan over `experts + 1` ids (the shared expert's skipped) and bf16
        /// [rows * (top + 1), width], a pair's SwiGLU; W4A4's a pair's rows quantized for gate and up and for down.
        plan: ?experts.Plan = null,
        act: ?cuda.Buffer = null,
        quantized: ?experts.Quantized = null,

        pub fn init(weights: Moe, rows: usize) cuda.Error!Buffers {
            const slots = weights.top + 1;
            const dims = weights.routed.dims;
            var self: Buffers = .{ .rows = rows, .slots = slots, .logits = undefined, .picks = undefined, .weights = undefined, .y = undefined, .shared_y = undefined, .shared = undefined };
            self.logits = try .init(.f32, &.{ rows, weights.router.n });
            errdefer self.logits.deinit();
            self.picks = try .init(.i32, &.{ rows, slots });
            errdefer self.picks.deinit();
            self.weights = try .init(.f32, &.{ rows, slots });
            errdefer self.weights.deinit();
            self.y = try .init(.f32, &.{ rows * slots, dims });
            errdefer self.y.deinit();
            self.shared_y = try .init(.f32, &.{ rows, dims });
            errdefer self.shared_y.deinit();
            self.shared = try .init(weights.shared.width, rows);
            errdefer self.shared.deinit();
            const routed = weights.routed;
            self.plan = try .init(rows, slots, routed.count + 1);
            errdefer self.plan.?.deinit();
            // the bf16 SwiGLU whatever these weights' form: the MTP layer's A16 experts share the buffers
            self.act = try .init(.bf16, &.{ rows * slots, routed.width });
            errdefer self.act.?.deinit();
            if (routed.activations == .nvfp4) self.quantized = try .init(rows * slots, routed.dims, routed.width);
            return self;
        }

        pub fn deinit(self: *Buffers) void {
            self.logits.deinit();
            self.picks.deinit();
            self.weights.deinit();
            self.y.deinit();
            self.shared_y.deinit();
            self.shared.deinit();
            if (self.plan) |*one| one.deinit();
            if (self.act) |*one| one.deinit();
            if (self.quantized) |*one| one.deinit();
            self.* = undefined;
        }
    };

    /// x bf16 [rows, dims] -> out fp32 or bf16 [rows, dims], the rank's partial: the routed slots times their weights
    /// in slot order, then the shared expert. W4A4 experts (`.nvfp4`) take the rows before `from` A16 and the rows
    /// from it W4A4 (spec 115, the card's forms: `forward.zig` `bf16_rows`), each row's bits its form's alone.
    pub fn forward(self: Moe, buffers: Buffers, x: cuda.View, out: cuda.View, from: usize) cuda.Error!void {
        try self.route(buffers, x, from);
        try self.fill(buffers, x, 0, @intCast(x.size[0]), out);
    }

    /// `forward` to its routed slots, before the shared expert and the sum (`fill`).
    pub fn route(self: Moe, buffers: Buffers, x: cuda.View, from: usize) cuda.Error!void {
        const rows: usize = @intCast(x.size[0]);
        std.debug.assert(rows >= 1 and rows <= buffers.rows);
        const logits = buffers.logits.rows(rows);
        const picks = buffers.picks.rows(rows);
        const weights = buffers.weights.rows(rows);
        // the router's K slices are summed by the selection, in slice order; a weight of one slice has none
        const slices: ?cuda.Slices = if (self.router.part != null) try self.router.slices(x, logits) else null;
        if (slices == null) try self.router.forward(x, logits);
        try moe.select(logits, slices, self.bias.view, picks, weights, self.top, self.scale, self.norm);

        const y = buffers.y.rows(rows * buffers.slots);
        // the shared expert's id is past the routed ones: its slot of y stays as it was and is not read
        const routed = self.routed;
        const plan = buffers.plan.?;
        // the A16 and W4A4 rows' split is the experts' (`Experts.run`), shared with Flash Next
        try routed.run(plan, picks, x, buffers.act.?.rows(rows * buffers.slots), buffers.quantized, y, rows, from, routed.count);
    }

    /// `forward`'s end on the window's rows [first, first + count) alone, after `route` over all of x bf16
    /// [rows, dims]: the shared expert and the slots' sum into those rows of out fp32 or bf16 [rows, dims]. Every launch
    /// works a row at a time, so a row's bits are the whole's (spec 116 WS-03: the rows another rank reads first).
    pub fn fill(self: Moe, buffers: Buffers, x: cuda.View, first: usize, count: usize, out: cuda.View) cuda.Error!void {
        const rows: usize = @intCast(x.size[0]);
        std.debug.assert(count >= 1 and first + count <= rows and (out.dtype == .f32 or out.dtype == .bf16));
        const slots = buffers.slots;
        const shared_y = buffers.shared_y.span(first, count);
        try self.shared.run(buffers.shared, x.span(first, count), shared_y);
        try moe.combine(buffers.y.span(first * slots, count * slots), shared_y, buffers.weights.span(first, count), out.span(first, count));
    }
};
