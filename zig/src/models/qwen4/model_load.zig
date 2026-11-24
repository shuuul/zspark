//! The loader's helpers of `model.zig`: a layer's, the experts' and the hyper-connections' weights read
//! from the layout and uploaded.

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("blocks/attention.zig");
const bf16 = @import("../../weights/bf16.zig");
const cuda = @import("../../cuda/cuda.zig");
const experts = @import("../../weights/experts.zig");
const face = @import("../../weights/face.zig");
const Face = face.Face;
const layout = @import("layout.zig");
const loading = @import("../../weights/loading.zig");
const model = @import("model.zig");
const moe = @import("blocks/moe.zig");
const nvfp4 = @import("../../weights/nvfp4.zig");
const requantize = @import("../../weights/requantize.zig");
const share_file = @import("share.zig");
const share_layout = @import("share_layout.zig");
const safetensors = @import("../../weights/safetensors.zig");

pub fn cutShare(err: share_layout.Error) model.Model.LoadError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.Indivisible => error.UnsupportedShape,
    };
}

/// The tensors under `part` held in memory while they are packed and uploaded.
pub fn hold(gpa: Allocator, found: layout.Layout, part: anytype) Allocator.Error!safetensors.Held {
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(gpa);
    try layout.ranges(gpa, part, &list);
    return found.checkpoint.hold(gpa, list.items);
}

/// A block loader's error as `Model.LoadError`.
pub fn narrow(err: anyerror) model.Model.LoadError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.CudaFailure => error.CudaFailure,
        error.UnsupportedShape => error.UnsupportedShape,
        else => error.UnsupportedScheme,
    };
}

/// A BF16 linear's bytes as stored.
pub fn plain(linear: layout.Linear) error{UnsupportedScheme}![]const u8 {
    if (linear.scheme != .bf16 or linear.weight.dtype != .BF16) return error.UnsupportedScheme;
    return linear.weight.data;
}

/// Linears of one input, in column order, as one face: each BF16 or MXFP8, as stored.
pub fn faceOf(gpa: Allocator, linears: []const layout.Linear, scratch: cuda.Scratch.Rows) model.Model.LoadError!Face {
    var parts: [face.max_parts]face.Part = undefined;
    if (linears.len > parts.len) return error.UnsupportedShape;
    for (linears, parts[0..linears.len]) |linear, *part| {
        if (linear.cols != linears[0].cols) return error.UnsupportedShape;
        part.* = try linear.part();
    }
    return Face.initParts(gpa, parts[0..linears.len], linears[0].cols, scratch) catch |err| narrow(err);
}

/// A vector as fp32, with the + 1 a centred norm weight takes.
fn floats(gpa: Allocator, vector: layout.Vector, centred: bool) Allocator.Error![]f32 {
    const out = try gpa.alloc(f32, vector.len());
    const add: f32 = if (vector.gamma and centred) 1.0 else 0.0;
    for (out, 0..) |*value, i| value.* = vector.value(i) + add;
    return out;
}

/// An RMSNorm's gamma on the device, fp32.
pub fn gamma(gpa: Allocator, vector: layout.Vector, centred: bool) model.Model.LoadError!cuda.Buffer {
    const host = try floats(gpa, vector, centred);
    defer gpa.free(host);
    var buffer: cuda.Buffer = try .init(.f32, &.{host.len});
    errdefer buffer.deinit();
    try buffer.upload(std.mem.sliceAsBytes(host));
    return buffer;
}

pub fn loadHyperConnection(gpa: Allocator, source: layout.HyperConnection, centred: bool, scratch: cuda.Scratch.Rows) model.Model.LoadError!model.HyperConnection {
    const scale = try floats(gpa, source.norm, centred);
    defer gpa.free(scale);
    var down = try faceOf(gpa, if (source.inject) |inject| &.{ source.down, inject } else &.{source.down}, scratch);
    errdefer down.deinit();
    const up = try faceOf(gpa, &.{source.up}, scratch);
    return .fromFaces(down, up, scale, source.inject != null);
}

/// `columns`: the layer was cut by `share_layout.sliceLayerPlaced`, its routed experts' down whole, and the
/// device takes this rank's columns of it (`staging.place`). `activations`: the routed experts' pack, `.nvfp4`
/// for W4A4 rows in a prompt's window (`model.PromptForm.card`), which needs every expert NVFP4 with its
/// `input_scale`.
pub fn loadLayer(gpa: Allocator, staging: *experts.Staging, source: layout.Layer, shape: model.Shape, attention_shape: attention.Shape, centred: bool, scratch: cuda.Scratch.Rows, stored: bool, cut: ?share_file.Share, columns: ?share_file.Share, activations: experts.Activations) model.Model.LoadError!model.Layer {
    var attn_hc = try loadHyperConnection(gpa, source.attn_hc, centred, scratch);
    errdefer attn_hc.deinit();
    var mlp_hc = try loadHyperConnection(gpa, source.mlp_hc, centred, scratch);
    errdefer mlp_hc.deinit();
    var block: model.Block = switch (source.block) {
        .linear_attention => |dn| blk: {
            const a_log = try floats(gpa, dn.a_log, centred);
            defer gpa.free(a_log);
            const dt_bias = try floats(gpa, dn.dt_bias, centred);
            defer gpa.free(dt_bias);
            if (dn.norm.tensor.dtype != .BF16 or dn.conv.tensor.dtype != .BF16) return error.UnsupportedScheme;
            var projection = try faceOf(gpa, &.{ dn.qkv, dn.z, dn.b, dn.a }, scratch);
            errdefer projection.deinit();
            var out = try faceOf(gpa, &.{dn.out}, scratch);
            errdefer out.deinit();
            break :blk .{ .delta_net = model.DeltaNet.fromFaces(projection, out, .{
                .conv = dn.conv.tensor.data,
                .a_log = a_log,
                .dt_bias = dt_bias,
                .norm = dn.norm.tensor.data,
            }, shape) catch |err| return narrow(err) };
        },
        .full_attention => |tensors| .{ .attention = attention.Weights.load(gpa, attention_shape, tensors, centred, scratch) catch |err| return narrow(err) },
    };
    errdefer switch (block) {
        .delta_net => |*dn| dn.deinit(),
        .attention => |*weights| weights.deinit(),
    };
    const weights = try loadMoe(gpa, staging, source.moe, shape, scratch, stored, cut, columns, activations);
    return .{ .attn_hc = attn_hc, .mlp_hc = mlp_hc, .block = block, .moe = weights };
}

/// A layer's MoE. The routed experts reach the device as NVFP4: a main layer's are stored so and go as they are;
/// the MTP layer's go as they are when NVFP4 (FP4-Local's) and are requantized on the host when BF16 stacked, as
/// TensorFold's `moe_nvfp4` does ("draft-only NVFP4 packing").
/// `layout_builder.Site` keeps anything but NVFP4 out of a main layer. `unchanged` (the MTP layer's, under
/// `Options.drafts_stored`): nothing is requantized; NVFP4 experts run as ever, and any other storage is refused.
/// `cut` (the MTP layer's on a rank of two, `share_layout.sliceMtp`): `shape` is the rank's, and experts that are
/// requantized come whole, are quantized whole and then cut to the rank's half of the width, 320 of 640, so
/// both ranks hold codes of one quantization. A departure from TensorFold's order (`weights.py` slices the bf16
/// values and quantizes a rank's half under a scale of its own), as spec 090 names it; either only drafts.
fn loadMoe(gpa: Allocator, staging: *experts.Staging, source: layout.Moe, shape: model.Shape, scratch: cuda.Scratch.Rows, unchanged: bool, cut: ?share_file.Share, columns: ?share_file.Share, activations: experts.Activations) model.Model.LoadError!moe.Weights {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    if (unchanged) try storedAs(source.experts);
    // placed by the device where the loader keeps the bytes for it; the host packs them otherwise
    var placed: ?experts.Experts = switch (source.experts) {
        .each => |each| if (staging.place) try placedExperts(gpa, staging, each, shape, columns, activations) else null,
        .stacked => null,
    };
    errdefer if (placed) |*one| one.deinit();
    if (placed == null and activations == .nvfp4) {
        // the host's pack is A16's: W4A4 rows are the device's placing (the load's way for every main layer)
        std.log.err("the routed experts take W4A4 rows only where the device places them", .{});
        return error.UnsupportedScheme;
    }
    if (placed == null and columns != null) return error.UnsupportedScheme;
    const Pick = enum { gate, up, down };
    var matrices: [3]experts.Matrix = undefined;
    var quantized: [3]?requantize.Quantized = @splat(null);
    defer for (&quantized) |*slot| if (slot.*) |*owned| owned.deinit(gpa);
    const count = switch (source.experts) {
        .each => |each| each.len,
        .stacked => shape.routed,
    };
    // what is requantized is whole, on a rank of two as well
    const width = if (cut != null) shape.width * share_file.world else shape.width;
    // csf scales decoded here, a layer's at a time, into the layer's arena
    const decoding = loading.now();
    const decoded: ?[3][]const []const u8 = switch (source.experts) {
        .each => |each| if (placed == null) try blockScalesOf(arena, each) else null,
        .stacked => null,
    };
    loading.since(&loading.host.csf_ns, decoding);
    if (placed == null) for (&matrices, &quantized, [_]Pick{ .gate, .up, .down }, 0..) |*matrix, *owned, pick, p| {
        const n = if (pick == .down) shape.hidden else width;
        const k = if (pick == .down) width else shape.hidden;
        const weights = try arena.alloc(requantize.Weight, count);
        // experts stored as NVFP4
        var stored: usize = 0;
        switch (source.experts) {
            .each => |each| {
                const codes = try arena.alloc([]const u8, count);
                const block_scales = try arena.alloc([]const u8, count);
                const scales = try arena.alloc(f32, count);
                for (each, 0..) |expert, e| {
                    const linear = switch (pick) {
                        .gate => expert.gate,
                        .up => expert.up,
                        .down => expert.down,
                    };
                    switch (linear.scheme) {
                        .nvfp4 => {
                            codes[e] = linear.weight.data;
                            block_scales[e] = decoded.?[p][e];
                            const scale_2 = linear.scale_2.?;
                            if (scale_2.dtype != .F32 or scale_2.data.len != 4) return error.UnsupportedScheme;
                            scales[e] = @bitCast(std.mem.readInt(u32, scale_2.data[0..4], .little));
                            stored += 1;
                        },
                        .bf16 => weights[e] = .{ .bf16 = try plain(linear) },
                        // the layout keeps MXFP8 out of experts
                        .mxfp8 => return error.UnsupportedScheme,
                    }
                }
                matrix.* = .{ .codes = codes, .block_scales = block_scales, .scales = scales };
            },
            .stacked => |tensors| {
                // gate_up [E, 2 width, hidden]: an expert's gate rows, then its up rows; down [E, hidden, width]
                const tensor = if (pick == .down) tensors.down else tensors.gate_up;
                const each = tensor.data.len / count;
                const size = n * k * 2;
                if (tensor.dtype != .BF16 or each != (if (pick == .down) size else 2 * size)) return error.UnsupportedScheme;
                for (weights, 0..) |*weight, e| weight.* = .{ .bf16 = tensor.data[e * each + (if (pick == .up) size else 0) ..][0..size] };
            },
        }
        // a projection's experts are of one storage
        if (stored == count) continue;
        if (stored != 0) return error.UnsupportedScheme;
        owned.* = requantize.Quantized.init(gpa, weights, n, k) catch |err| return narrow(err);
        matrix.* = owned.*.?.matrix();
        if (cut) |one| matrix.* = if (pick == .down) try matrix.colsOf(arena, n, k, one.rank, share_file.world) else try matrix.rowsOf(arena, n, one.rank, share_file.world);
    };

    var router: bf16.Linear = try .initStacked(&.{ try plain(source.router), try plain(source.shared_gate) }, shape.hidden, scratch);
    errdefer router.deinit();
    var routed: moe.Weights.Routed = if (placed) |one| .{ .nvfp4 = one } else .{ .nvfp4 = experts.Experts.init(gpa, staging, count, shape.width, shape.hidden, matrices[0], matrices[1], matrices[2]) catch |err| switch (err) {
        error.UnsupportedShape => return error.UnsupportedShape,
        error.OutOfMemory => return error.OutOfMemory,
        error.CudaFailure => return error.CudaFailure,
    } };
    placed = null;
    errdefer routed.deinit();
    var gate_up = try faceOf(gpa, &.{ source.shared.gate, source.shared.up }, scratch);
    errdefer gate_up.deinit();
    const down = try faceOf(gpa, &.{source.shared.down}, scratch);
    return .{ .router = router, .stored = .{ .nvfp4 = .{ .routed = routed, .shared_gate_up = gate_up, .shared_down = down } } };
}

/// The routed experts placed by the device from where a part's bytes are staged (`Experts.initPlaced`: the bytes
/// `Experts.init` packs on the host): each expert's codes and e4m3 block scales where they lie, csf's decoded
/// into `staging.keep`'s memory. `columns`: down whole (`share_layout.sliceLayerPlaced`), this rank's inputs read
/// a row's stride apart. Null where an expert is not NVFP4.
fn placedExperts(gpa: Allocator, staging: *experts.Staging, each: []const layout.Expert, shape: model.Shape, columns: ?share_file.Share, activations: experts.Activations) model.Model.LoadError!?experts.Experts {
    for (each) |expert| for ([_]layout.Linear{ expert.gate, expert.up, expert.down }) |linear| if (linear.scheme != .nvfp4) return null;
    const count = each.len;
    // a swap trades down's columns: a rank's, the one it holds
    if (staging.swap) |one| if (columns == null or columns.?.rank != one.rank) return error.UnsupportedShape;
    const kept = staging.keep(gpa);
    const cut = try kept.alloc(layout.Expert, count);
    for (each, cut) |expert, *dest| {
        dest.* = expert;
        if (columns) |share| dest.down = share_layout.columnsOf(expert.down, share) catch |err| return cutShare(err);
    }
    const decoding = loading.now();
    const scales = try blockScalesOf(kept, cut);
    loading.since(&loading.host.csf_ns, decoding);
    var matrices: [3]experts.Experts.Placed = undefined;
    for (&matrices, 0..) |*matrix, p| {
        const codes = try kept.alloc(u64, count);
        const at = try kept.alloc(u64, count);
        const factors = try kept.alloc(f32, count);
        const inputs: []f32 = if (activations == .nvfp4) try kept.alloc(f32, count) else &.{};
        var code_stride: ?usize = null;
        var scale_stride: ?usize = null;
        for (cut, 0..) |expert, e| {
            const linear = switch (p) {
                0 => expert.gate,
                1 => expert.up,
                else => expert.down,
            };
            // a row as stored, and where this rank's inputs begin in it: down's columns where `columns` cuts it
            const of: usize = if (p == 2 and columns != null) share_file.world else 1;
            const stored: usize = linear.cols * of;
            const first: usize = if (p == 2) if (columns) |share| share.rank * linear.cols else 0 else 0;
            if (linear.cols % 32 != 0 or linear.weight.data.len != linear.rows * stored / 2) return error.UnsupportedShape;
            // under a swap down's codes are read from the rank's half of the outputs on (`experts.Swap`)
            const swapped: usize = if (p == 2 and staging.swap != null) staging.swap.?.rank * (linear.rows / share_file.world) * (stored / 2) else 0;
            codes[e] = @intFromPtr(linear.weight.data.ptr) + first / 2 + swapped;
            // stored scales are the whole row's; csf's decoded are the rank's window
            const as_stored = linear.scale != null;
            at[e] = @intFromPtr(scales[p][e].ptr) + if (as_stored) first / 16 else 0;
            const rows_apart = if (as_stored) stored / 16 else linear.cols / 16;
            if ((code_stride orelse stored / 2) != stored / 2 or (scale_stride orelse rows_apart) != rows_apart) return error.UnsupportedShape;
            code_stride = stored / 2;
            scale_stride = rows_apart;
            const scale_2 = linear.scale_2 orelse return error.UnsupportedScheme;
            if (scale_2.dtype != .F32 or scale_2.data.len != 4) return error.UnsupportedScheme;
            factors[e] = @bitCast(std.mem.readInt(u32, scale_2.data[0..4], .little));
            if (activations == .nvfp4) {
                // the card's A4 rows quantize with it (`Experts.finish`: gate and up's the layer's largest)
                const input = linear.input_scale orelse {
                    std.log.err("{s} stores no input_scale; W4A4 rows quantize with it", .{linear.name});
                    return error.UnsupportedScheme;
                };
                inputs[e] = @bitCast(std.mem.readInt(u32, input.data[0..4], .little));
            }
        }
        matrix.* = .{ .codes = codes, .code_stride = code_stride.?, .scales = .{ .plain = .{ .at = at, .stride = scale_stride.? } }, .factors = factors, .input_scales = inputs };
    }
    return experts.Experts.initPlaced(gpa, staging, 0, count, shape.width, shape.hidden, matrices[0], matrices[1], matrices[2], activations) catch |err| switch (err) {
        error.UnsupportedShape => return error.UnsupportedShape,
        error.OutOfMemory => return error.OutOfMemory,
        error.CudaFailure => return error.CudaFailure,
    };
}

/// Each NVFP4 expert's e4m3 block scales, gate's, up's and down's by expert (`Linear.blockScales`): the stored
/// bytes, or csf's decoded into `arena` on the host's cores, an expert's matrix a task (7 GiB a load: 2.9 s on
/// one core). Other storages' entries are empty.
fn blockScalesOf(arena: Allocator, each: []const layout.Expert) model.Model.LoadError![3][]const []const u8 {
    var out: [3][][]const u8 = undefined;
    const Task = struct { linear: layout.Linear, into: []u8 };
    var tasks: std.ArrayList(Task) = .empty;
    for (&out, 0..) |*by_expert, p| {
        by_expert.* = try arena.alloc([]const u8, each.len);
        for (each, by_expert.*) |expert, *scales| {
            const linear = switch (p) {
                0 => expert.gate,
                1 => expert.up,
                else => expert.down,
            };
            scales.* = &.{};
            if (linear.scheme != .nvfp4) continue;
            if (linear.scale) |stored| {
                scales.* = stored.data;
                continue;
            }
            const into = try arena.alloc(u8, linear.rows * (linear.cols / 16));
            scales.* = into;
            try tasks.append(arena, .{ .linear = linear, .into = into });
        }
    }
    const Share = struct {
        fn run(all: []const Task, first: usize, stride: usize, failed: *std.atomic.Value(usize)) void {
            var i = first;
            while (i < all.len) : (i += stride) {
                const task = all[i];
                task.linear.csf.?.decode(task.linear.rows, task.linear.cols, task.into) catch {
                    _ = failed.cmpxchgStrong(all.len, i, .monotonic, .monotonic);
                };
            }
        }
    };
    if (tasks.items.len == 0) return .{ out[0], out[1], out[2] };
    var failed: std.atomic.Value(usize) = .init(tasks.items.len);
    const workers = @min(tasks.items.len, @min(std.Thread.getCpuCount() catch 1, 32));
    var threads: [32]std.Thread = undefined;
    var spawned: usize = 0;
    for (1..workers) |w| {
        threads[spawned] = std.Thread.spawn(.{}, Share.run, .{ tasks.items, w, workers, &failed }) catch break;
        spawned += 1;
    }
    // the shares no thread took, this thread's own included
    Share.run(tasks.items, 0, workers, &failed);
    for (spawned + 1..workers) |w| Share.run(tasks.items, w, workers, &failed);
    for (threads[0..spawned]) |thread| thread.join();
    const bad = failed.load(.monotonic);
    if (bad < tasks.items.len) {
        std.log.err("{s}: its csf block scales do not decode (a base above 240 or an exception past the matrix)", .{tasks.items[bad].linear.name});
        return error.UnsupportedScheme;
    }
    return .{ out[0], out[1], out[2] };
}

/// Under `Options.drafts_stored`: nothing where every expert is NVFP4 (run as stored already), and a refusal of
/// anything else, which only a requantization would run.
fn storedAs(experts_of: layout.Experts) model.Model.LoadError!void {
    const each = switch (experts_of) {
        .each => |each| each,
        .stacked => |tensors| {
            std.log.err("the MTP layer's experts are one {t} tensor; drafts as stored (ZSPARK_DRAFTS_STORED) read NVFP4 experts, and nothing is requantized", .{tensors.gate_up.dtype});
            return error.UnsupportedScheme;
        },
    };
    for (each) |expert| for ([_]layout.Linear{ expert.gate, expert.up, expert.down }) |linear| if (linear.scheme != .nvfp4) {
        std.log.err("{s}: stored as {t}; drafts as stored (ZSPARK_DRAFTS_STORED) read NVFP4 experts, and nothing is requantized", .{ linear.name, linear.scheme });
        return error.UnsupportedScheme;
    };
}

/// Whether the MTP head's routed experts are placed by the device as a main layer's (`placedExperts`): every one
/// NVFP4. Others are requantized or run as stored, from whole tensors.
pub fn placesMtp(head: layout.Mtp) bool {
    const each = switch (head.layer.moe.experts) {
        .each => |each| each,
        .stacked => return false,
    };
    for (each) |expert| for ([_]layout.Linear{ expert.gate, expert.up, expert.down }) |linear| if (linear.scheme != .nvfp4) return false;
    return each.len > 0;
}

/// A load's parts read a part ahead into two kept buffers (`loading.Ahead`), the loaders taking each part's
/// tensors where they are staged (`layout.moved`): the embedding, then each layer (its n-gram table left in the
/// mapping), then the mixer and the head. On a rank of two a layer's routed experts' gate and up are read by the
/// rank's rows alone, and down by its half of the outputs where `swapping`; every other tensor whole, as the
/// share's cuts read it.
pub const Reading = struct {
    gpa: Allocator,
    found: layout.Layout,
    layers: usize,
    share: ?share_file.Share,
    /// On a rank of two: each reads half of a routed layer's down projections, its half of the outputs, and trades
    /// the inputs the peer keeps of them for its own of the peer's half (`experts.Swap`, as GLM's two ranks). Set
    /// before the first part is read, and `swap` before the load.
    swapping: bool = false,
    /// The trade `swapping` asks for: the ranks' communicator, once they have joined.
    swap: ?experts.Swap = null,
    /// Whether the load reads the MTP head, a part after the head (`Model.Options.mtp` where the layout has one):
    /// whole, or each rank its half (`Options.head_split`). Set before the first part is read.
    mtp: enum { none, whole, split } = .none,
    ahead: loading.Ahead(Reading) = .{},
    /// What `moved` copies, kept until the part after next.
    arenas: [2]std.heap.ArenaAllocator,
    turn: usize = 0,

    pub fn init(gpa: Allocator, found: layout.Layout, layers: usize, share: ?share_file.Share) Reading {
        return .{ .gpa = gpa, .found = found, .layers = layers, .share = share, .arenas = .{ .init(gpa), .init(gpa) } };
    }

    pub fn deinit(self: *Reading) void {
        self.ahead.deinit(self.gpa, self);
        for (&self.arenas) |*arena| arena.deinit();
    }

    /// Reads the first two parts on a thread of their own before the load asks for them (while the ranks join);
    /// the Reading stays where it is until it is freed.
    pub fn prefetch(self: *Reading) void {
        self.ahead.prefetch(self, embed);
    }

    pub const embed = 0;

    pub fn layer(index: usize) usize {
        return 1 + index;
    }

    /// The mixer and the head.
    pub fn top(self: *const Reading) usize {
        return 1 + self.layers;
    }

    /// The MTP head's part, after `top`, where the load reads one.
    pub fn mtpPart(self: *const Reading) usize {
        std.debug.assert(self.mtp != .none);
        return self.top() + 1;
    }

    pub fn after(self: *const Reading, part: usize) ?usize {
        const last = if (self.mtp != .none) self.top() + 1 else self.top();
        return if (part < last) part + 1 else null;
    }

    pub fn fill(self: *const Reading, slot: *loading.Slot, part: usize) anyerror!void {
        var list: std.ArrayList([]const u8) = .empty;
        defer list.deinit(self.gpa);
        var tensors: std.ArrayList([]const u8) = .empty;
        defer tensors.deinit(self.gpa);
        if (part == embed) {
            try layout.ranges(self.gpa, self.found.embed, &list);
        } else if (part == self.top()) {
            try layout.ranges(self.gpa, self.found.mixer, &list);
            try layout.ranges(self.gpa, self.found.head, &list);
        } else if (part == self.top() + 1) {
            // the MTP head: its layer's experts by the rank's rows where each rank holds half of it
            const head = self.found.mtp.?;
            inline for (comptime std.meta.fieldNames(layout.Mtp)) |name| {
                if (comptime !std.mem.eql(u8, name, "layer")) try layout.ranges(self.gpa, @field(head, name), &list);
            }
            try tensors.appendSlice(self.gpa, list.items);
            try self.layerRanges(head.layer, self.mtp == .split and placesMtp(head), false, &list, &tensors);
        } else if (self.found.layers[part - 1]) |whole| {
            try self.layerRanges(whole, self.share != null, self.swapping, &list, &tensors);
        }
        if (tensors.items.len < list.items.len) try tensors.appendSlice(self.gpa, list.items[tensors.items.len..]);
        try slot.stage.read(self.gpa, self.found.checkpoint, list.items);
        try slot.shift(self.gpa, list.items, tensors.items);
    }

    /// A layer's ranges into `list` and the whole tensors they are of into `tensors` (beside each other from the
    /// first), its n-gram table left out: every tensor whole, but where `halves` its routed experts' gate and up
    /// by the rank's rows, and where `swapping` down by its half of the outputs too.
    fn layerRanges(self: *const Reading, whole: layout.Layer, halves: bool, swapping: bool, list: *std.ArrayList([]const u8), tensors: *std.ArrayList([]const u8)) Allocator.Error!void {
        var reading = whole;
        if (reading.ple) |*branch| branch.table.shards = &.{};
        const each: []const layout.Expert = if (halves and reading.moe.experts == .each) reading.moe.experts.each else &.{};
        if (each.len > 0) reading.moe.experts = .{ .each = &.{} };
        const first = list.items.len;
        try layout.ranges(self.gpa, reading, list);
        try tensors.appendSlice(self.gpa, list.items[first..]);
        for (each) |expert| {
            // gate and up by the rank's rows (their codes; csf's streams are decoded for its window, whole); down
            // whole, or under a swap by the rank's half of its outputs, every input of them (`experts.Swap`)
            for ([_]layout.Linear{ expert.gate, expert.up, expert.down }, 0..) |linear, p| {
                const before = list.items.len;
                if (p == 2 and !swapping) {
                    try layout.ranges(self.gpa, linear, list);
                    try tensors.appendSlice(self.gpa, list.items[before..]);
                    continue;
                }
                var rest = linear;
                rest.weight.data = &.{};
                try layout.ranges(self.gpa, rest, list);
                try tensors.appendSlice(self.gpa, list.items[before..]);
                const data = linear.weight.data;
                const half = data.len / share_file.world;
                try list.append(self.gpa, data[self.share.?.rank * half ..][0..half]);
                try tensors.append(self.gpa, data);
            }
        }
    }

    /// `part`'s slot, read; the next part's read begun. What `moved` copies goes into the arena returned, kept
    /// until two parts on.
    pub fn begin(self: *Reading, part: usize) model.Model.LoadError!Allocator {
        _ = self.ahead.begin(self, part) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.CudaFailure => error.CudaFailure,
            else => error.MissingTensors,
        };
        self.turn = 1 - self.turn;
        _ = self.arenas[self.turn].reset(.retain_capacity);
        return self.arenas[self.turn].allocator();
    }

    /// `value`'s tensors where the current part staged them.
    pub fn moved(self: *const Reading, arena: Allocator, value: anytype) Allocator.Error!@TypeOf(value) {
        return layout.moved(arena, value, self.ahead.current.?);
    }

    /// The launches that read the current part are recorded; its slot is written over after them.
    pub fn end(self: *Reading) void {
        self.ahead.end();
    }
};
