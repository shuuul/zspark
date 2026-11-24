//! How `layout.zig`'s `Layout.init` resolves and checks a checkpoint's tensors (spec 104 WS-06 moved it here):
//! `Builder` finds each tensor by name, checks its dtype, shape and storage against the config and the site it
//! sits at (`Site`), and records what is missing.

const std = @import("std");
const Allocator = std.mem.Allocator;
const qwen4_config = @import("config.zig");
const Config = qwen4_config.Config;
const Diagnostic = qwen4_config.Diagnostic;
const Ngram = qwen4_config.Ngram;
const csf = @import("../../weights/csf.zig");
const nvfp4 = @import("../../weights/nvfp4.zig");
const safetensors = @import("../../weights/safetensors.zig");
const Checkpoint = safetensors.Checkpoint;
const Tensor = safetensors.Tensor;
const layout = @import("layout.zig");
const Attention = layout.Attention;
const Conv = layout.Conv;
const DeltaNet = layout.DeltaNet;
const Error = layout.Error;
const Expert = layout.Expert;
const Experts = layout.Experts;
const HyperConnection = layout.HyperConnection;
const Layer = layout.Layer;
const Linear = layout.Linear;
const Moe = layout.Moe;
const Mtp = layout.Mtp;
const Ple = layout.Ple;
const Scheme = layout.Scheme;
const Stats = layout.Stats;
const Table = layout.Table;
const Vector = layout.Vector;
const mxfp8_block = layout.mxfp8_block;
const floatAt = layout.floatAt;
const centredFromMeans = layout.centredFromMeans;

fn shaped(tensor: Tensor, dims: []const u64) bool {
    return std.mem.eql(u64, tensor.shape, dims);
}

/// A leaf standing in for a missing tensor inside a part that is then dropped.
const absent: Tensor = .{ .dtype = .U8, .shape = &.{}, .data = &.{} };

/// Where a linear sits, which decides the schemes it may take: those TensorFold's loader reads there.
pub const Site = enum {
    /// Read as bf16 rows alone.
    plain,
    /// The DeltaNet and attention projections, the shared expert and the head: bf16 or MXFP8.
    dense,
    /// Decoder experts: NVFP4 alone.
    routed,
    /// The MTP layer's experts: NVFP4 alone (the retired NVFP4 checkpoint's FP8 ones are no longer read).
    drafter,

    fn allowed(self: Site) std.enums.EnumSet(Scheme) {
        return switch (self) {
            .plain => .initOne(.bf16),
            .dense => .initMany(&.{ .bf16, .mxfp8 }),
            .routed, .drafter => .initOne(.nvfp4),
        };
    }

    /// `allowed` in the words a refusal says.
    fn reads(self: Site) []const u8 {
        return switch (self) {
            .plain => "bf16 only",
            .dense => "bf16 or MXFP8",
            .routed => "routed experts are NVFP4",
            .drafter => "MTP experts are NVFP4",
        };
    }
};

pub const Builder = struct {
    arena: Allocator,
    config: *const Config,
    checkpoint: *const Checkpoint,
    diagnostic: *Diagnostic,
    missing: std.ArrayList([]const u8) = .empty,
    stats: Stats = .{},

    pub fn fmt(b: *Builder, comptime format: []const u8, args: anytype) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(b.arena, format, args);
    }

    fn fail(b: *Builder, err: Error, comptime format: []const u8, args: anytype) Error {
        b.diagnostic.set(b.arena, format, args);
        return err;
    }

    pub fn anyStartsWith(b: *Builder, prefix: []const u8) bool {
        for (b.checkpoint.tensors.keys()) |name| if (std.mem.startsWith(u8, name, prefix)) return true;
        return false;
    }

    /// The tensor, or a stand-in with its name recorded as missing.
    fn need(b: *Builder, name: []const u8) Allocator.Error!?Tensor {
        if (b.checkpoint.get(name)) |tensor| return tensor;
        try b.missing.append(b.arena, name);
        return null;
    }

    /// `built` if none of its tensors was missing, else null. Evaluated by the caller between two reads
    /// of the missing count, so parts nest.
    pub fn part(b: *Builder, comptime T: type, built: Error!Checked(T)) Error!?T {
        const checked = try built;
        return if (b.missing.items.len == checked.mark) checked.value else null;
    }

    fn Checked(comptime T: type) type {
        return struct { value: T, mark: usize };
    }

    fn mismatch(b: *Builder, name: []const u8, tensor: Tensor, comptime expected: []const u8, args: anytype) Error {
        const want = b.fmt(expected, args) catch return error.OutOfMemory;
        return b.fail(error.InvalidCheckpoint, "{s}: {t} {any}, expected {s}", .{ name, tensor.dtype, tensor.shape, want });
    }

    pub fn linear(b: *Builder, name: []const u8, rows: u32, cols: u32, site: Site) Error!Checked(Linear) {
        const mark = b.missing.items.len;
        var out: Linear = .{ .name = name, .scheme = .bf16, .rows = rows, .cols = cols, .weight = absent };
        const weight_name = try b.fmt("{s}.weight", .{name});
        const weight = (try b.need(weight_name)) orelse return .{ .value = out, .mark = mark };
        out.weight = weight;

        const scale_name = try b.fmt("{s}.weight_scale", .{name});
        const scale = b.checkpoint.get(scale_name);
        if (scale == null and weight.dtype == .U8) {
            const declared = if (b.config.quant.find(name)) |declared_as| declared_as.scale_encoding else .plain;
            const fixed_name = try b.fmt("{s}{s}", .{ scale_name, csf.fixed_suffix });
            const exceptions_name = try b.fmt("{s}{s}", .{ scale_name, csf.exceptions_suffix });
            if (declared == .csf or b.checkpoint.get(fixed_name) != null or b.checkpoint.get(exceptions_name) != null)
                return b.csfLinear(out, mark, site, fixed_name, exceptions_name);
        }
        if (scale != null) if (b.config.quant.find(name)) |declared_as| if (declared_as.scale_encoding == .csf)
            return b.fail(error.InvalidCheckpoint, "{s}: plain block scales where the config declares csf", .{scale_name});
        // a scale may sit in a shard still downloading: the weight's dtype says which to wait for
        if (scale == null and weight.dtype != .BF16) {
            switch (weight.dtype) {
                .U8 => {
                    try b.missing.append(b.arena, scale_name);
                    _ = try b.need(try b.fmt("{s}.weight_scale_2", .{name}));
                },
                .F8_E4M3 => try b.missing.append(b.arena, scale_name),
                else => return b.fail(error.UnsupportedScheme, "{s}: stored as {t}; zspark reads bf16, MXFP8 and NVFP4 weights", .{ weight_name, weight.dtype }),
            }
            return .{ .value = out, .mark = mark };
        }
        out.scheme = Scheme.of(weight, scale) catch |err| switch (err) {
            error.UnknownStorage => return b.fail(error.UnsupportedScheme, "{s}: unknown weight storage {t}", .{ weight_name, weight.dtype }),
        };
        if (!site.allowed().contains(out.scheme))
            return b.fail(error.UnsupportedScheme, "{s}: stored as {t}, which is not read here ({s})", .{ weight_name, out.scheme, site.reads() });

        const n: u64, const k: u64 = .{ rows, cols };
        switch (out.scheme) {
            .bf16 => if (weight.dtype != .BF16 or !shaped(weight, &.{ n, k }))
                return b.mismatch(weight_name, weight, "BF16 {{ {d}, {d} }}", .{ n, k }),
            .nvfp4 => {
                const group = b.config.quant.nvfp4_group;
                try b.nvfp4Codes(weight_name, weight, n, k);
                if (!shaped(scale.?, &.{ n, k / group }))
                    return b.mismatch(scale_name, scale.?, "F8_E4M3 {{ {d}, {d} }}", .{ n, k / group });
                out.scale_2 = try b.nvfp4Scale2(name);
                out.input_scale = try b.nvfp4Input(name);
                out.scale = scale;
            },
            .mxfp8 => {
                if (k % mxfp8_block != 0 or weight.dtype != .F8_E4M3 or !shaped(weight, &.{ n, k }))
                    return b.mismatch(weight_name, weight, "F8_E4M3 {{ {d}, {d} }} with inputs in blocks of {d}", .{ n, k, mxfp8_block });
                if (!shaped(scale.?, &.{ n, k / mxfp8_block }))
                    return b.mismatch(scale_name, scale.?, "U8 {{ {d}, {d} }} (a ue8m0 scale every {d} inputs)", .{ n, k / mxfp8_block, mxfp8_block });
                if (b.config.quant.find(name)) |declared_as| if (declared_as.algo != .mxfp8)
                    return b.fail(error.InvalidCheckpoint, "{s}: MXFP8 tensors where the config declares {t}", .{ weight_name, declared_as.algo });
                out.scale = scale;
            },
        }
        if (b.missing.items.len == mark) {
            b.stats.linears.getPtr(out.scheme).* += 1;
            b.stats.bytes.getPtr(out.scheme).* += out.bytes();
        }
        return .{ .value = out, .mark = mark };
    }

    fn nvfp4Codes(b: *Builder, weight_name: []const u8, weight: Tensor, n: u64, k: u64) Error!void {
        if (k % b.config.quant.nvfp4_group != 0 or !shaped(weight, &.{ n, k / 2 }))
            return b.mismatch(weight_name, weight, "U8 {{ {d}, {d} }} (two codes a byte)", .{ n, k / 2 });
    }

    /// `name.weight_scale_2`, one F32; null and recorded where it is missing.
    fn nvfp4Scale2(b: *Builder, name: []const u8) Error!?Tensor {
        const scale_2_name = try b.fmt("{s}.weight_scale_2", .{name});
        const scale_2 = (try b.need(scale_2_name)) orelse return null;
        if (scale_2.dtype != .F32 or scale_2.count() != 1)
            return b.mismatch(scale_2_name, scale_2, "one F32", .{});
        return scale_2;
    }

    /// `name.input_scale`, one F32, or null where the checkpoint stores none (it is not waited for).
    fn nvfp4Input(b: *Builder, name: []const u8) Error!?Tensor {
        const input_name = try b.fmt("{s}.input_scale", .{name});
        const input = b.checkpoint.get(input_name) orelse return null;
        if (input.dtype != .F32 or input.count() != 1)
            return b.mismatch(input_name, input, "one F32", .{});
        return input;
    }

    /// An NVFP4 linear whose block scales are csf-encoded: both streams, their shapes checked against the
    /// weight's; the values are checked where they are decoded.
    fn csfLinear(b: *Builder, found: Linear, mark: usize, site: Site, fixed_name: []const u8, exceptions_name: []const u8) Error!Checked(Linear) {
        var out = found;
        out.scheme = .nvfp4;
        if (!site.allowed().contains(.nvfp4))
            return b.fail(error.UnsupportedScheme, "{s}: stored as nvfp4 with csf scales, which is not read here ({s})", .{ fixed_name, site.reads() });
        const n: u64, const k: u64 = .{ out.rows, out.cols };
        try b.nvfp4Codes(try b.fmt("{s}.weight", .{out.name}), out.weight, n, k);
        const group = b.config.quant.nvfp4_group;
        if (group != 16 or !csf.fits(n, k / group))
            return b.fail(error.UnsupportedScheme, "{s}: csf scales of {d} rows and {d} columns; csf takes whole slabs of 16 rows", .{ fixed_name, n, k / group });
        const shape: csf.Shape = .{ .rows = n, .cols = k / group };
        const fixed = try b.need(fixed_name);
        const exceptions = try b.need(exceptions_name);
        if (fixed) |tensor| {
            const dims = shape.fixedDims();
            if (tensor.dtype != .U8 or !shaped(tensor, &dims))
                return b.mismatch(fixed_name, tensor, "U8 {{ {d}, {d} }}", .{ dims[0], dims[1] });
        }
        if (exceptions) |tensor| {
            if (tensor.dtype != .U32 or tensor.shape.len != 1 or tensor.shape[0] > shape.count())
                return b.mismatch(exceptions_name, tensor, "U32 {{ count }}, at most {d}", .{shape.count()});
        }
        out.scale_2 = try b.nvfp4Scale2(out.name);
        out.input_scale = try b.nvfp4Input(out.name);
        if (fixed != null and exceptions != null) out.csf = .{ .fixed = fixed.?, .exceptions = exceptions.? };
        if (b.missing.items.len == mark) {
            b.stats.linears.getPtr(.nvfp4).* += 1;
            b.stats.bytes.getPtr(.nvfp4).* += out.bytes();
            b.stats.csf += 1;
        }
        return .{ .value = out, .mark = mark };
    }

    fn vector(b: *Builder, name: []const u8, len: u32, gamma: bool) Error!Vector {
        const tensor = (try b.need(name)) orelse return .{ .name = name, .tensor = absent, .gamma = gamma };
        if ((tensor.dtype != .BF16 and tensor.dtype != .F32 and tensor.dtype != .F16) or !shaped(tensor, &.{len}))
            return b.mismatch(name, tensor, "BF16, F16 or F32 {{ {d} }}", .{len});
        b.stats.small += 1;
        b.stats.small_bytes += tensor.data.len;
        return .{ .name = name, .tensor = tensor, .gamma = gamma };
    }

    fn conv(b: *Builder, name: []const u8, channels: u32, taps: u32) Error!Conv {
        const tensor = (try b.need(name)) orelse return .{ .name = name, .tensor = absent, .channels = channels, .taps = taps };
        if (tensor.dtype != .BF16 or tensor.count() != @as(u64, channels) * taps or tensor.shape.len == 0 or tensor.shape[0] != channels)
            return b.mismatch(name, tensor, "BF16 {{ {d}, 1, {d} }}", .{ channels, taps });
        b.stats.small += 1;
        b.stats.small_bytes += tensor.data.len;
        return .{ .name = name, .tensor = tensor, .channels = channels, .taps = taps };
    }

    /// A linear inside a larger part: the part's own mark decides whether it stands.
    fn sub(b: *Builder, base: []const u8, comptime field: []const u8, rows: u32, cols: u32, site: Site) Error!Linear {
        return (try b.linear(try b.fmt("{s}." ++ field, .{base}), rows, cols, site)).value;
    }

    pub fn hyperConnection(b: *Builder, name: []const u8, inject: bool) Error!Checked(HyperConnection) {
        const c = b.config;
        const mark = b.missing.items.len;
        return .{ .mark = mark, .value = .{
            .down = try b.sub(name, "input_mix_weight_down", c.low, c.wide(), .dense),
            .inject = if (inject) try b.sub(name, "block_inject_weight", c.streams, c.wide(), .dense) else null,
            .up = try b.sub(name, "input_mix_weight_up", c.wide(), c.low, .dense),
            .norm = try b.vector(try b.fmt("{s}.hc_norm.weight", .{name}), c.wide(), true),
        } };
    }

    fn deltaNet(b: *Builder, name: []const u8) Error!DeltaNet {
        const c = b.config;
        const values = c.value_heads * c.value_head_dim;
        return .{
            .qkv = try b.sub(name, "in_proj_qkv", c.convDim(), c.hidden, .dense),
            .z = try b.sub(name, "in_proj_z", values, c.hidden, .dense),
            .b = try b.sub(name, "in_proj_b", c.value_heads, c.hidden, .dense),
            .a = try b.sub(name, "in_proj_a", c.value_heads, c.hidden, .dense),
            .conv = try b.conv(try b.fmt("{s}.conv1d.weight", .{name}), c.convDim(), c.conv_kernel),
            .a_log = try b.vector(try b.fmt("{s}.A_log", .{name}), c.value_heads, false),
            .dt_bias = try b.vector(try b.fmt("{s}.dt_bias", .{name}), c.value_heads, false),
            .norm = try b.vector(try b.fmt("{s}.norm.weight", .{name}), c.value_head_dim, false),
            .out = try b.sub(name, "out_proj", c.hidden, values, .dense),
        };
    }

    fn attention(b: *Builder, name: []const u8) Error!Attention {
        const c = b.config;
        return .{
            .q = try b.sub(name, "q_proj", c.heads * 2 * c.head_dim, c.hidden, .dense),
            .k = try b.sub(name, "k_proj", c.kv_heads * c.head_dim, c.hidden, .dense),
            .v = try b.sub(name, "v_proj", c.kv_heads * c.head_dim, c.hidden, .dense),
            .index_qk = try b.sub(name, "indexer.index_qk_proj", (c.index_heads + c.index_kv_heads) * c.index_dim, c.hidden, .dense),
            .q_norm = try b.vector(try b.fmt("{s}.q_norm.weight", .{name}), c.head_dim, true),
            .k_norm = try b.vector(try b.fmt("{s}.k_norm.weight", .{name}), c.head_dim, true),
            .index_q_norm = try b.vector(try b.fmt("{s}.indexer.q_layernorm.weight", .{name}), c.index_dim, true),
            .index_k_norm = try b.vector(try b.fmt("{s}.indexer.k_layernorm.weight", .{name}), c.index_dim, true),
            .o = try b.sub(name, "o_proj", c.hidden, c.heads * c.head_dim, .dense),
        };
    }

    fn expert(b: *Builder, name: []const u8, width: u32, site: Site) Error!Expert {
        const c = b.config;
        return .{
            .gate = try b.sub(name, "gate_proj", width, c.hidden, site),
            .up = try b.sub(name, "up_proj", width, c.hidden, site),
            .down = try b.sub(name, "down_proj", c.hidden, width, site),
        };
    }

    fn moe(b: *Builder, name: []const u8, drafter: bool) Error!Moe {
        const c = b.config;
        const router = try b.sub(name, "gate", c.experts, c.hidden, .plain);
        const shared_gate = try b.sub(name, "shared_expert_gate", 1, c.hidden, .plain);
        const shared = try b.expert(try b.fmt("{s}.shared_expert", .{name}), c.shared_width, .dense);

        const stacked_gate_up = try b.fmt("{s}.experts.gate_up_proj", .{name});
        const stacked_down = try b.fmt("{s}.experts.down_proj", .{name});
        const per_expert = b.checkpoint.get(try b.fmt("{s}.experts.0.gate_proj.weight", .{name})) != null;
        // an MTP layer left out of quantization keeps its experts stacked in bf16
        const stacked = drafter and !per_expert and
            (b.checkpoint.get(stacked_gate_up) != null or b.checkpoint.get(stacked_down) != null or
                b.config.quant.find(try b.fmt("{s}.experts", .{name})) == null);
        var experts: Experts = undefined;
        if (stacked) {
            const e: u64, const w: u64, const d: u64 = .{ c.experts, c.moe_width, c.hidden };
            var found: Experts.Stacked = .{ .gate_up = absent, .down = absent };
            if (try b.need(stacked_gate_up)) |tensor| {
                if (tensor.dtype != .BF16 or !shaped(tensor, &.{ e, 2 * w, d }))
                    return b.mismatch(stacked_gate_up, tensor, "BF16 {{ {d}, {d}, {d} }}", .{ e, 2 * w, d });
                found.gate_up = tensor;
                b.stats.linears.getPtr(.bf16).* += 1;
                b.stats.bytes.getPtr(.bf16).* += tensor.data.len;
            }
            if (try b.need(stacked_down)) |tensor| {
                if (tensor.dtype != .BF16 or !shaped(tensor, &.{ e, d, w }))
                    return b.mismatch(stacked_down, tensor, "BF16 {{ {d}, {d}, {d} }}", .{ e, d, w });
                found.down = tensor;
                b.stats.linears.getPtr(.bf16).* += 1;
                b.stats.bytes.getPtr(.bf16).* += tensor.data.len;
            }
            experts = .{ .stacked = found };
        } else {
            const each = try b.arena.alloc(Expert, c.experts);
            for (each, 0..) |*slot, i| {
                slot.* = try b.expert(try b.fmt("{s}.experts.{d}", .{ name, i }), c.moe_width, if (drafter) .drafter else .routed);
            }
            experts = .{ .each = each };
        }
        return .{ .router = router, .shared_gate = shared_gate, .shared = shared, .experts = experts };
    }

    fn table(b: *Builder, base: []const u8, ngram: Ngram) Error!Table {
        const c = b.config;
        const shards = try b.arena.alloc(Table.Shard, c.ngram_shards);
        var scheme: ?Scheme = null;
        var rows: u64 = 0;
        var whole = true;
        for (shards, 0..) |*shard, i| {
            // the flat spelling, then the nested one some exports use
            var key = try b.fmt("{s}.shard_{d}", .{ base, i });
            var weight_name = try b.fmt("{s}.weight", .{key});
            if (b.checkpoint.get(weight_name) == null) {
                const nested = try b.fmt("{s}.shards.{d}", .{ base, i });
                const nested_weight = try b.fmt("{s}.weight", .{nested});
                if (b.checkpoint.get(nested_weight) != null) {
                    key = nested;
                    weight_name = nested_weight;
                }
            }
            shard.* = .{ .weight = absent };
            const weight = (try b.need(weight_name)) orelse {
                whole = false;
                continue;
            };
            if (b.checkpoint.get(try b.fmt("{s}.scales", .{key})) != null)
                return b.fail(error.UnsupportedScheme, "{s}: an MLX 4-bit n-gram table, which zspark does not read", .{weight_name});
            const block_name = try b.fmt("{s}.weight_scale", .{key});
            const kind: Scheme = switch (weight.dtype) {
                .U8 => .nvfp4,
                .BF16 => .bf16,
                else => return b.fail(error.UnsupportedScheme, "{s}: n-gram rows stored as {t}; zspark reads bf16 and NVFP4 tables", .{ weight_name, weight.dtype }),
            };
            if (scheme != null and scheme.? != kind)
                return b.fail(error.InvalidCheckpoint, "{s}: the n-gram shards mix {t} and {t}", .{ weight_name, scheme.?, kind });
            scheme = kind;
            const width: u64 = if (kind == .nvfp4) ngram.dims / 2 else ngram.dims;
            if (weight.shape.len != 2 or weight.shape[1] != width)
                return b.mismatch(weight_name, weight, "rows of {d}", .{width});
            shard.weight = weight;
            rows += weight.shape[0];
            b.stats.table_shards += 1;
            b.stats.table_bytes += weight.data.len;
            if (kind == .nvfp4) {
                if (try b.need(block_name)) |block| {
                    if (block.dtype != .F8_E4M3 or !shaped(block, &.{ weight.shape[0], ngram.dims / 16 }))
                        return b.mismatch(block_name, block, "F8_E4M3 {{ {d}, {d} }}", .{ weight.shape[0], ngram.dims / 16 });
                    shard.scale = block;
                    b.stats.table_bytes += block.data.len;
                }
            }
        }
        b.stats.table_scheme = scheme orelse b.stats.table_scheme;
        if (whole and rows != ngram.rows)
            return b.fail(error.InvalidCheckpoint, "{s}: the n-gram shards hold {d} rows, expected {d}", .{ base, rows, ngram.rows });

        // a quantized table's one scale; bf16 rows may carry one too
        var scale: ?Tensor = null;
        if (scheme) |kind| {
            const scale_name = try b.fmt("{s}.{s}", .{ base, if (kind == .nvfp4) "weight_scale_2" else "weight_scale" });
            scale = if (kind == .bf16) b.checkpoint.get(scale_name) else try b.need(scale_name);
            if (scale) |s| {
                if ((s.dtype != .F32 and s.dtype != .BF16) or s.count() != 1)
                    return b.mismatch(scale_name, s, "one float for the table", .{});
            }
        }
        return .{ .scheme = scheme orelse .bf16, .shards = shards, .scale = scale, .rows = ngram.rows, .dims = ngram.dims };
    }

    /// The hash constants the checkpoint ships must be the ones the config derives.
    fn checkNgram(b: *Builder, name: []const u8, comptime T: type, derived: []const T) Error!void {
        const tensor = (try b.need(name)) orelse return;
        if (tensor.dtype != .I64 or tensor.count() != derived.len)
            return b.mismatch(name, tensor, "I64 {{ {d} }}", .{derived.len});
        for (derived, 0..) |want, i| {
            const got = std.mem.readInt(i64, tensor.data[i * 8 ..][0..8], .little);
            if (got != @as(i64, @intCast(want)))
                return b.fail(error.InvalidCheckpoint, "{s}[{d}] is {d}; the config derives {d}", .{ name, i, got, want });
        }
    }

    fn ple(b: *Builder, name: []const u8, index: usize) Error!Ple {
        const c = b.config;
        const ngram: Ngram = try .init(b.arena, c.*, index);
        const embedding = try b.fmt("{s}.ple_embedding", .{name});
        try b.checkNgram(try b.fmt("{s}.layer_multipliers", .{embedding}), i64, ngram.multipliers);
        try b.checkNgram(try b.fmt("{s}.ngram_heads_offsets", .{embedding}), u64, ngram.head_offsets);
        try b.checkNgram(try b.fmt("{s}.ngram_heads_vocab_sizes", .{embedding}), u64, ngram.head_sizes);
        return .{
            .index = index,
            .ngram = ngram,
            .table = try b.table(try b.fmt("{s}.ngram_embedding", .{embedding}), ngram),
            .key = try b.sub(name, "key_proj", c.wide(), c.ple_dim, .plain),
            .value = try b.sub(name, "value_proj", c.hidden, c.ple_dim, .plain),
            .norm_key = try b.vector(try b.fmt("{s}.norm_key.weight", .{name}), c.wide(), true),
            .norm_query = try b.vector(try b.fmt("{s}.norm_query.weight", .{name}), c.wide(), true),
            .norm_conv = try b.vector(try b.fmt("{s}.norm_conv.weight", .{name}), c.wide(), true),
            .conv = try b.conv(try b.fmt("{s}.conv1d.weight", .{name}), c.wide(), c.ple_kernel),
        };
    }

    pub fn layer(b: *Builder, name: []const u8, kind: qwen4_config.LayerType, ple_index: ?usize, drafter: bool) Error!?Layer {
        const mark = b.missing.items.len;
        const found: Layer = .{
            .attn_hc = (try b.hyperConnection(try b.fmt("{s}.attn_hyper_connection", .{name}), true)).value,
            .mlp_hc = (try b.hyperConnection(try b.fmt("{s}.mlp_hyper_connection", .{name}), true)).value,
            .block = switch (kind) {
                .linear_attention => .{ .linear_attention = try b.deltaNet(try b.fmt("{s}.linear_attn", .{name})) },
                .full_attention => .{ .full_attention = try b.attention(try b.fmt("{s}.self_attn", .{name})) },
            },
            .ple = if (ple_index) |index| try b.ple(try b.fmt("{s}.ple", .{name}), index) else null,
            .moe = try b.moe(try b.fmt("{s}.mlp", .{name}), drafter),
        };
        return if (b.missing.items.len == mark) found else null;
    }

    pub fn mtp(b: *Builder) Error!?Mtp {
        const c = b.config;
        const mark = b.missing.items.len;
        const norm_embedding = try b.vector("mtp.pre_fc_norm_embedding.weight", c.hidden, true);
        const norm_hidden = try b.vector("mtp.pre_fc_norm_hidden.weight", c.wide(), true);
        // read as the head's other linears; TensorFold reads them BF16 alone
        const fc_embedding = try b.sub("mtp", "fc_embedding", c.hidden, c.hidden, .dense);
        const fc_hidden = try b.sub("mtp", "fc_hidden", c.hidden, c.hidden, .dense);
        const found = try b.layer("mtp.layers.0", .full_attention, null, true);
        const mixer = (try b.hyperConnection("mtp.hyper_connection_mixer", false)).value;
        if (b.missing.items.len != mark) return null;
        return .{
            .norm_embedding = norm_embedding,
            .norm_hidden = norm_hidden,
            .fc_embedding = fc_embedding,
            .fc_hidden = fc_hidden,
            .layer = found.?,
            .mixer = mixer,
        };
    }

    /// Whether norm weights are stored centred (gamma - 1) rather than as scales, from each layer's
    /// `attn_hyper_connection.hc_norm` mean, as TensorFold's reader.py `norms_around_one`.
    pub fn normsCentred(b: *Builder, prefix: []const u8) Error!?bool {
        const means = try b.arena.alloc(f64, b.config.layers);
        var count: usize = 0;
        for (0..b.config.layers) |i| {
            const name = try b.fmt("{s}layers.{d}.attn_hyper_connection.hc_norm.weight", .{ prefix, i });
            const tensor = b.checkpoint.get(name) orelse continue;
            if ((tensor.dtype != .BF16 and tensor.dtype != .F32 and tensor.dtype != .F16) or tensor.count() == 0)
                return b.mismatch(name, tensor, "BF16, F16 or F32 {{ {d} }}", .{b.config.wide()});
            var sum: f64 = 0;
            const len: usize = @intCast(tensor.count());
            for (0..len) |at| sum += floatAt(tensor, at);
            means[count] = sum / @as(f64, @floatFromInt(len));
            count += 1;
        }
        return centredFromMeans(means[0..count]) catch |err| switch (err) {
            error.Undecided => b.fail(error.InvalidCheckpoint, "cannot tell how the norm weights are stored: the hc_norm means sit neither around 1 nor around 0", .{}),
        };
    }
};
