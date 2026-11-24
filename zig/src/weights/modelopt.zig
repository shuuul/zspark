//! A ModelOpt checkpoint's quantization map and its linears as stored (TensorFold's `core/modelopt.zig` names
//! the format). `quantized_layers` names a module and how its weights are stored; a module it does not name is
//! BF16. A linear is read as stored, every part checked against the shape its outputs and inputs make, and cut
//! for a rank by its split (`weights/share.zig`) part by part. Shared by Flash Next's config (the map), GLM's
//! layout (the map and the linears) and Qwen3-ASR's (a single-format export: one `quant_algo` and the modules it
//! excludes, `Uniform`). Host only: bytes in, bytes out.
//!
//! The parts, for a matrix of `rows` outputs and `cols` inputs:
//!
//! - BF16: `weight` BF16 [rows, cols].
//! - MXFP8: `weight` F8_E4M3 [rows, cols], `weight_scale` U8 (ue8m0) [rows, cols / 32].
//! - FP8_PB_WO: `weight` F8_E4M3 [rows, cols], `weight_scale` F32 [rows / 128, cols / 128], stored 2-D or as
//!   [rows / 128, 1, cols / 128, 1] (vLLM's `KFp8Block128` squeezes the ones; read here as the 2-D).
//! - NVFP4 and W4A16_NVFP4: `weight` U8 [rows, cols / 2] (two E2M1 codes a byte), `weight_scale_2` F32 [], and
//!   the block scales e4m3 [rows, cols / 16] as `weight_scale` (plain) or as `weight_scale.csf32_*`
//!   (`csf_blocks.zig`). `input_scale` F32 [] is the activation scale: NVFP4 experts consume it for
//!   W4A4; a W4A16 caller keeps bf16 activations.
//! - FP8: `weight` F8_E4M3 [rows, cols], `weight_scale` F32 [] (the tensor's one scale, held in `scale_2`),
//!   `input_scale` F32 [] is retained for static FP8 activations (Nemotron's Mamba projections).
//!   Callers that keep bf16 activations leave it unused.
//!
//! A rank's cut: half of the rows (`row`, `vocab`) or of the columns (`col`) of the weight and of a scale of two
//! dimensions alike, by whole scale blocks; `weight_scale_2` whole; csf streams whole with the rank's window of
//! the plane (vLLM's `nvfp4_csf.py` stages a rank's rows of w1 and w3, its columns of w2).

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const safetensors = @import("safetensors.zig");
const share = @import("share.zig");
const csf_blocks = @import("csf_blocks.zig");
const Tensor = safetensors.Tensor;
const Split = share.Split;

/// How ModelOpt stored one module's weights.
pub const Algo = enum {
    /// E2M1 codes, an e4m3 scale per `group_size` inputs, one fp32 scale for the tensor (NVFP4, W4A16_NVFP4).
    nvfp4,
    /// e4m3 with one scale for the tensor.
    fp8,
    /// e4m3 with an fp32 scale per `group_size` x `group_size` block (FP8_PB_WO).
    fp8_block,
    /// e4m3 with a power-of-two (ue8m0) scale per 32 inputs.
    mxfp8,

    /// ModelOpt's `quant_algo`, upper case; null where it names none of these.
    pub fn of(name: []const u8) ?Algo {
        if (std.mem.eql(u8, name, "NVFP4") or std.mem.eql(u8, name, "W4A16_NVFP4")) return .nvfp4;
        if (std.mem.eql(u8, name, "FP8_PB_WO")) return .fp8_block;
        if (std.mem.eql(u8, name, "FP8")) return .fp8;
        if (std.mem.eql(u8, name, "MXFP8")) return .mxfp8;
        return null;
    }
};

/// How an NVFP4 module's block scales are stored (`weight_scale_encoding`).
pub const ScaleEncoding = enum {
    /// e4m3 bytes, `weight_scale`.
    plain,
    /// "csf": slabs of 16 rows and exceptions (`weights/csf.zig`, Flash Next FP4-Local).
    csf,
    /// "csf_32x2_v1": blocks of 32 rows by 2 columns (`weights/csf_blocks.zig`, GLM FP4-Local).
    csf32,
};

pub const QuantizedLayer = struct {
    /// A module path; it covers every linear beneath it.
    name: []const u8,
    algo: Algo,
    /// Inputs a scale covers: `group_size`, or `weight_block_size`'s second (MXFP8's [1, 32], FP8_PB_WO's
    /// [128, 128]); null where neither is given.
    group_size: ?u32,
    scale_encoding: ScaleEncoding = .plain,
    /// W4A16_NVFP4: an NVFP4 weight whose activations stay 16-bit; NVFP4's are quantized as well where a model
    /// runs them so (vLLM's ModelOpt mixed precision; GLM FP4-Local's routed experts, spec 114 WS-03).
    a16: bool = false,
};

/// The entry covering module or tensor `name` in `layers`, if any: a name that is the entry's or lies beneath it.
pub fn find(layers: []const QuantizedLayer, name: []const u8) ?QuantizedLayer {
    return layers[findIndex(layers, name) orelse return null];
}

/// `find`'s entry by its place in `layers`.
pub fn findIndex(layers: []const QuantizedLayer, name: []const u8) ?usize {
    for (layers, 0..) |layer, at| {
        if (!std.mem.startsWith(u8, name, layer.name)) continue;
        if (name.len == layer.name.len or name[layer.name.len] == '.') return at;
    }
    return null;
}

pub const EntryError = error{ UnknownAlgo, UnknownEncoding, InvalidEntry };

/// One `quantized_layers` entry, `value` its object. A `weight_block_size` of two counts is read where there is
/// no `group_size`. The encodings are the names `ScaleEncoding` gives; which algorithm may carry which is the
/// model's to refuse.
pub fn entry(name: []const u8, value: json.ObjectMap) EntryError!QuantizedLayer {
    const algo_name = value.get("quant_algo") orelse return error.InvalidEntry;
    if (algo_name != .string) return error.InvalidEntry;
    var upper: [32]u8 = undefined;
    if (algo_name.string.len > upper.len) return error.UnknownAlgo;
    const algo = Algo.of(std.ascii.upperString(&upper, algo_name.string)) orelse return error.UnknownAlgo;
    var group_size: ?u32 = null;
    if (value.get("group_size")) |size| {
        if (size != .integer or size.integer <= 0 or size.integer > 1 << 16) return error.InvalidEntry;
        group_size = @intCast(size.integer);
    } else if (value.get("weight_block_size")) |block| {
        if (block != .array or block.array.items.len != 2) return error.InvalidEntry;
        const inputs = block.array.items[1];
        if (inputs != .integer or inputs.integer <= 0 or inputs.integer > 1 << 16) return error.InvalidEntry;
        group_size = @intCast(inputs.integer);
    }
    var scale_encoding: ScaleEncoding = .plain;
    if (value.get("weight_scale_encoding")) |named| {
        if (named != .string) return error.InvalidEntry;
        if (std.mem.eql(u8, named.string, "csf")) scale_encoding = .csf else if (std.mem.eql(u8, named.string, csf_blocks.encoding)) scale_encoding = .csf32 else return error.UnknownEncoding;
    }
    const a16 = std.ascii.eqlIgnoreCase(algo_name.string, "W4A16_NVFP4");
    return .{ .name = name, .algo = algo, .group_size = group_size, .scale_encoding = scale_encoding, .a16 = a16 };
}

/// A checkpoint quantized by one algorithm throughout but for the modules it excludes, as ModelOpt's
/// `export_hf_checkpoint` writes a single-format build (`W4A16_NVFP4`, spec 140): no `quantized_layers`, a
/// `quant_algo` and a list of wildcards. vLLM's `ModelOptQuantConfigBase.from_config` and `is_layer_excluded`
/// (modelopt.py, read 2026-10-28) are what is read and how a module is matched.
pub const Uniform = struct {
    /// The entry every linear not excluded takes; its name is the module's when `of` gives it.
    layer: QuantizedLayer,
    /// `ignore` (config.json) or `exclude_modules` (hf_quant_config.json): names and `*`/`?` wildcards.
    exclude: []const []const u8,

    /// The entry of `module` (null: BF16, excluded).
    pub fn of(self: Uniform, module: []const u8) ?QuantizedLayer {
        if (excluded(self.exclude, module)) return null;
        var layer = self.layer;
        layer.name = module;
        return layer;
    }
};

pub const UniformError = EntryError || error{Mixed};

/// config.json's `quantization_config` or hf_quant_config.json's root, as vLLM's `from_config` reads either: a
/// `quantization` object (`quant_algo`, `group_size`, `exclude_modules`), or the flat form (`quant_algo`, `ignore`).
/// The flat form keeps its group in `config_groups`' first group's `weights.group_size` where it gives none.
/// `MIXED_PRECISION` is `error.Mixed`: a map of `quantized_layers` the model reads with `entry`.
pub fn uniform(arena: Allocator, value: json.ObjectMap) (UniformError || Allocator.Error)!Uniform {
    const nested = if (value.get("quantization")) |inner| (if (inner == .object) inner.object else null) else null;
    const block = nested orelse value;
    if (block.get("quant_algo")) |algo| if (algo == .string and std.ascii.eqlIgnoreCase(algo.string, "MIXED_PRECISION")) return error.Mixed;
    var layer = try entry("", block);
    if (layer.group_size == null and nested == null) layer.group_size = groupOfFirst(value);
    const listed = if (nested != null) block.get("exclude_modules") else value.get("ignore") orelse value.get("exclude_modules");
    const list = listed orelse return .{ .layer = layer, .exclude = &.{} };
    if (list != .array) return error.InvalidEntry;
    const exclude = try arena.alloc([]const u8, list.array.items.len);
    for (exclude, list.array.items) |*name, item| {
        if (item != .string) return error.InvalidEntry;
        name.* = item.string;
    }
    return .{ .layer = layer, .exclude = exclude };
}

/// compressed-tensors' `config_groups`: the first group's weights' `group_size`, if any.
fn groupOfFirst(value: json.ObjectMap) ?u32 {
    const groups = value.get("config_groups") orelse return null;
    if (groups != .object or groups.object.count() == 0) return null;
    const first = groups.object.values()[0];
    if (first != .object) return null;
    const weights = first.object.get("weights") orelse return null;
    if (weights != .object) return null;
    const size = weights.object.get("group_size") orelse return null;
    if (size != .integer or size.integer <= 0 or size.integer > 1 << 16) return null;
    return @intCast(size.integer);
}

/// vLLM's `is_layer_excluded`: a pattern inside the module's name (an equal one included; the legacy match vLLM
/// keeps for exports before ModelOpt 0.39), or matching it as `fnmatch` does (`*` any run of characters, dots
/// included, `?` one).
pub fn excluded(patterns: []const []const u8, module: []const u8) bool {
    for (patterns) |pattern| {
        if (std.mem.indexOf(u8, module, pattern) != null) return true;
        if (wildcard(pattern, module)) return true;
    }
    return false;
}

/// `fnmatch.fnmatchcase(name, pattern)` for `*` and `?`; any other character stands for itself.
pub fn wildcard(pattern: []const u8, name: []const u8) bool {
    var p: usize = 0;
    var n: usize = 0;
    // the last `*` and the place in the name it took up to, to let it take one more on a mismatch
    var star: ?usize = null;
    var taken: usize = 0;
    while (n < name.len) {
        if (p < pattern.len and (pattern[p] == '?' or pattern[p] == name[n])) {
            p += 1;
            n += 1;
        } else if (p < pattern.len and pattern[p] == '*') {
            star = p;
            p += 1;
            taken = n;
        } else if (star) |at| {
            p = at + 1;
            taken += 1;
            n = taken;
        } else return false;
    }
    while (p < pattern.len and pattern[p] == '*') p += 1;
    return p == pattern.len;
}

/// A linear's storage, as `read` finds it.
pub const Scheme = enum {
    bf16,
    mxfp8,
    fp8_block,
    nvfp4,
    /// e4m3 with one fp32 scale for the tensor (`weight_scale`).
    fp8,

    /// The scheme of a module the map gives `layer` for (null: BF16); null for what no linear here is stored as
    /// (Flash Next's csf).
    pub fn of(layer: ?QuantizedLayer) ?Scheme {
        const found = layer orelse return .bf16;
        return switch (found.algo) {
            .mxfp8 => if ((found.group_size orelse mxfp8_block) == mxfp8_block and found.scale_encoding == .plain) .mxfp8 else null,
            .fp8_block => if ((found.group_size orelse fp8_block) == fp8_block and found.scale_encoding == .plain) .fp8_block else null,
            .nvfp4 => if ((found.group_size orelse nvfp4_block) == nvfp4_block and found.scale_encoding != .csf) .nvfp4 else null,
            .fp8 => if (found.group_size == null and found.scale_encoding == .plain) .fp8 else null,
        };
    }
};

/// Inputs an MXFP8 scale covers, an FP8_PB_WO block's side, inputs an NVFP4 scale covers.
pub const mxfp8_block = 32;
pub const fp8_block = 128;
pub const nvfp4_block = 16;

/// A linear as stored, or a rank's cut of it.
pub const Linear = struct {
    scheme: Scheme,
    /// The matrix's outputs and inputs.
    rows: u64,
    cols: u64,
    weight: Tensor,
    /// MXFP8: U8 [rows, cols / 32]. FP8_PB_WO: F32 [rows / 128, cols / 128]. NVFP4 with plain scales: F8_E4M3
    /// [rows, cols / 16].
    scale: ?Tensor = null,
    /// NVFP4's `weight_scale_2` or FP8's `weight_scale`: F32 [], the tensor's one scale.
    scale_2: ?Tensor = null,
    /// NVFP4 with csf_32x2_v1 scales: the streams, whole, and the plane's window this linear holds.
    csf: ?Csf = null,
    /// NVFP4/FP8: the activations' F32 [] scale; consumed by quantized-activation callers.
    input_scale: ?Tensor = null,

    pub const Csf = struct {
        shape: csf_blocks.Shape,
        streams: csf_blocks.Streams,
        window: csf_blocks.Window,
    };

    /// The tensors stored for it: how many, their bytes, and what of them a rank of `share.ranks` holds by
    /// `split` (the weight and a scale of two dimensions cut, the rest whole; `input_scale` none).
    pub fn held(self: Linear, split: Split) struct { count: usize, bytes: u64, rank: u64 } {
        var count: usize = 1;
        var bytes: u64 = self.weight.data.len;
        var rank: u64 = split.share(self.weight.data.len);
        if (self.scale) |scale| {
            count += 1;
            bytes += scale.data.len;
            rank += split.share(scale.data.len);
        }
        if (self.scale_2) |scale_2| {
            count += 1;
            bytes += scale_2.data.len;
            rank += if (split == .drop) 0 else scale_2.data.len;
        }
        if (self.csf) |stored| {
            const streams = stored.streams;
            const all = streams.bases.len + streams.codes.len + streams.page_bases.len + streams.offsets.len + streams.exceptions.len;
            count += 5;
            bytes += all;
            rank += if (split == .drop) 0 else all;
        }
        if (self.input_scale) |input| {
            count += 1;
            bytes += input.data.len;
        }
        return .{ .count = count, .bytes = bytes, .rank = rank };
    }

    /// NVFP4: the block scales this linear holds, e4m3 [rows, cols / 16]: the stored bytes, or the csf
    /// window decoded into `arena`.
    pub fn blockScales(self: Linear, arena: Allocator) (csf_blocks.Error || Allocator.Error)![]const u8 {
        std.debug.assert(self.scheme == .nvfp4);
        if (self.scale) |scale| return scale.data;
        const stored = self.csf.?;
        const out = try arena.alloc(u8, @intCast(stored.window.count()));
        try csf_blocks.decodeWindow(stored.shape, stored.streams, stored.window, out);
        return out;
    }
};

pub const Error = error{ MissingTensor, UnexpectedTensor, Indivisible } || Allocator.Error;

/// The linear `module` (`module.weight` and its parts) of `source` (anything with `get(name) ?Tensor`), stored as
/// `layer` says (null: BF16), each part's dtype and shape checked against `rows` outputs and `cols` inputs. A
/// part it does not find, or finds otherwise, is set in `diagnostic` (anything with `set(arena, format, args)`).
pub fn read(arena: Allocator, source: anytype, module: []const u8, layer: ?QuantizedLayer, rows: u64, cols: u64, diagnostic: anytype) Error!Linear {
    const scheme = Scheme.of(layer) orelse {
        diagnostic.set(arena, "{s} is stored {t} in blocks of {?d}, scales {t}; zspark reads BF16, MXFP8 (32), FP8_PB_WO (128) and NVFP4 (16, scales plain or csf_32x2_v1) here", .{ module, layer.?.algo, layer.?.group_size, layer.?.scale_encoding });
        return error.UnexpectedTensor;
    };
    const r: Parts(@TypeOf(source), @TypeOf(diagnostic)) = .{ .arena = arena, .source = source, .module = module, .diagnostic = diagnostic };
    var out: Linear = .{ .scheme = scheme, .rows = rows, .cols = cols, .weight = undefined };
    switch (scheme) {
        .bf16 => out.weight = try r.part("weight", .BF16, &.{ rows, cols }),
        .fp8 => {
            out.weight = try r.part("weight", .F8_E4M3, &.{ rows, cols });
            out.scale_2 = try r.part("weight_scale", .F32, &.{});
            if (r.find("input_scale") != null) out.input_scale = try r.part("input_scale", .F32, &.{});
        },
        .mxfp8 => {
            if (cols % mxfp8_block != 0) return r.fail("{s}: {d} inputs are not whole blocks of {d}", .{ module, cols, mxfp8_block });
            out.weight = try r.part("weight", .F8_E4M3, &.{ rows, cols });
            out.scale = try r.part("weight_scale", .U8, &.{ rows, cols / mxfp8_block });
        },
        .fp8_block => {
            if (rows % fp8_block != 0 or cols % fp8_block != 0) return r.fail("{s}: [{d}, {d}] is not whole blocks of {d}", .{ module, rows, cols, fp8_block });
            out.weight = try r.part("weight", .F8_E4M3, &.{ rows, cols });
            const blocks = [2]u64{ rows / fp8_block, cols / fp8_block };
            const four = [4]u64{ blocks[0], 1, blocks[1], 1 };
            var scale = try r.get("weight_scale");
            if (scale.dtype != .F32 or !(std.mem.eql(u64, scale.shape, &blocks) or std.mem.eql(u64, scale.shape, &four)))
                return r.fail("{s}.weight_scale is {t} {any}; [{d}, {d}] makes it F32 {any}", .{ module, scale.dtype, scale.shape, rows, cols, blocks });
            scale.shape = try arena.dupe(u64, &blocks);
            out.scale = scale;
        },
        .nvfp4 => {
            if (cols % nvfp4_block != 0) return r.fail("{s}: {d} inputs are not whole blocks of {d}", .{ module, cols, nvfp4_block });
            out.weight = try r.part("weight", .U8, &.{ rows, cols / 2 });
            out.scale_2 = try r.part("weight_scale_2", .F32, &.{});
            const plane: csf_blocks.Shape = .{ .rows = rows, .cols = cols / nvfp4_block };
            if (layer.?.scale_encoding == .csf32) {
                if (!plane.fits()) return r.fail("{s}: {d} rows are not whole csf blocks of {d}", .{ module, rows, csf_blocks.block_rows });
                const exceptions = try r.get("weight_scale" ++ csf_blocks.exceptions_suffix);
                if (exceptions.dtype != .U16 or exceptions.shape.len != 1)
                    return r.fail("{s}.weight_scale{s} is {t} {any}; a vector of U16", .{ module, csf_blocks.exceptions_suffix, exceptions.dtype, exceptions.shape });
                out.csf = .{ .shape = plane, .window = .{ .row = 0, .rows = plane.rows, .col = 0, .cols = plane.cols }, .streams = .{
                    .bases = (try r.part("weight_scale" ++ csf_blocks.bases_suffix, .U8, &.{plane.rows})).data,
                    .codes = (try r.part("weight_scale" ++ csf_blocks.codes_suffix, .U8, &.{plane.codeBytes()})).data,
                    .page_bases = (try r.part("weight_scale" ++ csf_blocks.page_bases_suffix, .U32, &.{plane.pages()})).data,
                    .offsets = (try r.part("weight_scale" ++ csf_blocks.offsets_suffix, .U16, &.{plane.blocks() + plane.pages()})).data,
                    .exceptions = exceptions.data,
                } };
            } else out.scale = try r.part("weight_scale", .F8_E4M3, &.{ plane.rows, plane.cols });
            if (r.find("input_scale") != null) out.input_scale = try r.part("input_scale", .F32, &.{});
        },
    }
    return out;
}

/// A module's parts by their leaf names, each checked as `read` asks.
fn Parts(comptime Source: type, comptime Diagnostic: type) type {
    return struct {
        arena: Allocator,
        source: Source,
        module: []const u8,
        diagnostic: Diagnostic,

        const Self = @This();

        fn fail(self: Self, comptime format: []const u8, args: anytype) Error {
            self.diagnostic.set(self.arena, format, args);
            return error.UnexpectedTensor;
        }

        fn find(self: Self, leaf: []const u8) ?Tensor {
            var name: [512]u8 = undefined;
            return self.source.get(std.fmt.bufPrint(&name, "{s}.{s}", .{ self.module, leaf }) catch return null);
        }

        fn get(self: Self, leaf: []const u8) Error!Tensor {
            return self.find(leaf) orelse {
                self.diagnostic.set(self.arena, "the checkpoint has no {s}.{s}", .{ self.module, leaf });
                return error.MissingTensor;
            };
        }

        fn part(self: Self, leaf: []const u8, dtype: safetensors.DType, shape: []const u64) Error!Tensor {
            const tensor = try self.get(leaf);
            if (tensor.dtype != dtype or !std.mem.eql(u64, tensor.shape, shape))
                return self.fail("{s}.{s} is {t} {any}; the config makes it {t} {any}", .{ self.module, leaf, tensor.dtype, tensor.shape, dtype, shape });
            return tensor;
        }
    };
}

/// `linear` as `rank` of `of` holds it by `split`: the weight and a scale of two dimensions by the same half,
/// on whole scale blocks; `weight_scale_2` whole; the csf streams whole with the window narrowed. Half of the
/// rows is a slice of the stored bytes; half of the columns is copied into `arena`. A cut that is not whole
/// blocks is set in `diagnostic` and refused.
pub fn cut(arena: Allocator, linear: Linear, split: Split, rank: usize, of: usize, diagnostic: anytype) Error!Linear {
    const by: u64 = of;
    switch (split) {
        .whole => return linear,
        .drop => return linear,
        .row, .vocab => {
            // the rows a scale block covers; csf streams stay whole and any window of rows decodes (vLLM's
            // `nvfp4_csf.py` takes `tp_rank * local_intermediate` on)
            const block: u64 = if (linear.scheme == .fp8_block) fp8_block else 1;
            if (linear.rows % (by * block) != 0)
                return refuse(arena, diagnostic, "{d} rows do not halve on blocks of {d}", .{ linear.rows, block });
            var out = linear;
            out.rows = linear.rows / by;
            out.weight = try share.take(arena, linear.weight, .row, rank, of);
            if (linear.scale) |scale| out.scale = try share.take(arena, scale, .row, rank, of);
            if (linear.csf) |stored| {
                out.csf.?.window.rows = stored.window.rows / by;
                out.csf.?.window.row = stored.window.row + rank * out.csf.?.window.rows;
            }
            return out;
        },
        .col => {
            const block: u64 = switch (linear.scheme) {
                .bf16, .fp8 => 1,
                .mxfp8 => mxfp8_block,
                .fp8_block => fp8_block,
                .nvfp4 => nvfp4_block,
            };
            if (linear.cols % (by * block) != 0)
                return refuse(arena, diagnostic, "{d} inputs do not halve on blocks of {d}", .{ linear.cols, block });
            var out = linear;
            out.cols = linear.cols / by;
            out.weight = try share.take(arena, linear.weight, .col, rank, of);
            if (linear.scale) |scale| out.scale = try share.take(arena, scale, .col, rank, of);
            if (linear.csf) |stored| {
                out.csf.?.window.cols = stored.window.cols / by;
                out.csf.?.window.col = stored.window.col + rank * out.csf.?.window.cols;
            }
            return out;
        },
    }
}

fn refuse(arena: Allocator, diagnostic: anytype, comptime format: []const u8, args: anytype) Error {
    diagnostic.set(arena, format, args);
    return error.Indivisible;
}
