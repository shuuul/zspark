//! Where a DFlash2 drafter's weights are in its one safetensors file: every tensor, its shape, what a rank of
//! two holds of it as `families/glm5_next/cuda/dflash2.py:Drafter.__init__` slices it, how TensorFold holds
//! it after load and how zspark does. Host only. GLM and Qwen Flash Next both read this table; the kinds are
//! the architecture's, not a family's.
//!
//! WS-03's Qwen drafter imports `Layout`, `kinds`, `find`, `rule`, `Split`, `Found`, `Storage` and `Dim`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const safetensors = @import("../../weights/safetensors.zig");
const share = @import("../../weights/share.zig");
const config_file = @import("config.zig");
const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;

pub const Split = share.Split;

/// A kind's tensors as found: how many, their bytes, and a rank's.
pub const Found = struct {
    count: usize = 0,
    bytes: u64 = 0,
    rank: u64 = 0,
};

pub const Error = error{ MissingTensor, UnexpectedTensor, UnknownTensor } || Allocator.Error;

/// How a tensor is held once loaded.
pub const Storage = enum {
    /// As stored.
    bf16,
    /// Four bits a weight made at load, in memory only: TensorFold's affine 4-bit in groups of 64 with a bf16
    /// scale and bias a group (`qmm.quantize4`), zspark's NVFP4 with an e4m3 scale a group of 16
    /// (`requantize.zig`). Both are 4.5 bits a weight.
    four,
    /// fp32 on the host: the selector's codebooks, read a row at a time by the chain.
    host,

    /// The bytes `bf16_bytes` of BF16 become.
    pub fn bytes(self: Storage, bf16_bytes: u64) u64 {
        return switch (self) {
            .bf16 => bf16_bytes,
            .four => bf16_bytes * 9 / 32,
            .host => bf16_bytes * 2,
        };
    }
};

/// A dimension of a tensor, by what the config makes it.
pub const Dim = enum {
    none,
    two,
    hidden,
    vocab,
    /// `fc`'s input: the taps side by side.
    taps,
    q_inner,
    kv_inner,
    head_dim,
    intermediate,
    /// A kernel projection's outputs.
    kernel,
    selector,

    pub fn of(self: Dim, c: Config) u64 {
        return switch (self) {
            .none => 0,
            .two => 2,
            .hidden => c.hidden,
            .vocab => c.vocab,
            .taps => c.tapWidth(),
            .q_inner => c.heads * c.head_dim,
            .kv_inner => c.kv_heads * c.head_dim,
            .head_dim => c.head_dim,
            .intermediate => c.intermediate,
            .kernel => c.kernelWidth(),
            .selector => c.selector_rank,
        };
    }
};

pub const Kind = struct {
    /// The name after `layers.N.`; the whole name where `layer` is false.
    leaf: []const u8,
    dims: [3]Dim,
    split: Split,
    layer: bool,
    /// How TensorFold holds it (`dflash2.py`: what goes through `_quantize4`), and how zspark does unless the
    /// drafter is asked for as stored.
    tensorfold: Storage,
    storage: Storage,

    pub fn shape(self: Kind, c: Config, out: *[3]u64) []const u64 {
        var n: usize = 0;
        for (self.dims) |dim| if (dim != .none) {
            out[n] = dim.of(c);
            n += 1;
        };
        return out[0..n];
    }
};

fn kind(leaf: []const u8, dims: [3]Dim, split: Split, layer: bool, tensorfold: Storage, storage: Storage) Kind {
    return .{ .leaf = leaf, .dims = dims, .split = split, .layer = layer, .tensorfold = tensorfold, .storage = storage };
}

/// Every tensor of the drafter, all BF16 in the file. What zspark keeps as stored where TensorFold holds four
/// bits: the context's K and V (a context row's stay in the cache for the window's 2,047 positions), the kernel
/// projections (their output multiplies the activations, tap by tap) and nothing of size; `fc`, the queries,
/// both partials' projections and the MLP are nine tenths of a rank's bytes and are read whole each pass.
pub const kinds = [_]Kind{
    kind("fc.weight", .{ .hidden, .taps, .none }, .whole, false, .four, .four),
    kind("hidden_norm.weight", .{ .hidden, .none, .none }, .whole, false, .bf16, .bf16),
    kind("norm.weight", .{ .hidden, .none, .none }, .whole, false, .bf16, .bf16),
    kind("candidate_selector.hidden_projection.weight", .{ .selector, .hidden, .none }, .whole, false, .bf16, .bf16),
    kind("candidate_selector.predecessor_codebook", .{ .vocab, .selector, .none }, .whole, false, .host, .host),
    kind("candidate_selector.successor_codebook", .{ .vocab, .selector, .none }, .whole, false, .host, .host),
    kind("input_layernorm.weight", .{ .hidden, .none, .none }, .whole, true, .bf16, .bf16),
    kind("post_attention_layernorm.weight", .{ .hidden, .none, .none }, .whole, true, .bf16, .bf16),
    // the two dynamic convolutions: [branch, tap, hidden] and the projection that adds to it a row
    kind("attention_conv.base_kernel", .{ .two, .two, .hidden }, .whole, true, .bf16, .bf16),
    kind("attention_conv.kernel_projection.weight", .{ .kernel, .hidden, .none }, .whole, true, .four, .bf16),
    kind("mlp_conv.base_kernel", .{ .two, .two, .hidden }, .whole, true, .bf16, .bf16),
    kind("mlp_conv.kernel_projection.weight", .{ .kernel, .hidden, .none }, .whole, true, .four, .bf16),
    // attention: a rank has half the query heads and half the KV heads
    kind("self_attn.q_proj.weight", .{ .q_inner, .hidden, .none }, .row, true, .four, .four),
    kind("self_attn.k_proj.weight", .{ .kv_inner, .hidden, .none }, .row, true, .four, .bf16),
    kind("self_attn.v_proj.weight", .{ .kv_inner, .hidden, .none }, .row, true, .four, .bf16),
    kind("self_attn.q_norm.weight", .{ .head_dim, .none, .none }, .whole, true, .bf16, .bf16),
    kind("self_attn.k_norm.weight", .{ .head_dim, .none, .none }, .whole, true, .bf16, .bf16),
    kind("self_attn.o_proj.weight", .{ .hidden, .q_inner, .none }, .col, true, .four, .four),
    kind("mlp.gate_proj.weight", .{ .intermediate, .hidden, .none }, .row, true, .four, .four),
    kind("mlp.up_proj.weight", .{ .intermediate, .hidden, .none }, .row, true, .four, .four),
    kind("mlp.down_proj.weight", .{ .hidden, .intermediate, .none }, .col, true, .four, .four),
};

/// The entry of `kinds` a tensor's name is, by its name alone; null where the drafter has no such tensor.
pub fn find(name: []const u8) ?usize {
    var leaf = name;
    const layers = "layers.";
    const in_layer = std.mem.startsWith(u8, name, layers);
    if (in_layer) {
        const rest = name[layers.len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        _ = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
        leaf = rest[dot + 1 ..];
    }
    for (kinds, 0..) |one, at| if (one.layer == in_layer and std.mem.eql(u8, one.leaf, leaf)) return at;
    return null;
}

/// What a rank holds of the tensor `name`.
pub fn rule(name: []const u8) ?Split {
    return kinds[find(name) orelse return null].split;
}

pub const Layout = struct {
    /// One a `kinds` entry: its tensors, their bytes and a rank's, as stored.
    found: []Found,
    tensors: usize = 0,
    bytes: u64 = 0,
    rank: u64 = 0,

    /// Finds every tensor the config asks for in `checkpoint`, BF16 and of its shape, and refuses one it did
    /// not ask for. The result lives in `arena`.
    pub fn init(arena: Allocator, c: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic) Error!Layout {
        var self: Layout = .{ .found = try arena.alloc(Found, kinds.len) };
        @memset(self.found, .{});
        var name: [128]u8 = undefined;
        for (kinds, self.found) |one, *found| {
            if (!one.layer) {
                try self.take(arena, c, checkpoint, diagnostic, one, found, one.leaf);
            } else for (0..c.layers) |layer| {
                try self.take(arena, c, checkpoint, diagnostic, one, found, std.fmt.bufPrint(&name, "layers.{d}.{s}", .{ layer, one.leaf }) catch unreachable);
            }
        }
        if (checkpoint.tensors.count() != self.tensors) {
            var it = checkpoint.tensors.iterator();
            while (it.next()) |entry| if (find(entry.key_ptr.*) == null) {
                diagnostic.set(arena, "the drafter's file holds {s}, which the DFlash2 layout has no place for", .{entry.key_ptr.*});
                return error.UnknownTensor;
            };
            diagnostic.set(arena, "the drafter's file holds {d} tensors where its config asks for {d}", .{ checkpoint.tensors.count(), self.tensors });
            return error.UnknownTensor;
        }
        return self;
    }

    fn take(self: *Layout, arena: Allocator, c: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic, one: Kind, found: *Found, name: []const u8) Error!void {
        const tensor = checkpoint.get(name) orelse {
            diagnostic.set(arena, "the drafter's file has no {s}", .{name});
            return error.MissingTensor;
        };
        var dims: [3]u64 = undefined;
        const shape = one.shape(c.*, &dims);
        if (tensor.dtype != .BF16 or !std.mem.eql(u64, tensor.shape, shape)) {
            diagnostic.set(arena, "{s} is {t} {any} where the drafter's config makes it BF16 {any}", .{ name, tensor.dtype, tensor.shape, shape });
            return error.UnexpectedTensor;
        }
        const rank = one.split.share(tensor.data.len);
        found.count += 1;
        found.bytes += tensor.data.len;
        found.rank += rank;
        self.tensors += 1;
        self.bytes += tensor.data.len;
        self.rank += rank;
    }

    /// A rank's bytes as TensorFold holds the drafter (`tensorfold` true) or as zspark does; `stored`: every
    /// device tensor as the file has it. The codebooks' host bytes are not among them: `host`.
    pub fn device(self: Layout, tensorfold: bool, stored: bool) u64 {
        var sum: u64 = 0;
        for (kinds, self.found) |one, found| {
            const storage = if (tensorfold) one.tensorfold else one.storage;
            if (storage == .host) continue;
            sum += if (stored) found.rank else storage.bytes(found.rank);
        }
        return sum;
    }

    pub fn host(self: Layout) u64 {
        var sum: u64 = 0;
        for (kinds, self.found) |one, found| {
            if (one.storage == .host) sum += one.storage.bytes(found.rank);
        }
        return sum;
    }
};
