//! A checkpoint's layout as one rank holds it (spec 050). The tensors are `share.zig`'s slices; the config's
//! head counts and expert width are halved to match. The vocabulary stays whole: the embedding is, and only
//! `lm_head` is cut. BF16, MXFP8 and NVFP4 are sliced: a scale tensor of two dimensions is cut as its weight
//! is, an MXFP8 exponent a 32 inputs with its codes. NVFP4 scales stored
//! csf stay whole and the rank decodes its window of them at load (`layout.Csf.window`, spec 105). A tensor
//! whose cut is not a whole number of rows or columns is refused. The MTP layer's routed experts, where the
//! head is split (`sliceMtp`, spec 090), are cut here when stored NVFP4 and otherwise left whole, and cut after
//! the load has requantized them.

const std = @import("std");
const Allocator = std.mem.Allocator;
const layout = @import("layout.zig");
const safetensors = @import("../../weights/safetensors.zig");
const share_file = @import("share.zig");
const Config = @import("config.zig").Config;
const Tensor = safetensors.Tensor;

pub const Error = error{Indivisible} || Allocator.Error;

const Share = share_file.Share;
const Heads = share_file.Heads;

/// `config` with the widths a rank holds. `vocab` stays the embedding's.
pub fn halved(config: Config) Config {
    var out = config;
    out.heads /= share_file.world;
    out.kv_heads /= share_file.world;
    out.key_heads /= share_file.world;
    out.value_heads /= share_file.world;
    out.moe_width /= share_file.world;
    out.shared_width /= share_file.world;
    return out;
}

/// `found` as `share` holds it. `config` is the unsplit one (`Heads.of` reads its full head counts).
/// `load_mtp` false leaves the MTP head out.
pub fn slice(arena: Allocator, found: layout.Layout, config: Config, share: Share, load_mtp: bool) Error!layout.Layout {
    const heads = Heads.of(config);
    var out = found;
    if (found.embed) |linear| out.embed = try sliceLinear(arena, linear, share, heads);
    const layers = try arena.alloc(?layout.Layer, found.layers.len);
    for (found.layers, layers) |layer, *dest| dest.* = if (layer) |one| try sliceLayer(arena, one, share, heads) else null;
    out.layers = layers;
    if (found.mixer) |mixer| out.mixer = try sliceHyper(arena, mixer, share, heads);
    if (found.head) |linear| out.head = try sliceLinear(arena, linear, share, heads);
    out.mtp = if (load_mtp and found.mtp != null) try sliceMtp(arena, found.mtp.?, share, heads) else null;
    return out;
}

pub fn sliceLayer(arena: Allocator, layer: layout.Layer, share: Share, heads: Heads) Error!layout.Layer {
    return cutLayer(arena, layer, share, heads, false, false);
}

/// `sliceLayer` for a layer whose routed experts the device places (`model_load.loadLayer`'s `columns`): their
/// down is left whole, its columns read where they lie (`columnsOf`) rather than copied here.
pub fn sliceLayerPlaced(arena: Allocator, layer: layout.Layer, share: Share, heads: Heads) Error!layout.Layer {
    return cutLayer(arena, layer, share, heads, false, true);
}

/// A routed expert's down as the device cuts it: its inputs and its csf window the rank's columns, its codes and
/// any stored scales whole, each row's half found by its stride.
pub fn columnsOf(linear: layout.Linear, share: Share) Error!layout.Linear {
    var out = linear;
    out.cols = linear.cols / share_file.world;
    if (linear.csf) |stored| out.csf = try csfWindow(stored, linear, .col, share);
    return out;
}

fn cutLayer(arena: Allocator, layer: layout.Layer, share: Share, heads: Heads, requantized: bool, down_whole: bool) Error!layout.Layer {
    var out = layer;
    out.attn_hc = try sliceHyper(arena, layer.attn_hc, share, heads);
    out.mlp_hc = try sliceHyper(arena, layer.mlp_hc, share, heads);
    out.block = switch (layer.block) {
        .linear_attention => |dn| .{ .linear_attention = try sliceDelta(arena, dn, share, heads) },
        .full_attention => |attn| .{ .full_attention = try sliceAttn(arena, attn, share, heads) },
    };
    if (layer.ple) |branch| out.ple = try slicePle(arena, branch, share, heads);
    out.moe = try sliceMoe(arena, layer.moe, share, heads, requantized, down_whole);
    return out;
}

/// The head as `share` holds it where it is split (TensorFold's `weights.py`): the attention's heads halved,
/// the shared expert's width, the norms, `fc_*`, the read-outs and the indexer whole. Its routed experts are
/// cut here where they are stored NVFP4 and left whole in any other storage: the load requantizes those
/// whole and then takes the rank's half (`model.zig` `loadMoe`).
pub fn sliceMtp(arena: Allocator, mtp: layout.Mtp, share: Share, heads: Heads) Error!layout.Mtp {
    var out = mtp;
    out.norm_embedding = try sliceVector(arena, mtp.norm_embedding, share, heads);
    out.norm_hidden = try sliceVector(arena, mtp.norm_hidden, share, heads);
    out.fc_embedding = try sliceLinear(arena, mtp.fc_embedding, share, heads);
    out.fc_hidden = try sliceLinear(arena, mtp.fc_hidden, share, heads);
    out.layer = try cutLayer(arena, mtp.layer, share, heads, true, false);
    out.mixer = try sliceHyper(arena, mtp.mixer, share, heads);
    return out;
}

/// `sliceMtp` with its layer's NVFP4 routed experts cut as `sliceLayerPlaced` cuts a main layer's: down whole, the
/// device taking the rank's columns. Experts of another storage stay whole, as `sliceMtp` leaves them.
pub fn sliceMtpPlaced(arena: Allocator, mtp: layout.Mtp, share: Share, heads: Heads) Error!layout.Mtp {
    var out = try sliceMtp(arena, mtp, share, heads);
    out.layer = try cutLayer(arena, mtp.layer, share, heads, true, true);
    return out;
}

fn slicePle(arena: Allocator, branch: layout.Ple, share: Share, heads: Heads) Error!layout.Ple {
    var out = branch;
    out.key = try sliceLinear(arena, branch.key, share, heads);
    out.value = try sliceLinear(arena, branch.value, share, heads);
    out.norm_key = try sliceVector(arena, branch.norm_key, share, heads);
    out.norm_query = try sliceVector(arena, branch.norm_query, share, heads);
    out.norm_conv = try sliceVector(arena, branch.norm_conv, share, heads);
    out.conv = try sliceConv(arena, branch.conv, share, heads);
    return out;
}

pub fn sliceHyper(arena: Allocator, connection: layout.HyperConnection, share: Share, heads: Heads) Error!layout.HyperConnection {
    var out = connection;
    out.down = try sliceLinear(arena, connection.down, share, heads);
    if (connection.inject) |inject| out.inject = try sliceLinear(arena, inject, share, heads);
    out.up = try sliceLinear(arena, connection.up, share, heads);
    out.norm = try sliceVector(arena, connection.norm, share, heads);
    return out;
}

fn sliceDelta(arena: Allocator, dn: layout.DeltaNet, share: Share, heads: Heads) Error!layout.DeltaNet {
    return .{
        .qkv = try sliceLinear(arena, dn.qkv, share, heads),
        .z = try sliceLinear(arena, dn.z, share, heads),
        .b = try sliceLinear(arena, dn.b, share, heads),
        .a = try sliceLinear(arena, dn.a, share, heads),
        .conv = try sliceConv(arena, dn.conv, share, heads),
        .a_log = try sliceVector(arena, dn.a_log, share, heads),
        .dt_bias = try sliceVector(arena, dn.dt_bias, share, heads),
        .norm = try sliceVector(arena, dn.norm, share, heads),
        .out = try sliceLinear(arena, dn.out, share, heads),
    };
}

fn sliceAttn(arena: Allocator, attn: layout.Attention, share: Share, heads: Heads) Error!layout.Attention {
    return .{
        .q = try sliceLinear(arena, attn.q, share, heads),
        .k = try sliceLinear(arena, attn.k, share, heads),
        .v = try sliceLinear(arena, attn.v, share, heads),
        .index_qk = try sliceLinear(arena, attn.index_qk, share, heads),
        .q_norm = try sliceVector(arena, attn.q_norm, share, heads),
        .k_norm = try sliceVector(arena, attn.k_norm, share, heads),
        .index_q_norm = try sliceVector(arena, attn.index_q_norm, share, heads),
        .index_k_norm = try sliceVector(arena, attn.index_k_norm, share, heads),
        .o = try sliceLinear(arena, attn.o, share, heads),
    };
}

/// `requantized`: routed experts that are not NVFP4 stay whole (the MTP layer's, cut after their load).
fn sliceMoe(arena: Allocator, moe: layout.Moe, share: Share, heads: Heads, requantized: bool, down_whole: bool) Error!layout.Moe {
    var out = moe;
    out.router = try sliceLinear(arena, moe.router, share, heads);
    out.shared_gate = try sliceLinear(arena, moe.shared_gate, share, heads);
    out.shared = try sliceExpert(arena, moe.shared, share, heads, false);
    switch (moe.experts) {
        .each => |each| {
            const sliced = try arena.alloc(layout.Expert, each.len);
            for (each, sliced) |expert, *dest| {
                dest.* = if (requantized and expert.gate.scheme != .nvfp4) expert else try sliceExpert(arena, expert, share, heads, down_whole);
            }
            out.experts = .{ .each = sliced };
        },
        .stacked => if (!requantized) {
            std.log.err("a share reads experts one tensor each; a stacked table is not split", .{});
            return error.Indivisible;
        },
    }
    return out;
}

fn sliceExpert(arena: Allocator, expert: layout.Expert, share: Share, heads: Heads, down_whole: bool) Error!layout.Expert {
    return .{
        .gate = try sliceLinear(arena, expert.gate, share, heads),
        .up = try sliceLinear(arena, expert.up, share, heads),
        .down = if (down_whole) expert.down else try sliceLinear(arena, expert.down, share, heads),
    };
}

pub fn sliceLinear(arena: Allocator, linear: layout.Linear, share: Share, heads: Heads) Error!layout.Linear {
    if (linear.weight.shape.len == 0) return linear;
    const group = share_file.groupOf(linear.name);
    var out = linear;
    out.weight = try take(arena, linear.weight, group, share, heads, linear.name);
    if (linear.scale) |scale| {
        if (scale.shape.len <= 1) out.scale = scale else out.scale = try take(arena, scale, group, share, heads, linear.name);
    }
    if (out.weight.shape.len >= 1) out.rows = @intCast(out.weight.shape[0]);
    if (out.weight.shape.len >= 2) out.cols = @intCast(out.weight.shape[1]);
    // NVFP4 codes are two a byte: the stored columns are half the inputs
    if (linear.scheme == .nvfp4) out.cols = linear.cols / @as(u32, if (group.cut() == .col) share_file.world else 1);
    if (linear.csf) |stored| out.csf = try csfWindow(stored, linear, group.cut(), share);
    return out;
}

/// csf streams stay whole: the rank decodes its window of the scales at load, its rows of a row cut and its
/// columns of a column cut (vLLM's `CsfSource.select` for a tensor-parallel rank, the card's b12x loader).
fn csfWindow(stored: layout.Csf, linear: layout.Linear, cut: share_file.Cut, share: Share) Error!layout.Csf {
    const rows: u64 = linear.rows;
    const cols: u64 = linear.cols / 16;
    const world: u64 = share_file.world;
    var out = stored;
    out.window = switch (cut) {
        .row => if (rows % world == 0) .{ .row = share.rank * rows / world, .rows = rows / world, .col = 0, .cols = cols } else null,
        .col => if (cols % (2 * world) == 0) .{ .row = 0, .rows = rows, .col = share.rank * cols / world, .cols = cols / world } else null,
        .whole, .unread => return stored,
        .qkv => null,
    };
    if (stored.window != null or out.window == null) {
        std.log.err("{s}: its csf scales [{d}, {d}] are not cut {t}", .{ linear.name, rows, cols, cut });
        return error.Indivisible;
    }
    return out;
}

fn sliceVector(arena: Allocator, vector: layout.Vector, share: Share, heads: Heads) Error!layout.Vector {
    if (vector.tensor.shape.len == 0) return vector;
    var out = vector;
    out.tensor = try take(arena, vector.tensor, share_file.groupOf(vector.name), share, heads, vector.name);
    return out;
}

fn sliceConv(arena: Allocator, conv: layout.Conv, share: Share, heads: Heads) Error!layout.Conv {
    if (conv.tensor.shape.len == 0) return conv;
    var out = conv;
    out.tensor = try take(arena, conv.tensor, share_file.groupOf(conv.name), share, heads, conv.name);
    if (out.tensor.shape.len >= 1) out.channels = @intCast(out.tensor.shape[0]);
    return out;
}

fn take(arena: Allocator, tensor: Tensor, group: share_file.Group, share: Share, heads: Heads, name: []const u8) Error!Tensor {
    const cut = group.cut();
    const axis: usize = if (cut == .col) 1 else 0;
    if (cut != .whole and cut != .unread and cut != .qkv) {
        if (tensor.shape.len <= axis or tensor.shape[axis] % share_file.world != 0) {
            std.log.err("{s}: {d} on axis {d} does not halve", .{ name, if (tensor.shape.len > axis) tensor.shape[axis] else 0, axis });
            return error.Indivisible;
        }
    }
    return share_file.take(arena, tensor, group, share, heads);
}
