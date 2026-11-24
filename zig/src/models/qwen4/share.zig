//! What one rank of two holds of a Flash Next checkpoint (spec 050). Host only.
//!
//! TensorFold's NVFP4 loader already slices by rank (`families/qwen4_exp/cuda/weights.py`: `gdn_nvfp4`,
//! `attention_nvfp4`, `moe_nvfp4`, and `lm_head` in `load`) although the engine refuses a world of 2. A rank
//! holds half of every wide matrix and all of the embedding, the norms, the hyper-connections, the router and
//! the indexer. The n-gram table stays mapped, as on one rank, and the vision tower is not read.
//!
//! DeltaNet's `in_proj_qkv` and convolution are three segments (q, then k, then v), each halved by head, so a
//! rank's slice is a copy. An expert's `gate` and `up` halve their rows; `down` halves its columns, packed
//! NVFP4 included (two codes a byte, a scale every 16 inputs). `weight_scale_2` is one number and stays whole.

const std = @import("std");
const units = @import("../../cuda/units.zig");
const Allocator = std.mem.Allocator;
const safetensors = @import("../../weights/safetensors.zig");
const weights_share = @import("../../weights/share.zig");
const qwen4_config = @import("config.zig");
const Config = qwen4_config.Config;

/// Ranks this checkpoint is split over.
pub const world: u32 = 2;

/// One rank of `world`.
pub const Share = struct {
    rank: u32,

    pub fn of(rank: u32) Share {
        std.debug.assert(rank < world);
        return .{ .rank = rank };
    }
};

/// q and v channels of a DeltaNet layer. k has q's count: the convolution is q, then k, then v.
pub const Heads = struct {
    q: u32,
    v: u32,

    pub fn of(c: Config) Heads {
        return .{
            .q = c.key_heads * c.key_head_dim,
            .v = c.value_heads * c.value_head_dim,
        };
    }
};

/// How a tensor is cut, and the group it is reported in. `unread` is not loaded.
pub const Group = enum {
    /// `in_proj_qkv` and the convolution: half the key heads and half the value heads.
    qkv,
    /// `in_proj_z`, `in_proj_b`, `in_proj_a`, `A_log`, `dt_bias`: half the value heads, by rows.
    deltanet_rows,
    /// `out_proj`: half the value channels, by columns.
    deltanet_out,
    /// `q_proj`, `k_proj`, `v_proj`: half the heads, by rows.
    attention_rows,
    /// `o_proj`: half the heads, by columns.
    attention_out,
    /// Every expert's `gate` and `up`, the shared expert's too: half the width, by rows.
    expert_rows,
    /// Every expert's `down`: half the width, by columns.
    expert_cols,
    /// `lm_head`: half the vocabulary, by rows.
    head,
    /// The embedding, the norms, the mixes, the router, the indexer, the PLE, every `weight_scale_2`.
    whole,
    /// The vision tower, the n-gram table, an activation `input_scale`.
    unread,

    pub fn cut(self: Group) Cut {
        return switch (self) {
            .qkv => .qkv,
            .deltanet_rows, .attention_rows, .expert_rows, .head => .row,
            .deltanet_out, .attention_out, .expert_cols => .col,
            .whole => .whole,
            .unread => .unread,
        };
    }
};

pub const Cut = enum {
    whole,
    row,
    col,
    /// Three segments, each halved. The bytes a rank holds are still half.
    qkv,
    unread,

    pub fn held(self: Cut, total: u64) u64 {
        return switch (self) {
            .whole => total,
            .row, .col, .qkv => total / world,
            .unread => 0,
        };
    }
};

/// Which group `name` belongs to. A name the forward does not read is `unread` or `whole` by what the loader
/// keeps: unknown tensors of the language model are whole, so a new one is counted until it is classified.
pub fn groupOf(name: []const u8) Group {
    if (std.mem.startsWith(u8, name, "model.visual.") or std.mem.indexOf(u8, name, "ngram_embedding") != null)
        return .unread;
    if (std.mem.endsWith(u8, name, ".input_scale")) return .unread;
    if (std.mem.endsWith(u8, name, ".weight_scale_2")) return .whole;
    // A layout's name is the module (`...in_proj_qkv`); a checkpoint's is that plus `.weight`.
    if (std.mem.indexOf(u8, name, "linear_attn.in_proj_qkv") != null or std.mem.indexOf(u8, name, "linear_attn.conv1d") != null)
        return .qkv;
    if (std.mem.indexOf(u8, name, "linear_attn.out_proj") != null) return .deltanet_out;
    if (std.mem.indexOf(u8, name, "in_proj_z") != null or std.mem.indexOf(u8, name, "in_proj_b") != null or
        std.mem.indexOf(u8, name, "in_proj_a") != null or std.mem.indexOf(u8, name, "linear_attn.A_log") != null or
        std.mem.indexOf(u8, name, "linear_attn.dt_bias") != null)
        return .deltanet_rows;
    if (std.mem.indexOf(u8, name, "self_attn.o_proj") != null) return .attention_out;
    if (std.mem.indexOf(u8, name, "self_attn.q_proj") != null or std.mem.indexOf(u8, name, "self_attn.k_proj") != null or
        std.mem.indexOf(u8, name, "self_attn.v_proj") != null)
        return .attention_rows;
    if (std.mem.indexOf(u8, name, "down_proj") != null) return .expert_cols;
    if (std.mem.indexOf(u8, name, "gate_proj") != null or std.mem.indexOf(u8, name, "up_proj") != null)
        return .expert_rows;
    if (std.mem.indexOf(u8, name, "lm_head") != null) return .head;
    return .whole;
}

pub const Sums = struct {
    count: [10]u32 = @splat(0),
    bytes: [10]u64 = @splat(0),

    pub fn add(self: *Sums, group: Group, n: u64) void {
        const i = @backingInt(group);
        self.count[i] += 1;
        self.bytes[i] += n;
    }

    /// Bytes a rank holds: half of every split tensor, all of every whole one.
    pub fn rank(self: Sums) u64 {
        var n: u64 = 0;
        for (std.enums.values(Group)) |group| n += group.cut().held(self.bytes[@backingInt(group)]);
        return n;
    }

    pub fn split(self: Sums) u64 {
        var n: u64 = 0;
        for (std.enums.values(Group)) |group| {
            if (group.cut() != .whole and group.cut() != .unread) n += self.bytes[@backingInt(group)];
        }
        return n;
    }

    pub fn whole(self: Sums) u64 {
        return self.bytes[@backingInt(Group.whole)];
    }
};

/// Every tensor of `checkpoint`, classified.
pub fn sumsOf(checkpoint: *const safetensors.Checkpoint) Sums {
    var sums: Sums = .{};
    for (checkpoint.tensors.keys(), checkpoint.tensors.values()) |name, tensor| {
        sums.add(groupOf(name), tensor.data.len);
    }
    return sums;
}

/// `tensor` as `share` holds it. `heads` is read for `.qkv` only. `.unread` is the tensor unchanged: the
/// caller does not load it.
pub fn take(arena: Allocator, tensor: safetensors.Tensor, group: Group, share: Share, heads: Heads) Allocator.Error!safetensors.Tensor {
    return switch (group.cut()) {
        .whole, .unread => tensor,
        .row => weights_share.take(arena, tensor, .row, share.rank, world),
        .col => weights_share.take(arena, tensor, .col, share.rank, world),
        .qkv => takeQkv(arena, tensor, share.rank, heads),
    };
}

fn takeQkv(arena: Allocator, tensor: safetensors.Tensor, rank: u32, heads: Heads) Allocator.Error!safetensors.Tensor {
    const q: usize = heads.q;
    const v: usize = heads.v;
    const full: usize = @intCast(tensor.shape[0]);
    std.debug.assert(full == 2 * q + v and q % world == 0 and v % world == 0);
    const qh = q / world;
    const vh = v / world;
    const inner = tensor.data.len / full;
    const rows = 2 * qh + vh;
    const shape = try arena.dupe(u64, tensor.shape);
    shape[0] = rows;
    const data = try arena.alignedAlloc(u8, .@"64", rows * inner);
    const segments = [_]struct { at: usize, n: usize, src: usize }{
        .{ .at = 0, .n = qh, .src = rank * qh },
        .{ .at = qh, .n = qh, .src = q + rank * qh },
        .{ .at = 2 * qh, .n = vh, .src = 2 * q + rank * vh },
    };
    for (segments) |part| {
        @memcpy(data[part.at * inner ..][0 .. part.n * inner], tensor.data[part.src * inner ..][0 .. part.n * inner]);
    }
    return .{ .dtype = tensor.dtype, .shape = shape, .data = data };
}

/// One line a group, then the bytes a rank holds.
pub fn report(stdout: *std.Io.Writer, sums: Sums) std.Io.Writer.Error!void {
    const lines = [_]struct { group: Group, text: []const u8 }{
        .{ .group = .qkv, .text = "in_proj_qkv, conv1d: half the key heads, half the value heads" },
        .{ .group = .deltanet_rows, .text = "in_proj_z, in_proj_b, in_proj_a, A_log, dt_bias: rows" },
        .{ .group = .deltanet_out, .text = "out_proj: columns" },
        .{ .group = .attention_rows, .text = "q_proj, k_proj, v_proj: rows" },
        .{ .group = .attention_out, .text = "o_proj: columns" },
        .{ .group = .expert_rows, .text = "gate, up, shared: rows 0:320 or 320:640" },
        .{ .group = .expert_cols, .text = "down, shared down: those columns, scales with them" },
        .{ .group = .head, .text = "lm_head: half the vocabulary" },
        .{ .group = .whole, .text = "embedding, norms, mixes, router, indexer, PLE, weight_scale_2" },
        .{ .group = .unread, .text = "vision, n-gram table, input_scale: not loaded" },
    };
    for (lines) |line| {
        const i = @backingInt(line.group);
        const held = line.group.cut().held(sums.bytes[i]);
        try stdout.print("  {s}\n    {t:<6} {d:>6} tensors  {d:>8.3} GiB, a rank {d:>8.3}\n", .{
            line.text, line.group.cut(), sums.count[i], units.gib(sums.bytes[i]), units.gib(held),
        });
    }
    try stdout.print("a rank holds {d:.3} GiB ({d:.3} split in half, {d:.3} whole)\n", .{
        units.gib(sums.rank()),
        units.gib(sums.split()),
        units.gib(sums.whole()),
    });
}
