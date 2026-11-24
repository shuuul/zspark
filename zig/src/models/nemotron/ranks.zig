//! What one rank of two holds of Nemotron-3.5-Transcribe's backbone (spec 151 WS-06), and how the split is
//! cut. The scheme is TensorFold's (`families/nemotron_h/cuda/tp.py`, python-0.6 `1b31cc74`): every wide
//! projection halves its heads, its Mamba heads and groups, or every expert's width, and `lm_head` halves its
//! vocabulary rows; the embeddings, every norm, the routers and `e_score_correction_bias`, `eh_proj` and the
//! MTP block's concat norms stay whole on both ranks. A rank computes its half of every layer's product as an
//! fp32 partial; the partials are gathered and added rank 0 first, rounded once into the residual
//! (`common/rank_sum.cu`), so a row's bits are the serial model's (`ranks_forward.zig`).
//!
//! Per layer, the cut is:
//!
//! - attention: q, k and v each by this rank's half of the heads (rows of the stored projection), o_proj by
//!   the same half of its input columns.
//! - mamba: `in_proj`'s row segments — z by heads, the conv channels x, then each group's B and C by its half
//!   of the groups, dt by heads — `out_proj` by the same half of its input columns; the conv's taps, bias and
//!   the group norm's weights sliced by the same channel and head indices.
//! - moe: every routed expert's `up_proj` rows and `down_proj` columns halved (`width` / 2 a rank); the one
//!   shared expert the same, its two pseudo-experts each halving width, so a rank keeps 2 pseudo-experts of
//!   `width` / 2. The router and bias stay whole (every rank routes every row and takes the same picks).
//! - `lm_head` (and the draft head, which is `lm_head` itself): half the vocabulary's rows.
//! - the MTP block: its attention and MoE split as the backbone's; `eh_proj`, `enorm`, `hnorm`, the two layer
//!   norms and `final_layernorm` whole.
//!
//! TensorFold keeps a config of the whole model and slices the weights; zspark's ranks instead hold
//! rank-shaped `Shape`s (halved `heads`, `kv_heads`, `mamba_heads`, `groups`, expert `width`) so the same
//! kernels run a rank's widths — a rank's state (KV caches, Mamba ssm) is its half's own, which is what makes
//! each rank hold about half the weights and half the per-sequence state.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const units = @import("../../cuda/units.zig");
const config_file = @import("config.zig");
const mamba_file = @import("mamba.zig");
const attention_file = @import("attention.zig");
const moe_file = @import("moe.zig");
const modelopt = @import("../../weights/modelopt.zig");
const safetensors = @import("../../weights/safetensors.zig");
const share_file = @import("../../weights/share.zig");

pub const Error = modelopt.Error || error{UnsupportedShape} || Allocator.Error;

/// Ranks this checkpoint is split over.
pub const world: usize = share_file.ranks;

/// One rank of `world`.
pub const Share = struct {
    rank: usize,

    pub fn of(rank: usize) Share {
        std.debug.assert(rank < world);
        return .{ .rank = rank };
    }
};

/// The shapes a rank's mixers run: the backbone's halved over `share`'s heads and groups.
pub const Shapes = struct {
    mamba: mamba_file.Shape,
    attention: attention_file.Shape,
    moe: moe_file.Shape,

    /// `b` the whole model's backbone; null `share` the whole shapes.
    pub fn of(b: config_file.Backbone, share: ?Share) Shapes {
        const by: usize = if (share) |_| world else 1;
        if (b.heads % by != 0 or b.kv_heads % by != 0 or b.mamba_heads % by != 0 or b.groups % by != 0 or b.moe_width % by != 0 or b.vocab % by != 0)
            unreachable; // refused at the load's `refuse`
        return .{
            .mamba = .{ .xd = b.mambaInner() / by, .heads = b.mamba_heads / by, .head_dim = b.mamba_head_dim, .groups = b.groups / by, .state = b.ssm_state },
            .attention = .{ .hidden = b.hidden, .heads = b.heads / by, .kv_heads = b.kv_heads / by, .head_dim = b.head_dim, .eps = b.eps },
            .moe = .{ .dims = b.hidden, .routed = b.experts, .width = b.moe_width / by, .top = b.top_k, .scale = b.routed_scale, .norm = b.norm_topk, .eps = b.eps },
        };
    }
};

/// Whether the config's heads, groups, expert width and vocabulary halve evenly; `diagnostic` names the one
/// that does not.
pub fn divisible(arena: Allocator, b: config_file.Backbone, diagnostic: anytype) Error!void {
    const fields = .{
        .{ "num_attention_heads", b.heads },       .{ "num_key_value_heads", b.kv_heads },
        .{ "mamba_num_heads", b.mamba_heads },     .{ "n_groups", b.groups },
        .{ "moe_intermediate_size", b.moe_width }, .{ "vocab_size", b.vocab },
    };
    inline for (fields) |field| {
        if (field[1] % world != 0) {
            diagnostic.set(arena, "{s} is {d}: zspark's two-rank split asks an even count", .{ field[0], field[1] });
            return error.Indivisible;
        }
    }
    // a column half is whole NVFP4 blocks of 16 inputs (out_proj of mambaInner, o_proj of heads * head_dim,
    // down of moe_width, shared down of shared_width)
    const cols = .{ b.mambaInner(), b.heads * b.head_dim, b.moe_width, b.shared_width };
    const names = .{ "mamba out_proj", "o_proj", "the experts' down_proj", "the shared down_proj" };
    inline for (cols, names) |width, name| {
        if (width % (world * modelopt.nvfp4_block) != 0) {
            diagnostic.set(arena, "{s} input width {d} does not halve on NVFP4 blocks of {d}", .{ name, width, modelopt.nvfp4_block });
            return error.Indivisible;
        }
    }
}

/// The Mamba layer's `in_proj` row indices a rank holds: z's head channels, the conv channels (x by heads,
/// then each group's B and C by its half of the groups), dt by heads. Fills `rows` (of `s.proj()` /
/// `world`), returns it.
pub fn mambaInRows(b: config_file.Backbone, s: mamba_file.Shape, share: Share, rows: []usize) []usize {
    const full: mamba_file.Shape = .of(b);

    const rank = share.rank;
    const xd_half = s.xd; // this rank's x/z width
    const grp = s.state; // a group's B or C width (unchanged)
    var at: usize = 0;
    // z, then x: this rank's heads' channels of the full xd
    for (0..2) |part| {
        const base = part * full.xd + rank * xd_half;
        for (0..xd_half) |i| {
            rows[at] = base + i;
            at += 1;
        }
    }
    // each group's B, then each group's C: this rank's groups
    for (0..2) |part| {
        const base = 2 * full.xd + part * full.groups * s.state + rank * s.groups * grp;
        for (0..s.groups * grp) |i| {
            rows[at] = base + i;
            at += 1;
        }
    }
    // dt, a head
    const dt = 2 * full.xd + 2 * full.groups * s.state;
    for (0..s.heads) |i| {
        rows[at] = dt + rank * s.heads + i;
        at += 1;
    }
    std.debug.assert(at == rows.len and at == s.proj());
    return rows;
}

/// The head indices a rank holds (`s.heads` of them): for A_log, D, dt_bias.
pub fn mambaHeadRows(s: mamba_file.Shape, share: Share, rows: []usize) []usize {
    for (0..s.heads) |i| rows[i] = share.rank * s.heads + i;
    return rows[0..s.heads];
}

/// The group-norm weight's row indices a rank holds: the rank's heads' `head_dim` channels of xd (for
/// `mixer.norm.weight`, [xd]).
pub fn mambaNormRows(s: mamba_file.Shape, share: Share, rows: []usize) []usize {
    for (0..s.xd) |i| rows[i] = share.rank * s.xd + i;
    return rows[0..s.xd];
}

/// The Mamba conv channel indices a rank holds (x, the groups' B, the groups' C): `s.cd()` of them, the same
/// segments of `mambaInRows` after the z part.
pub fn mambaConvCols(b: config_file.Backbone, s: mamba_file.Shape, share: Share, cols: []usize) []usize {
    const full: mamba_file.Shape = .of(b);
    var at: usize = 0;
    for (0..s.xd) |i| {
        cols[at] = share.rank * s.xd + i;
        at += 1;
    }
    for (0..2) |part| {
        const base = full.xd + part * full.groups * s.state + share.rank * s.groups * s.state;
        for (0..s.groups * s.state) |i| {
            cols[at] = base + i;
            at += 1;
        }
    }
    std.debug.assert(at == cols.len and at == s.cd());
    return cols;
}

/// A bf16 vector `t` ([n] or [n, 1] bytes as u16) sliced to the indices `rows`.
pub fn bf16Rows(arena: Allocator, t: safetensors.Tensor, rows: []const usize) Error!safetensors.Tensor {
    if (t.dtype != .BF16) return error.UnexpectedTensor;
    const n: usize = @intCast(t.count());
    for (rows) |r| if (r >= n) return error.UnexpectedTensor;
    const data = try arena.alignedAlloc(u8, .@"64", rows.len * 2);
    for (rows, 0..) |r, at| @memcpy(data[at * 2 ..][0..2], t.data[r * 2 ..][0..2]);
    const shape = try arena.dupe(u64, t.shape);
    shape[0] = rows.len;
    return .{ .dtype = t.dtype, .shape = shape, .data = data };
}

/// A stored `Linear`'s rows gathered to `idx` (in arbitrary order, duplicates allowed): the weight's rows,
/// and a two-dimensional scale's. csf-encoded NVFP4 scales decode to the plain plane first (a gather
/// reorders what a window cannot name); `scale_2` and `input_scale` keep as they are.
pub fn linearRows(arena: Allocator, linear: modelopt.Linear, idx: []const usize, diagnostic: *config_file.Diagnostic) Error!modelopt.Linear {
    var out = linear;
    const wcols: usize = @intCast(linear.weight.shape[1]);
    const w = try arena.alignedAlloc(u8, .@"64", idx.len * wcols);
    for (idx, 0..) |r, at| {
        if (r >= linear.rows) {
            diagnostic.set(arena, "linearRows: row {d} of {d}", .{ r, linear.rows });
            return error.UnexpectedTensor;
        }
        @memcpy(w[at * wcols ..][0..wcols], linear.weight.data[r * wcols ..][0..wcols]);
    }
    const shape = try arena.dupe(u64, linear.weight.shape);
    shape[0] = idx.len;
    out.weight = .{ .dtype = linear.weight.dtype, .shape = shape, .data = w };
    out.rows = idx.len;
    switch (linear.scheme) {
        .nvfp4 => {
            const scales = linear.blockScales(arena) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.UnexpectedTensor,
            };
            const scols: usize = @intCast(linear.cols / modelopt.nvfp4_block);
            const s = try arena.alignedAlloc(u8, .@"64", idx.len * scols);
            for (idx, 0..) |r, at| @memcpy(s[at * scols ..][0..scols], scales[r * scols ..][0..scols]);
            out.scale = .{ .dtype = .F8_E4M3, .shape = &.{ idx.len, scols }, .data = s };
            out.csf = null;
        },
        .mxfp8 => {
            if (linear.scale) |scale| {
                if (idx.len % modelopt.mxfp8_block != 0) {
                    diagnostic.set(arena, "linearRows: MXFP8 gathers row blocks of {d}", .{modelopt.mxfp8_block});
                    return error.UnsupportedShape;
                }
                const scols: usize = @intCast(scale.shape[1]);
                const s = try arena.alignedAlloc(u8, .@"64", (idx.len / modelopt.mxfp8_block) * scols);
                for (0..idx.len / modelopt.mxfp8_block) |g| {
                    const src = idx[g * modelopt.mxfp8_block] / modelopt.mxfp8_block;
                    @memcpy(s[g * scols ..][0..scols], scale.data[src * scols ..][0..scols]);
                }
                const sshape = try arena.dupe(u64, scale.shape);
                sshape[0] = idx.len / modelopt.mxfp8_block;
                out.scale = .{ .dtype = scale.dtype, .shape = sshape, .data = s };
            }
        },
        .fp8_block => {
            if (linear.scale) |scale| {
                if (idx.len % modelopt.fp8_block != 0) {
                    diagnostic.set(arena, "linearRows: FP8_PB_WO gathers row blocks of {d}", .{modelopt.fp8_block});
                    return error.UnsupportedShape;
                }
                const scols: usize = @intCast(scale.shape[1]);
                const s = try arena.alignedAlloc(u8, .@"64", (idx.len / modelopt.fp8_block) * scols);
                for (0..idx.len / modelopt.fp8_block) |g| {
                    const src = idx[g * modelopt.fp8_block] / modelopt.fp8_block;
                    @memcpy(s[g * scols ..][0..scols], scale.data[src * scols ..][0..scols]);
                }
                const sshape = try arena.dupe(u64, scale.shape);
                sshape[0] = idx.len / modelopt.fp8_block;
                out.scale = .{ .dtype = scale.dtype, .shape = sshape, .data = s };
            }
        },
        else => {},
    }
    return out;
}

/// A stored `Linear`'s columns gathered to `idx` in units of `block` input channels: NVFP4's code bytes
/// come in blocks of 16 channels (8 bytes), its scale one byte a block; FP8/BF16 gather per column;
/// MXFP8 blocks of 32. csf scales decode to the plane first as `linearRows`'s.
pub fn linearCols(arena: Allocator, linear: modelopt.Linear, idx: []const usize, block: usize, diagnostic: *config_file.Diagnostic) Error!modelopt.Linear {
    var out = linear;
    const n: usize = @intCast(linear.rows);
    const wcols: usize = @intCast(linear.weight.shape[1]);
    switch (linear.scheme) {
        .nvfp4 => {
            if (block != modelopt.nvfp4_block) {
                diagnostic.set(arena, "linearCols: NVFP4 gathers by {d} channels", .{modelopt.nvfp4_block});
                return error.UnsupportedShape;
            }
            const scales = linear.blockScales(arena) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.UnexpectedTensor,
            };
            const kept = idx.len * (modelopt.nvfp4_block / 2);
            const w = try arena.alignedAlloc(u8, .@"64", n * kept);
            const s = try arena.alignedAlloc(u8, .@"64", n * idx.len);
            for (0..n) |r| {
                for (idx, 0..) |b, at| {
                    @memcpy(w[r * kept + at * 8 ..][0..8], linear.weight.data[r * wcols + b * 8 ..][0..8]);
                    s[r * idx.len + at] = scales[r * (wcols / 8) + b];
                }
            }
            const wshape = try arena.dupe(u64, linear.weight.shape);
            wshape[1] = kept;
            out.weight = .{ .dtype = linear.weight.dtype, .shape = wshape, .data = w };
            out.scale = .{ .dtype = .F8_E4M3, .shape = &.{ n, idx.len }, .data = s };
            out.csf = null;
        },
        .fp8, .bf16 => {
            if (block != 1) {
                diagnostic.set(arena, "linearCols: FP8/BF16 gather per column", .{});
                return error.UnsupportedShape;
            }
            const each: usize = if (linear.scheme == .bf16) 2 else 1; // bytes per column
            const w = try arena.alignedAlloc(u8, .@"64", n * idx.len * each);
            for (0..n) |r| {
                for (idx, 0..) |c, at| {
                    @memcpy(w[(r * idx.len + at) * each ..][0..each], linear.weight.data[r * wcols + c * each ..][0..each]);
                }
            }
            const wshape = try arena.dupe(u64, linear.weight.shape);
            wshape[1] = idx.len * each;
            out.weight = .{ .dtype = linear.weight.dtype, .shape = wshape, .data = w };
        },
        .mxfp8 => {
            if (block != modelopt.mxfp8_block) {
                diagnostic.set(arena, "linearCols: MXFP8 gathers by {d} channels", .{modelopt.mxfp8_block});
                return error.UnsupportedShape;
            }
            const each = modelopt.mxfp8_block;
            const w = try arena.alignedAlloc(u8, .@"64", n * idx.len * each);
            const s = try arena.alignedAlloc(u8, .@"64", n * idx.len);
            for (0..n) |r| {
                for (idx, 0..) |b, at| {
                    @memcpy(w[(r * idx.len + at) * each ..][0..each], linear.weight.data[r * wcols + b * each ..][0..each]);
                    if (linear.scale) |scale| s[r * idx.len + at] = scale.data[r * (wcols / each) + b];
                }
            }
            const wshape = try arena.dupe(u64, linear.weight.shape);
            wshape[1] = idx.len * each;
            out.weight = .{ .dtype = linear.weight.dtype, .shape = wshape, .data = w };
            if (linear.scale) |scale| out.scale = .{ .dtype = scale.dtype, .shape = &.{ n, idx.len }, .data = s };
        },
        else => {
            diagnostic.set(arena, "linearCols: unsupported scheme {t}", .{linear.scheme});
            return error.UnsupportedShape;
        },
    }
    out.cols = idx.len * block;
    return out;
}

/// How a tensor is cut for the byte account (`report`). Host only.
pub const Group = enum {
    /// q/k/v, in_proj, up_proj, router — rows of the stored matrix (the router's are whole, counted here not).
    rows,
    /// o_proj, out_proj, down_proj — columns.
    cols,
    /// `lm_head`'s vocabulary rows.
    head,
    /// Embeddings, norms, gates, biases, eh_proj, `weight_scale_2`, `weight_scale` scalars.
    whole,
    /// Not read: the CTC heads, the diarization encoder under this manifest, `input_scale` parts.
    unread,

    pub fn cut(self: Group) Cut {
        return switch (self) {
            .rows, .head => .row,
            .cols => .col,
            .whole => .whole,
            .unread => .unread,
        };
    }
};

pub const Cut = enum {
    whole,
    row,
    col,
    unread,

    pub fn held(self: Cut, total: u64) u64 {
        return switch (self) {
            .whole => total,
            .row, .col => total / world,
            .unread => 0,
        };
    }
};

/// Which group `name` (a checkpoint tensor's) belongs to. Norms, embeddings, routers, scalars and the MTP
/// block's concat tensors are whole; the wide projections split by their axis.
pub fn groupOf(name: []const u8) Group {
    if (std.mem.startsWith(u8, name, "ctc_timestamp.") or std.mem.startsWith(u8, name, "perception.") or
        std.mem.endsWith(u8, name, ".input_scale")) return .unread;
    if (std.mem.endsWith(u8, name, ".weight_scale_2") or std.mem.endsWith(u8, name, ".weight_scale") or
        std.mem.indexOf(u8, name, "csf_") != null) return .whole;
    if (std.mem.indexOf(u8, name, "lm_head") != null) return .head;
    if (std.mem.indexOf(u8, name, "out_proj") != null or std.mem.indexOf(u8, name, "o_proj") != null or
        std.mem.indexOf(u8, name, "down_proj") != null) return .cols;
    if (std.mem.indexOf(u8, name, "in_proj") != null or std.mem.indexOf(u8, name, "q_proj") != null or
        std.mem.indexOf(u8, name, "k_proj") != null or std.mem.indexOf(u8, name, "v_proj") != null or
        std.mem.indexOf(u8, name, "up_proj") != null) return .rows;
    return .whole;
}

pub const Sums = struct {
    count: [5]u32 = @splat(0),
    bytes: [5]u64 = @splat(0),

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
            if (group.cut() == .row or group.cut() == .col) n += self.bytes[@backingInt(group)];
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

/// One line a group, then the bytes a rank holds.
pub fn report(stdout: *std.Io.Writer, sums: Sums) std.Io.Writer.Error!void {
    const lines = [_]struct { group: Group, text: []const u8 }{
        .{ .group = .rows, .text = "q/k/v, in_proj, up_proj, conv and head vectors: rows, a rank's half" },
        .{ .group = .cols, .text = "o_proj, out_proj, down_proj: columns" },
        .{ .group = .head, .text = "lm_head: half the vocabulary" },
        .{ .group = .whole, .text = "embeddings, norms, routers, eh_proj, scalar scales" },
        .{ .group = .unread, .text = "perception, CTC, input_scale: not split here" },
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
