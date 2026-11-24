//! `profile_test.zig`'s cases (spec 104 WS-06 moved them out of `profile_test.zig`): Flash Next's sizes, the kinds
//! of launch and `cases`, each kind at the shapes a traced round launched it at. How a case is built, launched
//! and timed stays in `profile_test.zig`.

const std = @import("std");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const gdn = @import("../../../../src/models/qwen4/blocks/gdn.zig");

/// Flash Next's sizes (its checkpoints' `config.json`).
pub const flash = struct {
    pub const dims = 2560;
    pub const streams = 4;
    pub const wide = streams * dims;
    pub const low = 320;
    pub const routed = 512;
    pub const top = 10;
    pub const slots = top + 1;
    pub const width = 640;
    pub const vocab = 248320;
    pub const eps = 1e-6;
    pub const delta: gdn.Shape = .{};
    pub const attending: attention.Shape = .{ .dims = dims, .heads = 24, .kv_heads = 2, .head_dim = 256, .index_heads = 4, .index_dim = 128, .rotary_dim = 64, .rope_theta = 1e7, .budget = 2048, .ratio = 4, .eps = eps };
    pub const ple_heads = 8;
    pub const ple_dims = dims / ple_heads;
    pub const ple_taps = 4;
    pub const ple_dilation = 3;
};

const Kind = enum {
    embed,
    /// `zs_qwen4_hc_writeback`: the squares alone, a bf16 branch, the MoE's slots with the shared expert in y's
    /// last slot or in a buffer of its own.
    writeback_none,
    writeback_branch,
    writeback_slots,
    writeback_shared,
    normed,
    /// `zs_qwen4_hc_site`: the write-back of a branch, or of the slots, and the norm in one launch
    /// (`ZSPARK_HC_JOINED`); `pair_*`: the two launches it stands for.
    site_branch,
    site_slots,
    pair_branch,
    pair_slots,
    /// The slots already summed (`hc.Pending.sum`): an fp32 row in place of 11 slots.
    site_sum,
    pair_sum,
    after_site_sum,
    /// The same after a memory-bound launch (`mix` over a prompt chunk's rows, 12.5 MB), as in a round, where
    /// the launch before a site is a linear's: what the launch before leaves a small kernel is part of its time.
    after_site_branch,
    after_site_slots,
    after_pair_branch,
    after_pair_slots,
    /// The down projection as 32 K slices (`act_kernel` sums them).
    act,
    mix,
    /// `zs_qwen4_gdn_chain`, or the wide chain's three launches from `gdn.wide_rows` rows.
    chain,
    replay,
    shift,
    /// A round's 36 windows shifted by one launch (`state.once`): against 36 of `shift`.
    shift_each,
    /// The router's logits as its 8 K slices.
    topk,
    shared_act,
    prep,
    pool,
    /// The products' GEMM and `zs_qwen4_attn_select`.
    select,
    attend,
    /// `attend` behind a `mix` over 512 rows, a memory-bound neighbour.
    after_attend,
    /// `attend` as a served step launches it: sized for the graph's bucket, the lists there, the rows dense.
    served,
    gate,
    ple_embed,
    ple_gate,
    ple_conv,
    rmsnorm,
    add_streams,
    pick,
    plan,
    columns,
};

pub fn isSite(kind: Kind) bool {
    return switch (kind) {
        .site_branch, .site_slots, .site_sum, .after_site_branch, .after_site_slots, .after_site_sum => true,
        else => false,
    };
}

pub fn isBranch(kind: Kind) bool {
    return switch (kind) {
        .site_branch, .pair_branch, .after_site_branch, .after_pair_branch => true,
        else => false,
    };
}

pub fn isSum(kind: Kind) bool {
    return switch (kind) {
        .site_sum, .pair_sum, .after_site_sum => true,
        else => false,
    };
}

/// Flash Next's DeltaNet layers, whose windows a round's commit shifts.
pub const shift_layers = 36;

pub fn isAfter(kind: Kind) bool {
    return switch (kind) {
        .after_site_branch, .after_site_slots, .after_site_sum, .after_pair_branch, .after_pair_slots => true,
        else => false,
    };
}

/// K slices a read-out down projection and a router leave unsummed (a traced round's).
pub const act_slices = 32;
pub const topk_slices = 8;
pub const filler_rows = 512;
/// The smallest context a captured graph is sized for (`Graphs.bucket`).
const served_bucket = 8192;

pub const Case = struct {
    kind: Kind,
    rows: usize,
    /// The attention's: positions the caches hold at the window's end.
    context: usize = 0,
    /// The caches' storage.
    kv: attention.Storage = .bf16,
    /// Left out of the smoke.
    heavy: bool = false,

    /// What the launch is sized for: a served step's is its graph's bucket.
    pub fn sized(self: Case) usize {
        return if (self.kind == .served) @max(self.context, served_bucket) else self.context;
    }

    pub fn name(self: Case, buffer: []u8) []const u8 {
        if (self.context == 0) return std.fmt.bufPrint(buffer, "{t}:{d}", .{ self.kind, self.rows }) catch unreachable;
        if (self.kv == .bf16) return std.fmt.bufPrint(buffer, "{t}:{d}:{d}", .{ self.kind, self.rows, self.context }) catch unreachable;
        return std.fmt.bufPrint(buffer, "{t}:{d}:{d}:{t}", .{ self.kind, self.rows, self.context, self.kv }) catch unreachable;
    }
};

pub const cases = blk: {
    @setEvalBranchQuota(100_000);
    const row_kinds = [_]Kind{ .embed, .writeback_none, .writeback_branch, .writeback_slots, .writeback_shared, .normed, .site_branch, .site_slots, .pair_branch, .pair_slots, .after_site_branch, .after_site_slots, .after_pair_branch, .after_pair_slots, .site_sum, .pair_sum, .after_site_sum, .act, .mix, .chain, .replay, .shift, .shift_each, .topk, .shared_act, .gate, .ple_embed, .ple_gate, .ple_conv, .rmsnorm, .add_streams, .plan, .columns };
    const counts = [_]usize{ 1, 2, 3, 4, 6, 8, 16, 512 };
    const attending = [_]Kind{ .prep, .pool, .select, .attend, .after_attend };
    const contexts = [_]usize{ 512, 2048, 4096, 16384 };
    var list: [row_kinds.len * counts.len + attending.len * counts.len * contexts.len + 44 + 96]Case = undefined;
    var at: usize = 0;
    for (row_kinds) |kind| {
        for (counts) |rows| {
            // a replay is of a window's prefix; the MTP head and the pick run a row a draft step
            if (kind == .replay and rows == 512) continue;
            if ((kind == .rmsnorm or kind == .add_streams) and rows > 1) continue;
            list[at] = .{ .kind = kind, .rows = rows, .heavy = rows == 512 };
            at += 1;
        }
    }
    for (attending) |kind| {
        for (contexts) |context| {
            for (counts) |rows| {
                if (rows > context) continue;
                // a row selects once it has more complete blocks than it attends
                if (kind == .select and !flash.attending.selects(context)) continue;
                if (kind == .after_attend and (rows > 16 or context > 4096)) continue;
                list[at] = .{ .kind = kind, .rows = rows, .context = context, .heavy = rows == 512 or context > 4096 };
                at += 1;
            }
        }
    }
    for ([_]attention.Storage{ .fp8, .fp8_nvfp4 }) |kv| {
        for ([_]Kind{ .prep, .attend }) |kind| {
            list[at] = .{ .kind = kind, .rows = 1, .context = 4096, .kv = kv };
            at += 1;
        }
        for ([_]usize{ 2, 3, 4, 6, 8, 16, 512 }) |rows| {
            list[at] = .{ .kind = .attend, .rows = rows, .context = 4096, .kv = kv, .heavy = rows == 512 };
            at += 1;
        }
        for ([_]usize{ 1, 8, 16 }) |rows| {
            list[at] = .{ .kind = .attend, .rows = rows, .context = 512, .kv = kv };
            at += 1;
        }
        // a prompt window's rows over a 16k prompt's lists (spec 105 WS-05), and a reply's (WS-06)
        for ([_]usize{ 1, 4, 16, 32, 64, 128, 512, 2048 }) |rows| {
            list[at] = .{ .kind = .attend, .rows = rows, .context = 16384, .kv = kv, .heavy = true };
            at += 1;
        }
    }
    for ([_]attention.Storage{ .bf16, .fp8, .fp8_nvfp4 }) |kv| {
        for ([_]usize{ 256, 512, 1024, 4096 }) |context| {
            for ([_]usize{ 1, 2, 3, 4, 6, 8, 12, 16 }) |rows| {
                list[at] = .{ .kind = .served, .rows = rows, .context = context, .kv = kv, .heavy = true };
                at += 1;
            }
        }
    }
    list[at] = .{ .kind = .pick, .rows = 1 };
    at += 1;
    const made = list[0..at].*;
    break :blk made;
};
