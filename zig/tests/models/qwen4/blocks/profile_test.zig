//! Flash Next's own kernels (`zig/kernels/qwen4/`, `common/`) each launched alone at the shapes the model launches
//! them at, on random inputs: what a profiler is given (spec 073 WS-03). No checkpoint and no model. The cases
//! are `profile_cases.zig`'s.
//!
//! `ZSPARK_QWEN4_PROFILE` names a case of `cases` (`chain:1`, `attend:4:4096`, ...), a kernel (`chain`: its
//! cases), or `all` (`docs/environment.md`). Unset, the test is a smoke of the same launches, each case once on
//! one copy, the 512-row and long-context cases left out, for a tier.
//!
//! A case is one entry of the C ABI at one shape: `rows` (1, a draft window's, a prompt chunk's 512) and, for the
//! attention's, the context (positions held, in caches of 4,096 or more). The shapes are those a traced round
//! launched (spec 073's entry has the histogram: a reply's rounds are of 2 to 6 rows, hardly ever 1), not a grid;
//! an attention case's bytes are those its context holds. A named case makes `copies` sets of its buffers, taken in turn, so that what a
//! launch reads was not left in L2 (24 MB) by the launch before, where the set is large enough to say so (a case
//! prints its bytes and copies: a small set stays in L2, as it does in a round, where the launch before wrote
//! it); then a 64 MB tail, since the pages the kernel zeroed last read faster for a while (`cuda-gb10`); then
//! `passes` passes over the copies. A profiler skips the first pass.
//!
//! A case's launches lie between two `cudaMemset`s of a marker buffer, of `mark + 2 i` and `mark + 2 i + 1`
//! bytes for case `i`: a trace's kernels are cut into cases by them (the kernels of a case's own setup come
//! before its first marker).
//!
//!   ZSPARK_QWEN4_PROFILE=all nsys profile --trace=cuda --sample=none --cpuctxsw=none TEST
//!   ZSPARK_QWEN4_PROFILE=chain:1 ncu --kernel-name regex:chain_kernel --launch-skip 24 --launch-count 1 TEST
//!
//! Each case prints its copies, the bytes of one set, and a launch's time by the loop (the kernel's own duration
//! is the trace's).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const experts = @import("../../../../src/weights/experts.zig");
const picking = @import("../../../../src/decode/pick.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const gdn = @import("../../../../src/models/qwen4/blocks/gdn.zig");
const hc = @import("../../../../src/models/qwen4/blocks/hc.zig");
const moe = @import("../../../../src/models/qwen4/blocks/moe.zig");
const mtp = @import("../../../../src/models/qwen4/mtp.zig");
const ple = @import("../../../../src/models/qwen4/blocks/ple.zig");
const state = @import("../../../../src/models/qwen4/blocks/state.zig");
const profile_cases = @import("profile_cases.zig");
const gpu = @import("../../../cuda/testing.zig");
const profile_fixture = @import("../../../weights/profile_fixture.zig");
const nowNs = profile_fixture.nowNs;
const Fill = profile_fixture.Fill;
const Noise = profile_fixture.Noise;
const Marker = profile_fixture.Marker;
const expectFinite = profile_fixture.expectFinite;
const uploaded = gpu.device;
const flash = profile_cases.flash;
const isSite = profile_cases.isSite;
const isBranch = profile_cases.isBranch;
const isSum = profile_cases.isSum;
const shift_layers = profile_cases.shift_layers;
const isAfter = profile_cases.isAfter;
const act_slices = profile_cases.act_slices;
const topk_slices = profile_cases.topk_slices;
const filler_rows = profile_cases.filler_rows;
const Case = profile_cases.Case;
const cases = profile_cases.cases;

const passes = 9;
const mark = 7000;

/// One copy of a case's buffers, in the order its launch names them.
const Set = struct {
    buffers: [24]cuda.Buffer = undefined,
    len: usize = 0,
    bytes: usize = 0,

    fn deinit(self: *Set) void {
        for (self.buffers[0..self.len]) |*buffer| buffer.deinit();
    }

    fn add(self: *Set, gpa: Allocator, noise: Noise, dtype: cuda.DType, shape: []const usize, fill: Fill) !cuda.View {
        var buffer: cuda.Buffer = try .init(dtype, shape);
        errdefer buffer.deinit();
        const count = buffer.view.count();
        const start = noise.random.uintLessThan(usize, Noise.count);
        switch (dtype) {
            .bf16, .f16 => {
                const host = try gpa.alloc(u16, count);
                defer gpa.free(host);
                for (host, 0..) |*bits, i| {
                    const x = noise.value(fill, start + i);
                    bits.* = if (dtype == .bf16) numeric.bf16FromF32(x) else @bitCast(@as(f16, @floatCast(x)));
                }
                try buffer.upload(std.mem.sliceAsBytes(host));
            },
            .f32 => {
                const host = try gpa.alloc(f32, count);
                defer gpa.free(host);
                for (host, 0..) |*x, i| x.* = noise.value(fill, start + i);
                try buffer.upload(std.mem.sliceAsBytes(host));
            },
            else => {
                std.debug.assert(fill == .zero);
                try buffer.zero();
            },
        }
        self.buffers[self.len] = buffer;
        self.len += 1;
        self.bytes += buffer.bytes();
        return buffer.view;
    }

    /// A buffer of the caller's own filling.
    fn keep(self: *Set, buffer: cuda.Buffer) cuda.View {
        self.buffers[self.len] = buffer;
        self.len += 1;
        self.bytes += buffer.bytes();
        return buffer.view;
    }

    fn view(self: Set, index: usize) cuda.View {
        return self.buffers[index].view;
    }
};

/// The attention's weights without its projections, which no launch here reads.
const Attending = struct {
    shape: attention.Shape,
    weights: attention.Weights,
    rotary: attention.Rotary,

    fn init(gpa: Allocator, kv: attention.Storage) !Attending {
        var shape = flash.attending;
        shape.kv = kv;
        const ones = try gpa.alloc(f32, 256);
        defer gpa.free(ones);
        @memset(ones, 1.0);
        var weights: attention.Weights = undefined;
        weights.q_scale = try uploaded(f32, .f32, &.{shape.head_dim}, ones[0..shape.head_dim]);
        weights.k_scale = try uploaded(f32, .f32, &.{shape.head_dim}, ones[0..shape.head_dim]);
        weights.index_q_scale = try uploaded(f32, .f32, &.{shape.index_dim}, ones[0..shape.index_dim]);
        weights.index_k_scale = try uploaded(f32, .f32, &.{shape.index_dim}, ones[0..shape.index_dim]);
        return .{ .shape = shape, .weights = weights, .rotary = try .init(shape) };
    }

    fn deinit(self: *Attending) void {
        self.weights.q_scale.deinit();
        self.weights.k_scale.deinit();
        self.weights.index_q_scale.deinit();
        self.weights.index_k_scale.deinit();
        self.rotary.deinit();
    }
};

/// One copy of an attention case: a layer's caches holding `context - rows` committed positions, and a window's
/// scratch with its queries, lists and output as the launches before the timed one left them.
const held_capacity = 4096;

const Layer = struct {
    cache: attention.Cache,
    scratch: attention.Scratch,
    position: attention.Position,
    bytes: usize,

    fn init(gpa: Allocator, noise: Noise, a: Attending, case: Case) !Layer {
        const shape = a.shape;
        // a served sequence's caches are longer than what it holds, and a launch's chunks go by the capacity
        const capacity = @max(case.sized(), held_capacity);
        var cache: attention.Cache = try .contiguous(shape, capacity);
        errdefer cache.deinit();
        var scratch: attention.Scratch = try .init(shape, @max(case.rows, attention.chunk), capacity);
        errdefer scratch.deinit();
        var position: attention.Position = try .init();
        errdefer position.deinit();
        const host = try gpa.alloc(u16, @max(attention.chunk, case.rows) * shape.projection());
        defer gpa.free(host);
        // the prompt before the window, in chunks, then the window itself up to the launch the case times
        var pos: usize = 0;
        while (pos < case.context) {
            const rows = if (pos < case.context - case.rows) @min(attention.chunk, case.context - case.rows - pos) else case.rows;
            const start = noise.random.uintLessThan(usize, Noise.count);
            for (host[0 .. rows * shape.projection()], 0..) |*bits, i| bits.* = numeric.bf16FromF32(noise.value(.normal, start + i));
            try scratch.pa.uploadAt(0, std.mem.sliceAsBytes(host[0 .. rows * shape.projection()]));
            try position.set(pos);
            const context = pos + rows;
            const pa = scratch.pa.rows(rows);
            try attention.prep(shape, a.weights, a.rotary, cache, scratch, pa, position.device.view, rows, context, null);
            try attention.pool(shape, a.weights, a.rotary, cache, position.device.view, rows, context, null);
            pos = context;
        }
        const selection = selectionOf(shape, scratch, case.sized());
        if (selection) |lists| try attention.select(shape, lists, scratch.iq.rows(case.rows), cache.pooled.view, position.device.view, 0, case.rows, case.sized(), null);
        try attention.attend(shape, cache, scratch, selection, scratch.q.rows(case.rows), position.device.view, scratch.attn.rows(case.rows), case.sized());
        try cuda.synchronize();
        var bytes = cache.k.bytes() + cache.v.bytes() + cache.index_keys.bytes() + cache.pooled.bytes();
        if (cache.k_scale) |b| bytes += 2 * b.bytes();
        return .{ .cache = cache, .scratch = scratch, .position = position, .bytes = bytes };
    }

    fn deinit(self: *Layer) void {
        self.cache.deinit();
        self.scratch.deinit();
        self.position.deinit();
    }

    fn launch(self: Layer, a: Attending, case: Case) !void {
        const shape = a.shape;
        const rows = case.rows;
        const pos = self.position.device.view;
        const selection = selectionOf(shape, self.scratch, case.sized());
        switch (case.kind) {
            .prep => try attention.prep(shape, a.weights, a.rotary, self.cache, self.scratch, self.scratch.pa.rows(rows), pos, rows, case.context, null),
            .pool => try attention.pool(shape, a.weights, a.rotary, self.cache, pos, rows, case.context, null),
            .select => try attention.select(shape, selection.?, self.scratch.iq.rows(rows), self.cache.pooled.view, pos, 0, rows, case.context, null),
            .attend, .after_attend, .served => try attention.attend(shape, self.cache, self.scratch, selection, self.scratch.q.rows(rows), pos, self.scratch.attn.rows(rows), case.sized()),
            else => unreachable,
        }
    }
};

fn selectionOf(shape: attention.Shape, scratch: attention.Scratch, context: usize) ?attention.Selection {
    return if (shape.selects(context)) scratch.selection.? else null;
}

/// A row's `top` distinct routed experts and the shared one, as the selection writes them.
fn randomPicks(gpa: Allocator, random: std.Random, rows: usize) !cuda.Buffer {
    const host = try gpa.alloc(i32, rows * flash.slots);
    defer gpa.free(host);
    for (0..rows) |row| {
        const first = random.uintLessThan(usize, flash.routed);
        for (0..flash.top) |slot| host[row * flash.slots + slot] = @intCast((first + slot * 37) % flash.routed);
        host[row * flash.slots + flash.top] = flash.routed;
    }
    return uploaded(i32, .i32, &.{ rows, flash.slots }, host);
}

/// The buffers of one copy of a case that is no attention's.
fn build(gpa: Allocator, noise: Noise, case: Case, set: *Set, table: cuda.View) !void {
    const rows = case.rows;
    const dn = flash.delta;
    switch (case.kind) {
        .embed => {
            const ids = try gpa.alloc(i32, rows);
            defer gpa.free(ids);
            for (ids) |*id| id.* = @intCast(noise.random.uintLessThan(usize, @intCast(table.size[0])));
            _ = set.keep(try uploaded(i32, .i32, &.{rows}, ids));
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .zero);
        },
        .writeback_none, .writeback_branch, .writeback_slots, .writeback_shared => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ rows, flash.dims / hc.block, flash.streams }, .zero);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.streams }, .small);
            if (case.kind == .writeback_branch) _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.dims }, .small);
            if (case.kind == .writeback_slots or case.kind == .writeback_shared) {
                _ = try set.add(gpa, noise, .f32, &.{ rows * flash.slots, flash.dims }, .small);
                _ = try set.add(gpa, noise, .f32, &.{ rows, flash.slots }, .small);
            }
            if (case.kind == .writeback_shared) _ = try set.add(gpa, noise, .f32, &.{ rows, flash.dims }, .small);
        },
        .site_branch, .site_slots, .pair_branch, .pair_slots, .after_site_branch, .after_site_slots, .after_pair_branch, .after_pair_slots, .site_sum, .pair_sum, .after_site_sum => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ rows, flash.dims / hc.block, flash.streams }, .zero);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.streams }, .small);
            _ = try set.add(gpa, noise, .f32, &.{flash.wide}, .unit);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .zero);
            if (isBranch(case.kind)) {
                _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.dims }, .small);
            } else if (isSum(case.kind)) {
                _ = try set.add(gpa, noise, .f32, &.{ rows, flash.dims }, .small);
            } else {
                _ = try set.add(gpa, noise, .f32, &.{ rows * flash.slots, flash.dims }, .small);
                _ = try set.add(gpa, noise, .f32, &.{ rows, flash.slots }, .small);
            }
        },
        .normed => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ rows, flash.dims / hc.block, flash.streams }, .unit);
            _ = try set.add(gpa, noise, .f32, &.{flash.wide}, .unit);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .zero);
        },
        .act => {
            _ = try set.add(gpa, noise, .f32, &.{ act_slices, rows, flash.low + flash.streams }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.low }, .zero);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.streams }, .zero);
        },
        .mix => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.dims }, .zero);
        },
        .chain, .replay => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, dn.projection() }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ gdn.taps - 1, dn.conv() }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ dn.value_heads, gdn.value_dim, gdn.key_dim }, .small);
            _ = try set.add(gpa, noise, .f32, &.{ dn.value_heads, gdn.value_dim, gdn.key_dim }, .zero);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, dn.out() }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ rows, dn.value_heads * 4 }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ rows, dn.key_heads, gdn.key_dim }, .zero);
            // a replay's scratch as a chain leaves it: keys of norm one, gates under one
            _ = try set.add(gpa, noise, .f32, &.{ rows, dn.key_heads, gdn.key_dim }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, dn.value_heads, gdn.value_dim }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ rows, dn.value_heads }, .unit);
            _ = try set.add(gpa, noise, .f32, &.{ rows, dn.value_heads }, .unit);
        },
        .shift => {
            _ = try set.add(gpa, noise, .bf16, &.{ 1, gdn.taps - 1, dn.conv() }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, dn.projection() }, .normal);
        },
        .shift_each => {
            _ = try set.add(gpa, noise, .bf16, &.{ shift_layers, gdn.taps - 1, dn.conv() }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ shift_layers * rows, dn.projection() }, .normal);
        },
        .topk => {
            _ = try set.add(gpa, noise, .f32, &.{ rows, flash.routed + 1 }, .normal);
            _ = try set.add(gpa, noise, .i32, &.{ rows, flash.slots }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ rows, flash.slots }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ topk_slices, rows, flash.routed + 1 }, .normal);
        },
        .shared_act => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, 2 * flash.width }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.width }, .zero);
        },
        .gate => {
            const a = flash.attending;
            _ = try set.add(gpa, noise, .bf16, &.{ rows, a.heads, a.head_dim }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, a.projection() }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, a.out() }, .zero);
        },
        .ple_embed => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows * flash.ple_heads, flash.ple_dims }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.dims }, .zero);
        },
        .ple_gate => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.dims }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{flash.wide}, .unit);
            _ = try set.add(gpa, noise, .f32, &.{flash.wide}, .unit);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ rows, flash.streams }, .zero);
        },
        .ple_conv => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ rows, flash.streams }, .unit);
            _ = try set.add(gpa, noise, .f32, &.{flash.wide}, .unit);
            _ = try set.add(gpa, noise, .bf16, &.{ (flash.ple_taps - 1) * flash.ple_dilation, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ flash.wide, flash.ple_taps }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .zero);
        },
        .rmsnorm => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows * flash.streams, flash.dims }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{flash.dims}, .unit);
            _ = try set.add(gpa, noise, .bf16, &.{ rows * flash.streams, flash.dims }, .zero);
        },
        .add_streams => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.dims }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows * flash.streams, flash.dims }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.wide }, .zero);
        },
        .pick => {
            _ = try set.add(gpa, noise, .bf16, &.{flash.vocab}, .normal);
            _ = try set.add(gpa, noise, .u32, &.{picking.Picked.wordsFor(flash.vocab)}, .zero);
        },
        .plan => _ = set.keep(try randomPicks(gpa, noise.random, rows)),
        .columns => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.width }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, flash.attending.projection() }, .zero);
        },
        .prep, .pool, .select, .attend, .after_attend, .served => unreachable,
    }
}

/// What every copy of a case shares: weights no larger than L2 would not tell copies apart.
const Shared = struct {
    table: cuda.View,
    conv: cuda.View,
    a_log: cuda.View,
    dt_bias: cuda.View,
    norm: cuda.View,
    plan: ?experts.Plan,
    /// `mix`'s buffers over `filler_rows` rows, for the cases that follow one.
    filler: ?[3]cuda.View = null,
};

fn launch(case: Case, set: Set, shared: Shared) !void {
    const rows = case.rows;
    const dn = flash.delta;
    switch (case.kind) {
        .embed => try hc.embed(set.view(0), shared.table, set.view(1), flash.streams),
        .writeback_none => try hc.writeback(set.view(0), set.view(1), .none, flash.streams),
        .writeback_branch => try hc.writeback(set.view(0), set.view(1), .{ .branch = .{ .branch = set.view(3), .inject = set.view(2) } }, flash.streams),
        .writeback_slots => try hc.writeback(set.view(0), set.view(1), .{ .slots = .{ .y = set.view(3), .shared = null, .weights = set.view(4), .inject = set.view(2) } }, flash.streams),
        .writeback_shared => try hc.writeback(set.view(0), set.view(1), .{ .slots = .{ .y = set.view(3), .shared = set.view(5), .weights = set.view(4), .inject = set.view(2) } }, flash.streams),
        .normed => try hc.normed(set.view(0), set.view(1), set.view(2), set.view(3), flash.eps, flash.streams),
        .site_branch, .site_slots, .pair_branch, .pair_slots, .after_site_branch, .after_site_slots, .after_pair_branch, .after_pair_slots, .site_sum, .pair_sum, .after_site_sum => {
            if (shared.filler) |f| try hc.mix(f[0], f[1], f[2], flash.streams);
            const pending: hc.Pending = if (isBranch(case.kind))
                .{ .branch = .{ .branch = set.view(5), .inject = set.view(2) } }
            else if (isSum(case.kind))
                .{ .sum = .{ .sum = set.view(5), .inject = set.view(2) } }
            else
                .{ .slots = .{ .y = set.view(5), .shared = null, .weights = set.view(6), .inject = set.view(2) } };
            if (isSite(case.kind)) {
                try hc.site(set.view(0), set.view(1), pending, set.view(3), set.view(4), flash.eps, flash.streams);
            } else {
                try hc.writeback(set.view(0), set.view(1), pending, flash.streams);
                try hc.normed(set.view(0), set.view(1), set.view(3), set.view(4), flash.eps, flash.streams);
            }
        },
        .act => try hc.activate(.of(set.view(0), act_slices, rows, flash.low + flash.streams), set.view(1), set.view(2), flash.streams, flash.low),
        .mix => try hc.mix(set.view(0), set.view(1), set.view(2), flash.streams),
        .chain => {
            const scratch: gdn.Scratch = .{ .k = set.buffers[7], .v = set.buffers[8], .g = set.buffers[9], .beta = set.buffers[10] };
            try gdn.chain(.{ .conv = shared.conv, .a_log = shared.a_log, .dt_bias = shared.dt_bias, .norm = shared.norm }, .{
                .projection = set.view(0),
                .conv_state = set.view(1),
                .state_in = set.view(2),
                .state_out = set.view(3),
                .out = set.view(4),
                .sums = set.view(5),
                .scratch = &scratch,
                .work = set.view(6),
            }, rows, flash.eps);
        },
        .replay => {
            const scratch: gdn.Scratch = .{ .k = set.buffers[7], .v = set.buffers[8], .g = set.buffers[9], .beta = set.buffers[10] };
            // a window's prefix: all but its last row, one row of a window of one
            try gdn.replay(set.view(2), scratch, @max(rows - 1, 1), set.view(3));
        },
        .shift => try state.shiftWindows(set.view(0), .{ .view = set.view(1), .layer = 0, .row = dn.projection(), .rows = rows }, rows),
        .shift_each => {
            var olds: [shift_layers]cuda.View = undefined;
            var freshes: [shift_layers]cuda.View = undefined;
            const old = set.view(0);
            const fresh = set.view(1);
            for (&olds, &freshes, 0..) |*o, *f, i| {
                o.* = .init(@ptrFromInt(@intFromPtr(old.ptr.?) + i * (gdn.taps - 1) * dn.conv() * 2), .bf16, &.{ 1, gdn.taps - 1, dn.conv() });
                f.* = .init(@ptrFromInt(@intFromPtr(fresh.ptr.?) + i * rows * dn.projection() * 2), .bf16, &.{ rows, dn.projection() });
            }
            try state.shiftEach(&olds, &freshes, dn.projection(), rows, rows);
        },
        .topk => try cuda.launched(moe.zs_qwen4_moe_topk(&set.view(0), &set.view(1), &set.view(2), flash.top, &set.view(3))),
        .shared_act => try cuda.launched(zs_qwen4_moe_shared_act(&set.view(0), &set.view(1))),
        .gate => try attention.gate(set.view(0), set.view(1), set.view(2)),
        .ple_embed => try cuda.launched(cuda.kernels.zs_qwen4_ple_embed(&set.view(0), &set.view(1), flash.ple_heads, 0.5)),
        .ple_gate => try cuda.launched(cuda.kernels.zs_qwen4_ple_gate(&set.view(0), &set.view(1), &set.view(2), &set.view(3), &set.view(4), &set.view(5), &set.view(6), flash.eps, flash.streams)),
        .ple_conv => try cuda.launched(zs_qwen4_ple_conv(&set.view(0), &set.view(1), &set.view(2), &set.view(3), &set.view(4), &set.view(5), &set.view(6), flash.eps, flash.streams, flash.ple_dilation)),
        .rmsnorm => try mtp.rmsnorm(set.view(0), set.view(1), set.view(2), flash.eps),
        .add_streams => try mtp.addStreams(set.view(0), set.view(1), set.view(2), flash.streams),
        .pick => try picking.pick(set.view(0), set.view(1)),
        .plan => try shared.plan.?.route(set.view(0), rows),
        .columns => try cuda.launched(cuda.kernels.zs_columns_place(&set.view(0), &set.view(1), 0)),
        .prep, .pool, .select, .attend, .after_attend, .served => unreachable,
    }
}

extern fn zs_qwen4_moe_shared_act(gu: *const cuda.View, act: *const cuda.View) c_int;
extern fn zs_qwen4_ple_conv(gated: *const cuda.View, pss: *const cuda.View, norm_conv: *const cuda.View, tail: *const cuda.View, conv: *const cuda.View, h: *const cuda.View, nrow: *const cuda.View, eps: f64, streams: i64, dilation: i64) c_int;

/// The copies of a set of `bytes` bytes: past L2 four times over where 64 copies reach it, two at least.
fn copiesOf(bytes: usize, full: bool) usize {
    if (!full) return 1;
    return std.math.clamp((96 << 20) / @max(bytes, 1), 2, 64);
}

fn report(case: Case, copies: usize, bytes: usize, times: []u64) void {
    var buffer: [64]u8 = undefined;
    std.mem.sort(u64, times, {}, std.sort.asc(u64));
    std.debug.print("PROFILE {s}: {d} launches a pass, {d} bytes a set; a launch by the loop {d:.1} us median, {d:.1} best\n", .{
        case.name(&buffer),                                                                    copies,                                                                    bytes,
        @as(f64, @floatFromInt(times[times.len / 2])) / 1e3 / @as(f64, @floatFromInt(copies)), @as(f64, @floatFromInt(times[0])) / 1e3 / @as(f64, @floatFromInt(copies)),
    });
}

fn run(gpa: Allocator, noise: Noise, case: Case, index: usize, marker: Marker, full: bool) !void {
    var times: [passes]u64 = undefined;
    const timed = if (full) times[0..] else times[0..1];
    switch (case.kind) {
        .prep, .pool, .select, .attend, .after_attend, .served => {
            var a: Attending = try .init(gpa, case.kv);
            defer a.deinit();
            var beside: Set = .{};
            defer beside.deinit();
            const filler: ?[3]cuda.View = if (case.kind == .after_attend) .{
                try beside.add(gpa, noise, .bf16, &.{ filler_rows, flash.wide }, .normal),
                try beside.add(gpa, noise, .bf16, &.{ filler_rows, flash.wide }, .normal),
                try beside.add(gpa, noise, .bf16, &.{ filler_rows, flash.dims }, .zero),
            } else null;
            const bytes = case.context * (a.shape.kv_heads * (a.shape.head_dim * a.shape.kv.dtype().size() + a.shape.kv.valueWidth(a.shape.head_dim) * a.shape.kv.valueDtype().size()) + a.shape.index_dim * 2);
            // by the bytes the context holds, which is what a launch may read; the caches are larger
            const copies = @min(copiesOf(bytes, full), if (case.rows > 16) @as(usize, 3) else if (case.context > 4096) @as(usize, 12) else 48);
            const layers = try gpa.alloc(Layer, copies);
            defer gpa.free(layers);
            var made: usize = 0;
            defer for (layers[0..made]) |*layer| layer.deinit();
            for (layers) |*layer| {
                layer.* = try .init(gpa, noise, a, case);
                made += 1;
            }
            var tail: ?cuda.Buffer = if (full) try .init(.u8, &.{64 << 20}) else null;
            defer if (tail) |*buffer| buffer.deinit();
            try marker.at(index, false);
            for (timed) |*time| {
                try cuda.synchronize();
                const t0 = nowNs();
                for (layers) |layer| {
                    if (filler) |f| try hc.mix(f[0], f[1], f[2], flash.streams);
                    try layer.launch(a, case);
                }
                try cuda.synchronize();
                time.* = nowNs() - t0;
            }
            try marker.at(index, true);
            try expectFinite(gpa, layers[0].scratch.attn.rows(case.rows));
            if (full) report(case, copies, bytes, timed);
        },
        else => {
            const dn = flash.delta;
            var common: Set = .{};
            defer common.deinit();
            // an embedding table past L2 several times over; the model's is 1.27 GB, read a row a token
            const table = try common.add(gpa, noise, .bf16, &.{ if (case.kind == .embed) 32768 else 1, flash.dims }, .normal);
            var plan: ?experts.Plan = if (case.kind == .plan) try .init(case.rows, flash.slots, flash.routed + 1) else null;
            defer if (plan) |*made| made.deinit();
            const filler: ?[3]cuda.View = if (isAfter(case.kind)) .{
                try common.add(gpa, noise, .bf16, &.{ filler_rows, flash.wide }, .normal),
                try common.add(gpa, noise, .bf16, &.{ filler_rows, flash.wide }, .normal),
                try common.add(gpa, noise, .bf16, &.{ filler_rows, flash.dims }, .zero),
            } else null;
            const shared: Shared = .{
                .filler = filler,
                .table = table,
                .conv = try common.add(gpa, noise, .bf16, &.{ dn.conv(), gdn.taps }, .small),
                .a_log = try common.add(gpa, noise, .f32, &.{dn.value_heads}, .small),
                .dt_bias = try common.add(gpa, noise, .f32, &.{dn.value_heads}, .small),
                .norm = try common.add(gpa, noise, .bf16, &.{gdn.value_dim}, .unit),
                .plan = plan,
            };
            var first: Set = .{};
            try build(gpa, noise, case, &first, table);
            const copies = copiesOf(first.bytes, full);
            const sets = try gpa.alloc(Set, copies);
            defer gpa.free(sets);
            sets[0] = first;
            var made: usize = 1;
            defer for (sets[0..made]) |*set| set.deinit();
            for (sets[1..]) |*set| {
                set.* = .{};
                made += 1;
                try build(gpa, noise, case, set, table);
            }
            var tail: ?cuda.Buffer = if (full) try .init(.u8, &.{64 << 20}) else null;
            defer if (tail) |*buffer| buffer.deinit();
            try marker.at(index, false);
            for (timed) |*time| {
                try cuda.synchronize();
                const t0 = nowNs();
                for (sets) |set| try launch(case, set, shared);
                try cuda.synchronize();
                time.* = nowNs() - t0;
            }
            try marker.at(index, true);
            for (sets[0].buffers[0..sets[0].len]) |buffer| try expectFinite(gpa, buffer.view);
            if (full) report(case, copies, first.bytes, timed);
        },
    }
}

test "PROFILE Flash Next's own kernels launched alone at the model's shapes: a row, a draft window, a prompt chunk" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(73);
    var noise: Noise = try .init(gpa, prng.random());
    defer noise.deinit(gpa);
    const asked: ?[]const u8 = if (std.c.getenv("ZSPARK_QWEN4_PROFILE")) |text| std.mem.span(text) else null;
    var marker: Marker = .{ .buffer = try .init(.u8, &.{mark + 2 * cases.len + 2}), .mark = mark };
    defer marker.buffer.deinit();

    // a switch as `main` reads it: 0 times the base kernels
    if (std.c.getenv("ZSPARK_GDN_QUADS")) |text| gdn.setQuads(text[0] != '0');
    const default_window = attention.window_rows;
    defer attention.window_rows = default_window;
    if (std.c.getenv("ZSPARK_ATTN_WINDOW")) |text| attention.window_rows = std.fmt.parseInt(usize, std.mem.span(text), 10) catch 0;
    var ran: usize = 0;
    for (cases, 0..) |case, index| {
        if (asked) |want| {
            var buffer: [64]u8 = undefined;
            const kind = @tagName(case.kind);
            if (!std.mem.eql(u8, want, "all") and !std.mem.eql(u8, want, case.name(&buffer)) and !std.mem.eql(u8, want, kind)) continue;
        } else if (case.heavy or (case.rows != 1 and case.rows != 4)) continue;
        ran += 1;
        try run(gpa, noise, case, index, marker, asked != null);
    }
    // a name that is no case's is a mistake, not a pass
    try std.testing.expect(ran > 0);
}
