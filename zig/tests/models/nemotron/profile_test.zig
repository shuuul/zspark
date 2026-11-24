//! Nemotron-3.5-Transcribe's own kernels (`zig/kernels/nemotron/` and the `common/` and `nvfp4/` entries its
//! decode step launches) each launched alone at a rank's shapes, on random inputs: what a profiler is given
//! (spec 153 WS-00). No checkpoint, no model and no drafter's tensor.
//!
//! `ZSPARK_NEMOTRON_PROFILE` names a case (`conv:4`, `scan:1:hot`, `attend:8`, ...), a kernel (`moeup`: its
//! cases), or `all` (`docs/environment.md`). Unset, the test is a smoke of the same launches, each kernel once
//! at 1 and 4 rows on one copy, for a tier.
//!
//! A case is one entry of the C ABI at `rows` rows (a serial step's 1, a draft window's up to 16, a prompt
//! chunk's 512). The shapes are Transcribe's on one rank: hidden 2688; the Mamba-2 mixer 64 heads of 64 over
//! 8 groups of state 128 (conv channels 6144, projection 12224, an fp32 state of 64 x 64 x 128); the NoPE
//! attention 32 query over 2 KV heads of 128; the relu2-MoE 128 routed experts of 1856 plus the shared
//! expert's two halves, 6 routed and 8 slots a row, W4A4; the DFlash2 drafter rank's 16 heads over a context
//! of 512 and a vocabulary half of 65536. A named case makes `copies` sets of its buffers, a site's constants
//! among them (one MoE layer's packed W4A4 weights, ~0.7 GB, are shared across the sets: every launch reads
//! the stream whole, so L2 keeps none of it either way), then a 64 MB tail, then `passes` passes over the
//! copies; `:hot` is one set launched 16 times a pass, what the kernel costs when memory does not bound it.
//!
//! `attend`'s caches hold `pos` = 32,768 positions (a reply's later windows) in 512-token chunks; `attendg`
//! is the same launch taken by the group kernel from 64 rows. A case's launches lie between two `cudaMemset`s
//! of `mark + 2 i` and `mark + 2 i + 1` bytes, as in `tests/models/qwen4/blocks/profile_test.zig`:
//!
//!   ZSPARK_NEMOTRON_PROFILE=all nsys profile --trace=cuda --sample=none --cpuctxsw=none TEST

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const experts_file = @import("../../../src/weights/experts.zig");
const nvfp4 = @import("../../../src/weights/nvfp4.zig");
const bf16_file = @import("../../../src/weights/bf16.zig");
const glue = @import("../../../src/models/nemotron/glue.zig");
const mamba = @import("../../../src/models/nemotron/mamba.zig");
const moe = @import("../../../src/models/nemotron/moe.zig");
const df2 = @import("../../../src/decode/dflash2/kernels.zig");
const gpu = @import("../../cuda/testing.zig");
const profile_fixture = @import("../../weights/profile_fixture.zig");
const nowNs = profile_fixture.nowNs;
const Fill = profile_fixture.Fill;
const Noise = profile_fixture.Noise;
const Marker = profile_fixture.Marker;
const expectFinite = profile_fixture.expectFinite;

/// Transcribe's dimensions as one rank holds them (`llm_backbone/config.json`; `mamba.zig`, `moe.zig`,
/// `attention.zig`, `dflash2.zig`).
const nemo = struct {
    const hidden = 2688;
    const vocab = 131072;
    const eps = 1e-5;
    // Mamba-2
    const xd = 4096;
    const heads = 64;
    const head_dim = 64;
    const groups = 8;
    const state = 128;
    const cd = xd + 2 * groups * state;
    const proj = 2 * xd + 2 * groups * state + heads;
    const rmax = mamba.window_rows;
    // attention: 32 query over 2 KV heads of 128; a served sequence's position
    const a_heads = 32;
    const a_kv = 2;
    const a_dim = 128;
    const qkv = (a_heads + 2 * a_kv) * a_dim;
    const pos = 32768;
    const capacity = 33792;
    const chunks = (capacity + 512 - 1) / 512;
    // relu2-MoE: 128 routed of 1856 + the shared's two halves, 6 routed and 8 slots
    const routed = 128;
    const experts = routed + 2;
    const width = 1856;
    const top = 6;
    const slots = top + 2;
    const slices = 6; // routerSlices(2688)
    const scale = 2.5;
    // DFlash2's drafter rank: hidden 2688, block 16, 16 query heads over 1 KV of 128, conv groups of 16,
    // the vocabulary's half of 65536, 16 picks a row
    const d_hidden = 2688;
    const block = 16;
    const d_heads = 16;
    const d_kv = 1;
    const d_dim = 128;
    const d_groups = d_hidden / 16;
    const d_context = 512;
    const half_vocab = 65536;
    const d_top = 16;
};

const Kind = enum {
    /// h = x + r, normed: every layer's entry (`norms.cu`).
    norm,
    /// The MoE layer's end: its slots and the norm (`norms.cu`).
    moenorm,
    /// The mamba mixer's group RMSNorm (`norms.cu`).
    gnorm,
    /// A window's depthwise conv and SiLU (`mamba.cu`).
    conv,
    /// A window's selective scan (`mamba.cu`).
    scan,
    /// A prompt chunk's conv and scan (`mamba_wide.cu`).
    convw,
    scanw,
    /// The K/V writes and head split (`common/attention_prep.cu`).
    attprep,
    /// The narrow attend, rows under the group limit (`common/attention.cu`).
    attend,
    /// The group attend of 64 and more rows (`common/attention_group.cu`).
    attendg,
    /// The router's logits and the top-k (`route.cu`).
    route,
    /// The routed pairs' plan (`nvfp4/experts.cu`'s plan kernel).
    moeplan,
    /// W4A4: the rows' quantization, then gate and up relu-squared (`experts_quant.cu`, `experts_a4.cu`).
    moeup,
    /// Down the rows gate and up quantized (`experts_a4.cu`).
    moedown,
    /// The stacked qkv and o projections (`nvfp4/qmmf.cu`): W4A4 as the backbone runs them — the rows
    /// through `zs_nvfp4_linear_quant` first, as `quantizeActivations` makes every window.
    qkv,
    oproj,
    /// The mamba projections, per-tensor FP8 (`nvfp4/qmmf.cu`).
    inproj,
    outproj,
    /// `lm_head`, the checkpoint's one BF16 linear (`common/gemm.cu`): [131,072, 2688], every window's last
    /// launch, the decode row's largest single weight.
    head,
    /// DFlash2's dynamic convolution, a block's rows (`common/dflash2.cu`).
    dconv,
    /// Its attention over the drafter context (`common/dflash2_attend.cu`).
    dattend,
    /// A draft row's top picks of the vocabulary half (`common/dflash2.cu`).
    dtop,
    /// The picks dotted again against the stored head rows (`common/dflash2.cu`).
    drescore,

    /// Fixed-shape launches (a DFlash2 block's), which no 1- or 4-row smoke can take: the smoke runs
    /// them at their own rows.
    fn fixed(self: Kind) bool {
        return switch (self) {
            .dconv, .dattend, .dtop, .drescore => true,
            else => false,
        };
    }
};

const Case = struct {
    kind: Kind,
    rows: usize,
    hot: bool = false,

    fn name(self: Case, buffer: []u8) []const u8 {
        return std.fmt.bufPrint(buffer, "{t}:{d}{s}", .{ self.kind, self.rows, if (self.hot) ":hot" else "" }) catch unreachable;
    }
};

const cases = blk: {
    const kinds = std.enums.values(Kind);
    var list: [kinds.len * 10]Case = undefined;
    var at: usize = 0;
    for (kinds) |kind| {
        const counts: []const usize = switch (kind) {
            // a window's rows: a step to a draft window's 16
            .attend => &.{ 1, 4, 16 },
            // the group kernel is what a prompt chunk's rows take
            .attendg => &.{ 1, 4, 64, 512 },
            // a block pass's fixed width; the draft rows are the block's but its first
            .dconv, .dattend => &.{nemo.block},
            .dtop, .drescore => &.{nemo.block - 1},
            // the wide mamba kernels a prompt chunk takes; the small rows the smoke's
            .convw, .scanw => &.{ 1, 4, 512 },
            else => &.{ 1, 4, 16, 512 },
            // the window kernels bound rows at RMAX 16; the wide ones take a prompt chunk's
            .conv, .scan => &.{ 1, 4, 16 },
        };
        for (counts) |rows| {
            for ([_]bool{ false, true }) |hot| {
                list[at] = .{ .kind = kind, .rows = rows, .hot = hot };
                at += 1;
            }
        }
    }
    const made = list[0..at].*;
    break :blk made;
};

const passes = 9;
const hot_launches = 16;
const mark = 7600;

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
            .bf16 => {
                const host = try gpa.alloc(u16, count);
                defer gpa.free(host);
                for (host, 0..) |*bits, i| bits.* = numeric.bf16FromF32(noise.value(fill, start + i));
                try buffer.upload(std.mem.sliceAsBytes(host));
            },
            .f32 => {
                const host = try gpa.alloc(f32, count);
                defer gpa.free(host);
                for (host, 0..) |*x, i| x.* = noise.value(fill, start + i);
                try buffer.upload(std.mem.sliceAsBytes(host));
            },
            .i32 => {
                const host = try gpa.alloc(i32, count);
                defer gpa.free(host);
                for (host, 0..) |*x, i| x.* = @intFromFloat(noise.value(fill, start + i));
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

    fn ints(self: *Set, shape: []const usize, values: []const i32) !cuda.View {
        var buffer: cuda.Buffer = try .init(.i32, shape);
        errdefer buffer.deinit();
        try buffer.upload(std.mem.sliceAsBytes(values));
        return self.keep(buffer);
    }

    /// f32 of `base + spread * normal`: a scan's a (all negative, as decay writes it) or its dt_bias.
    fn addF32(self: *Set, gpa: Allocator, noise: Noise, shape: []const usize, base: f32, spread: f32) !cuda.View {
        var buffer: cuda.Buffer = try .init(.f32, shape);
        errdefer buffer.deinit();
        const count = buffer.view.count();
        const start = noise.random.uintLessThan(usize, Noise.count);
        const host = try gpa.alloc(f32, count);
        defer gpa.free(host);
        for (host, 0..) |*x, i| x.* = base + spread * noise.value(.normal, start + i);
        try buffer.upload(std.mem.sliceAsBytes(host));
        self.buffers[self.len] = buffer;
        self.len += 1;
        self.bytes += buffer.bytes();
        return buffer.view;
    }

    fn view(self: Set, index: usize) cuda.View {
        return self.buffers[index].view;
    }
};

/// The mamba meta row a mid-sequence window runs with: position `pos`, parity 0, and the previous window's
/// `pk` kept rows pending in `raw/xc/dt[1 - parity]` to replay first (DFlash2 keeps ~97%, so a decode
/// window's scan steps are its rows plus the last kept window's). `pk` bounded by the parity buffers' room.
fn addMeta(set: *Set, rows: usize) !cuda.View {
    const pk: i32 = if (rows >= nemo.rmax) 1 else @intCast(@min(nemo.rmax - rows, nemo.rmax));
    return set.ints(&.{4}, &.{ nemo.pos, 0, pk, 0 });
}

/// A row's `top` routed picks and the two shared halves, distinct.
fn addPicks(gpa: Allocator, noise: Noise, set: *Set, rows: usize) !cuda.View {
    const host = try gpa.alloc(i32, rows * nemo.slots);
    defer gpa.free(host);
    for (0..rows) |row| {
        const first = noise.random.uintLessThan(usize, nemo.routed);
        for (0..nemo.top) |slot| host[row * nemo.slots + slot] = @intCast((first + slot * 37) % nemo.routed);
        host[row * nemo.slots + nemo.top] = nemo.routed;
        host[row * nemo.slots + nemo.top + 1] = nemo.routed + 1;
    }
    return set.ints(&.{ rows, nemo.slots }, host);
}

/// The buffers of one copy of a case that is no MoE or DFlash2 case's.
fn build(gpa: Allocator, noise: Noise, case: Case, set: *Set) !void {
    const rows = case.rows;
    switch (case.kind) {
        .norm => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{nemo.hidden}, .unit);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .zero);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ rows, nemo.hidden / 64 }, .zero);
        },
        .moenorm => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ rows, nemo.slots, nemo.hidden }, .small);
            _ = try set.add(gpa, noise, .f32, &.{ rows, nemo.slots }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{nemo.hidden}, .unit);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .zero);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ rows, nemo.hidden / 64 }, .zero);
        },
        .gnorm => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.xd }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{nemo.xd}, .unit);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.xd }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ rows, nemo.xd / 64 }, .zero);
        },
        .conv, .convw => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.proj }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ 3, nemo.cd }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ 2, nemo.rmax, nemo.cd }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ 2, nemo.rmax, nemo.cd }, .small);
            if (case.kind == .convw) _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.cd }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ 4, nemo.cd }, .small);
            _ = try set.add(gpa, noise, .f32, &.{nemo.cd}, .small);
            _ = try addMeta(set, rows);
        },
        .scan, .scanw => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.proj }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ 2, nemo.rmax, nemo.cd }, .small);
            if (case.kind == .scanw) _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.cd }, .small);
            _ = try set.add(gpa, noise, .f32, &.{ 2, nemo.rmax, nemo.heads }, .small);
            _ = try set.add(gpa, noise, .f32, &.{ nemo.heads, nemo.head_dim, nemo.state }, .small);
            // a = -exp(A_log) < 0, as `decay` writes it: a positive a grows ssm without bound over the
            // passes; d near one, dt_bias in softplus's middle (softplus(-2) ~ 0.13)
            _ = try set.addF32(gpa, noise, &.{nemo.heads}, -1.0, 0.05);
            _ = try set.add(gpa, noise, .f32, &.{nemo.heads}, .unit);
            _ = try set.addF32(gpa, noise, &.{nemo.heads}, -2.0, 0.5);
            _ = try addMeta(set, rows);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.xd }, .zero);
        },
        .attprep => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.qkv }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.capacity, nemo.a_kv, nemo.a_dim }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.capacity, nemo.a_kv, nemo.a_dim }, .small);
            _ = try set.ints(&.{1}, &.{nemo.pos});
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.a_heads, nemo.a_dim }, .zero);
        },
        .attend, .attendg => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.a_heads, nemo.a_dim }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.capacity, nemo.a_kv, nemo.a_dim }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.capacity, nemo.a_kv, nemo.a_dim }, .small);
            _ = try set.ints(&.{1}, &.{nemo.pos});
            if (case.kind == .attend) {
                _ = try set.add(gpa, noise, .f32, &.{ rows, nemo.chunks, nemo.a_heads, nemo.a_dim }, .zero);
                _ = try set.add(gpa, noise, .f32, &.{ rows, nemo.chunks, nemo.a_heads }, .zero);
                _ = try set.add(gpa, noise, .f32, &.{ rows, nemo.chunks, nemo.a_heads }, .zero);
            }
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.a_heads, nemo.a_dim }, .zero);
        },
        .route => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.routed, nemo.hidden }, .small);
            _ = try set.add(gpa, noise, .f32, &.{nemo.routed}, .small);
            _ = try set.add(gpa, noise, .f32, &.{ nemo.slices, rows, nemo.routed }, .zero);
            _ = try set.add(gpa, noise, .i32, &.{ rows, nemo.slots }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{ rows, nemo.slots }, .zero);
        },
        .moeplan, .moeup, .moedown => {
            _ = try addPicks(gpa, noise, set, rows);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ rows * nemo.slots, nemo.hidden }, .zero);
        },
        .qkv => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.qkv }, .zero);
        },
        .oproj => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.a_heads * nemo.a_dim }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .zero);
        },
        .inproj => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.proj }, .zero);
        },
        .outproj => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.xd }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .zero);
        },
        .head => {
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.hidden }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ rows, nemo.vocab }, .zero);
        },
        .dconv => {
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.block, nemo.d_hidden }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.block, 4 * nemo.d_groups }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ 2, 2, nemo.d_hidden }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.block, nemo.d_hidden }, .zero);
        },
        .dattend => {
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.d_heads, nemo.block, nemo.d_dim }, .normal);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.d_kv, nemo.d_context, nemo.d_dim }, .small);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.d_kv, nemo.d_context, nemo.d_dim }, .small);
            _ = try set.ints(&.{1}, &.{nemo.d_context - nemo.block});
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.block, nemo.d_heads * nemo.d_dim }, .zero);
            _ = try set.add(gpa, noise, .f32, &.{df2.attendParts(nemo.d_heads, nemo.block, nemo.d_context)}, .zero);
        },
        .dtop => {
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.block - 1, nemo.half_vocab }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ nemo.block - 1, nemo.d_top }, .zero);
            _ = try set.add(gpa, noise, .i32, &.{ nemo.block - 1, nemo.d_top }, .zero);
            const scratch = df2.topScratch(nemo.block - 1, nemo.half_vocab, nemo.d_top);
            if (scratch > 0) _ = try set.add(gpa, noise, .i32, &.{scratch}, .zero);
        },
        .drescore => {
            const host = try gpa.alloc(i32, (nemo.block - 1) * nemo.d_top);
            defer gpa.free(host);
            for (host, 0..) |*token, i| token.* = @intCast((i * 4099) % nemo.half_vocab);
            _ = try set.ints(&.{ nemo.block - 1, nemo.d_top }, host);
            _ = try set.add(gpa, noise, .bf16, &.{ nemo.block - 1, nemo.d_hidden }, .normal);
            _ = try set.add(gpa, noise, .f32, &.{ nemo.block - 1, nemo.d_top }, .zero);
        },
    }
}

/// What every copy of a case shares: weights no larger than L2 would not tell copies apart, but a layer's
/// packed experts and a drafter head's vocabulary rows outweigh it many times over.
const Shared = struct {
    experts: ?experts_file.Experts = null,
    qkv: ?nvfp4.Fp4Linear = null,
    o: ?nvfp4.Fp4Linear = null,
    in_proj: ?nvfp4.Fp8BlockLinear = null,
    out_proj: ?nvfp4.Fp8BlockLinear = null,
    head: ?bf16_file.Linear = null,
    vocab: ?cuda.Buffer = null,
    plan: ?experts_file.Plan = null,
    quantized: ?experts_file.Quantized = null,

    fn deinit(self: *Shared, gpa: Allocator) void {
        if (self.experts) |*one| one.deinit();
        if (self.qkv) |*one| one.deinit();
        if (self.o) |*one| one.deinit();
        if (self.in_proj) |*one| one.deinit();
        if (self.out_proj) |*one| one.deinit();
        if (self.vocab) |*one| one.deinit();
        if (self.head) |*one| one.deinit();
        if (self.plan) |*one| one.deinit();
        if (self.quantized) |*one| one.deinit();
        _ = gpa;
    }
};

/// One expert's random NVFP4 projection, which every expert of the layer then is (`nvfp4_profile_test.zig`'s
/// `OneMatrix`): the bytes differ by address only, which is all a timing asks.
const OneMatrix = struct {
    codes: []u8,
    block_scales: []u8,
    scales: []f32,
    inputs: []f32,
    code_parts: [][]const u8,
    scale_parts: [][]const u8,

    fn init(gpa: Allocator, random: std.Random, count: usize, n: usize, k: usize) !OneMatrix {
        const codes = try gpa.alloc(u8, n * k / 2);
        errdefer gpa.free(codes);
        random.bytes(codes);
        const block_scales = try gpa.alloc(u8, n * k / 16);
        errdefer gpa.free(block_scales);
        for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
        const scales = try gpa.alloc(f32, count);
        errdefer gpa.free(scales);
        for (scales) |*scale| scale.* = 0.01 + 0.02 * random.float(f32);
        const inputs = try gpa.alloc(f32, count);
        errdefer gpa.free(inputs);
        @memset(inputs, 0.0015);
        const code_parts = try gpa.alloc([]const u8, count);
        errdefer gpa.free(code_parts);
        @memset(code_parts, codes);
        const scale_parts = try gpa.alloc([]const u8, count);
        errdefer gpa.free(scale_parts);
        @memset(scale_parts, block_scales);
        return .{ .codes = codes, .block_scales = block_scales, .scales = scales, .inputs = inputs, .code_parts = code_parts, .scale_parts = scale_parts };
    }

    fn deinit(self: OneMatrix, gpa: Allocator) void {
        gpa.free(self.codes);
        gpa.free(self.block_scales);
        gpa.free(self.scales);
        gpa.free(self.inputs);
        gpa.free(self.code_parts);
        gpa.free(self.scale_parts);
    }

    fn matrix(self: OneMatrix) experts_file.Matrix {
        return .{ .codes = self.code_parts, .block_scales = self.scale_parts, .scales = self.scales, .input_scales = self.inputs };
    }
};

/// A layer's relu2 experts, W4A4 (`initRelu2Quantized`), at the rank's count of 130.
fn makeExperts(gpa: Allocator, random: std.Random) !experts_file.Experts {
    const up: OneMatrix = try .init(gpa, random, nemo.experts, nemo.width, nemo.hidden);
    defer up.deinit(gpa);
    const down: OneMatrix = try .init(gpa, random, nemo.experts, nemo.hidden, nemo.width);
    defer down.deinit(gpa);
    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    return .initRelu2Quantized(gpa, &staging, nemo.experts, nemo.width, nemo.hidden, up.matrix(), down.matrix());
}

/// An NVFP4 projection of n x k, random codes.
fn makeFp4(gpa: Allocator, random: std.Random, n: usize, k: usize) !nvfp4.Fp4Linear {
    const codes = try gpa.alloc(u8, n * k / 2);
    defer gpa.free(codes);
    random.bytes(codes);
    const block_scales = try gpa.alloc(u8, n * k / 16);
    defer gpa.free(block_scales);
    for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
    return .init(gpa, codes, block_scales, n, k, 0.02);
}

/// A per-tensor FP8 projection on its checkpoint's static activation scale (the mamba linears as the
/// checkpoint stores and the model runs them: Fp8BlockLinear + quantizeStaticActivations -> qmm_a8.cu).
fn makeFp8(gpa: Allocator, random: std.Random, n: usize, k: usize, rows: cuda.Scratch.Rows) !nvfp4.Fp8BlockLinear {
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    random.bytes(codes);
    for (codes) |*b| {
        if (b.* & 0x7f == 0x7f) b.* ^= 1; // e4m3's NaN patterns, as a real checkpoint lacks
    }
    const scales = try gpa.alloc(f32, 1);
    defer gpa.free(scales);
    scales[0] = 0.0015;
    var linear = try nvfp4.Fp8BlockLinear.init(gpa, codes, scales, n, k, .{ .outputs = n, .inputs = k });
    errdefer linear.deinit();
    try linear.quantizeStaticActivations(rows, 0.0015);
    return linear;
}

/// The drafter head's stored rows of the vocabulary half, bf16.
fn makeVocab(gpa: Allocator, noise: Noise) !cuda.Buffer {
    const host = try gpa.alloc(u16, nemo.half_vocab * nemo.d_hidden);
    defer gpa.free(host);
    const start = noise.random.uintLessThan(usize, Noise.count);
    for (host, 0..) |*bits, i| bits.* = numeric.bf16FromF32(noise.value(.small, start + i));
    return gpu.device(u16, .bf16, &.{ nemo.half_vocab, nemo.d_hidden }, host);
}

fn launch(case: Case, set: Set, shared: Shared) !void {
    const rows = case.rows;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(nemo.a_dim)));
    switch (case.kind) {
        .norm => try glue.addRmsnorm(set.view(0), set.view(1), set.view(2), set.view(3), set.view(4), set.view(5), nemo.eps),
        .moenorm => try glue.addMoeNorm(set.view(0), set.view(1), set.view(2), set.view(3), set.view(4), set.view(5), set.view(6), nemo.eps, nemo.top),
        .gnorm => try glue.groupRmsnorm(set.view(0), set.view(1), set.view(2), set.view(3), nemo.eps, nemo.groups),
        .conv => try mamba.conv(set.view(0), set.view(1), set.view(2), set.view(3), set.view(4), set.view(5), set.view(6), nemo.xd),
        .convw => try mamba.convWide(set.view(0), set.view(1), set.view(2), set.view(3), set.view(4), set.view(5), set.view(6), set.view(7), nemo.xd),
        .scan => try mamba.scan(set.view(0), set.view(1), set.view(2), set.view(3), set.view(4), set.view(5), set.view(6), set.view(7), set.view(8), 0, std.math.inf(f32), nemo.groups),
        .scanw => try mamba.scanWide(set.view(0), set.view(1), set.view(2), set.view(3), set.view(4), set.view(5), set.view(6), set.view(7), set.view(8), set.view(9), 0, std.math.inf(f32), nemo.groups),
        .attprep => try cuda.launched(cuda.kernels.zs_attn_prep(&set.view(0), null, null, null, &set.view(3), &set.view(4), &set.view(1), &set.view(2), nemo.eps)),
        .attend => try cuda.launched(cuda.kernels.zs_attn_attend(&set.view(0), &set.view(1), &set.view(2), &set.view(3), &set.view(4), &set.view(5), &set.view(6), &set.view(7), null, null, null, null, null, null, scale, @intCast(nemo.pos + rows))),
        .attendg => try cuda.launched(cuda.kernels.zs_attn_attend_group(&set.view(0), &set.view(1), &set.view(2), &set.view(3), &set.view(4), scale, @intCast(nemo.pos + rows))),
        .route => try glue.route(set.view(0), set.view(1), set.view(2), set.view(3), set.view(4), set.view(5), nemo.scale, nemo.top, nemo.slots, true),
        .moeplan => try shared.plan.?.routeAs(set.view(0), rows, experts_file.Activations.passOf(rows)),
        .moeup => {
            try shared.plan.?.routeAs(set.view(0), rows, experts_file.Activations.passOf(rows));
            try shared.experts.?.gateUpQuantizedAs(set.view(1), shared.plan.?, shared.quantized.?, null, rows, null, experts_file.Activations.passOf(rows));
        },
        .moedown => {
            try shared.plan.?.routeAs(set.view(0), rows, experts_file.Activations.passOf(rows));
            try shared.experts.?.downProjectQuantizedAs(shared.plan.?, shared.quantized.?, set.view(2), rows, null, experts_file.Activations.passOf(rows));
        },
        .qkv => try shared.qkv.?.forward(set.view(0), set.view(1)),
        .oproj => try shared.o.?.forward(set.view(0), set.view(1)),
        .inproj => try shared.in_proj.?.forward(set.view(0), set.view(1)),
        .outproj => try shared.out_proj.?.forward(set.view(0), set.view(1)),
        .head => try shared.head.?.forward(set.view(0), set.view(1)),
        .dconv => try df2.dconv(set.view(0), set.view(1), set.view(2), 0, null, set.view(3)),
        .dattend => try df2.attend(set.view(0), set.view(1), set.view(2), .{ .pos = set.view(3), .ring = false }, set.view(4), set.view(5), nemo.d_context, false),
        .dtop => try df2.top(set.view(0), set.view(1), set.view(2), if (set.len > 3) set.view(3) else null, 0),
        .drescore => try df2.rescore(set.view(0), shared.vocab.?.view, set.view(1), set.view(2), 0),
    }
}

/// The copies of a set of `bytes` bytes: past L2 four times over where 64 copies reach it, two at least.
fn copiesOf(bytes: usize, full: bool) usize {
    if (!full) return 1;
    return std.math.clamp((96 << 20) / @max(bytes, 1), 2, 64);
}

fn run(gpa: Allocator, noise: Noise, case: Case, index: usize, marker: Marker, full: bool, shared: *Shared) !void {
    var first: Set = .{};
    try build(gpa, noise, case, &first);
    const copies = copiesOf(first.bytes, full);
    const sets = try gpa.alloc(Set, copies);
    defer gpa.free(sets);
    sets[0] = first;
    var made: usize = 1;
    defer for (sets[0..made]) |*set| set.deinit();
    for (sets[1..]) |*set| {
        set.* = .{};
        made += 1;
        try build(gpa, noise, case, set);
    }
    // moedown's rows_q down halves are what its own gate and up just quantized: seed them once, before
    // the markers, as a launch's setup (the copies' picks differ; the values are valid NVFP4 either way)
    if (case.kind == .moedown) {
        try shared.plan.?.routeAs(first.view(0), case.rows, experts_file.Activations.passOf(case.rows));
        try shared.experts.?.gateUpQuantizedAs(first.view(1), shared.plan.?, shared.quantized.?, null, case.rows, null, experts_file.Activations.passOf(case.rows));
    }
    var tail: ?cuda.Buffer = if (full) try .init(.u8, &.{64 << 20}) else null;
    // the pages the kernel zeroed last read faster for a while; the tail is written once so no case's
    // first pass finds them so (cuda-gb10)
    if (tail) |*buffer| try buffer.zero();
    defer if (tail) |*buffer| buffer.deinit();
    const launches = if (full and case.hot) hot_launches else copies;
    var times: [passes]u64 = undefined;
    const timed = if (full) times[0..] else times[0..1];
    try marker.at(index, false);
    for (timed) |*time| {
        try cuda.synchronize();
        const t0 = nowNs();
        // :hot is one set in L2 launched `hot_launches` times a pass; cold the copies in turn
        for (0..launches) |i| try launch(case, if (case.hot) sets[0] else sets[i % copies], shared.*);
        try cuda.synchronize();
        time.* = nowNs() - t0;
    }
    try marker.at(index, true);
    // every value finite: the launch ran on what it was given — the scan's ssm among them, decaying as it
    // should (a < 0) rather than growing over the passes
    for (sets[0].buffers[0..sets[0].len]) |buffer| try expectFinite(gpa, buffer.view);
    if (!full) return;
    var buffer: [64]u8 = undefined;
    std.mem.sort(u64, timed, {}, std.sort.asc(u64));
    std.debug.print("PROFILE {s}: {d} launches a pass, {d} bytes a set; a launch by the loop {d:.1} us median, {d:.1} best\n", .{
        case.name(&buffer),                                                                      launches,                                                                    first.bytes,
        @as(f64, @floatFromInt(timed[timed.len / 2])) / 1e3 / @as(f64, @floatFromInt(launches)), @as(f64, @floatFromInt(timed[0])) / 1e3 / @as(f64, @floatFromInt(launches)),
    });
}

test "PROFILE Nemotron-3.5-Transcribe's own kernels launched alone at a rank's shapes: a step, a draft window, a prompt chunk" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(73);
    var noise: Noise = try .init(gpa, prng.random());
    defer noise.deinit(gpa);
    const asked: ?[]const u8 = if (std.c.getenv("ZSPARK_NEMOTRON_PROFILE")) |text| std.mem.span(text) else null;
    var marker: Marker = .{ .buffer = try .init(.u8, &.{mark + 2 * cases.len + 2}), .mark = mark };
    defer marker.buffer.deinit();

    var shared: Shared = .{};
    defer shared.deinit(gpa);
    // the linears' W4A4 work rows come of a scratch pool, as a model's own (`quantizeActivations`)
    var pool = try cuda.Scratch.create(gpa);
    defer pool.destroy();
    var ran: usize = 0;
    for (cases, 0..) |case, index| {
        if (asked) |want| {
            var buffer: [64]u8 = undefined;
            const kind = @tagName(case.kind);
            if (!std.mem.eql(u8, want, "all") and !std.mem.eql(u8, want, case.name(&buffer)) and !std.mem.eql(u8, want, kind)) continue;
        } else if (case.hot or (!case.kind.fixed() and case.rows != 1 and case.rows != 4)) continue;
        ran += 1;
        // the shared fixtures a kind needs, made on its first run
        switch (case.kind) {
            .moeplan, .moeup, .moedown => {
                // the plan and the quantized rows sized once for the most pairs a case takes
                // (`routeAs` bounds a call's rows under the plan's)
                if (shared.plan == null) shared.plan = try .init(512, nemo.slots, nemo.experts);
                if (shared.quantized == null) shared.quantized = try .init(512 * nemo.slots, nemo.hidden, nemo.width);
                if ((case.kind == .moeup or case.kind == .moedown) and shared.experts == null) shared.experts = try makeExperts(gpa, noise.random);
            },
            .qkv => if (shared.qkv == null) {
                shared.qkv = try makeFp4(gpa, noise.random, nemo.qkv, nemo.hidden);
                try shared.qkv.?.quantizeActivations(pool.rows(512), 0.0015);
            },
            .oproj => if (shared.o == null) {
                shared.o = try makeFp4(gpa, noise.random, nemo.hidden, nemo.a_heads * nemo.a_dim);
                try shared.o.?.quantizeActivations(pool.rows(512), 0.0015);
            },
            .inproj => if (shared.in_proj == null) {
                shared.in_proj = try makeFp8(gpa, noise.random, nemo.proj, nemo.hidden, pool.rows(512));
            },
            .outproj => if (shared.out_proj == null) {
                shared.out_proj = try makeFp8(gpa, noise.random, nemo.hidden, nemo.xd, pool.rows(512));
            },
            .head => if (shared.head == null) {
                // the checkpoint's bf16 bytes, random but finite
                const weight = try gpa.alloc(u16, nemo.vocab * nemo.hidden);
                defer gpa.free(weight);
                const start = noise.random.uintLessThan(usize, Noise.count);
                for (weight, 0..) |*bits, i| bits.* = numeric.bf16FromF32(noise.value(.small, start + i));
                shared.head = try .init(std.mem.sliceAsBytes(weight), nemo.vocab, nemo.hidden, pool.rows(512));
            },
            .drescore => if (shared.vocab == null) {
                shared.vocab = try makeVocab(gpa, noise);
            },
            else => {},
        }
        try run(gpa, noise, case, index, marker, asked != null, &shared);
    }
    // a name that is no case's is a mistake, not a pass
    try std.testing.expect(ran > 0);
}
