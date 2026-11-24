//! GLM-5.3-Flash's own kernels (`zig/kernels/glm5/`) each launched alone at the shapes a rank launches them at,
//! on random inputs: what a profiler is given (spec 073 WS-05). No checkpoint, no model and no drafter's tensor.
//!
//! `ZSPARK_GLM5_PROFILE` names a case (`pre:4`, `mix:8:hot`, ...), a kernel (`mix`: its cases), or `all`
//! (`docs/environment.md`). Unset, the test is a smoke of the same launches, each kernel once at 1 and 4 rows on
//! one copy, for a tier.
//!
//! A case is one entry of the C ABI at `rows` rows (a round's are 2 to 8, spec 073's entry has the histogram; a
//! prompt chunk's 512). A named case makes `copies` sets of its buffers, a site's constants among them (a
//! rank's 90 sites hold 71 MB of mix dots and its 34 KDA layers 143 MB of states, so a round finds none of them
//! in L2's 24 MB), then a 64 MB tail, then `passes` passes over the copies; `:hot` is one set launched 16 times
//! a pass, what the kernel costs when memory does not bound it. `bare` is `pre` with one Sinkhorn round for its twenty, to say
//! what of `pre` its first thread's prologue is. `attend` is DSA's rows past the dense limit, each over its own list
//! of 2,051 keys of a cache of 33,792 (twelve caches, so that L2 holds none of a launch's keys); `keys`, `pool`,
//! `score` and `select` are the indexer's launches at that position, sized as a graph's (spec 093 WS-04). A case's launches lie between two `cudaMemset`s of `mark + 2 i`
//! and `mark + 2 i + 1` bytes, as in `tests/models/qwen4/blocks/profile_test.zig`:
//!
//!   ZSPARK_GLM5_PROFILE=all nsys profile --trace=cuda --sample=none --cpuctxsw=none TEST

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const dsa = @import("../../../src/models/glm5/layers/dsa.zig");
const hc = @import("../../../src/models/glm5/layers/hc.zig");
const kda = @import("../../../src/models/glm5/layers/kda.zig");
const index_launch = @import("../../../src/models/glm5/layers/index_launch.zig");
const profile_fixture = @import("../../weights/profile_fixture.zig");
const nowNs = profile_fixture.nowNs;
const Fill = profile_fixture.Fill;
const Noise = profile_fixture.Noise;
const Marker = profile_fixture.Marker;
const expectFinite = profile_fixture.expectFinite;

/// GLM-5.3-Flash's sizes as one rank holds them (`config.json` of FP4-Local; `layers/config.zig`).
const glm = struct {
    const dims = 4096;
    const wide = hc.streams * dims;
    const mix = 24;
    const iters = 20;
    const eps = 1e-6;
    const hc_eps = 1e-6;
    const world = 2;
    /// KDA: 32 of the 64 heads, 128 a head; a projection row is q | k | v | f_a | g_a | b.
    const heads = 32;
    const head = 128;
    const inner = heads * head;
    const projection = 3 * inner + 2 * head + heads;
    /// DSA: 32 heads of 256 keys and 256 values over a latent of 512.
    const key = 256;
    const latent = 512;
    /// The indexer: 32 heads of 128, the 512 best pools of four kept and the unfinished pool's three.
    const index_heads = 32;
    const index_dim = 128;
    const kept = 512;
    const width = kept * 4 + 3;
    /// A long prompt's position (spec 093's request holds 32,768 tokens), in a cache of 66 extents; a
    /// graph's launches are sized for `bucket` positions and the scores for the context's pools.
    const pos = 32798;
    const held = 66 * 512;
    const bucket = 65536;
    const pools = 262144 / 4;
    /// `window`'s: the last 4,096 rows of a 16,384-token prompt.
    const window_pos = 12288;
    const window_context = 16384;
};

const Kind = enum {
    mix,
    pre,
    bare,
    post,
    chain,
    absorb,
    expand,
    /// A round's rows past the dense limit (spec 093 WS-04): each attends its list's 2,048 to 2,051 latents.
    attend,
    /// The same over an FP8 cache (`--kv-dtype fp8`, spec 107 WS-03): a token's 528-byte record.
    attend8,
    /// The indexer's launches at that position: the rows' keys and gates written, the pool they complete, the
    /// scores over the context's pools, the lists.
    keys,
    pool,
    score,
    select,
    /// The score over a prompt window of 4,096 rows at 12,288, a context of 16,384 (spec 116 WS-05).
    window,

    fn listed(self: Kind) bool {
        return switch (self) {
            .attend, .attend8, .keys, .pool, .score, .select, .window => true,
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
    const counts = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 64, 128, 512, 2048, 4096 };
    var list: [kinds.len * counts.len * 2]Case = undefined;
    var at: usize = 0;
    for (kinds) |kind| {
        for (counts) |rows| {
            // from 32 rows the chain is the chunked one's three launches, a prompt's and not a round's
            // 2,048: a prompt window at FP4-Local's default rows, the read-out's kernels and KDA's chunked chain only
            // (spec 107 WS-03)
            if (kind == .window and rows != 4096) continue;
            if (rows == 2048 and kind != .mix and kind != .pre and kind != .chain) continue;
            // 64, 128: where a window kernel starts paying; 4,096: a 16k prompt's layer-window (spec 116 WS-01,
            // WS-05; the chunked chain alone, WS-06)
            if ((rows == 64 or rows == 128) and kind != .attend and kind != .attend8) continue;
            if (rows == 4096 and kind != .attend and kind != .attend8 and kind != .window and kind != .chain) continue;
            if (rows == 512 and (kind == .chain or kind == .bare or (kind.listed() and kind != .attend and kind != .attend8))) continue;
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
const mark = 7400;

/// One copy of a case's buffers, in the order its launch names them.
const Set = struct {
    buffers: [18]cuda.Buffer = undefined,
    len: usize = 0,
    bytes: usize = 0,

    fn deinit(self: *Set) void {
        for (self.buffers[0..self.len]) |*buffer| buffer.deinit();
    }

    fn add(self: *Set, gpa: Allocator, noise: Noise, dtype: cuda.DType, shape: []const usize, fill: Fill) !void {
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
            // e4m3 codes of the values times 16
            .u8 => {
                const host = try gpa.alloc(u8, count);
                defer gpa.free(host);
                for (host, 0..) |*code, i| code.* = numeric.e4m3FromF32(std.math.clamp(16 * noise.value(fill, start + i), -448, 448));
                try buffer.upload(host);
            },
            else => unreachable,
        }
        self.buffers[self.len] = buffer;
        self.len += 1;
        self.bytes += buffer.bytes();
    }

    fn addInts(self: *Set, shape: []const usize, values: []const i32) !void {
        var buffer: cuda.Buffer = try .init(.i32, shape);
        errdefer buffer.deinit();
        try buffer.upload(std.mem.sliceAsBytes(values));
        self.buffers[self.len] = buffer;
        self.len += 1;
        self.bytes += buffer.bytes();
    }

    /// An FP8 cache of `positions` tokens: a token's codes of `.small` values over a scale a group.
    fn addRecords(self: *Set, gpa: Allocator, noise: Noise, positions: usize) !void {
        var buffer: cuda.Buffer = try .init(.u8, &.{ positions, dsa.fp8_record });
        errdefer buffer.deinit();
        const host = try gpa.alloc(u8, positions * dsa.fp8_record);
        defer gpa.free(host);
        const start = noise.random.uintLessThan(usize, Noise.count);
        const scale: f32 = 0.4 / 448.0;
        for (0..positions) |t| {
            const rec = host[t * dsa.fp8_record ..][0..dsa.fp8_record];
            for (rec[0..dsa.latent], 0..) |*code, i| code.* = numeric.e4m3FromF32(std.math.clamp(noise.value(.small, start + t * dsa.latent + i) / scale, -448, 448));
            for (0..dsa.latent / 128) |g| @memcpy(rec[dsa.latent + g * 4 ..][0..4], std.mem.asBytes(&scale));
        }
        try buffer.upload(host);
        self.buffers[self.len] = buffer;
        self.len += 1;
        self.bytes += buffer.bytes();
    }

    /// The position `at` and the identity table over the held extents.
    fn addPlace(self: *Set, at: usize, positions: usize) !void {
        try self.addInts(&.{1}, &.{@intCast(at)});
        var table: [glm.bucket / 512]i32 = undefined;
        for (&table, 0..) |*extent, i| extent.* = @intCast(i);
        try self.addInts(&.{positions / 512}, table[0 .. positions / 512]);
    }

    fn view(self: Set, index: usize) cuda.View {
        return self.buffers[index].view;
    }
};

/// DSA's first row's position: `glm.pos`, or less where the window's rows would pass the held cache.
fn attendAt(rows: usize) usize {
    return @min(glm.pos, glm.held - rows);
}

fn build(gpa: Allocator, noise: Noise, case: Case, set: *Set) !void {
    const rows = case.rows;
    switch (case.kind) {
        .mix => {
            try set.add(gpa, noise, .bf16, &.{ rows, glm.wide }, .normal);
            try set.add(gpa, noise, .bf16, &.{ glm.mix, glm.wide }, .small);
            try set.add(gpa, noise, .f32, &hc.partShape(rows), .zero);
        },
        .pre, .bare => {
            try set.add(gpa, noise, .bf16, &.{ rows, glm.wide }, .normal);
            // parts as `mix` leaves them near enough: dots about one, the squares' sum above zero
            try set.add(gpa, noise, .f32, &hc.partShape(rows), .unit);
            try set.add(gpa, noise, .f32, &.{glm.mix}, .small);
            try set.add(gpa, noise, .f32, &.{3}, .unit);
            try set.add(gpa, noise, .bf16, &.{glm.dims}, .unit);
            try set.add(gpa, noise, .bf16, &.{ rows, glm.dims }, .zero);
            try set.add(gpa, noise, .f32, &.{ rows, hc.streams }, .zero);
            try set.add(gpa, noise, .f32, &.{ rows, hc.streams, hc.streams }, .zero);
        },
        .post => {
            try set.add(gpa, noise, .bf16, &.{ rows, glm.wide }, .normal);
            try set.add(gpa, noise, .f32, &.{ glm.world, rows, glm.dims }, .small);
            try set.add(gpa, noise, .f32, &.{ rows, hc.streams }, .unit);
            try set.add(gpa, noise, .f32, &.{ rows, hc.streams, hc.streams }, .small);
        },
        .chain => {
            try set.add(gpa, noise, .bf16, &.{ rows, glm.projection }, .normal);
            try set.add(gpa, noise, .bf16, &.{ rows, glm.inner }, .normal);
            try set.add(gpa, noise, .bf16, &.{ rows, glm.inner }, .normal);
            try set.add(gpa, noise, .bf16, &.{ 3, 3 * glm.inner }, .normal);
            try set.add(gpa, noise, .bf16, &.{ 3 * glm.inner, 4 }, .small);
            try set.add(gpa, noise, .f32, &.{ glm.heads, glm.head, glm.head }, .small);
            try set.add(gpa, noise, .f32, &.{glm.heads}, .small);
            try set.add(gpa, noise, .f32, &.{glm.inner}, .small);
            try set.add(gpa, noise, .bf16, &.{glm.head}, .unit);
            try set.add(gpa, noise, .bf16, &.{ rows, glm.inner }, .zero);
            try set.add(gpa, noise, .f32, &.{ glm.heads, glm.head, glm.head }, .zero);
            try set.add(gpa, noise, .f32, &.{ rows, glm.heads, glm.head }, .zero);
            try set.add(gpa, noise, .bf16, &.{ rows, glm.heads, glm.head }, .zero);
            try set.add(gpa, noise, .f32, &.{ rows, glm.heads, glm.head }, .zero);
            try set.add(gpa, noise, .f32, &.{ rows, glm.heads }, .zero);
            if (rows >= 32) {
                // the chunked chain's scratch, decays and read-out
                const many: usize = @intCast(zs_glm5_kda_chunks(@intCast(rows), 0) * glm.heads);
                try set.add(gpa, noise, .bf16, &.{ many, @intCast(zs_glm5_kda_chunk_elems()) }, .zero);
                try set.add(gpa, noise, .f32, &.{ many, glm.head }, .zero);
                try set.add(gpa, noise, .bf16, &.{ rows, glm.heads, glm.head }, .zero);
            }
        },
        .absorb => {
            try set.add(gpa, noise, .bf16, &.{ rows, glm.heads * glm.key }, .normal);
            try set.add(gpa, noise, .bf16, &.{ glm.heads, 2 * glm.key, glm.latent }, .small);
            try set.add(gpa, noise, .bf16, &.{ rows, glm.heads, glm.latent }, .zero);
        },
        .expand => {
            try set.add(gpa, noise, .bf16, &.{ rows, glm.heads, glm.latent }, .normal);
            try set.add(gpa, noise, .bf16, &.{ glm.heads, 2 * glm.key, glm.latent }, .small);
            try set.add(gpa, noise, .bf16, &.{ rows, glm.heads * glm.key }, .zero);
        },
        .attend, .attend8 => {
            const chunks = (glm.width + dsa.chunk - 1) / dsa.chunk;
            try set.add(gpa, noise, .bf16, &.{ rows, glm.heads, glm.latent }, .normal);
            if (case.kind == .attend) try set.add(gpa, noise, .bf16, &.{ glm.held, glm.latent }, .small) else try set.addRecords(gpa, noise, glm.held);
            try set.addPlace(attendAt(rows), glm.held);
            // the partials as `dsa.Buffers` holds them: a window's running output alone from 128 rows
            const held = dsa.Buffers.parted(rows, chunks);
            try set.add(gpa, noise, .f32, &.{ held, chunks, glm.heads, glm.latent }, .zero);
            try set.add(gpa, noise, .f32, &.{ held, chunks, glm.heads }, .zero);
            try set.add(gpa, noise, .f32, &.{ held, chunks, glm.heads }, .zero);
            try set.add(gpa, noise, .bf16, &.{ rows, glm.heads, glm.latent }, .zero);
            // a row's list: `kept` of its complete pools in pool order, then the unfinished pool's tokens
            const tokens = try gpa.alloc(i32, rows * glm.width);
            defer gpa.free(tokens);
            const counts = try gpa.alloc(i32, rows);
            defer gpa.free(counts);
            const chosen = try gpa.alloc(u32, glm.kept);
            defer gpa.free(chosen);
            @memset(tokens, -1);
            for (0..rows) |r| {
                const end = attendAt(rows) + r + 1;
                const complete = end / 4;
                const stride = complete / glm.kept;
                for (chosen, 0..) |*pool, i| pool.* = @intCast(i * stride + noise.random.uintLessThan(usize, stride));
                const list = tokens[r * glm.width ..][0..glm.width];
                for (chosen, 0..) |pool, i| {
                    for (0..4) |k| list[i * 4 + k] = @intCast(pool * 4 + k);
                }
                const rest = end - 4 * complete;
                for (0..rest) |k| list[glm.kept * 4 + k] = @intCast(4 * complete + k);
                counts[r] = @intCast(glm.kept * 4 + rest);
            }
            try set.addInts(&.{ rows, glm.width }, tokens);
            try set.addInts(&.{rows}, counts);
        },
        .keys => {
            try set.add(gpa, noise, .bf16, &.{ rows, glm.index_dim }, .normal);
            try set.add(gpa, noise, .f32, &.{ rows, glm.index_dim }, .normal);
            try set.add(gpa, noise, .bf16, &.{glm.index_dim}, .unit);
            try set.add(gpa, noise, .bf16, &.{glm.index_dim}, .small);
            try set.add(gpa, noise, .bf16, &.{ glm.held, glm.index_dim }, .zero);
            try set.add(gpa, noise, .bf16, &.{ glm.held, glm.index_dim }, .zero);
            try set.addPlace(glm.pos, glm.held);
        },
        .pool => {
            try set.add(gpa, noise, .bf16, &.{ glm.bucket, glm.index_dim }, .normal);
            try set.add(gpa, noise, .bf16, &.{ glm.bucket, glm.index_dim }, .normal);
            try set.add(gpa, noise, .bf16, &.{ 4, glm.index_dim }, .small);
            try set.add(gpa, noise, .u8, &.{ glm.bucket / 4, glm.index_dim }, .zero);
            try set.add(gpa, noise, .f32, &.{glm.bucket / 4}, .zero);
            try set.addPlace(glm.pos, glm.bucket);
        },
        .score => {
            try set.add(gpa, noise, .u8, &.{ rows, glm.index_heads, glm.index_dim }, .normal);
            try set.add(gpa, noise, .f32, &.{ rows, glm.index_heads }, .normal);
            try set.add(gpa, noise, .u8, &.{ glm.bucket / 4, glm.index_dim }, .normal);
            try set.add(gpa, noise, .f32, &.{glm.bucket / 4}, .unit);
            try set.add(gpa, noise, .f32, &.{ rows, glm.pools }, .zero);
            try set.addPlace(glm.pos, glm.held);
        },
        // the queries coded, then scored: what the bf16 score was before spec 116 WS-05
        .window => {
            try set.add(gpa, noise, .bf16, &.{ rows, glm.index_heads, glm.index_dim }, .normal);
            try set.add(gpa, noise, .f32, &.{ rows, glm.index_heads }, .normal);
            try set.add(gpa, noise, .u8, &.{ rows, glm.index_heads, glm.index_dim }, .zero);
            try set.add(gpa, noise, .f32, &.{ rows, glm.index_heads }, .zero);
            try set.add(gpa, noise, .u8, &.{ glm.window_context / 4, glm.index_dim }, .normal);
            try set.add(gpa, noise, .f32, &.{glm.window_context / 4}, .unit);
            try set.add(gpa, noise, .f32, &.{ rows, glm.window_context / 4 }, .zero);
            try set.addInts(&.{1}, &.{glm.window_pos});
        },
        .select => {
            try set.add(gpa, noise, .f32, &.{ rows, glm.pools }, .normal);
            try set.addInts(&.{1}, &.{glm.pos});
            const tokens = try gpa.alloc(i32, rows * glm.width);
            defer gpa.free(tokens);
            @memset(tokens, 0);
            try set.addInts(&.{ rows, glm.width }, tokens);
            try set.addInts(&.{rows}, tokens[0..rows]);
        },
    }
}

fn launch(case: Case, set: Set) !void {
    switch (case.kind) {
        .mix => try hc.mix(set.view(0), set.view(1), set.view(2)),
        .pre, .bare => try hc.pre(set.view(0), set.view(1), .{ .dots = set.view(0), .base = set.view(2), .scale = set.view(3), .norm = set.view(4) }, set.view(5), set.view(6), set.view(7), glm.eps, glm.hc_eps, if (case.kind == .pre) glm.iters else 1),
        .post => try hc.post(set.view(0), set.view(1), set.view(2), set.view(3)),
        .chain => if (case.rows >= 32)
            try cuda.launched(zs_glm5_kda_chunked(&set.view(0), 3 * glm.inner + 2 * glm.head, &set.view(1), &set.view(2), &set.view(3), &set.view(4), &set.view(5), &set.view(6), &set.view(7), &set.view(8), glm.eps, -5.0, @intCast(case.rows), 0, &set.view(9), &set.view(10), &set.view(15), &set.view(16), &set.view(17), 0, &no_rows, &no_views))
        else
            try cuda.launched(zs_glm5_kda_chain(&set.view(0), 3 * glm.inner + 2 * glm.head, &set.view(1), &set.view(2), &set.view(3), &set.view(4), &set.view(5), &set.view(6), &set.view(7), &set.view(8), glm.eps, 0.1, @intCast(case.rows), &set.view(9), &set.view(10), &set.view(11), &set.view(12), &set.view(13), &set.view(14))),
        .absorb => try dsa.absorb(set.view(0), set.view(1), glm.key, set.view(2)),
        .expand => try dsa.expand(set.view(0), set.view(1), glm.key, set.view(2)),
        .attend, .attend8 => try dsa.attendRows(set.view(0), set.view(1), set.view(2), &set.view(3), .{ set.view(4), set.view(5), set.view(6) }, set.view(7), .{ set.view(8), set.view(9) }, 1.0 / 16.0, attendAt(case.rows) + case.rows),
        .keys => try index_launch.write(&set.view(0), &set.view(1), &set.view(2), &set.view(3), &set.view(4), &set.view(5), &set.view(7), &set.view(6), case.rows, glm.held),
        .pool => try index_launch.pool(&set.view(0), &set.view(1), &set.view(6), &set.view(2), &set.view(3), &set.view(4), &set.view(5), case.rows, glm.bucket),
        .score => try index_launch.score(&set.view(0), &set.view(1), &set.view(2), &set.view(3), &set.view(5), &set.view(4), glm.bucket),
        .window => {
            try index_launch.query(&set.view(0), &set.view(1), &set.view(2), &set.view(3));
            try index_launch.score(&set.view(2), &set.view(3), &set.view(4), &set.view(5), &set.view(7), &set.view(6), glm.window_context);
        },
        .select => try index_launch.select(&set.view(0), &set.view(1), &set.view(2), &set.view(3), glm.kept, glm.pos + case.rows),
    }
}

extern fn zs_glm5_kda_chain(p: *const cuda.View, b_off: i64, a: *const cuda.View, g: *const cuda.View, cs: *const cuda.View, cw: *const cuda.View, state_in: *const cuda.View, a_log: *const cuda.View, dt_bias: *const cuda.View, norm_w: *const cuda.View, eps: f64, lower: f64, rows: i64, out: *const cuda.View, state_out: ?*const cuda.View, k_save: ?*const cuda.View, v_save: ?*const cuda.View, g_save: ?*const cuda.View, b_save: ?*const cuda.View) c_int;
extern fn zs_glm5_kda_chunked(p: *const cuda.View, b_off: i64, a: *const cuda.View, g: ?*const cuda.View, cs: *const cuda.View, cw: *const cuda.View, state_in: *const cuda.View, a_log: *const cuda.View, dt_bias: *const cuda.View, norm_w: ?*const cuda.View, eps: f64, lower: f64, rows: i64, phase: i64, out: ?*const cuda.View, state_out: *const cuda.View, scratch: *const cuda.View, last: *const cuda.View, y_tmp: ?*const cuda.View, asked_n: i64, asked_rows: [*]const i64, asked_to: [*]const cuda.View) c_int;
const no_rows = [_]i64{0};
const no_views = [_]cuda.View{.{}};
extern fn zs_glm5_kda_chunks(rows: i64, phase: i64) i64;
extern fn zs_glm5_kda_chunk_elems() i64;

fn run(gpa: Allocator, noise: Noise, case: Case, index: usize, marker: Marker, full: bool) !void {
    var first: Set = .{};
    try build(gpa, noise, case, &first);
    // a row gathers 2 MB of a cache: twelve caches, so that L2 holds no launch's keys for the next
    const copies: usize = if (!full or case.hot) 1 else if (case.kind == .attend or case.kind == .attend8) (if (case.rows > 16) 2 else 12) else std.math.clamp((96 << 20) / first.bytes, 2, 64);
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
    var tail: ?cuda.Buffer = if (full) try .init(.u8, &.{64 << 20}) else null;
    defer if (tail) |*buffer| buffer.deinit();
    const launches = if (full and case.hot) hot_launches else copies;
    var times: [passes]u64 = undefined;
    const timed = if (full) times[0..] else times[0..1];
    try marker.at(index, false);
    for (timed) |*time| {
        try cuda.synchronize();
        const t0 = nowNs();
        for (0..launches) |i| try launch(case, sets[i % copies]);
        try cuda.synchronize();
        time.* = nowNs() - t0;
    }
    try marker.at(index, true);
    // a score past a row's complete pools is -inf; DSA's partials are scratch a window's form may leave as it likes
    for (sets[0].buffers[0..sets[0].len], 0..) |buffer, at| {
        if (case.kind == .score and at == 4) continue;
        if (case.kind == .window and at == 6) continue;
        if ((case.kind == .attend or case.kind == .attend8) and at >= 4 and at <= 6) continue;
        try expectFinite(gpa, buffer.view);
    }
    if (!full) return;
    var buffer: [64]u8 = undefined;
    // the attention's output as bits, what two builds' forms are held to (spec 116 WS-01)
    if (case.kind == .attend or case.kind == .attend8) {
        const out = sets[0].view(7);
        const bytes = try gpa.alloc(u8, out.count() * out.dtype.size());
        defer gpa.free(bytes);
        try cuda.download(out, bytes);
        std.debug.print("PROFILE {s}: output hash {x:0>16}\n", .{ case.name(&buffer), std.hash.Wyhash.hash(0, bytes) });
    }
    std.mem.sort(u64, timed, {}, std.sort.asc(u64));
    std.debug.print("PROFILE {s}: {d} launches a pass, {d} bytes a set; a launch by the loop {d:.1} us median, {d:.1} best\n", .{
        case.name(&buffer),                                                                      launches,                                                                    first.bytes,
        @as(f64, @floatFromInt(timed[timed.len / 2])) / 1e3 / @as(f64, @floatFromInt(launches)), @as(f64, @floatFromInt(timed[0])) / 1e3 / @as(f64, @floatFromInt(launches)),
    });
}

test "PROFILE GLM-5.3-Flash's own kernels launched alone at a rank's shapes: a row, a draft window, a prompt chunk" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(73);
    var noise: Noise = try .init(gpa, prng.random());
    defer noise.deinit(gpa);
    const asked: ?[]const u8 = if (std.c.getenv("ZSPARK_GLM5_PROFILE")) |text| std.mem.span(text) else null;
    var marker: Marker = .{ .buffer = try .init(.u8, &.{mark + 2 * cases.len + 2}), .mark = mark };
    defer marker.buffer.deinit();

    // the switches as `main` reads them: a form's cases are the base's under its switch
    hc.lanes = if (std.c.getenv("ZSPARK_GLM_HC_LANES")) |text| text[0] != '0' else false;
    hc.octs = if (std.c.getenv("ZSPARK_GLM_HC_OCTS")) |text| text[0] != '0' else false;
    hc.setForms();
    dsa.setAsk(if (std.c.getenv("ZSPARK_GLM_MLA_ASK")) |text| std.fmt.parseInt(i32, std.mem.span(text), 10) catch 0 else 0);
    defer dsa.setAsk(0);
    dsa.setQ8(if (std.c.getenv("ZSPARK_GLM_MLA_Q8")) |text| text[0] != '0' else true);
    defer dsa.setQ8(true);
    kda.setQuads(if (std.c.getenv("ZSPARK_GLM_KDA_QUADS")) |text| text[0] != '0' else false);
    defer kda.setQuads(false);
    defer {
        hc.lanes = false;
        hc.octs = false;
        hc.setForms();
    }
    var ran: usize = 0;
    for (cases, 0..) |case, index| {
        if (asked) |want| {
            var buffer: [64]u8 = undefined;
            if (!std.mem.eql(u8, want, "all") and !std.mem.eql(u8, want, case.name(&buffer)) and !std.mem.eql(u8, want, @tagName(case.kind))) continue;
        } else if (case.hot or (case.rows != 1 and case.rows != 4)) continue;
        ran += 1;
        try run(gpa, noise, case, index, marker, asked != null);
    }
    // a name that is no case's is a mistake, not a pass
    try std.testing.expect(ran > 0);
}
