//! `kernels.zig`'s tests on random rows at DFlash2's sizes: the convolution, the prep and the selection against
//! `kernels_ref.zig` bit for bit, a context row alone against the row in a batch; a block's attention near the host's
//! (an online softmax with bf16 probabilities, `near`'s bound) and bit for bit flat against the ring, with NaN
//! in every row of the context that no query of the launch sees.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const host = @import("../../../src/decode/dflash2/kernels_ref.zig");
const kernels = @import("../../../src/decode/dflash2/kernels.zig");
const shared = @import("../../models/glm5/testing.zig");
const device = shared.device;
const fetch = shared.fetch;

const fx = struct {
    const dims = 4096;
    /// `conv_group_size` 16: 256 groups of channels.
    const groups = dims / 16;
    /// A rank's query heads and KV heads, four query heads a KV head.
    const heads = 16;
    const kv = 4;
    const block = 8;
    const window = 2047;
    /// Window, block and a tile of 64, in whole tiles.
    const ring = 2176;
    const eps = 1e-5;
    const theta = 10000.0;
    /// A NaN: what a row holds that nothing may read.
    const nan: u16 = 0x7fc0;
};

const head = host.head;

/// A position on the device, int32 [1].
fn positionOn(pos: usize) !cuda.Buffer {
    const value: i32 = @intCast(pos);
    return device(i32, .i32, &.{1}, &.{value});
}

fn move(buffer: cuda.Buffer, pos: usize) !void {
    try cuda.synchronize();
    const value: i32 = @intCast(pos);
    try buffer.upload(std.mem.asBytes(&value));
}

fn allNan(row: []const u16) bool {
    for (row) |bits| if (!std.math.isNan(numeric.f32FromBf16(bits))) return false;
    return true;
}

/// A row of the attention against the host's, which takes the kernel's steps but for the order of an mma's
/// fp32 sums and `expf`. A rounding apart there leaves the result its bf16 neighbour (2^-7 of it), and now and
/// then turns a probability to its bf16 neighbour: 2^-7 of that key's share of the largest value it holds,
/// a share that is small among many keys; `slack` is that share's bound for the scene.
fn near(want: []const u16, got: []const u16, largest: f32, slack: f32) !void {
    for (want, got) |a, b| {
        const w = numeric.f32FromBf16(a);
        const bound = @abs(w) / 128.0 + largest * slack / 128.0;
        if (!(@abs(w - numeric.f32FromBf16(b)) <= bound)) {
            std.debug.print("attend: host {d} device {d}, bound {d}\n", .{ w, numeric.f32FromBf16(b), bound });
            return error.TestExpectedApproxEqAbs;
        }
    }
}

fn largestOf(values: []const u16) f32 {
    var most: f32 = 0;
    for (values) |bits| most = @max(most, @abs(numeric.f32FromBf16(bits)));
    return most;
}

fn noNan(row: []const u16) bool {
    for (row) |bits| if (!std.math.isFinite(numeric.f32FromBf16(bits))) return false;
    return true;
}

test "DFlash2 dconv: both branches, with and without the residual, are the host's bits; two rows alone and onto the residual too" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();
    const rows = fx.block;
    const d = fx.dims;
    const g = 4 * fx.groups;
    const x = try gpa.alloc(u16, rows * d);
    defer gpa.free(x);
    shared.randomBf16(random, x, 1.0);
    const dyn = try gpa.alloc(u16, rows * g);
    defer gpa.free(dyn);
    shared.randomBf16(random, dyn, 0.4);
    const base = try gpa.alloc(u16, 4 * d);
    defer gpa.free(base);
    shared.randomBf16(random, base, 0.6);
    const res = try gpa.alloc(u16, rows * d);
    defer gpa.free(res);
    shared.randomBf16(random, res, 2.0);
    var x_dev = try device(u16, .bf16, &.{ rows, d }, x);
    defer x_dev.deinit();
    var dyn_dev = try device(u16, .bf16, &.{ rows, g }, dyn);
    defer dyn_dev.deinit();
    var base_dev = try device(u16, .bf16, &.{ 2, 2, d }, base);
    defer base_dev.deinit();
    var res_dev = try device(u16, .bf16, &.{ rows, d }, res);
    defer res_dev.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ rows, d });
    defer out.deinit();
    var few: cuda.Buffer = try .init(.bf16, &.{ 2, d });
    defer few.deinit();
    const want = try gpa.alloc(u16, d);
    defer gpa.free(want);

    for ([_]u1{ 0, 1 }) |branch| for ([_]bool{ false, true }) |with| {
        try kernels.dconv(x_dev.view, dyn_dev.view, base_dev.view, branch, if (with) res_dev.view else null, out.view);
        const got = try fetch(u16, gpa, out);
        defer gpa.free(got);
        for (0..rows) |r| {
            const before: ?[]const u16 = if (r > 0) x[(r - 1) * d ..][0..d] else null;
            const residual: ?[]const u16 = if (with) res[r * d ..][0..d] else null;
            host.dconvRow(x[r * d ..][0..d], before, dyn[r * g ..][0..g], base, branch, residual, want);
            try testing.expectEqualSlices(u16, want, got[r * d ..][0..d]);
        }
        // the block's first two rows in a launch of their own
        try kernels.dconv(x_dev.span(0, 2), dyn_dev.span(0, 2), base_dev.view, branch, if (with) res_dev.span(0, 2) else null, few.view);
        const two = try fetch(u16, gpa, few);
        defer gpa.free(two);
        try testing.expectEqualSlices(u16, got[0 .. 2 * d], two);
        if (with) {
            var onto = try device(u16, .bf16, &.{ rows, d }, res);
            defer onto.deinit();
            try kernels.dconv(x_dev.view, dyn_dev.view, base_dev.view, branch, onto.view, onto.view);
            const placed = try fetch(u16, gpa, onto);
            defer gpa.free(placed);
            try testing.expectEqualSlices(u16, got, placed);
        }
    };
}

/// Rows as the drafter's two linears leave them: a row's query heads (none for context rows), and its key heads
/// then its value heads; the heads of different sizes so that a norm shows, and the two norms' weights.
const Rows = struct {
    count: usize,
    query_heads: usize,
    /// bf16 [count, query_heads * 128] and [count, 2 * kv * 128]
    q_rows: []u16,
    kv_rows: []u16,
    q_norm: [head]u16,
    k_norm: [head]u16,
    q_rows_dev: ?cuda.Buffer,
    kv_rows_dev: cuda.Buffer,
    q_norm_dev: cuda.Buffer,
    k_norm_dev: cuda.Buffer,

    fn init(gpa: Allocator, seed: u64, count: usize, query_heads: usize) !Rows {
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        const q_rows = try gpa.alloc(u16, count * query_heads * head);
        errdefer gpa.free(q_rows);
        for (0..count * query_heads) |h| shared.randomBf16(random, q_rows[h * head ..][0..head], 0.25 + @as(f32, @floatFromInt(h % 7)));
        const kv_rows = try gpa.alloc(u16, count * 2 * fx.kv * head);
        errdefer gpa.free(kv_rows);
        for (0..count * 2 * fx.kv) |h| shared.randomBf16(random, kv_rows[h * head ..][0..head], 0.25 + @as(f32, @floatFromInt(h % 5)));
        var q_norm: [head]u16 = undefined;
        var k_norm: [head]u16 = undefined;
        for (&q_norm, &k_norm) |*q, *k| {
            q.* = numeric.bf16FromF32(1.0 + 0.3 * random.floatNorm(f32));
            k.* = numeric.bf16FromF32(0.8 + 0.3 * random.floatNorm(f32));
        }
        var q_rows_dev: ?cuda.Buffer = if (query_heads > 0) try device(u16, .bf16, &.{ count, query_heads * head }, q_rows) else null;
        errdefer if (q_rows_dev) |*buffer| buffer.deinit();
        var kv_rows_dev = try device(u16, .bf16, &.{ count, 2 * fx.kv * head }, kv_rows);
        errdefer kv_rows_dev.deinit();
        var q_norm_dev = try device(u16, .bf16, &.{head}, &q_norm);
        errdefer q_norm_dev.deinit();
        const k_norm_dev = try device(u16, .bf16, &.{head}, &k_norm);
        return .{ .count = count, .query_heads = query_heads, .q_rows = q_rows, .kv_rows = kv_rows, .q_norm = q_norm, .k_norm = k_norm, .q_rows_dev = q_rows_dev, .kv_rows_dev = kv_rows_dev, .q_norm_dev = q_norm_dev, .k_norm_dev = k_norm_dev };
    }

    fn deinit(self: *Rows, gpa: Allocator) void {
        self.k_norm_dev.deinit();
        self.q_norm_dev.deinit();
        self.kv_rows_dev.deinit();
        if (self.q_rows_dev) |*buffer| buffer.deinit();
        gpa.free(self.kv_rows);
        gpa.free(self.q_rows);
    }

    /// Query head `h` of row `r`.
    fn query(self: *const Rows, r: usize, h: usize) *const [head]u16 {
        return self.q_rows[(r * self.query_heads + h) * head ..][0..head];
    }

    /// Key head `h` of row `r`, and its value head.
    fn key(self: *const Rows, r: usize, h: usize) *const [head]u16 {
        return self.kv_rows[(r * 2 * fx.kv + h) * head ..][0..head];
    }

    fn value(self: *const Rows, r: usize, h: usize) *const [head]u16 {
        return self.kv_rows[(r * 2 * fx.kv + fx.kv + h) * head ..][0..head];
    }

    /// Rows first .. first + count - 1 at positions from `place`'s on.
    fn prep(self: *const Rows, first: usize, count: usize, place: kernels.Place, q: ?cuda.View, k: cuda.View, v: cuda.View) !void {
        const q_rows: ?cuda.View = if (self.q_rows_dev) |buffer| buffer.span(first, count) else null;
        try kernels.prep(q_rows, self.kv_rows_dev.span(first, count), self.q_norm_dev.view, self.k_norm_dev.view, place, q, k, v, fx.eps, fx.theta);
    }
};

/// A context's keys or values, bf16 [kv, capacity, 128], every row NaN.
fn emptyContext(gpa: Allocator, capacity: usize) !cuda.Buffer {
    const fill = try gpa.alloc(u16, fx.kv * capacity * head);
    defer gpa.free(fill);
    @memset(fill, fx.nan);
    return device(u16, .bf16, &.{ fx.kv, capacity, head }, fill);
}

test "DFlash2 prep: a block's queries, keys and values are the host's bits at their positions' rows, flat and in a ring" {
    const gpa = testing.allocator;
    const rows = fx.block;
    var case: Rows = try .init(gpa, 68, rows, fx.heads);
    defer case.deinit(gpa);
    var q_dev: cuda.Buffer = try .init(.bf16, &.{ fx.heads, rows, head });
    defer q_dev.deinit();
    var pos_dev = try positionOn(0);
    defer pos_dev.deinit();
    var want: [head]u16 = undefined;
    // the first position, one inside the window, a block across the ring's end, one whose phases take many turns
    for ([_]usize{ 0, 100, 2 * fx.ring - 3, 1_000_003 }) |pos| for ([_]bool{ true, false }) |ring| {
        if (!ring and pos + rows > fx.ring) continue;
        var k_dev = try emptyContext(gpa, fx.ring);
        defer k_dev.deinit();
        var v_dev = try emptyContext(gpa, fx.ring);
        defer v_dev.deinit();
        try move(pos_dev, pos);
        try case.prep(0, rows, .{ .pos = pos_dev.view, .ring = ring }, q_dev.view, k_dev.view, v_dev.view);
        const q = try fetch(u16, gpa, q_dev);
        defer gpa.free(q);
        const k = try fetch(u16, gpa, k_dev);
        defer gpa.free(k);
        const v = try fetch(u16, gpa, v_dev);
        defer gpa.free(v);
        for (0..rows) |r| {
            const slot = (pos + r) % fx.ring;
            for (0..fx.heads) |h| {
                host.prepHead(case.query(r, h), &case.q_norm, pos + r, fx.eps, fx.theta, &want);
                try testing.expectEqualSlices(u16, &want, q[(h * rows + r) * head ..][0..head]);
            }
            for (0..fx.kv) |h| {
                host.prepHead(case.key(r, h), &case.k_norm, pos + r, fx.eps, fx.theta, &want);
                try testing.expectEqualSlices(u16, &want, k[(h * fx.ring + slot) * head ..][0..head]);
                try testing.expectEqualSlices(u16, case.value(r, h), v[(h * fx.ring + slot) * head ..][0..head]);
            }
        }
        // no other row of the context was written
        for (0..fx.kv * fx.ring) |row| {
            if ((row % fx.ring + fx.ring - pos % fx.ring) % fx.ring < rows) continue;
            try testing.expect(allNan(k[row * head ..][0..head]) and allNan(v[row * head ..][0..head]));
        }
    };
}

test "DFlash2 prep: a context row's key and value are the same bits alone, in a batch of 64, at any offset of a launch, flat and through the ring's wrap" {
    const gpa = testing.allocator;
    const rows = 64;
    // the batch's row 20 is the ring's row 0
    const pos = 3 * fx.ring - 20;
    var case: Rows = try .init(gpa, 69, rows, 0);
    defer case.deinit(gpa);
    var pos_dev = try positionOn(pos);
    defer pos_dev.deinit();

    var k_batch = try emptyContext(gpa, fx.ring);
    defer k_batch.deinit();
    var v_batch = try emptyContext(gpa, fx.ring);
    defer v_batch.deinit();
    try case.prep(0, rows, .{ .pos = pos_dev.view, .ring = true }, null, k_batch.view, v_batch.view);
    const k = try fetch(u16, gpa, k_batch);
    defer gpa.free(k);
    const v = try fetch(u16, gpa, v_batch);
    defer gpa.free(v);

    var want: [head]u16 = undefined;
    for (0..rows) |r| for (0..fx.kv) |h| {
        const at = (h * fx.ring + (pos + r) % fx.ring) * head;
        host.prepHead(case.key(r, h), &case.k_norm, pos + r, fx.eps, fx.theta, &want);
        try testing.expectEqualSlices(u16, &want, k[at..][0..head]);
        try testing.expectEqualSlices(u16, case.value(r, h), v[at..][0..head]);
    };

    // a row a launch, then launches of 7 rows: the whole ring is the batch's, the rows not written too
    for ([_]usize{ 1, 7 }) |step| {
        var k_dev = try emptyContext(gpa, fx.ring);
        defer k_dev.deinit();
        var v_dev = try emptyContext(gpa, fx.ring);
        defer v_dev.deinit();
        var first: usize = 0;
        while (first < rows) : (first += step) {
            try move(pos_dev, pos + first);
            try case.prep(first, @min(step, rows - first), .{ .pos = pos_dev.view, .ring = true }, null, k_dev.view, v_dev.view);
        }
        const k_got = try fetch(u16, gpa, k_dev);
        defer gpa.free(k_got);
        const v_got = try fetch(u16, gpa, v_dev);
        defer gpa.free(v_got);
        try testing.expectEqualSlices(u16, k, k_got);
        try testing.expectEqualSlices(u16, v, v_got);
    }

    // flat: position p is row p
    const capacity = pos + rows + 3;
    var k_flat = try emptyContext(gpa, capacity);
    defer k_flat.deinit();
    var v_flat = try emptyContext(gpa, capacity);
    defer v_flat.deinit();
    try move(pos_dev, pos);
    try case.prep(0, rows, .{ .pos = pos_dev.view, .ring = false }, null, k_flat.view, v_flat.view);
    const k_got = try fetch(u16, gpa, k_flat);
    defer gpa.free(k_got);
    const v_got = try fetch(u16, gpa, v_flat);
    defer gpa.free(v_got);
    for (0..fx.kv) |h| for (0..capacity) |p| {
        const flat = (h * capacity + p) * head;
        if (p < pos or p >= pos + rows) {
            try testing.expect(allNan(k_got[flat..][0..head]) and allNan(v_got[flat..][0..head]));
            continue;
        }
        const at = (h * fx.ring + p % fx.ring) * head;
        try testing.expectEqualSlices(u16, k[at..][0..head], k_got[flat..][0..head]);
        try testing.expectEqualSlices(u16, v[at..][0..head], v_got[flat..][0..head]);
    };
}

/// A block's queries and a context by position: keys and values bf16 [kv][positions, 128] on the host.
const Scene = struct {
    block: host.Block,
    q: []u16,
    keys: []u16,
    values: []u16,
    q_dev: cuda.Buffer,
    pos_dev: cuda.Buffer,
    out_dev: cuda.Buffer,

    fn init(gpa: Allocator, seed: u64, pos: usize, window: usize) !Scene {
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        const positions = pos + fx.block;
        const q = try gpa.alloc(u16, fx.heads * fx.block * head);
        errdefer gpa.free(q);
        // scores a few units apart
        shared.randomBf16(random, q, 1.5);
        const keys = try gpa.alloc(u16, fx.kv * positions * head);
        errdefer gpa.free(keys);
        shared.randomBf16(random, keys, 1.5);
        const values = try gpa.alloc(u16, fx.kv * positions * head);
        errdefer gpa.free(values);
        shared.randomBf16(random, values, 1.0);
        var q_dev = try device(u16, .bf16, &.{ fx.heads, fx.block, head }, q);
        errdefer q_dev.deinit();
        var pos_dev = try positionOn(pos);
        errdefer pos_dev.deinit();
        const out_dev: cuda.Buffer = try .init(.bf16, &.{ fx.block, fx.heads * head });
        return .{ .block = .{ .pos = pos, .rows = fx.block, .window = window }, .q = q, .keys = keys, .values = values, .q_dev = q_dev, .pos_dev = pos_dev, .out_dev = out_dev };
    }

    fn deinit(self: *Scene, gpa: Allocator) void {
        self.out_dev.deinit();
        self.pos_dev.deinit();
        self.q_dev.deinit();
        gpa.free(self.values);
        gpa.free(self.keys);
        gpa.free(self.q);
    }

    /// The first position a query of the block sees.
    fn first(self: *const Scene) usize {
        return self.block.seen(0).first;
    }

    /// The block's attention over a context of `capacity` rows, flat or a ring, that holds the positions a
    /// query of the block sees, and NaN in every other row. The positions from `lost` to `lost_end` hold
    /// another key (the sign turned) and a value of 1000s: a query that sees one moves in every head, and one
    /// that does not keeps its bits (a NaN there would reach it: its p = 0 is multiplied, not skipped). The
    /// caller frees the result, bf16 [rows, heads * 128].
    fn run(self: *const Scene, gpa: Allocator, capacity: usize, ring: bool, causal: bool, lost: usize, lost_end: usize) ![]u16 {
        const positions = self.block.pos + fx.block;
        const k = try gpa.alloc(u16, fx.kv * capacity * head);
        defer gpa.free(k);
        const v = try gpa.alloc(u16, fx.kv * capacity * head);
        defer gpa.free(v);
        @memset(k, fx.nan);
        @memset(v, fx.nan);
        for (0..fx.kv) |h| for (self.first()..positions) |p| {
            const slot = if (ring) p % capacity else p;
            @memcpy(k[(h * capacity + slot) * head ..][0..head], self.keys[(h * positions + p) * head ..][0..head]);
            @memcpy(v[(h * capacity + slot) * head ..][0..head], self.values[(h * positions + p) * head ..][0..head]);
            if (p >= lost and p < lost_end) {
                for (k[(h * capacity + slot) * head ..][0..head]) |*bits| bits.* ^= 0x8000;
                @memset(v[(h * capacity + slot) * head ..][0..head], numeric.bf16FromF32(1000.0));
            }
        };
        var parts: cuda.Buffer = try .init(.f32, &.{kernels.attendParts(fx.heads, fx.block, self.block.window)});
        defer parts.deinit();
        var k_dev = try device(u16, .bf16, &.{ fx.kv, capacity, head }, k);
        defer k_dev.deinit();
        var v_dev = try device(u16, .bf16, &.{ fx.kv, capacity, head }, v);
        defer v_dev.deinit();
        try kernels.attend(self.q_dev.view, k_dev.view, v_dev.view, .{ .pos = self.pos_dev.view, .ring = ring }, self.out_dev.view, parts.view, self.block.window, causal);
        return fetch(u16, gpa, self.out_dev);
    }

    /// Whether every head of the rows differs between two results.
    fn moved(a: []const u16, b: []const u16) bool {
        var at: usize = 0;
        while (at < a.len) : (at += head) if (std.mem.eql(u16, a[at..][0..head], b[at..][0..head])) return false;
        return true;
    }

    /// Row `row` of head `h` on the host.
    fn hostRow(self: *const Scene, causal: bool, row: usize, h: usize, out: *[head]u16) void {
        const positions = self.block.pos + fx.block;
        const kvh = h / (fx.heads / fx.kv);
        var block = self.block;
        block.causal = causal;
        host.attendRow(self.q[(h * fx.block + row) * head ..][0..head], self.keys[kvh * positions * head ..][0 .. positions * head], self.values[kvh * positions * head ..][0 .. positions * head], block, row, out);
    }
};

test "DFlash2 attend: a block is near the host's and the same bits flat, in the ring and in a ring of another size, under the window, at it and past the ring's wrap, with NaN in every row no query sees" {
    const gpa = testing.allocator;
    var want: [head]u16 = undefined;
    // no context; under the window; the window's last whole one and the first cut; a window across the ring's
    // end; a block across it
    for ([_]usize{ 0, 100, fx.window, fx.window + 1, 2 * fx.ring + 500, 3 * fx.ring - 4 }, 0..) |pos, n| {
        var scene: Scene = try .init(gpa, 70 + pos, pos, fx.window);
        defer scene.deinit(gpa);
        const flat = try scene.run(gpa, pos + fx.block + 5, false, false, 0, 0);
        defer gpa.free(flat);
        try testing.expect(noNan(flat));
        // every query on the host for half of the positions, the block's ends for the others
        for (0..fx.block) |row| {
            if (n % 2 == 1 and row != 0 and row != fx.block - 1) continue;
            for (0..fx.heads) |h| {
                scene.hostRow(false, row, h, &want);
                try near(&want, flat[(row * fx.heads + h) * head ..][0..head], largestOf(scene.values), 0.05);
            }
        }
        for ([_]usize{ fx.ring, 4096, fx.window + fx.block }) |capacity| {
            const ringed = try scene.run(gpa, capacity, true, false, 0, 0);
            defer gpa.free(ringed);
            try testing.expectEqualSlices(u16, flat, ringed);
        }
    }
}

test "DFlash2 attend: a query sees a context key 2047 positions back and not the one before it; causal, a row sees the block to itself" {
    const gpa = testing.allocator;
    const pos = 2 * fx.ring + 500;
    var scene: Scene = try .init(gpa, 71, pos, fx.window);
    defer scene.deinit(gpa);
    const row_bytes = fx.heads * head;
    const whole = try scene.run(gpa, fx.ring, true, false, 0, 0);
    defer gpa.free(whole);
    // rows 0 .. r - 1 see the positions pos - window .. pos - window + r - 1, and row r sees none of them
    for ([_]usize{ 1, 4, 7 }) |r| {
        const got = try scene.run(gpa, fx.ring, true, false, pos - fx.window, pos - fx.window + r);
        defer gpa.free(got);
        try testing.expect(Scene.moved(whole[0 .. r * row_bytes], got[0 .. r * row_bytes]));
        try testing.expectEqualSlices(u16, whole[r * row_bytes ..], got[r * row_bytes ..]);
    }

    var want: [head]u16 = undefined;
    const causal = try scene.run(gpa, fx.ring, true, true, 0, 0);
    defer gpa.free(causal);
    for (0..fx.block) |row| for ([_]usize{ 0, 6, 15 }) |h| {
        scene.hostRow(true, row, h, &want);
        try near(&want, causal[(row * fx.heads + h) * head ..][0..head], largestOf(scene.values), 0.05);
    };
    try testing.expect(!std.mem.eql(u16, whole[0..row_bytes], causal[0..row_bytes]));
    // the block's last key lost: its own row alone reads it
    const got = try scene.run(gpa, fx.ring, true, true, pos + fx.block - 1, pos + fx.block);
    defer gpa.free(got);
    try testing.expectEqualSlices(u16, causal[0 .. (fx.block - 1) * row_bytes], got[0 .. (fx.block - 1) * row_bytes]);
    try testing.expect(Scene.moved(causal[(fx.block - 1) * row_bytes ..], got[(fx.block - 1) * row_bytes ..]));
    // and not causal every row reads it
    const open = try scene.run(gpa, fx.ring, true, false, pos + fx.block - 1, pos + fx.block);
    defer gpa.free(open);
    try testing.expect(Scene.moved(whole, open));
}

test "DFlash2 attend: under a window shorter than the block a query still sees the whole block, and no context past its window" {
    const gpa = testing.allocator;
    const window = 3;
    const pos = fx.ring + 40;
    var scene: Scene = try .init(gpa, 72, pos, window);
    defer scene.deinit(gpa);
    var want: [head]u16 = undefined;
    const flat = try scene.run(gpa, pos + fx.block, false, false, 0, 0);
    defer gpa.free(flat);
    for (0..fx.block) |row| for (0..fx.heads) |h| {
        scene.hostRow(false, row, h, &want);
        try near(&want, flat[(row * fx.heads + h) * head ..][0..head], largestOf(scene.values), 0.05);
    };
    // a ring of the fewest rows: window + block
    const ringed = try scene.run(gpa, window + fx.block, true, false, 0, 0);
    defer gpa.free(ringed);
    try testing.expectEqualSlices(u16, flat, ringed);
    // the context's last key lost: rows 0 to 2 read it, rows 3 to 7 stand past its window
    const got = try scene.run(gpa, fx.ring, true, false, pos - 1, pos);
    defer gpa.free(got);
    const row_bytes = fx.heads * head;
    try testing.expect(Scene.moved(flat[0 .. 3 * row_bytes], got[0 .. 3 * row_bytes]));
    try testing.expectEqualSlices(u16, flat[3 * row_bytes ..], got[3 * row_bytes ..]);
}

test "DFlash2 top: a row's largest are chain.largest's, value and token: among many equals, with NaN, infinities and both zeros, a row short of numbers, and rows narrower than the block's threads" {
    const chain = @import("../../../src/decode/dflash2/chain.zig");
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();
    // a rank's half of the vocabulary (GLM's, Flash Next's) and the selector's 16; then what divides nothing, a row
    // of one slice and of two, and more picks than a slice's numbers
    for ([_][3]usize{ .{ 7, 77440, 16 }, .{ 7, 124160, 16 }, .{ 3, 1000, 16 }, .{ 2, 1025, 1 }, .{ 4, 16, 16 }, .{ 1, 3001, 64 }, .{ 5, 4097, 16 }, .{ 5, 8200, 40 } }) |shape| {
        const rows = shape[0];
        const cols = shape[1];
        const top = shape[2];
        const offset = 77440;
        const logits = try arena.alloc(u16, rows * cols);
        // bf16 of a normal: a few thousand values, so equals at every rank of a row
        shared.randomBf16(random, logits, 3.0);
        if (rows > 1) {
            // NaN of both signs, infinities and zeros among them
            const row = logits[cols..][0..cols];
            for (0..cols / 7 + 3) |_| row[random.uintLessThan(usize, cols)] = ([_]u16{ 0x7fc0, 0xffc0, 0x7f80, 0xff80, 0x0000, 0x8000 })[random.uintLessThan(usize, 6)];
        }
        if (rows > 2) {
            // numbers fewer than `top`, when the row is wide enough to hold them apart
            const row = logits[2 * cols ..][0..cols];
            @memset(row, fx.nan);
            for (0..@min(5, cols / 3)) |n| row[cols - 1 - 2 * n] = numeric.bf16FromF32(@floatFromInt(n % 3));
        }
        if (rows > 3) @memset(logits[3 * cols ..][0..cols], numeric.bf16FromF32(-1.5));
        if (rows > 4) for (logits[4 * cols ..][0..cols]) |*bits| {
            bits.* = if (random.boolean()) 0x0000 else 0x8000;
        };
        var logits_dev = try device(u16, .bf16, &.{ rows, cols }, logits);
        defer logits_dev.deinit();
        var values_dev: cuda.Buffer = try .init(.f32, &.{ rows, top });
        defer values_dev.deinit();
        var tokens_dev: cuda.Buffer = try .init(.i32, &.{ rows, top });
        defer tokens_dev.deinit();
        var scratch: cuda.Buffer = try .init(.i32, &.{@max(1, kernels.topScratch(rows, cols, top))});
        defer scratch.deinit();
        try kernels.top(logits_dev.view, values_dev.view, tokens_dev.view, scratch.view, offset);
        const values = try fetch(f32, arena, values_dev);
        const tokens = try fetch(i32, arena, tokens_dev);
        const want_values = try arena.alloc(f32, top);
        const want_tokens = try arena.alloc(i32, top);
        for (0..rows) |row| {
            chain.largest(logits[row * cols ..][0..cols], offset, want_values, want_tokens);
            try testing.expectEqualSlices(i32, want_tokens, tokens[row * top ..][0..top]);
            // a zero's sign is the picked token's
            try testing.expectEqualSlices(u32, @ptrCast(want_values), @ptrCast(values[row * top ..][0..top]));
        }
    }
}

test "DFlash2 rescore: a pick's value is its row's dot with the draft's, `host.rescore`'s bits; a pick out of the rank's half is -inf" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();
    // the model's dims and selector's 16, an odd dims, and a pick list at the kernel's 128
    for ([_][3]usize{ .{ 5, 6144, 16 }, .{ 3, 1000, 16 }, .{ 2, 4097, 1 }, .{ 1, 128, 128 } }) |shape| {
        const rows = shape[0];
        const dims = shape[1];
        const top = shape[2];
        const half = 77440;
        const offset = 77440;
        const head_rows = try arena.alloc(u16, half * dims);
        for (head_rows) |*bits| bits.* = numeric.bf16FromF32(random.float(f32) * 2 - 1);
        const final = try arena.alloc(u16, rows * dims);
        for (final) |*bits| bits.* = numeric.bf16FromF32(random.float(f32) * 2 - 1);
        const tokens = try arena.alloc(i32, rows * top);
        for (tokens) |*token| {
            // most in the rank's half, some before it and some past it
            token.* = @intCast(offset + random.intRangeAtMost(i64, -3, half + 2));
        }
        var rows_dev = try device(u16, .bf16, &.{ half, dims }, head_rows);
        defer rows_dev.deinit();
        var final_dev = try device(u16, .bf16, &.{ rows, dims }, final);
        defer final_dev.deinit();
        var tokens_dev = try device(i32, .i32, &.{ rows, top }, tokens);
        defer tokens_dev.deinit();
        var values_dev: cuda.Buffer = try .init(.f32, &.{ rows, top });
        defer values_dev.deinit();
        try kernels.rescore(tokens_dev.view, rows_dev.view, final_dev.view, values_dev.view, offset);
        const values = try fetch(f32, arena, values_dev);
        const want = try arena.alloc(f32, rows * top);
        host.rescore(tokens, head_rows, final, want, top, dims, offset);
        try testing.expectEqualSlices(u32, @ptrCast(want), @ptrCast(values));
    }
}
