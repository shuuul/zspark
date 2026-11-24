//! `attention.zig`'s tests on caches in order: each kernel against the same steps on the host, a row alone
//! against the row among others, and windows (with dropped rows) against serial steps through the whole block,
//! across the dense and sparse regimes. The tests through a table are `attention_table_test.zig`'s, the
//! selection's `attention_selection_test.zig`'s.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const fx = fixtures.attending.fx;
const Model = fixtures.attending.Model;
const Run = fixtures.attending.Run;
const Stored = fixtures.attending.Stored;
const e2m1FromF32 = fixtures.attending.e2m1FromF32;
const fetch = fixtures.attending.fetch;
const expectStored = fixtures.attending.expectStored;
const widen = fixtures.attending.widen;
const randomBf16 = fixtures.attending.randomBf16;
const runWindowsOver = fixtures.attending.runWindowsOver;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const bf = numeric.roundBf16;
const Cache = attention.Cache;
const Position = attention.Position;
const Rotary = attention.Rotary;
const Scratch = attention.Scratch;
const Shape = attention.Shape;
const Storage = attention.Storage;
const Weights = attention.Weights;
const attend = attention.attend;
const gate = attention.gate;
const pool = attention.pool;
const prep = attention.prep;

/// A head on the host: x / rms(x) * scale rounded to bf16, the first values rotated at `pos`. `slack` is the
/// magnitude each value's error scales with: a rounding of either operand of the rotation.
fn hostHead(x: []const f32, scale: []const f32, inv_freq: []const f32, pos: usize, eps: f32, want: []f32, slack: []f32) void {
    const half = inv_freq.len;
    var squares: f64 = 0;
    for (x) |v| squares += @as(f64, v) * v;
    const rinv = 1.0 / @sqrt(@as(f32, @floatCast(squares)) / @as(f32, @floatFromInt(x.len)) + eps);
    for (want, slack, 0..) |*out_value, *mag, d| {
        const xn = bf(x[d] * rinv * scale[d]);
        if (d >= 2 * half) {
            out_value.* = xn;
            mag.* = @abs(xn);
            continue;
        }
        const partner = if (d < half) d + half else d - half;
        const xpn = bf(x[partner] * rinv * scale[partner]);
        const angle = @as(f32, @floatFromInt(pos)) * inv_freq[if (d < half) d else d - half];
        out_value.* = if (d < half) xn * @cos(angle) - xpn * @sin(angle) else xpn * @sin(angle) + xn * @cos(angle);
        mag.* = @abs(xn) + @abs(xpn);
    }
}

test "prep normalizes, rotates and stores each head as the host does" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(31);
    const random = prng.random();
    const shape = fx.small;
    const rows = 3;
    const capacity = 5008;
    var model: Model = try .init(gpa, random, shape, rows);
    defer model.deinit();
    var cache: Cache = try .contiguous(shape, capacity);
    defer cache.deinit();
    var scratch: Scratch = try .init(shape, rows, capacity);
    defer scratch.deinit();
    var position: Position = try .init();
    defer position.deinit();

    const hd = shape.head_dim;
    const id = shape.index_dim;
    const width = shape.projection();
    const keys_at = shape.heads * 2 * hd;
    const values_at = keys_at + shape.kv_heads * hd;
    const index_at = values_at + shape.kv_heads * hd;
    const want = try gpa.alloc(f32, @max(hd, id));
    const slack = try gpa.alloc(f32, @max(hd, id));
    for ([_]usize{ 0, 37, 5000 }) |pos0| {
        const pa = try randomBf16(gpa, random, rows * width, 1.5);
        try scratch.pa.upload(std.mem.sliceAsBytes(pa));
        try position.set(pos0);
        try prep(shape, model.weights, model.rotary, cache, scratch, scratch.pa.view, position.device.view, rows, pos0 + rows, null);
        const q = try fetch(u16, gpa, scratch.q);
        const iq = try fetch(u16, gpa, scratch.iq);
        const k = try fetch(u16, gpa, cache.k);
        const v = try fetch(u16, gpa, cache.v);
        const index_keys = try fetch(u16, gpa, cache.index_keys);
        for (0..rows) |r| {
            const row = pa[r * width ..][0..width];
            const pos = pos0 + r;
            for (0..shape.heads) |h| {
                hostHead(try widen(gpa, row[h * 2 * hd ..][0..hd]), model.q_norm, model.inv_freq, pos, shape.eps, want[0..hd], slack[0..hd]);
                for (0..hd) |d| try expectStored(want[d], slack[d], q[(r * shape.heads + h) * hd + d]);
            }
            for (0..shape.kv_heads) |h| {
                hostHead(try widen(gpa, row[keys_at + h * hd ..][0..hd]), model.k_norm, model.inv_freq, pos, shape.eps, want[0..hd], slack[0..hd]);
                const slot = (pos * shape.kv_heads + h) * hd;
                for (0..hd) |d| try expectStored(want[d], slack[d], k[slot + d]);
                try testing.expectEqualSlices(u16, row[values_at + h * hd ..][0..hd], v[slot..][0..hd]);
            }
            for (0..shape.index_heads) |h| {
                hostHead(try widen(gpa, row[index_at + h * id ..][0..id]), model.index_q_norm, model.inv_freq, pos, shape.eps, want[0..id], slack[0..id]);
                for (0..id) |d| try expectStored(want[d], slack[d], iq[(r * shape.index_heads + h) * id + d]);
            }
            try testing.expectEqualSlices(u16, row[index_at + shape.index_heads * id ..][0..id], index_keys[pos * id ..][0..id]);
        }
    }
}

/// A head's codes and scale against the bf16 head they were taken from: the scale is the largest magnitude over
/// 448 and each code the e4m3 nearest value / largest * 448.
fn expectCodes(from: []const u16, codes: []const u8, scale: f32) !void {
    var top: f32 = 0;
    for (from) |bits| top = @max(top, @abs(numeric.f32FromBf16(bits)));
    try testing.expectEqual(top / 448.0, scale);
    for (from, codes) |bits, code| {
        const want: u8 = if (top > 0) numeric.e4m3FromF32(numeric.f32FromBf16(bits) / top * 448.0) else 0;
        try testing.expectEqual(want, code);
    }
}

/// A head's NVFP4 bytes and scale against the bf16 values they were taken from: the scale as FP8's, a 16's scale
/// code the e4m3 nearest (its largest magnitude) / (the head's) * 448, a value the e2m1 code of x over
/// (that code's value) * scale / 6.
fn expectNvfp4(from: []const u16, bytes: []const u8, scale: f32) !void {
    var top: f32 = 0;
    for (from) |bits| top = @max(top, @abs(numeric.f32FromBf16(bits)));
    try testing.expectEqual(top / 448.0, scale);
    for (0..from.len / 16) |block| {
        var most: f32 = 0;
        for (from[block * 16 ..][0..16]) |bits| most = @max(most, @abs(numeric.f32FromBf16(bits)));
        const scale_code: u8 = if (top > 0) numeric.e4m3FromF32(most / top * 448.0) else 0;
        try testing.expectEqual(scale_code, bytes[from.len / 2 + block]);
        const u = numeric.f32FromE4m3(scale_code) * top / 448.0 / 6.0;
        for (from[block * 16 ..][0..16], block * 16..) |bits, d| {
            const want: u8 = if (u > 0) e2m1FromF32(numeric.f32FromBf16(bits) / u) else 0;
            try testing.expectEqual(want, (bytes[d / 2] >> @intCast(d % 2 * 4)) & 15);
        }
    }
}

test "prep with FP8 caches stores the codes of what the bf16 caches hold, each head over its largest magnitude, and NVFP4 values each 16 over its own" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(38);
    const random = prng.random();
    const rows = 4;
    const capacity = 5008;
    for ([_]Shape{ fx.small, fx.wide }) |plain| for ([_]Storage{ .fp8, .fp8_nvfp4 }) |storage| {
        var shape = plain;
        shape.kv = storage;
        var model: Model = try .init(gpa, random, shape, rows);
        defer model.deinit();
        var plain_cache: Cache = try .contiguous(plain, capacity);
        defer plain_cache.deinit();
        var plain_scratch: Scratch = try .init(plain, rows, capacity);
        defer plain_scratch.deinit();
        var cache: Cache = try .contiguous(shape, capacity);
        defer cache.deinit();
        var scratch: Scratch = try .init(shape, rows, capacity);
        defer scratch.deinit();
        var position: Position = try .init();
        defer position.deinit();

        const hd = shape.head_dim;
        const width = shape.projection();
        for ([_]usize{ 0, 37, 5000 }) |pos0| {
            const pa = try randomBf16(gpa, random, rows * width, 1.5);
            // row 1 has a key and a value of zeros: a scale of zero, codes of zero
            @memset(pa[width + shape.heads * 2 * hd ..][0..hd], 0);
            @memset(pa[width + shape.heads * 2 * hd + shape.kv_heads * hd ..][0..hd], 0);
            // row 2's values span e4m3's range, down through its subnormals to what rounds to zero
            for (pa[2 * width + shape.heads * 2 * hd + shape.kv_heads * hd ..][0..hd], 0..) |*bits, d| {
                bits.* = numeric.bf16FromF32(std.math.ldexp(@as(f32, 1.0 + random.float(f32)), -@as(i32, @intCast(d % 22))));
            }
            // row 3's first 16 values lie halfway between e2m1's steps under a scale of a quarter (the head's
            // largest is 1.5 and the rest stay below it), where a tie goes to the even mantissa
            const ties = [16]f32{ 6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, -6 };
            for (pa[3 * width + shape.heads * 2 * hd + shape.kv_heads * hd ..][0..hd], 0..) |*bits, d| {
                bits.* = numeric.bf16FromF32(if (d < 16) ties[d] * 0.25 else random.float(f32) - 0.5);
            }
            try position.set(pos0);
            try plain_scratch.pa.upload(std.mem.sliceAsBytes(pa));
            try prep(plain, model.weights, model.rotary, plain_cache, plain_scratch, plain_scratch.pa.view, position.device.view, rows, pos0 + rows, null);
            try scratch.pa.upload(std.mem.sliceAsBytes(pa));
            try prep(shape, model.weights, model.rotary, cache, scratch, scratch.pa.view, position.device.view, rows, pos0 + rows, null);
            const q_from = try fetch(u16, gpa, plain_scratch.q);
            const k_from = try fetch(u16, gpa, plain_cache.k);
            const v_from = try fetch(u16, gpa, plain_cache.v);
            const q = try fetch(u8, gpa, scratch.q);
            const q_scale = try fetch(f32, gpa, scratch.q_scale.?);
            const k = try fetch(u8, gpa, cache.k);
            const k_scale = try fetch(f32, gpa, cache.k_scale.?);
            const v = try fetch(u8, gpa, cache.v);
            const v_scale = try fetch(f32, gpa, cache.v_scale.?);
            for (0..rows * shape.heads) |i| try expectCodes(q_from[i * hd ..][0..hd], q[i * hd ..][0..hd], q_scale[i]);
            for (pos0 * shape.kv_heads..(pos0 + rows) * shape.kv_heads) |i| {
                try expectCodes(k_from[i * hd ..][0..hd], k[i * hd ..][0..hd], k_scale[i]);
                if (storage == .fp8) {
                    try expectCodes(v_from[i * hd ..][0..hd], v[i * hd ..][0..hd], v_scale[i]);
                } else {
                    const wide = storage.valueWidth(hd);
                    try expectNvfp4(v_from[i * hd ..][0..hd], v[i * wide ..][0..wide], v_scale[i]);
                }
            }
            if (storage == .fp8_nvfp4) {
                const tied = v[(pos0 + 3) * shape.kv_heads * storage.valueWidth(hd) ..];
                try testing.expectEqualSlices(u8, &.{ 0x07, 0x22, 0x44, 0x66, 0xa0, 0xca, 0xec, 0xfe }, tied[0..8]);
                try testing.expectEqual(@as(u8, 0x7e), tied[hd / 2]);
            }
            try testing.expectEqual(@as(f32, 0), k_scale[(pos0 + 1) * shape.kv_heads]);
            try testing.expectEqual(@as(f32, 0), v_scale[(pos0 + 1) * shape.kv_heads]);
            // the indexer's rows are bf16 under either storage
            try testing.expectEqualSlices(u16, try fetch(u16, gpa, plain_scratch.iq), try fetch(u16, gpa, scratch.iq));
            try testing.expectEqualSlices(u16, try fetch(u16, gpa, plain_cache.index_keys), try fetch(u16, gpa, cache.index_keys));
        }
    };
}

test "pooling takes the blocks a window completes, and no other" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(32);
    const random = prng.random();
    const shape = fx.small;
    const capacity = 40;
    const id = shape.index_dim;
    var model: Model = try .init(gpa, random, shape, 1);
    defer model.deinit();
    var cache: Cache = try .contiguous(shape, capacity);
    defer cache.deinit();
    var position: Position = try .init();
    defer position.deinit();

    const keys = try randomBf16(gpa, random, capacity * id, 1.0);
    try cache.index_keys.upload(std.mem.sliceAsBytes(keys));
    const blocks = capacity / shape.ratio;
    const untouched = try gpa.alloc(u16, blocks * id);
    @memset(untouched, 0x7fc0);
    const mean = try gpa.alloc(f32, id);
    const want = try gpa.alloc(f32, id);
    const slack = try gpa.alloc(f32, id);
    // (first position, rows): nothing complete; block 0; a block begun before the window; two of three; whole blocks
    for ([_][2]usize{ .{ 0, 3 }, .{ 0, 4 }, .{ 5, 3 }, .{ 6, 9 }, .{ 8, 16 }, .{ 39, 1 } }) |window| {
        const pos0, const rows = window;
        try cache.pooled.upload(std.mem.sliceAsBytes(untouched));
        try position.set(pos0);
        try pool(shape, model.weights, model.rotary, cache, position.device.view, rows, pos0 + rows, null);
        const pooled = try fetch(u16, gpa, cache.pooled);
        for (0..blocks) |b| {
            const got = pooled[b * id ..][0..id];
            if (b < pos0 / shape.ratio or (b + 1) * shape.ratio > pos0 + rows) {
                try testing.expectEqualSlices(u16, untouched[0..id], got);
                continue;
            }
            for (mean, 0..) |*m, d| {
                var acc: f32 = 0;
                for (0..shape.ratio) |j| acc += numeric.f32FromBf16(keys[(b * shape.ratio + j) * id + d]);
                m.* = bf(acc / @as(f32, @floatFromInt(shape.ratio)));
            }
            hostHead(mean, model.index_k_norm, model.inv_freq, b * shape.ratio, shape.eps, want, slack);
            for (0..id) |d| try expectStored(want[d], slack[d], got[d]);
        }
    }
}

fn hostScore(shape: Shape, q: []const f32, k: []const f32, hk: usize, key: i32) f64 {
    const hd = shape.head_dim;
    const row = k[(@as(usize, @intCast(key)) * shape.kv_heads + hk) * hd ..][0..hd];
    var sum: f64 = 0;
    for (q, row) |a, b| sum += @as(f64, a) * b;
    return sum;
}

/// A query head against `keys` on the host in fp64: the softmax-weighted values, and the weighted magnitudes the
/// kernel's error scales with.
fn hostAttend(shape: Shape, q: []const f32, k: []const f32, v: []const f32, hk: usize, keys: []const i32, want: []f64, mag: []f64) void {
    const hd = shape.head_dim;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(hd)));
    var largest = -std.math.inf(f64);
    for (keys) |key| largest = @max(largest, hostScore(shape, q, k, hk, key) * scale);
    @memset(want, 0);
    @memset(mag, 0);
    var total: f64 = 0;
    for (keys) |key| {
        const p = @exp(hostScore(shape, q, k, hk, key) * scale - largest);
        total += p;
        const row = v[(@as(usize, @intCast(key)) * shape.kv_heads + hk) * hd ..][0..hd];
        for (want, mag, row) |*w, *m, value| {
            w.* += p * value;
            m.* += p * @abs(value);
        }
    }
    for (want, mag) |*w, *m| {
        w.* /= total;
        m.* /= total;
    }
}

/// The device's rows against the host's over each row's `lists` entry. The tolerance is 1/64 of the weighted
/// magnitudes: the weights are rounded to bf16 before they meet the values, and the result once more. With FP8
/// caches it is 1/14: the weights are rounded to e4m3, a 16th of each at most, and a weight far below the tile's
/// largest to e4m3's smallest step, which a floor allows for (it shows where the heaviest key's value is zero).
fn expectAttention(gpa: Allocator, shape: Shape, q: []const f32, k: []const f32, v: []const f32, lists: []const []const i32, got: []const u16) !void {
    const hd = shape.head_dim;
    const group = shape.heads / shape.kv_heads;
    const want = try gpa.alloc(f64, hd);
    defer gpa.free(want);
    const mag = try gpa.alloc(f64, hd);
    defer gpa.free(mag);
    const floor = try gpa.alloc(f64, hd);
    defer gpa.free(floor);
    for (lists, 0..) |keys, r| for (0..shape.heads) |h| {
        const at = (r * shape.heads + h) * hd;
        hostAttend(shape, q[at..][0..hd], k, v, h / group, keys, want, mag);
        // what the codes of small weights cannot resolve: half of e4m3's smallest step, 2^-10 of 448, of the
        // tile's largest weight times scale, on every key; a value's scale is within 5.4 of the largest
        @memset(floor, 0);
        if (shape.kv.scaled()) for (keys) |key| {
            for (floor, v[(@as(usize, @intCast(key)) * shape.kv_heads + h / group) * hd ..][0..hd]) |*f, value| f.* += @abs(value) * (5.4 / 1024.0 / 448.0);
        };
        for (0..hd) |d| try testing.expectApproxEqAbs(want[d], @as(f64, numeric.f32FromBf16(got[at + d])), mag[d] / @as(f64, if (shape.kv.scaled()) 14.0 else 64.0) + floor[d] + 1e-30);
    };
}

test "attention matches the host over dense and listed keys, and a row keeps its bits alone" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(34);
    const random = prng.random();
    const rows = 5;
    const capacity = 720;
    for ([_]Shape{ fx.small, fx.wide, fx.full }) |base| for ([_]Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        // the small shape's budget, so every shape reads up to two chunks
        var shape = base;
        shape.budget = fx.small.budget;
        shape.kv = storage;
        const hd = shape.head_dim;
        var cache: Cache = try .contiguous(shape, capacity);
        defer cache.deinit();
        var scratch: Scratch = try .init(shape, rows, capacity);
        defer scratch.deinit();
        var alone: Scratch = try .init(shape, 1, capacity);
        defer alone.deinit();
        var position: Position = try .init();
        defer position.deinit();
        // queries and keys are FP8 beside NVFP4 values
        const coded: Storage = if (storage == .bf16) .bf16 else .fp8;
        const k: Stored = try .random(gpa, random, coded, capacity, shape.kv_heads, hd, 1.0);
        const v: Stored = try .random(gpa, random, storage, capacity, shape.kv_heads, hd, 1.0);
        try k.upload(cache.k, cache.k_scale);
        try v.upload(cache.v, cache.v_scale);
        const q: Stored = try .random(gpa, random, coded, rows, shape.heads, hd, 1.5);
        try q.upload(scratch.q, scratch.q_scale);
        const all = try gpa.alloc(i32, capacity);
        for (all, 0..) |*key, i| key.* = @intCast(i);

        // dense rows of 1 to 5 keys, around a tile's end, around a chunk's end, and into a second chunk
        for ([_]usize{ 0, 61, 509, 695 }) |pos0| {
            try position.set(pos0);
            try attend(shape, cache, scratch, null, scratch.q.view, position.device.view, scratch.attn.view, pos0 + rows);
            const got = try fetch(u16, gpa, scratch.attn);
            var lists: [rows][]const i32 = undefined;
            for (&lists, 0..) |*list, r| list.* = all[0 .. pos0 + r + 1];
            try expectAttention(gpa, shape, q.values, k.values, v.values, &lists, got);
            for (0..rows) |r| {
                try q.slice(hd, r * shape.heads, shape.heads).upload(alone.q, alone.q_scale);
                try position.set(pos0 + r);
                try attend(shape, cache, alone, null, alone.q.view, position.device.view, alone.attn.view, pos0 + r + 1);
                const one = try fetch(u16, gpa, alone.attn);
                try testing.expectEqualSlices(u16, one, got[r * shape.heads * hd ..][0 .. shape.heads * hd]);
            }
        }

        // listed rows: the longest list, a dense row among them, a whole tile, a key past a chunk, a single key
        const pos0 = 600;
        const width = shape.listWidth();
        const counts = [rows]usize{ width - 1, pos0 + 2, 64, 513, 1 };
        const ids = try gpa.alloc(i32, rows * width);
        @memset(ids, -1);
        var nk: [rows]i32 = undefined;
        var sparse: [rows]i32 = undefined;
        var lists: [rows][]const i32 = undefined;
        for (0..rows) |r| {
            nk[r] = @intCast(counts[r]);
            sparse[r] = @intFromBool(r != 1);
            const list = ids[r * width ..][0..@min(counts[r], width)];
            for (list) |*key| key.* = random.intRangeLessThan(i32, 0, capacity);
            lists[r] = if (r == 1) all[0..counts[r]] else list;
        }
        const selection = scratch.selection.?;
        try selection.ids.upload(std.mem.sliceAsBytes(ids));
        try selection.nk.upload(std.mem.sliceAsBytes(&nk));
        try selection.sparse.upload(std.mem.sliceAsBytes(&sparse));
        try position.set(pos0);
        try attend(shape, cache, scratch, selection, scratch.q.view, position.device.view, scratch.attn.view, pos0 + rows);
        const got = try fetch(u16, gpa, scratch.attn);
        try expectAttention(gpa, shape, q.values, k.values, v.values, &lists, got);
        // row 3's list alone, in a scratch of one row
        const one_list = alone.selection.?;
        try one_list.ids.upload(std.mem.sliceAsBytes(ids[3 * width ..][0..width]));
        try one_list.nk.upload(std.mem.asBytes(&nk[3]));
        try one_list.sparse.upload(std.mem.asBytes(&sparse[3]));
        try q.slice(hd, 3 * shape.heads, shape.heads).upload(alone.q, alone.q_scale);
        try position.set(pos0 + 3);
        try attend(shape, cache, alone, one_list, alone.q.view, position.device.view, alone.attn.view, pos0 + 4);
        const one = try fetch(u16, gpa, alone.attn);
        try testing.expectEqualSlices(u16, one, got[3 * shape.heads * hd ..][0 .. shape.heads * hd]);

        // a row that lists one key gets that key's values: its weight is one, so nothing but the stored value's
        // own rounding is in the way (with NVFP4, every scale code's and magnitude's e4m3 code)
        for (0..40) |_| {
            for (0..rows) |r| {
                nk[r] = 1;
                sparse[r] = 1;
                ids[r * width] = random.intRangeLessThan(i32, 0, capacity);
            }
            try selection.ids.upload(std.mem.sliceAsBytes(ids));
            try selection.nk.upload(std.mem.sliceAsBytes(&nk));
            try selection.sparse.upload(std.mem.sliceAsBytes(&sparse));
            try attend(shape, cache, scratch, selection, scratch.q.view, position.device.view, scratch.attn.view, pos0 + rows);
            const single = try fetch(u16, gpa, scratch.attn);
            const group = shape.heads / shape.kv_heads;
            for (0..rows) |r| for (0..shape.heads) |h| {
                const want = v.values[(@as(usize, @intCast(ids[r * width])) * shape.kv_heads + h / group) * hd ..][0..hd];
                for (want, single[(r * shape.heads + h) * hd ..][0..hd]) |value, bits| try testing.expectApproxEqRel(value, numeric.f32FromBf16(bits), 1.0 / 100.0);
            };
        }
    };
}

test "a window's rows a block (attention_window.cu) give the chunks' and their merge's bits, dense and listed, every storage" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(105);
    const random = prng.random();
    const rows = 8;
    const capacity = 2600;
    const default_rows = attention.window_rows;
    defer attention.window_rows = default_rows;
    // the model's heads and budget: 12 query heads a KV head, lists of 2,052 keys, a fifth chunk of 4
    var model_like = fx.full;
    model_like.heads = 24;
    model_like.budget = 2048;
    for ([_]Shape{ fx.small, fx.wide, model_like }) |base| for ([_]Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        var shape = base;
        shape.kv = storage;
        const hd = shape.head_dim;
        var cache: Cache = try .contiguous(shape, capacity);
        defer cache.deinit();
        var scratch: Scratch = try .init(shape, rows, capacity);
        defer scratch.deinit();
        var position: Position = try .init();
        defer position.deinit();
        const coded: Storage = if (storage == .bf16) .bf16 else .fp8;
        const k: Stored = try .random(gpa, random, coded, capacity, shape.kv_heads, hd, 1.0);
        const v: Stored = try .random(gpa, random, storage, capacity, shape.kv_heads, hd, 1.0);
        try k.upload(cache.k, cache.k_scale);
        try v.upload(cache.v, cache.v_scale);
        // queries wide enough that a later chunk's largest score passes an earlier one's
        const q: Stored = try .random(gpa, random, coded, rows, shape.heads, hd, 4.0);
        try q.upload(scratch.q, scratch.q_scale);

        // dense rows from the first position, around a chunk's end, and over four chunks; under a selection a
        // dense row holds fewer keys than the lists' width
        for ([_]usize{ 0, 507, 1530, 2040 }) |pos0| {
            try position.set(pos0);
            const selection: ?attention.Selection = if (shape.selects(pos0 + rows)) scratch.selection.? else null;
            if (selection != null and pos0 + rows > shape.listWidth() - 1) continue;
            if (selection) |s| {
                // every row dense under a selection
                const none: [rows]i32 = @splat(0);
                var nk: [rows]i32 = undefined;
                for (&nk, 0..) |*n, r| n.* = @intCast(pos0 + r + 1);
                try s.nk.upload(std.mem.sliceAsBytes(&nk));
                try s.sparse.upload(std.mem.sliceAsBytes(&none));
            }
            const got = try bothWays(gpa, shape, cache, scratch, selection, position.device.view, pos0 + rows);
            try testing.expectEqualSlices(u16, got[0], got[1]);
        }
        if (!shape.selects(capacity)) continue;
        // listed rows: the longest list, chunks exactly whole, a tile, a single key, a dense row among them
        // the dense row as long as a dense row under a selection is: its list's width less one
        const pos0 = shape.listWidth() - 1 - rows;
        const width = shape.listWidth();
        const counts = [rows]usize{ width - 1, 512, 1024, 2048, 64, 1, 0, 777 };
        const ids = try gpa.alloc(i32, rows * width);
        @memset(ids, -1);
        var nk: [rows]i32 = undefined;
        var sparse: [rows]i32 = undefined;
        for (0..rows) |r| {
            const count = @min(counts[r], width - 1);
            nk[r] = @intCast(if (r == 6) pos0 + r + 1 else count);
            sparse[r] = @intFromBool(r != 6);
            for (ids[r * width ..][0..count]) |*key| key.* = random.intRangeLessThan(i32, 0, capacity);
        }
        const selection = scratch.selection.?;
        try selection.ids.upload(std.mem.sliceAsBytes(ids));
        try selection.nk.upload(std.mem.sliceAsBytes(&nk));
        try selection.sparse.upload(std.mem.sliceAsBytes(&sparse));
        try position.set(pos0);
        const got = try bothWays(gpa, shape, cache, scratch, selection, position.device.view, pos0 + rows);
        try testing.expectEqualSlices(u16, got[0], got[1]);
    };
}

/// `attend` by attention.cu's chunks and merge, then by attention_window.cu.
fn bothWays(gpa: Allocator, shape: Shape, cache: Cache, scratch: Scratch, selection: ?attention.Selection, pos: cuda.View, context: usize) ![2][]u16 {
    var out: [2][]u16 = undefined;
    for ([_]usize{ 0, 1 }, &out) |window, *got| {
        attention.window_rows = window;
        try attend(shape, cache, scratch, selection, scratch.q.view, pos, scratch.attn.view, context);
        got.* = try fetch(u16, gpa, scratch.attn);
    }
    return out;
}

/// The score of position `p` under a query head of factor 1, for the test below: the largest at position 0, the
/// first tile within 5 of it, and the second tile (64 .. 127) 85 and more under it, position 100 a further 30.
fn farScore(random: std.Random, p: usize) f32 {
    if (p == 0) return 45.0;
    if (p == 100) return -70.0;
    if (p >= 64 and p < 128) return -40.0 - 0.05 * @as(f32, @floatFromInt(p - 64));
    return 40.0 + 5.0 * random.float(f32);
}

test "a tile far under the row's largest score adds nothing, and no NaN: FP8 and NVFP4 values, dense and listed rows" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(43);
    const random = prng.random();
    const rows = 5;
    const capacity = 720;
    for ([_]Shape{ fx.small, fx.full }) |base| for ([_]Storage{ .fp8, .fp8_nvfp4 }) |storage| {
        var shape = base;
        shape.budget = fx.small.budget;
        shape.kv = storage;
        const hd = shape.head_dim;
        const root = @sqrt(@as(f32, @floatFromInt(hd)));
        var cache: Cache = try .contiguous(shape, capacity);
        defer cache.deinit();
        var scratch: Scratch = try .init(shape, rows, capacity);
        defer scratch.deinit();
        var alone: Scratch = try .init(shape, 1, capacity);
        defer alone.deinit();
        var position: Position = try .init();
        defer position.deinit();

        // Scores made by hand. Every value of a query is its head's factor (0.95, 1 or 1.05: the largest code
        // under a scale of factor / 448) and every value of a key S / sqrt(head_dim), so a key's score is the
        // factor times its S. Under the row's largest score, 45 times the factor, the second tile's best key is
        // 80.75 to 89.25 lower: exp of that times the tile's largest value scale (0.03 at most) is 2.6e-37 at
        // most, and 448 over it is not an fp32. That tile's weights are then the largest code where they are
        // not zero, under a unit of e-39 or less, and where they are zero they were zero times infinity:
        // position 100's (30 lower still, its weight underflows), position 101's (its values are zeros, so
        // its scale is zero) and those of a key past the row's list.
        const k: Stored = try .random(gpa, random, .fp8, capacity, shape.kv_heads, hd, 1.0);
        for (0..capacity) |pos| {
            const score = farScore(random, pos);
            for (0..shape.kv_heads) |hk| {
                const h = pos * shape.kv_heads + hk;
                k.scales[h] = @abs(score) / (448.0 * root);
                @memset(k.bytes[h * hd ..][0..hd], numeric.e4m3FromF32(if (score < 0) -448.0 else 448.0));
                @memset(k.values[h * hd ..][0..hd], score / root);
            }
        }
        const q: Stored = try .random(gpa, random, .fp8, rows, shape.heads, hd, 1.0);
        for (0..rows * shape.heads) |h| {
            const factor = 0.95 + 0.05 * @as(f32, @floatFromInt(h % shape.heads % 3));
            q.scales[h] = factor / 448.0;
            @memset(q.bytes[h * hd ..][0..hd], numeric.e4m3FromF32(448.0));
            @memset(q.values[h * hd ..][0..hd], factor);
        }
        // any values, under scales of 0.01 to 0.03, and zeros at position 101
        const v: Stored = try .random(gpa, random, storage, capacity, shape.kv_heads, hd, 1.0);
        for (v.scales, 0..) |*scale, h| {
            const to: f32 = if (h / shape.kv_heads == 101) 0.0 else 0.01 + 0.02 * random.float(f32);
            for (v.values[h * hd ..][0..hd]) |*value| value.* = value.* / scale.* * to;
            scale.* = to;
        }
        try k.upload(cache.k, cache.k_scale);
        try v.upload(cache.v, cache.v_scale);
        try q.upload(scratch.q, scratch.q_scale);
        try q.slice(hd, 0, shape.heads).upload(alone.q, alone.q_scale);

        // Listed rows, every one with the first tile's keys first:
        //   0: those alone, what the others must equal bit for bit;
        //   1: then a whole tile of far keys whose weights and scales are not zero (finite before spec 043);
        //   2: then position 100 alone, and 63 places past the list;
        //   3: then the second tile as it lies, positions 100 and 101 in it;
        //   4: then positions 128 .. 189, what a dense row of 190 keys must equal.
        const width = shape.listWidth();
        const ids = try gpa.alloc(i32, rows * width);
        @memset(ids, -1);
        for (0..rows) |r| for (ids[r * width ..][0..64], 0..) |*key, i| {
            key.* = @intCast(i);
        };
        for (ids[1 * width + 64 ..][0..64], 0..) |*key, i| key.* = @intCast(64 + (if (i == 36 or i == 37) i - 36 else i));
        ids[2 * width + 64] = 100;
        for (ids[3 * width + 64 ..][0..64], 0..) |*key, i| key.* = @intCast(64 + i);
        for (ids[4 * width + 64 ..][0..62], 0..) |*key, i| key.* = @intCast(128 + i);
        const nk = [rows]i32{ 64, 128, 65, 128, 126 };
        const sparse = [rows]i32{ 1, 1, 1, 1, 1 };
        const selection = scratch.selection.?;
        try selection.ids.upload(std.mem.sliceAsBytes(ids));
        try selection.nk.upload(std.mem.sliceAsBytes(&nk));
        try selection.sparse.upload(std.mem.sliceAsBytes(&sparse));
        const pos0 = 600;
        try position.set(pos0);
        try attend(shape, cache, scratch, selection, scratch.q.view, position.device.view, scratch.attn.view, pos0 + rows);
        const got = try fetch(u16, gpa, scratch.attn);
        const each = shape.heads * hd;
        for (got) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
        // the first tile alone is what the host makes of it
        try expectAttention(gpa, shape, q.values[0..each], k.values, v.values, &.{ids[0..64]}, got[0..each]);
        for (1..4) |r| try testing.expectEqualSlices(u16, got[0..each], got[r * each ..][0..each]);

        // dense rows: 70 keys (six of the far tile, then places past the row), 128 (the far tile whole), 190
        for ([_]usize{ 70, 128, 190 }) |n| {
            try position.set(n - 1);
            try attend(shape, cache, alone, null, alone.q.view, position.device.view, alone.attn.view, n);
            const one = try fetch(u16, gpa, alone.attn);
            for (one) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
            try testing.expectEqualSlices(u16, got[(if (n == 190) @as(usize, 4) else 0) * each ..][0..each], one);
        }
        _ = arena_state.reset(.retain_capacity);
    };
}

test "the gate scales each head's output by the sigmoid of its gate" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(35);
    const random = prng.random();
    const shape = fx.small;
    const rows = 3;
    const hd = shape.head_dim;
    var scratch: Scratch = try .init(shape, rows, 16);
    defer scratch.deinit();
    const pa = try randomBf16(gpa, random, rows * shape.projection(), 2.0);
    defer gpa.free(pa);
    const o = try randomBf16(gpa, random, rows * shape.out(), 1.0);
    defer gpa.free(o);
    try scratch.pa.upload(std.mem.sliceAsBytes(pa));
    try scratch.attn.upload(std.mem.sliceAsBytes(o));
    try gate(scratch.attn.view, scratch.pa.view, scratch.gated.view);
    const got = try fetch(u16, gpa, scratch.gated);
    defer gpa.free(got);
    for (0..rows) |r| for (0..shape.heads) |h| for (0..hd) |d| {
        const value = numeric.f32FromBf16(o[(r * shape.heads + h) * hd + d]);
        const g = numeric.f32FromBf16(pa[r * shape.projection() + h * 2 * hd + hd + d]);
        try expectStored(value / (1.0 + @exp(-g)), @abs(value), got[(r * shape.heads + h) * hd + d]);
    };
}

/// Decodes `total` positions of `inputs` [total, dims] through the block in windows of `sizes` (rows, rows kept),
/// taken in turn. A window's rows past the kept ones carry `noise` instead, as rejected drafts would, and the next
/// window starts after the kept rows.
fn runWindows(gpa: Allocator, shape: Shape, weights: Weights, rotary: Rotary, inputs: []const u16, noise: []const u16, total: usize, capacity: usize, bound: ?usize, sizes: []const [2]usize) !Run {
    return runWindowsOver(gpa, shape, weights, rotary, inputs, noise, total, capacity, bound, sizes, null);
}

test "windows with dropped rows equal serial steps bit for bit, dense, sparse and across the transition" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(36);
    const random = prng.random();
    const Case = struct { shape: Shape, total: usize };
    // windows of 5 rows and more a row a block (attention_window.cu, spec 105 WS-05), steps by chunks: the same bits
    const Way = struct { window_rows: usize };
    const default_rows = attention.window_rows;
    defer attention.window_rows = default_rows;
    for ([_]Way{ .{ .window_rows = 0 }, .{ .window_rows = 5 } }) |way| for ([_]Case{ .{ .shape = fx.small, .total = 720 }, .{ .shape = fx.wide, .total = 190 } }) |case| for ([_]Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        attention.window_rows = way.window_rows;
        var shape = case.shape;
        shape.kv = storage;
        const total = case.total;
        var model: Model = try .init(gpa, random, shape, 16);
        defer model.deinit();
        const inputs = try randomBf16(gpa, random, (total + 16) * shape.dims, 1.0);
        const noise = try randomBf16(gpa, random, (total + 16) * shape.dims, 1.0);

        // serial steps in caches that hold exactly the sequence
        const serial = try runWindows(gpa, shape, model.weights, model.rotary, inputs, noise, total, total, null, &.{.{ 1, 1 }});
        try testing.expectEqual(total - (shape.budget + shape.ratio - 1), serial.sparse_rows);
        // the selection met equal scores at its cut: blocks no indexer head likes all score zero
        try testing.expect(serial.tied_rows > 0);
        for (serial.branch) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));

        // windows of every kind in larger caches: whole, with rows dropped, one row, sixteen rows
        const mixed = try runWindows(gpa, shape, model.weights, model.rotary, inputs, noise, total, total + 300, null, &.{ .{ 5, 5 }, .{ 7, 3 }, .{ 1, 1 }, .{ 16, 9 }, .{ 4, 1 }, .{ 3, 3 }, .{ 16, 16 } });
        try serial.expectEqual(mixed, shape, total);
        try testing.expectEqual(serial.sparse_rows, mixed.sparse_rows);

        // a launch bounded by more context than the sequence has, as a captured graph's is: the same bits
        const bounded = try runWindows(gpa, shape, model.weights, model.rotary, inputs, noise, total, total + 300, total + 300, &.{ .{ 1, 1 }, .{ 7, 3 }, .{ 16, 16 } });
        try serial.expectEqual(bounded, shape, total);
        try testing.expectEqual(serial.sparse_rows, bounded.sparse_rows);

        // caches no larger than the budget have no selection scratch: the dense rows' bits are the same
        const dense = try runWindows(gpa, shape, model.weights, model.rotary, inputs, noise, shape.budget, shape.budget, null, &.{ .{ 16, 16 }, .{ 6, 2 } });
        try testing.expectEqual(@as(usize, 0), dense.sparse_rows);
        try serial.expectEqual(dense, shape, shape.budget);
    };
}

test "a checkpoint's first attention layer: finite results, and windows cut differently agree past the budget" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory; skipped without.
    const path = std.mem.span(std.c.getenv("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest);
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const c, const resolved = try opened.open(testing.allocator, gpa, path);
    defer opened.close();
    const shape = try Shape.of(&c);
    const tensors = for (resolved.layers) |maybe| {
        const layer = maybe orelse continue;
        if (layer.block == .full_attention) break layer.block.full_attention;
    } else return error.SkipZigTest;
    var scratch: cuda.Scratch = .init(gpa);
    defer scratch.deinit();
    var weights: Weights = try .load(gpa, shape, tensors, resolved.norms_centred.?, scratch.rows(16));
    defer weights.deinit();
    var rotary: Rotary = try .init(shape);
    defer rotary.deinit();

    var prng: std.Random.DefaultPrng = .init(37);
    const random = prng.random();
    const total = shape.budget + 80;
    const inputs = try randomBf16(gpa, random, (total + 16) * shape.dims, 1.0);
    const noise = try randomBf16(gpa, random, (total + 16) * shape.dims, 1.0);
    const whole = try runWindows(gpa, shape, weights, rotary, inputs, noise, total, total + 16, null, &.{.{ 16, 16 }});
    const cut = try runWindows(gpa, shape, weights, rotary, inputs, noise, total, total + 4096, null, &.{ .{ 7, 3 }, .{ 16, 9 }, .{ 1, 1 }, .{ 5, 5 }, .{ 1, 1 } });
    try whole.expectEqual(cut, shape, total);
    try testing.expectEqual(total - (shape.budget + shape.ratio - 1), whole.sparse_rows);
    try testing.expectEqual(whole.sparse_rows, cut.sparse_rows);
    var largest: f32 = 0;
    for (whole.branch) |bits| {
        const value = numeric.f32FromBf16(bits);
        try testing.expect(std.math.isFinite(value));
        largest = @max(largest, @abs(value));
    }
    try testing.expect(largest > 0);
    // with random inputs the real indexer leaves blocks no head likes, so the cut falls among equal (zero) scores
    try testing.expect(whole.tied_rows > 0);
}
