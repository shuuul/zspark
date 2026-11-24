//! `attention.zig`'s rotary positions for an image prompt (spec 019, `Rope`): the prep and the pooled key rotate
//! each pair by the axis TensorFold 0.6.5's `rope_axis` (MODE 2) picks, against the host; a row's bits are the
//! same alone and among others under a table; a table of zeros, or one the rows are past with delta 0, gives the
//! text path's bits.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const fx = fixtures.attending.fx;
const Model = fixtures.attending.Model;
const fetch = fixtures.attending.fetch;
const expectStored = fixtures.attending.expectStored;
const widen = fixtures.attending.widen;
const randomBf16 = fixtures.attending.randomBf16;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const bf = numeric.roundBf16;
const Cache = attention.Cache;
const Position = attention.Position;
const Rope = attention.Rope;
const Scratch = attention.Scratch;
const Shape = attention.Shape;
const pool = attention.pool;
const prep = attention.prep;

/// The small shape with sections that end inside its 8 pairs: pairs 1 and 4 take h, 2 takes w, the rest t.
fn sectioned() Shape {
    var shape = fx.small;
    shape.sections = .{ 2, 2, 1 };
    return shape;
}

/// TensorFold's `rope_axis` on the host: the position pair `i` of the row at `pos` rotates by.
fn axis(positions: []const [3]i32, delta: i32, pos: usize, i: usize, sections: [3]u32) i64 {
    if (pos >= positions.len) return @as(i64, @intCast(pos)) + delta;
    const at = positions[pos];
    if (i % 3 == 1 and i < 3 * sections[1]) return at[1];
    if (i % 3 == 2 and i < 3 * sections[2]) return at[2];
    return at[0];
}

/// `attention_test.zig`'s host head, each pair rotated at its own axis.
fn hostHead(x: []const f32, scale: []const f32, inv_freq: []const f32, axes: []const i64, eps: f32, want: []f32, slack: []f32) void {
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
        const pair = if (d < half) d else d - half;
        const angle = @as(f32, @floatFromInt(axes[pair])) * inv_freq[pair];
        out_value.* = if (d < half) xn * @cos(angle) - xpn * @sin(angle) else xpn * @sin(angle) + xn * @cos(angle);
        mag.* = @abs(xn) + @abs(xpn);
    }
}

/// A table as an image prompt makes one: text, a grid of `h` x `w` at one t, text again; `delta` continues it.
fn imageTable(gpa: Allocator, before: usize, h: usize, w: usize, after: usize) !struct { positions: [][3]i32, delta: i32 } {
    const positions = try gpa.alloc([3]i32, before + h * w + after);
    for (positions[0..before], 0..) |*p, i| p.* = @splat(@intCast(i));
    const t: i32 = @intCast(before);
    for (0..h) |y| for (0..w) |x| {
        positions[before + y * w + x] = .{ t, t + @as(i32, @intCast(y)), t + @as(i32, @intCast(x)) };
    };
    const next: i32 = t + @as(i32, @intCast(@max(h, w)));
    for (positions[before + h * w ..], 0..) |*p, i| p.* = @splat(next + @as(i32, @intCast(i)));
    return .{ .positions = positions, .delta = next - @as(i32, @intCast(before + h * w)) };
}

test "under an image's rotary positions prep rotates each pair by its axis as the host does, a row's bits the same alone" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(61);
    const random = prng.random();
    const shape = sectioned();
    const rows = 8;
    const capacity = 64;
    var model: Model = try .init(gpa, random, shape, rows);
    defer model.deinit();
    var cache: Cache = try .contiguous(shape, capacity);
    defer cache.deinit();
    var scratch: Scratch = try .init(shape, rows, capacity);
    defer scratch.deinit();
    var position: Position = try .init();
    defer position.deinit();
    var rope: Rope = try .init(capacity);
    defer rope.deinit();
    const table = try imageTable(gpa, 3, 5, 6, 4);
    try rope.set(table.positions, table.delta);
    try testing.expectEqual(@as(usize, 37), table.positions.len);

    const hd = shape.head_dim;
    const id = shape.index_dim;
    const half = shape.rotary_dim / 2;
    const width = shape.projection();
    const keys_at = shape.heads * 2 * hd;
    const values_at = keys_at + shape.kv_heads * hd;
    const index_at = values_at + shape.kv_heads * hd;
    const want = try gpa.alloc(f32, @max(hd, id));
    const slack = try gpa.alloc(f32, @max(hd, id));
    const axes = try gpa.alloc(i64, half);
    // inside the table, across its end, past it
    for ([_]usize{ 0, 33, 50 }) |pos0| {
        const pa = try randomBf16(gpa, random, rows * width, 1.5);
        try scratch.pa.upload(std.mem.sliceAsBytes(pa));
        try position.set(pos0);
        try prep(shape, model.weights, model.rotary, cache, scratch, scratch.pa.view, position.device.view, rows, pos0 + rows, rope.view());
        const q = try fetch(u16, gpa, scratch.q);
        const iq = try fetch(u16, gpa, scratch.iq);
        const k = try fetch(u16, gpa, cache.k);
        for (0..rows) |r| {
            const row = pa[r * width ..][0..width];
            const pos = pos0 + r;
            for (axes, 0..) |*a, i| a.* = axis(table.positions, table.delta, pos, i, shape.sections);
            for (0..shape.heads) |h| {
                hostHead(try widen(gpa, row[h * 2 * hd ..][0..hd]), model.q_norm, model.inv_freq, axes, shape.eps, want[0..hd], slack[0..hd]);
                for (0..hd) |d| try expectStored(want[d], slack[d], q[(r * shape.heads + h) * hd + d]);
            }
            for (0..shape.kv_heads) |h| {
                hostHead(try widen(gpa, row[keys_at + h * hd ..][0..hd]), model.k_norm, model.inv_freq, axes, shape.eps, want[0..hd], slack[0..hd]);
                for (0..hd) |d| try expectStored(want[d], slack[d], k[(pos * shape.kv_heads + h) * hd + d]);
            }
            for (0..shape.index_heads) |h| {
                hostHead(try widen(gpa, row[index_at + h * id ..][0..id]), model.index_q_norm, model.inv_freq, axes, shape.eps, want[0..id], slack[0..id]);
                for (0..id) |d| try expectStored(want[d], slack[d], iq[(r * shape.index_heads + h) * id + d]);
            }
        }
        // each row alone at its position: the same bits
        const q_row = shape.heads * hd;
        const iq_row = shape.index_heads * id;
        const k_row = shape.kv_heads * hd;
        for (0..rows) |r| {
            try scratch.pa.uploadAt(0, std.mem.sliceAsBytes(pa[r * width ..][0..width]));
            try position.set(pos0 + r);
            try prep(shape, model.weights, model.rotary, cache, scratch, scratch.pa.rows(1), position.device.view, 1, pos0 + r + 1, rope.view());
            const q_alone = try fetch(u16, gpa, scratch.q);
            const iq_alone = try fetch(u16, gpa, scratch.iq);
            const k_alone = try fetch(u16, gpa, cache.k);
            try testing.expectEqualSlices(u16, q[r * q_row ..][0..q_row], q_alone[0..q_row]);
            try testing.expectEqualSlices(u16, iq[r * iq_row ..][0..iq_row], iq_alone[0..iq_row]);
            try testing.expectEqualSlices(u16, k[(pos0 + r) * k_row ..][0..k_row], k_alone[(pos0 + r) * k_row ..][0..k_row]);
        }
    }
}

test "a text sequence's rotary words (zeros) give prep and the pool the bits of no table at all" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(62);
    const random = prng.random();
    const shape = sectioned();
    const rows = 9;
    const capacity = 64;
    var model: Model = try .init(gpa, random, shape, rows);
    defer model.deinit();
    var scratch: Scratch = try .init(shape, rows, capacity);
    defer scratch.deinit();
    var position: Position = try .init();
    defer position.deinit();
    var zeros: Rope = try .init(capacity);
    defer zeros.deinit();
    // a table set and then cleared is zeros again
    try zeros.set((try imageTable(gpa, 1, 2, 2, 1)).positions, 5);
    try zeros.clear();
    const width = shape.projection();
    var caches: [2]Cache = .{ try .contiguous(shape, capacity), try .contiguous(shape, capacity) };
    defer for (&caches) |*c| c.deinit();
    for ([_]usize{ 0, 27 }) |pos0| {
        const pa = try randomBf16(gpa, random, rows * width, 1.5);
        var got: [2][3][]u16 = undefined;
        for (&caches, 0..) |cache, which| {
            try scratch.pa.upload(std.mem.sliceAsBytes(pa));
            try position.set(pos0);
            const rope: ?cuda.View = if (which == 0) null else zeros.view();
            try prep(shape, model.weights, model.rotary, cache, scratch, scratch.pa.view, position.device.view, rows, pos0 + rows, rope);
            try pool(shape, model.weights, model.rotary, cache, position.device.view, rows, pos0 + rows, rope);
            got[which] = .{ try fetch(u16, gpa, scratch.q), try fetch(u16, gpa, cache.k), try fetch(u16, gpa, cache.pooled) };
        }
        for (got[0], got[1]) |a, b| try testing.expectEqualSlices(u16, a, b);
    }
}

test "under an image's rotary positions the pooled key rotates at its block's first position's axes, as the host does" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(63);
    const random = prng.random();
    const shape = sectioned();
    const capacity = 64;
    const id = shape.index_dim;
    const half = shape.rotary_dim / 2;
    var model: Model = try .init(gpa, random, shape, 1);
    defer model.deinit();
    var cache: Cache = try .contiguous(shape, capacity);
    defer cache.deinit();
    var position: Position = try .init();
    defer position.deinit();
    var rope: Rope = try .init(capacity);
    defer rope.deinit();
    const table = try imageTable(gpa, 2, 4, 7, 3);
    try rope.set(table.positions, table.delta);

    const keys = try randomBf16(gpa, random, capacity * id, 1.0);
    try cache.index_keys.upload(std.mem.sliceAsBytes(keys));
    const blocks = capacity / shape.ratio;
    const mean = try gpa.alloc(f32, id);
    const want = try gpa.alloc(f32, id);
    const slack = try gpa.alloc(f32, id);
    const axes = try gpa.alloc(i64, half);
    try position.set(0);
    try pool(shape, model.weights, model.rotary, cache, position.device.view, capacity, capacity, rope.view());
    const pooled = try fetch(u16, gpa, cache.pooled);
    var moved: usize = 0;
    for (0..blocks) |b| {
        for (mean, 0..) |*m, d| {
            var acc: f32 = 0;
            for (0..shape.ratio) |j| acc += numeric.f32FromBf16(keys[(b * shape.ratio + j) * id + d]);
            m.* = bf(acc / @as(f32, @floatFromInt(shape.ratio)));
        }
        const first = b * shape.ratio;
        for (axes, 0..) |*a, i| a.* = axis(table.positions, table.delta, first, i, shape.sections);
        for (axes) |a| moved += @intFromBool(a != first);
        hostHead(mean, model.index_k_norm, model.inv_freq, axes, shape.eps, want, slack);
        for (0..id) |d| try expectStored(want[d], slack[d], pooled[b * id + d]);
    }
    // the table moves some blocks' axes off their positions: the test sees the table
    try testing.expect(moved > 0);

    // a window of one row completing each block: the same bits
    for (0..blocks) |b| {
        try position.set(b * shape.ratio + shape.ratio - 1);
        try pool(shape, model.weights, model.rotary, cache, position.device.view, 1, (b + 1) * shape.ratio, rope.view());
    }
    try testing.expectEqualSlices(u16, pooled, try fetch(u16, gpa, cache.pooled));
}
