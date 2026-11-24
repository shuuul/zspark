//! `testing.zig`'s `attending` (spec 104 WS-06 moved it out of `testing.zig`, which names it so): an attention
//! block of random weights, its shapes, a run of windows over caches in order or behind a table, and the host's
//! readers, for `attention_test.zig`, `attention_selection_test.zig` and `attention_table_test.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const gpu = @import("../../cuda/testing.zig");
const attention = @import("../../../src/models/qwen4/blocks/attention.zig");
const Cache = attention.Cache;
const Position = attention.Position;
const Rotary = attention.Rotary;
const Scratch = attention.Scratch;
const Shape = attention.Shape;
const Storage = attention.Storage;
const Tabled = attention.Tabled;
const Weights = attention.Weights;
const chunk = attention.chunk;
const forward = attention.forward;

pub const fetch = gpu.fetch;

/// A buffer's bytes, whatever its elements.
pub fn fetchBytes(gpa: Allocator, buffer: cuda.Buffer) ![]u8 {
    try cuda.synchronize();
    const host = try gpa.alloc(u8, buffer.bytes());
    errdefer gpa.free(host);
    try buffer.download(host);
    return host;
}

/// A stored bf16 value against the host's: within two roundings of `slack`.
pub fn expectStored(want: f32, slack: f32, got_bits: u16) !void {
    try testing.expectApproxEqAbs(want, numeric.f32FromBf16(got_bits), slack / 64.0 + 1e-30);
}

pub fn widen(gpa: Allocator, bits: []const u16) ![]f32 {
    const wide = try gpa.alloc(f32, bits.len);
    for (wide, bits) |*o, b| o.* = numeric.f32FromBf16(b);
    return wide;
}

pub fn randomBf16(gpa: Allocator, random: std.Random, count: usize, spread: f32) ![]u16 {
    const bits = try gpa.alloc(u16, count);
    for (bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32) * spread);
    return bits;
}

fn randomScales(gpa: Allocator, random: std.Random, count: usize) ![]f32 {
    const scales = try gpa.alloc(f32, count);
    for (scales) |*s| s.* = 1.0 + 0.2 * random.floatNorm(f32);
    return scales;
}

/// Shapes small enough for the host, with lists that still cross the 64-key tiles and the 512-key chunks.
pub const fx = struct {
    /// 3 query heads a KV head (one padded lane head), selection from position 643 on.
    pub const small: Shape = .{ .dims = 128, .heads = 6, .kv_heads = 2, .head_dim = 32, .index_heads = 2, .index_dim = 64, .rotary_dim = 16, .rope_theta = 1e7, .budget = 640, .ratio = 4, .eps = 1e-6 };
    /// 12 query heads on one KV head, as the model's ratio; selection from position 99 on.
    /// The model's head: 256 values, the widest the kernels take.
    pub const full: Shape = .{ .dims = 64, .heads = 4, .kv_heads = 2, .head_dim = 256, .index_heads = 2, .index_dim = 64, .rotary_dim = 64, .rope_theta = 1e7, .budget = 640, .ratio = 4, .eps = 1e-6 };
    pub const wide: Shape = .{ .dims = 192, .heads = 12, .kv_heads = 1, .head_dim = 64, .index_heads = 4, .index_dim = 128, .rotary_dim = 32, .rope_theta = 1e7, .budget = 96, .ratio = 4, .eps = 1e-6 };
};

/// Random weights of a shape, with the norm scales kept for the host.
pub const Model = struct {
    weights: Weights,
    scratch: *cuda.Scratch,
    rotary: Rotary,
    q_norm: []f32,
    k_norm: []f32,
    index_q_norm: []f32,
    index_k_norm: []f32,
    inv_freq: []f32,

    pub fn init(arena: Allocator, random: std.Random, shape: Shape, max_rows: usize) !Model {
        const spread = 1.0 / @sqrt(@as(f32, @floatFromInt(shape.dims)));
        const hd = shape.head_dim;
        var self: Model = undefined;
        self.q_norm = try randomScales(arena, random, hd);
        self.k_norm = try randomScales(arena, random, hd);
        self.index_q_norm = try randomScales(arena, random, shape.index_dim);
        self.index_k_norm = try randomScales(arena, random, shape.index_dim);
        self.inv_freq = try arena.alloc(f32, shape.rotary_dim / 2);
        Rotary.frequencies(shape, self.inv_freq);
        self.scratch = try .create(testing.allocator);
        errdefer self.scratch.destroy();
        self.weights = try .init(shape, .{
            .q = std.mem.sliceAsBytes(try randomBf16(arena, random, shape.heads * 2 * hd * shape.dims, spread)),
            .k = std.mem.sliceAsBytes(try randomBf16(arena, random, shape.kv_heads * hd * shape.dims, spread)),
            .v = std.mem.sliceAsBytes(try randomBf16(arena, random, shape.kv_heads * hd * shape.dims, spread)),
            .index_qk = std.mem.sliceAsBytes(try randomBf16(arena, random, (shape.index_heads + 1) * shape.index_dim * shape.dims, spread)),
            .o = std.mem.sliceAsBytes(try randomBf16(arena, random, shape.dims * shape.out(), 0.1)),
            .q_norm = self.q_norm,
            .k_norm = self.k_norm,
            .index_q_norm = self.index_q_norm,
            .index_k_norm = self.index_k_norm,
        }, self.scratch.rows(max_rows));
        errdefer self.weights.deinit();
        self.rotary = try .init(shape);
        return self;
    }

    pub fn deinit(self: *Model) void {
        self.weights.deinit();
        self.scratch.destroy();
        self.rotary.deinit();
    }
};

const e2m1_magnitudes = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

/// The e2m1 code nearest x, a tie to the even mantissa, as `e2m1_code` of the kernel.
pub fn e2m1FromF32(x: f32) u8 {
    const a = @abs(x);
    if (a <= 0.25) return 0;
    const code: u8 = if (a < 0.75) 1 else if (a <= 1.25) 2 else if (a < 1.75) 3 else if (a <= 2.5) 4 else if (a < 3.5) 5 else if (a <= 5.0) 6 else 7;
    return if (x < 0) code | 8 else code;
}

/// The e4m3 code attend makes of an NVFP4 value: its e2m1 magnitude times its 16's scale over 6, rounded.
fn e4m3FromNvfp4(four_bits: u8, scale_code: u8) u8 {
    const magnitude = four_bits & 7;
    const code: u8 = if (magnitude == 0) 0 else numeric.e4m3FromF32(numeric.f32FromE4m3(scale_code) * e2m1_magnitudes[magnitude] / 6.0);
    return code | (four_bits & 8) << 4;
}

/// The host's list for a row ending at `end` from scores over its complete blocks; false for a dense row.
pub fn hostList(gpa: Allocator, scores: []const f32, end: usize, ratio: usize, top: usize, ids: *std.ArrayList(i32)) !bool {
    const complete = end / ratio;
    ids.clearRetainingCapacity();
    if (complete <= top) return false;
    const order = try gpa.alloc(u32, complete);
    defer gpa.free(order);
    for (order, 0..) |*o, b| o.* = @intCast(b);
    const byScore = struct {
        fn before(s: []const f32, a: u32, b: u32) bool {
            return s[a] > s[b] or (s[a] == s[b] and a < b);
        }
    }.before;
    std.mem.sort(u32, order, scores, byScore);
    std.mem.sort(u32, order[0..top], {}, std.sort.asc(u32));
    for (order[0..top]) |b| for (0..ratio) |j| try ids.append(gpa, @intCast(b * ratio + j));
    for (complete * ratio..end) |tail| try ids.append(gpa, @intCast(tail));
    return true;
}

/// Heads as a storage holds them, and the values they stand for.
pub const Stored = struct {
    /// bf16 bits or e4m3 codes, `width` values a head, or a head's NVFP4 bytes
    bytes: []u8,
    /// bytes a head
    each: usize,
    /// A head's scale unless the storage is bf16.
    scales: []f32,
    values: []f32,

    /// `rows` rows of `group` heads of about `spread` in magnitude. FP8 heads have scales that differ from head
    /// to head, and three times larger in the rows of a tile's second half: the largest of a tile is there.
    pub fn random(gpa: Allocator, r: std.Random, storage: Storage, rows: usize, group: usize, width: usize, spread: f32) !Stored {
        const heads = rows * group;
        const values = try gpa.alloc(f32, heads * width);
        switch (storage) {
            .bf16 => {
                const bits = try randomBf16(gpa, r, heads * width, spread);
                for (values, bits) |*value, b| value.* = numeric.f32FromBf16(b);
                return .{ .bytes = std.mem.sliceAsBytes(bits), .each = 2 * width, .scales = &.{}, .values = values };
            },
            .fp8 => {
                const codes = try gpa.alloc(u8, heads * width);
                const scales = try gpa.alloc(f32, heads);
                for (scales, 0..) |*scale, h| {
                    scale.* = spread * (2.5 + 2.0 * r.float(f32)) / 448.0;
                    if (h / group % 64 >= 32) scale.* *= 3.0;
                    for (codes[h * width ..][0..width], values[h * width ..][0..width]) |*code, *value| {
                        code.* = numeric.e4m3FromF32(std.math.clamp(r.floatNorm(f32) * 448.0 / 3.5, -448.0, 448.0));
                        value.* = numeric.f32FromE4m3(code.*) * scale.*;
                    }
                }
                return .{ .bytes = codes, .each = width, .scales = scales, .values = values };
            },
            .fp8_nvfp4 => {
                // any bytes: e2m1 codes, then the 16s' scale codes, then what pads a head, which nothing reads. Every
                // head's second 16 has small scales, down through e4m3's subnormals, the others larger ones, and one
                // in 16 is zero: values that small beside large ones under one key would fall below what the
                // softmax weight's code resolves, which no tolerance in the weighted magnitudes allows for.
                const each = storage.valueWidth(width);
                const bytes = try gpa.alloc(u8, heads * each);
                r.bytes(bytes);
                const scales = try gpa.alloc(f32, heads);
                for (scales, 0..) |*scale, h| {
                    scale.* = spread * (2.5 + 2.0 * r.float(f32)) / 448.0;
                    if (h / group % 64 >= 32) scale.* *= 3.0;
                    const head_bytes = bytes[h * each ..][0..each];
                    for (head_bytes[width / 2 ..][0 .. width / 16], 0..) |*code, block| code.* = if (r.uintLessThan(u8, 16) == 0) 0 else if (block == 1) 1 + r.uintLessThan(u8, 0x2f) else 0x30 + r.uintLessThan(u8, 0x4f);
                    for (head_bytes[0 .. width / 2]) |*two| {
                        if (two.* & 0x0f == 8) two.* &= 0xf0;
                        if (two.* & 0xf0 == 0x80) two.* &= 0x0f;
                    }
                    for (values[h * width ..][0..width], 0..) |*value, d| {
                        const four_bits = (head_bytes[d / 2] >> @intCast(d % 2 * 4)) & 15;
                        value.* = numeric.f32FromE4m3(e4m3FromNvfp4(four_bits, head_bytes[width / 2 + d / 16])) * scale.*;
                    }
                }
                return .{ .bytes = bytes, .each = each, .scales = scales, .values = values };
            },
        }
    }

    /// Heads [from, from + count) of `width` values.
    pub fn slice(self: Stored, width: usize, from: usize, count: usize) Stored {
        return .{
            .bytes = self.bytes[from * self.each ..][0 .. count * self.each],
            .each = self.each,
            .scales = if (self.scales.len == 0) self.scales else self.scales[from..][0..count],
            .values = self.values[from * width ..][0 .. count * width],
        };
    }

    pub fn upload(self: Stored, buffer: cuda.Buffer, scale: ?cuda.Buffer) !void {
        try buffer.upload(self.bytes);
        if (scale) |to| try to.upload(std.mem.sliceAsBytes(self.scales));
    }
};

/// What a run of windows leaves: each committed row's results and the caches' committed rows.
pub const Run = struct {
    gated: []u16,
    branch: []u16,
    /// The caches' rows as bytes, and their scales (empty for bf16).
    k: []u8 = &.{},
    v: []u8 = &.{},
    k_scale: []f32 = &.{},
    v_scale: []f32 = &.{},
    index_keys: []u16 = &.{},
    pooled: []u16 = &.{},
    /// Committed rows that read a selected list, and those whose selection had equal scores at the cut.
    sparse_rows: usize = 0,
    tied_rows: usize = 0,

    pub fn expectEqual(want: Run, got: Run, shape: Shape, rows: usize) !void {
        const blocks = rows / shape.ratio;
        const kv = rows * shape.kv_heads * shape.head_dim * shape.kv.dtype().size();
        const values = rows * shape.kv_heads * shape.kv.valueWidth(shape.head_dim) * shape.kv.valueDtype().size();
        const scales = if (shape.kv.scaled()) rows * shape.kv_heads else 0;
        try testing.expectEqualSlices(f32, want.k_scale[0..scales], got.k_scale[0..scales]);
        try testing.expectEqualSlices(f32, want.v_scale[0..scales], got.v_scale[0..scales]);
        try testing.expectEqualSlices(u16, want.gated[0 .. rows * shape.out()], got.gated[0 .. rows * shape.out()]);
        try testing.expectEqualSlices(u16, want.branch[0 .. rows * shape.dims], got.branch[0 .. rows * shape.dims]);
        try testing.expectEqualSlices(u8, want.k[0..kv], got.k[0..kv]);
        try testing.expectEqualSlices(u8, want.v[0..values], got.v[0..values]);
        try testing.expectEqualSlices(u16, want.index_keys[0 .. rows * shape.index_dim], got.index_keys[0 .. rows * shape.index_dim]);
        try testing.expectEqualSlices(u16, want.pooled[0 .. blocks * shape.index_dim], got.pooled[0 .. blocks * shape.index_dim]);
    }
};

/// A cache's array as the sequence's rows in order: itself, or its rows gathered through the table.
fn inOrder(gpa: Allocator, cache: Cache, buffer: cuda.Buffer) ![]align(4) u8 {
    const stored = try fetchBytes(gpa, buffer);
    const each = stored.len / @as(usize, @intCast(buffer.view.size[0]));
    const out = try gpa.alignedAlloc(u8, .@"4", cache.capacity * each);
    const table = try gpa.alloc(i32, if (cache.table) |view| @intCast(view.size[0]) else 0);
    if (cache.table) |view| try cuda.download(view, std.mem.sliceAsBytes(table));
    for (0..cache.capacity) |pos| {
        const at = if (cache.table == null) pos else @as(usize, @intCast(table[pos / chunk])) * chunk + pos % chunk;
        @memcpy(out[pos * each ..][0..each], stored[at * each ..][0..each]);
    }
    return out;
}

/// `runWindows` over caches in order (`mode` null) or behind a table of their own (`Tabled`).
pub fn runWindowsOver(gpa: Allocator, shape: Shape, weights: Weights, rotary: Rotary, inputs: []const u16, noise: []const u16, total: usize, capacity: usize, bound: ?usize, sizes: []const [2]usize, mode: ?Tabled) !Run {
    var max_rows: usize = 0;
    for (sizes) |size| max_rows = @max(max_rows, size[0]);
    var cache: Cache = if (mode) |m| try .initTabled(shape, capacity, m) else try .contiguous(shape, capacity);
    defer cache.deinit();
    var scratch: Scratch = try .init(shape, max_rows, capacity);
    defer scratch.deinit();
    var position: Position = try .init();
    defer position.deinit();
    var x: cuda.Buffer = try .init(.bf16, &.{ max_rows, shape.dims });
    defer x.deinit();
    const staged = try gpa.alloc(u16, max_rows * shape.dims);
    defer gpa.free(staged);
    @memset(staged, 0);

    var run: Run = .{ .gated = try gpa.alloc(u16, total * shape.out()), .branch = try gpa.alloc(u16, total * shape.dims) };
    var turn: usize = 0;
    while (position.host < total) : (turn += 1) {
        const pos = position.host;
        const rows = @min(sizes[turn % sizes.len][0], capacity - pos);
        const keep = @min(sizes[turn % sizes.len][1], rows, total - pos);
        for (0..rows) |r| {
            const source = if (r < keep) inputs else noise;
            @memcpy(staged[r * shape.dims ..][0..shape.dims], source[(pos + r) * shape.dims ..][0..shape.dims]);
        }
        try x.upload(std.mem.sliceAsBytes(staged));
        try forward(shape, weights, rotary, cache, scratch, position, x.rows(rows), rows, bound, null, null);
        const gated = try fetch(u16, gpa, scratch.gated);
        defer gpa.free(gated);
        const branch = try fetch(u16, gpa, scratch.branch);
        defer gpa.free(branch);
        @memcpy(run.gated[pos * shape.out() ..][0 .. keep * shape.out()], gated[0 .. keep * shape.out()]);
        @memcpy(run.branch[pos * shape.dims ..][0 .. keep * shape.dims], branch[0 .. keep * shape.dims]);
        if (shape.selects(bound orelse pos + rows)) {
            // what the selection left for the kept rows: the regime, the list's length, and whether scores tied
            const selection = scratch.selection.?;
            const nk = try fetch(i32, gpa, selection.nk);
            defer gpa.free(nk);
            const sparse = try fetch(i32, gpa, selection.sparse);
            defer gpa.free(sparse);
            const scores = try fetch(f32, gpa, selection.scores);
            defer gpa.free(scores);
            for (0..keep) |r| {
                const end = pos + r + 1;
                try testing.expectEqual(@as(i32, @intFromBool(shape.selects(end))), sparse[r]);
                if (!shape.selects(end)) {
                    try testing.expectEqual(@as(i32, @intCast(end)), nk[r]);
                    continue;
                }
                try testing.expectEqual(@as(i32, @intCast(shape.budget + end % shape.ratio)), nk[r]);
                run.sparse_rows += 1;
                run.tied_rows += @intFromBool(try tiedAtCut(gpa, scores[r * selection.blocks ..][0 .. end / shape.ratio], shape.top()));
            }
        }
        try position.advance(keep);
    }
    try cuda.synchronize();
    run.k = try inOrder(gpa, cache, cache.k);
    run.v = try inOrder(gpa, cache, cache.v);
    if (cache.k_scale) |scale| run.k_scale = (std.mem.bytesAsSlice(f32, try inOrder(gpa, cache, scale)));
    if (cache.v_scale) |scale| run.v_scale = (std.mem.bytesAsSlice(f32, try inOrder(gpa, cache, scale)));
    run.index_keys = (std.mem.bytesAsSlice(u16, try inOrder(gpa, cache, cache.index_keys)));
    run.pooled = try fetch(u16, gpa, cache.pooled);
    return run;
}

/// Whether more blocks share the score of the `top`-th best than the selection can take: the lower block decides.
pub fn tiedAtCut(gpa: Allocator, scores: []const f32, top: usize) !bool {
    const sorted = try gpa.dupe(f32, scores);
    defer gpa.free(sorted);
    std.mem.sort(f32, sorted, {}, std.sort.desc(f32));
    return sorted[top] == sorted[top - 1];
}
