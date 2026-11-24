//! The caches of `attention.zig`'s block: how they hold a position's keys and values (`Storage`, an option of
//! the load, not of the checkpoint), the rows every sequence of a layer shares (`Pool`), the device table of
//! where a sequence's positions lie (`Table`), and one sequence's view of them (`Cache`). `attention.zig` holds
//! the block's shape and its launches, which read the caches through these, and `Tabled`, the layout a test
//! makes a cache in (`Cache.initTabled`).

const std = @import("std");
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const attention_file = @import("attention.zig");
const Shape = attention_file.Shape;
const chunk = attention_file.chunk;
const Tabled = attention_file.Tabled;

/// How the caches hold a position's keys and values.
pub const Storage = enum {
    /// As the model computes them.
    bf16,
    /// A head as e4m3 codes of x / s, s the head's largest magnitude over 448: half the bytes, and the scores
    /// run on the codes (`zig/kernels/qwen4/attention.cu`). The queries are quantized the same way.
    fp8,
    /// Keys and queries as `fp8`; a head's values as NVFP4: e2m1 codes, two a byte, then an e4m3 scale code a 16
    /// values, under the head's scale. 4.5 bits a value. Attend turns them into e4m3 codes as it loads them.
    fp8_nvfp4,

    /// Of the queries and keys.
    pub fn dtype(self: Storage) cuda.DType {
        return switch (self) {
            .bf16 => .bf16,
            .fp8, .fp8_nvfp4 => .e4m3,
        };
    }

    pub fn valueDtype(self: Storage) cuda.DType {
        return switch (self) {
            .bf16 => .bf16,
            .fp8 => .e4m3,
            .fp8_nvfp4 => .u8,
        };
    }

    /// The last dimension of the value cache: values, or the bytes of a head as NVFP4 (up to a multiple of 8).
    pub fn valueWidth(self: Storage, head_dim: usize) usize {
        return if (self == .fp8_nvfp4) (head_dim / 2 + head_dim / 16 + 7) / 8 * 8 else head_dim;
    }

    /// Whether heads carry scales.
    pub fn scaled(self: Storage) bool {
        return self != .bf16;
    }

    /// The flag's word: what `--kv-dtype` reads and the logs print.
    pub fn name(self: Storage) []const u8 {
        return switch (self) {
            .bf16 => "bf16",
            .fp8 => "fp8",
            .fp8_nvfp4 => "fp8-nvfp4",
        };
    }

    pub fn parse(text: []const u8) ?Storage {
        for (std.meta.tags(Storage)) |storage| if (std.mem.eql(u8, text, storage.name())) return storage;
        return null;
    }
};

/// Spec 018: one attention layer's cache rows for every sequence, as extents of `chunk` positions. Extent `e` is
/// rows [512 e, 512 e + 512) of each array, the same index in every layer's pool (the MTP head's too), so that a
/// sequence has one table for all of them (`Table`). Made once and never moved: a captured graph holds its
/// addresses. Which sequence names which extent is the host's (`zig/src/decode/extents.zig`); nothing here knows a count.
pub const Pool = struct {
    extents: usize,
    /// The arrays below, one after another: one allocation a layer (spec 101), whose arrays are views into it.
    store: cuda.Buffer,
    /// bf16 or e4m3 [extents 512, kv_heads, head_dim]
    k: cuda.Buffer,
    /// As `Cache.v`, [extents 512, kv_heads, ..]
    v: cuda.Buffer,
    /// fp32 [extents 512, kv_heads] unless the storage is bf16
    k_scale: ?cuda.Buffer,
    v_scale: ?cuda.Buffer,
    /// bf16 [extents 512, index_dim]
    index_keys: cuda.Buffer,

    pub fn init(shape: Shape, count: usize) cuda.Error!Pool {
        const rows = count * chunk;
        var store: cuda.Buffer = try .init(.u8, &.{bytesFor(shape, count)});
        errdefer store.deinit();
        var at: usize = 0;
        const k = carve(store, &at, shape.kv.dtype(), &.{ rows, shape.kv_heads, shape.head_dim });
        const v = carve(store, &at, shape.kv.valueDtype(), &.{ rows, shape.kv_heads, shape.kv.valueWidth(shape.head_dim) });
        const k_scale: ?cuda.Buffer = if (shape.kv.scaled()) carve(store, &at, .f32, &.{ rows, shape.kv_heads }) else null;
        const v_scale: ?cuda.Buffer = if (shape.kv.scaled()) carve(store, &at, .f32, &.{ rows, shape.kv_heads }) else null;
        const index_keys = carve(store, &at, .bf16, &.{ rows, shape.index_dim });
        std.debug.assert(at == store.bytes());
        // Not zeroed (spec 101): no launch reads a row its sequence has not written. The attention loads a key,
        // a value and their scales for a row's own list alone and fills the rest of a tile with zeros
        // (`attention.cu`), the selection pools complete blocks alone (`qsa.cu`), a fork's
        // copy, a carry and a spill take a chain's rows below its end, and an extent a sequence frees is the
        // next one's as it was left. Under `cuda.Scratch.poisoned` the arrays hold NaN patterns instead, against
        // which the tests hold the bits (`attention_table_test.zig`, `rounds_test.zig`).
        try cuda.Scratch.poison(store);
        // A head's NVFP4 bytes past its codes and scales, up to a multiple of 8, are written by no launch and
        // read by none, but a row's bytes are copied and hashed whole (a spill's file, the ranks' checks): zeros,
        // as a launch. A head of 128 or 256 values has none.
        if (shape.kv == .fp8_nvfp4 and shape.kv.valueWidth(shape.head_dim) > shape.head_dim / 2 + shape.head_dim / 16) try v.zeroLaunched();
        return .{ .extents = count, .store = store, .k = k, .v = v, .k_scale = k_scale, .v_scale = v_scale, .index_keys = index_keys };
    }

    /// `shape`'s array at byte `at.*` of `store`, a view `deinit` leaves alone; `at` moves past it. Each array's
    /// bytes are a multiple of 512 rows' and so of `cudaMalloc`'s 256-byte alignment.
    fn carve(store: cuda.Buffer, at: *usize, dtype: cuda.DType, shape: []const usize) cuda.Buffer {
        std.debug.assert(at.* % 256 == 0);
        const base: [*]u8 = @ptrCast(store.view.ptr.?);
        const array: cuda.Buffer = .{ .view = .init(base + at.*, dtype, shape) };
        at.* += array.bytes();
        std.debug.assert(at.* <= store.bytes());
        return array;
    }

    /// Bytes one position of `shape` costs in the pool. A rank passes its own shape: half the KV heads, the
    /// index keys whole (`share_layout.zig`).
    pub fn positionBytes(shape: Shape) usize {
        return bytesFor(shape, 1) / chunk;
    }

    /// The device bytes `init` asks for `count` extents of `shape`, with nothing made: its arrays' own sizes.
    pub fn bytesFor(shape: Shape, count: usize) usize {
        const kv = shape.kv;
        var row = shape.kv_heads * shape.head_dim * kv.dtype().size();
        row += shape.kv_heads * kv.valueWidth(shape.head_dim) * kv.valueDtype().size();
        if (kv.scaled()) row += 2 * shape.kv_heads * cuda.DType.f32.size();
        row += shape.index_dim * cuda.DType.bf16.size();
        return count * chunk * row;
    }

    pub fn deinit(self: *Pool) void {
        self.store.deinit();
        self.* = undefined;
    }

    /// A fork's copy in this layer (`extents.Copy`): the first `rows` rows of extent `from` into extent `to`, of
    /// every array, as launches in the stream's order. The caller runs it in every layer's pool before a call
    /// that can free `from`, and before the new sequence's first window. The pooled keys are the sequence's own
    /// and are copied by `Cache.inherit`.
    pub fn copy(self: Pool, c: extents.Copy) cuda.Error!void {
        const from: usize = @intCast(c.from);
        const to: usize = @intCast(c.to);
        std.debug.assert(from != to and from < self.extents and to < self.extents and c.rows <= chunk);
        for ([_]?cuda.Buffer{ self.k, self.v, self.k_scale, self.v_scale, self.index_keys }) |buffer| {
            if (buffer) |b| try b.copyRows(to * chunk, b, from * chunk, c.rows);
        }
    }
};

/// Spec 018: where a sequence's positions lie in the pools, on the device: int32 [capacity / 512], entry
/// `pos >> 9` the extent of position `pos` (`extents.Chain.table`). One for a sequence, read by every layer's
/// kernels and by the MTP head's, whose entries are the same positions and may run ahead of the main model's by
/// its drafts: a round grows the chain past its window by the most rows a head chains (`extents.Pool.grow`
/// in `hall_engine.zig`'s `roundInner`), and `set` after it. Entries past the chain's extents are never read by a row.
pub const Table = struct {
    device: cuda.Buffer,

    /// For sequences of up to `capacity` positions.
    pub fn init(capacity: usize) cuda.Error!Table {
        var device: cuda.Buffer = try .init(.i32, &.{@max(extents.extentsFor(capacity), 1)});
        errdefer device.deinit();
        try device.zero();
        return .{ .device = device };
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(capacity: usize) usize {
        return cuda.Buffer.bytesOf(.i32, &.{@max(extents.extentsFor(capacity), 1)});
    }

    /// Uploads a chain's table (`chain.table.items`), after `grow`, a fork or a revival changed it: an upload
    /// of at most 4 bytes an extent, in the stream's order as any copy from the host is.
    pub fn set(self: Table, entries: []const i32) cuda.Error!void {
        std.debug.assert(entries.len <= @as(usize, @intCast(self.device.view.size[0])));
        if (entries.len > 0) try self.device.uploadAt(0, std.mem.sliceAsBytes(entries));
    }

    pub fn deinit(self: *Table) void {
        self.device.deinit();
        self.* = undefined;
    }
};

/// One attention layer's caches for a sequence; rows 0 .. position - 1 are committed.
pub const Cache = struct {
    capacity: usize,
    /// Spec 018, null wherever zspark serves: int32 [capacity / 512] (`Table`), and then `k`, `v`, their scales
    /// and `index_keys` are a pool's arrays, position p at row `table[p >> 9] * 512 + p % 512`. `pooled` is
    /// the sequence's own either way.
    table: ?cuda.View = null,
    /// What `deinit` frees: every buffer, or `pooled` alone over a `Pool` that others share.
    lent: bool = false,
    /// The table and the pool of a cache made by `initTabled`.
    own_table: ?cuda.Buffer = null,
    own_pool: ?Pool = null,
    /// bf16 or e4m3 [capacity, kv_heads, head_dim]: normalized, rotated keys
    k: cuda.Buffer,
    /// bf16 or e4m3 [capacity, kv_heads, head_dim], or uint8 [capacity, kv_heads, `Storage.valueWidth`]: values as
    /// projected
    v: cuda.Buffer,
    /// fp32 [capacity, kv_heads] unless the storage is bf16: what a key's codes are multiplied by
    k_scale: ?cuda.Buffer,
    /// fp32 [capacity, kv_heads] unless the storage is bf16
    v_scale: ?cuda.Buffer,
    /// bf16 [capacity, index_dim]: the indexer's keys as projected
    index_keys: cuda.Buffer,
    /// bf16 [capacity / ratio, index_dim]: each complete block's pooled, normalized, rotated indexer key
    pooled: cuda.Buffer,

    pub fn init(shape: Shape, capacity: usize) cuda.Error!Cache {
        return contiguous(shape, capacity);
    }

    /// The sequence's rows in order.
    pub fn contiguous(shape: Shape, capacity: usize) cuda.Error!Cache {
        var k: cuda.Buffer = try .init(shape.kv.dtype(), &.{ capacity, shape.kv_heads, shape.head_dim });
        errdefer k.deinit();
        var v: cuda.Buffer = try .init(shape.kv.valueDtype(), &.{ capacity, shape.kv_heads, shape.kv.valueWidth(shape.head_dim) });
        errdefer v.deinit();
        var k_scale: ?cuda.Buffer = if (shape.kv.scaled()) try .init(.f32, &.{ capacity, shape.kv_heads }) else null;
        errdefer if (k_scale) |*buffer| buffer.deinit();
        var v_scale: ?cuda.Buffer = if (shape.kv.scaled()) try .init(.f32, &.{ capacity, shape.kv_heads }) else null;
        errdefer if (v_scale) |*buffer| buffer.deinit();
        var index_keys: cuda.Buffer = try .init(.bf16, &.{ capacity, shape.index_dim });
        errdefer index_keys.deinit();
        var pooled: cuda.Buffer = try .init(.bf16, &.{ @max(capacity / shape.ratio, 1), shape.index_dim });
        errdefer pooled.deinit();
        for ([_]?cuda.Buffer{ k, v, k_scale, v_scale, index_keys, pooled }) |buffer| if (buffer) |b| try b.zero();
        return .{ .capacity = capacity, .k = k, .v = v, .k_scale = k_scale, .v_scale = v_scale, .index_keys = index_keys, .pooled = pooled };
    }

    /// A sequence's caches in `pool`, a layer's, behind `table` (the sequence's `Table.device.view`): the
    /// pool's arrays lent, the pooled keys its own. `capacity` is the positions the sequence may hold.
    pub fn over(shape: Shape, pool_: Pool, table: cuda.View, capacity: usize) cuda.Error!Cache {
        std.debug.assert(extents.extentsFor(capacity) <= @as(usize, @intCast(table.size[0])) and table.dtype == .i32);
        std.debug.assert(chunk % shape.ratio == 0);
        var pooled: cuda.Buffer = try .init(.bf16, &.{ @max(capacity / shape.ratio, 1), shape.index_dim });
        errdefer pooled.deinit();
        try pooled.zero();
        return .{ .capacity = capacity, .table = table, .lent = true, .k = pool_.k, .v = pool_.v, .k_scale = pool_.k_scale, .v_scale = pool_.v_scale, .index_keys = pool_.index_keys, .pooled = pooled };
    }

    /// The device bytes `over` asks, with nothing made: the sequence's own pooled keys, the rest being the pool's.
    pub fn bytesOver(shape: Shape, capacity: usize) usize {
        return cuda.Buffer.bytesOf(.bf16, &.{ @max(capacity / shape.ratio, 1), shape.index_dim });
    }

    /// A cache in the `Tabled` layout: a pool and a table of this cache's own.
    pub fn initTabled(shape: Shape, capacity: usize, mode: Tabled) cuda.Error!Cache {
        const count = @max(extents.extentsFor(capacity), 1);
        var places: [1024]i32 = undefined;
        // a context past 524,288 positions is not asked of this facility
        std.debug.assert(count <= places.len);
        for (places[0..count], 0..) |*place, i| place.* = @intCast(i);
        if (mode.scattered) {
            var prng: std.Random.DefaultPrng = .init(0x018b2 + count);
            prng.random().shuffle(i32, places[0..count]);
            // two extents or more never lie in order
            const in_order = for (places[0..count], 0..) |place, i| {
                if (place != i) break false;
            } else true;
            if (in_order) std.mem.reverse(i32, places[0..count]);
        }
        var table: Table = try .init(capacity);
        errdefer table.deinit();
        try table.set(places[0..count]);
        var pool_: Pool = try .init(shape, count);
        errdefer pool_.deinit();
        var self = try over(shape, pool_, table.device.view, capacity);
        self.own_table = table.device;
        self.own_pool = pool_;
        return self;
    }

    /// A fork's pooled keys: the blocks complete within `source`'s first `at` positions, copied as launches.
    /// The block `at` lies in is pooled again by the window that completes it, from the extents' index keys.
    pub fn inherit(self: Cache, shape: Shape, source: Cache, at: usize) cuda.Error!void {
        std.debug.assert(at <= self.capacity and at <= source.capacity);
        if (at / shape.ratio > 0) try self.pooled.copyRows(0, source.pooled, 0, at / shape.ratio);
    }

    pub fn deinit(self: *Cache) void {
        if (!self.lent) {
            self.k.deinit();
            self.v.deinit();
            if (self.k_scale) |*buffer| buffer.deinit();
            if (self.v_scale) |*buffer| buffer.deinit();
            self.index_keys.deinit();
        }
        if (self.own_table) |*buffer| buffer.deinit();
        if (self.own_pool) |*pool| pool.deinit();
        self.pooled.deinit();
        self.* = undefined;
    }
};
