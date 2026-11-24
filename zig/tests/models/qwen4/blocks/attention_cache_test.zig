//! `attention_cache.zig`'s tests: a fork inside an extent copies its rows on the device (`Pool.copy`, the pooled
//! keys `Cache.inherit`), each side then writing where the other does not read, both against their own run over
//! caches in order.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const fx = fixtures.attending.fx;
const Model = fixtures.attending.Model;
const Run = fixtures.attending.Run;
const fetch = fixtures.attending.fetch;
const fetchBytes = fixtures.attending.fetchBytes;
const randomBf16 = fixtures.attending.randomBf16;
const runWindowsOver = fixtures.attending.runWindowsOver;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const Cache = attention.Cache;
const Pool = attention.Pool;
const Position = attention.Position;
const Scratch = attention.Scratch;
const Shape = attention.Shape;
const Storage = attention.Storage;
const Table = attention.Table;
const chunk = attention.chunk;
const forward = attention.forward;
/// A sequence decoded through one attention block over a layer's pool, its extents the host pool's.
const Stream = struct {
    chain: *extents.Chain,
    table: Table,
    cache: Cache,
    scratch: Scratch,
    position: Position,
    x: cuda.Buffer,
    gated: std.ArrayList(u16) = .empty,
    branch: std.ArrayList(u16) = .empty,

    const window = 16;

    fn init(shape: Shape, layer: Pool, chain: *extents.Chain, capacity: usize) !Stream {
        var table: Table = try .init(capacity);
        errdefer table.deinit();
        var cache: Cache = try .over(shape, layer, table.device.view, capacity);
        errdefer cache.deinit();
        var scratch: Scratch = try .init(shape, window, capacity);
        errdefer scratch.deinit();
        var position: Position = try .init();
        errdefer position.deinit();
        return .{ .chain = chain, .table = table, .cache = cache, .scratch = scratch, .position = position, .x = try .init(.bf16, &.{ window, shape.dims }) };
    }

    fn deinit(self: *Stream) void {
        self.table.deinit();
        self.cache.deinit();
        self.scratch.deinit();
        self.position.deinit();
        self.x.deinit();
    }

    /// One window of `rows` of `inputs` [.., dims] at the stream's position, all kept: the chain grown to it
    /// first, as a round does.
    fn step(self: *Stream, gpa: Allocator, host: *extents.Pool, shape: Shape, model: Model, inputs: []const u16, rows: usize) !void {
        const pos = self.position.host;
        std.debug.assert(pos == self.chain.len());
        try host.grow(self.chain, pos + rows);
        try self.table.set(self.chain.table.items);
        try self.x.uploadAt(0, std.mem.sliceAsBytes(inputs[pos * shape.dims ..][0 .. rows * shape.dims]));
        try forward(shape, model.weights, model.rotary, self.cache, self.scratch, self.position, self.x.rows(rows), rows, null, null, null);
        const gated = try fetch(u16, gpa, self.scratch.gated);
        const branch = try fetch(u16, gpa, self.scratch.branch);
        try self.gated.appendSlice(gpa, gated[0 .. rows * shape.out()]);
        try self.branch.appendSlice(gpa, branch[0 .. rows * shape.dims]);
        try self.position.advance(rows);
        const tokens = try gpa.alloc(i32, rows);
        @memset(tokens, 7);
        host.append(self.chain, tokens);
    }

    /// The stream's outputs from position `from` on, and its caches' rows through its chain, against a run over
    /// caches in order.
    fn expectRun(self: *Stream, gpa: Allocator, shape: Shape, want: Run, from: usize) !void {
        const total = self.chain.len();
        try testing.expectEqualSlices(u16, want.gated[from * shape.out() .. total * shape.out()], self.gated.items);
        try testing.expectEqualSlices(u16, want.branch[from * shape.dims .. total * shape.dims], self.branch.items);
        const pairs = [_]struct { ?cuda.Buffer, []const u8 }{ .{ self.cache.k, want.k }, .{ self.cache.v, want.v }, .{ self.cache.k_scale, std.mem.sliceAsBytes(want.k_scale) }, .{ self.cache.v_scale, std.mem.sliceAsBytes(want.v_scale) }, .{ self.cache.index_keys, std.mem.sliceAsBytes(want.index_keys) } };
        for (pairs) |pair| {
            const stored = try fetchBytes(gpa, pair[0] orelse continue);
            const each = stored.len / @as(usize, @intCast(pair[0].?.view.size[0]));
            for (0..total) |pos| try testing.expectEqualSlices(u8, pair[1][pos * each ..][0..each], stored[self.chain.row(pos) * each ..][0..each]);
        }
        const pooled = try fetch(u16, gpa, self.cache.pooled);
        const blocks = total / shape.ratio * shape.index_dim;
        try testing.expectEqualSlices(u16, want.pooled[0..blocks], pooled[0..blocks]);
    }
};

test "a fork inside an extent copies its rows on the device, and each side then writes where the other does not read: both equal their own run in order" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(185);
    const random = prng.random();
    // forks inside the second extent: within a block of four, at a block's end, and at an extent's end, where
    // nothing is copied
    for ([_]usize{ 701, 700, 1024 }) |at| for ([_]Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        var shape = fx.small;
        shape.kv = storage;
        const total = at + 230;
        const capacity = 4 * chunk;
        var model: Model = try .init(gpa, random, shape, Stream.window);
        defer model.deinit();
        // the two sequences share their first `at` inputs and part after
        const first = try randomBf16(gpa, random, total * shape.dims, 1.0);
        const second = try randomBf16(gpa, random, total * shape.dims, 1.0);
        @memcpy(second[0 .. at * shape.dims], first[0 .. at * shape.dims]);
        const want_first = try runWindowsOver(gpa, shape, model.weights, model.rotary, first, first, total, total, null, &.{.{ 1, 1 }}, null);
        const want_second = try runWindowsOver(gpa, shape, model.weights, model.rotary, second, second, total, total, null, &.{.{ 1, 1 }}, null);
        try testing.expect(!std.mem.eql(u16, want_first.branch[at * shape.dims ..], want_second.branch[at * shape.dims ..]));

        var host: extents.Pool = try .init(testing.allocator, .{ .extents = 12, .capacity = capacity, .slots = 3 });
        defer host.deinit();
        var layer: Pool = try .init(shape, 12);
        defer layer.deinit();
        // another stream takes extents between this one's, so that a table is not in order
        const filler = try host.open();
        try host.grow(filler, 1);
        var a: Stream = try .init(shape, layer, try host.open(), capacity);
        defer a.deinit();
        var filled: usize = 1;
        while (a.chain.len() < at) {
            const rows = @min(Stream.window, at - a.chain.len());
            if ((a.chain.len() + rows - 1) / chunk + 1 > a.chain.table.items.len) {
                // the filler first, where the stream is about to ask for an extent
                filled += chunk;
                try host.grow(filler, filled);
            }
            try a.step(gpa, &host, shape, model, first, rows);
        }
        try testing.expect(a.chain.table.items[0] != 0 and a.chain.table.items[1] != a.chain.table.items[0] + 1);

        const forked = try host.fork(a.chain, at);
        var b: Stream = try .init(shape, layer, forked.chain, capacity);
        defer b.deinit();
        try testing.expectEqual(at % chunk != 0, forked.copy != null);
        if (forked.copy) |c| {
            try testing.expectEqual(at % chunk, c.rows);
            try layer.copy(c);
        }
        try b.cache.inherit(shape, a.cache, at);
        try b.position.set(at);
        // each in turn, in windows of different sizes, the filler taking an extent between theirs
        var turn: usize = 0;
        while (a.chain.len() < total or b.chain.len() < total) : (turn += 1) {
            if (turn == 9) try host.grow(filler, filled + chunk);
            if (a.chain.len() < total) try a.step(gpa, &host, shape, model, first, @min(@as(usize, if (turn % 3 == 0) 1 else Stream.window), total - a.chain.len()));
            if (b.chain.len() < total) try b.step(gpa, &host, shape, model, second, @min(@as(usize, if (turn % 2 == 0) 7 else 1), total - b.chain.len()));
        }
        try cuda.synchronize();
        try host.check(gpa);
        // the first extent is named by both; past the fork each has its own
        try testing.expectEqual(a.chain.table.items[0], b.chain.table.items[0]);
        try testing.expect(a.chain.table.items[at / chunk] != b.chain.table.items[at / chunk]);
        try a.expectRun(gpa, shape, want_first, 0);
        try b.expectRun(gpa, shape, want_second, at);
        _ = arena_state.reset(.retain_capacity);
    };
}
