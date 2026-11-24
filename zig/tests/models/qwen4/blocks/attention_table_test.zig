//! `attention.zig`'s tests through a table of scattered extents: the launchers of each kind against the caches
//! in order, bit for bit.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const fx = fixtures.attending.fx;
const Model = fixtures.attending.Model;
const Stored = fixtures.attending.Stored;
const fetch = fixtures.attending.fetch;
const randomBf16 = fixtures.attending.randomBf16;
const fetchBytes = fixtures.attending.fetchBytes;
const runWindowsOver = fixtures.attending.runWindowsOver;
const tiedAtCut = fixtures.attending.tiedAtCut;
const hostList = fixtures.attending.hostList;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");
const Cache = attention.Cache;
const Position = attention.Position;
const Scratch = attention.Scratch;
const Selection = attention.Selection;
const Shape = attention.Shape;
const Storage = attention.Storage;
const chunk = attention.chunk;
const pick = attention.pick;
const uploaded = attention.uploaded;
const numeric = @import("../../../../src/weights/numeric.zig");
const Tabled = attention.Tabled;
const attend = attention.attend;
const pool = attention.pool;
const prep = attention.prep;

/// A sequence's extents scattered in a pool: `table[e]` is where positions [512 e, 512 e + 512) lie.
const Scattered = struct {
    table: []i32,
    device: cuda.Buffer,
    /// The pool's rows.
    capacity: usize,

    /// `extents` of a sequence at places drawn from twice as many and one more, no two alike.
    fn init(gpa: Allocator, random: std.Random, count: usize) !Scattered {
        const places = try gpa.alloc(i32, 2 * count + 1);
        defer gpa.free(places);
        for (places, 0..) |*place, i| place.* = @intCast(i);
        random.shuffle(i32, places);
        const table = try gpa.dupe(i32, places[0..count]);
        var device: cuda.Buffer = try .init(.i32, &.{count});
        errdefer device.deinit();
        try device.upload(std.mem.sliceAsBytes(table));
        return .{ .table = table, .device = device, .capacity = places.len * chunk };
    }

    fn row(self: Scattered, pos: usize) usize {
        return @as(usize, @intCast(self.table[pos / chunk])) * chunk + pos % chunk;
    }

    /// `rows` of `each` bytes in the sequence's order, as the pool holds them; what no row lies at is `fill`.
    fn spread(self: Scattered, gpa: Allocator, rows: []const u8, each: usize, fill: u8) ![]u8 {
        const out = try gpa.alloc(u8, self.capacity * each);
        @memset(out, fill);
        for (0..rows.len / each) |pos| @memcpy(out[self.row(pos) * each ..][0..each], rows[pos * each ..][0..each]);
        return out;
    }

    fn deinit(self: *Scattered) void {
        self.device.deinit();
    }
};

/// `zs_attn_attend_window_rows` over `cache`'s pool as a round of one stream: rows `pos0 ..` of slot 0, whose
/// table is `table`'s.
fn attendWindowRows(gpa: Allocator, table: Scattered, cache: Cache, scratch: Scratch, selection: ?Selection, pos0: usize, rows: usize, end: usize) !void {
    const sid = try gpa.alloc(i32, rows);
    const posr = try gpa.alloc(i32, rows);
    for (sid, posr, 0..) |*s, *p, r| {
        s.* = 0;
        p.* = @intCast(pos0 + r);
    }
    var sid_device: cuda.Buffer = try .init(.i32, &.{rows});
    defer sid_device.deinit();
    try sid_device.upload(std.mem.sliceAsBytes(sid));
    var posr_device: cuda.Buffer = try .init(.i32, &.{rows});
    defer posr_device.deinit();
    try posr_device.upload(std.mem.sliceAsBytes(posr));
    const address = struct {
        fn of(view: cuda.View) i64 {
            return @intCast(@intFromPtr(view.ptr.?));
        }
    }.of;
    var tables: cuda.Buffer = try .init(.i64, &.{1});
    defer tables.deinit();
    try tables.upload(std.mem.sliceAsBytes(&[_]i64{address(table.device.view)}));
    var bases: cuda.Buffer = try .init(.i64, &.{ 1, 3 });
    defer bases.deinit();
    try bases.upload(std.mem.sliceAsBytes(&[_]i64{ address(cache.k.view), address(cache.v.view), 0 }));
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(cache.k.view.size[2])));
    const ids = if (selection) |*lists| &lists.ids.view else null;
    const nk = if (selection) |*lists| &lists.nk.view else null;
    const sparse = if (selection) |*lists| &lists.sparse.view else null;
    const viewOf = attention.viewOf;
    try cuda.launched(cuda.kernels.zs_attn_attend_window_rows(&sid_device.view, &tables.view, &bases.view, &scratch.q.rows(rows), &cache.k.view, &cache.v.view, &posr_device.view, &scratch.attn.rows(rows), ids, nk, sparse, viewOf(&scratch.q_scale), viewOf(&cache.k_scale), viewOf(&cache.v_scale), scale, @intCast(end)));
}

test "attention through a table of scattered extents gives the contiguous caches' bits: three storages, dense and listed rows, 1, 16 and 512 rows" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(180);
    const random = prng.random();
    // five extents, the last one in part
    const capacity = 2400;
    for ([_]Shape{ fx.small, fx.wide, fx.full }) |base| for ([_]Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        var table: Scattered = try .init(gpa, random, (capacity + chunk - 1) / chunk);
        defer table.deinit();
        const hd = base.head_dim;
        const coded: Storage = if (storage == .bf16) .bf16 else .fp8;
        const k: Stored = try .random(gpa, random, coded, capacity, base.kv_heads, hd, 1.0);
        const v: Stored = try .random(gpa, random, storage, capacity, base.kv_heads, hd, 1.0);
        // a budget past the caches: every row is dense, over as many chunks as it has keys; and the small one,
        // under which every row here reads a list
        for ([_]usize{ 4096, fx.small.budget }) |budget| {
            var shape = base;
            shape.budget = budget;
            shape.kv = storage;
            var cache: Cache = try .contiguous(shape, capacity);
            defer cache.deinit();
            var pooled: Cache = try .contiguous(shape, table.capacity);
            defer pooled.deinit();
            try k.upload(cache.k, cache.k_scale);
            try v.upload(cache.v, cache.v_scale);
            // what no position lies at holds other bytes than any row's place in order would
            try pooled.k.upload(try table.spread(gpa, k.bytes, k.each * shape.kv_heads, 0x3c));
            try pooled.v.upload(try table.spread(gpa, v.bytes, v.each * shape.kv_heads, 0x3c));
            if (shape.kv.scaled()) {
                try pooled.k_scale.?.upload(try table.spread(gpa, std.mem.sliceAsBytes(k.scales), 4 * shape.kv_heads, 0x3c));
                try pooled.v_scale.?.upload(try table.spread(gpa, std.mem.sliceAsBytes(v.scales), 4 * shape.kv_heads, 0x3c));
            }
            var position: Position = try .init();
            defer position.deinit();
            for ([_]usize{ 1, 16, 512 }) |rows| {
                var scratch: Scratch = try .init(shape, rows, capacity);
                defer scratch.deinit();
                var other: Scratch = try .init(shape, rows, capacity);
                defer other.deinit();
                const q: Stored = try .random(gpa, random, coded, rows, shape.heads, hd, 1.5);
                try q.upload(scratch.q, scratch.q_scale);
                try q.upload(other.q, other.q_scale);
                // windows that end inside the first extent, at an extent's end, one row past it, and in the last
                for ([_]usize{ @max(rows, 300), 1024, 1025, capacity }) |end| {
                    const pos0 = end - rows;
                    try position.set(pos0);
                    var selection: ?Selection = null;
                    var selection_other: ?Selection = null;
                    if (scratch.selection) |lists| {
                        // listed rows of every length to the longest, dense rows among them where a row has no
                        // more keys than a list holds, keys from anywhere before the row's end
                        const width = shape.listWidth();
                        const ids = try gpa.alloc(i32, rows * width);
                        const nk = try gpa.alloc(i32, rows);
                        const sparse = try gpa.alloc(i32, rows);
                        for (0..rows) |r| {
                            sparse[r] = @intFromBool(r % 7 != 3 or pos0 + r + 1 > width - 1);
                            // a list is no longer than the positions before its row's end, as the launch's chunks are
                            const count = if (sparse[r] == 0 or r % 5 == 0) width - 1 else 1 + random.uintLessThan(usize, width - 1);
                            nk[r] = @intCast(@min(count, pos0 + r + 1));
                            for (ids[r * width ..][0..width]) |*key| key.* = random.intRangeLessThan(i32, 0, @intCast(pos0 + r + 1));
                        }
                        for ([_]Selection{ lists, other.selection.? }) |to| {
                            try to.ids.upload(std.mem.sliceAsBytes(ids));
                            try to.nk.upload(std.mem.sliceAsBytes(nk));
                            try to.sparse.upload(std.mem.sliceAsBytes(sparse));
                        }
                        selection = lists;
                        selection_other = other.selection.?;
                    }
                    try attend(shape, cache, scratch, selection, scratch.q.view, position.device.view, scratch.attn.view, end);
                    const want = try fetch(u16, gpa, scratch.attn);
                    // the same rows in a pool, as a cache behind the table: lists of the pool's rows, as the
                    // selection writes them through the table
                    var behind = pooled;
                    behind.table = table.device.view;
                    if (selection_other) |lists| {
                        const ids = try fetch(i32, gpa, lists.ids);
                        for (ids) |*key| key.* = @intCast(table.row(@intCast(key.*)));
                        try lists.ids.upload(std.mem.sliceAsBytes(ids));
                    }
                    try other.attn.zero();
                    try attend(shape, behind, other, selection_other, other.q.view, position.device.view, other.attn.view, end);
                    try testing.expectEqualSlices(u16, want, try fetch(u16, gpa, other.attn));
                    // the window's form over a round's rows (attention_window_rows.cu, spec 149 WS-02), every row
                    // the one stream's at its own position, whatever the rows: the same bits
                    try other.attn.zero();
                    try attendWindowRows(gpa, table, behind, other, selection_other, pos0, rows, end);
                    try testing.expectEqualSlices(u16, want, try fetch(u16, gpa, other.attn));
                    // Every value is finite, so the equal bits are not NaN against NaN. These random rows hold
                    // tiles 80 and more under their row's largest score (fp8, 32 values a head, the dense row at
                    // position 930 of the window ending at 1024, head 1), which gave a head of NaN before
                    // spec 043.
                    for (want) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
                }
            }
        }
        _ = arena_state.reset(.retain_capacity);
    };
}

test "prep through a table of scattered extents writes the contiguous caches' rows at their places, and nothing else" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(181);
    const random = prng.random();
    const capacity = 2048;
    for ([_]Shape{ fx.small, fx.wide, fx.full }) |base| for ([_]Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        var shape = base;
        shape.kv = storage;
        var model: Model = try .init(gpa, random, shape, 16);
        defer model.deinit();
        var position: Position = try .init();
        defer position.deinit();
        // a row alone at an extent's last position and at the next one's first, 16 rows over an extent's end, 512
        // rows over two ends
        for ([_][2]usize{ .{ 1, 1023 }, .{ 1, 1024 }, .{ 16, 505 }, .{ 512, 700 }, .{ 16, 1017 } }) |window| {
            const rows = window[0];
            const pos0 = window[1];
            var table: Scattered = try .init(gpa, random, capacity / chunk);
            defer table.deinit();
            var cache: Cache = try .contiguous(shape, capacity);
            defer cache.deinit();
            var scratch: Scratch = try .init(shape, rows, capacity);
            defer scratch.deinit();
            const pa = try randomBf16(gpa, random, rows * shape.projection(), 1.0);
            try scratch.pa.upload(std.mem.sliceAsBytes(pa));
            try position.set(pos0);
            try prep(shape, model.weights, model.rotary, cache, scratch, scratch.pa.view, position.device.view, rows, pos0 + rows, null);
            // a cache behind the table through `prep`
            var behind: Cache = try .contiguous(shape, table.capacity);
            defer behind.deinit();
            behind.table = table.device.view;
            var third: Scratch = try .init(shape, rows, capacity);
            defer third.deinit();
            try prep(shape, model.weights, model.rotary, behind, third, scratch.pa.view, position.device.view, rows, pos0 + rows, null);
            try testing.expectEqualSlices(u8, try fetchBytes(gpa, scratch.q), try fetchBytes(gpa, third.q));
            try testing.expectEqualSlices(u16, try fetch(u16, gpa, scratch.iq), try fetch(u16, gpa, third.iq));
            if (scratch.q_scale) |scale| try testing.expectEqualSlices(f32, try fetch(f32, gpa, scale), try fetch(f32, gpa, third.q_scale.?));
            // each cache: the contiguous one's rows spread by the table are the pool, whose other rows are the
            // zeros it started as (so are the contiguous cache's outside the window)
            const pairs = [_][2]?cuda.Buffer{ .{ cache.k, behind.k }, .{ cache.v, behind.v }, .{ cache.k_scale, behind.k_scale }, .{ cache.v_scale, behind.v_scale }, .{ cache.index_keys, behind.index_keys } };
            for (pairs) |pair| {
                const ordered = try fetchBytes(gpa, pair[0] orelse continue);
                const got = try fetchBytes(gpa, pair[1].?);
                const want = try table.spread(gpa, ordered, ordered.len / capacity, 0);
                try testing.expectEqualSlices(u8, want, got);
                var written: usize = 0;
                for (ordered) |byte| written += @intFromBool(byte != 0);
                try testing.expect(written > rows);
            }
        }
        _ = arena_state.reset(.retain_capacity);
    };
}

test "pooling through a table reads each block's index keys where they lie, and writes the pooled keys in order" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(182);
    const random = prng.random();
    const capacity = 2048;
    // blocks of 4, of 2 and of 1 position: an extent holds whole blocks of each
    for ([_]usize{ 4, 2, 1 }) |ratio| {
        var shape = fx.small;
        shape.ratio = ratio;
        const id = shape.index_dim;
        var model: Model = try .init(gpa, random, shape, 1);
        defer model.deinit();
        var table: Scattered = try .init(gpa, random, capacity / chunk);
        defer table.deinit();
        var cache: Cache = try .contiguous(shape, capacity);
        defer cache.deinit();
        var behind: Cache = try .contiguous(shape, table.capacity);
        defer behind.deinit();
        behind.table = table.device.view;
        behind.capacity = capacity;
        var position: Position = try .init();
        defer position.deinit();
        const keys = try randomBf16(gpa, random, capacity * id, 1.0);
        try cache.index_keys.upload(std.mem.sliceAsBytes(keys));
        // what no position lies at would pool to other keys
        try behind.index_keys.upload(try table.spread(gpa, std.mem.sliceAsBytes(keys), 2 * id, 0x3c));
        const untouched = try gpa.alloc(u16, (table.capacity / ratio) * id);
        @memset(untouched, 0x7fc0);
        // (first position, rows): a block that ends an extent, one that begins the next, windows over one end
        // and over two, a window that completes nothing
        for ([_][2]usize{ .{ 508, 4 }, .{ 512, 4 }, .{ 500, 16 }, .{ 700, 512 }, .{ 1023, 1 }, .{ 1024, 1 }, .{ 0, 2048 } }) |window| {
            const pos0, const rows = window;
            try position.set(pos0);
            for ([_]Cache{ cache, behind }) |to| {
                try to.pooled.upload(std.mem.sliceAsBytes(untouched[0 .. to.pooled.bytes() / 2]));
                try pool(shape, model.weights, model.rotary, to, position.device.view, rows, pos0 + rows, null);
            }
            const want = try fetch(u16, gpa, cache.pooled);
            const got = try fetch(u16, gpa, behind.pooled);
            try testing.expectEqualSlices(u16, want, got[0..want.len]);
            var written: usize = 0;
            for (want) |bits| written += @intFromBool(bits != 0x7fc0);
            try testing.expect(written >= ((pos0 + rows) / ratio - pos0 / ratio) * id / 2);
            // every pooled key written is finite: the untouched ones are the NaN they were set to
            for (want) |bits| try testing.expect(bits == 0x7fc0 or std.math.isFinite(numeric.f32FromBf16(bits)));
        }
        _ = arena_state.reset(.retain_capacity);
    }
}

test "the selection through a table lists the pool's rows of the positions it lists without: ties, tails and blocks that end an extent" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(183);
    const random = prng.random();
    const heads = 3;
    const dim = 64;
    const Case = struct { ratio: usize, top: usize, pos0: usize, rows: usize };
    const cases = [_]Case{
        // rows on both sides of the first sparse row (end 644), tails of 0 to 3 positions
        .{ .ratio = 4, .top = 160, .pos0 = 640, .rows = 9 },
        // tails that begin an extent (ends 2049 to 2051), and rows whose last block ends one
        .{ .ratio = 4, .top = 300, .pos0 = 2040, .rows = 16 },
        // more blocks than the kernel has threads
        .{ .ratio = 4, .top = 5, .pos0 = 12000, .rows = 6 },
        // no tail, and a single block to take; blocks of two
        .{ .ratio = 1, .top = 1, .pos0 = 1500, .rows = 6 },
        .{ .ratio = 2, .top = 700, .pos0 = 3066, .rows = 12 },
        // a wide window
        .{ .ratio = 4, .top = 512, .pos0 = 3900, .rows = 512 },
    };
    var position: Position = try .init();
    defer position.deinit();
    // lists that run from an extent's last position into the next one's first
    var crossings: usize = 0;
    defer std.debug.assert(crossings > 0);
    for (cases) |case| {
        // the arena resets at the end of the case, so the list lives in this iteration
        var host_ids: std.ArrayList(i32) = .empty;
        var shape = fx.small;
        shape.ratio = case.ratio;
        shape.budget = case.top * case.ratio;
        const context = case.pos0 + case.rows;
        const blocks = context / case.ratio;
        var table: Scattered = try .init(gpa, random, extents.extentsFor(context));
        defer table.deinit();
        const back = try gpa.alloc(usize, table.capacity / chunk);
        @memset(back, std.math.maxInt(usize));
        for (table.table, 0..) |place, e| back[@intCast(place)] = e;
        var selection: Selection = try .init(shape, case.rows, context + 40);
        defer selection.deinit();
        var through: Selection = try .init(shape, case.rows, context + 40);
        defer through.deinit();
        // the products of the test without a table: built to tie
        const dots = try gpa.alloc(f32, case.rows * heads * blocks);
        for (0..case.rows) |r| for (0..heads) |h| for (0..blocks) |b| {
            const value: f32 = switch (r % 8) {
                0 => @as(f32, @floatFromInt(random.uintLessThan(u8, 4))) - 1.0,
                1 => 0.75,
                2 => -1.0 - random.float(f32),
                3 => if (b % 7 == 3) 2.0 else if (b % 5 == 0) 1.0 else @as(f32, @floatFromInt(b % 3)) * 0.125,
                else => random.floatNorm(f32),
            };
            dots[(r * heads + h) * blocks + b] = value;
        };
        var dots_dev = try uploaded(f32, .f32, &.{ case.rows * heads, blocks }, dots);
        defer dots_dev.deinit();
        try position.set(case.pos0);
        try pick(shape, selection, dots_dev.view, position.device.view, 0, case.rows, heads, dim, null);
        try pick(shape, through, dots_dev.view, position.device.view, 0, case.rows, heads, dim, table.device.view);
        const want_ids = try fetch(i32, gpa, selection.ids);
        const want_nk = try fetch(i32, gpa, selection.nk);
        const want_sparse = try fetch(i32, gpa, selection.sparse);
        const got_ids = try fetch(i32, gpa, through.ids);
        try testing.expectEqualSlices(i32, want_nk, try fetch(i32, gpa, through.nk));
        try testing.expectEqualSlices(i32, want_sparse, try fetch(i32, gpa, through.sparse));
        const width = shape.listWidth();
        var sparse_rows: usize = 0;
        var tied_rows: usize = 0;
        var crossing: usize = 0;
        const through_scores = try fetch(f32, gpa, through.scores);
        const scores = try fetch(f32, gpa, selection.scores);
        for (0..case.rows) |r| {
            if (want_sparse[r] == 0) continue;
            sparse_rows += 1;
            const end = case.pos0 + r + 1;
            tied_rows += @intFromBool(try tiedAtCut(gpa, scores[r * selection.blocks ..][0 .. end / case.ratio], case.top));
            try testing.expectEqualSlices(u32, @ptrCast(scores[r * selection.blocks ..][0 .. end / case.ratio]), @ptrCast(through_scores[r * selection.blocks ..][0 .. end / case.ratio]));
            const count: usize = @intCast(want_nk[r]);
            // the host's list from the same scores: the lower block among equal ones, in rising order
            try testing.expect(try hostList(gpa, through_scores[r * through.blocks ..][0 .. end / case.ratio], end, case.ratio, case.top, &host_ids));
            try testing.expectEqualSlices(i32, host_ids.items, want_ids[r * width ..][0..count]);
            for (want_ids[r * width ..][0..count], got_ids[r * width ..][0..count], 0..) |pos, at, i| {
                // a row of the pool, at the place the table gives the position: mapped back, the same list in
                // the same order
                const stored: usize = @intCast(at);
                try testing.expectEqual(table.row(@intCast(pos)), stored);
                try testing.expectEqual(@as(usize, @intCast(pos)), back[stored / chunk] * chunk + stored % chunk);
                if (i > 0 and want_ids[r * width + i - 1] + 1 == pos and @as(usize, @intCast(pos)) % chunk == 0) crossing += 1;
            }
        }
        try testing.expect(sparse_rows >= 6 and tied_rows >= 1);
        crossings += crossing;
        _ = arena_state.reset(.retain_capacity);
    }
}

test "windows through a table of scattered extents equal serial steps over caches in order, across the dense limit and the extents' ends: three storages, both kinds of list" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(184);
    const random = prng.random();
    // the small shape, whose selection starts at 643 and runs over the end of the first extent and of the second;
    // and the model's budget, whose first listed row ends at position 2,052, in the fifth extent
    var limit = fx.wide;
    limit.budget = 2048;
    const Case = struct { shape: Shape, total: usize };
    for ([_]Case{ .{ .shape = fx.small, .total = 1100 }, .{ .shape = limit, .total = 2100 } }) |case| for ([_]Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        var shape = case.shape;
        shape.kv = storage;
        const total = case.total;
        var model: Model = try .init(gpa, random, shape, 512);
        defer model.deinit();
        const inputs = try randomBf16(gpa, random, (total + 512) * shape.dims, 1.0);
        const noise = try randomBf16(gpa, random, (total + 512) * shape.dims, 1.0);
        const serial = try runWindowsOver(gpa, shape, model.weights, model.rotary, inputs, noise, total, total, null, &.{.{ 1, 1 }}, null);
        try testing.expectEqual(total - (shape.budget + shape.ratio - 1), serial.sparse_rows);
        try testing.expect(serial.tied_rows > 0);
        // every value finite: equal bits are not NaN against NaN
        for (serial.branch) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
        for (serial.gated) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
        for ([_]Tabled{ .{}, .{ .scattered = false } }) |mode| {
            // serial steps through the table, then windows of 16 and of 512 rows, whole and with rows dropped
            const steps = try runWindowsOver(gpa, shape, model.weights, model.rotary, inputs, noise, total, total + 300, null, &.{.{ 1, 1 }}, mode);
            try serial.expectEqual(steps, shape, total);
            const mixed = try runWindowsOver(gpa, shape, model.weights, model.rotary, inputs, noise, total, total + 600, null, &.{ .{ 16, 16 }, .{ 7, 3 }, .{ 512, 300 }, .{ 1, 1 }, .{ 16, 9 }, .{ 512, 512 }, .{ 5, 5 } }, mode);
            try serial.expectEqual(mixed, shape, total);
            try testing.expectEqual(serial.sparse_rows, mixed.sparse_rows);
            // a launch bounded past the sequence, as a captured graph's is
            const bounded = try runWindowsOver(gpa, shape, model.weights, model.rotary, inputs, noise, total, total + 600, total + 600, &.{ .{ 1, 1 }, .{ 16, 16 }, .{ 512, 211 } }, mode);
            try serial.expectEqual(bounded, shape, total);
        }
        _ = arena_state.reset(.retain_capacity);
    };
}

test "windows over extents never zeroed, their rows NaN patterns until written, equal serial steps over zeroed caches in order: three storages, both kinds of list" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(101);
    const random = prng.random();
    // the small shape's selection starts at 643 and crosses the first extent's end; the last extent's rows past
    // the sequence, and the second extent's beyond a window's rows, are read by no row
    const shape_base = fx.small;
    const total = 1100;
    for ([_]Storage{ .bf16, .fp8, .fp8_nvfp4 }) |storage| {
        var shape = shape_base;
        shape.kv = storage;
        var model: Model = try .init(gpa, random, shape, 512);
        defer model.deinit();
        const inputs = try randomBf16(gpa, random, (total + 512) * shape.dims, 1.0);
        const noise = try randomBf16(gpa, random, (total + 512) * shape.dims, 1.0);
        const serial = try runWindowsOver(gpa, shape, model.weights, model.rotary, inputs, noise, total, total, null, &.{.{ 1, 1 }}, null);
        try testing.expect(serial.sparse_rows > 0);
        for (serial.branch) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
        for (serial.gated) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
        // the pool's arrays and every scratch filled with NaN patterns as they are made (`Pool.init`)
        cuda.Scratch.poisoned = true;
        defer cuda.Scratch.poisoned = false;
        for ([_]Tabled{ .{}, .{ .scattered = false } }) |mode| {
            const mixed = try runWindowsOver(gpa, shape, model.weights, model.rotary, inputs, noise, total, total + 600, null, &.{ .{ 16, 16 }, .{ 7, 3 }, .{ 512, 300 }, .{ 1, 1 }, .{ 16, 9 }, .{ 512, 512 }, .{ 5, 5 } }, mode);
            try serial.expectEqual(mixed, shape, total);
            const bounded = try runWindowsOver(gpa, shape, model.weights, model.rotary, inputs, noise, total, total + 600, total + 600, &.{ .{ 1, 1 }, .{ 16, 16 }, .{ 512, 211 } }, mode);
            try serial.expectEqual(bounded, shape, total);
        }
        cuda.Scratch.poisoned = false;
        _ = arena_state.reset(.retain_capacity);
    }
}
