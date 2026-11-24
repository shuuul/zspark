//! `index.cu` against `index.zig`: bit for bit (the keys, the pooled keys' codes and scales, the queries' codes
//! and folded weights) but the scores, whose dot is the FP8 tensor cores'. A window, the serial steps across a
//! pool's end and an extent's, and a row alone. The table is swapped, so a key written as if the extents were
//! contiguous would not match.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const index = @import("../../../../src/models/glm5/layers/index.zig");
const launch = @import("../../../../src/models/glm5/layers/index_launch.zig");
const shared = @import("../testing.zig");

const dim = 16;
const heads = 2;
const kept = 2;
const tokens_n = 520;
const extent = launch.extent;
const extents = (tokens_n + extent - 1) / extent;
const pools = tokens_n / index.pool;
const width = index.listWidth(kept, index.pool);

fn logical(storage: []const u16, table: []const i32, pos: usize) []const u16 {
    const physical: usize = @intCast(table[pos / extent]);
    const row = physical * extent + pos % extent;
    return storage[row * dim ..][0..dim];
}

fn setPos(buffer: cuda.Buffer, pos: usize) !void {
    const word: i32 = @intCast(pos);
    try buffer.upload(std.mem.asBytes(&word));
}

test "the indexer's window is its serial steps, and a row alone, through a swapped extent table" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(46);
    const random = prng.random();

    const k = try arena.alloc(u16, tokens_n * dim);
    const gate = try arena.alloc(f32, tokens_n * dim);
    const ln_w = try arena.alloc(u16, dim);
    const ln_b = try arena.alloc(u16, dim);
    const ape = try arena.alloc(u16, index.pool * dim);
    const qi = try arena.alloc(u16, tokens_n * heads * dim);
    const weights = try arena.alloc(f32, tokens_n * heads);
    shared.randomBf16(random, k, 0.4);
    shared.randomF32(random, gate, 0.3);
    shared.randomBf16(random, ln_b, 0.05);
    shared.randomBf16(random, ape, 0.2);
    shared.randomBf16(random, qi, 0.4);
    shared.randomF32(random, weights, 0.5);
    for (ln_w) |*one| one.* = numeric.bf16FromF32(1.0 + random.floatNorm(f32) * 0.05);
    const table = [_]i32{ 1, 0 };
    const host_key = try arena.alloc(u16, tokens_n * dim);
    const host_gate = try arena.alloc(u16, tokens_n * dim);
    for (0..tokens_n) |pos| {
        index.normKey(k[pos * dim ..][0..dim], ln_w, ln_b, host_key[pos * dim ..][0..dim]);
        index.storeGate(gate[pos * dim ..][0..dim], host_gate[pos * dim ..][0..dim]);
    }
    const host_pool = try arena.alloc(u16, pools * dim);
    for (0..pools) |p| {
        var rows: [index.pool][]const u16 = undefined;
        var gates: [index.pool][]const u16 = undefined;
        for (0..index.pool) |t| {
            const pos = p * index.pool + t;
            rows[t] = host_key[pos * dim ..][0..dim];
            gates[t] = host_gate[pos * dim ..][0..dim];
        }
        index.poolRow(rows, gates, ape, host_pool[p * dim ..][0..dim]);
    }
    const host_codes = try arena.alloc(u8, pools * dim);
    const host_scales = try arena.alloc(f32, pools);
    for (0..pools) |p| host_scales[p] = index.encodeKey(host_pool[p * dim ..][0..dim], host_codes[p * dim ..][0..dim]);
    const host_q8 = try arena.alloc(u8, tokens_n * heads * dim);
    const host_wq = try arena.alloc(f32, tokens_n * heads);
    for (0..tokens_n) |at| index.encodeQuery(qi[at * heads * dim ..][0 .. heads * dim], weights[at * heads ..][0..heads], host_q8[at * heads * dim ..][0 .. heads * dim], host_wq[at * heads ..][0..heads]);

    var k_dev = try shared.device(u16, .bf16, &.{ tokens_n, dim }, k);
    defer k_dev.deinit();
    var gate_dev = try shared.device(f32, .f32, &.{ tokens_n, dim }, gate);
    defer gate_dev.deinit();
    var ln_w_dev = try shared.device(u16, .bf16, &.{dim}, ln_w);
    defer ln_w_dev.deinit();
    var ln_b_dev = try shared.device(u16, .bf16, &.{dim}, ln_b);
    defer ln_b_dev.deinit();
    var ape_dev = try shared.device(u16, .bf16, &.{ index.pool, dim }, ape);
    defer ape_dev.deinit();
    var qi_dev = try shared.device(u16, .bf16, &.{ tokens_n, heads, dim }, qi);
    defer qi_dev.deinit();
    var w_dev = try shared.device(f32, .f32, &.{ tokens_n, heads }, weights);
    defer w_dev.deinit();
    var table_dev = try shared.device(i32, .i32, &.{extents}, &table);
    defer table_dev.deinit();
    var pos_dev = try cuda.Buffer.init(.i32, &.{1});
    defer pos_dev.deinit();

    const rows = extents * extent;
    var keys = try cuda.Buffer.init(.bf16, &.{ rows, dim });
    defer keys.deinit();
    var gates = try cuda.Buffer.init(.bf16, &.{ rows, dim });
    defer gates.deinit();
    var keys_serial = try cuda.Buffer.init(.bf16, &.{ rows, dim });
    defer keys_serial.deinit();
    var gates_serial = try cuda.Buffer.init(.bf16, &.{ rows, dim });
    defer gates_serial.deinit();
    var alone = try cuda.Buffer.init(.bf16, &.{ rows, dim });
    defer alone.deinit();
    try keys.zero();
    try gates.zero();
    try keys_serial.zero();
    try gates_serial.zero();
    try alone.zero();

    try setPos(pos_dev, 0);
    try launch.write(&k_dev.view, &gate_dev.view, &ln_w_dev.view, &ln_b_dev.view, &keys.view, &gates.view, &table_dev.view, &pos_dev.view, tokens_n, tokens_n);
    for (0..tokens_n) |pos| {
        var k_row = k_dev.span(pos, 1);
        var g_row = gate_dev.span(pos, 1);
        try setPos(pos_dev, pos);
        try launch.write(&k_row, &g_row, &ln_w_dev.view, &ln_b_dev.view, &keys_serial.view, &gates_serial.view, &table_dev.view, &pos_dev.view, 1, pos + 1);
    }
    const got_key = try shared.fetch(u16, arena, keys);
    const got_gate = try shared.fetch(u16, arena, gates);
    const serial_key = try shared.fetch(u16, arena, keys_serial);
    try testing.expectEqualSlices(u16, got_key, serial_key);
    for (0..tokens_n) |pos| {
        try testing.expectEqualSlices(u16, host_key[pos * dim ..][0..dim], logical(got_key, &table, pos));
        try testing.expectEqualSlices(u16, host_gate[pos * dim ..][0..dim], logical(got_gate, &table, pos));
    }
    // token 0 lives in physical extent 1, token 512 in physical extent 0
    try testing.expectEqualSlices(u16, host_key[0..dim], got_key[table[0] * extent * dim ..][0..dim]);
    try testing.expectEqualSlices(u16, host_key[512 * dim ..][0..dim], got_key[0..dim]);

    const alone_at = 510;
    var k_alone = k_dev.span(alone_at, 1);
    var g_alone = gate_dev.span(alone_at, 1);
    try setPos(pos_dev, alone_at);
    try launch.write(&k_alone, &g_alone, &ln_w_dev.view, &ln_b_dev.view, &alone.view, &gates_serial.view, &table_dev.view, &pos_dev.view, 1, alone_at + 1);
    const got_alone = try shared.fetch(u16, arena, alone);
    try testing.expectEqualSlices(u16, logical(got_key, &table, alone_at), logical(got_alone, &table, alone_at));

    var codes = try cuda.Buffer.init(.u8, &.{ pools, dim });
    defer codes.deinit();
    var scales = try cuda.Buffer.init(.f32, &.{pools});
    defer scales.deinit();
    var codes_serial = try cuda.Buffer.init(.u8, &.{ pools, dim });
    defer codes_serial.deinit();
    var scales_serial = try cuda.Buffer.init(.f32, &.{pools});
    defer scales_serial.deinit();
    inline for (.{ &codes, &scales, &codes_serial, &scales_serial }) |buffer| try buffer.zero();
    try setPos(pos_dev, 0);
    try launch.pool(&keys.view, &gates.view, &table_dev.view, &ape_dev.view, &codes.view, &scales.view, &pos_dev.view, tokens_n, tokens_n);
    for (0..tokens_n) |pos| {
        try setPos(pos_dev, pos);
        try launch.pool(&keys.view, &gates.view, &table_dev.view, &ape_dev.view, &codes_serial.view, &scales_serial.view, &pos_dev.view, 1, pos + 1);
    }
    try testing.expectEqualSlices(u8, host_codes, try shared.fetch(u8, arena, codes));
    try testing.expectEqualSlices(f32, host_scales, try shared.fetch(f32, arena, scales));
    try testing.expectEqualSlices(u8, host_codes, try shared.fetch(u8, arena, codes_serial));
    try testing.expectEqualSlices(f32, host_scales, try shared.fetch(f32, arena, scales_serial));

    // the queries coded over every row at once and a row alone
    var q8 = try cuda.Buffer.init(.u8, &.{ tokens_n, heads, dim });
    defer q8.deinit();
    var wq = try cuda.Buffer.init(.f32, &.{ tokens_n, heads });
    defer wq.deinit();
    try launch.query(&qi_dev.view, &w_dev.view, &q8.view, &wq.view);
    try testing.expectEqualSlices(u8, host_q8, try shared.fetch(u8, arena, q8));
    try testing.expectEqualSlices(f32, host_wq, try shared.fetch(f32, arena, wq));
    var q8_one = try cuda.Buffer.init(.u8, &.{ 1, heads, dim });
    defer q8_one.deinit();
    var wq_one = try cuda.Buffer.init(.f32, &.{ 1, heads });
    defer wq_one.deinit();
    var qi_alone = qi_dev.span(alone_at, 1);
    var w_alone = w_dev.span(alone_at, 1);
    try launch.query(&qi_alone, &w_alone, &q8_one.view, &wq_one.view);
    try testing.expectEqualSlices(u8, host_q8[alone_at * heads * dim ..][0 .. heads * dim], try shared.fetch(u8, arena, q8_one));
    try testing.expectEqualSlices(f32, host_wq[alone_at * heads ..][0..heads], try shared.fetch(f32, arena, wq_one));

    const host: Host = .{ .q8 = host_q8, .wq = host_wq, .codes = host_codes, .scales = host_scales };
    const dev: Dev = .{ .q8 = q8, .wq = wq, .codes = codes, .scales = scales, .pos = pos_dev };
    try expectSelect(arena, host, dev, 4, 12);
    try expectSelect(arena, host, dev, 508, 8);
}

test "a window's rows split in blocks pick the full window's lists" {
    // the two ranks' blocks of `dsa.indexRowsSplit` (spec 152 WS-10): a block's rows scored and selected
    // against the same pooled keys with its position base, then the lists beside each other. Every rank's
    // kernels and inputs are the full indexer's, so the gathered lists equal it row for row.
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(7);
    const random = prng.random();

    const tokens_split = 96;
    const pools_split = tokens_split / index.pool;
    var qi = try arena.alloc(u16, tokens_split * heads * dim);
    var weights = try arena.alloc(f32, tokens_split * heads);
    var pooled = try arena.alloc(u16, pools_split * dim);
    shared.randomBf16(random, qi, 0.4);
    shared.randomF32(random, weights, 0.5);
    shared.randomBf16(random, pooled, 0.4);
    const codes_host = try arena.alloc(u8, pools_split * dim);
    const scales_host = try arena.alloc(f32, pools_split);
    for (0..pools_split) |p| scales_host[p] = index.encodeKey(pooled[p * dim ..][0..dim], codes_host[p * dim ..][0..dim]);
    const q8_host = try arena.alloc(u8, tokens_split * heads * dim);
    const wq_host = try arena.alloc(f32, tokens_split * heads);
    for (0..tokens_split) |at| index.encodeQuery(qi[at * heads * dim ..][0 .. heads * dim], weights[at * heads ..][0..heads], q8_host[at * heads * dim ..][0 .. heads * dim], wq_host[at * heads ..][0..heads]);

    var qi_dev = try shared.device(u16, .bf16, &.{ tokens_split, heads, dim }, qi);
    defer qi_dev.deinit();
    var w_dev = try shared.device(f32, .f32, &.{ tokens_split, heads }, weights);
    defer w_dev.deinit();
    var codes = try shared.device(u8, .u8, &.{ pools_split, dim }, codes_host);
    defer codes.deinit();
    var scales = try shared.device(f32, .f32, &.{pools_split}, scales_host);
    defer scales.deinit();
    var q8 = try cuda.Buffer.init(.u8, &.{ tokens_split, heads, dim });
    defer q8.deinit();
    var wq = try cuda.Buffer.init(.f32, &.{ tokens_split, heads });
    defer wq.deinit();
    try launch.query(&qi_dev.view, &w_dev.view, &q8.view, &wq.view);
    var pos_dev = try cuda.Buffer.init(.i32, &.{1});
    defer pos_dev.deinit();

    // the window of `rows` at `pos`: the full indexer's lists
    const pos = 8;
    const rows = 40;
    var scores = try cuda.Buffer.init(.f32, &.{ rows, pools_split });
    defer scores.deinit();
    var lists = try cuda.Buffer.init(.i32, &.{ rows, width });
    defer lists.deinit();
    var counts = try cuda.Buffer.init(.i32, &.{rows});
    defer counts.deinit();
    var q8_rows = q8.span(pos, rows);
    var wq_rows = wq.span(pos, rows);
    try setPos(pos_dev, pos);
    try launch.score(&q8_rows, &wq_rows, &codes.view, &scales.view, &pos_dev.view, &scores.view, pos + rows);
    try launch.select(&scores.view, &pos_dev.view, &lists.view, &counts.view, kept, pos + rows);

    // rank r's block of `per` rows: scored and selected alone, its position base its own word
    const world = 2;
    const per = (rows + world - 1) / world;
    var split_lists = try cuda.Buffer.init(.i32, &.{ rows, width });
    defer split_lists.deinit();
    var split_counts = try cuda.Buffer.init(.i32, &.{rows});
    defer split_counts.deinit();
    for (0..world) |rank| {
        const lo = @min(rows, rank * per);
        const hi = @min(rows, lo + per);
        var q8_block = q8.span(pos + lo, hi - lo);
        var wq_block = wq.span(pos + lo, hi - lo);
        var sc = try cuda.Buffer.init(.f32, &.{ hi - lo, pools_split });
        defer sc.deinit();
        var tl = split_lists.span(lo, hi - lo);
        var ct = split_counts.span(lo, hi - lo);
        try setPos(pos_dev, pos + lo);
        try launch.score(&q8_block, &wq_block, &codes.view, &scales.view, &pos_dev.view, &sc.view, pos + rows);
        try launch.select(&sc.view, &pos_dev.view, &tl, &ct, kept, pos + rows);
    }
    const got = try shared.fetch(i32, arena, lists);
    const got_counts = try shared.fetch(i32, arena, counts);
    try testing.expectEqualSlices(i32, got, try shared.fetch(i32, arena, split_lists));
    try testing.expectEqualSlices(i32, got_counts, try shared.fetch(i32, arena, split_counts));
}

const Host = struct { q8: []const u8, wq: []const f32, codes: []const u8, scales: []const f32 };
const Dev = struct { q8: cuda.Buffer, wq: cuda.Buffer, codes: cuda.Buffer, scales: cuda.Buffer, pos: cuda.Buffer };

fn scoreRows(dev: Dev, pos: usize, rows: usize, scores: cuda.Buffer) !void {
    var q8_rows = dev.q8.span(pos, rows);
    var wq_rows = dev.wq.span(pos, rows);
    try setPos(dev.pos, pos);
    try launch.score(&q8_rows, &wq_rows, &dev.codes.view, &dev.scales.view, &dev.pos.view, &scores.view, pos + rows);
}

fn expectSelect(arena: std.mem.Allocator, host: Host, dev: Dev, pos: usize, rows: usize) !void {
    var scores = try cuda.Buffer.init(.f32, &.{ rows, pools });
    defer scores.deinit();
    var lists = try cuda.Buffer.init(.i32, &.{ rows, width });
    defer lists.deinit();
    var counts = try cuda.Buffer.init(.i32, &.{rows});
    defer counts.deinit();
    try scoreRows(dev, pos, rows, scores);
    try launch.select(&scores.view, &dev.pos.view, &lists.view, &counts.view, kept, pos + rows);
    const got_scores = try shared.fetch(f32, arena, scores);
    const got_lists = try shared.fetch(i32, arena, lists);
    const got_counts = try shared.fetch(i32, arena, counts);

    var one_scores = try cuda.Buffer.init(.f32, &.{ 1, pools });
    defer one_scores.deinit();
    var one_lists = try cuda.Buffer.init(.i32, &.{ 1, width });
    defer one_lists.deinit();
    var one_counts = try cuda.Buffer.init(.i32, &.{1});
    defer one_counts.deinit();
    var serial_scores = try arena.alloc(f32, rows * pools);
    var serial_lists = try arena.alloc(i32, rows * width);
    var serial_counts = try arena.alloc(i32, rows);

    for (0..rows) |r| {
        const at = pos + r;
        const complete = index.completePools(at, index.pool);
        // The dot is the tensor cores' (32 products a step), the host model's one product at a time: a score is
        // held to the products' mass, and the list is the host's selection over the device's scores.
        const host_scores = got_scores[r * pools ..][0..complete];
        const q8 = host.q8[at * heads * dim ..][0 .. heads * dim];
        const wq = host.wq[at * heads ..][0..heads];
        for (0..complete) |p| {
            const key = host.codes[p * dim ..][0..dim];
            const host_score = index.scorePool(q8, wq, key, host.scales[p]);
            var mass: f32 = 0;
            for (0..heads) |h| {
                var products: f32 = 0;
                for (q8[h * dim ..][0..dim], key) |q, k| products += @abs(numeric.f32FromE4m3(q) * numeric.f32FromE4m3(k));
                mass += @abs(wq[h]) * products;
            }
            try testing.expect(@abs(host_score - host_scores[p]) <= 4e-6 * mass * host.scales[p]);
        }
        if (complete <= kept) {
            try testing.expectEqual(@as(i32, 0), got_counts[r]);
        } else {
            const selected = try arena.alloc(u32, kept);
            try index.selectPools(arena, host_scores, kept, selected);
            var host_tokens: [width]i32 = undefined;
            const count = index.selectTokens(selected, at, index.pool, &host_tokens);
            try testing.expectEqual(@as(i32, @intCast(count)), got_counts[r]);
            try testing.expectEqualSlices(i32, &host_tokens, got_lists[r * width ..][0..width]);
        }

        // the row alone: its own serial step
        try scoreRows(dev, at, 1, one_scores);
        try launch.select(&one_scores.view, &dev.pos.view, &one_lists.view, &one_counts.view, kept, at + 1);
        try one_scores.download(std.mem.sliceAsBytes(serial_scores[r * pools ..][0..pools]));
        // a window's scores are its serial steps' to the bit
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(host_scores), std.mem.sliceAsBytes(serial_scores[r * pools ..][0..complete]));
        try one_lists.download(std.mem.sliceAsBytes(serial_lists[r * width ..][0..width]));
        try one_counts.download(std.mem.sliceAsBytes(serial_counts[r..][0..1]));
    }
    try testing.expectEqualSlices(i32, got_lists, serial_lists);
    try testing.expectEqualSlices(i32, got_counts, serial_counts);
}

test "the indexer's tie at the cut keeps the lower pool" {
    const nb = 4;
    const scores = [_]f32{ 9, 1, 1, 0.5, -0.0, 0.0, -1, -2 };
    // two rows: the tie at the cut, and -0 counted as +0
    var scores_dev = try cuda.Buffer.init(.f32, &.{ 2, nb });
    defer scores_dev.deinit();
    try scores_dev.upload(std.mem.sliceAsBytes(&scores));
    var pos_dev = try cuda.Buffer.init(.i32, &.{1});
    defer pos_dev.deinit();
    var lists = try cuda.Buffer.init(.i32, &.{ 2, width });
    defer lists.deinit();
    var counts = try cuda.Buffer.init(.i32, &.{2});
    defer counts.deinit();

    // row 0 stands at 15: four complete pools. row 1 is a second launch at 7, two complete pools, kept 1.
    try setPos(pos_dev, 15);
    var row0 = scores_dev.span(0, 1);
    var lists0 = lists.span(0, 1);
    try launch.select(&row0, &pos_dev.view, &lists0, &counts.view, kept, 16);

    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var selected: [kept]u32 = undefined;
    try index.selectPools(arena, scores[0..nb], kept, &selected);
    var host_tokens: [width]i32 = undefined;
    const count = index.selectTokens(&selected, 15, index.pool, &host_tokens);
    const got = try shared.fetch(i32, arena, lists);
    const got_counts = try shared.fetch(i32, arena, counts);
    try testing.expectEqual(@as(i32, @intCast(count)), got_counts[0]);
    try testing.expectEqualSlices(i32, &host_tokens, got[0..width]);
    try testing.expectEqualSlices(u32, &.{ 0, 1 }, &selected);

    const kept_one = 1;
    const width_one = index.listWidth(kept_one, index.pool);
    var lists_one = try cuda.Buffer.init(.i32, &.{ 1, width_one });
    defer lists_one.deinit();
    var count_one = try cuda.Buffer.init(.i32, &.{1});
    defer count_one.deinit();
    var row1 = scores_dev.span(1, 1);
    try setPos(pos_dev, 7);
    try launch.select(&row1, &pos_dev.view, &lists_one.view, &count_one.view, kept_one, 8);
    var selected_one: [1]u32 = undefined;
    try index.selectPools(arena, scores[nb..][0..2], kept_one, &selected_one);
    const host_one = try arena.alloc(i32, width_one);
    const count_b = index.selectTokens(&selected_one, 7, index.pool, host_one);
    const got_one = try shared.fetch(i32, arena, lists_one);
    const got_one_count = try shared.fetch(i32, arena, count_one);
    try testing.expectEqual(@as(i32, @intCast(count_b)), got_one_count[0]);
    try testing.expectEqualSlices(i32, host_one, got_one);
    try testing.expectEqual(@as(u32, 0), selected_one[0]);
}
