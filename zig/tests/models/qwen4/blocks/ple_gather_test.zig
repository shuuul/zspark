//! `ple.zig`'s tests of the n-gram ids and the gather (spec 104 WS-06 moved them out of `ple_test.zig`): the ids
//! against TensorFold's, an NVFP4 table's rows against direct dequantization, a bf16 table's as stored.

const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const Drive = @import("../../../../src/weights/drive.zig").Drive;
const numeric = @import("../../../../src/weights/numeric.zig");
const nvfp4 = @import("../../../../src/weights/nvfp4.zig");
const layout = @import("../../../../src/models/qwen4/layout.zig");
const ple_file = @import("../../../../src/models/qwen4/blocks/ple.zig");
const Table = ple_file.Table;
const gather = ple_file.gather;
const ngramIds = ple_file.ngramIds;
const uploaded = ple_file.uploaded;
const zs_qwen4_ple_embed = cuda.kernels.zs_qwen4_ple_embed;

const testing = std.testing;
const rows_fixture = @import("ple_fixture.zig");
const fetch = rows_fixture.fetch;
const ngramOf = rows_fixture.ngramOf;

test "n-gram ids equal TensorFold's" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // The constants below are `NGram.ids` of TensorFold 0.6.5's cuda/ngram.py for the same tokens.

    // nvidia/Qwen3.8-Flash-Next-NVFP4's table: 3-grams, 8 heads each; eos resets (alone, doubled, mid-window)
    {
        const eos = 248044;
        const ngram = try ngramOf(arena, 248320, 3, 8, 20_000_000, 128, 1234, 2560, 0);
        const tokens = [_]u32{ 151644, 872, 198, 9707, eos, 248319, 0, 1, eos, eos, 5 };
        const want = [tokens.len][16]u64{
            .{ 9213577, 25503567, 59963735, 67280845, 97263283, 108604657, 126879742, 153604170, 163991649, 192963728, 210734887, 224850479, 244505954, 265606139, 292960741, 312140635 },
            .{ 6954764, 37428000, 54939688, 74004108, 95568253, 116575022, 138540160, 142009604, 165349151, 193458441, 202840010, 231812024, 252625381, 274082424, 294711253, 318927564 },
            .{ 6746895, 31942778, 54550641, 78212738, 81361934, 102413837, 123993646, 146103739, 175246506, 190873805, 206877969, 222022136, 242132242, 262461054, 282619366, 303774792 },
            .{ 14498898, 33339661, 52935564, 72566389, 92432492, 112425343, 132449694, 152547667, 161849319, 183634012, 212589728, 233690506, 255012619, 276375658, 283505765, 319224510 },
            .{ 13256830, 36293698, 52307262, 75760785, 91967280, 118328464, 138769205, 154368323, 170188935, 193004505, 205075451, 229064096, 251797085, 275608664, 283785543, 306467417 },
            .{ 11864469, 21951351, 48107725, 61973071, 83586526, 111028494, 132413971, 148010049, 176514610, 187742443, 211837434, 238017737, 255699936, 273579109, 286249325, 309928373 },
            .{ 3745007, 25177969, 59639583, 68080282, 84924354, 118004746, 128374230, 156931756, 177662890, 188011405, 216311023, 225542225, 253648089, 262985372, 299705082, 305354803 },
            .{ 9602235, 25898701, 54046987, 77454603, 83232550, 118491972, 131380991, 141899668, 166421853, 194516160, 219112511, 228756966, 256583578, 264769977, 280912202, 302222744 },
            .{ 2760535, 21194529, 54821219, 66837341, 96008662, 113339850, 120218497, 157703057, 177150885, 198316379, 215026412, 225946768, 247641779, 270395100, 284881440, 319076668 },
            .{ 9663979, 26558231, 56120240, 74755659, 80459717, 109265651, 132697467, 151022725, 170054832, 192967038, 200687722, 225763581, 259275737, 272983544, 297596484, 300986548 },
            .{ 15389869, 39778609, 55713969, 62213332, 88817728, 118483999, 133731511, 155458159, 179763390, 197956758, 205378969, 220499474, 242466248, 265658744, 293662119, 315720898 },
        };
        var got: [tokens.len * 16]u64 = undefined;
        ngramIds(ngram, eos, &.{ eos, eos }, &tokens, &got);
        for (want, 0..) |row, i| try testing.expectEqualSlices(u64, &row, got[i * 16 ..][0..16]);
        // the same rows a token at a time, the history carried as `commit` carries it
        var history = [2]u32{ eos, eos };
        for (tokens, 0..) |token, i| {
            var one: [16]u64 = undefined;
            ngramIds(ngram, eos, &history, &.{token}, &one);
            try testing.expectEqualSlices(u64, &want[i], &one);
            history = .{ history[1], token };
        }
    }
    // a small table, two heads an n-gram
    {
        const ngram = try ngramOf(arena, 300, 3, 2, 100, 8, 0, 64, 0);
        const tokens = [_]u32{ 5, 17, 9, 3, 3, 299, 9, 9, 1 };
        const want = [tokens.len][4]u64{
            .{ 52, 153, 278, 398 }, .{ 68, 173, 219, 313 }, .{ 14, 135, 208, 343 },
            .{ 8, 112, 283, 406 },  .{ 10, 171, 233, 313 }, .{ 46, 135, 266, 355 },
            .{ 52, 179, 204, 364 }, .{ 11, 146, 217, 364 }, .{ 99, 171, 228, 387 },
        };
        var got: [tokens.len * 4]u64 = undefined;
        ngramIds(ngram, 9, &.{ 9, 9 }, &tokens, &got);
        for (want, 0..) |row, i| try testing.expectEqualSlices(u64, &row, got[i * 4 ..][0..4]);
    }
    // 4-grams of the second PLE layer, from a history with an eos inside it
    {
        const ngram = try ngramOf(arena, 1000, 4, 1, 50, 4, 7, 30, 1);
        const tokens = [_]u32{ 999, 4, 6, 2, 8, 10, 12 };
        const want = [tokens.len][3]u64{
            .{ 38, 129, 204 }, .{ 30, 84, 200 }, .{ 0, 84, 198 },  .{ 31, 68, 176 },
            .{ 41, 85, 203 },  .{ 22, 77, 194 }, .{ 55, 96, 148 },
        };
        var got: [tokens.len * 3]u64 = undefined;
        ngramIds(ngram, 2, &.{ 11, 2, 13 }, &tokens, &got);
        for (want, 0..) |row, i| try testing.expectEqualSlices(u64, &row, got[i * 3 ..][0..3]);
    }
}

test "the NVFP4 gather is each code times its block scale and the table's, rounded once to bf16" {
    const gpa = testing.allocator;
    const dims = 32;
    const counts = [_]usize{ 3, 2 };
    var prng: std.Random.DefaultPrng = .init(6);
    const random = prng.random();
    var codes: [5 * dims / 2]u8 = undefined;
    random.bytes(&codes);
    var blocks: [5 * dims / 16]u8 = undefined;
    for (&blocks) |*byte| byte.* = 0x20 + random.uintLessThan(u8, 0x40);
    // row 0: codes 0 to 15 twice; its first block scale 1, its second 0.5
    for (codes[0 .. dims / 2], 0..) |*byte, i| byte.* = @intCast((2 * i % 16) | ((2 * i + 1) % 16) << 4);
    blocks[0..2].* = .{ 0x38, 0x30 };
    var shards: [counts.len]layout.Table.Shard = undefined;
    var shapes: [counts.len][2][2]u64 = undefined;
    var first: usize = 0;
    for (counts, &shards, &shapes) |count, *shard, *shape| {
        shape.* = .{ .{ count, dims / 2 }, .{ count, dims / 16 } };
        shard.* = .{
            .weight = .{ .dtype = .U8, .shape = &shape[0], .data = codes[first * dims / 2 ..][0 .. count * dims / 2] },
            .scale = .{ .dtype = .F8_E4M3, .shape = &shape[1], .data = blocks[first * dims / 16 ..][0 .. count * dims / 16] },
        };
        first += count;
    }
    const global: f32 = 0.0173;
    var table: Table = try .init(gpa, .{ .scheme = .nvfp4, .shards = &shards, .scale = .{ .dtype = .F32, .shape = &.{1}, .data = std.mem.asBytes(&global) }, .rows = 5, .dims = dims });
    defer table.deinit(gpa);
    // the table's scale is in the rows, as TensorFold's NVFP4Table gives them
    try testing.expectEqual(@as(f32, 1.0), table.scale);

    const ids = [_]u64{ 4, 0, 3, 2 };
    var got: [ids.len * dims]u16 = undefined;
    try table.gather(&ids, &got);
    for (ids, 0..) |id, i| for (0..dims) |d| {
        const at = @as(usize, @intCast(id)) * dims + d;
        const code = if (at % 2 == 0) codes[at / 2] & 0xf else codes[at / 2] >> 4;
        const product = nvfp4.e2m1[code] * numeric.f32FromE4m3(blocks[at / 16]);
        try testing.expectEqual(numeric.bf16FromF32(product * global), got[i * dims + d]);
    };
    try testing.expectError(error.RowOutOfRange, table.gather(&.{5}, got[0..dims]));

    // row 0 under a table scale of 1: the E2M1 values, then halved
    var plain: Table = try .init(gpa, .{ .scheme = .nvfp4, .shards = &shards, .scale = null, .rows = 5, .dims = dims });
    defer plain.deinit(gpa);
    try plain.gather(&.{0}, got[0..dims]);
    for (0..dims) |d| try testing.expectEqual(numeric.bf16FromF32(nvfp4.e2m1[d % 16] * @as(f32, if (d < 16) 1.0 else 0.5)), got[d]);
    try testing.expectEqualSlices(u16, &.{ 0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0 }, got[0..8]);
    try testing.expectEqualSlices(u16, &.{ 0x8000, 0xbe80, 0xbf00, 0xbf40, 0xbf80, 0xbfc0, 0xc000, 0xc040 }, got[24..32]);
}

test "a bf16 table's rows are gathered as stored, its scale left to the embedding" {
    const gpa = testing.allocator;
    const dims = 4;
    var values: [6 * dims]u16 = undefined;
    for (&values, 0..) |*bits, i| bits.* = numeric.bf16FromF32(@as(f32, @floatFromInt(i)) - 7.5);
    const bytes = std.mem.sliceAsBytes(&values);
    const scale: f32 = 0.37;
    var table: Table = try .init(gpa, .{
        .scheme = .bf16,
        .shards = &.{
            .{ .weight = .{ .dtype = .BF16, .shape = &.{ 2, dims }, .data = bytes[0 .. 2 * dims * 2] } },
            .{ .weight = .{ .dtype = .BF16, .shape = &.{ 4, dims }, .data = bytes[2 * dims * 2 ..] } },
        },
        .scale = .{ .dtype = .F32, .shape = &.{1}, .data = std.mem.asBytes(&scale) },
        .rows = 6,
        .dims = dims,
    });
    defer table.deinit(gpa);
    try testing.expectEqual(scale, table.scale);
    const ids = [_]u64{ 5, 1, 2 };
    var got: [ids.len * dims]u16 = undefined;
    try table.gather(&ids, &got);
    for (ids, 0..) |id, i| try testing.expectEqualSlices(u16, values[@as(usize, @intCast(id)) * dims ..][0..dims], got[i * dims ..][0..dims]);

    // a table whose drive is kept awake tells it of every gather, a copy of the table too (a session's, the
    // pool's and the thread ahead each hold one), and gathers the same bytes
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "rows", .data = &@as([Drive.block]u8, @splat(0)) });
    const fd: isize = @bitCast(std.os.linux.openat(tmp.dir.handle, "rows", .{ .ACCMODE = .RDONLY, .DIRECT = true, .CLOEXEC = true }, 0));
    defer if (fd >= 0) {
        _ = std.os.linux.close(@intCast(fd));
    };
    if (Drive.open(gpa, @intCast(fd), Drive.block, .{})) |drive| {
        defer drive.close(gpa);
        var kept = table;
        kept.drive = drive;
        const copy = kept;
        var same: [ids.len * dims]u16 = undefined;
        try kept.gather(&ids, &same);
        try copy.gather(ids[0..1], same[0..dims]);
        try testing.expectEqual(@as(u32, 2), drive.touched.load(.monotonic));
        try testing.expectEqualSlices(u16, &got, &same);
        try testing.expectEqual(null, table.drive);
    }

    // the embedding of those rows: each value times the scale, rounded once
    var staged = try uploaded(.bf16, &.{ ids.len, dims }, std.mem.sliceAsBytes(&got));
    defer staged.deinit();
    var emb: cuda.Buffer = try .init(.bf16, &.{ 1, ids.len * dims });
    defer emb.deinit();
    try cuda.launched(zs_qwen4_ple_embed(&staged.view, &emb.view, ids.len, table.scale));
    const scaled = try fetch(u16, gpa, emb.view);
    defer gpa.free(scaled);
    for (got, scaled) |bits, out| try testing.expectEqual(numeric.bf16FromF32(numeric.f32FromBf16(bits) * scale), out);
}

test "a wide gather cut into parts on threads gives each row's bytes, and refuses a row out of range" {
    const gpa = testing.allocator;
    const dims = 32;
    const rows = 97;
    var prng: std.Random.DefaultPrng = .init(149);
    const random = prng.random();
    var codes: [rows * dims / 2]u8 = undefined;
    random.bytes(&codes);
    var blocks: [rows * dims / 16]u8 = undefined;
    for (&blocks) |*byte| byte.* = 0x20 + random.uintLessThan(u8, 0x40);
    const shape = [2][2]u64{ .{ rows, dims / 2 }, .{ rows, dims / 16 } };
    const shards = [_]layout.Table.Shard{.{
        .weight = .{ .dtype = .U8, .shape = &shape[0], .data = &codes },
        .scale = .{ .dtype = .F8_E4M3, .shape = &shape[1], .data = &blocks },
    }};
    var table: Table = try .init(gpa, .{ .scheme = .nvfp4, .shards = &shards, .scale = null, .rows = rows, .dims = dims });
    defer table.deinit(gpa);
    // wide enough for every part and a last part shorter than the others
    const ids = try gpa.alloc(u64, Table.part_ids * Table.most_parts + 333);
    defer gpa.free(ids);
    for (ids) |*id| id.* = random.uintLessThan(u64, rows);
    const whole = try gpa.alloc(u16, ids.len * dims);
    defer gpa.free(whole);
    try table.gather(ids, whole);
    // a row at a time, on this thread
    var one: [dims]u16 = undefined;
    for (ids, 0..) |id, i| {
        try table.gather(&.{id}, &one);
        try testing.expectEqualSlices(u16, &one, whole[i * dims ..][0..dims]);
    }
    // a row out of range in a part on another thread fails the gather
    ids[ids.len - 1] = rows;
    try testing.expectError(error.RowOutOfRange, table.gather(ids, whole));
}

test "the checkpoint's table read ahead through its files gathers the bytes advised through the mapping" {
    const fixtures = @import("../testing.zig");
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var opened: fixtures.checkpoints.Held = undefined;
    _, const found = try opened.open(gpa, arena_state.allocator(), path);
    defer opened.close();
    const branch = for (found.layers) |maybe| {
        if (maybe) |layer| if (layer.ple) |held| break held;
    } else return error.SkipZigTest;
    var mapped: Table = try .init(gpa, branch.table);
    defer mapped.deinit(gpa);
    var filed: Table = try .init(gpa, branch.table);
    defer filed.deinit(gpa);
    try filed.readFrom(gpa, &opened.checkpoint);
    try testing.expectEqual(mapped.shards.len, filed.files.len);
    for (filed.files) |file| try testing.expect(file.fd >= 0);
    // a prompt's window of tokens, its rows over every part; the same rows a part at a time through the mapping
    const tokens = 2048;
    const heads = branch.ngram.head_sizes.len;
    const ids = try gpa.alloc(u64, tokens * heads);
    defer gpa.free(ids);
    const toks = try gpa.alloc(u32, tokens);
    defer gpa.free(toks);
    var prng: std.Random.DefaultPrng = .init(@intCast(std.Io.Clock.awake.now(testing.io).toNanoseconds()));
    const out = try gpa.alloc(u16, ids.len * filed.stride());
    defer gpa.free(out);
    const want = try gpa.alloc(u16, ids.len * filed.stride());
    defer gpa.free(want);
    for (toks) |*token| token.* = prng.random().uintLessThan(u32, 150_000);
    ngramIds(branch.ngram, 248044, &.{ 248044, 248044 }, toks, ids);
    try filed.gather(ids, out);
    var at: usize = 0;
    while (at < ids.len) : (at += Table.part_ids) {
        const to = @min(at + Table.part_ids, ids.len);
        try mapped.gather(ids[at..to], want[at * filed.stride() .. to * filed.stride()]);
    }
    try testing.expectEqualSlices(u16, want, out);
}
