//! `safetensors.zig`'s tests.

const std = @import("std");
const safetensors = @import("../../src/weights/safetensors.zig");
const Io = std.Io;
const Checkpoint = safetensors.Checkpoint;
const DType = safetensors.DType;
const File = safetensors.File;

fn writeFixture(io: Io, dir: Io.Dir, name: []const u8, header: []const u8, data: []const u8) !void {
    var bytes: [512]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memcpy(bytes[8 + header.len ..][0..data.len], data);
    try dir.writeFile(io, .{ .sub_path = name, .data = bytes[0 .. 8 + header.len + data.len] });
}

test "reads tensors by name" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header =
        \\{"__metadata__":{"format":"pt"},"a.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
        \\"b.codes":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[16,19]}}
    ;
    const values = [4]f32{ 1.0, -2.0, 3.5, 0.25 };
    var data: [19]u8 = undefined;
    @memcpy(data[0..16], std.mem.sliceAsBytes(&values));
    @memcpy(data[16..], &[_]u8{ 0x38, 0x7e, 0xbc });
    try writeFixture(io, tmp.dir, "t.safetensors", header, &data);

    var file: File = try .open(std.testing.allocator, io, tmp.dir, "t.safetensors");
    defer file.close();
    try std.testing.expectEqual(@as(usize, 2), file.tensors.count());
    const a = file.get("a.weight").?;
    try std.testing.expectEqual(DType.F32, a.dtype);
    try std.testing.expectEqualSlices(u64, &.{ 2, 2 }, a.shape);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), a.data);
    const b = file.get("b.codes").?;
    try std.testing.expectEqualSlices(u8, &.{ 0x38, 0x7e, 0xbc }, b.data);
    try std.testing.expect(file.get("missing") == null);
}

test "refuses a range past the file" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header =
        \\{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}
    ;
    try writeFixture(io, tmp.dir, "short.safetensors", header, &@as([8]u8, @splat(0)));
    try std.testing.expectError(error.InvalidSafetensors, File.open(std.testing.allocator, io, tmp.dir, "short.safetensors"));
}

test "a directory is refused as a file, by name" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "reference", .default_dir);
    try std.testing.expectError(error.IsDir, File.open(std.testing.allocator, io, tmp.dir, "reference"));
}

test "a checkpoint gathers its shards" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeFixture(io, tmp.dir, "model-00001-of-00002.safetensors",
        \\{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}
    , &.{ 1, 2 });
    try writeFixture(io, tmp.dir, "model-00002-of-00002.safetensors",
        \\{"b":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}
    , &.{ 3, 4, 5 });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });

    var checkpoint: Checkpoint = try .open(std.testing.allocator, io, tmp.dir, ".");
    defer checkpoint.close();
    try std.testing.expectEqual(@as(usize, 2), checkpoint.shards.len);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2 }, checkpoint.get("a").?.data);
    try std.testing.expectEqualSlices(u8, &.{ 3, 4, 5 }, checkpoint.get("b").?.data);
}

test "shards in a subdirectory are found by the index's names, those still downloading skipped" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "tensors", .default_dir);
    try writeFixture(io, tmp.dir, "tensors/model-00001-of-00003.safetensors",
        \\{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}
    , &.{ 1, 2 });
    try writeFixture(io, tmp.dir, "tensors/model-00002-of-00003.safetensors",
        \\{"b":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}
    , &.{ 3, 4, 5 });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"metadata":{"total_size":5},"weight_map":{"a":"tensors/model-00001-of-00003.safetensors",
        \\"b":"tensors/model-00002-of-00003.safetensors","a2":"tensors/model-00001-of-00003.safetensors",
        \\"c":"tensors/model-00003-of-00003.safetensors"}}
    });

    var checkpoint: Checkpoint = try .open(std.testing.allocator, io, tmp.dir, ".");
    defer checkpoint.close();
    try std.testing.expectEqual(@as(usize, 2), checkpoint.shards.len);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2 }, checkpoint.get("a").?.data);
    try std.testing.expectEqualSlices(u8, &.{ 3, 4, 5 }, checkpoint.get("b").?.data);

    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const names = try Checkpoint.indexedShards(arena.allocator(), io, tmp.dir);
    try std.testing.expectEqual(@as(usize, 3), names.len);
    try std.testing.expectEqualStrings("tensors/model-00003-of-00003.safetensors", names[2]);
}

test "a subdirectory's shards the index names join the directory's own" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "tensors", .default_dir);
    try writeFixture(io, tmp.dir, "model-00001-of-00001.safetensors",
        \\{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}
    , &.{ 1, 2 });
    try writeFixture(io, tmp.dir, "tensors/native-QB.safetensors",
        \\{"b":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}
    , &.{ 3, 4, 5 });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"metadata":{"total_size":5},"weight_map":{"a":"model-00001-of-00001.safetensors",
        \\"b":"tensors/native-QB.safetensors"}}
    });

    var checkpoint: Checkpoint = try .open(std.testing.allocator, io, tmp.dir, ".");
    defer checkpoint.close();
    try std.testing.expectEqual(@as(usize, 2), checkpoint.shards.len);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2 }, checkpoint.get("a").?.data);
    try std.testing.expectEqualSlices(u8, &.{ 3, 4, 5 }, checkpoint.get("b").?.data);
}

test "staged ranges read as the file does, to the file's last page; the next read reuses the buffer" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const page = std.heap.pageSize();

    // Three tensors over 40 pages and a bit: the first two share a page, the last ends the file mid-page.
    const sizes = [_]usize{ 3 * page + 100, 20 * page - 100 + 7, 17 * page + 513 };
    const header_text = try std.fmt.allocPrint(gpa,
        \\{{"a":{{"dtype":"U8","shape":[{d}],"data_offsets":[0,{d}]}},"b":{{"dtype":"U8","shape":[{d}],"data_offsets":[{d},{d}]}},"c":{{"dtype":"U8","shape":[{d}],"data_offsets":[{d},{d}]}}}}
    , .{ sizes[0], sizes[0], sizes[1], sizes[0], sizes[0] + sizes[1], sizes[2], sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2] });
    defer gpa.free(header_text);
    const total = sizes[0] + sizes[1] + sizes[2];
    const bytes = try gpa.alloc(u8, 8 + header_text.len + total);
    defer gpa.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header_text.len, .little);
    @memcpy(bytes[8..][0..header_text.len], header_text);
    var prng: std.Random.DefaultPrng = .init(41);
    prng.random().bytes(bytes[8 + header_text.len ..]);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });

    var checkpoint: Checkpoint = try .open(gpa, io, tmp.dir, ".");
    defer checkpoint.close();
    const data = bytes[8 + header_text.len ..];
    const a = checkpoint.get("a").?.data;
    const b = checkpoint.get("b").?.data;
    const c = checkpoint.get("c").?.data;
    // a tensor, and a part of one, are in their shard's mapping; memory that is no file's is in none
    try std.testing.expectEqual(@as(?*const File, &checkpoint.shards[0]), checkpoint.fileOf(c));
    try std.testing.expectEqual(@as(?*const File, &checkpoint.shards[0]), checkpoint.fileOf(b[5..9]));
    try std.testing.expectEqual(@as(?*const File, null), checkpoint.fileOf(bytes[0..16]));
    // Without direct reads (a tmpfs) nothing is staged, and the mapping still reads right.
    const direct = checkpoint.shards[0].direct >= 0;

    var stage: safetensors.Stage = .{};
    defer stage.deinit(gpa);
    try stage.read(gpa, &checkpoint, &.{ c, a[0 .. page + 3] });
    if (direct) {
        try std.testing.expectEqual(@as(usize, 2), stage.spans.items.len);
        try std.testing.expectEqualSlices(u8, data[0 .. page + 3], stage.moved(a[0 .. page + 3]).?);
        try std.testing.expectEqualSlices(u8, data[sizes[0] + sizes[1] ..], stage.moved(c).?);
        // a part of a staged range, and none of what was not staged
        try std.testing.expectEqualSlices(u8, data[sizes[0] + sizes[1] + 7 ..][0..9], stage.moved(c[7..16]).?);
        try std.testing.expectEqual(@as(?[]const u8, null), stage.moved(b));
        try std.testing.expectEqual(@as(?[]const u8, null), stage.moved(a));
        const shift = stage.shift(c).?;
        try std.testing.expectEqual(@intFromPtr(stage.moved(c).?.ptr), @as(usize, @bitCast(@as(isize, @bitCast(@intFromPtr(c.ptr))) + shift)));
    } else try std.testing.expectEqual(@as(?[]const u8, null), stage.moved(c));
    try std.testing.expectEqualSlices(u8, data[0..sizes[0]], a);
    const first = stage.buffer;

    // Adjacent and overlapping ranges become one span; a range outside every shard and an empty one are skipped.
    try stage.read(gpa, &checkpoint, &.{ b, a[10..20], c, a, bytes[0..16], a[0..0] });
    if (direct) {
        try std.testing.expectEqual(@as(usize, 1), stage.spans.items.len);
        try std.testing.expect(stage.bytes() >= total);
        try std.testing.expectEqualSlices(u8, data, stage.moved(@as([*]const u8, a.ptr)[0..total]).?);
        // the buffer was large enough, and is the same
        try std.testing.expectEqual(first.ptr, stage.buffer.ptr);
        try std.testing.expectEqual(@as(usize, 0), @intFromPtr(stage.buffer.ptr) % safetensors.Stage.huge);
    }
    try std.testing.expectEqualSlices(u8, bytes, checkpoint.shards[0].map[0..bytes.len]);
}
test "a checkpoint of another's heads opens no file, and stages ranges laid out as that one lays them" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const page = std.heap.pageSize();
    // two shards, a tensor of the first ending in the page the second's file would follow it in
    var prng: std.Random.DefaultPrng = .init(43);
    const names = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
    for (names, [_][2]usize{ .{ 2 * page + 9, page + 5 }, .{ 7, 3 * page } }) |name, sizes| {
        const header_text = try std.fmt.allocPrint(gpa,
            \\{{"{s}a":{{"dtype":"U8","shape":[{d}],"data_offsets":[0,{d}]}},"{s}b":{{"dtype":"U8","shape":[{d}],"data_offsets":[{d},{d}]}}}}
        , .{ name[10..11], sizes[0], sizes[0], name[10..11], sizes[1], sizes[0], sizes[0] + sizes[1] });
        defer gpa.free(header_text);
        const bytes = try gpa.alloc(u8, 8 + header_text.len + sizes[0] + sizes[1]);
        defer gpa.free(bytes);
        std.mem.writeInt(u64, bytes[0..8], header_text.len, .little);
        @memcpy(bytes[8..][0..header_text.len], header_text);
        prng.random().bytes(bytes[8 + header_text.len ..]);
        try tmp.dir.writeFile(io, .{ .sub_path = name, .data = bytes });
    }

    var checkpoint: Checkpoint = try .open(gpa, io, tmp.dir, ".");
    defer checkpoint.close();
    var heads: [2][]const u8 = undefined;
    var lengths: [2]u64 = undefined;
    for (checkpoint.shards, &heads, &lengths) |shard, *head, *length| {
        head.* = shard.head;
        length.* = shard.map.len;
    }
    var shell: Checkpoint = try .fromHeads(gpa, &heads, &lengths);
    defer shell.close();
    try std.testing.expectEqual(checkpoint.tensors.count(), shell.tensors.count());
    for (shell.shards) |shard| try std.testing.expectEqual(@as(std.posix.fd_t, -1), shard.direct);

    // the same parts of the same tensors on both; the bytes this one stages, sent whole to the other
    const wanted = [_]struct { name: []const u8, from: usize, len: usize }{ .{ .name = "2b", .from = 0, .len = 3 * page }, .{ .name = "1a", .from = 5, .len = page }, .{ .name = "1b", .from = 0, .len = page + 5 }, .{ .name = "2a", .from = 1, .len = 6 } };
    var mine: [wanted.len][]const u8 = undefined;
    var theirs: [wanted.len][]const u8 = undefined;
    for (wanted, &mine, &theirs) |one, *a, *b| {
        const here = checkpoint.get(one.name).?;
        const there = shell.get(one.name).?;
        try std.testing.expectEqual(here.dtype, there.dtype);
        try std.testing.expectEqualSlices(u64, here.shape, there.shape);
        a.* = here.data[one.from..][0..one.len];
        b.* = there.data[one.from..][0..one.len];
    }
    var sending: safetensors.Stage = .{};
    defer sending.deinit(gpa);
    const sent = (try sending.lay(gpa, &checkpoint, &mine, true)).?;
    // what a direct read would bring (a tmpfs has none): the file's bytes, from its mapping
    for (mine) |range| @memcpy(@constCast(sending.moved(range).?), range);
    var receiving: safetensors.Stage = .{};
    defer receiving.deinit(gpa);
    const received = (try receiving.lay(gpa, &shell, &theirs, true)).?;
    try std.testing.expectEqual(sent.len, received.len);
    @memcpy(received, sent);
    for (mine, theirs) |a, b| try std.testing.expectEqualSlices(u8, a, receiving.moved(b).?);
    // without `every` a file without direct reads stages nothing
    try std.testing.expectEqual(@as(usize, 0), (try receiving.lay(gpa, &shell, &theirs, false)).?.len);
}

test "held ranges read as the file does, during and after, to the file's last page" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const page = std.heap.pageSize();

    // Three tensors over 40 pages and a bit: the first two share a page, the last ends the file mid-page.
    const sizes = [_]usize{ 3 * page + 100, 20 * page - 100 + 7, 17 * page + 513 };
    const header_text = try std.fmt.allocPrint(gpa,
        \\{{"a":{{"dtype":"U8","shape":[{d}],"data_offsets":[0,{d}]}},"b":{{"dtype":"U8","shape":[{d}],"data_offsets":[{d},{d}]}},"c":{{"dtype":"U8","shape":[{d}],"data_offsets":[{d},{d}]}}}}
    , .{ sizes[0], sizes[0], sizes[1], sizes[0], sizes[0] + sizes[1], sizes[2], sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2] });
    defer gpa.free(header_text);
    const total = sizes[0] + sizes[1] + sizes[2];
    const bytes = try gpa.alloc(u8, 8 + header_text.len + total);
    defer gpa.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header_text.len, .little);
    @memcpy(bytes[8..][0..header_text.len], header_text);
    var prng: std.Random.DefaultPrng = .init(41);
    prng.random().bytes(bytes[8 + header_text.len ..]);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });

    var checkpoint: Checkpoint = try .open(gpa, io, tmp.dir, ".");
    defer checkpoint.close();
    const data = bytes[8 + header_text.len ..];
    const a = checkpoint.get("a").?.data;
    const b = checkpoint.get("b").?.data;
    const c = checkpoint.get("c").?.data;
    // a tensor, and a part of one, are in their shard's mapping; memory that is no file's is in none
    try std.testing.expectEqual(@as(?*const File, &checkpoint.shards[0]), checkpoint.fileOf(c));
    try std.testing.expectEqual(@as(?*const File, &checkpoint.shards[0]), checkpoint.fileOf(b[5..9]));
    try std.testing.expectEqual(@as(?*const File, null), checkpoint.fileOf(bytes[0..16]));
    // Without direct reads (a tmpfs) hold is a no-op, and the rest still has to read right.
    const direct = checkpoint.shards[0].direct >= 0;

    var held = try checkpoint.hold(gpa, &.{ c, a });
    if (direct) try std.testing.expectEqual(@as(usize, 2), held.spans.len);
    try std.testing.expectEqualSlices(u8, data[0..sizes[0]], a);
    try std.testing.expectEqualSlices(u8, data[sizes[0]..][0..sizes[1]], b);
    try std.testing.expectEqualSlices(u8, data[sizes[0] + sizes[1] ..], c);
    held.release();
    try std.testing.expectEqualSlices(u8, data[0..sizes[0]], a);
    try std.testing.expectEqualSlices(u8, data[sizes[0] + sizes[1] ..], c);

    // Adjacent and overlapping ranges become one span; a range outside every shard and an empty one are skipped.
    var all = try checkpoint.hold(gpa, &.{ b, a[10..20], c, a, bytes[0..16], a[0..0] });
    if (direct) {
        try std.testing.expectEqual(@as(usize, 1), all.spans.len);
        try std.testing.expect(all.bytes() >= total);
    }
    try std.testing.expectEqualSlices(u8, data, @as([*]const u8, a.ptr)[0..total]);
    // The header's page was replaced too, where a tensor shares it.
    try std.testing.expectEqualSlices(u8, bytes, checkpoint.shards[0].map[0..bytes.len]);
    all.release();
    all.release();
    try std.testing.expectEqualSlices(u8, bytes, checkpoint.shards[0].map[0..bytes.len]);
}

test "a hold beside another leaves out the pages that one holds, and both read as the file does" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const page = std.heap.pageSize();

    // Two tensors that share a page, as a layer's last and the next one's first may.
    const sizes = [_]usize{ 5 * page + 300, 9 * page + 11 };
    const header_text = try std.fmt.allocPrint(gpa,
        \\{{"a":{{"dtype":"U8","shape":[{d}],"data_offsets":[0,{d}]}},"b":{{"dtype":"U8","shape":[{d}],"data_offsets":[{d},{d}]}}}}
    , .{ sizes[0], sizes[0], sizes[1], sizes[0], sizes[0] + sizes[1] });
    defer gpa.free(header_text);
    const bytes = try gpa.alloc(u8, 8 + header_text.len + sizes[0] + sizes[1]);
    defer gpa.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header_text.len, .little);
    @memcpy(bytes[8..][0..header_text.len], header_text);
    var prng: std.Random.DefaultPrng = .init(43);
    prng.random().bytes(bytes[8 + header_text.len ..]);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });

    var checkpoint: Checkpoint = try .open(gpa, io, tmp.dir, ".");
    defer checkpoint.close();
    const data = bytes[8 + header_text.len ..];
    const a = checkpoint.get("a").?.data;
    const b = checkpoint.get("b").?.data;
    const direct = checkpoint.shards[0].direct >= 0;

    var alone = try checkpoint.hold(gpa, &.{b});
    const whole = alone.bytes();
    alone.release();
    var first = try checkpoint.hold(gpa, &.{a});
    var second = try checkpoint.holdBeside(gpa, &.{b}, &first);
    // the page both tensors are on is the first's alone
    if (direct) try std.testing.expectEqual(whole - page, second.bytes());
    try std.testing.expectEqualSlices(u8, data[0..sizes[0]], a);
    try std.testing.expectEqualSlices(u8, data[sizes[0]..], b);
    first.release();
    try std.testing.expectEqualSlices(u8, data[sizes[0]..], b);
    second.release();
    try std.testing.expectEqualSlices(u8, bytes, checkpoint.shards[0].map[0..bytes.len]);
}
