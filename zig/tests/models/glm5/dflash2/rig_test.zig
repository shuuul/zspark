//! `rig.zig`'s tests: the drafter crossing from rank 0 to rank 1 under `serve --weights-from-peer`
//! (`Source.cross`), its NCCL transfer a copy in one process here. The published drafter's file is read where it
//! is (`ZSPARK_GLM_DRAFTER`); nothing of it is written.

const std = @import("std");
const testing = std.testing;
const safetensors = @import("../../../../src/weights/safetensors.zig");
const rig = @import("../../../../src/models/glm5/dflash2/rig.zig");

test "the published drafter crossed whole: rank 1's every tensor is the file's bytes, from no mapping, no GPU" {
    const path = testing.environ.getPosix("ZSPARK_GLM_DRAFTER") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var checkpoint: safetensors.Checkpoint = try .open(gpa, testing.io, .cwd(), path);
    defer checkpoint.close();
    if (checkpoint.shards[0].direct < 0) return error.SkipZigTest;
    // rank 1's: rank 0's headers alone, whose tensors' bytes no read reaches
    const heads = try gpa.alloc([]const u8, checkpoint.shards.len);
    defer gpa.free(heads);
    const lengths = try gpa.alloc(u64, checkpoint.shards.len);
    defer gpa.free(lengths);
    for (checkpoint.shards, heads, lengths) |shard, *head, *length| {
        head.* = shard.head;
        length.* = shard.map.len;
    }
    var shell: safetensors.Checkpoint = try .fromHeads(gpa, heads, lengths);
    defer shell.close();
    var sending: rig.Source = .{ .checkpoint = &checkpoint };
    defer sending.deinit(gpa);
    var receiving: rig.Source = .{ .checkpoint = &shell };
    defer receiving.deinit(gpa);
    const sent = try sending.lay(gpa, true, null);
    const into = try receiving.lay(gpa, false, null);
    try testing.expectEqual(sent.len, into.len);
    @memcpy(into, sent);
    var bytes: usize = 0;
    for (checkpoint.tensors.keys(), checkpoint.tensors.values()) |name, tensor| {
        for ([_]*const rig.Source{ &sending, &receiving }) |source| {
            const got = source.get(name).?;
            try testing.expectEqual(tensor.dtype, got.dtype);
            try testing.expectEqualSlices(u64, tensor.shape, got.shape);
            if (tensor.data.len == 0) continue;
            // staged: neither the file's mapping nor the shell's unreadable one
            try testing.expect(checkpoint.fileOf(got.data) == null and shell.fileOf(got.data) == null);
            try testing.expectEqualSlices(u8, tensor.data, got.data);
        }
        bytes += tensor.data.len;
    }
    std.debug.print("the drafter crossed: {d} tensors, {d} bytes of them in {d} laid out\n", .{ checkpoint.tensors.count(), bytes, sent.len });
}
