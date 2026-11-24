//! `learnt.zig`'s test, on FP4-Local (`ZSPARK_GLM_FP4_LOCAL`, its caches' latents FP8 records): its first four
//! layers with both ranks in the
//! process, the states that leave written to a store in a temporary directory. A conversation that another
//! parts from leaves its state on disk; its next turn starts from the file and gives a fresh run's reply; a
//! file cut short is dropped and the prompt starts from the shorter, to the same reply. 13 GiB on the device.

const std = @import("std");
const testing = std.testing;
const shared = @import("../testing.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const spill = @import("../../../../src/decode/spill.zig");
const learned = @import("../../../../src/decode/learned.zig");
const learned_blocks = @import("../../../../src/decode/learned_blocks.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const session_file = @import("../../../../src/models/glm5/session/session.zig");
const learnt = @import("../../../../src/models/glm5/session/learnt.zig");

const Session = session_file.SessionOf(layout.world);

/// A prompt of `count` tokens, one of a family by `seed`.
fn promptOf(out: []i32, seed: usize) void {
    for (out, 0..) |*token, i| token.* = @intCast((i * 7919 + seed * 104729 + 17) % 150000 + 10);
}

test "a conversation's state that leaves goes to disk, and its next turn starts from the file with a fresh run's reply: FP4-Local, its caches FP8" {
    try spiltBack(testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest);
}

fn spiltBack(path: []const u8) !void {
    const gpa = testing.allocator;
    const io = testing.io;
    const floor, const worth, const block = .{ spill.floor, spill.worth, learned_blocks.block_tokens };
    defer spill.floor, spill.worth, learned_blocks.block_tokens = .{ floor, worth, block };
    spill.floor, spill.worth, learned_blocks.block_tokens = .{ 16, 16, 16 };
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const where = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(where);
    const session = try Session.create(gpa, arena, io, &checkpoint, &c, .{}, .{ .layers = 4, .capacity = 128, .snapshots = 4, .spill = true, .kv = .stored(&c) }, "test");
    defer session.destroy();
    try session.meet();
    // as `serve` opens it: the identity's probe, then the store and its writer, the session left holding nothing
    session.learning = try learnt.open(session, .{ .spill = .{ .dir = where, .gib = 1 }, .home = null, .checkpoint = path, .commit = "test", .dirty = false, .min_free = 0 });
    const store = session.spilled orelse return error.NoStore;
    try testing.expectEqual(@as(usize, 0), session.state.pos);
    try testing.expectEqual(@as(usize, 0), session.marksHeld());

    for ([_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 7, .temperature = 0.8 } }, 0..) |settings, round| {
        var a: [48]i32 = undefined;
        promptOf(&a, 1 + 3 * round);
        var b: [40]i32 = undefined;
        promptOf(&b, 2 + 3 * round);
        var more: [8]i32 = undefined;
        promptOf(&more, 3 + 3 * round);
        var reply_a: [8]i32 = undefined;
        _ = try session.reply(&a, settings, .{ .eos = &.{}, .fresh = true }, &reply_a);
        // the mark two tokens before the prompt's end, the highest of its prompt's, goes when another parts
        const written = store.written;
        var reply_b: [8]i32 = undefined;
        _ = try session.reply(&b, settings, .{ .eos = &.{} }, &reply_b);
        try testing.expectEqual(written + 1, store.written);
        try testing.expect(store.has(a[0 .. a.len - 1]));

        var next: std.ArrayList(i32) = .empty;
        try next.appendSlice(arena, &a);
        try next.appendSlice(arena, &reply_a);
        try next.appendSlice(arena, &more);
        const read = store.read;
        var onward: [10]i32 = undefined;
        const went = try session.reply(next.items, settings, .{ .eos = &.{} }, &onward);
        try testing.expectEqual(a.len - 2, went.cached);
        try testing.expectEqual(read + 1, store.read);
        var fresh: [10]i32 = undefined;
        const again = try session.reply(next.items, settings, .{ .eos = &.{}, .fresh = true }, &fresh);
        try testing.expectEqual(@as(usize, 0), again.cached);
        try testing.expectEqualSlices(i32, &fresh, &onward);

        // a file cut short is dropped, and the prompt starts from the shorter one, to the same reply
        _ = try session.reply(&b, settings, .{ .eos = &.{}, .fresh = true }, &reply_b);
        // the writer's files on disk before one is cut
        learnt.settle(session);
        const len = store.longest(next.items, next.items.len).?;
        const name = learned.fileOf(.spilled, learned.keyOf(next.items[0..len]), false);
        const file = try store.dir.openFile(io, name.get(), .{ .mode = .read_write });
        try file.setLength(io, try file.length(io) - 1);
        file.close(io);
        var whole: [10]i32 = undefined;
        const ran = try session.reply(next.items, settings, .{ .eos = &.{} }, &whole);
        try testing.expect(!store.has(next.items[0..len]));
        try testing.expectEqual(a.len - 2, ran.cached);
        try testing.expect(len - 1 > a.len - 2);
        try testing.expectEqualSlices(i32, &fresh, &whole);
    }
}
