//! A cache whose extent table is not the identity gives the same rows as one whose table is. The admission
//! predicate refuses a context before any buffer is asked for.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const dsa = @import("../../../../src/models/glm5/layers/dsa.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const loopback = @import("../../../../src/models/glm5/layers/loopback.zig");
const shared = @import("../testing.zig");

const fx = @import("../../../../src/models/glm5/checks/random.zig").fx;
const Fixture = shared.Fixture;
const Loop = shared.Loop;

test "an extent's context larger than the device is refused" {
    try testing.expect(!dsa.admits(cuda.reserve, 1));
    try testing.expect(dsa.admits(cuda.reserve + 64, 32));
    try testing.expect(!dsa.admits(cuda.available(testing.io), std.math.maxInt(usize) / 4));
}

fn play(loop: *Loop, tokens: []const i32, logits: []u16) !void {
    const half = fx.vocab / layout.world;
    var row: [fx.vocab]u16 = undefined;
    var at: usize = 0;
    while (at < tokens.len) {
        const n = @min(16, tokens.len - at);
        const window = tokens[at..][0..n];
        try loopback.forward(loop.m, &loop.state, &loop.b, window);
        try cuda.synchronize();
        for (0..n) |r| {
            for (loop.b.ranks, 0..) |b, rank| {
                try b.logits.downloadAt(r * half * 2, std.mem.sliceAsBytes(row[0..half]));
                @memcpy(logits[(at + r) * fx.vocab + rank * half ..][0..half], row[0..half]);
            }
        }
        try loopback.commit(loop.m, &loop.state, loop.b, window, n);
        at += n;
    }
}

test "scattered extents give the identity table's rows" {
    var fixture: Fixture = undefined;
    try fixture.init(46, 528);
    defer fixture.deinit();
    const tokens = fixture.tokens;
    const gpa = testing.allocator;
    const logits = try gpa.alloc(u16, tokens.len * fx.vocab);
    defer gpa.free(logits);
    const again = try gpa.alloc(u16, tokens.len * fx.vocab);
    defer gpa.free(again);

    var identity = try fixture.loop(1024);
    defer identity.deinit();
    try play(&identity, tokens, logits);

    var scattered = try fixture.loop(1024);
    defer scattered.deinit();
    const swapped = [_]i32{ 1, 0 };
    for (&scattered.state.ranks) |*rank| try rank.dsa.?.retarget(&swapped);
    try play(&scattered, tokens, again);
    try testing.expectEqualSlices(u16, logits, again);
}
