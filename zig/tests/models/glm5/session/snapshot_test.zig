//! A snapshot restored and replayed equals a cold run of the same tokens: the suffix after a cut, an edited
//! suffix, the rows after a hang-up's last mark, and a shot taken inside a window and restored once the KDA
//! state has turned side, and one replayed inside a window (spec 112 WS-06). Both ranks hold the same position. The thinning:
//! `tests/decode/marks_test.zig`.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const dsa = @import("../../../../src/models/glm5/layers/dsa.zig");
const forward = @import("../../../../src/models/glm5/layers/forward.zig");
const kda = @import("../../../../src/models/glm5/layers/kda.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const loopback = @import("../../../../src/models/glm5/layers/loopback.zig");
const snapshot = @import("../../../../src/models/glm5/session/snapshot.zig");
const shared = @import("../testing.zig");

const fx = @import("../../../../src/models/glm5/checks/random.zig").fx;
const Fixture = shared.Fixture;
const Loop = shared.Loop;

/// One token a step. `logits` is bf16 [tokens, vocab], both ranks' halves, when it has room.
fn run(loop: *Loop, tokens: []const i32, logits: []u16) !void {
    const half = fx.vocab / layout.world;
    var row: [fx.vocab]u16 = undefined;
    for (tokens, 0..) |token, i| {
        const window = [_]i32{token};
        try loopback.forward(loop.m, &loop.state, &loop.b, &window);
        try cuda.synchronize();
        if (logits.len > 0) {
            for (loop.b.ranks, 0..) |b, rank| {
                try cuda.download(b.logits.rows(1), std.mem.sliceAsBytes(row[0..half]));
                @memcpy(logits[i * fx.vocab + rank * half ..][0..half], row[0..half]);
            }
        }
        try loopback.commit(loop.m, &loop.state, loop.b, &window, 1);
    }
}

fn take(loop: *Loop) ![layout.world]snapshot.Shot {
    return takeInto(loop, @splat(null));
}

/// `take` into each rank's `spares`, a dropped shot's KDA buffers where one is given (`Session.spare`).
fn takeInto(loop: *Loop, spares: [layout.world]?kda.Mark) ![layout.world]snapshot.Shot {
    var shots: [layout.world]snapshot.Shot = undefined;
    var done: usize = 0;
    errdefer for (shots[0..done]) |*shot| shot.deinit();
    errdefer for (spares[done + 1 ..]) |spare| if (spare) |kept| {
        var gone = kept;
        gone.deinit();
    };
    for (&shots, loop.state.ranks, spares) |*shot, rank, spare| {
        shot.* = try snapshot.takeRank(rank, spare);
        done += 1;
    }
    return shots;
}

fn release(shots: [layout.world]snapshot.Shot) void {
    var copy = shots;
    for (&copy) |*shot| shot.deinit();
}

fn restore(loop: *Loop, shots: [layout.world]snapshot.Shot) !void {
    for (shots, &loop.state.ranks) |shot, *rank| try snapshot.restoreRank(shot, rank);
    loop.state.pos = shots[0].pos;
}

fn expectState(loop: *Loop, cold: *Loop) !void {
    try testing.expectEqual(cold.state.pos, loop.state.pos);
    try testing.expectEqual(loop.state.ranks[0].pos, loop.state.ranks[1].pos);
    const gpa = testing.allocator;
    for (loop.state.ranks, cold.state.ranks) |got, want| {
        const chains = got.kda.?;
        const one = chains.heads * kda.head_dim * kda.head_dim;
        const a = try gpa.alloc(u32, chains.layers * one);
        defer gpa.free(a);
        const b = try gpa.alloc(u32, chains.layers * one);
        defer gpa.free(b);
        for (0..chains.layers) |index| {
            try cuda.download(chains.committed(index), std.mem.sliceAsBytes(a[index * one ..][0..one]));
            try cuda.download(want.kda.?.committed(index), std.mem.sliceAsBytes(b[index * one ..][0..one]));
        }
        try testing.expectEqualSlices(u32, b, a);
        const cache = got.dsa.?;
        const rows = got.pos * dsa.latent;
        const lat_a = try gpa.alloc(u16, cache.layers * rows);
        defer gpa.free(lat_a);
        const lat_b = try gpa.alloc(u16, cache.layers * rows);
        defer gpa.free(lat_b);
        for (0..cache.layers) |index| {
            var view = cache.layer(index);
            view.size[0] = @intCast(got.pos);
            var other = want.dsa.?.layer(index);
            other.size[0] = @intCast(got.pos);
            try cuda.download(view, std.mem.sliceAsBytes(lat_a[index * rows ..][0..rows]));
            try cuda.download(other, std.mem.sliceAsBytes(lat_b[index * rows ..][0..rows]));
        }
        try testing.expectEqualSlices(u16, lat_b, lat_a);
    }
}

test "a restored suffix, an edited one and a hang-up equal a cold run" {
    var fixture: Fixture = undefined;
    try fixture.init(46, 40);
    defer fixture.deinit();
    var live = try fixture.loop(64);
    defer live.deinit();
    var cold = try fixture.loop(64);
    defer cold.deinit();
    const tokens = fixture.tokens;
    const gpa = testing.allocator;

    const cut: usize = 16;
    try run(&live, tokens[0..cut], &.{});
    const shots = try take(&live);
    defer release(shots);
    for (shots) |shot| try testing.expectEqual(cut, shot.pos);

    const suffix = tokens[cut..32];
    const live_logits = try gpa.alloc(u16, suffix.len * fx.vocab);
    defer gpa.free(live_logits);
    try restore(&live, shots);
    try run(&live, suffix, live_logits);
    const recorded = try gpa.alloc(u16, 32 * fx.vocab);
    defer gpa.free(recorded);
    try run(&cold, tokens[0..32], recorded);
    try testing.expectEqualSlices(u16, recorded[cut * fx.vocab ..], live_logits);
    try expectState(&live, &cold);

    // an edited suffix from the same cut
    var edited: [8]i32 = undefined;
    for (&edited, 0..) |*id, i| id.* = @mod(tokens[cut + i] + 17, @as(i32, fx.vocab));
    try restore(&live, shots);
    const edit_logits = try gpa.alloc(u16, edited.len * fx.vocab);
    defer gpa.free(edit_logits);
    try run(&live, &edited, edit_logits);
    try cold.state.reset();
    var edited_prompt: [24]i32 = undefined;
    @memcpy(edited_prompt[0..cut], tokens[0..cut]);
    @memcpy(edited_prompt[cut..], &edited);
    const edit_cold = try gpa.alloc(u16, edited_prompt.len * fx.vocab);
    defer gpa.free(edit_cold);
    try run(&cold, &edited_prompt, edit_cold);
    try testing.expectEqualSlices(u16, edit_cold[cut * fx.vocab ..], edit_logits);
    try expectState(&live, &cold);

    // a hang-up at 20: the last mark at or below it is 16, and the rows from there equal a cold run
    try live.state.reset();
    try run(&live, tokens[0..8], &.{});
    var at8 = try take(&live);
    defer release(at8);
    try run(&live, tokens[8..16], &.{});
    const at16 = try take(&live);
    defer release(at16);
    try run(&live, tokens[16..20], &.{});
    const held = [_]usize{ at8[0].pos, at16[0].pos };
    try testing.expectEqualSlices(usize, &.{ 8, 16 }, &held);
    try testing.expect(at16[0].pos <= 20 and at16[1].pos == at16[0].pos);
    try restore(&live, at16);
    const hung = try gpa.alloc(u16, 4 * fx.vocab);
    defer gpa.free(hung);
    try run(&live, tokens[16..20], hung);
    try cold.state.reset();
    const hung_cold = try gpa.alloc(u16, 20 * fx.vocab);
    defer gpa.free(hung_cold);
    try run(&cold, tokens[0..20], hung_cold);
    try testing.expectEqualSlices(u16, hung_cold[16 * fx.vocab ..], hung);
    try expectState(&live, &cold);

    // a shot taken into a dropped one's buffers, at8's and stale, is a fresh one's
    var spares: [layout.world]?kda.Mark = undefined;
    for (&spares, &at8) |*spare, *shot| {
        spare.* = shot.kda;
        shot.kda = null;
    }
    const at20 = try takeInto(&live, spares);
    defer release(at20);
    try run(&live, tokens[20..28], &.{});
    try restore(&live, at20);
    const resumed = try gpa.alloc(u16, 4 * fx.vocab);
    defer gpa.free(resumed);
    try run(&live, tokens[20..24], resumed);
    try cold.state.reset();
    const resumed_cold = try gpa.alloc(u16, 24 * fx.vocab);
    defer gpa.free(resumed_cold);
    try run(&cold, tokens[0..24], resumed_cold);
    try testing.expectEqualSlices(u16, resumed_cold[20 * fx.vocab ..], resumed);
    try expectState(&live, &cold);

    // a shot holds the side of the KDA state a window reads, taken between a window and its commit
    // (a plan's at a window's first row, `marks.zig`) and restored once the state's side has turned; then a window kept in part (a
    // replay from the restored side) and steps
    try live.state.reset();
    try run(&live, tokens[0..5], &.{});
    const window = tokens[5..12];
    try loopback.forward(live.m, &live.state, &live.b, window);
    const at5 = try take(&live);
    defer release(at5);
    const side = live.state.ranks[0].kda.?.cur;
    try loopback.commit(live.m, &live.state, live.b, window, window.len);
    try run(&live, tokens[12..14], &.{});
    try testing.expect(live.state.ranks[0].kda.?.cur != side);
    try restore(&live, at5);
    try testing.expect(live.state.ranks[0].kda.?.cur != side);
    try loopback.forward(live.m, &live.state, &live.b, window);
    try loopback.commit(live.m, &live.state, live.b, window, 4);
    try cold.state.reset();
    try run(&cold, tokens[0..9], &.{});
    try expectState(&live, &cold);
    const replayed = try gpa.alloc(u16, 3 * fx.vocab);
    defer gpa.free(replayed);
    try run(&live, tokens[9..12], replayed);
    try cold.state.reset();
    const replayed_cold = try gpa.alloc(u16, 12 * fx.vocab);
    defer gpa.free(replayed_cold);
    try run(&cold, tokens[0..12], replayed_cold);
    try testing.expectEqualSlices(u16, replayed_cold[9 * fx.vocab ..], replayed);
    try expectState(&live, &cold);
}

/// `take` of the state after the first `rows` rows of the window that has run and is not committed: into the
/// marks `asked` before it ran, where it ran chunked (`ask`).
fn takeWithin(loop: *Loop, rows: usize, asked: ?[layout.world]kda.Mark) ![layout.world]snapshot.Shot {
    var shots: [layout.world]snapshot.Shot = undefined;
    var done: usize = 0;
    errdefer for (shots[0..done]) |*shot| shot.deinit();
    for (&shots, loop.state.ranks, 0..) |*shot, rank, r| {
        shot.* = try snapshot.takeWithin(rank, rows, if (asked) |marks| marks[r] else null);
        done += 1;
    }
    return shots;
}

/// Each rank's KDA state asked, before the next window runs, to write the state after its first `rows` rows.
fn ask(loop: *Loop, rows: usize) ![layout.world]kda.Mark {
    var marks: [layout.world]kda.Mark = undefined;
    var done: usize = 0;
    errdefer for (marks[0..done]) |*one| one.deinit();
    for (&marks, &loop.state.ranks) |*mark, *rank| {
        const chains = &rank.kda.?;
        mark.* = try .init(chains.layers, chains.heads);
        done += 1;
        chains.ask(rows, mark.*);
    }
    return marks;
}

test "a mark replayed inside a window is the state the window leaves there, and the window commits as it would have" {
    var fixture: Fixture = undefined;
    try fixture.init(47, 96);
    defer fixture.deinit();
    var live = try fixture.loop(128);
    defer live.deinit();
    var cold = try fixture.loop(128);
    defer cold.deinit();
    const tokens = fixture.tokens;
    const gpa = testing.allocator;

    // A window of bf16 rows, whose rows are serial steps', and one of the quantized forms and the chunked chain
    // (`forward.bf16_rows`, `kda.chunks`), whose rows are another such window's cut at a chunk's start (spec 115,
    // 116): marks inside each, the chunked one's at its chunk's start, position 64 (`session/marks.zig` snaps them).
    const wide = 70;
    try testing.expect(kda.chunks(wide) and wide > forward.bf16_rows and wide <= fx.max_rows and kda.chunks(kda.chunk - 12));
    const Window = struct { from: usize, stop: usize, within: []const usize };
    const windows = [_]Window{ .{ .from = 5, .stop = 12, .within = &.{ 1, 3, 6 } }, .{ .from = 12, .stop = 12 + wide, .within = &.{kda.chunk - 12} } };
    var shots: [4][layout.world]snapshot.Shot = undefined;
    var taken: usize = 0;
    defer for (shots[0..taken]) |one| release(one);
    try run(&live, tokens[0..5], &.{});
    for (windows) |w| {
        const window = tokens[w.from..w.stop];
        // a chunked window's marks are asked before it runs; the asked buffers become the shots' (spec 116)
        var asked: [4]?[layout.world]kda.Mark = @splat(null);
        if (kda.chunks(window.len)) for (w.within, 0..) |rows, i| {
            asked[i] = try ask(&live, rows);
        };
        try loopback.forward(live.m, &live.state, &live.b, window);
        for (w.within, 0..) |rows, i| {
            shots[taken] = try takeWithin(&live, rows, asked[i]);
            taken += 1;
        }
        try loopback.commit(live.m, &live.state, live.b, window, window.len);
    }
    // the window committed whole is the same with the marks taken as without
    try run(&cold, tokens[0..5], &.{});
    for (windows) |w| {
        try loopback.forward(cold.m, &cold.state, &cold.b, tokens[w.from..w.stop]);
        try loopback.commit(cold.m, &cold.state, cold.b, tokens[w.from..w.stop], w.stop - w.from);
    }
    try expectState(&live, &cold);
    try expectConv(&live, &cold);

    // Each restored, and the steps after it, equal a cold run to its position: serial steps in the bf16 window,
    // a quantized window to the mark's row in the chunked one, committed whole. The highest
    // first, as a session drops the marks above the one it restores: the steps after a lower one write over
    // the cache's rows above it.
    const after = 4;
    const resumed = try gpa.alloc(u16, after * fx.vocab);
    defer gpa.free(resumed);
    const recorded = try gpa.alloc(u16, after * fx.vocab);
    defer gpa.free(recorded);
    var left = taken;
    while (left > 0) {
        left -= 1;
        const shot = shots[left];
        const at = shot[0].pos;
        try testing.expect(shot[1].pos == at);
        try restore(&live, shot);
        try cold.state.reset();
        if (at <= windows[0].stop) {
            try run(&cold, tokens[0..at], &.{});
        } else {
            const from = windows[1].from;
            try run(&cold, tokens[0..from], &.{});
            const window = tokens[from..at];
            try loopback.forward(cold.m, &cold.state, &cold.b, window);
            try loopback.commit(cold.m, &cold.state, cold.b, window, window.len);
        }
        try expectState(&live, &cold);
        try expectConv(&live, &cold);
        try run(&live, tokens[at..][0..after], resumed);
        try run(&cold, tokens[at..][0..after], recorded);
        try testing.expectEqualSlices(u16, recorded, resumed);
        try expectState(&live, &cold);
    }
}

/// The KDA layers' conv windows of both ranks alike.
fn expectConv(loop: *Loop, cold: *Loop) !void {
    const gpa = testing.allocator;
    for (loop.state.ranks, cold.state.ranks) |got, want| {
        const n = got.kda.?.conv.view.count();
        const a = try gpa.alloc(u16, n);
        defer gpa.free(a);
        const b = try gpa.alloc(u16, n);
        defer gpa.free(b);
        try cuda.download(got.kda.?.conv.view, std.mem.sliceAsBytes(a));
        try cuda.download(want.kda.?.conv.view, std.mem.sliceAsBytes(b));
        try testing.expectEqualSlices(u16, b, a);
    }
}
