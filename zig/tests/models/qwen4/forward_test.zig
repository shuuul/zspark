//! `forward.zig`'s tests on a small model of random weights (`testing.zig`'s `random_model`): a window equals its serial steps through the whole forward, a kept prefix leaves the state the serial
//! steps leave, graphs give the eager step's bits. Each checkpoint's first layers: `forward_checkpoint_test.zig`;
//! the helpers both use: `forward_fixture.zig`.

const std = @import("std");
const hc = @import("../../../src/models/qwen4/blocks/hc.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("testing.zig");
const testModel = fixtures.random_model.testModel;
const fx = fixtures.random_model.fx;
const randomBf16 = fixtures.random_model.randomBf16;
const randomFloats = fixtures.random_model.randomFloats;
const cuda = @import("../../../src/cuda/cuda.zig");
const face = @import("../../../src/weights/face.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const forward_file = @import("../../../src/models/qwen4/forward.zig");
const gdn = @import("../../../src/models/qwen4/blocks/gdn.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const Buffers = forward_file.Buffers;
const Snapshot = forward_file.Snapshot;
const State = forward_file.State;
const commit = forward_file.commit;
const forward = forward_file.forward;
const forward_fixture = @import("forward_fixture.zig");
const decode = forward_fixture.decode;
const decodeWith = forward_fixture.decodeWith;
const expectFoldedAsBefore = forward_fixture.expectFoldedAsBefore;
const expectOwnedAsShared = forward_fixture.expectOwnedAsShared;
const expectWideAsKept = forward_fixture.expectWideAsKept;
const expectSameSnapshot = forward_fixture.expectSameSnapshot;
const expectPoisonedAsClean = forward_fixture.expectPoisonedAsClean;

test "captured graphs give the eager step's bits, one graph for each window size and state side" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try testModel(gpa, arena.allocator());
    defer m.deinit();

    const tokens = [_]i32{ 5, 300, 17, 17, 0, 251, 88, 9, 123, 4, 4, 200, 31 };
    const eager = try decode(gpa, m, &tokens, 1, 1);
    defer gpa.free(eager);
    const Case = struct { window: usize, keep: usize, captures: usize };
    // serial steps replay two graphs, the DeltaNet states being on one side and then the other; a window whose
    // rows are not all kept is run again from where the kept ones end, and the sequence's last window is short
    for ([_]Case{ .{ .window = 1, .keep = 1, .captures = 2 }, .{ .window = 4, .keep = 4, .captures = 3 }, .{ .window = 3, .keep = 2, .captures = 3 } }) |case| {
        var captures: usize = 0;
        const graphed = try decodeWith(gpa, m, &tokens, case.window, case.keep, &captures);
        defer gpa.free(graphed);
        try testing.expectEqualSlices(u16, eager, graphed);
        try testing.expectEqual(case.captures, captures);
    }
}

test "a window kept in part, or a window of drafts kept whole, is not replayed by its commit: the states wait behind, and a state taken there is the serial steps'" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try testModel(gpa, arena.allocator());
    defer m.deinit();
    const tokens = [_]i32{ 5, 300, 17, 17, 0, 251, 88, 9 };
    var serial: State = try .init(gpa, m, m.max_rows, 512);
    defer serial.deinit();
    var state: State = try .init(gpa, m, m.max_rows, 512);
    defer state.deinit();
    var b: Buffers = try .init(gpa, m, m.max_rows, 512);
    defer b.deinit();
    var one: Buffers = try .init(gpa, m, m.max_rows, 512);
    defer one.deinit();
    // windows of 3, 2, 2, 3 and 1 rows, of which 2, 1, 2, 1 and 1 are kept: the window of two kept whole wrote no
    // state (`stores`), so it waits behind as the others do; the row's window writes it
    const Window = struct { rows: usize, keep: usize };
    var at: usize = 0;
    for ([_]Window{ .{ .rows = 3, .keep = 2 }, .{ .rows = 2, .keep = 1 }, .{ .rows = 2, .keep = 2 }, .{ .rows = 3, .keep = 1 }, .{ .rows = 1, .keep = 1 } }) |window| {
        const side = state.current[0];
        const behind = if (forward_file.lands(state, b, window.rows)) 0 else state.behind;
        try forward(m, &state, &b, tokens[at..][0..window.rows]);
        try commit(m, &state, b, tokens[at..][0..window.rows], window.keep);
        if (window.keep < window.rows or !forward_file.stores(window.rows)) {
            // nothing replayed: the committed side is where the window found it, or where landing left it
            try testing.expectEqual(behind + window.keep, state.behind);
            try testing.expect(behind == 0 or state.current[0] == side);
        } else try testing.expectEqual(0, state.behind);
        for (tokens[at..][0..window.keep]) |*token| {
            try forward(m, &serial, &one, token[0..1]);
            try commit(m, &serial, one, token[0..1], 1);
        }
        at += window.keep;
        var want: Snapshot = try .take(gpa, serial);
        defer want.deinit();
        var got: Snapshot = try .take(gpa, state);
        defer got.deinit();
        try expectSameSnapshot(gpa, want, got);
    }
    try testing.expect(state.kept >= 4);
}

test "a window through the whole forward equals its serial steps, whatever prefix is kept" {
    try windowEqualsSerial();
}

/// The read-out's up and `mix` by `hc.upMixWindow` (spec 095), its gate lowered to the test's rows: steps of one
/// row through it give the launches' bits, and windows through it the serial steps' without it.
fn windowReadout(m: model.Model, tokens: []const i32) !void {
    const gpa = testing.allocator;
    const kept = hc.window_rows;
    defer hc.window_rows = kept;
    hc.window_rows = 0;
    const launches = try decode(gpa, m, tokens, 1, 1);
    defer gpa.free(launches);
    hc.window_rows = 1;
    var before = hc.window_readouts;
    const fused = try decode(gpa, m, tokens, 1, 1);
    defer gpa.free(fused);
    try testing.expect(hc.window_readouts > before);
    try testing.expectEqualSlices(u16, launches, fused);
    // from 2 rows: the windows by it, the steps of one row without it
    hc.window_rows = 2;
    for ([_][2]usize{ .{ 4, 4 }, .{ 3, 2 }, .{ tokens.len, tokens.len } }) |case| {
        before = hc.window_readouts;
        const windowed = try decode(gpa, m, tokens, case[0], case[1]);
        defer gpa.free(windowed);
        try testing.expect(hc.window_readouts > before);
        try testing.expectEqualSlices(u16, launches, windowed);
    }
}

test "with the read-out's up and mix as one launch a window equals the serial steps, and a step keeps its bits" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // a prompt's width of rows, the whole run as one window
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 16);
    defer m.deinit();
    try windowReadout(m, &.{ 5, 300, 17, 17, 0, 251, 88, 9, 123 });
}

fn windowEqualsSerial() !void {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try testModel(gpa, arena.allocator());
    defer m.deinit();

    const tokens = [_]i32{ 5, 300, 17, 17, 0, 251, 88, 9, 123 };
    const serial = try decode(gpa, m, &tokens, 1, 1);
    defer gpa.free(serial);
    var finite = true;
    var spread: f32 = 0;
    for (serial) |bits| {
        const v = numeric.f32FromBf16(bits);
        finite = finite and std.math.isFinite(v);
        spread = @max(spread, @abs(v));
    }
    try testing.expect(finite and spread > 0);
    // later rows differ from the first: the state carries
    try testing.expect(!std.mem.eql(u16, serial[0..fx.shape.vocab], serial[3 * fx.shape.vocab ..][0..fx.shape.vocab]));

    const Case = struct { window: usize, keep: usize };
    for ([_]Case{ .{ .window = 4, .keep = 4 }, .{ .window = 3, .keep = 3 }, .{ .window = 4, .keep = 1 }, .{ .window = 4, .keep = 2 }, .{ .window = 4, .keep = 3 } }) |case| {
        const windowed = try decode(gpa, m, &tokens, case.window, case.keep);
        defer gpa.free(windowed);
        try testing.expectEqualSlices(u16, serial, windowed);
    }
    try expectFoldedAsBefore(gpa, m, &tokens, serial, m.max_rows);
    try expectPoisonedAsClean(gpa, m, &tokens, serial, &tokens, serial, m.max_rows);
    try expectWideAsKept(gpa, m, &tokens, m.max_rows);
    cuda.Scratch.shared = false;
    defer cuda.Scratch.shared = true;
    var owned = try testModel(gpa, arena.allocator());
    defer owned.deinit();
    try expectOwnedAsShared(gpa, m, owned, &tokens, serial, &tokens, serial, m.max_rows);
}

test "prepare restores every state side on error, and holds predicts the key after staging lands deferred rows" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try testModel(gpa, arena.allocator());
    defer m.deinit();
    var state: State = try .init(gpa, m, m.max_rows, 512);
    defer state.deinit();
    var b: Buffers = try .init(gpa, m, m.max_rows, 512);
    defer b.deinit();
    var failing: testing.FailingAllocator = .init(gpa, .{ .fail_index = 0 });
    var rejected: forward_file.Graphs = try .init(failing.allocator());
    defer rejected.deinit();
    try testing.expectError(error.OutOfMemory, rejected.prepare(m, &state, &b, 2, 1, 0));
    try testing.expect(std.mem.allEqual(u1, state.current, 0));
    try testing.expectEqual(0, rejected.cache.count());

    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    state.kept = 3;
    try graphs.forward(m, &state, &b, &.{ 5, 17, 200 });
    try commit(m, &state, b, &.{ 5, 17, 200 }, 2);
    try testing.expectEqual(2, state.behind);
    try testing.expectEqual(0, state.current[0]);
    try graphs.prepare(m, &state, &b, 2, 1, 0);
    try testing.expect(std.mem.allEqual(u1, state.current, 0));
    try testing.expect(graphs.holds(state, b, 2));
    try testing.expectEqual(2, graphs.cache.count());
    try graphs.forward(m, &state, &b, &.{ 200, 3 });
    try testing.expectEqual(1, state.current[0]);
    try testing.expectEqual(2, graphs.cache.count());
    try testing.expect(graphs.cache.contains(.{ .rows = 3, .current = 0, .context = 0 }));
    try testing.expect(graphs.cache.contains(.{ .rows = 2, .current = 1, .context = 0 }));
}
