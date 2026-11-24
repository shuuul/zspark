//! What `forward_test.zig` and `forward_checkpoint_test.zig` share (spec 104 WS-06 moved it out of
//! `forward_test.zig`): a sequence decoded in windows, and the checks that a window, a fold, a poisoned buffer,
//! a wide window and a shared model each give the serial steps' bits. And the check on a checkpoint's first
//! layers that `checkpoint_prefix_test.zig` runs on its shared load (`firstLayers`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const forward_file = @import("../../../src/models/qwen4/forward.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const gdn = @import("../../../src/models/qwen4/blocks/gdn.zig");
const qwen4_config = @import("../../../src/models/qwen4/config.zig");
const layout = @import("../../../src/models/qwen4/layout.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const checkpoints = @import("testing.zig").checkpoints;
const Buffers = forward_file.Buffers;
const Graphs = forward_file.Graphs;
const Snapshot = forward_file.Snapshot;
const State = forward_file.State;
const commit = forward_file.commit;
const downloadRow = forward_file.downloadRow;
const forward = forward_file.forward;
const forwardScoring = forward_file.forwardScoring;
const widerThanKept = forward_file.widerThanKept;

/// The logits of each of `tokens`, run in windows of `window` rows with `keep` of each kept; rows past `keep`
/// are run again by the next window.
pub fn decode(gpa: Allocator, m: model.Model, tokens: []const i32, window: usize, keep: usize) ![]u16 {
    return decodeWith(gpa, m, tokens, window, keep, null);
}

/// `decode`, through `graphs` when given; `captures` then takes how many graphs that made.
pub fn decodeWith(gpa: Allocator, m: model.Model, tokens: []const i32, window: usize, keep: usize, captures: ?*usize) ![]u16 {
    const vocab = m.shape.vocab;
    var state: State = try .init(gpa, m, m.max_rows, 512);
    defer state.deinit();
    var b: Buffers = try .init(gpa, m, m.max_rows, 512);
    defer b.deinit();
    var graphs: ?Graphs = if (captures != null) try .init(gpa) else null;
    defer if (graphs) |*g| g.deinit();
    const out = try gpa.alloc(u16, tokens.len * vocab);
    errdefer gpa.free(out);
    var at: usize = 0;
    while (at < tokens.len) {
        const rows = @min(window, tokens.len - at);
        const kept = @min(keep, rows);
        if (graphs) |*g| try g.forward(m, &state, &b, tokens[at..][0..rows]) else try forward(m, &state, &b, tokens[at..][0..rows]);
        try cuda.synchronize();
        for (0..kept) |r| try downloadRow(b.logits, r, vocab, out[(at + r) * vocab ..][0..vocab]);
        try commit(m, &state, b, tokens[at..][0..rows], kept);
        at += kept;
    }
    if (graphs) |g| captures.?.* = g.cache.count();
    return out;
}

/// Spec 028 folded the fp16 conversions and the K slices' sums into the kernels beside them. `stepped` is
/// `tokens`' logits from one-row steps with the folded launches; the same tokens as steps, as 7-row verify
/// windows and as windows of `wide` rows, each with the launches as they were and as they are, give those bits.
pub fn expectFoldedAsBefore(gpa: Allocator, m: model.Model, tokens: []const i32, stepped: []const u16, wide: usize) !void {
    defer cuda.Slices.folded = true;
    for ([_]bool{ false, true }) |folded| {
        cuda.Slices.folded = folded;
        const verify = @min(7, m.max_rows);
        for ([_]usize{ 1, verify, wide }) |window| {
            if (folded and window == 1) continue;
            var captures: usize = 0;
            // a verify window as a reply runs it, through a graph
            const got = try decodeWith(gpa, m, tokens, window, window, if (window == verify) &captures else null);
            defer gpa.free(got);
            try testing.expectEqualSlices(u16, stepped, got);
        }
    }
}

/// `owned` is `shared`'s weights loaded with `cuda.Scratch.shared` off, each linear's scratch its own as before
/// spec 033. `tokens` as verify windows with a row left over, through graphs, and `long` in windows of `wide`
/// rows give on it the bits `stepped` and `long_stepped` hold, which `shared` gave.
pub fn expectOwnedAsShared(gpa: Allocator, shared: model.Model, owned: model.Model, tokens: []const i32, stepped: []const u16, long: []const i32, long_stepped: []const u16, wide: usize) !void {
    try testing.expect(owned.scratch.bytes() > shared.scratch.bytes());
    try testing.expect(owned.scratch.lent.items.len > shared.scratch.lent.items.len);
    const verify = @min(7, owned.max_rows);
    var captures: usize = 0;
    const windows = try decodeWith(gpa, owned, tokens, verify, @max(verify, 2) - 1, &captures);
    defer gpa.free(windows);
    try testing.expectEqualSlices(u16, stepped[0..windows.len], windows);
    const prompt = try decode(gpa, owned, long, wide, wide);
    defer gpa.free(prompt);
    try testing.expectEqualSlices(u16, long_stepped[0..prompt.len], prompt);
}

/// Spec 033: a window wider than the rows each DeltaNet layer keeps runs on one projection and one scratch for
/// every layer, is committed whole and writes the states asked of it as it runs (`Snapshot.ahead`). After four
/// tokens kept whole, the next `wide` of `tokens` as such a window give the states inside it, the last row's
/// logits and the state after it that the same window gives where every layer keeps all of it.
pub fn expectWideAsKept(gpa: Allocator, m: model.Model, tokens: []const i32, wide: usize) !void {
    const capacity = 512;
    const first = 4;
    const run = tokens[first..][0..@min(wide, tokens.len - first)];
    var all_state: State = try .init(gpa, m, m.max_rows, capacity);
    defer all_state.deinit();
    var all: Buffers = try .init(gpa, m, m.max_rows, capacity);
    defer all.deinit();
    var state: State = try .initKeeping(gpa, m, m.max_rows, 1, capacity);
    defer state.deinit();
    var b: Buffers = try .initFor(gpa, m, m.layers, m.max_rows, 1, capacity);
    defer b.deinit();
    try testing.expect(widerThanKept(state, b, first) and widerThanKept(state, b, run.len));
    try testing.expect(!widerThanKept(all_state, all, run.len) and !widerThanKept(state, b, gdn.taps - 1));
    try forwardScoring(m, &all_state, &all, tokens[0..first], .none);
    try commit(m, &all_state, all, tokens[0..first], first);
    try forwardScoring(m, &state, &b, tokens[0..first], .none);
    try commit(m, &state, b, tokens[0..first], first);

    // rows whose conv inputs still hold rows from before the window, the middle, and the end
    const keeps = [_]usize{ 1, 2, 3, run.len / 2 + 1, run.len - 1, run.len };
    var asked: [keeps.len]Snapshot = undefined;
    var filled: usize = 0;
    defer for (asked[0..filled]) |*snapshot| snapshot.deinit();
    for (keeps) |keep| {
        asked[filled] = try .ahead(gpa, state, keep);
        filled += 1;
    }
    state.wanted = &asked;
    try forwardScoring(m, &state, &b, run, .last);
    state.wanted = &.{};
    for (&asked) |*snapshot| try snapshot.finish(state, b, run);
    try forwardScoring(m, &all_state, &all, run, .last);
    for (keeps, asked) |keep, got| {
        var want: Snapshot = try .within(gpa, m, all_state, all, run, keep);
        defer want.deinit();
        try expectSameSnapshot(gpa, want, got);
    }
    const vocab = m.shape.vocab;
    const rows = try gpa.alloc(u16, 2 * vocab);
    defer gpa.free(rows);
    try cuda.synchronize();
    try downloadRow(all.logits, 0, vocab, rows[0..vocab]);
    try downloadRow(b.logits, 0, vocab, rows[vocab..]);
    try testing.expectEqualSlices(u16, rows[0..vocab], rows[vocab..]);
    try commit(m, &all_state, all, run, run.len);
    try commit(m, &state, b, run, run.len);
    var want: Snapshot = try .take(gpa, all_state);
    defer want.deinit();
    var got: Snapshot = try .take(gpa, state);
    defer got.deinit();
    try expectSameSnapshot(gpa, want, got);
}

pub fn expectSameSnapshot(gpa: Allocator, want: Snapshot, got: Snapshot) !void {
    try testing.expectEqual(want.pos, got.pos);
    try cuda.synchronize();
    for ([_][2][]cuda.Buffer{ .{ want.recurrent, got.recurrent }, .{ want.conv, got.conv }, .{ want.tails, got.tails } }) |pair| {
        try testing.expectEqual(pair[0].len, pair[1].len);
        for (pair[0], pair[1]) |x, y| {
            try testing.expectEqual(x.bytes(), y.bytes());
            const bytes = try gpa.alloc(u8, 2 * x.bytes());
            defer gpa.free(bytes);
            try x.downloadAt(0, bytes[0..x.bytes()]);
            try y.downloadAt(0, bytes[x.bytes()..]);
            try testing.expect(std.mem.eql(u8, bytes[0..x.bytes()], bytes[x.bytes()..]));
        }
    }
    try testing.expectEqual(want.history.len, got.history.len);
    for (want.history, got.history) |x, y| try testing.expectEqualSlices(u32, x, y);
}

/// Spec 033's proof that a linear's scratch is dead once the kernel that reads it has run: with every buffer of
/// the model's pool filled with NaN patterns before each linear's first launch (`cuda.Scratch.poisoned`),
/// `tokens` as one-row steps and as verify windows with a row left over give `stepped`'s bits, and `long` in
/// windows of `wide` rows gives `long_stepped`'s, each eagerly and through graphs, which record the fills.
/// `stepped` and `long_stepped` may hold more rows than the tokens given.
pub fn expectPoisonedAsClean(gpa: Allocator, m: model.Model, tokens: []const i32, stepped: []const u16, long: []const i32, long_stepped: []const u16, wide: usize) !void {
    cuda.Scratch.poisoned = true;
    defer cuda.Scratch.poisoned = false;
    const verify = @min(7, m.max_rows);
    const Case = struct { tokens: []const i32, want: []const u16, window: usize, keep: usize };
    for ([_]Case{
        .{ .tokens = tokens, .want = stepped, .window = 1, .keep = 1 },
        .{ .tokens = tokens, .want = stepped, .window = verify, .keep = @max(verify, 2) - 1 },
        .{ .tokens = long, .want = long_stepped, .window = wide, .keep = wide },
    }) |case| for ([_]bool{ false, true }) |graphed| {
        var captures: usize = 0;
        const got = try decodeWith(gpa, m, case.tokens, case.window, case.keep, if (graphed) &captures else null);
        defer gpa.free(got);
        try testing.expectEqualSlices(u16, case.want[0..got.len], got);
    };
}

/// `forward_checkpoint_test.zig`'s check on a checkpoint's first layers, to its first attention layer, which
/// `checkpoint_prefix_test.zig` runs on its shared load (spec 133 WS-08 moved it here): `m` the trunk of
/// `opened`'s first layers for windows of 400 rows, MXFP8 faces and the NVFP4 table through graphs, wide windows
/// and a snapshot. A second load of the same layers lends each linear a buffer of its own.
pub fn firstLayers(gpa: Allocator, m: model.Model, opened: *const checkpoints.Held, config: qwen4_config.Config, found: layout.Layout) !void {
    try testing.expectEqual(@as(usize, 400), m.max_rows);
    const count = m.layers.len;
    var with_ple = false;
    for (m.layers) |layer| with_ple = with_ple or layer.ple != null;
    try testing.expect(with_ple and m.attention != null);

    const tokens = [_]i32{ 248045, 846, 198, 826, 279, 1118, 4097, 9944, 4947, 11, 18101, 539, 73982 };
    const eager = try decode(gpa, m, &tokens, 1, 1);
    defer gpa.free(eager);
    for (eager) |bits| try testing.expect(std.math.isFinite(numeric.f32FromBf16(bits)));
    // the table's rows come from the checkpoint's file: where it reads directly, its drive is kept awake by
    // one reader, which each step's gather has told (`weights/drive.zig`)
    var direct = false;
    for (m.layers) |layer| if (layer.ple) |branch| {
        try testing.expectEqual(m.drive, branch.table.drive);
        direct = direct or opened.checkpoint.fileOf(branch.table.shards[0]).?.direct >= 0;
    };
    try testing.expectEqual(direct, m.drive != null);
    if (m.drive) |drive| try testing.expect(drive.touched.load(.monotonic) >= tokens.len and drive.reads.load(.monotonic) >= 1);
    const Case = struct { window: usize, keep: usize };
    for ([_]Case{ .{ .window = 1, .keep = 1 }, .{ .window = 4, .keep = 3 } }) |case| {
        var captures: usize = 0;
        const graphed = try decodeWith(gpa, m, &tokens, case.window, case.keep, &captures);
        defer gpa.free(graphed);
        try testing.expectEqualSlices(u16, eager, graphed);
        try testing.expect(captures >= 2);
    }
    // A prompt's wide windows give each row the bits its one-row step gives: every kernel's wider tiles, the
    // PLE's and the attention's among them, and at 400 rows a pack's experts on their wide kernel.
    var long: [400]i32 = undefined;
    for (&long, 0..) |*token, i| token.* = @intCast(1000 + i * 37 + (i * i) % 911);
    const stepped = try decode(gpa, m, &long, 1, 1);
    defer gpa.free(stepped);
    for ([_]usize{ 400, 130, 33 }) |window| {
        const wide = try decode(gpa, m, &long, window, window);
        defer gpa.free(wide);
        try testing.expectEqualSlices(u16, stepped, wide);
    }
    // the read-outs and the router as they launched before spec 028, on these layers' own weights
    try expectFoldedAsBefore(gpa, m, &long, stepped, 320);
    // and no linear of these layers reads scratch that another's call came after
    try expectPoisonedAsClean(gpa, m, tokens[0..6], eager, &long, stepped, 320);
    try expectWideAsKept(gpa, m, &long, 320);
    // nor does a buffer lent to many linears give other bits than each linear's own did
    {
        cuda.Scratch.shared = false;
        defer cuda.Scratch.shared = true;
        var owned: model.Model = try .load(gpa, config, found, count, 320);
        defer owned.deinit();
        try expectOwnedAsShared(gpa, m, owned, &tokens, eager, &long, stepped, 320);
    }

    // A snapshot's state, after other tokens went over it, gives the rows after it as the sequence itself does.
    const vocab = m.shape.vocab;
    const row = try gpa.alloc(u16, vocab);
    defer gpa.free(row);
    var state: State = try .init(gpa, m, m.max_rows, 64);
    defer state.deinit();
    var b: Buffers = try .init(gpa, m, m.max_rows, 64);
    defer b.deinit();
    const at = 6;
    try forward(m, &state, &b, tokens[0..4]);
    try commit(m, &state, b, tokens[0..4], 4);
    try forward(m, &state, &b, tokens[4..at]);
    try commit(m, &state, b, tokens[4..at], at - 4);
    var snapshot: Snapshot = try .take(gpa, state);
    defer snapshot.deinit();
    try testing.expect(snapshot.tails.len > 0 and snapshot.bytes() > 0);
    const other = [_]i32{ 9944, 198, 198, 846, 11 };
    try forward(m, &state, &b, other[0..3]);
    try commit(m, &state, b, other[0..3], 3);
    try forward(m, &state, &b, other[3..]);
    try commit(m, &state, b, other[3..], 2);
    try snapshot.restore(&state);
    try testing.expectEqual(at, state.pos);
    try forward(m, &state, &b, tokens[at..][0..4]);
    try cuda.synchronize();
    for (0..4) |r| {
        try downloadRow(b.logits, r, vocab, row);
        try testing.expectEqualSlices(u16, eager[(at + r) * vocab ..][0..vocab], row);
    }

    // And a snapshot taken from inside a window that is then committed whole (`Snapshot.within`): the state at
    // its row, the DeltaNet states replayed and the conv inputs, PLE tails and histories slid to it.
    try state.reset();
    try forward(m, &state, &b, tokens[0..4]);
    try commit(m, &state, b, tokens[0..4], 4);
    try forward(m, &state, &b, tokens[4..11]);
    var inside: Snapshot = try .within(gpa, m, state, b, tokens[4..11], at - 4);
    defer inside.deinit();
    var whole: Snapshot = try .within(gpa, m, state, b, tokens[4..11], 7);
    defer whole.deinit();
    // and each written over a snapshot of the other row (`Snapshot.again`), in the buffers that one had
    var over: Snapshot = try .within(gpa, m, state, b, tokens[4..11], 7);
    defer over.deinit();
    var back: Snapshot = try .within(gpa, m, state, b, tokens[4..11], at - 4);
    defer back.deinit();
    const held = .{ over.recurrent[0].view.ptr, over.conv[0].view.ptr, over.tails[0].view.ptr, over.bytes() };
    try over.again(m, state, b, tokens[4..11], at - 4);
    try back.again(m, state, b, tokens[4..11], 7);
    try testing.expectEqual(held, .{ over.recurrent[0].view.ptr, over.conv[0].view.ptr, over.tails[0].view.ptr, over.bytes() });
    try testing.expectEqual(at, over.pos);
    try commit(m, &state, b, tokens[4..11], 7);
    try forward(m, &state, &b, other[0..3]);
    try commit(m, &state, b, other[0..3], 3);
    for ([_]Snapshot{ inside, whole, over, back }) |kept| {
        try kept.restore(&state);
        try testing.expectEqual(kept.pos, state.pos);
        const rest = tokens[kept.pos..];
        try forward(m, &state, &b, rest);
        try cuda.synchronize();
        for (0..rest.len) |r| {
            try downloadRow(b.logits, r, vocab, row);
            try testing.expectEqualSlices(u16, eager[(kept.pos + r) * vocab ..][0..vocab], row);
        }
    }
    try testing.expectEqual(at, inside.pos);
    try testing.expectEqual(11, whole.pos);
}
