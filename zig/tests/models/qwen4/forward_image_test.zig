//! `forward.zig`'s image prompt (spec 019 WS-03) on `testing.zig`'s tiny model (a DeltaNet layer with a PLE,
//! then an attention layer), random weights and random features: the placeholder rows take the features before
//! the PLE and rotate by the image's positions, and a row's bits are the same in serial steps, in one window, in
//! windows of 16 and through graphs; a text prompt after it is a fresh state's.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("testing.zig");
const tinyModel = fixtures.tiny_model.tinyModel;
const tiny = fixtures.tiny_model.tiny;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const forward_file = @import("../../../src/models/qwen4/forward.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const Buffers = forward_file.Buffers;
const Graphs = forward_file.Graphs;
const State = forward_file.State;
const commit = forward_file.commit;
const downloadRow = forward_file.downloadRow;
const forward = forward_file.forward;

const capacity = 128;
const total = 34;
/// Rows 4 to 15 are the placeholders of a 3 x 4 grid.
const before = 4;
const grid_h = 3;
const grid_w = 4;

const Prompt = struct {
    tokens: [total]i32,
    positions: [total][3]i32,
    delta: i32,
    rows: [grid_h * grid_w]u32,
    features: cuda.Buffer,
};

fn prompt(random: std.Random) !Prompt {
    var p: Prompt = undefined;
    for (&p.tokens) |*token| token.* = random.intRangeLessThan(i32, 0, @intCast(tiny.shape.vocab));
    for (p.positions[0..before], 0..) |*q, i| q.* = @splat(@intCast(i));
    for (0..grid_h) |y| for (0..grid_w) |x| {
        p.positions[before + y * grid_w + x] = .{ before, before + @as(i32, @intCast(y)), before + @as(i32, @intCast(x)) };
    };
    const next = before + @max(grid_h, grid_w);
    for (p.positions[before + grid_h * grid_w ..], 0..) |*q, i| q.* = @splat(next + @as(i32, @intCast(i)));
    p.delta = next - (before + grid_h * grid_w);
    for (&p.rows, 0..) |*row, i| row.* = @intCast(before + i);
    var bits: [grid_h * grid_w * tiny.shape.hidden]u16 = undefined;
    for (&bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32));
    p.features = try .init(.bf16, &.{ grid_h * grid_w, tiny.shape.hidden });
    try p.features.upload(std.mem.sliceAsBytes(&bits));
    return p;
}

/// Every row's logits of `tokens` in windows of `window` rows (the image attached when `image`), eager or
/// through `graphs`.
fn decode(gpa: Allocator, m: model.Model, state: *State, b: *Buffers, p: *const Prompt, image: bool, window: usize, graphs: ?*Graphs) ![]u16 {
    const vocab = m.shape.vocab;
    if (image) try state.attach(.{ .features = p.features.view, .rows = &p.rows }, &p.positions, p.delta);
    const out = try gpa.alloc(u16, total * vocab);
    errdefer gpa.free(out);
    var at: usize = 0;
    while (at < total) {
        const rows = @min(window, total - at);
        const tokens = p.tokens[at..][0..rows];
        if (graphs) |g| {
            // a window over a placeholder is never one a graph holds
            const placeholder = image and at < before + grid_h * grid_w and at + rows > before;
            if (placeholder) try testing.expect(!g.holds(state.*, b.*, rows));
            const captured = g.cache.count();
            try g.forward(m, state, b, tokens);
            if (placeholder) try testing.expectEqual(captured, g.cache.count());
        } else try forward(m, state, b, tokens);
        try cuda.synchronize();
        for (0..rows) |r| try downloadRow(b.logits, r, vocab, out[(at + r) * vocab ..][0..vocab]);
        try commit(m, state, b.*, tokens, rows);
        at += rows;
    }
    // the features are let go once the last placeholder is committed
    try testing.expect(state.image == null);
    return out;
}

fn fresh(gpa: Allocator, m: model.Model, p: *const Prompt, image: bool, window: usize, graphed: bool) ![]u16 {
    var state: State = try .init(gpa, m, m.max_rows, capacity);
    defer state.deinit();
    var b: Buffers = try .init(gpa, m, m.max_rows, capacity);
    defer b.deinit();
    var graphs: Graphs = try .init(gpa);
    defer graphs.deinit();
    return decode(gpa, m, &state, &b, p, image, window, if (graphed) &graphs else null);
}

test "an image prompt's rows: serial steps, one window, windows of 16 and graphs give the same bits, and the image is read" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try tinyModel(gpa, arena.allocator());
    defer m.deinit();
    // sections that end inside the 8 pairs: pairs 1 and 4 take h, 2 takes w
    m.attention.?.shape.sections = .{ 2, 2, 1 };
    var prng: std.Random.DefaultPrng = .init(19);
    var p = try prompt(prng.random());
    defer p.features.deinit();
    const vocab = m.shape.vocab;

    const serial = try fresh(gpa, m, &p, true, 1, false);
    defer gpa.free(serial);
    for ([_]usize{ total, 16, 5 }) |window| {
        const got = try fresh(gpa, m, &p, true, window, false);
        defer gpa.free(got);
        try testing.expectEqualSlices(u16, serial, got);
    }
    for ([_]usize{ 1, 4 }) |window| {
        const got = try fresh(gpa, m, &p, true, window, true);
        defer gpa.free(got);
        try testing.expectEqualSlices(u16, serial, got);
    }

    // the rows before the image are text's; every row from the first placeholder on is not
    const text = try fresh(gpa, m, &p, false, 1, false);
    defer gpa.free(text);
    try testing.expectEqualSlices(u16, text[0 .. before * vocab], serial[0 .. before * vocab]);
    for (before..total) |r| try testing.expect(!std.mem.eql(u16, text[r * vocab ..][0..vocab], serial[r * vocab ..][0..vocab]));
}

test "a text prompt after an image prompt, on the same state after reset, is a fresh state's" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try tinyModel(gpa, arena.allocator());
    defer m.deinit();
    var prng: std.Random.DefaultPrng = .init(20);
    var p = try prompt(prng.random());
    defer p.features.deinit();

    const text = try fresh(gpa, m, &p, false, 4, true);
    defer gpa.free(text);
    var state: State = try .init(gpa, m, m.max_rows, capacity);
    defer state.deinit();
    var b: Buffers = try .init(gpa, m, m.max_rows, capacity);
    defer b.deinit();
    var graphs: Graphs = try .init(gpa);
    defer graphs.deinit();
    const image = try decode(gpa, m, &state, &b, &p, true, 4, &graphs);
    defer gpa.free(image);
    try state.reset();
    const after = try decode(gpa, m, &state, &b, &p, false, 4, &graphs);
    defer gpa.free(after);
    try testing.expectEqualSlices(u16, text, after);
}

test "attach refuses an image that does not fit its prompt" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try tinyModel(gpa, arena.allocator());
    defer m.deinit();
    var prng: std.Random.DefaultPrng = .init(21);
    var p = try prompt(prng.random());
    defer p.features.deinit();
    var state: State = try .init(gpa, m, m.max_rows, capacity);
    defer state.deinit();
    const features = p.features.view;
    // rows out of order, a row past the positions, fewer features than rows
    try testing.expectError(error.InvalidImage, state.attach(.{ .features = features, .rows = &.{ 5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 } }, &p.positions, p.delta));
    try testing.expectError(error.InvalidImage, state.attach(.{ .features = features, .rows = &.{ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, total } }, &p.positions, p.delta));
    try testing.expectError(error.InvalidImage, state.attach(.{ .features = features, .rows = &.{ 4, 5 } }, &p.positions, p.delta));
}
