//! Image prompts in the pool (spec 019 WS-05): an image seat beside text seats gives its serial reply bit for
//! bit, and the text seats theirs; it goes on from a text state at or below its first placeholder row and
//! keeps none above it, so another image of the same grid and tokens is never served another's rows; alone in
//! the pool it gives a session's tokens and drafts; and spec 086's round one row short of the wide kernels
//! holds with an image prompt as its chunk. The features are the test's (`hall.Sight` without a tower).

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const hall_fixture = @import("hall_fixture.zig");
const greedy = hall_fixture.greedy;
const Collect = hall_fixture.Collect;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const gdn = @import("../../../../src/models/qwen4/blocks/gdn.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const session = @import("../../../../src/models/qwen4/session/session.zig");
const Engine = hall_engine.Engine;
const Ask = hall_file.Ask;
const Out = hall_file.Out;

const context: usize = 256;
/// Rows 4 to 15 of a prompt are the placeholders of a 3 x 4 grid, as `forward_image_test.zig`'s.
const before = 4;
const grid_h = 3;
const grid_w = 4;
const placeholders = grid_h * grid_w;

/// A prompt of `len` tokens with its image's positions and features.
const Image = struct {
    tokens: []i32,
    positions: [][3]i32,
    delta: i32,
    rows: [placeholders]u32,
    bits: [placeholders * fixtures.tiny_model.tiny.shape.hidden]u16,
    features: cuda.Buffer,

    fn init(gpa: Allocator, random: std.Random, len: usize, vocab: usize) !Image {
        std.debug.assert(len > before + placeholders);
        var self: Image = undefined;
        self.tokens = try gpa.alloc(i32, len);
        errdefer gpa.free(self.tokens);
        self.positions = try gpa.alloc([3]i32, len);
        errdefer gpa.free(self.positions);
        for (self.tokens) |*token| token.* = random.intRangeLessThan(i32, 1, @intCast(vocab));
        for (self.positions[0..before], 0..) |*q, i| q.* = @splat(@intCast(i));
        for (0..grid_h) |y| for (0..grid_w) |x| {
            self.positions[before + y * grid_w + x] = .{ before, before + @as(i32, @intCast(y)), before + @as(i32, @intCast(x)) };
        };
        const next = before + @max(grid_h, grid_w);
        for (self.positions[before + placeholders ..], 0..) |*q, i| q.* = @splat(next + @as(i32, @intCast(i)));
        self.delta = next - (before + placeholders);
        for (&self.rows, 0..) |*row, i| row.* = @intCast(before + i);
        for (&self.bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32));
        self.features = try .init(.bf16, &.{ placeholders, fixtures.tiny_model.tiny.shape.hidden });
        try self.features.upload(std.mem.sliceAsBytes(&self.bits));
        return self;
    }

    fn deinit(self: *Image, gpa: Allocator) void {
        self.features.deinit();
        gpa.free(self.positions);
        gpa.free(self.tokens);
    }

    fn picture(self: *const Image) hall_file.Picture {
        return .{ .pixels = &.{}, .grids = &.{}, .rows = &self.rows, .positions = self.positions, .delta = self.delta };
    }

    /// Its features where the pool reads them: what the tower would have written there.
    fn show(self: *const Image, sight: *hall_file.Sight) !void {
        try sight.features.uploadAt(0, std.mem.sliceAsBytes(&self.bits));
    }

    fn forState(self: *const Image) forward.Image {
        return .{ .features = self.features.view, .rows = &self.rows };
    }
};

/// The reply to an image prompt by one-row steps from an empty state, `fixtures.serial.serialReply`'s way.
fn serialImageReply(gpa: Allocator, m: model_file.Model, image: *const Image, count: usize) ![]i32 {
    var state: forward.State = try .init(gpa, m, m.max_rows, context);
    defer state.deinit();
    var b: forward.Buffers = try .init(gpa, m, m.max_rows, context);
    defer b.deinit();
    var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, m.shape.vocab);
    defer gpa.free(logits);
    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    try state.attach(image.forState(), image.positions, image.delta);
    for (image.tokens) |token| {
        try forward.forward(m, &state, &b, &.{token});
        try forward.commit(m, &state, b, &.{token}, 1);
    }
    try testing.expect(state.image == null);
    for (out, 0..) |*token, made| {
        try forward.logitsRow(b, 0, logits);
        token.* = try sampler.choose(logits, image.tokens.len + made, greedy);
        try forward.forward(m, &state, &b, &.{token.*});
        try forward.commit(m, &state, b, &.{token.*}, 1);
    }
    return out;
}

/// Its pool is bound by the caller, where the engine lies.
fn engineWith(gpa: Allocator, m: model_file.Model, sight: *hall_file.Sight, slots: usize, chunk: usize) !Engine {
    return Engine.open(gpa, m, &.{}, .{
        .slots = slots,
        .session = .{ .capacity = context, .prompt_rows = 32 },
        .pool = extents.lowerBound(context, slots) * extents.positions,
        .chunk = chunk,
        .lazy_graphs = false,
        .sight = sight,
    });
}

/// Rounds until every slot of `slots` has ended.
fn drain(engine: *Engine, collect: *Collect, slots: []const usize) !void {
    const out: Out = .{ .ctx = collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail };
    var spins: usize = 0;
    while (spins < 200) : (spins += 1) {
        const done = for (slots) |slot| {
            if (!collect.ended[slot]) break false;
        } else true;
        if (done) break;
        switch (engine.round(out)) {
            .ran => engine.publishAhead(out),
            .short => return error.Short,
            .failed => return error.Failed,
        }
    }
    for (slots) |slot| try testing.expect(collect.ended[slot]);
}

fn askOf(engine: *const Engine, prompt: []const i32, count: usize, starts: []const usize, image: ?hall_file.Picture) Ask {
    return .{ .prompt = prompt, .settings = greedy, .max_tokens = count, .depth = engine.depth, .opening = engine.depth, .starts = starts, .image = image };
}

test "an image prompt in the pool beside text replies is its serial reply bit for bit; it keeps no state above its first image row, and another image of its tokens goes on from the text state below" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    // the random model: the tiny one's greedy tokens do not move with an image's features
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    // the trunk's attention layer rotates the image rows by all three axes
    m.attention.?.shape.sections = .{ 2, 2, 1 };
    var prng: std.Random.DefaultPrng = .init(195);
    const random = prng.random();
    const count: usize = 8;
    var first = try Image.init(gpa, random, 30, 250);
    defer first.deinit(gpa);
    // the same tokens and grid, other features
    var second = try Image.init(gpa, random, 30, 250);
    defer second.deinit(gpa);
    @memcpy(second.tokens, first.tokens);
    const texts = [_][]const i32{ &.{ 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }, &.{ 30, 31, 32 } };

    const serial_first = try serialImageReply(gpa, m, &first, count);
    defer gpa.free(serial_first);
    const serial_second = try serialImageReply(gpa, m, &second, count);
    defer gpa.free(serial_second);
    // the image is read: its tokens alone, as text, reply otherwise
    const as_text = try fixtures.serial.serialReply(gpa, m, first.tokens, count, greedy);
    defer gpa.free(as_text);
    try testing.expect(!std.mem.eql(i32, as_text, serial_first));
    var serial_texts: [2][]i32 = undefined;
    serial_texts[0] = try fixtures.serial.serialReply(gpa, m, texts[0], count, greedy);
    defer gpa.free(serial_texts[0]);
    serial_texts[1] = try fixtures.serial.serialReply(gpa, m, texts[1], count, greedy);
    defer gpa.free(serial_texts[1]);

    var sight: hall_file.Sight = try .init(gpa, null, placeholders, fixtures.tiny_model.tiny.shape.hidden, 3, context);
    defer sight.deinit();
    var engine = try engineWith(gpa, m, &sight, 3, 8);
    defer engine.deinit();
    engine.bindPool();
    var lists: [3]std.ArrayList(i32) = @splat(.empty);
    defer for (&lists) |*list| list.deinit(gpa);
    var ended: [3]bool = @splat(false);
    var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended };

    // two text seats and the image seat, whose prompt fills in chunks of 8 beside their replies; its message
    // starts at 3, below the image, where its one state is kept
    try first.show(&sight);
    _ = try engine.begin(0, askOf(&engine, texts[0], count, &.{}, null), 0);
    _ = try engine.begin(1, askOf(&engine, texts[1], count, &.{}, null), 0);
    try testing.expectEqual(0, try engine.begin(2, askOf(&engine, first.tokens, count, &.{3}, first.picture()), 0));
    const chain = engine.seats[2].?.chain;
    try drain(&engine, &collect, &.{ 0, 1, 2 });
    try testing.expectEqualSlices(i32, serial_texts[0], lists[0].items);
    try testing.expectEqualSlices(i32, serial_texts[1], lists[1].items);
    try testing.expectEqualSlices(i32, serial_first, lists[2].items);
    for (0..3) |slot| _ = engine.retire(slot, .ended, count);
    var below = false;
    for (engine.keeps.list.items) |keep| if (keep.chain == chain) {
        try testing.expect(keep.pos <= before);
        below = below or keep.pos == 3;
    };
    try testing.expect(below);

    // the second image goes on from the text state at 3, and reads its own features
    try second.show(&sight);
    for (&lists) |*list| list.clearRetainingCapacity();
    ended = @splat(false);
    try testing.expectEqual(3, try engine.begin(0, askOf(&engine, second.tokens, count, &.{3}, second.picture()), 0));
    try drain(&engine, &collect, &.{0});
    try testing.expectEqualSlices(i32, serial_second, lists[0].items);
    _ = engine.retire(0, .ended, count);

    // and a text prompt after it on the same slot is at its own positions again
    for (&lists) |*list| list.clearRetainingCapacity();
    ended = @splat(false);
    _ = try engine.begin(0, askOf(&engine, texts[0], count, &.{}, null), 0);
    try drain(&engine, &collect, &.{0});
    try testing.expectEqualSlices(i32, serial_texts[0], lists[0].items);
    _ = engine.retire(0, .ended, count);
}

test "an image prompt alone in the pool gives the session's tokens and drafts: the head's rows read the slot's positions" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 48);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    var prng: std.Random.DefaultPrng = .init(196);
    const count: usize = 12;
    var image = try Image.init(gpa, prng.random(), 24, 250);
    defer image.deinit(gpa);
    const serial = try serialImageReply(gpa, m, &image, count);
    defer gpa.free(serial);

    const alone: *session.Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .capacity = context, .prompt_rows = 32 });
    defer alone.destroy();
    var record: [count]i32 = undefined;
    var counting: fixtures.sessions.Counting = .{};
    const from_session = try alone.reply(image.tokens, greedy, .{ .max_tokens = count, .record = &record, .image = .{ .image = image.forState(), .positions = image.positions, .delta = image.delta } }, counting.sink());
    try testing.expectEqualSlices(i32, serial, record[0..from_session.tokens]);
    try testing.expect(from_session.stats.drafted > 0);

    var sight: hall_file.Sight = try .init(gpa, null, placeholders, fixtures.tiny_model.tiny.shape.hidden, 2, context);
    defer sight.deinit();
    try image.show(&sight);
    var engine = try engineWith(gpa, m, &sight, 2, 8);
    defer engine.deinit();
    engine.bindPool();
    var lists: [2]std.ArrayList(i32) = @splat(.empty);
    defer for (&lists) |*list| list.deinit(gpa);
    var ended: [2]bool = @splat(false);
    var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended };
    _ = try engine.begin(1, .{ .prompt = image.tokens, .settings = greedy, .max_tokens = count, .depth = session.default_depth, .opening = session.default_depth, .image = image.picture() }, 0);
    try drain(&engine, &collect, &.{1});
    try testing.expectEqualSlices(i32, serial, lists[1].items);
    const report = engine.retire(1, .ended, count).?;
    try testing.expectEqual(from_session.drafts, report.drafts);
}

test "an image prompt as the chunk one row short of the wide kernels beside three full windows: within each layer's short projection, every reply the serial one (spec 086)" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 128);
    defer m.deinit();
    // no head: the windows' drafts are the test's, as `hall_test.zig`'s spec 086 test
    const head = m.mtp;
    m.mtp = null;
    defer m.mtp = head;
    const count: usize = 40;
    var prng: std.Random.DefaultPrng = .init(86);
    const random = prng.random();
    var tokens: [3][5]i32 = undefined;
    for (&tokens) |*row| {
        for (row) |*token| token.* = random.intRangeLessThan(i32, 1, 250);
    }
    const chunk = gdn.wide_rows - 1;
    var image = try Image.init(gpa, random, chunk, 250);
    defer image.deinit(gpa);
    const prompts = [_][]const i32{ tokens[0][0..3], tokens[1][0..4], tokens[2][0..5] };
    var serial: [4][]i32 = undefined;
    var made: usize = 0;
    defer for (serial[0..made]) |reply| gpa.free(reply);
    for (prompts) |prompt| {
        serial[made] = try fixtures.serial.serialReply(gpa, m, prompt, count, greedy);
        made += 1;
    }
    serial[3] = try serialImageReply(gpa, m, &image, count);
    made += 1;

    var sight: hall_file.Sight = try .init(gpa, null, placeholders, fixtures.tiny_model.tiny.shape.hidden, 4, context);
    defer sight.deinit();
    try image.show(&sight);
    var engine = try engineWith(gpa, m, &sight, 4, 64);
    defer engine.deinit();
    engine.bindPool();
    var lists: [4]std.ArrayList(i32) = @splat(.empty);
    defer for (&lists) |*list| list.deinit(gpa);
    var ended: [4]bool = @splat(false);
    var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended };
    const out: Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail };
    for (prompts, 0..) |prompt, i| _ = try engine.begin(i, askOf(&engine, prompt, count, &.{}, null), 0);
    var begun = false;
    var widest: usize = 0;
    var window: usize = 0;
    for (0..200) |spin| {
        if (begun and std.mem.allEqual(bool, &ended, true)) break;
        var decoding: usize = 0;
        for (engine.seats) |*maybe| {
            const seat = if (maybe.*) |*held| held else continue;
            if (seat.phase != .decode) continue;
            decoding += 1;
            for (&seat.drafts, 0..) |*draft, d| draft.* = @intCast(1 + (spin * 7 + d * 13) % 249);
            seat.draft_n = seat.drafts.len - 1;
            window = seat.drafts.len;
        }
        if (!begun and decoding == 3) {
            _ = try engine.begin(3, askOf(&engine, image.tokens, count, &.{}, image.picture()), 0);
            begun = true;
        }
        switch (engine.round(out)) {
            .ran => engine.publishAhead(out),
            .short => return error.Short,
            .failed => return error.Failed,
        }
        widest = @max(widest, engine.staged.used);
    }
    try testing.expectEqual(3 * window + chunk, widest);
    try testing.expect(widest <= engine.buffers.kept);
    for (serial, lists, ended, 0..) |want, got, done, i| {
        try testing.expect(done);
        try testing.expectEqualSlices(i32, want, got.items);
        _ = engine.retire(i, .ended, count);
    }
}
