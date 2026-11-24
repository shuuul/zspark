//! `draft_test.zig`'s drafted replies, which `checkpoint_prefix_test.zig` and `forward_checkpoint_test.zig` also
//! run on the checkpoint loads they share (spec 133 WS-08 moved them here): drafts from the serial reply, right
//! ones, wrong ones at chosen places, none.

const std = @import("std");
const fixtures = @import("testing.zig");
const cuda = @import("../../../src/cuda/cuda.zig");
const masks_file = @import("../../../src/decode/masks.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const draft_file = @import("../../../src/models/qwen4/draft.zig");
const forward = @import("../../../src/models/qwen4/forward.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const Decoder = draft_file.Decoder;
const Drafter = draft_file.Drafter;
const Stats = draft_file.Stats;
const forward_max_rows = draft_file.forward_max_rows;
const serialReply = fixtures.serial.serialReply;

/// Drafts from a reply known beforehand, spoiled where `wrong` says.
pub const Oracle = struct {
    /// `reply[i]` sits at position `first + i`.
    reply: []const i32,
    first: usize,
    /// Drafts a call gives, at most.
    count: usize,
    /// Whether the draft for `position` is spoiled.
    wrong: *const fn (position: usize) bool,
    calls: usize = 0,

    pub fn drafter(self: *Oracle) Drafter {
        return .{ .context = self, .draft = draft };
    }

    fn draft(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, _: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Oracle = @ptrCast(@alignCast(context));
        self.calls += 1;
        try testing.expectEqual(@as(i64, @intCast(tokens.len)), streams.size[0]);
        // the newest token is where the caller says
        try testing.expectEqual(self.reply[position - self.first], tokens[tokens.len - 1]);
        var n: usize = 0;
        while (n < @min(out.len, self.count)) : (n += 1) {
            const at = position + 1 + n;
            if (at - self.first >= self.reply.len) break;
            const right = self.reply[at - self.first];
            out[n] = if (self.wrong(at)) @mod(right + 7, 300) else right;
        }
        return n;
    }
};

const draft_fixture = @import("../../decode/draft_fixture.zig");
const never = draft_fixture.never;
const always = draft_fixture.always;
const everyThird = draft_fixture.everyThird;
const everyFifth = draft_fixture.everyFifth;

pub const Reply = struct { tokens: []i32, stats: Stats };

/// `count` tokens after `prompt`: serial steps without an oracle, drafted rounds with one.
pub fn reply(gpa: Allocator, m: model.Model, prompt: []const i32, count: usize, settings: sampling.Sampling, eos: []const u32, oracle: ?*Oracle, graphed: bool) !Reply {
    return replyWith(gpa, m, prompt, count, settings, eos, if (oracle) |o| o.drafter() else null, graphed, null, null);
}

/// `reply` under `masks`, where that is given, with any drafter.
pub fn replyWith(gpa: Allocator, m: model.Model, prompt: []const i32, count: usize, settings: sampling.Sampling, eos: []const u32, drafter: ?Drafter, graphed: bool, masks: ?*masks_file.Masks, follow: ?draft_file.Follow) !Reply {
    var state: forward.State = try .init(gpa, m, m.max_rows, 256);
    defer state.deinit();
    var b: forward.Buffers = try .init(gpa, m, m.max_rows, 256);
    defer b.deinit();
    var graphs: forward.Graphs = try .init(gpa);
    defer graphs.deinit();
    var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, m.shape.vocab);
    defer gpa.free(logits);
    var decoder: Decoder = .{ .m = m, .state = &state, .b = &b, .graphs = if (graphed) &graphs else null, .sampler = &sampler, .settings = settings, .eos = eos, .logits = logits, .masks = masks, .follow = follow };

    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    var at: usize = 0;
    while (at < prompt.len) {
        const rows = @min(m.max_rows, prompt.len - at);
        try forward.forward(m, &state, &b, prompt[at..][0..rows]);
        if (at + rows == prompt.len) try forward.logitsRow(b, rows - 1, logits);
        try forward.commit(m, &state, b, prompt[at..][0..rows], rows);
        at += rows;
    }
    // the reply's first token, as `Decoder.prompt` chooses it
    if (masks) |held| {
        _ = try held.window(&.{});
        try held.apply(0, logits);
    }
    out[0] = try sampler.choose(logits, prompt.len, settings);
    if (follow) |text| out[0] = text.at(prompt.len, out[0]);
    if (masks) |held| try held.commit(1, out[0]);
    var made: usize = 1;
    var drafts: [forward_max_rows]i32 = undefined;
    var chosen: [forward_max_rows]i32 = undefined;
    // the prompt's last row is the only one kept so far whose choice is the newest token
    var kept: []const i32 = out[0..1];
    while (made < count and std.mem.indexOfScalar(u32, eos, @intCast(out[made - 1])) == null) {
        const n = if (drafter) |d| try decoder.draft(d, kept, count - made - 1, &drafts) else 0;
        const keep = try decoder.round(out[made - 1], drafts[0..n], &chosen);
        @memcpy(out[made..][0..keep], chosen[0..keep]);
        made += keep;
        kept = out[made - keep .. made];
    }
    return .{ .tokens = try gpa.realloc(out, made), .stats = decoder.stats };
}

/// The decoder's reply to `prompt` on `m` in windows of `m.max_rows` rows is the serial one: with no drafts, with
/// an oracle's right, wrong and mixed, eager and through graphs, and a reply that ends early.
pub fn expectDrafted(gpa: Allocator, m: model.Model, prompt: []const i32, count: usize, settings: sampling.Sampling) !void {
    const none: []const u32 = &.{};
    const serial: Reply = .{ .tokens = try serialReply(gpa, m, prompt, count, settings), .stats = .{ .rounds = count - 1 } };
    defer gpa.free(serial.tokens);
    // the decoder with no drafts is serial decoding
    const undrafted = try reply(gpa, m, prompt, count, settings, none, null, false);
    defer gpa.free(undrafted.tokens);
    try testing.expectEqualSlices(i32, serial.tokens, undrafted.tokens);
    try testing.expectEqual(count - 1, undrafted.stats.rounds);
    // the reply is not one token over and over, or a spoiled draft could be right
    var distinct: usize = 0;
    for (serial.tokens[1..], serial.tokens[0 .. count - 1]) |a, b| distinct += @intFromBool(a != b);
    try testing.expect(distinct > count / 2);

    const depth = m.max_rows - 1;
    const Case = struct { wrong: *const fn (usize) bool, count: usize, graphed: bool };
    for ([_]Case{
        .{ .wrong = never, .count = depth, .graphed = true },
        .{ .wrong = never, .count = 1, .graphed = false },
        .{ .wrong = always, .count = depth, .graphed = true },
        .{ .wrong = everyThird, .count = depth, .graphed = true },
        .{ .wrong = everyFifth, .count = depth, .graphed = false },
        .{ .wrong = everyFifth, .count = 2, .graphed = true },
    }) |case| {
        var oracle: Oracle = .{ .reply = serial.tokens, .first = prompt.len, .count = case.count, .wrong = case.wrong };
        const drafted = try reply(gpa, m, prompt, count, settings, none, &oracle, case.graphed);
        defer gpa.free(drafted.tokens);
        try testing.expectEqualSlices(i32, serial.tokens, drafted.tokens);
        try testing.expectEqual(drafted.stats.rounds, oracle.calls);
        if (case.wrong == &never) {
            // every draft is kept, so a round makes one token more than it drafts
            try testing.expectEqual(drafted.stats.drafted, drafted.stats.accepted);
            try testing.expectEqual(count - 1, drafted.stats.rounds + drafted.stats.accepted);
            try testing.expect(drafted.stats.rounds < serial.stats.rounds);
        } else if (case.wrong == &always) {
            try testing.expectEqual(@as(usize, 0), drafted.stats.accepted);
            try testing.expectEqual(serial.stats.rounds, drafted.stats.rounds);
        } else {
            try testing.expect(drafted.stats.accepted > 0 and drafted.stats.accepted < drafted.stats.drafted);
            try testing.expectEqual(count - 1, drafted.stats.rounds + drafted.stats.accepted);
        }
    }

    // a reply that ends: the row that chooses the end token is the last kept, whatever was drafted after it
    const end = [1]u32{@intCast(serial.tokens[count / 2])};
    const ended = try reply(gpa, m, prompt, count, settings, &end, null, false);
    defer gpa.free(ended.tokens);
    try testing.expect(ended.tokens.len < count);
    try testing.expectEqualSlices(i32, serial.tokens[0..ended.tokens.len], ended.tokens);
    var oracle: Oracle = .{ .reply = serial.tokens, .first = prompt.len, .count = depth, .wrong = never };
    const drafted = try reply(gpa, m, prompt, count, settings, &end, &oracle, true);
    defer gpa.free(drafted.tokens);
    try testing.expectEqualSlices(i32, ended.tokens, drafted.tokens);
}

/// The checkpoint's prompt and sampled reply `draft_test.zig`'s check on its first layers ran: a few layers of the
/// model repeat themselves greedily, so only a sampled reply is checked, on `m` narrowed to windows of four rows.
pub fn checkpointDrafted(gpa: Allocator, m: model.Model) !void {
    const prompt = [_]i32{ 248045, 846, 198, 826, 279, 1118, 4097 };
    var four = fixtures.checkpoints.narrowed(m, 4);
    // the trunk alone, as the check's own load had it (`.mtp = false`): the oracle drafts
    four.mtp = null;
    try expectDrafted(gpa, four, &prompt, 24, .{ .seed = 5, .temperature = 1.2, .top_k = 60, .top_p = 0.98, .min_p = 0 });
}
