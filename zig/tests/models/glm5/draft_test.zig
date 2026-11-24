//! `draft.zig`'s tests: the shared decoder over both ranks of a random GLM (`testing.zig`'s `Fixture`)
//! against serial steps, with drafts a test makes up from the serial reply: right ones, wrong ones at chosen
//! places, none. `reply` is also what `mtp_test.zig` runs the MTP layer's drafts through.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const masks_file = @import("../../../src/decode/masks.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const draft_file = @import("../../../src/models/glm5/draft.zig");
const loopback = @import("../../../src/models/glm5/layers/loopback.zig");
const random_file = @import("../../../src/models/glm5/checks/random.zig");
const shared = @import("testing.zig");
const Drafter = draft_file.Drafter;
const forward_max_rows = draft_file.forward_max_rows;
const Fixture = shared.Fixture;
const Loop = shared.Loop;
const reply = shared.reply;
const fx = random_file.fx;

/// Drafts from a reply known beforehand, spoiled where `wrong` says.
const Oracle = struct {
    /// `reply[i]` sits at position `first + i`.
    reply: []const i32,
    first: usize,
    /// Drafts a call gives, at most.
    count: usize,
    /// Whether the draft for `position` is spoiled.
    wrong: *const fn (position: usize) bool,
    calls: usize = 0,

    fn drafter(self: *Oracle) Drafter {
        return .{ .context = self, .draft = draft };
    }

    fn draft(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, _: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Oracle = @ptrCast(@alignCast(context));
        try testing.expectEqual(@as(i64, @intCast(tokens.len)), streams.size[0]);
        // a prompt's window before its last: absorbed, nothing drafted
        if (position < self.first) return 0;
        self.calls += 1;
        // the newest token is where the caller says
        try testing.expectEqual(self.reply[position - self.first], tokens[tokens.len - 1]);
        var n: usize = 0;
        while (n < @min(out.len, self.count)) : (n += 1) {
            const at = position + 1 + n;
            if (at - self.first >= self.reply.len) break;
            const right = self.reply[at - self.first];
            out[n] = if (self.wrong(at)) @mod(right + 7, fx.vocab) else right;
        }
        return n;
    }
};

const draft_fixture = @import("../../decode/draft_fixture.zig");
const never = draft_fixture.never;
const always = draft_fixture.always;
const everyThird = draft_fixture.everyThird;
const everyFifth = draft_fixture.everyFifth;

fn expectDrafted(gpa: Allocator, loop: *Loop, prompt: []const i32, count: usize, settings: sampling.Sampling) !void {
    const none: []const u32 = &.{};
    const serial = try shared.serialReply(gpa, loop.m, fx.vocab, prompt, count, settings);
    defer gpa.free(serial);
    // the decoder with no drafts is serial decoding, its prompt in windows
    const undrafted = try reply(gpa, loop, prompt, count, settings, none, null, null);
    defer gpa.free(undrafted.tokens);
    try testing.expectEqualSlices(i32, serial, undrafted.tokens);
    try testing.expectEqual(count - 1, undrafted.stats.rounds);
    // the reply is not one token over and over, or a spoiled draft could be right
    var distinct: usize = 0;
    for (serial[1..], serial[0 .. count - 1]) |a, b| distinct += @intFromBool(a != b);
    try testing.expect(distinct > count / 2);

    var graphs: loopback.Graphs = try .init(gpa);
    defer graphs.deinit();
    const depth = forward_max_rows - 1;
    const Case = struct { wrong: *const fn (usize) bool, count: usize, graphed: bool };
    for ([_]Case{
        .{ .wrong = never, .count = depth, .graphed = true },
        .{ .wrong = never, .count = 1, .graphed = false },
        .{ .wrong = always, .count = depth, .graphed = true },
        .{ .wrong = everyThird, .count = depth, .graphed = true },
        .{ .wrong = everyFifth, .count = depth, .graphed = false },
        .{ .wrong = everyFifth, .count = 2, .graphed = true },
    }) |case| {
        var oracle: Oracle = .{ .reply = serial, .first = prompt.len, .count = case.count, .wrong = case.wrong };
        const drafted = try reply(gpa, loop, prompt, count, settings, none, oracle.drafter(), if (case.graphed) &graphs else null);
        defer gpa.free(drafted.tokens);
        try testing.expectEqualSlices(i32, serial, drafted.tokens);
        try testing.expectEqual(drafted.stats.rounds, oracle.calls);
        if (case.wrong == &never) {
            // every draft is kept, so a round makes one token more than it drafts
            try testing.expectEqual(drafted.stats.drafted, drafted.stats.accepted);
            try testing.expectEqual(count - 1, drafted.stats.rounds + drafted.stats.accepted);
            try testing.expect(drafted.stats.rounds < count - 1);
        } else if (case.wrong == &always) {
            try testing.expectEqual(@as(usize, 0), drafted.stats.accepted);
            try testing.expectEqual(count - 1, drafted.stats.rounds);
        } else {
            try testing.expect(drafted.stats.accepted > 0 and drafted.stats.accepted < drafted.stats.drafted);
            try testing.expectEqual(count - 1, drafted.stats.rounds + drafted.stats.accepted);
        }
    }

    // a reply that ends: the row that chooses the end token is the last kept, whatever was drafted after it
    const end = [1]u32{@intCast(serial[count / 2])};
    const ended = try reply(gpa, loop, prompt, count, settings, &end, null, null);
    defer gpa.free(ended.tokens);
    try testing.expect(ended.tokens.len < count);
    try testing.expectEqualSlices(i32, serial[0..ended.tokens.len], ended.tokens);
    var oracle: Oracle = .{ .reply = serial, .first = prompt.len, .count = depth, .wrong = never };
    const drafted = try reply(gpa, loop, prompt, count, settings, &end, oracle.drafter(), &graphs);
    defer gpa.free(drafted.tokens);
    try testing.expectEqualSlices(i32, ended.tokens, drafted.tokens);
}

test "drafted decoding over both ranks gives the serial reply, greedy and sampled, whatever the drafts" {
    const gpa = testing.allocator;
    const count = 40;
    var fixture: Fixture = undefined;
    // a prompt of more than one window, so its first rows are advanced over
    try fixture.init(61, 23);
    defer fixture.deinit();
    var loop = try fixture.loop(fixture.tokens.len + count + forward_max_rows);
    defer loop.deinit();
    try expectDrafted(gpa, &loop, fixture.tokens, count, .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 });
    try expectDrafted(gpa, &loop, fixture.tokens, count, .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 });
}
