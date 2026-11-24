//! `session.zig`'s tests of the states kept inside a reply: where its reasoning ends, and on its stride.

const std = @import("std");
const testing = std.testing;
const fixtures = @import("../testing.zig");
const Counting = fixtures.sessions.Counting;
const cuda = @import("../../../../src/cuda/cuda.zig");
const masks_file = @import("../../../../src/decode/masks.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const grammar = @import("../../../../src/text/grammar.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const draft = @import("../../../../src/models/qwen4/draft.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const parked_states = @import("../../../../src/models/qwen4/session/parked.zig").states;
const session_file = @import("../../../../src/models/qwen4/session/session.zig");
const Options = session_file.Options;
const Session = session_file.Session;

test "the state where a reply's reasoning ends is kept: a prompt that parts in the content runs from there, as an empty state would" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    // a prompt of 40 tokens takes one snapshot, the mark before its end, and the reply one
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 4, .stride = 32 });
    defer session.destroy();
    // one at most: the reply's goes in for the prompt's
    const tight: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 1, .stride = 32 });
    defer tight.destroy();
    const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 0 });
    defer fresh.destroy();

    const count = 12;
    const open = 251;
    var random: std.Random.DefaultPrng = .init(35);
    var prompt: [40]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    prompt[prompt.len - 1] = open;
    var counting: Counting = .{};
    const each = [_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 9, .temperature = 0.8 } };
    var tried: usize = 0;
    // a constraint that takes any token, after three it forces
    const Any = struct {
        const table: grammar.Constraint.Table = .{ .start = start, .fill = fill, .accept = accept, .ended = ended, .deinit = destroy };
        fn start(_: *anyopaque) grammar.Error!grammar.Position {
            return @fromBackingInt(@as(u64, 0));
        }
        fn fill(_: *anyopaque, _: grammar.Position, mask: []u32) grammar.Error!void {
            @memset(mask, std.math.maxInt(u32));
        }
        fn accept(_: *anyopaque, position: grammar.Position, _: u32) grammar.Error!?grammar.Position {
            return position;
        }
        fn ended(_: *anyopaque, _: grammar.Position) bool {
            return false;
        }
        fn destroy(_: *anyopaque) void {}
    };
    var nothing: u8 = 0;
    const led = try grammar.Led.wrap(gpa, .{ .context = &nothing, .words = (m.shape.vocab + 31) / 32, .table = &Any.table }, &.{ 7, 6, 5 });
    defer led.deinit();
    var forced: usize = 0;
    // The reply's token `index` stands for `</think>`: the pending token after the prompt, then tokens that
    // are run as each row of a window of drafts.
    for (0..count - 1) |index| {
        const settings = each[index % 2];
        const expected = try fixtures.serial.serialReply(gpa, m, &prompt, count, settings);
        defer gpa.free(expected);
        const close = expected[index];
        if (close == open or std.mem.indexOfScalar(i32, expected[0..index], close) != null) continue;
        tried += 1;
        const at = prompt.len + index + 1;
        for ([_]*Session{ session, tight }) |one| {
            errdefer std.debug.print("token {d} closes the reasoning, {d} snapshots\n", .{ index, one.options.snapshots });
            one.think = .{ .open = open, .close = close };
            one.trusted = false;
            var record: [count]i32 = undefined;
            const reply = try one.reply(&prompt, settings, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
            try testing.expectEqualSlices(i32, expected, &record);
            try testing.expectEqual(2, reply.taken);
            const held = try fixtures.sessions.snapshotPositions(one, gpa);
            defer gpa.free(held);
            try testing.expectEqualSlices(usize, if (one == tight) &.{at} else &.{ 38, at }, held);

            // The next prompt has the reasoning and the token after its close as they were decoded, and then
            // other tokens: it goes on from the reasoning's end, and replies as a session that runs it whole.
            var next: std.ArrayList(i32) = .empty;
            defer next.deinit(gpa);
            try next.appendSlice(gpa, &prompt);
            try next.appendSlice(gpa, record[0 .. index + 2]);
            try next.appendSlice(gpa, &.{ 7, 6, 5, open });
            var after: [count]i32 = undefined;
            const second = try one.reply(next.items, each[1 - index % 2], .{ .max_tokens = count, .depth = 3, .record = &after }, counting.sink());
            try testing.expectEqual(at, second.cached);
            fresh.trusted = false;
            var whole: [count]i32 = undefined;
            const cold = try fresh.reply(next.items, each[1 - index % 2], .{ .max_tokens = count, .depth = 3, .record = &whole }, counting.sink());
            try testing.expectEqual(0, cold.cached);
            try testing.expectEqualSlices(i32, &whole, &after);
            try testing.expectEqual(cold.drafts, second.drafts);
            try testing.expectEqual(cold.stats.accepted, second.stats.accepted);

            // A prompt that parts at the token after the close is not served by that state (the head's entry
            // for the close was absorbed with the token it no longer has): it runs from the prompt's own mark,
            // where that is still held.
            one.trusted = false;
            _ = try one.reply(&prompt, settings, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
            next.shrinkRetainingCapacity(prompt.len + index + 1);
            try next.appendSlice(gpa, &.{ 7, 6, 5, open });
            const parted = try one.reply(next.items, settings, .{ .max_tokens = count, .depth = 3, .record = &after }, counting.sink());
            try testing.expectEqual(@as(usize, if (one == tight) 0 else 38), parted.cached);
            fresh.trusted = false;
            _ = try fresh.reply(next.items, settings, .{ .max_tokens = count, .depth = 3, .record = &whole }, counting.sink());
            try testing.expectEqualSlices(i32, &whole, &after);
        }

        // Under a constraint that forces what follows the close, as a document's lead does, the window that
        // holds the close keeps the forced rows after it: the state is the one after the close all the same.
        errdefer std.debug.print("token {d} closes the reasoning, a lead forced after it\n", .{index});
        session.think = .{ .open = open, .close = close };
        session.trusted = false;
        var record: [count + 4]i32 = undefined;
        const reply = try session.reply(&prompt, settings, .{ .max_tokens = record.len, .depth = 3, .record = &record, .constraint = led }, counting.sink());
        try testing.expectEqualSlices(i32, expected[0 .. index + 1], record[0 .. index + 1]);
        try testing.expectEqualSlices(i32, &.{ 7, 6, 5 }, record[index + 1 .. index + 4]);
        forced += reply.masks.forced;
        const held = try fixtures.sessions.snapshotPositions(session, gpa);
        defer gpa.free(held);
        try testing.expectEqualSlices(usize, &.{ 38, at }, held);
        var next: std.ArrayList(i32) = .empty;
        defer next.deinit(gpa);
        try next.appendSlice(gpa, &prompt);
        try next.appendSlice(gpa, record[0 .. index + 3]);
        try next.appendSlice(gpa, &.{ 9, open });
        var after: [count]i32 = undefined;
        try testing.expectEqual(at, (try session.reply(next.items, settings, .{ .max_tokens = count, .depth = 3, .record = &after }, counting.sink())).cached);
        fresh.trusted = false;
        var whole: [count]i32 = undefined;
        _ = try fresh.reply(next.items, settings, .{ .max_tokens = count, .depth = 3, .record = &whole }, counting.sink());
        try testing.expectEqualSlices(i32, &whole, &after);
    }
    try testing.expect(tried >= 6);
    // the lead's tokens were drafted, so windows kept rows past the close
    try testing.expect(forced >= 2 * tried);

    // A reply that does not think keeps none, and one whose reasoning never closes keeps none either.
    session.think = null;
    session.trusted = false;
    var record: [count]i32 = undefined;
    try testing.expectEqual(1, (try session.reply(&prompt, each[0], .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink())).taken);
    session.think = .{ .open = open, .close = 253 };
    session.trusted = false;
    try testing.expectEqual(1, (try session.reply(&prompt, each[0], .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink())).taken);
}

test "states are kept on a reply's stride: a prompt that parts late in a long reply runs from the newest it agrees past, as an empty state would" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    // a state each 16 tokens of a reply, the two newest held beside the four of the prompts
    const stride = 16;
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 4, .stride = 32, .reply_stride = stride });
    defer session.destroy();
    // one of each at most; its windows are of four rows, most of them kept in part (the head's chains go on
    // whatever its confidence, and a random model takes few of its drafts)
    const tight: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 1, .stride = 32, .reply_stride = stride, .reply_marks = 1 });
    defer tight.destroy();
    tight.head.?.confidence = 0;
    tight.head.?.chain = 0;
    const fresh: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 0 });
    defer fresh.destroy();
    const confident: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 0 });
    defer confident.destroy();
    confident.head.?.confidence = 0;
    confident.head.?.chain = 0;

    const count = 60;
    var random: std.Random.DefaultPrng = .init(36);
    var prompt: [40]i32 = undefined;
    for (&prompt) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    const sampled: sampling.Sampling = .{ .seed = 9, .temperature = 0.8 };
    const expected = try fixtures.serial.serialReply(gpa, m, &prompt, count, sampled);
    defer gpa.free(expected);
    // The prompt's one state stands at 38, the reply's 60 tokens at 40 to 99: states at 54, 70 and 86.
    const end = prompt.len + count - 1;

    // Where the next prompt parts from the sequence held, and the state each session goes on from: one at
    // position p serves a prompt that agrees through token p.
    const Case = struct { shared: usize, cached: [2]usize, count: usize = 20 };
    const cases = [_]Case{
        // the whole reply and more: the state as it stands
        .{ .shared = end + 1, .cached = .{ end, end } },
        // late in the reply, as after a stop string or a hang-up: the newest on the stride; and the reply
        // after it takes its own
        .{ .shared = 95, .cached = .{ 86, 86 }, .count = count },
        // right after the newest, which is then where the prompt parts less one: the state `plan` would have
        // kept there joins the snapshots, and outlives the strides of the long reply that follows
        .{ .shared = 87, .cached = .{ 86, 86 }, .count = count },
        // at the newest: the one before it, where two are held, else the prompt's
        .{ .shared = 86, .cached = .{ 70, 38 } },
        .{ .shared = 71, .cached = .{ 70, 38 } },
        // below both: the prompt's own mark, as before
        .{ .shared = 70, .cached = .{ 38, 38 } },
        .{ .shared = 45, .cached = .{ 38, 38 } },
    };
    for (cases) |case| {
        for ([_]*Session{ session, tight }, case.cached, 0..) |one, cached, nth| {
            errdefer std.debug.print("parted at {d}, session {d}\n", .{ case.shared, nth });
            one.trusted = false;
            var record: [count]i32 = undefined;
            const reply = try one.reply(&prompt, sampled, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
            try testing.expectEqualSlices(i32, expected, &record);
            try testing.expectEqual(0, reply.cached);
            // the prompt's mark and the three of the reply; the prompt's is not dropped for them
            try testing.expectEqual(4, reply.taken);
            const held = try fixtures.sessions.snapshotPositions(one, gpa);
            defer gpa.free(held);
            try testing.expectEqualSlices(usize, &.{38}, held);
            const strided = try fixtures.sessions.stridePositions(one, gpa);
            defer gpa.free(strided);
            try testing.expectEqualSlices(usize, if (nth == 0) &.{ 70, 86 } else &.{86}, strided);
            try testing.expectEqual(end, one.state.pos);

            var next: std.ArrayList(i32) = .empty;
            defer next.deinit(gpa);
            try next.appendSlice(gpa, one.known.items[0..case.shared]);
            // another token than the one held there
            try next.append(gpa, if (case.shared < one.known.items.len and one.known.items[case.shared] == 7) 8 else 7);
            try next.appendSlice(gpa, &.{ 6, 5, 9 });
            var after: [count]i32 = undefined;
            const second = try one.reply(next.items, greedy, .{ .max_tokens = case.count, .depth = 3, .record = &after }, counting.sink());
            try testing.expectEqual(cached, second.cached);
            const cold = if (nth == 0) fresh else confident;
            cold.trusted = false;
            var whole: [count]i32 = undefined;
            const other = try cold.reply(next.items, greedy, .{ .max_tokens = case.count, .depth = 3, .record = &whole }, counting.sink());
            try testing.expectEqual(0, other.cached);
            try testing.expectEqualSlices(i32, whole[0..case.count], after[0..case.count]);
            try testing.expectEqual(other.drafts, second.drafts);
            try testing.expectEqual(other.stats.accepted, second.stats.accepted);

            // The states on the stride below where the prompt parted stay, until the reply takes others:
            // each a stride past the newest held, which after the prompt is the mark two before its end.
            const now = try fixtures.sessions.stridePositions(one, gpa);
            defer gpa.free(now);
            var buffer: [3]usize = undefined;
            var want: std.ArrayList(usize) = .initBuffer(&buffer);
            for (strided) |pos| {
                if (pos < case.shared and pos != case.shared - 1) want.appendAssumeCapacity(pos);
            }
            var at = next.items.len - 2 + stride;
            while (at <= next.items.len + case.count - 1) : (at += stride) {
                want.appendAssumeCapacity(at);
                if (want.items.len > one.options.reply_marks) _ = want.orderedRemove(0);
            }
            try testing.expectEqualSlices(usize, want.items, now);
            // and the buffers of those the prompt parted below are the next ones', or wait for them: as many
            // are held as before, but where one joined the snapshots
            if (std.mem.indexOfScalar(usize, strided, case.shared - 1) == null) {
                try testing.expectEqual(strided.len, now.len + one.spare.items.len);
            } else {
                try testing.expect(now.len + one.spare.items.len <= strided.len);
            }
            if (case.shared == 95) try testing.expectEqualSlices(usize, if (nth == 0) &.{ 129, 145 } else &.{145}, now);
            if (case.shared != 87) continue;
            // The state at 86 was the part point's: it is a snapshot now where the budget has room (with one,
            // the turn's mark took its place, as it would have before), and the reply's strides are all new.
            try testing.expectEqualSlices(usize, if (nth == 0) &.{ 121, 137 } else &.{137}, now);
            const kept = try fixtures.sessions.snapshotPositions(one, gpa);
            defer gpa.free(kept);
            try testing.expectEqualSlices(usize, if (nth == 0) &.{ 38, 86, 89 } else &.{89}, kept);
            // so a prompt that parts there again goes on from it, where the strides alone would give 38 (and with
            // a budget of one nothing is left below it, as before)
            next.items[case.shared] += 2;
            const third = try one.reply(next.items, greedy, .{ .max_tokens = 20, .depth = 3, .record = &after }, counting.sink());
            try testing.expectEqual(@as(usize, if (nth == 0) 86 else 0), third.cached);
            cold.trusted = false;
            _ = try cold.reply(next.items, greedy, .{ .max_tokens = 20, .depth = 3, .record = &whole }, counting.sink());
            try testing.expectEqualSlices(i32, whole[0..20], after[0..20]);
        }
    }

    // A reply that thinks: the stride is counted from the reasoning's end once that is taken, and the end
    // stays among the prompt's states.
    const open = 251;
    prompt[prompt.len - 1] = open;
    const thought = try fixtures.serial.serialReply(gpa, m, &prompt, count, greedy);
    defer gpa.free(thought);
    // the token that closes the reasoning: one the reply has not written before, 58 to 66 tokens in
    const index = for (17..26) |i| {
        if (thought[i] != open and std.mem.indexOfScalar(i32, thought[0..i], thought[i]) == null) break i;
    } else return error.TestUnexpectedResult;
    const close = thought[index];
    const closed = prompt.len + index + 1;
    session.think = .{ .open = open, .close = close };
    session.trusted = false;
    var record: [count]i32 = undefined;
    const reply = try session.reply(&prompt, greedy, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
    try testing.expectEqualSlices(i32, thought, &record);
    // the prompt's, one in the reasoning at 54, the reasoning's end, two after it
    try testing.expectEqual(5, reply.taken);
    const held = try fixtures.sessions.snapshotPositions(session, gpa);
    defer gpa.free(held);
    try testing.expectEqualSlices(usize, &.{ 38, closed }, held);
    const strided = try fixtures.sessions.stridePositions(session, gpa);
    defer gpa.free(strided);
    try testing.expectEqualSlices(usize, &.{ closed + stride, closed + 2 * stride }, strided);
    // a prompt that parts in the content before the first of them goes on from the reasoning's end
    var next: std.ArrayList(i32) = .empty;
    defer next.deinit(gpa);
    try next.appendSlice(gpa, session.known.items[0 .. closed + stride]);
    try next.appendSlice(gpa, &.{ if (session.known.items[closed + stride] == 7) 8 else 7, 6, open });
    var after: [count]i32 = undefined;
    const second = try session.reply(next.items, sampled, .{ .max_tokens = 20, .depth = 3, .record = &after }, counting.sink());
    try testing.expectEqual(closed, second.cached);
    fresh.think = session.think;
    fresh.trusted = false;
    var whole: [count]i32 = undefined;
    const other = try fresh.reply(next.items, sampled, .{ .max_tokens = 20, .depth = 3, .record = &whole }, counting.sink());
    try testing.expectEqualSlices(i32, whole[0..20], after[0..20]);
    try testing.expectEqual(other.drafts, second.drafts);

    // No stride, or no state of the prompt's to count it from: none is taken.
    for ([_]Options{
        .{ .window = 4, .capacity = 256, .snapshots = 4, .stride = 32, .reply_stride = 0 },
        .{ .window = 4, .capacity = 256, .snapshots = 4, .stride = 32, .reply_stride = stride, .reply_marks = 0 },
        .{ .window = 4, .capacity = 256, .snapshots = 0, .reply_stride = stride },
    }) |options| {
        const none: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, options);
        defer none.destroy();
        const plain = try none.reply(&prompt, greedy, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
        try testing.expectEqual(@as(usize, @intFromBool(options.snapshots > 0)), plain.taken);
        try testing.expectEqual(0, none.strides.items.len);
    }

    // A stride no window is a multiple of (and wider than one: a window gives one state at most), with drafts that are kept (the random head's are not): each state
    // stands on its own row inside the windows that keep several, and not where a window happens to end.
    for ([_]usize{ 0, 7 }) |spoiled| {
        const narrow: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 4, .stride = 32, .reply_stride = 5 });
        defer narrow.destroy();
        // the head drafts as deep as it is asked, whatever it thinks of its drafts
        narrow.head.?.confidence = 0;
        narrow.head.?.chain = 0;
        var told: Told = .{ .head = narrow.head.?.drafter(), .reply = thought, .first = prompt.len, .spoiled = spoiled };
        narrow.drafting = told.drafter();
        const plain = try narrow.reply(&prompt, greedy, .{ .max_tokens = count, .depth = 3, .record = &record }, counting.sink());
        try testing.expectEqualSlices(i32, thought, &record);
        // windows of several rows, whole and in part
        if (spoiled == 0) {
            try testing.expectEqual(plain.stats.drafted, plain.stats.accepted);
        } else {
            try testing.expect(plain.stats.accepted > count / 2 and plain.stats.accepted < plain.stats.drafted);
        }
        try testing.expectEqual(1 + (end - 38) / 5, plain.taken);
        const on = try fixtures.sessions.stridePositions(narrow, gpa);
        defer gpa.free(on);
        try testing.expectEqualSlices(usize, &.{ 93, 98 }, on);
        next.clearRetainingCapacity();
        try next.appendSlice(gpa, narrow.known.items[0..97]);
        try next.appendSlice(gpa, &.{ if (narrow.known.items[97] == 7) 8 else 7, 6, 5 });
        narrow.drafting = null;
        const again = try narrow.reply(next.items, sampled, .{ .max_tokens = 20, .depth = 3, .record = &after }, counting.sink());
        try testing.expectEqual(93, again.cached);
        confident.trusted = false;
        const from = try confident.reply(next.items, sampled, .{ .max_tokens = 20, .depth = 3, .record = &whole }, counting.sink());
        try testing.expectEqualSlices(i32, whole[0..20], after[0..20]);
        try testing.expectEqual(from.drafts, again.drafts);
    }
}

/// A drafter that knows the reply: the head is asked first, so that it absorbs the window as it would, and its
/// drafts are replaced by the reply's own tokens, each `spoiled`th position's by another where that is not 0.
const Told = struct {
    head: draft.Drafter,
    /// `reply[i]` sits at position `first + i`.
    reply: []const i32,
    first: usize,
    spoiled: usize,

    fn drafter(self: *Told) draft.Drafter {
        return .{ .context = self, .draft = tell };
    }

    fn tell(context: *anyopaque, streams: cuda.View, kept: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Told = @ptrCast(@alignCast(context));
        const n = try self.head.draft(self.head.context, streams, kept, position, guide, out);
        if (position < self.first) return n;
        for (out[0..n], position + 1..) |*token, at| {
            if (at - self.first >= self.reply.len) return at - position - 1;
            const right = self.reply[at - self.first];
            token.* = if (self.spoiled != 0 and at % self.spoiled == 0) @mod(right + 7, 250) else right;
        }
        return n;
    }
};

/// A reply's tokens as they are taken, with the buffers the device had been asked for by the first and by
/// the last: every round of the reply lies between the two.
const Allocating = struct {
    first: ?usize = null,
    last: usize = 0,

    fn take(context: *anyopaque, _: i32) void {
        const self: *Allocating = @ptrCast(@alignCast(context));
        self.last = cuda.account.made_ever;
        if (self.first == null) self.first = self.last;
    }

    fn halt(_: *anyopaque) bool {
        return false;
    }

    fn tap(self: *Allocating) session_file.Tap {
        return .{ .context = self, .takeFn = take, .haltFn = halt };
    }
};

test "a reply's rounds allocate nothing on the device: its strides' buffers are made behind the prompt's last window" {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena.allocator(), null, 32);
    defer m.deinit();
    var diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
    defer tokens.deinit(gpa);
    const stride = 16;
    const session: *Session = try .create(gpa, testing.io, m, &tokens, &.{}, .{ .window = 4, .capacity = 256, .snapshots = 4, .stride = 32, .reply_stride = stride });
    defer session.destroy();

    const count = 60;
    var random: std.Random.DefaultPrng = .init(89);
    var prompts: [2][40]i32 = undefined;
    for (&prompts) |*prompt| for (prompt) |*token| {
        token.* = random.random().intRangeLessThan(i32, 1, 250);
    };
    var counting: Counting = .{};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    var record: [count]i32 = undefined;
    // The session's first reply, another conversation's, and the first's again: each takes three states on
    // its stride, the first two into buffers the session did not have when its prompt came.
    for ([_]usize{ 0, 1, 0 }, 0..) |which, nth| {
        errdefer std.debug.print("reply {d}\n", .{nth});
        const expected = try fixtures.serial.serialReply(gpa, m, &prompts[which], count, greedy);
        defer gpa.free(expected);
        var seen: Allocating = .{};
        const reply = try session.reply(&prompts[which], greedy, .{ .max_tokens = count, .depth = 3, .record = &record, .tap = seen.tap() }, counting.sink());
        try testing.expectEqualSlices(i32, expected, &record);
        try testing.expect(reply.taken >= 3);
        try testing.expectEqual(session.options.reply_marks, session.strides.items.len);
        try testing.expectEqual(0, session.spare.items.len);
        try testing.expectEqual(seen.first.?, seen.last);
    }

    // The next turn, with states that went waiting in the lot while two stand on the stride: the reply's
    // states are written over the stride's own, and no more than `reply_marks` are held.
    for (0..parked_states + 2) |_| {
        var blank: forward.Snapshot = try .ahead(gpa, session.state, 1);
        errdefer blank.deinit();
        try blank.reserve(session.state);
        session.lot.give(blank);
    }
    var next: std.ArrayList(i32) = .empty;
    defer next.deinit(gpa);
    try next.appendSlice(gpa, session.known.items);
    try next.appendSlice(gpa, &.{ 6, 5, 9 });
    const expected = try fixtures.serial.serialReply(gpa, m, next.items, count, greedy);
    defer gpa.free(expected);
    var seen: Allocating = .{};
    _ = try session.reply(next.items, greedy, .{ .max_tokens = count, .depth = 3, .record = &record, .tap = seen.tap() }, counting.sink());
    try testing.expectEqualSlices(i32, expected, &record);
    try testing.expectEqual(session.options.reply_marks, session.strides.items.len);
    try testing.expectEqual(seen.first.?, seen.last);
}
