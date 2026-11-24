//! `session.zig`'s test, on FP4-Local, the checkpoint `ZSPARK_GLM_FP4_LOCAL` names: its first four layers with both ranks in the
//! process. A prompt that goes on from the caches gives the reply of a fresh one, drafted and serial; an
//! edited last message and a hang-up inside a reply do too, from the positions held; a tap takes every token
//! and ends a reply; a record's chains leave the reply and the session's rule as they were; a follower's marks
//! hold both ranks' shots at one position, on the same session (spec 133). With the DFlash2
//! drafter `ZSPARK_GLM_DRAFTER` names, a prompt resumed at a mark inside a window or on a reply's grid drafts
//! the serial reply, by the drafts of a fresh one. 13 GiB on the device.

const std = @import("std");
const testing = std.testing;
const shared = @import("../testing.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const flash_config = @import("../../../../src/decode/dflash2/config.zig");
const flash_layout = @import("../../../../src/decode/dflash2/layout.zig");
const rig = @import("../../../../src/models/glm5/dflash2/rig.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const session_file = @import("../../../../src/models/glm5/session/session.zig");

const Session = session_file.SessionOf(layout.world);

/// A tap that keeps what it is given and halts once it holds `limit`.
const Taken = struct {
    tokens: [64]i32 = undefined,
    count: usize = 0,
    limit: usize = 64,
    cached: ?usize = null,

    fn tap(self: *Taken) session_file.Tap {
        return .{ .context = self, .take = take, .halt = halt, .held = held };
    }

    fn take(context: *anyopaque, token: i32) void {
        const self: *Taken = @ptrCast(@alignCast(context));
        self.tokens[self.count] = token;
        self.count += 1;
    }

    fn halt(context: *anyopaque) bool {
        const self: *Taken = @ptrCast(@alignCast(context));
        return self.count >= self.limit;
    }

    fn held(context: *anyopaque, cached: usize) void {
        const self: *Taken = @ptrCast(@alignCast(context));
        self.cached = cached;
    }
};

/// What `ReplyOptions.chains` was given: a chain's position and length, and whether every pick is a token
/// with a probability.
const Recorded = struct {
    positions: [64]usize = undefined,
    lengths: [64]usize = undefined,
    count: usize = 0,
    sound: bool = true,

    fn chains(self: *Recorded) session_file.Chains {
        return .{ .context = self, .chain = chain };
    }

    fn chain(context: *anyopaque, position: usize, trail: []const session_file.Chains.Trail, noisy: []const f64) anyerror!void {
        const self: *Recorded = @ptrCast(@alignCast(context));
        if (noisy.len != 0) self.sound = false;
        self.positions[self.count] = position;
        self.lengths[self.count] = trail.len;
        self.count += 1;
        for (trail) |one| {
            if (one.token < 0 or !(one.p > 0 and one.p <= 1)) self.sound = false;
        }
    }
};

test "FP4-Local's first layers in a session: a prompt that goes on from the caches gives the fresh reply, a tap ends one, and a follower's snapshot marks hold both ranks' shots at one position" {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    const session = try Session.create(gpa, arena, io, &checkpoint, &c, .{}, .{ .layers = 4, .capacity = 96, .snapshots = 4 }, "test");
    defer session.destroy();
    try session.meet();

    const first = [_]i32{ 154822, 154824, 154827, 852, 279, 1156, 4236, 10245, 5109, 11, 18630, 553, 75994, 13, 154828, 154841, 154842 };
    const more = [_]i32{ 852, 279, 1156 };
    for ([_]sampling.Sampling{ .{ .seed = 0, .temperature = 0 }, .{ .seed = 7, .temperature = 0.8 } }) |settings| {
        for ([_]bool{ true, false }) |drafts| {
            var a: [12]i32 = undefined;
            var taken: Taken = .{};
            const one = try session.reply(&first, settings, .{ .eos = &.{}, .drafts = drafts, .fresh = true, .tap = taken.tap() }, &a);
            try testing.expectEqual(@as(usize, a.len), one.made);
            try testing.expectEqual(session_file.Finish.length, one.finish);
            try testing.expectEqual(@as(?usize, 0), taken.cached);
            try testing.expectEqualSlices(i32, &a, taken.tokens[0..taken.count]);
            // the same prompt again starts two tokens before its end and gives the same reply
            var resumed_out: [12]i32 = undefined;
            const resumed = try session.reply(&first, settings, .{ .eos = &.{}, .drafts = drafts }, &resumed_out);
            try testing.expectEqualSlices(i32, &a, &resumed_out);
            try testing.expect(resumed.cached > 0 and resumed.cached < first.len);
            // the turn after: the prompt, its reply, and more
            var next: std.ArrayList(i32) = .empty;
            try next.appendSlice(arena, &first);
            try next.appendSlice(arena, &a);
            try next.appendSlice(arena, &more);
            var onward: [10]i32 = undefined;
            const went = try session.reply(next.items, settings, .{ .eos = &.{}, .drafts = drafts }, &onward);
            try testing.expectEqual(first.len + a.len - 1, went.cached);
            var fresh: [10]i32 = undefined;
            const again = try session.reply(next.items, settings, .{ .eos = &.{}, .drafts = drafts, .fresh = true }, &fresh);
            try testing.expectEqual(@as(usize, 0), again.cached);
            try testing.expectEqualSlices(i32, &fresh, &onward);
            // a prompt the caches hold whole, or one that parts from them, starts again
            try testing.expectEqual(@as(usize, 0), session.cachedOf(next.items, drafts));
            try testing.expectEqual(@as(usize, 0), session.cachedOf(&first, drafts));

            // a record's chains: the reply is the same, a chain a round from each position in turn, as long as
            // the drafter goes, and the rule the session drafts by is back afterwards
            if (drafts) {
                const rule = session.head.confidence;
                var recorded: Recorded = .{};
                var c2: [12]i32 = undefined;
                const plain = try session.reply(&first, settings, .{ .eos = &.{}, .fresh = true, .chains = recorded.chains() }, &c2);
                try testing.expectEqualSlices(i32, &a, &c2);
                try testing.expectEqual(@as(usize, 0), plain.stats.accepted);
                try testing.expectEqual(a.len - 1, recorded.count);
                try testing.expect(recorded.sound);
                for (recorded.positions[0..recorded.count], recorded.lengths[0..recorded.count], 0..) |position, length, i| {
                    try testing.expectEqual(first.len + i, position);
                    // the chain ends at the reply's room and no sooner: its floor is no probability's
                    try testing.expect(recorded.lengths[0] > 1);
                    try testing.expectEqual(@min(recorded.lengths[0], a.len - 2 - i), length);
                }
                try testing.expectEqual(rule, session.head.confidence);
                var after: [12]i32 = undefined;
                const drafted = try session.reply(&first, settings, .{ .eos = &.{}, .fresh = true }, &after);
                try testing.expectEqualSlices(i32, &a, &after);
                try testing.expectEqual(one.stats.drafted, drafted.stats.drafted);
            }

            // a reply the tap ends is the fresh reply's start, and an end token ends one with itself
            var cut: Taken = .{ .limit = 3 };
            var b: [12]i32 = undefined;
            const short = try session.reply(&first, settings, .{ .eos = &.{}, .drafts = drafts, .tap = cut.tap() }, &b);
            try testing.expectEqual(session_file.Finish.cut, short.finish);
            try testing.expect(short.made >= 3 and short.made < a.len);
            try testing.expectEqualSlices(i32, a[0..short.made], b[0..short.made]);
            const ends = [_]u32{@intCast(a[4])};
            const stopped = try session.reply(&first, settings, .{ .eos = &ends, .drafts = drafts }, &b);
            try testing.expectEqual(session_file.Finish.stop, stopped.finish);
            const at = std.mem.indexOfScalar(i32, &a, a[4]).?;
            try testing.expectEqual(at + 1, stopped.made);
            try testing.expectEqualSlices(i32, a[0 .. at + 1], b[0 .. at + 1]);

            // an edited last token parts at the mark two before the prompt's end and gives the fresh reply
            var edited = first;
            edited[16] +%= 17;
            var edited_out: [12]i32 = undefined;
            const edited_reply = try session.reply(&edited, settings, .{ .eos = &.{}, .drafts = drafts }, &edited_out);
            try testing.expect(edited_reply.cached > 0 and edited_reply.cached < edited.len);
            var edited_fresh: [12]i32 = undefined;
            const edited_again = try session.reply(&edited, settings, .{ .eos = &.{}, .drafts = drafts, .fresh = true }, &edited_fresh);
            try testing.expectEqual(@as(usize, 0), edited_again.cached);
            try testing.expectEqualSlices(i32, &edited_fresh, &edited_out);

            // a hang-up inside the reply, then the tokens it wrote: the same as a cold session, from a
            // position it held
            var hung: Taken = .{ .limit = 3 };
            var partial: [12]i32 = undefined;
            const hung_reply = try session.reply(&first, settings, .{ .eos = &.{}, .drafts = drafts, .fresh = true, .tap = hung.tap() }, &partial);
            try testing.expectEqual(session_file.Finish.cut, hung_reply.finish);
            try testing.expect(hung_reply.made >= 3 and hung_reply.made < a.len);
            var continued: std.ArrayList(i32) = .empty;
            try continued.appendSlice(arena, &first);
            try continued.appendSlice(arena, partial[0..hung_reply.made]);
            var after_hang: [8]i32 = undefined;
            const went_on = try session.reply(continued.items, settings, .{ .eos = &.{}, .drafts = drafts }, &after_hang);
            try testing.expect(went_on.cached > 0 and went_on.cached < continued.items.len);
            var cold_hang: [8]i32 = undefined;
            const cold = try session.reply(continued.items, settings, .{ .eos = &.{}, .drafts = drafts, .fresh = true }, &cold_hang);
            try testing.expectEqual(@as(usize, 0), cold.cached);
            try testing.expectEqualSlices(i32, &cold_hang, &after_hang);
        }
    }

    // a follower's snapshot marks, on the same session (spec 133: one load): a fresh reply starts with none
    {
        // a fresh drafted reply: the mark two tokens before the prompt's end, inside its last window, and the reply's
        var a: [12]i32 = undefined;
        _ = try session.reply(&first, sampling.Sampling{ .seed = 0, .temperature = 0 }, .{ .eos = &.{}, .drafts = true, .fresh = true }, &a);
        try testing.expect(session.marksHeld() >= 2);
        var before_prompt: ?usize = null;
        for (0..session.marksHeld()) |at| {
            const one = session.markAt(at);
            // each rank's shot stands where the mark does, and took the KDA chains the model's first layers keep
            for (one.ranks) |position| try testing.expectEqual(one.pos, position);
            for (one.kda) |taken| try testing.expect(taken);
            if (one.pos < first.len) before_prompt = one.pos;
        }
        // the mark before the prompt's end is there: the same prompt resumes from it
        try testing.expect(before_prompt != null and before_prompt.? > 0);
        var again: [12]i32 = undefined;
        const resumed = try session.reply(&first, sampling.Sampling{ .seed = 0, .temperature = 0 }, .{ .eos = &.{} }, &again);
        try testing.expectEqualSlices(i32, &a, &again);
        try testing.expectEqual(before_prompt.?, resumed.cached);
        // the marks stand oldest first
        var earlier: usize = 0;
        for (0..session.marksHeld()) |at| {
            try testing.expect(session.markAt(at).pos >= earlier);
            earlier = session.markAt(at).pos;
        }
    }
}

test "a mark inside a chunked prompt window, asked of it before it runs, resumes to the fresh reply" {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    const session = try Session.create(gpa, arena, io, &checkpoint, &c, .{}, .{ .layers = 4, .rows = 128, .capacity = 192, .snapshots = 4 }, "test");
    defer session.destroy();
    try session.meet();

    // 150 tokens, run as windows of 128 (KDA chunked), 6 and 16: the message that starts at 100 is marked at
    // its chunk's start, 64, which the first window writes as it passes it (spec 116)
    var first: [150]i32 = undefined;
    for (&first, 0..) |*one, i| one.* = @intCast(1000 + (i * 7919) % 50000);
    const starts = [_]usize{100};
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    var a: [8]i32 = undefined;
    _ = try session.reply(&first, greedy, .{ .eos = &.{}, .drafts = false, .fresh = true, .starts = &starts }, &a);
    var held = false;
    for (0..session.marksHeld()) |at| held = held or session.markAt(at).pos == 64;
    try testing.expect(held);
    // a prompt that parts from it inside that message goes on from 64, and replies as from an empty sequence
    var second: [130]i32 = undefined;
    @memcpy(second[0..110], first[0..110]);
    for (second[110..], 0..) |*one, i| one.* = @intCast(2000 + i * 13);
    var fresh: [8]i32 = undefined;
    var resumed: [8]i32 = undefined;
    const went = try session.reply(&second, greedy, .{ .eos = &.{}, .drafts = false, .starts = &starts }, &resumed);
    try testing.expectEqual(@as(usize, 64), went.cached);
    _ = try session.reply(&second, greedy, .{ .eos = &.{}, .drafts = false, .fresh = true }, &fresh);
    try testing.expectEqualSlices(i32, &fresh, &resumed);
}

test "DFlash2 resumed at a mark inside a prompt's window, and at one on a reply's grid, drafts the serial replies" {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const where = testing.environ.getPosix("ZSPARK_GLM_DRAFTER") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c, var checkpoint = try shared.openPack(gpa, arena, path);
    defer checkpoint.close();
    // the drafter's directory as a command opens it: read from where it is, nothing of it written
    var drafting: rig.Opened = .{};
    defer drafting.close(io);
    drafting.path = where;
    drafting.dir = try std.Io.Dir.cwd().openDir(io, where, .{});
    drafting.config = try flash_config.Config.read(arena, io, drafting.dir.?, &drafting.diagnostic);
    drafting.checkpoint = try .open(gpa, io, .cwd(), where);
    drafting.layout = try flash_layout.Layout.init(arena, &drafting.config, &drafting.checkpoint.?, &drafting.diagnostic);
    try testing.expect(try rig.fit(arena, &drafting, &c, 4));
    const session = try Session.create(gpa, arena, io, &checkpoint, &c, .{}, .{ .layers = 4, .capacity = 96, .snapshots = 4, .stride = 4, .drafter = &drafting }, "test");
    defer session.destroy();
    try session.meet();

    // 20 tokens, run as a window of 4 and the last of 16: the second message starts at 9, inside the last (the
    // client set's second request resumed there, spec 114)
    const first = [_]i32{ 154822, 154824, 154827, 852, 279, 1156, 4236, 10245, 5109, 154827, 18630, 553, 75994, 13, 1156, 4236, 10245, 154828, 154841, 154842 };
    const starts = [_]usize{9};
    var a: [12]i32 = undefined;
    // parts from `first` inside its second message
    var second: [18]i32 = undefined;
    @memcpy(second[0..12], first[0..12]);
    @memcpy(second[12..], &[_]i32{ 279, 852, 13, 154828, 154841, 154842 });
    const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0 };
    for ([_]sampling.Sampling{ greedy, .{ .seed = 7, .temperature = 0.8 } }) |settings| {
        _ = try session.reply(&first, settings, .{ .eos = &.{}, .drafts = false, .fresh = true }, &a);
        // goes on from `first`'s reply past a mark on its grid
        var third: [first.len + 11]i32 = undefined;
        @memcpy(third[0..first.len], &first);
        @memcpy(third[first.len..][0..9], a[0..9]);
        @memcpy(third[first.len + 9 ..], &[_]i32{ 852, 279 });
        var serial_second: [10]i32 = undefined;
        _ = try session.reply(&second, settings, .{ .eos = &.{}, .drafts = false, .fresh = true }, &serial_second);
        var serial_third: [8]i32 = undefined;
        _ = try session.reply(&third, settings, .{ .eos = &.{}, .drafts = false, .fresh = true }, &serial_third);
        // drafted from an empty sequence: a resumed reply's context is this one's, so are its drafts
        var scratch: [10]i32 = undefined;
        const fresh_second = try session.reply(&second, settings, .{ .eos = &.{}, .fresh = true }, &scratch);
        const fresh_third = try session.reply(&third, settings, .{ .eos = &.{}, .fresh = true }, scratch[0..8]);

        for (0..2) |case| {
            var drafted_first: [12]i32 = undefined;
            const one = try session.reply(&first, settings, .{ .eos = &.{}, .fresh = true, .starts = &starts }, &drafted_first);
            try testing.expectEqualSlices(i32, &a, &drafted_first);
            try testing.expect(one.stats.drafted > 0);
            if (case == 0) {
                var out: [10]i32 = undefined;
                const went = try session.reply(&second, settings, .{ .eos = &.{}, .starts = &starts }, &out);
                try testing.expectEqual(@as(usize, 9), went.cached);
                try testing.expectEqualSlices(i32, &serial_second, &out);
                try testing.expect(went.stats.drafted > 0);
                try testing.expectEqual(fresh_second.stats, went.stats);
            } else {
                var out: [8]i32 = undefined;
                const went = try session.reply(&third, settings, .{ .eos = &.{}, .starts = &starts }, &out);
                // a mark the reply kept on its grid, after a round whose rows the drafter had not absorbed
                try testing.expect(went.cached > first.len and went.cached <= first.len + 9);
                try testing.expectEqualSlices(i32, &serial_third, &out);
                try testing.expect(went.stats.drafted > 0);
                try testing.expectEqual(fresh_third.stats, went.stats);
            }
        }
    }
}
