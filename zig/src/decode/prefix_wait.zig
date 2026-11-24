//! The prefix wait (spec 152 WS-06, TensorFold 1.2.1's `lane_host_prefix.zig`): a request whose prompt
//! shares a cut another prompt is filling waits for that fill instead of running the shared rows again,
//! then goes on from the state the fill kept. The wait lives in the decoder's `admit`, which returns
//! `error.Wait` (the caller queues the request again): it is bound by `since`, never by a sleep, and ends
//! when the fill stands at the cut, is gone, or `ns` has passed — then the request fills for itself.
//! A request waits once (`Waiting.waited`): its next admit runs whatever the fill left.

const std = @import("std");

/// The wait's bound (ZSPARK_PREFIX_WAIT_MS, whole milliseconds; 0: never wait) and what it did.
pub const Wait = struct {
    ns: i96 = @as(i96, default_ms) * std.time.ns_per_ms,
    waits: u64 = 0, // requests that waited for another fill's cut
    timeouts: u64 = 0, // of them, released at the bound with the cut not reached

    pub const default_ms: u32 = 60_000;

    pub fn fromEnv() Wait {
        return fromText(std.mem.span(std.c.getenv("ZSPARK_PREFIX_WAIT_MS") orelse return .{}));
    }

    /// A bound from its text: whole milliseconds, 0 off; anything else is the default.
    pub fn fromText(text: []const u8) Wait {
        const t = std.mem.trim(u8, text, " \t");
        if (t.len == 0 or t[0] < '0' or t[0] > '9') return .{};
        const ms = std.fmt.parseInt(u32, t, 10) catch return .{};
        return .{ .ns = @as(i96, ms) * std.time.ns_per_ms };
    }
};

/// A waiting request's record: the fill it waits on (`leader`, `*const anyopaque` so the pools' own seat
/// types stay theirs), a token that names it still the same fill, the position it must reach, and when the
/// wait began (`std.Io.Clock.awake` nanoseconds, or `testNow` where the caller has no clock).
pub const Waiting = struct {
    leader: ?*const anyopaque = null,
    id: usize = 0,
    at: usize = 0,
    since: i96 = 0,
    /// Set once a request waited: it never waits again, so two prompts cannot wait on each other.
    waited: bool = false,
};

/// Set where the caller's clock is unknown: each wait times out one admit after it began.
pub const testNow: i96 = 0;

/// A fill a waiter may wait on.
pub const Filler = struct {
    context: *const anyopaque,
    /// Its tokens as long as it fills.
    prompt: []const i32,
    /// How far its fill stands.
    reached: usize,
    /// The positions its prompt keeps a state at, ascending.
    marks: []const usize,
    /// A token the caller gives, which tells a later fill at the same address apart.
    id: usize = 0,
};

/// Whether `prompt` waits for a fill: the cut wanted is the fill's highest mark the state serves `prompt`
/// from, which a kept state does where the token after it is the fill's own (`hall_keep.standing`'s rule);
/// it must stand past `have`, what a kept state already holds of `prompt`, or the fill serves nothing a kept
/// state does not. Returns the wait's record, or null.
pub fn pick(wait: *const Wait, fillers: []const Filler, prompt: []const i32, have: usize, now: i96) ?Waiting {
    if (wait.ns <= 0) return null;
    var best: ?Filler = null;
    var at: usize = 0;
    for (fillers) |f| {
        for (f.marks) |mark| {
            // a mark at or under where the fill stands is kept already and needs no wait
            if (mark <= f.reached or mark <= have or mark <= at) continue;
            if (mark + 1 > prompt.len or mark + 1 > f.prompt.len) continue;
            if (!std.mem.eql(i32, f.prompt[0 .. mark + 1], prompt[0 .. mark + 1])) continue;
            best = f;
            at = mark;
        }
    }
    const leader = best orelse return null;
    return .{ .leader = leader.context, .id = leader.id, .at = at, .since = now, .waited = true };
}

/// Whether `waiting` still waits: the fill it names is among `fillers` (same pointer and `id`), short of
/// `waiting.at`, and less than `wait.ns` has passed since `waiting.since`. A released wait counts a timeout.
pub fn holds(wait: *Wait, waiting: *const Waiting, fillers: []const Filler, now: i96) bool {
    const leader = waiting.leader orelse return false;
    for (fillers) |f| {
        if (f.context != leader or f.id != waiting.id) continue;
        if (f.reached >= waiting.at) return false;
        if (now - waiting.since < wait.ns) return true;
        wait.timeouts += 1;
        return false;
    }
    return false;
}

const testing = std.testing;

fn filler(ctx: *const anyopaque, prompt: []const i32, reached: usize, marks: []const usize) Filler {
    return .{ .context = ctx, .prompt = prompt, .reached = reached, .marks = marks, .id = 1 };
}

test "Wait.fromText: whole milliseconds, 0 off, anything else the default" {
    try testing.expectEqual(@as(i96, 60_000) * std.time.ns_per_ms, Wait.fromText("").ns);
    try testing.expectEqual(@as(i96, 0), Wait.fromText("0").ns);
    try testing.expectEqual(@as(i96, 250) * std.time.ns_per_ms, Wait.fromText(" 250\t").ns);
    try testing.expectEqual(@as(i96, 60_000) * std.time.ns_per_ms, Wait.fromText("soon").ns);
}

test "a prompt waits for the fill whose cut it shares" {
    var wait: Wait = .{};
    const a: usize = 0;
    const b: usize = 0;
    const fills = [_]Filler{
        filler(&a, &.{ 1, 2, 3, 4, 5, 6 }, 2, &.{5}),
        filler(&b, &.{ 9, 9, 9, 9 }, 0, &.{2}),
    };
    // the cut at 5 is no cut of this prompt's: the token after it is its own, not the fill's
    try testing.expect(pick(&wait, &fills, &.{ 1, 2, 3, 4, 5, 7 }, 0, testNow) == null);
    const w = pick(&wait, &fills, &.{ 1, 2, 3, 4, 5, 6, 7 }, 0, testNow).?;
    try testing.expectEqual(@as(usize, 5), w.at);
    try testing.expectEqual(@as(?*const anyopaque, &a), w.leader);
    // an unrelated prompt waits for nothing
    try testing.expect(pick(&wait, &fills, &.{ 8, 8, 8 }, 0, testNow) == null);
}

test "a cut already kept serves nothing a kept state does not" {
    var wait: Wait = .{};
    const a: usize = 0;
    const fills = [_]Filler{filler(&a, &.{ 1, 2, 3, 4, 5, 6 }, 2, &.{4})};
    try testing.expect(pick(&wait, &fills, &.{ 1, 2, 3, 4, 5 }, 4, testNow) == null);
    try testing.expect(pick(&wait, &fills, &.{ 1, 2, 3, 4, 5 }, 3, testNow) != null);
}

test "release: the cut reached, the leader gone, or the bound passed" {
    var wait: Wait = .{ .ns = 10 };
    const a: usize = 0;
    const prompt = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const w: Waiting = .{ .leader = &a, .id = 1, .at = 5, .since = 0, .waited = true };
    var fills = [_]Filler{filler(&a, &prompt, 2, &.{5})};
    try testing.expect(holds(&wait, &w, &fills, 5));
    fills[0].reached = 5;
    try testing.expect(!holds(&wait, &w, &fills, 5));
    fills[0].reached = 2;
    try testing.expect(!holds(&wait, &w, &fills, 10));
    try testing.expectEqual(@as(u64, 1), wait.timeouts);
    // a fill gone is a wait over, whatever its marks were
    const gone = [_]Filler{};
    try testing.expect(!holds(&wait, &w, &gone, 1));
    // a different fill at the same address (`id` moved on) is the leader gone
    fills[0].id = 2;
    try testing.expect(!holds(&wait, &w, &fills, 1));
}

test "off waits for nothing" {
    var wait: Wait = .{ .ns = 0 };
    const a: usize = 0;
    const fills = [_]Filler{filler(&a, &.{ 1, 2, 3, 4, 5, 6 }, 0, &.{5})};
    try testing.expect(pick(&wait, &fills, &.{ 1, 2, 3, 4, 5, 6, 7 }, 0, testNow) == null);
}
