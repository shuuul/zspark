//! Tests of the prefix wait on GLM's hall (spec 152 WS-06, `decode/prefix_wait.zig`): a request whose
//! prompt shares the cut another prompt fills waits (`admit` returns null, `pending` holds the record),
//! is served once the fill's state stands at the end, is freed by the bound, and waits on nothing when
//! the wait is off or the prompts share nothing. `check.sh glm5 hall` runs it; the random GLM in loopback.

const std = @import("std");
const testing = std.testing;
const hall_file = @import("../../../../src/models/glm5/pool/hall.zig");
const hall_fixture = @import("hall_fixture.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");

const greedy = hall_fixture.greedy;
const Fixture = hall_fixture.Fixture;
const Hall = hall_file.HallOf(layout.world);

test "a request waits for the prompt fill that shares its cut, then takes the state at its end" {
    var f: Fixture = undefined;
    try f.init(0x5eed, 400, 160);
    defer f.deinit();
    const ids = f.fixture.tokens;
    var hall = try f.hall(2, 160, false);
    defer hall.deinit();
    hall.prefix_wait = .{ .ns = 5 * std.time.ns_per_s };
    const none: []const u32 = &.{};

    const leader: hall_file.Request = .{ .prompt = ids[0..80], .settings = greedy, .eos = none, .max_tokens = 4 };
    _ = (try hall.admit(leader)).?;
    // the same prompt waits on the fill
    try testing.expect((try hall.admit(leader)) == null);
    try testing.expectEqual(@as(usize, 1), hall.pending.items.len);
    try testing.expectEqual(@as(u64, 1), hall.prefix_wait.waits);

    // a round feeds the prompt; still short of the end, the waiter waits
    var sink = struct {
        pub fn tokens(_: *@This(), _: usize, _: []const i32) void {}
        pub fn ended(_: *@This(), _: usize, _: hall_file.Ended) void {}
    }{};
    var spins: usize = 0;
    var taken: ?hall_file.Admitted = null;
    while (hall.streams[0].fed < hall.streams[0].prompt) : (spins += 1) {
        _ = try hall.round(&sink);
        try testing.expect(spins < 200);
        if ((try hall.admit(leader))) |admitted| {
            taken = admitted;
            break;
        }
    }
    taken = taken orelse try hall.admit(leader);
    // the kept state at the cut (the prompt's last position) served it: one row of it is run again
    try testing.expectEqual(@as(usize, 79), taken.?.cached);
    hall.retire(taken.?.slot);
}

test "the bound frees the waiter, and an unrelated prompt waits on nothing" {
    var f: Fixture = undefined;
    try f.init(0xbeef, 400, 160);
    defer f.deinit();
    const ids = f.fixture.tokens;
    var hall = try f.hall(2, 160, false);
    defer hall.deinit();
    hall.prefix_wait = .{ .ns = 1 }; // a nanosecond: every second admit is over it
    const none: []const u32 = &.{};

    const leader: hall_file.Request = .{ .prompt = ids[0..80], .settings = greedy, .eos = none, .max_tokens = 4 };
    _ = (try hall.admit(leader)).?;
    try testing.expect((try hall.admit(leader)) == null);
    // over the bound at once: the waiter takes a slot and fills
    const taken = try hall.admit(leader);
    try testing.expect(taken != null);
    try testing.expectEqual(@as(usize, 0), hall.pending.items.len);
    try testing.expectEqual(@as(u64, 1), hall.prefix_wait.timeouts);
    hall.retire(taken.?.slot);

    // an unrelated prompt waits on nothing
    const other: hall_file.Request = .{ .prompt = ids[200..240], .settings = greedy, .eos = none, .max_tokens = 4 };
    try testing.expect((try hall.admit(other)) != null);
}

test "off: the same prompt fills at once" {
    var f: Fixture = undefined;
    try f.init(0xf00d, 400, 160);
    defer f.deinit();
    const ids = f.fixture.tokens;
    var hall = try f.hall(2, 160, false);
    defer hall.deinit();
    hall.prefix_wait = .{ .ns = 0 };
    const none: []const u32 = &.{};
    const leader: hall_file.Request = .{ .prompt = ids[0..80], .settings = greedy, .eos = none, .max_tokens = 4 };
    _ = (try hall.admit(leader)).?;
    try testing.expect((try hall.admit(leader)) != null);
    try testing.expectEqual(@as(u64, 0), hall.prefix_wait.waits);
}
