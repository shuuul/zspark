//! `pool.zig`'s test on the random GLM in loopback: requests on tasks of their own through the scheduler's
//! decoding thread, more of them than slots. Each takes the serial reply of the model alone, token for token;
//! the token that ends a reply is its tail's and none of its tokens; a taker that says stop is given no more
//! and leaves the others theirs.

const std = @import("std");
const testing = std.testing;
const request_file = @import("../../../../src/server/request.zig");
const scheduler_file = @import("../../../../src/server/scheduler.zig");
const hall_file = @import("../../../../src/models/glm5/pool/hall.zig");
const hall_fixture = @import("hall_fixture.zig");
const shared = @import("../testing.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const pool_file = @import("../../../../src/models/glm5/pool/pool.zig");

const fx = @import("../../../../src/models/glm5/checks/random.zig").fx;
const Pool = pool_file.PoolOf(layout.world);

/// A request's own side: it keeps what it is given and says stop once it holds `limit`.
const Client = struct {
    pool: *Pool,
    job: Pool.Job,
    limit: usize = std.math.maxInt(usize),
    tokens: [64]i32 = undefined,
    count: usize = 0,
    cached: ?usize = null,
    outcome: request_file.Error!scheduler_file.Done = error.Failed,
    said: request_file.Diagnostic = .{},
    task: ?std.Io.Future(void) = null,

    fn run(self: *Client) void {
        self.outcome = self.pool.decode(&self.job, .{ .context = self, .take = take, .probe = probe, .began = began }, &self.said);
    }

    fn began(context: *anyopaque, cached: usize, _: i96) void {
        const self: *Client = @ptrCast(@alignCast(context));
        self.cached = cached;
    }

    fn take(context: *anyopaque, token: i32) request_file.Sink.Signal {
        const self: *Client = @ptrCast(@alignCast(context));
        self.tokens[self.count] = token;
        self.count += 1;
        return if (self.count >= self.limit) .stop else .more;
    }

    fn probe(_: *anyopaque) request_file.Sink.Signal {
        return .more;
    }
};

test "the pool: requests of their own tasks decoded several at once are each the serial reply, an end token is the tail's, a taker's stop ends its reply alone" {
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // held to serial steps: one form, a step's (`testing.oneForm`)
    const forms = shared.oneForm();
    defer forms.restore();
    const capacity = 160;
    var f: hall_fixture.Fixture = undefined;
    try f.init(0x4a21, 300, capacity);
    defer f.deinit();
    f.head.confidence = 0;
    const ids = f.fixture.tokens;
    const none: []const u32 = &.{};

    const requests = [_]hall_file.Request{
        .{ .prompt = ids[0..23], .settings = hall_fixture.greedy, .eos = none, .max_tokens = 30 },
        .{ .prompt = ids[30..37], .settings = hall_fixture.sampled(71), .eos = none, .max_tokens = 41 },
        .{ .prompt = ids[40..41], .settings = hall_fixture.sampled(72), .eos = none, .max_tokens = 9 },
        .{ .prompt = ids[50..163], .settings = hall_fixture.greedy, .eos = none, .max_tokens = 17 },
        .{ .prompt = ids[190..250], .settings = hall_fixture.sampled(74), .eos = none, .max_tokens = 33, .drafts = false },
        .{ .prompt = ids[260..269], .settings = hall_fixture.sampled(75), .eos = none, .max_tokens = 25 },
    };
    // before the decoding thread has the device
    var serial: [requests.len][]i32 = undefined;
    for (&serial, requests) |*want, r| want.* = try shared.serialReply(arena, f.loop.m, fx.vocab, r.prompt, r.max_tokens, r.settings);
    // the second reply ends at its first token after the third that it has not made before; the last is cut
    // after four
    const stops = for (3..serial[1].len) |at| {
        if (std.mem.indexOfScalar(i32, serial[1][0..at], serial[1][at]) == null) break at;
    } else return error.TestUnexpectedResult;
    const end = [_]u32{@intCast(serial[1][stops])};

    var hall = try f.hall(3, capacity, true);
    defer hall.deinit();
    const pool = try Pool.create(gpa, io, &hall);
    defer pool.destroy();
    var clients: [requests.len]Client = undefined;
    for (&clients, requests) |*one, r| one.* = .{ .pool = pool, .job = .{ .request = r } };
    clients[1].job.request.eos = &end;
    clients[5].limit = 4;
    for (&clients) |*one| one.task = try io.concurrent(Client.run, .{one});
    for (&clients) |*one| one.task.?.await(io);
    pool.close();

    for (&clients, serial, requests, 0..) |*one, want, r, index| {
        const done = try one.outcome;
        // told before a reply's first token: a reply that is its end token alone has none
        try testing.expectEqual(@as(?usize, if (one.count > 0) 0 else null), one.cached);
        try testing.expectEqual(one.count, done.tokens);
        switch (index) {
            1 => {
                try testing.expectEqual(request_file.Finish.stop, done.finish);
                try testing.expectEqualSlices(i32, want[0..stops], one.tokens[0..one.count]);
                try testing.expectEqual(@as(?i32, want[stops]), done.tail.end);
            },
            5 => {
                try testing.expectEqual(request_file.Finish.cut, done.finish);
                try testing.expectEqualSlices(i32, want[0..4], one.tokens[0..one.count]);
            },
            else => {
                try testing.expectEqual(request_file.Finish.length, done.finish);
                try testing.expectEqualSlices(i32, want, one.tokens[0..one.count]);
                try testing.expectEqual(@as(?i32, null), done.tail.end);
                try testing.expect(done.tail.rounds >= 1 and done.tail.rounds < r.max_tokens + 1 and done.tail.prompt_seconds != null);
                if (!r.drafts) try testing.expectEqual(@as(u64, 0), done.tail.drafted);
            },
        }
    }
    const counts = pool.scheduler.counts();
    try testing.expectEqual(@as(u64, requests.len), counts.admitted);
    try testing.expectEqual(@as(u64, 0), counts.preempted);
    try testing.expect(!pool.broken.load(.acquire));
    try testing.expectEqual(@as(usize, 0), hall.live());
}
