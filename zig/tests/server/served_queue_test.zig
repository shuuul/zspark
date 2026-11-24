//! `served.zig`'s tests of the queue on the host: an engine's Busy and `--max-queued` as the client sees
//! them, a session's turn taken in order, and a pool's queue left by a client that disconnects. Cut from
//! `served_test.zig` by whole declarations (spec 121); the model and the server are `served_fixture.zig`'s.

const std = @import("std");
const api = @import("engine_api");
const host = @import("server");
const request_file = @import("../../src/server/request.zig");
const sched = @import("../../src/server/scheduler.zig");
const served_file = @import("../../src/server/served.zig");
const text_frontend = @import("../../src/text/frontend.zig");
const Allocator = std.mem.Allocator;
const posix = std.posix;
const testing = std.testing;
const fixture = @import("served_fixture.zig");
const Bytes = fixture.Bytes;
const sent = fixture.sent;
const heard = fixture.heard;
const posted = fixture.posted;
const expectFull = fixture.expectFull;
const user = fixture.user;
const Kept = fixture.Kept;

/// Any engine behind TensorFold's server on a port of the loopback, over the tests' text.
const Front = struct {
    text: Bytes,
    server: *host.Server,
    listener: host.listener.Listener,
    stop: std.atomic.Value(bool),
    accepting: std.Thread,
    port: u16,

    fn init(self: *Front, gpa: Allocator, engine: api.Engine) !void {
        self.text = .{};
        self.server = try .init(gpa, testing.io, engine, self.text.text(), .{ .served_name = "hi", .model_ids = &.{"hi"}, .enable_thinking = false, .keep_warm_s = 0 }, null);
        errdefer self.server.deinit();
        self.listener = try .open(.{ .ip4 = .loopback(0) });
        errdefer self.listener.close();
        self.port = self.listener.port();
        self.stop = .init(false);
        self.accepting = try std.Thread.spawn(.{}, host.Server.serve, .{ self.server, self.listener, &self.stop });
    }

    /// Before the engine is closed; the server after it (`finish`), as `Stand.deinit`.
    fn deinit(self: *Front) void {
        self.stop.store(true, .release);
        self.accepting.join();
        while (self.server.open_connections.load(.acquire) != 0) std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        self.listener.close();
    }

    fn finish(self: *Front) void {
        self.server.deinit();
    }

    fn post(self: *Front, arena: Allocator, path: []const u8, body: []const u8) ![]const u8 {
        return heard(arena, try sent(self.port, try posted(arena, path, body)));
    }
};

/// An engine that takes nothing: every `submit` is `error.Busy`.
const Full = struct {
    asked: std.atomic.Value(u32) = .init(0),

    fn engine(self: *Full) api.Engine {
        return .{ .ctx = self, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .name = "full", .lanes = 1, .context_window = 64 };
    }

    fn submit(ctx: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        const self: *Full = @ptrCast(@alignCast(ctx));
        _ = self.asked.fetchAdd(1, .acq_rel);
        return error.Busy;
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

test "an engine's Busy reaches a client as a 503 with the host's words on the three APIs, Messages' an overloaded_error, streamed or not, before any header" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var full: Full = .{};
    var front: Front = undefined;
    try front.init(gpa, full.engine());
    defer front.finish();
    defer front.deinit();

    const chat = "{\"model\":\"hi\"," ++ user;
    const completion = "{\"model\":\"hi\",\"prompt\":\"x\"";
    const responses = "{\"model\":\"hi\",\"input\":\"x\"";
    const messages = "{\"model\":\"hi\",\"max_tokens\":8," ++ user;
    // `chat.zig` names it a capacity refusal, `errors.zig` a 503: `Run.fail` sends the kind's status (`PATCHES.md`)
    try expectFull(try front.post(arena, "/v1/chat/completions", chat ++ "}"), busy_body);
    try expectFull(try front.post(arena, "/v1/completions", completion ++ "}"), busy_body);
    try expectFull(try front.post(arena, "/v1/responses", responses ++ "}"), busy_body);
    try expectFull(try front.post(arena, "/v1/messages", messages ++ "}"), overloaded);
    // since 1.1.1 a stream is submitted before it opens (TensorFold #912): the same 503 before its headers
    try expectFull(try front.post(arena, "/v1/chat/completions", chat ++ ",\"stream\":true}"), busy_body);
    try expectFull(try front.post(arena, "/v1/completions", completion ++ ",\"stream\":true}"), busy_body);
    try expectFull(try front.post(arena, "/v1/responses", responses ++ ",\"stream\":true}"), busy_body);
    try expectFull(try front.post(arena, "/v1/messages", messages ++ ",\"stream\":true}"), overloaded);
    try testing.expectEqual(8, full.asked.load(.acquire));
}

const busy = "the engine is busy; retry shortly";
const overloaded = "{\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\", \"message\": \"" ++ busy ++ "\"}}";
const busy_body = "{\"error\": {\"message\": \"" ++ busy ++ "\", \"type\": \"invalid_request_error\", \"code\": \"engine_busy\"}}";

/// Holds whoever passes until the test lets as many through.
const Gate = struct {
    mutex: std.Io.Mutex = .init,
    opened: std.Io.Condition = .init,
    permits: usize = 0,

    fn pass(self: *Gate) void {
        self.mutex.lockUncancelable(testing.io);
        defer self.mutex.unlock(testing.io);
        while (self.permits == 0) self.opened.waitUncancelable(testing.io, &self.mutex);
        self.permits -= 1;
    }

    fn allow(self: *Gate, more: usize) void {
        self.mutex.lockUncancelable(testing.io);
        defer self.mutex.unlock(testing.io);
        self.permits += more;
        self.opened.broadcast(testing.io);
    }
};

/// One more millisecond of waiting for what a test's other threads must do; five seconds fail it.
fn soon(waited: *usize) !void {
    waited.* += 1;
    if (waited.* > 5000) return error.TestTimedOut;
    try testing.io.sleep(.fromMilliseconds(1), .awake);
}

/// A model on a session: a reply takes the session's turn, is held at the gate, and is `hi`.
const Turns = struct {
    pub const Behind = struct {
        gate: Gate = .{},
        /// Every request the model was given, and those that took the turn: their `max_tokens`, in order.
        replies: std.atomic.Value(u32) = .init(0),
        entered: std.atomic.Value(usize) = .init(0),
        order: [8]usize = undefined,
    };
    const Self = served_file.Served(Turns);

    pub fn reply(live: *Self.Live, _: Allocator, ask: served_file.Ask, _: *request_file.Diagnostic) !served_file.Ended {
        const behind = &live.served.behind;
        _ = behind.replies.fetchAdd(1, .acq_rel);
        if (!live.enters()) return .{ .reason = .cancelled };
        defer live.leaves();
        behind.order[behind.entered.load(.acquire)] = ask.max_tokens;
        _ = behind.entered.fetchAdd(1, .acq_rel);
        behind.gate.pass();
        for ("hi") |byte| _ = live.give(byte);
        return .{ .reason = .stop };
    }

    pub fn counts(_: *Behind) ?sched.Counts {
        return null;
    }

    pub fn closed(_: *Behind) void {}

    /// How many stand in the line for the session's turn.
    fn lined(served: *Self) usize {
        served.mutex.lockUncancelable(testing.io);
        defer served.mutex.unlock(testing.io);
        var count: usize = 0;
        var at = served.first;
        while (at) |one| : (at = one.after) count += 1;
        return count;
    }

    /// Requests the engine holds: decoding and waiting.
    fn held(served: *Self) u32 {
        var status: api.Status = .{};
        served.engine().status(&status, &.{});
        return status.running + status.waiting;
    }
};

test "a session's requests take its turn in the order they came; one cancelled while it waits ends at once, is never decoded, and those behind it move up" {
    const gpa = testing.allocator;
    var frontend: text_frontend.Frontend = undefined;
    var device_bytes: usize = 0;
    var served: Turns.Self = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = null, .behind = .{}, .capacity = 64, .vocab = 256, .lanes = 1, .think = null, .device_bytes = &device_bytes };
    defer served.close();
    const engine = served.engine();
    const behind = &served.behind;
    const prompt = [_]u32{ 1, 2, 3 };
    var kept: [5]Kept = @splat(.{});
    var waited: usize = 0;

    // the first decodes, held at the gate; three wait behind it, each in line before the next is sent
    try engine.submit(0, &.{ .prompt = &prompt, .max_tokens = 10 }, kept[0].sink());
    while (behind.entered.load(.acquire) != 1) try soon(&waited);
    for (1..4) |id| {
        try engine.submit(id, &.{ .prompt = &prompt, .max_tokens = @intCast(10 + id) }, kept[id].sink());
        while (Turns.lined(&served) != id) try soon(&waited);
    }
    // the second of the three leaves: its end comes while the first still decodes
    engine.cancel(2);
    try kept[2].wait();
    try testing.expectEqual(api.Reason.cancelled, kept[2].reason.?);
    try testing.expectEqual(0, kept[2].count);
    try testing.expectEqual(1, behind.entered.load(.acquire));
    try testing.expectEqual(2, Turns.lined(&served));
    try testing.expectEqual(3, Turns.held(&served));
    // a cancel of one that is gone, and of one never sent, is nothing
    engine.cancel(2);
    engine.cancel(9);
    // one that comes now stands last
    try engine.submit(4, &.{ .prompt = &prompt, .max_tokens = 14 }, kept[4].sink());
    while (Turns.lined(&served) != 3) try soon(&waited);
    behind.gate.allow(4);
    for ([_]usize{ 0, 1, 3, 4 }) |id| {
        try kept[id].wait();
        try testing.expectEqual(api.Reason.stop, kept[id].reason.?);
        try testing.expectEqualSlices(u32, &[_]u32{ 'h', 'i' }, kept[id].tokens[0..kept[id].count]);
    }
    try testing.expectEqualSlices(usize, &.{ 10, 11, 13, 14 }, behind.order[0..behind.entered.load(.acquire)]);
    try testing.expectEqual(5, behind.replies.load(.acquire));
}

test "with --max-queued a request beyond those decoded and that many waiting is refused with the host's 503, streamed or not, and never reaches the model; a waiting client that disconnects frees its place at once" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var frontend: text_frontend.Frontend = undefined;
    var device_bytes: usize = 0;
    var served: Turns.Self = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = null, .behind = .{}, .capacity = 64, .vocab = 256, .lanes = 1, .max_queued = 1, .think = null, .device_bytes = &device_bytes };
    var front: Front = undefined;
    try front.init(gpa, served.engine());
    defer front.finish();
    defer served.close();
    defer front.deinit();
    const behind = &served.behind;
    const ask = "{\"model\":\"hi\"," ++ user ++ ",\"max_tokens\":";
    var waited: usize = 0;

    // one decodes and one waits: the server is full
    const first = try sent(front.port, try posted(arena, "/v1/chat/completions", ask ++ "10}"));
    while (behind.entered.load(.acquire) != 1) try soon(&waited);
    const second = try sent(front.port, try posted(arena, "/v1/chat/completions", ask ++ "11}"));
    while (Turns.lined(&served) != 1) try soon(&waited);
    try expectFull(try front.post(arena, "/v1/chat/completions", ask ++ "12}"), busy_body);
    try expectFull(try front.post(arena, "/v1/messages", ask ++ "12}"), overloaded);
    // streamed, the same 503 before any header of a stream (TensorFold #912), as vLLM's `_preflight`
    try expectFull(try front.post(arena, "/v1/chat/completions", ask ++ "12,\"stream\":true}"), busy_body);
    try expectFull(try front.post(arena, "/v1/responses", "{\"model\":\"hi\",\"input\":\"x\",\"stream\":true}"), busy_body);
    try testing.expectEqual(2, behind.replies.load(.acquire));
    try testing.expectEqual(2, Turns.held(&served));

    // the one that waits hangs up: the host finds it gone and cancels, and it has left before the first ends
    _ = posix.system.close(second);
    while (Turns.held(&served) != 1) try soon(&waited);
    try testing.expectEqual(1, behind.entered.load(.acquire));
    try testing.expectEqual(0, Turns.lined(&served));
    // so the next request is taken, and waits its turn
    const third = try sent(front.port, try posted(arena, "/v1/chat/completions", ask ++ "13}"));
    while (Turns.lined(&served) != 1) try soon(&waited);
    try expectFull(try front.post(arena, "/v1/chat/completions", ask ++ "14}"), busy_body);

    behind.gate.allow(2);
    for ([_]posix.socket_t{ first, third }) |fd| {
        const reply = try heard(arena, fd);
        errdefer std.debug.print("the reply:\n{s}\n", .{reply});
        try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 OK\r\n"));
        try testing.expect(std.mem.indexOf(u8, reply, "\"content\": \"hi\"") != null);
    }
    try testing.expectEqualSlices(usize, &.{ 10, 13 }, behind.order[0..behind.entered.load(.acquire)]);
    // a reply's end frees its place before the client has it: the next request of the same client is taken
    behind.gate.allow(1);
    const after = try front.post(arena, "/v1/chat/completions", ask ++ "15}");
    try testing.expect(std.mem.startsWith(u8, after, "HTTP/1.1 200 OK\r\n"));
    // the three refused were never the model's; the one that hung up was, and took no turn
    try testing.expectEqual(4, behind.replies.load(.acquire));
    try testing.expectEqual(3, behind.entered.load(.acquire));
}

/// A pool of one slot whose round is held at the gate and then gives `hi`: the scheduler's decoder.
const Deck = struct {
    gate: Gate = .{},
    admits: std.atomic.Value(u32) = .init(0),
    rounds: std.atomic.Value(u32) = .init(0),

    fn decoder(self: *Deck) sched.Decoder {
        return .{ .context = self, .table = &.{ .admit = admit, .round = round, .retire = retire, .preempt = preempt } };
    }

    fn admit(context: *anyopaque, _: *sched.Job, _: []const i32, _: *request_file.Diagnostic) sched.Admission {
        const self: *Deck = @ptrCast(@alignCast(context));
        _ = self.admits.fetchAdd(1, .acq_rel);
        return .{ .admitted = .{ .slot = 0 } };
    }

    fn round(context: *anyopaque, told: *sched.Round) sched.Outcome {
        const self: *Deck = @ptrCast(@alignCast(context));
        self.gate.pass();
        _ = self.rounds.fetchAdd(1, .acq_rel);
        told.emit(0, &.{ 'h', 'i' });
        told.end(0, .stop);
        return .ran;
    }

    fn retire(_: *anyopaque, _: usize, _: sched.Leave, _: usize) sched.Tail {
        return .{};
    }

    fn preempt(_: *anyopaque, _: usize) void {
        unreachable;
    }
};

/// A model on a pool: a reply is the scheduler's.
const Pooled = struct {
    pub const Behind = struct { scheduler: *sched.Scheduler };
    const Self = served_file.Served(Pooled);

    pub fn reply(live: *Self.Live, _: Allocator, _: served_file.Ask, said: *request_file.Diagnostic) !served_file.Ended {
        const done = try live.served.behind.scheduler.decode(@ptrCast(live), live.taker(), said);
        live.closing(done.finish, done.tail.end);
        return .{ .reason = live.reasonOf(done.finish) };
    }

    pub fn counts(behind: *Behind) ?sched.Counts {
        return behind.scheduler.counts();
    }

    pub fn closed(_: *Behind) void {}
};

test "in a pool a queued request whose client disconnects leaves the scheduler's queue at once: before the round being run ends, without the probe's interval, and no decoder is offered it" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var deck: Deck = .{};
    // no question by the clock within the test: only `cancel`'s wake can take the request out
    var scheduler: sched.Scheduler = try .init(gpa, testing.io, deck.decoder(), .{ .slots = 1, .probe = .fromMilliseconds(60_000) });
    defer scheduler.deinit();
    try scheduler.start();
    defer scheduler.close();
    var frontend: text_frontend.Frontend = undefined;
    var device_bytes: usize = 0;
    var served: Pooled.Self = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = null, .behind = .{ .scheduler = &scheduler }, .capacity = 64, .vocab = 256, .lanes = 1, .think = null, .device_bytes = &device_bytes };
    var front: Front = undefined;
    try front.init(gpa, served.engine());
    defer front.finish();
    defer served.close();
    defer front.deinit();
    const ask = "{\"model\":\"hi\"," ++ user ++ "}";
    var waited: usize = 0;

    const first = try sent(front.port, try posted(arena, "/v1/chat/completions", ask));
    while (deck.admits.load(.acquire) != 1) try soon(&waited);
    // streamed and not: both are found gone while they wait
    for ([_][]const u8{ ask, "{\"model\":\"hi\"," ++ user ++ ",\"stream\":true}" }) |body| {
        const second = try sent(front.port, try posted(arena, "/v1/chat/completions", body));
        while (scheduler.counts().queued != 1) try soon(&waited);
        _ = posix.system.close(second);
        while (scheduler.counts().queued != 0) try soon(&waited);
        // its request ends too, while the first's round is still held
        var status: api.Status = .{};
        while (true) {
            served.engine().status(&status, &.{});
            if (status.running + status.waiting == 1) break;
            try soon(&waited);
        }
    }
    try testing.expectEqual(0, deck.rounds.load(.acquire));
    try testing.expectEqual(1, deck.admits.load(.acquire));
    deck.gate.allow(1);
    const reply = try heard(arena, first);
    errdefer std.debug.print("the reply:\n{s}\n", .{reply});
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 OK\r\n"));
    try testing.expect(std.mem.indexOf(u8, reply, "\"content\": \"hi\"") != null);
    try testing.expectEqual(1, scheduler.counts().admitted);
}
