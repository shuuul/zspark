//! `scheduler.zig`'s tests, on `Fake` (`scheduler_fake.zig`): each scheduling rule of the file's header, every
//! reply through `Scheduler.decode`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const testing = std.testing;
const request_file = @import("../../src/server/request.zig");
const scheduler_file = @import("../../src/server/scheduler.zig");
const Fake = @import("scheduler_fake.zig").Fake;
const Scheduler = scheduler_file.Scheduler;

// -- tests -------------------------------------------------------------------------------------------------------

const text_fixtures = @import("../text/testing.zig");

const tx = struct {
    const io = testing.io;
    const gpa = testing.allocator;

    /// A document's schema and two tools. (The tiny tokenizer has no `z`.)
    const person = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}," ++
        "\"tags\":{\"type\":\"array\",\"items\":{\"enum\":[\"x\",\"xy\",null]},\"maxItems\":3}},\"required\":[\"name\",\"age\"],\"additionalProperties\":false}";
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"add\",\"parameters\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}," ++
        "\"b\":{\"type\":\"integer\"}},\"required\":[\"a\",\"b\"],\"additionalProperties\":false}}}," ++
        "{\"type\":\"function\",\"function\":{\"name\":\"sub\",\"parameters\":{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"],\"additionalProperties\":false}}}]";

    /// What a client is given, and what it answers.
    const Kept = struct {
        reasoning: std.ArrayList(u8) = .empty,
        content: std.ArrayList(u8) = .empty,
        /// Every piece with its kind, a line each.
        trace: std.ArrayList(u8) = .empty,
        pieces: std.atomic.Value(usize) = .init(0),
        /// Answers `.stop` at this piece (the first is 1): a stop string met.
        stop_at: ?usize = null,
        /// Before that answer, waits until the fake has run this many rounds: the stop is heard late.
        late: ?struct { fake: *Fake, rounds: u64 } = null,
        /// The client has gone: every piece and every question is answered `.stop`.
        gone: std.atomic.Value(bool) = .init(false),
        probes: std.atomic.Value(usize) = .init(0),

        fn deinit(self: *Kept) void {
            self.reasoning.deinit(gpa);
            self.content.deinit(gpa);
            self.trace.deinit(gpa);
        }

        fn sink(self: *Kept) request_file.Sink {
            return .{ .context = self, .writeFn = write };
        }

        fn write(context: *anyopaque, kind: request_file.Kind, text: []const u8) request_file.Sink.Signal {
            const self: *Kept = @ptrCast(@alignCast(context));
            if (self.gone.load(.acquire)) return .stop;
            if (text.len == 0) {
                _ = self.probes.fetchAdd(1, .acq_rel);
                return .more;
            }
            const list = if (kind == .reasoning) &self.reasoning else &self.content;
            list.appendSlice(gpa, text) catch @panic("out of memory");
            self.trace.print(gpa, "{t}:{s}\n", .{ kind, text }) catch @panic("out of memory");
            const pieces = self.pieces.fetchAdd(1, .acq_rel) + 1;
            if (self.stop_at != pieces) return .more;
            if (self.late) |late| while (late.fake.rounds.load(.acquire) < late.rounds) io.sleep(.fromMilliseconds(1), .awake) catch break;
            return .stop;
        }
    };

    const Ask = struct {
        script: u64 = 0,
        prompt: usize = 8,
        max_tokens: ?usize = null,
        thinking: bool = false,
        fail_at: ?usize = null,
        structure: enum { free, document, call } = .free,
        stop_at: ?usize = null,
    };

    const Client = struct {
        ask: Ask,
        kept: Kept = .{},
        outcome: request_file.Error!Fake.Result = error.Failed,
        said: request_file.Diagnostic = .{},
        task: ?Io.Future(void) = null,
        done: std.atomic.Value(bool) = .init(false),
        /// The tokens and rows its connection takes, under `Fake.logprobs`.
        heard: ?*Fake.Heard = null,

        fn send(self: *Client, model: Fake.Model) !void {
            self.task = try io.concurrent(run, .{ self, model });
        }

        fn wait(self: *Client) void {
            if (self.task) |*task| task.await(io);
            self.task = null;
        }

        fn deinit(self: *Client) void {
            self.wait();
            self.kept.deinit();
        }

        fn run(self: *Client, model: Fake.Model) void {
            var arena_state: std.heap.ArenaAllocator = .init(gpa);
            defer arena_state.deinit();
            const arena = arena_state.allocator();
            self.outcome = self.asked(arena, model);
            self.done.store(true, .release);
        }

        fn asked(self: *Client, arena: Allocator, model: Fake.Model) request_file.Error!Fake.Result {
            const ask = self.ask;
            var text: std.ArrayList(u8) = .empty;
            try text.print(arena, "s={d} p={d}", .{ ask.script, ask.prompt });
            if (ask.fail_at) |at| try text.print(arena, " f={d}", .{at});
            self.kept.stop_at = ask.stop_at;
            return model.generate(arena, .{
                .messages = .{ .string = text.items },
                .tools = if (ask.structure == .call) std.json.parseFromSliceLeaky(std.json.Value, arena, tools, .{}) catch return error.Failed else null,
                .thinking = ask.thinking,
                .max_tokens = ask.max_tokens,
                .structure = switch (ask.structure) {
                    .free => null,
                    .document => .{ .document = .{ .spec = .{ .json_schema = person } } },
                    .call => .{ .call = .{} },
                },
                .heard = self.heard,
            }, self.kept.sink(), &self.said);
        }
    };

    /// A fake model behind a scheduler.
    const Bed = struct {
        fake: Fake,
        scheduler: Scheduler,
        /// `Options.probe`; a test that holds a request to `Scheduler.wake` alone sets it long.
        probe: Io.Duration,

        /// The fake, for the test to set; then `start`.
        fn make(self: *Bed, slots: usize) !void {
            self.probe = .fromMilliseconds(2);
            try self.fake.init(gpa, io, slots);
        }

        fn start(self: *Bed) !void {
            errdefer self.fake.deinit();
            self.scheduler = try .init(gpa, io, self.fake.decoder(), .{ .slots = self.fake.slots.len, .probe = self.probe });
            errdefer self.scheduler.deinit();
            try self.scheduler.start();
        }

        fn stop(self: *Bed) void {
            self.scheduler.close();
            self.scheduler.deinit();
            // every job that was made was let go, whatever became of its request
            testing.expectEqual(self.fake.prepared.load(.acquire), self.fake.released.load(.acquire)) catch @panic("a job was not released");
            for (self.fake.slots) |slot| if (slot != null) @panic("a slot was not left");
            self.fake.deinit();
        }

        fn model(self: *Bed) Fake.Model {
            return self.fake.model(&self.scheduler);
        }

        const What = enum { queued, live, admitted, arrived };

        /// Waits for the scheduler to say `n` of `what`; `arrived` is every request it has seen.
        fn until(self: *Bed, what: What, n: u64) !void {
            for (0..10_000) |_| {
                const counts = self.scheduler.counts();
                const now: u64 = switch (what) {
                    .queued => counts.queued,
                    .live => counts.live,
                    .admitted => counts.admitted,
                    .arrived => counts.queued + counts.admitted - counts.preempted,
                };
                if (now == n) return;
                try io.sleep(.fromMilliseconds(1), .awake);
            }
            return error.TestUnexpectedResult;
        }

        /// Lets a round run every few milliseconds until `client` has its reply's end: the rounds do not
        /// outrun a connection that is about to ask for the end.
        fn paced(self: *Bed, client: *Client) !void {
            while (!client.done.load(.acquire)) {
                try io.sleep(.fromMilliseconds(3), .awake);
                self.fake.allow(1);
            }
        }

        /// The reply to `ask` with nobody beside it.
        fn alone(ask: Ask) !Client {
            var bed: Bed = undefined;
            try bed.make(1);
            bed.fake.width = 1;
            try bed.start();
            defer bed.stop();
            var client: Client = .{ .ask = ask };
            client.run(bed.model());
            return client;
        }
    };

    /// `got` is the reply its request has alone: the text, the reasoning, the result or the refusal.
    fn expectAlone(got: *Client) !void {
        var expected = try Bed.alone(got.ask);
        defer expected.deinit();
        try expectSame(&expected, got);
    }

    fn expectSame(expected: *Client, got: *Client) !void {
        errdefer std.debug.print("ask: {any}\n", .{got.ask});
        try testing.expectEqualStrings(expected.kept.reasoning.items, got.kept.reasoning.items);
        try testing.expectEqualStrings(expected.kept.content.items, got.kept.content.items);
        try testing.expectEqualStrings(expected.kept.trace.items, got.kept.trace.items);
        // what a client is given; the decoder's count of its own work is not the same beside others
        if (expected.outcome) |alone| {
            var want = alone;
            var given = try got.outcome;
            want.rounds = 0;
            given.rounds = 0;
            try testing.expectEqual(want, given);
        } else |_| try testing.expectEqual(expected.outcome, got.outcome);
        try testing.expectEqualStrings(expected.said.message(), got.said.message());
    }

    fn expectLog(fake: *Fake, expected: []const Fake.Event) !void {
        errdefer for (fake.log.items) |event| std.debug.print("{t} {d}\n", .{ event.what, event.seed });
        try testing.expectEqual(expected.len, fake.log.items.len);
        for (expected, fake.log.items) |want, got| {
            try testing.expectEqual(want.what, got.what);
            try testing.expectEqual(want.seed, got.seed);
        }
    }
};

test "a reply alone is its script: reasoning, then content, the result counting what was taken" {
    var got = try tx.Bed.alone(.{ .script = 1, .prompt = 20, .thinking = true });
    defer got.deinit();
    try testing.expectEqualStrings("Thinking.", got.kept.reasoning.items);
    try testing.expectEqualStrings(Fake.default_scripts[1], got.kept.content.items);
    const result = try got.outcome;
    try testing.expectEqual(.stop, result.finish);
    try testing.expectEqual(20, result.prompt_tokens);
    try testing.expectEqual(10, result.cached_tokens);
    try testing.expect(result.reasoning_tokens != 0 and result.completion_tokens > result.reasoning_tokens);

    var short = try tx.Bed.alone(.{ .script = 1, .max_tokens = 3 });
    defer short.deinit();
    try testing.expectEqual(.length, (try short.outcome).finish);
    try testing.expectEqual(3, (try short.outcome).completion_tokens);
    try testing.expect(std.mem.startsWith(u8, Fake.default_scripts[1], short.kept.content.items));
}

test "requests are admitted in the order they came, into as many slots as there are" {
    var bed: tx.Bed = undefined;
    try bed.make(2);
    bed.fake.manual = true;
    try bed.start();
    defer bed.stop();
    var clients: [6]tx.Client = undefined;
    for (&clients, 0..) |*client, index| client.* = .{ .ask = .{ .script = 5 - index } };
    defer for (&clients) |*client| client.deinit();
    for (&clients, 1..) |*client, sent| {
        try client.send(bed.model());
        try bed.until(.arrived, sent);
    }
    bed.fake.free();
    for (&clients) |*client| client.wait();
    try testing.expectEqualSlices(u64, &.{ 5, 4, 3, 2, 1, 0 }, bed.fake.order.items);
    try testing.expectEqual(2, bed.fake.most_live.load(.acquire));
    for (&clients) |*client| try tx.expectAlone(client);
}

test "a waiting request is admitted when a slot is left, and a reply's end finds its slot free" {
    var bed: tx.Bed = undefined;
    try bed.make(1);
    bed.fake.manual = true;
    try bed.start();
    defer bed.stop();
    var clients = [_]tx.Client{ .{ .ask = .{ .script = 1 } }, .{ .ask = .{ .script = 0 } }, .{ .ask = .{ .script = 3 } } };
    defer for (&clients) |*client| client.deinit();
    for (&clients, 1..) |*client, sent| {
        try client.send(bed.model());
        try bed.until(.arrived, sent);
    }
    try testing.expectEqual(2, bed.scheduler.counts().queued);
    bed.fake.free();
    for (&clients) |*client| client.wait();
    try tx.expectLog(&bed.fake, &.{
        .{ .what = .admitted, .seed = 1 }, .{ .what = .retired, .seed = 1 },
        .{ .what = .admitted, .seed = 0 }, .{ .what = .retired, .seed = 0 },
        .{ .what = .admitted, .seed = 3 }, .{ .what = .retired, .seed = 3 },
    });
    try testing.expectEqual(1, bed.fake.most_live.load(.acquire));
    // the last reply's end was told after its slot was left
    try testing.expectEqual(0, bed.scheduler.counts().live);
    for (&clients) |*client| try tx.expectAlone(client);
}

test "a request the pool cannot hold yet waits first in line, and nothing passes it" {
    var bed: tx.Bed = undefined;
    try bed.make(3);
    bed.fake.manual = true;
    bed.fake.pool = 200;
    try bed.start();
    defer bed.stop();
    // two prompts that do not fit together, and a third that would fit beside either
    var clients = [_]tx.Client{ .{ .ask = .{ .script = 1, .prompt = 120 } }, .{ .ask = .{ .script = 7, .prompt = 120 } }, .{ .ask = .{ .script = 3, .prompt = 10 } } };
    defer for (&clients) |*client| client.deinit();
    for (&clients, 1..) |*client, sent| {
        try client.send(bed.model());
        try bed.until(.arrived, sent);
    }
    bed.fake.allow(2);
    try bed.until(.queued, 2);
    try testing.expectEqual(1, bed.scheduler.counts().admitted);
    bed.fake.free();
    for (&clients) |*client| client.wait();
    try testing.expectEqualSlices(u64, &.{ 1, 7, 3 }, bed.fake.order.items);
    try tx.expectLog(&bed.fake, &.{
        .{ .what = .admitted, .seed = 1 }, .{ .what = .retired, .seed = 1 },
        .{ .what = .admitted, .seed = 7 }, .{ .what = .admitted, .seed = 3 },
        .{ .what = .retired, .seed = 3 },  .{ .what = .retired, .seed = 7 },
    });
    for (&clients) |*client| try tx.expectAlone(client);
}

test "a request that can never be held is refused as the client's, and the next one is served" {
    var bed: tx.Bed = undefined;
    try bed.make(2);
    bed.fake.pool = 200;
    bed.fake.capacity = 1000;
    try bed.start();
    defer bed.stop();
    for ([_]struct { usize, []const u8 }{
        .{ 300, "the prompt is 300 tokens and the cache pool holds 200: it can never be held" },
        .{ 1000, "This server's maximum context length is 1000 tokens; the prompt is 1000" },
        // it fits the pool and not with a round's room: nothing is live to end and make it
        .{ 199, "the prompt is larger than the cache pool can hold" },
    }) |case| {
        var client: tx.Client = .{ .ask = .{ .prompt = case[0] } };
        defer client.deinit();
        client.run(bed.model());
        try testing.expectError(error.BadRequest, client.outcome);
        try testing.expectEqualStrings(case[1], client.said.message());
        try testing.expectEqualStrings("", client.kept.trace.items);
    }
    try testing.expectEqual(0, bed.scheduler.counts().admitted);
    var served: tx.Client = .{ .ask = .{ .script = 1 } };
    defer served.deinit();
    served.run(bed.model());
    try tx.expectAlone(&served);
}

test "a client that goes while its request waits leaves the queue, and is never decoded" {
    var bed: tx.Bed = undefined;
    try bed.make(1);
    bed.fake.manual = true;
    try bed.start();
    defer bed.stop();
    var clients = [_]tx.Client{ .{ .ask = .{ .script = 1 } }, .{ .ask = .{ .script = 2 } }, .{ .ask = .{ .script = 0 } } };
    defer for (&clients) |*client| client.deinit();
    for (&clients, 1..) |*client, sent| {
        try client.send(bed.model());
        try bed.until(.arrived, sent);
    }
    clients[1].kept.gone.store(true, .release);
    // it leaves on its own thread at its next question: the round being run is still held, and nothing waits for it
    clients[1].wait();
    try testing.expectEqual(0, bed.fake.rounds.load(.acquire));
    const result = try clients[1].outcome;
    try testing.expectEqual(.cut, result.finish);
    try testing.expectEqual(0, result.completion_tokens);
    try testing.expectEqual(0, clients[1].kept.pieces.load(.acquire));
    // the one behind it stays, and is the queue's head and tail
    try testing.expectEqual(1, bed.scheduler.counts().queued);
    bed.fake.free();
    for (&clients) |*client| client.wait();
    try testing.expectEqualSlices(u64, &.{ 1, 0 }, bed.fake.order.items);
    try tx.expectAlone(&clients[0]);
    try tx.expectAlone(&clients[2]);
}

test "a reply cut by its client leaves its slot to the next request, and the reply beside it is its own" {
    var bed: tx.Bed = undefined;
    try bed.make(2);
    bed.fake.manual = true;
    try bed.start();
    defer bed.stop();
    var clients = [_]tx.Client{ .{ .ask = .{ .script = 2, .stop_at = 3 } }, .{ .ask = .{ .script = 4 } }, .{ .ask = .{ .script = 0 } } };
    defer for (&clients) |*client| client.deinit();
    for (&clients, 1..) |*client, sent| {
        try client.send(bed.model());
        try bed.until(.arrived, sent);
    }
    try bed.paced(&clients[0]);
    bed.fake.free();
    for (&clients) |*client| client.wait();
    try testing.expectEqual(.cut, (try clients[0].outcome).finish);
    try testing.expectEqual(3, (try clients[0].outcome).completion_tokens);
    try testing.expectEqual(.stop, (try clients[2].outcome).finish);
    for (&clients) |*client| try tx.expectAlone(client);
    var cancelled: usize = 0;
    for (bed.fake.retired.items) |left| {
        if (left.seed != 2) {
            try testing.expectEqual(.ended, left.why);
            continue;
        }
        cancelled += 1;
        try testing.expectEqual(.cancelled, left.why);
        try testing.expectEqual(3, left.kept);
        try testing.expect(left.generated >= left.kept);
    }
    try testing.expectEqual(1, cancelled);
    try testing.expectEqual(3, bed.fake.retired.items.len);
}

test "a stop heard rounds late: the tokens decoded past it are discarded, the text and the usage a lone reply's" {
    var bed: tx.Bed = undefined;
    try bed.make(2);
    bed.fake.width = 6;
    bed.fake.manual = true;
    try bed.start();
    defer bed.stop();
    var late: tx.Client = .{ .ask = .{ .script = 4, .stop_at = 2 } };
    defer late.deinit();
    late.kept.late = .{ .fake = &bed.fake, .rounds = 5 };
    var beside: tx.Client = .{ .ask = .{ .script = 2 } };
    defer beside.deinit();
    try late.send(bed.model());
    try bed.until(.arrived, 1);
    try beside.send(bed.model());
    try bed.until(.arrived, 2);
    // five rounds and no more until the stop has been said
    bed.fake.allow(5);
    try bed.paced(&late);
    bed.fake.free();
    late.wait();
    beside.wait();
    try testing.expectEqual(2, (try late.outcome).completion_tokens);
    try testing.expectEqual(.cut, (try late.outcome).finish);
    try tx.expectAlone(&late);
    try tx.expectAlone(&beside);
    for (bed.fake.retired.items) |left| if (left.seed == 4) {
        try testing.expectEqual(.cancelled, left.why);
        try testing.expectEqual(2, left.kept);
        // three rounds of tokens at least were decoded while the stop was on its way
        try testing.expect(left.generated >= left.kept + 2);
        // and the reply's counts are the decoder's as far as it ran, though its connection left it
        try testing.expectEqual(left.generated, (try late.outcome).rounds);
    };
}

test "a waiting request asked by wake leaves at once, without the probe's interval or a round's end; the last of the queue and the first alike" {
    var bed: tx.Bed = undefined;
    try bed.make(1);
    bed.fake.manual = true;
    // no question is asked by the clock within the test
    bed.probe = .fromMilliseconds(60_000);
    try bed.start();
    defer bed.stop();
    var clients = [_]tx.Client{ .{ .ask = .{ .script = 1 } }, .{ .ask = .{ .script = 2 } }, .{ .ask = .{ .script = 0 } }, .{ .ask = .{ .script = 3 } } };
    defer for (&clients) |*client| client.deinit();
    for (&clients, 1..) |*client, sent| {
        try client.send(bed.model());
        try bed.until(.arrived, sent);
    }
    // a wake with nobody gone asks everyone and changes nothing
    bed.scheduler.wake();
    while (clients[1].kept.probes.load(.acquire) == 0 or clients[3].kept.probes.load(.acquire) == 0) try tx.io.sleep(.fromMilliseconds(1), .awake);
    try testing.expectEqual(3, bed.scheduler.counts().queued);
    // the last in line, then the first: the queue's tail and its head are set right
    for ([_]usize{ 3, 1 }, [_]usize{ 2, 1 }) |gone, left| {
        clients[gone].kept.gone.store(true, .release);
        bed.scheduler.wake();
        clients[gone].wait();
        try testing.expectEqual(.cut, (try clients[gone].outcome).finish);
        try testing.expectEqual(left, bed.scheduler.counts().queued);
    }
    try testing.expectEqual(0, bed.fake.rounds.load(.acquire));
    // one that comes now stands behind the one that stayed
    var late: tx.Client = .{ .ask = .{ .script = 2 } };
    defer late.deinit();
    try late.send(bed.model());
    try bed.until(.queued, 2);
    bed.fake.free();
    for (&clients) |*client| client.wait();
    late.wait();
    // only those that stayed were ever offered to the decoder
    try testing.expectEqualSlices(u64, &.{ 1, 0, 2 }, bed.fake.order.items);
    try testing.expectEqual(3, bed.scheduler.counts().admitted);
    try tx.expectAlone(&clients[0]);
    try tx.expectAlone(&clients[2]);
    try tx.expectAlone(&late);
}

test "a client that hangs up while its prompt is filled is found by the question, and its slot is left" {
    var bed: tx.Bed = undefined;
    try bed.make(1);
    bed.fake.manual = true;
    bed.fake.chunk = 1;
    try bed.start();
    defer bed.stop();
    var filling: tx.Client = .{ .ask = .{ .script = 0, .prompt = 5000 } };
    defer filling.deinit();
    var next: tx.Client = .{ .ask = .{ .script = 1 } };
    defer next.deinit();
    try filling.send(bed.model());
    try bed.until(.live, 1);
    try next.send(bed.model());
    try bed.until(.queued, 1);
    bed.fake.allow(3);
    while (filling.kept.probes.load(.acquire) == 0) try tx.io.sleep(.fromMilliseconds(1), .awake);
    filling.kept.gone.store(true, .release);
    // a round at a time, so that the prompt is not filled before the question is asked again
    try bed.paced(&filling);
    bed.fake.free();
    filling.wait();
    next.wait();
    try testing.expectEqual(.cut, (try filling.outcome).finish);
    try testing.expectEqualStrings("", filling.kept.trace.items);
    try testing.expectEqual(.cancelled, bed.fake.retired.items[0].why);
    try testing.expectEqual(0, bed.fake.retired.items[0].kept);
    try testing.expect(bed.fake.rounds.load(.acquire) < 5000);
    try tx.expectAlone(&next);
}

test "a round the pool cannot serve takes the newest request off; it runs again and its client sees nothing twice" {
    var bed: tx.Bed = undefined;
    try bed.make(2);
    bed.fake.manual = true;
    bed.fake.pool = 120;
    try bed.start();
    defer bed.stop();
    // two long replies that do not fit together to their ends, and a short one that came after them
    var clients = [_]tx.Client{ .{ .ask = .{ .script = 2, .prompt = 10 } }, .{ .ask = .{ .script = 8, .prompt = 10 } }, .{ .ask = .{ .script = 3 } } };
    defer for (&clients) |*client| client.deinit();
    for (&clients, 1..) |*client, sent| {
        try client.send(bed.model());
        try bed.until(.arrived, sent);
    }
    bed.fake.free();
    for (&clients) |*client| client.wait();
    for (&clients) |*client| try tx.expectAlone(client);
    try testing.expectEqualStrings(Fake.default_scripts[2], clients[1].kept.content.items);
    try testing.expectEqual(1, bed.fake.preempted.load(.acquire));
    try testing.expectEqual(1, bed.fake.replayed.load(.acquire));
    try testing.expectEqual(1, bed.scheduler.counts().preempted);
    // the older reply went on, and the newer came back from the head of the queue once there was room: the
    // request behind it did not pass
    try tx.expectLog(&bed.fake, &.{
        .{ .what = .admitted, .seed = 2 }, .{ .what = .admitted, .seed = 8 }, .{ .what = .preempted, .seed = 8 },
        .{ .what = .retired, .seed = 2 },  .{ .what = .admitted, .seed = 8 }, .{ .what = .admitted, .seed = 3 },
        .{ .what = .retired, .seed = 3 },  .{ .what = .retired, .seed = 8 },
    });
}

test "a reply's log probabilities: each token's row is taken before it and the end's after the reply, through a preemption" {
    var bed: tx.Bed = undefined;
    try bed.make(2);
    bed.fake.manual = true;
    bed.fake.pool = 120;
    bed.fake.logprobs = true;
    try bed.start();
    defer bed.stop();
    // as the preemption's test: the second is taken off and runs again from what its client has
    var clients = [_]tx.Client{ .{ .ask = .{ .script = 2, .prompt = 10 } }, .{ .ask = .{ .script = 8, .prompt = 10 } }, .{ .ask = .{ .script = 3 } } };
    var heard: [clients.len]Fake.Heard = @splat(.{});
    defer for (&heard) |*one| one.deinit(tx.gpa);
    defer for (&clients) |*client| client.deinit();
    for (&clients, &heard) |*client, *one| client.heard = one;
    for (&clients, 1..) |*client, sent| {
        try client.send(bed.model());
        try bed.until(.arrived, sent);
    }
    bed.fake.free();
    for (&clients) |*client| client.wait();
    try testing.expectEqual(1, bed.fake.preempted.load(.acquire));
    for (&clients, &heard) |*client, *one| {
        try tx.expectAlone(client);
        // a row a token, its own, taken just before it
        try testing.expectEqual(0, one.pending);
        try testing.expectEqual(one.tokens.items.len, one.rows.items.len);
        for (one.tokens.items, one.rows.items, 0..) |token, row, at| {
            const want = Fake.rowOf(token, at);
            try testing.expectEqual(want.token, row.token);
            try testing.expectEqual(want.logprob, row.logprob);
            try testing.expectEqual(want.count, row.count);
            try testing.expectEqual(want.ids[0], row.ids[0]);
        }
        const end = one.end_row orelse return error.TestUnexpectedResult;
        try testing.expectEqual(@as(u32, @intCast(Fake.end_token)), end.token);
        try testing.expectEqual(-@as(f32, @floatFromInt(one.tokens.items.len)), end.logprob);
    }
}

test "a lone reply the pool cannot hold fails with what it had sent, and the server goes on" {
    var bed: tx.Bed = undefined;
    try bed.make(2);
    bed.fake.pool = 50;
    try bed.start();
    defer bed.stop();
    var long: tx.Client = .{ .ask = .{ .script = 2, .prompt = 10 } };
    defer long.deinit();
    long.run(bed.model());
    try testing.expectError(error.Failed, long.outcome);
    try testing.expectEqualStrings("the cache pool cannot hold this reply", long.said.message());
    try testing.expect(long.kept.content.items.len != 0 and std.mem.startsWith(u8, Fake.default_scripts[2], long.kept.content.items));
    try testing.expectEqual(.failed, bed.fake.retired.items[0].why);
    var served: tx.Client = .{ .ask = .{ .script = 1 } };
    defer served.deinit();
    served.run(bed.model());
    try tx.expectAlone(&served);
}

test "a request that fails leaves the one beside it whole; a round that fails ends every reply and not the server" {
    var bed: tx.Bed = undefined;
    try bed.make(2);
    bed.fake.manual = true;
    try bed.start();
    defer bed.stop();
    {
        var clients = [_]tx.Client{ .{ .ask = .{ .script = 2, .fail_at = 5 } }, .{ .ask = .{ .script = 4 } } };
        defer for (&clients) |*client| client.deinit();
        for (&clients, 1..) |*client, sent| {
            try client.send(bed.model());
            try bed.until(.arrived, sent);
        }
        bed.fake.free();
        for (&clients) |*client| client.wait();
        try testing.expectError(error.Failed, clients[0].outcome);
        try testing.expectEqualStrings("the scripted reply failed", clients[0].said.message());
        try tx.expectAlone(&clients[0]);
        try tx.expectAlone(&clients[1]);
    }
    {
        bed.fake.hold();
        bed.fake.fail_round = bed.fake.rounds.load(.acquire) + 3;
        var clients = [_]tx.Client{ .{ .ask = .{ .script = 2 } }, .{ .ask = .{ .script = 4 } } };
        defer for (&clients) |*client| client.deinit();
        for (&clients, 1..) |*client, sent| {
            try client.send(bed.model());
            try bed.until(.live, 1);
            try bed.until(.queued, sent - 1);
        }
        bed.fake.free();
        for (&clients) |*client| {
            client.wait();
            try testing.expectError(error.Failed, client.outcome);
            try testing.expectEqualStrings("decoding failed", client.said.message());
        }
    }
    var served: tx.Client = .{ .ask = .{ .script = 1 } };
    defer served.deinit();
    served.run(bed.model());
    try tx.expectAlone(&served);
}

test "closing ends the waiting request with an error and cuts the live one; nothing is taken after" {
    var bed: tx.Bed = undefined;
    try bed.make(1);
    bed.fake.manual = true;
    try bed.start();
    var clients = [_]tx.Client{ .{ .ask = .{ .script = 2 } }, .{ .ask = .{ .script = 0 } } };
    defer for (&clients) |*client| client.deinit();
    for (&clients, 1..) |*client, sent| {
        try client.send(bed.model());
        try bed.until(.arrived, sent);
    }
    var closing = try tx.io.concurrent(Scheduler.close, .{&bed.scheduler});
    while (!bed.scheduler.counts().closing) try tx.io.sleep(.fromMilliseconds(1), .awake);
    bed.fake.allow(1);
    closing.await(tx.io);
    for (&clients) |*client| client.wait();
    try testing.expectEqual(.cut, (try clients[0].outcome).finish);
    try testing.expectError(error.Failed, clients[1].outcome);
    try testing.expectEqualStrings("the server is stopping", clients[1].said.message());
    var after: tx.Client = .{ .ask = .{} };
    defer after.deinit();
    after.run(bed.model());
    try testing.expectError(error.Failed, after.outcome);
    try testing.expectEqualStrings("the server is stopping", after.said.message());
    bed.stop();
}

test "a document and a required call are decoded under their masks beside free replies, each as alone" {
    var bed: tx.Bed = undefined;
    try bed.make(4);
    try bed.start();
    defer bed.stop();
    var arena_state: std.heap.ArenaAllocator = .init(tx.gpa);
    defer arena_state.deinit();
    var clients = [_]tx.Client{
        .{ .ask = .{ .script = 2 } },
        .{ .ask = .{ .script = 11, .structure = .document } },
        .{ .ask = .{ .script = 4, .thinking = true } },
        .{ .ask = .{ .script = 12, .structure = .call } },
        .{ .ask = .{ .script = 13, .structure = .document, .thinking = true } },
        .{ .ask = .{ .script = 14, .structure = .call } },
        .{ .ask = .{ .script = 0 } },
        .{ .ask = .{ .script = 15, .structure = .document } },
    };
    defer for (&clients) |*client| client.deinit();
    for (&clients) |*client| try client.send(bed.model());
    for (&clients) |*client| client.wait();
    try testing.expectEqual(5, bed.fake.constrained.load(.acquire));
    for (&clients) |*client| {
        errdefer std.debug.print("content: {s}\nsaid: {s}\n", .{ client.kept.content.items, client.said.message() });
        try testing.expectEqual(.stop, (try client.outcome).finish);
        switch (client.ask.structure) {
            .free => try testing.expectEqualStrings(Fake.default_scripts[client.ask.script % Fake.default_scripts.len], client.kept.content.items),
            .document => try testing.expect(try text_fixtures.validates(arena_state.allocator(), tx.person, client.kept.content.items)),
            .call => try testing.expect(std.mem.indexOf(u8, client.kept.content.items, "<tool_call>") != null),
        }
        try tx.expectAlone(client);
    }
}

test "many clients at once, whatever the slots, the pool and the rounds' widths: each has the reply it has alone" {
    const iterations: u64 = 48;
    // The lone replies, by request: the same request is asked in many iterations.
    for (0..iterations) |iteration| {
        errdefer std.debug.print("iteration {d}\n", .{iteration});
        var prng: std.Random.DefaultPrng = .init(0xb6 + iteration);
        const random = prng.random();
        const tight = iteration % 3 == 2;
        var bed: tx.Bed = undefined;
        try bed.make(random.intRangeAtMost(usize, 1, 5));
        bed.fake.width = random.intRangeAtMost(usize, 1, 6);
        bed.fake.chunk = random.intRangeAtMost(usize, 2, 16);
        // a tight pool holds any one of its replies and not two long ones: requests wait and are taken off
        if (tight) bed.fake.pool = 150;
        try bed.start();
        defer bed.stop();
        var clients: [10]tx.Client = undefined;
        const count = random.intRangeAtMost(usize, 1, clients.len);
        for (clients[0..count]) |*client| client.* = .{ .ask = .{
            .script = random.uintLessThan(u64, 1000),
            .prompt = random.intRangeAtMost(usize, 1, 20),
            .max_tokens = if (random.uintLessThan(u8, 4) == 0) random.intRangeAtMost(usize, 1, 30) else null,
            .thinking = random.boolean(),
            .structure = if (tight) .free else switch (random.uintLessThan(u8, 8)) {
                0 => .document,
                1 => .call,
                else => .free,
            },
            .stop_at = if (random.uintLessThan(u8, 4) == 0) random.intRangeAtMost(usize, 1, 12) else null,
        } };
        defer for (clients[0..count]) |*client| client.deinit();
        for (clients[0..count]) |*client| {
            try client.send(bed.model());
            // some arrive together, some while others are decoded
            if (random.boolean()) try tx.io.sleep(.fromMicroseconds(random.uintLessThan(u32, 300)), .awake);
        }
        for (clients[0..count]) |*client| client.wait();
        for (clients[0..count]) |*client| try tx.expectAlone(client);
        // a request taken off whose client then says stop leaves from the queue, in no slot
        try testing.expect(bed.fake.retired.items.len <= count and bed.fake.retired.items.len + bed.fake.preempted.load(.acquire) >= count);
        try testing.expect(bed.fake.most_live.load(.acquire) <= bed.fake.slots.len);
        for (bed.fake.retired.items) |left| try testing.expect(left.why != .failed and left.kept <= left.generated);
        const counts = bed.scheduler.counts();
        try testing.expectEqual(0, counts.queued + counts.live);
        try testing.expect(counts.admitted >= count and counts.admitted <= count + counts.preempted);
    }
}

test "a decoder that learns the held tokens in its first round tells them there, and the request is told the same" {
    var bed: tx.Bed = undefined;
    try bed.make(1);
    bed.fake.width = 1;
    bed.fake.held_late = true;
    try bed.start();
    defer bed.stop();
    var client: tx.Client = .{ .ask = .{ .script = 1, .prompt = 20 } };
    defer client.deinit();
    client.run(bed.model());
    const result = try client.outcome;
    try testing.expectEqual(20, result.prompt_tokens);
    try testing.expectEqual(10, result.cached_tokens);
}
