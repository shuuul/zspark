//! The pool's replies against the session's and the hall's lone stream's, and the pool's sizes (`hall.zig`'s
//! tests beside `hall_test.zig`): a reply's tokens, count and end the same from a session, from the lone stream
//! and from the pool; the prompt windows `serve` loads for; the pool's positions; a prompt chunk one row short
//! of the wide kernels beside three full windows.

const std = @import("std");
const testing = std.testing;
const fixtures = @import("../testing.zig");
const hall_fixture = @import("hall_fixture.zig");
const Bench = hall_fixture.Bench;
const Collect = hall_fixture.Collect;
const engineFor = hall_fixture.engineFor;
const greedy = hall_fixture.greedy;
const openPool = hall_fixture.openPool;
const extents = @import("../../../../src/decode/extents.zig");
const request_file = @import("../../../../src/server/request.zig");
const sched = @import("../../../../src/server/scheduler.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const gdn = @import("../../../../src/models/qwen4/blocks/gdn.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_front = @import("../../../../src/models/qwen4/pool/hall_front.zig");
const session = @import("../../../../src/models/qwen4/session/session.zig");
const Out = hall_file.Out;

/// A reply's tokens as `Scheduler.decode` gives them to the thread that asked.
const Taken = struct {
    tokens: [16]i32 = undefined,
    count: usize = 0,
    /// The prompt's tokens that were not run, once told.
    cached: ?usize = null,

    fn taker(self: *Taken) sched.Taker {
        return .{ .context = self, .take = take, .probe = probe, .began = began };
    }

    fn take(context: *anyopaque, token: i32) request_file.Sink.Signal {
        const self: *Taken = @ptrCast(@alignCast(context));
        self.tokens[self.count] = token;
        self.count += 1;
        return .more;
    }

    fn probe(_: *anyopaque) request_file.Sink.Signal {
        return .more;
    }

    fn began(context: *anyopaque, cached_tokens: usize, _: i96) void {
        const self: *Taken = @ptrCast(@alignCast(context));
        self.cached = cached_tokens;
    }
};

test "a reply's tokens, their count and its end are the same from a session, from the hall's lone stream and from the pool: ended by eos, at its first token, and by max_tokens" {
    const template_file = @import("../../../../src/text/template.zig");
    const text_frontend = @import("../../../../src/text/frontend.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var m = try fixtures.random_model.testModelRows(gpa, arena, null, 48);
    defer m.deinit();
    const source = "{% for message in messages %}<|end|>{{ message.role }} {{ message.content }}\n{% endfor %}<|end|>assistant ";
    var template_diagnostic: template_file.Diagnostic = .{};
    const parsed = try template_file.Template.parse(gpa, source, &template_diagnostic);
    var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
    const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
    var frontend: text_frontend.Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
    defer frontend.deinit(gpa);
    const list = try std.json.parseFromSliceLeaky(std.json.Value, arena, "[{\"role\":\"user\",\"content\":\"the cat sat on the mat\"}]", .{});
    var text_diagnostic: text_frontend.Diagnostic = .{};
    const ids = try frontend.prompt(arena, .{ .messages = list, .thinking = false }, &text_diagnostic);
    const prompt = try arena.alloc(i32, ids.len);
    for (prompt, ids) |*token, id| token.* = @intCast(id);
    const starts = try frontend.starts(arena, prompt);
    const serial = try fixtures.serial.serialReply(gpa, m, prompt, 12, greedy);
    defer gpa.free(serial);

    // the reply ends at its sixth token or before, at its first, and at `max_tokens` with no end token
    const Case = struct { eos: []const u32, max_tokens: usize, tokens: usize, finish: request_file.Finish };
    const sixth = [1]u32{@intCast(serial[5])};
    const first = [1]u32{@intCast(serial[0])};
    for ([_]Case{
        .{ .eos = &sixth, .max_tokens = 12, .tokens = std.mem.indexOfScalar(i32, serial, serial[5]).?, .finish = .stop },
        .{ .eos = &first, .max_tokens = 12, .tokens = 0, .finish = .stop },
        .{ .eos = &.{}, .max_tokens = 7, .tokens = 7, .finish = .length },
    }) |case| {
        errdefer std.debug.print("a reply of {d} tokens that ends as {t}\n", .{ case.tokens, case.finish });
        var said: request_file.Diagnostic = .{};
        // the token that ended the reply, which is none of its tokens
        const end: ?i32 = if (case.finish == .stop) serial[case.tokens] else null;

        // `--parallel 1`: `Session.reply`, as `served.zig` calls it
        const alone: *session.Session = try .create(gpa, testing.io, m, &frontend.tokenizer, case.eos, .{ .capacity = 256, .prompt_rows = 32 });
        defer alone.destroy();
        var record: [12]i32 = undefined;
        var counting: fixtures.sessions.Counting = .{};
        const from_session = try alone.reply(prompt, greedy, .{ .max_tokens = case.max_tokens, .record = &record, .starts = starts }, counting.sink());
        try testing.expectEqual(case.finish, from_session.finish);
        try testing.expectEqual(case.tokens, from_session.tokens);
        try testing.expectEqualSlices(i32, serial[0..case.tokens], record[0..from_session.tokens]);
        try testing.expectEqual(end, from_session.end);
        try testing.expectEqual(0, from_session.reasoning_tokens);

        // `--parallel 2`: its lone stream, then the pool, opened by two requests at once; each a job made
        // and given to `Scheduler.decode`, as `served.zig` does
        const hall = try hall_file.Hall.create(gpa, testing.io, m, &frontend.tokenizer, case.eos, .{
            .slots = 2,
            .session = .{ .capacity = 256, .prompt_rows = 32 },
            .pool = extents.lowerBound(256, 2) * extents.positions,
            .chunk = 8,
        });
        defer hall.destroy();
        var bench: Bench = .{ .gpa = gpa };
        defer bench.deinit();
        for ([_]bool{ false, true }) |pooled| {
            errdefer std.debug.print("{s}\n", .{if (pooled) "in the pool" else "the lone stream"});
            // The decoding thread retired the slot before `decode` returned and waits for work: the pool is
            // opened from here, by the decoder's own calls.
            if (pooled) try openPool(&bench, hall.core);
            const job = try hall_front.makeJob(hall, .{ .prompt = prompt, .settings = greedy, .max_tokens = case.max_tokens, .thinking = false, .constraint = null }, session.default_depth, session.default_depth, starts, null, null, null);
            defer hall_front.releaseJob(hall, job);
            var taken: Taken = .{};
            const done = try hall.scheduler.decode(job, taken.taker(), &said);
            try testing.expectEqual(pooled, hall.core.engine != null);
            try testing.expectEqual(case.finish, done.finish);
            try testing.expectEqual(case.tokens, done.tokens);
            try testing.expectEqualSlices(i32, record[0..from_session.tokens], taken.tokens[0..taken.count]);
            try testing.expectEqual(end, done.tail.end);
            // told before the first token, where one came
            if (case.tokens != 0) try testing.expect(taken.cached != null);
        }
    }
}

test "serve's session runs a prompt in windows of what was asked whatever --parallel loads the model for: unset 4096 on one rank, the pool's rounds 2048" {
    const config: @import("../../../../src/models/qwen4/config.zig").Config = undefined;
    const lone = session.lonePromptRows(config);
    const round = session.promptRows(config);
    // one rank's default (spec 148 WS-02), the widest `--max-num-batched-tokens` takes, and the least (spec 105 WS-07)
    try testing.expectEqual(@as(usize, 4096), lone);
    for ([_]usize{ lone, round, session.most_prompt_rows }) |want| {
        const options = hall_file.sessionFor(want, 4096, 256);
        try testing.expectEqual(4096, options.capacity);
        try testing.expectEqual(256, options.reply_stride);
        for ([_]usize{ 1, 2, 4, 16, 64 }) |parallel| {
            errdefer std.debug.print("--parallel {d}, rows {d}\n", .{ parallel, want });
            // `--parallel 1`'s session and the hall's lone stream's are this one, on the model `serve` loads
            try testing.expectEqual(want, options.promptWindow(hall_file.rowsFor(want, parallel)));
        }
    }
    // the pool of four's rounds stay 2048 rows, whatever the lone stream's windows
    try testing.expectEqual(@as(usize, 2048), hall_file.rowsFor(round, hall_file.default_parallel));
    // outside 2048 to 4096 none is taken
    for ([_]usize{ round, lone, session.most_prompt_rows }) |rows| try testing.expect(session.refusedRows(config, rows) == null);
    try testing.expect(session.refusedRows(config, session.most_prompt_rows + 1) != null);
    try testing.expect(session.refusedRows(config, round - 1) != null);
}

test "serve's pool is the streams' contexts: the defaults' is 1,048,576 positions, and a small context's no less than the lower bound" {
    const poolFor = hall_file.poolFor;
    try testing.expectEqual(@as(usize, 1_048_576), poolFor(hall_file.default_context, hall_file.default_parallel));
    try testing.expectEqual(@as(usize, 327_680), poolFor(81920, 4));
    try testing.expectEqual(@as(usize, 2_097_152), poolFor(262144, 8));
    // what `openAfter` and `extents.Pool.init` ask of any pool
    for ([_]usize{ 64, 256, 512, 600, 81920 }) |context| {
        for ([_]usize{ 2, 4, 64 }) |parallel| {
            const pool = poolFor(context, parallel);
            try testing.expect(pool % extents.positions == 0);
            try testing.expect(pool / extents.positions >= extents.lowerBound(context, parallel));
            try testing.expect(pool >= parallel * context);
        }
    }
    try testing.expectEqual(extents.lowerBound(64, 4) * extents.positions, poolFor(64, 4));
}

test "in the pool a prompt chunk one row short of the wide kernels beside three full windows is within each layer's short projection: every reply is the serial one" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 128);
    defer m.deinit();
    // no head: the windows' drafts are the test's, each window as wide as one gets whatever a head would chain
    const head = m.mtp;
    m.mtp = null;
    defer m.mtp = head;
    const count: usize = 40;
    var random: std.Random.DefaultPrng = .init(86);
    var tokens: [4][64]i32 = undefined;
    for (&tokens) |*row| {
        for (row) |*token| token.* = random.random().intRangeLessThan(i32, 1, 250);
    }
    // spec 086: three replies under way and a prompt of 29 to 31 tokens, which is not a wide segment
    const chunk = gdn.wide_rows - 1;
    try testing.expect(chunk <= tokens[3].len);
    const prompts = [_][]const i32{ tokens[0][0..3], tokens[1][0..4], tokens[2][0..5], tokens[3][0..chunk] };
    var serial: [4][]i32 = undefined;
    var made: usize = 0;
    defer for (serial[0..made]) |reply| gpa.free(reply);
    for (prompts) |prompt| {
        serial[made] = try fixtures.serial.serialReply(gpa, m, prompt, count, greedy);
        made += 1;
    }

    var engine = try engineFor(gpa, m, 4, 64);
    engine.bindPool();
    defer engine.deinit();
    var lists: [4]std.ArrayList(i32) = @splat(.empty);
    defer for (&lists) |*list| list.deinit(gpa);
    var ended: [4]bool = @splat(false);
    var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended };
    const out: Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail };
    for (prompts[0..3], 0..) |prompt, i| {
        _ = try engine.begin(i, .{ .prompt = prompt, .settings = greedy, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, 0);
    }
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
            // drafts no model made: the round keeps of them what the model follows, and its tokens are its own
            for (&seat.drafts, 0..) |*draft, d| draft.* = @intCast(1 + (spin * 7 + d * 13) % 249);
            seat.draft_n = seat.drafts.len - 1;
            window = seat.drafts.len;
        }
        if (!begun and decoding == 3) {
            _ = try engine.begin(3, .{ .prompt = prompts[3], .settings = greedy, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, 0);
            begun = true;
        }
        switch (engine.round(out)) {
            .ran => engine.publishAhead(out),
            .short => return error.Short,
            .failed => return error.Failed,
        }
        widest = @max(widest, engine.staged.used);
    }
    // the round that took the prompt had the three windows beside it, and no segment of it was wide
    try testing.expectEqual(3 * window + chunk, widest);
    try testing.expect(widest <= engine.buffers.kept);
    for (serial, lists, ended, 0..) |want, got, done, i| {
        try testing.expect(done);
        try testing.expectEqualSlices(i32, want, got.items);
        _ = engine.retire(i, .ended, count);
    }
}
