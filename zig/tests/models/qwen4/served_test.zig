//! `served.zig`'s tests: a reply's tokens through TensorFold's engine interface are the serial reply's, from
//! a session and from the pool's lone stream; a required call and a document open as the template writes them.

const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const host = @import("server");
const command = @import("../../../src/server/command.zig");
const fixtures = @import("testing.zig");
const Allocator = std.mem.Allocator;
const posix = std.posix;
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const extents = @import("../../../src/decode/extents.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const tokenizer = @import("../../../src/text/tokenizer.zig");
const hall_file = @import("../../../src/models/qwen4/pool/hall.zig");
const served_file = @import("../../../src/models/qwen4/served.zig");
const session = @import("../../../src/models/qwen4/session/session.zig");

const greedy: sampling.Sampling = .{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };

/// A request's events as its server would keep them.
const Heard = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    changed: std.Io.Condition = .init,
    tokens: [64]u32 = undefined,
    count: usize = 0,
    /// The rows of `Event.logprobs`, and whether each came alone just before its token's `tokens` event.
    rows: [64]api.LogprobRow = undefined,
    row_count: usize = 0,
    out_of_step: bool = false,
    /// The cached count, and how many tokens had come when it was told.
    prefilled: ?u32 = null,
    before: usize = 0,
    reason: ?api.Reason = null,
    rounds: u64 = 0,
    prefill_seconds: ?f64 = null,
    /// Asks for the end once this many tokens have come.
    cancel_at: ?usize = null,
    engine: ?api.Engine = null,

    fn sink(self: *Heard) api.Sink {
        return .{ .ctx = self, .event = event };
    }

    fn event(ctx: *anyopaque, id: api.Id, given: *const api.Event) void {
        const self: *Heard = @ptrCast(@alignCast(ctx));
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        switch (given.*) {
            .prefilled => |cached| {
                self.prefilled = cached;
                self.before = self.count;
            },
            .logprobs => |rows| {
                if (rows.len != 1 or self.row_count != self.count) self.out_of_step = true;
                for (rows) |row| {
                    self.rows[self.row_count] = row;
                    self.row_count += 1;
                }
            },
            .prompt_logprobs, .audio => {},
            .tokens => |tokens| for (tokens) |token| {
                self.tokens[self.count] = token;
                self.count += 1;
                if (self.row_count != 0 and self.row_count != self.count) self.out_of_step = true;
            },
            .finished => |finished| {
                self.reason = finished.reason;
                self.rounds = finished.stats.rounds;
                self.prefill_seconds = finished.stats.prefill_seconds;
            },
        }
        if (self.cancel_at) |at| if (self.count >= at and self.reason == null) self.engine.?.cancel(id);
        self.changed.broadcast(self.io);
    }

    fn wait(self: *Heard) api.Reason {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        while (self.reason == null) self.changed.waitUncancelable(self.io, &self.mutex);
        return self.reason.?;
    }
};

test "a reply through the engine interface is the serial reply's tokens: from a session and from the pool's lone stream, ended by eos (the end token after it), past an eos the request ignores, by max_tokens, by a cancel and by a stop string" {
    const template_file = @import("../../../src/text/template.zig");
    const text_frontend = @import("../../../src/text/frontend.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var m = try fixtures.random_model.testModelRows(gpa, arena, null, 48);
    defer m.deinit();
    var template_diagnostic: template_file.Diagnostic = .{};
    const parsed = try template_file.Template.parse(gpa, "{{ messages[0].content }}", &template_diagnostic);
    var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
    const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
    var frontend: text_frontend.Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
    defer frontend.deinit(gpa);
    var ids: std.ArrayList(u32) = .empty;
    try frontend.tokenizer.encode(arena, "the cat sat on the mat", &ids);
    const prompt = try arena.alloc(i32, ids.items.len);
    for (prompt, ids.items) |*token, id| token.* = @intCast(id);
    const serial = try fixtures.serial.serialReply(gpa, m, prompt, 12, greedy);
    defer gpa.free(serial);

    // `eos` is the checkpoint's, `asked` the request's where it differs (`ignore_eos`); `tokens` are the
    // serial reply's that come, the end token among them.
    const Case = struct { eos: []const u32, asked: ?[]const u32 = null, max_tokens: u32, cancel_at: ?usize = null, stop_at: ?usize = null, tokens: usize, reason: api.Reason };
    // a stop string that is whole once this many tokens have come
    const Stop = struct {
        fn check(ctx: *anyopaque, emitted: []const u32) bool {
            const at: *const usize = @ptrCast(@alignCast(ctx));
            return emitted.len >= at.*;
        }
    };
    const sixth = [1]u32{@intCast(serial[5])};
    for ([_]Case{
        .{ .eos = &sixth, .max_tokens = 12, .tokens = std.mem.indexOfScalar(i32, serial, serial[5]).? + 1, .reason = .stop },
        .{ .eos = &sixth, .asked = &.{}, .max_tokens = 7, .tokens = 7, .reason = .length },
        .{ .eos = &.{}, .max_tokens = 7, .tokens = 7, .reason = .length },
        .{ .eos = &.{}, .max_tokens = 12, .cancel_at = 1, .tokens = 0, .reason = .cancelled },
        .{ .eos = &.{}, .max_tokens = 12, .stop_at = 3, .tokens = 3, .reason = .stop },
    }) |case| {
        errdefer std.debug.print("a reply of {d} tokens that ends as {t}\n", .{ case.tokens, case.reason });
        var stop_at: usize = case.stop_at orelse 0;
        const request: api.Request = .{ .prompt = ids.items, .max_tokens = case.max_tokens, .eos = case.asked orelse case.eos, .stop = if (case.stop_at != null) .{ .ctx = &stop_at, .check = Stop.check } else null };

        const alone: *session.Session = try .create(gpa, testing.io, m, &frontend.tokenizer, case.eos, .{ .capacity = 256, .prompt_rows = 32 });
        defer alone.destroy();
        const hall = try hall_file.Hall.create(gpa, testing.io, m, &frontend.tokenizer, case.eos, .{
            .slots = 2,
            .session = .{ .capacity = 256, .prompt_rows = 32 },
            .pool = extents.lowerBound(256, 2) * extents.positions,
            .chunk = 8,
        });
        defer hall.destroy();
        var over_session: served_file.Served = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = null, .behind = .{ .session = alone, .depth = fixtures.sessions.deepest }, .capacity = 256, .vocab = @intCast(m.shape.vocab), .lanes = 1, .think = null, .device_bytes = &cuda.account.bytes };
        defer over_session.close();
        var over_hall: served_file.Served = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = null, .behind = .{ .hall = hall, .depth = fixtures.sessions.deepest }, .capacity = 256, .vocab = @intCast(m.shape.vocab), .lanes = 2, .think = null, .device_bytes = &cuda.account.bytes };
        defer over_hall.close();

        // what a second turn finds held, from each
        var held: [2]u32 = @splat(0);
        for ([_]*served_file.Served{ &over_session, &over_hall }, 0..) |served, which| {
            errdefer std.debug.print("{s}\n", .{if (which == 0) "over a session" else "over the pool"});
            var heard: Heard = .{ .io = testing.io, .cancel_at = case.cancel_at, .engine = served.engine() };
            try served.engine().submit(7, &request, heard.sink());
            try testing.expectEqual(case.reason, heard.wait());
            // told once the prompt had run, before any token, with nothing held of a first prompt
            try testing.expectEqual(0, heard.prefilled.?);
            try testing.expectEqual(0, heard.before);
            if (case.cancel_at) |at| {
                // a cancel is heard between windows: the tokens before it are the serial reply's, and not all came
                try testing.expect(heard.count >= at and heard.count < case.max_tokens);
                for (heard.tokens[0..heard.count], serial[0..heard.count]) |got, want| try testing.expectEqual(@as(u32, @intCast(want)), got);
                continue;
            }
            try testing.expectEqual(case.tokens, heard.count);
            for (heard.tokens[0..heard.count], serial[0..heard.count]) |got, want| try testing.expectEqual(@as(u32, @intCast(want)), got);
            // what the server's usage and its `/stats` read: the reply's windows and the prompt's time, and
            // nothing live once the reply is finished
            // (a reply a stop string ended has them too, through a session and in the pool)
            try testing.expect(heard.rounds > 0 and heard.prefill_seconds.? > 0);
            var status: api.Status = .{};
            var streams: [4]u32 = undefined;
            served.engine().status(&status, &streams);
            try testing.expect(status.decode_tokens_per_second > 0 and status.prefill_tokens_per_second > 0);
            const memory = served.engine().memory(false).?;
            try testing.expect(memory.active > 0 and memory.peak >= memory.active);

            // The next turn, the prompt and its reply with two tokens more: the server is told what was held
            // before the turn's first token, and the tokens are the serial reply's still.
            var next: std.ArrayList(u32) = .empty;
            try next.appendSlice(arena, ids.items);
            for (serial[0 .. case.tokens + 2]) |token| try next.append(arena, @intCast(token));
            const turn: api.Request = .{ .prompt = next.items, .max_tokens = 3, .eos = case.eos };
            var again: Heard = .{ .io = testing.io };
            try served.engine().submit(8, &turn, again.sink());
            _ = again.wait();
            try testing.expectEqual(0, again.before);
            held[which] = again.prefilled.?;
            if (case.eos.len == 0 or case.asked != null) for (again.tokens[0..again.count], serial[case.tokens + 2 ..][0..again.count]) |got, want| try testing.expectEqual(@as(u32, @intCast(want)), got);
        }
        // (after a stop string the session holds its window and the pool what it gave)
        if (case.cancel_at == null and case.stop_at == null) {
            try testing.expect(held[0] >= ids.items.len);
            try testing.expectEqual(held[0], held[1]);
        }
    }
}

/// `got` is `want` to the bit: the token, its log probability and its best.
fn expectRow(want: api.LogprobRow, got: api.LogprobRow) !void {
    try testing.expectEqual(want.token, got.token);
    try testing.expectEqual(@as(u32, @bitCast(want.logprob)), @as(u32, @bitCast(got.logprob)));
    try testing.expectEqual(want.count, got.count);
    try testing.expectEqualSlices(u32, want.ids[0..want.count], got.ids[0..got.count]);
    for (want.logprobs[0..want.count], got.logprobs[0..got.count]) |a, b| try testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
}

test "logprobs: a token's row is the same bits from a serial step, a verify window, the pool's lone stream, a round beside another stream and a second turn's prompt, the end token's included" {
    const template_file = @import("../../../src/text/template.zig");
    const text_frontend = @import("../../../src/text/frontend.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var m = try fixtures.random_model.testModelRows(gpa, arena, null, 48);
    defer m.deinit();
    var template_diagnostic: template_file.Diagnostic = .{};
    const parsed = try template_file.Template.parse(gpa, "{{ messages[0].content }}", &template_diagnostic);
    var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
    const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
    var frontend: text_frontend.Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
    defer frontend.deinit(gpa);
    var ids: std.ArrayList(u32) = .empty;
    try frontend.tokenizer.encode(arena, "the cat sat on the mat", &ids);
    const prompt = try arena.alloc(i32, ids.items.len);
    for (prompt, ids.items) |*token, id| token.* = @intCast(id);
    const serial = try fixtures.serial.serialReply(gpa, m, prompt, 12, greedy);
    defer gpa.free(serial);
    // the reply ends at its tenth token's first place, which is given its row too
    const eos = [1]u32{@intCast(serial[9])};
    const length = std.mem.indexOfScalar(i32, serial, serial[9]).? + 1;
    const k = 5;
    const request: api.Request = .{ .prompt = ids.items, .max_tokens = 12, .eos = &eos, .logprobs = k };

    const serially: *session.Session = try .create(gpa, testing.io, m, &frontend.tokenizer, &eos, .{ .capacity = 256, .prompt_rows = 32 });
    defer serially.destroy();
    const drafted: *session.Session = try .create(gpa, testing.io, m, &frontend.tokenizer, &eos, .{ .capacity = 256, .prompt_rows = 32 });
    defer drafted.destroy();
    const hall = try hall_file.Hall.create(gpa, testing.io, m, &frontend.tokenizer, &eos, .{
        .slots = 2,
        .session = .{ .capacity = 256, .prompt_rows = 32 },
        .pool = extents.lowerBound(256, 2) * extents.positions,
        .chunk = 8,
    });
    defer hall.destroy();
    const vocab: u32 = @intCast(m.shape.vocab);
    var over_serial: served_file.Served = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = null, .behind = .{ .session = serially, .depth = 0 }, .capacity = 256, .vocab = vocab, .lanes = 1, .think = null, .device_bytes = &cuda.account.bytes, .logprobs = true };
    defer over_serial.close();
    var over_drafted: served_file.Served = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = null, .behind = .{ .session = drafted, .depth = fixtures.sessions.deepest }, .capacity = 256, .vocab = vocab, .lanes = 1, .think = null, .device_bytes = &cuda.account.bytes, .logprobs = true };
    defer over_drafted.close();
    var over_hall: served_file.Served = .{ .gpa = gpa, .io = testing.io, .frontend = &frontend, .compiler = null, .behind = .{ .hall = hall, .depth = fixtures.sessions.deepest }, .capacity = 256, .vocab = vocab, .lanes = 2, .think = null, .device_bytes = &cuda.account.bytes, .logprobs = true };
    defer over_hall.close();

    // serial steps: the rows every other form is held to, a row a token and the end's after them
    var want: Heard = .{ .io = testing.io };
    try over_serial.engine().submit(1, &request, want.sink());
    try testing.expectEqual(.stop, want.wait());
    try testing.expect(!want.out_of_step);
    try testing.expectEqual(length, want.count);
    try testing.expectEqual(length, want.row_count);
    for (want.tokens[0..want.count], serial[0..want.count], want.rows[0..want.count]) |got, token, row| {
        try testing.expectEqual(@as(u32, @intCast(token)), got);
        try testing.expectEqual(got, row.token);
        try testing.expectEqual(k, row.count);
        // the greedy token is its row's best, its log probability below zero
        try testing.expectEqual(got, row.ids[0]);
        try testing.expect(row.logprob < 0 and row.logprob == row.logprobs[0]);
    }

    const Form = struct { served: *served_file.Served, name: []const u8 };
    for ([_]Form{ .{ .served = &over_drafted, .name = "verify windows" }, .{ .served = &over_hall, .name = "the pool's lone stream" } }, 2..) |form, id| {
        errdefer std.debug.print("{s}\n", .{form.name});
        var heard: Heard = .{ .io = testing.io };
        try form.served.engine().submit(id, &request, heard.sink());
        try testing.expectEqual(.stop, heard.wait());
        try testing.expect(!heard.out_of_step);
        try testing.expectEqualSlices(u32, want.tokens[0..want.count], heard.tokens[0..heard.count]);
        try testing.expectEqual(want.row_count, heard.row_count);
        for (want.rows[0..want.row_count], heard.rows[0..heard.row_count]) |a, b| try expectRow(a, b);
    }

    // two at once in the pool: the first moves from the lone stream to the rounds the second shares
    var pair: [2]Heard = .{ .{ .io = testing.io }, .{ .io = testing.io } };
    for (&pair, 4..) |*heard, id| try over_hall.engine().submit(id, &request, heard.sink());
    for (&pair) |*heard| {
        errdefer std.debug.print("a round of two\n", .{});
        try testing.expectEqual(.stop, heard.wait());
        try testing.expect(!heard.out_of_step);
        try testing.expectEqualSlices(u32, want.tokens[0..want.count], heard.tokens[0..heard.count]);
        try testing.expectEqual(want.row_count, heard.row_count);
        for (want.rows[0..want.row_count], heard.rows[0..heard.row_count]) |a, b| try expectRow(a, b);
    }

    // the pool was opened for them
    try testing.expect(hall.core.engine != null);

    // a second turn whose prompt holds the reply's first tokens: its first row is a prompt window's last
    var next: std.ArrayList(u32) = .empty;
    try next.appendSlice(arena, ids.items);
    for (serial[0..4]) |token| try next.append(arena, @intCast(token));
    const turn: api.Request = .{ .prompt = next.items, .max_tokens = 2, .eos = &eos, .logprobs = k };
    for ([_]*served_file.Served{ &over_serial, &over_drafted, &over_hall }, 6..) |served, id| {
        var heard: Heard = .{ .io = testing.io };
        try served.engine().submit(id, &turn, heard.sink());
        _ = heard.wait();
        try testing.expect(!heard.out_of_step);
        try testing.expectEqual(2, heard.row_count);
        for (want.rows[4..6], heard.rows[0..2]) |a, b| try expectRow(a, b);
    }
}
test "the capacity the server says is the pool's applied states' budget: what the load's fit left, to zero" {
    const gpa = testing.allocator;
    // `serve` fits the budget it was not given after `Hall.create` and writes the pool's options; nothing
    // runs, so the core and the hall stand on the stack and the engine is asked as `/memory` asks it
    const kept: usize = 7 << 30;
    var core: hall_file.Core = .{
        .gpa = gpa,
        .io = testing.io,
        .m = undefined,
        .tokens = undefined,
        .eos = &.{},
        .options = .{ .slots = 2, .pool = 0, .session = .{ .capacity = 256, .prompt_rows = 32 }, .states_bytes = kept },
        .prefix_wait = .{},
        .minimum_state_bytes = 1 << 20,
    };
    var hall: hall_file.Hall = .{ .gpa = gpa, .io = testing.io, .options = core.options, .core = &core, .scheduler = undefined, .think_open = null, .think_close = null };
    var served: served_file.Served = .{ .gpa = gpa, .io = testing.io, .frontend = undefined, .compiler = null, .behind = .{ .hall = &hall, .depth = 0 }, .capacity = 256, .vocab = 256, .lanes = 2, .think = null, .device_bytes = &cuda.account.bytes };
    // a budget given is held whole
    var capacity = served.engine().info().prompt_cache_capacity.?;
    try testing.expectEqual(kept, capacity.budget_bytes);
    try testing.expectEqual(@as(usize, 1 << 20), capacity.minimum_state_bytes);
    // a budget the fit shrank, and none where the pool alone takes what is left: `serve`'s three writes
    for ([_]usize{ 3 << 20, 0 }) |states| {
        hall.options.states_bytes = states;
        core.options.states_bytes = states;
        if (core.session) |held| held.lot.cap = states;
        capacity = served.engine().info().prompt_cache_capacity.?;
        try testing.expectEqual(states, capacity.budget_bytes);
    }
}

test "a served required call and a served document open with the template's lead after thinking, and with themselves where the reply does not think" {
    const json_schema = @import("../../../src/text/json_schema.zig");
    const template_file = @import("../../../src/text/template.zig");
    const text_frontend = @import("../../../src/text/frontend.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var m = try fixtures.random_model.testModelDrafting(gpa, arena, null);
    defer m.deinit();
    // the prompt ends with `<t>`, which stands for `<think>`: the tiny tokenizer has no think block
    const source = "{% for message in messages %}<l>{{ message.role }} {{ message.content }}\n{% endfor %}<l>assistant <t>";
    var template_diagnostic: template_file.Diagnostic = .{};
    const parsed = try template_file.Template.parse(gpa, source, &template_diagnostic);
    var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
    const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
    var frontend: text_frontend.Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
    defer frontend.deinit(gpa);
    const eos = [1]u32{frontend.tokenizer.addedId("<|end|>").?};
    const alone: *session.Session = try .create(gpa, testing.io, m, &frontend.tokenizer, &eos, .{ .window = m.max_rows, .capacity = 256 });
    defer alone.destroy();
    var engine: json_schema.Engine = try .init(gpa, &frontend.tokenizer, m.shape.vocab, &eos);
    defer engine.deinit(gpa);
    var served: served_file.Served = .{
        .gpa = gpa,
        .io = testing.io,
        .frontend = &frontend,
        .compiler = .{ .engine = engine.interface(), .tokenizer = &frontend.tokenizer, .stops = &eos },
        .behind = .{ .session = alone, .depth = 3 },
        .capacity = 256,
        .vocab = @intCast(m.shape.vocab),
        .lanes = 1,
        .think = null,
        .device_bytes = &cuda.account.bytes,
    };
    defer served.close();

    const list = try std.json.parseFromSliceLeaky(std.json.Value, arena, "[{\"role\":\"user\",\"content\":\"the cat sat on the mat\"}]", .{});
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"sub\",\"parameters\":" ++
        "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"],\"additionalProperties\":false}}}]";
    var text_diagnostic: text_frontend.Diagnostic = .{};
    const prompt = try frontend.prompt(arena, .{ .messages = list }, &text_diagnostic);
    const open = frontend.tokenizer.addedId("<t>").?;
    try testing.expectEqual(open, prompt[prompt.len - 1]);

    // The free reply's first token stands for `</think>`: the reasoning is that token, and the gate begins after.
    var free: Heard = .{ .io = testing.io };
    try served.engine().submit(1, &.{ .prompt = prompt, .max_tokens = 4, .eos = &eos }, free.sink());
    _ = free.wait();
    const close = free.tokens[0];
    try testing.expect(close != open and close != eos[0]);

    // The server's gate names the template's opener; of it this engine reads that a call is required, and
    // compiles its own over the offered tools.
    const call: api.Request = .{ .prompt = prompt, .max_tokens = 24, .eos = &eos, .tools_json = tools, .call = .{ .opener = 0 } };
    const object: api.Request = .{ .prompt = prompt, .max_tokens = 24, .eos = &eos, .structure = .{ .kind = .json } };
    const Case = struct { thinking: bool, lead: []const u8, begins: []const u8 };
    for ([_]Case{
        // the lead the frontend read off its template, after the reasoning's close
        .{ .thinking = true, .lead = "\n\n", .begins = "\n\n<tool_call>\n" },
        .{ .thinking = true, .lead = "\n", .begins = "\n<tool_call>\n" },
        // a template that writes nothing there
        .{ .thinking = true, .lead = "", .begins = "<tool_call>\n" },
        // a reply that does not think: the prompt already ends with what the template writes
        .{ .thinking = false, .lead = "\n\n", .begins = "<tool_call>\n" },
    }) |case| {
        errdefer std.debug.print("thinking {}, lead {d} bytes\n", .{ case.thinking, case.lead.len });
        served.think = if (case.thinking) .{ .open = @intCast(open), .close = @intCast(close) } else null;
        alone.think = served.think;
        defer alone.think = null;
        const from: usize = @intFromBool(case.thinking);

        // the reply's tokens, as the next turn's prompt would have to render them
        frontend.call_lead = case.lead;
        var called: Heard = .{ .io = testing.io };
        try served.engine().submit(2, &call, called.sink());
        try testing.expect(called.wait() != .failed);
        if (case.thinking) try testing.expectEqual(close, called.tokens[0]);
        var written: std.ArrayList(u8) = .empty;
        try frontend.tokenizer.decode(arena, called.tokens[from..called.count], .{}, &written);
        try testing.expect(written.items.len >= case.begins.len);
        try testing.expectEqualStrings(case.begins, written.items[0..case.begins.len]);

        // A document opens the same way (spec 035): after the lead the template writes before content where
        // the reply thinks, and at once where it does not.
        frontend.content_lead = case.lead;
        var document: Heard = .{ .io = testing.io };
        try served.engine().submit(3, &object, document.sink());
        try testing.expect(document.wait() != .failed);
        if (case.thinking) try testing.expectEqual(close, document.tokens[0]);
        var decoded: std.ArrayList(u8) = .empty;
        try frontend.tokenizer.decode(arena, document.tokens[from..document.count], .{}, &decoded);
        const opens = try std.mem.concat(arena, u8, &.{ if (case.thinking) case.lead else "", "{" });
        try testing.expect(decoded.items.len >= opens.len);
        try testing.expectEqualStrings(opens, decoded.items[0..opens.len]);
    }
}

test "a request's tokens on their way to the server: nothing of a window after the one a stop string ended at, and the end token after the reply's own, in the rate and read by no stop string" {
    const gpa = testing.allocator;
    var served: served_file.Served = .{ .gpa = gpa, .io = testing.io, .frontend = undefined, .compiler = null, .behind = .{ .depth = 0 }, .capacity = 256, .vocab = 256, .lanes = 1, .think = null, .device_bytes = &cuda.account.bytes };
    const prompt = [_]u32{ 1, 2, 3 };
    // a stop string whole at the reply's second token, and at an end token were it read
    const Stop = struct {
        fn check(_: *anyopaque, emitted: []const u32) bool {
            return emitted.len >= 2 or emitted[emitted.len - 1] == 9;
        }
    };
    var nothing: u8 = 0;
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 8, .eos = &.{9}, .stop = .{ .ctx = &nothing, .check = Stop.check } };

    var heard: Heard = .{ .io = testing.io };
    var live: served_file.Served.Live = .{ .served = &served, .id = 1, .request = &request, .sink = heard.sink() };
    defer live.emitted.deinit(gpa);
    try testing.expect(!live.give(5));
    try testing.expect(live.give(6));
    // the window's rest, and an end the model still names
    try testing.expect(live.give(7));
    live.closing(.stop, 9);
    try testing.expectEqualSlices(u32, &.{ 5, 6 }, heard.tokens[0..heard.count]);
    try testing.expectEqual(.stop, live.reasonOf(.cut));

    var ended: Heard = .{ .io = testing.io };
    var other: served_file.Served.Live = .{ .served = &served, .id = 2, .request = &request, .sink = ended.sink() };
    defer other.emitted.deinit(gpa);
    try testing.expect(!other.give(5));
    other.closing(.stop, 9);
    try testing.expect(!other.stopped);
    try testing.expectEqualSlices(u32, &.{ 5, 9 }, ended.tokens[0..ended.count]);
    // two of the first request and two of the second, the end token among them
    var total: u32 = 0;
    for (served.given) |count| total += count;
    try testing.expectEqual(4, total);
}

test "an image request on FP4-Local (spec 019 WS-04 and WS-05): the host's prepared image through the eye and a session is the forward's reply from the tower's features, drafted and serial, twice; in the pool beside a text reply, the same tokens, the tower run by the seat" {
    // Runs on the whole tier with ZSPARK_FLASH_NEXT_FP4_LOCAL, the checkpoint's directory, and
    // ZSPARK_REFERENCE_VISION, WS-03's exports, for `shapes`' prompt: TensorFold's ids, 64 placeholders.
    if (!@import("tier").whole) return error.SkipZigTest;
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const reference_path = testing.environ.getPosix("ZSPARK_REFERENCE_VISION") orelse return error.SkipZigTest;
    const safetensors = @import("../../../src/weights/safetensors.zig");
    const qwen4_config = @import("../../../src/models/qwen4/config.zig");
    const layout = @import("../../../src/models/qwen4/layout.zig");
    const model = @import("../../../src/models/qwen4/model.zig");
    const forward = @import("../../../src/models/qwen4/forward.zig");
    const compare = @import("../../../src/models/qwen4/compare.zig");
    const tower_file = @import("../../../src/models/qwen4/vision.zig");
    const text_vision = @import("../../../src/text/vision.zig");
    const request_file = @import("../../../src/server/request.zig");
    const Bench = @import("compare_fixture.zig").Bench;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var checkpoint: safetensors.Checkpoint = try .open(gpa, io, .cwd(), path);
    defer checkpoint.close();
    var diagnostic: qwen4_config.Diagnostic = .{};
    const config = try qwen4_config.Config.read(arena, io, dir, &diagnostic);
    try testing.expect(config.servesImages());
    const found = try layout.Layout.init(arena, &config, &checkpoint, &diagnostic);
    const rows = 96;
    var m: model.Model = try .loadWith(gpa, config, found, .{ .layers = config.layers, .max_rows = rows, .mtp = true });
    defer m.deinit();
    var tower = tower_file.Tower.load(gpa, arena, &checkpoint, tower_file.max_patches_default, &diagnostic) catch |err| {
        std.debug.print("vision: {s}\n", .{diagnostic.message});
        return err;
    };
    defer tower.deinit();
    var refused: text_vision.Problem = .{};
    const settings = (try text_vision.Settings.load(arena, io, dir, &refused)).?;
    var eye: served_file.Eye = try .init(gpa, &tower, settings, 2, 512);
    defer eye.deinit();

    // the host's part: the picture's bytes prepared by the offer, the image at its first placeholder
    var references = try std.Io.Dir.cwd().openDir(io, reference_path, .{});
    defer references.close(io);
    const summary = try std.json.parseFromSliceLeaky(std.json.Value, arena, try references.readFileAlloc(io, "shapes.f32.json", arena, .limited(1 << 24)), .{});
    const ids = summary.object.get("ids").?.array.items;
    const prompt = try arena.alloc(i32, ids.len);
    for (prompt, ids) |*token, id| token.* = @intCast(id.integer);
    const prepared = try eye.offer.prepare(arena, @embedFile("fixtures/vision/shapes.png"), @import("../../../src/server/images.zig").image_tokens);
    const at = std.mem.indexOfScalar(i32, prompt, @intCast(settings.image_token)).?;
    try testing.expectEqual(64, prepared.tokens);
    try testing.expectEqual(64, std.mem.count(i32, prompt, &.{@intCast(settings.image_token)}));
    const given = [1]api.Image{.{ .pixels = prepared.pixels, .gh = prepared.gh, .gw = prepared.gw, .tokens = prepared.tokens, .at = @intCast(at) }};

    // the forward's reply from the same features, as WS-03's whole-model test decodes one
    var features: cuda.Buffer = try .init(.bf16, &.{ prepared.tokens, tower_file.out_hidden });
    defer features.deinit();
    try tower.encode(gpa, prepared.pixels, &.{.{ .t = 1, .h = prepared.gh, .w = prepared.gw }}, features.view);
    var said: request_file.Diagnostic = .{};
    const image_prompt = try eye.prompt(gpa, arena, prompt, &given, &said);
    const reference: compare.ReferenceImage = .{ .features = features, .rows = @constCast(image_prompt.image.rows), .positions = @constCast(image_prompt.positions), .delta = image_prompt.delta };
    var b: forward.Buffers = try .init(gpa, m, rows, Bench.capacity);
    defer b.deinit();
    var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
    defer sampler.deinit(gpa);
    var wide: cuda.Buffer = try .init(.f32, &.{ 1, m.shape.vocab });
    defer wide.deinit();
    const count = 24;
    const bench: Bench = .{ .m = m, .b = &b, .sampler = &sampler, .logits = try arena.alloc(u16, m.shape.vocab), .wide = &wide, .gpa = gpa, .image = &reference };
    const want = try arena.alloc(i32, count);
    _ = try bench.drafted(prompt, want, greedy);

    const bytes = try dir.readFileAlloc(io, "tokenizer.json", arena, .limited(1 << 26));
    var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
    var tokens = try tokenizer.Tokenizer.parse(gpa, bytes, &tokenizer_diagnostic);
    defer tokens.deinit(gpa);
    const held: *session.Session = try .create(gpa, io, m, &tokens, &.{}, .{ .capacity = 512, .prompt_rows = 32 });
    defer held.destroy();
    var counting: fixtures.sessions.Counting = .{};
    // drafted, serial, and drafted again after a reply that held the prompt: each from an empty state
    for ([_]usize{ fixtures.sessions.deepest, 0, fixtures.sessions.deepest }, 0..) |depth, turn| {
        errdefer std.debug.print("turn {d}, depth {d}\n", .{ turn, depth });
        var record: [count]i32 = undefined;
        const ip = try eye.prompt(gpa, arena, prompt, &given, &said);
        const done = try held.reply(prompt, greedy, .{ .max_tokens = count, .depth = depth, .record = &record, .image = ip }, counting.sink());
        try testing.expectEqual(0, done.cached);
        try testing.expectEqualSlices(i32, want[0..done.tokens], record[0..done.tokens]);
        std.debug.print("shapes on FP4-Local, depth {d}: {d} tokens, the forward's; {d} of {d} drafts kept\n", .{ depth, done.tokens, done.stats.accepted, done.stats.drafted });
    }

    // WS-05: the text before the image, the session's reply to it, then both in the pool's one round, the
    // image seat's features written by the tower when its chunk reaches them
    const text = prompt[0..at];
    var text_record: [count]i32 = undefined;
    const text_done = try held.reply(text, greedy, .{ .max_tokens = count, .record = &text_record }, counting.sink());
    const hall_engine = @import("../../../src/models/qwen4/pool/hall_engine.zig");
    const Collect = @import("pool/hall_fixture.zig").Collect;
    var engine: hall_engine.Engine = try .open(gpa, m, &.{}, .{
        .slots = 2,
        .session = .{ .capacity = 512, .prompt_rows = 32 },
        .pool = extents.lowerBound(512, 2) * extents.positions,
        .chunk = 32,
        .lazy_graphs = false,
        .sight = &eye.sight,
    });
    defer engine.deinit();
    engine.bindPool();
    var lists: [2]std.ArrayList(i32) = @splat(.empty);
    defer for (&lists) |*list| list.deinit(gpa);
    var ended: [2]bool = @splat(false);
    var collect: Collect = .{ .gpa = gpa, .lists = &lists, .ended = &ended };
    const out: hall_file.Out = .{ .ctx = &collect, .emit = Collect.emit, .end = Collect.end, .fail = Collect.fail };
    const picture = try eye.picture(arena, prompt, &given, &said);
    // the session's encode left its own features there: the seat runs the tower again
    _ = try engine.begin(0, .{ .prompt = text, .settings = greedy, .max_tokens = count, .depth = engine.depth, .opening = engine.depth }, 0);
    _ = try engine.begin(1, .{ .prompt = prompt, .settings = greedy, .max_tokens = count, .depth = engine.depth, .opening = engine.depth, .image = picture }, 0);
    var spins: usize = 0;
    while (!(ended[0] and ended[1]) and spins < 400) : (spins += 1) switch (engine.round(out)) {
        .ran => engine.publishAhead(out),
        .short, .failed => return error.Failed,
    };
    // the engine was given no end tokens: the session's replies are where its own stop
    try testing.expect(lists[0].items.len >= text_done.tokens);
    try testing.expectEqualSlices(i32, text_record[0..text_done.tokens], lists[0].items[0..text_done.tokens]);
    try testing.expectEqualSlices(i32, want, lists[1].items);
    for (0..2) |slot| _ = engine.retire(slot, .ended, count);
    std.debug.print("shapes on FP4-Local in the pool beside a text reply: {d} tokens, the forward's\n", .{lists[1].items.len});
}

/// A checkpoint folder that is a `generation_config.json` alone, so `samplingDefaults` and
/// `Startup.run`'s instruct resolution read it as a checkpoint's.
fn generationDir(arena: Allocator, io: std.Io, tmp: *std.testing.TmpDir, name: []const u8, config: []const u8) ![]const u8 {
    try tmp.dir.createDirPath(io, name);
    try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(arena, "{s}/generation_config.json", .{name}), .data = config });
    var opened = try tmp.dir.openDir(io, name, .{});
    defer opened.close(io);
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    return try arena.dupe(u8, buffer[0..try opened.realPath(io, &buffer)]);
}

/// The host's `model_text.Text` over the request's own bytes: `render` is the message's content verbatim,
/// `encode` a byte a token — the same scripted text `served_fixture.zig`'s `Bytes` gives the host's tests, so
/// a prompt's ids are its bytes and the serial reply can be computed on them. The engine behind it is the
/// random model's real session, so the reply's tokens come through the real sampler.
/// Replies render each full-width token as eight hex digits; HTTP token_ids carry the exact comparison.
const ByteText = struct {
    vocab: u32,
    fn text(self: *ByteText) host.model_text.Text {
        return .{ .ctx = self, .vtable = &.{
            .encode = encode,
            .decode = decode,
            .token_id = tokenId,
            .token_string = tokenString,
            .vocab_size = vocabSize,
            .eos_ids = eosIds,
            .render = render,
            .template_source = templateSource,
        } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) host.model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len * 8);
        for (ids, 0..) |id, i| _ = std.fmt.bufPrint(decoded[i * 8 ..][0..8], "{x:0>8}", .{id}) catch unreachable;
        return decoded;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{x:0>8}", .{id});
    }
    fn vocabSize(ctx: *anyopaque) u32 {
        const self: *ByteText = @ptrCast(@alignCast(ctx));
        return self.vocab;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: Allocator, messages: host.json.Value, _: host.model_text.RenderOptions, _: *[]const u8) host.model_text.Error!host.model_text.Rendered {
        var out: std.ArrayList(u8) = .empty;
        if (messages == .array) for (messages.array) |message| {
            const content = message.get("content") orelse continue;
            if (content == .string) try out.appendSlice(a, content.string);
        };
        return .{ .text = try out.toOwnedSlice(a) };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// `command.Startup.run`'s server over the model's engine on a thread: `begin` waits for its `PORT`, `stop`
/// raises the SIGTERM `zspark serve` takes and joins; an early return still stops the server it started, and
/// the signal handlers the run installed are put back when the file's tests end.
const StartupRun = struct {
    code: u8 = 255,
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    fn main(self: *StartupRun, startup: *command.Startup, gpa: Allocator, io: std.Io, args: host.cli.Args, setup: host.serve.Setup) void {
        self.code = startup.run(gpa, io, args, setup);
        self.done.store(true, .release);
    }

    /// The port once the server listens.
    fn begin(self: *StartupRun, io: std.Io, startup: *command.Startup, gpa: Allocator, args: host.cli.Args, setup: host.serve.Setup) !u16 {
        startup_port.store(0, .release);
        var s = setup;
        s.on_listen = startupListened;
        self.thread = try std.Thread.spawn(.{}, StartupRun.main, .{ self, startup, gpa, io, args, s });
        while (startup_port.load(.acquire) == 0) {
            if (self.done.load(.acquire)) return error.NoPort;
            std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
        }
        return startup_port.load(.acquire);
    }

    /// SIGTERM as a client end takes it, then the thread's join; the exit code `run` returned.
    fn stop(self: *StartupRun) !u8 {
        if (self.thread) |t| {
            self.thread = null;
            if (!self.done.load(.acquire)) try posix.raise(.TERM);
            t.join();
        }
        return self.code;
    }
};

/// The `PORT` a `serve` takes, for `StartupRun`'s `on_listen`.
var startup_port: std.atomic.Value(u16) = .init(0);
fn startupListened(port: u16) void {
    startup_port.store(port, .release);
}

/// `body` POSTed to `/v1/chat/completions` on 127.0.0.1:`port`, the reply to the connection's end; a reply
/// that never ends fails in ten seconds (`served_fixture.zig`'s `sent`/`heard`, written where this file has
/// no socket of its own).
fn chatPost(arena: Allocator, port: u16, body: []const u8) ![]const u8 {
    const asked = try std.fmt.allocPrint(arena, "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}", .{ body.len, body });
    const made = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(made) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(made);
    defer _ = posix.system.close(fd);
    const timeout: posix.timeval = .{ .sec = 10, .usec = 0 };
    try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.RCVTIMEO, std.mem.asBytes(&timeout));
    const address: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    if (posix.errno(posix.system.connect(fd, @ptrCast(&address), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.ConnectFailed;
    var at: usize = 0;
    while (at < asked.len) {
        const wrote = posix.system.write(fd, asked[at..].ptr, asked.len - at);
        if (posix.errno(wrote) != .SUCCESS) return error.WriteFailed;
        at += @intCast(wrote);
    }
    var reply: std.ArrayList(u8) = .empty;
    var buffer: [4096]u8 = undefined;
    while (true) {
        const got = try posix.read(fd, &buffer);
        if (got == 0) return reply.items;
        try reply.appendSlice(arena, buffer[0..got]);
    }
}

/// The reply's `tensorfold.sampling` label ("exact" over a sampling set, "greedy" without) and its
/// `token_ids` (the same slice `token_sha` hashes), checked against the serial reply's `want`.
fn replyIs(arena: Allocator, got: []const u8, label: []const u8, want: []const i32) !void {
    const cut = std.mem.lastIndexOf(u8, got, "\r\n\r\n") orelse return error.NoReply;
    const parsed = try host.json.parse(arena, got[cut + 4 ..]);
    try testing.expect(parsed == .ok);
    const runtime = parsed.ok.get("tensorfold") orelse return error.NoTensorfold;
    try testing.expectEqualStrings(label, (runtime.get("sampling") orelse return error.NoSampling).string);
    const ids = (runtime.get("token_ids") orelse return error.NoTokenIds).array;
    try testing.expectEqual(want.len, ids.len);
    for (want, ids) |t, v| try testing.expectEqual(@as(u32, @intCast(t)), try std.fmt.parseInt(u32, v.int, 10));
}

test "--instruct-sampling and generation_config decide a chat's sampling over HTTP: qwen3_5's card for thinking-off chats, the request's fields over it, none, do_sample false (spec 168 WS03)" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const template_file = @import("../../../src/text/template.zig");
    const text_frontend = @import("../../../src/text/frontend.zig");
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // The random model behind its own session: the real sampler, whose draws the serial reply shares.
    var m = try fixtures.random_model.testModelRows(gpa, arena, null, 48);
    defer m.deinit();
    var template_diagnostic: template_file.Diagnostic = .{};
    const parsed_template = try template_file.Template.parse(gpa, "{{ messages[0].content }}", &template_diagnostic);
    var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
    const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
    var frontend: text_frontend.Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed_template, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
    defer frontend.deinit(gpa);

    const alone: *session.Session = try .create(gpa, io, m, &frontend.tokenizer, &.{}, .{ .capacity = 256, .prompt_rows = 32 });
    defer alone.destroy();
    var served: served_file.Served = .{ .gpa = gpa, .io = io, .frontend = &frontend, .compiler = null, .behind = .{ .session = alone, .depth = fixtures.sessions.deepest }, .capacity = 256, .vocab = @intCast(m.shape.vocab), .lanes = 1, .think = null, .device_bytes = &cuda.account.bytes };
    defer served.close();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const sampled_dir = try generationDir(arena, io, &tmp, "sampled", "{\"do_sample\":true,\"temperature\":0.5,\"top_p\":0.6,\"top_k\":77}");
    const greedy_dir = try generationDir(arena, io, &tmp, "greedy", "{\"do_sample\":false}");

    // The request's fixed seed makes the serial reply decide: the checkpoint's set, qwen3_5's card
    // (0.7/0.8/20; this engine serves no penalties), the request's temperature/top_k over it, and greedy.
    const prompt = try arena.alloc(i32, "the cat sat on the mat".len);
    for ("the cat sat on the mat", prompt) |byte, *id| id.* = byte;
    const want_defaults = try fixtures.serial.serialReply(gpa, m, prompt, 8, .{ .seed = 424242, .temperature = 0.5, .top_k = 77, .top_p = 0.6, .min_p = 0 });
    defer gpa.free(want_defaults);
    const want_card = try fixtures.serial.serialReply(gpa, m, prompt, 8, .{ .seed = 424242, .temperature = 0.7, .top_k = 20, .top_p = 0.8, .min_p = 0 });
    defer gpa.free(want_card);
    const want_over = try fixtures.serial.serialReply(gpa, m, prompt, 8, .{ .seed = 424242, .temperature = 0.8, .top_k = 40, .top_p = 0.8, .min_p = 0 });
    defer gpa.free(want_over);
    const want_greedy = try fixtures.serial.serialReply(gpa, m, prompt, 8, .{ .seed = 424242, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 });
    defer gpa.free(want_greedy);
    // The competitors differ: a path that dropped the instruct set or the request's fields could not answer
    // any of these correctly.
    try testing.expect(!std.mem.eql(i32, want_defaults, want_card));
    try testing.expect(!std.mem.eql(i32, want_card, want_over));
    try testing.expect(!std.mem.eql(i32, want_card, want_greedy));

    var environ: std.process.Environ.Map = .init(gpa);
    defer environ.deinit();
    var bytes: ByteText = .{ .vocab = @intCast(m.shape.vocab) };

    // `serve.run` installs the process's stop handlers; the file's other tests get theirs back.
    var old_term: posix.Sigaction = undefined;
    var old_int: posix.Sigaction = undefined;
    var old_pipe: posix.Sigaction = undefined;
    posix.sigaction(.TERM, null, &old_term);
    posix.sigaction(.INT, null, &old_int);
    posix.sigaction(.PIPE, null, &old_pipe);
    defer {
        posix.sigaction(.TERM, &old_term, null);
        posix.sigaction(.INT, &old_int, null);
        posix.sigaction(.PIPE, &old_pipe, null);
    }

    const Round = struct { flags: []const []const u8, dir: []const u8, asks: []const struct { []const u8, []const u8, []const i32 } };
    // `msg` is the message the scripted text renders; each ask is {extra JSON fields, label, expected ids}.
    const msg = ",\"messages\":[{\"role\":\"user\",\"content\":\"the cat sat on the mat\"}]}";
    const head = "{\"seed\":424242,\"return_token_ids\":true,\"max_tokens\":8";
    for ([_]Round{
        // --instruct-sampling card: plain and explicitly thinking chats draw the checkpoint's set;
        // chat_template_kwargs.enable_thinking:false takes the card, below request sampling fields.
        .{ .flags = &.{ "--instruct-sampling", "card" }, .dir = sampled_dir, .asks = &.{
            .{ "", "exact", want_defaults },
            .{ ",\"chat_template_kwargs\":{\"enable_thinking\":true}", "exact", want_defaults },
            .{ ",\"chat_template_kwargs\":{\"enable_thinking\":false}", "exact", want_card },
            .{ ",\"chat_template_kwargs\":{\"enable_thinking\":false},\"temperature\":0.8,\"top_k\":40", "exact", want_over },
        } },
        // --instruct-sampling none: thinking off takes the checkpoint's set, no card.
        .{ .flags = &.{ "--instruct-sampling", "none" }, .dir = sampled_dir, .asks = &.{
            .{ ",\"chat_template_kwargs\":{\"enable_thinking\":false}", "exact", want_defaults },
        } },
        // do_sample false: every chat draws greedily, card or not.
        .{ .flags = &.{ "--instruct-sampling", "card" }, .dir = greedy_dir, .asks = &.{
            .{ "", "greedy", want_greedy },
            .{ ",\"chat_template_kwargs\":{\"enable_thinking\":false}", "greedy", want_greedy },
        } },
    }) |round| {
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.append(arena, round.dir);
        try argv.appendSlice(arena, &.{ "--host", "127.0.0.1", "--port", "0", "--prompt-seed" });
        try argv.appendSlice(arena, round.flags);
        var usage: host.cli.Usage = .{};
        const args = try host.cli.parse(arena, argv.items, &usage);
        var startup = (try command.Startup.open(arena, &environ)).?;
        defer startup.deinit();
        var run: StartupRun = .{};
        const port = try run.begin(io, &startup, gpa, args, .{
            .identity = command.identity(),
            .engine = served.engine(),
            .text = bytes.text(),
            .served = "random",
            .model_dir = round.dir,
            .model_type = "qwen3_5",
            .sampling = try command.samplingDefaults(arena, io, round.dir, args),
            .environ = &environ,
        });
        defer _ = run.stop() catch {};
        for (round.asks) |ask| {
            errdefer std.debug.print("{s} {s}: the reply's tokens were not the serial reply's\n", .{ round.flags[1], ask[0] });
            try replyIs(arena, try chatPost(arena, port, try std.fmt.allocPrint(arena, "{s}{s}{s}", .{ head, ask[0], msg })), ask[1], ask[2]);
        }
        try testing.expectEqual(@as(u8, 0), try run.stop());
    }
}
