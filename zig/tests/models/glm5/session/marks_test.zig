//! `marks.zig`'s tests, on the host: where GLM's session keeps a prompt's states and a reply's, over the
//! client set's conversations (`tools/reference/client_bench.json`'s `prompt` and `turns`) laid out as GLM's
//! template lays them, and on FP4-Local's own template (`ZSPARK_GLM_FP4_LOCAL`). A session of no device's
//! (`Sim`) resumes as `Session.reply` does and takes each planned mark where the window that holds it runs.
//! That a mark taken inside a window is the state there: `snapshot_test.zig`; the session's: `session_test.zig`.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const shelf = @import("../../../../src/decode/shelf.zig");
const book_file = @import("../../../../src/decode/book.zig");
const frontend_file = @import("../../../../src/text/frontend.zig");
const marking = @import("../../../../src/models/glm5/session/marks.zig");
const snapshot = @import("../../../../src/models/glm5/session/snapshot.zig");

/// A mark of no device's.
const Mark = struct {
    pos: usize,
    reply: bool = false,

    pub fn deinit(self: *Mark) void {
        self.* = undefined;
    }

    pub fn release(self: *Mark, gpa: Allocator, spare: *shelf.Spares(Mark)) void {
        spare.give(gpa, self.*);
    }
};

/// FP4-Local's windows (`Decoder.promptRows`): `--rows` 4096, and the last as wide as `chosen`'s 16.
const Rows = struct {
    wide: usize = 4096,

    pub fn promptRows(self: Rows, pos: usize, len: usize, chosen: usize) usize {
        const left = len - pos;
        const last = @min(self.wide, chosen);
        return if (left <= last) left else @min(self.wide, left - last);
    }
};
const room = 16;

/// What `marks.zig` reads of a session, and `Session.reply`'s resume.
const Sim = struct {
    gpa: Allocator,
    options: struct { snapshots: usize = snapshot.held_default, stride: usize = snapshot.stride_default } = .{},
    state: struct { pos: usize = 0 } = .{},
    marks: shelf.Shelf(Mark, Mark),
    book: book_file.Book,
    lead: Lead = .{},
    known: std.ArrayList(i32) = .empty,

    const Lead = struct {
        pub fn tell(_: *Lead, _: u32, _: []const u32, _: []const i32) !void {}
    };

    fn init(gpa: Allocator) !Sim {
        return .{ .gpa = gpa, .marks = .{ .gpa = gpa }, .book = try .init(gpa, marking.bookOptions(@as(@FieldType(Sim, "options"), .{}))) };
    }

    fn deinit(self: *Sim) void {
        self.book.deinit(self.gpa);
        self.marks.deinit();
        self.known.deinit(self.gpa);
    }

    pub fn leads(_: *const Sim) bool {
        return false;
    }

    pub fn takeMark(self: *Sim) !Mark {
        return .{ .pos = self.state.pos };
    }

    /// No drafter's context to bring to a mark.
    pub fn flashTo(_: *Sim, _: usize, _: ?usize) !?usize {
        return null;
    }

    pub fn flashBack(_: *Sim, _: ?usize) !void {}

    const Turn = struct { cached: usize, planned: []usize };

    /// A reply of `made` tokens to `prompt`: the state resumes as `Session.reply`'s (`cachedOf`,
    /// `marks.reached`), the prompt's marks are planned and taken, the reply keeps one each stride, a token
    /// a round, and at its end. The caller frees `planned`.
    fn turn(self: *Sim, prompt: []const i32, made: usize, starts: []const usize) !Turn {
        const known = self.known.items;
        var same: usize = 0;
        while (same < known.len and same < prompt.len and known[same] == prompt[same]) same += 1;
        var cached: usize = 0;
        if (known.len > 0 and known.len < prompt.len and same == known.len) {
            cached = known.len;
        } else {
            if (try marking.reached(self, prompt.len, same)) |pos| {
                self.marks.dropAbove(pos);
                cached = pos;
            }
        }
        if (cached == 0) self.marks.clear();
        self.state.pos = cached;
        const planned = try marking.plan(self, Rows{}, room, .{ .pos = cached, .len = prompt.len, .shared = @max(same, cached), .starts = starts });
        errdefer self.gpa.free(planned);
        // each where the window that holds it runs (`takeIn`)
        for (planned) |at| try self.marks.held.append(self.gpa, .{ .pos = at });
        self.state.pos = prompt.len;
        self.book.gridFrom(self.state.pos);
        for (1..made) |_| {
            self.state.pos += 1;
            while (self.book.onGrid(self.state.pos)) try marking.keep(self, true);
        }
        try marking.keep(self, true);
        self.known.clearRetainingCapacity();
        try self.known.appendSlice(self.gpa, prompt);
        for (1..made) |i| try self.known.append(self.gpa, @intCast(900_000 + i + prompt.len));
        try self.expectBudgets();
        return .{ .cached = cached, .planned = planned };
    }

    fn held(self: *const Sim, out: []usize) []usize {
        return self.marks.positions(out);
    }

    fn expectBudgets(self: *const Sim) !void {
        var prompts: usize = 0;
        var replies: usize = 0;
        for (self.marks.held.items, 0..) |mark, i| {
            if (mark.reply) replies += 1 else prompts += 1;
            if (i > 0) try testing.expect(self.marks.held.items[i - 1].pos < mark.pos);
        }
        try testing.expect(prompts <= self.options.snapshots and replies <= snapshot.reply_extra);
    }
};

// GLM's layout: `[gMASK]<sop><|system|>` and the reasoning's effort, then each message's opener
const user = 154827;
const assistant = 154828;
const think = 154841;
const header = [_]i32{ 154822, 154824, 154823, 1, 2, 3 };

/// `header`, then each of `messages` opened by its role's token, then the reply's header; and where each
/// message starts.
fn render(arena: Allocator, messages: []const []const i32) !struct { ids: []i32, starts: []usize } {
    var ids: std.ArrayList(i32) = .empty;
    var starts: std.ArrayList(usize) = .empty;
    try ids.appendSlice(arena, &header);
    for (messages, 0..) |body, i| {
        try starts.append(arena, ids.items.len);
        try ids.append(arena, if (i % 2 == 0) user else assistant);
        try ids.appendSlice(arena, body);
    }
    try starts.append(arena, ids.items.len);
    try ids.appendSlice(arena, &.{ assistant, think });
    return .{ .ids = ids.items, .starts = starts.items };
}

/// `count` tokens of text, one of a family by `seed`.
fn text(arena: Allocator, count: usize, seed: usize) ![]i32 {
    const out = try arena.alloc(i32, count);
    for (out, 0..) |*id, i| id.* = @intCast(10 + (i * 7919 + seed * 104729) % 150000);
    return out;
}

test "a cold prompt keeps its messages' starts and two before its end, inside the windows that run, and its reply a stride and its end" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var sim: Sim = try .init(gpa);
    defer sim.deinit();
    // turns-500's first: 484 tokens, its reply 320
    const first = try render(arena, &.{try text(arena, 475, 1)});
    try testing.expectEqual(484, first.ids.len);
    try testing.expectEqualSlices(usize, &.{ 6, 482 }, first.starts);
    const one = try sim.turn(first.ids, 320, first.starts);
    defer gpa.free(one.planned);
    try testing.expectEqual(0, one.cached);
    // the windows [0, 468) and [468, 484): the user's start inside the first, the reply's header inside the last
    try testing.expectEqual(468, (Rows{}).promptRows(0, 484, room));
    // and the last window's start, the committed state there (GLM's own); the user's start, inside a window
    // that runs KDA chunked, moves down to its chunk's start, the prompt's: none (spec 116 WS-06)
    try testing.expectEqualSlices(usize, &.{ 468, 482 }, one.planned);
    var buffer: [16]usize = undefined;
    // the reply's stride 256 tokens on, and its end, where the next prompt's message starts
    try testing.expectEqualSlices(usize, &.{ 468, 482, 740, 803 }, sim.held(&buffer));
}

test "the client set's conversations: a turn goes on from the caches, an edit from its message's start, a new conversation from the user's" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var sim: Sim = try .init(gpa);
    defer sim.deinit();
    var buffer: [16]usize = undefined;
    const excerpt = try text(arena, 475, 1);
    const first = try render(arena, &.{excerpt});
    gpa.free((try sim.turn(first.ids, 320, first.starts)).planned);

    // the second turn: the first reply as the client sends it back (the reply's own tokens), then a question
    const reply = try arena.dupe(i32, sim.known.items[first.ids.len - 2 ..]);
    const second_ask = try text(arena, 22, 2);
    const second = try render(arena, &.{ excerpt, reply[1..], second_ask });
    try testing.expectEqual(828, second.ids.len);
    const two = try sim.turn(second.ids, 320, second.starts);
    defer gpa.free(two.planned);
    // from the caches, 803; the first reply's end joins the prompt's marks, where the second message starts
    try testing.expectEqual(803, two.cached);
    try testing.expectEqualSlices(usize, &.{ 812, 826 }, two.planned);
    try testing.expectEqualSlices(usize, &.{ 468, 482, 803, 812, 826, 1084, 1147 }, sim.held(&buffer));
    for (sim.marks.held.items) |mark| try testing.expectEqual(mark.pos >= 1084, mark.reply);

    // the third turn, its reply 16
    const reply_two = try arena.dupe(i32, sim.known.items[second.ids.len - 2 ..]);
    const third = try render(arena, &.{ excerpt, reply[1..], second_ask, reply_two[1..], try text(arena, 14, 3) });
    const three = try sim.turn(third.ids, 16, third.starts);
    defer gpa.free(three.planned);
    try testing.expectEqual(1147, three.cached);
    // 17 rows, a window of 1 and the last 16: its start too
    try testing.expectEqualSlices(usize, &.{ third.ids.len - 16, third.ids.len - 2 }, three.planned);

    // the third message edited: from the third prompt's last window's start, a row past the message's (the
    // second reply's end): one window of 16 rows, where from the start it ran 1 and 16, all bf16 either way
    var edited_ask = try arena.dupe(i32, try text(arena, 14, 3));
    edited_ask[8] +%= 1;
    const third_edited = try render(arena, &.{ excerpt, reply[1..], second_ask, reply_two[1..], edited_ask });
    const four = try sim.turn(third_edited.ids, 16, third_edited.starts);
    defer gpa.free(four.planned);
    try testing.expectEqual(1148, four.cached);

    // the second message edited: from the second prompt's last window's start, a row before the edit (before
    // spec 116 WS-06, 803, its start: the user's start at 6, moved into the prompt's chunk, freed a mark's
    // room; before spec 112 WS-06, 742: the first reply's stride below it, its window 70 rows)
    var second_edited_ask = try arena.dupe(i32, second_ask);
    second_edited_ask[9] +%= 1;
    const second_edited = try render(arena, &.{ excerpt, reply[1..], second_edited_ask });
    const five = try sim.turn(second_edited.ids, 16, second_edited.starts);
    defer gpa.free(five.planned);
    try testing.expectEqual(812, five.cached);

    // a new conversation over the same lead: cold, the user's start at 6 being inside the first prompt's first
    // chunk (spec 116 WS-06; before it, from 6, the first prompt's mark there)
    var other = try arena.dupe(i32, try text(arena, 476, 4));
    @memcpy(other[0..20], excerpt[0..20]);
    const cold = try render(arena, &.{other});
    const six = try sim.turn(cold.ids, 320, cold.starts);
    defer gpa.free(six.planned);
    try testing.expectEqual(0, six.cached);
    try testing.expectEqualSlices(usize, &.{ cold.ids.len - 16, cold.ids.len - 2 }, six.planned);
}

test "a long prompt keeps a mark where a window that crossed a stride ends, and eight at most" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var sim: Sim = try .init(gpa);
    defer sim.deinit();
    // prompt-16000: one message over four windows of 4096
    const long = try render(arena, &.{try text(arena, 16000, 5)});
    const one = try sim.turn(long.ids, 16, long.starts);
    defer gpa.free(one.planned);
    // the first window holds the user's start; each after it crosses a stride of 512 (`decode/marks.zig`)
    try testing.expectEqualSlices(usize, &.{ 8192, 12288, 15993, long.ids.len - 2 }, one.planned);

    // two hundred short messages: eight of the prompt's, thinned toward the start, the last turn's kept
    var messages: [200][]const i32 = undefined;
    for (&messages, 0..) |*body, i| body.* = try text(arena, 30, 100 + i);
    const many = try render(arena, &messages);
    const two = try sim.turn(many.ids, 16, many.starts);
    defer gpa.free(two.planned);
    var prompts: usize = 0;
    for (sim.marks.held.items) |mark| prompts += @intFromBool(!mark.reply);
    try testing.expectEqual(snapshot.held_default, prompts);
    try testing.expectEqual(many.ids.len - 2, two.planned[two.planned.len - 1]);
}

test "FP4-Local's template: its messages start at `<|user|>` and `<|assistant|>`, and a new conversation goes on from the user's start" {
    const path = testing.environ.getPosix("ZSPARK_GLM_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try std.Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    var diagnostic: frontend_file.Diagnostic = .{};
    var frontend: frontend_file.Frontend = try .load(gpa, testing.io, dir, null, &diagnostic);
    defer frontend.deinit(gpa);
    const ids = struct {
        fn of(f: *const frontend_file.Frontend, name: []const u8) u32 {
            return f.tokenizer.addedId(name).?;
        }
    };
    try testing.expectEqual(2, frontend.openers.len);
    try testing.expect(std.mem.indexOfScalar(u32, frontend.openers, ids.of(&frontend, "<|user|>")) != null);
    try testing.expect(std.mem.indexOfScalar(u32, frontend.openers, ids.of(&frontend, "<|assistant|>")) != null);

    // the client set's `prompt` requests: a lead, an excerpt and a question, one user message
    const lead = "Below is an excerpt of an engineering log. Read it, then answer the question after it.\n\n";
    const question = "\n\nQuestion: in one sentence, what is the excerpt about?";
    var sim: Sim = try .init(gpa);
    defer sim.deinit();
    var first_user: ?usize = null;
    for ([_][]const u8{ "The pump at station four ran hot twice on Tuesday.", "Release 2.3 moved the cache to the second disk." }, 0..) |excerpt, i| {
        var list: std.json.Array = .init(arena);
        var message: std.json.ObjectMap = .empty;
        try message.put(arena, "role", .{ .string = "user" });
        try message.put(arena, "content", .{ .string = try std.mem.concat(arena, u8, &.{ lead, excerpt, question }) });
        try list.append(.{ .object = message });
        const tokens = try frontend.prompt(arena, .{ .messages = .{ .array = list } }, &diagnostic);
        const prompt = try arena.alloc(i32, tokens.len);
        for (prompt, tokens) |*one, id| one.* = @intCast(id);
        const starts = try frontend.starts(arena, prompt);
        // the user's message, and the reply's header two tokens before the end
        try testing.expectEqual(2, starts.len);
        try testing.expectEqual(ids.of(&frontend, "<|user|>"), tokens[starts[0]]);
        try testing.expectEqual(prompt.len - 2, starts[1]);
        if (first_user) |at| try testing.expectEqual(at, starts[0]) else first_user = starts[0];
        const done = try sim.turn(prompt, 16, starts);
        defer gpa.free(done.planned);
        // cold from nothing, then from the first conversation's mark at the user's start
        try testing.expectEqual(if (i == 0) 0 else starts[0], done.cached);
        std.log.info("FP4-Local: prompt {d} tokens, messages at {any}, resumed at {d}, marks planned at {any}", .{ prompt.len, starts, done.cached, done.planned });
    }
}
