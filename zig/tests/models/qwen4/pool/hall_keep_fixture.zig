//! What the tests of the states a pool keeps share (`hall_keep_test.zig`, `hall_carry_test.zig`): the tokens
//! an engine gave each slot, an engine of a given shape, a round, a request, and the check that every state an
//! engine holds along a sequence is a serial slot's (`expectSerial`); and the check that a pool stream's kept
//! states are a serial slot's (`keptAsSerial`), which `checkpoint_prefix_test.zig` runs on a checkpoint too.

const std = @import("std");
const fixtures = @import("../testing.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const request_file = @import("../../../../src/server/request.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const forward = @import("../../../../src/models/qwen4/forward.zig");
const rounds = @import("../../../../src/models/qwen4/pool/rounds.zig");
const Engine = hall_engine.Engine;
const Keep = @import("../../../../src/models/qwen4/pool/hall_keep.zig").Keep;
const Out = hall_file.Out;
const fetchBytes = fixtures.attending.fetchBytes;
const greedy = @import("hall_fixture.zig").greedy;
const prefix = @import("../../../../src/decode/prefix.zig");

/// The sampled settings of `hall_keep_test.zig`'s replies.
pub const sampled: sampling.Sampling = .{ .seed = 7, .temperature = 0.8 };

/// The tokens the pool gave each slot, and which have ended.
pub const Collect = struct {
    gpa: Allocator,
    lists: [4]std.ArrayList(i32) = @splat(.empty),
    ended: [4]bool = @splat(false),

    fn emit(ctx: *anyopaque, slot: usize, tokens: []const i32) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.lists[slot].appendSlice(self.gpa, tokens) catch @panic("oom");
    }

    fn end(ctx: *anyopaque, slot: usize, _: request_file.Finish) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
    }

    fn fail(ctx: *anyopaque, slot: usize, message: []const u8) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
        std.debug.print("slot {d} failed: {s}\n", .{ slot, message });
    }

    pub fn out(self: *Collect) Out {
        return .{ .ctx = self, .emit = emit, .end = end, .fail = fail };
    }

    pub fn clear(self: *Collect, slot: usize) void {
        self.lists[slot].clearRetainingCapacity();
        self.ended[slot] = false;
    }

    pub fn deinit(self: *Collect) void {
        for (&self.lists) |*list| list.deinit(self.gpa);
    }
};

pub const Shape = struct {
    slots: usize = 1,
    context: usize = 512,
    prompt_rows: usize = 32,
    chunk: usize = 24,
    reply_stride: usize = 32,
    states_bytes: usize = hall_file.default_states_bytes,
};

pub fn open(gpa: Allocator, m: model_file.Model, shape: Shape) !Engine {
    return Engine.open(gpa, m, &.{}, .{
        .slots = shape.slots,
        .session = .{ .capacity = shape.context, .prompt_rows = shape.prompt_rows, .reply_stride = shape.reply_stride },
        .pool = (extents.lowerBound(shape.context, shape.slots) + 24) * extents.positions,
        .chunk = shape.chunk,
        .states_bytes = shape.states_bytes,
    });
}

pub fn step(engine: *Engine, collect: *Collect) !void {
    switch (engine.round(collect.out())) {
        .ran => engine.publishAhead(collect.out()),
        .short => return error.Short,
        .failed => return error.Failed,
    }
}

pub fn ask(prompt: []const i32, settings: sampling.Sampling, count: usize, starts: []const usize, gate: ?i32) hall_file.Ask {
    return .{ .prompt = prompt, .settings = settings, .max_tokens = count, .depth = 6, .opening = 6, .starts = starts, .gate = gate };
}

fn expectBuffer(gpa: Allocator, want: cuda.Buffer, got: cuda.Buffer) !void {
    const a = try fetchBytes(gpa, want);
    defer gpa.free(a);
    const b = try fetchBytes(gpa, got);
    defer gpa.free(b);
    try testing.expect(std.mem.eql(u8, a, b));
}

/// The first rows of `whole` are `part`'s.
pub fn expectLeading(gpa: Allocator, whole: cuda.Buffer, part: cuda.Buffer) !void {
    const a = try fetchBytes(gpa, whole);
    defer gpa.free(a);
    const b = try fetchBytes(gpa, part);
    defer gpa.free(b);
    try testing.expect(b.len > 0 and b.len <= a.len);
    try testing.expect(std.mem.eql(u8, a[0..b.len], b));
}

/// A pool's kept state is one allocation and its pooled keys another (spec 146 WS-04): each part a view that
/// lies inside its block.
pub fn expectOneAllocation(keep: Keep) !void {
    try expectInside(keep.state.block, &.{ keep.state.recurrent, keep.state.conv, keep.state.tails });
    if (keep.pooled.list.len > 0 and keep.pooled.list[0].view.ptr != null) {
        try expectInside(keep.pooled.block, &.{keep.pooled.list});
    } else try testing.expect(keep.pooled.block.view.ptr == null);
}

fn expectInside(block: cuda.Buffer, lists: []const []const cuda.Buffer) !void {
    try testing.expect(block.view.ptr != null);
    const first = @intFromPtr(block.view.ptr.?);
    var bytes: usize = 0;
    for (lists) |list| for (list) |part| {
        const at = @intFromPtr(part.view.ptr.?);
        try testing.expect(at >= first and at + part.bytes() <= first + block.bytes());
        bytes += part.bytes();
    };
    try testing.expect(bytes <= block.bytes());
}

/// Every state `engine` holds along `sequence` is the state of a slot that ran `sequence` a row a round to its
/// position: the recurrent side, the conv inputs, the PLE's tail and history, the pooled keys below it, the
/// head's length and the token after. Returns how many were compared.
pub fn expectSerial(gpa: Allocator, m: model_file.Model, engine: *Engine, sequence: []const i32, shape: Shape) !usize {
    var serial = try open(gpa, m, .{ .slots = 1, .context = shape.context, .prompt_rows = 1, .reply_stride = 0 });
    serial.bindPool();
    defer serial.deinit();
    serial.keeps.chunks = false;
    var collect: Collect = .{ .gpa = gpa };
    defer collect.deinit();
    const longer = try gpa.alloc(i32, sequence.len + 1);
    defer gpa.free(longer);
    @memcpy(longer[0..sequence.len], sequence);
    longer[sequence.len] = 1;
    _ = try serial.begin(0, ask(longer, greedy, 1, &.{}, null), 0);
    var compared: usize = 0;
    const slot = &serial.slots[0];
    while (slot.pos + 1 < sequence.len) {
        try step(&serial, &collect);
        const pos = slot.pos;
        for (engine.keeps.list.items) |*keep| {
            if (keep.pos != pos or !std.mem.eql(i32, keep.chain.tokens.items[0..pos], sequence[0..pos])) continue;
            // a chain that goes on with another token parted from `sequence` here: its state is not on it
            if (keep.chain.tokens.items.len > pos and keep.chain.tokens.items[pos] != sequence[pos]) continue;
            errdefer std.debug.print("the state at {d}, {t}\n", .{ pos, keep.kind });
            try testing.expectEqual(sequence[pos], keep.next);
            try testing.expectEqual(pos, keep.state.pos);
            for (keep.state.recurrent, keep.state.conv, slot.conv, 0..) |recurrent, conv, source, linear| {
                // the slot's state at its position (`rounds.committed`, spec 144 WS-09)
                var at = try forward.Snapshot.likeOf(recurrent);
                defer at.deinit();
                try rounds.committed(slot.*, linear, at);
                try expectBuffer(gpa, at, recurrent);
                try expectBuffer(gpa, source, conv);
            }
            for (keep.state.tails, keep.state.history, slot.ple) |tail, history, branch| {
                if (tail.bytes() > 0) try expectBuffer(gpa, branch.tail, tail);
                try testing.expectEqualSlices(u32, branch.history, history);
            }
            const blocks = if (m.attention) |shared| pos / shared.shape.ratio else 0;
            try testing.expectEqual(slot.caches.len + @intFromBool(slot.head != null), keep.pooled.list.len);
            for (keep.pooled.list, 0..) |pooled, i| {
                if (blocks == 0) {
                    try testing.expect(pooled.view.ptr == null);
                    continue;
                }
                try testing.expectEqual(blocks, @as(usize, @intCast(pooled.view.size[0])));
                try expectLeading(gpa, if (i < slot.caches.len) slot.caches[i].pooled else slot.head.?.pooled, pooled);
            }
            if (slot.head != null) try testing.expectEqual(slot.head_len, keep.head_len);
            compared += 1;
        }
    }
    serial.preempt(0);
    return compared;
}

pub const Got = struct { tokens: []i32, cached: usize };

/// One request alone on slot 0, to its end: the reply's tokens, the caller's.
pub fn one(engine: *Engine, collect: *Collect, asked: hall_file.Ask) !Got {
    collect.clear(0);
    const cached = try engine.begin(0, asked, 0);
    var spins: usize = 0;
    while (!collect.ended[0]) : (spins += 1) {
        if (spins == 4096) return error.Stuck;
        try step(engine, collect);
    }
    const tokens = try engine.gpa.dupe(i32, collect.lists[0].items);
    _ = engine.retire(0, .ended, tokens.len);
    return .{ .tokens = tokens, .cached = cached };
}

pub fn countOf(engine: *const Engine, kind: prefix.Kind) usize {
    var n: usize = 0;
    for (engine.keeps.list.items) |keep| n += @intFromBool(keep.kind == kind);
    return n;
}

pub fn positions(gpa: Allocator, engine: *const Engine, chunks: bool) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(gpa);
    for (engine.keeps.list.items) |keep| if (chunks or keep.kind != .chunk) try list.append(gpa, keep.pos);
    std.mem.sort(usize, list.items, {}, std.sort.asc(usize));
    return list.toOwnedSlice(gpa);
}

pub fn randomTokens(random: std.Random, out: []i32) void {
    for (out) |*token| token.* = random.intRangeLessThan(i32, 1, 250);
}

pub const starts_of_150 = [_]usize{ 0, 37, 90, 141 };

/// A prompt of 150 tokens and a reply of 44 on `m`, taken within the rounds and by chunks cut at each state:
/// both hold the states a row at a time gives, at the same positions. With `beside`, a reply decodes on the
/// next slot while the prompt runs, and its tokens and its end are the serial ones.
pub fn keptAsSerial(gpa: Allocator, m: model_file.Model, shape: Shape, beside: bool) !void {
    var random: std.Random.DefaultPrng = .init(54);
    var prompt: [150]i32 = undefined;
    randomTokens(random.random(), &prompt);
    const other = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const replied = 44;
    var planned: [2][]usize = undefined;
    var made: usize = 0;
    defer for (planned[0..made]) |list| gpa.free(list);
    var replies: [2][]i32 = undefined;
    defer for (replies[0..made]) |reply| gpa.free(reply);
    for ([_]bool{ false, true }) |cut| {
        errdefer std.debug.print("cut {}, beside {}, prompt windows of {d}, chunks of {d}\n", .{ cut, beside, shape.prompt_rows, shape.chunk });
        var engine = try open(gpa, m, shape);
        engine.bindPool();
        defer engine.deinit();
        engine.keeps.cut = cut;
        var collect: Collect = .{ .gpa = gpa };
        defer collect.deinit();
        if (beside) {
            _ = try engine.begin(1, ask(&other, sampled, 40, &.{}, null), 0);
            for (0..3) |_| try step(&engine, &collect);
        }
        _ = try engine.begin(0, ask(&prompt, sampled, replied, &starts_of_150, null), 0);
        var spins: usize = 0;
        while (!(collect.ended[0] and (!beside or collect.ended[1]))) : (spins += 1) {
            if (spins == 1024) return error.Stuck;
            try step(&engine, &collect);
        }
        _ = engine.retire(0, .ended, replied);
        if (beside) _ = engine.retire(1, .ended, 40);
        try testing.expectEqual(replied, collect.lists[0].items.len);
        replies[made] = try gpa.dupe(i32, collect.lists[0].items);
        planned[made] = try positions(gpa, &engine, false);
        made += 1;

        var sequence: std.ArrayList(i32) = .empty;
        defer sequence.deinit(gpa);
        try sequence.appendSlice(gpa, &prompt);
        try sequence.appendSlice(gpa, collect.lists[0].items);
        const held = engine.keeps.list.items.len;
        var compared = try expectSerial(gpa, m, &engine, sequence.items, shape);
        if (beside) {
            const serial = try fixtures.serial.serialReply(gpa, m, &other, 40, sampled);
            defer gpa.free(serial);
            try testing.expectEqualSlices(i32, serial, collect.lists[1].items);
            sequence.clearRetainingCapacity();
            try sequence.appendSlice(gpa, &other);
            try sequence.appendSlice(gpa, serial);
            compared += try expectSerial(gpa, m, &engine, sequence.items, shape);
        }
        try testing.expectEqual(held, compared);
        for (engine.keeps.list.items) |keep| try expectOneAllocation(keep);
        // the messages' starts above the first, the mark before the end, a stride and the end
        try testing.expect(countOf(&engine, .message) >= 2 and countOf(&engine, .tail) == @as(usize, if (beside) 2 else 1) and countOf(&engine, .stride) >= 1);
        try testing.expectEqual(@as(usize, if (beside) 2 else 1), countOf(&engine, .end));
        if (cut) try testing.expectEqual(0, engine.keeps.took.within) else try testing.expect(engine.keeps.took.within >= 3);
    }
    try testing.expectEqualSlices(i32, replies[0], replies[1]);
    try testing.expectEqualSlices(usize, planned[1], planned[0]);
}
