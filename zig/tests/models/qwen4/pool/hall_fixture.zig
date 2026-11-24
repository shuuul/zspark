//! What the tests of the pool's shared rounds share (`hall_test.zig`, `hall_streams_test.zig`,
//! `hall_draft_test.zig`): the tokens an engine gave each slot, a small engine over a model, the greedy sampling,
//! and `Bench`, a core's slots drained in rounds (`coreFor`, `askFor`, `openPool`); and a reply the serial one
//! before a second stream, beside others and after (`sameBeforeBesideAfter`), which `checkpoint_prefix_test.zig`
//! runs on a checkpoint too.

const std = @import("std");
const Allocator = std.mem.Allocator;
const extents = @import("../../../../src/decode/extents.zig");
const sampling = @import("../../../../src/decode/sampling.zig");
const request_file = @import("../../../../src/server/request.zig");
const tokenizer = @import("../../../../src/text/tokenizer.zig");
const hall_engine = @import("../../../../src/models/qwen4/pool/hall_engine.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const model_file = @import("../../../../src/models/qwen4/model.zig");
const session = @import("../../../../src/models/qwen4/session/session.zig");
const rig = @import("../../../../src/models/qwen4/dflash2/rig.zig");
const Engine = hall_engine.Engine;
const Ask = hall_file.Ask;
const Core = hall_file.Core;
const Out = hall_file.Out;
const testing = std.testing;
const fixtures = @import("../testing.zig");
const cuda = @import("../../../../src/cuda/cuda.zig");
const attention = @import("../../../../src/models/qwen4/blocks/attention.zig");

pub const greedy: sampling.Sampling = .{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };

pub const Collect = struct {
    gpa: Allocator,
    lists: []std.ArrayList(i32),
    ended: []bool,

    pub fn emit(ctx: *anyopaque, slot: usize, tokens: []const i32) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.lists[slot].appendSlice(self.gpa, tokens) catch @panic("oom");
    }

    pub fn end(ctx: *anyopaque, slot: usize, _: request_file.Finish) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
    }

    pub fn fail(ctx: *anyopaque, slot: usize, message: []const u8) void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
        std.debug.print("slot {d} failed: {s}\n", .{ slot, message });
    }
};

pub fn engineFor(gpa: Allocator, m: model_file.Model, slots: usize, chunk: usize) !Engine {
    return engineDrafting(gpa, m, slots, chunk, null);
}

/// `engineFor`, its buffers tapped for `drafter` (`session.Options.drafter`), which `Engine.seatDrafter` seats.
pub fn engineDrafting(gpa: Allocator, m: model_file.Model, slots: usize, chunk: usize, drafter: ?*const rig.Opened) !Engine {
    const context: usize = 256;
    const places = extents.lowerBound(context, slots);
    return Engine.open(gpa, m, &.{}, .{
        .slots = slots,
        .session = .{ .capacity = context, .prompt_rows = 32, .drafter = drafter, .copies = drafter == null },
        .pool = places * extents.positions,
        .chunk = chunk,
        .lazy_graphs = false,
    });
}

/// What a decoder's rounds gave, a slot each, and what the scheduler would answer it.
pub const Bench = struct {
    gpa: Allocator,
    lists: [8]std.ArrayList(i32) = @splat(.empty),
    ended: [8]bool = @splat(false),
    finish: [8]?request_file.Finish = @splat(null),
    held: [8]?usize = @splat(null),
    failed: bool = false,
    /// A request waits once slot 0 has been given this many tokens.
    wait_at: ?usize = null,

    pub fn deinit(self: *Bench) void {
        for (&self.lists) |*list| list.deinit(self.gpa);
    }

    pub fn reset(self: *Bench, slot: usize) void {
        self.lists[slot].clearRetainingCapacity();
        self.ended[slot] = false;
        self.finish[slot] = null;
        self.held[slot] = null;
    }

    pub fn out(self: *Bench) Out {
        return .{ .ctx = self, .emit = emit, .end = end, .fail = fail, .held = told, .waiting = waiting };
    }

    pub fn emit(ctx: *anyopaque, slot: usize, tokens: []const i32) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.lists[slot].appendSlice(self.gpa, tokens) catch @panic("oom");
    }

    pub fn end(ctx: *anyopaque, slot: usize, finish: request_file.Finish) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
        self.finish[slot] = finish;
    }

    pub fn fail(ctx: *anyopaque, slot: usize, message: []const u8) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.ended[slot] = true;
        self.failed = true;
        std.debug.print("slot {d} failed: {s}\n", .{ slot, message });
    }

    pub fn told(ctx: *anyopaque, slot: usize, cached: usize) void {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        self.held[slot] = cached;
    }

    pub fn waiting(ctx: *anyopaque) bool {
        const self: *Bench = @ptrCast(@alignCast(ctx));
        return if (self.wait_at) |n| self.lists[0].items.len >= n else false;
    }

    /// Rounds until every slot of `slots` has ended.
    pub fn drain(self: *Bench, core: *Core, slots: []const usize) !void {
        var spins: usize = 0;
        while (spins < 400) : (spins += 1) {
            const done = for (slots) |slot| {
                if (!self.ended[slot]) break false;
            } else true;
            if (done) break;
            switch (core.round(self.out())) {
                .ran => {},
                .short => return error.Short,
                .failed => return error.Failed,
            }
        }
        for (slots) |slot| try testing.expect(self.ended[slot]);
        try testing.expect(!self.failed);
    }

    pub const Got = struct { tokens: []i32, cached: usize, finish: request_file.Finish };

    /// One request from its admission to its end, with nothing beside it.
    pub fn one(self: *Bench, core: *Core, ask: Ask) !Got {
        const taken = try core.admit(ask, &.{});
        self.reset(taken.slot);
        try self.drain(core, &.{taken.slot});
        const tokens = try self.gpa.dupe(i32, self.lists[taken.slot].items);
        _ = core.retire(taken.slot, .ended, tokens.len);
        return .{ .tokens = tokens, .cached = self.held[taken.slot] orelse taken.cached, .finish = self.finish[taken.slot].? };
    }
};

pub fn coreFor(gpa: Allocator, m: model_file.Model, tokens: *const tokenizer.Tokenizer, eos: []const u32, slots: usize, context: usize) !*Core {
    return coreDrafting(gpa, m, tokens, eos, slots, context, null);
}

/// `coreFor` whose session drafts by `drafter` (DFlash2), as `serve --drafter` makes it.
pub fn coreDrafting(gpa: Allocator, m: model_file.Model, tokens: *const tokenizer.Tokenizer, eos: []const u32, slots: usize, context: usize, drafter: ?*const rig.Opened) !*Core {
    return Core.create(gpa, testing.io, m, tokens, eos, .{
        .slots = slots,
        .session = .{ .capacity = context, .prompt_rows = 32, .drafter = drafter, .copies = drafter == null },
        .pool = extents.lowerBound(context, slots) * extents.positions,
        .chunk = 8,
    });
}

pub fn askFor(prompt: []const i32, settings: sampling.Sampling, count: usize) Ask {
    return .{ .prompt = prompt, .settings = settings, .max_tokens = count, .depth = session.default_depth, .opening = session.default_depth };
}

/// Opens `core`'s pool: two requests at once, run to their ends.
pub fn openPool(bench: *Bench, core: *Core) !void {
    const a = try core.admit(askFor(&.{ 9, 8, 7 }, greedy, 2), &.{});
    const b = try core.admit(askFor(&.{ 6, 5 }, greedy, 2), &.{});
    bench.reset(a.slot);
    bench.reset(b.slot);
    try testing.expect(core.engine != null and core.session == null);
    try bench.drain(core, &.{ a.slot, b.slot });
    _ = core.retire(a.slot, .ended, 2);
    _ = core.retire(b.slot, .ended, 2);
}

/// Two streams admitted at once to `core`, whose session drafts by DFlash2 (`coreDrafting`, `serve --drafter`):
/// the second opens the pool, which seats the session's drafter; each reply is the serial one on `m`, and
/// drafts were verified in its windows.
pub fn twoDrafted(gpa: Allocator, core: *Core, m: model_file.Model, prompts: [2][]const i32, settings: [2]sampling.Sampling, count: usize) !void {
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();
    var slots: [2]usize = undefined;
    for (settings, prompts, &slots) |one, prompt, *slot| {
        slot.* = (try core.admit(askFor(prompt, one, count), &.{})).slot;
        bench.reset(slot.*);
    }
    try testing.expect(core.session == null and core.engine.?.flash != null);
    try bench.drain(core, &slots);
    for (settings, prompts, slots) |one, prompt, slot| {
        const expected = try fixtures.serial.serialReply(gpa, m, prompt, count, one);
        defer gpa.free(expected);
        try testing.expectEqualSlices(i32, expected, bench.lists[slot].items);
        const tail = core.retire(slot, .ended, count);
        try testing.expect(tail.drafted > tail.rounds);
    }
}

pub const Storages = struct { kv: attention.Storage, head: attention.Storage };

/// On `m`, loaded with `case`'s storages: a sampled reply is the serial one from the session before a second
/// stream, as three others arrive mid-reply, beside three others from its first row, and alone in the pool
/// after; the session's head and the pool's hold their cache as `case.head`. With `kept`, the head's pool
/// arrays at the end are compared with an all-bf16 load's (kept by the first such call), where the head is
/// bf16: for a model whose trunk does not attend.
pub fn sameBeforeBesideAfter(gpa: Allocator, m: model_file.Model, tokens: *const tokenizer.Tokenizer, case: Storages, kept: ?*[3]?[]u8) !void {
    const storage = case.kv;
    const core = try coreFor(gpa, m, tokens, &.{}, 4, 256);
    defer core.destroy();
    var bench: Bench = .{ .gpa = gpa };
    defer bench.deinit();

    const count = 24;
    const sampled: sampling.Sampling = .{ .seed = 5, .temperature = 0.9 };
    const prompt = [_]i32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    const others = [_][]const i32{ &.{ 1, 2, 3, 4 }, &.{ 8, 9, 10 }, &.{ 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 } };
    const expected = try fixtures.serial.serialReply(gpa, m, &prompt, count, sampled);
    defer gpa.free(expected);

    // before a second stream: the session's caches
    try testing.expectEqual(storage, core.session.?.m.attention.?.shape.kv);
    try testing.expectEqual(case.head.dtype(), core.session.?.head.?.state.cache.k.view.dtype);
    try testing.expectEqual(storage.dtype(), core.session.?.buffers.attention.?.q.view.dtype);
    const before = try bench.one(core, askFor(&prompt, sampled, count));
    defer gpa.free(before.tokens);
    try testing.expectEqualSlices(i32, expected, before.tokens);
    try testing.expect(core.engine == null);

    // as it comes: the stream is live, some of its tokens sent, when three others arrive
    const live = try core.admit(askFor(&prompt, sampled, count), &.{});
    bench.reset(live.slot);
    bench.wait_at = 3;
    _ = core.round(bench.out());
    bench.wait_at = null;
    const sent = bench.lists[live.slot].items.len;
    try testing.expect(sent >= 3 and sent < count and !bench.ended[live.slot]);
    var slots: [4]usize = undefined;
    slots[0] = live.slot;
    for (others, slots[1..]) |other, *slot| {
        slot.* = (try core.admit(askFor(other, greedy, count), &.{})).slot;
        bench.reset(slot.*);
    }
    try testing.expect(core.session == null);
    // the pool's caches are the storage the session's were
    try testing.expectEqual(storage, core.engine.?.m.attention.?.shape.kv);
    try testing.expectEqual(case.head.dtype(), core.engine.?.head_pool.?.k.view.dtype);
    try testing.expectEqual(case.head.scaled(), core.engine.?.head_pool.?.k_scale != null);
    try bench.drain(core, &slots);
    try testing.expectEqualSlices(i32, expected, bench.lists[live.slot].items);
    for (slots) |slot| _ = core.retire(slot, .ended, count);

    // beside three others, from its first row
    for (others, slots[0..3]) |other, *slot| {
        slot.* = (try core.admit(askFor(other, sampled, count), &.{})).slot;
        bench.reset(slot.*);
    }
    slots[3] = (try core.admit(askFor(&prompt, sampled, count), &.{})).slot;
    bench.reset(slots[3]);
    try bench.drain(core, &slots);
    try testing.expectEqualSlices(i32, expected, bench.lists[slots[3]].items);
    for (slots) |slot| _ = core.retire(slot, .ended, count);

    // and after, alone in the pool
    const after = try bench.one(core, askFor(&prompt, sampled, count));
    defer gpa.free(after.tokens);
    try testing.expectEqualSlices(i32, expected, after.tokens);

    const plain = kept orelse return;
    if (case.head != .bf16) return;
    try cuda.synchronize();
    const head_pool = core.engine.?.head_pool.?;
    for ([_]cuda.Buffer{ head_pool.k, head_pool.v, head_pool.index_keys }, plain) |buffer, *held| {
        const bytes = try gpa.alloc(u8, buffer.bytes());
        errdefer gpa.free(bytes);
        try buffer.download(bytes);
        if (held.*) |want| {
            try testing.expect(storage != .bf16);
            try testing.expect(std.mem.eql(u8, want, bytes));
            gpa.free(bytes);
        } else {
            try testing.expectEqual(attention.Storage.bf16, storage);
            // the rounds left rows there
            try testing.expect(!std.mem.allEqual(u8, bytes, 0));
            held.* = bytes;
        }
    }
}
