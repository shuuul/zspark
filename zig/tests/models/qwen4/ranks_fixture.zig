//! `ranks_test.zig`'s drafted reply on two ranks with the MTP head whole beside rank 0's share, which
//! `mtp_ranks_test.zig` also runs on the checkpoint's first layers it loads once for both head forms (spec 133
//! WS-08 moved it here).

const std = @import("std");
const testing = std.testing;
const draft = @import("../../../src/decode/draft.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const qwen4_config = @import("../../../src/models/qwen4/config.zig");
const mtp_file = @import("../../../src/models/qwen4/mtp.zig");
const ranks = @import("../../../src/models/qwen4/ranks_forward.zig");

const Allocator = std.mem.Allocator;
const Pair = ranks.Held(2);

const ns = struct {
    pub const Model = Pair.Model;
    pub const State = Pair.State;
    pub const Buffers = Pair.Buffers;
    pub const Graphs = Pair.Graphs;
    pub const Error = Pair.Error;
    pub const forward = Pair.forward;
    pub const forwardScoring = Pair.forwardScoring;
    pub const commit = Pair.commit;
    pub const logitsRow = Pair.logitsRow;
    pub const streams = Pair.streams;
};

const Decoder = draft.Decoder(ns);

pub const Drafted = struct { tokens: []i32, stats: draft.Stats };

/// `count` tokens after `prompt` on both ranks in loopback: one row a round, or in rounds that rank 0's head
/// drafts for (`chain` its floor, 0 filling every window). The head's graphs with `graphed`; the trunk is
/// launched.
pub fn replyOn(gpa: Allocator, held: Pair.Model, prompt: []const i32, count: usize, settings: sampling.Sampling, drafting: bool, graphed: bool, chain: f32, capacity: usize) !Drafted {
    const rows = held.max_rows;
    const vocab = held.ranks[0].shape.vocab;
    var state: Pair.State = try .init(gpa, held, rows, capacity);
    defer state.deinit();
    var b: Pair.Buffers = try .init(gpa, held, rows, capacity);
    defer b.deinit();
    var sampler: sampling.Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, vocab);
    defer gpa.free(logits);
    var decoder: Decoder = .{ .m = held, .state = &state, .b = &b, .graphs = null, .sampler = &sampler, .settings = settings, .eos = &.{}, .logits = logits };
    var head: ?mtp_file.Head = if (drafting) try .init(gpa, held.ranks[0], &sampler, settings, .{ .rows = rows, .capacity = capacity, .graphs = graphed }) else null;
    defer if (head) |*h| h.deinit();
    if (head) |*h| {
        h.chain = chain;
        // the head's attention computes in a scratch of its own, sized for the whole model's heads
        try testing.expect(h.b.layer.attention != null and !h.b.layer.attention_lent);
    }
    const drafter: ?draft.Drafter = if (head) |*h| h.drafter() else null;

    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    var chosen: [draft.forward_max_rows]i32 = undefined;
    var drafts: [draft.forward_max_rows]i32 = undefined;
    var kept = try decoder.prompt(drafter, prompt, &chosen);
    out[0] = chosen[kept - 1];
    var made: usize = 1;
    while (made < count) {
        const n = if (drafter) |d| try decoder.draft(d, chosen[0..kept], count - made - 1, &drafts) else 0;
        kept = try decoder.round(chosen[kept - 1], drafts[0..n], &chosen);
        @memcpy(out[made..][0..kept], chosen[0..kept]);
        made += kept;
    }
    // the reply's last window absorbed, as a session leaves the head: it stands where the state does
    if (drafter) |d| _ = try decoder.draft(d, chosen[0..kept], 0, &drafts);
    if (head) |h| try testing.expectEqual(state.pos, h.state.position.host);
    return .{ .tokens = out, .stats = decoder.stats };
}

pub const reply_settings = [_]sampling.Sampling{
    .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 },
    .{ .seed = 11, .temperature = 1.5, .top_k = 40, .top_p = 0.95, .min_p = 0 },
};

/// A checkpoint's first layers on two ranks in `held`, its MTP head whole beside rank 0's share and both ranks
/// loaded with the same options: rank 1 holds nothing of the head, and the drafted reply is the serial one.
/// `ids` the draft vocabulary the load was given.
pub fn wholeHeadDrafted(gpa: Allocator, held: Pair.Model, config: qwen4_config.Config, ids: []const u32) !void {
    try testing.expect(held.ranks[0].mtp != null and held.ranks[1].mtp == null);
    const whole = held.ranks[0].mtp.?.whole.?;
    try testing.expectEqual(config.moe_width, whole.shape.width);
    try testing.expectEqual(config.moe_width / 2, held.ranks[0].shape.width);
    try testing.expectEqual(config.heads, whole.attention.shape.heads);
    try testing.expectEqual(config.heads / 2, held.ranks[0].attention.?.shape.heads);
    try testing.expectEqual(ids.len, held.ranks[0].mtp.?.draft.?.ids.len);

    const prompt = [_]i32{ 151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198 };
    const count = 24;
    for (reply_settings) |settings| {
        const serial = try replyOn(gpa, held, &prompt, count, settings, false, false, 0, 4096);
        defer gpa.free(serial.tokens);
        const got = try replyOn(gpa, held, &prompt, count, settings, true, true, 0, 4096);
        defer gpa.free(got.tokens);
        std.debug.print("two ranks, four layers, the head beside rank 0: {d} rounds, {d} of {d} drafts kept\n", .{ got.stats.rounds, got.stats.accepted, got.stats.drafted });
        try testing.expectEqualSlices(i32, serial.tokens, got.tokens);
        try testing.expectEqual(count - 1, got.stats.rounds + got.stats.accepted);
    }
}
