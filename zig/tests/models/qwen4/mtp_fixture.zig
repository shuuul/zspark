//! What `mtp_test.zig`, `mtp_pick_test.zig`, `mtp_checkpoint_test.zig` and `checkpoint_prefix_test.zig` call:
//! random bf16 values, a buffer made from host values, the ids a draft head is given, a reply the head drafts
//! for, and that reply held to the serial one on a checkpoint's first layers (`draftedAsSerial`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric_fixture = @import("../../weights/numeric_fixture.zig");
const gpu = @import("../../cuda/testing.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const draft = @import("../../../src/models/qwen4/draft.zig");
const forward = @import("../../../src/models/qwen4/forward.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const mtp_file = @import("../../../src/models/qwen4/mtp.zig");
const Head = mtp_file.Head;
const testing = std.testing;
const fixtures = @import("testing.zig");

pub const randomBf16 = numeric_fixture.randomBf16;

pub const device = gpu.device;

/// The ids below `vocab` that are not multiples of three.
pub fn testSubset(arena: Allocator, vocab: u32) ![]u32 {
    var list: std.ArrayList(u32) = .empty;
    for (0..vocab) |id| if (id % 3 != 0) try list.append(arena, @intCast(id));
    return list.items;
}

pub const Reply = struct { tokens: []i32, stats: draft.Stats };

/// `count` tokens after `prompt`, decoded in rounds the head drafts for.
pub fn headReply(gpa: Allocator, m: model.Model, prompt: []const i32, count: usize, settings: sampling.Sampling, graphed: bool, confidence: f32) !Reply {
    return headReplyBeside(gpa, m, prompt, count, settings, graphed, confidence, false, null);
}

/// `headReply`; with `lent` the head's attention computes in the main model's scratch, as a session's does.
/// With `costs` the head drafts by them (`Paying`), as a checkpoint's does.
pub fn headReplyBeside(gpa: Allocator, m: model.Model, prompt: []const i32, count: usize, settings: sampling.Sampling, graphed: bool, confidence: f32, lent: bool, costs: ?mtp_file.Paying.Costs) !Reply {
    var state: forward.State = try .init(gpa, m, m.max_rows, 256);
    defer state.deinit();
    var b: forward.Buffers = try .init(gpa, m, m.max_rows, 256);
    defer b.deinit();
    var graphs: forward.Graphs = try .init(gpa);
    defer graphs.deinit();
    var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, m.shape.vocab);
    defer gpa.free(logits);
    var decoder: draft.Decoder = .{ .m = m, .state = &state, .b = &b, .graphs = if (graphed) &graphs else null, .sampler = &sampler, .settings = settings, .eos = &.{}, .logits = logits };
    var head: Head = try .init(gpa, m, &sampler, settings, .{ .rows = m.max_rows, .capacity = 256, .graphs = graphed, .beside = if (lent) &b else null });
    defer head.deinit();
    try testing.expectEqual(m.attention.?.head.dtype(), head.state.cache.k.view.dtype);
    try testing.expectEqual(m.attention.?.head.scaled(), head.state.cache.k_scale != null);
    if (lent) try testing.expectEqual(b.attention.?.pa.view.ptr, head.b.layer.attention.?.pa.view.ptr);
    head.confidence = confidence;
    head.chain = 0;
    if (costs) |given| {
        head.costs = given;
        head.fresh();
    }

    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    var chosen: [draft.forward_max_rows]i32 = undefined;
    var drafts: [draft.forward_max_rows]i32 = undefined;
    var kept = try decoder.prompt(head.drafter(), prompt, &chosen);
    out[0] = chosen[kept - 1];
    var made: usize = 1;
    while (made < count) {
        const n = try decoder.draft(head.drafter(), chosen[0..kept], count - made - 1, &drafts);
        kept = try decoder.round(chosen[kept - 1], drafts[0..n], &chosen);
        @memcpy(out[made..][0..kept], chosen[0..kept]);
        made += kept;
    }
    return .{ .tokens = out, .stats = decoder.stats };
}

/// On a checkpoint's first layers with its head: the reply drafted through graphs and eagerly, the head's
/// attention in the trunk's scratch as a session's is, is the serial one, and the head drafted.
pub fn draftedAsSerial(gpa: Allocator, m: model.Model) !void {
    const prompt = [_]i32{ 248045, 846, 198, 826, 279, 1118, 4097 };
    const settings: sampling.Sampling = .{ .seed = 5, .temperature = 1.2, .top_k = 60, .top_p = 0.98, .min_p = 0 };
    try testing.expect(m.mtp != null);
    const serial = try fixtures.serial.serialReply(gpa, m, &prompt, 24, settings);
    defer gpa.free(serial);
    for ([_]bool{ true, false }) |graphed| {
        const got = try headReplyBeside(gpa, m, &prompt, 24, settings, graphed, 0, true, null);
        defer gpa.free(got.tokens);
        try testing.expectEqualSlices(i32, serial, got.tokens);
        try testing.expect(got.stats.drafted > got.stats.rounds);
    }
}
