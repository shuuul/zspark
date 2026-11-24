//! `mtp.zig`'s tests on a checkpoint (spec 104 WS-06 moved them out of `mtp_test.zig`): drafts as stored. Its
//! first layers with its head, the head's cache bf16 beside FP8 trunk caches, are `checkpoint_prefix_test.zig`'s,
//! on the load each cache storage shares (spec 133 WS-02).

const std = @import("std");
const fixtures = @import("testing.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const draft = @import("../../../src/models/qwen4/draft.zig");
const model = @import("../../../src/models/qwen4/model.zig");

const testing = std.testing;
const heads = @import("mtp_fixture.zig");
const headReplyBeside = heads.headReplyBeside;

test "drafts as stored: a checkpoint's first layers draft with no draft head and the MTP experts in their stored format, and the reply is the serial one" {
    // On ZSPARK_FLASH_NEXT_FP4_LOCAL, whose MTP experts are NVFP4 as stored. The trunk is the default load's, and a
    // default load's drafted reply is its serial one (`checkpoint_prefix_test.zig`), so the serial reply here is the default's reply.
    const gpa = testing.allocator;
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    const count = fixtures.checkpoints.attentionLayers(found) orelse return error.SkipZigTest;
    const prompt = [_]i32{ 248045, 846, 198, 826, 279, 1118, 4097 };
    const settings: sampling.Sampling = .{ .seed = 5, .temperature = 1.2, .top_k = 60, .top_p = 0.98, .min_p = 0 };
    // the draft vocabulary is given, as a command gives it, and left unused
    var m: model.Model = try .loadWith(gpa, config, found, .{ .layers = count, .max_rows = draft.forward_max_rows, .draft_vocab = try model.DraftHead.defaultIds(arena), .drafts_stored = true, .kv = .fp8, .head_kv = .bf16 });
    defer m.deinit();
    try testing.expect(m.mtp.?.draft == null);
    try testing.expect(m.mtp.?.layer.moe.stored.nvfp4.routed == .nvfp4);
    // no fitted calibration: the checkpoint drafts by `Paying.published`
    try testing.expect(m.mtp.?.costs.?.landing == null);
    // a main layer's experts are NVFP4 as stored, whatever the switch
    try testing.expect(m.layers[0].moe.stored.nvfp4.routed == .nvfp4);
    const serial = try fixtures.serial.serialReply(gpa, m, &prompt, 24, settings);
    defer gpa.free(serial);
    for ([_]bool{ true, false }) |graphed| {
        const got = try headReplyBeside(gpa, m, &prompt, 24, settings, graphed, 0, true, null);
        defer gpa.free(got.tokens);
        try testing.expectEqualSlices(i32, serial, got.tokens);
        try testing.expect(got.stats.drafted > got.stats.rounds);
    }
}
