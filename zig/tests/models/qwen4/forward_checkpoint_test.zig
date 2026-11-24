//! `forward.zig`'s test on a checkpoint's first layers in the card's prompt form (spec 104 WS-06 moved it out of
//! `forward_test.zig`): 4096-row windows. The same layers' graphs, wide windows and snapshot in the bf16 form are
//! `checkpoint_prefix_test.zig`'s, on the load its checks share (`forward_fixture.firstLayers`, spec 133 WS-08).

const std = @import("std");
const testing = std.testing;
const fixtures = @import("testing.zig");
const cuda = @import("../../../src/cuda/cuda.zig");
const forward_file = @import("../../../src/models/qwen4/forward.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const drafted_fixture = @import("draft_fixture.zig");
const Buffers = forward_file.Buffers;
const State = forward_file.State;
const commit = forward_file.commit;
const downloadRow = forward_file.downloadRow;
const forward = forward_file.forward;

test "a mixed-precision checkpoint's first layers, to its first attention layer, in the card's prompt form: drafted decoding gives the serial reply; in prompt windows of 4096 rows each row the bits of 2048-row windows in either form, and of its serial step in one" {
    // Spec 105 WS-07: `--max-num-batched-tokens 4096` loads the model for 4096 rows. Runs with
    // ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory; skipped without. Every kernel of the first
    // layers runs at 4096 rows here, the routed experts, the MXFP8 linears, the PLE and the attention's chunks.
    // Loaded in the card's form (spec 143 WS-06): its prompt windows' rows FP8 and W4A4, each row's bits its own
    // within the form (a 4096-row window's as 2048-row windows'); then one form, every window bf16, as the steps.
    // One load for `draft_test.zig`'s drafted reply in the card's form too (spec 133 WS-08), which loaded these
    // layers for windows of four rows: its windows are four rows here (`checkpoints.narrowed`).
    const session = @import("../../../src/models/qwen4/session/session.zig");
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    const count = fixtures.checkpoints.attentionLayers(found) orelse return error.SkipZigTest;
    const wide = session.most_prompt_rows;
    // the pool's rounds' windows, the least `--max-num-batched-tokens` takes
    const narrow = session.promptRows(config);
    try testing.expect(session.refusedRows(config, wide) == null and narrow < wide);
    const form = model.prompt_form;
    model.prompt_form = .card;
    defer model.prompt_form = form;
    var m: model.Model = try .load(gpa, config, found, count, wide);
    defer m.deinit();
    {
        // the experts on the W4A4 pack, whose steps and verify windows run A16: one form, so drafting stays exact
        errdefer std.debug.print("the card's prompt form: the drafted reply\n", .{});
        try drafted_fixture.checkpointDrafted(gpa, m);
    }
    errdefer std.debug.print("the card's prompt form: 4096-row windows\n", .{});
    var long: [wide + 104]i32 = undefined;
    for (&long, 0..) |*token, i| token.* = @intCast(1000 + i * 37 + (i * i) % 911);
    const card_widest = try rowHashes(gpa, m, &long, &.{wide});
    defer gpa.free(card_widest);
    const card_halves = try rowHashes(gpa, m, &long, &.{narrow});
    defer gpa.free(card_halves);
    try testing.expectEqualSlices(u64, card_halves, card_widest);

    const bound = forward_file.bf16_rows;
    forward_file.bf16_rows = std.math.maxInt(usize);
    defer forward_file.bf16_rows = bound;
    const widest = try rowHashes(gpa, m, &long, &.{wide});
    defer gpa.free(widest);
    const halves = try rowHashes(gpa, m, &long, &.{narrow});
    defer gpa.free(halves);
    try testing.expectEqualSlices(u64, halves, widest);
    try testing.expect(widest[0] != widest[1] and widest[narrow] != widest[wide]);
    // the forms differ: the card's rows are quantized
    var differ: usize = 0;
    for (card_widest, widest) |a, b| differ += @intFromBool(a != b);
    try testing.expect(differ > long.len / 2);
    // serial steps across both windows' ends, from states the windows before them left: 2016 to 2080 and
    // 4064 to the end, every other row in windows
    var plan: [2 + 64 + 136]usize = undefined;
    plan[0] = narrow - 32;
    @memset(plan[1..65], 1);
    plan[65] = wide - 32 - narrow - 32;
    @memset(plan[66..], 1);
    const stepped = try rowHashes(gpa, m, &long, &plan);
    defer gpa.free(stepped);
    try testing.expectEqualSlices(u64, widest[narrow - 32 ..][0..64], stepped[narrow - 32 ..][0..64]);
    try testing.expectEqualSlices(u64, widest[wide - 32 ..], stepped[wide - 32 ..]);
}

/// A hash of each row's logits, `tokens` run in windows of `plan`'s widths in turn (its last again to the
/// end), every row kept.
fn rowHashes(gpa: std.mem.Allocator, m: model.Model, tokens: []const i32, plan: []const usize) ![]u64 {
    const vocab = m.shape.vocab;
    var state: State = try .init(gpa, m, m.max_rows, 4608);
    defer state.deinit();
    var b: Buffers = try .init(gpa, m, m.max_rows, 4608);
    defer b.deinit();
    const row = try gpa.alloc(u16, vocab);
    defer gpa.free(row);
    const out = try gpa.alloc(u64, tokens.len);
    errdefer gpa.free(out);
    var at: usize = 0;
    var step: usize = 0;
    while (at < tokens.len) : (step += 1) {
        const rows = @min(plan[@min(step, plan.len - 1)], tokens.len - at);
        try forward(m, &state, &b, tokens[at..][0..rows]);
        try cuda.synchronize();
        for (0..rows) |r| {
            try downloadRow(b.logits, r, vocab, row);
            out[at + r] = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(row));
        }
        try commit(m, &state, b, tokens[at..][0..rows], rows);
        at += rows;
    }
    return out;
}
