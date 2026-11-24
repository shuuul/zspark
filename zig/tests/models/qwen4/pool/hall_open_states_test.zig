//! `hall_open.Fit` with the states' budget (spec 101, the owner's 2026-10-16: the load's check counts the
//! states, as vLLM reserves its cache at the load): the budget is held out of what is left at both of the
//! hand-over's steps, to the byte, and what a refusal names in its place fits under the same budget.

const std = @import("std");
const fixtures = @import("../testing.zig");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const extents = @import("../../../../src/decode/extents.zig");
const hall_file = @import("../../../../src/models/qwen4/pool/hall.zig");
const hall_open = @import("../../../../src/models/qwen4/pool/hall_open.zig");
const Fit = hall_open.Fit;
const Rest = hall_open.Rest;
const extentsBytes = hall_open.extentsBytes;

test "the load's check holds the states' budget out of what is left, at the extents' step and at the rest's, to the byte" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    const context: usize = 8192;
    const pool = extentsBytes(m, extents.wanted(context, 4));
    const rest = Rest.of(m, 4, context).bytes();
    const states = hall_file.default_states_bytes;
    // no session to give back: the extents, the rest and the states, the reserve kept
    const both = Fit.of(m, context, 4, pool + rest + states + cuda.reserve, 0, states);
    try testing.expect(both.fits and both.extents_fit);
    try testing.expectEqual(states, both.states);
    try testing.expectEqual(pool + rest + states + cuda.reserve, both.left);
    // one byte less refuses what the check without the states passed
    const short = Fit.of(m, context, 4, pool + rest + states + cuda.reserve - 1, 0, states);
    try testing.expect(!short.fits and short.extents_fit);
    try testing.expect(Fit.of(m, context, 4, short.left, 0, 0).fits);
    // a session of the rest or more gives the rest back: the extents and the states decide
    try testing.expect(Fit.of(m, context, 4, pool + states + cuda.reserve, rest, states).fits);
    const extents_short = Fit.of(m, context, 4, pool + states + cuda.reserve - 1, 1 << 40, states);
    try testing.expect(!extents_short.fits and !extents_short.extents_fit);
    try testing.expect(Fit.of(m, context, 4, extents_short.left, 1 << 40, 0).extents_fit);
    // a budget wider than what is left: nothing but the session
    const none = Fit.of(m, context, 4, states, 0, states + 1);
    try testing.expect(!none.fits and !none.extents_fit);
    try testing.expectEqual(@as(usize, 1), none.parallel);
    try testing.expectEqual(@as(usize, 0), none.context);
}

test "what a refusal names under a states' budget fits under it, and the next above it does not" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    const context: usize = 8192;
    const pool = extentsBytes(m, extents.wanted(context, 4));
    const rest = Rest.of(m, 4, context).bytes();
    const half = rest / 2;
    const states = (pool + rest) / 3;
    var left = states + cuda.reserve;
    while (left < pool + (rest - half) + states + cuda.reserve) : (left += (pool + rest) / 37 + 1) {
        const fit = Fit.of(m, context, 4, left, half, states);
        try testing.expect(!fit.fits);
        // the same as the check without the budget, asked of what the budget leaves
        const plain = Fit.of(m, context, 4, left - states, half, 0);
        try testing.expectEqual(plain.parallel, fit.parallel);
        try testing.expectEqual(plain.context, fit.context);
        try testing.expectEqual(plain.extents_fit, fit.extents_fit);
        if (fit.parallel > 1) try testing.expect(Fit.of(m, context, fit.parallel, left, half, states).fits);
        try testing.expect(fit.parallel < 4 and !Fit.of(m, context, fit.parallel + 1, left, half, states).fits);
        if (fit.context > 0) try testing.expect(Fit.of(m, fit.context, 4, left, half, states).fits);
        try testing.expect(fit.context < context and !Fit.of(m, fit.context + extents.positions, 4, left, half, states).fits);
    }
}

test "the widest states' budget a pool leaves fits beside it to the byte, and one byte more does not" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var m = try fixtures.random_model.testModelRows(gpa, arena_state.allocator(), null, 48);
    defer m.deinit();
    const context: usize = 8192;
    const pool = extentsBytes(m, extents.wanted(context, 4));
    const rest = Rest.of(m, 4, context).bytes();
    // the rest's step decides with no session, the extents' with a session of the rest or more
    for ([_]usize{ 0, rest / 2, rest, 1 << 40 }) |session| {
        const left = pool + rest + cuda.reserve + (3 << 30);
        const widest = Fit.statesWithin(m, context, 4, left, session);
        try testing.expect(widest >= 3 << 30);
        try testing.expect(Fit.of(m, context, 4, left, session, widest).fits);
        try testing.expect(!Fit.of(m, context, 4, left, session, widest + 1).fits);
    }
    try testing.expectEqual(@as(usize, 3 << 30), Fit.statesWithin(m, context, 4, pool + rest + cuda.reserve + (3 << 30), 0));
    // a pool that does not fit alone leaves no budget
    try testing.expectEqual(@as(usize, 0), Fit.statesWithin(m, context, 4, pool + rest + cuda.reserve - 1, 0));
    try testing.expectEqual(@as(usize, 0), Fit.statesWithin(m, context, 4, pool + rest + cuda.reserve, 0));
}
