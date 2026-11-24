//! Nemotron-3.5-Transcribe's host admission, cancellation and explicit thinking flags
//! (spec 151 WS-08). Real routes run in the GPU smoke.

const std = @import("std");
const testing = std.testing;
const api = @import("engine_api");
const served = @import("../../../src/models/nemotron/served.zig");
const serve_cmd = @import("../../../src/models/nemotron/commands/serve.zig");

/// An engine never started: admission, cancellation and status do not dereference GPU resources.
fn engineForTest(gpa: std.mem.Allocator) served.Serve {
    return .{
        .gpa = gpa,
        .io = testing.io,
        .model = undefined,
        .perception = undefined,
        .tok = undefined,
        .state = undefined,
        .buffers = undefined,
        .graphs = undefined,
        .sampler = undefined,
        .features = undefined,
        .locator = 131_072,
        .context = 1 << 20,
        .most_waiting = 2,
    };
}

const Events = struct {
    ids: [2]api.Id = undefined,
    reasons: [2]api.Reason = undefined,
    n: usize = 0,
    unexpected: bool = false,
};

fn sinkOf(seen: *Events) api.Sink {
    return .{ .ctx = seen, .event = struct {
        fn call(ctx: *anyopaque, id: api.Id, event: *const api.Event) void {
            const seen_: *Events = @ptrCast(@alignCast(ctx));
            switch (event.*) {
                .finished => |finished| {
                    if (seen_.n == seen_.ids.len) {
                        seen_.unexpected = true;
                        return;
                    }
                    seen_.ids[seen_.n] = id;
                    seen_.reasons[seen_.n] = finished.reason;
                    seen_.n += 1;
                },
                else => seen_.unexpected = true,
            }
        }
    }.call };
}

test "served: requests queue to most_waiting, a further one is Busy; cancel and close leave quietly" {
    var e = engineForTest(testing.allocator);
    const gpa = testing.allocator;
    var seen: Events = .{};
    const sink = sinkOf(&seen);
    var result: api.TranscriptionResult = .{};
    defer result.deinit(gpa);
    var r: api.TranscriptionRequest = .{ .audio = "not audio", .max_tokens = 4, .result = &result, .gpa = gpa };
    const engine = e.engine();
    // nothing started: submissions queue behind a still-running slot
    try engine.vtable.submit_transcription.?(engine.ctx, 1, &r, sink);
    try engine.vtable.submit_transcription.?(engine.ctx, 2, &r, sink);
    try testing.expectError(error.Busy, engine.vtable.submit_transcription.?(engine.ctx, 3, &r, sink));
    var status: api.Status = .{};
    var tokens: [8]u32 = undefined;
    engine.vtable.status(engine.ctx, &status, &tokens);
    try testing.expectEqual(@as(u32, 0), status.running); // worker never started
    try testing.expectEqual(@as(u32, 2), status.waiting);
    // a waiting cancel is dequeued and finished at once, not parked until the queue drains
    engine.vtable.cancel(engine.ctx, 1);
    engine.vtable.status(engine.ctx, &status, &tokens);
    try testing.expectEqual(@as(u32, 1), status.waiting);
    engine.vtable.cancel(engine.ctx, 2);
    engine.vtable.status(engine.ctx, &status, &tokens);
    try testing.expectEqual(@as(u32, 0), status.waiting);
    e.close();
    try testing.expectError(error.Closed, engine.vtable.submit_transcription.?(engine.ctx, 4, &r, sink));
    try testing.expect(!seen.unexpected);
    try testing.expectEqualSlices(api.Id, &.{ 1, 2 }, seen.ids[0..seen.n]);
    try testing.expectEqualSlices(api.Reason, &.{ .cancelled, .cancelled }, seen.reasons[0..seen.n]);
}

test "served: thinkingAsked reads an explicit --thinking or --no-thinking, the default left out" {
    const asked = serve_cmd.thinkingAsked;
    try testing.expect(!asked(&.{ "--port", "8080" }));
    try testing.expect(asked(&.{"--thinking"}));
    try testing.expect(asked(&.{"--no-thinking"}));
    try testing.expect(asked(&.{ "--parallel", "2", "--thinking" }));
    try testing.expect(asked(&.{"--thinking=true"}));
    try testing.expect(!asked(&.{"--thinking-budget"}));
    try testing.expect(!asked(&.{"--thinking-reserve"}));
}
