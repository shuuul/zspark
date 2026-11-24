//! Decoding Qwen3-ASR: `zig/src/decode/draft.zig`'s decoder over this model's forward (`namespace`). It has no
//! drafter (no chain): a reply is serial steps, a window of one row each, and a prompt is windows of the buffers'
//! rows. What a caller here names is that file's, instantiated once. Below, the serial greedy reply the command
//! and the tests run on the forward alone, as the package decodes (greedy, `max_new_tokens`).

const std = @import("std");
const cuda = @import("../../cuda/cuda.zig");
const forward_file = @import("forward.zig");
const sampling = @import("../../decode/sampling.zig");
const shared = @import("../../decode/draft.zig");

/// Qwen3-ASR as `shared.Decoder`'s `M`.
pub const namespace = struct {
    pub const Model = forward_file.Model;
    pub const State = forward_file.State;
    pub const Buffers = forward_file.Buffers;
    pub const Graphs = forward_file.Graphs;
    pub const Error = forward_file.Error;
    pub const forward = forward_file.forward;
    pub const forwardScoring = forward_file.forwardScoring;
    pub const commit = forward_file.commit;
    pub const logitsRow = forward_file.logitsRow;
    pub const logitRows = forward_file.logitRows;
    pub const streams = forward_file.streams;
};

pub const forward_max_rows = shared.forward_max_rows;
pub const Error = shared.Error(namespace);
pub const Chooser = shared.Chooser(namespace);
pub const Decoder = shared.Decoder(namespace);

/// When `greedy` had the prompt's logits: what follows is the reply's steps. `stop`, when set, ends the reply
/// before its next step (a client gone, `served.zig`); `on_first`, when set, is called at that moment (the
/// engine's `prefilled` event, which the host's /metrics reads as the first token's time). A chat reply (spec 150)
/// sets `on_token`, called with each token as it is chosen, the end token too; true ends the reply at it (a stop
/// string); and `eos`, the request's end tokens in place of the model's (empty under `ignore_eos`).
pub const Timing = struct {
    first: u64 = 0,
    stop: ?*const std.atomic.Value(bool) = null,
    on_first: ?Hook = null,
    on_token: ?TokenHook = null,
    eos: ?[]const u32 = null,

    pub const Hook = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque) void };
    pub const TokenHook = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque, token: u32) bool };

    fn ends(t: ?*const Timing, m: forward_file.Model, token: u32) bool {
        if (t) |one| if (one.eos) |eos| return std.mem.indexOfScalar(u32, eos, token) != null;
        return m.config.isEos(token);
    }
};

/// The serial greedy reply to `prompt` from an empty state: the prompt in windows of the buffers' rows, then a step
/// a token, each the largest logit, until an end token or `max` tokens. `logits` holds a row. The reply's tokens go
/// to `out`, the end token among them. The steps run as captured graphs (`forward.Graphs`), held for this reply:
/// one capture a bucket of context, the same bits as launching the kernels.
pub fn greedy(gpa: std.mem.Allocator, m: forward_file.Model, state: *forward_file.State, b: *forward_file.Buffers, prompt: []const i32, max: usize, logits: []u16, out: *std.ArrayList(u32), timing: ?*Timing) !void {
    std.debug.assert(state.pos == 0 and prompt.len >= 1);
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    while (state.pos < prompt.len) try shared.promptWindow(namespace, m, state, b, prompt);
    while (out.items.len < max) {
        try forward_file.logitsRow(b.*, 0, logits);
        if (out.items.len == 0) if (timing) |t| {
            t.first = cuda.monotonic();
            if (t.on_first) |h| h.call(h.ctx);
        };
        const token: u32 = @intCast(try sampling.argmax(logits));
        try out.append(gpa, token);
        if (timing) |t| if (t.on_token) |h| if (h.call(h.ctx, token)) break;
        if (Timing.ends(timing, m, token) or out.items.len == max) break;
        if (timing) |t| if (t.stop) |stop| if (stop.load(.acquire)) break;
        const step = [_]i32{@intCast(token)};
        try graphs.forward(m, state, b, &step);
        try forward_file.commit(m, state, b.*, &step, 1);
    }
}
