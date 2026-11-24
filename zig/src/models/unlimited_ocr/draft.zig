//! Decoding Unlimited-OCR: `zig/src/decode/draft.zig`'s decoder over this model's forward (`namespace`). It has
//! no drafter (no chain): a reply is serial steps, a window of one row each, and a prompt is windows of the
//! buffers' rows. What a caller here names is that file's, instantiated once. Below, the serial greedy reply
//! the command and the tests run on the forward alone.

const std = @import("std");
const cuda = @import("../../cuda/cuda.zig");
const forward_file = @import("forward.zig");
const requests = @import("requests.zig");
const sampling = @import("../../decode/sampling.zig");
const shared = @import("../../decode/draft.zig");

/// Unlimited-OCR as `shared.Decoder`'s `M`.
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
    pub const streams = forward_file.streams;

    /// Rows `row` to `row + n` of the window's logits, bf16 [n, vocab].
    pub fn logitRows(b: forward_file.Buffers, row: usize, n: usize) cuda.View {
        return b.logits.span(row, n);
    }
};

pub const forward_max_rows = shared.forward_max_rows;
pub const Error = shared.Error(namespace);
pub const Chooser = shared.Chooser(namespace);
pub const Decoder = shared.Decoder(namespace);

/// When `greedy` had the prompt's logits: what follows is the reply's steps.
pub const Timing = struct { first: u64 = 0 };

/// The serial greedy reply to `prompt` from an empty state: the prompt in windows of the buffers' rows, then a
/// step a token, each chosen by the largest logit after `rules` (`requests.Rules`: the n-gram rule over the
/// reply, then the PageCountGuard), until the config's end token or `max` tokens.
/// `logits` holds a row. The reply's tokens go to `out`, the end token among them. The steps are one captured
/// graph (`forward.Graphs`), captured while the prompt's last window runs: the same launches, the same bits.
pub fn greedy(gpa: std.mem.Allocator, m: forward_file.Model, state: *forward_file.State, b: *forward_file.Buffers, prompt: []const i32, max: usize, given: requests.Rules, logits: []u16, out: *std.ArrayList(u32), timing: ?*Timing) !void {
    std.debug.assert(state.pos == 0 and prompt.len >= 1);
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    while (state.pos < prompt.len) try shared.promptWindow(namespace, m, state, b, prompt);
    try graphs.prepare(m, state, b, 1, .last);
    var seen: std.ArrayList(u32) = .empty;
    defer seen.deinit(gpa);
    for (prompt) |id| try seen.append(gpa, @intCast(id));
    var rules = given;
    while (out.items.len < max) {
        try forward_file.logitsRow(b.*, 0, logits);
        if (out.items.len == 0) if (timing) |t| {
            t.first = cuda.monotonic();
        };
        rules.apply(seen.items, prompt.len, logits);
        const token: u32 = @intCast(try sampling.argmax(logits));
        try out.append(gpa, token);
        try seen.append(gpa, token);
        if (token == m.config.eos or out.items.len == max) break;
        const step = [_]i32{@intCast(token)};
        try graphs.forwardScoring(m, state, b, &step, .last);
        try forward_file.commit(m, state, b.*, &step, 1);
    }
}
