//! A long clip's pieces decoded at once (spec 142 WS-06), as vLLM runs the pieces it cuts a transcription
//! into (`audio/split.zig`): each piece is a request of its own, with its own prompt and no other piece's
//! context, and they run together. Here a piece takes a lane, a sequence's caches of its own (`Lanes`): its
//! prompt runs alone through the forward's windows (`decode/draft.zig`'s `promptWindow`), then every lane
//! holding a reply steps in one window, a row a lane (`forward.steps`), so the weights are read once a step for
//! all of them. A finished piece frees its lane for the next. Each row's attention reads only its lane's
//! caches and the rest is row by row, so a piece's tokens are those it decodes alone (`greedy`'s), whatever
//! runs beside it: `forward_test.zig`'s "pieces" test holds that.
//!
//! A step is one captured graph (`forward.StepGraphs`) over every lane, a free lane's row computed and dropped,
//! so the graph never changes as pieces end; it holds two attention launches a lane a layer beside the shared
//! ones. The prompts run one piece at a time, eagerly, between steps.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const placeholders = @import("../../decode/placeholders.zig");
const sampling = @import("../../decode/sampling.zig");
const shared = @import("../../decode/draft.zig");
const config_file = @import("config.zig");
const draft = @import("draft.zig");
const forward_file = @import("forward.zig");

const State = forward_file.State;

/// The sequences pieces decode in, `count` of `positions` each.
pub const Lanes = struct {
    states: []State,
    gpa: Allocator,

    pub fn init(gpa: Allocator, config: config_file.Config, count: usize, positions: usize) !Lanes {
        const states = try gpa.alloc(State, count);
        errdefer gpa.free(states);
        var made: usize = 0;
        errdefer for (states[0..made]) |*s| s.deinit();
        for (states) |*s| {
            s.* = try .init(gpa, config, positions);
            made += 1;
        }
        return .{ .states = states, .gpa = gpa };
    }

    pub fn deinit(self: *Lanes) void {
        for (self.states) |*s| s.deinit();
        self.gpa.free(self.states);
        self.* = undefined;
    }

    pub fn capacity(self: Lanes) usize {
        return self.states[0].capacity;
    }
};

/// One piece: its prompt with the tower's embeddings over its `<|audio_pad|>` rows, at most `limit` reply tokens.
pub const Piece = struct {
    prompt: []const i32,
    audio: placeholders.Placeholders,
    limit: usize,
    /// The reply's tokens, an end token among them.
    reply: std.ArrayList(u32) = .empty,

    fn ended(self: Piece, config: config_file.Config) bool {
        const n = self.reply.items.len;
        return n >= self.limit or (n > 0 and config.isEos(self.reply.items[n - 1]));
    }
};

/// Every piece's greedy reply, pieces taking free lanes in order. `b.scored` must hold a row a lane. `timing`'s
/// `first` and `on_first` mark the first piece's first logits; its `stop` ends every reply before its next step
/// and leaves the pieces not begun empty.
pub fn greedy(gpa: Allocator, m: forward_file.Model, lanes: Lanes, b: *forward_file.Buffers, pieces: []Piece, timing: ?*draft.Timing) !void {
    const count = @min(lanes.states.len, b.scored);
    std.debug.assert(count >= 1);
    const vocab = m.config.vocab;
    const logits = try gpa.alloc(u16, count * vocab);
    defer gpa.free(logits);
    // the piece each lane holds; a free lane null
    const held = try gpa.alloc(?usize, count);
    defer gpa.free(held);
    @memset(held, null);
    // every lane steps, a free one on a token nobody reads and nothing committed, so one graph serves all steps
    const states = try gpa.alloc(*State, count);
    defer gpa.free(states);
    for (states, lanes.states[0..count]) |*s, *lane| s.* = lane;
    var many: forward_file.Many = .{ .states = states };
    var graphs: forward_file.StepGraphs = try .init(gpa);
    defer graphs.deinit();
    const tokens = try gpa.alloc(i32, count);
    defer gpa.free(tokens);
    var next: usize = 0;
    var first = true;
    while (true) {
        const stopped = if (timing) |t| if (t.stop) |s| s.load(.acquire) else false else false;
        if (!stopped) for (held, 0..) |*h, lane| {
            if (h.* != null or next == pieces.len) continue;
            const p = &pieces[next];
            next += 1;
            const state = &lanes.states[lane];
            try state.reset();
            state.audio = p.audio;
            defer state.audio = null;
            while (state.pos < p.prompt.len) try shared.promptWindow(draft.namespace, m, state, b, p.prompt);
            try forward_file.logitsRow(b.*, 0, logits[0..vocab]);
            if (first) if (timing) |t| {
                first = false;
                t.first = cuda.monotonic();
                if (t.on_first) |hook| hook.call(hook.ctx);
            };
            try p.reply.append(gpa, @intCast(try sampling.argmax(logits[0..vocab])));
            h.* = if (p.ended(m.config)) null else next - 1;
            if (h.* == null) try state.reset();
        };
        var live: usize = 0;
        for (held, tokens) |h, *token| {
            token.* = if (h) |piece| @intCast(pieces[piece].reply.getLast()) else 0;
            live += @intFromBool(h != null);
        }
        if (live == 0) {
            if (next == pieces.len or stopped) return;
            continue;
        }
        if (stopped) return;
        try graphs.forward(m, &many, b, tokens);
        try cuda.download(b.logits.rows(count), std.mem.sliceAsBytes(logits));
        for (held, states, tokens, 0..) |*h, state, token, j| {
            const piece = h.* orelse continue;
            const step = [_]i32{token};
            try forward_file.commit(m, state, b.*, &step, 1);
            const p = &pieces[piece];
            try p.reply.append(gpa, @intCast(try sampling.argmax(logits[j * vocab ..][0..vocab])));
            if (p.ended(m.config)) {
                h.* = null;
                try state.reset();
            }
        }
    }
}
