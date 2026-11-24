//! What `compare_vllm_test.zig` and `served_test.zig` share: a whole model's decoding on one window's buffers
//! (`Bench`), drafted and forced along another's reply, and where a token stands in a row.

const std = @import("std");
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const compare = @import("../../../src/models/qwen4/compare.zig");
const forward = @import("../../../src/models/qwen4/forward.zig");
const draft = @import("../../../src/models/qwen4/draft.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const mtp = @import("../../../src/models/qwen4/mtp.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const Allocator = std.mem.Allocator;

pub fn commonPrefix(a: []const i32, b: []const i32) usize {
    var n: usize = 0;
    while (n < @min(a.len, b.len) and a[n] == b[n]) n += 1;
    return n;
}

/// Where a token stands in a row the sampler chose from.
pub const Standing = struct {
    logit: f64,
    /// The scaled logit plus the key's Gumbel noise: among the tokens the cuts keep, the largest wins.
    score: f64,
    /// Tokens ahead of it in the sampler's order; top-k keeps ranks below `top_k`.
    rank: usize,
    /// The probability, over the top-k at the temperature, of the tokens ahead of it: top-p drops a token once
    /// this reaches `top_p`.
    ahead: f64,
};

pub fn standing(logits: []const u16, token: i32, position: u64, s: sampling.Sampling) Standing {
    const id: usize = @intCast(token);
    const mine: f64 = numeric.f32FromBf16(logits[id]);
    // a greedy row's probabilities are read at temperature 1
    const temperature = if (s.greedy()) 1 else @max(s.temperature, 1e-6);
    // the row's largest logits, descending
    var best: [64]f64 = @splat(-std.math.inf(f64));
    const kept: usize = if (s.top_k == 0) best.len else @min(s.top_k, best.len);
    var rank: usize = 0;
    for (logits, 0..) |bits, other| {
        const v: f64 = numeric.f32FromBf16(bits);
        if (v > mine or (v == mine and other < id)) rank += 1;
        if (!(v > best[kept - 1])) continue;
        var i = kept - 1;
        while (i > 0 and best[i - 1] < v) : (i -= 1) best[i] = best[i - 1];
        best[i] = v;
    }
    var total: f64 = 0;
    var ahead: f64 = 0;
    for (best[0..kept], 0..) |v, i| {
        const p = @exp((v - best[0]) / temperature);
        total += p;
        if (i < rank) ahead += p;
    }
    // greedy decoding draws nothing: the logit decides
    const noise = if (s.greedy()) 0 else -@log(-@log(sampling.uniform(s.seed, position, id)));
    return .{ .logit = mine, .score = if (s.greedy()) mine else mine / temperature + noise, .rank = rank, .ahead = ahead / total };
}

/// What the whole-model test's decoding shares: the model, one window's buffers, the sampler and a row of
/// logits on the host. Each reply takes a state and graphs of its own.
pub const Bench = struct {
    m: model.Model,
    b: *forward.Buffers,
    sampler: *sampling.Sampler,
    logits: []u16,
    /// fp32 [1, vocab]: a row's logits before their bf16 rounding, asked for where a choice differs.
    wide: *cuda.Buffer,
    gpa: Allocator,
    /// An image prompt's (spec 019): attached to each reply's state, and to its head.
    image: ?*const compare.ReferenceImage = null,

    pub const capacity = 4096;

    /// The head again on row `row` of the window just run, unrounded: `tokens`' logits as fp32. Two logits that
    /// tie as bf16 are told apart here, and how near each is to the edge of its bf16 step.
    pub fn unrounded(self: Bench, row: usize, tokens: [2]i32) ![2]f32 {
        try self.m.head.forward(self.b.mixed.span(row, 1), self.wide.view);
        try cuda.synchronize();
        var out: [2]f32 = undefined;
        for (&out, tokens) |*v, token| try self.wide.downloadAt(@as(usize, @intCast(token)) * 4, std.mem.asBytes(v));
        return out;
    }

    /// `out.len` tokens after `prompt` in rounds the MTP head drafts for, as `zspark chat` decodes by default
    /// (serial rounds where the checkpoint has no head).
    pub fn drafted(self: Bench, prompt: []const i32, out: []i32, settings: sampling.Sampling) !draft.Stats {
        var state: forward.State = try .init(self.gpa, self.m, self.b.rows, capacity);
        defer state.deinit();
        var graphs: forward.Graphs = try .init(self.gpa);
        defer graphs.deinit();
        var head: ?mtp.Head = if (self.m.mtp != null) try .init(self.gpa, self.m, self.sampler, settings, .{ .rows = self.b.rows, .capacity = capacity }) else null;
        defer if (head) |*h| h.deinit();
        if (self.image) |image| {
            try image.attach(&state);
            if (head) |*h| try h.attach(state.rope);
        }
        const drafter: ?draft.Drafter = if (head) |*h| h.drafter() else null;
        var decoder: draft.Decoder = .{ .m = self.m, .state = &state, .b = self.b, .graphs = &graphs, .sampler = self.sampler, .settings = settings, .eos = &.{}, .logits = self.logits };
        var chosen: [draft.forward_max_rows]i32 = undefined;
        var drafts: [draft.forward_max_rows]i32 = undefined;
        var kept = try decoder.prompt(drafter, prompt, &chosen);
        out[0] = chosen[kept - 1];
        var made: usize = 1;
        while (made < out.len) {
            const n = if (drafter) |d| try decoder.draft(d, chosen[0..kept], @min(6, out.len - made - 1), &drafts) else 0;
            kept = try decoder.round(chosen[kept - 1], drafts[0..n], &chosen);
            @memcpy(out[made..][0..kept], chosen[0..kept]);
            made += kept;
        }
        return decoder.stats;
    }

    pub const Forced = struct {
        /// Tokens of `theirs` that were zspark's choice too.
        agree: usize = 0,
        /// The first that was not, and zspark's choice there: where its own reply leaves theirs.
        first: ?usize = null,
        mine: i32 = 0,
    };

    /// zspark's choice at every position of the reply `theirs`, each from its own row after their tokens up to
    /// there (one-row steps through graphs), so a close call later in the reply is seen too. Prints both
    /// tokens' standing in zspark's row wherever the two differ.
    pub fn forced(self: Bench, label: []const u8, prompt: []const i32, theirs: []const i32, settings: sampling.Sampling) !Forced {
        var state: forward.State = try .init(self.gpa, self.m, self.b.rows, capacity);
        defer state.deinit();
        var graphs: forward.Graphs = try .init(self.gpa);
        defer graphs.deinit();
        if (self.image) |image| try image.attach(&state);
        try forward.forward(self.m, &state, self.b, prompt);
        try forward.logitsRow(self.b.*, prompt.len - 1, self.logits);
        try forward.commit(self.m, &state, self.b.*, prompt, prompt.len);
        var result: Forced = .{};
        for (theirs, 0..) |want, i| {
            // the key's position is the place in the sequence of the token being chosen, as TensorFold's is
            const position = prompt.len + i;
            const token = try self.sampler.choose(self.logits, position, settings);
            if (token == want) {
                result.agree += 1;
            } else {
                const a = standing(self.logits, token, position, settings);
                const t = standing(self.logits, want, position, settings);
                // the prompt's last row for the first token, the step's one row after
                const wide = try self.unrounded(if (i == 0) prompt.len - 1 else 0, .{ token, want });
                std.debug.print("{s}: token {d} (position {d}): {d} (logit {d:.4}, rank {d}, {d:.4} ahead, score {d:.4}), theirs {d} (logit {d:.4}, rank {d}, {d:.4} ahead, score {d:.4}); before the bf16 rounding {d:.5} and {d:.5}\n", .{ label, i, position, token, a.logit, a.rank, a.ahead, a.score, want, t.logit, t.rank, t.ahead, t.score, wide[0], wide[1] });
                if (result.first == null) {
                    result.first = i;
                    result.mine = token;
                }
            }
            if (i + 1 == theirs.len) break;
            const step = [1]i32{want};
            try graphs.forward(self.m, &state, self.b, &step);
            try forward.logitsRow(self.b.*, 0, self.logits);
            try forward.commit(self.m, &state, self.b.*, &step, 1);
        }
        return result;
    }
};
