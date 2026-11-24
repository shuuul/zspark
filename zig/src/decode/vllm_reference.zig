//! The files `tools/reference/vllm_reference.py` fetches from a checkpoint's vLLM container (`NAME.json`,
//! `score.json`): a checkpoint no TensorFold reads is held to them (specs 105 and 107). Each model's checks keep
//! their own cases and bounds (`models/qwen4/checks/vllm_reference.zig`, `models/glm5/checks/ranks_vllm.zig`).
//! Host only.

const std = @import("std");
const numeric = @import("../weights/numeric.zig");
const sampling = @import("sampling.zig");

/// A greedy reply vLLM gave (`NAME.json`), and the tokens of it zspark's reply begins with, at least: each
/// model's checks give their own cases.
pub const Case = struct { name: []const u8, least: usize };

/// A case's two replies, as both models' checks run them: greedy, and sampled.
pub const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
pub const sampled: sampling.Sampling = .{ .seed = 11, .temperature = 0.8, .top_k = 40, .top_p = 0.95, .min_p = 0 };

/// Whether `reference_path` holds vLLM's files rather than TensorFold's export, which holds `exported`.
pub fn holds(io: std.Io, reference_path: []const u8, exported: []const u8) bool {
    var references = std.Io.Dir.cwd().openDir(io, reference_path, .{}) catch return false;
    defer references.close(io);
    references.access(io, exported, .{}) catch {
        references.access(io, "score.json", .{}) catch return false;
        return true;
    };
    return false;
}

/// `NAME.json`: the prompt, and each run's tokens with the top logprobs of each step.
pub const Reply = struct { prompt: []i32, runs: []const struct { tokens: []i32, steps: []const struct { top: std.json.ArrayHashMap(f64) } } };
/// `score.json`: the text, and each run's logprob of each position's token (none for the first).
pub const Scored = struct { prompt: []i32, runs: []const struct { positions: []const ?struct { logprob: ?f64 } } };

pub fn reply(arena: std.mem.Allocator, text: []const u8) !Reply {
    return std.json.parseFromSliceLeaky(Reply, arena, text, .{ .ignore_unknown_fields = true });
}

pub fn scored(arena: std.mem.Allocator, text: []const u8) !Scored {
    return std.json.parseFromSliceLeaky(Scored, arena, text, .{ .ignore_unknown_fields = true });
}

/// vLLM's logprob of `token` at a step of its first run, or null where it is not among the step's top.
pub fn logprobOf(theirs: Reply, step: usize, token: i32) ?f64 {
    var key_buffer: [16]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buffer, "{d}", .{token}) catch return null;
    return theirs.runs[0].steps[step].top.map.get(key);
}

/// The score text's sums: zspark's loss a token, vLLM's two runs', |zspark - vLLM| and |vLLM - vLLM|; zspark's
/// distance from the second run and from the runs' mean, printed beside (what of `apart` is vLLM's own noise).
pub const Sums = struct {
    zspark: f64 = 0,
    vllm: [2]f64 = @splat(0),
    apart: f64 = 0,
    theirs: f64 = 0,
    second: f64 = 0,
    mean: f64 = 0,
    /// sum of |vLLM - vLLM|^2: the standard error of `theirs` a token
    theirs_squares: f64 = 0,
    /// sums of zspark's loss less the runs' mean, and its square: the paired standard error of the nll's difference
    gap: f64 = 0,
    gap_squares: f64 = 0,
    count: usize = 0,

    /// The token at `next` given the bf16 row of logits before it.
    pub fn add(self: *Sums, s: Scored, logits: []const u16, next: usize) void {
        var highest = -std.math.inf(f32);
        for (logits) |bits| highest = @max(highest, numeric.f32FromBf16(bits));
        var total: f64 = 0;
        for (logits) |bits| total += @exp(@as(f64, numeric.f32FromBf16(bits) - highest));
        const loss = @log(total) - @as(f64, numeric.f32FromBf16(logits[@intCast(s.prompt[next])]) - highest);
        const v0 = -(s.runs[0].positions[next].?.logprob.?);
        const v1 = -(s.runs[1].positions[next].?.logprob.?);
        self.zspark += loss;
        self.vllm[0] += v0;
        self.vllm[1] += v1;
        self.apart += @abs(loss - v0);
        self.theirs += @abs(v1 - v0);
        self.theirs_squares += (v1 - v0) * (v1 - v0);
        self.second += @abs(loss - v1);
        self.mean += @abs(loss - (v0 + v1) / 2);
        self.gap += loss - (v0 + v1) / 2;
        self.gap_squares += (loss - (v0 + v1) / 2) * (loss - (v0 + v1) / 2);
        self.count += 1;
    }

    pub fn print(self: Sums, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const n: f64 = @floatFromInt(self.count);
        try w.print("score: {d} tokens; nll zspark {d:.5}, vLLM {d:.5} and {d:.5}; a token's |zspark - vLLM| {d:.4}, vLLM's between its runs {d:.4}; zspark's from the second {d:.4}, from their mean {d:.4}; zspark's nll less theirs {d:.5}, a standard error {d:.5}\n", .{ self.count, self.zspark / n, self.vllm[0] / n, self.vllm[1] / n, self.apart / n, self.theirs / n, self.second / n, self.mean / n, self.gap / n, @sqrt(@max(self.gap_squares / n - (self.gap / n) * (self.gap / n), 0) / n) });
    }

    /// Within `nll_apart` a token of vLLM's first run, and as near vLLM as vLLM is to itself.
    pub fn within(self: Sums, nll_apart: f64) bool {
        const n: f64 = @floatFromInt(self.count);
        return self.count > 0 and @abs(self.zspark - self.vllm[0]) / n <= nll_apart and self.apart <= self.theirs;
    }

    /// For a reference whose runs differ as much as quantized activations make them (GLM FP4-Local's W4A4): within
    /// `nll_apart` a token of the runs' mean, and zspark from either run no further than the runs from each other
    /// plus two standard errors of that mean, as a third run would be (one is further than the pair half the time).
    /// Fable's rule, spec 114 WS-06: zspark's distance from the runs' mean, 0.2040, is a third run's 0.2334 * sqrt(3/4).
    /// `across`, where zspark's form is not the reference's own kernel, is the least the runs' distance is taken as:
    /// vLLM's own runs in another form of at least the reference's precision are that far from it (spec 115).
    pub fn exchangeable(self: Sums, nll_apart: f64, across: f64) bool {
        if (self.count < 2) return false;
        const n: f64 = @floatFromInt(self.count);
        const most = self.bound(across) * n;
        return @abs(self.zspark - (self.vllm[0] + self.vllm[1]) / 2) / n <= nll_apart and self.apart <= most and self.second <= most;
    }

    /// The bound `exchangeable` holds zspark's distances to, a token.
    pub fn bound(self: Sums, across: f64) f64 {
        const n: f64 = @floatFromInt(self.count);
        const pair = self.theirs / n;
        return @max(pair, across) + 2 * @sqrt(@max(self.theirs_squares / n - pair * pair, 0) / n);
    }
};
