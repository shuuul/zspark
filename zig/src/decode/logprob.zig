//! A reply's log probabilities (OpenAI `logprobs`, spec 101 WS-02): each chosen token's under the model's raw
//! distribution and its row's best tokens, found on the device from the row of logits the token was chosen from
//! (`zig/kernels/common/logprob.cu`, TensorFold 1.0.12's row kernel), as TensorFold's lanes find them. The
//! distribution is the raw one, as TensorFold's and vLLM's by default (`logprobs_mode` `raw_logprobs`); SGLang's
//! sampled path takes the logits over the temperature first.
//!
//! A row's words are a function of its logits alone, so its log probabilities keep the row rule as its token
//! does: the same bits from a serial step, a verify window, a prompt's last window or a round of several streams.
//! The chosen token's logit is read on the host from the row it was chosen from, and its log probability is
//! that minus the row's log-sum-exp in fp32, as TensorFold's `Row.fromWords` subtracts.

const std = @import("std");
const cuda = @import("../cuda/cuda.zig");
const numeric = @import("../weights/numeric.zig");

/// A reply's token with its log probability and its row's best tokens (OpenAI `logprobs`, spec 101 WS-02), as
/// TensorFold's `lanes.LogprobRow` holds it: under the model's raw distribution (its logits as the read-out
/// wrote them, before the sampling's temperature and filters and any constraint's mask), a log-sum-exp in fp32.
/// `server/served.zig` hands it to the host as that type (`server/request.zig`'s `LogprobRow`).
pub const Row = struct {
    /// OpenAI's `top_logprobs` limit.
    pub const most = 20;

    token: u32,
    logprob: f32,
    /// Of `ids` and `logprobs`: the request's `top_logprobs`.
    count: u8 = 0,
    /// Logit down, the lower id among equals.
    ids: [most]u32 = undefined,
    logprobs: [most]f32 = undefined,
};

/// The words a row's launch writes (TensorFold's `LP_WORDS`): [0] the log-sum-exp, [2 + j] and [22 + j] the
/// j-th best id and its logit.
pub const words = 3 + 2 * Row.most;

extern fn zs_logprob_rows(logits: *const cuda.View, k: i64, out: *const cuda.View) c_int;

/// `logits` bf16 [rows, V] -> `out` u32 [rows, `words`]: each row's log-sum-exp and its `k` best, as above.
pub fn rows(logits: cuda.View, k: u8, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_logprob_rows(&logits, k, &out));
}

/// What a reply that asks for log probabilities is lent: the device's words for a window's rows and their copy
/// on the host. Made once with its decoder, for the most rows a window scores.
pub const Scratch = struct {
    /// u32 [most rows, `words`].
    out: cuda.Buffer,
    host: []u32,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, most_rows: usize) !Scratch {
        var out: cuda.Buffer = try .init(.u32, &.{ most_rows, words });
        errdefer out.deinit();
        return .{ .out = out, .host = try gpa.alloc(u32, most_rows * words), .gpa = gpa };
    }

    pub fn deinit(self: *Scratch) void {
        self.out.deinit();
        self.gpa.free(self.host);
    }

    pub fn bytesOf(most_rows: usize) usize {
        return cuda.Buffer.bytesOf(.u32, &.{ most_rows, words });
    }

    /// Launches the words of `logits`' rows, bf16 [n, V], n at most the scratch's rows.
    pub fn launch(self: *Scratch, logits: cuda.View, k: u8) cuda.Error!void {
        std.debug.assert(logits.size[0] <= self.out.view.size[0]);
        return rows(logits, k, self.out.view);
    }

    /// Waits for the launch and copies its first `n` rows' words to the host.
    pub fn read(self: *Scratch, n: usize) cuda.Error!void {
        try cuda.synchronize();
        return self.out.downloadAt(0, std.mem.sliceAsBytes(self.host[0 .. n * words]));
    }

    /// Row `i` of the words read, for `token`, whose logit is `logit` (bf16 bits, the row's own), with `k` best.
    pub fn row(self: *const Scratch, i: usize, token: i32, logit: u16, k: u8) Row {
        return of(self.host[i * words ..][0..words], token, logit, k);
    }
};

/// A row from its words (TensorFold's `Row.fromWords`, the pick's logit given).
pub fn of(w: *const [words]u32, token: i32, logit: u16, k: u8) Row {
    std.debug.assert(k <= Row.most);
    const lse: f32 = @bitCast(w[0]);
    var r: Row = .{ .token = @intCast(token), .logprob = numeric.f32FromBf16(logit) - lse, .count = k };
    for (0..k) |j| {
        r.ids[j] = w[2 + j];
        r.logprobs[j] = @as(f32, @bitCast(w[2 + Row.most + j])) - lse;
    }
    return r;
}
