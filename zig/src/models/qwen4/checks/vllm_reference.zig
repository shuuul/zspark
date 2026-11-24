//! A checkpoint no TensorFold reads, held to the vLLM container its card names (spec 105): the files
//! `tools/reference/vllm_reference.py` fetched (`NAME.json`, `score.json`), the cases and the bounds. One Spark's
//! whole-model test (`compare_vllm_test.zig`) and two ranks' (`ranks_vllm.zig`, `ranks-check DIR reference=REF`)
//! read them here, so both are held to the same numbers; the files' forms are `decode/vllm_reference.zig`'s. Host only.

const shared = @import("../../../decode/vllm_reference.zig");

/// A greedy reply vLLM gave twice (`NAME.json`), and the tokens of it zspark's reply begins with, at least.
pub const Case = shared.Case;

/// Measured 2026-10-18 (spec 105 WS-02, zspark 000c579 against the card's image `sha256:03a956544e9f…3595`):
/// `sky` and `primes` every token vLLM's; 256 tokens of `sky` vLLM's to token 100, where vLLM's first two are
/// an exact tie (1056 and 28750, both -0.8556) and zspark takes 28750. Since the routed experts fold their
/// scales into the weights (spec 107 WS-03) `sky` leaves vLLM's at token 38, 22687 here against 3404 (24.31994
/// and 24.29638 before the bf16 rounding), 63 of the 64 choices after vLLM's tokens zspark's own; `primes` at
/// token 19, past the answer, where the turn ends: 248045 here against vLLM's 248046 (16.13143 and 15.55509), the
/// made-up turns' rows that no implementation settles (`compare_test.zig`'s `published`); 256 tokens of `sky`
/// leave it at the same token 38, 253 of the 256 choices after vLLM's tokens zspark's own.
pub const cases = [_]Case{
    .{ .name = "sky-drafted", .least = 38 },
    .{ .name = "primes-drafted", .least = 19 },
    .{ .name = "sky-long-drafted", .least = 38 },
};

/// Where zspark's reply may leave vLLM's: where vLLM's own first two choices lie within this of each other
/// (its logprobs): twice the score text's mean |zspark - vLLM| a token, 0.195, measured as `cases`' notes say.
pub const tie = 0.39;

/// The score text's nll, zspark's against vLLM's first run, at most this apart: measured -0.0088.
pub const nll_apart = 0.02;

pub const Reply = shared.Reply;
pub const Scored = shared.Scored;
pub const reply = shared.reply;
pub const scored = shared.scored;
pub const logprobOf = shared.logprobOf;
pub const Sums = shared.Sums;
pub const holds = shared.holds;
pub const greedy = shared.greedy;
pub const sampled = shared.sampled;
