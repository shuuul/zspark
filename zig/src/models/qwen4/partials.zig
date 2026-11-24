//! A rank's fp32 partial and the sum of the ranks' (spec 050). `roundSum` is the branch: rank 0 first, one
//! rounding to bf16, then hc write-back mode 1. `add` is the MoE: the same order, left fp32, then mode 3.
//! `fold` is one rank's slots by weight, mode 2's fma chain without its rounding.

const cuda = @import("../../cuda/cuda.zig");

/// gathered fp32 [ranks, rows, dims] -> into bf16 [rows, dims].
pub fn roundSum(gathered: cuda.View, into: cuda.View) cuda.Error!void {
    return cuda.launched(zs_qwen4_partial_add(&gathered, &into, 1));
}

/// gathered fp32 [ranks, rows, dims] -> into fp32 [rows, dims].
pub fn add(gathered: cuda.View, into: cuda.View) cuda.Error!void {
    return cuda.launched(zs_qwen4_partial_add(&gathered, &into, 0));
}

/// y fp32 [rows * slots, dims], shared fp32 [rows, dims], weights fp32 [rows, slots] -> out fp32 [rows, dims].
pub fn fold(y: cuda.View, shared: cuda.View, weights: cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_qwen4_moe_fold(&y, &shared, &weights, &out));
}

extern fn zs_qwen4_partial_add(gathered: *const cuda.View, into: *const cuda.View, to_bf16: i64) c_int;
extern fn zs_qwen4_moe_fold(y: *const cuda.View, shared: *const cuda.View, weights: *const cuda.View, out: *const cuda.View) c_int;
