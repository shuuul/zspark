//! Launches of `zig/kernels/glm5/index.cu`. The arithmetic is `index.zig`'s: a window is the serial steps, and
//! a row alone is that row in the window. Keys and gates are addressed through the extent table; the pooled
//! keys are contiguous, as e4m3 codes and a scale a pool, and the queries are coded the same way before the score.

const cuda = @import("../../../cuda/cuda.zig");

/// Positions in one extent. A pool of four never straddles two.
pub const extent = 512;

pub fn write(k: *const cuda.View, gate: *const cuda.View, ln_w: *const cuda.View, ln_b: *const cuda.View, keys: *const cuda.View, gates: *const cuda.View, table: *const cuda.View, pos: *const cuda.View, rows: usize, context: usize) cuda.Error!void {
    try cuda.launched(zs_glm5_index_write(k, gate, ln_w, ln_b, keys, gates, table, pos, @intCast(rows), @intCast(context)));
}

/// The pools the window completes, into `codes` u8 [pools, D] and `scales` fp32 [pools].
pub fn pool(keys: *const cuda.View, gates: *const cuda.View, table: *const cuda.View, ape: *const cuda.View, codes: *const cuda.View, scales: *const cuda.View, pos: *const cuda.View, rows: usize, context: usize) cuda.Error!void {
    try cuda.launched(zs_glm5_index_pool(keys, gates, table, ape, codes, scales, pos, @intCast(rows), @intCast(context)));
}

/// qi bf16 [R, H, D] and weights fp32 [R, H] into q8 u8 [R, H, D] and wq fp32 [R, H].
pub fn query(qi: *const cuda.View, weights: *const cuda.View, q8: *const cuda.View, wq: *const cuda.View) cuda.Error!void {
    try cuda.launched(zs_glm5_index_query(qi, weights, q8, wq));
}

pub fn score(q8: *const cuda.View, wq: *const cuda.View, codes: *const cuda.View, scales: *const cuda.View, pos: *const cuda.View, scores: *const cuda.View, context: usize) cuda.Error!void {
    try cuda.launched(zs_glm5_index_score(q8, wq, codes, scales, pos, scores, @intCast(context)));
}

pub fn select(scores: *const cuda.View, pos: *const cuda.View, tokens: *const cuda.View, counts: *const cuda.View, kept: usize, context: usize) cuda.Error!void {
    try cuda.launched(zs_glm5_index_select(scores, pos, tokens, counts, @intCast(kept), @intCast(context)));
}

extern fn zs_glm5_index_write(k: *const cuda.View, gate: *const cuda.View, ln_w: *const cuda.View, ln_b: *const cuda.View, keys: *const cuda.View, gates: *const cuda.View, table: *const cuda.View, pos: *const cuda.View, rows: i64, context: i64) c_int;
extern fn zs_glm5_index_pool(keys: *const cuda.View, gates: *const cuda.View, table: *const cuda.View, ape: *const cuda.View, codes: *const cuda.View, scales: *const cuda.View, pos: *const cuda.View, rows: i64, context: i64) c_int;
extern fn zs_glm5_index_query(qi: *const cuda.View, weights: *const cuda.View, q8: *const cuda.View, wq: *const cuda.View) c_int;
extern fn zs_glm5_index_score(q8: *const cuda.View, wq: *const cuda.View, codes: *const cuda.View, scales: *const cuda.View, pos: *const cuda.View, scores: *const cuda.View, context: i64) c_int;
extern fn zs_glm5_index_select(scores: *const cuda.View, pos: *const cuda.View, tokens: *const cuda.View, counts: *const cuda.View, kept: i64, context: i64) c_int;
