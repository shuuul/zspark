//! DFlash2's own kernels on the device (`zig/kernels/common/dflash2.cu`): the dynamic convolution over a
//! block's rows, a row's q, k and v normed, rotated and put where the context keeps them, the block's
//! attention over the context, and the head's picks dotted again against the stored rows.
//! `kernels_ref.zig` has the same steps on the host. C ABI names stay `zs_dflash2_*` (the first
//! consumer's); the kernels are shared.
//!
//! WS-03's Qwen drafter imports `Place`, `dconv`, `prep`, `attend`, `attendParts`, `top`, `rescore` and `head`.

const cuda = @import("../../cuda/cuda.zig");
const host = @import("kernels_ref.zig");

pub const head = host.head;

/// Where a launch's first row stands, and how the context stores a position.
pub const Place = struct {
    /// int32 [1], read on the device: a captured graph holds no position.
    pos: cuda.View,
    /// Position p is the context's row p mod its capacity; flat, it is row p and the caller keeps every
    /// position of a launch under the capacity.
    ring: bool,
};

/// x bf16 [rows, dims], dyn bf16 [rows, 4 * groups] (`kernel_projection`'s output: a row's branch, tap,
/// group), base bf16 [2, 2, dims] -> out bf16 [rows, dims]: row r mixes rows r and r - 1 with the branch's two
/// taps, and takes `residual` bf16 [rows, dims] when there is one. `out` may be the residual, not x.
pub fn dconv(x: cuda.View, dyn: cuda.View, base: cuda.View, branch: u1, residual: ?cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_dflash2_dconv(&x, &dyn, &base, if (residual) |*res| res else null, &out, branch));
}

/// A row's heads at position pos + r: q_rows bf16 [rows, heads * 128], its query heads (null for context rows,
/// and `q` with it), and kv_rows bf16 [rows, 2 * kv * 128], its key heads, then its value heads. q bf16 [heads,
/// rows or more, 128] takes the queries normed with `q_norm` and rotated; k and v bf16 [kv, capacity, 128] the
/// keys normed with `k_norm` and rotated and the values as they are, at the position's row.
pub fn prep(q_rows: ?cuda.View, kv_rows: cuda.View, q_norm: cuda.View, k_norm: cuda.View, at: Place, q: ?cuda.View, k: cuda.View, v: cuda.View, eps: f32, theta: f64) cuda.Error!void {
    return cuda.launched(zs_dflash2_prep(if (q_rows) |*view| view else null, &kv_rows, &q_norm, &k_norm, &at.pos, if (q) |*view| view else null, &k, &v, eps, theta, @intFromBool(at.ring)));
}

/// q bf16 [heads, rows, 128], the block at positions pos .. pos + rows - 1, over k and v bf16 [kv, capacity,
/// 128] -> out bf16 [rows, heads * 128]. A query sees the context `window` positions back and the whole block
/// (to its own row when `causal`).
/// `parts` is scratch, fp32 of `attendParts` values. An online softmax on the tensor cores
/// (`zig/kernels/common/dflash2_attend.cu`): the same bits wherever the context's rows lie, and near `host.attendRow`.
pub fn attend(q: cuda.View, k: cuda.View, v: cuda.View, at: Place, out: cuda.View, parts: cuda.View, window: usize, causal: bool) cuda.Error!void {
    return cuda.launched(zs_dflash2_attend(&q, &k, &v, &at.pos, &out, &parts, @intCast(window), @intFromBool(causal), @intFromBool(at.ring)));
}

/// The values `attend`'s scratch holds for `heads` query heads, `rows` rows and a window.
pub fn attendParts(heads: usize, rows: usize, window: usize) usize {
    return @intCast(zs_dflash2_attend_parts(@intCast(heads), @intCast(rows), @intCast(window)));
}

/// A row's `values.size[1]` largest of logits bf16 [rows, tokens] -> values fp32 and tokens int32 [rows, top],
/// largest first and the lower token among equals, each token `offset` on: `chain.largest` on the device.
/// `scratch` is int32 of `topScratch` values (a row split in slices, spec 143 WS-09), or null where that is 0.
pub fn top(logits: cuda.View, values: cuda.View, tokens: cuda.View, scratch: ?cuda.View, offset: usize) cuda.Error!void {
    return cuda.launched(zs_dflash2_top(&logits, &values, &tokens, if (scratch) |s| &s else null, @intCast(offset)));
}

/// The int32 values `top`'s scratch holds for `rows` rows of `cols` logits and `count` picks a row.
pub fn topScratch(rows: usize, cols: usize, count: usize) usize {
    return @intCast(zs_dflash2_top_scratch(@intCast(rows), @intCast(cols), @intCast(count)));
}

/// The head's picks dotted again against the stored rows: tokens int32 [rows, top] of whole-vocabulary ids
/// (`top`'s), rows bf16 [half, dims] the rank's rows of the vocabulary as the main checkpoint stores them,
/// final bf16 [rows, dims] the normed draft rows, `offset` the rank's first token -> values fp32 [rows, top],
/// each pick's dot with its row, one `fmaf` a value: `host.rescore` on the device.
pub fn rescore(tokens: cuda.View, rows: cuda.View, final: cuda.View, values: cuda.View, offset: usize) cuda.Error!void {
    return cuda.launched(zs_dflash2_rescore(&tokens, &rows, &final, &values, @intCast(offset)));
}

extern fn zs_dflash2_dconv(x: *const cuda.View, dyn: *const cuda.View, base: *const cuda.View, res: ?*const cuda.View, out: *const cuda.View, branch: i64) c_int;
extern fn zs_dflash2_prep(q_rows: ?*const cuda.View, kv_rows: *const cuda.View, q_norm: *const cuda.View, k_norm: *const cuda.View, pos: *const cuda.View, q: ?*const cuda.View, k: *const cuda.View, v: *const cuda.View, eps: f64, theta: f64, ring: i64) c_int;
extern fn zs_dflash2_attend(q: *const cuda.View, k: *const cuda.View, v: *const cuda.View, pos: *const cuda.View, out: *const cuda.View, parts: *const cuda.View, window: i64, causal: i64, ring: i64) c_int;
extern fn zs_dflash2_attend_parts(heads: i64, rows: i64, window: i64) i64;
extern fn zs_dflash2_top(logits: *const cuda.View, values: *const cuda.View, tokens: *const cuda.View, scratch: ?*const cuda.View, offset: i64) c_int;
extern fn zs_dflash2_top_scratch(rows: i64, cols: i64, top: i64) i64;
extern fn zs_dflash2_rescore(tokens: *const cuda.View, rows: *const cuda.View, final: *const cuda.View, values: *const cuda.View, offset: i64) c_int;
