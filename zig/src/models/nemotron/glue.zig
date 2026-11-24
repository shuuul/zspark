//! The NemotronH backbone's glue launches (`zig/kernels/nemotron/`): the residual RMS norms, the router and
//! its top-k with the shared experts' slots, the serial round's feed, the routed experts' plan, a row's rest
//! add, and the MTP head's keyed greedy draw. TensorFold's kernels and bits; `glue_ref.zig` is their host twin.
//! Wrappers take views and return `cuda.launched`'s error.

const std = @import("std");
const cuda = @import("../../cuda/cuda.zig");

/// The router's K slices: the largest divisor of D / 64 up to 6, as TensorFold's `routerShape` picks it
/// (2688 -> 6 slices of 448). The slices fix the logits' sum order; PART is fp32 [SK, R, E].
pub fn routerSlices(d: usize) usize {
    std.debug.assert(d % 64 == 0);
    var g = @min(6, d / 64);
    while (g > 1) : (g -= 1) if ((d / 64) % g == 0) {
        break;
    };
    return g;
}

/// h = bf16(x + r) (x alone without r, and h then untouched), y = bf16(rmsnorm(h) * w), xs = y's fp32
/// 64-group sums ([rows, D / 64]; nothing reads them yet — the 4-bit matmuls they fed are not ported).
pub fn addRmsnorm(x: cuda.View, r: ?cuda.View, w: cuda.View, h: cuda.View, y: cuda.View, xs: cuda.View, eps: f64) cuda.Error!void {
    const h_ptr: ?*const cuda.View = if (r != null) &h else null;
    return cuda.launched(zs_nemotron_add_rmsnorm(&x, if (r) |*rr| rr else null, &w, h_ptr, &y, &xs, eps));
}

/// h = bf16(x + delta) where delta is the slots' sum in order (the first `nr` routed experts weighted by wt,
/// the rest the shared halves'), then addRmsnorm's tail. yk fp32 or bf16 [rows, ns, D], wt fp32 [rows, ns].
pub fn addMoeNorm(x: cuda.View, yk: cuda.View, wt: cuda.View, w: cuda.View, hn: cuda.View, y: cuda.View, xs: cuda.View, eps: f64, nr: usize) cuda.Error!void {
    return cuda.launched(zs_nemotron_add_moe_norm(&x, &yk, &wt, &w, &hn, &y, &xs, eps, @intCast(nr)));
}

/// The MTP block's input [rmsnorm(e) * we | rmsnorm(h) * wh] -> out bf16 [rows, 2 D], xs fp32 [rows, 2 D / 64].
pub fn concatNorms(e: cuda.View, hd: cuda.View, we: cuda.View, wh: cuda.View, out: cuda.View, xs: cuda.View, eps: f64) cuda.Error!void {
    return cuda.launched(zs_nemotron_concat_norms(&e, &hd, &we, &wh, &out, &xs, eps));
}

/// The Mamba blocks' group RMSNorm: x bf16 [rows, XD] over `groups` groups of XD / groups (a multiple of 64,
/// up to 1,024), out bf16 [rows, XD], xs fp32 [rows, XD / 64]; w bf16 [groups * GS].
pub fn groupRmsnorm(x: cuda.View, w: cuda.View, out: cuda.View, xs: cuda.View, eps: f64, groups: usize) cuda.Error!void {
    return cuda.launched(zs_nemotron_group_rmsnorm(&x, &w, &out, &xs, eps, @intCast(groups)));
}

/// The router's logits in K slices: x bf16 [rows, D], w bf16 [E, D] -> part fp32 [SK, rows, E].
pub fn router(x: cuda.View, w: cuda.View, part: cuda.View) cuda.Error!void {
    return cuda.launched(zs_nemotron_router(&x, &w, &part));
}

/// Each row's picks and weights from the slices: idx int32 and wt fp32 [rows, ns]; a row's `top` experts by
/// sigmoid score plus bias (the lower id on a tie), weights prob / (sum + 1e-20) * scaling when `norm`, then
/// ns - top shared slots (ids E.., weight 1). top and ns up to 8, E up to 256.
pub fn topk(part: cuda.View, bias: cuda.View, idx: cuda.View, wt: cuda.View, scaling: f64, top: usize, ns: usize, norm: bool) cuda.Error!void {
    return cuda.launched(zs_nemotron_topk(&part, &bias, &idx, &wt, scaling, @intCast(top), @intCast(ns), @intFromBool(norm)));
}

/// router then topk on one scratch `part` (fp32 [SK, rows, E]).
pub fn route(x: cuda.View, w: cuda.View, bias: cuda.View, part: cuda.View, idx: cuda.View, wt: cuda.View, scaling: f64, top: usize, ns: usize, norm: bool) cuda.Error!void {
    try router(x, w, part);
    return topk(part, bias, idx, wt, scaling, top, ns, norm);
}

/// One serial round's end on the device: ids[0] = sampled, history[pos + 1] = it, meta advances
/// {pos + 1, parity ^ 1, keep 1}.
pub fn serialFeed(sampled: cuda.View, ids: cuda.View, meta: cuda.View, history: cuda.View) cuda.Error!void {
    return cuda.launched(zs_nemotron_serial_feed(&sampled, &ids, &meta, &history));
}

/// The routed slots' plan: members (pick indices by expert, order kept), items triples {expert, first, count}
/// of at most `tile`, counts {items, experts used}. 16 rows of 8 slots, the `routed` first; experts up to 128.
pub fn planRouted(picks: cuda.View, members: cuda.View, items: cuda.View, counts: cuda.View, rows: usize, slots: usize, routed: usize, experts: usize, tile: usize) cuda.Error!void {
    return cuda.launched(zs_nemotron_plan_routed(&picks, &members, &items, &counts, @intCast(rows), @intCast(slots), @intCast(routed), @intCast(experts), @intCast(tile)));
}

/// y[row * row_mul + row_add] += r @ x[row] for row < rows: r bf16 [N, K], x and y (fp32 or bf16) at their
/// views' row strides; K and the strides multiples of 8 (16-byte reads).
pub fn restRows(x: cuda.View, r: cuda.View, y: cuda.View, rows: usize, row_mul: usize, row_add: usize) cuda.Error!void {
    return cuda.launched(zs_nemotron_rest_rows(&x, &r, &y, @intCast(rows), @intCast(row_mul), @intCast(row_add)));
}

/// The MTP draw's greedy pick over up to 32 candidates a row: vals fp32 and ids int64 [rows, C] -> out int32
/// [rows] the rank-0 id (value descending, id ascending), prob fp32 [rows] 1 / the top-k mass share.
pub fn keyedGreedy(vals: cuda.View, ids: cuda.View, out: cuda.View, prob: ?cuda.View, k: usize) cuda.Error!void {
    const count: usize = @intCast(vals.size[1]);
    return cuda.launched(zs_nemotron_keyed_greedy(&vals, &ids, &out, if (prob) |*p| p else null, @intCast(@min(k, count))));
}

/// `yk`'s fp32 [rows, ns, D] MoE slots combined in slot order (the first `nr` weighted by `wt`, the rest —
/// the shared experts' rows — added), into this rank's fp32 partial `out` [rows, D] for the ranks' sum.
pub fn moePartial(yk: cuda.View, wt: cuda.View, out: cuda.View, nr: usize) cuda.Error!void {
    return cuda.launched(zs_nemotron_moe_partial(&yk, &wt, &out, @intCast(nr)));
}
extern fn zs_nemotron_add_rmsnorm(x: *const cuda.View, r: ?*const cuda.View, w: *const cuda.View, h: ?*const cuda.View, y: *const cuda.View, xs: *const cuda.View, eps: f64) c_int;
extern fn zs_nemotron_add_moe_norm(x: *const cuda.View, yk: *const cuda.View, wt: *const cuda.View, w: *const cuda.View, hn: *const cuda.View, y: *const cuda.View, xs: *const cuda.View, eps: f64, nr: i64) c_int;
extern fn zs_nemotron_concat_norms(e: *const cuda.View, hd: *const cuda.View, we: *const cuda.View, wh: *const cuda.View, out: *const cuda.View, xs: *const cuda.View, eps: f64) c_int;
extern fn zs_nemotron_group_rmsnorm(x: *const cuda.View, w: *const cuda.View, out: *const cuda.View, xs: *const cuda.View, eps: f64, groups: i64) c_int;
extern fn zs_nemotron_router(x: *const cuda.View, w: *const cuda.View, part: *const cuda.View) c_int;
extern fn zs_nemotron_topk(part: *const cuda.View, bias: *const cuda.View, idx: *const cuda.View, wt: *const cuda.View, scaling: f64, topk: i64, ns: i64, norm: i64) c_int;
extern fn zs_nemotron_serial_feed(sampled: *const cuda.View, ids: *const cuda.View, meta: *const cuda.View, history: *const cuda.View) c_int;
extern fn zs_nemotron_plan_routed(picks: *const cuda.View, members: *const cuda.View, items: *const cuda.View, counts: *const cuda.View, rows: i64, slots: i64, routed: i64, e: i64, tile: i64) c_int;
extern fn zs_nemotron_rest_rows(x: *const cuda.View, r: *const cuda.View, y: *const cuda.View, rows: i64, row_mul: i64, row_add: i64) c_int;
extern fn zs_nemotron_keyed_greedy(vals: *const cuda.View, ids: *const cuda.View, out: *const cuda.View, prob: ?*const cuda.View, k: i64) c_int;
extern fn zs_nemotron_moe_partial(yk: *const cuda.View, wt: *const cuda.View, out: *const cuda.View, nr: i64) c_int;
