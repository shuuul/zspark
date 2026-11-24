//! GLM-5.3-Flash's MoE glue on the device: the selection from the router's logits, the SwiGLU of the dense and
//! shared MLPs, and a rank's share of the block. `glue_ref.zig` has the same steps on the host.

const cuda = @import("../../../cuda/cuda.zig");
const glue = @import("../../../decode/glue.zig");

/// logits fp32 [rows, experts] and bias fp32 [experts] -> picks int32 and weights fp32 [rows, top + 1], the
/// shared expert (id `experts`, weight 1) in the last slot. With `slices`, the router's K slices unsummed
/// ([sk, rows, experts], `cuda.Slices`), the selection sums them and writes `logits` too.
/// One slice (a window's router, summed in its GEMM) is `logits` itself: read there, not passed as `part`.
pub fn select(logits: cuda.View, slices: ?cuda.Slices, bias: cuda.View, picks: cuda.View, weights: cuda.View, top: usize, scale: f32, norm: bool) cuda.Error!void {
    return cuda.launched(zs_glm5_moe_topk(&logits, &bias, &picks, &weights, @intCast(top), scale, @intFromBool(norm), if (slices) |*s| (if (s.view.size[0] > 1) &s.view else null) else null));
}

/// gu bf16 [rows, 2 * width] (gate | up) -> act bf16 [rows, width], the gate held under `limit` and the up
/// within it (`decode/glue.zig`, shared with Flash Next's DFlash2).
pub const swiglu = glue.swiglu;

/// y fp32 [rows * slots, dims] and weights fp32 [rows, slots] -> out fp32 [rows, dims]: the slots times their
/// weights in order, this rank's partial of the block. With `shared`, fp32 [rows, dims], a row's last slot is
/// read there and not in `y`: the shared expert's rows, which a BF16 linear writes on their own.
pub fn combine(y: cuda.View, shared: ?cuda.View, weights: cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_glm5_moe_combine(&y, &weights, &out, if (shared) |*s| s else null));
}

extern fn zs_glm5_moe_topk(logits: *const cuda.View, bias: *const cuda.View, picks: *const cuda.View, weights: *const cuda.View, top: i64, scale: f64, norm: i64, part: ?*const cuda.View) c_int;
extern fn zs_glm5_moe_combine(y: *const cuda.View, weights: *const cuda.View, out: *const cuda.View, shared: ?*const cuda.View) c_int;
