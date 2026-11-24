//! The row glue the families launch (`zig/kernels/common/glue.cu`, spec 136): a row's RMS norm, the ranks'
//! partials of a branch added in rank order, a branch added to a residual (alone, or with the norm that follows in
//! one launch), and the SwiGLU of a dense MLP. GLM-5.3-Flash's layers, Flash Next's DFlash2 drafter, Qwen3-ASR and
//! Unlimited-OCR call them; each a row's bits alone.
//! `glue_ref.zig` has the branch and the SwiGLU on the host.

const cuda = @import("../cuda/cuda.zig");

/// out bf16 [rows, dims] = w bf16 or fp32 [dims] times x bf16 over its rms; x and out may be the first columns of
/// wider rows. dims to 4096.
pub fn rmsnorm(x: cuda.View, w: cuda.View, out: cuda.View, eps: f32) cuda.Error!void {
    return cuda.launched(zs_rmsnorm(&x, &w, &out, eps));
}

/// gathered fp32 [world, rows or more, dims] -> out bf16 [rows, dims]: the ranks' partials added rank 0 first.
pub fn branch(gathered: cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_branch(&gathered, &out));
}

/// x bf16 [rows, dims] takes the branch in place: x = bf16(x + bf16(the ranks' partials added rank 0 first)),
/// gathered fp32 [world, rows or more, dims]; or x = bf16(x + branch), one rank's branch bf16 [rows, dims].
pub fn residualAdd(x: cuda.View, gathered: cuda.View) cuda.Error!void {
    return cuda.launched(zs_residual_add(&x, &gathered));
}

/// `residualAdd` of one rank's branch, then `rmsnorm` of the new x, in one launch and with their bits: x bf16
/// [rows, dims] = bf16(x + bf16(branch)), the branch a matmul's K slices (`Linear.slices`) added in slice order,
/// then out bf16 [rows, dims] the norm of x with w. dims to 4096.
pub fn residualNorm(x: cuda.View, branch_slices: cuda.Slices, w: cuda.View, out: cuda.View, eps: f32) cuda.Error!void {
    return cuda.launched(zs_residual_rmsnorm(&x, &branch_slices.view, &w, &out, eps));
}

/// `residualAdd` of a bf16 branch [rows, dims] as its product wrote it, then the norm of the new x by w bf16 [dims]
/// into out bf16 [rows, dims], in one launch and with `rmsnorm`'s bits. dims 257 to 4096.
pub fn residualNormBf16(x: cuda.View, product: cuda.View, w: cuda.View, out: cuda.View, eps: f32) cuda.Error!void {
    return cuda.launched(zs_residual_rmsnorm(&x, &product, &w, &out, eps));
}

/// gu bf16 [rows, 2 * width] (gate | up) -> act bf16 [rows, width], the gate held under `limit` and the up
/// within it.
pub fn swiglu(gu: cuda.View, act: cuda.View, limit: f32) cuda.Error!void {
    return cuda.launched(zs_swiglu(&gu, &act, limit));
}

extern fn zs_rmsnorm(x: *const cuda.View, w: *const cuda.View, out: *const cuda.View, eps: f64) c_int;
extern fn zs_branch(gathered: *const cuda.View, out: *const cuda.View) c_int;
extern fn zs_residual_add(x: *const cuda.View, gathered: *const cuda.View) c_int;
extern fn zs_residual_rmsnorm(x: *const cuda.View, part: *const cuda.View, w: *const cuda.View, out: *const cuda.View, eps: f64) c_int;
extern fn zs_swiglu(gu: *const cuda.View, act: *const cuda.View, limit: f64) c_int;
