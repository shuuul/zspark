//! GLM-5.3-Flash's embedding, hyper-connections and norms on the device. The residual is four streams of
//! `dims` values a row; a site reads them out (`mix`, then `pre`) and, once the branch has run on every rank,
//! writes it back (`post`). `glue_ref.zig` has the same steps on the host.

const cuda = @import("../../../cuda/cuda.zig");
const host = @import("glue_ref.zig");
const glue = @import("../../../decode/glue.zig");

pub const streams = host.streams;

/// The shape of `mix`'s result for `rows` rows: fp32 [rows, 16, 32].
pub fn partShape(rows: usize) [3]usize {
    return .{ rows, host.parts, host.entry };
}

/// ids int32 [rows], table bf16 [vocab, dims] -> out bf16 [rows, 4 * dims].
pub fn embed(ids: cuda.View, table: cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(cuda.kernels.zs_embed(&ids, &table, &out, streams));
}

/// x bf16 [rows, 4 * dims], dots bf16 [24, 4 * dims] (`hc_*_fn`) -> part fp32 [rows, 16, 32]: the mix dots and
/// the streams' squares in 16 blocks, which `pre` adds.
pub fn mix(x: cuda.View, dots: cuda.View, part: cuda.View) cuda.Error!void {
    return cuda.launched(zs_glm5_hc_mix(&x, &dots, &part));
}

/// The read-out's two launches by their other forms (spec 073 WS-05), each the same bits as its base and off
/// unless `main` reads its switch: `lanes` (`ZSPARK_GLM_HC_LANES`), `pre`'s gates and Sinkhorn on a warp's lanes
/// and not on one thread; `octs` (`ZSPARK_GLM_HC_OCTS`), `mix`'s values loaded eight at a time. Said before a
/// step's graph is captured.
pub var lanes = false;
pub var octs = false;

pub fn setForms() void {
    zs_glm5_hc_forms(@intFromBool(lanes), @intFromBool(octs));
}

/// A site's constants on the device: `hc_*_fn` bf16 [24, 4 * dims], `hc_*_base` fp32 [24], `hc_*_scale` fp32
/// [3], and the norm the collapsed row takes, bf16 [dims].
pub const Site = struct {
    dots: cuda.View,
    base: cuda.View,
    scale: cuda.View,
    norm: cuda.View,
};

/// The read-out after `mix`: out bf16 [rows, dims], the collapsed row normed, with post fp32 [rows, 4] and
/// comb fp32 [rows, 4, 4] kept for `post`.
pub fn pre(x: cuda.View, part: cuda.View, site: Site, out: cuda.View, post_gates: cuda.View, comb: cuda.View, eps: f32, hc_eps: f32, iters: usize) cuda.Error!void {
    return cuda.launched(zs_glm5_hc_pre(&x, &part, &site.base, &site.scale, &site.norm, &out, &post_gates, &comb, eps, hc_eps, @intCast(iters)));
}

/// x bf16 [rows, 4 * dims] takes the branch in place: gathered fp32 or bf16 [world, rows or more, dims], every
/// rank's partial, added rank 0 first in fp32.
pub fn post(x: cuda.View, gathered: cuda.View, post_gates: cuda.View, comb: cuda.View) cuda.Error!void {
    return cuda.launched(zs_glm5_hc_post(&x, &gathered, &post_gates, &comb));
}

/// p fp32 [rows, dims] in place, each value rounded to bf16 and kept fp32: what a bf16 partial gives `post`.
pub fn roundBf16(p: cuda.View) cuda.Error!void {
    return cuda.launched(zs_glm5_round_bf16(&p));
}

/// x bf16 [rows, dims] takes the branch in place, the MTP layer's plain residual (`decode/glue.zig`).
pub const residualAdd = glue.residualAdd;

/// gathered fp32 [world, rows or more, dims] -> out bf16 [rows, dims]: the ranks' partials added rank 0 first
/// (`decode/glue.zig`, shared with Flash Next's DFlash2).
pub const branch = glue.branch;

/// x bf16 [rows, 4 * dims] -> out bf16 [rows, dims], the streams' mean.
pub fn streamMean(x: cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_glm5_stream_mean(&x, &out));
}

/// out bf16 [rows, dims] = w bf16 [dims] times x over its rms; x and out may be the first columns of wider rows
/// (`decode/glue.zig`).
pub const rmsnorm = glue.rmsnorm;

extern fn zs_glm5_hc_mix(x: *const cuda.View, dots: *const cuda.View, part: *const cuda.View) c_int;
extern fn zs_glm5_hc_pre(x: *const cuda.View, part: *const cuda.View, base: *const cuda.View, scale: *const cuda.View, norm: *const cuda.View, out: *const cuda.View, post: *const cuda.View, comb: *const cuda.View, eps: f64, hc_eps: f64, iters: i64) c_int;
extern fn zs_glm5_hc_forms(lanes: c_int, octs: c_int) void;
extern fn zs_glm5_hc_post(x: *const cuda.View, gathered: *const cuda.View, post: *const cuda.View, comb: *const cuda.View) c_int;
extern fn zs_glm5_round_bf16(p: *const cuda.View) c_int;
extern fn zs_glm5_stream_mean(x: *const cuda.View, out: *const cuda.View) c_int;
