//! Unlimited-OCR-2's DASH attend (spec 147 WS-04): `kernels/unlimited_ocr/dash_attention.cu`'s two kernels. A
//! row at position i attends its causal window of `window` keys (no bias) and the 32-key blocks its router chose
//! (`ids` [R, S], a slot < 0 absent; `bias` [R, S] fp32 added to the block's scores outside the window). The tile
//! kernel (64 rows a block, a prompt's chunk) and the step kernel (a block a 32-key unit of a row, an ordered fold;
//! steps and verify windows) give a row the same bits: the caller picks one by cost alone.

const cuda = @import("../../cuda/cuda.zig");

/// Keys a router block: the kernels' unit.
pub const block = 32;

pub const Form = enum { tile, step };

/// The step kernel's units a row at most: `selected` chosen blocks and the window's blocks.
pub fn slots(selected: usize, window: usize) usize {
    return selected + (window + block - 1) / block + 1;
}

/// The step kernel's scratch: `parts` fp32 [rows heads slots, 130] or more, `done` int32 [rows heads] or more,
/// zero before the first call and left zero by each.
pub const Scratch = struct {
    parts: cuda.View,
    done: cuda.View,
};

/// out bf16 [R, H 128] of q bf16 [R, H, 128] at positions at[0] .. at[0] + R - 1 against the caches bf16
/// [cap, H, 128] (holding those positions and every one before), `scale` the scores' (1 / sqrt(128)).
pub fn attend(form: Form, q: cuda.View, kc: cuda.View, vc: cuda.View, at: cuda.View, ids: cuda.View, bias: cuda.View, window: usize, scale: f64, scratch: ?Scratch, out: cuda.View) cuda.Error!void {
    switch (form) {
        .tile => try cuda.launched(zs_unlimited_ocr_dash_tile(&q, &kc, &vc, &at, &ids, &bias, @intCast(window), scale, &out)),
        .step => {
            const s = scratch.?;
            try cuda.launched(zs_unlimited_ocr_dash_step(&q, &kc, &vc, &at, &ids, &bias, @intCast(window), scale, &s.parts, &s.done, &out));
        },
    }
}

pub extern fn zs_unlimited_ocr_dash_tile(q: *const cuda.View, kc: *const cuda.View, vc: *const cuda.View, at: *const cuda.View, ids: *const cuda.View, bias: *const cuda.View, window: i64, scale: f64, out: *const cuda.View) c_int;
pub extern fn zs_unlimited_ocr_dash_step(q: *const cuda.View, kc: *const cuda.View, vc: *const cuda.View, at: *const cuda.View, ids: *const cuda.View, bias: *const cuda.View, window: i64, scale: f64, parts: *const cuda.View, done: *const cuda.View, out: *const cuda.View) c_int;
