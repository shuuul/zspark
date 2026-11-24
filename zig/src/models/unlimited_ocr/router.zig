//! Unlimited-OCR-2's DASH router on the device (spec 147 WS-03, `kernels/unlimited_ocr/router.cu`): a window's
//! rows projected by the layer's stacked `q_proj`/`k_proj` (`model.Router.qk`, the CUTLASS bf16 GEMM, bf16 out as
//! HF's `F.linear`), then `prep` (Gemma RMSNorm and page2d RoPE: the row's index q, its index key into the cache by
//! position), `kbar` (the summary key of each block a window's row completes) and `route` (a row's 12 remote
//! blocks and their bias). `forward.zig` calls them in that order on Unlimited-OCR-2's layers, over the state's
//! index keys and k_bar by position, its blocks' image pages and the window's coordinates (`pages.zig`); the
//! model holds the `Rope` (`Model.page2d`), which the main attention's q and k share.
//!
//! The index q and k are bf16: HF's are fp32 tensors of bf16 values (RoPE runs in bf16, then `.float()`), so
//! nothing is lost. A row's results read its own projection and coordinates and the k_bar of blocks that end
//! 128 keys or more before it: the same bits alone, in a chunk, in a window or as a step.

const std = @import("std");
const cuda = @import("../../cuda/cuda.zig");
const config_file = @import("config.zig");
const model_file = @import("model.zig");
const pages_file = @import("pages.zig");

const Dash = config_file.Dash;
const Page2d = config_file.Page2d;

/// The index head's width, the kernels' (`router.cu` `D`).
pub const dim = 128;
/// Keys a block, the kernels' (`BLOCK`).
pub const block = 32;
/// The entmax's candidates, the kernels' (`TOP_M`).
pub const candidates = 16;

/// A row's coordinates, int32 [rows, 3] on the device (`pages.zig`): its page2d RoPE's token and page (HF's
/// `_build_page_rope_pos` in a prompt, the decode state machine after), and its query page: the image page whose
/// blocks it picks among (`_build_page_aware_ids`' `query_token_page_ids` in a prompt, `marker count - 1 - prompt
/// pages` in a reply), -1 for none (every remote block).
pub const Coords = pages_file.Coords;

/// A block's least and largest image page, int32 [blocks, 2]; -1 both for a block without an image token
/// (`_uocr_dash_decode_allow`'s `bmin`, `bmax`).
pub const Pages = pages_file.Pages;

/// page2d RoPE's inverse frequencies a pair (fp32 [dim / 2]), as `DeepseekV2IndexerRotary2D.__init__` computes them
/// in fp32: the first `n_tok` pairs `1 / tok_theta^(i / n_tok)` ("resample"), the last `page_pairs`
/// `1 / page_theta^(i / page_pairs)`. The powers are taken in f64 and rounded once.
pub const Rope = struct {
    inv: cuda.Buffer,
    n_tok: usize,
    page_mod: usize,

    pub fn init(rope: Page2d) !Rope {
        const half = dim / 2;
        const n_tok = half - rope.page_pairs;
        var freq: [half]f32 = undefined;
        for (freq[0..n_tok], 0..) |*f, i| f.* = inverse(rope.tok_theta, i, n_tok);
        for (freq[n_tok..], 0..) |*f, i| f.* = inverse(rope.page_theta, i, rope.page_pairs);
        var inv: cuda.Buffer = try .init(.f32, &.{half});
        errdefer inv.deinit();
        try inv.upload(std.mem.sliceAsBytes(&freq));
        return .{ .inv = inv, .n_tok = n_tok, .page_mod = rope.page_mod };
    }

    fn inverse(theta: f64, i: usize, n: usize) f32 {
        const exponent: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
        const power: f32 = @floatCast(std.math.pow(f64, theta, exponent));
        return 1.0 / power;
    }

    pub fn deinit(self: *Rope) void {
        self.inv.deinit();
        self.* = undefined;
    }
};

/// A layer's routing constants from the config (`configure_dash_inference`'s `_infer_params`).
pub const Params = struct {
    /// The entmax input's factor, `dash_scale / sqrt(index_dim)`.
    scale: f64,
    floor: f64,
    sigma: f64,

    pub fn of(dash: Dash, layer: usize) Params {
        return .{ .scale = dash.scale[layer] / @sqrt(@as(f64, @floatFromInt(dash.index_dim))), .floor = dash.floor[layer], .sigma = dash.sigma };
    }
};

pub extern fn zs_unlimited_ocr_router_prep(qk: *const cuda.View, at: *const cuda.View, coords: *const cuda.View, q_norm: *const cuda.View, k_norm: *const cuda.View, inv: *const cuda.View, n_tok: i64, page_mod: i64, eps: f64, idx_q: *const cuda.View, idx_k: *const cuda.View) c_int;
pub extern fn zs_unlimited_ocr_router_kbar(idx_k: *const cuda.View, head_cls: *const cuda.View, at: *const cuda.View, rows: i64, kbar: *const cuda.View) c_int;
pub extern fn zs_unlimited_ocr_router_route(idx_q: *const cuda.View, kbar: *const cuda.View, at: *const cuda.View, coords: *const cuda.View, pages: *const cuda.View, scale: f64, floor: f64, sigma: f64, ids: *const cuda.View, bias: *const cuda.View, scores: ?*const cuda.View) c_int;

/// The window's rows `x` (bf16 [rows, hidden], after the input norm) through q_proj and k_proj: bf16 [rows, 2 dim].
pub fn project(router: model_file.Router, x: cuda.View, qk: cuda.View) cuda.Error!void {
    return router.qk.forward(x, qk);
}

/// `qk` [rows, 2 dim] to the rows' index q (bf16 [rows, dim]) and their index keys (`idx_k`, bf16 [capacity, dim],
/// row r at position `at[0] + r`). `coords` int32 [rows, 3] (`Coords`).
pub fn prep(router: model_file.Router, rope: Rope, eps: f32, qk: cuda.View, at: cuda.View, coords: cuda.View, idx_q: cuda.View, idx_k: cuda.View) cuda.Error!void {
    return cuda.launched(zs_unlimited_ocr_router_prep(&qk, &at, &coords, &router.q_norm.view, &router.k_norm.view, &rope.inv.view, @intCast(rope.n_tok), @intCast(rope.page_mod), eps, &idx_q, &idx_k));
}

/// k_bar (fp32 [blocks, dim]) of each block one of the window's `rows` completes, from the index keys.
pub fn kbar(router: model_file.Router, idx_k: cuda.View, at: cuda.View, rows: usize, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_unlimited_ocr_router_kbar(&idx_k, &router.head_cls.view, &at, @intCast(rows), &out));
}

/// A row's remote blocks: `ids` int32 [rows, read_blocks] (in the entmax's order, -1 an empty slot) and `bias` fp32
/// alike (0 an empty slot); `scores` fp32 [rows, W] the block scores where a test asks. `kbar`'s rows bound the
/// blocks a row may pick and its shared memory (4 B a block).
pub fn route(params: Params, idx_q: cuda.View, kbar_: cuda.View, at: cuda.View, coords: cuda.View, pages: cuda.View, ids: cuda.View, bias: cuda.View, scores: ?cuda.View) cuda.Error!void {
    const s: ?*const cuda.View = if (scores) |*v| v else null;
    return cuda.launched(zs_unlimited_ocr_router_route(&idx_q, &kbar_, &at, &coords, &pages, params.scale, params.floor, params.sigma, &ids, &bias, s));
}
