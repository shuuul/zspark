// Entry points of the NVFP4 / FP8 kernels: argument checks, then the launchers in qmmf.cu, experts.cu and prompt.cu.
// From TensorFold 0.6.5 (609ca41) qmmf.cpp and prompt.cpp, Apache-2.0; changed for zspark: a C ABI over zs_view.

#include "../zs/host.h"

void qmmf_cuda(const zs::View&, const zs::View&, const zs::View&, double, zs::View&, const zs::View&, int64_t,
               int64_t, int64_t, int64_t, int64_t, int64_t, bool);
void nvfp4_experts_cuda(int64_t, int64_t, int64_t, const zs::View&, int64_t, int64_t, const zs::View&, const zs::View&,
                        int64_t, int64_t, const zs::View&, const zs::View&, const zs::View&, zs::View&, int64_t,
                        double, int64_t, int64_t);
void prompt16_cuda(const zs::View&, const zs::View&, const zs::View&, double, zs::View&, int64_t, int64_t, int64_t,
                   int64_t, bool);

namespace {

bool aligned16(const zs::View& x) {
    return reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0);
}

}  // namespace

extern "C" {

// out (M, N) = x (M, K) bf16 @ W: mode 0 NVFP4 (tiled words, e4m3 block scales [npad/64, K/64, 64, 4]), 1 FP8
// (fragment-order bytes), 2 MXFP8 (those with e8m0 scales [npad/64, K/64, 64, 2]), 3 FP8 with one fp32 scale a
// column and 64 inputs; ``scale`` the per-tensor factor.
int zs_nvfp4_qmmf(const zs_view* x_, const zs_view* w_, const zs_view* bs_, double scale, const zs_view* out_,
                  const zs_view* part_, int64_t mode, int64_t n, int64_t sk, int64_t npad, int64_t bm, int f32) {
    zs::begin();
    const zs::View x(x_), w(w_), bs(bs_), part(part_);
    zs::View out(out_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2, "x: (M, K) bf16");
    ZS_REQUIRE(mode >= 0 && mode <= 3, "mode 0-3");
    const int64_t m = x.size(0), k = x.size(1);
    ZS_REQUIRE(sk >= 1 && k % 64 == 0 && (k / 64) % sk == 0, "K in whole groups of 64, split evenly");
    ZS_REQUIRE(w.defined() && w.numel() * w.element_size() == npad * k / (mode == 0 ? 2 : 1),
               "weight bytes do not match n and K");
    ZS_REQUIRE(mode == 1 || (bs.defined() && bs.numel() * bs.element_size() ==
                                                 (k / 64) * npad * (mode == 0 || mode == 3 ? 4 : 2)),
               "block scales [npad/64, K/64, 64, 4|2] bytes (mode 3: one fp32 a column and 64 inputs)");
    ZS_REQUIRE(out.defined() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
               out.dtype() == (f32 ? ZS_F32 : ZS_BF16), "out: (M, n)");
    ZS_REQUIRE(bm == 0 || bm == 16 || bm == 32 || bm == 64 || bm == -16 || bm == -32 || bm == 128, "bm: 16, 32, 64, 0, -16, -32 or 128");
    ZS_REQUIRE(bm != 128 || mode == 2 || mode == 3, "bm 128: MXFP8 and block FP8 only");
    ZS_REQUIRE(sk <= 8 || bm <= 0 || bm == 128 || (part.defined() && part.numel() >= sk * m * n), "part: (SK, M, n) fp32");
    qmmf_cuda(x, w, bs, scale, out, part, mode, n, k, sk, npad, bm, f32 != 0);
    return zs::status();
}

// Grouped experts on a plan: blocks [E, N/32, K/32, M, 144] int32, scales [E, M] fp32. tiles: the m16 tiles of
// rows one reading of an expert's weights serves, 1 (a step, a narrow window), 2 or 4; even: with 1, the units in
// whole rounds over the warps (experts.cu). Neither changes a bit.
int zs_nvfp4_experts(int64_t epi, int64_t tiles, int64_t even, const zs_view* x_, int64_t x_stride, int64_t slots,
                     const zs_view* w_, const zs_view* scale_, int64_t kg, int64_t nb, const zs_view* items_,
                     const zs_view* counts_, const zs_view* members_, const zs_view* out_, int64_t n, double limit,
                     int64_t skip, int64_t max_units) {
    zs::begin();
    const zs::View x(x_), w(w_), scale(scale_), items(items_), counts(counts_), members(members_);
    zs::View out(out_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x_stride % 4 == 0, "x: bf16 rows, 8-byte aligned");
    ZS_REQUIRE(w.defined() && w.dtype() == ZS_I32 && w.size(-1) == 144, "w: [E, N/32, K/32, M, 144]");
    ZS_REQUIRE(scale.defined() && scale.dtype() == ZS_F32 && scale.size(0) == w.size(0), "scale: [E, M] fp32");
    ZS_REQUIRE(items.defined() && counts.defined() && members.defined(), "plan: items, counts and members");
    ZS_REQUIRE(out.defined() && out.dtype() == (epi == 0 ? ZS_F32 : ZS_BF16), "out dtype");
    ZS_REQUIRE(tiles == 1 || tiles == 2 || tiles == 4, "tiles: 1, 2 or 4 tiles of rows a pass");
    nvfp4_experts_cuda(epi, tiles, even, x, x_stride, slots, w, scale, kg, nb, items, counts, members, out, n, limit,
                       skip, max_units);
    return zs::status();
}

// out (M, n) = x (M, K) bf16 @ W for prompt rows, modes 0-2 as zs_nvfp4_qmmf; ``tile`` never changes a row's bits.
int zs_nvfp4_prompt16(const zs_view* x_, const zs_view* w_, const zs_view* bs_, double scale, const zs_view* out_,
                      int64_t mode, int64_t n, int64_t npad, int64_t tile, int f32) {
    zs::begin();
    const zs::View x(x_), w(w_), bs(bs_);
    zs::View out(out_);
    ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.size(0) >= 1, "x: (M, K) bf16");
    ZS_REQUIRE(aligned16(x), "x rows must start on 16-byte boundaries");
    ZS_REQUIRE(mode >= 0 && mode <= 2 && tile >= 0 && tile <= 5, "mode 0-2, tile 0-5");
    const int64_t m = x.size(0), k = x.size(1);
    ZS_REQUIRE(k % 64 == 0 && npad % 128 == 0 && n <= npad, "K in groups of 64, n padded to 128");
    ZS_REQUIRE(w.defined() && w.numel() * w.element_size() == npad * k / (mode == 0 ? 2 : 1),
               "weight bytes do not match n and K");
    ZS_REQUIRE(mode == 1 || (bs.defined() && bs.numel() == (k / 64) * npad * (mode == 0 ? 4 : 2)),
               "block scales [npad/64, K/64, 64, 4|2]");
    ZS_REQUIRE(out.defined() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
               out.dtype() == (f32 ? ZS_F32 : ZS_BF16), "out: (M, n)");
    prompt16_cuda(x, w, bs, scale, out, mode, n, npad, tile, f32 != 0);
    return zs::status();
}

}  // extern "C"
