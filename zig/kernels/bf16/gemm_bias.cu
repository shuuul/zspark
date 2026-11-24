// BF16 linear with its bias, and an activation after it, on CUTLASS: out (M, N) = act(x (M, K) @ W^T + bias) with
// W [N, K] as stored (CUTLASS's A row-major times B column-major), for the vision tower's linears (spec 019
// WS-02). What PyTorch's `nn.Linear` then a GELU does in two kernels is one here: the product summed in fp32,
// the bias added in fp32 and the sum rounded to bf16 (cuBLASLt's bias epilogue), then, when asked, the GELU of
// that bf16 value in fp32 rounded again (PyTorch's elementwise GELU on a bf16 tensor). The bias reaches the
// epilogue as CUTLASS's C with a row stride of 0 (CUTLASS's example 12_gemm_bias_relu), so every row reads the
// same N values.
//
// One tile for every row count and no K split: an element's K products are summed in the order the tile's
// K steps give, which depends on K alone, so a row's bits are its own whatever the rows beside it (the row rule).
// K need not be a multiple of the tile's 32 (fc2's 4,304): CUTLASS's multistage loop predicates the residue.
// The tile is gemm.cu's `Half`: 128 x 128 x 32, 2 x 2 warps of 64 x 64, 3 stages.
//
// Shared memory a block (CUTLASS's multistage loop): 3 stages of A 128 x 32 and B 128 x 32 bf16, 3 x (8,192 +
// 8,192) = 49,152 B, two blocks an SM; the epilogue reuses it. Nothing else is staged.
//
// `zs_b16_gemm_bias_residual` (spec 142 WS-08): the same tile and mainloop, CUTLASS's broadcast epilogue
// (`GemmUniversalWithBroadcast`) with the bias as its column vector and the residual stream h as C and D, in
// place: h = bf(h + bf(x W^T + bias)), the bits of `zs_b16_gemm_bias` then the add, the linear's output never
// written. Its epilogue stages the accumulators through the mainloop's 49,152 B as the other's does.
//
// Provenance: zspark's own (gemm.cu's tile, CUTLASS 4.8's LinearCombinationGeneric); the arithmetic is
// transformers 5.18's Qwen3_5VisionMLP / Qwen3_5VisionPatchMerger as TensorFold 0.6.5 runs them.

#include <stdint.h>

#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"

#include "../zs/host.h"

namespace {

__device__ __forceinline__ float bf_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// The activations on the bias-added sum, rounded to bf16 first as PyTorch's second kernel reads it. CUTLASS
// instantiates a functor on float (its parameter traits) and on Array<float, N> (the epilogue's fragments).
template <int ACT> struct Act {
  CUTLASS_HOST_DEVICE static float one(float x) {
#if defined(__CUDA_ARCH__)
    if (ACT == 0) return x;
    const float v = bf_round(x);
    if (ACT == 1) {
      // gelu_pytorch_tanh: PyTorch's GeluType::Tanh, 0.5 x (1 + tanh(sqrt(2) (2 / sqrt(pi)) 0.5 (x + 0.044715 x^3)))
      const float kBeta = 1.41421356237309504880f * 1.12837916709551257390f * 0.5f;
      const float inner = kBeta * (v + 0.044715f * v * v * v);
      return 0.5f * v * (1.f + tanhf(inner));
    }
    if (ACT == 2) {
      // nn.GELU(): x * 0.5 (1 + erf(x / sqrt(2)))
      return v * 0.5f * (1.f + erff(v * 0.70710678118654752440f));
    }
    // quick_gelu (Unlimited-OCR's CLIP, spec 137 WS-03): x * sigmoid(1.702 x), each PyTorch op on bf16 rounded
    const float s = bf_round(1.f / (1.f + expf(-bf_round(1.702f * v))));
    return v * s;
#else
    return x;
#endif
  }
};

template <class T> struct Identity {
  CUTLASS_HOST_DEVICE T operator()(T const& x) const { return x; }
};
template <int N> struct Identity<cutlass::Array<float, N>> {
  CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& x) const { return x; }
};
template <class T> struct GeluTanh {
  CUTLASS_HOST_DEVICE T operator()(T const& x) const { return Act<1>::one(x); }
};
template <int N> struct GeluTanh<cutlass::Array<float, N>> {
  CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& x) const {
    cutlass::Array<float, N> y;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) y[i] = Act<1>::one(x[i]);
    return y;
  }
};
template <class T> struct GeluErf {
  CUTLASS_HOST_DEVICE T operator()(T const& x) const { return Act<2>::one(x); }
};
template <int N> struct GeluErf<cutlass::Array<float, N>> {
  CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& x) const {
    cutlass::Array<float, N> y;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) y[i] = Act<2>::one(x[i]);
    return y;
  }
};

template <class T> struct QuickGelu {
  CUTLASS_HOST_DEVICE T operator()(T const& x) const { return Act<3>::one(x); }
};
template <int N> struct QuickGelu<cutlass::Array<float, N>> {
  CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& x) const {
    cutlass::Array<float, N> y;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) y[i] = Act<3>::one(x[i]);
    return y;
  }
};

template <template <class> class A>
using Gemm = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGeneric<A, cutlass::bfloat16_t, 8, float, float,
                                                        cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3>;

template <class Op>
void run(const zs::View& x, const zs::View& w, const zs::View& bias, zs::View& out, int M, int N, int K) {
  typename Op::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, cutlass::gemm::GemmCoord(M, N, K), 1,
                              {1.f, 0.f}, x.data_ptr(), w.data_ptr(), bias.data_ptr(), out.data_ptr(), 0, 0, 0, 0,
                              x.stride(0), static_cast<int64_t>(K), 0, static_cast<int64_t>(N));
  Op op;
  if (op.can_implement(args) != cutlass::Status::kSuccess)
    return zs::fail("b16 gemm bias: CUTLASS cannot take this shape");
  if (op.initialize(args, nullptr, zs::stream()) != cutlass::Status::kSuccess)
    return zs::fail("b16 gemm bias: CUTLASS initialize failed");
  if (op.run(zs::stream()) != cutlass::Status::kSuccess) return zs::fail("b16 gemm bias: CUTLASS launch failed");
}

// The residual after the bias (spec 142 WS-08): z = bf(c + bf(ab + v)), with c the residual stream h read where z
// is written, v the bias broadcast down the columns. The two roundings are those of the linear's bf16 output and
// of the add after it (`ocr_norm_kernel`'s `h + branch`), so h's bits are the two launches'. The epilogue reads a
// thread's c and writes its z over the same elements, in the same iteration: in place is safe.
struct ResidualAfterBias
    : cutlass::epilogue::thread::LinearCombinationBiasElementwise<cutlass::bfloat16_t, float, float, cutlass::bfloat16_t,
                                                                  cutlass::bfloat16_t, 8, cutlass::epilogue::thread::Identity<float>,
                                                                  cutlass::plus<float>, false, cutlass::bfloat16_t> {
  using Base = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
      cutlass::bfloat16_t, float, float, cutlass::bfloat16_t, cutlass::bfloat16_t, 8,
      cutlass::epilogue::thread::Identity<float>, cutlass::plus<float>, false, cutlass::bfloat16_t>;
  using Base::Base;
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE void operator()(FragmentZ& z, FragmentT&, FragmentAccumulator const& ab, FragmentC const& c,
                                      FragmentCompute const& v) const {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      const float branch = static_cast<float>(cutlass::bfloat16_t(ab[i] + v[i]));
      z[i] = cutlass::bfloat16_t(static_cast<float>(c[i]) + branch);
    }
  }
  // never called (`is_source_needed`), there for the epilogue's other path to compile
  CUTLASS_HOST_DEVICE void operator()(FragmentZ& z, FragmentT&, FragmentAccumulator const& ab,
                                      FragmentCompute const& v) const {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) z[i] = cutlass::bfloat16_t(ab[i] + v[i]);
  }
};

using GemmResidual = cutlass::gemm::device::GemmUniversalWithBroadcast<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    ResidualAfterBias, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3>;

}  // namespace

// out bf16 (M, N) = act(x bf16 (M, K) @ w bf16 [N, K]^T + bias bf16 [N]); act 0 none, 1 GELU tanh, 2 exact GELU,
// 3 quick_gelu, each on the bias-added sum rounded to bf16. K and N in whole 8s (16-byte rows), any M.
extern "C" int zs_b16_gemm_bias(const zs_view* x_, const zs_view* w_, const zs_view* bias_, const zs_view* out_,
                                int64_t act) {
  zs::begin();
  const zs::View x(x_), w(w_), bias(bias_);
  zs::View out(out_);
  ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.dim() == 2 && w.is_contiguous(), "w: [N, K] bf16");
  const int64_t n = w.size(0), k = w.size(1);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.size(1) == k && x.size(0) >= 1 &&
                 x.stride(1) == 1,
             "x: (M, K) bf16");
  ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_BF16 && bias.is_contiguous() && bias.numel() == n, "bias: [N] bf16");
  ZS_REQUIRE(k % 8 == 0 && n % 8 == 0 && x.stride(0) % 8 == 0, "K, N and x's row stride in whole 8s");
  ZS_REQUIRE(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
             "x, w, bias and out on 16-byte boundaries");
  const int64_t m = x.size(0);
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_BF16 && out.is_contiguous() && out.dim() == 2 && out.size(0) == m &&
                 out.size(1) == n,
             "out: (M, N) bf16");
  ZS_REQUIRE(m <= INT32_MAX && n <= INT32_MAX && k <= INT32_MAX, "shape within int32");
  ZS_REQUIRE(act >= 0 && act <= 3, "act: 0 none, 1 GELU tanh, 2 exact GELU, 3 quick_gelu");
  const int M = static_cast<int>(m), N = static_cast<int>(n), K = static_cast<int>(k);
  if (act == 0) run<Gemm<Identity>>(x, w, bias, out, M, N, K);
  else if (act == 1) run<Gemm<GeluTanh>>(x, w, bias, out, M, N, K);
  else if (act == 2) run<Gemm<GeluErf>>(x, w, bias, out, M, N, K);
  else run<Gemm<QuickGelu>>(x, w, bias, out, M, N, K);
  return zs::status();
}

// h bf16 (M, N) = bf(h + bf(x bf16 (M, K) @ w bf16 [N, K]^T + bias bf16 [N])), in place (spec 142 WS-08): the
// linear before a residual add, its output never written. The same tile and K order as `zs_b16_gemm_bias`, so h
// holds the bits of that call followed by the add. K and N in whole 8s, any M.
extern "C" int zs_b16_gemm_bias_residual(const zs_view* x_, const zs_view* w_, const zs_view* bias_, const zs_view* h_) {
  zs::begin();
  const zs::View x(x_), w(w_), bias(bias_);
  zs::View h(h_);
  ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.dim() == 2 && w.is_contiguous(), "w: [N, K] bf16");
  const int64_t n = w.size(0), k = w.size(1);
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.size(1) == k && x.size(0) >= 1 &&
                 x.stride(1) == 1,
             "x: (M, K) bf16");
  ZS_REQUIRE(bias.defined() && bias.dtype() == ZS_BF16 && bias.is_contiguous() && bias.numel() == n, "bias: [N] bf16");
  ZS_REQUIRE(k % 8 == 0 && n % 8 == 0 && x.stride(0) % 8 == 0, "K, N and x's row stride in whole 8s");
  ZS_REQUIRE(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0 &&
                 reinterpret_cast<uintptr_t>(h.data_ptr()) % 16 == 0,
             "x, w, bias and h on 16-byte boundaries");
  const int64_t m = x.size(0);
  ZS_REQUIRE(h.defined() && h.dtype() == ZS_BF16 && h.is_contiguous() && h.dim() == 2 && h.size(0) == m && h.size(1) == n,
             "h: (M, N) bf16");
  ZS_REQUIRE(m <= INT32_MAX && n <= INT32_MAX && k <= INT32_MAX, "shape within int32");
  // C is h, read where D writes it; the broadcast vector the bias; no T
  typename GemmResidual::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
                                        cutlass::gemm::GemmCoord(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)), 1,
                                        {1.f, 1.f}, x.data_ptr(), w.data_ptr(), h.data_ptr(), h.data_ptr(), bias.data_ptr(),
                                        nullptr, 0, 0, 0, 0, 0, 0, x.stride(0), k, n, n, 0, 0);
  GemmResidual op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) zs::fail("b16 gemm bias residual: CUTLASS cannot take this shape");
  else if (op.initialize(args, nullptr, zs::stream()) != cutlass::Status::kSuccess)
    zs::fail("b16 gemm bias residual: CUTLASS initialize failed");
  else if (op.run(zs::stream()) != cutlass::Status::kSuccess) zs::fail("b16 gemm bias residual: CUTLASS launch failed");
  return zs::status();
}
