// Rank MTP candidates by value descending, id ascending; compute the top-K share in fp64.
// From TensorFold 1.1.5 (792cdbab, unchanged at 1b31cc74) zig/kernels/cuda/nemotron_keyed.cu, Apache-2.0;
// changed for zspark: the zs launcher at the file's end. The kernel's device code is upstream's byte for byte.

#include <math.h>
#include <stdint.h>

#include "../zs/host.h"

// One warp a row of C <= 32 candidates: OUT the rank-0 id; PROB, if given, 1 / sum over the top K of e^(v - v0).
extern "C" __global__ void __launch_bounds__(32) tf_nemo_keyed_greedy(const float* __restrict__ VALS, const long long* __restrict__ IDS,
                                                                        int* __restrict__ OUT, float* __restrict__ PROB, int C, int K) {
    constexpr unsigned full = 0xffffffffu;
    const int64_t r = blockIdx.x;
    const int lane = threadIdx.x;
    const bool ok = lane < C;
    const double v = ok ? static_cast<double>(VALS[r * C + lane]) : -INFINITY;
    const long long id = ok ? IDS[r * C + lane] : (1LL << 40);
    int rank = 0;
    for (int j = 0; j < 32; ++j) {
        const double vj = __shfl_sync(full, v, j);
        const long long idj = __shfl_sync(full, id, j);
        if (j < C && (vj > v || (vj == v && idj < id))) ++rank;
    }
    const unsigned first = __ballot_sync(full, ok && rank == 0);
    const long long tok = __shfl_sync(full, id, first ? __ffs(first) - 1 : 0);
    if (lane == 0) OUT[r] = static_cast<int>(tok);
    if (PROB == nullptr) return;
    const bool kept = ok && rank < K;
    double top = kept ? v : -INFINITY;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) top = fmax(top, __shfl_xor_sync(full, top, off));
    double e = kept ? exp(v - top) : 0.0;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) e = e + __shfl_xor_sync(full, e, off);
    if (lane == 0) PROB[r] = static_cast<float>(1.0 / e);
}

extern "C" {

// vals fp32 and ids int64 [R, C], C up to 32 -> out int32 [R] the rank-0 id (value descending, id ascending),
// and prob fp32 [R] if given: 1 / the top-K sum of e^(v - v0) in fp64 (the candidates' mass share).
int zs_nemotron_keyed_greedy(const zs_view* vals_, const zs_view* ids_, const zs_view* out_,
                             const zs_view* prob_, int64_t k) {
    zs::begin();
    const zs::View vals(vals_), ids(ids_), out(out_), prob(prob_);
    ZS_REQUIRE(vals.defined() && vals.dtype() == ZS_F32 && vals.dim() == 2 && vals.is_contiguous(),
               "vals: fp32 [R, C]");
    const int64_t rows = vals.size(0), c = vals.size(1);
    ZS_REQUIRE(c >= 1 && c <= 32, "C: up to 32 candidates a row");
    ZS_REQUIRE(ids.defined() && ids.dtype() == ZS_I64 && ids.is_contiguous() && ids.numel() >= rows * c,
               "ids: int64 [R, C]");
    ZS_REQUIRE(out.defined() && out.dtype() == ZS_I32 && out.numel() >= rows, "out: int32 [R]");
    if (prob.defined())
        ZS_REQUIRE(prob.dtype() == ZS_F32 && prob.numel() >= rows, "prob: fp32 [R]");
    ZS_REQUIRE(k >= 0 && k <= c, "k: the top of the candidates");
    if (rows > 0)
        tf_nemo_keyed_greedy<<<static_cast<unsigned>(rows), 32, 0, zs::stream()>>>(
            vals.data_ptr<float>(), ids.data_ptr<long long>(), out.data_ptr<int>(),
            prob.defined() ? prob.data_ptr<float>() : nullptr, static_cast<int>(c), static_cast<int>(k));
    ZS_CUDA_LAUNCH_CHECK();
    return zs::status();
}

}  // extern "C"
