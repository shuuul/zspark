// BF16 matmul on CUTLASS: out (M, N) = x (M, K) @ W^T with W [N, K] as stored, which is CUTLASS's A row-major
// times B column-major (both K-contiguous), so the checkpoint's bytes are used as they are.
// CUTLASS 4.8's SM120 collective builder covers the F8F6F4 MMA alone, so BF16 x BF16 goes through the classic
// tensor-op GEMM (mma.sync m16n8k16, cp.async multistage), which sm_121 runs.
// An element adds its K products 16 at a time in order in fp32 whatever the tile, and the K slices are the same
// K / sk parallel slices as matmul.cu's, summed by the same reduction: the two kernels and both tiles here give an
// element the same bits, so choosing among them by the row count changes no arithmetic (tested in zig/src/weights/bf16.zig).

#include <stdint.h>
#include <type_traits>

#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "../zs/host.h"
#include "reduce.cuh"

namespace {

// Threadblock tile (TM, TN, TK) of warps (WM, WN). The epilogue stores one element at a time so N need not be
// a multiple of a vector.
template <class El, class Out, int TM, int TN, int TK, int WM, int WN, int STAGES,
          cutlass::epilogue::thread::ScaleType::Kind SCALE = cutlass::epilogue::thread::ScaleType::Nothing>
using Gemm = cutlass::gemm::device::GemmUniversal<
    El, cutlass::layout::RowMajor, El, cutlass::layout::ColumnMajor, Out,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, TK>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Out, 1, float, float, SCALE>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, STAGES>;

// Prompt rows. On the GB10 256x128x32, 128x256x32 and 128x128x32 measured within a tenth of this one.
template <class El, class Out> using Wide = Gemm<El, Out, 128, 128, 64, 64, 64, 3>;
// Up to 16 rows: one MMA row tile, so a call reads the weight once and multiplies nothing but those rows.
template <class El, class Out> using Narrow = Gemm<El, Out, 16, 128, 64, 16, 32, 4>;

// One GEMM into ``d``: the output itself, or the (sk, M, N) fp32 slices.
template <class Op>
void run(const zs::View& x, const zs::View& w, void* d, int M, int N, int K, int SK) {
  const bool split = SK > 1;
  typename Op::Arguments args(
      split ? cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel : cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmCoord(M, N, K), SK, {1.f, 0.f}, x.data_ptr(), w.data_ptr(), nullptr, d, 0, 0, 0,
      static_cast<int64_t>(M) * N, x.stride(0), static_cast<int64_t>(K), static_cast<int64_t>(N),
      static_cast<int64_t>(N));
  Op op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return zs::fail("b16 gemm: CUTLASS cannot take this shape");
  if (op.initialize(args, nullptr, zs::stream()) != cutlass::Status::kSuccess)
    return zs::fail("b16 gemm: CUTLASS initialize failed");
  if (op.run(zs::stream()) != cutlass::Status::kSuccess) return zs::fail("b16 gemm: CUTLASS launch failed");
}

// Two more tiles of one MMA row tile (spec 073 WS-04), for a decoding round's launches: the same sums as Narrow,
// since who computes an element changes and its arithmetic does not (bf16.zig's `Tile.of` chooses by the weight's
// K slices, never by the rows). Every GEMM here reads its weight once at the memory's rate (ncu: sectors filled
// from memory are the weight's bytes to 1%), so what a tile changes is how the reads are asked for. CUTLASS's
// multistage loop holds STAGES * (TN + 16) * 64 * 2 bytes of shared memory a block, A beside B:
//
//   tile     columns  warps  stages  bytes a block                  blocks an SM  K values on their way a block
//   Narrow   128      4      4       73,728 (B 65,536, A 8,192)     1             256
//   Slim     32       1x2    4       24,576 (B 16,384, A 8,192)     4             256
//   Deep     8        1      8       24,576 (B 8,192, A 16,384)     4             512
//
// A's stage is 16 rows whatever the window (CUTLASS's tile): with Deep it is two thirds of a block, and could be
// less only in a kernel that is not CUTLASS's. Measured alone (`ZSPARK_BF16_PROFILE`, a kernel's own duration,
// past L2, 1 to 16 rows): Deep on a weight of one K slice, 16512 x 2560 367 us for 375 with Narrow, 13952 x 2560
// 311 for 318, 10240 x 320 28.8 for 30.2, the head 5,439 for 5,587; Slim on a weight split over K, 320 x 10240
// in 32 slices 30.7 for 31.6, 1280 x 2560 in 8 30.5 for 31.4, 2560 x 6144 in 4 142.5 for 144.5. Each loses on the
// other's shapes (Deep 31.5 on 320 x 10240), and 16 columns by 12 stages, 32 by 8, 64 by 4 and 128 by 2 or 3
// were between them or behind.
template <class El, class Out> using Slim = Gemm<El, Out, 16, 32, 64, 16, 16, 4>;
template <class El, class Out> using Deep = Gemm<El, Out, 16, 8, 64, 16, 8, 8>;

// A window's tile with stages of 32 values of K (spec 073 WS-04): Wide's shared memory is 3 * (128 + 128) * 64 * 2
// = 98,304 B, B's stages and A's alike, so one block an SM (ncu at 512 rows: 8% occupancy, the SMs' compute at
// 64%, a warp waiting on the tensor pipe); Half's is 49,152 B, two blocks an SM, 96 K values on their way a
// block for 192. The same sums. It wins where the blocks at once still find their weight in L2: every shape
// from 2048 rows (5 to 10% off a launch alone: 16512 x 2560 1,870 to 1,771 us, 2560 x 6144 1,020 to 958), all
// but a weight in 32 slices from 1024, a weight of one slice that L2 holds from 256 (10240 x 320: 9 to 18%);
// under 1024 rows it loses 5 to 16% on the large weights (the same columns are read again by fewer row tiles),
// which keep Wide. `bf16.Tile.of` holds the rule.
template <class El, class Out> using Half = Gemm<El, Out, 128, 128, 32, 64, 64, 3>;

// A window's K slices summed inside the GEMM (spec 095 WS-03): CUTLASS's serial split-K (kGemm with SK slices),
// fp32 out. Slice z's block clears its accumulators and walks the same K range as in kGemmSplitKParallel
// (gemm_universal.h: offset_k and problem_size_k from one gemm_k_size, params_universal_base.h), then waits on
// its tile's semaphore for slice z - 1 and stores fma(1, acc_z, D): slice 0 stores 1 * acc (beta 0, the source
// not read), slice z > 0 beta 1 (`set_k_partition`), one fp32 rounding of acc_z + D. That is b16_slices' sum,
// from slice 0 in slice order, and no mma ever adds to a carried total: the bits of the slices plus b16_reduce
// (tested in zig/tests/weights/bf16_test.zig at windows of 17 to 2,048 rows and 2 to 32 slices). What changes
// is the traffic: D (rows x N fp32, 2.6 MB at 2,048 rows of the read-out) stays in L2 across the slices, where
// the split wrote SK of them, 84 MB at 32, and a reader read them back. Only D must stay fp32: a bf16 D would
// round at every slice. The semaphores, an int a 128 x 128 tile, CUTLASS zeroes each call (init_workspace).
template <class El> using WideSerial =
    Gemm<El, float, 128, 128, 64, 64, 64, 3, cutlass::epilogue::thread::ScaleType::Default>;
template <class El> using HalfSerial =
    Gemm<El, float, 128, 128, 32, 64, 64, 3, cutlass::epilogue::thread::ScaleType::Default>;

template <class Op>
void serial(const zs::View& x, const zs::View& w, float* d, int* locks, int M, int N, int K, int SK) {
  typename Op::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, cutlass::gemm::GemmCoord(M, N, K), SK,
                              {1.f, 0.f}, x.data_ptr(), w.data_ptr(), d, d, 0, 0, 0, 0, x.stride(0),
                              static_cast<int64_t>(K), static_cast<int64_t>(N), static_cast<int64_t>(N));
  Op op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return zs::fail("b16 gemm: CUTLASS cannot take this shape");
  if (op.initialize(args, locks, zs::stream()) != cutlass::Status::kSuccess)
    return zs::fail("b16 gemm: CUTLASS initialize failed");
  if (op.run(zs::stream()) != cutlass::Status::kSuccess) return zs::fail("b16 gemm: CUTLASS launch failed");
}

// From 1,024 rows the 128-row tiles sum their slices in the block; fewer rows keep the parallel split and
// b16_reduce or the reader's sum, the same bits. A slice waits for the one before it, so the chain is SK
// epilogues long whatever the rows, and pays only where the slices it saves are large (ncu, main against the
// serial form, a GEMM and its reduce: the read-out's 32 slices 52 to 136 us at 64 rows, 191 to 177 at 512,
// 354 to 250 at 1,024; the router's 8 on 5 column tiles 64 to 90 at 512, 144 to 124 at 1,024; every split
// shape gains at 1,024 and 2,048). Bf16.zig's `sums_from` is this bound.
constexpr int SUMS_FROM = 1024;
bool sums_inside(int tile, int M) { return (tile == 2 || tile == 5) && M >= SUMS_FROM; }

// One GEMM on the tile asked: 1 Narrow, 2 Wide, 3 Slim, 4 Deep, 5 Half.
template <class El, class Out>
void pick(int tile, const zs::View& x, const zs::View& w, void* d, int M, int N, int K, int SK) {
  switch (tile) {
    case 1: return run<Narrow<El, Out>>(x, w, d, M, N, K, SK);
    case 2: return run<Wide<El, Out>>(x, w, d, M, N, K, SK);
    case 3: return run<Slim<El, Out>>(x, w, d, M, N, K, SK);
    case 5: return run<Half<El, Out>>(x, w, d, M, N, K, SK);
    default: return run<Deep<El, Out>>(x, w, d, M, N, K, SK);
  }
}

// The K slices alone, fp32 (SK, M, N) into ``part``, for a kernel that sums them itself.
template <class El>
void slices(const zs::View& x, const zs::View& w, const zs::View& part, int N, int K, int SK, int tile) {
  pick<El, float>(tile, x, w, part.data_ptr(), static_cast<int>(x.size(0)), N, K, SK);
}

template <class El, bool F32>
void launch(const zs::View& x, const zs::View& w, zs::View& out, const zs::View& part, int N, int K, int SK,
            int tile) {
  using Out = std::conditional_t<F32, float, cutlass::bfloat16_t>;
  const int M = static_cast<int>(x.size(0));
  if (SK == 1) return pick<El, Out>(tile, x, w, out.data_ptr(), M, N, K, 1);
  const long long total = static_cast<long long>(M) * N;
  if (sums_inside(tile, M)) {
    // D fp32: ``out`` itself, or part's first M x N, rounded to bf16 by b16_reduce of one slice; the semaphores
    // after D in ``part`` (SK >= 2 slices of M x N hold both)
    float* d = F32 ? out.data_ptr<float>() : part.data_ptr<float>();
    int* locks = reinterpret_cast<int*>(part.data_ptr<float>() + (F32 ? 0 : (total + 3) / 4 * 4));
    if (tile == 5) serial<HalfSerial<El>>(x, w, d, locks, M, N, K, SK);
    else serial<WideSerial<El>>(x, w, d, locks, M, N, K, SK);
    if (F32 || zs::status()) return;
    b16_reduce<false><<<static_cast<unsigned>((total + 255) / 256), 256, 0, zs::stream()>>>(part.data_ptr<float>(),
                                                                                           out.data_ptr(), total, 1);
    ZS_CUDA_LAUNCH_CHECK();
    return;
  }
  pick<El, float>(tile, x, w, part.data_ptr(), M, N, K, SK);
  if (zs::status()) return;
  b16_reduce<F32><<<static_cast<unsigned>((total + 255) / 256), 256, 0, zs::stream()>>>(
      part.data_ptr<float>(), out.data_ptr(), total, SK);
  ZS_CUDA_LAUNCH_CHECK();
}

// x and w of ``dtype`` (ZS_BF16 with El bfloat16_t).
template <class El>
int gemm(const zs_view* x_, const zs_view* w_, const zs_view* out_, const zs_view* part_, int64_t sk, int64_t tile,
         int dtype) {
  zs::begin();
  const zs::View x(x_), w(w_), part(part_);
  zs::View out(out_);
  ZS_REQUIRE(w.defined() && w.dtype() == dtype && w.dim() == 2 && w.is_contiguous(), "w: [N, K] of the entry's dtype");
  const int64_t n = w.size(0), k = w.size(1);
  ZS_REQUIRE(x.defined() && x.dtype() == dtype && x.dim() == 2 && x.size(1) == k && x.size(0) >= 1,
             "x: (M, K) of the entry's dtype");
  ZS_REQUIRE(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x.stride(0) % 8 == 0 &&
             reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
             "x rows and w must start on 16-byte boundaries");
  // slices of whole cache lines, so CUTLASS's K slices are matmul.cu's
  ZS_REQUIRE(sk >= 1 && k % 64 == 0 && k % sk == 0 && (k / sk) % 64 == 0, "K in whole steps of 64 a slice");
  const int64_t m = x.size(0);
  ZS_REQUIRE(m <= INT32_MAX && n <= INT32_MAX && k <= INT32_MAX, "shape within int32");
  ZS_REQUIRE(tile >= 0 && tile <= 5, "tile: 0 to 5");
  ZS_REQUIRE(tile <= 2 || tile == 5 || m <= 16, "the slim and deep tiles: 16 rows at most");
  const int narrow = tile == 0 ? (m <= 16 ? 1 : 2) : static_cast<int>(tile);
  const int N = static_cast<int>(n), K = static_cast<int>(k), SK = static_cast<int>(sk);
  if (!out.defined()) {
    // the slices unsummed: with one slice, the product itself
    ZS_REQUIRE(part.defined() && part.dtype() == ZS_F32 && part.is_contiguous() && part.numel() >= sk * m * n,
               "part: (SK, M, N) fp32");
    slices<El>(x, w, part, N, K, SK, narrow);
    return zs::status();
  }
  ZS_REQUIRE(out.is_contiguous() && out.dim() == 2 && out.size(0) == m && out.size(1) == n &&
             (out.dtype() == ZS_F32 || out.dtype() == ZS_BF16), "out: (M, N) bf16 or fp32");
  ZS_REQUIRE(sk == 1 || (part.defined() && part.dtype() == ZS_F32 && part.numel() >= sk * m * n),
             "part: (SK, M, N) fp32");
  if (out.dtype() == ZS_F32) launch<El, true>(x, w, out, part, N, K, SK, narrow);
  else launch<El, false>(x, w, out, part, N, K, SK, narrow);
  return zs::status();
}

// Several GEMMs of one shape in one launch, each with its own weight, and those a device count calls empty
// skipped (spec 144, port 5: the pool's indexer GEMM a slot). CUTLASS's batched mode, entry z = blockIdx.z: its
// x and out regions at fixed strides, its weight given to its blocks here (batch_stride_B 0). A block whose
// entry's count is not positive returns before its first load, block-uniform and before any barrier. Past that
// check the body is Kernel2's on the tile `zs_b16_gemm` takes for M (tile 0, one K slice): an element's sums are
// the same instructions in the same order, so its bits are that call's. A graph holds the grid, and which
// entries run is read on the device each replay. ENTRIES bounds the weights a launch carries (the pool's 64 slots).
constexpr int ENTRIES = 64;
struct EntryWeights {
  const void* w[ENTRIES];
};

template <class Kernel>
__global__ void b16_gemm_entries_kernel(typename Kernel::Params params, EntryWeights weights, const int* live,
                                        int live_stride) {
  const int z = static_cast<int>(blockIdx.z);
  if (live[z * live_stride] <= 0) return;
  typename Kernel::Params own = params;
  own.ptr_B = const_cast<void*>(weights.w[z]);
  extern __shared__ int entries_storage[];
  Kernel::invoke(own, *reinterpret_cast<typename Kernel::SharedStorage*>(entries_storage));
}

template <class Op>
void entries(const zs::View& x, int64_t x_stride, const EntryWeights& weights, float* out, int64_t out_stride,
             const int* live, int live_stride, int M, int N, int K, int count) {
  using Kernel = typename Op::GemmKernel;
  typename Kernel::Arguments args(cutlass::gemm::GemmUniversalMode::kBatched, cutlass::gemm::GemmCoord(M, N, K), count,
                                  {1.f, 0.f}, x.data_ptr(), weights.w[0], nullptr, out, x_stride, 0, 0, out_stride,
                                  x.stride(0), static_cast<int64_t>(K), static_cast<int64_t>(N),
                                  static_cast<int64_t>(N));
  if (Kernel::can_implement(args) != cutlass::Status::kSuccess)
    return zs::fail("b16 gemm entries: CUTLASS cannot take this shape");
  // the identity swizzle reads neither the SM count nor the occupancy
  const typename Kernel::Params params(args, 0, 0);
  constexpr int bytes = static_cast<int>(sizeof(typename Kernel::SharedStorage));
  if constexpr (bytes >= (48 << 10)) {
    // Kernel2's opt-in, which CUTLASS makes for Kernel2 alone
    static thread_local bool opted = false;
    if (!opted) {
      const cudaError_t code = cudaFuncSetAttribute(b16_gemm_entries_kernel<Kernel>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
      if (code != cudaSuccess) return zs::fail_cuda(code, "b16 gemm entries: shared memory");
      opted = true;
    }
  }
  b16_gemm_entries_kernel<Kernel><<<params.get_grid_dims(), dim3(Kernel::kThreadCount, 1, 1), bytes, zs::stream()>>>(
      params, weights, live, live_stride);
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

// The hand-written matmul's arguments and bits (zig/tests/kernels/bf16/matmul.cu). ``tile``: 0 the narrow tile up to 16 rows and the wide one above, 1 narrow,
// 2 wide, 3 slim and 4 deep (16 rows at most). Without ``out`` the K slices are left unsummed in ``part``, fp32 (SK, M, N), one slice the product
// itself: the kernel that reads them next adds them in slice order (b16_reduce's sum) and a launch is saved.
extern "C" int zs_b16_gemm(const zs_view* x_, const zs_view* w_, const zs_view* out_, const zs_view* part_,
                           int64_t sk, int64_t tile) {
  return gemm<cutlass::bfloat16_t>(x_, w_, out_, part_, sk, tile, ZS_BF16);
}

// ``count`` GEMMs of one shape in one launch, entry z skipped when live[z, live_column] <= 0. x bf16 [*, K]: entry
// z's [m, K] at element z * x_stride. w: ``count`` views, each bf16 [N, K]. out fp32: entry z's [m, N] at element
// z * out_stride. live int32 [count or more, *], read on the device. A running entry's products are zs_b16_gemm's
// bits for (its x, its w, an fp32 out, one slice, tile 0); a skipped entry's out region is left as it was.
extern "C" int zs_b16_gemm_entries(const zs_view* x_, int64_t m, int64_t x_stride, const zs_view* w_, int64_t count,
                                   const zs_view* out_, int64_t out_stride, const zs_view* live_, int64_t live_column) {
  zs::begin();
  const zs::View x(x_), out(out_), live(live_);
  ZS_REQUIRE(count >= 1 && count <= ENTRIES && w_ != nullptr, "count: 1 to 64 entries, a weight each");
  const zs::View w0(&w_[0]);
  ZS_REQUIRE(w0.defined() && w0.dtype() == ZS_BF16 && w0.dim() == 2 && w0.is_contiguous(), "w: bf16 [N, K]");
  const int64_t n = w0.size(0), k = w0.size(1);
  EntryWeights weights{};
  for (int64_t z = 0; z < count; ++z) {
    const zs::View w(&w_[z]);
    ZS_REQUIRE(w.defined() && w.dtype() == ZS_BF16 && w.dim() == 2 && w.is_contiguous() && w.size(0) == n &&
                   w.size(1) == k && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
               "w: every entry's bf16 [N, K], on a 16-byte boundary");
    weights.w[z] = w.data_ptr();
  }
  ZS_REQUIRE(k % 64 == 0 && m >= 1 && n >= 1 && m <= INT32_MAX && n <= INT32_MAX && k <= INT32_MAX,
             "K in whole steps of 64, the shape within int32");
  ZS_REQUIRE(x.defined() && x.dtype() == ZS_BF16 && x.dim() == 2 && x.is_contiguous() && x.size(1) == k &&
                 reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x_stride >= m * k && x_stride % 8 == 0 &&
                 x.numel() >= (count - 1) * x_stride + m * k,
             "x: bf16 [*, K], an entry's m rows every x_stride elements, on 16-byte boundaries");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_F32 && out.is_contiguous() && out_stride >= m * n &&
                 out.numel() >= (count - 1) * out_stride + m * n,
             "out: fp32, an entry's [m, N] every out_stride elements");
  ZS_REQUIRE(live.defined() && live.dtype() == ZS_I32 && live.dim() == 2 && live.is_contiguous() &&
                 live.size(0) >= count && live_column >= 0 && live_column < live.size(1),
             "live: int32 [count, *], the column read");
  const int M = static_cast<int>(m), N = static_cast<int>(n), K = static_cast<int>(k), Z = static_cast<int>(count);
  const int* counts = live.data_ptr<int>() + live_column;
  const int stride = static_cast<int>(live.size(1));
  float* d = out.data_ptr<float>();
  // tile 0's choice, as zs_b16_gemm makes it for these rows
  if (M <= 16) entries<Narrow<cutlass::bfloat16_t, float>>(x, x_stride, weights, d, out_stride, counts, stride, M, N, K, Z);
  else entries<Wide<cutlass::bfloat16_t, float>>(x, x_stride, weights, d, out_stride, counts, stride, M, N, K, Z);
  return zs::status();
}
