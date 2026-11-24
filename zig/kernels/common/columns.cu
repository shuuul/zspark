// Rows of one buffer copied into a column range of another's rows: how a face whose parts are stored differently
// (block FP8 beside BF16) puts a part's result into its columns of the one output row. zspark's own, not taken from
// TensorFold, which does this with a tensor copy (cuda/nvfp4/linear.py, Concat). Hand-written because it is a copy,
// not arithmetic.

#include <stdint.h>

#include "../zs/host.h"

namespace {

template <class T>
__global__ void place_kernel(const T* __restrict__ src, T* __restrict__ dst, long long total, int width, int n,
                             int at) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const long long row = i / width;
  const int col = static_cast<int>(i % width);
  dst[row * n + at + col] = src[i];
}

template <class T>
void place(const zs::View& src, zs::View& dst, int64_t at) {
  const long long total = src.numel();
  place_kernel<T><<<static_cast<unsigned>((total + 255) / 256), 256, 0, zs::stream()>>>(
      src.data_ptr<T>(), dst.data_ptr<T>(), total, static_cast<int>(src.size(1)), static_cast<int>(dst.size(1)),
      static_cast<int>(at));
  ZS_CUDA_LAUNCH_CHECK();
}

}  // namespace

// dst [M, n] columns at .. at + w - 1 = src [M, w], bit for bit; both contiguous, of one dtype of 2 or 4 bytes.
extern "C" int zs_columns_place(const zs_view* src_, const zs_view* dst_, int64_t at) {
  zs::begin();
  const zs::View src(src_);
  zs::View dst(dst_);
  ZS_REQUIRE(src.defined() && src.dim() == 2 && src.is_contiguous() && src.size(0) >= 1 && src.size(1) >= 1,
             "src: [M, w]");
  ZS_REQUIRE(dst.defined() && dst.dim() == 2 && dst.is_contiguous() && dst.size(0) == src.size(0) &&
             dst.dtype() == src.dtype(), "dst: [M, n] of src's dtype");
  ZS_REQUIRE(at >= 0 && at + src.size(1) <= dst.size(1) && dst.size(1) <= INT32_MAX, "columns at .. at + w within n");
  const int64_t width = src.element_size();
  ZS_REQUIRE(width == 2 || width == 4, "elements of 2 or 4 bytes");
  if (width == 2) place<uint16_t>(src, dst, at);
  else place<uint32_t>(src, dst, at);
  return zs::status();
}
