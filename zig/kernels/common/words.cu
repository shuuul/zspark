// A few words from the host written into a buffer as a launch: the words travel in the launch's arguments, so the
// host's copy may change at once, the write is in the stream's order (no wait, unlike a synchronous copy from the
// host) and a capture records it with its values. On the GB10 the GPU reads pageable host memory through the host's
// page tables, so an asynchronous copy from it may read the source after the call returns: this does not. zspark's
// own, not taken from TensorFold (spec 114 WS-02).

#include <stdint.h>

#include "../zs/host.h"

namespace {

constexpr int most_words = 32;

struct Words {
  uint32_t w[most_words];
};

__global__ void words_kernel(uint32_t* __restrict__ dst, Words words, int n) {
  const int i = threadIdx.x;
  if (i < n) dst[i] = words.w[i];
}

}  // namespace

// dst's 4-byte words at .. at + n - 1 = words[0 .. n - 1]; dst contiguous, of a dtype of 4 bytes; n 1 to 32.
extern "C" int zs_words_set(const zs_view* dst_, int64_t at, const uint32_t* words, int64_t n) {
  zs::begin();
  zs::View dst(dst_);
  ZS_REQUIRE(dst.defined() && dst.is_contiguous() && dst.element_size() == 4, "dst: contiguous, 4-byte elements");
  ZS_REQUIRE(n >= 1 && n <= most_words && at >= 0 && at + n <= dst.numel(), "1 to 32 words within dst");
  Words held;
  for (int i = 0; i < n; ++i) held.w[i] = words[i];
  words_kernel<<<1, most_words, 0, zs::stream()>>>(static_cast<uint32_t*>(dst.data_ptr()) + at, held,
                                                    static_cast<int>(n));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
