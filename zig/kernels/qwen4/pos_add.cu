// A later tile's window position: out = pos + add, both int32 [1]. One thread. A captured round fits in one
// tile of the selection and never launches it (attention.select). zspark's own; not from TensorFold.

#include <stdint.h>

#include "../zs/host.h"

namespace {

__global__ void pos_add_kernel(const int* __restrict__ pos, int* __restrict__ out, int add) { *out = *pos + add; }

}  // namespace

extern "C" int zs_qwen4_attn_pos_add(const zs_view* pos_, const zs_view* out_, int64_t add) {
  zs::begin();
  const zs::View pos(pos_), out(out_);
  ZS_REQUIRE(pos.defined() && pos.dtype() == ZS_I32 && pos.numel() == 1, "pos: int32 [1]");
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_I32 && out.numel() == 1, "out: int32 [1]");
  ZS_REQUIRE(add >= 0 && add <= INT32_MAX, "a tile's offset is not negative");
  pos_add_kernel<<<1, 1, 0, zs::stream()>>>(pos.data_ptr<int>(), out.data_ptr<int>(), static_cast<int>(add));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
