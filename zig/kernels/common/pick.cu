// A scored row's start, found on the device: what a choice reads of a row of logits without reading it back.
// zspark's own (spec 028), written for Flash Next's MTP head and shared since GLM's ranks pick from their halves
// of the vocabulary (spec 045). Every value is a function of the one row.

#include <cuda_bf16.h>
#include <stdint.h>

#include "math.cuh"
#include "../zs/host.h"

namespace {

using common_math::bf16_low;

// What the host needs of a scored row to choose a draft from it without reading it back (spec 028): the row's
// first logits in the sampler's order, its extremes, and the sum a probability divides by. Four launches over one
// row of N bf16 logits, a block of PICK_THREADS threads over PICK_SPAN columns, into ``w``, uint32 (several rows
// in the same four launches, spec 144 WS-03: ``blockIdx.y`` the row, its logits N and its words S further on, each
// row's words those of a launch of it alone, as nothing a block reads or writes is another row's):
//   [0], [1]  the row's largest and smallest key
//   [2]       how many of the row's keys are at or above the cut, the key of rank [3]
//   [3]       read, not written: the rank the host asks to, 1 to PICK_MOST (clamped to that and to N); where
//             the launch is given ``asks``, the row's byte there instead
//   [4]       an fp32: the sum over columns of exp(v - largest) for v > largest - 24, as fp32 terms added a
//             thread, a warp, a block and then block by block in order; 0 where the largest key is not finite
//   [6]       the cut
//   [8 + 2j], [9 + 2j]  for j < min([2], PICK_SLOTS): a key at or above the cut and its column, in no order
//   the rest is the launches' own: PICK_HIGH and PICK_LOW the keys counted by high byte and, for the cut's high
//   byte, by low byte; PICK_TAKEN the slots taken; from PICK_BLOCKS each block's largest key, smallest key and sum.
// Where [2] is over PICK_SLOTS some keys at the cut were dropped, whichever came last: the host reads the row
// then. Otherwise every word is a count of integers, a set, or a sum in a fixed order, so no order of threads or
// blocks shows in what the host makes of them.
constexpr int PICK_THREADS = 256;
constexpr int PICK_EACH = 4;  // columns a thread reads: tid, tid + 256, ... of its block's span
constexpr int PICK_SPAN = PICK_THREADS * PICK_EACH;
constexpr int PICK_MOST = 64;
constexpr int PICK_SLOTS = 128;
constexpr int PICK_HIGH = 8 + 2 * PICK_SLOTS;
constexpr int PICK_LOW = PICK_HIGH + 256;
constexpr int PICK_TAKEN = PICK_LOW + 256;
constexpr int PICK_BLOCKS = PICK_TAKEN + 8;

// bf16 bits as the key the host's sampler orders by (sampling.zig `keyOf`): the values' order, -0 one below +0,
// the infinities just outside the finite keys 0x0080 to 0xff7f and the NaNs beyond.
__device__ __forceinline__ unsigned key_of(uint16_t bits) {
  return (bits & 0x8000u) ? (~static_cast<unsigned>(bits) & 0xffffu) : (bits | 0x8000u);
}
__device__ __forceinline__ float value_of(unsigned key) {
  const unsigned bits = key >= 0x8000u ? (key & 0x7fffu) : (~key & 0xffffu);
  return bf16_low(bits);
}
__device__ __forceinline__ unsigned warp_max(unsigned v) {
  for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ unsigned warp_min(unsigned v) {
  for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// The rank the host asks to, and the high byte that key has: the bins above it hold ``above`` keys.
__device__ __forceinline__ int cut_byte(const uint32_t* counts, unsigned want, unsigned& above) {
  int byte = 255;
  for (; byte > 0; --byte) {
    if (above + counts[byte] >= want) break;
    above += counts[byte];
  }
  return byte;
}
__device__ __forceinline__ unsigned wanted(const uint32_t* w, const uint8_t* asks, int N) {
  const unsigned most = N < PICK_MOST ? N : PICK_MOST;
  const unsigned asked = asks != nullptr ? static_cast<unsigned>(asks[blockIdx.y]) : w[3];
  return min(max(asked, 1u), most);
}

// Each block's largest and smallest key; block 0 also clears the counts the next launches add to.
__global__ void __launch_bounds__(PICK_THREADS)
    pick_scan(const uint16_t* __restrict__ row, uint32_t* __restrict__ w, int N, int NB, long long S,
              const uint8_t* __restrict__ asks) {
  row += static_cast<long long>(blockIdx.y) * N;  // a launch's row: its logits and its words
  w += static_cast<long long>(blockIdx.y) * S;
  __shared__ unsigned highs[PICK_THREADS / 32], lows[PICK_THREADS / 32];
  const int tid = threadIdx.x, blk = blockIdx.x;
  unsigned largest = 0, smallest = 0xffffu;
  for (int j = 0; j < PICK_EACH; ++j) {
    const long long i = (long long)blk * PICK_SPAN + j * PICK_THREADS + tid;
    if (i >= N) break;
    const unsigned key = key_of(row[i]);
    largest = max(largest, key);
    smallest = min(smallest, key);
  }
  largest = warp_max(largest);
  smallest = warp_min(smallest);
  if ((tid & 31) == 0) {
    highs[tid >> 5] = largest;
    lows[tid >> 5] = smallest;
  }
  if (blk == 0) {
    w[PICK_HIGH + tid] = 0;
    w[PICK_LOW + tid] = 0;
    if (tid == 0) w[PICK_TAKEN] = 0;
  }
  __syncthreads();
  if (tid == 0) {
    for (int k = 1; k < PICK_THREADS / 32; ++k) {
      largest = max(largest, highs[k]);
      smallest = min(smallest, lows[k]);
    }
    w[PICK_BLOCKS + blk] = largest;
    w[PICK_BLOCKS + NB + blk] = smallest;
  }
}

// The keys counted by high byte, and each block's sum of exp(v - largest).
__global__ void __launch_bounds__(PICK_THREADS)
    pick_count(const uint16_t* __restrict__ row, uint32_t* __restrict__ w, int N, int NB, long long S,
               const uint8_t* __restrict__ asks) {
  row += static_cast<long long>(blockIdx.y) * N;  // a launch's row: its logits and its words
  w += static_cast<long long>(blockIdx.y) * S;
  __shared__ unsigned hist[256];
  __shared__ unsigned highs[PICK_THREADS / 32];
  __shared__ float sums[PICK_THREADS / 32];
  __shared__ unsigned s_largest;
  const int tid = threadIdx.x, blk = blockIdx.x;
  hist[tid] = 0;
  unsigned largest = 0;
  for (int i = tid; i < NB; i += PICK_THREADS) largest = max(largest, w[PICK_BLOCKS + i]);
  largest = warp_max(largest);
  if ((tid & 31) == 0) highs[tid >> 5] = largest;
  __syncthreads();
  if (tid == 0) {
    for (int k = 1; k < PICK_THREADS / 32; ++k) largest = max(largest, highs[k]);
    s_largest = largest;
  }
  __syncthreads();
  largest = s_largest;
  const bool finite = largest >= 0x0080u && largest <= 0xff7fu;
  const float top = value_of(largest), floor = top - 24.0f;
  float sum = 0.f;
  for (int j = 0; j < PICK_EACH; ++j) {
    const long long i = (long long)blk * PICK_SPAN + j * PICK_THREADS + tid;
    if (i >= N) break;
    const unsigned key = key_of(row[i]);
    atomicAdd(&hist[key >> 8], 1u);
    const float v = value_of(key);
    if (finite && v > floor) sum += expf(v - top);
  }
  // a lane and the lane it meets add the same two values, so every lane of a warp ends with one sum
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if ((tid & 31) == 0) sums[tid >> 5] = sum;
  __syncthreads();
  if (tid == 0) {
    for (int k = 1; k < PICK_THREADS / 32; ++k) sum += sums[k];
    w[PICK_BLOCKS + 2 * NB + blk] = __float_as_uint(sum);
  }
  if (hist[tid] != 0) atomicAdd(&w[PICK_HIGH + tid], hist[tid]);
}

// The keys of the cut's high byte counted by low byte.
__global__ void __launch_bounds__(PICK_THREADS)
    pick_low(const uint16_t* __restrict__ row, uint32_t* __restrict__ w, int N, int NB, long long S,
             const uint8_t* __restrict__ asks) {
  row += static_cast<long long>(blockIdx.y) * N;  // a launch's row: its logits and its words
  w += static_cast<long long>(blockIdx.y) * S;
  __shared__ unsigned s_high;
  const int tid = threadIdx.x, blk = blockIdx.x;
  if (tid == 0) {
    unsigned above = 0;
    s_high = static_cast<unsigned>(cut_byte(w + PICK_HIGH, wanted(w, asks, N), above));
  }
  __syncthreads();
  const unsigned high = s_high;
  if ((w[PICK_BLOCKS + blk] >> 8) < high) return;  // none of this block's keys reaches it
  for (int j = 0; j < PICK_EACH; ++j) {
    const long long i = (long long)blk * PICK_SPAN + j * PICK_THREADS + tid;
    if (i >= N) break;
    const unsigned key = key_of(row[i]);
    if ((key >> 8) == high) atomicAdd(&w[PICK_LOW + (key & 0xffu)], 1u);
  }
}

// The cut, and every key at or above it with its column; block 0 writes the row's extremes and sum.
__global__ void __launch_bounds__(PICK_THREADS)
    pick_emit(const uint16_t* __restrict__ row, uint32_t* __restrict__ w, int N, int NB, long long S,
              const uint8_t* __restrict__ asks) {
  row += static_cast<long long>(blockIdx.y) * N;  // a launch's row: its logits and its words
  w += static_cast<long long>(blockIdx.y) * S;
  __shared__ unsigned s_cut;
  const int tid = threadIdx.x, blk = blockIdx.x;
  if (tid == 0) {
    const unsigned want = wanted(w, asks, N);
    unsigned above = 0;
    const int high = cut_byte(w + PICK_HIGH, want, above);
    const int low = cut_byte(w + PICK_LOW, want, above);
    s_cut = (static_cast<unsigned>(high) << 8) | static_cast<unsigned>(low);
    if (blk == 0) {
      unsigned largest = 0, smallest = 0xffffu;
      float sum = 0.f;
      for (int k = 0; k < NB; ++k) {
        largest = max(largest, w[PICK_BLOCKS + k]);
        smallest = min(smallest, w[PICK_BLOCKS + NB + k]);
        sum += __uint_as_float(w[PICK_BLOCKS + 2 * NB + k]);
      }
      w[0] = largest;
      w[1] = smallest;
      w[2] = above + w[PICK_LOW + low];
      w[4] = __float_as_uint(sum);
      w[6] = s_cut;
    }
  }
  __syncthreads();
  const unsigned cut = s_cut;
  if (w[PICK_BLOCKS + blk] < cut) return;
  for (int j = 0; j < PICK_EACH; ++j) {
    const long long i = (long long)blk * PICK_SPAN + j * PICK_THREADS + tid;
    if (i >= N) break;
    const unsigned key = key_of(row[i]);
    if (key < cut) continue;
    const unsigned slot = atomicAdd(&w[PICK_TAKEN], 1u);
    if (slot < PICK_SLOTS) {
      w[8 + 2 * slot] = key;
      w[9 + 2 * slot] = static_cast<uint32_t>(i);
    }
  }
}

// The four launches over `rows` rows of n logits a row apart into words `S` apart; `asks` null or u8 [rows].
int launch(const uint16_t* row, uint32_t* w, int n, int rows, long long S, const uint8_t* asks) {
  const int blocks = (n + PICK_SPAN - 1) / PICK_SPAN;
  const dim3 grid(blocks, rows);
  pick_scan<<<grid, PICK_THREADS, 0, zs::stream()>>>(row, w, n, blocks, S, asks);
  ZS_CUDA_LAUNCH_CHECK();
  pick_count<<<grid, PICK_THREADS, 0, zs::stream()>>>(row, w, n, blocks, S, asks);
  ZS_CUDA_LAUNCH_CHECK();
  pick_low<<<grid, PICK_THREADS, 0, zs::stream()>>>(row, w, n, blocks, S, asks);
  ZS_CUDA_LAUNCH_CHECK();
  pick_emit<<<grid, PICK_THREADS, 0, zs::stream()>>>(row, w, n, blocks, S, asks);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

}  // namespace

extern "C" {

// logits bf16, one row of N columns (any shape of N elements) -> out uint32 [536 + 3 ceil(N / 1024)], as the
// pick kernels say; out[3] is the caller's.
int zs_pick(const zs_view* logits_, const zs_view* out_) {
  zs::begin();
  const zs::View logits(logits_), out(out_);
  ZS_REQUIRE(logits.defined() && logits.dtype() == ZS_BF16 && logits.is_contiguous() && logits.numel() >= 1 &&
             logits.numel() <= (1 << 30), "logits: bf16, one row");
  const int n = static_cast<int>(logits.numel()), blocks = (n + PICK_SPAN - 1) / PICK_SPAN;
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_U32 && out.is_contiguous() &&
             out.numel() >= PICK_BLOCKS + 3 * blocks, "out: uint32 [536 + 3 ceil(N / 1024)]");
  return launch(logits.data_ptr<uint16_t>(), out.data_ptr<uint32_t>(), n, 1, 0, nullptr);
}

// zs_pick of each row of logits bf16 [rows, N] into the same row of out uint32 [rows, S >= 536 + 3 ceil(N /
// 1024)], in four launches for all of them (spec 144 WS-03); asks, u8 [rows] or null, the rank each row is cut
// at in place of its out[3].
int zs_pick_rows(const zs_view* logits_, const zs_view* out_, const zs_view* asks_) {
  zs::begin();
  const zs::View logits(logits_), out(out_), asks(asks_);
  ZS_REQUIRE(logits.defined() && logits.dtype() == ZS_BF16 && logits.dim() == 2 && logits.is_contiguous() &&
             logits.size(0) >= 1 && logits.size(0) <= 65535 && logits.size(1) >= 1 && logits.size(1) <= (1 << 30),
             "logits: bf16 [rows, N], contiguous, rows 1 to 65535");
  const int rows = static_cast<int>(logits.size(0)), n = static_cast<int>(logits.size(1));
  const int blocks = (n + PICK_SPAN - 1) / PICK_SPAN;
  ZS_REQUIRE(out.defined() && out.dtype() == ZS_U32 && out.dim() == 2 && out.is_contiguous() &&
             out.size(0) == rows && out.size(1) >= PICK_BLOCKS + 3 * blocks,
             "out: uint32 [rows, 536 + 3 ceil(N / 1024) or more]");
  ZS_REQUIRE(!asks.defined() || (asks.dtype() == ZS_U8 && asks.numel() == rows), "asks: u8 [rows]");
  return launch(logits.data_ptr<uint16_t>(), out.data_ptr<uint32_t>(), n, rows, out.size(1),
                asks.defined() ? asks.data_ptr<uint8_t>() : nullptr);
}

}  // extern "C"
