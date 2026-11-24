// What GLM-5.3-Flash's DSA kernels share: `mla.cu`'s (the bf16 and FP8 caches' chunks, the merge),
// `mla_proj.cu`'s (absorb, expand, the cache writes) and `mla_q8.cu`'s (the FP8 cache's chunks with vLLM's FP8
// queries, spec 114 WS-05).
#pragma once

#include <cuda_runtime.h>

#include "../common/math.cuh"

namespace glm5_mla {

using namespace common_math;

constexpr int LW = 512;        // a latent's values
constexpr int CHUNK = 512;     // keys a chunk block attends
constexpr int TILE = 32;       // keys a step of the online softmax takes
constexpr int HEADS = 32;      // a chunk block's heads of one row: two groups of 16, an mma's rows
constexpr int THREADS = 256;   // its two groups of four warps: a warp 8 keys of a tile and a quarter of the latent
// An FP8 cache's token (`--kv-dtype fp8`, spec 107 WS-03), vLLM's GLM record (b12x `fp8_ds_mla`, 528 bytes): the
// latent's 512 values as E4M3 codes, then GROUPS fp32 scales, one a GROUP of 128 values, each its group's largest
// |value| / 448 (1 for a group of zeros). A code is its own token's alone, so a window's rows and a serial step
// write and read the same bytes.
constexpr int GROUP = 128;
constexpr int GROUPS = LW / GROUP;
constexpr int REC = LW + GROUPS * 4;

// Blocks of 256 threads for `count` threads.
inline unsigned blocks(long long count) { return static_cast<unsigned>((count + 255) / 256); }

// Logical token to the cache's row. A null table is the identity: position p is row p.
__device__ __forceinline__ int placed(const int* table, int tok) {
  if (!table) return tok;
  return table[tok >> 9] * 512 + (tok & 511);
}

// The keys a row attends: its list's count where the indexer gave it one, else every position up to its own.
__device__ __forceinline__ int keys_of(const int* pos, const int* counts, int r) {
  if (counts && counts[r] > 0) return counts[r];
  return pos[0] + r + 1;
}

// A line asked for and not waited for (`prefetch.global.L1`, `cuda-gb10`): no arithmetic sees it.
__device__ __forceinline__ void ask(const void* p) {
  asm volatile("prefetch.global.L1 [%0];\n" : : "l"(p) : "memory");
}

// The merge's step, `mla_merge_kernel`'s and a window kernel's (spec 116 WS-01): a chunk's m_c, l_c, o_c folded
// into a row's running m, l, o in chunk order from m = -inf, l = o = 0. A chunk of l_c > 0 is folded:
//   m' = max(m, m_c); a = exp(m - m') (0 where m is -inf); b = exp(m_c - m'); o = fma(o, a, o_c b), l alike.
// The pairing is written out (`__fmaf_rn`, `__fmul_rn`) as `mla_merge_kernel`'s object had it under nvcc's
// contraction (cuobjdump -sass, CUDA 13.0: FMUL o_c b, then FFMA o a + that): another object is not bound to
// contract `o * a + o_c * b` the same way.
struct Fold {
  float a, b, m;
};

__device__ __forceinline__ Fold fold_of(float m, float cm, float cl) {
  const bool active = cl > 0.f;
  const float next = active ? fmaxf(m, cm) : m;
  return {active ? (m == minus_inf() ? 0.f : expf(m - next)) : 1.f, active ? expf(cm - next) : 0.f, next};
}

__device__ __forceinline__ float folded(float o, float co, Fold w) { return __fmaf_rn(o, w.a, __fmul_rn(co, w.b)); }

// A line of a row's running output read for the last time: L2 drops it unwritten (`discard.global.L2`).
__device__ __forceinline__ void drop_line(const void* p) {
  asm volatile("discard.global.L2 [%0], 128;\n" : : "l"(p) : "memory");
}

// `mla_q8.cu`: the FP8 cache's chunk partials with FP8 queries (grid rows x head blocks x chunks), on `stream`.
// The arguments are `mla_chunk_kernel`'s; the caller checked them. With `out` (grid rows x head blocks) a block
// runs its row's chunks in order and folds each into the row's output as the merge does (`out` bf16 [R, H, LW]),
// po taken as fp32 [R, H, LW] the running output, pm and pl untouched.
void q8_chunks(dim3 grid, cudaStream_t stream, const void* qa, const unsigned char* cache, const int* pos,
               const int* table, float* po, float* pm, float* pl, const int* tokens, const int* counts, int width,
               float scale, int h, int nch, void* out);

}  // namespace glm5_mla
