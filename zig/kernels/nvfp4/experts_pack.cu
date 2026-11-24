// zspark's own (2026-10-20, the load's critical path): a layer's routed NVFP4 experts placed for the FP4 mma (or
// the A16 kernels) on the device, where the host did it at load (`weights/experts.zig` `packA4`, `pack`, `weights/csf_blocks.zig`
// `decodeWindow`). The bytes are the host's: both kernels only move bytes, and the host's functions stay as the
// tests' reference (`tests/models/glm5/session/pack_test.zig`, `tests/models/qwen4/model_test.zig`).
//
// The sources are the checkpoint's bytes as a load staged them in host memory (`safetensors.Stage`), which the
// GB10 reads through the host's page tables (`pageableMemoryAccess`), each expert's at its own address: a
// pointer table [experts, fields] of u64 in device memory. The checkpoint aligns its tensors to 2 bytes only, so
// every multi-byte read goes through `load_u32`, which reads the aligned words around it.
//
// zs_nvfp4_csf_decode: one thread a 32 x 2 block of a csf_32x2_v1 plane (rows a multiple of 128, columns of 4:
// whole tiles, the experts' planes), which validates it as `csf_blocks.validate` does (each block's records
// chained from the one before, a page's first at its base, records sorted on its columns, the bases at most
// 240) and writes its scales inside the window. A plane that fails sets `failed` to its call's mark.
//
// zs_nvfp4_experts_pack_a4: one warp a (32 columns, 64 inputs) block, a lane a column: its 32 code bytes and four
// scales into the block's 72 uint4 as `packA4` places them. A matrix's outputs may come from two places (down's,
// where each rank reads half of them and receives the other half from its peer).
//
// zs_nvfp4_experts_pack_a16: one warp a (32 columns, 32 inputs) block, lane (gq, t) its four code words of the n8
// tiles j (an output's inputs 4 t .. 4 t + 3 and 16 + 4 t .. of the step, nibbles interleaved as the bf16 mma's
// widening reads them) and lanes 0 to 15 a word of its 64 scales, as `pack` places them (2026-10-20, Flash Next's
// load: the same placement for its A16 experts, the same sources and split as `pack_a4`).
//
// zs_nvfp4_rows_gather: a rank's rows of the other's columns made contiguous, for the peer.
//
// Shared memory: none.

#include <stdint.h>

#include "../zs/host.h"

namespace {

constexpr int THREADS = 256;

// The little-endian word at p, which may sit on any byte.
__device__ __forceinline__ uint32_t load_u32(const uint8_t* p) {
  const uintptr_t at = reinterpret_cast<uintptr_t>(p);
  const uint32_t* base = reinterpret_cast<const uint32_t*>(at & ~uintptr_t(3));
  const uint32_t shift = 8 * (at & 3);
  if (shift == 0) return base[0];
  return __funnelshift_r(base[0], base[1], shift);
}

__device__ __forceinline__ uint32_t load_u16(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8; }

// A plane's fields in its row of the table.
enum { BASES, CODES, PAGE_BASES, OFFSETS, EXCEPTIONS, RECORDS, PLANE_FIELDS };

__global__ void __launch_bounds__(THREADS)
    csf_decode_kernel(const uint64_t* __restrict__ planes, int64_t experts, int rows, int cols, int wrow, int wrows,
                      int wcol, int wcols, uint8_t* __restrict__ out, uint32_t* __restrict__ failed, uint32_t mark) {
  const int block_rows = rows / 32, block_cols = cols / 2;
  const int64_t blocks = (int64_t)block_rows * block_cols;
  const int64_t id = (int64_t)blockIdx.x * THREADS + threadIdx.x;
  if (id >= experts * blocks) return;
  const int64_t e = id / blocks;
  const int64_t i = id - e * blocks;  // the block's place in the visit
  const uint64_t* plane = planes + e * PLANE_FIELDS;
  const uint8_t* bases = reinterpret_cast<const uint8_t*>(plane[BASES]);
  const uint8_t* codes = reinterpret_cast<const uint8_t*>(plane[CODES]);
  const uint8_t* page_bases = reinterpret_cast<const uint8_t*>(plane[PAGE_BASES]);
  const uint8_t* offsets = reinterpret_cast<const uint8_t*>(plane[OFFSETS]);
  const uint8_t* exceptions = reinterpret_cast<const uint8_t*>(plane[EXCEPTIONS]);
  const uint64_t records = plane[RECORDS];
  // whole tiles of 4 block rows by 2 block columns, a tile's blocks column by column, 32 tiles a page
  const int64_t tiles_across = block_cols / 2;
  const int64_t tile = i / 8, local = i % 8;
  const int br = (int)(tile / tiles_across) * 4 + (int)(local % 4);
  const int bc = (int)(tile % tiles_across) * 2 + (int)(local / 4);
  const int64_t page = tile / 32;
  const int64_t at = i + page;
  const uint32_t here = load_u16(offsets + 2 * at);
  const uint64_t first = load_u32(page_bases + 4 * page) / 2 + here;
  const uint32_t count = (load_u16(offsets + 2 * (at + 1)) - here) & 0xffff;
  bool bad = count > 64;
  if (i == 0 && first != 0) bad = true;
  if (tile % 32 == 0 && local == 0 && (here != 0 || (load_u32(page_bases + 4 * page) & 1))) bad = true;
  if (i + 1 < blocks) {
    const int64_t next = i + 1, next_page = (next / 8) / 32, next_at = next + next_page;
    const uint64_t next_first = load_u32(page_bases + 4 * next_page) / 2 + load_u16(offsets + 2 * next_at);
    if (!bad && next_first != first + count) bad = true;
  } else if (!bad && first + count != records) {
    bad = true;
  }
  if (bc == 0)
    for (int r = 0; r < 32; ++r)
      if (bases[br * 32 + r] > 240) bad = true;
  // the block's 64 scales: its row's base plus its offset, then its records over them
  // only a block the window meets is decoded; every block is validated
  const bool inside = br * 32 < wrow + wrows && (br + 1) * 32 > wrow && bc * 2 < wcol + wcols && (bc + 1) * 2 > wcol;
  uint8_t value[64];
  if (!bad && inside) {
    const uint8_t* block = codes + i * 32;
    for (int w = 0; w < 8; ++w) {
      const uint32_t word = load_u32(block + 4 * w);
      for (int b = 0; b < 8; ++b) {
        const int field = w * 8 + b;
        value[field] = bases[br * 32 + field / 2] + ((word >> (4 * b)) & 0xf);
      }
    }
  }
  if (!bad) {
    int last = -1;
    for (uint32_t k = 0; k < count && !bad; ++k) {
      const uint32_t record = load_u16(exceptions + 2 * (first + k));
      const int position = record & 0xff;
      if (position >= 64 || position <= last) {
        bad = true;
        break;
      }
      last = position;
      if (inside) value[position] = record >> 8;
    }
  }
  if (bad) {
    atomicCAS(failed, 0u, mark);
    return;
  }
  if (!inside) return;
  for (int r = 0; r < 32; ++r) {
    const int row = br * 32 + r;
    if (row < wrow || row >= wrow + wrows) continue;
    for (int c = 0; c < 2; ++c) {
      const int col = bc * 2 + c;
      if (col < wcol || col >= wcol + wcols) continue;
      out[(e * wrows + (row - wrow)) * (int64_t)wcols + (col - wcol)] = value[2 * r + c];
    }
  }
}

// A matrix's fields in its row of the table: the codes of its outputs below `split`, of those from `split` on (each
// from its own first output), its scales.
enum { MATRIX_CODES, MATRIX_HIGH, MATRIX_SCALES, MATRIX_FIELDS };

__global__ void __launch_bounds__(THREADS)
    pack_a4_kernel(const uint64_t* __restrict__ sources, int64_t experts, int n, int k, int split, int64_t code_stride,
                   int64_t high_stride, int64_t scale_stride, uint32_t* __restrict__ out, int per, int m) {
  const int nb = n / 32, kg = k / 64;
  const int64_t warp = ((int64_t)blockIdx.x * THREADS + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= experts * nb * kg) return;
  const int64_t e = warp / (nb * kg);
  const int rest = (int)(warp - e * nb * kg);
  const int cb = rest / kg, g = rest % kg;
  const int j = lane / 8, q = lane % 8;
  const int64_t column = (int64_t)cb * 32 + j * 8 + q;
  const uint64_t* source = sources + e * MATRIX_FIELDS;
  // a block's 32 outputs lie on one side of `split`, a multiple of 32
  const uint8_t* codes = column < split
                             ? reinterpret_cast<const uint8_t*>(source[MATRIX_CODES]) + column * code_stride
                             : reinterpret_cast<const uint8_t*>(source[MATRIX_HIGH]) + (column - split) * high_stride;
  codes += g * 32;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(source[MATRIX_SCALES]) + column * scale_stride + g * 4;
  uint32_t* block = out + ((((e * nb + cb) * kg + g) * per + m) * 288);
  uint32_t word[8];
  for (int w = 0; w < 8; ++w) word[w] = load_u32(codes + 4 * w);
  for (int c = 0; c < 4; ++c) {
    const int at = q * 4 + c;
    const int base = (j < 2 ? at : 32 + at) * 4 + (j & 1) * 2;
    block[base] = word[c];
    block[base + 1] = word[c + 4];
  }
  block[(64 + q) * 4 + j] = load_u32(scales);
}

__global__ void __launch_bounds__(THREADS)
    pack_a16_kernel(const uint64_t* __restrict__ sources, int64_t experts, int n, int k, int split, int64_t code_stride,
                    int64_t high_stride, int64_t scale_stride, uint32_t* __restrict__ out, int per, int m) {
  const int nb = n / 32, kg = k / 32;
  const int64_t warp = ((int64_t)blockIdx.x * THREADS + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= experts * nb * kg) return;
  const int64_t e = warp / (nb * kg);
  const int rest = (int)(warp - e * nb * kg);
  const int cb = rest / kg, g = rest % kg;
  const uint64_t* source = sources + e * MATRIX_FIELDS;
  uint32_t* block = out + ((((e * nb + cb) * kg + g) * per + m) * 144);
  // a block's 32 outputs lie on one side of `split`, a multiple of 32
  const int64_t first = (int64_t)cb * 32;
  const uint8_t* codes = first < split
                             ? reinterpret_cast<const uint8_t*>(source[MATRIX_CODES]) + first * code_stride
                             : reinterpret_cast<const uint8_t*>(source[MATRIX_HIGH]) + (first - split) * high_stride;
  const int64_t stride = first < split ? code_stride : high_stride;
  const int gq = lane / 4, t = lane % 4;
  uint32_t word[4];
  for (int j = 0; j < 4; ++j) {
    const uint8_t* row = codes + (j * 8 + gq) * stride + g * 16 + t * 2;
    uint32_t w = 0;
    for (int h = 0; h < 2; ++h) {
      // inputs q = 0 .. 3 of the half: bytes (q0 | q1 << 4), (q2 | q3 << 4) -> nibbles 2h, 2h + 4, 2h + 1, 2h + 5
      const uint32_t b0 = row[h * 8], b1 = row[h * 8 + 1];
      w |= (b0 & 0xf) << (4 * (2 * h)) | (b0 >> 4) << (4 * (2 * h + 4)) | (b1 & 0xf) << (4 * (2 * h + 1)) |
           (b1 >> 4) << (4 * (2 * h + 5));
    }
    word[j] = w;
  }
  reinterpret_cast<uint4*>(block)[lane] = make_uint4(word[0], word[1], word[2], word[3]);
  if (lane < 16) {
    // word w: (t, h, j / 2) = (w / 4, w / 2 % 2, w % 2), its bytes (j % 2, c), output 8 j + 2 t + c
    const int st = lane / 4, sh = (lane / 2) % 2, jh = lane % 2;
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(source[MATRIX_SCALES]);
    uint32_t w = 0;
    for (int b = 0; b < 4; ++b) {
      const int j = 2 * jh + b / 2, c = b % 2;
      const int64_t output = first + j * 8 + st * 2 + c;
      w |= uint32_t(scales[output * scale_stride + g * 2 + sh]) << (8 * b);
    }
    block[128 + lane] = w;
  }
}

// One thread a word of out [experts, rows, width]: row r of expert e from sources[e] + r * stride.
__global__ void __launch_bounds__(THREADS)
    rows_gather_kernel(const uint64_t* __restrict__ sources, int64_t experts, int64_t rows, int64_t width,
                       int64_t stride, uint32_t* __restrict__ out) {
  const int64_t words = width / 4;
  const int64_t id = (int64_t)blockIdx.x * THREADS + threadIdx.x;
  if (id >= experts * rows * words) return;
  const int64_t e = id / (rows * words);
  const int64_t r = (id / words) % rows;
  const int64_t w = id % words;
  out[id] = load_u32(reinterpret_cast<const uint8_t*>(sources[e]) + r * stride + 4 * w);
}

}  // namespace

// out [experts, wrows, wcols] e4m3 = each plane's scales in the window (wrow, wrows, wcol, wcols) of a [rows, cols]
// plane; planes i64 [experts, 6]: bases, codes, page_bases, offsets, exceptions (addresses the device reads) and
// the exceptions' records. failed u32 [1] is set to mark where a plane does not validate and was 0.
extern "C" int zs_nvfp4_csf_decode(const zs_view* planes_, int64_t rows, int64_t cols, int64_t wrow, int64_t wrows,
                                   int64_t wcol, int64_t wcols, const zs_view* out_, const zs_view* failed_,
                                   int64_t mark) {
  zs::begin();
  zs::View planes(planes_), out(out_), failed(failed_);
  ZS_REQUIRE(planes.defined() && planes.dtype() == ZS_I64 && planes.dim() == 2 && planes.size(1) == PLANE_FIELDS,
             "planes: i64 [experts, 6]");
  ZS_REQUIRE(rows > 0 && rows % 128 == 0 && cols > 0 && cols % 4 == 0, "a plane of whole tiles");
  ZS_REQUIRE(wrow >= 0 && wrows > 0 && wrow + wrows <= rows && wcol >= 0 && wcols > 0 && wcol + wcols <= cols,
             "a window inside the plane");
  const int64_t experts = planes.size(0);
  ZS_REQUIRE(out.defined() && out.is_contiguous() && out.element_size() == 1 && out.numel() == experts * wrows * wcols,
             "out: [experts, wrows, wcols] bytes");
  ZS_REQUIRE(failed.defined() && failed.dtype() == ZS_U32 && mark > 0, "failed: u32, a mark not 0");
  const int64_t threads = experts * (rows / 32) * (cols / 2);
  csf_decode_kernel<<<(unsigned)((threads + THREADS - 1) / THREADS), THREADS, 0, zs::stream()>>>(
      static_cast<const uint64_t*>(planes.data_ptr()), experts, (int)rows, (int)cols, (int)wrow, (int)wrows,
      (int)wcol, (int)wcols, static_cast<uint8_t*>(out.data_ptr()), static_cast<uint32_t*>(failed.data_ptr()),
      (uint32_t)mark);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// out i32 [experts, n/32, k/64, per, 288], matrix m of per: each expert's [n, k] codes (k/2 bytes a row) and e4m3
// scales (k/16 a row, scale_stride bytes from one row to the next) as `packA4` places them; sources i64 [experts, 3]:
// each expert's first code byte of output 0, of output `split` (a multiple of 32; n where every output is in the
// first), code_stride and high_stride bytes from one row to the next on either side, and its first scale byte:
// addresses the device reads.
extern "C" int zs_nvfp4_experts_pack_a4(const zs_view* sources_, int64_t n, int64_t k, int64_t split,
                                        int64_t code_stride, int64_t high_stride, int64_t scale_stride,
                                        const zs_view* out_, int64_t per, int64_t m) {
  zs::begin();
  zs::View sources(sources_), out(out_);
  ZS_REQUIRE(sources.defined() && sources.dtype() == ZS_I64 && sources.dim() == 2 && sources.size(1) == MATRIX_FIELDS,
             "sources: i64 [experts, 3]");
  ZS_REQUIRE(n > 0 && n % 32 == 0 && k > 0 && k % 64 == 0, "n in 32s, k in 64s");
  ZS_REQUIRE(split >= 0 && split <= n && split % 32 == 0, "split: outputs in 32s");
  ZS_REQUIRE(code_stride >= k / 2 && high_stride >= k / 2 && scale_stride >= k / 16, "rows at least a row apart");
  ZS_REQUIRE(per >= 1 && m >= 0 && m < per, "matrix m of per");
  const int64_t experts = sources.size(0);
  ZS_REQUIRE(out.defined() && out.is_contiguous() && out.element_size() == 4 &&
                 out.numel() == experts * (n / 32) * (k / 64) * per * 288,
             "out: i32 [experts, n/32, k/64, per, 288]");
  const int64_t threads = experts * (n / 32) * (k / 64) * 32;
  pack_a4_kernel<<<(unsigned)((threads + THREADS - 1) / THREADS), THREADS, 0, zs::stream()>>>(
      static_cast<const uint64_t*>(sources.data_ptr()), experts, (int)n, (int)k, (int)split, code_stride, high_stride,
      scale_stride, static_cast<uint32_t*>(out.data_ptr()), (int)per, (int)m);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// out i32 [experts, n/32, k/32, per, 144], matrix m of per: `zs_nvfp4_experts_pack_a4`'s sources, split and
// strides, placed as `pack` places them for the A16 kernels (`experts.cu`).
extern "C" int zs_nvfp4_experts_pack_a16(const zs_view* sources_, int64_t n, int64_t k, int64_t split,
                                         int64_t code_stride, int64_t high_stride, int64_t scale_stride,
                                         const zs_view* out_, int64_t per, int64_t m) {
  zs::begin();
  zs::View sources(sources_), out(out_);
  ZS_REQUIRE(sources.defined() && sources.dtype() == ZS_I64 && sources.dim() == 2 && sources.size(1) == MATRIX_FIELDS,
             "sources: i64 [experts, 3]");
  ZS_REQUIRE(n > 0 && n % 32 == 0 && k > 0 && k % 32 == 0, "n and k in 32s");
  ZS_REQUIRE(split >= 0 && split <= n && split % 32 == 0, "split: outputs in 32s");
  ZS_REQUIRE(code_stride >= k / 2 && high_stride >= k / 2 && scale_stride >= k / 16, "rows at least a row apart");
  ZS_REQUIRE(per >= 1 && m >= 0 && m < per, "matrix m of per");
  const int64_t experts = sources.size(0);
  ZS_REQUIRE(out.defined() && out.is_contiguous() && out.element_size() == 4 &&
                 out.numel() == experts * (n / 32) * (k / 32) * per * 144,
             "out: i32 [experts, n/32, k/32, per, 144]");
  const int64_t threads = experts * (n / 32) * (k / 32) * 32;
  pack_a16_kernel<<<(unsigned)((threads + THREADS - 1) / THREADS), THREADS, 0, zs::stream()>>>(
      static_cast<const uint64_t*>(sources.data_ptr()), experts, (int)n, (int)k, (int)split, code_stride, high_stride,
      scale_stride, static_cast<uint32_t*>(out.data_ptr()), (int)per, (int)m);
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}

// out u8 [experts, rows, width]: each expert's `rows` rows of `width` bytes (a multiple of 4), `stride` bytes apart
// from sources i64 [experts] on: a rank's part of a matrix, where the device reads it, made contiguous to be sent.
extern "C" int zs_nvfp4_rows_gather(const zs_view* sources_, int64_t rows, int64_t width, int64_t stride,
                                    const zs_view* out_) {
  zs::begin();
  zs::View sources(sources_), out(out_);
  ZS_REQUIRE(sources.defined() && sources.dtype() == ZS_I64 && sources.dim() == 1, "sources: i64 [experts]");
  ZS_REQUIRE(rows > 0 && width > 0 && width % 4 == 0 && stride >= width, "rows of whole words, a row apart");
  const int64_t experts = sources.size(0);
  ZS_REQUIRE(out.defined() && out.is_contiguous() && out.element_size() == 1 && out.numel() == experts * rows * width,
             "out: u8 [experts, rows, width]");
  const int64_t threads = experts * rows * (width / 4);
  rows_gather_kernel<<<(unsigned)((threads + THREADS - 1) / THREADS), THREADS, 0, zs::stream()>>>(
      static_cast<const uint64_t*>(sources.data_ptr()), experts, rows, width, stride,
      static_cast<uint32_t*>(out.data_ptr()));
  ZS_CUDA_LAUNCH_CHECK();
  return zs::status();
}
