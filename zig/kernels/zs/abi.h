// The C ABI between the nvcc-built launchers and zig/src/cuda/cuda.zig; keep the two in step: the entries here
// are those of cuda.zig's `kernels`. An entry one Zig file wraps is declared beside its wrapper instead.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum zs_dtype {
    ZS_U8 = 0,
    ZS_I32 = 1,
    ZS_F32 = 2,
    ZS_BF16 = 3,
    ZS_F16 = 4,
    ZS_E4M3 = 5,
    ZS_I64 = 6,
    ZS_U16 = 7,
    ZS_U32 = 8,
} zs_dtype;

// A device tensor as the launchers read it. Only the rows of a 2-D view may be strided.
typedef struct zs_view {
    void* ptr;
    int64_t size[6];
    int64_t row_stride;  // elements from one row to the next; 0: contiguous
    int32_t ndim;
    int32_t dtype;       // zs_dtype
} zs_view;

// Every launcher returns 0, or nonzero with the reason in zs_last_error() (this thread's, until its next call).
const char* zs_last_error(void);
// The stream this thread's launches go to (a cudaStream_t; null: the default stream).
void zs_set_stream(void* stream);

// out (M, N) bf16 or fp32 = x (M, K) bf16 times w [N, K] bf16, sk K slices summed in slice order (part fp32
// [sk, M, N] when sk > 1); without out the slices stay in part. tile: 0 by the row count, 1 narrow, 2 wide,
// 3 slim, 4 deep, 5 half (bf16.Tile). Tiles never change bits.
int zs_b16_gemm(const zs_view* x, const zs_view* w, const zs_view* out, const zs_view* part, int64_t sk,
                int64_t tile);
// count GEMMs of one shape in one launch (the pool's indexer GEMM a slot): entry z's out fp32 [m, N] at element
// z * out_stride = x bf16 [m, K] at element z * x_stride times w[z] bf16 [N, K], zs_b16_gemm's bits (one slice,
// tile 0); an entry whose live[z, live_column] (int32, read on the device) is not positive returns at once.
// count is 1 to 64.
int zs_b16_gemm_entries(const zs_view* x, int64_t m, int64_t x_stride, const zs_view* w, int64_t count,
                        const zs_view* out, int64_t out_stride, const zs_view* live, int64_t live_column);
// out bf16 (M, N) = act(x bf16 (M, K) @ w bf16 [N, K]^T + bias bf16 [N]); act 0 none, 1 GELU tanh, 2 exact
// GELU, 3 quick_gelu, each on the bias-added sum rounded to bf16 (bf16/gemm_bias.cu). K and N in whole 8s.
int zs_b16_gemm_bias(const zs_view* x, const zs_view* w, const zs_view* bias, const zs_view* out, int64_t act);
// h bf16 [P, W] += branch (rounded to bf16 and written back), then y = LayerNorm(h) by weight and bias bf16
// [W]; W a multiple of 8, at most 8,192 (common/layernorm.cu). No view but h need be given.
int zs_layernorm(const zs_view* h, const zs_view* branch, const zs_view* weight, const zs_view* bias,
                 const zs_view* y, double eps);
// ids int32 [R], table bf16 [V, D] -> out bf16 [R, streams * D], the row in each stream.
int zs_embed(const zs_view* ids, const zs_view* table, const zs_view* out, int64_t streams);
// features bf16 [N, D], pairs int32 [K, 2] (a row of out, its feature's row) -> those rows of out bf16
// [R, streams * D] in each stream: a prompt's placeholder rows take an encoder's features.
int zs_embed_rows(const zs_view* features, const zs_view* pairs, const zs_view* out, int64_t streams);
// dst [M, n] columns at .. at + w - 1 = src [M, w], bit for bit: a face's part into its columns of the output row.
int zs_columns_place(const zs_view* src, const zs_view* dst, int64_t at);
// zs_nvfp4_qmmf's mode 3 for 1 to 16 rows of a weight with one scale a (64-column tile, 64 inputs), ts fp32
// [npad/64, K/64]: the same bits; fuse: a block sums its tile's K slices itself (qmmf_tile.cu).
int zs_nvfp4_qmmf_tile(const zs_view* x, const zs_view* w, const zs_view* ts, const zs_view* out, int64_t n,
                       int64_t sk, int64_t npad, int fuse, int f32);
// The text attention's prep two families launch (zig/kernels/common/attention_prep.cu): q, k each over its norm or
// none, split-half RoPE at the device's position, k and v into the caches.
int zs_attn_prep(const zs_view* qkv, const zs_view* q_norm, const zs_view* k_norm, const zs_view* inv,
                 const zs_view* pos, const zs_view* q, const zs_view* kc, const zs_view* vc, double eps);
// The text attention two families launch (zig/kernels/common/attention.cu, attention_window.cu): q against a bf16
// or FP8 cache, a row's keys its own positions or its listed ids; the window's form a row a block, the same bits.
int zs_attn_attend(const zs_view* q, const zs_view* kc, const zs_view* vc, const zs_view* pos, const zs_view* po,
                   const zs_view* pm, const zs_view* pl, const zs_view* out, const zs_view* ids, const zs_view* nk,
                   const zs_view* sparse, const zs_view* qs, const zs_view* ks, const zs_view* vs, double scale,
                   int64_t context);
int zs_attn_attend_window(const zs_view* q, const zs_view* kc, const zs_view* vc, const zs_view* pos,
                          const zs_view* out, const zs_view* ids, const zs_view* nk, const zs_view* sparse,
                          const zs_view* qs, const zs_view* ks, const zs_view* vs, double scale, int64_t context);
// Flash Next's that two Zig files launch (zig/kernels/qwen4/; each file's comment says what it takes).
// attention_rows.cu: attention.cu's prep and common/attention.cu's attend for several streams in one round, a
// row's stream, position and extent table replacing the one sequence's.
int zs_qwen4_attn_prep_rows(const zs_view* sid, const zs_view* tables, const zs_view* bases, const zs_view* p,
                            const zs_view* pos, const zs_view* q_scale, const zs_view* k_scale,
                            const zs_view* iq_scale, const zs_view* inv_freq, const zs_view* q, const zs_view* kc,
                            const zs_view* vc, const zs_view* iq, const zs_view* ikc, const zs_view* qs,
                            const zs_view* ks, const zs_view* vs, double eps, int64_t context,
                            const zs_view* ropes, int64_t s1, int64_t s2);
int zs_attn_attend_rows(const zs_view* sid, const zs_view* tables, const zs_view* bases, const zs_view* q,
                        const zs_view* kc, const zs_view* vc, const zs_view* pos, const zs_view* po,
                        const zs_view* pm, const zs_view* pl, const zs_view* out, const zs_view* ids,
                        const zs_view* nk, const zs_view* sparse, const zs_view* qs, const zs_view* ks,
                        const zs_view* vs, double scale, int64_t context);
// attention_window_rows.cu: attention_window.cu's form of zs_attn_attend_rows, the same bits, no partials.
int zs_attn_attend_window_rows(const zs_view* sid, const zs_view* tables, const zs_view* bases, const zs_view* q,
                               const zs_view* kc, const zs_view* vc, const zs_view* pos, const zs_view* out,
                               const zs_view* ids, const zs_view* nk, const zs_view* sparse, const zs_view* qs,
                               const zs_view* ks, const zs_view* vs, double scale, int64_t context);
// qsa_rows.cu: the indexer's pooling for each slot, its matrix and extent table pointers in ptrs.
int zs_qwen4_attn_pool_slots(const zs_view* seg, const zs_view* ptrs, int64_t stride, int64_t pooled_col,
                             int64_t table_col, const zs_view* bases, const zs_view* ikc, const zs_view* scale,
                             const zs_view* inv_freq, double eps, int64_t ratio, int64_t blocks,
                             const zs_view* ropes, int64_t s1, int64_t s2);
// qsa_table.cu: qsa.cu's selection writing a listed row's keys as the pool's rows.
int zs_qwen4_attn_select_table(const zs_view* table, const zs_view* dots, const zs_view* pos, const zs_view* scores,
                               const zs_view* ids, const zs_view* nk, const zs_view* sparse, int64_t heads,
                               int64_t dim, int64_t ratio, int64_t top);
// and its window's selection at once so (spec 149 WS-02): zs_qwen4_attn_select_window's lists, the pool's rows.
int zs_qwen4_attn_select_window_table(const zs_view* table, const zs_view* iq, const zs_view* pooled,
                                      const zs_view* pos, const zs_view* scores, const zs_view* ids,
                                      const zs_view* nk, const zs_view* sparse, int64_t rows, int64_t heads,
                                      int64_t dim, int64_t nbl, int64_t ratio, int64_t top);
// ple.cu: the gathered table rows scaled into the embedding, and the gate with each stream's squares.
int zs_qwen4_ple_embed(const zs_view* v, const zs_view* out, int64_t heads, double scale);
int zs_qwen4_ple_gate(const zs_view* keys, const zs_view* vals, const zs_view* h, const zs_view* norm_key,
                      const zs_view* norm_query, const zs_view* gated, const zs_view* pss, double eps,
                      int64_t streams);

#ifdef __cplusplus
}
#endif
