// Spec 018 WS-B3, the pool's (`--parallel` N): attention.cu's prep and common/attention.cu's attend for several
// streams in one round (zs_qwen4_attn_prep_rows, zs_attn_attend_rows). A row's stream, position and extent table
// replace the one sequence's. ZS_ROWS there says what differs, which is where a row looks, and nothing of the arithmetic.
// A second object, so attention.cu's own stays, bit for bit, the one it was.

#define ZS_EXTENTS 2
#define ZS_ROWS 1
#include "attention.cu"
#include "../common/attention.cu"
