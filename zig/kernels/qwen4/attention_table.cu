// Spec 018's layout, the pool's (`--parallel` N): attention.cu's prep and common/attention.cu's attend kernels
// reaching the caches through a table of 512-position extents, a listed row's ids being the pool's rows that
// qsa_table.cu's selection wrote (zs_qwen4_attn_prep_table, zs_attn_attend_table). Every line is
// attention.cu's; ZS_EXTENTS there says what differs, which is where a position's row lies and nothing of the
// arithmetic. A second object, so that attention.cu's own stays, bit for bit, the one it was.

#define ZS_EXTENTS 2
#include "attention.cu"
#include "../common/attention.cu"
