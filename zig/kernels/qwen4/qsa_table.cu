// Spec 018's layout, the pool's (`--parallel` N): qsa.cu's pooling reading the indexer's
// keys through a table of 512-position extents, and its selection writing a listed row's keys as the pool's rows
// (zs_qwen4_attn_pool_table, zs_qwen4_attn_select_table). Every line is qsa.cu's; ZS_EXTENTS there says what
// differs. A second object, so that qsa.cu's own stays, bit for bit, the one it was.

#define ZS_EXTENTS 2
#include "qsa.cu"
