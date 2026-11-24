// Spec 149 WS-02, the pool's (`--parallel` N) prompt rounds: common/attention_window.cu's attend, a row a block over
// its chunks, for a round's rows as attention_rows.cu's attend takes them (a row's stream, position and extent
// table; zs_attn_attend_window_rows). ZS_ROWS there says where a row looks, nothing of the arithmetic: a row's bits
// are zs_attn_attend_rows'. A second object, so attention_window.cu's own stays the one it was.

#define ZS_ROWS 1
#include "../common/attention_window.cu"
