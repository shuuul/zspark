# `zig/kernels/glm5/` instructions

Extends `zig/kernels/AGENTS.md` and the root `AGENTS.md`. GLM-5.3-Flash's own kernels (spec 044).

## What an edit here must keep

- `glm5/`: a sum over a vector folds its upper half onto its lower (`fold`, `fold_shared`; four values are
  `(a + c) + (b + d)`), as spec 014 found Triton's `tl.sum`; a product added to a running sum is one `fmaf`,
  the chain's first product rounded alone. Both are zspark's reading of the Triton source until WS-07 holds
  them against exported tensors; `zig/src/models/glm5/layers/glue_ref.zig` is the same arithmetic on the host:
  edit the two together. The kernels are for four streams and a D that is a power of two to 4096.
- `glm5/hc.cu`, `mla_proj.cu`: `pre_lanes_kernel`, `mix_octs_kernel` (a window's `mix_rows_kernel`) and `expand_asked_kernel` (spec 073) are their
  bases' bits by other loads and lanes, behind `zs_glm5_hc_forms` and `zs_glm5_mla_ask`: a change to a base's
  arithmetic is made in its form too, and `hc_test` and `dsa_test` hold the pairs together. `kda.cu` and
  `kda_quads.cu` compile one chain and replay, `kda_common.cuh`, the state by scalars or by `float4` (`load4`,
  `store4`; `zs_glm5_kda_quads`).
- `glm5/kda.cu`, `kda_common.cuh`: an edit to their device code is a departure from TensorFold, said in its header; the replay keeps
  the chain's bits. A window of more than 48 rows runs `kda_chunk.cu` (spec 116 WS-06, FLA's `chunk_kda`): chunks
  of 64 rows on absolute positions, a row's bits its position's whatever the window's start (`phase`); its conv
  and gate arithmetic is `kda.cu`'s, an edit to one is made in the other, and its shared memory tables are in
  its header. `glm5/mla.cu` is `common/attention.cu`'s steps
  with the latent (512 values) as key and value: 32 keys a tile, the heads of one row as an mma's rows, the
  chunks by position; a block a row's 32 heads on one staged tile (69 KB); absorb and expand (`mla_proj.cu`, with the cache writes) are mma chains over k. `mla_q8.cu` (spec 114 WS-05) is
  its chunks over an FP8 cache in vLLM's FP8 form (`mla.cuh` what the three files share): an edit to the tiles, the softmax or
  the partials of one is made in the other. Both have a window form (`WHOLE`, spec 116 WS-01, `zs_glm5_mla_window`):
  a block a row's chunks in order, each folded by `mla.cuh` `fold_of`/`folded`, which `mla_merge_kernel` uses too,
  so the window and the chunk blocks and merge stay one set of bits (`dsa_test`); an edit to the fold is made there.
