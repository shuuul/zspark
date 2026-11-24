# Kernels

The tables describe the CUDA entry points used by zspark and the tests. An entry is an `extern "C"` function named `zs_*`. Its source file and the C ABI declarations give the launch contract. The [environment guide](environment.md) lists switches and defaults.

## `zs/`

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_last_error` | nothing (`host.cu`) | the calling thread's last error text | models using the shared ABI | none |
| `zs_set_stream` | nothing (`host.cu`) | the stream this thread's launches go to | models using the shared ABI | none |

## `common/`

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_columns_place` | `columns.cu` `place_kernel` | a face's part copied into its columns of the output row, where a face mixes storages (P's trellis runs wrote theirs themselves, spec 089 WS-03: 1.0 a chain launch for 4.4, traced) | any face that mixes storages | none |
| `zs_embed` | `embed.cu` `embed_kernel` | a token's embedding row into each stream | models using the shared ABI | none |
| `zs_embed_rows` | `embed.cu` `embed_rows_kernel` | a prompt's placeholder rows of the embedded streams take an encoder's features, every stream (Flash Next's image, spec 019; Qwen3-ASR's audio, spec 138); was `zs_qwen4_embed_image`; a copy, no shared memory | image placeholders; Qwen3-ASR prompts | none |
| `zs_pick` | `pick.cu` `pick_count`, `pick_low`, `pick_scan`, `pick_emit` | a scored row's start: what a choice needs of a row of logits, without reading it back | models using the shared ABI | none |
| `zs_pick_rows` | `pick.cu`, the same four kernels over `blockIdx.y` rows | `zs_pick` of every row of a round at once, each cut at its own rank (spec 144 WS-03, `decode/settle.zig` `launchRound`): a round's reply rows settled in one pass | pool; GLM's and Flash Next's pair settle (`gatherPicks`) | none |
| `zs_residual_add` | `glue.cu` `residual_kernel` | a branch added to one stream's rows (the ranks' fp32 partials, or one bf16 branch); was `zs_glm5_residual_add`, and `zs_unlimited_ocr_add` | GLM FP4-Local, Qwen3-ASR, Unlimited-OCR | none |
| `zs_residual_rmsnorm` | `glue.cu` `residual_rmsnorm_kernel` | a split matmul's K slices summed in slice order (`b16_reduce`'s), added to one stream's rows, then the next RMS norm, in one launch; `zs_residual_add` then `zs_rmsnorm`'s bits; a block of 1,024 threads a row, 16 KB shared (the row's squares). A bf16 branch [R, D] takes `residual_held_norm_kernel` instead (`glue.cuh`'s `norm_row`: 256 threads, the row in registers, 1 KB shared; moved from Unlimited-OCR's `moe.cu`, spec 142 WS-04) | Qwen3-ASR's language model, each add but the last; Unlimited-OCR's post-attention and dense adds | none |
| `zs_rmsnorm` | `glue.cu` `rmsnorm_kernel` | a row over its RMS, times a weight (bf16, or fp32 as stored); was `zs_glm5_rmsnorm` | GLM FP4-Local, Flash Next's DFlash2 | none |
| `zs_branch` | `glue.cu` `branch_kernel` | the ranks' partials added, rank 0 first; was `zs_glm5_branch` | GLM FP4-Local, Flash Next's DFlash2 | none |
| `zs_swiglu` | `glue.cu` `swiglu_kernel` | SwiGLU of a dense or shared MLP's gate and up, held to a limit; was `zs_glm5_swiglu` | GLM FP4-Local, Flash Next's DFlash2 | none |
| `zs_layernorm` | `layernorm.cu` `layernorm_kernel` | a pending branch added into h and written back, then LayerNorm with weight and bias, any width in 8s to 8,192, a block a row; 128 B shared (the warps' sums) | Qwen3-ASR's audio tower | none |
| `zs_experts_plan` | `plan.cu` `plan_kernel` | a call's picks grouped by expert, for the NVFP4 experts | FP4-Local | none |
| `zs_dflash2_prep` | `dflash2.cu` `dflash2_prep_kernel` | the drafter's q and k normed and rotated, v copied | GLM FP4-Local with the drafter | none |
| `zs_dflash2_dconv` | `dflash2.cu` `dconv_kernel` | the two-tap grouped dynamic convolution over a block's rows | GLM FP4-Local with the drafter | none |
| `zs_dflash2_top` | `dflash2.cu` `top_slice_kernel`, `top_merge_kernel` | a row's largest logits: slices of 4,096 columns a block, then a row's first of their candidates (spec 143 WS-09; was a block a row) | GLM FP4-Local with the drafter, Flash Next's DFlash2 | none |
| `zs_dflash2_rescore` | `dflash2.cu` `rescore_kernel` | the four-bit head's picks dotted again with the stored head's rows, one `fmaf` a value (`host.rescore` on the device); a block a pick, its two rows staged by a warp, lane 0 the sum (`rescore_staged_kernel`, spec 143 WS-09); rows past 12,288 values a thread a pick (`rescore_kernel`) | GLM FP4-Local with the drafter, Flash Next's DFlash2 | none |
| `zs_dflash2_attend` | `dflash2_attend.cu` `dflash2_chunk_kernel`, `dflash2_merge_kernel` | a block's attention over the context's keys and values | GLM FP4-Local with the drafter | none |
| `zs_dflash2_attend_parts` | nothing | the scratch `attend` needs, in values | GLM FP4-Local with the drafter | none |

## `bf16/`

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_b16_gemm` | `gemm.cu`: CUTLASS `GemmUniversal` (`Kernel2`), then `reduce.cuh` `b16_reduce` where the weight splits K and `out` is given; a window's 128-row tiles sum the slices inside (serial split-K, a memset of its semaphores before it, `b16_reduce` of one slice after it for a bf16 `out`) | bf16 rows times a bf16 weight, K slices summed in slice order | models using the shared ABI | `ZSPARK_GEMM_TILES` (1) picks the tile |
| `zs_b16_gemm_entries` | `gemm.cu` `b16_gemm_entries_kernel`: CUTLASS `GemmUniversal` in batched mode on `zs_b16_gemm`'s tile 0 (narrow to 16 rows, wide above), Kernel2's body behind a device check | GEMMs of one shape a weight each, fp32, an entry whose count in a device table is not positive returning at once; a running entry `zs_b16_gemm`'s bits | the pool's captured selection (`rounds_select.selectFixed`, spec 144 port 5) | none |
| `zs_b16_gemm_bias` | `gemm_bias.cu`: CUTLASS `GemmUniversal` (`Kernel2`), 128 x 128 x 32, 3 stages, the bias as C with a row stride of 0 | bf16 rows times a bf16 weight plus its bias, rounded, then GELU (tanh or exact) or quick_gelu (Unlimited-OCR's CLIP, spec 137) of the rounded sum: the vision towers' linears, one tile at every row count | image and OCR processing | none |
| `zs_b16_gemm_bias_residual` | `gemm_bias.cu`: CUTLASS `GemmUniversalWithBroadcast`, `zs_b16_gemm_bias`'s tile and mainloop, the bias as the broadcast vector, h as C and D in place | h = bf(h + bf(x W^T + bias)): a linear and the residual add after it in its epilogue, `zs_b16_gemm_bias` then the add to the bit, its output never written | SAM's proj and lin2 (Unlimited-OCR's tower, spec 142 WS-08) | none |
| `zs_bf16_experts_up` | `experts.cu` `bf16_experts_kernel<UP>`: a 64 x 64 x 64 `m16n8k16` tile, 4 warps, two `cp.async` stages, 32 KB | the routed experts' gate and up products of their rows grouped by `common/plan.cu`'s plan, SwiGLU in the epilogue, bf16 act at each pair's plan place | Unlimited-OCR's decoder | none |
| `zs_bf16_experts_down` | `experts.cu` `bf16_experts_kernel<DOWN>` | the down product of each pair's act, bf16, written at the pair's row | Unlimited-OCR's decoder | none |
| `zs_b16_matmul` | `zig/tests/kernels/bf16/matmul.cu` `b16_kernel` | `zs_b16_gemm`'s product by hand, the same bits | tests (`bf16_test.zig`), linked into the test binaries only | none |

The tiles of `zs_b16_gemm` (`weights/bf16.zig` `Tile.of`): up to 16 rows the slim tile on a
weight split over K, the deep tile on a weight of one slice whose input is 1,024 wide or more, the narrow
(base) tile on a one-slice weight under 1,024 wide (Flash Next's read-out up); a window on the half tile from
2,048 rows, from 1,024 but for a weight in 32 slices, from 256 for a one-slice weight L2 holds, else on the
wide tile.

A window's K slices (spec 095 WS-03, from `bf16.sums_from`, 1,024 rows): the wide and half tiles run as
CUTLASS's serial split-K. Slice z's block walks the K range of the parallel split from zero, waits on its tile's
semaphore and stores `fma(1, acc_z, D)` into an fp32 D (slice 0: `1 * acc_z`), which is `b16_slices`' sum in
slice order: the parallel split's bits, tested at 1 to 2,048 rows on 2 to 32 slices. D is `out` itself when
it is fp32 (the read-out's and the router's `slices` then hand their reader one slice), else `part`'s first
rows x N, rounded to bf16 by `b16_reduce` of one slice. The semaphores, an `int` a 128 x 128 tile, sit in `part`
after D; CUTLASS zeroes them each call (`init_workspace`). Shared memory: the tile's own, unchanged (wide
98,304 B, half 49,152 B, the epilogue's staging in the same union); D waits in L2 (2.6 MB for the read-out at
2,048 rows, where the parallel split wrote 84 MB of slices). What could be less: none in the block; the
memset (a launch a call) could go, since the last slice leaves its semaphore at 0, if the semaphores had a
buffer of their own. Why 1,024: the chain is SK epilogues long whatever the rows, so under it the parallel
split's slices are cheaper (ncu, the read-out's 32 slices 52 to 136 us at 64 rows, 191 to 177 at 512, 354 to
250 at 1,024, 674 to 353 at 2,048; the router's 8 slices 64 to 90 at 512, 144 to 124 at 1,024).

`b16_slices` (`reduce.cuh`; `b16_reduce`, the read-out's `act_kernel`, the routers' selections read it) loads
a thread's slices eight at a time, then adds them in slice order: an add after each load
kept two loads in flight, a serial chain of L2 latencies. The same adds, so the same bits (slice 0 taken as
it is, a -0 keeps its sign); two slices are a load and an add as before. Standalone, after a writer that leaves
the slices in L2: 32 slices at 1 to 16 rows of the read-out 3.4 to 2.0 us, 8 slices 1.8 to 1.3, 4 slices 1.6
to 1.3; `b16_reduce` of 2,560 columns in 8 slices 1.6 to 1.1-1.3. No shared memory.

## `nvfp4/`

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_nvfp4_qmmf` | `qmmf.cu` `qmmf_kernel` (with `reduce_kernel`) | bf16 rows times an NVFP4, FP8 or MXFP8 weight as stored (the lane matmul) | the quantized draft head; the drafter's linears; FP4-Local's MXFP8 and FP8_PB_WO linears and its MXFP8 head, FP8_PB_WO above 16 rows. FP4-Local's 384 MXFP8 linears (mode 2), decode rows and prompt windows alike (spec 105: 14.3 ms of a 37.9 ms serial step, spark-a `scratch-105/ws02/trace4`, not one of the three traces). A K-split weight of 40 tiles or more at 1 to 32 rows sums its slices in the block, not a cluster: the same bits, 2560 x 6144 MXFP8 70 us against 83, NVFP4 46 against 56 (spec 105 WS-03, `nvfp4.lane_fused_tiles`). An MXFP8 window of 256 rows or more takes the 128-row tile (bm 128, a warp's decoded weights for 128 rows): the same bits, 9 to 18% less a launch at 2,048 rows, a 16k cold prompt 2.6% (spec 105 WS-03, `nvfp4.lane_wide_rows`). Fixture: block FP8 with `ZSPARK_FP8_TILES=0`. GLM's since spec 114 WS-04 take `zs_nvfp4_qmm_a8` | none |
| `zs_nvfp4_qmmf_tile` | `qmmf_tile.cu` `qmmt_kernel` | a block-FP8 weight with one scale a tile, 1 to 16 rows, `qmmf`'s bits | GLM FP4-Local's FP8_PB_WO linears (MLA, the shared experts) at 1 to 16 rows until spec 114 WS-04 | `ZSPARK_FP8_TILES` (16) |
| `zs_nvfp4_qmm_a8` | `qmm_a8.cu` `qmma8_kernel`, 4 warps and 3 stages of 128 inputs in both forms: 1 to 16 rows bm 16 (quantizes in the block, 39,744 B), more rows `quant_a8_kernel` then bm 64 (49,920 B); two blocks an SM, K slices by shape (a cluster, or in the block from 32 tiles) | bf16 rows quantized as the card's vLLM quantizes them (MXFP8: e4m3 under an e8m0 a 32, b12x; block FP8: e4m3 under an fp32 a token's 128, `per_token_group_fp8_quant`), times an MXFP8 or block-FP8 weight as stored on `mma.sync m16n8k32 kind::mxf8f6f4.block_scale` (W8A8) | GLM FP4-Local: every MXFP8 and FP8_PB_WO linear of FP4-Local (KDA, the dense MLPs, MLA, the shared experts, the MTP layer's, the indexer's) and its MXFP8 head, every row count (spec 114 WS-04, `nvfp4.Activations`; `share.faceOf`). P: every MXFP8 linear of Flash Next's trunk, the shared expert's gate and up and down included, in prompt windows of more than 80 rows (spec 143 WS-06 and WS-08, `Layer.quantizePrompts`); DeltaNet's b and a (`in_proj_ba`, the projection's last 64-column tiles) keep bf16 rows on the lane matmul (`Mx8Linear.keepBf16From`). A 16k cold prompt at 2,048-row windows: 1.25 s of its 7.0 s of kernels; DeltaNet's in and out projections (2,560 by 16,512, 6,144 by 2,560) at 98 to 102 TFLOPS, the shared expert's (2,560 by 1,280, 640 by 2,560) 31 to 46 (spec 148 WS-01, `nsys`) | none |
| `zs_nvfp4_experts` | `experts.cu` `nvfp4_expert_kernel`; the decode gate and up with every half unit on a warp `experts_half.cu` `nvfp4_expert_half` (two warps a unit, TF 0afe9dd5); from 768 rows `experts_wide.cu` `nvfp4_expert_wide_kernel`, from 1024 `experts_tile.cu` `nvfp4_expert_tile_kernel` (105 WS-03: 23% less at 2048 rows alone; 107 WS-03: the e4m3 scale folded into the bf16 weights before the mma, as b12x does, a departure from TensorFold, 4 to 5% less at 2048 rows, 11 to 12% at 4096) | a layer's routed NVFP4 experts on a plan: gate and up with SwiGLU, or one up matrix with relu squared (epilogue 1, Nemotron's, spec 151), then down | GLM FP4-Local's MTP layer's experts (`W4A16_NVFP4`), at the rank's half width with GLM's clip (spec 107 WS-02; its routed experts are W4A4 since spec 114 WS-03, below) | `ZSPARK_EVEN_EXPERTS` (1): the units in whole rounds over the warps, and the half dispatch |
| `zs_nvfp4_csf_decode`, `zs_nvfp4_experts_pack_a4`, `zs_nvfp4_rows_gather` | `experts_pack.cu` `csf_decode_kernel` (a thread a csf block), `pack_a4_kernel` (a warp a block), `rows_gather_kernel` | at load: a W4A4 layer's csf_32x2_v1 scales validated and decoded, its blocks placed as `experts.packA4` does, read from host memory through its page tables; down's peer inputs gathered to trade (`experts.Swap`); bytes the host's (`session/pack_test.zig`) | GLM FP4-Local's load, layers 3 to 44 (`experts.Experts.initPlaced`) | none |
| `zs_nvfp4_experts_quant` | `experts_quant.cu` `nvfp4_experts_quant_kernel` | a W4A4 call's rows: each routed pair's bf16 row of X quantized to NVFP4 under the layer's gate and up global scale (1 over its experts' largest `input_scale`, the card's `a1.amin()`, spec 115), as b12x's `quantize_block_fp4` (spec 114 WS-03, zspark's own), into the plan's member order; down's rows are gate and up's epilogue's since WS-07 | GLM FP4-Local's routed experts of layers 3 to 44 (the map's `NVFP4`), before gate and up | none |
| `zs_nvfp4_linear_quant` | `experts_quant.cu` `nvfp4_linear_quant_kernel`, one thread a 16-input group, no shared memory | checkpoint NVFP4 activation rounding, normalized FP4 times e4m3 block scales widened exactly to bf16; the global factor stays in the linear GEMM's alpha | Nemotron backbone attention q/k/v and o, every window | none |
| `zs_nvfp4_experts_a4` | `experts_a4.cu` `nvfp4_expert_a4_kernel` (16 pairs a reading); from 256 rows `experts_tile_a4.cu` `nvfp4_expert_tile_a4_kernel` (64 a reading), from 1,024 its 16-warp form (128 a reading, one block an SM) | the routed experts W4A4: NVFP4 rows times NVFP4 weights on `mma.m16n8k64 kind::mxf4nvf4` with their e4m3 scales, times alpha = weight_scale_2 / gs; gate and up's epilogue makes down's NVFP4 rows under its expert's scale from g and u rounded to bf16 and clipped, silu (`ex2.approx`, `rcp.approx`) and the product each rounded, the card's `prefill_a4` (`swiglu_card`, WS-07, spec 115); a quantized window's (GLM FP4-Local's of more than 48 rows), down's is `zs_nvfp4_experts`'; weights packed for the instruction (`experts.packA4`) | as `zs_nvfp4_experts_quant` | none |
| `zs_nvfp4_experts_a16` | `experts_a4.cu` `nvfp4_expert_a16_kernel` (16 pairs a reading) | the routed experts A16 on the W4A4 pack: bf16 rows times the `packA4` codes widened to bf16 in registers (`fp4pair`, `scale_pair`), `mma.m16n8k16`, times `weight_scale_2`; epilogue 5 writes gate and up's SwiGLU in fp32, clipped, rounded to bf16 once at its pair's row, 0 and 3 down's sums | GLM FP4-Local's routed experts in a bf16 window (a step, a verify window, a prompt window of up to 48 rows), the card's captured A16 | none |
| `zs_nvfp4_prompt16` | `prompt.cu` `prompt_kernel` | prompt rows times an NVFP4, FP8 or MXFP8 weight | no Flash Next checkpoint (WS-04; FP4-Local's MXFP8 windows take `qmmf`, whose bits are a row's at any count); the fixture's prompt rows | none |
| `zs_nvfp4_prompt_bias` | `prompt_bias.cu` `prompt_bias_kernel`, tile 4 (128 x 128, 4 warps, 2 stages: 41,984 B NVFP4, 49,664 MXFP8, two blocks an SM) | bf16 rows times an NVFP4 or MXFP8 weight as stored, `prompt.cu`'s chain, then `gemm_bias.cu`'s epilogue: the bias added before the one rounding, GELU (tanh or exact) of the rounded sum | a quantized vision tower's linears (MXFP8 qkv and proj, NVFP4 fc1, fc2 and the merger's two; spec 019 WS-08) | none |

Nemotron's Mamba projections also reach `zs_nvfp4_qmm_a8` with their stored static FP8 activation
scale; zero selects the dynamic MX/block-FP8 behavior above. Its relu2 backbone experts reach
`zs_nvfp4_experts_a4` in every window, with separate routed/shared input scales and a zero-padded
down K for TP2's width 928. These spec 151 precision paths have arithmetic and row-independence
checks; no speed measurements are claimed.
`Fp4Linear` borrows activation scratch by capacity and gives the quantizer a dense `[rows, K]`
view; an allocation reused from a larger linear must not supply its original shape.


## `qwen4/`

Flash Next's own kernels, on both of its checkpoints. `*_table`, `*_rows` and `*_slots` are second objects of
the same device code: no trace has the pool's rounds, so each is "not measured".

The hyper-connections, the MoE's glue, the state:

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_qwen4_hc_site` | `hc.cu` `site_kernel` | a site's write-back and the norm after it in one launch (modes: none, a branch, the MoE's slots, the MoE's sum) | Flash Next FP4-Local | `ZSPARK_HC_JOINED` (on); mode 3 was `ZSPARK_MOE_SUM`'s (removed, spec 121) |
| `zs_qwen4_hc_writeback` | `hc.cu` `writeback_kernel` | a branch added into the streams with its inject gates | base; the pool; where a site does not join | the base of `site` |
| `zs_qwen4_hc_normed` | `hc.cu` `normed_kernel` | the streams over their RMS, times the scale | as above | the base of `site` |
| `zs_qwen4_hc_act` | `hc.cu` `act_kernel` | the down projection's K slices summed, the activation, the inject gates | Flash Next FP4-Local | none |
| `zs_qwen4_hc_mix` | `hc.cu` `mix_kernel` | the up projection mixed with the normed streams into one row | Flash Next FP4-Local | none |
| `zs_qwen4_hc_up_mix_window` | `hc_window.cu` `up_mix_window_kernel` | the bf16 up product and `mix` in one launch at a prompt window's rows, the product never written; the GEMM's and `mix`'s bits. Shared memory: act and weight, 128 rows by 32 of K each, 80 B a row, 20,480 B a stage, 2 stages (40,960 B, two blocks an SM); `normed` to registers in the epilogue, prefetched into L2 | Flash Next FP4-Local, windows of 128 rows and more | none (`hc.window_rows`) |
| `zs_qwen4_hc_down_window` | `hc_down.cu` `down_window_kernel` | the streams normed in the down tile's shared memory, the down product over the whole K with its K slices folded in order, the activation and inject gates in the epilogue: `normed`, the GEMM and `act` in one launch, their bits. Shared memory: h 64 rows and the weight 112 by 32 of K, 80 B a row, a stage's 32 scale values, 14,208 B a stage, 3 stages, 1,024 B of 1 / rms (43,648 B, two blocks an SM) | Flash Next FP4-Local, windows of 512 rows and more | `ZSPARK_HC_FUSED` (on; `hc.fused_rows`) |
| `zs_qwen4_hc_up_mix_streams` | `hc_window.cu` `up_mix_window_kernel` (its streams form) | `zs_qwen4_hc_up_mix_window` with the streams in place of the normed rows, normed in the epilogue (the rows' 1 / rms in 2,048 B after the stages: 43,008 B, two blocks an SM); `normed`'s, the GEMM's and `mix`'s bits | as above | as above |
| `zs_qwen4_moe_topk` | `moe.cu` `topk_kernel` | the router's slices summed, the top experts and their weights | Flash Next FP4-Local | none |
| `zs_qwen4_moe_shared_act` | `moe.cu` `shared_act_kernel` | the shared expert's SwiGLU where it is a bf16 linear | Flash Next FP4-Local | none |
| `zs_qwen4_gdn_chain` | `gdn.cu` `chain_kernel` | DeltaNet's chain of a window's rows from the state, under 32 rows; first it replays in registers the rows the rounds before kept in part (`behind`, read from the device; spec 089 WS-03: 30.4 us a launch to 36.4 with them, traced), eight rows' saves staged in shared memory at a time (spec 096 WS-02: 1.9 us a row to 0.9, alone); a row's four k.state and four q.state sums in one butterfly each (spec 143 WS-05, upstream 995f98af: 101.8 to 89.5 ms over a traced decode's 2,664 launches, 2.1 to 1.6 us a row). It writes the state only for a row's window (`forward.stores`): a window of drafts is kept whole a round in five, and its 3 MB written for nothing cost `out_proj` after it 16 to 18 us (dirty lines; spec 096 WS-02), so its rows wait behind too | Flash Next FP4-Local | `ZSPARK_GDN_QUADS` (on) |
| `zs_qwen4_gdn_replay` | `gdn.cu` `replay_kernel` | the state after rows kept in part, for a state read meanwhile (`forward.committed`) and a pool's selected states (`keepWithin`); a landing is `zs_qwen4_gdn_land`'s; the chain's staged replay | Flash Next FP4-Local, outside the graph | `ZSPARK_GDN_QUADS` (on) |
| `zs_qwen4_gdn_land` | `gdn.cu` `land_kernel` | a landing (`forward.lands`, a pool's slot in `rounds.stage`): every DeltaNet layer of every stream that lands, `replay`'s bits, one launch for 64 states; a head's 128 rows split over 4 blocks of 256 threads (a row's update reads its own dot alone), four blocks an SM at 54 to 56 registers. Was a `replay_kernel` a layer, 36 in series, 40 us each on a rank (spec 146 WS-02: 1.44 ms a landing, 0.59 / 0.20 ms a round at 4 / 1 streams) | Flash Next FP4-Local, pool, outside the graph | `ZSPARK_GDN_QUADS` (on) |
| `zs_qwen4_gdn_quads` | nothing | sets the two above and the two `_slots` below to rows of four floats and butterfly sums | Flash Next FP4-Local, pool | `ZSPARK_GDN_QUADS` |
| `zs_qwen4_gdn_chain_wide` | `gdn_wide.cu` `front_kernel`, `wide_chain_kernel`, `back_kernel` | the chain of a window of 32 rows and more, the narrow chain's bits | Flash Next FP4-Local, a prompt's windows | none (`gdn.wide_rows`) |
| `zs_qwen4_gdn_replay_wide` | `gdn_wide.cu` `wide_chain_kernel` | the replay through the wide chain's layout | Flash Next FP4-Local, windows of 32 rows and more | none |
| `zs_qwen4_gdn_chain_slots` | `gdn_slots.cu` `chain_slots_kernel` | one chain a slot of a shared round, both row forms in one launch; first it replays in registers the slot's deferred rows, and a window of drafts (2 to 31 rows) saves its rows after them into the slot's own scratch and writes no state, as a session's chain (`behind`) | pool | `ZSPARK_GDN_QUADS` (on) |
| `zs_qwen4_gdn_replay_slots` | `gdn_slots.cu` `replay_slots_kernel` | each slot's state after its kept rows, a window the chain deferred left (spec 144 WS-09: only a window of 32 rows or more kept in part is left to it) | pool, outside the graph | `ZSPARK_GDN_QUADS` (on) |
| `zs_qwen4_shift_each` | `state.cu` `shift_each_kernel` | a commit's conv windows of every DeltaNet layer shifted in one launch | Flash Next FP4-Local | `ZSPARK_SHIFT_ONCE` (on) |
| `zs_qwen4_shift_windows` | `state.cu` `shift_kernel` | the same shift, a launch a call | base; the calls `once` does not take | the base of `shift_each` |
| `zs_qwen4_shift_slots` | `state_slots.cu` `shift_slots_kernel` | every slot's window shifted in one launch | pool | none |
| `zs_qwen4_shift_slots_each` | `state_slots.cu` `shift_slots_each_kernel` | every slot's conv window of every DeltaNet layer shifted in one launch, where no wide segment shares one projection | pool | none |

The attention, the selection, PLE and the MTP head:

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_qwen4_attn_prep` | `attention.cu` `attn_prep_kernel` | q, k, v normed and rotated, the keys and values written to the caches; an image prompt's rows rotate by their t, h and w (`attention_rope.cuh`, spec 019) | Flash Next FP4-Local | none |
| `zs_attn_prep` | `common/attention_prep.cu` `attn_prep_split_kernel` | the q, k and v of each row: q and k over their per-head RMSNorm where the model has one (HF's roundings), split-half RoPE from the device position (cos and sin bf16), or none where `inv` is null (NemotronH's NoPE, spec 151), q to its rows, k and v into the bf16 caches; a block a row and head, D / 2 threads, 16 B of shared memory. Was `zs_qwen3_asr_attn_prep` and `zs_unlimited_ocr_attn_prep`, the same bits. `zs_attn_prep_page2d` is the same kernel without norms, turned by page2d RoPE at a row's coordinates (int32 [R, 3], token and page; Unlimited-OCR-2, spec 147 WS-05) | Qwen3-ASR's language model, Unlimited-OCR's decoder, Nemotron's attention (tests only so far): every window and step | none |
| `zs_attn_attend` | `common/attention.cu` `attn_chunk_kernel`, `attn_merge_kernel` | a row's online softmax over its keys, bf16, FP8 or FP8 keys with NVFP4 values (spec 138 moved it to `common/`: Qwen3-ASR's language model launches it too) | Flash Next FP4-Local | the base of `ZSPARK_ATTN_WINDOW` |
| `zs_qwen4_attn_gate` | `attention.cu` `attn_gate_kernel` | the heads' outputs gated by the stacked projection | Flash Next FP4-Local | none |
| `zs_attn_attend_group` | `common/attention_group.cu` `attn_group_kernel` | `attend_window`'s softmax for a group of rows a block: a kv head's G query heads of 16 / G rows fill an `m16n8k16` tile, two tiles a block, each row its own key range and chunk merge (the window kernel's order, so a row's bits are its chunks' alone); 29,952 B, three blocks an SM | Qwen3-ASR's language model, a prompt window of 64 rows or more | none |
| `zs_attn_attend_window` | `common/attention_window.cu` `attn_window_kernel` | `attend`'s softmax and merge, a block a row over its chunks, the merge in registers; `attend`'s bits | 64 rows or more on contiguous caches | `ZSPARK_ATTN_WINDOW` (64) |
| `zs_qwen4_attn_prep_table` | `attention_table.cu` `attn_prep_table_kernel` | `prep` through a table of 512-position extents | caches behind a table (the pool's layout; tests) | none |
| `zs_attn_attend_table` | `attention_table.cu` `attn_chunk_table_kernel`, `attn_merge_table_kernel` | `attend` through the table | as above | none |
| `zs_qwen4_attn_prep_rows` | `attention_rows.cu` `attn_prep_rows_kernel` | `prep` for several streams in one round | pool | none |
| `zs_attn_attend_rows` | `attention_rows.cu` `attn_chunk_rows_kernel`, `attn_merge_rows_kernel` | `attend` for several streams | pool, a round under 64 rows | none |
| `zs_attn_attend_window_rows` | `qwen4/attention_window_rows.cu` `attn_window_rows_kernel` | `attend_window` for several streams, `attend_rows`' bits | pool, a round of 64 rows or more | `ZSPARK_ATTN_WINDOW` (64) |
| `zs_qwen4_attn_pool` | `qsa.cu` `pool_kernel` | the indexer's keys pooled a block of positions, rotated at the block's first position (its t, h and w for an image prompt) | Flash Next FP4-Local | none |
| `zs_qwen4_attn_select` | `qsa.cu` `scores_kernel`, `select_kernel` | a row's best blocks, in block order, as its list of keys | Flash Next FP4-Local | none |
| `zs_qwen4_attn_select_window` | `qsa.cu` `scores_window_kernel`, `select_kernel` | a prompt window's lists at once: every row's scores from the indexer queries and the pooled keys (the selection GEMM's k16 chain and `scores_kernel`'s sum, no `dots`), then one selection over the rows; the tiles' bits. Shared memory: 64 pooled keys, DI + 8 values a row (17,408 B at 128) | Flash Next FP4-Local, windows of 33 rows and more, caches in order | none (`attention.whole_rows`) |
| `zs_qwen4_attn_pos_add` | `pos_add.cu` `pos_add_kernel` | a later tile's window position | Flash Next FP4-Local, a selection past one tile; never in a captured round | none |
| `zs_qwen4_attn_pool_table` | `qsa_table.cu` `pool_kernel` | `pool` through the table | as `prep_table` | none |
| `zs_qwen4_attn_select_table` | `qsa_table.cu` `scores_kernel`, `select_kernel` | `select` writing the pool's rows | as `prep_table` | none |
| `zs_qwen4_attn_select_window_table` | `qsa_table.cu` `scores_window_kernel`, `select_kernel` | `select_window` writing the pool's rows | pool, a stream's 33 rows or more in a round | none (`attention.whole_rows`) |
| `zs_qwen4_attn_pool_slots` | `qsa_rows.cu` `pool_slots_kernel` | `pool` for several streams | pool | none |
| `zs_qwen4_attn_pack_queries` | `select_slots.cu` `pack_queries_kernel` | a round's indexer queries packed a slot | pool | none |
| `zs_qwen4_attn_select_slots` | `select_slots.cu` `scores_slots_kernel`, `select_slots_kernel` | `select` a round row at a time over the slots | pool | none |
| `zs_qwen4_ple_embed` | `ple.cu` `ple_embed_kernel` | the gathered n-gram rows side by side, scaled | Flash Next FP4-Local | none |
| `zs_qwen4_ple_gate` | `ple.cu` `ple_gate_kernel` | the value projection gated by each stream's key-query product | Flash Next FP4-Local | none |
| `zs_qwen4_ple_conv` | `ple.cu` `conv_kernel` | the dilated convolution over the gated rows, added to the streams | Flash Next FP4-Local | none |
| `zs_qwen4_ple_conv_slots` | `ple_slots.cu` `conv_slots_kernel` | the convolution for several streams | pool | none |
| `zs_qwen4_mtp_rmsnorm` | `mtp.cu` `rmsnorm_kernel`; at width 10,240 and 4 rows or fewer `rmsnorm_tiles_kernel` (10 tile blocks a row, same bytes; spec 152 WS-07, TensorFold ca8f24bb) | the MTP head's norms of the next embedding and the final streams | Flash Next FP4-Local, a draft step | none |
| `zs_qwen4_mtp_add_streams` | `mtp.cu` `add_streams_kernel` | the head's own streams from the two | Flash Next FP4-Local, a draft step | none |

The vision tower, an image's patches to its features; no decoding round reaches it:

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_qwen4_vision_pos` | `vision.cu` `pos_kernel` | the patches' bias, then the learned position table resampled to the grid (four taps) added | Flash Next FP4-Local, an image | none |
| `zs_qwen4_vision_rope` | `vision.cu` `rope_kernel` | the 2D rope on q and k in place, a block a patch | Flash Next FP4-Local, an image | none |
| `zs_qwen4_vision_norm` | `vision.cu` `norm_kernel` | a pending branch added into h, then LayerNorm | Flash Next FP4-Local, an image | none |
| `zs_qwen4_vision_attention` | `vision_attention.cu` `vision_attention_kernel` | softmax attention of each image over its own patches, 128 queries a tile cut at each image's start, online softmax on `m16n8k16`; 45,056 B, two blocks an SM | Flash Next FP4-Local, an image | none |

## `unlimited_ocr/`

Unlimited-OCR-2's shared vision tower, a page's views to the language model's image rows; no decoding round
reaches it. Historical times below are from v1's shared tower (`vision_test.zig`, spark-b, 2026-10-25,
spec 137 WS-03); the gundam page took 90 ms, about 45 of them the CUTLASS linears.

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_unlimited_ocr_vision_patches` | `vision.cu` `ocr_patches_kernel` | the 16 x 16 patches' columns for the patch conv's GEMM | an OCR page | none |
| `zs_unlimited_ocr_vision_im2col` | `vision.cu` `ocr_im2col_kernel` | a 3 x 3 conv's columns (padding 1, stride 1 or 2), channels last, for the neck's GEMMs | an OCR page | none |
| `zs_unlimited_ocr_vision_sam_pos` | `vision.cu` `ocr_sam_pos_kernel` | the patch conv's bias, then SAM's position table (resampled on the host for a crop) | an OCR page | none |
| `zs_unlimited_ocr_vision_norm` | `vision.cu` `ocr_norm_kernel` | a pending branch added (CLIP's; SAM's proj and lin2 add theirs in `zs_b16_gemm_bias_residual`, spec 142 WS-08), then LayerNorm, a row a token (SAM's window blocks read their windows through the attention's map, spec 142 WS-03); the neck's LayerNorm2d; CLIP's on its fp32 stream | an OCR page | none |
| `zs_unlimited_ocr_vision_clip_embed` | `vision.cu` `ocr_clip_embed_kernel` | CLIP's class token and SAM's features plus its position table, then `pre_layrnorm` to fp32 | an OCR page | none |
| `zs_unlimited_ocr_vision_concat` | `vision.cu` `ocr_concat_kernel` | CLIP's rows past the class token beside SAM's features, bf16, the projector's input | an OCR page | none |
| `zs_unlimited_ocr_vision_place` | `vision.cu` `ocr_place_kernel` | the projector's rows, `image_newline` and `view_seperator` to their rows of `lm_features` | an OCR page | none |
| `zs_unlimited_ocr_vision_relpos` | `vision_attention.cu` `ocr_relpos_kernel` | SAM's decomposed relative positions, `q . table` for a row's kh + W places and every head (a window's row through its token map, zero on padding); a lane a place, its table row widened in registers, the row's q loaded 16 bytes a lane into shared memory | an OCR page | none |
| `zs_unlimited_ocr_vision_attention` | `vision_attention.cu` `ocr_attention_kernel` | softmax attention within each segment (a 14 x 14 window, a view), online softmax on `m16n8k16`; a window's rows gathered from the token rows by a map, its padding qkv's bias (a zero row's q, k, v), its output scattered back; SAM's relative-position bias: rel_w staged a block's rows beside K and V (in registers on the 64 x 64 view), rel_h read from L1; CLIP and the 64 x 64 view 36,864 B, SAM's windows 43,008, a crop's global blocks 47,104: two blocks an SM in every form | an OCR page | none |

Unlimited-OCR-2's DASH decoder uses the shared MoE kernels.

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_unlimited_ocr_dash_tile` | `dash_attention.cu` `unlimited_ocr_dash_tile_kernel` | Unlimited-OCR-2's DASH attend: a row's causal 128-key window (no bias) and its router's chosen 32-key blocks (a bias a block, the window winning), on `m16n8k16`; 64 rows a block over their union of blocks in ascending order, each block's partial folded in order; 30,756 B | Unlimited-OCR-2's decoder (WS-05) | none |
| `zs_unlimited_ocr_dash_step` | `dash_attention.cu` `unlimited_ocr_dash_step_kernel` | the same rows a block a (unit, head, row), a warp a 32-key block, the last block folding the row's partials in order: the tile kernel's bits; 16,473 B | Unlimited-OCR-2's decoder (WS-05) | none |
| `zs_unlimited_ocr_moe_route` | `moe.cu` `unlimited_ocr_route_kernel` | the gate's fp32 softmax over 64 experts and the top 6 (the lower id among equals), unnormalized, ascending; at one row the experts' plan too, as `zs_experts_plan` writes it | Unlimited-OCR's decoder | none |
| `zs_unlimited_ocr_moe_combine` | `moe.cu` `unlimited_ocr_combine_kernel`, `unlimited_ocr_combine_norm_kernel` | h += bf16(bf16(sum of weight x y in fp32) + shared); in a decoder's chain, with the next layer's (or the final) RMSNorm of h in the same launch, `zs_rmsnorm`'s bits | Unlimited-OCR's decoder | none |
| `zs_unlimited_ocr_router_prep` | `router.cu` `ocr_router_prep_kernel` | Unlimited-OCR-2's DASH router past its GEMM: a row's index q and k from the stacked `q_proj`/`k_proj` (bf16 CUTLASS) through Gemma RMSNorm over 128 (fp32, clamp 1e4) and page2d RoPE from the row's coordinates (cos, sin and the products bf16, as HF's), bf16 out; k into the index cache at `at[0] + r`; a block a row and side, 128 threads, 528 B | Unlimited-OCR-2's decoder (WS-05 to wire) | none |
| `zs_unlimited_ocr_router_kbar` | `router.cu` `ocr_router_kbar_kernel` | the k_bar (fp32) of a block a window's row completes: `head_cls`'s softmax over the block's 32 keys, summed in key order; 128 threads, 128 B | Unlimited-OCR-2's decoder (WS-05 to wire) | none |
| `zs_unlimited_ocr_router_route` | `router.cu` `ocr_router_route_kernel` | a row's remote blocks: scores `q . k_bar` over blocks ending 128 keys before it, the page-aware candidates, top 16 (the lower block among equals), 1.5-entmax sort-based in a fixed order, top 12, bias `(log(clamp(p)) - mean) / sigma`; 128 threads, 672 B and 4 B a block dynamic | Unlimited-OCR-2's decoder (WS-05 to wire) | none |

## `qwen3_asr/`

Qwen3-ASR-1.7B's audio tower, a clip's mel to its embeddings; no decoding round reaches it; and
its language model's attention prep (WS-04), whose attend is `common/`'s. The
convolutions are im2col and `zs_b16_gemm_bias`, the linears `zs_b16_gemm_bias`, the norms `zs_layernorm`. Its
time: 193.6 ms for 600 s of audio (7,847 tokens), a fast build, the machine alone (2026-10-25).

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_qwen3_asr_tower_patches` | `tower.cu` `tower_patches_kernel`, `tower_patches1_kernel` | a 3x3 stride-2 convolution's columns from channels-last rows (the mel's one channel padded to 16 columns); no shared memory | an audio clip | none |
| `zs_qwen3_asr_tower_bias_gelu` | `tower.cu` `tower_bias_gelu_kernel` | a convolution's bias added and rounded, then exact GELU, in place (PyTorch's two roundings); no shared memory | an audio clip | none |
| `zs_qwen3_asr_tower_positions` | `tower.cu` `tower_positions_kernel` | the sinusoid table's row of each token's place in its chunk added, rounded to bf16; no shared memory | an audio clip | none |
| `zs_qwen3_asr_tower_attention` | `tower_attention.cu` `tower_attention_kernel` | softmax attention of each 104-token block over itself, 128 queries a tile, online softmax on `m16n8k16`, `vision_attention.cu` at head 64; 36,864 B, two blocks an SM | an audio clip | none |

## `glm5/`

GLM-5.3-Flash's own kernels.

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_glm5_hc_mix` | `hc.cu` `mix_octs_kernel`, from 16 rows `mix_rows_kernel`; base `mix_kernel` | the read-out's 24 dots a block of the four streams | GLM FP4-Local | `ZSPARK_GLM_HC_OCTS` (1) |
| `zs_glm5_hc_pre` | `hc.cu` `pre_lanes_kernel`; base `pre_kernel` | the part sums, the gates, the Sinkhorn's 4x4, the collapsed row normed (its weight bf16, or fp32 as FP4-Local stores it) | GLM FP4-Local | `ZSPARK_GLM_HC_LANES` (1) |
| `zs_glm5_hc_forms` | nothing | sets the two above to their forms or their bases | GLM FP4-Local | the two above |
| `zs_glm5_hc_post` | `hc.cu` `post_kernel` | the ranks' partials written back into the streams | GLM FP4-Local | none |
| `zs_glm5_stream_mean` | `hc.cu` `mean_kernel` | the streams' mean | GLM FP4-Local | none |
| `zs_glm5_kda_chain` | `kda_quads.cu` `chain_kernel`; base `kda.cu` `chain_kernel` | KDA's chain of a window's rows from the state, up to 48 rows | GLM FP4-Local | `ZSPARK_GLM_KDA_QUADS` (1) |
| `zs_glm5_kda_chunked` | `kda_chunk.cu` `intra_kernel`, `state_kernel`, `out_kernel` | KDA's chain of a window of more than 48 rows, chunks of 64 on absolute positions; the states asked at chunk starts inside it written on the way (marks); the state alone, `out` null | GLM FP4-Local | `kda.chunked_windows` (tests) |
| `zs_glm5_kda_replay_layers` | `replay_layers_kernel` of the same two files | every layer's state after the rows a round kept, one launch | GLM FP4-Local | `ZSPARK_GLM_KDA_QUADS` (1) |
| `zs_glm5_kda_quads` | nothing | sets the two above to `kda_quads.cu` or `kda.cu` | GLM FP4-Local | `ZSPARK_GLM_KDA_QUADS` |
| `zs_glm5_kda_conv_shift` | `kda_state.cu` `conv_shift_kernel` | the conv's window after a commit | GLM FP4-Local | none |
| `zs_glm5_mla_absorb` | `mla_proj.cu` `absorb_kernel` | DSA's queries through `kv_b` into the latent | GLM FP4-Local | none |
| `zs_glm5_mla_expand` | `mla_proj.cu` `expand_asked_kernel` to 16 rows; base `expand_kernel` | the attended latent through `kv_b` to the heads' values | GLM FP4-Local | `ZSPARK_GLM_MLA_ASK` (2) |
| `zs_glm5_mla_ask` | nothing | sets how far ahead `expand` asks, 0 the base | GLM FP4-Local | `ZSPARK_GLM_MLA_ASK` |
| `zs_glm5_mla_q8` | nothing | an FP8 cache's chunks with FP8 queries (1, the default) or bf16 queries (0): not the same bits | GLM FP4-Local | `ZSPARK_GLM_MLA_Q8` |
| `zs_glm5_mla_window` | nothing | from how many rows `attend` runs its window form: 128, fixed (`dsa.zig` `Buffers` sizes po's partials for fewer rows alone); another count (0 never) in tests alone, the same bits | GLM FP4-Local | none |
| `zs_glm5_mla_write` | `mla_proj.cu` `write_kernel`; `write_fp8_kernel` | a row's latent into the cache at its position; under `--kv-dtype fp8` as vLLM's 528-byte record, E4M3 codes and a scale a 128 values | GLM FP4-Local | none |
| `zs_glm5_mla_attend` | `mla.cu` `mla_chunk_kernel`, `mla_merge_kernel`; from 128 rows their window form, `mla_chunk_kernel<_, true>` or `q8_chunk_kernel<true>` alone | a row's online softmax over the latent cache on the tensor cores: every key up to its position, or the indexer's list past the dense limit, the keys gathered by the list; a block a row's 32 heads over one staged tile; an FP8 cache's records read raw by `mla_q8.cu` `q8_chunk_kernel`: the queries e4m3 per (head, 128), the scores by the block-scaled e4m3 mma, P.V by HIGH and LOW e4m3 weights, as vLLM's b12x (spec 114 WS-05: alone 80 to 60 us at 1 row, 93 to 73 at 8, a 512-row window 4.74 ms to 3.12); `ZSPARK_GLM_MLA_Q8=0` the records widened exactly to bf16 under bf16 queries (`mla_chunk_kernel<true>`, spec 107 WS-03) (spec 093: alone, a list of 2,051 keys, 89 to 105 us at 1 to 6 rows to 65 to 75, a 512-row window 7.0 ms to 3.5); from 128 rows a block a row runs its chunks in order and folds each into the row's output as the merge does, with no partials but the running output and no merge launch, the merge's bits (spec 116 WS-01: alone, a 4,096-row window FP8 24.9 ms to 12.7, bf16 cache 34.0 to 30.5; 512 rows 3.1 to 1.67) | GLM FP4-Local | none |
| `zs_glm5_index_write` | `index.cu` `write_kernel` | a row's index key after LayerNorm, and its gate, through the extent table: a block a row, a thread a value, the sums one thread's in order | tests | none |
| `zs_glm5_index_pool` | `index.cu` `pool_kernel` | a pool of four as vLLM's: the gates' softmax summed in fp32, rounded once to bf16, rotated (Walsh-Hadamard) and stored e4m3 codes with an fp32 power-of-two scale; a warp a pool, rows / 4 + 1 pools from the window's first | tests | none |
| `zs_glm5_index_query` | `index.cu` `query_kernel` | each head's query rotated and coded as a pooled key, its scale folded into the head's fp32 weight with 1 / sqrt(D H); a warp a head of a row | tests | none |
| `zs_glm5_index_score` | `index.cu` `score_kernel` | a row's scores over the pools complete before it on the FP8 tensor cores (m16n8k32 e4m3): a block 64 pools and 8 rows, the rows' coded queries staged once, the keys' fragments in registers; relu times the folded weight over the heads, times the key's scale; the grid the context's pools | tests | none |
| `zs_glm5_index_select` | `index.cu` `select_kernel` | the kept pools, ties to the lower, then the unfinished pool's tail | tests | none |
| `zs_glm5_moe_topk` | `moe.cu` `topk_kernel` | the router's selection and weights | GLM FP4-Local | none |
| `zs_glm5_moe_combine` | `moe.cu` `moe_combine_kernel` | a rank's share of the block from its slots and the shared expert | GLM FP4-Local | none |

## `nemotron/`

Nemotron-3.5-Transcribe's backbone glue, taken from TensorFold's `nemotron_*.cu` with the kernels'
device code byte for byte, and `mamba_wide.cu`, zspark's prompt path for the Mamba mixer with the step's bits;
nothing reaches them but their tests yet. The norms write their rows' fp32 64-group sums (`xs`), which the
4-bit matmuls they fed upstream are not ported to read.

| Entry | Launches | Computes | Reached by | Switch |
| --- | --- | --- | --- | --- |
| `zs_nemotron_add_rmsnorm` | `norms.cu` `tf_nemo_add_rmsnorm` | h = bf16(x + r) (x alone without r), y = bf16(rmsnorm(h) * w), xs the row's group sums; a block of 256 a row | tests | none |
| `zs_nemotron_add_moe_norm` | `norms.cu` `tf_nemo_add_moe_norm` | the experts' slots summed in order (the routed by fma through wt, the shared added), h = bf16(x + delta), then the norm | tests | none |
| `zs_nemotron_concat_norms` | `norms.cu` `tf_nemo_concat_norms` | the MTP input [rmsnorm(e) * we ; rmsnorm(h) * wh] side by side and its group sums, (row, part) blocks | tests | none |
| `zs_nemotron_group_rmsnorm` | `norms.cu` `tf_nemo_group_rmsnorm` | the Mamba blocks' group RMSNorm: a block a (row, group) of up to 1,024, n = bf16(v * inv) then bf16(w * n) | tests | none |
| `zs_nemotron_router` | `route.cu` `tf_nemo_router` | fp32 logits in SK K-slices (16x16 staged tiles), one fma chain a slice; SK divides D / 64, picked as `glue.routerSlices` | tests | none |
| `zs_nemotron_topk` | `route.cu` `tf_nemo_topk` | a warp a row: the slices summed in order, sigmoid scores plus bias ranked (the lower id on a tie), weights prob / (sum + 1e-20) * scaling, then the shared slots | tests | none |
| `zs_nemotron_serial_feed` | `ops.cu` `tf_serial_feed` | a serial round's end on the device: the token into ids and history, pos + 1, parity flipped, keep 1 | tests | none |
| `zs_nemotron_plan_routed` | `ops.cu` `tf_plan_routed` | the routed slots' plan: members grouped by expert (order kept), items of at most a tile, the counts; one block of 128 | tests | none |
| `zs_nemotron_rest_rows` | `ops.cu` `tf_rest_rows` | y[row * row_mul + row_add] += r @ x[row], a warp a column over 8-value chunks, fp32 or bf16 y | tests | none |
| `zs_nemotron_keyed_greedy` | `keyed.cu` `tf_nemo_keyed_greedy` | a warp a row of up to 32 MTP candidates: the rank-0 id (value desc, id asc), the top-k mass share in fp64 | tests | none |
| `zs_nemotron_mamba_conv` | `mamba.cu` `tf_nemo_conv` | a window's conv (up to 16 rows), a thread a channel: the kept rows of raw[1 - parity] replay into base, the window's rows into raw/xc[parity]; 4 taps by fma, SiLU | tests | none |
| `zs_nemotron_mamba_scan` | `mamba.cu` `tf_nemo_scan` | a window's selective scan, block (head, 32 value rows) of 128: the kept rows replay and commit the fp32 state, then the window's rows; y = bf16(silu(z)) * bf16(C . s + D x) | tests | none |
| `zs_nemotron_mamba_conv_wide` | `mamba_wide.cu` `nemo_conv_wide`, `nemo_conv_wide_commit` | a prompt chunk's conv of any rows, thread (channel, 16 rows), each row tf_nemo_conv's bits; base committed through the row before the last, which waits in the parity buffers | tests | none |
| `zs_nemotron_mamba_scan_wide` | `mamba_wide.cu` `nemo_scan_wide` | a prompt chunk's scan of any rows, tf_nemo_scan's blocks and bits, 16 rows staged in shared memory while the next 16 load; the state committed through the row before the last | tests | none |
| `zs_nemotron_mamba_a` | `mamba_wide.cu` `nemo_mamba_a` | a = -exp(A_log) at load, TensorFold's `tf_nemotron_mamba_a_f32_kernel`'s arithmetic | tests | none |
| `zs_nemotron_perception_prep` | `perception.cu` `nemotron_prep_kernel` | the encoder's qk LayerNorm a head (a block a row, a warp a head) then RoPE on the first ROT dims, in place on the w_qkv rows; a row's position its index inside its `offsets` segment | tests | none |
| `zs_nemotron_perception_attention` | `perception_attention.cu` `perception_attention_kernel` | a segment's tokens attend to their own alone (`qwen3_asr/tower_attention.cu`'s tiling at head 128): a block (128 queries of a segment, one head), keys 64 at a time; `tiles` names the segments, `w` a key length where the PEE's padding mask cuts | tests | none |
| `zs_nemotron_speaker_conv_input` | `speaker.cu` `nemotron_speaker_conv_input_kernel` | packs [T,H] into caller-owned [T,H*3] for the full-channel subpixel Conv1d, masking rows past the valid length; shared BF16 tensor-core GEMM computes all output channels, followed by ReLU | Sortformer | none |

## The kernel contract

A kernel that sums or masks registers with `zig/tests/cuda/contract.zig`, and runs under
`tools/check.sh weights contract` on the GB10. The harness runs five gates over the shapes the adapter
declares — the shapes the engine serves it at and their tile edges — each against the kernel's own formula
in f64:

1. **Non-finite.** NaN and both infinities go into the inputs at known places; every output's class (finite,
   NaN, +Inf, -Inf) must equal the f64 reference's, with no tolerance.
2. **Repeat and aliasing.** Five runs on the same inputs give the same bytes, the inputs are unchanged, the
   sentinel rows around the output are intact, and no output overlaps an input unless the spec says in
   place.
3. **Shapes.** Every declared shape, each output within `|want| * half an ulp + gamma(depth) * magnitude`:
   the element's reference value, the number of roundings after its sums, and the magnitude of the terms it
   sums.
4. **fp32 sums.** Integer inputs whose fp32 partials are exact; the kernel stays inside gate 3's bound,
   while the same terms replayed in f16 and in bf16 on the host miss the *fp32* bound by 100 times. The
   output's own step is not the measure: a bf16 output's rounding (2^-8) is coarser than an f16 sum's error
   at the norm's shapes, so it cannot show that accumulator, and the gate judges the sum itself.
5. **The launch.** Threads, registers, shared memory and blocks an SM against the GB10's limits
   (`cuda-gb10`: 1,536 threads and 65,536 registers an SM, 102,400 B of shared memory an SM, 101,376 B a
   block opted in, 24 blocks an SM). This gate needs no GPU. The figures an adapter declares are read from
   `cuobjdump -res-usage` of the linked test binary and kept by hand: gate 5 compares them with the chip's
   limits, not with the binary, and a launcher that picks among kernels declares the worst its shapes take.

Registered: `zs_b16_gemm` (`bf16/gemm.cu`, on CUTLASS) and `zs_rmsnorm` (`common/glue.cu`). A kernel
registers in the change that adds it, with an adapter in `zig/tests/cuda/contract_test.zig`; the adapter is
its inputs, its launch, its f64 formula, the magnitude of its terms and its guard rows. The controls in
`zig/tests/kernels/contract/controls.cu` each break one gate on purpose (a NaN dropped by `fmaxf`, a write
into its input, a lost ragged tail, a bf16 sum); a control that passes the gate it was planted for fails
the test, because then the gate proves nothing.

## Closed until a new measurement

- Closed until a new measurement: `experts_wide.cu`'s other shapes and its splits summed in the block; a branching compare in `moe.cu`'s count.
- Closed until a new measurement: the values' transpose of `attention.cu`'s chunk kernel (and the pool's
  `attention_rows.cu`, `attention_table.cu`) in `attention_window.cu`'s lane order: a reply's
  rows wait on their loads, not on the stores' banks, and that order made 4 to 32 rows slower alone (16 rows
  over 16k fp8 86.8 to 102.9 us); chunk and merge are 1.4 to 1.8% of a decoding round's kernels at 4k and 16k.
