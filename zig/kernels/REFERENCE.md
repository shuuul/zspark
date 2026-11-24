# `zig/kernels/` reference

Read when a task touches what a section names; the `AGENTS.md` beside this file binds every task here.

## The directories

- `zs/`: the C ABI (`abi.h`), the host shim (`host.h`, `host.cu`) that stands where ATen and c10 stood in
  TensorFold, and the toolchain probe.
- `nvfp4/`: TensorFold's lane matmul, prompt rows and routed experts, device code unchanged but `experts.cu`'s
  EVEN form (073: `experts.even_units`) and the code decoders, one copy in `codes.cuh` (spec 121; the e4m3 value and the signed e2m1 code, spec 136). zspark's own: `experts_wide.cu` (021, 032, from
  `experts.Pass.wide_rows` rows), `prompt_bias.cu` (spec 019
  WS-08: `prompt.cu`'s loop copied with a bias and GELU epilogue, the vision tower's quantized linears),
  `qmm_a8.cu` (spec 114 WS-04: MXFP8 and block-FP8 weights times rows quantized as vLLM's, on FP8 tensor cores)
  and the W4A4 routed experts (spec 114 WS-03: `experts_a4.cuh` the block-scaled FP4 mma, its operand lanes as
  probed and b12x's activation quantization, and `swiglu_a4`, down's rows made in gate and up's epilogue from their
  fp32 SwiGLU (WS-07); `experts_quant.cu` gate and up's rows quantized; `experts_a4.cu`, `experts_tile_a4.cu` the
  products, the latter's RH 2 form 128 pairs a reading in dynamic shared memory; `experts.Activations`).
- `common/`: TensorFold's routing plan and shared headers; `columns.cu`, `embed.cu` (`zs_embed`, a token's
  row into each stream, both models'; `zs_embed_rows`, an encoder's features over a prompt's placeholder rows) and `pick.cu` (`zs_pick`) are zspark's, and so is `math.cuh`, the device one-liners
  (`bf`, `bf16_low`, `warp_sum`, `sigmoidf_` and the like) that were the same text in many kernels, and `copies.cuh`,
  `cp16`, `cp_commit`, `cp_wait` (`experts.cuh`'s) and the `ldmatrix` family — `ldmatrix2`, `ldmatrix4` (`qmm_frag.cuh`'s, re-exported),
  `ldmatrix2t`, `ldmatrix4t` (spec 121; the one home since spec 158 WS-01). `dflash2.cu`
  (spec 067, 122) is rewritten from `dflash2.py`, `--fmad=false`; `dflash2_attend.cu` is `mla.cu`'s steps: both
  models' DFlash2 drafters launch them. `glue.cu` (spec 136) is the row glue from `glue.py` that GLM and Flash Next's
  DFlash2 launch: `zs_rmsnorm`, `zs_branch`, `zs_swiglu`, moved from `glm5/hc.cu` and `moe.cu`, their
  helpers in `glue.cuh`, which `hc.cu` includes too; the residual adds (`zs_residual_add`, `zs_residual_rmsnorm`)
  serve Qwen3-ASR and Unlimited-OCR too, the latter's bf16 form and `glue.cuh`'s `norm_row` moved from
  `unlimited_ocr/moe.cu` (spec 142 WS-04). `layernorm.cu` (spec 138) is zspark's LayerNorm with the residual add
  before it, any width in 8s; `qwen4/vision.cu`'s `norm_kernel` is its arithmetic at 1,152. `attention.cu` and `attention_window.cu` (spec 138 WS-04) are
  the text attention moved from `qwen4/` when Qwen3-ASR's language model became its second family, the text
  unchanged and the entries `zs_attn_*`; `attention.cuh` and `attention_mma.cuh` are what they and `qwen4/attention.cu`'s
  prep share. `qwen4/attention_table.cu` and `attention_rows.cu` compile both halves again for the pool.
- `bf16/`: `gemm.cu`, one CUTLASS GEMM on bf16, its tile by rows and weight (073: `bf16.Tile.of`); `reduce.cuh`. `gemm_bias.cu` (spec 019): a linear with its bias
  and GELU in CUTLASS's epilogue, one tile at every row count, for the vision tower. The hand-written matmul of the same bits is a test's (`zig/tests/kernels/bf16/`).
- `qwen4/`: Qwen3.8 Flash Next's own, rewritten from TensorFold's Triton sources or taken from the family's
  `.cu`. `gdn_wide.cu` (spec 031) is the DeltaNet chain from `gdn.wide_rows` rows. `gdn.cuh` holds what
  `gdn.cu`, `gdn_slots.cu` and `gdn_wide.cu` share: the head's shape and `update` (spec 121), and `spread_sum` from TensorFold 1.1.4 (995f98af) with `update_spread` (spec 143 WS-05; the slots' kernels too, WS-10). `vision.cu` and
  `vision_attention.cu` (spec 019) are the vision tower's, zspark's own for transformers' `Qwen3_5VisionModel`. `*_table.cu`,
  `*_rows.cu` and `*_slots.cu` are spec 018's shared rounds: each a second object, so that
  the kernel it follows stays the one it was. `hc_down.cu` (spec 148 WS-03) is the read-out's norm, down and
  activation in one launch at a prompt window's rows, the norm in the down tile after Mia's AI Lab's recipe
  (MIT, patch 0014's `hcn`), a block over the whole K; its bits `hc.cu`'s and `bf16/gemm.cu`'s.
- `qwen3_asr/`: Qwen3-ASR-1.7B's audio tower (spec 138), zspark's own for the `qwen-asr` package's
  `Qwen3ASRAudioEncoder`: `tower.cu` (the convolutions' columns, bias and GELU, the positions) and
  `tower_attention.cu`, `qwen4/vision_attention.cu` at head 64 over 104-token blocks. Its language model's prep (WS-04:
  the q and k norms, RoPE and the caches before `common/attention.cu`'s attend, from transformers' `Qwen3Attention`;
  TensorFold has no Qwen3 dense) is `common/attention_prep.cu`, which Unlimited-OCR's decoder takes without the
  norms (spec 142 WS-04).
- `nemotron/`: Nemotron-3.5-Transcribe's backbone glue (spec 151), taken from TensorFold's `nemotron_*.cu`
  with the kernels' device code byte for byte: `norms.cu` (the residual RMS norms and group RMSNorm),
  `route.cu` (the sliced router and the top-k with the shared experts' slots), `ops.cu` (the serial feed,
  the routed plan, `tf_rest_rows`), `keyed.cu` (the MTP draw's greedy pick) and `mamba.cu` (the Mamba-2
  window's conv and scan, without upstream's prompt conv). `mamba_wide.cu` is zspark's: a prompt chunk of any
  rows with the window kernels' bits, and A = -exp(A_log) at load. `glue_math.cuh` is
  TensorFold's one-ulp transcendentals, kept beside `common/math.cuh`'s expf forms because the bits differ.
  Dropped as dead under the NVFP4 checkpoint: `tf_nemo_embed` (MLX 4-bit token rows), `tf_pack_dense` and
  `tf_transpose_pad16` (the lane matmul's repack); the mamba and attention kernels come in their own slices.
- `unlimited_ocr/router.cu`: Unlimited-OCR-2's DASH router past its GEMM (spec 147 WS-03), rewritten from the checkpoint's
  `dash_attention.py` and `modeling_deepseekv2.py` with Triton off: prep (Gemma norm, page2d RoPE), k_bar, and the route
  (candidates, top 16, 1.5-entmax, top 12, bias), sums in a fixed order and no contraction in the entmax.
- `glm5/`: GLM-5.3-Flash's own (spec 044), from `families/glm5_next/cuda/` at `cb2ebf0`. Rewritten from
  Triton: `hc.cu` (the four streams' read-out in parts, write-back, mean) and `moe.cu` (selection, combine;
  the RMSNorm and SwiGLU are `common/glue.cu`'s) from `glue.py`, `mla.cu` (DSA's dense rows over the latent cache) and `mla_proj.cu`
  (absorb, expand, the cache writes; split from it in spec 121) from `latent.py`. Taken:
  `kda.cu`, the family's `kda.cu` with `kda.cpp`'s launchers on the C ABI, its device code upstream's in
  `kda_common.cuh` (which `kda_quads.cu` compiles with the state by `float4`), compiled `--fmad=false` as upstream. `kda_state.cu` is zspark's (the conv's window after a commit).
  `mla_q8.cu` (spec 114 WS-05) is zspark's: an FP8 cache's chunks in vLLM's FP8 form (b12x `decode_math.py`'s
  query quantization, block-scaled scores and HIGH and LOW weights), `mla.cu`'s tiles; `mla.cuh` what the three share.
  `index.cu` (spec 046) is the indexer, `--fmad=false`, its score an mma: a row's list from its own query, ties to the lower pool,
  keys and gates through the extent table, the pooled keys contiguous.
- `tools/check.sh sanitize AREA FILTER [TOOL]` does the above in one verb (spec 103). A served load under the
  tool (spec 086, 103): `compute-sanitizer --tool memcheck --padding 1048576` before `zspark serve`,
  `--context 16384`; end the server, not the tool, or no summary is written.

## Reading a kernel with the counters

What Nsight Compute gives on the GB10 (spec 028 WS-04, `ncu` 2025.3.1 under `/usr/local/cuda/bin`):

- On a loaded model use `--replay-mode application` with `--graph-profiling node`: 13 to 18 minutes for a
  six-token reply. Kernel replay, the default, saves the device's memory before each launch: hours.
- TensorFold is read the same way in its container, the host's `/opt/nvidia/nsight-compute` mounted read
  only, `--nvtx --nvtx-include gen/`.
- `--set full` works. No counter for DRAM or the TLB: `lts__d_sectors_fill_sysmem` is the fill from memory.
- Equal counts do not mean equal time. The pack's kernels ran TensorFold's instructions, misses and fills to
  0.1% in 3 to 6% more cycles: the cause was where the buffers lay (`cuda.arena`), which no counter names.
