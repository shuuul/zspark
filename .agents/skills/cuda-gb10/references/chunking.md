# Chunking and windows on the GB10

How zspark cuts a prompt into windows and a window's work into chunks, what each width bought and cost on this
chip, and how every cut keeps the row rule. Part of the `cuda-gb10` skill. A window is the rows one forward
runs (a prompt's part, a verify window, a serial step); a chunk is a cut inside a kernel (64 rows of KDA, 512
keys of attention).

## The rule every cut keeps

- **A row's bits never depend on the rows beside it** (root `AGENTS.md`). A chunk grid that comes from the
  row's own position (KDA's 64 on absolute positions) or from the row's own key list (attention's 512-key
  chunks) keeps it; one that counts from a window's first row would not. A kernel's tile order and its K
  slices are chosen by the shape alone, never by how many rows share the launch (`exactness.md`).
- **GLM FP4-Local's form follows the window's kind** (spec 115, the owner's option B): a step, a verify window
  and a prompt window of 48 rows or fewer run bf16 rows (routed experts A16, MXFP8 linears on bf16 rows); a
  prompt window of more than 48 runs the card's quantized forms. 48 is the card's own bound (the b12x MXFP8
  plans' largest bf16 M; its MoE graphs capture up to 24 rows). Within a form a row's bits stay its own; a
  verify window and a serial step share one form, so drafted replies stay bit-exact. Cost, accepted: a prompt
  whose windows fall on both sides of 48 is not bit for bit the same prompt cut otherwise (a hall's part beside
  other streams against alone). [`precision-forms.md`](precision-forms.md) has the forms.
- **Test a cut by cutting**: `glm5 kda` runs a chunked window cut at chunk starts from leads 0, 5 and 63
  against the uncut one; `dsa_test` lists that end a chunk exactly and 1, 2, 3 keys into the next;
  `ranks_windows.zig` plays 2,085 rows as one 2,048-row window, as 512-row windows and as serial steps (specs
  116 WS-01, WS-06, 111 WS-01).

## A prompt's window rows

Why wider pays: a window reads every routed expert it touches once a layer, so the experts' bytes a row fall
with the width. What it costs: buffers a row (logits, slots, KDA's saves, attention partials) and a pool that
must still fit.

| Checkpoint, ranks | Widths | What it bought, what it cost | Source |
| --- | --- | --- | --- |
| Flash Next FP4-Local, one Spark | 2,048 default; `--max-num-batched-tokens 4096` off by default | 16k cold 7.83 to 7.41 s (median) for 2.56 GiB; at 4,096 the pool's rounds took 25.1 GiB of slots against 13.5, so the pool keeps 2,048 | 110 WS-01 |
| the same, a layer's experts alone (`skew`) | 512 / 1,024 / 2,048 / 4,096 | 5.71 / 6.10 / 7.68 / 11.67 ms: twice the rows for 1.5x the time | 105 WS-05 |
| Flash Next FP4-Local, two ranks | 512 to 2,048 (the control frame's bound, not the link's) | 16k cold 1810 to 2553-2598 tok/s; a rank 42.86 to 46.95 GiB; 4,096 2764 tok/s at 52.40 GiB | 111 WS-01 |
| GLM FP4-Local, two ranks | 512 to 2,048 | 4k and 16k cold 1.63 to 1.67x, 5.3 GiB a rank | 107 WS-03 |
| GLM FP4-Local, two ranks | 2,048 to 4,096, the default since | 16k 1.08x, 3,694 tokens 1.13x; 100,955 MiB a rank after load against 108,296 | 107 WS-03, 114 WS-08 |

- **A width's cost can hide on the host**: at 4,096 rows GLM's short prompts were slower (483 tokens 0.86 to
  1.04 s), not by any kernel (the 483-row window is 650 ms of kernels at either width) but by each snapshot's
  `cudaMalloc` of 142 MB, 50 to 100 ms with 109 GB of the device held against 5 at 2,048. Kept buffers fixed it
  (107 WS-03; [`states-and-marks.md`](states-and-marks.md)).
- **What a GLM window row costs a rank**: 3.54 MiB measured, of which KDA's per-layer saves (proj, k, v, g,
  beta of 34 layers) 2.15 MiB, DSA's chunk partials 0.33 MB, logits 0.15, slots 0.13 (107 WS-03). Spec 116 cut
  the saves to 48 rows (5.70 GB less a rank at 4,096) and the partials to what is used (1,028 MiB less), by
  the sizes, not measured on a load.
- At GLM's 4,096 rows a layer-window of routed experts took 10.8 ms against 8.2 of its weights at the bus (114
  WS-08), 12.5 after WS-09's deal ([`experts-nvfp4.md`](experts-nvfp4.md)): wider windows would read them less
  often again; not tried (`--rows` is 4,096 at most, 107 WS-03).
- **A narrow window is mostly fixed cost**: on two ranks 7 rows 40 ms, 9 rows 53, a 16-row graphed window 68.5
  (110 WS-02). A pair that chose into 16 rows ran a 35-token turn as two windows; the chosen rows now span the
  window (a held turn 0.144 to 0.083 s).
- **A prompt beside replies** (074 WS-04): the 3.05 pack, 25,900 tokens beside 3 replies — chunks of 512 rows
  made a round wait 670 to 702 ms and the prompt's first token 33 s; 128 rows 292 ms and 53 s; 64 rows 212 ms
  and about 75 s (reckoned). A pool round with a prompt is never a graph (075).
- **Kernel forms by width, chosen by time, never by bits**: DeltaNet's wide chain from `gdn.wide_rows` 32; the
  wide experts from 768 rows; split-K summed inside the GEMM (CUTLASS serial split-K, the same bits) from
  1,024 rows (095 WS-03: the read-out at 2,048 rows 674 to 353 us, but at 24 rows 57 to 134, so decode keeps
  the parallel split); the indexer's selection as one launch from 33 rows (095).
- **The GLM indexer's dense limit does not pay to fuse** (spec 143 WS-04): a window whose rows all lie under
  `index_topk` (2,051 tokens) still runs `query_kernel`, `score_kernel` and `select_kernel` for lists nobody
  reads — 91.5, 73.8 and 76.1 us at 2,032 rows (`select` writes the rows' -1s, 16.7 MB), 13.9, 11.7 and 18.2
  at 496 — but twelve DSA launches a prompt make the scores 0.08% of a 2,048-token first token and 0.05% of a
  512's, all three 0.27%: under the 0.9% spread.
- **DeltaNet's block in one launch loses** (spec 143 WS-01): 384 threads (256 on the chain, 128 producing
  16-row stages), one block a head, 82,496 B — alone it lost to the three launches above 32 rows (1,100 rows
  1,294 against 1,235 us; 300 rows 378 to 409 against 321; 33 rows 64 against 48, nsys): the chain at one
  block a head runs 1.18 us a row against `wide_chain_kernel`'s 0.71. It wins only where it saves replays of
  wanted states (1,100 rows with marks at 511 and 1,024: 1,298 against about 1,993). Served 4k / 16k cold
  first tokens inside the spread (2.011 / 7.851 s off, 1.983 / 7.871 on).
- **A wide-window bug needs a wide test**: conv windows went wrong in every DeltaNet layer but the last, seen
  only by a test of at least 2 layers and 32 rows (051); a bound that was a `debug.assert` let a fast build
  write 79 rows into 65 (086, 93160e2).
- The pool's prompt chunk is what the replies' rows leave of the loaded rows (`docs/serving.md`; vLLM and
  SGLang make that budget 8,192, spec 101). The card's vLLM ran GLM's prompts in chunks of 4,096 (115 WS-02).

## KDA: chunked on a prompt, recurrent on steps (GLM, spec 116 WS-06)

- **Which rows.** Windows of more than 48 rows run `kda_chunk.cu`: chunks of 64 on absolute positions, the
  window's first chunk from its phase (pos % 64). Steps and verify windows (at most 48) keep the fp32
  recurrent chain and its replay. vLLM and SGLang do the same (FLA's `chunk_kda`, BT 64, on extend;
  `fused_recurrent_kda` on decode and verify); TensorFold is recurrent everywhere.
- **Arithmetic, never below vLLM's readable fallback** (FLA's Triton: tf32 dots, a bf16 inverse): the A blocks
  fp32 on the CUDA cores, the 16-row solve fp32, the WY products bf16 into `m16n8k16` with fp32 sums, the
  state fp32 between chunks. Chunked against the recurrent chain: outputs 9.9e-3 and 1.1e-2 of 2.65 and 2.26.
- **The cut.** `decode/draft.zig` `promptCut` cuts every prompt window but the last at a multiple of a model's
  `prompt_align` (GLM 64, Flash Next 1, its bits unchanged), shared, not copied. A hall part that cannot end
  at a chunk start within its share runs 48 rows, recurrent; the graphs' key holds a chunked part's phase.
- **Time**, a 4,096-row layer-window, a rank's 32 heads, alone: 7.81 to 8.00 ms to 4.68 to 4.76 (intra 3.07,
  state 1.16, out 0.46), some 0.42 s a rank over a 16k prompt's 136 layer-windows. The first build's 7.8 ms
  (intra 5.8) fell when the conv read 16 rows ahead and prefetched the next chunk's L2. Intra is one block an
  SM at 0.26 issued a scheduler: latency is the next lever.
- **The recurrent wide chain's cost**: `step_kernel` holds 25,664 B static (16 staged rows of fp32 k, q,
  decay), so 3 blocks an SM and 144 of its 256 blocks resident: two waves, each the whole serial chain (1.27
  us a row, about 2.5 TFLOPS) (116 WS-02).
- **Memory**, a rank at 4,096 rows: the saves 5.70 GB less (needed only for a replay, so 48 rows).

## Attention's key chunks and their merge

| Kernel | Before | Window form | Source |
| --- | --- | --- | --- |
| GLM DSA/MLA, FP8 cache, 4,096 rows | 24.96 ms (5 chunks of 512 keys a row, 1.34 GB fp32 partials a layer-window, merge at 210 GB/s) | 12.68 ms, from 128 rows | 116 WS-01 |
| the same, bf16 cache | 34.12 ms | 30.39 | 116 WS-01 |
| the same, 512 rows, FP8 | 3.12 ms | 1.67 | 116 WS-01 |
| Flash Next `attention_window.cu`, 16k context, FP8, 2,048 rows | 7.47 ms | 3.97, from 64 rows | 105 WS-05 |

- **The form**: a block a row (and its heads) runs the row's chunks in order, each from nothing as its chunk
  block did, and folds it into the running output with the merge's own contraction (`fma(o, a, co * b)` read
  in the merge's SASS); the last fold divides once. The same bits as chunks plus merge, and no partials
  written.
- **The cut was swept, not guessed**: GLM FP8 128 rows 0.74 to 0.53 ms but 8 rows 0.073 to 0.226; Flash Next
  64 rows 0.234 to 0.154 but 16 rows 0.098 to 0.121. Steps, drafts and short windows keep the chunks and the
  merge.
- A gathered key set (a DASH row's window plus 12 chosen 32-key blocks) keeps both forms' bits by one unit, a
  router block, in both: each unit's partial from zero, folded in ascending block order with
  `__fmaf_rn(o, a, __fmul_rn(co, b))` pinned; a tile's unit no row of it takes folds as the identity. The
  step form (a block a unit and head, the last folding) 12.4 us at 1 row, 23.2 at 4, 74.5 at 16; the 64-row
  tile 39 at 1 to 4 rows: the cut near 8 rows (147 WS-04, a quick timing).
- Served: GLM's 16k cold prompt 0.97x on two ranks (some 0.27 s, against 0.48 projected from the standalone
  figures); Flash Next's -0.34 s (-4%) by `generate`, the 16k row at par with the card's vLLM.
- A row's list of 2,048 to 2,051 keys makes a fifth chunk of 0 to 3 keys: run as one tile of the row's block,
  its keys asked for under the fourth chunk's last tile.
- References: vLLM's b12x sparse MLA prefill is one pass without a split; TensorFold splits by 512 in parts of
  128 rows (its partials some 40 MB, in L2). Not built: the fusion paid first.
- At decoding rows the chunks and merge are 1.4 to 1.8% of a Flash Next round (105 WS-06); a fix there cannot
  show outside a load's spread ([`measuring.md`](measuring.md)).

## Two ranks split a window's glue by rows from 256 rows

- From `split_rows` (256) each rank makes and reads out only its half of the rows between layers; below it the
  partials are all-gathered whole as TensorFold does. Flash Next: 16k first token 9.98 to 9.10 s (106 WS-02).
  GLM: 16k under `nsys` 13.60 to 12.80 s, the same bits (107 WS-03, 7860efd).
- Spec 116 WS-03 cut the kernel before each scatter in two row parts so the first part's scatter runs beside
  the rest: [`two-ranks.md`](two-ranks.md).

## Sources

Specs 051, 068, 074, 075, 086, 087, 095, 105 to 107, 110, 111, 114 to 116, 143, 147 (`specs/archive/`, 107 in
`specs/`); `docs/serving.md` ("Windows", "A conversation's states"); `docs/kernels.md` (`zs_glm5_kda_chunked`,
`zs_glm5_mla_attend`, `zs_attn_attend_window`).
