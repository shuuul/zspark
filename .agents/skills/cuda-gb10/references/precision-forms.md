# Precision forms by window on the GB10

Which precision each operand takes, by checkpoint and by window kind, what the narrower forms bought on this
chip, what their error was, and what was refused. Part of the `cuda-gb10` skill. The instructions themselves
(`mxf8f6f4`, `mxf4nvf4`, their fragments) are in `SKILL.md`.

## The rules (the owner)

| Date | Rule | Source |
| --- | --- | --- |
| 2026-10-10 | The model that emits a reply's tokens runs as its checkpoint stores it: zspark requantizes no weight of it. A drafter may be quantized at load | root `AGENTS.md` |
| 2026-10-17 | Flash Next FP4-Local: MXFP8 linears and routed experts on bf16 rows (W8A16, A16), as QAD trained them and TensorFold runs them | 105, Decisions |
| 2026-10-19 | Activations may take the precision of the card's launch configuration (the Hugging Face page's command), read in vLLM's source, never less in any phase. A rounding vLLM makes in one phase only is not taken where the row rule would carry it into the other | 114, 115 |
| 2026-10-19 | GLM: a row's form follows its window's kind (option B): bf16 to 48 rows, quantized above as the card | 115 |

## GLM FP4-Local's forms (spec 115 WS-01, T = 48)

| Operand | Step, verify window, prompt window of 48 rows or fewer | Prompt window of more than 48 rows |
| --- | --- | --- |
| Routed experts (NVFP4 weights) | A16: bf16 rows, codes widened in registers, `m16n8k16` (`zs_nvfp4_experts_a16`) | W4A4 as b12x's `prefill_a4`: gate/up rows under the layer's one global scale, down's per expert, `mxf4nvf4` m16n8k64 |
| MXFP8 linears (KDA, dense MLPs, head) | bf16 rows (lane matmul) | MXFP8 rows, an e8m0 a 32 (`qmm_a8`) |
| Block-FP8 linears (MLA, shared experts) | FP8 rows, a token's 128-group, every phase | the same |
| DSA's queries on the FP8 cache | e4m3 per (head, 128), every phase (114 WS-05) | the same |
| The indexer | Hadamard-rotated queries and pooled keys, e4m3 under a ue8m0 a 128, 132 B a key, every phase (116 WS-05) | the same |
| The ranks' partials | fp32 | bf16, as vLLM and SGLang (116 WS-04) |
| Down's per-route sums, the MoE combine | fp32 | fp32 (above the card's bf16 atomics) |

- The 48 covers both of the card's thresholds: b12x's A4 prompt parts are dropped whenever a CUDA graph is
  being captured, GLM's MoE is captured inside piecewise graphs up to 24 rows with MTP, and the MXFP8 "auto"
  plans keep bf16 rows on small M — so no window is below the card's (read in vLLM's image).
- The MTP layer and DFlash2 run with every row quantized: they only draft. Flash Next's forms are unchanged.
- Kept above the card: down's per-route sums fp32, no flush of tiny weight scales, IEEE exp and quotient in the
  A16 SwiGLU, prompt windows of 25 to 48 rows A16 where the card's are A4.

## What the narrower forms bought (GLM, two ranks, 16k cold prompt unless named)

| Change | Gain | Source |
| --- | --- | --- |
| Routed experts W4A4 at a 2,048-row window | layer 3's real picks 13.68-14.25 ms (A16) to 11.88: not the products' speed (gate and up are at the bus in both) but 128 pairs a reading the small rows allow, the weights read 1.03 times for 1.37 | 114 WS-03 |
| FP8 rows on block-FP8 and MXFP8 linears | 1.29 to 1.69x at 2,048 rows (63.9 to 86.2 TFLOPS on KDA's qkv); decoding at the bus either way | 114 WS-04 |
| WS-03 and WS-04 served together | 16k first token 12.59 to 11.3 s (0.90x), decode unchanged | 114, 2026-10-19 |
| DSA's queries FP8 | 512-row window 4.75 to 3.12 ms, 1 row 80 to 60 us; 16k 0.95x with WS-07 | 114 WS-05 |
| The indexer FP8 | 4,096-row window's scores 8.66-8.93 ms to 1.97-1.99 (about 60 TFLOPS FP8); 4 rows 34.7 to 13.8 us; 0.09 GiB less a rank at 262,144 | 116 WS-05 |
| Partials bf16 | the side stream's scatters 526 to 301 ms on a 9k prompt, first token 163 ms (3%) less | 116 WS-04 |
| FP8 KV cache (`fp8_ds_mla`, the default) | 1,364 MiB less a rank at 262,144; cold prompts 0.99x | 107 WS-03 |

## Errors, and how to judge them

- The bar is vLLM's own spread, not a tolerance chosen: two runs of the card's command are 0.1272 a token apart
  on the 5,132-token score text; vLLM at A16 is 0.20 from the card, its image default (W4A4 every phase) 0.233
  from itself. So zspark's distance is bounded by vLLM's across forms (0.2304, two standard errors 0.2377) and
  nll within 0.02 (115 Decisions, `ranks_vllm.zig` `across`).
- The A4 residual is A4's sensitivity, not a step zspark lacks: one last-bit change of down's pairs moves the
  score as far (0.2317) as zspark is from the card (0.2321), as vLLM's W4A4 runs part 0.233 (115 WS-05).
- Two A4 runs are farther apart than an A4 run is from an A16 run. Rows that differ upstream in the last bits
  flip codes independently, so the distances add roughly in squares. Flash Next (143 WS-07): zspark's A4 is
  0.3085 from vLLM's A4 and 0.2798 from its A16, where zspark's bf16 is 0.2636 and 0.1980. A clause "no farther
  from the A4 reference than bf16 is" fails for any correct A4, so judge an A4 form by nll and by its distance
  from the A16 reference. To test a kernel, feed vLLM's own dumped rows through zspark's: on Flash Next's
  layers 3, 15 and 45, x's and down's input codes and scales were b12x's byte for byte, and each route's output
  agreed within 0.0017 (b12x stores a route bf16).
- Flash Next's A16 reference is not deterministic, its A4 is (143 WS-08): four A16 runs of the score text lie
  0.212 to 0.220 a token apart pairwise, and vLLM's A4 (two runs bit-equal) 0.2754 to 0.2786 from them. Judge
  against the runs' mean — there the card and the A4 sit at 0.2428 and 0.2431 — or inside the A4's own spread
  across runs; per-run gates are decided by that noise (the card's excess ranged +0.0044 to -0.0062, standard
  error 0.0055; a 2^-9 rounding moved the card 0.004 on fixed runs).
- The card's per-linear choice of A16 against quantized prompts is raced, not fixed: b12x's
  `gemm.blockscaled_precision` times them per linear (`gemm/blockscaled/_tuning.py`), at the 4,096-row capacity
  regime every uncaptured chunk runs; captured counts (1, 2 rows) have their own. The winners sit in
  `~/.cache/b12x/compile/preparation/<digest>.json` (root's; copy out with a container), keyed by a digest; log
  `B12xBlockscaledLinear.run` to learn which linear took which. Flash Next: every MXFP8 linear quantized at
  1,241 to 4,096 rows except DeltaNet's `in_proj_ba` (2560 -> 96, A16, tile_n 128, split_k 2); all A16 at 1 and
  2 rows. vLLM's linears quantize with Triton `_quantize` (`_quantize.py:80`), byte-equal to `qmm_a8.cu`'s (0
  codes and scales of 13 dumped linears differ); its GEMM is an fp32-accurate product of those codes.
- Layer 3, W4A4 against A16: a pair's expert output 13.4% relative L2 (at most 20%), a row's routed sum 7.9%;
  no group's scale saturates (down's largest 0.78 of 448) (114 WS-03).
- FP8 linears, four layers and the head in loopback: logits 3.3% RMS off, top token kept 45 of 48, KL 1.6e-3
  nats a row (114 WS-04). Under T = 16 and the card's A4 form: 5.0%, 46 of 48, KL 4.4e-3 (115 WS-03).
- The FP8 cache: nll 2.21824 against bf16's 2.22185, vLLM 2.22567 and 2.22472 (107 WS-03).
- The FP8 indexer moves selections: TensorFold's inputs coded FP8 on the host move 8 of 512 pools at layer 3
  and 15 at layer 7 (116 WS-05).

## Flash Next and the caches (older checkpoints included)

| Form | Finding | Source |
| --- | --- | --- |
| KV cache bytes a position and layer | bf16 2,304, fp8 1,296, fp8-nvfp4 1,072, NVFP4 848 | 100, 2026-10-17 |
| fp8-nvfp4 cache, 4.05-bit pack | +0.0088 +- 0.0019 nats; over 133,623 tokens FP8 +0.0028 +- 0.0078, fp8-nvfp4 -0.0275 +- 0.0068 (unexplained) | 100 |
| NVFP4 keys | closed by the owner, reckoned: 0.66 GiB a stream at 262,144, attention 0.56% of a round's trunk, about +0.015 nats or more | 100 |
| An FP8 cache and drafts | costs kept drafts, so the MTP head's cache is bf16 ([`drafting.md`](drafting.md)) | 052, 100 |
| Block FP8 for bf16 weights | +0.0075 nats (noise 0.005), prose 40.5 to 46.1 tok/s; refused as a default (run as stored), drafts only | 070, 071 |
| Router gates | never quantized: 2.8e-3 of error moves an expert in 383 of 2,448 rows | 070 |
| FP4 activations on Flash Next | does not pay: 1.08 to 1.10x slower | 015 |

- How small a score difference is noise: under 0.007 nats is rounding order (079 WS-02); `score`'s own
  spread is 0.0032 to 0.0037, and any perturbation, even 5.3e-5 at layer 0, draws afresh (about 18% of expert
  picks differ, replies part after 8 to 165 tokens). Judge by seeds: `score ... perturb=1e-3 seeds=8` (100).
- A two-rank head's experts are quantized whole then cut, not TensorFold's slice then quantize: within 0.0069
  relative rms, each rank's part keeping its own tensor scale (090 WS-04).

## Refused

- Down's slots in bf16 in every phase (114 WS-10): b12x rounds a pair to bf16 on prompt rows but sums in fp32
  at decode; one form for all windows would carry the rounding into decoding, below vLLM's. Forms follow window
  kinds, so the quantized prompt form stores each route bf16 (fp32 of bf16, EPI 6), as the card's A4 FC2 and
  its A16 prefill (`kernel.py:283`, `B12X_W4A16_PREFILL_FUSED_SUM` unset) (143 WS-08).
- W4A4 experts and FP8 rows in every phase (114, 115): below the card at decoding and a 6-row verify once the
  card's environment was read. A16 at decode lost nothing: code-long 44.0 to 46.5 tok/s.
- A16 decode's split of one row's gate and up over four warps (114 WS-02): 173.9 for 180.5 us alone, nothing a
  load could separate.
- The card's launch environment sets `B12X_W4A16_A4_PREFILL=1`, so its 16k prompt rows are A4, not A16 (114
  WS-06): read the launch environment, not the image's defaults.

## Sources

Specs 015, 052, 070, 071, 079, 090, 100, 105, 107, 114, 115 (full text at `47d745aa`), 116, 143; `docs/recipes/glm-5.3-flash-fp4-local.md`.
