# `zig/src/models/nemotron/` instructions

Extends `../AGENTS.md`, `zig/src/AGENTS.md` and the root `AGENTS.md`. Nemotron-3.5-Transcribe-30B-A4B
(`model_type` `nemo_speechlm`, TensorFold's family `nemotron`) on NVIDIA's NVFP4 checkpoint (spec 151): a NeMo
perception module before a NemotronH backbone, served with DFlash2 at two ranks. Storage, defaults and
figures: `docs/recipes/`'s page when it lands (WS-10). Behaviour reference for the backbone: TensorFold's
`families/nemotron/` engine (the NVFP4 checkpoint itself, run as stored: no MLX export and no requantizing);
vLLM 0.32 with NeMo's `nemo_speechlm` plugin for perception, prompt and routes.

## The model

- Backbone: 52 layers of hidden 2,688 by `layers_block_type` (23 Mamba-2, 23 MoE, 6 full attention), an MTP
  block of attention then MoE (`llm.mtp.layers.{0,1}`). Mamba-2: 64 heads of 64, 8 groups, state 128, conv 4,
  fp32 SSM state. MoE: 128 routed experts of 1,856 (top 6, `relu2`, scale 2.5, sigmoid +
  `e_score_correction_bias`), one shared of 3,712. Attention: 32 query over 2 KV heads of 128, RoPE 1e4.
  Untied head, vocab 131,072.
- Perception: 128-bin log-mel (16 kHz, n_fft 512, hann), 8x feature stacking, a 32-layer encoder of 1,280
  (10 heads, qk_norm, RoPE on half a head, FFN x4), then `perception.proj` to 2,688. The checkpoint mounts
  a ParallelExpertEncoder (`config.pe_encoder`): the clip's mel runs once, `per_feature` normalized, and
  the encoder sees context-extended windows (`pe_core` 500 rows' core + 50 rows' context a side; without a
  PEE NeMo chunks the mel at `encoder_chunk_size_seconds`). Sortformer's streaming speaker encoder
  contributes the speaker infusion below; the CTC heads remain unread (`layout.dropped`).
- Storage (`hf_quant_config.json`, ModelOpt MIXED_PRECISION): attention's q/k/v/o and the experts NVFP4
  group 16; the Mamba projections, the MTP block and the encoder's linears per-tensor FP8; embeddings, conv1d,
  norms, gates and `lm_head` BF16.
  Mamba projections use their stored FP8 input scales in every window through shared FP8 tensor cores.
  Backbone relu2 experts use W4A4 in every window; MTP's requantized experts retain A16. Routed up
  projections use their group's largest input_scale; shared halves keep their own linear's scale.
  TP2: down-product K 928 pads to 960; MTP requantizes sliced shapes; conv indices cover `s.cd()` only.
  Attention's NVFP4 projections use checkpoint activation scales in every window (largest q/k/v
  input scale), widening normalized FP4 times block scales exactly for the existing BF16 GEMM.

## Files

- `config.zig`: the checkpoint's three config files — `config.json` (speechlm + `perception`),
  `llm_backbone/config.json` (the backbone), `hf_quant_config.json` (the map, `Quant.find` drops a module's
  `llm.` prefix to match it). `layout.zig`: every tensor's name, dtype and shape (`kinds`, a `Where` per layer
  family), resolved and counted by `Layout.init`, which refuses a tensor it has no place for; `home` is a
  tensor's shard. Tests: `zig/tests/models/nemotron/` (`fixture.zig` builds a small synthetic checkpoint);
  the host tier, and `test-nemotron` (`tools/check.sh nemotron`).
- The published backbone config can omit `num_hidden_layers`: derive it from `layers_block_type`, and
  reject an explicit count that disagrees. MTP q/k/v FP8 scales can differ: shared `linear.Linear.stored`
  preserves each through `fp8_block`, retaining the per-tensor path for a common scale. Never rescale the
  codes to the largest factor merely to fuse the projections.
  MTP token staging covers a whole prompt window; only chained draft positions use 16 slots.
- `glue.zig` wraps `zig/kernels/nemotron/`'s `zs_nemotron_*` launchers — the residual RMS norms, the sliced
  router and its top-k (shared experts as the last slots), the serial feed, the routed plan, `rest_rows` and
  the MTP's keyed greedy — TensorFold's kernels byte for byte. `glue_ref.zig` is their host twin, op for op
  (its `f32ToBf16` is `cvt.rn`'s, not `numeric`'s: a NaN is 0x7fff). `glue_test.zig` holds each launcher to
  the reference and a row alone to the row among others, bit for bit; `glue_ref_test.zig` is the host tier's.
- `mamba.zig`: the Mamba-2 mixer between in_proj and out_proj (`kernels/nemotron/mamba.cu`, TensorFold's
  `tf_nemo_conv`/`tf_nemo_scan` byte for byte, and zspark's `mamba_wide.cu`). TensorFold's state model: the
  fp32 SSM state and the conv's three-input base lag a window; a window replays the previous one's kept rows
  (meta {pos, parity, prev_keep}, int32 [4], from the parity buffers raw/xc/dt[1 - parity]) and commits
  through them, then writes its own rows to buffer `parity`, so rejected rows never touch committed state.
  `Mixer.window` (up to 16 rows, a step or a verify window) then `Meta.afterWindow(keep)`; `Mixer.prompt` (a
  chunk of any rows, `Scratch.rows` at most) then `Meta.afterPrompt(rows)`: it commits through the row before
  the last and leaves that row pending, exactly as its serial steps do, so a prompt's rows and state are the
  steps' bits and the next window replays as after a step. Both files build with `--fmad=false` (upstream's
  flags). A, D, dt_bias and the conv are widened to fp32 at load as TensorFold does (the conv tap-major, A =
  -exp(A_log) by `zs_nemotron_mamba_a`). dt's clamp is `time_step_limit` ([0, inf) on Transcribe), not
  `time_step_min`/`max` (the init range). `mamba_test.zig`: the window against `glue_ref.zig`'s conv and scan,
  window, rollback and prompts of 1 to 2,049 rows against serial steps, inputs at softplus's and exp's ends.
  `attention.zig`: one NoPE GQA layer (q/k/v stacked NVFP4/FP8 + o_proj through `linear.Linear.stored`), a bf16
  KV cache a sequence, `window` (≤16 rows) and `prompt` (a chunk) over `common/attention_prep.cu` (a null `inv`,
  no rotation) and `common/attention*.cu`; a row's bits are the same in a serial step, a window and a chunk.
- `prompt.zig`: the text around a clip, host only — the locator's expansion to a clip's audio rows
  (`clipRows`, `mel.zig`'s `rows` at `config.perception`'s params; a single pass where the checkpoint mounts
  a ParallelExpertEncoder, as this one does — `config.pe_encoder`; the 3,600 s cap) and `expand`, the ChatML
  text the checkpoint's template renders for a system-less request with thinking off
  (the transcription route's prompt; the card's 40 languages for `language`), the fields' sanitizing
  (`audio/text.zig`, shared with `qwen3_asr/transcript.zig`), and the `nemotron_v3` reasoning split.
  `prompt_test.zig` is the host tier's; `zig/tests/audio/text_test.zig` the helpers'.
- `mel.zig`: the featurizer's log-mel on the host (pre-emphasis 0.97, the centered STFT in f32 with the
  checkpoint's `fb` and `window` tables, `log(x + 2^-24)`; no dither, no `pad_to`), `normalizePerFeature`
  the PEE's `asr_normalize_type`, `chunkSpans` the `encoder_chunk_size_seconds` chunks with NeMo's fold of
  a short tail, `rows` a clip's audio rows as `audio_token_estimator.py` counts them (`check.sh nemotron
  mel`; the WS-04 fixture clips under `ZSPARK_REFERENCE_NEMOTRON`). `perception.zig`: a clip's 16 kHz
  samples to the bf16 [rows, `output_dim`] rows the forward writes over the prompt's audio placeholder
  rows (`decode/placeholders.zig`). Under the PEE (`config.pe_encoder`, what this checkpoint mounts) the
  whole clip's mel, normed, runs `pe_core`-row cores with `pe_left`/`pe_right` context a segment each —
  a segment's keys masked at its clamped length — the kept core rows compacted by `zs_embed_rows`;
  without one, `chunk_seconds` sample chunks. The stack: the host stacking 8 frames a row,
  `pre_encode.proj`, `embed_norm`, the FP8-layer encoder (w_qkv by `zs_nvfp4_prompt16`, the qk norms and
  RoPE by `kernels/nemotron/perception.cu`, the per-segment full attention `perception_attention.cu`,
  the biased linears `prompt_bias.cu`'s FP8 mode), `final_norm`, the PEE's `asr_norm` plus the
  speaker infusion (`speaker.zig` below), and `perception.proj`. `Perception.rows` is `clipRows`' count.
  `perception_test.zig`: the prep and attention kernels against the host's fp32, the whole stack on the
  fixture's synthetic FP8 weights both paths (cosine against a host reference, a chunk alone the same
  bits as beside others).
- `speaker.zig` (WS-11): the PEE's speaker path as vLLM serving runs it — Sortformer's streaming
  forward a window (spkcache, FIFO, `_compress_spkcache`) to the speaker activity, then
  `_fuse_diar_and_asr` (`asr_norm` + `spk_kernel_scale` x `diar_norm(activity > 0.5) @ diar_kernel`)
  before `perception.proj`. The call chain and every threshold: `REFERENCE.md`, "The speaker path".
  Subpixel upsampling is a full-channel Conv1d (`groups=1`), packed to the shared BF16 tensor-core
  GEMM. PEE overrides its speaker output stride to the ASR subsampling (8), regardless of the nested
  checkpoint value 1. Fusion uses natural-log sinusoid frequencies and a single-warp LayerNorm
  reduction; its regression uses the real 1,280-wide ASR state and checks row independence.
- `moe.zig`: a MoE layer — `glue.zig`'s router and top-k (sigmoid + `e_score_correction_bias`, scaling 2.5),
  `routed + 2` relu-squared NVFP4 experts on `weights/experts.zig` `initRelu2` (the shared expert folded in as
  TensorFold's two pseudo-experts: its up's row halves, its down's column halves), the slots' fp32 rows and
  weights, and `combine` = `glue.addMoeNorm` (routed weighted, the shared added) into the next block's normed
  rows. Stateless per row: `window` and `prompt` are one `run`, the experts' pass picked by rows; zspark plans
  every slot in one call and keeps the slots' yk fp32 in a prompt chunk too (both depart from TensorFold,
  whose decode plans the shared halves aside and whose prefill writes yk bf16). The MTP layer's experts
  store FP8 (`mtp.layers.1.mixer.*`): `load` requantizes them to NVFP4 on the host
  (`weights/requantize.zig`, TensorFold's recipe) — a drafter-only conversion under the root AGENTS.md's
  quantization exception (it moves offers, never a token; WS-07 measured 109/420 drafts kept and
  60.06 tok/s on the 39 s bench); an FP8 expert under `backbone.` is refused. `moe_test.zig`: the slots
  and weights against an fp64 host, ties to the lower id, and a row's bits a serial step's in a window
  and prompt chunks of 1 to 2,049 rows.
- `ranks.zig` (WS-06): the two-ranks cut — `Share` (a rank), `Shapes.of` (the mixers' halves: 16 query/1 KV
  attention head, 32 Mamba heads over 4 groups, expert width 928, `lm_head` half the vocabulary), `divisible`,
  the `in_proj` segment and conv/head/norm index sets (`mambaInRows`, `mambaConvCols`, `mambaHeadRows`,
  `mambaNormRows`), the `linearRows`/`linearCols` gathers (csf scales decode to the plane first — a gather
  reorders what a csf window cannot name), and `report`'s byte account. `ranks_forward.zig`: the forward over
  `ranks/`'s `Gather` — every layer's product an fp32 partial in `Ranks.parts` (`moePartial` folds a rank's
  slots first), met then `sumRows`'d rank 0 first and rounded once into the residual. `Held` drives one
  process's ranks in lockstep — every held rank's layer before the fold's gather — so the loopback meets
  real partials; `.nccl` all-gathers per layer in a pair. `logitsRow` gathers a scored row's halves;
  `stage`/`forwardRows`/`Op.window` split the embed from the layers for a lead's told windows.
  `commands/ranks.zig`: `zspark ranks DIR check ID..` runs the prompt serial and split and compares each
  row's picks equal, cosine at least 0.999 and RMS difference (the partials' sum order differs from the
  serial product's); `generate` decodes greedy on the split path. `rank=R at=ADDR:PORT` makes this process
  one rank of two — `join.zig`'s channel and NCCL, the gathered logits' digest met over the channel
  (`check`) and rank 0's reply printed.
- `dflash2.zig`: the model's side of `decode/dflash2/` for
  `NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2` (spec 151 WS-07) — `Opened` (null `sliding_window`,
  `Config.parseFull`), `published`, `fit`/`enter`/`given`; `checks/random_drafter.zig` is the tests'
  synthetic drafter. The shared code takes this drafter's selector and convolution fields as they are;
  the config's null window is `Config.window == null` (whole context, flat, never a ring) and its GQA's
  q_inner (4,096) is not hidden — `Window` gained `.full` for it.

- `dflash2/` is the target's side: `pass.zig` the `Target` (one rank, `wide` false, `owns_heads`),
  `drafter.zig` `Of(held)` (both drafter ranks read the same embedding and taps of a one-rank target,
  but each head and its stored rescore rows cover only that rank's half of the vocabulary) and
  `draftHeads` (four-bit copies or stored), `rig.zig` `initRig` (it calls
  `state.Buffers.setTaps` itself) and `drafter_stored`. `generate drafter=DIR` verifies DFlash2's drafts
  against the backbone; `mtp` and `serial` stay selectable. `REFERENCE.md` holds the tap-index
  convention (vLLM's `target_layer_ids` + 1, `hidden_states + residual` after block `idx`); the
  forward_test's serial-vs-window taps hold it.

- What the target owes DFlash2 (WS-07; taps, `Target.Of`, a rejected draft): `REFERENCE.md`.

## When editing here

- Do: keep the layout to what the published index and headers say (`~/.cache/zspark/scratch-151/manifest.tsv`
  is the manifest table); a tensor this checkpoint gains or loses joins or leaves `kinds`.
- Don't: resolve `ctc_timestamp.*` (a spec non-goal), or quantize at load. `perception.encoder.
  diarization_model.*` is WS-11's, served bf16 as stored.
- The draft head is `lm_head` whole: TensorFold's `draft_ids.txt` was built for Nemotron-3.5-Lightning's
  tokenizer (its ids 100 to 141 are Transcribe's `<spk:>`/`<ev:>`/`<sty:>` tokens — 42 of the list's ids
  map to different tokens), so the MTP scores the full head instead. Keep `draft_ids.txt` out of this
  tree; a draft subset built for Transcribe's tokenizer could return.
