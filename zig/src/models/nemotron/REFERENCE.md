# `zig/src/models/nemotron/` reference

Read when a task touches what a section names; the `AGENTS.md` beside this file binds every task here.

## The speaker path (WS-11)

What vLLM serving runs for the ParallelExpertEncoder's speaker infusion (`speaker.zig`). The call chain:
`salm/model.py._process_audio` wraps `perception(...)` in `encoder.online_inference()` — no
`spk_targets`, no `return_ctc_timestamp_inputs` on the plain transcription route — so `forward`
takes `_forward_online_impl` (`parallel_expert_encoder.py` 1881): `ceil(total/chunk_feat_len)`
windows of `online_inference_length` 500 core rows + `chunk_left/right_context` 50, mel frames
(sub 8). ASR applies `per_feature` normalization once per clip; Sortformer takes the raw mel
(`diar_normalize_type=NA`). Both use the full mel tensor width, including its masked tail; valid lengths
mask attention. `_encode_asr_windows` keeps each window's core rows; the diarization branch
(`run_streaming_diar`, spk_targets None serving) runs `forward_streaming_step` a window on the
same span, stateful: `chunk_len`/`fifo_len`/`spkcache_update_period`/`spkcache_len` =
online_inference_length / `diar_fifo_len` 40 / `diar_spkcache_update_period` 300 /
`diar_spkcache_len` 188 (not the modules' defaults). A step: FeatureStacking of the window's mel
rows (grid at `context_start`, the kept core may land mid-row), concat [spkcache | fifo | chunk]
pre-encoded embs, the 31-layer encoder (bypass_pre_encode), `encoder_proj`, `forward_infer` (no
transformer_encoder; `upsample_hidden` the full-channel subpixel Conv1d k=3 x8), sigmoid speaker logits masked
at the packed length, `streaming_update` (FIFO append, pop `update_period` into spkcache,
`_compress_spkcache` when over; skipped on the last window) and the chunk's native preds at
(spkcache + fifo + lc_enc) x `upsample_factor` — `lc_enc = round(left/8)`, `rc_enc =
ceil(right/8)`. The PEE constructor overrides `output_subsampling_factor` with the ASR subsampling
factor (8), pooling eight native 10 ms predictions per ASR row before `_align_diar_frames`.
The stored nested value 1 does not control this mounted path. Then `_fuse_diar_and_asr`: `_speaker_features` thresholded
at `speaker_activity_threshold` 0.5, `out = asr_norm(states) + spk_kernel_scale(1.0) x
(diar_norm(features) @ diar_kernel)` — `diar_kernel` the sinusoid [8, asr_d_model], the norms
LayerNorms eps 1e-5 — then `perception.proj`. Sortformer n_spk 8, fc_d_model 512, tf_d_model 192,
sil via `learnable_sil_emb` (`spkcache_sil_frames_per_spk` 1), `sil_threshold` 0.2,
`pred_score_threshold` 0.25, `scores_boost_latest` 0.05, `strong/weak_boost_rate` 0.75/1.5,
`min_pos_scores_rate` 0.5, `max_index` 99999; eval `_permute_speakers` never runs. The
CUDA-graph path (`encoder.online_graphs`) replays the same calls; long audio also pads the clip
to full windows plus a `last_window_bucket` tail — masked trailing frames only.
The subpixel convolution uses `groups=1`: every output channel reads all 192 input channels and
three temporal taps. Native execution packs reusable [T, 192*3] scratch and multiplies by the
checkpoint's unchanged [8*192, 192*3] weights through the shared BF16 tensor-core GEMM, then ReLU.
The speaker sinusoid uses natural-log frequencies `10000^(-2i/D)`.


Only each new chunk's stacked mel passes through `pre_encode`; cache and FIFO rows are already
embeddings and are prepended afterward. Streaming state retains those pre-encoder embeddings before
the encoder overwrites its working rows. A request resets the streaming state before its first window.

## The accuracy compare (WS-05): `zspark compare`

`commands/compare.zig` + `checks/reference.zig` replay a clip's reference trajectory on the
native stack: `export_model.py`'s `NAME.safetensors` (prompt.ids, audio.placeholder_rows,
forward.num_rows/positions/input_ids, hidden.Lxx + residual.Lxx committed streams, tokens.greedy,
logits.stepNN, enc.out) beside `NAME.mel.safetensors`' `pcm.16k`. The command encodes the PCM,
compares it with `enc.out`, injects the features over the placeholders (`state.audio`), runs the
prefill and the first 32 teacher-forced serial steps (greedy ids fed, every decoder layer tapped),
and folds each layer's completed stream against the fixture's committed rows only.
- Honest mapping: `checks/reference.zig`'s `rows` marks a verify row mapped only while its window's
  ids continue the committed `prompt.ids`+`tokens.greedy` chain and its position precedes the next
  forward's start; a partial verify's candidate rows (rejected drafts) never compare. Rows the
  reply produced but no committed reference row covers are counted and skipped, never silently
  folded. `forward.input_ids` absent entirely maps by position alone (the exporter's inputs_embeds
  case); the prefill then takes `prompt.ids` as its own chain.
- Logits: `logits.step{f}` stores the forward's *last* `shape[0]` rows only (the prefill's last row;
  a verify window's all). A stored row is comparable at the serial step whose fed position equals
  the row's position, only where the window's rows through it held the chain. The tensors carry the
  tokenizer's 131082 columns against the checkpoint's 131072 — the first 131072 compare; the note
  prints once.
- `ref_enc` injects the fixture's `enc.out`, isolating the backbone from the native perception path.
- Teacher-forced picks are diagnostics. The command resets state and generates a separate free greedy
  reply, feeding its own picks until EOS or the card's context is exhausted. Only that reply is scored.
  `truth=` supplies ground truth; otherwise the reference transcript is the yardstick. English uses the
  fixture scorer's lowercase/punctuation removal; Chinese uses CER without spaces and retains punctuation.
- `steps=N` bounds teacher-forced diagnostics, not free generation. All prefill chunks must cover
  positions 0..prompt_len-1; partial captures fail loudly. Layer tensors stay BF16 in the file mapping;
  native prefill uses at most 2,048 rows per window and downloads only active tap rows.
- Layer taps compare BF16(hidden + residual) at committed positions. Measured tolerances belong to
  the accuracy evidence in the checkpoint recipe and spec; they are not inferred from the diagnostic.

## DFlash2 (WS-07): what the target owes the drafter

The target's side lives in `dflash2/` (`pass.zig`'s `Target`, `drafter.zig`'s `Of(held)` and `draftHeads`,
`rig.zig`'s `initRig`): `decode/dflash2/pass.zig`'s contract — `ranks`, `lead`, `Op` with
`flash_taps`/`flash_block`/`flash_round` for a leading rank, `Of(held)`, `Job`, `owns_heads`,
`initRig`/`initRigHeld`, `wide`.
- Taps: `setTaps(c.taps)` — every kept row's hidden after decoder layers 1, 5, 19, 29, 41 and 51
  (zero-based block ids; vLLM `gpu_model_runner.py` maps `target_layer_ids` + 1 to
  `aux_hidden_state_layers`, `nemotron_h.py` capturing `hidden_states + residual` after block `idx` at
  slot `idx + 1`) — the stream itself bf16, one residual stream, no mean — into bf16 [rows, 6 x 2688],
  `target.taps()` the same buffer on both drafter ranks (one-rank target). `forward.zig` captures after
  each block's fold (the last layer's at the final fold). A row's taps are that row's alone (the row
  rule; `forward_test.zig`'s serial-vs-window equality).
- `Target.Of(held)`: `tapLayers()` = `c.taps`; `mains(heads, widened)` each drafter rank's `Main` —
  `embed` (the model's table bf16 [131072, 2688]), `head` (a four-bit copy or stored BF16 rows of
  that drafter rank's vocabulary half) and `rows` (the same half's stored BF16 rows for `pass.rescore`).
  On a whole target `draftHeads` cuts the head in half; a TP2 target already holds its own half.
  `taps()` returns the buffers above; `deinit` owns nothing (the model is the caller's).
- A rejected draft: the verify `commit(keep)` is the target's own — `mamba.zig`'s parity replay rolls
  the conv and SSM state of a rejected window back by replaying the kept rows alone (`Mixer.window` +
  `Meta.afterWindow(keep)`), so no DFlash2-specific rollback is owed. The drafter's own context stays
  as written: its context rows past the kept end are rewritten by the next `addTaps` (a block never
  moves the context).
- MTP stays selectable: the drafter is the `draft.Drafter` the session is given, not a fixed part of
  the model.
- The drafter directory's weights are OpenMDW-1.1, the user's own copy on disk; nothing writes them.
