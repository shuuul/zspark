# `src/models/qwen3_asr/` instructions

Extends `src/AGENTS.md` and the root `AGENTS.md`. Qwen3-ASR-1.7B and 0.6B (`model_type` `qwen3_asr`, spec 138;
0.6B spec 140 WS-09), bf16 as stored: an audio tower before a Qwen3 language model, served as
`/v1/audio/transcriptions`, and chat with audio parts as vLLM serves it (spec 150). No
TensorFold source: the `qwen-asr` 0.0.6 package's transformers backend (transformers 4.57.6,
`modeling_qwen3_asr.py`; Apache-2.0) is the behaviour reference, vLLM's `qwen3_asr.py` what the route takes beside it.
Fixtures: `tools/reference/qwen3_asr/` (its `README.md` lists the package's quirks). Kernels: `kernels/qwen3_asr/`,
C symbols `zs_qwen3_asr_*`. Tests: `zig/tests/models/qwen3_asr/`, the area `tools/check.sh qwen3_asr`. Storage,
defaults and figures: `docs/recipes/qwen3-asr-1.7b.md`, `qwen3-asr-0.6b.md`; the owner's W4A16_NVFP4 builds (spec
140): `docs/recipes/qwen3-asr-1.7b-nvfp4.md`, `qwen3-asr-0.6b-nvfp4.md`.

## The model

- Audio tower: the 128-bin log-mel in chunks of 100 frames (13 tokens each; the sinusoid's rows 0 to 12 restart in
  every chunk); three Conv2d k3 s2 p1 with GELU (1, 480, 480, 480), `conv_out` 7,680 to d_model; pre-LN layers
  (heads of 64) attending in block-diagonal blocks of 104 tokens; `ln_post`, `proj1` with GELU, `proj2` to
  `output_dim`. `tower.Shape` comes from `audio_config`: 1.7B d_model 1,024, 16 heads, 24 layers, FFN 4,096,
  `output_dim` 2,048; 0.6B 896, 14, 18, 3,584, 1,024. Another shape is refused (`checkConfig`); a new d_model needs
  its instantiation in `tower_attention.cu` and `tower.attention_widths`.
- Language model: Qwen3-1.7B, 28 layers of 2,048, 16 query and 8 KV heads of 128, q and k RMSNorm, theta 1e6 (0.6B:
  28 layers of 1,024, FFN 3,072, the same heads; config only). MRoPE's
  axes take one position for audio: plain split-half 1D RoPE. `thinker.lm_head` is the head and the embedding table
  (`embed_tokens` is the same bytes, not loaded).
- The tower's embeddings replace the prompt's `<|audio_pad|>` rows (`decode/placeholders.zig`, shared with Flash
  Next's images). Greedy decoding, no drafter (`draft.zig`: `Decoder` with no chain).

## Files

- `config.zig`: `text_config` and the audio ids; refuses an untied head, `attention_bias`, a `hidden_act` but silu.
  `Config.quantization`: `quantization_config`, null or ModelOpt's W4A16_NVFP4 (`weights/modelopt.zig`'s
  `Uniform`, its excluded modules bf16), anything else refused. `layout.zig`: the language model's tensors by
  name, dtype and shape, a linear through `modelopt.read` (the tower checks its own); the head is `embed_tokens`
  where an export wrote no `lm_head`. `model.zig`: its device weights, q, k, v and gate, up stacked as
  `linear.Linear.stored` (bf16, or NVFP4 into `qmmf` with bf16 activations).
- `forward.zig`: a window of rows after a `State` (caches, the position a device word); `Graphs` captures a step per
  (rows, context bucket); `steps` a row in each of several states, `StepGraphs` a graph per lane count.
  `draft.zig`: `greedy`. `pieces.zig`: a long clip's pieces, each greedy in its own lane, stepped together. `tower.zig`: `Tower.encode(features, frames, out)`, mel as bf16.
- `transcript.zig`: the package's prompt, language names, `parse_asr_output` and its repetition fold; host only.
- `served.zig`: the engine of `Engine.submitTranscription`, a file to its transcript, one request at a time; 1,200 s
  of audio, 4,096 reply tokens unless the request names fewer. `commands.zig`, `commands/`: `generate` and `serve`.

## Kernels

`kernels/qwen3_asr/`: `tower.cu` (convolution patches, bias and GELU, positions), `tower_attention.cu` (the 104-token
blocks; the width a template argument, 1,024 and 896, chosen by the output's columns). From `common/`: `attention_prep.cu` (q and k norms, RoPE, k and v into the caches; Unlimited-OCR's too,
without norms), `attention.cu` (windows under 64 rows), `attention_group.cu` (from 64, the same bits, a key read once
for a group of rows), `glue.cu`
(`zs_residual_rmsnorm`: the K slices of `o` and `down` summed in slice order, added, then the next norm),
`layernorm.cu`, `embed.cu`; `bf16/gemm_bias.cu` for the tower's GEMMs.

## Exactness and tests

- Row rule, language model: K slices follow the weight's shape and key chunks merge in index order, so a window, a
  prompt chunk and a graphed step give a row its bits. The two `row rule` tests of `forward_test.zig` hold it (final
  rows, a verify window's logits, a long prompt against groups of rows): a new window or attention form passes them.
- Tower: a chunk (convs, GEMM rows, norms) or a block (attention) alone decides its values; `tower_test.zig` holds a
  104-token block to the same bits alone, beside other audio and as another block.
- Against the package: near, not equal (its sdpa is bf16, here softmax and sums are fp32; norms take `rsqrtf` where
  torch takes `rsqrt`). Cosine floors: 0.999 a layer or tower stage (0.6B's tower 0.998: one zh_30s row is unstable
  in bf16 after its last layer, the package's own bf16 against its fp32 0.99912 at `ln_post`), carried layers 0.9,
  logits 0.99; a reply may leave the package's only at a near tie (`tie_steps`, 2 bf16 steps).
- A test opens a fixture scenario (`en_30s.free`: `prompt.ids`, `enc.proj2`, `tokens.greedy`) and runs on every
  checkpoint set (a `Subject`: `ZSPARK_QWEN3_ASR` with `ZSPARK_REFERENCE_QWEN3_ASR`, `ZSPARK_QWEN3_ASR_0_6B` with
  `ZSPARK_REFERENCE_QWEN3_ASR_0_6B`, each NVFP4 build), skipping when none is; `forward_test.zig` loads the model once, so none is in
  the `whole` row of `tools/test_areas.tsv`; its replies through zspark's own tower decode only under `ZSPARK_QWEN3_ASR_TOWER_REPLIES`. `check.sh qwen3_asr [FILTER]` takes the GPU; `transcript_test.zig` runs without it on `text_cases.json`
  (`text_cases.py`). The route's host test and chat's audio parts: `zig/tests/server/transcriptions_test.zig`;
  `served_test.zig` runs `served.zig` behind the host on `forward_test.zig`'s subjects (its loads), a chat's reply
  against the route's transcript of one clip.

## Departures from the package and vLLM (spec 138)

- The package's prompt and parsing, not vLLM's: the empty system turn kept, the transcript stripped and its
  repetitions folded. `language` takes vLLM's ISO codes and the package's names (and `yue`, `fil`). Greedy only, no
  stream, `verbose_json` refused.
- No 30 s clipping by default; past 1,200 s a 400, not a split. `serve --max-audio-clip-s N` cuts as vLLM does
  (`audio/split.zig`), the pieces at once in lanes (`pieces.zig`), joined as vLLM joins them (`transcript.join`);
  each piece keeps the package's prompt and parse. The audio layer owns the supported formats.
- The tower attends in 104-token blocks where the shipped sdpa and eager paths take the whole clip (it changes 28 of
  52 long transcripts); the fixtures follow the blocks, as FlashAttention-2 varlen does.
- On the route the engine owns the prompt and the reply's text. The directory and C prefix take `model_type`, not a
  TensorFold family name (`docs/naming.md`).

## Chat (spec 150)

- `/v1/chat/completions` and `/v1/completions` as vLLM serves this model: the host renders `chat_template.json`
  (all system text the context; an audio part of any turn a placeholder in the user turn; user text dropped), reads
  `audio_url` (a data URL; HTTPS with `--vision-urls`) and `input_audio` parts (the vendor's `server/audios.zig`), and
  `Info.audio`'s `prepare` decodes a part as the route decodes a file; each placeholder becomes its clip's rows. A
  text-only chat is answered too, as vLLM answers it (`supports_transcription_only` is unset in `qwen3_asr.py`).
- The reply is the model's text as it is (`language X<asr_text>...`), not parsed; usage counts the audio rows. With an
  empty context the template's prompt is the route's, token for token (`served_test.zig`).
- Departures: greedy only (`Info.greedy_only`, the route's and the checkpoint's `generation_config`); no
  `--max-audio-clip-s` cut (vLLM cuts on the route alone); a request's audio up to 1,200 s in all (vLLM's decoder
  stops at 600 s a clip, `VLLM_MAX_AUDIO_DECODE_DURATION_S`); an audio part in a system message refused, where the
  template would count it.
- A clip under 100 frames is padded to a chunk, its convolution rows past the valid length zeroed; the package runs
  it unpadded.

## When editing here

- Do: say a departure from the package in the code, the commit and the spec's handoff.
- Do: run `check.sh qwen3_asr` and the WER set (`tools/reference/qwen3_asr/serve_wer.py`) after a change of bits.
- Do not: quantize at load; an NVFP4 checkpoint is its own (spec 140), read as stored. Its test (`forward_test.zig`,
  `ZSPARK_QWEN3_ASR_NVFP4`) holds it to the same build dequantized to bf16 and to both row rules.
- Do: run both checkpoints' tests after a change to the tower or the language model; 1.7B's figure lines are
  compared with the last full tier's to show its bits unchanged.
