# Qwen3-ASR-1.7B reference (spec 138)

Fixtures and the WER/CER set for `zig/src/audio/`, the audio tower and the language model of Qwen3-ASR-1.7B. The
oracle is the `qwen-asr` 0.0.6 package's transformers backend (transformers 4.57.6, torch 2.14.1+cu130, bf16,
GPU, batch 1, greedy), because TensorFold has no ASR model. Python only here, never at runtime.

```bash
PY=$(tools/reference/qwen3_asr/env.sh)          # builds ~/.cache/zspark/qwen3-asr-env/venv (uv, no container) once
RUN="flock -s ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock $PY -I tools/reference/qwen3_asr/export.py"
$RUN fixtures   # 8 clips, 12 scenarios, 2 resampler files: about 6 minutes
$RUN wer --also-full   # the WER/CER set: about 40 minutes with --also-full
$RUN attn       # windowed vs the shipped full audio attention
$RUN tied       # embed_tokens vs lm_head
# 0.6B (spec 140 WS-09): the same commands with its checkpoint and its own output directory
ZSPARK_QWEN3_ASR_MODEL=~/models/Qwen3-ASR-0.6B ZSPARK_REFERENCE_QWEN3_ASR=~/.cache/zspark/reference-qwen3-asr-0.6b $RUN fixtures
```

Output: `~/.cache/zspark/reference-qwen3-asr/` (`ZSPARK_REFERENCE_QWEN3_ASR`, `docs/environment.md`). Sources are
`--src DIR` (default `~/.cache/zspark/scratch-138/dl`): LibriSpeech test-clean (openslr.org/resources/12, CC BY
4.0) and FLEURS `cmn_hans_cn` test (huggingface.co/datasets/google/fleurs, CC BY 4.0).

## The facts the port needs (read from the package and from transformers 4.57.6)

**Audio in.** `qwen_asr/inference/utils.py`: a path goes through `librosa.load(sr=None, mono=False)`; mono is the
mean of the channels; the resample to 16 kHz is `librosa.resample(audio, orig_sr, target_sr=16000)` whose default
`res_type` is `soxr_hq` (the `soxr` package, 1.1.0); then `float_range_normalize`: if the peak is above 1 divide
by it, then clip to [-1, 1] (a no-op for decoded int16). Audio above 1200 s is split near the 1200 s marks (not
exported), and only those split pieces are zero-padded to 0.5 s: a clip under 1200 s goes in whole and unpadded,
however short (`split_audio_into_chunks` returns it first). `export.py short` exports two such clips (0.6 s and
0.35 s, one chunk of under 100 frames).

**Mel** (`WhisperFeatureExtractor`, `preprocessor_config.json`, torch path with device `cpu`: torch is installed,
so `_torch_extract_fbank_features` runs, not the numpy one):

| Step | Value |
| --- | --- |
| Filters | `mel_filter_bank(num_frequency_bins=201, num_mel_filters=128, min_frequency=0.0, max_frequency=8000.0, sampling_rate=16000, norm="slaney", mel_scale="slaney")`: Slaney scale and Slaney area norm, float64, cast to float32 |
| STFT | `torch.stft(wave, 400, 160, window=torch.hann_window(400))`: periodic Hann, `center=True` (reflect pad 200), complex |
| Power | the last frame dropped (`stft[..., :-1]`), `abs()**2`: frames = samples // 160 |
| Mel | `mel_filters.T @ power` in float32 |
| Log | `clamp(min=1e-10).log10()`, then `max(x, x.max() - 8.0)` with the max over the whole clip, then `(x + 4) / 4` |
| No pad | `padding=True`, `truncation=False`: no 30 s pad; `feature_attention_mask` has `samples // 160` ones |
| Dither | 0 |

The extractor's output is float32 `[128, T]`; the package then runs `inputs.to(model.dtype)`, so **the tower is
fed the mel rounded to bf16** (`mel.bf16`; the rounding error against `mel.f32` is up to 4e-3). A float64
recomputation of the recipe differs from `mel.f32` by 1.5e-5 at most (`scratch-138/melcheck.txt`), so the 1e-4 bar
of the spec is above the torch path's own noise; the numpy path agrees with it to about 1e-5.

**Audio tower** (`modeling_qwen3_asr.py` 609-737), as it runs:
- The mel is cut into chunks of 100 frames; the tail chunk is **zero-padded to 100 frames in the mel domain
  (value 0, not the floor) and goes through the three convs padded**, and only the first
  `((((n-1)//2+1)-1)//2+1-1)//2+1` positions of its output are kept. Its last tokens therefore see zeros where an
  unpadded conv would see its zero border after layer 1 and 2: a port must pad and mask the same way.
  A clip of one chunk under 100 frames is the exception: the chunks are padded to the longest, itself, so it goes
  through the convs unpadded and its last tokens see the convs' zero border (`tower.zig` zeroes the rows past it).
- Positions: the sinusoid table `[0..12]` is added per chunk after `conv_out`.
- Attention blocks: `window = 13 * (800 // 100) = 104` tokens, the last block the remainder (`enc.cu_seqlens`).
  **The shipped backend only applies them under FlashAttention-2.** Under sdpa or eager (the default without
  flash-attn) the tower is given no mask and every token attends to the whole clip, which the package never
  warns about. The reference here is the intended block attention (what FA2 varlen computes, and what vLLM and
  SGLang run): `export.py` registers an attention function that runs sdpa once per block. `attn_windowed_vs_full.json`
  holds the cosine of the two outputs and whether the transcripts differ; they agree for clips of at most
  104 tokens (about 8 s). `--audio-attn full` exports the shipped behaviour.
- `enc.proj1` is the linear's output before its GELU; `enc.proj2` is the audio embedding that replaces the pads.

**Prompt** (`apply_chat_template` of the package's template): the system turn is always present, empty or the
context; `language X<asr_text>` is appended to the generation prompt for a forced language. `<|audio_pad|>`
(151676) is repeated `N` times, `N = enc.cu_seqlens[-1]`; ids are in `prompt.ids`. Positions are `0..seq-1`
(`get_rope_index` gives the three MRoPE axes the same value), so the LM is plain RoPE, theta 1e6.

**Decoding**: greedy, `max_new_tokens` 4096 (the package's vLLM default; its transformers default of 512 cuts a
600 s transcript), EOS 151645 or 151643. `raw_text` is the decode with `skip_special_tokens=True`, which keeps
`<asr_text>`; `parsed_*` is the package's `parse_asr_output`. The reference environment sets
`TORCH_DISABLE_NATIVE_JIT=1`: torch 2.14 otherwise routes some ops (the RoPE outer product, norms, top-k) to
Triton kernels that cannot build here (no Python headers) and that are not what the other references ran.

**`embed_tokens` and `lm_head`**: `thinker.model.embed_tokens.weight` and `thinker.lm_head.weight` are both stored
(bf16, `[151936, 2048]`, shard 1) and are **bitwise identical** (sha256 d7d2c2a8...17f9; `export.py tied`). One copy
is enough in memory.

## Files

| File | Contents |
| --- | --- |
| `clips/NAME.flac` | the 16 kHz mono int16 clip (the resampler files are 48 kHz stereo and 44.1 kHz mono FLAC) |
| `manifest.json` | per clip: source, utterance files, seconds, ground truth; per scenario: tokens, new tokens; provenance |
| `NAME.SCENARIO.safetensors` | tensors below, metadata: versions, GPU, model index sha256, attention mode, resampler |
| `NAME.SCENARIO.json` | prompt text, raw and parsed text, shapes, timings, the two generate runs' logit difference |
| `rs_*.audio.safetensors` | `pcm.src` (`[T]` or `[T, 2]`), `pcm.16k`, `mel.f32` for the resampler/downmix tests |
| `wer/` | `refs.jsonl`, `clips/*.flac`, `provenance.json` |

Tensors of `NAME.SCENARIO.safetensors` (`T` mel frames, `N` audio tokens, `S` prompt length, `V` 151936, `K` new tokens):

| Name | Type, shape | Is |
| --- | --- | --- |
| `pcm.src`, `pcm.16k` | F32 `[n]` | decoded at the source rate; after the package's mono + `soxr_hq` path |
| `mel.f32`, `mel.bf16` | F32, BF16 `[128, T]` | the extractor's output; what the tower receives |
| `enc.chunk_lengths`, `enc.tokens_per_chunk` | I32 `[ceil(T/100)]` | mel frames and tokens of each 100-frame chunk |
| `enc.cu_seqlens` | I32 `[blocks+1]` | attention block bounds in tokens (asserted equal to the ones the tower built) |
| `enc.conv1_gelu`..`conv3_gelu`, `enc.conv_out` | BF16 `[chunks, 480, 64/32/16, 50/25/13]`, `[chunks, 13, 1024]` | after each conv + GELU, and `conv_out` before the positions; the `*_5s` clips only |
| `enc.after_conv_pos` | BF16 `[N, 1024]` | the first layer's input: padded rows dropped, positions added |
| `enc.layer00,05,11,17,23` | BF16 `[N, 1024]` | outputs of those layers |
| `enc.ln_post`, `enc.proj1`, `enc.proj2` | BF16 `[N, 1024/1024/2048]` | `proj2` is the audio embedding |
| `prompt.ids`, `prompt.audio_pad_first` | I32 `[S]`, `[1]` | the ids with `N` pads, and the first pad's index |
| `lm.prefill.hidden.00`..`28` | BF16 `[S, 2048]` | HF `hidden_states` of the prefill: 0 the embeddings with the audio merged, 1..27 layer outputs, 28 after the final norm; for `S > 1024` layer 00 is whole and the others hold the last 32 rows (`lm.prefill.rows_from` = `S-32`, else 0) |
| `lm.decode.hidden` | BF16 `[steps, 29, 2048]` | the same 29 states of each of the first 32 decode forwards (fewer if EOS comes first) |
| `logits.first33` | F32 `[min(K,33), V]` | step 0 (prefill's last row) and the first 32 decode steps |
| `logits.top5_ids`, `logits.top5_vals` | I32, F32 `[K, 5]` | top-5 of every step |
| `tokens.greedy` | I32 `[K]` | the generated ids, EOS included |

Scenarios: `free` (no language, no context), `forced` (`English` / `Chinese`), `context` (a system text). Clips:
`en_5s`, `en_30s`, `en_140s`, `en_600s` (LibriSpeech test-clean, whole utterances joined in reading order; the
600 s one joins two speakers) and `zh_5s`, `zh_30s`, `zh_140s`, `zh_600s` (FLEURS sentences, different speakers
joined, in a seeded order, repeated sentences dropped); durations are whole utterances, so a "30 s" clip is up to
a few seconds longer. `rs_*` are derived from `en_5s` and `zh_5s` (soxr_hq upsampled, stereo as `[x, 0.5x]`),
because the sources are 16 kHz: they exercise decode, downmix and the resampler, not recorded 48 kHz speech.

## The WER/CER set

`wer/refs.jsonl`, one JSON per clip: `id`, `lang` (`en`/`zh`), `mode` (`free`, `forced`, `context`),
`forced_language`, `context`, `seconds`, `source`, `files`, `ground_truth` (LibriSpeech upper case; FLEURS raw
text with punctuation), `hf_raw`, `hf_language`, `hf_text` (the package's parse), `n_audio_tokens`, `n_new`,
`eos_hit`, and for clips above 104 tokens `hf_full_attn_raw` (the shipped unmasked attention). The reference
transcripts are `hf_text` / `hf_raw` of the windowed run. Score English with WER on lower-cased text without
punctuation, Chinese with CER on text without punctuation or spaces, both against `hf_*` and against the ground
truth.
