---
checkpoint: Qwen/Qwen3-ASR-1.7B
revision: 7278e1e70fe206f11671096ffdd38061171dd6e5
model: Qwen3-ASR-1.7B
family: qwen3_asr
storage: BF16 as stored (the audio tower, the Qwen3 1.7B language model, one copy of the tied head and embedding)
license: apache-2.0
disk_gb: 4.7
machines: 1
device_gib: 7.19
status: supported
drafters: none (greedy `Decoder(M)` with an empty chain; spec 138)
variables: ZSPARK_QWEN3_ASR, ZSPARK_REFERENCE_QWEN3_ASR
---

# Qwen3-ASR-1.7B (`Qwen/Qwen3-ASR-1.7B`)

**Machines: one DGX Spark**, which it fills by 7.19 GiB of device memory after a `serve` load at the defaults
below (spark-b, 2026-10-25, spec 138 WS-05). It transcribes speech through `POST /v1/audio/transcriptions`
([`docs/serving-media.md`](../serving-media.md#audio)) and, as vLLM serves it, `/v1/chat/completions` with audio
parts ([Chat](../serving-media.md#chat), spec 150). Its area (`check.sh qwen3_asr`, 7 tests)
and `check.sh audio` pass; no full tier is recorded for it.

## The checkpoint

| | |
| --- | --- |
| Repository, revision | `Qwen/Qwen3-ASR-1.7B` at `7278e1e70fe206f11671096ffdd38061171dd6e5` (the repository's head, last modified 2026-01-30, read 2026-10-26; the directory was fetched before that read, 2026-08-24) |
| Licence | Apache-2.0 |
| On disk | 4.70 GB: two `model-0000N-of-00002.safetensors`, `config.json`, `preprocessor_config.json`, `chat_template.json`, and `vocab.json`, `merges.txt` and `tokenizer_config.json` (no `tokenizer.json`) |
| TensorFold family | none (TensorFold has no ASR model); in zspark `zig/src/models/qwen3_asr/`, named by the checkpoint's `model_type` (`docs/naming.md`) |
| Variables | `ZSPARK_QWEN3_ASR`, `ZSPARK_REFERENCE_QWEN3_ASR` (`docs/environment.md`) |
| Behaviour reference | the `qwen-asr` 0.0.6 package's transformers backend; vLLM's `qwen3_asr.py` for the route (spec 138 Decisions) |

The model, from `config.json` (thinker only: `thinker.audio_tower`, `thinker.model`, `thinker.lm_head`, about
2.35 B parameters):

- The audio tower (317.5 M): the Whisper mel (128 bins, 16 kHz) in chunks of 100 frames, three stride-2 convs, 24
  pre-LN layers (width 1024, 16 heads of 64, GELU FFN 4096) that attend in blocks of 104 tokens (800 frames), then
  `ln_post` and a two-layer projector to 2048. A 100-frame chunk gives 13 tokens.
- The language model: Qwen3-1.7B, 28 layers, hidden 2048, 16 query and 8 kv heads of 128, FFN 6144, q/k RMSNorm,
  theta 1e6. MRoPE's three axes take one position for audio, so it is a plain split-half RoPE.

## What is stored, and how it is multiplied

Everything is BF16 and runs as stored; nothing is quantized (the owner's NVFP4 build is [its own page](qwen3-asr-1.7b-nvfp4.md), spec 140). The tower's
linears and convs are `zs_b16_gemm_bias` (the convs through im2col), the language model's are `bf16.Linear`.
`embed_tokens` and `lm_head` are both stored and are bitwise identical.

## What changes while it loads

- One copy of the tied pair: `thinker.lm_head` is the head and the embedding table, `embed_tokens` is not read.
- A layer's q, k and v are stacked into one linear, gate and up into another.
- The tower's conv weights are repacked to (tap, channel) columns, `conv_out`'s columns are reordered for
  channels-last convs, and q, k, v are fused into one `[3072, 1024]` linear.
- `config.json` is checked (`thinker_config.text_config`, `audio_config`), and the tower's 397 tensors by name,
  dtype and shape.
- The tokenizer is built from `vocab.json`, `merges.txt` and `tokenizer_config.json`, as transformers'
  `Qwen2Converter` makes its `tokenizer.json`.

## Defaults that follow this checkpoint

| Setting | Default | Why |
| --- | --- | --- |
| `--name` | `Qwen3-ASR-1.7B` (the directory's name) | what aimd sends |
| `--rows` | 2048 | a prompt window's rows (zspark's own flag) |
| `--context` | 20,208 tokens | a 1,200 s clip's 15,600 audio rows, 512 for the prompt's words and 4,096 reply tokens |
| reply tokens | 4,096 | the package's `max_new_tokens` on vLLM; `max_completion_tokens` lowers it |
| clip length | 1,200 s | the package's `MAX_ASR_INPUT_SECONDS`; longer is a 400 (no splitting at the lowest-energy point) |
| `--max-audio-clip-s` | off: a clip runs whole | the package's way; `30` (the checkpoint's `chunk_length`) is vLLM's: pieces cut at the quietest 100 ms, run at once in `--parallel` lanes (16) of 2,048 positions; [below](#long-clips-in-pieces) |
| upload | 25 MiB | vLLM's `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` |
| formats | WAV, FLAC, MP3, Ogg Vorbis, M4A/MP4 (AAC-LC and ALAC), ADTS AAC-LC, Ogg/WebM Opus, PCM AIFF | the shared host audio decoders, with the format detected from the file bytes |
| decoding | greedy, one transcription at a time | a temperature above 0 is a 400; `--max-queued` (64) others wait |
| chat | audio parts, 999 a request, 1,200 s in all; the reply's text unparsed; no `--max-audio-clip-s` cut | vLLM's chat for this model (`--limit-mm-per-prompt`'s default; it cuts on the route alone); greedy as the route; a chat waits in the same queue |

Decoding steps run as CUDA graphs keyed by rows and a context bucket (4,096 to the capacity); a prompt window of
64 rows or more uses the grouped attention kernel. Rows keep the row rule: a window, a chunk and
a serial step give the same bits (`forward_test.zig`).

## Measured

All on spark-b, fast builds, the machine held alone, `zspark generate` on the WS-01 fixtures or `serve` over HTTP
(spec 138 WS-05 and WS-06; raw lines `~/.cache/zspark/scratch-138/`; WS-06's head 23b6124c).

| Figure | Value |
| --- | --- |
| Load | 1.0 s warm (`serve`), 7.19 GiB on the device |
| In `zspark stack` (spec 139 WS-03, spark-a; since 2026-10-28 the stack runs [the NVFP4 build](qwen3-asr-1.7b-nvfp4.md) under this id, spec 140 WS-05) | 7,534 MiB (`nvidia-smi`), 7.5 to 7.8 GiB of `MemAvailable` at `--parallel 1`; Flash Next beside it takes `--parallel 3`, 2 with OCR too (its recipe's "Beside OCR and ASR") |
| Beside Flash Next in `zspark stack` (spec 139 WS-04, WS-06; RTF on 600 s segments) | alone 0.058 (0.061 under the default `--mps 25`); with a Flash Next stream beside it 0.117 time-sliced, 0.221 under `--mps 25` (3.8x; 0.141 on two Sparks, 2.4x), 0.123 at `--mps 50`; the stream keeps 41 of 53 tok/s instead of 21 ([the table](qwen3.8-flash-next-fp4-local.md#beside-ocr-and-asr-zspark-stack)). On two Sparks ASR runs on the rank 1 Spark; its text equals the plain `serve`'s |
| Decode, en_140s, 525 tokens | 58.8 to 59.1 tok/s (55.4 to 55.8 before WS-06's graphs and fused norm); the weights at the bus would be about 65 |
| Prompt, 1,869 rows (en_140s) | 188 to 190 ms |
| Prompt, 600 s clip (7,862 rows, 512-row windows) | 952 to 979 ms (2,756 to 2,768 before) |
| Real-time factor over HTTP | 0.051 (0.0566 before WS-06); 0.053 on WS-05's 4,202 s of audio in 222 s |
| A 600 s clip over HTTP | en_u00 (613.9 s, 1,656 tokens) 35.0 s; zh_u00 (608.4 s, 1,303 tokens) 27.4 s |
| The audio tower, 600 s | 193.6 ms (under 1% of a 600 s request, so left unoptimized) |

Accuracy, the 64-clip set (32 English from LibriSpeech test-clean, 32 Chinese from FLEURS cmn_hans_cn, 5.5 to
613.9 s; free, forced-language and context cases), against the ground truth, normalized alike (WS-05):

| | zspark | `qwen-asr` 0.0.6 (HF bf16) |
| --- | --- | --- |
| English WER | 1.618% | 1.675% |
| Chinese CER | 9.728% | 9.668% |

56 of the 64 texts equal the package's; the others leave at a near tie of the language model (a gap of 0 or 1
bf16 step), first at characters 8 to 3,675. The tower follows the reference at `proj2` cosine 0.9995 to 0.9999
over a clip, with scattered rows lower on long clips (WS-03); the 5 s and 30 s transcripts equal the package's
token for token. Not measured: the owner's own recordings.

### Speed against vLLM

Spec 142 WS-05 (spark-b, machine alone, loads zspark, vLLM, zspark, vLLM, two passes of the 64 clips each, the second
kept). No 0.27 image exists here: vLLM is `eugr/spark-vllm-b12x:latest`, the fork `local-inference-lab/vllm`
`dev/karmic-kraken` @8244d1c44 (`v0.29.1rc0-603`), the card's `vllm serve`, with `vllm[audio]`'s packages added (av,
scipy, soundfile, soxr; every request was a 400 without) and `VLLM_MAX_AUDIO_DECODE_DURATION_S=1200`.

| | zspark | vLLM (fork) |
| --- | --- | --- |
| RTF, all 64 / 30 s or less (28) / longer (36) | 0.0486 / 0.0519 / 0.0484 | 0.0198 / 0.0692 / 0.0167 |
| First token, median / 30 s or less / the 600 s clip | 70 / 33 / 1,117 ms | 43 / 36 / 540 ms |
| Tokens a second, one stream (30 s or less) | 62.2 / 61.6 | 46.1 / 46.4 |
| English WER / Chinese CER | 1.618% / 9.728% | 1.887% / 9.4% |
| Texts equal to the `qwen-asr` package's | 56 of 64 | 26 of 64 |

On one stream zspark decodes 34% faster and first tokens are within the spread; vLLM is 2.45x faster on the set
because it cuts a clip over 30 s into 30 s pieces and runs them at once (zspark's default runs the whole clip as the
package does; the option below does the cut). vLLM's texts differ between loads, zspark's are the same on all four
passes. vLLM loads in about 180 s, zspark in 2 s.

### Long clips in pieces

`serve --max-audio-clip-s 30` (spec 142 WS-06, spark-a held alone, WS-05's 64 clips, the second of two passes;
raw lines `~/.cache/zspark/scratch-142/ws06/`) against the default and vLLM 0.27 (its own 30 s pieces, WS-05):

| | zspark whole (default) | zspark in pieces | vLLM |
| --- | --- | --- | --- |
| RTF, all / to 30 s / longer | 0.0486 / 0.0519 / 0.0484 | 0.0219 / 0.0527 / 0.0199 | 0.0198 / 0.0686 / 0.0166 |
| first token, median / 600 s clip | 70 ms / 1,117 ms | 91 ms / 1,226 ms | 42 ms / 512 ms |
| A 600 s clip | en_u00 35.0 s | en_u00 5.7 s, zh_u00 4.4 s | 2.8 s, 2.3 s |
| English WER / Chinese CER | 1.618% / 9.728% | 1.945% / 10.029% | 1.887% / 9.442% |
| texts equal to vLLM's (its second load) | 25 of 64, 1 of 36 past 30 s | 37 of 64, 13 of 36 past 30 s | its two loads: 56 of 64, 28 of 36 |

Up to 16 pieces run at once (`--parallel`), each in a lane of 2,048 positions: 10.70 GiB on the device instead of
7.19, the load 1.4 s. A clip to 30 s is untouched (its texts equal the default's). A piece's text is the one the
default gives that piece sent alone (en_u00's first and zh_m04's 0.36 s last, checked), so where the pieces leave
vLLM's texts (punctuation, numerals as digits or words, a dropped accent, an empty last piece where vLLM hears
"嗯。") the model takes another branch under zspark's prompt (the package's, vLLM drops the empty system turn) and
numerics, as on short clips (24 of 28 equal). The cut is vLLM's to the sample (`zig/tests/audio/split_test.zig`).
The rest of the gap on a 600 s clip is the 21 prompts run one after another: vLLM batches them (first token 512
ms against 1,226).

## Tests

`ZSPARK_QWEN3_ASR=DIR ZSPARK_REFERENCE_QWEN3_ASR=~/.cache/zspark/reference-qwen3-asr tools/check.sh qwen3_asr`:
the tower against the package's per-stage exports, the language model's layers, logits and greedy tokens, the
row-rule tests (a block of the tower, windows against steps, graphed steps), and the prompt and transcript
parsing against `text_cases.json`. `check.sh audio` holds decoding, resampling and the mel to the fixtures;
`tests/server/transcriptions_test.zig` holds the route and chat's audio parts (host tier); `served_test.zig` (in the
area) the engine behind the host, a chat's reply against the route's transcript of one clip. The fixtures are `tools/reference/qwen3_asr/`
(`env.sh` builds the `qwen-asr` environment; `export.py`; `serve_wer.py` runs the WER set through a server).
