# Nemotron-3.5-Transcribe-30B-A4B-NVFP4 reference (spec 151 WS-04)

The oracle is the model card's serving stack: **vLLM 0.32.0 + NeMo Speech main's `nemo_speechlm`
plugin**, on GB10 (aarch64, CUDA 13). Everything weightless — the mel of the fixed clips, the rendered
chat prompt, the audio placeholder count, the clip set itself — is written without the checkpoint; the
encoder rows, backbone hiddens, logits, transcripts and per-clip WER wait for WS-00's download.

## The environment

`Dockerfile` + `build_image.sh` build `zspark151-nemo-vllm:local` on spark-b over
`zspark142-vllm-audio:local` (spec 142's vLLM audio image: Ubuntu 24.04, python3.12, torch 2.13.0+cu130,
nvcc 13.0, flashinfer with the sm_121a JIT cache). The card's recipe (`uv sync --extra asr --extra vllm
--locked` in NVIDIA-NeMo/Speech) resolves the same vllm and torch; the Dockerfile installs the PyPI
wheels over the image's matching torch and `pip install -e`'s the pinned NeMo checkout
(`~/.cache/zspark/scratch-151/nemo`, main `774bc30`, 2026-10-30). The vLLM wheel is staged into the
build context by `build_image.sh` (the 321 MB PyPI reach from the Sparks truncated on the first
attempt). The card's `qwen3_dflash2.py` anchor-clamp patch is applied at build.

```bash
ssh spark-b 'cd ~/repo/zspark-151r && bash tools/reference/nemotron_transcribe/build_image.sh'
# smoke: imports and the serve flags the card's launch needs
ssh spark-b 'docker run --rm zspark151-nemo-vllm:local \
  python3 -c "import vllm,nemo.collections.speechlm2.vllm.salm as s; s.register(); print(vllm.__version__)"'
```

## The fixtures

`export.py weightless` (inside the image, or anywhere NeMo's preprocessor imports — it needs only
`soundfile`, `torch`, `safetensors`, `transformers`, `nemo`) writes under `~/.cache/zspark/scratch-151/
fixtures/` (`ZSPARK_REFERENCE_NEMOTRON`, mirrored on both Sparks):

- `clips/NAME.flac` — spec 138's clip set (LibriSpeech test-clean `en_5s/30s/140s/600s`, FLEURS
  `cmn_hans_cn` `zh_5s/30s/140s/600s`, same chapters and seeds) plus `en_3500s`, whole LibriSpeech
  chapters joined to 3,494 s, near the checkpoint's hour context. All CC BY 4.0.
- `NAME.mel.safetensors` — `pcm.16k` F32 `[n]` and `mel.f32` `[128, T]`, the checkpoint's
  `perception.preprocessor` (AudioToMelSpectrogramPreprocessor: 16 kHz, n_fft 512, 25/10 ms hann, log,
  dither 1e-5, no pad).
- `manifest.json` — per clip: source, utterance files, seconds, mel frames, ground truth and
  `audio_placeholders`, the plugin's own estimate (`_estimate_audio_tokens`: `(n + 2·256 − 512) / 160`
  frames then ⌈/8⌉ feature stacking, one single pass — the checkpoint mounts a ParallelExpertEncoder,
  so no 30 s chunk split).
- `prompt.chat.json` — `chat_template.jinja` rendered for the card's chat example
  (`Transcribe the following audio.` plus one `audio_url` part), `enable_thinking` false and true, the
  tokenizer's ids and the placeholder token's index.

```bash
ssh spark-b 'docker run --rm -v $HOME:$HOME zspark151-nemo-vllm:local \
  python3 ~/repo/zspark-151r/tools/reference/nemotron_transcribe/export.py weightless'
rsync -a ~/.cache/zspark/scratch-151/fixtures/ 198.51.100.1:~/.cache/zspark/scratch-151/fixtures/   # both Sparks
```

## With the weights (WS-00)

The checkpoint is the HF hub snapshot `66c6adc` under `~/.cache/huggingface/hub/` (both Sparks;
`~/models/NVIDIA-…-NVFP4` is metadata-only — do not use it). `serve.sh` and `export.py`/`export_model.py`
default to the snapshot; `ZSPARK_NEMOTRON_NVFP4` overrides.

- `serve.sh` — the card's `vllm serve` profile (bf16 activations, fp8 KV cache, ReplaySSM on
  flashinfer's Mamba backend, MTP8, `nemotron_v3`, the CTC-timestamp middleware; loopback only).
- `export.py wer [--serve FILE]` — transcripts of every clip through `127.0.0.1:8000`'s chat
  completions, WER (en, lowercased no-punctuation jiwer) / CER (zh, no spaces) against ground truth,
  into `wer/refs.jsonl` + `wer/summary.json`. `--serve` names a file holding the serve command to
  start (and stop) around the run; without it an already-running server is used.
- `export_model.py` — the model fixtures (`model/NAME.safetensors`), one in-process `vllm.LLM`
  offline load (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `enforce_eager=True` so hooks fire; same kernels,
  greedy): `enc.out` the encoder's output rows, `forward.num_rows`/`forward.input_ids`/
  `forward.positions` per backbone forward, `hidden.L*`/`residual.L*` per layer per forward
  (the block's mixer output and its completed row — `fused_add_rms_norm` returns residual as
  fp32(in+residual)→bf16), `logits.*`/`logits.top5_*`, `tokens.greedy` and the transcript.
  The exporter spools each forward to host files and streams the final safetensors assembly, so the
  hour fixture does not require a second full tensor copy in memory. It captures every prefill chunk
  and the first 32 decode forwards; generation uses the remaining room in the card's 65,536-token
  context. Metadata records `finish_reason` and `stop_reason`; `length` must not be reported as EOS.
  `manifest.json`'s `clips` are the clip set: `accuracy_test.zig` replays `model/NAME.safetensors` for
  each declared name alone, so an auxiliary safetensors left beside them (a debug or diagnostic export)
  is not a fixture.

```bash
# image (re)build — stages vllm + flashinfer wheels, no GPU needed
ssh spark-b 'cd ~/repo/zspark-151-reference && bash tools/reference/nemotron_transcribe/build_image.sh'
# model fixtures, one load (GPU, under both locks):
ssh spark-b 'flock ~/.cache/zspark/machine.lock -c "flock ~/.cache/zspark/gpu.lock -c \
  \"docker run --rm --gpus all -e ZSPARK_HOST_HOME=\\$HOME -v \\$HOME:\\$HOME \
     zspark151-nemo-vllm:local python3 \
     \\$HOME/repo/zspark-151-reference/tools/reference/nemotron_transcribe/export_model.py\""'
```

### GB10 environment notes (first-load findings, `~/.cache/zspark/scratch-151/report-ReferenceLoad.md`)

- `flashinfer-python` must match `flashinfer-jit-cache` or `import flashinfer.fused_moe` hard-fails
  inside NVFP4 MoE backend selection — image installs 0.7.1 to pair with the base's jit-cache 0.7.1
  (vLLM 0.32.0's `==0.7.0.post1` is a requirements pin, its call sites are version-agnostic).
- A snapshot copy can leave `tokenizer.json` as a dangling nested blob symlink; every file under the
  snapshot must resolve (`[ -e $(readlink -f f) ]`) before a load.
