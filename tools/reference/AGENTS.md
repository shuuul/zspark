# `tools/reference/` instructions

This file extends `tools/AGENTS.md` and the root `AGENTS.md`. Python test fixtures that export reference tensors
and expected values from TensorFold. They run inside the TensorFold container (PyTorch and Triton), never at
zspark's runtime, and the build does not know them; the parent's "builds for the host with no libc" rule is for
the Zig tools. One fixture has another oracle and another language: `grammar_masks` (xgrammar, Zig and a C++
shim).

**Python is allowed in the repository only here**, as fixtures that export references from TensorFold (and
the checks and measurements below, run by hand); never at runtime.

`README.md` here is for whoever reads a reference file or runs a script: how the vision files read, what a script's docstring does not say, and the other export commands.

## Purpose

Each script's docstring or header is its manual; `README.md`, "The map in detail", says more of each. The map:

- `flash_next_export.py`: TensorFold's decode-path forward by block, as safetensors (no checkpoint supported since spec 121).
- `flash_next_tap_export.py`: DFlash2 completed-layer taps (spec 122). `--synthetic` writes made 4-stream means, no checkpoint; `--from-sglang` prints the SGLang command and stops; `--from-vllm` with `flash_next_taps_vllm.sh` and `.patch` is the real capture in the card's vLLM (spec 125).
- `qwen3_asr/`: Qwen3-ASR-1.7B's fixtures and WER set from the `qwen-asr` package (spec 138; `env.sh`, `export.py`, `README.md`).
- `nemotron_transcribe/`: Nemotron-3.5-Transcribe-30B-A4B-NVFP4's reference environment and fixtures (spec
  151 WS-04): `Dockerfile`/`build_image.sh` build `zspark151-nemo-vllm:local` on spark-b (vLLM 0.32.0 +
  NeMo Speech main's `nemo_speechlm` plugin over `zspark142-vllm-audio:local`); `export.py` writes the
  weightless fixtures (NeMo mel, the card's chat prompt and ids, the plugin's audio-placeholder count,
  the spec-138 clip set plus an hour-long one) under `~/.cache/zspark/scratch-151/fixtures/` on both
  Sparks, and after WS-00 the per-clip WER; `export_model.py` the weight fixtures (encoder rows, per-layer
  hiddens, logits, transcripts) by hooking vLLM's `LLM` in-process; `serve.sh` is the card's launch,
  loopback only.
- `vision_export.py`, `vision_cases.py`: Flash Next's image path and TensorFold's image processor (spec 019).
- `glm5_export.py`, `glm5_drafter_export.py`: GLM's first layers and DFlash2 drafter from TensorFold 0.6.6.
- `hf_env.sh`, `unlimited_ocr/`: the HF-code reference environment and Unlimited-OCR-2's fixtures. See "The HF environment".
- `vllm_reference.sh`, `vllm_reference.py`, `vllm_reference_ranks.sh`: the card's vLLM container's greedy replies and logprobs (specs 105, 107).
- `template_cases.py`, `message_cases.py`, `tokenizer_cases.py`, `image_cases.py`: `zig/src/text/`'s and `image/`'s fixtures; no GPU.
- `sampling_cases.py`, `requantize_cases.py`: `zig/tests/fixtures/*.bin`.
- `ngram_cases.py`: the no-repeat n-gram ban sets of Unlimited-OCR's HF processor, ported; `zig/tests/fixtures/ngram_cases.bin` (spec 137); no GPU, no dependencies.
- `audio_cases.py`: `zig/src/audio/`'s fixture, the `qwen-asr` package's decode, `soxr_hq` and Whisper features on files made there (spec 138); no GPU, `uv run`.
- `audio_split_cases.py`: `audio/split.zig`'s fixture, vLLM's `split_audio` copied and run on numpy over a generated
  signal and the WER set's clips (spec 142 WS-06); no GPU, `uv run`.
- `csf_blocks_cases.py`: csf_32x2_v1 planes through flashinfer's loader (spec 107); no GPU.
- `grammar_masks.zig`, `grammar_masks.cc`, `grammar_masks.sh`: xgrammar's masks, the oracle of `json_schema.zig` (spec 017).
- `openai_exchange.py`, `responses_exchange.py`, `anthropic_exchange.py`: the clients' packages against a running `serve`, by hand.
- Measurements, not fixtures:
  - `client_bench.py`, `client_bench.json`, `client_bench.sh`, `required_call_bench.json`: what a client sees of a server.
  - `glm5_client_bench.sh`, `glm5_serve_check.py`: the same on GLM over two ranks (spec 068).
  - `shared_acceptance.sh`, `pool_states.*`, `spill_states.*`, `fill_gaps.*`, `readme_table.*`: the pool, the states and the first table.
  - `serve_lib.sh`: what those scripts share, sourced (spec 053).
  - `draft_policy.py`, `pool_policy.py`, `lands_cases.py`, `lands_fit.py`: draft rules played over recorded chains; host only.
  - `device_memory.py`, `turns_part.py`, `long_conversation.py`: device memory by kind; where a request parts from what `serve` holds; a conversation past 100,000 tokens.

## Local commands

The outputs live outside the repository, under `~/.cache/zspark/`. Kernels compile on the first run into
`~/.cache/zspark/tensorfold-kernels/` (about 2.5 minutes), which later runs reuse.

- TensorFold's loader drops the checkpoint's pages from the page cache as it reads, so the next zspark load after
  an export reads from disk again.
- The other scripts' commands are in their docstrings and headers.

## The HF environment

For a checkpoint whose behaviour reference is its own HF code (no TensorFold family: Unlimited-OCR-2, Qwen3-ASR).
`source tools/reference/hf_env.sh` makes `~/.cache/zspark/venv-reference` once and activates it: uv, Python 3.12,
torch and torchvision from `https://download.pytorch.org/whl/cu130` (aarch64 + CUDA 13; 2.14.1 on 2026-10-25, the
card says 2.10), transformers 4.57.6 (5.x removes APIs the OCR code imports), Pillow 12.3.0 (13.x changes the resize fixture's bytes), einops, addict, easydict, pymupdf (`import pymupdf`; `fitz` warns),
safetensors; `HF_ENV_EXTRA="qwen-asr"` adds more. The first call downloads about 6 GB (15 minutes) and takes no
lock; run a script under `flock -s machine.lock flock gpu.lock`. `ZSPARK_REFERENCE_VENV` moves it. A reference of
this kind runs the checkpoint's own entry point (`infer`), hooks it from outside and wraps `generate` only.

## Local conventions

- TensorFold is not edited. The exporter wraps module-level functions of `families/qwen4_exp/cuda/forward.py`
  from outside and clones the buffers they leave; a capture is a clone because the buffers are reused.
- The exporter is pinned to TensorFold 0.6.5 (commit 609ca41). When TensorFold moves, reread `forward.py`,
  `state.py` and `decode.py` against the list in the exporter's docstring before trusting a file.
- Tensors keep the dtype TensorFold holds them in (BF16, F32, I32), rows on the first axis, plain dotted names.
- Every file carries its provenance in the safetensors metadata: TensorFold's version and commit, the
  checkpoint's revision, torch and triton versions, the mode and the ids.
- A bench starts its server on the loopback, 127.0.0.1:18888, holding the machine alone. Only
  `client_bench.sh ... sparkdash` listens on a LAN address, spark-b's (192.0.2.227:8888, which sparkDash
  knows), for the run. A bench never listens on 192.0.2.114:8888: that is the LAN's model endpoint on
  spark-a, where other clients' requests would reach the bench, and `serve_lib.sh` refuses to start there.
- `client_bench.json` is fixed: changing it means measuring both servers again, and a new request goes last, so
  that the ones before it stay the bytes they were.
- An expectation is what the oracle returned, not written in Zig. A new schema class is an entry in
  `grammar_masks.zig`'s `corpus`.
- xgrammar is never vendored, linked into zspark or needed by a test: only the two text files are. What is
  allowed stays xgrammar's word; regenerate after the matcher changes what it reads, and read the diff.
- `device_memory.py` reads a buffer's kind from its variable's name in the source (`SCRATCH`): a renamed
  buffer needs its name added there.
- Run `turns_part.py --set all` after a change to the template's rendering, the gate's form of a call or what
  `session.zig` keeps.

## When editing here

- Do: record a new capture point in the exporter's docstring and say which buffer it clones.
- Do: keep serial steps and windows in one process, so the comparison between them is of one load.
- Don't: import anything from here in `zig/src/`, `zig/kernels/` or `build.zig`, or add Python outside this directory.
