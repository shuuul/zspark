<p align="center">
  <img src="assets/logo.svg" alt="zspark: Z in a DGX Spark box with a spark" width="160" />
</p>

# zspark

[![Zig](https://img.shields.io/badge/Zig-0.17.1-F7A41D?logo=zig&logoColor=white)](https://ziglang.org/download/)
[![CUDA](https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware](https://img.shields.io/badge/Hardware-DGX%20Spark%20GB10-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

> *"On my desk, there are two machines: one is a DGX Spark, and the other is also a DGX Spark."*

My personal recipes for local LLM, OCR, and ASR inference on two NVIDIA DGX Sparks (GB10, sm_121a), backed by a decoding engine written in Zig and CUDA C++. The checkpoint choices, serving configurations, and measurements reflect what I use on my own machines.

zspark derives from [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0). It retains TensorFold's optimized CUDA kernels while replacing the Python, PyTorch, and Triton runtime layers with native Zig and CUDA C++; `zspark serve` provides TensorFold 1.3.4's HTTP host (bundled under `zig/vendor/tensorfold/`) over zspark's native engine. There is no Apple or MLX support.

Checkpoints are decoded in the format they store. zspark applies no extra quantization of its own at load or inference: weights left unquantized by publishers (such as FP4-Local's BF16 linears and F32 norms) are computed directly as stored, and tensors are held as TensorFold holds them (which only requantizes weights that draft, such as MTP layer experts). Speed comes directly from the pipeline: fused kernels, low-overhead launches, prefix caching, and speculative drafting.

## Key Points

- **One stream's speed first.** Optimization prioritizes minimizing time-to-first-token (including cold prompts) and maximizing single-stream decode throughput; multi-stream concurrency builds on top of this foundation.
- **The checkpoint's precision is kept.** Models execute in their native checkpoint precision. Speculatively drafted tokens are strictly verified against serial decode token for token: drafts are verified, never trusted.
- **Optimizations that lose nothing.** Acceleration comes from kernels, launch graphs, prefix caches, and drafting. No quality-degrading approximations are taken by default.
- **Two Sparks on one desk.** GLM-5.3-Flash runs across two Sparks as two ranks joined over NCCL. Flash Next fits on a single Spark, or can be split across two ranks to halve per-device memory footprint and accelerate decode with distributed MTP drafting. Alternatively, two whole copies can run behind a built-in router for concurrent multi-stream serving.
- **LLM, OCR, and ASR in one stack.** I have explicit needs for document OCR and speech transcription alongside LLM inference. `zspark stack` starts the selected engines and a shared HTTP router, so clients can use one address for chat, page extraction, and audio transcription. The [personal recipes](docs/recipes/README.md) record how these workloads fit together on my two Sparks.

## Build

Requires Zig 0.17.1, Rust and Cargo 1.85 or later, and the CUDA 13 toolkit (`nvcc`) on an NVIDIA DGX Spark. The first build fetches CUTLASS 4.8.1 and the locked ASR decoder crates. TensorFold 1.3.4's host is bundled in-tree (`zig/vendor/tensorfold/`).

```bash
zig build
./zig-out/bin/zspark probe
```

- `-Dcuda=PATH`: Sets the toolkit root (default `/usr/local/cuda`).
- `-Doptimize=debug|safe|fast|small`: Sets the build mode (default `fast` for production serving; `safe` adds runtime safety checks).
- `-Dnccl=PATH`: Enables NCCL for multi-machine TP=2 setups (e.g. `-Dnccl=/usr`).

The [deployment guide](docs/deployment.md) covers a clean build, two-Spark setup, and the optional xgrammar library for structured output.

Versions use `YYYY.M.D` (for example, `2026.11.12` or `2027.1.5`), maintained in `build.zig.zon`.
Month and day have no leading zero, as Zig's SemVer parser requires. Both
`zspark version` and `zspark-router version` report that version with the build mode, full Git commit
and optional `dirty` marker; an unauthenticated backend `/health` reports the same identity.
The date is the release date, never the build machine's clock. Release tags use `vYYYY.M.D`
and name a clean, verified commit after checkpoint retirements are complete.

## Checkpoints

Each supported checkpoint has a dedicated recipe in [`docs/recipes/`](docs/recipes/README.md) detailing its revision, tensor precision, memory footprint, load times, and measured decode throughput against reference baselines (vLLM / TensorFold), along with [how figures are measured](docs/recipes/README.md#how-the-figures-are-taken). [docs/weights.md](docs/weights.md) records the Hugging Face source of each supported checkpoint, the revision pins, and how to download one into a local directory — zspark never downloads weights itself.

| Checkpoint | Storage | Machines | Recipe |
| --- | --- | --- | --- |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | NVFP4 experts (csf scales), MXFP8 side layers | 1, or 2 as two ranks | [qwen3.8-flash-next-fp4-local.md](docs/recipes/qwen3.8-flash-next-fp4-local.md) |
| `nvidia/GLM-5.3-Flash-FP4-Local` | NVFP4 experts (csf_32x2_v1 scales), MXFP8 and FP8_PB_WO side layers | **2, no fewer** | [glm-5.3-flash-fp4-local.md](docs/recipes/glm-5.3-flash-fp4-local.md) |
| `Qwen/Qwen3-ASR-1.7B` | BF16 as stored (speech to text) | 1 | [qwen3-asr-1.7b.md](docs/recipes/qwen3-asr-1.7b.md) |
| `Qwen/Qwen3-ASR-0.6B` | BF16 as stored (speech to text) | 1 | [qwen3-asr-0.6b.md](docs/recipes/qwen3-asr-0.6b.md) |
| `shuuul/Qwen3-ASR-1.7B-NVFP4-W4A16` | NVFP4 W4A16 language model, the rest BF16 (speech to text) | 1 | [qwen3-asr-1.7b-nvfp4.md](docs/recipes/qwen3-asr-1.7b-nvfp4.md) |
| `shuuul/Qwen3-ASR-0.6B-NVFP4-W4A16` | NVFP4 W4A16 language model, the rest BF16 (speech to text) | 1 | [qwen3-asr-0.6b-nvfp4.md](docs/recipes/qwen3-asr-0.6b-nvfp4.md) |
| `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` | NVFP4 attention/MoE, FP8 Mamba/perception (speech to text) | 2 | [nemotron-3.5-transcribe-30b-a4b-nvfp4.md](docs/recipes/nemotron-3.5-transcribe-30b-a4b-nvfp4.md) |
| `baidu/Unlimited-OCR-2` | BF16 as stored (pages to tagged text, DASH attention) | 1 | [unlimited-ocr-2.md](docs/recipes/unlimited-ocr-2.md) |

- **Flash Next FP4-Local**: NVIDIA's QAD checkpoint of Qwen3.8 Flash Next. On a single Spark, it loads in 17 s (vs 1,000 s in vLLM), responds to a held turn in 0.109 s (vs 0.261 s), and streams replies at 61 tok/s (vs 46.6 tok/s). Cold 16,000-token prompt prefill runs at 2,080 tok/s. Prompts support inline images.
- **GLM-5.3-Flash FP4-Local**: NVIDIA's QAD checkpoint of GLM-5.3-Flash (177 GB). Runs across two Sparks (87.8 GiB per rank), held bit-compatible with reference vLLM at TP=2.
- **Qwen3-ASR-1.7B / 0.6B**: High-speed speech transcription via `POST /v1/audio/transcriptions` (WAV, FLAC, MP3, Ogg Vorbis, M4A, AAC-LC, Opus, and more, up to 1,200 s). BF16 as stored (7.2 GiB), loads in 1 s, real-time factor 0.051, and 59 tok/s decode (WER and CER within 0.1 points of `qwen-asr` on 64 clips). Also supports `/v1/chat/completions` with audio inputs ([serving-media.md](docs/serving-media.md#chat)).
- **Unlimited-OCR-2**: Multi-page document OCR (up to 128 pages per request) using DASH attention with PageCountGuard and `max_page_tokens`: 14.71 GiB at 131,072-token context, ~160 tok/s decode through `serve`.

## Running it

The quickest path is the stack behind one address: copy [`zspark.local.example.sh`](zspark.local.example.sh) to `zspark.local.sh`, point `STACK_ARGS` at a checkpoint directory (see [docs/weights.md](docs/weights.md) for the pinned downloads), then:

```bash
./start.sh            # zspark stack detached, log under ~/.cache/zspark/logs
./stop.sh             # stops the recorded stack, and only that stack

# Client query, once the log shows the router up (default 127.0.0.1:8888)
curl localhost:8888/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"stream":true}'
```

`DRY_RUN=1 ./start.sh` prints the exact `zspark stack` command it would run. The launchers are wrappers — the [stack section](#flash-next-ocr-and-asr-behind-one-address) and [docs/serving.md](docs/serving.md) describe what they start and every flag.

Individual commands run directly too:

```bash
# Interactive CLI chat
./zig-out/bin/zspark chat DIR --no-thinking "List the first five prime numbers, separated by commas."

# OpenAI-compatible API server
./zig-out/bin/zspark serve DIR --port 8000 --name qwen3.8-flash-next

# Client query
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"stream":true}'
```

`DIR` is a checkpoint directory. `serve` provides OpenAI Chat Completions, Responses, and Anthropic Messages compatibility through TensorFold 1.3.4's host. OpenAI chat and completion requests support one choice per request (`n=1`). zspark rejects `cache_salt` requests. See [docs/serving.md](docs/serving.md) for drafting options, routing flags, state retention (prefix caching), request concurrency (`--parallel`), context and KV cache configuration (`--context`, `--kv-dtype`), tool calls, inline images, and audio transcription.

### sparkDash monitoring

[sparkDash](https://github.com/MiaAI-Lab/sparkDash) can monitor `zspark serve` through its TensorFold backend: point a unit's LLM port at the server, or at `zspark stack`'s router port (8888 by default). `/v1/models` identifies the backend; `/metrics` supplies token totals, running and waiting requests, KV usage, draft acceptance, and latency statistics.

- **Prefix Cache Hit Rate**: Counters report prompt tokens restored from retained states. The hit rate is restored prompt tokens divided by all measured prompt tokens. sparkDash's **Token totals** view displays cached vs computed prefill.
- **Memory & Budgets**: Retained-state occupancy and applied byte budgets are exposed on `/metrics`; `/memory` reports the prompt-cache layout.

## Tests

Tests are organized into host and GPU tiers:

- `zig build test-host`: Host tests for tokenization, chat templates, grammar masks, server routing and sampling. Runs without a GPU or CUDA toolkit.
- `zig build test-weights`, `zig build test-qwen4`, `zig build test-glm5`: Targeted GPU kernel and forward tests against random weights or local reference checkpoints.
- `zig build test`: Full test suite validating end-to-end forward passes, greedy tokens, and sampled generation against reference outputs.

See [docs/environment.md](docs/environment.md) for environment variables (`ZSPARK_FLASH_NEXT_FP4_LOCAL`, `ZSPARK_GLM_FP4_LOCAL`, etc.). Pre-commit hooks in `.githooks` validate code formatting and specifications (`git config core.hooksPath .githooks`).

## Flash Next on two Sparks

One Spark is the default. Two ranks require building with NCCL (`zig build -Dnccl=/usr`):

```bash
tools/two_ranks.sh serve DIR --host 127.0.0.1 --port 18888
```

`DIR` is the FP4-Local checkpoint, located at the same path on both Sparks. Rank 0 serves requests; rank 1 participates in tensor-parallel compute. Each rank holds ~48.5 GiB of device memory (compared to 81.0 GiB on a single Spark). The MTP drafting head is also split across ranks, achieving 70.4 tok/s streaming decode (vs 62.0 on one Spark) and cold prefill of 2,598 tok/s (vs 2,068 tok/s).

**Two whole copies**: An alternative multi-Spark configuration ([docs/serving.md](docs/serving.md#two-copies-behind-one-address)):
```bash
tools/two_copies.sh serve DIR --parallel 4
```
Runs one independent model copy per Spark behind a local routing proxy (`127.0.0.1:18887`), doubling total throughput for concurrent conversations.

## Flash Next, OCR and ASR behind one address

`zspark stack` runs the models for my LLM, document OCR, and speech transcription workloads behind one address (default `127.0.0.1:8888`). Each engine runs as its own process; the stack manages startup, health checks, routing, and shutdown:

```bash
zspark stack FLASH_DIR --ocr OCR_DIR --asr ASR_DIR
```

- Audio requests (`/v1/audio/*`) route to the configured ASR engine.
- Chat requests select Flash Next or Unlimited-OCR-2 by the request's `model` field.
- CUDA MPS (`--mps PCT`, 25% default) limits the small OCR/ASR models' SM share while Flash Next uses the whole GPU.
- Set `--peer` to the second Spark's direct-link IPv4 address to run Flash Next over two ranks. Single-rank ASR runs on the peer, while Nemotron ASR spans both Sparks.

For a stack that runs in the background across sessions, `./start.sh` and `./stop.sh` wrap `zspark stack`: `start.sh` detaches it, logs under `~/.cache/zspark/logs`, and records the process's identity (pid plus its start tick) so `stop.sh` signals only the stack this tree started — a stale or reused pid is never signalled. Copy `zspark.local.example.sh` to `zspark.local.sh` (untracked) and put the machine's checkpoint paths and flags in its `STACK_ARGS`; one Spark behind loopback is the default, while `--ocr`, `--asr`, `--peer` (two Sparks, needs `zig build -Dnccl=/usr`), `--drafter`, and a LAN `--listen` are all explicit opt-ins. `DRY_RUN=1 ./start.sh` prints the exact command.

The [stack guide](docs/serving.md#flash-next-ocr-and-asr-from-one-command) shows how to run Flash Next with DFlash2 and Nemotron transcription across two Sparks, and how to add OCR with `--ocr`.

## GLM-5.3-Flash

GLM-5.3-Flash requires two DGX Sparks joined over NCCL (`zig build -Dnccl=/usr`), utilizing MTP drafting (or DFlash2 via `--drafter` for 1.16x throughput). `serve` supports a 262,144 token context window. Prompts with common prefixes resume from saved cache snapshots. See [docs/recipes/glm-5.3-flash-fp4-local.md](docs/recipes/glm-5.3-flash-fp4-local.md) for launch parameters, cluster setup, and benchmark data.

## Layout

`build.zig` defines the build pipeline; all Zig and CUDA source code resides under `zig/`:

- `zig/src/`: Zig engine (memory management, runtime, scheduler, model architectures).
- `zig/kernels/`: Optimized CUDA kernels compiled by `nvcc` for `sm_121a` (`zs/` C ABI, `nvfp4/`, `bf16/`, `qwen4/`, `glm5/`).
- `zig/vendor/tensorfold/`: Upstream TensorFold 1.3.4 host integration (server, tokenizer, templates).
- `zig/tests/`: Comprehensive unit, integration, and reference tests.
- `tools/`: Host utilities, benchmark scripts, and test runners (`check.sh`).
- `specs/`: Execution specifications and architectural decisions (see [specs/README.md](specs/README.md)).

## Contributing and security

[CONTRIBUTING.md](CONTRIBUTING.md) has the clean-checkout workflow (host tests with no GPU, targeted GPU areas, formatting, spec checks) that matches what public CI runs. Security vulnerabilities are reported privately through the channel in [SECURITY.md](SECURITY.md) — never a public issue. To verify a running server end to end, `tools/http_smoke.sh` exercises `/health`, `/v1/models`, streamed chat and a required tool call ([docs/http-smoke.md](docs/http-smoke.md)).

## Upstream contributions

Issues reported to [ashhart/TensorFold](https://github.com/ashhart/TensorFold):

- [#2082](https://github.com/ashhart/TensorFold/issues/2082): optional engine refusal, returning HTTP 400 before a stream starts.
- [#2083](https://github.com/ashhart/TensorFold/issues/2083): PNG tRNS key low byte handling.
- [#2084](https://github.com/ashhart/TensorFold/issues/2084): client constraint field name in `api.Structure`.

All three issues are closed as completed upstream. Their fixes are included in the bundled TensorFold host; [UPSTREAM](zig/vendor/tensorfold/UPSTREAM) records the import and [PATCHES.md](zig/vendor/tensorfold/PATCHES.md) lists the remaining local changes.

zspark is licensed under Apache-2.0 (`LICENSE`), consistent with TensorFold. See `NOTICE` for upstream attributions.
