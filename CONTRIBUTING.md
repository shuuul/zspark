# Contributing to zspark

Thanks for your interest. zspark is a personal-research inference engine for the NVIDIA
DGX Spark (GB10, sm_121a) — Zig plus CUDA, no Python at runtime. Contributions are
welcome as issues, patches, and pull requests; this file is the workflow that works from a
clean checkout on any machine, without the maintainer's environment.

## Ground rules

- **Correctness before speed.** The engine's contract is the *row rule*: a row's bits never
  depend on the rows beside it — serial decoding, a verify window and a concurrent round
  produce identical tokens. A change that trades that for speed will not be merged; see
  `AGENTS.md` ("The rule everything serves") and `docs/architecture.md`.
- **Apache-2.0.** Contributions are under the repository's license (`LICENSE`). Do not
  submit code you cannot license.
- **No weights, no credentials, no machine-specific paths.** Model weights are never
  committed (GLM's DFlash2 drafter must not be copied or redistributed in any form —
  see `docs/weights.md`). Keep personal paths, LAN addresses and tokens out of files,
  tests and commit messages.
- **TensorFold is the behaviour reference.** `zspark serve` is TensorFold 1.2.7's host
  vendored under `zig/vendor/tensorfold/`; where behaviour is ambiguous, match upstream
  (`zig/vendor/tensorfold/PATCHES.md` lists intentional departures).
- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `build:`, `refactor:`,
  `chore:`), one logical change a commit.

## Set up

Requirements are **Zig 0.17.1**, **Rust and Cargo 1.85 or later** for the audio decoders, and the
**CUDA 13 toolkit** (`nvcc`) on a DGX Spark for device code. The first build fetches CUTLASS and
the audio crates pinned in `zig/src/audio/Cargo.lock`. Host tests run on native Linux without CUDA.

```bash
git clone https://github.com/shuuul/zspark.git && cd zspark
zig build                 # nvcc compiles kernels for sm_121a; -Doptimize=fast is default
./zig-out/bin/zspark probe
```

- `zig build -Doptimize=debug|safe|fast|small` selects the mode (`safe` adds runtime
  checks — good for development).
- `zig build -Dnccl=/usr` enables NCCL for two-Spark tensor-parallel runs.
- Checkpoint directories come from Hugging Face, at the revisions each recipe pins:
  `docs/weights.md` has the download commands (`hf download REPO --revision COMMIT`).

## What to run before sending a change

Pick the cheapest check that covers the change; commands are runnable from a clean
checkout with no maintainer-specific environment:

| Change | Command |
| --- | --- |
| Device code or build graph | `zig build` (compile everything; CUDA toolkit required) |
| Any host-side code, on **any machine incl. no GPU / no CUDA toolkit** | `zig build test-host` |
| One area touched (tokenizer, server, audio, weights, a model) | `zig build test-text` / `test-server` / `test-image` / `test-audio` / `test-weights` / `test-qwen4` / `test-glm5` / `test-unlimited_ocr` / `test-qwen3_asr` / `test-nemotron` — each `-Dtest-filter=NAME` narrows to matching tests |
| Test area inventory or Qwen4 file coverage | `tools/check.sh inventory` (no Zig or CUDA required) |
| Docs / specs only | `zig build check-specs` |

- CI runs `tools/check.sh inventory` and then `zig build test-host` on matching pull requests
  and manual dispatch (`.github/workflows/specs.yml`). `test-host` includes the spec validator and
  its tests and needs neither a GPU nor `nvcc`. Checkpoint tests skip when their `ZSPARK_*`
  variable (`docs/environment.md`) is unset.
- GPU area steps need a DGX Spark; checkpoint-dependent tests additionally need the
  variable pointing at a local checkpoint directory (see above). If your change touches
  device code and you have no Spark, say so in the pull request — the maintainer runs the
  device tiers before merge.
- **Formatting:** `zig fmt --exclude zig/vendor zig build.zig tools` before committing.
- **Local hooks:** `git config core.hooksPath .githooks` installs `pre-commit` /
  `pre-push` hooks that run the spec checks automatically.
- **Smoke-test a running server** after serving changes: `tools/http_smoke.sh` against a
  local `zspark serve` or stack (`docs/http-smoke.md`).
- On the maintainer's shared Sparks, use `tools/check.sh` for builds/tests: it takes the
  machine, GPU and cache locks described in `specs/WORKER.md`. Direct build commands
  need the shared machine and cache locks; device runs also need the GPU lock.

## Specs and where things go

`specs/` holds the project's execution records — read `specs/AGENTS.md` before editing
one, and run `zig build check-specs` after. Most external contributions do not need a
spec; open a discussion issue first for anything that changes behaviour, an interface, a
default, or a model's served output.

Other conventions worth knowing before writing code:

- Every `ZSPARK_*` variable gets a row in `docs/environment.md` in the same commit.
- A new test file is registered in `zig/*_tests.zig`; a new `zig build` test area is a row
  of `tools/test_areas.tsv`.
- Python lives only under `tools/reference/` (fixtures exported from TensorFold); the
  runtime never calls it.
- Serving behaviour a client sees is documented in `docs/serving.md` /
  `docs/serving-media.md` — update them in the same commit as the behaviour.

## Reporting bugs and security issues

- Ordinary bugs: open a GitHub issue with the commit (`zspark version`), the command, and
  the observed output.
- Security vulnerabilities: **do not** open a public issue — follow
  [SECURITY.md](SECURITY.md) for the private reporting channel.
