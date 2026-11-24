# zspark

LLM decoding for the NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory) in Zig and CUDA. It derives from
TensorFold 0.6.5 (commit 609ca41, the references' and recordings'): it keeps its CUDA kernels and replaces
the Python, PyTorch and Triton layers. `serve` is TensorFold 1.3.4's host, in `zig/vendor/` (072, 113, 141, 145, 167, 174, 175, 176).
No Apple or MLX.

## Scope

- What is served and on which checkpoints: `README.md`, `docs/recipes/`.
- Python only under `tools/reference/` (fixtures exported from TensorFold), never at runtime.

## Before any work

Read `specs/WORKER.md`: the standing rules (where to work, locks, the LAN server,
checks, load budgets, spec entries, commits, what goes to the owner). Then the nearest `AGENTS.md` below and your row of the Active spec.

## Build and test

```bash
zig build   # nvcc compiles the kernels for sm_121a, Zig links them; -Doptimize=debug|safe|fast (default fast)
tools/check.sh compile   # everything compiled, nothing run, no GPU: while editing
tools/check.sh text|server|weights|qwen4|glm5|whole [FILTER]   # an area (test-text ...); whole: 2 whole-model tests
tools/check.sh host|full|ranks # a tier: no GPU; all, half on each Spark (--one: here); two ranks. FILTER is -Dtest-filter=TEXT
# check.sh takes locks, logs runs; ~/.cache/zspark/env.sh sets paths below (unset skips those tests).
#   ZSPARK_FLASH_NEXT_FP4_LOCAL=<checkpoint dir>  ZSPARK_REFERENCE_FP4_LOCAL=~/.cache/zspark/reference-fp4-local
#   ZSPARK_GLM_FP4_LOCAL=<checkpoint dir>  ZSPARK_REFERENCE_GLM_FP4_LOCAL=...-glm-fp4-local-card
#   ZSPARK_GLM_DRAFTER, ZSPARK_REFERENCE_GLM_DRAFTER, ZSPARK_REFERENCE_VISION (docs/environment.md)
zig build check-specs   # specs/ and the size budgets (no GPU); test-specs: the validator's tests
zig fmt --exclude zig/vendor zig build.zig tools   # before a commit
```

Zig 0.17.1 (`/home/linuxbrew/.linuxbrew/bin`), the CUDA 13 toolkit at `/usr/local/cuda`. Zig 0.17's
standard library is not older releases': read it under `$(zig env)`'s lib directory, guess no API. Version:
`build.zig.zon`: unpadded `YYYY.M.D` + mode/commit/dirty; tags `vYYYY.M.D`.
Outside `check.sh`, take the locks in `specs/WORKER.md`; run cache pruning outside a build driver.

## The rule everything serves

**A row's bits never depend on the rows beside it.** A verify window, a prompt chunk and a serial step give the
same bits for the same row. Draft verification is only exact because of it, so no change may trade it for speed.
Exactness is against zspark's own serial decoding; against TensorFold compare with a tolerance. How kernels
keep the rule and tests check it: `zig/kernels/AGENTS.md`, `zig/src/AGENTS.md`. FP4-Local (specs 115, 143):
a row's form follows its window's rows, bf16 to 48 on GLM, 80 on Flash Next (steps, verify), quantized
above as the card's vLLM; a verify window and a serial step stay one form, and within a form a row's bits stay its own.

## The owner's standing orders

- Order of work: Flash Next until it passes TensorFold on what a client sees (load, first token, tokens a
  second); GLM after. 2026-10-12: one stream's speed first, a prompt's included, the checkpoint's precision
  kept, by optimizations that lose nothing; on two Sparks two ranks first, two whole copies an extra scheme.
- The model that emits a reply's tokens is run in the format its checkpoint stores: zspark quantizes nothing
  of it; only the pipeline (kernels, launches, caches, drafting) is optimized. What only drafts may be
  quantized at load, chosen by tokens a second and drafts kept (2026-10-10).
- 2026-10-19: activations may take the precision of the card's launch configuration (the Hugging Face page's
  command), read in vLLM's source, never less in any phase; GLM's rows take their window kind's form (115).
- Speed first, on tensor cores where a product fits. A host model or a kernel replaced is held near, not to
  the bit; the row rule stays bits and stays closed.
- An open decision is taken as the coordinator recommends and reported, not waited for; the next optimization
  spec is opened by expected gain; a bit-exact speed-up is on whatever its size once
  it is measured outside the loads' spread (2026-10-11; the 2% bar is gone). What would cost the row rule, or
  change behaviour, an interface or a default, goes to the owner.
- TensorFold is the reference; where it gives none, what both vLLM and SGLang do, read in their source.
- A measurement has a plan and a budget of loads before it runs. A round costs the least that answers it:
  `check.sh compile`, then the test touched; a worker runs no tier. 2026-10-20: the full tier and `ranks` at a
  stage (a spec's close, a batch of merges), not each WS; a merge whose tests already cover it runs nothing more.
- 2026-10-21: parallelize independent work across both Sparks, with disjoint files and a named machine per
  task. Reuse check evidence; run minimal filters. TP=2 timings reserve both; do not disturb servers.
- GLM's DFlash2 drafter's weights are never committed or redistributed in any form, and no copy is written;
  2026-10-20: sent over the link into rank 1's memory between the owner's two Sparks is allowed.
- Sizes: a source file and a spec 60,000 bytes, this file 10,000, a nested `AGENTS.md` 14,000 (2026-10-10).
- 2026-10-21: TensorFold, vLLM, SGLang, [NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference) and Mia's recipe (148)
  may always be read and followed, for behaviour and for how a thing is built. An open question reads the
  relevant paths in these sources and says what each does or does not cover; TensorFold remains the
  behaviour reference above. Hardware, precision, exactness and provenance rules still apply.
- For OCR and ASR, offline NVFP4 checkpoints may keep sensitive layers in higher precision.

Active: none.
See `specs/README.md` for the active index and workflow.

## Specs

Work that spans sessions or agents runs under a spec (`specs/README.md`: the index, the rules). The
coordinator reserves the ID, writes the spec and merges branches; a worker claims its row and appends a dated
entry. A spec is an execution record: what must outlive it goes into code, tests, `README.md`, `docs/serving.md`
(what `chat` and `serve` do for a client) and the nearest `AGENTS.md`.

## Layout

- `zig/src/`: the program. `cuda/`: the runtime, buffers; `weights/`: checkpoints, linears; `decode/`: shared
  decoding; `text/`, `server/`, `image/`, `audio/` (decoders): host only; `ranks/`: NCCL; `models/`: a directory
  a model, under TensorFold's family name (`models/qwen4/`, `models/glm5/`).
- `zig/tests/`: every test, at its source's path (`tests/text/x_test.zig` for `src/text/x.zig`), and
  `fixtures/`. The roots in `zig/` list them (`tests.zig`, `host_tests.zig`, an area's): add a new file there.
- `zig/kernels/`: CUDA sources, listed in `build.zig`. `zs/` is the C ABI; `nvfp4/`, `common/` are taken from TensorFold;
  `bf16/`, `qwen4/`, `glm5/` are rewritten from its Triton sources.
- `zig/vendor/tensorfold/`: TensorFold's host as upstream has it and zspark's patches (its own `AGENTS.md`);
  `stb/`, `libwebp/`: the C decoders of `image/`, the same way.
- `specs/`: tracked execution specs; `tools/`: host-only tools,
  the validator (`check_specs.zig`), `reference/` the Python fixtures and benches.

The data flow between these: `docs/architecture.md`, "Layers".

## Local AGENTS.md hierarchy

Read the nearest first; each extends its parent. A `REFERENCE.md` beside one holds what a task reads only
when it touches what a section names.

- `zig/kernels/AGENTS.md`: the C ABI, exactness in device code, provenance, compiler flags, CUTLASS.
  `zig/kernels/glm5/AGENTS.md`: GLM's kernels' own rules.
- `zig/src/AGENTS.md`: Zig 0.17 habits, device buffers and views, how tests are written.
- `zig/src/models/AGENTS.md`: what a model gives the shared decoder and the dispatch; then `qwen4/`,
  `glm5/`, `unlimited_ocr/AGENTS.md`: a model's structure, its sources, shapes, naming.
- `zig/src/text/AGENTS.md`, `server/AGENTS.md`: tokenizer and template; the scheduler and constraints.
- `specs/AGENTS.md`: who edits what in a spec, and when. `specs/WORKER.md`: a worker's standing rules.
- `tools/AGENTS.md`, `tools/reference/AGENTS.md`: host-only tools; the Python fixtures and the benches.
- `zig/vendor/tensorfold/AGENTS.md`: TensorFold's host: an import, a patch, a release.

## Conventions

- Match the surrounding code's comment density.
- Before adding a file, a directory or a model: `docs/architecture.md` (what a change tidies), `docs/naming.md`.
- Before writing a rule into one model, look for it in the other models and in `decode/`: what it does
  apart from what the model's state holds (which snapshot goes, a dropped state's buffers reused, a stride) is
  shared, called from both, not copied; the commit says which. A copy two models already hold is moved to
  `decode/` in the change that would touch it twice, or gets a spec row (2026-10-18; spec 112 is the first).
- Every `ZSPARK_*` variable has its row in `docs/environment.md`, and what a checkpoint decides (storage,
  what loads how, defaults, figures) is in its page under `docs/recipes/`: changed in the same commit.
- Diagrams are `mermaid` blocks, no ASCII art.
- TensorFold is the reference for behaviour. When a port departs from it, say so in the code and in the commit.
