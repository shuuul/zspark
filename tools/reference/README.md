# The reference files and the scripts that write them

What TensorFold's exported references hold and how to read them, then what each script here needs known beyond
its own docstring or header, then the export commands; `AGENTS.md` beside this file has the rules for editing.

## Reading the vision reference

`vision_export.py` (spec 019 WS-06) on the NVFP4 checkpoint; its docstring has every file's names. What the
files showed on 2026-10-17 (spark-b, `~/.cache/zspark/scratch-019/*.log`):

- `zig/tests/fixtures/vision/cases.json` holds what is small: each image file's bytes, decoder and RGB sha256
  with TensorFold's `content_hash`, and each case's rendered prompt, expanded ids, grids, spans, positions
  (3, n), delta and the replies. The RGB, `pixel_values` and every row are in `ZSPARK_REFERENCE_VISION`.
- The least pixels the processor is given are 32 x 32 (TensorFold's default in `prepare`), not the checkpoint's
  `shortest_edge` 65536: `tiny` (48 x 40) becomes a 1 x 2 x 4 grid, 2 tokens. Where the resize keeps the size
  (`shapes`), `pixel_values` is the RGB in float64 rescaled and normalized, then rounded to float32.
- The tower gives an image alone the bits it gives it among the case's others for `shapes` and `bands`, not
  for `tiny` beside `bands` (1 ULP of bf16 at most, 0.00098): TensorFold's tower does not keep zspark's row
  rule across images. `CASE.tower`'s `features` are the bits the engine embeds; each GPU stage gave the same.
- On the decode path the serial steps, the prompt as one window and windows of 16 hold the same bits at all 653
  points of both cases, both slots, image rows included; a repeated run too. The `layers` stage's files equal
  the whole load's at layers 0 to 3.
- The resize is torchvision's (transformers 5.18's `Qwen2VLImageProcessor` is a torch backend): uint8 bicubic
  with antialias, torch's integer path, not Pillow's (6 bytes of `bands` differ by 1 under `Image.resize`). The
  normalization is torchvision's fused `(f32(x) - 127.5) / 127.5`. `vision_cases.py` (spec 019 WS-01) gives
  the processor's grids and refusals at more sizes and budgets, and 12 random images' resized bytes.
- Replies (32 tokens, no stop at the end of turn): on `pair` the engine's (prompt kernels, bf16 slots) and the
  bf16 decode path's are the same tokens, the `.f32` decode path's part at token 3; on `shapes` the engine's
  and the decode path's part at token 2 ("There are three shapes" against "There is a red circle"), and the two
  decode paths at token 21, past the end of the answer.

## The scripts

Each script's docstring or header is its manual: what it wraps, its file's format, its command. Below is only
what those do not say.

- `flash_next_export.py`: shared helpers for `vision_export.py` and `sampling_cases.py`. Its docstring describes the tensor names and TensorFold internals.
- `flash_next_tap_export.py` (spec 122): `--synthetic` writes made 4-stream residuals and both means (GLM f32-acc and sequential bf16) as safetensors under `~/.cache/zspark/scratch-122/`; no checkpoint, no GPU. `--from-sglang --target DIR` prints the SGLang-patch capture command and exits 2: TensorFold does not read FP4-Local (spec 121) and a real capture needs that load. zspark's mean is f32-acc (spec 125 WS-01). `--from-vllm fetch|assemble` is the real capture (spec 125 WS-02), run by `flash_next_taps_vllm.sh MODEL_DIR REFERENCE RAW`: the card's vLLM booted eager by `vllm_reference.sh` with `flash_next_taps_vllm.patch` on a scratch copy of the image's `qwen4_exp/nvidia/model.py`, layer i's taps the `combined` of layer i+1's `combine_and_mix`; RAW/fixture/ (a case's `NAME.safetensors`, `MANIFEST.json` with vLLM's spreads and the bound) goes to `$ZSPARK_REFERENCE_FP4_LOCAL/taps/`, which `compare_vllm_test.zig`'s Z1 reads. The docstring is the manual.
- `glm5_export.py`: GLM's first layers and MTP layer, the two ranks as two threads of one process whose
  `comm` is a barrier; pinned to TensorFold 0.6.6, the last with this family in Python. `glm5_drafter_export.py` imports it.
- `glm5_drafter_export.py`: TensorFold's `dflash2.Drafter` for both ranks on made taps (a seed a case), its
  4-bit copies replaced by BF16 faces, the pack's embedding and head and no main layer: 9 s to load, 7.4 GiB.
  A case is `NAME=POSITIONS,PENDING,SEED`; it runs twice, flat in one update and in the ring in pieces, and
  the file is the ring's (TensorFold's two do not differ). `zig/tests/models/glm5/dflash2/chain_reference_test.zig` reads
  it from `ZSPARK_REFERENCE_GLM_DRAFTER`. The drafter's weights are CC BY-NC-ND: the files hold
  activations and stay under `~/.cache/zspark/`. Its docstring has the names; the command is under "The
  commands".
- `openai_exchange.py`: the structured replies are validated with `jsonschema`, a Pydantic model's schema
  (`$ref`, `Optional`) is parsed back into the model, and the refused schema is the 400 for a `pattern`. The
  streamed calls (spec 027) are each sent whole and streamed with the same seed and compared (content, names,
  arguments, finish, usage), two calls are also asked with one allowed, a call's name is required before its
  last argument delta and a long string in eight deltas or more; it prints when each call's name and last
  delta came.
- `responses_exchange.py`, `anthropic_exchange.py`: each also runs the package's own stream accumulator
  (`responses.stream`, `messages.stream`); the Responses one checks that `store: false` stores nothing.
  `ZSPARK_URL` names a server elsewhere than `http://127.0.0.1:8011`, for all three exchanges.
- `client_bench.py`, `client_bench.json`, `client_bench.sh`: the script keeps the server's log and its memory.
  The context is 81920 unless `CONTEXT` says otherwise, the one TensorFold's recorded loads were taken at. The
  `turns` requests (spec 030) were added after spec 024's, so those stay the bytes they were; TensorFold has
  not been measured on them. `required_call_bench.json` is the set `SET` names for spec 034.
- `glm5_client_bench.sh`'s `WRAP` wraps zspark's rank 0 and `WRAP1` its rank 1 (the report is written on the
  peer). A traced round is not a served one's: spec 093's 3.8 ms a round under `nsys` was nothing untraced.
  A change under 2% is timed by `AFTER` sending the replay ten times a load, more than one load a build.
- `glm5_client_bench.sh`'s `TF_WRAP=1` traces TensorFold's rank 0 under `nsys` inside its container (spec 091):
  the host's Nsight Systems is mounted read-only at `/nsys` (`NSYS_HOME`, 2025.3.2) and `TF_TRACE_DIR`
  (default `~/.cache/zspark/scratch-091`) at `/trace`, so the report is `/trace/rank0.nsys-rep` on the host,
  written when `docker stop` ends the rank (1.3 s of the 60 s grace on the wrapper's trial). Read it with
  `nsys export --type sqlite`; a decode window is found by launches a second, which jump when a prefill ends.
- `shared_acceptance.sh`: the server's log in OUT_DIR has the pool's account of its device buffers.
- `readme_table.sh`, `readme_table.py`: the load is `zspark generate DIR 0 rows=N score=FILE cases=FILE`. The
  two plain (serial) replies are 128 tokens; the reply at which the device memory is read is 32 tokens; the
  4000-token prompt is the harness's at its rendered length, two tokens decoded. `score=FILE` scores those ids
  on the same load, in windows of 16, as `zspark score` does. The load seconds are printed once the first
  reply's graphs are captured, which is where `chat` prints `loaded in`.
- `draft_policy.py`: one reply moves by several percent either way under any change of floor, so a rule is
  judged over a set, recorded on one load with `zspark generate DIR 0 cases=FILE`. When variants are timed on
  one load, a text's first reply runs 1 to 4% slower than the same text decoded again (every window, at equal
  rows), so each variant is run twice and the second passes are compared.
- `lands_cases.py`, `lands_fit.py` (spec 085): `lands_cases.py ZSPARK DIR OUT.cases` writes 120 cases, a
  fitting set and a held-out set of prose and code; `zspark generate DIR 0 cases=OUT.cases > RUN.out` records
  their chains on one load (about 25 minutes a pack), and `lands_fit.py RUN.out OUT.cases` prints the seven
  numbers of `mtp_rule.Paying.Landing` fitted on the fitting replies, how they sort and calibrate the held-out
  drafts beside the head's probability, and the rule that reads them replayed on the held-out replies at 1, 2
  and 4 seats (standard library only). `chains` prints a probability to four decimals, and the packs'
  numbers in `mtp_rule.zig` were fitted on the probabilities whole: a fit from `RUN.out` gives a slope 0.012 to
  0.016 lower and intercepts within 0.04, and its replay stands within 1.2 points of theirs. Spec 102 added
  the numbers refitted while serving, as vLLM's `OnlineAcceptanceEstimator` does, played against them fixed;
  and `lands_fit.py costs RUN.out`: `mtp_rule.Paying.Costs`' `row` and `level` from the load's cases drafted
  with `rounds` (names with `-drafted-`), the windows' ms by their rows and the head's seconds a level. `lands_cases.py held OUT_DIR` writes
  the held-out prompts as `client_bench.py streams --prompts FILE` reads them; with `--all` (`STREAM_ALL=1`
  of `client_bench.sh`, whose `STREAM_PROMPTS` takes such files, comma separated, a `streams-SET.json` each)
  a count runs every group of that many prompts, their tokens over their seconds: the timing of a draft rule,
  one reply moving by several percent either way.
- `device_memory.py`: `ZSPARK_ALLOCATIONS=1 zspark serve ...` writes the lines, with each buffer's return
  addresses. A kind is read from a source line's variable name (`SCRATCH`), so a renamed buffer needs its name
  added there.
- `turns_part.py`: `--set all` is the conversations and the shapes together; one load serves all of it, and the
  whole set holds the GPU for half an hour.
- `requantize_cases.py`: it must run on the device (PyTorch divides by a scalar differently on the CPU), where
  it holds a few MiB.
- `grammar_masks.zig`, `grammar_masks.cc`, `grammar_masks.sh`: `grammar_masks.txt` is the tiny tokenizer with
  every mask whole (155 KB); `grammar_masks_checkpoint.txt` the checkpoint's vocabulary with each mask as its
  count and two hashes (94 KB). Both are from xgrammar 8262b5c:

  ```bash
  git clone https://github.com/mlc-ai/xgrammar X && git -C X checkout 8262b5c
  cmake -S X -B XB -DXGRAMMAR_BUILD_PYTHON_BINDINGS=OFF -DCMAKE_BUILD_TYPE=Release && make -C XB -j20
  tools/reference/grammar_masks.sh X XB "$CK"      # GRAMMAR_MASKS_SHOW=1 prints each schema's differing tokens
  ```

  At 8262b5c the mask after an object reached through a recursive `$ref` lacks `},` and the like, which
  xgrammar's acceptor takes: 32 tokens of the checkpoint fixture.

## The commands

The vision reference's three stages are in `vision_export.py`'s docstring (another mount for the checkpoint).

The drafter-only fixture can be regenerated with the TensorFold-compatible embedding and head used by the reference exporter (`PACK`). Run under the locks in `specs/WORKER.md`:

```bash
. ~/.cache/zspark/env.sh; mkdir -p ~/.cache/zspark/reference-glm-drafter
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/cache -e PYTHONPATH=/tf/src \
  -v ~/.cache/zspark/tensorfold-cb2ebf0:/tf:ro -v ~/.cache/huggingface:/hf:ro \
  -v ~/.cache/zspark/tensorfold-kernels:/cache -v "$PWD/tools/reference":/ref:ro \
  -v ~/.cache/zspark/reference-glm-drafter:/out --entrypoint python tensorfold-glm53:v0.6.0 \
  /ref/glm5_drafter_export.py "/hf/${PACK#$HOME/.cache/huggingface/}" \
  --drafter "/hf/${ZSPARK_GLM_DRAFTER#$HOME/.cache/huggingface/}" --out-dir /out \
  --tensorfold-commit cb2ebf0540f42604e2759b2ddef497861e928248 --case short=37,1234,1 --case long=2376,1234,2
```

Unlimited-OCR-2's fixtures (spec 147) come from the checkpoint's HF `infer` in the `hf_env.sh`
environment, with Triton explicitly off. `one_page` and `two_pages` record the image views, each layer's
DASH router outputs and the greedy reply. Fixture page bytes retain spec 137's provenance; only OCR2
checkpoint and reference paths are used by current tools.

`dry` imports the remote code and builds the model with parameters on the meta device, without weights;
`--smoke` checks the hooks on the CPU with three new tokens (not a fixture):

```bash
source tools/reference/hf_env.sh
python tools/reference/unlimited_ocr/export_v2.py dry
flock -s ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock \
  python tools/reference/unlimited_ocr/export_v2.py fixtures --pages PAGE_IMAGES
flock -s ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock \
  python tools/reference/unlimited_ocr/export_v2.py cer --cer CER_INPUT
python tools/reference/unlimited_ocr/cer.py --self ~/.cache/zspark/reference-unlimited-ocr-2/cer
```

`PAGE_IMAGES` contains `paper_p01.png` and `paper_p03.png`; the exporter copies them into its output's
`pages/`. `CER_INPUT` contains `manifest.json` (`{"pages":[{"name":"...","image":"..."}]}`) and its
images. Outputs keep image paths relative and copy the inputs beside the manifest. Defaults read
the current OCR2 output directory's `pages/` or `cer/`.
Existing `shared_tensors.json` files record historical v1/v2 tensor comparisons; new exports do not
produce them.

With `fixtures --only`, an input page shared with a skipped case must keep the same bytes; changed
shared inputs require regenerating all cases. `cer --only` preserves the unselected manifest entries,
including when the input and output are the same directory.

## The map in detail

The map in `AGENTS.md` has one line a script; this is the longer form (each script's docstring or header is its
manual: what it wraps, its file's format, its command).

- `hf_env.sh`: the HF-code reference environment.
- `unlimited_ocr/export_v2.py`: Unlimited-OCR-2's DASH router and attend, layer by layer, for a prompt and its steps
  (`prefill.*`, `step.*` at every decode call, `detail.*` at the first steps, at each reply `<PAGE>` and late),
  recomputing what HF's functions keep inside (`k_bar`, the top-16, the entmax weights) with HF's ops and
  asserting them against what HF returns; the near-tie gaps that let a kernel's block ids be told from a tie.
- `flash_next_export.py`: TensorFold's decode-path forward by block, serial steps and windows, as safetensors
  with a JSON summary; the MTP head, a sampled reply, on a checkpoint TensorFold reads (none supported since spec
  121; `vision_export.py` and `sampling_cases.py` import it).
- `vision_export.py`: Flash Next's BF16 vision fixture generator from TensorFold 0.6.5: its own
  images, their decoded RGB, the processor's outputs, the tower's points, two image prompts' rows by block in
  `flash_next_export.py`'s form (which it imports) and their greedy replies; `zig/tests/fixtures/vision/` and
  `~/.cache/zspark/reference-vision/`, by stages (`processor` without a GPU). `README.md`, "Reading the vision
  reference".
- `vision_cases.py`: TensorFold 0.6.5's image processor alone on sizes and budgets of its own (spec 019 WS-01):
  grids, lengths, deltas and refusals in `zig/tests/fixtures/vision/processor_cases.json`, random images and
  their resized bytes in `processor_more.safetensors`. No GPU.
- `glm5_export.py`: GLM-5.3-Flash's first N layers and MTP layer from TensorFold 0.6.6 (commit cb2ebf0, an
  archive), both ranks in one process over a stand-in `comm`. `glm5_drafter_export.py` imports it.
- `glm5_drafter_export.py`: GLM's DFlash2 drafter alone from the same TensorFold, on made taps with BF16
  faces; activations only, under `~/.cache/zspark/`, and no file of the drafter's weights.
- `vllm_reference.sh`, `vllm_reference.py`: a checkpoint no TensorFold reads, against the vLLM container its card
  names (spec 105): the container on 127.0.0.1 with one sequence and no MTP, greedy replies and top-20 logprobs of
  `generate cases=` lines and the score text, twice for vLLM's spread; `compare` holds a `generate` output to them.
  `vllm_reference_ranks.sh`: the same at `--tensor-parallel-size 2`, a node a Spark (GLM FP4-Local, spec 107),
  holding both machines' locks whole.
- `template_cases.py`, `message_cases.py`, `tokenizer_cases.py`: `zig/src/text/`'s fixtures; no GPU. `zig/src/text/AGENTS.md` says how they are used.
- `image_cases.py`: `zig/src/image/`'s fixture, `zig/tests/fixtures/image/cases.json`: pictures made there and
  TensorFold 0.6.5's `_decode` over Pillow 12.3.0, in TensorFold's container (its docstring); no GPU.
- `unlimited_ocr/resize_cases.py`: Pillow's bicubic resize and `ImageOps.pad` on made pictures
  (`zig/tests/fixtures/image/resize_cases.json`); no model or GPU. The v1 preprocessing fixture and its
  exporter were removed in spec 170; OCR2's host checks use its page fixtures and request behavior.
- `sampling_cases.py`, `requantize_cases.py` (on the device, never the CPU) (no GPU): `zig/tests/fixtures/*.bin`.
- `csf_blocks_cases.py`: `zig/tests/fixtures/csf_blocks_cases.bin`, csf_32x2_v1 planes encoded and decoded by
  flashinfer's b12x loader in the card's vLLM container (`--network none`, its docstring), and the SHA-256 of six
  of GLM FP4-Local's planes it decoded (spec 107); no GPU.
- `grammar_masks.zig`, `grammar_masks.cc`, `grammar_masks.sh`: xgrammar's masks, the oracle of
  `zig/src/text/json_schema.zig`, into `zig/tests/fixtures/text/grammar_masks*.txt` (spec 017); no GPU.
- `openai_exchange.py`, `responses_exchange.py`, `anthropic_exchange.py`: checks by hand, not fixtures: the
  clients' own packages against a running `zspark serve`.
- Measurements, not fixtures:
  - `client_bench.py`, `client_bench.json`, `client_bench.sh`, `required_call_bench.json`: what a client sees
    of `zspark serve` and of TensorFold's server, one load a run (specs 024, 030, 034). With `STREAMS`, the
    clients' prompts are `STREAM_PROMPTS`: `prose` (unset), `chains`, four lists whose draft chains run
    long, or `code`, four short functions no draft rule was fitted on (spec 077).
  - `glm5_client_bench.sh`, `glm5_serve_check.py`: the same harness on GLM over two ranks (spec 068), zspark's or
    TensorFold 0.6.6's, both machines held alone, 2,051 tokens of context; the check asks a served load for a
    stream, a tool call, a stop string and a hang-up.
  - `shared_acceptance.sh`: shared rounds as a client sees them; it judges nothing (spec 051).
  - `pool_states.sh`, `pool_states.py`: what a conversation holds in the session and in the pool, and the
    seconds to a first token, for each way a request follows another; one load (spec 054).
  - `spill_states.sh`, `spill_states.py`: what a conversation that left the pool costs its next prompt with
    `--spill-gib` and without, after the pool let it go and after a restart; a load a call (spec 074).
  - `fill_gaps.sh`, `fill_gaps.py`: the waits between the deltas of replies that stream while a long prompt is
    filled beside them in the pool, and that prompt's first token; a load a call (spec 074).
  - `readme_table.sh`, `readme_table.py`: README's first measured table from one `generate` load (spec 049).
  - `serve_lib.sh`: what those scripts share, sourced: the locks' re-exec, the check that the GPU is
    free, the refusal of the LAN's endpoint, `zspark serve` started, waited for and stopped (spec 053).
  - `draft_policy.py`: rules for ending a draft chain, played over recorded chains (specs 028, 037, 061:
    `depth`, `split`; 067: `glm`, both GLM drafters' from `generate ... chains`).
  - `pool_policy.py`: the pool's rule (`mtp_rule.Paying` with its round's price) played over chains recorded at
    every position, S replies sharing rounds; its counts are a load's (spec 078).
  - `lands_cases.py`, `lands_fit.py`: a checkpoint's calibration of a draft landing (`mtp_rule.Paying.Landing`,
    spec 085): the cases, a fitting and a held-out set, and the seven numbers fitted on `generate ... chains`'
    lines, with the rule that reads them played through `pool_policy.py`, and refitted while serving (vLLM's);
    `costs`: a storage's `Paying.Costs` row and level from drafted cases. Standard library, no GPU.
  - `device_memory.py`: zspark's device memory by kind (spec 030).
  - `turns_part.py`: where a later request parts from what `zspark serve` holds (specs 030, 035, 036).
  - `long_conversation.py`: a conversation past 100,000 tokens over zspark's own files, compacted by the host,
    then its turns' prompts and its whole text as `generate` cases and scored ids, a storage a load (spec 100).
