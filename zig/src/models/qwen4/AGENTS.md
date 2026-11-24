# `src/models/qwen4/` instructions

Extends `src/AGENTS.md` and the root `AGENTS.md`. Qwen3.8 Flash Next (`model_type` `qwen4_exp` or
`qwen3_8_flash_next`) on its FP4-Local checkpoint: NVFP4 experts, MXFP8 linears beside BF16 (spec 105). Its
kernels are `kernels/qwen4/`, C symbols `zs_qwen4_*` (spec 040).

Its forms follow a window's rows (spec 143 WS-06, `decode/forms.zig` as GLM's): a window of more than
`forward.bf16_rows` (80) is a prompt's, its MXFP8 linears on FP8 rows as b12x's race chose each at 4,096 rows
(`Layer.quantizePrompts`, `qmm_a8.cu`; DeltaNet's b and a alone keep bf16 rows, spec 143 WS-08) and its routed
experts W4A4 (each route stored bf16 before the sum, as the card's FC2); steps, verify windows and shorter prompt
windows run bf16 rows, one form. `compute`,
`ranks_forward` and `pool/rounds.zig` (`formFrom`) set it; `model.prompt_form` is `card` in the commands, `bf16`
in a test that does not set it. A test holding a window of more than 80 rows to its serial steps raises the bound.
On two ranks a window or a pool's round of `ranks_forward.split_rows` (256) or more splits its glue by rows from
the last PLE (spec 106, 149 WS-02): one rank's part is `ranks_forward.Glued`, run by the pair's `splitLayer` and
by `rounds.computeOver` (`Parts.glued`), never copied; both ranks decide from what they share, and only `mixed`
and `h` are whole after it. A test lowers `split_rows`; `ranks-check` `streams` and `hall` run at 32.

## Purpose

A file's `//!` header says what it does; tests: `NAME_test.zig` under `zig/tests/models/qwen4/`.
A test's name starts with its file's: `models.qwen4.blocks.ple_test.`.

The file map, the model's shape (48 layers, four streams, the pending branch) and the TensorFold file behind each
file here: `REFERENCE.md`.

`commands/serve.zig` owns the one-rank startup and the shared command options; `commands/serve_ranks.zig`
owns the two-rank load and join, including its pool. Both retain their model resources and borrow the shared
text-loading operations from `server/command.zig`; tokenization and device load order stay explicit.

## Local conventions

Checkpoints and loading:

- Names follow the checkpoint and TensorFold (`mixed`, `pss`, `inject`), spelled out where TensorFold
  abbreviates (`dims` for `D`, `streams` for `S`, `low` for the hyper-connection's rank).
- A module's tensors are spread over several shards: resolve by name through the checkpoint, never by shard.
- RMSNorm weights are stored centred (gamma - 1). `layout.Vector.gamma` marks those that take
  + 1 at upload when `Layout.norms_centred`; DeltaNet's `norm`, `A_log` and `dt_bias` are used as stored.
- The MTP experts' `weight_scale_inv` and the n-gram table's `weight_scale` are BF16: widen at upload.
- Published (FP4-Local, spec 105): every linear but the routed experts MXFP8, the experts NVFP4.
- A mixed-precision checkpoint (`quant_algo` MIXED_PRECISION, spec 105): `Scheme.mxfp8` (e4m3, a ue8m0 byte a
  32 inputs) at `Site.dense`, a face's part (`nvfp4.Mx8Linear`, the lane matmul's mode 2 for rows and windows,
  bf16 activations); NVFP4 scales `csf`-encoded (`Linear.csf`, `weights/csf.zig`) are decoded in `loadMoe`, a
  layer at a time over the host's cores (`blockScalesOf`), a rank its window of them (`Csf.window`), and `ranges` holds only the streams. The n-gram
  table may be NVFP4 (`ple.Table` `.nvfp4`: code x block x `weight_scale_2`, one bf16 rounding). Shards may lie under `tensors/` (`Checkpoint.indexedShards`).
- Block FP8 runs on its stored bytes, never dequantized. In a `face.Face` (`model.faceOf`) a part's 128-row
  blocks count from its own first row, not the stack's.
- The MTP layer's experts are requantized to NVFP4 (`requantize.zig`, TensorFold's codes); they and
  `model.DraftHead` only draft. `Options.drafts_stored` runs both as stored.
- The n-gram table (51 GB) stays mapped and out of `Checkpoint.hold`, always. Its rows are advised through the
  shard's descriptor, never `madvise` where a file is known, and a wide gather runs in parts on threads
  (`ple.Table.gather`, spec 149 WS-05): one path for every caller.
- Nothing at load may read through a mapping what `Checkpoint.hold` later reads directly: `hold` drops the page
  cache behind it.

Exactness:

- A prompt row must be the serial step's bits, so a window's width and its kernel variant are chosen by time
  and never change a bit. Prompts run in windows of `session.promptRows` (2048; spec 031) in the
  pool's rounds, of `session.lonePromptRows` (4096; spec 148 WS-02) in one rank's lone stream and a pair on two
  ranks (spec 149 WS-04); `--max-num-batched-tokens` 2048 to 4096 sets either (spec 105 WS-07), and the pool's rounds on two ranks alone
  (`common.poolRows`, spec 149 WS-06: one Spark's pool keeps 2,048 beside its lone session);
  from `gdn.wide_rows` (32) rows the chain is `kernels/qwen4/gdn_wide.cu`, and the experts have wide kernels
  by row count too (spec 032). A new variant comes with a test holding its outputs to the kernel it replaces.
- TensorFold's prompt path (`gdn_prefill.cu`, its prefill GEMMs) has other bits than its decode path: take its
  layout, never its arithmetic, and compare greedy tokens with `greedy_decode_path`.
- The sparse selection's cut often lands among blocks that score exactly zero, so its tie-break to the
  lower block id decides real rows. Do not reorder it.
- A `compare.Bounds` value is what was measured. Lower one only with the row's numbers in its doc comment, and
  with `zspark compare ... free perturb=3e-4 seeds=8` showing that the row changes its token under noise.
- Drafts choose nothing: `mtp.zig` is held to TensorFold by tolerance, and a draft rule (`mtp_rule.Paying`,
  spec 061; `Head.chain` without costs) changes the rows a round verifies, never a token. In the pool a
  draft is judged against the round its seats share (`Paying.Round`, spec 077: the seats' two sums and the
  round's rows cross seats, nothing else), for a storage with a round of several seats in its costs (FP4-Local prices each seat alone). A seat alone is priced as spec 061's
  lone stream, to the bit (`mtp_paying_test.zig`); drafts beside other seats may differ between runs, tokens never.
  Where a checkpoint has a fitted calibration (`Paying.Landing`, spec 085), a
  draft pays by the square and by the calibrated product; the trust, a round's price and what a round is
  told stay the square's.
- A head pass of the pool scores its seats in one launch (`hall_head.scoreRows`, spec 078). `hall_score_fixture.zig`'s
  check, on the random model and a checkpoint, holds each seat's row and `picked` to the row scored alone; a kernel chosen by rows must pass it first.
- A reply never depends on what the session decoded before. Whatever leaves the state, the head and `known`
  out of step clears `Session.trusted`.

The forward, its graphs and buffers:

- `forward` is `stage` (the host's part: ids, the PLE's table rows; the next window's and round's are
  gathered on a thread meanwhile, `forward.expect` and `ask`, taken for the same ids alone: `ple.Ahead`) then `compute` (launches only). `Graphs` captures `compute` for each window size, DeltaNet state side and context
  bucket. Whatever else changes a launch's arguments between steps becomes a graph key or is read from device
  memory, as the position is; a host copy inside `compute` fails a capture.
- Graphs are captured at load (`Session.prepare`, spec 020). A new graph key, or a new place the step or the
  head runs from, goes into `prepare` too; `session_test.zig` holds a reply to no new graph.
- A window wider than `forward.keptRows` shares one projection and scratch among the layers, so it is committed
  whole and a state inside it is asked for before it runs (`Snapshot.ahead`, `State.wanted`); a pair asks
  both ranks (`Op.keep_ahead`).
- The pool's short projection holds the widest round with no segment of `gdn.wide_rows`: the others' windows
  and a prompt chunk one row short (`hall_open.shortRows`, spec 086). What widens such a round widens it.
- No forward between a prompt's last window's commit and `Session.settle`: it would overwrite what
  deferred replays start from. Under `State.behind` a state is `forward.committed`'s, and a pool's slot under
  `Slot.behind` `rounds.committed`'s (spec 144 WS-09).
- `Decoder.promptRows` must stay `prompt`'s own arithmetic: `Session.plan` and GLM's `session/marks.zig` place snapshots by it.
- Attention: at most 16 query heads a KV head, `head_dim` 32, 64, 128 or 256 (`attention.Shape.of`). The
  caches are `.bf16`, `.fp8` or `.fp8_nvfp4` (zspark's formats, specs 012 and 013: no TensorFold reference);
  a random-model test loops over all three (`random_model.caches`). The MTP head's cache is bf16 whatever
  `--kv-dtype` asks (spec 052), so a launcher takes its queries from `Scratch.queries(cache)`: `scratch.q`
  read directly gives a bf16 cache coded queries.
- A load's caches are one storage (`layout.storageFor`) and one live stream under `--parallel` is the session's
  code (spec 051): a reply must not change with the number of streams.
- A device buffer the pool's opening makes is in the `bytesFor` beside its `init` (`cuda.Buffer.bytesOf`),
  which `hall_open.Rest` sums for the load's check: a new one goes in both, `hall_open_test.` holds
  them to the byte.

Which test goes with a change, by `-Dtest-filter`:

- Which snapshots are taken or dropped (`plan`, `thinned`, `unused`): `session_snapshot_test.zig`, turns
  and edits (`snapshots`, `edited`, `the mark before`).
- The sequences left (`parked.Lot`: `park`, `makeRoom`, `last`, `victim`): `session_parked_test`.
- `Ends`, `keepWithin`: `reasoning ends`. `keepStride`, `Snapshot.again`: `reply's stride`
  (`session_stride_test.zig`), and `first layers`.
- `Snapshot.within`: `forward_test.zig`'s first-layers tests (a PLE).
- `decode/masks.zig`, or where `draft.zig` calls it: `constraint`.

## Local commands

Usage: `zspark`'s own text (`src/main.zig`); the tiers: the root file.

- `zspark compare DIR REF/primes.serial.f32.safetensors [layers=4]`: every block against TensorFold, each fed
  the reference's input. REF is one `.f32` file, not the directory. Run it after touching a kernel of the
  forward: each point within 2e-3 relative rms; the streams, projections, PLE output and logits exact.
  `free` is no test of closeness.
- `zspark generate DIR N [serial|eager|lazy] ID..`: a prompt and N greedy tokens, timed; `cases=FILE`
  runs many on one load.
- `zspark score DIR [KV] [each] [perturb=SIZE seeds=N] ID..`: mean nll, how a change of arithmetic is judged.
  A run of the 5336 tokens lies 0.0035 from another (spec 100): judge by seeds; a long text by blocks of
  thousands of tokens.

DFlash2 taps (spec 122): `taps.zig` is the host mean (default GLM f32-acc; `Mean.bf16` rounds after each add).
`Buffers.setTaps` reserves the five slots at session setup, never on the decode path. `compute`, `Held.launch`
and the pool's rounds (`rounds.computeOver`, one rank or a share; spec 126 WS-04, WS-05) write them through
`forward.capture`, after an explicit MoE writeback of a tap layer, not through `hc.site`: no bit of h moves. Two
ranks in one process (`Held(2)`, `DrafterOf(2).initHeld`) is the NCCL reference: drafted==serial;
rank 1 follows `Op.flash_*` and proposes nothing.
Stored DFlash2 heads borrow the target's BF16 weight spans and pool scratch; only four-bit proposer copies enter the rig's owned-head cleanup.
`serve --drafter` (spec 126): the session's heap rig migrates into the pool (`Session.takeFlash`,
`Core.migrate`; `Engine.seatDrafter` over `SeatsOf(2)`, buffers tapped in `openAfter`), `flashRound` after
`keepRound`, a seat's context cleared in `resetSlot` at every sequence start and fork. On two ranks one
stream is `ranks_drafter.Given(1)`, eager; above `--parallel 1` each process holds a `rig.Pool` (`Rig(1)` and
`SeatsOf(1)` on its share, bound by `seatRankDrafter` after the pool's fit), rank 0 alone proposing and
`Op.flash_round` told after its commit, checked against it on rank 1 (`Link.committed_parts`, spec 162).

The mean rule (spec 125 WS-01, settled from PyTorch's source and one op on the GPU): `mean` on a bf16 CUDA
tensor accumulates in float and rounds once to bf16. `mean_kernel_cuda` calls `mean_kernel_impl<at::BFloat16,
float>` (pytorch v2.13.0, `aten/src/ATen/native/cuda/ReduceMomentKernel.cu:53-59`), and `MeanOps` adds in
`acc_t` and multiplies by `factor` at the projection (`aten/src/ATen/native/SharedReduceOps.h:151-165`). One
op on inputs built to separate the two modes, in the card's vLLM image (`eugr/spark-vllm-b12x`, torch
2.13.0+cu130; receipt `~/.cache/zspark/scratch-125/mean_one_op.stdout`), printed the op's bits equal to
`Mean.f32_acc` on all outputs and on 32 inputs where the modes differ, never `Mean.bf16`. `Mean.f32_acc`, the
default `forward_state.Buffers.tap_mean` already holds, follows it.

## When editing here

- Do: give each block its own file with its wrappers and tests, compared with TensorFold at its boundary.
- Do: record a departure from TensorFold in the code and the spec's handoff.
- Do: keep a constrained reply the serial one: a row's mask is made from the tokens before that row
  (`testing.zig`'s `serial.serialConstrained` is the reference).
- Don't: let a reply without a constraint pay for one: no matcher call, no mask buffer touched.
- What a rank holds of the model: `REFERENCE.md`, "Two ranks".
- Don't: add MLX 4-bit branches here. Vision is spec 019's (lifted for its WS-03, 2026-10-17): an image prompt's
  rows enter as `forward_state.Image` over the embedding and rotate by `attention.Rope`; the tower is `vision.zig`.
