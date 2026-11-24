# `zig/src/` reference

Read when a task touches what a section names; the `AGENTS.md` beside this file binds every task here.

## The map

A file's header (`//!`) says what it holds; this is the map.

- `main.zig`: usage, the table of models, `tokens`. `decode/dispatch.zig`: a command
  goes to the model that `config.json`'s `model_type` names; a model registers the commands it has
  (`models/qwen4/commands.zig`). `cli.zig`: what commands share that names no model: `key=value` words, token
  ids, the `--flag` table (`parse`; `read` is `parse` without the log, for a test), a checkpoint opened in
  order (`Opened`).
- `decode/draft.zig`: drafted decoding over any model: `Decoder(M)` and `Chooser(M)`, generic at compile time over a
  model's namespace (the contract is `Decoder`'s doc comment), `Drafter`, `Chain`. Its tests run it on the
  host over a model without a device. `copied.zig`: a `Drafter` around a model's that copies from the context;
  `recall.zig`: `serve --recall`'s record of earlier requests, which `copied.zig` asks too.
- `cuda/cuda.zig`: the CUDA runtime bindings, `Buffer`, `View`, `DType`, the shared launchers' `extern` decls.
  `block.zig`: where `cuda.cut` places the parts of one allocation (host only).
- `weights/`: `safetensors.zig`: memory-mapped shards and checkpoint directories. `drive.zig`: their drive kept
  out of its deep idle state while mapped rows are read (host only; spec 089). `numeric.zig`: bf16 and
  e4m3 on the host. `bf16.zig`, `nvfp4.zig`, `experts.zig` (with `experts_pack.zig`, its staging and packing): linears and routed experts, in their kernels' order;
  `csf.zig`: NVFP4 block scales stored csf-compressed, decoded
  to their e4m3 bytes at load (host only; spec 105). `face.zig`:
  linears of one input as one projection. `requantize.zig`: FP8 or BF16 experts made NVFP4 (drafts); `linear.zig`: a linear as stored or so made.
  `share.zig`: a rank's half of a tensor's rows or columns (GLM and Flash Next).
- `decode/sampling.zig`: keyed sampling on the host. `masks.zig`: a reply's constraint. `guard.zig`: the loop guard and thinking budget, a constraint. `confidence.zig`: a draft's probability.
  `paying.zig`: the clock-free EWMA arithmetic used by Qwen MTP; costs, trust and confidence decisions stay family-local.
  `marks.zig`: which held state goes; `shelf.zig`: the states held and the buffers of those dropped (both models' sessions, spec 112).
  Two ranks alike (spec 136 WS-05): `settle.zig`, a row of logits from two halves and a window's greedy rows settled on the device; `graphs.zig`, a forward over the ranks as captured graphs; `dflash2/follow.zig`, rank 1's drafter operations.
  `decode/dflash2/`: DFlash2's shared mechanisms (spec 122): config schema, layout, weights, selector/chain, SWA ring wrap, and the kernel wrappers over `zig/kernels/common/dflash2*.cu`. GLM imports them; Qwen's capture of taps stays in `qwen4/forward.zig`.
- `decode/extents.zig`, `prefix.zig`: the attention caches' extents and the prefix store over them, for
  `serve --parallel` above 1 once a second stream is live (018, 051). A harness's end
  (`Mark.shared`) goes last; `learned.zig`: states on disk (064); `spill.zig`: which that left go there (074);
  `spill_writer.zig`: the thread that writes them, and the host memory they wait in (no device call on it);
  `learned_parts.zig`: a model's state as a file's parts, copied out and read back, and the stores opened (both models).
- `models/qwen4/`: Flash Next: its forward, decoder, session and pool, and `served.zig`, the engine TensorFold's host
  serves. `models/glm5/`: GLM-5.3-Flash on two ranks, its two drafters and its pool (a model's `AGENTS.md`).
- `ranks/`: two processes over NCCL, one a Spark. `check.zig`: what every model's `ranks-check` does between its runs and at its end.
- `text/`, `server/`: host only, each with its `AGENTS.md`: the grammar, the schema matcher and the call gate behind a
  constrained reply, a request's messages to a prompt; the scheduler of several replies and the compiler of a
  constraint. `image/`: pictures to RGB, host only.
- `router/`: the two whole copies' one address (`zspark-router`), host only: it imports no other layer and names no
  checkpoint.
- `stack/`: `zspark stack` (spec 139), host only: `plan.zig` the children a command line gives (OCR, ASR, Flash
  Next's ranks, the router) in their waves and each one's MPS variables, `supervisor.zig` their start, health,
  lines, signals and stop, `command.zig` the binaries, the link's NCCL settings, the MPS daemons (started before
  the children, quit after them; spec 139 WS-06) and the memory left between waves.

## Counting device memory

- `cuda.account` counts the bytes and buffers alive; `zspark serve` logs them once loaded. With
  `ZSPARK_ALLOCATIONS` set, each buffer made or freed writes a line with its call path, which
  `tools/reference/device_memory.py` sums by kind. `nvidia-smi`'s figure for the process is those bytes, the
  CUDA context (170 MiB) and the graphs; pinned host memory and the host's own are not in it.

## What the areas miss

What the areas (the retired quick tier's tests) cannot catch, tried by mutation: a layer past the fourth reading another layer's tensors
(the full tier's blocks fail), and a rounding left out of the router's weights (every block stays within its
tolerance in both tiers, and only a sampled reply of the full tier parts early).
