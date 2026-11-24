# `zig/kernels/` instructions

Extends the root `AGENTS.md`. CUDA sources compiled by nvcc for `sm_121a`, linked into the Zig
executable; `build.zig` lists each with its flags. A source's header says what it computes, where it comes
from and why it is hand-written: read it first. `docs/kernels.md` has a row an entry.

Each directory's sources, which are TensorFold's unchanged and which zspark's: `REFERENCE.md`; each entry point:
`docs/kernels.md`. `glm5/AGENTS.md` has GLM's kernels' own rules.

What an edit here must keep:

- A kernel is placed by its layer and its precision, never by a checkpoint (the owner, 2026-10-17): a linear or
  an expert GEMM by the format it reads (`bf16/`, `nvfp4/`, a new format's own directory such as
  `mxfp8/`); a layer's kernel (attention, DeltaNet, MoE routing, norms) under the family it serves, named by the
  layer and, where it has variants, the precision (`attention.cu`, `mla_q8.cu`). No directory, file or
  entry point names a checkpoint or a recipe (`fp4_local`, `nvidia_*`), so that the next checkpoint of a format
  shares it.
- A model's own layers go in a directory under TensorFold's family name (`qwen4/`, `glm5/`), with no
  `models/` level. A kernel two families launch belongs in a shared directory (`common/` or its format's).
- `gemm.cu`'s K split is zspark's (`bf16.zig`'s `splitK`), not TensorFold's `f16_split`.
- A GEMM called without `out` leaves its K slices unsummed in `part`. A kernel that reads them
  (`zs_qwen4_hc_act`, `zs_qwen4_moe_topk`) sums them through `b16_slices` (`bf16/reduce.cuh`, `b16_reduce`'s
  sum), in slice order, never in its own. A test holds the slices' shortcut against the launches it replaced.
- `qwen4/hc.cu`: the write-back takes a null shared buffer when the shared expert is y's last slot. A
  thread moves four values as 8 bytes, and a block's 256 squares still fold halves onto halves (`hc.zig`).
- `qwen4/gdn_wide.cu` keeps the bits of `gdn.cu`'s step. TensorFold's prefill chain contracts to FMA and sums
  in strides of four, so its prompts' states differ from its steps'; zspark does not take that.
- A count or a selection (`mtp.cu`'s pick, `experts_pairs.cu`, `moe.cu`) leaves nothing that depends on the
  order the threads ran in.
- The grouped experts' grid is a bound, not a count: a graph holds its launches' grids, and the member
  tiles' number is known on the device alone.

## Local commands

- One file while iterating: `nvcc -c -O3 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -o /tmp/x.o FILE.cu`
  with its flags from `build.zig`.
- A new source is built once it is in `kernels` in `build.zig`.
- Under `compute-sanitizer`, and Nsight Compute on the GB10: `REFERENCE.md`.

## Mega kernels, and where a round goes

The owner, 2026-10-10: fuse actively. A layer's chain of launches becomes one kernel wherever that
shortens a round: its weights read once, its intermediates in registers and shared memory, no start and
tail between its steps. Qwen3.8 Flash Next first, GLM after.

- Find the bottleneck first. A mega kernel is opened from a round's trace (by kernel, the gaps between nodes,
  each kernel against the GB10's bounds), retaken at the current commit. Spec 038
  Record the trace configuration and report the full round.
- 038 and 070: joining two kernels saves under 2% (a replayed graph pays no launch); the large kernels are
  their weights' bytes at the memory's rate. Time a kernel by its own duration, in a round.
- A new file beside the kernels it replaces, which stay. The row rule holds; their bits need not (the owner,
  2026-10-10): its error against them is measured.
- Its header lists what its shared memory holds and why none can be less (`cuda-gb10`).
- What is closed until a new measurement: `docs/kernels.md`, "Closed until a new measurement".

## Exactness in device code

- Fixed reduction order. Sums over K, over streams, over slots or over a block run in an order that depends on the
  shapes alone. A shared-memory tree over a fixed thread count is fine; an order that depends on the grid is not.
- A rank's partial is fp32. Two ranks add them in rank order, rank 0 first, and round once (`qwen4/partial.cu`).
  A world of one does not call that kernel: its path is unchanged.
- K slices come from the weight's shape, never from the row count. A row tile or block shape may change with the
  row count only if it cannot change an element's arithmetic.
- Keep every bf16 rounding the model has, where it has it (`bf(x)` helpers). Do not fold two roundings into one.
  Where the model's rounding is, is settled by TensorFold's exported tensors, not by its Triton source alone: the
  compiled write-back (`qwen4/hc.cu`) rounds `h + branch * inject` once where `glue.py` writes two (spec 014).
  A point near but not exact against the reference is replayed on the host both ways before it is accepted.
- A kernel whose replay must reproduce its serial steps is compiled with `--fmad=false` (DeltaNet). Elsewhere nvcc
  contracts to FMA, which is deterministic and allowed.
- No library call whose algorithm changes with the row count (cuBLAS GEMM) on a path that defines a row's bits.
- A kernel's form (A16 or A4 experts, bf16 or FP8 rows) is chosen by its caller from the window, never inside a
  kernel by its row count: GLM FP4-Local's window of more than 48 rows takes the quantized forms (spec 115), and
  each form keeps a row's bits its own (`nvfp4/experts_a4.cu`'s two kernels, `qmm_a8.cu` beside the lane matmul).
- A quotient that serves as a scale is read at fp32's ends before it is trusted: `448 / most` is infinite for
  every `most` under 1.3e-36, not only for zero, and zero times it is NaN (spec 043): guarding the divisor
  against zero does not cover it. Such a kernel's test holds rows made by hand for each end of a range (a tile
  far under the largest score, a head of zeros, keys past a row's list, a weight that underflows), and on
  random rows asks for every value finite.

## CUTLASS

Prefer CUTLASS to hand-written MMA code wherever one of its kernels can meet the rules above. A hand-written
kernel says in its header why CUTLASS does not serve there (a storage layout it does not read, a recurrence,
a selection).

- Which CUTLASS kernels the GB10 takes: the `cuda-gb10` skill, "CUTLASS".
- The SM120 kernels take FP4 or FP8 on both sides, in M tiles of 128 contiguous rows. They do not multiply
  bf16 rows by NVFP4 weights, which is this repository's exact arithmetic (`nvfp4/qmmf.cu`,
  `nvfp4/experts.cu`), nor serve rows that each gather their own keys (attend's FP8 mma is by hand).
- Fix a CUTLASS kernel's K partitioning by the weight's shape, and put it through the same row-alone test as
  any other. Its parallel split-K writes partials in the output type: keep slices fp32 and sum them with
  `bf16/reduce.cuh`.
- CUTLASS is a Zig package (`build.zig.zon`); a kernel that uses it sets `.cutlass = true` in `build.zig`.
- Measure a kernel against the GB10's bounds before rewriting it (spec 011): a bf16 `m16n8k16` chain peaks
  near 33 TFLOPS, memory serves about 200 GB/s. Attend's shared memory a block decides an SM's blocks (012).

## The C ABI

- Entry points are `extern "C"`, named `zs_<area>_<what>` (a model's own after its directory: `zs_qwen4_*`),
  take `const zs_view*` (null for an absent optional), and return 0, or nonzero with the reason in
  `zs_last_error()`. No exception crosses.
- An entry point calls `zs::begin()`, checks every argument with `ZS_REQUIRE` (dtype, shape, contiguity,
  alignment) before anything launches, then returns `zs::status()`. Launches use `zs::stream()` and are followed
  by `ZS_CUDA_LAUNCH_CHECK()`.
- An entry more than one Zig file calls is declared in `zs/abi.h` and in `cuda.zig`'s `kernels`; one that a
  single Zig file wraps is an `extern fn` beside its wrapper. A kernel's name is its own across the tree
  (`ple_gate_kernel`, not a second `gate_kernel`): a trace has only the name.
  `zs_view` and `zs_dtype` mirror `View` and `DType` in `zig/src/cuda/cuda.zig`: change both together.

## Provenance

- A file taken or rewritten from TensorFold says so in its first lines: the source path, the commit (609ca41 for
  0.6.5) and what changed. It stays Apache-2.0.
- Do: keep a taken kernel's device code byte for byte, changes only in the host launchers at the file's end, so
  an upstream fix still applies.
- Don't: reformat a taken file, rename its symbols, or mix new kernels into it. New code goes in a new file.

## When editing here

- Do: document a packed layout (tile order, block contents) with its shape where the kernel reads it and where
  the host packs it.
- Do: take `ldmatrix2`, `ldmatrix4`, `ldmatrix2t` and `ldmatrix4t` from `common/copies.cuh`; no kernel defines
  one of its own (spec 158 WS-01).
- Do: register a kernel that sums or masks with the kernel contract in the change that adds it
  (`zig/tests/cuda/contract_test.zig`; `docs/kernels.md`, "The kernel contract"). Its adapter declares the
  shapes, the f64 formula, the magnitude of the terms, the guard rows and the launch; the launch's registers
  and shared memory are read from `cuobjdump -res-usage`, and read again when the launch or its tile changes.
- Do: carry upstream's flags over with a taken kernel (`extra_cuda_cflags` at its `load`).
- Don't: allocate in a launcher. Scratch comes from the caller as a view.
