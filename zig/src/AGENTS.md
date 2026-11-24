# `zig/src/` instructions

This file extends the root `AGENTS.md`. The Zig program: CLI, checkpoint reading, device buffers, the host side of
every kernel, and the models (`models/`).

## Purpose

A file's header (`//!`) says what it holds; the map of `zig/src/` is `REFERENCE.md`, "The map".

- `models/`: a directory a model (`models/AGENTS.md`: its names, what it gives `dispatch.zig` and `draft.zig`,
  what still waits). What two models share lies in `cuda/`, `weights/` and `decode/`. Nothing else in `zig/src/` imports a model's
  directory but `main.zig` and the tests' roots.

Which kernel a call takes is chosen by its row count and never changes a bit: `experts.Pass` (items of 32 and
`experts_wide.cu` from `Pass.wide_rows`, 768; of 64 and `experts_tile.cu` from `tile_rows`, 1024). `Scratch.launch`
and `cuda.Slices.folded = false` (a split matmul's slices summed by the next kernel, spec 028) force one path,
for the tests that hold the paths against each other; nothing else sets them.

## Zig 0.17

- Read the standard library's source (`zig env` gives the lib directory) instead of recalling an API: `std.Io`
  carries files, directories and clocks; `main` takes `std.process.Init`; tests use `std.testing.io` and
  `std.testing.tmpDir`.
- `**` is gone: use `@splat`. `@typeInfo(T).@"struct".fields` is gone: use `std.meta.fieldNames`.
- A function parameter may not shadow a file-level declaration, a test constant included. Keep fixture constants
  in a namespace (`const fx = struct { ... };`).
- A file under `/proc` says its size is 0: `Dir.readFileAlloc` returns nothing for it and no error. Read it
  with `file.readerStreaming` (`cuda.systemAvailable`; spec 082).

## Device buffers and views

- `cuda.Buffer` owns device memory; `cuda.View` is what a launcher reads. A wrapper takes views, never raw
  pointers, and turns the launcher's status into an error with `cuda.launched`.
  A pool's kept state and its pooled keys are each one allocation (`cuda.cut`, `cuda.Parts`; `Snapshot.block`),
  its parts views freed with it alone: 86 `cudaMalloc` a state were 5.5 ms with the device idle (spec 146 WS-04).
- Weights are uploaded once from the checkpoint's mapping. A format whose kernel reads the stored order (BF16)
  uploads the mapped bytes as they are; the others pack on the host first.
- A loader holds the tensors it is about to read (`Checkpoint.hold` over `layout.ranges`) and releases them
  after. A hold reads the file with direct I/O into memory laid over the mapping, so the page cache stays empty:
  on the Spark it shares memory with the device, and a cached checkpoint slows every device allocation.
- Memory for packing that every layer needs (`experts.Staging`) is made once and passed down, not per layer.
- No allocation on the decode path: scratch is made at init for the most rows a call takes. A linear's (a split
  matmul's K slices, rows rounded to fp16 or rotated, a face's staging) is lent by its model's pool
  (`cuda.Scratch`, spec 033): one buffer for each role, element type and shape, and a part by capacity, the
  smallest buffer of that type whose bytes cover the request (spec 047), whatever the number of linears that
  ask, never moved or freed before the model (a captured graph keeps its address). Each linear owning its own
  was 5 to 6 GiB at 512 rows. That stands on one rule: what a linear leaves in scratch is dead once the kernel
  that reads it has run, which is before the next linear's first launch (`slices` hands its sums to the kernel
  after it; a face's fp16 rows are written by the kernel before it and read after its trellis runs, hence the
  role in the key, and why halves and rotated rows stay keyed on the exact shape). A kernel reads a part by
  its pointer and its element count, so a larger part serves a smaller shape. A new linear takes its scratch
  from `rows.pool.lend`, calls `scratch.enter(x)` before its first launch, and keeps as its own whatever is
  read later or must hold a value between calls. `cuda.Scratch.poisoned`, or
  `ZSPARK_POISON_SCRATCH=1` for a whole test run, fills the pool with NaN patterns at every `enter`, and the
  tiers must pass with it; `cuda.Scratch.shared` off gives each linear its own, to compare bits. Other types
  own their scratch; one `attention.Scratch` serves every attention layer and the MTP head's
  (`forward.Buffers.initBeside`), poisoned before each block in the same mode. Its selection `dots` holds the
  rows a captured round asks for (`gdn.wide_rows`), and a wider window scores that many at a time.
- Counting what is alive on the device (`cuda.account`, `ZSPARK_ALLOCATIONS`, `nvidia-smi`'s figure):
  `REFERENCE.md`. On the Spark a device buffer's addresses lie among the host's mappings, and once freed they
  can be the host's next: a write through a stale pointer (a view kept, a graph replayed) then lands in host
  memory with no CUDA error. `ZSPARK_QUARANTINE` holds freed buffers back and reads them again.
- A copy from the device into host memory is `Buffer.download`, `downloadAt` or `cuda.download`, which land in
  `cuda.staging`'s page-locked memory and copy on (spec 132), or `downloadLaunched` into a `cuda.HostBuffer`:
  never a bare `cudaMemcpy` into pageable memory. On the Spark one sometimes misses its pages, with no error,
  and writes others of the process (a test's buffer kept the allocator's 0xAA, its metadata was changed).
- `cuda.Stream` and `cuda.Graph` capture launches and replay them. Kernels launch on the default stream except
  inside a capture; a launcher must take its stream from `zs::stream()` and never wait or copy to the host.
  A capture records and runs nothing, so a graph can be captured before its buffers hold anything.
- Launches are asynchronous. Call `cuda.synchronize()` before reading results back, which also surfaces a
  kernel's failure.

## Tests

A file's tests are `zig/tests/` at its path, in `NAME_test.zig` (`zig/tests/text/template_test.zig` for
`zig/src/text/template.zig`; several where one would pass 50,000 bytes, named for what they hold). No source
holds a test and no test imports another: a test imports its file by path (`../../src/text/template.zig`) and
names what it reads (`const Head = mtp_file.Head;`), under an alias a test's local does not shadow; a name a
test reaches becomes `pub`. What two test files share has one home under `zig/tests/`, the area's `testing.zig`
or a `NAME_fixture.zig`. Only what the program itself reads stays in `zig/src/` (`glm5/checks/`).
The roots stand in `zig/`, above both trees, since a module reaches only what is under its root. Register a
new file, a test file too, in its area's root, in the `comptime` block (a `test` block would count as a test):
`text_tests.zig`, `server_tests.zig`, `image_tests.zig`, `audio_tests.zig`, `weights_tests.zig`,
`qwen4_tests.zig`, `glm5_tests.zig`, `unlimited_ocr_tests.zig`, `qwen3_asr_tests.zig` or `nemotron_tests.zig`,
which `zig build test-text` and the others run alone and `tests.zig` imports for the full tier. A shared file
outside those whose imports reach no `cuda.zig` goes in `host_tests.zig`'s `comptime` block alone: the host
tier's root, linked without the kernels, imported by `tests.zig` beside the areas' so the shared inventory is
listed once (a file reached twice is compiled once, its tests run once). A shared file that does reach
`cuda.zig` goes in `tests.zig`'s own `test` block; a binary holds the tests of every file its root reaches.

Two shares of Flash Next compared in one process (loopback) are the exact reference for those shares over NCCL
(`models/qwen4/checks/ranks_check.zig`). A window's rows are that reference's serial steps. Several streams in
one round (`ranks-check ... streams`, every layer kind, each cache storage) each equal that stream alone.

The full tier and the areas (`specs/WORKER.md` says which a change needs) differ in one thing: an area's binary
is compiled with `@import("tier").whole` false, and a test that loads a whole checkpoint returns
`error.SkipZigTest` when it is; `test-whole` runs those alone, by the names in `tools/test_areas.tsv` (spec
133), so host, areas and whole hold the full tier's tests. A new test on a real checkpoint loads its first layers
(`Model.loadWith` with `.layers`), is gated on its `ZSPARK_*` variable alone and so runs in its area; only
what cannot be shown on a few layers is gated on the tier as well, and its name goes into the inventory's `whole` row (`check.sh sets` checks it, and refuses a gate written outside a test's body). A check on Flash Next's first layers up to
the first attention layer is a fixture function run by `checkpoint_prefix_test.zig` on its cache storage's one
load, at the rows it was written for (`checkpoints.narrowed`), not a load of its own (spec 133). Do not put a tier or a path into the run
step's environment in `build.zig`: a configured build is cached with the environment it was configured in.

What the areas cannot catch, found by mutation: `REFERENCE.md`.

A `failed command:` line beside a passing summary is not a failure: a test wrote to stderr (`std.debug.print`),
and Zig 0.17's build runner shows a run step's stderr through its error printer, which adds the command a step
with `has_side_effects` (ours all have it) never clears on success (`compiler/Maker/Step/Run.zig`); the
summary and the exit code are what say whether the tier passed.

Every kernel wrapper has both of:

- **Against the host.** The same computation in Zig on random inputs: fp64 for a sum, with a tolerance stated as
  a multiple of the summed magnitudes; fp32 with the model's roundings for glue, within a bf16 step.
- **A row alone against the row among others**, bit for bit (`expectEqualSlices` on the raw bits), across the row
  counts that select different tiles, slices or plans. `linears.checkLinear` does both for anything with
  `forward` and `prefill`.

Those two are what a model's own tests check at the shapes it serves. The properties inside one call — NaN
class, repeated bytes, guard rows, tile edges, an fp32 sum, the launch's registers and shared memory — are the
kernel contract's (`zig/tests/cuda/contract.zig`; `docs/kernels.md`, "The kernel contract"): a kernel that sums
or masks registers in the change that adds it, and `tools/check.sh weights contract` runs it.

A bug fix comes with a test that fails without the fix: restore the bug once, see the test fail, and say so in
the spec's handoff. A selection or anything with ties gets inputs built to tie, and a rounding that a later
cast hides on random data gets inputs searched for until it shows (`ple.zig`'s rows whose last add lies an
fp32 step from the middle of two fp16 values, spec 042).

A chain or recurrence adds a third: a window equals its serial steps, and a replayed prefix equals the serial
state, bit for bit. GLM FP4-Local's windows of more than `forward.bf16_rows` take the quantized forms (spec 115):
there a window of bf16 rows equals its serial steps, and a quantized window's row equals the same row in a window
of another size (`pack_test.zig` lowers the bound and restores it).

## When editing here

- Do: name a packed layout's shape in the doc comment of the field that holds it.
- Do: refuse an unsupported checkpoint with a message that says what was found and what is read.
- Don't: call a launcher without a wrapper, or keep a view past its buffer's `deinit`.
