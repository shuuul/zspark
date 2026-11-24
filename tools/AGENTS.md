# `tools/` instructions

This file extends the root `AGENTS.md`. Host-only programs the build runs, and `spark_copy.sh`, which copies
weights between the two Sparks (bash, `dd`, `nc`, `rsync` and `ssh`: nothing to build; its header is its manual),
`check.sh`, the one entry for a check or a measurement (bash, `flock`, `jq` and `nvidia-smi`; `check.sh help` prints
its verbs, the header below them has the details: read that, never the script whole). Its responsibilities are
the sourced files beside it (spec 171): `check_areas.sh` loads and validates the one test-area inventory
`tools/test_areas.tsv` — each step's name, host/GPU kind, tier and split membership, fallback seconds, model,
Zig root and `-Dtest-filter`s, which `build.zig` embeds and every step-listing verb consumes, so a new or
moved test step is one row there and nothing else — and proves `check_qwen4_files.txt`'s coverage;
`check_tier.sh` runs a step under the locks, `check_scope.sh` selects affected steps and owns pass records,
`check_schedule.sh` balances and runs those steps, `check_sets.sh` checks compiled test membership, and
`check_measure.sh` owns the checkpoint verbs. `check.sh inventory` runs the area and Qwen4 coverage checks
without a build or machine configuration. `sets` uses the same checks before listing the compiled tests. `check_qwen4_files.txt` is
`qwen4-files`' list of a quickest test a runnable `zig/tests/models/qwen4/` file, one entry a file.
`two_copies.sh`, which serves two whole copies of Flash Next behind the router's one address (spec 088;
its header too), and `two_ranks.sh`, which starts both ranks of a command, rank 1 on the other Spark over
ssh (its header too; a new two-rank command is one line of its `case`, with the lock it takes and its time
limit). The two scripts and `reference/glm5_client_bench.sh` share `link_lib.sh`, sourced: `link_env` (NCCL's
settings for the link, also shipped to rank 1's shell with `declare -f`) and `gpu_guard` (the refusal of a busy
GPU, shipped into `bash -c`, `check.sh`'s too).

`http_smoke.sh` (bash, curl and jq; its header is its manual, `docs/http-smoke.md` the
longer form) is the client's minimal check of a running `zspark serve` or the stack's
router: `/health`, `/v1/models`, a streamed chat's SSE events, a required tool call, an
unknown-model refusal, and opt-in `--asr`/`--ocr` media checks against caller files. It
starts and stops nothing, takes no lock and never loads the GPU; the key goes through a
private curl config, never argv.

`sparkdash_probe.mjs` (node, no build; spec 160) drives sparkDash's `LlmProbe` (`$SPARKDASH_DIR`, default
`/tmp/sparkDash`) against a serving engine and prints the snapshot it would show: `node tools/sparkdash_probe.mjs
[host] [port] [sleep_s]`; the optional sleep polls twice around one repeat-image request so the fields that need
an in-session delta (per-stream tok/s, cached-prefill split) fill.

`held_turn_bench.zig` (`zig build bench-held-turn`, spec 072) is the one program here that stands on the
tree: TensorFold's server from `zig/vendor/`, an engine that ends a reply at once and
`zig/src/server/segments.zig`, to time what the host does before a prompt reaches the engine. No GPU, no
weights; a checkpoint's tokenizer and template. `compile` builds it.

`cache.zig` (spec 168 WS-04; `zspark-cache`, `zig build cache -- ...` report-only, `test-cache` its
`cache_test.zig`) is the one inspector and pruner of this machine's generated caches. No argument reports
the known roots — every `~/repo/zspark*` checkout's `.zig-cache` and `zig-out`, `~/.cache/zig`,
`~/.cache/zspark`, `~/.cache/huggingface` — each child's allocated bytes (st_blocks, not the apparent
size), its newest mtime and its class. With a root and no bound it reports that root alone; nothing is
ever removed without `--apply`, and pruning needs the root named plus `--older-than DAYS`,
`--max-bytes BYTES` or both: eligible entries go oldest-first until the byte bound holds, a bound no
eligible entry can meet is reported unmet, and an entry touched in the last minute is kept regardless.
`--apply` accepts only a canonical generated root: `~/.cache/zig`, `~/.cache/zspark` and its
regenerable children (`*-cache`, `*_cache`, `upstream*`, `tensorfold-*`), or a `.zig-cache`/`zig-cache*`/
`zig-out*` whose parent is a verified zspark checkout (`build.zig.zon`, `zig/`, `tools/` present). Every
other positional root is report-only, whatever it is named — a `model-cache` under `~/models` is weights,
not a candidate. A path through a protected component (`deploy-*`, `scratch-*`, `reference-*`, `models`,
`weights`, `hub`/`xet`, `logs`, `zig-pkg*`, `.git`, locks, env, venvs, `full-passed`, serving state
`learned`/`spilled`/`recall`) refuses the root entirely; a nested `.zig-cache`/`zig-out` is a root of its
own, never an entry of its parent. Inside a zig cache `o`/`tmp`/`z` are eligible, the `b`/`p`/`h`/`c`
package stores never; unknown names are kept.
The tool holds `~/.cache/zspark/machine.lock` shared for its whole run — the hold every heavy
command owes, which waits out an exclusive timing and is never refused by a server's own shared hold —
and `--apply` takes `~/.cache/zspark/cache.lock` exclusively inside it without blocking: a build holding
`cache.lock` shared (as every `zig build` under `check.sh` does, after `machine.lock` and never across a
`gpu.lock` wait) refuses the run. `--apply` also refuses a caller anywhere below a `zig` build driver,
and re-measures each candidate and re-surveys `/proc` immediately before its removal, so an entry
touched, replaced or newly held since the plan is kept. Symlinks are never followed; open fds, mappings, executables and a cwd inside an entry, and
a `zig` process building in its checkout, keep it. Prune the installed `zspark-cache` only: `zig build
cache` report is fine, its apply refuses the driver. Tests run in temporary directories they fill and
remove: no real deletion, no shared lock taken.

Local checks use `check.sh` and `.githooks/`. Install the hooks with `git config core.hooksPath .githooks`.
`pre-commit` runs `check-specs` and `pre-push` also runs `test-specs`. `.github/workflows/specs.yml` runs the
spec checks and `test-host` for pull requests and manual runs on a CPU-only runner. `test-host` links no CUDA
kernels and needs no CUDA toolkit. A manual run or a pull request changing vendor, import or build inputs also
runs `test-golden` and `test-vendor`. Host tests that need a checkpoint skip when their `ZSPARK_*` variable is unset.

`test_runner.zig` (spec 133) is compiled into every test binary (`build.zig`'s `runner`, with the test's own
target and optimization), not built as a tool: Zig 0.17's `lib/compiler/test_runner.zig` as it ships, each change
marked `zspark:`, so a Zig upgrade copies the new upstream file and applies those again. It keeps upstream's
protocol, skips, leak and logged-error checks; it adds each test's milliseconds, written to `DIR/PID.tsv`
(milliseconds, status, name, the MiB of `MemAvailable` gained over the test: negative where it kept memory,
the device's included on GB10) where `ZSPARK_TEST_TIMES=DIR`, and `ZSPARK_TEST_LIST=1`, which runs no test and
reports each skipped with its name (a binary's tests counted with no GPU: `ZSPARK_TEST_LIST=1
ZSPARK_TEST_TIMES=DIR zig build STEP`). `check.sh` sets `ZSPARK_TEST_TIMES` to the log's `.times/` and appends
the twenty slowest to the log, then the tests that lost over 1 GiB. A run killed by a signal (zig's build
runner, or a test binary zig names and starts again past) prints `KILLED`, the signal and the last test the
runner's lines hold, never a count of steps that succeeded (spec 133 WS-04). Each test step has a `-build` twin that compiles and runs nothing: a GPU verb
compiles through it under the shared machine lock, then takes `gpu.lock` for the run, a cache hit.

`full` (spec 133 WS-06) runs host + areas + whole on both Sparks at once: the clean head pushed over the link
into the other's clone (a ref under `refs/check/`, deleted once checked out in `~/repo/zspark-check`, a worktree
made once and reused), each step whole on one Spark through the internal verb `half`, which builds its binaries
in one `zig build` and runs each step as its verb does, under that machine's locks and env.sh. The split takes
each step's seconds from the newest `TIME-full-STEP.times` (the other's copied back), else the figures in
`tools/test_areas.tsv`; `check.sh split` prints it. A new test step is a row of the inventory, its root at
`zig/STEP_tests.zig`. `--one full` is the single-Spark tier.

`full` runs only the steps a change reaches (WS-07): the paths changed since the nearest ancestor of HEAD marked
passed with the same optimization mode, `--bare` setting and `ZSPARK_*` environment. `--bare` clears all
incoming `ZSPARK_*` variables on each Spark, including those sourced from `env.sh`. Shared code (`vendor/`,
`cuda/`, `decode/`, `weights/`, `ranks/`, `main.zig`, kernels outside a model) and shared build inputs (including
`tools/host_descriptor.zig`, the runner and `tools/test_areas.tsv`) reach every step. Other Zig or embedded
files reach steps that import them transitively. `kernels/MODEL/` reaches GPU steps that import MODEL.
Check scripts and host tool test inputs reach `host` alone. `whole` accompanies checks from runtime paths.
The first line names the base and each step run or skipped. A scoped pass marks
the head; no step reached marks a clean head with no run. `--all` is the whole tier; `split [BASE [HEAD]]` the dry
choice. `cuda/` stays shared because a model's `zs_*` externs live there; moving them under `models/MODEL/` would
scope them (spec 133, a follow-up with no row).

`qwen4-files` covers each of `zig/tests/models/qwen4/`'s test files the area runs once: a test a file in about
  1 minute and 45 s of compile for the original 57-file list, where the area ran in 12 minutes
  (2026-10-17, spark-a). Spec 171 adds eight source-chosen selections, not retimed. Its list is
  `tools/check_qwen4_files.txt` (was `check.sh`'s `QWEN4_FILES`); its original entries were chosen from a run of the area's binary with each test's end stamped. A file whose tests are all
gated on `tier.whole` (`compare_vllm_test.zig`) takes no entry, since they all skip under the area's `tier.whole`
false, and a new test file adds its quickest test. A filter that names no test of its file, or matches none,
fails the verb.

## Local conventions

- A tool builds for the host with no libc, no CUDA and no dependency on `zig/src/`, so CI without a GPU or the CUDA
  toolkit can run it. `build.zig` gives each its own module and step.
- `check_specs.zig` collects every problem and prints them together. A new rule comes with a test that runs the
  validator on a tree in a temporary directory and asserts the message, in `check_specs_test.zig`.
- `check_specs.zig` also holds the size budgets (`budget`) and the one list of files allowed over a source's
  60,000 bytes (`exceptions`): a generated table or a fixture for good, a file not yet split until the
  workstream named. An entry whose file is gone or within the budget fails the check: remove it in the commit
  that splits the file.
- Below `zig/vendor/` (`check_specs.zig`'s `vendored`) a file is another project's as it publishes it: the
  60,000 bytes do not hold there and it takes no entry in `exceptions`. An `AGENTS.md` there keeps its budget.
- `check_vendor.zig` (run by `check_specs.zig`) holds each `zig/vendor/NAME/` to its `MANIFEST.sha256`: a file
  that differs from the import, is added or is gone needs a row in that directory's `PATCHES.md`, and a row
  needs such a file. It reads no network and no package. `vendor_import.sh` (bash, `rsync`, `sha256sum`; its
  header is its manual) is the one writer of TensorFold's manifest: it takes the host's files from a checkout
  of upstream, and `--compare` checks them against one.
- `host_descriptor.zig` owns TensorFold's module roots, imports, aliases and import selection; `build.zig` and
  `vendor_import.sh` consume it. `test-host-descriptor` rejects malformed graphs. `profile` owns
  `core/startup_receipt.zig`; `core` reexports that API, never a second module for the file.
- `vendor_lib.sh` stages and verifies before publication, records preimages for interrupted publication,
  and refuses subsequent edits on resume. Replay uses an isolated worktree and preserves its pending queue.
  `zig build test-vendor` runs `vendor_import_test.sh` in test-owned repositories and staging caches.
  The full lifecycle and manual recovery commands are in `zig/vendor/tensorfold/AGENTS.md`.
- `check_areas_test.sh` (`zig build test-areas`, spec 171) is the inventory's and the curated coverage's
  contract: `check.sh split`, the scope and the validators over temporary repositories it writes and
  removes, including a step that fails so `check.sh`'s report of it is the proof. Its argument is the
  tools directory read (its own by default), so a changed copy can be exercised without touching the tree's.
  Host/full tiers run it too; `ZSPARK_TEST_LIST=1` skips its fixtures so `sets` remains a listing-only check.

