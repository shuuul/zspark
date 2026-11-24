# A worker's standing rules

What binds every worker on this repository, a subagent included. A brief names a spec's row and this file; it
does not restate them. Root `AGENTS.md` has the build commands and the row rule, the nearest `AGENTS.md` the
local rules. `zig build check-specs` holds this file to 7,000 bytes.

1. **Where to work.** In your own worktree on a branch named after the spec (`git worktree add ../zspark-NNNx -b
   spec-NNN-topic main`), never in main's tree. Push only that branch; the coordinator merges. spark-a and
   spark-b each hold a clone (`~/repo/zspark`); the spec's Decisions say which machine a workstream runs on.
   Leave no untracked file behind.

2. **The locks**, under `~/.cache/zspark/`; take `machine.lock` first, then `gpu.lock` when needed.
   `tools/check.sh` takes them; a command run outside it takes them itself.
   - `gpu.lock`: every command that touches the GPU runs under `flock ~/.cache/zspark/gpu.lock`, one model
     load at a time.
   - `machine.lock`: every build, test or other heavy command takes it shared (`flock -s`); a timing takes it
     whole (`flock -x`) with the GPU lock inside, for the timing only, not for a session. A speed is recorded
     only from a run that held the machine alone, repeated, with `uptime` beside it. A download takes neither
     lock (the owner, 2026-10-10).
   - `cache.lock`: every build driver takes it shared, after the machine lock (and after the GPU lock for a
     test run). Never hold it while waiting for the GPU. Cache pruning takes it exclusive and nonblocking,
     then revalidates candidates and active mappings; it refuses an active build rather than waiting on a
     serving process. Direct `zig build` commands use `flock -s ~/.cache/zspark/cache.lock zig build ...`.
     Build/install the cache tool first; run its prune executable directly, outside a build driver's locks.

3. **The servers.** Both Sparks normally serve a model that fills their GPUs. Run `nvidia-smi` before anything
   on the GPU. Stop that server only through its own repository's `stop.sh` (`~/repo/*-TensorFold/`), never by
   killing the process, and do not start it again: the owner restores it. Kill no process that is not yours. A `zspark` that you or your
   run started and left holding the GPU or `check.sh` you end yourself (`kill -TERM`), once `ps` shows its parent,
   binary and loopback port are yours; it is not the owner's to wait for. A bench listens on 127.0.0.1:18888
   and uses a separate port from the live service.

4. **The check a change needs**, through `tools/check.sh` (its header is its manual): the least that answers
   the question (the owner, 2026-10-10).
   - While editing: `check.sh compile` (everything compiled, nothing run, no GPU, no queue), then the test you
     touched, `check.sh --debug AREA FILTER` (3 to 5 s to compile). `zspark compare DIR
     REF/primes.serial.f32.safetensors layers=4` prints four layers against TensorFold in 3 s. Build a
     shortcut before paying an expensive step a third time.
   - No tier and no baseline: the brief gives the counts before; the coordinator runs one `full` a merge, the
     committed head split over both Sparks (9 min; `--one full`, one Spark, 22) and compares; `ranks` after. An area whole (`qwen4` 11 min, `glm5` 6) only where the brief
     asks, on the Spark it names. A queued check is waited for or left to end, never killed.
   - A stage pays for no test twice (spec 133): `full` alone, or `whole` (the two whole-model tests) beside the
     areas a round already ran on that head; there is no `quick`. A GPU verb compiles under the shared machine
     lock and holds `gpu.lock` for the run alone. A tier's log ends with its 20 slowest tests, all of them under
     `.times/`: a cost or a saving is quoted from it, not guessed. A stage starts with `check.sh sets` (lists
     only, no GPU): host, areas and whole hold `full`'s tests, and the `whole` row in `tools/test_areas.tsv` every `tier.whole` test.
   - Hashes, where the brief says: `accept` on one checkpoint for a move, `bench` on all three otherwise;
     both refuse a binary that is not the clean head's, and a figure names its commit.

5. **A measurement has a plan and a budget**, written in the spec before it runs.
   - What recorded data can answer is answered on the host (`tools/reference/draft_policy.py` over recorded
     chains, a choice among candidates from one trace).
   - The device times two things: the base and the one candidate chosen. A sweep on the device needs a reason
     no recording gives.
   - The cases are those the question names, then one pass of the set on each checkpoint the change can reach.
   - One build a side, one full tier, on the final rebased head. The brief gives the number of loads; a worker
     that needs more says why in its entry first.

6. **The row rule.** Root `AGENTS.md`, "The rule everything serves".

7. **The spec.** Read `specs/AGENTS.md`. Claim your workstream's row, then append one dated entry under
   "Progress and handoff": about 40 lines plus tables, with what changed, the evidence, what remains. Raw
   output goes under `~/.cache/zspark/scratch-NNN/` and is named in the entry, not pasted. Tick only what you
   showed. Do not rewrite another worker's entry or close the spec. `zig build check-specs` before a commit.

8. **Commits.** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `build:`, `refactor:`, `chore:`, a
   scope such as `feat(qwen4):` where it helps), then what changed and why in prose, then the co-author trailer
   your session was given. One move or one group of files a commit. `zig fmt --exclude zig/vendor zig build.zig tools` first. A change
   leaves nothing to tidy: `docs/architecture.md`, "What every change does".

9. **Weights and Python.** Weights and references move between the Sparks over the direct link with `tools/spark_copy.sh SRC PEER DEST_DIR`, under both machines'
   `machine.lock`, never by a second download. A hybrid directory's files that are the published checkpoint's
   are symbolic links, which the script keeps as links: link them again on the other side. Python lives under
   `tools/reference/` alone and never runs at runtime. No weight is committed; GLM's DFlash2 drafter's are
   not copied either (spec 067).

10. **What goes to the owner**, through the coordinator, before it is done: anything that costs the row rule
    or changes behaviour, an interface or a default; a removal the spec does not list; a branch or tree deleted
    that is not merged; a speed-up that is not bit-exact. A fault met on the way gets its own spec.

11. **The report.** Branch and head; what changed, a line a file; the figures with the command and the commit
    they came from; test counts by tier; what was dropped or not done, said plainly; what the owner must decide.
