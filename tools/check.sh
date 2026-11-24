#!/bin/bash
# One entry for a check or a measurement (spec 053). It takes the locks, names the checkpoints, keeps the
# whole output in a log and prints a few lines: on success one line and the log's path; on failure the failing
# tests' names and the first error lines. `tools/check.sh help` prints this short form, to the Logs line.
#
#   tools/check.sh compile            zig build compile, debug: the program and every test compiled, none run; no GPU
#   tools/check.sh host [FILTER]      zig build test-host: no GPU
#   tools/check.sh text [FILTER]      zig/src/text/ alone (test-text); server: zig/src/server/ alone (test-server)
#   tools/check.sh image [FILTER]     zig/src/image/ alone (test-image): the decoders against Pillow, damaged files
#   tools/check.sh audio [FILTER]     zig/src/audio/ alone (test-audio): decoders, resampler and mel against the reference
#   tools/check.sh weights [FILTER]   zig/src/cuda/, weights/ and ranks/ alone (test-weights); the areas below
#   tools/check.sh qwen4 [FILTER]     zig/src/models/qwen4/ alone (test-qwen4), its whole-model tests skipped
#   tools/check.sh qwen4-files        test-qwen4's quickest passing test of each test file it can run, one binary
#   tools/check.sh glm5 [FILTER]      zig/src/models/glm5/ alone (test-glm5)
#   tools/check.sh unlimited_ocr [FILTER]   zig/src/models/unlimited_ocr/'s device tests alone (test-unlimited_ocr)
#   tools/check.sh qwen3_asr [FILTER] zig/src/models/qwen3_asr/ alone (test-qwen3_asr)
#   tools/check.sh nemotron [FILTER]  zig/src/models/nemotron/ alone (test-nemotron)
#   tools/check.sh whole [FILTER]     zig build test-whole: the two whole-model tests alone (host + areas + whole = full)
#   tools/check.sh full               the steps the paths changed since the last head marked passed can reach (`--all`:
#                                     every test), once a commit (see `--again`): the committed head on both Sparks,
#                                     host + areas + whole split between them by seconds; `--one full` here alone, a
#                                     tree with changes too (every step: `zig build test`)
#   tools/check.sh split [BASE [HEAD]]   the steps `full` would run, why, and their halves; nothing run
#   tools/check.sh inventory          validate the test-area inventory and Qwen4 file coverage, without a build
#   tools/check.sh sets               host + areas + whole hold full's tests, whole every tier.whole one: listed, none run
#   tools/check.sh ranks [CKPT]       the two ranks' checks, two_ranks.sh: the random model's session,
#                                     states and hall, the checkpoint's first layers through the pool
#                                     (`hall`) and in prompt windows of 2,048 rows (`windows`), the whole
#                                     checkpoint against its reference, `replies` bf16 and fp8; CKPT a
#                                     directory both machines hold at one path: by default $ZSPARK_RANKS_DIR,
#                                     else ~/models/Qwen3.8-Flash-Next-FP4-Local; its reference
#                                     $ZSPARK_REFERENCE_FP4_LOCAL (the vLLM container's) where CKPT is
#                                     $ZSPARK_FLASH_NEXT_FP4_LOCAL; skipped where no peer answers, and the
#                                     reference check where none is set; CKPT glm-fp4-local ($ZSPARK_GLM_FP4_LOCAL)
#                                     runs GLM FP4-Local whole against $ZSPARK_REFERENCE_GLM_FP4_LOCAL alone
#   tools/check.sh accept CKPT [serve args...]   tools/reference/shared_acceptance.sh on one load
#   tools/check.sh bench CKPT [OUT]   tools/reference/client_bench.sh (zspark, seed 1) and OUT/hashes
#   tools/check.sh build              zig build -Doptimize=fast into zig-out/fast, and its stamp
#   tools/check.sh sanitize AREA FILTER [TOOL]   AREA's tests of FILTER (weights, qwen4, glm5, unlimited_ocr, qwen3_asr, whole) run, then
#                                     their binary under compute-sanitizer TOOL (memcheck unless given)
#   tools/check.sh help               this short form
#
# FILTER is -Dtest-filter's text: the tests whose names hold it. CKPT is glm-fp4-local or a directory.
# Before the command: `--debug` builds the tests debug (a FILTER's compile 3 to 5 s, host code up to 20x slower);
# `--again` runs `full` on a head that already passed it; `--all` runs every step of `full`; `--one` runs `full` on this Spark alone; `--bare` unsets the ZSPARK_* variables; `--stale`
# lets `accept` and `bench` measure a binary that is not the clean head's. Variables: ~/.cache/zspark/env.sh.
# A run killed by a signal (the build runner, or a test binary) prints KILLED, the signal and the last test.
# Logs: ~/.cache/zspark/logs/TIME-COMMAND[-CKPT]: .log, 20 slowest tests last, all in .times/; accept/bench a dir.
#
# A tier's times come from tools/test_runner.zig (spec 133).
#
# The details, read only when a verb's behaviour is in question, follow. The verbs' bodies and the data they
# consult live in the files this one sources (spec 171): check_areas.sh reads the one test-area inventory,
# tools/test_areas.tsv (an area's step, kind, tier, split and seconds, model, Zig root and filters — every
# verb dispatching on an area, `full`'s split and scope, `sets`, `sanitize` and `qwen4-files`' coverage from
# it); check_tier.sh the run under the locks; check_schedule.sh the scope and split of `full`;
# check_sets.sh the sets verb; check_measure.sh the checkpoint verbs.
#
# `--debug` before the command builds the tests with -Doptimize=debug: the compile is three to five times
# shorter (3 to 5 s for one file's tests, `server.structured_test.` or `models.qwen4.blocks.ple_test.`, where the safe build takes 12
# to 17), and a test that computes on the host runs up to twenty times longer (all of `text` 101 s for 24,
# `weights` 65 for 25). So: with a FILTER while editing; an area whole, and every tier, without it. `compile`
# runs no test and is always built debug: 25 s from nothing where the safe build takes one to two minutes.
#
# `full` refuses a clean tree whose head already passed it with the same ZSPARK_* variables (the mark is
# $CACHE/full-passed/SHA-KEY, KEY a hash of mode and environment, written when such a run passes); `--again` before the
# command runs it anyway, scoped against the mark before.
#
# `full` runs the steps a change can reach (spec 133 WS-07): the paths changed (`git diff --no-renames`, the tree's
# untracked files too) since the nearest ancestor of HEAD marked with the same KEY, mapped to steps by the rules
# above `scope()` in tools/check_schedule.sh: shared code (vendor/, src/cuda/, decode/, weights/, ranks/, main.zig,
# kernels outside a model's directory, build.zig and its tool-side descriptor, runner and inventory) reaches
# every step. A Zig source or embedded file reaches the steps whose test roots import it, transitively.
# kernels/MODEL/ reaches GPU steps that import MODEL. Check scripts and host tool test inputs reach host alone.
# `whole` accompanies checks from runtime paths. The first line names the base, the steps
# run with a path that reaches each, and those skipped. No mark, every step; `--all`, every step. No step
# reached: nothing runs, and the clean head is marked. A scoped run that passes marks the head, as a whole one
# does, and the next diffs from it. `split [BASE [HEAD]]` prints the choice dry: since BASE (by default the mark)
# to HEAD (by default the tree), then the halves.
#
# `full` on two Sparks (spec 133 WS-06) refuses a tree with changes. It pushes the head over the link into the
# other Spark's clone (PEER and optionally PEER_REPO from .env.local),
# checks it out in PEER_CHECK (PEER_REPO-check, a worktree made once and reused), and runs `check.sh half STEPS`
# there and here at once: each Spark its own locks, env.sh and binaries, the steps in turn through the verbs' code.
# The caller's own ZSPARK_* and `--debug`/`--bare` reach both. One log here (`TIME-full.log`): each step's line
# named with its Spark, both halves' wall times, the 20 slowest of both; each step's own log and `.times/` are
# `TIME-full-STEP`, the other's copied here (they balance the next run). A step the other did not finish (no
# answer, the link lost, its GPU in use) runs here after, said so; SIGINT or SIGTERM to `full` is meant to end
# both halves (not yet exercised). SPLIT="a b |
# c d" (here | there) replaces the balance and is never marked passed.
#
# The checkpoints and references are the ZSPARK_* variables below, read from ~/.cache/zspark/env.sh (a
# file a machine, not tracked: `export ZSPARK_FLASH_NEXT_FP4_LOCAL=...` and the others; `ZSPARK_GLM_DRAFTER` is
# the directory of GLM's drafter, `ZSPARK_REFERENCE_GLM_DRAFTER` its reference). A variable whose path is not on
# this machine is unset for the Flash Next and GLM paths checked below, so their tests skip. Without the file
# the variables of the caller's environment are used. `--bare` clears every ZSPARK_* input from the caller and
# machine file on both Sparks. The test runner still receives its internal timing and listing settings.
#
# Locks: `machine.lock` shared for every build and test, `gpu.lock` inside it for what needs the GPU, which
# must be free of other processes once the lock is held. A GPU verb compiles first (`zig build STEP-build`, the
# shared lock alone) and takes `gpu.lock` for the run, which finds its binary in the cache (spec 133); the log
# stamps each phase (`check.sh: HH:MM:SS ...`). `accept` and `bench` build under the shared lock and
# then run their script, which holds the machine alone (`machine.lock` whole): so neither is called from
# under a lock.
# Every build driver also takes cache.lock shared (after machine.lock; after gpu.lock when a run needs it).
# Explicit cache pruning takes cache.lock exclusively and nonblocking. No cache lock is held while waiting
# for gpu.lock. Direct zig commands must use the same shared cache lock when pruning could run concurrently.
#
# `sanitize` (spec 103) builds and runs the area's tests of FILTER as the area's verb does, then takes the test
# binary that holds FILTER (a filter is compiled in) and runs it again under `compute-sanitizer --tool TOOL
# --leak-check no --show-backtrace yes`, both locks held, its report in the log beside the tier's (`.san.log`);
# it prints the tool's ERROR SUMMARY. memcheck sees no shared memory and no neighbour inside an arena chunk; a
# whole served load under the tool is `zig/kernels/REFERENCE.md`'s, not this verb's.
#
# `qwen4-files` (spec 103) is `qwen4` with one filter a test file of `zig/tests/models/qwen4/`
# (tools/check_qwen4_files.txt; -Dtest-filter given again, any of them), each file's quickest test that runs
# where env.sh names every checkpoint, chosen from one stamped run of the area. A file whose tests are all
# gated on `tier.whole` takes no entry: under the area's `tier.whole` false they all skip
# (`compare_vllm_test.zig`; `whole` runs it). A test file added there is added to the list, with its
# quickest test; `sets` and `qwen4-files` refuse a missing, doubled or stale entry.
#
# `accept` and `bench` measure zig-out/fast/bin/zspark, built here with -Doptimize=fast (`zig build` alone
# builds zig-out/bin/zspark fast too, the default; the tiers build the tests `-Doptimize=safe`). They refuse a binary whose stamp (`zspark version`) is not the
# head's commit, or is dirty: a tree with a change not committed gives one. `--stale` before the command lets
# it through and says so; ZSPARK=<binary> measures that binary without building, under the same rule.
# CONTEXT, EXTRA, SET, ONLY, STREAMS and MAX_TOKENS reach client_bench.sh as its header says.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
. "$HERE/link_lib.sh"
. "$HERE/check_areas.sh"
. "$HERE/check_tier.sh"
. "$HERE/check_scope.sh"
. "$HERE/check_schedule.sh"
. "$HERE/check_sets.sh"
. "$HERE/check_measure.sh"
GUARD=$(gpu_guard check.sh plain)
CACHE=$HOME/.cache/zspark
ML=$CACHE/machine.lock
GL=$CACHE/gpu.lock
CL=$CACHE/cache.lock
usage() { sed -n '2,/^# Logs:/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }

STALE=
BARE=
DEBUG=
AGAIN=
ALL=
ONE=
while :; do
  case ${1:-} in
    --stale) STALE=1; shift ;;
    --again) AGAIN=1; shift ;;
    --all) ALL=1; shift ;;
    --one) ONE=1; shift ;;
    --debug) DEBUG=1; shift ;;
    --bare) BARE=1; shift ;;
    *) break ;;
  esac
done
[ $# -ge 1 ] || usage
COMMAND=$1; shift

if [ "$COMMAND" = inventory ]; then
  areas_validate inventory || exit 1
  echo 'inventory: test areas and Qwen4 file coverage passed'
  exit 0
fi

export PATH=/home/linuxbrew/.linuxbrew/bin:$PATH
. "$HERE/local_env.sh"
load_local_env "$REPO" || exit $?

# The variables from the machine's file, each kept only if its path is here; one the caller exported wins
# (`ZSPARK_REFERENCE_GLM_FP4_LOCAL=... tools/check.sh ranks glm-fp4-local`).
callers=$(export -p | grep -E '^declare -x ZSPARK_[A-Z0-9_]+=' || true)
# shellcheck disable=SC1091
[ -f "$CACHE/env.sh" ] && . "$CACHE/env.sh"
eval "$callers"
if [ -n "$BARE" ]; then
  for name in ${!ZSPARK_@}; do unset "$name"; done
  callers=
else
  for name in ZSPARK_GLM_DRAFTER ZSPARK_REFERENCE_GLM_DRAFTER ZSPARK_REFERENCE_VISION ZSPARK_FLASH_NEXT_FP4_LOCAL \
      ZSPARK_REFERENCE_FP4_LOCAL; do
    if [ -n "${!name:-}" ] && [ -e "${!name}" ]; then export "${name?}"; else unset "$name"; fi
  done
fi

mkdir -p "$CACHE/logs"
cd "$REPO" || exit 1
# a half of `full` on the other Spark takes the stamp of the run that started it, so their logs pair up
STAMP=${CHECK_STAMP:-$(date +%Y%m%d-%H%M%S)}

case $COMMAND in
  help | -h | --help) usage 0 ;;
  compile) DEBUG=1; tier compile compile host ;;
  build)
    Z=$(binary) || exit 1
    echo "$("$Z" version): $Z"
    ;;
  ranks) verb_ranks "$@" ;;
  accept) verb_accept "$@" ;;
  bench) verb_bench "$@" ;;
  sets) verb_sets ;;
  *)
    # The verbs over the inventory: its load refuses a malformed or contradictory test_areas.tsv first.
    areas_load || { echo "check.sh $COMMAND: the test-area inventory failed (above); $AREAS_FILE" >&2; exit 2; }
    case $COMMAND in
      split) verb_split "$@" ;;
      half) verb_half "$@" ;;
      full) verb_full ;;
      qwen4-files) verb_qwen4_files ;;
      sanitize) verb_sanitize "$@" ;;
      *)
        # an area's name is its verb: `check.sh qwen4 FILTER` is `zig build test-qwen4` under its kind's locks
        if areas_step "$COMMAND"; then
          tier "$COMMAND" "test-$COMMAND" "$(areas_kind "$COMMAND")" "${1:-}"
        else
          usage
        fi ;;
    esac ;;
esac
