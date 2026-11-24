#!/bin/bash
# tools/test_areas.tsv and tools/check_qwen4_files.txt's contract, run on temporary repositories written
# here (spec 171): the inventory's load and refusal under `check.sh split` (scheduling, the scope's shared
# and scoped paths, the halves from the fallback costs), the Qwen4 coverage's acceptance and every
# refusal, and a failing step's report. Nothing it makes is inside this repository; run it as
# `tools/check_areas_test.sh` (needs git, awk, zig). `split` and the validators are shell alone;
# `host`/`qwen4-files`/`sets` exercise failing builds in temporary repositories without a build.zig. The tools checked
# are this script's own directory, or the one named as its argument (a copy for a run against changed
# sources). Each repository's copy of check.sh has its CACHE pointed inside the repository, so no
# machine's env, logs or marks reach a run; $HOME is never repurposed.
set -u
# Tier membership checks list tests without running the fixture's builds or repositories.
if [ "${ZSPARK_TEST_LIST:-}" = 1 ]; then exit 0; fi
export LC_ALL=C
TOOLS=${1:-$(cd "$(dirname "$0")" && pwd)}
failures=0
ok()   { printf 'ok   %s\n' "$1"; }
bad()  { printf 'FAIL %s\n' "$1"; failures=$((failures + 1)); }
want() { if [ "$2" -eq "$3" ]; then ok "$1"; else bad "$1 (wanted $3, got $2)"; fi; }
want_eq() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (wanted $3, got $2)"; fi; }

export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t

for tool in git awk mktemp; do
    command -v "$tool" >/dev/null || { echo "$tool is needed" >&2; exit 2; }
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# A working repository: check.sh and its sources, an inventory with two split GPU rows beside the host's
# and the whole step's, a non-split host subarea, the Qwen4 test files (one runnable, one gated) and their
# coverage entry. $1 the directory.
make_repo() {
    local r=$1
    mkdir -p "$r/tools" "$r/zig/src/models/qwen4" "$r/zig/kernels/qwen4" "$r/zig/tests/models/qwen4" "$r/zig/tests/fixtures"
    for f in check.sh check_areas.sh check_tier.sh check_scope.sh check_schedule.sh check_sets.sh check_measure.sh link_lib.sh local_env.sh test_areas.tsv check_qwen4_files.txt; do
        cp "$TOOLS/$f" "$r/tools/$f"
    done
    cat >"$r/tools/test_areas.tsv" <<'EOF'
#step	kind	tier	split	seconds	model	root	filters	about
host	host	yes	yes	57	-	zig/host_tests.zig	-	the host tier
text	host	no	no	-	-	zig/text_tests.zig	-	text alone
weights	gpu	no	yes	21	-	zig/weights_tests.zig	-	the shared device code
qwen4	gpu	no	yes	456	qwen4	zig/qwen4_tests.zig	tests.models.qwen4.	qwen4 alone
whole	whole	no	yes	153	-	zig/whole_tests.zig	tests.models.qwen4.gated_test.test.gated case	the whole-model tests
EOF
    # the roots, with the imports `scope` follows: qwen4's root reaches its test file and a source file
    for f in host_tests text_tests weights_tests whole_tests; do
        printf 'comptime {}\n' >"$r/zig/${f}.zig"
    done
    cat >"$r/zig/qwen4_tests.zig" <<'EOF'
comptime {
    _ = @import("tests/models/qwen4/alpha_test.zig");
    _ = @import("src/models/qwen4/thing.zig");
}
EOF
    printf 'pub const x = 1;\n' >"$r/zig/src/models/qwen4/thing.zig"
    cat >"$r/zig/tests/models/qwen4/alpha_test.zig" <<'EOF'
test "the alpha case passes" {
}
EOF
    cat >"$r/zig/tests/models/qwen4/gated_test.zig" <<'EOF'
const whole_tier = @import("tier").whole;
test "gated case" {
    if (!whole_tier) return error.SkipZigTest;
}
EOF
    printf 'qwen4.alpha_test.test.the alpha case passes\n' >"$r/tools/check_qwen4_files.txt"
    git -C "$r" init -q
    printf '/.cache/\n' >>"$r/.git/info/exclude"
    # the copied script keeps its state inside the repository (committed so a checkout keeps it): no
    # machine's env.sh, logs or marks
    sed -i 's|^CACHE=$HOME/.cache/zspark|CACHE='"$r"'/.cache/zspark|' "$r/tools/check.sh"
    git -C "$r" add -A
    git -C "$r" commit -qm init
}

# check.sh in the repository $1 with $2.. as arguments. Output on stdout, the status returned.
check() {
    local r=$1 n; shift
    for n in ${!ZSPARK_@}; do unset "$n"; done 2>/dev/null
    (cd "$r" && tools/check.sh "$@") 2>&1
}

# The coverage validator alone, in the repository $1: what `sets` and `qwen4-files` run before a build,
# through the repository's own copy of check_areas.sh.
coverage() {
    local r=$1
    bash -c 'set -u; REPO=$1; . "$REPO/tools/check_areas.sh"; areas_check_qwen4' _ "$r" 2>&1
}


# --- a valid inventory and its scheduling ---

R=$WORK/valid
make_repo "$R"
base=$(git -C "$R" rev-parse HEAD)

out=$(check "$R" split); st=$?
want "split on a valid inventory" $st 0
case $out in *"no ancestor of HEAD marked passed"*) ok "split: no mark names every step" ;; *) bad "split: no-mark line missing: $out" ;; esac
case $out in *"here 456 s: qwen4 456 s (fixed)"*"there 231 s"*) ok "split: halves from the fallback costs" ;; *) bad "split: halves wrong: $out" ;; esac

# a change to the inventory is a build input: shared, every step
echo '# a comment more' >>"$R/tools/test_areas.tsv"
git -C "$R" add -A; git -C "$R" commit -qm bump
head=$(git -C "$R" rev-parse HEAD)
out=$(check "$R" split "$base" "$head"); st=$?
want "split: inventory change schedules" $st 0
case $out in *"every step: tools/test_areas.tsv shared"*) ok "split: the inventory is shared" ;; *) bad "split: inventory not shared: $out" ;; esac

# Check tooling has a host gate. A tools-only change needs no whole model.
git -C "$R" checkout -q "$base"
echo '# nothing' >>"$R/tools/check.sh"
git -C "$R" add -A; git -C "$R" commit -qm script
head=$(git -C "$R" rev-parse HEAD)
out=$(check "$R" split "$base" "$head"); st=$?
want "split: a check script alone" $st 0
case $out in *"run host: tools/check.sh (host tools); skip weights, qwen4, whole"*) ok "split: a check script selects host only" ;; *) bad "split: tools script scoped wrongly: $out" ;; esac

git -C "$R" switch -q --detach "$base"
printf '\n# fixture change\n' >>"$R/tools/local_env.sh"
out=$(check "$R" split "$base"); st=$?
want "split: loader-only working tree" "$st" 0
case $out in *"run host: tools/local_env.sh (host tools); skip weights, qwen4, whole"*) ok "split: local loader selects host only" ;; *) bad "split: loader scoped wrongly: $out" ;; esac
git -C "$R" restore tools/local_env.sh
out=$(check "$R" split no-such-base); st=$?
want "split: invalid base fails" "$st" 2
case $out in *"invalid base revision: no-such-base"*) ok "split: names invalid base" ;; *) bad "split: invalid base: $out" ;; esac
out=$(check "$R" split "$base" no-such-head); st=$?
want "split: invalid head fails" "$st" 2
case $out in *"invalid head revision: no-such-head"*) ok "split: names invalid head" ;; *) bad "split: invalid head: $out" ;; esac

# The descriptor is a build import even when build.zig and the vendored tree are unchanged.
git -C "$R" checkout -q "$base"
printf 'comptime {}\n' >"$R/tools/host_descriptor.zig"
git -C "$R" add tools/host_descriptor.zig; git -C "$R" commit -qm descriptor
head=$(git -C "$R" rev-parse HEAD)
out=$(check "$R" split "$base" "$head"); st=$?
want "split: the descriptor schedules" "$st" 0
case $out in *"every step: tools/host_descriptor.zig shared"*) ok "split: the descriptor is shared" ;; *) bad "split: descriptor scoped wrongly: $out" ;; esac

# A safe-mode success can scope a safe run. A Debug run must find its own evidence.
git -C "$R" checkout -q "$base"
printf 'documentation\n' >"$R/README.md"
git -C "$R" add README.md; git -C "$R" commit -qm docs
marks=$R/.cache/zspark/full-passed
mkdir -p "$marks"
key() (
    local name DEBUG=$1 BARE=$2
    for name in ${!ZSPARK_@}; do unset "$name"; done
    . "$TOOLS/check_scope.sh"
    pass_key
)
printf '%s\n' 'safe log' >"$marks/$base-$(key '' '')"
out=$(check "$R" split)
case $out in *"nothing: no path a test builds"*) ok "split: safe reuses safe evidence" ;; *) bad "split: safe did not reuse its evidence: $out" ;; esac
out=$(check "$R" --debug split)
case $out in *"no ancestor of HEAD marked passed"*) ok "split: Debug cannot reuse safe evidence" ;; *) bad "split: Debug reused safe evidence: $out" ;; esac

# Bare mode drops new ZSPARK_* names from both the caller and the machine's env.sh.
printf '%s\n' 'export ZSPARK_TOOL_TEST_FROM_ENV=machine' >"$R/.cache/zspark/env.sh"
printf '%s\n' 'bare log' >"$marks/$base-$(key '' 1)"
out=$(cd "$R" && ZSPARK_TOOL_TEST_FROM_CALLER=caller tools/check.sh --bare split)
case $out in *"nothing: no path a test builds"*) ok "split: bare drops caller and machine variables" ;; *) bad "split: bare kept ZSPARK_* variables: $out" ;; esac
out=$(check "$R" split)
case $out in *"no ancestor of HEAD marked passed"*) ok "split: regular runs keep machine variables" ;; *) bad "split: regular run lost machine variables: $out" ;; esac
rm "$R/.cache/zspark/env.sh"

out=$(check "$R" --one full); st=$?
want "full: documentation-only run passes" "$st" 0
mark="$marks/$(git -C "$R" rev-parse HEAD)-$(key '' '')"
if [[ -f $mark && $(<"$mark") == "no step reached since $base: safe log" ]]; then
    ok "full: inherited pass evidence recorded"
else
    bad "full: inherited pass evidence missing: $out"
fi
printf '\n# host test change\n' >>"$R/tools/local_env.sh"
git -C "$R" add tools/local_env.sh; git -C "$R" commit -qm loader
out=$(check "$R" --one full); st=$?
want "full: failing host build propagates" "$st" 1
mark="$marks/$(git -C "$R" rev-parse HEAD)-$(key '' '')"
if [[ ! -e $mark ]]; then ok "full: a failed run writes no mark"; else bad "full: a failed run wrote $mark"; fi

# a test file its area's root imports reaches that step and whole
git -C "$R" checkout -q "$base"
echo 'test "another" {}' >>"$R/zig/tests/models/qwen4/alpha_test.zig"
git -C "$R" add -A; git -C "$R" commit -qm test
head=$(git -C "$R" rev-parse HEAD)
out=$(check "$R" split "$base" "$head"); st=$?
want "split: an imported test file" $st 0
case $out in *"qwen4: zig/tests/models/qwen4/alpha_test.zig (imported)"*"whole: always with runtime checks"*) ok "split: qwen4 and whole run" ;; *) bad "split: imported file scoped wrongly: $out" ;; esac

# a changed file beside the model's .zig (not itself a .zig) reaches the model's step; a new .zig no
# root imports reaches nothing; a kernel of the model's reaches it
git -C "$R" checkout -q "$base"
printf 'data\n' >"$R/zig/src/models/qwen4/side.dat"
git -C "$R" add -A; git -C "$R" commit -qm model
head=$(git -C "$R" rev-parse HEAD)
out=$(check "$R" split "$base" "$head")
case $out in *"qwen4: zig/src/models/qwen4/side.dat (beside its .zig)"*) ok "split: a file beside the model's reaches qwen4" ;; *) bad "split: model-dir scoping wrong: $out" ;; esac
prev=$head
printf 'pub const y = 2;\n' >"$R/zig/src/models/qwen4/other.zig"
git -C "$R" add -A; git -C "$R" commit -qm unimported
head=$(git -C "$R" rev-parse HEAD)
out=$(check "$R" split "$prev" "$head")
case $out in *"no path a test builds"*) ok "split: an unimported .zig runs nothing" ;; *) bad "split: unimported .zig scoped wrongly: $out" ;; esac
prev=$head
printf 'x\n' >"$R/zig/kernels/qwen4/k.cu"
git -C "$R" add -A; git -C "$R" commit -qm kernel
head=$(git -C "$R" rev-parse HEAD)
out=$(check "$R" split "$prev" "$head")
case $out in *"qwen4: zig/kernels/qwen4/k.cu (kernels/qwen4)"*) ok "split: a kernel reaches qwen4" ;; *) bad "split: kernel scoping wrong: $out" ;; esac

# --- the inventory's refusals ---

# $1 name, $2 the sed/edits run on the valid inventory, $3 the text the refusal must hold
reject() {
    local r=$WORK/rej-$1
    make_repo "$r"
    eval "$2"
    local out st
    out=$(check "$r" split); st=$?
    want "$1: refused" $st 2
    case $out in *"$3"*) ok "$1: message" ;; *) bad "$1: wrong message: $out" ;; esac
}

reject dup-row 'printf "host\thost\tyes\tyes\t57\t-\tzig/host_tests.zig\t-\tagain\n" >>"$r/tools/test_areas.tsv"' "given twice"
reject bad-kind 'sed -i "s/^weights\tgpu/weights\tdev/" "$r/tools/test_areas.tsv"' "not host, gpu or whole"
reject gpu-no-split 'sed -i "s/^qwen4\tgpu\tno\tyes\t456/qwen4\tgpu\tno\tno\t-/" "$r/tools/test_areas.tsv"' "full would never run it"
reject no-split-seconds 'sed -i "s/^text\thost\tno\tno\t-/text\thost\tno\tno\t9/" "$r/tools/test_areas.tsv"' "needs - for seconds"
reject split-no-seconds 'sed -i "s/^weights\tgpu\tno\tyes\t21/weights\tgpu\tno\tyes\t-/" "$r/tools/test_areas.tsv"' "needs its fallback seconds"
reject bad-model 'sed -i "s/^qwen4\tgpu\tno\tyes\t456\tqwen4/qwen4\tgpu\tno\tyes\t456\tgone/" "$r/tools/test_areas.tsv"' "no zig/src/models/gone"
reject model-on-host 'sed -i "s/^text\thost\tno\tno\t-\t-/text\thost\tno\tno\t-\tqwen4/" "$r/tools/test_areas.tsv"' "on a host step"
reject missing-root 'rm "$r/zig/weights_tests.zig"' "is not a file"
reject wrong-root 'sed -i "s|zig/weights_tests.zig|zig/other_tests.zig|" "$r/tools/test_areas.tsv"' "is not zig/weights_tests.zig"
reject short-row 'printf "fewer\thost\n" >>"$r/tools/test_areas.tsv"' "fewer than nine columns"
reject no-whole 'grep -v "^whole" "$r/tools/test_areas.tsv" >"$r/t" && mv "$r/t" "$r/tools/test_areas.tsv"' "exactly one needed"
reject no-host 'grep -v "^host" "$r/tools/test_areas.tsv" >"$r/t" && mv "$r/t" "$r/tools/test_areas.tsv"' "no host step"
reject bad-tier 'sed -i "s/^text\thost\tno/text\thost\tyes/; s/^qwen4\tgpu\tno/qwen4\tgpu\tyes/" "$r/tools/test_areas.tsv"' "tier=yes on a gpu step"

# an unknown verb still prints the usage; an area verb without zig fails the tier, not the dispatch
out=$(check "$R" bogus); st=$?
want "bogus verb" $st 2

# --- the Qwen4 coverage ---

out=$(coverage "$R"); st=$?
want "coverage: valid" $st 0
if [ "$#" -eq 0 ]; then
    out=$(coverage "$(dirname "$TOOLS")"); st=$?
    want "coverage: the checkout's Qwen4 shortcuts" "$st" 0
    [ "$st" -eq 0 ] || printf '%s\n' "$out"
fi

# $1 name, $2 edits, $3 the text the refusal must hold
cover_reject() {
    local r=$WORK/cov-$1
    make_repo "$r"
    eval "$2"
    local out st
    out=$(coverage "$r"); st=$?
    want "$1: refused" $st 1
    case $out in *"$3"*) ok "$1: message" ;; *) bad "$1: wrong message: $out" ;; esac
}

cover_reject missing 'printf "test \"the gamma case\" {}\n" >"$r/zig/tests/models/qwen4/gamma_test.zig"' "have no entry"
cover_reject dup-entry 'printf "qwen4.alpha_test.test.the alpha case passes\n" >>"$r/tools/check_qwen4_files.txt"' "given twice"
cover_reject dup-file 'printf "test \"the beta case\" {}\n" >>"$r/zig/tests/models/qwen4/alpha_test.zig"; printf "qwen4.alpha_test.test.the beta case\n" >>"$r/tools/check_qwen4_files.txt"' "has an entry already"
cover_reject stale-file 'printf "qwen4.gone_test.test.the gone case\n" >>"$r/tools/check_qwen4_files.txt"' "no test file"
cover_reject stale-name 'printf "qwen4.alpha_test.test.no such test\n" >>"$r/tools/check_qwen4_files.txt"' "no test of"
cover_reject commented-name 'printf "// test \"hidden\" {}\n" >>"$r/zig/tests/models/qwen4/alpha_test.zig"; printf "qwen4.alpha_test.test.hidden\n" >>"$r/tools/check_qwen4_files.txt"' "no test of"
cover_reject gated-file 'printf "qwen4.gated_test.test.gated case\n" >>"$r/tools/check_qwen4_files.txt"' "all gated on tier.whole"
cover_reject outside-qwen4 'printf "glm5.alpha_test.test.the alpha case passes\n" >>"$r/tools/check_qwen4_files.txt"' "outside qwen4"
cover_reject gated-selection 'cat >"$r/zig/tests/models/qwen4/mixed_test.zig" <<'"'"'EOF'"'"'
const whole_tier = @import("tier").whole;
test "runnable case" {
}
test "the gated one" {
    if (!whole_tier) return error.SkipZigTest;
}
EOF
printf "qwen4.mixed_test.test.the gated one\n" >>"$r/tools/check_qwen4_files.txt"' "names a tier.whole test"

# a file whose tests all gate is exempt; a new runnable file beside it is not hidden by the exemption
r=$WORK/cov-exempt
make_repo "$r"
printf 'test "all gated one" {\n    if (!whole_tier) return error.SkipZigTest;\n}\n' >>"$r/zig/tests/models/qwen4/gated_test.zig"
out=$(coverage "$r"); st=$?
want "coverage: a gated file exempt" $st 0
printf 'test "the delta case" {}\n' >"$r/zig/tests/models/qwen4/delta_test.zig"
out=$(coverage "$r"); st=$?
want "coverage: new runnable file not hidden" $st 1
case $out in *"have no entry"*) ok "coverage: the new file is seen" ;; *) bad "coverage: wrong message: $out" ;; esac

# `check.sh qwen4-files` runs the same coverage before the tier: refused there, the tier never built
out=$(check "$r" qwen4-files); st=$?
want "qwen4-files: coverage refused first" $st 2
out=$(check "$R" qwen4-files); st=$?
want "qwen4-files: valid coverage reaches the build" $st 1
case $out in *"FAILED"*) ok "qwen4-files: dispatch reaches tier" ;; *) bad "qwen4-files: odd failure: $out" ;; esac

# Inventory and sets share static failures. With valid inputs inventory never runs the failing fixture build.
out=$(check "$r" inventory); st=$?
want "inventory: stale Qwen coverage refused" "$st" 1
case $out in *"curated file coverage is inconsistent"*) ok "inventory: names Qwen coverage" ;; *) bad "inventory: wrong message: $out" ;; esac
out=$(check "$r" sets); st=$?
want "sets: stale Qwen coverage refused" "$st" 1
case $out in *"curated file coverage is inconsistent"*) ok "sets: names Qwen coverage" ;; *) bad "sets: wrong message: $out" ;; esac

# `check.sh sets` runs the inventory and the coverage before the lists
r=$WORK/sets-bad
make_repo "$r"
printf "fewer\thost\n" >>"$r/tools/test_areas.tsv"
out=$(check "$r" sets); st=$?
want "sets: bad inventory refused" $st 1
case $out in *"inventory is inconsistent"*) ok "sets: names the inventory" ;; *) bad "sets: wrong message: $out" ;; esac

out=$(check "$r" inventory); st=$?
want "inventory: bad TSV refused" "$st" 1
case $out in *"inventory is inconsistent"*) ok "inventory: names the TSV" ;; *) bad "inventory: wrong message: $out" ;; esac

r=$WORK/sets-ok
make_repo "$r"
out=$(check "$r" inventory); st=$?
want "inventory: valid tree passes without a build" "$st" 0
case $out in *"file coverage passed"*) ok "inventory: prints success" ;; *) bad "inventory: wrong message: $out" ;; esac
out=$(check "$r" sets); st=$?
want "sets: valid reaches the lists" $st 1
case $out in *"did not build or list"*) ok "sets: dispatch reaches zig" ;; *) bad "sets: odd failure: $out" ;; esac

# --- the tier's failure propagation ---

# `check.sh host` runs `zig build test-host` under the locks; the fixture has no build.zig, so zig fails
# and the failure, not a pass, is what the verb reports (a dropped `|| status=$?` would turn it into one)
out=$(check "$R" host); st=$?
want "host: zig's failure reaches the caller" $st 1
case $out in *"host FAILED"*) ok "host: failure is reported" ;; *) bad "host: a failing build passed: $out" ;; esac

# `half` runs each step through the same code: the failed step says fail, and half's exit is 1
out=$(check "$R" half host); st=$?
want "half: a failing step fails the half" $st 1
case $out in *"half: host fail"*) ok "half: the step says fail" ;; *) bad "half: step not failed: $out" ;; esac

# `ranks` validates the checkpoint, probes the peer, then builds and checks the stamp.
rank_repo=$WORK/ranks
mkdir -p "$rank_repo/zig-out/bin" "$rank_repo/tools" "$rank_repo/checkpoint" "$rank_repo/cache/logs"
cat >"$rank_repo/zig-out/bin/zspark" <<'EOF'
#!/bin/bash
printf 'stamp\n' >>"$TRACE"
printf 'zspark test fixture-head\n'
EOF
cat >"$rank_repo/tools/two_ranks.sh" <<'EOF'
#!/bin/bash
printf 'rank\n' >>"$TRACE"
EOF
chmod +x "$rank_repo/zig-out/bin/zspark" "$rank_repo/tools/two_ranks.sh"
rank_case() (
    local scenario=$1 checkpoint=$rank_repo/checkpoint
    set -u
    REPO=$rank_repo HERE=$rank_repo/tools CACHE=$rank_repo/cache STAMP=$scenario ML=machine CL=cache
    export TRACE=$rank_repo/$scenario.trace
    : > "$TRACE"
    unset ZSPARK_FLASH_NEXT_FP4_LOCAL ZSPARK_GLM_FP4_LOCAL
    PEER_REPO=$rank_repo
    case $scenario in invalid) checkpoint=$rank_repo/missing; unset PEER ;; unset-peer) unset PEER ;; *) PEER=192.0.2.10 ;; esac
    ssh() { printf 'peer\n' >>"$TRACE"; [ "$scenario" != unreachable ]; }
    flock() { printf 'build\n' >>"$TRACE"; [ "$scenario" != build-fails ]; }
    git() { printf 'fixture-head\n'; }
    . "$TOOLS/check_measure.sh"
    verb_ranks "$checkpoint"
)
for scenario in invalid unset-peer unreachable build-fails reachable; do
    out=$(rank_case "$scenario" 2>&1); st=$?
    trace=$(paste -sd' ' "$rank_repo/$scenario.trace")
    case $scenario in
        invalid) expected_status=2; expected_trace=; diagnostic='no such checkpoint' ;;
        unset-peer) expected_status=0; expected_trace=; diagnostic='PEER is unset' ;;
        unreachable) expected_status=0; expected_trace=peer; diagnostic='skipped' ;;
        build-fails) expected_status=1; expected_trace='peer build'; diagnostic='build FAILED' ;;
        reachable) expected_status=0; expected_trace='peer build stamp rank rank rank rank rank rank rank'; diagnostic='passed' ;;
    esac
    want "ranks: $scenario status" "$st" "$expected_status"
    want_eq "ranks: $scenario order" "$trace" "$expected_trace"
    case $out in *"$diagnostic"*) ok "ranks: $scenario diagnostic" ;; *) bad "ranks: $scenario output: $out" ;; esac
done

echo
if [ $failures -gt 0 ]; then echo "$failures failures" >&2; exit 1; fi
echo "check_areas_test.sh: all cases passed"
