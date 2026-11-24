# The `sets` verb, sourced by check.sh (spec 171): a stage's checks that run no test. The steps' test names
# in list mode (ZSPARK_TEST_LIST: compiled debug, nothing run, no GPU): a name of `full` that host, the
# areas and whole do not hold fails, but a root's anonymous test (`tests.test_0`) and the tools' tests
# (such as `test-specs`), which run beside every tier. And an area lists
# a whole-model test it skips, so the names cannot see a gate: each test whose body reads
# `@import("tier").whole` (or a constant of it) must be in test-whole's list. Since spec 171 the inventory
# itself is checked first (check_areas.sh's areas_load: the schema tools/test_areas.tsv's header gives,
# and every split row's root and model), and `qwen4-files`' curated coverage
# (tools/check_qwen4_files.txt against zig/tests/models/qwen4/) the same way.

verb_sets() {
  areas_validate sets || exit 1
  log=$CACHE/logs/$STAMP-sets.log start=$SECONDS
  lists=${log%.log}.lists
  mkdir -p "$lists/full" "$lists/steps" "$lists/whole"
  : > "$log"
  status=0
  list() { flock -s "$ML" flock -s "$CL" env ZSPARK_TEST_LIST=1 "ZSPARK_TEST_TIMES=$lists/$1" zig build "${@:2}" -Doptimize=debug --summary all >> "$log" 2>&1 || status=$?; }
  list full test
  local -a step_names=()
  for step in "${AREA_STEPS[@]}"; do [ "$step" != "$AREA_WHOLE" ] && step_names+=("test-$step"); done
  [ $status = 0 ] && list steps "${step_names[@]}"
  [ $status = 0 ] && list whole "test-$AREA_WHOLE"
  if [ $status != 0 ]; then
    echo "sets FAILED: a step did not build or list"; grep -n -E 'error:' "$log" | head -12; echo "log: $log"; exit 1
  fi
  names() { cat "$@" | cut -f3 | grep -v -E '^[a-z0-9_]+\.test_[0-9]+$|^check_[a-z_]+\.' | sort -u; }
  names "$lists"/full/*.tsv > "$lists/full.names"
  names "$lists"/steps/*.tsv "$lists"/whole/*.tsv > "$lists/steps.names"
  names "$lists"/whole/*.tsv > "$lists/whole.names"
  # the gated tests, by their full names (`tests.DIR.FILE.test.NAME`)
  gated_find zig/tests | sort -u > "$lists/gated.all"
  # a gate outside a test's body (a helper the test calls) hides the test from this check: refused
  stray=$(grep '^STRAY' "$lists/gated.all" | cut -f2)
  grep -v '^STRAY' "$lists/gated.all" | awk -F '\t' '{ print $1 ".test." $2 }' > "$lists/gated.names"
  missing=$(comm -23 "$lists/full.names" "$lists/steps.names")
  ungated=$(comm -23 "$lists/gated.names" "$lists/whole.names")
  extra=$(comm -13 "$lists/full.names" "$lists/steps.names" | wc -l)
  { echo "check.sh: full's tests no step holds:"; echo "$missing"; echo "check.sh: tier.whole tests not in test-whole:"; echo "$ungated"
    echo "check.sh: tests the steps hold that full does not: $extra"; comm -13 "$lists/full.names" "$lists/steps.names"; } >> "$log"
  if [ -n "$missing" ] || [ -n "$ungated" ] || [ -n "$stray" ]; then
    [ -n "$stray" ] && { echo "sets FAILED: tier.whole read outside a test's body (write the gate in the test):"; head -20 <<<"$stray"; }
    [ -n "$missing" ] && { echo "sets FAILED: $(wc -l <<<"$missing") of full's $(wc -l < "$lists/full.names") tests in no step:"; head -20 <<<"$missing"; }
    [ -n "$ungated" ] && { echo "sets FAILED: $(wc -l <<<"$ungated") tests gated on tier.whole not in the whole step's filters:"; head -20 <<<"$ungated"; }
    echo "log: $log"; exit 1
  fi
  echo "sets: full's $(wc -l < "$lists/full.names") tests all in host, the areas and whole ($extra held there alone), its $(wc -l < "$lists/gated.names") tier.whole tests in whole; $((SECONDS - start)) s"
  echo "log: $log"
}
