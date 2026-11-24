# What runs a step, sourced by check.sh (spec 171): `tier` is `zig build` under the locks with the run's
# output in the log and its verdict on stdout (every tier and area verb, and `half`, go through it);
# `binary` the fast build held to the head's stamp (`build`, `accept`, `bench`); the `sanitize` and
# `qwen4-files` verbs. FILTERS is tier()'s extra -Dtest-filter list; the verbs' dispatch sets it.

FILTERS=()

# A tier or an area: `zig build` under the locks, its output in the log. On the GPU, the step's `-build` twin
# first under the machine lock alone, then the run under both, with the same options: a cache hit.
tier() { # label step gpu [filter]
  local label=$1 step=$2 gpu=$3 filter=${4:-} log=$CACHE/logs/$STAMP-$1.log status=0 start=$SECONDS
  local times=${log%.log}.times
  mkdir -p "$times"
  local -a opts=(--summary all)
  [ -n "$filter" ] && opts+=("-Dtest-filter=$filter")
  for each in "${FILTERS[@]}"; do opts+=("-Dtest-filter=$each"); done
  if [ -n "$DEBUG" ]; then opts+=(-Doptimize=debug); label="$label, debug"; else opts+=(-Doptimize=safe); fi
  local -a run=(flock -s "$CL" env "ZSPARK_TEST_TIMES=$times" zig build "$step" "${opts[@]}")
  stamp() { echo "check.sh: $(date +%T) $*" >> "$log"; }
  : > "$log"
  if [ "$gpu" = gpu ]; then
    stamp "compiling $step-build (machine.lock shared)"
    flock -s "$ML" flock -s "$CL" zig build "$step-build" "${opts[@]}" >> "$log" 2>&1 || status=$?
    if [ $status = 0 ]; then
      stamp "compiled; waiting for gpu.lock"
      # shellcheck disable=SC2016
      flock -s "$ML" flock "$GL" bash -c "$GUARD
        echo \"check.sh: \$(date +%T) gpu.lock held, running $step\"
        exec \"\$@\"" check "${run[@]}" >> "$log" 2>&1 || status=$?
      stamp "ran; gpu.lock released"
    fi
  else
    flock -s "$ML" "${run[@]}" >> "$log" 2>&1 || status=$?
  fi
  # The slowest tests, from the runner's lines (milliseconds, status, name), at the log's end.
  if compgen -G "$times/*.tsv" > /dev/null; then
    { echo; echo "check.sh: the 20 slowest of $(cat "$times"/*.tsv | wc -l) tests ($times):"
      cat "$times"/*.tsv | sort -t $'\t' -k1,1 -g -r | head -20 | awk -F '\t' '{ printf "%10.1f s  %-4s  %s\n", $1 / 1000, $2, $3 }'
      # what a test kept: MemAvailable lost over it, the device's too on GB10 (the 4th column, spec 133 WS-04)
      local kept
      kept=$(cat "$times"/*.tsv | awk -F '\t' '$4 != "" && $4 < -1024' | sort -t $'\t' -k4,4 -g | head -10 | awk -F '\t' '{ printf "%10.1f GiB  %s\n", $4 / 1024, $3 }')
      [ -n "$kept" ] && { echo "check.sh: the tests that lost over 1 GiB of MemAvailable:"; echo "$kept"; }
    } >> "$log"
  fi
  rmdir "$times" 2> /dev/null
  if [ $status = 75 ] && grep -q '^check.sh: the GPU is in use' "$log"; then
    grep '^check.sh: the GPU is in use' "$log" >&2; echo "log: $log" >&2; exit 1
  fi
  # The step's own binary is the run with the most tests (the spec validator's twelve are the other).
  local counts
  counts=$(grep -o -E 'run test [0-9]+ pass[^)]*\)' "$log" | awk '{ n = $0; sub(/.*\(/, "", n); if (n + 0 >= most) { most = n + 0; line = $0 } } END { print line }' | sed 's/^run test //')
  local compile
  compile=$(grep -E 'compile test (safe|fast|small|debug) [^ ]+ (success|cached)' "$log" | head -1 | sed -E 's/.*compile test [a-z]+ [^ ]+ //; s/ MaxRSS.*//; s/^success //; s/^cached.*/cached/')
  # the run's summary, not the `-build` compile's before it: a run that died has none
  local summary
  if [ "$gpu" = gpu ]; then
    summary=$(sed -n '/^check.sh: [0-9:]* gpu.lock held/,$p' "$log" | grep -m1 '^Build Summary:')
  else
    summary=$(grep -m1 '^Build Summary:' "$log")
  fi
  # Killed (spec 133 WS-04): zig's build runner by a signal (the status is 128 and its number, through flock),
  # or a test binary, which zig names and starts again past it. Said as killed, with the last test the
  # runner's lines hold: the newest .tsv's last line, not the tools' tests' (they run beside the step).
  local signal=
  [ $status -gt 128 ] && signal="zig's build runner by SIG$(kill -l $((status - 128)) 2> /dev/null || echo $((status - 128)))"
  local died
  died=$(grep -m1 -o -E "^error: '.*' terminated with signal [A-Z0-9]+" "$log" | sed -E "s/^error: '(.*)' terminated with signal ([A-Z0-9]+)$/SIG\2: \1/")
  if [ -n "$signal" ] || [ -n "$died" ]; then
    local last= newest
    newest=$(grep -L -E $'^[^\t]*\t[^\t]*\tcheck_[a-z_]+\\.' "$times"/*.tsv 2> /dev/null | xargs -r ls -t 2> /dev/null | head -1)
    [ -n "$newest" ] && last=$(tail -1 "$newest" | cut -f3)
    echo "$label${filter:+ \"$filter\"} KILLED: $signal${signal:+${died:+; }}${died:+the test binary by $died}${counts:+; $counts}"
    if [ $status = 137 ] || [[ $died = SIGKILL* ]]; then echo "  SIGKILL: often the OOM killer; the kernel log says (journalctl -k | grep -i oom)"; fi
    [ -n "$signal" ] && echo "  the last test finished: ${last:-none recorded}${newest:+ ($newest)}"
    echo "log: $log"
    return 1
  fi
  # A filter that matches no test passes with fewer tests: a run that held fewer than the filters asked for fails.
  if [ $status = 0 ] && { [ -n "$filter" ] || [ ${#FILTERS[@]} -gt 0 ]; }; then
    local total want=${#FILTERS[@]}
    total=$(sed -n 's/.*(\([0-9]*\) total).*/\1/p' <<<"$counts")
    [ "$want" -gt 0 ] || want=1
    if [ "${total:-0}" -lt "$want" ]; then
      echo "$label${filter:+ \"$filter\"} FAILED: ${counts:-no test ran}; a filter matches no test (${total:-0} ran, at least $want asked for)"
      echo "log: $log"
      return 1
    fi
  fi
  if [ $status = 0 ]; then
    summary=${summary#Build Summary: }
    echo "$label${filter:+ \"$filter\"}: ${counts:-no test ran}; ${summary%%;*}; $((SECONDS - start)) s (compile ${compile:-none})"
    echo "log: $log"
    return 0
  fi
  echo "$label${filter:+ \"$filter\"} FAILED: ${counts:-no test ran}; ${summary#Build Summary: }; $((SECONDS - start)) s"
  # The failing tests' names, then the first error lines (a compile error has no test's name).
  grep -E "^error: '.*' (failed|leaked|crashed|timed out)|^error: while executing test '" "$log" | head -20
  grep -n -E 'error:|panic:|expected .*, found' "$log" | grep -v -E "^[0-9]+:error: '.*' failed|the following (command|build)|^[0-9]+:error: while executing" | head -12
  echo "log: $log"
  return 1
}

# zig-out/fast/bin/zspark at the head, or ZSPARK as given: its path, after the stamp is held against the head.
binary() {
  local log=$CACHE/logs/$STAMP-build.log z=${ZSPARK:-}
  if [ -z "$z" ]; then
    z=$REPO/zig-out/fast/bin/zspark
    if ! flock -s "$ML" flock -s "$CL" zig build -Doptimize=fast -p "$REPO/zig-out/fast" --summary all > "$log" 2>&1; then
      echo "build FAILED" >&2; grep -n -E 'error:' "$log" | head -12 >&2; echo "log: $log" >&2; exit 1
    fi
  fi
  local head stamp commit dirty
  head=$(git -C "$REPO" rev-parse HEAD)
  stamp=$("$z" version 2>/dev/null) || stamp=
  commit=$(echo "$stamp" | awk '{ print $3 }'); dirty=$(echo "$stamp" | awk '{ print $4 }')
  if [ "$commit" != "$head" ] || [ -n "$dirty" ]; then
    if [ -z "$STALE" ]; then
      echo "refused: $z is \"${stamp:-not stamped}\" and the head is $head: commit or stash the change, or say --stale" >&2
      exit 1
    fi
    echo "--stale: measuring \"${stamp:-not stamped}\", the head being $head" >&2
  fi
  echo "$z"
}

# `qwen4-files` (spec 103): `qwen4` with one filter a test file of `zig/tests/models/qwen4/`
# (tools/check_qwen4_files.txt; -Dtest-filter given again, any of them). Every entry is held against its
# file and the files' coverage against the area's tests before anything is built (check_areas.sh's
# areas_check_qwen4): a file whose tests are all gated on `tier.whole` takes no entry.
verb_qwen4_files() {
  areas_check_qwen4 || exit 2
  FILTERS=("${QWEN4_FILES[@]}")
  tier qwen4-files test-qwen4 gpu
}

verb_sanitize() {
  AREA=${1:-}; FILTER=${2:-}; TOOL=${3:-memcheck}
  local gpu_steps=() step
  for step in "${SPLIT_STEPS[@]}"; do [ "$(areas_kind "$step")" = gpu ] && gpu_steps+=("$step"); done
  case " ${gpu_steps[*]} " in *" $AREA "*) ;; *) echo "sanitize: AREA is ${gpu_steps[*]// /, }: a GPU step" >&2; exit 2 ;; esac
  [ -n "$FILTER" ] || { echo "sanitize: a FILTER, or the whole area runs under the tool for hours" >&2; exit 2; }
  tier "$AREA" "test-$AREA" gpu "$FILTER" || exit 1
  # the newest test binary that holds the filter's text
  BIN=$(grep -l -F -- "$FILTER" "$REPO"/.zig-cache/o/*/test 2>/dev/null | xargs -r ls -t | head -1)
  [ -n "$BIN" ] || { echo "sanitize: no test binary holds \"$FILTER\"" >&2; exit 1; }
  SLOG=$CACHE/logs/$STAMP-$AREA.san.log; start=$SECONDS
  # shellcheck disable=SC2016
  flock -s "$ML" flock "$GL" bash -c "$GUARD
    exec \"\$@\"" check /usr/local/cuda/bin/compute-sanitizer --tool "$TOOL" --leak-check no --show-backtrace yes \
    --log-file "$SLOG" "$BIN" > "$SLOG.out" 2>&1
  status=$?
  echo "sanitize $TOOL \"$FILTER\": exit $status; $(grep -h 'ERROR SUMMARY' "$SLOG" | tail -1); $((SECONDS - start)) s"
  echo "log: $SLOG"
  [ $status = 0 ] && grep -q 'ERROR SUMMARY: 0 errors' "$SLOG"
}
