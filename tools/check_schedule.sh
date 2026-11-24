# `full`'s scheduling, sourced by check.sh (spec 171): the split of a tier's steps over the two Sparks
# (step_seconds, split_plan) and the `split`, `half` and `full` verbs. Scope and pass evidence
# come from check_scope.sh. The inventory supplies SPLIT_STEPS and SPLIT_FIXED, the fallback
# seconds. A step's seconds are its tests' in the newest table a split
# left (`*-full-STEP.times`, never a FILTER's run, its log holding the run's summary), else the
# inventory's figure.

step_seconds() { # STEP: "SECONDS SOURCE"
  local dir
  for dir in $(ls -d "$CACHE"/logs/*-full-"$1".times 2> /dev/null | sort -r); do
    grep -q '^Build Summary:.* tests' "${dir%.times}.log" 2> /dev/null || continue
    echo "$(cat "$dir"/*.tsv | awk -F '\t' '{ t += $1 } END { printf "%d", t / 1000 + 0.5 }') ${dir##*/}"
    return
  done
  echo "${SPLIT_FIXED[$1]:-60} fixed"
}
# The two halves of STEPS (by default every step), "HERE | THERE": every way of splitting them tried, the one whose
# larger half is least kept; the larger half here, as the other pays for the push and its checkout. SPLIT="a b |
# c d" gives it instead.
split_plan() { # [STEP...]
  if [ -n "${SPLIT:-}" ]; then echo "$SPLIT"; return; fi
  local -a steps=("$@") secs=()
  [ $# -gt 0 ] || steps=("${SPLIT_STEPS[@]}")
  local n=${#steps[@]} i mask a b best=-1 pick=0 here= there=
  for i in "${!steps[@]}"; do secs[i]=$(step_seconds "${steps[i]}" | cut -d' ' -f1); done
  for ((mask = 0; mask < (1 << n); mask++)); do
    a=0; b=0
    for ((i = 0; i < n; i++)); do if (((mask >> i) & 1)); then ((a += secs[i])); else ((b += secs[i])); fi; done
    ((a < b)) && continue
    if ((best < 0 || a < best)); then best=$a; pick=$mask; fi
  done
  for ((i = 0; i < n; i++)); do if (((pick >> i) & 1)); then here+=" ${steps[i]}"; else there+=" ${steps[i]}"; fi; done
  echo "${here# } | ${there# }"
}

# `split BASE [HEAD]` prints the choice dry: since BASE (by default the mark) to HEAD (by default the
# tree), then the halves.
verb_split() {
  base=${1:-}
  [ -n "$base" ] || base=$(scope_base)
  if [ -n "$ALL" ]; then
    SCOPE_STEPS=("${SPLIT_STEPS[@]}"); echo "scope: every step (--all)"
  elif [ -z "$base" ]; then
    SCOPE_STEPS=("${SPLIT_STEPS[@]}"); echo "scope: every step (no ancestor of HEAD marked passed with these ZSPARK_* variables)"
  else
    scope "$base" "${2:-}" || exit 2
    echo "scope ${2:+to $(git rev-parse --short "$2") }$SCOPE_LINE"
  fi
  [ ${#SCOPE_STEPS[@]} -gt 0 ] || exit 0
  plan=$(split_plan "${SCOPE_STEPS[@]}")
  for side in "here:${plan%%|*}" "there:${plan#*|}"; do
    total=0 line=
    for step in ${side#*:}; do read -r sec src <<<"$(step_seconds "$step")"; ((total += sec)); line+=" $step $sec s ($src);"; done
    echo "${side%%:*} $total s:${line%;}"
  done
}

# `full`'s steps given, run here in turn (WS-06): their binaries built first in one `zig build`, then each
# through tier(); after each a line `half: STEP pass|fail SECONDS`, which `full` reads. Its process group is
# in $CACHE/full-half-STAMP.pid while it runs, for `full` to end it.
verb_half() {
  [ $# -ge 1 ] || usage
  echo $$ > "$CACHE/full-half-$STAMP.pid"
  trap 'rm -f "$CACHE/full-half-$STAMP.pid"' EXIT
  builds=()
  for step in "$@"; do builds+=("test-$step-build"); done
  optimize=-Doptimize=safe; [ -n "$DEBUG" ] && optimize=-Doptimize=debug
  start=$SECONDS
  flock -s "$ML" flock -s "$CL" zig build "${builds[@]}" "$optimize" --summary all > "$CACHE/logs/$STAMP-full-$(hostname -s)-build.log" 2>&1 ||
    echo "half: not every binary built ($CACHE/logs/$STAMP-full-$(hostname -s)-build.log); each step says why"
  echo "half: compiled $((SECONDS - start))"
  status=0
  for step in "$@"; do
    start=$SECONDS
    if tier "full-$step" "test-$step" "$(areas_kind "$step")"; then result=pass; else result=fail; status=1; fi
    echo "half: $step $result $((SECONDS - start))"
  done
  echo "half: ran $SECONDS"
  exit $status
}

verb_full() {
  # A tree with any change, untracked files included, is never marked: its head does not name what was built.
  MARK=$(pass_current_mark) || MARK=
  if [ -n "$MARK" ] && [ -f "$MARK" ] && [ -z "$AGAIN" ]; then
    echo "full: $(git rev-parse --short HEAD) already passed with this optimization mode and ZSPARK_* environment: $(cat "$MARK"); --again runs it anyway" >&2
    exit 1
  fi
  [ -n "$MARK" ] || [ -n "$ONE" ] || { echo "full: the tree has changes, and both Sparks run one commit: commit them, or --one full runs this tree here" >&2; exit 1; }
  # The steps the changes since the last passed ancestor reach (WS-07); `--all`, or no such mark, every step.
  BASE=
  [ -n "$ALL" ] || BASE=$(scope_base)
  if [ -n "$BASE" ]; then
    scope "$BASE" || exit 2
    echo "full scope $SCOPE_LINE"
  else
    SCOPE_STEPS=("${SPLIT_STEPS[@]}")
    echo "full scope: every step ($([ -n "$ALL" ] && echo "--all" || echo "no ancestor of HEAD marked passed with these ZSPARK_* variables"))"
  fi
  if [ ${#SCOPE_STEPS[@]} = 0 ]; then
    if [ -n "$MARK" ]; then
      evidence=$(cat "$(pass_mark_path "$BASE")") || exit 1
      pass_record "$MARK" "no step reached since $BASE: $evidence" || exit 1
      echo "full passed: no step reached; $(git rev-parse --short HEAD) marked as its base $(git rev-parse --short "$BASE") (--all runs every step)"
    else
      echo "full: no step reached (the tree has changes: not marked)"
    fi
    exit 0
  fi
  if [ -n "$ONE" ]; then
    LOG=$CACHE/logs/$STAMP-full.log
    if [ ${#SCOPE_STEPS[@]} = ${#SPLIT_STEPS[@]} ]; then
      tier full test gpu || exit 1
    else
      # the chosen steps alone, here, as a half runs them: each step's line, its log `TIME-full-STEP`
      FLAGS=
      [ -n "$DEBUG" ] && FLAGS+=" --debug"
      [ -n "$BARE" ] && FLAGS+=" --bare"
      # shellcheck disable=SC2086
      env CHECK_STAMP="$STAMP" "$0"$FLAGS half "${SCOPE_STEPS[@]}" 2>&1 | tee "$LOG" | grep -v '^half: '
      if [ "${PIPESTATUS[0]}" != 0 ]; then
        echo "full FAILED: $(sed -n 's/^half: \([a-z0-9_]*\) fail .*/\1/p' "$LOG" | paste -sd' ')"; echo "log: $LOG"; exit 1
      fi
    fi
    pass_record "$MARK" "$LOG" || exit 1
    exit 0
  fi
  # Two Sparks (WS-06): the committed head on both, a half each, one log here.
  SHA=$(git rev-parse HEAD)
  LOG=$CACHE/logs/$STAMP-full.log
  ME=$(hostname -s)
  # the other Spark on the link, its clone (the main tree's path here) and the worktree a run checks out there
  if [ -z "${PEER:-}" ]; then
    echo 'full: set PEER in .env.local for two Sparks, or use --one full' >&2
    exit 2
  fi
  [ -n "${PEER_REPO:-}" ] || PEER_REPO=$(git worktree list --porcelain | sed -n 's/^worktree //p' | head -1)
  PEER_CHECK=${PEER_CHECK:-$PEER_REPO-check}
  SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER")
  PLAN=$(split_plan "${SCOPE_STEPS[@]}")
  read -r -a HERE_STEPS <<<"${PLAN%%|*}"
  read -r -a THERE_STEPS <<<"${PLAN#*|}"
  : > "$LOG"
  say() { echo "$*" | tee -a "$LOG"; }
  THEM=
  [ ${#THERE_STEPS[@]} -gt 0 ] && { THEM=$("${SSH[@]}" hostname -s 2> /dev/null) || THEM=; }
  say "full at $(git rev-parse --short HEAD) on two Sparks: $ME ${HERE_STEPS[*]} | ${THEM:-$PEER} ${THERE_STEPS[*]}"
  # The head into the other's clone over the link, under a ref of its own that goes once it is checked out.
  SYNCED= start=$SECONDS
  if [ -n "$THEM" ] && [ ${#THERE_STEPS[@]} -gt 0 ]; then
    # shellcheck disable=SC2029
    if GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=5" git push -q --no-verify --force "$PEER:$PEER_REPO" "$SHA:refs/check/$SHA" >> "$LOG" 2>&1 &&
      "${SSH[@]}" "set -e
        git -C $PEER_REPO worktree list --porcelain | grep -qx 'worktree $PEER_CHECK' || git -C $PEER_REPO worktree add -q --detach $PEER_CHECK $SHA
        git -C $PEER_CHECK checkout -q --detach --force $SHA; git -C $PEER_CHECK clean -fdq
        git -C $PEER_REPO update-ref -d refs/check/$SHA
        [ \"\$(git -C $PEER_CHECK rev-parse HEAD)\" = $SHA ]" >> "$LOG" 2>&1; then
      SYNCED=1
    fi
  fi
  SYNC=$((SECONDS - start))
  # the caller's own ZSPARK_* and the flags reach the other half. Bare runs forward no ZSPARK_* overrides.
  FORWARD=
  for name in $(sed -n 's/^declare -x \(ZSPARK_[A-Z0-9_]*\)=.*/\1/p' <<<"$callers"); do FORWARD+=" $(printf '%q' "$name=${!name:-}")"; done
  FLAGS=
  [ -n "$DEBUG" ] && FLAGS+=" --debug"
  [ -n "$BARE" ] && FLAGS+=" --bare"
  OUT_HERE=$CACHE/logs/$STAMP-full.$ME.out
  OUT_THERE=$CACHE/logs/$STAMP-full.${THEM:-peer}.out
  : > "$OUT_THERE"
  REMOTE= THERE_START=$SECONDS
  if [ -n "$SYNCED" ]; then
    # shellcheck disable=SC2029,SC2086
    "${SSH[@]}" "cd $PEER_CHECK && exec env CHECK_STAMP=$STAMP$FORWARD setsid -w tools/check.sh$FLAGS half ${THERE_STEPS[*]}" > "$OUT_THERE" 2>&1 &
    REMOTE=$!
  elif [ ${#THERE_STEPS[@]} -gt 0 ]; then
    say "${THEM:-$PEER}: $([ -n "$THEM" ] && echo "its checkout of $SHA failed (the log says why)" || echo "does not answer"); its half runs here after this one"
  fi
  LOCAL=
  stop() {
    [ -n "$REMOTE" ] && "${SSH[@]}" "f=\$HOME/.cache/zspark/full-half-$STAMP.pid; [ -f \$f ] && kill -TERM -- -\$(cat \$f)" 2> /dev/null
    [ -n "$LOCAL" ] && [ -f "$CACHE/full-half-$STAMP.pid" ] && kill -TERM -- -"$(cat "$CACHE/full-half-$STAMP.pid")" 2> /dev/null
    say "full: stopped, both halves ended"; exit 130
  }
  trap stop INT TERM
  # this half in a session of its own, so that stopping `full` ends its tests too
  HERE_START=$SECONDS
  # shellcheck disable=SC2086
  env CHECK_STAMP="$STAMP" setsid -w "$0"$FLAGS half "${HERE_STEPS[@]}" > "$OUT_HERE" 2>&1 &
  LOCAL=$!
  wait $LOCAL; HERE_WAITED=$((SECONDS - HERE_START)); LOCAL=
  THERE_STATUS=
  if [ -n "$REMOTE" ]; then wait $REMOTE; THERE_STATUS=$?; THERE_WAITED=$((SECONDS - THERE_START)); REMOTE=; fi
  # what the other Spark left: its steps' logs and tables, at the same paths here
  [ -n "$SYNCED" ] && rsync -a -e "ssh -o BatchMode=yes" "$PEER:$CACHE/logs/$STAMP-full-*" "$CACHE/logs/" >> "$LOG" 2>&1
  # a step the other did not finish (no answer, the link lost, its GPU in use) runs here now, and is said to
  MISSING=()
  for step in "${THERE_STEPS[@]}"; do grep -q "^half: $step " "$OUT_THERE" || MISSING+=("$step"); done
  OUT_AFTER=$CACHE/logs/$STAMP-full.$ME-after.out
  if [ ${#MISSING[@]} -gt 0 ]; then
    if [ -n "$SYNCED" ]; then
      say "${THEM}: ${MISSING[*]} not run there (ssh exit $THERE_STATUS; its last lines below): run here now"
      grep -v '^half: ' "$OUT_THERE" | tail -5 | sed "s/^/  $THEM: /" | tee -a "$LOG"
    fi
    AFTER_START=$SECONDS
    # shellcheck disable=SC2086
    env CHECK_STAMP="$STAMP" setsid -w "$0"$FLAGS half "${MISSING[@]}" > "$OUT_AFTER" 2>&1 &
    LOCAL=$!
    wait $LOCAL; LOCAL=
    AFTER_SECONDS=$((SECONDS - AFTER_START))
  fi
  trap - INT TERM
  # each step's lines named with its Spark; the markers give the steps' sides and results
  FAILED=()
  SIDE=()
  report() { # SPARK OUT
    [ -s "$2" ] || return 0
    grep -v '^half: ' "$2" | sed "s/^/$1 /" | tee -a "$LOG"
    while read -r _ step result _; do
      SIDE+=("$step $1")
      [ "$result" = pass ] || FAILED+=("$1 $step")
    done < <(grep -E '^half: [a-z0-9_]+ (pass|fail) ' "$2")
  }
  report "$ME" "$OUT_HERE"
  report "${THEM:-$PEER}" "$OUT_THERE"
  report "$ME" "$OUT_AFTER"
  for step in "${HERE_STEPS[@]}" "${THERE_STEPS[@]}"; do
    printf '%s\n' "${SIDE[@]}" | grep -q "^$step " || FAILED+=("$step not run")
  done
  compiled() { sed -n 's/^half: compiled //p' "$1" 2> /dev/null; }
  # a half's own wall time, as it measured it; the other's, where it said none, as seen from here
  ran() { sed -n 's/^half: ran //p' "$1" 2> /dev/null; }
  HERE_SECONDS=$(ran "$OUT_HERE"); HERE_SECONDS=${HERE_SECONDS:-$HERE_WAITED}
  [ -n "${THERE_WAITED:-}" ] && { THERE_SECONDS=$(ran "$OUT_THERE"); THERE_SECONDS=${THERE_SECONDS:-$THERE_WAITED}; }
  say "halves: $ME ${HERE_SECONDS} s (compile $(compiled "$OUT_HERE") s)${THERE_SECONDS:+; $THEM ${THERE_SECONDS} s (compile $(compiled "$OUT_THERE") s; its checkout $SYNC s before)}${AFTER_SECONDS:+; ${MISSING[*]} after, here, ${AFTER_SECONDS} s}; $SECONDS s in all"
  # the slowest tests of both halves together, each with its step and Spark
  {
    echo; echo "check.sh: the 20 slowest of both halves' tests ($CACHE/logs/$STAMP-full-*.times):"
    for entry in "${SIDE[@]}"; do
      read -r step spark <<<"$entry"
      cat "$CACHE/logs/$STAMP-full-$step.times"/*.tsv 2> /dev/null | awk -F '\t' -v at="$spark $step" 'BEGIN { OFS = "\t" } { print $1, $2, at, $3, $4 }'
    done > "$LOG.all"
    sort -t $'\t' -k1,1 -g -r "$LOG.all" | head -20 | awk -F '\t' '{ printf "%10.1f s  %-4s  %-22s %s\n", $1 / 1000, $2, $3, $4 }'
    kept=$(awk -F '\t' '$5 != "" && $5 < -1024' "$LOG.all" | sort -t $'\t' -k5,5 -g | head -10 | awk -F '\t' '{ printf "%10.1f GiB  %-22s %s\n", $5 / 1024, $3, $4 }')
    [ -n "$kept" ] && { echo "check.sh: the tests that lost over 1 GiB of MemAvailable:"; echo "$kept"; }
    rm -f "$LOG.all"
  } >> "$LOG"
  if [ ${#FAILED[@]} -gt 0 ]; then
    say "full FAILED: $(printf '%s, ' "${FAILED[@]}" | sed 's/, $//')"
    echo "log: $LOG"
    exit 1
  fi
  say "full passed: ${#SIDE[@]} steps, on $(printf '%s\n' "${SIDE[@]}" | cut -d' ' -f2 | sort -u | paste -sd' ')"
  if [ -z "${SPLIT:-}" ]; then pass_record "$MARK" "$LOG" || exit 1; fi
  echo "log: $LOG"
}
