# The checkpoint verbs, sourced by check.sh (spec 171): `ranks`, `accept` and `bench`, each a measurement
# against a checkpoint on the GPU with its own lock discipline, none of it inside this script's holds.
# `busy` says what holds the GPU; `checkpoint` resolves a name or a directory to a directory on this
# machine.

busy() { nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null; }
checkpoint() { # glm-fp4-local or a directory: the directory
  local dir
  case $1 in
    glm-fp4-local) dir=${ZSPARK_GLM_FP4_LOCAL:-} ;;
    *) dir=$1 ;;
  esac
  if [ -z "$dir" ] || [ ! -d "$dir" ]; then echo "$1: no such checkpoint on this machine (see $CACHE/env.sh)" >&2; exit 2; fi
  echo "$dir"
}

verb_ranks() {
  # The two ranks' checks, each run by two_ranks.sh, which takes both machines' locks itself and refuses
  # binaries that differ: nothing here is run under this script's locks (as `accept` and `bench`). The
  # binary is the safe build the two ranks run, over NCCL as two_ranks.sh says, held to the head.
  # rank 1 opens the same path on its machine: the default is one both Sparks hold, not
  # $ZSPARK_FLASH_NEXT_FP4_LOCAL, whose path may differ between them; ZSPARK_RANKS_DIR, from env.sh or the
  # caller, wins
  CKPT=$(checkpoint "${1:-${ZSPARK_RANKS_DIR:-$HOME/models/Qwen3.8-Flash-Next-FP4-Local}}") || exit 2
  # An unavailable rank 1 skips the tier before it needs a build.
  if [ -z "${PEER:-}" ]; then
    echo 'ranks: PEER is unset (see .env.local.example): skipped'; exit 0
  fi
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" true 2>/dev/null; then
    echo "ranks: no rank 1 on $PEER: skipped"; exit 0
  fi
  # a checkpoint's own reference, at one path on both Sparks: FP4-Local's the vLLM container's replies (no
  # TensorFold reads it, spec 105); none for another checkpoint
  RREF=
  GLM=
  real=$(realpath "$CKPT")
  if [ -n "${ZSPARK_FLASH_NEXT_FP4_LOCAL:-}" ] && [ "$real" = "$(realpath "$ZSPARK_FLASH_NEXT_FP4_LOCAL")" ]; then
    RREF=${ZSPARK_REFERENCE_FP4_LOCAL:-}
  elif [ -n "${ZSPARK_GLM_FP4_LOCAL:-}" ] && [ "$real" = "$(realpath "$ZSPARK_GLM_FP4_LOCAL")" ]; then
    RREF=${ZSPARK_REFERENCE_GLM_FP4_LOCAL:-}
    GLM=1
  fi
  RLOG=$CACHE/logs/$STAMP-ranks-build.log
  if ! flock -s "$ML" flock -s "$CL" zig build -Dnccl="${NCCL:-/usr}" --summary all > "$RLOG" 2>&1; then
    echo "build FAILED" >&2; grep -n -E 'error:' "$RLOG" | head -12 >&2; echo "log: $RLOG" >&2; exit 1
  fi
  RZ=$REPO/zig-out/bin/zspark
  RHEAD=$(git -C "$REPO" rev-parse HEAD)
  RSTAMP=$("$RZ" version 2>/dev/null) || RSTAMP=
  if [ "$(echo "$RSTAMP" | awk '{ print $3 }')" != "$RHEAD" ] || [ -n "$(echo "$RSTAMP" | awk '{ print $4 }')" ]; then
    echo "refused: $RZ is \"${RSTAMP:-not stamped}\" and the head is $RHEAD: commit or stash the change" >&2
    exit 1
  fi
  # the peer's clone is the main tree's path, not this worktree's (two_ranks.sh defaults PEER_REPO to $REPO)
  [ -n "${PEER_REPO:-}" ] || PEER_REPO=$(git -C "$REPO" worktree list --porcelain | sed -n 's/^worktree //p' | head -1)
  export PEER_REPO
  RSTATUS=0
  RAN=0
  RUNS=("ranks-check random model=flash-next session" "ranks-check random model=flash-next states"
    "ranks-check random model=flash-next hall" "ranks-check $CKPT hall" "ranks-check $CKPT windows"
    "ranks-check $CKPT reference=$RREF"
    "replies $CKPT" "replies $CKPT kv=fp8")
  # GLM FP4-Local (spec 107 WS-02): the whole model against the vLLM container, its own run of 10 to 20 min; FP8
  # caches, serve's default and the card's (spec 115)
  [ -n "$GLM" ] && RUNS=("ranks-check $CKPT kv=fp8 reference=$RREF")
  for runcmd in "${RUNS[@]}"; do
    case $runcmd in *reference=) [ -n "$RREF" ] || { echo "ranks: $CKPT's reference unset, its reference= check skipped"; continue; } ;; esac
    # the whole model's replies and score text take longer than two_ranks.sh's 300 s
    limit=${LIMIT:-300}
    case $runcmd in *reference=*) limit=${LIMIT:-1200} ;; esac
    [ -n "$GLM" ] && limit=${LIMIT:-2400}
    # shellcheck disable=SC2086
    LIMIT=$limit "$HERE/two_ranks.sh" $runcmd || { echo "ranks FAILED: $runcmd" >&2; RSTATUS=1; break; }
    echo "ranks: $runcmd passed"
  done
  exit $RSTATUS
}

verb_accept() {
  [ $# -ge 1 ] || usage
  NAME=$(basename "$1"); CKPT=$(checkpoint "$1") || exit 2; shift
  OUT=$CACHE/logs/$STAMP-accept-$NAME
  if [ -n "$(busy)" ]; then echo "the GPU is in use, nothing is started: $(busy)" >&2; exit 1; fi
  Z=$(binary) || exit 1
  mkdir -p "$OUT"
  if ZSPARK=$Z "$HERE/reference/shared_acceptance.sh" "$CKPT" "$OUT" "$@" > "$OUT/acceptance.txt" 2> "$OUT/acceptance.err"; then
    # A line a check, without the requests' seconds; the texts' hashes alone are OUT/hashes, for `diff`.
    awk '/^turn[12]:/ { head = $0; sub(/, usage .*/, "", head); rest = $0; held = ""
        while (match(rest, /"cached_tokens":[0-9]+/)) { held = held " " substr(rest, RSTART + 16, RLENGTH - 16); rest = substr(rest, RSTART + RLENGTH) }
        print head ", cached_tokens of the session and the pool:" held; next }
      /^(prompt [0-9]|memory)/' "$OUT/acceptance.txt"
    grep -E '^(prompt [0-9]|turn[12]:)' "$OUT/acceptance.txt" | sed -E 's/, (tokens|usage) .*//' > "$OUT/hashes"
    echo "$("$Z" version | cut -c1-27); log: $OUT (hashes, acceptance.txt, server.log)"
  else
    echo "accept FAILED" >&2; head -5 "$OUT/acceptance.err" >&2; tail -3 "$OUT/server.log" 2>/dev/null >&2
    echo "log: $OUT" >&2; exit 1
  fi
}

verb_bench() {
  [ $# -ge 1 ] || usage
  NAME=$(basename "$1"); CKPT=$(checkpoint "$1") || exit 2
  OUT=${2:-$CACHE/logs/$STAMP-bench-$NAME}
  if [ -n "$(busy)" ]; then echo "the GPU is in use, nothing is started: $(busy)" >&2; exit 1; fi
  Z=$(binary) || exit 1
  mkdir -p "$OUT"
  if ZSPARK=$Z "$HERE/reference/client_bench.sh" zspark "$CKPT" "$OUT" 1 > "$OUT/bench.txt" 2>&1; then
    # A line a request, as spec 053's base was written: what a later run is held against with `diff`.
    [ -f "$OUT/harness.json" ] && jq -r '.requests[] | [(.name + "#" + (.repeat | tostring)), .kind, .request_sha, .sha,
      .status, .prompt_tokens, .cached_tokens, .completion_tokens, .finish] | @tsv' "$OUT/harness.json" > "$OUT/hashes"
    tail -1 "$OUT/bench.txt"
    [ -f "$OUT/hashes" ] && echo "$(wc -l < "$OUT/hashes") requests, $(grep -c -v -P '\t200\t' "$OUT/hashes") not 200; sha256 of hashes $(sha256sum "$OUT/hashes" | cut -c1-16)"
    echo "$("$Z" version | cut -c1-27); log: $OUT (hashes, harness.json, meta.json, server.log)"
  else
    echo "bench FAILED" >&2; tail -5 "$OUT/bench.txt" >&2; echo "log: $OUT" >&2; exit 1
  fi
}
