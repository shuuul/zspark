#!/bin/bash
# Two whole copies of `zspark serve` behind the router's one address (spec 088).
#
#   tools/two_copies.sh serve DIR [FLAG..]   the stack: a copy here on 127.0.0.1:18888, a copy on the other
#                                            Spark on its link address (198.51.100.2:18888, the owner's answer
#                                            to question 5; never the LAN's address), the router on
#                                            127.0.0.1:18887. LIMIT seconds (a day). SIGINT ends them,
#                                            the router first, then this copy, then the other's
#   tools/two_copies.sh check DIR [FLAG..]   the stack, then a conversation's two turns through the router
#                                            and two on the copy here alone, the same shape and no shared
#                                            word: each second turn's cached_tokens is printed, and the
#                                            router's is held equal to the copy's own (the WS-02 check)#
#   tools/two_copies.sh serve-stop DIR        the stack's stop, the same three patterns stop_stack ends
#                                            `serve` with: for a caller that ran `serve` in the
#                                            background, whose SIGINT the script never sees
#
# DIR is the checkpoint's directory here, COPY_DIR its path on the other Spark when it differs (default
# DIR, the same path on both); FLAG.. are the copies' own `serve` flags (--parallel 4 is the recipe's).
# ZSPARK (zig-out/bin/zspark) and ROUTER (zig-out/bin/zspark-router) name the binaries, as check.sh's
# ZSPARK= does — a timing builds -Doptimize=fast and says both. PEER (198.51.100.2) is the other Spark's link
# address, PEER_REPO the repository on it when its path differs, ROUTER_PORT (18887) and COPY_PORT (18888)
# the ports, COPY_B_HOST the address the other copy listens on (its link's, PEER). DRY_RUN=1 prints the
# commands, starts nothing.
#
# Locks, as two_ranks.sh takes them: machine.lock here and on the other Spark and gpu.lock inside each, all
# held for the run; MACHINE=-x holds both machines alone (a timing, WS-03's loads). A GPU another process
# holds stops the script before anything runs. Both machines run this repository's path's binaries, built;
# two copies whose commit differ are refused (a dirty tree is said, the check being no timing).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
. "$HERE/link_lib.sh"
. "$HERE/local_env.sh"
load_local_env "$REPO" || exit $?
CACHE=$HOME/.cache/zspark
PEER_REPO=${PEER_REPO:-$REPO}
ROUTER_PORT=${ROUTER_PORT:-18887}
COPY_PORT=${COPY_PORT:-18888}
export PATH=/home/linuxbrew/.linuxbrew/bin:$PATH
usage() { sed -n 2,14p "$0" | sed 's/^# \{0,1\}//'; exit 2; }

[ $# -ge 2 ] || usage
COMMAND=$1; shift
DIR=$1; shift
case $COMMAND in serve | check | serve-stop) ;; *) usage ;; esac
: "${PEER:?Set PEER in .env.local}"
COPY_B_HOST=${COPY_B_HOST:-$PEER}
LIMIT=${LIMIT:-86400}
MACHINE=${MACHINE:--s}
COPY_DIR=${COPY_DIR:-$DIR}

Z=${ZSPARK:-$REPO/zig-out/bin/zspark}
R=${ROUTER:-$REPO/zig-out/bin/zspark-router}
# the other Spark runs the same repository-relative binary ZSPARK names here
case $Z in "$REPO"/*) Z_PEER=$PEER_REPO/${Z#"$REPO"/} ;; *) Z_PEER=$Z ;; esac
[ -x "$Z" ] || { echo "two_copies.sh: no $Z: zig build" >&2; exit 2; }
[ -x "$R" ] || { echo "two_copies.sh: no $R: zig build" >&2; exit 2; }
if [ "$COMMAND" != serve-stop ]; then
  [ -d "$DIR" ] || { echo "two_copies.sh: no checkpoint directory $DIR" >&2; exit 2; }
fi

SSH=(ssh -o BatchMode=yes "$PEER")
if [ "$COMMAND" != serve-stop ]; then
  # a stop needs no binaries and no checkpoint anywhere; a start checks both sides' builds first
  here=$("$Z" version) || exit 1
  there=$("${SSH[@]}" "$Z_PEER version" 2>/dev/null) || { echo "two_copies.sh: no zspark at $PEER:$Z_PEER" >&2; exit 2; }
  # the commit is what the two copies must share; a dirty tree is said, this check being no timing
  commit() { printf '%s' "$1" | awk '{ print $3 }'; }
  [ "$(commit "$here")" = "$(commit "$there")" ] || { echo "two_copies.sh: '$here' here, '$there' on $PEER: build the same commit on both" >&2; exit 2; }
  [ "${here##* }" = dirty ] && echo "two_copies.sh: the copy here is a dirty build: $here" >&2
  "${SSH[@]}" test -d "$COPY_DIR" || { echo "two_copies.sh: $COPY_DIR is not on $PEER" >&2; exit 2; }
fi

free_gpu=$(gpu_guard two_copies.sh)

mkdir -p "$CACHE/logs"
LOG=$CACHE/logs/$(date +%Y%m%d-%H%M%S)-two-copies-$COMMAND
remote=
local_pid=
router_pid=

# The router first, then this copy, then the other's, whoever called for the end. A kill by the pid takes
# one link of the flock's and timeout's chain, so each process is ended by its own command line: the
# pattern is this run's exact argv, this machine's alone.
stop_stack() {
  [ -n "${router_pid:-}" ] && pkill -TERM -f "zspark-router --listen 127.0.0.1:$ROUTER_PORT" 2>/dev/null
  [ -n "${local_pid:-}" ] && pkill -TERM -f "zspark serve $DIR --host 127.0.0.1 --port $COPY_PORT" 2>/dev/null
  if [ -n "${remote:-}" ]; then
    # the session's end is the other copy's SIGHUP; the named command is this run's alone
    kill "$remote" 2>/dev/null
    "${SSH[@]}" "pkill -TERM -f 'zspark serve $COPY_DIR --host $COPY_B_HOST --port $COPY_PORT' 2>/dev/null" || true
  fi
  wait 2>/dev/null
  return 0
}
trap stop_stack EXIT
trap 'stop_stack; exit 130' INT TERM

# The copies take the locks with them, the GPU inside: their load is the run's own.
start_stack() { # the copies' flags; the pids left in remote, local_pid and router_pid
  local flags=$* theirs mine router
  theirs="timeout $((LIMIT + 60)) $Z_PEER serve $COPY_DIR --host $COPY_B_HOST --port $COPY_PORT $flags"
  mine="timeout $LIMIT $Z serve $DIR --host 127.0.0.1 --port $COPY_PORT $flags"
  router="timeout $LIMIT $R --listen 127.0.0.1:$ROUTER_PORT copy-a=127.0.0.1:$COPY_PORT copy-b=$COPY_B_HOST:$COPY_PORT"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "# on $PEER: flock $MACHINE ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock bash -c \"$(echo "$free_gpu; exec $theirs")\""
    echo "# here: flock $MACHINE $CACHE/machine.lock flock $CACHE/gpu.lock bash -c \"$(echo "$free_gpu; exec $mine")\""
    echo "# here: $router"
    echo "# the client talks to 127.0.0.1:$ROUTER_PORT"
    return 1
  fi
  # the other copy first: it is the one ssh has to reach
  # shellcheck disable=SC2029
  "${SSH[@]}" "flock $MACHINE ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock bash -c '$free_gpu
    exec $theirs'" > "$LOG.copy-b.log" 2>&1 &
  remote=$!
  # shellcheck disable=SC2016
  flock $MACHINE "$CACHE/machine.lock" flock "$CACHE/gpu.lock" bash -c "$free_gpu
    exec $mine" > "$LOG.copy-a.log" 2>&1 &
  local_pid=$!
  sleep 2
  kill -0 "$remote" 2>/dev/null || { echo "two_copies.sh: the copy on $PEER did not stay up: $LOG.copy-b.log" >&2; return 1; }
  kill -0 "$local_pid" 2>/dev/null || { echo "two_copies.sh: the copy here did not stay up: $LOG.copy-a.log" >&2; return 1; }
  wait_healthy "$local_pid" "127.0.0.1:$COPY_PORT" "$LOG.copy-a.log" || return 1
  "${SSH[@]}" "for i in \$(seq 300); do curl -sf http://$COPY_B_HOST:$COPY_PORT/health > /dev/null 2>&1 && exit 0; sleep 1; done; exit 1" \
    || { echo "two_copies.sh: the copy on $PEER never answered /health: $LOG.copy-b.log" >&2; return 1; }
  echo "two_copies.sh: both copies answer /health; the router next" >&2
  $router > "$LOG.router.log" 2>&1 &
  router_pid=$!
  for _ in $(seq 60); do curl -sf "http://127.0.0.1:$ROUTER_PORT/health" > /dev/null 2>&1 && return 0; kill -0 "$router_pid" 2>/dev/null || break; sleep 1; done
  echo "two_copies.sh: the router never answered: $LOG.router.log" >&2
  return 1
}

# This copy's /health, until it answers or three hundred seconds are gone.
wait_healthy() { # pid address log
  for _ in $(seq 300); do
    kill -0 "$1" 2>/dev/null || { echo "two_copies.sh: the copy at $2 died at once: $3" >&2; return 1; }
    curl -sf "http://$2/health" > /dev/null 2>&1 && return 0
    sleep 1
  done
  echo "two_copies.sh: the copy at $2 never answered /health: $3" >&2
  return 1
}

# A prompt of about four hundred tokens that names its subject: the two sides' prompts are the same shape
# and share no word of the content, so one conversation's rows are no other's.
prompt() { # subject: the text, not the request's body
  local line="The colour of the water changed with the hour, and the gulls kept their own accounts of it."
  local text="" i
  for i in $(seq 12); do text="$text $line"; done
  printf '%s Tell me of the %s in one sentence.' "$text" "$1"
}

# One conversation through the router, one on the copy here alone, the same shape and no shared word: the
# router's second turn's cached_tokens is the copy's own figure or the check fails.
check() {
  local router_t2= copy_t2= side port turn body out figure failed=0
  local first second reply
  for side in router copy; do
    port=$ROUTER_PORT
    [ $side = copy ] && port=$COPY_PORT
    [ $side = router ] && first=$(prompt sea) || first=$(prompt hill)
    second="And once more, in other words, of the same."
    turn=1
    while :; do
      if [ $turn = 1 ]; then
        body=$(jq -n --arg content "$first" '{messages:[{role:"user",content:$content}]}')
      else
        body=$(jq -n --arg content "$first" --arg reply "$reply" --arg next "$second" \
          '{messages:[{role:"user",content:$content},{role:"assistant",content:$reply},{role:"user",content:$next}]}')
      fi
      out=$(curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'content-type: application/json' -d "$body") \
        || { echo "check: the $side's turn $turn got no reply" >&2; failed=1; break; }
      figure=$(printf '%s' "$out" | jq -r '.usage.prompt_tokens_details.cached_tokens')
      printf 'check: %s turn %d cached_tokens %s\n' "$side" "$turn" "$figure"
      if [ -z "$figure" ] || [ "$figure" = null ]; then
        echo "check: the $side's turn $turn said no cached_tokens" >&2
        failed=1
        break
      fi
      if [ $turn = 2 ]; then
        [ $side = router ] && router_t2=$figure || copy_t2=$figure
        break
      fi
      reply=$(printf '%s' "$out" | jq -r '.choices[0].message.content')
      [ -n "$reply" ] || { echo "check: the $side's turn 1 said nothing" >&2; failed=1; break; }
      turn=2
    done
  done
  if [ -n "$router_t2" ] && [ "$router_t2" != "$copy_t2" ]; then
    echo "check FAILED: the router's second turn held $router_t2, the copy's own $copy_t2" >&2
    return 1
  fi
  [ "$failed" = 0 ] && echo "check: the router's second turn's cached_tokens is the copy's own ($router_t2)"
  return $failed
}

status=0
case $COMMAND in
  serve)
    start_stack "$@" || status=$?
    [ -n "${DRY_RUN:-}" ] && exit 0 # the stack was printed, nothing was started
    if [ $status = 0 ]; then
      wait "$router_pid" || status=$?
      echo "two_copies.sh: the stack is stopped; logs: $LOG.router.log $LOG.copy-a.log $LOG.copy-b.log" >&2
    fi
    ;;
  check)
    start_stack "$@" || status=$?
    [ -n "${DRY_RUN:-}" ] && exit 0
    if [ $status = 0 ]; then
      check || status=$?
    fi
    ;;
  serve-stop)
    # the stack's stop without a signal to this script: a caller that runs the stack in the background
    # ends it here, because SIGINT to a background job's shell is ignored on entry and the traps of
    # `serve` never fire in it. stop_stack's own guards are on `serve`'s pids, unset here, so the same
    # three patterns are killed by hand.
    pkill -TERM -f "zspark-router --listen 127.0.0.1:$ROUTER_PORT" 2>/dev/null || true
    pkill -TERM -f "zspark serve $DIR --host 127.0.0.1 --port $COPY_PORT" 2>/dev/null || true
    "${SSH[@]}" "pkill -TERM -f 'zspark serve $COPY_DIR --host $COPY_B_HOST --port $COPY_PORT' 2>/dev/null" || true
    echo "two_copies.sh: the stack is stopped; logs: $LOG.router.log $LOG.copy-a.log $LOG.copy-b.log" >&2
    ;;
esac
stop_stack
exit $status
