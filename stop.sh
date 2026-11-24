#!/usr/bin/env bash
# stop.sh — stop the `zspark stack` ./start.sh started.
#
#   ./stop.sh            SIGTERM to the recorded stack: the same stop as Ctrl-C — the
#                        router first, then each model and its MPS daemon, and over
#                        --peer the other Spark's children when their ssh stdin
#                        closes — then a wait of STOP_S (120 unset). Still up after
#                        that is said, not forced: the record stays and ./stop.sh
#                        repeats. Nothing else is ever signalled.
#   DRY_RUN=1 ./stop.sh  say what the pid file would do, change nothing
#   ./stop.sh help
#
# Only the process start.sh recorded is signalled, and only while it is still that
# process: the pid's /proc start tick must equal the recorded one (a reused pid fails
# it) and argv must be this tree's zspark "stack" — rechecked under the lock just
# before the signal. A record whose process ended or whose pid was taken is stale:
# dropped, never signalled. A record naming a living process that is not this tree's
# stack is foreign (another checkout's, a hand-started one): refused, left alone.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$ROOT/stack_lib.sh"

for a in "$@"; do case $a in help|-h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; *) echo "stop.sh: $a: help lists the usage" >&2; exit 2 ;; esac; done
[[ $STOP_S =~ ^[0-9]+$ && $STOP_S -gt 0 ]] || { echo "stop.sh: STOP_S is seconds, a positive number, not $STOP_S" >&2; exit 2; }

if [ "$DRY_RUN" != 1 ]; then
  # The lock before any read or mutation: a start or another stop between the read and
  # the act would change whom this is.
  mkdir -p "$STATE_DIR"
  exec 9>"$LOCK"
  flock -w 60 9 || { echo "stop.sh: $LOCK busy (another start.sh or stop.sh holding it over 60 s)" >&2; exit 1; }
fi

record=$(stack_record "$PID_FILE")
pid= tick=
case $record in
  ours*) set -- $record; pid=$2; tick=$3 ;;
  foreign*)
    set -- $record
    echo "stop.sh: $PID_FILE's pid $2 is a running process that is not this tree's 'zspark stack' ($(pid_argv "$2" 0 2>/dev/null | head -c 120)): leaving it and the record alone" >&2
    exit 1 ;;
  malformed)
    if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: $PID_FILE is malformed: it would be dropped, nothing signalled"; exit 0; fi
    echo "stop.sh: $PID_FILE is malformed: dropping it without signalling anything" >&2
    rm -f "$PID_FILE"
    exit 0 ;;
  dead)
    if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: $PID_FILE's process is gone: it would be dropped, nothing signalled"; exit 0; fi
    rm -f "$PID_FILE" "$ARGS_FILE"
    echo "stop.sh: the recorded process is gone; dropped the stale record: nothing to stop"
    exit 0 ;;
  none)
    echo "stop.sh: no stack of $ROOT is recorded: nothing to stop"
    exit 0 ;;
esac

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: would send SIGTERM to pid $pid ($(pid_argv "$pid" 0 2>/dev/null) stack), then wait up to ${STOP_S}s; nothing is signalled"
  exit 0
fi

# Re-read the record under the lock (the case above already ran under it for real
# runs): identity once more just before the signal.
is_our_stack "$pid" "$tick" || { echo "stop.sh: pid $pid is no longer the recorded stack; nothing signalled — run ./stop.sh again" >&2; exit 1; }

kill -TERM "$pid" 2>/dev/null || { rm -f "$PID_FILE" "$ARGS_FILE"; echo "stop.sh: pid $pid was already gone; dropped the record, nothing signalled"; exit 0; }
waited=0
while :; do
  # Still the same process: running (not a zombie) at the recorded tick.
  if [ "$(pid_state "$pid" 2>/dev/null)" = Z ] || [ "$(pid_tick "$pid" 2>/dev/null)" != "$tick" ]; then
    break
  fi
  if [ "$waited" -ge "$STOP_S" ]; then
    echo "stop.sh: pid $pid is still up after ${STOP_S}s; nothing more is signalled — ./stop.sh again retries, the log says why it waits" >&2
    exit 1
  fi
  sleep 1; waited=$((waited + 1))
done
rm -f "$PID_FILE" "$ARGS_FILE"
echo "stop.sh: stack stopped (pid $pid): the router, the serves, the MPS daemon and the peer's children ended with it"
