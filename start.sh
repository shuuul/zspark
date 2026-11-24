#!/usr/bin/env bash
# start.sh — start `zspark stack` in the background: Flash Next (and the OCR and ASR
# checkpoints the configuration names) as their own serves behind zspark-router.
#
#   ./start.sh                  the configured stack; already running is said, not an error
#   ./start.sh ARG..            this once, those words as `zspark stack`'s own
#                               (FLASH_DIR .. [-- ..]), the configuration left out
#   ./start.sh restart [ARG..]  ./stop.sh, then start again (a changed zspark.local.sh)
#   DRY_RUN=1 ./start.sh        print the stack's command, start nothing (restart too
#                               signals nothing)
#   ./start.sh help
#
# The stack's words come from zspark.local.sh's STACK_ARGS (zspark.local.example.sh has
# every flag and where the checkpoints' downloads land), or from the command line, which
# replaces them. The defaults stay the stack's: one Spark, 127.0.0.1:8888, Flash Next
# alone; --peer, --ocr, --asr, --listen and the serve flags after -- are all opt-ins.
#
# The stack runs detached, its lines in LOG_DIR/stack-*.log (LOG_DIR/stack-latest.log
# links the newest; ~/.cache/zspark/logs unset) and its pid in a per-checkout STATE_DIR's
# stack.pid with the process's start tick beside it (~/.cache/zspark/stack-CKSUM unset,
# the sum this tree's path hashes to). stop.sh signals that process alone: a pid file
# whose start tick or argv no longer matches — the pid reused by another program — is
# stale and dropped, never signalled, and a live record that is not this tree's stack
# is refused, never dropped. start.sh returns once the stack is spawned; the models
# load behind it, and the router answers on --listen when every configured model is
# healthy (the log says each wave).
#
# ZSPARK_BIN names the zspark binary (./zig-out/bin/zspark unset, zspark-router beside
# it: zig build). STOP_S (120) is stop.sh's wait. Nothing here builds or downloads.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$ROOT/stack_lib.sh"

usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

restart=0
case ${1:-} in
  help|-h|--help) usage 0 ;;
  restart) restart=1; shift ;;
esac
[ $# -gt 0 ] && STACK_ARGS=("$@")
declare -p STACK_ARGS 2>/dev/null | grep -q '^declare \-a' || { echo "start.sh: zspark.local.sh must set STACK_ARGS as an array: STACK_ARGS=(DIR FLAG..)" >&2; exit 2; }
if [ ${#STACK_ARGS[@]} -eq 0 ]; then
  [ "$restart" = 1 ] || { echo "start.sh: no stack arguments: copy zspark.local.example.sh to zspark.local.sh and set STACK_ARGS, or pass them here" >&2; usage 2; }
  echo "start.sh: nothing to restart with" >&2; exit 2
fi
case ${STACK_ARGS[0]} in -h|--help) usage 0 ;; --*) echo "start.sh: the first word is FLASH_DIR; stack flags come after it" >&2; exit 2 ;; esac
if [ "$restart" = 1 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN: restart would run ./stop.sh first; signalling nothing"
  else
    "$ROOT/stop.sh" || exit $?
  fi
fi

if [ "$DRY_RUN" != 1 ]; then
  [ -x "$ZSPARK_BIN" ] || { echo "start.sh: no $ZSPARK_BIN: zig build" >&2; exit 2; }
  [ -x "$(dirname "$ZSPARK_BIN")/zspark-router" ] || { echo "start.sh: no zspark-router beside $ZSPARK_BIN: zig build" >&2; exit 2; }
fi

cmd=("$ZSPARK_BIN" stack "${STACK_ARGS[@]}")
if [ "$DRY_RUN" = 1 ]; then
  printf 'DRY_RUN: would run detached, log under %s, pid under %s:\n  ' "$LOG_DIR" "$STATE_DIR"
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 9>"$LOCK"
flock -w 60 9 || { echo "start.sh: $LOCK busy (another start.sh or stop.sh holding it over 60 s)" >&2; exit 1; }

record=$(stack_record "$PID_FILE")
case $record in
  ours*)
    set -- $record
    echo "start.sh: the stack already runs (pid $2, log $(readlink -f "$LOG_DIR/stack-latest.log" 2>/dev/null || echo "$LOG_DIR")); ./stop.sh stops it"
    exit 0 ;;
  foreign*)
    set -- $record
    echo "start.sh: $PID_FILE's pid $2 is a running process that is not this tree's 'zspark stack' ($(pid_argv "$2" 0 2>/dev/null | head -c 120)): leaving it and the record alone" >&2
    exit 1 ;;
  malformed)
    echo "start.sh: $PID_FILE is malformed: dropping it without signalling anything" >&2
    rm -f "$PID_FILE" ;;
  dead)
    rm -f "$PID_FILE" ;;
  none) ;;
esac

log="$LOG_DIR/stack-$(date +%Y%m%d-%H%M%S).log"
ln -sfn "$log" "$LOG_DIR/stack-latest.log"
nohup "${cmd[@]}" >>"$log" 2>&1 </dev/null 9>&- &
pid=$!
tick=
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  [ -d "/proc/$pid" ] || break
  [ "$(pid_state "$pid" 2>/dev/null)" = Z ] && break
  t=$(pid_tick "$pid" 2>/dev/null) && is_our_stack "$pid" "$t" && { tick=$t; break; }
  unset t
  sleep 0.2
done
if [ -z "$tick" ]; then
  echo "start.sh: the stack exited or never exec'd; its log: $log" >&2
  cat "$log" >&2
  exit 1
fi
printf '%s %s\n' "$pid" "$tick" >"$PID_FILE"
printf '%q ' "${cmd[@]}" >"$ARGS_FILE"; printf '\n' >>"$ARGS_FILE"
echo "start.sh: stack started (pid $pid): $log"
echo "start.sh: 'started', not 'ready': the models load behind it; the router answers once every configured model is healthy. Follow: tail -f $log"
