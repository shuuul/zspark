# stack_lib.sh — sourced by start.sh and stop.sh, never run alone. ROOT must be set.
#
# What both wrappers share: the machine's configuration file, the state and log paths,
# the zspark binary the identity check compares argv[0] against, and that check itself —
# the pid, the start tick /proc/PID/stat gave it, and argv (this zspark's "stack"), so a
# pid since reused by another program is never signalled.

# The user's file, where the variables below are set where the environment did not
# (the environment wins: a one-off `STATE_DIR=... ./stop.sh` beats the file), and
# STACK_ARGS, start.sh's `zspark stack` words (zspark.local.example.sh has them all).
. "$ROOT/tools/local_env.sh"
load_local_env "$ROOT" || exit $?
declare -A _env_keep
for _v in STATE_DIR LOG_DIR STOP_S DRY_RUN ZSPARK_BIN; do
  [[ -v $_v ]] && _env_keep[$_v]=${!_v}
done
[ -f "$ROOT/zspark.local.sh" ] && . "$ROOT/zspark.local.sh"
for _v in "${!_env_keep[@]}"; do printf -v "$_v" '%s' "${_env_keep[$_v]}"; done
unset _env_keep _v

# State lives under a per-checkout directory: a second zspark checkout has its own and
# neither drops the other's record nor reads it. An operator who sets one STATE_DIR for
# two checkouts is answered by "foreign", below, not by a dropped record.
STATE_DIR=${STATE_DIR:-"$HOME/.cache/zspark/stack-$(echo -n "$ROOT" | cksum | cut -d' ' -f1)"}
LOG_DIR=${LOG_DIR:-"$HOME/.cache/zspark/logs"}
PID_FILE=$STATE_DIR/stack.pid
ARGS_FILE=$STATE_DIR/stack.argv
LOCK=$STATE_DIR/stack.lock
STOP_S=${STOP_S:-120}
DRY_RUN=${DRY_RUN:-0}

ZSPARK_BIN=${ZSPARK_BIN:-"$ROOT/zig-out/bin/zspark"}
ZSPARK_BIN=$(readlink -f "$ZSPARK_BIN" 2>/dev/null || echo "$ZSPARK_BIN")

# The process's start tick (field 22 of /proc/PID/stat, after the ") " that ends comm):
# the tell a pid reuse cannot pass — a new process on an old pid has another tick.
pid_tick() { local s; s=$(<"/proc/$1/stat") || return 1; s=${s##*) }; set -- $s; echo "${20}"; }
pid_state() { local s; s=$(<"/proc/$1/stat") || return 1; s=${s##*) }; set -- $s; echo "$1"; }

# argv word N of a process, read NUL-separated (a path's own bytes, newline included,
# never split): the word or nothing where the process or the word is not there.
pid_argv() { local pid=$1 n=$2 i=0 word
  while IFS= read -r -d '' word; do
    if [ "$i" = "$n" ]; then printf '%s' "$word"; return 0; fi
    i=$((i + 1))
  done <"/proc/$pid/cmdline"
  return 1
}

# The process start.sh recorded is still that process — alive, not a zombie, its tick
# unchanged — and its argv is this tree's "zspark stack": either argv0 the binary itself
# (the native build) or argv1 (an interpreter's argv0 keeps the script path), so a
# scripted ZSPARK_BIN identifies the same way.
is_our_stack() { local tick=$2
  [ -d "/proc/$1" ] || return 1
  [ "$(pid_state "$1" 2>/dev/null)" != Z ] || return 1
  [ "$(pid_tick "$1" 2>/dev/null)" = "$tick" ] || return 1
  if [ "$(readlink -f "$(pid_argv "$1" 0)" 2>/dev/null)" = "$ZSPARK_BIN" ]; then
    [ "$(pid_argv "$1" 1)" = stack ]; return
  fi
  [ "$(readlink -f "$(pid_argv "$1" 1)" 2>/dev/null)" = "$ZSPARK_BIN" ] &&
  [ "$(pid_argv "$1" 2)" = stack ]
}
# Classify the recorded process without mutating state or signalling it.
stack_record() {
  local pid tick
  [ -f "$1" ] || { echo none; return; }
  read -r pid tick <"$1" 2>/dev/null || { echo malformed; return; }
  if [[ ! $pid =~ ^[1-9][0-9]*$ || ! $tick =~ ^[0-9]+$ ]]; then echo malformed; return; fi
  if [ ! -d "/proc/$pid" ] || [ "$(pid_state "$pid" 2>/dev/null)" = Z ] || [ "$(pid_tick "$pid" 2>/dev/null)" != "$tick" ]; then
    echo dead; return
  fi
  if is_our_stack "$pid" "$tick"; then echo "ours $pid $tick"; else echo "foreign $pid $tick"; fi
}
