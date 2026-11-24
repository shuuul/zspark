# What the scripts that load a model share (spec 053), sourced by client_bench.sh, shared_acceptance.sh and
# readme_table.sh: the locks' re-exec, the check that the GPU is free, the refusal of the LAN's model endpoint,
# and `zspark serve` started, waited for and stopped. Each function prints what its caller printed when it held
# the lines itself; a caller keeps its own messages where the three differed.
#
#   serve_locked NAME "$@"        run the calling script again under the locks, unless NAME is set
#   serve_gpu_free                exit 1 when a process holds the GPU
#   serve_refuse_lan HOST PORT    exit 2 on the configured LAN model endpoint
#   serve_start HOST PORT LOG ZSPARK CKPT [serve args...]    SERVE_PID
#   serve_wait URL TRIES SECONDS  0: URL answered; 1: the server ended; 2: it never answered
#   serve_stop SIGNAL             the signal, then the wait for the process

# The machine alone (`machine.lock` whole, `gpu.lock` inside) for the whole script: NAME, exported, says the
# script already runs under them.
serve_locked() {
  local name=$1; shift
  if [ -n "${!name:-}" ]; then return 0; fi
  export "$name=1"
  exec flock -x ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock "$0" "$@"
}

serve_gpu_free() {
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "the GPU is in use: nothing is started" >&2; exit 1
  fi
}

# Benchmarks must leave the LAN model endpoint available to other clients.
. "$(dirname "${BASH_SOURCE[0]}")/../local_env.sh"
load_local_env "$(dirname "${BASH_SOURCE[0]}")/../.." || exit $?

serve_refuse_lan() {
  local host=$1 port=$2 protected_host=${MODEL_LAN_HOST:-}
  [ "$port" = "${MODEL_LAN_PORT:-8888}" ] || return 0
  if [ -z "$protected_host" ]; then
    case $host in 127.*|localhost|::1) return 0 ;; esac
    echo 'Set MODEL_LAN_HOST in .env.local before benchmarking on a LAN at the model port' >&2
    exit 2
  fi
  if [ "$host" = "$protected_host" ] ||
     { [ "$host" = 0.0.0.0 ] && ip -4 -o addr | awk -v host="$protected_host" '
         { split($4, addr, "/"); if (addr[1] == host) found = 1 } END { exit !found }'; }; then
    echo "$protected_host:$port is the LAN's model endpoint: a bench does not listen on it" >&2
    exit 2
  fi
}

# `ZSPARK serve CKPT --host HOST --port PORT [serve args...]` in the background, its output in LOG.
serve_start() {
  local host=$1 port=$2 log=$3 zspark=$4 ckpt=$5; shift 5
  serve_refuse_lan "$host" "$port"
  "$zspark" serve "$ckpt" --host "$host" --port "$port" "$@" > "$log" 2>&1 &
  SERVE_PID=$!
}

# Asks URL every 0.2 s, TRIES times, each answer waited for SECONDS. SERVE_PID may be another server's.
serve_wait() {
  local url=$1 tries=$2 each=$3
  for _ in $(seq 1 "$tries"); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then return 1; fi
    if curl -s -m "$each" -o /dev/null "$url"; then return 0; fi
    sleep 0.2
  done
  return 2
}

serve_stop() {
  kill "-$1" "$SERVE_PID" 2>/dev/null || true
  wait "$SERVE_PID" 2>/dev/null || true
}
