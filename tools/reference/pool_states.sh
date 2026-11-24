#!/bin/bash
# One load of `zspark serve` under its defaults, measured by tools/reference/pool_states.py (spec 054): what a
# conversation holds in the session and in the shared pool, and the seconds to a request's first token. It
# judges nothing. The machine is held alone (`machine.lock` whole, `gpu.lock` inside), as client_bench.sh
# holds it; the server listens on 127.0.0.1:18888.
#
#   tools/reference/pool_states.sh CHECKPOINT_DIR OUT_DIR [serve args...]
#
# OUT_DIR gets server.log, states.json (every request), table.md, before.txt and after.txt (`uptime`), and the
# device memory after the load, at the end and at its most (`nvidia-smi` asked every 100 ms: mem_trace). ZSPARK
# is the binary (a `-Doptimize=fast` build of the clean head: `tools/check.sh build`); SIZES, REPEATS, CUT and
# ONLY reach pool_states.py as its options.
set -uo pipefail
if [ $# -lt 2 ]; then sed -n 2,11p "$0"; exit 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=tools/reference/serve_lib.sh
. "$HERE/serve_lib.sh"
serve_locked POOL_STATES_LOCKED "$@"
CK=$(realpath "$1"); O=$2; shift 2
Z=${ZSPARK:-$HERE/../../zig-out/fast/bin/zspark}
BASE=http://127.0.0.1:18888
serve_gpu_free
mkdir -p "$O"; rm -f "$O"/states.json "$O"/table.md "$O"/mem_* "$O"/server.log
{ date -Is; uptime; "$Z" version; } > "$O/before.txt"
START=$(date +%s.%N)
serve_start 127.0.0.1 18888 "$O/server.log" "$Z" "$CK" --name m "$@"
trap 'serve_stop INT' EXIT
serve_wait $BASE/health 600 1
if [ $? = 1 ]; then echo "the server ended: $O/server.log" >&2; exit 1; fi
echo "load $(echo "$(date +%s.%N)-$START" | bc) s" > "$O/load"
mem() { nvidia-smi --query-compute-apps=used_memory --format=csv,noheader > "$O/mem_$1"; }
mem load
nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits -lms 100 > "$O/mem_trace" 2>/dev/null &
TRACE=$!
python3 "$HERE/pool_states.py" --url $BASE --log "$O/server.log" --out "$O/states.json" \
  --sizes "${SIZES:-300,6000}" --repeats "${REPEATS:-3}" --cut "${CUT:-600}" ${ONLY:+--only "$ONLY"} \
  > "$O/run.txt" 2> "$O/run.err"
STATUS=$?
mem end
kill "$TRACE" 2>/dev/null; wait "$TRACE" 2>/dev/null
sort -n "$O/mem_trace" | tail -1 > "$O/mem_peak"
trap - EXIT
serve_stop INT
{ date -Is; uptime; } > "$O/after.txt"
sed -n '/^| phase/,$p' "$O/run.txt" > "$O/table.md"
cat "$O/table.md"
echo "memory: after load $(cat "$O/mem_load"), at the end $(cat "$O/mem_end"), at its most $(cat "$O/mem_peak") MiB; $(cat "$O/load"); $(sed -n 2p "$O/before.txt")"
grep -E "the pool's device buffers|shared pool is open|states held" "$O/server.log" | tail -3
[ $STATUS = 0 ] || { echo "pool_states.py failed: $O/run.err" >&2; tail -3 "$O/run.err" >&2; exit 1; }
