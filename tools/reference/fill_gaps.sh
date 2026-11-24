#!/bin/bash
# One load of `zspark serve` measured by tools/reference/fill_gaps.py (spec 074, WS-04): the waits between the
# deltas of replies that stream while a long prompt is filled beside them in the pool, and the prompt's own
# first token. It judges nothing. The machine is held alone (`machine.lock` whole, `gpu.lock` inside), as
# pool_states.sh holds it; the server listens on 127.0.0.1:18888.
#
#   tools/reference/fill_gaps.sh CHECKPOINT_DIR OUT_DIR [serve args...]
#   python3 tools/reference/fill_gaps.py --compare off=OUT_A/gaps.json on=OUT_B/gaps.json
#
# OUT_DIR gets server.log, gaps.json (every request and each delta's time), run.txt (the table), before.txt
# and after.txt (`uptime`, the binary, the flags). ZSPARK is the binary (a `-Doptimize=fast` build of the clean
# head: `tools/check.sh build`); SIZE, STREAMS, MAX_TOKENS, AFTER and REPEATS reach fill_gaps.py as its options.
set -uo pipefail
if [ $# -lt 2 ]; then sed -n 2,11p "$0"; exit 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=tools/reference/serve_lib.sh
. "$HERE/serve_lib.sh"
serve_locked FILL_GAPS_LOCKED "$@"
CK=$(realpath "$1"); O=$2; shift 2
Z=${ZSPARK:-$HERE/../../zig-out/fast/bin/zspark}
BASE=http://127.0.0.1:18888
serve_gpu_free
mkdir -p "$O"; rm -f "$O"/gaps.json "$O"/server.log
{ date -Is; uptime; "$Z" version; echo "serve $CK --name m $*"; } > "$O/before.txt"
START=$(date +%s.%N)
serve_start 127.0.0.1 18888 "$O/server.log" "$Z" "$CK" --name m "$@"
trap 'serve_stop INT' EXIT
serve_wait $BASE/health 600 1
if [ $? = 1 ]; then echo "the server ended: $O/server.log" >&2; exit 1; fi
echo "load $(echo "$(date +%s.%N)-$START" | bc) s" > "$O/load"
python3 "$HERE/fill_gaps.py" --url $BASE --log "$O/server.log" --out "$O/gaps.json" \
  --size "${SIZE:-25000}" --streams "${STREAMS:-3}" --max-tokens "${MAX_TOKENS:-512}" --after "${AFTER:-2}" \
  --repeats "${REPEATS:-2}" > "$O/run.txt" 2> "$O/run.err"
STATUS=$?
trap - EXIT
serve_stop INT
{ date -Is; uptime; } > "$O/after.txt"
cat "$O/run.txt"
echo "$(cat "$O/load"); $(sed -n 2p "$O/before.txt")"
grep -E "zspark: switches:|shared pool is open" "$O/server.log" | tail -4
[ $STATUS = 0 ] || { echo "fill_gaps.py failed: $O/run.err" >&2; tail -3 "$O/run.err" >&2; exit 1; }
