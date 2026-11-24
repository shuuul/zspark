#!/bin/bash
# One load of `zspark serve` measured by tools/reference/spill_states.py (spec 074): what a conversation that
# left the pool costs its next prompt, with the states on disk (`--spill-gib`) and without. It judges nothing.
# The machine is held alone (`machine.lock` whole, `gpu.lock` inside), as pool_states.sh holds it; the server
# listens on 127.0.0.1:18888.
#
#   PHASE=leave tools/reference/spill_states.sh CHECKPOINT_DIR OUT_DIR [serve args...]
#   PHASE=restart FROM=OUT_DIR_OF_A_LEAVE/states.json tools/reference/spill_states.sh CHECKPOINT_DIR OUT_DIR [serve args...]
#
# OUT_DIR gets server.log, states.json, run.txt, before.txt and after.txt, `load` and `stop` (the seconds each
# took: a stop writes what the server holds). ZSPARK is the binary (a `-Doptimize=fast` build of the clean
# head: `tools/check.sh build`); SIZE and CONVERSATIONS reach spill_states.py. COLD=DIR asks the kernel to
# drop DIR's files from its page cache before the load, so a state is read from the disk (`cold.txt`: the
# pages `fincore` still finds).
set -uo pipefail
if [ $# -lt 2 ]; then sed -n 2,14p "$0"; exit 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=tools/reference/serve_lib.sh
. "$HERE/serve_lib.sh"
serve_locked SPILL_STATES_LOCKED "$@"
CK=$(realpath "$1"); O=$2; shift 2
Z=${ZSPARK:-$HERE/../../zig-out/fast/bin/zspark}
BASE=http://127.0.0.1:18888
PHASE=${PHASE:-leave}
serve_gpu_free
mkdir -p "$O"; rm -f "$O"/states.json "$O"/server.log "$O"/cold.txt
{ date -Is; uptime; "$Z" version; echo "serve $CK --name m $*"; } > "$O/before.txt"
if [ -n "${COLD:-}" ]; then
  find "$COLD" \( -name '*.st' -o -name '*.blk' \) -exec dd if={} iflag=nocache count=0 status=none \;
  find "$COLD" \( -name '*.st' -o -name '*.blk' \) -exec fincore {} + > "$O/cold.txt" 2>&1 || true
fi
START=$(date +%s.%N)
serve_start 127.0.0.1 18888 "$O/server.log" "$Z" "$CK" --name m "$@"
trap 'serve_stop INT' EXIT
serve_wait $BASE/health 600 1
if [ $? = 1 ]; then echo "the server ended: $O/server.log" >&2; exit 1; fi
echo "load $(echo "$(date +%s.%N)-$START" | bc) s" > "$O/load"
python3 "$HERE/spill_states.py" --phase "$PHASE" --url $BASE --log "$O/server.log" --out "$O/states.json" \
  --size "${SIZE:-16000}" --conversations "${CONVERSATIONS:-8}" ${FROM:+--from "$FROM"} \
  > "$O/run.txt" 2> "$O/run.err"
STATUS=$?
trap - EXIT
START=$(date +%s.%N)
serve_stop INT
echo "stop $(echo "$(date +%s.%N)-$START" | bc) s" > "$O/stop"
{ date -Is; uptime; } > "$O/after.txt"
cat "$O/run.txt"
echo "$(cat "$O/load"); $(cat "$O/stop"); $(sed -n 2p "$O/before.txt")"
grep -E "zspark: switches:|shared pool is open|states are kept|is written|is read|is not " "$O/server.log" | tail -40
[ $STATUS = 0 ] || { echo "spill_states.py failed: $O/run.err" >&2; tail -3 "$O/run.err" >&2; exit 1; }
