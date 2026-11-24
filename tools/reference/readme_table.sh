#!/bin/bash
# One `zspark generate` load, every row of README's first measured table (spec 049). The prompts are rendered
# before the lock; the load holds the machine alone (`machine.lock` whole, `gpu.lock` inside).
#
#   tools/reference/readme_table.sh CHECKPOINT_DIR OUT_DIR
#
# OUT_DIR gets cases.txt, generate.txt, meta.json and table.md. SCORE is the ids file (the reference's
# score_ids.txt). ZSPARK is a `-Doptimize=fast` build. The prompt windows are the session's: 2048.
set -euo pipefail
if [ $# -ne 2 ]; then sed -n 2,8p "$0"; exit 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
CKPT=$(realpath "$1"); OUT=$2
ZSPARK=${ZSPARK:-$REPO/zig-out/bin/zspark}
SCORE=${SCORE:-$HOME/.cache/zspark/reference/score_ids.txt}
export ZSPARK
# shellcheck source=tools/reference/serve_lib.sh
. "$HERE/serve_lib.sh"
if [ -z "${README_TABLE_LOCKED:-}" ]; then
  mkdir -p "$OUT"
  python3 "$HERE/readme_table.py" prepare "$CKPT" "$OUT"
fi
serve_locked README_TABLE_LOCKED "$@"
mkdir -p "$OUT"
{ date -Is; uptime; free -m; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; } > "$OUT/before.txt"
serve_gpu_free
python3 "$HERE/readme_table.py" measure "$CKPT" "$OUT" "$SCORE"
{ date -Is; uptime; free -m; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; } > "$OUT/after.txt"
echo "readme table: $OUT"
