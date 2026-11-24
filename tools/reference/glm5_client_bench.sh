#!/bin/bash
# One load of GLM-5.3-Flash on two ranks, measured as a client sees it (spec 068): rank 0 here and rank 1 on the
# other Spark, `zspark serve` or TensorFold 0.6.6's server, tools/reference/client_bench.py against rank 0, the
# load's time and both machines' memory and `uptime`, and both ranks stopped. client_bench.sh is one Spark's.
#
#   tools/reference/glm5_client_bench.sh zspark|tensorfold OUT_DIR SEED [LABEL=MODEL ...]
#
# Each LABEL=MODEL is a pass of the harness on the one load, its requests asked of MODEL, its files in
# OUT_DIR/LABEL (harness.json, harness.txt, harness.log: rank 0's lines while it ran, uptime.txt: both machines'
# before and after). TensorFold's draft policy is a request's, after `@` in the model's name
# (`families/glm5_next/cuda/app.py`): `mtp=glm-5.3-flash@auto:2:8:0.03 dflash2=glm-5.3-flash` times the MTP layer
# and the drafter on one load (`auto` with the drafter is `fc5:0.3`, spelled out it stays the MTP layer's).
# zspark's is the load's: with DRAFTER the drafter drafts, without it the MTP layer. Without a pass: run=NAME.
#
# CKPT is the checkpoint (`ZSPARK_GLM_FP4_LOCAL` unless set; the EXL3 pack's default was retired, spec 121
# WS-00); CKPT1 is rank 1's, CKPT unless set (`EXTRA=--weights-from-peer CKPT1=/nonexistent`: rank 1 opens
# nothing of it). DRAFTER=1 loads `ZSPARK_GLM_DRAFTER` on both ranks (read where it is; nothing of it is written);
# DRAFTER1 is rank 1's path, the same unless set (with `--weights-from-peer` rank 1 is sent rank 0's drafter and
# opens nothing of it: `DRAFTER1=/nonexistent`).
# CONTEXT is zspark's `--context` (2051: the window TensorFold serves without `--context`, DSA's rows all dense)
# and is given to TensorFold only when TF_CONTEXT is set. ONLY names the requests (the default: what fits 2,051
# tokens of client_bench.json, the table the retired pack fitted; `ONLY=` the whole set, with CONTEXT=262144); SET another set. LAYERS=N is zspark's `--layers N`, a trial of this script under shared
# locks. EXTRA holds more arguments for both ranks, RANK_ENV more environment (`A=1 B=2`: rank 1 runs over ssh
# and inherits none of this shell's; each rank's log is the proof of what it took). WRAP is a command zspark's
# rank 0 runs under (`nsys profile ... -o FILE`: a trace of that rank, which SIGTERM ends and writes; its
# figures are not a timing's), WRAP1 rank 1's on the other Spark (its file is written there). TF_WRAP=1 is its TensorFold counterpart (spec 091): rank 0's container starts
# under `/nsys/bin/nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none
# --force-overwrite true -o /trace/rank0 python -m tensorfold serve ...`, the host's Nsight Systems
# (NSYS_HOME, 2025.3.2) mounted read-only at /nsys and TF_TRACE_DIR (default ~/.cache/zspark/scratch-091)
# writable at /trace, so the report is /trace/rank0.nsys-rep on the host; nsys writes it when `docker stop`
# ends the rank. Rank 1 is wrapped by WRAP1 alone. AFTER is a command run after the passes with the server's
# URL and NAME as its arguments, its output in OUT_DIR/after.txt. KILL=1 ends zspark's rank 0 with SIGKILL and
# records how long rank 1 took to end (`peer_end_s`); otherwise SIGTERM, TensorFold's containers `docker stop`.
#
# Locks: this machine's `machine.lock` then `gpu.lock`, then the other's over ssh, held for the load; both
# machines alone unless LAYERS. A GPU another process holds stops the script before anything is started. Rank
# 0 listens on 127.0.0.1:18888 (HOST, PORT; never the LAN's endpoint). The ranks meet on the link: zspark's
# rank 1 listens on PEER:29551, TensorFold's TCPStore is rank 0's link address, port 29561. NCCL's settings
# are `tools/two_ranks.sh`'s (`link_env`). TensorFold is TENSORFOLD (0.6.6, read only) in IMAGE. PEER_REPO is this repository on the other Spark; both binaries are one commit's.
set -uo pipefail
if [ $# -lt 3 ]; then sed -n 2,6p "$0" | sed 's/^# \{0,1\}//'; exit 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
. "$HERE/../link_lib.sh"
CACHE=$HOME/.cache/zspark
# shellcheck source=tools/reference/serve_lib.sh
. "$HERE/serve_lib.sh"
# shellcheck disable=SC1091
. "$CACHE/env.sh"
SERVER=$1; OUT=$(realpath -m "$2"); SEED=$3; shift 3
: "${PEER:?Set PEER in .env.local}"
PEER_REPO=${PEER_REPO:-$REPO}
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-18888}
serve_refuse_lan "$HOST" "$PORT"
CONTEXT=${CONTEXT:-2051}
CKPT=${CKPT:-$ZSPARK_GLM_FP4_LOCAL}
NAME=glm-5.3-flash
TENSORFOLD=${TENSORFOLD:-$CACHE/tensorfold-cb2ebf0}
IMAGE=${IMAGE:-tensorfold-glm53:v0.6.0}
ONLY=${ONLY-warm,decode,prompt-50-cold,prompt-50-second,prompt-50-edited,prompt-500-cold,prompt-500-second,prompt-500-edited,turns-500-first,turns-500-second,turns-500-third,turns-500-third-edited,turns-500-second-edited}
[ $# -gt 0 ] || set -- "run=$NAME"
MACHINE=-x; [ -n "${LAYERS:-}" ] && MACHINE=-s
LIMIT=${LIMIT:-7200}
SSH=(ssh -o BatchMode=yes "$PEER")
mine=$(link_env "$PEER" 2 4) || exit 1
ME=$(ip -o route get "$PEER" | sed -n 's/.* src \([^ ]*\).*/\1/p')
theirs=$("${SSH[@]}" bash -s <<<"$(declare -f link_env); link_env $ME 2 4") || exit 1
mkdir -p "$OUT"

if [ "$SERVER" = zspark ]; then
  [ -z "${LAYERS:-}" ] || EXTRA="--layers $LAYERS ${EXTRA:-}"
  here=$("$REPO/zig-out/bin/zspark" version) || exit 1
  there=$("${SSH[@]}" "$PEER_REPO/zig-out/bin/zspark version") || exit 2
  [ "$here" = "$there" ] || { echo "'$here' here, '$there' on $PEER: build the same commit on both" >&2; exit 2; }
  flags="--host $HOST --port $PORT --name $NAME --context $CONTEXT"
  flags="$flags ${EXTRA:-}"
  R0="env $mine ${RANK_ENV:-} ${WRAP:-} timeout $LIMIT $REPO/zig-out/bin/zspark serve $CKPT $flags ${DRAFTER:+--drafter $ZSPARK_GLM_DRAFTER} rank=0 at=$PEER:29551"
  R1="env $theirs ${RANK_ENV:-} ${WRAP1:-} timeout $((LIMIT + 60)) $PEER_REPO/zig-out/bin/zspark serve ${CKPT1:-$CKPT} $flags ${DRAFTER:+--drafter ${DRAFTER1:-$ZSPARK_GLM_DRAFTER}} rank=1 at=$PEER:29551"
  VERSION=$here
elif [ "$SERVER" = tensorfold ]; then
  [ -z "${LAYERS:-}" ] || { echo "TensorFold's server has no --layers: its load is the whole model" >&2; exit 2; }
  hf() { echo "/hf/${1#"$HOME"/.cache/huggingface/}"; }
  rank_cmd() {  # $1 the rank, $2 its NCCL settings
    local e env= entry=python nsys= mounts=
    for e in $2; do env="$env -e $e"; done
    if [ "$1" = 0 ] && [ -n "${TF_WRAP:-}" ]; then
      # nsys inside the container (spec 091): the host's Nsight Systems read-only at /nsys, the report's
      # directory writable at /trace; nsys writes the report when `docker stop` ends the rank.
      entry=/nsys/bin/nsys
      nsys="profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite true -o /trace/rank0 python"
      mounts="-v ${NSYS_HOME:-/opt/nvidia/nsight-systems/2025.3.2}:/nsys:ro -v ${TF_TRACE_DIR:-$CACHE/scratch-091}:/trace"
    fi
    echo "docker run --rm --name zspark-glm-rank$1 --gpus all --ipc=host --network host --shm-size 16g" \
      "--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864" \
      "--user \$(id -u):\$(id -g) -e HOME=/cache -e PYTHONPATH=/tf/src -e PYTHONUNBUFFERED=1" \
      "-e TENSORFOLD_NO_UPDATE_CHECK=1 $env -v $TENSORFOLD:/tf:ro -v \$HOME/.cache/huggingface:/hf:ro $mounts" \
      "-v \$HOME/.cache/zspark/tensorfold-kernels:/cache --entrypoint $entry $IMAGE $nsys -m tensorfold serve" \
      "$(hf "$CKPT") --backend cuda --tp 2 --rank $1 --master $ME --master-port 29561" \
      "--host $HOST --port $PORT --name $NAME --snapshot-dir none --no-update-check" \
      "--drafter $([ -n "${DRAFTER:-}" ] && hf "$ZSPARK_GLM_DRAFTER" || echo none)" \
      "${TF_CONTEXT:+--context $TF_CONTEXT} ${EXTRA:-}"
  }
  R0="timeout $LIMIT $(rank_cmd 0 "$mine")"; R1="timeout $((LIMIT + 60)) $(rank_cmd 1 "$theirs")"
  VERSION="TensorFold $(git -C "$TENSORFOLD" rev-parse --short HEAD 2>/dev/null || basename "$TENSORFOLD")"
else
  echo "zspark or tensorfold, not $SERVER" >&2; exit 2
fi
echo "rank 0 $(hostname) $ME: $mine"; echo "rank 1 $PEER: $theirs"
free_gpu=$(gpu_guard glm5_client_bench.sh)
export PASSES="$*"
export SERVER OUT SEED PEER HOST PORT CONTEXT CKPT NAME ONLY HERE R0 R1 VERSION MACHINE free_gpu
export SET=${SET:-} AFTER=${AFTER:-} KILL=${KILL:-} DRAFTER=${DRAFTER:-} LAYERS=${LAYERS:-} EXTRA=${EXTRA:-}

# shellcheck disable=SC2016,SC2086
flock $MACHINE "$CACHE/machine.lock" flock "$CACHE/gpu.lock" bash -c "$free_gpu"'
  . "$HERE/serve_lib.sh"
  peer() { ssh -o BatchMode=yes "$PEER" "$@"; }
  both() { echo "$(hostname): $(uptime)"; echo "$PEER: $(peer uptime)"; }                   # uptime of both machines
  device() { echo "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk "{s+=\$1} END {print s+0}")" \
    "$(peer nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk "{s+=\$1} END {print s+0}")"; }
  LOG=$OUT/server.log
  { date -Is; both; free -m; } > "$OUT/before.txt"
  START=$(date +%s.%N)
  peer "flock $MACHINE ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock bash -c '\''$free_gpu
    echo glm5_client_bench.sh: locked; exec $R1'\''" > "$OUT/rank1.log" 2>&1 &
  remote=$!
  until grep -q "^glm5_client_bench.sh: locked" "$OUT/rank1.log"; do kill -0 $remote 2> /dev/null || { cat "$OUT/rank1.log"; exit 1; }; sleep 1; done
  LOCKED=$(date +%s.%N)
  ( eval "exec $R0" ) > "$LOG" 2>&1 &
  SERVE_PID=$!
  stop() {  # rank 0 first; rank 1 ends with it, and how long that took is the answer to a killed rank
    local t0; t0=$(date +%s.%N)
    if [ "$SERVER" = zspark ]; then
      if [ -n "$KILL" ]; then pkill -KILL -P "$SERVE_PID"; else kill -TERM "$SERVE_PID" 2> /dev/null; fi
      for _ in $(seq 1 300); do kill -0 $remote 2> /dev/null || break; sleep 0.2; done
      PEER_END=$(echo "$(date +%s.%N) - $t0" | bc)
      kill -0 $remote 2> /dev/null && peer "pkill -TERM -f \"[z]spark serve .* rank=1 at=\""
    else
      docker stop -t 60 zspark-glm-rank0 > /dev/null 2>&1; peer docker stop -t 60 zspark-glm-rank1 > /dev/null 2>&1
      PEER_END=0
    fi
    wait $remote 2> /dev/null; wait "$SERVE_PID" 2> /dev/null
  }
  trap stop EXIT
  ANSWERED=0
  serve_wait "http://$HOST:$PORT/v1/models" 6000 2 || ANSWERED=$?
  if [ "$ANSWERED" != 0 ]; then echo "the server did not answer ($ANSWERED): $LOG" >&2; tail -5 "$LOG" "$OUT/rank1.log" >&2; exit 1; fi
  READY=$(date +%s.%N)
  read -r DEV0 DEV1 <<< "$(device)"
  STATUS=0
  for pass in $PASSES; do
    label=${pass%%=*} model=${pass#*=}
    mkdir -p "$OUT/$label"
    { date -Is; both; } > "$OUT/$label/uptime.txt"
    LINES=$(wc -l < "$LOG")
    python3 "$HERE/client_bench.py" run --url "http://$HOST:$PORT" --model "$model" --seed "$SEED" \
      ${ONLY:+--only $ONLY} ${SET:+--set $SET} --out "$OUT/$label/harness.json" > "$OUT/$label/harness.txt" 2>&1 || STATUS=$?
    tail -n +"$((LINES + 1))" "$LOG" > "$OUT/$label/harness.log"
    { date -Is; both; } >> "$OUT/$label/uptime.txt"
    echo "$label ($model): harness status $STATUS"
  done
  read -r DEV2 DEV3 <<< "$(device)"
  if [ -n "$AFTER" ]; then $AFTER "http://$HOST:$PORT" "$NAME" > "$OUT/after.txt" 2>&1 || STATUS=$?; fi
  { date -Is; both; free -m; } > "$OUT/after-uptime.txt"
  trap - EXIT
  stop
  # zspark: "its share 82.33 GiB in 36.1 s"; TensorFold: "loaded in 25.6s)"
  LOAD=$(grep -o -E "(GiB in|loaded in) [0-9.]+ ?s" "$LOG" | head -1 | grep -o -E "[0-9.]+" | tail -1)
  python3 - "$OUT/meta.json" <<PY
import json, os, sys
e = os.environ
json.dump({"server": e["SERVER"], "version": e["VERSION"], "checkpoint": e["CKPT"], "seed": int(e["SEED"]), "context": int(e["CONTEXT"]),
           "drafter": bool(e["DRAFTER"]), "layers": e["LAYERS"], "extra": e["EXTRA"], "passes": "$PASSES".split(),
           "locked_s": round($LOCKED - $START, 2), "ready_s": round($READY - $LOCKED, 2), "load_s": float("${LOAD:-0}" or 0),
           "device_mib_loaded": [$DEV0, $DEV1], "device_mib_end": [$DEV2, $DEV3], "peer_end_s": float("$PEER_END"),
           "killed": bool(e["KILL"]), "rank0": e["R0"], "rank1": e["R1"],
           "before": open(e["OUT"] + "/before.txt").read().splitlines()[1:3]}, open(sys.argv[1], "w"), indent=1)
PY
  echo "$SERVER seed $SEED: ready $(echo "$READY - $LOCKED" | bc) s after both machines were held, rank 1 ended $PEER_END s after rank 0, status $STATUS, $OUT"
  exit $STATUS' bench
