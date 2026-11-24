#!/bin/bash
# One load of one server, measured as a client sees it (spec 024): starts `zspark serve` or TensorFold's server on
# a checkpoint, runs tools/reference/client_bench.py against it, optionally sparkDash's two benchmarks, records
# load time and memory, and stops the server. One load serves every request. The machine is held alone for all
# of it (`machine.lock` whole, `gpu.lock` inside), so run nothing else under those locks around it.
#
#   tools/reference/client_bench.sh zspark|tensorfold|vllm CHECKPOINT_DIR OUT_DIR SEED [sparkdash]
#
# OUT_DIR gets server.log, harness.log (the server's lines while the harness ran), harness.json, meta.json and,
# with `sparkdash`, sparkdash.json. SEED is the load's number: the same seed gives both servers the same bytes.
#
# Without `sparkdash` the server listens on 127.0.0.1:18888. With it, on the machine's LAN address (HOST, PORT:
# 192.0.2.227:8888 on spark-b, the address and port sparkDash knows) without a key, for the run only: on
# spark-a that address and port are the LAN's model endpoint, and other clients' requests then reach the bench
# (one did, spec 032), so sparkDash's benchmarks are taken on spark-b, and the script starts nothing on
# 192.0.2.114:8888, whatever HOST and PORT say. Both servers get the same context (CONTEXT, 81920 unless
# set: the context TensorFold's recorded loads and every table of spec 024 were taken at, so a load with the
# default stands beside them) and the name `qwen3.8-flash-next`. TensorFold runs from TENSORFOLD (read only)
# in its container IMAGE, one rank, with `--snapshot-dir none --no-update-check` and otherwise its defaults;
# zspark is ZSPARK (a `-Doptimize=fast` build) with its defaults. vLLM is the container the card of
# nvidia/Qwen3.8-Flash-Next-FP4-Local names (VLLM_IMAGE), with its flags (VLLM=card) or its flags at zspark's
# precision (VLLM=a16), both drafting with MTP as the card does (spec 105 WS-03). EXTRA holds more arguments for the server;
# SET names another set of requests than client_bench.json (required_call_bench.json: a call free and
# required, with thinking and without, spec 034). STREAMS, when set (1,2,4), runs that many clients at once
# for each count on this one load and skips the spec-024 set; MAX_TOKENS is each reply's length (128).
# STREAM_PROMPTS names their prompts: `prose` (unset: a paragraph each), `chains` (four lists the model knows
# by heart, whose draft chains run long: spec 077) or `code` (four short functions), or a JSON file of prompts;
# several, comma separated, are run one after another on the load, a `streams-SET.json` each. STREAM_ALL=1
# runs a count over every group of that many prompts of the set, not its first ones alone.
# RANKS=2 (zspark only, spec 126) serves the checkpoint on two ranks, rank 0 here and rank 1 on the other Spark,
# through `MACHINE=-x tools/two_ranks.sh serve` (its own session): that script holds both machines alone for the
# serve's life and runs each repository's `zig-out/bin/zspark` (built with -Dnccl=/usr, one commit; ZSPARK is not
# read), so this script takes no lock itself; CHECKPOINT_DIR is a path both Sparks hold, its links unresolved.
# The peer's device memory and uptime are recorded beside this machine's; the stop is SIGTERM to rank 0's
# zspark, which ends rank 1 (the peer's is ended after 120 s).
set -euo pipefail
if [ $# -lt 4 ]; then sed -n 2,35p "$0"; exit 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=tools/reference/serve_lib.sh
. "$HERE/serve_lib.sh"
RANKS=${RANKS:-1}
case $RANKS in 1) serve_locked CLIENT_BENCH_LOCKED "$@" ;; 2) [ "$1" = zspark ] || { echo "RANKS=2 is zspark's" >&2; exit 2; } ;;
  *) echo "RANKS is 1 or 2, not $RANKS" >&2; exit 2 ;; esac
if [ "$RANKS" = 2 ]; then : "${PEER:?Set PEER in .env.local}"; fi
peer_device() {  # the other Spark's device MiB on two ranks, 0 on one
  if [ "$RANKS" = 2 ]; then ssh -o BatchMode=yes "$PEER" nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}'; else echo 0; fi
}
peer_uptime() { [ "$RANKS" = 2 ] && echo "$PEER: $(ssh -o BatchMode=yes "$PEER" uptime)" || true; }
SERVER=$1; CKPT=$(realpath "$2"); OUT=$3; SEED=$4; DASH=${5:-}
# rank 1 opens the path as given: a link here may resolve to what the other Spark lacks (spark-a's HF snapshot)
[ "$RANKS" = 1 ] || CKPT=$(realpath -s "$2")
REPO=$(cd "$HERE/../.." && pwd)
if [ "$DASH" = sparkdash ]; then
  HOST=${HOST:-${SPARKDASH_HOST:?Set SPARKDASH_HOST in .env.local or pass HOST}}
  PORT=${PORT:-8888}
else HOST=${HOST:-127.0.0.1}; PORT=${PORT:-18888}; fi
serve_refuse_lan "$HOST" "$PORT"
CONTEXT=${CONTEXT:-81920}
ZSPARK=${ZSPARK:-$REPO/zig-out/bin/zspark}
TENSORFOLD=${TENSORFOLD:-$HOME/repo/TensorFold}
IMAGE=${IMAGE:-tensorfold-glm53:v0.6.0}
SPARK=${SPARK:-${SPARKDASH_SPARK:-spark-b}}
NAME=qwen3.8-flash-next
CONTAINER=zspark-024-tensorfold
mkdir -p "$OUT"
LOG=$OUT/server.log
{ date -Is; uptime; free -m; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; peer_uptime; } > "$OUT/before.txt"
serve_gpu_free

START=$(date +%s.%N)
if [ "$SERVER" = zspark ] && [ "$RANKS" = 2 ]; then
  serve_refuse_lan "$HOST" "$PORT"
  # shellcheck disable=SC2086
  MACHINE=-x setsid "$REPO/tools/two_ranks.sh" serve "$CKPT" --host "$HOST" --port "$PORT" --name "$NAME" \
    --context "$CONTEXT" ${EXTRA:-} > "$LOG" 2>&1 &
  PID=$!
  SERVE_PID=$PID
  stop() {  # rank 0's zspark in the script's session; rank 1 ends with it
    pkill -TERM -x zspark -s "$PID" 2>/dev/null || true
    for _ in $(seq 1 600); do kill -0 "$PID" 2>/dev/null || break; sleep 0.2; done
    if kill -0 "$PID" 2>/dev/null; then
      ssh -o BatchMode=yes "$PEER" "pkill -TERM -f \"[z]spark serve .* rank=1 at=\"" || true
      kill -TERM -- "-$PID" 2>/dev/null || true
    fi
    wait "$PID" 2>/dev/null || true
  }
  rss_pid() { pgrep -x zspark -s "$PID" | head -1; }
elif [ "$SERVER" = zspark ]; then
  # shellcheck disable=SC2086
  serve_start "$HOST" "$PORT" "$LOG" "$ZSPARK" "$CKPT" --name "$NAME" --context "$CONTEXT" ${EXTRA:-}
  PID=$SERVE_PID
  stop() { serve_stop TERM; }
  rss_pid() { echo "$PID"; }
elif [ "$SERVER" = vllm ]; then
  # The card's command for the DGX Spark (vllm_reference.sh has it with the reference's changes), served as the
  # bench names it on HOST:PORT. VLLM=card is the card's flags as written: MTP of 4, the head requantized to
  # MXFP8, prefill activations to FP4. VLLM=a16 keeps the MTP and zspark's precision: the head bf16
  # (VLLM_MXFP8_LM_HEAD=0) and prefill bf16 (B12X_W4A16_A4_PREFILL=0), as the reference runs.
  CONTAINER=zspark-bench-vllm
  case "${VLLM:-card}" in
    card) HEAD8=1; PREFILL4=1 ;;
    a16) HEAD8=0; PREFILL4=0 ;;
    *) echo "VLLM is card or a16, not ${VLLM}" >&2; exit 2 ;;
  esac
  mkdir -p "$HOME/.cache/vllm" "$HOME/.cache/flashinfer" "$HOME/.triton" "$HOME/.cache/b12x"
  docker rm -f "$CONTAINER" > /dev/null 2>&1 || true
  # shellcheck disable=SC2086
  docker run --rm --name "$CONTAINER" --gpus all --ipc host --network host --shm-size=16g --ulimit memlock=-1 \
    --ulimit stack=67108864 --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$CKPT:/model:ro" -v "$HOME/.cache/vllm:/root/.cache/vllm" -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "$HOME/.triton:/root/.triton" -v "$HOME/.cache/b12x:/root/.cache/b12x" \
    -e VLLM_PLE_TABLE_MEMORY=disk -e B12X_DISK_BACKEND=io_uring -e CUTE_DSL_ARCH=sm_121a -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e VLLM_USE_AOT_COMPILE=1 \
    -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e B12X_POLICY_MODE=auto \
    -e VLLM_MXFP8_LM_HEAD=$HEAD8 -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
    -e B12X_W4A16_A4_PREFILL=$PREFILL4 -e HF_HUB_OFFLINE=1 \
    --entrypoint /bin/bash "${VLLM_IMAGE:-eugr/spark-vllm-b12x:latest}" -c 'exec vllm serve /model "$@"' vllm \
    --served-model-name "$NAME" --host "$HOST" --port "$PORT" --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16 --kv-cache-dtype fp8 \
    --quantization modelopt_mixed --block-size 16 --load-format b12x --max-model-len "$CONTEXT" --max-num-seqs 8 \
    --max-num-batched-tokens 4096 --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
    --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x --no-enable-flashinfer-autotune \
    --mm-encoder-tp-mode data --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --enable-prompt-tokens-details --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    --gpu-memory-utilization 0.8 ${EXTRA:-} > "$LOG" 2>&1 &
  PID=$!
  SERVE_PID=$PID
  stop() { docker stop -t 60 "$CONTAINER" > /dev/null 2>&1 || true; wait "$PID" 2>/dev/null || true; }
  rss_pid() { docker inspect -f '{{.State.Pid}}' "$CONTAINER"; }
else
  # shellcheck disable=SC2086
  docker run --rm --name "$CONTAINER" --gpus all --network host --user "$(id -u):$(id -g)" -e HOME=/cache \
    -e PYTHONPATH=/tf/src -e PYTHONUNBUFFERED=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 -v "$TENSORFOLD":/tf:ro \
    -v "$HOME/.cache/huggingface":"$HOME/.cache/huggingface":ro -v "$HOME/models":"$HOME/models":ro \
    -v "$HOME/.cache/zspark/tensorfold-kernels":/cache --entrypoint python "$IMAGE" -m tensorfold serve "$CKPT" \
    --backend cuda --host "$HOST" --port "$PORT" --name "$NAME" --context "$CONTEXT" --snapshot-dir none \
    --no-update-check ${EXTRA:-} > "$LOG" 2>&1 &
  PID=$!
  SERVE_PID=$PID
  stop() { docker stop -t 60 "$CONTAINER" > /dev/null 2>&1 || true; wait "$PID" 2>/dev/null || true; }
  rss_pid() { docker inspect -f '{{.State.Pid}}' "$CONTAINER"; }
fi
trap stop EXIT

READY=
ANSWERED=0
# vLLM compiles for some minutes before it answers (854 s on its first load, spec 105)
serve_wait "http://$HOST:$PORT/v1/models" "$([ "$SERVER" = vllm ] && echo 9000 || echo 1800)" 2 || ANSWERED=$?
if [ "$ANSWERED" = 1 ]; then echo "the server ended while loading: $LOG" >&2; tail -5 "$LOG" >&2; exit 1; fi
if [ "$ANSWERED" = 0 ]; then READY=$(date +%s.%N); fi
[ -n "$READY" ] || { echo "the server did not answer in time" >&2; exit 1; }

memory() {  # device MiB of every compute process, the server's resident host MiB, the machine's used MiB
  local device rss used
  device=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  rss=$(awk '/VmRSS/ {print int($2/1024)}' "/proc/$(rss_pid)/status" 2>/dev/null || echo 0)
  used=$(free -m | awk '/^Mem:/ {print $3}')
  echo "$device ${rss:-0} $used"
}
read -r DEV0 RSS0 USED0 <<< "$(memory)"
PDEV0=$(peer_device)
LINES=$(wc -l < "$LOG")
STATUS=0
# shellcheck disable=SC2086
if [ -n "${STREAMS:-}" ]; then
  STREAM_PROMPTS=${STREAM_PROMPTS:-prose}
  # one set writes streams.json; several, comma separated, a file each, named after the set
  : > "$OUT/harness.txt"
  for PROMPTS in ${STREAM_PROMPTS//,/ }; do
    JSON=streams.json
    case "$STREAM_PROMPTS" in *,*) JSON=streams-$(basename "$PROMPTS" .json).json ;; esac
    python3 "$HERE/client_bench.py" streams --url "http://$HOST:$PORT" --model "$NAME" --seed "$SEED" \
      --counts "$STREAMS" --max-tokens "${MAX_TOKENS:-128}" --prompts "$PROMPTS" ${STREAM_ALL:+--all} \
      --out "$OUT/$JSON" >> "$OUT/harness.txt" 2>&1 || STATUS=$?
  done
else
  python3 "$HERE/client_bench.py" run --url "http://$HOST:$PORT" --model "$NAME" --seed "$SEED" ${ONLY:+--only $ONLY} \
    ${SET:+--set $SET} \
    --out "$OUT/harness.json" > "$OUT/harness.txt" 2>&1 || STATUS=$?
fi
tail -n +"$((LINES + 1))" "$LOG" > "$OUT/harness.log"
read -r DEV1 RSS1 USED1 <<< "$(memory)"
PDEV1=$(peer_device)
if [ "$DASH" = sparkdash ]; then
  # shellcheck disable=SC2086
  python3 "$HERE/client_bench.py" sparkdash --spark "$SPARK" --port "$PORT" --model "$NAME" ${DASH_ARGS:-} \
    --out "$OUT/sparkdash.json" > "$OUT/sparkdash.txt" 2>&1 || STATUS=$?
fi
read -r DEV2 RSS2 USED2 <<< "$(memory)"
{ date -Is; uptime; free -m; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv; peer_uptime; } > "$OUT/after.txt"
trap - EXIT
stop
# zspark: "loaded in 10.9 s" (two ranks: "its share 37.36 GiB in 20.1 s"); TensorFold: "loaded in 25.6s)"; vLLM's is its ready_s
LOAD=$(grep -o -E '(loaded|GiB) in [0-9.]+ ?s' "$LOG" | head -1 | grep -o -E '[0-9.]+' | head -1 || true)
python3 - "$OUT/meta.json" <<PY
import json, sys
json.dump({"server": "$SERVER", "checkpoint": "$CKPT", "seed": $SEED, "context": $CONTEXT, "extra": "${EXTRA:-}", "vllm": "${VLLM:-}", "ranks": $RANKS,
           "peer_device_mib_loaded": $PDEV0, "peer_device_mib_end": $PDEV1,
           "ready_s": round($READY - $START, 2), "load_s": float("${LOAD:-0}" or 0),
           "device_mib_loaded": $DEV0, "host_rss_mib_loaded": $RSS0, "host_used_mib_loaded": $USED0,
           "device_mib_end": $DEV1, "host_rss_mib_end": $RSS1, "host_used_mib_end": $USED1,
           "device_mib_dash": $DEV2, "host_rss_mib_dash": $RSS2, "host_used_mib_dash": $USED2,
           "uptime_before": open("$OUT/before.txt").read().splitlines()[1].strip(),
           "uptime_after": open("$OUT/after.txt").read().splitlines()[1].strip()}, open(sys.argv[1], "w"), indent=1)
PY
echo "$SERVER seed $SEED: ready in $(echo "$READY - $START" | bc) s, harness status $STATUS, $OUT"
exit $STATUS
