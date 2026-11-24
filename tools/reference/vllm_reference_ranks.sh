#!/bin/bash
# vllm_reference.sh's two-node form (spec 107 WS-02): the card's vLLM container at --tensor-parallel-size 2, a
# node a Spark over the ConnectX link, as the card of a checkpoint that runs on two Sparks only launches it
# (GLM-5.3-Flash FP4-Local). Node 1 (headless) is started on PEER first, node 0 here serves 127.0.0.1:18888;
# vllm_reference.py fetches the cases' greedy replies and the score text's prompt logprobs; both are stopped.
#
#   tools/reference/vllm_reference_ranks.sh MODEL_DIR CASES SCORE_IDS OUT [IMAGE]
#   tools/reference/vllm_reference_ranks.sh bench MODEL_DIR OUT SEED [IMAGE]
#
# `bench` (spec 107 WS-03) serves it as the card does, with its MTP of 5 (`humming` experts, B12X attention) and
# `--max-num-seqs 4`, and runs client_bench.py against node 0 (SEED, ONLY, SET as glm5_client_bench.sh's): OUT
# gets harness.json, harness.txt, meta.json (ready_s from the first container's start, both nodes' device MiB,
# both machines' uptime) and the nodes' logs. VLLM=card keeps the card's B12X_W4A16_A4_PREFILL=1 (a prompt's
# expert activations FP4); VLLM=a16 sets it to 0, zspark's precision. The served name is glm-5.3-flash.
#
# MODEL_DIR is a path both Sparks hold (on each a Hugging Face snapshot whose files link into its blobs, its hub
# directory mounted, or a plain directory): one container path on both; IMAGE as vllm_reference.sh's, on both.
# Run it on spark-a with no model on either GPU: it
# takes this machine's `machine.lock` whole then `gpu.lock`, then the peer's the same way over ssh, and holds
# them to the end. The command is the card's two-node one but for vllm_reference.sh's departures: --host
# 127.0.0.1 --port 18888 on node 0, --max-num-seqs 1, no --speculative-config, B12X_W4A16_A4_PREFILL=0 (the card
# quantizes prefill activations to FP4), VLLM_SERVER_DEV_MODE=1, HF_HUB_OFFLINE=1; and NCCL_IB_GID_INDEX, the
# RoCE v2 GID of the link's IPv4 address, which two_ranks.sh gives zspark's ranks too. The link's netdev and its
# RoCE devices (both rails) are read from `ip` and sysfs on each machine, as two_ranks.sh reads them.
# PRECISION=image (spec 114 WS-06) leaves out VLLM_B12X_MOE_FP4_FORCE_A16 and B12X_W4A16_A4_PREFILL: the image's
# defaults, the routed experts W4A4 (each expert's input_scale) in both phases, zspark's precision since spec 114.
# PRECISION=card (spec 115) keeps the card's launch environment whole: VLLM_B12X_MOE_FP4_FORCE_A16=1 and
# B12X_W4A16_A4_PREFILL=1, a prompt's expert activations FP4 and decoding's bf16, zspark's bar since spec 115.
set -euo pipefail
mode=reference
if [ "${1:-}" = bench ]; then
  mode=bench; shift
  given=$(realpath -s "$1"); model=$(realpath "$1"); out=$(realpath -m "$2"); seed=$3; image=${4:-eugr/spark-vllm-b12x:latest}
  case "${VLLM:-card}" in card) prefill4=1 ;; a16) prefill4=0 ;; *) echo "VLLM is card or a16, not $VLLM" >&2; exit 2 ;; esac
else
  given=$(realpath -s "$1"); model=$(realpath "$1"); cases=$(realpath "$2"); score=$(realpath "$3"); out=$(realpath -m "$4"); image=${5:-eugr/spark-vllm-b12x:latest}
fi
here=$(cd "$(dirname "$0")" && pwd)
. "$here/../local_env.sh"
load_local_env "$here/../.." || exit $?
: "${PEER:?Set PEER in .env.local}"
CACHE=$HOME/.cache/zspark
name=zspark-vllm-reference
port=18888
master_port=29501
SSH=(ssh -o BatchMode=yes "$PEER")
mkdir -p "$out"

# The netdev, its RoCE devices and their GID index for the route to $1, on the machine this runs on.
link() {
  local route dev addr hca gid= hex n hcas
  route=$(ip -o route get "$1"); dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  addr=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  hca=$(ls "/sys/class/net/$dev/device/infiniband" | head -1)
  hex=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' ${addr//./ })
  for n in "/sys/class/infiniband/$hca/ports/1/gid_attrs/types/"*; do
    [ "$(cat "$n" 2>/dev/null)" = "RoCE v2" ] && [ "$(cat "/sys/class/infiniband/$hca/ports/1/gids/${n##*/}" 2>/dev/null)" = "$hex" ] && { gid=${n##*/}; break; }
  done
  hcas=$hca
  for n in /sys/class/infiniband/*; do
    [ "${n##*/}" != "$hca" ] && [ "$(cat "$n/ports/1/gid_attrs/types/$gid" 2>/dev/null)" = "RoCE v2" ] || continue
    case $(cat "$n/ports/1/gids/$gid" 2>/dev/null) in 0000:0000:0000:0000:0000:ffff:*) hcas=$hcas,${n##*/} ;; esac
  done
  echo "$dev $hcas $gid $addr"
}

# The `docker run` of node $1 (0 or 1), its link "$2" and the head's address $3, as a command line on stdout.
# The snapshot is mounted at one path on both machines: where its files link into blobs, the hub directory
# its repository lies in (a blob may link on into the hub's shared `blobs/`).
node() {
  local rank=$1 dev hcas gid addr head=$3 mount target serve
  read -r dev hcas gid addr <<<"$2"
  target=/hf/$(basename "$(dirname "$(dirname "$model")")")/snapshots/$(basename "$model")
  # decided on the machine that runs it: spark-a's snapshot links into the Hugging Face cache, spark-b's is plain
  if [ "$rank" = 0 ]; then
    mount="$model:$target:ro"
    [ -L "$(ls -d "$model"/* | head -1)" ] && mount="$(cd "$model/../../.." && pwd):/hf:ro"
  else
    mount="$PEER_MODEL:$target:ro"
    [ -n "$PEER_LINKED" ] && mount="$PEER_LINKED:/hf:ro"
  fi
  serve=(--host 0.0.0.0 --port 8000 --headless)
  [ "$rank" = 0 ] && serve=(--host 127.0.0.1 --port $port)
  # the reference's departures from the card, or the card's own serve (`bench`)
  local served seqs=1 a4=0 devmode=(-e VLLM_SERVER_DEV_MODE=1) spec=() moe=()
  served=$(basename "$given")
  if [ "$mode" = bench ]; then
    served=glm-5.3-flash seqs=4 a4=$prefill4 devmode=()
    spec=(--speculative-config '{"method":"mtp","num_speculative_tokens":5,"moe_backend":"humming","attention_backend":"B12X"}')
  fi
  [ "${PRECISION:-a16}" = card ] && a4=1
  [ "${PRECISION:-a16}" = image ] || moe=(-e VLLM_B12X_MOE_FP4_FORCE_A16=1 -e B12X_W4A16_A4_PREFILL=$a4)
  printf '%q ' docker run -d --name "$name" --privileged --gpus all --ipc host --network host --shm-size=16g \
    --ulimit nofile=1048576:1048576 --ulimit memlock=-1 --ulimit stack=67108864 \
    -e MN_IF_NAME="$dev" -e UCX_NET_DEVICES="$dev" -e NCCL_SOCKET_IFNAME="$dev" -e NCCL_IB_HCA="$hcas" \
    -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX="$gid" -e OMPI_MCA_btl_tcp_if_include="$dev" -e GLOO_SOCKET_IFNAME="$dev" \
    -e TP_SOCKET_IFNAME="$dev" -e NCCL_IGNORE_CPU_AFFINITY=1 -e HEAD_IP="$head" -e MASTER_PORT=$master_port \
    -e CUTE_DSL_ARCH=sm_121a -e SAFETENSORS_FAST_GPU=1 -e VLLM_ENABLE_ROCE_ALLREDUCE=1 \
    -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e NCCL_BUFFSIZE=1048576 -e NCCL_MAX_CTAS=8 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e B12X_POLICY_MODE=auto "${moe[@]}" \
    "${devmode[@]}" -e HF_HUB_OFFLINE=1 \
    -v "$mount" -v "$HOME/.cache/vllm:/root/.cache/vllm" -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "$HOME/.triton:/root/.triton" -v "$HOME/.cache/b12x:/root/.cache/b12x" \
    --entrypoint /bin/bash "$image" -c 'exec vllm serve "$@"' vllm "$target" --served-model-name "$served" \
    --tensor-parallel-size 2 --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16 \
    --kv-cache-dtype fp8 --quantization modelopt_mixed --attention-backend B12X --block-size 256 --moe-backend b12x \
    --linear-backend b12x --no-enable-flashinfer-autotune --load-format b12x --max-model-len 500000 --max-num-seqs $seqs \
    --max-num-batched-tokens 4096 "${spec[@]}" --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --gpu-memory-utilization 0.87 --nnodes 2 --node-rank "$rank" --master-addr "$head" --master-port $master_port "${serve[@]}"
}

mine=$(link "$PEER")
me=$(awk '{ print $4 }' <<<"$mine")
theirs=$("${SSH[@]}" bash -s <<<"$(declare -f link); link $me")
# the peer's snapshot: where its files link into blobs, the hub directory to mount
PEER_MODEL=$("${SSH[@]}" "realpath '$given'")
PEER_LINKED=$("${SSH[@]}" "[ -L \"\$(ls -d '$PEER_MODEL'/* | head -1)\" ] && cd '$PEER_MODEL/../../..' && pwd" || true)
free='[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "the GPU of $(hostname) is in use" >&2; exit 75; }'
digest=${DIGEST:-$(docker image inspect "$image" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo unknown)}
id=$(docker image inspect "$image" --format '{{.Id}}')
[ "$id" = "$("${SSH[@]}" docker image inspect "$image" --format '{{.Id}}')" ] || { echo "the image differs on $PEER" >&2; exit 2; }
echo "node 0 $(hostname): $mine"; echo "node 1 $PEER: $theirs"

exec 8>"$CACHE/machine.lock"; flock -x 8; exec 9>"$CACHE/gpu.lock"; flock 9
eval "$free"
# the peer's locks, held by a process this script ends: it prints once it holds them
"${SSH[@]}" "exec flock -x ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock bash -c '$free; echo held \$\$; exec sleep 86400'" > "$out/peer-locks.log" 2>&1 &
holder=$!
cleanup() {
  docker stop -t 30 "$name" >/dev/null 2>&1 || true; docker rm -f "$name" >/dev/null 2>&1 || true
  "${SSH[@]}" "docker stop -t 30 $name >/dev/null 2>&1; docker rm -f $name >/dev/null 2>&1; kill $(awk '/^held/ { print $2 }' "$out/peer-locks.log") 2>/dev/null; true" || true
  kill "$holder" 2>/dev/null || true
}
trap cleanup EXIT
until grep -q '^held ' "$out/peer-locks.log"; do kill -0 "$holder" 2>/dev/null || { cat "$out/peer-locks.log" >&2; exit 1; }; sleep 2; done
mkdir -p "$HOME/.cache/vllm" "$HOME/.cache/flashinfer" "$HOME/.triton" "$HOME/.cache/b12x"
"${SSH[@]}" 'mkdir -p ~/.cache/vllm ~/.cache/flashinfer ~/.triton ~/.cache/b12x'
docker rm -f "$name" >/dev/null 2>&1 || true; "${SSH[@]}" "docker rm -f $name >/dev/null 2>&1 || true"
started=$(date +%s)
# the worker first, as the card says
"${SSH[@]}" "$(node 1 "$theirs" "$me")" > /dev/null
"${SSH[@]}" "docker logs -f $name" > "$out/server.rank1.log" 2>&1 &
eval "$(node 0 "$mine" "$me")" > /dev/null
docker logs -f "$name" > "$out/server.log" 2>&1 &
until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; do
  docker ps --format '{{.Names}}' | grep -qx "$name" || { echo "node 0's container stopped while loading" >&2; exit 1; }
  "${SSH[@]}" "docker ps --format '{{.Names}}'" | grep -qx "$name" || { echo "node 1's container stopped while loading" >&2; exit 1; }
  sleep 5
done
ready=$(date +%s)
if [ "$mode" = bench ]; then
  both() { echo "$(hostname): $(uptime)"; echo "$PEER: $("${SSH[@]}" uptime)"; }
  mib='nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk "{s+=\$1} END {print s+0}"'
  loaded="$(eval "$mib"), $("${SSH[@]}" "$mib")"
  { date -Is; both; } > "$out/uptime.txt"
  lines=$(wc -l < "$out/server.log")
  status=0
  # shellcheck disable=SC2086
  python3 -I "$here/client_bench.py" run --url "http://127.0.0.1:$port" --model glm-5.3-flash --seed "$seed" \
    ${ONLY:+--only $ONLY} ${SET:+--set $SET} --out "$out/harness.json" > "$out/harness.txt" 2>&1 || status=$?
  tail -n +"$((lines + 1))" "$out/server.log" > "$out/harness.log"
  { date -Is; both; } >> "$out/uptime.txt"
  printf '{"server": "vllm", "vllm": "%s", "image": "%s", "digest": "%s", "checkpoint": "%s", "seed": %s, "ready_s": %d, "device_mib_loaded": [%s], "device_mib_end": [%s], "tensor_parallel": 2}\n' \
    "${VLLM:-card}" "$image" "$digest" "$model" "$seed" $((ready - started)) "$loaded" "$(eval "$mib"), $("${SSH[@]}" "$mib")" > "$out/meta.json"
  echo "vllm ${VLLM:-card} seed $seed: ready $((ready - started)) s, harness status $status, $out"
  exit $status
fi
python3 -I "$here/vllm_reference.py" fetch "http://127.0.0.1:$port" "$cases" "$score" "$out"
printf '{"image": "%s", "digest": "%s", "id": "%s", "model": "%s", "load_seconds": %d, "hosts": ["%s", "%s"], "tensor_parallel": 2}\n' \
  "$image" "$digest" "$id" "$model" $((ready - started)) "$(hostname)" "$("${SSH[@]}" hostname)" > "$out/provenance.json"
