#!/bin/bash
# The vLLM container a checkpoint's card names, as a reference (spec 105 WS-02): started on 127.0.0.1:18888 with
# one sequence, no MTP and every activation the card quantizes kept bf16, asked by vllm_reference.py for the
# cases' greedy replies and the score text's prompt logprobs, then stopped.
#
#   tools/reference/vllm_reference.sh MODEL_DIR CASES SCORE_IDS OUT [IMAGE]
#
# IMAGE is eugr/spark-vllm-b12x:latest unless given; its digest (DIGEST where the image was loaded) and ID go to
# OUT/provenance.json. Run it on the Spark that holds MODEL_DIR, under that machine's `machine.lock` whole and
# `gpu.lock` (specs/WORKER.md), with no other model on the GPU. The command is the card's for the DGX Spark, `vllm serve` from the card's shell
# (/bin/bash), but for: --host 127.0.0.1 --port 18888, --max-num-seqs 1, no --speculative-config,
# VLLM_MXFP8_LM_HEAD=0 (the card requantizes the head), B12X_W4A16_A4_PREFILL=0 (the card quantizes prefill
# activations to FP4), VLLM_SERVER_DEV_MODE=1 (the prefix cache reset between runs), HF_HUB_OFFLINE=1 and the
# model from MODEL_DIR.
#
# Spec 125: DOCKER_EXTRA and SERVE_EXTRA, words, are added to the container's and to `vllm serve`'s arguments, and
# CLIENT, where set, is run in place of the fetch with the server's URL as its last word (flash_next_taps_vllm.sh).
set -euo pipefail
model=$(realpath "$1"); cases=$(realpath "$2"); score=$(realpath "$3"); out=$4; image=${5:-eugr/spark-vllm-b12x:latest}
here=$(cd "$(dirname "$0")" && pwd)
name=zspark-vllm-reference
port=18888
mkdir -p "$out" "$HOME/.cache/vllm" "$HOME/.cache/flashinfer" "$HOME/.triton" "$HOME/.cache/b12x"
# a loaded image (docker save | docker load) has no registry digest: DIGEST gives it
digest=${DIGEST:-$(docker image inspect "$image" --format '{{index .RepoDigests 0}}' 2>/dev/null | tr -d '\n' || true)}
digest=${digest:-unknown}
id=$(docker image inspect "$image" --format '{{.Id}}')
cleanup() { docker stop -t 30 "$name" >/dev/null 2>&1 || true; docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
started=$(date +%s)
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all --ipc host --network host --shm-size=16g --ulimit memlock=-1 \
    --ulimit stack=67108864 --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    -v "$model:/model:ro" -v "$HOME/.cache/vllm:/root/.cache/vllm" -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
    -v "$HOME/.triton:/root/.triton" -v "$HOME/.cache/b12x:/root/.cache/b12x" \
    -e VLLM_PLE_TABLE_MEMORY=disk -e B12X_DISK_BACKEND=io_uring -e CUTE_DSL_ARCH=sm_121a -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e VLLM_USE_AOT_COMPILE=1 \
    -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e B12X_POLICY_MODE=auto -e VLLM_MXFP8_LM_HEAD=0 \
    -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 -e VLLM_B12X_MOE_FP4_FORCE_A16=1 -e B12X_W4A16_A4_PREFILL=0 \
    -e VLLM_SERVER_DEV_MODE=1 -e HF_HUB_OFFLINE=1 ${DOCKER_EXTRA:-} \
    --entrypoint /bin/bash "$image" -c 'exec vllm serve /model "$@"' vllm --served-model-name "$(basename "$model")" \
    --host 127.0.0.1 --port $port --tensor-parallel-size 1 --pipeline-parallel-size 1 --mamba-cache-mode align \
    --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16 --kv-cache-dtype fp8 \
    --quantization modelopt_mixed --block-size 16 --load-format b12x --max-model-len 262144 --max-num-seqs 1 \
    --max-num-batched-tokens 4096 --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x \
    --no-enable-flashinfer-autotune --mm-encoder-tp-mode data --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' --gpu-memory-utilization 0.8 ${SERVE_EXTRA:-} >/dev/null
# the log from the start, so that a load that fails says why
docker logs -f "$name" > "$out/server.log" 2>&1 &
until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; do
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then echo "the container stopped while loading" >&2; exit 1; fi
    sleep 5
done
ready=$(date +%s)
if [ -n "${CLIENT:-}" ]; then
    $CLIENT "http://127.0.0.1:$port"
else
    python3 -I "$here/vllm_reference.py" fetch "http://127.0.0.1:$port" "$cases" "$score" "$out"
fi
printf '{"image": "%s", "digest": "%s", "id": "%s", "model": "%s", "load_seconds": %d, "host": "%s"}\n' \
    "$image" "$digest" "$id" "$model" $((ready - started)) "$(hostname)" > "$out/provenance.json"
