#!/bin/bash
# Flash Next DFlash2's completed-layer taps of FP4-Local in the card's vLLM, one boot (spec 125 WS-02, V1).
#
#   tools/reference/flash_next_taps_vllm.sh MODEL_DIR REFERENCE RAW [IMAGE]
#
# vllm_reference.sh's boot of MODEL_DIR with flash_next_taps_vllm.patch applied to a copy of the image's
# vllm/models/qwen4_exp/nvidia/model.py (RAW/model.py, mounted read-only over the image's; the repository's
# file is never patched), RAW/taps mounted for the hook's dumps (ZSPARK_TAPS_DIR) and --enforce-eager: a compiled
# graph or a CUDA graph's replay runs no Python. Its client is `flash_next_tap_export.py --from-vllm fetch` over
# REFERENCE's prompts (vllm_reference.sh's output for MODEL_DIR, $ZSPARK_REFERENCE_FP4_LOCAL); then
# `--from-vllm assemble` in the image, on the CPU, writes RAW/fixture/ (NAME.safetensors, MANIFEST.json), which
# goes to $ZSPARK_REFERENCE_FP4_LOCAL/taps/. Locks and GPU as vllm_reference.sh's: the caller holds
# `machine.lock` whole and `gpu.lock`, with no other model on the GPU.
set -euo pipefail
model=$1; reference=$(realpath "$2"); raw=$(realpath -m "$3"); image=${4:-eugr/spark-vllm-b12x:latest}
here=$(cd "$(dirname "$0")" && pwd)
target=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py
mkdir -p "$raw/taps"
: > "$raw/taps/label"
docker run --rm --entrypoint cat "$image" "$target" > "$raw/model.py"
patch --quiet "$raw/model.py" "$here/flash_next_taps_vllm.patch"
sha256sum "$here/flash_next_taps_vllm.patch" > "$raw/patch.sha256"
DOCKER_EXTRA="-v $raw/model.py:$target:ro -v $raw/taps:/taps -e ZSPARK_TAPS_DIR=/taps" SERVE_EXTRA=--enforce-eager \
    CLIENT="python3 -I $here/flash_next_tap_export.py --from-vllm fetch --reference $reference --raw $raw" \
    "$here/vllm_reference.sh" "$model" /dev/null /dev/null "$raw" "$image"
docker run --rm -v "$raw:/raw" -v "$here:/here:ro" --entrypoint python3 "$image" -I /here/flash_next_tap_export.py \
    --from-vllm assemble --raw /raw --owner "$(id -u):$(id -g)"
