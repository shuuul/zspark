#!/usr/bin/env bash
# Spec 151 WS-04: build the Nemotron-3.5-Transcribe reference image `zspark151-nemo-vllm:local`
# (vLLM 0.32.0 + NeMo Speech's nemo_speechlm plugin) on spark-b (GB10, aarch64, CUDA 13).
#
#   build_image.sh [DOCKERFILE_DIR]
#
# The Dockerfile COPYies `nemo/`, so this script stages a context under
# ~/.cache/zspark/scratch-151/nemo-image/: the Dockerfile plus a clean export of the pinned NeMo Speech
# checkout (NEMO_SRC, default ~/.cache/zspark/scratch-151/nemo, git archive at HEAD = 774bc30 main
# 2026-10-30). No docker build-cache is trusted for the nemo layer: the archive's hash feeds the tag.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CTX=${ZSPARK_NEMOTRON_IMAGE_CTX:-$HOME/.cache/zspark/scratch-151/nemo-image}
NEMO_SRC=${NEMO_SRC:-$HOME/.cache/zspark/scratch-151/nemo}
IMAGE=zspark151-nemo-vllm:local

# Stage the wheels the Dockerfile COPYies (a PyPI reach from the Sparks truncates large transfers;
# fetch once into scratch-151/wheels, copy into the context). The vllm wheel may already sit in the
# context from the first build's manual staging — seed the cache from it before downloading.
WHEELS=${ZSPARK_NEMOTRON_WHEELS:-$HOME/.cache/zspark/scratch-151/wheels}
mkdir -p "$WHEELS" "$CTX"
[ -f "$CTX/vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl" ] && \
  [ ! -f "$WHEELS/vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl" ] && \
  cp "$CTX/vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl" "$WHEELS/" || true
for spec in "vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl:vllm==0.32.0" \
            "flashinfer_python-0.7.1-py3-none-any.whl:flashinfer-python==0.7.1"; do
  f=${spec%%:*}; pkg=${spec##*:}
  if [ ! -f "$WHEELS/$f" ]; then
    docker run --rm -v "$WHEELS:/wheels" zspark142-vllm-audio:local \
      pip download --no-deps -d /wheels "$pkg" || \
      pip download --no-deps -d "$WHEELS" "$pkg"
  fi
  cp -f "$WHEELS/$f" "$CTX/$f"
done
mkdir -p "$CTX/nemo"
rsync -a --delete --exclude .git "$NEMO_SRC/" "$CTX/nemo/"
cp "$HERE/Dockerfile" "$CTX/Dockerfile"
sha=$(cd "$NEMO_SRC" && git rev-parse HEAD 2>/dev/null || echo "nogit")
echo "nemo at $sha"
docker build -t "$IMAGE" "$CTX" 2>&1 | tee "$CTX/build.log"
echo "built $IMAGE; log: $CTX/build.log"
